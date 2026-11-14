// ink2 attention of a prefill chunk on tensor cores. Included by ink2_attn.cuh, inside namespace ink2.
//
// The rows are one run of one slot at consecutive positions (the caller checks), so 16 rows share their keys.
// CTA = (16 rows, K / V head); a warp = one query head of that K / V head, its 16 rows. The CTA walks the keys
// from the first row's window start to the last row's position in tiles of 64: K row-major and V transposed in
// shared memory, loaded once for every query head of the group.
//   scores   [16][64] = q k^T / sqrt(hd): mma m16n8k16, q fragments in registers (q is BF16 after its clip)
//   rel bias G [16][80] = r proj[:, dist]: one mma per 8 distances into the warp's shared rows; the score of
//            (row i, key j of the tile) reads column 63 + i - j. Distances past rel_win load zero.
//   softmax  online, from (maximum = sink, sum = 1, acc = 0) as the stock merge with the sink ends
//   acc      [16][128] += p v: p = hi + lo, two BF16 fragments, so the product keeps about 16 bits of p
// The sums are in the tensor cores' order and p is split, so the output is close to the stock kernels' and not
// bit-equal to it. q and r are prepared in place (k_ink2_attn_qr_prep) with the stock kernel's arithmetic.
#pragma once

constexpr int TC_HD = 128, TC_RD = 16, TC_KT = 64, TC_KLD = TC_HD + 8, TC_VLD = TC_KT + 8, TC_GC = 80, TC_MAX_GQ = 8;
inline size_t attn_tc_smem(int gq) { return static_cast<size_t>(TC_KT) * TC_KLD * 2 + static_cast<size_t>(TC_HD) * TC_VLD * 2 + static_cast<size_t>(gq) * 16 * TC_GC * 4; }

__device__ __forceinline__ void tc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// q_raw -> q and r_raw -> r of every (row, query head), in place. grid rows * nh_q.
__global__ void __launch_bounds__(ACT) k_ink2_attn_qr_prep(uint16_t* q, uint16_t* r, const uint16_t* q_clip, const uint16_t* rel_clip, Rows rw, int nh_q, int hd, int rd, int base,
                                                           int global)
{
    __shared__ float rs[ATTN_MAX_REL], red[ACT];
    const int t = threadIdx.x, row = blockIdx.x / nh_q, lq = blockIdx.x % nh_q;
    const float tq = global ? attn_tau(rw.pos[row], base) : 1.0f;
    const size_t qo = (static_cast<size_t>(row) * nh_q + lq) * hd;
    const float tmp = t < hd ? mul_rn(bf(q[qo + t]), bf(q_clip[lq * hd + t])) : 0.0f;
    const float qf = l2_clip(blk_sum(mul_rn(tmp, tmp), red, ACT), hd);
    if (t < hd)
    {
        float v = rbf(mul_rn(tmp, qf));
        if (global) v = rbf(mul_rn(v, tq));
        q[qo + t] = bf16(v);
    }
    if (t == 0)
    {
        uint16_t* rr = r + (static_cast<size_t>(row) * nh_q + lq) * rd;
        float ss = 0.0f;
        for (int d = 0; d < rd; ++d)
        {
            rs[d] = mul_rn(bf(rr[d]), bf(rel_clip[lq * rd + d]));
            ss = add_rn(ss, mul_rn(rs[d], rs[d]));
        }
        const float rf = l2_clip(ss, rd);
        for (int d = 0; d < rd; ++d)
        {
            float v = rbf(mul_rn(rs[d], rf));
            if (global) v = rbf(mul_rn(v, tq));
            rr[d] = bf16(v);
        }
    }
}

// a.q_raw / a.r_raw hold q and r (k_ink2_attn_qr_prep); a.rows = M, a.m0 = 0. grid (ceil(M / 16), nh_kv), block gq * 32
// threads, attn_tc_smem(gq) bytes of dynamic shared memory. hd = TC_HD, rel_dim = TC_RD.
__global__ void k_ink2_attn_prefill_tc(AttnCoreArgs a, int gq)
{
    extern __shared__ __align__(16) unsigned char tc_sm[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, nthr = gq * 32;
    uint16_t* sK = reinterpret_cast<uint16_t*>(tc_sm);                                // [key][cell]
    uint16_t* sV = sK + TC_KT * TC_KLD;                                               // [cell][key]
    float* sG = reinterpret_cast<float*>(sV + TC_HD * TC_VLD) + warp * 16 * TC_GC;    // this warp's [row][distance column]
    const int lkv = blockIdx.y, r0 = blockIdx.x * 16, M = a.rows;
    const int lq0 = max(0, (a.kv0 + lkv) * a.gqa - a.q0), lq1 = min(a.nh_q, (a.kv0 + lkv + 1) * a.gqa - a.q0);
    const int lq = lq0 + warp;
    const bool on = lq < lq1;
    const int ra = r0 + g, rb = ra + 8;   // this lane's two rows
    const bool va = on && ra < M, vb = on && rb < M;
    const bool glob = a.global != 0;
    const int pr0 = a.rw.pos[r0], pa = pr0 + g, pb = pa + 8;
    const int plast = pr0 + min(16, M - r0) - 1;
    const int loa = attn_lo(glob, a.window, pa), lob = attn_lo(glob, a.window, pb), kstart = attn_lo(glob, a.window, pr0);
    const float root_hd = sqrtf(static_cast<float>(TC_HD));

    uint32_t qf[8][4], rf[4];
    {
        const uint16_t* qpa = a.q_raw + (static_cast<size_t>(ra) * a.nh_q + lq) * TC_HD;
        const uint16_t* qpb = a.q_raw + (static_cast<size_t>(rb) * a.nh_q + lq) * TC_HD;
        for (int kk = 0; kk < 8; ++kk)
        {
            const int c = kk * 16 + 2 * t;
            qf[kk][0] = va ? ld32(qpa + c) : 0u;
            qf[kk][1] = vb ? ld32(qpb + c) : 0u;
            qf[kk][2] = va ? ld32(qpa + c + 8) : 0u;
            qf[kk][3] = vb ? ld32(qpb + c + 8) : 0u;
        }
        const uint16_t* rpa = a.r_raw + (static_cast<size_t>(ra) * a.nh_q + lq) * TC_RD;
        const uint16_t* rpb = a.r_raw + (static_cast<size_t>(rb) * a.nh_q + lq) * TC_RD;
        rf[0] = va ? ld32(rpa + 2 * t) : 0u;
        rf[1] = vb ? ld32(rpb + 2 * t) : 0u;
        rf[2] = va ? ld32(rpa + 2 * t + 8) : 0u;
        rf[3] = vb ? ld32(rpb + 2 * t + 8) : 0u;
    }
    float oc[16][4];
    for (int j = 0; j < 16; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.0f;
    // the running maximum starts at the sink, whose weight 1 is in lane t = 0's part of the sum
    float ma = va ? bf(a.sink[static_cast<size_t>(ra) * a.nh_q + lq]) : 0.0f, mb = vb ? bf(a.sink[static_cast<size_t>(rb) * a.nh_q + lq]) : 0.0f;
    float la = t == 0 ? 1.0f : 0.0f, lb = la;

    for (int kb = kstart; kb <= plast; kb += TC_KT)
    {
        __syncthreads();   // every warp is done with the last tile
        for (int v = tid; v < TC_KT * (TC_HD / 8); v += nthr)
        {
            const int r = v / (TC_HD / 8), c = (v % (TC_HD / 8)) * 8, kpos = kb + r;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = make_uint4(0, 0, 0, 0);
            if (kpos <= plast)
            {
                size_t voff;
                const uint16_t* kr = attn_key(a, r0, kpos, lkv, voff);
                kx = *reinterpret_cast<const uint4*>(kr + c);
                vx = *reinterpret_cast<const uint4*>(kr + voff + c);
            }
            *reinterpret_cast<uint4*>(sK + r * TC_KLD + c) = kx;
            const uint16_t* ve = reinterpret_cast<const uint16_t*>(&vx);
            for (int i = 0; i < 8; ++i) sV[(c + i) * TC_VLD + r] = ve[i];
        }
        // the rel bias of this tile: column col is distance d0 + col
        const int d0 = pr0 - kb - (TC_KT - 1);
        const bool rel = on && d0 < a.rel_win;
        if (rel)
        {
            for (int nt = 0; nt < TC_GC / 8; ++nt)
            {
                const int dist = d0 + nt * 8 + g;
                uint32_t b0 = 0u, b1 = 0u;
                if (dist >= 0 && dist < a.rel_win)
                {
                    const uint16_t* pp = a.rel_proj + dist;
                    const size_t rwn = static_cast<size_t>(a.rel_win);
                    b0 = static_cast<uint32_t>(pp[(2 * t) * rwn]) | (static_cast<uint32_t>(pp[(2 * t + 1) * rwn]) << 16);
                    b1 = static_cast<uint32_t>(pp[(2 * t + 8) * rwn]) | (static_cast<uint32_t>(pp[(2 * t + 9) * rwn]) << 16);
                }
                float gc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                tc_mma(gc, rf, b0, b1);
                float* ga = sG + g * TC_GC + nt * 8 + 2 * t;
                ga[0] = gc[0];
                ga[1] = gc[1];
                ga[8 * TC_GC] = gc[2];
                ga[8 * TC_GC + 1] = gc[3];
            }
        }
        __syncthreads();
        if (!on) continue;

        // the scores [16][64] = q k^T: 8 blocks of 8 keys
        float sr[8][4];
        for (int j = 0; j < 8; ++j)
        {
            sr[j][0] = sr[j][1] = sr[j][2] = sr[j][3] = 0.0f;
            const uint16_t* kr = sK + (j * 8 + g) * TC_KLD + 2 * t;
            for (int kk = 0; kk < 8; ++kk) tc_mma(sr[j], qf[kk], ld32(kr + kk * 16), ld32(kr + kk * 16 + 8));
        }
        float xa = -INFINITY, xb = -INFINITY;
        for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 2; ++e)
            {
                const int kcol = j * 8 + 2 * t + e, kpos = kb + kcol;
                float za = div_rn(sr[j][e], root_hd), zb = div_rn(sr[j][2 + e], root_hd);
                if (rel)
                {
                    const float* gp = sG + g * TC_GC + (TC_KT - 1) + g - kcol;
                    za = add_rn(za, gp[0]);
                    zb = add_rn(zb, gp[8 * TC_GC + 8]);
                }
                za = va && kpos <= pa && kpos >= loa ? za : -INFINITY;
                zb = vb && kpos <= pb && kpos >= lob ? zb : -INFINITY;
                sr[j][e] = za;
                sr[j][2 + e] = zb;
                xa = fmaxf(xa, za);
                xb = fmaxf(xb, zb);
            }
        for (int o2 = 1; o2 < 4; o2 <<= 1)
        {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
        }
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        const float aa = expf(ma - na), ab = expf(mb - nb);
        ma = na;
        mb = nb;
        la *= aa;
        lb *= ab;
        for (int j = 0; j < 16; ++j)
        {
            oc[j][0] *= aa;
            oc[j][1] *= aa;
            oc[j][2] *= ab;
            oc[j][3] *= ab;
        }
        // acc [16][128] += p v, 16 keys a step; p = hi + lo
        for (int kk = 0; kk < 4; ++kk)
        {
            uint32_t hi[4], lw[4];
            for (int half = 0; half < 2; ++half)
            {
                const float* c = sr[2 * kk + half];
                const float p0 = expf(c[0] - na), p1 = expf(c[1] - na), p2 = expf(c[2] - nb), p3 = expf(c[3] - nb);
                const float h0 = rbf(p0), h1 = rbf(p1), h2 = rbf(p2), h3 = rbf(p3);
                const float w0 = rbf(p0 - h0), w1 = rbf(p1 - h1), w2 = rbf(p2 - h2), w3 = rbf(p3 - h3);
                la += (h0 + w0) + (h1 + w1);
                lb += (h2 + w2) + (h3 + w3);
                hi[half * 2] = static_cast<uint32_t>(bf16(h0)) | (static_cast<uint32_t>(bf16(h1)) << 16);
                hi[half * 2 + 1] = static_cast<uint32_t>(bf16(h2)) | (static_cast<uint32_t>(bf16(h3)) << 16);
                lw[half * 2] = static_cast<uint32_t>(bf16(w0)) | (static_cast<uint32_t>(bf16(w1)) << 16);
                lw[half * 2 + 1] = static_cast<uint32_t>(bf16(w2)) | (static_cast<uint32_t>(bf16(w3)) << 16);
            }
            for (int j = 0; j < 16; ++j)
            {
                const uint16_t* vr = sV + (j * 8 + g) * TC_VLD + 2 * t + kk * 16;
                const uint32_t b0 = ld32(vr), b1 = ld32(vr + 8);
                tc_mma(oc[j], hi, b0, b1);
                tc_mma(oc[j], lw, b0, b1);
            }
        }
    }
    if (!on) return;
    for (int o2 = 1; o2 < 4; o2 <<= 1)
    {
        la += __shfl_xor_sync(0xffffffffu, la, o2);
        lb += __shfl_xor_sync(0xffffffffu, lb, o2);
    }
    const size_t qa = (static_cast<size_t>(ra) * a.nh_q + lq) * TC_HD, qb = (static_cast<size_t>(rb) * a.nh_q + lq) * TC_HD;
    for (int j = 0; j < 16; ++j)
    {
        const int c = j * 8 + 2 * t;
        if (va)
        {
            a.og[qa + c] = attn_out(oc[j][0], la, a.gate[qa + c]);
            a.og[qa + c + 1] = attn_out(oc[j][1], la, a.gate[qa + c + 1]);
        }
        if (vb)
        {
            a.og[qb + c] = attn_out(oc[j][2], lb, a.gate[qb + c]);
            a.og[qb + c + 1] = attn_out(oc[j][3], lb, a.gate[qb + c + 1]);
        }
    }
}
