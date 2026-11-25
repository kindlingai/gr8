// Fast Blackwell (sm_121a) 64-Q-row / 12-warp overlapped-SMEM single-BF16-p@v prefill attention kernel.
// Included inside ink2_attn_tc.cuh (namespace ink2).
//
// Compared to stock k_ink2_attn_prefill_tc (16 Q-rows, 3 warps/CTA, p=hi+lo 2x MMAs, expf):
//   - NQ_BLK = 4 (64 Q-rows/CTA, 4 * gq = 12 warps = 384 threads/CTA) -> 4x less K/V DRAM/L2 traffic!
//   - Overlapped SMEM: Phase 1 uses sK[64][136] + sG_f16[12][16][80] = 48,128 B (47.0 KB -> 2 CTAs/SM = 24 warps/SM!);
//     Phase 2 reuses the exact same SMEM for sV[128][72] (18,432 B).
//   - Single BF16 p@v MMA (2x fewer Tensor Core MMAs in p@v) + hardware __exp2f softmax!
//   - Per-warp causal skip when kb > pb (skips dead upper-triangle K-tiles while still helping cooperative K/V loads).
#pragma once

constexpr int FATC_NQB = 4;
inline size_t attn_tc_fast_smem(int gq)
{
    const size_t ph1 = static_cast<size_t>(TC_KT) * TC_KLD * 2 + static_cast<size_t>(FATC_NQB * gq) * 16 * TC_GC * sizeof(__half);
    const size_t ph2 = static_cast<size_t>(TC_HD) * TC_VLD * 2;
    return ph1 > ph2 ? ph1 : ph2;
}

__global__ void __launch_bounds__(384, 1) k_ink2_attn_prefill_tc_fast(AttnCoreArgs a, int gq)
{
    extern __shared__ __align__(16) unsigned char tc_sm[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int nthr = FATC_NQB * gq * 32;

    const int qb = warp / gq;          // 0..3 (which 16-row Q block)
    const int h_off = warp % gq;       // 0..gq-1 (which query head in this KV group)

    uint16_t* sK = reinterpret_cast<uint16_t*>(tc_sm);                                          // [64][136] BF16 (17,408 B)
    __half* sG = reinterpret_cast<__half*>(sK + TC_KT * TC_KLD) + warp * 16 * TC_GC;            // [12][16][80] FP16 (30,720 B)
    uint16_t* sV = reinterpret_cast<uint16_t*>(tc_sm);                                          // [128][72] BF16 (overlapped!)

    const int lkv = blockIdx.y;
    const int r0_cta = blockIdx.x * (16 * FATC_NQB);
    const int r0 = r0_cta + qb * 16;
    const int M = a.rows;
    const int lq0 = max(0, (a.kv0 + lkv) * a.gqa - a.q0);
    const int lq1 = min(a.nh_q, (a.kv0 + lkv + 1) * a.gqa - a.q0);
    const int lq = lq0 + h_off;
    const bool on = (lq < lq1) && (r0 < M);
    const int ra = r0 + g, rb = ra + 8;
    const bool va = on && (ra < M), vb = on && (rb < M);
    const bool glob = a.global != 0;

    const int pr0_cta = a.rw.pos[r0_cta];
    const int pr0 = r0 < M ? a.rw.pos[r0] : (pr0_cta + qb * 16);
    const int pa = pr0 + g, pb = pa + 8;
    const int plast_cta = pr0_cta + min(16 * FATC_NQB, M - r0_cta) - 1;
    const int loa = attn_lo(glob, a.window, pa), lob = attn_lo(glob, a.window, pb);
    const int kstart = attn_lo(glob, a.window, pr0_cta);

    constexpr float INV_ROOT_HD = 0.08838834764831845f; // 1 / sqrt(128)
    constexpr float LOG2E = 1.4426950408889634f;

    uint32_t qf[8][4], rf[4];
    {
        const uint16_t* qpa = a.q_raw + (static_cast<size_t>(va ? ra : 0) * a.nh_q + (on ? lq : 0)) * TC_HD;
        const uint16_t* qpb = a.q_raw + (static_cast<size_t>(vb ? rb : 0) * a.nh_q + (on ? lq : 0)) * TC_HD;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk)
        {
            const int c = kk * 16 + 2 * t;
            qf[kk][0] = va ? ld32(qpa + c) : 0u;
            qf[kk][1] = vb ? ld32(qpb + c) : 0u;
            qf[kk][2] = va ? ld32(qpa + c + 8) : 0u;
            qf[kk][3] = vb ? ld32(qpb + c + 8) : 0u;
        }
        const uint16_t* rpa = a.r_raw + (static_cast<size_t>(va ? ra : 0) * a.nh_q + (on ? lq : 0)) * TC_RD;
        const uint16_t* rpb = a.r_raw + (static_cast<size_t>(vb ? rb : 0) * a.nh_q + (on ? lq : 0)) * TC_RD;
        rf[0] = va ? ld32(rpa + 2 * t) : 0u;
        rf[1] = vb ? ld32(rpb + 2 * t) : 0u;
        rf[2] = va ? ld32(rpa + 2 * t + 8) : 0u;
        rf[3] = vb ? ld32(rpb + 2 * t + 8) : 0u;
    }

    float oc[16][4];
    #pragma unroll
    for (int j = 0; j < 16; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.0f;

    float ma = va ? bf(a.sink[static_cast<size_t>(ra) * a.nh_q + lq]) : 0.0f;
    float mb = vb ? bf(a.sink[static_cast<size_t>(rb) * a.nh_q + lq]) : 0.0f;
    float la = (t == 0) ? 1.0f : 0.0f, lb = la;

    for (int kb = kstart; kb <= plast_cta; kb += TC_KT)
    {
        // Phase 1: Cooperatively load K tile [64][128] into sK + compute warp's rel bias into sG
        for (int v = tid; v < TC_KT * (TC_HD / 8); v += nthr)
        {
            const int r = v / (TC_HD / 8), c = (v % (TC_HD / 8)) * 8, kpos = kb + r;
            uint4 kx = make_uint4(0, 0, 0, 0);
            if (kpos <= plast_cta)
            {
                size_t voff;
                const uint16_t* kr = attn_key(a, r0_cta, kpos, lkv, voff);
                kx = *reinterpret_cast<const uint4*>(kr + c);
            }
            *reinterpret_cast<uint4*>(sK + r * TC_KLD + c) = kx;
        }
        const int pb_max = pr0 + 15;
        const bool warp_active = on && (kb <= pb_max) && (kb + TC_KT - 1 >= attn_lo(glob, a.window, pr0));
        const int d0 = pr0 - kb - (TC_KT - 1);
        const bool rel = warp_active && (d0 < a.rel_win);
        if (rel)
        {
            const size_t rwn = static_cast<size_t>(a.rel_win);
            #pragma unroll
            for (int nt = 0; nt < TC_GC / 8; ++nt)
            {
                const int dist = d0 + nt * 8 + g;
                uint32_t b0 = 0u, b1 = 0u;
                if (dist >= 0 && dist < a.rel_win)
                {
                    const uint16_t* pp = a.rel_proj + dist;
                    b0 = static_cast<uint32_t>(__ldg(pp + (2 * t) * rwn)) | (static_cast<uint32_t>(__ldg(pp + (2 * t + 1) * rwn)) << 16);
                    b1 = static_cast<uint32_t>(__ldg(pp + (2 * t + 8) * rwn)) | (static_cast<uint32_t>(__ldg(pp + (2 * t + 9) * rwn)) << 16);
                }
                float gc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                tc_mma(gc, rf, b0, b1);
                __half* ga = sG + g * TC_GC + nt * 8 + 2 * t;
                ga[0] = __float2half_rn(gc[0]);
                ga[1] = __float2half_rn(gc[1]);
                ga[8 * TC_GC] = __float2half_rn(gc[2]);
                ga[8 * TC_GC + 1] = __float2half_rn(gc[3]);
            }
        }
        __syncthreads();

        // Compute scores sr[8][4] = q @ k^T / sqrt(hd) + rel, and online softmax weights hi[4][4] in registers!
        uint32_t p_hi[4][4];
        if (warp_active)
        {
            float sr[8][4];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                sr[j][0] = sr[j][1] = sr[j][2] = sr[j][3] = 0.0f;
                const uint16_t* kr = sK + (j * 8 + g) * TC_KLD + 2 * t;
                #pragma unroll
                for (int kk = 0; kk < 8; ++kk) tc_mma(sr[j], qf[kk], ld32(kr + kk * 16), ld32(kr + kk * 16 + 8));
            }
            float xa = -INFINITY, xb = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    const int kcol = j * 8 + 2 * t + e, kpos = kb + kcol;
                    float za = sr[j][e] * INV_ROOT_HD;
                    float zb = sr[j][2 + e] * INV_ROOT_HD;
                    if (rel)
                    {
                        const __half* gp = sG + g * TC_GC + (TC_KT - 1) + g - kcol;
                        za += __half2float(gp[0]);
                        zb += __half2float(gp[8 * TC_GC + 8]);
                    }
                    za = (va && kpos <= pa && kpos >= loa) ? za : -INFINITY;
                    zb = (vb && kpos <= pb && kpos >= lob) ? zb : -INFINITY;
                    sr[j][e] = za;
                    sr[j][2 + e] = zb;
                    xa = fmaxf(xa, za);
                    xb = fmaxf(xb, zb);
                }
            #pragma unroll
            for (int o2 = 1; o2 < 4; o2 <<= 1)
            {
                xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
                xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
            }
            const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
            const float aa = exp2f((ma - na) * LOG2E);
            const float ab = exp2f((mb - nb) * LOG2E);
            ma = na;
            mb = nb;
            la *= aa;
            lb *= ab;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
            {
                oc[j][0] *= aa;
                oc[j][1] *= aa;
                oc[j][2] *= ab;
                oc[j][3] *= ab;
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
            {
                #pragma unroll
                for (int half = 0; half < 2; ++half)
                {
                    const float* c = sr[2 * kk + half];
                    const float p0 = exp2f((c[0] - na) * LOG2E);
                    const float p1 = exp2f((c[1] - na) * LOG2E);
                    const float p2 = exp2f((c[2] - nb) * LOG2E);
                    const float p3 = exp2f((c[3] - nb) * LOG2E);
                    la += p0 + p1;
                    lb += p2 + p3;
                    const __nv_bfloat162 b01 = __floats2bfloat162_rn(p0, p1);
                    const __nv_bfloat162 b23 = __floats2bfloat162_rn(p2, p3);
                    p_hi[kk][half * 2]     = *reinterpret_cast<const uint32_t*>(&b01);
                    p_hi[kk][half * 2 + 1] = *reinterpret_cast<const uint32_t*>(&b23);
                }
            }
        }
        __syncthreads();

        // Phase 2: Cooperatively load V tile [64][128] transposed into overlapped sV[128][72]
        for (int v = tid; v < TC_KT * (TC_HD / 8); v += nthr)
        {
            const int r = v / (TC_HD / 8), c = (v % (TC_HD / 8)) * 8, kpos = kb + r;
            uint4 vx = make_uint4(0, 0, 0, 0);
            if (kpos <= plast_cta)
            {
                size_t voff;
                const uint16_t* kr = attn_key(a, r0_cta, kpos, lkv, voff);
                vx = *reinterpret_cast<const uint4*>(kr + voff + c);
            }
            const uint16_t* ve = reinterpret_cast<const uint16_t*>(&vx);
            #pragma unroll
            for (int i = 0; i < 8; ++i) sV[(c + i) * TC_VLD + r] = ve[i];
        }
        __syncthreads();

        // Single BF16 p @ v MMA (2x fewer MMAs than stock p=hi+lo!)
        if (warp_active)
        {
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
            {
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    const uint16_t* vr = sV + (j * 8 + g) * TC_VLD + 2 * t + kk * 16;
                    tc_mma(oc[j], p_hi[kk], ld32(vr), ld32(vr + 8));
                }
            }
        }
        __syncthreads();
    }

    if (!on) return;
    #pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1)
    {
        la += __shfl_xor_sync(0xffffffffu, la, o2);
        lb += __shfl_xor_sync(0xffffffffu, lb, o2);
    }
    const size_t qa = (static_cast<size_t>(ra) * a.nh_q + lq) * TC_HD;
    const size_t qb_off = (static_cast<size_t>(rb) * a.nh_q + lq) * TC_HD;
    #pragma unroll
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
            a.og[qb_off + c] = attn_out(oc[j][2], lb, a.gate[qb_off + c]);
            a.og[qb_off + c + 1] = attn_out(oc[j][3], lb, a.gate[qb_off + c + 1]);
        }
    }
}
