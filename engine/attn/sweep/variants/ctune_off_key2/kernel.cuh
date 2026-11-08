// Compile/launch tuning of our kernels at 1ed4501c. No arithmetic edits.
#pragma once
namespace detail_ctune_off_key2 {
using namespace ds41a;
template <bool FL, bool CQ = false>
__global__ void __launch_bounds__(AT_THREADS, 1) k_ctune_off_key2(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);   // two lists of 64 slots: the block in use and the next live one
    float* sc_s = (float*)(rowp + 128);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
    {
        const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
        for (int c = tid; c < 16 * HD / 8; c += AT_THREADS)
            cp_async16(Qs + (c >> 6) * AT_QLD + (c & 63) * 8, src + c, 16);
    }
    float m_run = -1e30f, sum_run = 0.0f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int start = start_of(p.start, p.dstart);
    const int L = p.WL + p.TK;
    const int nblk = (L + 63) >> 6;
    const int R = p.ring_rows ? p.ring_rows : WIN;
    const int tpos = pos_of(start, p.dstart, p.dsess, t);   // this row's position (decode)
    const bf16 *kvr = p.kvw, *ckr = p.ck;                   // this row's ring and compressed rows (its session's slab at SESS > 1)
    if (p.dsess)
    {
        const int k = p.dsess[t];
        kvr += (size_t)k * R * HD;
        if (ckr) ckr = cq_slab<CQ>(ckr, p.ck_ss, k);
    }
    // slot list of block blk into lst; the return value (uniform) says whether any slot is live. A block with no live slot (the -1 tail
    // of a short top-k list, the masked front of a young window) changes nothing: P = 0, scale = exp(0) = 1 -> skipped
    auto build = [&](int blk, const bf16** lst) -> int
    {
        int has = 0;
        if (tid < 64)
        {
            const int i = blk * 64 + tid;
            const bf16* rp = nullptr;
            if (i < p.WL)
            {
                if (!p.decode)
                {
                    const int id = max(t - (WIN - 1), 0) + i;
                    if (id <= t) rp = p.kvw + (size_t)id * HD;
                }
                else
                {
                    const int pos = tpos - (WIN - 1) + i;   // list slot i holds position tpos - 127 + i; negative = masked
                    if (pos >= (FL ? p.floor : 0)) rp = kvr + (size_t)(pos % R) * HD;   // FL (bounded replay): positions below p.floor masked
                }
            }
            else if (i < L)
            {
                const int j = p.topk[(size_t)t * p.ldtk + (i - p.WL)];
                if (j >= 0) rp = cq_slot<CQ>(ckr, j);
            }
            lst[tid] = rp;
            has = rp != nullptr;
        }
        return __syncthreads_or(has);
    };
    int cblk = 0, cur = 0;
    while (cblk < nblk && !build(cblk, rowp)) cblk++;
#pragma unroll 2
    while (cblk < nblk)
    {
        const bf16** cl = rowp + 64 * cur;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            const int c = tid + i * AT_THREADS;
            const bf16* rp = cl[c >> 6];
            if (CQ && cq_tag(rp)) continue;   // cq_store below
            cp_async16(Ks + (c >> 6) * AT_KLD + (c & 63) * 8, rp ? (const void*)((const uint4*)rp + (c & 63)) : (const void*)p.q, rp ? 16 : 0);
        }
        cp_async_commit();
        uint2 cd[8];
        uint32_t sb[8];
        const bool cqb = CQ && (cblk + 1) * 64 > p.WL;   // the block holds compressed slots (uniform)
        if (cqb) cq_load(cl, w, lane, cd, sb);
        int nb = cblk + 1;
        const int nx = cur ^ 1;
        while (nb < nblk && !build(nb, rowp + 64 * nx)) nb++;
        if (nb < nblk)
        {
            const bf16* rp = rowp[64 * nx + (tid >> 2)];
            if (CQ && cq_tag(rp)) prefetch_l2(cq_row(rp) + ((tid & 3) == 3 ? CQ_ROW - 1 : (tid & 3) * 128));   // the 288 B row's lines
            else if (rp)
            {
                prefetch_l2((const char*)rp + (tid & 3) * 256);
                prefetch_l2((const char*)rp + (tid & 3) * 256 + 128);
            }
        }
        cp_async_wait0();
        if (cqb) cq_store(Ks, AT_KLD, cl, w, lane, cd, sb);
        __syncthreads();
        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks++)
            {
                uint32_t a[4];
                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                mma16816(cc, a, b0, b1);
            }
            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
            const bool v0 = cl[col] != nullptr, v1 = cl[col + 1] != nullptr;
            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
        }
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
            }
#pragma unroll
            for (int kj = 0; kj < 4; kj++)
            {
                uint32_t a[4];
                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; np++)
                {
                    uint32_t b[4];
                    const int mi = lane >> 3, r = lane & 7;
                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
        cblk = nb;
        cur = nx;
    }
    cp_async_commit();
    cp_async_wait0();   // Q's copies when no block was live
    if (sl == 0) fin_s[sh] = sum_run + expf(p.sink[hg * 16 + sh] - m_run);
    __syncthreads();
    {
        const int r0 = lane >> 2;
        const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
        bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
        bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);
            *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);
        }
    }
}

}
