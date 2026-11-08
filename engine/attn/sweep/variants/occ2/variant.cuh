#pragma once
// Occupancy experiment derived solely from our ds41_attn.cuh CPA core.
// Keep the 64-slot online-softmax tile and the exact QK/PV MMA order.
// Stage 256 feature dimensions at a time; never combine partial sums.
namespace occ2_detail {
using namespace ds41a;
constexpr int OCC_LD = 256 + 8;
constexpr size_t OCC_SMEM = (16 + 64) * OCC_LD * sizeof(bf16)
    + 16 * AT_SLD * sizeof(float) + 16 * AT_PLD * sizeof(bf16)
    + 128 * sizeof(void*) + 32 * sizeof(float);
static_assert(OCC_SMEM == 50048, "two half-width staging tiles fit two CTAs");

template <bool CQ>
__device__ __forceinline__ void occ_load_half(bf16* Ks, const bf16* const* rows,
                                             int half, int tid, const bf16* zero_src)
{
    // One lane per eight-value vector: preserve the baseline dq4 arithmetic.
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int c = tid + i * AT_THREADS;
        const int row = c >> 5, col = (c & 31) * 8;
        const bf16* rp = rows[row];
        bf16* dst = Ks + row * OCC_LD + col;
        if (CQ && cq_tag(rp))
        {
            const uint8_t* packed = cq_row(rp);
            const int dim = half * 256 + col;
            const uint32_t codes = __ldg((const uint32_t*)(packed + dim / 2));
            const __nv_bfloat162 scale = __float2bfloat162_rn(f_e4m3(__ldg(packed + CQ_CODES + dim / 16)));
            uint32_t v[4];
            cq_dq4(codes, scale, v[0], v[1]);
            cq_dq4(codes >> 16, scale, v[2], v[3]);
            *(uint4*)dst = make_uint4(v[0], v[1], v[2], v[3]);
        }
        else
            cp_async16(dst, rp ? rp + half * 256 + col : zero_src, rp ? 16 : 0);
    }
    cp_async_commit();
    cp_async_wait0();
    __syncthreads();
}
template <bool FL, bool CQ = false>
__global__ void __launch_bounds__(AT_THREADS, 2) k_attn_occ2_t(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * OCC_LD;
    float* Ss = (float*)(Ks + 64 * OCC_LD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);   // two lists of 64 slots: the block in use and the next live one
    float* sc_s = (float*)(rowp + 128);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
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
    while (cblk < nblk)
    {
        const bf16** cl = rowp + 64 * cur;
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
        {   // One unchanged QK accumulator chain; only operand staging is halved.
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const bf16* q = p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD;
                for (int c = tid; c < 16 * 256 / 8; c += AT_THREADS)
                    cp_async16(Qs + (c >> 5) * OCC_LD + (c & 31) * 8,
                               q + (c >> 5) * HD + half * 256 + (c & 31) * 8, 16);
                occ_load_half<CQ>(Ks, cl, half, tid, p.q);
                const bf16* krow = Ks + (w * 8 + (lane >> 2)) * OCC_LD + 2 * (lane & 3);
#pragma unroll 4
                for (int ks = 0; ks < 256 / 16; ks++)
                {
                    uint32_t a[4];
                    ldsm_x4(a, Qs + (lane & 15) * OCC_LD + ks * 16 + (lane >> 4) * 8);
                    const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                    const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                    mma16816(cc, a, b0, b1);
                }
                __syncthreads();
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
            // All warps help fill each half; four own its output dimensions.
            // Each output keeps the original kj=0,1,2,3 MMA sequence.
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                occ_load_half<CQ>(Ks, cl, half, tid, p.q);
                if ((w >> 2) == half)
                {
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
                            ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * OCC_LD
                                        + (w & 3) * 64 + (np * 2 + (mi >> 1)) * 8);
                            mma16816(acc[np * 2], a, b[0], b[1]);
                            mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                        }
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
        cblk = nb;
        cur = nx;
    }
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
} // namespace occ2_detail

namespace sweep_variant {
inline const void* kernel(const ds41a::AttnP& p) {
    using namespace occ2_detail;
    if (p.floor > 0) return p.ckq ? (const void*)k_attn_occ2_t<true, true> : (const void*)k_attn_occ2_t<true, false>;
    return p.ckq ? (const void*)k_attn_occ2_t<false, true> : (const void*)k_attn_occ2_t<false, false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    if (p.floor > 0) return p.ckq ? "k_attn_occ2_tILb1ELb1EE" : "k_attn_occ2_tILb1ELb0EE";
    return p.ckq ? "k_attn_occ2_tILb0ELb1EE" : "k_attn_occ2_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return occ2_detail::OCC_SMEM; }
inline cudaError_t init(const ds41a::AttnP& p) {
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p, cudaStream_t stream) {
    using namespace occ2_detail;
    const dim3 grid(p.n, p.heads / 16);
    if (p.floor > 0) {
        if (p.ckq) k_attn_occ2_t<true, true><<<grid, AT_THREADS, OCC_SMEM, stream>>>(p);
        else k_attn_occ2_t<true, false><<<grid, AT_THREADS, OCC_SMEM, stream>>>(p);
    } else {
        if (p.ckq) k_attn_occ2_t<false, true><<<grid, AT_THREADS, OCC_SMEM, stream>>>(p);
        else k_attn_occ2_t<false, false><<<grid, AT_THREADS, OCC_SMEM, stream>>>(p);
    }
}
} // namespace sweep_variant
