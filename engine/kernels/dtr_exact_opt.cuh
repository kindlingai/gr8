// Bit-exact dense trellis o_proj & q_b (+ fused idx_q) decode kernels for GLM-5.3 (GR8_DTR_EXACT_OPT=1, default off).
//
// Preserves exact IEEE-754 left-to-right reduction order (0 ULP vs stock dtr::k_dtr) while optimizing:
//   - k_dtr_rot_par: parallelizes Hadamard-128 pre-rotation across (K/128, M) instead of 3 CTAs serializing M.
//   - k_op_sub2_xsmem<NT, PARTS>: 512-thread SUB=2, UNR=4, 1-barrier exact trellis o_proj with cooperative smem-staged
//     xa and paired 16-byte BF16 stores direct to the send slot (0 global partials) for stock_parts in {2, 4}
//     (falls back to stock for stock_parts == 32, e.g. TP=4 M=1).
//   - k_qb_sub1_ks4<NT>: 256-thread ks=4, sub=1, UNR=4, 0-dynamic-smem exact 16-bucket trellis q_b with direct __ldcg
//     epilogue and paired 16-byte stores (solo q_b at small/mid M).
//   - k_qb_ks2_sub2<NT>: 512-thread ks=2, sub=2, UNR=4, 4-slot smem exact 16-bucket trellis q_b with direct __ldcg
//     epilogue and paired 16-byte stores (solo q_b at larger M).
//   - k_qb_idxq_ks2<NT>: fuses MXFP8 idx_q (ks=32, last-CTA L2->slot gather publish) with exact ks=2, sub=2, UNR=4
//     trellis q_b into one 512-thread launch on indexer layers.
//
// Self-contained: leaves dtr.cuh, densev.cuh, bind.cuh, glue.cuh and dec_dense_opt.cuh unmodified.
#pragma once
#include <algorithm>
#include <cstdint>
#include "dec_dense_opt.cuh"

namespace dtr_exact_opt
{

#if defined(DEC_SLOT) && DEC_SLOT >= 2 && !G6_MODEL_FLASH && !G6_MODEL_DS41 && !G6_MODEL_INK2

constexpr int SMP = 132;  // padded smem stride (floats) for bank-conflict-free 128-col rows

// ------------------------------------------------------------------------------------------------ pre-rotation

__global__ void __launch_bounds__(256) k_dtr_rot_par(const __half* x, int ldx, const __half* suh, u32* xa, int K, int M)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nblk_k = K >> 7;
    const int task = blockIdx.x * 8 + warp;
    if (task >= M * nblk_k) return;
    const int blk = task / M;
    const int m = task - blk * M;
    const int k = blk * 128 + lane * 4, K16 = K >> 4;
    float s[4], xv[4];
    dtr::ld4h(suh + k, s);
    dtr::ld4h(x + static_cast<size_t>(m) * ldx + k, xv);
    float v[4] = {xv[0] * s[0], xv[1] * s[1], xv[2] * s[2], xv[3] * s[3]};
    had128(v, lane);
    store_frag_row(xa + static_cast<size_t>(m >> 3) * K16 * 64, blk * 8, m & 7, lane, v);
}

inline void launch_rot_par(const __half* x, int ldx, const __half* suh, u32* xa, int K, int M, cudaStream_t st)
{
    if (!x || !suh || !xa || K <= 0 || M <= 0) return;
    const int tasks = M * (K >> 7);
    k_dtr_rot_par<<<static_cast<unsigned>((tasks + 7) >> 3), 256, 0, st>>>(x, ldx, suh, xa, K, M);
}

// ------------------------------------------------------------------------------------------------ device helpers

template <int NT>
__device__ __forceinline__ void store_acc_smem(float* dst, int stride, int M, int col, int c, const float (&acc)[NT][4])
{
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M) { dst[m0 * stride + col] = acc[t][0]; dst[m0 * stride + col + 8] = acc[t][2]; }
        if (m0 + 1 < M) { dst[(m0 + 1) * stride + col] = acc[t][1]; dst[(m0 + 1) * stride + col + 8] = acc[t][3]; }
    }
}

template <int NT>
__device__ __forceinline__ void store_acc_global(float* dst, int M, int col, int c, const float (&acc)[NT][4])
{
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int m0 = t * 8 + 2 * c;
        if (m0 < M) { dst[m0 * 128 + col] = acc[t][0]; dst[m0 * 128 + col + 8] = acc[t][2]; }
        if (m0 + 1 < M) { dst[(m0 + 1) * 128 + col] = acc[t][1]; dst[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
    }
}

// Cooperative smem-staged xa loop with 4-tile double-buffered weight prefetch (exact sequential MMA order).
template <int NT>
__device__ __forceinline__ void run_groups_xsmem_u4(const Tr3Lane<dtr::BITS, true>& ld,
                                                    const uint8_t* gb, long long sG,
                                                    const u32* xa_base, int s0_grp, int xts,
                                                    int ng, int j, int lane,
                                                    uint2* sm_xq,
                                                    float (&acc)[NT][4])
{
    constexpr int TB = 32 * dtr::BITS;
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    u32 na[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);

    for (int g = 0; g < ng; ++g)
    {
        const int sg = s0_grp + g * 8;
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            sm_xq[t * 256 + j * 32 + lane] = __ldg(reinterpret_cast<const uint2*>(
                xa_base + static_cast<size_t>(t) * xts + static_cast<size_t>(sg + j) * 64 + lane * 2));
        __syncthreads();

        u32 wa[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) wa[u] = na[u];
        const uint8_t* gb_hi = gb + 4 * 8 * TB;
        #pragma unroll
        for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb_hi + u * (8 * TB), lane);
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = sm_xq[t * 256 + u * 32 + lane];
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }

        #pragma unroll
        for (int u = 0; u < 4; ++u) wa[u] = na[u];
        if (g + 1 < ng)
        {
            gb += sG;
            #pragma unroll
            for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = sm_xq[t * 256 + (u + 4) * 32 + lane];
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
        __syncthreads();
    }
}

// Exact q_b group loop with 4-tile double-buffered weight prefetch across group boundaries.
template <int NT, int NG, class Emit>
__device__ __forceinline__ void run_qb_groups_u4(const Tr3Lane<dtr::BITS, true>& ld,
                                                 const uint8_t* gb, long long sG,
                                                 const u32* xb, int xts,
                                                 int lane, Emit&& emit)
{
    constexpr int TB = 32 * dtr::BITS;
    u32 na[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
    for (int g = 0; g < NG; ++g)
    {
        float gacc[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) gacc[t][0] = gacc[t][1] = gacc[t][2] = gacc[t][3] = 0.f;

        u32 wa[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) wa[u] = na[u];
        const uint8_t* gb_hi = gb + 4 * 8 * TB;
        #pragma unroll
        for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb_hi + u * (8 * TB), lane);
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + u * 64));
                mma16816(gacc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }

        #pragma unroll
        for (int u = 0; u < 4; ++u) wa[u] = na[u];
        if (g + 1 < NG)
        {
            gb += sG;
            #pragma unroll
            for (int u = 0; u < 4; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + (u + 4) * 64));
                mma16816(gacc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
        xb += 8 * 64;
        emit(g, gacc);
    }
}

// ------------------------------------------------------------------------------------------------ o_proj (E5)

inline size_t op_sub2_smem(int parts, int M, int nt)
{
    const int nslots = 1 + (parts / 2);
    size_t b = static_cast<size_t>(nslots) * M * SMP * sizeof(float);
    b = (b + 15) & ~size_t(15);
    b += static_cast<size_t>(2) * nt * 256 * sizeof(uint2);
    return b + 16;
}

template <int NT, int PARTS>
__global__ void __launch_bounds__(512, 1) k_op_sub2_xsmem(dec_dense_opt::DtrG a)
{
    static_assert(PARTS == 2 || PARTS == 4, "PARTS must be 2 or 4");
    constexpr int TB = 32 * dtr::BITS, NW = 16, HP = PARTS / 2;
    extern __shared__ __align__(16) float smf[];
    const int nblk = a.N >> 7, M = a.M;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, qq = warp >> 3;
    const int nb = blockIdx.x;
    const int K16 = a.K >> 4, sp = K16 / PARTS, ng = sp >> 3;
    const int xts = K16 * 64;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    const uint8_t* w0 = a.w + nb * a.sB + j * TB;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);

    size_t off_f = static_cast<size_t>(1 + HP) * M * SMP;
    off_f = (off_f + 3) & ~size_t(3);
    uint2* sm_xq = reinterpret_cast<uint2*>(smf + off_f) + qq * NT * 256;

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    #pragma unroll
    for (int pp = 0; pp < HP; ++pp)
    {
        const int part_id = qq * HP + pp;
        const int s0 = part_id * sp;
        const uint8_t* gb = w0 + static_cast<long long>(s0 >> 3) * a.sG;
        float gacc[NT][4];
        run_groups_xsmem_u4<NT>(ld, gb, a.sG, a.xa, s0, xts, ng, j, lane, sm_xq, gacc);

        if (qq == 0)
        {
            if (pp == 0)
            {
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] = gacc[t][0]; acc[t][1] = gacc[t][1];
                    acc[t][2] = gacc[t][2]; acc[t][3] = gacc[t][3];
                }
            }
            else
            {
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] += gacc[t][0]; acc[t][1] += gacc[t][1];
                    acc[t][2] += gacc[t][2]; acc[t][3] += gacc[t][3];
                }
            }
        }
        else
        {
            store_acc_smem<NT>(smf + static_cast<size_t>(1 + pp) * M * SMP, SMP, M, col, c, gacc);
        }
    }
    if (qq == 0) store_acc_smem<NT>(smf, SMP, M, col, c, acc);
    __syncthreads();

    const int n0 = nb * 128 + lane * 4;
    uint16_t* yo = reinterpret_cast<uint16_t*>(dslot::ptr(a.slot));
    float s[4];
    dtr::ld4h(a.svh + n0, s);
    for (int m = warp; m < M; m += NW)
    {
        const float* b0 = smf + static_cast<size_t>(m) * SMP + lane * 4;
        float v[4] = {0.f + b0[0], 0.f + b0[1], 0.f + b0[2], 0.f + b0[3]};
        #pragma unroll
        for (int r = 1; r <= HP; ++r)
        {
            const float* br = smf + (static_cast<size_t>(r) * M + m) * SMP + lane * 4;
            v[0] += br[0]; v[1] += br[1]; v[2] += br[2]; v[3] += br[3];
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        const uint2 mine = make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]));
        const u32 ox = __shfl_xor_sync(0xffffffffu, mine.x, 1), oy = __shfl_xor_sync(0xffffffffu, mine.y, 1);
        if (!(lane & 1))
            *reinterpret_cast<uint4*>(yo + static_cast<size_t>(m) * a.ldy + n0) = make_uint4(mine.x, mine.y, ox, oy);
    }
    dslot::publish(a.slot, !(lane & 1) && warp < M, static_cast<unsigned>(nblk));
}

// ------------------------------------------------------------------------------------------------ q_b + idx_q (F4)

inline size_t qb_ks2_smem(int M) { return static_cast<size_t>(4) * M * SMP * sizeof(float) + 32; }

template <int NT>
__global__ void __launch_bounds__(256, 2) k_qb_sub1_ks4(dec_dense_opt::DtrG a)
{
    constexpr int TB = 32 * dtr::BITS, NW = 8, KS = 4, NG = 4, NPARTS = 13;
    __shared__ int last_p;
    const int nblk = a.N >> 7, M = a.M;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp;
    const int nb = blockIdx.x % nblk, p = blockIdx.x / nblk;
    const int K16 = a.K >> 4, xts = K16 * 64;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    const int g0 = p * NG;
    const uint8_t* gb = a.w + nb * a.sB + j * TB + static_cast<long long>(g0) * a.sG;
    const u32* xb = a.xa + static_cast<size_t>(g0 * 8) * 64 + lane * 2;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    run_qb_groups_u4<NT, NG>(ld, gb, a.sG, xb, xts, lane, [&](int g, const float (&gacc)[NT][4]) {
        if (p == 0)
        {
            if (g == 0)
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] = gacc[t][0]; acc[t][1] = gacc[t][1];
                    acc[t][2] = gacc[t][2]; acc[t][3] = gacc[t][3];
                }
            else
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] += gacc[t][0]; acc[t][1] += gacc[t][1];
                    acc[t][2] += gacc[t][2]; acc[t][3] += gacc[t][3];
                }
        }
        else
        {
            const int slot_i = 1 + (p - 1) * NG + g;
            store_acc_global<NT>(a.part + (static_cast<size_t>(nb) * NPARTS + slot_i) * M * 128, M, col, c, gacc);
        }
    });
    if (p == 0) store_acc_global<NT>(a.part + static_cast<size_t>(nb) * NPARTS * M * 128, M, col, c, acc);

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last_p = atomicAdd(a.cnt + nb, 1) == (KS - 1);
    __syncthreads();
    if (!last_p) return;
    __threadfence();

    const int n0 = nb * 128 + lane * 4;
    uint16_t* yo = static_cast<uint16_t*>(a.y);
    const bool bf = a.odt == dtr::O_BF16;
    float s[4];
    dtr::ld4h(a.svh + n0, s);
    for (int m = warp; m < M; m += NW)
    {
        const float* pb = a.part + static_cast<size_t>(nb) * NPARTS * M * 128 + static_cast<size_t>(m) * 128 + lane * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 4
        for (int r = 0; r < NPARTS; ++r)
        {
            const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        const uint2 mine = bf ? make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]))
                              : make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
        const u32 ox = __shfl_xor_sync(0xffffffffu, mine.x, 1), oy = __shfl_xor_sync(0xffffffffu, mine.y, 1);
        if (!(lane & 1))
            *reinterpret_cast<uint4*>(yo + static_cast<size_t>(m) * a.ldy + n0) = make_uint4(mine.x, mine.y, ox, oy);
    }
    if (threadIdx.x == 0) a.cnt[nb] = 0;
}

template <int NT>
__device__ __forceinline__ void qb_ks2_sub2_body(const dec_dense_opt::DtrG& a, int bid, float* smf)
{
    constexpr int TB = 32 * dtr::BITS, NW = 16, NPARTS = 9;
    const int nblk = a.N >> 7, M = a.M;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, qq = warp >> 3;
    const int nb = bid % nblk, p = bid / nblk;
    const int K16 = a.K >> 4, xts = K16 * 64;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    const int g0 = p * 8 + qq * 4;
    const uint8_t* gb = a.w + nb * a.sB + j * TB + static_cast<long long>(g0) * a.sG;
    const u32* xb = a.xa + static_cast<size_t>(g0 * 8) * 64 + lane * 2;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    run_qb_groups_u4<NT, 4>(ld, gb, a.sG, xb, xts, lane, [&](int g, const float (&gacc)[NT][4]) {
        if (p == 0 && qq == 0)
        {
            if (g == 0)
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] = gacc[t][0]; acc[t][1] = gacc[t][1];
                    acc[t][2] = gacc[t][2]; acc[t][3] = gacc[t][3];
                }
            else
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    acc[t][0] += gacc[t][0]; acc[t][1] += gacc[t][1];
                    acc[t][2] += gacc[t][2]; acc[t][3] += gacc[t][3];
                }
        }
        else if (p == 0)
        {
            store_acc_smem<NT>(smf + static_cast<size_t>(g) * M * SMP, SMP, M, col, c, gacc);
        }
        else
        {
            const int slot_i = 1 + qq * 4 + g;
            store_acc_global<NT>(a.part + (static_cast<size_t>(nb) * NPARTS + slot_i) * M * 128, M, col, c, gacc);
        }
    });

    if (p == 0)
    {
        __syncthreads();
        if (qq == 0)
        {
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                const float* src = smf + static_cast<size_t>(r) * M * SMP;
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const int m0 = t * 8 + 2 * c;
                    if (m0 < M) { acc[t][0] += src[m0 * SMP + col]; acc[t][2] += src[m0 * SMP + col + 8]; }
                    if (m0 + 1 < M) { acc[t][1] += src[(m0 + 1) * SMP + col]; acc[t][3] += src[(m0 + 1) * SMP + col + 8]; }
                }
            }
            store_acc_global<NT>(a.part + static_cast<size_t>(nb) * NPARTS * M * 128, M, col, c, acc);
        }
    }

    __threadfence();
    __syncthreads();
    int* last = reinterpret_cast<int*>(smf + static_cast<size_t>(4) * M * SMP);
    if (threadIdx.x == 0) *last = atomicAdd(a.cnt + nb, 1) == 1;
    __syncthreads();
    if (!*last) return;
    __threadfence();

    const int n0 = nb * 128 + lane * 4;
    uint16_t* yo = static_cast<uint16_t*>(a.y);
    const bool bf = a.odt == dtr::O_BF16;
    float s[4];
    dtr::ld4h(a.svh + n0, s);
    for (int m = warp; m < M; m += NW)
    {
        const float* pb = a.part + static_cast<size_t>(nb) * NPARTS * M * 128 + static_cast<size_t>(m) * 128 + lane * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 4
        for (int r = 0; r < NPARTS; ++r)
        {
            const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        const uint2 mine = bf ? make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]))
                              : make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
        const u32 ox = __shfl_xor_sync(0xffffffffu, mine.x, 1), oy = __shfl_xor_sync(0xffffffffu, mine.y, 1);
        if (!(lane & 1))
            *reinterpret_cast<uint4*>(yo + static_cast<size_t>(m) * a.ldy + n0) = make_uint4(mine.x, mine.y, ox, oy);
    }
    if (threadIdx.x == 0) a.cnt[nb] = 0;
}

template <int NT>
__global__ void __launch_bounds__(512, 1) k_qb_ks2_sub2(dec_dense_opt::DtrG a)
{
    extern __shared__ __align__(16) float smf[];
    qb_ks2_sub2_body<NT>(a, blockIdx.x, smf);
}

template <int NT>
__global__ void __launch_bounds__(512, 1) k_qb_idxq_ks2(dec_dense_opt::FGv g, dec_dense_opt::DtrG d)
{
    extern __shared__ __align__(16) uint8_t sm_raw[];
    if (static_cast<int>(blockIdx.x) < g.ctas)
        dec_dense_opt::gv_fused_body<NT, 2>(g, blockIdx.x, reinterpret_cast<float4 (*)[32]>(sm_raw));
    else
        qb_ks2_sub2_body<NT>(d, blockIdx.x - g.ctas, reinterpret_cast<float*>(sm_raw));
}

// ------------------------------------------------------------------------------------------------ host dispatch

struct State
{
    bool on = false;
};

inline cudaError_t init_attrs()
{
    cudaError_t err = cudaSuccess;
    auto set = [&](const void* fn) {
        const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dtr::SMEM_MAX);
        if (err == cudaSuccess) err = e;
    };
    const int max_nt = g6mx::NT < 4 ? g6mx::NT : 4;
    for (int nt = 1; nt <= max_nt; ++nt)
    {
        g6mx::nt_dispatch(nt, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if constexpr (NTv <= 4)
            {
                set(reinterpret_cast<const void*>(k_op_sub2_xsmem<NTv, 2>));
                set(reinterpret_cast<const void*>(k_op_sub2_xsmem<NTv, 4>));
                set(reinterpret_cast<const void*>(k_qb_ks2_sub2<NTv>));
                set(reinterpret_cast<const void*>(k_qb_idxq_ks2<NTv>));
            }
        });
    }
    return err;
}

inline cudaError_t setup(State& s)
{
    if (!knob::dtr_exact_opt()) return cudaSuccess;
    if (s.on) return cudaSuccess;
    if (const cudaError_t e = init_attrs(); e != cudaSuccess) return e;
    s.on = true;
    return cudaSuccess;
}

inline bool qb_eligible(const bind::Layer& b, const bind::Scratch& sc, int M, int ks)
{
    if (M < 1 || M > 32) return false;
    if (!b.q_b_t.on || b.q_b_t.a.K != geom::QA || b.q_b_t.a.ks != 16 || b.q_b_t.sub != 1) return false;
    if ((b.q_b_t.a.N % 128) != 0 || (b.q_b_t.a.ldy % 8) != 0) return false;
    const int nblk = b.q_b_t.a.N / 128;
    if (nblk > sc.dt.cnt_n) return false;
    if (ks == 2)
    {
        if (qb_ks2_smem(M) > static_cast<size_t>(dtr::SMEM_MAX)) return false;
        return static_cast<size_t>(nblk) * 9 * M * 128 <= sc.dt.part_n;
    }
    if (ks == 4)
        return static_cast<size_t>(nblk) * 13 * M * 128 <= sc.dt.part_n;
    return false;
}

inline dec_dense_opt::DtrG make_qb_args(const bind::Layer& b, const bind::Scratch& sc, int M, int ks)
{
    dec_dense_opt::DtrG d;
    d.w = b.q_b_t.a.w[0]; d.sB = b.q_b_t.a.sB; d.sG = b.q_b_t.a.sG;
    d.K = b.q_b_t.a.K; d.N = b.q_b_t.a.N; d.ks = ks; d.M = M;
    d.xa = sc.xf_eh; d.svh = b.q_b_t.a.svh[0];
    d.odt = b.q_b_t.a.odt; d.y = b.q_b_t.a.y; d.ldy = b.q_b_t.a.ldy;
    d.part = sc.dt.part; d.cnt = sc.dt.cnt;
    return d;
}

// Returns true if this layer's trellis q_b (and idx_q when fused/present) was launched.
// Leaves non-trellis q_b layers unhandled so dec_dense_opt or stock handles them.
inline bool launch_qb_idxq(const State& s, const bind::Layer& b, const bind::Scratch& sc, const dtr::Mat& m_qb,
                           const __half* q_a_n, int M, bool noiq, bool qa_rot_done,
                           const g6::FabricDev* qg, const u64* stepp, unsigned* qtickets, int ciq,
                           bool& did_qs_put, cudaStream_t st)
{
    did_qs_put = false;
    if (!s.on || M < 1 || M > 32) return false;
    const bool qb_ks2_ok = qb_eligible(b, sc, M, 2);
    const bool qb_ks4_ok = qb_eligible(b, sc, M, 4);
    if (!qb_ks2_ok && !qb_ks4_ok) return false;

    const bool iq_on = !noiq && b.idx_q.on && b.idx_q.codec == bind::MXFP8 && b.idx_q.ks == 32 && (b.idx_q.O % 16) == 0;
    const bool can_fuse = iq_on && qb_ks2_ok;

    if (!qa_rot_done)
    {
        bind::launch(b.xf_qa, st);
        launch_rot_par(q_a_n, geom::QA, m_qb.suh, sc.xf_eh, geom::QA, M, st);
    }

    const int nt = g6mx::tiles(M);
    if (can_fuse)
    {
        const dec_dense_opt::FGv g = dec_dense_opt::make_idxq_args(b, M, qg, stepp, qtickets, ciq);
        const dec_dense_opt::DtrG d = make_qb_args(b, sc, M, 2);
        const size_t smem = std::max(static_cast<size_t>(32 * 32 * sizeof(float4)), qb_ks2_smem(M));
        const int grid = g.ctas + (d.N / 128) * 2;
        g6mx::nt_dispatch(nt, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if constexpr (NTv <= 4) k_qb_idxq_ks2<NTv><<<grid, 512, smem, st>>>(g, d);
        });
        did_qs_put = (qg && ciq >= 0);
        return true;
    }

    if (!noiq && b.idx_q.on)
    {
        if (iq_on)
        {
            const dec_dense_opt::FGv g = dec_dense_opt::make_idxq_args(b, M, qg, stepp, qtickets, ciq);
            g6mx::nt_dispatch(nt, [&](auto N) {
                constexpr int NTv = decltype(N)::value;
                dec_dense_opt::k_gv_fused<NTv><<<g.ctas, 512, 0, st>>>(g);
            });
            did_qs_put = (qg && ciq >= 0);
        }
        else
        {
            bind::launch(b.idx_q, st);
        }
    }

    const bool use_ks2_solo = (M >= 16) || (b.q_b_t.a.N <= 2816 && M >= 8);
    if ((use_ks2_solo && qb_ks2_ok) || !qb_ks4_ok)
    {
        const dec_dense_opt::DtrG d = make_qb_args(b, sc, M, 2);
        const size_t smem = qb_ks2_smem(M);
        const int grid = (d.N / 128) * 2;
        g6mx::nt_dispatch(nt, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if constexpr (NTv <= 4) k_qb_ks2_sub2<NTv><<<grid, 512, smem, st>>>(d);
        });
    }
    else
    {
        const dec_dense_opt::DtrG d = make_qb_args(b, sc, M, 4);
        const int grid = (d.N / 128) * 4;
        g6mx::nt_dispatch(nt, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if constexpr (NTv <= 4) k_qb_sub1_ks4<NTv><<<grid, 256, 0, st>>>(d);
        });
    }
    return true;
}

// Returns true if o_proj was launched with exact 0-scratch k_op_sub2_xsmem (+ k_dtr_rot_par).
inline bool launch_o_proj(const State& s, const bind::Layer& b, const bind::Scratch& sc, int M, cudaStream_t st)
{
    if (!s.on || M < 1 || M > 32) return false;
    if (!b.o_proj_t.on || !b.o_proj_t.rot || !b.o_proj_t.a.slot.send || b.o_proj_t.a.odt != dtr::O_BF16) return false;
    if ((b.o_proj_t.a.N % 128) != 0 || (b.o_proj_t.a.ldy % 8) != 0) return false;
    const int stock_parts = b.o_proj_t.a.ks * b.o_proj_t.sub;
    if (stock_parts != 2 && stock_parts != 4) return false;
    const int K16 = b.o_proj_t.a.K / 16;
    if (K16 % (8 * stock_parts) != 0) return false;
    const int nt = g6mx::tiles(M);
    const size_t smem = op_sub2_smem(stock_parts, M, nt);
    if (smem > static_cast<size_t>(dtr::SMEM_MAX)) return false;

    launch_rot_par(b.o_proj_t.a.x, b.o_proj_t.a.ldx, b.o_proj_t.a.suh[0], sc.xf_eh, b.o_proj_t.a.K, M, st);
    dec_dense_opt::DtrG d;
    d.w = b.o_proj_t.a.w[0]; d.sB = b.o_proj_t.a.sB; d.sG = b.o_proj_t.a.sG;
    d.K = b.o_proj_t.a.K; d.N = b.o_proj_t.a.N; d.ks = 1; d.M = M;
    d.xa = sc.xf_eh; d.svh = b.o_proj_t.a.svh[0];
    d.odt = dtr::O_BF16; d.ldy = b.o_proj_t.a.ldy;
    d.slot = b.o_proj_t.a.slot;
    const int grid = d.N / 128;
    g6mx::nt_dispatch(nt, [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        if constexpr (NTv <= 4)
        {
            if (stock_parts == 4) k_op_sub2_xsmem<NTv, 4><<<grid, 512, smem, st>>>(d);
            else k_op_sub2_xsmem<NTv, 2><<<grid, 512, smem, st>>>(d);
        }
    });
    return true;
}

#else

struct State { bool on = false; };
inline cudaError_t setup(State&) { return cudaSuccess; }
inline bool launch_qb_idxq(const State&, const bind::Layer&, const bind::Scratch&, const dtr::Mat&, const __half*, int, bool, bool,
                           const g6::FabricDev*, const u64*, unsigned*, int, bool& did_qs_put, cudaStream_t) { did_qs_put = false; return false; }
inline bool launch_o_proj(const State&, const bind::Layer&, const bind::Scratch&, int, cudaStream_t) { return false; }

#endif

}  // namespace dtr_exact_opt
