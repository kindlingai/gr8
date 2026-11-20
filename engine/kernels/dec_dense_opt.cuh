// Bit-exact fused decode dense kernels for GLM-5.3 (GR8_DEC_DENSE_OPT=1, default off).
//
// Keeps every output bit-identical (0 ULP) to the stock decode path while cutting launch overhead, L2/DRAM traffic,
// and pinned-slot writes across the attention projections:
//   - S1+S2 (layer input under GR8_QA_SPLIT=2): k_gv_fused_1p fuses q_a (MXFP8 ks=16, last-CTA L2->pinned gather
//     publish), kv_a (MXFP8 ks=16), and idx_k + idx_w (BF16W ks=32 as 2-CTA k-pairs with inline fp16->bf16 fragment
//     conversion) into one 512-thread launch.
//   - q_a_norm: k_qa_norm_rot fuses RMSNorm(q_a) + xf_qa fragment table + H128(q_a_n * suh_qb) rotation into one pass.
//   - S3 (q_b + idx_q): k_qb_idxq_exact fuses idx_q (MXFP8 ks=32, last-CTA gather publish) with exact 16-bucket
//     trellis q_b (ks=1, 0 global partials, stock IEEE-754 left-to-right sum order) for M <= 16; k_dtr_gb10_exact
//     handles solo q_b (ks=2, M <= 32) and k_gv_fused handles solo idx_q.
//   - S4 (o_proj): k_dtr_gb10_op runs 0-scratch ks=1 trellis o_proj with paired 16-byte BF16 stores direct to the
//     send slot wherever stock_parts == sub (exact sum order) and M is in the faster regime.
//
// Self-contained: leaves dtr.cuh, densev.cuh, bind.cuh and glue.cuh unmodified.
#pragma once
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "bind.cuh"
#include "../comms/fabric.h"

namespace dec_dense_opt
{

#if defined(DEC_SLOT) && DEC_SLOT >= 2 && !G6_MODEL_FLASH && !G6_MODEL_DS41 && !G6_MODEL_INK2

static_assert(geom::QA == 2048 && geom::IDX_DIM == 128 && geom::IDX_HEADS == 32,
              "dec_dense_opt: GLM-5.3 attention geometry required");

constexpr int PAIR_T = geom::IDX_DIM / 16 + geom::IDX_HEADS / 16;   // 10 tiles (8 idx_k + 2 idx_w)
constexpr int SMP = 132;                                            // padded sm_blk row (floats)

enum { C_MX = 1, C_BF16 = 2, C_BF16CVT = 3 };

struct FPart
{
    const uint8_t* w = nullptr;
    const u32* xfrag = nullptr;
    void* y = nullptr;
    int codec = C_MX, K = 0, ot = 0, ks = 1, ld = 0, odt = gv::O_F16, cta0 = 0;
    int k_pair = 0;
    float4* pair_part = nullptr;      // [PAIR_T][17][g6mx::NT][32] float4
    unsigned* pair_cnt = nullptr;     // [nl][PAIR_T], zeroed once at boot and reset by last CTA
};

struct FGv
{
    int n = 0, M = 0, ctas = 0;
    FPart p[4];
    dslot::Slot slot;                 // send != nullptr: part slot_part copies L2 -> send slot on last CTA, then rings
    int slot_part = -1, slot_ld = 0;
    int slot_last_cta = 0;
    int slot_ctas = 0;
};

struct DtrG
{
    const uint8_t* w = nullptr;
    long long sB = 0, sG = 0;
    int K = 0, N = 0, ks = 1, M = 0;
    const u32* xa = nullptr;
    const __half* svh = nullptr;
    int odt = dtr::O_F16;
    void* y = nullptr;
    int ldy = 0;
    dslot::Slot slot;                 // o_proj direct-to-slot BF16 publish
    float* part = nullptr;            // ks == 2 exact q_b partials
    int* cnt = nullptr;               // ks == 2 exact q_b arrival counters
};

inline int part_ctas(const FPart& p) { return p.k_pair ? 2 * p.ot : p.ot; }
inline size_t dtrg_smem(int M) { return static_cast<size_t>(M) * SMP * 4 + 16; }
inline size_t dtrg_exact_smem(int ks, int M)
{
    const int nslots = (ks == 2) ? 5 : 9;
    return static_cast<size_t>(nslots) * M * SMP * 4 + 16;
}

inline dslot::Slot make_slot(const g6::FabricDev& f, const u64* stepp, unsigned* tickets, int c)
{
    dslot::Slot s;
    s.send = f.send; s.bell = f.doorbell; s.stepp = stepp;
    s.half = g6::fab_slot_base(f, f.ncoll); s.off = g6::fab_slot_base(f, c);
    s.ncoll = f.ncoll; s.c = c; s.ticket = tickets + c; s.xt = f.xt;
    return s;
}

// ------------------------------------------------------------------------------------------------ device helpers

template <int NT>
__device__ __forceinline__ void run_cvt(const GvGroup<Bf16w>& g, const u32* bf, int ks, float (&acc)[NT][4], int ts)
{
    u32 a[4];
    Bf16w::dec(g.c, a);
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        Bf16w::mma_a(acc[t], a, bind::h2_to_bf2(bf[t * ts + ks * 64]), bind::h2_to_bf2(bf[t * ts + ks * 64 + 1]));
}

template <int NT>
__device__ __forceinline__ void tile_part_cvt(const uint8_t* tile, const u32* bf, int ts, int g0, int g1, float (&acc)[NT][4])
{
    constexpr int PFB = 2;
    const int lane = threadIdx.x & 31, Q = g1 - g0;
    GvGroup<Bf16w> nx[PFB];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
        if (p < Q) nx[p].load(tile + static_cast<size_t>(g0 + p) * Bf16w::GR, lane);
    for (int q = 0; q < Q; q += PFB)
    {
        GvGroup<Bf16w> cur[PFB];
        #pragma unroll
        for (int p = 0; p < PFB; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
            if (q + PFB + p < Q) nx[p].load(tile + static_cast<size_t>(g0 + q + PFB + p) * Bf16w::GR, lane);
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
            if (q + p < Q) run_cvt<NT>(cur[p], bf, g0 + q + p, acc, ts);
    }
}

template <int NT>
__device__ __forceinline__ void run_part(const FPart& p, const uint8_t* tile, const u32* bf, int ts, int g0, int g1, float (&acc)[NT][4])
{
    if (p.codec == C_MX) gv_tile_part<Mxfp8, NT>(tile, bf, ts, g0, g1, acc);
    else if (p.codec == C_BF16) gv_tile_part<Bf16w, NT>(tile, bf, ts, g0, g1, acc);
    else tile_part_cvt<NT>(tile, bf, ts, g0, g1, acc);
}

__device__ __forceinline__ void ring_bell(const dslot::Slot& s)
{
    const u64 step = *s.stepp;
    u64* xr = s.xt ? reinterpret_cast<u64*>(s.xt) + ((size_t)(step % g6::XT_STEPS) * s.ncoll + s.c) * g6::XT_W : nullptr;
    if (xr) { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); xr[g6::XT_PUB] = t; }
    *s.ticket = 0;
    __threadfence_system();
    const u64 seq = step * (u64)s.ncoll + s.c + 1;
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(s.bell + s.c), "l"(seq) : "memory");
    if (xr) { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); xr[g6::XT_DB] = t; xr[g6::XT_SEQ] = seq; }
}

__device__ __forceinline__ void slot_gather_last(const dslot::Slot& s, unsigned nct, const uint4* src, int ld16, int w16, int M,
                                                 int* flag, bool wrote)
{
    if (wrote) __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *flag = atomicAdd(s.ticket, 1u) == nct - 1u;
    __syncthreads();
    if (!*flag) return;
    if (threadIdx.x == 0) __threadfence();
    __syncthreads();
    uint4* dst = reinterpret_cast<uint4*>(dslot::ptr(s));
    const int nu = M * w16;
    for (int u = threadIdx.x; u < nu; u += blockDim.x)
    {
        const int r = u / w16, k = u - r * w16;
        dst[u] = __ldcg(src + static_cast<size_t>(r) * ld16 + k);
    }
    if (static_cast<int>(threadIdx.x) < nu) __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) ring_bell(s);
}

__device__ __forceinline__ void put_tile(const FPart& p, int M, int ot, int t, int lane, float a0, float a1, float a2, float a3)
{
    const int r = lane >> 2, c = lane & 3;
    const size_t o = static_cast<size_t>(ot) * 16 + r;
    const int m0 = 8 * t + 2 * c;
    if (m0 < M)
    {
        gv::put(p.y, static_cast<size_t>(m0) * p.ld + o, a0, p.odt);
        gv::put(p.y, static_cast<size_t>(m0) * p.ld + o + 8, a2, p.odt);
    }
    if (m0 + 1 < M)
    {
        gv::put(p.y, static_cast<size_t>(m0 + 1) * p.ld + o, a1, p.odt);
        gv::put(p.y, static_cast<size_t>(m0 + 1) * p.ld + o + 8, a3, p.odt);
    }
}

// ------------------------------------------------------------------------------------------------ k_gv_fused / k_gv_fused_1p

template <int NT>
__device__ __forceinline__ void gv_pair_body(const FPart& p, int M, int rel, float4 (*red)[32])
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = rel >> 1, hk = rel & 1;
    const int G = p.codec == C_MX ? 2 : 1, GR = p.codec == C_MX ? 528 : 512;
    const int NG2 = p.K / (16 * G), NGP = NG2 / p.ks, ts = p.K * 4;
    const uint8_t* tile = p.w + static_cast<size_t>(ot) * NG2 * GR;
    const u32* bf = p.xfrag + lane * 2;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    const int part = hk * 16 + warp;
    run_part<NT>(p, tile, bf, ts, part * NGP, (part + 1) * NGP, acc);
    float4* base = p.pair_part + static_cast<size_t>(ot) * 17 * NT * 32;
    if (hk == 0)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            red[warp][lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
            __syncthreads();
            if (warp == 0)
            {
                for (int q = 1; q < 16; ++q)
                {
                    const float4 v = red[q][lane];
                    acc[t][0] += v.x; acc[t][1] += v.y; acc[t][2] += v.z; acc[t][3] += v.w;
                }
                base[t * 32 + lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
            }
            __syncthreads();
        }
    }
    else
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            base[static_cast<size_t>(1 + warp) * NT * 32 + t * 32 + lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
    }
    if (hk == 1 || warp == 0) __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(&red[0][0]);
    if (threadIdx.x == 0) *flag = atomicAdd(p.pair_cnt + ot, 1u) == 1u;
    __syncthreads();
    if (!*flag || warp != 0) return;
    __threadfence();
    if (lane == 0) p.pair_cnt[ot] = 0;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        float4 s = __ldcg(base + t * 32 + lane);
        for (int q = 1; q <= 16; ++q)
        {
            const float4 v = __ldcg(base + static_cast<size_t>(q) * NT * 32 + t * 32 + lane);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        put_tile(p, M, ot, t, lane, s.x, s.y, s.z, s.w);
    }
}

template <int NT, int MAXP>
__device__ __forceinline__ void gv_fused_body(const FGv& a, int bid, float4 (*red)[32])
{
    constexpr int NW = 16;
    int pi = 0;
    #pragma unroll
    for (int k = 1; k < 4; ++k)
        if (k < a.n && bid >= a.p[k].cta0) pi = k;
    const FPart p = a.p[pi];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (p.k_pair) { gv_pair_body<NT>(p, a.M, bid - p.cta0, red); return; }
    const int ot = bid - p.cta0;
    const int ppw = p.ks / NW;
    const int G = p.codec == C_MX ? 2 : 1, GR = p.codec == C_MX ? 528 : 512;
    const int NG2 = p.K / (16 * G), NGP = NG2 / p.ks, ts = p.K * 4;
    const uint8_t* tile = p.w + static_cast<size_t>(ot) * NG2 * GR;
    const u32* bf = p.xfrag + lane * 2;
    float acc[MAXP][NT][4];
    #pragma unroll
    for (int pp = 0; pp < MAXP; ++pp)
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[pp][t][0] = acc[pp][t][1] = acc[pp][t][2] = acc[pp][t][3] = 0.f;
    #pragma unroll
    for (int pp = 0; pp < MAXP; ++pp)
    {
        if (pp >= ppw) break;
        const int part = warp + pp * NW;
        run_part<NT>(p, tile, bf, ts, part * NGP, (part + 1) * NGP, acc[pp]);
    }
    const int r = lane >> 2, c = lane & 3;
    const size_t o = static_cast<size_t>(ot) * 16 + r;
    const bool slot_on = a.slot.send && pi == a.slot_part;
    __half* srow = slot_on && !a.slot_last_cta ? reinterpret_cast<__half*>(dslot::ptr(a.slot)) : nullptr;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        if (p.ks > 1)
        {
            #pragma unroll
            for (int pp = 0; pp < MAXP; ++pp)
                if (pp < ppw) red[warp + pp * NW][lane] = make_float4(acc[pp][t][0], acc[pp][t][1], acc[pp][t][2], acc[pp][t][3]);
            __syncthreads();
            if (warp == 0)
                for (int q = 1; q < p.ks; ++q)
                {
                    const float4 v = red[q][lane];
                    acc[0][t][0] += v.x; acc[0][t][1] += v.y; acc[0][t][2] += v.z; acc[0][t][3] += v.w;
                }
            __syncthreads();
        }
        if (warp == 0)
        {
            const int m0 = 8 * t + 2 * c;
            put_tile(p, a.M, ot, t, lane, acc[0][t][0], acc[0][t][1], acc[0][t][2], acc[0][t][3]);
            if (srow)
            {
                if (m0 < a.M)
                {
                    srow[static_cast<size_t>(m0) * a.slot_ld + o] = __float2half_rn(acc[0][t][0]);
                    srow[static_cast<size_t>(m0) * a.slot_ld + o + 8] = __float2half_rn(acc[0][t][2]);
                }
                if (m0 + 1 < a.M)
                {
                    srow[static_cast<size_t>(m0 + 1) * a.slot_ld + o] = __float2half_rn(acc[0][t][1]);
                    srow[static_cast<size_t>(m0 + 1) * a.slot_ld + o + 8] = __float2half_rn(acc[0][t][3]);
                }
            }
        }
    }
    if (slot_on)
    {
        const unsigned nct = static_cast<unsigned>(a.slot_ctas ? a.slot_ctas : a.ctas);
        if (a.slot_last_cta)
            slot_gather_last(a.slot, nct, static_cast<const uint4*>(p.y), p.ld / 8, a.slot_ld / 8, a.M,
                             reinterpret_cast<int*>(&red[0][0]), warp == 0);
        else dslot::publish(a.slot, warp == 0, nct);
    }
}

template <int NT>
__global__ void __launch_bounds__(512) k_gv_fused_1p(FGv a)
{
    __shared__ float4 red[16][32];
    gv_fused_body<NT, 1>(a, blockIdx.x, red);
}

template <int NT>
__global__ void __launch_bounds__(512) k_gv_fused(FGv a)
{
    __shared__ float4 red[32][32];
    gv_fused_body<NT, 2>(a, blockIdx.x, red);
}

// ------------------------------------------------------------------------------------------------ k_qa_norm_rot

__global__ void __launch_bounds__(512) k_qa_norm_rot(const __half* in, const __nv_bfloat16* w, __half* out, u32* xf16,
                                                     const __half* suh_qb, u32* xa_qb)
{
    constexpr int QAW = geom::QA;
    __shared__ float wsum[8];
    __shared__ __align__(16) __half sm_out[QAW];
    const int m = blockIdx.x, M = gridDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const __half* row = in + static_cast<size_t>(m) * QAW;
    float xv[8];
    if (tid < 256)
    {
        float ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            xv[k] = __half2float(row[tid + k * 256]);
            ss = fmaf(xv[k], xv[k], ss);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) wsum[warp] = ss;
    }
    __syncthreads();
    if (tid < 256)
    {
        float tot = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) tot += wsum[i];
        const float inv = rsqrtf(tot / static_cast<float>(QAW) + geom::RMS_EPS);
        #pragma unroll
        for (int k = 0; k < 8; ++k) sm_out[tid + k * 256] = __float2half_rn(xv[k] * inv * __bfloat162float(w[tid + k * 256]));
    }
    __syncthreads();
    if (tid < 256) reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * QAW)[tid] = reinterpret_cast<const uint4*>(sm_out)[tid];
    const int tile = m >> 3, r = m & 7;
    constexpr int NKS = QAW >> 4;
    if (xf16 && tid < NKS)
    {
        const int ks = tid;
        const uint4* s = reinterpret_cast<const uint4*>(sm_out + ks * 16);
        const uint4 lo = s[0], hi = s[1];
        uint4* d = reinterpret_cast<uint4*>(xf16 + static_cast<size_t>(tile) * NKS * 64 + (static_cast<size_t>(ks) * 32 + r * 4) * 2);
        d[0] = make_uint4(lo.x, hi.x, lo.y, hi.y);
        d[1] = make_uint4(lo.z, hi.z, lo.w, hi.w);
        if (m == M - 1)
            for (int z = r + 1; z < 8; ++z)
            {
                uint4* dz = reinterpret_cast<uint4*>(xf16 + static_cast<size_t>(tile) * NKS * 64 + (static_cast<size_t>(ks) * 32 + z * 4) * 2);
                dz[0] = make_uint4(0u, 0u, 0u, 0u);
                dz[1] = make_uint4(0u, 0u, 0u, 0u);
            }
    }
    if (suh_qb && xa_qb)
    {
        const int k = warp * 128 + lane * 4;
        const uint2 rx = *reinterpret_cast<const uint2*>(sm_out + k);
        const uint2 rs = *reinterpret_cast<const uint2*>(suh_qb + k);
        const float2 ax = __half22float2(as_h2(rx.x)), bx = __half22float2(as_h2(rx.y));
        const float2 as = __half22float2(as_h2(rs.x)), bs = __half22float2(as_h2(rs.y));
        float v[4] = {ax.x * as.x, ax.y * as.y, bx.x * bs.x, bx.y * bs.y};
        had128(v, lane);
        store_frag_row(xa_qb + static_cast<size_t>(tile) * NKS * 64, warp * 8, r, lane, v);
    }
}

// ------------------------------------------------------------------------------------------------ k_dtr_gb10_exact / k_qb_idxq_exact

template <int NT>
__device__ __forceinline__ void dtr_gb10_exact_body(const DtrG& a, int bid, float* smf)
{
    constexpr int TB = 32 * dtr::BITS, SUB = 2, NW = 8 * SUB, UNR = 8;
    const int nblk = a.N >> 7, M = a.M;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, qq = warp >> 3;
    const int nb = bid % nblk, p = bid / nblk;
    const int K16 = a.K >> 4, sc = K16 / a.ks, sw = sc / SUB, s0 = p * sc + qq * sw, ng = sw >> 3;
    const uint8_t* gb = a.w + nb * a.sB + j * TB + static_cast<long long>(s0 >> 3) * a.sG;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);
    u32 na[UNR];
    #pragma unroll
    for (int u = 0; u < UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
    const u32* xb = a.xa + static_cast<size_t>(s0) * 64 + lane * 2;
    const int xts = K16 * 64;
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    for (int g = 0; g < ng; ++g)
    {
        u32 wa[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u) wa[u] = na[u];
        if (g + 1 < ng)
        {
            gb += a.sG;
            #pragma unroll
            for (int u = 0; u < UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
        }
        float gacc[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) gacc[t][0] = gacc[t][1] = gacc[t][2] = gacc[t][3] = 0.f;
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
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
        xb += UNR * 64;
        if (p == 0 && qq == 0)
        {
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                acc[t][0] += gacc[t][0]; acc[t][1] += gacc[t][1];
                acc[t][2] += gacc[t][2]; acc[t][3] += gacc[t][3];
            }
        }
        else if (p == 0)
        {
            float* dst = smf + static_cast<size_t>(1 + g) * M * SMP;
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const int m0 = t * 8 + 2 * c;
                if (m0 < M) { dst[m0 * SMP + col] = gacc[t][0]; dst[m0 * SMP + col + 8] = gacc[t][2]; }
                if (m0 + 1 < M) { dst[(m0 + 1) * SMP + col] = gacc[t][1]; dst[(m0 + 1) * SMP + col + 8] = gacc[t][3]; }
            }
        }
        else
        {
            float* dst = a.part + (static_cast<size_t>(nb) * 9 + 1 + qq * ng + g) * M * 128;
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const int m0 = t * 8 + 2 * c;
                if (m0 < M) { dst[m0 * 128 + col] = gacc[t][0]; dst[m0 * 128 + col + 8] = gacc[t][2]; }
                if (m0 + 1 < M) { dst[(m0 + 1) * 128 + col] = gacc[t][1]; dst[(m0 + 1) * 128 + col + 8] = gacc[t][3]; }
            }
        }
    }

    if (a.ks == 1)
    {
        __syncthreads();
        if (qq == 0)
        {
            for (int r = 0; r < ng; ++r)
            {
                const float* src = smf + static_cast<size_t>(1 + r) * M * SMP;
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const int m0 = t * 8 + 2 * c;
                    if (m0 < M) { acc[t][0] += src[m0 * SMP + col]; acc[t][2] += src[m0 * SMP + col + 8]; }
                    if (m0 + 1 < M) { acc[t][1] += src[(m0 + 1) * SMP + col]; acc[t][3] += src[(m0 + 1) * SMP + col + 8]; }
                }
            }
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const int m0 = t * 8 + 2 * c;
                if (m0 < M) { smf[m0 * SMP + col] = acc[t][0]; smf[m0 * SMP + col + 8] = acc[t][2]; }
                if (m0 + 1 < M) { smf[(m0 + 1) * SMP + col] = acc[t][1]; smf[(m0 + 1) * SMP + col + 8] = acc[t][3]; }
            }
        }
        __syncthreads();
    }
    else
    {
        if (p == 0)
        {
            __syncthreads();
            if (qq == 0)
            {
                for (int r = 0; r < ng; ++r)
                {
                    const float* src = smf + static_cast<size_t>(1 + r) * M * SMP;
                    #pragma unroll
                    for (int t = 0; t < NT; ++t)
                    {
                        const int m0 = t * 8 + 2 * c;
                        if (m0 < M) { acc[t][0] += src[m0 * SMP + col]; acc[t][2] += src[m0 * SMP + col + 8]; }
                        if (m0 + 1 < M) { acc[t][1] += src[(m0 + 1) * SMP + col]; acc[t][3] += src[(m0 + 1) * SMP + col + 8]; }
                    }
                }
                float* dst = a.part + static_cast<size_t>(nb) * 9 * M * 128;
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const int m0 = t * 8 + 2 * c;
                    if (m0 < M) { dst[m0 * 128 + col] = acc[t][0]; dst[m0 * 128 + col + 8] = acc[t][2]; }
                    if (m0 + 1 < M) { dst[(m0 + 1) * 128 + col] = acc[t][1]; dst[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
                }
            }
        }
        __threadfence();
        __syncthreads();
        int* last = reinterpret_cast<int*>(smf + static_cast<size_t>(5) * M * SMP);
        if (threadIdx.x == 0) *last = atomicAdd(a.cnt + nb, 1) == 1;
        __syncthreads();
        if (!*last) return;
        __threadfence();
        const int nq = M * 32;
        for (int i = threadIdx.x; i < nq; i += NW * 32)
        {
            const int m = i >> 5, l4 = i & 31;
            const float* s = a.part + static_cast<size_t>(nb) * 9 * M * 128 + m * 128 + l4 * 4;
            float4 v = __ldcg(reinterpret_cast<const float4*>(s));
            #pragma unroll
            for (int r = 1; r <= 8; ++r)
            {
                const float4 w = __ldcg(reinterpret_cast<const float4*>(s + static_cast<size_t>(r) * M * 128));
                v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
            }
            float* d = smf + static_cast<size_t>(m) * SMP + l4 * 4;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        __syncthreads();
        if (threadIdx.x == 0) a.cnt[nb] = 0;
    }

    const int n0 = nb * 128 + lane * 4;
    uint16_t* yo = static_cast<uint16_t*>(a.y);
    const bool bf = a.odt == dtr::O_BF16;
    float s[4];
    dtr::ld4h(a.svh + n0, s);
    for (int m = warp; m < M; m += NW)
    {
        const float* b = smf + static_cast<size_t>(m) * SMP + lane * 4;
        float v[4] = {0.f + b[0], 0.f + b[1], 0.f + b[2], 0.f + b[3]};
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s[e];
        const uint2 mine = bf ? make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3])) : make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
        const u32 ox = __shfl_xor_sync(0xffffffffu, mine.x, 1), oy = __shfl_xor_sync(0xffffffffu, mine.y, 1);
        if (!(lane & 1))
            *reinterpret_cast<uint4*>(yo + static_cast<size_t>(m) * a.ldy + n0) = make_uint4(mine.x, mine.y, ox, oy);
    }
}

template <int NT>
__global__ void __launch_bounds__(512) k_dtr_gb10_exact(DtrG a)
{
    extern __shared__ __align__(16) float smf[];
    dtr_gb10_exact_body<NT>(a, blockIdx.x, smf);
}

template <int NT>
__global__ void __launch_bounds__(512) k_qb_idxq_exact(FGv g, DtrG d)
{
    extern __shared__ __align__(16) uint8_t sm_raw[];
    if (static_cast<int>(blockIdx.x) < g.ctas)
        gv_fused_body<NT, 2>(g, blockIdx.x, reinterpret_cast<float4 (*)[32]>(sm_raw));
    else
        dtr_gb10_exact_body<NT>(d, blockIdx.x - g.ctas, reinterpret_cast<float*>(sm_raw));
}

// ------------------------------------------------------------------------------------------------ k_dtr_gb10_op (o_proj ks=1 direct slot)

template <int NT, int SUB> struct OpBounds
{
    static constexpr int MINB = (SUB == 2 && NT <= 2) ? 2 : 1;
};

template <int NT, int SUB>
__global__ void __launch_bounds__(256 * SUB, OpBounds<NT, SUB>::MINB) k_dtr_gb10_op(DtrG a)
{
    constexpr int TB = 32 * dtr::BITS, NW = 8 * SUB, UNR = 8;
    extern __shared__ __align__(16) float smf[];
    const int nblk = a.N >> 7, M = a.M;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, qq = warp >> 3;
    const int nb = blockIdx.x;
    const int K16 = a.K >> 4, sw = K16 / SUB, s0 = qq * sw, ng = sw >> 3;
    const uint8_t* gb = a.w + nb * a.sB + j * TB + static_cast<long long>(s0 >> 3) * a.sG;
    Tr3Lane<dtr::BITS, true> ld;
    ld.init(lane);
    u32 na[UNR];
    #pragma unroll
    for (int u = 0; u < UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
    const u32* xb = a.xa + static_cast<size_t>(s0) * 64 + lane * 2;
    const int xts = K16 * 64;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int g = 0; g < ng; ++g)
    {
        u32 wa[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u) wa[u] = na[u];
        if (g + 1 < ng)
        {
            gb += a.sG;
            #pragma unroll
            for (int u = 0; u < UNR; ++u) na[u] = ld.loadw(gb + u * (8 * TB), lane);
        }
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const uint2 b = __ldg(reinterpret_cast<const uint2*>(xb + t * xts + u * 64));
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b.x, b.y);
            }
        }
        xb += UNR * 64;
    }
    const int col = j * 16 + (lane >> 2), c = lane & 3;
    #pragma unroll
    for (int r = 0; r < SUB; ++r)
    {
        if (qq == r)
        {
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const int m0 = t * 8 + 2 * c;
                if (m0 < M)
                {
                    float* d = smf + m0 * SMP + col;
                    if (r == 0) { d[0] = acc[t][0]; d[8] = acc[t][2]; }
                    else { d[0] += acc[t][0]; d[8] += acc[t][2]; }
                }
                if (m0 + 1 < M)
                {
                    float* d = smf + (m0 + 1) * SMP + col;
                    if (r == 0) { d[0] = acc[t][1]; d[8] = acc[t][3]; }
                    else { d[0] += acc[t][1]; d[8] += acc[t][3]; }
                }
            }
        }
        __syncthreads();
    }
    const int n0 = nb * 128 + lane * 4;
    uint16_t* yo = reinterpret_cast<uint16_t*>(dslot::ptr(a.slot));
    float s[4];
    dtr::ld4h(a.svh + n0, s);
    for (int m = warp; m < M; m += NW)
    {
        const float* b = smf + static_cast<size_t>(m) * SMP + lane * 4;
        float v[4] = {0.f + b[0], 0.f + b[1], 0.f + b[2], 0.f + b[3]};
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

// ------------------------------------------------------------------------------------------------ host state & launchers

struct State
{
    bool on = false;
    int nl = 0;
    float4* pair_part = nullptr;      // [PAIR_T][17][g6mx::NT][32] float4
    unsigned* pair_cnt = nullptr;     // [nl][PAIR_T], zero between launches
};

inline cudaError_t init_attrs()
{
    cudaError_t err = cudaSuccess;
    auto set = [&](const void* fn) {
        const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dtr::SMEM_MAX);
        if (err == cudaSuccess) err = e;
    };
    for (int nt = 1; nt <= g6mx::NT; ++nt)
    {
        g6mx::nt_dispatch(nt, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            set(reinterpret_cast<const void*>(k_dtr_gb10_exact<NTv>));
            if constexpr (NTv <= 2) set(reinterpret_cast<const void*>(k_qb_idxq_exact<NTv>));
        });
    }
    return err;
}

inline cudaError_t setup(State& s, int nl)
{
    if (!knob::dec_dense_opt()) return cudaSuccess;
    if (s.on) return cudaSuccess;
    if (const cudaError_t e = init_attrs(); e != cudaSuccess) return e;
    const size_t pp_bytes = static_cast<size_t>(PAIR_T) * 17 * g6mx::NT * 32 * sizeof(float4);
    const size_t pc_bytes = static_cast<size_t>(nl > 0 ? nl : 1) * PAIR_T * sizeof(unsigned);
    if (const cudaError_t e = cudaMalloc(&s.pair_part, pp_bytes); e != cudaSuccess) return e;
    if (const cudaError_t e = cudaMalloc(&s.pair_cnt, pc_bytes); e != cudaSuccess) return e;
    if (const cudaError_t e = cudaMemset(s.pair_cnt, 0, pc_bytes); e != cudaSuccess) return e;
    if (const cudaError_t e = cudaStreamSynchronize(nullptr); e != cudaSuccess) return e;
    s.nl = nl;
    s.on = true;
    return cudaSuccess;
}

inline bool xin_eligible(const State& s, int i, const bind::Layer& b, bool indexer)
{
    if (!s.on || i < 0 || i >= s.nl) return false;
    if (!b.q_a.on || b.q_a.codec != bind::MXFP8 || b.q_a.ks != 16 || (b.q_a.O % 16) != 0 || b.q_a.ld != geom::QA) return false;
    if (!b.kv_a.on || b.kv_a.codec != bind::MXFP8 || b.kv_a.ks != 16) return false;
    if (indexer)
    {
        if (!b.idx_k.on || b.idx_k.codec != bind::BF16W || b.idx_k.ks != 32 || b.idx_k.O != geom::IDX_DIM) return false;
        if (!b.idx_w.on || b.idx_w.codec != bind::BF16W || b.idx_w.ks != 32 || b.idx_w.O != geom::IDX_HEADS) return false;
    }
    return true;
}

inline void launch_xin(const State& s, int i, const bind::Layer& b, bool indexer, int M,
                       const g6::FabricDev& qg, const u64* stepp, unsigned* qtickets, int cqa, cudaStream_t st)
{
    FGv a;
    a.M = M;
    FPart& pq = a.p[0];
    pq.w = b.q_a.w; pq.xfrag = b.q_a.xfrag; pq.y = b.q_a.y; pq.codec = C_MX;
    pq.K = b.q_a.K; pq.ot = b.q_a.O / 16; pq.ks = 16; pq.ld = b.q_a.ld; pq.odt = gv::O_F16; pq.cta0 = 0;
    const int qct = pq.ot;
    a.slot = make_slot(qg, stepp, qtickets, cqa);
    a.slot_part = 0;
    a.slot_ld = b.q_a.O;
    a.slot_last_cta = 1;
    a.slot_ctas = qct;

    int c0 = qct;
    FPart& pkv = a.p[1];
    pkv.w = b.kv_a.w; pkv.xfrag = b.kv_a.xfrag; pkv.y = b.kv_a.y; pkv.codec = C_MX;
    pkv.K = b.kv_a.K; pkv.ot = b.kv_a.O / 16; pkv.ks = 16; pkv.ld = b.kv_a.ld; pkv.odt = gv::O_F16; pkv.cta0 = c0;
    c0 += pkv.ot;
    a.n = 2;

    if (indexer)
    {
        FPart& pk = a.p[2];
        pk.w = b.idx_k.w; pk.xfrag = b.q_a.xfrag; pk.y = b.idx_k.y; pk.codec = C_BF16CVT;
        pk.K = b.idx_k.K; pk.ot = b.idx_k.O / 16; pk.ks = 32; pk.ld = b.idx_k.ld; pk.odt = gv::O_F16; pk.cta0 = c0;
        pk.k_pair = 1;
        pk.pair_part = s.pair_part;
        pk.pair_cnt = s.pair_cnt + static_cast<size_t>(i) * PAIR_T;
        c0 += part_ctas(pk);

        FPart& pw = a.p[3];
        pw.w = b.idx_w.w; pw.xfrag = b.q_a.xfrag; pw.y = b.idx_w.y; pw.codec = C_BF16CVT;
        pw.K = b.idx_w.K; pw.ot = b.idx_w.O / 16; pw.ks = 32; pw.ld = b.idx_w.ld; pw.odt = gv::O_F16; pw.cta0 = c0;
        pw.k_pair = 1;
        pw.pair_part = s.pair_part + static_cast<size_t>(geom::IDX_DIM / 16) * 17 * g6mx::NT * 32;
        pw.pair_cnt = s.pair_cnt + static_cast<size_t>(i) * PAIR_T + geom::IDX_DIM / 16;
        c0 += part_ctas(pw);
        a.n = 4;
    }
    a.ctas = c0;
    g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        k_gv_fused_1p<NTv><<<a.ctas, 512, 0, st>>>(a);
    });
}

inline void launch_qa_norm_rot(const __half* q_a, const __nv_bfloat16* q_a_norm, __half* q_a_n,
                               const bind::Layer& b, const bind::Scratch& sc, const dtr::Mat& m_qb, int M, cudaStream_t st)
{
    k_qa_norm_rot<<<M, 512, 0, st>>>(q_a, q_a_norm, q_a_n,
                                     b.xf_qa.on ? sc.xf_qa : nullptr,
                                     b.q_b_t.on ? m_qb.suh : nullptr,
                                     b.q_b_t.on ? sc.xf_eh : nullptr);
}

inline bool qb_exact_eligible(const bind::Layer& b, const bind::Scratch& sc, int M, int ks)
{
    if (!b.q_b_t.on || b.q_b_t.a.K != geom::QA || b.q_b_t.a.ks != 16 || b.q_b_t.sub != 1) return false;
    if (ks != 1 && ks != 2) return false;
    if (dtrg_exact_smem(ks, M) > static_cast<size_t>(dtr::SMEM_MAX)) return false;
    if (ks == 2)
    {
        const int nblk = b.q_b_t.a.N / 128;
        if (static_cast<size_t>(nblk) * 9 * M * 128 > sc.dt.part_n || nblk > sc.dt.cnt_n) return false;
    }
    return true;
}

inline DtrG make_qb_args(const bind::Layer& b, const bind::Scratch& sc, int M, int ks)
{
    DtrG d;
    d.w = b.q_b_t.a.w[0]; d.sB = b.q_b_t.a.sB; d.sG = b.q_b_t.a.sG;
    d.K = b.q_b_t.a.K; d.N = b.q_b_t.a.N; d.ks = ks; d.M = M;
    d.xa = sc.xf_eh; d.svh = b.q_b_t.a.svh[0];
    d.odt = b.q_b_t.a.odt; d.y = b.q_b_t.a.y; d.ldy = b.q_b_t.a.ldy;
    if (ks == 2) { d.part = sc.dt.part; d.cnt = sc.dt.cnt; }
    return d;
}

inline FGv make_idxq_args(const bind::Layer& b, int M, const g6::FabricDev* qg, const u64* stepp, unsigned* qtickets, int ciq)
{
    FGv g;
    g.n = 1; g.M = M;
    FPart& p = g.p[0];
    p.w = b.idx_q.w; p.xfrag = b.idx_q.xfrag; p.y = b.idx_q.y; p.codec = C_MX;
    p.K = b.idx_q.K; p.ot = b.idx_q.O / 16; p.ks = 32; p.ld = b.idx_q.ld; p.odt = gv::O_F16; p.cta0 = 0;
    g.ctas = p.ot;
    if (qg && ciq >= 0)
    {
        g.slot = make_slot(*qg, stepp, qtickets, ciq);
        g.slot_part = 0;
        g.slot_ld = b.idx_q.O;
        g.slot_last_cta = 1;
        g.slot_ctas = p.ot;
    }
    return g;
}

// Returns true if the caller's idx_q + q_b (including qs_put when ciq >= 0) were launched.
// Caller still handles l2pf_window and qs_get(ciq) when ciq >= 0.
inline bool launch_qb_idxq(const State& s, const bind::Layer& b, const bind::Scratch& sc, const dtr::Mat& m_qb,
                           const __half* q_a_n, int M, bool noiq, bool qa_rot_done,
                           const g6::FabricDev* qg, const u64* stepp, unsigned* qtickets, int ciq,
                           bool& did_qs_put, cudaStream_t st)
{
    did_qs_put = false;
    if (!s.on) return false;
    const bool iq_on = !noiq && b.idx_q.on && b.idx_q.codec == bind::MXFP8 && b.idx_q.ks == 32 && (b.idx_q.O % 16) == 0;
    const bool can_fuse = iq_on && M <= 16 && qb_exact_eligible(b, sc, M, 1);
    const bool can_qb_solo = !can_fuse && qb_exact_eligible(b, sc, M, 2);
    if (!can_fuse && !can_qb_solo && !iq_on) return false;

    if (!qa_rot_done)
    {
        bind::launch(b.xf_qa, st);
        if (can_fuse || can_qb_solo) dtr::launch_rot(q_a_n, geom::QA, m_qb.suh, sc.xf_eh, geom::QA, M, st);
    }

    if (can_fuse)
    {
        const FGv g = make_idxq_args(b, M, qg, stepp, qtickets, ciq);
        const DtrG d = make_qb_args(b, sc, M, 1);
        const size_t smem = std::max(static_cast<size_t>(32 * 32 * sizeof(float4)), dtrg_exact_smem(1, M));
        const int grid = g.ctas + (d.N / 128);
        g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if constexpr (NTv <= 2) k_qb_idxq_exact<NTv><<<grid, 512, smem, st>>>(g, d);
        });
        did_qs_put = (qg && ciq >= 0);
        return true;
    }

    if (iq_on)
    {
        const FGv g = make_idxq_args(b, M, qg, stepp, qtickets, ciq);
        g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            k_gv_fused<NTv><<<g.ctas, 512, 0, st>>>(g);
        });
        did_qs_put = (qg && ciq >= 0);
    }
    else if (!noiq && b.idx_q.on)
    {
        bind::launch(b.idx_q, st);
    }

    if (can_qb_solo)
    {
        const DtrG d = make_qb_args(b, sc, M, 2);
        const size_t smem = dtrg_exact_smem(2, M);
        const int grid = (d.N / 128) * 2;
        g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            k_dtr_gb10_exact<NTv><<<grid, 512, smem, st>>>(d);
        });
    }
    else
    {
        bind::launch_q_b(b, st);
    }
    return true;
}

// Returns true if o_proj was launched with exact 0-scratch k_dtr_gb10_op.
inline bool launch_o_proj(const State& s, const bind::Layer& b, const bind::Scratch& sc, int M, cudaStream_t st)
{
    if (!s.on || !b.o_proj_t.on || !b.o_proj_t.rot || !b.o_proj_t.a.slot.send) return false;
    const int stock_parts = b.o_proj_t.a.ks * b.o_proj_t.sub;
    if (stock_parts != 2 && stock_parts != 4) return false;
    const int K16 = b.o_proj_t.a.K / 16;
    if (K16 % (8 * stock_parts) != 0) return false;
    const bool fast_regime = (stock_parts == 4) ? (M >= 16) : (M >= 8);
    if (!fast_regime) return false;

    dtr::launch_rot(b.o_proj_t.a.x, b.o_proj_t.a.ldx, b.o_proj_t.a.suh[0], sc.xf_eh, b.o_proj_t.a.K, M, st);
    DtrG d;
    d.w = b.o_proj_t.a.w[0]; d.sB = b.o_proj_t.a.sB; d.sG = b.o_proj_t.a.sG;
    d.K = b.o_proj_t.a.K; d.N = b.o_proj_t.a.N; d.ks = 1; d.M = M;
    d.xa = sc.xf_eh; d.svh = b.o_proj_t.a.svh[0];
    d.odt = dtr::O_BF16; d.ldy = b.o_proj_t.a.ldy;
    d.slot = b.o_proj_t.a.slot;
    const int grid = d.N / 128;
    const size_t smem = dtrg_smem(M);
    g6mx::nt_dispatch(g6mx::tiles(M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        if (stock_parts == 4) k_dtr_gb10_op<NTv, 4><<<grid, 1024, smem, st>>>(d);
        else k_dtr_gb10_op<NTv, 2><<<grid, 512, smem, st>>>(d);
    });
    return true;
}

#else

struct State { bool on = false; };
inline cudaError_t setup(State&, int) { return cudaSuccess; }
inline bool xin_eligible(const State&, int, const bind::Layer&, bool) { return false; }
inline void launch_xin(const State&, int, const bind::Layer&, bool, int, const g6::FabricDev&, const u64*, unsigned*, int, cudaStream_t) {}
inline void launch_qa_norm_rot(const __half*, const __nv_bfloat16*, __half*, const bind::Layer&, const bind::Scratch&, const dtr::Mat&, int, cudaStream_t) {}
inline bool launch_qb_idxq(const State&, const bind::Layer&, const bind::Scratch&, const dtr::Mat&, const __half*, int, bool, bool,
                           const g6::FabricDev*, const u64*, unsigned*, int, bool& did_qs_put, cudaStream_t) { did_qs_put = false; return false; }
inline bool launch_o_proj(const State&, const bind::Layer&, const bind::Scratch&, int, cudaStream_t) { return false; }

#endif

}   // namespace dec_dense_opt
