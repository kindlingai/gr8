// Inkling-2-Mini / Inkling-2 exlr8-v2 trellis routed experts (pmcgx / mul1 / mcg, E4M3-rounded, K=2/3/4) on sm_121a
// tensor cores, with Inkling-2 softcapped SwiGLU (4*tanh(g/4) * sigmoid(g) * 25*tanh(u/25)).
//
// Self-contained: parameterized on runtime (H, eb, NE, TK) with zero dependency on GLM compile-time constants
// (common.cuh / k1args.cuh). Uses ink2::Plan (ink2_router.cuh) directly and writes the same ypart [tile][8][H]
// layout as ink2_moe.cuh so k_reduce (TP=1) and k_ink2rt_picks + sum (TP>1) work identically.
//
// Expects standard v2plan::v2_layer tensors (no shared expert -> bs=false):
//   tr3.fc1    [fc1_bytes] u8    [nblk][2][H/16][8][32*K]
//   tr3.fc2    [fc2_bytes] u8    [nblk][H/16][8][32*K]  (already Item128-transposed to gr8 order)
//   tr3.v1     [nblk][2][H] f16  signed suh for gate (0) and up (1)
//   tr3.v2     [nblk][3][128] f16  svh gate (0), svh up (1), suh down (2)
//   tr3.v3     [nblk][H] f16     signed svh for down
//   tr3.table  [NE][6] i64       [fc1_off, fc2_off, row0, nblk, k_down, first_half]
//   tr3.kmix   [NE] i64          0 when uniform per expert, else 4 bits per fc1 unit (2*b + mat)
#pragma once
#include "ink2_moe.cuh"
#include "../runtime/ink2_prof.cuh"
#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace ink2
{
namespace tr3
{
enum { CB_MCG = 0, CB_MUL1 = 1, CB_PMCGX = 2 };

#ifdef __CUDACC__
static constexpr float SQ128 = 0.08838834764831844f;   // 1 / sqrt(128)

__device__ __forceinline__ __half2 as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ float rnd_h(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ void mma16816_f16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    gr8_sm70::mma16816_f16(c, a0, a1, a2, a3, b0, b1);
}

__device__ __forceinline__ void had128(float (&v)[4], int lane)
{
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? (p - v[e]) : (v[e] + p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= SQ128;
}

__device__ __forceinline__ u32 round_e4m3(u32 r) { return gr8_sm70::round_f16x2_e4m3_bounded(r); }

__device__ __forceinline__ u32 cb_mcg2(u32 w0, u32 w1)
{
    u32 x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const u32 lo = __byte_perm(x0, x1, 0x5410);
    const u32 hi = __byte_perm(x0, x1, 0x7632);
    return round_e4m3(as_u32(__hadd2(as_h2(lo), as_h2(hi))));
}

__device__ __forceinline__ u32 cb_mul1x2(u32 w0, u32 w1)
{
    const u32 s0 = __dp4a(w0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const u32 s1 = __dp4a(w1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 v = __hfma2(as_h2(__byte_perm(s0, s1, 0x5410)), as_h2(0x1eee1eeeu), as_h2(0xc931c931u));
    return round_e4m3(as_u32(v));
}

__device__ __forceinline__ u32 cb_pmcgx(u32 w)
{
    u32 x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    return round_e4m3(as_u32(__hadd2(as_h2(x0), as_h2(__byte_perm(x1, 0, 0x1032)))));
}

template <int BITS, int CB>
struct LaneDec
{
    int i0, i1, sh;
    __device__ __forceinline__ void init(int lane)
    {
        const int b0 = (lane * 8 + 1) * BITS - 16 + 256 * BITS;
        i0 = (b0 / 32) % (8 * BITS);
        i1 = (i0 + 1) % (8 * BITS);
        sh = 48 - (b0 % 32);
    }
    __device__ __forceinline__ u32 loadw(const uint8_t* tile, int lane) const
    {
        if constexpr (BITS == 4) return nv4::rd32(tile + lane * 4);
        else return nv4::rd32(tile + (lane % (8 * BITS)) * 4);
    }
    __device__ __forceinline__ void decodew(u32 raw, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        u64 m;
        int s;
        if constexpr (BITS == 4)
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0);
            m = (static_cast<u64>(a) << 32) | raw;
            s = 28;
        }
        else
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0);
            const u32 b = __shfl_sync(0xffffffffu, raw, i1);
            m = (static_cast<u64>(a) << 32) | b;
            s = sh;
        }
        if constexpr (CB == CB_PMCGX)
        {
            u32 w[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) w[p] = static_cast<u32>(m >> (s - BITS * (2 * p + 1))) & 0xfffffu;
            f0[0] = cb_pmcgx(w[0]);
            f0[1] = cb_pmcgx(w[1]);
            f1[0] = cb_pmcgx(w[2]);
            f1[1] = cb_pmcgx(w[3]);
        }
        else
        {
            u32 w[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (s - BITS * j)) & 0xffffu;
            if constexpr (CB == CB_MUL1)
            {
                f0[0] = cb_mul1x2(w[0], w[1]);
                f0[1] = cb_mul1x2(w[2], w[3]);
                f1[0] = cb_mul1x2(w[4], w[5]);
                f1[1] = cb_mul1x2(w[6], w[7]);
            }
            else
            {
                f0[0] = cb_mcg2(w[0], w[1]);
                f0[1] = cb_mcg2(w[2], w[3]);
                f1[0] = cb_mcg2(w[4], w[5]);
                f1[1] = cb_mcg2(w[6], w[7]);
            }
        }
    }
};

__device__ __forceinline__ int fc1_unit(const uint8_t* fc1, const long long* kmix, int e, const long long* tr, int u, int H, const uint8_t*& mw)
{
    const size_t mb = static_cast<size_t>(H) * 16;   // (H/16) * 8 * 32 bytes per bit of width
    const long long km = kmix ? kmix[e] : 0;
    if (!km)
    {
        const int k = static_cast<int>(tr[4]);
        mw = fc1 + static_cast<size_t>(tr[0]) + static_cast<size_t>(u) * static_cast<size_t>(k) * mb;
        return k;
    }
    if (km < 0)
    {
        const int gk = static_cast<int>(km & 15);
        const int uk = static_cast<int>((km >> 4) & 15);
        const int b = u >> 1;
        const int mat = u & 1;
        mw = fc1 + static_cast<size_t>(tr[0]) +
             (static_cast<size_t>(b) * static_cast<size_t>(gk + uk) + (mat ? static_cast<size_t>(gk) : 0u)) * mb;
        return mat ? uk : gk;
    }
    size_t off = static_cast<size_t>(tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * mb;
    mw = fc1 + off;
    return static_cast<int>((km >> (4 * u)) & 15);
}

__device__ __forceinline__ const uint8_t* fc2_block(const uint8_t* fc2, const long long* tr, int b, int bits, int H)
{
    const size_t mb = static_cast<size_t>(H) * 16;
    return fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * static_cast<size_t>(bits) * mb;
}

template <int BITS, int CB>
__device__ __forceinline__ void fc1_hb_mma(const uint8_t* mw, int hb, int warp, int lane,
                                           const __half (&x_rot)[8][136], int mask, float (&c)[4])
{
    const size_t tb = static_cast<size_t>(32 * BITS);
    LaneDec<BITS, CB> tl;
    tl.init(lane);
    const int ks0 = hb * 8;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int r = (lane & 3) | ((lane & 16) >> 2), qp = (lane >> 2) & 3;
    const bool act = ((mask >> r) & 1) != 0;
    gr8_sm70::Mma16816Acc acc;
    acc.zero();
    #pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(ks0 + p) * 8 + warp) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        uint32_t qa0_0, qa0_1, qa1_0, qa1_1;
        gr8_sm70::Mma16816Acc::prep_a(f0[0], f1[0], f0[1], f1[1], qa0_0, qa0_1, qa1_0, qa1_1);
        const uint2 bv = act ? *reinterpret_cast<const uint2*>(x_rot[r] + p * 16 + 4 * qp) : make_uint2(0u, 0u);
        acc.mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
    }
    acc.flush_add(c);
#else
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = ((mask >> row0) & 1) != 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(ks0 + p) * 8 + warp) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
        mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
#endif
}

template <int BITS, int CB>
__device__ __forceinline__ void fc2_ntile_mma(const uint8_t* mw, int n_tile, int lane,
                                              const __half (&x2)[8][136], int mask, float (&c)[4])
{
    const size_t tb = static_cast<size_t>(32 * BITS);
    LaneDec<BITS, CB> tl;
    tl.init(lane);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int r = (lane & 3) | ((lane & 16) >> 2), qp = (lane >> 2) & 3;
    const bool act = ((mask >> r) & 1) != 0;
    gr8_sm70::Mma16816Acc acc;
    acc.zero();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        uint32_t qa0_0, qa0_1, qa1_0, qa1_1;
        gr8_sm70::Mma16816Acc::prep_a(f0[0], f1[0], f0[1], f1[1], qa0_0, qa0_1, qa1_0, qa1_1);
        const uint2 bv = act ? *reinterpret_cast<const uint2*>(x2[r] + ks * 16 + 4 * qp) : make_uint2(0u, 0u);
        acc.mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
    }
    acc.flush_add(c);
#else
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = ((mask >> row0) & 1) != 0;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col + 8) : 0u;
        mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
#endif
}

struct Tr3Args
{
    const uint8_t* fc1;
    const uint8_t* fc2;
    const __half* v1;         // [nblk][2][H]
    const __half* v2;         // [nblk][3][128]
    const __half* v3;         // [nblk][H]
    const long long* table;   // [NE][6]
    const long long* kmix;    // [NE] or nullptr
    const uint16_t* x;        // [M][H] BF16
    Plan plan;
    __half* a_rot;            // [max_tiles * 8][eb][128] f16
    void* ypart;              // [max_tiles * 8][H] fp32 (or BF16 if y_bf16)
    uint16_t* out;            // [M][H] BF16
    int H, eb, M, TK, y_bf16;
    // Optional Wave 3 factorized + wide-load fields (null/0 = fallback to stock)
    const __half* sg;         // [2][H] (in, out signs)
    const __half* v1b;        // [nblk][2][HB]
    const __half* v3b;        // [nblk][HB]
    __half* x_had;            // [M][H] f16 pre-rotated H128(x * sg_in)
    int wide;                 // 1 if fc1/fc2 widened in-place
};

// ---------------------------------------------------------------------------------------------- Wave 3: wide-load helpers & boot setup
__device__ __forceinline__ u64 ld_nc64_tr3(const void* p)
{
    u64 v;
    asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ld_nc128_tr3(const void* p) { return gr8_sm70::ld_nc_uint4(p); }

template <int BITS>
__device__ __forceinline__ u32 trw_chunk_rt(const u64 (&w)[3], int p)
{
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if constexpr (CB > 16)
        if (sh > 64 - CB) v |= w[n + 1] << (64 - sh);
    return static_cast<u32>(v) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__device__ __forceinline__ u32 trw_tile_chunk_rt(const uint8_t* tile, int lane)
{
    constexpr int CB = 8 * BITS, NW = 8 * BITS;
    const int bit = lane * CB, wi = bit >> 5, off = bit & 31;
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi + 1 < NW ? wi + 1 : NW - 1];
    return static_cast<u32>(m >> (64 - off - CB)) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__device__ __forceinline__ void trw_pack_rt(u64 (&v)[3], int p, u32 chunk)
{
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if constexpr (CB > 16)
        if (sh > 64 - CB) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
}

template <int BITS>
__device__ __forceinline__ void decode_pmcgx_m(u64 m, int s, u32 (&f0)[2], u32 (&f1)[2])
{
    u32 w[4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) w[p] = static_cast<u32>(m >> (s - BITS * (2 * p + 1))) & 0xfffffu;
    f0[0] = cb_pmcgx(w[0]);
    f0[1] = cb_pmcgx(w[1]);
    f1[0] = cb_pmcgx(w[2]);
    f1[1] = cb_pmcgx(w[3]);
}

// In-place widen FC1 for runtime H (HB = H / 128). Run EXACTLY ONCE per physical layer!
__global__ void __launch_bounds__(256) k_ink2_tr3_widen_fc1(uint8_t* fc1, const long long* tab, const long long* kmix, int eb, int H)
{
    const int HB = H / 128;
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int hb = blockIdx.x % HB;
    const int bm = (blockIdx.x / HB) % (2 * eb);
    const int e = blockIdx.x / (HB * 2 * eb);
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (bm >= 2 * static_cast<int>(tr[3])) return;
    const uint8_t* mw_c = nullptr;
    const int k = fc1_unit(fc1, kmix, e, tr, bm, H, mw_c);
    uint8_t* grp = const_cast<uint8_t*>(mw_c) + static_cast<size_t>(hb) * (2048 * k);
    if (k == 3)
    {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack_rt<3>(v, p, trw_tile_chunk_rt<3>(grp + (p * 8 + j) * 96, lane));
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
    }
    else if (k == 2)
    {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack_rt<2>(v, p, trw_tile_chunk_rt<2>(grp + (p * 8 + j) * 64, lane));
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 2; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
    }
    else if (k == 4)
    {
        u32 w[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) w[p] = *reinterpret_cast<const u32*>(grp + (p * 8 + j) * 128 + lane * 4);
        __syncthreads();
        *reinterpret_cast<uint4*>(grp + j * 1024 + lane * 16) = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(grp + j * 1024 + 512 + lane * 16) = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

// In-place widen FC2 for runtime H (HB = H / 128). Run EXACTLY ONCE per physical layer!
__global__ void __launch_bounds__(256) k_ink2_tr3_widen_fc2(uint8_t* fc2, const long long* tab, int eb, int H)
{
    const int HB = H / 128;
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int hb = blockIdx.x % HB;
    const int b = (blockIdx.x / HB) % eb;
    const int e = blockIdx.x / (HB * eb);
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int k = static_cast<int>(tr[4]);
    const int n_tile = hb * 8 + j;
    uint8_t* nt = const_cast<uint8_t*>(fc2_block(fc2, tr, b, k, H)) + static_cast<size_t>(n_tile) * (256 * k);
    if (k == 3)
    {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack_rt<3>(v, p, trw_tile_chunk_rt<3>(nt + p * 96, lane));
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(nt + n * 256 + lane * 8) = v[n];
    }
    else if (k == 2)
    {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack_rt<2>(v, p, trw_tile_chunk_rt<2>(nt + p * 64, lane));
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < 2; ++n) *reinterpret_cast<u64*>(nt + n * 256 + lane * 8) = v[n];
    }
    else if (k == 4)
    {
        u32 w[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) w[p] = *reinterpret_cast<const u32*>(nt + p * 128 + lane * 4);
        __syncwarp();
        *reinterpret_cast<uint4*>(nt + lane * 16) = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(nt + 512 + lane * 16) = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

// Extract sg[2][H] from v1/v3 (first non-zero row per channel). Grid: ceil(H / 256), Block: 256.
__global__ void k_ink2_tr3_extract_sg(const __half* v1, const __half* v3, __half* sg, int nblk, int H)
{
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= H) return;
    uint16_t s_in = 0x3c00u;   // +1.0h
    for (int r = 0; r < nblk * 2; ++r)
    {
        const uint16_t u = __half_as_ushort(v1[static_cast<size_t>(r) * H + c]);
        if ((u & 0x7fffu) != 0)
        {
            s_in = (u & 0x8000u) ? 0xbc00u : 0x3c00u;
            break;
        }
    }
    uint16_t s_out = 0x3c00u;
    for (int r = 0; r < nblk; ++r)
    {
        const uint16_t u = __half_as_ushort(v3[static_cast<size_t>(r) * H + c]);
        if ((u & 0x7fffu) != 0)
        {
            s_out = (u & 0x8000u) ? 0xbc00u : 0x3c00u;
            break;
        }
    }
    sg[c] = __ushort_as_half(s_in);
    sg[H + c] = __ushort_as_half(s_out);
}

// Extract v1b[nblk][2][HB], v3b[nblk][HB] and bit-exact verify every cell of v1 and v3!
__global__ void k_ink2_tr3_extract_verify_bs(const __half* v1, const __half* v3, const __half* sg,
                                             __half* v1b, __half* v3b, int* d_ok, int nblk, int H)
{
    const int HB = H / 128;
    const int idx = blockIdx.x;
    const int hb = idx % HB;
    const int brow = idx / HB;
    if (brow >= nblk) return;
    const int c = hb * 128 + threadIdx.x;
    const uint16_t sin_neg = __half_as_ushort(sg[c]) & 0x8000u;
    const uint16_t sout_neg = __half_as_ushort(sg[H + c]) & 0x8000u;
    const uint16_t sin0_neg = __half_as_ushort(sg[hb * 128]) & 0x8000u;
    const uint16_t sout0_neg = __half_as_ushort(sg[H + hb * 128]) & 0x8000u;

    for (int mat = 0; mat < 2; ++mat)
    {
        const size_t row = static_cast<size_t>(brow) * 2 + mat;
        const uint16_t b0 = __half_as_ushort(v1[row * H + hb * 128]) ^ sin0_neg;
        if (threadIdx.x == 0) v1b[row * HB + hb] = __ushort_as_half(b0);
        const uint16_t got = __half_as_ushort(v1[row * H + c]);
        const uint16_t exp = b0 ^ sin_neg;
        if (got != exp && !(((got & 0x7fffu) == 0) && ((exp & 0x7fffu) == 0))) atomicAnd(d_ok, 0);
    }
    const uint16_t b3 = __half_as_ushort(v3[static_cast<size_t>(brow) * H + hb * 128]) ^ sout0_neg;
    if (threadIdx.x == 0) v3b[static_cast<size_t>(brow) * HB + hb] = __ushort_as_half(b3);
    const uint16_t got3 = __half_as_ushort(v3[static_cast<size_t>(brow) * H + c]);
    const uint16_t exp3 = b3 ^ sout_neg;
    if (got3 != exp3 && !(((got3 & 0x7fffu) == 0) && ((exp3 & 0x7fffu) == 0))) atomicAnd(d_ok, 0);
}

// Pre-rotate H128(x * sg_in) ONCE per token. Grid: M * HB, Block: 32.
__global__ void __launch_bounds__(32) k_ink2_tr3_prerot_x(const uint16_t* __restrict__ x, const __half* __restrict__ sg_in,
                                                          __half* __restrict__ x_had, int M, int H)
{
    const int HB = H / 128;
    const int hb = blockIdx.x % HB;
    const int m = blockIdx.x / HB;
    if (m >= M) return;
    const int lane = threadIdx.x;
    const size_t off = static_cast<size_t>(m) * H + hb * 128 + lane * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x + off);
    const uint2 sv = *reinterpret_cast<const uint2*>(sg_in + hb * 128 + lane * 4);
    const float2 s0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
    const float2 s1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
    float v[4] = {
        bf(static_cast<uint16_t>(xv.x)) * s0.x,
        bf(static_cast<uint16_t>(xv.x >> 16)) * s0.y,
        bf(static_cast<uint16_t>(xv.y)) * s1.x,
        bf(static_cast<uint16_t>(xv.y >> 16)) * s1.y,
    };
    had128(v, lane);
    uint2 ov;
    *reinterpret_cast<__half2*>(&ov.x) = __floats2half2_rn(v[0], v[1]);
    *reinterpret_cast<__half2*>(&ov.y) = __floats2half2_rn(v[2], v[3]);
    *reinterpret_cast<uint2*>(x_had + off) = ov;
}

template <int BITS, int NT>
__device__ __forceinline__ void pmcgx_fc1_hb_wide(const uint8_t* mw, int hb, int warp, int lane,
                                                  const __half (&x_rot)[NT * 8][136], const int (&masks)[NT], int nt_here,
                                                  float (&c)[NT][4])
{
    const int prev_lane = (lane + 31) & 31;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int r = (lane & 3) | ((lane & 16) >> 2), qp = (lane >> 2) & 3;
    gr8_sm70::Mma16816Acc acc[NT <= 2 ? NT : 1];
    if constexpr (NT <= 2)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[t].zero();
    }
    auto step = [&](const u32 (&f0)[2], const u32 (&f1)[2], int kp) {
        uint32_t qa0_0, qa0_1, qa1_0, qa1_1;
        gr8_sm70::Mma16816Acc::prep_a(f0[0], f1[0], f0[1], f1[1], qa0_0, qa0_1, qa1_0, qa1_1);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const bool act = ((masks[t] >> r) & 1) != 0;
                const uint2 bv = act ? *reinterpret_cast<const uint2*>(x_rot[t * 8 + r] + kp * 16 + 4 * qp) : make_uint2(0u, 0u);
                if constexpr (NT <= 2) acc[t].mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
                else gr8_sm70::Mma16816Acc::mma_prepared_add(c[t], qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
            }
        }
    };
#else
    const int row0 = lane >> 2, col = lane & 3;
    auto step = [&](const u32 (&f0)[2], const u32 (&f1)[2], int kp) {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const bool act = ((masks[t] >> row0) & 1) != 0;
                const __half* xr = x_rot[t * 8 + row0] + kp * 16 + 2 * col;
                const u32 b0 = act ? *reinterpret_cast<const u32*>(xr) : 0u;
                const u32 b1 = act ? *reinterpret_cast<const u32*>(xr + 8) : 0u;
                mma16816_f16(c[t], f0[0], f1[0], f0[1], f1[1], b0, b1);
            }
        }
    };
#endif
    if constexpr (BITS == 4)
    {
        const uint8_t* wg = mw + static_cast<size_t>(hb) * 8192 + warp * 1024 + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const uint4 wh = ld_nc128_tr3(wg + h * 512);
            const u32 w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                const u32 ch = w[p];
                const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                u32 f0[2], f1[2];
                decode_pmcgx_m<4>((static_cast<u64>(pc) << 32) | ch, 28, f0, f1);
                step(f0, f1, h * 4 + p);
            }
        }
    }
    else
    {
        const uint8_t* wg = mw + static_cast<size_t>(hb) * (2048 * BITS) + warp * 256 + lane * 8;
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = ld_nc64_tr3(wg + n * 2048);
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 ch = trw_chunk_rt<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            decode_pmcgx_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
            step(f0, f1, p);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    if constexpr (NT <= 2)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t) if (t < nt_here) acc[t].flush_add(c[t]);
    }
#endif
}

template <int BITS, int NT>
__device__ __forceinline__ void pmcgx_fc2_ntile_wide(const uint8_t* mw, int n_tile, int lane,
                                                     const __half* x2_b_base, const int (&masks)[NT], int nt_here,
                                                     float (&c)[NT][4])
{
    const int prev_lane = (lane + 31) & 31;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const int r = (lane & 3) | ((lane & 16) >> 2), qp = (lane >> 2) & 3;
    gr8_sm70::Mma16816Acc acc[NT <= 2 ? NT : 1];
    if constexpr (NT <= 2)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[t].zero();
    }
    auto step = [&](const u32 (&f0)[2], const u32 (&f1)[2], int ks) {
        uint32_t qa0_0, qa0_1, qa1_0, qa1_1;
        gr8_sm70::Mma16816Acc::prep_a(f0[0], f1[0], f0[1], f1[1], qa0_0, qa0_1, qa1_0, qa1_1);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const bool act = ((masks[t] >> r) & 1) != 0;
                const uint2 bv = act ? *reinterpret_cast<const uint2*>(x2_b_base + (t * 8 + r) * 136 + ks * 16 + 4 * qp) : make_uint2(0u, 0u);
                if constexpr (NT <= 2) acc[t].mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
                else gr8_sm70::Mma16816Acc::mma_prepared_add(c[t], qa0_0, qa0_1, qa1_0, qa1_1, bv.x, bv.y);
            }
        }
    };
#else
    const int row0 = lane >> 2, col = lane & 3;
    auto step = [&](const u32 (&f0)[2], const u32 (&f1)[2], int ks) {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const bool act = ((masks[t] >> row0) & 1) != 0;
                const __half* xr = x2_b_base + (t * 8 + row0) * 136 + ks * 16 + 2 * col;
                const u32 b0 = act ? *reinterpret_cast<const u32*>(xr) : 0u;
                const u32 b1 = act ? *reinterpret_cast<const u32*>(xr + 8) : 0u;
                mma16816_f16(c[t], f0[0], f1[0], f0[1], f1[1], b0, b1);
            }
        }
    };
#endif
    if constexpr (BITS == 4)
    {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * 1024 + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const uint4 wh = ld_nc128_tr3(wg + h * 512);
            const u32 w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                const u32 ch = w[p];
                const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                u32 f0[2], f1[2];
                decode_pmcgx_m<4>((static_cast<u64>(pc) << 32) | ch, 28, f0, f1);
                step(f0, f1, h * 4 + p);
            }
        }
    }
    else
    {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * (256 * BITS) + lane * 8;
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = ld_nc64_tr3(wg + n * 256);
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 ch = trw_chunk_rt<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            decode_pmcgx_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
            step(f0, f1, p);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    if constexpr (NT <= 2)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t) if (t < nt_here) acc[t].flush_add(c[t]);
    }
#endif
}

// Wide-load multi-tile PMCGX FC1 using pre-rotated x_had and block scales v1b.
// Grid: (slots, eb), Block: 256 (8 warps). Processes NT * 8 tokens per trellis weight decode!
template <int NT>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc1_wide(Tr3Args a)
{
    const int slot = blockIdx.x, b = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int HB = a.H / 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;

    const uint8_t *mw0 = nullptr, *mw1 = nullptr;
    const int bits0 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 0, a.H, mw0);
    const int bits1 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 1, a.H, mw1);

    __shared__ __half x_rot[NT * 8][136];

    for (int tg = 0; tg < nt; tg += NT)
    {
        const int nt_here = min(NT, nt - tg);
        int masks[NT];
        int w_tok[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const int tile = tile0 + tg + t;
                masks[t] = a.plan.tile_mask[tile];
                w_tok[t] = ((masks[t] >> warp) & 1) ? a.plan.tile_tok[tile * 8 + warp] : 0;
            }
            else
            {
                masks[t] = 0;
                w_tok[t] = 0;
            }
        }

        // Pass 0: GATE -> transpose through x_rot -> had128 * sg -> stage in a.a_rot (halves static smem!)
        {
            float c[NT][4];
            #pragma unroll
            for (int t = 0; t < NT; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;
            for (int hb = 0; hb < HB; ++hb)
            {
                const __half2 bsh = __half2half2(a.v1b[(brow * 2 + 0) * HB + hb]);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    if (t < nt_here && ((masks[t] >> warp) & 1))
                    {
                        const uint2 xv = *reinterpret_cast<const uint2*>(a.x_had + static_cast<size_t>(w_tok[t]) * a.H + hb * 128 + lane * 4);
                        uint2 ov;
                        *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
                        *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
                        *reinterpret_cast<uint2*>(x_rot[t * 8 + warp] + lane * 4) = ov;
                    }
                }
                __syncthreads();
                if (bits0 == 3) pmcgx_fc1_hb_wide<3, NT>(mw0, hb, warp, lane, x_rot, masks, nt_here, c);
                else if (bits0 == 4) pmcgx_fc1_hb_wide<4, NT>(mw0, hb, warp, lane, x_rot, masks, nt_here, c);
                else pmcgx_fc1_hb_wide<2, NT>(mw0, hb, warp, lane, x_rot, masks, nt_here, c);
                __syncthreads();
            }
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here)
                {
                    const int t0 = 2 * col, t1 = 2 * col + 1;
                    if ((masks[t] >> t0) & 1)
                    {
                        x_rot[t * 8 + t0][warp * 16 + row0] = __float2half_rn(c[t][0]);
                        x_rot[t * 8 + t0][warp * 16 + row0 + 8] = __float2half_rn(c[t][2]);
                    }
                    if ((masks[t] >> t1) & 1)
                    {
                        x_rot[t * 8 + t1][warp * 16 + row0] = __float2half_rn(c[t][1]);
                        x_rot[t * 8 + t1][warp * 16 + row0 + 8] = __float2half_rn(c[t][3]);
                    }
                }
            }
            __syncthreads();
            const uint2 sgv = *reinterpret_cast<const uint2*>(a.v2 + (brow * 3 + 0) * 128 + lane * 4);
            const float2 sg0 = __half22float2(*reinterpret_cast<const __half2*>(&sgv.x));
            const float2 sg1 = __half22float2(*reinterpret_cast<const __half2*>(&sgv.y));
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here && ((masks[t] >> warp) & 1))
                {
                    const uint2 yv = *reinterpret_cast<const uint2*>(x_rot[t * 8 + warp] + lane * 4);
                    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&yv.x));
                    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&yv.y));
                    float vg[4] = {f0.x, f0.y, f1.x, f1.y};
                    had128(vg, lane);
                    uint2 ov;
                    *reinterpret_cast<__half2*>(&ov.x) = __floats2half2_rn(vg[0] * sg0.x, vg[1] * sg0.y);
                    *reinterpret_cast<__half2*>(&ov.y) = __floats2half2_rn(vg[2] * sg1.x, vg[3] * sg1.y);
                    const int tile = tile0 + tg + t;
                    __half* dst = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                    *reinterpret_cast<uint2*>(dst) = ov;
                }
            }
            __syncthreads();
        }

        // Pass 1: UP + softcapped SwiGLU + suh_d -> had128 -> a_rot
        {
            float c[NT][4];
            #pragma unroll
            for (int t = 0; t < NT; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;
            for (int hb = 0; hb < HB; ++hb)
            {
                const __half2 bsh = __half2half2(a.v1b[(brow * 2 + 1) * HB + hb]);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    if (t < nt_here && ((masks[t] >> warp) & 1))
                    {
                        const uint2 xv = *reinterpret_cast<const uint2*>(a.x_had + static_cast<size_t>(w_tok[t]) * a.H + hb * 128 + lane * 4);
                        uint2 ov;
                        *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
                        *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
                        *reinterpret_cast<uint2*>(x_rot[t * 8 + warp] + lane * 4) = ov;
                    }
                }
                __syncthreads();
                if (bits1 == 3) pmcgx_fc1_hb_wide<3, NT>(mw1, hb, warp, lane, x_rot, masks, nt_here, c);
                else if (bits1 == 4) pmcgx_fc1_hb_wide<4, NT>(mw1, hb, warp, lane, x_rot, masks, nt_here, c);
                else pmcgx_fc1_hb_wide<2, NT>(mw1, hb, warp, lane, x_rot, masks, nt_here, c);
                __syncthreads();
            }
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here)
                {
                    const int t0 = 2 * col, t1 = 2 * col + 1;
                    if ((masks[t] >> t0) & 1)
                    {
                        x_rot[t * 8 + t0][warp * 16 + row0] = __float2half_rn(c[t][0]);
                        x_rot[t * 8 + t0][warp * 16 + row0 + 8] = __float2half_rn(c[t][2]);
                    }
                    if ((masks[t] >> t1) & 1)
                    {
                        x_rot[t * 8 + t1][warp * 16 + row0] = __float2half_rn(c[t][1]);
                        x_rot[t * 8 + t1][warp * 16 + row0 + 8] = __float2half_rn(c[t][3]);
                    }
                }
            }
            __syncthreads();
            const uint2 suv = *reinterpret_cast<const uint2*>(a.v2 + (brow * 3 + 1) * 128 + lane * 4);
            const uint2 sdv = *reinterpret_cast<const uint2*>(a.v2 + (brow * 3 + 2) * 128 + lane * 4);
            const float2 su0 = __half22float2(*reinterpret_cast<const __half2*>(&suv.x));
            const float2 su1 = __half22float2(*reinterpret_cast<const __half2*>(&suv.y));
            const float2 sd0 = __half22float2(*reinterpret_cast<const __half2*>(&sdv.x));
            const float2 sd1 = __half22float2(*reinterpret_cast<const __half2*>(&sdv.y));
            const float su[4] = {su0.x, su0.y, su1.x, su1.y};
            const float sd[4] = {sd0.x, sd0.y, sd1.x, sd1.y};
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here && ((masks[t] >> warp) & 1))
                {
                    const uint2 uv = *reinterpret_cast<const uint2*>(x_rot[t * 8 + warp] + lane * 4);
                    const float2 uf0 = __half22float2(*reinterpret_cast<const __half2*>(&uv.x));
                    const float2 uf1 = __half22float2(*reinterpret_cast<const __half2*>(&uv.y));
                    float vu[4] = {uf0.x, uf0.y, uf1.x, uf1.y};
                    had128(vu, lane);
                    const int tile = tile0 + tg + t;
                    __half* dst = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                    const uint2 gv = *reinterpret_cast<const uint2*>(dst);
                    const float2 gf0 = __half22float2(*reinterpret_cast<const __half2*>(&gv.x));
                    const float2 gf1 = __half22float2(*reinterpret_cast<const __half2*>(&gv.y));
                    const float g_arr[4] = {gf0.x, gf0.y, gf1.x, gf1.y};
                    float va[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        const float u = rnd_h(vu[i] * su[i]);
                        const float act = bf(static_cast<uint16_t>(swiglu(g_arr[i], u)));
                        va[i] = act * sd[i];
                    }
                    had128(va, lane);
                    uint2 ov;
                    *reinterpret_cast<__half2*>(&ov.x) = __floats2half2_rn(va[0], va[1]);
                    *reinterpret_cast<__half2*>(&ov.y) = __floats2half2_rn(va[2], va[3]);
                    *reinterpret_cast<uint2*>(dst) = ov;
                }
            }
            __syncthreads();
        }
    }
}

// Wide-load fused-b multi-HB PMCGX FC2 using dynamic smem for a_rot [nb][NT*8][136].
// Grid: (slots, ceil(HB / HB_PER_CTA)), Block: 256 (8 warps).
// Accumulates all b=0..nb-1 in Hadamard space BEFORE a SINGLE had128 (9x fewer had128, 0 barriers in b-loop)!
template <int NT, int HB_PER_CTA>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc2_wide_mhb(Tr3Args a)
{
    const int slot = blockIdx.x, hbg = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int HB = a.H / 128;
    const int hb0 = hbg * HB_PER_CTA;
    if (hb0 >= HB) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;

    extern __shared__ __half x2_dyn[];   // [nb][NT * 8][136]
    __shared__ float y_raw[8][128];

    for (int tg = 0; tg < nt; tg += NT)
    {
        const int nt_here = min(NT, nt - tg);
        int masks[NT];
        float rws[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const int tile = tile0 + tg + t;
                masks[t] = a.plan.tile_mask[tile];
                rws[t] = ((masks[t] >> warp) & 1) ? a.plan.tile_w[tile * 8 + warp] : 0.f;
            }
            else
            {
                masks[t] = 0;
                rws[t] = 0.f;
            }
        }

        // Preload ALL nb blocks of a_rot for this tile-group into dynamic smem ONCE!
        for (int b = 0; b < nb; ++b)
        {
            __half* xb = x2_dyn + static_cast<size_t>(b) * (NT * 8 * 136);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here && ((masks[t] >> warp) & 1))
                {
                    const int tile = tile0 + tg + t;
                    const __half* src = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                    *reinterpret_cast<uint2*>(xb + (t * 8 + warp) * 136 + lane * 4) = *reinterpret_cast<const uint2*>(src);
                }
            }
        }
        __syncthreads();

        for (int hi = 0; hi < HB_PER_CTA; ++hi)
        {
            const int hb = hb0 + hi;
            if (hb >= HB) break;
            const int n_tile = hb * 8 + warp;

            float c_tot[NT][4];
            #pragma unroll
            for (int t = 0; t < NT; ++t) c_tot[t][0] = c_tot[t][1] = c_tot[t][2] = c_tot[t][3] = 0.f;

            for (int b = 0; b < nb; ++b)
            {
                const uint8_t* mw = fc2_block(a.fc2, tr, b, bits, a.H);
                const __half* xb = x2_dyn + static_cast<size_t>(b) * (NT * 8 * 136);
                float cb[NT][4];
                #pragma unroll
                for (int t = 0; t < NT; ++t) cb[t][0] = cb[t][1] = cb[t][2] = cb[t][3] = 0.f;
                if (bits == 3) pmcgx_fc2_ntile_wide<3, NT>(mw, n_tile, lane, xb, masks, nt_here, cb);
                else if (bits == 4) pmcgx_fc2_ntile_wide<4, NT>(mw, n_tile, lane, xb, masks, nt_here, cb);
                else pmcgx_fc2_ntile_wide<2, NT>(mw, n_tile, lane, xb, masks, nt_here, cb);
                const size_t brow = static_cast<size_t>(tr[2]) + b;
                const float bscale = __half2float(a.v3b[brow * HB + hb]);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    if (t < nt_here)
                    {
                        c_tot[t][0] = fmaf(rnd_h(cb[t][0]), bscale, c_tot[t][0]);
                        c_tot[t][1] = fmaf(rnd_h(cb[t][1]), bscale, c_tot[t][1]);
                        c_tot[t][2] = fmaf(rnd_h(cb[t][2]), bscale, c_tot[t][2]);
                        c_tot[t][3] = fmaf(rnd_h(cb[t][3]), bscale, c_tot[t][3]);
                    }
                }
            }

            const uint2 sv = *reinterpret_cast<const uint2*>(a.sg + a.H + hb * 128 + lane * 4);
            const float2 sf0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
            const float2 sf1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
            const float sg_v[4] = {sf0.x, sf0.y, sf1.x, sf1.y};

            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here)
                {
                    const int t0 = 2 * col, t1 = 2 * col + 1;
                    if ((masks[t] >> t0) & 1)
                    {
                        y_raw[t0][warp * 16 + row0] = c_tot[t][0];
                        y_raw[t0][warp * 16 + row0 + 8] = c_tot[t][2];
                    }
                    if ((masks[t] >> t1) & 1)
                    {
                        y_raw[t1][warp * 16 + row0] = c_tot[t][1];
                        y_raw[t1][warp * 16 + row0 + 8] = c_tot[t][3];
                    }
                    __syncthreads();

                    if ((masks[t] >> warp) & 1)
                    {
                        float v[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
                        had128(v, lane);
                        const float rw = rws[t];
                        const int tile = tile0 + tg + t;
                        const size_t o = static_cast<size_t>(tile * 8 + warp) * a.H + hb * 128 + lane * 4;
                        if (a.y_bf16)
                        {
                            uint16_t* dst = static_cast<uint16_t*>(a.ypart) + o;
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) dst[i] = static_cast<uint16_t>(nv4::bf16_rne(rw * (v[i] * sg_v[i])));
                        }
                        else
                        {
                            *reinterpret_cast<float4*>(static_cast<float*>(a.ypart) + o) =
                                make_float4(rw * (v[0] * sg_v[0]), rw * (v[1] * sg_v[1]), rw * (v[2] * sg_v[2]), rw * (v[3] * sg_v[3]));
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

// High-occupancy streamed PMCGX FC2 for large M (prefill): loads 1 block b of a_rot into small static smem
// x2_sm[NT*8][136] (12.8KB for NT=4, 21.5KB for NT=8 -> 3-4x higher occupancy than 82KB dynamic smem,
// zero cudaFuncSetAttribute, and 2x fewer trellis weight decodes at M>=384!). 100% bit-identical to fc2_wide_mhb.
template <int NT>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc2_wide_stream(Tr3Args a)
{
    const int slot = blockIdx.x, hb = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int HB = a.H / 128;
    if (hb >= HB) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    __shared__ __half x2_sm[NT * 8][136];
    __shared__ float y_raw[8][128];

    const uint2 sv = *reinterpret_cast<const uint2*>(a.sg + a.H + hb * 128 + lane * 4);
    const float2 sf0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
    const float2 sf1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
    const float sg_v[4] = {sf0.x, sf0.y, sf1.x, sf1.y};

    for (int tg = 0; tg < nt; tg += NT)
    {
        const int nt_here = min(NT, nt - tg);
        int masks[NT];
        float rws[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const int tile = tile0 + tg + t;
                masks[t] = a.plan.tile_mask[tile];
                rws[t] = ((masks[t] >> warp) & 1) ? a.plan.tile_w[tile * 8 + warp] : 0.f;
            }
            else
            {
                masks[t] = 0;
                rws[t] = 0.f;
            }
        }

        float c_tot[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) c_tot[t][0] = c_tot[t][1] = c_tot[t][2] = c_tot[t][3] = 0.f;

        for (int b = 0; b < nb; ++b)
        {
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here && ((masks[t] >> warp) & 1))
                {
                    const int tile = tile0 + tg + t;
                    const __half* src = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                    *reinterpret_cast<uint2*>(x2_sm[t * 8 + warp] + lane * 4) = *reinterpret_cast<const uint2*>(src);
                }
            }
            __syncthreads();

            const uint8_t* mw = fc2_block(a.fc2, tr, b, bits, a.H);
            float cb[NT][4];
            #pragma unroll
            for (int t = 0; t < NT; ++t) cb[t][0] = cb[t][1] = cb[t][2] = cb[t][3] = 0.f;
            if (bits == 3) pmcgx_fc2_ntile_wide<3, NT>(mw, n_tile, lane, &x2_sm[0][0], masks, nt_here, cb);
            else if (bits == 4) pmcgx_fc2_ntile_wide<4, NT>(mw, n_tile, lane, &x2_sm[0][0], masks, nt_here, cb);
            else pmcgx_fc2_ntile_wide<2, NT>(mw, n_tile, lane, &x2_sm[0][0], masks, nt_here, cb);
            const size_t brow = static_cast<size_t>(tr[2]) + b;
            const float bscale = __half2float(a.v3b[brow * HB + hb]);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (t < nt_here)
                {
                    c_tot[t][0] = fmaf(rnd_h(cb[t][0]), bscale, c_tot[t][0]);
                    c_tot[t][1] = fmaf(rnd_h(cb[t][1]), bscale, c_tot[t][1]);
                    c_tot[t][2] = fmaf(rnd_h(cb[t][2]), bscale, c_tot[t][2]);
                    c_tot[t][3] = fmaf(rnd_h(cb[t][3]), bscale, c_tot[t][3]);
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            if (t < nt_here)
            {
                const int t0 = 2 * col, t1 = 2 * col + 1;
                if ((masks[t] >> t0) & 1)
                {
                    y_raw[t0][warp * 16 + row0] = c_tot[t][0];
                    y_raw[t0][warp * 16 + row0 + 8] = c_tot[t][2];
                }
                if ((masks[t] >> t1) & 1)
                {
                    y_raw[t1][warp * 16 + row0] = c_tot[t][1];
                    y_raw[t1][warp * 16 + row0 + 8] = c_tot[t][3];
                }
                __syncthreads();

                if ((masks[t] >> warp) & 1)
                {
                    float v[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
                    had128(v, lane);
                    const float rw = rws[t];
                    const int tile = tile0 + tg + t;
                    const size_t o = static_cast<size_t>(tile * 8 + warp) * a.H + hb * 128 + lane * 4;
                    if (a.y_bf16)
                    {
                        uint16_t* dst = static_cast<uint16_t*>(a.ypart) + o;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) dst[i] = static_cast<uint16_t>(nv4::bf16_rne(rw * (v[i] * sg_v[i])));
                    }
                    else
                    {
                        *reinterpret_cast<float4*>(static_cast<float*>(a.ypart) + o) =
                            make_float4(rw * (v[0] * sg_v[0]), rw * (v[1] * sg_v[1]), rw * (v[2] * sg_v[2]), rw * (v[3] * sg_v[3]));
                    }
                }
                __syncthreads();
            }
        }
    }
}

// Vectorized float4 deterministic route combine (H % 4 == 0, y_bf16 == 0). Bit-exact with k_combine.
__global__ void __launch_bounds__(256) k_ink2_tr3_combine_vec4(MoeArgs a)
{
    const size_t H4 = static_cast<size_t>(a.H >> 2);
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(a.M) * H4) return;
    const int m = static_cast<int>(i / H4), n4 = static_cast<int>(i % H4);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const float* yp = static_cast<const float*>(a.ypart);
    for (int k = 0; k < a.TK; ++k)
    {
        const size_t row = static_cast<size_t>(a.plan.tok_slot[m * a.TK + k] * 8 + a.plan.tok_col[m * a.TK + k]);
        const float4 v = reinterpret_cast<const float4*>(yp + row * a.H)[n4];
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }
    uint2 ov;
    ov.x = nv4::bf16_rne(s0) | (nv4::bf16_rne(s1) << 16);
    ov.y = nv4::bf16_rne(s2) | (nv4::bf16_rne(s3) << 16);
    reinterpret_cast<uint2*>(a.out + static_cast<size_t>(m) * a.H)[n4] = ov;
}

// grid: (max_slots, eb), block: 256 (8 warps).
// Computes fc1 (gate + up) -> H128 * svh -> Inkling-2 softcapped SwiGLU -> * suh_down -> H128 -> a_rot [row][eb][128].
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc1(Tr3Args a)
{
    const int slot = blockIdx.x, b = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int HB = a.H / 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;

    const uint8_t *mw0 = nullptr, *mw1 = nullptr;
    const int bits0 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 0, a.H, mw0);
    const int bits1 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 1, a.H, mw1);

    __shared__ __half x_rot[8][136];
    __shared__ float y_gate[8][128];
    __shared__ float y_up[8][128];

    for (int tg = 0; tg < nt; ++tg)
    {
        const int tile = tile0 + tg;
        const int mask = a.plan.tile_mask[tile];
        const bool w_act = ((mask >> warp) & 1) != 0;
        const int w_tok = w_act ? a.plan.tile_tok[tile * 8 + warp] : 0;

        for (int mat = 0; mat < 2; ++mat)
        {
            const uint8_t* mw = mat == 0 ? mw0 : mw1;
            const int bits = mat == 0 ? bits0 : bits1;
            const __half* suh = a.v1 + (brow * 2 + mat) * a.H;
            float c[4] = {0.f, 0.f, 0.f, 0.f};

            for (int hb = 0; hb < HB; ++hb)
            {
                if (w_act)
                {
                    const uint16_t* xp = a.x + static_cast<size_t>(w_tok) * a.H + hb * 128 + lane * 4;
                    const __half* sp = suh + hb * 128 + lane * 4;
                    float v[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) v[i] = bf(xp[i]) * __half2float(sp[i]);
                    had128(v, lane);
                    __half* dst = x_rot[warp] + lane * 4;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(v[i]);
                }
                __syncthreads();
                if (bits == 3) fc1_hb_mma<3, CB>(mw, hb, warp, lane, x_rot, mask, c);
                else if (bits == 4) fc1_hb_mma<4, CB>(mw, hb, warp, lane, x_rot, mask, c);
                else fc1_hb_mma<2, CB>(mw, hb, warp, lane, x_rot, mask, c);
                __syncthreads();
            }

            float (*ydst)[128] = mat == 0 ? y_gate : y_up;
            const int t0 = 2 * col, t1 = 2 * col + 1;
            if ((mask >> t0) & 1)
            {
                ydst[t0][warp * 16 + row0] = rnd_h(c[0]);
                ydst[t0][warp * 16 + row0 + 8] = rnd_h(c[2]);
            }
            if ((mask >> t1) & 1)
            {
                ydst[t1][warp * 16 + row0] = rnd_h(c[1]);
                ydst[t1][warp * 16 + row0 + 8] = rnd_h(c[3]);
            }
            __syncthreads();
        }

        if (w_act)
        {
            float vg[4], vu[4];
            const __half* svh_g = a.v2 + (brow * 3 + 0) * 128 + lane * 4;
            const __half* svh_u = a.v2 + (brow * 3 + 1) * 128 + lane * 4;
            const __half* suh_d = a.v2 + (brow * 3 + 2) * 128 + lane * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                vg[i] = y_gate[warp][lane * 4 + i];
                vu[i] = y_up[warp][lane * 4 + i];
            }
            had128(vg, lane);
            had128(vu, lane);
            float va[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float g = rnd_h(vg[i] * __half2float(svh_g[i]));
                const float u = rnd_h(vu[i] * __half2float(svh_u[i]));
                const float act = bf(static_cast<uint16_t>(swiglu(g, u)));
                va[i] = act * __half2float(suh_d[i]);
            }
            had128(va, lane);
            __half* dst = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(va[i]);
        }
        __syncthreads();
    }
}

// grid: (max_slots, HB), block: 256 (8 warps).
// Accumulates down projection over b = 0..nb-1, applies H128 * v3[brow] * route_weight, writes ypart [tile*8+col][H].
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc2(Tr3Args a)
{
    const int slot = blockIdx.x, hb = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    __shared__ __half x2_sm[8][136];
    __shared__ float y_raw[8][128];

    for (int tg = 0; tg < nt; ++tg)
    {
        const int tile = tile0 + tg;
        const int mask = a.plan.tile_mask[tile];
        const bool w_act = ((mask >> warp) & 1) != 0;
        const float rw = w_act ? a.plan.tile_w[tile * 8 + warp] : 0.f;
        float sum_v[4] = {0.f, 0.f, 0.f, 0.f};

        for (int b = 0; b < nb; ++b)
        {
            if (w_act)
            {
                const __half* src = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                *reinterpret_cast<uint2*>(x2_sm[warp] + lane * 4) = *reinterpret_cast<const uint2*>(src);
            }
            __syncthreads();

            const uint8_t* mw = fc2_block(a.fc2, tr, b, bits, a.H);
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            if (bits == 3) fc2_ntile_mma<3, CB>(mw, n_tile, lane, x2_sm, mask, c);
            else if (bits == 4) fc2_ntile_mma<4, CB>(mw, n_tile, lane, x2_sm, mask, c);
            else fc2_ntile_mma<2, CB>(mw, n_tile, lane, x2_sm, mask, c);

            const int t0 = 2 * col, t1 = 2 * col + 1;
            if ((mask >> t0) & 1)
            {
                y_raw[t0][warp * 16 + row0] = rnd_h(c[0]);
                y_raw[t0][warp * 16 + row0 + 8] = rnd_h(c[2]);
            }
            if ((mask >> t1) & 1)
            {
                y_raw[t1][warp * 16 + row0] = rnd_h(c[1]);
                y_raw[t1][warp * 16 + row0 + 8] = rnd_h(c[3]);
            }
            __syncthreads();

            if (w_act)
            {
                float v[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
                had128(v, lane);
                const size_t brow = static_cast<size_t>(tr[2]) + b;
                const __half* svh = a.v3 + brow * a.H + hb * 128 + lane * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) sum_v[i] += rw * (v[i] * __half2float(svh[i]));
            }
            __syncthreads();
        }

        if (w_act)
        {
            const size_t o = static_cast<size_t>(tile * 8 + warp) * a.H + hb * 128 + lane * 4;
            if (a.y_bf16)
            {
                uint16_t* dst = static_cast<uint16_t*>(a.ypart) + o;
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = static_cast<uint16_t>(nv4::bf16_rne(sum_v[i]));
            }
            else
            {
                *reinterpret_cast<float4*>(static_cast<float*>(a.ypart) + o) = make_float4(sum_v[0], sum_v[1], sum_v[2], sum_v[3]);
            }
        }
    }
}

// Set once at runtime startup; standalone kernel users retain the default dispatch.
inline bool& tr3_wide_enabled()
{
    static bool on = true;
    return on;
}

template <int CB>
inline void launch_tr3(const Tr3Args& a, int NE, cudaStream_t st, Prof* prof = nullptr)
{
    const int slots = plan_max_slots(a.M, a.TK, NE);
    const int HB = a.H / 128;
    const bool use_wide = (CB == CB_PMCGX && tr3_wide_enabled() && a.wide && a.sg != nullptr && a.v1b != nullptr && a.v3b != nullptr && a.x_had != nullptr);
    if (use_wide)
    {
        {
            ProfScope _ps(prof, PS_MOE_FC1, st);
            k_ink2_tr3_prerot_x<<<a.M * HB, 32, 0, st>>>(a.x, a.sg, a.x_had, a.M, a.H);
            if (a.M <= 16) k_ink2_tr3_fc1_wide<1><<<dim3(slots, a.eb), 256, 0, st>>>(a);
            else if (a.M < 128) k_ink2_tr3_fc1_wide<2><<<dim3(slots, a.eb), 256, 0, st>>>(a);
            else if (a.M < 384) k_ink2_tr3_fc1_wide<4><<<dim3(slots, a.eb), 256, 0, st>>>(a);
            else k_ink2_tr3_fc1_wide<8><<<dim3(slots, a.eb), 256, 0, st>>>(a);
        }
        {
            ProfScope _ps(prof, PS_MOE_FC2, st);
            if (a.M <= 16)
            {
                constexpr int HB_C = 2;
                const size_t smem = static_cast<size_t>(a.eb) * 8 * 136 * sizeof(__half);
                static size_t set1 = 0;
                if (smem > 40000 && smem > set1)
                {
                    cudaFuncSetAttribute(k_ink2_tr3_fc2_wide_mhb<1, HB_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
                    set1 = smem;
                }
                k_ink2_tr3_fc2_wide_mhb<1, HB_C><<<dim3(slots, (HB + HB_C - 1) / HB_C), 256, smem, st>>>(a);
            }
            else if (a.M < 128)
            {
                constexpr int HB_C = 3;
                const size_t smem = static_cast<size_t>(a.eb) * 16 * 136 * sizeof(__half);
                static size_t set2 = 0;
                if (smem > 40000 && smem > set2)
                {
                    cudaFuncSetAttribute(k_ink2_tr3_fc2_wide_mhb<2, HB_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
                    set2 = smem;
                }
                k_ink2_tr3_fc2_wide_mhb<2, HB_C><<<dim3(slots, (HB + HB_C - 1) / HB_C), 256, smem, st>>>(a);
            }
            else if (a.M < 384)
            {
                k_ink2_tr3_fc2_wide_stream<4><<<dim3(slots, HB), 256, 0, st>>>(a);
            }
            else
            {
                k_ink2_tr3_fc2_wide_stream<8><<<dim3(slots, HB), 256, 0, st>>>(a);
            }
        }
    }
    else
    {
        {
            ProfScope _ps(prof, PS_MOE_FC1, st);
            k_ink2_tr3_fc1<CB><<<dim3(slots, a.eb), 256, 0, st>>>(a);
        }
        {
            ProfScope _ps(prof, PS_MOE_FC2, st);
            k_ink2_tr3_fc2<CB><<<dim3(slots, HB), 256, 0, st>>>(a);
        }
    }
    if (a.out != nullptr)
    {
        ProfScope _ps(prof, PS_MOE_COMBINE, st);
        MoeArgs ma{};
        ma.H = a.H;
        ma.M = a.M;
        ma.TK = a.TK;
        ma.plan = a.plan;
        ma.ypart = a.ypart;
        ma.y_bf16 = a.y_bf16;
        ma.out = a.out;
        if (!a.y_bf16 && (a.H & 3) == 0)
            k_ink2_tr3_combine_vec4<<<static_cast<unsigned>((static_cast<size_t>(a.M) * (a.H >> 2) + 255) / 256), 256, 0, st>>>(ma);
        else
            k_combine<<<static_cast<unsigned>((static_cast<size_t>(a.M) * a.H + 255) / 256), 256, 0, st>>>(ma);
    }
}

inline void moe_tr3(const Tr3Args& a, int cb, int NE, cudaStream_t st, Prof* prof = nullptr)
{
    if (a.H % 128 != 0 || a.eb < 1 || a.M < 1) throw std::runtime_error("ink2::tr3::moe_tr3: H, eb or M");
    if (cb == CB_PMCGX) launch_tr3<CB_PMCGX>(a, NE, st, prof);
    else if (cb == CB_MUL1) launch_tr3<CB_MUL1>(a, NE, st, prof);
    else launch_tr3<CB_MCG>(a, NE, st, prof);
}
#endif   // __CUDACC__
}   // namespace tr3
}   // namespace ink2
