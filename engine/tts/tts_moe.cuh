// Native grouped EXLR8 trellis MoE kernels for Drama 3 (g6tts --weight-mode native).
// Adapted from `engine/kernels/qw38_moe.cuh` and `engine/kernels/tr3.cuh` for Drama 3's TP=1 geometry:
//   - HID = 2048 (HB = 16 blocks of 128, KSH = 128 k16-steps)
//   - NEXP = 256 routed experts + shared expert at table row 256
//   - EXP_I = 512 (EB = 4 blocks of 128), TOPK = 8
//   - Supports BITS in {2, 3, 4} per (block, matrix) via `tr3.kmix` and codebooks TR3_MCG / TR3_MUL1 / TR3_PMCGX
//   - Zero host sync: on-GPU group builder -> pre-rotated x_had -> grouped fc1 -> grouped fc2 -> deterministic combine
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "tts_config.h"

namespace g6tts {

constexpr int MOE_HID = geom::HID;            // 2048
constexpr int MOE_KSH = geom::HID / 16;       // 128
constexpr int MOE_HB = geom::HID / 128;       // 16
constexpr int MOE_EB = geom::EXP_I / 128;     // 4
constexpr int MOE_NEXP = geom::NEXP;          // 256
constexpr int MOE_TOPK = geom::TOPK;          // 8
constexpr int MOE_SHARED_E = geom::NEXP;      // 256 (table row of trellis shared expert)
constexpr float MOE_SQ128 = 0.088388347648f;  // 1 / sqrt(128)

enum TtsTr3Cb {
    TTS_TR3_MCG = 0,
    TTS_TR3_MUL1 = 1,
    TTS_TR3_PMCGX = 2,
    TTS_TR3_NCB = 3,
};

struct TtsTr3LayerW {
    bool valid = false;
    bool has_shared = false;
    int cb = TTS_TR3_MCG;
    bool e4 = true;
    int n_rows = 0; // 256 or 257 (with shared expert at row 256)

    const uint8_t* fc1 = nullptr;   // [fc1_bytes]
    const uint8_t* fc2 = nullptr;   // [fc2_bytes] (Item128-permuted)
    const __half* sg = nullptr;     // [2, 2048] (in, out signs as +-1 fp16)
    const __half* v1b = nullptr;    // [nblk, 2, 16]
    const __half* v2 = nullptr;     // [nblk, 3, 128]
    const __half* v3b = nullptr;    // [nblk, 16]
    const int64_t* tab = nullptr;   // [n_rows, 6]
    const int64_t* kmix = nullptr;  // [n_rows]
};

// ------------------------------------------------------------------------------------------------ PTX & codebook helpers

__device__ __forceinline__ uint32_t moe_ld_nc(const void* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ __half2 moe_as_h2(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ uint32_t moe_as_u32(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

__device__ __forceinline__ void moe_mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void moe_had128(float (&v)[4], int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? (p - v[e]) : (v[e] + p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= MOE_SQ128;
}

template <bool E4>
__device__ __forceinline__ uint32_t moe_tr3_e4(uint32_t r) {
    if constexpr (E4) {
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    }
    return r;
}

template <bool E4>
__device__ __forceinline__ uint32_t moe_tr3_mcg2(uint32_t w0, uint32_t w1) {
    uint32_t x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const uint32_t lo = __byte_perm(x0, x1, 0x5410);
    const uint32_t hi = __byte_perm(x0, x1, 0x7632);
    return moe_tr3_e4<E4>(moe_as_u32(__hadd2(moe_as_h2(lo), moe_as_h2(hi))));
}

template <bool E4>
__device__ __forceinline__ uint32_t moe_tr3_mul1x2(uint32_t w0, uint32_t w1) {
    const uint32_t s0 = __dp4a(w0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(w1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 v = __hfma2(moe_as_h2(__byte_perm(s0, s1, 0x5410)), moe_as_h2(0x1eee1eeeu), moe_as_h2(0xc931c931u));
    return moe_tr3_e4<E4>(moe_as_u32(v));
}

template <bool E4>
__device__ __forceinline__ uint32_t moe_tr3_pmcgx(uint32_t w) {
    uint32_t x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    return moe_tr3_e4<E4>(moe_as_u32(__hadd2(moe_as_h2(x0), moe_as_h2(__byte_perm(x1, 0, 0x1032)))));
}

// Matches `engine/kernels/tr3.cuh::Tr3Lane<BITS, E4, CB>` bit-for-bit.
template <int BITS, bool E4, int CB>
struct Tr3Lane {
    int i0, i1, sh;

    __device__ __forceinline__ void init(int lane) {
        const int b0 = (lane * 8 + 1) * BITS - 16 + 256 * BITS;
        i0 = (b0 / 32) % (8 * BITS);
        i1 = (i0 + 1) % (8 * BITS);
        sh = 48 - (b0 % 32);
    }

    __device__ __forceinline__ uint32_t loadw(const uint8_t* tile, int lane) const {
        if constexpr (BITS == 4) return moe_ld_nc(tile + lane * 4);
        else return moe_ld_nc(tile + (lane % (8 * BITS)) * 4);
    }

    __device__ __forceinline__ void decodew(uint32_t raw, uint32_t (&f0)[2], uint32_t (&f1)[2]) const {
        if constexpr (BITS == 4) {
            const uint32_t a = __shfl_sync(0xffffffffu, raw, i0);
            decode_m((static_cast<uint64_t>(a) << 32) | raw, 28, f0, f1);
        } else {
            const uint32_t a = __shfl_sync(0xffffffffu, raw, i0);
            const uint32_t b = __shfl_sync(0xffffffffu, raw, i1);
            decode_m((static_cast<uint64_t>(a) << 32) | b, sh, f0, f1);
        }
    }

    __device__ static __forceinline__ void decode_m(uint64_t m, int s, uint32_t (&f0)[2], uint32_t (&f1)[2]) {
        if constexpr (CB == TTS_TR3_PMCGX) {
            uint32_t w[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) w[p] = static_cast<uint32_t>(m >> (s - BITS * (2 * p + 1))) & 0xfffffu;
            f0[0] = moe_tr3_pmcgx<E4>(w[0]);
            f0[1] = moe_tr3_pmcgx<E4>(w[1]);
            f1[0] = moe_tr3_pmcgx<E4>(w[2]);
            f1[1] = moe_tr3_pmcgx<E4>(w[3]);
        } else {
            uint32_t w[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = static_cast<uint32_t>(m >> (s - BITS * j)) & 0xffffu;
            if constexpr (CB == TTS_TR3_MUL1) {
                f0[0] = moe_tr3_mul1x2<E4>(w[0], w[1]);
                f0[1] = moe_tr3_mul1x2<E4>(w[2], w[3]);
                f1[0] = moe_tr3_mul1x2<E4>(w[4], w[5]);
                f1[1] = moe_tr3_mul1x2<E4>(w[6], w[7]);
            } else {
                f0[0] = moe_tr3_mcg2<E4>(w[0], w[1]);
                f0[1] = moe_tr3_mcg2<E4>(w[2], w[3]);
                f1[0] = moe_tr3_mcg2<E4>(w[4], w[5]);
                f1[1] = moe_tr3_mcg2<E4>(w[6], w[7]);
            }
        }
    }
};

// Unit offset & width resolver in `tr3.fc1` (matches `v2plan::v2_layer` for NH=4 <= 8).
__device__ __forceinline__ int tts_tr3_fc1_unit(const uint8_t* t_fc1, const int64_t* t_kmix,
                                                int e, const int64_t* tr, int u, const uint8_t*& mw) {
    constexpr size_t MB = size_t(MOE_KSH) * 8 * 32; // 32768 bytes per bit of width
    const int64_t km = t_kmix ? t_kmix[e] : 0ll;
    if (km == 0ll) {
        const int bits = static_cast<int>(tr[4]);
        mw = t_fc1 + static_cast<size_t>(tr[0]) + size_t(u) * MB * size_t(bits);
        return bits;
    }
    size_t off = static_cast<size_t>(tr[0]);
    for (int j = 0; j < u; ++j) {
        const int bj = static_cast<int>((km >> (4 * j)) & 15ll);
        off += MB * size_t(bj);
    }
    mw = t_fc1 + off;
    return static_cast<int>((km >> (4 * u)) & 15ll);
}

__device__ __forceinline__ const uint8_t* tts_tr3_fc2_block(const uint8_t* t_fc2, const int64_t* tr, int b, int bits) {
    constexpr size_t MB = size_t(MOE_KSH) * 8 * 32; // 32768 bytes per bit of width
    return t_fc2 + static_cast<size_t>(tr[1]) + size_t(b) * MB * size_t(bits);
}

// ------------------------------------------------------------------------------------------------ Grouping & rotation

// Group `n_routes = M * 8` routes by expert (0..255) into chunks of <= 8 routes per expert.
// Single CTA of 256 threads (1 thread per expert), zero host sync!
__global__ void k_tts_moe_build_groups(const int32_t* __restrict__ topk_idx,
                                       int32_t* __restrict__ d_n_groups,
                                       int32_t* __restrict__ grp_exp,
                                       int32_t* __restrict__ grp_cnt,
                                       int32_t* __restrict__ grp_routes,
                                       int n_routes) {
    const int tid = threadIdx.x; // 0..255 = expert id
    __shared__ int cnt[MOE_NEXP];
    __shared__ int g_off[MOE_NEXP];
    __shared__ int cur[MOE_NEXP];

    cnt[tid] = 0;
    cur[tid] = 0;
    __syncthreads();

    for (int r = tid; r < n_routes; r += MOE_NEXP) {
        atomicAdd(&cnt[topk_idx[r]], 1);
    }
    __syncthreads();

    // Exclusive prefix sum of ceil(cnt[e] / 8) over 256 experts (8 warps x 32 lanes)
    const int ng = (cnt[tid] + 7) >> 3;
    const int lane = tid & 31, warp = tid >> 5;
    int val = ng;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, val, off);
        if (lane >= off) val += n;
    }
    __shared__ int wsum[8];
    if (lane == 31) wsum[warp] = val;
    __syncthreads();
    if (warp == 0) {
        int wv = (lane < 8) ? wsum[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, wv, off);
            if (lane >= off) wv += n;
        }
        if (lane < 8) wsum[lane] = wv;
    }
    __syncthreads();
    const int base = (warp > 0 ? wsum[warp - 1] : 0) + (val - ng);
    g_off[tid] = base;
    if (tid == MOE_NEXP - 1) d_n_groups[0] = base + ng;

    const int c_tot = cnt[tid];
    for (int gi = 0; gi < ng; ++gi) {
        const int g = base + gi;
        const int rem = c_tot - gi * 8;
        grp_exp[g] = tid;
        grp_cnt[g] = (rem < 8) ? rem : 8;
    }
    __syncthreads();

    for (int r = tid; r < n_routes; r += MOE_NEXP) {
        const int e = topk_idx[r];
        const int pos = atomicAdd(&cur[e], 1);
        const int g = g_off[e] + (pos >> 3);
        grp_routes[g * 8 + (pos & 7)] = r;
    }
}

// Fast 1-token group builder for M == 1 (8 distinct experts -> 8 single-route groups, 1 warp).
__global__ void k_tts_moe_groups_m1(const int32_t* __restrict__ topk_idx,
                                    int32_t* __restrict__ d_n_groups,
                                    int32_t* __restrict__ grp_exp,
                                    int32_t* __restrict__ grp_cnt,
                                    int32_t* __restrict__ grp_routes) {
    const int k = threadIdx.x;
    if (k < MOE_TOPK) {
        grp_exp[k] = topk_idx[k];
        grp_cnt[k] = 1;
        grp_routes[k * 8] = k;
    }
    if (k == 0) d_n_groups[0] = MOE_TOPK;
}

// Group builder for trellis shared expert (table row 256): ceil(M / 8) groups of up to 8 tokens.
__global__ void k_tts_moe_groups_shared(int M, int shared_e,
                                        int32_t* __restrict__ d_n_groups,
                                        int32_t* __restrict__ grp_exp,
                                        int32_t* __restrict__ grp_cnt,
                                        int32_t* __restrict__ grp_routes) {
    const int ng = (M + 7) >> 3;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g == 0) d_n_groups[0] = ng;
    if (g < ng) {
        const int m0 = g * 8;
        const int cnt = min(8, M - m0);
        grp_exp[g] = shared_e;
        grp_cnt[g] = cnt;
        for (int i = 0; i < cnt; ++i) grp_routes[g * 8 + i] = m0 + i;
    }
}

// Pre-rotate X[M, 2048] (BF16) by layer input signs sg[0, :] + 128-point Sylvester Hadamard -> x_had[M, 2048] (FP16).
// Grid: (M * HB), Block: 32 (1 warp per (token, 128-block)).
__global__ void k_tts_x_had128(const __nv_bfloat16* __restrict__ x,
                               const __half* __restrict__ t_sg,
                               __half* __restrict__ x_had,
                               int M) {
    const int idx = blockIdx.x;
    const int hb = idx % MOE_HB;
    const int m = idx / MOE_HB;
    if (m >= M) return;
    const int lane = threadIdx.x;
    const int c0 = hb * 128 + lane * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x + size_t(m) * MOE_HID + c0);
    const uint2 sv = *reinterpret_cast<const uint2*>(t_sg + c0);
    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
    const float2 s01 = __half22float2(moe_as_h2(sv.x)), s23 = __half22float2(moe_as_h2(sv.y));
    float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
    moe_had128(v, lane);
    *reinterpret_cast<uint2*>(x_had + size_t(m) * MOE_HID + c0) =
        make_uint2(moe_as_u32(__floats2half2_rn(v[0], v[1])), moe_as_u32(__floats2half2_rn(v[2], v[3])));
}

// ------------------------------------------------------------------------------------------------ Grouped FC1 & FC2

template <int BITS, bool E4, int CB>
__device__ __forceinline__ void moe_fc1_hb_stock(const uint8_t* mw, int hb, int warp, int lane,
                                                 const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const size_t tb = size_t(32 * BITS);
    Tr3Lane<BITS, E4, CB> tl;
    tl.init(lane);
    const int ks0 = hb * 8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t* tile = mw + (size_t(ks0 + p) * 8 + warp) * tb;
        const uint32_t raw = tl.loadw(tile, lane);
        uint32_t f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const uint32_t b0 = act ? *reinterpret_cast<const uint32_t*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
        const uint32_t b1 = act ? *reinterpret_cast<const uint32_t*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
        moe_mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

template <int BITS, bool E4, int CB>
__device__ __forceinline__ void moe_fc2_ntile_stock(const uint8_t* mw, int n_tile, int lane,
                                                    const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const size_t tb = size_t(32 * BITS);
    Tr3Lane<BITS, E4, CB> tl;
    tl.init(lane);
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const uint8_t* tile = mw + (size_t(n_tile) * 8 + ks) * tb;
        const uint32_t raw = tl.loadw(tile, lane);
        uint32_t f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const uint32_t b0 = act ? *reinterpret_cast<const uint32_t*>(x2[row0] + ks * 16 + 2 * col) : 0u;
        const uint32_t b1 = act ? *reinterpret_cast<const uint32_t*>(x2[row0] + ks * 16 + 2 * col + 8) : 0u;
        moe_mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

// Grouped FC1 (gate + up + SwiGLU + suh_d + had128 -> fc1_act[route, b, 128] FP16).
// Grid: (max_groups * MOE_EB), Block: 256 (8 warps = 8 n-tiles covering 128 channels).
// `route_div` = MOE_TOPK for routed experts, 1 for trellis shared expert.
template <bool E4, int CB>
__global__ void __launch_bounds__(256) k_tts_moe_fc1_grp(
    const __half* __restrict__ x_had,
    const int32_t* __restrict__ d_n_groups,
    const int32_t* __restrict__ grp_exp,
    const int32_t* __restrict__ grp_cnt,
    const int32_t* __restrict__ grp_routes,
    const int64_t* __restrict__ t_tab,
    const int64_t* __restrict__ t_kmix,
    const uint8_t* __restrict__ t_fc1,
    const __half* __restrict__ t_v1b,
    const __half* __restrict__ t_v2,
    __half* __restrict__ fc1_act,
    int route_div) {
    const int idx = blockIdx.x;
    const int b = idx % MOE_EB;
    const int g = idx / MOE_EB;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const int64_t* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const int cnt = grp_cnt[g];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = lane >> 2, col = lane & 3;

    __shared__ int s_tok[8];
    __shared__ int s_route[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_tok[tid] = r / route_div;
    }
    __syncthreads();

    __shared__ __half x_rot[8][136];
    __shared__ float y_gate[8][128];
    __shared__ float y_up[8][128];

    for (int mat = 0; mat < 2; ++mat) {
        const uint8_t* mw = nullptr;
        const int bits = tts_tr3_fc1_unit(t_fc1, t_kmix, e, tr, 2 * b + mat, mw);
        float c[4] = {0.f, 0.f, 0.f, 0.f};

        for (int hb = 0; hb < MOE_HB; ++hb) {
            if (warp < cnt) {
                const float bscale = __half2float(t_v1b[(brow * 2 + mat) * MOE_HB + hb]);
                const __half2 bsh = __float2half2_rn(bscale);
                const uint2 xv = *reinterpret_cast<const uint2*>(
                    x_had + (size_t(s_tok[warp]) * MOE_HB + hb) * 128 + lane * 4);
                uint2 ov;
                *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
                *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
                *reinterpret_cast<uint2*>(x_rot[warp] + lane * 4) = ov;
            }
            __syncthreads();
            if (bits == 4) moe_fc1_hb_stock<4, E4, CB>(mw, hb, warp, lane, x_rot, cnt, c);
            else if (bits == 3) moe_fc1_hb_stock<3, E4, CB>(mw, hb, warp, lane, x_rot, cnt, c);
            else moe_fc1_hb_stock<2, E4, CB>(mw, hb, warp, lane, x_rot, cnt, c);
            __syncthreads();
        }

        float (*ydst)[128] = (mat == 0) ? y_gate : y_up;
        const int t0 = 2 * col, t1 = 2 * col + 1;
        if (t0 < cnt) {
            ydst[t0][warp * 16 + row0] = c[0];
            ydst[t0][warp * 16 + row0 + 8] = c[2];
        }
        if (t1 < cnt) {
            ydst[t1][warp * 16 + row0] = c[1];
            ydst[t1][warp * 16 + row0 + 8] = c[3];
        }
        __syncthreads();
    }

    if (warp < cnt) {
        float vg[4], vu[4];
        const __half* svh_g = t_v2 + (brow * 3 + 0) * 128 + lane * 4;
        const __half* svh_u = t_v2 + (brow * 3 + 1) * 128 + lane * 4;
        const __half* suh_d = t_v2 + (brow * 3 + 2) * 128 + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = y_gate[warp][lane * 4 + i];
            vu[i] = y_up[warp][lane * 4 + i];
        }
        moe_had128(vg, lane);
        moe_had128(vu, lane);
        float va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float g_val = vg[i] * __half2float(svh_g[i]);
            const float u_val = vu[i] * __half2float(svh_u[i]);
            const float act = (g_val / (1.0f + expf(-g_val))) * u_val;
            va[i] = act * __half2float(suh_d[i]);
        }
        moe_had128(va, lane);
        __half* dst = fc1_act + (size_t(s_route[warp]) * MOE_EB + b) * 128 + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(va[i]);
    }
}

// Grouped FC2 (down projection): 1 CTA per (group g, hb in 0..15).
// Sums all nb = 4 blocks in Hadamard space weighted by `t_v3b`, applies one `had128`, multiplies by
// `t_sg[1, hb*128..]` and `topk_w[route]` (or 1.0f if `topk_w == nullptr` for shared expert), writes FP32 `out_part`.
template <bool E4, int CB>
__global__ void __launch_bounds__(256) k_tts_moe_fc2_grp(
    const __half* __restrict__ fc1_act,
    const int32_t* __restrict__ d_n_groups,
    const int32_t* __restrict__ grp_exp,
    const int32_t* __restrict__ grp_cnt,
    const int32_t* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const int64_t* __restrict__ t_tab,
    const uint8_t* __restrict__ t_fc2,
    const __half* __restrict__ t_sg,
    const __half* __restrict__ t_v3b,
    float* __restrict__ out_part) {
    const int idx = blockIdx.x;
    const int hb = idx % MOE_HB;
    const int g = idx / MOE_HB;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const int64_t* tr = t_tab + 6 * e;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int cnt = grp_cnt[g];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    __shared__ int s_route[8];
    __shared__ float s_rw[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_rw[tid] = topk_w ? topk_w[r] : 1.0f;
    }
    __syncthreads();

    float sg_v[4];
    if (warp < cnt) {
        const uint2 sv = *reinterpret_cast<const uint2*>(t_sg + MOE_HID + hb * 128 + lane * 4);
        const float2 sf0 = __half22float2(moe_as_h2(sv.x));
        const float2 sf1 = __half22float2(moe_as_h2(sv.y));
        sg_v[0] = sf0.x; sg_v[1] = sf0.y; sg_v[2] = sf1.x; sg_v[3] = sf1.y;
    }

    __shared__ __half x2_sm[MOE_EB][8][136];
    __shared__ float y_raw[8][128];

    if (warp < cnt) {
        const size_t rbase = size_t(s_route[warp]) * MOE_EB;
        for (int b = 0; b < nb; ++b) {
            const __half* x2 = fc1_act + (rbase + b) * 128 + lane * 4;
            *reinterpret_cast<uint2*>(x2_sm[b][warp] + lane * 4) = *reinterpret_cast<const uint2*>(x2);
        }
    }
    __syncthreads();

    float c_tot[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nb; ++b) {
        const uint8_t* mw = tts_tr3_fc2_block(t_fc2, tr, b, bits);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        if (bits == 4) moe_fc2_ntile_stock<4, E4, CB>(mw, n_tile, lane, x2_sm[b], cnt, c);
        else if (bits == 3) moe_fc2_ntile_stock<3, E4, CB>(mw, n_tile, lane, x2_sm[b], cnt, c);
        else moe_fc2_ntile_stock<2, E4, CB>(mw, n_tile, lane, x2_sm[b], cnt, c);
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        const float bscale = __half2float(t_v3b[brow * MOE_HB + hb]);
        c_tot[0] = fmaf(c[0], bscale, c_tot[0]);
        c_tot[1] = fmaf(c[1], bscale, c_tot[1]);
        c_tot[2] = fmaf(c[2], bscale, c_tot[2]);
        c_tot[3] = fmaf(c[3], bscale, c_tot[3]);
    }

    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        y_raw[t0][warp * 16 + row0] = c_tot[0];
        y_raw[t0][warp * 16 + row0 + 8] = c_tot[2];
    }
    if (t1 < cnt) {
        y_raw[t1][warp * 16 + row0] = c_tot[1];
        y_raw[t1][warp * 16 + row0 + 8] = c_tot[3];
    }
    __syncthreads();

    if (warp < cnt) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
        moe_had128(v, lane);
        const float rw = s_rw[warp];
        float* dst = out_part + size_t(s_route[warp]) * MOE_HID + hb * 128 + lane * 4;
        *reinterpret_cast<float4*>(dst) = make_float4(
            rw * (v[0] * sg_v[0]), rw * (v[1] * sg_v[1]),
            rw * (v[2] * sg_v[2]), rw * (v[3] * sg_v[3]));
    }
}

// Deterministic per-route reduction: writes sum_{k=0..7} fc2_part[m*8 + k, :] into FP32 routed_out[m, :].
// Grid: (M * 8), Block: 64 (each thread handles float4 -> 8 * 64 * 4 = 2048 exact).
__global__ void __launch_bounds__(64) k_tts_moe_combine_f32(
    const float* __restrict__ fc2_part,
    float* __restrict__ routed_out,
    int M) {
    const int m = blockIdx.x >> 3;
    const int blk = blockIdx.x & 7;
    if (m >= M) return;
    const int i = blk * 64 + threadIdx.x; // 0..511 (float4 -> 2048)
    const int r0 = m * MOE_TOPK;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
        const float4 v = reinterpret_cast<const float4*>(fc2_part + size_t(r0 + k) * MOE_HID)[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    reinterpret_cast<float4*>(routed_out + size_t(m) * MOE_HID)[i] = acc;
}

// ------------------------------------------------------------------------------------------------ Scratch & dispatch

struct TtsMoeScratch {
    bool initialized = false;
    int max_M = 0;
    int32_t* d_n_groups = nullptr;   // [1]
    int32_t* grp_exp = nullptr;      // [max_M * 8]
    int32_t* grp_cnt = nullptr;      // [max_M * 8]
    int32_t* grp_routes = nullptr;   // [max_M * 8 * 8]
    __half* x_had = nullptr;         // [max_M, 2048]
    __half* fc1_act = nullptr;       // [max_M * 8, 4, 128]
    float* fc2_part = nullptr;       // [max_M * 8, 2048]
    size_t scratch_bytes = 0;

    ~TtsMoeScratch() { release(); }

    void release() {
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        freep(d_n_groups); freep(grp_exp); freep(grp_cnt); freep(grp_routes);
        freep(x_had); freep(fc1_act); freep(fc2_part);
        scratch_bytes = 0;
        max_M = 0;
        initialized = false;
    }

    void init(int max_chunk) {
        release();
        max_M = std::max(16, max_chunk);
        const size_t max_routes = size_t(max_M) * MOE_TOPK;
        auto alloc_d = [&](auto*& p, size_t n) {
            const size_t nb = n * sizeof(*p);
            if (cudaMalloc(&p, nb) != cudaSuccess) {
                throw std::runtime_error("g6tts TtsMoeScratch: cudaMalloc failed");
            }
            scratch_bytes += nb;
        };
        alloc_d(d_n_groups, 1);
        alloc_d(grp_exp, max_routes);
        alloc_d(grp_cnt, max_routes);
        alloc_d(grp_routes, max_routes * 8);
        alloc_d(x_had, size_t(max_M) * MOE_HID);
        alloc_d(fc1_act, max_routes * MOE_EB * 128);
        alloc_d(fc2_part, max_routes * MOE_HID);
        initialized = true;
    }
};

template <class F>
inline void tts_tr3_dispatch(int cb, bool e4, F&& fn) {
    if (e4) {
        if (cb == TTS_TR3_PMCGX) fn(std::integral_constant<bool, true>{}, std::integral_constant<int, TTS_TR3_PMCGX>{});
        else if (cb == TTS_TR3_MUL1) fn(std::integral_constant<bool, true>{}, std::integral_constant<int, TTS_TR3_MUL1>{});
        else fn(std::integral_constant<bool, true>{}, std::integral_constant<int, TTS_TR3_MCG>{});
    } else {
        if (cb == TTS_TR3_PMCGX) fn(std::integral_constant<bool, false>{}, std::integral_constant<int, TTS_TR3_PMCGX>{});
        else if (cb == TTS_TR3_MUL1) fn(std::integral_constant<bool, false>{}, std::integral_constant<int, TTS_TR3_MUL1>{});
        else fn(std::integral_constant<bool, false>{}, std::integral_constant<int, TTS_TR3_MCG>{});
    }
}

// Pre-rotates x[M, 2048] into `sc.x_had` once per MoE layer (reused by both trellis shared expert and routed experts).
inline void tts_moe_prepare_x_had(const __nv_bfloat16* d_norm, const TtsTr3LayerW& tr3, int M,
                                  TtsMoeScratch& sc, cudaStream_t st) {
    k_tts_x_had128<<<M * MOE_HB, 32, 0, st>>>(d_norm, tr3.sg, sc.x_had, M);
}

// Evaluates the trellis shared expert (table row 256) into FP32 `d_shared_out[M, 2048]` (zero host sync).
inline void tts_moe_shared_forward(const TtsTr3LayerW& tr3, int M, TtsMoeScratch& sc,
                                   float* d_shared_out, cudaStream_t st) {
    const int ng = (M + 7) >> 3;
    k_tts_moe_groups_shared<<<(ng + 63) / 64, 64, 0, st>>>(
        M, MOE_SHARED_E, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes);
    tts_tr3_dispatch(tr3.cb, tr3.e4, [&](auto e4_c, auto cb_c) {
        constexpr bool E4 = decltype(e4_c)::value;
        constexpr int CB = decltype(cb_c)::value;
        k_tts_moe_fc1_grp<E4, CB><<<ng * MOE_EB, 256, 0, st>>>(
            sc.x_had, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            tr3.tab, tr3.kmix, tr3.fc1, tr3.v1b, tr3.v2, sc.fc1_act, 1);
        k_tts_moe_fc2_grp<E4, CB><<<ng * MOE_HB, 256, 0, st>>>(
            sc.fc1_act, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            nullptr, tr3.tab, tr3.fc2, tr3.sg, tr3.v3b, d_shared_out);
    });
}

// Evaluates all M * 8 routed experts and deterministically adds weighted outputs into FP32 `d_residual[M, 2048]`.
inline void tts_moe_routed_forward(const TtsTr3LayerW& tr3,
                                   const int32_t* d_topk_ids, const float* d_topk_w,
                                   int M, TtsMoeScratch& sc, float* d_residual, cudaStream_t st) {
    const int n_routes = M * MOE_TOPK;
    const int max_groups = (M == 1) ? MOE_TOPK : std::min(n_routes, MOE_NEXP + n_routes / 8);
    if (M == 1) {
        k_tts_moe_groups_m1<<<1, 32, 0, st>>>(d_topk_ids, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes);
    } else {
        k_tts_moe_build_groups<<<1, MOE_NEXP, 0, st>>>(
            d_topk_ids, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
    }
    tts_tr3_dispatch(tr3.cb, tr3.e4, [&](auto e4_c, auto cb_c) {
        constexpr bool E4 = decltype(e4_c)::value;
        constexpr int CB = decltype(cb_c)::value;
        k_tts_moe_fc1_grp<E4, CB><<<max_groups * MOE_EB, 256, 0, st>>>(
            sc.x_had, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            tr3.tab, tr3.kmix, tr3.fc1, tr3.v1b, tr3.v2, sc.fc1_act, MOE_TOPK);
        k_tts_moe_fc2_grp<E4, CB><<<max_groups * MOE_HB, 256, 0, st>>>(
            sc.fc1_act, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            d_topk_w, tr3.tab, tr3.fc2, tr3.sg, tr3.v3b, sc.fc2_part);
    });
    k_tts_moe_combine_f32<<<M * 8, 64, 0, st>>>(sc.fc2_part, d_residual, M);
}

} // namespace g6tts
