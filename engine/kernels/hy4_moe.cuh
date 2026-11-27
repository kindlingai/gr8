// Tencent Hy4-flash Dense L0 MLP + Latent MoE (routed_down -> 512 PMCGX routed experts top-16 -> routed_up)
// + PMCGX Shared Expert (K3/K4).
// Verified verbatim against HF PR 50903 modeling_hy_v4_flash.py:
//   - Layer 0 Dense MLP: I = 13824 (TP-sharded: 13824 at TP=1, 6912 at TP=2), silu(gate)*up (NO swiglu clamp!),
//     followed by TP AllReduce + post_fc2_norm.
//   - Layers 1..41 Router: FP32 logits -> sigmoid -> choice = scores + e_score_correction_bias (FP32) ->
//     top-16 (stable lowest-index tie-break) -> gather pre-bias scores -> normalize (+1e-20) * 3.998.
//   - Layers 1..41 Routed Experts (EH=1536, EHB=12, EXP_I=1536, EB=12/nr):
//     PMCGX E4M3-rounded trellis (K=2/3/4 + per-matrix kmix, wide-load tile order), swiglu_limit=10.0
//     (gate <= 10.0, up in [-10.0, 10.0]).
//   - Layers 1..41 Shared Expert (H=3072, HB=24, SHARED_I=1536, EB=12/nr):
//     PMCGX E4M3-rounded trellis (K=3/4, wide-load tile order), NO swiglu clamp!
//   - Single TP AllReduce optimization: because routed_expert_up_proj is linear, each rank computes
//     out_part_r = routed_up(routed_lat_part_r) + shared_part_r -> ONE AllReduce -> combine_output_norm!
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include "../model_geom_hy4.h"
#include "hy4_hc.cuh"
#include "hy4_mxfp8.cuh"

namespace hy4 {

using u32 = uint32_t;
using u64 = uint64_t;

static constexpr float HY4_SQ128 = 0.08838834764831844f; // 1 / sqrt(128)

__device__ __forceinline__ __half2 as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ u32 rd32_u8(const uint8_t* p) { return *reinterpret_cast<const u32*>(p); }
__device__ __forceinline__ u64 rd64_u8(const uint8_t* p) { return *reinterpret_cast<const u64*>(p); }

__device__ __forceinline__ void mma16816_f16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void had128(float (&v)[4], int lane) {
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
    for (int e = 0; e < 4; ++e) v[e] *= HY4_SQ128;
}

__device__ __forceinline__ u32 pmcgx_e4m3(u32 w) {
    u32 x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    u32 r = as_u32(__hadd2(as_h2(x0), as_h2(__byte_perm(x1, 0, 0x1032))));
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

// Decode 8 PMCGX E4M3 weights (4 pair states) from shifted 64-bit window `m` -> mma A fragment (f0[0..1], f1[0..1]).
template <int BITS>
__device__ __forceinline__ void pmcgx_decode_m(u64 m, u32 (&f0)[2], u32 (&f1)[2]) {
    constexpr int B0 = 7 * BITS;
    f0[0] = pmcgx_e4m3(static_cast<u32>(m >> (B0 - 1 * BITS)) & 0xfffffu);
    f0[1] = pmcgx_e4m3(static_cast<u32>(m >> (B0 - 3 * BITS)) & 0xfffffu);
    f1[0] = pmcgx_e4m3(static_cast<u32>(m >> (B0 - 5 * BITS)) & 0xfffffu);
    f1[1] = pmcgx_e4m3(static_cast<u32>(m >> (B0 - 7 * BITS)) & 0xfffffu);
}

// Wide-load helpers (exact match to gr8 tr3_wide.cuh / ink2_tr3_moe.cuh).
template <int BITS>
__device__ __forceinline__ u32 trw_tile_chunk(const uint8_t* tile, int lane) {
    constexpr int CB = 8 * BITS, NW = 8 * BITS;
    const int bit = lane * CB, wi = bit >> 5, off = bit & 31;
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi + 1 < NW ? wi + 1 : NW - 1];
    return static_cast<u32>(m >> (64 - off - CB)) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__device__ __forceinline__ void trw_pack(u64 (&v)[3], int p, u32 chunk) {
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if (CB > 16 && sh > 64 - CB) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
}

template <int BITS>
__device__ __forceinline__ u32 trw_chunk(const u64 (&w)[3], int p) {
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if (CB > 16 && sh > 64 - CB) v |= w[n + 1] << (64 - sh);
    return static_cast<u32>(v) & static_cast<u32>((1ull << CB) - 1);
}

// Resolve FC1 unit (unit = 2*b + mat) byte pointer and width `bits`, supporting ALL kmix encodings from v2plan.rs:
//   km == 0: uniform expert width tr[4]
//   km <  0: bit-63 uniform-per-matrix mode (gate = km & 0xf, up = (km >> 4) & 0xf) — used by Hy4 TP=1 (nb=12)!
//   km >  0: per-unit 4-bit width at shift 4*u — used by Hy4 TP=2 (nb=6)!
template <int HB>
__device__ __forceinline__ int hy4_fc1_unit(const uint8_t* fc1, const long long* kmix, int e,
                                            const long long* tr, int unit, const uint8_t*& mw) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(HB) * 2048; // 16 * H = 16 * 128 * HB
    const long long km = kmix ? kmix[e] : 0LL;
    if (km == 0LL) {
        const int bits = static_cast<int>(tr[4]);
        mw = fc1 + static_cast<size_t>(tr[0]) + static_cast<size_t>(unit) * UNIT_PER_BIT * static_cast<size_t>(bits);
        return bits;
    }
    if (km < 0LL) {
        const int gk = static_cast<int>(km & 0xfull);
        const int uk = static_cast<int>((km >> 4) & 0xfull);
        const int b = unit >> 1, mat = unit & 1;
        const size_t off = (static_cast<size_t>(b) * (gk + uk) + (mat ? static_cast<size_t>(gk) : 0ull)) * UNIT_PER_BIT;
        mw = fc1 + static_cast<size_t>(tr[0]) + off;
        return mat ? uk : gk;
    }
    size_t sum_k = 0;
    for (int u = 0; u < unit; ++u) sum_k += static_cast<size_t>((km >> (4 * u)) & 0xfull);
    mw = fc1 + static_cast<size_t>(tr[0]) + sum_k * UNIT_PER_BIT;
    return static_cast<int>((km >> (4 * unit)) & 0xfull);
}

// =================================================================================================
// One-time startup in-place widen for FC1 and FC2 (works for both HB=12 routed and HB=24 shared!)
// =================================================================================================
template <int HB>
__global__ void __launch_bounds__(256) k_hy4_tr3_widen_fc1(
    uint8_t* fc1, const long long* tab, const long long* kmix, int ne, int eb) {
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int hb = blockIdx.x % HB;
    const int u  = (blockIdx.x / HB) % (2 * eb);
    const int e  = blockIdx.x / (HB * 2 * eb);
    if (e >= ne) return;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if ((u >> 1) >= static_cast<int>(tr[3])) return;

    const uint8_t* mw_c = nullptr;
    const int k = hy4_fc1_unit<HB>(fc1, kmix, e, tr, u, mw_c);
    uint8_t* grp = const_cast<uint8_t*>(mw_c) + static_cast<size_t>(hb) * (2048 * k);
    if (k == 3) {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack<3>(v, p, trw_tile_chunk<3>(grp + (p * 8 + j) * 96, lane));
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
    } else if (k == 2) {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack<2>(v, p, trw_tile_chunk<2>(grp + (p * 8 + j) * 64, lane));
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 2; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
    } else if (k == 4) {
        u32 w[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) w[p] = *reinterpret_cast<const u32*>(grp + (p * 8 + j) * 128 + lane * 4);
        __syncthreads();
        *reinterpret_cast<uint4*>(grp + j * 1024 + lane * 16)       = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(grp + j * 1024 + 512 + lane * 16) = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

template <int HB>
__global__ void __launch_bounds__(256) k_hy4_tr3_widen_fc2(
    uint8_t* fc2, const long long* tab, int ne, int eb) {
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int hb = blockIdx.x % HB;
    const int b  = (blockIdx.x / HB) % eb;
    const int e  = blockIdx.x / (HB * eb);
    if (e >= ne) return;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int k = static_cast<int>(tr[4]);
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(HB) * 2048;
    const int n_tile = hb * 8 + j;
    uint8_t* nt = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * k +
                  static_cast<size_t>(n_tile) * (256 * k);
    if (k == 3) {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack<3>(v, p, trw_tile_chunk<3>(nt + p * 96, lane));
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(nt + n * 256 + lane * 8) = v[n];
    } else if (k == 2) {
        u64 v[3] = {0, 0, 0};
        #pragma unroll
        for (int p = 0; p < 8; ++p) trw_pack<2>(v, p, trw_tile_chunk<2>(nt + p * 64, lane));
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < 2; ++n) *reinterpret_cast<u64*>(nt + n * 256 + lane * 8) = v[n];
    } else if (k == 4) {
        u32 w[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) w[p] = *reinterpret_cast<const u32*>(nt + p * 128 + lane * 4);
        __syncwarp();
        *reinterpret_cast<uint4*>(nt + lane * 16)       = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(nt + 512 + lane * 16) = make_uint4(w[4], w[5], w[6], w[7]);
    }
}

// =================================================================================================
// Core PMCGX Wide-Order Tile MMA Helpers (up to 8 tokens per group in x_rot[8][136] FP16!)
// =================================================================================================
template <int BITS>
__device__ __forceinline__ void hy4_pmcgx_fc1_hb_wide(
    const uint8_t* mw, int hb, int warp, int lane,
    const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* grp = mw + static_cast<size_t>(hb) * (2048 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = *reinterpret_cast<const uint4*>(grp + warp * 1024 + lane * 16);
        const uint4 w1 = *reinterpret_cast<const uint4*>(grp + warp * 1024 + 512 + lane * 16);
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    } else {
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = rd64_u8(grp + (n * 8 + warp) * 256 + lane * 8);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

template <int BITS>
__device__ __forceinline__ void hy4_pmcgx_fc2_ntile_wide(
    const uint8_t* mw, int n_tile, int lane,
    const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* nt = mw + static_cast<size_t>(n_tile) * (256 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = *reinterpret_cast<const uint4*>(nt + lane * 16);
        const uint4 w1 = *reinterpret_cast<const uint4*>(nt + 512 + lane * 16);
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    } else {
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = rd64_u8(nt + n * 256 + lane * 8);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

// Double-B-fragment (up to 16 routes/tokens per weight decode!) PMCGX wide helpers for prefill (M >= 16).
template <int BITS>
__device__ __forceinline__ void hy4_pmcgx_fc1_hb_wide16(
    const uint8_t* mw, int hb, int warp, int lane,
    const __half (&x_rot)[16][136], int cnt, float (&c0)[4], float (&c1)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act0 = (row0 < cnt);
    const bool act1 = (row0 + 8 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* grp = mw + static_cast<size_t>(hb) * (2048 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = *reinterpret_cast<const uint4*>(grp + warp * 1024 + lane * 16);
        const uint4 w1 = *reinterpret_cast<const uint4*>(grp + warp * 1024 + 512 + lane * 16);
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act0 ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act0 ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c0, f0[0], f1[0], f0[1], f1[1], b0, b1);
            if (cnt > 8) {
                const u32 b2 = act1 ? *reinterpret_cast<const u32*>(x_rot[row0 + 8] + p * 16 + 2 * col) : 0u;
                const u32 b3 = act1 ? *reinterpret_cast<const u32*>(x_rot[row0 + 8] + p * 16 + 2 * col + 8) : 0u;
                mma16816_f16(c1, f0[0], f1[0], f0[1], f1[1], b2, b3);
            }
        }
    } else {
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = rd64_u8(grp + (n * 8 + warp) * 256 + lane * 8);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act0 ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act0 ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c0, f0[0], f1[0], f0[1], f1[1], b0, b1);
            if (cnt > 8) {
                const u32 b2 = act1 ? *reinterpret_cast<const u32*>(x_rot[row0 + 8] + p * 16 + 2 * col) : 0u;
                const u32 b3 = act1 ? *reinterpret_cast<const u32*>(x_rot[row0 + 8] + p * 16 + 2 * col + 8) : 0u;
                mma16816_f16(c1, f0[0], f1[0], f0[1], f1[1], b2, b3);
            }
        }
    }
}

template <int BITS>
__device__ __forceinline__ void hy4_pmcgx_fc2_ntile_wide16(
    const uint8_t* mw, int n_tile, int lane,
    const __half (&x2)[16][136], int cnt, float (&c0)[4], float (&c1)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act0 = (row0 < cnt);
    const bool act1 = (row0 + 8 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* nt = mw + static_cast<size_t>(n_tile) * (256 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = *reinterpret_cast<const uint4*>(nt + lane * 16);
        const uint4 w1 = *reinterpret_cast<const uint4*>(nt + 512 + lane * 16);
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act0 ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act0 ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c0, f0[0], f1[0], f0[1], f1[1], b0, b1);
            if (cnt > 8) {
                const u32 b2 = act1 ? *reinterpret_cast<const u32*>(x2[row0 + 8] + p * 16 + 2 * col) : 0u;
                const u32 b3 = act1 ? *reinterpret_cast<const u32*>(x2[row0 + 8] + p * 16 + 2 * col + 8) : 0u;
                mma16816_f16(c1, f0[0], f1[0], f0[1], f1[1], b2, b3);
            }
        }
    } else {
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = rd64_u8(nt + n * 256 + lane * 8);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            pmcgx_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act0 ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act0 ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c0, f0[0], f1[0], f0[1], f1[1], b0, b1);
            if (cnt > 8) {
                const u32 b2 = act1 ? *reinterpret_cast<const u32*>(x2[row0 + 8] + p * 16 + 2 * col) : 0u;
                const u32 b3 = act1 ? *reinterpret_cast<const u32*>(x2[row0 + 8] + p * 16 + 2 * col + 8) : 0u;
                mma16816_f16(c1, f0[0], f1[0], f0[1], f1[1], b2, b3);
            }
        }
    }
}

// =================================================================================================
// Pre-rotate input x [M, H] BF16 -> x_had [M, HB, 128] FP16 = H128(x * sg_in).
// Grid: (M * HB), Block: 32.
// =================================================================================================
template <int HB>
__global__ void __launch_bounds__(32) k_hy4_prerot_x(
    const __nv_bfloat16* __restrict__ x,
    const __half* __restrict__ sg_in,
    __half* __restrict__ x_had, int M) {
    const int idx = blockIdx.x;
    if (idx >= M * HB) return;
    const int m = idx / HB, hb = idx % HB;
    const int lane = threadIdx.x;
    const __nv_bfloat16* xm = x + (static_cast<size_t>(m) * HB + hb) * 128;
    const __half* sg = sg_in + hb * 128;
    float v[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int c = lane * 4 + e;
        v[e] = __bfloat162float(xm[c]) * __half2float(sg[c]);
    }
    had128(v, lane);
    __half* dst = x_had + (static_cast<size_t>(m) * HB + hb) * 128;
    #pragma unroll
    for (int e = 0; e < 4; ++e) dst[lane * 4 + e] = __float2half_rn(v[e]);
}

// =================================================================================================
// Layer 0 Dense MLP (I = 13824, TP-sharded, NO swiglu clamp)
// =================================================================================================
struct DenseMlpWeights {
    mx8::LinW            gate_proj{};            // [dense_i_loc, 3072]
    mx8::LinW            up_proj{};              // [dense_i_loc, 3072]
    mx8::LinW            down_proj{};            // [3072, dense_i_loc]
    const __nv_bfloat16* post_fc2_ln  = nullptr; // [3072]
    int dense_i_loc = 0;
};

struct DenseMlpScratch {
    __nv_bfloat16* gate = nullptr; // [max_M, dense_i_loc]
    __nv_bfloat16* up   = nullptr; // [max_M, dense_i_loc]
};

__global__ void k_hy4_silu_mul_bf16(__nv_bfloat16* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ up, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        gate[i] = __float2bfloat16(silu_f(g) * u);
    }
}

inline void dense_mlp_forward(cublasHandle_t blas, const DenseMlpWeights& w, DenseMlpScratch& sc,
                              const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
                              int M, uint8_t* x8_scratch, cudaStream_t st) {
    const uint8_t* x_x8 = nullptr;
    if (M >= 16 && w.gate_proj.is_mx8() && x8_scratch != nullptr) {
        mx8::bf16_to_mx8_a(x_norm, x8_scratch, M, geom::HID, st);
        x_x8 = x8_scratch;
    }
    mx8::linear_w(blas, x_norm, w.gate_proj, sc.gate, M, w.dense_i_loc, geom::HID, x8_scratch, x_x8, st);
    mx8::linear_w(blas, x_norm, w.up_proj,   sc.up,   M, w.dense_i_loc, geom::HID, x8_scratch, x_x8, st);
    const size_t n = static_cast<size_t>(M) * w.dense_i_loc;
    k_hy4_silu_mul_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(sc.gate, sc.up, n);
    mx8::linear_w(blas, sc.gate, w.down_proj, out_part, M, geom::HID, w.dense_i_loc, x8_scratch, nullptr, st);
}

// =================================================================================================
// Layers 1..41 MoE (Router + Latent PMCGX Routed Experts + PMCGX Shared Expert)
// =================================================================================================
struct Tr3Pack {
    const uint8_t*   fc1  = nullptr;
    const uint8_t*   fc2  = nullptr;
    const long long* tab  = nullptr; // [ne, 6]
    const long long* kmix = nullptr; // [ne]
    const __half*    sg   = nullptr; // [2, H] (in, out signs)
    const __half*    v1b  = nullptr; // [nblk, 2, HB]
    const __half*    v2   = nullptr; // [nblk, 3, 128]
    const __half*    v3b  = nullptr; // [nblk, HB]
    int ne = 0;
    int eb = 0;
};

struct MoeWeights {
    const __nv_bfloat16* router_w       = nullptr; // [512, 3072] BF16 (kept exact FP32-accum)
    const float*         router_bias    = nullptr; // [512] FP32 (e_score_correction_bias)
    mx8::LinW            routed_down{};            // [1536, 3072]
    mx8::LinW            routed_up{};              // [3072, 1536]
    const __nv_bfloat16* combine_ln     = nullptr; // [3072] BF16
    Tr3Pack routed;                                // ne=512, HB=12, eb=12/nr
    Tr3Pack shared;                                // ne=1,   HB=24, eb=12/nr
};

struct MoeScratch {
    float*         router_logits = nullptr; // [max_M, 512] FP32
    int*           topk_idx      = nullptr; // [max_M, 16] INT32
    float*         topk_w        = nullptr; // [max_M, 16] FP32
    __nv_bfloat16* routed_in     = nullptr; // [max_M, 1536] BF16
    __nv_bfloat16* routed_lat    = nullptr; // [max_M, 1536] BF16
    __nv_bfloat16* routed_up_out = nullptr; // [max_M, 3072] BF16
    __nv_bfloat16* shared_out    = nullptr; // [max_M, 3072] BF16

    // Grouped routing buffers (groups of <= 8 routes sharing the same expert!)
    int*    d_n_groups = nullptr; // [1]
    int*    grp_exp    = nullptr; // [max_M * 16]
    int*    grp_cnt    = nullptr; // [max_M * 16]
    int*    grp_routes = nullptr; // [max_M * 16 * 8]
    __half* x_had_r    = nullptr; // [max_M, 12, 128] FP16 (routed latent)
    __half* x_had_s    = nullptr; // [max_M, 24, 128] FP16 (shared full H=3072)
    __half* fc1_act_r  = nullptr; // [max_M * 16, max_eb, 128] FP16
    __half* fc1_act_s  = nullptr; // [max_M, max_eb, 128] FP16
    float*  fc2_part_r = nullptr; // [max_M * 16, 1536] FP32 (deterministic per-route sum!)
};

// Exact HYV4FlashTopkRouter: sigmoid(FP32) + e_score_correction_bias (FP32) -> top-16 -> normalize * 3.998!
// Grid: (M), Block: 256 (each thread owns 2 experts: tid and tid + 256).
__global__ void __launch_bounds__(256) k_hy4_router_topk16(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const float* row = logits + static_cast<size_t>(m) * geom::NEXP;

    __shared__ float s_sc[geom::NEXP];
    __shared__ float s_ch[geom::NEXP];
    for (int k = 0; k < 2; ++k) {
        const int e = tid + k * 256;
        const float sc = 1.0f / (1.0f + expf(-row[e]));
        s_sc[e] = sc;
        s_ch[e] = sc + bias[e];
    }
    __syncthreads();

    __shared__ int   sel_e[geom::TOPK];
    __shared__ float sel_w[geom::TOPK];
    __shared__ float w_val[8];
    __shared__ int   w_idx[8];
    const int warp = tid >> 5, lane = tid & 31;

    for (int k = 0; k < geom::TOPK; ++k) {
        float bv = s_ch[tid];
        int   bi = tid;
        const float v1 = s_ch[tid + 256];
        if (v1 > bv) { bv = v1; bi = tid + 256; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { w_val[warp] = bv; w_idx[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float mv = w_val[0]; int mi = w_idx[0];
            for (int w = 1; w < 8; ++w) {
                if (w_val[w] > mv || (w_val[w] == mv && w_idx[w] < mi)) { mv = w_val[w]; mi = w_idx[w]; }
            }
            sel_e[k] = mi;
            sel_w[k] = s_sc[mi]; // pre-bias sigmoid score!
            s_ch[mi] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid < geom::TOPK) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < geom::TOPK; ++k) sum += sel_w[k];
        const float inv = geom::ROUTED_SCALE / (sum + 1e-20f);
        topk_idx[static_cast<size_t>(m) * geom::TOPK + tid] = sel_e[tid];
        topk_w[static_cast<size_t>(m) * geom::TOPK + tid]   = sel_w[tid] * inv;
    }
}

// Fast M=1 group builder (16 single-route groups, 1 warp!).
__global__ void k_hy4_moe_groups_m1(const int* __restrict__ topk_idx,
                                    int* __restrict__ d_n_groups,
                                    int* __restrict__ grp_exp,
                                    int* __restrict__ grp_cnt,
                                    int* __restrict__ grp_routes) {
    const int t = threadIdx.x;
    if (t < geom::TOPK) {
        grp_exp[t] = topk_idx[t];
        grp_cnt[t] = 1;
        grp_routes[t * 8] = t;
    }
    if (t == 0) d_n_groups[0] = geom::TOPK;
}

// Parallel 2-pass shared-memory group builder for arbitrary M >= 2 (512x fewer global loads than serial scan!).
// Splits each hit expert into chunks of <= GMAX routes (GMAX=8 for small M, GMAX=16 for prefill).
// Grid: 1, Block: 512.
template <int GMAX>
__global__ void __launch_bounds__(512) k_hy4_moe_build_groups(
    const int* __restrict__ topk_idx,
    int* __restrict__ d_n_groups,
    int* __restrict__ grp_exp,
    int* __restrict__ grp_cnt,
    int* __restrict__ grp_routes,
    int n_routes) {
    const int tid = threadIdx.x; // 0..511
    __shared__ int s_cnt[512];
    __shared__ int s_base[512];
    __shared__ int s_cur[512];
    __shared__ int s_wsum[16];

    s_cnt[tid] = 0;
    s_cur[tid] = 0;
    __syncthreads();

    for (int r = tid; r < n_routes; r += 512) {
        const int e = topk_idx[r];
        atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();

    const int cnt = s_cnt[tid];
    const int ng = (cnt + GMAX - 1) / GMAX;
    // Intra-warp inclusive scan
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int v = ng;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int prev = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += prev;
    }
    if (lane == 31) s_wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int ws = (lane < 16) ? s_wsum[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            const int prev = __shfl_up_sync(0xffffffffu, ws, off);
            if (lane >= off) ws += prev;
        }
        if (lane < 16) s_wsum[lane] = ws;
        if (lane == 15) d_n_groups[0] = ws;
    }
    __syncthreads();
    const int w_off = (warp > 0) ? s_wsum[warp - 1] : 0;
    s_base[tid] = w_off + v - ng; // exclusive prefix sum
    __syncthreads();

    for (int r = tid; r < n_routes; r += 512) {
        const int e = topk_idx[r];
        const int pos = atomicAdd(&s_cur[e], 1);
        const int g = s_base[e] + (pos / GMAX);
        const int sub = pos % GMAX;
        if (sub == 0) {
            grp_exp[g] = e;
            grp_cnt[g] = min(GMAX, s_cnt[e] - (pos - sub));
        }
        grp_routes[g * GMAX + sub] = r;
    }
}

// Grouped PMCGX Routed FC1 (HB=12, SWIGLU_CLAMP=10.0):
// Computes gate + up -> had128 * svh -> clamp & silu(g)*u -> * suh_d -> had128 -> fc1_act_r!
// Grid: (grid_G * eb), Block: 256 (8 warps = 8 n-tiles = 128 output cols).
__global__ void __launch_bounds__(256) k_hy4_routed_fc1_grp(
    const __half* __restrict__ x_had,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1b,
    const __half* __restrict__ v2,
    __half* __restrict__ fc1_act,
    int eb) {
    const int b = blockIdx.x % eb;
    const int g = blockIdx.x / eb;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int cnt = grp_cnt[g];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ int s_tok[8], s_route[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_tok[tid] = r / geom::TOPK;
    }
    __syncthreads();

    const uint8_t* mw_g = nullptr;
    const uint8_t* mw_u = nullptr;
    const int kg = hy4_fc1_unit<geom::EHB>(fc1, kmix, e, tr, 2 * b + 0, mw_g);
    const int ku = hy4_fc1_unit<geom::EHB>(fc1, kmix, e, tr, 2 * b + 1, mw_u);

    __shared__ __half x_rot[8][136];
    float cg[4] = {0, 0, 0, 0};
    float cu[4] = {0, 0, 0, 0};

    // Gate pass over HB = 12
    for (int hb = 0; hb < geom::EHB; ++hb) {
        if (warp < cnt) {
            const __half2 bsh = __half2half2(v1b[(brow * 2 + 0) * geom::EHB + hb]);
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had + (static_cast<size_t>(s_tok[warp]) * geom::EHB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[warp] + lane * 4) = ov;
        }
        __syncthreads();
        if (kg == 3)      hy4_pmcgx_fc1_hb_wide<3>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        else if (kg == 4) hy4_pmcgx_fc1_hb_wide<4>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        else              hy4_pmcgx_fc1_hb_wide<2>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        __syncthreads();
    }
    // Up pass over HB = 12
    for (int hb = 0; hb < geom::EHB; ++hb) {
        if (warp < cnt) {
            const __half2 bsh = __half2half2(v1b[(brow * 2 + 1) * geom::EHB + hb]);
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had + (static_cast<size_t>(s_tok[warp]) * geom::EHB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[warp] + lane * 4) = ov;
        }
        __syncthreads();
        if (ku == 3)      hy4_pmcgx_fc1_hb_wide<3>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        else if (ku == 4) hy4_pmcgx_fc1_hb_wide<4>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        else              hy4_pmcgx_fc1_hb_wide<2>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        __syncthreads();
    }

    // Store cg/cu into smem [8][128], then each warp (0..cnt-1) runs had128 -> svh -> clamped SwiGLU -> suh_d -> had128!
    __shared__ float s_g[8][128];
    __shared__ float s_u[8][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        s_g[t0][warp * 16 + row0]     = cg[0];
        s_g[t0][warp * 16 + row0 + 8] = cg[2];
        s_u[t0][warp * 16 + row0]     = cu[0];
        s_u[t0][warp * 16 + row0 + 8] = cu[2];
    }
    if (t1 < cnt) {
        s_g[t1][warp * 16 + row0]     = cg[1];
        s_g[t1][warp * 16 + row0 + 8] = cg[3];
        s_u[t1][warp * 16 + row0]     = cu[1];
        s_u[t1][warp * 16 + row0 + 8] = cu[3];
    }
    __syncthreads();

    if (warp < cnt) {
        float vg[4], vu[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = s_g[warp][lane * 4 + i];
            vu[i] = s_u[warp][lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        const __half* sv_g = v2 + (brow * 3 + 0) * 128;
        const __half* sv_u = v2 + (brow * 3 + 1) * 128;
        const __half* su_d = v2 + (brow * 3 + 2) * 128;
        float va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = lane * 4 + i;
            float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            // HYV4FlashExperts._apply_gate: clamp gate <= 10.0, up in [-10.0, 10.0]
            gf = fminf(gf, geom::SWIGLU_LIMIT);
            uf = fmaxf(-geom::SWIGLU_LIMIT, fminf(geom::SWIGLU_LIMIT, uf));
            const float act = __bfloat162float(__float2bfloat16(silu_f(gf) * uf));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

// Grouped PMCGX Routed FC2 (HB=12): sums over b=0..nb-1 per route -> writes fc2_part_r[route, 1536] FP32!
// Grid: (grid_G * EHB), Block: 256 (8 warps = 8 n-tiles = 128 output cols).
__global__ void __launch_bounds__(256) k_hy4_routed_fc2_grp(
    const __half* __restrict__ fc1_act,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ sg,   // [2, 1536]
    const __half* __restrict__ v3b,  // [nblk, 12]
    float* __restrict__ fc2_part,
    int eb) {
    const int hb = blockIdx.x % geom::EHB;
    const int g  = blockIdx.x / geom::EHB;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int cnt = grp_cnt[g];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ int s_route[8];
    if (tid < cnt) s_route[tid] = grp_routes[g * 8 + tid];
    __syncthreads();

    __shared__ __half x2[8][136];
    float acc[4] = {0, 0, 0, 0};
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(geom::EHB) * 2048;
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        if (warp < cnt) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[warp] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * bits;
        float c[4] = {0, 0, 0, 0};
        if (bits == 3)      hy4_pmcgx_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 4) hy4_pmcgx_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else                hy4_pmcgx_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
        const float sc = __half2float(v3b[brow * geom::EHB + hb]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(c[i], sc, acc[i]);
        __syncthreads();
    }

    __shared__ float s_out[8][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) { s_out[t0][warp * 16 + row0] = acc[0]; s_out[t0][warp * 16 + row0 + 8] = acc[2]; }
    if (t1 < cnt) { s_out[t1][warp * 16 + row0] = acc[1]; s_out[t1][warp * 16 + row0 + 8] = acc[3]; }
    __syncthreads();

    if (warp < cnt) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = s_out[warp][lane * 4 + i];
        had128(v, lane);
        const int r = s_route[warp];
        const float rw = topk_w[r];
        const __half* sg_out = sg + geom::EH + hb * 128;
        float* dst = fc2_part + static_cast<size_t>(r) * geom::EH + hb * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v[i] * __half2float(sg_out[lane * 4 + i]) * rw;
    }
}

// Deterministic sum over all 16 routes -> routed_lat [M, 1536] BF16!
// Grid: (M * 6), Block: 256 (6 * 256 = 1536 exact).
__global__ void __launch_bounds__(256) k_hy4_routed_reduce(
    const float* __restrict__ fc2_part,
    __nv_bfloat16* __restrict__ routed_lat, int M) {
    const int m = blockIdx.x / 6;
    const int d = (blockIdx.x % 6) * 256 + threadIdx.x;
    if (m >= M || d >= geom::EH) return;
    float acc = 0.0f;
    for (int k = 0; k < geom::TOPK; ++k) {
        const int r = m * geom::TOPK + k;
        acc += fc2_part[static_cast<size_t>(r) * geom::EH + d];
    }
    routed_lat[static_cast<size_t>(m) * geom::EH + d] = __float2bfloat16(acc);
}

// =================================================================================================
// PMCGX Shared Expert (H=3072, HB=24, eb=12/nr, K3 or K4, NO swiglu clamp!)
// Processes tokens in chunks of 8!
// =================================================================================================
__global__ void __launch_bounds__(256) k_hy4_shared_fc1(
    const __half* __restrict__ x_had_s,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1b,
    const __half* __restrict__ v2,
    __half* __restrict__ fc1_act_s,
    int M, int eb) {
    const int b   = blockIdx.x % eb;
    const int m8  = blockIdx.x / eb;
    const int m0  = m8 * 8;
    const int cnt = min(8, M - m0);
    if (cnt <= 0) return;

    const long long* tr = tab;
    if (b >= static_cast<int>(tr[3])) return;
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const uint8_t* mw_g = nullptr;
    const uint8_t* mw_u = nullptr;
    const int kg = hy4_fc1_unit<geom::HB>(fc1, kmix, 0, tr, 2 * b + 0, mw_g);
    const int ku = hy4_fc1_unit<geom::HB>(fc1, kmix, 0, tr, 2 * b + 1, mw_u);

    __shared__ __half x_rot[8][136];
    float cg[4] = {0, 0, 0, 0};
    float cu[4] = {0, 0, 0, 0};

    for (int hb = 0; hb < geom::HB; ++hb) {
        if (warp < cnt) {
            const __half2 bsh = __half2half2(v1b[(brow * 2 + 0) * geom::HB + hb]);
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + warp) * geom::HB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[warp] + lane * 4) = ov;
        }
        __syncthreads();
        if (kg == 4)      hy4_pmcgx_fc1_hb_wide<4>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        else if (kg == 3) hy4_pmcgx_fc1_hb_wide<3>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        else              hy4_pmcgx_fc1_hb_wide<2>(mw_g, hb, warp, lane, x_rot, cnt, cg);
        __syncthreads();
    }
    for (int hb = 0; hb < geom::HB; ++hb) {
        if (warp < cnt) {
            const __half2 bsh = __half2half2(v1b[(brow * 2 + 1) * geom::HB + hb]);
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + warp) * geom::HB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[warp] + lane * 4) = ov;
        }
        __syncthreads();
        if (ku == 4)      hy4_pmcgx_fc1_hb_wide<4>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        else if (ku == 3) hy4_pmcgx_fc1_hb_wide<3>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        else              hy4_pmcgx_fc1_hb_wide<2>(mw_u, hb, warp, lane, x_rot, cnt, cu);
        __syncthreads();
    }

    __shared__ float s_g[8][128];
    __shared__ float s_u[8][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        s_g[t0][warp * 16 + row0] = cg[0]; s_g[t0][warp * 16 + row0 + 8] = cg[2];
        s_u[t0][warp * 16 + row0] = cu[0]; s_u[t0][warp * 16 + row0 + 8] = cu[2];
    }
    if (t1 < cnt) {
        s_g[t1][warp * 16 + row0] = cg[1]; s_g[t1][warp * 16 + row0 + 8] = cg[3];
        s_u[t1][warp * 16 + row0] = cu[1]; s_u[t1][warp * 16 + row0 + 8] = cu[3];
    }
    __syncthreads();

    if (warp < cnt) {
        float vg[4], vu[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { vg[i] = s_g[warp][lane * 4 + i]; vu[i] = s_u[warp][lane * 4 + i]; }
        had128(vg, lane);
        had128(vu, lane);
        const __half* sv_g = v2 + (brow * 3 + 0) * 128;
        const __half* sv_u = v2 + (brow * 3 + 1) * 128;
        const __half* su_d = v2 + (brow * 3 + 2) * 128;
        float va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = lane * 4 + i;
            // Shared expert is HYV4FlashMLP: NO swiglu_limit clamp!
            const float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            const float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            const float act = __bfloat162float(__float2bfloat16(silu_f(gf) * uf));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act_s + (static_cast<size_t>(m0 + warp) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_hy4_shared_fc2(
    const __half* __restrict__ fc1_act_s,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ sg,   // [2, 3072]
    const __half* __restrict__ v3b,  // [nblk, 24]
    __nv_bfloat16* __restrict__ shared_out,
    int M, int eb) {
    const int hb  = blockIdx.x % geom::HB;
    const int m8  = blockIdx.x / geom::HB;
    const int m0  = m8 * 8;
    const int cnt = min(8, M - m0);
    if (cnt <= 0) return;

    const long long* tr = tab;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ __half x2[8][136];
    float acc[4] = {0, 0, 0, 0};
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(geom::HB) * 2048;
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        if (warp < cnt) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act_s + (static_cast<size_t>(m0 + warp) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[warp] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * bits;
        float c[4] = {0, 0, 0, 0};
        if (bits == 4)      hy4_pmcgx_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 3) hy4_pmcgx_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else                hy4_pmcgx_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
        const float sc = __half2float(v3b[brow * geom::HB + hb]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(c[i], sc, acc[i]);
        __syncthreads();
    }

    __shared__ float s_out[8][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) { s_out[t0][warp * 16 + row0] = acc[0]; s_out[t0][warp * 16 + row0 + 8] = acc[2]; }
    if (t1 < cnt) { s_out[t1][warp * 16 + row0] = acc[1]; s_out[t1][warp * 16 + row0 + 8] = acc[3]; }
    __syncthreads();

    if (warp < cnt) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = s_out[warp][lane * 4 + i];
        had128(v, lane);
        const __half* sg_out = sg + geom::HID + hb * 128;
        __nv_bfloat16* dst = shared_out + static_cast<size_t>(m0 + warp) * geom::HID + hb * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2bfloat16(v[i] * __half2float(sg_out[lane * 4 + i]));
    }
}

// 16-route double-B-fragment PMCGX Routed FC1 for prefill (M >= 16): halves weight reads & decodes!
__global__ void __launch_bounds__(256) k_hy4_routed_fc1_grp16(
    const __half* __restrict__ x_had,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1b,
    const __half* __restrict__ v2,
    __half* __restrict__ fc1_act,
    int eb) {
    const int b = blockIdx.x % eb;
    const int g = blockIdx.x / eb;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int cnt = grp_cnt[g];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ int s_tok[16], s_route[16];
    if (tid < cnt) {
        const int r = grp_routes[g * 16 + tid];
        s_route[tid] = r;
        s_tok[tid] = r / geom::TOPK;
    }
    __syncthreads();

    const uint8_t* mw_g = nullptr;
    const uint8_t* mw_u = nullptr;
    const int kg = hy4_fc1_unit<geom::EHB>(fc1, kmix, e, tr, 2 * b + 0, mw_g);
    const int ku = hy4_fc1_unit<geom::EHB>(fc1, kmix, e, tr, 2 * b + 1, mw_u);

    __shared__ __half x_rot[16][136];
    float cg0[4] = {0, 0, 0, 0}, cg1[4] = {0, 0, 0, 0};
    float cu0[4] = {0, 0, 0, 0}, cu1[4] = {0, 0, 0, 0};

    for (int hb = 0; hb < geom::EHB; ++hb) {
        const __half2 bsh = __half2half2(v1b[(brow * 2 + 0) * geom::EHB + hb]);
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had + (static_cast<size_t>(s_tok[w]) * geom::EHB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[w] + lane * 4) = ov;
        }
        __syncthreads();
        if (kg == 3)      hy4_pmcgx_fc1_hb_wide16<3>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else if (kg == 4) hy4_pmcgx_fc1_hb_wide16<4>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else              hy4_pmcgx_fc1_hb_wide16<2>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        __syncthreads();
    }
    for (int hb = 0; hb < geom::EHB; ++hb) {
        const __half2 bsh = __half2half2(v1b[(brow * 2 + 1) * geom::EHB + hb]);
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had + (static_cast<size_t>(s_tok[w]) * geom::EHB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[w] + lane * 4) = ov;
        }
        __syncthreads();
        if (ku == 3)      hy4_pmcgx_fc1_hb_wide16<3>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else if (ku == 4) hy4_pmcgx_fc1_hb_wide16<4>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else              hy4_pmcgx_fc1_hb_wide16<2>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        __syncthreads();
    }

    __shared__ float s_g[16][128];
    __shared__ float s_u[16][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        s_g[t0][warp * 16 + row0] = cg0[0]; s_g[t0][warp * 16 + row0 + 8] = cg0[2];
        s_u[t0][warp * 16 + row0] = cu0[0]; s_u[t0][warp * 16 + row0 + 8] = cu0[2];
    }
    if (t1 < cnt) {
        s_g[t1][warp * 16 + row0] = cg0[1]; s_g[t1][warp * 16 + row0 + 8] = cg0[3];
        s_u[t1][warp * 16 + row0] = cu0[1]; s_u[t1][warp * 16 + row0 + 8] = cu0[3];
    }
    if (t0 + 8 < cnt) {
        s_g[t0 + 8][warp * 16 + row0] = cg1[0]; s_g[t0 + 8][warp * 16 + row0 + 8] = cg1[2];
        s_u[t0 + 8][warp * 16 + row0] = cu1[0]; s_u[t0 + 8][warp * 16 + row0 + 8] = cu1[2];
    }
    if (t1 + 8 < cnt) {
        s_g[t1 + 8][warp * 16 + row0] = cg1[1]; s_g[t1 + 8][warp * 16 + row0 + 8] = cg1[3];
        s_u[t1 + 8][warp * 16 + row0] = cu1[1]; s_u[t1 + 8][warp * 16 + row0 + 8] = cu1[3];
    }
    __syncthreads();

    const __half* sv_g = v2 + (brow * 3 + 0) * 128;
    const __half* sv_u = v2 + (brow * 3 + 1) * 128;
    const __half* su_d = v2 + (brow * 3 + 2) * 128;
    for (int w = warp; w < cnt; w += 8) {
        float vg[4], vu[4], va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { vg[i] = s_g[w][lane * 4 + i]; vu[i] = s_u[w][lane * 4 + i]; }
        had128(vg, lane);
        had128(vu, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = lane * 4 + i;
            float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            gf = fminf(gf, geom::SWIGLU_LIMIT);
            uf = fmaxf(-geom::SWIGLU_LIMIT, fminf(geom::SWIGLU_LIMIT, uf));
            const float act = __bfloat162float(__float2bfloat16(silu_f(gf) * uf));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(s_route[w]) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_hy4_routed_fc2_grp16(
    const __half* __restrict__ fc1_act,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ sg,
    const __half* __restrict__ v3b,
    float* __restrict__ fc2_part,
    int eb) {
    const int hb = blockIdx.x % geom::EHB;
    const int g  = blockIdx.x / geom::EHB;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int cnt = grp_cnt[g];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ int s_route[16];
    if (tid < cnt) s_route[tid] = grp_routes[g * 16 + tid];
    __syncthreads();

    __shared__ __half x2[16][136];
    float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0};
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(geom::EHB) * 2048;
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act + (static_cast<size_t>(s_route[w]) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[w] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * bits;
        float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
        if (bits == 3)      hy4_pmcgx_fc2_ntile_wide16<3>(mw, n_tile, lane, x2, cnt, c0, c1);
        else if (bits == 4) hy4_pmcgx_fc2_ntile_wide16<4>(mw, n_tile, lane, x2, cnt, c0, c1);
        else                hy4_pmcgx_fc2_ntile_wide16<2>(mw, n_tile, lane, x2, cnt, c0, c1);
        const float sc = __half2float(v3b[brow * geom::EHB + hb]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc0[i] = fmaf(c0[i], sc, acc0[i]);
            acc1[i] = fmaf(c1[i], sc, acc1[i]);
        }
        __syncthreads();
    }

    __shared__ float s_out[16][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt)     { s_out[t0][warp * 16 + row0] = acc0[0];     s_out[t0][warp * 16 + row0 + 8] = acc0[2]; }
    if (t1 < cnt)     { s_out[t1][warp * 16 + row0] = acc0[1];     s_out[t1][warp * 16 + row0 + 8] = acc0[3]; }
    if (t0 + 8 < cnt) { s_out[t0 + 8][warp * 16 + row0] = acc1[0]; s_out[t0 + 8][warp * 16 + row0 + 8] = acc1[2]; }
    if (t1 + 8 < cnt) { s_out[t1 + 8][warp * 16 + row0] = acc1[1]; s_out[t1 + 8][warp * 16 + row0 + 8] = acc1[3]; }
    __syncthreads();

    const __half* sg_out = sg + geom::EH + hb * 128;
    for (int w = warp; w < cnt; w += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = s_out[w][lane * 4 + i];
        had128(v, lane);
        const int r = s_route[w];
        const float rw = topk_w[r];
        float* dst = fc2_part + static_cast<size_t>(r) * geom::EH + hb * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v[i] * __half2float(sg_out[lane * 4 + i]) * rw;
    }
}

__global__ void __launch_bounds__(256) k_hy4_shared_fc1_16(
    const __half* __restrict__ x_had_s,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1b,
    const __half* __restrict__ v2,
    __half* __restrict__ fc1_act_s,
    int M, int eb) {
    const int b   = blockIdx.x % eb;
    const int m16 = blockIdx.x / eb;
    const int m0  = m16 * 16;
    const int cnt = min(16, M - m0);
    if (cnt <= 0) return;

    const long long* tr = tab;
    if (b >= static_cast<int>(tr[3])) return;
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const uint8_t* mw_g = nullptr;
    const uint8_t* mw_u = nullptr;
    const int kg = hy4_fc1_unit<geom::HB>(fc1, kmix, 0, tr, 2 * b + 0, mw_g);
    const int ku = hy4_fc1_unit<geom::HB>(fc1, kmix, 0, tr, 2 * b + 1, mw_u);

    __shared__ __half x_rot[16][136];
    float cg0[4] = {0, 0, 0, 0}, cg1[4] = {0, 0, 0, 0};
    float cu0[4] = {0, 0, 0, 0}, cu1[4] = {0, 0, 0, 0};

    for (int hb = 0; hb < geom::HB; ++hb) {
        const __half2 bsh = __half2half2(v1b[(brow * 2 + 0) * geom::HB + hb]);
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + w) * geom::HB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[w] + lane * 4) = ov;
        }
        __syncthreads();
        if (kg == 4)      hy4_pmcgx_fc1_hb_wide16<4>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else if (kg == 3) hy4_pmcgx_fc1_hb_wide16<3>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else              hy4_pmcgx_fc1_hb_wide16<2>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        __syncthreads();
    }
    for (int hb = 0; hb < geom::HB; ++hb) {
        const __half2 bsh = __half2half2(v1b[(brow * 2 + 1) * geom::HB + hb]);
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + w) * geom::HB + hb) * 128 + lane * 4);
            uint2 ov;
            *reinterpret_cast<__half2*>(&ov.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsh);
            *reinterpret_cast<__half2*>(&ov.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsh);
            *reinterpret_cast<uint2*>(x_rot[w] + lane * 4) = ov;
        }
        __syncthreads();
        if (ku == 4)      hy4_pmcgx_fc1_hb_wide16<4>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else if (ku == 3) hy4_pmcgx_fc1_hb_wide16<3>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else              hy4_pmcgx_fc1_hb_wide16<2>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        __syncthreads();
    }

    __shared__ float s_g[16][128];
    __shared__ float s_u[16][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt)     { s_g[t0][warp * 16 + row0] = cg0[0]; s_g[t0][warp * 16 + row0 + 8] = cg0[2]; s_u[t0][warp * 16 + row0] = cu0[0]; s_u[t0][warp * 16 + row0 + 8] = cu0[2]; }
    if (t1 < cnt)     { s_g[t1][warp * 16 + row0] = cg0[1]; s_g[t1][warp * 16 + row0 + 8] = cg0[3]; s_u[t1][warp * 16 + row0] = cu0[1]; s_u[t1][warp * 16 + row0 + 8] = cu0[3]; }
    if (t0 + 8 < cnt) { s_g[t0 + 8][warp * 16 + row0] = cg1[0]; s_g[t0 + 8][warp * 16 + row0 + 8] = cg1[2]; s_u[t0 + 8][warp * 16 + row0] = cu1[0]; s_u[t0 + 8][warp * 16 + row0 + 8] = cu1[2]; }
    if (t1 + 8 < cnt) { s_g[t1 + 8][warp * 16 + row0] = cg1[1]; s_g[t1 + 8][warp * 16 + row0 + 8] = cg1[3]; s_u[t1 + 8][warp * 16 + row0] = cu1[1]; s_u[t1 + 8][warp * 16 + row0 + 8] = cu1[3]; }
    __syncthreads();

    const __half* sv_g = v2 + (brow * 3 + 0) * 128;
    const __half* sv_u = v2 + (brow * 3 + 1) * 128;
    const __half* su_d = v2 + (brow * 3 + 2) * 128;
    for (int w = warp; w < cnt; w += 8) {
        float vg[4], vu[4], va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { vg[i] = s_g[w][lane * 4 + i]; vu[i] = s_u[w][lane * 4 + i]; }
        had128(vg, lane);
        had128(vu, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = lane * 4 + i;
            const float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            const float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            const float act = __bfloat162float(__float2bfloat16(silu_f(gf) * uf));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act_s + (static_cast<size_t>(m0 + w) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_hy4_shared_fc2_16(
    const __half* __restrict__ fc1_act_s,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ sg,
    const __half* __restrict__ v3b,
    __nv_bfloat16* __restrict__ shared_out,
    int M, int eb) {
    const int hb  = blockIdx.x % geom::HB;
    const int m16 = blockIdx.x / geom::HB;
    const int m0  = m16 * 16;
    const int cnt = min(16, M - m0);
    if (cnt <= 0) return;

    const long long* tr = tab;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    __shared__ __half x2[16][136];
    float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0};
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(geom::HB) * 2048;
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act_s + (static_cast<size_t>(m0 + w) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[w] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * bits;
        float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
        if (bits == 4)      hy4_pmcgx_fc2_ntile_wide16<4>(mw, n_tile, lane, x2, cnt, c0, c1);
        else if (bits == 3) hy4_pmcgx_fc2_ntile_wide16<3>(mw, n_tile, lane, x2, cnt, c0, c1);
        else                hy4_pmcgx_fc2_ntile_wide16<2>(mw, n_tile, lane, x2, cnt, c0, c1);
        const float sc = __half2float(v3b[brow * geom::HB + hb]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc0[i] = fmaf(c0[i], sc, acc0[i]);
            acc1[i] = fmaf(c1[i], sc, acc1[i]);
        }
        __syncthreads();
    }

    __shared__ float s_out[16][128];
    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt)     { s_out[t0][warp * 16 + row0] = acc0[0];     s_out[t0][warp * 16 + row0 + 8] = acc0[2]; }
    if (t1 < cnt)     { s_out[t1][warp * 16 + row0] = acc0[1];     s_out[t1][warp * 16 + row0 + 8] = acc0[3]; }
    if (t0 + 8 < cnt) { s_out[t0 + 8][warp * 16 + row0] = acc1[0]; s_out[t0 + 8][warp * 16 + row0 + 8] = acc1[2]; }
    if (t1 + 8 < cnt) { s_out[t1 + 8][warp * 16 + row0] = acc1[1]; s_out[t1 + 8][warp * 16 + row0 + 8] = acc1[3]; }
    __syncthreads();

    const __half* sg_out = sg + geom::HID + hb * 128;
    for (int w = warp; w < cnt; w += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = s_out[w][lane * 4 + i];
        had128(v, lane);
        __nv_bfloat16* dst = shared_out + static_cast<size_t>(m0 + w) * geom::HID + hb * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2bfloat16(v[i] * __half2float(sg_out[lane * 4 + i]));
    }
}

__global__ void k_hy4_add_bf16(const __nv_bfloat16* __restrict__ a,
                               const __nv_bfloat16* __restrict__ b,
                               __nv_bfloat16* __restrict__ out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// =================================================================================================
// Full MoE Forward (Layers 1..41) -> out_part [M, 3072] BF16 partial!
// =================================================================================================
inline void moe_forward(cublasHandle_t blas, const MoeWeights& w, MoeScratch& sc,
                        const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
                        int M, uint8_t* x8_scratch, cudaStream_t st, bool skip_add = false) {
    // 1. Router (FP32 logits + sigmoid + bias -> top-16)
    linear_bf16_f32(blas, x_norm, w.router_w, sc.router_logits, M, geom::NEXP, geom::HID, st);
    k_hy4_router_topk16<<<M, 256, 0, st>>>(sc.router_logits, w.router_bias, sc.topk_idx, sc.topk_w, M);

    // 2. Latent down proj + Routed PMCGX experts + Latent up proj
    mx8::linear_w(blas, x_norm, w.routed_down, sc.routed_in, M, geom::EH, geom::HID, x8_scratch, nullptr, st);
    const int n_routes = M * geom::TOPK;
    k_hy4_prerot_x<geom::EHB><<<M * geom::EHB, 32, 0, st>>>(sc.routed_in, w.routed.sg, sc.x_had_r, M);

    if (M == 1) {
        k_hy4_moe_groups_m1<<<1, 32, 0, st>>>(sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes);
        const int grid_G = geom::TOPK;
        k_hy4_routed_fc1_grp<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        k_hy4_routed_fc2_grp<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
    } else if (M < 16) {
        k_hy4_moe_build_groups<8><<<1, geom::NEXP, 0, st>>>(
            sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
        const int grid_G = std::min(n_routes, geom::NEXP + (n_routes + 7) / 8);
        k_hy4_routed_fc1_grp<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        k_hy4_routed_fc2_grp<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
    } else {
        k_hy4_moe_build_groups<16><<<1, geom::NEXP, 0, st>>>(
            sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
        const int grid_G = std::min(n_routes, geom::NEXP + (n_routes + 15) / 16);
        k_hy4_routed_fc1_grp16<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        k_hy4_routed_fc2_grp16<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
    }
    k_hy4_routed_reduce<<<M * 6, 256, 0, st>>>(sc.fc2_part_r, sc.routed_lat, M);
    mx8::linear_w(blas, sc.routed_lat, w.routed_up, sc.routed_up_out, M, geom::HID, geom::EH, x8_scratch, nullptr, st);

    // 3. Shared PMCGX expert (H=3072, HB=24, NO swiglu clamp)
    k_hy4_prerot_x<geom::HB><<<M * geom::HB, 32, 0, st>>>(x_norm, w.shared.sg, sc.x_had_s, M);
    if (M < 16) {
        const int n_m8 = (M + 7) / 8;
        k_hy4_shared_fc1<<<n_m8 * w.shared.eb, 256, 0, st>>>(
            sc.x_had_s, w.shared.tab, w.shared.kmix, w.shared.fc1, w.shared.v1b, w.shared.v2,
            sc.fc1_act_s, M, w.shared.eb);
        k_hy4_shared_fc2<<<n_m8 * geom::HB, 256, 0, st>>>(
            sc.fc1_act_s, w.shared.tab, w.shared.fc2, w.shared.sg, w.shared.v3b, sc.shared_out, M, w.shared.eb);
    } else {
        const int n_m16 = (M + 15) / 16;
        k_hy4_shared_fc1_16<<<n_m16 * w.shared.eb, 256, 0, st>>>(
            sc.x_had_s, w.shared.tab, w.shared.kmix, w.shared.fc1, w.shared.v1b, w.shared.v2,
            sc.fc1_act_s, M, w.shared.eb);
        k_hy4_shared_fc2_16<<<n_m16 * geom::HB, 256, 0, st>>>(
            sc.fc1_act_s, w.shared.tab, w.shared.fc2, w.shared.sg, w.shared.v3b, sc.shared_out, M, w.shared.eb);
    }

    // 4. Combine routed_up_out + shared_out -> out_part [M, 3072] BF16 partial (skipped if caller fuses add+norm+inject!)
    if (!skip_add && out_part != nullptr) {
        const size_t n = static_cast<size_t>(M) * geom::HID;
        k_hy4_add_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(
            sc.routed_up_out, sc.shared_out, out_part, n);
    }
}

} // namespace hy4
