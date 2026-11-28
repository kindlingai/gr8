// ERNIE-5.5-Flash-Preview Dense L0 SiTU MLP + Layers 1..42 Latent MoE (FP32 split-feature router,
// fc1_latent_proj 4096->2048, 512 mul1 0x83DCD12D E4M3 routed experts top-10 with optional fixed-address
// d_ptab[512][EB][2] for hybrid 5090 offload, latent_norm + fc2_latent_proj 2048->4096, and VRAM-resident
// mul1 shared expert 4096x2048).
//
// Exact ERNIE 5.5 math invariants verified against config.json & CPU reference:
//   - Activation: SiTU-GLU everywhere (L0 dense, routed experts, shared expert):
//       situ_mul(g, u) = (4 * tanh(g/4) * sigmoid(g)) * (25 * tanh(u/25))
//   - Split-feature FP32 router:
//       scores = sigmoid(x_f32 @ w0^T) + sigmoid(x_f32 @ w1^T)    (w0, w1 are F32 [512, 4096])
//       choice = scores + bias                                    (bias is F32 [512])
//       ids    = topk(choice, 10)  (stable lowest-index tie-break)
//       wts    = scores[ids] / max(sum(scores[ids]), 1e-20) * scale[ids]   (per-expert F32 scale[512])
//   - Routed experts (EH=2048, EHB=16, EXP_I=2048, EB=16/nr):
//       mul1 codebook (0x83DCD12D) + FP8 E4M3 rounding, K=2/3/4 + per-matrix bit-63 kmix, wide-load order.
//       Supports BOTH resident contiguous fc1/fc2 (d_ptab == nullptr) AND fixed-address d_ptab[512][EB][2]
//       for hybrid VRAM hot-cache + pageable host offload (identical math, zero duplication!).
//   - Shared expert (H=4096, HB=32, SHARED_I=2048, EB=16/nr):
//       mul1 E4M3 wide-load order, always VRAM-resident (~246 MiB across all 42 layers at K4).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include "../model_geom_ernie55.h"
#include "ernie55_mhc.cuh"
#include "ernie55_mxfp8.cuh"

namespace ernie55 {

using u32 = uint32_t;
using u64 = uint64_t;

static constexpr float ERNIE55_SQ128 = 0.08838834764831844f; // 1 / sqrt(128)

__device__ __forceinline__ __half2 as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ u64 rd64_u8(const uint8_t* p) { return __ldg(reinterpret_cast<const u64*>(p)); }

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
    for (int e = 0; e < 4; ++e) v[e] *= ERNIE55_SQ128;
}

// Exact exlr8 mul1 (0x83DCD12D) codebook with FP8 E4M3 round-to-nearest-even saturating finite.
// Two 16-bit trellis states -> half2.
__device__ __forceinline__ u32 mul1_e4m3x2(u32 w0, u32 w1) {
    const u32 s0 = __dp4a(w0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const u32 s1 = __dp4a(w1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 v = __hfma2(as_h2(__byte_perm(s0, s1, 0x5410)), as_h2(0x1eee1eeeu), as_h2(0xc931c931u));
    u32 r = as_u32(v);
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

// Decode 8 mul1 E4M3 weights (8 16-bit scalar trellis states) from wide-order 64-bit window `m`.
// m = (chunk[lane-1] << (8*BITS)) | chunk[lane]; state j = (m >> (7*BITS - BITS*j)) & 0xffff.
template <int BITS>
__device__ __forceinline__ void mul1_decode_m(u64 m, u32 (&f0)[2], u32 (&f1)[2]) {
    constexpr int B0 = 7 * BITS;
    u32 w[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (B0 - BITS * j)) & 0xffffu;
    f0[0] = mul1_e4m3x2(w[0], w[1]);
    f0[1] = mul1_e4m3x2(w[2], w[3]);
    f1[0] = mul1_e4m3x2(w[4], w[5]);
    f1[1] = mul1_e4m3x2(w[6], w[7]);
}

// Wide-load bit-packing helpers (host + device, shared with ernie55_hybrid.cuh).
template <int BITS>
__host__ __device__ __forceinline__ u32 trw_tile_chunk(const uint8_t* tile, int lane) {
    constexpr int CB = 8 * BITS, NW = 8 * BITS;
    const int bit = lane * CB, wi = bit >> 5, off = bit & 31;
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi + 1 < NW ? wi + 1 : NW - 1];
    return static_cast<u32>(m >> (64 - off - CB)) & static_cast<u32>((1ull << CB) - 1);
}

template <int BITS>
__host__ __device__ __forceinline__ void trw_pack(u64 (&v)[3], int p, u32 chunk) {
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if constexpr (CB > 16) {
        if (sh > 64 - CB) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
    }
}

template <int BITS>
__host__ __device__ __forceinline__ u32 trw_chunk(const u64 (&w)[3], int p) {
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if constexpr (CB > 16) {
        if (sh > 64 - CB) v |= w[n + 1] << (64 - sh);
    }
    return static_cast<u32>(v) & static_cast<u32>((1ull << CB) - 1);
}

// Resolve FC1 unit (unit = 2*b + mat) byte pointer and width `bits`, supporting ALL kmix encodings from v2plan.rs
// AND optional fixed-address ptab[(e * eb + b) * 2 + 0] for hybrid offload!
template <int HB>
__device__ __forceinline__ int ernie55_fc1_unit(
    const uint8_t* fc1, const uint8_t* const* ptab, const long long* kmix,
    int e, const long long* tr, int unit, int eb, const uint8_t*& mw) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(HB) * 2048; // 16 * H = 16 * 128 * HB
    const long long km = kmix ? kmix[e] : 0LL;
    const int b = unit >> 1, mat = unit & 1;

    if (ptab != nullptr) {
        const uint8_t* blk_base = ptab[(static_cast<size_t>(e) * eb + b) * 2];
        int gk, uk;
        if (km == 0LL) {
            gk = uk = static_cast<int>(tr[4]);
        } else if (km < 0LL) {
            gk = static_cast<int>(km & 0xfull);
            uk = static_cast<int>((km >> 4) & 0xfull);
        } else {
            gk = static_cast<int>((km >> (8 * b)) & 0xfull);
            uk = static_cast<int>((km >> (8 * b + 4)) & 0xfull);
        }
        mw = blk_base + (mat ? static_cast<size_t>(gk) * UNIT_PER_BIT : 0ull);
        return mat ? uk : gk;
    }

    if (km == 0LL) {
        const int bits = static_cast<int>(tr[4]);
        mw = fc1 + static_cast<size_t>(tr[0]) + static_cast<size_t>(unit) * UNIT_PER_BIT * static_cast<size_t>(bits);
        return bits;
    }
    if (km < 0LL) {
        const int gk = static_cast<int>(km & 0xfull);
        const int uk = static_cast<int>((km >> 4) & 0xfull);
        const size_t off = (static_cast<size_t>(b) * (gk + uk) + (mat ? static_cast<size_t>(gk) : 0ull)) * UNIT_PER_BIT;
        mw = fc1 + static_cast<size_t>(tr[0]) + off;
        return mat ? uk : gk;
    }
    size_t sum_k = 0;
    for (int u = 0; u < unit; ++u) sum_k += static_cast<size_t>((km >> (4 * u)) & 0xfull);
    mw = fc1 + static_cast<size_t>(tr[0]) + sum_k * UNIT_PER_BIT;
    return static_cast<int>((km >> (4 * unit)) & 0xfull);
}

template <int HB>
__device__ __forceinline__ const uint8_t* ernie55_fc2_block(
    const uint8_t* fc2, const uint8_t* const* ptab, int e, const long long* tr, int b, int bits, int eb) {
    if (ptab != nullptr) return ptab[(static_cast<size_t>(e) * eb + b) * 2 + 1];
    constexpr size_t BLK_PER_BIT = static_cast<size_t>(HB) * 2048;
    return fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * static_cast<size_t>(bits);
}

// =================================================================================================
// One-time startup GPU in-place widen for FC1 and FC2 (works for K2, K3, K4; HB=16 routed & HB=32 shared)
// =================================================================================================
template <int HB>
__global__ void __launch_bounds__(256) k_ernie55_tr3_widen_fc1(
    uint8_t* fc1, const long long* tab, const long long* kmix, int ne, int eb) {
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int hb = blockIdx.x % HB;
    const int u  = (blockIdx.x / HB) % (2 * eb);
    const int e  = blockIdx.x / (HB * 2 * eb);
    if (e >= ne) return;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if ((u >> 1) >= static_cast<int>(tr[3])) return;

    const uint8_t* mw_c = nullptr;
    const int k = ernie55_fc1_unit<HB>(fc1, nullptr, kmix, e, tr, u, eb, mw_c);
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
__global__ void __launch_bounds__(256) k_ernie55_tr3_widen_fc2(
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
// Core mul1 Wide-Order Tile MMA Helpers (8-route and 16-route double-B-fragment)
// =================================================================================================
template <int BITS>
__device__ __forceinline__ void ernie55_mul1_fc1_hb_wide(
    const uint8_t* mw, int hb, int warp, int lane,
    const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* grp = mw + static_cast<size_t>(hb) * (2048 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(grp + warp * 1024 + lane * 16));
        const uint4 w1 = __ldg(reinterpret_cast<const uint4*>(grp + warp * 1024 + 512 + lane * 16));
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
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
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

// Dual gate+up mul1 FC1 helper (same BITS): issues both gate & up DRAM weight loads concurrently
// before decoding gate, hiding 100% of up weight latency behind gate decode + 8 MMAs (100% bit-exact!).
template <int BITS>
__device__ __forceinline__ void ernie55_mul1_fc1_hb_pair_wide(
    const uint8_t* mw_g, const uint8_t* mw_u, int hb, int warp, int lane,
    const __half (&x_rot_g)[8][136], const __half (&x_rot_u)[8][136],
    int cnt, float (&cg)[4], float (&cu)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* grpg = mw_g + static_cast<size_t>(hb) * (2048 * BITS);
    const uint8_t* grpu = mw_u + static_cast<size_t>(hb) * (2048 * BITS);

    if constexpr (BITS == 4) {
        const uint4 g0 = __ldg(reinterpret_cast<const uint4*>(grpg + warp * 1024 + lane * 16));
        const uint4 g1 = __ldg(reinterpret_cast<const uint4*>(grpg + warp * 1024 + 512 + lane * 16));
        const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(grpu + warp * 1024 + lane * 16));
        const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(grpu + warp * 1024 + 512 + lane * 16));
        const u32 chg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        const u32 chu[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chg[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot_g[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot_g[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(cg, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chu[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot_u[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot_u[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(cu, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    } else {
        u64 wg[3] = {0, 0, 0}, wu[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) wg[n] = rd64_u8(grpg + (n * 8 + warp) * 256 + lane * 8);
        #pragma unroll
        for (int n = 0; n < BITS; ++n) wu[n] = rd64_u8(grpu + (n * 8 + warp) * 256 + lane * 8);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(wg, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot_g[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot_g[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(cg, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(wu, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot_u[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot_u[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(cu, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

template <int BITS>
__device__ __forceinline__ void ernie55_mul1_fc2_ntile_wide(
    const uint8_t* mw, int n_tile, int lane,
    const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* nt = mw + static_cast<size_t>(n_tile) * (256 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(nt + lane * 16));
        const uint4 w1 = __ldg(reinterpret_cast<const uint4*>(nt + 512 + lane * 16));
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
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
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

template <int BITS>
__device__ __forceinline__ void ernie55_mul1_fc1_hb_wide16(
    const uint8_t* mw, int hb, int warp, int lane,
    const __half (&x_rot)[16][136], int cnt, float (&c0)[4], float (&c1)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act0 = (row0 < cnt);
    const bool act1 = (row0 + 8 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* grp = mw + static_cast<size_t>(hb) * (2048 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(grp + warp * 1024 + lane * 16));
        const uint4 w1 = __ldg(reinterpret_cast<const uint4*>(grp + warp * 1024 + 512 + lane * 16));
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
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
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
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
__device__ __forceinline__ void ernie55_mul1_fc2_ntile_wide16(
    const uint8_t* mw, int n_tile, int lane,
    const __half (&x2)[16][136], int cnt, float (&c0)[4], float (&c1)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act0 = (row0 < cnt);
    const bool act1 = (row0 + 8 < cnt);
    const int prev_lane = (lane - 1) & 31;
    const uint8_t* nt = mw + static_cast<size_t>(n_tile) * (256 * BITS);

    if constexpr (BITS == 4) {
        const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(nt + lane * 16));
        const uint4 w1 = __ldg(reinterpret_cast<const uint4*>(nt + 512 + lane * 16));
        const u32 chs[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = chs[p];
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            mul1_decode_m<4>((static_cast<u64>(pc) << 32) | ch, f0, f1);
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
            mul1_decode_m<BITS>((static_cast<u64>(pc) << (8 * BITS)) | ch, f0, f1);
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

// Pre-rotate input x [M, H] BF16 -> x_had [M, HB, 128] FP16 = H128(x * sg_in).
template <int HB>
__global__ void __launch_bounds__(32) k_ernie55_prerot_x(
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
// Layer 0 Dense SiTU-GLU MLP (I = 14336, TP-sharded)
// =================================================================================================
struct DenseMlpWeights {
    mx8::LinW gate_proj{}; // [dense_i_loc, 4096]
    mx8::LinW up_proj{};   // [dense_i_loc, 4096]
    mx8::LinW down_proj{}; // [4096, dense_i_loc]
    int dense_i_loc = 0;
};

struct DenseMlpScratch {
    __nv_bfloat16* gate = nullptr; // [max_M, dense_i_loc]
    __nv_bfloat16* up   = nullptr; // [max_M, dense_i_loc]
};

__global__ void k_ernie55_situ_mul_bf16(__nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        gate[i] = __float2bfloat16(situ_mul_f(__bfloat162float(gate[i]), __bfloat162float(up[i])));
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
    k_ernie55_situ_mul_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(sc.gate, sc.up, n);
    mx8::linear_w(blas, sc.gate, w.down_proj, out_part, M, geom::HID, w.dense_i_loc, x8_scratch, nullptr, st);
}

// =================================================================================================
// Layers 1..42 Latent MoE (FP32 Split-Feature Router + mul1 Routed + mul1 Shared)
// =================================================================================================
struct Tr3Pack {
    const uint8_t*        fc1  = nullptr;
    const uint8_t*        fc2  = nullptr;
    const uint8_t* const* ptab = nullptr; // optional fixed-address [ne * eb * 2] for hybrid offload
    const long long*      tab  = nullptr; // [ne, 6]
    const long long*      kmix = nullptr; // [ne]
    const __half*         sg   = nullptr; // [2, H] (in, out signs)
    const __half*         v1b  = nullptr; // [nblk, 2, HB]
    const __half*         v2   = nullptr; // [nblk, 3, 128]
    const __half*         v3b  = nullptr; // [nblk, HB]
    int ne = 0;
    int eb = 0;
};

struct MoeWeights {
    const float*         router_w0   = nullptr; // [512, 4096] F32
    const float*         router_w1   = nullptr; // [512, 4096] F32
    const float*         router_bias = nullptr; // [512] F32 (e_score_correction_bias)
    const float*         router_sc   = nullptr; // [512] F32 (routed_scaling_factor_param)
    mx8::LinW            fc1_lat{};             // [2048, 4096]
    const __nv_bfloat16* lat_norm    = nullptr; // [2048] BF16
    mx8::LinW            fc2_lat{};             // [4096, 2048]
    Tr3Pack routed;                             // ne=512, HB=16, eb=16/nr
    Tr3Pack shared;                             // ne=1,   HB=32, eb=16/nr
};

struct MoeScratch {
    float*         x_f32         = nullptr; // [max_M, 4096] F32 (for prefill FP32 router SGEMM)
    float*         router_l0     = nullptr; // [max_M, 512] F32
    float*         router_l1     = nullptr; // [max_M, 512] F32
    int*           topk_idx      = nullptr; // [max_M, 10] INT32
    float*         topk_w        = nullptr; // [max_M, 10] F32
    __nv_bfloat16* routed_in     = nullptr; // [max_M, 2048] BF16
    __nv_bfloat16* routed_lat    = nullptr; // [max_M, 2048] BF16
    __nv_bfloat16* routed_normed = nullptr; // [max_M, 2048] BF16
    __nv_bfloat16* routed_up_out = nullptr; // [max_M, 4096] BF16
    __nv_bfloat16* shared_out    = nullptr; // [max_M, 4096] BF16

    int*    d_n_groups = nullptr; // [1]
    int*    grp_exp    = nullptr; // [max_M * 10]
    int*    grp_cnt    = nullptr; // [max_M * 10]
    int*    grp_routes = nullptr; // [(max_M * 10 + 512) * 16]
    __half* x_had_r    = nullptr; // [max_M, 16, 128] FP16 (routed latent)
    __half* x_had_s    = nullptr; // [max_M, 32, 128] FP16 (shared full H=4096)
    __half* fc1_act_r  = nullptr; // [max_M * 10, max_eb, 128] FP16
    __half* fc1_act_s  = nullptr; // [max_M, max_eb, 128] FP16
    float*  fc2_part_r = nullptr; // [max_M * 10, 2048] FP32
    float*  sh_fc1_part = nullptr; // [16, max_eb, 4, 2, 128] FP32 (shared FC1 K-split=4 scratch)
    float*  sh_fc2_part = nullptr; // [16, HB, 2, 128] FP32 (shared FC2 K-split=2 scratch)
};

// Fused FP32 Router (W0[512,4096] + W1[512,4096], 256 heavy CTAs FIRST) + MXFP8 fc1_lat ([2048,4096], 512 light CTAs)
// on shared x_norm [M, 4096]! Total 768 CTAs = exact 16.00 waves on 48 SMs (eliminates 11.1% router wave tail + 42 launches).
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_ernie55_moe_router_fc1lat_mx8(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ W0,
    const float* __restrict__ W1,
    const uint8_t* __restrict__ lat_q,
    const uint8_t* __restrict__ lat_sc,
    float* __restrict__ L0,
    float* __restrict__ L1,
    __nv_bfloat16* __restrict__ routed_in,
    int M, int m_off) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int bx = blockIdx.x;
    const __nv_bfloat16* Xm0 = X + static_cast<size_t>(m_off) * geom::HID;

    if (bx < 256) {
        const int e_ext = bx * 4 + warp;
        if (e_ext >= 2 * geom::NEXP) return;
        const bool is_w1 = (e_ext >= geom::NEXP);
        const int e = is_w1 ? (e_ext - geom::NEXP) : e_ext;
        const float4* wr = reinterpret_cast<const float4*>((is_w1 ? W1 : W0) + static_cast<size_t>(e) * geom::HID);
        float a[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) a[m] = 0.0f;

        for (int i = lane; i < (geom::HID >> 2); i += 128) {
            const float4 f0 = __ldg(wr + i);
            const float4 f1 = __ldg(wr + i + 32);
            const float4 f2 = __ldg(wr + i + 64);
            const float4 f3 = __ldg(wr + i + 96);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (EXACT_M || m < M) {
                    const uint2* xm = reinterpret_cast<const uint2*>(Xm0 + static_cast<size_t>(m) * geom::HID);
                    const uint2 r0 = __ldg(xm + i);
                    const uint2 r1 = __ldg(xm + i + 32);
                    const uint2 r2 = __ldg(xm + i + 64);
                    const uint2 r3 = __ldg(xm + i + 96);
                    auto dot4 = [](uint2 raw, float4 f, float acc) -> float {
                        const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                        const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                        return fmaf(x01.x, f.x, fmaf(x01.y, f.y, fmaf(x23.x, f.z, fmaf(x23.y, f.w, acc))));
                    };
                    a[m] = dot4(r0, f0, a[m]);
                    a[m] = dot4(r1, f1, a[m]);
                    a[m] = dot4(r2, f2, a[m]);
                    a[m] = dot4(r3, f3, a[m]);
                }
            }
        }
        float* L = is_w1 ? L1 : L0;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const float s = warp_sum(a[m]);
                if (lane == 0) L[static_cast<size_t>(m_off + m) * geom::NEXP + e] = s;
            }
        }
    } else {
        const int row = (bx - 256) * 4 + warp;
        if (row >= geom::EH) return;
        constexpr int K = geom::HID;
        constexpr int K16 = K >> 4;
        constexpr int NB  = K >> 5;
        const uint4* qrow = reinterpret_cast<const uint4*>(lat_q + static_cast<size_t>(row) * K);
        const uint8_t* srow = lat_sc + static_cast<size_t>(row) * NB;
        float acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

        #pragma unroll
        for (int i = lane; i < K16; i += 128) {
            const uint4 w0 = __ldg(qrow + i);
            const uint4 w1 = __ldg(qrow + i + 32);
            const uint4 w2 = __ldg(qrow + i + 64);
            const uint4 w3 = __ldg(qrow + i + 96);
            const uint8_t s0 = __ldg(srow + (i >> 1));
            const uint8_t s1 = __ldg(srow + ((i + 32) >> 1));
            const uint8_t s2 = __ldg(srow + ((i + 64) >> 1));
            const uint8_t s3 = __ldg(srow + ((i + 96) >> 1));
            mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Xm0, w0, s0, i,      M, K, acc);
            mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Xm0, w1, s1, i + 32, M, K, acc);
            mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Xm0, w2, s2, i + 64, M, K, acc);
            mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Xm0, w3, s3, i + 96, M, K, acc);
        }
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const float tot = warp_sum(acc[m]);
                if (lane == 0) routed_in[static_cast<size_t>(m_off + m) * geom::EH + row] = __float2bfloat16(tot);
            }
        }
    }
}

// Fast coalesced F32 GEMV for small-M router (M <= 12): splits W0 (0..511) & W1 (512..1023) across
// Grid: (256), Block: 128 (4 warps/CTA, 5.33 waves on 48 SMs) with 4x-unrolled float4 loads (64 B/lane in flight).
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ernie55_router_f32_gemv(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ W0,
    const float* __restrict__ W1,
    float* __restrict__ L0,
    float* __restrict__ L1,
    int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e_ext = blockIdx.x * 4 + warp;
    if (e_ext >= 2 * geom::NEXP) return;
    const bool is_w1 = (e_ext >= geom::NEXP);
    const int e = is_w1 ? (e_ext - geom::NEXP) : e_ext;
    const float4* wr = reinterpret_cast<const float4*>((is_w1 ? W1 : W0) + static_cast<size_t>(e) * geom::HID);
    float a[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) a[m] = 0.0f;

    // HID / 4 = 1024 = 8 * 128 exact! 4x-unrolled __ldg float4 loads.
    for (int i = lane; i < (geom::HID >> 2); i += 128) {
        const float4 f0 = __ldg(wr + i);
        const float4 f1 = __ldg(wr + i + 32);
        const float4 f2 = __ldg(wr + i + 64);
        const float4 f3 = __ldg(wr + i + 96);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint2* xm = reinterpret_cast<const uint2*>(X + static_cast<size_t>(m) * geom::HID);
                const uint2 r0 = __ldg(xm + i);
                const uint2 r1 = __ldg(xm + i + 32);
                const uint2 r2 = __ldg(xm + i + 64);
                const uint2 r3 = __ldg(xm + i + 96);
                auto dot4 = [](uint2 raw, float4 f, float acc) -> float {
                    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
                    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
                    return fmaf(x01.x, f.x, fmaf(x01.y, f.y, fmaf(x23.x, f.z, fmaf(x23.y, f.w, acc))));
                };
                a[m] = dot4(r0, f0, a[m]);
                a[m] = dot4(r1, f1, a[m]);
                a[m] = dot4(r2, f2, a[m]);
                a[m] = dot4(r3, f3, a[m]);
            }
        }
    }
    float* L = is_w1 ? L1 : L0;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = warp_sum(a[m]);
            if (lane == 0) L[static_cast<size_t>(m) * geom::NEXP + e] = s;
        }
    }
}

// Exact ERNIE 5.5 Split-Feature Top-10 Router:
//   scores = sigmoid(l0) + sigmoid(l1)
//   choice = scores + bias -> top-10 (stable lowest-index tie-break) -> renorm * scale[ids]
// Optionally increments d_exp_hits[e] for hybrid adaptive cache heat tracking.
// Grid: (M), Block: 256 (each thread owns 2 experts: tid and tid + 256).
__global__ void __launch_bounds__(256) k_ernie55_router_topk10(
    const float* __restrict__ l0,
    const float* __restrict__ l1,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    uint32_t* __restrict__ d_exp_hits,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const float* r0 = l0 + static_cast<size_t>(m) * geom::NEXP;
    const float* r1 = l1 + static_cast<size_t>(m) * geom::NEXP;

    __shared__ float s_sc[geom::NEXP];
    __shared__ float s_ch[geom::NEXP];
    for (int k = 0; k < 2; ++k) {
        const int e = tid + k * 256;
        const float sc = (1.0f / (1.0f + expf(-r0[e]))) + (1.0f / (1.0f + expf(-r1[e])));
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
            sel_w[k] = s_sc[mi];
            s_ch[mi] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid < geom::TOPK) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < geom::TOPK; ++k) sum += sel_w[k];
        const float denom = fmaxf(sum, 1e-20f);
        const int ei = sel_e[tid];
        topk_idx[static_cast<size_t>(m) * geom::TOPK + tid] = ei;
        topk_w[static_cast<size_t>(m) * geom::TOPK + tid]   = (sel_w[tid] / denom) * scale[ei];
        if (d_exp_hits != nullptr) atomicAdd(&d_exp_hits[ei], 1u);
    }
}

// Fast M=1 group builder (10 single-route groups, 1 warp).
__global__ void k_ernie55_moe_groups_m1(const int* __restrict__ topk_idx,
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

// Parallel 2-pass shared-memory group builder for arbitrary M >= 2.
template <int GMAX>
__global__ void __launch_bounds__(512) k_ernie55_moe_build_groups(
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

    for (int r = tid; r < n_routes; r += 512) atomicAdd(&s_cnt[topk_idx[r]], 1);
    __syncthreads();

    const int cnt = s_cnt[tid];
    const int ng = (cnt + GMAX - 1) / GMAX;
    const int lane = tid & 31, warp = tid >> 5;
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
    s_base[tid] = w_off + v - ng;
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

// Grouped mul1 Routed FC1 (EHB=16, SiTU-GLU, supports resident fc1 OR hybrid ptab!).
__global__ void __launch_bounds__(256) k_ernie55_routed_fc1_grp(
    const __half* __restrict__ x_had,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const uint8_t* const* __restrict__ ptab,
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
    const int kg = ernie55_fc1_unit<geom::EHB>(fc1, ptab, kmix, e, tr, 2 * b + 0, eb, mw_g);
    const int ku = ernie55_fc1_unit<geom::EHB>(fc1, ptab, kmix, e, tr, 2 * b + 1, eb, mw_u);

    __shared__ __half x_rot_g[8][136];
    __shared__ __half x_rot_u[8][136];
    float cg[4] = {0, 0, 0, 0};
    float cu[4] = {0, 0, 0, 0};

    for (int hb = 0; hb < geom::EHB; ++hb) {
        if (warp < cnt) {
            const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
                x_had + (static_cast<size_t>(s_tok[warp]) * geom::EHB + hb) * 128 + lane * 4));
            const __half2 bsg = __half2half2(__ldg(v1b + (brow * 2 + 0) * geom::EHB + hb));
            const __half2 bsu = __half2half2(__ldg(v1b + (brow * 2 + 1) * geom::EHB + hb));
            uint2 ovg, ovu;
            *reinterpret_cast<__half2*>(&ovg.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsg);
            *reinterpret_cast<__half2*>(&ovg.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsg);
            *reinterpret_cast<__half2*>(&ovu.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsu);
            *reinterpret_cast<__half2*>(&ovu.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsu);
            *reinterpret_cast<uint2*>(x_rot_g[warp] + lane * 4) = ovg;
            *reinterpret_cast<uint2*>(x_rot_u[warp] + lane * 4) = ovu;
        }
        __syncthreads();
        if (kg == ku) {
            if (kg == 2)      ernie55_mul1_fc1_hb_pair_wide<2>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
            else if (kg == 3) ernie55_mul1_fc1_hb_pair_wide<3>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
            else              ernie55_mul1_fc1_hb_pair_wide<4>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
        } else {
            if (kg == 3)      ernie55_mul1_fc1_hb_wide<3>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            else if (kg == 4) ernie55_mul1_fc1_hb_wide<4>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            else              ernie55_mul1_fc1_hb_wide<2>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            if (ku == 3)      ernie55_mul1_fc1_hb_wide<3>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
            else if (ku == 4) ernie55_mul1_fc1_hb_wide<4>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
            else              ernie55_mul1_fc1_hb_wide<2>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
        }
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
            const float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            const float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            const float act = __bfloat162float(__float2bfloat16(situ_mul_f(gf, uf)));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

// Grouped mul1 Routed FC2 (EHB=16): sums over b=0..nb-1 -> writes fc2_part[route, 2048] FP32.
__global__ void __launch_bounds__(256) k_ernie55_routed_fc2_grp(
    const __half* __restrict__ fc1_act,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const uint8_t* const* __restrict__ ptab,
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

    __shared__ int s_route[8];
    if (tid < cnt) s_route[tid] = grp_routes[g * 8 + tid];
    __syncthreads();

    __shared__ __half x2[8][136];
    float acc[4] = {0, 0, 0, 0};
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        if (warp < cnt) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[warp] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = ernie55_fc2_block<geom::EHB>(fc2, ptab, e, tr, b, bits, eb);
        float c[4] = {0, 0, 0, 0};
        if (bits == 3)      ernie55_mul1_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 4) ernie55_mul1_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else                ernie55_mul1_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
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

// 16-route double-B-fragment mul1 Routed FC1/FC2 for prefill (M >= 16).
__global__ void __launch_bounds__(256) k_ernie55_routed_fc1_grp16(
    const __half* __restrict__ x_had,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const uint8_t* const* __restrict__ ptab,
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
    const int kg = ernie55_fc1_unit<geom::EHB>(fc1, ptab, kmix, e, tr, 2 * b + 0, eb, mw_g);
    const int ku = ernie55_fc1_unit<geom::EHB>(fc1, ptab, kmix, e, tr, 2 * b + 1, eb, mw_u);

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
        if (kg == 3)      ernie55_mul1_fc1_hb_wide16<3>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else if (kg == 4) ernie55_mul1_fc1_hb_wide16<4>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else              ernie55_mul1_fc1_hb_wide16<2>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
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
        if (ku == 3)      ernie55_mul1_fc1_hb_wide16<3>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else if (ku == 4) ernie55_mul1_fc1_hb_wide16<4>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else              ernie55_mul1_fc1_hb_wide16<2>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
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
            const float act = __bfloat162float(__float2bfloat16(situ_mul_f(gf, uf)));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(s_route[w]) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_ernie55_routed_fc2_grp16(
    const __half* __restrict__ fc1_act,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const uint8_t* const* __restrict__ ptab,
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
    const int n_tile = hb * 8 + warp;

    for (int b = 0; b < nb; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        for (int w = warp; w < cnt; w += 8) {
            const uint2 xv = *reinterpret_cast<const uint2*>(
                fc1_act + (static_cast<size_t>(s_route[w]) * eb + b) * 128 + lane * 4);
            *reinterpret_cast<uint2*>(x2[w] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = ernie55_fc2_block<geom::EHB>(fc2, ptab, e, tr, b, bits, eb);
        float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
        if (bits == 3)      ernie55_mul1_fc2_ntile_wide16<3>(mw, n_tile, lane, x2, cnt, c0, c1);
        else if (bits == 4) ernie55_mul1_fc2_ntile_wide16<4>(mw, n_tile, lane, x2, cnt, c0, c1);
        else                ernie55_mul1_fc2_ntile_wide16<2>(mw, n_tile, lane, x2, cnt, c0, c1);
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

// Deterministic sum over all 10 routes + fused latent_norm -> routed_normed [M, 2048] BF16!
// Grid: (M), Block: 256 (2 x float4 = 8 elements per thread for EH=2048 exact, coalesced float4 __ldg).
__global__ void __launch_bounds__(256) k_ernie55_routed_reduce_and_norm(
    const float* __restrict__ fc2_part,
    const __nv_bfloat16* __restrict__ lat_norm_w,
    __nv_bfloat16* __restrict__ routed_normed, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    constexpr int EH4 = geom::EH >> 2; // 512

    float4 vals[2];
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int d4 = tid + j * 256;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        #pragma unroll
        for (int k = 0; k < geom::TOPK; ++k) {
            const int r = m * geom::TOPK + k;
            const float4 v = __ldg(reinterpret_cast<const float4*>(fc2_part) + static_cast<size_t>(r) * EH4 + d4);
            a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
        }
        const float b0 = __bfloat162float(__float2bfloat16(a0));
        const float b1 = __bfloat162float(__float2bfloat16(a1));
        const float b2 = __bfloat162float(__float2bfloat16(a2));
        const float b3 = __bfloat162float(__float2bfloat16(a3));
        vals[j] = make_float4(b0, b1, b2, b3);
        ss = fmaf(b0, b0, fmaf(b1, b1, fmaf(b2, b2, fmaf(b3, b3, ss))));
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::EH)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    uint2* dst2 = reinterpret_cast<uint2*>(routed_normed + static_cast<size_t>(m) * geom::EH);
    const uint2* nw2 = reinterpret_cast<const uint2*>(lat_norm_w);
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int d4 = tid + j * 256;
        const uint2 raw = __ldg(nw2 + d4);
        const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        uint2 out;
        *reinterpret_cast<__nv_bfloat162*>(&out.x) = __floats2bfloat162_rn(vals[j].x * inv * w01.x, vals[j].y * inv * w01.y);
        *reinterpret_cast<__nv_bfloat162*>(&out.y) = __floats2bfloat162_rn(vals[j].z * inv * w23.x, vals[j].w * inv * w23.y);
        dst2[d4] = out;
    }
}

// =================================================================================================
// Shared mul1 Expert (H=4096, HB=32, eb=16/nr, SiTU-GLU, always VRAM-resident)
// =================================================================================================
__global__ void __launch_bounds__(256) k_ernie55_shared_fc1(
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
    const int kg = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 0, eb, mw_g);
    const int ku = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 1, eb, mw_u);

    __shared__ __half x_rot_g[8][136];
    __shared__ __half x_rot_u[8][136];
    float cg[4] = {0, 0, 0, 0};
    float cu[4] = {0, 0, 0, 0};

    for (int hb = 0; hb < geom::HB; ++hb) {
        if (warp < cnt) {
            const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + warp) * geom::HB + hb) * 128 + lane * 4));
            const __half2 bsg = __half2half2(__ldg(v1b + (brow * 2 + 0) * geom::HB + hb));
            const __half2 bsu = __half2half2(__ldg(v1b + (brow * 2 + 1) * geom::HB + hb));
            uint2 ovg, ovu;
            *reinterpret_cast<__half2*>(&ovg.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsg);
            *reinterpret_cast<__half2*>(&ovg.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsg);
            *reinterpret_cast<__half2*>(&ovu.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsu);
            *reinterpret_cast<__half2*>(&ovu.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsu);
            *reinterpret_cast<uint2*>(x_rot_g[warp] + lane * 4) = ovg;
            *reinterpret_cast<uint2*>(x_rot_u[warp] + lane * 4) = ovu;
        }
        __syncthreads();
        if (kg == 4)      ernie55_mul1_fc1_hb_wide<4>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
        else if (kg == 3) ernie55_mul1_fc1_hb_wide<3>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
        else              ernie55_mul1_fc1_hb_wide<2>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
        if (ku == 4)      ernie55_mul1_fc1_hb_wide<4>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
        else if (ku == 3) ernie55_mul1_fc1_hb_wide<3>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
        else              ernie55_mul1_fc1_hb_wide<2>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
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
            const float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
            const float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
            const float act = __bfloat162float(__float2bfloat16(situ_mul_f(gf, uf)));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act_s + (static_cast<size_t>(m0 + warp) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_ernie55_shared_fc2(
    const __half* __restrict__ fc1_act_s,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ sg,
    const __half* __restrict__ v3b,
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
        if (bits == 4)      ernie55_mul1_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 3) ernie55_mul1_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else                ernie55_mul1_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
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

__global__ void __launch_bounds__(256) k_ernie55_shared_fc1_16(
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
    const int kg = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 0, eb, mw_g);
    const int ku = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 1, eb, mw_u);

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
        if (kg == 4)      ernie55_mul1_fc1_hb_wide16<4>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else if (kg == 3) ernie55_mul1_fc1_hb_wide16<3>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
        else              ernie55_mul1_fc1_hb_wide16<2>(mw_g, hb, warp, lane, x_rot, cnt, cg0, cg1);
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
        if (ku == 4)      ernie55_mul1_fc1_hb_wide16<4>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else if (ku == 3) ernie55_mul1_fc1_hb_wide16<3>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
        else              ernie55_mul1_fc1_hb_wide16<2>(mw_u, hb, warp, lane, x_rot, cnt, cu0, cu1);
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
            const float act = __bfloat162float(__float2bfloat16(situ_mul_f(gf, uf)));
            va[i] = act * __half2float(su_d[c]);
        }
        had128(va, lane);
        __half* dst = fc1_act_s + (static_cast<size_t>(m0 + w) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

__global__ void __launch_bounds__(256) k_ernie55_shared_fc2_16(
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
        if (bits == 4)      ernie55_mul1_fc2_ntile_wide16<4>(mw, n_tile, lane, x2, cnt, c0, c1);
        else if (bits == 3) ernie55_mul1_fc2_ntile_wide16<3>(mw, n_tile, lane, x2, cnt, c0, c1);
        else                ernie55_mul1_fc2_ntile_wide16<2>(mw, n_tile, lane, x2, cnt, c0, c1);
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

__global__ void k_ernie55_add_bf16(const __nv_bfloat16* __restrict__ a,
                                   const __nv_bfloat16* __restrict__ b,
                                   __nv_bfloat16* __restrict__ out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// K-split=4 Shared FC1 (64 CTAs for M<=8 instead of 16 CTAs -> fills all 48 SMs!)
__global__ void __launch_bounds__(256) k_ernie55_shared_fc1_part(
    const __half* __restrict__ x_had_s,
    const long long* __restrict__ tab,
    const long long* __restrict__ kmix,
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1b,
    float* __restrict__ sh_fc1_part, // [M, eb, 4, 2, 128]
    int M, int eb) {
    const int sp  = blockIdx.x & 3;
    const int rem = blockIdx.x >> 2;
    const int b   = rem % eb;
    const int m8  = rem / eb;
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
    const int kg = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 0, eb, mw_g);
    const int ku = ernie55_fc1_unit<geom::HB>(fc1, nullptr, kmix, 0, tr, 2 * b + 1, eb, mw_u);

    __shared__ __half x_rot_g[8][136];
    __shared__ __half x_rot_u[8][136];
    float cg[4] = {0, 0, 0, 0};
    float cu[4] = {0, 0, 0, 0};

    constexpr int CHUNK = geom::HB / 4; // 8
    const int hb0 = sp * CHUNK;
    for (int hb = hb0; hb < hb0 + CHUNK; ++hb) {
        if (warp < cnt) {
            const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
                x_had_s + (static_cast<size_t>(m0 + warp) * geom::HB + hb) * 128 + lane * 4));
            const __half2 bsg = __half2half2(__ldg(v1b + (brow * 2 + 0) * geom::HB + hb));
            const __half2 bsu = __half2half2(__ldg(v1b + (brow * 2 + 1) * geom::HB + hb));
            uint2 ovg, ovu;
            *reinterpret_cast<__half2*>(&ovg.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsg);
            *reinterpret_cast<__half2*>(&ovg.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsg);
            *reinterpret_cast<__half2*>(&ovu.x) = __hmul2(*reinterpret_cast<const __half2*>(&xv.x), bsu);
            *reinterpret_cast<__half2*>(&ovu.y) = __hmul2(*reinterpret_cast<const __half2*>(&xv.y), bsu);
            *reinterpret_cast<uint2*>(x_rot_g[warp] + lane * 4) = ovg;
            *reinterpret_cast<uint2*>(x_rot_u[warp] + lane * 4) = ovu;
        }
        __syncthreads();
        if (kg == ku) {
            if (kg == 4)      ernie55_mul1_fc1_hb_pair_wide<4>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
            else if (kg == 3) ernie55_mul1_fc1_hb_pair_wide<3>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
            else              ernie55_mul1_fc1_hb_pair_wide<2>(mw_g, mw_u, hb, warp, lane, x_rot_g, x_rot_u, cnt, cg, cu);
        } else {
            if (kg == 4)      ernie55_mul1_fc1_hb_wide<4>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            else if (kg == 3) ernie55_mul1_fc1_hb_wide<3>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            else              ernie55_mul1_fc1_hb_wide<2>(mw_g, hb, warp, lane, x_rot_g, cnt, cg);
            if (ku == 4)      ernie55_mul1_fc1_hb_wide<4>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
            else if (ku == 3) ernie55_mul1_fc1_hb_wide<3>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
            else              ernie55_mul1_fc1_hb_wide<2>(mw_u, hb, warp, lane, x_rot_u, cnt, cu);
        }
        __syncthreads();
    }

    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        const size_t base = (((static_cast<size_t>(m0 + t0) * eb + b) * 4 + sp) * 2) * 128;
        sh_fc1_part[base + warp * 16 + row0]           = cg[0];
        sh_fc1_part[base + warp * 16 + row0 + 8]       = cg[2];
        sh_fc1_part[base + 128 + warp * 16 + row0]     = cu[0];
        sh_fc1_part[base + 128 + warp * 16 + row0 + 8] = cu[2];
    }
    if (t1 < cnt) {
        const size_t base = (((static_cast<size_t>(m0 + t1) * eb + b) * 4 + sp) * 2) * 128;
        sh_fc1_part[base + warp * 16 + row0]           = cg[1];
        sh_fc1_part[base + warp * 16 + row0 + 8]       = cg[3];
        sh_fc1_part[base + 128 + warp * 16 + row0]     = cu[1];
        sh_fc1_part[base + 128 + warp * 16 + row0 + 8] = cu[3];
    }
}

__global__ void __launch_bounds__(32) k_ernie55_shared_fc1_epilogue(
    const float* __restrict__ sh_fc1_part,
    const long long* __restrict__ tab,
    const __half* __restrict__ v2,
    __half* __restrict__ fc1_act_s,
    int M, int eb) {
    const int idx = blockIdx.x;
    if (idx >= M * eb) return;
    const int m = idx / eb;
    const int b = idx % eb;
    if (b >= static_cast<int>(tab[3])) return;
    const size_t brow = static_cast<size_t>(tab[2]) + b;
    const int lane = threadIdx.x;

    float vg[4] = {0, 0, 0, 0};
    float vu[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int sp = 0; sp < 4; ++sp) {
        const size_t base = (((static_cast<size_t>(m) * eb + b) * 4 + sp) * 2) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] += sh_fc1_part[base + lane * 4 + i];
            vu[i] += sh_fc1_part[base + 128 + lane * 4 + i];
        }
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
        const float gf = __bfloat162float(__float2bfloat16(vg[i] * __half2float(sv_g[c])));
        const float uf = __bfloat162float(__float2bfloat16(vu[i] * __half2float(sv_u[c])));
        const float act = __bfloat162float(__float2bfloat16(situ_mul_f(gf, uf)));
        va[i] = act * __half2float(su_d[c]);
    }
    had128(va, lane);
    __half* dst = fc1_act_s + (static_cast<size_t>(m) * eb + b) * 128;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
}

// K-split=2 Shared FC2 (64 CTAs for M<=8 instead of 32 CTAs -> fills all 48 SMs!)
__global__ void __launch_bounds__(256) k_ernie55_shared_fc2_part(
    const __half* __restrict__ fc1_act_s,
    const long long* __restrict__ tab,
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ v3b,
    float* __restrict__ sh_fc2_part, // [M, HB, 2, 128]
    int M, int eb) {
    const int sp  = blockIdx.x & 1;
    const int rem = blockIdx.x >> 1;
    const int hb  = rem % geom::HB;
    const int m8  = rem / geom::HB;
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

    const int half_nb = (nb + 1) >> 1;
    const int b_start = (sp == 0) ? 0 : half_nb;
    const int b_end   = (sp == 0) ? half_nb : nb;
    for (int b = b_start; b < b_end; ++b) {
        const size_t brow = static_cast<size_t>(tr[2]) + b;
        if (warp < cnt) {
            const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
                fc1_act_s + (static_cast<size_t>(m0 + warp) * eb + b) * 128 + lane * 4));
            *reinterpret_cast<uint2*>(x2[warp] + lane * 4) = xv;
        }
        __syncthreads();
        const uint8_t* mw = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * BLK_PER_BIT * bits;
        float c[4] = {0, 0, 0, 0};
        if (bits == 4)      ernie55_mul1_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 3) ernie55_mul1_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else                ernie55_mul1_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
        const float sc = __half2float(__ldg(v3b + brow * geom::HB + hb));
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(c[i], sc, acc[i]);
        __syncthreads();
    }

    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        const size_t base = ((static_cast<size_t>(m0 + t0) * geom::HB + hb) * 2 + sp) * 128;
        sh_fc2_part[base + warp * 16 + row0]     = acc[0];
        sh_fc2_part[base + warp * 16 + row0 + 8] = acc[2];
    }
    if (t1 < cnt) {
        const size_t base = ((static_cast<size_t>(m0 + t1) * geom::HB + hb) * 2 + sp) * 128;
        sh_fc2_part[base + warp * 16 + row0]     = acc[1];
        sh_fc2_part[base + warp * 16 + row0 + 8] = acc[3];
    }
}

__global__ void __launch_bounds__(32) k_ernie55_shared_fc2_epilogue(
    const float* __restrict__ sh_fc2_part,
    const __half* __restrict__ sg,
    __nv_bfloat16* __restrict__ shared_out,
    int M) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::HB) return;
    const int m  = idx / geom::HB;
    const int hb = idx % geom::HB;
    const int lane = threadIdx.x;

    const size_t base = (static_cast<size_t>(m) * geom::HB + hb) * 2 * 128;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = sh_fc2_part[base + lane * 4 + i] + sh_fc2_part[base + 128 + lane * 4 + i];
    }
    had128(v, lane);
    const __half* sg_out = sg + geom::HID + hb * 128;
    __nv_bfloat16* dst = shared_out + static_cast<size_t>(m) * geom::HID + hb * 128;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2bfloat16(v[i] * __half2float(sg_out[lane * 4 + i]));
}

// =================================================================================================
// MoE Stages: Router, Shared Expert, Routed Experts (resident OR hybrid callback!)
// =================================================================================================
inline void moe_route(cublasHandle_t blas, const MoeWeights& w, MoeScratch& sc,
                      const __nv_bfloat16* x_norm, int M, uint32_t* d_exp_hits, cudaStream_t st) {
    if (M <= 12) {
        constexpr unsigned G = (2 * geom::NEXP + 3) / 4; // 256
        if (M <= 1)      k_ernie55_router_f32_gemv<1> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, sc.router_l0, sc.router_l1, M);
        else if (M <= 2) k_ernie55_router_f32_gemv<2> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, sc.router_l0, sc.router_l1, M);
        else if (M <= 4) k_ernie55_router_f32_gemv<4> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, sc.router_l0, sc.router_l1, M);
        else if (M <= 8) k_ernie55_router_f32_gemv<8> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, sc.router_l0, sc.router_l1, M);
        else             k_ernie55_router_f32_gemv<12><<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, sc.router_l0, sc.router_l1, M);
    } else {
        const size_t n = static_cast<size_t>(M) * geom::HID;
        k_ernie55_bf16_to_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(x_norm, sc.x_f32, n);
        cublasSetStream(blas, st);
        gemm_f32_nt(blas, M, geom::NEXP, geom::HID, sc.x_f32, w.router_w0, sc.router_l0);
        gemm_f32_nt(blas, M, geom::NEXP, geom::HID, sc.x_f32, w.router_w1, sc.router_l1);
    }
    k_ernie55_router_topk10<<<M, 256, 0, st>>>(
        sc.router_l0, sc.router_l1, w.router_bias, w.router_sc, sc.topk_idx, sc.topk_w, d_exp_hits, M);
}

inline void moe_shared(const MoeWeights& w, MoeScratch& sc, const __nv_bfloat16* x_norm, int M, cudaStream_t st) {
    k_ernie55_prerot_x<geom::HB><<<M * geom::HB, 32, 0, st>>>(x_norm, w.shared.sg, sc.x_had_s, M);
    if (M < 16) {
        const int n_m8 = (M + 7) / 8;
        if (sc.sh_fc1_part != nullptr && sc.sh_fc2_part != nullptr) {
            k_ernie55_shared_fc1_part<<<n_m8 * w.shared.eb * 4, 256, 0, st>>>(
                sc.x_had_s, w.shared.tab, w.shared.kmix, w.shared.fc1, w.shared.v1b, sc.sh_fc1_part, M, w.shared.eb);
            k_ernie55_shared_fc1_epilogue<<<M * w.shared.eb, 32, 0, st>>>(
                sc.sh_fc1_part, w.shared.tab, w.shared.v2, sc.fc1_act_s, M, w.shared.eb);
            k_ernie55_shared_fc2_part<<<n_m8 * geom::HB * 2, 256, 0, st>>>(
                sc.fc1_act_s, w.shared.tab, w.shared.fc2, w.shared.v3b, sc.sh_fc2_part, M, w.shared.eb);
            k_ernie55_shared_fc2_epilogue<<<M * geom::HB, 32, 0, st>>>(
                sc.sh_fc2_part, w.shared.sg, sc.shared_out, M);
        } else {
            k_ernie55_shared_fc1<<<n_m8 * w.shared.eb, 256, 0, st>>>(
                sc.x_had_s, w.shared.tab, w.shared.kmix, w.shared.fc1, w.shared.v1b, w.shared.v2,
                sc.fc1_act_s, M, w.shared.eb);
            k_ernie55_shared_fc2<<<n_m8 * geom::HB, 256, 0, st>>>(
                sc.fc1_act_s, w.shared.tab, w.shared.fc2, w.shared.sg, w.shared.v3b, sc.shared_out, M, w.shared.eb);
        }
    } else {
        const int n_m16 = (M + 15) / 16;
        k_ernie55_shared_fc1_16<<<n_m16 * w.shared.eb, 256, 0, st>>>(
            sc.x_had_s, w.shared.tab, w.shared.kmix, w.shared.fc1, w.shared.v1b, w.shared.v2,
            sc.fc1_act_s, M, w.shared.eb);
        k_ernie55_shared_fc2_16<<<n_m16 * geom::HB, 256, 0, st>>>(
            sc.fc1_act_s, w.shared.tab, w.shared.fc2, w.shared.sg, w.shared.v3b, sc.shared_out, M, w.shared.eb);
    }
}

inline void moe_routed_finish(cublasHandle_t blas, const MoeWeights& w, MoeScratch& sc,
                              __nv_bfloat16* out_part, int M, uint8_t* x8_scratch,
                              cudaStream_t st, bool skip_add) {
    k_ernie55_routed_reduce_and_norm<<<M, 256, 0, st>>>(sc.fc2_part_r, w.lat_norm, sc.routed_normed, M);
    mx8::linear_w(blas, sc.routed_normed, w.fc2_lat, sc.routed_up_out, M, geom::HID, geom::EH, x8_scratch, nullptr, st);
    if (!skip_add && out_part != nullptr) {
        const size_t n = static_cast<size_t>(M) * geom::HID;
        k_ernie55_add_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(
            sc.routed_up_out, sc.shared_out, out_part, n);
    }
}

struct MoeProfDetail {
    double router_ms = 0, fc1_lat_ms = 0, prerot_r_ms = 0, routed_fc1_ms = 0;
    double routed_fc2_ms = 0, shared_ms = 0, finish_ms = 0;
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    void mark(double& acc, cudaStream_t st) {
        cudaEventRecord(ev_b, st);
        cudaEventSynchronize(ev_b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_a, ev_b);
        acc += ms;
        cudaEventRecord(ev_a, st);
    }
};

// Resident MoE forward (Spark TP=1 or TP=2/4). Hybrid offload calls moe_route + fc1_lat + prerot +
// hybrid.run_routed_layer(...) + moe_shared + moe_routed_finish!
// When stream_sh/ev_fork/ev_join are provided (and !pd), runs shared expert in parallel with routed path
// (including native fork/join inside CUDA Graph capture) to fill small-grid SM wave tails!
inline void moe_forward_resident(cublasHandle_t blas, const MoeWeights& w, MoeScratch& sc,
                                 const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
                                 int M, uint8_t* x8_scratch, cudaStream_t st, bool skip_add = false,
                                 MoeProfDetail* pd = nullptr,
                                 cudaStream_t stream_sh = nullptr,
                                 cudaEvent_t ev_fork = nullptr,
                                 cudaEvent_t ev_join = nullptr) {
    const bool parallel_sh = (!pd && stream_sh != nullptr && ev_fork != nullptr && ev_join != nullptr);
    if (parallel_sh) {
        cudaEventRecord(ev_fork, st);
        cudaStreamWaitEvent(stream_sh, ev_fork, 0);
        moe_shared(w, sc, x_norm, M, stream_sh);
        cudaEventRecord(ev_join, stream_sh);
    }
    if (pd) cudaEventRecord(pd->ev_a, st);
    if (M <= 15 && w.fc1_lat.is_mx8()) {
        auto launch_rf = [&](int cnt, int m_off) {
            constexpr unsigned G = (2 * geom::NEXP + geom::EH) / 4; // 768 = exact 16.00 waves on 48 SMs
            if (cnt == 1)      k_ernie55_moe_router_fc1lat_mx8<1, true> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, 1, m_off);
            else if (cnt == 2) k_ernie55_moe_router_fc1lat_mx8<2, true> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, 2, m_off);
            else if (cnt == 4) k_ernie55_moe_router_fc1lat_mx8<4, true> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, 4, m_off);
            else if (cnt == 6) k_ernie55_moe_router_fc1lat_mx8<6, true> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, 6, m_off);
            else if (cnt == 8) k_ernie55_moe_router_fc1lat_mx8<8, true> <<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, 8, m_off);
            else               k_ernie55_moe_router_fc1lat_mx8<8, false><<<G, 128, 0, st>>>(x_norm, w.router_w0, w.router_w1, w.fc1_lat.mx8.q, w.fc1_lat.mx8.sc, sc.router_l0, sc.router_l1, sc.routed_in, cnt, m_off);
        };
        if (M <= 8) launch_rf(M, 0);
        else { launch_rf(8, 0); launch_rf(M - 8, 8); }
        k_ernie55_router_topk10<<<M, 256, 0, st>>>(
            sc.router_l0, sc.router_l1, w.router_bias, w.router_sc, sc.topk_idx, sc.topk_w, nullptr, M);
        if (pd) pd->mark(pd->router_ms, st);
    } else {
        moe_route(blas, w, sc, x_norm, M, nullptr, st);
        if (pd) pd->mark(pd->router_ms, st);
        mx8::linear_w(blas, x_norm, w.fc1_lat, sc.routed_in, M, geom::EH, geom::HID, x8_scratch, nullptr, st);
        if (pd) pd->mark(pd->fc1_lat_ms, st);
    }
    const int n_routes = M * geom::TOPK;
    k_ernie55_prerot_x<geom::EHB><<<M * geom::EHB, 32, 0, st>>>(sc.routed_in, w.routed.sg, sc.x_had_r, M);
    if (pd) pd->mark(pd->prerot_r_ms, st);

    if (M == 1) {
        k_ernie55_moe_groups_m1<<<1, 32, 0, st>>>(sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes);
        const int grid_G = geom::TOPK;
        k_ernie55_routed_fc1_grp<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.ptab, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc1_ms, st);
        k_ernie55_routed_fc2_grp<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.ptab, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc2_ms, st);
    } else if (M < 16) {
        k_ernie55_moe_build_groups<8><<<1, geom::NEXP, 0, st>>>(
            sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
        const int grid_G = std::min(n_routes, geom::NEXP + (n_routes + 7) / 8);
        k_ernie55_routed_fc1_grp<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.ptab, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc1_ms, st);
        k_ernie55_routed_fc2_grp<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.ptab, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc2_ms, st);
    } else {
        k_ernie55_moe_build_groups<16><<<1, geom::NEXP, 0, st>>>(
            sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
        const int grid_G = std::min(n_routes, geom::NEXP + (n_routes + 15) / 16);
        k_ernie55_routed_fc1_grp16<<<grid_G * w.routed.eb, 256, 0, st>>>(
            sc.x_had_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            w.routed.tab, w.routed.kmix, w.routed.fc1, w.routed.ptab, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc1_ms, st);
        k_ernie55_routed_fc2_grp16<<<grid_G * geom::EHB, 256, 0, st>>>(
            sc.fc1_act_r, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.routed.tab, w.routed.fc2, w.routed.ptab, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
        if (pd) pd->mark(pd->routed_fc2_ms, st);
    }
    if (!parallel_sh) {
        moe_shared(w, sc, x_norm, M, st);
        if (pd) pd->mark(pd->shared_ms, st);
    }
    k_ernie55_routed_reduce_and_norm<<<M, 256, 0, st>>>(sc.fc2_part_r, w.lat_norm, sc.routed_normed, M);
    mx8::linear_w(blas, sc.routed_normed, w.fc2_lat, sc.routed_up_out, M, geom::HID, geom::EH, x8_scratch, nullptr, st);
    if (parallel_sh) cudaStreamWaitEvent(st, ev_join, 0);
    if (!skip_add && out_part != nullptr) {
        const size_t n = static_cast<size_t>(M) * geom::HID;
        k_ernie55_add_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(
            sc.routed_up_out, sc.shared_out, out_part, n);
    }
    if (pd) pd->mark(pd->finish_ms, st);
}

} // namespace ernie55
