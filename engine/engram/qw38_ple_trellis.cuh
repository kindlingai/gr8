// GPU on-the-fly decode kernel for exlr8 FORMAT v4 Record 9 (Row Trellis) PLE lookup tables.
// Supports pmcgx (E4M3-rounded) and pmcgx-f16 (unrounded FP16) at K4, K4.5, K5 (specialized templates)
// and any valid pair width K0.5..K6 (generic fallback).
//
// Decode math (FORMAT.md "Row trellis"):
//   Row r has C/256 tiles; tile t holds values 256*t..256*t+255 in stream order (no tile_perm).
//   Each 20-bit pair state p emits weights (v[2p], v[2p+1]) via pmcgx; 128-point Sylvester Hadamard
//   runs on each 128-value half; W[r, c] = (H128 v_r)[c] * suh[c] * svh[r] -> BF16 (first `out_cols`
//   columns written, e.g. out_cols = 160 for Qwen 3.8 PLE with C = 256).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace qw38 {

template <bool E4>
__device__ __forceinline__ uint32_t ple_pmcgx_dev(uint32_t w) {
    uint32_t x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const uint32_t x1s = __byte_perm(x1, 0, 0x1032);
    const __half2 s = __hadd2(*reinterpret_cast<const __half2*>(&x0), *reinterpret_cast<const __half2*>(&x1s));
    uint32_t r = *reinterpret_cast<const uint32_t*>(&s);
    if constexpr (E4) {
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    }
    return r;
}

// 128-point Sylvester Hadamard across one 32-lane warp (4 consecutive elements per lane).
__device__ __forceinline__ void ple_warp_fwht128_4(float& v0, float& v1, float& v2, float& v3, int lane) {
    const float a0 = v0 + v1, a1 = v0 - v1;
    const float a2 = v2 + v3, a3 = v2 - v3;
    v0 = a0 + a2; v1 = a1 + a3;
    v2 = a0 - a2; v3 = a1 - a3;
    #pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, s);
        const float p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, s);
        const float p2 = __shfl_xor_sync(0xFFFFFFFFu, v2, s);
        const float p3 = __shfl_xor_sync(0xFFFFFFFFu, v3, s);
        const bool hi = (lane & s) != 0;
        v0 = hi ? (p0 - v0) : (v0 + p0);
        v1 = hi ? (p1 - v1) : (v1 + p1);
        v2 = hi ? (p2 - v2) : (v2 + p2);
        v3 = hi ? (p3 - v3) : (v3 + p3);
    }
}

__device__ __forceinline__ void ple_store_scaled_4(const __half* __restrict__ suh,
                                                   __nv_bfloat16* __restrict__ dst_row,
                                                   int c0, int out_cols, float scale_r,
                                                   float v0, float v1, float v2, float v3) {
    if (c0 + 3 < out_cols) {
        const __half2 s01 = *reinterpret_cast<const __half2*>(suh + c0);
        const __half2 s23 = *reinterpret_cast<const __half2*>(suh + c0 + 2);
        *reinterpret_cast<__nv_bfloat162*>(dst_row + c0) =
            __floats2bfloat162_rn(v0 * __low2float(s01) * scale_r, v1 * __high2float(s01) * scale_r);
        *reinterpret_cast<__nv_bfloat162*>(dst_row + c0 + 2) =
            __floats2bfloat162_rn(v2 * __low2float(s23) * scale_r, v3 * __high2float(s23) * scale_r);
    } else if (c0 < out_cols) {
        const float vs[4] = {v0, v1, v2, v3};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (c0 + j < out_cols)
                dst_row[c0 + j] = __float2bfloat16(vs[j] * __half2float(suh[c0 + j]) * scale_r);
        }
    }
}

// Specialized fast path for fixed K_BITS (8=K4, 9=K4.5, 10=K5).
template <int K_BITS, bool E4, int WARPS_PER_CTA>
__global__ void k_ple_rowtrellis_decode(const uint8_t* __restrict__ codes,
                                        const __half* __restrict__ svh,
                                        const __half* __restrict__ suh,
                                        __nv_bfloat16* __restrict__ out,
                                        int n_rows, int n_tiles, int out_cols) {
    constexpr int NW32 = 4 * K_BITS;
    constexpr int NB = 128 * K_BITS;
    constexpr int TILE_BYTES = 16 * K_BITS;
    constexpr float INV_SQRT128 = 0.08838834764831844f;

    __shared__ uint32_t sm_all[WARPS_PER_CTA][NW32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tid = blockIdx.x * WARPS_PER_CTA + warp;
    if (tid >= n_rows * n_tiles) return;

    const int r = tid / n_tiles;
    const int t = tid - r * n_tiles;
    uint32_t* sm = sm_all[warp];

    const uint32_t* src = reinterpret_cast<const uint32_t*>(codes + (size_t(r) * n_tiles + t) * TILE_BYTES);
    #pragma unroll
    for (int w = lane; w < NW32; w += 32) sm[w] = __ldg(src + w);
    __syncwarp();

    auto decode_pair = [&](int p, float& out0, float& out1) __attribute__((always_inline)) {
        int start = (p + 1) * K_BITS - 20;
        if (start < 0) start += NB;
        const int w0 = start >> 5;
        const int w1 = (w0 + 1 < NW32) ? (w0 + 1) : 0;
        const int off = start & 31;
        const uint64_t m = (uint64_t(sm[w0]) << 32) | uint64_t(sm[w1]);
        const uint32_t st = uint32_t(m >> (44 - off)) & 0xFFFFFu;
        const uint32_t h2u = ple_pmcgx_dev<E4>(st);
        const __half2 h2 = *reinterpret_cast<const __half2*>(&h2u);
        out0 = __low2float(h2);
        out1 = __high2float(h2);
    };

    const float scale_r = INV_SQRT128 * __half2float(__ldg(svh + r));
    __nv_bfloat16* dst_row = out + size_t(r) * out_cols;
    const int col_tile = t * 256;

    if (col_tile < out_cols) {
        float v0, v1, v2, v3;
        decode_pair(2 * lane, v0, v1);
        decode_pair(2 * lane + 1, v2, v3);
        ple_warp_fwht128_4(v0, v1, v2, v3, lane);
        ple_store_scaled_4(suh, dst_row, col_tile + 4 * lane, out_cols, scale_r, v0, v1, v2, v3);
    }
    if (col_tile + 128 < out_cols) {
        float v0, v1, v2, v3;
        decode_pair(64 + 2 * lane, v0, v1);
        decode_pair(64 + 2 * lane + 1, v2, v3);
        ple_warp_fwht128_4(v0, v1, v2, v3, lane);
        ple_store_scaled_4(suh, dst_row, col_tile + 128 + 4 * lane, out_cols, scale_r, v0, v1, v2, v3);
    }
}

// Generic runtime K_BITS fallback for any valid pmcgx width K0.5..K6 (k_bits in 1..12).
template <bool E4, int WARPS_PER_CTA>
__global__ void k_ple_rowtrellis_decode_dyn(const uint8_t* __restrict__ codes,
                                            const __half* __restrict__ svh,
                                            const __half* __restrict__ suh,
                                            __nv_bfloat16* __restrict__ out,
                                            int n_rows, int n_tiles, int out_cols, int k_bits) {
    constexpr float INV_SQRT128 = 0.08838834764831844f;
    const int nw32 = 4 * k_bits;
    const int nb = 128 * k_bits;
    const int tile_bytes = 16 * k_bits;

    __shared__ uint32_t sm_all[WARPS_PER_CTA][48]; // max nw32 = 4 * 12 = 48 (K6)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tid = blockIdx.x * WARPS_PER_CTA + warp;
    if (tid >= n_rows * n_tiles) return;

    const int r = tid / n_tiles;
    const int t = tid - r * n_tiles;
    uint32_t* sm = sm_all[warp];

    const uint32_t* src = reinterpret_cast<const uint32_t*>(codes + (size_t(r) * n_tiles + t) * tile_bytes);
    for (int w = lane; w < nw32; w += 32) sm[w] = __ldg(src + w);
    __syncwarp();

    auto decode_pair = [&](int p, float& out0, float& out1) __attribute__((always_inline)) {
        int start = (p + 1) * k_bits - 20;
        if (start < 0) start += nb;
        const int w0 = start >> 5;
        const int w1 = (w0 + 1 < nw32) ? (w0 + 1) : 0;
        const int off = start & 31;
        const uint64_t m = (uint64_t(sm[w0]) << 32) | uint64_t(sm[w1]);
        const uint32_t st = uint32_t(m >> (44 - off)) & 0xFFFFFu;
        const uint32_t h2u = ple_pmcgx_dev<E4>(st);
        const __half2 h2 = *reinterpret_cast<const __half2*>(&h2u);
        out0 = __low2float(h2);
        out1 = __high2float(h2);
    };

    const float scale_r = INV_SQRT128 * __half2float(__ldg(svh + r));
    __nv_bfloat16* dst_row = out + size_t(r) * out_cols;
    const int col_tile = t * 256;

    if (col_tile < out_cols) {
        float v0, v1, v2, v3;
        decode_pair(2 * lane, v0, v1);
        decode_pair(2 * lane + 1, v2, v3);
        ple_warp_fwht128_4(v0, v1, v2, v3, lane);
        ple_store_scaled_4(suh, dst_row, col_tile + 4 * lane, out_cols, scale_r, v0, v1, v2, v3);
    }
    if (col_tile + 128 < out_cols) {
        float v0, v1, v2, v3;
        decode_pair(64 + 2 * lane, v0, v1);
        decode_pair(64 + 2 * lane + 1, v2, v3);
        ple_warp_fwht128_4(v0, v1, v2, v3, lane);
        ple_store_scaled_4(suh, dst_row, col_tile + 128 + 4 * lane, out_cols, scale_r, v0, v1, v2, v3);
    }
}

template <bool E4>
inline void ple_rowtrellis_decode_e4(const uint8_t* codes, const __half* svh, const __half* suh,
                                     __nv_bfloat16* out, int n_rows, int n_tiles, int out_cols,
                                     int k_bits, cudaStream_t st) {
    const int tot = n_rows * n_tiles;
    if (tot <= 0) return;
    if (tot <= 256) {
        switch (k_bits) {
            case 8:  k_ple_rowtrellis_decode<8, E4, 1><<<tot, 32, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            case 9:  k_ple_rowtrellis_decode<9, E4, 1><<<tot, 32, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            case 10: k_ple_rowtrellis_decode<10, E4, 1><<<tot, 32, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            default: k_ple_rowtrellis_decode_dyn<E4, 1><<<tot, 32, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols, k_bits); return;
        }
    } else {
        const int grid = (tot + 3) / 4;
        switch (k_bits) {
            case 8:  k_ple_rowtrellis_decode<8, E4, 4><<<grid, 128, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            case 9:  k_ple_rowtrellis_decode<9, E4, 4><<<grid, 128, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            case 10: k_ple_rowtrellis_decode<10, E4, 4><<<grid, 128, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols); return;
            default: k_ple_rowtrellis_decode_dyn<E4, 4><<<grid, 128, 0, st>>>(codes, svh, suh, out, n_rows, n_tiles, out_cols, k_bits); return;
        }
    }
}

// Decode `n_rows` compressed row-trellis rows on `st` directly into `out` [n_rows, out_cols] BF16.
inline void ple_rowtrellis_decode(const uint8_t* codes, const __half* svh, const __half* suh,
                                  __nv_bfloat16* out, int n_rows, int n_tiles, int out_cols,
                                  int k_bits, bool e4m3, cudaStream_t st) {
    if (e4m3) ple_rowtrellis_decode_e4<true>(codes, svh, suh, out, n_rows, n_tiles, out_cols, k_bits, st);
    else      ple_rowtrellis_decode_e4<false>(codes, svh, suh, out, n_rows, n_tiles, out_cols, k_bits, st);
}

} // namespace qw38
