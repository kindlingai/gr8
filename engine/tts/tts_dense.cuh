// Native mixed-format dense linear dispatch for Drama 3 (g6tts --weight-mode native).
// Reuses `engine/embed/embed_gemm.cuh` kernels directly on BF16 activations (zero FP16<->BF16 conversion
// overhead!) for Trellis K4 (mcg), INT4, and BF16 without modifying `embed_gemm.cuh`, and adds Drama 3-specific
// FP8_ROW -> FP32/BF16 kernels (supporting arbitrary/odd N such as 4097-row sliced `lm_head` and 1024-row
// residual codebook heads) plus FP8_ROW backbone embedding gather.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../embed/embed_gemm.cuh"
#include "tts_config.h"

namespace g6tts {

using bf16 = __nv_bfloat16;

// Exact round-to-BF16 in FP32 value space (matches upstream fish-speech BF16 tensor ops).
__device__ __forceinline__ float tts_rbf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// ------------------------------------------------------------------------------------------------ Store & scaling helpers

__global__ void k_tts_bf16_to_f32_store(const bf16* __restrict__ src, float* __restrict__ dst, size_t n, int add_res) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(src[i]);
    if (add_res) dst[i] += v;
    else dst[i] = v;
}

__global__ void k_tts_bf16_to_f32_store_scaled(const bf16* __restrict__ src,
                                               const float* __restrict__ row_scale,
                                               float* __restrict__ dst,
                                               int M, int N, int add_res) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = size_t(M) * size_t(N);
    if (i >= n) return;
    const int m = static_cast<int>(i / size_t(N));
    const float v = __bfloat162float(src[i]) * row_scale[m];
    if (add_res) dst[i] += v;
    else dst[i] = v;
}

// BF16 residual update: dst[i] = add_res ? rbf16(dst[i] + bf16(src[i])) : bf16(src[i]).
__global__ void k_tts_res_update_bf16(const bf16* __restrict__ src, float* __restrict__ dst, size_t n, int add_res) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(src[i]);
    dst[i] = add_res ? tts_rbf16(dst[i] + v) : v;
}

// Lossless BF16 <-> FP32 conversion helpers for RefF32 exact-weight verification mode.
__global__ void k_tts_bf16_to_f32_buf(const bf16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

__global__ void k_tts_f32_to_bf16_buf(const float* __restrict__ src, bf16* __restrict__ dst, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16_rn(src[i]);
}

// Bit-exact Parseval-bounded (`suh`-aware) power-of-two per-row dynamic scaling for Trellis K4 activations:
// By Parseval's theorem, ||H128(u)||_2 = ||u||_2, so max_j |H128(u)_j| <= ||u||_2 for every 128-element block u.
// Measuring bm = max_{kb} ||X[m, kb*128..] * max(|suh0..2|)||_2 guarantees ZERO FP16 overflow when bm <= 32768
// while leaving normal/moderate rows completely unscaled (s=1) so small channels never underflow into FP16 subnormals!
// Grid: (M), Block: 256 (8 warps; each warp reduces full 128-element blocks).
__global__ void k_tts_row_scale_pow2_bf16(const bf16* __restrict__ X,
                                          const __half* __restrict__ suh0,
                                          const __half* __restrict__ suh1,
                                          const __half* __restrict__ suh2,
                                          bf16* __restrict__ X_scaled,
                                          float* __restrict__ row_scale,
                                          int M, int K) {
    __shared__ float red[8];
    __shared__ float s_inv;
    const int m = blockIdx.x;
    if (m >= M) return;
    const bf16* xr = X + size_t(m) * K;
    bf16* yr = X_scaled + size_t(m) * K;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nblk = K >> 7;
    float max_l2_sq = 0.0f;
    for (int kb = warp; kb < nblk; kb += 8) {
        float bsq = 0.0f;
        const int base = kb * 128 + lane * 4;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int k = base + e;
            const float xv = __bfloat162float(xr[k]);
            float sm = suh0 ? fabsf(__half2float(suh0[k])) : 1.0f;
            if (suh1) sm = fmaxf(sm, fabsf(__half2float(suh1[k])));
            if (suh2) sm = fmaxf(sm, fabsf(__half2float(suh2[k])));
            const float t = xv * sm;
            bsq += t * t;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) bsq += __shfl_xor_sync(0xffffffffu, bsq, off);
        max_l2_sq = fmaxf(max_l2_sq, bsq);
    }
    if (lane == 0) red[warp] = max_l2_sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        float bsq_max = (threadIdx.x < 8) ? red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) bsq_max = fmaxf(bsq_max, __shfl_xor_sync(0xffffffffu, bsq_max, off));
        if (threadIdx.x == 0) {
            const float bm = sqrtf(bsq_max);
            if (bm > 32768.0f && isfinite(bm)) {
                int exp_i = 0;
                (void)frexpf(bm, &exp_i);
                // bm in [0.5, 1) * 2^exp_i => p = exp_i - 15 puts bm / 2^p in [16384.0, 32768.0) (< 65504 FP16 max)
                const int p = exp_i - 15;
                row_scale[m] = ldexpf(1.0f, p);
                s_inv = ldexpf(1.0f, -p);
            } else if (bm > 0.0f && bm < 0.0625f && isfinite(bm)) {
                int exp_i = 0;
                (void)frexpf(bm, &exp_i);
                const int p = exp_i - 1;
                row_scale[m] = ldexpf(1.0f, p);
                s_inv = ldexpf(1.0f, -p);
            } else {
                row_scale[m] = 1.0f;
                s_inv = 1.0f;
            }
        }
    }
    __syncthreads();
    const float inv = s_inv;
    for (int k = threadIdx.x; k < K; k += 256) {
        yr[k] = __float2bfloat16_rn(__bfloat162float(xr[k]) * inv);
    }
}

// Bit-exact power-of-two per-row unscaling in-place on BF16 output Y[M, N].
__global__ void k_tts_row_unscale_bf16(bf16* __restrict__ Y,
                                       const float* __restrict__ row_scale,
                                       int M, int N) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = size_t(M) * size_t(N);
    if (i >= n) return;
    const int m = static_cast<int>(i / size_t(N));
    Y[i] = __float2bfloat16_rn(__bfloat162float(Y[i]) * row_scale[m]);
}

// True FP32-output Split-K reducer + H128 * svh * row_scale (+ optional FP32 residual add) for Trellis K4.
// Eliminates intermediate BF16 output quantization on >65504 residual projections (e.g. tower `w2`).
// Grid: (NB = N / 128, ceil(M / 8)), Block: 256 (8 warps, 1 warp per output row).
__global__ void __launch_bounds__(256) k_tts_splitk_reduce_f32_scaled(
    const float* __restrict__ partials,
    const __half* __restrict__ svh,
    const float* __restrict__ row_scale,
    float* __restrict__ y,
    int M, int N, int ks_parts, int add_res) {
    const int bx = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.y * 8 + warp;
    if (row >= M) return;

    const int c0 = bx * 128;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    const float* base = partials + size_t(bx) * ks_parts * size_t(M) * 128 + size_t(row) * 128 + lane * 4;
    for (int kp = 0; kp < ks_parts; ++kp) {
        const float4 p = *reinterpret_cast<const float4*>(base + size_t(kp) * size_t(M) * 128);
        v[0] += p.x; v[1] += p.y; v[2] += p.z; v[3] += p.w;
    }
    g6emb::emb_had128(v, lane);
    const uint2 sr = *reinterpret_cast<const uint2*>(svh + c0 + lane * 4);
    const float2 s01 = __half22float2(g6emb::emb_as_h2(sr.x));
    const float2 s23 = __half22float2(g6emb::emb_as_h2(sr.y));
    const float rsc = row_scale[row];
    v[0] = v[0] * s01.x * rsc;
    v[1] = v[1] * s01.y * rsc;
    v[2] = v[2] * s23.x * rsc;
    v[3] = v[3] * s23.y * rsc;
    float* dst = y + size_t(row) * N + c0 + lane * 4;
    if (add_res) {
        float4 cur = *reinterpret_cast<const float4*>(dst);
        cur.x += v[0]; cur.y += v[1]; cur.z += v[2]; cur.w += v[3];
        *reinterpret_cast<float4*>(dst) = cur;
    } else {
        *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
    }
}

// Exact FP32-dequant, FP32-output INT4 g32 GEMM: Y[M, N] = (add_res ? Y : 0) + X[M, K] @ dequant_f32(q4, s4)^T.
// Avoids both BF16 weight rounding and BF16 output quantization on residual projections (e.g. o_proj, out_proj, sh0.w2).
// Grid: (ceil(N / 4), ceil(M / 8)), Block: 128 (4 warps, 1 warp per output row).
__global__ void __launch_bounds__(128) k_tts_int4_gemm_f32(
    const bf16* __restrict__ X,
    const uint8_t* __restrict__ q4,
    const __half* __restrict__ s4,
    float* __restrict__ Y,
    int M, int N, int K, int add_res) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    const int m0 = blockIdx.y * 8;
    if (n >= N || m0 >= M) return;
    const int mcnt = min(8, M - m0);

    const int K32 = K >> 5;
    const uint4* qrow = reinterpret_cast<const uint4*>(q4 + size_t(n) * (K >> 1));
    const __half* srow = s4 + size_t(n) * K32;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = lane; g < K32; g += 32) {
        const float sc = __half2float(srow[g]);
        const uint4 qv = __ldg(qrow + g);
        const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
        float wf[32];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t word = qw[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t byte = (word >> (b * 8)) & 0xffu;
                wf[w * 8 + b * 2 + 0] = static_cast<float>(static_cast<int>(byte & 15u) - 8);
                wf[w * 8 + b * 2 + 1] = static_cast<float>(static_cast<int>(byte >> 4) - 8);
            }
        }
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < mcnt) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + size_t(m0 + m) * K + size_t(g) * 32);
                float gs = 0.f;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint4 xv = xm[u];
                    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
                    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
                    const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.z));
                    const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.w));
                    const int base = u * 8;
                    gs = fmaf(f0.x, wf[base + 0], fmaf(f0.y, wf[base + 1], gs));
                    gs = fmaf(f1.x, wf[base + 2], fmaf(f1.y, wf[base + 3], gs));
                    gs = fmaf(f2.x, wf[base + 4], fmaf(f2.y, wf[base + 5], gs));
                    gs = fmaf(f3.x, wf[base + 6], fmaf(f3.y, wf[base + 7], gs));
                }
                acc[m] = fmaf(gs, sc, acc[m]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m < mcnt) {
            float v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) {
                const size_t idx = size_t(m0 + m) * N + n;
                if (add_res) Y[idx] += v;
                else Y[idx] = v;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ FP8_ROW kernels

// Native FP8_ROW backbone embedding lookup + semantic codebook c1 mix (/ sqrt(2)) into BF16-rounded FP32 residual.
// Grid: (M), Block: 256.
__global__ void k_backbone_embed_fp8(const int32_t* __restrict__ rows,
                                     const uint8_t* __restrict__ embed_fp8,
                                     const float* __restrict__ embed_scale,
                                     const bf16* __restrict__ cb_embed,
                                     float* __restrict__ out_f32,
                                     int M, int H, int rw,
                                     int sem0, int sem1, int cb_vocab) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int32_t* row = rows + size_t(m) * rw;
    const int main_id = row[0];
    const bool is_sem = (main_id >= sem0 && main_id <= sem1);
    const int c1 = is_sem ? max(0, min(row[2], cb_vocab - 1)) : 0;
    const uint8_t* te_q = embed_fp8 + size_t(main_id) * H;
    const float sc = embed_scale[main_id];
    const bf16* ce = cb_embed + size_t(cb_vocab + c1) * H;
    float* dst = out_f32 + size_t(m) * H;
    constexpr float inv_sqrt2 = 0.7071067811865475f;

    for (int d = threadIdx.x * 2; d < H; d += blockDim.x * 2) {
        const uint16_t raw = *reinterpret_cast<const uint16_t*>(te_q + d);
        uint32_t h2 = 0;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
        const float2 fv = __half22float2(*reinterpret_cast<const __half2*>(&h2));
        float v0 = fv.x * sc;
        float v1 = fv.y * sc;
        if (is_sem) {
            v0 = (v0 + __bfloat162float(ce[d])) * inv_sqrt2;
            v1 = (v1 + __bfloat162float(ce[d + 1])) * inv_sqrt2;
        }
        dst[d] = tts_rbf16(v0);
        dst[d + 1] = tts_rbf16(v1);
    }
}

// Exact FP32 backbone embedding lookup (for WeightMode::RefF32 verification).
__global__ void k_backbone_embed_f32(const int32_t* __restrict__ rows,
                                     const float* __restrict__ embed_f32,
                                     const bf16* __restrict__ cb_embed,
                                     float* __restrict__ out_f32,
                                     int M, int H, int rw,
                                     int sem0, int sem1, int cb_vocab) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int32_t* row = rows + size_t(m) * rw;
    const int main_id = row[0];
    const bool is_sem = (main_id >= sem0 && main_id <= sem1);
    const int c1 = is_sem ? max(0, min(row[2], cb_vocab - 1)) : 0;
    const float* te = embed_f32 + size_t(main_id) * H;
    const bf16* ce = cb_embed + size_t(cb_vocab + c1) * H;
    float* dst = out_f32 + size_t(m) * H;
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float v = te[d];
        if (is_sem) {
            v = (v + __bfloat162float(ce[d])) * inv_sqrt2;
        }
        dst[d] = tts_rbf16(v);
    }
}

// Exact FP8_ROW GEMM: Y[M, N] (FP32) = (add_res ? Y : 0) + X[M, K] (BF16) @ (Wq[N, K] * Wsc[N])^T.
// Supports arbitrary N (including odd N = 4097 for sliced lm_head, N = 1024 for residual heads, N = 1)
// and any K divisible by 16. Grid: (ceil(N / 4), ceil(M / 8)), Block: 128 (4 warps, 1 warp per output row).
__global__ void __launch_bounds__(128) k_tts_fp8_row_gemm_f32(
    const bf16* __restrict__ X,
    const uint8_t* __restrict__ Wq,
    const float* __restrict__ Wsc,
    float* __restrict__ Y,
    int M, int N, int K, int add_res) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    const int m0 = blockIdx.y * 8;
    if (n >= N || m0 >= M) return;
    const int mcnt = min(8, M - m0);

    const uint4* wrow = reinterpret_cast<const uint4*>(Wq + size_t(n) * K);
    const int K16 = K >> 4;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int i = lane; i < K16; i += 32) {
        const uint4 wv = __ldg(wrow + i);
        const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        float2 wf[8];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint16_t lo = static_cast<uint16_t>(words[w] & 0xffffu);
            const uint16_t hi = static_cast<uint16_t>(words[w] >> 16);
            uint32_t h0 = 0, h1 = 0;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(lo));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(hi));
            wf[w * 2 + 0] = __half22float2(*reinterpret_cast<const __half2*>(&h0));
            wf[w * 2 + 1] = __half22float2(*reinterpret_cast<const __half2*>(&h1));
        }

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < mcnt) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + size_t(m0 + m) * K);
                const uint4 x0 = xm[2 * i + 0];
                const uint4 x1 = xm[2 * i + 1];
                const uint32_t xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                float s = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[u]));
                    s = fmaf(xv.x, wf[u].x, fmaf(xv.y, wf[u].y, s));
                }
                acc[m] += s;
            }
        }
    }

    const float sc = Wsc[n];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m < mcnt) {
            float v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) {
                const size_t idx = size_t(m0 + m) * N + n;
                if (add_res) Y[idx] += v * sc;
                else Y[idx] = v * sc;
            }
        }
    }
}

// Exact FP8_ROW GEMM with BF16 output (for non-128-aligned dense projections if any fall back to FP8_ROW).
__global__ void __launch_bounds__(128) k_tts_fp8_row_gemm_bf16(
    const bf16* __restrict__ X,
    const uint8_t* __restrict__ Wq,
    const float* __restrict__ Wsc,
    bf16* __restrict__ Y,
    int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    const int m0 = blockIdx.y * 8;
    if (n >= N || m0 >= M) return;
    const int mcnt = min(8, M - m0);

    const uint4* wrow = reinterpret_cast<const uint4*>(Wq + size_t(n) * K);
    const int K16 = K >> 4;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int i = lane; i < K16; i += 32) {
        const uint4 wv = __ldg(wrow + i);
        const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        float2 wf[8];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint16_t lo = static_cast<uint16_t>(words[w] & 0xffffu);
            const uint16_t hi = static_cast<uint16_t>(words[w] >> 16);
            uint32_t h0 = 0, h1 = 0;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(lo));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(hi));
            wf[w * 2 + 0] = __half22float2(*reinterpret_cast<const __half2*>(&h0));
            wf[w * 2 + 1] = __half22float2(*reinterpret_cast<const __half2*>(&h1));
        }

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < mcnt) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + size_t(m0 + m) * K);
                const uint4 x0 = xm[2 * i + 0];
                const uint4 x1 = xm[2 * i + 1];
                const uint32_t xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                float s = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[u]));
                    s = fmaf(xv.x, wf[u].x, fmaf(xv.y, wf[u].y, s));
                }
                acc[m] += s;
            }
        }
    }

    const float sc = Wsc[n];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m < mcnt) {
            float v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) Y[size_t(m0 + m) * N + n] = __float2bfloat16_rn(v * sc);
        }
    }
}

// ------------------------------------------------------------------------------------------------ Workspace & dispatch

struct DenseWorkspace {
    bool initialized = false;
    g6emb::GemmWorkspace emb_ws;
    bf16* d_y0_bf16 = nullptr;   // [max_M, 9216] scratch for Trellis/INT4 -> FP32 store
    bf16* d_x_scaled = nullptr;  // [max_M, 6144] power-of-two scaled activations for FP16-safe Trellis K4
    float* d_row_scale = nullptr;// [max_M] exact power-of-two row scales
    size_t scratch_bytes = 0;

    ~DenseWorkspace() { release(); }

    void release() {
        if (d_y0_bf16) { cudaFree(d_y0_bf16); d_y0_bf16 = nullptr; }
        if (d_x_scaled) { cudaFree(d_x_scaled); d_x_scaled = nullptr; }
        if (d_row_scale) { cudaFree(d_row_scale); d_row_scale = nullptr; }
        emb_ws.release();
        scratch_bytes = 0;
        initialized = false;
    }

    void init(int max_M) {
        release();
        const int M_cap = std::max(16, max_M);
        // Reserve 3x x_rot for fused triple Trellis K4 GEMMs (Q+K+V) up to K=6144.
        emb_ws.init(M_cap * 3, 6144, size_t(8) << 20);
        const size_t nb_y = size_t(M_cap) * 9216 * sizeof(bf16);
        const size_t nb_x = size_t(M_cap) * 6144 * sizeof(bf16);
        const size_t nb_s = size_t(M_cap) * sizeof(float);
        if (cudaMalloc(&d_y0_bf16, nb_y) != cudaSuccess ||
            cudaMalloc(&d_x_scaled, nb_x) != cudaSuccess ||
            cudaMalloc(&d_row_scale, nb_s) != cudaSuccess) {
            release();
            throw std::runtime_error("g6tts DenseWorkspace: cudaMalloc failed");
        }
        scratch_bytes = nb_y + nb_x + nb_s;
        initialized = true;
    }
};

// Native linear projection with BF16 output: Y[M, W.out] = X[M, W.inp] @ W^T.
inline void tts_native_linear_bf16(const bf16* X, const g6emb::DevMatrix& W, bf16* Y,
                                   int M, DenseWorkspace& ws, cublasHandle_t blas, cudaStream_t st) {
    if (M <= 0) return;
    if (W.fmt == g6emb::FMT_FP8_ROW) {
        const dim3 grid((W.out + 3) / 4, (M + 7) / 8);
        k_tts_fp8_row_gemm_bf16<<<grid, 128, 0, st>>>(X, W.data, W.scale, Y, M, W.out, W.inp);
        return;
    }
    if (W.fmt == g6emb::FMT_BF16) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSetStream(blas, st);
        cublasStatus_t s = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, W.out, M, W.inp,
                                        &alpha, W.bf16, CUDA_R_16BF, W.inp,
                                        X, CUDA_R_16BF, W.inp,
                                        &beta, Y, CUDA_R_16BF, W.out,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("tts_native_linear_bf16: cublasGemmEx failed");
        return;
    }
    if (W.fmt == g6emb::FMT_TRELLIS) {
        k_tts_row_scale_pow2_bf16<<<M, 256, 0, st>>>(X, W.suh, nullptr, nullptr, ws.d_x_scaled, ws.d_row_scale, M, W.inp);
        g6emb::linear_forward(ws.d_x_scaled, W, nullptr, Y, M, false, ws.emb_ws, st);
        const size_t y_n = size_t(M) * W.out;
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((y_n + 255) / 256), 256, 0, st>>>(Y, ws.d_row_scale, M, W.out);
        return;
    }
    g6emb::linear_forward(X, W, nullptr, Y, M, false, ws.emb_ws, st);
}

// Native linear projection with FP32 output (and optional residual add): Y[M, out_rows] = (add_res ? Y : 0) + X @ W^T.
// Supports slicing `out_rows <= W.out` (e.g. 1024 for residual heads or 4097 for lm_head).
inline void tts_native_linear_f32(const bf16* X, const g6emb::DevMatrix& W, float* Y,
                                  int M, int out_rows, bool add_res,
                                  DenseWorkspace& ws, cublasHandle_t blas, cudaStream_t st) {
    if (M <= 0) return;
    const int N = (out_rows > 0) ? std::min(out_rows, W.out) : W.out;
    if (W.fmt == g6emb::FMT_FP8_ROW) {
        const dim3 grid((N + 3) / 4, (M + 7) / 8);
        k_tts_fp8_row_gemm_f32<<<grid, 128, 0, st>>>(X, W.data, W.scale, Y, M, N, W.inp, add_res ? 1 : 0);
        return;
    }
    if (W.fmt == g6emb::FMT_BF16) {
        const float alpha = 1.0f, beta = add_res ? 1.0f : 0.0f;
        cublasSetStream(blas, st);
        cublasStatus_t s = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, W.inp,
                                        &alpha, W.bf16, CUDA_R_16BF, W.inp,
                                        X, CUDA_R_16BF, W.inp,
                                        &beta, Y, CUDA_R_32F, N,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("tts_native_linear_f32: cublasGemmEx failed");
        return;
    }
    // Trellis K4 / INT4: direct true-FP32 output kernels (zero intermediate BF16 output quantization).
    if (N != W.out) {
        throw std::runtime_error("tts_native_linear_f32: partial row slice unsupported for format " + std::to_string(W.fmt));
    }
    if (W.fmt == g6emb::FMT_TRELLIS) {
        k_tts_row_scale_pow2_bf16<<<M, 256, 0, st>>>(X, W.suh, nullptr, nullptr, ws.d_x_scaled, ws.d_row_scale, M, W.inp);
        const int NB = W.out / 128;
        const int nst = std::max(1, W.inp / 32);
        // Chunk M in slices of <= 64 rows so k_emb_tr_h16_splitk (BM=4 -> 64 rows) produces exact FP32 partials.
        for (int r0 = 0; r0 < M; r0 += 64) {
            const int rows = std::min(64, M - r0);
            int ks = (W.inp >= 4096) ? 6 : 4;
            if (NB * ks > 128) ks = std::max(1, 96 / std::max(1, NB));
            ks = std::min(ks, std::max(1, nst / 4));
            const size_t max_ks = std::max<size_t>(1, ws.emb_ws.acc_floats / (size_t(NB) * size_t(rows) * 128));
            ks = std::max(1, std::min(ks, static_cast<int>(max_ks)));
            g6emb::launch_rot_h16(ws.d_x_scaled + size_t(r0) * W.inp, W.suh, rows, W.inp, ws.emb_ws, st);
            g6emb::k_emb_tr_h16_splitk<<<dim3(NB, ks), g6emb::EmbTrH16Cfg<4>::NTH, g6emb::EmbTrH16Cfg<4>::SMEM, st>>>(
                ws.emb_ws.x_rot, W.data, ws.emb_ws.f32_acc, rows, W.inp, W.sB, W.sG, ks);
            k_tts_splitk_reduce_f32_scaled<<<dim3(NB, (rows + 7) / 8), 256, 0, st>>>(
                ws.emb_ws.f32_acc, W.svh, ws.d_row_scale + r0, Y + size_t(r0) * W.out, rows, W.out, ks, add_res ? 1 : 0);
        }
        return;
    }
    if (W.fmt == g6emb::FMT_INT4) {
        const dim3 grid((W.out + 3) / 4, (M + 7) / 8);
        k_tts_int4_gemm_f32<<<grid, 128, 0, st>>>(X, W.data, W.s4, Y, M, W.out, W.inp, add_res ? 1 : 0);
        return;
    }
    const size_t y_n = size_t(M) * W.out;
    g6emb::linear_forward(X, W, nullptr, ws.d_y0_bf16, M, false, ws.emb_ws, st);
    k_tts_bf16_to_f32_store<<<static_cast<unsigned>((y_n + 255) / 256), 256, 0, st>>>(
        ws.d_y0_bf16, Y, y_n, add_res ? 1 : 0);
}

// Fused dual native projection (e.g. w1+w3 or in_proj_qkv+in_proj_z): reads X once via k_emb_rot_h16_multi<2>.
inline void tts_native_linear_dual_bf16(const bf16* X,
                                        const g6emb::DevMatrix& W0, const g6emb::DevMatrix& W1,
                                        bf16* Y0, bf16* Y1, int M,
                                        DenseWorkspace& ws, cublasHandle_t blas, cudaStream_t st) {
    if (M <= 0) return;
    if (W0.fmt == g6emb::FMT_TRELLIS && W1.fmt == g6emb::FMT_TRELLIS && W0.inp == W1.inp) {
        k_tts_row_scale_pow2_bf16<<<M, 256, 0, st>>>(X, W0.suh, W1.suh, nullptr, ws.d_x_scaled, ws.d_row_scale, M, W0.inp);
        g6emb::linear_forward_dual(ws.d_x_scaled, W0, W1, Y0, Y1, M, ws.emb_ws, st);
        const size_t n0 = size_t(M) * W0.out;
        const size_t n1 = size_t(M) * W1.out;
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((n0 + 255) / 256), 256, 0, st>>>(Y0, ws.d_row_scale, M, W0.out);
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((n1 + 255) / 256), 256, 0, st>>>(Y1, ws.d_row_scale, M, W1.out);
        return;
    }
    tts_native_linear_bf16(X, W0, Y0, M, ws, blas, st);
    tts_native_linear_bf16(X, W1, Y1, M, ws, blas, st);
}

// Fused triple native projection (FullAttn Q+K+V): reads X once via k_emb_rot_h16_multi<3>.
inline void tts_native_linear_triple_bf16(const bf16* X,
                                          const g6emb::DevMatrix& W0, const g6emb::DevMatrix& W1, const g6emb::DevMatrix& W2,
                                          bf16* Y0, bf16* Y1, bf16* Y2, int M,
                                          DenseWorkspace& ws, cublasHandle_t blas, cudaStream_t st) {
    if (M <= 0) return;
    if (W0.fmt == g6emb::FMT_TRELLIS && W1.fmt == g6emb::FMT_TRELLIS && W2.fmt == g6emb::FMT_TRELLIS &&
        W0.inp == W1.inp && W0.inp == W2.inp) {
        k_tts_row_scale_pow2_bf16<<<M, 256, 0, st>>>(X, W0.suh, W1.suh, W2.suh, ws.d_x_scaled, ws.d_row_scale, M, W0.inp);
        g6emb::linear_forward_triple(ws.d_x_scaled, W0, W1, W2, Y0, Y1, Y2, M, ws.emb_ws, st);
        const size_t n0 = size_t(M) * W0.out;
        const size_t n1 = size_t(M) * W1.out;
        const size_t n2 = size_t(M) * W2.out;
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((n0 + 255) / 256), 256, 0, st>>>(Y0, ws.d_row_scale, M, W0.out);
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((n1 + 255) / 256), 256, 0, st>>>(Y1, ws.d_row_scale, M, W1.out);
        k_tts_row_unscale_bf16<<<static_cast<unsigned>((n2 + 255) / 256), 256, 0, st>>>(Y2, ws.d_row_scale, M, W2.out);
        return;
    }
    tts_native_linear_bf16(X, W0, Y0, M, ws, blas, st);
    tts_native_linear_bf16(X, W1, Y1, M, ws, blas, st);
    tts_native_linear_bf16(X, W2, Y2, M, ws, blas, st);
}

} // namespace g6tts
