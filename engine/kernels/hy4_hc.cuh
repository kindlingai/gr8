// Tencent Hy4-flash hyper-connections (non-combined iHC), RMSNorm, and BF16 GEMV/GEMM helpers.
// Verified verbatim against HF PR 50903 modeling_hy_v4_flash.py:
//   - HYV4FlashRMSNorm: weight * bf16(x_f32 * rsqrt(mean(x_f32^2) + eps))  (standard T5 RMSNorm, NOT 1+w).
//   - HYV4FlashHyperConnection (forced FP32):
//       flat   = hyper.reshape(M, 12288).float()
//       inv    = rsqrt(mean(flat^2) + 1e-5)                   (HYV4FlashUnweightedRMSNorm, eps = rms_norm_eps)
//       mixes  = (flat @ fn.T) * inv                          [M, 8] FP32
//       pre    = sigmoid(mixes[:, 0:4] * scale[0] + base[0:4]) + 1e-6
//       post   = 2.0 * sigmoid(mixes[:, 4:8] * scale[1] + base[4:8]) + 1e-6
//       out    = bf16(sum_{s=0..3} pre[:, s] * hyper[:, s, :])
//     Sublayer residual consume (after TP AllReduce + sandwich norm):
//       hyper[:, s, :] = bf16(post[:, s] * sublayer_out + hyper[:, s, :])
//   - HYV4FlashHyperHead (forced FP32, before final model.norm):
//       mixes  = (flat @ hc_head_fn.T) * inv                  [M, 4] FP32
//       pre    = sigmoid(mixes * hc_head_scale[0] + hc_head_base) + 1e-6
//       out    = bf16(sum_{s=0..3} pre[:, s] * hyper[:, s, :])
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include "../model_geom_hy4.h"

namespace hy4 {

struct StepSeg {
    int slot = 0;
    int row0 = 0;
    int pos0 = 0;
    int len  = 0;
};

struct HcWeights {
    const float* fn    = nullptr; // [8, 12288] FP32
    const float* base  = nullptr; // [8] FP32
    const float* scale = nullptr; // [2] FP32
};

struct HcHeadWeights {
    const float* fn    = nullptr; // [4, 12288] FP32
    const float* base  = nullptr; // [4] FP32
    const float* scale = nullptr; // [1] FP32
};

struct HcScratch {
    float* flat_f32  = nullptr; // [max_M, 12288] FP32 (for prefill SGEMM path M > 16)
    float* mixes     = nullptr; // [max_M, 8] FP32
    float* post      = nullptr; // [max_M, 4] FP32
};

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// BF16 row-major GEMM: C[M, N] = A[M, K] @ B[N, K]^T (FP32 accumulate).
inline cublasStatus_t gemm_bf16_nt(cublasHandle_t h, int M, int N, int K,
                                   const __nv_bfloat16* A, const __nv_bfloat16* B,
                                   __nv_bfloat16* C, float alpha = 1.0f, float beta = 0.0f) {
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
                        &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

inline cublasStatus_t gemm_bf16_nt_f32(cublasHandle_t h, int M, int N, int K,
                                       const __nv_bfloat16* A, const __nv_bfloat16* B,
                                       float* C, float alpha = 1.0f, float beta = 0.0f) {
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
                        &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// FP32 row-major GEMM: C[M, N] = A[M, K] @ B[N, K]^T (exact FP32, no TF32 truncation).
inline cublasStatus_t gemm_f32_nt(cublasHandle_t h, int M, int N, int K,
                                  const float* A, const float* B,
                                  float* C, float alpha = 1.0f, float beta = 0.0f) {
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        &alpha, B, CUDA_R_32F, K, A, CUDA_R_32F, K,
                        &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
}

__device__ __forceinline__ float dot_bf16x8(uint4 a, uint4 b) {
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(pa[i]);
        const float2 fb = __bfloat1622float2(pb[i]);
        s = fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, s));
    }
    return s;
}

// Fast coalesced BF16 GEMV for small decode batches (M <= 12):
//   C[M, N] = A[M, K] @ B[N, K]^T + optional bias[N] (K must be multiple of 8).
template <int MAX_M, bool OUT_F32 = false, bool HAS_BIAS = false>
__global__ void __launch_bounds__(128) k_hy4_bf16_gemv_smallm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    void* __restrict__ C,
    int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;
    const int K8 = K >> 3;
    const uint4* brow = reinterpret_cast<const uint4*>(B + static_cast<size_t>(n) * K);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    int i = lane;
    for (; i + 96 < K8; i += 128) {
        const uint4 b0 = __ldg(brow + i);
        const uint4 b1 = __ldg(brow + i + 32);
        const uint4 b2 = __ldg(brow + i + 64);
        const uint4 b3 = __ldg(brow + i + 96);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* am = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K);
                acc[m] += dot_bf16x8(am[i], b0) + dot_bf16x8(am[i + 32], b1)
                        + dot_bf16x8(am[i + 64], b2) + dot_bf16x8(am[i + 96], b3);
            }
        }
    }
    for (; i < K8; i += 32) {
        const uint4 bv = __ldg(brow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K)[i];
                acc[m] += dot_bf16x8(av, bv);
            }
        }
    }
    const float bv_bias = HAS_BIAS ? __bfloat162float(bias[n]) : 0.0f;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = warp_sum(acc[m]) + bv_bias;
            if (lane == 0) {
                if constexpr (OUT_F32) static_cast<float*>(C)[static_cast<size_t>(m) * N + n] = s;
                else static_cast<__nv_bfloat16*>(C)[static_cast<size_t>(m) * N + n] = __float2bfloat16(s);
            }
        }
    }
}

template <bool OUT_F32 = false, bool HAS_BIAS = false>
inline void launch_bf16_gemv_smallm(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                    void* C, int M, int N, int K, cudaStream_t st,
                                    const __nv_bfloat16* bias = nullptr) {
    const int grid = (N + 3) / 4;
    if (M <= 1)      k_hy4_bf16_gemv_smallm<1,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 2) k_hy4_bf16_gemv_smallm<2,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 4) k_hy4_bf16_gemv_smallm<4,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 8) k_hy4_bf16_gemv_smallm<8,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else             k_hy4_bf16_gemv_smallm<12, OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
}

__global__ void k_hy4_add_bias_bf16(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ b,
                                    int M, int N) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t tot = static_cast<size_t>(M) * N;
    if (idx >= tot) return;
    const int n = static_cast<int>(idx % static_cast<size_t>(N));
    y[idx] = __float2bfloat16(__bfloat162float(y[idx]) + __bfloat162float(b[n]));
}

inline void linear_bf16(cublasHandle_t blas, const __nv_bfloat16* A, const __nv_bfloat16* B,
                        __nv_bfloat16* C, int M, int N, int K, cudaStream_t st,
                        const __nv_bfloat16* bias = nullptr) {
    if (M <= 12 && (K & 7) == 0) {
        if (bias) launch_bf16_gemv_smallm<false, true>(A, B, C, M, N, K, st, bias);
        else      launch_bf16_gemv_smallm<false, false>(A, B, C, M, N, K, st, nullptr);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, N, K, A, B, C);
        if (bias) {
            const size_t tot = static_cast<size_t>(M) * N;
            k_hy4_add_bias_bf16<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(C, bias, M, N);
        }
    }
}

inline void linear_bf16_f32(cublasHandle_t blas, const __nv_bfloat16* A, const __nv_bfloat16* B,
                            float* C, int M, int N, int K, cudaStream_t st) {
    if (M <= 12 && (K & 7) == 0) {
        launch_bf16_gemv_smallm<true, false>(A, B, C, M, N, K, st, nullptr);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt_f32(blas, M, N, K, A, B, C);
    }
}

// Token embedding lookup + 4-stream HC expansion: hyper[m, s, d] = embed[tok[m], d] for s = 0..3.
// Grid: (M * 12), Block: 256 (12 * 256 = 3072 exact).
__global__ void k_hy4_embed_expand(const int* __restrict__ d_tok,
                                   const __nv_bfloat16* __restrict__ embed,
                                   __nv_bfloat16* __restrict__ hyper, int M) {
    const int m = blockIdx.x / 12;
    const int d = (blockIdx.x % 12) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HID) return;
    const int tok = d_tok[m];
    const __nv_bfloat16 v = embed[static_cast<size_t>(tok) * geom::HID + d];
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) hm[static_cast<size_t>(s) * geom::HID + d] = v;
}

// Standalone HYV4FlashRMSNorm: y = weight * bf16(x_f32 * rsqrt(mean(x_f32^2) + eps)).
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_rmsnorm(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ w,
                                                     __nv_bfloat16* __restrict__ y,
                                                     int M, int D, float eps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const __nv_bfloat16* xm = x + static_cast<size_t>(m) * D;
    __nv_bfloat16* ym = y + static_cast<size_t>(m) * D;

    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = __bfloat162float(xm[d]);
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum(ss);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < 8) ? sm[tid] : 0.0f;
        tot = warp_sum(tot);
        if (tid == 0) sm[0] = rsqrtf(tot / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = sm[0];
    for (int d = tid; d < D; d += 256) {
        const __nv_bfloat16 xn = __float2bfloat16(__bfloat162float(xm[d]) * inv);
        ym[d] = __float2bfloat16(__bfloat162float(w[d]) * __bfloat162float(xn));
    }
}

// Fused HC Pre + Layer RMSNorm (small decode M <= 16).
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_hc_pre_and_norm_small(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    const __nv_bfloat16* __restrict__ ln_w,
    float* __restrict__ post_out,
    __nv_bfloat16* __restrict__ mixed_normed,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;
    float ss = 0.0f;
    float dot[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    const uint2* hm4 = reinterpret_cast<const uint2*>(hm);
    for (int i = tid; i < (geom::HC_D >> 2); i += 256) {
        const uint2 raw = hm4[i];
        const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        ss = fmaf(a01.x, a01.x, fmaf(a01.y, a01.y, fmaf(a23.x, a23.x, fmaf(a23.y, a23.y, ss))));
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 f = __ldg(reinterpret_cast<const float4*>(hc_fn + static_cast<size_t>(j) * geom::HC_D) + i);
            dot[j] = fmaf(a01.x, f.x, fmaf(a01.y, f.y, fmaf(a23.x, f.z, fmaf(a23.y, f.w, dot[j]))));
        }
    }

    __shared__ float sm_ss[8];
    __shared__ float sm_dot[8][8];
    ss = warp_sum(ss);
    if (lane == 0) sm_ss[warp] = ss;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float dj = warp_sum(dot[j]);
        if (lane == 0) sm_dot[j][warp] = dj;
    }
    __syncthreads();

    __shared__ float s_pre[4];
    if (tid == 0) {
        float tot_ss = 0.0f;
        for (int w = 0; w < 8; ++w) tot_ss += sm_ss[w];
        const float inv = rsqrtf(tot_ss / static_cast<float>(geom::HC_D) + geom::RMS_EPS);
        const float sc_pre = hc_scale[0];
        const float sc_post = hc_scale[1];
        for (int s = 0; s < 4; ++s) {
            float d_pre = 0.0f, d_post = 0.0f;
            for (int w = 0; w < 8; ++w) {
                d_pre += sm_dot[s][w];
                d_post += sm_dot[4 + s][w];
            }
            s_pre[s] = sigmoid_f(d_pre * inv * sc_pre + hc_base[s]) + geom::HC_EPS;
            post_out[static_cast<size_t>(m) * 4 + s] =
                geom::HC_MAGNITUDE * sigmoid_f(d_post * inv * sc_post + hc_base[4 + s]) + geom::HC_EPS;
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float mss = 0.0f;
    float tmp[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float v = p0 * __bfloat162float(hm[0 * geom::HID + d]) +
                        p1 * __bfloat162float(hm[1 * geom::HID + d]) +
                        p2 * __bfloat162float(hm[2 * geom::HID + d]) +
                        p3 * __bfloat162float(hm[3 * geom::HID + d]);
        const float vbf = __bfloat162float(__float2bfloat16(v));
        tmp[k] = vbf;
        mss = fmaf(vbf, vbf, mss);
    }
    mss = warp_sum(mss);
    if (lane == 0) sm_ss[warp] = mss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        sm_ss[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv_ln = sm_ss[0];
    __nv_bfloat16* ym = mixed_normed + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(tmp[k] * inv_ln);
        ym[d] = __float2bfloat16(__bfloat162float(ln_w[d]) * __bfloat162float(xn));
    }
}

__global__ void k_hy4_bf16_to_f32(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void __launch_bounds__(256) k_hy4_hc_pre_and_norm_pf(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ mixes_raw,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    const __nv_bfloat16* __restrict__ ln_w,
    float* __restrict__ post_out,
    __nv_bfloat16* __restrict__ mixed_normed,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;

    float ss = 0.0f;
    const uint2* hm4 = reinterpret_cast<const uint2*>(hm);
    for (int i = tid; i < (geom::HC_D >> 2); i += 256) {
        const uint2 raw = hm4[i];
        const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        ss = fmaf(a01.x, a01.x, fmaf(a01.y, a01.y, fmaf(a23.x, a23.x, fmaf(a23.y, a23.y, ss))));
    }
    __shared__ float sm_ss[8];
    ss = warp_sum(ss);
    if (lane == 0) sm_ss[warp] = ss;
    __syncthreads();

    __shared__ float s_pre[4];
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        const float inv = rsqrtf(tot / static_cast<float>(geom::HC_D) + geom::RMS_EPS);
        const float* mx = mixes_raw + static_cast<size_t>(m) * 8;
        const float sc0 = hc_scale[0], sc1 = hc_scale[1];
        for (int s = 0; s < 4; ++s) {
            s_pre[s] = sigmoid_f(mx[s] * inv * sc0 + hc_base[s]) + geom::HC_EPS;
            post_out[static_cast<size_t>(m) * 4 + s] =
                geom::HC_MAGNITUDE * sigmoid_f(mx[4 + s] * inv * sc1 + hc_base[4 + s]) + geom::HC_EPS;
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float mss = 0.0f;
    float tmp[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float v = p0 * __bfloat162float(hm[0 * geom::HID + d]) +
                        p1 * __bfloat162float(hm[1 * geom::HID + d]) +
                        p2 * __bfloat162float(hm[2 * geom::HID + d]) +
                        p3 * __bfloat162float(hm[3 * geom::HID + d]);
        const float vbf = __bfloat162float(__float2bfloat16(v));
        tmp[k] = vbf;
        mss = fmaf(vbf, vbf, mss);
    }
    mss = warp_sum(mss);
    if (lane == 0) sm_ss[warp] = mss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        sm_ss[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv_ln = sm_ss[0];
    __nv_bfloat16* ym = mixed_normed + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(tmp[k] * inv_ln);
        ym[d] = __float2bfloat16(__bfloat162float(ln_w[d]) * __bfloat162float(xn));
    }
}

// 8-block parallel FP32 dot product for small decode batches (M <= 16): 8x SM parallelism on hc_fn[8, 12288]!
// Grid: dim3(8, M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_hc_dot_small(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ hc_fn,
    float* __restrict__ mixes_raw,
    int M) {
    const int j = blockIdx.x;
    const int m = blockIdx.y;
    if (j >= geom::HC_MIX || m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const uint2* hm4 = reinterpret_cast<const uint2*>(hyper + static_cast<size_t>(m) * geom::HC_D);
    const float4* fn4 = reinterpret_cast<const float4*>(hc_fn + static_cast<size_t>(j) * geom::HC_D);
    float d0 = 0.0f, d1 = 0.0f;
    // HC_D >> 2 = 3072 = 12 * 256 exact -> 6 iterations of 2x unrolled 256-stride!
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        const int i0 = tid + (k * 2) * 256;
        const int i1 = tid + (k * 2 + 1) * 256;
        const uint2 r0 = hm4[i0], r1 = hm4[i1];
        const float4 f0 = __ldg(fn4 + i0), f1 = __ldg(fn4 + i1);
        const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r0.x));
        const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r0.y));
        const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r1.x));
        const float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r1.y));
        d0 = fmaf(a0.x, f0.x, fmaf(a0.y, f0.y, fmaf(a1.x, f0.z, fmaf(a1.y, f0.w, d0))));
        d1 = fmaf(b0.x, f1.x, fmaf(b0.y, f1.y, fmaf(b1.x, f1.z, fmaf(b1.y, f1.w, d1))));
    }
    float acc = warp_sum(d0 + d1);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w];
        mixes_raw[static_cast<size_t>(m) * geom::HC_MIX + j] = tot;
    }
}

inline void hc_pre_and_norm(cublasHandle_t blas, const HcWeights& w, HcScratch& sc,
                            const __nv_bfloat16* hyper, const __nv_bfloat16* ln_w,
                            __nv_bfloat16* mixed_normed, int M, cudaStream_t st) {
    if (M <= 16) {
        k_hy4_hc_dot_small<<<dim3(geom::HC_MIX, M), 256, 0, st>>>(hyper, w.fn, sc.mixes, M);
        k_hy4_hc_pre_and_norm_pf<<<M, 256, 0, st>>>(
            hyper, sc.mixes, w.base, w.scale, ln_w, sc.post, mixed_normed, M);
    } else {
        const size_t n = static_cast<size_t>(M) * geom::HC_D;
        k_hy4_bf16_to_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(hyper, sc.flat_f32, n);
        cublasSetStream(blas, st);
        gemm_f32_nt(blas, M, geom::HC_MIX, geom::HC_D, sc.flat_f32, w.fn, sc.mixes);
        k_hy4_hc_pre_and_norm_pf<<<M, 256, 0, st>>>(
            hyper, sc.mixes, w.base, w.scale, ln_w, sc.post, mixed_normed, M);
    }
}

// Fused TP Partial Sum (exact rank order 0..nr-1) + Sandwich RMSNorm + HC Residual Inject.
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_sum_norm_inject(
    const __nv_bfloat16* __restrict__ parts,
    const __nv_bfloat16* __restrict__ post_norm_w,
    const float* __restrict__ post,
    __nv_bfloat16* __restrict__ hyper,
    int nr, int M, int part_m_stride) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const size_t rank_stride = static_cast<size_t>(part_m_stride) * geom::HID;

    float y[12];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        float acc = 0.0f;
        for (int r = 0; r < nr; ++r) {
            acc += __bfloat162float(parts[static_cast<size_t>(r) * rank_stride + static_cast<size_t>(m) * geom::HID + d]);
        }
        const float ybf = __bfloat162float(__float2bfloat16(acc));
        y[k] = ybf;
        ss = fmaf(ybf, ybf, ss);
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    const float p0 = post[static_cast<size_t>(m) * 4 + 0];
    const float p1 = post[static_cast<size_t>(m) * 4 + 1];
    const float p2 = post[static_cast<size_t>(m) * 4 + 2];
    const float p3 = post[static_cast<size_t>(m) * 4 + 3];
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(y[k] * inv);
        const float ny = __bfloat162float(__float2bfloat16(__bfloat162float(post_norm_w[d]) * __bfloat162float(xn)));
        hm[0 * geom::HID + d] = __float2bfloat16(fmaf(p0, ny, __bfloat162float(hm[0 * geom::HID + d])));
        hm[1 * geom::HID + d] = __float2bfloat16(fmaf(p1, ny, __bfloat162float(hm[1 * geom::HID + d])));
        hm[2 * geom::HID + d] = __float2bfloat16(fmaf(p2, ny, __bfloat162float(hm[2 * geom::HID + d])));
        hm[3 * geom::HID + d] = __float2bfloat16(fmaf(p3, ny, __bfloat162float(hm[3 * geom::HID + d])));
    }
}

// Final HYV4FlashHyperHead + model.norm: hyper[M, 4, 3072] -> out[M, 3072] BF16 for lm_head.
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_hc_head_and_norm(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ head_fn,
    const float* __restrict__ head_base,
    const float* __restrict__ head_scale,
    const __nv_bfloat16* __restrict__ final_norm_w,
    __nv_bfloat16* __restrict__ out,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;

    float ss = 0.0f;
    float dot[4] = {0, 0, 0, 0};
    const uint2* hm4 = reinterpret_cast<const uint2*>(hm);
    for (int i = tid; i < (geom::HC_D >> 2); i += 256) {
        const uint2 raw = hm4[i];
        const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        ss = fmaf(a01.x, a01.x, fmaf(a01.y, a01.y, fmaf(a23.x, a23.x, fmaf(a23.y, a23.y, ss))));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 f = __ldg(reinterpret_cast<const float4*>(head_fn + static_cast<size_t>(j) * geom::HC_D) + i);
            dot[j] = fmaf(a01.x, f.x, fmaf(a01.y, f.y, fmaf(a23.x, f.z, fmaf(a23.y, f.w, dot[j]))));
        }
    }
    __shared__ float sm_ss[8];
    __shared__ float sm_dot[4][8];
    ss = warp_sum(ss);
    if (lane == 0) sm_ss[warp] = ss;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float dj = warp_sum(dot[j]);
        if (lane == 0) sm_dot[j][warp] = dj;
    }
    __syncthreads();

    __shared__ float s_pre[4];
    if (tid == 0) {
        float tot_ss = 0.0f;
        for (int w = 0; w < 8; ++w) tot_ss += sm_ss[w];
        const float inv = rsqrtf(tot_ss / static_cast<float>(geom::HC_D) + geom::RMS_EPS);
        const float sc = head_scale[0];
        for (int s = 0; s < 4; ++s) {
            float ds = 0.0f;
            for (int w = 0; w < 8; ++w) ds += sm_dot[s][w];
            s_pre[s] = sigmoid_f(ds * inv * sc + head_base[s]) + geom::HC_EPS;
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float mss = 0.0f;
    float tmp[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float v = p0 * __bfloat162float(hm[0 * geom::HID + d]) +
                        p1 * __bfloat162float(hm[1 * geom::HID + d]) +
                        p2 * __bfloat162float(hm[2 * geom::HID + d]) +
                        p3 * __bfloat162float(hm[3 * geom::HID + d]);
        const float vbf = __bfloat162float(__float2bfloat16(v));
        tmp[k] = vbf;
        mss = fmaf(vbf, vbf, mss);
    }
    mss = warp_sum(mss);
    if (lane == 0) sm_ss[warp] = mss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        sm_ss[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv_ln = sm_ss[0];
    __nv_bfloat16* ym = out + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(tmp[k] * inv_ln);
        ym[d] = __float2bfloat16(__bfloat162float(final_norm_w[d]) * __bfloat162float(xn));
    }
}

// Fused TP=1 two-source sum (a + b) + sandwich RMSNorm + HC residual inject (saves a full BF16 round-trip on MoE!).
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_add_norm_inject(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ post_norm_w,
    const float* __restrict__ post_w,
    __nv_bfloat16* __restrict__ hyper,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* am = a + static_cast<size_t>(m) * geom::HID;
    const __nv_bfloat16* bm = b + static_cast<size_t>(m) * geom::HID;

    float y[12];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float v = __bfloat162float(__float2bfloat16(__bfloat162float(am[d]) + __bfloat162float(bm[d])));
        y[k] = v;
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    const float p0 = post_w[static_cast<size_t>(m) * 4 + 0];
    const float p1 = post_w[static_cast<size_t>(m) * 4 + 1];
    const float p2 = post_w[static_cast<size_t>(m) * 4 + 2];
    const float p3 = post_w[static_cast<size_t>(m) * 4 + 3];
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_D;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(y[k] * inv);
        const float ny = __bfloat162float(__float2bfloat16(__bfloat162float(post_norm_w[d]) * __bfloat162float(xn)));
        hm[0 * geom::HID + d] = __float2bfloat16(fmaf(p0, ny, __bfloat162float(hm[0 * geom::HID + d])));
        hm[1 * geom::HID + d] = __float2bfloat16(fmaf(p1, ny, __bfloat162float(hm[1 * geom::HID + d])));
        hm[2 * geom::HID + d] = __float2bfloat16(fmaf(p2, ny, __bfloat162float(hm[2 * geom::HID + d])));
        hm[3 * geom::HID + d] = __float2bfloat16(fmaf(p3, ny, __bfloat162float(hm[3 * geom::HID + d])));
    }
}

// Residual add with optional sandwich RMSNorm (for DFly2 single-stream layers):
//   if post_norm_w != nullptr: h[m,d] += post_norm_w[d] * rmsnorm(sum_r parts[r,m,d])
//   else:                      h[m,d] += sum_r parts[r,m,d]
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_sum_norm_add1(
    const __nv_bfloat16* __restrict__ parts,
    const __nv_bfloat16* __restrict__ post_norm_w,
    __nv_bfloat16* __restrict__ h,
    int nr, int stride_rows, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    float y[12];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        float acc = 0.0f;
        for (int r = 0; r < nr; ++r) {
            acc += __bfloat162float(parts[(static_cast<size_t>(r) * stride_rows + m) * geom::HID + d]);
        }
        const float v = __bfloat162float(__float2bfloat16(acc));
        y[k] = v;
        ss = fmaf(v, v, ss);
    }
    __nv_bfloat16* hm = h + static_cast<size_t>(m) * geom::HID;
    if (post_norm_w != nullptr) {
        __shared__ float sm[8];
        ss = warp_sum(ss);
        if (lane == 0) sm[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < 8; ++w) tot += sm[w];
            sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
        }
        __syncthreads();
        const float inv = sm[0];
        #pragma unroll
        for (int k = 0; k < 12; ++k) {
            const int d = tid + k * 256;
            const __nv_bfloat16 xn = __float2bfloat16(y[k] * inv);
            const float ny = __bfloat162float(__float2bfloat16(__bfloat162float(post_norm_w[d]) * __bfloat162float(xn)));
            hm[d] = __float2bfloat16(__bfloat162float(hm[d]) + ny);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 12; ++k) {
            const int d = tid + k * 256;
            hm[d] = __float2bfloat16(__bfloat162float(hm[d]) + y[k]);
        }
    }
}

} // namespace hy4
