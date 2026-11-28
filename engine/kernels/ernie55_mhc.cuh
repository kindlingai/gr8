// ERNIE-5.5-Flash-Preview 4-stream Manifold Hyper-Connections (mHC, Sinkhorn-20), RMSNorm,
// mhc_contract head, multimax LM-head transform, and BF16 GEMV/GEMM helpers.
//
// Exact ERNIE 5.5 math invariants verified against config.json & CPU reference:
//   - Flat mHC RMS norm puts rms_eps OUTSIDE sqrt:
//       denom = sqrt(mean(flat^2)) + RMS_EPS           (NOT rsqrt(mean + eps)!)
//   - Checkpoint `mapping_proj.weight` is F32 [16384, 24] and `hc_head_fn` is F32 [16384, 4];
//     boot-time `transpose_f32_inplace` transposes them to coalesced row-major [24, 16384] / [4, 16384].
//   - mHC Pre (20 Sinkhorn iterations):
//       mixes    = (flat @ fn^T) / denom               [M, 24] F32
//       pre_mix  = sigmoid(mixes[:, 0:4] * scale[0] + bias[0:4])
//       post_mix = 2.0 * sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
//       comb     = softmax_row(reshape(mixes[:, 8:24] * scale[2] + bias[8:24], 4, 4)) + HC_EPS
//       comb     = col_norm(comb, HC_EPS); repeat 19x { comb = row_norm(comb); comb = col_norm(comb); }
//       layer_in = bf16(sum_s (pre_mix[s] + HC_EPS) * hyper[s])
//       out      = rms_norm(layer_in, ln_w, RMS_EPS)
//   - mHC Post (no post-projection sandwich norm in ERNIE 5.5):
//       new_hyper[i] = bf16(sum_j comb[i, j] * hyper[j] + post_mix[i] * sublayer_out)
//   - mhc_contract Head + final model.norm:
//       mixes = (flat @ head_fn^T) / (sqrt(mean(flat^2)) + RMS_EPS)
//       pre   = sigmoid(mixes * head_scale[0] + head_base) + HC_EPS
//       out   = rms_norm(bf16(sum_s pre[s] * hyper[s]), final_norm_w, RMS_EPS)
//   - multimax LM-head piecewise adjustment (F32 logits, BF16 ranges[4] and ts[4]):
//       y = x + t0*relu(r0 - x) + t1*relu(x - r1) + t2*relu(r2 - x)^2 + t3*relu(x - r3)^2
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include "../model_geom_ernie55.h"

namespace ernie55 {

struct StepSeg {
    int slot = 0;
    int row0 = 0;
    int pos0 = 0;
    int len  = 0;
};

struct HcWeights {
    const float* fn    = nullptr; // [24, 16384] F32 (boot-transposed from checkpoint [16384, 24])
    const float* base  = nullptr; // [24] F32 (bias)
    const float* scale = nullptr; // [3] F32 ({alpha_pre, alpha_post, alpha_res})
};

struct HcHeadWeights {
    const float* fn    = nullptr; // [4, 16384] F32 (boot-transposed from checkpoint [16384, 4])
    const float* base  = nullptr; // [4] F32
    const float* scale = nullptr; // [1] F32
};

struct HcScratch {
    float* flat_f32 = nullptr; // [max_M, 16384] F32 (for prefill SGEMM path M > 16)
    float* mixes    = nullptr; // [max_M, 24] F32
    float* post     = nullptr; // [max_M, 4] F32
    float* comb     = nullptr; // [max_M, 16] F32 (4x4 doubly-stochastic Sinkhorn-20 matrix)
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

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// Exact ERNIE 5.5 SiTU-GLU activation:
//   g_act = beta * tanh(g / beta) * sigmoid(g)
//   u_act = linear_beta * tanh(u / linear_beta)  (when linear_beta > 0)
__device__ __forceinline__ float situ_mul_f(float g, float u) {
    const float ga = geom::SITU_BETA * tanhf(g * (1.0f / geom::SITU_BETA)) * sigmoid_f(g);
    const float ua = geom::SITU_LINEAR_BETA * tanhf(u * (1.0f / geom::SITU_LINEAR_BETA));
    return ga * ua;
}

// =================================================================================================
// Boot-time F32 matrix transpose: src[R, C] -> dst[C, R]
// =================================================================================================
__global__ void __launch_bounds__(256) k_ernie55_transpose_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int R, int C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t tot = static_cast<size_t>(R) * C;
    if (idx >= tot) return;
    const int r = static_cast<int>(idx / static_cast<size_t>(C));
    const int c = static_cast<int>(idx % static_cast<size_t>(C));
    dst[static_cast<size_t>(c) * R + r] = src[idx];
}

inline void transpose_f32_inplace(float* mat, float* scratch, int R, int C, cudaStream_t st) {
    const size_t tot = static_cast<size_t>(R) * C;
    k_ernie55_transpose_f32<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(mat, scratch, R, C);
    cudaMemcpyAsync(mat, scratch, tot * sizeof(float), cudaMemcpyDeviceToDevice, st);
}

// =================================================================================================
// BF16 & F32 GEMV / GEMM Helpers
// =================================================================================================
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

template <int MAX_M, bool OUT_F32 = false, bool HAS_BIAS = false>
__global__ void __launch_bounds__(128) k_ernie55_bf16_gemv_smallm(
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
    if (M <= 1)      k_ernie55_bf16_gemv_smallm<1,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 2) k_ernie55_bf16_gemv_smallm<2,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 4) k_ernie55_bf16_gemv_smallm<4,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else if (M <= 8) k_ernie55_bf16_gemv_smallm<8,  OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
    else             k_ernie55_bf16_gemv_smallm<12, OUT_F32, HAS_BIAS><<<grid, 128, 0, st>>>(A, B, bias, C, M, N, K);
}

__global__ void k_ernie55_add_bias_bf16(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ b,
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
            k_ernie55_add_bias_bf16<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(C, bias, M, N);
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

// =================================================================================================
// Embedding Expand + RMSNorm + mHC Pre/Post + mhc_contract + multimax
// =================================================================================================

// Token embedding lookup + 4-stream mHC expansion: hyper[m, s, d] = embed[tok[m], d] for s = 0..3.
// Grid: (M * 16), Block: 256 (16 * 256 = 4096 exact).
__global__ void __launch_bounds__(256) k_ernie55_embed_expand(
    const int* __restrict__ d_tok,
    const __nv_bfloat16* __restrict__ embed,
    __nv_bfloat16* __restrict__ hyper, int M) {
    const int m = blockIdx.x / 16;
    const int d = (blockIdx.x % 16) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HID) return;
    const int tok = d_tok[m];
    const __nv_bfloat16 v = embed[static_cast<size_t>(tok) * geom::HID + d];
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) hm[static_cast<size_t>(s) * geom::HID + d] = v;
}

// Hybrid-mode 4-stream expansion from host-fetched BF16 embeddings [M, 4096] already copied to GPU.
// Grid: (M * 16), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_expand_from_rows(
    const __nv_bfloat16* __restrict__ rows,
    __nv_bfloat16* __restrict__ hyper, int M) {
    const int m = blockIdx.x / 16;
    const int d = (blockIdx.x % 16) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HID) return;
    const __nv_bfloat16 v = rows[static_cast<size_t>(m) * geom::HID + d];
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) hm[static_cast<size_t>(s) * geom::HID + d] = v;
}

// Standard RMSNorm (optional weight): y = bf16(x_f32 * rsqrt(mean(x_f32^2) + eps) * w_f32).
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_rmsnorm(
    const __nv_bfloat16* __restrict__ x,
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
        float val = __bfloat162float(xm[d]) * inv;
        if (w != nullptr) val *= __bfloat162float(w[d]);
        ym[d] = __float2bfloat16(val);
    }
}

__global__ void k_ernie55_bf16_to_f32(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// 24-block parallel FP32 dot product for small decode batches (M <= 16): 24x SM parallelism on hc_fn[24, 16384].
// Grid: dim3(24, M), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_mhc_dot_small(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ hc_fn,
    float* __restrict__ mixes_raw,
    int M) {
    const int j = blockIdx.x;
    const int m = blockIdx.y;
    if (j >= geom::HC_MIX || m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const uint2* hm4 = reinterpret_cast<const uint2*>(hyper + static_cast<size_t>(m) * geom::HC_DIM);
    const float4* fn4 = reinterpret_cast<const float4*>(hc_fn + static_cast<size_t>(j) * geom::HC_DIM);
    float d0 = 0.0f, d1 = 0.0f;
    // HC_DIM >> 2 = 4096 = 16 * 256 exact -> 8 iterations of 2x unrolled 256-stride.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
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

// Exact ERNIE 5.5 mHC Sinkhorn-20 + Stream Pre-Mix + Layer RMSNorm.
// Grid: (M), Block: 256 (16 elements per thread for HID=4096 exact).
__global__ void __launch_bounds__(256) k_ernie55_mhc_pre_and_norm(
    const __nv_bfloat16* __restrict__ hyper,
    const float* __restrict__ mixes_raw,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    const __nv_bfloat16* __restrict__ ln_w,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ mixed_normed,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;

    float ss = 0.0f;
    const uint2* hm4 = reinterpret_cast<const uint2*>(hm);
    for (int i = tid; i < (geom::HC_DIM >> 2); i += 256) {
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
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        // ERNIE 5.5 invariant: eps OUTSIDE sqrt!
        const float inv = 1.0f / (sqrtf(tot * (1.0f / static_cast<float>(geom::HC_DIM))) + geom::RMS_EPS);
        const float* mx = mixes_raw + static_cast<size_t>(m) * geom::HC_MIX;
        const float sc_pre  = hc_scale[0];
        const float sc_post = hc_scale[1];
        const float sc_res  = hc_scale[2];

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            s_pre[s] = sigmoid_f(mx[s] * inv * sc_pre + hc_base[s]) + geom::HC_EPS;
            post_out[static_cast<size_t>(m) * 4 + s] =
                geom::HC_POST_ALPHA * sigmoid_f(mx[4 + s] * inv * sc_post + hc_base[4 + s]);
        }

        // 4x4 Sinkhorn-20 doubly-stochastic manifold projection
        float C[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float rmax = -INFINITY;
            float z[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = 8 + i * 4 + j;
                z[j] = mx[idx] * inv * sc_res + hc_base[idx];
                rmax = fmaxf(rmax, z[j]);
            }
            float rsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                z[j] = expf(z[j] - rmax);
                rsum += z[j];
            }
            const float inv_r = 1.0f / rsum;
            #pragma unroll
            for (int j = 0; j < 4; ++j) C[i][j] = z[j] * inv_r + geom::HC_EPS;
        }
        // First column normalization
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float csum = C[0][j] + C[1][j] + C[2][j] + C[3][j] + geom::HC_EPS;
            const float inv_c = 1.0f / csum;
            #pragma unroll
            for (int i = 0; i < 4; ++i) C[i][j] *= inv_c;
        }
        // Remaining HC_ITERS - 1 = 19 row + col normalizations
        for (int it = 0; it < geom::HC_ITERS - 1; ++it) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float rsum = C[i][0] + C[i][1] + C[i][2] + C[i][3] + geom::HC_EPS;
                const float inv_r = 1.0f / rsum;
                #pragma unroll
                for (int j = 0; j < 4; ++j) C[i][j] *= inv_r;
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float csum = C[0][j] + C[1][j] + C[2][j] + C[3][j] + geom::HC_EPS;
                const float inv_c = 1.0f / csum;
                #pragma unroll
                for (int i = 0; i < 4; ++i) C[i][j] *= inv_c;
            }
        }
        float* co = comb_out + static_cast<size_t>(m) * 16;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) co[i * 4 + j] = C[i][j];
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float mss = 0.0f;
    float tmp[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
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
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        sm_ss[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::HID)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv_ln = sm_ss[0];
    __nv_bfloat16* ym = mixed_normed + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = tid + k * 256;
        ym[d] = __float2bfloat16(tmp[k] * inv_ln * __bfloat162float(ln_w[d]));
    }
}

inline void mhc_pre_and_norm(cublasHandle_t blas, const HcWeights& w, HcScratch& sc,
                             const __nv_bfloat16* hyper, const __nv_bfloat16* ln_w,
                             __nv_bfloat16* mixed_normed, int M, cudaStream_t st) {
    if (M <= 16) {
        k_ernie55_mhc_dot_small<<<dim3(geom::HC_MIX, M), 256, 0, st>>>(hyper, w.fn, sc.mixes, M);
    } else {
        const size_t n = static_cast<size_t>(M) * geom::HC_DIM;
        k_ernie55_bf16_to_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(hyper, sc.flat_f32, n);
        cublasSetStream(blas, st);
        gemm_f32_nt(blas, M, geom::HC_MIX, geom::HC_DIM, sc.flat_f32, w.fn, sc.mixes);
    }
    k_ernie55_mhc_pre_and_norm<<<M, 256, 0, st>>>(
        hyper, sc.mixes, w.base, w.scale, ln_w, sc.post, sc.comb, mixed_normed, M);
}

// Fused TP Partial Sum + mHC Post Residual Combine (in-place on hyper[M, 4, 4096]).
// Grid: (M * 16), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_sum_mhc_post(
    const __nv_bfloat16* __restrict__ parts,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ hyper,
    int nr, int M, int part_m_stride) {
    const int m = blockIdx.x / 16;
    const int d = (blockIdx.x % 16) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HID) return;
    const size_t rank_stride = static_cast<size_t>(part_m_stride) * geom::HID;
    float y = 0.0f;
    for (int r = 0; r < nr; ++r) {
        y += __bfloat162float(parts[static_cast<size_t>(r) * rank_stride + static_cast<size_t>(m) * geom::HID + d]);
    }
    const float ybf = __bfloat162float(__float2bfloat16(y));
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;
    const float h0 = __bfloat162float(hm[0 * geom::HID + d]);
    const float h1 = __bfloat162float(hm[1 * geom::HID + d]);
    const float h2 = __bfloat162float(hm[2 * geom::HID + d]);
    const float h3 = __bfloat162float(hm[3 * geom::HID + d]);
    const float* C = comb + static_cast<size_t>(m) * 16;
    const float* P = post + static_cast<size_t>(m) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = C[i * 4 + 0] * h0 + C[i * 4 + 1] * h1 +
                        C[i * 4 + 2] * h2 + C[i * 4 + 3] * h3 + P[i] * ybf;
        hm[static_cast<size_t>(i) * geom::HID + d] = __float2bfloat16(v);
    }
}

// Fused TP=1 MoE (routed_up_out + shared_out) + mHC Post Residual Combine (saves a BF16 round-trip).
// Grid: (M * 16), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_add_mhc_post(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ hyper,
    int M) {
    const int m = blockIdx.x / 16;
    const int d = (blockIdx.x % 16) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HID) return;
    const float ybf = __bfloat162float(__float2bfloat16(
        __bfloat162float(a[static_cast<size_t>(m) * geom::HID + d]) +
        __bfloat162float(b[static_cast<size_t>(m) * geom::HID + d])));
    __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;
    const float h0 = __bfloat162float(hm[0 * geom::HID + d]);
    const float h1 = __bfloat162float(hm[1 * geom::HID + d]);
    const float h2 = __bfloat162float(hm[2 * geom::HID + d]);
    const float h3 = __bfloat162float(hm[3 * geom::HID + d]);
    const float* C = comb + static_cast<size_t>(m) * 16;
    const float* P = post + static_cast<size_t>(m) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = C[i * 4 + 0] * h0 + C[i * 4 + 1] * h1 +
                        C[i * 4 + 2] * h2 + C[i * 4 + 3] * h3 + P[i] * ybf;
        hm[static_cast<size_t>(i) * geom::HID + d] = __float2bfloat16(v);
    }
}

// Final mhc_contract Head + model.norm: hyper[M, 4, 4096] -> out[M, 4096] BF16 for lm_head.
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_hc_head_and_norm(
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
    const __nv_bfloat16* hm = hyper + static_cast<size_t>(m) * geom::HC_DIM;

    float ss = 0.0f;
    float dot[4] = {0, 0, 0, 0};
    const uint2* hm4 = reinterpret_cast<const uint2*>(hm);
    for (int i = tid; i < (geom::HC_DIM >> 2); i += 256) {
        const uint2 raw = hm4[i];
        const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        ss = fmaf(a01.x, a01.x, fmaf(a01.y, a01.y, fmaf(a23.x, a23.x, fmaf(a23.y, a23.y, ss))));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 f = __ldg(reinterpret_cast<const float4*>(head_fn + static_cast<size_t>(j) * geom::HC_DIM) + i);
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
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot_ss += sm_ss[w];
        // ERNIE 5.5 invariant: eps OUTSIDE sqrt!
        const float inv = 1.0f / (sqrtf(tot_ss * (1.0f / static_cast<float>(geom::HC_DIM))) + geom::RMS_EPS);
        const float sc = head_scale[0];
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            float ds = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) ds += sm_dot[s][w];
            s_pre[s] = sigmoid_f(ds * inv * sc + head_base[s]) + geom::HC_EPS;
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float mss = 0.0f;
    float tmp[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
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
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm_ss[w];
        sm_ss[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::HID)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv_ln = sm_ss[0];
    __nv_bfloat16* ym = out + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = tid + k * 256;
        ym[d] = __float2bfloat16(tmp[k] * inv_ln * __bfloat162float(final_norm_w[d]));
    }
}

// In-place ERNIE 5.5 multimax LM-head transform on F32 logits[M, vr].
// ranges and ts are BF16 [4] in the checkpoint.
__global__ void __launch_bounds__(256) k_ernie55_multimax(
    float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ ranges,
    const __nv_bfloat16* __restrict__ ts,
    size_t total) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float r0 = __bfloat162float(ranges[0]);
    const float r1 = __bfloat162float(ranges[1]);
    const float r2 = __bfloat162float(ranges[2]);
    const float r3 = __bfloat162float(ranges[3]);
    const float t0 = __bfloat162float(ts[0]);
    const float t1 = __bfloat162float(ts[1]);
    const float t2 = __bfloat162float(ts[2]);
    const float t3 = __bfloat162float(ts[3]);
    const float x = logits[i];
    const float a = fmaxf(0.0f, r0 - x);
    const float b = fmaxf(0.0f, x - r1);
    const float c = fmaxf(0.0f, r2 - x);
    const float d = fmaxf(0.0f, x - r3);
    logits[i] = x + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
}

inline void apply_multimax(float* logits, const __nv_bfloat16* ranges, const __nv_bfloat16* ts,
                           int M, int vr, cudaStream_t st) {
    if (!ranges || !ts || M <= 0 || vr <= 0) return;
    const size_t total = static_cast<size_t>(M) * vr;
    k_ernie55_multimax<<<static_cast<unsigned>((total + 255) / 256), 256, 0, st>>>(logits, ranges, ts, total);
}

// Fused BF16 LM-head GEMV + multimax transform for small-M decode (M <= 12).
// Saves a full [M, 201216] F32 read/write pass and keeps multimax scalars in registers.
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ernie55_lm_head_bf16_multimax(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ ranges,
    const __nv_bfloat16* __restrict__ ts,
    float* __restrict__ C,
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
                acc[m] += dot_bf16x8(__ldg(am + i), b0) + dot_bf16x8(__ldg(am + i + 32), b1)
                        + dot_bf16x8(__ldg(am + i + 64), b2) + dot_bf16x8(__ldg(am + i + 96), b3);
            }
        }
    }
    for (; i < K8; i += 32) {
        const uint4 bv = __ldg(brow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K) + i);
                acc[m] += dot_bf16x8(av, bv);
            }
        }
    }
    const bool has_mm = (ranges != nullptr && ts != nullptr);
    const float r0 = has_mm ? __bfloat162float(__ldg(ranges + 0)) : 0.0f;
    const float r1 = has_mm ? __bfloat162float(__ldg(ranges + 1)) : 0.0f;
    const float r2 = has_mm ? __bfloat162float(__ldg(ranges + 2)) : 0.0f;
    const float r3 = has_mm ? __bfloat162float(__ldg(ranges + 3)) : 0.0f;
    const float t0 = has_mm ? __bfloat162float(__ldg(ts + 0)) : 0.0f;
    const float t1 = has_mm ? __bfloat162float(__ldg(ts + 1)) : 0.0f;
    const float t2 = has_mm ? __bfloat162float(__ldg(ts + 2)) : 0.0f;
    const float t3 = has_mm ? __bfloat162float(__ldg(ts + 3)) : 0.0f;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float x = warp_sum(acc[m]);
            if (lane == 0) {
                if (has_mm) {
                    const float a = fmaxf(0.0f, r0 - x);
                    const float b = fmaxf(0.0f, x - r1);
                    const float c = fmaxf(0.0f, r2 - x);
                    const float d = fmaxf(0.0f, x - r3);
                    x = x + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d);
                }
                C[static_cast<size_t>(m) * N + n] = x;
            }
        }
    }
}

inline void linear_bf16_f32_multimax(cublasHandle_t blas, const __nv_bfloat16* A, const __nv_bfloat16* B,
                                     const __nv_bfloat16* ranges, const __nv_bfloat16* ts,
                                     float* C, int M, int N, int K, cudaStream_t st) {
    if (M <= 12 && (K & 7) == 0) {
        const unsigned grid = static_cast<unsigned>((N + 3) / 4);
        if (M <= 1)      k_ernie55_lm_head_bf16_multimax<1> <<<grid, 128, 0, st>>>(A, B, ranges, ts, C, M, N, K);
        else if (M <= 2) k_ernie55_lm_head_bf16_multimax<2> <<<grid, 128, 0, st>>>(A, B, ranges, ts, C, M, N, K);
        else if (M <= 4) k_ernie55_lm_head_bf16_multimax<4> <<<grid, 128, 0, st>>>(A, B, ranges, ts, C, M, N, K);
        else if (M <= 8) k_ernie55_lm_head_bf16_multimax<8> <<<grid, 128, 0, st>>>(A, B, ranges, ts, C, M, N, K);
        else             k_ernie55_lm_head_bf16_multimax<12><<<grid, 128, 0, st>>>(A, B, ranges, ts, C, M, N, K);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt_f32(blas, M, N, K, A, B, C);
        apply_multimax(C, ranges, ts, M, N, st);
    }
}

} // namespace ernie55
