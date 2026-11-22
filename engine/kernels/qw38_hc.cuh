// Qwen3.8-Flash-Next hyper-connections (Qwen4ExpTextGatedResidual) and RMSNorm kernels.
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py:
//   - Qwen4ExpTextRMSNorm: zero-centered (1.0f + weight.float()), optional group_size = HID (2560).
//   - Initial state: inputs_embeds.repeat(1, 1, 4) -> [M, 4, 2560] BF16.
//   - Qwen4ExpTextGatedResidual (use_combine = true):
//       hn    = hc_norm(hyper_in)                               [M, 4, 2560] BF16 (grouped per 2560)
//       down  = silu(hn @ W_down^T / 4.0)                       [M, 320] BF16
//       mix_w = sigmoid(down @ W_up^T)                          [M, 4, 2560]
//       mixed = mean_{s=0..3}(mix_w[:, s, :] * hn[:, s, :])     [M, 2560] BF16
//       inj_w = 2.0 * sigmoid(hn @ W_inj^T / 4.0)               [M, 4] FP32
//     After sublayer y [M, 2560]:
//       hyper_out[:, s, :] = hyper_in[:, s, :] + y * inj_w[:, s]
//   - Final hyper_connection_mixer (use_combine = false): returns mixed [M, 2560] BF16 directly for lm_head.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "../model_geom_qw38.h"

namespace qw38 {

struct HcWeights {
    const __nv_bfloat16* norm = nullptr;    // [10240] BF16 (zero-centered: 1 + w)
    const __nv_bfloat16* down = nullptr;    // [320, 10240] BF16
    const __nv_bfloat16* up = nullptr;      // [10240, 320] BF16
    const __nv_bfloat16* inject = nullptr;  // [4, 10240] BF16 (nullptr for final mixer)
};

struct HcScratch {
    __nv_bfloat16* hn = nullptr;      // [max_M, 10240] BF16
    __nv_bfloat16* down = nullptr;    // [max_M, 320] BF16
    __nv_bfloat16* up = nullptr;      // [max_M, 10240] BF16
    float* inj = nullptr;             // [max_M, 4] FP32
};

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// BF16 row-major GEMM: C[M, N] = A[M, K] @ B[N, K]^T (all row-major, FP32 accumulate).
// In cuBLAS column-major terms: C^T[N, M] = B^T[N, K] @ A^T[K, M] => op(B)=T, op(A)=N.
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

// Replicate token embeddings into 4 hyper-connection streams: out[m, s, d] = embed[tok[m], d]
__global__ void k_embed_hc4(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ embed,
                            __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* src = embed + static_cast<size_t>(tok[m]) * geom::HID;
    __nv_bfloat16* dst = out + static_cast<size_t>(m) * geom::HC_D;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        const __nv_bfloat16 v = src[d];
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) dst[s * geom::HID + d] = v;
    }
}

// Grouped zero-centered RMSNorm: x[M, G, D] -> out[M, G, D] BF16, weight[G*D] BF16.
// One CTA per (m, g). Exact match to Qwen4ExpTextRMSNorm: rsqrt(mean(x^2)+eps) * x * (1 + w) -> bf16.
template <int D>
__global__ void k_rmsnorm_1pw(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                              __nv_bfloat16* __restrict__ out, int M, int G, float eps) {
    const int mg = blockIdx.x;
    const int m = mg / G;
    const int g = mg % G;
    if (m >= M) return;
    const __nv_bfloat16* xr = x + (static_cast<size_t>(m) * G + g) * D;
    const __nv_bfloat16* wr = w + static_cast<size_t>(g) * D;
    __nv_bfloat16* yr = out + (static_cast<size_t>(m) * G + g) * D;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    ss = warp_sum(ss);
    __shared__ float sm[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        float tot = lane < nw ? sm[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) sm[0] = rsqrtf(tot / float(D) + eps);
    }
    __syncthreads();
    const float inv = sm[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]) * inv * (1.0f + __bfloat162float(wr[i]));
        yr[i] = __float2bfloat16(v);
    }
}

// Elementwise SiLU(x / 4.0) in-place on [M, 320] BF16
__global__ void k_div4_silu_bf16(__nv_bfloat16* __restrict__ x, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float v = __bfloat162float(x[i]) * 0.25f;
        x[i] = __float2bfloat16(silu_f(v));
    }
}

// Elementwise 2.0 * sigmoid(x / 4.0) in-place on [M, 4] FP32
__global__ void k_hc_inject_act(float* __restrict__ inj, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) inj[i] = 2.0f * sigmoid_f(inj[i] * 0.25f);
}

// mixed[m, d] = bf16(0.25f * sum_{s=0..3} sigmoid(up[m, s, d]) * hn[m, s, d])
__global__ void k_hc_mix(const __nv_bfloat16* __restrict__ hn, const __nv_bfloat16* __restrict__ up,
                         __nv_bfloat16* __restrict__ mixed, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* h = hn + static_cast<size_t>(m) * geom::HC_D;
    const __nv_bfloat16* u = up + static_cast<size_t>(m) * geom::HC_D;
    __nv_bfloat16* out = mixed + static_cast<size_t>(m) * geom::HID;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) {
            const float mw = __bfloat162float(__float2bfloat16(sigmoid_f(__bfloat162float(u[s * geom::HID + d]))));
            acc += mw * __bfloat162float(h[s * geom::HID + d]);
        }
        out[d] = __float2bfloat16(acc * 0.25f);
    }
}

// Residual injection: hyper[m, s, d] += bf16(y[m, d] * inj[m, s])
// When parts != nullptr and nr > 1, sums rank-ordered BF16 partials parts[r][m][d] in FP32 -> BF16 first.
__global__ void k_hc_inject(__nv_bfloat16* __restrict__ hyper, const __nv_bfloat16* __restrict__ y_or_parts,
                            const float* __restrict__ inj, int M, int nr, size_t rank_stride_elems) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __nv_bfloat16* h = hyper + static_cast<size_t>(m) * geom::HC_D;
    float w[geom::HC];
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) w[s] = inj[m * geom::HC + s];

    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        float yv = 0.0f;
        for (int r = 0; r < nr; ++r) {
            yv += __bfloat162float(y_or_parts[static_cast<size_t>(r) * rank_stride_elems + static_cast<size_t>(m) * geom::HID + d]);
        }
        const float ybf = __bfloat162float(__float2bfloat16(yv));
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) {
            const float prev = __bfloat162float(h[s * geom::HID + d]);
            const float add = __bfloat162float(__float2bfloat16(ybf * w[s]));
            h[s * geom::HID + d] = __float2bfloat16(prev + add);
        }
    }
}

// Full GatedResidual prepare: hyper_in [M, 10240] -> mixed [M, 2560] (+ optional inj [M, 4]).
inline void hc_prepare(cublasHandle_t blas, const HcWeights& w, const __nv_bfloat16* hyper_in,
                       __nv_bfloat16* mixed, HcScratch& sc, int M, cudaStream_t st) {
    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(hyper_in, w.norm, sc.hn, M, geom::HC, geom::RMS_EPS);
    cublasSetStream(blas, st);
    // down = silu(hn @ W_down^T / 4)
    gemm_bf16_nt(blas, M, geom::HC_RANK, geom::HC_D, sc.hn, w.down, sc.down);
    {
        const int n = M * geom::HC_RANK;
        k_div4_silu_bf16<<<(n + 255) / 256, 256, 0, st>>>(sc.down, n);
    }
    // up = down @ W_up^T
    gemm_bf16_nt(blas, M, geom::HC_D, geom::HC_RANK, sc.down, w.up, sc.up);
    // mixed = mean_s(sigmoid(up) * hn)
    k_hc_mix<<<M, 256, 0, st>>>(sc.hn, sc.up, mixed, M);
    // inject weights (skipped for final mixer where w.inject == nullptr)
    if (w.inject) {
        gemm_bf16_nt_f32(blas, M, geom::HC, geom::HC_D, sc.hn, w.inject, sc.inj);
        const int n = M * geom::HC;
        k_hc_inject_act<<<(n + 63) / 64, 64, 0, st>>>(sc.inj, n);
    }
}

} // namespace qw38
