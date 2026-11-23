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
#include "qw38_fp8.cuh"

namespace qw38 {

struct HcWeights {
    const __nv_bfloat16* norm = nullptr;    // [10240] BF16 (zero-centered: 1 + w)
    const __nv_bfloat16* down = nullptr;    // [320, 10240] BF16
    const __nv_bfloat16* up = nullptr;      // [10240, 320] BF16
    const __nv_bfloat16* inject = nullptr;  // [4, 10240] BF16 (nullptr for final mixer)
    Fp8Mat fp8_down;                        // optional on-load FP8 [320, 10240]
    Fp8Mat fp8_up;                          // optional on-load FP8 [10240, 320]
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

// Fast dot product of 8 BF16 elements packed in two uint4 registers (FP32 accumulate via __bfloat1622float2).
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

// Reusable coalesced BF16 GEMV for small decode batches (M <= 8):
//   C[M, N] = A[M, K] @ B[N, K]^T (K must be a multiple of 8; N rows -> N CTAs, 256 threads/CTA).
// Each CTA reads one row of B[n, 0..K-1] ONCE with 128-bit uint4 loads across all M tokens.
template <int MAX_M, bool OUT_F32 = false>
__global__ void k_bf16_gemv_smallm(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   void* __restrict__ C,
                                   int M, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int k8 = K >> 3;
    const uint4* brow = reinterpret_cast<const uint4*>(B + static_cast<size_t>(n) * K);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int i = threadIdx.x; i < k8; i += blockDim.x) {
        const uint4 wv = __ldg(brow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K)[i];
                acc[m] += dot_bf16x8(av, wv);
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float sm[8][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float v = warp_sum(acc[m]);
            if (lane == 0) sm[warp][m] = v;
        }
    }
    __syncthreads();
    if (warp == 0 && lane < M) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w][lane];
        if constexpr (OUT_F32) {
            static_cast<float*>(C)[static_cast<size_t>(lane) * N + n] = tot;
        } else {
            static_cast<__nv_bfloat16*>(C)[static_cast<size_t>(lane) * N + n] = __float2bfloat16(tot);
        }
    }
}

// Warp-per-row coalesced BF16 GEMV for small decode batches (M <= 8):
//   C[M, N] = A[M, K] @ B[N, K]^T (K multiple of 8; 4 warps/CTA = 4 output dims/CTA, zero smem, zero __syncthreads).
template <int MAX_M, bool OUT_F32 = false>
__global__ void k_bf16_gemv_1warp(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B,
                                  void* __restrict__ C,
                                  int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;
    const int k8 = K >> 3;
    const uint4* brow = reinterpret_cast<const uint4*>(B + static_cast<size_t>(n) * K);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k8; i += 32) {
        const uint4 wv = __ldg(brow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K)[i];
                acc[m] += dot_bf16x8(av, wv);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float tot = warp_sum(acc[m]);
            if (lane == 0) {
                if constexpr (OUT_F32) {
                    static_cast<float*>(C)[static_cast<size_t>(m) * N + n] = tot;
                } else {
                    static_cast<__nv_bfloat16*>(C)[static_cast<size_t>(m) * N + n] = __float2bfloat16(tot);
                }
            }
        }
    }
}

template <bool OUT_F32 = false>
inline void launch_bf16_gemv_smallm(const __nv_bfloat16* A, const __nv_bfloat16* B, void* C,
                                    int M, int N, int K, cudaStream_t st) {
    const int grid = (N + 3) / 4;
    if (M == 1) k_bf16_gemv_1warp<1, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
    else if (M <= 4) k_bf16_gemv_1warp<4, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
    else if (M <= 8) k_bf16_gemv_1warp<8, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
    else if (M <= 12) k_bf16_gemv_1warp<12, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
    else if (M <= 16) k_bf16_gemv_1warp<16, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
    else k_bf16_gemv_1warp<20, OUT_F32><<<grid, 128, 0, st>>>(A, B, C, M, N, K);
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

// Replicate token or staged vision embeddings into 4 hyper-connection streams:
// when vis_map[m] >= 0, reads vis_pool[vis_map[m], d]; otherwise reads embed[tok[m], d].
__global__ void k_embed_hc4_vis(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ embed,
                                const int* __restrict__ vis_map, const __nv_bfloat16* __restrict__ vis_pool,
                                __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int vr = vis_map ? vis_map[m] : -1;
    const __nv_bfloat16* src = (vr >= 0) ? (vis_pool + static_cast<size_t>(vr) * geom::HID)
                                         : (embed + static_cast<size_t>(tok[m]) * geom::HID);
    __nv_bfloat16* dst = out + static_cast<size_t>(m) * geom::HC_D;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        const __nv_bfloat16 v = src[d];
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) dst[s * geom::HID + d] = v;
    }
}

// Grouped zero-centered RMSNorm (vectorized uint4): x[M, G, D] -> out[M, G, D] BF16, weight[G*D] BF16.
// One CTA per (m, g). Exact match to Qwen4ExpTextRMSNorm: rsqrt(mean(x^2)+eps) * x * (1 + w) -> bf16.
template <int D>
__global__ void k_rmsnorm_1pw(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                              __nv_bfloat16* __restrict__ out, int M, int G, float eps) {
    const int mg = blockIdx.x;
    const int m = mg / G;
    const int g = mg % G;
    if (m >= M) return;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (static_cast<size_t>(m) * G + g) * D);
    const uint4* wr = reinterpret_cast<const uint4*>(w + static_cast<size_t>(g) * D);
    uint4* yr = reinterpret_cast<uint4*>(out + (static_cast<size_t>(m) * G + g) * D);
    constexpr int D8 = D >> 3;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        const uint4 xv = xr[i];
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(xp[k]);
            ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
        }
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
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        const uint4 xv = xr[i];
        const uint4 wv = __ldg(wr + i);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        uint4 yv;
        __nv_bfloat162* yp = reinterpret_cast<__nv_bfloat162*>(&yv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fx = __bfloat1622float2(xp[k]);
            const float2 fw = __bfloat1622float2(wp[k]);
            yp[k] = __floats2bfloat162_rn(fx.x * inv * (1.0f + fw.x), fx.y * inv * (1.0f + fw.y));
        }
        yr[i] = yv;
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
// Multi-CTA vectorized uint4: Grid (M * 5), Block 64 -> 5 * 64 * 8 = 2560 EXACT (100% coalesced, bit-exact!).
__global__ void k_hc_mix(const __nv_bfloat16* __restrict__ hn, const __nv_bfloat16* __restrict__ up,
                         __nv_bfloat16* __restrict__ mixed, int M) {
    const int m = blockIdx.x / 5;
    const int blk = blockIdx.x % 5;
    if (m >= M) return;
    const int d8 = blk * 64 + threadIdx.x; // 0..319 (each uint4 = 8 BF16s)
    const __nv_bfloat16* h = hn + static_cast<size_t>(m) * geom::HC_D;
    const __nv_bfloat16* u = up + static_cast<size_t>(m) * geom::HC_D;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) {
        const uint4 uv = reinterpret_cast<const uint4*>(u + s * geom::HID)[d8];
        const uint4 hv = reinterpret_cast<const uint4*>(h + s * geom::HID)[d8];
        const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
        const __nv_bfloat162* hp2 = reinterpret_cast<const __nv_bfloat162*>(&hv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fu = __bfloat1622float2(up2[k]);
            const float2 fh = __bfloat1622float2(hp2[k]);
            const float mw0 = __bfloat162float(__float2bfloat16(sigmoid_f(fu.x)));
            const float mw1 = __bfloat162float(__float2bfloat16(sigmoid_f(fu.y)));
            acc[2 * k] += mw0 * fh.x;
            acc[2 * k + 1] += mw1 * fh.y;
        }
    }
    uint4 out;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        op[k] = __floats2bfloat162_rn(acc[2 * k] * 0.25f, acc[2 * k + 1] * 0.25f);
    }
    reinterpret_cast<uint4*>(mixed + static_cast<size_t>(m) * geom::HID)[d8] = out;
}

// Residual injection (multi-CTA vectorized uint4): hyper[m, s, d] += bf16(y[m, d] * inj[m, s])
// Grid: (M * 5), Block: 64 -> 5 * 64 * 8 = 2560 EXACT (zero loop, 100% coalesced uint4!).
__global__ void k_hc_inject(__nv_bfloat16* __restrict__ hyper, const __nv_bfloat16* __restrict__ y_or_parts,
                            const float* __restrict__ inj, int M, int nr, size_t rank_stride_elems) {
    const int m = blockIdx.x / 5;
    const int blk = blockIdx.x % 5;
    if (m >= M) return;
    const int d8 = blk * 64 + threadIdx.x; // 0..319 (each uint4 = 8 BF16s)

    float w[geom::HC];
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) w[s] = inj[m * geom::HC + s];

    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int r = 0; r < nr; ++r) {
        const uint4 pv = reinterpret_cast<const uint4*>(
            y_or_parts + static_cast<size_t>(r) * rank_stride_elems + static_cast<size_t>(m) * geom::HID)[d8];
        const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(pp[k]);
            y[2 * k] += f.x;
            y[2 * k + 1] += f.y;
        }
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = __bfloat162float(__float2bfloat16(y[k]));

    __nv_bfloat16* h = hyper + static_cast<size_t>(m) * geom::HC_D;
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) {
        uint4* hp = reinterpret_cast<uint4*>(h + s * geom::HID) + d8;
        const uint4 prev = *hp;
        const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&prev);
        uint4 out;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out);
        const float ws = w[s];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fp = __bfloat1622float2(pr[k]);
            const float a0 = __bfloat162float(__float2bfloat16(y[2 * k] * ws));
            const float a1 = __bfloat162float(__float2bfloat16(y[2 * k + 1] * ws));
            op[k] = __floats2bfloat162_rn(fp.x + a0, fp.y + a1);
        }
        *hp = out;
    }
}

// Iter 1 Fused HC Decode Kernels (M <= 8):
//   1. k_hc_down_silu_inj<MAX_M>: Grid (has_inj ? 324 : 320), Block 256.
//      Computes down[m, n] = bf16(silu(bf16(hn[m] . W_down[n]) * 0.25)) for n < 320,
//      and inj[m, n-320] = 2.0 * sigmoid((hn[m] . W_inj[n-320]) * 0.25) for n >= 320.
//      Reads W_down (6.55 MB) + W_inj (80 KB) ONCE across all M tokens with 128-bit coalesced loads!
template <int MAX_M>
__global__ void k_hc_down_silu_inj(const __nv_bfloat16* __restrict__ hn,
                                   const __nv_bfloat16* __restrict__ w_down,
                                   const __nv_bfloat16* __restrict__ w_inj,
                                   __nv_bfloat16* __restrict__ down_out,
                                   float* __restrict__ inj_out,
                                   int M) {
    const int n = blockIdx.x;
    const bool is_inj = (n >= geom::HC_RANK);
    const uint4* wrow = reinterpret_cast<const uint4*>(
        is_inj ? (w_inj + static_cast<size_t>(n - geom::HC_RANK) * geom::HC_D)
               : (w_down + static_cast<size_t>(n) * geom::HC_D));
    constexpr int K8 = geom::HC_D / 8; // 1280 = 5 * 256 (exact!)

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    #pragma unroll
    for (int it = 0; it < K8 / 256; ++it) {
        const int i = it * 256 + threadIdx.x;
        const uint4 wv = __ldg(wrow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(hn + static_cast<size_t>(m) * geom::HC_D)[i];
                acc[m] += dot_bf16x8(av, wv);
            }
        }
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float sm[8][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float v = warp_sum(acc[m]);
            if (lane == 0) sm[warp][m] = v;
        }
    }
    __syncthreads();
    if (warp == 0 && lane < M) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w][lane];
        if (is_inj) {
            inj_out[lane * geom::HC + (n - geom::HC_RANK)] = 2.0f * sigmoid_f(tot * 0.25f);
        } else {
            const float bf = __bfloat162float(__float2bfloat16(tot));
            down_out[static_cast<size_t>(lane) * geom::HC_RANK + n] = __float2bfloat16(silu_f(bf * 0.25f));
        }
    }
}

//   2. k_hc_up_mix_fused<MAX_M>: Grid (2560 / 8 = 320), Block 256 (8 warps, 1 warp per hidden dim d).
//      Loads down[0..M-1, 0..319] into shared memory once, computes up[m, s, d] = down[m] . W_up[s*2560+d]
//      in registers (NEVER writing/reading the 10240-wide up tensor in HBM!), and writes mixed[m, d] directly!
template <int MAX_M>
__global__ void k_hc_up_mix_fused(const __nv_bfloat16* __restrict__ hn,
                                  const __nv_bfloat16* __restrict__ down,
                                  const __nv_bfloat16* __restrict__ w_up,
                                  __nv_bfloat16* __restrict__ mixed,
                                  int M) {
    const int warp = threadIdx.x >> 5;   // 0..7
    const int lane = threadIdx.x & 31;   // 0..31
    const int d = blockIdx.x * 8 + warp;
    if (d >= geom::HID) return;

    // Cache down[M, 320] in shared memory (at most 8 * 640 = 5120 bytes)
    __shared__ uint4 s_down[MAX_M][geom::HC_RANK / 8]; // [MAX_M][40]
    constexpr int R8 = geom::HC_RANK / 8;              // 40
    const int tot_u4 = M * R8;
    for (int i = threadIdx.x; i < tot_u4; i += blockDim.x) {
        const int m = i / R8;
        const int c = i % R8;
        s_down[m][c] = reinterpret_cast<const uint4*>(down + static_cast<size_t>(m) * geom::HC_RANK)[c];
    }
    __syncthreads();

    float mix_acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) mix_acc[m] = 0.0f;

    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) {
        const uint4* wrow = reinterpret_cast<const uint4*>(w_up + static_cast<size_t>(s * geom::HID + d) * geom::HC_RANK);
        const uint4 w0 = __ldg(wrow + lane);                          // 0..255 (all 32 lanes)
        const uint4 w1 = (lane < 8) ? __ldg(wrow + 32 + lane) : uint4{0, 0, 0, 0}; // 256..319 (lanes 0..7)
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float p = dot_bf16x8(s_down[m][lane], w0);
                if (lane < 8) p += dot_bf16x8(s_down[m][32 + lane], w1);
                const float u_val = warp_sum(p);
                if (lane == 0) {
                    const float u_bf = __bfloat162float(__float2bfloat16(u_val));
                    const float mw = __bfloat162float(__float2bfloat16(sigmoid_f(u_bf)));
                    mix_acc[m] += mw * __bfloat162float(hn[static_cast<size_t>(m) * geom::HC_D + s * geom::HID + d]);
                }
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) mixed[static_cast<size_t>(m) * geom::HID + d] = __float2bfloat16(mix_acc[m] * 0.25f);
        }
    }
}

// Iter 6 Roofline FP8 HC Decode Kernels (M <= 8):
//   1. k_fp8_hc_down_silu_inj<MAX_M>: Block=128 (4 warps -> 640/128 = 5 EXACT iterations, 0% idle threads!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_fp8_hc_down_silu_inj(const __nv_bfloat16* __restrict__ hn,
                                       const uint8_t* __restrict__ dq, const __half* __restrict__ dsc,
                                       const __nv_bfloat16* __restrict__ w_inj,
                                       __nv_bfloat16* __restrict__ down_out,
                                       float* __restrict__ inj_out,
                                       int M) {
    const int n = blockIdx.x;
    const bool is_inj = (n >= geom::HC_RANK);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if (is_inj) {
        const uint4* wrow = reinterpret_cast<const uint4*>(w_inj + static_cast<size_t>(n - geom::HC_RANK) * geom::HC_D);
        constexpr int K8 = geom::HC_D / 8; // 1280 = 10 * 128
        #pragma unroll
        for (int it = 0; it < K8 / 128; ++it) {
            const int i = it * 128 + threadIdx.x;
            const uint4 wv = __ldg(wrow + i);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const uint4 av = reinterpret_cast<const uint4*>(hn + static_cast<size_t>(m) * geom::HC_D)[i];
                    acc[m] += dot_bf16x8(av, wv);
                }
            }
        }
    } else {
        constexpr int K16 = geom::HC_D >> 4; // 640 = 5 * 128 EXACT!
        constexpr int NB = geom::HC_D >> 5;  // 320
        const uint4* qrow = reinterpret_cast<const uint4*>(dq + static_cast<size_t>(n) * geom::HC_D);
        const __half* srow = dsc + static_cast<size_t>(n) * NB;
        #pragma unroll
        for (int it = 0; it < K16 / 128; ++it) {
            const int i = it * 128 + threadIdx.x;
            const uint4 wv = __ldg(qrow + i);
            const float sc = __half2float(__ldg(srow + (i >> 1)));
            float2 wf[8];
            fp8x16_unpack(wv, sc, wf);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const uint4* am = reinterpret_cast<const uint4*>(hn + static_cast<size_t>(m) * geom::HC_D);
                    acc[m] += dot_bf16x16_f2x8(am[2 * i], am[2 * i + 1], wf);
                }
            }
        }
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float sm[4][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float v = warp_sum(acc[m]);
            if (lane == 0) sm[warp][m] = v;
        }
    }
    __syncthreads();
    if (warp == 0 && lane < M) {
        const float tot = sm[0][lane] + sm[1][lane] + sm[2][lane] + sm[3][lane];
        if (is_inj) {
            inj_out[lane * geom::HC + (n - geom::HC_RANK)] = 2.0f * sigmoid_f(tot * 0.25f);
        } else {
            const float bf = __bfloat162float(__float2bfloat16(tot));
            down_out[static_cast<size_t>(lane) * geom::HC_RANK + n] = __float2bfloat16(silu_f(bf * 0.25f));
        }
    }
}

//   2. k_fp8_hc_up_mix_fused<MAX_M> using contiguous transposed fp8_up [2560, 4, 320]!
//      Grid: (2560 / 2 = 1280), Block: 256 (8 warps = 2 dims x 4 streams in parallel, ZERO s-loop, 100% contiguous burst!).
template <int MAX_M>
__global__ void __launch_bounds__(256) k_fp8_hc_up_mix_fused(const __nv_bfloat16* __restrict__ hn,
                                      const __nv_bfloat16* __restrict__ down,
                                      const uint8_t* __restrict__ uq, const __half* __restrict__ usc,
                                      __nv_bfloat16* __restrict__ mixed,
                                      int M) {
    const int warp = threadIdx.x >> 5;   // 0..7 -> dim_local = warp >> 2 (0..1), s = warp & 3 (0..3)
    const int lane = threadIdx.x & 31;
    const int dl = warp >> 2;
    const int s = warp & 3;
    const int d = blockIdx.x * 2 + dl;
    if (d >= geom::HID) return;

    __shared__ uint4 s_down[MAX_M][geom::HC_RANK / 8]; // [MAX_M][40]
    constexpr int R8 = geom::HC_RANK / 8;              // 40
    const int tot_u4 = M * R8;
    for (int i = threadIdx.x; i < tot_u4; i += 256) {
        s_down[i / R8][i % R8] = reinterpret_cast<const uint4*>(down + static_cast<size_t>(i / R8) * geom::HC_RANK)[i % R8];
    }
    __syncthreads();

    constexpr int R16 = geom::HC_RANK >> 4; // 20
    constexpr int NB = geom::HC_RANK >> 5;  // 10
    const size_t row = static_cast<size_t>(d) * 4 + s; // contiguous [2560, 4, 320]!
    float2 wf[8];
    if (lane < R16) {
        const uint4 wv = __ldg(reinterpret_cast<const uint4*>(uq + row * geom::HC_RANK) + lane);
        const float sc = __half2float(__ldg(usc + row * NB + (lane >> 1)));
        fp8x16_unpack(wv, sc, wf);
    }

    __shared__ float s_contrib[2][4][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float p = (lane < R16) ? dot_bf16x16_f2x8(s_down[m][2 * lane], s_down[m][2 * lane + 1], wf) : 0.0f;
            const float u_val = warp_sum(p);
            if (lane == 0) {
                const float u_bf = __bfloat162float(__float2bfloat16(u_val));
                const float mw = __bfloat162float(__float2bfloat16(sigmoid_f(u_bf)));
                s_contrib[dl][s][m] = mw * __bfloat162float(hn[static_cast<size_t>(m) * geom::HC_D + s * geom::HID + d]);
            }
        }
    }
    __syncthreads();
    if (s == 0 && lane < M) {
        const float acc = s_contrib[dl][0][lane] + s_contrib[dl][1][lane] + s_contrib[dl][2][lane] + s_contrib[dl][3][lane];
        mixed[static_cast<size_t>(lane) * geom::HID + d] = __float2bfloat16(acc * 0.25f);
    }
}

// Full GatedResidual prepare: hyper_in [M, 10240] -> mixed [M, 2560] (+ optional inj [M, 4]).
inline void hc_prepare(cublasHandle_t blas, const HcWeights& w, const __nv_bfloat16* hyper_in,
                       __nv_bfloat16* mixed, HcScratch& sc, int M, cudaStream_t st) {
    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(hyper_in, w.norm, sc.hn, M, geom::HC, geom::RMS_EPS);
    if (M <= 16 && w.fp8_down.q && w.fp8_up.q) {
        const int down_grid = w.inject ? (geom::HC_RANK + geom::HC) : geom::HC_RANK;
        const int up_grid = geom::HID / 2;
        if (M == 1) {
            k_fp8_hc_down_silu_inj<1><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, 1);
            k_fp8_hc_up_mix_fused<1><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, 1);
        } else if (M <= 4) {
            k_fp8_hc_down_silu_inj<4><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, M);
            k_fp8_hc_up_mix_fused<4><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, M);
        } else if (M == 5) {
            k_fp8_hc_down_silu_inj<5><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, M);
            k_fp8_hc_up_mix_fused<5><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, M);
        } else if (M <= 8) {
            k_fp8_hc_down_silu_inj<8><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, M);
            k_fp8_hc_up_mix_fused<8><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, M);
        } else if (M <= 12) {
            k_fp8_hc_down_silu_inj<12><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, M);
            k_fp8_hc_up_mix_fused<12><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, M);
        } else {
            k_fp8_hc_down_silu_inj<16><<<down_grid, 128, 0, st>>>(sc.hn, w.fp8_down.q, w.fp8_down.sc, w.inject, sc.down, sc.inj, M);
            k_fp8_hc_up_mix_fused<16><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.fp8_up.q, w.fp8_up.sc, mixed, M);
        }
        return;
    }
    if (M <= 8) {
        const int down_grid = w.inject ? (geom::HC_RANK + geom::HC) : geom::HC_RANK;
        const int up_grid = geom::HID / 8;
        if (M == 1) {
            k_hc_down_silu_inj<1><<<down_grid, 256, 0, st>>>(sc.hn, w.down, w.inject, sc.down, sc.inj, 1);
            k_hc_up_mix_fused<1><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.up, mixed, 1);
        } else if (M <= 4) {
            k_hc_down_silu_inj<4><<<down_grid, 256, 0, st>>>(sc.hn, w.down, w.inject, sc.down, sc.inj, M);
            k_hc_up_mix_fused<4><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.up, mixed, M);
        } else {
            k_hc_down_silu_inj<8><<<down_grid, 256, 0, st>>>(sc.hn, w.down, w.inject, sc.down, sc.inj, M);
            k_hc_up_mix_fused<8><<<up_grid, 256, 0, st>>>(sc.hn, sc.down, w.up, mixed, M);
        }
        return;
    }
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
    k_hc_mix<<<M * 5, 64, 0, st>>>(sc.hn, sc.up, mixed, M);
    // inject weights (skipped for final mixer where w.inject == nullptr)
    if (w.inject) {
        gemm_bf16_nt_f32(blas, M, geom::HC, geom::HC_D, sc.hn, w.inject, sc.inj);
        const int n = M * geom::HC;
        k_hc_inject_act<<<(n + 63) / 64, 64, 0, st>>>(sc.inj, n);
    }
}

} // namespace qw38
