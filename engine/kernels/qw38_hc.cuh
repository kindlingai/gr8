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

// Residual injection (vectorized uint4): hyper[m, s, d] += bf16(y[m, d] * inj[m, s])
// When parts != nullptr and nr > 1, sums rank-ordered BF16 partials parts[r][m][d] in FP32 -> BF16 first.
__global__ void k_hc_inject(__nv_bfloat16* __restrict__ hyper, const __nv_bfloat16* __restrict__ y_or_parts,
                            const float* __restrict__ inj, int M, int nr, size_t rank_stride_elems) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __nv_bfloat16* h = hyper + static_cast<size_t>(m) * geom::HC_D;
    float w[geom::HC];
    #pragma unroll
    for (int s = 0; s < geom::HC; ++s) w[s] = inj[m * geom::HC + s];

    constexpr int D2 = geom::HID / 2;
    for (int d2 = threadIdx.x; d2 < D2; d2 += blockDim.x) {
        float y0 = 0.0f, y1 = 0.0f;
        for (int r = 0; r < nr; ++r) {
            const __nv_bfloat162 pv = reinterpret_cast<const __nv_bfloat162*>(
                y_or_parts + static_cast<size_t>(r) * rank_stride_elems + static_cast<size_t>(m) * geom::HID)[d2];
            const float2 pf = __bfloat1622float2(pv);
            y0 += pf.x;
            y1 += pf.y;
        }
        const float ybf0 = __bfloat162float(__float2bfloat16(y0));
        const float ybf1 = __bfloat162float(__float2bfloat16(y1));
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) {
            __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(h + s * geom::HID) + d2;
            const float2 prev = __bfloat1622float2(*hp);
            const float a0 = __bfloat162float(__float2bfloat16(ybf0 * w[s]));
            const float a1 = __bfloat162float(__float2bfloat16(ybf1 * w[s]));
            *hp = __floats2bfloat162_rn(prev.x + a0, prev.y + a1);
        }
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

// Full GatedResidual prepare: hyper_in [M, 10240] -> mixed [M, 2560] (+ optional inj [M, 4]).
inline void hc_prepare(cublasHandle_t blas, const HcWeights& w, const __nv_bfloat16* hyper_in,
                       __nv_bfloat16* mixed, HcScratch& sc, int M, cudaStream_t st) {
    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(hyper_in, w.norm, sc.hn, M, geom::HC, geom::RMS_EPS);
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
    k_hc_mix<<<M, 256, 0, st>>>(sc.hn, sc.up, mixed, M);
    // inject weights (skipped for final mixer where w.inject == nullptr)
    if (w.inject) {
        gemm_bf16_nt_f32(blas, M, geom::HC, geom::HC_D, sc.hn, w.inject, sc.inj);
        const int n = M * geom::HC;
        k_hc_inject_act<<<(n + 63) / 64, 64, 0, st>>>(sc.inj, n);
    }
}

} // namespace qw38
