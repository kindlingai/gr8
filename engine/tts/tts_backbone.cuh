// Drama 3 (fish_omni) backbone forward pass in CUDA + cuBLAS BF16 (reference & native paths).
//
// Exact match to fish-speech @ 0f8a12e8 (`fish_omni/{backbone,layers,moe}.py`):
//   - Backbone input: E_LM[main], and for semantic rows (SEM0 <= main <= SEM1)
//     (E_LM[main] + codebook_embeddings[4096 + c1]) / sqrt(2), held in an FP32 residual stream.
//   - 40 layers: l % 4 == 3 -> FullAttention (16 Q heads x 256, 2 KV heads x 256, q_proj=[q|gate] per head,
//     ZeroCenteredRMSNorm 1+w on q/k, NeoX half-split RoPE on first 64 dims theta=1e7, o_proj(attn*sigmoid(gate)));
//     l % 4 != 3 -> GatedDeltaNet (16 KH / 32 VH x 128, V_per_K=2, depthwise conv1d k=4 + SiLU, FP32 state [32,128,128],
//     GatedRMSNorm with plain weight * silu(z), out_proj).
//   - Every layer has SparseMoE: FP32 router softmax -> top-8 -> renormalize, 256 routed experts (either FP8_BLOCK
//     dequantized on the fly per active expert or pre-decoded BF16, or native EXLR8 trellis), sigmoid-gated shared expert.
//   - Final ZeroCenteredRMSNorm + sliced lm_head over the 4097 legal main-token candidates.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts_config.h"
#include "tts_dense.cuh"
#include "tts_moe.cuh"
#include "tts_weights.h"

namespace g6tts {

inline void tts_blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("g6tts cuBLAS error: ") + what + " (" + std::to_string(int(s)) + ")");
    }
}
#define TTS_BLAS_CK(x) ::g6tts::tts_blas_ck((x), #x)

// Row-major GEMM: Y[M, N] = alpha * X[M, K] @ W[N, K]^T + beta * Y[M, N]
// W and X are BF16; Y may be BF16 (`out_f32 == false`) or FP32 (`out_f32 == true`).
inline void gemm_bf16(cublasHandle_t blas, const bf16* X, const bf16* W, void* Y,
                      int M, int N, int K, bool out_f32, float beta = 0.0f) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const float alpha = 1.0f;
    TTS_BLAS_CK(cublasGemmEx(blas,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             W, CUDA_R_16BF, K,
                             X, CUDA_R_16BF, K,
                             &beta,
                             Y, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Unified Ref GEMM: runs exact FP32 cublasSgemm when W.f32 is populated (WeightMode::RefF32),
// or standard BF16 cublasGemmEx when W.bf16 is populated (WeightMode::Ref / RefFp16).
inline void gemm_ref(cublasHandle_t blas, const bf16* X, const RefMat& W, void* Y,
                     int M, int N, int K, bool out_f32, float beta,
                     float* d_x_f32, float* d_y_f32, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (W.f32 != nullptr) {
        const size_t x_n = size_t(M) * size_t(K);
        k_tts_bf16_to_f32_buf<<<static_cast<unsigned>((x_n + 255) / 256), 256, 0, stream>>>(X, d_x_f32, x_n);
        const float alpha = 1.0f;
        float* dst_f32 = out_f32 ? static_cast<float*>(Y) : d_y_f32;
        TTS_BLAS_CK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                N, M, K, &alpha, W.f32, K, d_x_f32, K, &beta, dst_f32, N));
        if (!out_f32) {
            const size_t y_n = size_t(M) * size_t(N);
            k_tts_f32_to_bf16_buf<<<static_cast<unsigned>((y_n + 255) / 256), 256, 0, stream>>>(
                d_y_f32, static_cast<bf16*>(Y), y_n);
        }
        return;
    }
    gemm_bf16(blas, X, W.bf16, Y, M, N, K, out_f32, beta);
}

// ------------------------------------------------------------------------------------------------ Common CUDA kernels

__device__ __forceinline__ float tts_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float tts_block_sum256(float v, float* red) {
    v = tts_warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x < 32) {
        tot = (threadIdx.x < 8) ? red[threadIdx.x] : 0.0f;
        tot = tts_warp_sum(tot);
        if (threadIdx.x == 0) red[0] = tot;
    }
    __syncthreads();
    const float out = red[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float tts_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float tts_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float tts_softplus(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float tts_e4m3_to_f32(uint8_t b) {
    const uint32_t e = (b >> 3) & 0x0fu;
    const uint32_t m = b & 0x07u;
    float v = 0.0f;
    if (e == 0u) {
        v = static_cast<float>(m) * (1.0f / 512.0f);
    } else if (e == 15u && m == 7u) {
        v = 0.0f;
    } else {
        v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    }
    return (b & 0x80u) ? -v : v;
}

// Dequantizes one FP8_BLOCK [out, in] matrix with 128x128 FP32 scales into BF16 [out, in].
// Matches `fish_omni/moe.py:dequantize_fp8_blocks(..., torch.bfloat16)` with full FP32 exponent range.
__global__ void k_dequant_fp8_block128(const uint8_t* __restrict__ w_fp8,
                                       const float* __restrict__ w_sc,
                                       bf16* __restrict__ out_bf16,
                                       int out_dim, int in_dim, int bi) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(out_dim) * size_t(in_dim);
    if (idx >= total) return;
    const int r = static_cast<int>(idx / size_t(in_dim));
    const int c = static_cast<int>(idx % size_t(in_dim));
    const float s = w_sc[(r >> 7) * bi + (c >> 7)];
    out_bf16[idx] = __float2bfloat16_rn(tts_e4m3_to_f32(w_fp8[idx]) * s);
}

// Backbone input embedding into FP32 residual stream:
//   rows[M, 11]: [main, c0, c1, ..., c9]
//   if SEM0 <= main <= SEM1: (E_LM[main] + CB_EMB[4096 + c1]) / sqrt(2)
//   else:                    E_LM[main]
__global__ void k_backbone_embed(const int32_t* __restrict__ rows,
                                 const bf16* __restrict__ embed_tokens,
                                 const bf16* __restrict__ cb_emb,
                                 float* __restrict__ out_f32,
                                 int M, int H, int row_w, int sem0, int sem1, int cb_vocab) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int32_t* r = rows + size_t(m) * row_w;
    const int main_tok = r[0];
    const bool is_sem = (main_tok >= sem0 && main_tok <= sem1);
    const int c1 = is_sem ? max(0, min(cb_vocab - 1, r[2])) : 0;
    const bf16* e_tok = embed_tokens + size_t(main_tok) * H;
    const bf16* e_cb = cb_emb + size_t(cb_vocab + c1) * H;
    float* dst = out_f32 + size_t(m) * H;
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float v = __bfloat162float(e_tok[d]);
        if (is_sem) {
            v = tts_rbf16(tts_rbf16(v + __bfloat162float(e_cb[d])) * inv_sqrt2);
        }
        dst[d] = v;
    }
}

// ZeroCenteredRMSNorm from FP32 input -> BF16 output: y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)).
// Grid: (M), Block: 256.
__global__ void k_zc_rmsnorm_f32_to_bf16(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         bf16* __restrict__ y,
                                         int M, int H, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + size_t(m) * H;
    bf16* yr = y + size_t(m) * H;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < H; d += 256) {
        const float v = xr[d];
        ss += v * v;
    }
    const float rs = rsqrtf(tts_block_sum256(ss, red) / float(H) + eps);
    for (int d = threadIdx.x; d < H; d += 256) {
        yr[d] = __float2bfloat16_rn(xr[d] * rs * (1.0f + w[d]));
    }
}

// ZeroCenteredRMSNorm from FP32 input -> BF16-rounded FP32 output (for teacher-forced hidden dump).
__global__ void k_zc_rmsnorm_f32_to_f32(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ y,
                                        int M, int H, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + size_t(m) * H;
    float* yr = y + size_t(m) * H;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < H; d += 256) {
        const float v = xr[d];
        ss += v * v;
    }
    const float rs = rsqrtf(tts_block_sum256(ss, red) / float(H) + eps);
    for (int d = threadIdx.x; d < H; d += 256) {
        yr[d] = tts_rbf16(xr[d] * rs * (1.0f + w[d]));
    }
}

// SwiGLU activation: out[i] = bf16(bf16(silu(gate[i])) * up[i]).
__global__ void k_silu_mul_bf16(const bf16* __restrict__ gate,
                                const bf16* __restrict__ up,
                                bf16* __restrict__ out,
                                size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    out[i] = __float2bfloat16_rn(tts_rbf16(tts_silu(g)) * u);
}

// ------------------------------------------------------------------------------------------------ Full Attention kernels

// Backbone FullAttention Q/K ZeroCenteredRMSNorm + NeoX half-split partial RoPE + KV cache append.
// qg_in: [M, nq, 2*hd] (per-head [Q(256) | Gate(256)])
// k_in, v_in: [M, nkv, hd]
// Writes Q_out: [M, nq, hd], K_cache[slot, pos, nkv, hd], V_cache[slot, pos, nkv, hd].
// Grid: (M, nq + nkv), Block: 256 (1 thread per head_dim element for hd=256).
__global__ void k_bb_qk_norm_rope_kv(const bf16* __restrict__ qg_in,
                                     const bf16* __restrict__ k_in,
                                     const bf16* __restrict__ v_in,
                                     const float* __restrict__ q_norm_w,
                                     const float* __restrict__ k_norm_w,
                                     bf16* __restrict__ q_out,
                                     bf16* __restrict__ k_cache,
                                     bf16* __restrict__ v_cache,
                                     const int* __restrict__ d_slots,
                                     const int* __restrict__ d_pos,
                                     int M, int nq, int nkv, int hd, int rotary_dim,
                                     float eps, float theta, int cap) {
    __shared__ float red[8];
    __shared__ float s_buf[256];
    const int m = blockIdx.x;
    const int h_glob = blockIdx.y;
    const int d = threadIdx.x;
    if (m >= M || h_glob >= nq + nkv || d >= hd) return;

    const bool is_k = (h_glob >= nq);
    const int h = is_k ? (h_glob - nq) : h_glob;
    const int slot = d_slots[m];
    const int pos = d_pos[m];

    const bf16* src = is_k
        ? (k_in + (size_t(m) * nkv + h) * hd)
        : (qg_in + (size_t(m) * nq + h) * (2 * hd)); // Q is first half of [Q | Gate] per head
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float xv = __bfloat162float(src[d]);
    const float rs = rsqrtf(tts_block_sum256(xv * xv, red) / float(hd) + eps);
    const float normed = tts_rbf16(xv * rs * (1.0f + nw[d]));
    s_buf[d] = normed;
    __syncthreads();

    float out_v = normed;
    const int half_rot = rotary_dim >> 1;
    if (d < rotary_dim) {
        const int j = (d < half_rot) ? d : (d - half_rot);
        const float inv_freq = 1.0f / powf(theta, float(2 * j) / float(rotary_dim));
        float sn, cs;
        sincosf(float(pos) * inv_freq, &sn, &cs);
        const float first = s_buf[j];
        const float second = s_buf[j + half_rot];
        out_v = (d < half_rot) ? (first * cs - second * sn) : (second * cs + first * sn);
    }

    if (is_k) {
        const size_t kv_off = ((size_t(slot) * cap + pos) * nkv + h) * hd + d;
        k_cache[kv_off] = __float2bfloat16_rn(out_v);
        v_cache[kv_off] = v_in[(size_t(m) * nkv + h) * hd + d];
    } else {
        q_out[(size_t(m) * nq + h) * hd + d] = __float2bfloat16_rn(out_v);
    }
}

// Causal GQA attention with sigmoid output gate (head_dim=256).
// Gate layout: if `gate_interleaved_per_head == true` (Backbone): gate is second half of `gate_src[m, hq, 2*hd]`.
//              if `gate_interleaved_per_head == false` (Tower):   gate is `gate_src[m * gate_stride + hq * hd + d]`.
// Grid: (M, nq), Block: 256 (1 thread per head_dim element).
__global__ void k_gqa_attn_gated(const bf16* __restrict__ q_in,
                                 const bf16* __restrict__ gate_src,
                                 const bf16* __restrict__ k_cache,
                                 const bf16* __restrict__ v_cache,
                                 bf16* __restrict__ out,
                                 const int* __restrict__ d_slots,
                                 const int* __restrict__ d_pos,
                                 int M, int nq, int nkv, int hd, int cap,
                                 bool gate_interleaved_per_head, int gate_stride, int gate_offset) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    const int hq = blockIdx.y;
    const int d = threadIdx.x;
    if (m >= M || hq >= nq || d >= hd) return;

    const int hkv = hq / (nq / nkv);
    const int slot = d_slots[m];
    const int pos = d_pos[m];
    const float scale = rsqrtf(float(hd));

    const float qv = __bfloat162float(q_in[(size_t(m) * nq + hq) * hd + d]);
    const size_t slot_base = size_t(slot) * cap * nkv * hd;
    const size_t kv_stride = size_t(nkv) * hd;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; ++t) {
        const size_t koff = slot_base + size_t(t) * kv_stride + size_t(hkv) * hd + d;
        const float kv = __bfloat162float(k_cache[koff]);
        const float dot = tts_block_sum256(qv * kv, red) * scale;
        const float nmax = fmaxf(m_max, dot);
        const float alpha = __expf(m_max - nmax);
        const float p = __expf(dot - nmax);
        m_max = nmax;
        l_sum = l_sum * alpha + p;
        acc = acc * alpha + p * __bfloat162float(v_cache[koff]);
    }

    const float attn_v = (l_sum > 0.0f) ? (acc / l_sum) : 0.0f;
    float g_val = 0.0f;
    if (gate_interleaved_per_head) {
        g_val = __bfloat162float(gate_src[(size_t(m) * nq + hq) * (2 * hd) + hd + d]);
    } else {
        g_val = __bfloat162float(gate_src[size_t(m) * gate_stride + gate_offset + size_t(hq) * hd + d]);
    }
    out[(size_t(m) * nq + hq) * hd + d] = __float2bfloat16_rn(tts_rbf16(attn_v) * tts_rbf16(tts_sigmoid(g_val)));
}

// ------------------------------------------------------------------------------------------------ GDN kernels

// Causal depthwise conv1d (k=4) + SiLU over a contiguous segment of `len` rows for one slot.
// Grid: ((qkv_dim + 255) / 256), Block: 256.
__global__ void k_gdn_conv1d_seg(const bf16* __restrict__ qkv_raw,
                                 const float* __restrict__ conv_w,
                                 float* __restrict__ conv_hist,
                                 bf16* __restrict__ qkv_out,
                                 int qkv_dim, int slot, int row0, int pos0, int len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) return;
    const float* wc = conv_w + size_t(c) * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float* hb = conv_hist + size_t(slot) * 3 * qkv_dim;
    float h0 = (pos0 > 0) ? hb[0 * qkv_dim + c] : 0.0f;
    float h1 = (pos0 > 0) ? hb[1 * qkv_dim + c] : 0.0f;
    float h2 = (pos0 > 0) ? hb[2 * qkv_dim + c] : 0.0f;

    for (int t = 0; t < len; ++t) {
        const size_t idx = size_t(row0 + t) * qkv_dim + c;
        const float x = __bfloat162float(qkv_raw[idx]);
        const float y = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        qkv_out[idx] = __float2bfloat16_rn(tts_silu(tts_rbf16(y)));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    hb[0 * qkv_dim + c] = h0;
    hb[1 * qkv_dim + c] = h1;
    hb[2 * qkv_dim + c] = h2;
}

// Batched single-token decode conv1d (k=4) + SiLU for M active slots in one launch.
__global__ void k_gdn_conv1d_dec(const bf16* __restrict__ qkv_raw,
                                 const float* __restrict__ conv_w,
                                 float* __restrict__ conv_hist,
                                 bf16* __restrict__ qkv_out,
                                 const int* __restrict__ d_slots,
                                 const int* __restrict__ d_pos,
                                 int M, int qkv_dim) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(M) * qkv_dim) return;
    const int m = static_cast<int>(idx / size_t(qkv_dim));
    const int c = static_cast<int>(idx % size_t(qkv_dim));
    const int slot = d_slots[m];
    const int pos = d_pos[m];

    const float* wc = conv_w + size_t(c) * 4;
    float* hb = conv_hist + size_t(slot) * 3 * qkv_dim;
    const float h0 = (pos > 0) ? hb[0 * qkv_dim + c] : 0.0f;
    const float h1 = (pos > 0) ? hb[1 * qkv_dim + c] : 0.0f;
    const float h2 = (pos > 0) ? hb[2 * qkv_dim + c] : 0.0f;
    const float x = __bfloat162float(qkv_raw[idx]);
    const float y = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3];
    qkv_out[idx] = __float2bfloat16_rn(tts_silu(tts_rbf16(y)));
    hb[0 * qkv_dim + c] = h1;
    hb[1 * qkv_dim + c] = h2;
    hb[2 * qkv_dim + c] = x;
}

// Recurrent Gated Delta Rule + GatedRMSNorm(plain weight * silu(z)) over a contiguous segment for one slot.
// Grid: (nh_v = 32), Block: 256 (v = tid & 127, kh = tid >> 7, each thread holds 64 rows of S[128,128]).
// Uses distinct sp_kv / sp_delta / sp_out shared arrays to guarantee zero WAR/WAW hazards across kh=0/1 warps.
__global__ void k_gdn_rec_norm_seg(const bf16* __restrict__ qkv_conv,
                                   const float* __restrict__ a_in,
                                   const float* __restrict__ b_in,
                                   const bf16* __restrict__ z_in,
                                   const float* __restrict__ A_log,
                                   const float* __restrict__ dt_bias,
                                   const float* __restrict__ norm_w,
                                   float* __restrict__ S_all,
                                   bf16* __restrict__ core_out,
                                   int nh_k, int nh_v, int slot, int row0, int pos0, int len, float eps) {
    const int hv = blockIdx.x;
    if (hv >= nh_v) return;
    const int hk = hv / (nh_v / nh_k); // V_per_K = 2
    const int qkv_dim = (2 * nh_k + nh_v) * 128;

    const int tid = threadIdx.x;
    const int v = tid & 127;
    const int kh = tid >> 7;
    const int k_base = kh * 64;

    __shared__ float sq[128];
    __shared__ float sk[128];
    __shared__ float sp_kv[128];
    __shared__ float sp_delta[128];
    __shared__ float sp_out[128];
    __shared__ float s_meta[2];
    __shared__ float qwarp[4];
    __shared__ float kwarp[4];

    float* S_head = S_all + (size_t(slot) * nh_v + hv) * (128 * 128);
    float s_col[64];
    if (pos0 == 0) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = S_head[(k_base + i) * 128 + v];
    }

    const float neg_A = -expf(A_log[hv]);
    const float dtb = dt_bias[hv];
    constexpr float inv_sqrt_dk = 0.08838834764831845f; // 1 / sqrt(128)

    for (int t = 0; t < len; ++t) {
        const int r = row0 + t;
        const bf16* qkv = qkv_conv + size_t(r) * qkv_dim;
        const bf16* q_ptr = qkv + size_t(hk) * 128;
        const bf16* k_ptr = qkv + size_t(nh_k + hk) * 128;
        const bf16* v_ptr = qkv + size_t(2 * nh_k + hv) * 128;

        if (kh == 0) sq[v] = __bfloat162float(q_ptr[v]);
        else sk[v] = __bfloat162float(k_ptr[v]);
        if (tid == 0) {
            const float av = tts_rbf16(a_in[size_t(r) * nh_v + hv]);
            const float bv = tts_rbf16(b_in[size_t(r) * nh_v + hv]);
            s_meta[0] = expf(neg_A * tts_softplus(av + dtb));
            s_meta[1] = tts_sigmoid(bv);
        }
        __syncthreads();

        float qv = 0.0f, kv = 0.0f;
        if (kh == 0) {
            qv = sq[v];
            const float q2 = tts_warp_sum(qv * qv);
            if ((v & 31) == 0) qwarp[v >> 5] = q2;
        } else {
            kv = sk[v];
            const float k2 = tts_warp_sum(kv * kv);
            if ((v & 31) == 0) kwarp[v >> 5] = k2;
        }
        __syncthreads();
        if (kh == 0) {
            const float qsum = qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3];
            sq[v] = qv * rsqrtf(qsum + 1e-6f) * inv_sqrt_dk;
        } else {
            const float ksum = kwarp[0] + kwarp[1] + kwarp[2] + kwarp[3];
            sk[v] = kv * rsqrtf(ksum + 1e-6f);
        }
        __syncthreads();

        const float decay = s_meta[0];
        const float beta = s_meta[1];
        float kv_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float sv = s_col[i] * decay;
            s_col[i] = sv;
            kv_part = fmaf(sv, sk[k_base + i], kv_part);
        }
        if (kh == 1) sp_kv[v] = kv_part;
        __syncthreads();
        if (kh == 0) {
            const float vv = __bfloat162float(v_ptr[v]);
            sp_delta[v] = (vv - (kv_part + sp_kv[v])) * beta;
        }
        __syncthreads();
        const float delta_v = sp_delta[v];

        float out_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float sv = fmaf(sk[k_base + i], delta_v, s_col[i]);
            s_col[i] = sv;
            out_part = fmaf(sv, sq[k_base + i], out_part);
        }
        if (kh == 1) sp_out[v] = out_part;
        __syncthreads();

        // GatedRMSNorm: bf16(norm_w * bf16(rmsnorm(bf16(o)))) * silu(z)
        float ov = 0.0f;
        if (kh == 0) {
            ov = tts_rbf16(out_part + sp_out[v]);
            const float ss = tts_warp_sum(ov * ov);
            if ((v & 31) == 0) qwarp[v >> 5] = ss;
        }
        __syncthreads();
        if (kh == 0) {
            const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
            const float normed = tts_rbf16(norm_w[v] * tts_rbf16(ov * inv));
            const size_t zoff = (size_t(r) * nh_v + hv) * 128 + v;
            const float gate = tts_silu(__bfloat162float(z_in[zoff]));
            core_out[zoff] = __float2bfloat16_rn(normed * gate);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 64; ++i) S_head[(k_base + i) * 128 + v] = s_col[i];
}

// Batched single-token decode GDN recurrence + GatedRMSNorm across M slots in ONE launch.
// Grid: (M * nh_v), Block: 256.
__global__ void k_gdn_rec_norm_dec(const bf16* __restrict__ qkv_conv,
                                   const float* __restrict__ a_in,
                                   const float* __restrict__ b_in,
                                   const bf16* __restrict__ z_in,
                                   const float* __restrict__ A_log,
                                   const float* __restrict__ dt_bias,
                                   const float* __restrict__ norm_w,
                                   float* __restrict__ S_all,
                                   bf16* __restrict__ core_out,
                                   const int* __restrict__ d_slots,
                                   const int* __restrict__ d_pos,
                                   int M, int nh_k, int nh_v, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_v) return;
    const int m = idx / nh_v;
    const int hv = idx % nh_v;
    const int hk = hv / (nh_v / nh_k);
    const int slot = d_slots[m];
    const int pos = d_pos[m];
    const int qkv_dim = (2 * nh_k + nh_v) * 128;

    const int tid = threadIdx.x;
    const int v = tid & 127;
    const int kh = tid >> 7;
    const int k_base = kh * 64;

    __shared__ float sq[128], sk[128], sp_kv[128], sp_delta[128], sp_out[128], s_meta[2], qwarp[4], kwarp[4];
    const bf16* qkv = qkv_conv + size_t(m) * qkv_dim;
    const bf16* q_ptr = qkv + size_t(hk) * 128;
    const bf16* k_ptr = qkv + size_t(nh_k + hk) * 128;
    const bf16* v_ptr = qkv + size_t(2 * nh_k + hv) * 128;

    if (kh == 0) sq[v] = __bfloat162float(q_ptr[v]);
    else sk[v] = __bfloat162float(k_ptr[v]);
    if (tid == 0) {
        const float av = tts_rbf16(a_in[size_t(m) * nh_v + hv]);
        const float bv = tts_rbf16(b_in[size_t(m) * nh_v + hv]);
        s_meta[0] = expf(-expf(A_log[hv]) * tts_softplus(av + dt_bias[hv]));
        s_meta[1] = tts_sigmoid(bv);
    }
    __syncthreads();

    constexpr float inv_sqrt_dk = 0.08838834764831845f;
    float qv = 0.0f, kv = 0.0f;
    if (kh == 0) {
        qv = sq[v];
        const float q2 = tts_warp_sum(qv * qv);
        if ((v & 31) == 0) qwarp[v >> 5] = q2;
    } else {
        kv = sk[v];
        const float k2 = tts_warp_sum(kv * kv);
        if ((v & 31) == 0) kwarp[v >> 5] = k2;
    }
    __syncthreads();
    if (kh == 0) {
        sq[v] = qv * rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) + 1e-6f) * inv_sqrt_dk;
    } else {
        sk[v] = kv * rsqrtf((kwarp[0] + kwarp[1] + kwarp[2] + kwarp[3]) + 1e-6f);
    }
    __syncthreads();

    const float decay = s_meta[0];
    const float beta = s_meta[1];
    float* S_head = S_all + (size_t(slot) * nh_v + hv) * (128 * 128);
    float s_col[64];
    float kv_part = 0.0f;
    if (pos == 0) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float sv = S_head[(k_base + i) * 128 + v] * decay;
            s_col[i] = sv;
            kv_part = fmaf(sv, sk[k_base + i], kv_part);
        }
    }
    if (kh == 1) sp_kv[v] = kv_part;
    __syncthreads();
    if (kh == 0) {
        sp_delta[v] = (__bfloat162float(v_ptr[v]) - (kv_part + sp_kv[v])) * beta;
    }
    __syncthreads();
    const float delta_v = sp_delta[v];

    float out_part = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const float sv = fmaf(sk[k_base + i], delta_v, s_col[i]);
        S_head[(k_base + i) * 128 + v] = sv;
        out_part = fmaf(sv, sq[k_base + i], out_part);
    }
    if (kh == 1) sp_out[v] = out_part;
    __syncthreads();

    float ov = 0.0f;
    if (kh == 0) {
        ov = tts_rbf16(out_part + sp_out[v]);
        const float ss = tts_warp_sum(ov * ov);
        if ((v & 31) == 0) qwarp[v >> 5] = ss;
    }
    __syncthreads();
    if (kh == 0) {
        const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
        const float normed = tts_rbf16(norm_w[v] * tts_rbf16(ov * inv));
        const size_t zoff = size_t(idx) * 128 + v;
        core_out[zoff] = __float2bfloat16_rn(normed * tts_silu(__bfloat162float(z_in[zoff])));
    }
}

// ------------------------------------------------------------------------------------------------ MoE kernels

// Router: BF16-rounded gate logits -> FP32 softmax over 256 experts -> top-K -> optional renormalize -> BF16 topk_w.
// Grid: (M), Block: 256 (one thread per expert for NEXP=256).
__global__ void k_moe_router_topk(const float* __restrict__ logits,
                                  int32_t* __restrict__ topk_ids,
                                  float* __restrict__ topk_w,
                                  int M, int nexp, int topk, bool norm_topk) {
    __shared__ float s_p[256];
    __shared__ float red[8];
    __shared__ float s_row_max;
    const int m = blockIdx.x;
    const int e = threadIdx.x;
    if (m >= M) return;

    const float v = (e < nexp) ? tts_rbf16(logits[size_t(m) * nexp + e]) : -1e30f;
    float mx = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((e & 31) == 0) red[e >> 5] = mx;
    __syncthreads();
    if (e < 32) {
        float bm = (e < 8) ? red[e] : -1e30f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off));
        if (e == 0) s_row_max = bm;
    }
    __syncthreads();
    const float row_max = s_row_max;

    const float ex = (e < nexp) ? expf(v - row_max) : 0.0f;
    const float sum_ex = tts_block_sum256(ex, red);
    s_p[e] = (e < nexp && sum_ex > 0.0f) ? (ex / sum_ex) : 0.0f;
    __syncthreads();

    if (e == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            float best_p = -1.0f;
            int best_e = 0;
            for (int j = 0; j < nexp; ++j) {
                if (s_p[j] > best_p) {
                    best_p = s_p[j];
                    best_e = j;
                }
            }
            topk_ids[size_t(m) * topk + k] = best_e;
            topk_w[size_t(m) * topk + k] = best_p;
            wsum += best_p;
            s_p[best_e] = -2.0f;
        }
        const float inv = (norm_topk && wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
        for (int k = 0; k < topk; ++k) {
            topk_w[size_t(m) * topk + k] = tts_rbf16(topk_w[size_t(m) * topk + k] * inv);
        }
    }
}

__global__ void k_gather_rows_bf16(const bf16* __restrict__ src,
                                   const int32_t* __restrict__ row_ids,
                                   bf16* __restrict__ dst,
                                   int cnt, int H) {
    const int i = blockIdx.x;
    if (i >= cnt) return;
    const bf16* s = src + size_t(row_ids[i]) * H;
    bf16* d = dst + size_t(i) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) d[c] = s[c];
}

__global__ void k_scatter_add_f32(const float* __restrict__ src,
                                  const int32_t* __restrict__ row_ids,
                                  const float* __restrict__ weights,
                                  float* __restrict__ dst,
                                  int cnt, int H) {
    const int i = blockIdx.x;
    if (i >= cnt) return;
    const float w = weights[i];
    const float* s = src + size_t(i) * H;
    float* d = dst + size_t(row_ids[i]) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        d[c] += s[c] * w;
    }
}

// Fused MoE + gated shared expert + FP32 residual update (avoids 512-ULP BF16 boundary flips on outlier channels).
__global__ void k_moe_finalize_res_f32(const float* __restrict__ routed_f32,
                                       const float* __restrict__ shared_out,
                                       const float* __restrict__ shared_gate_logit,
                                       float* __restrict__ residual,
                                       int M, int H) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float g = tts_sigmoid(shared_gate_logit[m]);
    const float* rt = routed_f32 + size_t(m) * H;
    const float* sh = shared_out + size_t(m) * H;
    float* r = residual + size_t(m) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        r[c] += rt[c] + g * sh[c];
    }
}

// Dequantize 4 packed FP8 E4M3 bytes in `w4` scaled by `s` into 4 FP32 values.
__device__ __forceinline__ float4 tts_deq4_e4m3(uint32_t w4, float s) {
    const uint16_t lo = static_cast<uint16_t>(w4 & 0xffffu);
    const uint16_t hi = static_cast<uint16_t>(w4 >> 16);
    uint32_t h0 = 0, h1 = 0;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(lo));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(hi));
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
    return make_float4(f0.x * s, f0.y * s, f1.x * s, f1.y * s);
}

__device__ __forceinline__ float tts_dot4_bf16_f4(uint2 xv, float4 w) {
    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
    return fmaf(x01.x, w.x, fmaf(x01.y, w.y, fmaf(x23.x, w.z, x23.y * w.w)));
}

// On-GPU zero-host-sync Ref MoE kernels for store-0 (FP8_BLOCK) or BF16 experts.
// Single-token / ungrouped path (used for M == 1, 8 distinct experts) with 128-bit LDG.E.128 loads.
// Grid: (M * TOPK, I / 4), Block: 128 (4 warps, 1 warp per intermediate channel).
__global__ void __launch_bounds__(128) k_ref_moe_fc1_route(
    const bf16* __restrict__ x_norm,
    const int32_t* __restrict__ topk_ids,
    const uint8_t* __restrict__ gate_fp8, const float* __restrict__ gate_sc,
    const uint8_t* __restrict__ up_fp8,   const float* __restrict__ up_sc,
    const bf16* __restrict__ gate_bf16,   const bf16* __restrict__ up_bf16,
    bf16* __restrict__ route_act,
    int M, int H, int I, int topk, bool is_fp8) {
    const int route = blockIdx.x;
    if (route >= M * topk) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.y * 4 + warp;
    if (i >= I) return;

    const int m = route / topk;
    const int e = topk_ids[route];
    const bf16* xr = x_norm + size_t(m) * H;
    const size_t mat_elems = size_t(I) * H;
    float ag = 0.0f, au = 0.0f;

    if (is_fp8) {
        const int bo = (I + 127) / 128;
        const int bi = (H + 127) / 128;
        const size_t sc_base = (size_t(e) * bo + (i >> 7)) * bi;
        const uint8_t* gr = gate_fp8 + size_t(e) * mat_elems + size_t(i) * H;
        const uint8_t* ur = up_fp8 + size_t(e) * mat_elems + size_t(i) * H;
        // 16 elements/lane * 32 lanes = 512 elements/iter; each uint4 sits strictly inside one 128-elem scale block.
        for (int k = lane * 16; k < H; k += 512) {
            const float gs = gate_sc[sc_base + (k >> 7)];
            const float us = up_sc[sc_base + (k >> 7)];
            const uint4 gw = *reinterpret_cast<const uint4*>(gr + k);
            const uint4 uw = *reinterpret_cast<const uint4*>(ur + k);
            const uint4 xv0 = *reinterpret_cast<const uint4*>(xr + k);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(xr + k + 8);
            ag += tts_dot4_bf16_f4(make_uint2(xv0.x, xv0.y), tts_deq4_e4m3(gw.x, gs));
            ag += tts_dot4_bf16_f4(make_uint2(xv0.z, xv0.w), tts_deq4_e4m3(gw.y, gs));
            ag += tts_dot4_bf16_f4(make_uint2(xv1.x, xv1.y), tts_deq4_e4m3(gw.z, gs));
            ag += tts_dot4_bf16_f4(make_uint2(xv1.z, xv1.w), tts_deq4_e4m3(gw.w, gs));
            au += tts_dot4_bf16_f4(make_uint2(xv0.x, xv0.y), tts_deq4_e4m3(uw.x, us));
            au += tts_dot4_bf16_f4(make_uint2(xv0.z, xv0.w), tts_deq4_e4m3(uw.y, us));
            au += tts_dot4_bf16_f4(make_uint2(xv1.x, xv1.y), tts_deq4_e4m3(uw.z, us));
            au += tts_dot4_bf16_f4(make_uint2(xv1.z, xv1.w), tts_deq4_e4m3(uw.w, us));
        }
    } else {
        const bf16* gr = gate_bf16 + size_t(e) * mat_elems + size_t(i) * H;
        const bf16* ur = up_bf16 + size_t(e) * mat_elems + size_t(i) * H;
        for (int k = lane * 8; k < H; k += 256) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xr + k);
            const uint4 gv = *reinterpret_cast<const uint4*>(gr + k);
            const uint4 uv = *reinterpret_cast<const uint4*>(ur + k);
            const uint32_t xa[4] = {xv.x, xv.y, xv.z, xv.w};
            const uint32_t ga[4] = {gv.x, gv.y, gv.z, gv.w};
            const uint32_t ua[4] = {uv.x, uv.y, uv.z, uv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa[j]));
                const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ga[j]));
                const float2 uf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ua[j]));
                ag = fmaf(xf.x, gf.x, fmaf(xf.y, gf.y, ag));
                au = fmaf(xf.x, uf.x, fmaf(xf.y, uf.y, au));
            }
        }
    }

    ag = tts_warp_sum(ag);
    au = tts_warp_sum(au);
    if (lane == 0) {
        route_act[size_t(route) * I + i] = __float2bfloat16_rn(tts_rbf16(tts_silu(tts_rbf16(ag))) * tts_rbf16(au));
    }
}

// Grid: (M * TOPK, H / 4), Block: 128 (4 warps, 1 warp per hidden channel).
__global__ void __launch_bounds__(128) k_ref_moe_fc2_route(
    const bf16* __restrict__ route_act,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w,
    const uint8_t* __restrict__ down_fp8, const float* __restrict__ down_sc,
    const bf16* __restrict__ down_bf16,
    float* __restrict__ route_out,
    int M, int H, int I, int topk, bool is_fp8) {
    const int route = blockIdx.x;
    if (route >= M * topk) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = blockIdx.y * 4 + warp;
    if (h >= H) return;

    const int e = topk_ids[route];
    const float rw = topk_w[route];
    const bf16* ar = route_act + size_t(route) * I;
    const size_t mat_elems = size_t(H) * I;
    float ad = 0.0f;

    if (is_fp8) {
        const int bo = (H + 127) / 128;
        const int bi = (I + 127) / 128;
        const size_t sc_base = (size_t(e) * bo + (h >> 7)) * bi;
        const uint8_t* dr = down_fp8 + size_t(e) * mat_elems + size_t(h) * I;
        for (int k = lane * 16; k < I; k += 512) {
            const float ds = down_sc[sc_base + (k >> 7)];
            const uint4 dw = *reinterpret_cast<const uint4*>(dr + k);
            const uint4 av0 = *reinterpret_cast<const uint4*>(ar + k);
            const uint4 av1 = *reinterpret_cast<const uint4*>(ar + k + 8);
            ad += tts_dot4_bf16_f4(make_uint2(av0.x, av0.y), tts_deq4_e4m3(dw.x, ds));
            ad += tts_dot4_bf16_f4(make_uint2(av0.z, av0.w), tts_deq4_e4m3(dw.y, ds));
            ad += tts_dot4_bf16_f4(make_uint2(av1.x, av1.y), tts_deq4_e4m3(dw.z, ds));
            ad += tts_dot4_bf16_f4(make_uint2(av1.z, av1.w), tts_deq4_e4m3(dw.w, ds));
        }
    } else {
        const bf16* dr = down_bf16 + size_t(e) * mat_elems + size_t(h) * I;
        for (int k = lane * 8; k < I; k += 256) {
            const uint4 av = *reinterpret_cast<const uint4*>(ar + k);
            const uint4 dv = *reinterpret_cast<const uint4*>(dr + k);
            const uint32_t aa[4] = {av.x, av.y, av.z, av.w};
            const uint32_t da[4] = {dv.x, dv.y, dv.z, dv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&aa[j]));
                const float2 df = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&da[j]));
                ad = fmaf(af.x, df.x, fmaf(af.y, df.y, ad));
            }
        }
    }

    ad = tts_warp_sum(ad);
    if (lane == 0) {
        route_out[size_t(route) * H + h] = rw * ad;
    }
}

// Grouped zero-host-sync Ref MoE FC1 for M >= 2 (prefill M=2..128 AND batched decode M=2,4,8).
// Reads each active expert's gate+up weights ONCE from DRAM per group of up to 8 tokens!
// Grid: (max_groups, I / 4), Block: 128 (4 warps, 1 warp per intermediate channel).
__global__ void __launch_bounds__(128) k_ref_moe_fc1_grouped(
    const bf16* __restrict__ x_norm,
    const int32_t* __restrict__ d_n_groups,
    const int32_t* __restrict__ grp_exp,
    const int32_t* __restrict__ grp_cnt,
    const int32_t* __restrict__ grp_routes,
    const uint8_t* __restrict__ gate_fp8, const float* __restrict__ gate_sc,
    const uint8_t* __restrict__ up_fp8,   const float* __restrict__ up_sc,
    const bf16* __restrict__ gate_bf16,   const bf16* __restrict__ up_bf16,
    bf16* __restrict__ route_act,
    int H, int I, int topk, bool is_fp8) {
    const int g = blockIdx.x;
    if (g >= d_n_groups[0]) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.y * 4 + warp;
    if (i >= I) return;

    const int e = grp_exp[g];
    const int cnt = grp_cnt[g]; // 1..8
    int r_ids[8];
    int m_ids[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < cnt) {
            r_ids[b] = grp_routes[g * 8 + b];
            m_ids[b] = r_ids[b] / topk;
        } else {
            r_ids[b] = 0;
            m_ids[b] = 0;
        }
    }

    float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const size_t mat_elems = size_t(I) * H;

    if (is_fp8) {
        const int bo = (I + 127) / 128;
        const int bi = (H + 127) / 128;
        const size_t sc_base = (size_t(e) * bo + (i >> 7)) * bi;
        const uint8_t* gr = gate_fp8 + size_t(e) * mat_elems + size_t(i) * H;
        const uint8_t* ur = up_fp8 + size_t(e) * mat_elems + size_t(i) * H;
        for (int k = lane * 16; k < H; k += 512) {
            const float gs = gate_sc[sc_base + (k >> 7)];
            const float us = up_sc[sc_base + (k >> 7)];
            const uint4 gw = *reinterpret_cast<const uint4*>(gr + k);
            const uint4 uw = *reinterpret_cast<const uint4*>(ur + k);
            const uint32_t gwa[4] = {gw.x, gw.y, gw.z, gw.w};
            const uint32_t uwa[4] = {uw.x, uw.y, uw.z, uw.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 gf = tts_deq4_e4m3(gwa[j], gs);
                const float4 uf = tts_deq4_e4m3(uwa[j], us);
                const int kj = k + j * 4;
                for (int b = 0; b < cnt; ++b) {
                    const uint2 xv = *reinterpret_cast<const uint2*>(x_norm + size_t(m_ids[b]) * H + kj);
                    ag[b] += tts_dot4_bf16_f4(xv, gf);
                    au[b] += tts_dot4_bf16_f4(xv, uf);
                }
            }
        }
    } else {
        const bf16* gr = gate_bf16 + size_t(e) * mat_elems + size_t(i) * H;
        const bf16* ur = up_bf16 + size_t(e) * mat_elems + size_t(i) * H;
        for (int k = lane * 8; k < H; k += 256) {
            const uint4 gv = *reinterpret_cast<const uint4*>(gr + k);
            const uint4 uv = *reinterpret_cast<const uint4*>(ur + k);
            const uint32_t ga[4] = {gv.x, gv.y, gv.z, gv.w};
            const uint32_t ua[4] = {uv.x, uv.y, uv.z, uv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ga[j]));
                const float2 uf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ua[j]));
                const int kj = k + j * 2;
                for (int b = 0; b < cnt; ++b) {
                    const uint32_t xv = *reinterpret_cast<const uint32_t*>(x_norm + size_t(m_ids[b]) * H + kj);
                    const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv));
                    ag[b] = fmaf(xf.x, gf.x, fmaf(xf.y, gf.y, ag[b]));
                    au[b] = fmaf(xf.x, uf.x, fmaf(xf.y, uf.y, au[b]));
                }
            }
        }
    }

    for (int b = 0; b < cnt; ++b) {
        const float g_sum = tts_warp_sum(ag[b]);
        const float u_sum = tts_warp_sum(au[b]);
        if (lane == 0) {
            route_act[size_t(r_ids[b]) * I + i] =
                __float2bfloat16_rn(tts_rbf16(tts_silu(tts_rbf16(g_sum))) * tts_rbf16(u_sum));
        }
    }
}

// Grouped zero-host-sync Ref MoE FC2 for M >= 2.
// Reads each active expert's down weight ONCE from DRAM per group of up to 8 tokens!
// Grid: (max_groups, H / 4), Block: 128 (4 warps, 1 warp per hidden channel).
__global__ void __launch_bounds__(128) k_ref_moe_fc2_grouped(
    const bf16* __restrict__ route_act,
    const int32_t* __restrict__ d_n_groups,
    const int32_t* __restrict__ grp_exp,
    const int32_t* __restrict__ grp_cnt,
    const int32_t* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const uint8_t* __restrict__ down_fp8, const float* __restrict__ down_sc,
    const bf16* __restrict__ down_bf16,
    float* __restrict__ route_out,
    int H, int I, bool is_fp8) {
    const int g = blockIdx.x;
    if (g >= d_n_groups[0]) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = blockIdx.y * 4 + warp;
    if (h >= H) return;

    const int e = grp_exp[g];
    const int cnt = grp_cnt[g]; // 1..8
    int r_ids[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        r_ids[b] = (b < cnt) ? grp_routes[g * 8 + b] : 0;
    }

    float ad[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const size_t mat_elems = size_t(H) * I;

    if (is_fp8) {
        const int bo = (H + 127) / 128;
        const int bi = (I + 127) / 128;
        const size_t sc_base = (size_t(e) * bo + (h >> 7)) * bi;
        const uint8_t* dr = down_fp8 + size_t(e) * mat_elems + size_t(h) * I;
        for (int k = lane * 16; k < I; k += 512) {
            const float ds = down_sc[sc_base + (k >> 7)];
            const uint4 dw = *reinterpret_cast<const uint4*>(dr + k);
            const uint32_t dwa[4] = {dw.x, dw.y, dw.z, dw.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 df = tts_deq4_e4m3(dwa[j], ds);
                const int kj = k + j * 4;
                for (int b = 0; b < cnt; ++b) {
                    const uint2 av = *reinterpret_cast<const uint2*>(route_act + size_t(r_ids[b]) * I + kj);
                    ad[b] += tts_dot4_bf16_f4(av, df);
                }
            }
        }
    } else {
        const bf16* dr = down_bf16 + size_t(e) * mat_elems + size_t(h) * I;
        for (int k = lane * 8; k < I; k += 256) {
            const uint4 dv = *reinterpret_cast<const uint4*>(dr + k);
            const uint32_t da[4] = {dv.x, dv.y, dv.z, dv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 df = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&da[j]));
                const int kj = k + j * 2;
                for (int b = 0; b < cnt; ++b) {
                    const uint32_t av = *reinterpret_cast<const uint32_t*>(route_act + size_t(r_ids[b]) * I + kj);
                    const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&av));
                    ad[b] = fmaf(af.x, df.x, fmaf(af.y, df.y, ad[b]));
                }
            }
        }
    }

    for (int b = 0; b < cnt; ++b) {
        const float d_sum = tts_warp_sum(ad[b]);
        if (lane == 0) {
            const int r = r_ids[b];
            route_out[size_t(r) * H + h] = topk_w[r] * d_sum;
        }
    }
}

// ------------------------------------------------------------------------------------------------ Backbone runner

struct BackboneProfile {
    bool enabled = false;
    double embed_ms = 0.0;
    double attn_ms = 0.0;
    double gdn_ms = 0.0;
    double moe_ms = 0.0;
    double head_ms = 0.0;
    void reset() { embed_ms = attn_ms = gdn_ms = moe_ms = head_ms = 0.0; }
};

struct BackboneLayerTrace {
    bool enabled = false;
    bool replay_router = false;
    std::vector<float> embed_f32;
    std::vector<float> final_hidden_f32;
    // Fine-grained sub-op traces for L0 (GDN) and L3 (FullAttention) Gate E1 diagnostics.
    std::vector<float> l0_qkv_raw;
    std::vector<float> l0_qkv_conv;
    std::vector<float> l0_core_out;
    std::vector<float> l3_q_ready;
    std::vector<float> l3_attn_out;
    std::vector<std::vector<float>> mixer_residuals;
    std::vector<std::vector<float>> router_logits;
    std::vector<std::vector<float>> residuals;
    std::vector<std::vector<int32_t>> topk_ids;
    std::vector<std::vector<float>> topk_w;
    std::vector<std::vector<int32_t>> forced_topk_ids;
    std::vector<std::vector<float>> forced_topk_w;
    void reset(int L) {
        embed_f32.clear();
        final_hidden_f32.clear();
        l0_qkv_raw.clear();
        l0_qkv_conv.clear();
        l0_core_out.clear();
        l3_q_ready.clear();
        l3_attn_out.clear();
        mixer_residuals.assign(size_t(L), {});
        router_logits.assign(size_t(L), {});
        residuals.assign(size_t(L), {});
        topk_ids.assign(size_t(L), {});
        topk_w.assign(size_t(L), {});
    }
};

struct BackboneRunner {
    static constexpr size_t CUBLAS_WS_BYTES = size_t(32) << 20;

    const TtsWeights* W = nullptr;
    cublasHandle_t blas = nullptr;
    void* d_cublas_ws = nullptr;
    int max_slots = 8;
    int cap = 4096;
    int max_chunk = 128;

    // Persistent per-slot caches
    std::vector<bf16*> k_caches;    // [L], non-null for full-attn layers: [max_slots, cap, 2, 256] BF16
    std::vector<bf16*> v_caches;    // [L]
    std::vector<float*> gdn_S;      // [L], non-null for GDN layers: [max_slots, 32, 128, 128]
    std::vector<float*> gdn_conv;   // [L], non-null for GDN layers: [max_slots, 3, 8192]

    // Scratch buffers (allocated for `max_chunk` rows)
    int32_t* d_rows = nullptr;          // [max_chunk, 11]
    int32_t* d_slots = nullptr;         // [max_chunk]
    int32_t* d_pos = nullptr;           // [max_chunk]
    float* d_residual = nullptr;        // [max_chunk, H] (holds exact BF16-rounded residual values)
    bf16* d_res_delta = nullptr;        // [max_chunk, H] scratch for BF16 residual delta
    float* d_moe_routed_f32 = nullptr;  // [max_chunk, H] FP32 routed MoE accumulator before BF16 finalize
    float* d_ref_x_f32 = nullptr;       // [max_chunk, 4096] FP32 scratch (WeightMode::RefF32 only)
    float* d_ref_y_f32 = nullptr;       // [max_chunk, 8192] FP32 scratch (WeightMode::RefF32 only)
    bf16* d_norm = nullptr;             // [max_chunk, H]
    bf16* d_qkv = nullptr;              // [max_chunk, 8192] (reused by FullAttn qg & GDN qkv_raw)
    bf16* d_qkv2 = nullptr;             // [max_chunk, 8192] (GDN qkv_conv)
    bf16* d_q_ready = nullptr;          // [max_chunk, 4096] (FullAttn Q & GDN z / core)
    bf16* d_kv_tmp = nullptr;           // [max_chunk, 1024] (K[512] + V[512])
    bf16* d_attn_out = nullptr;         // [max_chunk, 4096]
    float* d_ab = nullptr;              // [max_chunk, 64] (GDN a[32] + b[32])
    float* d_router_logits = nullptr;   // [max_chunk, 256]
    int32_t* d_topk_ids = nullptr;      // [max_chunk, 8]
    float* d_topk_w = nullptr;          // [max_chunk, 8]
    bf16* d_shared_g = nullptr;         // [max_chunk, 512]
    bf16* d_shared_u = nullptr;         // [max_chunk, 512]
    float* d_shared_out = nullptr;      // [max_chunk, H]
    float* d_shared_gate = nullptr;     // [max_chunk]
    bf16* d_exp_x = nullptr;            // [max_chunk, H]
    bf16* d_exp_g = nullptr;            // [max_chunk, 512]
    bf16* d_exp_u = nullptr;            // [max_chunk, 512]
    float* d_exp_out = nullptr;         // [max_chunk, H]
    int32_t* d_exp_rows = nullptr;      // [max_chunk * top_k]
    float* d_exp_w = nullptr;           // [max_chunk * top_k]
    bf16* d_deq_g = nullptr;            // [512, 2048] scratch for on-the-fly FP8 expert dequant
    bf16* d_deq_u = nullptr;            // [512, 2048]
    bf16* d_deq_d = nullptr;            // [2048, 512]
    bf16* d_route_act = nullptr;        // [max_chunk * top_k, I] (on-GPU Ref MoE for prefill & decode)
    float* d_route_out = nullptr;       // [max_chunk * top_k, H]
    int32_t* d_ref_n_groups = nullptr;  // [1]
    int32_t* d_ref_grp_exp = nullptr;   // [max_chunk * top_k]
    int32_t* d_ref_grp_cnt = nullptr;   // [max_chunk * top_k]
    int32_t* d_ref_grp_routes = nullptr;// [max_chunk * top_k * 8]
    int max_graph_M = 16;
    bf16* d_final_bf16 = nullptr;       // [max_chunk, H]
    float* d_final_f32 = nullptr;       // [max_chunk, H]
    float* d_main_logits = nullptr;     // [max_chunk, 4097]

    // Pinned host scratch for tiny MoE routing table & row/slot/pos staging
    int32_t* h_topk_ids = nullptr;
    float* h_topk_w = nullptr;
    int32_t* h_exp_rows = nullptr;      // [max_chunk * top_k]
    float* h_exp_w = nullptr;           // [max_chunk * top_k]
    int32_t* h_rows_stage = nullptr;    // [max_chunk, row_width]
    int32_t* h_slots = nullptr;
    int32_t* h_pos = nullptr;

    size_t state_bytes = 0;
    size_t scratch_bytes = 0;
    DenseWorkspace dense_ws;
    TtsMoeScratch moe_sc;
    BackboneProfile prof;
    BackboneLayerTrace trace;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev_upload = nullptr;
    bool upload_pending = false;

    ~BackboneRunner() { release(); }

    void release() {
        if (upload_pending && ev_upload) {
            cudaEventSynchronize(ev_upload);
            upload_pending = false;
        }
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto freeh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        dense_ws.release();
        moe_sc.release();
        for (auto*& p : k_caches) freep(p);
        for (auto*& p : v_caches) freep(p);
        for (auto*& p : gdn_S) freep(p);
        for (auto*& p : gdn_conv) freep(p);
        k_caches.clear(); v_caches.clear(); gdn_S.clear(); gdn_conv.clear();
        freep(d_rows); freep(d_slots); freep(d_pos); freep(d_residual); freep(d_res_delta);
        freep(d_moe_routed_f32); freep(d_ref_x_f32); freep(d_ref_y_f32); freep(d_norm);
        freep(d_qkv); freep(d_qkv2); freep(d_q_ready); freep(d_kv_tmp); freep(d_attn_out);
        freep(d_ab); freep(d_router_logits); freep(d_topk_ids); freep(d_topk_w);
        freep(d_shared_g); freep(d_shared_u); freep(d_shared_out); freep(d_shared_gate);
        freep(d_exp_x); freep(d_exp_g); freep(d_exp_u); freep(d_exp_out);
        freep(d_exp_rows); freep(d_exp_w);
        freep(d_deq_g); freep(d_deq_u); freep(d_deq_d);
        freep(d_route_act); freep(d_route_out);
        freep(d_ref_n_groups); freep(d_ref_grp_exp); freep(d_ref_grp_cnt); freep(d_ref_grp_routes);
        freep(d_final_bf16); freep(d_final_f32); freep(d_main_logits);
        freeh(h_topk_ids); freeh(h_topk_w); freeh(h_exp_rows); freeh(h_exp_w);
        freeh(h_rows_stage); freeh(h_slots); freeh(h_pos);
        if (ev0) { cudaEventDestroy(ev0); ev0 = nullptr; }
        if (ev1) { cudaEventDestroy(ev1); ev1 = nullptr; }
        if (ev_upload) { cudaEventDestroy(ev_upload); ev_upload = nullptr; }
        if (blas) { cublasDestroy(blas); blas = nullptr; }
        freep(d_cublas_ws);
        scratch_bytes = 0;
    }

    void init(const TtsWeights& weights, int streams, int kv_cap, int chunk = 128) {
        release();
        W = &weights;
        max_slots = std::max(1, streams);
        cap = std::max(16, kv_cap);
        max_chunk = std::max(16, chunk);

        TTS_BLAS_CK(cublasCreate(&blas));
        TTS_CK(cudaMalloc(&d_cublas_ws, CUBLAS_WS_BYTES));
        TTS_BLAS_CK(cublasSetWorkspace(blas, d_cublas_ws, CUBLAS_WS_BYTES));
        scratch_bytes = CUBLAS_WS_BYTES;
        TTS_CK(cudaEventCreate(&ev0));
        TTS_CK(cudaEventCreate(&ev1));
        TTS_CK(cudaEventCreate(&ev_upload));
        upload_pending = false;

        const auto& c = W->cfg;
        const int L = c.num_layers;
        const int H = c.hidden_size;
        const int I = c.moe_intermediate;
        const int qkv_dim = (2 * c.gdn_kh + c.gdn_vh) * c.gdn_kd;
        const int val_dim = c.gdn_vh * c.gdn_vd;
        const int n_cand = c.main_candidates();

        k_caches.assign(size_t(L), nullptr);
        v_caches.assign(size_t(L), nullptr);
        gdn_S.assign(size_t(L), nullptr);
        gdn_conv.assign(size_t(L), nullptr);

        state_bytes = 0;
        for (int l = 0; l < L; ++l) {
            if (c.is_full_attn[size_t(l)]) {
                const size_t kv_b = size_t(max_slots) * cap * c.num_kv_heads * c.head_dim * sizeof(bf16);
                TTS_CK(cudaMalloc(&k_caches[size_t(l)], kv_b));
                TTS_CK(cudaMalloc(&v_caches[size_t(l)], kv_b));
                TTS_CK(cudaMemset(k_caches[size_t(l)], 0, kv_b));
                TTS_CK(cudaMemset(v_caches[size_t(l)], 0, kv_b));
                state_bytes += 2 * kv_b;
            } else {
                const size_t s_b = size_t(max_slots) * c.gdn_vh * c.gdn_kd * c.gdn_vd * sizeof(float);
                const size_t cv_b = size_t(max_slots) * (c.gdn_conv - 1) * qkv_dim * sizeof(float);
                TTS_CK(cudaMalloc(&gdn_S[size_t(l)], s_b));
                TTS_CK(cudaMalloc(&gdn_conv[size_t(l)], cv_b));
                TTS_CK(cudaMemset(gdn_S[size_t(l)], 0, s_b));
                TTS_CK(cudaMemset(gdn_conv[size_t(l)], 0, cv_b));
                state_bytes += s_b + cv_b;
            }
        }

        auto alloc_d = [&](auto*& p, size_t n) {
            const size_t nb = n * sizeof(*p);
            TTS_CK(cudaMalloc(&p, nb));
            scratch_bytes += nb;
        };
        const int qg_dim = std::max(qkv_dim, c.num_heads * 2 * c.head_dim);
        const int a_dim = std::max(val_dim, c.num_heads * c.head_dim);
        alloc_d(d_rows, size_t(max_chunk) * c.row_width());
        alloc_d(d_slots, size_t(max_chunk));
        alloc_d(d_pos, size_t(max_chunk));
        alloc_d(d_residual, size_t(max_chunk) * H);
        alloc_d(d_res_delta, size_t(max_chunk) * H);
        alloc_d(d_moe_routed_f32, size_t(max_chunk) * H);
        if (W->mode == WeightMode::RefF32) {
            alloc_d(d_ref_x_f32, size_t(max_chunk) * 4096);
            alloc_d(d_ref_y_f32, size_t(max_chunk) * 8192);
        }
        alloc_d(d_norm, size_t(max_chunk) * H);
        alloc_d(d_qkv, size_t(max_chunk) * qg_dim);
        alloc_d(d_qkv2, size_t(max_chunk) * qkv_dim);
        alloc_d(d_q_ready, size_t(max_chunk) * a_dim);
        alloc_d(d_kv_tmp, size_t(max_chunk) * 2 * c.num_kv_heads * c.head_dim);
        alloc_d(d_attn_out, size_t(max_chunk) * a_dim);
        alloc_d(d_ab, size_t(max_chunk) * 2 * c.gdn_vh);
        alloc_d(d_router_logits, size_t(max_chunk) * c.num_experts);
        alloc_d(d_topk_ids, size_t(max_chunk) * c.top_k);
        alloc_d(d_topk_w, size_t(max_chunk) * c.top_k);
        alloc_d(d_shared_g, size_t(max_chunk) * c.shared_intermediate);
        alloc_d(d_shared_u, size_t(max_chunk) * c.shared_intermediate);
        alloc_d(d_shared_out, size_t(max_chunk) * H);
        alloc_d(d_shared_gate, size_t(max_chunk));
        alloc_d(d_exp_x, size_t(max_chunk) * H);
        alloc_d(d_exp_g, size_t(max_chunk) * I);
        alloc_d(d_exp_u, size_t(max_chunk) * I);
        alloc_d(d_exp_out, size_t(max_chunk) * H);
        alloc_d(d_exp_rows, size_t(max_chunk) * c.top_k);
        alloc_d(d_exp_w, size_t(max_chunk) * c.top_k);
        alloc_d(d_deq_g, size_t(I) * H);
        alloc_d(d_deq_u, size_t(I) * H);
        alloc_d(d_deq_d, size_t(H) * I);
        max_graph_M = std::min(max_chunk, std::max(16, max_slots));
        const size_t max_ref_routes = size_t(max_chunk) * c.top_k;
        alloc_d(d_route_act, max_ref_routes * I);
        alloc_d(d_route_out, max_ref_routes * H);
        alloc_d(d_ref_n_groups, 1);
        alloc_d(d_ref_grp_exp, max_ref_routes);
        alloc_d(d_ref_grp_cnt, max_ref_routes);
        alloc_d(d_ref_grp_routes, max_ref_routes * 8);
        alloc_d(d_final_bf16, size_t(max_chunk) * H);
        alloc_d(d_final_f32, size_t(max_chunk) * H);
        alloc_d(d_main_logits, size_t(max_chunk) * n_cand);

        TTS_CK(cudaHostAlloc(&h_topk_ids, size_t(max_chunk) * c.top_k * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_topk_w, size_t(max_chunk) * c.top_k * sizeof(float), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_exp_rows, size_t(max_chunk) * c.top_k * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_exp_w, size_t(max_chunk) * c.top_k * sizeof(float), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_rows_stage, size_t(max_chunk) * c.row_width() * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_slots, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_pos, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));

        if (W->mode == WeightMode::Native) {
            dense_ws.init(max_chunk);
            scratch_bytes += dense_ws.scratch_bytes;
        }
        if (W->n_tr3_expert_layers > 0) {
            moe_sc.init(max_chunk);
            scratch_bytes += moe_sc.scratch_bytes;
        }
    }

    // Unified projection helpers: route through native Trellis/FP8_ROW/INT4/BF16 kernels when loaded natively,
    // or through cuBLAS BF16/FP32 in Ref/RefF32 mode.
    void proj_bf16(const bf16* X, const RefMat& ref_w, const g6emb::DevMatrix& nat_w,
                   bf16* Y, int M, int N, int K, cudaStream_t stream) {
        if (W->mode == WeightMode::Native && nat_w.valid()) {
            tts_native_linear_bf16(X, nat_w, Y, M, dense_ws, blas, stream);
        } else {
            gemm_ref(blas, X, ref_w, Y, M, N, K, false, 0.0f, d_ref_x_f32, d_ref_y_f32, stream);
        }
    }

    void proj_f32(const bf16* X, const RefMat& ref_w, const g6emb::DevMatrix& nat_w,
                  float* Y, int M, int N, int K, bool add_res, cudaStream_t stream) {
        if (W->mode == WeightMode::Native && nat_w.valid()) {
            tts_native_linear_f32(X, nat_w, Y, M, N, add_res, dense_ws, blas, stream);
        } else {
            gemm_ref(blas, X, ref_w, Y, M, N, K, true, add_res ? 1.0f : 0.0f, d_ref_x_f32, d_ref_y_f32, stream);
        }
    }

    // BF16 residual projection: Y_res = add_res ? bf16(Y_res + bf16(X @ W^T)) : bf16(X @ W^T).
    void proj_res_bf16(const bf16* X, const RefMat& ref_w, const g6emb::DevMatrix& nat_w,
                       float* Y_res, int M, int N, int K, bool add_res, cudaStream_t stream) {
        proj_bf16(X, ref_w, nat_w, d_res_delta, M, N, K, stream);
        const size_t y_n = size_t(M) * size_t(N);
        k_tts_res_update_bf16<<<static_cast<unsigned>((y_n + 255) / 256), 256, 0, stream>>>(
            d_res_delta, Y_res, y_n, add_res ? 1 : 0);
    }

    void compute_main_logits(const bf16* d_hid, int M, cudaStream_t stream) {
        TTS_BLAS_CK(cublasSetStream(blas, stream));
        const auto& c = W->cfg;
        proj_f32(d_hid, W->lm_head_cand, W->lm_head_nat, d_main_logits,
                 M, c.main_candidates(), c.hidden_size, false, stream);
    }

    // Zero-host-sync on-GPU Ref MoE (FP8_BLOCK or BF16) shared by both prefill (`forward`) and decode (`forward_device`).
    // Uses 128-bit LDG.E.128 per-route kernels for M == 1 (8 distinct experts) and on-GPU grouped
    // weight-sharing kernels for M >= 2 (up to 8 tokens/group, reading each active expert ONCE from DRAM).
    void run_ref_moe(const BackboneLayerW& LW, int M, cudaStream_t stream) {
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int I = c.moe_intermediate;
        const int n_routes = M * c.top_k;
        if (M == 1 || c.num_experts != MOE_NEXP || c.top_k != MOE_TOPK) {
            k_ref_moe_fc1_route<<<dim3(n_routes, (I + 3) / 4), 128, 0, stream>>>(
                d_norm, d_topk_ids,
                LW.exp_gate_fp8, LW.exp_gate_sc, LW.exp_up_fp8, LW.exp_up_sc,
                LW.exp_gate_bf16, LW.exp_up_bf16, d_route_act,
                M, H, I, c.top_k, LW.experts_fp8);
            k_ref_moe_fc2_route<<<dim3(n_routes, (H + 3) / 4), 128, 0, stream>>>(
                d_route_act, d_topk_ids, d_topk_w,
                LW.exp_down_fp8, LW.exp_down_sc, LW.exp_down_bf16, d_route_out,
                M, H, I, c.top_k, LW.experts_fp8);
        } else {
            const int max_groups = std::min(n_routes, MOE_NEXP + n_routes / 8);
            k_tts_moe_build_groups<<<1, MOE_NEXP, 0, stream>>>(
                d_topk_ids, d_ref_n_groups, d_ref_grp_exp, d_ref_grp_cnt, d_ref_grp_routes, n_routes);
            k_ref_moe_fc1_grouped<<<dim3(max_groups, (I + 3) / 4), 128, 0, stream>>>(
                d_norm, d_ref_n_groups, d_ref_grp_exp, d_ref_grp_cnt, d_ref_grp_routes,
                LW.exp_gate_fp8, LW.exp_gate_sc, LW.exp_up_fp8, LW.exp_up_sc,
                LW.exp_gate_bf16, LW.exp_up_bf16, d_route_act,
                H, I, c.top_k, LW.experts_fp8);
            k_ref_moe_fc2_grouped<<<dim3(max_groups, (H + 3) / 4), 128, 0, stream>>>(
                d_route_act, d_ref_n_groups, d_ref_grp_exp, d_ref_grp_cnt, d_ref_grp_routes,
                d_topk_w, LW.exp_down_fp8, LW.exp_down_sc, LW.exp_down_bf16, d_route_out,
                H, I, LW.experts_fp8);
        }
        k_tts_moe_combine_f32<<<M * 8, 64, 0, stream>>>(d_route_out, d_moe_routed_f32, M);
    }

    // 100% on-GPU, zero-host-sync, CUDA-graph-capturable batched decode forward.
    // All inputs (`dev_rows [M, 11]`, `dev_slots [M]`, `dev_pos [M]`) live on device.
    // Produces `d_final_bf16 [M, H]` (and `d_main_logits [M, 4097]` if `compute_logits`).
    void forward_device(const int32_t* dev_rows, const int32_t* dev_slots, const int32_t* dev_pos,
                        int M, bool compute_logits, bool want_f32_hidden, cudaStream_t stream) {
        if (M <= 0 || M > max_graph_M) {
            throw std::runtime_error("BackboneRunner::forward_device: M out of bounds");
        }
        TTS_BLAS_CK(cublasSetStream(blas, stream));
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int qkv_dim = (2 * c.gdn_kh + c.gdn_vh) * c.gdn_kd;
        const int val_dim = c.gdn_vh * c.gdn_vd;
        const int kv_dim = c.num_kv_heads * c.head_dim;

        auto tic = [&]() { if (prof.enabled) cudaEventRecord(ev0, stream); };
        auto toc = [&](double& dst) {
            if (prof.enabled) {
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                dst += double(ms);
            }
        };

        tic();
        if (W->embed_tokens_nat.valid() && W->embed_tokens_nat.fmt == g6emb::FMT_FP8_ROW) {
            k_backbone_embed_fp8<<<M, 256, 0, stream>>>(
                dev_rows, W->embed_tokens_nat.data, W->embed_tokens_nat.scale, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        } else if (W->embed_tokens.f32 != nullptr) {
            k_backbone_embed_f32<<<M, 256, 0, stream>>>(
                dev_rows, W->embed_tokens.f32, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        } else {
            k_backbone_embed<<<M, 256, 0, stream>>>(
                dev_rows, W->embed_tokens.bf16, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        }
        toc(prof.embed_ms);

        for (int l = 0; l < c.num_layers; ++l) {
            const BackboneLayerW& LW = W->layers[size_t(l)];
            k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, LW.input_ln, d_norm, M, H, c.rms_norm_eps);

            if (LW.is_full_attn) {
                tic();
                if (W->mode == WeightMode::Native && LW.q_proj_nat.valid() && LW.k_proj_nat.valid() && LW.v_proj_nat.valid()) {
                    tts_native_linear_triple_bf16(d_norm, LW.q_proj_nat, LW.k_proj_nat, LW.v_proj_nat,
                                                  d_qkv, d_kv_tmp, d_kv_tmp + size_t(M) * kv_dim, M, dense_ws, blas, stream);
                } else {
                    proj_bf16(d_norm, LW.q_proj, LW.q_proj_nat, d_qkv, M, c.num_heads * 2 * c.head_dim, H, stream);
                    proj_bf16(d_norm, LW.k_proj, LW.k_proj_nat, d_kv_tmp, M, kv_dim, H, stream);
                    proj_bf16(d_norm, LW.v_proj, LW.v_proj_nat, d_kv_tmp + size_t(M) * kv_dim, M, kv_dim, H, stream);
                }

                dim3 qk_grid(M, c.num_heads + c.num_kv_heads);
                k_bb_qk_norm_rope_kv<<<qk_grid, 256, 0, stream>>>(
                    d_qkv, d_kv_tmp, d_kv_tmp + size_t(M) * kv_dim,
                    LW.q_norm, LW.k_norm, d_q_ready, k_caches[size_t(l)], v_caches[size_t(l)],
                    dev_slots, dev_pos, M, c.num_heads, c.num_kv_heads, c.head_dim, c.rotary_dim,
                    c.rms_norm_eps, c.rope_theta, cap);

                dim3 attn_grid(M, c.num_heads);
                k_gqa_attn_gated<<<attn_grid, 256, 0, stream>>>(
                    d_q_ready, d_qkv, k_caches[size_t(l)], v_caches[size_t(l)], d_attn_out,
                    dev_slots, dev_pos, M, c.num_heads, c.num_kv_heads, c.head_dim, cap,
                    true, 0, 0);

                proj_f32(d_attn_out, LW.o_proj, LW.o_proj_nat, d_residual, M, H, c.num_heads * c.head_dim, true, stream);
                toc(prof.attn_ms);
            } else {
                tic();
                if (W->mode == WeightMode::Native && LW.in_proj_qkv_nat.valid() && LW.in_proj_z_nat.valid()) {
                    tts_native_linear_dual_bf16(d_norm, LW.in_proj_qkv_nat, LW.in_proj_z_nat,
                                                d_qkv, d_q_ready, M, dense_ws, blas, stream);
                } else {
                    proj_bf16(d_norm, LW.in_proj_qkv, LW.in_proj_qkv_nat, d_qkv, M, qkv_dim, H, stream);
                    proj_bf16(d_norm, LW.in_proj_z, LW.in_proj_z_nat, d_q_ready, M, val_dim, H, stream);
                }
                proj_f32(d_norm, LW.in_proj_a, LW.in_proj_a_nat, d_ab, M, c.gdn_vh, H, false, stream);
                proj_f32(d_norm, LW.in_proj_b, LW.in_proj_b_nat, d_ab + size_t(M) * c.gdn_vh, M, c.gdn_vh, H, false, stream);

                const size_t tot = size_t(M) * qkv_dim;
                k_gdn_conv1d_dec<<<int((tot + 255) / 256), 256, 0, stream>>>(
                    d_qkv, LW.conv1d, gdn_conv[size_t(l)], d_qkv2, dev_slots, dev_pos, M, qkv_dim);
                k_gdn_rec_norm_dec<<<M * c.gdn_vh, 256, 0, stream>>>(
                    d_qkv2, d_ab, d_ab + size_t(M) * c.gdn_vh, d_q_ready,
                    LW.A_log, LW.dt_bias, LW.gdn_norm, gdn_S[size_t(l)], d_attn_out,
                    dev_slots, dev_pos, M, c.gdn_kh, c.gdn_vh, c.rms_norm_eps);

                proj_f32(d_attn_out, LW.out_proj, LW.out_proj_nat, d_residual, M, H, val_dim, true, stream);
                toc(prof.gdn_ms);
            }

            // Sparse MoE + Shared Expert (100% on-GPU, zero host sync)
            tic();
            k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, LW.post_attn_ln, d_norm, M, H, c.rms_norm_eps);

            if (LW.tr3.valid) {
                tts_moe_prepare_x_had(d_norm, LW.tr3, M, moe_sc, stream);
            }
            if (LW.tr3.valid && LW.tr3.has_shared) {
                tts_moe_shared_forward(LW.tr3, M, moe_sc, d_shared_out, stream);
            } else {
                proj_bf16(d_norm, LW.shared_gate_proj, LW.shared_gate_nat, d_shared_g, M, c.shared_intermediate, H, stream);
                proj_bf16(d_norm, LW.shared_up_proj, LW.shared_up_nat, d_shared_u, M, c.shared_intermediate, H, stream);
                const size_t sh_n = size_t(M) * c.shared_intermediate;
                k_silu_mul_bf16<<<int((sh_n + 255) / 256), 256, 0, stream>>>(d_shared_g, d_shared_u, d_shared_g, sh_n);
                proj_f32(d_shared_g, LW.shared_down_proj, LW.shared_down_nat, d_shared_out, M, H, c.shared_intermediate, false, stream);
            }
            proj_f32(d_norm, LW.shared_expert_gate, LW.shared_expert_gate_nat, d_shared_gate, M, 1, H, false, stream);

            proj_f32(d_norm, LW.router_gate, LW.router_gate_nat, d_router_logits, M, c.num_experts, H, false, stream);
            k_moe_router_topk<<<M, 256, 0, stream>>>(
                d_router_logits, d_topk_ids, d_topk_w, M, c.num_experts, c.top_k, c.norm_topk_prob);

            if (LW.tr3.valid) {
                tts_moe_routed_forward(LW.tr3, d_topk_ids, d_topk_w, M, moe_sc, d_moe_routed_f32, stream);
            } else {
                run_ref_moe(LW, M, stream);
            }
            k_moe_finalize_res_f32<<<M, 256, 0, stream>>>(d_moe_routed_f32, d_shared_out, d_shared_gate, d_residual, M, H);
            toc(prof.moe_ms);
        }

        tic();
        k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_bf16, M, H, c.rms_norm_eps);
        if (want_f32_hidden) {
            k_zc_rmsnorm_f32_to_f32<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_f32, M, H, c.rms_norm_eps);
        }
        if (compute_logits) {
            compute_main_logits(d_final_bf16, M, stream);
        }
        toc(prof.head_ms);
    }

    // Runs backbone on `M` rows:
    //   - `is_single_slot_seg == true`: rows `0..M-1` belong to one `seg_slot` at positions `seg_pos0..seg_pos0+M-1` (prefill chunk).
    //   - `is_single_slot_seg == false`: row `m` belongs to `slots_host[m]` at position `pos_host[m]` (batched decode).
    // Produces:
    //   - `d_final_bf16` [M, H] (and `d_final_f32` [M, H] if `want_f32_hidden`)
    //   - `d_main_logits` [M, 4097] FP32 if `compute_logits`
    void forward(const int32_t* h_rows, int M,
                 bool is_single_slot_seg, int seg_slot, int seg_pos0,
                 const int32_t* slots_host, const int32_t* pos_host,
                 bool compute_logits, bool want_f32_hidden, cudaStream_t stream) {
        if (M <= 0 || M > max_chunk) {
            throw std::runtime_error("BackboneRunner::forward: M out of bounds");
        }
        TTS_BLAS_CK(cublasSetStream(blas, stream));
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int qkv_dim = (2 * c.gdn_kh + c.gdn_vh) * c.gdn_kd;
        const int val_dim = c.gdn_vh * c.gdn_vd;
        const int kv_dim = c.num_kv_heads * c.head_dim;

        if (upload_pending) {
            TTS_CK(cudaEventSynchronize(ev_upload));
            upload_pending = false;
        }
        std::memcpy(h_rows_stage, h_rows, size_t(M) * c.row_width() * sizeof(int32_t));
        for (int m = 0; m < M; ++m) {
            h_slots[m] = is_single_slot_seg ? seg_slot : slots_host[m];
            h_pos[m] = is_single_slot_seg ? (seg_pos0 + m) : pos_host[m];
        }
        TTS_CK(cudaMemcpyAsync(d_rows, h_rows_stage, size_t(M) * c.row_width() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_slots, h_slots, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_pos, h_pos, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaEventRecord(ev_upload, stream));
        upload_pending = true;

        auto tic = [&]() { if (prof.enabled) cudaEventRecord(ev0, stream); };
        auto toc = [&](double& dst) {
            if (prof.enabled) {
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                dst += double(ms);
            }
        };

        tic();
        if (W->embed_tokens_nat.valid() && W->embed_tokens_nat.fmt == g6emb::FMT_FP8_ROW) {
            k_backbone_embed_fp8<<<M, 256, 0, stream>>>(
                d_rows, W->embed_tokens_nat.data, W->embed_tokens_nat.scale, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        } else if (W->embed_tokens.f32 != nullptr) {
            k_backbone_embed_f32<<<M, 256, 0, stream>>>(
                d_rows, W->embed_tokens.f32, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        } else {
            k_backbone_embed<<<M, 256, 0, stream>>>(
                d_rows, W->embed_tokens.bf16, W->codebook_embeddings, d_residual,
                M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        }
        toc(prof.embed_ms);
        if (trace.enabled) {
            trace.reset(c.num_layers);
            trace.embed_f32.resize(size_t(M) * H);
            TTS_CK(cudaMemcpyAsync(trace.embed_f32.data(), d_residual, size_t(M) * H * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
            TTS_CK(cudaStreamSynchronize(stream));
        }

        for (int l = 0; l < c.num_layers; ++l) {
            const BackboneLayerW& LW = W->layers[size_t(l)];
            k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, LW.input_ln, d_norm, M, H, c.rms_norm_eps);

            if (LW.is_full_attn) {
                tic();
                if (W->mode == WeightMode::Native && LW.q_proj_nat.valid() && LW.k_proj_nat.valid() && LW.v_proj_nat.valid()) {
                    tts_native_linear_triple_bf16(d_norm, LW.q_proj_nat, LW.k_proj_nat, LW.v_proj_nat,
                                                  d_qkv, d_kv_tmp, d_kv_tmp + size_t(M) * kv_dim, M, dense_ws, blas, stream);
                } else {
                    proj_bf16(d_norm, LW.q_proj, LW.q_proj_nat, d_qkv, M, c.num_heads * 2 * c.head_dim, H, stream);
                    proj_bf16(d_norm, LW.k_proj, LW.k_proj_nat, d_kv_tmp, M, kv_dim, H, stream);
                    proj_bf16(d_norm, LW.v_proj, LW.v_proj_nat, d_kv_tmp + size_t(M) * kv_dim, M, kv_dim, H, stream);
                }

                dim3 qk_grid(M, c.num_heads + c.num_kv_heads);
                k_bb_qk_norm_rope_kv<<<qk_grid, 256, 0, stream>>>(
                    d_qkv, d_kv_tmp, d_kv_tmp + size_t(M) * kv_dim,
                    LW.q_norm, LW.k_norm, d_q_ready, k_caches[size_t(l)], v_caches[size_t(l)],
                    d_slots, d_pos, M, c.num_heads, c.num_kv_heads, c.head_dim, c.rotary_dim,
                    c.rms_norm_eps, c.rope_theta, cap);

                dim3 attn_grid(M, c.num_heads);
                k_gqa_attn_gated<<<attn_grid, 256, 0, stream>>>(
                    d_q_ready, d_qkv, k_caches[size_t(l)], v_caches[size_t(l)], d_attn_out,
                    d_slots, d_pos, M, c.num_heads, c.num_kv_heads, c.head_dim, cap,
                    true, 0, 0);

                // o_proj accumulates into FP32 residual
                if (trace.enabled && l == 3) {
                    auto dl_bf = [&](const bf16* p, size_t n, std::vector<float>& dst) {
                        std::vector<uint16_t> raw(n);
                        TTS_CK(cudaMemcpyAsync(raw.data(), p, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
                        TTS_CK(cudaStreamSynchronize(stream));
                        dst.resize(n);
                        for (size_t i = 0; i < n; ++i) dst[i] = host_bf16_to_f32(raw[i]);
                    };
                    dl_bf(d_q_ready, size_t(M) * c.num_heads * c.head_dim, trace.l3_q_ready);
                    dl_bf(d_attn_out, size_t(M) * c.num_heads * c.head_dim, trace.l3_attn_out);
                }
                proj_f32(d_attn_out, LW.o_proj, LW.o_proj_nat, d_residual, M, H, c.num_heads * c.head_dim, true, stream);
                toc(prof.attn_ms);
            } else {
                tic();
                if (W->mode == WeightMode::Native && LW.in_proj_qkv_nat.valid() && LW.in_proj_z_nat.valid()) {
                    tts_native_linear_dual_bf16(d_norm, LW.in_proj_qkv_nat, LW.in_proj_z_nat,
                                                d_qkv, d_q_ready, M, dense_ws, blas, stream);
                } else {
                    proj_bf16(d_norm, LW.in_proj_qkv, LW.in_proj_qkv_nat, d_qkv, M, qkv_dim, H, stream);
                    proj_bf16(d_norm, LW.in_proj_z, LW.in_proj_z_nat, d_q_ready, M, val_dim, H, stream);
                }
                proj_f32(d_norm, LW.in_proj_a, LW.in_proj_a_nat, d_ab, M, c.gdn_vh, H, false, stream);
                proj_f32(d_norm, LW.in_proj_b, LW.in_proj_b_nat, d_ab + size_t(M) * c.gdn_vh, M, c.gdn_vh, H, false, stream);

                if (is_single_slot_seg) {
                    k_gdn_conv1d_seg<<<(qkv_dim + 255) / 256, 256, 0, stream>>>(
                        d_qkv, LW.conv1d, gdn_conv[size_t(l)], d_qkv2, qkv_dim, seg_slot, 0, seg_pos0, M);
                    k_gdn_rec_norm_seg<<<c.gdn_vh, 256, 0, stream>>>(
                        d_qkv2, d_ab, d_ab + size_t(M) * c.gdn_vh, d_q_ready,
                        LW.A_log, LW.dt_bias, LW.gdn_norm, gdn_S[size_t(l)], d_attn_out,
                        c.gdn_kh, c.gdn_vh, seg_slot, 0, seg_pos0, M, c.rms_norm_eps);
                } else {
                    const size_t tot = size_t(M) * qkv_dim;
                    k_gdn_conv1d_dec<<<int((tot + 255) / 256), 256, 0, stream>>>(
                        d_qkv, LW.conv1d, gdn_conv[size_t(l)], d_qkv2, d_slots, d_pos, M, qkv_dim);
                    k_gdn_rec_norm_dec<<<M * c.gdn_vh, 256, 0, stream>>>(
                        d_qkv2, d_ab, d_ab + size_t(M) * c.gdn_vh, d_q_ready,
                        LW.A_log, LW.dt_bias, LW.gdn_norm, gdn_S[size_t(l)], d_attn_out,
                        d_slots, d_pos, M, c.gdn_kh, c.gdn_vh, c.rms_norm_eps);
                }
                if (trace.enabled && l == 0) {
                    auto dl_bf = [&](const bf16* p, size_t n, std::vector<float>& dst) {
                        std::vector<uint16_t> raw(n);
                        TTS_CK(cudaMemcpyAsync(raw.data(), p, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
                        TTS_CK(cudaStreamSynchronize(stream));
                        dst.resize(n);
                        for (size_t i = 0; i < n; ++i) dst[i] = host_bf16_to_f32(raw[i]);
                    };
                    dl_bf(d_qkv, size_t(M) * qkv_dim, trace.l0_qkv_raw);
                    dl_bf(d_qkv2, size_t(M) * qkv_dim, trace.l0_qkv_conv);
                    dl_bf(d_attn_out, size_t(M) * val_dim, trace.l0_core_out);
                }
                // out_proj accumulates into FP32 residual
                proj_f32(d_attn_out, LW.out_proj, LW.out_proj_nat, d_residual, M, H, val_dim, true, stream);
                toc(prof.gdn_ms);
            }

            // Sparse MoE + Shared Expert
            if (trace.enabled) {
                trace.mixer_residuals[size_t(l)].resize(size_t(M) * H);
                TTS_CK(cudaMemcpyAsync(trace.mixer_residuals[size_t(l)].data(), d_residual,
                                       size_t(M) * H * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
            tic();
            k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, LW.post_attn_ln, d_norm, M, H, c.rms_norm_eps);

            if (LW.tr3.valid) {
                tts_moe_prepare_x_had(d_norm, LW.tr3, M, moe_sc, stream);
            }

            // Shared expert (trellis shared row 256 when planned, else dense/BF16 shared expert)
            if (LW.tr3.valid && LW.tr3.has_shared) {
                tts_moe_shared_forward(LW.tr3, M, moe_sc, d_shared_out, stream);
            } else {
                proj_bf16(d_norm, LW.shared_gate_proj, LW.shared_gate_nat, d_shared_g, M, c.shared_intermediate, H, stream);
                proj_bf16(d_norm, LW.shared_up_proj, LW.shared_up_nat, d_shared_u, M, c.shared_intermediate, H, stream);
                const size_t sh_n = size_t(M) * c.shared_intermediate;
                k_silu_mul_bf16<<<int((sh_n + 255) / 256), 256, 0, stream>>>(d_shared_g, d_shared_u, d_shared_g, sh_n);
                proj_f32(d_shared_g, LW.shared_down_proj, LW.shared_down_nat, d_shared_out, M, H, c.shared_intermediate, false, stream);
            }
            proj_f32(d_norm, LW.shared_expert_gate, LW.shared_expert_gate_nat, d_shared_gate, M, 1, H, false, stream);

            // Router
            proj_f32(d_norm, LW.router_gate, LW.router_gate_nat, d_router_logits, M, c.num_experts, H, false, stream);
            k_moe_router_topk<<<M, 256, 0, stream>>>(
                d_router_logits, d_topk_ids, d_topk_w, M, c.num_experts, c.top_k, c.norm_topk_prob);
            if (trace.enabled) {
                trace.router_logits[size_t(l)].resize(size_t(M) * c.num_experts);
                TTS_CK(cudaMemcpyAsync(trace.router_logits[size_t(l)].data(), d_router_logits,
                                       size_t(M) * c.num_experts * sizeof(float), cudaMemcpyDeviceToHost, stream));
                trace.topk_ids[size_t(l)].resize(size_t(M) * c.top_k);
                trace.topk_w[size_t(l)].resize(size_t(M) * c.top_k);
                TTS_CK(cudaMemcpyAsync(trace.topk_ids[size_t(l)].data(), d_topk_ids,
                                       size_t(M) * c.top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                TTS_CK(cudaMemcpyAsync(trace.topk_w[size_t(l)].data(), d_topk_w,
                                       size_t(M) * c.top_k * sizeof(float), cudaMemcpyDeviceToHost, stream));
                TTS_CK(cudaStreamSynchronize(stream));
            }
            if (trace.replay_router && size_t(l) < trace.forced_topk_ids.size() &&
                trace.forced_topk_ids[size_t(l)].size() == size_t(M) * c.top_k) {
                TTS_CK(cudaMemcpyAsync(d_topk_ids, trace.forced_topk_ids[size_t(l)].data(),
                                       size_t(M) * c.top_k * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                TTS_CK(cudaMemcpyAsync(d_topk_w, trace.forced_topk_w[size_t(l)].data(),
                                       size_t(M) * c.top_k * sizeof(float), cudaMemcpyHostToDevice, stream));
            }

            if (LW.tr3.valid) {
                // Native grouped EXLR8 trellis routed experts (zero host sync!)
                tts_moe_routed_forward(LW.tr3, d_topk_ids, d_topk_w, M, moe_sc, d_moe_routed_f32, stream);
            } else {
                // Zero-host-sync on-GPU grouped Ref MoE (FP8_BLOCK or BF16)
                run_ref_moe(LW, M, stream);
            }
            k_moe_finalize_res_f32<<<M, 256, 0, stream>>>(d_moe_routed_f32, d_shared_out, d_shared_gate, d_residual, M, H);
            toc(prof.moe_ms);
            if (trace.enabled) {
                trace.residuals[size_t(l)].resize(size_t(M) * H);
                TTS_CK(cudaMemcpyAsync(trace.residuals[size_t(l)].data(), d_residual,
                                       size_t(M) * H * sizeof(float), cudaMemcpyDeviceToHost, stream));
                TTS_CK(cudaStreamSynchronize(stream));
            }
        }

        // Final norm + optional sliced lm_head
        tic();
        k_zc_rmsnorm_f32_to_bf16<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_bf16, M, H, c.rms_norm_eps);
        if (want_f32_hidden) {
            k_zc_rmsnorm_f32_to_f32<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_f32, M, H, c.rms_norm_eps);
            if (trace.enabled) {
                trace.final_hidden_f32.resize(size_t(M) * H);
                TTS_CK(cudaMemcpyAsync(trace.final_hidden_f32.data(), d_final_f32,
                                       size_t(M) * H * sizeof(float), cudaMemcpyDeviceToHost, stream));
                TTS_CK(cudaStreamSynchronize(stream));
            }
        }
        if (compute_logits) {
            compute_main_logits(d_final_bf16, M, stream);
        }
        toc(prof.head_ms);
    }
};

} // namespace g6tts
