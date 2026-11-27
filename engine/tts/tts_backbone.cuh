// Drama 3 (fish_omni) backbone forward pass in CUDA + cuBLAS FP16 (reference path).
//
// Exact match to fish-speech @ 0f8a12e8 (`fish_omni/{backbone,layers,moe}.py`):
//   - Backbone input: E_LM[main], and for semantic rows (SEM0 <= main <= SEM1)
//     (E_LM[main] + codebook_embeddings[4096 + c1]) / sqrt(2), held in an FP32 residual stream.
//   - 40 layers: l % 4 == 3 -> FullAttention (16 Q heads x 256, 2 KV heads x 256, q_proj=[q|gate] per head,
//     ZeroCenteredRMSNorm 1+w on q/k, NeoX half-split RoPE on first 64 dims theta=1e7, o_proj(attn*sigmoid(gate)));
//     l % 4 != 3 -> GatedDeltaNet (16 KH / 32 VH x 128, V_per_K=2, depthwise conv1d k=4 + SiLU, FP32 state [32,128,128],
//     GatedRMSNorm with plain weight * silu(z), out_proj).
//   - Every layer has SparseMoE: FP32 router softmax -> top-8 -> renormalize, 256 routed experts (either FP8_BLOCK
//     dequantized on the fly per active expert or pre-decoded FP16), sigmoid-gated BF16/FP16 shared expert.
//   - Final ZeroCenteredRMSNorm + sliced lm_head over the 4097 legal main-token candidates.
#pragma once

#include <cublas_v2.h>
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
#include "tts_weights.h"

namespace g6tts {

inline void tts_blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("g6tts cuBLAS error: ") + what + " (" + std::to_string(int(s)) + ")");
    }
}
#define TTS_BLAS_CK(x) ::g6tts::tts_blas_ck((x), #x)

// Row-major GEMM: Y[M, N] = alpha * X[M, K] @ W[N, K]^T + beta * Y[M, N]
// W and X are FP16; Y may be FP16 (`out_f32 == false`) or FP32 (`out_f32 == true`).
inline void gemm_f16(cublasHandle_t blas, const __half* X, const __half* W, void* Y,
                     int M, int N, int K, bool out_f32, float beta = 0.0f) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const float alpha = 1.0f;
    TTS_BLAS_CK(cublasGemmEx(blas,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             W, CUDA_R_16F, K,
                             X, CUDA_R_16F, K,
                             &beta,
                             Y, out_f32 ? CUDA_R_32F : CUDA_R_16F, N,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

// Dequantizes one FP8_BLOCK [out, in] matrix with 128x128 FP32 scales into FP16 [out, in].
// Clamps to [-65504, 65504] matching fish-speech clamp_to_float16.
__global__ void k_dequant_fp8_block128(const uint8_t* __restrict__ w_fp8,
                                       const float* __restrict__ w_sc,
                                       __half* __restrict__ out_f16,
                                       int out_dim, int in_dim, int bi) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(out_dim) * size_t(in_dim);
    if (idx >= total) return;
    const int r = static_cast<int>(idx / size_t(in_dim));
    const int c = static_cast<int>(idx % size_t(in_dim));
    const float s = w_sc[(r >> 7) * bi + (c >> 7)];
    float v = tts_e4m3_to_f32(w_fp8[idx]) * s;
    v = fminf(65504.0f, fmaxf(-65504.0f, v));
    out_f16[idx] = __float2half_rn(v);
}

// Backbone input embedding into FP32 residual stream:
//   rows[M, 11]: [main, c0, c1, ..., c9]
//   if SEM0 <= main <= SEM1: (E_LM[main] + CB_EMB[4096 + c1]) / sqrt(2)
//   else:                    E_LM[main]
__global__ void k_backbone_embed(const int32_t* __restrict__ rows,
                                 const __half* __restrict__ embed_tokens,
                                 const __half* __restrict__ cb_emb,
                                 float* __restrict__ out_f32,
                                 int M, int H, int row_w, int sem0, int sem1, int cb_vocab) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int32_t* r = rows + size_t(m) * row_w;
    const int main_tok = r[0];
    const bool is_sem = (main_tok >= sem0 && main_tok <= sem1);
    const int c1 = is_sem ? max(0, min(cb_vocab - 1, r[2])) : 0;
    const __half* e_tok = embed_tokens + size_t(main_tok) * H;
    const __half* e_cb = cb_emb + size_t(cb_vocab + c1) * H;
    float* dst = out_f32 + size_t(m) * H;
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float v = __half2float(e_tok[d]);
        if (is_sem) {
            v = (v + __half2float(e_cb[d])) * inv_sqrt2;
        }
        dst[d] = v;
    }
}

// ZeroCenteredRMSNorm from FP32 input -> FP16 output: y = fp16(x * rsqrt(mean(x^2) + eps) * (1 + w)).
// Grid: (M), Block: 256.
__global__ void k_zc_rmsnorm_f32_to_f16(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        __half* __restrict__ y,
                                        int M, int H, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + size_t(m) * H;
    __half* yr = y + size_t(m) * H;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < H; d += 256) {
        const float v = xr[d];
        ss += v * v;
    }
    const float rs = rsqrtf(tts_block_sum256(ss, red) / float(H) + eps);
    for (int d = threadIdx.x; d < H; d += 256) {
        const float v = xr[d] * rs * (1.0f + w[d]);
        yr[d] = __float2half_rn(fminf(65504.0f, fmaxf(-65504.0f, v)));
    }
}

// ZeroCenteredRMSNorm from FP32 input -> FP32 output (for teacher-forced hidden dump).
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
        yr[d] = xr[d] * rs * (1.0f + w[d]);
    }
}

// SwiGLU activation: out[i] = fp16(silu(gate[i]) * up[i]).
__global__ void k_silu_mul_f16(const __half* __restrict__ gate,
                               const __half* __restrict__ up,
                               __half* __restrict__ out,
                               size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float v = tts_silu(g) * u;
    out[i] = __float2half_rn(fminf(65504.0f, fmaxf(-65504.0f, v)));
}

// ------------------------------------------------------------------------------------------------ Full Attention kernels

// Backbone FullAttention Q/K ZeroCenteredRMSNorm + NeoX half-split partial RoPE + KV cache append.
// qg_in: [M, nq, 2*hd] (per-head [Q(256) | Gate(256)])
// k_in, v_in: [M, nkv, hd]
// Writes Q_out: [M, nq, hd], K_cache[slot, pos, nkv, hd], V_cache[slot, pos, nkv, hd].
// Grid: (M, nq + nkv), Block: 256 (1 thread per head_dim element for hd=256).
__global__ void k_bb_qk_norm_rope_kv(const __half* __restrict__ qg_in,
                                     const __half* __restrict__ k_in,
                                     const __half* __restrict__ v_in,
                                     const float* __restrict__ q_norm_w,
                                     const float* __restrict__ k_norm_w,
                                     __half* __restrict__ q_out,
                                     __half* __restrict__ k_cache,
                                     __half* __restrict__ v_cache,
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

    const __half* src = is_k
        ? (k_in + (size_t(m) * nkv + h) * hd)
        : (qg_in + (size_t(m) * nq + h) * (2 * hd)); // Q is first half of [Q | Gate] per head
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float xv = __half2float(src[d]);
    const float rs = rsqrtf(tts_block_sum256(xv * xv, red) / float(hd) + eps);
    const float normed = xv * rs * (1.0f + nw[d]);
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
        k_cache[kv_off] = __float2half_rn(out_v);
        v_cache[kv_off] = v_in[(size_t(m) * nkv + h) * hd + d];
    } else {
        q_out[(size_t(m) * nq + h) * hd + d] = __float2half_rn(out_v);
    }
}

// Causal GQA attention with sigmoid output gate (head_dim=256).
// Gate layout: if `gate_interleaved_per_head == true` (Backbone): gate is second half of `gate_src[m, hq, 2*hd]`.
//              if `gate_interleaved_per_head == false` (Tower):   gate is `gate_src[m * gate_stride + hq * hd + d]`.
// Grid: (M, nq), Block: 256 (1 thread per head_dim element).
__global__ void k_gqa_attn_gated(const __half* __restrict__ q_in,
                                 const __half* __restrict__ gate_src,
                                 const __half* __restrict__ k_cache,
                                 const __half* __restrict__ v_cache,
                                 __half* __restrict__ out,
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

    const float qv = __half2float(q_in[(size_t(m) * nq + hq) * hd + d]);
    const size_t slot_base = size_t(slot) * cap * nkv * hd;
    const size_t kv_stride = size_t(nkv) * hd;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= pos; ++t) {
        const size_t koff = slot_base + size_t(t) * kv_stride + size_t(hkv) * hd + d;
        const float kv = __half2float(k_cache[koff]);
        const float dot = tts_block_sum256(qv * kv, red) * scale;
        const float nmax = fmaxf(m_max, dot);
        const float alpha = __expf(m_max - nmax);
        const float p = __expf(dot - nmax);
        m_max = nmax;
        l_sum = l_sum * alpha + p;
        acc = acc * alpha + p * __half2float(v_cache[koff]);
    }

    const float attn_v = (l_sum > 0.0f) ? (acc / l_sum) : 0.0f;
    float g_val = 0.0f;
    if (gate_interleaved_per_head) {
        g_val = __half2float(gate_src[(size_t(m) * nq + hq) * (2 * hd) + hd + d]);
    } else {
        g_val = __half2float(gate_src[size_t(m) * gate_stride + gate_offset + size_t(hq) * hd + d]);
    }
    out[(size_t(m) * nq + hq) * hd + d] = __float2half_rn(attn_v * tts_sigmoid(g_val));
}

// ------------------------------------------------------------------------------------------------ GDN kernels

// Causal depthwise conv1d (k=4) + SiLU over a contiguous segment of `len` rows for one slot.
// Grid: ((qkv_dim + 255) / 256), Block: 256.
__global__ void k_gdn_conv1d_seg(const __half* __restrict__ qkv_raw,
                                 const float* __restrict__ conv_w,
                                 float* __restrict__ conv_hist,
                                 __half* __restrict__ qkv_out,
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
        const float x = __half2float(qkv_raw[idx]);
        const float y = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        qkv_out[idx] = __float2half_rn(tts_silu(y));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    hb[0 * qkv_dim + c] = h0;
    hb[1 * qkv_dim + c] = h1;
    hb[2 * qkv_dim + c] = h2;
}

// Batched single-token decode conv1d (k=4) + SiLU for M active slots in one launch.
__global__ void k_gdn_conv1d_dec(const __half* __restrict__ qkv_raw,
                                 const float* __restrict__ conv_w,
                                 float* __restrict__ conv_hist,
                                 __half* __restrict__ qkv_out,
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
    const float x = __half2float(qkv_raw[idx]);
    const float y = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3];
    qkv_out[idx] = __float2half_rn(tts_silu(y));
    hb[0 * qkv_dim + c] = h1;
    hb[1 * qkv_dim + c] = h2;
    hb[2 * qkv_dim + c] = x;
}

// Recurrent Gated Delta Rule + GatedRMSNorm(plain weight * silu(z)) over a contiguous segment for one slot.
// Grid: (nh_v = 32), Block: 256 (v = tid & 127, kh = tid >> 7, each thread holds 64 rows of S[128,128]).
__global__ void k_gdn_rec_norm_seg(const __half* __restrict__ qkv_conv,
                                   const float* __restrict__ a_in,
                                   const float* __restrict__ b_in,
                                   const __half* __restrict__ z_in,
                                   const float* __restrict__ A_log,
                                   const float* __restrict__ dt_bias,
                                   const float* __restrict__ norm_w,
                                   float* __restrict__ S_all,
                                   __half* __restrict__ core_out,
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
    __shared__ float sp[128];
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
        const __half* qkv = qkv_conv + size_t(r) * qkv_dim;
        const __half* q_ptr = qkv + size_t(hk) * 128;
        const __half* k_ptr = qkv + size_t(nh_k + hk) * 128;
        const __half* v_ptr = qkv + size_t(2 * nh_k + hv) * 128;

        if (kh == 0) sq[v] = __half2float(q_ptr[v]);
        else sk[v] = __half2float(k_ptr[v]);
        if (tid == 0) {
            const float av = a_in[size_t(r) * nh_v + hv];
            const float bv = b_in[size_t(r) * nh_v + hv];
            s_meta[0] = expf(neg_A * tts_softplus(av + dtb));
            s_meta[1] = tts_sigmoid(bv);
        }
        __syncthreads();

        if (kh == 0) {
            const float qv = sq[v];
            const float q2 = tts_warp_sum(qv * qv);
            if ((v & 31) == 0) qwarp[v >> 5] = q2;
            __syncthreads();
            const float qsum = qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3];
            sq[v] = qv * rsqrtf(qsum + 1e-6f) * inv_sqrt_dk;
        } else {
            const float kv = sk[v];
            const float k2 = tts_warp_sum(kv * kv);
            if ((v & 31) == 0) kwarp[v >> 5] = k2;
            __syncthreads();
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
        if (kh == 1) sp[v] = kv_part;
        __syncthreads();
        if (kh == 0) {
            const float vv = __half2float(v_ptr[v]);
            sp[v] = (vv - (kv_part + sp[v])) * beta;
        }
        __syncthreads();
        const float delta_v = sp[v];

        float out_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float sv = fmaf(sk[k_base + i], delta_v, s_col[i]);
            s_col[i] = sv;
            out_part = fmaf(sv, sq[k_base + i], out_part);
        }
        if (kh == 1) sp[v] = out_part;
        __syncthreads();

        // GatedRMSNorm: plain norm_w * rmsnorm(o) * silu(z)
        if (kh == 0) {
            const float ov = out_part + sp[v];
            const float ss = tts_warp_sum(ov * ov);
            if ((v & 31) == 0) qwarp[v >> 5] = ss;
            __syncthreads();
            const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
            const float normed = ov * inv * norm_w[v];
            const size_t zoff = (size_t(r) * nh_v + hv) * 128 + v;
            const float gate = tts_silu(__half2float(z_in[zoff]));
            core_out[zoff] = __float2half_rn(normed * gate);
        } else {
            __syncthreads();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 64; ++i) S_head[(k_base + i) * 128 + v] = s_col[i];
}

// Batched single-token decode GDN recurrence + GatedRMSNorm across M slots in ONE launch.
// Grid: (M * nh_v), Block: 256.
__global__ void k_gdn_rec_norm_dec(const __half* __restrict__ qkv_conv,
                                   const float* __restrict__ a_in,
                                   const float* __restrict__ b_in,
                                   const __half* __restrict__ z_in,
                                   const float* __restrict__ A_log,
                                   const float* __restrict__ dt_bias,
                                   const float* __restrict__ norm_w,
                                   float* __restrict__ S_all,
                                   __half* __restrict__ core_out,
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

    __shared__ float sq[128], sk[128], sp[128], s_meta[2], qwarp[4], kwarp[4];
    const __half* qkv = qkv_conv + size_t(m) * qkv_dim;
    const __half* q_ptr = qkv + size_t(hk) * 128;
    const __half* k_ptr = qkv + size_t(nh_k + hk) * 128;
    const __half* v_ptr = qkv + size_t(2 * nh_k + hv) * 128;

    if (kh == 0) sq[v] = __half2float(q_ptr[v]);
    else sk[v] = __half2float(k_ptr[v]);
    if (tid == 0) {
        const float av = a_in[size_t(m) * nh_v + hv];
        const float bv = b_in[size_t(m) * nh_v + hv];
        s_meta[0] = expf(-expf(A_log[hv]) * tts_softplus(av + dt_bias[hv]));
        s_meta[1] = tts_sigmoid(bv);
    }
    __syncthreads();

    constexpr float inv_sqrt_dk = 0.08838834764831845f;
    if (kh == 0) {
        const float qv = sq[v];
        const float q2 = tts_warp_sum(qv * qv);
        if ((v & 31) == 0) qwarp[v >> 5] = q2;
        __syncthreads();
        sq[v] = qv * rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) + 1e-6f) * inv_sqrt_dk;
    } else {
        const float kv = sk[v];
        const float k2 = tts_warp_sum(kv * kv);
        if ((v & 31) == 0) kwarp[v >> 5] = k2;
        __syncthreads();
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
    if (kh == 1) sp[v] = kv_part;
    __syncthreads();
    if (kh == 0) {
        sp[v] = (__half2float(v_ptr[v]) - (kv_part + sp[v])) * beta;
    }
    __syncthreads();
    const float delta_v = sp[v];

    float out_part = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const float sv = fmaf(sk[k_base + i], delta_v, s_col[i]);
        S_head[(k_base + i) * 128 + v] = sv;
        out_part = fmaf(sv, sq[k_base + i], out_part);
    }
    if (kh == 1) sp[v] = out_part;
    __syncthreads();

    if (kh == 0) {
        const float ov = out_part + sp[v];
        const float ss = tts_warp_sum(ov * ov);
        if ((v & 31) == 0) qwarp[v >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
        const size_t zoff = size_t(idx) * 128 + v;
        core_out[zoff] = __float2half_rn(ov * inv * norm_w[v] * tts_silu(__half2float(z_in[zoff])));
    }
}

// ------------------------------------------------------------------------------------------------ MoE kernels

// Router: FP32 softmax over 256 experts -> top-K -> optional renormalize.
// Grid: (M), Block: 256 (one thread per expert for NEXP=256).
__global__ void k_moe_router_topk(const float* __restrict__ logits,
                                  int32_t* __restrict__ topk_ids,
                                  float* __restrict__ topk_w,
                                  int M, int nexp, int topk, bool norm_topk) {
    __shared__ float s_p[256];
    __shared__ float red[8];
    const int m = blockIdx.x;
    const int e = threadIdx.x;
    if (m >= M) return;

    const float v = (e < nexp) ? logits[size_t(m) * nexp + e] : -1e30f;
    // Block max
    float mx = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((e & 31) == 0) red[e >> 5] = mx;
    __syncthreads();
    if (e < 32) {
        float bm = (e < 8) ? red[e] : -1e30f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off));
        if (e == 0) red[0] = bm;
    }
    __syncthreads();
    const float row_max = red[0];

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
        if (norm_topk && wsum > 0.0f) {
            const float inv = 1.0f / wsum;
            for (int k = 0; k < topk; ++k) topk_w[size_t(m) * topk + k] *= inv;
        }
    }
}

__global__ void k_gather_rows_f16(const __half* __restrict__ src,
                                  const int32_t* __restrict__ row_ids,
                                  __half* __restrict__ dst,
                                  int cnt, int H) {
    const int i = blockIdx.x;
    if (i >= cnt) return;
    const __half* s = src + size_t(row_ids[i]) * H;
    __half* d = dst + size_t(i) * H;
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

__global__ void k_add_gated_shared_f32(const float* __restrict__ shared_out,
                                       const float* __restrict__ shared_gate_logit,
                                       float* __restrict__ residual,
                                       int M, int H) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float g = tts_sigmoid(shared_gate_logit[m]);
    const float* s = shared_out + size_t(m) * H;
    float* r = residual + size_t(m) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        r[c] += g * s[c];
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

struct BackboneRunner {
    const TtsWeights* W = nullptr;
    cublasHandle_t blas = nullptr;
    int max_slots = 8;
    int cap = 4096;
    int max_chunk = 128;

    // Persistent per-slot caches
    std::vector<__half*> k_caches;  // [L], non-null for full-attn layers: [max_slots, cap, 2, 256]
    std::vector<__half*> v_caches;  // [L]
    std::vector<float*> gdn_S;      // [L], non-null for GDN layers: [max_slots, 32, 128, 128]
    std::vector<float*> gdn_conv;   // [L], non-null for GDN layers: [max_slots, 3, 8192]

    // Scratch buffers (allocated for `max_chunk` rows)
    int32_t* d_rows = nullptr;          // [max_chunk, 11]
    int32_t* d_slots = nullptr;         // [max_chunk]
    int32_t* d_pos = nullptr;           // [max_chunk]
    float* d_residual = nullptr;        // [max_chunk, H]
    __half* d_norm = nullptr;           // [max_chunk, H]
    __half* d_qkv = nullptr;            // [max_chunk, 8192] (reused by FullAttn qg & GDN qkv_raw)
    __half* d_qkv2 = nullptr;           // [max_chunk, 8192] (GDN qkv_conv)
    __half* d_q_ready = nullptr;        // [max_chunk, 4096] (FullAttn Q & GDN z / core)
    __half* d_kv_tmp = nullptr;         // [max_chunk, 1024] (K[512] + V[512])
    __half* d_attn_out = nullptr;       // [max_chunk, 4096]
    float* d_ab = nullptr;              // [max_chunk, 64] (GDN a[32] + b[32])
    float* d_router_logits = nullptr;   // [max_chunk, 256]
    int32_t* d_topk_ids = nullptr;      // [max_chunk, 8]
    float* d_topk_w = nullptr;          // [max_chunk, 8]
    __half* d_shared_g = nullptr;       // [max_chunk, 512]
    __half* d_shared_u = nullptr;       // [max_chunk, 512]
    float* d_shared_out = nullptr;      // [max_chunk, H]
    float* d_shared_gate = nullptr;     // [max_chunk]
    __half* d_exp_x = nullptr;          // [max_chunk, H]
    __half* d_exp_g = nullptr;          // [max_chunk, 512]
    __half* d_exp_u = nullptr;          // [max_chunk, 512]
    float* d_exp_out = nullptr;         // [max_chunk, H]
    int32_t* d_exp_rows = nullptr;      // [max_chunk]
    float* d_exp_w = nullptr;           // [max_chunk]
    __half* d_deq_g = nullptr;          // [512, 2048] scratch for on-the-fly FP8 expert dequant
    __half* d_deq_u = nullptr;          // [512, 2048]
    __half* d_deq_d = nullptr;          // [2048, 512]
    __half* d_final_f16 = nullptr;      // [max_chunk, H]
    float* d_final_f32 = nullptr;       // [max_chunk, H]
    float* d_main_logits = nullptr;     // [max_chunk, 4097]

    // Pinned host scratch for tiny MoE routing table & slot/pos staging
    int32_t* h_topk_ids = nullptr;
    float* h_topk_w = nullptr;
    int32_t* h_exp_rows = nullptr;
    float* h_exp_w = nullptr;
    int32_t* h_slots = nullptr;
    int32_t* h_pos = nullptr;

    size_t state_bytes = 0;
    size_t scratch_bytes = 0;
    BackboneProfile prof;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    ~BackboneRunner() { release(); }

    void release() {
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto freeh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        for (auto*& p : k_caches) freep(p);
        for (auto*& p : v_caches) freep(p);
        for (auto*& p : gdn_S) freep(p);
        for (auto*& p : gdn_conv) freep(p);
        k_caches.clear(); v_caches.clear(); gdn_S.clear(); gdn_conv.clear();
        freep(d_rows); freep(d_slots); freep(d_pos); freep(d_residual); freep(d_norm);
        freep(d_qkv); freep(d_qkv2); freep(d_q_ready); freep(d_kv_tmp); freep(d_attn_out);
        freep(d_ab); freep(d_router_logits); freep(d_topk_ids); freep(d_topk_w);
        freep(d_shared_g); freep(d_shared_u); freep(d_shared_out); freep(d_shared_gate);
        freep(d_exp_x); freep(d_exp_g); freep(d_exp_u); freep(d_exp_out);
        freep(d_exp_rows); freep(d_exp_w);
        freep(d_deq_g); freep(d_deq_u); freep(d_deq_d);
        freep(d_final_f16); freep(d_final_f32); freep(d_main_logits);
        freeh(h_topk_ids); freeh(h_topk_w); freeh(h_exp_rows); freeh(h_exp_w);
        freeh(h_slots); freeh(h_pos);
        if (ev0) { cudaEventDestroy(ev0); ev0 = nullptr; }
        if (ev1) { cudaEventDestroy(ev1); ev1 = nullptr; }
        if (blas) { cublasDestroy(blas); blas = nullptr; }
    }

    void init(const TtsWeights& weights, int streams, int kv_cap, int chunk = 128) {
        release();
        W = &weights;
        max_slots = std::max(1, streams);
        cap = std::max(16, kv_cap);
        max_chunk = std::max(16, chunk);

        TTS_BLAS_CK(cublasCreate(&blas));
        TTS_CK(cudaEventCreate(&ev0));
        TTS_CK(cudaEventCreate(&ev1));

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
                const size_t kv_b = size_t(max_slots) * cap * c.num_kv_heads * c.head_dim * sizeof(__half);
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
        alloc_d(d_exp_rows, size_t(max_chunk));
        alloc_d(d_exp_w, size_t(max_chunk));
        alloc_d(d_deq_g, size_t(I) * H);
        alloc_d(d_deq_u, size_t(I) * H);
        alloc_d(d_deq_d, size_t(H) * I);
        alloc_d(d_final_f16, size_t(max_chunk) * H);
        alloc_d(d_final_f32, size_t(max_chunk) * H);
        alloc_d(d_main_logits, size_t(max_chunk) * n_cand);

        TTS_CK(cudaHostAlloc(&h_topk_ids, size_t(max_chunk) * c.top_k * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_topk_w, size_t(max_chunk) * c.top_k * sizeof(float), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_exp_rows, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_exp_w, size_t(max_chunk) * sizeof(float), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_slots, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_pos, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
    }

    // Runs backbone on `M` rows:
    //   - `is_single_slot_seg == true`: rows `0..M-1` belong to one `seg_slot` at positions `seg_pos0..seg_pos0+M-1` (prefill chunk).
    //   - `is_single_slot_seg == false`: row `m` belongs to `slots_host[m]` at position `pos_host[m]` (batched decode).
    // Produces:
    //   - `d_final_f16` [M, H] (and `d_final_f32` [M, H] if `want_f32_hidden`)
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
        const int I = c.moe_intermediate;
        const int qkv_dim = (2 * c.gdn_kh + c.gdn_vh) * c.gdn_kd;
        const int val_dim = c.gdn_vh * c.gdn_vd;
        const int kv_dim = c.num_kv_heads * c.head_dim;

        for (int m = 0; m < M; ++m) {
            h_slots[m] = is_single_slot_seg ? seg_slot : slots_host[m];
            h_pos[m] = is_single_slot_seg ? (seg_pos0 + m) : pos_host[m];
        }
        TTS_CK(cudaMemcpyAsync(d_rows, h_rows, size_t(M) * c.row_width() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_slots, h_slots, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_pos, h_pos, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

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
        k_backbone_embed<<<M, 256, 0, stream>>>(
            d_rows, W->embed_tokens, W->codebook_embeddings, d_residual,
            M, H, c.row_width(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        toc(prof.embed_ms);

        // Reuse grouping vectors across layers
        std::vector<std::vector<std::pair<int32_t, float>>> exp_buckets(static_cast<size_t>(c.num_experts));

        for (int l = 0; l < c.num_layers; ++l) {
            const BackboneLayerW& LW = W->layers[size_t(l)];
            k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_residual, LW.input_ln, d_norm, M, H, c.rms_norm_eps);

            if (LW.is_full_attn) {
                tic();
                gemm_f16(blas, d_norm, LW.q_proj, d_qkv, M, c.num_heads * 2 * c.head_dim, H, false, 0.0f);
                gemm_f16(blas, d_norm, LW.k_proj, d_kv_tmp, M, kv_dim, H, false, 0.0f);
                gemm_f16(blas, d_norm, LW.v_proj, d_kv_tmp + size_t(M) * kv_dim, M, kv_dim, H, false, 0.0f);

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

                // o_proj accumulates directly into FP32 residual
                gemm_f16(blas, d_attn_out, LW.o_proj, d_residual, M, H, c.num_heads * c.head_dim, true, 1.0f);
                toc(prof.attn_ms);
            } else {
                tic();
                gemm_f16(blas, d_norm, LW.in_proj_qkv, d_qkv, M, qkv_dim, H, false, 0.0f);
                gemm_f16(blas, d_norm, LW.in_proj_z, d_q_ready, M, val_dim, H, false, 0.0f);
                gemm_f16(blas, d_norm, LW.in_proj_a, d_ab, M, c.gdn_vh, H, true, 0.0f);
                gemm_f16(blas, d_norm, LW.in_proj_b, d_ab + size_t(M) * c.gdn_vh, M, c.gdn_vh, H, true, 0.0f);

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
                // out_proj accumulates directly into FP32 residual
                gemm_f16(blas, d_attn_out, LW.out_proj, d_residual, M, H, val_dim, true, 1.0f);
                toc(prof.gdn_ms);
            }

            // Sparse MoE + Shared Expert
            tic();
            k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_residual, LW.post_attn_ln, d_norm, M, H, c.rms_norm_eps);

            // Shared expert
            gemm_f16(blas, d_norm, LW.shared_gate_proj, d_shared_g, M, c.shared_intermediate, H, false, 0.0f);
            gemm_f16(blas, d_norm, LW.shared_up_proj, d_shared_u, M, c.shared_intermediate, H, false, 0.0f);
            const size_t sh_n = size_t(M) * c.shared_intermediate;
            k_silu_mul_f16<<<int((sh_n + 255) / 256), 256, 0, stream>>>(d_shared_g, d_shared_u, d_shared_g, sh_n);
            gemm_f16(blas, d_shared_g, LW.shared_down_proj, d_shared_out, M, H, c.shared_intermediate, true, 0.0f);
            gemm_f16(blas, d_norm, LW.shared_expert_gate, d_shared_gate, M, 1, H, true, 0.0f);
            k_add_gated_shared_f32<<<M, 256, 0, stream>>>(d_shared_out, d_shared_gate, d_residual, M, H);

            // Router
            gemm_f16(blas, d_norm, LW.router_gate, d_router_logits, M, c.num_experts, H, true, 0.0f);
            k_moe_router_topk<<<M, 256, 0, stream>>>(
                d_router_logits, d_topk_ids, d_topk_w, M, c.num_experts, c.top_k, c.norm_topk_prob);

            // Reference grouped routed experts (syncs tiny M*8 routing table)
            TTS_CK(cudaMemcpyAsync(h_topk_ids, d_topk_ids, size_t(M) * c.top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            TTS_CK(cudaMemcpyAsync(h_topk_w, d_topk_w, size_t(M) * c.top_k * sizeof(float), cudaMemcpyDeviceToHost, stream));
            TTS_CK(cudaStreamSynchronize(stream));

            for (auto& b : exp_buckets) b.clear();
            for (int m = 0; m < M; ++m) {
                for (int k = 0; k < c.top_k; ++k) {
                    const int e = h_topk_ids[m * c.top_k + k];
                    const float w = h_topk_w[m * c.top_k + k];
                    exp_buckets[size_t(e)].push_back({m, w});
                }
            }

            const int bo_gu = (I + 127) / 128;
            const int bi_gu = (H + 127) / 128;
            const int bo_dn = (H + 127) / 128;
            const int bi_dn = (I + 127) / 128;
            const size_t gu_elems = size_t(I) * H;
            const size_t gu_sc_elems = size_t(bo_gu) * bi_gu;
            const size_t dn_sc_elems = size_t(bo_dn) * bi_dn;

            for (int e = 0; e < c.num_experts; ++e) {
                const auto& bucket = exp_buckets[size_t(e)];
                if (bucket.empty()) continue;
                const int cnt = static_cast<int>(bucket.size());
                for (int i = 0; i < cnt; ++i) {
                    h_exp_rows[i] = bucket[size_t(i)].first;
                    h_exp_w[i] = bucket[size_t(i)].second;
                }
                TTS_CK(cudaMemcpyAsync(d_exp_rows, h_exp_rows, size_t(cnt) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                TTS_CK(cudaMemcpyAsync(d_exp_w, h_exp_w, size_t(cnt) * sizeof(float), cudaMemcpyHostToDevice, stream));
                k_gather_rows_f16<<<cnt, 256, 0, stream>>>(d_norm, d_exp_rows, d_exp_x, cnt, H);

                const __half* wg = nullptr;
                const __half* wu = nullptr;
                const __half* wd = nullptr;
                if (LW.experts_fp8) {
                    const int blocks = int((gu_elems + 255) / 256);
                    k_dequant_fp8_block128<<<blocks, 256, 0, stream>>>(
                        LW.exp_gate_fp8 + size_t(e) * gu_elems, LW.exp_gate_sc + size_t(e) * gu_sc_elems,
                        d_deq_g, I, H, bi_gu);
                    k_dequant_fp8_block128<<<blocks, 256, 0, stream>>>(
                        LW.exp_up_fp8 + size_t(e) * gu_elems, LW.exp_up_sc + size_t(e) * gu_sc_elems,
                        d_deq_u, I, H, bi_gu);
                    k_dequant_fp8_block128<<<blocks, 256, 0, stream>>>(
                        LW.exp_down_fp8 + size_t(e) * gu_elems, LW.exp_down_sc + size_t(e) * dn_sc_elems,
                        d_deq_d, H, I, bi_dn);
                    wg = d_deq_g; wu = d_deq_u; wd = d_deq_d;
                } else {
                    wg = LW.exp_gate_f16 + size_t(e) * gu_elems;
                    wu = LW.exp_up_f16 + size_t(e) * gu_elems;
                    wd = LW.exp_down_f16 + size_t(e) * gu_elems;
                }

                gemm_f16(blas, d_exp_x, wg, d_exp_g, cnt, I, H, false, 0.0f);
                gemm_f16(blas, d_exp_x, wu, d_exp_u, cnt, I, H, false, 0.0f);
                const size_t act_n = size_t(cnt) * I;
                k_silu_mul_f16<<<int((act_n + 255) / 256), 256, 0, stream>>>(d_exp_g, d_exp_u, d_exp_g, act_n);
                gemm_f16(blas, d_exp_g, wd, d_exp_out, cnt, H, I, true, 0.0f);
                k_scatter_add_f32<<<cnt, 256, 0, stream>>>(d_exp_out, d_exp_rows, d_exp_w, d_residual, cnt, H);
            }
            toc(prof.moe_ms);
        }

        // Final norm + optional sliced lm_head
        tic();
        k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_f16, M, H, c.rms_norm_eps);
        if (want_f32_hidden) {
            k_zc_rmsnorm_f32_to_f32<<<M, 256, 0, stream>>>(d_residual, W->final_norm, d_final_f32, M, H, c.rms_norm_eps);
        }
        if (compute_logits) {
            gemm_f16(blas, d_final_f16, W->lm_head_cand, d_main_logits, M, c.main_candidates(), H, true, 0.0f);
        }
        toc(prof.head_ms);
    }
};

} // namespace g6tts
