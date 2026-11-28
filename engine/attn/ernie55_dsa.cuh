// ERNIE-5.5-Flash-Preview 6x Sparse MLA-DSA Attention (L7, 14, 21, 28, 35, 42) + Dynamic Paged KV Pool
// (BF16 / FP8 / INT4 KV + BF16 / FP8 Lightning Indexer K).
//
// Exact ERNIE 5.5 math invariants verified against config.json & CPU reference:
//   - Plain RoPE (theta = 8,000,000.0, NO YaRN!):
//       Main MLA Q/K: interleaved RoPE on trailing 64 dims ([192..255] of Q, [512..575] of KV)
//       Lightning Indexer Q/K: NeoX split-half partial RoPE on LEADING 64 of 128 dims + Hadamard-128!
//   - Lightning Indexer (64 heads x 128 dim, top-2048 over [0, seq_len - 128) merged with SWA-128):
//       iq = had128(neox_rope64(idx_wq_b(q_c).view(M, 64, 128)))
//       ik = had128(neox_rope64(layer_norm(idx_wk(x), k_norm_w, k_norm_b, 1e-6)))
//       iw = idx_weights_proj(x) * (128^-0.5) * (64^-0.5)
//       score[t] = sum_{h=0..63} iw[h] * relu(iq[h] . ik[t])   for t in [0 .. pos - 128]
//       sel = topk(score, 2048) union [max(0, pos - 127) .. pos]   (<= 2176 keys!)
//   - Absorbed MLA (kv_b unpacked at boot -> w_uk_t [nh_loc*512, 192] + w_uv [nh_loc*256, 512]):
//       q_abs = [q_nope @ w_uk, q_rope]  [M, nh_loc, 576]
//       online softmax initialized with F32 softmax_offset sink (m = sink[h], l = 1.0)
//       v = (probs @ kv_c) @ w_uv^T      [M, nh_loc, 256]
//       o = VHAPostMix(v, I + V @ U^T)   [M, nh_loc, 256]
//       out = o_proj(o.reshape(M, nh_loc*256) * sigmoid(gate_proj(q_c)))
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../model_geom_ernie55.h"
#include "../kernels/ernie55_mhc.cuh"
#include "../kernels/ernie55_mxfp8.cuh"
#include "../kernels/ernie55_moe.cuh"
#include "ernie55_hca.cuh"

namespace ernie55 {

enum class KvFmt : int { BF16 = 0, FP8 = 1, INT4 = 2 };

constexpr int KV_FP8_STRIDE  = 672; // 512 E4M3 nope + 32 B (16 half scales) + 128 B (64 bf16 rope)
constexpr int KV_INT4_STRIDE = 416; // 256 packed INT4 nope + 32 B (16 half scales) + 128 B (64 bf16 rope)
constexpr int IDX_SCORE_BATCH = 32;
constexpr int DSA_MAX_SPLITS  = 32;
constexpr int DSA_TC_SMEM     = 32 * 584 * 2 + 8 * 16 * 32 * 4; // 53,760 B

inline size_t dsa_kv_tok_bytes(KvFmt f) {
    if (f == KvFmt::FP8)  return KV_FP8_STRIDE;
    if (f == KvFmt::INT4) return KV_INT4_STRIDE;
    return geom::KVA * sizeof(__nv_bfloat16);
}

struct DsaWeights {
    int nh_loc = 0;                               // 64 / nr
    int h_off  = 0;                               // rank * nh_loc
    mx8::LinW            q_a_proj{};              // [1024, 4096]
    const __nv_bfloat16* q_a_ln       = nullptr;  // [1024]
    mx8::LinW            q_b_proj{};              // [nh_loc * 256, 1024]
    mx8::LinW            kv_a_proj{};             // [576, 4096]
    const __nv_bfloat16* kv_a_ln      = nullptr;  // [512]
    mx8::LinW            w_uk_t{};                // [nh_loc * 512, 192]
    mx8::LinW            w_uv{};                  // [nh_loc * 256, 512]
    mx8::LinW            gate_proj{};             // [nh_loc * 256, 1024] (from q_c!)
    mx8::LinW            o_proj{};                // [4096, nh_loc * 256]
    const float*         sinks        = nullptr;  // [nh_loc] F32 (converted from BF16 softmax_offset at boot)
    const float*         vha_mix      = nullptr;  // [64, 64] F32

    // Lightning Indexer
    mx8::LinW            idx_wq_b{};              // [64 * 128, 1024]
    mx8::LinW            idx_wk{};                // [128, 4096]
    const __nv_bfloat16* idx_k_norm_w = nullptr;  // [128]
    const __nv_bfloat16* idx_k_norm_b = nullptr;  // [128]
    const __nv_bfloat16* idx_w_proj   = nullptr;  // [64, 4096] BF16 (kept exact F32-accum)
};

struct DsaLayerKv {
    KvFmt fmt = KvFmt::FP8;
    uint8_t*       kv_pool  = nullptr; // [pool_pages][64][dsa_kv_tok_bytes(fmt)]
    __nv_bfloat16* idx_bf16 = nullptr; // BF16 mode: [pool_pages][64][128] BF16
    uint8_t*       idx_q    = nullptr; // FP8/INT4 mode: [pool_pages][64][128] E4M3
    __half*        idx_sc   = nullptr; // FP8/INT4 mode: [pool_pages][64][4] FP16 (per-32 scales)
};

struct DsaScratch {
    __nv_bfloat16* q_c      = nullptr; // [max_M, 1024]
    __nv_bfloat16* q_raw    = nullptr; // [max_M, nh_loc * 256]
    __nv_bfloat16* q_nope_h = nullptr; // [nh_loc, max_M, 192]
    __nv_bfloat16* q_abs_h  = nullptr; // [nh_loc, max_M, 512]
    __nv_bfloat16* q_full   = nullptr; // [max_M, nh_loc, 576]
    __nv_bfloat16* kv_raw   = nullptr; // [max_M, 576]
    __nv_bfloat16* gate     = nullptr; // [max_M, nh_loc * 256]
    __nv_bfloat16* u_h      = nullptr; // [nh_loc, max_M, 512]
    __nv_bfloat16* v_h      = nullptr; // [nh_loc, max_M, 256]
    __nv_bfloat16* attn_out = nullptr; // [max_M, nh_loc * 256]
    __nv_bfloat16* kv_b_tmp = nullptr; // [nh_loc * 512 * 256] (on-the-fly MXFP8 dequant for M > 12)

    float* split_m = nullptr; // [max_dec_M, DSA_MAX_SPLITS, nh_loc]
    float* split_l = nullptr; // [max_dec_M, DSA_MAX_SPLITS, nh_loc]
    float* split_o = nullptr; // [max_dec_M, DSA_MAX_SPLITS, nh_loc, 512]

    // Lightning Indexer scratch
    __nv_bfloat16* idx_k   = nullptr; // [max_M, 128]
    __nv_bfloat16* idx_q   = nullptr; // [max_M, 64, 128]
    float*         idx_w   = nullptr; // [max_M, 64] F32
    float*         idx_sc  = nullptr; // [IDX_SCORE_BATCH, max_cap] F32
    int*           sel_tok = nullptr; // [max_M, IDX_TOPK] INT32 (2048)
    int*           sel_cnt = nullptr; // [max_M] INT32 (-1 = dense all [0..pos])
};

// Boot unpack of kv_b_proj [nh_loc, 448, 512] -> w_uk_t [nh_loc, 512, 192] + w_uv [nh_loc, 256, 512].
__global__ void __launch_bounds__(256) k_ernie55_unpack_kv_b(
    const __nv_bfloat16* __restrict__ kv_b_src,
    __nv_bfloat16* __restrict__ w_uk_t,
    __nv_bfloat16* __restrict__ w_uv,
    int nh_loc) {
    const int idx = blockIdx.x;
    if (idx >= nh_loc * geom::KVB_HEAD) return;
    const int h = idx / geom::KVB_HEAD;
    const int r = idx % geom::KVB_HEAD;
    const __nv_bfloat16* row = kv_b_src + static_cast<size_t>(idx) * geom::KV_LORA;
    if (r < geom::QK_NOPE) {
        for (int c = threadIdx.x; c < geom::KV_LORA; c += 256) {
            w_uk_t[(static_cast<size_t>(h) * geom::KV_LORA + c) * geom::QK_NOPE + r] = row[c];
        }
    } else {
        const int vr = r - geom::QK_NOPE;
        __nv_bfloat16* dst = w_uv + (static_cast<size_t>(h) * geom::V_HEAD + vr) * geom::KV_LORA;
        for (int c = threadIdx.x; c < geom::KV_LORA; c += 256) dst[c] = row[c];
    }
}

__global__ void k_ernie55_bf16_to_f32_vec(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Plain DSA RoPE (theta = 8e6, NO YaRN).
__device__ __forceinline__ void dsa_rope_pair(float& x0, float& x1, int pos, int pair_idx) {
    const float inv_freq = powf(geom::DSA_ROPE_THETA, -static_cast<float>(pair_idx * 2) * (1.0f / 64.0f));
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float y0 = x0 * c - x1 * s;
    const float y1 = x1 * c + x0 * s;
    x0 = y0;
    x1 = y1;
}

__global__ void __launch_bounds__(256) k_ernie55_dsa_q_split_rope(
    const __nv_bfloat16* __restrict__ q_raw,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_nope_h,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int d = threadIdx.x;
    const __nv_bfloat16* src = q_raw + static_cast<size_t>(idx) * geom::QK_HEAD;
    if (d < geom::QK_NOPE) {
        q_nope_h[(static_cast<size_t>(h) * M + m) * geom::QK_NOPE + d] = src[d];
    } else if (d < geom::QK_NOPE + geom::QK_ROPE / 2) {
        const int p = d - geom::QK_NOPE;
        float x0 = __bfloat162float(src[geom::QK_NOPE + 2 * p]);
        float x1 = __bfloat162float(src[geom::QK_NOPE + 2 * p + 1]);
        dsa_rope_pair(x0, x1, d_pos[m], p);
        __nv_bfloat16* dst = q_full + static_cast<size_t>(idx) * geom::KVA + geom::KV_LORA;
        dst[2 * p]     = __float2bfloat16(x0);
        dst[2 * p + 1] = __float2bfloat16(x1);
    }
}

__global__ void __launch_bounds__(256) k_ernie55_dsa_q_abs_scatter(
    const __nv_bfloat16* __restrict__ q_abs_h,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const uint2* src = reinterpret_cast<const uint2*>(q_abs_h + (static_cast<size_t>(h) * M + m) * geom::KV_LORA);
    uint2* dst = reinterpret_cast<uint2*>(q_full + static_cast<size_t>(idx) * geom::KVA);
    if (threadIdx.x < 128) dst[threadIdx.x] = src[threadIdx.x];
}

// Fused Small-M Absorbed Q + RoPE (BF16 and MXFP8)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ernie55_dsa_q_absorb_rope_bf16(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ w_uk_t,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh) {
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.y * 4 + warp;
    if (h >= nh || c >= geom::KV_LORA) return;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if (lane < 24) {
        const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
            w_uk_t + (static_cast<size_t>(h) * geom::KV_LORA + c) * geom::QK_NOPE) + lane);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(
                    q_raw + (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD)[lane];
                acc[m] = dot_bf16x8(av, bv);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = warp_sum(acc[m]);
            if (lane == 0) q_full[(static_cast<size_t>(m) * nh + h) * geom::KVA + c] = __float2bfloat16(s);
        }
    }
    if (blockIdx.y == 0 && warp == 0) {
        const float inv_freq = powf(geom::DSA_ROPE_THETA, -static_cast<float>(lane * 2) * (1.0f / 64.0f));
        #pragma unroll 1
        for (int m = 0; m < M; ++m) {
            const size_t q_off = (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD + geom::QK_NOPE;
            const size_t d_off = (static_cast<size_t>(m) * nh + h) * geom::KVA + geom::KV_LORA;
            const float x0 = __bfloat162float(q_raw[q_off + 2 * lane]);
            const float x1 = __bfloat162float(q_raw[q_off + 2 * lane + 1]);
            float s, cv;
            sincosf(static_cast<float>(d_pos[m]) * inv_freq, &s, &cv);
            q_full[d_off + 2 * lane]     = __float2bfloat16(x0 * cv - x1 * s);
            q_full[d_off + 2 * lane + 1] = __float2bfloat16(x1 * cv + x0 * s);
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_ernie55_dsa_q_absorb_rope_mx8(
    const __nv_bfloat16* __restrict__ q_raw,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wsc,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh) {
    const int h = blockIdx.x;
    const int hwarp = threadIdx.x >> 4;
    const int hlane = threadIdx.x & 15;
    const int c0 = blockIdx.y * 16 + hwarp * 2;
    if (h >= nh || c0 >= geom::KV_LORA) return;

    float acc0[MAX_M], acc1[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    if (hlane < 12) {
        const size_t r0 = static_cast<size_t>(h) * geom::KV_LORA + c0;
        const size_t r1 = r0 + 1;
        // Issue 2x concurrent uint4 weight loads + scales per lane, read q_raw ONCE for both rows!
        const uint4 wv0 = __ldg(reinterpret_cast<const uint4*>(wq + r0 * geom::QK_NOPE) + hlane);
        const uint4 wv1 = __ldg(reinterpret_cast<const uint4*>(wq + r1 * geom::QK_NOPE) + hlane);
        const float sc0 = __uint_as_float(static_cast<uint32_t>(__ldg(wsc + r0 * 6 + (hlane >> 1))) << 23);
        const float sc1 = __uint_as_float(static_cast<uint32_t>(__ldg(wsc + r1 * 6 + (hlane >> 1))) << 23);
        float2 wf0[8], wf1[8];
        mx8::fp8x4_unpack(wv0.x, wf0[0], wf0[1]); mx8::fp8x4_unpack(wv0.y, wf0[2], wf0[3]);
        mx8::fp8x4_unpack(wv0.z, wf0[4], wf0[5]); mx8::fp8x4_unpack(wv0.w, wf0[6], wf0[7]);
        mx8::fp8x4_unpack(wv1.x, wf1[0], wf1[1]); mx8::fp8x4_unpack(wv1.y, wf1[2], wf1[3]);
        mx8::fp8x4_unpack(wv1.z, wf1[4], wf1[5]); mx8::fp8x4_unpack(wv1.w, wf1[6], wf1[7]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* qm = reinterpret_cast<const uint4*>(
                    q_raw + (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD);
                const uint4 q0 = __ldg(qm + 2 * hlane);
                const uint4 q1 = __ldg(qm + 2 * hlane + 1);
                acc0[m] = sc0 * mx8::dot_bf16x16(q0, q1, wf0);
                acc1[m] = sc1 * mx8::dot_bf16x16(q0, q1, wf1);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s0 = mx8::h16_sum(acc0[m]);
            const float s1 = mx8::h16_sum(acc1[m]);
            if (hlane == 0) {
                __nv_bfloat16* dst = q_full + (static_cast<size_t>(m) * nh + h) * geom::KVA + c0;
                dst[0] = __float2bfloat16(s0);
                dst[1] = __float2bfloat16(s1);
            }
        }
    }
    if (blockIdx.y == 0 && threadIdx.x < 32) {
        const int p = threadIdx.x;
        const float inv_freq = powf(geom::DSA_ROPE_THETA, -static_cast<float>(p * 2) * (1.0f / 64.0f));
        #pragma unroll 1
        for (int m = 0; m < M; ++m) {
            const size_t q_off = (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD + geom::QK_NOPE;
            const size_t d_off = (static_cast<size_t>(m) * nh + h) * geom::KVA + geom::KV_LORA;
            const float x0 = __bfloat162float(__ldg(q_raw + q_off + 2 * p));
            const float x1 = __bfloat162float(__ldg(q_raw + q_off + 2 * p + 1));
            float s, cv;
            sincosf(static_cast<float>(d_pos[m]) * inv_freq, &s, &cv);
            q_full[d_off + 2 * p]     = __float2bfloat16(x0 * cv - x1 * s);
            q_full[d_off + 2 * p + 1] = __float2bfloat16(x1 * cv + x0 * s);
        }
    }
}

// =================================================================================================
// DSA Paged KV Write (BF16 / FP8-E4M3 / INT4-KV4)
// Grid: (M), Block: 256.
// =================================================================================================
template <KvFmt FMT>
__global__ void __launch_bounds__(256) k_ernie55_dsa_kv_write(
    const __nv_bfloat16* __restrict__ kv_raw,
    const __nv_bfloat16* __restrict__ kv_a_ln,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    uint8_t* __restrict__ kv_dst,
    int M, int max_pps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const __nv_bfloat16* src = kv_raw + static_cast<size_t>(m) * geom::KVA;

    const float v0 = __bfloat162float(src[tid]);
    const float v1 = __bfloat162float(src[tid + 256]);
    float ss = warp_sum(v0 * v0 + v1 * v1);
    __shared__ float sm[8];
    if (lane == 0) sm[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < 8) ? sm[tid] : 0.0f;
        tot = warp_sum(tot);
        if (tid == 0) sm[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::KV_LORA)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];

    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    const int lpage = pos / geom::KV_PAGE;
    const int off   = pos % geom::KV_PAGE;
    const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
    constexpr size_t STRIDE = (FMT == KvFmt::FP8) ? KV_FP8_STRIDE
                            : (FMT == KvFmt::INT4) ? KV_INT4_STRIDE
                            : (geom::KVA * sizeof(__nv_bfloat16));
    uint8_t* out_u8 = kv_dst + (static_cast<size_t>(ppage) * geom::KV_PAGE + off) * STRIDE;

    __shared__ float s_nope[512];
    const float qn0 = __bfloat162float(__float2bfloat16(v0 * inv * __bfloat162float(kv_a_ln[tid])));
    const float qn1 = __bfloat162float(__float2bfloat16(v1 * inv * __bfloat162float(kv_a_ln[tid + 256])));

    if constexpr (FMT == KvFmt::BF16) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_u8);
        out[tid]       = __float2bfloat16(qn0);
        out[tid + 256] = __float2bfloat16(qn1);
    } else {
        s_nope[tid]       = qn0;
        s_nope[tid + 256] = qn1;
        __syncthreads();
        const int warp = tid >> 5;
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int blk = pass * 8 + warp;
            const float val = s_nope[blk * 32 + lane];
            const float amax = warp_max(fabsf(val));
            if constexpr (FMT == KvFmt::FP8) {
                const float sc = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
                const float inv_sc = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
                out_u8[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val * inv_sc, __NV_SATFINITE, __NV_E4M3));
                if (lane == 0) reinterpret_cast<__half*>(out_u8 + 512)[blk] = __float2half_rn(sc);
            } else {
                const float sc = (amax > 0.0f) ? (amax * (1.0f / 7.0f)) : 1.0f;
                const float inv_sc = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
                const int qi = max(-7, min(7, __float2int_rn(val * inv_sc))) + 8;
                const int q_hi = __shfl_down_sync(0xffffffffu, qi, 1);
                if ((lane & 1) == 0) {
                    out_u8[blk * 16 + (lane >> 1)] = static_cast<uint8_t>((qi & 0xf) | ((q_hi & 0xf) << 4));
                }
                if (lane == 0) reinterpret_cast<__half*>(out_u8 + 256)[blk] = __float2half_rn(sc);
            }
        }
    }

    if (tid < 32) {
        float x0 = __bfloat162float(src[geom::KV_LORA + 2 * tid]);
        float x1 = __bfloat162float(src[geom::KV_LORA + 2 * tid + 1]);
        dsa_rope_pair(x0, x1, pos, tid);
        constexpr size_t ROPE_OFF = (FMT == KvFmt::BF16) ? (geom::KV_LORA * 2)
                                  : (FMT == KvFmt::FP8) ? (512 + 32) : (256 + 32);
        __nv_bfloat16* rdst = reinterpret_cast<__nv_bfloat16*>(out_u8 + ROPE_OFF);
        rdst[2 * tid]     = __float2bfloat16(x0);
        rdst[2 * tid + 1] = __float2bfloat16(x1);
    }
}

// =================================================================================================
// Lightning Indexer: LayerNorm + NeoX Partial RoPE (leading 64 of 128) + Hadamard-128 + TC Scorer + Top-2048
// =================================================================================================

// Indexer Q: NeoX partial RoPE on [0..63] + Hadamard-128 in-place.
// Grid: (M * 64), Block: 32 (1 warp = 1 head of 128 dims!).
__global__ void __launch_bounds__(32) k_ernie55_idx_q_rope_had(
    __nv_bfloat16* __restrict__ idx_q,
    const int* __restrict__ d_pos, int M) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::IDX_HEADS) return;
    const int m = idx / geom::IDX_HEADS;
    const int lane = threadIdx.x; // 0..31
    const int pos = d_pos[m];
    __nv_bfloat16* q = idx_q + static_cast<size_t>(idx) * geom::IDX_DIM;

    __shared__ float s[128];
    // NeoX split-half RoPE on leading 64 dims: pairs (lane, lane + 32)
    float x0 = __bfloat162float(q[lane]);
    float x1 = __bfloat162float(q[lane + 32]);
    dsa_rope_pair(x0, x1, pos, lane);
    s[lane]      = __bfloat162float(__float2bfloat16(x0));
    s[lane + 32] = __bfloat162float(__float2bfloat16(x1));
    s[lane + 64] = __bfloat162float(q[lane + 64]);
    s[lane + 96] = __bfloat162float(q[lane + 96]);
    __syncwarp();

    float v[4] = { s[lane * 4 + 0], s[lane * 4 + 1], s[lane * 4 + 2], s[lane * 4 + 3] };
    had128(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; ++e) q[lane * 4 + e] = __float2bfloat16(v[e]);
}

// Indexer K: LayerNorm(k, w, b, 1e-6) + NeoX partial RoPE on [0..63] + Hadamard-128 -> write paged idx pool.
// Grid: (M), Block: 32 (1 warp = 128 dims!).
template <bool IDX_FP8>
__global__ void __launch_bounds__(32) k_ernie55_idx_k_write(
    const __nv_bfloat16* __restrict__ idx_k,
    const __nv_bfloat16* __restrict__ ln_w,
    const __nv_bfloat16* __restrict__ ln_b,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    __nv_bfloat16* __restrict__ idx_bf16,
    uint8_t* __restrict__ idx_q,
    __half* __restrict__ idx_sc,
    int M, int max_pps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int lane = threadIdx.x; // 0..31
    const __nv_bfloat16* src = idx_k + static_cast<size_t>(m) * geom::IDX_DIM;

    float vals[4];
    float sum = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        vals[e] = __bfloat162float(src[lane * 4 + e]);
        sum += vals[e];
    }
    const float mean = warp_sum(sum) * (1.0f / 128.0f);
    float var = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float diff = vals[e] - mean;
        var = fmaf(diff, diff, var);
    }
    const float inv = rsqrtf(warp_sum(var) * (1.0f / 128.0f) + geom::RMS_EPS);

    __shared__ float s[128];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int c = lane * 4 + e;
        const float y = (vals[e] - mean) * inv * __bfloat162float(ln_w[c]) + __bfloat162float(ln_b[c]);
        s[c] = __bfloat162float(__float2bfloat16(y));
    }
    __syncwarp();

    const int pos = d_pos[m];
    float x0 = s[lane];
    float x1 = s[lane + 32];
    dsa_rope_pair(x0, x1, pos, lane);
    s[lane]      = __bfloat162float(__float2bfloat16(x0));
    s[lane + 32] = __bfloat162float(__float2bfloat16(x1));
    __syncwarp();

    float v[4] = { s[lane * 4 + 0], s[lane * 4 + 1], s[lane * 4 + 2], s[lane * 4 + 3] };
    had128(v, lane);

    const int slot  = d_slot[m];
    const int lpage = pos / geom::KV_PAGE;
    const int off   = pos % geom::KV_PAGE;
    const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
    const size_t to = static_cast<size_t>(ppage) * geom::KV_PAGE + off;

    if constexpr (!IDX_FP8) {
        __nv_bfloat16* dst = idx_bf16 + to * geom::IDX_DIM;
        #pragma unroll
        for (int e = 0; e < 4; ++e) dst[lane * 4 + e] = __float2bfloat16(v[e]);
    } else {
        // Store to s[] so 4 blocks of 32 dims can be quantized cleanly
        #pragma unroll
        for (int e = 0; e < 4; ++e) s[lane * 4 + e] = __bfloat162float(__float2bfloat16(v[e]));
        __syncwarp();
        uint8_t* dq = idx_q + to * geom::IDX_DIM;
        __half*  ds = idx_sc + to * 4;
        #pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            const float qv = s[blk * 32 + lane];
            const float amax = warp_max(fabsf(qv));
            const float sc = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
            const float inv_sc = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
            dq[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(qv * inv_sc, __NV_SATFINITE, __NV_E4M3));
            if (lane == 0) ds[blk] = __float2half_rn(sc);
        }
    }
}

// Tensor-Core Lightning Indexer Scorer (64 indexer heads, 32 tokens per CTA):
// Loops over 4 head-groups of 16 heads (h0 = 0, 16, 32, 48), scoring 8 tokens/warp x 4 warps = 32 tokens/CTA!
// Grid: dim3((max_old + 31) / 32, n_m), Block: 128 (4 warps).
template <bool IDX_FP8>
__global__ void __launch_bounds__(128) k_ernie55_idx_score_tc(
    const __nv_bfloat16* __restrict__ idx_q,
    const float* __restrict__ idx_w,
    const __nv_bfloat16* __restrict__ idx_bf16,
    const uint8_t* __restrict__ idx_pool_q,
    const __half* __restrict__ idx_pool_sc,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    float* __restrict__ idx_sc,
    int m_off, int n_m, int max_pps, int max_cap) {
    const int lm = blockIdx.y;
    if (lm >= n_m) return;
    const int m = m_off + lm;
    const int pos = d_pos[m];
    const int n_old = pos + 1 - geom::SLIDING_WIN;
    if (n_old <= geom::IDX_TOPK) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t0 = (blockIdx.x * 4 + warp) * 8;
    if (t0 >= n_old) return;

    const int rg = lane >> 2;
    const int tc = lane & 3;
    const int slot = d_slot[m];
    const int tk = t0 + rg;
    const bool valid_k = (tk < n_old);

    size_t to = 0;
    if (valid_k) {
        const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + (tk / geom::KV_PAGE)];
        to = static_cast<size_t>(ppage) * geom::KV_PAGE + (tk % geom::KV_PAGE);
    }

    // Preload K[8, 128] for this warp's 8 tokens (reused across all 4 head-groups of 16!)
    uint32_t kb0[8], kb1[8];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int kbase = ks * 16;
        if (valid_k) {
            if constexpr (!IDX_FP8) {
                const __nv_bfloat16* kp = idx_bf16 + to * geom::IDX_DIM + kbase;
                kb0[ks] = __ldg(reinterpret_cast<const uint32_t*>(kp + 2 * tc));
                kb1[ks] = __ldg(reinterpret_cast<const uint32_t*>(kp + 2 * tc + 8));
            } else {
                const float sc = __half2float(__ldg(idx_pool_sc + to * 4 + (ks >> 1)));
                const uint8_t* qp = idx_pool_q + to * geom::IDX_DIM + kbase;
                const uint16_t raw0 = __ldg(reinterpret_cast<const uint16_t*>(qp + 2 * tc));
                const uint16_t raw1 = __ldg(reinterpret_cast<const uint16_t*>(qp + 2 * tc + 8));
                uint32_t h0, h1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(raw0));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(raw1));
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
                kb0[ks] = pack_bf2(f0.x * sc, f0.y * sc);
                kb1[ks] = pack_bf2(f1.x * sc, f1.y * sc);
            }
        } else {
            kb0[ks] = 0u;
            kb1[ks] = 0u;
        }
    }

    constexpr float IW_SCALE = 0.011048543456039806f; // (128 * 64)^-0.5
    const __nv_bfloat16* qm = idx_q + static_cast<size_t>(m) * geom::IDX_HEADS * geom::IDX_DIM;
    const float* iw = idx_w + static_cast<size_t>(m) * geom::IDX_HEADS;

    float s0 = 0.0f, s1 = 0.0f;
    #pragma unroll
    for (int hg = 0; hg < 4; ++hg) {
        const int hbase = hg * 16;
        const float w0 = iw[hbase + rg]     * IW_SCALE;
        const float w8 = iw[hbase + rg + 8] * IW_SCALE;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int kbase = ks * 16;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(qm + (hbase + rg) * 128     + kbase + 2 * tc);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(qm + (hbase + rg + 8) * 128 + kbase + 2 * tc);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(qm + (hbase + rg) * 128     + kbase + 2 * tc + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(qm + (hbase + rg + 8) * 128 + kbase + 2 * tc + 8);
            mma16816_bf16(d, a0, a1, a2, a3, kb0[ks], kb1[ks]);
        }
        s0 = fmaf(w0, fmaxf(d[0], 0.0f), fmaf(w8, fmaxf(d[2], 0.0f), s0));
        s1 = fmaf(w0, fmaxf(d[1], 0.0f), fmaf(w8, fmaxf(d[3], 0.0f), s1));
    }

    s0 += __shfl_xor_sync(0xffffffffu, s0, 4);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
    if (rg == 0) {
        const int t_even = t0 + 2 * tc;
        if (t_even < n_old)     idx_sc[static_cast<size_t>(lm) * max_cap + t_even]     = s0;
        if (t_even + 1 < n_old) idx_sc[static_cast<size_t>(lm) * max_cap + t_even + 1] = s1;
    }
}

__device__ __forceinline__ uint32_t dsa_score_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exact 3-pass radix top-2048 selector over [0 .. n_old - 1] (preserving ascending token order for ties).
// Grid: (n_m), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_idx_topk2048(
    const float* __restrict__ idx_sc,
    const int* __restrict__ d_pos,
    int* __restrict__ sel_tok,
    int* __restrict__ sel_cnt,
    int m_off, int n_m, int max_cap) {
    const int lm = blockIdx.x;
    if (lm >= n_m) return;
    const int m = m_off + lm;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int pos = d_pos[m];
    const int n_old = pos + 1 - geom::SLIDING_WIN;
    constexpr int KSEL = geom::IDX_TOPK; // 2048

    if (n_old <= KSEL) {
        if (tid == 0) sel_cnt[m] = -1;
        return;
    }

    int* dst = sel_tok + static_cast<size_t>(m) * KSEL;
    const float* sc = idx_sc + static_cast<size_t>(lm) * max_cap;
    __shared__ int hist[2048];
    __shared__ int w_tot[8];
    __shared__ int s_p1, s_need1, s_cnt1, s_p2, s_need2, s_cnt2, s_eq_need, s_out_pos, s_eq_used;
    __shared__ uint32_t s_strict_thr, s_eq_thr;

    for (int i = tid; i < 2048; i += 256) hist[i] = 0;
    __syncthreads();
    for (int b = tid; b < n_old; b += 256) atomicAdd(&hist[(dsa_score_key(sc[b]) >> 21) & 2047], 1);
    __syncthreads();
    {
        int my_sum = 0;
        const int b_hi = 2047 - tid * 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) my_sum += hist[b_hi - k];
        int ws = my_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, ws, off);
            if (lane >= off) ws += n;
        }
        if (lane == 31) w_tot[warp] = ws;
        __syncthreads();
        if (tid == 0) {
            int run = 0, tw = 7;
            for (int w = 0; w < 8; ++w) { if (run + w_tot[w] >= KSEL) { tw = w; break; } run += w_tot[w]; }
            int p1 = 0;
            for (int b = 2047 - tw * 256; b >= 0; --b) { if (run + hist[b] >= KSEL) { p1 = b; break; } run += hist[b]; }
            s_p1 = p1; s_need1 = KSEL - run; s_cnt1 = hist[p1];
            if (s_cnt1 == s_need1) { s_strict_thr = static_cast<uint32_t>(p1) << 21; s_eq_thr = 0u; s_eq_need = 0; }
        }
    }
    __syncthreads();

    if (s_cnt1 > s_need1) {
        const uint32_t p1 = static_cast<uint32_t>(s_p1);
        for (int i = tid; i < 2048; i += 256) hist[i] = 0;
        __syncthreads();
        for (int b = tid; b < n_old; b += 256) {
            const uint32_t u = dsa_score_key(sc[b]);
            if ((u >> 21) == p1) atomicAdd(&hist[(u >> 10) & 2047], 1);
        }
        __syncthreads();
        int my_sum = 0;
        const int b_hi = 2047 - tid * 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) my_sum += hist[b_hi - k];
        int ws = my_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, ws, off);
            if (lane >= off) ws += n;
        }
        if (lane == 31) w_tot[warp] = ws;
        __syncthreads();
        if (tid == 0) {
            int run = 0, tw = 7;
            for (int w = 0; w < 8; ++w) { if (run + w_tot[w] >= s_need1) { tw = w; break; } run += w_tot[w]; }
            int p2 = 0;
            for (int b = 2047 - tw * 256; b >= 0; --b) { if (run + hist[b] >= s_need1) { p2 = b; break; } run += hist[b]; }
            s_p2 = p2; s_need2 = s_need1 - run; s_cnt2 = hist[p2];
            if (s_cnt2 == s_need2) {
                s_strict_thr = (p1 << 21) | (static_cast<uint32_t>(p2) << 10);
                s_eq_thr = 0u; s_eq_need = 0;
            }
        }
        __syncthreads();

        if (s_cnt2 > s_need2) {
            const uint32_t p12 = (p1 << 11) | static_cast<uint32_t>(s_p2);
            for (int i = tid; i < 1024; i += 256) hist[i] = 0;
            __syncthreads();
            for (int b = tid; b < n_old; b += 256) {
                const uint32_t u = dsa_score_key(sc[b]);
                if ((u >> 10) == p12) atomicAdd(&hist[u & 1023], 1);
            }
            __syncthreads();
            int my_sum3 = 0;
            const int b3_hi = 1023 - tid * 4;
            #pragma unroll
            for (int k = 0; k < 4; ++k) my_sum3 += hist[b3_hi - k];
            int ws3 = my_sum3;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, ws3, off);
                if (lane >= off) ws3 += n;
            }
            if (lane == 31) w_tot[warp] = ws3;
            __syncthreads();
            if (tid == 0) {
                int run = 0, tw = 7;
                for (int w = 0; w < 8; ++w) { if (run + w_tot[w] >= s_need2) { tw = w; break; } run += w_tot[w]; }
                int p3 = 0;
                for (int b = 1023 - tw * 128; b >= 0; --b) { if (run + hist[b] >= s_need2) { p3 = b; break; } run += hist[b]; }
                const int need3 = s_need2 - run;
                const uint32_t thr_u = (p12 << 10) | static_cast<uint32_t>(p3);
                if (hist[p3] == need3) { s_strict_thr = thr_u; s_eq_thr = 0u; s_eq_need = 0; }
                else { s_strict_thr = thr_u + 1u; s_eq_thr = thr_u; s_eq_need = need3; }
            }
            __syncthreads();
        }
    }

    const uint32_t strict_thr = s_strict_thr;
    const uint32_t eq_thr = s_eq_thr;
    const int eq_need = s_eq_need;
    const uint32_t lane_mask = (1u << lane) - 1u;
    if (tid == 0) { s_out_pos = 0; s_eq_used = 0; }
    __syncthreads();

    for (int base = 0; base < n_old; base += 256) {
        if (s_out_pos >= KSEL) break;
        const int b = base + tid;
        const bool valid = (b < n_old);
        const uint32_t u = valid ? dsa_score_key(sc[b]) : 0u;
        const bool is_gt = valid && (u >= strict_thr);
        const bool is_eq = valid && (eq_need > 0) && (u == eq_thr);
        bool take_eq = false;
        if (__syncthreads_or(is_eq)) {
            const uint32_t m_eq = __ballot_sync(0xffffffffu, is_eq);
            const int lp = __popc(m_eq & lane_mask);
            if (lane == 31) w_tot[warp] = __popc(m_eq);
            __syncthreads();
            int w_off = 0, c_tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { if (w < warp) w_off += w_tot[w]; c_tot += w_tot[w]; }
            if (is_eq && (s_eq_used + w_off + lp < eq_need)) take_eq = true;
            __syncthreads();
            if (tid == 0) s_eq_used += c_tot;
            __syncthreads();
        }
        const bool take = is_gt || take_eq;
        if (__syncthreads_or(take)) {
            const uint32_t m_tk = __ballot_sync(0xffffffffu, take);
            const int lp = __popc(m_tk & lane_mask);
            if (lane == 31) w_tot[warp] = __popc(m_tk);
            __syncthreads();
            int w_off = 0, c_tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { if (w < warp) w_off += w_tot[w]; c_tot += w_tot[w]; }
            if (take) {
                const int p = s_out_pos + w_off + lp;
                if (p < KSEL) dst[p] = b;
            }
            __syncthreads();
            if (tid == 0) s_out_pos += c_tot;
            __syncthreads();
        }
    }
    if (tid == 0) sel_cnt[m] = min(s_out_pos, KSEL);
}

// =================================================================================================
// Tensor-Core Absorbed MLA-DSA Decode / Sparse / Prefill (BF16 / FP8 / INT4 KV)
// =================================================================================================
template <KvFmt FMT>
__device__ __forceinline__ void dsa_load_kv_tile(
    __nv_bfloat16 (*sKV)[584],
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_page_table,
    int slot, int max_pps,
    const int* __restrict__ sel, int nsel, int swa_start,
    int k0, int nk, int tid) {
    auto tok_pos = [&](int ki) -> int {
        if (nsel < 0) return ki;
        if (ki < nsel) return sel[ki];
        return swa_start + (ki - nsel);
    };
    if constexpr (FMT == KvFmt::BF16) {
        const int total_u4 = nk * 72;
        for (int idx4 = tid; idx4 < total_u4; idx4 += 256) {
            const int tk = idx4 / 72;
            const int u4 = idx4 - tk * 72;
            const int tpos = tok_pos(k0 + tk);
            const int ppage = __ldg(d_page_table + static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE));
            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(kv_mem) +
                                       (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * geom::KVA;
            *reinterpret_cast<uint4*>(sKV[tk] + u4 * 8) = __ldg(reinterpret_cast<const uint4*>(src) + u4);
        }
    } else if constexpr (FMT == KvFmt::FP8) {
        const int warp = tid >> 5, lane = tid & 31;
        for (int tk = warp; tk < nk; tk += 8) {
            const int tpos = tok_pos(k0 + tk);
            const int ppage = __ldg(d_page_table + static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE));
            const uint8_t* src = kv_mem + (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * KV_FP8_STRIDE;
            const __half* scs = reinterpret_cast<const __half*>(src + 512);
            const uint4 qv = __ldg(reinterpret_cast<const uint4*>(src) + lane);
            const float sc = __half2float(__ldg(scs + (lane >> 1)));
            float2 f[8];
            mx8::fp8x4_unpack(qv.x, f[0], f[1]);
            mx8::fp8x4_unpack(qv.y, f[2], f[3]);
            mx8::fp8x4_unpack(qv.z, f[4], f[5]);
            mx8::fp8x4_unpack(qv.w, f[6], f[7]);
            uint4 o0, o1;
            o0.x = pack_bf2(f[0].x * sc, f[0].y * sc); o0.y = pack_bf2(f[1].x * sc, f[1].y * sc);
            o0.z = pack_bf2(f[2].x * sc, f[2].y * sc); o0.w = pack_bf2(f[3].x * sc, f[3].y * sc);
            o1.x = pack_bf2(f[4].x * sc, f[4].y * sc); o1.y = pack_bf2(f[5].x * sc, f[5].y * sc);
            o1.z = pack_bf2(f[6].x * sc, f[6].y * sc); o1.w = pack_bf2(f[7].x * sc, f[7].y * sc);
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16)     = o0;
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16 + 8) = o1;
            *reinterpret_cast<uint32_t*>(sKV[tk] + 512 + lane * 2) = __ldg(reinterpret_cast<const uint32_t*>(src + 544) + lane);
        }
    } else { // KvFmt::INT4
        const int warp = tid >> 5, lane = tid & 31;
        for (int tk = warp; tk < nk; tk += 8) {
            const int tpos = tok_pos(k0 + tk);
            const int ppage = __ldg(d_page_table + static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE));
            const uint8_t* src = kv_mem + (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * KV_INT4_STRIDE;
            const __half* scs = reinterpret_cast<const __half*>(src + 256);
            const uint2 raw = __ldg(reinterpret_cast<const uint2*>(src) + lane);
            const float sc = __half2float(__ldg(scs + (lane >> 1)));
            auto unpack_u32 = [&](uint32_t w, uint4& dst) {
                uint32_t out[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b = (w >> (8 * i)) & 0xffu;
                    out[i] = pack_bf2(static_cast<float>(static_cast<int>(b & 0xfu) - 8) * sc,
                                      static_cast<float>(static_cast<int>(b >> 4) - 8) * sc);
                }
                dst = make_uint4(out[0], out[1], out[2], out[3]);
            };
            uint4 o0, o1;
            unpack_u32(raw.x, o0);
            unpack_u32(raw.y, o1);
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16)     = o0;
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16 + 8) = o1;
            *reinterpret_cast<uint32_t*>(sKV[tk] + 512 + lane * 2) = reinterpret_cast<const uint32_t*>(src + 288)[lane];
        }
    }
}

template <KvFmt FMT>
__global__ void __launch_bounds__(256) k_ernie55_dsa_mla_decode_tc(
    const __nv_bfloat16* __restrict__ q_full,
    const float* __restrict__ sinks,
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    const int* __restrict__ sel_tok,
    const int* __restrict__ sel_cnt,
    __nv_bfloat16* __restrict__ u_h,
    float* __restrict__ split_m,
    float* __restrict__ split_l,
    float* __restrict__ split_o,
    int M, int nh, int max_pps, int NS) {
    const int n_hg = nh >> 4;
    const int idx = blockIdx.x;
    const int sp = idx % NS;
    const int rem = idx / NS;
    const int hg = rem % n_hg;
    const int m  = rem / n_hg;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg = lane >> 2;
    const int tc = lane & 3;
    const int h0 = hg * 16;

    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    const int nsel = sel_cnt ? sel_cnt[m] : -1;
    int total_tok, swa_start = 0;
    const int* sel = nullptr;
    if (nsel < 0) {
        total_tok = pos + 1;
    } else {
        swa_start = max(0, pos + 1 - geom::SLIDING_WIN);
        total_tok = nsel + (pos + 1 - swa_start);
        sel = sel_tok + static_cast<size_t>(m) * geom::IDX_TOPK;
    }

    const int per = (total_tok + NS - 1) / NS;
    const int k_start = sp * per;
    const int k_end   = min(total_tok, k_start + per);

    const __nv_bfloat16* qp_a = q_full + (static_cast<size_t>(m) * nh + h0 + rg) * geom::KVA;
    const __nv_bfloat16* qp_b = q_full + (static_cast<size_t>(m) * nh + h0 + rg + 8) * geom::KVA;
    uint32_t qn[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c = warp * 64 + ks * 16 + 2 * tc;
        qn[ks][0] = *reinterpret_cast<const uint32_t*>(qp_a + c);
        qn[ks][1] = *reinterpret_cast<const uint32_t*>(qp_b + c);
        qn[ks][2] = *reinterpret_cast<const uint32_t*>(qp_a + c + 8);
        qn[ks][3] = *reinterpret_cast<const uint32_t*>(qp_b + c + 8);
    }
    uint32_t qr[4] = {0u, 0u, 0u, 0u};
    if (warp < 4) {
        const int c = geom::KV_LORA + warp * 16 + 2 * tc;
        qr[0] = *reinterpret_cast<const uint32_t*>(qp_a + c);
        qr[1] = *reinterpret_cast<const uint32_t*>(qp_b + c);
        qr[2] = *reinterpret_cast<const uint32_t*>(qp_a + c + 8);
        qr[3] = *reinterpret_cast<const uint32_t*>(qp_b + c + 8);
    }

    float m_a = (sp == 0) ? sinks[h0 + rg]     : -INFINITY;
    float m_b = (sp == 0) ? sinks[h0 + rg + 8] : -INFINITY;
    float l_a = (sp == 0) ? 1.0f : 0.0f;
    float l_b = (sp == 0) ? 1.0f : 0.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 (*sKV)[584] = reinterpret_cast<__nv_bfloat16 (*)[584]>(sm_raw);
    float (*sPart)[16][32] = reinterpret_cast<float (*)[16][32]>(sm_raw + 32 * 584 * 2);

    constexpr int TILE_K = 32;
    constexpr float ATTN_SCALE = 0.0625f; // 256^-0.5

    for (int k0 = k_start; k0 < k_end; k0 += TILE_K) {
        const int nk = min(TILE_K, k_end - k0);
        dsa_load_kv_tile<FMT>(sKV, kv_mem, d_page_table, slot, max_pps, sel, nsel, swa_start, k0, nk, tid);
        __syncthreads();

        float s_loc[4][4];
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) s_loc[kt][0] = s_loc[kt][1] = s_loc[kt][2] = s_loc[kt][3] = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int tk = kt * 8 + rg;
            const bool valid_tk = (tk < nk);
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int c = warp * 64 + ks * 16 + 2 * tc;
                const uint32_t b0 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c)     : 0u;
                const uint32_t b1 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c + 8) : 0u;
                mma16816_bf16(s_loc[kt], qn[ks][0], qn[ks][1], qn[ks][2], qn[ks][3], b0, b1);
            }
            if (warp < 4) {
                const int c = geom::KV_LORA + warp * 16 + 2 * tc;
                const uint32_t b0 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c)     : 0u;
                const uint32_t b1 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c + 8) : 0u;
                mma16816_bf16(s_loc[kt], qr[0], qr[1], qr[2], qr[3], b0, b1);
            }
            sPart[warp][rg][kt * 8 + 2 * tc]         = s_loc[kt][0];
            sPart[warp][rg][kt * 8 + 2 * tc + 1]     = s_loc[kt][1];
            sPart[warp][rg + 8][kt * 8 + 2 * tc]     = s_loc[kt][2];
            sPart[warp][rg + 8][kt * 8 + 2 * tc + 1] = s_loc[kt][3];
        }
        __syncthreads();

        float sc_a[8], sc_b[8];
        float tmax_a = -INFINITY, tmax_b = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int c0 = kt * 8 + 2 * tc, c1 = c0 + 1;
            float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sa0 += sPart[w][rg][c0];     sa1 += sPart[w][rg][c1];
                sb0 += sPart[w][rg + 8][c0]; sb1 += sPart[w][rg + 8][c1];
            }
            sc_a[2 * kt]     = (c0 < nk) ? (sa0 * ATTN_SCALE) : -INFINITY;
            sc_a[2 * kt + 1] = (c1 < nk) ? (sa1 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt]     = (c0 < nk) ? (sb0 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt + 1] = (c1 < nk) ? (sb1 * ATTN_SCALE) : -INFINITY;
            tmax_a = fmaxf(tmax_a, fmaxf(sc_a[2 * kt], sc_a[2 * kt + 1]));
            tmax_b = fmaxf(tmax_b, fmaxf(sc_b[2 * kt], sc_b[2 * kt + 1]));
        }
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 1));
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 2));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 1));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 2));

        const float m_new_a = fmaxf(m_a, tmax_a);
        const float m_new_b = fmaxf(m_b, tmax_b);
        const float alpha_a = (m_a > -1e30f) ? __expf(m_a - m_new_a) : 0.0f;
        const float alpha_b = (m_b > -1e30f) ? __expf(m_b - m_new_b) : 0.0f;

        float l_tile_a = 0.0f, l_tile_b = 0.0f;
        float pa[8], pb[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa[i] = (sc_a[i] > -1e30f) ? __expf(sc_a[i] - m_new_a) : 0.0f;
            pb[i] = (sc_b[i] > -1e30f) ? __expf(sc_b[i] - m_new_b) : 0.0f;
            l_tile_a += pa[i]; l_tile_b += pb[i];
        }
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 1);
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 2);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 1);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 2);

        l_a = l_a * alpha_a + l_tile_a; l_b = l_b * alpha_b + l_tile_b;
        m_a = m_new_a;                  m_b = m_new_b;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha_a; o_acc[nt][1] *= alpha_a;
            o_acc[nt][2] *= alpha_b; o_acc[nt][3] *= alpha_b;
        }

        const uint32_t pf0[4] = { pack_bf2(pa[0], pa[1]), pack_bf2(pb[0], pb[1]), pack_bf2(pa[2], pa[3]), pack_bf2(pb[2], pb[3]) };
        const uint32_t pf1[4] = { pack_bf2(pa[4], pa[5]), pack_bf2(pb[4], pb[5]), pack_bf2(pa[6], pa[7]), pack_bf2(pb[6], pb[7]) };

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dcol = warp * 64 + nt * 8 + rg;
            const int k00 = 2 * tc, k01 = k00 + 1, k02 = k00 + 8, k03 = k02 + 1;
            const uint32_t vb0 = pack_bf2(k00 < nk ? __bfloat162float(sKV[k00][dcol]) : 0.0f,
                                          k01 < nk ? __bfloat162float(sKV[k01][dcol]) : 0.0f);
            const uint32_t vb1 = pack_bf2(k02 < nk ? __bfloat162float(sKV[k02][dcol]) : 0.0f,
                                          k03 < nk ? __bfloat162float(sKV[k03][dcol]) : 0.0f);
            mma16816_bf16(o_acc[nt], pf0[0], pf0[1], pf0[2], pf0[3], vb0, vb1);
            if (nk > 16) {
                const int k10 = 16 + 2 * tc, k11 = k10 + 1, k12 = k10 + 8, k13 = k12 + 1;
                const uint32_t vb2 = pack_bf2(k10 < nk ? __bfloat162float(sKV[k10][dcol]) : 0.0f,
                                              k11 < nk ? __bfloat162float(sKV[k11][dcol]) : 0.0f);
                const uint32_t vb3 = pack_bf2(k12 < nk ? __bfloat162float(sKV[k12][dcol]) : 0.0f,
                                              k13 < nk ? __bfloat162float(sKV[k13][dcol]) : 0.0f);
                mma16816_bf16(o_acc[nt], pf1[0], pf1[1], pf1[2], pf1[3], vb2, vb3);
            }
        }
        __syncthreads();
    }

    if (NS == 1) {
        const float inv_a = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
        const float inv_b = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
        __nv_bfloat16* da = u_h + (static_cast<size_t>(h0 + rg)     * M + m) * geom::KV_LORA + warp * 64;
        __nv_bfloat16* db = u_h + (static_cast<size_t>(h0 + rg + 8) * M + m) * geom::KV_LORA + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            *reinterpret_cast<uint32_t*>(da + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][0] * inv_a, o_acc[nt][1] * inv_a);
            *reinterpret_cast<uint32_t*>(db + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][2] * inv_b, o_acc[nt][3] * inv_b);
        }
    } else {
        const size_t sia = (static_cast<size_t>(m) * NS + sp) * nh + (h0 + rg);
        const size_t sib = (static_cast<size_t>(m) * NS + sp) * nh + (h0 + rg + 8);
        if (warp == 0 && tc == 0) {
            split_m[sia] = m_a; split_l[sia] = l_a;
            split_m[sib] = m_b; split_l[sib] = l_b;
        }
        float* soa = split_o + sia * geom::KV_LORA + warp * 64;
        float* sob = split_o + sib * geom::KV_LORA + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            soa[nt * 8 + 2 * tc] = o_acc[nt][0]; soa[nt * 8 + 2 * tc + 1] = o_acc[nt][1];
            sob[nt * 8 + 2 * tc] = o_acc[nt][2]; sob[nt * 8 + 2 * tc + 1] = o_acc[nt][3];
        }
    }
}

template <KvFmt FMT>
__global__ void __launch_bounds__(256) k_ernie55_dsa_mla_prefill_tc(
    const __nv_bfloat16* __restrict__ q_full,
    const float* __restrict__ sinks,
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_page_table,
    __nv_bfloat16* __restrict__ u_h,
    int M, int nh, int max_pps, int slot, int row0, int pos0, int seg_len) {
    const int qb = blockIdx.x, h = blockIdx.y;
    const int r_base = qb * 16;
    if (r_base >= seg_len || h >= nh) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rg = lane >> 2, tc = lane & 3;
    const int ra = r_base + rg, rb = ra + 8;
    const bool va = (ra < seg_len), vb = (rb < seg_len);
    const int pa = pos0 + ra, pb = pos0 + rb;
    const int k_end = pos0 + min(seg_len, r_base + 16);

    const __nv_bfloat16* qp_a = q_full + (static_cast<size_t>(row0 + (va ? ra : 0)) * nh + h) * geom::KVA;
    const __nv_bfloat16* qp_b = q_full + (static_cast<size_t>(row0 + (vb ? rb : 0)) * nh + h) * geom::KVA;
    uint32_t qn[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c = warp * 64 + ks * 16 + 2 * tc;
        qn[ks][0] = va ? *reinterpret_cast<const uint32_t*>(qp_a + c)     : 0u;
        qn[ks][1] = vb ? *reinterpret_cast<const uint32_t*>(qp_b + c)     : 0u;
        qn[ks][2] = va ? *reinterpret_cast<const uint32_t*>(qp_a + c + 8) : 0u;
        qn[ks][3] = vb ? *reinterpret_cast<const uint32_t*>(qp_b + c + 8) : 0u;
    }
    uint32_t qr[4] = {0u, 0u, 0u, 0u};
    if (warp < 4) {
        const int c = geom::KV_LORA + warp * 16 + 2 * tc;
        qr[0] = va ? *reinterpret_cast<const uint32_t*>(qp_a + c)     : 0u;
        qr[1] = vb ? *reinterpret_cast<const uint32_t*>(qp_b + c)     : 0u;
        qr[2] = va ? *reinterpret_cast<const uint32_t*>(qp_a + c + 8) : 0u;
        qr[3] = vb ? *reinterpret_cast<const uint32_t*>(qp_b + c + 8) : 0u;
    }

    const float sink_h = sinks[h];
    float m_a = sink_h, m_b = sink_h, l_a = 1.0f, l_b = 1.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 (*sKV)[584] = reinterpret_cast<__nv_bfloat16 (*)[584]>(sm_raw);
    float (*sPart)[16][32] = reinterpret_cast<float (*)[16][32]>(sm_raw + 32 * 584 * 2);

    constexpr int TILE_K = 32;
    constexpr float ATTN_SCALE = 0.0625f;

    for (int k0 = 0; k0 < k_end; k0 += TILE_K) {
        const int nk = min(TILE_K, k_end - k0);
        dsa_load_kv_tile<FMT>(sKV, kv_mem, d_page_table, slot, max_pps, nullptr, -1, 0, k0, nk, tid);
        __syncthreads();

        float s_loc[4][4];
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) s_loc[kt][0] = s_loc[kt][1] = s_loc[kt][2] = s_loc[kt][3] = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int tk = kt * 8 + rg;
            const bool valid_tk = (tk < nk);
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int c = warp * 64 + ks * 16 + 2 * tc;
                const uint32_t b0 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c)     : 0u;
                const uint32_t b1 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c + 8) : 0u;
                mma16816_bf16(s_loc[kt], qn[ks][0], qn[ks][1], qn[ks][2], qn[ks][3], b0, b1);
            }
            if (warp < 4) {
                const int c = geom::KV_LORA + warp * 16 + 2 * tc;
                const uint32_t b0 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c)     : 0u;
                const uint32_t b1 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c + 8) : 0u;
                mma16816_bf16(s_loc[kt], qr[0], qr[1], qr[2], qr[3], b0, b1);
            }
            sPart[warp][rg][kt * 8 + 2 * tc]         = s_loc[kt][0];
            sPart[warp][rg][kt * 8 + 2 * tc + 1]     = s_loc[kt][1];
            sPart[warp][rg + 8][kt * 8 + 2 * tc]     = s_loc[kt][2];
            sPart[warp][rg + 8][kt * 8 + 2 * tc + 1] = s_loc[kt][3];
        }
        __syncthreads();

        float sc_a[8], sc_b[8];
        float tmax_a = -INFINITY, tmax_b = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int c0 = kt * 8 + 2 * tc, c1 = c0 + 1;
            const int kp0 = k0 + c0, kp1 = k0 + c1;
            float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sa0 += sPart[w][rg][c0];     sa1 += sPart[w][rg][c1];
                sb0 += sPart[w][rg + 8][c0]; sb1 += sPart[w][rg + 8][c1];
            }
            sc_a[2 * kt]     = (va && c0 < nk && kp0 <= pa) ? (sa0 * ATTN_SCALE) : -INFINITY;
            sc_a[2 * kt + 1] = (va && c1 < nk && kp1 <= pa) ? (sa1 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt]     = (vb && c0 < nk && kp0 <= pb) ? (sb0 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt + 1] = (vb && c1 < nk && kp1 <= pb) ? (sb1 * ATTN_SCALE) : -INFINITY;
            tmax_a = fmaxf(tmax_a, fmaxf(sc_a[2 * kt], sc_a[2 * kt + 1]));
            tmax_b = fmaxf(tmax_b, fmaxf(sc_b[2 * kt], sc_b[2 * kt + 1]));
        }
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 1));
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 2));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 1));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 2));

        const float m_new_a = fmaxf(m_a, tmax_a);
        const float m_new_b = fmaxf(m_b, tmax_b);
        const float alpha_a = __expf(m_a - m_new_a);
        const float alpha_b = __expf(m_b - m_new_b);
        float l_tile_a = 0.0f, l_tile_b = 0.0f, pa_v[8], pb_v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa_v[i] = (sc_a[i] > -1e30f) ? __expf(sc_a[i] - m_new_a) : 0.0f;
            pb_v[i] = (sc_b[i] > -1e30f) ? __expf(sc_b[i] - m_new_b) : 0.0f;
            l_tile_a += pa_v[i]; l_tile_b += pb_v[i];
        }
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 1);
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 2);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 1);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 2);

        l_a = l_a * alpha_a + l_tile_a; l_b = l_b * alpha_b + l_tile_b;
        m_a = m_new_a;                  m_b = m_new_b;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha_a; o_acc[nt][1] *= alpha_a;
            o_acc[nt][2] *= alpha_b; o_acc[nt][3] *= alpha_b;
        }
        const uint32_t pf0[4] = { pack_bf2(pa_v[0], pa_v[1]), pack_bf2(pb_v[0], pb_v[1]), pack_bf2(pa_v[2], pa_v[3]), pack_bf2(pb_v[2], pb_v[3]) };
        const uint32_t pf1[4] = { pack_bf2(pa_v[4], pa_v[5]), pack_bf2(pb_v[4], pb_v[5]), pack_bf2(pa_v[6], pa_v[7]), pack_bf2(pb_v[6], pb_v[7]) };
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dcol = warp * 64 + nt * 8 + rg;
            const int k00 = 2 * tc, k01 = k00 + 1, k02 = k00 + 8, k03 = k02 + 1;
            const uint32_t vb0 = pack_bf2(k00 < nk ? __bfloat162float(sKV[k00][dcol]) : 0.0f,
                                          k01 < nk ? __bfloat162float(sKV[k01][dcol]) : 0.0f);
            const uint32_t vb1 = pack_bf2(k02 < nk ? __bfloat162float(sKV[k02][dcol]) : 0.0f,
                                          k03 < nk ? __bfloat162float(sKV[k03][dcol]) : 0.0f);
            mma16816_bf16(o_acc[nt], pf0[0], pf0[1], pf0[2], pf0[3], vb0, vb1);
            if (nk > 16) {
                const int k10 = 16 + 2 * tc, k11 = k10 + 1, k12 = k10 + 8, k13 = k12 + 1;
                const uint32_t vb2 = pack_bf2(k10 < nk ? __bfloat162float(sKV[k10][dcol]) : 0.0f,
                                              k11 < nk ? __bfloat162float(sKV[k11][dcol]) : 0.0f);
                const uint32_t vb3 = pack_bf2(k12 < nk ? __bfloat162float(sKV[k12][dcol]) : 0.0f,
                                              k13 < nk ? __bfloat162float(sKV[k13][dcol]) : 0.0f);
                mma16816_bf16(o_acc[nt], pf1[0], pf1[1], pf1[2], pf1[3], vb2, vb3);
            }
        }
        __syncthreads();
    }

    const float inv_a = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
    const float inv_b = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
    if (va) {
        __nv_bfloat16* da = u_h + (static_cast<size_t>(h) * M + row0 + ra) * geom::KV_LORA + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            *reinterpret_cast<uint32_t*>(da + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][0] * inv_a, o_acc[nt][1] * inv_a);
    }
    if (vb) {
        __nv_bfloat16* db = u_h + (static_cast<size_t>(h) * M + row0 + rb) * geom::KV_LORA + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            *reinterpret_cast<uint32_t*>(db + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][2] * inv_b, o_acc[nt][3] * inv_b);
    }
}

inline void ensure_dsa_tc_smem() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_ernie55_dsa_mla_decode_tc<KvFmt::BF16>,  cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
    cudaFuncSetAttribute(k_ernie55_dsa_mla_decode_tc<KvFmt::FP8>,   cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
    cudaFuncSetAttribute(k_ernie55_dsa_mla_decode_tc<KvFmt::INT4>,  cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
    cudaFuncSetAttribute(k_ernie55_dsa_mla_prefill_tc<KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
    cudaFuncSetAttribute(k_ernie55_dsa_mla_prefill_tc<KvFmt::FP8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
    cudaFuncSetAttribute(k_ernie55_dsa_mla_prefill_tc<KvFmt::INT4>, cudaFuncAttributeMaxDynamicSharedMemorySize, DSA_TC_SMEM);
}

__global__ void __launch_bounds__(256) k_ernie55_dsa_split_reduce(
    const float* __restrict__ split_m,
    const float* __restrict__ split_l,
    const float* __restrict__ split_o,
    __nv_bfloat16* __restrict__ u_h,
    int M, int nh, int NS) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh, h = idx % nh, tid = threadIdx.x;
    float gm = -INFINITY;
    for (int sp = 0; sp < NS; ++sp) gm = fmaxf(gm, split_m[(static_cast<size_t>(m) * NS + sp) * nh + h]);
    float gl = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int sp = 0; sp < NS; ++sp) {
        const size_t sidx = (static_cast<size_t>(m) * NS + sp) * nh + h;
        const float sl = split_l[sidx];
        if (sl <= 0.0f) continue;
        const float w = __expf(split_m[sidx] - gm);
        gl += sl * w;
        const float* so = split_o + sidx * geom::KV_LORA;
        o0 = fmaf(so[tid], w, o0);
        o1 = fmaf(so[tid + 256], w, o1);
    }
    const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    __nv_bfloat16* dst = u_h + (static_cast<size_t>(h) * M + m) * geom::KV_LORA;
    dst[tid] = __float2bfloat16(o0 * inv_l);
    dst[tid + 256] = __float2bfloat16(o1 * inv_l);
}

// Fused Small-M MXFP8 q_a [1024, 4096] + kv_a [576, 4096] + idx_wk [128, 4096] = 1728 rows = 432 CTAs (exact 9.00 waves on 48 SMs)!
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_ernie55_dsa_qkv_idx_mx8(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ qa_q,  const uint8_t* __restrict__ qa_sc,
    const uint8_t* __restrict__ kva_q, const uint8_t* __restrict__ kva_sc,
    const uint8_t* __restrict__ ik_q,  const uint8_t* __restrict__ ik_sc,
    __nv_bfloat16* __restrict__ q_c,
    __nv_bfloat16* __restrict__ kv_raw,
    __nv_bfloat16* __restrict__ idx_k,
    int M, int m_off) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    constexpr int N0 = geom::QA;              // 1024
    constexpr int N1 = N0 + geom::KVA;        // 1600
    constexpr int N2 = N1 + geom::IDX_DIM;    // 1728
    if (o >= N2) return;

    const int kind = (o < N0) ? 0 : ((o < N1) ? 1 : 2);
    const int row  = (kind == 0) ? o : ((kind == 1) ? (o - N0) : (o - N1));
    const uint8_t* Bq  = (kind == 0) ? qa_q  : ((kind == 1) ? kva_q  : ik_q);
    const uint8_t* Bsc = (kind == 0) ? qa_sc : ((kind == 1) ? kva_sc : ik_sc);

    constexpr int K = geom::HID; // 4096
    constexpr int K16 = K >> 4;  // 256
    constexpr int NB  = K >> 5;  // 128
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(row) * K);
    const uint8_t* srow = Bsc + static_cast<size_t>(row) * NB;
    const __nv_bfloat16* Xm0 = X + static_cast<size_t>(m_off) * K;

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
            if (lane == 0) {
                const size_t gm = static_cast<size_t>(m_off + m);
                if (kind == 0)      q_c[gm * geom::QA + row]        = __float2bfloat16(tot);
                else if (kind == 1) kv_raw[gm * geom::KVA + row]    = __float2bfloat16(tot);
                else                idx_k[gm * geom::IDX_DIM + row] = __float2bfloat16(tot);
            }
        }
    }
}

// Fused Small-M MXFP8 q_b [nh_loc*256, 1024] + gate [nh_loc*256, 1024] + optional idx_wq_b [8192, 1024] on shared q_c!
// 32-lane/row, 2x-unrolled (K=1024 -> K16=64 -> single unrolled step, zero loop overhead).
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_ernie55_dsa_qbg_mx8(
    const __nv_bfloat16* __restrict__ q_c,
    const uint8_t* __restrict__ qb_q,   const uint8_t* __restrict__ qb_sc,
    const uint8_t* __restrict__ gate_q, const uint8_t* __restrict__ gate_sc,
    const uint8_t* __restrict__ iq_q,   const uint8_t* __restrict__ iq_sc,
    __nv_bfloat16* __restrict__ q_raw,
    __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ idx_q,
    int M, int m_off, int N_q, int N_g, int N_iq) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= N_q + N_g + N_iq) return;

    const int kind = (o < N_q) ? 0 : ((o < N_q + N_g) ? 1 : 2);
    const int row  = (kind == 0) ? o : ((kind == 1) ? (o - N_q) : (o - N_q - N_g));
    const uint8_t* Bq  = (kind == 0) ? qb_q  : ((kind == 1) ? gate_q  : iq_q);
    const uint8_t* Bsc = (kind == 0) ? qb_sc : ((kind == 1) ? gate_sc : iq_sc);

    constexpr int K = geom::QA; // 1024
    constexpr int NB = K >> 5;  // 32
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(row) * K);
    const uint8_t* srow = Bsc + static_cast<size_t>(row) * NB;
    const __nv_bfloat16* Qm0 = q_c + static_cast<size_t>(m_off) * K;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const uint4 w0 = __ldg(qrow + lane);
    const uint4 w1 = __ldg(qrow + lane + 32);
    const uint8_t s0 = __ldg(srow + (lane >> 1));
    const uint8_t s1 = __ldg(srow + ((lane + 32) >> 1));
    mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Qm0, w0, s0, lane,      M, K, acc);
    mx8::mx8_gemv_step_u4<MAX_M, EXACT_M>(Qm0, w1, s1, lane + 32, M, K, acc);

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = warp_sum(acc[m]);
            if (lane == 0) {
                const size_t gm = static_cast<size_t>(m_off + m);
                if (kind == 0)      q_raw[gm * N_q + row]  = __float2bfloat16(tot);
                else if (kind == 1) gate[gm * N_g + row]   = __float2bfloat16(tot);
                else                idx_q[gm * N_iq + row] = __float2bfloat16(tot);
            }
        }
    }
}

// Direct Small-M MXFP8 V-expand: v_h[h, m, 256] = u_h[h, m, 512] @ w_uv[h, 256, 512]^T.
// 4 rows/warp (4x DRAM MLP, reads u_h ONCE per 4 output rows!):
// Grid: dim3(nh_loc, V_HEAD / 16) = 1024 CTAs, Block: 128 (4 warps x 4 rows = 16 rows/CTA).
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_ernie55_dsa_v_expand_mx8(
    const __nv_bfloat16* __restrict__ u_h, // [nh_loc, M, 512]
    const uint8_t* __restrict__ wq,        // [nh_loc * 256, 512]
    const uint8_t* __restrict__ wsc,       // [nh_loc * 256, 16]
    __nv_bfloat16* __restrict__ v_h,       // [nh_loc, M, 256]
    int M, int nh) {
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v0 = (blockIdx.y * 4 + warp) * 4;
    if (h >= nh || v0 >= geom::V_HEAD) return;

    const size_t base_row = static_cast<size_t>(h) * geom::V_HEAD + v0;
    // Issue 4 concurrent uint4 weight loads + 4 scale loads per lane for 4x DRAM MLP!
    uint4 wv[4];
    float sc[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t rid = base_row + r;
        wv[r] = __ldg(reinterpret_cast<const uint4*>(wq + rid * geom::KV_LORA) + lane);
        sc[r] = __uint_as_float(static_cast<uint32_t>(__ldg(wsc + rid * (geom::KV_LORA >> 5) + (lane >> 1))) << 23);
    }

    if constexpr (MAX_M == 1) {
        const uint4* um = reinterpret_cast<const uint4*>(
            u_h + static_cast<size_t>(h) * geom::KV_LORA);
        const uint4 u0 = __ldg(um + 2 * lane);
        const uint4 u1 = __ldg(um + 2 * lane + 1);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float2 wf[8];
            mx8::fp8x4_unpack(wv[r].x, wf[0], wf[1]);
            mx8::fp8x4_unpack(wv[r].y, wf[2], wf[3]);
            mx8::fp8x4_unpack(wv[r].z, wf[4], wf[5]);
            mx8::fp8x4_unpack(wv[r].w, wf[6], wf[7]);
            const float s = warp_sum(sc[r] * mx8::dot_bf16x16(u0, u1, wf));
            if (lane == 0) v_h[static_cast<size_t>(h) * geom::V_HEAD + v0 + r] = __float2bfloat16(s);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float2 wf[8];
            mx8::fp8x4_unpack(wv[r].x, wf[0], wf[1]);
            mx8::fp8x4_unpack(wv[r].y, wf[2], wf[3]);
            mx8::fp8x4_unpack(wv[r].z, wf[4], wf[5]);
            mx8::fp8x4_unpack(wv[r].w, wf[6], wf[7]);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (EXACT_M || m < M) {
                    const uint4* um = reinterpret_cast<const uint4*>(
                        u_h + (static_cast<size_t>(h) * M + m) * geom::KV_LORA);
                    const float partial = sc[r] * mx8::dot_bf16x16(__ldg(um + 2 * lane), __ldg(um + 2 * lane + 1), wf);
                    const float s = warp_sum(partial);
                    if (lane == 0) v_h[(static_cast<size_t>(h) * M + m) * geom::V_HEAD + v0 + r] = __float2bfloat16(s);
                }
            }
        }
    }
}

// Fused VHAPostMix + sigmoid(gate) on v_h [nh, M, 256] -> attn_out [M, nh * 256].
// Grid: (M * nh), Block: 256 (1 thread per V_HEAD=256 dim).
__global__ void __launch_bounds__(256) k_ernie55_dsa_vha_gate(
    const __nv_bfloat16* __restrict__ v_h,
    const float* __restrict__ vha_mix,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ attn_out,
    int M, int nh, int h_off) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int d = threadIdx.x;
    const float* mrow = vha_mix + static_cast<size_t>(h_off + h) * geom::HEADS + h_off;
    float s = 0.0f;
    for (int g = 0; g < nh; ++g) {
        s = fmaf(__ldg(mrow + g), __bfloat162float(__ldg(v_h + (static_cast<size_t>(g) * M + m) * geom::V_HEAD + d)), s);
    }
    const float vbf = __bfloat162float(__float2bfloat16(s));
    const float gv  = sigmoid_f(__bfloat162float(__ldg(gate + static_cast<size_t>(idx) * geom::V_HEAD + d)));
    attn_out[static_cast<size_t>(idx) * geom::V_HEAD + d] = __float2bfloat16(vbf * gv);
}

struct DsaProfDetail {
    double qkv_idx_ms = 0, prep_ms = 0, qbg_ms = 0, q_abs_ms = 0;
    double idx_score_ms = 0, mla_tc_ms = 0, v_exp_ms = 0, vha_gate_ms = 0, o_proj_ms = 0;
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

// Full MLA-DSA Forward -> out_part [M, 4096] BF16.
inline void dsa_forward(
    cublasHandle_t blas, const DsaWeights& w, DsaScratch& sc, DsaLayerKv& kv,
    const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
    const StepSeg* segs, int nsegs,
    const int* d_slot, const int* d_pos, const int* d_page_table,
    int M, int max_pps, int max_cap, int max_pos,
    uint8_t* x8_scratch, cudaStream_t st,
    int graph_idx_cap = 0, int graph_ns = 0,
    DsaProfDetail* pd = nullptr) {
    if (pd) cudaEventRecord(pd->ev_a, st);
    const bool need_sparse = (max_pos + 1 - geom::SLIDING_WIN > geom::IDX_TOPK) || (graph_idx_cap > 0);
    const bool small_mx8   = (M <= 15 && w.q_a_proj.is_mx8());

    // 1. Q_a + KV_a + Indexer K projections (fused 432 CTAs = exact 9.00 waves on small-M MXFP8!)
    if (small_mx8) {
        auto launch_qkv_idx = [&](int cnt, int m_off) {
            constexpr unsigned G = (geom::QA + geom::KVA + geom::IDX_DIM) / 4; // 432
            if (cnt == 1)      k_ernie55_dsa_qkv_idx_mx8<1, true> <<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, 1, m_off);
            else if (cnt == 2) k_ernie55_dsa_qkv_idx_mx8<2, true> <<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, 2, m_off);
            else if (cnt == 4) k_ernie55_dsa_qkv_idx_mx8<4, true> <<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, 4, m_off);
            else if (cnt == 6) k_ernie55_dsa_qkv_idx_mx8<6, true> <<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, 6, m_off);
            else if (cnt == 8) k_ernie55_dsa_qkv_idx_mx8<8, true> <<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, 8, m_off);
            else               k_ernie55_dsa_qkv_idx_mx8<8, false><<<G, 128, 0, st>>>(x_norm, w.q_a_proj.mx8.q, w.q_a_proj.mx8.sc, w.kv_a_proj.mx8.q, w.kv_a_proj.mx8.sc, w.idx_wk.mx8.q, w.idx_wk.mx8.sc, sc.q_c, sc.kv_raw, sc.idx_k, cnt, m_off);
        };
        if (M <= 8) launch_qkv_idx(M, 0);
        else { launch_qkv_idx(8, 0); launch_qkv_idx(M - 8, 8); }
    } else {
        const uint8_t* x_x8 = nullptr;
        if (M >= 16 && w.q_a_proj.is_mx8() && x8_scratch != nullptr) {
            mx8::bf16_to_mx8_a(x_norm, x8_scratch, M, geom::HID, st);
            x_x8 = x8_scratch;
        }
        mx8::linear_w(blas, x_norm, w.q_a_proj,  sc.q_c,    M, geom::QA,      geom::HID, x8_scratch, x_x8, st);
        mx8::linear_w(blas, x_norm, w.kv_a_proj, sc.kv_raw, M, geom::KVA,     geom::HID, x8_scratch, x_x8, st);
        mx8::linear_w(blas, x_norm, w.idx_wk,    sc.idx_k,  M, geom::IDX_DIM, geom::HID, x8_scratch, x_x8, st);
    }
    if (pd) pd->mark(pd->qkv_idx_ms, st);

    // 2. Q_c RMSNorm, KV paged write, Indexer K paged write
    k_ernie55_rmsnorm<<<M, 256, 0, st>>>(sc.q_c, w.q_a_ln, sc.q_c, M, geom::QA, geom::RMS_EPS);
    if (kv.fmt == KvFmt::FP8)       k_ernie55_dsa_kv_write<KvFmt::FP8> <<<M, 256, 0, st>>>(sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, kv.kv_pool, M, max_pps);
    else if (kv.fmt == KvFmt::INT4) k_ernie55_dsa_kv_write<KvFmt::INT4><<<M, 256, 0, st>>>(sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, kv.kv_pool, M, max_pps);
    else                            k_ernie55_dsa_kv_write<KvFmt::BF16><<<M, 256, 0, st>>>(sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, kv.kv_pool, M, max_pps);
    if (kv.fmt == KvFmt::BF16) {
        k_ernie55_idx_k_write<false><<<M, 32, 0, st>>>(
            sc.idx_k, w.idx_k_norm_w, w.idx_k_norm_b, d_slot, d_pos, d_page_table,
            kv.idx_bf16, nullptr, nullptr, M, max_pps);
    } else {
        k_ernie55_idx_k_write<true><<<M, 32, 0, st>>>(
            sc.idx_k, w.idx_k_norm_w, w.idx_k_norm_b, d_slot, d_pos, d_page_table,
            nullptr, kv.idx_q, kv.idx_sc, M, max_pps);
    }
    if (pd) pd->mark(pd->prep_ms, st);

    // 3. Q_b, gate_proj(q_c), and (when sparse & small_mx8) fused Indexer Q on shared q_c!
    const int N_q  = w.nh_loc * geom::QK_HEAD;
    const int N_g  = w.nh_loc * geom::V_HEAD;
    const int N_iq = geom::IDX_HEADS * geom::IDX_DIM;
    const uint8_t* qc_x8 = nullptr;
    bool idx_q_done = false;
    if (small_mx8) {
        const int niq = need_sparse ? N_iq : 0;
        const unsigned G = static_cast<unsigned>((N_q + N_g + niq + 3) / 4);
        auto launch_qbg = [&](int cnt, int m_off) {
            if (cnt == 1)      k_ernie55_dsa_qbg_mx8<1, true> <<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, 1, m_off, N_q, N_g, niq);
            else if (cnt == 2) k_ernie55_dsa_qbg_mx8<2, true> <<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, 2, m_off, N_q, N_g, niq);
            else if (cnt == 4) k_ernie55_dsa_qbg_mx8<4, true> <<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, 4, m_off, N_q, N_g, niq);
            else if (cnt == 6) k_ernie55_dsa_qbg_mx8<6, true> <<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, 6, m_off, N_q, N_g, niq);
            else if (cnt == 8) k_ernie55_dsa_qbg_mx8<8, true> <<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, 8, m_off, N_q, N_g, niq);
            else               k_ernie55_dsa_qbg_mx8<8, false><<<G, 128, 0, st>>>(sc.q_c, w.q_b_proj.mx8.q, w.q_b_proj.mx8.sc, w.gate_proj.mx8.q, w.gate_proj.mx8.sc, w.idx_wq_b.mx8.q, w.idx_wq_b.mx8.sc, sc.q_raw, sc.gate, sc.idx_q, cnt, m_off, N_q, N_g, niq);
        };
        if (M <= 8) launch_qbg(M, 0);
        else { launch_qbg(8, 0); launch_qbg(M - 8, 8); }
        idx_q_done = need_sparse;
    } else {
        if (M >= 16 && w.q_b_proj.is_mx8() && x8_scratch != nullptr) {
            mx8::bf16_to_mx8_a(sc.q_c, x8_scratch, M, geom::QA, st);
            qc_x8 = x8_scratch;
        }
        mx8::linear_w(blas, sc.q_c, w.q_b_proj,  sc.q_raw, M, N_q, geom::QA, x8_scratch, qc_x8, st);
        mx8::linear_w(blas, sc.q_c, w.gate_proj, sc.gate,  M, N_g, geom::QA, x8_scratch, qc_x8, st);
    }
    if (pd) pd->mark(pd->qbg_ms, st);

    // 4. Q absorption + RoPE
    if (M <= 12) {
        if (w.w_uk_t.is_mx8()) {
            const dim3 g(w.nh_loc, geom::KV_LORA / 16);
            if (M <= 1)      k_ernie55_dsa_q_absorb_rope_mx8<1> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.mx8.q, w.w_uk_t.mx8.sc, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 2) k_ernie55_dsa_q_absorb_rope_mx8<2> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.mx8.q, w.w_uk_t.mx8.sc, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 4) k_ernie55_dsa_q_absorb_rope_mx8<4> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.mx8.q, w.w_uk_t.mx8.sc, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 8) k_ernie55_dsa_q_absorb_rope_mx8<8> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.mx8.q, w.w_uk_t.mx8.sc, d_pos, sc.q_full, M, w.nh_loc);
            else             k_ernie55_dsa_q_absorb_rope_mx8<12><<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.mx8.q, w.w_uk_t.mx8.sc, d_pos, sc.q_full, M, w.nh_loc);
        } else {
            const dim3 g(w.nh_loc, geom::KV_LORA / 4);
            if (M <= 1)      k_ernie55_dsa_q_absorb_rope_bf16<1> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.bf16, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 2) k_ernie55_dsa_q_absorb_rope_bf16<2> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.bf16, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 4) k_ernie55_dsa_q_absorb_rope_bf16<4> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.bf16, d_pos, sc.q_full, M, w.nh_loc);
            else if (M <= 8) k_ernie55_dsa_q_absorb_rope_bf16<8> <<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.bf16, d_pos, sc.q_full, M, w.nh_loc);
            else             k_ernie55_dsa_q_absorb_rope_bf16<12><<<g, 128, 0, st>>>(sc.q_raw, w.w_uk_t.bf16, d_pos, sc.q_full, M, w.nh_loc);
        }
    } else {
        k_ernie55_dsa_q_split_rope<<<M * w.nh_loc, 256, 0, st>>>(sc.q_raw, d_pos, sc.q_nope_h, sc.q_full, M, w.nh_loc);
        const __nv_bfloat16* wuk_bf16 = w.w_uk_t.bf16;
        if (w.w_uk_t.is_mx8()) {
            mx8::dequant_to_bf16(w.w_uk_t.mx8, sc.kv_b_tmp, st);
            wuk_bf16 = sc.kv_b_tmp;
        }
        cublasSetStream(blas, st);
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            geom::KV_LORA, M, geom::QK_NOPE,
            &alpha, wuk_bf16, CUDA_R_16BF, geom::QK_NOPE, static_cast<long long>(geom::KV_LORA) * geom::QK_NOPE,
            sc.q_nope_h, CUDA_R_16BF, geom::QK_NOPE, static_cast<long long>(M) * geom::QK_NOPE,
            &beta, sc.q_abs_h, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(M) * geom::KV_LORA,
            w.nh_loc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        k_ernie55_dsa_q_abs_scatter<<<M * w.nh_loc, 256, 0, st>>>(sc.q_abs_h, sc.q_full, M, w.nh_loc);
    }
    if (pd) pd->mark(pd->q_abs_ms, st);

    // 5. Sparse Lightning Indexer scoring & top-2048 selection (only when pos + 1 - 128 > 2048!)
    if (need_sparse) {
        if (!idx_q_done) {
            mx8::linear_w(blas, sc.q_c, w.idx_wq_b, sc.idx_q, M, N_iq, geom::QA, x8_scratch, qc_x8, st);
        }
        k_ernie55_idx_q_rope_had<<<M * geom::IDX_HEADS, 32, 0, st>>>(sc.idx_q, d_pos, M);
        linear_bf16_f32(blas, x_norm, w.idx_w_proj, sc.idx_w, M, geom::IDX_HEADS, geom::HID, st);

        const int score_lim = (graph_idx_cap > 0) ? graph_idx_cap : std::max(1, max_pos + 1 - geom::SLIDING_WIN);
        for (int m0 = 0; m0 < M; m0 += IDX_SCORE_BATCH) {
            const int nm = std::min(IDX_SCORE_BATCH, M - m0);
            const dim3 grid((score_lim + 31) / 32, nm);
            if (kv.fmt == KvFmt::BF16) {
                k_ernie55_idx_score_tc<false><<<grid, 128, 0, st>>>(
                    sc.idx_q, sc.idx_w, kv.idx_bf16, nullptr, nullptr,
                    d_slot, d_pos, d_page_table, sc.idx_sc, m0, nm, max_pps, max_cap);
            } else {
                k_ernie55_idx_score_tc<true><<<grid, 128, 0, st>>>(
                    sc.idx_q, sc.idx_w, nullptr, kv.idx_q, kv.idx_sc,
                    d_slot, d_pos, d_page_table, sc.idx_sc, m0, nm, max_pps, max_cap);
            }
            k_ernie55_idx_topk2048<<<nm, 256, 0, st>>>(
                sc.idx_sc, d_pos, sc.sel_tok, sc.sel_cnt, m0, nm, max_cap);
        }
    }
    if (pd) pd->mark(pd->idx_score_ms, st);

    // 6. Tensor-Core Absorbed MLA Attention
    ensure_dsa_tc_smem();
    const bool use_prefill_tc = (!need_sparse && M > 12 && graph_ns == 0);
    if (use_prefill_tc) {
        for (int s = 0; s < nsegs; ++s) {
            const StepSeg& sg = segs[s];
            if (sg.len <= 0) continue;
            const dim3 grid((sg.len + 15) / 16, w.nh_loc);
            if (kv.fmt == KvFmt::FP8)       k_ernie55_dsa_mla_prefill_tc<KvFmt::FP8> <<<grid, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_page_table, sc.u_h, M, w.nh_loc, max_pps, sg.slot, sg.row0, sg.pos0, sg.len);
            else if (kv.fmt == KvFmt::INT4) k_ernie55_dsa_mla_prefill_tc<KvFmt::INT4><<<grid, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_page_table, sc.u_h, M, w.nh_loc, max_pps, sg.slot, sg.row0, sg.pos0, sg.len);
            else                            k_ernie55_dsa_mla_prefill_tc<KvFmt::BF16><<<grid, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_page_table, sc.u_h, M, w.nh_loc, max_pps, sg.slot, sg.row0, sg.pos0, sg.len);
        }
    } else {
        const int n_hg = w.nh_loc >> 4;
        const int est_k = need_sparse ? geom::DSA_ATTN_TOPK : (max_pos + 1);
        int NS = 1;
        if (graph_ns > 0) {
            NS = graph_ns;
        } else if (M <= 12 && est_k >= 256) {
            NS = std::min(DSA_MAX_SPLITS, std::max(1, std::min(16 / M, est_k / 128)));
        }
        const int* sel_p = need_sparse ? sc.sel_tok : nullptr;
        const int* cnt_p = need_sparse ? sc.sel_cnt : nullptr;
        if (kv.fmt == KvFmt::FP8)       k_ernie55_dsa_mla_decode_tc<KvFmt::FP8> <<<M * n_hg * NS, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_slot, d_pos, d_page_table, sel_p, cnt_p, sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, w.nh_loc, max_pps, NS);
        else if (kv.fmt == KvFmt::INT4) k_ernie55_dsa_mla_decode_tc<KvFmt::INT4><<<M * n_hg * NS, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_slot, d_pos, d_page_table, sel_p, cnt_p, sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, w.nh_loc, max_pps, NS);
        else                            k_ernie55_dsa_mla_decode_tc<KvFmt::BF16><<<M * n_hg * NS, 256, DSA_TC_SMEM, st>>>(sc.q_full, w.sinks, kv.kv_pool, d_slot, d_pos, d_page_table, sel_p, cnt_p, sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, w.nh_loc, max_pps, NS);
        if (NS > 1) {
            k_ernie55_dsa_split_reduce<<<M * w.nh_loc, 256, 0, st>>>(sc.split_m, sc.split_l, sc.split_o, sc.u_h, M, w.nh_loc, NS);
        }
    }
    if (pd) pd->mark(pd->mla_tc_ms, st);

    // 7. V expand: v_h[h, m, 256] = u_h[h, m, 512] @ w_uv[h, 256, 512]^T (4 rows/warp -> 1024 CTAs)
    if (M <= 12 && w.w_uv.is_mx8()) {
        const dim3 gv(w.nh_loc, geom::V_HEAD / 16);
        if (M == 1)      k_ernie55_dsa_v_expand_mx8<1,  true> <<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, 1, w.nh_loc);
        else if (M == 2) k_ernie55_dsa_v_expand_mx8<2,  true> <<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, 2, w.nh_loc);
        else if (M == 4) k_ernie55_dsa_v_expand_mx8<4,  true> <<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, 4, w.nh_loc);
        else if (M == 8) k_ernie55_dsa_v_expand_mx8<8,  true> <<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, 8, w.nh_loc);
        else if (M <  8) k_ernie55_dsa_v_expand_mx8<8,  false><<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, M, w.nh_loc);
        else             k_ernie55_dsa_v_expand_mx8<12, false><<<gv, 128, 0, st>>>(sc.u_h, w.w_uv.mx8.q, w.w_uv.mx8.sc, sc.v_h, M, w.nh_loc);
    } else {
        const __nv_bfloat16* wuv_bf16 = w.w_uv.bf16;
        if (w.w_uv.is_mx8()) {
            mx8::dequant_to_bf16(w.w_uv.mx8, sc.kv_b_tmp, st);
            wuv_bf16 = sc.kv_b_tmp;
        }
        cublasSetStream(blas, st);
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            geom::V_HEAD, M, geom::KV_LORA,
            &alpha, wuv_bf16, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(geom::V_HEAD) * geom::KV_LORA,
            sc.u_h, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(M) * geom::KV_LORA,
            &beta, sc.v_h, CUDA_R_16BF, geom::V_HEAD, static_cast<long long>(M) * geom::V_HEAD,
            w.nh_loc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    if (pd) pd->mark(pd->v_exp_ms, st);

    // 8. VHAPostMix + * sigmoid(gate) -> o_proj -> out_part [M, 4096]
    k_ernie55_dsa_vha_gate<<<M * w.nh_loc, 256, 0, st>>>(sc.v_h, w.vha_mix, sc.gate, sc.attn_out, M, w.nh_loc, w.h_off);
    if (pd) pd->mark(pd->vha_gate_ms, st);
    mx8::linear_w(blas, sc.attn_out, w.o_proj, out_part, M, geom::HID, w.nh_loc * geom::V_HEAD, x8_scratch, nullptr, st);
    if (pd) pd->mark(pd->o_proj_ms, st);
}

} // namespace ernie55
