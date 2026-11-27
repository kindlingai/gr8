// Tencent Hy4-flash Dual-Shape Gated Absorbed MLA + Compressed Group Indexer + 1M Dynamic Paged KV Pool.
// Upgrades over baseline:
//   - Multi-precision DSA KV cache (--kv-format fp8|int4|bf16):
//       BF16: 576*2 = 1152 B/tok + 256 B/grp idx (14.25 GiB / 1M)
//       FP8 : 512 E4M3 + 16 FP16 scales + 64 BF16 rope = 672 B/tok + FP8 idx 136 B/grp (7.91 GiB / 1M, -44.5% VRAM!)
//       INT4: 256 packed INT4 + 16 FP16 scales + 64 BF16 rope = 416 B/tok + FP8 idx 136 B/grp (5.05 GiB / 1M, -64.6% VRAM!)
//   - Blackwell MXFP8 dense projections (q_a, q_b, kv_a, w_uk_t, w_uv, linear_gate, o_proj, idx_q, idx_tk) with
//     fused small-M MXFP8 q_absorb_rope and v_expand_gate kernels (2x less dense BW, ~5.1 GiB VRAM reclaimed!).
//   - Tensor-Core mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 kernels for:
//       1. Compressed Group Indexer scoring (16 heads x 8 groups per warp, zero idx_q reloads)
//       2. Absorbed MLA Prefill (16 query tokens x 32 KV tokens tiled FlashMLA, 16x KV reuse!)
//       3. Absorbed MLA Decode/Verify/Sparse (16 query heads x 32 KV tokens, 16x KV reuse + native FP8/INT4 dequant!)
//   - DFly2 Index Snapshot hook for target DSA layers [25, 29, 33, 37, 41].
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../model_geom_hy4.h"
#include "../kernels/hy4_hc.cuh"
#include "../kernels/hy4_mxfp8.cuh"

namespace hy4 {

enum class KvFmt : int { BF16 = 0, FP8 = 1, INT4 = 2 };

constexpr int KV_FP8_STRIDE  = 672; // 512 E4M3 nope + 32 B (16 half scales) + 128 B (64 bf16 rope)
constexpr int KV_INT4_STRIDE = 416; // 256 packed INT4 nope + 32 B (16 half scales) + 128 B (64 bf16 rope)
constexpr int IDX_SCORE_BATCH = 64;
constexpr int MLA_MAX_SPLITS  = 32;

inline size_t dsa_kv_tok_bytes(KvFmt f) {
    if (f == KvFmt::FP8)  return KV_FP8_STRIDE;
    if (f == KvFmt::INT4) return KV_INT4_STRIDE;
    return geom::KVA * sizeof(__nv_bfloat16);
}

__device__ __forceinline__ void mma16816_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

struct AttnWeights {
    bool is_dsa = false;
    int nh_loc  = 0;                               // DSA: 32/nr, SWA: 64/nr
    mx8::LinW            q_a_proj{};               // [1024, 3072]
    const __nv_bfloat16* q_a_ln        = nullptr;  // [1024]
    mx8::LinW            q_b_proj{};               // [nh_loc * 256, 1024]
    mx8::LinW            kv_a_proj{};              // [576, 3072]
    const __nv_bfloat16* kv_a_ln       = nullptr;  // [512]
    // Unpacked from kv_b_proj [nh_loc, 448, 512]:
    mx8::LinW            w_uk_t{};                 // [nh_loc * 512, 192]
    mx8::LinW            w_uv{};                   // [nh_loc * 256, 512]
    mx8::LinW            gate_proj{};              // [nh_loc * 256, 3072]
    mx8::LinW            o_proj{};                 // [3072, nh_loc * 256]
    const __nv_bfloat16* post_proj_ln  = nullptr;  // [3072]
    const float*         sinks         = nullptr;  // [nh_loc] FP32

    // Compressed group indexer (DSA layers only)
    mx8::LinW            idx_q_proj{};             // [2048, 1024]
    const __nv_bfloat16* idx_hw_proj   = nullptr;  // [16, 3072] BF16
    mx8::LinW            idx_tk_proj{};            // [128, 3072]
    const __nv_bfloat16* idx_tk_norm   = nullptr;  // [128]
    const float*         idx_kv_proj   = nullptr;  // [128, 128] FP32
    const float*         idx_kg_proj   = nullptr;  // [128, 128] FP32
    const float*         idx_k_norm    = nullptr;  // [128] FP32
    const float*         idx_ape       = nullptr;  // [4, 128] FP32
};

// Dynamic paged KV for one DSA layer.
struct DsaLayerKv {
    KvFmt fmt = KvFmt::FP8;
    uint8_t*       kv_pool  = nullptr; // [pool_pages][64][dsa_kv_tok_bytes(fmt)]
    __nv_bfloat16* idx_bf16 = nullptr; // BF16 mode: [pool_pages][16][128] BF16
    uint8_t*       idx_q    = nullptr; // FP8/INT4 mode: [pool_pages][16][128] E4M3
    __half*        idx_sc   = nullptr; // FP8/INT4 mode: [pool_pages][16][4] FP16 (per-32 scales)
    __nv_bfloat16* idx_raw  = nullptr; // [slots][4][128] BF16 (pending raw token keys for incomplete 4-tok group)
};

// Fixed sliding-window ring for one SWA layer (always BF16: 2048*576*2B = 2.25 MiB/slot/layer).
struct SwaLayerKv {
    __nv_bfloat16* ring = nullptr; // [slots][SWA_RING][576] BF16
};

struct AttnScratch {
    __nv_bfloat16* q_a      = nullptr; // [max_M, 1024]
    __nv_bfloat16* q_resid  = nullptr; // [max_M, 1024]
    __nv_bfloat16* q_raw    = nullptr; // [max_M, max_nh * 256]
    __nv_bfloat16* q_nope_h = nullptr; // [max_nh, max_M, 192]
    __nv_bfloat16* q_abs_h  = nullptr; // [max_nh, max_M, 512]
    __nv_bfloat16* q_full   = nullptr; // [max_M, max_nh, 576]
    __nv_bfloat16* kv_raw   = nullptr; // [max_M, 576]
    __nv_bfloat16* gate     = nullptr; // [max_M, max_nh * 256]
    __nv_bfloat16* u_h      = nullptr; // [max_nh, max_M, 512]
    __nv_bfloat16* v_h      = nullptr; // [max_nh, max_M, 256]
    __nv_bfloat16* attn_out = nullptr; // [max_M, max_nh * 256]
    __nv_bfloat16* kv_b_tmp = nullptr; // [max_nh * 512 * 256] (on-the-fly MXFP8 dequant for M > 12 batched w_uk/w_uv)

    // Split-KV scratch for small-M decode/verify
    float* split_m = nullptr; // [max_dec_M, MLA_MAX_SPLITS, max_nh]
    float* split_l = nullptr; // [max_dec_M, MLA_MAX_SPLITS, max_nh]
    float* split_o = nullptr; // [max_dec_M, MLA_MAX_SPLITS, max_nh, 512]

    // Compressed indexer scratch
    __nv_bfloat16* idx_tk  = nullptr; // [max_M, 128]
    __nv_bfloat16* idx_q   = nullptr; // [max_M, 16, 128]
    float*         idx_hw  = nullptr; // [max_M, 16]
    float*         idx_sc  = nullptr; // [IDX_SCORE_BATCH, max_groups]
    int*           sel_grp = nullptr; // [max_M, 511]
    int*           sel_cnt = nullptr; // [max_M]
};

// =================================================================================================
// Startup in-place unpack of kv_b_proj [nh_loc, 448, 512] -> w_uk_t [nh_loc, 512, 192] + w_uv [nh_loc, 256, 512]
// =================================================================================================
__global__ void __launch_bounds__(256) k_hy4_unpack_kv_b(
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
        for (int c = threadIdx.x; c < geom::KV_LORA; c += 256) {
            dst[c] = row[c];
        }
    }
}

// =================================================================================================
// Interleaved RoPE on 64 dims (32 pairs: (2i, 2i+1))
// =================================================================================================
__device__ __forceinline__ void hy4_rope_pair(float& x0, float& x1, int pos, int pair_idx, float theta) {
    const float inv_freq = powf(theta, -static_cast<float>(pair_idx * 2) * (1.0f / 64.0f));
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = __bfloat162float(__float2bfloat16(__cosf(ang)));
    const float s = __bfloat162float(__float2bfloat16(__sinf(ang)));
    const float y0 = x0 * c - x1 * s;
    const float y1 = x1 * c + x0 * s;
    x0 = y0;
    x1 = y1;
}

__global__ void __launch_bounds__(256) k_hy4_q_split_rope(
    const __nv_bfloat16* __restrict__ q_raw,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_nope_h,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh, float theta) {
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
        hy4_rope_pair(x0, x1, d_pos[m], p, theta);
        __nv_bfloat16* dst = q_full + static_cast<size_t>(idx) * geom::KVA + geom::KV_LORA;
        dst[2 * p]     = __float2bfloat16(x0);
        dst[2 * p + 1] = __float2bfloat16(x1);
    }
}

__global__ void __launch_bounds__(256) k_hy4_q_abs_scatter(
    const __nv_bfloat16* __restrict__ q_abs_h,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const uint2* src = reinterpret_cast<const uint2*>(q_abs_h + (static_cast<size_t>(h) * M + m) * geom::KV_LORA);
    uint2* dst = reinterpret_cast<uint2*>(q_full + static_cast<size_t>(idx) * geom::KVA);
    const int t = threadIdx.x;
    if (t < 128) dst[t] = src[t];
}

// =================================================================================================
// KV Write (Supports BF16, FP8-E4M3 per-32 block scales, and INT4-KV4 per-32 block scales!)
// Grid: (M), Block: 256.
// =================================================================================================
template <KvFmt FMT, bool IS_DSA>
__global__ void __launch_bounds__(256) k_hy4_kv_write(
    const __nv_bfloat16* __restrict__ kv_raw,
    const __nv_bfloat16* __restrict__ kv_a_ln,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    uint8_t* __restrict__ kv_dst,
    int M, int max_pps, float theta) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const __nv_bfloat16* src = kv_raw + static_cast<size_t>(m) * geom::KVA;

    // 512 elements = 2 per thread
    const float v0 = __bfloat162float(src[tid]);
    const float v1 = __bfloat162float(src[tid + 256]);
    float ss = warp_sum(v0 * v0 + v1 * v1);
    __shared__ float sm[8];
    if (lane == 0) sm[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < 8) ? sm[tid] : 0.0f;
        tot = warp_sum(tot);
        if (tid == 0) sm[0] = rsqrtf(tot / static_cast<float>(geom::KV_LORA) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];

    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    uint8_t* out_u8;
    if constexpr (IS_DSA) {
        const int lpage = pos / geom::KV_PAGE;
        const int off   = pos % geom::KV_PAGE;
        const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
        constexpr size_t STRIDE = (FMT == KvFmt::FP8) ? KV_FP8_STRIDE
                                : (FMT == KvFmt::INT4) ? KV_INT4_STRIDE
                                : (geom::KVA * sizeof(__nv_bfloat16));
        out_u8 = kv_dst + (static_cast<size_t>(ppage) * geom::KV_PAGE + off) * STRIDE;
    } else {
        const int off = pos & (geom::SWA_RING - 1);
        out_u8 = kv_dst + (static_cast<size_t>(slot) * geom::SWA_RING + off) * (geom::KVA * sizeof(__nv_bfloat16));
    }

    // Normed nope values (512 dims) stored in smem for clean 32-dim block quantization
    __shared__ float s_nope[512];
    const float n0 = __bfloat162float(__float2bfloat16(v0 * inv)) * __bfloat162float(kv_a_ln[tid]);
    const float n1 = __bfloat162float(__float2bfloat16(v1 * inv)) * __bfloat162float(kv_a_ln[tid + 256]);
    const float qn0 = __bfloat162float(__float2bfloat16(n0));
    const float qn1 = __bfloat162float(__float2bfloat16(n1));

    if constexpr (!IS_DSA || FMT == KvFmt::BF16) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(out_u8);
        out[tid]       = __float2bfloat16(qn0);
        out[tid + 256] = __float2bfloat16(qn1);
    } else {
        s_nope[tid]       = qn0;
        s_nope[tid + 256] = qn1;
        __syncthreads();
        // 8 warps process 16 blocks of 32 dims (2 passes of 8 blocks = 1 warp per 32-dim block!)
        const int warp = tid >> 5;
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int blk = pass * 8 + warp; // 0..15
            const float val = s_nope[blk * 32 + lane];
            const float amax = warp_max(fabsf(val));
            if constexpr (FMT == KvFmt::FP8) {
                const float sc = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
                const float inv_sc = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
                out_u8[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val * inv_sc, __NV_SATFINITE, __NV_E4M3));
                if (lane == 0) reinterpret_cast<__half*>(out_u8 + 512)[blk] = __float2half_rn(sc);
            } else { // KvFmt::INT4: symmetric 4-bit offset 8 in [-7..+7] -> u4 in [1..15], (q - 8) * sc
                const float sc = (amax > 0.0f) ? (amax * (1.0f / 7.0f)) : 1.0f;
                const float inv_sc = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
                const int qi = max(-7, min(7, __float2int_rn(val * inv_sc))) + 8; // 1..15
                const int q_hi = __shfl_down_sync(0xffffffffu, qi, 1);
                if ((lane & 1) == 0) {
                    out_u8[blk * 16 + (lane >> 1)] = static_cast<uint8_t>((qi & 0xf) | ((q_hi & 0xf) << 4));
                }
                if (lane == 0) reinterpret_cast<__half*>(out_u8 + 256)[blk] = __float2half_rn(sc);
            }
        }
    }

    // RoPE on 64 dims (always exact BF16!)
    if (tid < 32) {
        float x0 = __bfloat162float(src[geom::KV_LORA + 2 * tid]);
        float x1 = __bfloat162float(src[geom::KV_LORA + 2 * tid + 1]);
        hy4_rope_pair(x0, x1, pos, tid, theta);
        constexpr size_t ROPE_OFF = (!IS_DSA || FMT == KvFmt::BF16) ? (geom::KV_LORA * 2)
                                  : (FMT == KvFmt::FP8) ? (512 + 32) : (256 + 32);
        __nv_bfloat16* rdst = reinterpret_cast<__nv_bfloat16*>(out_u8 + ROPE_OFF);
        rdst[2 * tid]     = __float2bfloat16(x0);
        rdst[2 * tid + 1] = __float2bfloat16(x1);
    }
}

// =================================================================================================
// Compressed Group Indexer Key Update (BF16 or FP8 idx_pool) + Tensor-Core Group Scorer + Radix Top-511
// =================================================================================================
template <bool IDX_FP8>
__device__ __forceinline__ void hy4_pool_4_keys_f32(
    const __nv_bfloat16 (&s_raw)[4][128],
    const float* __restrict__ kv_proj,
    const float* __restrict__ kg_proj,
    const float* __restrict__ ape,
    const float* __restrict__ k_norm_w,
    int group_pos,
    __nv_bfloat16* __restrict__ dst_bf16,
    uint8_t* __restrict__ dst_q,
    __half* __restrict__ dst_sc,
    float (&s_norm)[128]) {
    const int d = threadIdx.x; // 0..127
    const float* vr = kv_proj + static_cast<size_t>(d) * 128;
    const float* gr = kg_proj + static_cast<size_t>(d) * 128;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float g[4] = {ape[0 * 128 + d], ape[1 * 128 + d], ape[2 * 128 + d], ape[3 * 128 + d]};
    for (int c = 0; c < 128; ++c) {
        const float wv = vr[c];
        const float wg = gr[c];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float x = __bfloat162float(s_raw[k][c]);
            v[k] = fmaf(x, wv, v[k]);
            g[k] = fmaf(x, wg, g[k]);
        }
    }
    const float mg = fmaxf(fmaxf(g[0], g[1]), fmaxf(g[2], g[3]));
    const float e0 = expf(g[0] - mg), e1 = expf(g[1] - mg), e2 = expf(g[2] - mg), e3 = expf(g[3] - mg);
    const float inv_e = 1.0f / (e0 + e1 + e2 + e3);
    const float pooled = (e0 * v[0] + e1 * v[1] + e2 * v[2] + e3 * v[3]) * inv_e;

    // HYV4FlashRMSNorm(pooled, eps=1e-5) in FP32 with FP32 key_norm_w, then cast to BF16!
    float ss = warp_sum(pooled * pooled);
    __shared__ float sm[4];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) * (1.0f / 128.0f) + geom::RMS_EPS);
    s_norm[d] = __bfloat162float(__float2bfloat16(pooled * inv * k_norm_w[d]));
    __syncthreads();

    float out_val = s_norm[d];
    if (d >= 64) {
        // Compute rotated pair cleanly for trailing 64 dims
        const int p = (d - 64) >> 1;
        float x0 = s_norm[64 + 2 * p];
        float x1 = s_norm[64 + 2 * p + 1];
        hy4_rope_pair(x0, x1, group_pos, p, geom::ROPE_THETA_DSA);
        out_val = ((d & 1) == 0) ? x0 : x1;
    }

    if constexpr (!IDX_FP8) {
        dst_bf16[d] = __float2bfloat16(out_val);
    } else {
        // 4 warps = 4 blocks of 32 dims -> E4M3 + FP16 scale
        const float qv = __bfloat162float(__float2bfloat16(out_val));
        const float amax = warp_max(fabsf(qv));
        const float sc = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
        const float inv_sc = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
        dst_q[d] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(qv * inv_sc, __NV_SATFINITE, __NV_E4M3));
        if ((d & 31) == 0) dst_sc[d >> 5] = __float2half_rn(sc);
    }
}

template <bool IDX_FP8>
__global__ void __launch_bounds__(128) k_hy4_idx_k_update_dec(
    const __nv_bfloat16* __restrict__ idx_tk,
    const float* __restrict__ kv_proj,
    const float* __restrict__ kg_proj,
    const float* __restrict__ ape,
    const float* __restrict__ k_norm_w,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    __nv_bfloat16* __restrict__ idx_raw,
    __nv_bfloat16* __restrict__ idx_bf16,
    uint8_t* __restrict__ idx_q,
    __half* __restrict__ idx_sc,
    int M, int max_pps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int d = threadIdx.x;
    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    const int sub  = pos & 3;
    __nv_bfloat16* rslot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    const __nv_bfloat16 v = idx_tk[static_cast<size_t>(m) * 128 + d];
    rslot[sub * 128 + d] = v;
    if (sub != 3) return;

    __shared__ __nv_bfloat16 s_raw[4][128];
    __shared__ float s_norm[128];
    #pragma unroll
    for (int k = 0; k < 3; ++k) s_raw[k][d] = rslot[k * 128 + d];
    s_raw[3][d] = v;
    __syncthreads();

    const int g = pos >> 2;
    const int lpage = g >> 4;
    const int goff  = g & 15;
    const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
    const size_t go = static_cast<size_t>(ppage) * 16 + goff;
    hy4_pool_4_keys_f32<IDX_FP8>(
        s_raw, kv_proj, kg_proj, ape, k_norm_w, g * 4,
        idx_bf16 ? (idx_bf16 + go * 128) : nullptr,
        idx_q    ? (idx_q    + go * 128) : nullptr,
        idx_sc   ? (idx_sc   + go * 4)   : nullptr,
        s_norm);
}

template <bool IDX_FP8>
__global__ void __launch_bounds__(128) k_hy4_idx_k_update_seg_complete(
    const __nv_bfloat16* __restrict__ idx_tk,
    const float* __restrict__ kv_proj,
    const float* __restrict__ kg_proj,
    const float* __restrict__ ape,
    const float* __restrict__ k_norm_w,
    const int* __restrict__ d_page_table,
    const __nv_bfloat16* __restrict__ idx_raw,
    __nv_bfloat16* __restrict__ idx_bf16,
    uint8_t* __restrict__ idx_q,
    __half* __restrict__ idx_sc,
    int max_pps, int slot, int row0, int pos0, int len) {
    const int d = threadIdx.x;
    const int g = (pos0 >> 2) + blockIdx.x;
    const int gp0 = g * 4;
    if (gp0 + 4 > pos0 + len) return;

    const __nv_bfloat16* rslot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    __shared__ __nv_bfloat16 s_raw[4][128];
    __shared__ float s_norm[128];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = gp0 + k;
        s_raw[k][d] = (p < pos0) ? rslot[k * 128 + d] : idx_tk[static_cast<size_t>(row0 + (p - pos0)) * 128 + d];
    }
    __syncthreads();

    const int lpage = g >> 4;
    const int goff  = g & 15;
    const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
    const size_t go = static_cast<size_t>(ppage) * 16 + goff;
    hy4_pool_4_keys_f32<IDX_FP8>(
        s_raw, kv_proj, kg_proj, ape, k_norm_w, gp0,
        idx_bf16 ? (idx_bf16 + go * 128) : nullptr,
        idx_q    ? (idx_q    + go * 128) : nullptr,
        idx_sc   ? (idx_sc   + go * 4)   : nullptr,
        s_norm);
}

__global__ void __launch_bounds__(128) k_hy4_idx_k_update_seg_tail(
    const __nv_bfloat16* __restrict__ idx_tk,
    __nv_bfloat16* __restrict__ idx_raw,
    int slot, int row0, int pos0, int len) {
    const int d = threadIdx.x;
    const int seg_end = pos0 + len;
    const int tail_g0 = (seg_end >> 2) << 2;
    if (tail_g0 >= seg_end) return;
    __nv_bfloat16* rslot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    for (int k = 0; k < 4; ++k) {
        const int p = tail_g0 + k;
        if (p >= pos0 && p < seg_end) rslot[k * 128 + d] = idx_tk[static_cast<size_t>(row0 + (p - pos0)) * 128 + d];
    }
}

// Device-driven small-M indexer update (handles both decode len=1 and DFly2 verify blocks len<=12!).
// Reads d_slot[m] and d_pos[m] solely from device memory -> 100% CUDA-Graph-capture-safe!
// Grid: 1, Block: 128 (exact IDX_DIM = 128).
template <bool IDX_FP8>
__global__ void __launch_bounds__(128) k_hy4_idx_k_update_smallm(
    const __nv_bfloat16* __restrict__ idx_tk,
    const float* __restrict__ kv_proj,
    const float* __restrict__ kg_proj,
    const float* __restrict__ ape,
    const float* __restrict__ k_norm_w,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    __nv_bfloat16* __restrict__ idx_raw,
    __nv_bfloat16* __restrict__ idx_bf16,
    uint8_t* __restrict__ idx_q,
    __half* __restrict__ idx_sc,
    int M, int max_pps) {
    const int d = threadIdx.x;
    __shared__ __nv_bfloat16 s_raw[4][128];
    __shared__ float s_norm[128];
    for (int m = 0; m < M; ++m) {
        const int slot = d_slot[m];
        const int pos  = d_pos[m];
        const int sub  = pos & 3;
        __nv_bfloat16* rslot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
        rslot[sub * 128 + d] = idx_tk[static_cast<size_t>(m) * 128 + d];
        __syncthreads();
        if (sub == 3) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) s_raw[k][d] = rslot[k * 128 + d];
            __syncthreads();
            const int g = pos >> 2;
            const int lpage = g >> 4;
            const int goff  = g & 15;
            const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + lpage];
            const size_t go = static_cast<size_t>(ppage) * 16 + goff;
            hy4_pool_4_keys_f32<IDX_FP8>(
                s_raw, kv_proj, kg_proj, ape, k_norm_w, g * 4,
                idx_bf16 ? (idx_bf16 + go * 128) : nullptr,
                idx_q    ? (idx_q    + go * 128) : nullptr,
                idx_sc   ? (idx_sc   + go * 4)   : nullptr,
                s_norm);
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(32) k_hy4_idx_q_rope(
    __nv_bfloat16* __restrict__ idx_q,
    const int* __restrict__ d_pos, int M) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::IDX_HEADS) return;
    const int m = idx / geom::IDX_HEADS;
    const int p = threadIdx.x;
    __nv_bfloat16* q = idx_q + static_cast<size_t>(idx) * 128 + 64;
    float x0 = __bfloat162float(q[2 * p]);
    float x1 = __bfloat162float(q[2 * p + 1]);
    hy4_rope_pair(x0, x1, d_pos[m], p, geom::ROPE_THETA_DSA);
    q[2 * p]     = __float2bfloat16(x0);
    q[2 * p + 1] = __float2bfloat16(x1);
}

// Tensor-Core Compressed Group Indexer Scorer (mma.m16n8k16):
//   16 indexer heads = M=16, 8 consecutive compressed groups = N=8, dim=128 = 8 K-steps!
//   Each warp keeps ALL 16 heads of idx_q[m] in registers and scores 8 groups in 8 MMA calls!
//   4 warps/CTA = 32 groups/CTA -> 4x-8x faster at 1M context!
// Grid: dim3((max_ng + 31) / 32, n_m), Block: 128 (4 warps).
template <bool IDX_FP8>
__global__ void __launch_bounds__(128) k_hy4_idx_score_tc(
    const __nv_bfloat16* __restrict__ idx_q,
    const float* __restrict__ idx_hw,
    const __nv_bfloat16* __restrict__ idx_bf16,
    const uint8_t* __restrict__ idx_pool_q,
    const __half* __restrict__ idx_pool_sc,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    float* __restrict__ idx_sc,
    int m_off, int n_m, int max_pps, int max_groups) {
    const int lm = blockIdx.y;
    if (lm >= n_m) return;
    const int m = m_off + lm;
    const int pos = d_pos[m];
    if (pos < geom::IDX_TOPK) return;
    const int n_earlier = pos >> 2;

    const int warp = threadIdx.x >> 5; // 0..3
    const int lane = threadIdx.x & 31;
    const int g0 = (blockIdx.x * 4 + warp) * 8;
    if (g0 >= n_earlier) return;

    const int rg = lane >> 2; // 0..7
    const int tc = lane & 3;  // 0..3
    const int slot = d_slot[m];
    const int gk = g0 + rg;
    const bool valid_k = (gk < n_earlier);

    const __nv_bfloat16* qm = idx_q + static_cast<size_t>(m) * geom::IDX_HEADS * 128;
    const float* hw = idx_hw + static_cast<size_t>(m) * geom::IDX_HEADS;
    constexpr float HW_SCALE = 0.02209708691207961f;
    const float w0 = hw[rg]     * HW_SCALE;
    const float w8 = hw[rg + 8] * HW_SCALE;

    // Resolve key row pointer for group gk (lane rg owns row rg of B[8, 128]!)
    size_t go = 0;
    if (valid_k) {
        const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + (gk >> 4)];
        go = static_cast<size_t>(ppage) * 16 + (gk & 15);
    }

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int kbase = ks * 16;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(qm + rg * 128       + kbase + 2 * tc);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(qm + (rg + 8) * 128 + kbase + 2 * tc);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(qm + rg * 128       + kbase + 2 * tc + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(qm + (rg + 8) * 128 + kbase + 2 * tc + 8);

        uint32_t b0 = 0u, b1 = 0u;
        if (valid_k) {
            if constexpr (!IDX_FP8) {
                const __nv_bfloat16* kp = idx_bf16 + go * 128 + kbase;
                b0 = *reinterpret_cast<const uint32_t*>(kp + 2 * tc);
                b1 = *reinterpret_cast<const uint32_t*>(kp + 2 * tc + 8);
            } else {
                const float sc = __half2float(idx_pool_sc[go * 4 + (ks >> 1)]);
                const uint8_t* qp = idx_pool_q + go * 128 + kbase;
                const uint16_t raw0 = *reinterpret_cast<const uint16_t*>(qp + 2 * tc);
                const uint16_t raw1 = *reinterpret_cast<const uint16_t*>(qp + 2 * tc + 8);
                uint32_t h0, h1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(raw0));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(raw1));
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
                b0 = pack_bf2(f0.x * sc, f0.y * sc);
                b1 = pack_bf2(f1.x * sc, f1.y * sc);
            }
        }
        mma16816_bf16(d, a0, a1, a2, a3, b0, b1);
    }

    // d[0]=dot(h=rg, n=2*tc), d[1]=dot(h=rg, n=2*tc+1), d[2]=dot(h=rg+8, n=2*tc), d[3]=dot(h=rg+8, n=2*tc+1)
    float s0 = fmaf(w0, fmaxf(d[0], 0.0f), w8 * fmaxf(d[2], 0.0f));
    float s1 = fmaf(w0, fmaxf(d[1], 0.0f), w8 * fmaxf(d[3], 0.0f));
    // Reduce across rg = 0..7 (stride 4, 8, 16 in lane!)
    s0 += __shfl_xor_sync(0xffffffffu, s0, 4);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
    if (rg == 0) {
        const int g_even = g0 + 2 * tc;
        if (g_even < n_earlier)     idx_sc[static_cast<size_t>(lm) * max_groups + g_even]     = s0;
        if (g_even + 1 < n_earlier) idx_sc[static_cast<size_t>(lm) * max_groups + g_even + 1] = s1;
    }
}

__device__ __forceinline__ uint32_t hy4_score_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(256) k_hy4_idx_topk511(
    const float* __restrict__ idx_sc,
    const int* __restrict__ d_pos,
    int* __restrict__ sel_grp,
    int* __restrict__ sel_cnt,
    int m_off, int n_m, int max_groups) {
    const int lm = blockIdx.x;
    if (lm >= n_m) return;
    const int m = m_off + lm;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int pos = d_pos[m];
    constexpr int KSEL = geom::IDX_GROUP_TOPK - 1; // 511

    if (pos < geom::IDX_TOPK) {
        if (tid == 0) sel_cnt[m] = -1;
        return;
    }
    const int nblk = pos >> 2;
    int* dst = sel_grp + static_cast<size_t>(m) * KSEL;
    if (nblk <= KSEL) {
        for (int b = tid; b < nblk; b += 256) dst[b] = b;
        if (tid == 0) sel_cnt[m] = nblk;
        return;
    }

    const float* sc = idx_sc + static_cast<size_t>(lm) * max_groups;
    __shared__ int hist[2048];
    __shared__ int w_tot[8];
    __shared__ int s_p1, s_need1, s_cnt1, s_p2, s_need2, s_cnt2, s_eq_need, s_out_pos, s_eq_used;
    __shared__ uint32_t s_strict_thr, s_eq_thr;

    for (int i = tid; i < 2048; i += 256) hist[i] = 0;
    __syncthreads();
    for (int b = tid; b < nblk; b += 256) atomicAdd(&hist[(hy4_score_key(sc[b]) >> 21) & 2047], 1);
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
        for (int b = tid; b < nblk; b += 256) {
            const uint32_t u = hy4_score_key(sc[b]);
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
            for (int b = tid; b < nblk; b += 256) {
                const uint32_t u = hy4_score_key(sc[b]);
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

    for (int base = 0; base < nblk; base += 256) {
        if (s_out_pos >= KSEL) break;
        const int b = base + tid;
        const bool valid = (b < nblk);
        const uint32_t u = valid ? hy4_score_key(sc[b]) : 0u;
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

// Per-slot isolated DFly2 index snapshot driven by device array d_snap_dst[M] (>= 0 selects destination slot*W + off).
// 100% CUDA-Graph-capture-safe AND completely prevents cross-slot clobbering during multi-session serving!
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_idx_snapshot_dst(
    const int* __restrict__ sel_grp,
    const int* __restrict__ sel_cnt,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_snap_dst,
    int* __restrict__ dst_grp, // [slots * dfly_w_max, 511]
    int* __restrict__ dst_cnt, // [slots * dfly_w_max]
    int* __restrict__ dst_pos, // [slots * dfly_w_max]
    int M, bool sparse_used) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int di = d_snap_dst[m];
    if (di < 0) return;
    const int tid = threadIdx.x;
    constexpr int KSEL = geom::IDX_GROUP_TOPK - 1;
    if (!sparse_used) {
        if (tid == 0) { dst_cnt[di] = -1; dst_pos[di] = d_pos[m]; }
        return;
    }
    const int cnt = sel_cnt[m];
    if (tid == 0) { dst_cnt[di] = cnt; dst_pos[di] = d_pos[m]; }
    if (cnt > 0) {
        const int* src = sel_grp + static_cast<size_t>(m) * KSEL;
        int* dst = dst_grp + static_cast<size_t>(di) * KSEL;
        for (int i = tid; i < cnt; i += 256) dst[i] = src[i];
    }
}

// Per-slot isolated save of post-norm idx_tk rows for exact partial-accept idx_raw rollback.
// Grid: (M), Block: 128.
__global__ void __launch_bounds__(128) k_hy4_scatter_idx_tk(
    const __nv_bfloat16* __restrict__ idx_tk,
    const int* __restrict__ d_snap_dst,
    __nv_bfloat16* __restrict__ dst_tk,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int di = d_snap_dst[m];
    if (di < 0) return;
    const int d = threadIdx.x;
    dst_tk[static_cast<size_t>(di) * geom::IDX_DIM + d] = idx_tk[static_cast<size_t>(m) * geom::IDX_DIM + d];
}

// =================================================================================================
// Helper: Load 1 KV token (576 BF16) from BF16 / FP8 / INT4 pool into sKV[tk][0..575] BF16!
// Cooperatively executed by 256 threads loading `nk` tokens (nk <= 32).
// =================================================================================================
template <bool IS_DSA, KvFmt FMT>
__device__ __forceinline__ void hy4_load_kv_tile(
    __nv_bfloat16 (*sKV)[584],
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_page_table,
    int slot, int max_pps,
    const int* __restrict__ sg, int nsel, int tail_start, int win_start,
    int k0, int nk, int tid) {
    if constexpr (!IS_DSA || FMT == KvFmt::BF16) {
        const int total_u4 = nk * 72; // 576 bf16 = 72 uint4
        for (int idx4 = tid; idx4 < total_u4; idx4 += 256) {
            const int tk = idx4 / 72;
            const int u4 = idx4 - tk * 72;
            const int ki = k0 + tk;
            int tpos;
            if constexpr (IS_DSA) {
                if (nsel < 0) tpos = ki;
                else if (ki < nsel * 4) tpos = sg[ki >> 2] * 4 + (ki & 3);
                else tpos = tail_start + (ki - nsel * 4);
            } else {
                tpos = win_start + ki;
            }
            const __nv_bfloat16* src;
            if constexpr (IS_DSA) {
                const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE)];
                src = reinterpret_cast<const __nv_bfloat16*>(kv_mem) +
                      (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * geom::KVA;
            } else {
                src = reinterpret_cast<const __nv_bfloat16*>(kv_mem) +
                      (static_cast<size_t>(slot) * geom::SWA_RING + (tpos & (geom::SWA_RING - 1))) * geom::KVA;
            }
            *reinterpret_cast<uint4*>(sKV[tk] + u4 * 8) = reinterpret_cast<const uint4*>(src)[u4];
        }
    } else if constexpr (FMT == KvFmt::FP8) {
        // Each token has 16 blocks of 32 nope FP8 + 8 uint4 rope BF16 = 24 work units per token
        // 256 threads: 8 threads per token (or loop over tokens * 32 chunks of 16 nope + 8 rope)
        // Even cleaner: 1 warp per token (8 warps process 8 tokens per pass, 4 passes for nk=32)!
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int tk = warp; tk < nk; tk += 8) {
            const int ki = k0 + tk;
            int tpos;
            if (nsel < 0) tpos = ki;
            else if (ki < nsel * 4) tpos = sg[ki >> 2] * 4 + (ki & 3);
            else tpos = tail_start + (ki - nsel * 4);
            const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE)];
            const uint8_t* src = kv_mem + (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * KV_FP8_STRIDE;
            const __half* scs = reinterpret_cast<const __half*>(src + 512);
            // 32 lanes load 16 bytes (16 FP8) each = 512 FP8 exact! lane>>1 = block 0..15
            const uint4 qv = reinterpret_cast<const uint4*>(src)[lane];
            const float sc = __half2float(scs[lane >> 1]);
            float2 f[8];
            mx8::fp8x4_unpack(qv.x, f[0], f[1]);
            mx8::fp8x4_unpack(qv.y, f[2], f[3]);
            mx8::fp8x4_unpack(qv.z, f[4], f[5]);
            mx8::fp8x4_unpack(qv.w, f[6], f[7]);
            uint4 o0, o1;
            o0.x = pack_bf2(f[0].x * sc, f[0].y * sc);
            o0.y = pack_bf2(f[1].x * sc, f[1].y * sc);
            o0.z = pack_bf2(f[2].x * sc, f[2].y * sc);
            o0.w = pack_bf2(f[3].x * sc, f[3].y * sc);
            o1.x = pack_bf2(f[4].x * sc, f[4].y * sc);
            o1.y = pack_bf2(f[5].x * sc, f[5].y * sc);
            o1.z = pack_bf2(f[6].x * sc, f[6].y * sc);
            o1.w = pack_bf2(f[7].x * sc, f[7].y * sc);
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16)     = o0;
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16 + 8) = o1;
            // Rope 64 BF16 = 32 uint32s -> 1 uint32 per lane!
            *reinterpret_cast<uint32_t*>(sKV[tk] + 512 + lane * 2) =
                reinterpret_cast<const uint32_t*>(src + 544)[lane];
        }
    } else { // KvFmt::INT4
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int tk = warp; tk < nk; tk += 8) {
            const int ki = k0 + tk;
            int tpos;
            if (nsel < 0) tpos = ki;
            else if (ki < nsel * 4) tpos = sg[ki >> 2] * 4 + (ki & 3);
            else tpos = tail_start + (ki - nsel * 4);
            const int ppage = d_page_table[static_cast<size_t>(slot) * max_pps + (tpos / geom::KV_PAGE)];
            const uint8_t* src = kv_mem + (static_cast<size_t>(ppage) * geom::KV_PAGE + (tpos % geom::KV_PAGE)) * KV_INT4_STRIDE;
            const __half* scs = reinterpret_cast<const __half*>(src + 256);
            // 32 lanes load 8 bytes (16 packed INT4) each = 512 INT4 exact!
            const uint2 raw = reinterpret_cast<const uint2*>(src)[lane];
            const float sc = __half2float(scs[lane >> 1]);
            auto unpack_u32 = [&](uint32_t w, uint4& dst) {
                uint32_t out[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t b = (w >> (8 * i)) & 0xffu;
                    const float x0 = static_cast<float>(static_cast<int>(b & 0xfu) - 8) * sc;
                    const float x1 = static_cast<float>(static_cast<int>(b >> 4) - 8) * sc;
                    out[i] = pack_bf2(x0, x1);
                }
                dst = make_uint4(out[0], out[1], out[2], out[3]);
            };
            uint4 o0, o1;
            unpack_u32(raw.x, o0);
            unpack_u32(raw.y, o1);
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16)     = o0;
            *reinterpret_cast<uint4*>(sKV[tk] + lane * 16 + 8) = o1;
            *reinterpret_cast<uint32_t*>(sKV[tk] + 512 + lane * 2) =
                reinterpret_cast<const uint32_t*>(src + 288)[lane];
        }
    }
}

// =================================================================================================
// Tensor-Core Absorbed MLA Decode / Verify / Sparse Kernel (mma.m16n8k16):
//   Packs 16 query heads into M=16, 32 KV tokens into 4 x N=8!
//   8 warps split D=512 (64 dims/warp = 8 m16n8 tiles); warps 0..3 also own 16 rope dims each!
//   ZERO Q reloads, 16x KV reuse per CTA, native BF16/FP8/INT4 KV dequant on load!
// Grid: (M * (nh / 16) * NS), Block: 256 (8 warps).
// =================================================================================================
template <bool IS_DSA, KvFmt FMT>
__global__ void __launch_bounds__(256) k_hy4_mla_decode_tc(
    const __nv_bfloat16* __restrict__ q_full,  // [M, nh, 576]
    const float* __restrict__ sinks,           // [nh] FP32
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const int* __restrict__ d_page_table,
    const int* __restrict__ sel_grp,
    const int* __restrict__ sel_cnt,
    __nv_bfloat16* __restrict__ u_h,           // [nh, M, 512] (NS == 1)
    float* __restrict__ split_m,               // [M, NS, nh] (NS > 1)
    float* __restrict__ split_l,               // [M, NS, nh] (NS > 1)
    float* __restrict__ split_o,               // [M, NS, nh, 512] (NS > 1)
    int M, int nh, int max_pps, int NS) {
    const int n_hg = nh >> 4; // nh is always 16, 32, or 64!
    const int idx = blockIdx.x;
    const int sp = idx % NS;
    const int rem = idx / NS;
    const int hg = rem % n_hg;
    const int m  = rem / n_hg;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5; // 0..7 -> owns nope dims [warp*64 .. warp*64+63]
    const int lane = tid & 31;
    const int rg = lane >> 2;  // 0..7
    const int tc = lane & 3;   // 0..3
    const int h0 = hg * 16;

    const int slot = d_slot[m];
    const int pos  = d_pos[m];

    int total_tok = 0, win_start = 0, nsel = -1, tail_start = 0;
    const int* sg = nullptr;
    if constexpr (IS_DSA) {
        nsel = sel_cnt ? sel_cnt[m] : -1;
        if (nsel < 0) {
            total_tok = pos + 1;
        } else {
            tail_start = (pos >> 2) << 2;
            total_tok = nsel * 4 + (pos + 1 - tail_start);
            sg = sel_grp + static_cast<size_t>(m) * (geom::IDX_GROUP_TOPK - 1);
        }
    } else {
        win_start = max(0, pos - geom::SWA_WINDOW + 1);
        total_tok = pos - win_start + 1;
    }

    const int per = (total_tok + NS - 1) / NS;
    const int k_start = sp * per;
    const int k_end = min(total_tok, k_start + per);

    // Preload this warp's slice of Q[16, 64] (+ Q_rope[16, 16] for warps 0..3) into registers!
    const __nv_bfloat16* qp_a = q_full + (static_cast<size_t>(m) * nh + h0 + rg) * geom::KVA;
    const __nv_bfloat16* qp_b = q_full + (static_cast<size_t>(m) * nh + h0 + rg + 8) * geom::KVA;
    uint32_t qn[4][4]; // 4 K-steps of 16 = 64 dims
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

    // Online softmax state for head (h0 + rg) and head (h0 + rg + 8)
    float m_a = (sp == 0) ? sinks[h0 + rg]     : -INFINITY;
    float m_b = (sp == 0) ? sinks[h0 + rg + 8] : -INFINITY;
    float l_a = (sp == 0) ? 1.0f : 0.0f;
    float l_b = (sp == 0) ? 1.0f : 0.0f;

    float o_acc[8][4]; // 8 N-tiles of 8 = 64 output dims for this warp!
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 (*sKV)[584] = reinterpret_cast<__nv_bfloat16 (*)[584]>(sm_raw); // 32 * 584 * 2 = 37,376 B
    float (*sPart)[16][32] = reinterpret_cast<float (*)[16][32]>(sm_raw + 32 * 584 * 2); // 8 * 16 * 32 * 4 = 16,384 B

    constexpr int TILE_K = 32;
    constexpr float ATTN_SCALE = 0.0625f; // 256^-0.5

    for (int k0 = k_start; k0 < k_end; k0 += TILE_K) {
        const int nk = min(TILE_K, k_end - k0);
        hy4_load_kv_tile<IS_DSA, FMT>(sKV, kv_mem, d_page_table, slot, max_pps, sg, nsel, tail_start, win_start, k0, nk, tid);
        __syncthreads();

        // 1. Each warp computes partial S[16, 32] across its 64 nope dims (+ 16 rope dims for warps 0..3)
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

        // 2. Sum partials across 8 warps, scale, online softmax -> direct MMA A-fragments p_frag[2][4]!
        float sc_a[8], sc_b[8];
        float tmax_a = -INFINITY, tmax_b = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int c0 = kt * 8 + 2 * tc;
            const int c1 = c0 + 1;
            float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sa0 += sPart[w][rg][c0];
                sa1 += sPart[w][rg][c1];
                sb0 += sPart[w][rg + 8][c0];
                sb1 += sPart[w][rg + 8][c1];
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
            l_tile_a += pa[i];
            l_tile_b += pb[i];
        }
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 1);
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 2);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 1);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 2);

        l_a = l_a * alpha_a + l_tile_a;
        l_b = l_b * alpha_b + l_tile_b;
        m_a = m_new_a;
        m_b = m_new_b;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha_a; o_acc[nt][1] *= alpha_a;
            o_acc[nt][2] *= alpha_b; o_acc[nt][3] *= alpha_b;
        }

        // 3. P[16, 32] @ sKV[32, warp*64..+63] -> 2 K-steps of 16 x 8 N-tiles of 8!
        const uint32_t pf0[4] = { pack_bf2(pa[0], pa[1]), pack_bf2(pb[0], pb[1]), pack_bf2(pa[2], pa[3]), pack_bf2(pb[2], pb[3]) };
        const uint32_t pf1[4] = { pack_bf2(pa[4], pa[5]), pack_bf2(pb[4], pb[5]), pack_bf2(pa[6], pa[7]), pack_bf2(pb[6], pb[7]) };

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dcol = warp * 64 + nt * 8 + rg;
            const int k00 = 2 * tc, k01 = k00 + 1, k02 = k00 + 8, k03 = k02 + 1;
            const uint32_t vb0 = pack_bf2(
                k00 < nk ? __bfloat162float(sKV[k00][dcol]) : 0.0f,
                k01 < nk ? __bfloat162float(sKV[k01][dcol]) : 0.0f);
            const uint32_t vb1 = pack_bf2(
                k02 < nk ? __bfloat162float(sKV[k02][dcol]) : 0.0f,
                k03 < nk ? __bfloat162float(sKV[k03][dcol]) : 0.0f);
            mma16816_bf16(o_acc[nt], pf0[0], pf0[1], pf0[2], pf0[3], vb0, vb1);

            if (nk > 16) {
                const int k10 = 16 + 2 * tc, k11 = k10 + 1, k12 = k10 + 8, k13 = k12 + 1;
                const uint32_t vb2 = pack_bf2(
                    k10 < nk ? __bfloat162float(sKV[k10][dcol]) : 0.0f,
                    k11 < nk ? __bfloat162float(sKV[k11][dcol]) : 0.0f);
                const uint32_t vb3 = pack_bf2(
                    k12 < nk ? __bfloat162float(sKV[k12][dcol]) : 0.0f,
                    k13 < nk ? __bfloat162float(sKV[k13][dcol]) : 0.0f);
                mma16816_bf16(o_acc[nt], pf1[0], pf1[1], pf1[2], pf1[3], vb2, vb3);
            }
        }
        __syncthreads();
    }

    // Write output
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
            soa[nt * 8 + 2 * tc]     = o_acc[nt][0];
            soa[nt * 8 + 2 * tc + 1] = o_acc[nt][1];
            sob[nt * 8 + 2 * tc]     = o_acc[nt][2];
            sob[nt * 8 + 2 * tc + 1] = o_acc[nt][3];
        }
    }
}

// =================================================================================================
// Tensor-Core Absorbed MLA Prefill Kernel (mma.m16n8k16):
//   For contiguous segments (ALL 30 SWA layers + dense DSA layers when pos < 2048):
//   1 CTA = (16 query tokens, 1 head), 8 warps split D=512 (64 dims/warp), TILE_K = 32 KV tokens!
//   16x fewer KV memory reads + full BF16 Tensor Core throughput!
// Grid: dim3((seg_len + 15) / 16, nh), Block: 256 (8 warps).
// =================================================================================================
template <bool IS_DSA, KvFmt FMT>
__global__ void __launch_bounds__(256) k_hy4_mla_prefill_tc(
    const __nv_bfloat16* __restrict__ q_full, // [M, nh, 576]
    const float* __restrict__ sinks,          // [nh] FP32
    const uint8_t* __restrict__ kv_mem,
    const int* __restrict__ d_page_table,
    __nv_bfloat16* __restrict__ u_h,          // [nh, M, 512]
    int M, int nh, int max_pps, int slot, int row0, int pos0, int seg_len) {
    const int qb = blockIdx.x;
    const int h  = blockIdx.y;
    const int r_base = qb * 16;
    if (r_base >= seg_len || h >= nh) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg = lane >> 2; // 0..7
    const int tc = lane & 3;  // 0..3

    const int ra = r_base + rg;
    const int rb = ra + 8;
    const bool va = (ra < seg_len);
    const bool vb = (rb < seg_len);
    const int pa = pos0 + ra;
    const int pb = pos0 + rb;

    const int p_first = pos0 + r_base;
    const int p_last  = pos0 + min(seg_len, r_base + 16) - 1;
    const int k_start = IS_DSA ? 0 : max(0, p_first - geom::SWA_WINDOW + 1);
    const int k_end   = p_last + 1;

    // Preload Q[16, 64] (+ Q_rope[16, 16] for warps 0..3)
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
    float m_a = sink_h, m_b = sink_h;
    float l_a = 1.0f,   l_b = 1.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 (*sKV)[584] = reinterpret_cast<__nv_bfloat16 (*)[584]>(sm_raw);
    float (*sPart)[16][32] = reinterpret_cast<float (*)[16][32]>(sm_raw + 32 * 584 * 2);

    constexpr int TILE_K = 32;
    constexpr float ATTN_SCALE = 0.0625f;
    const int lo_a = IS_DSA ? 0 : max(0, pa - geom::SWA_WINDOW + 1);
    const int lo_b = IS_DSA ? 0 : max(0, pb - geom::SWA_WINDOW + 1);

    for (int k0 = k_start; k0 < k_end; k0 += TILE_K) {
        const int nk = min(TILE_K, k_end - k0);
        hy4_load_kv_tile<IS_DSA, FMT>(sKV, kv_mem, d_page_table, slot, max_pps, nullptr, -1, 0, 0, k0, nk, tid);
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
            const int c0 = kt * 8 + 2 * tc;
            const int c1 = c0 + 1;
            const int kp0 = k0 + c0, kp1 = k0 + c1;
            float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sa0 += sPart[w][rg][c0];     sa1 += sPart[w][rg][c1];
                sb0 += sPart[w][rg + 8][c0]; sb1 += sPart[w][rg + 8][c1];
            }
            const bool oka0 = va && (c0 < nk) && (kp0 >= lo_a) && (kp0 <= pa);
            const bool oka1 = va && (c1 < nk) && (kp1 >= lo_a) && (kp1 <= pa);
            const bool okb0 = vb && (c0 < nk) && (kp0 >= lo_b) && (kp0 <= pb);
            const bool okb1 = vb && (c1 < nk) && (kp1 >= lo_b) && (kp1 <= pb);
            sc_a[2 * kt]     = oka0 ? (sa0 * ATTN_SCALE) : -INFINITY;
            sc_a[2 * kt + 1] = oka1 ? (sa1 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt]     = okb0 ? (sb0 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt + 1] = okb1 ? (sb1 * ATTN_SCALE) : -INFINITY;
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

        float l_tile_a = 0.0f, l_tile_b = 0.0f;
        float pa_v[8], pb_v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa_v[i] = (sc_a[i] > -1e30f) ? __expf(sc_a[i] - m_new_a) : 0.0f;
            pb_v[i] = (sc_b[i] > -1e30f) ? __expf(sc_b[i] - m_new_b) : 0.0f;
            l_tile_a += pa_v[i];
            l_tile_b += pb_v[i];
        }
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 1);
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 2);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 1);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 2);

        l_a = l_a * alpha_a + l_tile_a;
        l_b = l_b * alpha_b + l_tile_b;
        m_a = m_new_a;
        m_b = m_new_b;

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
            const uint32_t vb0 = pack_bf2(
                k00 < nk ? __bfloat162float(sKV[k00][dcol]) : 0.0f,
                k01 < nk ? __bfloat162float(sKV[k01][dcol]) : 0.0f);
            const uint32_t vb1 = pack_bf2(
                k02 < nk ? __bfloat162float(sKV[k02][dcol]) : 0.0f,
                k03 < nk ? __bfloat162float(sKV[k03][dcol]) : 0.0f);
            mma16816_bf16(o_acc[nt], pf0[0], pf0[1], pf0[2], pf0[3], vb0, vb1);

            if (nk > 16) {
                const int k10 = 16 + 2 * tc, k11 = k10 + 1, k12 = k10 + 8, k13 = k12 + 1;
                const uint32_t vb2 = pack_bf2(
                    k10 < nk ? __bfloat162float(sKV[k10][dcol]) : 0.0f,
                    k11 < nk ? __bfloat162float(sKV[k11][dcol]) : 0.0f);
                const uint32_t vb3 = pack_bf2(
                    k12 < nk ? __bfloat162float(sKV[k12][dcol]) : 0.0f,
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
        for (int nt = 0; nt < 8; ++nt) {
            *reinterpret_cast<uint32_t*>(da + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][0] * inv_a, o_acc[nt][1] * inv_a);
        }
    }
    if (vb) {
        __nv_bfloat16* db = u_h + (static_cast<size_t>(h) * M + row0 + rb) * geom::KV_LORA + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            *reinterpret_cast<uint32_t*>(db + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][2] * inv_b, o_acc[nt][3] * inv_b);
        }
    }
}

constexpr int MLA_TC_SMEM = 32 * 584 * 2 + 8 * 16 * 32 * 4; // 53,760 B

inline void ensure_mla_tc_smem() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_hy4_mla_decode_tc<true,  KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_decode_tc<true,  KvFmt::FP8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_decode_tc<true,  KvFmt::INT4>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_decode_tc<false, KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_prefill_tc<true,  KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_prefill_tc<true,  KvFmt::FP8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_prefill_tc<true,  KvFmt::INT4>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
    cudaFuncSetAttribute(k_hy4_mla_prefill_tc<false, KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_TC_SMEM);
}

__global__ void __launch_bounds__(256) k_hy4_mla_split_reduce(
    const float* __restrict__ split_m,
    const float* __restrict__ split_l,
    const float* __restrict__ split_o,
    __nv_bfloat16* __restrict__ u_h,
    int M, int nh, int NS) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int tid = threadIdx.x;

    float gm = -INFINITY;
    for (int sp = 0; sp < NS; ++sp) {
        gm = fmaxf(gm, split_m[(static_cast<size_t>(m) * NS + sp) * nh + h]);
    }
    float gl = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int sp = 0; sp < NS; ++sp) {
        const size_t sidx = (static_cast<size_t>(m) * NS + sp) * nh + h;
        const float sl = split_l[sidx];
        if (sl <= 0.0f) continue;
        const float w = __expf(split_m[sidx] - gm);
        gl += sl * w;
        const float* so = split_o + sidx * geom::KV_LORA;
        o0 = fmaf(so[tid],       w, o0);
        o1 = fmaf(so[tid + 256], w, o1);
    }
    const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    __nv_bfloat16* dst = u_h + (static_cast<size_t>(h) * M + m) * geom::KV_LORA;
    dst[tid]       = __float2bfloat16(o0 * inv_l);
    dst[tid + 256] = __float2bfloat16(o1 * inv_l);
}

__global__ void __launch_bounds__(256) k_hy4_attn_gate_transpose(
    const __nv_bfloat16* __restrict__ v_h,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ attn_out,
    int M, int nh) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int d = threadIdx.x;
    const float v = __bfloat162float(v_h[(static_cast<size_t>(h) * M + m) * geom::V_HEAD + d]);
    const float g = __bfloat162float(gate[static_cast<size_t>(idx) * geom::V_HEAD + d]);
    const float sg = __bfloat162float(__float2bfloat16(sigmoid_f(g)));
    attn_out[static_cast<size_t>(idx) * geom::V_HEAD + d] = __float2bfloat16(v * sg);
}

// =================================================================================================
// Fused Small-M Absorbed Q + RoPE (BF16 and MXFP8!)
// =================================================================================================
template <int MAX_M>
__global__ void __launch_bounds__(128) k_hy4_q_absorb_rope_smallm(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ w_uk_t,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh, float theta) {
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
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const __nv_bfloat16* src = q_raw + (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD + geom::QK_NOPE;
                float x0 = __bfloat162float(src[2 * lane]);
                float x1 = __bfloat162float(src[2 * lane + 1]);
                hy4_rope_pair(x0, x1, d_pos[m], lane, theta);
                __nv_bfloat16* dst = q_full + (static_cast<size_t>(m) * nh + h) * geom::KVA + geom::KV_LORA;
                dst[2 * lane]     = __float2bfloat16(x0);
                dst[2 * lane + 1] = __float2bfloat16(x1);
            }
        }
    }
}

// MXFP8 Fused Small-M Absorbed Q + RoPE: K=192 = 12 uint4s of FP8 (6 blocks of 32)! 12 lanes/row (16-lane half-warp)!
template <int MAX_M>
__global__ void __launch_bounds__(128) k_hy4_q_absorb_rope_mx8(
    const __nv_bfloat16* __restrict__ q_raw,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wsc,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ q_full,
    int M, int nh, float theta) {
    const int h = blockIdx.x;
    const int hwarp = threadIdx.x >> 4; // 0..7 (8 rows/CTA -> grid.y = 64!)
    const int hlane = threadIdx.x & 15; // 0..15 (lanes 0..11 own 16 FP8 each = 192 exact!)
    const int c = blockIdx.y * 8 + hwarp;
    if (h >= nh || c >= geom::KV_LORA) return;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if (hlane < 12) {
        const size_t row_id = static_cast<size_t>(h) * geom::KV_LORA + c;
        const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(wq + row_id * geom::QK_NOPE) + hlane);
        const float sc = __uint_as_float(static_cast<uint32_t>(__ldcs(wsc + row_id * 6 + (hlane >> 1))) << 23);
        float2 wf[8];
        mx8::fp8x4_unpack(wv.x, wf[0], wf[1]);
        mx8::fp8x4_unpack(wv.y, wf[2], wf[3]);
        mx8::fp8x4_unpack(wv.z, wf[4], wf[5]);
        mx8::fp8x4_unpack(wv.w, wf[6], wf[7]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* qm = reinterpret_cast<const uint4*>(
                    q_raw + (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD);
                acc[m] = sc * mx8::dot_bf16x16(__ldg(qm + 2 * hlane), __ldg(qm + 2 * hlane + 1), wf);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = mx8::h16_sum(acc[m]);
            if (hlane == 0) q_full[(static_cast<size_t>(m) * nh + h) * geom::KVA + c] = __float2bfloat16(s);
        }
    }
    if (blockIdx.y == 0 && threadIdx.x < 32) {
        const int lane = threadIdx.x;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const __nv_bfloat16* src = q_raw + (static_cast<size_t>(m) * nh + h) * geom::QK_HEAD + geom::QK_NOPE;
                float x0 = __bfloat162float(src[2 * lane]);
                float x1 = __bfloat162float(src[2 * lane + 1]);
                hy4_rope_pair(x0, x1, d_pos[m], lane, theta);
                __nv_bfloat16* dst = q_full + (static_cast<size_t>(m) * nh + h) * geom::KVA + geom::KV_LORA;
                dst[2 * lane]     = __float2bfloat16(x0);
                dst[2 * lane + 1] = __float2bfloat16(x1);
            }
        }
    }
}

inline void launch_q_absorb_rope_smallm(const __nv_bfloat16* q_raw, const mx8::LinW& w_uk_t,
                                        const int* d_pos, __nv_bfloat16* q_full,
                                        int M, int nh, float theta, cudaStream_t st) {
    for (int m0 = 0; m0 < M; m0 += 12) {
        const int nm = std::min(12, M - m0);
        const __nv_bfloat16* qr = q_raw  + static_cast<size_t>(m0) * nh * geom::QK_HEAD;
        const int*           dp = d_pos  + m0;
        __nv_bfloat16*       qf = q_full + static_cast<size_t>(m0) * nh * geom::KVA;
        if (w_uk_t.is_mx8()) {
            const dim3 grid(nh, geom::KV_LORA / 8);
            if (nm <= 1)      k_hy4_q_absorb_rope_mx8<1> <<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
            else if (nm <= 2) k_hy4_q_absorb_rope_mx8<2> <<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
            else if (nm <= 4) k_hy4_q_absorb_rope_mx8<4> <<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
            else if (nm <= 6) k_hy4_q_absorb_rope_mx8<6> <<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
            else if (nm <= 8) k_hy4_q_absorb_rope_mx8<8> <<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
            else              k_hy4_q_absorb_rope_mx8<12><<<grid, 128, 0, st>>>(qr, w_uk_t.mx8.q, w_uk_t.mx8.sc, dp, qf, nm, nh, theta);
        } else {
            const dim3 grid(nh, geom::KV_LORA / 4);
            if (nm <= 1)      k_hy4_q_absorb_rope_smallm<1> <<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
            else if (nm <= 2) k_hy4_q_absorb_rope_smallm<2> <<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
            else if (nm <= 4) k_hy4_q_absorb_rope_smallm<4> <<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
            else if (nm <= 6) k_hy4_q_absorb_rope_smallm<6> <<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
            else if (nm <= 8) k_hy4_q_absorb_rope_smallm<8> <<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
            else              k_hy4_q_absorb_rope_smallm<12><<<grid, 128, 0, st>>>(qr, w_uk_t.bf16, dp, qf, nm, nh, theta);
        }
    }
}

// =================================================================================================
// Fused Small-M V-Expand + Sigmoid Gate + Transpose (BF16 and MXFP8!)
// =================================================================================================
template <int MAX_M>
__global__ void __launch_bounds__(128) k_hy4_v_expand_gate_smallm(
    const __nv_bfloat16* __restrict__ u_h,
    const __nv_bfloat16* __restrict__ w_uv,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ attn_out,
    int M_stride, int m0, int nm, int nh) {
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = blockIdx.y * 4 + warp;
    if (h >= nh || d >= geom::V_HEAD) return;

    const uint4* wrow = reinterpret_cast<const uint4*>(
        w_uv + (static_cast<size_t>(h) * geom::V_HEAD + d) * geom::KV_LORA);
    const uint4 b0 = __ldg(wrow + lane);
    const uint4 b1 = __ldg(wrow + lane + 32);

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < nm) {
            const uint4* um = reinterpret_cast<const uint4*>(
                u_h + (static_cast<size_t>(h) * M_stride + (m0 + m)) * geom::KV_LORA);
            acc[m] = dot_bf16x8(um[lane], b0) + dot_bf16x8(um[lane + 32], b1);
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < nm) {
            const float s = warp_sum(acc[m]);
            if (lane == 0) {
                const size_t oidx = (static_cast<size_t>(m0 + m) * nh + h) * geom::V_HEAD + d;
                const float vbf = __bfloat162float(__float2bfloat16(s));
                const float g = __bfloat162float(gate[oidx]);
                const float sg = __bfloat162float(__float2bfloat16(sigmoid_f(g)));
                attn_out[oidx] = __float2bfloat16(vbf * sg);
            }
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_hy4_v_expand_gate_mx8(
    const __nv_bfloat16* __restrict__ u_h,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wsc,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ attn_out,
    int M_stride, int m0, int nm, int nh) {
    const int h = blockIdx.x;
    const int hwarp = threadIdx.x >> 4; // 0..7 -> grid.y = 32
    const int hlane = threadIdx.x & 15; // 16 lanes x 2 passes x 16 FP8 = 512 exact!
    const int d = blockIdx.y * 8 + hwarp;
    if (h >= nh || d >= geom::V_HEAD) return;

    const size_t row_id = static_cast<size_t>(h) * geom::V_HEAD + d;
    const uint4* qrow = reinterpret_cast<const uint4*>(wq + row_id * geom::KV_LORA);
    const uint8_t* srow = wsc + row_id * 16;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int i = pass * 16 + hlane;
        const uint4 wv = __ldcs(qrow + i);
        const float sc = __uint_as_float(static_cast<uint32_t>(__ldcs(srow + (i >> 1))) << 23);
        float2 wf[8];
        mx8::fp8x4_unpack(wv.x, wf[0], wf[1]);
        mx8::fp8x4_unpack(wv.y, wf[2], wf[3]);
        mx8::fp8x4_unpack(wv.z, wf[4], wf[5]);
        mx8::fp8x4_unpack(wv.w, wf[6], wf[7]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < nm) {
                const uint4* um = reinterpret_cast<const uint4*>(
                    u_h + (static_cast<size_t>(h) * M_stride + (m0 + m)) * geom::KV_LORA);
                acc[m] = fmaf(sc, mx8::dot_bf16x16(__ldg(um + 2 * i), __ldg(um + 2 * i + 1), wf), acc[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < nm) {
            const float s = mx8::h16_sum(acc[m]);
            if (hlane == 0) {
                const size_t oidx = (static_cast<size_t>(m0 + m) * nh + h) * geom::V_HEAD + d;
                const float vbf = __bfloat162float(__float2bfloat16(s));
                const float g = __bfloat162float(gate[oidx]);
                const float sg = __bfloat162float(__float2bfloat16(sigmoid_f(g)));
                attn_out[oidx] = __float2bfloat16(vbf * sg);
            }
        }
    }
}

inline void launch_v_expand_gate_smallm(const __nv_bfloat16* u_h, const mx8::LinW& w_uv,
                                        const __nv_bfloat16* gate, __nv_bfloat16* attn_out,
                                        int M, int nh, cudaStream_t st) {
    for (int m0 = 0; m0 < M; m0 += 12) {
        const int nm = std::min(12, M - m0);
        if (w_uv.is_mx8()) {
            const dim3 grid(nh, geom::V_HEAD / 8);
            if (nm <= 1)      k_hy4_v_expand_gate_mx8<1> <<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 2) k_hy4_v_expand_gate_mx8<2> <<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 4) k_hy4_v_expand_gate_mx8<4> <<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 6) k_hy4_v_expand_gate_mx8<6> <<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 8) k_hy4_v_expand_gate_mx8<8> <<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
            else              k_hy4_v_expand_gate_mx8<12><<<grid, 128, 0, st>>>(u_h, w_uv.mx8.q, w_uv.mx8.sc, gate, attn_out, M, m0, nm, nh);
        } else {
            const dim3 grid(nh, geom::V_HEAD / 4);
            if (nm <= 1)      k_hy4_v_expand_gate_smallm<1> <<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 2) k_hy4_v_expand_gate_smallm<2> <<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 4) k_hy4_v_expand_gate_smallm<4> <<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 6) k_hy4_v_expand_gate_smallm<6> <<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
            else if (nm <= 8) k_hy4_v_expand_gate_smallm<8> <<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
            else              k_hy4_v_expand_gate_smallm<12><<<grid, 128, 0, st>>>(u_h, w_uv.bf16, gate, attn_out, M, m0, nm, nh);
        }
    }
}

// =================================================================================================
// Full Gated Absorbed MLA Attention Forward
// =================================================================================================
inline void attn_forward(cublasHandle_t blas, const AttnWeights& w, AttnScratch& sc,
                         DsaLayerKv* dsa_kv, SwaLayerKv* swa_kv,
                         const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
                         const StepSeg* segs, int nsegs,
                         const int* d_slot, const int* d_pos, const int* d_page_table,
                         int M, int max_pps, int max_groups, int max_pos_in_step,
                         uint8_t* x8_scratch, cudaStream_t st,
                         const int* d_snap_dst = nullptr,
                         int* dfly_snap_grp = nullptr, int* dfly_snap_cnt = nullptr, int* dfly_snap_pos = nullptr,
                         __nv_bfloat16* idx_tk_save = nullptr,
                         int graph_idx_groups = 0, int graph_ns_override = 0) {
    ensure_mla_tc_smem();
    const int nh = w.nh_loc;
    const float theta = w.is_dsa ? geom::ROPE_THETA_DSA : geom::ROPE_THETA_SWA;

    // Pre-quantize x_norm ONCE when M >= 16 and MXFP8 is active (reused by q_a, kv_a, gate, idx_tk!)
    const uint8_t* xnorm_x8 = nullptr;
    if (M >= 16 && w.q_a_proj.is_mx8()) {
        mx8::bf16_to_mx8_a(x_norm, x8_scratch, M, geom::HID, st);
        xnorm_x8 = x8_scratch;
    }

    // 1. Q projection + Absorb W_uk_t + RoPE
    mx8::linear_w(blas, x_norm, w.q_a_proj, sc.q_a, M, geom::QA, geom::HID, x8_scratch, xnorm_x8, st);
    k_hy4_rmsnorm<<<M, 256, 0, st>>>(sc.q_a, w.q_a_ln, sc.q_resid, M, geom::QA, geom::RMS_EPS);

    const uint8_t* qresid_x8 = nullptr;
    if (M >= 16 && w.q_b_proj.is_mx8()) {
        // Use second region of x8_scratch after xnorm_x8? Simpler: run gate + kv_a + idx_tk BEFORE reusing x8_scratch!
    }
    // Run all x_norm projections while xnorm_x8 is valid!
    mx8::linear_w(blas, x_norm, w.kv_a_proj, sc.kv_raw, M, geom::KVA, geom::HID, x8_scratch, xnorm_x8, st);
    mx8::linear_w(blas, x_norm, w.gate_proj, sc.gate, M, nh * geom::V_HEAD, geom::HID, x8_scratch, xnorm_x8, st);
    if (w.is_dsa) {
        mx8::linear_w(blas, x_norm, w.idx_tk_proj, sc.idx_tk, M, geom::IDX_DIM, geom::HID, x8_scratch, xnorm_x8, st);
    }

    // Now x8_scratch is free for q_resid!
    if (M >= 16 && w.q_b_proj.is_mx8()) {
        mx8::bf16_to_mx8_a(sc.q_resid, x8_scratch, M, geom::QA, st);
        qresid_x8 = x8_scratch;
    }
    mx8::linear_w(blas, sc.q_resid, w.q_b_proj, sc.q_raw, M, nh * geom::QK_HEAD, geom::QA, x8_scratch, qresid_x8, st);

    if (M <= 12) {
        launch_q_absorb_rope_smallm(sc.q_raw, w.w_uk_t, d_pos, sc.q_full, M, nh, theta, st);
    } else {
        k_hy4_q_split_rope<<<M * nh, 256, 0, st>>>(sc.q_raw, d_pos, sc.q_nope_h, sc.q_full, M, nh, theta);
        const __nv_bfloat16* uk_bf16 = w.w_uk_t.bf16;
        if (w.w_uk_t.is_mx8()) {
            mx8::dequant_to_bf16(w.w_uk_t.mx8, sc.kv_b_tmp, st);
            uk_bf16 = sc.kv_b_tmp;
        }
        cublasSetStream(blas, st);
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            geom::KV_LORA, M, geom::QK_NOPE,
            &alpha,
            uk_bf16, CUDA_R_16BF, geom::QK_NOPE, static_cast<long long>(geom::KV_LORA) * geom::QK_NOPE,
            sc.q_nope_h, CUDA_R_16BF, geom::QK_NOPE, static_cast<long long>(M) * geom::QK_NOPE,
            &beta,
            sc.q_abs_h, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(M) * geom::KV_LORA,
            nh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        k_hy4_q_abs_scatter<<<M * nh, 256, 0, st>>>(sc.q_abs_h, sc.q_full, M, nh);
    }

    // 2. Write KV latent into paged DSA pool (BF16 / FP8 / INT4) or SWA ring (BF16)
    if (w.is_dsa) {
        if (dsa_kv->fmt == KvFmt::FP8) {
            k_hy4_kv_write<KvFmt::FP8, true><<<M, 256, 0, st>>>(
                sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, dsa_kv->kv_pool, M, max_pps, theta);
        } else if (dsa_kv->fmt == KvFmt::INT4) {
            k_hy4_kv_write<KvFmt::INT4, true><<<M, 256, 0, st>>>(
                sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, dsa_kv->kv_pool, M, max_pps, theta);
        } else {
            k_hy4_kv_write<KvFmt::BF16, true><<<M, 256, 0, st>>>(
                sc.kv_raw, w.kv_a_ln, d_slot, d_pos, d_page_table, dsa_kv->kv_pool, M, max_pps, theta);
        }
    } else {
        k_hy4_kv_write<KvFmt::BF16, false><<<M, 256, 0, st>>>(
            sc.kv_raw, w.kv_a_ln, d_slot, d_pos, nullptr, reinterpret_cast<uint8_t*>(swa_kv->ring), M, max_pps, theta);
    }

    // 3. Compressed Group Indexer (DSA layers only)
    bool use_sparse_sel = false;
    if (w.is_dsa) {
        const bool idx_fp8 = (dsa_kv->fmt != KvFmt::BF16);
        k_hy4_rmsnorm<<<M, 256, 0, st>>>(sc.idx_tk, w.idx_tk_norm, sc.idx_tk, M, geom::IDX_DIM, geom::RMS_EPS);

        if (idx_tk_save != nullptr && d_snap_dst != nullptr) {
            k_hy4_scatter_idx_tk<<<M, geom::IDX_DIM, 0, st>>>(sc.idx_tk, d_snap_dst, idx_tk_save, M);
        }

        if (M <= 12) {
            if (idx_fp8) {
                k_hy4_idx_k_update_smallm<true><<<1, 128, 0, st>>>(
                    sc.idx_tk, w.idx_kv_proj, w.idx_kg_proj, w.idx_ape, w.idx_k_norm,
                    d_slot, d_pos, d_page_table, dsa_kv->idx_raw, nullptr, dsa_kv->idx_q, dsa_kv->idx_sc, M, max_pps);
            } else {
                k_hy4_idx_k_update_smallm<false><<<1, 128, 0, st>>>(
                    sc.idx_tk, w.idx_kv_proj, w.idx_kg_proj, w.idx_ape, w.idx_k_norm,
                    d_slot, d_pos, d_page_table, dsa_kv->idx_raw, dsa_kv->idx_bf16, nullptr, nullptr, M, max_pps);
            }
        } else {
            for (int s = 0; s < nsegs; ++s) {
                if (segs[s].len <= 0) continue;
                const int g0 = segs[s].pos0 >> 2;
                const int g_comp_end = (segs[s].pos0 + segs[s].len) >> 2;
                if (g_comp_end > g0) {
                    if (idx_fp8) {
                        k_hy4_idx_k_update_seg_complete<true><<<g_comp_end - g0, 128, 0, st>>>(
                            sc.idx_tk, w.idx_kv_proj, w.idx_kg_proj, w.idx_ape, w.idx_k_norm,
                            d_page_table, dsa_kv->idx_raw, nullptr, dsa_kv->idx_q, dsa_kv->idx_sc,
                            max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                    } else {
                        k_hy4_idx_k_update_seg_complete<false><<<g_comp_end - g0, 128, 0, st>>>(
                            sc.idx_tk, w.idx_kv_proj, w.idx_kg_proj, w.idx_ape, w.idx_k_norm,
                            d_page_table, dsa_kv->idx_raw, dsa_kv->idx_bf16, nullptr, nullptr,
                            max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                    }
                }
                if (((segs[s].pos0 + segs[s].len) & 3) != 0) {
                    k_hy4_idx_k_update_seg_tail<<<1, 128, 0, st>>>(
                        sc.idx_tk, dsa_kv->idx_raw, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                }
            }
        }

        if (max_pos_in_step >= geom::IDX_TOPK) {
            use_sparse_sel = true;
            mx8::linear_w(blas, sc.q_resid, w.idx_q_proj, sc.idx_q, M, geom::IDX_HEADS * geom::IDX_DIM, geom::QA, x8_scratch, qresid_x8, st);
            k_hy4_idx_q_rope<<<M * geom::IDX_HEADS, 32, 0, st>>>(sc.idx_q, d_pos, M);
            linear_bf16_f32(blas, x_norm, w.idx_hw_proj, sc.idx_hw, M, geom::IDX_HEADS, geom::HID, st);

            const int max_ng = std::min(max_groups, (graph_idx_groups > 0) ? graph_idx_groups : (max_pos_in_step >> 2));
            for (int m0 = 0; m0 < M; m0 += IDX_SCORE_BATCH) {
                const int nm = std::min(IDX_SCORE_BATCH, M - m0);
                if (idx_fp8) {
                    k_hy4_idx_score_tc<true><<<dim3((max_ng + 31) / 32, nm), 128, 0, st>>>(
                        sc.idx_q, sc.idx_hw, nullptr, dsa_kv->idx_q, dsa_kv->idx_sc,
                        d_slot, d_pos, d_page_table, sc.idx_sc, m0, nm, max_pps, max_groups);
                } else {
                    k_hy4_idx_score_tc<false><<<dim3((max_ng + 31) / 32, nm), 128, 0, st>>>(
                        sc.idx_q, sc.idx_hw, dsa_kv->idx_bf16, nullptr, nullptr,
                        d_slot, d_pos, d_page_table, sc.idx_sc, m0, nm, max_pps, max_groups);
                }
                k_hy4_idx_topk511<<<nm, 256, 0, st>>>(sc.idx_sc, d_pos, sc.sel_grp, sc.sel_cnt, m0, nm, max_groups);
            }
        }

        // Optional per-slot isolated DFly2 index snapshot (100% graph-safe & multi-session race-free!)
        if (d_snap_dst && dfly_snap_grp && dfly_snap_cnt && dfly_snap_pos) {
            k_hy4_idx_snapshot_dst<<<M, 256, 0, st>>>(
                sc.sel_grp, sc.sel_cnt, d_pos, d_snap_dst,
                dfly_snap_grp, dfly_snap_cnt, dfly_snap_pos,
                M, use_sparse_sel);
        }
    }

    // 4. Tensor-Core Absorbed MLA Attention (Prefill TC for contiguous segments >= 16; Decode TC for small-M or sparse!)
    const bool can_prefill_tc = (!use_sparse_sel) && (M >= 16);
    if (can_prefill_tc) {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len <= 0) continue;
            const dim3 grid((segs[s].len + 15) / 16, nh);
            if (w.is_dsa) {
                if (dsa_kv->fmt == KvFmt::FP8) {
                    k_hy4_mla_prefill_tc<true, KvFmt::FP8><<<grid, 256, MLA_TC_SMEM, st>>>(
                        sc.q_full, w.sinks, dsa_kv->kv_pool, d_page_table, sc.u_h,
                        M, nh, max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                } else if (dsa_kv->fmt == KvFmt::INT4) {
                    k_hy4_mla_prefill_tc<true, KvFmt::INT4><<<grid, 256, MLA_TC_SMEM, st>>>(
                        sc.q_full, w.sinks, dsa_kv->kv_pool, d_page_table, sc.u_h,
                        M, nh, max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                } else {
                    k_hy4_mla_prefill_tc<true, KvFmt::BF16><<<grid, 256, MLA_TC_SMEM, st>>>(
                        sc.q_full, w.sinks, dsa_kv->kv_pool, d_page_table, sc.u_h,
                        M, nh, max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                }
            } else {
                k_hy4_mla_prefill_tc<false, KvFmt::BF16><<<grid, 256, MLA_TC_SMEM, st>>>(
                    sc.q_full, w.sinks, reinterpret_cast<const uint8_t*>(swa_kv->ring), nullptr, sc.u_h,
                    M, nh, max_pps, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
            }
        }
    } else {
        const int n_hg = nh >> 4;
        int NS = 1;
        if (graph_ns_override > 0) {
            NS = std::min(MLA_MAX_SPLITS, graph_ns_override);
        } else if (M <= 12) {
            if (max_pos_in_step + 1 >= 256) {
                const int want = (w.is_dsa && max_pos_in_step >= geom::IDX_TOPK) ? 16 : 8;
                NS = std::min(MLA_MAX_SPLITS, std::max(1, want / std::max(1, M)));
            }
        } else if (M <= 16) {
            const int eff_kv = w.is_dsa ? std::min(max_pos_in_step + 1, geom::IDX_TOPK)
                                        : std::min(max_pos_in_step + 1, geom::SWA_WINDOW);
            if (eff_kv >= 256) {
                const int want = (eff_kv >= 1024) ? 16 : 8;
                NS = std::min(MLA_MAX_SPLITS, std::max(1, want / std::max(1, M)));
            }
        }
        if (w.is_dsa) {
            if (dsa_kv->fmt == KvFmt::FP8) {
                k_hy4_mla_decode_tc<true, KvFmt::FP8><<<M * n_hg * NS, 256, MLA_TC_SMEM, st>>>(
                    sc.q_full, w.sinks, dsa_kv->kv_pool, d_slot, d_pos, d_page_table,
                    use_sparse_sel ? sc.sel_grp : nullptr, use_sparse_sel ? sc.sel_cnt : nullptr,
                    sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, nh, max_pps, NS);
            } else if (dsa_kv->fmt == KvFmt::INT4) {
                k_hy4_mla_decode_tc<true, KvFmt::INT4><<<M * n_hg * NS, 256, MLA_TC_SMEM, st>>>(
                    sc.q_full, w.sinks, dsa_kv->kv_pool, d_slot, d_pos, d_page_table,
                    use_sparse_sel ? sc.sel_grp : nullptr, use_sparse_sel ? sc.sel_cnt : nullptr,
                    sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, nh, max_pps, NS);
            } else {
                k_hy4_mla_decode_tc<true, KvFmt::BF16><<<M * n_hg * NS, 256, MLA_TC_SMEM, st>>>(
                    sc.q_full, w.sinks, dsa_kv->kv_pool, d_slot, d_pos, d_page_table,
                    use_sparse_sel ? sc.sel_grp : nullptr, use_sparse_sel ? sc.sel_cnt : nullptr,
                    sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, nh, max_pps, NS);
            }
        } else {
            k_hy4_mla_decode_tc<false, KvFmt::BF16><<<M * n_hg * NS, 256, MLA_TC_SMEM, st>>>(
                sc.q_full, w.sinks, reinterpret_cast<const uint8_t*>(swa_kv->ring), d_slot, d_pos, nullptr,
                nullptr, nullptr, sc.u_h, sc.split_m, sc.split_l, sc.split_o, M, nh, max_pps, NS);
        }
        if (NS > 1) {
            k_hy4_mla_split_reduce<<<M * nh, 256, 0, st>>>(sc.split_m, sc.split_l, sc.split_o, sc.u_h, M, nh, NS);
        }
    }

    // 5+6. Expand V + Sigmoid Gate + Transpose -> attn_out [M, nh, 256] -> o_proj
    if (M <= 12) {
        launch_v_expand_gate_smallm(sc.u_h, w.w_uv, sc.gate, sc.attn_out, M, nh, st);
    } else {
        const __nv_bfloat16* uv_bf16 = w.w_uv.bf16;
        if (w.w_uv.is_mx8()) {
            mx8::dequant_to_bf16(w.w_uv.mx8, sc.kv_b_tmp, st);
            uv_bf16 = sc.kv_b_tmp;
        }
        cublasSetStream(blas, st);
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            geom::V_HEAD, M, geom::KV_LORA,
            &alpha,
            uv_bf16, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(geom::V_HEAD) * geom::KV_LORA,
            sc.u_h, CUDA_R_16BF, geom::KV_LORA, static_cast<long long>(M) * geom::KV_LORA,
            &beta,
            sc.v_h, CUDA_R_16BF, geom::V_HEAD, static_cast<long long>(M) * geom::V_HEAD,
            nh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        k_hy4_attn_gate_transpose<<<M * nh, 256, 0, st>>>(sc.v_h, sc.gate, sc.attn_out, M, nh);
    }
    mx8::linear_w(blas, sc.attn_out, w.o_proj, out_part, M, geom::HID, nh * geom::V_HEAD, x8_scratch, nullptr, st);
}

} // namespace hy4
