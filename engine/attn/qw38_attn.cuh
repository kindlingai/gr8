// Qwen3.8-Flash-Next full attention (Qwen4ExpTextAttention) + QSA Block-4 Sparse Indexer (Qwen4ExpTextQSAIndexer).
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py:
//   - 12 trunk layers (l % 4 == 3) + 1 MTP layer (L048).
//   - 24 Q heads, 2 KV heads (GQA=12), HEAD_DIM=256, partial rotate_half mRoPE on first 64 dims (theta=1e7).
//   - q_proj [nh_q * 512, 2560]: per-head interleaved [q_h (256), gate_h (256)].
//   - q_norm [256], k_norm [256]: Qwen4ExpTextRMSNorm (1.0 + weight).
//   - QSA Indexer:
//       index_qk_proj [640, 2560] -> q_idx [4, 128] + k_raw [128].
//       q_idx = apply_rope(q_layernorm(q_idx), pos)   (1+w, partial rotate_half RoPE on first 64 dims).
//       Block b = 4 consecutive tokens [4b..4b+3]:
//         pooled = bf16(mean_{j=0..3}(k_raw[4b+j].float()))
//         block_k[b] = apply_rope(k_layernorm(pooled), pos=4b)  (stored directly in idx_pages!)
//       Score(b) = sum_{h=0..3} relu(dot(q_idx[h].float(), block_k[b].float())) / sqrt(128.0f).
//       Select top min(512, num_complete_blocks) blocks -> 2048 tokens + remaining 0..3 tail tokens.
//       When num_complete_blocks <= 512 (i.e. kv_len <= 2051), all causal tokens are visible (dense GQA!).
//   - Attention output: attn_out = attn_out * sigmoid(gate) -> o_proj [2560, nh_q * 256].
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include "../kernels/qw38_hc.cuh"
#include "../kernels/qw38_gdn.cuh"

namespace qw38 {

struct FullAttnWeights {
    const __nv_bfloat16* q_proj = nullptr;      // [nh_q * 512, 2560]
    const __nv_bfloat16* k_proj = nullptr;      // [nh_kv * 256, 2560]
    const __nv_bfloat16* v_proj = nullptr;      // [nh_kv * 256, 2560]
    const __nv_bfloat16* o_proj = nullptr;      // [2560, nh_q * 256]
    const __nv_bfloat16* q_norm = nullptr;      // [256] (1 + w)
    const __nv_bfloat16* k_norm = nullptr;      // [256] (1 + w)
    const __nv_bfloat16* idx_qk_proj = nullptr; // [640, 2560]
    const __nv_bfloat16* idx_q_norm = nullptr;  // [128] (1 + w)
    const __nv_bfloat16* idx_k_norm = nullptr;  // [128] (1 + w)
    // Optional on-load block-32 FP8 decode weights (GR8_QW38_DENSE_FP8=1)
    Fp8Mat fp8_q, fp8_k, fp8_v, fp8_idx, fp8_o;
};

// Paged KV + QSA compressed block-key cache matching qw38_kv_state(...) & qw38_ckpt_rewidth.h byte-for-byte:
//   kv_pages:  [slots][max_pages][2 (K,V)][64][nh_kv][256] BF16
//   idx_pages: [slots][max_pages][16][128] BF16 (pre-pooled, k_layernorm'd, RoPE'd at pos=4*b)
//   idx_raw:   [slots][4][128] BF16 (up to 3 pending raw k_raw vectors for the current incomplete block)
struct FullLayerKv {
    __nv_bfloat16* kv_pages = nullptr;
    __nv_bfloat16* idx_pages = nullptr;
    __nv_bfloat16* idx_raw = nullptr;
    __nv_bfloat16* idx_raw_snap = nullptr; // optional [slots, max_K, 4, 128] BF16 (MTP rollback)
    // Optional staging pages for StashPool partial-page checkpoints
    __nv_bfloat16* kv_staging = nullptr;
    __nv_bfloat16* idx_staging = nullptr;
};

struct FullAttnScratch {
    __nv_bfloat16* q_gate = nullptr;  // [max_M, nh_q * 512]
    __nv_bfloat16* q = nullptr;       // [max_M, nh_q * 256] (normed + RoPE'd)
    __nv_bfloat16* gate = nullptr;    // [max_M, nh_q * 256]
    __nv_bfloat16* k = nullptr;       // [max_M, nh_kv * 256]
    __nv_bfloat16* v = nullptr;       // [max_M, nh_kv * 256]
    __nv_bfloat16* idx_qk = nullptr;  // [max_M, 640]
    __nv_bfloat16* idx_q = nullptr;   // [max_M, 4 * 128] (normed + RoPE'd)
    __nv_bfloat16* attn_out = nullptr;// [max_M, nh_q * 256]
    float* idx_scores = nullptr;      // [max_M, max_blocks] FP32 (max_blocks = cap / 4)
    int* sel_blocks = nullptr;        // [max_M, 512] INT32 (selected block indices per query row)
    int* sel_counts = nullptr;        // [max_M] INT32 (number of selected blocks per query row; -1 = dense)
    // Split-KV scratch for GQA-12 FlashDecode (M <= 8, up to 24 splits)
    float* split_m = nullptr;         // [8 * 24 * 24] FP32
    float* split_l = nullptr;         // [8 * 24 * 24] FP32
    float* split_o = nullptr;         // [8 * 24 * 24 * 256] FP32
};

__device__ __forceinline__ void apply_partial_rope_pair(float& x0, float& x1, int pos, int pair_idx) {
    // inv_freq[i] = 1 / (1e7 ^ (2*i / 64))
    const float exp_val = static_cast<float>(2 * pair_idx) / static_cast<float>(geom::ROPE_DIM);
    const float inv_freq = 1.0f / powf(geom::ROPE_THETA, exp_val);
    const float angle = static_cast<float>(pos) * inv_freq;
    float s, c;
    sincosf(angle, &s, &c);
    // rotate_half on [x0 (first 32), x1 (second 32)]:
    //   out0 = x0 * c - x1 * s
    //   out1 = x1 * c + x0 * s
    const float r0 = x0 * c - x1 * s;
    const float r1 = x1 * c + x0 * s;
    x0 = r0;
    x1 = r1;
}

// Split q_gate [M, nh_q, 2, 256] -> q [M, nh_q, 256] (q_norm + partial RoPE) and gate [M, nh_q, 256].
// Grid: (M * nh_q), Block: 256 threads.
__global__ void k_qw38_q_norm_rope(const __nv_bfloat16* __restrict__ q_gate,
                                   const __nv_bfloat16* __restrict__ q_norm_w,
                                   const int* __restrict__ row_pos,
                                   __nv_bfloat16* __restrict__ q_out,
                                   __nv_bfloat16* __restrict__ gate_out,
                                   int M, int nh_q, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_q) return;
    const int m = idx / nh_q;
    const int d = threadIdx.x; // 0..255
    const __nv_bfloat16* src_q = q_gate + static_cast<size_t>(idx) * 512;
    const __nv_bfloat16* src_g = src_q + 256;
    __nv_bfloat16* dst_q = q_out + static_cast<size_t>(idx) * 256;
    __nv_bfloat16* dst_g = gate_out + static_cast<size_t>(idx) * 256;

    dst_g[d] = src_g[d];

    const float v = __bfloat162float(src_q[d]);
    float ss = warp_sum(v * v);
    __shared__ float sm[8];
    __shared__ float qf[256];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    if (d < 32) {
        float tot = (d < 8) ? sm[d] : 0.0f;
        tot = warp_sum(tot);
        if (d == 0) sm[0] = rsqrtf(tot / 256.0f + eps);
    }
    __syncthreads();
    const float normed = __bfloat162float(__float2bfloat16(v * sm[0] * (1.0f + __bfloat162float(q_norm_w[d]))));
    qf[d] = normed;
    __syncthreads();
    if (d < geom::ROPE_DIM / 2) {
        float x0 = qf[d], x1 = qf[d + geom::ROPE_DIM / 2];
        apply_partial_rope_pair(x0, x1, row_pos[m], d);
        dst_q[d] = __float2bfloat16(x0);
        dst_q[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
    } else if (d >= geom::ROPE_DIM) {
        dst_q[d] = __float2bfloat16(normed);
    }
}

// K norm + partial RoPE and scatter K & V into paged KV cache:
// kv_pages[slot][page][0=K, 1=V][tok_in_page][lkv][256] BF16.
// Grid: (M * nh_kv), Block: 256 threads.
__global__ void k_qw38_kv_write(const __nv_bfloat16* __restrict__ k_in,
                                const __nv_bfloat16* __restrict__ v_in,
                                const __nv_bfloat16* __restrict__ k_norm_w,
                                const int* __restrict__ row_slot,
                                const int* __restrict__ row_pos,
                                __nv_bfloat16* __restrict__ kv_pages,
                                int M, int nh_kv, int max_pages, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_kv) return;
    const int m = idx / nh_kv;
    const int lkv = idx % nh_kv;
    const int d = threadIdx.x; // 0..255
    const int slot = row_slot[m];
    const int pos = row_pos[m];
    const int page = pos / geom::KV_PAGE;
    const int off = pos % geom::KV_PAGE;

    const __nv_bfloat16* src_k = k_in + static_cast<size_t>(idx) * 256;
    const __nv_bfloat16* src_v = v_in + static_cast<size_t>(idx) * 256;

    const float v = __bfloat162float(src_k[d]);
    float ss = warp_sum(v * v);
    __shared__ float sm[8];
    __shared__ float kf[256];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    if (d < 32) {
        float tot = (d < 8) ? sm[d] : 0.0f;
        tot = warp_sum(tot);
        if (d == 0) sm[0] = rsqrtf(tot / 256.0f + eps);
    }
    __syncthreads();
    const float normed = __bfloat162float(__float2bfloat16(v * sm[0] * (1.0f + __bfloat162float(k_norm_w[d]))));
    kf[d] = normed;
    __syncthreads();

    const size_t page_stride = static_cast<size_t>(2) * geom::KV_PAGE * nh_kv * 256;
    const size_t plane_stride = static_cast<size_t>(geom::KV_PAGE) * nh_kv * 256;
    __nv_bfloat16* page_base = kv_pages + (static_cast<size_t>(slot) * max_pages + page) * page_stride;
    __nv_bfloat16* dst_k = page_base + (static_cast<size_t>(off) * nh_kv + lkv) * 256;
    __nv_bfloat16* dst_v = page_base + plane_stride + (static_cast<size_t>(off) * nh_kv + lkv) * 256;

    dst_v[d] = src_v[d];
    if (d < geom::ROPE_DIM / 2) {
        float x0 = kf[d], x1 = kf[d + geom::ROPE_DIM / 2];
        apply_partial_rope_pair(x0, x1, pos, d);
        dst_k[d] = __float2bfloat16(x0);
        dst_k[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
    } else if (d >= geom::ROPE_DIM) {
        dst_k[d] = __float2bfloat16(normed);
    }
}

// QSA Indexer Q norm + partial RoPE: idx_qk [M, 640] -> idx_q [M, 4, 128] BF16.
// Grid: (M * 4), Block: 128 threads.
__global__ void k_qsa_q_norm_rope(const __nv_bfloat16* __restrict__ idx_qk,
                                  const __nv_bfloat16* __restrict__ q_norm_w,
                                  const int* __restrict__ row_pos,
                                  __nv_bfloat16* __restrict__ idx_q,
                                  int M, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::IDX_HEADS) return;
    const int m = idx / geom::IDX_HEADS;
    const int h = idx % geom::IDX_HEADS;
    const int d = threadIdx.x; // 0..127
    const __nv_bfloat16* src = idx_qk + static_cast<size_t>(m) * geom::IDX_PROJ_OUT + h * 128;
    __nv_bfloat16* dst = idx_q + static_cast<size_t>(idx) * 128;

    const float v = __bfloat162float(src[d]);
    float ss = warp_sum(v * v);
    __shared__ float sm[4];
    __shared__ float qf[128];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / 128.0f + eps);
    const float normed = __bfloat162float(__float2bfloat16(v * inv * (1.0f + __bfloat162float(q_norm_w[d]))));
    qf[d] = normed;
    __syncthreads();
    if (d < geom::ROPE_DIM / 2) {
        float x0 = qf[d], x1 = qf[d + geom::ROPE_DIM / 2];
        apply_partial_rope_pair(x0, x1, row_pos[m], d);
        dst[d] = __float2bfloat16(x0);
        dst[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
    } else if (d >= geom::ROPE_DIM) {
        dst[d] = __float2bfloat16(normed);
    }
}

// QSA Indexer K accumulation & block-4 pooling for one segment (slot, row0, pos0, len).
// Runs sequentially over t=0..len-1 in one 128-thread CTA per segment (tiny: 128 dims!).
// Writes raw k to idx_raw[slot][pos % 4], and whenever (pos % 4 == 3) averages the 4 raw keys,
// applies k_layernorm (1+w), applies partial RoPE at pos = block * 4, and writes idx_pages[slot][page][blk%16].
__global__ void k_qsa_k_update_seg(const __nv_bfloat16* __restrict__ idx_qk,
                                   const __nv_bfloat16* __restrict__ k_norm_w,
                                   __nv_bfloat16* __restrict__ idx_raw,
                                   __nv_bfloat16* __restrict__ idx_raw_snap,
                                   __nv_bfloat16* __restrict__ idx_pages,
                                   int max_pages, int slot, int row0, int pos0, int len, int snap_k, float eps) {
    const int d = threadIdx.x; // 0..127
    __nv_bfloat16* raw_slot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    __shared__ float sm[4];
    __shared__ float kf[128];

    for (int t = 0; t < len; ++t) {
        const int pos = pos0 + t;
        const int sub = pos & 3;
        const __nv_bfloat16 kv = idx_qk[static_cast<size_t>(row0 + t) * geom::IDX_PROJ_OUT + geom::IDX_HEADS * 128 + d];
        raw_slot[sub * 128 + d] = kv;
        __syncthreads();
        if (sub == 3) {
            const float avg = 0.25f * (
                __bfloat162float(raw_slot[0 * 128 + d]) +
                __bfloat162float(raw_slot[1 * 128 + d]) +
                __bfloat162float(raw_slot[2 * 128 + d]) +
                __bfloat162float(raw_slot[3 * 128 + d]));
            const float avg_bf = __bfloat162float(__float2bfloat16(avg));
            float ss = warp_sum(avg_bf * avg_bf);
            if ((d & 31) == 0) sm[d >> 5] = ss;
            __syncthreads();
            const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / 128.0f + eps);
            const float normed = __bfloat162float(__float2bfloat16(avg_bf * inv * (1.0f + __bfloat162float(k_norm_w[d]))));
            kf[d] = normed;
            __syncthreads();

            const int blk = pos >> 2;
            const int page = blk >> 4;   // 16 blocks per 64-token page
            const int boff = blk & 15;
            __nv_bfloat16* dst = idx_pages + ((static_cast<size_t>(slot) * max_pages + page) * 16 + boff) * 128;
            if (d < geom::ROPE_DIM / 2) {
                float x0 = kf[d], x1 = kf[d + geom::ROPE_DIM / 2];
                apply_partial_rope_pair(x0, x1, blk * 4, d);
                dst[d] = __float2bfloat16(x0);
                dst[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
            } else if (d >= geom::ROPE_DIM) {
                dst[d] = __float2bfloat16(normed);
            }
            __syncthreads();
        }
        if (idx_raw_snap && t < len - 1 && t < snap_k) {
            __nv_bfloat16* sb = idx_raw_snap + (static_cast<size_t>(slot) * snap_k + t) * 4 * 128;
            #pragma unroll
            for (int k = 0; k < 4; ++k) sb[k * 128 + d] = raw_slot[k * 128 + d];
        }
    }
}

// Batched single-token QSA K update across all M decode rows in ONE launch!
// Grid: (M), Block: 128 threads.
__global__ void k_qsa_k_update_dec_batch(const __nv_bfloat16* __restrict__ idx_qk,
                                         const __nv_bfloat16* __restrict__ k_norm_w,
                                         __nv_bfloat16* __restrict__ idx_raw,
                                         __nv_bfloat16* __restrict__ idx_pages,
                                         const int* __restrict__ row_slot,
                                         const int* __restrict__ row_pos,
                                         int max_pages, int M, float eps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int d = threadIdx.x; // 0..127
    const int slot = row_slot[m];
    const int pos = row_pos[m];
    const int sub = pos & 3;
    __nv_bfloat16* raw_slot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    const __nv_bfloat16 kv = idx_qk[static_cast<size_t>(m) * geom::IDX_PROJ_OUT + geom::IDX_HEADS * 128 + d];
    raw_slot[sub * 128 + d] = kv;
    __syncthreads();
    if (sub == 3) {
        __shared__ float sm[4];
        __shared__ float kf[128];
        const float avg = 0.25f * (
            __bfloat162float(raw_slot[0 * 128 + d]) +
            __bfloat162float(raw_slot[1 * 128 + d]) +
            __bfloat162float(raw_slot[2 * 128 + d]) +
            __bfloat162float(raw_slot[3 * 128 + d]));
        const float avg_bf = __bfloat162float(__float2bfloat16(avg));
        const float ss = warp_sum(avg_bf * avg_bf);
        if ((d & 31) == 0) sm[d >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / 128.0f + eps);
        const float normed = __bfloat162float(__float2bfloat16(avg_bf * inv * (1.0f + __bfloat162float(k_norm_w[d]))));
        kf[d] = normed;
        __syncthreads();

        const int blk = pos >> 2;
        const int page = blk >> 4;
        const int boff = blk & 15;
        __nv_bfloat16* dst = idx_pages + ((static_cast<size_t>(slot) * max_pages + page) * 16 + boff) * 128;
        if (d < geom::ROPE_DIM / 2) {
            float x0 = kf[d], x1 = kf[d + geom::ROPE_DIM / 2];
            apply_partial_rope_pair(x0, x1, blk * 4, d);
            dst[d] = __float2bfloat16(x0);
            dst[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
        } else if (d >= geom::ROPE_DIM) {
            dst[d] = __float2bfloat16(normed);
        }
    }
}

// QSA Indexer scoring + top-512 block selection per query row m.
// Only runs work when num_complete_blocks = (pos + 1) / 4 > 512; otherwise sets sel_counts[m] = -1 (dense).
// Grid: (M), Block: 256 threads.
__global__ void k_qsa_score_blocks(const __nv_bfloat16* __restrict__ idx_q,
                                   const __nv_bfloat16* __restrict__ idx_pages,
                                   const int* __restrict__ row_slot,
                                   const int* __restrict__ row_pos,
                                   float* __restrict__ idx_scores,
                                   int max_pages, int max_blocks, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int pos = row_pos[m];
    const int nblk = (pos + 1) / geom::IDX_COMPRESS;
    if (nblk <= geom::IDX_TOP_BLOCKS) return;

    const int slot = row_slot[m];
    __shared__ float sq[geom::IDX_HEADS][128];
    for (int i = threadIdx.x; i < geom::IDX_HEADS * 128; i += blockDim.x) {
        sq[i >> 7][i & 127] = __bfloat162float(idx_q[static_cast<size_t>(m) * geom::IDX_HEADS * 128 + i]);
    }
    __syncthreads();

    constexpr float inv_sqrt_d = 0.08838834764831845f; // 1 / sqrt(128)
    float* out = idx_scores + static_cast<size_t>(m) * max_blocks;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;

    for (int b = warp; b < nblk; b += nwarps) {
        const int page = b >> 4;
        const int boff = b & 15;
        const __nv_bfloat16* kb = idx_pages + ((static_cast<size_t>(slot) * max_pages + page) * 16 + boff) * 128;
        float k0 = __bfloat162float(kb[lane]);
        float k1 = __bfloat162float(kb[lane + 32]);
        float k2 = __bfloat162float(kb[lane + 64]);
        float k3 = __bfloat162float(kb[lane + 96]);
        float score = 0.0f;
        #pragma unroll
        for (int h = 0; h < geom::IDX_HEADS; ++h) {
            float dot = sq[h][lane] * k0 + sq[h][lane + 32] * k1 + sq[h][lane + 64] * k2 + sq[h][lane + 96] * k3;
            dot = warp_sum(dot);
            if (dot > 0.0f) score += dot;
        }
        if (lane == 0) out[b] = score * inv_sqrt_d;
    }
}

// Select top-512 blocks from idx_scores[m, 0..nblk) using binary search on threshold + compaction.
// Exact tie-breaking matches top-512 selection. Grid: (M), Block: 256 threads.
__global__ void k_qsa_select_top512(const float* __restrict__ idx_scores,
                                    const int* __restrict__ row_pos,
                                    int* __restrict__ sel_blocks,
                                    int* __restrict__ sel_counts,
                                    int max_blocks, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int pos = row_pos[m];
    const int nblk = (pos + 1) / geom::IDX_COMPRESS;
    if (nblk <= geom::IDX_TOP_BLOCKS) {
        if (threadIdx.x == 0) sel_counts[m] = -1; // dense causal
        return;
    }
    const float* sc = idx_scores + static_cast<size_t>(m) * max_blocks;
    int* dst = sel_blocks + static_cast<size_t>(m) * geom::IDX_TOP_BLOCKS;

    // 1. Find min/max score
    __shared__ float s_lo, s_hi;
    __shared__ int s_cnt;
    float lo = 1e30f, hi = -1e30f;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        const float v = sc[b];
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, off));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, off));
    }
    __shared__ float rlo[8], rhi[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { rlo[warp] = lo; rhi[warp] = hi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ml = rlo[0], mh = rhi[0];
        for (int w = 1; w < 8; ++w) { ml = fminf(ml, rlo[w]); mh = fmaxf(mh, rhi[w]); }
        s_lo = ml;
        s_hi = mh;
    }
    __syncthreads();

    // 2. Binary search for threshold where count(score > mid) <= 512
    for (int it = 0; it < 28; ++it) {
        const float mid = 0.5f * (s_lo + s_hi);
        int c = 0;
        for (int b = threadIdx.x; b < nblk; b += blockDim.x) c += (sc[b] > mid) ? 1 : 0;
        c = static_cast<int>(warp_sum(static_cast<float>(c)));
        if (threadIdx.x == 0) s_cnt = 0;
        __syncthreads();
        if (lane == 0) atomicAdd(&s_cnt, c);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (s_cnt > geom::IDX_TOP_BLOCKS) s_lo = mid;
            else s_hi = mid;
        }
        __syncthreads();
    }
    const float thr = s_hi;

    // 3. Emit all strictly > thr, then fill ties == thr up to 512
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        if (sc[b] > thr) {
            const int p = atomicAdd(&s_cnt, 1);
            if (p < geom::IDX_TOP_BLOCKS) dst[p] = b;
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        if (sc[b] <= thr && sc[b] >= s_lo) {
            const int p = atomicAdd(&s_cnt, 1);
            if (p < geom::IDX_TOP_BLOCKS) dst[p] = b;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) sel_counts[m] = min(s_cnt, geom::IDX_TOP_BLOCKS);
}

// GQA-12 Tiled FlashAttention + Split-KV (works for ALL TP=1..4, both dense & QSA sparse!).
// 1 CTA = (m, lkv, split) with 384 threads (12 warps = up to 12 Q heads sharing 1 KV head!).
// Loads 16 KV tokens per tile into 16 KB smem with uint4 coalesced loads -> 12x less KV bandwidth & 1 barrier/16 tokens!
// When NS == 1: writes final gated BF16 output directly to `out`.
// When NS > 1:  writes partial (m_max, l_sum, acc[256]) to split_{m,l,o} for k_qw38_gqa_split_reduce.
__global__ void __launch_bounds__(384) k_qw38_gqa_flash_tiled(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ kv_pages,
    const int* __restrict__ row_slot,
    const int* __restrict__ row_pos,
    const int* __restrict__ sel_blocks,
    const int* __restrict__ sel_counts,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ split_m,
    float* __restrict__ split_l,
    float* __restrict__ split_o,
    int M, int nh_q, int nh_kv, int q0_global, int kv0_global, int max_pages, int NS) {
    const int idx = blockIdx.x;
    const int sp = idx % NS;
    const int rem = idx / NS;
    const int lkv = rem % nh_kv;
    const int m = rem / nh_kv;
    if (m >= M) return;

    const int tid = threadIdx.x;   // 0..383
    const int warp = tid >> 5;     // 0..11
    const int lane = tid & 31;     // 0..31 -> dims [lane*8 .. lane*8+7]

    const int lq = (kv0_global + lkv) * geom::GQA + warp - q0_global;
    const bool active_q = (lq >= 0 && lq < nh_q);

    const int slot = row_slot[m];
    const int pos = row_pos[m];
    const int nsel = sel_counts ? sel_counts[m] : -1;

    const int n_blk_tok = (nsel >= 0) ? (nsel * geom::IDX_COMPRESS) : 0;
    const int tail_start = ((pos + 1) / geom::IDX_COMPRESS) * geom::IDX_COMPRESS;
    const int total_tok = (nsel < 0) ? (pos + 1) : (n_blk_tok + (pos + 1 - tail_start));
    const int* sb = (nsel >= 0) ? (sel_blocks + static_cast<size_t>(m) * geom::IDX_TOP_BLOCKS) : nullptr;

    const int per = (total_tok + NS - 1) / NS;
    const int k_start = sp * per;
    const int k_end = min(total_tok, k_start + per);

    float qr[8];
    if (active_q) {
        const size_t qh = static_cast<size_t>(m) * nh_q + lq;
        const uint4 qv = reinterpret_cast<const uint4*>(q + qh * 256)[lane];
        const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(qp[i]);
            qr[2 * i] = f.x;
            qr[2 * i + 1] = f.y;
        }
    }

    const size_t page_stride = static_cast<size_t>(2) * geom::KV_PAGE * nh_kv * 256;
    const size_t plane_stride = static_cast<size_t>(geom::KV_PAGE) * nh_kv * 256;
    const __nv_bfloat16* slot_base = kv_pages + static_cast<size_t>(slot) * max_pages * page_stride;

    __shared__ uint4 sK[16][32]; // 8 KB
    __shared__ uint4 sV[16][32]; // 8 KB
    constexpr float scale = 0.0625f; // 1 / sqrt(256)

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int t0 = k_start; t0 < k_end; t0 += 16) {
        const int n_tok = min(16, k_end - t0);
        const int n_u4 = n_tok * 32;
        for (int i = tid; i < n_u4; i += 384) {
            const int j = i >> 5;
            const int l = i & 31;
            const int k = t0 + j;
            int kv_pos;
            if (nsel < 0) kv_pos = k;
            else if (k < n_blk_tok) kv_pos = sb[k >> 2] * geom::IDX_COMPRESS + (k & 3);
            else kv_pos = tail_start + (k - n_blk_tok);

            const int page = kv_pos / geom::KV_PAGE;
            const int off = kv_pos % geom::KV_PAGE;
            const __nv_bfloat16* pbase = slot_base + static_cast<size_t>(page) * page_stride;
            const size_t h_off = (static_cast<size_t>(off) * nh_kv + lkv) * 256;
            sK[j][l] = reinterpret_cast<const uint4*>(pbase + h_off)[l];
            sV[j][l] = reinterpret_cast<const uint4*>(pbase + plane_stride + h_off)[l];
        }
        __syncthreads();

        if (active_q) {
            for (int j = 0; j < n_tok; ++j) {
                const uint4 kv = sK[j][lane];
                const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kv);
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 fk = __bfloat1622float2(kp[i]);
                    dot = fmaf(qr[2 * i], fk.x, fmaf(qr[2 * i + 1], fk.y, dot));
                }
                const float score = warp_sum(dot) * scale;
                const float new_max = fmaxf(m_max, score);
                const float alpha = __expf(m_max - new_max);
                const float p = __expf(score - new_max);

                const uint4 vv = sV[j][lane];
                const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vv);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 fv = __bfloat1622float2(vp[i]);
                    acc[2 * i] = fmaf(p, fv.x, acc[2 * i] * alpha);
                    acc[2 * i + 1] = fmaf(p, fv.y, acc[2 * i + 1] * alpha);
                }
                l_sum = l_sum * alpha + p;
                m_max = new_max;
            }
        }
        __syncthreads();
    }

    if (!active_q) return;
    const size_t qh = static_cast<size_t>(m) * nh_q + lq;
    if (NS == 1) {
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        const uint4 gv = reinterpret_cast<const uint4*>(gate + qh * 256)[lane];
        const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
        uint4 ov;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 fg = __bfloat1622float2(gp[i]);
            const float o0 = __bfloat162float(__float2bfloat16(acc[2 * i] * inv_l));
            const float o1 = __bfloat162float(__float2bfloat16(acc[2 * i + 1] * inv_l));
            op[i] = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
        }
        reinterpret_cast<uint4*>(out + qh * 256)[lane] = ov;
    } else {
        const size_t sidx = qh * NS + sp;
        if (lane == 0) {
            split_m[sidx] = m_max;
            split_l[sidx] = l_sum;
        }
        float* od = split_o + sidx * 256 + lane * 8;
        *reinterpret_cast<float4*>(od) = make_float4(acc[0], acc[1], acc[2], acc[3]);
        *reinterpret_cast<float4*>(od + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
}

// Merge NS Split-KV partials via exact log-sum-exp + apply sigmoid output gate -> attn_out.
// Grid: (M * nh_q), Block: 32 threads (1 warp = 32 lanes x 8 dims = 256 dims).
__global__ void k_qw38_gqa_split_reduce(const float* __restrict__ split_m,
                                        const float* __restrict__ split_l,
                                        const float* __restrict__ split_o,
                                        const __nv_bfloat16* __restrict__ gate,
                                        __nv_bfloat16* __restrict__ out,
                                        int M, int nh_q, int NS) {
    const int qh = blockIdx.x;
    if (qh >= M * nh_q) return;
    const int lane = threadIdx.x;

    const float* sm = split_m + static_cast<size_t>(qh) * NS;
    const float* sl = split_l + static_cast<size_t>(qh) * NS;

    float g_max = -1e30f;
    for (int sp = 0; sp < NS; ++sp) {
        if (sl[sp] > 0.0f) g_max = fmaxf(g_max, sm[sp]);
    }

    float l_tot = 0.0f;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int sp = 0; sp < NS; ++sp) {
        const float ls = sl[sp];
        if (ls <= 0.0f) continue;
        const float w = __expf(sm[sp] - g_max);
        l_tot += ls * w;
        const float* so = split_o + (static_cast<size_t>(qh) * NS + sp) * 256 + lane * 8;
        const float4 v0 = *reinterpret_cast<const float4*>(so);
        const float4 v1 = *reinterpret_cast<const float4*>(so + 4);
        acc[0] = fmaf(w, v0.x, acc[0]); acc[1] = fmaf(w, v0.y, acc[1]);
        acc[2] = fmaf(w, v0.z, acc[2]); acc[3] = fmaf(w, v0.w, acc[3]);
        acc[4] = fmaf(w, v1.x, acc[4]); acc[5] = fmaf(w, v1.y, acc[5]);
        acc[6] = fmaf(w, v1.z, acc[6]); acc[7] = fmaf(w, v1.w, acc[7]);
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    const uint4 gv = reinterpret_cast<const uint4*>(gate + static_cast<size_t>(qh) * 256)[lane];
    const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
    uint4 ov;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fg = __bfloat1622float2(gp[i]);
        const float o0 = __bfloat162float(__float2bfloat16(acc[2 * i] * inv_l));
        const float o1 = __bfloat162float(__float2bfloat16(acc[2 * i + 1] * inv_l));
        op[i] = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
    }
    reinterpret_cast<uint4*>(out + static_cast<size_t>(qh) * 256)[lane] = ov;
}

inline int gqa_split_count(int M, int nh_kv, int max_pos, bool has_split) {
    if (M <= 8 && max_pos >= 128 && has_split) {
        int ns = (max_pos >= 512) ? (24 / std::min(M * nh_kv, 6)) : 8;
        if (ns > 24) ns = 24;
        if (ns < 2) ns = 2;
        return ns;
    }
    return 1;
}

// Graph-safe batched MTP verify QSA K update across S sessions x V tokens/session!
// Grid: (S), Block: 128 threads.
__global__ void k_qsa_k_update_verify_batch(const __nv_bfloat16* __restrict__ idx_qk,
                                            const __nv_bfloat16* __restrict__ k_norm_w,
                                            __nv_bfloat16* __restrict__ idx_raw,
                                            __nv_bfloat16* __restrict__ idx_raw_snap,
                                            __nv_bfloat16* __restrict__ idx_pages,
                                            const int* __restrict__ row_slot,
                                            const int* __restrict__ row_pos,
                                            int max_pages, int S, int V, int snap_k, float eps) {
    const int s = blockIdx.x;
    if (s >= S) return;
    const int d = threadIdx.x; // 0..127
    const int row0 = s * V;
    const int slot = row_slot[row0];
    const int pos0 = row_pos[row0];
    __nv_bfloat16* raw_slot = idx_raw + static_cast<size_t>(slot) * 4 * 128;
    __shared__ float sm[4];
    __shared__ float kf[128];

    for (int t = 0; t < V; ++t) {
        const int pos = pos0 + t;
        const int sub = pos & 3;
        const __nv_bfloat16 kv = idx_qk[static_cast<size_t>(row0 + t) * geom::IDX_PROJ_OUT + geom::IDX_HEADS * 128 + d];
        raw_slot[sub * 128 + d] = kv;
        __syncthreads();
        if (sub == 3) {
            const float avg = 0.25f * (
                __bfloat162float(raw_slot[0 * 128 + d]) +
                __bfloat162float(raw_slot[1 * 128 + d]) +
                __bfloat162float(raw_slot[2 * 128 + d]) +
                __bfloat162float(raw_slot[3 * 128 + d]));
            const float avg_bf = __bfloat162float(__float2bfloat16(avg));
            const float ss = warp_sum(avg_bf * avg_bf);
            if ((d & 31) == 0) sm[d >> 5] = ss;
            __syncthreads();
            const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / 128.0f + eps);
            const float normed = __bfloat162float(__float2bfloat16(avg_bf * inv * (1.0f + __bfloat162float(k_norm_w[d]))));
            kf[d] = normed;
            __syncthreads();

            const int blk = pos >> 2;
            const int page = blk >> 4;
            const int boff = blk & 15;
            __nv_bfloat16* dst = idx_pages + ((static_cast<size_t>(slot) * max_pages + page) * 16 + boff) * 128;
            if (d < geom::ROPE_DIM / 2) {
                float x0 = kf[d], x1 = kf[d + geom::ROPE_DIM / 2];
                apply_partial_rope_pair(x0, x1, blk * 4, d);
                dst[d] = __float2bfloat16(x0);
                dst[d + geom::ROPE_DIM / 2] = __float2bfloat16(x1);
            } else if (d >= geom::ROPE_DIM) {
                dst[d] = __float2bfloat16(normed);
            }
            __syncthreads();
        }
        if (idx_raw_snap && t < V - 1 && t < snap_k) {
            __nv_bfloat16* sb = idx_raw_snap + (static_cast<size_t>(slot) * snap_k + t) * 4 * 128;
            #pragma unroll
            for (int k = 0; k < 4; ++k) sb[k * 128 + d] = raw_slot[k * 128 + d];
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Tensor-Core QSA Block Scorer + Tensor-Core GQA-12 Prefill FlashAttention (sm_80+)
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void attn_mma_bf16_m16n8k16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void attn_ldsm4(uint32_t (&a)[4], const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
#endif
}

__device__ __forceinline__ void attn_ldsm2(uint32_t (&b)[2], const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(s));
#endif
}

__device__ __forceinline__ void attn_ldsm2_trans(uint32_t (&b)[2], const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(s));
#endif
}

// Per-segment Tensor-Core Tiled QSA Block Scorer (BM=64 queries x BN=64 compressed blocks).
// Reads idx_pages[seg_slot] once per 64 queries (64x less HBM traffic) and computes 4-head ReLU-sum via BF16 MMA.
// Grid: dim3((seg_max_nblk + 63) / 64, (seg_len + 63) / 64), Block: 256 (8 warps = 4x2 tiles).
__global__ void __launch_bounds__(256) k_qsa_score_blocks_tc_seg(
    const __nv_bfloat16* __restrict__ idx_q,
    const __nv_bfloat16* __restrict__ idx_pages,
    float* __restrict__ idx_scores,
    int max_pages, int max_blocks,
    int seg_row0, int seg_len, int seg_slot, int seg_pos0) {
    const int b0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    if (m0 >= seg_len) return;

    const int n_m = min(64, seg_len - m0);
    const int tile_max_nblk = (seg_pos0 + m0 + n_m) / geom::IDX_COMPRESS;
    if (tile_max_nblk <= geom::IDX_TOP_BLOCKS || b0 >= tile_max_nblk) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int wm = (warp >> 1) * 16; // 0, 16, 32, 48
    const int wn = (warp & 1) * 32;  // 0, 32

    __shared__ __nv_bfloat16 sK[64][128];
    __shared__ __nv_bfloat16 sQ[64][128];

    for (int idx = tid; idx < 64 * 16; idx += 256) {
        const int lb = idx >> 4, d8 = (idx & 15) << 3;
        const int b = b0 + lb;
        uint4 kv = make_uint4(0, 0, 0, 0);
        if (b < tile_max_nblk) {
            const int page = b >> 4, boff = b & 15;
            const __nv_bfloat16* kb = idx_pages + ((static_cast<size_t>(seg_slot) * max_pages + page) * 16 + boff) * 128;
            kv = *reinterpret_cast<const uint4*>(kb + d8);
        }
        *reinterpret_cast<uint4*>(sK[lb] + d8) = kv;
    }

    float tot[4][4] = {};

    for (int h = 0; h < geom::IDX_HEADS; ++h) {
        for (int idx = tid; idx < 64 * 16; idx += 256) {
            const int lm = idx >> 4, d8 = (idx & 15) << 3;
            const int lm_g = m0 + lm;
            uint4 qv = make_uint4(0, 0, 0, 0);
            if (lm_g < seg_len) {
                qv = *reinterpret_cast<const uint4*>(
                    idx_q + (static_cast<size_t>(seg_row0 + lm_g) * geom::IDX_HEADS + h) * 128 + d8);
            }
            *reinterpret_cast<uint4*>(sQ[lm] + d8) = qv;
        }
        __syncthreads();

        float acc[4][4] = {};
        #pragma unroll
        for (int k0 = 0; k0 < 128; k0 += 16) {
            uint32_t qa[4];
            qa[0] = *reinterpret_cast<const uint32_t*>(sQ[wm + gid] + k0 + 2 * tid4);
            qa[1] = *reinterpret_cast<const uint32_t*>(sQ[wm + gid + 8] + k0 + 2 * tid4);
            qa[2] = *reinterpret_cast<const uint32_t*>(sQ[wm + gid] + k0 + 2 * tid4 + 8);
            qa[3] = *reinterpret_cast<const uint32_t*>(sQ[wm + gid + 8] + k0 + 2 * tid4 + 8);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int lb = wn + j * 8 + gid;
                uint32_t kb[2];
                kb[0] = *reinterpret_cast<const uint32_t*>(sK[lb] + k0 + 2 * tid4);
                kb[1] = *reinterpret_cast<const uint32_t*>(sK[lb] + k0 + 2 * tid4 + 8);
                attn_mma_bf16_m16n8k16(acc[j], qa, kb);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            tot[j][0] += fmaxf(acc[j][0], 0.0f);
            tot[j][1] += fmaxf(acc[j][1], 0.0f);
            tot[j][2] += fmaxf(acc[j][2], 0.0f);
            tot[j][3] += fmaxf(acc[j][3], 0.0f);
        }
        __syncthreads();
    }

    constexpr float inv_sqrt_d = 0.08838834764831845f;
    const int la = m0 + wm + gid;
    const int lb = la + 8;
    const int nblk_a = (la < seg_len) ? ((seg_pos0 + la + 1) / geom::IDX_COMPRESS) : 0;
    const int nblk_b = (lb < seg_len) ? ((seg_pos0 + lb + 1) / geom::IDX_COMPRESS) : 0;
    const bool act_a = (nblk_a > geom::IDX_TOP_BLOCKS);
    const bool act_b = (nblk_b > geom::IDX_TOP_BLOCKS);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = b0 + wn + j * 8 + 2 * tid4;
        if (act_a) {
            const size_t row_off = static_cast<size_t>(seg_row0 + la) * max_blocks;
            if (c0 < nblk_a)     idx_scores[row_off + c0]     = tot[j][0] * inv_sqrt_d;
            if (c0 + 1 < nblk_a) idx_scores[row_off + c0 + 1] = tot[j][1] * inv_sqrt_d;
        }
        if (act_b) {
            const size_t row_off = static_cast<size_t>(seg_row0 + lb) * max_blocks;
            if (c0 < nblk_b)     idx_scores[row_off + c0]     = tot[j][2] * inv_sqrt_d;
            if (c0 + 1 < nblk_b) idx_scores[row_off + c0 + 1] = tot[j][3] * inv_sqrt_d;
        }
    }
}

// Zero-bank-conflict Tensor-Core GQA-12 Head-MMA FlashAttention (works for BOTH Dense & QSA Sparse, TP=1..4).
// 1 CTA = (m, lkv), Grid: (nh_kv * M), Block: 256 (8 warps). Uses ldmatrix.x4 / ldmatrix.x2 / ldmatrix.x2.trans.
__global__ void __launch_bounds__(256, 2) k_qw38_gqa_tc_head12(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ kv_pages,
    const int* __restrict__ row_slot,
    const int* __restrict__ row_pos,
    const int* __restrict__ sel_blocks,
    const int* __restrict__ sel_counts,
    __nv_bfloat16* __restrict__ out,
    int M, int nh_q, int nh_kv, int q0_global, int kv0_global, int max_pages) {
    const int idx = blockIdx.x;
    const int lkv = idx / M;
    const int m = idx % M;
    if (lkv >= nh_kv) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int gq_base = (kv0_global + lkv) * geom::GQA - q0_global;

    const int slot = row_slot[m];
    const int pos = row_pos[m];
    const int nsel = sel_counts ? sel_counts[m] : -1;
    const int n_blk_tok = (nsel >= 0) ? (nsel * geom::IDX_COMPRESS) : 0;
    const int tail_start = ((pos + 1) / geom::IDX_COMPRESS) * geom::IDX_COMPRESS;
    const int total_tok = (nsel < 0) ? (pos + 1) : (n_blk_tok + (pos + 1 - tail_start));
    const int* sb = (nsel >= 0) ? (sel_blocks + static_cast<size_t>(m) * geom::IDX_TOP_BLOCKS) : nullptr;

    constexpr int LD = 264; // +8 BF16 padding -> 16B aligned & bank-conflict-free
    __shared__ __nv_bfloat16 sQ[16][LD];
    __shared__ __nv_bfloat16 sKV[64][LD];
    __shared__ __nv_bfloat16 sP[16][72];
    __shared__ float s_wm[16][8], s_wl[16][8], s_mmax[16], s_lsum[16], s_alpha[16];
    __shared__ int s_kvpos[64];

    for (int i = tid; i < 16 * 32; i += 256) {
        const int qh = i >> 5, d8 = (i & 31) << 3;
        const int lq = gq_base + qh;
        uint4 qv = make_uint4(0, 0, 0, 0);
        if (qh < geom::GQA && lq >= 0 && lq < nh_q)
            qv = *reinterpret_cast<const uint4*>(q + (static_cast<size_t>(m) * nh_q + lq) * 256 + d8);
        *reinterpret_cast<uint4*>(sQ[qh] + d8) = qv;
    }
    if (tid < 16) { s_mmax[tid] = -1e30f; s_lsum[tid] = 0.0f; }

    const size_t page_stride = static_cast<size_t>(2) * geom::KV_PAGE * nh_kv * 256;
    const size_t plane_stride = static_cast<size_t>(geom::KV_PAGE) * nh_kv * 256;
    const __nv_bfloat16* slot_base = kv_pages + static_cast<size_t>(slot) * max_pages * page_stride;
    constexpr float scale = 0.0625f;

    float oc[4][4] = {};
    __syncthreads();

    for (int t0 = 0; t0 < total_tok; t0 += 64) {
        const int n_tok = min(64, total_tok - t0);
        if (tid < 64) {
            const int k = t0 + tid;
            int kvp = -1;
            if (tid < n_tok) {
                if (nsel < 0) kvp = k;
                else if (k < n_blk_tok) kvp = sb[k >> 2] * geom::IDX_COMPRESS + (k & 3);
                else kvp = tail_start + (k - n_blk_tok);
            }
            s_kvpos[tid] = kvp;
        }
        __syncthreads();

        for (int i = tid; i < 64 * 32; i += 256) {
            const int j = i >> 5, d8 = (i & 31) << 3;
            const int kvp = s_kvpos[j];
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (kvp >= 0) {
                const int page = kvp / geom::KV_PAGE, off = kvp % geom::KV_PAGE;
                kv = *reinterpret_cast<const uint4*>(
                    slot_base + static_cast<size_t>(page) * page_stride + (static_cast<size_t>(off) * nh_kv + lkv) * 256 + d8);
            }
            *reinterpret_cast<uint4*>(sKV[j] + d8) = kv;
        }
        __syncthreads();

        float sc[4] = {0.f, 0.f, 0.f, 0.f};
        const int k_base = warp * 8;
        if (k_base < n_tok) {
            const int q_r = lane & 15, q_c = (lane >> 4) * 8;
            const int k_r = lane & 7,  k_c = ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int d0 = 0; d0 < 256; d0 += 16) {
                uint32_t qa[4], kb[2];
                attn_ldsm4(qa, sQ[q_r] + d0 + q_c);
                attn_ldsm2(kb, sKV[k_base + k_r] + d0 + k_c);
                attn_mma_bf16_m16n8k16(sc, qa, kb);
            }
        }
        const int c0 = k_base + 2 * tid4, c1 = c0 + 1;
        const float s0 = (c0 < n_tok) ? (sc[0] * scale) : -1e30f;
        const float s1 = (c1 < n_tok) ? (sc[1] * scale) : -1e30f;
        const float s2 = (c0 < n_tok) ? (sc[2] * scale) : -1e30f;
        const float s3 = (c1 < n_tok) ? (sc[3] * scale) : -1e30f;

        float ma = fmaxf(s0, s1), mb = fmaxf(s2, s3);
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
        if (tid4 == 0) { s_wm[gid][warp] = ma; s_wm[gid + 8][warp] = mb; }
        __syncthreads();

        if (tid < 16) {
            float tmax = s_mmax[tid];
            #pragma unroll
            for (int w = 0; w < 8; ++w) tmax = fmaxf(tmax, s_wm[tid][w]);
            s_alpha[tid] = __expf(s_mmax[tid] - tmax);
            s_mmax[tid] = tmax;
        }
        __syncthreads();

        const float nma = s_mmax[gid], nmb = s_mmax[gid + 8];
        const float p0 = (c0 < n_tok) ? __expf(s0 - nma) : 0.0f;
        const float p1 = (c1 < n_tok) ? __expf(s1 - nma) : 0.0f;
        const float p2 = (c0 < n_tok) ? __expf(s2 - nmb) : 0.0f;
        const float p3 = (c1 < n_tok) ? __expf(s3 - nmb) : 0.0f;
        *reinterpret_cast<__nv_bfloat162*>(sP[gid] + c0)     = __floats2bfloat162_rn(p0, p1);
        *reinterpret_cast<__nv_bfloat162*>(sP[gid + 8] + c0) = __floats2bfloat162_rn(p2, p3);

        float la = p0 + p1, lb = p2 + p3;
        la += __shfl_xor_sync(0xffffffffu, la, 1); la += __shfl_xor_sync(0xffffffffu, la, 2);
        lb += __shfl_xor_sync(0xffffffffu, lb, 1); lb += __shfl_xor_sync(0xffffffffu, lb, 2);
        if (tid4 == 0) { s_wl[gid][warp] = la; s_wl[gid + 8][warp] = lb; }
        __syncthreads();

        if (tid < 16) {
            float l_add = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) l_add += s_wl[tid][w];
            s_lsum[tid] = s_lsum[tid] * s_alpha[tid] + l_add;
        }
        const float aa = s_alpha[gid], ab = s_alpha[gid + 8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            oc[j][0] *= aa; oc[j][1] *= aa;
            oc[j][2] *= ab; oc[j][3] *= ab;
        }

        for (int i = tid; i < 64 * 32; i += 256) {
            const int j = i >> 5, d8 = (i & 31) << 3;
            const int kvp = s_kvpos[j];
            uint4 vv = make_uint4(0, 0, 0, 0);
            if (kvp >= 0) {
                const int page = kvp / geom::KV_PAGE, off = kvp % geom::KV_PAGE;
                vv = *reinterpret_cast<const uint4*>(
                    slot_base + static_cast<size_t>(page) * page_stride + plane_stride + (static_cast<size_t>(off) * nh_kv + lkv) * 256 + d8);
            }
            *reinterpret_cast<uint4*>(sKV[j] + d8) = vv;
        }
        __syncthreads();

        const int d_base = warp * 32;
        const int p_r = lane & 15, p_c = (lane >> 4) * 8;
        const int v_r = lane & 15;
        for (int k0 = 0; k0 < n_tok; k0 += 16) {
            uint32_t pa[4];
            attn_ldsm4(pa, sP[p_r] + k0 + p_c);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t vb[2];
                attn_ldsm2_trans(vb, sKV[k0 + v_r] + d_base + j * 8);
                attn_mma_bf16_m16n8k16(oc[j], pa, vb);
            }
        }
        __syncthreads();
    }

    const int d_base = warp * 32;
    const int lq_a = gq_base + gid, lq_b = gq_base + gid + 8;
    const bool va = (gid < geom::GQA && lq_a >= 0 && lq_a < nh_q);
    const bool vb = (gid + 8 < geom::GQA && lq_b >= 0 && lq_b < nh_q);
    const float inv_a = (s_lsum[gid] > 0.f) ? (1.0f / s_lsum[gid]) : 0.0f;
    const float inv_b = (s_lsum[gid + 8] > 0.f) ? (1.0f / s_lsum[gid + 8]) : 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dc = d_base + j * 8 + 2 * tid4;
        if (va) {
            const size_t qh = static_cast<size_t>(m) * nh_q + lq_a;
            const float2 fg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + qh * 256 + dc));
            const float o0 = __bfloat162float(__float2bfloat16(oc[j][0] * inv_a));
            const float o1 = __bfloat162float(__float2bfloat16(oc[j][1] * inv_a));
            *reinterpret_cast<__nv_bfloat162*>(out + qh * 256 + dc) = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
        }
        if (vb) {
            const size_t qh = static_cast<size_t>(m) * nh_q + lq_b;
            const float2 fg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + qh * 256 + dc));
            const float o0 = __bfloat162float(__float2bfloat16(oc[j][2] * inv_b));
            const float o1 = __bfloat162float(__float2bfloat16(oc[j][3] * inv_b));
            *reinterpret_cast<__nv_bfloat162*>(out + qh * 256 + dc) = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
        }
    }
}

// Per-segment Dense Causal Tensor-Core FlashAttention (BM=16 queries x 4 Q-heads per CTA -> 16x KV reuse).
// Grid: dim3((seg_len + 15) / 16, nh_kv, 3), Block: 128 (4 warps = 4 Q-heads). Works for TP=1..4.
__global__ void __launch_bounds__(128, 2) k_qw38_gqa_dense_tc16_seg(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ kv_pages,
    __nv_bfloat16* __restrict__ out,
    int seg_row0, int seg_len, int seg_slot, int seg_pos0,
    int nh_q, int nh_kv, int q0_global, int kv0_global, int max_pages) {
    const int m0 = blockIdx.x * 16;
    const int lkv = blockIdx.y;
    const int qgrp = blockIdx.z; // 0..2 -> 4 Q-heads each
    if (m0 >= seg_len || lkv >= nh_kv) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int qh_local = qgrp * 4 + warp;
    const int lq = (kv0_global + lkv) * geom::GQA + qh_local - q0_global;
    const bool active_q = (qh_local < geom::GQA && lq >= 0 && lq < nh_q);

    const int pos0 = seg_pos0 + m0;
    const int n_m = min(16, seg_len - m0);
    const int plast = pos0 + n_m - 1;
    const int ra = m0 + gid, rb = ra + 8;
    const int ga = seg_row0 + ra, gb = seg_row0 + rb;
    const bool va = active_q && (ra < seg_len), vb = active_q && (rb < seg_len);
    const int pa = pos0 + gid, pb = pa + 8;

    constexpr int LD = 264;
    __shared__ __nv_bfloat16 sKV[32][LD];

    const size_t page_stride = static_cast<size_t>(2) * geom::KV_PAGE * nh_kv * 256;
    const size_t plane_stride = static_cast<size_t>(geom::KV_PAGE) * nh_kv * 256;
    const __nv_bfloat16* slot_base = kv_pages + static_cast<size_t>(seg_slot) * max_pages * page_stride;
    const __nv_bfloat16* qpa = q + (static_cast<size_t>(va ? ga : 0) * nh_q + (active_q ? lq : 0)) * 256;
    const __nv_bfloat16* qpb = q + (static_cast<size_t>(vb ? gb : 0) * nh_q + (active_q ? lq : 0)) * 256;

    float oc[32][4];
    #pragma unroll
    for (int j = 0; j < 32; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.0f;
    float ma = -1e30f, mb = -1e30f, la = 0.0f, lb = 0.0f;
    constexpr float scale = 0.0625f;

    const int k_r = lane & 7, k_c = ((lane >> 3) & 1) * 8;
    const int v_r = lane & 15;

    for (int kb = 0; kb <= plast; kb += 32) {
        for (int i = tid; i < 32 * 32; i += 128) {
            const int j = i >> 5, d8 = (i & 31) << 3;
            const int kpos = kb + j;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (kpos <= plast) {
                const int page = kpos / geom::KV_PAGE, off = kpos % geom::KV_PAGE;
                kv = *reinterpret_cast<const uint4*>(
                    slot_base + static_cast<size_t>(page) * page_stride + (static_cast<size_t>(off) * nh_kv + lkv) * 256 + d8);
            }
            *reinterpret_cast<uint4*>(sKV[j] + d8) = kv;
        }
        __syncthreads();

        float sr[4][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) sr[j][0] = sr[j][1] = sr[j][2] = sr[j][3] = 0.0f;
        if (active_q) {
            for (int d0 = 0; d0 < 256; d0 += 16) {
                uint32_t qa[4];
                const int c = d0 + 2 * tid4;
                qa[0] = va ? *reinterpret_cast<const uint32_t*>(qpa + c) : 0u;
                qa[1] = vb ? *reinterpret_cast<const uint32_t*>(qpb + c) : 0u;
                qa[2] = va ? *reinterpret_cast<const uint32_t*>(qpa + c + 8) : 0u;
                qa[3] = vb ? *reinterpret_cast<const uint32_t*>(qpb + c + 8) : 0u;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t kb_r[2];
                    attn_ldsm2(kb_r, sKV[j * 8 + k_r] + d0 + k_c);
                    attn_mma_bf16_m16n8k16(sr[j], qa, kb_r);
                }
            }
        }
        __syncthreads();

        for (int i = tid; i < 32 * 32; i += 128) {
            const int j = i >> 5, d8 = (i & 31) << 3;
            const int kpos = kb + j;
            uint4 vv = make_uint4(0, 0, 0, 0);
            if (kpos <= plast) {
                const int page = kpos / geom::KV_PAGE, off = kpos % geom::KV_PAGE;
                vv = *reinterpret_cast<const uint4*>(
                    slot_base + static_cast<size_t>(page) * page_stride + plane_stride + (static_cast<size_t>(off) * nh_kv + lkv) * 256 + d8);
            }
            *reinterpret_cast<uint4*>(sKV[j] + d8) = vv;
        }

        uint32_t pf[2][4];
        if (active_q) {
            float xa = -1e30f, xb = -1e30f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k0 = kb + j * 8 + 2 * tid4, k1 = k0 + 1;
                sr[j][0] = (va && k0 <= pa) ? (sr[j][0] * scale) : -1e30f;
                sr[j][1] = (va && k1 <= pa) ? (sr[j][1] * scale) : -1e30f;
                sr[j][2] = (vb && k0 <= pb) ? (sr[j][2] * scale) : -1e30f;
                sr[j][3] = (vb && k1 <= pb) ? (sr[j][3] * scale) : -1e30f;
                xa = fmaxf(xa, fmaxf(sr[j][0], sr[j][1]));
                xb = fmaxf(xb, fmaxf(sr[j][2], sr[j][3]));
            }
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 1));
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 2));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 1));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 2));
            const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
            const float aa = __expf(ma - na), ab = __expf(mb - nb);
            ma = na; mb = nb;
            la *= aa; lb *= ab;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                oc[j][0] *= aa; oc[j][1] *= aa;
                oc[j][2] *= ab; oc[j][3] *= ab;
            }
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float* c = sr[2 * kk + h];
                    const float p0 = (c[0] > -1e29f) ? __expf(c[0] - na) : 0.f;
                    const float p1 = (c[1] > -1e29f) ? __expf(c[1] - na) : 0.f;
                    const float p2 = (c[2] > -1e29f) ? __expf(c[2] - nb) : 0.f;
                    const float p3 = (c[3] > -1e29f) ? __expf(c[3] - nb) : 0.f;
                    la += p0 + p1; lb += p2 + p3;
                    const __nv_bfloat162 b01 = __floats2bfloat162_rn(p0, p1);
                    const __nv_bfloat162 b23 = __floats2bfloat162_rn(p2, p3);
                    pf[kk][h * 2 + 0] = *reinterpret_cast<const uint32_t*>(&b01);
                    pf[kk][h * 2 + 1] = *reinterpret_cast<const uint32_t*>(&b23);
                }
            }
        }
        __syncthreads();

        if (active_q) {
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    uint32_t vb_r[2];
                    attn_ldsm2_trans(vb_r, sKV[kk * 16 + v_r] + j * 8);
                    attn_mma_bf16_m16n8k16(oc[j], pf[kk], vb_r);
                }
            }
        }
        __syncthreads();
    }

    if (!active_q) return;
    la += __shfl_xor_sync(0xffffffffu, la, 1); la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1); lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    const float inv_a = (la > 0.f) ? (1.0f / la) : 0.0f;
    const float inv_b = (lb > 0.f) ? (1.0f / lb) : 0.0f;
    const size_t qha = static_cast<size_t>(ga) * nh_q + lq;
    const size_t qhb = static_cast<size_t>(gb) * nh_q + lq;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int dc = j * 8 + 2 * tid4;
        if (va) {
            const float2 fg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + qha * 256 + dc));
            const float o0 = __bfloat162float(__float2bfloat16(oc[j][0] * inv_a));
            const float o1 = __bfloat162float(__float2bfloat16(oc[j][1] * inv_a));
            *reinterpret_cast<__nv_bfloat162*>(out + qha * 256 + dc) = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
        }
        if (vb) {
            const float2 fg = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + qhb * 256 + dc));
            const float o0 = __bfloat162float(__float2bfloat16(oc[j][2] * inv_b));
            const float o1 = __bfloat162float(__float2bfloat16(oc[j][3] * inv_b));
            *reinterpret_cast<__nv_bfloat162*>(out + qhb * 256 + dc) = __floats2bfloat162_rn(o0 * sigmoid_f(fg.x), o1 * sigmoid_f(fg.y));
        }
    }
}

// Execute one full-attention + QSA layer on rank `rank` of `nr`.
// Writes this rank's partial output [M, 2560] BF16 into `out_part`.
inline void full_attn_forward(cublasHandle_t blas, const FullAttnWeights& w, FullLayerKv& kv, FullAttnScratch& sc,
                              const __nv_bfloat16* x, __nv_bfloat16* out_part,
                              const int* row_slot_dev, const int* row_pos_dev,
                              const StepSeg* segs, int nsegs, int M, int nr, int rank,
                              int max_pages, int max_blocks, cudaStream_t st, int snap_k = 0) {
    const int nh_q = geom::nh_q(nr);
    const int nh_kv = geom::nh_kv(nr, rank);
    const int q0 = geom::q0(nr, rank);
    const int kv0 = geom::kv0(nr, rank);

    int max_pos = 0;
    bool all_len1 = (nsegs == M);
    const int V = snap_k + 1;
    bool all_verify = (snap_k > 0 && nsegs > 0 && nsegs * V == M);
    bool all_qsa_tc = true;
    bool all_dense_tc16 = (nsegs > 0);
    int seg_rows_sum = 0;
    for (int s = 0; s < nsegs; ++s) {
        if (segs[s].len > 0) {
            const int ep = segs[s].pos0 + segs[s].len - 1;
            if (ep > max_pos) max_pos = ep;
            const bool seg_qsa = ((ep + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS;
            if (seg_qsa && segs[s].len < 16) all_qsa_tc = false;
            if (segs[s].len < 16) all_dense_tc16 = false;
            seg_rows_sum += segs[s].len;
        }
        if (segs[s].len != 1 || segs[s].row0 != s) all_len1 = false;
        if (segs[s].len != V || segs[s].row0 != s * V) all_verify = false;
    }
    if (seg_rows_sum != M) { all_qsa_tc = false; all_dense_tc16 = false; }
    const bool need_qsa = ((max_pos + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS;

    // 1. Projections (Q+gate, K, V, and QSA index_qk)
    if ((M <= 12 && w.fp8_q.q) || M <= 8) {
        if (w.fp8_q.q) {
            launch_fp8_gemv_smallm<false>(x, w.fp8_q, sc.q_gate, M, st);
            launch_fp8_gemv_smallm<false>(x, w.fp8_k, sc.k, M, st);
            launch_fp8_gemv_smallm<false>(x, w.fp8_v, sc.v, M, st);
            launch_fp8_gemv_smallm<false>(x, w.fp8_idx, sc.idx_qk, M, st);
        } else {
            launch_bf16_gemv_smallm<false>(x, w.q_proj, sc.q_gate, M, nh_q * geom::HEAD_DIM * 2, geom::HID, st);
            launch_bf16_gemv_smallm<false>(x, w.k_proj, sc.k, M, nh_kv * geom::HEAD_DIM, geom::HID, st);
            launch_bf16_gemv_smallm<false>(x, w.v_proj, sc.v, M, nh_kv * geom::HEAD_DIM, geom::HID, st);
            launch_bf16_gemv_smallm<false>(x, w.idx_qk_proj, sc.idx_qk, M, geom::IDX_PROJ_OUT, geom::HID, st);
        }
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, nh_q * geom::HEAD_DIM * 2, geom::HID, x, w.q_proj, sc.q_gate);
        gemm_bf16_nt(blas, M, nh_kv * geom::HEAD_DIM, geom::HID, x, w.k_proj, sc.k);
        gemm_bf16_nt(blas, M, nh_kv * geom::HEAD_DIM, geom::HID, x, w.v_proj, sc.v);
        gemm_bf16_nt(blas, M, geom::IDX_PROJ_OUT, geom::HID, x, w.idx_qk_proj, sc.idx_qk);
    }

    // 2. Q norm + RoPE, K norm + RoPE + paged KV scatter
    k_qw38_q_norm_rope<<<M * nh_q, 256, 0, st>>>(sc.q_gate, w.q_norm, row_pos_dev, sc.q, sc.gate, M, nh_q, geom::RMS_EPS);
    k_qw38_kv_write<<<M * nh_kv, 256, 0, st>>>(sc.k, sc.v, w.k_norm, row_slot_dev, row_pos_dev, kv.kv_pages, M, nh_kv, max_pages, geom::RMS_EPS);

    // 3. QSA indexer K block-4 pooling update (always maintain idx_pages); skip Q norm/scoring/top-512 when max_pos < 2051!
    if (all_len1) {
        k_qsa_k_update_dec_batch<<<M, 128, 0, st>>>(sc.idx_qk, w.idx_k_norm, kv.idx_raw, kv.idx_pages,
                                                    row_slot_dev, row_pos_dev, max_pages, M, geom::RMS_EPS);
    } else if (all_verify) {
        k_qsa_k_update_verify_batch<<<nsegs, 128, 0, st>>>(sc.idx_qk, w.idx_k_norm, kv.idx_raw, kv.idx_raw_snap, kv.idx_pages,
                                                           row_slot_dev, row_pos_dev, max_pages, nsegs, V, snap_k, geom::RMS_EPS);
    } else {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len <= 0) continue;
            k_qsa_k_update_seg<<<1, 128, 0, st>>>(sc.idx_qk, w.idx_k_norm, kv.idx_raw,
                                                  snap_k > 0 ? kv.idx_raw_snap : nullptr, kv.idx_pages,
                                                  max_pages, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len, snap_k, geom::RMS_EPS);
        }
    }
    if (need_qsa) {
        k_qsa_q_norm_rope<<<M * geom::IDX_HEADS, 128, 0, st>>>(sc.idx_qk, w.idx_q_norm, row_pos_dev, sc.idx_q, M, geom::RMS_EPS);
        if (all_qsa_tc) {
            for (int s = 0; s < nsegs; ++s) {
                if (segs[s].len <= 0) continue;
                const int seg_max_nblk = (segs[s].pos0 + segs[s].len) / geom::IDX_COMPRESS;
                if (seg_max_nblk <= geom::IDX_TOP_BLOCKS) continue;
                const dim3 grid((seg_max_nblk + 63) / 64, (segs[s].len + 63) / 64);
                k_qsa_score_blocks_tc_seg<<<grid, 256, 0, st>>>(
                    sc.idx_q, kv.idx_pages, sc.idx_scores, max_pages, max_blocks,
                    segs[s].row0, segs[s].len, segs[s].slot, segs[s].pos0);
            }
        } else {
            k_qsa_score_blocks<<<M, 256, 0, st>>>(sc.idx_q, kv.idx_pages, row_slot_dev, row_pos_dev, sc.idx_scores, max_pages, max_blocks, M);
        }
        k_qsa_select_top512<<<M, 256, 0, st>>>(sc.idx_scores, row_pos_dev, sc.sel_blocks, sc.sel_counts, max_blocks, M);
    }

    // 4. GQA-12 FlashAttention:
    //    - Prefill (M > 8): Tensor-Core BM=16 dense FlashAttention when all segments are dense >=16,
    //      else Tensor-Core GQA-12 Head-MMA (works for QSA sparse and mixed dense/sparse).
    //    - Decode (M <= 8): Tiled FlashAttention + Split-KV.
    if (M > 8) {
        if (!need_qsa && all_dense_tc16) {
            for (int s = 0; s < nsegs; ++s) {
                if (segs[s].len <= 0) continue;
                const dim3 grid((segs[s].len + 15) / 16, nh_kv, 3);
                k_qw38_gqa_dense_tc16_seg<<<grid, 128, 0, st>>>(
                    sc.q, sc.gate, kv.kv_pages, sc.attn_out,
                    segs[s].row0, segs[s].len, segs[s].slot, segs[s].pos0,
                    nh_q, nh_kv, q0, kv0, max_pages);
            }
        } else {
            k_qw38_gqa_tc_head12<<<nh_kv * M, 256, 0, st>>>(
                sc.q, sc.gate, kv.kv_pages, row_slot_dev, row_pos_dev,
                need_qsa ? sc.sel_blocks : nullptr, need_qsa ? sc.sel_counts : nullptr,
                sc.attn_out, M, nh_q, nh_kv, q0, kv0, max_pages);
        }
    } else {
        const int NS = gqa_split_count(M, nh_kv, max_pos, sc.split_o != nullptr);
        k_qw38_gqa_flash_tiled<<<M * nh_kv * NS, 384, 0, st>>>(
            sc.q, sc.gate, kv.kv_pages, row_slot_dev, row_pos_dev,
            need_qsa ? sc.sel_blocks : nullptr, need_qsa ? sc.sel_counts : nullptr,
            sc.attn_out, sc.split_m, sc.split_l, sc.split_o,
            M, nh_q, nh_kv, q0, kv0, max_pages, NS);
        if (NS > 1) {
            k_qw38_gqa_split_reduce<<<M * nh_q, 32, 0, st>>>(
                sc.split_m, sc.split_l, sc.split_o, sc.gate, sc.attn_out, M, nh_q, NS);
        }
    }

    if ((M <= 12 && w.fp8_o.q) || M <= 8) {
        if (w.fp8_o.q) {
            launch_fp8_gemv_smallm<false>(sc.attn_out, w.fp8_o, out_part, M, st);
        } else {
            launch_bf16_gemv_smallm<false>(sc.attn_out, w.o_proj, out_part, M, geom::HID, nh_q * geom::HEAD_DIM, st);
        }
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, geom::HID, nh_q * geom::HEAD_DIM, sc.attn_out, w.o_proj, out_part);
    }
}

} // namespace qw38
