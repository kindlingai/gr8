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
                                   __nv_bfloat16* __restrict__ idx_pages,
                                   int max_pages, int slot, int row0, int pos0, int len, float eps) {
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

// Execute one full-attention + QSA layer on rank `rank` of `nr`.
// Writes this rank's partial output [M, 2560] BF16 into `out_part`.
inline void full_attn_forward(cublasHandle_t blas, const FullAttnWeights& w, FullLayerKv& kv, FullAttnScratch& sc,
                              const __nv_bfloat16* x, __nv_bfloat16* out_part,
                              const int* row_slot_dev, const int* row_pos_dev,
                              const StepSeg* segs, int nsegs, int M, int nr, int rank,
                              int max_pages, int max_blocks, cudaStream_t st) {
    const int nh_q = geom::nh_q(nr);
    const int nh_kv = geom::nh_kv(nr, rank);
    const int q0 = geom::q0(nr, rank);
    const int kv0 = geom::kv0(nr, rank);

    int max_pos = 0;
    bool all_len1 = (nsegs == M);
    for (int s = 0; s < nsegs; ++s) {
        if (segs[s].len > 0) {
            const int ep = segs[s].pos0 + segs[s].len - 1;
            if (ep > max_pos) max_pos = ep;
        }
        if (segs[s].len != 1 || segs[s].row0 != s) all_len1 = false;
    }
    const bool need_qsa = ((max_pos + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS;

    // 1. Projections (Q+gate, K, V, and QSA index_qk)
    if (M <= 8) {
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
    } else {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len <= 0) continue;
            k_qsa_k_update_seg<<<1, 128, 0, st>>>(sc.idx_qk, w.idx_k_norm, kv.idx_raw, kv.idx_pages,
                                                  max_pages, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len, geom::RMS_EPS);
        }
    }
    if (need_qsa) {
        k_qsa_q_norm_rope<<<M * geom::IDX_HEADS, 128, 0, st>>>(sc.idx_qk, w.idx_q_norm, row_pos_dev, sc.idx_q, M, geom::RMS_EPS);
        k_qsa_score_blocks<<<M, 256, 0, st>>>(sc.idx_q, kv.idx_pages, row_slot_dev, row_pos_dev, sc.idx_scores, max_pages, max_blocks, M);
        k_qsa_select_top512<<<M, 256, 0, st>>>(sc.idx_scores, row_pos_dev, sc.sel_blocks, sc.sel_counts, max_blocks, M);
    }

    // 4. GQA-12 Tiled FlashAttention (+ Split-KV for decode M <= 8 & ctx >= 128) + output projection
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

    if (M <= 8) {
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
