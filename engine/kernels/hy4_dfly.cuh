// Tencent Hy4-flash Built-in 0.85B DFly2 Speculative Drafter (model.dfly_layers.*).
// Verified verbatim against Tencent Hy4-flash checkpoint config.json and vLLM HYV4DFlyForCausalLM:
//   - Query block W = K + 1 (default K = 5 -> W = 6):
//       input_ids = [bonus_tok, 120009, 120009, 120009, 120009, 120009]
//       positions = [anchor, anchor+1, ..., anchor+W-1] where anchor = source_pos + 1.
//   - Slot-0 MTP fusion:
//       h[0] = h[0] + mtp_gate * eh_proj(cat(enorm(h[0]), hnorm(target_hidden)))
//   - 5 DFly layers (pre-norm residual, HC=1, no Engram, rms_norm_eps=1e-5):
//       1. Gated Absorbed MLA (32 heads, q_lora=1024, kv_lora=512, qk_nope=192, qk_rope=64, v=256,
//          rope_theta=1e7, learnable_sink_param [32], post_proj_norm):
//          Reuses target DSA layers [25, 29, 33, 37, 41] KV cache (BF16/FP8/INT4) and snapshotted
//          top-2048 history indices (zero extra KV cache allocation!), plus causal self-attention
//          over the W draft tokens [0..q_off] stored in a tiny scratch buffer!
//       2. Dense SwiGLU MLP (I=9216, TP-sharded 9216/nr, NO clamp, NO post_fc2_norm).
//   - Final norm: H_draft = final_norm(h) [W, 3072].
//   - Sequential hidden_correction + shared target lm_head (step = 0..K-1):
//       comb   = cat(hidden_norm(H_draft[step + 1]), embed_norm(embed_tokens[prev_tok]))
//       h_corr = H_draft[step + 1] + down_proj(silu(gate_proj(comb)) * up_proj(comb))
//       logits = lm_head(h_corr) -> draft_tok[step] (prev_tok = draft_tok[step]).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <vector>
#include "../model_geom_hy4.h"
#include "../attn/hy4_attn.cuh"
#include "hy4_hc.cuh"
#include "hy4_mxfp8.cuh"
#include "hy4_moe.cuh"

namespace hy4 {
namespace dfly {

constexpr int NUM_LAYERS = 5;
constexpr int HEADS      = 32;
constexpr int MLP_I      = 9216;
constexpr int HC_I       = 3072;
constexpr int MASK_TOK   = 120009;
constexpr int TARGET_DSA_LAYERS[NUM_LAYERS] = {25, 29, 33, 37, 41};

inline int target_dsa_to_dfly_idx(int layer) {
    for (int d = 0; d < NUM_LAYERS; ++d) {
        if (TARGET_DSA_LAYERS[d] == layer) return d;
    }
    return -1;
}

struct LayerWeights {
    int nh_loc = 0;
    int i_loc  = 0;
    const __nv_bfloat16* input_norm     = nullptr; // [3072]
    const __nv_bfloat16* post_attn_norm = nullptr; // [3072]
    mx8::LinW            q_a_proj{};               // [1024, 3072]
    const __nv_bfloat16* q_a_ln         = nullptr; // [1024]
    mx8::LinW            q_b_proj{};               // [nh_loc * 256, 1024]
    mx8::LinW            kv_a_proj{};              // [576, 3072]
    const __nv_bfloat16* kv_a_ln        = nullptr; // [512]
    mx8::LinW            w_uk_t{};                 // [nh_loc * 512, 192]
    mx8::LinW            w_uv{};                   // [nh_loc * 256, 512]
    mx8::LinW            gate_proj{};              // [nh_loc * 256, 3072]
    mx8::LinW            o_proj{};                 // [3072, nh_loc * 256]
    const __nv_bfloat16* post_proj_ln   = nullptr; // [3072]
    const float*         sinks          = nullptr; // [nh_loc] FP32 (converted from BF16 at boot!)
    mx8::LinW            mlp_gate{};               // [i_loc, 3072]
    mx8::LinW            mlp_up{};                 // [i_loc, 3072]
    mx8::LinW            mlp_down{};               // [3072, i_loc]
};

struct RankWeights {
    const __nv_bfloat16* mtp_enorm      = nullptr; // [3072]
    const __nv_bfloat16* mtp_hnorm      = nullptr; // [3072]
    mx8::LinW            mtp_eh_proj{};            // [3072, 6144]
    const __nv_bfloat16* mtp_gate       = nullptr; // [3072]
    const __nv_bfloat16* final_norm     = nullptr; // [3072]
    const __nv_bfloat16* hc_hidden_norm = nullptr; // [3072]
    const __nv_bfloat16* hc_embed_norm  = nullptr; // [3072]
    mx8::LinW            hc_gate_proj{};           // [3072, 6144]
    mx8::LinW            hc_up_proj{};             // [3072, 6144]
    mx8::LinW            hc_down_proj{};           // [3072, 3072]
    const __nv_bfloat16* conf_w         = nullptr; // [3072]
    const __nv_bfloat16* conf_b         = nullptr; // [1]
    LayerWeights         layers[NUM_LAYERS]{};
};

struct alignas(16) DflyGraphMeta {
    int slot;
    int bonus_tok;
    int anchor;
    int rope_delta;
    int snap_row;
    int pad[3];
};

struct alignas(16) DflyArgPair {
    float val;
    int id;
    int pad0;
    int pad1;
};

struct Scratch {
    int max_W = 0;
    __nv_bfloat16* h          = nullptr; // [max_W, 3072]
    __nv_bfloat16* x_ln       = nullptr; // [max_W, 3072]
    __nv_bfloat16* kv_draft   = nullptr; // [max_W, 576] BF16 (ephemeral draft KV for current layer!)
    __nv_bfloat16* mlp_gate   = nullptr; // [max_W, i_loc]
    __nv_bfloat16* mlp_up     = nullptr; // [max_W, i_loc]
    __nv_bfloat16* cat6144    = nullptr; // [6144]
    __nv_bfloat16* tmp3072    = nullptr; // [3072]
    __nv_bfloat16* hc_gate    = nullptr; // [3072]
    __nv_bfloat16* hc_up      = nullptr; // [3072]
    int*           d_q_pos    = nullptr; // [max_W]
    int*           d_draft_id = nullptr; // [max_W] (GPU-side argmax chain for zero-sync TP=1/TP=2 greedy!)
    float*         d_max_v    = nullptr; // [256] (partial argmax vals)
    int*           d_max_i    = nullptr; // [256] (partial argmax ids)
    DflyGraphMeta* d_meta     = nullptr; // [1] (device metadata for CUDA-Graph DFly2!)
    DflyArgPair*   d_arg_pair = nullptr; // [1] (16B local argmax pair for TP=2 GPU-side RDMA merge!)
};

// =================================================================================================
// Small helper kernels
// =================================================================================================

__global__ void k_dfly_bf16_to_f32(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Initialize DFly2 query block h[0..W-1, 3072] and positions d_q_pos[0..W-1] = anchor + 0..W-1.
// When d_meta != nullptr, reads (bonus_tok, anchor, rope_delta) on device and seeds d_draft_id[0] = bonus_tok (graph-safe!).
// Grid: (W * 12), Block: 256.
__global__ void __launch_bounds__(256) k_dfly_init_block(
    const __nv_bfloat16* __restrict__ embed,
    __nv_bfloat16* __restrict__ h,
    int* __restrict__ d_q_pos,
    int bonus_tok, int anchor, int rope_delta, int W,
    const DflyGraphMeta* __restrict__ d_meta = nullptr,
    int* __restrict__ d_draft_id = nullptr) {
    const int w = blockIdx.x / 12;
    const int d = (blockIdx.x % 12) * 256 + threadIdx.x;
    if (w >= W || d >= geom::HID) return;
    if (d_meta != nullptr) {
        bonus_tok  = d_meta->bonus_tok;
        anchor     = d_meta->anchor;
        rope_delta = d_meta->rope_delta;
    }
    const int tok = (w == 0) ? bonus_tok : MASK_TOK;
    h[static_cast<size_t>(w) * geom::HID + d] = embed[static_cast<size_t>(tok) * geom::HID + d];
    if (d == 0) {
        d_q_pos[w] = anchor + w + rope_delta;
        if (w == 0 && d_draft_id != nullptr) d_draft_id[0] = bonus_tok;
    }
}

// Compute cat(rmsnorm(a, w_a), rmsnorm(b, w_b)) -> out[6144] BF16 (2 blocks: block 0 does a -> [0..3071], block 1 does b -> [3072..6143]).
// Grid: 2, Block: 256.
__global__ void __launch_bounds__(256) k_dfly_cat_norms(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ wa,
    const __nv_bfloat16* __restrict__ b, const __nv_bfloat16* __restrict__ wb,
    __nv_bfloat16* __restrict__ out,
    const DflyGraphMeta* __restrict__ d_meta = nullptr) {
    const int which = blockIdx.x;
    const __nv_bfloat16* b_eff = (d_meta != nullptr) ? (b + static_cast<size_t>(d_meta->snap_row) * geom::HID) : b;
    const __nv_bfloat16* x = (which == 0) ? a : b_eff;
    const __nv_bfloat16* w = (which == 0) ? wa : wb;
    __nv_bfloat16* y = out + static_cast<size_t>(which) * geom::HID;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    float xs[12], ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const float v = __bfloat162float(x[tid + k * 256]);
        xs[k] = v;
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(xs[k] * inv);
        y[d] = __float2bfloat16(__bfloat162float(w[d]) * __bfloat162float(xn));
    }
}

// Same as k_dfly_cat_norms, but `b` is looked up on GPU from embed[d_tok_ptr[0]] (zero host sync in greedy chain!).
// Grid: 2, Block: 256.
__global__ void __launch_bounds__(256) k_dfly_cat_norms_embed(
    const __nv_bfloat16* __restrict__ h_row, const __nv_bfloat16* __restrict__ wh,
    const __nv_bfloat16* __restrict__ embed, const int* __restrict__ d_prev_tok, const __nv_bfloat16* __restrict__ we,
    __nv_bfloat16* __restrict__ out) {
    const int which = blockIdx.x;
    const __nv_bfloat16* x = (which == 0) ? h_row : (embed + static_cast<size_t>(d_prev_tok[0]) * geom::HID);
    const __nv_bfloat16* w = (which == 0) ? wh : we;
    __nv_bfloat16* y = out + static_cast<size_t>(which) * geom::HID;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    float xs[12], ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const float v = __bfloat162float(x[tid + k * 256]);
        xs[k] = v;
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const __nv_bfloat16 xn = __float2bfloat16(xs[k] * inv);
        y[d] = __float2bfloat16(__bfloat162float(w[d]) * __bfloat162float(xn));
    }
}

// Slot-0 MTP fusion residual: h0[d] = h0[d] + gate[d] * fused[d].
// Grid: 12, Block: 256.
__global__ void __launch_bounds__(256) k_dfly_mtp_add(
    __nv_bfloat16* __restrict__ h0,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ fused) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d < geom::HID) {
        const float g = __bfloat162float(gate[d]);
        const float f = __bfloat162float(fused[d]);
        const float gf = __bfloat162float(__float2bfloat16(g * f));
        h0[d] = __float2bfloat16(__bfloat162float(h0[d]) + gf);
    }
}

// Residual add for hidden_correction: out[d] = h_row[d] + delta[d].
// Grid: 12, Block: 256.
__global__ void __launch_bounds__(256) k_dfly_hc_add(
    const __nv_bfloat16* __restrict__ h_row,
    const __nv_bfloat16* __restrict__ delta,
    __nv_bfloat16* __restrict__ out) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d < geom::HID) {
        out[d] = __float2bfloat16(__bfloat162float(h_row[d]) + __bfloat162float(delta[d]));
    }
}

// Ephemeral DFly2 draft KV: RMSNorm(kv_raw[w, 0..511]) + DSA RoPE(kv_raw[w, 512..575], anchor + w) -> kv_draft[w, 576].
// Grid: (W), Block: 256.
__global__ void __launch_bounds__(256) k_dfly_kv_norm_rope(
    const __nv_bfloat16* __restrict__ kv_raw,
    const __nv_bfloat16* __restrict__ kv_ln,
    const int* __restrict__ d_q_pos,
    __nv_bfloat16* __restrict__ kv_draft,
    int W) {
    const int w = blockIdx.x;
    if (w >= W) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* src = kv_raw + static_cast<size_t>(w) * geom::KVA;
    __nv_bfloat16* dst = kv_draft + static_cast<size_t>(w) * geom::KVA;

    const float v0 = __bfloat162float(src[tid]);
    const float v1 = __bfloat162float(src[tid + 256]);
    float ss = v0 * v0 + v1 * v1;
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::KV_LORA) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    const __nv_bfloat16 n0 = __float2bfloat16(v0 * inv);
    const __nv_bfloat16 n1 = __float2bfloat16(v1 * inv);
    dst[tid]       = __float2bfloat16(__bfloat162float(kv_ln[tid])       * __bfloat162float(n0));
    dst[tid + 256] = __float2bfloat16(__bfloat162float(kv_ln[tid + 256]) * __bfloat162float(n1));

    if (tid < 32) {
        const int pos = d_q_pos[w];
        float r0 = __bfloat162float(src[geom::KV_LORA + 2 * tid]);
        float r1 = __bfloat162float(src[geom::KV_LORA + 2 * tid + 1]);
        hy4_rope_pair(r0, r1, pos, tid, geom::ROPE_THETA_DSA);
        dst[geom::KV_LORA + 2 * tid]     = __float2bfloat16(r0);
        dst[geom::KV_LORA + 2 * tid + 1] = __float2bfloat16(r1);
    }
}

// Fast 2-stage GPU argmax over FP32 logits[V] -> dst_id[0] (used for zero-sync TP=1 greedy DFly2 chain!).
__global__ void __launch_bounds__(256) k_dfly_argmax_p1(
    const float* __restrict__ logits, float* __restrict__ pval, int* __restrict__ pid, int V) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int stride = gridDim.x * 256;
    float bv = -INFINITY;
    int bi = 0;
    for (int i = bid * 256 + tid; i < V; i += stride) {
        const float v = logits[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi   = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        float mv = sv[0]; int mi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > mv || (sv[w] == mv && si[w] < mi)) { mv = sv[w]; mi = si[w]; }
        }
        pval[bid] = mv;
        pid[bid]  = mi;
    }
}

__global__ void __launch_bounds__(256) k_dfly_argmax_p2(
    const float* __restrict__ pval, const int* __restrict__ pid, int* __restrict__ dst_id, int nb) {
    const int tid = threadIdx.x;
    float bv = (tid < nb) ? pval[tid] : -INFINITY;
    int bi   = (tid < nb) ? pid[tid]  : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi   = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        float mv = sv[0]; int mi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > mv || (sv[w] == mv && si[w] < mi)) { mv = sv[w]; mi = si[w]; }
        }
        dst_id[0] = mi;
    }
}

// Stage-2 argmax producing a 16B (max_val, global_max_id) pair for zero-sync TP=2 GPU RDMA merge!
__global__ void __launch_bounds__(256) k_dfly_argmax_p2_pair(
    const float* __restrict__ pval, const int* __restrict__ pid,
    int v_offset, DflyArgPair* __restrict__ out_pair, int nb) {
    const int tid = threadIdx.x;
    float bv = (tid < nb) ? pval[tid] : -INFINITY;
    int bi   = (tid < nb) ? pid[tid]  : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi   = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        float mv = sv[0]; int mi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > mv || (sv[w] == mv && si[w] < mi)) { mv = sv[w]; mi = si[w]; }
        }
        out_pair->val  = mv;
        out_pair->id   = mi + v_offset;
        out_pair->pad0 = 0;
        out_pair->pad1 = 0;
    }
}

// =================================================================================================
// Tensor-Core DFly2 Absorbed MLA Kernel (mma.m16n8k16):
//   Attends causally over [target DSA history (< anchor)] + [kv_draft[0..q_off]]!
//   Target DSA history is read directly from target layer's BF16 / FP8 / INT4 KV pool with ZERO copy!
//   Grid: (W * n_hg * NS), Block: 256 (8 warps), Shared: DFLY_MLA_SMEM (58,560 B).
// =================================================================================================
template <KvFmt FMT>
__global__ void __launch_bounds__(256) k_dfly_mla_tc(
    const __nv_bfloat16* __restrict__ q_full,     // [W, nh, 576]
    const float* __restrict__ sinks,              // [nh] FP32
    const uint8_t* __restrict__ target_kv_pool,   // target DSA layer's KV pool (BF16 / FP8 / INT4)
    const int* __restrict__ d_page_table,         // [slots, max_pps]
    const int* __restrict__ snap_grp,             // [511] (or nullptr if dense)
    const int* __restrict__ snap_cnt_ptr,         // [1] (< 0 means dense 0..source_pos)
    const int* __restrict__ snap_pos_ptr,         // [1] (= source_pos)
    const __nv_bfloat16* __restrict__ kv_draft,   // [W, 576] BF16
    __nv_bfloat16* __restrict__ u_h,              // [nh, W, 512] (NS == 1)
    float* __restrict__ split_m,                  // [W, NS, nh] (NS > 1)
    float* __restrict__ split_l,
    float* __restrict__ split_o,                  // [W, NS, nh, 512]
    int slot, int W, int nh, int max_pps, int NS,
    const DflyGraphMeta* __restrict__ d_meta = nullptr) {
    const int n_hg = nh >> 4; // 32/nr / 16 -> 2 at TP=1, 1 at TP=2
    const int split_id = blockIdx.x % NS;
    const int rem = blockIdx.x / NS;
    const int hg = rem % n_hg;
    const int q_off = rem / n_hg;
    if (q_off >= W) return;

    if (d_meta != nullptr) {
        const int s_idx = d_meta->snap_row;
        slot         = d_meta->slot;
        snap_grp     = snap_grp + static_cast<size_t>(s_idx) * (geom::IDX_GROUP_TOPK - 1);
        snap_cnt_ptr = snap_cnt_ptr + s_idx;
        snap_pos_ptr = snap_pos_ptr + s_idx;
    }

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg   = lane >> 2; // 0..7
    const int tc   = lane & 3;  // 0..3
    const int h0   = hg * 16;

    const int source_pos = snap_pos_ptr[0];
    const int scnt = snap_cnt_ptr[0];
    const bool sparse = (scnt >= 0);
    const int tail_start = (source_pos >> 2) << 2;
    const int tail_len   = (source_pos >= 0) ? (source_pos - tail_start + 1) : 0;
    const int hist_len   = (source_pos < 0) ? 0 : (sparse ? (scnt * 4 + tail_len) : (source_pos + 1));
    const int total_kv   = hist_len + (q_off + 1);

    const int k_per_split = (total_kv + NS - 1) / NS;
    const int k_start = split_id * k_per_split;
    const int k_end   = min(total_kv, k_start + k_per_split);

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;                           // [16, 576]
    __nv_bfloat16* sKV = smem + 16 * 576;                // [32, 576]
    __nv_bfloat16* sP  = smem + (16 + 32) * 576;         // [16, 32]
    float* sS          = reinterpret_cast<float*>(sP + 16 * 32); // [16, 32]
    float* sM          = sS + 16 * 32;                   // [16]
    float* sL          = sM + 16;                        // [16]
    float* sAlpha      = sL + 16;                        // [16]

    const __nv_bfloat16* qm = q_full + (static_cast<size_t>(q_off) * nh + h0) * geom::KVA;
    for (int i = tid; i < 16 * 576 / 8; i += 256) {
        reinterpret_cast<uint4*>(sQ)[i] = reinterpret_cast<const uint4*>(qm)[i];
    }
    if (tid < 16) {
        sM[tid] = (split_id == 0) ? sinks[h0 + tid] : -INFINITY;
        sL[tid] = (split_id == 0) ? 1.0f : 0.0f;
    }
    __syncthreads();

    float acc_o[8][4];
    #pragma unroll
    for (int d = 0; d < 8; ++d) acc_o[d][0] = acc_o[d][1] = acc_o[d][2] = acc_o[d][3] = 0.0f;

    const int* ptab = d_page_table + static_cast<size_t>(slot) * max_pps;
    constexpr float SCALE = 0.0625f; // 1 / sqrt(256)

    for (int kb = k_start; kb < k_end; kb += 32) {
        const int nk = min(32, k_end - kb);
        // Load nk KV tokens into sKV[0..nk-1][0..575] (history from target_kv_pool, draft from kv_draft!)
        for (int tk = warp; tk < nk; tk += 8) {
            const int ki = kb + tk;
            __nv_bfloat16* dst = sKV + static_cast<size_t>(tk) * 576;
            if (ki >= hist_len) {
                const __nv_bfloat16* src = kv_draft + static_cast<size_t>(ki - hist_len) * 576;
                #pragma unroll
                for (int c = lane; c < 576 / 8; c += 32) {
                    reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(src)[c];
                }
            } else {
                int pos = ki;
                if (sparse) {
                    if (ki < scnt * 4) pos = snap_grp[ki >> 2] * 4 + (ki & 3);
                    else               pos = tail_start + (ki - scnt * 4);
                }
                const int ppage = ptab[pos >> 6];
                const size_t tidx = static_cast<size_t>(ppage) * geom::KV_PAGE + (pos & 63);
                if constexpr (FMT == KvFmt::BF16) {
                    const uint4* src = reinterpret_cast<const uint4*>(target_kv_pool + tidx * (576 * 2));
                    #pragma unroll
                    for (int c = lane; c < 576 / 8; c += 32) {
                        reinterpret_cast<uint4*>(dst)[c] = src[c];
                    }
                } else if constexpr (FMT == KvFmt::FP8) {
                    const uint8_t* base = target_kv_pool + tidx * KV_FP8_STRIDE;
                    const __half* scs = reinterpret_cast<const __half*>(base + 512);
                    const uint4 qv = reinterpret_cast<const uint4*>(base)[lane];
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
                    *reinterpret_cast<uint4*>(dst + lane * 16)     = o0;
                    *reinterpret_cast<uint4*>(dst + lane * 16 + 8) = o1;
                    *reinterpret_cast<uint32_t*>(dst + 512 + lane * 2) =
                        reinterpret_cast<const uint32_t*>(base + 544)[lane];
                } else {
                    const uint8_t* base = target_kv_pool + tidx * KV_INT4_STRIDE;
                    const __half* scs = reinterpret_cast<const __half*>(base + 256);
                    const uint2 raw = reinterpret_cast<const uint2*>(base)[lane];
                    const float sc = __half2float(scs[lane >> 1]);
                    auto unpack_u32 = [&](uint32_t w, uint4& od) {
                        uint32_t o[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t b = (w >> (8 * i)) & 0xffu;
                            o[i] = pack_bf2(static_cast<float>(static_cast<int>(b & 0xfu) - 8) * sc,
                                            static_cast<float>(static_cast<int>(b >> 4) - 8) * sc);
                        }
                        od = make_uint4(o[0], o[1], o[2], o[3]);
                    };
                    uint4 o0, o1;
                    unpack_u32(raw.x, o0);
                    unpack_u32(raw.y, o1);
                    *reinterpret_cast<uint4*>(dst + lane * 16)     = o0;
                    *reinterpret_cast<uint4*>(dst + lane * 16 + 8) = o1;
                    *reinterpret_cast<uint32_t*>(dst + 512 + lane * 2) =
                        reinterpret_cast<const uint32_t*>(base + 288)[lane];
                }
            }
        }
        __syncthreads();

        // Phase 1: QK^T via mma.m16n8k16 (warps 0..3 own 8 KV tokens each -> 32 KV tokens!)
        if (warp < 4) {
            const int nbase = warp * 8;
            float sc_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < 36; ++ks) {
                const int k0 = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sQ + rg * 576       + k0 + 2 * tc);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sQ + (rg + 8) * 576 + k0 + 2 * tc);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sQ + rg * 576       + k0 + 2 * tc + 8);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sQ + (rg + 8) * 576 + k0 + 2 * tc + 8);
                const int kn = nbase + rg;
                const uint32_t b0 = (kn < nk) ? *reinterpret_cast<const uint32_t*>(sKV + kn * 576 + k0 + 2 * tc)     : 0u;
                const uint32_t b1 = (kn < nk) ? *reinterpret_cast<const uint32_t*>(sKV + kn * 576 + k0 + 2 * tc + 8) : 0u;
                mma16816_bf16(sc_d, a0, a1, a2, a3, b0, b1);
            }
            const int c0 = nbase + 2 * tc;
            const int c1 = c0 + 1;
            sS[rg * 32       + c0] = (c0 < nk) ? (sc_d[0] * SCALE) : -INFINITY;
            sS[rg * 32       + c1] = (c1 < nk) ? (sc_d[1] * SCALE) : -INFINITY;
            sS[(rg + 8) * 32 + c0] = (c0 < nk) ? (sc_d[2] * SCALE) : -INFINITY;
            sS[(rg + 8) * 32 + c1] = (c1 < nk) ? (sc_d[3] * SCALE) : -INFINITY;
        }
        __syncthreads();

        // Phase 2: Online softmax over 16 heads (2 heads per warp!)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int hi = warp * 2 + hh;
            const float sv = sS[hi * 32 + lane];
            const float tmax = warp_max(sv);
            const float m_old = sM[hi];
            const float m_new = fmaxf(m_old, tmax);
            const float alpha = (m_old > -1e30f) ? __expf(m_old - m_new) : 0.0f;
            const float pv = (sv > -1e30f) ? __expf(sv - m_new) : 0.0f;
            const float psum = warp_sum(pv);
            sP[hi * 32 + lane] = __float2bfloat16(pv);
            if (lane == 0) {
                sM[hi] = m_new;
                sL[hi] = sL[hi] * alpha + psum;
                sAlpha[hi] = alpha;
            }
        }
        __syncthreads();

        // Rescale running O accumulators
        const float a0_s = sAlpha[rg];
        const float a8_s = sAlpha[rg + 8];
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            acc_o[d][0] *= a0_s; acc_o[d][1] *= a0_s;
            acc_o[d][2] *= a8_s; acc_o[d][3] *= a8_s;
        }

        // Phase 3: P[16, 32] @ V[32, 512] -> O[16, 512] (8 warps x 64 cols = 512 cols!)
        const int nbase_w = warp * 64;
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int k0 = ks * 16;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sP + rg * 32       + k0 + 2 * tc);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sP + (rg + 8) * 32 + k0 + 2 * tc);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sP + rg * 32       + k0 + 2 * tc + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sP + (rg + 8) * 32 + k0 + 2 * tc + 8);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                const int col = nbase_w + d * 8 + rg;
                const int r0 = k0 + 2 * tc;
                const int r2 = k0 + 2 * tc + 8;
                const uint32_t b0 = pack_bf2(
                    (r0 < nk)     ? __bfloat162float(sKV[r0 * 576 + col])       : 0.0f,
                    (r0 + 1 < nk) ? __bfloat162float(sKV[(r0 + 1) * 576 + col]) : 0.0f);
                const uint32_t b1 = pack_bf2(
                    (r2 < nk)     ? __bfloat162float(sKV[r2 * 576 + col])       : 0.0f,
                    (r2 + 1 < nk) ? __bfloat162float(sKV[(r2 + 1) * 576 + col]) : 0.0f);
                mma16816_bf16(acc_o[d], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    // Write final O (normalized if NS==1, else raw into split buffers)
    const int nbase_w = warp * 64;
    if (NS == 1) {
        const float inv0 = (sL[rg]     > 0.0f) ? (1.0f / sL[rg])     : 0.0f;
        const float inv8 = (sL[rg + 8] > 0.0f) ? (1.0f / sL[rg + 8]) : 0.0f;
        __nv_bfloat16* o0 = u_h + (static_cast<size_t>(h0 + rg)     * W + q_off) * 512;
        __nv_bfloat16* o8 = u_h + (static_cast<size_t>(h0 + rg + 8) * W + q_off) * 512;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const int c0 = nbase_w + d * 8 + 2 * tc;
            o0[c0]     = __float2bfloat16(acc_o[d][0] * inv0);
            o0[c0 + 1] = __float2bfloat16(acc_o[d][1] * inv0);
            o8[c0]     = __float2bfloat16(acc_o[d][2] * inv8);
            o8[c0 + 1] = __float2bfloat16(acc_o[d][3] * inv8);
        }
    } else {
        const size_t base_s = (static_cast<size_t>(q_off) * NS + split_id) * nh;
        if (tid < 16) {
            split_m[base_s + h0 + tid] = sM[tid];
            split_l[base_s + h0 + tid] = sL[tid];
        }
        float* o0 = split_o + (base_s + h0 + rg)     * 512;
        float* o8 = split_o + (base_s + h0 + rg + 8) * 512;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const int c0 = nbase_w + d * 8 + 2 * tc;
            o0[c0]     = acc_o[d][0];
            o0[c0 + 1] = acc_o[d][1];
            o8[c0]     = acc_o[d][2];
            o8[c0 + 1] = acc_o[d][3];
        }
    }
}

constexpr int DFLY_MLA_SMEM = (16 + 32) * 576 * 2 + 16 * 32 * 2 + 16 * 32 * 4 + 48 * 4; // 58,560 B

inline void ensure_dfly_mla_smem() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_dfly_mla_tc<KvFmt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, DFLY_MLA_SMEM);
    cudaFuncSetAttribute(k_dfly_mla_tc<KvFmt::FP8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, DFLY_MLA_SMEM);
    cudaFuncSetAttribute(k_dfly_mla_tc<KvFmt::INT4>, cudaFuncAttributeMaxDynamicSharedMemorySize, DFLY_MLA_SMEM);
}

// =================================================================================================
// Full DFly2 Forward for 1 request slot (W = K + 1 tokens, default K = 5 -> W = 6):
//   Runs slot-0 mtp_fusion -> 5 DFly MLA+SwiGLU layers (reusing target DSA layers [25,29,33,37,41])
//   -> final_norm -> H_draft [W, 3072] in sc.h!
//   Then caller runs K sequential hidden_correction + lm_head steps!
// =================================================================================================
inline void forward_body(
    cublasHandle_t blas,
    const std::vector<RankWeights>& rw,   // [nloc]
    Scratch& sc,
    AttnScratch& attn_sc,
    const DsaLayerKv* const* target_kv,   // [5 * nloc], indexed [d * nloc + i]
    const int* const* snap_grp_ptrs,      // [5], each points to [511] for this slot/source_row
    const int* const* snap_cnt_ptrs,      // [5], each points to [1]
    const int* const* snap_pos_ptrs,      // [5], each points to [1]
    const __nv_bfloat16* embed,
    const __nv_bfloat16* target_hidden,   // [3072] (mixed_ln at source_row!)
    __nv_bfloat16* parts,                 // [nr, max_rows, 3072]
    const int* d_page_table,
    uint8_t* x8_scratch,
    int nr, int rank0, int nloc,
    int slot, int bonus_tok, int anchor, int W, int max_pps,
    cudaStream_t st,
    const std::function<void(int c, int M, const __nv_bfloat16* post_ln)>& on_ar_add = nullptr,
    int rope_delta = 0,
    const DflyGraphMeta* d_meta = nullptr) {

    ensure_dfly_mla_smem();
    const RankWeights& w0 = rw[0];
    // 1. Embed [bonus_tok, MASK, ..., MASK] + positions [anchor + rope_delta .. anchor + W - 1 + rope_delta]
    k_dfly_init_block<<<W * 12, 256, 0, st>>>(
        embed, sc.h, sc.d_q_pos, bonus_tok, anchor, rope_delta, W, d_meta, d_meta ? sc.d_draft_id : nullptr);

    // 2. Slot-0 mtp_fusion: h[0] += gate * eh_proj(cat(enorm(h[0]), hnorm(target_hidden)))
    k_dfly_cat_norms<<<2, 256, 0, st>>>(sc.h, w0.mtp_enorm, target_hidden, w0.mtp_hnorm, sc.cat6144, d_meta);
    mx8::linear_w(blas, sc.cat6144, w0.mtp_eh_proj, sc.tmp3072, 1, geom::HID, 2 * geom::HID, x8_scratch, nullptr, st);
    k_dfly_mtp_add<<<12, 256, 0, st>>>(sc.h, w0.mtp_gate, sc.tmp3072);

    // 3. 5 DFly layers
    int coll = 0;
    const int NS = std::min(MLA_MAX_SPLITS, std::max(1, 16 / std::max(1, W)));
    for (int d = 0; d < NUM_LAYERS; ++d) {
        const LayerWeights& lw0 = rw[0].layers[d];

        // 3a. input_layernorm
        k_hy4_rmsnorm<<<W, 256, 0, st>>>(sc.h, lw0.input_norm, sc.x_ln, W, geom::HID, geom::RMS_EPS);

        // 3b. Gated Absorbed MLA reusing target DSA layer TARGET_DSA_LAYERS[d]
        for (int i = 0; i < nloc; ++i) {
            const int r = rank0 + i;
            const LayerWeights& lw = rw[size_t(i)].layers[d];
            const DsaLayerKv* tkv = target_kv[d * nloc + i];
            const int nh = lw.nh_loc;
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * W * geom::HID;

            mx8::linear_w(blas, sc.x_ln, lw.q_a_proj, attn_sc.q_a, W, geom::QA, geom::HID, x8_scratch, nullptr, st);
            k_hy4_rmsnorm<<<W, 256, 0, st>>>(attn_sc.q_a, lw.q_a_ln, attn_sc.q_resid, W, geom::QA, geom::RMS_EPS);
            mx8::linear_w(blas, attn_sc.q_resid, lw.q_b_proj, attn_sc.q_raw, W, nh * geom::QK_HEAD, geom::QA, x8_scratch, nullptr, st);
            launch_q_absorb_rope_smallm(attn_sc.q_raw, lw.w_uk_t, sc.d_q_pos, nullptr, attn_sc.q_full, W, nh, geom::ROPE_THETA_DSA, st);

            mx8::linear_w(blas, sc.x_ln, lw.kv_a_proj, attn_sc.kv_raw, W, geom::KVA, geom::HID, x8_scratch, nullptr, st);
            k_dfly_kv_norm_rope<<<W, 256, 0, st>>>(attn_sc.kv_raw, lw.kv_a_ln, sc.d_q_pos, sc.kv_draft, W);

            mx8::linear_w(blas, sc.x_ln, lw.gate_proj, attn_sc.gate, W, nh * geom::V_HEAD, geom::HID, x8_scratch, nullptr, st);

            const int n_hg = nh >> 4;
            const int grid_mla = W * n_hg * NS;
            if (tkv->fmt == KvFmt::FP8) {
                k_dfly_mla_tc<KvFmt::FP8><<<grid_mla, 256, DFLY_MLA_SMEM, st>>>(
                    attn_sc.q_full, lw.sinks, tkv->kv_pool, d_page_table,
                    snap_grp_ptrs[d], snap_cnt_ptrs[d], snap_pos_ptrs[d],
                    sc.kv_draft, attn_sc.u_h, attn_sc.split_m, attn_sc.split_l, attn_sc.split_o,
                    slot, W, nh, max_pps, NS, d_meta);
            } else if (tkv->fmt == KvFmt::INT4) {
                k_dfly_mla_tc<KvFmt::INT4><<<grid_mla, 256, DFLY_MLA_SMEM, st>>>(
                    attn_sc.q_full, lw.sinks, tkv->kv_pool, d_page_table,
                    snap_grp_ptrs[d], snap_cnt_ptrs[d], snap_pos_ptrs[d],
                    sc.kv_draft, attn_sc.u_h, attn_sc.split_m, attn_sc.split_l, attn_sc.split_o,
                    slot, W, nh, max_pps, NS, d_meta);
            } else {
                k_dfly_mla_tc<KvFmt::BF16><<<grid_mla, 256, DFLY_MLA_SMEM, st>>>(
                    attn_sc.q_full, lw.sinks, tkv->kv_pool, d_page_table,
                    snap_grp_ptrs[d], snap_cnt_ptrs[d], snap_pos_ptrs[d],
                    sc.kv_draft, attn_sc.u_h, attn_sc.split_m, attn_sc.split_l, attn_sc.split_o,
                    slot, W, nh, max_pps, NS, d_meta);
            }
            if (NS > 1) {
                k_hy4_mla_split_reduce<<<W * nh, 256, 0, st>>>(attn_sc.split_m, attn_sc.split_l, attn_sc.split_o, attn_sc.u_h, W, nh, NS);
            }
            launch_v_expand_gate_smallm(attn_sc.u_h, lw.w_uv, attn_sc.gate, attn_sc.attn_out, W, nh, st);
            mx8::linear_w(blas, attn_sc.attn_out, lw.o_proj, p_dst, W, geom::HID, nh * geom::V_HEAD, x8_scratch, nullptr, st);
        }

        // 3c. Attention AllReduce + post_proj_norm + residual add
        if (on_ar_add) {
            on_ar_add(coll++, W, lw0.post_proj_ln);
        } else {
            k_hy4_sum_norm_add1<<<W, 256, 0, st>>>(parts, lw0.post_proj_ln, sc.h, nr, W, W);
        }

        // 3d. post_attention_layernorm + Dense SwiGLU MLP (NO clamp, NO post_fc2_norm!)
        k_hy4_rmsnorm<<<W, 256, 0, st>>>(sc.h, lw0.post_attn_norm, sc.x_ln, W, geom::HID, geom::RMS_EPS);
        for (int i = 0; i < nloc; ++i) {
            const int r = rank0 + i;
            const LayerWeights& lw = rw[size_t(i)].layers[d];
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * W * geom::HID;
            mx8::linear_w(blas, sc.x_ln, lw.mlp_gate, sc.mlp_gate, W, lw.i_loc, geom::HID, x8_scratch, nullptr, st);
            mx8::linear_w(blas, sc.x_ln, lw.mlp_up,   sc.mlp_up,   W, lw.i_loc, geom::HID, x8_scratch, nullptr, st);
            const size_t n = static_cast<size_t>(W) * lw.i_loc;
            k_hy4_silu_mul_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(sc.mlp_gate, sc.mlp_up, n);
            mx8::linear_w(blas, sc.mlp_gate, lw.mlp_down, p_dst, W, geom::HID, lw.i_loc, x8_scratch, nullptr, st);
        }

        // 3e. MLP AllReduce + residual add (no post norm!)
        if (on_ar_add) {
            on_ar_add(coll++, W, nullptr);
        } else {
            k_hy4_sum_norm_add1<<<W, 256, 0, st>>>(parts, nullptr, sc.h, nr, W, W);
        }
    }

    // 4. Final norm -> sc.h [W, 3072]
    k_hy4_rmsnorm<<<W, 256, 0, st>>>(sc.h, w0.final_norm, sc.h, W, geom::HID, geom::RMS_EPS);
}

// One step of hidden_correction (using explicit host prev_tok or GPU d_prev_tok_ptr):
//   out_hidden [3072] = h_step + down_proj(silu(gate_proj(cat(hidden_norm(h_step), embed_norm(embed[prev_tok])))) * up_proj(...))
inline void hidden_correction_step(
    cublasHandle_t blas,
    const RankWeights& w0,
    Scratch& sc,
    const __nv_bfloat16* h_step,
    const __nv_bfloat16* embed,
    int prev_tok,
    const int* d_prev_tok_ptr, // if non-null, reads prev_tok on GPU (zero host sync!)
    __nv_bfloat16* out_hidden,
    uint8_t* x8_scratch,
    cudaStream_t st) {
    if (d_prev_tok_ptr != nullptr) {
        k_dfly_cat_norms_embed<<<2, 256, 0, st>>>(h_step, w0.hc_hidden_norm, embed, d_prev_tok_ptr, w0.hc_embed_norm, sc.cat6144);
    } else {
        const __nv_bfloat16* emb_row = embed + static_cast<size_t>(prev_tok) * geom::HID;
        k_dfly_cat_norms<<<2, 256, 0, st>>>(h_step, w0.hc_hidden_norm, emb_row, w0.hc_embed_norm, sc.cat6144);
    }
    mx8::linear_w(blas, sc.cat6144, w0.hc_gate_proj, sc.hc_gate, 1, HC_I, 2 * geom::HID, x8_scratch, nullptr, st);
    mx8::linear_w(blas, sc.cat6144, w0.hc_up_proj,   sc.hc_up,   1, HC_I, 2 * geom::HID, x8_scratch, nullptr, st);
    k_hy4_silu_mul_bf16<<<(HC_I + 255) / 256, 256, 0, st>>>(sc.hc_gate, sc.hc_up, HC_I);
    mx8::linear_w(blas, sc.hc_gate, w0.hc_down_proj, sc.tmp3072, 1, geom::HID, HC_I, x8_scratch, nullptr, st);
    k_dfly_hc_add<<<12, 256, 0, st>>>(h_step, sc.tmp3072, out_hidden);
}

} // namespace dfly
} // namespace hy4
