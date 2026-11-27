// Drama 3 Codebook-MTP tower forward pass in CUDA + cuBLAS FP16 (reference path).
//
// Exact match to fish-speech @ 0f8a12e8 (`fish_omni/tower.py`):
//   - 4 shared layers (`audio_decoder_mtp_shared.layers.0..3`) + 9 residual layers (`audio_decoder_mtp.layers.0..8`)
//     + 9 codebook heads (`audio_decoder_mtp.heads.0..8` [4096, 2048], residual sampling uses rows 0..1023).
//   - Layer: e = semantic ? embed(clamp(code, 0, 4095)) : 0; shared layer 0 adds summed codebook_embeddings for c2..c9
//     (masked to 0 if current input row is not semantic); mixed = eh_proj(cat(enorm(e), hnorm(hidden))) ([e | h]);
//     TowerTransformerLayer (FP32 residual, wqkv=[Q 4096 | Gate 4096 | K 512 | V 512], ZeroCenteredRMSNorm q/k,
//     interleaved-pair RoPE on first 64 dims, GQA * sigmoid(Gate), wo, SwiGLU with FP32-output w2); final_layernorm.
//   - Every layer maintains its own per-slot KV cache over all rows (prompt included).
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tts_backbone.cuh"
#include "tts_config.h"
#include "tts_weights.h"

namespace g6tts {

// Computes summed embeddings of unseen codes `first_cb..end_cb-1` (for Drama 3: c2..c9) for each row `frames[M, 11]`,
// masked to zero when `!is_semantic(frames[m, 0])`.
// Grid: (M), Block: 256.
__global__ void k_tower_unseen_sum(const int32_t* __restrict__ frames,
                                   const __half* __restrict__ cb_emb,
                                   float* __restrict__ out_f32,
                                   int M, int H, int row_w, int sem0, int sem1,
                                   int first_cb, int end_cb, int cb_vocab) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int32_t* f = frames + size_t(m) * row_w;
    const bool is_sem = (f[0] >= sem0 && f[0] <= sem1);
    float* dst = out_f32 + size_t(m) * H;
    for (int d = threadIdx.x; d < H; d += 256) {
        float acc = 0.0f;
        if (is_sem) {
            for (int cb = first_cb; cb < end_cb; ++cb) {
                const int code = max(0, min(cb_vocab - 1, f[1 + cb]));
                acc += __half2float(cb_emb[(size_t(cb) * cb_vocab + code) * H + d]);
            }
        }
        dst[d] = acc;
    }
}

// Fused TowerLayer entry:
//   e = semantic[m] ? emb_table[clamp(codes[m], 0, vocab-1)] : 0  (+ shared_unseen[m] if non-null)
//   cat_out[m, 0..H-1]   = fp16(ZeroCenteredRMSNorm(e, enorm_w))
//   cat_out[m, H..2H-1]  = fp16(ZeroCenteredRMSNorm(hidden[m], hnorm_w))
// Grid: (M), Block: 256.
__global__ void k_tower_eh_cat_norm(const __half* __restrict__ hidden,
                                    const int32_t* __restrict__ codes,
                                    const uint8_t* __restrict__ semantic,
                                    const __half* __restrict__ emb_table,
                                    const float* __restrict__ shared_unseen,
                                    const float* __restrict__ enorm_w,
                                    const float* __restrict__ hnorm_w,
                                    __half* __restrict__ cat_out,
                                    int M, int H, int vocab, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;

    const bool is_sem = (semantic[m] != 0);
    const int code = max(0, min(vocab - 1, codes[m]));
    const __half* e_row = emb_table + size_t(code) * H;
    const __half* h_row = hidden + size_t(m) * H;
    const float* u_row = shared_unseen ? (shared_unseen + size_t(m) * H) : nullptr;
    __half* dst = cat_out + size_t(m) * (2 * H);

    // 1. enorm(e)
    float ss_e = 0.0f;
    for (int d = threadIdx.x; d < H; d += 256) {
        float ev = is_sem ? __half2float(e_row[d]) : 0.0f;
        if (u_row) ev += u_row[d];
        ss_e += ev * ev;
    }
    const float rs_e = rsqrtf(tts_block_sum256(ss_e, red) / float(H) + eps);
    for (int d = threadIdx.x; d < H; d += 256) {
        float ev = is_sem ? __half2float(e_row[d]) : 0.0f;
        if (u_row) ev += u_row[d];
        const float y = ev * rs_e * (1.0f + enorm_w[d]);
        dst[d] = __float2half_rn(fminf(65504.0f, fmaxf(-65504.0f, y)));
    }

    // 2. hnorm(hidden)
    float ss_h = 0.0f;
    for (int d = threadIdx.x; d < H; d += 256) {
        const float hv = __half2float(h_row[d]);
        ss_h += hv * hv;
    }
    const float rs_h = rsqrtf(tts_block_sum256(ss_h, red) / float(H) + eps);
    for (int d = threadIdx.x; d < H; d += 256) {
        const float hv = __half2float(h_row[d]);
        const float y = hv * rs_h * (1.0f + hnorm_w[d]);
        dst[H + d] = __float2half_rn(fminf(65504.0f, fmaxf(-65504.0f, y)));
    }
}

// TowerAttention Q/K ZeroCenteredRMSNorm + INTERLEAVED-PAIR partial RoPE + KV cache append.
// wqkv_out: [M, 9216] contiguous [Q(4096) | Gate(4096) | K(512) | V(512)].
// Grid: (M, nq + nkv), Block: 256 (1 thread per head_dim element for hd=256).
__global__ void k_tower_qk_norm_rope_kv(const __half* __restrict__ wqkv_out,
                                        const float* __restrict__ q_norm_w,
                                        const float* __restrict__ k_norm_w,
                                        __half* __restrict__ q_out,
                                        __half* __restrict__ k_cache,
                                        __half* __restrict__ v_cache,
                                        const int* __restrict__ d_slots,
                                        const int* __restrict__ d_pos,
                                        int M, int nq, int nkv, int hd, int rotary_dim,
                                        float eps, float rope_base, int cap) {
    __shared__ float red[8];
    __shared__ float s_buf[256];
    const int m = blockIdx.x;
    const int h_glob = blockIdx.y;
    const int d = threadIdx.x;
    if (m >= M || h_glob >= nq + nkv || d >= hd) return;

    const int q_width = nq * hd;    // 4096
    const int kv_width = nkv * hd;  // 512
    const int total_w = 2 * q_width + 2 * kv_width; // 9216

    const bool is_k = (h_glob >= nq);
    const int h = is_k ? (h_glob - nq) : h_glob;
    const int slot = d_slots[m];
    const int pos = d_pos[m];

    const __half* row = wqkv_out + size_t(m) * total_w;
    const __half* src = is_k
        ? (row + 2 * q_width + size_t(h) * hd)
        : (row + size_t(h) * hd);
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float xv = __half2float(src[d]);
    const float rs = rsqrtf(tts_block_sum256(xv * xv, red) / float(hd) + eps);
    const float normed = xv * rs * (1.0f + nw[d]);
    s_buf[d] = normed;
    __syncthreads();

    float out_v = normed;
    if (d < rotary_dim) {
        // Interleaved pairs (2*j, 2*j + 1) matching `tower.py:180-190`
        const int j = d >> 1;
        const float inv_freq = 1.0f / powf(rope_base, float(2 * j) / float(rotary_dim));
        float sn, cs;
        sincosf(float(pos) * inv_freq, &sn, &cs);
        const float even = s_buf[2 * j];
        const float odd = s_buf[2 * j + 1];
        out_v = ((d & 1) == 0) ? (even * cs - odd * sn) : (odd * cs + even * sn);
    }

    if (is_k) {
        const size_t kv_off = ((size_t(slot) * cap + pos) * nkv + h) * hd + d;
        k_cache[kv_off] = __float2half_rn(out_v);
        v_cache[kv_off] = row[2 * q_width + kv_width + size_t(h) * hd + d];
    } else {
        q_out[(size_t(m) * nq + h) * hd + d] = __float2half_rn(out_v);
    }
}

// ------------------------------------------------------------------------------------------------ Tower runner

struct TowerProfile {
    bool enabled = false;
    double shared_ms = 0.0;
    double res_ms = 0.0;
    double head_ms = 0.0;
    void reset() { shared_ms = res_ms = head_ms = 0.0; }
};

struct TowerRunner {
    const TtsWeights* W = nullptr;
    cublasHandle_t blas = nullptr;
    int max_slots = 8;
    int cap = 4096;
    int max_chunk = 128;

    // 13 per-layer KV caches: [0..3] shared, [4..12] residual; each [max_slots, cap, 2, 256] FP16
    std::vector<__half*> k_caches;
    std::vector<__half*> v_caches;

    // Scratch buffers
    float* d_unseen = nullptr;      // [max_chunk, H]
    __half* d_hid_a = nullptr;      // [max_chunk, H]
    __half* d_hid_b = nullptr;      // [max_chunk, H]
    __half* d_eh_cat = nullptr;     // [max_chunk, 2*H]
    float* d_res_f32 = nullptr;     // [max_chunk, H]
    __half* d_norm = nullptr;       // [max_chunk, H]
    __half* d_wqkv = nullptr;       // [max_chunk, 9216]
    __half* d_q_ready = nullptr;    // [max_chunk, 4096]
    __half* d_attn_out = nullptr;   // [max_chunk, 4096]
    __half* d_ffn_1 = nullptr;      // [max_chunk, 6144]
    __half* d_ffn_3 = nullptr;      // [max_chunk, 6144]
    float* d_head_logits = nullptr; // [max_chunk, 4096] (or [max_chunk, 1024] slice)
    int32_t* d_codes = nullptr;     // [tower_res, max_chunk]
    uint8_t* d_sem = nullptr;       // [max_chunk]
    int32_t* d_frames = nullptr;    // [max_chunk, 11]
    int32_t* d_slots = nullptr;     // [max_chunk]
    int32_t* d_pos = nullptr;       // [max_chunk]

    // Persistent pinned host staging for async prefill uploads & optional head logit readback
    int32_t* h_slots = nullptr;     // [max_chunk]
    int32_t* h_pos = nullptr;       // [max_chunk]
    uint8_t* h_sem = nullptr;       // [max_chunk]
    int32_t* h_codes = nullptr;     // [tower_res, max_chunk]
    int32_t* h_frames = nullptr;    // [max_chunk, 11]
    float* h_head_tmp = nullptr;    // [max_chunk, RES_CB_SIZE]

    size_t state_bytes = 0;
    size_t scratch_bytes = 0;
    TowerProfile prof;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev_upload = nullptr;
    bool upload_pending = false;

    ~TowerRunner() { release(); }

    void release() {
        if (upload_pending && ev_upload) {
            cudaEventSynchronize(ev_upload);
            upload_pending = false;
        }
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto freeh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        for (auto*& p : k_caches) freep(p);
        for (auto*& p : v_caches) freep(p);
        k_caches.clear(); v_caches.clear();
        freep(d_unseen); freep(d_hid_a); freep(d_hid_b); freep(d_eh_cat);
        freep(d_res_f32); freep(d_norm); freep(d_wqkv); freep(d_q_ready);
        freep(d_attn_out); freep(d_ffn_1); freep(d_ffn_3); freep(d_head_logits);
        freep(d_codes); freep(d_sem); freep(d_frames); freep(d_slots); freep(d_pos);
        freeh(h_slots); freeh(h_pos); freeh(h_sem); freeh(h_codes); freeh(h_frames); freeh(h_head_tmp);
        if (ev0) { cudaEventDestroy(ev0); ev0 = nullptr; }
        if (ev1) { cudaEventDestroy(ev1); ev1 = nullptr; }
        if (ev_upload) { cudaEventDestroy(ev_upload); ev_upload = nullptr; }
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
        TTS_CK(cudaEventCreate(&ev_upload));
        upload_pending = false;

        const auto& c = W->cfg;
        const int n_tot = c.tower_shared + c.tower_res;
        k_caches.assign(size_t(n_tot), nullptr);
        v_caches.assign(size_t(n_tot), nullptr);
        state_bytes = 0;
        const size_t kv_b = size_t(max_slots) * cap * c.tower_kv_heads * c.tower_head_dim * sizeof(__half);
        for (int i = 0; i < n_tot; ++i) {
            TTS_CK(cudaMalloc(&k_caches[size_t(i)], kv_b));
            TTS_CK(cudaMalloc(&v_caches[size_t(i)], kv_b));
            TTS_CK(cudaMemset(k_caches[size_t(i)], 0, kv_b));
            TTS_CK(cudaMemset(v_caches[size_t(i)], 0, kv_b));
            state_bytes += 2 * kv_b;
        }

        const int H = c.hidden_size;
        const int qw = c.tower_heads * c.tower_head_dim;
        const int kvw = c.tower_kv_heads * c.tower_head_dim;
        auto alloc_d = [&](auto*& p, size_t n) {
            const size_t nb = n * sizeof(*p);
            TTS_CK(cudaMalloc(&p, nb));
            scratch_bytes += nb;
        };
        alloc_d(d_unseen, size_t(max_chunk) * H);
        alloc_d(d_hid_a, size_t(max_chunk) * H);
        alloc_d(d_hid_b, size_t(max_chunk) * H);
        alloc_d(d_eh_cat, size_t(max_chunk) * 2 * H);
        alloc_d(d_res_f32, size_t(max_chunk) * H);
        alloc_d(d_norm, size_t(max_chunk) * H);
        alloc_d(d_wqkv, size_t(max_chunk) * (2 * qw + 2 * kvw));
        alloc_d(d_q_ready, size_t(max_chunk) * qw);
        alloc_d(d_attn_out, size_t(max_chunk) * qw);
        alloc_d(d_ffn_1, size_t(max_chunk) * c.tower_intermediate);
        alloc_d(d_ffn_3, size_t(max_chunk) * c.tower_intermediate);
        alloc_d(d_head_logits, size_t(max_chunk) * c.tower_vocab);
        alloc_d(d_codes, size_t(c.tower_res) * max_chunk);
        alloc_d(d_sem, size_t(max_chunk));
        alloc_d(d_frames, size_t(max_chunk) * c.row_width());
        alloc_d(d_slots, size_t(max_chunk));
        alloc_d(d_pos, size_t(max_chunk));

        TTS_CK(cudaHostAlloc(&h_slots, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_pos, size_t(max_chunk) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_sem, size_t(max_chunk) * sizeof(uint8_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_codes, size_t(c.tower_res) * max_chunk * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_frames, size_t(max_chunk) * c.row_width() * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_head_tmp, size_t(max_chunk) * geom::RES_CB_SIZE * sizeof(float), cudaHostAllocDefault));
    }

    // Runs one TowerLayer (shared or residual) on `M` rows.
    void run_layer(const TowerLayerW& TW, int kv_idx,
                   const __half* hidden_in, __half* hidden_out,
                   const int32_t* dev_codes, const uint8_t* dev_sem,
                   const float* dev_unseen,
                   const int32_t* dev_slots, const int32_t* dev_pos,
                   int M, cudaStream_t stream) {
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int qw = c.tower_heads * c.tower_head_dim;
        const int kvw = c.tower_kv_heads * c.tower_head_dim;
        const int wqkv_dim = 2 * qw + 2 * kvw;

        // 1. enorm(e) || hnorm(hidden) -> eh_proj into FP32 residual
        k_tower_eh_cat_norm<<<M, 256, 0, stream>>>(
            hidden_in, dev_codes, dev_sem, TW.embedding, dev_unseen,
            TW.enorm, TW.hnorm, d_eh_cat, M, H, c.tower_vocab, c.tower_norm_eps);
        gemm_f16(blas, d_eh_cat, TW.eh_proj, d_res_f32, M, H, 2 * H, true, 0.0f);

        // 2. Attention
        k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_res_f32, TW.attn_norm, d_norm, M, H, c.tower_norm_eps);
        gemm_f16(blas, d_norm, TW.wqkv, d_wqkv, M, wqkv_dim, H, false, 0.0f);

        dim3 qk_grid(M, c.tower_heads + c.tower_kv_heads);
        k_tower_qk_norm_rope_kv<<<qk_grid, 256, 0, stream>>>(
            d_wqkv, TW.q_norm, TW.k_norm, d_q_ready,
            k_caches[size_t(kv_idx)], v_caches[size_t(kv_idx)],
            dev_slots, dev_pos, M, c.tower_heads, c.tower_kv_heads, c.tower_head_dim,
            c.tower_rotary_dim, c.tower_norm_eps, c.tower_rope_base, cap);

        dim3 attn_grid(M, c.tower_heads);
        k_gqa_attn_gated<<<attn_grid, 256, 0, stream>>>(
            d_q_ready, d_wqkv, k_caches[size_t(kv_idx)], v_caches[size_t(kv_idx)], d_attn_out,
            dev_slots, dev_pos, M, c.tower_heads, c.tower_kv_heads, c.tower_head_dim, cap,
            false, wqkv_dim, qw);

        // wo accumulates into FP32 residual
        gemm_f16(blas, d_attn_out, TW.wo, d_res_f32, M, H, qw, true, 1.0f);

        // 3. SwiGLU FFN with FP32-output w2 (weights exceed FP16 max)
        k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_res_f32, TW.ffn_norm, d_norm, M, H, c.tower_norm_eps);
        gemm_f16(blas, d_norm, TW.w1, d_ffn_1, M, c.tower_intermediate, H, false, 0.0f);
        gemm_f16(blas, d_norm, TW.w3, d_ffn_3, M, c.tower_intermediate, H, false, 0.0f);
        const size_t fn = size_t(M) * c.tower_intermediate;
        k_silu_mul_f16<<<int((fn + 255) / 256), 256, 0, stream>>>(d_ffn_1, d_ffn_3, d_ffn_1, fn);
        gemm_f16(blas, d_ffn_1, TW.w2, d_res_f32, M, H, c.tower_intermediate, true, 1.0f);

        // 4. final_layernorm -> FP16 hidden_out
        k_zc_rmsnorm_f32_to_f16<<<M, 256, 0, stream>>>(d_res_f32, TW.final_ln, hidden_out, M, H, c.tower_norm_eps);
    }

    // Runs all 4 shared layers; result lands in `d_hid_a` (returns pointer to active hidden buffer).
    __half* run_shared(const __half* bb_hidden,
                       const int32_t* dev_c0, const uint8_t* dev_sem,
                       const int32_t* dev_frames,
                       const int32_t* dev_slots, const int32_t* dev_pos,
                       int M, cudaStream_t stream) {
        TTS_BLAS_CK(cublasSetStream(blas, stream));
        const auto& c = W->cfg;
        k_tower_unseen_sum<<<M, 256, 0, stream>>>(
            dev_frames, W->codebook_embeddings, d_unseen,
            M, c.hidden_size, c.row_width(), c.semantic_start_id, c.semantic_end_id,
            1 + c.vq_backbone_codebooks, c.num_codebooks, c.tower_vocab);

        const __half* cur = bb_hidden;
        __half* dst = d_hid_a;
        for (int i = 0; i < c.tower_shared; ++i) {
            run_layer(W->shared_layers[size_t(i)], i, cur, dst,
                      dev_c0, dev_sem, (i == 0) ? d_unseen : nullptr,
                      dev_slots, dev_pos, M, stream);
            cur = dst;
            dst = (dst == d_hid_a) ? d_hid_b : d_hid_a;
        }
        return const_cast<__half*>(cur);
    }

    // Runs one residual layer `res_idx` (0..8) and computes head logits `[M, head_out_rows]` FP32 in `d_head_logits`.
    // Pass `head_out_rows = 1024` for residual sampling (or `4096` for full-vocab logits).
    __half* run_residual_step(int res_idx, __half* cur_hidden,
                              const int32_t* dev_prev_code, const uint8_t* dev_sem,
                              const int32_t* dev_slots, const int32_t* dev_pos,
                              int M, int head_out_rows, bool compute_head, cudaStream_t stream) {
        TTS_BLAS_CK(cublasSetStream(blas, stream));
        const auto& c = W->cfg;
        __half* next_hid = (cur_hidden == d_hid_a) ? d_hid_b : d_hid_a;
        run_layer(W->res_layers[size_t(res_idx)], c.tower_shared + res_idx,
                  cur_hidden, next_hid, dev_prev_code, dev_sem, nullptr,
                  dev_slots, dev_pos, M, stream);
        if (compute_head) {
            gemm_f16(blas, next_hid, W->heads[size_t(res_idx)], d_head_logits,
                     M, head_out_rows, c.hidden_size, true, 0.0f);
        }
        return next_hid;
    }

    // Teacher-forced prefill over `M` paired rows: `frames[0..M-1]` paired with `next_frames[0..M-1]`.
    // If `out_head_logits_host != nullptr`, writes `[M, 9, 1024]` FP32 head logits (for Gate E1 parity!).
    void prefill_chunk(const __half* bb_hidden,
                       const int32_t* h_frames_in, const int32_t* h_next_frames,
                       int slot, int pos0, int M,
                       float* out_head_logits_host, cudaStream_t stream) {
        if (M <= 0) return;
        if (M > max_chunk) {
            throw std::runtime_error("TowerRunner::prefill_chunk: M exceeds max_chunk");
        }
        // Ensure previous async H2D copies from our persistent pinned staging buffers have completed.
        if (upload_pending) {
            TTS_CK(cudaEventSynchronize(ev_upload));
            upload_pending = false;
        }
        const auto& c = W->cfg;
        const int rw = c.row_width();
        std::memcpy(h_frames, h_frames_in, size_t(M) * rw * sizeof(int32_t));
        for (int m = 0; m < M; ++m) {
            h_slots[m] = slot;
            h_pos[m] = pos0 + m;
            const int main_nxt = h_next_frames[size_t(m) * rw + 0];
            h_sem[m] = c.is_semantic(main_nxt) ? 1 : 0;
        }
        // Pack all `c.tower_res` code rows [i, m] contiguously so a single H2D upload feeds shared (c0 = i=0) + all 9 residual layers.
        for (int i = 0; i < c.tower_res; ++i) {
            int32_t* dst = h_codes + size_t(i) * M;
            for (int m = 0; m < M; ++m) {
                dst[m] = h_next_frames[size_t(m) * rw + 1 + i]; // c_i
            }
        }
        TTS_CK(cudaMemcpyAsync(d_slots, h_slots, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_pos, h_pos, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_sem, h_sem, size_t(M), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_codes, h_codes, size_t(c.tower_res) * M * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_frames, h_frames, size_t(M) * rw * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaEventRecord(ev_upload, stream));
        upload_pending = true;

        __half* hid = run_shared(bb_hidden, d_codes, d_sem, d_frames, d_slots, d_pos, M, stream);

        for (int i = 0; i < c.tower_res; ++i) {
            hid = run_residual_step(i, hid, d_codes + size_t(i) * M, d_sem, d_slots, d_pos,
                                    M, geom::RES_CB_SIZE, out_head_logits_host != nullptr, stream);
            if (out_head_logits_host) {
                TTS_CK(cudaMemcpyAsync(h_head_tmp, d_head_logits,
                                       size_t(M) * geom::RES_CB_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
                TTS_CK(cudaStreamSynchronize(stream));
                upload_pending = false;
                for (int m = 0; m < M; ++m) {
                    std::memcpy(out_head_logits_host + (size_t(m) * c.tower_res + i) * geom::RES_CB_SIZE,
                                h_head_tmp + size_t(m) * geom::RES_CB_SIZE,
                                geom::RES_CB_SIZE * sizeof(float));
                }
            }
        }
    }
};

} // namespace g6tts
