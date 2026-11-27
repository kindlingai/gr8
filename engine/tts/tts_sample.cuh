// On-GPU sampling for Drama 3 (main token with RAS + 9 residual codebook heads).
//
// Exact match to Plan 1.6 & fish-speech `sample_top_k_top_p` + RAS:
//   - Main token: over 4097 legal candidates (`{248046} U [248099, 252194]`), temperature, top_k, top_p
//     (always keep rank 0; drop when cumsum > top_p or rank >= top_k), exponential-race draw
//     `argmax(p / Exp(1))` from Philox4x32-10 keyed by `(seed, frame, codebook)`.
//   - RAS: if the first draw is a semantic token (SEM0..SEM1) present in the slot's last 16 main tokens,
//     redraw with T=1.0, top_k=30, top_p=0.95 (using retry sub-key).
//   - Residual heads 0..8: over ids < 1024 only with `top_k = min(top_k, 30)` (when top_k > 0).
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "tts_backbone.cuh"
#include "tts_config.h"

namespace g6tts {

// Standard Philox4x32-10 counter-based PRNG (deterministic across runs for fixed (seed, frame, codebook)).
__device__ __forceinline__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t hi0 = __umulhi(M0, ctr.x);
        const uint32_t lo0 = M0 * ctr.x;
        const uint32_t hi1 = __umulhi(M1, ctr.z);
        const uint32_t lo1 = M1 * ctr.z;
        ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += W0;
        key.y += W1;
    }
    return ctr;
}

// Draws an Exp(1) variate for candidate `idx` keyed by `(seed, frame, codebook_subkey)`.
__device__ __forceinline__ float philox_exp1(uint64_t seed, uint32_t frame, uint32_t cb_tag, uint32_t idx) {
    const uint2 key = make_uint2(static_cast<uint32_t>(seed & 0xffffffffull),
                                 static_cast<uint32_t>(seed >> 32));
    const uint4 ctr = make_uint4(idx >> 2, frame, cb_tag, 0x5A3C96E1u);
    const uint4 r = philox4x32_10(ctr, key);
    uint32_t w = 0u;
    switch (idx & 3u) {
        case 0: w = r.x; break;
        case 1: w = r.y; break;
        case 2: w = r.z; break;
        default: w = r.w; break;
    }
    // Map to (0, 1] strictly positive so -logf(u) is finite and > 0
    const float u = (static_cast<float>(w >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return -__logf(fmaxf(u, 1e-7f));
}

// Block-wide argmax over 256 threads; writes (best_val, best_idx) to shared scalars and returns both.
__device__ __forceinline__ void block_argmax256(float val, int idx,
                                                float* s_val, int* s_idx,
                                                float& out_val, int& out_idx) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, val, off);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        s_val[warp] = val;
        s_idx[warp] = idx;
    }
    __syncthreads();
    if (tid < 32) {
        float bv = (tid < 8) ? s_val[tid] : -1e30f;
        int bi = (tid < 8) ? s_idx[tid] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (tid == 0) {
            s_val[0] = bv;
            s_idx[0] = bi;
        }
    }
    __syncthreads();
    out_val = s_val[0];
    out_idx = s_idx[0];
    __syncthreads();
}

// Draws one index in [0, V) from `logits[V]` under (temperature, top_k, top_p) using Philox4x32-10.
// Matches `tower.py:sample_top_k_top_p`:
//   - temperature <= 1e-5 -> greedy argmax
//   - top_k in [1..128] -> exact top-K extraction, drop rank where cumsum > top_p (keep rank 0), exp-race draw
//   - top_k <= 0 or > 128 -> exact binary-searched top_p threshold (or full vocab if top_p >= 1.0), exp-race draw
__device__ int device_sample_row(const float* __restrict__ logits, int V,
                                 float temperature, int top_k, float top_p,
                                 uint64_t seed, uint32_t frame, uint32_t cb_tag,
                                 float* s_val, int* s_idx, float* red,
                                 float* s_top_l, int* s_top_i) {
    const int tid = threadIdx.x;

    // 1. Find global max & argmax
    float lmax = -1e30f;
    int larg = 0x7fffffff;
    for (int i = tid; i < V; i += 256) {
        const float v = logits[i];
        if (v > lmax || (v == lmax && i < larg)) {
            lmax = v;
            larg = i;
        }
    }
    float gmax = 0.0f;
    int garg = 0;
    block_argmax256(lmax, larg, s_val, s_idx, gmax, garg);

    if (temperature <= 1e-5f || top_k == 1) {
        return garg;
    }

    const float inv_t = 1.0f / temperature;

    // 2. Compute softmax denominator over all V candidates
    float sum_e = 0.0f;
    for (int i = tid; i < V; i += 256) {
        sum_e += __expf((logits[i] - gmax) * inv_t);
    }
    const float tot_e = tts_block_sum256(sum_e, red);
    if (!(tot_e > 0.0f) || !isfinite(tot_e)) return garg;

    // 3a. Exact top-K path (covers default top_k=30 and RAS top_k=30)
    if (top_k > 0 && top_k <= 128) {
        const int K = min(top_k, V);
        // Extract K largest logits iteratively (exact descending order, lowest-index tie-break)
        for (int k = 0; k < K; ++k) {
            float bv = -1e30f;
            int bi = 0x7fffffff;
            for (int i = tid; i < V; i += 256) {
                bool taken = false;
                for (int j = 0; j < k; ++j) {
                    if (s_top_i[j] == i) { taken = true; break; }
                }
                if (!taken) {
                    const float v = logits[i];
                    if (v > bv || (v == bv && i < bi)) {
                        bv = v;
                        bi = i;
                    }
                }
            }
            float kv = 0.0f;
            int ki = 0;
            block_argmax256(bv, bi, s_val, s_idx, kv, ki);
            if (tid == 0) {
                s_top_l[k] = kv;
                s_top_i[k] = ki;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float cum = 0.0f;
            float kept_sum = 0.0f;
            int kept_n = 0;
            for (int k = 0; k < K; ++k) {
                const float p = __expf((s_top_l[k] - gmax) * inv_t) / tot_e;
                cum += p;
                if (k == 0 || cum <= top_p || top_p >= 1.0f) {
                    kept_sum += p;
                    kept_n = k + 1;
                } else {
                    break;
                }
            }
            float best_score = -1.0f;
            int best_id = s_top_i[0];
            for (int k = 0; k < kept_n; ++k) {
                const float p = (__expf((s_top_l[k] - gmax) * inv_t) / tot_e) / kept_sum;
                const float e1 = philox_exp1(seed, frame, cb_tag, uint32_t(k));
                const float sc = p / e1;
                if (sc > best_score) {
                    best_score = sc;
                    best_id = s_top_i[k];
                }
            }
            s_idx[0] = best_id;
        }
        __syncthreads();
        return s_idx[0];
    }

    // 3b. Uncapped or large top_k path: find top_p cutoff threshold (if top_p < 1.0) and run exp-race over all V
    float cutoff = -1e30f;
    if (top_p < 1.0f) {
        float lo = gmax - 40.0f * temperature;
        float hi = gmax;
        for (int it = 0; it < 20; ++it) {
            const float mid = 0.5f * (lo + hi);
            float part = 0.0f;
            for (int i = tid; i < V; i += 256) {
                if (logits[i] >= mid) part += __expf((logits[i] - gmax) * inv_t);
            }
            const float mass = tts_block_sum256(part, red) / tot_e;
            if (mass > top_p) lo = mid;
            else hi = mid;
        }
        cutoff = hi;
    }

    float best_sc = -1.0f;
    int best_id = garg;
    for (int i = tid; i < V; i += 256) {
        if (i == garg || logits[i] >= cutoff) {
            const float p = __expf((logits[i] - gmax) * inv_t) / tot_e;
            const float e1 = philox_exp1(seed, frame, cb_tag, uint32_t(i));
            const float sc = p / e1;
            if (sc > best_sc || (sc == best_sc && i < best_id)) {
                best_sc = sc;
                best_id = i;
            }
        }
    }
    float dummy = 0.0f;
    int chosen = garg;
    block_argmax256(best_sc, best_id, s_val, s_idx, dummy, chosen);
    return chosen;
}

// Samples the main token for M rows with per-slot RAS (16-entry main-token ring).
// Outputs:
//   - `d_out_main[M]`: sampled token id (either EOS or in [SEM0..SEM1])
//   - `d_out_c0[M]`:   clamp(main - SEM0, 0, 4095)
//   - `d_out_sem[M]`:  1 if semantic else 0
// Also updates `d_ras_ring[slot, 0..15]` and `d_ras_len[slot]`.
// Grid: (M), Block: 256.
__global__ void k_sample_main_ras(const float* __restrict__ main_logits,
                                  const int32_t* __restrict__ cand_ids,
                                  int32_t* __restrict__ d_ras_ring,
                                  int32_t* __restrict__ d_ras_pos,
                                  const int32_t* __restrict__ d_slots,
                                  const float* __restrict__ d_temp,
                                  const float* __restrict__ d_top_p,
                                  const int32_t* __restrict__ d_top_k,
                                  const uint64_t* __restrict__ d_seed,
                                  const uint32_t* __restrict__ d_frame,
                                  int32_t* __restrict__ d_out_main,
                                  int32_t* __restrict__ d_out_c0,
                                  uint8_t* __restrict__ d_out_sem,
                                  int M, int n_cand, int sem0, int sem1, int tow_vocab) {
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    __shared__ float red[8];
    __shared__ float s_top_l[128];
    __shared__ int s_top_i[128];

    const int m = blockIdx.x;
    if (m >= M) return;
    const float* row = main_logits + size_t(m) * n_cand;
    const float temp = d_temp[m];
    const float tp = d_top_p[m];
    const int tk = d_top_k[m];
    const uint64_t seed = d_seed[m];
    const uint32_t frame = d_frame[m];
    const int slot = d_slots[m];

    const int c1 = device_sample_row(row, n_cand, temp, tk, tp, seed, frame, 0u,
                                     s_val, s_idx, red, s_top_l, s_top_i);
    int tok = cand_ids[c1];

    // RAS check: if first draw is semantic and present in the last 16 main tokens, redraw at 1.0 / 30 / 0.95
    const bool is_sem1 = (tok >= sem0 && tok <= sem1);
    bool repeated = false;
    if (is_sem1 && temp > 1e-5f) {
        const int32_t* ring = d_ras_ring + size_t(slot) * geom::RAS_WINDOW;
        for (int i = 0; i < geom::RAS_WINDOW; ++i) {
            if (ring[i] == tok) { repeated = true; break; }
        }
    }
    if (repeated) {
        const int c2 = device_sample_row(row, n_cand, geom::RAS_TEMPERATURE, geom::RAS_TOP_K, geom::RAS_TOP_P,
                                         seed, frame, 0x100u, s_val, s_idx, red, s_top_l, s_top_i);
        tok = cand_ids[c2];
    }

    if (threadIdx.x == 0) {
        d_out_main[m] = tok;
        const bool is_sem = (tok >= sem0 && tok <= sem1);
        d_out_sem[m] = is_sem ? 1 : 0;
        d_out_c0[m] = max(0, min(tow_vocab - 1, tok - sem0));
        // Push into 16-entry circular ring
        const int rp = d_ras_pos[slot];
        d_ras_ring[size_t(slot) * geom::RAS_WINDOW + (rp % geom::RAS_WINDOW)] = tok;
        d_ras_pos[slot] = rp + 1;
    }
}

// Samples residual codebook `cb_idx` (1..9) over ids `0..1023` with `top_k = min(top_k, 30)`.
// Grid: (M), Block: 256.
__global__ void k_sample_residual(const float* __restrict__ head_logits,
                                  const float* __restrict__ d_temp,
                                  const float* __restrict__ d_top_p,
                                  const int32_t* __restrict__ d_top_k,
                                  const uint64_t* __restrict__ d_seed,
                                  const uint32_t* __restrict__ d_frame,
                                  int32_t* __restrict__ d_out_code,
                                  int M, int head_stride, int cb_idx) {
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    __shared__ float red[8];
    __shared__ float s_top_l[128];
    __shared__ int s_top_i[128];

    const int m = blockIdx.x;
    if (m >= M) return;
    const float* row = head_logits + size_t(m) * head_stride;
    const int raw_tk = d_top_k[m];
    const int eff_tk = (raw_tk > 0) ? min(raw_tk, 30) : 30;
    const int code = device_sample_row(row, geom::RES_CB_SIZE, d_temp[m], eff_tk, d_top_p[m],
                                       d_seed[m], d_frame[m], uint32_t(cb_idx),
                                       s_val, s_idx, red, s_top_l, s_top_i);
    if (threadIdx.x == 0) d_out_code[m] = code;
}

} // namespace g6tts
