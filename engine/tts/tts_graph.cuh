// Fused per-frame CUDA graph & codec stride CUDA graph runner for Drama 3 (Wave 3.3).
//
// Captures:
//   1. One `cudaGraphExec_t` per batch bucket M in {1, 2, 4, 8} executing an entire autoregressive frame with
//      ZERO host round-trips:
//        gather slot inputs (hidden, last_row, pos, sampling params)
//        -> sliced lm_head -> on-GPU main-token RAS sampling
//        -> 4 shared tower layers
//        -> 9 x (residual tower layer -> head -> residual sampling)
//        -> commit [M, 11] output row, update device EOS active mask, push [c0..c9] to device code ring,
//           advance device slot pos & frame_idx
//        -> 40-layer backbone forward_device on newly sampled [M, 11] row at pos+1 -> scatter d_slot_hidden
//        -> async D2H copy of [M, 11] output rows and active mask to pinned host memory.
//      Rows address physical slots through a device table `d_slot_of_row[M]`, with unused bucket rows mapped
//      to a dedicated throwaway `PAD_SLOT = max_slots` so captured graphs are reused for any slot assignment.
//   2. One `cudaGraphExec_t` per (M in {1, 2, 4, 8}, stride in {1, 2, 4}) running:
//        gather newest `stride` frames from device code ring -> TtsCodec::decode_frames -> async D2H s16 PCM.
//
// Eager (`use_graph == false`) and graphed (`use_graph == true`) execute the exact same `record_frame_ops`
// and `record_codec_ops` sequences for bit-identical codes on fixed seeds.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tts_backbone.cuh"
#include "tts_config.h"
#include "tts_sample.cuh"
#include "tts_tower.cuh"
#include "tts_weights.h"

namespace g6tts {

constexpr int GRAPH_BUCKETS[4] = {1, 2, 4, 8};
constexpr int CODEC_STRIDES[3] = {1, 2, 4};
constexpr int CODE_RING_CAP = 16; // power-of-2 per-slot device code ring

inline int pick_graph_bucket(int active_m, int max_slots) {
    for (int b : GRAPH_BUCKETS) {
        if (b >= active_m && b <= max_slots) return b;
    }
    return std::min(max_slots, active_m);
}

inline int bucket_index(int b) {
    if (b == 1) return 0;
    if (b == 2) return 1;
    if (b == 4) return 2;
    if (b == 8) return 3;
    return -1;
}

inline int stride_index(int s) {
    if (s == 1) return 0;
    if (s == 2) return 1;
    if (s == 4) return 2;
    return -1;
}

// ------------------------------------------------------------------------------------------------ Device glue kernels

// Step 0: Zero-sync single-launch slot admission from pinned `AdmitParams`.
struct AdmitParams {
    int32_t last_row[16];
    int32_t ras_ring[geom::RAS_WINDOW];
    int32_t pos;
    int32_t ras_pos;
    float temperature;
    float top_p;
    int32_t top_k;
    uint64_t seed;
};

// Grid: 1, Block: 256.
__global__ void k_graph_admit_slot(
    int slot,
    const __nv_bfloat16* __restrict__ last_hidden,
    const AdmitParams* __restrict__ params,
    __nv_bfloat16* __restrict__ slot_hidden,
    int32_t* __restrict__ slot_last_row,
    int32_t* __restrict__ slot_pos,
    uint32_t* __restrict__ slot_fidx,
    uint8_t* __restrict__ slot_active,
    float* __restrict__ slot_temp,
    float* __restrict__ slot_top_p,
    int32_t* __restrict__ slot_top_k,
    uint64_t* __restrict__ slot_seed,
    int32_t* __restrict__ ras_ring,
    int32_t* __restrict__ ras_pos,
    int32_t* __restrict__ code_ring_wr,
    int H, int rw) {
    __nv_bfloat16* dst_h = slot_hidden + size_t(slot) * H;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        dst_h[d] = last_hidden[d];
    }
    if (threadIdx.x < rw) {
        slot_last_row[size_t(slot) * rw + threadIdx.x] = params->last_row[threadIdx.x];
    }
    if (threadIdx.x < geom::RAS_WINDOW) {
        ras_ring[size_t(slot) * geom::RAS_WINDOW + threadIdx.x] = params->ras_ring[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        slot_pos[slot] = params->pos;
        slot_fidx[slot] = 0u;
        slot_active[slot] = 1u;
        slot_temp[slot] = params->temperature;
        slot_top_p[slot] = params->top_p;
        slot_top_k[slot] = params->top_k;
        slot_seed[slot] = params->seed;
        ras_pos[slot] = params->ras_pos;
        code_ring_wr[slot] = 0;
    }
}

__global__ void k_graph_deactivate_slot(uint8_t* __restrict__ slot_active, int slot) {
    slot_active[slot] = 0u;
}

// Step 1: Gather per-row inputs for bucket rows 0..M-1 from slot-indexed persistent device state.
// Grid: (M), Block: 256.
__global__ void k_graph_gather_step_in(
    const int32_t* __restrict__ slot_of_row,
    const __nv_bfloat16* __restrict__ slot_hidden,
    const int32_t* __restrict__ slot_last_row,
    const int32_t* __restrict__ slot_pos,
    const uint32_t* __restrict__ slot_fidx,
    const float* __restrict__ slot_temp,
    const float* __restrict__ slot_top_p,
    const int32_t* __restrict__ slot_top_k,
    const uint64_t* __restrict__ slot_seed,
    __nv_bfloat16* __restrict__ step_hidden,
    int32_t* __restrict__ step_prev_row,
    int32_t* __restrict__ step_pos_tw,
    int32_t* __restrict__ step_pos_bb,
    float* __restrict__ step_temp,
    float* __restrict__ step_top_p,
    int32_t* __restrict__ step_top_k,
    uint64_t* __restrict__ step_seed,
    uint32_t* __restrict__ step_fidx,
    int M, int H, int rw, int cap) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int s = slot_of_row[m];
    const __nv_bfloat16* sh = slot_hidden + size_t(s) * H;
    __nv_bfloat16* dh = step_hidden + size_t(m) * H;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        dh[d] = sh[d];
    }
    if (threadIdx.x < rw) {
        step_prev_row[size_t(m) * rw + threadIdx.x] = slot_last_row[size_t(s) * rw + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        const int p = slot_pos[s];
        step_pos_tw[m] = p;
        step_pos_bb[m] = (p + 1 < cap) ? (p + 1) : (cap - 1);
        step_temp[m] = slot_temp[s];
        step_top_p[m] = slot_top_p[s];
        step_top_k[m] = slot_top_k[s];
        step_seed[m] = slot_seed[s];
        step_fidx[m] = slot_fidx[s];
    }
}

// Step 5: Pack [M, 11] output row, update EOS active mask, push [c0..c9] to device code ring,
// and advance device slot_last_row / slot_pos / slot_fidx.
// Grid: (M), Block: 32.
__global__ void k_graph_commit_frame(
    const int32_t* __restrict__ slot_of_row,
    const int32_t* __restrict__ out_main,
    const int32_t* __restrict__ out_c0,
    const int32_t* __restrict__ out_res_codes, // [9, M]
    int32_t* __restrict__ out_rows,            // [M, 11]
    int32_t* __restrict__ slot_last_row,       // [S_tot, 11]
    int32_t* __restrict__ slot_pos,            // [S_tot]
    uint32_t* __restrict__ slot_fidx,          // [S_tot]
    uint8_t* __restrict__ slot_active,         // [S_tot]
    int32_t* __restrict__ code_ring,           // [S_tot, CODE_RING_CAP, 10]
    int32_t* __restrict__ code_ring_wr,        // [S_tot]
    int M, int rw, int n_res, int eos_id, int cap) {
    const int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0) return;

    const int s = slot_of_row[m];
    const int32_t main_tok = out_main[m];
    const int32_t c0 = out_c0[m];

    int32_t* row = out_rows + size_t(m) * rw;
    row[0] = main_tok;
    row[1] = c0;
    for (int i = 0; i < n_res; ++i) {
        row[2 + i] = out_res_codes[size_t(i) * M + m];
    }

    if (slot_active[s] != 0) {
        if (main_tok == eos_id) {
            slot_active[s] = 0;
        } else {
            const int wr = code_ring_wr[s];
            int32_t* ring_dst = code_ring + (size_t(s) * CODE_RING_CAP + (wr & (CODE_RING_CAP - 1))) * geom::NCB;
            for (int cb = 0; cb < geom::NCB; ++cb) ring_dst[cb] = row[1 + cb];
            code_ring_wr[s] = wr + 1;

            int32_t* sl_row = slot_last_row + size_t(s) * rw;
            for (int j = 0; j < rw; ++j) sl_row[j] = row[j];
            const int p = slot_pos[s];
            slot_pos[s] = (p + 1 < cap) ? (p + 1) : (cap - 1);
            slot_fidx[s] += 1u;
        }
    }
}

// Step 7: Scatter newly computed backbone hidden state `bb_hidden[m, :]` -> `slot_hidden[slot_of_row[m], :]`.
// Grid: (M), Block: 256.
__global__ void k_graph_scatter_hidden(
    const int32_t* __restrict__ slot_of_row,
    const __nv_bfloat16* __restrict__ bb_hidden,
    __nv_bfloat16* __restrict__ slot_hidden,
    int M, int H) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int s = slot_of_row[m];
    const __nv_bfloat16* src = bb_hidden + size_t(m) * H;
    __nv_bfloat16* dst = slot_hidden + size_t(s) * H;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Codec graph step 1: Gather `[M, stride, 10]` codes from device code ring for `codec_slots[m]` starting at `codec_start[m]`.
// Grid: (M * stride), Block: 32.
__global__ void k_graph_gather_codec_codes(
    const int32_t* __restrict__ codec_slots,
    const int32_t* __restrict__ codec_start,
    const int32_t* __restrict__ code_ring,
    int32_t* __restrict__ codec_codes_out,
    int M, int stride) {
    const int idx = blockIdx.x;
    if (idx >= M * stride) return;
    const int m = idx / stride;
    const int f = idx % stride;
    const int cb = threadIdx.x;
    if (cb >= geom::NCB) return;
    const int s = codec_slots[m];
    const int ridx = (codec_start[m] + f) & (CODE_RING_CAP - 1);
    codec_codes_out[size_t(idx) * geom::NCB + cb] =
        code_ring[(size_t(s) * CODE_RING_CAP + ridx) * geom::NCB + cb];
}

// ------------------------------------------------------------------------------------------------ Profiling

struct FrameProfile {
    bool enabled = false;
    double sample_ms = 0.0;
    double codec_ms = 0.0;
    void reset() { sample_ms = codec_ms = 0.0; }
};

// ------------------------------------------------------------------------------------------------ Fused Graph Runner

template <typename CodecT>
struct TtsGraphRunner {
    const TtsWeights* W = nullptr;
    BackboneRunner* bb = nullptr;
    TowerRunner* tower = nullptr;
    CodecT* codec = nullptr;

    int max_slots = 8; // real slots 0..max_slots-1; slot `pad_slot = max_slots` is throwaway padding
    int pad_slot = 8;
    int total_slots = 9;
    int cap = 4096;
    bool use_graph = true;

    // Persistent slot-indexed device state (sized for `total_slots = max_slots + 1`)
    __nv_bfloat16* d_slot_hidden = nullptr;    // [total_slots, H]
    int32_t* d_slot_last_row = nullptr; // [total_slots, 11]
    int32_t* d_slot_pos = nullptr;      // [total_slots]
    uint32_t* d_slot_fidx = nullptr;    // [total_slots]
    uint8_t* d_slot_active = nullptr;   // [total_slots]
    float* d_slot_temp = nullptr;       // [total_slots]
    float* d_slot_top_p = nullptr;      // [total_slots]
    int32_t* d_slot_top_k = nullptr;    // [total_slots]
    uint64_t* d_slot_seed = nullptr;    // [total_slots]
    int32_t* d_ras_ring = nullptr;      // [total_slots, 16]
    int32_t* d_ras_pos = nullptr;       // [total_slots]
    int32_t* d_code_ring = nullptr;     // [total_slots, CODE_RING_CAP, 10]
    int32_t* d_code_ring_wr = nullptr;  // [total_slots]

    // Fixed-address bucket step device buffers (sized for `max_slots`)
    int32_t* d_slot_of_row = nullptr;   // [max_slots]
    __nv_bfloat16* d_step_hidden = nullptr;    // [max_slots, H]
    int32_t* d_step_prev_row = nullptr; // [max_slots, 11]
    int32_t* d_step_pos_tw = nullptr;   // [max_slots]
    int32_t* d_step_pos_bb = nullptr;   // [max_slots]
    float* d_step_temp = nullptr;       // [max_slots]
    float* d_step_top_p = nullptr;      // [max_slots]
    int32_t* d_step_top_k = nullptr;    // [max_slots]
    uint64_t* d_step_seed = nullptr;    // [max_slots]
    uint32_t* d_step_fidx = nullptr;    // [max_slots]
    int32_t* d_out_main = nullptr;      // [max_slots]
    int32_t* d_out_c0 = nullptr;        // [max_slots]
    uint8_t* d_out_sem = nullptr;       // [max_slots]
    int32_t* d_out_res_codes = nullptr; // [9, max_slots]
    int32_t* d_out_rows = nullptr;      // [max_slots, 11]

    // Fixed-address codec graph device buffers
    int32_t* d_codec_slots = nullptr;   // [max_slots]
    int32_t* d_codec_start = nullptr;   // [max_slots]
    int32_t* d_codec_codes = nullptr;   // [max_slots * 4 * 10]
    int16_t* d_codec_pcm = nullptr;     // [max_slots * 4 * 2048]

    // Pinned host staging (zero pageable DMA; fixed addresses for graph D2H nodes!)
    int32_t* h_slot_of_row = nullptr;   // [max_slots]
    int32_t* h_out_rows = nullptr;      // [max_slots, 11]
    uint8_t* h_out_active = nullptr;    // [total_slots]
    int32_t* h_codec_slots = nullptr;   // [max_slots]
    int32_t* h_codec_start = nullptr;   // [max_slots]
    int16_t* h_codec_pcm = nullptr;     // [max_slots * 4 * 2048]
    AdmitParams* h_admit_params = nullptr; // [max_slots] pinned per-slot admission staging
    AdmitParams* d_admit_params = nullptr; // [max_slots] device per-slot admission staging

    // Captured CUDA graphs: frame_exec[bucket_idx], split frame0 graphs, codec_exec[bucket_idx][stride_idx]
    std::array<cudaGraphExec_t, 4> frame_exec = {nullptr, nullptr, nullptr, nullptr};
    std::array<cudaGraphExec_t, 4> frame_tower_exec = {nullptr, nullptr, nullptr, nullptr};
    std::array<cudaGraphExec_t, 4> frame_bb_exec = {nullptr, nullptr, nullptr, nullptr};
    std::array<std::array<cudaGraphExec_t, 3>, 4> codec_exec = {{
        {nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr},
    }};

    cudaEvent_t ev_frame = nullptr;
    cudaEvent_t ev_codec = nullptr;
    cudaEvent_t ev_p0 = nullptr, ev_p1 = nullptr;
    FrameProfile prof;

    ~TtsGraphRunner() { release(); }

    void destroy_graphs() {
        for (auto& g : frame_exec) {
            if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        }
        for (auto& g : frame_tower_exec) {
            if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        }
        for (auto& g : frame_bb_exec) {
            if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        }
        for (auto& row : codec_exec) {
            for (auto& g : row) {
                if (g) { cudaGraphExecDestroy(g); g = nullptr; }
            }
        }
    }

    void release() {
        destroy_graphs();
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto freeh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        freep(d_slot_hidden); freep(d_slot_last_row); freep(d_slot_pos); freep(d_slot_fidx);
        freep(d_slot_active); freep(d_slot_temp); freep(d_slot_top_p); freep(d_slot_top_k);
        freep(d_slot_seed); freep(d_ras_ring); freep(d_ras_pos);
        freep(d_code_ring); freep(d_code_ring_wr);
        freep(d_slot_of_row); freep(d_step_hidden); freep(d_step_prev_row);
        freep(d_step_pos_tw); freep(d_step_pos_bb);
        freep(d_step_temp); freep(d_step_top_p); freep(d_step_top_k);
        freep(d_step_seed); freep(d_step_fidx);
        freep(d_out_main); freep(d_out_c0); freep(d_out_sem);
        freep(d_out_res_codes); freep(d_out_rows);
        freep(d_codec_slots); freep(d_codec_start); freep(d_codec_codes); freep(d_codec_pcm);
        freep(d_admit_params);
        freeh(h_slot_of_row); freeh(h_out_rows); freeh(h_out_active);
        freeh(h_codec_slots); freeh(h_codec_start); freeh(h_codec_pcm); freeh(h_admit_params);
        if (ev_frame) { cudaEventDestroy(ev_frame); ev_frame = nullptr; }
        if (ev_codec) { cudaEventDestroy(ev_codec); ev_codec = nullptr; }
        if (ev_p0) { cudaEventDestroy(ev_p0); ev_p0 = nullptr; }
        if (ev_p1) { cudaEventDestroy(ev_p1); ev_p1 = nullptr; }
    }

    // Note: Caller must initialize `bb`, `tower`, and `codec` with `max_slots + 1` slots so `pad_slot = max_slots` is valid!
    void init(const TtsWeights& weights, BackboneRunner& bb_in, TowerRunner& tower_in, CodecT& codec_in,
              int streams, int kv_cap, bool enable_graph) {
        release();
        W = &weights;
        bb = &bb_in;
        tower = &tower_in;
        codec = &codec_in;
        max_slots = std::max(1, streams);
        pad_slot = max_slots;
        total_slots = max_slots + 1;
        cap = std::max(16, kv_cap);
        use_graph = enable_graph;

        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int rw = c.row_width();

        auto alloc_d = [](auto*& p, size_t n) {
            TTS_CK(cudaMalloc(&p, n * sizeof(*p)));
            TTS_CK(cudaMemset(p, 0, n * sizeof(*p)));
        };
        alloc_d(d_slot_hidden, size_t(total_slots) * H);
        alloc_d(d_slot_last_row, size_t(total_slots) * rw);
        alloc_d(d_slot_pos, size_t(total_slots));
        alloc_d(d_slot_fidx, size_t(total_slots));
        alloc_d(d_slot_active, size_t(total_slots));
        alloc_d(d_slot_temp, size_t(total_slots));
        alloc_d(d_slot_top_p, size_t(total_slots));
        alloc_d(d_slot_top_k, size_t(total_slots));
        alloc_d(d_slot_seed, size_t(total_slots));
        alloc_d(d_ras_ring, size_t(total_slots) * geom::RAS_WINDOW);
        alloc_d(d_ras_pos, size_t(total_slots));
        alloc_d(d_code_ring, size_t(total_slots) * CODE_RING_CAP * geom::NCB);
        alloc_d(d_code_ring_wr, size_t(total_slots));

        alloc_d(d_slot_of_row, size_t(max_slots));
        alloc_d(d_step_hidden, size_t(max_slots) * H);
        alloc_d(d_step_prev_row, size_t(max_slots) * rw);
        alloc_d(d_step_pos_tw, size_t(max_slots));
        alloc_d(d_step_pos_bb, size_t(max_slots));
        alloc_d(d_step_temp, size_t(max_slots));
        alloc_d(d_step_top_p, size_t(max_slots));
        alloc_d(d_step_top_k, size_t(max_slots));
        alloc_d(d_step_seed, size_t(max_slots));
        alloc_d(d_step_fidx, size_t(max_slots));
        alloc_d(d_out_main, size_t(max_slots));
        alloc_d(d_out_c0, size_t(max_slots));
        alloc_d(d_out_sem, size_t(max_slots));
        alloc_d(d_out_res_codes, size_t(c.tower_res) * max_slots);
        alloc_d(d_out_rows, size_t(max_slots) * rw);

        alloc_d(d_codec_slots, size_t(max_slots));
        alloc_d(d_codec_start, size_t(max_slots));
        alloc_d(d_codec_codes, size_t(max_slots) * 4 * geom::NCB);
        alloc_d(d_codec_pcm, size_t(max_slots) * 4 * geom::FRAME_SAMPLES);
        alloc_d(d_admit_params, size_t(max_slots));

        TTS_CK(cudaHostAlloc(&h_slot_of_row, size_t(max_slots) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_out_rows, size_t(max_slots) * rw * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_out_active, size_t(total_slots) * sizeof(uint8_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_codec_slots, size_t(max_slots) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_codec_start, size_t(max_slots) * sizeof(int32_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_codec_pcm, size_t(max_slots) * 4 * geom::FRAME_SAMPLES * sizeof(int16_t), cudaHostAllocDefault));
        TTS_CK(cudaHostAlloc(&h_admit_params, size_t(max_slots) * sizeof(AdmitParams), cudaHostAllocDefault));

        TTS_CK(cudaEventCreate(&ev_frame));
        TTS_CK(cudaEventCreate(&ev_codec));
        TTS_CK(cudaEventCreate(&ev_p0));
        TTS_CK(cudaEventCreate(&ev_p1));
    }

    // Admits a newly prefilled stream into physical `slot` (zero host sync!).
    void admit_slot(int slot, const __nv_bfloat16* d_last_hidden, const int32_t* h_last_row,
                    int pos, const int32_t* h_ras_ring16, int ras_pos,
                    float temperature, float top_p, int top_k, uint64_t seed, cudaStream_t stream) {
        const auto& c = W->cfg;
        const int rw = c.row_width();
        AdmitParams& ap = h_admit_params[slot];
        std::memset(&ap, 0, sizeof(ap));
        std::memcpy(ap.last_row, h_last_row, size_t(rw) * sizeof(int32_t));
        std::memcpy(ap.ras_ring, h_ras_ring16, size_t(geom::RAS_WINDOW) * sizeof(int32_t));
        ap.pos = pos;
        ap.ras_pos = ras_pos;
        ap.temperature = temperature;
        ap.top_p = top_p;
        ap.top_k = top_k;
        ap.seed = seed;
        TTS_CK(cudaMemcpyAsync(d_admit_params + slot, &ap, sizeof(AdmitParams), cudaMemcpyHostToDevice, stream));
        k_graph_admit_slot<<<1, 256, 0, stream>>>(
            slot, d_last_hidden, d_admit_params + slot,
            d_slot_hidden, d_slot_last_row, d_slot_pos, d_slot_fidx, d_slot_active,
            d_slot_temp, d_slot_top_p, d_slot_top_k, d_slot_seed,
            d_ras_ring, d_ras_pos, d_code_ring_wr, c.hidden_size, rw);
    }

    // Marks a slot inactive on device (zero host sync).
    void deactivate_slot(int slot, cudaStream_t stream) {
        k_graph_deactivate_slot<<<1, 1, 0, stream>>>(d_slot_active, slot);
    }

    // Part 1 of a frame: gather -> lm_head + main RAS -> 4 shared + 9 residual tower layers -> commit row & code ring.
    // If `copy_d2h` is true, also copies `d_out_rows` and `d_slot_active` to pinned host buffers (for early frame-0 emit).
    void record_frame_tower_ops(int M, bool copy_d2h, cudaStream_t stream) {
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        const int rw = c.row_width();

        auto tic_s = [&]() { if (prof.enabled) cudaEventRecord(ev_p0, stream); };
        auto toc_s = [&](double& dst) {
            if (prof.enabled) {
                cudaEventRecord(ev_p1, stream);
                cudaEventSynchronize(ev_p1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_p0, ev_p1);
                dst += double(ms);
            }
        };

        // 1. Gather per-row state from persistent slot tables
        k_graph_gather_step_in<<<M, 256, 0, stream>>>(
            d_slot_of_row, d_slot_hidden, d_slot_last_row, d_slot_pos, d_slot_fidx,
            d_slot_temp, d_slot_top_p, d_slot_top_k, d_slot_seed,
            d_step_hidden, d_step_prev_row, d_step_pos_tw, d_step_pos_bb,
            d_step_temp, d_step_top_p, d_step_top_k, d_step_seed, d_step_fidx,
            M, H, rw, cap);

        // 2. Sliced lm_head + on-GPU main-token RAS sampling
        bb->compute_main_logits(d_step_hidden, M, stream);
        tic_s();
        k_sample_main_ras<<<M, 256, 0, stream>>>(
            bb->d_main_logits, W->main_cand_ids_dev, d_ras_ring, d_ras_pos,
            d_slot_of_row, d_step_temp, d_step_top_p, d_step_top_k, d_step_seed, d_step_fidx,
            d_out_main, d_out_c0, d_out_sem,
            M, c.main_candidates(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);
        toc_s(prof.sample_ms);

        // 3. 4 shared tower layers
        __nv_bfloat16* cur_hid = tower->run_shared(d_step_hidden, d_out_c0, d_out_sem,
                                                   d_step_prev_row, d_slot_of_row, d_step_pos_tw, M, stream);

        // 4. 9 x (residual tower layer -> head -> residual sampling)
        const int32_t* prev_code_dev = d_out_c0;
        for (int i = 0; i < c.tower_res; ++i) {
            cur_hid = tower->run_residual_step(i, cur_hid, prev_code_dev, d_out_sem,
                                               d_slot_of_row, d_step_pos_tw, M, geom::RES_CB_SIZE, true, stream);
            int32_t* dst_code = d_out_res_codes + size_t(i) * M;
            tic_s();
            k_sample_residual<<<M, 256, 0, stream>>>(
                tower->d_head_logits, d_step_temp, d_step_top_p, d_step_top_k,
                d_step_seed, d_step_fidx, dst_code, M, geom::RES_CB_SIZE, i + 1);
            toc_s(prof.sample_ms);
            prev_code_dev = dst_code;
        }

        // 5. Commit [M, 11] row, EOS active mask, device code ring, slot pos & frame_idx
        k_graph_commit_frame<<<M, 32, 0, stream>>>(
            d_slot_of_row, d_out_main, d_out_c0, d_out_res_codes,
            d_out_rows, d_slot_last_row, d_slot_pos, d_slot_fidx, d_slot_active,
            d_code_ring, d_code_ring_wr, M, rw, c.tower_res, c.eos_token_id, cap);

        if (copy_d2h) {
            TTS_CK(cudaMemcpyAsync(h_out_rows, d_out_rows, size_t(M) * rw * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            TTS_CK(cudaMemcpyAsync(h_out_active, d_slot_active, size_t(total_slots) * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        }
    }

    // Part 2 of a frame: 40-layer backbone forward on newly sampled [M, 11] row at pos+1 -> scatter d_slot_hidden.
    void record_frame_backbone_ops(int M, cudaStream_t stream) {
        const auto& c = W->cfg;
        const int H = c.hidden_size;
        bb->forward_device(d_out_rows, d_slot_of_row, d_step_pos_bb, M, false, false, stream);
        k_graph_scatter_hidden<<<M, 256, 0, stream>>>(d_slot_of_row, bb->d_final_bf16, d_slot_hidden, M, H);
    }

    // Records all GPU ops of ONE autoregressive frame for bucket size `M` (shared by eager & graph capture!).
    void record_frame_ops(int M, cudaStream_t stream) {
        const int rw = W->cfg.row_width();
        record_frame_tower_ops(M, false, stream);
        record_frame_backbone_ops(M, stream);
        TTS_CK(cudaMemcpyAsync(h_out_rows, d_out_rows, size_t(M) * rw * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        TTS_CK(cudaMemcpyAsync(h_out_active, d_slot_active, size_t(total_slots) * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    }

    void ensure_frame_split_graphs(int M, cudaStream_t stream) {
        const int bi = bucket_index(M);
        if (bi < 0 || (frame_tower_exec[size_t(bi)] != nullptr && frame_bb_exec[size_t(bi)] != nullptr)) return;

        for (int m = 0; m < M; ++m) h_slot_of_row[m] = pad_slot;
        TTS_CK(cudaMemcpyAsync(d_slot_of_row, h_slot_of_row, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaStreamSynchronize(stream));

        const bool saved_bb_prof = bb->prof.enabled;
        const bool saved_tw_prof = tower->prof.enabled;
        const bool saved_fr_prof = prof.enabled;
        bb->prof.enabled = false;
        tower->prof.enabled = false;
        prof.enabled = false;

        if (!frame_tower_exec[size_t(bi)]) {
            cudaGraph_t graph = nullptr;
            TTS_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            record_frame_tower_ops(M, true, stream);
            TTS_CK(cudaStreamEndCapture(stream, &graph));
            TTS_CK(cudaGraphInstantiate(&frame_tower_exec[size_t(bi)], graph, 0));
            TTS_CK(cudaGraphDestroy(graph));
        }
        if (!frame_bb_exec[size_t(bi)]) {
            cudaGraph_t graph = nullptr;
            TTS_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            record_frame_backbone_ops(M, stream);
            TTS_CK(cudaStreamEndCapture(stream, &graph));
            TTS_CK(cudaGraphInstantiate(&frame_bb_exec[size_t(bi)], graph, 0));
            TTS_CK(cudaGraphDestroy(graph));
        }

        bb->prof.enabled = saved_bb_prof;
        tower->prof.enabled = saved_tw_prof;
        prof.enabled = saved_fr_prof;
    }

    // Launches ONLY part 1 (lm_head + tower + sample + commit + D2H) so frame 0 audio can be decoded and
    // emitted BEFORE running the 40-layer backbone forward for frame 1.
    int step_frame_tower_only(const std::vector<int>& active_slots, cudaStream_t stream) {
        const int n_act = static_cast<int>(active_slots.size());
        if (n_act <= 0) return 0;

        const bool run_graph = use_graph && !prof.enabled && !bb->prof.enabled && !tower->prof.enabled;
        const int M = run_graph ? pick_graph_bucket(n_act, max_slots) : n_act;
        const int bi = bucket_index(M);
        if (run_graph && bi >= 0) {
            ensure_frame_split_graphs(M, stream);
        }

        for (int m = 0; m < n_act; ++m) h_slot_of_row[m] = active_slots[size_t(m)];
        for (int m = n_act; m < M; ++m) h_slot_of_row[m] = pad_slot;
        TTS_CK(cudaMemcpyAsync(d_slot_of_row, h_slot_of_row, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        if (run_graph && bi >= 0 && frame_tower_exec[size_t(bi)] != nullptr) {
            TTS_CK(cudaGraphLaunch(frame_tower_exec[size_t(bi)], stream));
        } else {
            record_frame_tower_ops(M, true, stream);
        }
        TTS_CK(cudaEventRecord(ev_frame, stream));
        TTS_CK(cudaEventSynchronize(ev_frame));
        return M;
    }

    // Launches ONLY part 2 (40-layer backbone forward + scatter hidden) matching the preceding
    // `step_frame_tower_only` bucket `M` (asynchronously pipelined on `stream`, zero host sync!).
    void step_frame_backbone_only(int M, cudaStream_t stream) {
        if (M <= 0) return;
        const bool run_graph = use_graph && !prof.enabled && !bb->prof.enabled && !tower->prof.enabled;
        const int bi = bucket_index(M);
        if (run_graph && bi >= 0 && frame_bb_exec[size_t(bi)] != nullptr) {
            TTS_CK(cudaGraphLaunch(frame_bb_exec[size_t(bi)], stream));
        } else {
            record_frame_backbone_ops(M, stream);
        }
    }

    // Pre-captures all bucket graphs during startup using throwaway `pad_slot` so the first request pays
    // zero cold graph-capture latency.
    void warmup_graphs(cudaStream_t stream) {
        if (!use_graph) return;
        for (int b : GRAPH_BUCKETS) {
            if (b > max_slots) break;
            ensure_frame_graph(b, stream);
            ensure_frame_split_graphs(b, stream);
            if (codec && codec->loaded()) {
                for (int s : CODEC_STRIDES) {
                    ensure_codec_graph(b, s, stream);
                }
            }
        }
        TTS_CK(cudaStreamSynchronize(stream));
    }

    // Lazily captures the frame graph for bucket `M` on `stream` using throwaway `pad_slot` so real slots are untouched!
    void ensure_frame_graph(int M, cudaStream_t stream) {
        const int bi = bucket_index(M);
        if (bi < 0 || frame_exec[size_t(bi)] != nullptr) return;

        // Point all M rows at throwaway `pad_slot` during capture (never touches any live slot's state!).
        for (int m = 0; m < M; ++m) h_slot_of_row[m] = pad_slot;
        TTS_CK(cudaMemcpyAsync(d_slot_of_row, h_slot_of_row, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaStreamSynchronize(stream));

        // Disable per-component event sync during graph capture (illegal inside capture)
        const bool saved_bb_prof = bb->prof.enabled;
        const bool saved_tw_prof = tower->prof.enabled;
        const bool saved_fr_prof = prof.enabled;
        bb->prof.enabled = false;
        tower->prof.enabled = false;
        prof.enabled = false;

        cudaGraph_t graph = nullptr;
        TTS_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        record_frame_ops(M, stream);
        TTS_CK(cudaStreamEndCapture(stream, &graph));
        TTS_CK(cudaGraphInstantiate(&frame_exec[size_t(bi)], graph, 0));
        TTS_CK(cudaGraphDestroy(graph));

        bb->prof.enabled = saved_bb_prof;
        tower->prof.enabled = saved_tw_prof;
        prof.enabled = saved_fr_prof;
    }

    // Launches ONE autoregressive frame over `active_slots` (automatically padded to bucket M in {1, 2, 4, 8}
    // when `use_graph` is true), waits on ONE event, and leaves `[M_active, 11]` rows in `h_out_rows`.
    int step_frame(const std::vector<int>& active_slots, cudaStream_t stream) {
        const int n_act = static_cast<int>(active_slots.size());
        if (n_act <= 0) return 0;

        const bool run_graph = use_graph && !prof.enabled && !bb->prof.enabled && !tower->prof.enabled;
        const int M = run_graph ? pick_graph_bucket(n_act, max_slots) : n_act;
        const int bi = bucket_index(M);

        if (run_graph && bi >= 0) {
            ensure_frame_graph(M, stream);
        }

        for (int m = 0; m < n_act; ++m) h_slot_of_row[m] = active_slots[size_t(m)];
        for (int m = n_act; m < M; ++m) h_slot_of_row[m] = pad_slot;
        TTS_CK(cudaMemcpyAsync(d_slot_of_row, h_slot_of_row, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        if (run_graph && bi >= 0 && frame_exec[size_t(bi)] != nullptr) {
            TTS_CK(cudaGraphLaunch(frame_exec[size_t(bi)], stream));
        } else {
            record_frame_ops(M, stream);
        }
        TTS_CK(cudaEventRecord(ev_frame, stream));
        TTS_CK(cudaEventSynchronize(ev_frame));
        return M;
    }

    // Records all GPU ops of ONE codec stride decode for `(M, stride)`.
    void record_codec_ops(int M, int stride, cudaStream_t stream) {
        k_graph_gather_codec_codes<<<M * stride, 32, 0, stream>>>(
            d_codec_slots, d_codec_start, d_code_ring, d_codec_codes, M, stride);
        codec->decode_frames(d_codec_codes, M, stride, d_codec_slots, d_codec_pcm, stream);
        TTS_CK(cudaMemcpyAsync(h_codec_pcm, d_codec_pcm,
                               size_t(M) * stride * geom::FRAME_SAMPLES * sizeof(int16_t),
                               cudaMemcpyDeviceToHost, stream));
    }

    void ensure_codec_graph(int M, int stride, cudaStream_t stream) {
        const int bi = bucket_index(M);
        const int si = stride_index(stride);
        if (bi < 0 || si < 0 || codec_exec[size_t(bi)][size_t(si)] != nullptr) return;

        for (int m = 0; m < M; ++m) {
            h_codec_slots[m] = pad_slot;
            h_codec_start[m] = 0;
        }
        TTS_CK(cudaMemcpyAsync(d_codec_slots, h_codec_slots, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_codec_start, h_codec_start, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaStreamSynchronize(stream));

        cudaGraph_t graph = nullptr;
        TTS_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        record_codec_ops(M, stride, stream);
        TTS_CK(cudaStreamEndCapture(stream, &graph));
        TTS_CK(cudaGraphInstantiate(&codec_exec[size_t(bi)][size_t(si)], graph, 0));
        TTS_CK(cudaGraphDestroy(graph));
    }

    // Decodes `stride` (in {1, 2, 4}) newest frames from the device code ring for `slots` starting at `ring_starts`.
    // Output lands in `h_codec_pcm[m * stride * 2048 .. +(stride*2048)]` for `m = 0..slots.size()-1`.
    void decode_codec_batch(const std::vector<int>& slots, const std::vector<int>& ring_starts,
                            int stride, cudaStream_t stream) {
        const int n = static_cast<int>(slots.size());
        if (n <= 0 || stride <= 0) return;

        const bool run_graph = use_graph && !prof.enabled && codec->loaded();
        const int M = run_graph ? pick_graph_bucket(n, max_slots) : n;
        const int bi = bucket_index(M);
        const int si = stride_index(stride);

        if (run_graph && bi >= 0 && si >= 0) {
            ensure_codec_graph(M, stride, stream);
        }

        for (int m = 0; m < n; ++m) {
            h_codec_slots[m] = slots[size_t(m)];
            h_codec_start[m] = ring_starts[size_t(m)];
        }
        for (int m = n; m < M; ++m) {
            h_codec_slots[m] = pad_slot;
            h_codec_start[m] = 0;
        }
        TTS_CK(cudaMemcpyAsync(d_codec_slots, h_codec_slots, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_codec_start, h_codec_start, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        if (prof.enabled) cudaEventRecord(ev_p0, stream);
        if (run_graph && bi >= 0 && si >= 0 && codec_exec[size_t(bi)][size_t(si)] != nullptr) {
            TTS_CK(cudaGraphLaunch(codec_exec[size_t(bi)][size_t(si)], stream));
        } else {
            record_codec_ops(M, stride, stream);
        }
        TTS_CK(cudaEventRecord(ev_codec, stream));
        TTS_CK(cudaEventSynchronize(ev_codec));
        if (prof.enabled) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_p0, ev_codec);
            prof.codec_ms += double(ms);
        }
    }
};

} // namespace g6tts
