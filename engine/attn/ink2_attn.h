// Inkling-2-Small trunk attention: the shapes, weights and state of one layer on one rank. Plain C++ (no CUDA); the
// kernels are in ink2_attn.cuh.
//
// A rank owns nh_q query heads q0 .. q0 + nh_q and the nh_kv K / V heads kv0 .. kv0 + nh_kv they read (query head h
// reads K / V head h / gqa). Every tensor below is the rank's slice, row major, BF16 cells:
//   wq, gate   [nh_q * hd][H]        rows of the rank's query heads
//   wr         [nh_q * rel_dim][H]
//   sink       [nh_q][H]
//   wk, wv     [nh_kv * hd][H]       rows of the rank's K / V heads
//   wo         [H][nh_q * hd]        the rank's columns, repacked
//   q_clip     [nh_q][hd]            rel_clip [nh_q][rel_dim]            rel_proj [rel_dim][rel_win]
//   local layer:   k_sconv, v_sconv [nh_kv * hd][4]
//   global layer:  k_static, v_static [4][nh_kv * hd] (the rank's columns, repacked), k_wproj, v_wproj [nh_kv * 4][H]
//
// State of one layer (nk = nh_kv * hd), all BF16:
//   hist_k, hist_v   [slots][3][nk]             the K / V convolution inputs, entry position % 3
//   local layer:   ring  [slots][2][window][nk] K (0) and V (1) after the head norms, entry position % window
//   global layer:  pages [pages][2][64][nk]     K and V of 64 positions; page_table [slots][max_pps] = the page of
//                                               positions 64 i .. 64 i + 63 of a slot
#pragma once
#include <cstdint>

namespace ink2
{
constexpr int KV_PAGE_TOKENS = 64;    // positions per K / V page of a global layer
constexpr int ATTN_SPAN = 256;        // keys one CTA of the attention core scores; more keys split over CTAs
constexpr int ATTN_MAX_HD = 128, ATTN_MAX_REL = 16;

struct AttnGeom
{
    int H = 0;
    int hd = 0;                // head_dim: a multiple of 16, <= 128
    int rel_dim = 0;           // <= 16
    int rel_win = 0;           // relative-position logits for distances 0 .. rel_win - 1
    int gqa = 0;               // query heads per K / V head
    int nh_q = 0, q0 = 0;      // the rank's query heads
    int nh_kv = 0, kv0 = 0;    // the rank's K / V heads
    int window = 0;            // local layer: q_pos - k_pos < window
    int log_scale_base = 0;    // global layer: tau(pos) = 1 + 0.1 ln(max(1, (pos + 1) / base))
    bool global = false;
};

struct AttnWeights
{
    const uint16_t *wq = nullptr, *wr = nullptr, *gate = nullptr, *sink = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const uint16_t *q_clip = nullptr, *rel_clip = nullptr, *rel_proj = nullptr;
    const uint16_t *k_sconv = nullptr, *v_sconv = nullptr;                                                   // local
    const uint16_t *k_static = nullptr, *v_static = nullptr, *k_wproj = nullptr, *v_wproj = nullptr;         // global
};

struct AttnState
{
    uint16_t *hist_k = nullptr, *hist_v = nullptr;
    uint16_t* ring = nullptr;             // local
    uint16_t* pages = nullptr;            // global
    const int* page_table = nullptr;
    int max_pps = 0;                      // pages per slot in page_table
};
}   // namespace ink2
