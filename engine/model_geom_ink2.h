// Inkling-2-Small geometry (-DG6_MODEL_INK2=1), included by model_geom.h. engine/tests/test_ink2_geom.cpp checks every
// value below, and reads them against the release's config.json when GR8_TEST_INK2_METADATA_DIR names its directory.
//   [cfg]   the config.json field of the same meaning (name in the comment)
//   [shape] read off the checkpoint's tensor shapes (engine/loader-rs/src/ink2.rs)
//   [der]   derived from [cfg] or [shape] values, as the comment says
// The names shared code already uses (HID, VOCAB, VTILES, KSH, HB, L_MAIN, N_NEXTN, MTP_LAYER, NLAYERS, RMS_EPS, HEADS,
// WINDOW, MAX_POS, NEXP, TOPK, EXP_I, EXP_BLOCKS, DENSE_I, ROUTE_SCALE, is_moe, DFL_*) keep their meaning; the names of
// another model's own blocks are not here.
#pragma once

namespace geom
{
// --- trunk ---
constexpr int HID = 4608;            // [cfg] hidden_size
constexpr int VOCAB_RAW = 201024;    // [cfg] vocab_size: the rows of embed and lm_head in the checkpoint
constexpr int VALID_VOCAB = 200058;  // [cfg] the ids a sampler may return; lm_head rows from here on give -inf
constexpr int VOCAB = 201216;        // [der] VOCAB_RAW rounded up to 128 * nr for nr = 2, 3, 4, 6, 12: 1572 tiles of 128
constexpr int VTILES = VOCAB / 128;  // [der] lm_head 128-row tiles: 1572
constexpr int KSH = HID / 16;        // [der] k16 steps over hidden: 288
constexpr int HB = HID / 128;        // [der] 128-column blocks over hidden: 36
constexpr int L_MAIN = 48;           // [cfg] num_hidden_layers: layers 0..47
constexpr int N_NEXTN = 5;           // [shape] the drafter layers of mtp.safetensors
constexpr int MTP_LAYER = L_MAIN;    // [der] pack index of the first drafter layer (48)
#define G6_MTP_L "L048"             // pack tensor prefix of the first drafter layer (string literal, for concatenation)
constexpr int NLAYERS = L_MAIN + N_NEXTN;   // [der] 53
constexpr int ATTN_RES_BLOCK = 6;    // [cfg] layers per block of the depth attention (AttnRes)
constexpr int ATTN_RES_DEPTH = L_MAIN / ATTN_RES_BLOCK + 1;   // [der] 9: the most rows a depth attention reads
constexpr float RMS_EPS = 1e-6f;     // [cfg] every RMSNorm
constexpr int DENSE_I = 0;           // [shape] no dense MLP layer: all 48 layers are MoE
constexpr bool is_moe(int l) { return l >= 0 && l < L_MAIN; }
constexpr int DFL_L = 5, DFL_NT = 5, DFL_NH = 32;   // runtime/dflash.cuh compiles in every build; no DFlash2 drafter here

// --- attention: 48 query heads over 8 K / V heads; 40 sliding-window layers and 8 full-attention layers ---
constexpr int HEADS = 48;            // [cfg] num_attention_heads
constexpr int KV_HEADS = 8;          // [cfg] num_key_value_heads
constexpr int GQA = HEADS / KV_HEADS;   // [der] 6 query heads per K / V head
constexpr int HEAD_DIM = 128;        // [cfg] head_dim
constexpr int REL_DIM = 16;          // [shape] wr.weight [48 * 16, 4608]
constexpr int REL_WIN = 1024;        // [shape] rel_logits_proj [16, 1024]
constexpr int WINDOW = 1024;         // [cfg] sliding_window of a local layer: q_pos - k_pos < 1024
constexpr int LOG_SCALE_BASE = 128000;   // [cfg] full-attention layers: tau(pos) = 1 + 0.1 ln(max(1, (pos + 1) / base))
constexpr int MAX_POS = 1048576;     // [cfg] max_position_embeddings
constexpr int KV_PAGE = 64;          // tokens per K / V page of a full-attention layer
constexpr int SCONV_K = 4;           // taps of every short convolution; its state keeps SCONV_K - 1 = 3 tokens

// --- MoE: 384 routed experts top-8, softcapped SwiGLU ---
constexpr int NEXP = 384;            // [cfg] n_routed_experts
constexpr int TOPK = 8;              // [cfg] num_experts_per_tok
constexpr int EXP_I = 2304;          // [cfg] moe_intermediate_size
constexpr int EXP_BLOCKS = EXP_I / 128;   // [der] 18
constexpr float ROUTE_SCALE = 8.0f;  // [cfg] route weights sum to 8
constexpr float LOGIT_SOFTCAP = 30.0f;    // logits = 30 tanh(z / 30)
constexpr float SWIGLU_GATE_CAP = 4.0f;   // gate -> 4 tanh(gate / 4)
constexpr float SWIGLU_UP_CAP = 25.0f;    // up -> 25 tanh(up / 25)

// --- tokens ---
constexpr int TOK_BOS = 199998;
constexpr int TOK_EOT = 199999;      // <|end_of_text|>
constexpr int TOK_PAD = 200000;
constexpr int TOK_EOS = 200006;      // <|content_model_end_sampling|>
constexpr int TOK_THINK_START = 200008, TOK_THINK_END = 200010;
constexpr int TOK_IMAGE = 200054;    // <|unused_200054|>: every row of an image span

// --- per-layer predicates ---
// full attention (paged K / V, dynamic K / V convolution, 3 depth-attention queries) on the last layer of each block
constexpr bool is_global(int l) { return l >= 0 && l < L_MAIN && l % 6 == 5; }
constexpr int global_slot(int l) { return l / 6; }        // index of full-attention layer l among them: 0..7
constexpr int local_slot(int l) { return l - l / 6; }     // index of sliding-window layer l among them: 0..39
constexpr bool is_mxfp8_moe(int l) { return l >= 40 && l < L_MAIN; }   // experts: layers 0..39 NVFP4, 40..47 MXFP8
constexpr bool is_block_start(int l) { return l % ATTN_RES_BLOCK == 0; }   // the layer input joins the depth rows
constexpr int N_GLOBAL = 8, N_LOCAL = 40, N_MXFP8 = 8;

// --- per-rank slices, nr ranks, rank r (head_ranges and vocab_split of engine/loader-rs/src/ink2.rs) ---
constexpr bool rank_count_ok(int nr) { return nr == 1 || nr == 2 || nr == 3 || nr == 4 || nr == 6 || nr == 12; }
constexpr int nh_q(int nr) { return HEADS / nr; }                  // query heads of a rank
constexpr int q0(int nr, int r) { return r * nh_q(nr); }           // its first query head
constexpr int kv0(int nr, int r) { return q0(nr, r) / GQA; }       // the K / V heads its query heads read: kv0 .. kv1
constexpr int kv1(int nr, int r) { return (q0(nr, r) + nh_q(nr) - 1) / GQA + 1; }
constexpr int nh_kv(int nr, int r) { return kv1(nr, r) - kv0(nr, r); }
constexpr int local_kv_head(int nr, int r, int lq) { return (q0(nr, r) + lq) / GQA - kv0(nr, r); }   // of local query head lq
constexpr int inter_slice(int nr) { return EXP_I / nr; }           // expert intermediate columns of a rank
constexpr int inter0(int nr, int r) { return r * inter_slice(nr); }
constexpr int vocab_padded(int nr) { return (VOCAB_RAW + 128 * nr - 1) / (128 * nr) * (128 * nr); }
constexpr int vocab_slice(int nr) { return vocab_padded(nr) / nr; }   // lm_head rows of a rank
constexpr int vocab0(int nr, int r) { return r * vocab_slice(nr); }

// --- compile-time checks ---
constexpr int count_global() { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += is_global(l); return n; }
constexpr int count_mxfp8() { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += is_mxfp8_moe(l); return n; }
constexpr bool kv_heads_all(int nr, int want) { for (int r = 0; r < nr; ++r) if (nh_kv(nr, r) != want) return false; return true; }
static_assert(NLAYERS == 53 && MTP_LAYER == 48, "48 trunk layers + 5 drafter layers");
static_assert(count_global() == N_GLOBAL && L_MAIN - count_global() == N_LOCAL && count_mxfp8() == N_MXFP8, "8 full-attention layers, 40 local, 8 MXFP8");
static_assert(is_global(5) && is_global(47) && !is_global(0) && !is_global(48), "full attention on 5, 11, .. 47");
static_assert(global_slot(47) == N_GLOBAL - 1 && local_slot(46) == N_LOCAL - 1 && local_slot(6) == 5, "slots 0..7 and 0..39");
static_assert(GQA == 6 && HEADS * HEAD_DIM == 6144 && KV_HEADS * HEAD_DIM == 1024, "wq [6144, 4608], wk [1024, 4608]");
static_assert(kv_heads_all(6, 2) && kv_heads_all(4, 2) && kv_heads_all(1, 8), "K / V heads per rank");
static_assert(vocab_padded(2) == VOCAB && vocab_padded(3) == VOCAB && vocab_padded(4) == VOCAB && vocab_padded(6) == VOCAB && vocab_padded(12) == VOCAB,
              "one padded vocabulary for 2, 3, 4, 6 and 12 ranks");
static_assert(vocab_padded(1) == 201088 && VOCAB >= VOCAB_RAW && VALID_VOCAB <= VOCAB_RAW, "vocabulary");
static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EXP_I % 128 == 0 && HEAD_DIM % 128 == 0, "128-wide tiles");
static_assert(inter_slice(6) % 64 == 0 && inter_slice(4) % 64 == 0 && inter_slice(12) % 64 == 0, "an NVFP4 group of 64 inside a rank's slice");
static_assert(WINDOW % KV_PAGE == 0 && ATTN_RES_DEPTH == 9, "window in pages; 8 finished blocks + the partial row");
}
