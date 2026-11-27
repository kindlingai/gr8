// Tencent Hy4-flash geometry (-DG6_MODEL_HY4=1), included by model_geom.h.
// engine/tests/test_hy4_geom.cpp and test_hy4_geom.py check every constant below against
// config.json (text_config) and all 1,964 checkpoint tensor shapes.
// Supports TP widths nr in {1, 2} (and cleanly divides for nr=4):
//   DSA attention (12 layers: 0,1,5,9,...,41): 32 heads (TP=1: 32, TP=2: 16), MLA latent 512+64, compressed indexer 16x128
//   SWA attention (30 layers):                 64 heads (TP=1: 64, TP=2: 32), MLA latent 512+64, window 512
//   Dense MLP (layer 0):                       I=13824 (108 x 128-blocks; TP=1: 13824, TP=2: 6912)
//   MoE routed (layers 1..41):                 512 experts, top-16, latent EH=1536, I=1536 (12 x 128-blocks, pmcgx)
//   MoE shared (layers 1..41):                 1 expert, H=3072, I=1536 (12 x 128-blocks, pmcgx)
//   Engram (layers 1, 11, 21, 31):             n-gram 2..3, 8 heads/order (16 heads), head_dim=96, FP8-E4M3 128B rowstore
//   mHC (4 streams, non-combined iHC):         fn [8, 12288], base [8], scale [2], magnitude 2.0, eps 1e-6; hc_head [4, 12288]
//   Vocab:                                     120832 = 944 tiles of 128 (TP=1: 944, TP=2: 472)
#pragma once

namespace geom
{
// --- trunk ---
constexpr int HID = 3072;               // [cfg] hidden_size
constexpr int VOCAB_RAW = 120832;       // [cfg] vocab_size
constexpr int VOCAB = 120832;           // [der] multiple of 128 (944 tiles)
constexpr int VTILES = VOCAB / 128;     // [der] 944
constexpr int KSH = HID / 16;           // [der] 192
constexpr int HB = HID / 128;           // [der] 24
constexpr int L_MAIN = 42;              // [cfg] num_hidden_layers (0..41)
constexpr int N_NEXTN = 0;              // [cfg] text-first boot (dfly deferred)
constexpr int MTP_LAYER = L_MAIN;       // [der] 42 (no MTP layers active)
#define G6_MTP_L "L042"
constexpr int NLAYERS = L_MAIN;         // [der] 42
constexpr float RMS_EPS = 1e-5f;        // [cfg] rms_norm_eps
constexpr int MAX_POS = 1048576;        // [cfg] max_position_embeddings (1M)
constexpr int KV_PAGE = 64;             // tokens per DSA paged KV block (16 indexer groups of 4)
constexpr int DENSE_I = 13824;          // [cfg] intermediate_size (layer 0 dense MLP)
constexpr int DENSE_BLOCKS = DENSE_I / 128; // [der] 108
constexpr int FIRST_MOE = 1;            // [cfg] first_k_dense_replace
constexpr int DFL_L = 5, DFL_NT = 5, DFL_NH = 32; // shared headers compile cleanly

// --- hyper-connections (non-combined iHC, 4 streams, FP32) ---
constexpr int HC = 4;                   // [cfg] hc_mult
constexpr int HC_D = HC * HID;          // [der] 12288
constexpr int HC_MIX = 2 * HC;          // [der] 8 (pre 4 + post 4, no comb)
constexpr float HC_EPS = 1e-6f;         // [cfg] hc_eps
constexpr float HC_MAGNITUDE = 2.0f;    // [cfg] hc_magnitude

// --- gated MLA attention (dual shape: 12 DSA layers + 30 SWA layers) ---
constexpr int HEADS_DSA = 32;           // [cfg] num_attention_heads (DSA layers)
constexpr int HEADS_SWA = 64;           // [cfg] swa_num_attention_heads (SWA layers)
constexpr int HEADS = HEADS_SWA;        // [der] max heads across layers (64, for g6nr.h)
constexpr int N_DSA = 12;               // [cfg] count of deepseek_sparse_attention layers
constexpr int N_SWA = 30;               // [cfg] count of sliding_attention layers
constexpr int SWA_WINDOW = 512;         // [cfg] sliding_window (511 preceding + current)
constexpr int SWA_RING   = 2048;        // power-of-2 ring >= SWA_WINDOW + max prefill chunk (collision-free!)
constexpr int QA = 1024;                // [cfg] q_lora_rank
constexpr int KV_LORA = 512;            // [cfg] kv_lora_rank
constexpr int QK_NOPE = 192;            // [cfg] qk_nope_head_dim (and swa_qk_nope_head_dim)
constexpr int QK_ROPE = 64;             // [cfg] qk_rope_head_dim (and swa_qk_rope_head_dim)
constexpr int QK_HEAD = QK_NOPE + QK_ROPE; // [der] 256
constexpr int V_HEAD = 256;             // [cfg] v_head_dim (and swa_v_head_dim)
constexpr int KVB_HEAD = QK_NOPE + V_HEAD; // [der] 448 (kv_b_proj rows per head)
constexpr int KVA = KV_LORA + QK_ROPE;  // [der] 576 (kv_a_proj_with_mqa output & latent cache width)
constexpr float ROPE_THETA_SWA = 10000.0f;    // [cfg] rope_parameters.sliding_attention.rope_theta
constexpr float ROPE_THETA_DSA = 10000000.0f; // [cfg] rope_parameters.deepseek_sparse_attention.rope_theta
constexpr int MROPE_SEC0 = 12, MROPE_SEC1 = 10, MROPE_SEC2 = 10; // [cfg] mrope_section (32 pairs = 64 dims)

// --- compressed group indexer (DSA layers only) ---
constexpr int IDX_HEADS = 16;           // [cfg] compressed_indexer_num_attention_heads
constexpr int IDX_DIM = 128;            // [cfg] compressed_indexer_head_dim
constexpr int IDX_RATIO = 4;            // [cfg] compressed_indexer_compress_ratio
constexpr int IDX_GROUP_TOPK = 512;     // [cfg] compressed_indexer_group_topk
constexpr int IDX_TOPK = 2048;          // [cfg] compressed_indexer_candidate_topk (= 512 * 4)
constexpr int IDX_ROPE_DIM = QK_ROPE;   // [der] 64 (trailing 64 of 128 dims rotated)
constexpr int IDX_GROUPS_PER_PAGE = KV_PAGE / IDX_RATIO; // [der] 16

// --- latent MoE (layers 1..41) ---
constexpr int NEXP = 512;               // [cfg] n_routed_experts
constexpr int TOPK = 16;                // [cfg] num_experts_per_tok
constexpr int EH = 1536;                // [cfg] routed_expert_hidden_size (latent dim)
constexpr int EH_B = EH / 128;
constexpr int EHB = EH_B;          // [der] 12 (128-col blocks over routed expert hidden)
constexpr int EH_KSH = EH / 16;         // [der] 96
constexpr int EXP_I = 1536;             // [cfg] moe_intermediate_size
constexpr int EXP_BLOCKS = EXP_I / 128; // [der] 12 (128-col blocks per routed expert)
constexpr int SHARED_I = 1536;          // [cfg] moe_intermediate_size * n_shared_experts (in/out = HID 3072)
constexpr int SHARED_BLOCKS = SHARED_I / 128; // [der] 12
constexpr float ROUTE_SCALE = 3.998f;
constexpr float ROUTED_SCALE = ROUTE_SCALE;   // [cfg] routed_scaling_factor
constexpr float SWIGLU_LIMIT = 10.0f;   // [cfg] swiglu_limit (routed experts only)

// --- Engram n-gram memory (decoder layers 1, 11, 21, 31; 1-indexed engram_layer_ids = [2,12,22,32]) ---
constexpr int N_ENGRAM = 4;
constexpr int ENGRAM_MAX_NGRAM = 3;     // [cfg] engram_max_ngram_size (bigrams + trigrams)
constexpr int ENGRAM_HEADS_PER_NGRAM = 8; // [cfg] engram_n_head_per_ngram
constexpr int ENGRAM_IDS = (ENGRAM_MAX_NGRAM - 1) * ENGRAM_HEADS_PER_NGRAM;
constexpr int ENGRAM_HEADS = ENGRAM_IDS; // [der] 16
constexpr int ENGRAM_HEAD_DIM = 96;     // [cfg] engram_n_embed_per_ngram (768) / 8
constexpr int ENGRAM_DIM = ENGRAM_IDS * ENGRAM_HEAD_DIM; // [der] 1536
constexpr int ENGRAM_LOOKUP_SIZE = 120818; // [shape] lookup_table.shape[0]
constexpr int ENGRAM_CVOCAB = 92620;    // [shape] lookup_table.max() + 1
constexpr int ENGRAM_PAD_ID = 120002;   // [cfg] engram_pad_id
constexpr int ENGRAM_SEED = 0;          // [cfg] engram_seed
constexpr int ENGRAM_ROW_BYTES = 128;   // 96 E4M3 + 2B BF16 scale + 30B zero pad (32 rows / 4 KiB page)
constexpr int ENGRAM_NGRAM_CTX = ENGRAM_MAX_NGRAM - 1; // [der] 2 prior token ids

// --- tokens ---
constexpr int TOK_BOS = 120000;         // [cfg] bos_token_id
constexpr int TOK_PAD = 120002;         // [cfg] pad_token_id
constexpr int TOK_EOS = 120025;         // [cfg] eos_token_id

// --- per-layer predicates ---
constexpr bool is_moe(int l) { return l >= FIRST_MOE && l < L_MAIN; }
constexpr bool is_dense(int l) { return l >= 0 && l < FIRST_MOE; }
constexpr bool is_dsa(int l) { return l == 0 || (l >= 1 && l < L_MAIN && ((l - 1) % 4 == 0)); }
constexpr bool is_dsa_layer(int l) { return is_dsa(l); }
constexpr bool is_swa(int l) { return l >= 0 && l < L_MAIN && !is_dsa(l); }
constexpr bool full_index(int l) { return is_dsa(l); }
constexpr int layer_heads(int l) { return is_dsa(l) ? HEADS_DSA : HEADS_SWA; }
constexpr int dsa_slot(int l) { return l == 0 ? 0 : (1 + (l - 1) / 4); } // 0..11
constexpr int swa_slot(int l) { return l - (l == 0 ? 0 : (1 + (l - 1) / 4)) - 1; } // 0..29
constexpr bool is_engram(int l) { return l == 1 || l == 11 || l == 21 || l == 31; }
constexpr int engram_slot(int l) { return l == 1 ? 0 : (l == 11 ? 1 : (l == 21 ? 2 : (l == 31 ? 3 : -1))); }
constexpr int engram_layer_id(int slot) { return slot == 0 ? 2 : (slot == 1 ? 12 : (slot == 2 ? 22 : 32)); }

// --- per-rank sharding for nr in {1, 2} (also exact for 4) ---
constexpr bool rank_count_ok(int nr) { return nr == 1 || nr == 2 || nr == 4; }
constexpr int nh_layer(int l, int nr) { return layer_heads(l) / nr; }
constexpr int h0_layer(int l, int nr, int r) { return r * nh_layer(l, nr); }
constexpr int dense_i_local(int nr) { return DENSE_I / nr; }
constexpr int exp_blocks_local(int nr) { return EXP_BLOCKS / nr; }       // 12 at TP=1, 6 at TP=2
constexpr int shared_blocks_local(int nr) { return SHARED_BLOCKS / nr; } // 12 at TP=1, 6 at TP=2
constexpr int vocab_tiles(int nr, int r) { return VTILES / nr + (r < VTILES % nr ? 1 : 0); }
constexpr int vocab_tile0(int nr, int r) { return r * (VTILES / nr) + (r < VTILES % nr ? r : VTILES % nr); }
constexpr int vocab_rows(int nr, int r) { return vocab_tiles(nr, r) * 128; }
constexpr int vocab_row0(int nr, int r) { return vocab_tile0(nr, r) * 128; }

// --- compile-time checks ---
constexpr int count_dsa() { int n = 0; for (int l = 0; l < L_MAIN; ++l) n += is_dsa(l); return n; }
constexpr int count_swa() { int n = 0; for (int l = 0; l < L_MAIN; ++l) n += is_swa(l); return n; }
constexpr int count_eng() { int n = 0; for (int l = 0; l < L_MAIN; ++l) n += is_engram(l); return n; }
constexpr bool check_swa_slots() {
    int s = 0;
    for (int l = 0; l < L_MAIN; ++l) if (is_swa(l)) { if (swa_slot(l) != s) return false; ++s; }
    return s == N_SWA;
}
constexpr bool check_dsa_slots() {
    int s = 0;
    for (int l = 0; l < L_MAIN; ++l) if (is_dsa(l)) { if (dsa_slot(l) != s) return false; ++s; }
    return s == N_DSA;
}
static_assert(count_dsa() == N_DSA && count_swa() == N_SWA && N_DSA + N_SWA == L_MAIN, "12 DSA + 30 SWA = 42");
static_assert(check_dsa_slots() && check_swa_slots(), "DSA/SWA slot bijection");
static_assert(count_eng() == N_ENGRAM && engram_slot(1) == 0 && engram_slot(31) == 3, "4 Engram layers");
static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EH % 128 == 0 && EXP_I % 128 == 0 && DENSE_I % 128 == 0, "128-wide tiles");
static_assert(MROPE_SEC0 + MROPE_SEC1 + MROPE_SEC2 == QK_ROPE / 2, "mRoPE sections sum to 32 pairs");
static_assert(KV_PAGE % IDX_RATIO == 0 && IDX_TOPK == IDX_GROUP_TOPK * IDX_RATIO, "indexer ratio alignment");
static_assert(HEADS_DSA % 2 == 0 && HEADS_SWA % 2 == 0 && EXP_BLOCKS % 2 == 0 && VTILES % 2 == 0, "TP=2 exact splits");
} // namespace geom
