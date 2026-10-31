// DeepSeek-V4.1-Flash-1017 geometry (-DG6_MODEL_DS41=1), included by model_geom.h. docs/DEEPSEEK.md is the card; every
// value below is checked against engine/ref/ds41/config.json (the release's config, sha256 8be45ce0...) by
// engine/tests/test_ds41_geom.py, and the tensor shapes by the static_asserts at the end (tensor map in the card).
//   [cfg]  the config.json field of the same meaning (name in the comment)
//   [shape] read off the checkpoint's safetensors headers (96,085 tensors; docs/DEEPSEEK.md section 3)
//   [der]  derived from [cfg] or [shape] values, as the comment says
// The names GLM and Flash code already use (HID, QA, NEXP, TOPK, HEADS, VOCAB, HB, KSH, VTILES, L_MAIN, MTP_LAYER,
// NLAYERS, HC*, SWIGLU_LIMIT, RMS_EPS, ROUTE_SCALE, EXP_I, EXP_BLOCKS, IDX_*, full_index, is_moe) keep their meaning, so
// shared kernels compile; DeepSeek-only names are new. Nothing here is the GLM or Flash value.
#pragma once

namespace geom
{
// --- trunk ---
constexpr int HID = 5120;            // [cfg] hidden_size
constexpr int VOCAB = 129280;        // [cfg] vocab_size
constexpr int VTILES = VOCAB / 128;  // [der] lm_head 128-row tiles: 1010
constexpr int KSH = HID / 16;        // [der] k16 steps over hidden: 320
constexpr int HB = HID / 128;        // [der] 128-column blocks over hidden: 40
constexpr int L_MAIN = 40;           // [cfg] num_hidden_layers: layers 0..39
constexpr int N_NEXTN = 3;           // [cfg] num_nextn_predict_layers: the DSpark layers mtp.0..2
constexpr int MTP_LAYER = L_MAIN;    // [der] pack index of the first DSpark layer (40); DSpark layer j is MTP_LAYER + j
#define G6_MTP_L "L040"             // pack tensor prefix of the first DSpark layer (string literal, for concatenation)
constexpr int NLAYERS = L_MAIN + N_NEXTN;   // [der] pack layers incl. DSpark: 43 (= len(compress_ratios))
constexpr float RMS_EPS = 1e-20f;    // [cfg] rms_norm_eps (not 1e-5)
constexpr float RMS_EPS_FLASH = RMS_EPS;    // the name engine/mhc uses for the mHC flat norm
constexpr int DENSE_I = 0;           // [shape] no dense MLP layer: all 40 layers are MoE
constexpr bool is_moe(int l) { return l >= 0 && l < NLAYERS; }

// --- attention: one shared KV head of 512 (K == V), 64 query heads, grouped low-rank output ---
constexpr int HEADS = 64;            // [cfg] num_attention_heads
constexpr int HEAD_DIM = 512;        // [cfg] head_dim: q/k/v width of every head (num_key_value_heads = 1)
constexpr int QK_ROPE = 64;          // [cfg] qk_rope_head_dim: the last 64 dims of a head carry RoPE
constexpr int QK_NOPE = HEAD_DIM - QK_ROPE;   // [der] 448
constexpr int KV_LORA = HEAD_DIM - QK_ROPE;   // [der] 448; DeepInfer names it kv_lora_rank
constexpr int KVA = HEAD_DIM;        // [shape] wkv.weight [512, 5120]: the one KV row per token, before compression
constexpr int V_HEAD = HEAD_DIM;     // [der] shared K/V: the value is the key
constexpr int QA = 1280;             // [cfg] q_lora_rank (wq_a [1280, 5120])
constexpr int Q_OUT = HEADS * HEAD_DIM;       // [shape] wq_b.weight [32768, 1280]
constexpr int O_LORA = 1024;         // [cfg] o_lora_rank
constexpr int O_GROUPS = 8;          // [cfg] o_groups
constexpr int O_GROUP_IN = HEADS / O_GROUPS * HEAD_DIM;   // [der] 4096: one group's attention output
constexpr int WINDOW = 128;          // [cfg] sliding_window: every layer attends to the last 128 tokens uncompressed
constexpr float ROPE_THETA = 10000.0f;           // [cfg] rope_theta
constexpr float COMPRESS_ROPE_THETA = 160000.0f; // [cfg] compress_rope_theta (compressed rows)
constexpr int MAX_POS = 1048576;     // [cfg] max_position_embeddings
// YaRN (rope_scaling; DeepInfer mscale = false)
constexpr int YARN_ORIG_POS = 65536; // [cfg] original_max_position_embeddings
constexpr int YARN_FACTOR = 16;      // [cfg] factor
constexpr int YARN_BETA_FAST = 32, YARN_BETA_SLOW = 1;   // [cfg]

// --- per-layer compression and the cross-layer sources ---
// compress_ratios (43 entries): layers 0,1 window only (0); 2..19 ratio 2 (18 layers); 20..39 ratio 1 (20 layers);
// the 3 DSpark layers window only (0). A ratio-2 layer makes one compressed row per 2 tokens, a ratio-1 layer one per token.
constexpr int compress_ratio(int l) { return l < 2 ? 0 : l < 20 ? 2 : l < L_MAIN ? 1 : 0; }
// kv_source_layer_ids [2, 8, 14, 20]: only these four layers own a compressor; layer l reads the nearest source <= l
// (layers 0, 1 and the DSpark layers have none: -1). Layers 3..7 read layer 2, 20+ read layer 20.
constexpr int kv_source(int l) { return l >= L_MAIN ? -1 : l >= 20 ? 20 : l >= 14 ? 14 : l >= 8 ? 8 : l >= 2 ? 2 : -1; }
constexpr bool is_kv_source(int l) { return l == 2 || l == 8 || l == 14 || l == 20; }
// index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]: the layers with their own indexer q projection and weights
constexpr int index_source(int l)
{
    return l >= L_MAIN ? -1 : l >= 36 ? 36 : l >= 32 ? 32 : l >= 28 ? 28 : l >= 24 ? 24 : l >= 20 ? 20 : l >= 14 ? 14
         : l >= 8 ? 8 : l >= 2 ? 2 : -1;
}
constexpr bool is_index_source(int l) { return l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 || l == 32 || l == 36; }
// the layers that run their own indexer (the binder rule of attn/bind.cu): the eight index sources
constexpr bool full_index(int l) { return is_index_source(l); }
// ratio-2 layers 2, 8, 14 own compressor.wgate (overlap gate); layer 20 (ratio 1) owns only compressor.wkv + norm
constexpr bool has_compressor_gate(int l) { return l == 2 || l == 8 || l == 14; }
constexpr int N_KV_SRC = 4, N_IDX_SRC = 8;
// DSA indexer
constexpr int IDX_HEADS = 32;        // [cfg] index_n_heads
constexpr int IDX_DIM = 128;         // [cfg] index_head_dim
constexpr int IDX_TOPK = 512;        // [cfg] index_topk
constexpr int IDX_WQB_OUT = IDX_HEADS * IDX_DIM;   // [shape] indexer.wq_b.weight [4096, 1280]
// candidate blocks: layers >= 20 select blocks of 8 tokens out of the candidate source's (layer 20) rows
constexpr int CAND_SOURCE = 20;      // [cfg] candidate_source_layer_id
constexpr int CAND_TOPK_BLOCKS = 2048;   // [cfg] candidate_topk_blocks
constexpr int CAND_BLOCK = 8;        // [cfg] candidate_block_size
constexpr int N_PREFILL_LAYER = 20;  // [der] DeepInfer n_prefill_layer = index of the first ratio-1 layer: layers >= 20 are consumers
// Bounded replay (YOCO prefill truncation): past the last kv source no layer makes state
// a later row reads for every position, so a prefill chunk runs layers >= LATE_LAYER_START over its last TAIL_ROWS rows only (G6_DS_PF_TAIL).
constexpr int LATE_LAYER_START = 20 + 1;   // [der] max(kv_source_layer_ids) + 1 (SGLang late_layer_start)
constexpr int TAIL_ROWS = WINDOW;          // [der] default tail: the window a decode step reaches back

// --- MoE: 384 routed experts top-6 + 1 shared, SwiGLU limit, sqrt-softplus noaux_tc router ---
constexpr int NEXP = 384;            // [cfg] n_routed_experts
constexpr int TOPK = 6;              // [cfg] num_experts_per_tok
constexpr int EXP_I = 2304;          // [cfg] moe_intermediate_size (routed and shared expert width)
constexpr int EXP_BLOCKS = EXP_I / 128;   // [der] 128-column blocks per routed expert: 18
constexpr int SHARED_I = 2304;       // [shape] shared_experts.w1 [2304, 5120]; n_shared_experts = 1
constexpr float ROUTE_SCALE = 1.5f;  // [cfg] routed_scaling_factor; scoring sqrtsoftplus, noaux_tc bias, norm_topk_prob true
constexpr float SWIGLU_LIMIT = 10.0f;    // [cfg] swiglu_limit (routed, shared, DSpark experts)
// quantization (quantization_config): fp8 E4M3 with UE8M0 scales per 32x32 block; routed experts MXFP4 (I8 = 2 codes)
constexpr int FP8_BLOCK = 32;        // [cfg] weight_block_size [32, 32]
constexpr int FP4_GROUP = 32;        // [shape] experts.w1.scale [2304, 160] = 5120 / 32 along K

// --- hyper-connections (mHC): 4 streams. the shifted variant, no hc_head tensor (engine/mhc's Flash block is the unshifted one) ---
constexpr int HC = 4;                // [cfg] hc_mult
constexpr int HC_MIX = (2 + HC) * HC;    // [der] 24: pre 4, post 4, comb 16 (hc_attn_base [24])
constexpr int HC_FLAT = HC * HID;    // [shape] hc_attn_fn [24, 20480]
constexpr int HC_ITERS = 20;         // [cfg] hc_sinkhorn_iters
constexpr float HC_EPS = 1e-6f;      // [cfg] hc_eps

// --- Engram (layers 1 and 14): hashed n-gram embedding rows, read from NVMe ---
constexpr int ENGRAM_L0 = 1, ENGRAM_L1 = 14;   // [cfg] engram_layer_ids
constexpr bool is_engram(int l) { return l == ENGRAM_L0 || l == ENGRAM_L1; }
constexpr int NGRAM_MAX = 4;         // [cfg] engram_max_ngram_size: tables for n = 2, 3, 4
constexpr int NGRAM_TABLES = NGRAM_MAX - 1;   // [der] 3 (DeepInfer n_ngram_tables)
constexpr int ENGRAM_HEADS = 8;      // [cfg] engram_n_heads per n-gram table
constexpr int ENGRAM_HEAD_DIM = 256; // [cfg] engram_head_dim
constexpr int ENGRAM_DIM = ENGRAM_HEADS * ENGRAM_HEAD_DIM;   // [der] 2048 per table; 3 tables = 6144 = wkv columns
constexpr int ENGRAM_VOCAB = 16000000;   // [cfg] engram_vocab_size (hash modulus basis per table)
constexpr int ENGRAM_CVOCAB = 99092; // [cfg] engram_compressed_vocab_size (tokenizer compression)
constexpr int ENGRAM_PAD = 2;        // [cfg] engram_pad_token_id
constexpr long long ENGRAM_ROWS_L0 = 384006168, ENGRAM_ROWS_L1 = 384016682;   // [cfg] engram_num_embeddings: 24 head tables each
constexpr int ENGRAM_ROW_BYTES = ENGRAM_HEAD_DIM + ENGRAM_HEAD_DIM / FP4_GROUP;   // [shape] embed [rows,256] E4M3 + scale [rows,8] E8M0: 264

// --- DSpark drafter (mtp.0..2): 3 layers, 128 experts top-3, block 5, taps layers 37..39 ---
constexpr int DS_NEXP = 128;         // [cfg] dspark_n_routed_experts
constexpr int DS_TOPK = 3;           // [cfg] dspark_num_experts_per_tok
constexpr int DS_BLOCK = 5;          // [cfg] dspark_block_size
constexpr int DS_TARGET0 = 37;       // [cfg] dspark_target_layer_ids [37, 38, 39]
constexpr int DS_MARKOV = 256;       // [cfg] dspark_markov_rank
constexpr int DS_NOISE_TOKEN = 128799;   // [cfg] dspark_noise_token_id
constexpr int DS_MAIN_PROJ_IN = 3 * HID;     // [shape] mtp.0.main_proj.weight [5120, 15360]: the three tapped hidden states
constexpr int DS_CONF_IN = HID + 256;        // [shape] confidence_head.proj [1, 5376]

// --- tokens ---
constexpr int TOK_BOS = 0, TOK_EOS = 1, TOK_PAD = 2;   // [cfg]
constexpr int TOK_IMAGE = 129264;    // [cfg] image_token_id (the vision tower is not served)
constexpr int TOK_THINK_END = 128822;    // [tok] </think> (<think> 128821), checked against the DeepSeek tokenizer

// --- compile-time checks: the tensor shapes of the checkpoint ---
constexpr int count_ratio(int r) { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += compress_ratio(l) == r; return n; }
constexpr int count_kv_src() { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += is_kv_source(l); return n; }
constexpr int count_idx_src() { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += is_index_source(l); return n; }
static_assert(NLAYERS == 43 && MTP_LAYER == 40, "40 trunk layers + 3 DSpark layers");
static_assert(count_ratio(2) == 18 && count_ratio(1) == 20 && count_ratio(0) == 5, "2 + 3 window-only, 18 ratio-2, 20 ratio-1");
static_assert(count_kv_src() == N_KV_SRC && count_idx_src() == N_IDX_SRC, "4 KV sources, 8 index sources");
static_assert(kv_source(39) == 20 && index_source(39) == 36 && kv_source(19) == 14 && index_source(23) == 20, "nearest source <= layer");
static_assert(Q_OUT == 32768 && O_GROUP_IN == 4096, "wq_b rows, wo_a columns");
static_assert(O_GROUPS * O_LORA == 8192, "wo_a.weight [8192, 4096], wo_b.weight [5120, 8192]");
static_assert(HC_FLAT == 20480 && HC_MIX == 24, "hc_*_fn [24, 20480]");
static_assert(DS_MAIN_PROJ_IN == 15360 && ENGRAM_DIM * NGRAM_TABLES == 6144, "main_proj [5120, 15360]; engram wkv [25600, 6144]");
static_assert(ENGRAM_ROW_BYTES == 264, "engram row: 256 E4M3 + 8 E8M0");
static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EXP_I % 128 == 0 && HEAD_DIM % 128 == 0, "128-wide tiles");
static_assert(HID % FP8_BLOCK == 0 && EXP_I % FP4_GROUP == 0 && QA % FP8_BLOCK == 0, "scale grids: wq_a.scale [40, 160]");
}
