// Qwen3.8-Flash-Next geometry (-DG6_MODEL_QW38=1), included by model_geom.h. engine/tests/test_qw38_geom.cpp checks
// every value below, and reads them against the release's config.json when GR8_TEST_QW38_METADATA_DIR is set.
//   [cfg]   the config.json text_config (or generation_config.json) field of the same meaning
//   [shape] read off the checkpoint's tensor shapes (Qwen/Qwen3.8-Flash-Next@812d96e)
//   [der]   derived from [cfg] or [shape] values
// Supports TP widths nr = 1, 2, 3, 4 cleanly:
//   full attention: 24 Q heads over 2 KV heads (GQA = 12). TP=1: 24Q/2KV; TP=2: 12Q/1KV; TP=3: 8Q over 1-2KV; TP=4: 6Q/1KV.
//   linear attn:    16 K heads, 48 V heads (3 V per K).    TP=1: 16K/48V; TP=2: 8K/24V; TP=3: 6K/16V; TP=4: 4K/12V.
//   MoE experts:    512 routed, top-10, 640 = 5 x 128-blocks, block-split rotated by (5*e + h + l) % nr.
//   vocab:          248,320 = 1,940 tiles of 128, split with g6nr::width(1940, r).
#pragma once

namespace geom
{
// --- trunk ---
constexpr int HID = 2560;             // [cfg] hidden_size
constexpr int VOCAB_RAW = 248320;     // [cfg] vocab_size
constexpr int VOCAB = 248320;         // [der] already a multiple of 128 (1940 tiles)
constexpr int VTILES = VOCAB / 128;   // [der] lm_head 128-row tiles: 1940
constexpr int KSH = HID / 16;         // [der] k16 steps over hidden: 160
constexpr int HB = HID / 128;         // [der] 128-column blocks over hidden: 20
constexpr int L_MAIN = 48;            // [cfg] num_hidden_layers: layers 0..47
constexpr int N_NEXTN = 1;            // [cfg] mtp_num_hidden_layers: 1 full-attention + MoE drafter layer
constexpr int MTP_LAYER = L_MAIN;     // [der] pack index of the MTP layer (48)
#define G6_MTP_L "L048"              // pack tensor prefix of the MTP layer
constexpr int NLAYERS = L_MAIN + N_NEXTN;   // [der] 49
constexpr float RMS_EPS = 1e-6f;      // [cfg] rms_norm_eps
constexpr int MAX_POS = 262144;       // [cfg] max_position_embeddings
constexpr int KV_PAGE = 64;           // tokens per full-attention KV page
constexpr int DENSE_I = 0;            // [shape] no dense MLP layers: all 48 trunk + 1 MTP layers are MoE
constexpr bool is_moe(int l) { return l >= 0 && l < NLAYERS; }
constexpr int DFL_L = 5, DFL_NT = 5, DFL_NH = 32;   // runtime/dflash.cuh compiles in every build; no DFlash2 here
constexpr int KV_LORA = 0, KVA = 0;                 // comms/cplayout.h compiles in every build; no MLA CP here

// --- hyper-connections: 4 parallel residual streams, low-rank 320 mixing ---
constexpr int HC = 4;                 // [cfg] hc_count
constexpr int HC_D = HC * HID;        // [shape] 10240: concatenated 4-stream state width
constexpr int HC_RANK = 320;          // [cfg] hc_lowrank: input_mix_weight_{down,up} inner dim

// --- full attention (layers 3, 7, ..., 47 + MTP layer 48): 24 Q heads, 2 KV heads, d=256, sigmoid gate ---
constexpr int HEADS = 24;             // [cfg] num_attention_heads
constexpr int KV_HEADS = 2;           // [cfg] num_key_value_heads
constexpr int GQA = HEADS / KV_HEADS; // [der] 12 query heads per KV head
constexpr int HEAD_DIM = 256;         // [cfg] head_dim
constexpr int ROPE_DIM = 64;          // [cfg] head_dim * partial_rotary_factor (256 * 0.25)
constexpr float ROPE_THETA = 1e7f;    // [cfg] rope_parameters.rope_theta
constexpr int MROPE_SEC0 = 11, MROPE_SEC1 = 11, MROPE_SEC2 = 10;   // [cfg] rope_parameters.mrope_section (sum = 32 pairs = 64 dims)
constexpr int Q_PROJ_OUT = HEADS * HEAD_DIM * 2;   // [shape] 12288: fused Q + sigmoid output gate
constexpr int KV_PROJ_OUT = KV_HEADS * HEAD_DIM;   // [shape] 512
constexpr int O_PROJ_IN = HEADS * HEAD_DIM;        // [shape] 6144

// --- QSA sparse indexer (full-attention layers + MTP): 4 Q heads x 128, 1 KV head, blocks of 4, budget 2048 ---
constexpr int IDX_HEADS = 4;          // [cfg] indexer_n_heads
constexpr int IDX_KV_HEADS = 1;       // [cfg] indexer_kv_heads
constexpr int IDX_DIM = 128;          // [cfg] indexer_head_dim
constexpr int IDX_COMPRESS = 4;       // [cfg] indexer_compress_ratio (tokens per compressed block)
constexpr int IDX_TOPK = 2048;        // [cfg] indexer_budget (tokens; no-op when ctx <= 2048)
constexpr int IDX_TOP_BLOCKS = IDX_TOPK / IDX_COMPRESS;   // [der] 512 blocks of 4
constexpr int IDX_PROJ_OUT = (IDX_HEADS + IDX_KV_HEADS) * IDX_DIM;   // [shape] 640: index_qk_proj.weight [640, 2560]

// --- linear attention (Gated DeltaNet, 36 layers): 16 K heads, 48 V heads, d=128, conv kernel 4, FP32 state ---
constexpr int GDN_KH = 16;            // [cfg] linear_num_key_heads
constexpr int GDN_VH = 48;            // [cfg] linear_num_value_heads
constexpr int GDN_V_PER_K = GDN_VH / GDN_KH;   // [der] 3 value heads per key head
constexpr int GDN_DK = 128;           // [cfg] linear_key_head_dim
constexpr int GDN_DV = 128;           // [cfg] linear_value_head_dim
constexpr int GDN_CONV = 4;           // [cfg] linear_conv_kernel_dim
constexpr int GDN_NHIST = GDN_CONV - 1;        // [der] 3-token conv1d history per slot
constexpr int GDN_Q_DIM = GDN_KH * GDN_DK;     // [der] 2048
constexpr int GDN_K_DIM = GDN_KH * GDN_DK;     // [der] 2048
constexpr int GDN_V_DIM = GDN_VH * GDN_DV;     // [der] 6144
constexpr int GDN_QKV_DIM = GDN_Q_DIM + GDN_K_DIM + GDN_V_DIM;   // [shape] 10240: in_proj_qkv [10240, 2560]
constexpr int N_GDN = 36;             // [der] 36 linear-attention layers
constexpr int N_FULL = 12;            // [der] 12 trunk full-attention layers

// --- PLE (Per-Layer N-gram Embedding) on 0-indexed layer 1 (config.json ple_layer_ids: [2] is 1-indexed) ---
constexpr int PLE_LAYER = 1;          // [shape] layers.1.ple.* (1-indexed ple_layer_ids = [2])
constexpr int PLE_NGRAM = 3;          // [cfg] ngram_size (bigrams + trigrams)
constexpr int PLE_HEADS_PER_NGRAM = 8;   // [cfg] heads_per_ngram
constexpr int PLE_HEADS = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM;   // [shape] 16: ngram_heads_offsets [16]
constexpr int PLE_DIM = 2560;         // [cfg] ple_embed_dim
constexpr int PLE_HEAD_DIM = PLE_DIM / PLE_HEADS;   // [shape] 160: shard_i.weight [*, 160]
constexpr int PLE_SHARDS = 128;       // [cfg] split_ngram_parts
constexpr int PLE_CONV = 4;           // [cfg] ple_conv_kernel_size
constexpr int PLE_CONV_DILATION = PLE_NGRAM;   // [code] modeling_qwen4_exp: dilation = ngram_size = 3
constexpr int PLE_NHIST = (PLE_CONV - 1) * PLE_CONV_DILATION;   // [der] 9-token dilated conv1d history
constexpr int PLE_NGRAM_CTX = PLE_NGRAM - 1;   // [der] 2 prior token ids for bigram/trigram hashing
constexpr int GDN_GATE_SIGMOID = 1;   // [cfg] output_gate_type = "sigmoid" (GDN RMSNormGated uses sigmoid)

// --- MoE: 512 routed experts top-10, intermediate 640 (5 blocks of 128) + gated shared expert (640) ---
constexpr int NEXP = 512;             // [cfg] num_experts
constexpr int TOPK = 10;              // [cfg] num_experts_per_tok
constexpr int EXP_I = 640;            // [cfg] moe_intermediate_size
constexpr int EXP_BLOCKS = EXP_I / 128;   // [der] 5 blocks of 128 (<= 8: fits tr3.kmix and ptab at every TP incl. TP=1)
constexpr int SHARED_I = 640;         // [cfg] shared_expert_intermediate_size
constexpr float ROUTE_SCALE = 1.0f;   // [cfg] softmax top-10 normalized

// --- tokens ---
constexpr int TOK_BOS = 248044;       // [cfg] bos_token_id
constexpr int TOK_EOS = 248044;       // [cfg] eos_token_id
constexpr int TOK_EOS_IM_END = 248046;// [cfg] generation_config.json eos_token_id[0] (<|im_end|>)
constexpr int TOK_IMAGE = 248056;     // [cfg] image_token_id

// --- per-layer predicates ---
constexpr bool is_full(int l) { return (l >= 0 && l < L_MAIN && l % 4 == 3) || l == MTP_LAYER; }
constexpr bool is_gdn(int l) { return l >= 0 && l < L_MAIN && l % 4 != 3; }
constexpr bool is_ple(int l) { return l == PLE_LAYER; }
constexpr int full_slot(int l) { return l == MTP_LAYER ? N_FULL : l / 4; }     // 0..11 trunk, 12 MTP
constexpr int gdn_slot(int l) { return l - l / 4; }                             // 0..35
constexpr bool full_index(int l) { return is_full(l); }

// --- per-rank sharding for nr in {1, 2, 3, 4} ---
constexpr bool rank_count_ok(int nr) { return nr == 1 || nr == 2 || nr == 3 || nr == 4; }

// Full attention: Q heads split evenly (24 / nr); KV heads are the contiguous range those Q heads read
constexpr int nh_q(int nr) { return HEADS / nr; }
constexpr int q0(int nr, int r) { return r * nh_q(nr); }
constexpr int kv0(int nr, int r) { return q0(nr, r) / GQA; }
constexpr int kv1(int nr, int r) { return (q0(nr, r) + nh_q(nr) - 1) / GQA + 1; }
constexpr int nh_kv(int nr, int r) { return kv1(nr, r) - kv0(nr, r); }
constexpr int nh_kv_max(int nr)
{
    int mx = 0;
    for (int r = 0; r < nr; ++r)
        if (nh_kv(nr, r) > mx) mx = nh_kv(nr, r);
    return mx;
}
constexpr int local_kv_head(int nr, int r, int lq) { return (q0(nr, r) + lq) / GQA - kv0(nr, r); }

// Linear attention (GDN): V heads split evenly (48 / nr); K/Q heads are the contiguous range those V heads read
// (at TP=1, 2, 4: exact disjoint split 16, 8, 4 K heads; at TP=3: 16 V heads reading 6 K heads per rank)
constexpr int gdn_vh(int nr) { return GDN_VH / nr; }
constexpr int gdn_v0(int nr, int r) { return r * gdn_vh(nr); }
constexpr int gdn_k0(int nr, int r) { return gdn_v0(nr, r) / GDN_V_PER_K; }
constexpr int gdn_k1(int nr, int r) { return (gdn_v0(nr, r) + gdn_vh(nr) - 1) / GDN_V_PER_K + 1; }
constexpr int gdn_kh(int nr, int r) { return gdn_k1(nr, r) - gdn_k0(nr, r); }
constexpr int gdn_kh_max(int nr)
{
    int mx = 0;
    for (int r = 0; r < nr; ++r)
        if (gdn_kh(nr, r) > mx) mx = gdn_kh(nr, r);
    return mx;
}
constexpr int gdn_local_k(int nr, int r, int lv) { return (gdn_v0(nr, r) + lv) / GDN_V_PER_K - gdn_k0(nr, r); }
constexpr int gdn_qkv_local(int nr, int r) { return (2 * gdn_kh(nr, r) + gdn_vh(nr)) * GDN_DK; }

// MoE: 5 blocks of 128 per expert, rotated across ranks: block h of expert e in layer l -> (5 * e + h + l) % nr
constexpr int exp_blocks_max(int nr) { return (EXP_BLOCKS + nr - 1) / nr; }   // 5, 3, 2, 2 for TP=1,2,3,4

// Vocabulary: 1940 tiles of 128, split with larger chunks first (g6nr::width)
constexpr int vocab_tiles(int nr, int r) { return VTILES / nr + (r < VTILES % nr ? 1 : 0); }
constexpr int vocab_tile0(int nr, int r) { return r * (VTILES / nr) + (r < VTILES % nr ? r : VTILES % nr); }
constexpr int vocab_rows(int nr, int r) { return vocab_tiles(nr, r) * 128; }
constexpr int vocab_row0(int nr, int r) { return vocab_tile0(nr, r) * 128; }

// --- compile-time checks ---
constexpr int count_gdn() { int n = 0; for (int l = 0; l < L_MAIN; ++l) n += is_gdn(l); return n; }
constexpr int count_full() { int n = 0; for (int l = 0; l < L_MAIN; ++l) n += is_full(l); return n; }
static_assert(count_gdn() == N_GDN && count_full() == N_FULL && N_GDN + N_FULL == L_MAIN, "36 GDN + 12 full");
static_assert(gdn_slot(46) == N_GDN - 1 && full_slot(47) == N_FULL - 1 && full_slot(MTP_LAYER) == N_FULL, "layer slots");
static_assert(is_ple(1) && !is_ple(0) && !is_ple(2), "PLE on 0-indexed layer 1");
static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EXP_I % 128 == 0 && HEAD_DIM % 128 == 0, "128-wide tiles");
static_assert(EXP_BLOCKS == 5 && exp_blocks_max(1) <= 8, "fits tr3.kmix and ptab at every TP including TP=1");
static_assert(nh_q(1) == 24 && nh_kv(1, 0) == 2, "TP1 full attn");
static_assert(nh_q(2) == 12 && nh_kv(2, 0) == 1 && nh_kv(2, 1) == 1 && kv0(2, 1) == 1, "TP2 full attn");
static_assert(nh_q(3) == 8 && nh_kv(3, 0) == 1 && nh_kv(3, 1) == 2 && nh_kv(3, 2) == 1, "TP3 full attn");
static_assert(nh_q(4) == 6 && nh_kv(4, 0) == 1 && kv0(4, 1) == 0 && kv0(4, 2) == 1 && kv0(4, 3) == 1, "TP4 full attn");
static_assert(gdn_vh(1) == 48 && gdn_kh(1, 0) == 16 && gdn_vh(2) == 24 && gdn_kh(2, 0) == 8, "TP1/TP2 GDN");
static_assert(gdn_vh(3) == 16 && gdn_kh(3, 0) == 6 && gdn_kh(3, 1) == 6 && gdn_kh(3, 2) == 6, "TP3 GDN");
static_assert(gdn_vh(4) == 12 && gdn_kh(4, 0) == 4 && gdn_k0(4, 3) == 12, "TP4 GDN");
static_assert(MROPE_SEC0 + MROPE_SEC1 + MROPE_SEC2 == ROPE_DIM / 2, "mRoPE sections sum to 32 pairs");
}   // namespace geom
