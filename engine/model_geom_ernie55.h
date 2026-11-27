// ERNIE-5.5-Flash-Preview (`Ernie5_5_MoeForCausalLM`) geometry (-DG6_MODEL_ERNIE55=1), included by model_geom.h.
// Single source of truth for loader, runtime, attention, mHC, MoE, hybrid offload, and host unit tests.
// Verified against config.json and the public exlr8 release (`kindlingai/ernie-5.5-flash-exlr8-k2-k3-k4`).
#pragma once
#include <cstddef>
#include <cstdint>

namespace geom
{
// --- core transformer dimensions ---
constexpr int HID = 4096;                   // [cfg] hidden_size
constexpr int VOCAB_RAW = 201216;           // [cfg] vocab_size
constexpr int VOCAB = 201216;               // [der] multiple of 128 (1572 tiles)
constexpr int VTILES = VOCAB / 128;         // [der] 1572
constexpr int KSH = HID / 16;               // [der] 256
constexpr int HB = HID / 128;               // [der] 32
constexpr int L_MAIN = 43;                  // [cfg] num_hidden_layers (0..42)
constexpr int N_NEXTN = 0;                  // [cfg] num_nextn_predict_layers
constexpr int MTP_LAYER = L_MAIN;           // [der] 43 (no MTP active)
#define G6_MTP_L "L043"
constexpr int NLAYERS = L_MAIN;             // [der] 43
constexpr int FIRST_MOE = 1;                // [cfg] first_k_dense_replace (L0 dense MLP, L1..L42 Latent MoE)
constexpr int N_MOE_LAYERS = NLAYERS - FIRST_MOE; // [der] 42
constexpr int DENSE_I = 14336;              // [cfg] intermediate_size (L0 dense SiTU-GLU MLP)
constexpr int DENSE_INTER = DENSE_I;        // alias
constexpr int DENSE_BLOCKS = DENSE_I / 128; // [der] 112
constexpr float RMS_EPS = 1e-6f;            // [cfg] rms_norm_eps
constexpr int MAX_POS = 1048576;            // [cfg] max_position_embeddings (1M)
constexpr int KV_PAGE = 64;                 // tokens per paged KV block
constexpr int DFL_L = 5, DFL_NT = 5, DFL_NH = 32; // shared gr8 header compatibility

// --- tokens ---
constexpr int TOK_PAD = 0;                  // [cfg] pad_token_id
constexpr int TOK_BOS = 1;                  // [cfg] bos_token_id
constexpr int TOK_EOS = 2;                  // [cfg] eos_token_id

// --- 4-stream Manifold Hyper-Connections (mHC) ---
// NOTE: ERNIE 5.5 mHC flat norm puts eps OUTSIDE sqrt: 1 / (sqrt(mean(x^2)) + RMS_EPS).
constexpr int HC = 4;                       // [cfg] hc_mult
constexpr int HC_DIM = HC * HID;            // [der] 16384
constexpr int HC_D = HC_DIM;                // alias
constexpr int HC_MIX = 2 * HC + HC * HC;    // [der] 24 (pre 4, post 4, comb 16)
constexpr int HC_ITERS = 20;                // [cfg] hc_sinkhorn_iters
constexpr float HC_EPS = 1e-6f;             // [cfg] hc_eps
constexpr float HC_POST_ALPHA = 2.0f;       // post_mix = 2 * sigmoid(...)

// --- SiTU-GLU activation ---
constexpr float SITU_BETA = 4.0f;           // [cfg] activation_situ_beta
constexpr float SITU_LINEAR_BETA = 25.0f;   // [cfg] activation_situ_linear_beta

// --- Latent MoE (L1..L42) ---
constexpr int NEXP = 512;                   // [cfg] n_routed_experts
constexpr int N_ROUTED = NEXP;              // alias
constexpr int N_SHARED = 1;                 // [cfg] n_shared_experts
constexpr int TOPK = 10;                    // [cfg] num_experts_per_tok
constexpr int TOP_K = TOPK;                 // alias
constexpr int EH = 2048;                    // [cfg] moe_latent_size
constexpr int MOE_LATENT = EH;              // alias
constexpr int EHB = EH / 128;               // [der] 16
constexpr int EH_KSH = EH / 16;             // [der] 128
constexpr int EXP_I = 2048;                 // [cfg] moe_intermediate_size
constexpr int MOE_INTER = EXP_I;            // alias
constexpr int EXP_BLOCKS = EXP_I / 128;     // [der] 16
constexpr int ROUTED_NH = EXP_BLOCKS;       // alias
constexpr int SHARED_I = EXP_I;             // [cfg] shared expert intermediate size (in/out = HID 4096)
constexpr int SHARED_BLOCKS = SHARED_I / 128; // [der] 16
constexpr int SHARED_NH = SHARED_BLOCKS;    // alias
constexpr int ROUTED_HBLK = EHB;            // [der] 16 (2048 / 128)
constexpr int SHARED_HBLK = HB;             // [der] 32 (4096 / 128)
constexpr float ROUTE_SCALE = 3.0f;         // [cfg] routed_scaling_factor (default; learnable per-expert param overrides)

// --- Hybrid Attention pattern: 6 sparse MLA-DSA layers (7,14,21,28,35,42) + 37 HCA C128 layers ---
constexpr bool is_dsa_layer(int l) {
    return l == 7 || l == 14 || l == 21 || l == 28 || l == 35 || l == 42;
}
constexpr bool is_dsa(int l) { return is_dsa_layer(l); }
constexpr bool is_hca(int l) { return l >= 0 && l < L_MAIN && !is_dsa_layer(l); }
constexpr bool is_moe(int l) { return l >= FIRST_MOE && l < L_MAIN; }
constexpr bool is_dense(int l) { return l >= 0 && l < FIRST_MOE; }
constexpr bool full_index(int l) { return is_dsa_layer(l); }
constexpr int compress_ratio(int l) { return is_dsa_layer(l) ? -2 : 128; }

constexpr int count_dsa_layers() {
    int n = 0;
    for (int l = 0; l < NLAYERS; ++l) if (is_dsa_layer(l)) ++n;
    return n;
}
constexpr int N_DSA_LAYERS = count_dsa_layers();        // 6
constexpr int N_DSA = N_DSA_LAYERS;                     // alias
constexpr int N_HCA_LAYERS = NLAYERS - N_DSA_LAYERS;    // 37
constexpr int N_HCA = N_HCA_LAYERS;                     // alias

constexpr int dsa_slot_of(int l) {
    int s = 0;
    for (int i = 0; i < l; ++i) if (is_dsa_layer(i)) ++s;
    return is_dsa_layer(l) ? s : -1;
}
constexpr int hca_slot_of(int l) {
    int s = 0;
    for (int i = 0; i < l; ++i) if (!is_dsa_layer(i)) ++s;
    return !is_dsa_layer(l) ? s : -1;
}

// --- 37x HCA compressed attention (compress_ratios[l] = 128) ---
constexpr int HEADS = 64;                   // [cfg] num_attention_heads (both HCA and DSA)
constexpr int HCA_HEADS = 64;
constexpr int HCA_KV_HEADS = 1;             // [cfg] num_key_value_heads
constexpr int HCA_HD = 512;                 // [cfg] head_dim & v_head_dim
constexpr int HCA_NOPE = 448;               // [cfg] qk_nope_head_dim
constexpr int HCA_ROPE = 64;                // [cfg] qk_rope_head_dim
constexpr int HCA_Q_LORA = 1024;            // [cfg] q_lora_rank
constexpr int HCA_O_GROUPS = 8;             // [cfg] o_groups
constexpr int HCA_HPG = HCA_HEADS / HCA_O_GROUPS; // [der] 8
constexpr int HCA_O_LORA = 1024;            // [cfg] o_lora_rank
constexpr int HCA_O_WIDTH = HCA_O_GROUPS * HCA_O_LORA; // [der] 8192
constexpr int HCA_CR = 128;                 // [cfg] compress_ratios[hca]
constexpr int SLIDING_WIN = 128;            // [cfg] sliding_window
constexpr int HCA_SWA_RING = 2048;          // power-of-2 ring >= SLIDING_WIN + max prefill chunk
constexpr int VHA_RANK = 16;                // [shape] vha_postmix_{U,V}: [64, 16]
constexpr float HCA_ROPE_THETA = 160000.0f; // [cfg] compress_rope_theta
constexpr int HCA_YARN_FACTOR = 16;         // [cfg] rope_scaling.factor
constexpr int HCA_YARN_ORIG = 65536;        // [cfg] rope_scaling.original_max_position_embeddings
constexpr float HCA_YARN_BETA_FAST = 32.0f; // [cfg] rope_scaling.beta_fast
constexpr float HCA_YARN_BETA_SLOW = 1.0f;  // [cfg] rope_scaling.beta_slow

// --- 6x MLA-DSA sparse attention (compress_ratios[l] = -2) ---
constexpr int DSA_HEADS = 64;               // [cfg] hybrid_mla_num_attention_heads
constexpr int QA = 1024;                    // [cfg] hybrid_mla_q_lora_rank
constexpr int DSA_Q_LORA = QA;
constexpr int KV_LORA = 512;                // [cfg] hybrid_mla_kv_lora_rank
constexpr int DSA_KV_LORA = KV_LORA;
constexpr int QK_NOPE = 192;                // [cfg] hybrid_mla_qk_nope_head_dim
constexpr int DSA_NOPE = QK_NOPE;
constexpr int QK_ROPE = 64;                 // [cfg] hybrid_mla_qk_rope_head_dim
constexpr int DSA_ROPE = QK_ROPE;
constexpr int QK_HEAD = QK_NOPE + QK_ROPE;  // [der] 256
constexpr int DSA_QK_DIM = QK_HEAD;
constexpr int V_HEAD = 256;                 // [cfg] hybrid_mla_v_head_dim
constexpr int DSA_V_DIM = V_HEAD;
constexpr int KVB_HEAD = QK_NOPE + V_HEAD;  // [der] 448
constexpr int KVA = KV_LORA + QK_ROPE;      // [der] 576
constexpr int DSA_O_WIDTH = DSA_HEADS * DSA_V_DIM; // [der] 16384
constexpr int IDX_HEADS = 64;               // [cfg] index_n_heads
constexpr int IDX_HD = 128;                 // [cfg] index_head_dim
constexpr int IDX_DIM = IDX_HD;             // alias
constexpr int IDX_TOPK = 2048;              // [cfg] index_topk
constexpr int DSA_ATTN_TOPK = IDX_TOPK + SLIDING_WIN; // [der] 2176
constexpr float DSA_ROPE_THETA = 8000000.0f; // [cfg] rope_theta

// --- TP sharding helpers (nr in {1, 2, 4}) ---
constexpr bool rank_count_ok(int nr) { return nr == 1 || nr == 2 || nr == 4; }
constexpr int nh_local(int nr) { return HEADS / nr; }
constexpr int hca_ogroups_local(int nr) { return HCA_O_GROUPS / nr; }
constexpr int dense_i_local(int nr) { return DENSE_I / nr; }
constexpr int exp_blocks_local(int nr) { return EXP_BLOCKS / nr; }
constexpr int shared_blocks_local(int nr) { return SHARED_BLOCKS / nr; }
constexpr int vocab_tiles(int nr, int r) { return VTILES / nr + (r < VTILES % nr ? 1 : 0); }
constexpr int vocab_tile0(int nr, int r) { return r * (VTILES / nr) + (r < VTILES % nr ? r : VTILES % nr); }
constexpr int vocab_rows(int nr, int r) { return vocab_tiles(nr, r) * 128; }
constexpr int vocab_row0(int nr, int r) { return vocab_tile0(nr, r) * 128; }

// --- exlr8 mul1 trellis byte helpers ---
constexpr size_t routed_expert_bytes(int k) {
    return 3ull * static_cast<size_t>(MOE_LATENT) * static_cast<size_t>(MOE_INTER) * static_cast<size_t>(k) / 8ull;
}
constexpr size_t shared_expert_bytes(int k) {
    return 3ull * static_cast<size_t>(HID) * static_cast<size_t>(MOE_INTER) * static_cast<size_t>(k) / 8ull;
}

static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EH % 128 == 0 && EXP_I % 128 == 0 && DENSE_I % 128 == 0,
              "128-wide tile alignment");
static_assert(N_DSA_LAYERS == 6 && N_HCA_LAYERS == 37 && N_DSA_LAYERS + N_HCA_LAYERS == NLAYERS,
              "6 DSA + 37 HCA = 43 layers");
static_assert(HCA_NOPE + HCA_ROPE == HCA_HD && HCA_O_GROUPS * HCA_HPG == HCA_HEADS, "HCA head geometry");
static_assert(DSA_NOPE + DSA_ROPE == DSA_QK_DIM && DSA_NOPE + DSA_V_DIM == KVB_HEAD, "DSA head geometry");
}  // namespace geom

namespace ernie55 {
namespace geom = ::geom;
}  // namespace ernie55
