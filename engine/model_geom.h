// The model geometry, the one compile-time definition of the served model's shape the engine shares.
// Default: GLM 5.3 (H6144, 256 experts, MLA + DSA, MTP layer 78). -DG6_MODEL_FLASH=1 builds GLM 5.3 Flash
// (H4096, 288 experts, MTP layer 45; scripts/build.sh MODEL=flash installs m4sim_flash / g6m4_flash).
// The GLM values equal the literals they replace, so the default build is SASS-identical to the tree before this header.
// A Flash build compiles; the Flash-only blocks (KDA, mHC, NoPE MLA, kpool indexer, 288-expert router) are not here yet.
#pragma once
#ifndef G6_MODEL_FLASH
#define G6_MODEL_FLASH 0
#endif

namespace geom
{
#if G6_MODEL_FLASH
constexpr int HID = 4096;            // hidden size
constexpr int QA = 1536;             // q_lora_rank
constexpr int QK_NOPE = 256, QK_ROPE = 0;   // per-head q/k dims: no RoPE part
constexpr int NEXP = 288;            // routed experts
constexpr int L_MAIN = 45;           // decoder layers 0..44; the MTP layer is 45
#define G6_MTP_L "L045"              // pack tensor prefix of the MTP layer (string literal, for concatenation)
constexpr int DFL_L = 5, DFL_NT = 5, DFL_NH = 32;   // DFlash2 drafter (runtime/dflash.cuh): layers, taps, q heads
#else
constexpr int HID = 6144;
constexpr int QA = 2048;
constexpr int QK_NOPE = 192, QK_ROPE = 64;
constexpr int NEXP = 256;
constexpr int L_MAIN = 78;
#define G6_MTP_L "L078"
constexpr int DFL_L = 6, DFL_NT = 6, DFL_NH = 64;
#endif
constexpr int HEADS = 64;            // attention heads
constexpr int KV_LORA = 512;         // MLA latent
constexpr int KVA = KV_LORA + QK_ROPE;   // kv_a output (and the absorbed query/key width): 576 / 512
constexpr int V_HEAD = 256;          // per-head value dim
constexpr int TOPK = 8;              // routed experts per token
constexpr int EXP_I = 2048;          // routed expert intermediate size
constexpr int EXP_BLOCKS = EXP_I / 128;  // 128-column blocks per routed expert: 16
constexpr int DENSE_I = 12288;       // dense MLP intermediate size (layers 0..2)
constexpr int VOCAB = 154880;
constexpr int VTILES = VOCAB / 128;  // lm_head 128-row tiles: 1210
constexpr int KSH = HID / 16;        // k16 steps over hidden: 384 / 256
constexpr int HB = HID / 128;        // 128-column blocks over hidden: 48 / 32
constexpr int MTP_LAYER = L_MAIN;    // 78 / 45
constexpr int NLAYERS = L_MAIN + 1;  // pack layers incl. MTP: 79 / 46
constexpr int IDX_HEADS = 32, IDX_DIM = 128, IDX_TOPK = 2048;   // DSA indexer: heads, head dim, selected keys
constexpr float RMS_EPS = 1e-5f;     // rms_norm_eps, both models
constexpr float ROUTE_SCALE = 2.5f;  // routed_scaling_factor, both models
static_assert(VOCAB % 128 == 0 && HID % 128 == 0 && EXP_I % 128 == 0, "128-wide tiles");

#if G6_MODEL_FLASH
// Flash-only blocks (engine INTERFACES.md section 1; values from the release config.json)
constexpr int KDA_HEADS = 64, KDA_DK = 128, KDA_DV = 128;   // linear_attn_config num_heads, head_dim
constexpr int KDA_D = KDA_HEADS * KDA_DK;                   // 8192
constexpr int KDA_CONV = 4;          // short_conv_kernel_size; the conv state keeps KDA_CONV - 1 = 3 tokens
constexpr float KDA_GATE_LO = -5.0f; // gate_lower_bound: g = KDA_GATE_LO * sigmoid(exp(A_log) * (f + dt_bias))
constexpr int HC = 4;                // mHC streams (hc_mult)
constexpr int HC_MIX = (2 + HC) * HC;    // 24: pre 4, post 4, comb 16
constexpr int HC_ITERS = 20;         // hc_sinkhorn_iters: 1 column pass, then 19 x (row, column)
constexpr float HC_EPS = 1e-6f;      // hc_eps
constexpr float RMS_EPS_FLASH = RMS_EPS;  // every RMSNorm incl. the mHC flat norm and the KDA o_norm
constexpr int KPOOL = 4;             // index_kpool: keys per indexer pool
constexpr float SWIGLU_LIMIT = 10.0f;    // swiglu_limit: gate <= 10, up in [-10, 10] (routed, shared and dense MLPs)
// layer_types: linear_attention (KDA) except every fourth from 3 (deepseek_sparse_attention); the MTP layer is DSA
constexpr bool is_kda(int l) { return l >= 0 && l < L_MAIN && l % 4 != 3; }
constexpr bool is_moe(int l) { return l >= 3; }         // first_k_dense_replace = 3; the MTP layer too
constexpr int kda_slot(int l) { return l - l / 4; }     // index of KDA layer l among the KDA layers
constexpr int count_kda() { int n = 0; for (int l = 0; l < NLAYERS; ++l) n += is_kda(l); return n; }
constexpr int N_KDA = 34;
static_assert(count_kda() == N_KDA && kda_slot(44) == N_KDA - 1, "34 KDA layers, slots 0..33");
#endif

// the layers that run their own indexer (engine/attn/bind.cu binder rule). GLM: the MTP layer, 0..2 and every fourth
// from 2; the others reuse the previous full layer's selection. Flash: every DSA layer has its own (kpool) indexer.
constexpr bool full_index(int l)
{
#if G6_MODEL_FLASH
    return l >= 0;
#else
    return l == MTP_LAYER || l <= 2 || (l - 2) % 4 == 0;
#endif
}
}
