// Drama 3 (fish_omni) weight structs and 64K-page-safe pinned-staging GPU loader for g6tts.
//
// Memory modes:
//   - WeightMode::Ref (default): dense/tower/vocab/shared in FP16 (~5 GB); store-0 routed experts kept
//     in block-128 FP8 E4M3 + F32 scales on GPU (~32.2 GB, ~37 GB total) and dequantized on the fly
//     for active routed experts. Store-1 (trellis) layers decode to FP16 on load.
//   - WeightMode::RefFp16: all matrices including every routed expert pre-decoded to FP16 on load (~72 GB).
//   - WeightMode::Native (Wave 3.2): dense/tower matrices kept in native EXLR8 formats (Trellis K4 mcg,
//     FP8_ROW, INT4, BF16), sliced FP8_ROW lm_head (4097 rows) & residual heads (1024 rows), native FP8_ROW
//     embed_tokens, and TP=1 grouped Trellis experts planned via `g6l_dr3_plan` (falling back cleanly to
//     Ref FP8_BLOCK/FP16 experts on layers without trellis experts).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "../embed/embed_weights.h"
#include "../loader-rs/g6load.h"
#include "tts_config.h"
#include "tts_moe.cuh"

namespace g6tts {

inline void tts_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("g6tts CUDA error: ") + what + ": " + cudaGetErrorString(e));
    }
}
#define TTS_CK(x) ::g6tts::tts_ck((x), #x)

inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

// Host BF16/FP16 conversion helpers (pure C++17, zero CUDA host-overload dependency).
inline float host_bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float host_f16_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x03ffu;
    uint32_t u = 0;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            uint32_t mm = m;
            int32_t ee = -14;
            while ((mm & 0x0400u) == 0) { mm <<= 1; --ee; }
            mm &= 0x03ffu;
            u = s | (uint32_t(ee + 127) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t host_f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline void host_f16_buf_to_bf16_inplace(void* buf, size_t n_elems) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < n_elems; ++i) p[i] = host_f32_to_bf16(host_f16_to_f32(p[i]));
}

// Reference decoded dense matrix pointer: BF16 in Ref/RefFp16 mode, or exact FP32 in RefF32 mode.
struct RefMat {
    const __nv_bfloat16* bf16 = nullptr;
    const float* f32 = nullptr;
    bool valid() const { return bf16 != nullptr || f32 != nullptr; }
};

struct BackboneLayerW {
    bool is_full_attn = false;
    bool experts_fp8 = false; // true: routed experts stored as FP8_BLOCK on device (~30 GB total)

    const float* input_ln = nullptr;     // [H]
    const float* post_attn_ln = nullptr; // [H]

    // Full attention (l % 4 == 3) — Ref BF16/FP32 + Native DevMatrix descriptors
    RefMat q_proj; // [HEADS * 2 * HEAD_DIM, H] = [8192, 2048]
    RefMat k_proj; // [KV_HEADS * HEAD_DIM, H] = [512, 2048]
    RefMat v_proj; // [KV_HEADS * HEAD_DIM, H] = [512, 2048]
    RefMat o_proj; // [H, HEADS * HEAD_DIM] = [2048, 4096]
    const float* q_norm = nullptr;         // [HEAD_DIM] = [256]
    const float* k_norm = nullptr;         // [HEAD_DIM] = [256]
    g6emb::DevMatrix q_proj_nat, k_proj_nat, v_proj_nat, o_proj_nat;

    // Gated DeltaNet (l % 4 != 3) — Ref BF16/FP32 + Native DevMatrix descriptors
    RefMat in_proj_qkv; // [8192, 2048]
    RefMat in_proj_z;   // [4096, 2048]
    RefMat in_proj_a;   // [32, 2048]
    RefMat in_proj_b;   // [32, 2048]
    RefMat out_proj;    // [2048, 4096]
    const float* conv1d = nullptr;              // [8192, 4]
    const float* A_log = nullptr;               // [32]
    const float* dt_bias = nullptr;             // [32]
    const float* gdn_norm = nullptr;            // [128] (plain weight, NOT 1+w)
    g6emb::DevMatrix in_proj_qkv_nat, in_proj_z_nat, in_proj_a_nat, in_proj_b_nat, out_proj_nat;

    // MoE router & shared expert (Ref BF16/FP32 + Native DevMatrix)
    RefMat router_gate;        // [NEXP, H] = [256, 2048]
    RefMat shared_gate_proj;   // [SHARED_I, H] = [512, 2048]
    RefMat shared_up_proj;     // [SHARED_I, H] = [512, 2048]
    RefMat shared_down_proj;   // [H, SHARED_I] = [2048, 512]
    RefMat shared_expert_gate; // [1, H] = [1, 2048]
    g6emb::DevMatrix router_gate_nat, shared_expert_gate_nat;
    g6emb::DevMatrix shared_gate_nat, shared_up_nat, shared_down_nat;

    // Native grouped Trellis routed + shared experts (Wave 3.2 via `g6l_dr3_plan`)
    TtsTr3LayerW tr3;

    // Fallback Ref routed experts: either FP8_BLOCK (`experts_fp8 == true`) OR pre-decoded BF16
    const uint8_t* exp_gate_fp8 = nullptr;
    const float* exp_gate_sc = nullptr;
    const uint8_t* exp_up_fp8 = nullptr;
    const float* exp_up_sc = nullptr;
    const uint8_t* exp_down_fp8 = nullptr;
    const float* exp_down_sc = nullptr;

    const __nv_bfloat16* exp_gate_bf16 = nullptr;
    const __nv_bfloat16* exp_up_bf16 = nullptr;
    const __nv_bfloat16* exp_down_bf16 = nullptr;
};

struct TowerLayerW {
    bool is_shared = false;
    const __nv_bfloat16* embedding = nullptr; // [4096, 2048] BF16 (semantic_embedding or codebook_embedding)
    const float* enorm = nullptr;             // [2048]
    const float* hnorm = nullptr;             // [2048]
    RefMat eh_proj;                           // [2048, 4096]
    g6emb::DevMatrix eh_proj_nat;

    const float* attn_norm = nullptr; // [2048]
    RefMat wqkv;                      // [9216, 2048] = [Q 4096 | Gate 4096 | K 512 | V 512]
    RefMat wo;                        // [2048, 4096]
    const float* q_norm = nullptr;    // [256]
    const float* k_norm = nullptr;    // [256]
    g6emb::DevMatrix wqkv_nat, wo_nat;

    const float* ffn_norm = nullptr; // [2048]
    RefMat w1;                       // [6144, 2048]
    RefMat w3;                       // [6144, 2048]
    RefMat w2;                       // [2048, 6144]
    g6emb::DevMatrix w1_nat, w3_nat, w2_nat;

    const float* final_ln = nullptr; // [2048]
};

struct TtsWeights {
    TtsConfig cfg;
    WeightMode mode = WeightMode::Ref;

    uint8_t* dense_arena = nullptr;
    size_t dense_bytes = 0;
    uint8_t* expert_arena = nullptr;
    size_t expert_bytes = 0;
    uint8_t* tr3_expert_arena = nullptr;
    size_t tr3_expert_bytes = 0;

    // Global backbone + tower weights
    RefMat embed_tokens;                                // [VOCAB_SIZE, H] BF16 or FP32 (when not native FP8_ROW)
    g6emb::DevMatrix embed_tokens_nat;                  // native FP8_ROW [VOCAB_SIZE, H] when available
    const float* final_norm = nullptr;                  // [H] FP32
    RefMat lm_head_cand;                                // [MAIN_CANDIDATES (4097), H] BF16 or FP32 (Ref)
    g6emb::DevMatrix lm_head_nat;                       // [MAIN_CANDIDATES (4097), H] native FP8_ROW/BF16
    const int32_t* main_cand_ids_dev = nullptr;         // [MAIN_CANDIDATES (4097)] INT32 on GPU
    const __nv_bfloat16* codebook_embeddings = nullptr; // [NCB * 4096, H] = [40960, 2048] BF16

    std::vector<BackboneLayerW> layers;          // [40]
    std::vector<TowerLayerW> shared_layers;      // [4]
    std::vector<TowerLayerW> res_layers;         // [9]
    std::vector<RefMat> heads;                   // [9], each [4096, 2048] BF16 or FP32 (Ref)
    std::vector<g6emb::DevMatrix> heads_nat;     // [9], each [1024, 2048] native FP8_ROW/BF16

    int n_fp8_expert_layers = 0;
    int n_f16_expert_layers = 0;
    int n_tr3_expert_layers = 0;
    int n_nat_trellis = 0, n_nat_fp8_row = 0, n_nat_int4 = 0, n_nat_bf16 = 0;

    TtsWeights() = default;
    TtsWeights(const TtsWeights&) = delete;
    TtsWeights& operator=(const TtsWeights&) = delete;
    TtsWeights(TtsWeights&& o) noexcept { *this = std::move(o); }
    TtsWeights& operator=(TtsWeights&& o) noexcept {
        if (this == &o) return *this;
        release();
        cfg = std::move(o.cfg);
        mode = o.mode;
        dense_arena = o.dense_arena; o.dense_arena = nullptr;
        dense_bytes = o.dense_bytes; o.dense_bytes = 0;
        expert_arena = o.expert_arena; o.expert_arena = nullptr;
        expert_bytes = o.expert_bytes; o.expert_bytes = 0;
        tr3_expert_arena = o.tr3_expert_arena; o.tr3_expert_arena = nullptr;
        tr3_expert_bytes = o.tr3_expert_bytes; o.tr3_expert_bytes = 0;
        embed_tokens = o.embed_tokens;
        embed_tokens_nat = o.embed_tokens_nat;
        final_norm = o.final_norm;
        lm_head_cand = o.lm_head_cand;
        lm_head_nat = o.lm_head_nat;
        main_cand_ids_dev = o.main_cand_ids_dev;
        codebook_embeddings = o.codebook_embeddings;
        layers = std::move(o.layers);
        shared_layers = std::move(o.shared_layers);
        res_layers = std::move(o.res_layers);
        heads = std::move(o.heads);
        heads_nat = std::move(o.heads_nat);
        n_fp8_expert_layers = o.n_fp8_expert_layers;
        n_f16_expert_layers = o.n_f16_expert_layers;
        n_tr3_expert_layers = o.n_tr3_expert_layers;
        n_nat_trellis = o.n_nat_trellis;
        n_nat_fp8_row = o.n_nat_fp8_row;
        n_nat_int4 = o.n_nat_int4;
        n_nat_bf16 = o.n_nat_bf16;
        return *this;
    }
    ~TtsWeights() { release(); }

    void release() {
        if (tr3_expert_arena) {
            cudaFree(tr3_expert_arena);
            tr3_expert_arena = nullptr;
            tr3_expert_bytes = 0;
        }
        if (expert_arena) {
            cudaFree(expert_arena);
            expert_arena = nullptr;
            expert_bytes = 0;
        }
        if (dense_arena) {
            cudaFree(dense_arena);
            dense_arena = nullptr;
            dense_bytes = 0;
        }
    }
};

inline bool is_routed_expert_name(const std::string& name) {
    return name.find(".mlp.experts.") != std::string::npos;
}

inline bool is_small_embedding_name(const std::string& name) {
    return name == "audio_decoder.codebook_embeddings.weight" ||
           name.find(".semantic_embedding.weight") != std::string::npos ||
           name.find(".codebook_embedding.weight") != std::string::npos;
}

inline bool is_res_head_name(const std::string& name) {
    return name.rfind("audio_decoder_mtp.heads.", 0) == 0;
}

// Minimal parser for `g6pack-v1` manifest JSON emitted by `g6l_dr3_plan`.
struct PlannedTensorEntry {
    std::string name;
    std::string expert_format;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    int64_t shape0 = 0;
};

struct PlannedDr3Manifest {
    uint64_t pack_bytes = 0;
    std::vector<PlannedTensorEntry> tensors;
};

inline PlannedDr3Manifest parse_dr3_plan_manifest(const std::string& text) {
    PlannedDr3Manifest out;
    auto find_u64 = [&](const std::string& s, const char* key, size_t pos, uint64_t& val) -> bool {
        const std::string k = std::string("\"") + key + "\"";
        size_t p = s.find(k, pos);
        if (p == std::string::npos) return false;
        p = s.find(':', p + k.size());
        if (p == std::string::npos) return false;
        ++p;
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
        char* endp = nullptr;
        val = std::strtoull(s.c_str() + p, &endp, 10);
        return endp != s.c_str() + p;
    };
    auto find_str = [&](const std::string& s, const char* key, size_t pos, std::string& val) -> bool {
        const std::string k = std::string("\"") + key + "\"";
        size_t p = s.find(k, pos);
        if (p == std::string::npos) return false;
        p = s.find(':', p + k.size());
        if (p == std::string::npos) return false;
        p = s.find('"', p + 1);
        if (p == std::string::npos) return false;
        size_t q = s.find('"', p + 1);
        if (q == std::string::npos) return false;
        val = s.substr(p + 1, q - p - 1);
        return true;
    };

    find_u64(text, "pack_bytes", 0, out.pack_bytes);
    const size_t t_start = text.find("\"tensors\"");
    if (t_start == std::string::npos) return out;
    size_t p = text.find('[', t_start);
    if (p == std::string::npos) return out;
    while (p < text.size()) {
        size_t ob = text.find('{', p);
        if (ob == std::string::npos) break;
        size_t cb = text.find('}', ob);
        if (cb == std::string::npos) break;
        const std::string obj = text.substr(ob, cb - ob + 1);
        PlannedTensorEntry e;
        if (find_str(obj, "name", 0, e.name)) {
            find_u64(obj, "offset", 0, e.offset);
            find_u64(obj, "nbytes", 0, e.nbytes);
            find_str(obj, "expert_format", 0, e.expert_format);
            size_t sp = obj.find("\"shape\"");
            if (sp != std::string::npos) {
                size_t lb = obj.find('[', sp);
                if (lb != std::string::npos) {
                    e.shape0 = std::strtoll(obj.c_str() + lb + 1, nullptr, 10);
                }
            }
            out.tensors.push_back(std::move(e));
        }
        p = cb + 1;
    }
    return out;
}

inline void parse_expert_format(const std::string& xf, int& cb, bool& e4) {
    e4 = true;
    if (xf == "exlr8-e4m3-pmcgx") {
        cb = TTS_TR3_PMCGX;
    } else if (xf == "exlr8-e4m3-mul1") {
        cb = TTS_TR3_MUL1;
    } else if (xf == "exlr8-e4m3") {
        cb = TTS_TR3_MCG;
    } else {
        throw std::runtime_error("g6tts: unsupported native trellis expert_format: " + xf);
    }
}

// Loads a Drama 3 exlr8 release into GPU arenas via `g6l_tts_*` (+ `g6l_dr3_plan` in native mode)
// through pinned host staging (64K-page safe, zero mmap).
inline TtsWeights load_tts_weights(const std::string& release_dir,
                                   const std::string& manifest_name = "",
                                   bool verify_checksums = true,
                                   WeightMode mode = WeightMode::Ref,
                                   uint32_t decode_threads = 8,
                                   bool strict_geometry = true) {
    const auto t0 = std::chrono::steady_clock::now();
    char err[512] = {};
    const char* mptr = manifest_name.empty() ? nullptr : manifest_name.c_str();
    void* h = g6l_tts_open(release_dir.c_str(), mptr, verify_checksums ? 1 : 0, err, sizeof(err));
    if (!h) {
        throw std::runtime_error(std::string("g6l_tts_open failed: ") + err);
    }
    struct Guard { void* p; ~Guard() { g6l_tts_close(p); } } guard{h};

    TtsWeights W;
    W.cfg = TtsConfig::from_release(h, release_dir, strict_geometry);
    W.mode = mode;

    const auto cand_ids = W.cfg.build_main_candidate_ids();
    const int n_cand = static_cast<int>(cand_ids.size());
    const bool native_mode = (mode == WeightMode::Native);
    const bool ref_f32 = (mode == WeightMode::RefF32);

    // 0. In Native mode, if any layer has trellis experts and a manifest is provided, plan them via `g6l_dr3_plan`.
    const int L = W.cfg.num_layers;
    std::vector<bool> layer_is_tr3(static_cast<size_t>(L), false);
    void* dr3_plan_h = nullptr;
    PlannedDr3Manifest dr3_man;
    struct PlanGuard { void*& p; ~PlanGuard() { if (p) { g6l_plan_free(p); p = nullptr; } } } plan_guard{dr3_plan_h};

    if (native_mode && !manifest_name.empty()) {
        bool any_trellis = false;
        for (int l = 0; l < L; ++l) {
            if (g6l_tts_expert_store(h, uint32_t(l)) == 1) {
                any_trellis = true;
                break;
            }
        }
        if (any_trellis) {
            static std::atomic<uint64_t> tmp_seq{0};
            const std::string tmp_man = "/tmp/g6tts_dr3_plan_" + std::to_string(::getpid()) + "_" +
                                        std::to_string(tmp_seq.fetch_add(1)) + ".json";
            struct TmpRm { std::string path; ~TmpRm() { if (!path.empty()) std::remove(path.c_str()); } } rm_guard{tmp_man};
            dr3_plan_h = g6l_dr3_plan(release_dir.c_str(), manifest_name.c_str(), nullptr,
                                      tmp_man.c_str(), err, sizeof(err));
            if (!dr3_plan_h) {
                throw std::runtime_error(std::string("g6l_dr3_plan failed: ") + err);
            }
            std::ifstream ifs(tmp_man);
            if (!ifs) throw std::runtime_error("g6tts: failed reading planned dr3 manifest " + tmp_man);
            std::stringstream ss;
            ss << ifs.rdbuf();
            dr3_man = parse_dr3_plan_manifest(ss.str());
            for (const auto& te : dr3_man.tensors) {
                if (te.name.size() >= 12 && te.name[0] == 'L' && te.name.find(".tr3.fc1") != std::string::npos) {
                    const int li = std::atoi(te.name.c_str() + 1);
                    if (li >= 0 && li < L) layer_is_tr3[size_t(li)] = true;
                }
            }
        }
    }

    // 1. Plan dense arena
    struct DensePlan {
        size_t item_idx = 0;
        std::string name;
        bool is_matrix = false;
        bool is_lm_head = false;
        bool is_res_head = false;
        bool is_embed_tokens = false;
        bool keep_native = false;
        bool decode_f32 = false;
        int fmt = 0, side = 0, k = 0;
        uint32_t ncomp = 0;
        std::string raw_dtype;
        size_t raw_bytes = 0;
        size_t out = 0, inp = 0;
        size_t dev_off = 0;
        size_t dev_bytes = 0;
        size_t comp_off[3] = {0, 0, 0};
        size_t comp_bytes[3] = {0, 0, 0};
    };
    const size_t nitems = g6l_tts_dense_item_count(h);
    std::vector<DensePlan> dplans;
    dplans.reserve(1024);

    size_t dense_total = 0;
    size_t max_stage = 64u << 20;

    // Reserve device buffer for main_cand_ids_dev
    const size_t cand_ids_off = dense_total;
    const size_t cand_ids_bytes = size_t(n_cand) * sizeof(int32_t);
    dense_total += align256(cand_ids_bytes);

    for (size_t i = 0; i < nitems; ++i) {
        g6l_embed_item it{};
        if (g6l_tts_dense_item_info(h, i, &it) != 0 || !it.hf_name) continue;
        const std::string name(it.hf_name);
        if (is_routed_expert_name(name)) continue;

        DensePlan p;
        p.item_idx = i;
        p.name = name;
        p.is_matrix = it.is_matrix != 0;
        p.is_lm_head = (name == "lm_head.weight");
        p.is_res_head = is_res_head_name(name);
        p.is_embed_tokens = (name == "model.language_model.embed_tokens.weight");
        p.fmt = it.fmt;
        p.side = it.side;
        p.k = it.k;
        p.ncomp = it.ncomp;

        if (p.is_matrix) {
            p.out = static_cast<size_t>(it.shape[0]);
            p.inp = static_cast<size_t>(it.shape[1]);
            const bool small_emb = is_small_embedding_name(name);
            const bool nat_ok = native_mode && !small_emb &&
                                ((p.is_embed_tokens && p.fmt == g6emb::FMT_FP8_ROW) ||
                                 (!p.is_embed_tokens &&
                                  (p.fmt == g6emb::FMT_TRELLIS || p.fmt == g6emb::FMT_FP8_ROW ||
                                   p.fmt == g6emb::FMT_INT4 || p.fmt == g6emb::FMT_BF16)));
            p.keep_native = nat_ok;
            if (p.keep_native) {
                if (p.is_lm_head) {
                    if (p.fmt == g6emb::FMT_FP8_ROW) {
                        p.ncomp = 2;
                        p.comp_bytes[0] = size_t(n_cand) * p.inp;
                        p.comp_bytes[1] = size_t(n_cand) * sizeof(float);
                        p.comp_off[0] = dense_total; dense_total += align256(p.comp_bytes[0]);
                        p.comp_off[1] = dense_total; dense_total += align256(p.comp_bytes[1]);
                        max_stage = std::max(max_stage, static_cast<size_t>(it.comps[0].nbytes));
                    } else if (p.fmt == g6emb::FMT_BF16) {
                        p.ncomp = 1;
                        p.comp_bytes[0] = size_t(n_cand) * p.inp * 2;
                        p.comp_off[0] = dense_total; dense_total += align256(p.comp_bytes[0]);
                        max_stage = std::max(max_stage, static_cast<size_t>(it.comps[0].nbytes));
                    } else {
                        p.keep_native = false;
                    }
                } else if (p.is_res_head) {
                    const size_t keep_rows = std::min<size_t>(p.out, geom::RES_CB_SIZE);
                    if (p.fmt == g6emb::FMT_FP8_ROW) {
                        p.ncomp = 2;
                        p.comp_bytes[0] = keep_rows * p.inp;
                        p.comp_bytes[1] = keep_rows * sizeof(float);
                        p.comp_off[0] = dense_total; dense_total += align256(p.comp_bytes[0]);
                        p.comp_off[1] = dense_total; dense_total += align256(p.comp_bytes[1]);
                        max_stage = std::max(max_stage, static_cast<size_t>(it.comps[0].nbytes));
                    } else if (p.fmt == g6emb::FMT_BF16) {
                        p.ncomp = 1;
                        p.comp_bytes[0] = keep_rows * p.inp * 2;
                        p.comp_off[0] = dense_total; dense_total += align256(p.comp_bytes[0]);
                        max_stage = std::max(max_stage, static_cast<size_t>(it.comps[0].nbytes));
                    } else {
                        p.keep_native = false;
                    }
                } else {
                    for (uint32_t c = 0; c < p.ncomp; ++c) {
                        p.comp_off[c] = dense_total;
                        p.comp_bytes[c] = static_cast<size_t>(it.comps[c].nbytes);
                        dense_total += align256(p.comp_bytes[c]);
                        max_stage = std::max(max_stage, p.comp_bytes[c]);
                    }
                }
            }
            if (!p.keep_native) {
                p.decode_f32 = ref_f32 && !small_emb;
                const size_t elem_b = p.decode_f32 ? 4u : 2u;
                const size_t full_bytes = p.out * p.inp * elem_b;
                p.dev_off = dense_total;
                p.dev_bytes = p.is_lm_head ? (size_t(n_cand) * p.inp * elem_b) : full_bytes;
                dense_total += align256(p.dev_bytes);
                max_stage = std::max(max_stage, full_bytes);
            }
        } else {
            p.raw_dtype = it.dtype ? it.dtype : "BF16";
            p.raw_bytes = static_cast<size_t>(it.comps[0].nbytes);
            size_t elems = 1;
            for (uint32_t d = 0; d < it.ndim; ++d) elems *= static_cast<size_t>(it.shape[d]);
            p.dev_off = dense_total;
            p.dev_bytes = elems * sizeof(float);
            dense_total += align256(p.dev_bytes);
            max_stage = std::max(max_stage, std::max(p.raw_bytes, p.dev_bytes));
        }
        dplans.push_back(std::move(p));
    }

    W.dense_bytes = std::max<size_t>(256, dense_total);
    TTS_CK(cudaMalloc(&W.dense_arena, W.dense_bytes));

    // 2. Plan expert arenas (native trellis via dr3_plan + fallback Ref FP8_BLOCK/FP16 arena for remaining layers)
    const int E = W.cfg.num_experts;
    const int H = W.cfg.hidden_size;
    const int I = W.cfg.moe_intermediate;
    const int bo_gu = (I + 127) / 128;
    const int bi_gu = (H + 127) / 128;
    const int bo_dn = (H + 127) / 128;
    const int bi_dn = (I + 127) / 128;

    struct LayerExpPlan {
        int store = -1;
        bool use_tr3 = false;
        bool use_fp8 = false;
        bool shared_from_store = false;
        size_t off_g = 0, off_g_sc = 0;
        size_t off_u = 0, off_u_sc = 0;
        size_t off_d = 0, off_d_sc = 0;
        size_t off_sg = 0, off_su = 0, off_sd = 0;
    };
    std::vector<LayerExpPlan> eplans(static_cast<size_t>(L));
    size_t exp_total = 0;
    const size_t sh_f16 = size_t(W.cfg.shared_intermediate) * H * 2;
    for (int l = 0; l < L; ++l) {
        if (layer_is_tr3[size_t(l)]) {
            eplans[size_t(l)].store = 1;
            eplans[size_t(l)].use_tr3 = true;
            ++W.n_tr3_expert_layers;
            continue;
        }
        const int st = g6l_tts_expert_store(h, uint32_t(l));
        if (st < 0) {
            throw std::runtime_error("g6tts: layer " + std::to_string(l) + " has no expert store");
        }
        eplans[size_t(l)].store = st;
        const bool use_fp8 = (st == 0) && (mode != WeightMode::RefFp16);
        eplans[size_t(l)].use_fp8 = use_fp8;
        if (use_fp8) {
            const size_t gu_fp8 = size_t(E) * I * H;
            const size_t gu_sc = size_t(E) * bo_gu * bi_gu * sizeof(float);
            const size_t dn_fp8 = size_t(E) * H * I;
            const size_t dn_sc = size_t(E) * bo_dn * bi_dn * sizeof(float);
            eplans[size_t(l)].off_g = exp_total; exp_total += align256(gu_fp8);
            eplans[size_t(l)].off_g_sc = exp_total; exp_total += align256(gu_sc);
            eplans[size_t(l)].off_u = exp_total; exp_total += align256(gu_fp8);
            eplans[size_t(l)].off_u_sc = exp_total; exp_total += align256(gu_sc);
            eplans[size_t(l)].off_d = exp_total; exp_total += align256(dn_fp8);
            eplans[size_t(l)].off_d_sc = exp_total; exp_total += align256(dn_sc);
            ++W.n_fp8_expert_layers;
        } else {
            const size_t mat_f16 = size_t(E) * I * H * 2;
            eplans[size_t(l)].off_g = exp_total; exp_total += align256(mat_f16);
            eplans[size_t(l)].off_u = exp_total; exp_total += align256(mat_f16);
            eplans[size_t(l)].off_d = exp_total; exp_total += align256(mat_f16);
            ++W.n_f16_expert_layers;
        }
        if (st == 1) {
            eplans[size_t(l)].shared_from_store = true;
            eplans[size_t(l)].off_sg = exp_total; exp_total += align256(sh_f16);
            eplans[size_t(l)].off_su = exp_total; exp_total += align256(sh_f16);
            eplans[size_t(l)].off_sd = exp_total; exp_total += align256(sh_f16);
        }
    }

    if (exp_total > 0) {
        W.expert_bytes = exp_total;
        TTS_CK(cudaMalloc(&W.expert_arena, W.expert_bytes));
    }
    if (dr3_plan_h && dr3_man.pack_bytes > 0) {
        W.tr3_expert_bytes = static_cast<size_t>(dr3_man.pack_bytes);
        TTS_CK(cudaMalloc(&W.tr3_expert_arena, W.tr3_expert_bytes));
    }

    // Pinned staging buffer for dense + expert chunks
    const size_t one_exp_f16 = size_t(I) * H * 2;
    const size_t exp_chunk_e = 16;
    max_stage = std::max(max_stage, exp_chunk_e * one_exp_f16);

    void* stage = nullptr;
    TTS_CK(cudaHostAlloc(&stage, max_stage, cudaHostAllocDefault));
    struct PinGuard { void* p; ~PinGuard() { if (p) cudaFreeHost(p); } } pin_guard{stage};

    cudaStream_t st = nullptr;
    TTS_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } st_guard{st};

    // Upload main_cand_ids
    std::memcpy(stage, cand_ids.data(), cand_ids_bytes);
    TTS_CK(cudaMemcpyAsync(W.dense_arena + cand_ids_off, stage, cand_ids_bytes, cudaMemcpyHostToDevice, st));
    TTS_CK(cudaStreamSynchronize(st));
    W.main_cand_ids_dev = reinterpret_cast<const int32_t*>(W.dense_arena + cand_ids_off);

    // 3. Upload dense / vocab / tower items
    std::unordered_map<std::string, RefMat> mat_map;
    std::unordered_map<std::string, g6emb::DevMatrix> nat_map;
    std::unordered_map<std::string, const float*> vec_map;
    std::vector<uint8_t> raw_tmp;

    for (const auto& p : dplans) {
        if (p.is_matrix) {
            if (p.keep_native) {
                g6emb::DevMatrix dm;
                dm.fmt = p.fmt;
                dm.out = static_cast<int>(p.out);
                dm.inp = static_cast<int>(p.inp);
                dm.k = p.k;
                dm.side = p.side;

                if (p.is_lm_head) {
                    dm.out = n_cand;
                    if (p.fmt == g6emb::FMT_FP8_ROW) {
                        const size_t full_q = p.out * p.inp;
                        const size_t full_s = p.out * sizeof(float);
                        if (g6l_tts_dense_read_component(h, p.item_idx, 0, stage, full_q, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(lm_head fp8): " + std::string(err));
                        }
                        std::vector<uint8_t> sq(size_t(n_cand) * p.inp);
                        const uint8_t* src_q = static_cast<const uint8_t*>(stage);
                        for (int c = 0; c < n_cand; ++c) {
                            std::memcpy(sq.data() + size_t(c) * p.inp, src_q + size_t(cand_ids[size_t(c)]) * p.inp, p.inp);
                        }
                        std::memcpy(stage, sq.data(), sq.size());
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[0], stage, sq.size(), cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));

                        if (g6l_tts_dense_read_component(h, p.item_idx, 1, stage, full_s, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(lm_head scale): " + std::string(err));
                        }
                        std::vector<float> ss(static_cast<size_t>(n_cand));
                        const float* src_s = static_cast<const float*>(stage);
                        for (int c = 0; c < n_cand; ++c) ss[size_t(c)] = src_s[cand_ids[size_t(c)]];
                        std::memcpy(stage, ss.data(), ss.size() * sizeof(float));
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[1], stage, ss.size() * sizeof(float), cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));

                        dm.data = W.dense_arena + p.comp_off[0];
                        dm.scale = reinterpret_cast<const float*>(W.dense_arena + p.comp_off[1]);
                        ++W.n_nat_fp8_row;
                    } else { // FMT_BF16
                        const size_t full_b = p.out * p.inp * 2;
                        if (g6l_tts_dense_read_component(h, p.item_idx, 0, stage, full_b, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(lm_head bf16): " + std::string(err));
                        }
                        std::vector<uint8_t> sb(size_t(n_cand) * p.inp * 2);
                        const uint8_t* src_b = static_cast<const uint8_t*>(stage);
                        for (int c = 0; c < n_cand; ++c) {
                            std::memcpy(sb.data() + size_t(c) * p.inp * 2, src_b + size_t(cand_ids[size_t(c)]) * p.inp * 2, p.inp * 2);
                        }
                        std::memcpy(stage, sb.data(), sb.size());
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[0], stage, sb.size(), cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));
                        dm.bf16 = reinterpret_cast<const __nv_bfloat16*>(W.dense_arena + p.comp_off[0]);
                        ++W.n_nat_bf16;
                    }
                } else if (p.is_res_head) {
                    const size_t keep_rows = std::min<size_t>(p.out, geom::RES_CB_SIZE);
                    dm.out = static_cast<int>(keep_rows);
                    if (p.fmt == g6emb::FMT_FP8_ROW) {
                        const size_t full_q = p.out * p.inp;
                        const size_t full_s = p.out * sizeof(float);
                        if (g6l_tts_dense_read_component(h, p.item_idx, 0, stage, full_q, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(" + p.name + "): " + err);
                        }
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));
                        if (g6l_tts_dense_read_component(h, p.item_idx, 1, stage, full_s, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(" + p.name + " scale): " + err);
                        }
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[1], stage, p.comp_bytes[1], cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));
                        dm.data = W.dense_arena + p.comp_off[0];
                        dm.scale = reinterpret_cast<const float*>(W.dense_arena + p.comp_off[1]);
                        ++W.n_nat_fp8_row;
                    } else {
                        const size_t full_b = p.out * p.inp * 2;
                        if (g6l_tts_dense_read_component(h, p.item_idx, 0, stage, full_b, err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(" + p.name + "): " + err);
                        }
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));
                        dm.bf16 = reinterpret_cast<const __nv_bfloat16*>(W.dense_arena + p.comp_off[0]);
                        ++W.n_nat_bf16;
                    }
                } else if (p.fmt == g6emb::FMT_INT4 && p.side == g6emb::SIDE_IN) {
                    std::vector<uint8_t> tmp_q4(p.comp_bytes[0]);
                    std::vector<uint8_t> tmp_s4(p.comp_bytes[1]);
                    if (g6l_tts_dense_read_component(h, p.item_idx, 0, tmp_q4.data(), tmp_q4.size(), err, sizeof(err)) != 0 ||
                        g6l_tts_dense_read_component(h, p.item_idx, 1, tmp_s4.data(), tmp_s4.size(), err, sizeof(err)) != 0) {
                        throw std::runtime_error("g6l_tts_dense_read_component(" + p.name + "): " + err);
                    }
                    std::vector<uint8_t> rm_s4(p.comp_bytes[1]);
                    g6emb::transpose_int4_side_in(tmp_q4.data(), reinterpret_cast<const uint16_t*>(tmp_s4.data()),
                                                  static_cast<uint8_t*>(stage), reinterpret_cast<uint16_t*>(rm_s4.data()),
                                                  dm.out, dm.inp);
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
                    TTS_CK(cudaStreamSynchronize(st));
                    std::memcpy(stage, rm_s4.data(), p.comp_bytes[1]);
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[1], stage, p.comp_bytes[1], cudaMemcpyHostToDevice, st));
                    TTS_CK(cudaStreamSynchronize(st));
                    dm.side = g6emb::SIDE_OUT;
                    dm.data = W.dense_arena + p.comp_off[0];
                    dm.s4 = reinterpret_cast<const __half*>(W.dense_arena + p.comp_off[1]);
                    ++W.n_nat_int4;
                } else {
                    for (uint32_t c = 0; c < p.ncomp; ++c) {
                        if (g6l_tts_dense_read_component(h, p.item_idx, c, stage, p.comp_bytes[c], err, sizeof(err)) != 0) {
                            throw std::runtime_error("g6l_tts_dense_read_component(" + p.name + "): " + err);
                        }
                        TTS_CK(cudaMemcpyAsync(W.dense_arena + p.comp_off[c], stage, p.comp_bytes[c], cudaMemcpyHostToDevice, st));
                        TTS_CK(cudaStreamSynchronize(st));
                    }
                    if (p.fmt == g6emb::FMT_TRELLIS) {
                        if (p.k != 4) throw std::runtime_error("g6tts native: only K4 dense trellis supported (" + p.name + ")");
                        dm.data = W.dense_arena + p.comp_off[0];
                        dm.suh = reinterpret_cast<const __half*>(W.dense_arena + p.comp_off[1]);
                        dm.svh = reinterpret_cast<const __half*>(W.dense_arena + p.comp_off[2]);
                        constexpr int64_t tb = 128; // 32 * 4
                        const int64_t eb = (p.side == g6emb::SIDE_IN)
                                               ? (int64_t(p.out) / 16 * 8 * tb)
                                               : (int64_t(p.inp) / 16 * 8 * tb);
                        dm.sB = (p.side == g6emb::SIDE_IN) ? (64 * tb) : eb;
                        dm.sG = (p.side == g6emb::SIDE_IN) ? eb : (64 * tb);
                        ++W.n_nat_trellis;
                    } else if (p.fmt == g6emb::FMT_FP8_ROW) {
                        dm.data = W.dense_arena + p.comp_off[0];
                        dm.scale = reinterpret_cast<const float*>(W.dense_arena + p.comp_off[1]);
                        ++W.n_nat_fp8_row;
                    } else if (p.fmt == g6emb::FMT_INT4) {
                        dm.data = W.dense_arena + p.comp_off[0];
                        dm.s4 = reinterpret_cast<const __half*>(W.dense_arena + p.comp_off[1]);
                        ++W.n_nat_int4;
                    } else {
                        dm.bf16 = reinterpret_cast<const __nv_bfloat16*>(W.dense_arena + p.comp_off[0]);
                        ++W.n_nat_bf16;
                    }
                }
                nat_map[p.name] = dm;
            } else if (p.decode_f32) {
                const size_t full_f32 = p.out * p.inp * 4;
                if (g6l_tts_dense_decode_f32(h, p.item_idx, stage, full_f32, decode_threads, err, sizeof(err)) != 0) {
                    throw std::runtime_error(std::string("g6l_tts_dense_decode_f32(") + p.name + "): " + err);
                }
                if (p.is_lm_head) {
                    std::vector<uint8_t> sliced(size_t(n_cand) * p.inp * 4);
                    const uint8_t* src = static_cast<const uint8_t*>(stage);
                    for (int c = 0; c < n_cand; ++c) {
                        const int row = cand_ids[size_t(c)];
                        if (row < 0 || size_t(row) >= p.out) {
                            throw std::runtime_error("g6tts: lm_head candidate id out of range");
                        }
                        std::memcpy(sliced.data() + size_t(c) * p.inp * 4, src + size_t(row) * p.inp * 4, p.inp * 4);
                    }
                    std::memcpy(stage, sliced.data(), sliced.size());
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, sliced.size(), cudaMemcpyHostToDevice, st));
                } else {
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, full_f32, cudaMemcpyHostToDevice, st));
                }
                TTS_CK(cudaStreamSynchronize(st));
                mat_map[p.name] = RefMat{nullptr, reinterpret_cast<const float*>(W.dense_arena + p.dev_off)};
            } else {
                const size_t full_bf16 = p.out * p.inp * 2;
                if (g6l_tts_dense_decode_bf16(h, p.item_idx, stage, full_bf16, decode_threads, err, sizeof(err)) != 0) {
                    throw std::runtime_error(std::string("g6l_tts_dense_decode_bf16(") + p.name + "): " + err);
                }
                if (p.is_lm_head) {
                    std::vector<uint8_t> sliced(size_t(n_cand) * p.inp * 2);
                    const uint8_t* src = static_cast<const uint8_t*>(stage);
                    for (int c = 0; c < n_cand; ++c) {
                        const int row = cand_ids[size_t(c)];
                        if (row < 0 || size_t(row) >= p.out) {
                            throw std::runtime_error("g6tts: lm_head candidate id out of range");
                        }
                        std::memcpy(sliced.data() + size_t(c) * p.inp * 2, src + size_t(row) * p.inp * 2, p.inp * 2);
                    }
                    std::memcpy(stage, sliced.data(), sliced.size());
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, sliced.size(), cudaMemcpyHostToDevice, st));
                } else {
                    TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, full_bf16, cudaMemcpyHostToDevice, st));
                }
                TTS_CK(cudaStreamSynchronize(st));
                mat_map[p.name] = RefMat{reinterpret_cast<const __nv_bfloat16*>(W.dense_arena + p.dev_off), nullptr};
            }
        } else {
            raw_tmp.resize(p.raw_bytes);
            if (g6l_tts_dense_read_component(h, p.item_idx, 0, raw_tmp.data(), p.raw_bytes, err, sizeof(err)) != 0) {
                throw std::runtime_error(std::string("g6l_tts_dense_read_component(") + p.name + "): " + err);
            }
            float* dst = static_cast<float*>(stage);
            const size_t elems = p.dev_bytes / sizeof(float);
            if (p.raw_dtype == "F32") {
                std::memcpy(dst, raw_tmp.data(), p.dev_bytes);
            } else if (p.raw_dtype == "BF16") {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw_tmp.data());
                for (size_t k = 0; k < elems; ++k) dst[k] = host_bf16_to_f32(src[k]);
            } else if (p.raw_dtype == "F16") {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw_tmp.data());
                for (size_t k = 0; k < elems; ++k) dst[k] = host_f16_to_f32(src[k]);
            } else {
                throw std::runtime_error("g6tts: unsupported raw vector dtype " + p.raw_dtype + " for " + p.name);
            }
            TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, p.dev_bytes, cudaMemcpyHostToDevice, st));
            TTS_CK(cudaStreamSynchronize(st));
            vec_map[p.name] = reinterpret_cast<const float*>(W.dense_arena + p.dev_off);
        }
    }

    // 3b. Upload planned native trellis expert image (64 MiB pinned chunks via `g6l_plan_read`)
    std::unordered_map<std::string, PlannedTensorEntry> tr3_map;
    if (dr3_plan_h && W.tr3_expert_arena) {
        const size_t chunk_b = std::min<size_t>(max_stage, 64u << 20);
        for (uint64_t off = 0; off < dr3_man.pack_bytes; off += chunk_b) {
            const uint64_t n = std::min<uint64_t>(chunk_b, dr3_man.pack_bytes - off);
            if (g6l_plan_read(dr3_plan_h, off, n, stage, err, sizeof(err)) != 0) {
                throw std::runtime_error(std::string("g6l_plan_read(dr3) failed: ") + err);
            }
            TTS_CK(cudaMemcpyAsync(W.tr3_expert_arena + off, stage, static_cast<size_t>(n), cudaMemcpyHostToDevice, st));
            TTS_CK(cudaStreamSynchronize(st));
        }
        for (const auto& te : dr3_man.tensors) tr3_map[te.name] = te;
    }

    auto opt_ref = [&](const std::string& k) -> RefMat {
        auto it = mat_map.find(k);
        return it == mat_map.end() ? RefMat{} : it->second;
    };
    auto opt_nat = [&](const std::string& k) -> g6emb::DevMatrix {
        auto it = nat_map.find(k);
        return it == nat_map.end() ? g6emb::DevMatrix{} : it->second;
    };
    auto bind_proj = [&](const std::string& k, RefMat& dst_ref, g6emb::DevMatrix& dst_nat) {
        dst_ref = opt_ref(k);
        dst_nat = opt_nat(k);
        if (!dst_ref.valid() && !dst_nat.valid()) {
            throw std::runtime_error("g6tts: missing matrix " + k);
        }
    };
    auto req_bf16 = [&](const std::string& k) -> const __nv_bfloat16* {
        const RefMat m = opt_ref(k);
        if (!m.bf16) throw std::runtime_error("g6tts: missing BF16 matrix " + k);
        return m.bf16;
    };
    auto req_vec = [&](const std::string& k) -> const float* {
        auto it = vec_map.find(k);
        if (it == vec_map.end()) throw std::runtime_error("g6tts: missing vector " + k);
        return it->second;
    };

    bind_proj("model.language_model.embed_tokens.weight", W.embed_tokens, W.embed_tokens_nat);
    W.final_norm = req_vec("model.language_model.norm.weight");
    bind_proj("lm_head.weight", W.lm_head_cand, W.lm_head_nat);
    W.codebook_embeddings = req_bf16("audio_decoder.codebook_embeddings.weight");

    // 4. Bind backbone layers + upload fallback ref routed experts
    W.layers.resize(size_t(L));
    const unsigned n_workers = std::max(1u, std::min<unsigned>(decode_threads, 16u));

    for (int l = 0; l < L; ++l) {
        const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
        BackboneLayerW& LW = W.layers[size_t(l)];
        LW.is_full_attn = W.cfg.is_full_attn[size_t(l)];
        LW.input_ln = req_vec(p + "input_layernorm.weight");
        LW.post_attn_ln = req_vec(p + "post_attention_layernorm.weight");

        if (LW.is_full_attn) {
            bind_proj(p + "self_attn.q_proj.weight", LW.q_proj, LW.q_proj_nat);
            bind_proj(p + "self_attn.k_proj.weight", LW.k_proj, LW.k_proj_nat);
            bind_proj(p + "self_attn.v_proj.weight", LW.v_proj, LW.v_proj_nat);
            bind_proj(p + "self_attn.o_proj.weight", LW.o_proj, LW.o_proj_nat);
            LW.q_norm = req_vec(p + "self_attn.q_norm.weight");
            LW.k_norm = req_vec(p + "self_attn.k_norm.weight");
        } else {
            bind_proj(p + "linear_attn.in_proj_qkv.weight", LW.in_proj_qkv, LW.in_proj_qkv_nat);
            bind_proj(p + "linear_attn.in_proj_z.weight", LW.in_proj_z, LW.in_proj_z_nat);
            bind_proj(p + "linear_attn.in_proj_a.weight", LW.in_proj_a, LW.in_proj_a_nat);
            bind_proj(p + "linear_attn.in_proj_b.weight", LW.in_proj_b, LW.in_proj_b_nat);
            bind_proj(p + "linear_attn.out_proj.weight", LW.out_proj, LW.out_proj_nat);
            LW.conv1d = req_vec(p + "linear_attn.conv1d.weight");
            LW.A_log = req_vec(p + "linear_attn.A_log");
            LW.dt_bias = req_vec(p + "linear_attn.dt_bias");
            LW.gdn_norm = req_vec(p + "linear_attn.norm.weight");
        }

        bind_proj(p + "mlp.gate.weight", LW.router_gate, LW.router_gate_nat);
        bind_proj(p + "mlp.shared_expert_gate.weight", LW.shared_expert_gate, LW.shared_expert_gate_nat);

        const LayerExpPlan& ep = eplans[size_t(l)];
        if (ep.use_tr3) {
            char lbuf[16];
            std::snprintf(lbuf, sizeof(lbuf), "L%03d.tr3.", l);
            const std::string tp(lbuf);
            auto req_tr3 = [&](const char* suf) -> const PlannedTensorEntry& {
                auto it = tr3_map.find(tp + suf);
                if (it == tr3_map.end()) throw std::runtime_error("g6tts: missing planned tensor " + tp + suf);
                return it->second;
            };
            const auto& e_fc1 = req_tr3("fc1");
            const auto& e_fc2 = req_tr3("fc2");
            const auto& e_sg = req_tr3("sg");
            const auto& e_v1b = req_tr3("v1b");
            const auto& e_v2 = req_tr3("v2");
            const auto& e_v3b = req_tr3("v3b");
            const auto& e_tab = req_tr3("table");
            const auto& e_kmix = req_tr3("kmix");

            LW.tr3.valid = true;
            parse_expert_format(e_fc1.expert_format, LW.tr3.cb, LW.tr3.e4);
            LW.tr3.n_rows = static_cast<int>(e_tab.shape0);
            LW.tr3.has_shared = (LW.tr3.n_rows > E);
            LW.tr3.fc1 = W.tr3_expert_arena + e_fc1.offset;
            LW.tr3.fc2 = W.tr3_expert_arena + e_fc2.offset;
            LW.tr3.sg = reinterpret_cast<const __half*>(W.tr3_expert_arena + e_sg.offset);
            LW.tr3.v1b = reinterpret_cast<const __half*>(W.tr3_expert_arena + e_v1b.offset);
            LW.tr3.v2 = reinterpret_cast<const __half*>(W.tr3_expert_arena + e_v2.offset);
            LW.tr3.v3b = reinterpret_cast<const __half*>(W.tr3_expert_arena + e_v3b.offset);
            LW.tr3.tab = reinterpret_cast<const int64_t*>(W.tr3_expert_arena + e_tab.offset);
            LW.tr3.kmix = reinterpret_cast<const int64_t*>(W.tr3_expert_arena + e_kmix.offset);
            if (!LW.tr3.has_shared) {
                bind_proj(p + "mlp.shared_expert.gate_proj.weight", LW.shared_gate_proj, LW.shared_gate_nat);
                bind_proj(p + "mlp.shared_expert.up_proj.weight", LW.shared_up_proj, LW.shared_up_nat);
                bind_proj(p + "mlp.shared_expert.down_proj.weight", LW.shared_down_proj, LW.shared_down_nat);
            }
            continue;
        }

        // Fallback non-trellis shared expert
        const bool has_dense_sh = (mat_map.count(p + "mlp.shared_expert.gate_proj.weight") > 0) ||
                                  (nat_map.count(p + "mlp.shared_expert.gate_proj.weight") > 0);
        if (has_dense_sh) {
            bind_proj(p + "mlp.shared_expert.gate_proj.weight", LW.shared_gate_proj, LW.shared_gate_nat);
            bind_proj(p + "mlp.shared_expert.up_proj.weight", LW.shared_up_proj, LW.shared_up_nat);
            bind_proj(p + "mlp.shared_expert.down_proj.weight", LW.shared_down_proj, LW.shared_down_nat);
        } else if (ep.shared_from_store) {
            const size_t sh_offs[3] = {ep.off_sg, ep.off_su, ep.off_sd};
            for (int proj = 0; proj < 3; ++proj) {
                if (g6l_tts_expert_decode_f16(h, uint32_t(l), uint32_t(E), uint32_t(proj),
                                              stage, sh_f16, err, sizeof(err)) != 0) {
                    throw std::runtime_error("g6tts: shared expert decode failed at layer " +
                                             std::to_string(l) + ": " + err);
                }
                host_f16_buf_to_bf16_inplace(stage, sh_f16 / 2);
                TTS_CK(cudaMemcpyAsync(W.expert_arena + sh_offs[proj], stage, sh_f16, cudaMemcpyHostToDevice, st));
                TTS_CK(cudaStreamSynchronize(st));
            }
            LW.shared_gate_proj = RefMat{reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_sg), nullptr};
            LW.shared_up_proj = RefMat{reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_su), nullptr};
            LW.shared_down_proj = RefMat{reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_sd), nullptr};
        } else {
            throw std::runtime_error("g6tts: shared expert missing at layer " + std::to_string(l));
        }

        LW.experts_fp8 = ep.use_fp8;
        if (ep.use_fp8) {
            LW.exp_gate_fp8 = W.expert_arena + ep.off_g;
            LW.exp_gate_sc = reinterpret_cast<const float*>(W.expert_arena + ep.off_g_sc);
            LW.exp_up_fp8 = W.expert_arena + ep.off_u;
            LW.exp_up_sc = reinterpret_cast<const float*>(W.expert_arena + ep.off_u_sc);
            LW.exp_down_fp8 = W.expert_arena + ep.off_d;
            LW.exp_down_sc = reinterpret_cast<const float*>(W.expert_arena + ep.off_d_sc);

            const size_t fp8_sz = size_t(I) * H;
            const size_t sc_sz_gu = size_t(bo_gu) * bi_gu * sizeof(float);
            const size_t sc_sz_dn = size_t(bo_dn) * bi_dn * sizeof(float);
            const size_t dev_fp8_offs[3] = {ep.off_g, ep.off_u, ep.off_d};
            const size_t dev_sc_offs[3] = {ep.off_g_sc, ep.off_u_sc, ep.off_d_sc};
            const size_t sc_sizes[3] = {sc_sz_gu, sc_sz_gu, sc_sz_dn};

            std::vector<uint8_t> sc_buf;
            for (int proj = 0; proj < 3; ++proj) {
                const size_t sc_sz = sc_sizes[proj];
                sc_buf.resize(size_t(E) * sc_sz);
                for (int e0 = 0; e0 < E; e0 += int(exp_chunk_e)) {
                    const int ec = std::min(int(exp_chunk_e), E - e0);
                    std::atomic<int> failed{0};
                    std::string first_err;
                    std::vector<std::thread> threads;
                    const unsigned nw = std::min<unsigned>(n_workers, unsigned(ec));
                    threads.reserve(nw);
                    for (unsigned t = 0; t < nw; ++t) {
                        threads.emplace_back([&, t] {
                            char terr[256] = {};
                            for (int idx = int(t); idx < ec; idx += int(nw)) {
                                if (failed.load(std::memory_order_relaxed)) return;
                                const int e = e0 + idx;
                                uint8_t* dst_f = static_cast<uint8_t*>(stage) + size_t(idx) * fp8_sz;
                                uint8_t* dst_s = sc_buf.data() + size_t(e) * sc_sz;
                                if (g6l_tts_expert_read_fp8(h, uint32_t(l), uint32_t(e), uint32_t(proj),
                                                            dst_f, fp8_sz, dst_s, sc_sz, terr, sizeof(terr)) != 0) {
                                    if (failed.exchange(1) == 0) first_err = terr;
                                    return;
                                }
                            }
                        });
                    }
                    for (auto& th : threads) th.join();
                    if (failed.load()) {
                        throw std::runtime_error("g6l_tts_expert_read_fp8 failed: " + first_err);
                    }
                    TTS_CK(cudaMemcpyAsync(W.expert_arena + dev_fp8_offs[proj] + size_t(e0) * fp8_sz,
                                           stage, size_t(ec) * fp8_sz, cudaMemcpyHostToDevice, st));
                    TTS_CK(cudaStreamSynchronize(st));
                }
                std::memcpy(stage, sc_buf.data(), sc_buf.size());
                TTS_CK(cudaMemcpyAsync(W.expert_arena + dev_sc_offs[proj], stage, sc_buf.size(), cudaMemcpyHostToDevice, st));
                TTS_CK(cudaStreamSynchronize(st));
            }
        } else {
            LW.exp_gate_bf16 = reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_g);
            LW.exp_up_bf16 = reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_u);
            LW.exp_down_bf16 = reinterpret_cast<const __nv_bfloat16*>(W.expert_arena + ep.off_d);
            const size_t dev_offs[3] = {ep.off_g, ep.off_u, ep.off_d};

            for (int proj = 0; proj < 3; ++proj) {
                for (int e0 = 0; e0 < E; e0 += int(exp_chunk_e)) {
                    const int ec = std::min(int(exp_chunk_e), E - e0);
                    std::atomic<int> failed{0};
                    std::string first_err;
                    std::vector<std::thread> threads;
                    const unsigned nw = std::min<unsigned>(n_workers, unsigned(ec));
                    threads.reserve(nw);
                    for (unsigned t = 0; t < nw; ++t) {
                        threads.emplace_back([&, t] {
                            char terr[256] = {};
                            for (int idx = int(t); idx < ec; idx += int(nw)) {
                                if (failed.load(std::memory_order_relaxed)) return;
                                const int e = e0 + idx;
                                uint8_t* dst = static_cast<uint8_t*>(stage) + size_t(idx) * one_exp_f16;
                                if (g6l_tts_expert_decode_f16(h, uint32_t(l), uint32_t(e), uint32_t(proj),
                                                              dst, one_exp_f16, terr, sizeof(terr)) != 0) {
                                    if (failed.exchange(1) == 0) first_err = terr;
                                    return;
                                }
                                host_f16_buf_to_bf16_inplace(dst, one_exp_f16 / 2);
                            }
                        });
                    }
                    for (auto& th : threads) th.join();
                    if (failed.load()) {
                        throw std::runtime_error("g6l_tts_expert_decode_f16 failed: " + first_err);
                    }
                    TTS_CK(cudaMemcpyAsync(W.expert_arena + dev_offs[proj] + size_t(e0) * one_exp_f16,
                                           stage, size_t(ec) * one_exp_f16, cudaMemcpyHostToDevice, st));
                    TTS_CK(cudaStreamSynchronize(st));
                }
            }
        }
    }

    // 5. Bind Codebook-MTP tower layers & heads
    auto bind_tower = [&](const std::string& p, bool is_shared) -> TowerLayerW {
        TowerLayerW T;
        T.is_shared = is_shared;
        T.embedding = req_bf16(p + (is_shared ? "semantic_embedding.weight" : "codebook_embedding.weight"));
        T.enorm = req_vec(p + "enorm.weight");
        T.hnorm = req_vec(p + "hnorm.weight");
        bind_proj(p + "eh_proj.weight", T.eh_proj, T.eh_proj_nat);
        T.attn_norm = req_vec(p + "transformer_layer.attention_norm.weight");
        bind_proj(p + "transformer_layer.attention.wqkv.weight", T.wqkv, T.wqkv_nat);
        bind_proj(p + "transformer_layer.attention.wo.weight", T.wo, T.wo_nat);
        T.q_norm = req_vec(p + "transformer_layer.attention.q_norm.weight");
        T.k_norm = req_vec(p + "transformer_layer.attention.k_norm.weight");
        T.ffn_norm = req_vec(p + "transformer_layer.ffn_norm.weight");
        bind_proj(p + "transformer_layer.feed_forward.w1.weight", T.w1, T.w1_nat);
        bind_proj(p + "transformer_layer.feed_forward.w3.weight", T.w3, T.w3_nat);
        bind_proj(p + "transformer_layer.feed_forward.w2.weight", T.w2, T.w2_nat);
        T.final_ln = req_vec(p + "final_layernorm.weight");
        return T;
    };

    W.shared_layers.resize(size_t(W.cfg.tower_shared));
    for (int l = 0; l < W.cfg.tower_shared; ++l) {
        W.shared_layers[size_t(l)] = bind_tower("audio_decoder_mtp_shared.layers." + std::to_string(l) + ".", true);
    }
    W.res_layers.resize(size_t(W.cfg.tower_res));
    W.heads.resize(size_t(W.cfg.tower_res));
    W.heads_nat.resize(size_t(W.cfg.tower_res));
    for (int l = 0; l < W.cfg.tower_res; ++l) {
        W.res_layers[size_t(l)] = bind_tower("audio_decoder_mtp.layers." + std::to_string(l) + ".", false);
        bind_proj("audio_decoder_mtp.heads." + std::to_string(l) + ".weight", W.heads[size_t(l)], W.heads_nat[size_t(l)]);
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "g6tts: loaded %s (%dL, %d+%d tower) in %.2f s: %.2f GiB VRAM "
                 "(dense=%.2f GiB [tr=%d fp8r=%d i4=%d bf16=%d], experts=%.2f GiB [tr3=%dL fp8=%dL f16=%dL], "
                 "mode=%s, verify=%d)\n",
                 W.cfg.model_name.c_str(), W.cfg.num_layers, W.cfg.tower_shared, W.cfg.tower_res, secs,
                 double(W.dense_bytes + W.expert_bytes + W.tr3_expert_bytes) / double(1ull << 30),
                 double(W.dense_bytes) / double(1ull << 30),
                 W.n_nat_trellis, W.n_nat_fp8_row, W.n_nat_int4, W.n_nat_bf16,
                 double(W.expert_bytes + W.tr3_expert_bytes) / double(1ull << 30),
                 W.n_tr3_expert_layers, W.n_fp8_expert_layers, W.n_f16_expert_layers,
                 weight_mode_name(W.mode), verify_checksums ? 1 : 0);
    return W;
}

} // namespace g6tts
