// Drama 3 (fish_omni) architecture constants, configuration parser, and weight-mode definitions for g6tts.
// Pure C++17 host header (no CUDA dependency) so host unit tests compile directly with g++.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../loader-rs/g6load.h"
#include "../runtime/json.h"

namespace g6tts {

// Verified Drama 3 (`fishaudio/drama-3` @ 78e5ab79) geometry constants.
namespace geom {
constexpr int HID = 2048;
constexpr int LAYERS = 40;
constexpr int FULL_ATTN_INTERVAL = 4; // layer l is full attention iff l % 4 == 3
constexpr int NEXP = 256;
constexpr int TOPK = 8;
constexpr int EXP_I = 512;
constexpr int SHARED_I = 512;

// Full attention
constexpr int HEADS = 16;
constexpr int KV_HEADS = 2;
constexpr int HEAD_DIM = 256;
constexpr int ROPE_DIM = 64;
constexpr float ROPE_THETA = 10000000.0f;
constexpr float RMS_EPS = 1e-6f;

// Gated DeltaNet (GDN)
constexpr int GDN_KH = 16;
constexpr int GDN_VH = 32;
constexpr int GDN_V_PER_K = 2;
constexpr int GDN_D = 128;
constexpr int GDN_CONV = 4;
constexpr int GDN_QKV_DIM = (2 * GDN_KH + GDN_VH) * GDN_D; // 8192
constexpr int GDN_VAL_DIM = GDN_VH * GDN_D;                // 4096

// Codebook-MTP tower
constexpr int TOW_SHARED = 4;
constexpr int TOW_RES = 9;
constexpr int TOW_LAYERS = TOW_SHARED + TOW_RES; // 13
constexpr int TOW_I = 6144;
constexpr int TOW_HEADS = 16;
constexpr int TOW_KV_HEADS = 2;
constexpr int TOW_HEAD_DIM = 256;
constexpr int TOW_ROPE_DIM = 64;
constexpr float TOW_ROPE_BASE = 10000000.0f;
constexpr float TOW_NORM_EPS = 1e-6f;
constexpr int TOW_VOCAB = 4096;
constexpr int RES_CB_SIZE = 1024;
constexpr int NCB = 10;
constexpr int ROW_WIDTH = 1 + NCB; // 11: [main, c0..c9]
constexpr int VQ_BACKBONE_CB = 1;  // c1 mixed into backbone input

// Vocabulary & special token IDs (verified from drama3-calib/contracts.json)
constexpr int VOCAB_SIZE = 252288;
constexpr int PAD = 248044;       // <|endoftext|>
constexpr int IM_START = 248045;  // <|im_start|>
constexpr int EOS = 248046;       // <|im_end|>
constexpr int VOICE = 248081;     // <|voice|>
constexpr int SPEAKER0 = 248083;  // <|speaker:0|>
constexpr int SEM0 = 248099;      // <|semantic:0|>
constexpr int SEM1 = 252194;      // <|semantic:4095|>
constexpr int MAIN_CANDIDATES = 1 + (SEM1 - SEM0 + 1); // 4097 = {EOS} U [SEM0..SEM1]

// Codec / streaming audio parameters
constexpr int SAMPLE_RATE = 44100;
constexpr int FRAME_SAMPLES = 2048;

// Sampling / RAS constants (fish-speech defaults)
constexpr int RAS_WINDOW = 16;
constexpr float RAS_TEMPERATURE = 1.0f;
constexpr int RAS_TOP_K = 30;
constexpr float RAS_TOP_P = 0.95f;
constexpr float DEFAULT_TEMPERATURE = 0.7f;
constexpr float DEFAULT_TOP_P = 0.7f;
constexpr int DEFAULT_TOP_K = 30;
} // namespace geom

enum class WeightMode {
    Ref = 0,     // BF16 dense + FP8-on-device routed experts when store==0 (~37 GB, dequant on the fly)
    RefFp16 = 1, // All weights (including routed experts) pre-decoded to BF16 on load (~72 GB)
    Native = 2,  // Keep native stored formats (trellis/FP8_ROW/INT4/BF16; Wave 3.2, falls back to Ref in Wave 1.6)
    RefF32 = 3,  // Exact FP32 dense weights + FP8-on-device routed experts (exact kernel reference)
};

inline const char* weight_mode_name(WeightMode m) {
    switch (m) {
        case WeightMode::Ref: return "ref";
        case WeightMode::RefFp16: return "ref-fp16";
        case WeightMode::Native: return "native";
        case WeightMode::RefF32: return "ref-f32";
    }
    return "unknown";
}

struct TtsConfig {
    std::string model_name = "drama-3";

    // Backbone geometry
    int vocab_size = geom::VOCAB_SIZE;
    int hidden_size = geom::HID;
    int num_layers = geom::LAYERS;
    int num_heads = geom::HEADS;
    int num_kv_heads = geom::KV_HEADS;
    int head_dim = geom::HEAD_DIM;
    int rotary_dim = geom::ROPE_DIM;
    float rope_theta = geom::ROPE_THETA;
    float rms_norm_eps = geom::RMS_EPS;
    int max_position_embeddings = 262144;

    // GDN geometry
    int gdn_kh = geom::GDN_KH;
    int gdn_vh = geom::GDN_VH;
    int gdn_kd = geom::GDN_D;
    int gdn_vd = geom::GDN_D;
    int gdn_conv = geom::GDN_CONV;

    // MoE geometry
    int num_experts = geom::NEXP;
    int top_k = geom::TOPK;
    int moe_intermediate = geom::EXP_I;
    int shared_intermediate = geom::SHARED_I;
    bool norm_topk_prob = true;

    // Layer types: true = full attention, false = linear attention (GDN)
    std::vector<bool> is_full_attn;

    // Tower geometry
    int tower_shared = geom::TOW_SHARED;
    int tower_res = geom::TOW_RES;
    int tower_intermediate = geom::TOW_I;
    int tower_heads = geom::TOW_HEADS;
    int tower_kv_heads = geom::TOW_KV_HEADS;
    int tower_head_dim = geom::TOW_HEAD_DIM;
    int tower_rotary_dim = geom::TOW_ROPE_DIM;
    float tower_rope_base = geom::TOW_ROPE_BASE;
    float tower_norm_eps = geom::TOW_NORM_EPS;
    int tower_vocab = geom::TOW_VOCAB;
    int num_codebooks = geom::NCB;
    int vq_backbone_codebooks = geom::VQ_BACKBONE_CB;

    // Special tokens
    int eos_token_id = geom::EOS;
    int pad_token_id = geom::PAD;
    int semantic_start_id = geom::SEM0;
    int semantic_end_id = geom::SEM1;

    // Serving limits
    int cap = 4096;
    int max_streams = 8;

    TtsConfig() { init_default_layer_types(); }

    void init_default_layer_types() {
        is_full_attn.resize(size_t(num_layers));
        for (int l = 0; l < num_layers; ++l) {
            is_full_attn[size_t(l)] = ((l % geom::FULL_ATTN_INTERVAL) == (geom::FULL_ATTN_INTERVAL - 1));
        }
    }

    int row_width() const { return 1 + num_codebooks; }
    int main_candidates() const { return 1 + (semantic_end_id - semantic_start_id + 1); }

    bool is_semantic(int tok) const { return tok >= semantic_start_id && tok <= semantic_end_id; }

    // Sorted candidate token ids matching `fish_omni/tower.py`: `torch.cat((eos, band)).sort().values`.
    std::vector<int> build_main_candidate_ids() const {
        std::vector<int> ids;
        ids.reserve(size_t(main_candidates()));
        ids.push_back(eos_token_id);
        for (int t = semantic_start_id; t <= semantic_end_id; ++t) ids.push_back(t);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    // Parses config.json from `path` (if present). Throws on malformed JSON.
    bool apply_config_json(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) return false;
        std::ostringstream ss;
        ss << ifs.rdbuf();
        const g6v::Json root = g6v::Json::parse(ss.str());
        if (root.t != g6v::Json::Obj) {
            throw std::runtime_error("tts_config: config.json root is not an object: " + path);
        }

        auto read_int = [](const g6v::Json& obj, const char* k, int& dst) {
            if (const auto* v = obj.find(k); v && v->t == g6v::Json::Num) dst = static_cast<int>(v->n);
        };
        auto read_float = [](const g6v::Json& obj, const char* k, float& dst) {
            if (const auto* v = obj.find(k); v && v->t == g6v::Json::Num) dst = static_cast<float>(v->n);
        };
        auto read_bool = [](const g6v::Json& obj, const char* k, bool& dst) {
            if (const auto* v = obj.find(k); v && v->t == g6v::Json::Bool) dst = v->b;
        };

        read_int(root, "eos_token_id", eos_token_id);
        read_int(root, "pad_token_id", pad_token_id);
        read_int(root, "semantic_start_token_id", semantic_start_id);
        read_int(root, "semantic_end_token_id", semantic_end_id);

        if (const auto* tc = root.find("text_config"); tc && tc->t == g6v::Json::Obj) {
            read_int(*tc, "vocab_size", vocab_size);
            read_int(*tc, "hidden_size", hidden_size);
            read_int(*tc, "num_hidden_layers", num_layers);
            read_int(*tc, "num_attention_heads", num_heads);
            read_int(*tc, "num_key_value_heads", num_kv_heads);
            read_int(*tc, "head_dim", head_dim);
            read_float(*tc, "rms_norm_eps", rms_norm_eps);
            read_int(*tc, "max_position_embeddings", max_position_embeddings);
            read_int(*tc, "linear_conv_kernel_dim", gdn_conv);
            read_int(*tc, "linear_key_head_dim", gdn_kd);
            read_int(*tc, "linear_value_head_dim", gdn_vd);
            read_int(*tc, "linear_num_key_heads", gdn_kh);
            read_int(*tc, "linear_num_value_heads", gdn_vh);
            read_int(*tc, "num_experts", num_experts);
            read_int(*tc, "num_experts_per_tok", top_k);
            read_int(*tc, "moe_intermediate_size", moe_intermediate);
            read_int(*tc, "shared_expert_intermediate_size", shared_intermediate);
            read_bool(*tc, "norm_topk_prob", norm_topk_prob);

            float prf = 0.25f;
            read_float(*tc, "partial_rotary_factor", prf);
            read_float(*tc, "rope_theta", rope_theta);
            if (const auto* rp = tc->find("rope_parameters"); rp && rp->t == g6v::Json::Obj) {
                read_float(*rp, "rope_theta", rope_theta);
                read_float(*rp, "partial_rotary_factor", prf);
            }
            rotary_dim = static_cast<int>(float(head_dim) * prf);

            if (const auto* lt = tc->find("layer_types"); lt && lt->t == g6v::Json::Arr && !lt->a.empty()) {
                is_full_attn.clear();
                is_full_attn.reserve(lt->a.size());
                for (const auto& entry : lt->a) {
                    if (entry.t != g6v::Json::Str) throw std::runtime_error("tts_config: non-string layer_types entry");
                    if (entry.s == "full_attention") is_full_attn.push_back(true);
                    else if (entry.s == "linear_attention") is_full_attn.push_back(false);
                    else throw std::runtime_error("tts_config: unknown layer type " + entry.s);
                }
            } else {
                init_default_layer_types();
            }
        }

        if (const auto* ac = root.find("audio_decoder_config"); ac && ac->t == g6v::Json::Obj) {
            read_int(*ac, "intermediate_size", tower_intermediate);
            read_int(*ac, "n_head", tower_heads);
            read_int(*ac, "n_local_heads", tower_kv_heads);
            read_int(*ac, "head_dim", tower_head_dim);
            read_float(*ac, "norm_eps", tower_norm_eps);
            read_float(*ac, "rope_base", tower_rope_base);
            float rp = 0.25f;
            read_float(*ac, "rotary_percent", rp);
            tower_rotary_dim = static_cast<int>(float(tower_head_dim) * rp);
            read_int(*ac, "vocab_size", tower_vocab);
            read_int(*ac, "num_codebooks", num_codebooks);
            read_int(*ac, "shared_enrichment_layers", tower_shared);
            read_int(*ac, "vq_backbone_codebooks", vq_backbone_codebooks);
            tower_res = num_codebooks - 1;
        }
        return true;
    }

    // Checks config against compile-time Drama 3 constants.
    void validate_drama3_geometry() const {
        auto eq = [](const char* name, int got, int want) {
            if (got != want) {
                throw std::runtime_error(std::string("g6tts geometry mismatch for ") + name + ": got " +
                                         std::to_string(got) + ", expected " + std::to_string(want));
            }
        };
        eq("hidden_size", hidden_size, geom::HID);
        eq("num_layers", num_layers, geom::LAYERS);
        eq("num_experts", num_experts, geom::NEXP);
        eq("top_k", top_k, geom::TOPK);
        eq("moe_intermediate", moe_intermediate, geom::EXP_I);
        eq("shared_intermediate", shared_intermediate, geom::SHARED_I);
        eq("num_heads", num_heads, geom::HEADS);
        eq("num_kv_heads", num_kv_heads, geom::KV_HEADS);
        eq("head_dim", head_dim, geom::HEAD_DIM);
        eq("rotary_dim", rotary_dim, geom::ROPE_DIM);
        eq("gdn_kh", gdn_kh, geom::GDN_KH);
        eq("gdn_vh", gdn_vh, geom::GDN_VH);
        eq("gdn_kd", gdn_kd, geom::GDN_D);
        eq("gdn_vd", gdn_vd, geom::GDN_D);
        eq("gdn_conv", gdn_conv, geom::GDN_CONV);
        eq("tower_shared", tower_shared, geom::TOW_SHARED);
        eq("tower_res", tower_res, geom::TOW_RES);
        eq("tower_intermediate", tower_intermediate, geom::TOW_I);
        eq("tower_heads", tower_heads, geom::TOW_HEADS);
        eq("tower_kv_heads", tower_kv_heads, geom::TOW_KV_HEADS);
        eq("tower_head_dim", tower_head_dim, geom::TOW_HEAD_DIM);
        eq("tower_rotary_dim", tower_rotary_dim, geom::TOW_ROPE_DIM);
        eq("tower_vocab", tower_vocab, geom::TOW_VOCAB);
        eq("num_codebooks", num_codebooks, geom::NCB);
        eq("vq_backbone_codebooks", vq_backbone_codebooks, geom::VQ_BACKBONE_CB);
        eq("semantic_start_id", semantic_start_id, geom::SEM0);
        eq("semantic_end_id", semantic_end_id, geom::SEM1);
        eq("eos_token_id", eos_token_id, geom::EOS);
        if (int(is_full_attn.size()) != num_layers) {
            throw std::runtime_error("g6tts: layer_types length mismatch");
        }
        for (int l = 0; l < num_layers; ++l) {
            const bool want = (l % geom::FULL_ATTN_INTERVAL) == (geom::FULL_ATTN_INTERVAL - 1);
            if (is_full_attn[size_t(l)] != want) {
                throw std::runtime_error("g6tts: unexpected layer_types at layer " + std::to_string(l));
            }
        }
    }

    static TtsConfig from_release(const void* h, const std::string& release_dir, bool strict = true) {
        TtsConfig cfg;
        const char* prof = h ? g6l_tts_profile(h) : nullptr;
        if (prof && std::string(prof) != "fish_omni") {
            throw std::runtime_error(std::string("g6tts: unsupported exlr8 profile: ") + prof);
        }
        cfg.apply_config_json(release_dir + "/config.json");
        if (h) {
            // Confirm vocab_size from release embed_tokens if config.json was absent
            const size_t nitems = g6l_tts_dense_item_count(h);
            for (size_t i = 0; i < nitems; ++i) {
                g6l_embed_item it{};
                if (g6l_tts_dense_item_info(h, i, &it) == 0 && it.hf_name &&
                    std::string(it.hf_name) == "model.language_model.embed_tokens.weight") {
                    cfg.vocab_size = static_cast<int>(it.shape[0]);
                    break;
                }
            }
        }
        if (strict) cfg.validate_drama3_geometry();
        return cfg;
    }
};

} // namespace g6tts
