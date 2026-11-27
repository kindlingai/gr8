// Configuration, geometry inference, and 3D M-RoPE position builder for the native g6embed dense embedding engine.
// Pure C++17 host header (no CUDA dependency) so it is unit-tested directly under g++ CPU gates.
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

namespace g6emb {

enum class EmbedProfile {
    Qwen3Vl = 0,
    EmbeddingGemma2 = 1,
};

enum class WeightMode {
    Native = 0,     // Keep packed TRELLIS K4 / INT4 / FP8_ROW / BF16 on GPU (minimum VRAM)
    Bf16 = 1,       // Decode quantized matrices to BF16 on load (exactness debug / cuBLAS baseline)
};

// Image span parsed from PROTOCOL.md v1 `embed.images`.
struct EmbedImage {
    int pos = 0;                 // First <|image_pad|> token index in prompt
    int vit_h = 0;               // Patch rows (multiple of merge_size=2)
    int vit_w = 0;               // Patch cols (multiple of merge_size=2)
    int span = 0;                // Merged visual tokens = (vit_h/2) * (vit_w/2)
    std::string key;
    std::vector<uint8_t> rgb;    // vit_h*16 * vit_w*16 * 3 uint8 RGB
};

struct MropePos {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct EmbedConfig {
    EmbedProfile profile = EmbedProfile::Qwen3Vl;
    std::string model_name = "qwen3-vl-embedding";

    // Text transformer geometry
    int vocab_size = 151936;
    int hidden_size = 2048;
    int intermediate_size = 6144;
    int num_text_layers = 28;
    int num_q_heads = 16;
    int num_kv_heads = 8;
    int head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 5000000.0f;
    std::array<int, 3> mrope_section = {24, 20, 20};
    int max_model_len = 16384;
    int eos_token_id = 151645;
    int image_token_id = 151655;

    // Vision tower geometry (Qwen3-VL)
    bool has_vision = false;
    int vit_layers = 0;
    int vit_hidden = 1024;
    int vit_intermediate = 4096;
    int vit_heads = 16;
    int vit_head_dim = 64;
    int vit_patch_size = 16;
    int vit_temporal_patch_size = 2;
    int vit_merge_size = 2;
    int vit_num_pos_embed = 2304;
    int vit_grid_size = 48;
    int vit_out_hidden = 2048;
    float vit_ln_eps = 1e-6f;
    float vit_rope_theta = 10000.0f;
    std::vector<int> deepstack_indexes; // 2B: {5, 11, 17}; 8B: {8, 16, 24}

    // Pooling & Matryoshka
    int embedding_dim = 2048;
    bool matryoshka = true;
    bool last_token_pool = true; // true: Qwen3-VL last non-pad token; false: EG2 mean pool + projection

    static constexpr int kQwen3VlPatch = 16;
    static constexpr int kQwen3VlMerge = 2;
    static constexpr int kQwen3VlMaxTokens = 4096; // production max_pixels=1310720 -> 1280 tokens; allow headroom

    static int qwen3vl_span_of(int vit_h, int vit_w) {
        if (vit_h < kQwen3VlMerge || vit_w < kQwen3VlMerge) return 0;
        if ((vit_h % kQwen3VlMerge) != 0 || (vit_w % kQwen3VlMerge) != 0) return 0;
        const int t = (vit_h / kQwen3VlMerge) * (vit_w / kQwen3VlMerge);
        if (t < 1 || t > kQwen3VlMaxTokens) return 0;
        return t;
    }

    static bool is_qwen3vl_8b_shape(int hidden, int text_layers) {
        return hidden == 4096 && text_layers == 36;
    }

    // Loads optional config.json overrides (when present alongside the exlr8 release).
    void apply_config_json(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) return;
        std::ostringstream ss;
        ss << ifs.rdbuf();
        const g6v::Json root = g6v::Json::parse(ss.str());
        if (root.t != g6v::Json::Obj) return;

        auto read_int = [](const g6v::Json& obj, const char* k, int& dst) {
            if (const auto* v = obj.find(k); v && v->t == g6v::Json::Num) dst = static_cast<int>(v->n);
        };
        auto read_float = [](const g6v::Json& obj, const char* k, float& dst) {
            if (const auto* v = obj.find(k); v && v->t == g6v::Json::Num) dst = static_cast<float>(v->n);
        };

        read_int(root, "image_token_id", image_token_id);
        if (const auto* tc = root.find("text_config"); tc && tc->t == g6v::Json::Obj) {
            read_int(*tc, "vocab_size", vocab_size);
            read_int(*tc, "hidden_size", hidden_size);
            read_int(*tc, "intermediate_size", intermediate_size);
            read_int(*tc, "num_hidden_layers", num_text_layers);
            read_int(*tc, "num_attention_heads", num_q_heads);
            read_int(*tc, "num_key_value_heads", num_kv_heads);
            read_int(*tc, "head_dim", head_dim);
            read_float(*tc, "rms_norm_eps", rms_norm_eps);
            read_float(*tc, "rope_theta", rope_theta);
            read_int(*tc, "eos_token_id", eos_token_id);
            if (const auto* rs = tc->find("rope_scaling"); rs && rs->t == g6v::Json::Obj) {
                if (const auto* ms = rs->find("mrope_section"); ms && ms->t == g6v::Json::Arr && ms->a.size() == 3) {
                    for (int i = 0; i < 3; ++i) mrope_section[i] = static_cast<int>(ms->a[size_t(i)].n);
                }
            }
            embedding_dim = hidden_size;
        }
        if (const auto* vc = root.find("vision_config"); vc && vc->t == g6v::Json::Obj) {
            read_int(*vc, "depth", vit_layers);
            read_int(*vc, "hidden_size", vit_hidden);
            read_int(*vc, "intermediate_size", vit_intermediate);
            read_int(*vc, "num_heads", vit_heads);
            read_int(*vc, "patch_size", vit_patch_size);
            read_int(*vc, "temporal_patch_size", vit_temporal_patch_size);
            read_int(*vc, "spatial_merge_size", vit_merge_size);
            read_int(*vc, "num_position_embeddings", vit_num_pos_embed);
            read_int(*vc, "out_hidden_size", vit_out_hidden);
            if (vit_heads > 0) vit_head_dim = vit_hidden / vit_heads;
            int g = 1;
            while (g * g < vit_num_pos_embed) ++g;
            vit_grid_size = g;
            if (const auto* ds = vc->find("deepstack_visual_indexes"); ds && ds->t == g6v::Json::Arr) {
                deepstack_indexes.clear();
                for (const auto& x : ds->a) deepstack_indexes.push_back(static_cast<int>(x.n));
            }
        }
    }

    // Infers authoritative dimensions directly from the discovered exlr8 release handle (and optional config.json).
    static EmbedConfig from_release(const void* h, const std::string& release_dir) {
        EmbedConfig cfg;
        const char* prof = g6l_embed_profile(h);
        const std::string prof_s = prof ? prof : "qwen3_vl";
        if (prof_s == "embedding_gemma2") {
            cfg.profile = EmbedProfile::EmbeddingGemma2;
            cfg.model_name = "embeddinggemma-2";
            cfg.last_token_pool = false;
        } else if (prof_s == "qwen3_vl") {
            cfg.profile = EmbedProfile::Qwen3Vl;
            cfg.model_name = "qwen3-vl-embedding";
            cfg.last_token_pool = true;
        } else {
            throw std::runtime_error("g6embed: unsupported exlr8 dense profile: " + prof_s);
        }

        // First apply config.json if present alongside the release.
        cfg.apply_config_json(release_dir + "/config.json");

        // Infer exact tensor shapes directly from the release items so stripped/synthetic releases work without config.json.
        int max_text_l = -1;
        int max_vis_l = -1;
        int max_ds_k = -1;
        int q_out = 0, k_out = 0;
        const size_t nitems = g6l_embed_item_count(h);
        for (size_t i = 0; i < nitems; ++i) {
            g6l_embed_item it{};
            if (g6l_embed_item_info(h, i, &it) != 0 || !it.hf_name) continue;
            const std::string name(it.hf_name);

            auto parse_layer = [&](const std::string& prefix) -> int {
                if (name.rfind(prefix, 0) != 0) return -1;
                size_t pos = prefix.size();
                size_t dot = name.find('.', pos);
                if (dot == std::string::npos || dot == pos) return -1;
                int v = 0;
                for (size_t k = pos; k < dot; ++k) {
                    if (name[k] < '0' || name[k] > '9') return -1;
                    v = v * 10 + (name[k] - '0');
                }
                return v;
            };

            if (name == "model.language_model.embed_tokens.weight" || name == "language_model.embed_tokens.weight") {
                cfg.vocab_size = static_cast<int>(it.shape[0]);
                cfg.hidden_size = static_cast<int>(it.shape[1]);
                cfg.embedding_dim = cfg.hidden_size;
            } else if (name == "language_model.embedding_projection.weight") {
                cfg.embedding_dim = static_cast<int>(it.shape[0]);
            }

            int tl = parse_layer("model.language_model.layers.");
            if (tl < 0) tl = parse_layer("language_model.layers.");
            if (tl >= 0) {
                max_text_l = std::max(max_text_l, tl);
                if (name.find(".self_attn.q_proj.weight") != std::string::npos) {
                    q_out = static_cast<int>(it.shape[0]);
                    cfg.hidden_size = static_cast<int>(it.shape[1]);
                } else if (name.find(".self_attn.k_proj.weight") != std::string::npos) {
                    k_out = static_cast<int>(it.shape[0]);
                } else if (name.find(".self_attn.q_norm.weight") != std::string::npos) {
                    cfg.head_dim = static_cast<int>(it.shape[0]);
                } else if (name.find(".mlp.gate_proj.weight") != std::string::npos) {
                    cfg.intermediate_size = static_cast<int>(it.shape[0]);
                }
            }

            int vl = parse_layer("model.visual.blocks.");
            if (vl >= 0) {
                cfg.has_vision = true;
                max_vis_l = std::max(max_vis_l, vl);
                if (name.find(".attn.proj.weight") != std::string::npos) {
                    cfg.vit_hidden = static_cast<int>(it.shape[0]);
                } else if (name.find(".mlp.linear_fc1.weight") != std::string::npos) {
                    cfg.vit_intermediate = static_cast<int>(it.shape[0]);
                }
            }
            if (name == "model.visual.pos_embed.weight") {
                cfg.has_vision = true;
                cfg.vit_num_pos_embed = static_cast<int>(it.shape[0]);
                cfg.vit_hidden = static_cast<int>(it.shape[1]);
                int g = 1;
                while (g * g < cfg.vit_num_pos_embed) ++g;
                cfg.vit_grid_size = g;
            } else if (name == "model.visual.merger.linear_fc2.weight") {
                cfg.vit_out_hidden = static_cast<int>(it.shape[0]);
            }
            int dsk = parse_layer("model.visual.deepstack_merger_list.");
            if (dsk >= 0) {
                max_ds_k = std::max(max_ds_k, dsk);
            }
        }

        if (max_text_l >= 0) cfg.num_text_layers = max_text_l + 1;
        if (max_vis_l >= 0) cfg.vit_layers = max_vis_l + 1;
        if (cfg.head_dim > 0) {
            if (q_out > 0) cfg.num_q_heads = q_out / cfg.head_dim;
            if (k_out > 0) cfg.num_kv_heads = k_out / cfg.head_dim;
        }
        if (cfg.profile == EmbedProfile::Qwen3Vl) {
            cfg.embedding_dim = cfg.hidden_size;
            cfg.vit_out_hidden = cfg.hidden_size;
            if (cfg.vit_layers > 0) {
                if (cfg.vit_heads <= 0 || cfg.vit_hidden % cfg.vit_heads != 0) {
                    cfg.vit_heads = (cfg.vit_hidden % 16 == 0) ? 16 : std::max(1, cfg.vit_hidden / 64);
                }
                cfg.vit_head_dim = cfg.vit_hidden / cfg.vit_heads;
            }
            if (cfg.deepstack_indexes.empty() && max_ds_k >= 0) {
                if (cfg.vit_layers == 24 && max_ds_k == 2) {
                    cfg.deepstack_indexes = {5, 11, 17};
                } else if (cfg.vit_layers == 27 && max_ds_k == 2) {
                    cfg.deepstack_indexes = {8, 16, 24};
                } else {
                    // Synthetic mini-model fallback: evenly spread taps within [0, vit_layers - 1]
                    for (int k = 0; k <= max_ds_k; ++k) {
                        const int idx = std::min(cfg.vit_layers - 1, ((k + 1) * cfg.vit_layers) / (max_ds_k + 2));
                        cfg.deepstack_indexes.push_back(idx);
                    }
                }
            }
            if (cfg.model_name == "qwen3-vl-embedding") {
                if (is_qwen3vl_8b_shape(cfg.hidden_size, cfg.num_text_layers)) {
                    cfg.model_name = "qwen3-vl-embedding-8b";
                } else if (cfg.hidden_size == 2048 && cfg.num_text_layers == 28) {
                    cfg.model_name = "qwen3-vl-embedding-2b";
                }
            }
        }
        return cfg;
    }
};

// Computes exact 3D interleaved M-RoPE (t, h, w) coordinates for one prompt sequence of length `seq_len`
// with ascending non-overlapping image spans, matching `Qwen3VLModel.get_rope_index` verbatim.
inline void build_qwen3vl_mrope_positions(int seq_len, const std::vector<EmbedImage>& images, MropePos* out) {
    int img_idx = 0;
    const int n_img = static_cast<int>(images.size());
    int delta = 0;
    for (int p = 0; p < seq_len; ++p) {
        while (img_idx < n_img && images[size_t(img_idx)].pos + images[size_t(img_idx)].span <= p) {
            const auto& im = images[size_t(img_idx)];
            const int lh = im.vit_h / EmbedConfig::kQwen3VlMerge;
            const int lw = im.vit_w / EmbedConfig::kQwen3VlMerge;
            delta += std::max(lh, lw) - im.span;
            ++img_idx;
        }
        if (img_idx < n_img && p >= images[size_t(img_idx)].pos) {
            const auto& im = images[size_t(img_idx)];
            const int lw = im.vit_w / EmbedConfig::kQwen3VlMerge;
            const int rel = p - im.pos;
            const int r = rel / lw;
            const int c = rel % lw;
            const int base = im.pos + delta;
            out[p] = {base, base + r, base + c};
        } else {
            const int cp = p + delta;
            out[p] = {cp, cp, cp};
        }
    }
}

} // namespace g6emb
