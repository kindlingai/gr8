// Multi-group exlr8 dense weight structs and 64K-page-safe pinned-staging GPU uploader for g6embed.
// Uses engine/loader-rs (g6l_embed_*) via pread -> pinned host staging -> cudaMemcpyAsync into one aligned device arena.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "embed_config.h"
#include "../loader-rs/g6load.h"

namespace g6emb {

// exlr8 v2 dense record format codes (matching engine/loader-rs/src/exlr8.rs)
constexpr int FMT_TRELLIS = 1;
constexpr int FMT_INT4 = 2;
constexpr int FMT_MXFP8 = 3;
constexpr int FMT_FP8_BLOCK = 4;
constexpr int FMT_FP8_ROW = 5;
constexpr int FMT_BF16 = 6;
constexpr int FMT_FP32 = 7;

constexpr int SIDE_REPLICATED = 0;
constexpr int SIDE_OUT = 1;
constexpr int SIDE_IN = 2;

inline void emb_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("g6embed CUDA error: ") + what + ": " + cudaGetErrorString(e));
    }
}
#define EMB_CK(x) ::g6emb::emb_ck((x), #x)

// One linear weight matrix W[out, inp] resident on GPU in its native exlr8 format (or BF16 in debug mode).
struct DevMatrix {
    int fmt = 0;         // FMT_TRELLIS | FMT_INT4 | FMT_MXFP8 | FMT_FP8_BLOCK | FMT_FP8_ROW | FMT_BF16 | FMT_FP32
    int out = 0;         // N (rows of HF weight)
    int inp = 0;         // K (cols of HF weight)
    int k = 0;           // trellis width (4)
    int side = 0;        // SIDE_REPLICATED | SIDE_OUT | SIDE_IN

    // Native component device pointers (only the fields for `fmt` are populated):
    const uint8_t* data = nullptr;       // trellis / q4 / fp8
    const __half* suh = nullptr;         // TRELLIS [inp]
    const __half* svh = nullptr;         // TRELLIS [out]
    const __half* s4 = nullptr;          // INT4 [out, inp/32] (always row-major on GPU!)
    const uint8_t* e8m0 = nullptr;       // MXFP8 [out, inp/32]
    const float* scale = nullptr;        // FP8_ROW [out] or FP8_BLOCK [ceil(out/128), ceil(inp/128)]
    const __nv_bfloat16* bf16 = nullptr; // FMT_BF16 [out, inp]
    const float* fp32 = nullptr;         // FMT_FP32 [out, inp]

    // Precomputed tile addressing strides for TRELLIS K4 (matching dtr.cuh):
    //   tile(nb, ks, c) = data + nb * sB + (ks >> 3) * sG + ((ks & 7) * 8 + c) * 128
    int64_t sB = 0;
    int64_t sG = 0;

    bool valid() const { return fmt != 0 && out > 0 && inp > 0; }
};

struct TextLayerW {
    const __nv_bfloat16* input_ln = nullptr;     // [H]
    const __nv_bfloat16* post_attn_ln = nullptr; // [H]
    const __nv_bfloat16* q_norm = nullptr;       // [head_dim]
    const __nv_bfloat16* k_norm = nullptr;       // [head_dim]
    DevMatrix q_proj;                            // [nq*hd, H]
    DevMatrix k_proj;                            // [nkv*hd, H]
    DevMatrix v_proj;                            // [nkv*hd, H]
    DevMatrix o_proj;                            // [H, nq*hd]
    DevMatrix gate_proj;                         // [I, H]
    DevMatrix up_proj;                           // [I, H]
    DevMatrix down_proj;                         // [H, I]
};

struct VisionBlockW {
    const __nv_bfloat16* norm1_w = nullptr;      // [H_v]
    const __nv_bfloat16* norm1_b = nullptr;      // [H_v]
    const __nv_bfloat16* norm2_w = nullptr;      // [H_v]
    const __nv_bfloat16* norm2_b = nullptr;      // [H_v]
    DevMatrix qkv_w;                             // [3*H_v, H_v]
    const __nv_bfloat16* qkv_b = nullptr;        // [3*H_v]
    DevMatrix proj_w;                            // [H_v, H_v]
    const __nv_bfloat16* proj_b = nullptr;       // [H_v]
    DevMatrix fc1_w;                             // [I_v, H_v] (Trellis K4 on 2B; FP8_ROW 4304 on 8B!)
    const __nv_bfloat16* fc1_b = nullptr;        // [I_v]
    DevMatrix fc2_w;                             // [H_v, I_v] (Trellis K4 on 2B; FP8_ROW 4304 on 8B!)
    const __nv_bfloat16* fc2_b = nullptr;        // [H_v]
};

struct MergerW {
    bool use_postshuffle_norm = false;
    const __nv_bfloat16* norm_w = nullptr;       // [H_v] (main) or [4*H_v] (deepstack)
    const __nv_bfloat16* norm_b = nullptr;
    DevMatrix fc1_w;                             // [4*H_v, 4*H_v]
    const __nv_bfloat16* fc1_b = nullptr;        // [4*H_v]
    DevMatrix fc2_w;                             // [H_text, 4*H_v]
    const __nv_bfloat16* fc2_b = nullptr;        // [H_text]
};

struct EmbedWeights {
    EmbedConfig cfg;
    WeightMode mode = WeightMode::Native;
    uint8_t* dev_arena = nullptr;
    size_t arena_bytes = 0;

    // Vocab / final norm
    DevMatrix embed_tokens;                      // [vocab, H] (FP8_ROW or BF16)
    const __nv_bfloat16* final_norm = nullptr;   // [H]
    DevMatrix eg2_projection;                    // optional [embedding_dim, H] for EG2

    // Text stack
    std::vector<TextLayerW> text_layers;

    // Vision stack (Qwen3-VL)
    const __nv_bfloat16* patch_proj_w = nullptr; // [H_v, 3, 2, 16, 16] flattened [H_v, 1536]
    const __nv_bfloat16* patch_proj_b = nullptr; // [H_v]
    const __nv_bfloat16* pos_embed = nullptr;    // [2304, H_v]
    std::vector<VisionBlockW> vision_blocks;
    MergerW main_merger;
    std::vector<MergerW> deepstack_mergers;

    // Format statistics for boot log
    int n_trellis = 0, n_int4 = 0, n_fp8_row = 0, n_bf16 = 0, n_other = 0;

    EmbedWeights() = default;
    EmbedWeights(const EmbedWeights&) = delete;
    EmbedWeights& operator=(const EmbedWeights&) = delete;
    EmbedWeights(EmbedWeights&& o) noexcept { *this = std::move(o); }
    EmbedWeights& operator=(EmbedWeights&& o) noexcept {
        if (this == &o) return *this;
        release();
        cfg = std::move(o.cfg);
        mode = o.mode;
        dev_arena = o.dev_arena; o.dev_arena = nullptr;
        arena_bytes = o.arena_bytes; o.arena_bytes = 0;
        embed_tokens = o.embed_tokens;
        final_norm = o.final_norm;
        eg2_projection = o.eg2_projection;
        text_layers = std::move(o.text_layers);
        patch_proj_w = o.patch_proj_w;
        patch_proj_b = o.patch_proj_b;
        pos_embed = o.pos_embed;
        vision_blocks = std::move(o.vision_blocks);
        main_merger = o.main_merger;
        deepstack_mergers = std::move(o.deepstack_mergers);
        n_trellis = o.n_trellis; n_int4 = o.n_int4; n_fp8_row = o.n_fp8_row; n_bf16 = o.n_bf16; n_other = o.n_other;
        return *this;
    }
    ~EmbedWeights() { release(); }

    void release() {
        if (dev_arena) {
            cudaFree(dev_arena);
            dev_arena = nullptr;
            arena_bytes = 0;
        }
    }
};

// Transposes INT4 side=IN [inp/128, out, 64] q4 & [inp/128, out, 4] s4 into row-major [out, inp/2] & [out, inp/32].
inline void transpose_int4_side_in(const uint8_t* src_q4, const uint16_t* src_s4,
                                   uint8_t* dst_q4, uint16_t* dst_s4, int out, int inp) {
    const int nb = inp / 128;
    const int q_row = inp / 2;
    const int s_row = inp / 32;
    for (int b = 0; b < nb; ++b) {
        for (int r = 0; r < out; ++r) {
            const size_t sb = (size_t(b) * out + r);
            std::memcpy(dst_q4 + size_t(r) * q_row + size_t(b) * 64, src_q4 + sb * 64, 64);
            std::memcpy(dst_s4 + size_t(r) * s_row + size_t(b) * 4, src_s4 + sb * 4, 4 * sizeof(uint16_t));
        }
    }
}

inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

// Loads a multi-group exlr8 dense release into a single aligned GPU arena through pinned host staging (64K-page safe).
inline EmbedWeights load_embed_weights(const std::string& release_dir, bool verify_checksums,
                                       WeightMode mode = WeightMode::Native, uint32_t decode_threads = 8) {
    const auto t0 = std::chrono::steady_clock::now();
    char err[512] = {};
    void* h = g6l_embed_open(release_dir.c_str(), verify_checksums ? 1 : 0, err, sizeof(err));
    if (!h) {
        throw std::runtime_error(std::string("g6l_embed_open failed: ") + err);
    }
    struct Guard { void* p; ~Guard() { g6l_embed_close(p); } } guard{h};

    EmbedWeights W;
    W.cfg = EmbedConfig::from_release(h, release_dir);
    W.mode = mode;

    const size_t nitems = g6l_embed_item_count(h);
    struct AllocPlan {
        size_t item_idx = 0;
        std::string name;
        bool is_matrix = false;
        int fmt = 0, out = 0, inp = 0, k = 0, side = 0;
        bool decode_to_bf16 = false;
        uint32_t ncomp = 0;
        size_t comp_off[3] = {0, 0, 0};
        size_t comp_bytes[3] = {0, 0, 0};
    };
    std::vector<AllocPlan> plans;
    plans.reserve(nitems);

    size_t total_bytes = 0;
    size_t max_stage_bytes = 64u << 20;
    for (size_t i = 0; i < nitems; ++i) {
        g6l_embed_item it{};
        if (g6l_embed_item_info(h, i, &it) != 0) throw std::runtime_error("g6l_embed_item_info failed");
        AllocPlan p;
        p.item_idx = i;
        p.name = it.hf_name;
        p.is_matrix = it.is_matrix != 0;
        p.fmt = it.fmt;
        p.side = it.side;
        p.k = it.k;
        if (p.is_matrix) {
            p.out = static_cast<int>(it.shape[0]);
            p.inp = static_cast<int>(it.shape[1]);
            if (mode == WeightMode::Bf16) {
                p.decode_to_bf16 = true;
                p.ncomp = 1;
                p.comp_off[0] = total_bytes;
                p.comp_bytes[0] = size_t(p.out) * size_t(p.inp) * 2;
                total_bytes += align256(p.comp_bytes[0]);
                max_stage_bytes = std::max(max_stage_bytes, p.comp_bytes[0]);
            } else {
                p.ncomp = it.ncomp;
                for (uint32_t c = 0; c < it.ncomp; ++c) {
                    p.comp_off[c] = total_bytes;
                    p.comp_bytes[c] = static_cast<size_t>(it.comps[c].nbytes);
                    total_bytes += align256(p.comp_bytes[c]);
                    max_stage_bytes = std::max(max_stage_bytes, p.comp_bytes[c]);
                }
            }
        } else {
            p.ncomp = 1;
            p.comp_off[0] = total_bytes;
            p.comp_bytes[0] = static_cast<size_t>(it.comps[0].nbytes);
            total_bytes += align256(p.comp_bytes[0]);
            max_stage_bytes = std::max(max_stage_bytes, p.comp_bytes[0]);
        }
        plans.push_back(std::move(p));
    }

    W.arena_bytes = std::max<size_t>(256, total_bytes);
    EMB_CK(cudaMalloc(&W.dev_arena, W.arena_bytes));

    // Allocate one reusable pinned host staging buffer (64K-page safe H2D DMA).
    void* stage = nullptr;
    EMB_CK(cudaHostAlloc(&stage, max_stage_bytes, cudaHostAllocDefault));
    struct PinGuard { void* p; ~PinGuard() { if (p) cudaFreeHost(p); } } pin_guard{stage};

    cudaStream_t st = nullptr;
    EMB_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } st_guard{st};

    std::unordered_map<std::string, DevMatrix> mat_map;
    std::unordered_map<std::string, const uint8_t*> raw_map;

    for (const auto& p : plans) {
        if (p.is_matrix) {
            DevMatrix dm;
            dm.out = p.out;
            dm.inp = p.inp;
            dm.k = p.k;
            dm.side = p.side;

            if (p.decode_to_bf16) {
                if (g6l_embed_decode_bf16(h, p.item_idx, stage, p.comp_bytes[0], decode_threads, err, sizeof(err)) != 0) {
                    throw std::runtime_error(std::string("g6l_embed_decode_bf16(") + p.name + "): " + err);
                }
                EMB_CK(cudaMemcpyAsync(W.dev_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
                EMB_CK(cudaStreamSynchronize(st));
                dm.fmt = FMT_BF16;
                dm.bf16 = reinterpret_cast<const __nv_bfloat16*>(W.dev_arena + p.comp_off[0]);
                ++W.n_bf16;
            } else if (p.fmt == FMT_INT4 && p.side == SIDE_IN) {
                // Transpose INT4 side=IN into uniform row-major [out, inp/2] & [out, inp/32] before H2D.
                std::vector<uint8_t> tmp_q4(p.comp_bytes[0]);
                std::vector<uint8_t> tmp_s4(p.comp_bytes[1]);
                if (g6l_embed_read_component(h, p.item_idx, 0, tmp_q4.data(), tmp_q4.size(), err, sizeof(err)) != 0 ||
                    g6l_embed_read_component(h, p.item_idx, 1, tmp_s4.data(), tmp_s4.size(), err, sizeof(err)) != 0) {
                    throw std::runtime_error(std::string("g6l_embed_read_component(") + p.name + "): " + err);
                }
                std::vector<uint8_t> rm_s4(p.comp_bytes[1]);
                transpose_int4_side_in(tmp_q4.data(), reinterpret_cast<const uint16_t*>(tmp_s4.data()),
                                       static_cast<uint8_t*>(stage), reinterpret_cast<uint16_t*>(rm_s4.data()),
                                       p.out, p.inp);
                EMB_CK(cudaMemcpyAsync(W.dev_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
                EMB_CK(cudaStreamSynchronize(st));
                std::memcpy(stage, rm_s4.data(), p.comp_bytes[1]);
                EMB_CK(cudaMemcpyAsync(W.dev_arena + p.comp_off[1], stage, p.comp_bytes[1], cudaMemcpyHostToDevice, st));
                EMB_CK(cudaStreamSynchronize(st));
                dm.fmt = FMT_INT4;
                dm.side = SIDE_OUT; // now row-major
                dm.data = W.dev_arena + p.comp_off[0];
                dm.s4 = reinterpret_cast<const __half*>(W.dev_arena + p.comp_off[1]);
                ++W.n_int4;
            } else {
                for (uint32_t c = 0; c < p.ncomp; ++c) {
                    if (g6l_embed_read_component(h, p.item_idx, c, stage, p.comp_bytes[c], err, sizeof(err)) != 0) {
                        throw std::runtime_error(std::string("g6l_embed_read_component(") + p.name + "): " + err);
                    }
                    EMB_CK(cudaMemcpyAsync(W.dev_arena + p.comp_off[c], stage, p.comp_bytes[c], cudaMemcpyHostToDevice, st));
                    EMB_CK(cudaStreamSynchronize(st));
                }
                dm.fmt = p.fmt;
                switch (p.fmt) {
                    case FMT_TRELLIS: {
                        if (p.k != 4) throw std::runtime_error("g6embed: only K4 trellis supported (" + p.name + ")");
                        dm.data = W.dev_arena + p.comp_off[0];
                        dm.suh = reinterpret_cast<const __half*>(W.dev_arena + p.comp_off[1]);
                        dm.svh = reinterpret_cast<const __half*>(W.dev_arena + p.comp_off[2]);
                        constexpr int64_t tb = 128; // 32 * 4
                        const int64_t eb = (p.side == SIDE_IN)
                                               ? (int64_t(p.out) / 16 * 8 * tb)
                                               : (int64_t(p.inp) / 16 * 8 * tb);
                        dm.sB = (p.side == SIDE_IN) ? (64 * tb) : eb;
                        dm.sG = (p.side == SIDE_IN) ? eb : (64 * tb);
                        ++W.n_trellis;
                        break;
                    }
                    case FMT_INT4:
                        dm.data = W.dev_arena + p.comp_off[0];
                        dm.s4 = reinterpret_cast<const __half*>(W.dev_arena + p.comp_off[1]);
                        ++W.n_int4;
                        break;
                    case FMT_MXFP8:
                        dm.data = W.dev_arena + p.comp_off[0];
                        dm.e8m0 = W.dev_arena + p.comp_off[1];
                        ++W.n_other;
                        break;
                    case FMT_FP8_BLOCK:
                    case FMT_FP8_ROW:
                        dm.data = W.dev_arena + p.comp_off[0];
                        dm.scale = reinterpret_cast<const float*>(W.dev_arena + p.comp_off[1]);
                        if (p.fmt == FMT_FP8_ROW) ++W.n_fp8_row; else ++W.n_other;
                        break;
                    case FMT_BF16:
                        dm.bf16 = reinterpret_cast<const __nv_bfloat16*>(W.dev_arena + p.comp_off[0]);
                        ++W.n_bf16;
                        break;
                    case FMT_FP32:
                        dm.fp32 = reinterpret_cast<const float*>(W.dev_arena + p.comp_off[0]);
                        ++W.n_other;
                        break;
                    default:
                        throw std::runtime_error("g6embed: unsupported matrix format for " + p.name);
                }
            }
            mat_map[p.name] = dm;
        } else {
            if (g6l_embed_read_component(h, p.item_idx, 0, stage, p.comp_bytes[0], err, sizeof(err)) != 0) {
                throw std::runtime_error(std::string("g6l_embed_read_component(") + p.name + "): " + err);
            }
            EMB_CK(cudaMemcpyAsync(W.dev_arena + p.comp_off[0], stage, p.comp_bytes[0], cudaMemcpyHostToDevice, st));
            EMB_CK(cudaStreamSynchronize(st));
            raw_map[p.name] = W.dev_arena + p.comp_off[0];
        }
    }

    auto req_mat = [&](const std::string& k) -> DevMatrix {
        auto it = mat_map.find(k);
        if (it == mat_map.end()) throw std::runtime_error("g6embed: missing matrix " + k);
        return it->second;
    };
    auto opt_mat = [&](const std::string& k) -> DevMatrix {
        auto it = mat_map.find(k);
        return it == mat_map.end() ? DevMatrix{} : it->second;
    };
    auto req_bf16 = [&](const std::string& k) -> const __nv_bfloat16* {
        auto it = raw_map.find(k);
        if (it != raw_map.end()) return reinterpret_cast<const __nv_bfloat16*>(it->second);
        auto mit = mat_map.find(k);
        if (mit != mat_map.end() && mit->second.fmt == FMT_BF16) return mit->second.bf16;
        throw std::runtime_error("g6embed: missing BF16 tensor " + k);
    };

    const bool is_q3 = (W.cfg.profile == EmbedProfile::Qwen3Vl);
    const std::string text_pfx = is_q3 ? "model.language_model.layers." : "language_model.layers.";
    W.embed_tokens = req_mat(is_q3 ? "model.language_model.embed_tokens.weight" : "language_model.embed_tokens.weight");
    W.final_norm = req_bf16(is_q3 ? "model.language_model.norm.weight" : "language_model.norm.weight");
    if (!is_q3) {
        W.eg2_projection = opt_mat("language_model.embedding_projection.weight");
    }

    W.text_layers.resize(size_t(W.cfg.num_text_layers));
    for (int l = 0; l < W.cfg.num_text_layers; ++l) {
        const std::string p = text_pfx + std::to_string(l) + ".";
        TextLayerW& L = W.text_layers[size_t(l)];
        L.input_ln = req_bf16(p + "input_layernorm.weight");
        L.post_attn_ln = req_bf16(p + "post_attention_layernorm.weight");
        L.q_norm = req_bf16(p + "self_attn.q_norm.weight");
        L.k_norm = req_bf16(p + "self_attn.k_norm.weight");
        L.q_proj = req_mat(p + "self_attn.q_proj.weight");
        L.k_proj = req_mat(p + "self_attn.k_proj.weight");
        L.v_proj = req_mat(p + "self_attn.v_proj.weight");
        L.o_proj = req_mat(p + "self_attn.o_proj.weight");
        L.gate_proj = req_mat(p + "mlp.gate_proj.weight");
        L.up_proj = req_mat(p + "mlp.up_proj.weight");
        L.down_proj = req_mat(p + "mlp.down_proj.weight");
    }

    if (is_q3 && W.cfg.has_vision) {
        W.patch_proj_w = req_bf16("model.visual.patch_embed.proj.weight");
        W.patch_proj_b = req_bf16("model.visual.patch_embed.proj.bias");
        W.pos_embed = req_bf16("model.visual.pos_embed.weight");
        W.vision_blocks.resize(size_t(W.cfg.vit_layers));
        for (int l = 0; l < W.cfg.vit_layers; ++l) {
            const std::string p = "model.visual.blocks." + std::to_string(l) + ".";
            VisionBlockW& B = W.vision_blocks[size_t(l)];
            B.norm1_w = req_bf16(p + "norm1.weight");
            B.norm1_b = req_bf16(p + "norm1.bias");
            B.norm2_w = req_bf16(p + "norm2.weight");
            B.norm2_b = req_bf16(p + "norm2.bias");
            B.qkv_w = req_mat(p + "attn.qkv.weight");
            B.qkv_b = req_bf16(p + "attn.qkv.bias");
            B.proj_w = req_mat(p + "attn.proj.weight");
            B.proj_b = req_bf16(p + "attn.proj.bias");
            B.fc1_w = req_mat(p + "mlp.linear_fc1.weight");
            B.fc1_b = req_bf16(p + "mlp.linear_fc1.bias");
            B.fc2_w = req_mat(p + "mlp.linear_fc2.weight");
            B.fc2_b = req_bf16(p + "mlp.linear_fc2.bias");
        }
        W.main_merger.use_postshuffle_norm = false;
        W.main_merger.norm_w = req_bf16("model.visual.merger.norm.weight");
        W.main_merger.norm_b = req_bf16("model.visual.merger.norm.bias");
        W.main_merger.fc1_w = req_mat("model.visual.merger.linear_fc1.weight");
        W.main_merger.fc1_b = req_bf16("model.visual.merger.linear_fc1.bias");
        W.main_merger.fc2_w = req_mat("model.visual.merger.linear_fc2.weight");
        W.main_merger.fc2_b = req_bf16("model.visual.merger.linear_fc2.bias");

        W.deepstack_mergers.resize(W.cfg.deepstack_indexes.size());
        for (size_t k = 0; k < W.cfg.deepstack_indexes.size(); ++k) {
            const std::string p = "model.visual.deepstack_merger_list." + std::to_string(k) + ".";
            MergerW& M = W.deepstack_mergers[k];
            M.use_postshuffle_norm = true;
            M.norm_w = req_bf16(p + "norm.weight");
            M.norm_b = req_bf16(p + "norm.bias");
            M.fc1_w = req_mat(p + "linear_fc1.weight");
            M.fc1_b = req_bf16(p + "linear_fc1.bias");
            M.fc2_w = req_mat(p + "linear_fc2.weight");
            M.fc2_b = req_bf16(p + "linear_fc2.bias");
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "g6embed: loaded %s (%s, %d text L, %d vis L, %zu ds) in %.2f s: %.1f MiB VRAM "
                 "[trellis=%d int4=%d fp8_row=%d bf16=%d other=%d, mode=%s, verify=%d]\n",
                 W.cfg.model_name.c_str(),
                 W.cfg.profile == EmbedProfile::Qwen3Vl ? "qwen3_vl" : "embedding_gemma2",
                 W.cfg.num_text_layers, W.cfg.vit_layers, W.cfg.deepstack_indexes.size(), secs,
                 double(W.arena_bytes) / (1024.0 * 1024.0),
                 W.n_trellis, W.n_int4, W.n_fp8_row, W.n_bf16, W.n_other,
                 mode == WeightMode::Bf16 ? "bf16" : "native",
                 verify_checksums ? 1 : 0);
    return W;
}

} // namespace g6emb
