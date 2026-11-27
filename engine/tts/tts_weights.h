// Drama 3 (fish_omni) weight structs and 64K-page-safe pinned-staging GPU loader for g6tts.
//
// Memory modes:
//   - WeightMode::Ref (default): dense/tower/vocab/shared in FP16 (~5 GB); store-0 routed experts kept
//     in block-128 FP8 E4M3 + F32 scales on GPU (~32.2 GB, ~37 GB total!) and dequantized on the fly
//     for only the active routed experts per step. Store-1 (trellis) layers decode to FP16 on load.
//   - WeightMode::RefFp16: all matrices including every routed expert pre-decoded to FP16 on load (~72 GB).
//   - WeightMode::Native: placeholder for Wave 3.2 (currently routes through Ref path).
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
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tts_config.h"
#include "../loader-rs/g6load.h"

namespace g6tts {

inline void tts_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("g6tts CUDA error: ") + what + ": " + cudaGetErrorString(e));
    }
}
#define TTS_CK(x) ::g6tts::tts_ck((x), #x)

inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

// Host BF16 -> FP32 exact conversion (all 1D norms, conv1d, A_log, dt_bias live in FP32 on GPU).
inline float host_bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct BackboneLayerW {
    bool is_full_attn = false;
    bool experts_fp8 = false; // true: routed experts stored as FP8_BLOCK on device (~32 GB total)

    const float* input_ln = nullptr;     // [H]
    const float* post_attn_ln = nullptr; // [H]

    // Full attention (l % 4 == 3)
    const __half* q_proj = nullptr; // [HEADS * 2 * HEAD_DIM, H] = [8192, 2048]
    const __half* k_proj = nullptr; // [KV_HEADS * HEAD_DIM, H] = [512, 2048]
    const __half* v_proj = nullptr; // [KV_HEADS * HEAD_DIM, H] = [512, 2048]
    const __half* o_proj = nullptr; // [H, HEADS * HEAD_DIM] = [2048, 4096]
    const float* q_norm = nullptr;  // [HEAD_DIM] = [256]
    const float* k_norm = nullptr;  // [HEAD_DIM] = [256]

    // Gated DeltaNet (l % 4 != 3)
    const __half* in_proj_qkv = nullptr; // [8192, 2048]
    const __half* in_proj_z = nullptr;   // [4096, 2048]
    const __half* in_proj_a = nullptr;   // [32, 2048]
    const __half* in_proj_b = nullptr;   // [32, 2048]
    const __half* out_proj = nullptr;    // [2048, 4096]
    const float* conv1d = nullptr;       // [8192, 4]
    const float* A_log = nullptr;        // [32]
    const float* dt_bias = nullptr;      // [32]
    const float* gdn_norm = nullptr;     // [128] (plain weight, NOT 1+w)

    // MoE router & shared expert (all FP16)
    const __half* router_gate = nullptr;        // [NEXP, H] = [256, 2048]
    const __half* shared_gate_proj = nullptr;   // [SHARED_I, H] = [512, 2048]
    const __half* shared_up_proj = nullptr;     // [SHARED_I, H] = [512, 2048]
    const __half* shared_down_proj = nullptr;   // [H, SHARED_I] = [2048, 512]
    const __half* shared_expert_gate = nullptr; // [1, H] = [1, 2048]

    // Routed experts: either FP8_BLOCK (`experts_fp8 == true`) OR pre-decoded FP16 (`experts_fp8 == false`)
    // FP8_BLOCK contiguous across experts [NEXP, out, in] and scales [NEXP, out/128, in/128]
    const uint8_t* exp_gate_fp8 = nullptr;
    const float* exp_gate_sc = nullptr;
    const uint8_t* exp_up_fp8 = nullptr;
    const float* exp_up_sc = nullptr;
    const uint8_t* exp_down_fp8 = nullptr;
    const float* exp_down_sc = nullptr;

    // FP16 contiguous across experts [NEXP, out, in]
    const __half* exp_gate_f16 = nullptr;
    const __half* exp_up_f16 = nullptr;
    const __half* exp_down_f16 = nullptr;
};

struct TowerLayerW {
    bool is_shared = false;
    const __half* embedding = nullptr; // [4096, 2048] (semantic_embedding or codebook_embedding)
    const float* enorm = nullptr;      // [2048]
    const float* hnorm = nullptr;      // [2048]
    const __half* eh_proj = nullptr;   // [2048, 4096]

    const float* attn_norm = nullptr; // [2048]
    const __half* wqkv = nullptr;     // [9216, 2048] = [Q 4096 | Gate 4096 | K 512 | V 512]
    const __half* wo = nullptr;       // [2048, 4096]
    const float* q_norm = nullptr;    // [256]
    const float* k_norm = nullptr;    // [256]

    const float* ffn_norm = nullptr; // [2048]
    const __half* w1 = nullptr;      // [6144, 2048]
    const __half* w3 = nullptr;      // [6144, 2048]
    const __half* w2 = nullptr;      // [2048, 6144] (computed with FP32 output!)

    const float* final_ln = nullptr; // [2048]
};

struct TtsWeights {
    TtsConfig cfg;
    WeightMode mode = WeightMode::Ref;

    uint8_t* dense_arena = nullptr;
    size_t dense_bytes = 0;
    uint8_t* expert_arena = nullptr;
    size_t expert_bytes = 0;

    // Global backbone + tower weights
    const __half* embed_tokens = nullptr;        // [VOCAB_SIZE, H] FP16
    const float* final_norm = nullptr;           // [H] FP32
    const __half* lm_head_cand = nullptr;        // [MAIN_CANDIDATES (4097), H] FP16 (sorted {EOS} U [SEM0..SEM1])
    const int32_t* main_cand_ids_dev = nullptr;  // [MAIN_CANDIDATES (4097)] INT32 on GPU
    const __half* codebook_embeddings = nullptr; // [NCB * 4096, H] = [40960, 2048] FP16

    std::vector<BackboneLayerW> layers;      // [40]
    std::vector<TowerLayerW> shared_layers;  // [4]
    std::vector<TowerLayerW> res_layers;     // [9]
    std::vector<const __half*> heads;        // [9], each [4096, 2048] FP16

    int n_fp8_expert_layers = 0;
    int n_f16_expert_layers = 0;

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
        embed_tokens = o.embed_tokens;
        final_norm = o.final_norm;
        lm_head_cand = o.lm_head_cand;
        main_cand_ids_dev = o.main_cand_ids_dev;
        codebook_embeddings = o.codebook_embeddings;
        layers = std::move(o.layers);
        shared_layers = std::move(o.shared_layers);
        res_layers = std::move(o.res_layers);
        heads = std::move(o.heads);
        n_fp8_expert_layers = o.n_fp8_expert_layers;
        n_f16_expert_layers = o.n_f16_expert_layers;
        return *this;
    }
    ~TtsWeights() { release(); }

    void release() {
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

// Loads a Drama 3 exlr8 release into GPU arenas via `g6l_tts_*` and pinned host staging (zero mmap).
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

    // 1. Plan dense arena: all non-routed-expert items (matrices -> FP16; raw vectors -> FP32; lm_head -> sliced [4097, H]).
    struct DensePlan {
        size_t item_idx = 0;
        std::string name;
        bool is_matrix = false;
        bool is_lm_head = false;
        std::string raw_dtype;
        size_t raw_bytes = 0;
        size_t out = 0, inp = 0;
        size_t dev_off = 0;
        size_t dev_bytes = 0;
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
        if (is_routed_expert_name(name)) continue; // handled in expert arena

        DensePlan p;
        p.item_idx = i;
        p.name = name;
        p.is_matrix = it.is_matrix != 0;
        p.is_lm_head = (name == "lm_head.weight");
        if (p.is_matrix) {
            p.out = static_cast<size_t>(it.shape[0]);
            p.inp = static_cast<size_t>(it.shape[1]);
            const size_t full_f16 = p.out * p.inp * 2;
            p.dev_off = dense_total;
            p.dev_bytes = p.is_lm_head ? (size_t(n_cand) * p.inp * 2) : full_f16;
            dense_total += align256(p.dev_bytes);
            max_stage = std::max(max_stage, full_f16);
        } else {
            p.raw_dtype = it.dtype ? it.dtype : "BF16";
            p.raw_bytes = static_cast<size_t>(it.comps[0].nbytes);
            size_t elems = 1;
            for (uint32_t d = 0; d < it.ndim; ++d) elems *= static_cast<size_t>(it.shape[d]);
            p.dev_off = dense_total;
            p.dev_bytes = elems * sizeof(float); // store all 1D/3D vectors as FP32 on GPU
            dense_total += align256(p.dev_bytes);
            max_stage = std::max(max_stage, std::max(p.raw_bytes, p.dev_bytes));
        }
        dplans.push_back(std::move(p));
    }

    W.dense_bytes = std::max<size_t>(256, dense_total);
    TTS_CK(cudaMalloc(&W.dense_arena, W.dense_bytes));

    // 2. Plan expert arena (one contiguous arena for all 40 layers' routed experts).
    const int L = W.cfg.num_layers;
    const int E = W.cfg.num_experts;
    const int H = W.cfg.hidden_size;
    const int I = W.cfg.moe_intermediate;
    const int bo_gu = (I + 127) / 128;
    const int bi_gu = (H + 127) / 128;
    const int bo_dn = (H + 127) / 128;
    const int bi_dn = (I + 127) / 128;

    struct LayerExpPlan {
        int store = -1;
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
        const int st = g6l_tts_expert_store(h, uint32_t(l));
        if (st < 0) {
            throw std::runtime_error("g6tts: layer " + std::to_string(l) + " has no expert store");
        }
        eplans[size_t(l)].store = st;
        // Use FP8-on-device when mode != RefFp16 and store == 0 (~32.2 GB instead of ~64.4 GB)
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
            // Reserve space in expert arena for trellis-encoded shared expert if not in dense files
            eplans[size_t(l)].shared_from_store = true;
            eplans[size_t(l)].off_sg = exp_total; exp_total += align256(sh_f16);
            eplans[size_t(l)].off_su = exp_total; exp_total += align256(sh_f16);
            eplans[size_t(l)].off_sd = exp_total; exp_total += align256(sh_f16);
        }
    }

    W.expert_bytes = std::max<size_t>(256, exp_total);
    TTS_CK(cudaMalloc(&W.expert_arena, W.expert_bytes));

    // Pinned staging buffer for dense + per-layer expert chunks
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
    std::unordered_map<std::string, const __half*> mat_map;
    std::unordered_map<std::string, const float*> vec_map;
    std::vector<uint8_t> raw_tmp;

    for (const auto& p : dplans) {
        if (p.is_matrix) {
            const size_t full_f16 = p.out * p.inp * 2;
            if (g6l_tts_dense_decode_f16(h, p.item_idx, stage, full_f16, decode_threads, err, sizeof(err)) != 0) {
                throw std::runtime_error(std::string("g6l_tts_dense_decode_f16(") + p.name + "): " + err);
            }
            if (p.is_lm_head) {
                // Gather only the 4097 legal main-token rows into a compact FP16 matrix
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
                TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, full_f16, cudaMemcpyHostToDevice, st));
            }
            TTS_CK(cudaStreamSynchronize(st));
            mat_map[p.name] = reinterpret_cast<const __half*>(W.dense_arena + p.dev_off);
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
                const __half* src = reinterpret_cast<const __half*>(raw_tmp.data());
                for (size_t k = 0; k < elems; ++k) dst[k] = __half2float(src[k]);
            } else {
                throw std::runtime_error("g6tts: unsupported raw vector dtype " + p.raw_dtype + " for " + p.name);
            }
            TTS_CK(cudaMemcpyAsync(W.dense_arena + p.dev_off, stage, p.dev_bytes, cudaMemcpyHostToDevice, st));
            TTS_CK(cudaStreamSynchronize(st));
            vec_map[p.name] = reinterpret_cast<const float*>(W.dense_arena + p.dev_off);
        }
    }

    auto req_mat = [&](const std::string& k) -> const __half* {
        auto it = mat_map.find(k);
        if (it == mat_map.end()) throw std::runtime_error("g6tts: missing matrix " + k);
        return it->second;
    };
    auto req_vec = [&](const std::string& k) -> const float* {
        auto it = vec_map.find(k);
        if (it == vec_map.end()) throw std::runtime_error("g6tts: missing vector " + k);
        return it->second;
    };

    W.embed_tokens = req_mat("model.language_model.embed_tokens.weight");
    W.final_norm = req_vec("model.language_model.norm.weight");
    W.lm_head_cand = req_mat("lm_head.weight");
    W.codebook_embeddings = req_mat("audio_decoder.codebook_embeddings.weight");

    // 4. Bind backbone layers + upload routed experts (in parallel across CPU workers per chunk)
    W.layers.resize(size_t(L));
    const unsigned n_workers = std::max(1u, std::min<unsigned>(decode_threads, 16u));

    for (int l = 0; l < L; ++l) {
        const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
        BackboneLayerW& LW = W.layers[size_t(l)];
        LW.is_full_attn = W.cfg.is_full_attn[size_t(l)];
        LW.input_ln = req_vec(p + "input_layernorm.weight");
        LW.post_attn_ln = req_vec(p + "post_attention_layernorm.weight");

        if (LW.is_full_attn) {
            LW.q_proj = req_mat(p + "self_attn.q_proj.weight");
            LW.k_proj = req_mat(p + "self_attn.k_proj.weight");
            LW.v_proj = req_mat(p + "self_attn.v_proj.weight");
            LW.o_proj = req_mat(p + "self_attn.o_proj.weight");
            LW.q_norm = req_vec(p + "self_attn.q_norm.weight");
            LW.k_norm = req_vec(p + "self_attn.k_norm.weight");
        } else {
            LW.in_proj_qkv = req_mat(p + "linear_attn.in_proj_qkv.weight");
            LW.in_proj_z = req_mat(p + "linear_attn.in_proj_z.weight");
            LW.in_proj_a = req_mat(p + "linear_attn.in_proj_a.weight");
            LW.in_proj_b = req_mat(p + "linear_attn.in_proj_b.weight");
            LW.out_proj = req_mat(p + "linear_attn.out_proj.weight");
            LW.conv1d = req_vec(p + "linear_attn.conv1d.weight");
            LW.A_log = req_vec(p + "linear_attn.A_log");
            LW.dt_bias = req_vec(p + "linear_attn.dt_bias");
            LW.gdn_norm = req_vec(p + "linear_attn.norm.weight");
        }

        LW.router_gate = req_mat(p + "mlp.gate.weight");
        LW.shared_expert_gate = req_mat(p + "mlp.shared_expert_gate.weight");

        // Shared expert: either already in dense arena (dev-ref) or decoded from trellis expert store
        const LayerExpPlan& ep = eplans[size_t(l)];
        auto sh_g = mat_map.find(p + "mlp.shared_expert.gate_proj.weight");
        if (sh_g != mat_map.end()) {
            LW.shared_gate_proj = sh_g->second;
            LW.shared_up_proj = req_mat(p + "mlp.shared_expert.up_proj.weight");
            LW.shared_down_proj = req_mat(p + "mlp.shared_expert.down_proj.weight");
        } else if (ep.shared_from_store) {
            const size_t sh_offs[3] = {ep.off_sg, ep.off_su, ep.off_sd};
            for (int proj = 0; proj < 3; ++proj) {
                if (g6l_tts_expert_decode_f16(h, uint32_t(l), uint32_t(E), uint32_t(proj),
                                              stage, sh_f16, err, sizeof(err)) != 0) {
                    throw std::runtime_error("g6tts: shared expert decode failed at layer " +
                                             std::to_string(l) + ": " + err);
                }
                TTS_CK(cudaMemcpyAsync(W.expert_arena + sh_offs[proj], stage, sh_f16, cudaMemcpyHostToDevice, st));
                TTS_CK(cudaStreamSynchronize(st));
            }
            LW.shared_gate_proj = reinterpret_cast<const __half*>(W.expert_arena + ep.off_sg);
            LW.shared_up_proj = reinterpret_cast<const __half*>(W.expert_arena + ep.off_su);
            LW.shared_down_proj = reinterpret_cast<const __half*>(W.expert_arena + ep.off_sd);
        } else {
            throw std::runtime_error("g6tts: shared expert missing from dense arena at layer " + std::to_string(l));
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
            LW.exp_gate_f16 = reinterpret_cast<const __half*>(W.expert_arena + ep.off_g);
            LW.exp_up_f16 = reinterpret_cast<const __half*>(W.expert_arena + ep.off_u);
            LW.exp_down_f16 = reinterpret_cast<const __half*>(W.expert_arena + ep.off_d);
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
        T.embedding = req_mat(p + (is_shared ? "semantic_embedding.weight" : "codebook_embedding.weight"));
        T.enorm = req_vec(p + "enorm.weight");
        T.hnorm = req_vec(p + "hnorm.weight");
        T.eh_proj = req_mat(p + "eh_proj.weight");
        T.attn_norm = req_vec(p + "transformer_layer.attention_norm.weight");
        T.wqkv = req_mat(p + "transformer_layer.attention.wqkv.weight");
        T.wo = req_mat(p + "transformer_layer.attention.wo.weight");
        T.q_norm = req_vec(p + "transformer_layer.attention.q_norm.weight");
        T.k_norm = req_vec(p + "transformer_layer.attention.k_norm.weight");
        T.ffn_norm = req_vec(p + "transformer_layer.ffn_norm.weight");
        T.w1 = req_mat(p + "transformer_layer.feed_forward.w1.weight");
        T.w3 = req_mat(p + "transformer_layer.feed_forward.w3.weight");
        T.w2 = req_mat(p + "transformer_layer.feed_forward.w2.weight");
        T.final_ln = req_vec(p + "final_layernorm.weight");
        return T;
    };

    W.shared_layers.resize(size_t(W.cfg.tower_shared));
    for (int l = 0; l < W.cfg.tower_shared; ++l) {
        W.shared_layers[size_t(l)] = bind_tower("audio_decoder_mtp_shared.layers." + std::to_string(l) + ".", true);
    }
    W.res_layers.resize(size_t(W.cfg.tower_res));
    W.heads.resize(size_t(W.cfg.tower_res));
    for (int l = 0; l < W.cfg.tower_res; ++l) {
        W.res_layers[size_t(l)] = bind_tower("audio_decoder_mtp.layers." + std::to_string(l) + ".", false);
        W.heads[size_t(l)] = req_mat("audio_decoder_mtp.heads." + std::to_string(l) + ".weight");
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "g6tts: loaded %s (%dL, %d+%d tower) in %.2f s: %.2f GiB VRAM "
                 "(dense=%.2f GiB, experts=%.2f GiB [fp8=%dL f16=%dL], mode=%s, verify=%d)\n",
                 W.cfg.model_name.c_str(), W.cfg.num_layers, W.cfg.tower_shared, W.cfg.tower_res, secs,
                 double(W.dense_bytes + W.expert_bytes) / double(1ull << 30),
                 double(W.dense_bytes) / double(1ull << 30),
                 double(W.expert_bytes) / double(1ull << 30),
                 W.n_fp8_expert_layers, W.n_f16_expert_layers,
                 weight_mode_name(W.mode), verify_checksums ? 1 : 0);
    return W;
}

} // namespace g6tts
