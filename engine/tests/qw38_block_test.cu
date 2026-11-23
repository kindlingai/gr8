// Per-block GPU exactness verification for Qwen3.8-Flash-Next CUDA kernels against
// HuggingFace transformers (modeling_qwen4_exp.py) BF16 reference fixtures dumped by
// tools/convert/qw38_ref_dump.py.
//
// Verifies:
//   1. Hyper-Connections (qw38_hc.cuh): layer-0 attn_hc mixed + inject weights & final mixer
//   2. Gated DeltaNet (qw38_gdn.cuh): layer-0 16K/48V conv1d+SiLU + FP32 delta rule + sigmoid RMSNormGated
//   3. Layer-1 PLE (qw38_ple.h & qw38_ple.cuh): exact n-gram row IDs (_shift_right_ignore_eos + signed mod)
//      + GPU key/value/signed-sqrt gate + dilated-3 causal conv1d (k=4, d=3, hist=9)
//   4. Full Attention + QSA (qw38_attn.cuh): layer-3 dense (seq=16) AND sparse QSA (seq=2200 > 2048 budget!)
//   5. MoE + Weight Paging (qw38_moe.cuh + ptab.cuh): layer-0 exlr8-e4m3 pmcgx routed experts + BF16 gated shared
//      expert vs BF16 PyTorch MoE output AND byte-identical parity across GR8_PTAB=0, 1, 2 (+ ptab::set!)
//
// Build on GB10 (sm_121a):
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_QW38=1 -DG6_NR=1 -DG6_RS_LOADER \
//     -Iengine engine/tests/qw38_block_test.cu engine/loader-rs/target/release/libg6load.a \
//     engine/knobs-rs/target/release/libgr8knobs.a engine/target/release/libg6host.a \
//     -lcublas -lpthread -ldl -lm -o /tmp/qw38_block_test
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "model_geom.h"
#include "kernels/qw38_hc.cuh"
#include "kernels/qw38_gdn.cuh"
#include "attn/qw38_attn.cuh"
#include "engram/qw38_ple.h"
#include "engram/qw38_ple.cuh"
#include "kernels/qw38_moe.cuh"
#include "loader/g6pack.h"
#include "loader-rs/g6load.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct Bf16Cmp {
    double cos_sim = 0.0;
    float max_abs = 0.0f;
    double rel_l2 = 0.0;
};

static std::vector<uint8_t> read_file_bytes(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    const size_t n = size_t(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(n);
    if (n && !f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n))) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    return buf;
}

static Bf16Cmp cmp_bf16_dev(const __nv_bfloat16* d_got, const std::string& ref_path, size_t n) {
    auto rb = read_file_bytes(ref_path);
    if (rb.size() != n * 2) { fprintf(stderr, "%s: size %zu != %zu\n", ref_path.c_str(), rb.size(), n * 2); exit(1); }
    std::vector<uint16_t> got(n), ref(n);
    std::memcpy(ref.data(), rb.data(), n * 2);
    CK(cudaMemcpy(got.data(), d_got, n * 2, cudaMemcpyDeviceToHost));
    auto bf2f = [](uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; };
    double dot = 0, ng = 0, nr = 0, diff2 = 0;
    float ma = 0;
    for (size_t i = 0; i < n; ++i) {
        const float g = bf2f(got[i]), r = bf2f(ref[i]);
        dot += double(g) * double(r);
        ng += double(g) * double(g);
        nr += double(r) * double(r);
        const float d = fabsf(g - r);
        if (d > ma) ma = d;
        diff2 += double(d) * double(d);
    }
    return {dot / (sqrt(ng) * sqrt(nr) + 1e-30), ma, sqrt(diff2) / (sqrt(nr) + 1e-30)};
}

static Bf16Cmp cmp_f32_dev(const float* d_got, const std::string& ref_path, size_t n) {
    auto rb = read_file_bytes(ref_path);
    if (rb.size() != n * 4) { fprintf(stderr, "%s: size %zu != %zu\n", ref_path.c_str(), rb.size(), n * 4); exit(1); }
    std::vector<float> got(n), ref(n);
    std::memcpy(ref.data(), rb.data(), n * 4);
    CK(cudaMemcpy(got.data(), d_got, n * 4, cudaMemcpyDeviceToHost));
    double dot = 0, ng = 0, nr = 0, diff2 = 0;
    float ma = 0;
    for (size_t i = 0; i < n; ++i) {
        const float g = got[i], r = ref[i];
        dot += double(g) * double(r);
        ng += double(g) * double(g);
        nr += double(r) * double(r);
        const float d = fabsf(g - r);
        if (d > ma) ma = d;
        diff2 += double(d) * double(d);
    }
    return {dot / (sqrt(ng) * sqrt(nr) + 1e-30), ma, sqrt(diff2) / (sqrt(nr) + 1e-30)};
}

// Simple minimal safetensors reader for BF16/I64 checkpoint tensors (used when testing before full release exists)
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t start = 0, end = 0;
    std::string file;
};

static std::map<std::string, StTensor> index_bf16_dir(const std::string& dir) {
    // Parse model.safetensors.index.json weight_map, then lazily read headers per file
    std::string idx_txt;
    {
        auto b = read_file_bytes(dir + "/model.safetensors.index.json");
        idx_txt.assign(b.begin(), b.end());
    }
    // Find unique .safetensors filenames referenced in weight_map (we only need layers 0,1,3 + mixer + embed)
    std::map<std::string, StTensor> out;
    auto want_key = [](const std::string& k) {
        if (k.find(".shard_") != std::string::npos) return false;
        return k.rfind("model.language_model.layers.0.", 0) == 0 ||
               k.rfind("model.language_model.layers.1.", 0) == 0 ||
               k.rfind("model.language_model.layers.3.", 0) == 0 ||
               k.rfind("model.language_model.hyper_connection_mixer.", 0) == 0 ||
               k == "model.language_model.embed_tokens.weight";
    };
    size_t wm = idx_txt.find("\"weight_map\"");
    size_t pos = idx_txt.find('{', wm);
    std::map<std::string, std::vector<std::string>> by_file;
    while (pos < idx_txt.size()) {
        size_t q1 = idx_txt.find('"', pos + 1);
        if (q1 == std::string::npos) break;
        size_t q2 = idx_txt.find('"', q1 + 1);
        std::string k = idx_txt.substr(q1 + 1, q2 - q1 - 1);
        size_t q3 = idx_txt.find('"', q2 + 1);
        size_t q4 = idx_txt.find('"', q3 + 1);
        std::string fn = idx_txt.substr(q3 + 1, q4 - q3 - 1);
        if (want_key(k)) by_file[fn].push_back(k);
        size_t next = idx_txt.find_first_of(",}", q4 + 1);
        if (next == std::string::npos || idx_txt[next] == '}') break;
        pos = next;
    }
    for (const auto& [fn, ks] : by_file) {
        const std::string full = dir + "/" + fn;
        std::ifstream f(full, std::ios::binary);
        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        std::string hdr(hlen, '\0');
        f.read(hdr.data(), std::streamsize(hlen));
        for (const std::string& k : ks) {
            const std::string needle = "\"" + k + "\":";
            size_t p = hdr.find(needle);
            if (p == std::string::npos) continue;
            size_t o = hdr.find("\"data_offsets\"", p);
            size_t b1 = hdr.find('[', o), b2 = hdr.find(',', b1), b3 = hdr.find(']', b2);
            StTensor t;
            t.file = full;
            t.start = 8 + hlen + std::stoull(hdr.substr(b1 + 1, b2 - b1 - 1));
            t.end = 8 + hlen + std::stoull(hdr.substr(b2 + 1, b3 - b2 - 1));
            out[k] = t;
        }
    }
    return out;
}

template <class T>
static T* load_dev(const std::map<std::string, StTensor>& idx, const std::string& k) {
    auto it = idx.find(k);
    if (it == idx.end()) { fprintf(stderr, "missing tensor %s\n", k.c_str()); exit(1); }
    const size_t n = size_t(it->second.end - it->second.start);
    std::vector<uint8_t> h(n);
    std::ifstream f(it->second.file, std::ios::binary);
    f.seekg(std::streamoff(it->second.start));
    f.read(reinterpret_cast<char*>(h.data()), std::streamsize(n));
    T* d = nullptr;
    CK(cudaMalloc(&d, n));
    CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
    return d;
}

static std::vector<int64_t> load_i64_host(const std::map<std::string, StTensor>& idx, const std::string& k) {
    auto it = idx.find(k);
    if (it == idx.end()) { fprintf(stderr, "missing tensor %s\n", k.c_str()); exit(1); }
    const size_t n = size_t(it->second.end - it->second.start);
    std::vector<int64_t> h(n / 8);
    std::ifstream f(it->second.file, std::ios::binary);
    f.seekg(std::streamoff(it->second.start));
    f.read(reinterpret_cast<char*>(h.data()), std::streamsize(n));
    return h;
}

template <class T>
static T* upload_bin(const std::string& path) {
    auto b = read_file_bytes(path);
    T* d = nullptr;
    CK(cudaMalloc(&d, b.size()));
    CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bf16_dir> <ref_dir> [release_dir]\n", argv[0]);
        return 2;
    }
    const std::string bf16_dir = argv[1];
    const std::string ref_dir = argv[2];
    const std::string rel_dir = argc >= 4 ? argv[3] : "";

    cublasHandle_t blas;
    cublasCreate(&blas);
    cudaStream_t st = 0;

    auto idx = index_bf16_dir(bf16_dir);
    printf("Indexed %zu BF16 tensors from %s\n", idx.size(), bf16_dir.c_str());

    const int S = 16;
    auto toks_bytes = read_file_bytes(ref_dir + "/toks.i32");
    std::vector<int> toks(S);
    std::memcpy(toks.data(), toks_bytes.data(), S * 4);

    __nv_bfloat16* d_hyper = upload_bin<__nv_bfloat16>(ref_dir + "/hyper_in.bf16");
    __nv_bfloat16* d_mixed = nullptr;
    CK(cudaMalloc(&d_mixed, size_t(S) * geom::HID * 2));

    // --- 1. Hyper-Connections (Layer 0 attn_hc + final mixer) ---
    {
        qw38::HcWeights hc{};
        hc.norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight");
        hc.down = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight");
        hc.up = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight");
        hc.inject = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight");
        qw38::HcScratch hsc{};
        CK(cudaMalloc(&hsc.hn, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&hsc.down, size_t(S) * geom::HC_RANK * 2));
        CK(cudaMalloc(&hsc.up, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&hsc.inj, size_t(S) * geom::HC * 4));

        qw38::hc_prepare(blas, hc, d_hyper, d_mixed, hsc, S, st);
        CK(cudaStreamSynchronize(st));
        auto c_mix = cmp_bf16_dev(d_mixed, ref_dir + "/l0_hc_mixed.bf16", size_t(S) * geom::HID);
        auto c_inj = cmp_f32_dev(hsc.inj, ref_dir + "/l0_hc_inj.f32", size_t(S) * geom::HC);
        printf("[1] L0 HC mixed:  cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_mix.cos_sim, c_mix.rel_l2, c_mix.max_abs);
        printf("[1] L0 HC inject: cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_inj.cos_sim, c_inj.rel_l2, c_inj.max_abs);
        if (c_mix.cos_sim < 0.9999 || c_inj.cos_sim < 0.9999) { fprintf(stderr, "FAIL L0 HC\n"); return 1; }

        // --- 2. Gated DeltaNet (Layer 0, TP=1) ---
        qw38::GdnWeights gw{};
        gw.in_proj_qkv = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
        gw.conv1d = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.conv1d.weight");
        gw.in_proj_z = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.in_proj_z.weight");
        gw.in_proj_a = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.in_proj_a.weight");
        gw.in_proj_b = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.in_proj_b.weight");
        gw.A_log = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.A_log");
        gw.dt_bias = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.dt_bias");
        gw.norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.norm.weight");
        gw.out_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.0.linear_attn.out_proj.weight");

        qw38::GdnLayerState gst{};
        CK(cudaMalloc(&gst.S, size_t(geom::GDN_VH) * 128 * 128 * 4));
        CK(cudaMalloc(&gst.conv, size_t(3) * geom::GDN_QKV_DIM * 2));
        qw38::GdnScratch gsc{};
        CK(cudaMalloc(&gsc.qkv_raw, size_t(S) * geom::GDN_QKV_DIM * 2));
        CK(cudaMalloc(&gsc.qkv_conv, size_t(S) * geom::GDN_QKV_DIM * 2));
        CK(cudaMalloc(&gsc.z, size_t(S) * geom::GDN_V_DIM * 2));
        CK(cudaMalloc(&gsc.a, size_t(S) * geom::GDN_VH * 2));
        CK(cudaMalloc(&gsc.b, size_t(S) * geom::GDN_VH * 2));
        CK(cudaMalloc(&gsc.core, size_t(S) * geom::GDN_V_DIM * 2));
        __nv_bfloat16* d_gdn_out = nullptr;
        CK(cudaMalloc(&d_gdn_out, size_t(S) * geom::HID * 2));

        // Feed exact ref l0_hc_mixed so we isolate GDN numerics cleanly!
        __nv_bfloat16* d_ref_m0 = upload_bin<__nv_bfloat16>(ref_dir + "/l0_hc_mixed.bf16");
        qw38::StepSeg seg{0, 0, 0, S};
        qw38::gdn_forward(blas, gw, gst, gsc, d_ref_m0, d_gdn_out, &seg, 1, S, 1, 0, true, st);
        CK(cudaStreamSynchronize(st));
        auto c_gdn = cmp_bf16_dev(d_gdn_out, ref_dir + "/l0_gdn_out.bf16", size_t(S) * geom::HID);
        printf("[2] L0 GDN (prefill 16): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_gdn.cos_sim, c_gdn.rel_l2, c_gdn.max_abs);
        if (c_gdn.cos_sim < 0.9995) { fprintf(stderr, "FAIL L0 GDN prefill\n"); return 1; }

        // Also verify token-by-token autoregressive GDN decode matches batched 16-token GDN!
        __nv_bfloat16* d_gdn_dec = nullptr;
        CK(cudaMalloc(&d_gdn_dec, size_t(S) * geom::HID * 2));
        for (int t = 0; t < S; ++t) {
            qw38::StepSeg dseg{0, 0, t, 1};
            qw38::gdn_forward(blas, gw, gst, gsc, d_ref_m0 + size_t(t) * geom::HID,
                              d_gdn_dec + size_t(t) * geom::HID, &dseg, 1, 1, 1, 0, true, st);
        }
        CK(cudaStreamSynchronize(st));
        auto c_gdn_dec = cmp_bf16_dev(d_gdn_dec, ref_dir + "/l0_gdn_out.bf16", size_t(S) * geom::HID);
        printf("[2] L0 GDN (decode 1x16): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_gdn_dec.cos_sim, c_gdn_dec.rel_l2, c_gdn_dec.max_abs);
        if (c_gdn_dec.cos_sim < 0.9995) { fprintf(stderr, "FAIL L0 GDN decode\n"); return 1; }
    }

    // --- 3. Layer-1 PLE: exact n-gram row IDs + GPU gating & dilated-3 conv1d ---
    {
        auto mult = load_i64_host(idx, "model.language_model.layers.1.ple.ple_embedding.layer_multipliers");
        auto offs = load_i64_host(idx, "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets");
        auto vs = load_i64_host(idx, "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes");
        qw38::PleStore store;
        store.set_hash_params(mult.data(), offs.data(), vs.data(), geom::TOK_EOS);
        int32_t ctx[2] = {geom::TOK_EOS, geom::TOK_EOS};
        std::vector<uint64_t> got_ids(size_t(S) * geom::PLE_HEADS);
        store.compute_ids(toks.data(), S, ctx, true, got_ids.data());

        auto ref_id_bytes = read_file_bytes(ref_dir + "/ple_ids.i64");
        std::vector<int64_t> ref_ids(size_t(S) * geom::PLE_HEADS);
        std::memcpy(ref_ids.data(), ref_id_bytes.data(), ref_id_bytes.size());
        int id_mismatches = 0;
        for (size_t i = 0; i < got_ids.size(); ++i) {
            if (int64_t(got_ids[i]) != ref_ids[i]) {
                if (id_mismatches < 5) fprintf(stderr, "PLE id[%zu]: got %llu want %lld\n", i, (unsigned long long)got_ids[i], (long long)ref_ids[i]);
                ++id_mismatches;
            }
        }
        printf("[3] L1 PLE n-gram IDs (with EOS reset): %d mismatches / %zu\n", id_mismatches, got_ids.size());
        if (id_mismatches != 0) return 1;

        // Also check incremental 1-by-1 decode produces the exact same IDs!
        int32_t ctx1[2] = {geom::TOK_EOS, geom::TOK_EOS};
        for (int t = 0; t < S; ++t) {
            uint64_t row16[16];
            store.compute_ids(&toks[t], 1, ctx1, true, row16);
            for (int h = 0; h < 16; ++h) {
                if (int64_t(row16[h]) != ref_ids[size_t(t) * 16 + h]) {
                    fprintf(stderr, "PLE 1-by-1 id mismatch at t=%d h=%d\n", t, h);
                    return 1;
                }
            }
        }
        printf("[3] L1 PLE n-gram IDs (1x16 decode): 0 mismatches\n");

        // GPU PLE forward on ref h0_out + ref ple_emb
        qw38::PleWeights pw{};
        pw.key_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.key_proj.weight");
        pw.value_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.value_proj.weight");
        pw.conv1d = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.conv1d.weight");
        pw.norm_query = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.norm_query.weight");
        pw.norm_key = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.norm_key.weight");
        pw.norm_conv = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.1.ple.norm_conv.weight");

        qw38::PleState pst{};
        CK(cudaMalloc(&pst.conv_hist, size_t(geom::PLE_NHIST) * geom::HC_D * 2));
        qw38::PleScratch psc{};
        psc.emb = upload_bin<__nv_bfloat16>(ref_dir + "/ple_emb.bf16");
        CK(cudaMalloc(&psc.key, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&psc.query, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&psc.val, size_t(S) * geom::HID * 2));
        CK(cudaMalloc(&psc.gv, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&psc.gv_norm, size_t(S) * geom::HC_D * 2));

        __nv_bfloat16* d_h0_out = upload_bin<__nv_bfloat16>(ref_dir + "/l0_out.bf16");
        qw38::StepSeg pseg{0, 0, 0, S};
        qw38::ple_forward(blas, pw, pst, psc, d_h0_out, &pseg, 1, S, true, st);
        CK(cudaStreamSynchronize(st));
        auto c_ple = cmp_bf16_dev(d_h0_out, ref_dir + "/l1_after_ple.bf16", size_t(S) * geom::HC_D);
        printf("[3] L1 PLE GPU (dilated-3 conv1d, prefill 16): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_ple.cos_sim, c_ple.rel_l2, c_ple.max_abs);
        if (c_ple.cos_sim < 0.9998) { fprintf(stderr, "FAIL L1 PLE GPU\n"); return 1; }

        // Verify 1-by-1 PLE decode matches prefill 16!
        __nv_bfloat16* d_h0_dec = upload_bin<__nv_bfloat16>(ref_dir + "/l0_out.bf16");
        __nv_bfloat16* emb_all = psc.emb;
        for (int t = 0; t < S; ++t) {
            psc.emb = emb_all + size_t(t) * geom::PLE_DIM;
            qw38::StepSeg dseg{0, 0, t, 1};
            qw38::ple_forward(blas, pw, pst, psc, d_h0_dec + size_t(t) * geom::HC_D, &dseg, 1, 1, true, st);
        }
        CK(cudaStreamSynchronize(st));
        auto c_ple_dec = cmp_bf16_dev(d_h0_dec, ref_dir + "/l1_after_ple.bf16", size_t(S) * geom::HC_D);
        printf("[3] L1 PLE GPU (decode 1x16): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_ple_dec.cos_sim, c_ple_dec.rel_l2, c_ple_dec.max_abs);
        if (c_ple_dec.cos_sim < 0.9998) { fprintf(stderr, "FAIL L1 PLE decode\n"); return 1; }
    }

    // --- 4. Layer-3 Full Attention + QSA (short dense seq=16 AND long sparse seq=2200 > 2048!) ---
    {
        qw38::FullAttnWeights aw{};
        aw.q_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.q_proj.weight");
        aw.k_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.k_proj.weight");
        aw.v_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.v_proj.weight");
        aw.o_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.o_proj.weight");
        aw.q_norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.q_norm.weight");
        aw.k_norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.k_norm.weight");
        aw.idx_qk_proj = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight");
        aw.idx_q_norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight");
        aw.idx_k_norm = load_dev<__nv_bfloat16>(idx, "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight");

        const int S_long = 2200;
        const int max_pages = (S_long + 63) / 64;
        const int max_blocks = max_pages * 16;
        const int nh_q = geom::HEADS, nh_kv = geom::KV_HEADS;

        qw38::FullLayerKv kv{};
        CK(cudaMalloc(&kv.kv_pages, size_t(max_pages) * 2 * 64 * nh_kv * 256 * 2));
        CK(cudaMalloc(&kv.idx_pages, size_t(max_pages) * 16 * 128 * 2));
        CK(cudaMalloc(&kv.idx_raw, size_t(4) * 128 * 2));

        qw38::FullAttnScratch asc{};
        CK(cudaMalloc(&asc.q_gate, size_t(S_long) * nh_q * 512 * 2));
        CK(cudaMalloc(&asc.q, size_t(S_long) * nh_q * 256 * 2));
        CK(cudaMalloc(&asc.gate, size_t(S_long) * nh_q * 256 * 2));
        CK(cudaMalloc(&asc.k, size_t(S_long) * nh_kv * 256 * 2));
        CK(cudaMalloc(&asc.v, size_t(S_long) * nh_kv * 256 * 2));
        CK(cudaMalloc(&asc.idx_qk, size_t(S_long) * geom::IDX_PROJ_OUT * 2));
        CK(cudaMalloc(&asc.idx_q, size_t(S_long) * geom::IDX_HEADS * 128 * 2));
        CK(cudaMalloc(&asc.attn_out, size_t(S_long) * nh_q * 256 * 2));
        CK(cudaMalloc(&asc.idx_scores, size_t(S_long) * max_blocks * 4));
        CK(cudaMalloc(&asc.sel_blocks, size_t(S_long) * geom::IDX_TOP_BLOCKS * 4));
        CK(cudaMalloc(&asc.sel_counts, size_t(S_long) * 4));

        std::vector<int> h_slot(S_long, 0), h_pos(S_long);
        for (int i = 0; i < S_long; ++i) h_pos[i] = i;
        int *d_slot = nullptr, *d_pos = nullptr;
        CK(cudaMalloc(&d_slot, S_long * 4));
        CK(cudaMalloc(&d_pos, S_long * 4));
        CK(cudaMemcpy(d_slot, h_slot.data(), S_long * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, h_pos.data(), S_long * 4, cudaMemcpyHostToDevice));

        // 4a: Short dense seq=16
        __nv_bfloat16* d_a3_in = upload_bin<__nv_bfloat16>(ref_dir + "/l3_attn_in.bf16");
        __nv_bfloat16* d_a3_out = nullptr;
        CK(cudaMalloc(&d_a3_out, size_t(S_long) * geom::HID * 2));
        qw38::StepSeg aseg{0, 0, 0, S};
        qw38::full_attn_forward(blas, aw, kv, asc, d_a3_in, d_a3_out, d_slot, d_pos, &aseg, 1, S, 1, 0, max_pages, max_blocks, st);
        CK(cudaStreamSynchronize(st));
        auto c_a3 = cmp_bf16_dev(d_a3_out, ref_dir + "/l3_attn_out.bf16", size_t(S) * geom::HID);
        printf("[4a] L3 Full Attn (dense seq=16): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_a3.cos_sim, c_a3.rel_l2, c_a3.max_abs);
        if (c_a3.cos_sim < 0.9998) { fprintf(stderr, "FAIL L3 dense attn\n"); return 1; }

        // 4b: Long sparse QSA seq=2200 (> 2048 budget -> exercises block-4 pooling, scoring, top-512 selection + tail!)
        __nv_bfloat16* d_long_in = upload_bin<__nv_bfloat16>(ref_dir + "/l3_long_in.bf16");
        qw38::StepSeg lseg{0, 0, 0, S_long};
        qw38::full_attn_forward(blas, aw, kv, asc, d_long_in, d_a3_out, d_slot, d_pos, &lseg, 1, S_long, 1, 0, max_pages, max_blocks, st);
        CK(cudaStreamSynchronize(st));
        auto c_qsa = cmp_bf16_dev(d_a3_out + size_t(S_long - 8) * geom::HID, ref_dir + "/l3_long_out_tail8.bf16", size_t(8) * geom::HID);
        printf("[4b] L3 Full Attn + QSA SPARSE (seq=2200, tail 8): cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_qsa.cos_sim, c_qsa.rel_l2, c_qsa.max_abs);
        if (c_qsa.cos_sim < 0.9990) { fprintf(stderr, "FAIL L3 QSA sparse attn\n"); return 1; }
    }

    // --- 5. Final HC mixer ---
    {
        qw38::HcWeights fhc{};
        fhc.norm = load_dev<__nv_bfloat16>(idx, "model.language_model.hyper_connection_mixer.hc_norm.weight");
        fhc.down = load_dev<__nv_bfloat16>(idx, "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight");
        fhc.up = load_dev<__nv_bfloat16>(idx, "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight");
        qw38::HcScratch hsc{};
        CK(cudaMalloc(&hsc.hn, size_t(S) * geom::HC_D * 2));
        CK(cudaMalloc(&hsc.down, size_t(S) * geom::HC_RANK * 2));
        CK(cudaMalloc(&hsc.up, size_t(S) * geom::HC_D * 2));
        __nv_bfloat16* d_in = upload_bin<__nv_bfloat16>(ref_dir + "/l1_after_ple.bf16");
        qw38::hc_prepare(blas, fhc, d_in, d_mixed, hsc, S, st);
        CK(cudaStreamSynchronize(st));
        auto c_fin = cmp_bf16_dev(d_mixed, ref_dir + "/final_mixed.bf16", size_t(S) * geom::HID);
        printf("[5] Final HC mixer: cos=%.7f rel_l2=%.6f max_abs=%.6f\n", c_fin.cos_sim, c_fin.rel_l2, c_fin.max_abs);
        if (c_fin.cos_sim < 0.9999) { fprintf(stderr, "FAIL Final HC mixer\n"); return 1; }
    }

    // --- 6. Layer-0 MoE PMCGX (exlr8-e4m3) + Weight Paging (GR8_PTAB=0, 1, 2 + ptab::set!) ---
    if (!rel_dir.empty()) {
        std::string wpath = rel_dir + "/exlr8-manifest-bit300.json";
        std::string mpath = "/tmp/qw38_block_test_manifest.json";
        char err[512] = {};
        void* plan = g6l_qw38_plan(rel_dir.c_str(), wpath.c_str(), 1, 0, 0, nullptr, mpath.c_str(), err, sizeof(err));
        if (!plan) { fprintf(stderr, "g6l_qw38_plan failed: %s\n", err); return 1; }
        g6::Pack pk;
        std::string perr;
        if (!pk.open_manifest(mpath, perr)) { fprintf(stderr, "open_manifest failed: %s\n", perr.c_str()); return 1; }
        auto load_plan_dev = [&](const std::string& name, size_t* out_bytes = nullptr) -> uint8_t* {
            const g6::Entry* te = pk.find(name);
            if (!te) { fprintf(stderr, "missing plan tensor %s\n", name.c_str()); exit(1); }
            std::vector<uint8_t> h(te->nbytes);
            if (g6l_plan_read(plan, te->offset, te->nbytes, h.data(), err, sizeof(err)) != 0) {
                fprintf(stderr, "g6l_plan_read %s failed: %s\n", name.c_str(), err);
                exit(1);
            }
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, te->nbytes));
            CK(cudaMemcpy(d, h.data(), te->nbytes, cudaMemcpyHostToDevice));
            if (out_bytes) *out_bytes = te->nbytes;
            return d;
        };

        qw38::MoeWeights mw{};
        mw.is_tr3 = true;
        mw.sh_i = geom::SHARED_I;
        mw.t_eb = g6nr::EB;
        mw.router = reinterpret_cast<const __nv_bfloat16*>(load_plan_dev("L000.router"));
        mw.sh_gate = reinterpret_cast<const __nv_bfloat16*>(load_plan_dev("L000.shared.gate_proj"));
        mw.sh_up = reinterpret_cast<const __nv_bfloat16*>(load_plan_dev("L000.shared.up_proj"));
        mw.sh_down = reinterpret_cast<const __nv_bfloat16*>(load_plan_dev("L000.shared.down_proj"));
        mw.sh_exp_gate = reinterpret_cast<const __nv_bfloat16*>(load_plan_dev("L000.shared.gate_w"));

        size_t fc1_b = 0, fc2_b = 0, tab_b = 0;
        uint8_t* d_fc1 = load_plan_dev("L000.tr3.fc1", &fc1_b);
        uint8_t* d_fc2 = load_plan_dev("L000.tr3.fc2", &fc2_b);
        mw.t_fc1 = d_fc1;
        mw.t_fc2 = d_fc2;
        mw.t_tab = reinterpret_cast<const long long*>(load_plan_dev("L000.tr3.table", &tab_b));
        mw.t_rows = int(tab_b / (6 * sizeof(long long)));
        mw.t_kmix = reinterpret_cast<const long long*>(load_plan_dev("L000.tr3.kmix"));
        mw.t_sg = reinterpret_cast<const __half*>(load_plan_dev("L000.tr3.sg"));
        mw.t_v1b = reinterpret_cast<const __half*>(load_plan_dev("L000.tr3.v1b"));
        mw.t_v2 = reinterpret_cast<const __half*>(load_plan_dev("L000.tr3.v2"));
        mw.t_v3b = reinterpret_cast<const __half*>(load_plan_dev("L000.tr3.v3b"));
        g6l_plan_free(plan);

        qw38::MoeScratch msc{};
        CK(cudaMalloc(&msc.router_logits, size_t(S) * geom::NEXP * 4));
        CK(cudaMalloc(&msc.topk_idx, size_t(S) * geom::TOPK * 4));
        CK(cudaMalloc(&msc.topk_w, size_t(S) * geom::TOPK * 4));
        CK(cudaMalloc(&msc.d_n_groups, 4));
        CK(cudaMalloc(&msc.grp_exp, size_t(S) * geom::TOPK * 4));
        CK(cudaMalloc(&msc.grp_cnt, size_t(S) * geom::TOPK * 4));
        CK(cudaMalloc(&msc.grp_routes, size_t(S) * geom::TOPK * 8 * 4));
        CK(cudaMalloc(&msc.fc1_part, size_t(8) * geom::TOPK * g6nr::EB * 2 * 2 * 128 * 4));
        CK(cudaMalloc(&msc.fc1_act, size_t(S) * geom::TOPK * g6nr::EB * 128 * 2));
        CK(cudaMalloc(&msc.fc2_part, size_t(S) * geom::TOPK * geom::HID * 4));
        CK(cudaMalloc(&msc.routed_acc, size_t(S) * geom::HID * 4));
        CK(cudaMalloc(&msc.sh_gate_buf, size_t(S) * geom::SHARED_I * 2));
        CK(cudaMalloc(&msc.sh_up_buf, size_t(S) * geom::SHARED_I * 2));
        CK(cudaMalloc(&msc.sh_out, size_t(S) * geom::HID * 2));
        CK(cudaMalloc(&msc.sh_scalar_gate, size_t(S) * 4));

        __nv_bfloat16* d_mlp_in = upload_bin<__nv_bfloat16>(ref_dir + "/l0_mlp_in.bf16");
        __nv_bfloat16* d_out0 = nullptr;
        __nv_bfloat16* d_out1 = nullptr;
        CK(cudaMalloc(&d_out0, size_t(S) * geom::HID * 2));
        CK(cudaMalloc(&d_out1, size_t(S) * geom::HID * 2));

        // GR8_PTAB=0 (contiguous)
        mw.t_ptab = nullptr;
        qw38::moe_forward(blas, mw, msc, d_mlp_in, d_out0, S, st);
        CK(cudaStreamSynchronize(st));
        auto c_moe = cmp_bf16_dev(d_out0, ref_dir + "/l0_moe_out.bf16", size_t(S) * geom::HID);
        printf("[6a] L0 MoE PMCGX (k3, GR8_PTAB=0) vs BF16 ref: cos=%.7f rel_l2=%.6f max_abs=%.6f\n",
               c_moe.cos_sim, c_moe.rel_l2, c_moe.max_abs);
        if (c_moe.cos_sim < 0.980) { fprintf(stderr, "FAIL L0 MoE PMCGX\n"); return 1; }

        std::vector<uint16_t> base(size_t(S) * geom::HID), cur(size_t(S) * geom::HID);
        CK(cudaMemcpy(base.data(), d_out0, base.size() * 2, cudaMemcpyDeviceToHost));

        // GR8_PTAB=1 (IMAGE piece table)
        ptab::Pool pool;
        ptab::Table pt1{};
        if (const char* pe = ptab::build(pt1, mw.t_tab, mw.t_kmix, d_fc1, d_fc2, mw.t_rows, g6nr::EB, ptab::IMAGE, pool, st)) {
            fprintf(stderr, "ptab::build IMAGE failed: %s\n", pe);
            return 1;
        }
        mw.t_ptab = pt1.dev;
        qw38::moe_forward(blas, mw, msc, d_mlp_in, d_out1, S, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(cur.data(), d_out1, cur.size() * 2, cudaMemcpyDeviceToHost));
        if (cur != base) { fprintf(stderr, "FAIL GR8_PTAB=1 parity\n"); return 1; }
        printf("[6b] L0 MoE GR8_PTAB=1 (IMAGE): 0 byte diffs vs PTAB=0\n");

        // GR8_PTAB=2 (SCATTER slab pool + live ptab::set relocation!)
        ptab::Table pt2{};
        if (const char* pe = ptab::build(pt2, mw.t_tab, mw.t_kmix, d_fc1, d_fc2, mw.t_rows, g6nr::EB, ptab::SCATTER, pool, st)) {
            fprintf(stderr, "ptab::build SCATTER failed: %s\n", pe);
            return 1;
        }
        mw.t_ptab = pt2.dev;
        mw.t_fc1 = nullptr;
        mw.t_fc2 = nullptr;
        qw38::moe_forward(blas, mw, msc, d_mlp_in, d_out1, S, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(cur.data(), d_out1, cur.size() * 2, cudaMemcpyDeviceToHost));
        if (cur != base) { fprintf(stderr, "FAIL GR8_PTAB=2 SCATTER parity\n"); return 1; }
        printf("[6c] L0 MoE GR8_PTAB=2 (SCATTER, %zu chunks, %zu MB slabs): 0 byte diffs vs PTAB=0\n",
               pool.chunks.size(), pt2.slab_bytes >> 20);

        // Live ptab::set relocation of the first active routed expert!
        int first_e = 0;
        CK(cudaMemcpy(&first_e, msc.topk_idx, 4, cudaMemcpyDeviceToHost));
        const int s = pt2.slot(first_e, 0);
        const size_t f1_b = pt2.bytes[s], f2_b = pt2.bytes[s + 1];
        const char* perr_t = nullptr;
        uint8_t* reloc1 = pool.take(f1_b, perr_t, true);
        uint8_t* reloc2 = pool.take(f2_b, perr_t, true);
        CK(cudaMemcpyAsync(reloc1, pt2.host[s], f1_b, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(reloc2, pt2.host[s + 1], f2_b, cudaMemcpyDeviceToDevice, st));
        ptab::Old old{};
        if (const char* pe = ptab::set(pt2, first_e, 0, reloc1, f1_b, reloc2, f2_b, st, &old)) {
            fprintf(stderr, "ptab::set failed: %s\n", pe);
            return 1;
        }
        qw38::moe_forward(blas, mw, msc, d_mlp_in, d_out1, S, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(cur.data(), d_out1, cur.size() * 2, cudaMemcpyDeviceToHost));
        if (cur != base) { fprintf(stderr, "FAIL GR8_PTAB=2 live ptab::set parity (e=%d)\n", first_e); return 1; }
        printf("[6d] L0 MoE live ptab::set relocation (expert %d, block 0): 0 byte diffs vs PTAB=0\n", first_e);
    }

    printf("ALL QW38 PER-BLOCK GPU EXACTNESS CHECKS PASSED!\n");
    return 0;
}
