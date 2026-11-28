// g6tts_bench: Teacher-forced parity dump (Gate E1), kernel parity verifier, decode microbenchmark,
// per-component profiler, and codec decode/encode harness for Drama 3 (g6tts).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts_backbone.cuh"
#include "tts_config.h"
#include "tts_sample.cuh"
#include "tts_tower.cuh"
#include "tts_weights.h"

#if __has_include("tts_codec.cuh")
#include "tts_codec.cuh"
#else
namespace g6tts {
struct TtsCodec {
    bool loaded_ = false;
    bool init(const std::string&, int, int = 4) { loaded_ = false; return false; }
    bool loaded() const { return loaded_; }
    void release() { loaded_ = false; }
    void reset_slot(int, cudaStream_t) {}
    void decode_frames(const int32_t*, int M, int stride, const int*, int16_t* d_pcm, cudaStream_t s) {
        cudaMemsetAsync(d_pcm, 0, size_t(M) * stride * geom::FRAME_SAMPLES * sizeof(int16_t), s);
    }
    int encode_pcm(const float*, int, std::vector<int32_t>& out_codes_10xF, cudaStream_t) {
        out_codes_10xF.clear();
        return -1;
    }
    void decode_whole(const int32_t*, int F, std::vector<int16_t>& out_pcm, cudaStream_t) {
        out_pcm.assign(size_t(F) * geom::FRAME_SAMPLES, 0);
    }
    void decode_streamed(const int32_t*, int F, const std::vector<int>&, std::vector<int16_t>& out_pcm, cudaStream_t) {
        out_pcm.assign(size_t(F) * geom::FRAME_SAMPLES, 0);
    }
};
} // namespace g6tts
#endif

#include "tts_graph.cuh"

namespace {

struct BenchOpts {
    std::string model_dir;
    std::string manifest;
    std::string codec_path;
    g6tts::WeightMode weight_mode = g6tts::WeightMode::Ref;
    bool weight_mode_explicit = false;
    bool verify_checksums = true;

    std::string teacher_in;
    std::string dump_out;
    std::string verify_rows;

    bool verify_kernels = false;
    bool verify_graph = false;
    bool run_scenarios = false;
    bool use_graph = true;
    bool profile = false;
    std::string scenarios = "m1,m4";
    std::string voc_strides = "1,1,2,4";
    int prompt_len = 64;
    int gen_frames = 32;
    int cap = 4096;

    std::string codec_decode_in;
    std::string codec_encode_in;
    std::string codec_out;
    std::string codec_strides;
};

void usage() {
    std::fprintf(stderr,
                 "usage: g6tts_bench --model-dir DIR [--manifest NAME] [--weight-mode ref|ref-fp16|ref-f32|native] [--no-hash]\n"
                 "                   [--teacher ROWS.bin --dump OUT.bin]\n"
                 "                   [--verify-kernels [--rows ROWS.bin]] [--verify-graph] [--scenarios [m1,m4]] [--graph|--no-graph] [--profile]\n"
                 "                   [--prompt-len N] [--frames N] [--cap N] [--codec-path PATH] [--voc-strides 1,1,2,4]\n"
                 "                   [--codec-decode CODES.bin --out PCM.bin [--codec-path PATH] [--codec-strides 1,1,2,4]]\n"
                 "                   [--codec-encode PCM.bin --out CODES.bin [--codec-path PATH]]\n");
}

bool parse_cli(int argc, char** argv, BenchOpts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* f) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "g6tts_bench: %s requires an argument\n", f);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            const char* v = need("--model-dir"); if (!v) return false; o.model_dir = v;
        } else if (a == "--manifest") {
            const char* v = need("--manifest"); if (!v) return false; o.manifest = v;
        } else if (a == "--codec-path") {
            const char* v = need("--codec-path"); if (!v) return false; o.codec_path = v;
        } else if (a == "--weight-mode") {
            const char* v = need("--weight-mode"); if (!v) return false;
            o.weight_mode_explicit = true;
            if (std::strcmp(v, "ref") == 0) o.weight_mode = g6tts::WeightMode::Ref;
            else if (std::strcmp(v, "ref-fp16") == 0) o.weight_mode = g6tts::WeightMode::RefFp16;
            else if (std::strcmp(v, "ref-f32") == 0) o.weight_mode = g6tts::WeightMode::RefF32;
            else if (std::strcmp(v, "native") == 0) o.weight_mode = g6tts::WeightMode::Native;
            else return false;
        } else if (a == "--no-hash") {
            o.verify_checksums = false;
        } else if (a == "--teacher") {
            const char* v = need("--teacher"); if (!v) return false; o.teacher_in = v;
        } else if (a == "--dump") {
            const char* v = need("--dump"); if (!v) return false; o.dump_out = v;
        } else if (a == "--rows") {
            const char* v = need("--rows"); if (!v) return false; o.verify_rows = v;
        } else if (a == "--verify-kernels") {
            o.verify_kernels = true;
        } else if (a == "--verify-graph") {
            o.verify_graph = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') o.scenarios = argv[++i];
        } else if (a == "--scenarios") {
            o.run_scenarios = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') o.scenarios = argv[++i];
        } else if (a == "--graph") {
            o.use_graph = true;
        } else if (a == "--no-graph") {
            o.use_graph = false;
        } else if (a == "--voc-strides") {
            const char* v = need("--voc-strides"); if (!v) return false; o.voc_strides = v;
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "--prompt-len") {
            const char* v = need("--prompt-len"); if (!v) return false; o.prompt_len = std::max(4, std::atoi(v));
        } else if (a == "--frames") {
            const char* v = need("--frames"); if (!v) return false; o.gen_frames = std::max(1, std::atoi(v));
        } else if (a == "--cap") {
            const char* v = need("--cap"); if (!v) return false; o.cap = std::max(64, std::atoi(v));
        } else if (a == "--codec-decode") {
            const char* v = need("--codec-decode"); if (!v) return false; o.codec_decode_in = v;
        } else if (a == "--codec-encode") {
            const char* v = need("--codec-encode"); if (!v) return false; o.codec_encode_in = v;
        } else if (a == "--out") {
            const char* v = need("--out"); if (!v) return false; o.codec_out = v;
        } else if (a == "--codec-strides") {
            const char* v = need("--codec-strides"); if (!v) return false; o.codec_strides = v;
        } else if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "g6tts_bench: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (!o.codec_decode_in.empty() || !o.codec_encode_in.empty()) return true;
    return !o.model_dir.empty();
}

// Binary format for `--teacher ROWS.bin`:
//   [4B magic "G6TR"][u32 n_seq]
//   for each seq: [u32 T][int32 rows[T * 11]]
std::vector<std::vector<int32_t>> read_teacher_rows(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("cannot open teacher rows file: " + path);
    char magic[4] = {};
    uint32_t n_seq = 0;
    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&n_seq), 4);
    if (!ifs || std::memcmp(magic, "G6TR", 4) != 0) {
        throw std::runtime_error("bad magic in teacher rows file (expected G6TR): " + path);
    }
    std::vector<std::vector<int32_t>> seqs(n_seq);
    for (uint32_t s = 0; s < n_seq; ++s) {
        uint32_t T = 0;
        ifs.read(reinterpret_cast<char*>(&T), 4);
        if (!ifs || T < 2) throw std::runtime_error("invalid sequence length in " + path);
        seqs[s].resize(size_t(T) * g6tts::geom::ROW_WIDTH);
        ifs.read(reinterpret_cast<char*>(seqs[s].data()), std::streamsize(seqs[s].size() * sizeof(int32_t)));
        if (!ifs) throw std::runtime_error("truncated teacher rows file: " + path);
    }
    return seqs;
}

struct TeacherSeqOut {
    int T = 0;
    std::vector<float> final_hidden; // [T, H]
    std::vector<float> main_logits;  // [T, 4097]
    std::vector<float> head_logits;  // [T-1, 9, 1024]
};

TeacherSeqOut run_teacher_seq(g6tts::BackboneRunner& bb, g6tts::TowerRunner& tower,
                              const std::vector<int32_t>& rows, cudaStream_t stream) {
    const auto& c = bb.W->cfg;
    const int rw = c.row_width();
    const int T = static_cast<int>(rows.size()) / rw;
    const int H = c.hidden_size;
    const int n_cand = c.main_candidates();

    TeacherSeqOut out;
    out.T = T;
    out.final_hidden.resize(size_t(T) * H);
    out.main_logits.resize(size_t(T) * n_cand);
    out.head_logits.resize(size_t(T - 1) * c.tower_res * g6tts::geom::RES_CB_SIZE);

    const int chunk = bb.max_chunk;
    for (int start = 0; start < T; start += chunk) {
        const int end = std::min(start + chunk, T);
        const int len = end - start;
        bb.forward(rows.data() + size_t(start) * rw, len,
                   true, 0, start, nullptr, nullptr, true, true, stream);
        TTS_CK(cudaMemcpyAsync(out.final_hidden.data() + size_t(start) * H,
                               bb.d_final_f32, size_t(len) * H * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
        TTS_CK(cudaMemcpyAsync(out.main_logits.data() + size_t(start) * n_cand,
                               bb.d_main_logits, size_t(len) * n_cand * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
        const int paired_end = std::min(end, T - 1);
        if (paired_end > start) {
            const int plen = paired_end - start;
            tower.prefill_chunk(bb.d_final_bf16,
                                rows.data() + size_t(start) * rw,
                                rows.data() + size_t(start + 1) * rw,
                                0, start, plen,
                                out.head_logits.data() + size_t(start) * c.tower_res * g6tts::geom::RES_CB_SIZE,
                                stream);
        }
        TTS_CK(cudaStreamSynchronize(stream));
    }
    return out;
}

void write_layer_trace(const std::string& path, const g6tts::TtsConfig& c,
                       int T, const g6tts::BackboneLayerTrace& bb_tr,
                       const g6tts::TowerLayerTrace& tw_tr) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("cannot open layer trace output: " + path);
    const char magic[4] = {'G', '6', 'L', '2'};
    const uint32_t uT = static_cast<uint32_t>(T);
    const uint32_t H = static_cast<uint32_t>(c.hidden_size);
    const uint32_t L = static_cast<uint32_t>(c.num_layers);
    const uint32_t K = static_cast<uint32_t>(c.top_k);
    const uint32_t n_tw = static_cast<uint32_t>(c.tower_shared + c.tower_res);
    ofs.write(magic, 4);
    ofs.write(reinterpret_cast<const char*>(&uT), 4);
    ofs.write(reinterpret_cast<const char*>(&H), 4);
    ofs.write(reinterpret_cast<const char*>(&L), 4);
    ofs.write(reinterpret_cast<const char*>(&K), 4);
    ofs.write(reinterpret_cast<const char*>(&n_tw), 4);
    auto wr_f32 = [&](const std::vector<float>& v) {
        ofs.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
    };
    wr_f32(bb_tr.embed_f32);
    wr_f32(bb_tr.final_hidden_f32);
    wr_f32(bb_tr.l0_qkv_raw);
    wr_f32(bb_tr.l0_qkv_conv);
    wr_f32(bb_tr.l0_core_out);
    wr_f32(bb_tr.l3_q_ready);
    wr_f32(bb_tr.l3_attn_out);
    for (uint32_t l = 0; l < L; ++l) {
        wr_f32(bb_tr.mixer_residuals[l]);
        wr_f32(bb_tr.residuals[l]);
        ofs.write(reinterpret_cast<const char*>(bb_tr.topk_ids[l].data()),
                  std::streamsize(bb_tr.topk_ids[l].size() * sizeof(int32_t)));
    }
    for (uint32_t tl = 0; tl < n_tw; ++tl) {
        wr_f32(tw_tr.hiddens[tl]);
    }
    for (int s = 0; s < 2; ++s) {
        wr_f32(tw_tr.sub[s].eh_cat);
        wr_f32(tw_tr.sub[s].eh_proj);
        wr_f32(tw_tr.sub[s].wqkv);
        wr_f32(tw_tr.sub[s].q_ready);
        wr_f32(tw_tr.sub[s].attn_out);
        wr_f32(tw_tr.sub[s].wo_res);
        wr_f32(tw_tr.sub[s].ffn_act);
        wr_f32(tw_tr.sub[s].w2_res);
    }
    if (!ofs) throw std::runtime_error("failed writing layer trace output: " + path);
}

void write_teacher_dump(const std::string& path, const g6tts::TtsConfig& c,
                        const std::vector<TeacherSeqOut>& outs) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("cannot open dump output: " + path);
    const char magic[4] = {'G', '6', 'T', 'D'};
    const uint32_t n_seq = static_cast<uint32_t>(outs.size());
    const uint32_t H = static_cast<uint32_t>(c.hidden_size);
    const uint32_t n_cand = static_cast<uint32_t>(c.main_candidates());
    const uint32_t n_heads = static_cast<uint32_t>(c.tower_res);
    const uint32_t h_vocab = static_cast<uint32_t>(g6tts::geom::RES_CB_SIZE);
    ofs.write(magic, 4);
    ofs.write(reinterpret_cast<const char*>(&n_seq), 4);
    ofs.write(reinterpret_cast<const char*>(&H), 4);
    ofs.write(reinterpret_cast<const char*>(&n_cand), 4);
    ofs.write(reinterpret_cast<const char*>(&n_heads), 4);
    ofs.write(reinterpret_cast<const char*>(&h_vocab), 4);
    for (const auto& o : outs) {
        const uint32_t T = static_cast<uint32_t>(o.T);
        ofs.write(reinterpret_cast<const char*>(&T), 4);
        ofs.write(reinterpret_cast<const char*>(o.final_hidden.data()), std::streamsize(o.final_hidden.size() * sizeof(float)));
        ofs.write(reinterpret_cast<const char*>(o.main_logits.data()), std::streamsize(o.main_logits.size() * sizeof(float)));
        ofs.write(reinterpret_cast<const char*>(o.head_logits.data()), std::streamsize(o.head_logits.size() * sizeof(float)));
    }
    if (!ofs) throw std::runtime_error("failed writing dump output: " + path);
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    return (na > 0.0 && nb > 0.0) ? (dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0;
}

std::vector<int32_t> make_synthetic_rows(int T, uint32_t seed) {
    std::vector<int32_t> rows(size_t(T) * g6tts::geom::ROW_WIDTH, 0);
    uint32_t s = 0x9e3779b9u ^ seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    const int text_prefix = std::max(2, T / 3);
    for (int t = 0; t < T; ++t) {
        if (t < text_prefix) {
            rows[size_t(t) * g6tts::geom::ROW_WIDTH] = 100 + int(rnd() % 50000u);
        } else {
            const int c0 = int(rnd() % 4096u);
            rows[size_t(t) * g6tts::geom::ROW_WIDTH] = g6tts::geom::SEM0 + c0;
            rows[size_t(t) * g6tts::geom::ROW_WIDTH + 1] = c0;
            for (int cb = 1; cb < g6tts::geom::NCB; ++cb) {
                rows[size_t(t) * g6tts::geom::ROW_WIDTH + 1 + cb] = int(rnd() % 1024u);
            }
        }
    }
    return rows;
}

int run_codec_cli(const BenchOpts& opts, cudaStream_t stream) {
    const std::string cpath = opts.codec_path.empty()
        ? (opts.model_dir + "/codec/exlr8-codec.safetensors")
        : opts.codec_path;
    if (opts.codec_out.empty()) {
        std::fprintf(stderr, "g6tts_bench: --out is required for --codec-decode / --codec-encode\n");
        return 2;
    }
    g6tts::TtsCodec codec;
    if (!codec.init(cpath, 1, 4)) {
        std::fprintf(stderr, "g6tts_bench: failed to load codec from %s\n", cpath.c_str());
        return 1;
    }
    if (!opts.codec_decode_in.empty()) {
        std::ifstream ifs(opts.codec_decode_in, std::ios::binary | std::ios::ate);
        if (!ifs) throw std::runtime_error("cannot open " + opts.codec_decode_in);
        const size_t nbytes = size_t(ifs.tellg());
        ifs.seekg(0);
        if (nbytes == 0 || (nbytes % (g6tts::geom::NCB * sizeof(int32_t))) != 0) {
            throw std::runtime_error("CODES.bin byte size must be a positive multiple of 10 * 4");
        }
        const int F = int(nbytes / (g6tts::geom::NCB * sizeof(int32_t)));
        std::vector<int32_t> codes(size_t(F) * g6tts::geom::NCB);
        ifs.read(reinterpret_cast<char*>(codes.data()), std::streamsize(nbytes));
        std::vector<int16_t> pcm;
        if (!opts.codec_strides.empty()) {
            std::vector<int> strides;
            std::stringstream ss(opts.codec_strides);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                const int v = std::atoi(tok.c_str());
                if (v >= 1) strides.push_back(v);
            }
            codec.decode_streamed(codes.data(), F, strides, pcm, stream);
        } else {
            codec.decode_whole(codes.data(), F, pcm, stream);
        }
        std::ofstream ofs(opts.codec_out, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(pcm.data()), std::streamsize(pcm.size() * sizeof(int16_t)));
        std::fprintf(stderr, "g6tts_bench: decoded %d frames -> %zu s16 samples (%s)\n",
                     F, pcm.size(), opts.codec_out.c_str());
        return 0;
    }
    // --codec-encode
    std::ifstream ifs(opts.codec_encode_in, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("cannot open " + opts.codec_encode_in);
    const size_t nbytes = size_t(ifs.tellg());
    ifs.seekg(0);
    if (nbytes == 0 || (nbytes % sizeof(float)) != 0) {
        throw std::runtime_error("PCM.bin byte size must be a positive multiple of 4");
    }
    const int n_samples = int(nbytes / sizeof(float));
    std::vector<float> pcm(static_cast<size_t>(n_samples));
    ifs.read(reinterpret_cast<char*>(pcm.data()), std::streamsize(nbytes));
    std::vector<int32_t> codes;
    const int F = codec.encode_pcm(pcm.data(), n_samples, codes, stream);
    if (F < 0) {
        std::fprintf(stderr, "g6tts_bench: codec encode failed\n");
        return 1;
    }
    std::ofstream ofs(opts.codec_out, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(codes.data()), std::streamsize(codes.size() * sizeof(int32_t)));
    std::fprintf(stderr, "g6tts_bench: encoded %d f32 samples -> [10, %d] codes (%s)\n",
                 n_samples, F, opts.codec_out.c_str());
    return 0;
}

uint64_t fnv1a_i32(const std::vector<int32_t>& v) {
    uint64_t h = 14695981039346656037ull;
    for (int32_t x : v) {
        const uint32_t u = static_cast<uint32_t>(x);
        for (int b = 0; b < 4; ++b) {
            h ^= uint64_t((u >> (b * 8)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

std::vector<int> parse_voc_strides_bench(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int v = std::atoi(tok.c_str());
        if (v == 1 || v == 2 || v == 4) out.push_back(v);
    }
    if (out.empty()) out = {1, 1, 2, 4};
    return out;
}

struct ScenarioRun {
    int M = 1;
    int frames = 0;
    bool graphed = false;
    bool codec_loaded = false;
    double total_ms = 0.0;
    double ms_per_step = 0.0;
    double frames_per_s = 0.0;
    double audio_s_per_s = 0.0;
    uint64_t codes_hash = 0;
    std::vector<int32_t> codes; // [frames, M, 11]
    std::vector<int16_t> pcm;   // concatenated decoded PCM when codec is loaded
    g6tts::BackboneProfile bb_prof;
    g6tts::TowerProfile tw_prof;
    g6tts::FrameProfile fr_prof;
};

ScenarioRun run_graph_scenario(const g6tts::TtsWeights& W, const BenchOpts& opts,
                               int M, bool enable_graph, bool enable_profile,
                               cudaStream_t stream) {
    const int cap_eff = std::max(opts.cap, opts.prompt_len + opts.gen_frames + 16);
    const int total_slots = M + 1; // slot M is throwaway pad_slot
    g6tts::BackboneRunner bb;
    g6tts::TowerRunner tw;
    g6tts::TtsCodec codec;
    g6tts::TtsGraphRunner<g6tts::TtsCodec> gr;

    bb.init(W, total_slots, cap_eff, std::max(128, opts.prompt_len));
    tw.init(W, total_slots, cap_eff, std::max(128, opts.prompt_len));

    const std::string cpath = opts.codec_path.empty()
        ? (opts.model_dir + "/codec/exlr8-codec.safetensors")
        : opts.codec_path;
    const bool has_codec = codec.init(cpath, total_slots, 4);
    const bool eff_graph = enable_graph && !enable_profile;
    gr.init(W, bb, tw, codec, M, cap_eff, eff_graph);

    const int H = W.cfg.hidden_size;
    const int rw = W.cfg.row_width();
    g6tts::bf16* d_tmp_hid = nullptr;
    TTS_CK(cudaMalloc(&d_tmp_hid, size_t(H) * sizeof(g6tts::bf16)));

    std::vector<int> active_slots(static_cast<size_t>(M));
    for (int m = 0; m < M; ++m) {
        active_slots[size_t(m)] = m;
        if (has_codec) codec.reset_slot(m, stream);
        const auto pr = make_synthetic_rows(opts.prompt_len, uint32_t(100 + m));
        bb.forward(pr.data(), opts.prompt_len, true, m, 0, nullptr, nullptr, false, false, stream);
        tw.prefill_chunk(bb.d_final_bf16, pr.data(), pr.data() + rw, m, 0, opts.prompt_len - 1, nullptr, stream);
        TTS_CK(cudaMemcpyAsync(d_tmp_hid, bb.d_final_bf16 + size_t(opts.prompt_len - 1) * H,
                               size_t(H) * sizeof(g6tts::bf16), cudaMemcpyDeviceToDevice, stream));
        int32_t ring[g6tts::geom::RAS_WINDOW];
        std::fill(std::begin(ring), std::end(ring), -1);
        const int take = std::min(opts.prompt_len, g6tts::geom::RAS_WINDOW);
        for (int i = 0; i < take; ++i) {
            ring[i] = pr[size_t(opts.prompt_len - take + i) * rw];
        }
        gr.admit_slot(m, d_tmp_hid, pr.data() + size_t(opts.prompt_len - 1) * rw,
                      opts.prompt_len - 1, ring, take,
                      0.7f, 0.7f, 30, 0x1000ull + uint64_t(m), stream);
    }
    cudaFree(d_tmp_hid);

    // Pre-capture CUDA graphs using throwaway pad_slot (does not perturb live slots 0..M-1!).
    if (eff_graph) {
        gr.ensure_frame_graph(M, stream);
        if (has_codec) {
            for (int st : g6tts::CODEC_STRIDES) gr.ensure_codec_graph(M, st, stream);
        }
    }

    bb.prof.enabled = enable_profile;
    tw.prof.enabled = enable_profile;
    gr.prof.enabled = enable_profile;
    bb.prof.reset();
    tw.prof.reset();
    gr.prof.reset();

    const auto voc_strides = parse_voc_strides_bench(opts.voc_strides);
    size_t stride_idx = 0;
    int ring_read_pos = 0;
    int ring_unflushed = 0;

    ScenarioRun out;
    out.M = M;
    out.frames = opts.gen_frames;
    out.graphed = eff_graph;
    out.codec_loaded = has_codec;
    out.codes.resize(size_t(opts.gen_frames) * M * rw);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    TTS_CK(cudaEventCreate(&ev0));
    TTS_CK(cudaEventCreate(&ev1));
    TTS_CK(cudaEventRecord(ev0, stream));

    for (int step = 0; step < opts.gen_frames; ++step) {
        gr.step_frame(active_slots, stream);
        std::memcpy(out.codes.data() + size_t(step) * M * rw,
                    gr.h_out_rows, size_t(M) * rw * sizeof(int32_t));
        if (has_codec) {
            ++ring_unflushed;
            const int target_stride = voc_strides[std::min(stride_idx, voc_strides.size() - 1)];
            if (ring_unflushed >= target_stride) {
                std::vector<int> starts(static_cast<size_t>(M), ring_read_pos);
                gr.decode_codec_batch(active_slots, starts, target_stride, stream);
                const size_t n_samp = size_t(M) * target_stride * g6tts::geom::FRAME_SAMPLES;
                out.pcm.insert(out.pcm.end(), gr.h_codec_pcm, gr.h_codec_pcm + n_samp);
                ring_read_pos += target_stride;
                ring_unflushed -= target_stride;
                if (stride_idx + 1 < voc_strides.size()) ++stride_idx;
            }
        }
    }
    if (has_codec && ring_unflushed > 0) {
        while (ring_unflushed > 0) {
            const int s = (ring_unflushed >= 4) ? 4 : ((ring_unflushed >= 2) ? 2 : 1);
            std::vector<int> starts(static_cast<size_t>(M), ring_read_pos);
            gr.decode_codec_batch(active_slots, starts, s, stream);
            const size_t n_samp = size_t(M) * s * g6tts::geom::FRAME_SAMPLES;
            out.pcm.insert(out.pcm.end(), gr.h_codec_pcm, gr.h_codec_pcm + n_samp);
            ring_read_pos += s;
            ring_unflushed -= s;
        }
    }

    TTS_CK(cudaEventRecord(ev1, stream));
    TTS_CK(cudaEventSynchronize(ev1));
    float total_ms = 0.0f;
    TTS_CK(cudaEventElapsedTime(&total_ms, ev0, ev1));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    out.total_ms = double(total_ms);
    out.ms_per_step = double(total_ms) / double(opts.gen_frames);
    out.frames_per_s = (double(M) * double(opts.gen_frames)) / std::max(1e-9, double(total_ms) * 1e-3);
    out.audio_s_per_s = out.frames_per_s / (double(g6tts::geom::SAMPLE_RATE) / double(g6tts::geom::FRAME_SAMPLES));
    out.codes_hash = fnv1a_i32(out.codes);
    out.bb_prof = bb.prof;
    out.tw_prof = tw.prof;
    out.fr_prof = gr.prof;
    return out;
}

} // namespace

int main(int argc, char** argv) {
    BenchOpts opts;
    if (!parse_cli(argc, argv, opts)) {
        usage();
        return 2;
    }

    try {
        cudaStream_t stream = nullptr;
        TTS_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        if (!opts.codec_decode_in.empty() || !opts.codec_encode_in.empty()) {
            const int rc = run_codec_cli(opts, stream);
            cudaStreamDestroy(stream);
            return rc;
        }

        if (!opts.teacher_in.empty()) {
            if (opts.dump_out.empty()) {
                std::fprintf(stderr, "g6tts_bench: --dump OUT.bin is required with --teacher\n");
                return 2;
            }
            const auto seqs = read_teacher_rows(opts.teacher_in);
            int max_T = 64;
            for (const auto& s : seqs) max_T = std::max(max_T, int(s.size()) / g6tts::geom::ROW_WIDTH);
            auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
            g6tts::BackboneRunner bb;
            g6tts::TowerRunner tower;
            bb.init(W, 1, std::max(opts.cap, max_T + 16), 128);
            tower.init(W, 1, std::max(opts.cap, max_T + 16), 128);
            std::vector<TeacherSeqOut> outs;
            outs.reserve(seqs.size());
            for (size_t i = 0; i < seqs.size(); ++i) {
                outs.push_back(run_teacher_seq(bb, tower, seqs[i], stream));
            }
            write_teacher_dump(opts.dump_out, W.cfg, outs);
            // Also write compact layer-by-layer trace for seq 0 (<= 64 rows, 1 chunk) for Gate E1 diagnostics.
            if (!seqs.empty()) {
                const int rw = W.cfg.row_width();
                const int T0 = std::min(64, int(seqs[0].size()) / rw);
                if (T0 >= 2) {
                    std::vector<int32_t> s0(seqs[0].begin(), seqs[0].begin() + size_t(T0) * rw);
                    bb.trace.enabled = true;
                    tower.trace.enabled = true;
                    (void)run_teacher_seq(bb, tower, s0, stream);
                    bb.trace.enabled = false;
                    tower.trace.enabled = false;
                    write_layer_trace(opts.dump_out + ".trace0", W.cfg, T0, bb.trace, tower.trace);
                }
            }
            std::fprintf(stderr, "g6tts_bench: dumped %zu teacher-forced sequences -> %s\n",
                         outs.size(), opts.dump_out.c_str());
            cudaStreamDestroy(stream);
            return 0;
        }

        if (opts.verify_kernels) {
            // Gate E3: compare Native (default) vs Ref on the exact same release across backbone, all 9 heads,
            // and standalone per-format linear unit probes on identical BF16 inputs.
            const g6tts::WeightMode test_mode = opts.weight_mode_explicit ? opts.weight_mode : g6tts::WeightMode::Native;
            std::vector<int32_t> rows;
            if (!opts.verify_rows.empty()) {
                const auto seqs = read_teacher_rows(opts.verify_rows);
                if (seqs.empty()) throw std::runtime_error("--rows file contains no sequences");
                const int rw = g6tts::geom::ROW_WIDTH;
                const int T = std::min(32, int(seqs[0].size()) / rw);
                rows.assign(seqs[0].begin(), seqs[0].begin() + size_t(T) * rw);
            } else {
                rows = make_synthetic_rows(32, 42);
            }

            // Deterministic BF16 probe input X[31, 6144] (odd M=31 matches TowerRunner prefill & covers w2 K=6144!)
            constexpr int OP_M = 31;
            constexpr int OP_K = 6144;
            std::vector<uint16_t> h_x_bf16(size_t(OP_M) * OP_K);
            {
                uint32_t s = 0x51a7b3c9u;
                for (size_t i = 0; i < h_x_bf16.size(); ++i) {
                    s = s * 1664525u + 1013904223u;
                    const float fv = (float(int32_t(s >> 8) - 8388608) / 8388608.0f) * 0.5f;
                    h_x_bf16[i] = g6tts::host_f32_to_bf16(fv);
                }
            }
            struct OpProbeSet {
                std::vector<float> tr_out;     // mtp.0.wqkv [9216, 2048]
                std::vector<float> tr_in;      // mtp.0.eh_proj [2048, 4096]
                std::vector<float> tr_dual;    // mtp.0.w1+w3 [6144+6144, 2048]
                std::vector<float> tr_triple;  // attn.3.q+k+v [8192+512+512, 2048]
                std::vector<float> int4_op;    // attn.3.o_proj [2048, 4096]
                std::vector<float> fp8_row;    // heads.0 [1024, 2048]
                std::vector<float> lm_head_op; // lm_head [4097, 2048]
                std::vector<float> bf16_op;    // gdn.0.router_gate [256, 2048]
                std::vector<float> gdn0_dual;  // gdn.0.in_proj_qkv + in_proj_z [8192+4096, 2048]
                std::vector<float> gdn0_out;   // gdn.0.out_proj [2048, 4096] (trellis side=IN)
                std::vector<float> gdn2_out_i4;// gdn.2.out_proj [2048, 4096] (int4)
                std::vector<float> sh0_w2_i4;            // mtp_shared.0.w2 [2048, 6144] (int4)
                std::vector<float> sh1_w2_tr;            // mtp_shared.1.w2 [2048, 6144] (trellis K=6144)
                std::vector<float> w2_outlier_tr;        // mtp.0.w2 [2048, 6144] with uniform >65504 outlier input
                std::vector<float> w2_sparse_outlier_tr; // mtp.0.w2 [2048, 6144] with 1 sparse 1e5 channel + 6143 O(0.5) channels
                std::vector<std::vector<float>> tw_all;  // all 13*6 = 78 tower matrices (shared 0..3, res 0..8)
                std::vector<std::string> tw_all_names;
                std::vector<int> tw_all_fmts;
            };
            std::vector<uint16_t> h_x_outlier_bf16(size_t(OP_M) * OP_K);
            std::vector<uint16_t> h_x_sparse_bf16 = h_x_bf16;
            for (size_t i = 0; i < h_x_outlier_bf16.size(); ++i) {
                h_x_outlier_bf16[i] = g6tts::host_f32_to_bf16(g6tts::host_bf16_to_f32(h_x_bf16[i]) * 200000.0f);
            }
            for (int m = 0; m < OP_M; ++m) {
                h_x_sparse_bf16[size_t(m) * OP_K + ((m * 137) % OP_K)] = g6tts::host_f32_to_bf16(100000.0f);
            }
            auto run_op_probes = [&](const g6tts::TtsWeights& W, g6tts::BackboneRunner& bb,
                                     g6tts::TowerRunner& tw) -> OpProbeSet {
                OpProbeSet out;
                const int H = W.cfg.hidden_size;
                const int n_cand = W.cfg.main_candidates();
                g6tts::bf16* d_x = nullptr;
                g6tts::bf16* d_y0 = nullptr;
                g6tts::bf16* d_y1 = nullptr;
                g6tts::bf16* d_y2 = nullptr;
                float* d_yf = nullptr;
                TTS_CK(cudaMalloc(&d_x, size_t(OP_M) * OP_K * sizeof(g6tts::bf16)));
                TTS_CK(cudaMalloc(&d_y0, size_t(OP_M) * 9216 * sizeof(g6tts::bf16)));
                TTS_CK(cudaMalloc(&d_y1, size_t(OP_M) * 6144 * sizeof(g6tts::bf16)));
                TTS_CK(cudaMalloc(&d_y2, size_t(OP_M) * 1024 * sizeof(g6tts::bf16)));
                TTS_CK(cudaMalloc(&d_yf, size_t(OP_M) * std::max(4096, n_cand) * sizeof(float)));
                TTS_CK(cudaMemcpyAsync(d_x, h_x_bf16.data(), h_x_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

                auto dl_bf16 = [&](const g6tts::bf16* d_p, size_t n) -> std::vector<float> {
                    std::vector<uint16_t> raw(n);
                    TTS_CK(cudaMemcpyAsync(raw.data(), d_p, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
                    TTS_CK(cudaStreamSynchronize(stream));
                    std::vector<float> v(n);
                    for (size_t i = 0; i < n; ++i) v[i] = g6tts::host_bf16_to_f32(raw[i]);
                    return v;
                };
                auto dl_f32 = [&](const float* d_p, size_t n) -> std::vector<float> {
                    std::vector<float> v(n);
                    TTS_CK(cudaMemcpyAsync(v.data(), d_p, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    TTS_CK(cudaStreamSynchronize(stream));
                    return v;
                };

                TTS_BLAS_CK(cublasSetStream(bb.blas, stream));
                TTS_BLAS_CK(cublasSetStream(tw.blas, stream));
                const auto& tw0 = W.res_layers[0];
                const auto& sh0 = W.shared_layers[0];
                const auto& sh1 = W.shared_layers[1];
                const auto& att3 = W.layers[3];
                const auto& gdn0 = W.layers[0];
                const auto& gdn2 = W.layers[2];

                // 1. TRELLIS side=OUT single (wqkv [9216, 2048])
                tw.proj_bf16(d_x, tw0.wqkv, tw0.wqkv_nat, d_y0, OP_M, 9216, H, stream);
                out.tr_out = dl_bf16(d_y0, size_t(OP_M) * 9216);

                // 2. TRELLIS side=IN split-K (eh_proj [2048, 4096])
                tw.proj_f32(d_x, tw0.eh_proj, tw0.eh_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.tr_in = dl_f32(d_yf, size_t(OP_M) * H);

                // 3. TRELLIS dual (w1 + w3 [6144, 2048] x 2)
                if (W.mode == g6tts::WeightMode::Native && tw0.w1_nat.valid() && tw0.w3_nat.valid()) {
                    g6tts::tts_native_linear_dual_bf16(d_x, tw0.w1_nat, tw0.w3_nat, d_y0, d_y1, OP_M, tw.dense_ws, tw.blas, stream);
                } else {
                    tw.proj_bf16(d_x, tw0.w1, tw0.w1_nat, d_y0, OP_M, 6144, H, stream);
                    tw.proj_bf16(d_x, tw0.w3, tw0.w3_nat, d_y1, OP_M, 6144, H, stream);
                }
                out.tr_dual = dl_bf16(d_y0, size_t(OP_M) * 6144);
                auto d1 = dl_bf16(d_y1, size_t(OP_M) * 6144);
                out.tr_dual.insert(out.tr_dual.end(), d1.begin(), d1.end());

                // 4. TRELLIS triple (attn.3 q+k+v [8192, 512, 512])
                if (W.mode == g6tts::WeightMode::Native && att3.q_proj_nat.valid() && att3.k_proj_nat.valid() && att3.v_proj_nat.valid()) {
                    g6tts::tts_native_linear_triple_bf16(d_x, att3.q_proj_nat, att3.k_proj_nat, att3.v_proj_nat,
                                                         d_y0, d_y1, d_y2, OP_M, bb.dense_ws, bb.blas, stream);
                } else {
                    bb.proj_bf16(d_x, att3.q_proj, att3.q_proj_nat, d_y0, OP_M, 8192, H, stream);
                    bb.proj_bf16(d_x, att3.k_proj, att3.k_proj_nat, d_y1, OP_M, 512, H, stream);
                    bb.proj_bf16(d_x, att3.v_proj, att3.v_proj_nat, d_y2, OP_M, 512, H, stream);
                }
                out.tr_triple = dl_bf16(d_y0, size_t(OP_M) * 8192);
                auto tk = dl_bf16(d_y1, size_t(OP_M) * 512);
                auto tv = dl_bf16(d_y2, size_t(OP_M) * 512);
                out.tr_triple.insert(out.tr_triple.end(), tk.begin(), tk.end());
                out.tr_triple.insert(out.tr_triple.end(), tv.begin(), tv.end());

                // 5. INT4 (attn.3.o_proj [2048, 4096])
                bb.proj_f32(d_x, att3.o_proj, att3.o_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.int4_op = dl_f32(d_yf, size_t(OP_M) * H);

                // 6. FP8_ROW (heads.0 [1024, 2048])
                tw.proj_f32(d_x, W.heads[0], W.heads_nat[0], d_yf, OP_M, g6tts::geom::RES_CB_SIZE, H, false, stream);
                out.fp8_row = dl_f32(d_yf, size_t(OP_M) * g6tts::geom::RES_CB_SIZE);

                // 7. FP8_ROW odd-N (lm_head [4097, 2048])
                bb.proj_f32(d_x, W.lm_head_cand, W.lm_head_nat, d_yf, OP_M, n_cand, H, false, stream);
                out.lm_head_op = dl_f32(d_yf, size_t(OP_M) * n_cand);

                // 8. BF16 (gdn.0.router_gate [256, 2048])
                bb.proj_f32(d_x, gdn0.router_gate, gdn0.router_gate_nat, d_yf, OP_M, W.cfg.num_experts, H, false, stream);
                out.bf16_op = dl_f32(d_yf, size_t(OP_M) * W.cfg.num_experts);

                // 9. gdn.0 in_proj_qkv + in_proj_z dual [8192+4096, 2048]
                if (W.mode == g6tts::WeightMode::Native && gdn0.in_proj_qkv_nat.valid() && gdn0.in_proj_z_nat.valid()) {
                    g6tts::tts_native_linear_dual_bf16(d_x, gdn0.in_proj_qkv_nat, gdn0.in_proj_z_nat, d_y0, d_y1, OP_M, bb.dense_ws, bb.blas, stream);
                } else {
                    bb.proj_bf16(d_x, gdn0.in_proj_qkv, gdn0.in_proj_qkv_nat, d_y0, OP_M, 8192, H, stream);
                    bb.proj_bf16(d_x, gdn0.in_proj_z, gdn0.in_proj_z_nat, d_y1, OP_M, 4096, H, stream);
                }
                out.gdn0_dual = dl_bf16(d_y0, size_t(OP_M) * 8192);
                auto gz = dl_bf16(d_y1, size_t(OP_M) * 4096);
                out.gdn0_dual.insert(out.gdn0_dual.end(), gz.begin(), gz.end());

                // 10. gdn.0 out_proj [2048, 4096] (trellis side=IN)
                bb.proj_f32(d_x, gdn0.out_proj, gdn0.out_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.gdn0_out = dl_f32(d_yf, size_t(OP_M) * H);

                // 11. gdn.2 out_proj [2048, 4096] (int4)
                bb.proj_f32(d_x, gdn2.out_proj, gdn2.out_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.gdn2_out_i4 = dl_f32(d_yf, size_t(OP_M) * H);

                // 12. mtp_shared.0 w2 [2048, 6144] (int4 K=6144)
                tw.proj_f32(d_x, sh0.w2, sh0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.sh0_w2_i4 = dl_f32(d_yf, size_t(OP_M) * H);

                // 13. mtp_shared.1 w2 [2048, 6144] (trellis K=6144)
                tw.proj_f32(d_x, sh1.w2, sh1.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.sh1_w2_tr = dl_f32(d_yf, size_t(OP_M) * H);

                // 14. mtp.0 w2 [2048, 6144] with uniform >65504 outlier activations (tests pow2 Trellis row scaling)
                TTS_CK(cudaMemcpyAsync(d_x, h_x_outlier_bf16.data(), h_x_outlier_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                tw.proj_f32(d_x, tw0.w2, tw0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.w2_outlier_tr = dl_f32(d_yf, size_t(OP_M) * H);

                // 15. mtp.0 w2 [2048, 6144] with 1 sparse 1e5 channel + 6143 normal channels (tests no FP16 subnormal underflow!)
                TTS_CK(cudaMemcpyAsync(d_x, h_x_sparse_bf16.data(), h_x_sparse_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                tw.proj_f32(d_x, tw0.w2, tw0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.w2_sparse_outlier_tr = dl_f32(d_yf, size_t(OP_M) * H);

                // 16. Exhaustive sweep of all 13 tower layers x 6 projections (78 matrices) on normal BF16 probe input.
                TTS_CK(cudaMemcpyAsync(d_x, h_x_bf16.data(), h_x_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                auto sweep_layer = [&](const g6tts::TowerLayerW& tl, const std::string& pfx) {
                    auto add_f32 = [&](const char* nm, const g6tts::RefMat& rm, const g6emb::DevMatrix& dm, int N, int K) {
                        tw.proj_f32(d_x, rm, dm, d_yf, OP_M, N, K, false, stream);
                        out.tw_all.push_back(dl_f32(d_yf, size_t(OP_M) * N));
                        out.tw_all_names.push_back(pfx + "." + nm);
                        out.tw_all_fmts.push_back(dm.valid() ? dm.fmt : -1);
                    };
                    auto add_bf16 = [&](const char* nm, const g6tts::RefMat& rm, const g6emb::DevMatrix& dm, int N, int K) {
                        tw.proj_bf16(d_x, rm, dm, d_y0, OP_M, N, K, stream);
                        out.tw_all.push_back(dl_bf16(d_y0, size_t(OP_M) * N));
                        out.tw_all_names.push_back(pfx + "." + nm);
                        out.tw_all_fmts.push_back(dm.valid() ? dm.fmt : -1);
                    };
                    add_f32("eh_proj", tl.eh_proj, tl.eh_proj_nat, H, 2 * H);
                    add_bf16("wqkv", tl.wqkv, tl.wqkv_nat, 9216, H);
                    add_f32("wo", tl.wo, tl.wo_nat, H, 2 * H);
                    add_bf16("w1", tl.w1, tl.w1_nat, W.cfg.tower_intermediate, H);
                    add_bf16("w3", tl.w3, tl.w3_nat, W.cfg.tower_intermediate, H);
                    add_f32("w2", tl.w2, tl.w2_nat, H, W.cfg.tower_intermediate);
                };
                for (int i = 0; i < W.cfg.tower_shared; ++i) {
                    sweep_layer(W.shared_layers[size_t(i)], "sh" + std::to_string(i));
                }
                for (int i = 0; i < W.cfg.tower_res; ++i) {
                    sweep_layer(W.res_layers[size_t(i)], "res" + std::to_string(i));
                }

                cudaFree(d_x); cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_yf);
                return out;
            };

            TeacherSeqOut ref_out, test_out, test_same_router_out;
            OpProbeSet ref_ops, test_ops;
            g6tts::BackboneLayerTrace ref_bb_tr, test_bb_tr, test_sr_bb_tr;
            g6tts::TowerLayerTrace ref_tw_tr, test_tw_tr, test_iso_tw_tr;
            std::vector<float> test_iso_heads;
            {
                // Use exact FP32 decoded dense weights as the mathematical reference for Gate E3
                // (eliminates 7-bit BF16 weight rounding noise vs un-rounded Native Trellis/INT4/FP8_ROW).
                auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums,
                                                 g6tts::WeightMode::RefF32);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, 1, 256, 128); tw.init(W, 1, 256, 128);
                ref_ops = run_op_probes(W, bb, tw);
                bb.trace.enabled = true;
                tw.trace.enabled = true;
                tw.trace.sub_kv[0] = 8;  // res 4
                tw.trace.sub_kv[1] = 11; // res 7
                ref_out = run_teacher_seq(bb, tw, rows, stream);
                ref_bb_tr = std::move(bb.trace);
                ref_tw_tr = std::move(tw.trace);
            }
            {
                auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, test_mode);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, 1, 256, 128); tw.init(W, 1, 256, 128);
                test_ops = run_op_probes(W, bb, tw);
                bb.trace.enabled = true;
                tw.trace.enabled = true;
                tw.trace.sub_kv[0] = 8;
                tw.trace.sub_kv[1] = 11;
                test_out = run_teacher_seq(bb, tw, rows, stream);
                test_bb_tr = std::move(bb.trace);
                test_tw_tr = std::move(tw.trace);

                // Controlled check A: Native backbone with Ref's exact router decisions (separates router flips vs kernel math)
                bb.trace.enabled = true;
                bb.trace.replay_router = true;
                bb.trace.forced_topk_ids = ref_bb_tr.topk_ids;
                bb.trace.forced_topk_w = ref_bb_tr.topk_w;
                tw.trace.enabled = false;
                test_same_router_out = run_teacher_seq(bb, tw, rows, stream);
                test_sr_bb_tr = std::move(bb.trace);

                // Controlled check B: Native tower fed Ref's exact backbone final_hidden (separates tower kernels vs bb drift)
                const int T = ref_out.T;
                const int H = W.cfg.hidden_size;
                const int rw = W.cfg.row_width();
                std::vector<uint16_t> h_ref_bf16(size_t(T) * H);
                for (size_t i = 0; i < h_ref_bf16.size(); ++i) {
                    h_ref_bf16[i] = g6tts::host_f32_to_bf16(ref_out.final_hidden[i]);
                }
                g6tts::bf16* d_ref_hid = nullptr;
                TTS_CK(cudaMalloc(&d_ref_hid, h_ref_bf16.size() * sizeof(uint16_t)));
                TTS_CK(cudaMemcpyAsync(d_ref_hid, h_ref_bf16.data(), h_ref_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                test_iso_heads.resize(size_t(T - 1) * W.cfg.tower_res * g6tts::geom::RES_CB_SIZE);
                tw.trace.enabled = true;
                tw.trace.sub_kv[0] = 8;
                tw.trace.sub_kv[1] = 11;
                tw.prefill_chunk(d_ref_hid, rows.data(), rows.data() + rw, 0, 0, T - 1, test_iso_heads.data(), stream);
                TTS_CK(cudaStreamSynchronize(stream));
                test_iso_tw_tr = std::move(tw.trace);
                cudaFree(d_ref_hid);
            }
            const double c_hid = cosine_sim(ref_out.final_hidden, test_out.final_hidden);
            const double c_main = cosine_sim(ref_out.main_logits, test_out.main_logits);
            const double c_head = cosine_sim(ref_out.head_logits, test_out.head_logits);
            const double c_hid_sr = cosine_sim(ref_out.final_hidden, test_same_router_out.final_hidden);
            const double c_main_sr = cosine_sim(ref_out.main_logits, test_same_router_out.main_logits);
            const int n_heads = g6tts::geom::TOW_RES;
            const int hv = g6tts::geom::RES_CB_SIZE;
            const int n_rows = std::max(0, ref_out.T - 1);
            double head_cos[g6tts::geom::TOW_RES];
            double head_iso_cos[g6tts::geom::TOW_RES];
            double head_min = 1.0, head_iso_min = 1.0;
            for (int h = 0; h < n_heads; ++h) {
                std::vector<float> ra, ta, ia;
                ra.reserve(size_t(n_rows) * hv);
                ta.reserve(size_t(n_rows) * hv);
                ia.reserve(size_t(n_rows) * hv);
                for (int r = 0; r < n_rows; ++r) {
                    const size_t base = (size_t(r) * n_heads + h) * hv;
                    ra.insert(ra.end(), ref_out.head_logits.begin() + base, ref_out.head_logits.begin() + base + hv);
                    ta.insert(ta.end(), test_out.head_logits.begin() + base, test_out.head_logits.begin() + base + hv);
                    ia.insert(ia.end(), test_iso_heads.begin() + base, test_iso_heads.begin() + base + hv);
                }
                head_cos[h] = cosine_sim(ra, ta);
                head_iso_cos[h] = cosine_sim(ra, ia);
                head_min = std::min(head_min, head_cos[h]);
                head_iso_min = std::min(head_iso_min, head_iso_cos[h]);
            }

            // Per-layer backbone & tower cosines + router top-k set overlap
            const int n_bb = static_cast<int>(std::min(ref_bb_tr.residuals.size(), test_bb_tr.residuals.size()));
            std::vector<double> bb_lcos(size_t(n_bb), 0.0);
            std::vector<double> bb_lcos_sr(size_t(n_bb), 0.0);
            std::vector<double> bb_mcos(size_t(std::min(4, n_bb)), 0.0);
            std::vector<double> bb_rcos(size_t(std::min(4, n_bb)), 0.0);
            int bb_first_div = -1;
            double bb_lcos_min = 1.0, bb_sr_min = 1.0;
            size_t router_hit = 0, router_tot = 0;
            int router_first_div = -1;
            for (int l = 0; l < n_bb; ++l) {
                bb_lcos[size_t(l)] = cosine_sim(ref_bb_tr.residuals[size_t(l)], test_bb_tr.residuals[size_t(l)]);
                bb_lcos_sr[size_t(l)] = cosine_sim(ref_bb_tr.residuals[size_t(l)], test_sr_bb_tr.residuals[size_t(l)]);
                bb_lcos_min = std::min(bb_lcos_min, bb_lcos[size_t(l)]);
                bb_sr_min = std::min(bb_sr_min, bb_lcos_sr[size_t(l)]);
                if (bb_first_div < 0 && bb_lcos[size_t(l)] < 0.9999) bb_first_div = l;
                if (l < 4) {
                    bb_mcos[size_t(l)] = cosine_sim(ref_bb_tr.mixer_residuals[size_t(l)], test_bb_tr.mixer_residuals[size_t(l)]);
                    bb_rcos[size_t(l)] = cosine_sim(ref_bb_tr.router_logits[size_t(l)], test_bb_tr.router_logits[size_t(l)]);
                }

                const auto& ra = ref_bb_tr.topk_ids[size_t(l)];
                const auto& ta = test_bb_tr.topk_ids[size_t(l)];
                const int K = g6tts::geom::TOPK;
                const int n_m = static_cast<int>(std::min(ra.size(), ta.size())) / K;
                size_t l_hit = 0, l_tot = size_t(n_m) * K;
                for (int m = 0; m < n_m; ++m) {
                    for (int k = 0; k < K; ++k) {
                        const int id = ra[size_t(m) * K + k];
                        for (int j = 0; j < K; ++j) {
                            if (ta[size_t(m) * K + j] == id) { ++l_hit; break; }
                        }
                    }
                }
                router_hit += l_hit;
                router_tot += l_tot;
                if (router_first_div < 0 && l_hit < l_tot) router_first_div = l;
            }
            const double router_agree = (router_tot > 0) ? (double(router_hit) / double(router_tot)) : 1.0;

            const int n_tw = static_cast<int>(std::min(ref_tw_tr.hiddens.size(), test_tw_tr.hiddens.size()));
            std::vector<double> tw_lcos(size_t(n_tw), 0.0);
            std::vector<double> tw_iso_lcos(size_t(n_tw), 0.0);
            int tw_first_div = -1;
            double tw_lcos_min = 1.0, tw_iso_min = 1.0;
            for (int l = 0; l < n_tw; ++l) {
                tw_lcos[size_t(l)] = cosine_sim(ref_tw_tr.hiddens[size_t(l)], test_tw_tr.hiddens[size_t(l)]);
                tw_iso_lcos[size_t(l)] = cosine_sim(ref_tw_tr.hiddens[size_t(l)], test_iso_tw_tr.hiddens[size_t(l)]);
                tw_lcos_min = std::min(tw_lcos_min, tw_lcos[size_t(l)]);
                tw_iso_min = std::min(tw_iso_min, tw_iso_lcos[size_t(l)]);
                if (tw_first_div < 0 && tw_lcos[size_t(l)] < 0.9999) tw_first_div = l;
            }

            double tw_all_min = 1.0;
            std::string tw_all_worst_name = "none";
            int tw_all_worst_fmt = -1;
            const size_t n_tw_ops = std::min(ref_ops.tw_all.size(), test_ops.tw_all.size());
            for (size_t i = 0; i < n_tw_ops; ++i) {
                const double c = cosine_sim(ref_ops.tw_all[i], test_ops.tw_all[i]);
                if (c < tw_all_min) {
                    tw_all_min = c;
                    tw_all_worst_name = test_ops.tw_all_names[i];
                    tw_all_worst_fmt = test_ops.tw_all_fmts[i];
                }
            }

            const bool ok = (c_hid >= 0.9999 && c_main >= 0.9999 && head_min >= 0.9999);
            std::printf("{\"verify_kernels\":true,\"gate\":\"E3\",\"pass\":%s,\"mode\":\"%s\",\"rows_src\":\"%s\","
                        "\"hidden_cos\":%.8f,\"main_cos\":%.8f,\"heads_cos\":%.8f,\"head_cos_min\":%.8f,\"head_cos\":[",
                        ok ? "true" : "false", g6tts::weight_mode_name(test_mode),
                        opts.verify_rows.empty() ? "synthetic" : "file",
                        c_hid, c_main, c_head, head_min);
            for (int h = 0; h < n_heads; ++h) {
                std::printf("%s%.8f", h ? "," : "", head_cos[h]);
            }
            std::printf("],\"same_router\":{\"hidden_cos\":%.8f,\"main_cos\":%.8f,\"bb_layer_cos_min\":%.8f},"
                        "\"tower_isolated\":{\"layer_cos_min\":%.8f,\"head_cos_min\":%.8f,\"layer_cos\":[",
                        c_hid_sr, c_main_sr, bb_sr_min, tw_iso_min, head_iso_min);
            for (int l = 0; l < n_tw; ++l) std::printf("%s%.8f", l ? "," : "", tw_iso_lcos[size_t(l)]);
            std::printf("],\"head_cos\":[");
            for (int h = 0; h < n_heads; ++h) std::printf("%s%.8f", h ? "," : "", head_iso_cos[h]);
            auto vec_max_abs = [](const std::vector<float>& v) -> double {
                double m = 0.0;
                for (float x : v) m = std::max(m, std::fabs(double(x)));
                return m;
            };
            std::printf("]},\"tower_iso_subops\":{");
            const char* sub_names[2] = {"res4", "res7"};
            const int sub_idx[2] = {8, 11};
            for (int s = 0; s < 2; ++s) {
                const auto& rs = ref_tw_tr.sub[s];
                const auto& ts = test_iso_tw_tr.sub[s];
                std::printf("%s\"%s\":{\"eh_cat\":%.8f,\"eh_proj\":%.8f,\"wqkv\":%.8f,\"q_ready\":%.8f,"
                            "\"attn_out\":%.8f,\"wo_res\":%.8f,\"ffn_act\":%.8f,\"w2_res\":%.8f,\"final_ln\":%.8f,"
                            "\"ffn_max\":%.2f,\"w2_max\":%.2f}",
                            s ? "," : "", sub_names[s],
                            cosine_sim(rs.eh_cat, ts.eh_cat),
                            cosine_sim(rs.eh_proj, ts.eh_proj),
                            cosine_sim(rs.wqkv, ts.wqkv),
                            cosine_sim(rs.q_ready, ts.q_ready),
                            cosine_sim(rs.attn_out, ts.attn_out),
                            cosine_sim(rs.wo_res, ts.wo_res),
                            cosine_sim(rs.ffn_act, ts.ffn_act),
                            cosine_sim(rs.w2_res, ts.w2_res),
                            (sub_idx[s] < n_tw) ? tw_iso_lcos[size_t(sub_idx[s])] : 0.0,
                            vec_max_abs(rs.ffn_act),
                            vec_max_abs(rs.w2_res));
            }
            std::printf("},\"tower_all_op_cos_min\":%.8f,\"tower_all_op_worst\":{\"name\":\"%s\",\"fmt\":%d,\"count\":%zu},"
                        "\"op_cos\":{\"embed_fp8\":%.8f,\"trellis_out\":%.8f,\"trellis_in\":%.8f,\"trellis_dual\":%.8f,"
                        "\"trellis_triple\":%.8f,\"int4\":%.8f,\"fp8_row\":%.8f,\"lm_head_fp8\":%.8f,\"bf16\":%.8f,"
                        "\"gdn0_dual\":%.8f,\"gdn0_out\":%.8f,\"gdn2_out_i4\":%.8f,\"sh0_w2_i4\":%.8f,\"sh1_w2_tr\":%.8f,\"w2_outlier_tr\":%.8f,\"w2_sparse_outlier_tr\":%.8f},"
                        "\"bb_mixer_cos_0_3\":[%.8f,%.8f,%.8f,%.8f],\"bb_router_cos_0_3\":[%.8f,%.8f,%.8f,%.8f],"
                        "\"router_topk_agree\":%.6f,\"router_first_div_layer\":%d,"
                        "\"bb_first_div_layer\":%d,\"bb_layer_cos_min\":%.8f,\"bb_layer_cos\":[",
                        tw_all_min, tw_all_worst_name.c_str(), tw_all_worst_fmt, n_tw_ops,
                        cosine_sim(ref_bb_tr.embed_f32, test_bb_tr.embed_f32),
                        cosine_sim(ref_ops.tr_out, test_ops.tr_out),
                        cosine_sim(ref_ops.tr_in, test_ops.tr_in),
                        cosine_sim(ref_ops.tr_dual, test_ops.tr_dual),
                        cosine_sim(ref_ops.tr_triple, test_ops.tr_triple),
                        cosine_sim(ref_ops.int4_op, test_ops.int4_op),
                        cosine_sim(ref_ops.fp8_row, test_ops.fp8_row),
                        cosine_sim(ref_ops.lm_head_op, test_ops.lm_head_op),
                        cosine_sim(ref_ops.bf16_op, test_ops.bf16_op),
                        cosine_sim(ref_ops.gdn0_dual, test_ops.gdn0_dual),
                        cosine_sim(ref_ops.gdn0_out, test_ops.gdn0_out),
                        cosine_sim(ref_ops.gdn2_out_i4, test_ops.gdn2_out_i4),
                        cosine_sim(ref_ops.sh0_w2_i4, test_ops.sh0_w2_i4),
                        cosine_sim(ref_ops.sh1_w2_tr, test_ops.sh1_w2_tr),
                        cosine_sim(ref_ops.w2_outlier_tr, test_ops.w2_outlier_tr),
                        cosine_sim(ref_ops.w2_sparse_outlier_tr, test_ops.w2_sparse_outlier_tr),
                        bb_mcos.size() > 0 ? bb_mcos[0] : 0.0, bb_mcos.size() > 1 ? bb_mcos[1] : 0.0,
                        bb_mcos.size() > 2 ? bb_mcos[2] : 0.0, bb_mcos.size() > 3 ? bb_mcos[3] : 0.0,
                        bb_rcos.size() > 0 ? bb_rcos[0] : 0.0, bb_rcos.size() > 1 ? bb_rcos[1] : 0.0,
                        bb_rcos.size() > 2 ? bb_rcos[2] : 0.0, bb_rcos.size() > 3 ? bb_rcos[3] : 0.0,
                        router_agree, router_first_div, bb_first_div, bb_lcos_min);
            for (int l = 0; l < n_bb; ++l) {
                std::printf("%s%.8f", l ? "," : "", bb_lcos[size_t(l)]);
            }
            std::printf("],\"tower_first_div_layer\":%d,\"tower_layer_cos_min\":%.8f,\"tower_layer_cos\":[",
                        tw_first_div, tw_lcos_min);
            for (int l = 0; l < n_tw; ++l) {
                std::printf("%s%.8f", l ? "," : "", tw_lcos[size_t(l)]);
            }
            std::printf("]}\n");
            cudaStreamDestroy(stream);
            return ok ? 0 : 1;
        }

        if (opts.verify_graph) {
            auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
            std::stringstream ss(opts.scenarios);
            std::string sc;
            bool all_ok = true;
            while (std::getline(ss, sc, ',')) {
                int M = 1;
                if (sc == "m1") M = 1;
                else if (sc == "m2") M = 2;
                else if (sc == "m4") M = 4;
                else if (sc == "m8") M = 8;
                else continue;

                const auto eager = run_graph_scenario(W, opts, M, false, false, stream);
                const auto graphed = run_graph_scenario(W, opts, M, true, false, stream);
                size_t mismatches = 0;
                const size_t n_codes = std::min(eager.codes.size(), graphed.codes.size());
                for (size_t i = 0; i < n_codes; ++i) {
                    if (eager.codes[i] != graphed.codes[i]) ++mismatches;
                }
                if (eager.codes.size() != graphed.codes.size()) mismatches += 1;

                size_t pcm_mismatches = 0;
                const size_t n_pcm = std::min(eager.pcm.size(), graphed.pcm.size());
                for (size_t i = 0; i < n_pcm; ++i) {
                    if (eager.pcm[i] != graphed.pcm[i]) ++pcm_mismatches;
                }
                if (eager.pcm.size() != graphed.pcm.size()) pcm_mismatches += 1;

                const bool ok = (mismatches == 0 && pcm_mismatches == 0);
                if (!ok) all_ok = false;
                const double speedup = (graphed.ms_per_step > 0.0) ? (eager.ms_per_step / graphed.ms_per_step) : 0.0;
                std::printf("{\"verify_graph\":true,\"scenario\":\"%s\",\"M\":%d,\"frames\":%d,"
                            "\"codec_loaded\":%s,\"exact_match\":%s,\"mismatches\":%zu,\"pcm_mismatches\":%zu,"
                            "\"codes_hash_eager\":\"%016llx\",\"codes_hash_graph\":\"%016llx\","
                            "\"eager_ms_per_step\":%.3f,\"graph_ms_per_step\":%.3f,\"speedup\":%.3f}\n",
                            sc.c_str(), M, opts.gen_frames,
                            graphed.codec_loaded ? "true" : "false",
                            ok ? "true" : "false", mismatches, pcm_mismatches,
                            static_cast<unsigned long long>(eager.codes_hash),
                            static_cast<unsigned long long>(graphed.codes_hash),
                            eager.ms_per_step, graphed.ms_per_step, speedup);
                std::fflush(stdout);
            }
            cudaStreamDestroy(stream);
            return all_ok ? 0 : 1;
        }

        if (opts.run_scenarios) {
            auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
            std::stringstream ss(opts.scenarios);
            std::string sc;
            while (std::getline(ss, sc, ',')) {
                int M = 1;
                if (sc == "m1") M = 1;
                else if (sc == "m2") M = 2;
                else if (sc == "m4") M = 4;
                else if (sc == "m8") M = 8;
                else continue;

                const auto r = run_graph_scenario(W, opts, M, opts.use_graph, opts.profile, stream);
                std::printf("{\"scenario\":\"%s\",\"M\":%d,\"frames\":%d,\"graph\":%s,\"codec_loaded\":%s,"
                            "\"ms_per_step\":%.3f,\"frames_per_s\":%.2f,\"audio_s_per_s\":%.2f,"
                            "\"codes_hash\":\"%016llx\","
                            "\"profile\":{\"embed_ms\":%.2f,\"attn_ms\":%.2f,\"gdn_ms\":%.2f,"
                            "\"moe_ms\":%.2f,\"head_ms\":%.2f,\"tower_shared_ms\":%.2f,"
                            "\"tower_res_ms\":%.2f,\"tower_head_ms\":%.2f,\"sample_ms\":%.2f,\"codec_ms\":%.2f}}\n",
                            sc.c_str(), M, opts.gen_frames,
                            r.graphed ? "true" : "false", r.codec_loaded ? "true" : "false",
                            r.ms_per_step, r.frames_per_s, r.audio_s_per_s,
                            static_cast<unsigned long long>(r.codes_hash),
                            r.bb_prof.embed_ms, r.bb_prof.attn_ms, r.bb_prof.gdn_ms,
                            r.bb_prof.moe_ms, r.bb_prof.head_ms, r.tw_prof.shared_ms,
                            r.tw_prof.res_ms, r.tw_prof.head_ms, r.fr_prof.sample_ms, r.fr_prof.codec_ms);
                std::fflush(stdout);
            }
            cudaStreamDestroy(stream);
            return 0;
        }

        usage();
        cudaStreamDestroy(stream);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "g6tts_bench: fatal: %s\n", e.what());
        return 1;
    }
}
