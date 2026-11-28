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
    bool run_scenarios = false;
    bool profile = false;
    std::string scenarios = "m1,m4";
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
                 "                   [--verify-kernels [--rows ROWS.bin]] [--scenarios [m1,m4]] [--profile]\n"
                 "                   [--prompt-len N] [--frames N] [--cap N]\n"
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
        } else if (a == "--scenarios") {
            o.run_scenarios = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') o.scenarios = argv[++i];
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

void write_layer_trace(const std::string& path, const g6tts::TtsConfig& c, int T,
                       const g6tts::BackboneLayerTrace& bb_tr, const g6tts::TowerLayerTrace& tw_tr) {
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
            // Gate E3: compare Native (default) vs exact RefF32 on the exact same release across backbone, all 9 heads,
            // controlled router/tower probes, and standalone per-format + 78-matrix tower linear unit probes.
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
                std::vector<float> tr_out, tr_in, tr_dual, tr_triple;
                std::vector<float> int4_op, fp8_row, lm_head_op, bf16_op;
                std::vector<float> gdn0_dual, gdn0_out, gdn2_out_i4, sh0_w2_i4, sh1_w2_tr;
                std::vector<float> w2_outlier_tr, w2_sparse_outlier_tr;
                std::vector<std::vector<float>> tw_all;
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
                g6tts::bf16 *d_x = nullptr, *d_y0 = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
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

                tw.proj_bf16(d_x, tw0.wqkv, tw0.wqkv_nat, d_y0, OP_M, 9216, H, stream);
                out.tr_out = dl_bf16(d_y0, size_t(OP_M) * 9216);

                tw.proj_f32(d_x, tw0.eh_proj, tw0.eh_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.tr_in = dl_f32(d_yf, size_t(OP_M) * H);

                if (W.mode == g6tts::WeightMode::Native && tw0.w1_nat.valid() && tw0.w3_nat.valid()) {
                    g6tts::tts_native_linear_dual_bf16(d_x, tw0.w1_nat, tw0.w3_nat, d_y0, d_y1, OP_M, tw.dense_ws, tw.blas, stream);
                } else {
                    tw.proj_bf16(d_x, tw0.w1, tw0.w1_nat, d_y0, OP_M, 6144, H, stream);
                    tw.proj_bf16(d_x, tw0.w3, tw0.w3_nat, d_y1, OP_M, 6144, H, stream);
                }
                out.tr_dual = dl_bf16(d_y0, size_t(OP_M) * 6144);
                auto d1 = dl_bf16(d_y1, size_t(OP_M) * 6144);
                out.tr_dual.insert(out.tr_dual.end(), d1.begin(), d1.end());

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

                bb.proj_f32(d_x, att3.o_proj, att3.o_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.int4_op = dl_f32(d_yf, size_t(OP_M) * H);

                tw.proj_f32(d_x, W.heads[0], W.heads_nat[0], d_yf, OP_M, g6tts::geom::RES_CB_SIZE, H, false, stream);
                out.fp8_row = dl_f32(d_yf, size_t(OP_M) * g6tts::geom::RES_CB_SIZE);

                bb.proj_f32(d_x, W.lm_head_cand, W.lm_head_nat, d_yf, OP_M, n_cand, H, false, stream);
                out.lm_head_op = dl_f32(d_yf, size_t(OP_M) * n_cand);

                bb.proj_f32(d_x, gdn0.router_gate, gdn0.router_gate_nat, d_yf, OP_M, W.cfg.num_experts, H, false, stream);
                out.bf16_op = dl_f32(d_yf, size_t(OP_M) * W.cfg.num_experts);

                if (W.mode == g6tts::WeightMode::Native && gdn0.in_proj_qkv_nat.valid() && gdn0.in_proj_z_nat.valid()) {
                    g6tts::tts_native_linear_dual_bf16(d_x, gdn0.in_proj_qkv_nat, gdn0.in_proj_z_nat, d_y0, d_y1, OP_M, bb.dense_ws, bb.blas, stream);
                } else {
                    bb.proj_bf16(d_x, gdn0.in_proj_qkv, gdn0.in_proj_qkv_nat, d_y0, OP_M, 8192, H, stream);
                    bb.proj_bf16(d_x, gdn0.in_proj_z, gdn0.in_proj_z_nat, d_y1, OP_M, 4096, H, stream);
                }
                out.gdn0_dual = dl_bf16(d_y0, size_t(OP_M) * 8192);
                auto gz = dl_bf16(d_y1, size_t(OP_M) * 4096);
                out.gdn0_dual.insert(out.gdn0_dual.end(), gz.begin(), gz.end());

                bb.proj_f32(d_x, gdn0.out_proj, gdn0.out_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.gdn0_out = dl_f32(d_yf, size_t(OP_M) * H);

                bb.proj_f32(d_x, gdn2.out_proj, gdn2.out_proj_nat, d_yf, OP_M, H, 2 * H, false, stream);
                out.gdn2_out_i4 = dl_f32(d_yf, size_t(OP_M) * H);

                tw.proj_f32(d_x, sh0.w2, sh0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.sh0_w2_i4 = dl_f32(d_yf, size_t(OP_M) * H);

                tw.proj_f32(d_x, sh1.w2, sh1.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.sh1_w2_tr = dl_f32(d_yf, size_t(OP_M) * H);

                TTS_CK(cudaMemcpyAsync(d_x, h_x_outlier_bf16.data(), h_x_outlier_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                tw.proj_f32(d_x, tw0.w2, tw0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.w2_outlier_tr = dl_f32(d_yf, size_t(OP_M) * H);

                TTS_CK(cudaMemcpyAsync(d_x, h_x_sparse_bf16.data(), h_x_sparse_bf16.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, stream));
                tw.proj_f32(d_x, tw0.w2, tw0.w2_nat, d_yf, OP_M, H, W.cfg.tower_intermediate, false, stream);
                out.w2_sparse_outlier_tr = dl_f32(d_yf, size_t(OP_M) * H);

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
                for (int i = 0; i < W.cfg.tower_shared; ++i) sweep_layer(W.shared_layers[size_t(i)], "sh" + std::to_string(i));
                for (int i = 0; i < W.cfg.tower_res; ++i) sweep_layer(W.res_layers[size_t(i)], "res" + std::to_string(i));

                cudaFree(d_x); cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_yf);
                return out;
            };

            TeacherSeqOut ref_out, test_out, test_same_router_out;
            OpProbeSet ref_ops, test_ops;
            g6tts::BackboneLayerTrace ref_bb_tr, test_bb_tr, test_sr_bb_tr;
            g6tts::TowerLayerTrace ref_tw_tr, test_tw_tr, test_iso_tw_tr;
            std::vector<float> test_iso_heads;
            {
                auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums,
                                                 g6tts::WeightMode::RefF32);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, 1, 256, 128); tw.init(W, 1, 256, 128);
                ref_ops = run_op_probes(W, bb, tw);
                bb.trace.enabled = true;
                tw.trace.enabled = true;
                tw.trace.sub_kv[0] = 8;
                tw.trace.sub_kv[1] = 11;
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

                bb.trace.enabled = true;
                bb.trace.replay_router = true;
                bb.trace.forced_topk_ids = ref_bb_tr.topk_ids;
                bb.trace.forced_topk_w = ref_bb_tr.topk_w;
                tw.trace.enabled = false;
                test_same_router_out = run_teacher_seq(bb, tw, rows, stream);
                test_sr_bb_tr = std::move(bb.trace);

                const int T = ref_out.T;
                const int H = W.cfg.hidden_size;
                const int rw = W.cfg.row_width();
                std::vector<uint16_t> h_ref_bf16(size_t(T) * H);
                for (size_t i = 0; i < h_ref_bf16.size(); ++i) h_ref_bf16[i] = g6tts::host_f32_to_bf16(ref_out.final_hidden[i]);
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
            double head_cos[g6tts::geom::TOW_RES], head_iso_cos[g6tts::geom::TOW_RES];
            double head_min = 1.0, head_iso_min = 1.0;
            for (int h = 0; h < n_heads; ++h) {
                std::vector<float> ra, ta, ia;
                ra.reserve(size_t(n_rows) * hv); ta.reserve(size_t(n_rows) * hv); ia.reserve(size_t(n_rows) * hv);
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

            const int n_bb = static_cast<int>(std::min(ref_bb_tr.residuals.size(), test_bb_tr.residuals.size()));
            std::vector<double> bb_lcos(size_t(n_bb), 0.0), bb_lcos_sr(size_t(n_bb), 0.0);
            int bb_first_div = -1, router_first_div = -1;
            double bb_lcos_min = 1.0, bb_sr_min = 1.0;
            size_t router_hit = 0, router_tot = 0;
            for (int l = 0; l < n_bb; ++l) {
                bb_lcos[size_t(l)] = cosine_sim(ref_bb_tr.residuals[size_t(l)], test_bb_tr.residuals[size_t(l)]);
                bb_lcos_sr[size_t(l)] = cosine_sim(ref_bb_tr.residuals[size_t(l)], test_sr_bb_tr.residuals[size_t(l)]);
                bb_lcos_min = std::min(bb_lcos_min, bb_lcos[size_t(l)]);
                bb_sr_min = std::min(bb_sr_min, bb_lcos_sr[size_t(l)]);
                if (bb_first_div < 0 && bb_lcos[size_t(l)] < 0.9999) bb_first_div = l;
                const auto& ra = ref_bb_tr.topk_ids[size_t(l)];
                const auto& ta = test_bb_tr.topk_ids[size_t(l)];
                const int K = g6tts::geom::TOPK;
                const int n_m = static_cast<int>(std::min(ra.size(), ta.size())) / K;
                size_t l_hit = 0, l_tot = size_t(n_m) * K;
                for (int m = 0; m < n_m; ++m) {
                    for (int k = 0; k < K; ++k) {
                        const int id = ra[size_t(m) * K + k];
                        for (int j = 0; j < K; ++j) if (ta[size_t(m) * K + j] == id) { ++l_hit; break; }
                    }
                }
                router_hit += l_hit; router_tot += l_tot;
                if (router_first_div < 0 && l_hit < l_tot) router_first_div = l;
            }
            const double router_agree = (router_tot > 0) ? (double(router_hit) / double(router_tot)) : 1.0;

            const int n_tw = static_cast<int>(std::min(ref_tw_tr.hiddens.size(), test_tw_tr.hiddens.size()));
            std::vector<double> tw_lcos(size_t(n_tw), 0.0), tw_iso_lcos(size_t(n_tw), 0.0);
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
                if (c < tw_all_min) { tw_all_min = c; tw_all_worst_name = test_ops.tw_all_names[i]; tw_all_worst_fmt = test_ops.tw_all_fmts[i]; }
            }

            const bool ok = (c_hid >= 0.9999 && c_main >= 0.9999 && head_min >= 0.9999);
            std::printf("{\"verify_kernels\":true,\"gate\":\"E3\",\"pass\":%s,\"mode\":\"%s\",\"rows_src\":\"%s\","
                        "\"hidden_cos\":%.8f,\"main_cos\":%.8f,\"heads_cos\":%.8f,\"head_cos_min\":%.8f,\"head_cos\":[",
                        ok ? "true" : "false", g6tts::weight_mode_name(test_mode),
                        opts.verify_rows.empty() ? "synthetic" : "file",
                        c_hid, c_main, c_head, head_min);
            for (int h = 0; h < n_heads; ++h) std::printf("%s%.8f", h ? "," : "", head_cos[h]);
            std::printf("],\"same_router\":{\"hidden_cos\":%.8f,\"main_cos\":%.8f,\"bb_layer_cos_min\":%.8f},"
                        "\"tower_isolated\":{\"layer_cos_min\":%.8f,\"head_cos_min\":%.8f,\"layer_cos\":[",
                        c_hid_sr, c_main_sr, bb_sr_min, tw_iso_min, head_iso_min);
            for (int l = 0; l < n_tw; ++l) std::printf("%s%.8f", l ? "," : "", tw_iso_lcos[size_t(l)]);
            std::printf("],\"head_cos\":[");
            for (int h = 0; h < n_heads; ++h) std::printf("%s%.8f", h ? "," : "", head_iso_cos[h]);
            std::printf("]},\"tower_all_op_cos_min\":%.8f,\"tower_all_op_worst\":{\"name\":\"%s\",\"fmt\":%d,\"count\":%zu},"
                        "\"op_cos\":{\"embed_fp8\":%.8f,\"trellis_out\":%.8f,\"trellis_in\":%.8f,\"trellis_dual\":%.8f,"
                        "\"trellis_triple\":%.8f,\"int4\":%.8f,\"fp8_row\":%.8f,\"lm_head_fp8\":%.8f,\"bf16\":%.8f,"
                        "\"gdn0_dual\":%.8f,\"gdn0_out\":%.8f,\"gdn2_out_i4\":%.8f,\"sh0_w2_i4\":%.8f,\"sh1_w2_tr\":%.8f,"
                        "\"w2_outlier_tr\":%.8f,\"w2_sparse_outlier_tr\":%.8f},"
                        "\"router_topk_agree\":%.6f,\"router_first_div_layer\":%d,"
                        "\"bb_first_div_layer\":%d,\"bb_layer_cos_min\":%.8f,\"tower_first_div_layer\":%d,\"tower_layer_cos_min\":%.8f}\n",
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
                        router_agree, router_first_div, bb_first_div, bb_lcos_min, tw_first_div, tw_lcos_min);
            cudaStreamDestroy(stream);
            return ok ? 0 : 1;
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

                const int cap_eff = std::max(opts.cap, opts.prompt_len + opts.gen_frames + 16);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, M, cap_eff, std::max(128, opts.prompt_len));
                tw.init(W, M, cap_eff, std::max(128, opts.prompt_len));
                bb.prof.enabled = opts.profile;

                const int H = W.cfg.hidden_size;
                const int rw = W.cfg.row_width();
                g6tts::bf16* d_hid = nullptr;
                int32_t *d_sl = nullptr, *d_ps = nullptr, *d_fr = nullptr, *d_c0 = nullptr, *d_cd = nullptr;
                uint8_t* d_sem = nullptr;
                TTS_CK(cudaMalloc(&d_hid, size_t(M) * H * sizeof(g6tts::bf16)));
                TTS_CK(cudaMalloc(&d_sl, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_ps, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_fr, size_t(M) * rw * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_c0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_cd, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_sem, size_t(M)));
                TTS_CK(cudaMemset(d_c0, 0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMemset(d_cd, 0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMemset(d_sem, 1, size_t(M)));

                std::vector<int32_t> h_slots(static_cast<size_t>(M)), h_rows(static_cast<size_t>(M) * rw, 0);
                // Keep per-slot prompt buffers alive until stream sync completes.
                std::vector<std::vector<int32_t>> prompts(static_cast<size_t>(M));
                for (int m = 0; m < M; ++m) {
                    prompts[size_t(m)] = make_synthetic_rows(opts.prompt_len, uint32_t(100 + m));
                    const auto& pr = prompts[size_t(m)];
                    bb.forward(pr.data(), opts.prompt_len, true, m, 0, nullptr, nullptr, false, false, stream);
                    tw.prefill_chunk(bb.d_final_bf16, pr.data(), pr.data() + rw, m, 0, opts.prompt_len - 1, nullptr, stream);
                    TTS_CK(cudaMemcpyAsync(d_hid + size_t(m) * H, bb.d_final_bf16 + size_t(opts.prompt_len - 1) * H,
                                           size_t(H) * sizeof(g6tts::bf16), cudaMemcpyDeviceToDevice, stream));
                    h_slots[size_t(m)] = m;
                    std::memcpy(h_rows.data() + size_t(m) * rw, pr.data() + size_t(opts.prompt_len - 1) * rw,
                                size_t(rw) * sizeof(int32_t));
                }
                TTS_CK(cudaStreamSynchronize(stream));
                prompts.clear();
                bb.prof.reset();

                // Pre-fill immutable per-step position table (no host buffer mutated while async H2D DMAs are in flight).
                std::vector<int32_t> h_pos_steps(size_t(opts.gen_frames + 1) * M);
                for (int step = 0; step <= opts.gen_frames; ++step) {
                    for (int m = 0; m < M; ++m) {
                        h_pos_steps[size_t(step) * M + m] = (opts.prompt_len - 1) + step;
                    }
                }
                TTS_CK(cudaMemcpyAsync(d_sl, h_slots.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                TTS_CK(cudaMemcpyAsync(d_fr, h_rows.data(), size_t(M) * rw * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                TTS_CK(cudaStreamSynchronize(stream));

                cudaEvent_t ev0 = nullptr, ev1 = nullptr;
                TTS_CK(cudaEventCreate(&ev0));
                TTS_CK(cudaEventCreate(&ev1));
                TTS_CK(cudaEventRecord(ev0, stream));
                for (int step = 0; step < opts.gen_frames; ++step) {
                    const int32_t* pos_tw = h_pos_steps.data() + size_t(step) * M;
                    const int32_t* pos_bb = h_pos_steps.data() + size_t(step + 1) * M;
                    TTS_CK(cudaMemcpyAsync(d_ps, pos_tw, size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    bb.compute_main_logits(d_hid, M, stream);
                    g6tts::bf16* ch = tw.run_shared(d_hid, d_c0, d_sem, d_fr, d_sl, d_ps, M, stream);
                    for (int i = 0; i < W.cfg.tower_res; ++i) {
                        ch = tw.run_residual_step(i, ch, d_cd, d_sem, d_sl, d_ps, M, g6tts::geom::RES_CB_SIZE, true, stream);
                    }
                    bb.forward(h_rows.data(), M, false, 0, 0, h_slots.data(), pos_bb, false, false, stream);
                    TTS_CK(cudaMemcpyAsync(d_hid, bb.d_final_bf16, size_t(M) * H * sizeof(g6tts::bf16), cudaMemcpyDeviceToDevice, stream));
                }
                TTS_CK(cudaEventRecord(ev1, stream));
                TTS_CK(cudaEventSynchronize(ev1));
                float total_ms = 0.0f;
                TTS_CK(cudaEventElapsedTime(&total_ms, ev0, ev1));
                const double ms_per_frame = double(total_ms) / double(opts.gen_frames);
                const double frames_per_s = (double(M) * double(opts.gen_frames)) / (double(total_ms) * 1e-3);
                const double audio_s_per_s = frames_per_s / (double(g6tts::geom::SAMPLE_RATE) / double(g6tts::geom::FRAME_SAMPLES));
                std::printf("{\"scenario\":\"%s\",\"M\":%d,\"frames\":%d,\"ms_per_step\":%.3f,"
                            "\"frames_per_s\":%.2f,\"audio_s_per_s\":%.2f,"
                            "\"profile\":{\"embed_ms\":%.2f,\"attn_ms\":%.2f,\"gdn_ms\":%.2f,\"moe_ms\":%.2f,\"head_ms\":%.2f}}\n",
                            sc.c_str(), M, opts.gen_frames, ms_per_frame, frames_per_s, audio_s_per_s,
                            bb.prof.embed_ms, bb.prof.attn_ms, bb.prof.gdn_ms, bb.prof.moe_ms, bb.prof.head_ms);
                std::fflush(stdout);

                cudaEventDestroy(ev0); cudaEventDestroy(ev1);
                cudaFree(d_hid); cudaFree(d_sl); cudaFree(d_ps); cudaFree(d_fr);
                cudaFree(d_c0); cudaFree(d_cd); cudaFree(d_sem);
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
