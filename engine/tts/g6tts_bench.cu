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
    bool verify_checksums = true;

    std::string teacher_in;
    std::string dump_out;

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
                 "usage: g6tts_bench --model-dir DIR [--manifest NAME] [--weight-mode ref|ref-fp16|native] [--no-hash]\n"
                 "                   [--teacher ROWS.bin --dump OUT.bin]\n"
                 "                   [--verify-kernels] [--scenarios [m1,m4]] [--profile]\n"
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
            if (std::strcmp(v, "ref") == 0) o.weight_mode = g6tts::WeightMode::Ref;
            else if (std::strcmp(v, "ref-fp16") == 0) o.weight_mode = g6tts::WeightMode::RefFp16;
            else if (std::strcmp(v, "native") == 0) o.weight_mode = g6tts::WeightMode::Native;
            else return false;
        } else if (a == "--no-hash") {
            o.verify_checksums = false;
        } else if (a == "--teacher") {
            const char* v = need("--teacher"); if (!v) return false; o.teacher_in = v;
        } else if (a == "--dump") {
            const char* v = need("--dump"); if (!v) return false; o.dump_out = v;
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
            tower.prefill_chunk(bb.d_final_f16,
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
            std::fprintf(stderr, "g6tts_bench: dumped %zu teacher-forced sequences -> %s\n",
                         outs.size(), opts.dump_out.c_str());
            cudaStreamDestroy(stream);
            return 0;
        }

        if (opts.verify_kernels) {
            // Compare Ref (FP8-on-device experts) vs RefFp16 (or Native vs Ref once Wave 3.2 lands)
            const auto rows = make_synthetic_rows(32, 42);
            TeacherSeqOut ref_out, test_out;
            {
                auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums,
                                                 g6tts::WeightMode::Ref);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, 1, 256, 128); tw.init(W, 1, 256, 128);
                ref_out = run_teacher_seq(bb, tw, rows, stream);
            }
            {
                auto W = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums,
                                                 opts.weight_mode);
                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, 1, 256, 128); tw.init(W, 1, 256, 128);
                test_out = run_teacher_seq(bb, tw, rows, stream);
            }
            const double c_hid = cosine_sim(ref_out.final_hidden, test_out.final_hidden);
            const double c_main = cosine_sim(ref_out.main_logits, test_out.main_logits);
            const double c_head = cosine_sim(ref_out.head_logits, test_out.head_logits);
            std::printf("{\"verify_kernels\":true,\"hidden_cos\":%.8f,\"main_cos\":%.8f,\"heads_cos\":%.8f}\n",
                        c_hid, c_main, c_head);
            cudaStreamDestroy(stream);
            return (c_hid >= 0.9999 && c_main >= 0.9999 && c_head >= 0.9999) ? 0 : 1;
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

                g6tts::BackboneRunner bb; g6tts::TowerRunner tw;
                bb.init(W, M, opts.cap, 128);
                tw.init(W, M, opts.cap, 128);
                bb.prof.enabled = opts.profile;

                const int H = W.cfg.hidden_size;
                const int rw = W.cfg.row_width();
                __half* d_hid = nullptr;
                int32_t *d_sl = nullptr, *d_ps = nullptr, *d_fr = nullptr, *d_c0 = nullptr, *d_cd = nullptr;
                uint8_t* d_sem = nullptr;
                TTS_CK(cudaMalloc(&d_hid, size_t(M) * H * sizeof(__half)));
                TTS_CK(cudaMalloc(&d_sl, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_ps, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_fr, size_t(M) * rw * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_c0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_cd, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMalloc(&d_sem, size_t(M)));
                TTS_CK(cudaMemset(d_c0, 0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMemset(d_cd, 0, size_t(M) * sizeof(int32_t)));
                TTS_CK(cudaMemset(d_sem, 1, size_t(M)));

                std::vector<int32_t> h_slots(static_cast<size_t>(M)), h_pos(static_cast<size_t>(M)), h_rows(static_cast<size_t>(M) * rw, 0);
                for (int m = 0; m < M; ++m) {
                    auto pr = make_synthetic_rows(opts.prompt_len, uint32_t(100 + m));
                    bb.forward(pr.data(), opts.prompt_len, true, m, 0, nullptr, nullptr, false, false, stream);
                    tw.prefill_chunk(bb.d_final_f16, pr.data(), pr.data() + rw, m, 0, opts.prompt_len - 1, nullptr, stream);
                    TTS_CK(cudaMemcpyAsync(d_hid + size_t(m) * H, bb.d_final_f16 + size_t(opts.prompt_len - 1) * H,
                                           size_t(H) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
                    h_slots[size_t(m)] = m;
                    h_pos[size_t(m)] = opts.prompt_len - 1;
                    std::memcpy(h_rows.data() + size_t(m) * rw, pr.data() + size_t(opts.prompt_len - 1) * rw,
                                size_t(rw) * sizeof(int32_t));
                }
                TTS_CK(cudaStreamSynchronize(stream));
                bb.prof.reset();

                cudaEvent_t ev0 = nullptr, ev1 = nullptr;
                TTS_CK(cudaEventCreate(&ev0));
                TTS_CK(cudaEventCreate(&ev1));
                TTS_CK(cudaEventRecord(ev0, stream));
                for (int step = 0; step < opts.gen_frames; ++step) {
                    TTS_CK(cudaMemcpyAsync(d_sl, h_slots.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    TTS_CK(cudaMemcpyAsync(d_ps, h_pos.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    TTS_CK(cudaMemcpyAsync(d_fr, h_rows.data(), size_t(M) * rw * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    g6tts::gemm_f16(bb.blas, d_hid, W.lm_head_cand, bb.d_main_logits, M, W.cfg.main_candidates(), H, true, 0.0f);
                    __half* ch = tw.run_shared(d_hid, d_c0, d_sem, d_fr, d_sl, d_ps, M, stream);
                    for (int i = 0; i < W.cfg.tower_res; ++i) {
                        ch = tw.run_residual_step(i, ch, d_cd, d_sem, d_sl, d_ps, M, g6tts::geom::RES_CB_SIZE, true, stream);
                    }
                    for (int m = 0; m < M; ++m) ++h_pos[size_t(m)];
                    bb.forward(h_rows.data(), M, false, 0, 0, h_slots.data(), h_pos.data(), false, false, stream);
                    TTS_CK(cudaMemcpyAsync(d_hid, bb.d_final_f16, size_t(M) * H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
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
