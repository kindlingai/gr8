// g6embed_bench: CUDA microbenchmark, per-op profiler, and native-vs-BF16 kernel parity verifier for g6embed.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "embed_config.h"
#include "embed_proto.h"
#include "embed_text.cuh"
#include "embed_vision.cuh"
#include "embed_weights.h"

namespace {

struct BenchOpts {
    std::string model_dir;
    g6emb::WeightMode weight_mode = g6emb::WeightMode::Native;
    bool verify_checksums = true;
    bool verify_kernels = false;
    bool profile = true;
    int warmup = 3;
    int iters = 10;
    int max_batched_tokens = 16384;
    std::string scenarios = "b1_l64,b16_l256,b64_l256,mm_512,mm_4x512";
};

void usage() {
    std::fprintf(stderr,
                 "usage: g6embed_bench --model-dir DIR [--weight-mode native|bf16] [--no-hash]\n"
                 "                     [--verify-kernels] [--profile|--no-profile]\n"
                 "                     [--warmup N] [--iters N] [--max-batched-tokens N]\n"
                 "                     [--scenarios b1_l64,b16_l256,b64_l256,mm_512,mm_4x512]\n");
}

bool parse_cli(int argc, char** argv, BenchOpts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* f) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "g6embed_bench: %s requires an argument\n", f);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            const char* v = need("--model-dir");
            if (!v) return false;
            o.model_dir = v;
        } else if (a == "--weight-mode") {
            const char* v = need("--weight-mode");
            if (!v) return false;
            if (std::strcmp(v, "native") == 0) o.weight_mode = g6emb::WeightMode::Native;
            else if (std::strcmp(v, "bf16") == 0) o.weight_mode = g6emb::WeightMode::Bf16;
            else return false;
        } else if (a == "--no-hash") {
            o.verify_checksums = false;
        } else if (a == "--verify-kernels") {
            o.verify_kernels = true;
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "--no-profile") {
            o.profile = false;
        } else if (a == "--warmup") {
            const char* v = need("--warmup");
            if (!v) return false;
            o.warmup = std::max(0, std::atoi(v));
        } else if (a == "--iters") {
            const char* v = need("--iters");
            if (!v) return false;
            o.iters = std::max(1, std::atoi(v));
        } else if (a == "--max-batched-tokens") {
            const char* v = need("--max-batched-tokens");
            if (!v) return false;
            o.max_batched_tokens = std::max(64, std::atoi(v));
        } else if (a == "--scenarios") {
            const char* v = need("--scenarios");
            if (!v) return false;
            o.scenarios = v;
        } else if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "g6embed_bench: unknown option %s\n", a.c_str());
            return false;
        }
    }
    return !o.model_dir.empty();
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    }
    return m;
}

std::vector<uint8_t> make_synthetic_rgb(int vit_h, int vit_w, int seed) {
    const int H = vit_h * 16;
    const int W = vit_w * 16;
    std::vector<uint8_t> rgb(size_t(H) * W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t off = (size_t(y) * W + x) * 3;
            rgb[off + 0] = static_cast<uint8_t>((x * 255 / std::max(1, W - 1) + seed * 17) & 255);
            rgb[off + 1] = static_cast<uint8_t>((y * 255 / std::max(1, H - 1) + seed * 31) & 255);
            rgb[off + 2] = static_cast<uint8_t>(((x ^ y) + seed * 53) & 255);
        }
    }
    return rgb;
}

std::vector<g6emb::EmbedRequest> make_text_batch(int B, int L, int vocab_size) {
    std::vector<g6emb::EmbedRequest> batch;
    batch.reserve(size_t(B));
    const int vmax = std::max(1024, std::min(vocab_size - 16, 120000));
    for (int b = 0; b < B; ++b) {
        g6emb::EmbedRequest r;
        r.id = b + 1;
        r.prompt.resize(size_t(L));
        uint32_t s = 0x9e3779b9u ^ static_cast<uint32_t>(b * 2654435761u + L);
        for (int i = 0; i < L; ++i) {
            s = s * 1664525u + 1013904223u;
            r.prompt[size_t(i)] = 100 + static_cast<int>((s >> 8) % static_cast<uint32_t>(vmax - 100));
        }
        batch.push_back(std::move(r));
    }
    return batch;
}

std::vector<g6emb::EmbedRequest> make_mm_batch(int B, int vit_h, int vit_w, int text_pad, int img_tok_id, int vocab_size) {
    std::vector<g6emb::EmbedRequest> batch;
    batch.reserve(size_t(B));
    const int n_vis_toks = (vit_h / 2) * (vit_w / 2);
    const int vmax = std::max(1024, std::min(vocab_size - 16, 120000));
    for (int b = 0; b < B; ++b) {
        g6emb::EmbedRequest r;
        r.id = b + 1;
        const int prefix = 8;
        const int suffix = std::max(4, text_pad);
        r.prompt.reserve(size_t(prefix + n_vis_toks + suffix));
        for (int i = 0; i < prefix; ++i) r.prompt.push_back(1000 + ((b * 13 + i * 7) % (vmax - 1000)));
        g6emb::EmbedImage im;
        im.pos = prefix;
        im.vit_h = vit_h;
        im.vit_w = vit_w;
        im.span = n_vis_toks;
        im.rgb = make_synthetic_rgb(vit_h, vit_w, b + 1);
        r.images.push_back(std::move(im));
        for (int i = 0; i < n_vis_toks; ++i) r.prompt.push_back(img_tok_id);
        for (int i = 0; i < suffix; ++i) r.prompt.push_back(2000 + ((b * 29 + i * 11) % (vmax - 2000)));
        batch.push_back(std::move(r));
    }
    return batch;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

int run_verify_kernels(const BenchOpts& opts, cudaStream_t stream) {
    std::fprintf(stderr, "g6embed_bench: verifying Native vs BF16 reference kernels on %s\n", opts.model_dir.c_str());

    // Run BF16 reference first, then free weights before loading Native so VRAM stays minimal.
    std::vector<g6emb::EmbedRequest> text_batch;
    std::vector<g6emb::EmbedRequest> mm_batch;
    std::vector<g6emb::EmbedOutput> ref_text, ref_mm;
    bool has_vision = false;

    {
        g6emb::EmbedWeights w_ref = g6emb::load_embed_weights(opts.model_dir, opts.verify_checksums, g6emb::WeightMode::Bf16);
        has_vision = w_ref.cfg.has_vision;
        text_batch = make_text_batch(4, 64, w_ref.cfg.vocab_size);
        if (has_vision) {
            mm_batch = make_mm_batch(1, 32, 32, 16, w_ref.cfg.image_token_id, w_ref.cfg.vocab_size);
        }
        g6emb::TextRunner tr;
        tr.init(w_ref, 2048);
        g6emb::VisionRunner vr;
        vr.init(w_ref);
        ref_text = tr.forward_batch(text_batch, nullptr, stream);
        if (has_vision) {
            auto vis = vr.encode_batch(mm_batch, tr.ws, nullptr, stream);
            ref_mm = tr.forward_batch(mm_batch, &vis, stream);
        }
    }

    std::vector<g6emb::EmbedOutput> nat_text, nat_mm;
    {
        g6emb::EmbedWeights w_nat = g6emb::load_embed_weights(opts.model_dir, opts.verify_checksums, g6emb::WeightMode::Native);
        g6emb::TextRunner tr;
        tr.init(w_nat, 2048);
        g6emb::VisionRunner vr;
        vr.init(w_nat);
        nat_text = tr.forward_batch(text_batch, nullptr, stream);
        if (has_vision) {
            auto vis = vr.encode_batch(mm_batch, tr.ws, nullptr, stream);
            nat_mm = tr.forward_batch(mm_batch, &vis, stream);
        }
    }

    bool ok = true;
    for (size_t i = 0; i < ref_text.size(); ++i) {
        const double cos = cosine_sim(ref_text[i].embedding, nat_text[i].embedding);
        const double mad = max_abs_diff(ref_text[i].embedding, nat_text[i].embedding);
        std::printf("{\"verify\":\"text_%zu\",\"cos\":%.8f,\"max_abs_diff\":%.6e}\n", i, cos, mad);
        if (cos < 0.9999) ok = false;
    }
    if (has_vision && !ref_mm.empty()) {
        const double cos = cosine_sim(ref_mm[0].embedding, nat_mm[0].embedding);
        const double mad = max_abs_diff(ref_mm[0].embedding, nat_mm[0].embedding);
        std::printf("{\"verify\":\"mm_512\",\"cos\":%.8f,\"max_abs_diff\":%.6e}\n", cos, mad);
        if (cos < 0.9999) ok = false;
    }
    std::fflush(stdout);
    if (!ok) {
        std::fprintf(stderr, "g6embed_bench: FAIL native vs bf16 parity below 0.9999\n");
        return 1;
    }
    std::fprintf(stderr, "g6embed_bench: PASS native vs bf16 parity >= 0.9999\n");
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
        EMB_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        if (opts.verify_kernels) {
            const int rc = run_verify_kernels(opts, stream);
            cudaStreamDestroy(stream);
            return rc;
        }

        g6emb::EmbedWeights weights = g6emb::load_embed_weights(opts.model_dir, opts.verify_checksums, opts.weight_mode);
        g6emb::TextRunner tr;
        tr.init(weights, opts.max_batched_tokens);
        tr.prof.enabled = opts.profile;
        g6emb::VisionRunner vr;
        vr.init(weights);

        const double w_mib = double(weights.arena_bytes) / (1024.0 * 1024.0);
        const double ws_mib = double(tr.workspace_bytes()) / (1024.0 * 1024.0);

        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        EMB_CK(cudaEventCreate(&ev0));
        EMB_CK(cudaEventCreate(&ev1));

        for (const std::string& sc : split_csv(opts.scenarios)) {
            std::vector<g6emb::EmbedRequest> batch;
            int n_imgs = 0;
            if (sc == "b1_l64") {
                batch = make_text_batch(1, 64, weights.cfg.vocab_size);
            } else if (sc == "b16_l256") {
                batch = make_text_batch(16, 256, weights.cfg.vocab_size);
            } else if (sc == "b64_l256") {
                batch = make_text_batch(64, 256, weights.cfg.vocab_size);
            } else if (sc == "mm_512") {
                if (!weights.cfg.has_vision) continue;
                batch = make_mm_batch(1, 32, 32, 24, weights.cfg.image_token_id, weights.cfg.vocab_size);
                n_imgs = 1;
            } else if (sc == "mm_4x512") {
                if (!weights.cfg.has_vision) continue;
                batch = make_mm_batch(4, 32, 32, 24, weights.cfg.image_token_id, weights.cfg.vocab_size);
                n_imgs = 4;
            } else {
                std::fprintf(stderr, "g6embed_bench: skipping unknown scenario %s\n", sc.c_str());
                continue;
            }

            int total_toks = 0;
            for (const auto& r : batch) total_toks += static_cast<int>(r.prompt.size());
            if (total_toks > opts.max_batched_tokens) {
                std::fprintf(stderr, "g6embed_bench: skipping %s (%d > max_batched_tokens %d)\n",
                             sc.c_str(), total_toks, opts.max_batched_tokens);
                continue;
            }

            auto run_once = [&]() -> std::vector<g6emb::EmbedOutput> {
                auto vis = vr.encode_batch(batch, tr.ws, &tr.prof, stream);
                return tr.forward_batch(batch, vis.n_img_tokens > 0 ? &vis : nullptr, stream);
            };

            // Warmup (profiling disabled)
            tr.prof.enabled = false;
            std::vector<g6emb::EmbedOutput> last_out;
            for (int w = 0; w < opts.warmup; ++w) last_out = run_once();

            // Timed iterations (unprofiled to avoid event sync overhead distorting wall latency)
            std::vector<double> times_ms;
            times_ms.reserve(size_t(opts.iters));
            for (int it = 0; it < opts.iters; ++it) {
                EMB_CK(cudaEventRecord(ev0, stream));
                last_out = run_once();
                EMB_CK(cudaEventRecord(ev1, stream));
                EMB_CK(cudaEventSynchronize(ev1));
                float ms = 0.0f;
                EMB_CK(cudaEventElapsedTime(&ms, ev0, ev1));
                times_ms.push_back(double(ms));
            }
            std::sort(times_ms.begin(), times_ms.end());
            const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / double(times_ms.size());
            const double p50_ms = times_ms[times_ms.size() / 2];
            const double min_ms = times_ms.front();

            // One profiled iteration for per-op breakdown
            tr.prof.reset();
            tr.prof.enabled = opts.profile;
            if (opts.profile) {
                (void)run_once();
                tr.prof.enabled = false;
            }

            // Deterministic embedding digest for quick sanity across iterations
            double emb_sum = 0.0, emb_l2 = 0.0;
            if (!last_out.empty()) {
                for (float v : last_out[0].embedding) {
                    emb_sum += double(v);
                    emb_l2 += double(v) * double(v);
                }
                emb_l2 = std::sqrt(emb_l2);
            }

            const double tok_s = double(total_toks) / (p50_ms * 1e-3);
            const double emb_s = double(batch.size()) / (p50_ms * 1e-3);

            std::printf(
                "{\"scenario\":\"%s\",\"mode\":\"%s\",\"B\":%zu,\"total_tokens\":%d,\"n_images\":%d,"
                "\"mean_ms\":%.3f,\"p50_ms\":%.3f,\"min_ms\":%.3f,\"tok_per_s\":%.1f,\"emb_per_s\":%.1f,"
                "\"vram_weights_mib\":%.1f,\"vram_ws_mib\":%.1f,\"emb0_l2\":%.6f,\"emb0_sum\":%.6f,"
                "\"profile\":{\"embed_ms\":%.3f,\"rmsnorm_ms\":%.3f,\"qkv_ms\":%.3f,\"qknorm_mrope_ms\":%.3f,"
                "\"attn_ms\":%.3f,\"o_proj_ms\":%.3f,\"mlp_gate_up_ms\":%.3f,\"swiglu_ms\":%.3f,"
                "\"mlp_down_ms\":%.3f,\"deepstack_ms\":%.3f,\"vision_ms\":%.3f,\"pool_ms\":%.3f}}\n",
                sc.c_str(), opts.weight_mode == g6emb::WeightMode::Bf16 ? "bf16" : "native",
                batch.size(), total_toks, n_imgs, mean_ms, p50_ms, min_ms, tok_s, emb_s,
                w_mib, ws_mib, emb_l2, emb_sum,
                tr.prof.embed_ms, tr.prof.rmsnorm_ms, tr.prof.qkv_ms, tr.prof.qknorm_mrope_ms,
                tr.prof.attn_ms, tr.prof.o_proj_ms, tr.prof.mlp_gate_up_ms, tr.prof.swiglu_ms,
                tr.prof.mlp_down_ms, tr.prof.deepstack_ms, tr.prof.vision_ms, tr.prof.pool_ms);
            std::fflush(stdout);
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "g6embed_bench: fatal: %s\n", e.what());
        return 1;
    }
}
