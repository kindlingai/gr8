// g6embed: Native CUDA TP=1 dense embedding engine for exlr8 multi-group releases (Qwen3-VL-Embedding 2B/8B).
// Speaks PROTOCOL.md v1 over a Unix domain socket (`hello` with `embeddings: true`, `embed`, `embedding`, `finish`).
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "embed_config.h"
#include "embed_proto.h"
#include "embed_text.cuh"
#include "embed_vision.cuh"
#include "embed_weights.h"

namespace {

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct CliOpts {
    std::string model_dir;
    std::string socket_path = "/run/g6/engine.sock";
    std::string model_name_override;
    int max_batched_tokens = 16384;
    int max_model_len = 16384;
    bool verify_checksums = true;
    bool dry_run_load = false;
    g6emb::WeightMode weight_mode = g6emb::WeightMode::Native;
};

void usage() {
    std::fprintf(stderr,
                 "usage: g6embed --model-dir DIR [--socket PATH] [--model-name NAME] "
                 "[--max-batched-tokens N] [--max-model-len N] [--weight-mode native|bf16] "
                 "[--no-hash] [--dry-run-load]\n");
}

bool parse_cli(int argc, char** argv, CliOpts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* f) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "g6embed: %s requires an argument\n", f);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            const char* v = need("--model-dir");
            if (!v) return false;
            o.model_dir = v;
        } else if (a == "--socket") {
            const char* v = need("--socket");
            if (!v) return false;
            o.socket_path = v;
        } else if (a == "--model-name") {
            const char* v = need("--model-name");
            if (!v) return false;
            o.model_name_override = v;
        } else if (a == "--max-batched-tokens") {
            const char* v = need("--max-batched-tokens");
            if (!v) return false;
            o.max_batched_tokens = std::max(1, std::atoi(v));
        } else if (a == "--max-model-len") {
            const char* v = need("--max-model-len");
            if (!v) return false;
            o.max_model_len = std::max(1, std::atoi(v));
        } else if (a == "--weight-mode") {
            const char* v = need("--weight-mode");
            if (!v) return false;
            if (std::strcmp(v, "native") == 0) o.weight_mode = g6emb::WeightMode::Native;
            else if (std::strcmp(v, "bf16") == 0) o.weight_mode = g6emb::WeightMode::Bf16;
            else {
                std::fprintf(stderr, "g6embed: unknown --weight-mode %s (expected native or bf16)\n", v);
                return false;
            }
        } else if (a == "--no-hash") {
            o.verify_checksums = false;
        } else if (a == "--dry-run-load") {
            o.dry_run_load = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "g6embed: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (o.model_dir.empty()) {
        std::fprintf(stderr, "g6embed: --model-dir is required\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    CliOpts opts;
    if (!parse_cli(argc, argv, opts)) {
        usage();
        return 2;
    }
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // no SA_RESTART so accept() unblocks cleanly on SIGINT/SIGTERM
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    std::signal(SIGPIPE, SIG_IGN);

    try {
        g6emb::EmbedWeights weights = g6emb::load_embed_weights(opts.model_dir, opts.verify_checksums, opts.weight_mode);
        weights.cfg.max_model_len = opts.max_model_len;
        if (!opts.model_name_override.empty()) {
            weights.cfg.model_name = opts.model_name_override;
        }
        if (opts.dry_run_load) {
            std::fprintf(stderr, "g6embed: --dry-run-load OK (%s)\n", weights.cfg.model_name.c_str());
            return 0;
        }

        const int lfd = g6emb::listen_unix_socket(opts.socket_path);
        if (lfd < 0) {
            std::fprintf(stderr, "g6embed: failed to bind Unix socket %s\n", opts.socket_path.c_str());
            return 1;
        }
        g6emb::TextRunner text_runner;
        text_runner.init(weights, opts.max_batched_tokens);
        g6emb::VisionRunner vision_runner;
        vision_runner.init(weights);

        cudaStream_t stream = nullptr;
        EMB_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::fprintf(stderr, "g6embed: text workspace %.1f MiB; listening on %s (model=%s, dim=%d, vision=%d)\n",
                     double(text_runner.workspace_bytes()) / (1024.0 * 1024.0),
                     opts.socket_path.c_str(), weights.cfg.model_name.c_str(),
                     weights.cfg.embedding_dim, weights.cfg.has_vision ? 1 : 0);

        g6emb::EmbedMetrics metrics;
        auto runner = [&](const std::vector<g6emb::EmbedRequest>& batch) -> std::vector<g6emb::EmbedOutput> {
            g6emb::PackedVisionFeatures vis = vision_runner.encode_batch(batch, text_runner.ws, &text_runner.prof, stream);
            return text_runner.forward_batch(batch, vis.n_img_tokens > 0 ? &vis : nullptr, stream);
        };

        while (!g_stop) {
            const int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            g6emb::serve_connection(cfd, weights.cfg, opts.max_batched_tokens, metrics, runner);
            ::close(cfd);
        }
        cudaStreamDestroy(stream);
        ::close(lfd);
        ::unlink(opts.socket_path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "g6embed: fatal: %s\n", e.what());
        return 1;
    }
}
