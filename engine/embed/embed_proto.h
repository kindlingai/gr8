// PROTOCOL.md v1 Unix domain socket server and token-budgeted dynamic batcher for g6embed.
// Pure C++17 host header (no CUDA dependency), speaking u32 big-endian length + UTF-8 JSON frames:
//   Engine -> Front end: hello (embeddings: true, embedding_dim, matryoshka, vision), embedding, finish, metrics
//   Front end -> Engine: embed (prompt, optional dimensions, normalize, optional images), cancel, metrics
#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "embed_config.h"
#include "../runtime/json.h"

namespace g6emb {

constexpr uint32_t MAX_PROTO_FRAME = 256u << 20;

struct EmbedRequest {
    std::string id;
    std::vector<int> prompt;
    int dimensions = 0; // 0 = native embedding_dim
    bool normalize = true;
    std::vector<EmbedImage> images;
};

struct EmbedOutput {
    std::string id;
    int prompt_tokens = 0;
    std::vector<float> embedding;
    std::string error; // non-empty -> emit finish reason="error"
    double prefill_s = 0.0;
};

struct EmbedMetrics {
    std::atomic<uint64_t> req_stop{0};
    std::atomic<uint64_t> req_abort{0};
    std::atomic<uint64_t> req_error{0};
    std::atomic<uint64_t> prompt_tokens{0};
    std::atomic<uint64_t> running{0};
    std::atomic<uint64_t> waiting{0};
};

// RFC 4648 standard base64 decoder for PROTOCOL.md `images[].rgb`.
inline bool decode_base64(const std::string& in, std::vector<uint8_t>& out) {
    static constexpr int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    if (in.size() % 4 != 0) return false;
    out.clear();
    out.reserve((in.size() / 4) * 3);
    for (size_t i = 0; i < in.size(); i += 4) {
        const int8_t a = T[uint8_t(in[i])];
        const int8_t b = T[uint8_t(in[i + 1])];
        const int8_t c = T[uint8_t(in[i + 2])];
        const int8_t d = T[uint8_t(in[i + 3])];
        if (a < 0 || b < 0 || c == -1 || d == -1) return false;
        if (c == -2 && (d != -2 || i + 4 != in.size())) return false;
        if (d == -2 && i + 4 != in.size()) return false;
        const uint32_t v = (uint32_t(a) << 18) | (uint32_t(b) << 12) |
                           (uint32_t(c == -2 ? 0 : c) << 6) | uint32_t(d == -2 ? 0 : d);
        out.push_back(uint8_t((v >> 16) & 0xffu));
        if (c != -2) out.push_back(uint8_t((v >> 8) & 0xffu));
        if (d != -2) out.push_back(uint8_t(v & 0xffu));
    }
    return true;
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(c));
                    out += buf;
                } else {
                    out.push_back(char(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

inline bool write_all_fd(int fd, const uint8_t* p, size_t n) {
    size_t done = 0;
    while (done < n) {
        const ssize_t w = ::write(fd, p + done, n - done);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        done += size_t(w);
    }
    return true;
}

inline bool read_exact_fd(int fd, uint8_t* p, size_t n, bool* clean_eof = nullptr) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::read(fd, p + done, n - done);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) {
            if (clean_eof && done == 0) *clean_eof = true;
            return false;
        }
        done += size_t(r);
    }
    return true;
}

inline bool send_frame(int fd, const std::string& json_body) {
    if (json_body.size() > MAX_PROTO_FRAME) return false;
    const uint32_t len_be = htonl(static_cast<uint32_t>(json_body.size()));
    uint8_t hdr[4];
    std::memcpy(hdr, &len_be, 4);
    return write_all_fd(fd, hdr, 4) &&
           write_all_fd(fd, reinterpret_cast<const uint8_t*>(json_body.data()), json_body.size());
}

inline bool recv_frame(int fd, std::string& out_body, bool* clean_eof = nullptr) {
    if (clean_eof) *clean_eof = false;
    uint8_t hdr[4];
    if (!read_exact_fd(fd, hdr, 4, clean_eof)) return false;
    uint32_t len_be = 0;
    std::memcpy(&len_be, hdr, 4);
    const uint32_t len = ntohl(len_be);
    if (len == 0 || len > MAX_PROTO_FRAME) return false;
    out_body.resize(len);
    return read_exact_fd(fd, reinterpret_cast<uint8_t*>(&out_body[0]), len, nullptr);
}

inline std::string make_hello_json(const EmbedConfig& cfg) {
    std::ostringstream os;
    os << "{\"type\":\"hello\",\"version\":1,\"model\":" << json_escape(cfg.model_name)
       << ",\"max_model_len\":" << cfg.max_model_len
       << ",\"vocab_size\":" << cfg.vocab_size
       << ",\"eos_ids\":[" << cfg.eos_token_id << "]"
       << ",\"max_logprobs\":0"
       << ",\"embeddings\":true"
       << ",\"embedding_dim\":" << cfg.embedding_dim
       << ",\"matryoshka\":" << (cfg.matryoshka ? "true" : "false");
    if (cfg.has_vision) os << ",\"vision\":true";
    os << "}";
    return os.str();
}

inline std::string make_embedding_json(const std::string& id, const std::vector<float>& emb, int prompt_tokens) {
    std::string out;
    out.reserve(96 + emb.size() * 14);
    out += "{\"type\":\"embedding\",\"id\":";
    out += json_escape(id);
    out += ",\"embedding\":[";
    char num[32];
    for (size_t i = 0; i < emb.size(); ++i) {
        if (i > 0) out.push_back(',');
        const float v = std::isfinite(emb[i]) ? emb[i] : 0.0f;
        std::snprintf(num, sizeof(num), "%.9g", double(v));
        out += num;
    }
    out += "],\"prompt_tokens\":";
    out += std::to_string(prompt_tokens);
    out += "}";
    return out;
}

inline std::string make_finish_json(const std::string& id, const char* reason, int prompt_tokens,
                                    const std::string& message = "", double prefill_s = -1.0) {
    std::ostringstream os;
    os << "{\"type\":\"finish\",\"id\":" << json_escape(id)
       << ",\"reason\":" << json_escape(reason)
       << ",\"stop_id\":null"
       << ",\"prompt_tokens\":" << prompt_tokens
       << ",\"cached_tokens\":0"
       << ",\"completion_tokens\":0";
    if (!message.empty()) os << ",\"message\":" << json_escape(message);
    if (prefill_s >= 0.0) {
        char tb[32];
        std::snprintf(tb, sizeof(tb), "%.6f", prefill_s);
        os << ",\"timing\":{\"prefill_s\":" << tb << ",\"decode_s\":0}";
    }
    os << "}";
    return os.str();
}

inline std::string make_metrics_json(uint64_t seq, const EmbedMetrics& m) {
    std::ostringstream os;
    os << "{\"type\":\"metrics\",\"seq\":" << seq
       << ",\"requests\":{\"running\":" << m.running.load()
       << ",\"waiting\":" << m.waiting.load()
       << ",\"finished\":{\"stop\":" << m.req_stop.load()
       << ",\"length\":0,\"abort\":" << m.req_abort.load()
       << ",\"error\":" << m.req_error.load() << "}}"
       << ",\"tokens\":{\"prompt\":" << m.prompt_tokens.load() << ",\"generation\":0}}";
    return os.str();
}

// Parses and validates one `embed` JSON frame against `cfg`. Returns empty string on success, or error message.
inline std::string parse_embed_request(const g6v::Json& obj, const EmbedConfig& cfg, EmbedRequest& req) {
    req = EmbedRequest{};
    const auto* id_v = obj.find("id");
    if (!id_v || id_v->t != g6v::Json::Str || id_v->s.empty()) return "embed: missing string id";
    req.id = id_v->s;

    const auto* pr_v = obj.find("prompt");
    if (!pr_v || pr_v->t != g6v::Json::Arr || pr_v->a.empty()) return "embed: prompt must be a non-empty array";
    if (static_cast<int>(pr_v->a.size()) > cfg.max_model_len) return "prompt too long";
    req.prompt.reserve(pr_v->a.size());
    for (const auto& t : pr_v->a) {
        if (t.t != g6v::Json::Num || !t.exact_integer) return "embed: prompt token id must be integer";
        const int64_t tok = t.i64();
        if (tok < 0 || tok >= cfg.vocab_size) return "embed: token id out of vocabulary range";
        req.prompt.push_back(static_cast<int>(tok));
    }

    if (const auto* dim_v = obj.find("dimensions"); dim_v && dim_v->t != g6v::Json::Null) {
        if (dim_v->t != g6v::Json::Num || !dim_v->exact_integer) return "embed: dimensions must be an integer";
        const int64_t d = dim_v->i64();
        if (d < 1 || d > cfg.embedding_dim) return "embed: dimensions out of range";
        if (!cfg.matryoshka && d != cfg.embedding_dim) return "embed: model does not support matryoshka truncation";
        req.dimensions = static_cast<int>(d);
    }

    if (const auto* norm_v = obj.find("normalize"); norm_v && norm_v->t != g6v::Json::Null) {
        if (norm_v->t != g6v::Json::Bool) return "embed: normalize must be boolean";
        req.normalize = norm_v->b;
    }

    if (const auto* im_v = obj.find("images"); im_v && im_v->t != g6v::Json::Null) {
        if (im_v->t != g6v::Json::Arr) return "embed: images must be an array";
        if (!im_v->a.empty() && !cfg.has_vision) return "embed: engine has no vision tower loaded";
        int prev_end = 0;
        for (const auto& elem : im_v->a) {
            if (elem.t != g6v::Json::Obj) return "embed: image entry must be an object";
            const auto* pos_v = elem.find("pos");
            const auto* vh_v = elem.find("vit_h");
            const auto* vw_v = elem.find("vit_w");
            const auto* rgb_v = elem.find("rgb");
            if (!pos_v || !vh_v || !vw_v || !rgb_v ||
                pos_v->t != g6v::Json::Num || !pos_v->exact_integer ||
                vh_v->t != g6v::Json::Num || !vh_v->exact_integer ||
                vw_v->t != g6v::Json::Num || !vw_v->exact_integer ||
                rgb_v->t != g6v::Json::Str) {
                return "embed: malformed image object";
            }
            EmbedImage im;
            im.pos = static_cast<int>(pos_v->i64());
            im.vit_h = static_cast<int>(vh_v->i64());
            im.vit_w = static_cast<int>(vw_v->i64());
            if (const auto* k_v = elem.find("key"); k_v && k_v->t == g6v::Json::Str) im.key = k_v->s;
            im.span = EmbedConfig::qwen3vl_span_of(im.vit_h, im.vit_w);
            if (im.span <= 0) return "embed: invalid vision grid";
            if (im.pos < prev_end || im.pos + im.span > static_cast<int>(req.prompt.size())) {
                return "embed: image span out of bounds or overlapping";
            }
            if (!decode_base64(rgb_v->s, im.rgb)) return "embed: invalid image base64";
            const size_t want_bytes = size_t(im.vit_h) * cfg.vit_patch_size *
                                      size_t(im.vit_w) * cfg.vit_patch_size * 3;
            if (im.rgb.size() != want_bytes) return "embed: image RGB byte count mismatch";
            prev_end = im.pos + im.span;
            req.images.push_back(std::move(im));
        }
    }
    return "";
}

// Token-budgeted dynamic batch queue shared between socket ingress thread and GPU worker thread.
class EmbedBatchQueue {
public:
    explicit EmbedBatchQueue(int max_batched_tokens, int max_batch = 256)
        : max_tokens_(std::max(1, max_batched_tokens)), max_batch_(std::max(1, max_batch)) {}

    bool push(EmbedRequest req) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return false;
        if (!in_flight_.insert(req.id).second) return false;
        q_.push_back(std::move(req));
        cv_.notify_one();
        return true;
    }

    // Cancels a queued (not yet running) request; returns true if removed before execution.
    bool cancel_queued(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = q_.begin(); it != q_.end(); ++it) {
            if (it->id == id) {
                q_.erase(it);
                in_flight_.erase(id);
                return true;
            }
        }
        return false;
    }

    void mark_done(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        in_flight_.erase(id);
    }

    size_t queued_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

    // Blocks until at least one request is available (or closed), then packs as many queued requests
    // as fit within max_batched_tokens and max_batch (always admitting at least the head request).
    std::vector<EmbedRequest> pop_batch() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return closed_ || !q_.empty(); });
        std::vector<EmbedRequest> batch;
        if (q_.empty()) return batch;
        int tokens = 0;
        while (!q_.empty()) {
            const int n = static_cast<int>(q_.front().prompt.size());
            if (!batch.empty() && (static_cast<int>(batch.size()) >= max_batch_ || tokens + n > max_tokens_)) break;
            tokens += n;
            batch.push_back(std::move(q_.front()));
            q_.pop_front();
        }
        return batch;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    int max_tokens_;
    int max_batch_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<EmbedRequest> q_;
    std::unordered_set<std::string> in_flight_;
    bool closed_ = false;
};

// Serves one accepted Unix domain socket connection speaking PROTOCOL.md v1.
// `run_batch` executes a packed batch of EmbedRequests and returns one EmbedOutput per request in order.
inline void serve_connection(int fd, const EmbedConfig& cfg, int max_batched_tokens, EmbedMetrics& metrics,
                             const std::function<std::vector<EmbedOutput>(const std::vector<EmbedRequest>&)>& run_batch,
                             int max_batch = 256) {
    if (!send_frame(fd, make_hello_json(cfg))) return;

    EmbedBatchQueue queue(max_batched_tokens, max_batch);
    std::mutex write_mu;
    std::atomic<bool> write_failed{false};

    auto safe_send = [&](const std::string& body) -> bool {
        std::lock_guard<std::mutex> lk(write_mu);
        if (write_failed.load()) return false;
        if (!send_frame(fd, body)) {
            write_failed.store(true);
            return false;
        }
        return true;
    };

    std::thread worker([&] {
        for (;;) {
            auto batch = queue.pop_batch();
            if (batch.empty()) break;
            metrics.waiting.store(queue.queued_count());
            metrics.running.store(batch.size());
            std::vector<EmbedOutput> outs;
            try {
                outs = run_batch(batch);
            } catch (const std::exception& e) {
                outs.clear();
                for (const auto& r : batch) {
                    EmbedOutput o;
                    o.id = r.id;
                    o.prompt_tokens = static_cast<int>(r.prompt.size());
                    o.error = e.what();
                    outs.push_back(std::move(o));
                }
            }
            metrics.running.store(0);
            for (size_t i = 0; i < batch.size(); ++i) {
                const auto& req = batch[i];
                const auto& o = (i < outs.size()) ? outs[i] : EmbedOutput{req.id, int(req.prompt.size()), {}, "internal output count mismatch", 0.0};
                queue.mark_done(req.id);
                if (!o.error.empty()) {
                    metrics.req_error.fetch_add(1);
                    safe_send(make_finish_json(req.id, "error", o.prompt_tokens, o.error));
                } else {
                    metrics.req_stop.fetch_add(1);
                    metrics.prompt_tokens.fetch_add(uint64_t(o.prompt_tokens));
                    if (safe_send(make_embedding_json(req.id, o.embedding, o.prompt_tokens))) {
                        safe_send(make_finish_json(req.id, "stop", o.prompt_tokens, "", o.prefill_s));
                    }
                }
            }
            metrics.waiting.store(queue.queued_count());
        }
    });

    std::string body;
    bool clean_eof = false;
    while (!write_failed.load() && recv_frame(fd, body, &clean_eof)) {
        g6v::Json obj;
        try {
            obj = g6v::Json::parse(body);
        } catch (...) {
            break; // PROTOCOL.md: malformed JSON closes connection
        }
        if (obj.t != g6v::Json::Obj) break;
        const auto* ty = obj.find("type");
        if (!ty || ty->t != g6v::Json::Str) continue;

        if (ty->s == "embed") {
            EmbedRequest req;
            const std::string err = parse_embed_request(obj, cfg, req);
            if (!err.empty()) {
                metrics.req_error.fetch_add(1);
                if (!req.id.empty()) {
                    safe_send(make_finish_json(req.id, "error", static_cast<int>(req.prompt.size()), err));
                }
                continue;
            }
            if (!queue.push(std::move(req))) {
                metrics.req_error.fetch_add(1);
                const auto* id_v = obj.find("id");
                if (id_v && id_v->t == g6v::Json::Str) {
                    safe_send(make_finish_json(id_v->s, "error", 0, "duplicate id"));
                }
            } else {
                metrics.waiting.store(queue.queued_count());
            }
        } else if (ty->s == "cancel") {
            if (const auto* id_v = obj.find("id"); id_v && id_v->t == g6v::Json::Str) {
                if (queue.cancel_queued(id_v->s)) {
                    metrics.req_abort.fetch_add(1);
                    metrics.waiting.store(queue.queued_count());
                    safe_send(make_finish_json(id_v->s, "abort", 0));
                }
            }
        } else if (ty->s == "metrics") {
            uint64_t seq = 0;
            if (const auto* sv = obj.find("seq"); sv && sv->t == g6v::Json::Num && sv->exact_integer && sv->i64() >= 0) {
                seq = static_cast<uint64_t>(sv->i64());
            }
            safe_send(make_metrics_json(seq, metrics));
        } else if (ty->s == "generate") {
            if (const auto* id_v = obj.find("id"); id_v && id_v->t == g6v::Json::Str) {
                metrics.req_error.fetch_add(1);
                safe_send(make_finish_json(id_v->s, "error", 0, "g6embed only supports embed requests"));
            }
        }
    }

    queue.close();
    worker.join();
}

// Binds and listens on a Unix domain socket path, serving connections sequentially until `stop` becomes true.
inline int listen_unix_socket(const std::string& sock_path) {
    ::unlink(sock_path.c_str());
    const int sfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (sock_path.size() >= sizeof(addr.sun_path)) {
        ::close(sfd);
        return -1;
    }
    std::memcpy(addr.sun_path, sock_path.c_str(), sock_path.size() + 1);
    if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(sfd);
        return -1;
    }
    if (::listen(sfd, 16) != 0) {
        ::close(sfd);
        return -1;
    }
    return sfd;
}

} // namespace g6emb
