// PROTOCOL.md v1 Unix domain socket server and continuous-batching request queue for g6tts.
// Pure C++17 host header (no CUDA dependency) so host unit tests compile directly with g++.
//
// Wire frames (u32 BE length + UTF-8 JSON):
//   Engine -> Front end:
//     hello:  {"type":"hello","version":1,"model":...,"max_model_len":CAP,"vocab_size":252288,
//              "eos_ids":[248046],"max_logprobs":0,"speech":true,"sample_rate":44100,"codebooks":10,"frame_samples":2048}
//     audio:  {"type":"audio","id":...,"frames":n,"pcm":"<b64 s16le>"?,"codes":[10*n frame-major]?}
//     codes:  {"type":"codes","id":...,"frames":F,"codes":[10*F frame-major]}
//     finish: {"type":"finish","id":...,"reason":"stop|length|abort|error","stop_id":248046|null,
//              "prompt_tokens":T,"cached_tokens":0,"completion_tokens":F,"message":...,"status":400}
//     metrics:{"type":"metrics","seq":N,...}
//   Front end -> Engine:
//     speech:       {"type":"speech","id":...,"rows":[11*T],"max_frames":N,"temperature":0.7,
//                    "top_p":0.7,"top_k":30,"seed":null,"emit":"pcm"|"codes"|"both"}
//     encode_audio: {"type":"encode_audio","id":...,"pcm":"<b64 f32le>"}
//     cancel:       {"type":"cancel","id":...}
//     metrics:      {"type":"metrics","seq":N}
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

#include "tts_config.h"
#include "../runtime/json.h"

namespace g6tts {

constexpr uint32_t MAX_PROTO_FRAME = 256u << 20;

enum class EmitMode {
    Pcm = 0,
    Codes = 1,
    Both = 2,
};

struct SpeechRequest {
    std::string id;
    std::vector<int32_t> rows; // flat frame-major [T, 11]
    int num_rows = 0;          // T = rows.size() / 11
    int max_frames = 600;
    float temperature = geom::DEFAULT_TEMPERATURE;
    float top_p = geom::DEFAULT_TOP_P;
    int top_k = geom::DEFAULT_TOP_K;
    uint64_t seed = 0;
    bool has_seed = false;
    EmitMode emit = EmitMode::Pcm;
};

struct EncodeAudioRequest {
    std::string id;
    std::vector<float> pcm; // mono 44.1 kHz f32le
};

struct TtsMetrics {
    std::atomic<uint64_t> req_stop{0};
    std::atomic<uint64_t> req_length{0};
    std::atomic<uint64_t> req_abort{0};
    std::atomic<uint64_t> req_error{0};
    std::atomic<uint64_t> prompt_tokens{0};
    std::atomic<uint64_t> gen_frames{0};
    std::atomic<uint64_t> running{0};
    std::atomic<uint64_t> waiting{0};
};

// RFC 4648 standard base64 encoder.
inline std::string encode_base64(const uint8_t* data, size_t len) {
    static constexpr char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = (i + 1 < len) ? data[i + 1] : 0u;
        const uint32_t c = (i + 2 < len) ? data[i + 2] : 0u;
        const uint32_t v = (a << 16) | (b << 8) | c;
        out.push_back(B64[(v >> 18) & 63u]);
        out.push_back(B64[(v >> 12) & 63u]);
        out.push_back((i + 1 < len) ? B64[(v >> 6) & 63u] : '=');
        out.push_back((i + 2 < len) ? B64[v & 63u] : '=');
    }
    return out;
}

// RFC 4648 standard base64 decoder.
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

inline std::string make_hello_json(const TtsConfig& cfg) {
    std::ostringstream os;
    os << "{\"type\":\"hello\",\"version\":1,\"model\":" << json_escape(cfg.model_name)
       << ",\"max_model_len\":" << cfg.cap
       << ",\"vocab_size\":" << cfg.vocab_size
       << ",\"eos_ids\":[" << cfg.eos_token_id << "]"
       << ",\"max_logprobs\":0"
       << ",\"speech\":true"
       << ",\"sample_rate\":" << geom::SAMPLE_RATE
       << ",\"codebooks\":" << cfg.num_codebooks
       << ",\"frame_samples\":" << geom::FRAME_SAMPLES
       << "}";
    return os.str();
}

// `pcm_s16` has `frames * 2048` samples when non-null; `codes_fx10` has `frames * 10` ints (frame-major) when non-null.
inline std::string make_audio_json(const std::string& id, int frames,
                                   const int16_t* pcm_s16, const int32_t* codes_fx10, int ncb = geom::NCB) {
    std::string out;
    out.reserve(128 + (pcm_s16 ? size_t(frames) * geom::FRAME_SAMPLES * 3 : 0) +
                (codes_fx10 ? size_t(frames) * ncb * 6 : 0));
    out += "{\"type\":\"audio\",\"id\":";
    out += json_escape(id);
    out += ",\"frames\":";
    out += std::to_string(frames);
    if (pcm_s16) {
        out += ",\"pcm\":\"";
        out += encode_base64(reinterpret_cast<const uint8_t*>(pcm_s16),
                             size_t(frames) * geom::FRAME_SAMPLES * sizeof(int16_t));
        out.push_back('"');
    }
    if (codes_fx10) {
        out += ",\"codes\":[";
        const size_t total = size_t(frames) * size_t(ncb);
        for (size_t i = 0; i < total; ++i) {
            if (i > 0) out.push_back(',');
            out += std::to_string(codes_fx10[i]);
        }
        out.push_back(']');
    }
    out.push_back('}');
    return out;
}

// `codes_fx10` has `frames * 10` ints in frame-major `[F, 10]` order.
inline std::string make_codes_json(const std::string& id, int frames,
                                   const int32_t* codes_fx10, int ncb = geom::NCB) {
    std::string out;
    const size_t total = size_t(frames) * size_t(ncb);
    out.reserve(96 + total * 6);
    out += "{\"type\":\"codes\",\"id\":";
    out += json_escape(id);
    out += ",\"frames\":";
    out += std::to_string(frames);
    out += ",\"codes\":[";
    for (size_t i = 0; i < total; ++i) {
        if (i > 0) out.push_back(',');
        out += std::to_string(codes_fx10[i]);
    }
    out += "]}";
    return out;
}

inline std::string make_finish_json(const std::string& id, const char* reason,
                                    int prompt_tokens, int completion_tokens,
                                    int stop_id = -1, const std::string& message = "",
                                    int http_status = 0, double prefill_s = -1.0, double decode_s = -1.0) {
    std::ostringstream os;
    os << "{\"type\":\"finish\",\"id\":" << json_escape(id)
       << ",\"reason\":" << json_escape(reason)
       << ",\"stop_id\":";
    if (stop_id >= 0) os << stop_id;
    else os << "null";
    os << ",\"prompt_tokens\":" << prompt_tokens
       << ",\"cached_tokens\":0"
       << ",\"completion_tokens\":" << completion_tokens;
    if (!message.empty()) os << ",\"message\":" << json_escape(message);
    if (http_status > 0) os << ",\"status\":" << http_status;
    if (prefill_s >= 0.0 && decode_s >= 0.0) {
        char tb[64];
        std::snprintf(tb, sizeof(tb), "{\"prefill_s\":%.6f,\"decode_s\":%.6f}", prefill_s, decode_s);
        os << ",\"timing\":" << tb;
    }
    os << "}";
    return os.str();
}

inline std::string make_metrics_json(uint64_t seq, const TtsMetrics& m) {
    std::ostringstream os;
    os << "{\"type\":\"metrics\",\"seq\":" << seq
       << ",\"requests\":{\"running\":" << m.running.load()
       << ",\"waiting\":" << m.waiting.load()
       << ",\"finished\":{\"stop\":" << m.req_stop.load()
       << ",\"length\":" << m.req_length.load()
       << ",\"abort\":" << m.req_abort.load()
       << ",\"error\":" << m.req_error.load() << "}}"
       << ",\"tokens\":{\"prompt\":" << m.prompt_tokens.load()
       << ",\"generation\":" << m.gen_frames.load() << "}}";
    return os.str();
}

// Parses and validates one `speech` frame against `cfg`. Returns error string (empty = OK) and sets `out_status`.
inline std::string parse_speech_request(const g6v::Json& obj, const TtsConfig& cfg,
                                        SpeechRequest& req, int& out_status) {
    req = SpeechRequest{};
    out_status = 400;

    const auto* id_v = obj.find("id");
    if (!id_v || id_v->t != g6v::Json::Str || id_v->s.empty()) return "speech: missing string id";
    req.id = id_v->s;

    const auto* rows_v = obj.find("rows");
    if (!rows_v || rows_v->t != g6v::Json::Arr || rows_v->a.empty()) {
        return "speech: rows must be a non-empty integer array";
    }
    const int rw = cfg.row_width();
    if (static_cast<int>(rows_v->a.size()) % rw != 0) {
        return "speech: rows length must be a multiple of " + std::to_string(rw);
    }
    req.num_rows = static_cast<int>(rows_v->a.size()) / rw;
    req.rows.reserve(rows_v->a.size());
    for (size_t i = 0; i < rows_v->a.size(); ++i) {
        const auto& elem = rows_v->a[i];
        if (elem.t != g6v::Json::Num || !elem.exact_integer) return "speech: rows entries must be integers";
        const int64_t v = elem.i64();
        const int col = static_cast<int>(i % size_t(rw));
        if (col == 0) {
            if (v < 0 || v >= cfg.vocab_size) return "speech: main token id out of range";
        } else {
            if (v < 0 || v >= cfg.tower_vocab) return "speech: codebook id out of range";
        }
        req.rows.push_back(static_cast<int32_t>(v));
    }

    if (const auto* mf_v = obj.find("max_frames"); mf_v && mf_v->t != g6v::Json::Null) {
        if (mf_v->t != g6v::Json::Num || !mf_v->exact_integer || mf_v->i64() < 1) {
            return "speech: max_frames must be an integer >= 1";
        }
        req.max_frames = static_cast<int>(mf_v->i64());
    }
    if (req.num_rows + req.max_frames > cfg.cap) {
        return "speech: prompt rows + max_frames exceeds KV capacity (" + std::to_string(cfg.cap) + ")";
    }

    if (const auto* t_v = obj.find("temperature"); t_v && t_v->t != g6v::Json::Null) {
        if (t_v->t != g6v::Json::Num || t_v->n < 0.0 || !std::isfinite(t_v->n)) {
            return "speech: temperature must be >= 0";
        }
        req.temperature = static_cast<float>(t_v->n);
    }
    if (const auto* tp_v = obj.find("top_p"); tp_v && tp_v->t != g6v::Json::Null) {
        if (tp_v->t != g6v::Json::Num || tp_v->n <= 0.0 || tp_v->n > 1.0 || !std::isfinite(tp_v->n)) {
            return "speech: top_p must be in (0, 1]";
        }
        req.top_p = static_cast<float>(tp_v->n);
    }
    if (const auto* tk_v = obj.find("top_k"); tk_v && tk_v->t != g6v::Json::Null) {
        if (tk_v->t != g6v::Json::Num || !tk_v->exact_integer || tk_v->i64() < 0) {
            return "speech: top_k must be an integer >= 0";
        }
        req.top_k = static_cast<int>(tk_v->i64());
    }
    if (const auto* s_v = obj.find("seed"); s_v && s_v->t != g6v::Json::Null) {
        if (s_v->t != g6v::Json::Num || !s_v->exact_integer) return "speech: seed must be an integer or null";
        req.seed = static_cast<uint64_t>(s_v->i64());
        req.has_seed = true;
    }
    if (const auto* e_v = obj.find("emit"); e_v && e_v->t != g6v::Json::Null) {
        if (e_v->t != g6v::Json::Str) return "speech: emit must be a string";
        if (e_v->s == "pcm") req.emit = EmitMode::Pcm;
        else if (e_v->s == "codes") req.emit = EmitMode::Codes;
        else if (e_v->s == "both") req.emit = EmitMode::Both;
        else return "speech: emit must be pcm, codes, or both";
    }
    out_status = 0;
    return "";
}

// Parses and validates one `encode_audio` frame. Returns error string (empty = OK).
inline std::string parse_encode_audio_request(const g6v::Json& obj, EncodeAudioRequest& req, int& out_status) {
    req = EncodeAudioRequest{};
    out_status = 400;
    const auto* id_v = obj.find("id");
    if (!id_v || id_v->t != g6v::Json::Str || id_v->s.empty()) return "encode_audio: missing string id";
    req.id = id_v->s;

    const auto* pcm_v = obj.find("pcm");
    if (!pcm_v || pcm_v->t != g6v::Json::Str || pcm_v->s.empty()) {
        return "encode_audio: pcm must be a non-empty base64 string";
    }
    std::vector<uint8_t> raw;
    if (!decode_base64(pcm_v->s, raw) || raw.empty() || (raw.size() % 4) != 0) {
        return "encode_audio: invalid base64 f32le pcm";
    }
    const size_t n = raw.size() / 4;
    if (n > size_t(geom::SAMPLE_RATE) * 120) {
        return "encode_audio: audio exceeds 120s limit";
    }
    req.pcm.resize(n);
    std::memcpy(req.pcm.data(), raw.data(), raw.size());
    out_status = 0;
    return "";
}

// Thread-safe queue connecting socket reader thread to GPU continuous-batching worker thread.
class TtsQueue {
public:
    bool push_speech(SpeechRequest req) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return false;
        if (!in_flight_.insert(req.id).second) return false;
        speech_q_.push_back(std::move(req));
        cv_.notify_one();
        return true;
    }

    bool push_encode(EncodeAudioRequest req) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return false;
        if (!in_flight_.insert(req.id).second) return false;
        encode_q_.push_back(std::move(req));
        cv_.notify_one();
        return true;
    }

    // Cancels a queued or active request. Returns 1 if removed from queue immediately,
    // 2 if marked for active-slot cancellation, 0 if unknown.
    int cancel(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = speech_q_.begin(); it != speech_q_.end(); ++it) {
            if (it->id == id) {
                speech_q_.erase(it);
                in_flight_.erase(id);
                return 1;
            }
        }
        for (auto it = encode_q_.begin(); it != encode_q_.end(); ++it) {
            if (it->id == id) {
                encode_q_.erase(it);
                in_flight_.erase(id);
                return 1;
            }
        }
        if (in_flight_.count(id)) {
            cancelled_.insert(id);
            cv_.notify_one();
            return 2;
        }
        return 0;
    }

    bool is_cancelled(const std::string& id) const {
        std::lock_guard<std::mutex> lk(mu_);
        return cancelled_.count(id) != 0;
    }

    void mark_done(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        in_flight_.erase(id);
        cancelled_.erase(id);
    }

    size_t queued_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return speech_q_.size() + encode_q_.size();
    }

    bool wait_for_work() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return closed_ || !speech_q_.empty() || !encode_q_.empty(); });
        return !closed_ || !speech_q_.empty() || !encode_q_.empty();
    }

    bool pop_encode(EncodeAudioRequest& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (encode_q_.empty()) return false;
        out = std::move(encode_q_.front());
        encode_q_.pop_front();
        return true;
    }

    bool pop_speech(SpeechRequest& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (speech_q_.empty()) return false;
        out = std::move(speech_q_.front());
        speech_q_.pop_front();
        return true;
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lk(mu_);
        return closed_;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<SpeechRequest> speech_q_;
    std::deque<EncodeAudioRequest> encode_q_;
    std::unordered_set<std::string> in_flight_;
    std::unordered_set<std::string> cancelled_;
    bool closed_ = false;
};

// Serves one accepted Unix domain socket connection.
// `run_worker(queue, safe_send, metrics)` runs on a dedicated worker thread until `queue.is_closed()`.
inline void serve_tts_connection(
    int fd,
    const TtsConfig& cfg,
    TtsMetrics& metrics,
    const std::function<void(TtsQueue&, const std::function<bool(const std::string&)>&, TtsMetrics&)>& run_worker) {
    if (!send_frame(fd, make_hello_json(cfg))) return;

    TtsQueue queue;
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

    std::thread worker([&] { run_worker(queue, safe_send, metrics); });

    std::string body;
    bool clean_eof = false;
    while (!write_failed.load() && recv_frame(fd, body, &clean_eof)) {
        g6v::Json obj;
        try {
            obj = g6v::Json::parse(body);
        } catch (...) {
            break; // malformed JSON closes connection
        }
        if (obj.t != g6v::Json::Obj) break;
        const auto* ty = obj.find("type");
        if (!ty || ty->t != g6v::Json::Str) continue;

        if (ty->s == "speech") {
            SpeechRequest req;
            int status = 0;
            const std::string err = parse_speech_request(obj, cfg, req, status);
            if (!err.empty()) {
                metrics.req_error.fetch_add(1);
                if (!req.id.empty()) {
                    safe_send(make_finish_json(req.id, "error", req.num_rows, 0, -1, err, status));
                }
                continue;
            }
            const std::string id_copy = req.id;
            const int pt = req.num_rows;
            if (!queue.push_speech(std::move(req))) {
                metrics.req_error.fetch_add(1);
                safe_send(make_finish_json(id_copy, "error", pt, 0, -1, "duplicate id", 400));
            } else {
                metrics.waiting.store(queue.queued_count());
            }
        } else if (ty->s == "encode_audio") {
            EncodeAudioRequest req;
            int status = 0;
            const std::string err = parse_encode_audio_request(obj, req, status);
            if (!err.empty()) {
                metrics.req_error.fetch_add(1);
                if (!req.id.empty()) {
                    safe_send(make_finish_json(req.id, "error", 0, 0, -1, err, status));
                }
                continue;
            }
            const std::string id_copy = req.id;
            if (!queue.push_encode(std::move(req))) {
                metrics.req_error.fetch_add(1);
                safe_send(make_finish_json(id_copy, "error", 0, 0, -1, "duplicate id", 400));
            } else {
                metrics.waiting.store(queue.queued_count());
            }
        } else if (ty->s == "cancel") {
            if (const auto* id_v = obj.find("id"); id_v && id_v->t == g6v::Json::Str) {
                if (queue.cancel(id_v->s) == 1) {
                    metrics.req_abort.fetch_add(1);
                    metrics.waiting.store(queue.queued_count());
                    safe_send(make_finish_json(id_v->s, "abort", 0, 0));
                }
            }
        } else if (ty->s == "metrics") {
            uint64_t seq = 0;
            if (const auto* sv = obj.find("seq"); sv && sv->t == g6v::Json::Num && sv->exact_integer && sv->i64() >= 0) {
                seq = static_cast<uint64_t>(sv->i64());
            }
            safe_send(make_metrics_json(seq, metrics));
        } else if (ty->s == "generate" || ty->s == "embed") {
            if (const auto* id_v = obj.find("id"); id_v && id_v->t == g6v::Json::Str) {
                metrics.req_error.fetch_add(1);
                safe_send(make_finish_json(id_v->s, "error", 0, 0, -1,
                                           "g6tts only supports speech and encode_audio requests", 400));
            }
        }
    }

    queue.close();
    worker.join();
}

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

} // namespace g6tts
