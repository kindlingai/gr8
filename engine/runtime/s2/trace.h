#pragma once
// GR8_S2_TRACE=PATH: a Chrome trace of one rank's S2 loop in PATH.r<rank>.json, for chrome://tracing or
// ui.perfetto.dev. Timestamps are wall-clock microseconds, so the ranks' files and a client's request times
// (tools/trace/merge.py) line up on one timeline. Each event is written and flushed as it happens, and the JSON array is
// never closed, which both viewers accept, so a trace survives a crash. Host clock only: no device syncs, no state.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace g6s2 {

class Trace {
    FILE* f_ = nullptr;
    std::mutex m_;
    unsigned pid_ = 0;

    template <class... A>
    void write(const char* fmt, A... a) {
        std::lock_guard<std::mutex> g(m_);
        fprintf(f_, fmt, a...);
        fflush(f_);
    }

public:
    // The steps thread (the boundary loop) and the head's control thread (arrivals, admissions, notices).
    static constexpr int STEPS = 0, CONTROL = 1;

    static uint64_t now() {
        using namespace std::chrono;
        return uint64_t(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
    }

    Trace() = default;
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;
    ~Trace() {
        if (f_) fclose(f_);
    }

    // Starts PATH.r<rank>.json; an empty path traces nothing.
    void open(const std::string& path, unsigned rank) {
        if (path.empty()) return;
        const std::string file = path + ".r" + std::to_string(rank) + ".json";
        f_ = fopen(file.c_str(), "w");
        if (!f_) {
            fprintf(stderr, "# s2 trace: cannot write %s, tracing nothing\n", file.c_str());
            return;
        }
        pid_ = rank;
        write("[\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%u,\"args\":{\"name\":\"rank %u\"}},\n", pid_, pid_);
        write("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%u,\"tid\":%d,\"args\":{\"name\":\"steps\"}},\n", pid_, STEPS);
        write("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%u,\"tid\":%d,\"args\":{\"name\":\"control\"}},\n", pid_, CONTROL);
    }

    explicit operator bool() const { return f_ != nullptr; }

    // A span from t0 to t1. args: the inside of a JSON object, "" for none.
    void span(const char* name, int tid, uint64_t t0, uint64_t t1, const std::string& args) {
        write("{\"name\":\"%s\",\"ph\":\"X\",\"pid\":%u,\"tid\":%d,\"ts\":%llu,\"dur\":%llu,\"args\":{%s}},\n", name, pid_, tid,
              (unsigned long long)t0, (unsigned long long)(t1 > t0 ? t1 - t0 : 0), args.c_str());
    }

    // A point in time on one thread.
    void instant(const char* name, int tid, const std::string& args) {
        write("{\"name\":\"%s\",\"ph\":\"i\",\"s\":\"t\",\"pid\":%u,\"tid\":%d,\"ts\":%llu,\"args\":{%s}},\n", name, pid_, tid,
              (unsigned long long)now(), args.c_str());
    }
};

// "key":value pairs for Trace's args.
inline std::string targ(const char* k, uint64_t v) { return std::string("\"") + k + "\":" + std::to_string(v); }
inline std::string targ(const char* k, const char* v) { return std::string("\"") + k + "\":\"" + v + "\""; }

}  // namespace g6s2
