#pragma once
// GR8_S2_TRACE=PATH: a Chrome trace of one rank's S2 loop in PATH.r<rank>.json, for chrome://tracing or
// ui.perfetto.dev. Timestamps are wall-clock microseconds, so the ranks' files and a client's request times
// (tools/trace/merge.py) line up on one timeline. Each event is written as it happens, and the JSON array is never
// closed, which both viewers accept, so a trace survives a crash. Host clock only: no device syncs, no state.
// The file is Rust (engine/scheduler-rs src/service/trace.rs); this is a facade.
#include "../../scheduler-rs/g6sched.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace g6s2 {

class Trace {
    struct Free { void operator()(g6svc_trace* t) const { g6svc_trace_free(t); } };
    std::unique_ptr<g6svc_trace, Free> h_;
    static const uint8_t* b(const char* s) { return reinterpret_cast<const uint8_t*>(s); }

public:
    // The steps thread (the boundary loop) and the head's control thread (arrivals, admissions, notices).
    static constexpr int STEPS = 0, CONTROL = 1;

    static uint64_t now() { return g6svc_trace_now(); }

    Trace() = default;
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;

    // Starts PATH.r<rank>.json; an empty path traces nothing.
    void open(const std::string& path, unsigned rank) {
        if (path.empty()) return;
        h_.reset(g6svc_trace_open(b(path.c_str()), std::strlen(path.c_str()), rank));
    }

    explicit operator bool() const { return h_ != nullptr; }

    // A span from t0 to t1. args: the inside of a JSON object, "" for none.
    void span(const char* name, int tid, uint64_t t0, uint64_t t1, const std::string& args) {
        g6svc_trace_span(h_.get(), b(name), std::strlen(name), tid, t0, t1, b(args.c_str()), std::strlen(args.c_str()));
    }

    // A point in time on one thread.
    void instant(const char* name, int tid, const std::string& args) {
        g6svc_trace_instant(h_.get(), b(name), std::strlen(name), tid, b(args.c_str()), std::strlen(args.c_str()));
    }
};

// "key":value pairs for Trace's args.
inline std::string targ(const char* k, uint64_t v) { return std::string("\"") + k + "\":" + std::to_string(v); }
inline std::string targ(const char* k, const char* v) { return std::string("\"") + k + "\":\"" + v + "\""; }

}  // namespace g6s2
