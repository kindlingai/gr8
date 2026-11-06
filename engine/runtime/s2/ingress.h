// File ingress for opt-in S2 shim. Control thread only, no per-round file polling.
// The files' rules are Rust (engine/scheduler-rs src/service/files.rs); these are facades.
#pragma once
#include "control_rs.h"
#include "../serve.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace g6s2 {
inline uint64_t payload_hash(const std::string& bytes) { return g6svc_payload_hash(ctl::bytes(bytes), bytes.size()); }
// paths end at their first NUL, as c_str() does
inline size_t path_n(const std::string& p) { return std::strlen(p.c_str()); }
inline bool exists(const std::string& p) { return g6svc_exists(ctl::bytes(p), path_n(p)); }
inline std::string read_file(const std::string& p) {
    std::string out;
    ctl::check(g6svc_read_file(ctl::bytes(p), path_n(p), ctl::grow_string, &out));
    return out;
}
inline uint64_t number(const std::string& s) {
    uint64_t n = 0;
    ctl::check(g6svc_number(ctl::bytes(s), s.size(), &n));
    return n;
}
// resize: a Resize event the head makes itself (Service share poll), never from the ingress files
struct Command { bool admit=false; uint64_t epoch=0, request=0, hash=0; std::optional<Resize> resize{}; };
namespace svc {
inline Command command(const g6svc_command& c) { return Command{c.admit, c.epoch, c.request, c.hash}; }
inline g6svc_command command(const Command& c) { return g6svc_command{c.epoch, c.request, c.hash, c.admit}; }
// what ready.s2 and the agreement read; valid while c is
struct Boot {
    SteeringView steering;
    g6svc_boot b{};
    Boot(const Config& c, unsigned ranks) : steering(c.steering) {
        b.scheduler = &c; b.steering = steering.v;
        b.max_sessions = MAX_SESSIONS;
        b.mtp0 = g6mod::has(G6_CAP_MTP0); b.pack = g6mod::has(G6_CAP_PACK); b.park = g6mod::has(G6_CAP_PARK);
        b.ranks = ranks; b.lp_max = uint32_t(g6rt::LP_MAX);
    }
    Boot(const Boot&) = delete;
    Boot& operator=(const Boot&) = delete;
};
} // namespace svc
inline Command parse_command(const std::string& text, uint64_t epoch) {
    g6svc_command c{};
    ctl::check(g6svc_parse_command(ctl::bytes(text), text.size(), epoch, &c));
    return svc::command(c);
}
// ready.s2's line. Version 3 adds insert rules. Version 4 appends the immutable steering receipt of one profile,
// version 5 that of several.
inline std::string capability(const Config& c, unsigned ranks) {
    const svc::Boot b(c, ranks);
    std::string out;
    ctl::check(g6svc_capability(&b.b, ctl::grow_string, &out));
    return out;
}
class Ingress {
    struct Free { void operator()(g6svc_ingress* i) const { g6svc_ingress_free(i); } };
    std::unique_ptr<g6svc_ingress, Free> h_;
    static std::optional<Command> got(int32_t rc, bool has, const g6svc_command& c) {
        ctl::check(rc);
        if (!has) return {};
        return svc::command(c);
    }
public:
    Ingress(std::string dir, uint64_t epoch) {
        g6svc_ingress* h = nullptr;
        const int32_t rc = g6svc_ingress_new(ctl::bytes(dir), path_n(dir), epoch, &h);
        h_.reset(h);
        ctl::check(rc);
    }
    std::optional<Command> next() {
        bool has = false; g6svc_command c{};
        const int32_t rc = g6svc_ingress_next(h_.get(), &has, &c);
        return got(rc, has, c);
    }
    // the next command without consuming it (the head's ADMIT_HOLD drain stops at the first one that is not an ADMIT)
    std::optional<Command> peek() const {
        bool has = false; g6svc_command c{};
        const int32_t rc = g6svc_ingress_peek(h_.get(), &has, &c);
        return got(rc, has, c);
    }
    std::string request(const Command& c) const {
        const g6svc_command x = svc::command(c);
        std::string out;
        ctl::check(g6svc_ingress_request(h_.get(), &x, ctl::grow_string, &out));
        return out;
    }
};
} // namespace g6s2
