// Persistent event-only host links, used by a native adapter with connected TCP
// sockets. CPU tests use connected local sockets. No socket activity per round.
// The channels are Rust (engine/scheduler-rs src/control/channel.rs): these classes are facades over one handle, and
// the ControlIO links they are given must outlive them. A callback's exception (the digest, room, make_commit,
// on_quit) fails the epoch and is rethrown here once the Rust returns.
#pragma once
#include "control_io.h"
#include "control_protocol.h"
#include <chrono>
#include <functional>
#include <map>
#include <utility>

namespace g6s2 {
// wall clock in microseconds, for GR8_S2_EVENTLOG lines only (comparable across ranks up to their clock skew)
inline long long wall_us() {
    return (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Shared event latch: computation only takes a mutex after before_work says due.
// A caller MUST apply the returned commit, then applied(); exceptions fail epoch.
class EventEndpoint {
protected:
    struct Free { void operator()(g6ctl_endpoint* e) const { g6ctl_endpoint_free(e); } };
    std::unique_ptr<g6ctl_endpoint, Free> h_;
    explicit EventEndpoint(g6ctl_endpoint* h) : h_(h) { if (!h_) throw std::runtime_error(g6ctl_error()); }
public:
    // optional, set before the first event: this rank's room for its ACKs (Ack::room), read on the compute thread
    std::function<uint32_t()> room;
    virtual ~EventEndpoint() = default;
    EventEndpoint(const EventEndpoint&) = delete;
    EventEndpoint& operator=(const EventEndpoint&) = delete;
    void fail() { g6ctl_fail(h_.get()); }
    // No digest, lock, allocation or I/O if no event is due. Call before departing
    // a LIVE boundary. Idle engines use wait_notice() instead of departing it.
    template<class Digest> std::optional<Commit> boundary(uint64_t b,Digest&& digest) {
        struct Ctx { Digest& digest; const std::function<uint32_t()>& room; std::exception_ptr err; } ctx{digest, room, nullptr};
        auto fn = [](void* p, uint64_t* d, uint32_t* r) -> int32_t {
            auto& c = *static_cast<Ctx*>(p);
            // digest before room: both only read the engine (flash_room), so the order does not matter today
            try { *d = c.digest(); *r = c.room ? c.room() : UINT32_MAX; return 0; }
            catch (...) { c.err = std::current_exception(); return 1; }
        };
        bool due = false; g6ctl_commit out;
        ctl::check(g6ctl_boundary(h_.get(), b, fn, &ctx, &due, &out), &ctx.err);
        if (!due) return {};
        return ctl::from_ctl(out);
    }
    Notice wait_notice() { g6ctl_notice n; ctl::check(g6ctl_wait_notice(h_.get(), &n)); return ctl::notice(n); }
    // the current event's installed/arrived stamps (wall_us); call between boundary() and applied()
    std::pair<long long,long long> stamps() { int64_t i = 0, a = 0; g6ctl_stamps(h_.get(), &i, &a); return {i, a}; }
    void applied() { ctl::check(g6ctl_applied(h_.get())); }
    // Compute thread, every rank at the same point with the same seq (1, 2, ...): return once every rank has arrived. For
    // host work whose time differs by rank (a tier's copies) before GPU work that waits for peers with a timeout.
    void sync(uint64_t seq) { ctl::check(g6ctl_sync(h_.get(), seq)); }
};
class WorkerChannel : public EventEndpoint {
public:
    std::function<void()> on_quit; // the head quit in order; set before receive_events starts
    WorkerChannel(uint64_t epoch,unsigned rank,ControlIO& io)
        : EventEndpoint((sch::check_abi(), g6ctl_worker_new(epoch, rank, io.handle()))) {}
    void receive_events() { // background control thread, one persistent connection
        struct Ctx { WorkerChannel& w; std::exception_ptr err; } ctx{*this, nullptr};
        auto fn = [](void* p) -> int32_t {
            auto& c = *static_cast<Ctx*>(p);
            try { if (c.w.on_quit) c.w.on_quit(); return 0; }
            catch (...) { c.err = std::current_exception(); return 1; }
        };
        ctl::check(g6ctl_receive_events(h_.get(), fn, &ctx), &ctx.err);
    }
    // The bytes the head sent for this request with its latest ADMIT notice (the caller checks their hash)
    std::string request(uint64_t id) { std::string s; ctl::check(g6ctl_request(h_.get(), id, ctl::grow_string, &s)); return s; }
    // Admission is FIFO: once a request is admitted, it and every earlier one (admitted, or cancelled while queued)
    // are done with
    void admitted(uint64_t id) { g6ctl_admitted(h_.get(), id); }
};
class LeaderChannel : public EventEndpoint {
    static g6ctl_endpoint* make(uint64_t epoch, const std::vector<ControlIO*>& peers) {
        sch::check_abi();
        std::vector<const g6ctl_io*> h;
        for (auto* p : peers) h.push_back(p ? p->handle() : nullptr);
        return g6ctl_head_new(epoch, h.data(), h.size());
    }
public:
    LeaderChannel(uint64_t epoch,const std::vector<ControlIO*>& peers) : EventEndpoint(make(epoch, peers)) {}   // none at TP1: the head agrees with itself
    // Control thread: announce caller-chosen FUTURE boundary, collect event ACKs,
    // then make one canonical allocator/scheduler commit while compute is paused.
    // Ordinary EOS, decode and prefill steps never call this function.
    // payload: an ADMIT's request bytes, sent to every worker just before the notice
    void event(const Notice& n,const std::function<Commit(uint64_t)>& make_commit,const std::string* payload=nullptr) {
        struct Ctx { const std::function<Commit(uint64_t)>& f; std::exception_ptr err; } ctx{make_commit, nullptr};
        auto fn = [](void* p, uint64_t hash, g6ctl_commit* out) -> int32_t {
            auto& c = *static_cast<Ctx*>(p);
            try { *out = ctl::to_ctl(c.f(hash)); return 0; }
            catch (...) { c.err = std::current_exception(); return 1; }
        };
        ctl::check(g6ctl_event(h_.get(), sch::c<g6ctl_notice>(&n), fn, &ctx, payload != nullptr,
                               payload ? ctl::bytes(*payload) : nullptr, payload ? payload->size() : 0), &ctx.err);
    }
    // inside make_commit: the least room every rank's ACK of this event reported (UINT32_MAX: no rank has a chunk map)
    uint32_t agreed_room() const {return g6ctl_agreed_room(h_.get());}
    // Pass the head's orderly quit to every worker. A worker already gone is not an error here.
    void quit() { g6ctl_quit(h_.get()); }
};
} // namespace g6s2
