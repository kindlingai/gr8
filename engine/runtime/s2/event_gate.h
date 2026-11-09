// ADMIT/CANCEL-only agreement. No GPU calls or per-round transport.
// Computational thread calls before_work(B) before departing safe boundary B.
// Control thread installs future notices and supplies commits. Per-event quiescence
// ACKs (arrive) follow stream completion; a commit must be sent only after all ACKs.
// The gate and the ACK set are Rust (engine/scheduler-rs src/control/gate.rs); these classes are facades.
#pragma once
#include "control_rs.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace g6s2 {
enum class EventKind : uint32_t { Admit = 1, Cancel = 2, Resize = 3 };
struct Notice {
    uint64_t epoch = 0, event = 0, boundary = 0, request = 0, payload_hash = 0;
    EventKind kind = EventKind::Admit;
    bool operator==(const Notice& n) const {
        return epoch == n.epoch && event == n.event && boundary == n.boundary &&
               request == n.request && payload_hash == n.payload_hash && kind == n.kind;
    }
};
// room: the rank's Flash KV chunks online (GR8_GLM53F_KV_CHUNKS, runtime/flash_kv.h), UINT32_MAX without a chunk map.
// Not replicated state: a recall the KV node did not release in time leaves one rank short (s2/flash_lend.h)
struct Ack { Notice notice; unsigned rank = 0; uint64_t state_hash = 0; uint32_t room = UINT32_MAX; };
static_assert(sizeof(Notice) == sizeof(g6ctl_notice) && offsetof(Notice, kind) == offsetof(g6ctl_notice, kind), "Notice layout");
static_assert(sizeof(Ack) == sizeof(g6ctl_ack) && offsetof(Ack, rank) == offsetof(g6ctl_ack, rank) &&
              offsetof(Ack, state_hash) == offsetof(g6ctl_ack, state_hash) && offsetof(Ack, room) == offsetof(g6ctl_ack, room),
              "Ack layout");
class EventGate {
    struct Free { void operator()(g6ctl_gate* g) const { g6ctl_gate_free(g); } };
    std::unique_ptr<g6ctl_gate, Free> h_;
public:
    EventGate(uint64_t epoch, unsigned rank) : h_((sch::check_abi(), g6ctl_gate_new(epoch, rank))) {
        if (!h_) throw std::runtime_error(g6ctl_error());
    }
    // No-event hot path: atomics only, no mutex, syscall, message, digest or allocation.
    bool before_work(uint64_t boundary) {
        bool due = false;
        ctl::check(g6ctl_gate_before_work(h_.get(), boundary, &due));
        return due;
    }
    void announce(const Notice& n) { ctl::check(g6ctl_gate_announce(h_.get(), sch::c<g6ctl_notice>(&n))); }
    Ack arrive(uint64_t boundary, uint64_t state_hash, uint32_t room = UINT32_MAX) {
        Ack a;
        ctl::check(g6ctl_gate_arrive(h_.get(), boundary, state_hash, room, sch::m<g6ctl_ack>(&a)));
        return a;
    }
    void commit(const Notice& n) { ctl::check(g6ctl_gate_commit(h_.get(), sch::c<g6ctl_notice>(&n))); }
    void wait_commit() { ctl::check(g6ctl_gate_wait_commit(h_.get())); }
    // hold (Commit::hold): stay stopped at this boundary, NOT departed, for the next event, which must name it. The
    // compute thread re-enters before_work(n.boundary), which returns true, and waits for that notice.
    void applied(const Notice& n, bool hold = false) { ctl::check(g6ctl_gate_applied(h_.get(), sch::c<g6ctl_notice>(&n), hold)); }
    void fail() { g6ctl_gate_fail(h_.get()); }
};

class Agreement { // rank0's per-event ACK set; never used in steady-state rounds
    struct Free { void operator()(g6ctl_agreement* a) const { g6ctl_agreement_free(a); } };
    std::unique_ptr<g6ctl_agreement, Free> h_;
public:
    Agreement(unsigned nr, Notice n) : h_(g6ctl_agreement_new(nr, sch::c<g6ctl_notice>(&n))) {
        if (!h_) throw std::runtime_error(g6ctl_error());
    }
    void ack(const Ack& a) { ctl::check(g6ctl_agreement_ack(h_.get(), sch::c<g6ctl_ack>(&a))); }
    bool ready() const { return g6ctl_agreement_ready(h_.get()); }
    uint64_t agreed_hash() const { uint64_t h = 0; ctl::check(g6ctl_agreement_hash(h_.get(), &h)); return h; }
    uint32_t room() const { uint32_t r = 0; ctl::check(g6ctl_agreement_room(h_.get(), &r)); return r; }   // the least of the ranks' rooms
};
} // namespace g6s2
