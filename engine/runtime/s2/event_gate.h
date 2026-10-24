// ADMIT/CANCEL-only agreement. No GPU calls or per-round transport.
// Computational thread calls before_work(B) before departing safe boundary B.
// Control thread installs future notices and supplies commits. Per-event quiescence
// ACKs (arrive) follow stream completion; a commit must be sent only after all ACKs.
#pragma once
#include "scheduler.h"
#include <atomic>
#include <condition_variable>
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
// room: the rank's Flash KV chunks online (G6_FLASH_KV_CHUNKS, runtime/flash_kv.h), UINT32_MAX without a chunk map.
// Not replicated state: a recall the KV node did not release in time leaves one rank short (s2/flash_lend.h)
struct Ack { Notice notice; unsigned rank = 0; uint64_t state_hash = 0; uint32_t room = UINT32_MAX; };
class EventGate {
    const uint64_t epoch_; const unsigned rank_;
    std::atomic<int64_t> passed_{-1};
    std::atomic<uint64_t> due_{UINT64_MAX};
    std::atomic<bool> failed_{false};
    std::mutex mu_; std::condition_variable cv_;
    std::optional<Notice> n_;
    uint64_t next_event_ = 1;
    bool arrived_ = false, committed_ = false;
    bool held_ = false;   // applied(n, true): stopped at due_, and the next notice must name it
public:
    EventGate(uint64_t epoch, unsigned rank) : epoch_(epoch), rank_(rank) { require(epoch, "bad epoch"); }
    // No-event hot path: atomic scalars only, no mutex, syscall, message, digest or allocation.
    bool before_work(uint64_t boundary) {
        require(boundary <= uint64_t(INT64_MAX), "boundary overflow");
        require(!failed_.load(), "control epoch failed");
        require(int64_t(boundary) > passed_.load(), "boundary already departed or regressed");
        const uint64_t due = due_.load();
        require(due >= boundary, "missed event boundary");
        // A stopped event boundary has NOT been departed. This permits a second
        // event at the same quiescent boundary before any new GPU work starts.
        if (due == boundary) return true;
        passed_.store(int64_t(boundary));
        // Sequentially consistent atomics pair with announce's recheck. In the
        // narrow install/depart race, fail-stop rather than execute past a notice.
        if (due_.load() <= boundary) {
            fail(); throw std::runtime_error("departure raced event notice");
        }
        return false;
    }
    void announce(const Notice& n) { // background control reader, not the decode loop
        std::lock_guard<std::mutex> lock(mu_);
        require(!failed_ && !n_, "pending event or failed epoch");
        require(n.epoch == epoch_ && n.event == next_event_ &&
                (n.kind == EventKind::Admit || n.kind == EventKind::Cancel || n.kind == EventKind::Resize),
                "notice epoch/sequence/kind mismatch");
        require(n.boundary <= uint64_t(INT64_MAX) && int64_t(n.boundary) > passed_.load(), "late event notice");
        require(!held_ || n.boundary == due_.load(), "notice does not name the held boundary");
        n_ = n; arrived_ = committed_ = held_ = false;
        due_.store(n.boundary);
        // Close race with before_work(): if it departed while notice was being installed,
        // fail the entire epoch. Never silently apply this event one round late.
        if (passed_.load() >= int64_t(n.boundary)) {
            failed_ = true; cv_.notify_all(); throw std::runtime_error("notice raced departed boundary");
        }
    }
    Ack arrive(uint64_t boundary, uint64_t state_hash, uint32_t room = UINT32_MAX) {
        std::lock_guard<std::mutex> lock(mu_);
        require(!failed_ && n_ && !arrived_ && n_->boundary == boundary && due_ == boundary, "invalid event arrival");
        arrived_ = true; return Ack{*n_, rank_, state_hash, room};
    }
    void commit(const Notice& n) { // only from a coordinator with all-rank ACK proof
        std::lock_guard<std::mutex> lock(mu_);
        require(!failed_ && n_ && *n_ == n && arrived_ && !committed_, "commit before event ACK or mismatched notice");
        committed_ = true; cv_.notify_all();
    }
    void wait_commit() { // called only at an installed event boundary
        std::unique_lock<std::mutex> lock(mu_);
        require(n_ && arrived_, "wait without event arrival");
        cv_.wait(lock, [&] { return committed_ || failed_.load(); });
        require(!failed_, "control epoch failed");
    }
    // hold (Commit::hold): stay stopped at this boundary, NOT departed, for the next event, which must name it. The
    // compute thread re-enters before_work(n.boundary), which returns true (due == boundary), and waits for that notice.
    void applied(const Notice& n, bool hold = false) { // caller applied canonical pool/scheduler changes successfully
        std::lock_guard<std::mutex> lock(mu_);
        require(n_ && *n_ == n && committed_ && !failed_, "invalid event application");
        require(next_event_ != UINT64_MAX, "event sequence exhausted");
        ++next_event_; n_.reset(); arrived_ = committed_ = false; held_ = hold; due_ = hold ? n.boundary : UINT64_MAX;
    }
    void fail() {
        // Share the waiter's mutex: an atomic predicate alone can lose a notify
        // between its predicate check and the condition-variable wait.
        { std::lock_guard<std::mutex> lock(mu_); failed_ = true; }
        cv_.notify_all();
    }
};

class Agreement { // rank0's per-event ACK set; never used in steady-state rounds
    unsigned nr_; Notice n_; std::vector<bool> seen_; std::optional<uint64_t> hash_; uint32_t room_ = UINT32_MAX;
public:
    Agreement(unsigned nr, Notice n) : nr_(nr), n_(n), seen_(nr, false) {
        require(nr == 2 || nr == 3 || nr == 4 || nr == 6, "expected TP2, TP3, TP4 or TP6 ranks");
    }
    void ack(const Ack& a) {
        require(a.rank < nr_ && a.notice == n_, "ACK rank/event mismatch");
        require(!seen_[a.rank], "duplicate event ACK");
        require(!hash_ || *hash_ == a.state_hash, "replicated state mismatch");
        hash_ = a.state_hash; seen_[a.rank] = true; room_ = std::min(room_, a.room);
    }
    bool ready() const { return std::all_of(seen_.begin(), seen_.end(), [](bool b) { return b; }); }
    uint64_t agreed_hash() const { require(ready(), "not all ranks quiescent"); return *hash_; }
    uint32_t room() const { require(ready(), "not all ranks quiescent"); return room_; }   // the least of the ranks' rooms
};
} // namespace g6s2
