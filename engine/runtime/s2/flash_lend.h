// FLASH_CAP_POLICY=borrow-idle: the head's rule for lending the Flash rank's KV chunks inside CAP
// (runtime/flash_kv.h) to the KV node beside it while the serve is idle, and recalling them before a request that needs
// them is admitted; the rule is engine/scheduler-rs/src/flash_lend.rs. The head drives both as Resize events on tier
// FLASH_KV_TIER (scheduler.h), so every rank applies them at the same commit. Each rank's ACK carries its chunks online
// (Ack::room); the head admits a request only when the least of them covers the request's worst-case rows. A recall the
// node does not release in time leaves that rank short: the request waits (it never runs on offline pages) and the head
// recalls again at the next attempt. A request that needs lent chunks waits for the recall before its admission (its
// TTFT includes it). Host only, the head's control thread.
#pragma once
#include "../../scheduler-rs/g6sched.h"
#include <chrono>
#include <cstdint>
#include <optional>

namespace g6s2 {

struct FlashLend {
    using Clock = std::chrono::steady_clock;
    long idle_ms = 0;              // idle this long before a lend; 0 = never lend (FLASH_CAP_POLICY=protected)
    uint32_t keep = 1;             // chunks left online while lent
    uint64_t chunk_rows = 0;       // rows one chunk holds (chunk pages x PAGE)
    uint32_t chunks = 0;           // chunks in the map

    // the chunks a run of `rows` rows needs online (at most all of them)
    uint32_t need(uint64_t rows) const { const g6sch_lend l = state(); return g6sch_lend_need(&l, rows); }
    // controller, every poll: the engine is idle (no resident session, none queued). The chunk count a lend takes the
    // ranks to, once per idle stretch of idle_ms, or nothing
    std::optional<uint32_t> poll(bool idle, Clock::time_point now)
    {
        g6sch_lend l = state();
        const int64_t n = g6sch_lend_poll(&l, idle, std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
        idle_ = l.idle; lent_ = l.lent; since_ns_ = l.since_ns;
        if (n < 0) return {};
        return uint32_t(n);
    }
    // admission of a request whose run may reach `rows` rows, with `room` the least chunks online over the ranks: 0 =
    // admit, else the chunk count to recall to first (the request waits for it)
    uint32_t recall(uint64_t rows, uint32_t room) const { const g6sch_lend l = state(); return g6sch_lend_recall(&l, rows, room); }

private:
    bool idle_ = false, lent_ = false;
    int64_t since_ns_ = 0;
    g6sch_lend state() const { return {int64_t(idle_ms), keep, chunk_rows, chunks, idle_, lent_, since_ns_}; }
};

}   // namespace g6s2
