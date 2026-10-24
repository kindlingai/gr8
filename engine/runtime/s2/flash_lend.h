// FLASH_CAP_POLICY=borrow-idle (D-331 M7 B-2): the head's rule for lending the Flash rank's KV chunks inside CAP
// (runtime/flash_kv.h) to the KV node beside it while the serve is idle, and recalling them before a request that needs
// them is admitted. The head drives both as Resize events on tier FLASH_KV_TIER (scheduler.h), so every rank applies
// them at the same commit. Each rank's ACK carries its chunks online (Ack::room); the head admits a request only when
// the least of them covers the request's worst-case rows. A recall the node does not release in time leaves that rank
// short: the request waits (it never runs on offline pages) and the head recalls again at the next attempt.
// Changed contract: a request that needs lent chunks waits for the recall before its admission (its TTFT includes it).
// Host only, the head's control thread.
#pragma once
#include <algorithm>
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
    uint32_t need(uint64_t rows) const
    {
        if (!chunk_rows) return 0;
        return (uint32_t)std::min<uint64_t>((rows + chunk_rows - 1) / chunk_rows, chunks);
    }
    // controller, every poll: the engine is idle (no resident session, none queued). The chunk count a lend takes the
    // ranks to, once per idle stretch of idle_ms, or nothing
    std::optional<uint32_t> poll(bool idle, Clock::time_point now)
    {
        if (!idle_ms || !chunks) return {};
        if (!idle) { idle_ = lent_ = false; return {}; }
        if (!idle_) { idle_ = true; since_ = now; }
        if (lent_ || now - since_ < std::chrono::milliseconds(idle_ms)) return {};
        lent_ = true;
        return std::min(keep, chunks);
    }
    // admission of a request whose run may reach `rows` rows, with `room` the least chunks online over the ranks: 0 =
    // admit, else the chunk count to recall to first (the request waits for it)
    uint32_t recall(uint64_t rows, uint32_t room) const
    {
        const uint32_t n = need(rows);
        return room >= n ? 0 : n;
    }

private:
    bool idle_ = false, lent_ = false;
    Clock::time_point since_{};
};

}   // namespace g6s2
