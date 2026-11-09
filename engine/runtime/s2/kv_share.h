#pragma once
// A pool tier whose store is a group lease on a KV node (g6kv_tier_open_group, kvman-rs REMOTE-TIER.md "Sharing a
// node"): clusters share the node and capacity moves between them by whole segments. Rank 0 (the head) reports to the
// node now and then (share_poll); when the node grants or asks back segments, the head announces a Resize event and
// every rank applies it at that one point of the call sequence (share_apply), so every rank's pool queues the same
// copies. Host only: the engine passes its KvTiers::run + io_done as run_io.
#include "pool.h"
#include "../../scheduler-rs/g6sched.h"
#include <algorithm>
#include <exception>
#include <functional>
#include <optional>

namespace g6s2 {

// Segments more a rank the head asks for in this report, from the tier's counters at the last report and now (per =
// blocks a segment): room for what pushed it since, or, after a spill or while busy (the engine has a request resident
// or queued), room for one more average entry before the tier fills. An idle tier that took no spill asks for
// nothing, so an idle group never takes room from a busy one. The rule is engine/scheduler-rs/src/kv_share.rs.
inline uint64_t share_want(const g6kv_tier_totals& last, const g6kv_tier_totals& now, uint64_t per, bool busy = false) {
    sch::check_abi();
    return g6sch_share_want(&last, &now, per, busy);
}

// The head's report: `want` segments more per rank (0 = none; share_want) and the ages of the tier's coldest blocks.
// offer = false (GR8_KV_SHARE_WHEN=idle while the engine is busy): no ages, so the node asks this group to give only for a
// group starting below its floor, never for another group's growth. The resize to announce (it carries the node's
// transaction), or nothing.
inline std::optional<Resize> share_poll(RunPool& pool, uint32_t tier, const g6kv_tier* store, uint64_t want,
                                        bool offer = true, size_t most = 8) {
    const uint64_t per = g6kv_store_seg_blocks(store), segs = g6kv_store_segs(store);
    require(per > 0, "share_poll: the store is not a group lease");
    const std::vector<uint64_t> ages = offer ? pool.coldest(tier, per, most) : std::vector<uint64_t>{};
    g6kv_report r{};
    require(g6kv_store_report(store, want, ages.data(), ages.size(), &r) == 0, g6kv_tier_error());
    g6sch_resize x;
    if (!g6sch_share_resize(tier, segs, per, r.grant, r.give, r.txn, &x)) return std::nullopt;
    return Resize{x.tier, x.segs, x.slots, x.txn};
}

// Every rank, at the event, with no copies pending. Grow: the lease first, then the tier, then the copies the pool
// had waiting for room. Shrink: the tier first (it evicts and queues moves into the slots that stay), the copies, then
// the lease gives its top segments back. run_io runs the pool's queued copies in order and calls io_done; the pool is
// not locked while it runs. false: the lease or tier keeps its size, on every rank alike: the tier refused a shrink (an
// indexed tier, or entries it may not evict), or the node refused the transaction (G6KV_STORE_REFUSED: a grant no rank
// took in time). A shrink the node refuses after the tier shrank leaves the lease larger than the tier, which is safe
// (the tier never addresses past its slots). A fault throws, as does run_io's own exception. The rule is
// engine/scheduler-rs/src/kv_share.rs apply.
inline bool share_apply(RunPool& pool, const g6kv_tier* store, const Resize& r, const std::function<void()>& run_io) {
    sch::check_abi();
    struct Io { const std::function<void()>& f; std::exception_ptr err; } io{run_io, nullptr};
    auto t = [](void* c) noexcept -> int32_t {
        Io& x = *static_cast<Io*>(c);
        try { x.f(); return 0; } catch (...) { x.err = std::current_exception(); return 1; }
    };
    const g6sch_resize x{r.tier, r.segs, r.slots, r.txn};
    const char* msg = "";
    const int32_t rc = g6s2_share_apply(pool.handle(), store, &x, t, &io, &msg);
    if (io.err) std::rethrow_exception(io.err);
    require(rc != 2, g6kv_tier_error());
    require(rc != 3, msg);
    require(rc != 4, "share_apply: the copies failed");
    return rc == 0;
}

} // namespace g6s2
