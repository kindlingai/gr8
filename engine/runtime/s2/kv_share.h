#pragma once
// A pool tier whose store is a group lease on a KV node (g6kv_tier_open_group, kvman-rs REMOTE-TIER.md "Sharing a
// node"): clusters share the node and capacity moves between them by whole segments. Rank 0 (the head) reports to the
// node now and then (share_poll); when the node grants or asks back segments, the head announces a Resize event and
// every rank applies it at that one point of the call sequence (share_apply), so every rank's pool queues the same
// copies. Host only: the engine passes its KvTiers::run + io_done as run_io.
#include "pool.h"
#include <algorithm>
#include <functional>
#include <optional>

namespace g6s2 {

// Segments more a rank the head asks for in this report, from the tier's counters at the last report and now (per =
// blocks a segment). Enough for what pushed it, not a fixed count:
// - an entry too big for the whole tier was offered since (oversize): room for it beside what the tier holds;
// - the tier filled since (it evicted to make room): the pages it pushed out (dropped, or moved to the next tier);
// - it took a spill since and the blocks left free are fewer than its entries hold on average, so the next spill of
//   that size would evict: ask BEFORE the tier is full, so the grant lands before an entry drops.
// At least one segment when any of these holds. A tier that took no spill asks for nothing, so an idle group never
// takes room from a busy one. The node grants up to the group's cap.
inline uint64_t share_want(const g6kv_tier_totals& last, const g6kv_tier_totals& now, uint64_t per) {
    if (per == 0) return 0;
    const auto segs = [per](uint64_t blocks) { return std::max<uint64_t>(1, (blocks + per - 1) / per); };
    uint64_t want = 0;
    if (now.oversize > last.oversize && now.oversize_last > now.free_slots)
        want = std::max(want, segs(now.oversize_last - now.free_slots));
    if (now.full > last.full)
        want = std::max(want, segs(now.dropped_pages - last.dropped_pages + now.moved_out_pages - last.moved_out_pages));
    else if (now.spilled_pages > last.spilled_pages && now.entries > 0) {
        const uint64_t avg = (now.slots - now.free_slots) / now.entries;
        if (now.free_slots < avg) want = std::max(want, segs(avg - now.free_slots));
    }
    return want;
}

// The head's report: `want` segments more per rank (0 = none; share_want) and the ages of the tier's coldest blocks.
// offer = false (KV_SHARE_WHEN=idle while the engine is busy): no ages, so the node asks this group to give only for a
// group starting below its floor, never for another group's growth. The resize to announce (it carries the node's
// transaction), or nothing.
inline std::optional<Resize> share_poll(RunPool& pool, uint32_t tier, const g6kv_tier* store, uint64_t want,
                                        bool offer = true, size_t most = 8) {
    const uint64_t per = g6kv_store_seg_blocks(store), segs = g6kv_store_segs(store);
    require(per > 0, "share_poll: the store is not a group lease");
    const std::vector<uint64_t> ages = offer ? pool.coldest(tier, per, most) : std::vector<uint64_t>{};
    g6kv_report r{};
    require(g6kv_store_report(store, want, ages.data(), ages.size(), &r) == 0, g6kv_tier_error());
    if (r.grant) return Resize{tier, segs + r.grant, (segs + r.grant) * per, r.txn};
    if (r.give && r.give < segs) return Resize{tier, segs - r.give, (segs - r.give) * per, r.txn};
    return std::nullopt;
}

// Every rank, at the event, with no copies pending. Grow: the lease first, then the tier, then the copies the pool
// had waiting for room. Shrink: the tier first (it evicts and queues moves into the slots that stay), the copies, then
// the lease gives its top segments back. run_io runs the pool's queued copies in order and calls io_done. false: the
// lease or tier keeps its size, on every rank alike: the tier refused a shrink (an indexed tier, or entries it may not
// evict), or the node refused the transaction (G6KV_STORE_REFUSED: a grant no rank took in time). A shrink the node
// refuses after the tier shrank leaves the lease larger than the tier, which is safe (the tier never addresses past
// its slots). A fault (-1) throws.
inline bool share_apply(RunPool& pool, const g6kv_tier* store, const Resize& r, const std::function<void()>& run_io) {
    const uint64_t segs = g6kv_store_segs(store);
    if (r.segs > segs) {
        uint64_t blocks = 0;
        const int rc = g6kv_store_grow(store, r.segs, r.txn, &blocks);
        if (rc == G6KV_STORE_REFUSED) return false;
        require(rc == 0, g6kv_tier_error());
        require(blocks == r.slots, "share_apply: the lease grew to a different size than the event's");
        require(pool.resize_tier(r.tier, blocks), "share_apply: the tier refused to grow");
        run_io();
    } else if (r.segs < segs) {
        if (!pool.resize_tier(r.tier, r.slots)) return false;
        run_io();
        uint64_t blocks = 0;
        const int rc = g6kv_store_shrink(store, r.segs, r.txn, &blocks);
        if (rc == G6KV_STORE_REFUSED) return false;
        require(rc == 0, g6kv_tier_error());
        require(blocks == r.slots, "share_apply: the lease shrank to a different size than the event's");
    }
    return true;
}

} // namespace g6s2
