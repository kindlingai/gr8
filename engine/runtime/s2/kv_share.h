#pragma once
// A pool tier whose store is a group lease on a KV node (g6kv_tier_open_group, kvman-rs REMOTE-TIER.md "Sharing a
// node"): clusters share the node and capacity moves between them by whole segments. Rank 0 (the head) reports to the
// node now and then (share_poll); when the node grants or asks back segments, the head announces a Resize event and
// every rank applies it at that one point of the call sequence (share_apply), so every rank's pool queues the same
// copies. Host only: the engine passes its KvTiers::run + io_done as run_io.
#include "pool.h"
#include <functional>
#include <optional>

namespace g6s2 {

// The head's report: `want` segments more per rank (0 = none; the tier filled since the last report, say) and the
// ages of the tier's coldest blocks. The resize to announce, or nothing.
inline std::optional<Resize> share_poll(RunPool& pool, uint32_t tier, const g6kv_tier* store, uint64_t want,
                                        size_t most = 8) {
    const uint64_t per = g6kv_store_seg_blocks(store), segs = g6kv_store_segs(store);
    require(per > 0, "share_poll: the store is not a group lease");
    const std::vector<uint64_t> ages = pool.coldest(tier, per, most);
    g6kv_report r{};
    require(g6kv_store_report(store, want, ages.data(), ages.size(), &r) == 0, g6kv_tier_error());
    if (r.grant) return Resize{tier, segs + r.grant, (segs + r.grant) * per};
    if (r.give && r.give < segs) return Resize{tier, segs - r.give, (segs - r.give) * per};
    return std::nullopt;
}

// Every rank, at the event, with no copies pending. Grow: the lease first, then the tier, then the copies the pool
// had waiting for room. Shrink: the tier first (it evicts and queues moves into the slots that stay), the copies, then
// the lease gives its top segments back. run_io runs the pool's queued copies in order and calls io_done. false: the
// tier refused a shrink (an indexed tier, or entries it may not evict) and the lease keeps its size.
inline bool share_apply(RunPool& pool, const g6kv_tier* store, const Resize& r, const std::function<void()>& run_io) {
    const uint64_t segs = g6kv_store_segs(store);
    if (r.segs > segs) {
        uint64_t blocks = 0;
        require(g6kv_store_grow(store, r.segs - segs, &blocks) == 0, g6kv_tier_error());
        require(blocks == r.slots, "share_apply: the lease grew to a different size than the event's");
        require(pool.resize_tier(r.tier, blocks), "share_apply: the tier refused to grow");
        run_io();
    } else if (r.segs < segs) {
        if (!pool.resize_tier(r.tier, r.slots)) return false;
        run_io();
        uint64_t blocks = 0;
        require(g6kv_store_shrink(store, r.segs, &blocks) == 0, g6kv_tier_error());
        require(blocks == r.slots, "share_apply: the lease shrank to a different size than the event's");
    }
    return true;
}

} // namespace g6s2
