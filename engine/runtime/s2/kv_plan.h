// How KvTiers::run (kv_tier.cuh) cuts the pool's copies into batches: host-only, so the host tests run it without CUDA.
// A batch is one kind of copy (g6kv_io kind: 0 spill, 1 move, 2 restore), at most `batch` blocks, each in one host slot,
// and at most `items_cap` page copies. A batch's store reads go out at once, then its writes, so a batch never holds
// two copies whose order matters.
#pragma once
#include "../../kvman-rs/g6kvman.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace g6s2 {
namespace kvt {
// one copy of a batch: page, part (g6kv_io), sub (part 2), and the host slot it goes to or comes from
struct Item { int page, part, sub, hslot; };
// per host slot j: block slots[j] of tier from[j] (a move: into block to_slots[j] of tier to[j]). items: the spills' or
// restores' page copies
struct Batch {
    uint32_t kind = 0;
    std::vector<Item> items;
    std::vector<uint32_t> from, to;
    std::vector<uint64_t> slots, to_slots;
};
// the batch from io[i] on; advances i past it. per_slot: replicated parts a block holds (0: no binding is CP)
inline Batch next_batch(const std::vector<g6kv_io>& io, size_t& i, int batch, uint32_t per_slot, int items_cap) {
    Batch b;
    b.kind = io.at(i).kind;
    const int need = int(per_slot ? per_slot : 1);   // a block's most copies: a page, a shard, or per_slot parts
    auto in = [&](uint32_t tier, uint64_t slot) {
        for (size_t j = 0; j < b.slots.size(); ++j) if (b.from[j] == tier && b.slots[j] == slot) return true;
        return false;
    };
    while (i < io.size() && io[i].kind == b.kind) {
        const g6kv_io& x = io[i];
        if (b.kind == 1) {
            // a move that reads a block an earlier move of the batch writes (the batch reads first), or writes a block
            // an earlier move writes (one write_many has no order): the next batch
            bool dep = false;
            for (size_t j = 0; j < b.slots.size() && !dep; ++j)
                dep = (b.to[j] == x.tier && b.to_slots[j] == x.slot) || (b.to[j] == x.to && b.to_slots[j] == x.to_slot);
            if (dep || (int)b.slots.size() == batch) break;
            b.from.push_back(x.tier); b.to.push_back(x.to); b.slots.push_back(x.slot); b.to_slots.push_back(x.to_slot); ++i;
            continue;
        }
        // a block's copies are next to each other: one host slot for them all. A spill's block starts at part 0, part 1
        // or part 2 sub 0, so only its later replicated parts join it; a block spilled twice in one call (a slot freed
        // and taken again) is two writes, in order: the second starts the next batch. Restores of one block share its
        // host slot (read once, copied to each page)
        const bool same = !b.slots.empty() && b.from.back() == x.tier && b.slots.back() == x.slot &&
                          (b.kind == 2 || (x.part == 2 && x.sub > 0));
        if (same) {
            if ((int)b.items.size() == items_cap) break;   // a restore only: the next batch reads the block again
        } else {
            if ((int)b.slots.size() == batch || (int)b.items.size() + need > items_cap) break;
            if (b.kind == 0 && in(x.tier, x.slot)) break;
            b.from.push_back(x.tier); b.slots.push_back(x.slot);
        }
        b.items.push_back({(int)x.page, (int)x.part, (int)x.sub, (int)b.slots.size() - 1}); ++i;
    }
    if (b.slots.empty()) throw std::runtime_error("KV store copies: a batch took no copy");
    return b;
}
} // namespace kvt
} // namespace g6s2
