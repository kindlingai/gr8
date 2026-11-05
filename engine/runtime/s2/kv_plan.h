// How KvTiers::run (kv_tier.cuh) cuts the pool's copies into batches; the rule is engine/scheduler-rs/src/kv_plan.rs.
// A batch is one kind of copy (g6kv_io kind: 0 spill, 1 move, 2 restore), at most `batch` blocks, each in one host slot,
// and at most `items_cap` page copies. A batch's store reads go out at once, then its writes, so a batch never holds
// two copies whose order matters.
#pragma once
#include "../../kvman-rs/g6kvman.h"
#include "../../scheduler-rs/g6sched.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace g6s2 {
namespace kvt {
// one copy of a batch: page, part (g6kv_io), sub (part 2), and the host slot it goes to or comes from
struct Item { int page, part, sub, hslot; };
static_assert(sizeof(Item) == sizeof(g6sch_item) && sizeof(g6kv_io) == sizeof(g6sch_io), "scheduler-rs layouts");
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
    (void)io.at(i);
    Batch b;
    const size_t ni = size_t(std::max(items_cap, 0)), ns = size_t(std::max(batch, 0));
    b.items.resize(ni); b.from.resize(ns); b.to.resize(ns); b.slots.resize(ns); b.to_slots.resize(ns);
    g6sch_batch out{0, reinterpret_cast<g6sch_item*>(b.items.data()), int32_t(ni), 0, b.from.data(), b.to.data(),
                    b.slots.data(), b.to_slots.data(), int32_t(ns), 0};
    const int rc = g6sch_kv_next_batch(reinterpret_cast<const g6sch_io*>(io.data()), io.size(), &i, batch, per_slot, items_cap, &out);
    if (rc == 1) throw std::runtime_error("KV store copies: a batch took no copy");
    if (rc) throw std::logic_error("KV store copies: batch buffers too small");
    b.kind = out.kind;
    b.items.resize(size_t(out.n_items));
    const size_t n = size_t(out.n_slots);
    b.from.resize(n); b.slots.resize(n);
    if (b.kind == 1) { b.to.resize(n); b.to_slots.resize(n); } else { b.to.clear(); b.to_slots.clear(); }
    return b;
}
} // namespace kvt
} // namespace g6s2
