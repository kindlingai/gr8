// The C-ABI functions of gr8's host-side S2 scheduling logic in Rust (for C++ callers: bool fields, <cstdint>) (engine/scheduler-rs, docs/HOST-CORE.md R2). Built as
// libg6sched.a with cargo; g6m4, m4sim and the host tests link it. The C++ headers in engine/runtime/s2 keep their
// types and call these; the structs here have the same layout as theirs (each wrapper checks with a static_assert).
// Fallible calls return 0 or a code the wrapper turns into its message, and leave their state unchanged on failure.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

// kv_plan.h: one copy the pool queued (g6kvman.h g6kv_io) and one copy of a batch
struct g6sch_io { uint32_t kind, page, tier, to; uint64_t slot, to_slot; uint32_t part, sub; };
struct g6sch_item { int32_t page, part, sub, hslot; };
// next_batch's result, in the caller's buffers: items_cap items, slots_cap entries of from, to, slots and to_slots
struct g6sch_batch {
    uint32_t kind;
    g6sch_item* items; int32_t items_cap, n_items;
    uint32_t *from, *to; uint64_t *slots, *to_slots; int32_t slots_cap, n_slots;
};
// the batch from io[*i] on, *i advanced past it: 0, 1 = no copy fits, 2 = a buffer is too small
int g6sch_kv_next_batch(const g6sch_io* io, size_t n, size_t* i, int batch, uint32_t per_slot, int items_cap, g6sch_batch* out);

// flash_lend.h: the borrow-idle rule's settings and state; since_ns in steady-clock nanoseconds
struct g6sch_lend { int64_t idle_ms; uint32_t keep; uint64_t chunk_rows; uint32_t chunks; bool idle, lent; int64_t since_ns; };
uint32_t g6sch_lend_need(const g6sch_lend* l, uint64_t rows);
int64_t g6sch_lend_poll(g6sch_lend* l, bool idle, int64_t now_ns);   // the chunk count a lend takes the ranks to, or -1
uint32_t g6sch_lend_recall(const g6sch_lend* l, uint64_t rows, uint32_t room);

// flash_ring.h: one session's KDA ring
struct g6sch_ring { int32_t first, rows; bool ring; };
int g6sch_ring_source(const g6sch_ring* r, int next, int* src);   // 0, 1 = outside the previous step
// 0, 1 = outside the previous step, 2 = a verify wider than the ring; kslot holds n entries
int g6sch_ring_rows(g6sch_ring* r, int per, int q, int pos, int n, bool verify, int* kslot, int* src);
// the Flash capture set as flags by row count (mx + 1 entries each): 0, 1 = invalid arguments, 2 = exceeds mx
int g6sch_flash_shapes(int sessions, int mtp, int chunk, bool adapt, int mx, uint8_t* mains, uint8_t* drafts, uint8_t* chains);

// cp_rows.h
uint64_t g6sch_cp_after(const int* bk, int nbk, int R);
int g6sch_cp_step_rows(int64_t v, int solo, int full, int* out);   // 0, or 1 = below solo
struct g6sch_cand { uint64_t id; bool past; };
void g6sch_cp_pick(const g6sch_cand* cand, size_t n, const uint64_t* served, size_t n_served, uint32_t rows, uint32_t per,
                   bool seg, bool* in_round, bool* capped, bool* fresh);

}
