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

// scheduler.h: the Scheduler's types (the same layout as g6s2's; Quantum holds SCHED_QMAX entries whatever MAX_SESSIONS
// is). The Scheduler calls return null or the refusal's message (a static string), and refuse before changing state,
// except where the C++ they replaced changed some first (complete_decode, complete_tail_pack's revision).
#define G6SCH_QMAX 32
struct g6sch_key { uint64_t epoch; uint32_t slot, generation; };
struct g6sch_identity { bool enabled; uint8_t profile_digest[32]; uint8_t scope; uint8_t schedule_digest[32]; uint32_t scale_bits, slot; };
struct g6sch_request { uint64_t id; uint32_t prompt_tokens, max_tokens; g6sch_identity steering; uint32_t r_rows; };
struct g6sch_run { uint32_t first_page, pages, table_version; uint64_t list_hash; };
struct g6sch_slot {
    g6sch_key key; g6sch_request req; g6sch_run run;
    uint32_t phase, prefilled, emitted, inserted, deferred;
    bool packable, pack_tail;
    uint64_t parked;
};
// Config without the steering receipt
struct g6sch_config {
    uint64_t epoch;
    uint32_t pool_rows, max_rows, max_sessions, mtp, guard_tokens, prefill_rows;
    bool two_lanes;
    uint32_t decode_prefill_rows, decode_rounds, window_pages, cp_step_rows;
    uint64_t cp_after;
    bool cp_seg;
    uint32_t r2_rows;
    bool tail_first;
    uint32_t notice_max;
    bool c8_pack;
    uint32_t mtp_sessions;
    bool sess_drafts, park;
};
struct g6sch_quantum {
    uint32_t kind; uint64_t boundary; g6sch_key keys[G6SCH_QMAX]; uint32_t count, first, rows;
    uint32_t firsts[G6SCH_QMAX], row_counts[G6SCH_QMAX];
};
// the Scheduler's counters and the quantum in flight
struct g6sch_state {
    uint64_t boundary, event, revision, last_id, pool;
    bool have_id, want_decode, busy, failed;
    uint64_t park_seq; uint32_t decode_run;
    g6sch_quantum work;
};
struct g6sch_grant { g6sch_key key; g6sch_request req; g6sch_run run; uint32_t reuse, restored, counted; };
struct g6sch_shrink { g6sch_key key; uint32_t r_rows; g6sch_run run; };
struct g6sch_commit {
    uint64_t epoch, event, boundary, before_hash;
    bool has_cancel; g6sch_key cancel;
    bool has_admit; g6sch_grant admit;
    bool has_shrink; g6sch_shrink shrink;
};
struct g6sch_completion { g6sch_key key; uint32_t emitted; bool finished; uint32_t inserted; };
// the Rust side's struct sizes and constants, which the C++ wrapper checks once (scheduler.h check_abi)
enum {
    G6SCH_ABI_SLOT, G6SCH_ABI_CONFIG, G6SCH_ABI_QUANTUM, G6SCH_ABI_STATE, G6SCH_ABI_COMMIT, G6SCH_ABI_COMPLETION,
    G6SCH_ABI_GRANT, G6SCH_ABI_PAGE, G6SCH_ABI_PREFILL_DEFER_MAX, G6SCH_ABI_NOTICE_MIN, G6SCH_ABI_NOTICE_MAX,
    G6SCH_ABI_NOTICE_ROWS, G6SCH_ABI_MAX_DECODE_ROUNDS, G6SCH_ABI_PACK_MAX_MEMBERS, G6SCH_ABI_QMAX, G6SCH_ABI_TOTALS, G6SCH_ABI_RESIZE,
};
uint64_t g6sch_abi(uint32_t what);

// Config::validate's scalar rules; mtp0, pack, park: the module's capabilities
const char* g6sch_validate(const g6sch_config* c, uint32_t max_sessions, bool mtp0, bool pack, bool park);
const char* g6sch_reserve_rows(const g6sch_config* c, const g6sch_request* r, uint64_t* out);
const char* g6sch_reserve_pages(const g6sch_config* c, const g6sch_request* r, uint32_t* out);
const char* g6sch_grant_rows(const g6sch_config* c, const g6sch_request* r, uint64_t* out);
const char* g6sch_grant_pages(const g6sch_config* c, const g6sch_request* r, uint32_t* out);
uint64_t g6sch_round_rows(const g6sch_config* c, uint64_t length);
const char* g6sch_grant_pages_ok(const g6sch_config* c, const g6sch_request* r, uint32_t pages, bool* out);
uint32_t g6sch_prefill_left(const g6sch_slot* s);
bool g6sch_prefill_first(const g6sch_slot* a, const g6sch_slot* b);

// every call below takes the slots as (s, n)
void g6sch_init(const g6sch_config* c, g6sch_slot* s, size_t n);
const char* g6sch_idle_boundary(const g6sch_slot* s, size_t n, g6sch_state* st, uint64_t target);
uint32_t g6sch_resident(const g6sch_slot* s, size_t n);
uint32_t g6sch_retiring(const g6sch_slot* s, size_t n);
uint32_t g6sch_parked(const g6sch_slot* s, size_t n);
bool g6sch_find(const g6sch_slot* s, size_t n, uint64_t id, g6sch_key* out);
// cancel: null for none
const char* g6sch_next_key(const g6sch_config* c, const g6sch_slot* s, size_t n, const g6sch_key* cancel, bool has_park,
                           uint32_t park, g6sch_key* out);
size_t g6sch_reclaimable(const g6sch_config* c, const g6sch_slot* s, size_t n, const g6sch_key* cancel, g6sch_run* out);   // out: n runs
// the digest is digest_head, then steering.h hash_boot, then digest_slots
uint64_t g6sch_digest_head(const g6sch_config* c, const g6sch_state* st);
uint64_t g6sch_digest_slots(const g6sch_config* c, const g6sch_slot* s, size_t n, uint64_t h);
// digest: the state's digest before the commit. freed: n runs
const char* g6sch_apply(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st, const g6sch_commit* e,
                        uint64_t digest, g6sch_run* freed, size_t* n_freed, bool* admitted);
// k and r must not point into s
const char* g6sch_grow(const g6sch_config* c, g6sch_slot* s, size_t n, const g6sch_state* st, const g6sch_key* k, const g6sch_run* r);
size_t g6sch_decoding(const g6sch_slot* s, size_t n, g6sch_key* out);   // out: n keys
void g6sch_alternate(const g6sch_config* c, uint32_t kind, bool* want, uint32_t* run);
// stalled: n flags, or null
void g6sch_plan(const g6sch_config* c, const g6sch_slot* s, size_t n, bool want, const bool* stalled, g6sch_quantum* out);
const char* g6sch_next(const g6sch_config* c, const g6sch_slot* s, size_t n, g6sch_state* st, const bool* stalled, g6sch_quantum* out);
void g6sch_advance(g6sch_slot* s, size_t n, const g6sch_quantum* q);
uint64_t g6sch_horizon(const g6sch_config* c, const g6sch_slot* s, size_t n, const g6sch_state* st);
// k must not point into s
const char* g6sch_skip_prefill(g6sch_slot* s, size_t n, const g6sch_state* st, const g6sch_key* k);
const char* g6sch_complete_prefill(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st);
const char* g6sch_complete_prefill_pack(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st);
const char* g6sch_complete_tail(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st, const g6sch_completion* r);
const char* g6sch_complete_tail_pack(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st, const g6sch_completion* r, size_t nr);
const char* g6sch_complete_decode(const g6sch_config* c, g6sch_slot* s, size_t n, g6sch_state* st, const g6sch_completion* r, size_t nr);

// kv_share.h: a tier's counters are g6kvman.h g6kv_tier_totals (the same layout); a Resize is scheduler.h's
struct g6sch_resize { uint32_t tier; uint64_t segs, slots, txn; };
struct g6kv_tier_totals;
uint64_t g6sch_share_want(const g6kv_tier_totals* last, const g6kv_tier_totals* now, uint64_t per, bool busy);
// true and *out written when the node's answer is a resize
bool g6sch_share_resize(uint32_t tier, uint64_t segs, uint64_t per, uint64_t grant, uint64_t give, uint64_t txn, g6sch_resize* out);

// scheduler.h AdmissionQueue: a Rust-owned queue behind an opaque pointer
struct g6sch_queue;
struct g6sch_status { uint32_t why, requested, room; };   // why: scheduler.h Stall
g6sch_queue* g6sch_queue_new(const g6sch_config* c);
void g6sch_queue_free(g6sch_queue* q);
const char* g6sch_queue_push(g6sch_queue* q, const g6sch_request* r);
bool g6sch_queue_cancel(g6sch_queue* q, uint64_t id);
const g6sch_request* g6sch_queue_front(const g6sch_queue* q);   // null when empty; valid until the queue changes
size_t g6sch_queue_size(const g6sch_queue* q);
uint64_t g6sch_queue_stalls(const g6sch_queue* q);
const char* g6sch_queue_admitted(g6sch_queue* q, uint64_t id);
const char* g6sch_queue_check(g6sch_queue* q, uint64_t revision, bool slot, bool fits, uint32_t room, bool* has, g6sch_status* out);

// sessions.h parse_request: the parsed request, its arrays owned by the handle the parse returned
struct g6sch_ints { const int32_t* p; size_t n; };
struct g6sch_parsed_req {
    g6sch_ints prompt, stop;
    int32_t n_gen, min_tokens;
    float temp, top_p;
    uint32_t seed;
    int32_t lp, plp;
    int64_t sys;
    g6sch_identity steering;
    int32_t ins_after, ins_unless;
    g6sch_ints ins_tokens;
    uint32_t gram_kind;
    bool gram_strict;
    int32_t gram_arm;
    const uint8_t* schema;
    size_t schema_n;
};
struct g6sch_parsed;
// the caller's steering resolution: null and *out written, or its refusal (valid until the caller's next call). It
// must not throw.
typedef const char* (*g6sch_resolve_fn)(void* ctx, bool has_profile, const uint8_t* profile, size_t profile_n,
                                        uint32_t scale_bits, g6sch_identity* out);
const char* g6sch_parse_request(const uint8_t* bytes, size_t n, bool grammar_ready, g6sch_resolve_fn resolve, void* ctx,
                                g6sch_parsed_req* out, g6sch_parsed** handle);
void g6sch_parsed_free(g6sch_parsed* handle);
// one `gram` field; *handle owns the schema bytes (g6sch_parsed_free)
const char* g6sch_parse_grammar(const uint8_t* s, size_t n, uint32_t* kind, bool* strict, int32_t* arm, const uint8_t** schema,
                                size_t* schema_n, g6sch_parsed** handle);

}
