// The C-ABI functions of gr8's host-side S2 scheduling logic in Rust (for C++ callers: bool fields, <cstdint>) (engine/scheduler-rs, docs/HOST-CORE.md R2). Built as
// libg6sched.a with cargo; g6m4, m4sim and the host tests link it. The C++ headers in engine/runtime/s2 keep their
// types and call these; the structs here have the same layout as theirs (static_asserts in each wrapper) and as the
// Rust's (the layout digest at the end, checked at startup). Fallible calls return 0 or a code, or null or the refusal's
// message, which the wrapper throws; see each call for what a refusal leaves changed.
#pragma once
#include <cstddef>
#include <cstdint>
#include "../kvman-rs/g6kvman.h"

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
struct g6sch_identity { bool enabled; uint8_t profile_digest[32]; uint8_t scope; uint8_t schedule_digest[32]; uint32_t scale_bits, slot;
                        bool images; uint8_t image_digest[32]; };
struct g6sch_request { uint64_t id; uint32_t prompt_tokens, max_tokens; g6sch_identity steering; uint32_t r_rows; };
struct g6sch_run { uint32_t first_page, pages, table_version; uint64_t list_hash; };
struct g6sch_slot {
    g6sch_key key; g6sch_request req; g6sch_run run;
    uint32_t phase, prefilled, emitted, inserted, deferred;
    bool packable, pack_tail;
    uint64_t parked;
};
// scheduler.h Config's scalar settings (its steering receipt stays C++). Config sets the defaults.
struct g6sch_config {
    uint64_t epoch;
    uint32_t pool_rows, max_rows, max_sessions, mtp;
    uint32_t guard_tokens;   // UINT32_MAX: 2 * mtp (guard())
    uint32_t prefill_rows;
    bool two_lanes;
    // While any session decodes, a prefill quantum takes at most this many rows on one lane, so a decode round comes
    // round sooner (each round waits out one prefill quantum). 0 = off: the full prefill_rows x lanes quantum. A multiple
    // of PAGE, at most prefill_rows (native.cuh prefill: lane A takes up to prefill_rows rows).
    uint32_t decode_prefill_rows;
    // While a prefill waits and sessions decode, this many decode rounds run per prefill quantum, so the decoders get K
    // rounds per quantum's wait instead of one (Scheduler::alternate). 0 or 1 = off: one round per quantum.
    uint32_t decode_rounds;
    // remote rows release (pool.h window, kvman-rs Pool::window): a run past R lists its worst case and this many window
    // pages besides. 0 = off
    uint32_t window_pages;
    // GR8_CP_STEP_ROWS (cp_rows.h): a step past R carries at most cp_step_rows rows (0 = off: every decoding session in
    // every round). cp_after: contexts above it run the CP block (cp_after()). cp_seg: GR8_CP_STEP_SEG. Read by
    // Driver::step only; the scheduler's quanta keep their kinds (a capped round is a Decode quantum with fewer sessions)
    uint32_t cp_step_rows;
    uint64_t cp_after;
    bool cp_seg;
    // R per grant: the smaller R a run may take (Request::r_rows; 0 = none). Steps switch to the CP block at one context
    // for every session (cp_after), whichever R it takes
    uint32_t r2_rows;
    // GR8_S2_TAIL_FIRST=1: a session whose prompt is prefilled (only its tail left) is served before any prefill, owed or
    // not, and its tail does not ask for a decode round (Scheduler::alternate), so a burst of short prompts reaches its
    // first tokens one tail per step instead of after every prefill and a decode round per tail.
    bool tail_first;
    // GR8_S2_NOTICE_MAX=<k>: the notice target's cap, NOTICE_MIN..NOTICE_MAX boundaries (horizon(), service.h event).
    // Lower only from a measured margin: a notice that lands after its boundary is fatal
    uint32_t notice_max;
    // DS41 only: short fresh prompts prefill together in one ragged pass, then share one prompt-tail step
    bool c8_pack;
    // 0: every session drafts mtp tokens a round. n: drafting stops while more than n sessions are active (one row each),
    // so up to max_rows sessions decode. Batch::set_mtp_sessions.
    uint32_t mtp_sessions;
    // DS41 only, GR8_DS_SESS_DRAFTS=1: a shared round's drafts a session go by its active count (s2/ds41_sess_drafts.h,
    // Batch::set_sess_drafts) instead of the mtp_sessions cut; needs mtp_sessions > 0
    bool sess_drafts;
    // DS41 only, GR8_DS_PARK=1: a finished session's slot is parked (Phase::Parked) instead of freed: its KV stays in the
    // slot's slabs, so a follow-up whose prompt extends it is granted that slot and prefills only the new rows. A new chat
    // takes a free slot below limit() first, else the oldest parked slot (Scheduler::next_key)
    bool park;
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
// the Rust side's constants, which the C++ wrapper checks once (scheduler.h check_abi) with the layout digest below
enum {
    G6SCH_ABI_PAGE, G6SCH_ABI_PREFILL_DEFER_MAX, G6SCH_ABI_NOTICE_MIN, G6SCH_ABI_NOTICE_MAX, G6SCH_ABI_NOTICE_ROWS,
    G6SCH_ABI_MAX_DECODE_ROUNDS, G6SCH_ABI_PACK_MAX_MEMBERS, G6SCH_ABI_QMAX,
};
uint64_t g6sch_abi(uint32_t what);

// the guard rows (guard_tokens, or 2 * mtp) and the most sessions at once: max_sessions, at most max_rows / the rows one
// session's round takes (one with mtp_sessions, else mtp + 1); max_rows bounds ONE step (s2_batch.h Batch::round)
uint32_t g6sch_guard(const g6sch_config* c);
uint32_t g6sch_limit(const g6sch_config* c);
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

// The layout digest of every type above (and g6kvman.h's g6kv_tier_totals), computed the way src/layout.rs computes
// it over the Rust types: per type its size and alignment, per field its offset, size and kind. The C++ compares it
// with g6sch_layout() once at startup (scheduler.h check_abi).
uint64_t g6sch_layout(void);
}
#include <type_traits>
namespace g6sch_abi_detail {
template <class T> constexpr uint64_t kind() {
    if constexpr (std::is_pointer_v<T>) return 0x70 | kind<std::remove_cv_t<std::remove_pointer_t<T>>>();
    else if constexpr (std::is_array_v<T>) return 0x50 | kind<std::remove_extent_t<T>>();
    else if constexpr (std::is_same_v<T, bool>) return 4;
    else if constexpr (std::is_floating_point_v<T>) return 3;
    else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) return 2;
    else if constexpr (std::is_integral_v<T>) return 1;
    else return 6;
}
template <class T> constexpr uint64_t psize() {
    if constexpr (std::is_pointer_v<T>) return sizeof(std::remove_pointer_t<T>);
    else return 0;
}
constexpr uint64_t mix(uint64_t h, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
    return h;
}
} // namespace g6sch_abi_detail
#define G6SCH_TYPE(T) h = g6sch_abi_detail::mix(g6sch_abi_detail::mix(h, sizeof(T)), alignof(T));
#define G6SCH_FIELD(T, f) \
    h = g6sch_abi_detail::mix(g6sch_abi_detail::mix(g6sch_abi_detail::mix(h, offsetof(T, f)), sizeof(decltype(T::f))), \
                              g6sch_abi_detail::kind<decltype(T::f)>());                                   \
    h = g6sch_abi_detail::mix(h, g6sch_abi_detail::psize<decltype(T::f)>());
constexpr uint64_t g6sch_layout_cpp() {
    uint64_t h = 14695981039346656037ull;
    G6SCH_TYPE(g6sch_status) G6SCH_FIELD(g6sch_status, why) G6SCH_FIELD(g6sch_status, requested) G6SCH_FIELD(g6sch_status, room)
    G6SCH_TYPE(g6sch_cand) G6SCH_FIELD(g6sch_cand, id) G6SCH_FIELD(g6sch_cand, past)
    G6SCH_TYPE(g6sch_batch) G6SCH_FIELD(g6sch_batch, kind) G6SCH_FIELD(g6sch_batch, items) G6SCH_FIELD(g6sch_batch, items_cap) G6SCH_FIELD(g6sch_batch, n_items) G6SCH_FIELD(g6sch_batch, from) G6SCH_FIELD(g6sch_batch, to) G6SCH_FIELD(g6sch_batch, slots) G6SCH_FIELD(g6sch_batch, to_slots) G6SCH_FIELD(g6sch_batch, slots_cap) G6SCH_FIELD(g6sch_batch, n_slots)
    G6SCH_TYPE(g6sch_ints) G6SCH_FIELD(g6sch_ints, p) G6SCH_FIELD(g6sch_ints, n)
    G6SCH_TYPE(g6sch_parsed_req) G6SCH_FIELD(g6sch_parsed_req, prompt) G6SCH_FIELD(g6sch_parsed_req, stop) G6SCH_FIELD(g6sch_parsed_req, n_gen) G6SCH_FIELD(g6sch_parsed_req, min_tokens) G6SCH_FIELD(g6sch_parsed_req, temp) G6SCH_FIELD(g6sch_parsed_req, top_p) G6SCH_FIELD(g6sch_parsed_req, seed) G6SCH_FIELD(g6sch_parsed_req, lp) G6SCH_FIELD(g6sch_parsed_req, plp) G6SCH_FIELD(g6sch_parsed_req, sys) G6SCH_FIELD(g6sch_parsed_req, steering) G6SCH_FIELD(g6sch_parsed_req, ins_after) G6SCH_FIELD(g6sch_parsed_req, ins_unless) G6SCH_FIELD(g6sch_parsed_req, ins_tokens) G6SCH_FIELD(g6sch_parsed_req, gram_kind) G6SCH_FIELD(g6sch_parsed_req, gram_strict) G6SCH_FIELD(g6sch_parsed_req, gram_arm) G6SCH_FIELD(g6sch_parsed_req, schema) G6SCH_FIELD(g6sch_parsed_req, schema_n)
    G6SCH_TYPE(g6sch_lend) G6SCH_FIELD(g6sch_lend, idle_ms) G6SCH_FIELD(g6sch_lend, keep) G6SCH_FIELD(g6sch_lend, chunk_rows) G6SCH_FIELD(g6sch_lend, chunks) G6SCH_FIELD(g6sch_lend, idle) G6SCH_FIELD(g6sch_lend, lent) G6SCH_FIELD(g6sch_lend, since_ns)
    G6SCH_TYPE(g6sch_ring) G6SCH_FIELD(g6sch_ring, first) G6SCH_FIELD(g6sch_ring, rows) G6SCH_FIELD(g6sch_ring, ring)
    G6SCH_TYPE(g6sch_io) G6SCH_FIELD(g6sch_io, kind) G6SCH_FIELD(g6sch_io, page) G6SCH_FIELD(g6sch_io, tier) G6SCH_FIELD(g6sch_io, to) G6SCH_FIELD(g6sch_io, slot) G6SCH_FIELD(g6sch_io, to_slot) G6SCH_FIELD(g6sch_io, part) G6SCH_FIELD(g6sch_io, sub)
    G6SCH_TYPE(g6sch_item) G6SCH_FIELD(g6sch_item, page) G6SCH_FIELD(g6sch_item, part) G6SCH_FIELD(g6sch_item, sub) G6SCH_FIELD(g6sch_item, hslot)
    G6SCH_TYPE(g6kv_tier_totals) G6SCH_FIELD(g6kv_tier_totals, slots) G6SCH_FIELD(g6kv_tier_totals, free_slots) G6SCH_FIELD(g6kv_tier_totals, entries) G6SCH_FIELD(g6kv_tier_totals, shared) G6SCH_FIELD(g6kv_tier_totals, spilled_pages) G6SCH_FIELD(g6kv_tier_totals, moved_in_pages) G6SCH_FIELD(g6kv_tier_totals, restored_pages) G6SCH_FIELD(g6kv_tier_totals, hits) G6SCH_FIELD(g6kv_tier_totals, dropped) G6SCH_FIELD(g6kv_tier_totals, dropped_pages) G6SCH_FIELD(g6kv_tier_totals, moved_out) G6SCH_FIELD(g6kv_tier_totals, moved_out_pages) G6SCH_FIELD(g6kv_tier_totals, refused) G6SCH_FIELD(g6kv_tier_totals, refused_pages) G6SCH_FIELD(g6kv_tier_totals, full) G6SCH_FIELD(g6kv_tier_totals, block_bytes) G6SCH_FIELD(g6kv_tier_totals, credit) G6SCH_FIELD(g6kv_tier_totals, oversize) G6SCH_FIELD(g6kv_tier_totals, oversize_last)
    G6SCH_TYPE(g6sch_resize) G6SCH_FIELD(g6sch_resize, tier) G6SCH_FIELD(g6sch_resize, segs) G6SCH_FIELD(g6sch_resize, slots) G6SCH_FIELD(g6sch_resize, txn)
    G6SCH_TYPE(g6sch_key) G6SCH_FIELD(g6sch_key, epoch) G6SCH_FIELD(g6sch_key, slot) G6SCH_FIELD(g6sch_key, generation)
    G6SCH_TYPE(g6sch_identity) G6SCH_FIELD(g6sch_identity, enabled) G6SCH_FIELD(g6sch_identity, profile_digest) G6SCH_FIELD(g6sch_identity, scope) G6SCH_FIELD(g6sch_identity, schedule_digest) G6SCH_FIELD(g6sch_identity, scale_bits) G6SCH_FIELD(g6sch_identity, slot) G6SCH_FIELD(g6sch_identity, images) G6SCH_FIELD(g6sch_identity, image_digest)
    G6SCH_TYPE(g6sch_request) G6SCH_FIELD(g6sch_request, id) G6SCH_FIELD(g6sch_request, prompt_tokens) G6SCH_FIELD(g6sch_request, max_tokens) G6SCH_FIELD(g6sch_request, steering) G6SCH_FIELD(g6sch_request, r_rows)
    G6SCH_TYPE(g6sch_run) G6SCH_FIELD(g6sch_run, first_page) G6SCH_FIELD(g6sch_run, pages) G6SCH_FIELD(g6sch_run, table_version) G6SCH_FIELD(g6sch_run, list_hash)
    G6SCH_TYPE(g6sch_slot) G6SCH_FIELD(g6sch_slot, key) G6SCH_FIELD(g6sch_slot, req) G6SCH_FIELD(g6sch_slot, run) G6SCH_FIELD(g6sch_slot, phase) G6SCH_FIELD(g6sch_slot, prefilled) G6SCH_FIELD(g6sch_slot, emitted) G6SCH_FIELD(g6sch_slot, inserted) G6SCH_FIELD(g6sch_slot, deferred) G6SCH_FIELD(g6sch_slot, packable) G6SCH_FIELD(g6sch_slot, pack_tail) G6SCH_FIELD(g6sch_slot, parked)
    G6SCH_TYPE(g6sch_config) G6SCH_FIELD(g6sch_config, epoch) G6SCH_FIELD(g6sch_config, pool_rows) G6SCH_FIELD(g6sch_config, max_rows) G6SCH_FIELD(g6sch_config, max_sessions) G6SCH_FIELD(g6sch_config, mtp) G6SCH_FIELD(g6sch_config, guard_tokens) G6SCH_FIELD(g6sch_config, prefill_rows) G6SCH_FIELD(g6sch_config, two_lanes) G6SCH_FIELD(g6sch_config, decode_prefill_rows) G6SCH_FIELD(g6sch_config, decode_rounds) G6SCH_FIELD(g6sch_config, window_pages) G6SCH_FIELD(g6sch_config, cp_step_rows) G6SCH_FIELD(g6sch_config, cp_after) G6SCH_FIELD(g6sch_config, cp_seg) G6SCH_FIELD(g6sch_config, r2_rows) G6SCH_FIELD(g6sch_config, tail_first) G6SCH_FIELD(g6sch_config, notice_max) G6SCH_FIELD(g6sch_config, c8_pack) G6SCH_FIELD(g6sch_config, mtp_sessions) G6SCH_FIELD(g6sch_config, sess_drafts) G6SCH_FIELD(g6sch_config, park)
    G6SCH_TYPE(g6sch_quantum) G6SCH_FIELD(g6sch_quantum, kind) G6SCH_FIELD(g6sch_quantum, boundary) G6SCH_FIELD(g6sch_quantum, keys) G6SCH_FIELD(g6sch_quantum, count) G6SCH_FIELD(g6sch_quantum, first) G6SCH_FIELD(g6sch_quantum, rows) G6SCH_FIELD(g6sch_quantum, firsts) G6SCH_FIELD(g6sch_quantum, row_counts)
    G6SCH_TYPE(g6sch_state) G6SCH_FIELD(g6sch_state, boundary) G6SCH_FIELD(g6sch_state, event) G6SCH_FIELD(g6sch_state, revision) G6SCH_FIELD(g6sch_state, last_id) G6SCH_FIELD(g6sch_state, pool) G6SCH_FIELD(g6sch_state, have_id) G6SCH_FIELD(g6sch_state, want_decode) G6SCH_FIELD(g6sch_state, busy) G6SCH_FIELD(g6sch_state, failed) G6SCH_FIELD(g6sch_state, park_seq) G6SCH_FIELD(g6sch_state, decode_run) G6SCH_FIELD(g6sch_state, work)
    G6SCH_TYPE(g6sch_grant) G6SCH_FIELD(g6sch_grant, key) G6SCH_FIELD(g6sch_grant, req) G6SCH_FIELD(g6sch_grant, run) G6SCH_FIELD(g6sch_grant, reuse) G6SCH_FIELD(g6sch_grant, restored) G6SCH_FIELD(g6sch_grant, counted)
    G6SCH_TYPE(g6sch_shrink) G6SCH_FIELD(g6sch_shrink, key) G6SCH_FIELD(g6sch_shrink, r_rows) G6SCH_FIELD(g6sch_shrink, run)
    G6SCH_TYPE(g6sch_commit) G6SCH_FIELD(g6sch_commit, epoch) G6SCH_FIELD(g6sch_commit, event) G6SCH_FIELD(g6sch_commit, boundary) G6SCH_FIELD(g6sch_commit, before_hash) G6SCH_FIELD(g6sch_commit, has_cancel) G6SCH_FIELD(g6sch_commit, cancel) G6SCH_FIELD(g6sch_commit, has_admit) G6SCH_FIELD(g6sch_commit, admit) G6SCH_FIELD(g6sch_commit, has_shrink) G6SCH_FIELD(g6sch_commit, shrink)
    G6SCH_TYPE(g6sch_completion) G6SCH_FIELD(g6sch_completion, key) G6SCH_FIELD(g6sch_completion, emitted) G6SCH_FIELD(g6sch_completion, finished) G6SCH_FIELD(g6sch_completion, inserted)
    return h;
}
#undef G6SCH_TYPE
#undef G6SCH_FIELD
