// The C-ABI functions of gr8's host-side S2 scheduling logic in Rust (for C++ callers: bool fields, <cstdint>) (engine/scheduler-rs, docs/HOST-CORE.md R2). Built as
// libg6host.a (engine/Cargo.toml); g6m4, m4sim and the host tests link it. The C++ headers in engine/runtime/s2 keep their
// types and call these; the structs here have the same layout as theirs (static_asserts in each wrapper) and as the
// Rust's (the layout digest at the end, checked at startup). Fallible calls return 0 or a code, or null or the refusal's
// message, which the wrapper throws; see each call for what a refusal leaves changed.
#pragma once
#include <cstddef>
#include <cstdint>
#include "../kvman-rs/g6kvman.h"
#include "../module_info.h"
#include "g6plan.h"

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
    // GR8_S2_NOTICE_MAX=<k>: the notice target's cap, NOTICE_MIN..NOTICE_MAX boundaries (horizon(), control.rs event).
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
    G6SCH_ABI_MAX_DECODE_ROUNDS, G6SCH_ABI_PACK_MAX_MEMBERS, G6SCH_ABI_QMAX, G6SCH_ABI_PARSE_FN, G6SCH_ABI_CONTROL_VERSION,
    G6SCH_ABI_PROFILES_MAX, G6SCH_ABI_SCALE_MAX_BITS,
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
// share_apply: every rank's change at a Resize commit, on the pool and the tier's store. The pool is locked only for
// the tier's resize, never around run_io (which runs the pool's copies and calls io_done; nonzero = it failed). 0
// applied, 1 the lease or tier keeps its size, 2 a store fault (g6kv_tier_error), 3 a pool error or a size the event
// did not name (*msg), 4 run_io failed
int32_t g6s2_share_apply(g6kv_pool* pool, const g6kv_tier* store, const g6sch_resize* r, int32_t (*run_io)(void* ctx),
                         void* ctx, const char** msg);

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
    float presence, frequency;   // penalties (runtime/penalty.h); 0 = off
    float tpresence, tfrequency; // added inside the think block, up to the think_end token
    int32_t think_end;           // -1: the prompt left no think block open
    int32_t boost_start;         // the think-end boost (runtime/penalty.h pen_boost); per1k 0 = off
    float boost_per1k, boost_max;
    float seq_mult[3], seq_base[3];   // the sequence penalty's scopes (penalty.h PEN_SEQ: thinking, answer, guard)
    int32_t seq_min[3];
    float seq_cap[3];
    g6sch_ints avoid;            // the guard scope's ids
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
// the caller's `img` field (DS41, ds41_image.h): null when kept, or its refusal (valid until the caller's next call).
// Called once per field, in order. It must not throw. resolve and image share ctx.
typedef const char* (*g6sch_image_fn)(void* ctx, const uint8_t* field, size_t field_n);
// The version of g6sch_parse_request's parameters and callback types, which no compiler checks across the C ABI:
// scheduler.h check_abi requires g6sch_abi(G6SCH_ABI_PARSE_FN) == G6SCH_PARSE_FN. Change any of them -> bump it here and
// PARSE_FN in src/ffi.rs. 2: the image callback.
#define G6SCH_PARSE_FN 2
const char* g6sch_parse_request(const uint8_t* bytes, size_t n, bool grammar_ready, g6sch_resolve_fn resolve,
                                g6sch_image_fn image, void* ctx, g6sch_parsed_req* out, g6sch_parsed** handle);
void g6sch_parsed_free(g6sch_parsed* handle);
// one `gram` field; *handle owns the schema bytes (g6sch_parsed_free)
const char* g6sch_parse_grammar(const uint8_t* s, size_t n, uint32_t* kind, bool* strict, int32_t* arm, const uint8_t** schema,
                                size_t* schema_n, g6sch_parsed** handle);

// s2_batch.h: the batch planner (docs/HOST-CORE.md R3b). Hooks must not call back into the planner. Rust owns its state
// behind g6s2_batch. The C++ s2::Batch is a facade: it passes settings in as the plain structs below, answers the hooks
// through the function pointers, and reads results and the learned state back. Calls that run steps return 0, 1 when a
// hook returned nonzero (the facade holds its exception), or 2 when the planner refused (g6s2_batch_error). A failed
// call leaves the state as far as it got, as the C++ planner's exceptions did.
struct g6s2_row { int32_t sess, tok, pos, spos, tgt; };
// a step as the run hook sees it, its arrays valid during the call (s2_batch.h Step less scheduler_decode)
struct g6s2_step {
    int32_t kind, event, cls, round_active;
    int64_t round;
    int32_t n_sess, n_rows;
    const int32_t *sess, *row0, *rows, *ctx, *ev;
    const g6s2_row* r;
};
struct g6s2_dfl_req { int32_t sess, rows, pos0, tok0, lo; bool query; int32_t phase; };
struct g6s2_dfl_drafts { int32_t tok[8]; float lp[8]; };
// each returns 0, or nonzero when the C++ hook threw. has_*: the C++ Hooks member is set
struct g6s2_hooks {
    void* ctx;
    int32_t (*run)(void* ctx, const g6s2_step* s, int32_t* out);          // out: n_rows tokens
    int32_t (*round)(void* ctx, int32_t q, int64_t n, int32_t base, int32_t a, int32_t k);
    int32_t (*prompt)(void* ctx, int32_t q, int32_t m, int32_t pos0);
    int32_t (*emit)(void* ctx, int32_t q, const int32_t* t, int32_t k, int32_t row, bool* done);
    int32_t (*dfl)(void* ctx, const g6s2_dfl_req* r, g6s2_dfl_drafts* out);
    int32_t (*dfl_start)(void* ctx, int32_t q, int32_t* lo);
    int32_t (*lp)(void* ctx, int32_t row, float* out);
    bool has_round, has_prompt, has_emit, has_dfl, has_dfl_start, has_lp;
};
// spec_len.h SpecLen, its levels inline (at most 16)
struct g6s2_spec_len {
    bool on;
    int32_t n_levels, levels[16];
    double draft_round, draft_each, table[9], seed[8], pn[8], pk[8], bn[8][8], bk[8][8], pos_decay, bucket_decay, prior;
    int64_t rounds, chosen[8];
    double pred_sum, got_sum;
};
struct g6s2_dsel_table { double v[9]; double d, m, e, p, g, k, n, w, r; };
struct g6s2_dsel {
    bool on;
    int32_t dwell;
    double margin;
    int32_t probe, model;
    g6s2_spec_len d, m;
    g6s2_dsel_table tab;
    double rn, rc, an, ag, prior, dn, dt, dc;
};
// KStop; KStopB's tau as b_tau_rows rows of 8, its widths by pointer
struct g6s2_kstop {
    int32_t mode, kmax;
    double tau[4], seed3;
    bool log, solo;
    int32_t b_mode, b_kmin, b_explore, b_tau_rows;
    const double* b_tau;
    const int32_t* b_widths;
    int32_t b_n_widths;
};
// NgramSrc and, when table, the table's entries (ngram_lookup.h parses it)
struct g6s2_ngram {
    bool table;
    int32_t n_ns;
    const int32_t* ns;
    bool has_cal;
    double cal[8];
    size_t n;
    const uint64_t* keys;
    const int32_t* toks;
    const double* ps;
    int32_t kmax, kmin;
    double margin;
    bool log;
};
struct g6s2_session {
    const int32_t* prompt;
    size_t n_prompt;
    int32_t first, chunk, n_gen;
    bool dflash;
    int32_t ins_after, ins_unless;
    const int32_t* ins_tokens;
    size_t n_ins;
};
// S2Cost, with Flash's solo table when flash (spec_len.h flash_cost)
struct g6s2_cost {
    double main0, main_row, draft0, draft_sess, chain0, chain_sess;
    bool flash;
    double flash_table[9], flash_draft_round, flash_draft_each;
};
// Stats. acc is valid until the planner's next call
struct g6s2_stats {
    int64_t rounds, tokens, drafted, mtp_rounds, switches, inserted, ngram_rounds, ngram_drafted, ngram_acc, dfl_rounds;
    int32_t dfl_from, dfl_lo, dfl_warm, promoted;
    const int64_t* acc;
    size_t n_acc;
};
// one slot's session and draft state, for the round log's learn record. The arrays are valid until the next call
struct g6s2_slot {
    int32_t first, chunk, n_gen;
    bool dflash;
    int32_t ins_after, ins_unless;
    size_t n_ins;
    const int32_t* ins_tokens;
    size_t n_tokens;
    const int32_t* tokens;
    size_t n_d;
    const int32_t* d;
    size_t n_dlm;
    const float* dlm;
    int32_t n;
    bool on, done;
    int32_t rounds, k;
    bool dfl, mtp;
    int32_t pb, pd[8];
    float pl[8];
    int32_t since, mn, lo, warm;
    bool armed;
    int32_t force, dd[8];
    float dl[8];
};
// the drafter's ring and the round counters
struct g6s2_learn {
    int64_t round_no, cur_round;
    int32_t cur_active, ring_q, ring_lo, ring_end, ring_top;
    size_t n_ring_tok;
    const int32_t* ring_tok;
};
// GR8_DS_SESS_DRAFTS's rule (s2/ds41_sess_drafts.h), the n-gram table's key (ngram_lookup.h), and a new SpecLen and its
// confidence bucket (for the check against spec_len.h): one implementation each, the planner's
int32_t g6s2_sess_drafts_kv(int32_t K, int32_t max_rows, int32_t n);
int32_t g6s2_sess_drafts_max(int32_t K, int32_t max_rows, int32_t slots);
uint64_t g6s2_ngram_key(const int32_t* ctx, int32_t n);
void g6s2_spec_len_default(g6s2_spec_len* out);
int32_t g6s2_spec_len_bucket(float lp);
struct g6s2_batch;
// null when slots is above 32
g6s2_batch* g6s2_batch_new(int32_t K, int32_t max_rows, int32_t slots, bool par, bool fresh_spec, const g6s2_cost* cost,
                           const g6s2_hooks* hooks);
void g6s2_batch_free(g6s2_batch* b);
const char* g6s2_batch_error(const g6s2_batch* b);
int32_t g6s2_batch_admit(g6s2_batch* b, const int32_t* qs, size_t n, const g6s2_session* ss, size_t n_ss);
int32_t g6s2_batch_admit_packed_tail(g6s2_batch* b, const int32_t* qs, size_t n, const g6s2_session* ss, size_t n_ss);
int32_t g6s2_batch_round(g6s2_batch* b, const int32_t* order, size_t n, bool* ran);
int32_t g6s2_batch_demote(g6s2_batch* b);
// a q out of range: 2
int32_t g6s2_batch_promote(g6s2_batch* b, int32_t q);
int32_t g6s2_batch_cancel(g6s2_batch* b, int32_t q);
int32_t g6s2_batch_release(g6s2_batch* b, int32_t q);
void g6s2_batch_set_mtp_sessions(g6s2_batch* b, int32_t n);
void g6s2_batch_set_sess_drafts(g6s2_batch* b, bool on);
void g6s2_batch_adapt(g6s2_batch* b, bool on);
void g6s2_batch_draft_par(g6s2_batch* b, bool on);
void g6s2_batch_fresh_spec(g6s2_batch* b, bool on);
void g6s2_batch_dflash(g6s2_batch* b, int32_t kv, const g6s2_spec_len* sl);
void g6s2_batch_dsel(g6s2_batch* b, const g6s2_dsel* d);
void g6s2_batch_kstop(g6s2_batch* b, const g6s2_kstop* k);
void g6s2_batch_dfl_reuse(g6s2_batch* b, int32_t mode, int32_t min);
int32_t g6s2_batch_ngram(g6s2_batch* b, const g6s2_ngram* n);   // 2: kmin / kmax refused
int32_t g6s2_batch_mtp_max_sess(const g6s2_batch* b);
// a q out of range reads as an empty slot: false, 0, no tokens, a zeroed struct
bool g6s2_batch_active(const g6s2_batch* b, int32_t q);
bool g6s2_batch_finished(const g6s2_batch* b, int32_t q);
int32_t g6s2_batch_length(const g6s2_batch* b, int32_t q);
void g6s2_batch_result(const g6s2_batch* b, int32_t q, const int32_t** p, size_t* n);   // valid until the next call
void g6s2_batch_stats(const g6s2_batch* b, int32_t q, g6s2_stats* out);
void g6s2_batch_state(const g6s2_batch* b, uint64_t* out);   // the round log's state record: 6 values a slot
void g6s2_batch_dsel_state(const g6s2_batch* b, g6s2_dsel* out);
void g6s2_batch_dsel_rates(const g6s2_batch* b, double* rm, double* rd);
// an estimate: 0 = the DFlash2 rounds' (dflash), 1 / 2 = the n-gram choice's DFlash2 / MTP, 3 + q = slot q's
void g6s2_batch_spec_len(const g6s2_batch* b, int32_t which, g6s2_spec_len* out);
void g6s2_batch_slot(const g6s2_batch* b, int32_t q, g6s2_slot* out);
void g6s2_batch_learn(const g6s2_batch* b, g6s2_learn* out);

// s2/driver.h: the driver. Rust owns the scheduler's state (config, slots, counters), the per-slot emitted counts,
// initialised flags, what it reads of each payload and the capped-round cycle behind g6s2_driver. The C++ Driver is a
// facade: it reads the scheduler's state through the view, answers the engine's hooks and the planner's calls through
// the tables below, and keeps the payloads. Hooks may read the driver back. Calls return 0, 1 when a hook returned
// nonzero (the facade holds its exception), or 2 when refused (g6s2_driver_error); after the scheduler changed, either
// fails it.
// the scheduler's n slots and its counters, where the driver keeps them for its life
struct g6s2_drv_view { const g6sch_slot* s; size_t n; const g6sch_state* st; };
struct g6s2_drv_payload { size_t prompt_len; int32_t chunk; bool decode_prompt; bool steering_ok; };
// the pool calls the driver makes itself on the pool passed to g6s2_driver_new (each locks it for that call, with no
// hook running): grow, runs before a round (then the grown hook); shrink, a worker's half of a shrink commit; release,
// a leaving session's run (no prefix reuse); head, this rank made the commit's cancel and shrink calls itself
struct g6s2_drv_pool { bool grow, shrink, release, head; };
// the engine's callbacks; each returns 0, or nonzero when the C++ hook threw. has_*: the C++ member is set. payload:
// the facade copies apply's payload into the slot (set) or clears it. prefill and prefill_pack find the payload by
// slot. grow decides growth itself; with g6s2_drv_pool::grow, grown follows a run that grew (the slot as it was)
struct g6s2_drv_hooks {
    void* ctx;
    int32_t (*bind)(void* ctx, const g6sch_grant* g);
    int32_t (*retire)(void* ctx, const g6sch_slot* s);
    int32_t (*prefill)(void* ctx, const g6sch_slot* s, uint32_t first, uint32_t rows);
    int32_t (*prefill_pack)(void* ctx, const uint32_t* slots, const uint32_t* firsts, const uint32_t* rows, uint32_t n);
    int32_t (*dfl_ok)(void* ctx, const g6sch_key* k, bool* ok);
    int32_t (*terminal)(void* ctx, const g6sch_key* k);
    int32_t (*grow)(void* ctx, const g6sch_slot* s, uint64_t rows, g6sch_run* out, bool* some);
    int32_t (*finished)(void* ctx, const g6sch_slot* s);
    int32_t (*resize)(void* ctx);
    int32_t (*shrink)(void* ctx, const g6sch_shrink* k, const g6sch_run* was, uint64_t length);
    int32_t (*settle)(void* ctx);
    int32_t (*payload)(void* ctx, uint32_t slot, bool set);
    int32_t (*grown)(void* ctx, const g6sch_slot* s, const g6sch_run* r);
    bool has_prefill_pack, has_dfl_ok, has_terminal, has_grow, has_finished, has_resize, has_shrink, has_settle, has_grown;
};
// the planner's calls, through the C++ s2::Batch (which writes the round log); admit builds the session from the slot's
// payload; inserted: Stats::inserted
struct g6s2_drv_planner {
    void* ctx;
    int32_t (*admit)(void* ctx, uint32_t slot, int32_t first, int32_t n_gen, bool dfl);
    int32_t (*admit_pack)(void* ctx, const uint32_t* slots, const int32_t* firsts, const int32_t* n_gens, uint32_t n);
    int32_t (*round)(void* ctx, const int32_t* order, uint32_t n, bool* ran);
    int32_t (*demote)(void* ctx);
    int32_t (*promote)(void* ctx, int32_t q);
    int32_t (*release)(void* ctx, int32_t q);
    int32_t (*length)(void* ctx, int32_t q, int32_t* out);
    int32_t (*finished)(void* ctx, int32_t q, bool* out);
    int32_t (*active)(void* ctx, int32_t q, bool* out);
    int32_t (*inserted)(void* ctx, int32_t q, int32_t* out);
};
struct g6s2_driver;
// a driver with n slots over c, each slot `empty` (the C++ default) before it is keyed, making the pool calls u names
// on pool (null: none; it must outlive the driver). Null when bind, retire, prefill or payload is missing, n is not in
// 1..G6SCH_QMAX, u names a call without a pool, or u.grow is set with the grow hook or without the grown hook
g6s2_driver* g6s2_driver_new(const g6s2_drv_hooks* h, const g6s2_drv_planner* p, const g6sch_config* c, size_t n,
                             const g6sch_slot* empty, g6kv_pool* pool, const g6s2_drv_pool* u);
void g6s2_driver_free(g6s2_driver* d);
const char* g6s2_driver_error(const g6s2_driver* d);
void g6s2_driver_view(const g6s2_driver* d, g6s2_drv_view* out);
// digest: the scheduler's before the commit; pay: null without an admit; freed: room for n runs
int32_t g6s2_driver_apply(const g6s2_driver* d, const g6sch_commit* e, uint64_t digest, bool resize,
                          const g6s2_drv_payload* pay, g6sch_run* freed, size_t* n_freed, bool* admitted);
int32_t g6s2_driver_step(const g6s2_driver* d, g6sch_quantum* out);
void g6s2_driver_fail(const g6s2_driver* d);
// refused from inside apply or step
int32_t g6s2_driver_idle_boundary(const g6s2_driver* d, uint64_t target);
int32_t g6s2_driver_pool_digest(const g6s2_driver* d, uint64_t v);
// the emit hook's accounting: *done = finished, or max_tokens emitted
int32_t g6s2_driver_emit(const g6s2_driver* d, uint32_t q, int32_t n, uint32_t count, bool finished, uint32_t max_tokens,
                         bool* done);
bool g6s2_driver_decoding(const g6s2_driver* d);   // the quantum in flight is a decode round
// a slot out of range reads 0 / false
uint32_t g6s2_driver_emitted(const g6s2_driver* d, uint32_t q);
bool g6s2_driver_initialized(const g6s2_driver* d, uint32_t q);
uint64_t g6s2_driver_cp_rounds(const g6s2_driver* d);

// s2/module.h: the service's calls on a model module (ModuleOps). A module fills one table. A null member is a call it
// does not have. ctx is the module's (in C++ a g6s2::ModuleCtx: it holds the exception a call threw). Each call returns
// 0, or 1 when the module threw. Calls on one module never overlap: they run on the compute thread, and on the head's
// control thread only while the compute thread waits at an event. A C caller passes the opaque types through. They are
// the C++ g6s2::Launch, Payload, std::vector<PrefillPart>, std::vector<int> (launch assigns its output there),
// g6rt::ServeReq and g6rt::PrefixReuse. ns: G6KV_NAMESPACE_BYTES bytes.
struct g6s2_launch;
struct g6s2_payload;
struct g6s2_parts;
struct g6s2_ids;
struct g6s2_serve_req;
struct g6s2_reuse;
struct g6s2_sampling;   // sessions.h Sampling, passed through
struct g6s2_lp_row { float lp; int32_t rank, n; int32_t id[20]; float tlp[20]; };  // serve.h LpRow
struct g6s2_module_ops {
    void* ctx;
    // a session, every rank: bound (its request first), grown, shrunk to the smaller R, retired. When it leaves, the
    // module keeps its run with its reuse state (retain) or parks its slot (park)
    int32_t (*bind)(void* ctx, const g6sch_grant* g);
    int32_t (*bind_request)(void* ctx, uint32_t slot, const g6s2_serve_req* r);
    int32_t (*grown)(void* ctx, uint32_t slot, const g6sch_run* r);
    int32_t (*shrink)(void* ctx, const g6sch_shrink* k, const g6sch_run* was, uint64_t length);
    int32_t (*retire)(void* ctx, const g6sch_slot* s);
    int32_t (*retain)(void* ctx, const g6sch_run* r, const g6s2_reuse* st, const uint8_t* ns);
    int32_t (*park)(void* ctx, uint32_t slot, const g6sch_run* r, const g6s2_reuse* st, const uint8_t* ns);
    // events and boundaries
    int32_t (*fence)(void* ctx);
    int32_t (*before_work)(void* ctx);
    int32_t (*drained)(void* ctx, bool* out);
    int32_t (*resize)(void* ctx, const g6sch_resize* r);
    int32_t (*flash_room)(void* ctx, uint32_t* out);
    // steps (s2/dispatch.h)
    int32_t (*bucket)(void* ctx, bool draft, int32_t context, int32_t* out);
    int32_t (*mode)(void* ctx, int32_t old_solo, int32_t new_solo);
    int32_t (*launch)(void* ctx, const g6s2_launch* l, g6s2_ids* out);
    int32_t (*prefill)(void* ctx, const g6sch_slot* s, const g6s2_payload* p, uint32_t first, uint32_t rows);
    int32_t (*prefill_seq)(void* ctx, const g6sch_slot* s, const g6s2_payload* p, uint32_t first, uint32_t rows, uint64_t seq);
    int32_t (*prefill_pack)(void* ctx, const g6s2_parts* parts, uint64_t seq);
    int32_t (*logprob)(void* ctx, int32_t row, bool target, int32_t k, g6s2_lp_row* out);
    int32_t (*draft_lp)(void* ctx, int32_t row, float* out);
    int32_t (*dfl_ok)(void* ctx, int32_t slot, bool* out);
    int32_t (*dfl_start)(void* ctx, int32_t slot, int32_t* out);
    int32_t (*dfl)(void* ctx, const g6s2_dfl_req* r, const g6s2_sampling* s, g6s2_dfl_drafts* out);
    // grammars: the host side of the mask arena the main head reads (null: none)
    int32_t* gm_on;
    uint32_t* gm_bits;
    int32_t gm_words;
    // penalties: the host side of the row table and the session table the main head reads (penalty.h PenRow and
    // PenSess; null: none)
    void* pen_rows;
    void* pen_sess;
};
// s2/pool_ops.h: a module's policy over its rank's pool (PoolOps). The service makes the plain calls on `pool` itself
// (free pages, room, reserve, fits, count, stats, runs), each run held to max_run_pages. ctx, the return value and the
// null members as in g6s2_module_ops. The calls run on the head's control thread (tiers also outside the event pause,
// while no other pool call runs) and on the compute thread. prompt: n tokens (r_rows: has_prompt false without prefix
// reuse). slots: MAX_SESSIONS of them. The C++ std::string and std::vector<g6kv_tier_totals> pass through.
struct g6s2_pool_hit { int32_t skip, lcp; const char* why; uint64_t age; bool stored; uint32_t tier; int32_t slot; };  // pool.h PoolHit
struct g6s2_str;
struct g6s2_tiers;
struct g6s2_pool_ops {
    void* ctx;
    // prefix reuse: the retained run (or parked slot) a prompt can continue, and claiming it as a grant
    int32_t (*match)(void* ctx, const int32_t* prompt, size_t n, uint64_t rows, uint64_t worst, int32_t limit, const uint8_t* ns,
                     uint32_t r, g6s2_pool_hit* out);
    int32_t (*take)(void* ctx, const g6s2_pool_hit* h, uint64_t rows, uint64_t worst, uint32_t r, g6sch_run* out);
    // the pool's state, compared across ranks at every event, and each tier's counters for status.s2
    int32_t (*digest)(void* ctx, uint64_t* out);
    int32_t (*tiers)(void* ctx, g6s2_tiers* out);
    // why a stalled request does not fit (empty: no reason), the R its run takes, and a decoding session shrunk to the
    // smaller R to make room (*made false: none)
    int32_t (*refusal)(void* ctx, uint64_t worst, uint32_t r, g6s2_str* out);
    int32_t (*r_rows)(void* ctx, const g6sch_request* req, uint64_t rows, uint64_t worst, const int32_t* prompt, size_t n,
                      bool has_prompt, uint32_t* out);
    int32_t (*make_room)(void* ctx, const g6sch_slot* slots, const g6sch_request* req, uint64_t rows, uint64_t worst, bool* made,
                         g6sch_shrink* out);
    void* pool;               // the rank's g6kv_pool (void: the layout digest needs the pointee's size)
    uint32_t max_run_pages;   // the most pages a run may hold (0: no limit): fits refuses past it, room clamps to it
    bool grow;                // runs grow on demand (the driver grows them before each round)
    bool count;               // admissions count their prompts toward the block stores' prefixes (g6kv_count_namespaced)
    bool status;              // status.s2 shows the growth totals and the live runs
};

// s2/bootstrap.h, control_io.h, control_protocol.h, control_channel.h, event_gate.h: the control links between the
// ranks. Rust owns each link's socket (g6ctl_io, the C++ ControlIO's), the event gate, the head's and workers' event
// channels (g6ctl_endpoint) and the bootstrap's exchanges. The C++ classes are facades. No call starts a thread. Fallible
// calls return 0, 1 when a callback returned nonzero (the facade holds its exception), or 2 when refused, with the
// message in g6ctl_error(), kept per thread. Bytes come back through a g6ctl_grow_fn. It appends n bytes to the caller's
// buffer and returns them.
struct g6ctl_notice { uint64_t epoch, event, boundary, request, payload_hash; uint32_t kind; };
struct g6ctl_ack { g6ctl_notice notice; uint32_t rank; uint64_t state_hash; uint32_t room; };
struct g6ctl_commit { g6sch_commit c; bool hold; bool has_resize; g6sch_resize resize; };
// any frame: tag picks the fields (notice: NOTICE and COMMIT; ack: ACK; commit: COMMIT; epoch, request and bytes:
// REQUEST; epoch: QUIT; epoch, rank and seq: SYNC; epoch and seq: GO). A decoded REQUEST's bytes point into the input
struct g6ctl_frame {
    uint32_t tag; g6ctl_notice notice; g6ctl_ack ack; g6ctl_commit commit;
    uint64_t epoch, request, seq; uint32_t rank; const uint8_t* bytes; size_t n;
};
// BootstrapConfig's fields the bootstrap reads (strings up to their first NUL, as c_str()) and its agreement()
struct g6ctl_boot {
    uint32_t rank, ranks, timeout; uint16_t port;
    const uint8_t* head_ip; size_t head_ip_n; const uint8_t* directory; size_t directory_n;
    const uint8_t* agreement; size_t agreement_n;
};
typedef uint8_t* (*g6ctl_grow_fn)(void* ctx, size_t n);
typedef int32_t (*g6ctl_digest_fn)(void* ctx, uint64_t* digest, uint32_t* room);
typedef int32_t (*g6ctl_commit_fn)(void* ctx, uint64_t hash, g6ctl_commit* out);
typedef int32_t (*g6ctl_quit_fn)(void* ctx);
const char* g6ctl_error(void);
// the socket constants and layouts the Rust declares, which the C++ compares with its headers (control_io.h check_sys)
enum {
    G6CTL_SYS_AF_INET, G6CTL_SYS_SOCK_STREAM, G6CTL_SYS_SOL_SOCKET, G6CTL_SYS_SO_REUSEADDR, G6CTL_SYS_SO_ERROR,
    G6CTL_SYS_SO_RCVTIMEO, G6CTL_SYS_SO_SNDTIMEO, G6CTL_SYS_SO_KEEPALIVE, G6CTL_SYS_IPPROTO_TCP, G6CTL_SYS_TCP_NODELAY,
    G6CTL_SYS_TCP_KEEPIDLE, G6CTL_SYS_TCP_KEEPINTVL, G6CTL_SYS_TCP_KEEPCNT, G6CTL_SYS_TCP_USER_TIMEOUT,
    G6CTL_SYS_SEND_FLAGS, G6CTL_SYS_SOCK_CLOEXEC, G6CTL_SYS_SOCK_NONBLOCK, G6CTL_SYS_SO_NOSIGPIPE, G6CTL_SYS_O_NONBLOCK,
    G6CTL_SYS_F_GETFL, G6CTL_SYS_F_SETFL, G6CTL_SYS_F_SETFD, G6CTL_SYS_FD_CLOEXEC, G6CTL_SYS_EINTR, G6CTL_SYS_EAGAIN,
    G6CTL_SYS_EWOULDBLOCK, G6CTL_SYS_EINPROGRESS, G6CTL_SYS_ECONNREFUSED, G6CTL_SYS_POLLIN, G6CTL_SYS_POLLOUT,
    G6CTL_SYS_POLLNVAL, G6CTL_SYS_SHUT_RDWR, G6CTL_SYS_SOCKADDR_IN, G6CTL_SYS_SIN_FAMILY, G6CTL_SYS_SIN_PORT,
    G6CTL_SYS_SIN_ADDR, G6CTL_SYS_TIMEVAL, G6CTL_SYS_TV_USEC, G6CTL_SYS_POLLFD,
    G6CTL_SYS_POLLFD_EVENTS, G6CTL_SYS_POLLFD_REVENTS, G6CTL_SYS_SOCKLEN_T, G6CTL_SYS_NFDS_T, G6CTL_SYS_N
};
int64_t g6ctl_sys(uint32_t what);
// frames: encode appends the frame, decode reads b as a frame of `tag`
int32_t g6ctl_encode(const g6ctl_frame* f, g6ctl_grow_fn grow, void* ctx);
int32_t g6ctl_decode(const uint8_t* b, size_t n, uint32_t tag, g6ctl_frame* out);
int32_t g6ctl_match(const g6ctl_notice* n, const g6ctl_commit* c);
// the Encoder's and Decoder's pieces: put appends, get reads at *i and moves it on (header: the magic and the tag)
void g6ctl_put_header(uint32_t tag, g6ctl_grow_fn grow, void* ctx);
void g6ctl_put_u64(uint64_t x, g6ctl_grow_fn grow, void* ctx);
void g6ctl_put_notice(const g6ctl_notice* n, g6ctl_grow_fn grow, void* ctx);
void g6ctl_put_key(const g6sch_key* k, g6ctl_grow_fn grow, void* ctx);
void g6ctl_put_steering(const g6sch_identity* s, g6ctl_grow_fn grow, void* ctx);
int32_t g6ctl_get_header(const uint8_t* b, size_t n, size_t* i, uint32_t* tag);
int32_t g6ctl_get_u64(const uint8_t* b, size_t n, size_t* i, uint64_t* out);
int32_t g6ctl_get_u32(const uint8_t* b, size_t n, size_t* i, uint32_t* out);
int32_t g6ctl_get_notice(const uint8_t* b, size_t n, size_t* i, g6ctl_notice* out);
int32_t g6ctl_get_key(const uint8_t* b, size_t n, size_t* i, g6sch_key* out);
int32_t g6ctl_get_steering(const uint8_t* b, size_t n, size_t* i, g6sch_identity* out);
int32_t g6ctl_get_end(const uint8_t* b, size_t n, size_t i);
// one link: owns fd (null when negative) and closes it at free
struct g6ctl_io;
g6ctl_io* g6ctl_io_new(int32_t fd);
void g6ctl_io_free(g6ctl_io* io);
void g6ctl_io_shutdown(const g6ctl_io* io);
int32_t g6ctl_io_timeout(const g6ctl_io* io, uint32_t seconds);
int32_t g6ctl_io_keepalive(const g6ctl_io* io, int32_t idle_s, int32_t intvl_s, int32_t cnt);
int32_t g6ctl_io_nodelay(const g6ctl_io* io);
int32_t g6ctl_io_send(const g6ctl_io* io, const uint8_t* b, size_t n);
int32_t g6ctl_io_receive(const g6ctl_io* io, bool idle, g6ctl_grow_fn grow, void* ctx);
void g6ctl_io_counts(const g6ctl_io* io, uint64_t* sent, uint64_t* received);
// the bootstrap: out has room for ranks - 1 links (the head's, in rank order) or 1 (a worker's)
int32_t g6ctl_boot_check(const uint8_t* ip, size_t ip_n, uint16_t port, uint32_t timeout);
int32_t g6ctl_bootstrap(const g6ctl_boot* c, g6ctl_io** out, size_t* n);
int32_t g6ctl_agree_lines(uint32_t rank, const g6ctl_io* const* links, size_t n, const uint8_t* mine, size_t mine_n,
                          bool idle, g6ctl_grow_fn grow, void* ctx);
int32_t g6ctl_capture_barrier(const g6ctl_boot* c, const g6ctl_io* const* links, size_t n);
void g6ctl_agreement_difference(uint32_t rank, const uint8_t* h, size_t hn, const uint8_t* w, size_t wn,
                                g6ctl_grow_fn grow, void* ctx);
// the event gate and the head's ACK set (null when refused)
struct g6ctl_gate;
g6ctl_gate* g6ctl_gate_new(uint64_t epoch, uint32_t rank);
void g6ctl_gate_free(g6ctl_gate* g);
int32_t g6ctl_gate_before_work(const g6ctl_gate* g, uint64_t b, bool* due);
int32_t g6ctl_gate_announce(const g6ctl_gate* g, const g6ctl_notice* n);
int32_t g6ctl_gate_arrive(const g6ctl_gate* g, uint64_t b, uint64_t hash, uint32_t room, g6ctl_ack* out);
int32_t g6ctl_gate_commit(const g6ctl_gate* g, const g6ctl_notice* n);
int32_t g6ctl_gate_wait_commit(const g6ctl_gate* g);
int32_t g6ctl_gate_applied(const g6ctl_gate* g, const g6ctl_notice* n, bool hold);
void g6ctl_gate_fail(const g6ctl_gate* g);
struct g6ctl_agreement;
g6ctl_agreement* g6ctl_agreement_new(uint32_t nr, const g6ctl_notice* n);
void g6ctl_agreement_free(g6ctl_agreement* a);
int32_t g6ctl_agreement_ack(g6ctl_agreement* a, const g6ctl_ack* k);
bool g6ctl_agreement_ready(const g6ctl_agreement* a);
int32_t g6ctl_agreement_hash(const g6ctl_agreement* a, uint64_t* out);
int32_t g6ctl_agreement_room(const g6ctl_agreement* a, uint32_t* out);
// one rank's event channel over links that outlive it (null when refused). The head's calls (event, agreed_room,
// quit) and the worker's (receive_events, request, admitted) refuse on the other's channel
struct g6ctl_endpoint;
g6ctl_endpoint* g6ctl_head_new(uint64_t epoch, const g6ctl_io* const* peers, size_t n);
g6ctl_endpoint* g6ctl_worker_new(uint64_t epoch, uint32_t rank, const g6ctl_io* io);
void g6ctl_endpoint_free(g6ctl_endpoint* e);
void g6ctl_fail(const g6ctl_endpoint* e);
int32_t g6ctl_boundary(const g6ctl_endpoint* e, uint64_t b, g6ctl_digest_fn digest, void* ctx, bool* due, g6ctl_commit* out);
int32_t g6ctl_wait_notice(const g6ctl_endpoint* e, g6ctl_notice* out);
void g6ctl_stamps(const g6ctl_endpoint* e, int64_t* installed, int64_t* arrived);
int32_t g6ctl_applied(const g6ctl_endpoint* e);
int32_t g6ctl_sync(const g6ctl_endpoint* e, uint64_t seq);
int32_t g6ctl_event(const g6ctl_endpoint* e, const g6ctl_notice* n, g6ctl_commit_fn make_commit, void* ctx, bool has_payload,
                    const uint8_t* payload, size_t payload_n);
uint32_t g6ctl_agreed_room(const g6ctl_endpoint* e);
void g6ctl_quit(const g6ctl_endpoint* e);
int32_t g6ctl_receive_events(const g6ctl_endpoint* e, g6ctl_quit_fn on_quit, void* ctx);
int32_t g6ctl_request(const g6ctl_endpoint* e, uint64_t id, g6ctl_grow_fn grow, void* ctx);
void g6ctl_admitted(const g6ctl_endpoint* e, uint64_t id);

// s2/ingress.h, trace.h, bootstrap.h's files and service.h's status.s2: the S2 files and the trace. Fallible calls
// return 0, or 2 when refused, with the message in g6ctl_error(). Strings are bytes and a length.
struct g6svc_command { uint64_t epoch, request, hash; bool admit; };
struct g6svc_run_use { uint32_t table_version, held, worst; };   // kvman-rs g6kv_use
struct g6svc_profile { const uint8_t* name; size_t name_n; uint8_t digest[32]; uint32_t active_layers; };
struct g6svc_steering { const g6svc_profile* profiles; size_t n; uint32_t scope, forced_scale_bits; bool loaded, forced; };
// BootstrapConfig's fields that ready.s2 and the agreement read, and the module's limits its config is checked against
struct g6svc_boot {
    const g6sch_config* scheduler; g6svc_steering steering;
    uint32_t max_sessions, rank, ranks, prefill_min, control_version, lp_max;
    const uint8_t* directory; size_t directory_n; const uint8_t* signature; size_t signature_n;
    bool mtp0, pack, park;
};
// status.s2; the growth counters, tiers and runs are read only with their has_ flag
struct g6svc_status {
    uint64_t epoch, queued, stalls, grows, stalled;
    const g6kv_tier_totals* tiers; size_t n_tiers; const g6svc_run_use* runs; size_t n_runs;
    uint32_t pool_pages, resident, retiring, free, room, held_pages, worst_pages;
    bool has_growth, has_tiers, has_runs;
};
uint64_t g6svc_payload_hash(const uint8_t* b, size_t n);
bool g6svc_exists(const uint8_t* p, size_t n);
int32_t g6svc_read_file(const uint8_t* p, size_t n, g6ctl_grow_fn grow, void* ctx);
int32_t g6svc_atomic_file(const uint8_t* p, size_t n, const uint8_t* b, size_t bn);
int32_t g6svc_number(const uint8_t* s, size_t n, uint64_t* out);
int32_t g6svc_parse_command(const uint8_t* t, size_t n, uint64_t epoch, g6svc_command* out);
// the head's reader of DIR's cmd.<k> and req.<id>: next consumes, peek does not; *has = false when none is there yet
struct g6svc_ingress;
int32_t g6svc_ingress_new(const uint8_t* dir, size_t n, uint64_t epoch, g6svc_ingress** out);
void g6svc_ingress_free(g6svc_ingress* i);
int32_t g6svc_ingress_next(const g6svc_ingress* i, bool* has, g6svc_command* out);
int32_t g6svc_ingress_peek(const g6svc_ingress* i, bool* has, g6svc_command* out);
int32_t g6svc_ingress_request(const g6svc_ingress* i, const g6svc_command* c, g6ctl_grow_fn grow, void* ctx);
int32_t g6svc_steering_check(const g6svc_steering* s);
int32_t g6svc_capability(const g6svc_boot* b, g6ctl_grow_fn grow, void* ctx);
int32_t g6svc_agreement(const g6svc_boot* b, g6ctl_grow_fn grow, void* ctx);
// links: the head's links, missing: one of them is absent
int32_t g6svc_publish_ready(const g6svc_boot* b, size_t links, bool missing);
int32_t g6svc_status_pool(uint32_t free, uint32_t room, uint32_t pool_pages);
int32_t g6svc_status_publish(const g6svc_status* s, const uint8_t* dir, size_t n);
// GR8_S2_TRACE: PATH.r<rank>.json, null when not traced; the steps and control threads write at once
struct g6svc_trace;
g6svc_trace* g6svc_trace_open(const uint8_t* p, size_t n, uint32_t rank);
void g6svc_trace_free(g6svc_trace* t);
void g6svc_trace_span(const g6svc_trace* t, const uint8_t* name, size_t name_n, int32_t tid, uint64_t t0, uint64_t t1,
                      const uint8_t* args, size_t args_n);
void g6svc_trace_instant(const g6svc_trace* t, const uint8_t* name, size_t name_n, int32_t tid, const uint8_t* args,
                         size_t args_n);
uint64_t g6svc_trace_now(void);

// s2/admit.h: the head's admission decisions. why: g6sch_status's (0 none, 1 slots, 2 capacity); recall_note: 0 none,
// 1 begin, 2 done
uint64_t g6svc_notice_target(uint64_t boundary, uint64_t horizon, bool held, uint64_t held_at, uint32_t notice_max);
bool g6svc_admit_retry(uint64_t head, uint64_t attempted_head, uint64_t revision, uint64_t attempted_revision);
uint64_t g6svc_attempted_revision(uint64_t revision);
bool g6svc_admit_shrink(bool fits, bool can_make_room, uint32_t live, uint32_t limit);
bool g6svc_admit_grant(uint32_t why, uint32_t short_of);
bool g6svc_admit_drain(bool admit_hold, uint32_t live, uint32_t limit);
bool g6svc_admit_hold(bool admit_hold, bool queued, uint32_t live, uint32_t limit);
bool g6svc_wait_line(uint32_t why, uint64_t last, uint64_t head);
bool g6svc_refusal_line(uint32_t why, uint64_t last, uint64_t head);
uint32_t g6svc_recall_note(uint32_t why, uint32_t short_of, uint64_t waiting, uint64_t head);
bool g6svc_share_idle(uint32_t resident, size_t queued);
bool g6svc_lend_idle(uint32_t resident, size_t queued, uint32_t retiring);
// Sessions::request, and steering.h's cache namespace (G6KV_NAMESPACE_BYTES into out)
void g6svc_request(uint64_t id, size_t prompt_tokens, int32_t n_gen, const g6sch_identity* steering,
                   g6sch_request* out);
void g6svc_cache_namespace(const g6sch_identity* s, uint8_t* out);

// s2/service.h: the head's control thread (controller, event and its commit, intake, the ADMIT_HOLD drain), every
// rank's watcher (monitor) and every rank's compute thread (the event boundary, the commit's application and the step
// loop, and the driver's hooks with service logic) run in Rust; the C++ Service starts the threads and calls in. The
// Rust borrows the Service's endpoint, ingress, queue, trace, the driver's slots and counters and the rank's pool for its
// life. What stays C++ is g6svc_host. Fallible calls return 0, 1 when a host or pool-table call threw (the C++ holds the
// exception), or 2 when refused, with the message in g6ctl_error().
// a request file the host parsed and checked: ok false = refused and answered, nothing queued. prompt: valid until the
// next host call
struct g6svc_checked {
    const int32_t* prompt; size_t n; int64_t sys; int32_t plp, n_gen; g6sch_identity steering; bool ok;
};
// one plain pool call of the head's control thread or a worker's bind, for a test's observer. kind: 0 room, 1 fits, 2 reserve, 3 count; out:
// room's or count's result, or fits' as 0 or 1; run: reserve's
struct g6svc_pool_call {
    uint32_t kind; uint64_t rows, worst; uint32_t r, out; g6sch_run run; const int32_t* prompt; size_t n; int64_t span;
    const uint8_t* ns;
};
// the request the compute thread staged for an ADMIT: req, the parser's g6rt::ServeReq (void: the layout digest needs
// the pointee's size), and what a worker's bind reads of it. Valid until the next stage or unstage
struct g6svc_staged { const void* req; const int32_t* prompt; size_t n; int64_t sys; int32_t plp; };
// what the service's threads call in C++, each thread with its own ctx. Each returns 0, or 1 when the C++ threw. The
// control thread: check, retire_run, share (null without a KV share), refusal and tiers (the pool policy's, whose bytes
// and counters stay valid until the next host call). The compute thread: retire_run, the driver facade (idle_boundary;
// digest, the pool's into the driver when has_pool, then the scheduler's; apply, with the staged request's payload when
// admit; step), the staged request (stage parses it, unstage drops it), and Sessions and Dispatch: abort (a cancelled
// session's terminal lines), mode (Dispatch::synchronize_mode), counters (the "m" lines due for these keys), begin
// (Dispatch::begin), bound (Sessions::bind with the staged request) and release (Sessions::retire, Dispatch::release)
struct g6svc_host {
    void* ctx;
    int32_t (*check)(void* ctx, uint64_t id, const uint8_t* bytes, size_t n, g6svc_checked* out);
    int32_t (*retire_run)(void* ctx, const g6sch_slot* s);
    int32_t (*share)(void* ctx, bool idle, bool* has, g6sch_resize* out);
    int32_t (*refusal)(void* ctx, uint64_t worst, uint32_t r, const uint8_t** bytes, size_t* n);
    int32_t (*tiers)(void* ctx, const g6kv_tier_totals** out, size_t* n);
    int32_t (*idle_boundary)(void* ctx, uint64_t boundary);
    int32_t (*stage)(void* ctx, const uint8_t* bytes, size_t n, g6svc_staged* out);
    int32_t (*unstage)(void* ctx);
    int32_t (*digest)(void* ctx, bool has_pool, uint64_t pool, uint64_t* out);
    int32_t (*abort)(void* ctx, const g6sch_key* k);
    int32_t (*apply)(void* ctx, const g6ctl_commit* c, bool admit);
    int32_t (*mode)(void* ctx);
    int32_t (*step)(void* ctx, g6sch_quantum* out);
    int32_t (*counters)(void* ctx, const g6sch_key* keys, uint32_t n);
    int32_t (*begin)(void* ctx, const g6sch_grant* g);
    int32_t (*bound)(void* ctx, const g6sch_grant* g);
    int32_t (*release)(void* ctx, const g6sch_slot* s);
};
// the control thread's settings, and the Service's objects it borrows (void: the C++ facades' handles, g6ctl_endpoint,
// g6svc_ingress, g6sch_queue, g6svc_trace or null)
struct g6svc_control_cfg {
    const g6sch_config* scheduler; const uint8_t* directory; size_t directory_n;
    const void* endpoint; const void* ingress; void* queue; const void* trace;
    const g6sch_slot* slots; size_t n_slots; const g6sch_state* state;
    g6s2_pool_ops pool; g6svc_host host; g6sch_lend lend;
    int64_t share_ms; uint32_t rank; bool head, admit_hold, elog;
};
struct g6svc_control;
int32_t g6svc_control_new(const g6svc_control_cfg* c, g6svc_control** out);
void g6svc_control_free(g6svc_control* c);
int32_t g6svc_control_run(const g6svc_control* c);                     // the head's control thread, until stopped
int32_t g6svc_control_monitor(const g6svc_control* c, uint32_t linger_ms);   // every rank's watcher
// the compute thread: its progress after each commit and step, and a Resize commit applied
void g6svc_control_publish(const g6svc_control* c, uint64_t boundary, uint64_t revision, uint64_t horizon,
                           uint32_t resident, uint32_t retiring);
void g6svc_control_resized(const g6svc_control* c);
uint32_t g6svc_control_resident(const g6svc_control* c);
bool g6svc_control_stopping(const g6svc_control* c);
void g6svc_control_stop(const g6svc_control* c);
void g6svc_control_shutdown(const g6svc_control* c);   // stop, and fail the channel so every waiter wakes
// status.s2 from the pool's values in s (free, room, growth, tiers, runs); the service's own fields are not read
int32_t g6svc_control_write_status(const g6svc_control* c, const g6svc_status* s);
// tests: an observer of the control thread's and a worker's bind's plain pool calls, set before the threads run
void g6svc_control_observe(const g6svc_control* c, void (*f)(void* ctx, const g6svc_pool_call* call), void* ctx);
// the compute thread's settings, and the Service's objects it borrows (void: the C++ facades' handles, g6svc_control,
// g6ctl_endpoint, g6svc_ingress (the head's; a worker may pass null), g6svc_trace or null). dfl_reuse and ngram: the
// head's terminal lines for GR8_DFL_REUSE and GR8_NGRAM
struct g6svc_compute_cfg {
    const void* control; const void* endpoint; const void* ingress; const void* trace;
    const g6sch_slot* slots; size_t n_slots; const g6sch_state* state;
    g6s2_module_ops module; g6svc_host host;
    uint32_t rank; bool elog, dfl_reuse, ngram;
};
struct g6svc_compute;
int32_t g6svc_compute_new(const g6svc_compute_cfg* c, g6svc_compute** out);
void g6svc_compute_free(g6svc_compute* c);   // before the control it borrows
int32_t g6svc_compute_run(const g6svc_compute* c);   // the compute thread, until stopped
// the driver's hooks (DriverHooks) whose logic is the service's, on the compute thread inside the host's apply and step:
// a grant (a worker checks its pool calls against the head's), a slot that leaves at a commit, a session that finished
// at its step, a session's first emission (tracing only), the barrier after tier copies, and a session's drafter lines
// at its completion
int32_t g6svc_compute_bind(const g6svc_compute* c, const g6sch_grant* g);
int32_t g6svc_compute_retire(const g6svc_compute* c, const g6sch_slot* s);
int32_t g6svc_compute_finished(const g6svc_compute* c, const g6sch_slot* s);
void g6svc_compute_first_token(const g6svc_compute* c, const g6sch_key* k);
int32_t g6svc_compute_settle(const g6svc_compute* c);
void g6svc_compute_terminal(const g6svc_compute* c, uint32_t slot, const g6s2_stats* s);

// s2/signature.h: SigInputs with C strings (null reads as empty); model is SigModel (0 GLM, 1 Flash, 2 DS41). The
// signature lives until the calling thread's next call.
struct g6sig_inputs {
    uint32_t model;
    int32_t cap; const char* layers_s; int32_t chunk, nc; const char* set_id; bool dfl; int32_t dfl_k, s2sess; const char* knobs;
    int32_t mp, mtp, dfl_reuse, dfl_reuse_min, lanes, gsplit; const char* ckpt; uint64_t ckpt_blocks; bool ckpt_persist;
    int32_t s2_below; bool two; int32_t pf_own; bool pf_qp; int32_t qp_ctx, qp_min; bool grammar; const char* ngram;
    int32_t kstop_mode, kstop_kmax; double kstop_tau[4]; double kstop_seed3; int32_t draft_par, qs_mode; bool iq_split, mtp_split;
    int32_t dhead_on, dhead_tpc, dhead_ks; bool mtp_last; int32_t dhead2; bool dfl_i4, dfl_body_i4; float dfl_alpha;
    int32_t dsel_model; const char* dsel_tab; const char* kstopb; int32_t cp_step; bool cp_seg; int32_t ctx_sp, posidx;
    bool steer_shared; int32_t ds_mtp; bool sh_tc; int32_t l2hint, fc2stg, fc1bstg; bool c8_pack, sess_drafts;
    int32_t pf_lanes, tk_main, tk_decode, idx_pipe_b, idx_q_trig, idx_pipe_qr; bool reuse, vis_on; const char* vis_id;
};
const char* g6sig_runtime(const g6sig_inputs* in);

// engine_options.h: g6m4's option parse (src/options.rs) over the build's description (module_info.h) and the knob
// table's C API (gr8_knobs.h; passed in, so this library needs nothing from libgr8knobs.a at link time). The host
// table holds what stays C++: the DSpark knob parse's result (DS41) and ds41_engram_colls, which dies on a bad Engram
// config as the boot does. The options' strings and lists live until g6opt_free; g6opt_options is null after a
// refusal, and g6opt_refusal is "" when the options pass.
struct g6opt_knobs {
    uint64_t table_hash;
    int (*flag)(uint32_t, uint32_t, int*);
    int (*int_)(uint32_t, uint32_t, int64_t*);
    int (*float_)(uint32_t, uint32_t, double*);
    const char* (*text)(uint32_t, uint32_t);
    ptrdiff_t (*ints)(uint32_t, uint32_t, const int64_t**);
    ptrdiff_t (*floats)(uint32_t, uint32_t, const double**);
    const char* (*tests_set)(void);
};
struct g6opt_host {
    void* ctx;
    int32_t (*engram_colls)(void*, const int32_t*, size_t);
    int32_t dsk_mtp;
    bool dsk_relaxed;
    double dsk_relax_topp;
    const char* dsk_err;
};
struct g6opt_strs {
    const char* const* p;
    size_t n;
};
struct g6opt_plan {
    g6sch_ints prompt;
    int32_t first;
    int32_t chunk;
    int32_t n_gen;
    float inv_temp;
    uint32_t seed;
    float top_p;
    int32_t lp;
    int32_t plp;
    bool rs;
};
struct g6opt_options {
    int32_t rank;
    g6opt_strs ips;
    g6opt_strs ips2;
    bool ring;
    g6opt_strs ring_devs;
    int32_t proxy_cpu;
    g6opt_strs pf_ips2;
    const char* manifest;
    const char* packfile;
    const char* v2_dir;
    const char* v2_slices;
    const char* v2_widths;
    const char* v2_dense;
    const char* v2_layout;
    bool v2_trellis;
    const char* qo_man;
    const char* qo_file;
    const char* layers_s;
    const char* levels_s;
    g6opt_strs prompts;
    int32_t co_cap;
    int32_t co_mp;
    int32_t co_top;
    int32_t co_pfrows;
    int32_t co_mtp;
    int32_t co_lanes;
    int32_t co_nc;
    float co_temp;
    bool co_graph;
    bool co_wide;
    bool co_pfv2;
    bool co_adapt;
    bool has_n_gen;
    int32_t n_gen;
    bool has_chunk;
    int32_t chunk;
    bool has_seed;
    uint32_t seed;
    bool has_top_p;
    float top_p;
    bool has_rs;
    bool rs;
    const char* qp_man;
    const char* qp_file;
    const char* dfl_man;
    const char* dfl_file;
    int32_t dfl_sel;
    int32_t dfl_k;
    bool spec_reset_per_request;
    bool spec_fixed;
    g6opt_plan plan;
    bool image;
    bool hash;
    bool reuse_prefix;
    bool check_args;
    bool lanes_explicit;
    const char* s2dir;
    const char* grammar_dir;
    g6opt_strs kv_tiers;
    const char* kv_keep;
    int32_t kv_system_after;
    double kv_write_limit;
    const char* kv_format;
    bool kv_old;
    bool kv_old_all;
    const char* set_id;
    const char* steering_dir;
    const char* steering_force_s;
    const char* steering_model_revision;
    const char* steering_tokenizer_revision;
    const char* steering_template_digest;
    const char* steering_weight_set;
    const char* steering_template_file;
    int32_t s2port;
    int32_t s2sess;
    uint64_t s2epoch;
    int32_t cap;
    int32_t top;
    int32_t pfrows;
    int32_t nc;
    int32_t mtp;
    int32_t mp;
    int32_t lanes;
    float temp;
    bool graph;
    bool wide;
    bool pfv2;
    bool adapt;
    bool release;
    bool s2mode;
    bool dfl;
    int32_t k;
    g6sch_ints layers;
    bool progress_isolate;
    int32_t route_split_mode;
    int32_t c8_pack_env;
    bool spec_on;
    g6sch_ints spec_levels;
    double spec_draft_round;
    double spec_draft_each;
    double spec_table[9];
    int32_t rt_win;
    int32_t kstop_mode;
    int32_t kstop_kmax;
    double kstop_tau[4];
    double kstop_seed3;
    bool kstop_log;
    bool kstop_solo;
    int32_t kstopb_mode;
    int32_t kstopb_kmin;
    const double* kstopb_tau;
    size_t kstopb_tau_n;
    g6sch_ints kstopb_widths;
    int32_t kstopb_explore;
    int32_t flash_cache;
    int64_t flash_budget_mib;
    int32_t ds_moe_main_topk;
    int32_t ds_moe_decode_topk;
    int32_t ds_idx_pipe_b;
    int32_t ds_idx_q_trig;
    int32_t ds_idx_pipe_qr;
    int32_t ds_admit_keep_scratch;
    bool ds_sess_drafts;
    int32_t ds_draft_cap;
    const g6opt_plan* plans;
    size_t n_plans;
    g6sch_ints row_counts;
    g6sch_ints draft_rows;
    int32_t nl;
    int32_t nx;
    int32_t ncoll;
    int32_t r;
    bool own4;
    int32_t qp_roll_s;
    bool qp_roll_ser;
    int32_t qp_wire_e;
};
struct g6opt_parsed;
g6opt_parsed* g6opt_parse(const G6ModuleInfo* m, const g6opt_knobs* k, const g6opt_host* h, int argc, const char* const* argv);
const char* g6opt_refusal(const g6opt_parsed* p);
const g6opt_options* g6opt_options_of(const g6opt_parsed* p);
void g6opt_free(g6opt_parsed* p);

// The layout digest of every type above (and g6kvman.h's g6kv_tier_totals), computed the way src/layout.rs computes
// it over the Rust types: per type its size and alignment, per field its offset, size and kind. The C++ compares it
// with g6sch_layout() once at startup (scheduler.h check_abi).
uint64_t g6sch_layout(void);
}
#include <type_traits>
namespace g6sch_abi_detail {
template <class T> constexpr uint64_t kind() {
    if constexpr (std::is_pointer_v<T> && std::is_function_v<std::remove_pointer_t<T>>) return 0x76;   // as a struct pointer
    else if constexpr (std::is_pointer_v<T>) return 0x70 | kind<std::remove_cv_t<std::remove_pointer_t<T>>>();
    else if constexpr (std::is_array_v<T>) return 0x50 | kind<std::remove_extent_t<T>>();
    else if constexpr (std::is_same_v<T, bool>) return 4;
    else if constexpr (std::is_floating_point_v<T>) return 3;
    else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) return 2;
    else if constexpr (std::is_integral_v<T>) return 1;
    else return 6;
}
template <class T> constexpr uint64_t psize() {
    // a function has no size, and void counts as 1 byte (Rust's c_void)
    if constexpr (std::is_pointer_v<T> && std::is_function_v<std::remove_pointer_t<T>>) return 0;
    else if constexpr (std::is_pointer_v<T> && std::is_void_v<std::remove_pointer_t<T>>) return 1;
    else if constexpr (std::is_pointer_v<T>) return sizeof(std::remove_pointer_t<T>);
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
    G6SCH_TYPE(g6sch_parsed_req) G6SCH_FIELD(g6sch_parsed_req, prompt) G6SCH_FIELD(g6sch_parsed_req, stop) G6SCH_FIELD(g6sch_parsed_req, n_gen) G6SCH_FIELD(g6sch_parsed_req, min_tokens) G6SCH_FIELD(g6sch_parsed_req, temp) G6SCH_FIELD(g6sch_parsed_req, top_p) G6SCH_FIELD(g6sch_parsed_req, presence) G6SCH_FIELD(g6sch_parsed_req, frequency) G6SCH_FIELD(g6sch_parsed_req, tpresence) G6SCH_FIELD(g6sch_parsed_req, tfrequency) G6SCH_FIELD(g6sch_parsed_req, think_end) G6SCH_FIELD(g6sch_parsed_req, boost_start) G6SCH_FIELD(g6sch_parsed_req, boost_per1k) G6SCH_FIELD(g6sch_parsed_req, boost_max) G6SCH_FIELD(g6sch_parsed_req, seq_mult) G6SCH_FIELD(g6sch_parsed_req, seq_base) G6SCH_FIELD(g6sch_parsed_req, seq_min) G6SCH_FIELD(g6sch_parsed_req, seq_cap) G6SCH_FIELD(g6sch_parsed_req, avoid) G6SCH_FIELD(g6sch_parsed_req, seed) G6SCH_FIELD(g6sch_parsed_req, lp) G6SCH_FIELD(g6sch_parsed_req, plp) G6SCH_FIELD(g6sch_parsed_req, sys) G6SCH_FIELD(g6sch_parsed_req, steering) G6SCH_FIELD(g6sch_parsed_req, ins_after) G6SCH_FIELD(g6sch_parsed_req, ins_unless) G6SCH_FIELD(g6sch_parsed_req, ins_tokens) G6SCH_FIELD(g6sch_parsed_req, gram_kind) G6SCH_FIELD(g6sch_parsed_req, gram_strict) G6SCH_FIELD(g6sch_parsed_req, gram_arm) G6SCH_FIELD(g6sch_parsed_req, schema) G6SCH_FIELD(g6sch_parsed_req, schema_n)
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
    G6SCH_TYPE(g6s2_row) G6SCH_FIELD(g6s2_row, sess) G6SCH_FIELD(g6s2_row, tok) G6SCH_FIELD(g6s2_row, pos) G6SCH_FIELD(g6s2_row, spos) G6SCH_FIELD(g6s2_row, tgt)
    G6SCH_TYPE(g6s2_step) G6SCH_FIELD(g6s2_step, kind) G6SCH_FIELD(g6s2_step, event) G6SCH_FIELD(g6s2_step, cls) G6SCH_FIELD(g6s2_step, round_active) G6SCH_FIELD(g6s2_step, round) G6SCH_FIELD(g6s2_step, n_sess) G6SCH_FIELD(g6s2_step, n_rows) G6SCH_FIELD(g6s2_step, sess) G6SCH_FIELD(g6s2_step, row0) G6SCH_FIELD(g6s2_step, rows) G6SCH_FIELD(g6s2_step, ctx) G6SCH_FIELD(g6s2_step, ev) G6SCH_FIELD(g6s2_step, r)
    G6SCH_TYPE(g6s2_dfl_req) G6SCH_FIELD(g6s2_dfl_req, sess) G6SCH_FIELD(g6s2_dfl_req, rows) G6SCH_FIELD(g6s2_dfl_req, pos0) G6SCH_FIELD(g6s2_dfl_req, tok0) G6SCH_FIELD(g6s2_dfl_req, lo) G6SCH_FIELD(g6s2_dfl_req, query) G6SCH_FIELD(g6s2_dfl_req, phase)
    G6SCH_TYPE(g6s2_dfl_drafts) G6SCH_FIELD(g6s2_dfl_drafts, tok) G6SCH_FIELD(g6s2_dfl_drafts, lp)
    G6SCH_TYPE(g6s2_spec_len) G6SCH_FIELD(g6s2_spec_len, on) G6SCH_FIELD(g6s2_spec_len, n_levels) G6SCH_FIELD(g6s2_spec_len, levels) G6SCH_FIELD(g6s2_spec_len, draft_round) G6SCH_FIELD(g6s2_spec_len, draft_each) G6SCH_FIELD(g6s2_spec_len, table) G6SCH_FIELD(g6s2_spec_len, seed) G6SCH_FIELD(g6s2_spec_len, pn) G6SCH_FIELD(g6s2_spec_len, pk) G6SCH_FIELD(g6s2_spec_len, bn) G6SCH_FIELD(g6s2_spec_len, bk) G6SCH_FIELD(g6s2_spec_len, pos_decay) G6SCH_FIELD(g6s2_spec_len, bucket_decay) G6SCH_FIELD(g6s2_spec_len, prior) G6SCH_FIELD(g6s2_spec_len, rounds) G6SCH_FIELD(g6s2_spec_len, chosen) G6SCH_FIELD(g6s2_spec_len, pred_sum) G6SCH_FIELD(g6s2_spec_len, got_sum)
    G6SCH_TYPE(g6s2_dsel_table) G6SCH_FIELD(g6s2_dsel_table, v) G6SCH_FIELD(g6s2_dsel_table, d) G6SCH_FIELD(g6s2_dsel_table, m) G6SCH_FIELD(g6s2_dsel_table, e) G6SCH_FIELD(g6s2_dsel_table, p) G6SCH_FIELD(g6s2_dsel_table, g) G6SCH_FIELD(g6s2_dsel_table, k) G6SCH_FIELD(g6s2_dsel_table, n) G6SCH_FIELD(g6s2_dsel_table, w) G6SCH_FIELD(g6s2_dsel_table, r)
    G6SCH_TYPE(g6s2_dsel) G6SCH_FIELD(g6s2_dsel, on) G6SCH_FIELD(g6s2_dsel, dwell) G6SCH_FIELD(g6s2_dsel, margin) G6SCH_FIELD(g6s2_dsel, probe) G6SCH_FIELD(g6s2_dsel, model) G6SCH_FIELD(g6s2_dsel, d) G6SCH_FIELD(g6s2_dsel, m) G6SCH_FIELD(g6s2_dsel, tab) G6SCH_FIELD(g6s2_dsel, rn) G6SCH_FIELD(g6s2_dsel, rc) G6SCH_FIELD(g6s2_dsel, an) G6SCH_FIELD(g6s2_dsel, ag) G6SCH_FIELD(g6s2_dsel, prior) G6SCH_FIELD(g6s2_dsel, dn) G6SCH_FIELD(g6s2_dsel, dt) G6SCH_FIELD(g6s2_dsel, dc)
    G6SCH_TYPE(g6s2_kstop) G6SCH_FIELD(g6s2_kstop, mode) G6SCH_FIELD(g6s2_kstop, kmax) G6SCH_FIELD(g6s2_kstop, tau) G6SCH_FIELD(g6s2_kstop, seed3) G6SCH_FIELD(g6s2_kstop, log) G6SCH_FIELD(g6s2_kstop, solo) G6SCH_FIELD(g6s2_kstop, b_mode) G6SCH_FIELD(g6s2_kstop, b_kmin) G6SCH_FIELD(g6s2_kstop, b_explore) G6SCH_FIELD(g6s2_kstop, b_tau_rows) G6SCH_FIELD(g6s2_kstop, b_tau) G6SCH_FIELD(g6s2_kstop, b_widths) G6SCH_FIELD(g6s2_kstop, b_n_widths)
    G6SCH_TYPE(g6s2_ngram) G6SCH_FIELD(g6s2_ngram, table) G6SCH_FIELD(g6s2_ngram, n_ns) G6SCH_FIELD(g6s2_ngram, ns) G6SCH_FIELD(g6s2_ngram, has_cal) G6SCH_FIELD(g6s2_ngram, cal) G6SCH_FIELD(g6s2_ngram, n) G6SCH_FIELD(g6s2_ngram, keys) G6SCH_FIELD(g6s2_ngram, toks) G6SCH_FIELD(g6s2_ngram, ps) G6SCH_FIELD(g6s2_ngram, kmax) G6SCH_FIELD(g6s2_ngram, kmin) G6SCH_FIELD(g6s2_ngram, margin) G6SCH_FIELD(g6s2_ngram, log)
    G6SCH_TYPE(g6s2_session) G6SCH_FIELD(g6s2_session, prompt) G6SCH_FIELD(g6s2_session, n_prompt) G6SCH_FIELD(g6s2_session, first) G6SCH_FIELD(g6s2_session, chunk) G6SCH_FIELD(g6s2_session, n_gen) G6SCH_FIELD(g6s2_session, dflash) G6SCH_FIELD(g6s2_session, ins_after) G6SCH_FIELD(g6s2_session, ins_unless) G6SCH_FIELD(g6s2_session, ins_tokens) G6SCH_FIELD(g6s2_session, n_ins)
    G6SCH_TYPE(g6s2_cost) G6SCH_FIELD(g6s2_cost, main0) G6SCH_FIELD(g6s2_cost, main_row) G6SCH_FIELD(g6s2_cost, draft0) G6SCH_FIELD(g6s2_cost, draft_sess) G6SCH_FIELD(g6s2_cost, chain0) G6SCH_FIELD(g6s2_cost, chain_sess) G6SCH_FIELD(g6s2_cost, flash) G6SCH_FIELD(g6s2_cost, flash_table) G6SCH_FIELD(g6s2_cost, flash_draft_round) G6SCH_FIELD(g6s2_cost, flash_draft_each)
    G6SCH_TYPE(g6s2_stats) G6SCH_FIELD(g6s2_stats, rounds) G6SCH_FIELD(g6s2_stats, tokens) G6SCH_FIELD(g6s2_stats, drafted) G6SCH_FIELD(g6s2_stats, mtp_rounds) G6SCH_FIELD(g6s2_stats, switches) G6SCH_FIELD(g6s2_stats, inserted) G6SCH_FIELD(g6s2_stats, ngram_rounds) G6SCH_FIELD(g6s2_stats, ngram_drafted) G6SCH_FIELD(g6s2_stats, ngram_acc) G6SCH_FIELD(g6s2_stats, dfl_rounds) G6SCH_FIELD(g6s2_stats, dfl_from) G6SCH_FIELD(g6s2_stats, dfl_lo) G6SCH_FIELD(g6s2_stats, dfl_warm) G6SCH_FIELD(g6s2_stats, promoted) G6SCH_FIELD(g6s2_stats, acc) G6SCH_FIELD(g6s2_stats, n_acc)
    G6SCH_TYPE(g6s2_slot) G6SCH_FIELD(g6s2_slot, first) G6SCH_FIELD(g6s2_slot, chunk) G6SCH_FIELD(g6s2_slot, n_gen) G6SCH_FIELD(g6s2_slot, dflash) G6SCH_FIELD(g6s2_slot, ins_after) G6SCH_FIELD(g6s2_slot, ins_unless) G6SCH_FIELD(g6s2_slot, n_ins) G6SCH_FIELD(g6s2_slot, ins_tokens) G6SCH_FIELD(g6s2_slot, n_tokens) G6SCH_FIELD(g6s2_slot, tokens) G6SCH_FIELD(g6s2_slot, n_d) G6SCH_FIELD(g6s2_slot, d) G6SCH_FIELD(g6s2_slot, n_dlm) G6SCH_FIELD(g6s2_slot, dlm) G6SCH_FIELD(g6s2_slot, n) G6SCH_FIELD(g6s2_slot, on) G6SCH_FIELD(g6s2_slot, done) G6SCH_FIELD(g6s2_slot, rounds) G6SCH_FIELD(g6s2_slot, k) G6SCH_FIELD(g6s2_slot, dfl) G6SCH_FIELD(g6s2_slot, mtp) G6SCH_FIELD(g6s2_slot, pb) G6SCH_FIELD(g6s2_slot, pd) G6SCH_FIELD(g6s2_slot, pl) G6SCH_FIELD(g6s2_slot, since) G6SCH_FIELD(g6s2_slot, mn) G6SCH_FIELD(g6s2_slot, lo) G6SCH_FIELD(g6s2_slot, warm) G6SCH_FIELD(g6s2_slot, armed) G6SCH_FIELD(g6s2_slot, force) G6SCH_FIELD(g6s2_slot, dd) G6SCH_FIELD(g6s2_slot, dl)
    G6SCH_TYPE(g6s2_learn) G6SCH_FIELD(g6s2_learn, round_no) G6SCH_FIELD(g6s2_learn, cur_round) G6SCH_FIELD(g6s2_learn, cur_active) G6SCH_FIELD(g6s2_learn, ring_q) G6SCH_FIELD(g6s2_learn, ring_lo) G6SCH_FIELD(g6s2_learn, ring_end) G6SCH_FIELD(g6s2_learn, ring_top) G6SCH_FIELD(g6s2_learn, n_ring_tok) G6SCH_FIELD(g6s2_learn, ring_tok)
    G6SCH_TYPE(g6s2_hooks) G6SCH_FIELD(g6s2_hooks, ctx) G6SCH_FIELD(g6s2_hooks, run) G6SCH_FIELD(g6s2_hooks, round) G6SCH_FIELD(g6s2_hooks, prompt) G6SCH_FIELD(g6s2_hooks, emit) G6SCH_FIELD(g6s2_hooks, dfl) G6SCH_FIELD(g6s2_hooks, dfl_start) G6SCH_FIELD(g6s2_hooks, lp) G6SCH_FIELD(g6s2_hooks, has_round) G6SCH_FIELD(g6s2_hooks, has_prompt) G6SCH_FIELD(g6s2_hooks, has_emit) G6SCH_FIELD(g6s2_hooks, has_dfl) G6SCH_FIELD(g6s2_hooks, has_dfl_start) G6SCH_FIELD(g6s2_hooks, has_lp)
    G6SCH_TYPE(g6s2_drv_view) G6SCH_FIELD(g6s2_drv_view, s) G6SCH_FIELD(g6s2_drv_view, n) G6SCH_FIELD(g6s2_drv_view, st)
    G6SCH_TYPE(g6s2_drv_payload) G6SCH_FIELD(g6s2_drv_payload, prompt_len) G6SCH_FIELD(g6s2_drv_payload, chunk) G6SCH_FIELD(g6s2_drv_payload, decode_prompt) G6SCH_FIELD(g6s2_drv_payload, steering_ok)
    G6SCH_TYPE(g6s2_drv_pool) G6SCH_FIELD(g6s2_drv_pool, grow) G6SCH_FIELD(g6s2_drv_pool, shrink) G6SCH_FIELD(g6s2_drv_pool, release) G6SCH_FIELD(g6s2_drv_pool, head)
    G6SCH_TYPE(g6s2_drv_hooks) G6SCH_FIELD(g6s2_drv_hooks, ctx) G6SCH_FIELD(g6s2_drv_hooks, bind) G6SCH_FIELD(g6s2_drv_hooks, retire) G6SCH_FIELD(g6s2_drv_hooks, prefill) G6SCH_FIELD(g6s2_drv_hooks, prefill_pack) G6SCH_FIELD(g6s2_drv_hooks, dfl_ok) G6SCH_FIELD(g6s2_drv_hooks, terminal) G6SCH_FIELD(g6s2_drv_hooks, grow) G6SCH_FIELD(g6s2_drv_hooks, finished) G6SCH_FIELD(g6s2_drv_hooks, resize) G6SCH_FIELD(g6s2_drv_hooks, shrink) G6SCH_FIELD(g6s2_drv_hooks, settle) G6SCH_FIELD(g6s2_drv_hooks, payload) G6SCH_FIELD(g6s2_drv_hooks, grown) G6SCH_FIELD(g6s2_drv_hooks, has_prefill_pack) G6SCH_FIELD(g6s2_drv_hooks, has_dfl_ok) G6SCH_FIELD(g6s2_drv_hooks, has_terminal) G6SCH_FIELD(g6s2_drv_hooks, has_grow) G6SCH_FIELD(g6s2_drv_hooks, has_finished) G6SCH_FIELD(g6s2_drv_hooks, has_resize) G6SCH_FIELD(g6s2_drv_hooks, has_shrink) G6SCH_FIELD(g6s2_drv_hooks, has_settle) G6SCH_FIELD(g6s2_drv_hooks, has_grown)
    G6SCH_TYPE(g6s2_drv_planner) G6SCH_FIELD(g6s2_drv_planner, ctx) G6SCH_FIELD(g6s2_drv_planner, admit) G6SCH_FIELD(g6s2_drv_planner, admit_pack) G6SCH_FIELD(g6s2_drv_planner, round) G6SCH_FIELD(g6s2_drv_planner, demote) G6SCH_FIELD(g6s2_drv_planner, promote) G6SCH_FIELD(g6s2_drv_planner, release) G6SCH_FIELD(g6s2_drv_planner, length) G6SCH_FIELD(g6s2_drv_planner, finished) G6SCH_FIELD(g6s2_drv_planner, active) G6SCH_FIELD(g6s2_drv_planner, inserted)
    G6SCH_TYPE(g6s2_lp_row) G6SCH_FIELD(g6s2_lp_row, lp) G6SCH_FIELD(g6s2_lp_row, rank) G6SCH_FIELD(g6s2_lp_row, n) G6SCH_FIELD(g6s2_lp_row, id) G6SCH_FIELD(g6s2_lp_row, tlp)
    G6SCH_TYPE(g6s2_module_ops) G6SCH_FIELD(g6s2_module_ops, ctx) G6SCH_FIELD(g6s2_module_ops, bind) G6SCH_FIELD(g6s2_module_ops, bind_request) G6SCH_FIELD(g6s2_module_ops, grown) G6SCH_FIELD(g6s2_module_ops, shrink) G6SCH_FIELD(g6s2_module_ops, retire) G6SCH_FIELD(g6s2_module_ops, retain) G6SCH_FIELD(g6s2_module_ops, park) G6SCH_FIELD(g6s2_module_ops, fence) G6SCH_FIELD(g6s2_module_ops, before_work) G6SCH_FIELD(g6s2_module_ops, drained) G6SCH_FIELD(g6s2_module_ops, resize) G6SCH_FIELD(g6s2_module_ops, flash_room) G6SCH_FIELD(g6s2_module_ops, bucket) G6SCH_FIELD(g6s2_module_ops, mode) G6SCH_FIELD(g6s2_module_ops, launch) G6SCH_FIELD(g6s2_module_ops, prefill) G6SCH_FIELD(g6s2_module_ops, prefill_seq) G6SCH_FIELD(g6s2_module_ops, prefill_pack) G6SCH_FIELD(g6s2_module_ops, logprob) G6SCH_FIELD(g6s2_module_ops, draft_lp) G6SCH_FIELD(g6s2_module_ops, dfl_ok) G6SCH_FIELD(g6s2_module_ops, dfl_start) G6SCH_FIELD(g6s2_module_ops, dfl) G6SCH_FIELD(g6s2_module_ops, gm_on) G6SCH_FIELD(g6s2_module_ops, gm_bits) G6SCH_FIELD(g6s2_module_ops, gm_words) G6SCH_FIELD(g6s2_module_ops, pen_rows) G6SCH_FIELD(g6s2_module_ops, pen_sess)
    G6SCH_TYPE(g6s2_pool_hit) G6SCH_FIELD(g6s2_pool_hit, skip) G6SCH_FIELD(g6s2_pool_hit, lcp) G6SCH_FIELD(g6s2_pool_hit, why) G6SCH_FIELD(g6s2_pool_hit, age) G6SCH_FIELD(g6s2_pool_hit, stored) G6SCH_FIELD(g6s2_pool_hit, tier) G6SCH_FIELD(g6s2_pool_hit, slot)
    G6SCH_TYPE(g6s2_pool_ops) G6SCH_FIELD(g6s2_pool_ops, ctx) G6SCH_FIELD(g6s2_pool_ops, match) G6SCH_FIELD(g6s2_pool_ops, take) G6SCH_FIELD(g6s2_pool_ops, digest) G6SCH_FIELD(g6s2_pool_ops, tiers) G6SCH_FIELD(g6s2_pool_ops, refusal) G6SCH_FIELD(g6s2_pool_ops, r_rows) G6SCH_FIELD(g6s2_pool_ops, make_room) G6SCH_FIELD(g6s2_pool_ops, pool) G6SCH_FIELD(g6s2_pool_ops, max_run_pages) G6SCH_FIELD(g6s2_pool_ops, grow) G6SCH_FIELD(g6s2_pool_ops, count) G6SCH_FIELD(g6s2_pool_ops, status)
    G6SCH_TYPE(g6ctl_notice) G6SCH_FIELD(g6ctl_notice, epoch) G6SCH_FIELD(g6ctl_notice, event) G6SCH_FIELD(g6ctl_notice, boundary) G6SCH_FIELD(g6ctl_notice, request) G6SCH_FIELD(g6ctl_notice, payload_hash) G6SCH_FIELD(g6ctl_notice, kind)
    G6SCH_TYPE(g6ctl_ack) G6SCH_FIELD(g6ctl_ack, notice) G6SCH_FIELD(g6ctl_ack, rank) G6SCH_FIELD(g6ctl_ack, state_hash) G6SCH_FIELD(g6ctl_ack, room)
    G6SCH_TYPE(g6ctl_commit) G6SCH_FIELD(g6ctl_commit, c) G6SCH_FIELD(g6ctl_commit, hold) G6SCH_FIELD(g6ctl_commit, has_resize) G6SCH_FIELD(g6ctl_commit, resize)
    G6SCH_TYPE(g6ctl_frame) G6SCH_FIELD(g6ctl_frame, tag) G6SCH_FIELD(g6ctl_frame, notice) G6SCH_FIELD(g6ctl_frame, ack) G6SCH_FIELD(g6ctl_frame, commit) G6SCH_FIELD(g6ctl_frame, epoch) G6SCH_FIELD(g6ctl_frame, request) G6SCH_FIELD(g6ctl_frame, seq) G6SCH_FIELD(g6ctl_frame, rank) G6SCH_FIELD(g6ctl_frame, bytes) G6SCH_FIELD(g6ctl_frame, n)
    G6SCH_TYPE(g6ctl_boot) G6SCH_FIELD(g6ctl_boot, rank) G6SCH_FIELD(g6ctl_boot, ranks) G6SCH_FIELD(g6ctl_boot, timeout) G6SCH_FIELD(g6ctl_boot, port) G6SCH_FIELD(g6ctl_boot, head_ip) G6SCH_FIELD(g6ctl_boot, head_ip_n) G6SCH_FIELD(g6ctl_boot, directory) G6SCH_FIELD(g6ctl_boot, directory_n) G6SCH_FIELD(g6ctl_boot, agreement) G6SCH_FIELD(g6ctl_boot, agreement_n)
    G6SCH_TYPE(g6svc_command) G6SCH_FIELD(g6svc_command, epoch) G6SCH_FIELD(g6svc_command, request) G6SCH_FIELD(g6svc_command, hash) G6SCH_FIELD(g6svc_command, admit)
    G6SCH_TYPE(g6svc_run_use) G6SCH_FIELD(g6svc_run_use, table_version) G6SCH_FIELD(g6svc_run_use, held) G6SCH_FIELD(g6svc_run_use, worst)
    G6SCH_TYPE(g6svc_profile) G6SCH_FIELD(g6svc_profile, name) G6SCH_FIELD(g6svc_profile, name_n) G6SCH_FIELD(g6svc_profile, digest) G6SCH_FIELD(g6svc_profile, active_layers)
    G6SCH_TYPE(g6svc_steering) G6SCH_FIELD(g6svc_steering, profiles) G6SCH_FIELD(g6svc_steering, n) G6SCH_FIELD(g6svc_steering, scope) G6SCH_FIELD(g6svc_steering, forced_scale_bits) G6SCH_FIELD(g6svc_steering, loaded) G6SCH_FIELD(g6svc_steering, forced)
    G6SCH_TYPE(g6svc_boot) G6SCH_FIELD(g6svc_boot, scheduler) G6SCH_FIELD(g6svc_boot, steering) G6SCH_FIELD(g6svc_boot, max_sessions) G6SCH_FIELD(g6svc_boot, rank) G6SCH_FIELD(g6svc_boot, ranks) G6SCH_FIELD(g6svc_boot, prefill_min) G6SCH_FIELD(g6svc_boot, control_version) G6SCH_FIELD(g6svc_boot, lp_max) G6SCH_FIELD(g6svc_boot, directory) G6SCH_FIELD(g6svc_boot, directory_n) G6SCH_FIELD(g6svc_boot, signature) G6SCH_FIELD(g6svc_boot, signature_n) G6SCH_FIELD(g6svc_boot, mtp0) G6SCH_FIELD(g6svc_boot, pack) G6SCH_FIELD(g6svc_boot, park)
    G6SCH_TYPE(g6svc_status) G6SCH_FIELD(g6svc_status, epoch) G6SCH_FIELD(g6svc_status, queued) G6SCH_FIELD(g6svc_status, stalls) G6SCH_FIELD(g6svc_status, grows) G6SCH_FIELD(g6svc_status, stalled) G6SCH_FIELD(g6svc_status, tiers) G6SCH_FIELD(g6svc_status, n_tiers) G6SCH_FIELD(g6svc_status, runs) G6SCH_FIELD(g6svc_status, n_runs) G6SCH_FIELD(g6svc_status, pool_pages) G6SCH_FIELD(g6svc_status, resident) G6SCH_FIELD(g6svc_status, retiring) G6SCH_FIELD(g6svc_status, free) G6SCH_FIELD(g6svc_status, room) G6SCH_FIELD(g6svc_status, held_pages) G6SCH_FIELD(g6svc_status, worst_pages) G6SCH_FIELD(g6svc_status, has_growth) G6SCH_FIELD(g6svc_status, has_tiers) G6SCH_FIELD(g6svc_status, has_runs)
    G6SCH_TYPE(g6svc_checked) G6SCH_FIELD(g6svc_checked, prompt) G6SCH_FIELD(g6svc_checked, n) G6SCH_FIELD(g6svc_checked, sys) G6SCH_FIELD(g6svc_checked, plp) G6SCH_FIELD(g6svc_checked, n_gen) G6SCH_FIELD(g6svc_checked, steering) G6SCH_FIELD(g6svc_checked, ok)
    G6SCH_TYPE(g6svc_pool_call) G6SCH_FIELD(g6svc_pool_call, kind) G6SCH_FIELD(g6svc_pool_call, rows) G6SCH_FIELD(g6svc_pool_call, worst) G6SCH_FIELD(g6svc_pool_call, r) G6SCH_FIELD(g6svc_pool_call, out) G6SCH_FIELD(g6svc_pool_call, run) G6SCH_FIELD(g6svc_pool_call, prompt) G6SCH_FIELD(g6svc_pool_call, n) G6SCH_FIELD(g6svc_pool_call, span) G6SCH_FIELD(g6svc_pool_call, ns)
    G6SCH_TYPE(g6svc_staged) G6SCH_FIELD(g6svc_staged, req) G6SCH_FIELD(g6svc_staged, prompt) G6SCH_FIELD(g6svc_staged, n) G6SCH_FIELD(g6svc_staged, sys) G6SCH_FIELD(g6svc_staged, plp)
    G6SCH_TYPE(g6svc_host) G6SCH_FIELD(g6svc_host, ctx) G6SCH_FIELD(g6svc_host, check) G6SCH_FIELD(g6svc_host, retire_run) G6SCH_FIELD(g6svc_host, share) G6SCH_FIELD(g6svc_host, refusal) G6SCH_FIELD(g6svc_host, tiers) G6SCH_FIELD(g6svc_host, idle_boundary) G6SCH_FIELD(g6svc_host, stage) G6SCH_FIELD(g6svc_host, unstage) G6SCH_FIELD(g6svc_host, digest) G6SCH_FIELD(g6svc_host, abort) G6SCH_FIELD(g6svc_host, apply) G6SCH_FIELD(g6svc_host, mode) G6SCH_FIELD(g6svc_host, step) G6SCH_FIELD(g6svc_host, counters) G6SCH_FIELD(g6svc_host, begin) G6SCH_FIELD(g6svc_host, bound) G6SCH_FIELD(g6svc_host, release)
    G6SCH_TYPE(g6svc_control_cfg) G6SCH_FIELD(g6svc_control_cfg, scheduler) G6SCH_FIELD(g6svc_control_cfg, directory) G6SCH_FIELD(g6svc_control_cfg, directory_n) G6SCH_FIELD(g6svc_control_cfg, endpoint) G6SCH_FIELD(g6svc_control_cfg, ingress) G6SCH_FIELD(g6svc_control_cfg, queue) G6SCH_FIELD(g6svc_control_cfg, trace) G6SCH_FIELD(g6svc_control_cfg, slots) G6SCH_FIELD(g6svc_control_cfg, n_slots) G6SCH_FIELD(g6svc_control_cfg, state) G6SCH_FIELD(g6svc_control_cfg, pool) G6SCH_FIELD(g6svc_control_cfg, host) G6SCH_FIELD(g6svc_control_cfg, lend) G6SCH_FIELD(g6svc_control_cfg, share_ms) G6SCH_FIELD(g6svc_control_cfg, rank) G6SCH_FIELD(g6svc_control_cfg, head) G6SCH_FIELD(g6svc_control_cfg, admit_hold) G6SCH_FIELD(g6svc_control_cfg, elog)
    G6SCH_TYPE(g6svc_compute_cfg) G6SCH_FIELD(g6svc_compute_cfg, control) G6SCH_FIELD(g6svc_compute_cfg, endpoint) G6SCH_FIELD(g6svc_compute_cfg, ingress) G6SCH_FIELD(g6svc_compute_cfg, trace) G6SCH_FIELD(g6svc_compute_cfg, slots) G6SCH_FIELD(g6svc_compute_cfg, n_slots) G6SCH_FIELD(g6svc_compute_cfg, state) G6SCH_FIELD(g6svc_compute_cfg, module) G6SCH_FIELD(g6svc_compute_cfg, host) G6SCH_FIELD(g6svc_compute_cfg, rank) G6SCH_FIELD(g6svc_compute_cfg, elog) G6SCH_FIELD(g6svc_compute_cfg, dfl_reuse) G6SCH_FIELD(g6svc_compute_cfg, ngram)
    G6SCH_TYPE(g6sig_inputs) G6SCH_FIELD(g6sig_inputs, model) G6SCH_FIELD(g6sig_inputs, cap) G6SCH_FIELD(g6sig_inputs, layers_s) G6SCH_FIELD(g6sig_inputs, chunk) G6SCH_FIELD(g6sig_inputs, nc) G6SCH_FIELD(g6sig_inputs, set_id) G6SCH_FIELD(g6sig_inputs, dfl) G6SCH_FIELD(g6sig_inputs, dfl_k) G6SCH_FIELD(g6sig_inputs, s2sess) G6SCH_FIELD(g6sig_inputs, knobs) G6SCH_FIELD(g6sig_inputs, mp) G6SCH_FIELD(g6sig_inputs, mtp) G6SCH_FIELD(g6sig_inputs, dfl_reuse) G6SCH_FIELD(g6sig_inputs, dfl_reuse_min) G6SCH_FIELD(g6sig_inputs, lanes) G6SCH_FIELD(g6sig_inputs, gsplit) G6SCH_FIELD(g6sig_inputs, ckpt) G6SCH_FIELD(g6sig_inputs, ckpt_blocks) G6SCH_FIELD(g6sig_inputs, ckpt_persist) G6SCH_FIELD(g6sig_inputs, s2_below) G6SCH_FIELD(g6sig_inputs, two) G6SCH_FIELD(g6sig_inputs, pf_own) G6SCH_FIELD(g6sig_inputs, pf_qp) G6SCH_FIELD(g6sig_inputs, qp_ctx) G6SCH_FIELD(g6sig_inputs, qp_min) G6SCH_FIELD(g6sig_inputs, grammar) G6SCH_FIELD(g6sig_inputs, ngram) G6SCH_FIELD(g6sig_inputs, kstop_mode) G6SCH_FIELD(g6sig_inputs, kstop_kmax) G6SCH_FIELD(g6sig_inputs, kstop_tau) G6SCH_FIELD(g6sig_inputs, kstop_seed3) G6SCH_FIELD(g6sig_inputs, draft_par) G6SCH_FIELD(g6sig_inputs, qs_mode) G6SCH_FIELD(g6sig_inputs, iq_split) G6SCH_FIELD(g6sig_inputs, mtp_split) G6SCH_FIELD(g6sig_inputs, dhead_on) G6SCH_FIELD(g6sig_inputs, dhead_tpc) G6SCH_FIELD(g6sig_inputs, dhead_ks) G6SCH_FIELD(g6sig_inputs, mtp_last) G6SCH_FIELD(g6sig_inputs, dhead2) G6SCH_FIELD(g6sig_inputs, dfl_i4) G6SCH_FIELD(g6sig_inputs, dfl_body_i4) G6SCH_FIELD(g6sig_inputs, dfl_alpha) G6SCH_FIELD(g6sig_inputs, dsel_model) G6SCH_FIELD(g6sig_inputs, dsel_tab) G6SCH_FIELD(g6sig_inputs, kstopb) G6SCH_FIELD(g6sig_inputs, cp_step) G6SCH_FIELD(g6sig_inputs, cp_seg) G6SCH_FIELD(g6sig_inputs, ctx_sp) G6SCH_FIELD(g6sig_inputs, posidx) G6SCH_FIELD(g6sig_inputs, steer_shared) G6SCH_FIELD(g6sig_inputs, ds_mtp) G6SCH_FIELD(g6sig_inputs, sh_tc) G6SCH_FIELD(g6sig_inputs, l2hint) G6SCH_FIELD(g6sig_inputs, fc2stg) G6SCH_FIELD(g6sig_inputs, fc1bstg) G6SCH_FIELD(g6sig_inputs, c8_pack) G6SCH_FIELD(g6sig_inputs, sess_drafts) G6SCH_FIELD(g6sig_inputs, pf_lanes) G6SCH_FIELD(g6sig_inputs, tk_main) G6SCH_FIELD(g6sig_inputs, tk_decode) G6SCH_FIELD(g6sig_inputs, idx_pipe_b) G6SCH_FIELD(g6sig_inputs, idx_q_trig) G6SCH_FIELD(g6sig_inputs, idx_pipe_qr) G6SCH_FIELD(g6sig_inputs, reuse) G6SCH_FIELD(g6sig_inputs, vis_on) G6SCH_FIELD(g6sig_inputs, vis_id)
    G6SCH_TYPE(G6ModuleInfo) G6SCH_FIELD(G6ModuleInfo, abi) G6SCH_FIELD(G6ModuleInfo, model) G6SCH_FIELD(G6ModuleInfo, name) G6SCH_FIELD(G6ModuleInfo, nr) G6SCH_FIELD(G6ModuleInfo, mx) G6SCH_FIELD(G6ModuleInfo, s2n) G6SCH_FIELD(G6ModuleInfo, max_sessions) G6SCH_FIELD(G6ModuleInfo, hidden) G6SCH_FIELD(G6ModuleInfo, layers) G6SCH_FIELD(G6ModuleInfo, mtp_layers) G6SCH_FIELD(G6ModuleInfo, vocab) G6SCH_FIELD(G6ModuleInfo, experts) G6SCH_FIELD(G6ModuleInfo, topk) G6SCH_FIELD(G6ModuleInfo, kv_row_bytes) G6SCH_FIELD(G6ModuleInfo, kv_page) G6SCH_FIELD(G6ModuleInfo, kv_format) G6SCH_FIELD(G6ModuleInfo, caps) G6SCH_FIELD(G6ModuleInfo, dfl_block) G6SCH_FIELD(G6ModuleInfo, pack_members) G6SCH_FIELD(G6ModuleInfo, dfl_drafts) G6SCH_FIELD(G6ModuleInfo, dfl_rep) G6SCH_FIELD(G6ModuleInfo, tp_sort) G6SCH_FIELD(G6ModuleInfo, kv4) G6SCH_FIELD(G6ModuleInfo, xgrammar) G6SCH_FIELD(G6ModuleInfo, ds_markov_maxs) G6SCH_FIELD(G6ModuleInfo, ds_block)
    G6SCH_TYPE(g6opt_knobs) G6SCH_FIELD(g6opt_knobs, table_hash) G6SCH_FIELD(g6opt_knobs, flag) G6SCH_FIELD(g6opt_knobs, int_) G6SCH_FIELD(g6opt_knobs, float_) G6SCH_FIELD(g6opt_knobs, text) G6SCH_FIELD(g6opt_knobs, ints) G6SCH_FIELD(g6opt_knobs, floats) G6SCH_FIELD(g6opt_knobs, tests_set)
    G6SCH_TYPE(g6opt_host) G6SCH_FIELD(g6opt_host, ctx) G6SCH_FIELD(g6opt_host, engram_colls) G6SCH_FIELD(g6opt_host, dsk_mtp) G6SCH_FIELD(g6opt_host, dsk_relaxed) G6SCH_FIELD(g6opt_host, dsk_relax_topp) G6SCH_FIELD(g6opt_host, dsk_err)
    G6SCH_TYPE(g6opt_strs) G6SCH_FIELD(g6opt_strs, p) G6SCH_FIELD(g6opt_strs, n)
    G6SCH_TYPE(g6opt_plan) G6SCH_FIELD(g6opt_plan, prompt) G6SCH_FIELD(g6opt_plan, first) G6SCH_FIELD(g6opt_plan, chunk) G6SCH_FIELD(g6opt_plan, n_gen) G6SCH_FIELD(g6opt_plan, inv_temp) G6SCH_FIELD(g6opt_plan, seed) G6SCH_FIELD(g6opt_plan, top_p) G6SCH_FIELD(g6opt_plan, lp) G6SCH_FIELD(g6opt_plan, plp) G6SCH_FIELD(g6opt_plan, rs)
    G6SCH_TYPE(g6opt_options) G6SCH_FIELD(g6opt_options, rank) G6SCH_FIELD(g6opt_options, ips) G6SCH_FIELD(g6opt_options, ips2) G6SCH_FIELD(g6opt_options, ring) G6SCH_FIELD(g6opt_options, ring_devs) G6SCH_FIELD(g6opt_options, proxy_cpu) G6SCH_FIELD(g6opt_options, pf_ips2) G6SCH_FIELD(g6opt_options, manifest) G6SCH_FIELD(g6opt_options, packfile) G6SCH_FIELD(g6opt_options, v2_dir) G6SCH_FIELD(g6opt_options, v2_slices) G6SCH_FIELD(g6opt_options, v2_widths) G6SCH_FIELD(g6opt_options, v2_dense) G6SCH_FIELD(g6opt_options, v2_layout) G6SCH_FIELD(g6opt_options, v2_trellis) G6SCH_FIELD(g6opt_options, qo_man) G6SCH_FIELD(g6opt_options, qo_file) G6SCH_FIELD(g6opt_options, layers_s) G6SCH_FIELD(g6opt_options, levels_s) G6SCH_FIELD(g6opt_options, prompts) G6SCH_FIELD(g6opt_options, co_cap) G6SCH_FIELD(g6opt_options, co_mp) G6SCH_FIELD(g6opt_options, co_top) G6SCH_FIELD(g6opt_options, co_pfrows) G6SCH_FIELD(g6opt_options, co_mtp) G6SCH_FIELD(g6opt_options, co_lanes) G6SCH_FIELD(g6opt_options, co_nc) G6SCH_FIELD(g6opt_options, co_temp) G6SCH_FIELD(g6opt_options, co_graph) G6SCH_FIELD(g6opt_options, co_wide) G6SCH_FIELD(g6opt_options, co_pfv2) G6SCH_FIELD(g6opt_options, co_adapt) G6SCH_FIELD(g6opt_options, has_n_gen) G6SCH_FIELD(g6opt_options, n_gen) G6SCH_FIELD(g6opt_options, has_chunk) G6SCH_FIELD(g6opt_options, chunk) G6SCH_FIELD(g6opt_options, has_seed) G6SCH_FIELD(g6opt_options, seed) G6SCH_FIELD(g6opt_options, has_top_p) G6SCH_FIELD(g6opt_options, top_p) G6SCH_FIELD(g6opt_options, has_rs) G6SCH_FIELD(g6opt_options, rs) G6SCH_FIELD(g6opt_options, qp_man) G6SCH_FIELD(g6opt_options, qp_file) G6SCH_FIELD(g6opt_options, dfl_man) G6SCH_FIELD(g6opt_options, dfl_file) G6SCH_FIELD(g6opt_options, dfl_sel) G6SCH_FIELD(g6opt_options, dfl_k) G6SCH_FIELD(g6opt_options, spec_reset_per_request) G6SCH_FIELD(g6opt_options, spec_fixed) G6SCH_FIELD(g6opt_options, plan) G6SCH_FIELD(g6opt_options, image) G6SCH_FIELD(g6opt_options, hash) G6SCH_FIELD(g6opt_options, reuse_prefix) G6SCH_FIELD(g6opt_options, check_args) G6SCH_FIELD(g6opt_options, lanes_explicit) G6SCH_FIELD(g6opt_options, s2dir) G6SCH_FIELD(g6opt_options, grammar_dir) G6SCH_FIELD(g6opt_options, kv_tiers) G6SCH_FIELD(g6opt_options, kv_keep) G6SCH_FIELD(g6opt_options, kv_system_after) G6SCH_FIELD(g6opt_options, kv_write_limit) G6SCH_FIELD(g6opt_options, kv_format) G6SCH_FIELD(g6opt_options, kv_old) G6SCH_FIELD(g6opt_options, kv_old_all) G6SCH_FIELD(g6opt_options, set_id) G6SCH_FIELD(g6opt_options, steering_dir) G6SCH_FIELD(g6opt_options, steering_force_s) G6SCH_FIELD(g6opt_options, steering_model_revision) G6SCH_FIELD(g6opt_options, steering_tokenizer_revision) G6SCH_FIELD(g6opt_options, steering_template_digest) G6SCH_FIELD(g6opt_options, steering_weight_set) G6SCH_FIELD(g6opt_options, steering_template_file) G6SCH_FIELD(g6opt_options, s2port) G6SCH_FIELD(g6opt_options, s2sess) G6SCH_FIELD(g6opt_options, s2epoch) G6SCH_FIELD(g6opt_options, cap) G6SCH_FIELD(g6opt_options, top) G6SCH_FIELD(g6opt_options, pfrows) G6SCH_FIELD(g6opt_options, nc) G6SCH_FIELD(g6opt_options, mtp) G6SCH_FIELD(g6opt_options, mp) G6SCH_FIELD(g6opt_options, lanes) G6SCH_FIELD(g6opt_options, temp) G6SCH_FIELD(g6opt_options, graph) G6SCH_FIELD(g6opt_options, wide) G6SCH_FIELD(g6opt_options, pfv2) G6SCH_FIELD(g6opt_options, adapt) G6SCH_FIELD(g6opt_options, release) G6SCH_FIELD(g6opt_options, s2mode) G6SCH_FIELD(g6opt_options, dfl) G6SCH_FIELD(g6opt_options, k) G6SCH_FIELD(g6opt_options, layers) G6SCH_FIELD(g6opt_options, progress_isolate) G6SCH_FIELD(g6opt_options, route_split_mode) G6SCH_FIELD(g6opt_options, c8_pack_env) G6SCH_FIELD(g6opt_options, spec_on) G6SCH_FIELD(g6opt_options, spec_levels) G6SCH_FIELD(g6opt_options, spec_draft_round) G6SCH_FIELD(g6opt_options, spec_draft_each) G6SCH_FIELD(g6opt_options, spec_table) G6SCH_FIELD(g6opt_options, rt_win) G6SCH_FIELD(g6opt_options, kstop_mode) G6SCH_FIELD(g6opt_options, kstop_kmax) G6SCH_FIELD(g6opt_options, kstop_tau) G6SCH_FIELD(g6opt_options, kstop_seed3) G6SCH_FIELD(g6opt_options, kstop_log) G6SCH_FIELD(g6opt_options, kstop_solo) G6SCH_FIELD(g6opt_options, kstopb_mode) G6SCH_FIELD(g6opt_options, kstopb_kmin) G6SCH_FIELD(g6opt_options, kstopb_tau) G6SCH_FIELD(g6opt_options, kstopb_tau_n) G6SCH_FIELD(g6opt_options, kstopb_widths) G6SCH_FIELD(g6opt_options, kstopb_explore) G6SCH_FIELD(g6opt_options, flash_cache) G6SCH_FIELD(g6opt_options, flash_budget_mib) G6SCH_FIELD(g6opt_options, ds_moe_main_topk) G6SCH_FIELD(g6opt_options, ds_moe_decode_topk) G6SCH_FIELD(g6opt_options, ds_idx_pipe_b) G6SCH_FIELD(g6opt_options, ds_idx_q_trig) G6SCH_FIELD(g6opt_options, ds_idx_pipe_qr) G6SCH_FIELD(g6opt_options, ds_admit_keep_scratch) G6SCH_FIELD(g6opt_options, ds_sess_drafts) G6SCH_FIELD(g6opt_options, ds_draft_cap) G6SCH_FIELD(g6opt_options, plans) G6SCH_FIELD(g6opt_options, n_plans) G6SCH_FIELD(g6opt_options, row_counts) G6SCH_FIELD(g6opt_options, draft_rows) G6SCH_FIELD(g6opt_options, nl) G6SCH_FIELD(g6opt_options, nx) G6SCH_FIELD(g6opt_options, ncoll) G6SCH_FIELD(g6opt_options, r) G6SCH_FIELD(g6opt_options, own4) G6SCH_FIELD(g6opt_options, qp_roll_s) G6SCH_FIELD(g6opt_options, qp_roll_ser) G6SCH_FIELD(g6opt_options, qp_wire_e)
    G6SCH_TYPE(g6plan_runs) G6SCH_FIELD(g6plan_runs, run1) G6SCH_FIELD(g6plan_runs, run2) G6SCH_FIELD(g6plan_runs, run3) G6SCH_FIELD(g6plan_runs, ncoll) G6SCH_FIELD(g6plan_runs, slot_bytes) G6SCH_FIELD(g6plan_runs, slot_bytes1) G6SCH_FIELD(g6plan_runs, slot_bytes2) G6SCH_FIELD(g6plan_runs, slot_bytes3) G6SCH_FIELD(g6plan_runs, rs_k) G6SCH_FIELD(g6plan_runs, quarter) G6SCH_FIELD(g6plan_runs, slice_stride)
    G6SCH_TYPE(g6plan_cp_dims) G6SCH_FIELD(g6plan_cp_dims, heads) G6SCH_FIELD(g6plan_cp_dims, kv_rank) G6SCH_FIELD(g6plan_cp_dims, qk_dim) G6SCH_FIELD(g6plan_cp_dims, topk)
    G6SCH_TYPE(g6plan_cfg) G6SCH_FIELD(g6plan_cfg, rank) G6SCH_FIELD(g6plan_cfg, stripe_min) G6SCH_FIELD(g6plan_cfg, slot_bytes) G6SCH_FIELD(g6plan_cfg, run1) G6SCH_FIELD(g6plan_cfg, run2) G6SCH_FIELD(g6plan_cfg, slot_bytes1) G6SCH_FIELD(g6plan_cfg, slot_bytes2) G6SCH_FIELD(g6plan_cfg, run3) G6SCH_FIELD(g6plan_cfg, slot_bytes3) G6SCH_FIELD(g6plan_cfg, pool) G6SCH_FIELD(g6plan_cfg, sge2) G6SCH_FIELD(g6plan_cfg, compact) G6SCH_FIELD(g6plan_cfg, ncoll) G6SCH_FIELD(g6plan_cfg, port) G6SCH_FIELD(g6plan_cfg, proxy_cpu) G6SCH_FIELD(g6plan_cfg, vcap) G6SCH_FIELD(g6plan_cfg, n_vcap) G6SCH_FIELD(g6plan_cfg, vdst_k) G6SCH_FIELD(g6plan_cfg, n_vdst_k) G6SCH_FIELD(g6plan_cfg, bulk_bytes) G6SCH_FIELD(g6plan_cfg, bulk_flags) G6SCH_FIELD(g6plan_cfg, xtrace) G6SCH_FIELD(g6plan_cfg, ring) G6SCH_FIELD(g6plan_cfg, relay_cpu)
    G6SCH_TYPE(g6plan_entry) G6SCH_FIELD(g6plan_entry, kind) G6SCH_FIELD(g6plan_entry, late) G6SCH_FIELD(g6plan_entry, use_pf_ips2) G6SCH_FIELD(g6plan_entry, port_off) G6SCH_FIELD(g6plan_entry, cfg)
    G6SCH_TYPE(g6plan_knobs) G6SCH_FIELD(g6plan_knobs, xtrace_pf) G6SCH_FIELD(g6plan_knobs, pf_stripe) G6SCH_FIELD(g6plan_knobs, pf_stripe_min) G6SCH_FIELD(g6plan_knobs, pf_rs) G6SCH_FIELD(g6plan_knobs, fpf_attn_p1) G6SCH_FIELD(g6plan_knobs, fpf_dsa_p1) G6SCH_FIELD(g6plan_knobs, fpf_p1) G6SCH_FIELD(g6plan_knobs, has_qa_split_cpu) G6SCH_FIELD(g6plan_knobs, qa_split_cpu) G6SCH_FIELD(g6plan_knobs, cp) G6SCH_FIELD(g6plan_knobs, cp_r) G6SCH_FIELD(g6plan_knobs, cp_step_rows) G6SCH_FIELD(g6plan_knobs, cp_s3) G6SCH_FIELD(g6plan_knobs, has_cp_mmax) G6SCH_FIELD(g6plan_knobs, cp_mmax) G6SCH_FIELD(g6plan_knobs, has_cp_r2) G6SCH_FIELD(g6plan_knobs, cp_r2) G6SCH_FIELD(g6plan_knobs, ds_prefill_w) G6SCH_FIELD(g6plan_knobs, ds_pf_tail) G6SCH_FIELD(g6plan_knobs, ds_pf_pool) G6SCH_FIELD(g6plan_knobs, ds_pf_lanes) G6SCH_FIELD(g6plan_knobs, ds_pf_l20_tail) G6SCH_FIELD(g6plan_knobs, ds_pf_rs) G6SCH_FIELD(g6plan_knobs, ds_mhc_row_shard) G6SCH_FIELD(g6plan_knobs, ds_mhc_fuse) G6SCH_FIELD(g6plan_knobs, ds_mhc_y_mxfp8) G6SCH_FIELD(g6plan_knobs, ds_pf_direct) G6SCH_FIELD(g6plan_knobs, ds_subdump) G6SCH_FIELD(g6plan_knobs, ds_stripe) G6SCH_FIELD(g6plan_knobs, ds_xtrace) G6SCH_FIELD(g6plan_knobs, ds_stripe_min)
    G6SCH_TYPE(g6plan_build) G6SCH_FIELD(g6plan_build, model) G6SCH_FIELD(g6plan_build, nr) G6SCH_FIELD(g6plan_build, mx) G6SCH_FIELD(g6plan_build, hid) G6SCH_FIELD(g6plan_build, ds_window) G6SCH_FIELD(g6plan_build, ds_tail_rows) G6SCH_FIELD(g6plan_build, ds_cand_source) G6SCH_FIELD(g6plan_build, ds_late_start) G6SCH_FIELD(g6plan_build, ds_pack_rows) G6SCH_FIELD(g6plan_build, ds_draft_ncoll) G6SCH_FIELD(g6plan_build, coll_bits) G6SCH_FIELD(g6plan_build, cp) G6SCH_FIELD(g6plan_build, qw_half) G6SCH_FIELD(g6plan_build, route_max_nr)
    G6SCH_TYPE(g6plan_in) G6SCH_FIELD(g6plan_in, b) G6SCH_FIELD(g6plan_in, cfg) G6SCH_FIELD(g6plan_in, n_ips) G6SCH_FIELD(g6plan_in, n_ips2) G6SCH_FIELD(g6plan_in, n_pf_ips2) G6SCH_FIELD(g6plan_in, nc) G6SCH_FIELD(g6plan_in, nx) G6SCH_FIELD(g6plan_in, nl) G6SCH_FIELD(g6plan_in, mp) G6SCH_FIELD(g6plan_in, mtp) G6SCH_FIELD(g6plan_in, lanes) G6SCH_FIELD(g6plan_in, drafts) G6SCH_FIELD(g6plan_in, chunk) G6SCH_FIELD(g6plan_in, s2sess) G6SCH_FIELD(g6plan_in, rt_win) G6SCH_FIELD(g6plan_in, route_split_mode) G6SCH_FIELD(g6plan_in, c8_pack_env) G6SCH_FIELD(g6plan_in, ds_draft_cap) G6SCH_FIELD(g6plan_in, qp_roll_s) G6SCH_FIELD(g6plan_in, qp_wire_e) G6SCH_FIELD(g6plan_in, s2mode) G6SCH_FIELD(g6plan_in, dfl) G6SCH_FIELD(g6plan_in, dsk_on) G6SCH_FIELD(g6plan_in, layers) G6SCH_FIELD(g6plan_in, n_layers) G6SCH_FIELD(g6plan_in, row_counts) G6SCH_FIELD(g6plan_in, n_row_counts) G6SCH_FIELD(g6plan_in, knobs) G6SCH_FIELD(g6plan_in, pf_sp) G6SCH_FIELD(g6plan_in, pool_owner_mode) G6SCH_FIELD(g6plan_in, dfl_ncoll) G6SCH_FIELD(g6plan_in, head_payload) G6SCH_FIELD(g6plan_in, trunk_bytes) G6SCH_FIELD(g6plan_in, full_index) G6SCH_FIELD(g6plan_in, n_full_index) G6SCH_FIELD(g6plan_in, cp_build) G6SCH_FIELD(g6plan_in, gather) G6SCH_FIELD(g6plan_in, draft_gather) G6SCH_FIELD(g6plan_in, pool_payload) G6SCH_FIELD(g6plan_in, route_payload) G6SCH_FIELD(g6plan_in, dsp_coll_max)
    G6SCH_TYPE(g6plan_cp) G6SCH_FIELD(g6plan_cp, nr) G6SCH_FIELD(g6plan_cp, nl) G6SCH_FIELD(g6plan_cp, mmax) G6SCH_FIELD(g6plan_cp, q_row) G6SCH_FIELD(g6plan_cp, cand_row) G6SCH_FIELD(g6plan_cp, part_row) G6SCH_FIELD(g6plan_cp, q) G6SCH_FIELD(g6plan_cp, n_q) G6SCH_FIELD(g6plan_cp, cand) G6SCH_FIELD(g6plan_cp, n_cand) G6SCH_FIELD(g6plan_cp, part) G6SCH_FIELD(g6plan_cp, n_part) G6SCH_FIELD(g6plan_cp, cap) G6SCH_FIELD(g6plan_cp, dst_k) G6SCH_FIELD(g6plan_cp, ncoll)
    G6SCH_TYPE(g6plan_plan) G6SCH_FIELD(g6plan_plan, refusal) G6SCH_FIELD(g6plan_plan, log) G6SCH_FIELD(g6plan_plan, n_log) G6SCH_FIELD(g6plan_plan, n) G6SCH_FIELD(g6plan_plan, entries) G6SCH_FIELD(g6plan_plan, dec_slot_bytes) G6SCH_FIELD(g6plan_plan, pf_rows) G6SCH_FIELD(g6plan_plan, tail) G6SCH_FIELD(g6plan_plan, i_late) G6SCH_FIELD(g6plan_plan, l20_tail) G6SCH_FIELD(g6plan_plan, rs_k) G6SCH_FIELD(g6plan_plan, quarter) G6SCH_FIELD(g6plan_plan, slice_stride) G6SCH_FIELD(g6plan_plan, rowshard) G6SCH_FIELD(g6plan_plan, y_mxfp8) G6SCH_FIELD(g6plan_plan, pf_direct) G6SCH_FIELD(g6plan_plan, pf_lanes_fab) G6SCH_FIELD(g6plan_plan, two) G6SCH_FIELD(g6plan_plan, pool_owner) G6SCH_FIELD(g6plan_plan, route_split) G6SCH_FIELD(g6plan_plan, cp_on) G6SCH_FIELD(g6plan_plan, cp_r) G6SCH_FIELD(g6plan_plan, cp_r2) G6SCH_FIELD(g6plan_plan, cp_s3) G6SCH_FIELD(g6plan_plan, cp_mmax) G6SCH_FIELD(g6plan_plan, cp_step) G6SCH_FIELD(g6plan_plan, cp_step_r) G6SCH_FIELD(g6plan_plan, cp)
    return h;
}
#undef G6SCH_TYPE
#undef G6SCH_FIELD
