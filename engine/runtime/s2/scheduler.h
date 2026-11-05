// The S2 scheduler, host-only state. No CUDA, allocator implementation, sampler, or transport.
// A caller may apply Commit ONLY after event-boundary all-rank quiescence (event_gate.h).
// TP4 owns run allocation/claim/release; the batch driver owns tokens, MTP and hidden state.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "../steering.h"
#include "../../module_info.h"
#include "../../scheduler-rs/g6sched.h"

namespace g6s2 {
inline void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
// a scheduler-rs refusal: null, or the message the C++ it replaced threw
inline void check(const char* refused) { if (refused) throw std::runtime_error(refused); }
constexpr unsigned MAX_SESSIONS = unsigned(g6mod::INFO.max_sessions), PAGE = unsigned(g6mod::INFO.kv_page);
// DS41: members of one ragged prefill pack (ds41::pack::MAX_SEGMENTS); the slots they name range over MAX_SESSIONS
constexpr unsigned PACK_MAX_MEMBERS = 8;
// The control link's version: the frame magic is the numeric base "G6S2E0" + this version (control_protocol.h), and the bootstrap agreement
// carries it (bootstrap.h), so ranks built with different commit frames refuse at bootstrap, not at the first event.
// 6 adds Commit::hold. 7 adds the Resize event (Commit::resize). 9 adds Ack::room (8 is the fail-soft store's, #159).
// 10 adds the steering profile slot. 11 adds Request::r_rows. 12 adds Commit::shrink. 13 adds 32 session slots and the
// DS41 prefill packs.
constexpr unsigned CONTROL_VERSION = 13;
constexpr uint32_t MAX_DECODE_ROUNDS = 64;   // Config::decode_rounds bound
// Prefill choice (Scheduler::next): of the sessions with prompt rows left, the one with the fewest runs next, so a short
// request admitted behind a long prefill is served instead of waiting out every quantum of it (#61). Shortest-first
// alone can defer a long prefill under a stream of short ones, so a session passed over this many times is owed a turn:
// it is served ahead of every session that is not owed, and once owed it is passed over at most once more per competing
// prefilling session (an owed rival that is served falls back to 0 and cannot catch the growing count again), so a long
// prompt always makes progress. 1 is one-for-one, i.e. round-robin; raise it for a more aggressive shortest-first. The
// cost is only how long a long prompt may wait, never correctness.
constexpr uint32_t PREFILL_DEFER_MAX = 3;
// A notice's target boundary (service.h event) is the first boundary whose work after the one in flight covers
// NOTICE_ROWS prefill rows, floored at NOTICE_MIN and capped at NOTICE_MAX. The constant this replaces was LOOKAHEAD = 8
// boundaries, the margin a decode round needs — and, during a prefill, eight ~6 s quanta for a propagation that takes
// milliseconds (#61), which is why a cancel could not land inside any below-R prefill: R is exactly eight 8,192-row
// quanta at the cluster's line. NOTICE_MIN: the boundary in flight when the head reads may be nearly done, so its
// remaining time is unknowable and the margin starts one boundary later. NOTICE_MAX: never a wider target than the
// constant bought, so the pure-decode case keeps exactly that behaviour (a decode round adds no coverage — stalls can
// shrink one to almost nothing, so the projection counts none). NOTICE_ROWS: 2,048 prefill rows is about 1.5 s at the
// ~1,300 rows/s the gate measures for a cold 8K prefill, an order of magnitude past the milliseconds a notice needs to
// reach every rank; and a rank cannot depart a boundary before every rank completed it (each step is one TP collective),
// so the collective, not the fastest rank, sets the margin.
constexpr uint64_t NOTICE_MIN = 2, NOTICE_MAX = 8;
constexpr uint32_t NOTICE_ROWS = 2048;
struct Config {
    uint64_t epoch = 1;
    uint32_t pool_rows = 614400, max_rows = 24, max_sessions = 8, mtp = 2;
    uint32_t guard_tokens = UINT32_MAX, prefill_rows = 4096;
    bool two_lanes = true;
    g6rt::steering::BootConfig steering{};
    // While any session decodes, a prefill quantum takes at most this many rows on one lane, so a decode round comes
    // round sooner (each round waits out one prefill quantum). 0 = off: the full prefill_rows x lanes quantum, as before.
    // A multiple of PAGE, at most prefill_rows (native.cuh prefill: lane A takes up to prefill_rows rows).
    uint32_t decode_prefill_rows = 0;   // last: tests build Config positionally
    // While a prefill waits and sessions decode, this many decode rounds run per prefill quantum, so the decoders get
    // K rounds per quantum's wait instead of one (Scheduler::alternate). 0 or 1 = off: one round per quantum, as before.
    uint32_t decode_rounds = 0;         // after decode_prefill_rows, for the same reason
    // remote rows release (pool.h window, kvman-rs Pool::window): a run past R lists its worst case and this many window
    // pages besides. 0 = off
    uint32_t window_pages = 0;          // last, for the same reason
    // GR8_CP_STEP_ROWS (cp_rows.h): a step past R carries at most cp_step_rows rows (0 = off: every decoding session in
    // every round, as before). cp_after: contexts above it run the CP block (cp_after()). cp_seg: GR8_CP_STEP_SEG. Read by
    // Driver::step only; the scheduler's quanta keep their kinds (a capped round is a Decode quantum with fewer sessions)
    uint32_t cp_step_rows = 0;          // after window_pages, for the same reason
    uint64_t cp_after = UINT64_MAX;
    bool cp_seg = false;
    // R per grant: the smaller R a run may take (Request::r_rows; 0 = none). Steps switch to the CP block at one context
    // for every session (cp_after), whichever R it takes
    uint32_t r2_rows = 0;
    // GR8_S2_TAIL_FIRST=1: a session whose prompt is prefilled (only its tail left) is served before any prefill, owed or
    // not, and its tail does not ask for a decode round (Scheduler::alternate), so a burst of short prompts reaches its
    // first tokens one tail per step instead of after every prefill and a decode round per tail. false = off: a tail
    // takes its turn like a prefill and asks for a decode round.
    bool tail_first = false;            // after cp_seg, for the same reason
    // GR8_S2_NOTICE_MAX=<k>: the notice target's cap, NOTICE_MIN..NOTICE_MAX boundaries (horizon(), service.h event).
    // NOTICE_MAX (8) is the default. Lower only from a measured margin: a notice that lands after its boundary is fatal
    uint32_t notice_max = uint32_t(NOTICE_MAX);   // after tail_first, for the same reason
    // DS41 only: short fresh prompts prefill together in one ragged pass, then share one prompt-tail step
    bool c8_pack = false;
    // 0: every session drafts mtp tokens a round. n: drafting stops while more than n sessions are active (one row each),
    // so up to max_rows sessions decode. Batch::set_mtp_sessions.
    uint32_t mtp_sessions = 0;
    // DS41 only, GR8_DS_SESS_DRAFTS=1: a shared round's drafts a session go by its active count (s2/ds41_sess_drafts.h,
    // Batch::set_sess_drafts) instead of the mtp_sessions cut; needs mtp_sessions > 0 (one row a session at most bounds limit())
    bool sess_drafts = false;
    // DS41 only, GR8_DS_PARK=1: a finished session's slot is parked (Phase::Parked) instead of freed: its KV stays in the
    // slot's slabs, so a follow-up whose prompt extends it is granted that slot and prefills only the new rows. A new chat
    // takes a free slot below limit() first, else the oldest parked slot (Scheduler::next_key). false = off: as before
    bool park = false;
    // max_rows bounds ONE step: a round is one main step of every session's mtp + 1 rows (s2_batch.h Batch::round)
    uint32_t guard() const { return guard_tokens == UINT32_MAX ? 2 * mtp : guard_tokens; }
    unsigned limit() const {
        const unsigned step = mtp_sessions ? 1u : (mtp + 1);
        return std::min(max_sessions, max_rows / step);
    }
    // the settings scheduler-rs reads: every field but the steering receipt
    g6sch_config scalars() const {
        g6sch_config x{};
        x.epoch = epoch; x.pool_rows = pool_rows; x.max_rows = max_rows; x.max_sessions = max_sessions; x.mtp = mtp;
        x.guard_tokens = guard_tokens; x.prefill_rows = prefill_rows; x.two_lanes = two_lanes;
        x.decode_prefill_rows = decode_prefill_rows; x.decode_rounds = decode_rounds; x.window_pages = window_pages;
        x.cp_step_rows = cp_step_rows; x.cp_after = cp_after; x.cp_seg = cp_seg; x.r2_rows = r2_rows;
        x.tail_first = tail_first; x.notice_max = notice_max; x.c8_pack = c8_pack; x.mtp_sessions = mtp_sessions;
        x.sess_drafts = sess_drafts; x.park = park;
        return x;
    }
    void validate() const {
        const g6sch_config c = scalars();
        check(g6sch_validate(&c, MAX_SESSIONS, g6mod::has(G6_CAP_MTP0), g6mod::has(G6_CAP_PACK), g6mod::has(G6_CAP_PARK)));
        require(steering.loaded == !steering.profiles.empty() && steering.profiles.size() <= g6rt::steering::PROFILES_MAX &&
                (steering.scope == g6rt::steering::Scope::Shared || steering.scope == g6rt::steering::Scope::Full),
                "invalid loaded steering receipt");
        for (const auto& p : steering.profiles)
            require(g6rt::steering::valid_profile_name(p.name) && p.active_layers, "invalid loaded steering receipt");
        require(!steering.forced || steering.profiles.size() == 1, "forced steering needs exactly one loaded profile");
        if (steering.forced) {
            const float scale = g6rt::steering::bits_float(steering.forced_scale_bits);
            require(std::isfinite(scale) && scale > 0.f && scale <= g6rt::steering::SCALE_MAX,
                    "forced steering needs a finite scale in (0,4]");
        } else require(!steering.forced_scale_bits, "unforced steering has a forced scale");
    }
};
struct Key {
    uint64_t epoch = 0; uint32_t slot = 0, generation = 0;
    bool operator==(const Key& b) const { return epoch == b.epoch && slot == b.slot && generation == b.generation; }
    bool operator!=(const Key& b) const { return !(*this == b); }
};
struct Request {
    uint64_t id = 0; // immutable native request ordinal; public HTTP ids may be reused
    uint32_t prompt_tokens = 0, max_tokens = 0;
    g6rt::steering::Identity steering{};
    uint32_t r_rows = 0;   // the R its run takes (pool.h), the head's choice at admission: 0 = the pool's, else Config::r2_rows
};
template <class T, class F> inline T sch_out(F f) { T v{}; check(f(&v)); return v; }
inline uint64_t reserve_rows(const Config& c, const Request& r);
inline uint32_t reserve_pages(const Config& c, const Request& r);
// rows a session holds from its admission: its prompt, the token the tail samples and the guard, which covers the prompt
// tail, its chain steps and the first round. A round at `length` tokens (prompt + output) writes rows below
// length + 2 * mtp - 1 (verify, first-form draft, chain), DFlash2 below length + 7, so round_rows = length + guard covers
// it (Config::validate keeps guard >= 2 * mtp, and DFlash2 builds set 14). Later rows come by growth (Driver::step).
inline uint64_t round_rows(const Config& c, uint64_t length);
inline uint64_t grant_rows(const Config& c, const Request& r);
inline uint32_t grant_pages(const Config& c, const Request& r);
// the pages a grant may list: what it needs now, its worst case, or (window runs past R) its worst case and the window
inline bool grant_pages_ok(const Config& c, const Request& r, uint32_t pages);
// a session's pages, as the control link carries them. Each rank's pool (pool.h) holds the list and fills it the same
// way. list_hash names the list, so a rank whose list differs fails at its grant. A run grows (Scheduler::grow) between
// events on every rank alike: same table version, more pages, a new hash.
struct Run {
    uint32_t first_page = 0, pages = 0, table_version = 0;
    uint64_t list_hash = 0;
    bool operator==(const Run& b) const {
        return first_page == b.first_page && pages == b.pages && table_version == b.table_version && list_hash == b.list_hash;
    }
    bool operator!=(const Run& b) const { return !(*this == b); }
};
// Parked (Config::park): a finished session whose slot keeps its KV for a follow-up. Not resident: it takes no step,
// does not count toward limit(), and leaves only by a grant (the follow-up's, or a new chat's that evicts it)
enum class Phase { Free, Prefill, Decode, Retiring, Parked };
struct Slot {
    Key key; Request req; Run run;
    Phase phase = Phase::Free;
    uint32_t prefilled = 0, emitted = 0;
    uint32_t inserted = 0;   // tokens the request's insert rule put in (s2_batch.h): in the digest, so the ranks agree it fired
    uint32_t deferred = 0;   // prefill quanta this session was passed over since it was last served (Scheduler::next): in the
                             // digest too, so a rank that counts differently fails at the grant rather than a step later
    bool packable = false;    // fresh, ordinary DS41 request (never a restored/reused/steered prompt)
    bool pack_tail = false;   // selected by one packed W pass; only these keys may join its tail
    uint64_t parked = 0;      // Config::park: the order it was parked in (1, 2, ..), so the oldest is evicted first
};
// prompt rows a session still has to prefill; 0 means only its prompt tail is left
inline uint32_t prefill_left(const Slot& s);
// the order this quantum's prefill candidates are served in. Owed (passed over PREFILL_DEFER_MAX times) outranks not
// owed; among the owed the largest count wins, and among equals the shortest prompt. Replicated state only: every rank
// holds the same slots and counters, so every rank names the same session, which the event digest then confirms.
inline bool prefill_first(const Slot& a, const Slot& b);
// reuse: prompt rows the run already holds from a retained session (pool.h match), so prefill starts there
// reuse: rows the prefill skips. restored: 0, or 1 + the tier those rows come from (pool.h add_tier). counted: the
// prefixes the request counted toward (pool.h count)
struct Grant { Key key; Request req; Run run; uint32_t reuse = 0, restored = 0, counted = 0; };
// a Resize event: every rank takes pool tier `tier` to `segs` segments of its group lease on a KV node, `slots` blocks
// (s2/kv_share.h share_apply), at the commit, so every rank's pool queues the same copies. txn: the node's transaction
// for it (each rank names it in its GROW or SHRINK)
// tier FLASH_KV_TIER (GR8_GLM53F_KV_CHUNKS, runtime/flash_kv.h): the Flash rank's GPU KV chunks instead, `segs` = the chunks
// online (slots and txn unused)
constexpr uint32_t FLASH_KV_TIER = 0xFFFFFFFFu;
// tier FLASH_CKPT_TIER (GR8_GLM53F_CACHE on a group lease, s2/flash_checkpoint.cuh): the Flash checkpoint store's lease
constexpr uint32_t FLASH_CKPT_TIER = 0xFFFFFFFEu;
struct Resize {
    uint32_t tier = 0;
    uint64_t segs = 0, slots = 0, txn = 0;
    bool operator==(const Resize& r) const { return tier == r.tier && segs == r.segs && slots == r.slots && txn == r.txn; }
};
// a live session's run takes the smaller R: its slot, that R, and its run after the change (same table version and
// length, pool.h shrink)
struct Shrink {
    Key key; uint32_t r_rows = 0; Run run;
    bool operator==(const Shrink& b) const { return key == b.key && r_rows == b.r_rows && run == b.run; }
};
struct Commit {
    uint64_t epoch = 0, event = 0, boundary = 0, before_hash = 0;
    std::optional<Key> cancel;
    std::optional<Grant> admit; // empty if FIFO head still cannot fit after reclaim
    // hold (an admit only): every rank stays stopped at this boundary after applying, and the head's next notice names
    // the SAME boundary (event_gate.h EventGate::applied), so queued admissions land one after another at one boundary
    // instead of one per NOTICE_MIN boundaries. A commit without hold releases the boundary. Never read by apply().
    bool hold = false;
    // a Resize event's change (no cancel, grant or hold); Driver::apply passes it to DriverHooks::resize
    std::optional<Resize> resize;
    // an admission's room: a decoding session takes the smaller R (Config::r2_rows) before the grant, which may then
    // use the pages it gave up (Driver::apply: DriverHooks::shrink before bind)
    std::optional<Shrink> shrink;
};
struct Changes {
    std::vector<Run> freed;
    std::optional<Grant> admitted;
};
enum class Kind { Idle, Prefill, PromptTail, Decode, PrefillPack, PromptTailPack };
struct Quantum {
    Kind kind = Kind::Idle;
    uint64_t boundary = 0;
    std::array<Key, MAX_SESSIONS> keys{};
    unsigned count = 0;
    uint32_t first = 0, rows = 0;
    std::array<uint32_t, MAX_SESSIONS> firsts{}, row_counts{}; // pack: per key; scalar first/rows stay the solo contract
    bool solo() const { return count == 1; }
};
struct Completion { Key key; uint32_t emitted = 0; bool finished = false; uint32_t inserted = 0; };   // inserted: in total

// scheduler-rs mirrors these types: the same layout, checked here
#define G6S2_SAME(T, C) static_assert(sizeof(T) == sizeof(C) && alignof(T) == alignof(C), #T " layout");
#define G6S2_AT(T, C, f) static_assert(offsetof(T, f) == offsetof(C, f), #T "::" #f " offset");
G6S2_SAME(Key, g6sch_key) G6S2_AT(Key, g6sch_key, slot) G6S2_AT(Key, g6sch_key, generation)
G6S2_SAME(g6rt::steering::Identity, g6sch_identity) G6S2_AT(g6rt::steering::Identity, g6sch_identity, scope)
G6S2_AT(g6rt::steering::Identity, g6sch_identity, schedule_digest) G6S2_AT(g6rt::steering::Identity, g6sch_identity, scale_bits)
G6S2_AT(g6rt::steering::Identity, g6sch_identity, slot)
G6S2_SAME(Request, g6sch_request) G6S2_AT(Request, g6sch_request, steering) G6S2_AT(Request, g6sch_request, r_rows)
G6S2_SAME(Run, g6sch_run) G6S2_AT(Run, g6sch_run, table_version) G6S2_AT(Run, g6sch_run, list_hash)
G6S2_SAME(Phase, uint32_t)
G6S2_SAME(Slot, g6sch_slot) G6S2_AT(Slot, g6sch_slot, req) G6S2_AT(Slot, g6sch_slot, run) G6S2_AT(Slot, g6sch_slot, phase)
G6S2_AT(Slot, g6sch_slot, deferred) G6S2_AT(Slot, g6sch_slot, packable) G6S2_AT(Slot, g6sch_slot, pack_tail)
G6S2_AT(Slot, g6sch_slot, parked)
G6S2_SAME(Grant, g6sch_grant) G6S2_AT(Grant, g6sch_grant, run) G6S2_AT(Grant, g6sch_grant, reuse)
G6S2_AT(Grant, g6sch_grant, counted)
G6S2_SAME(Shrink, g6sch_shrink) G6S2_AT(Shrink, g6sch_shrink, r_rows) G6S2_AT(Shrink, g6sch_shrink, run)
G6S2_SAME(Completion, g6sch_completion) G6S2_AT(Completion, g6sch_completion, finished)
G6S2_AT(Completion, g6sch_completion, inserted)
#undef G6S2_SAME
#undef G6S2_AT
static_assert(MAX_SESSIONS <= G6SCH_QMAX, "scheduler-rs quantum size");
namespace sch {
template <class C, class T> inline const C* c(const T* p) { return reinterpret_cast<const C*>(p); }
template <class C, class T> inline C* m(T* p) { return reinterpret_cast<C*>(p); }
inline g6sch_quantum to_rs(const Quantum& q) {
    g6sch_quantum o{};
    o.kind = uint32_t(q.kind); o.boundary = q.boundary; o.count = q.count; o.first = q.first; o.rows = q.rows;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        o.keys[i] = {q.keys[i].epoch, q.keys[i].slot, q.keys[i].generation};
        o.firsts[i] = q.firsts[i]; o.row_counts[i] = q.row_counts[i];
    }
    return o;
}
inline Quantum from_rs(const g6sch_quantum& o) {
    Quantum q;
    q.kind = Kind(o.kind); q.boundary = o.boundary; q.count = o.count; q.first = o.first; q.rows = o.rows;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        q.keys[i] = {o.keys[i].epoch, o.keys[i].slot, o.keys[i].generation};
        q.firsts[i] = o.firsts[i]; q.row_counts[i] = o.row_counts[i];
    }
    return q;
}
// the Rust side agrees on every size and constant both sides define
inline void check_abi() {
    static const bool ok = [] {
        const std::pair<uint32_t, uint64_t> want[] = {
            {G6SCH_ABI_SLOT, sizeof(g6sch_slot)}, {G6SCH_ABI_CONFIG, sizeof(g6sch_config)},
            {G6SCH_ABI_QUANTUM, sizeof(g6sch_quantum)}, {G6SCH_ABI_STATE, sizeof(g6sch_state)},
            {G6SCH_ABI_COMMIT, sizeof(g6sch_commit)}, {G6SCH_ABI_COMPLETION, sizeof(g6sch_completion)},
            {G6SCH_ABI_GRANT, sizeof(g6sch_grant)}, {G6SCH_ABI_PAGE, PAGE},
            {G6SCH_ABI_PREFILL_DEFER_MAX, PREFILL_DEFER_MAX}, {G6SCH_ABI_NOTICE_MIN, NOTICE_MIN},
            {G6SCH_ABI_NOTICE_MAX, NOTICE_MAX}, {G6SCH_ABI_NOTICE_ROWS, NOTICE_ROWS},
            {G6SCH_ABI_MAX_DECODE_ROUNDS, MAX_DECODE_ROUNDS}, {G6SCH_ABI_PACK_MAX_MEMBERS, PACK_MAX_MEMBERS},
            {G6SCH_ABI_QMAX, G6SCH_QMAX}};
        for (const auto& [what, v] : want) if (g6sch_abi(what) != v) return false;
        return true;
    }();
    require(ok, "scheduler-rs: struct sizes or constants differ from scheduler.h");
}
} // namespace sch
inline uint64_t reserve_rows(const Config& c, const Request& r) {
    const auto x = c.scalars();
    return sch_out<uint64_t>([&](uint64_t* v) { return g6sch_reserve_rows(&x, sch::c<g6sch_request>(&r), v); });
}
inline uint32_t reserve_pages(const Config& c, const Request& r) {
    const auto x = c.scalars();
    return sch_out<uint32_t>([&](uint32_t* v) { return g6sch_reserve_pages(&x, sch::c<g6sch_request>(&r), v); });
}
inline uint64_t round_rows(const Config& c, uint64_t length) { const auto x = c.scalars(); return g6sch_round_rows(&x, length); }
inline uint64_t grant_rows(const Config& c, const Request& r) {
    const auto x = c.scalars();
    return sch_out<uint64_t>([&](uint64_t* v) { return g6sch_grant_rows(&x, sch::c<g6sch_request>(&r), v); });
}
inline uint32_t grant_pages(const Config& c, const Request& r) {
    const auto x = c.scalars();
    return sch_out<uint32_t>([&](uint32_t* v) { return g6sch_grant_pages(&x, sch::c<g6sch_request>(&r), v); });
}
inline bool grant_pages_ok(const Config& c, const Request& r, uint32_t pages) {
    const auto x = c.scalars();
    return sch_out<bool>([&](bool* v) { return g6sch_grant_pages_ok(&x, sch::c<g6sch_request>(&r), pages, v); });
}
inline uint32_t prefill_left(const Slot& s) { return g6sch_prefill_left(sch::c<g6sch_slot>(&s)); }
inline bool prefill_first(const Slot& a, const Slot& b) { return g6sch_prefill_first(sch::c<g6sch_slot>(&a), sch::c<g6sch_slot>(&b)); }

// The scheduler's rules are engine/scheduler-rs/src/scheduler.rs; this class owns the state they act on.
class Scheduler {
    Config c_;
    g6sch_config x_;
    std::array<Slot, MAX_SESSIONS> s_{};
    g6sch_state st_{};
    const g6sch_slot* rs() const { return sch::c<g6sch_slot>(s_.data()); }
    g6sch_slot* rs() { return sch::m<g6sch_slot>(s_.data()); }
public:
    explicit Scheduler(Config c) : c_(c), x_(c.scalars()) {
        sch::check_abi();
        c_.validate();
        st_.want_decode = true;
        g6sch_init(&x_, rs(), MAX_SESSIONS);
    }
    const Config& config() const { return c_; }
    const auto& slots() const { return s_; }
    uint64_t boundary() const { return st_.boundary; }
    uint64_t next_event() const { return st_.event + 1; }
    uint64_t revision() const { return st_.revision; }
    bool busy() const { return st_.busy; }
    void idle_boundary(uint64_t target) { check(g6sch_idle_boundary(rs(), MAX_SESSIONS, &st_, target)); }   // event-only: no model work
    void fail() { st_.failed = true; }
    // the pool's digest (pool.h RunPool::digest) at this event boundary, so the ranks also agree on their pages
    void pool_digest(uint64_t d) { st_.pool = d; }
    unsigned resident() const { return g6sch_resident(rs(), MAX_SESSIONS); }
    unsigned retiring() const { return g6sch_retiring(rs(), MAX_SESSIONS); }
    unsigned parked() const { return g6sch_parked(rs(), MAX_SESSIONS); }
    std::optional<Key> find(uint64_t id) const {
        g6sch_key k;
        if (!g6sch_find(rs(), MAX_SESSIONS, id, &k)) return {};
        return Key{k.epoch, k.slot, k.generation};
    }
    // park: a parked slot (or a Retiring one, which the commit parks) the grant takes for a follow-up that extends it
    Key next_key(std::optional<Key> cancel = {}, std::optional<unsigned> park = {}) const {
        g6sch_key k;
        check(g6sch_next_key(&x_, rs(), MAX_SESSIONS, cancel ? sch::c<g6sch_key>(&*cancel) : nullptr, bool(park),
                             park.value_or(0), &k));
        return Key{k.epoch, k.slot, k.generation};
    }
    std::vector<Run> reclaimable(std::optional<Key> cancel = {}) const {
        std::vector<Run> out(MAX_SESSIONS);
        out.resize(g6sch_reclaimable(&x_, rs(), MAX_SESSIONS, cancel ? sch::c<g6sch_key>(&*cancel) : nullptr, sch::m<g6sch_run>(out.data())));
        return out;
    }
    uint64_t digest() const { // event diagnostics only; NEVER called by next()/complete()
        uint64_t h = g6sch_digest_head(&x_, &st_);
        g6rt::steering::hash_boot(h, c_.steering);
        return g6sch_digest_slots(&x_, rs(), MAX_SESSIONS, h);
    }
    Changes apply(const Commit& e) {
        g6sch_commit x{};
        x.epoch = e.epoch; x.event = e.event; x.boundary = e.boundary; x.before_hash = e.before_hash;
        if ((x.has_cancel = bool(e.cancel))) std::memcpy(&x.cancel, &*e.cancel, sizeof x.cancel);
        if ((x.has_admit = bool(e.admit))) std::memcpy(&x.admit, &*e.admit, sizeof x.admit);
        if ((x.has_shrink = bool(e.shrink))) std::memcpy(&x.shrink, &*e.shrink, sizeof x.shrink);
        Changes d;
        d.freed.resize(MAX_SESSIONS);
        size_t n = 0; bool admitted = false;
        check(g6sch_apply(&x_, rs(), MAX_SESSIONS, &st_, &x, digest(), sch::m<g6sch_run>(d.freed.data()), &n, &admitted));
        d.freed.resize(n);
        if (admitted) d.admitted = e.admit;
        return d; // allocator adapter now applies these canonical changes; no local allocation choice here
    }
    // a decoding session's run grew (Driver::step, every rank alike, between events)
    void grow(const Key& k, const Run& r) {
        check(g6sch_grow(&x_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_key>(&k), sch::c<g6sch_run>(&r)));
    }
    // the decoding sessions in ascending request id, as next() orders them
    std::vector<Key> decoding() const {
        std::vector<Key> out(MAX_SESSIONS);
        out.resize(g6sch_decoding(rs(), MAX_SESSIONS, sch::m<g6sch_key>(out.data())));
        return out;
    }
    // The alternation flag after a completed quantum of kind k: a prefill (or tail) asks for a decode round next; a decode
    // round asks for the prefill unless fewer than decode_rounds rounds have run since the last prefill. horizon() runs
    // the same function forward, so the notice target agrees with the steps the ranks will take.
    static void alternate(const Config& c, Kind k, bool& want, uint32_t& run) {
        const auto x = c.scalars();
        g6sch_alternate(&x, uint32_t(k), &want, &run);
    }
    // The next quantum's shape from slot state `s` and the alternation flag, without touching the scheduler. `stalled`:
    // decoding sessions left out of this round (the driver's); horizon() passes none.
    static Quantum plan(const Config& c, const std::array<Slot, MAX_SESSIONS>& s, bool want_decode,
                        const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        const auto x = c.scalars();
        g6sch_quantum q;
        g6sch_plan(&x, sch::c<g6sch_slot>(s.data()), MAX_SESSIONS, want_decode, stalled.data(), &q);
        return sch::from_rs(q);
    }
    // stalled[slot]: a decoding session left out of this round (its run could not grow); it keeps its place
    Quantum next(const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        g6sch_quantum q;
        check(g6sch_next(&x_, rs(), MAX_SESSIONS, &st_, stalled.data(), &q));
        return sch::from_rs(q);
    }
    // Advance a slot copy the way the completions do (prefilled, phase, the deferral counters), for horizon()'s projection
    static void advance(std::array<Slot, MAX_SESSIONS>& sim, const Quantum& q) {
        const g6sch_quantum x = sch::to_rs(q);
        g6sch_advance(sch::m<g6sch_slot>(sim.data()), MAX_SESSIONS, &x);
    }
    // The boundary a notice anchored now should target (service.h event): the first boundary whose work after the one
    // in flight covers NOTICE_ROWS prefill rows, within [boundary + NOTICE_MIN, boundary + notice_max]. A pure function
    // of the slots and the alternation flag, so every rank names the same boundary.
    uint64_t horizon() const { return g6sch_horizon(&x_, rs(), MAX_SESSIONS, &st_); }
    // a just-admitted session that runs its whole prompt as decode steps (prompt logprobs): straight to PromptTail
    void skip_prefill(const Key& k) { check(g6sch_skip_prefill(rs(), MAX_SESSIONS, &st_, sch::c<g6sch_key>(&k))); }
    void complete_prefill() { check(g6sch_complete_prefill(&x_, rs(), MAX_SESSIONS, &st_)); }   // both lanes and draft prefill
    void complete_prefill_pack() { check(g6sch_complete_prefill_pack(&x_, rs(), MAX_SESSIONS, &st_)); }
    void complete_tail(const Completion& r) {
        check(g6sch_complete_tail(&x_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(&r)));
    }
    void complete_tail_pack(const std::vector<Completion>& r) {
        check(g6sch_complete_tail_pack(&x_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(r.data()), r.size()));
    }
    void complete_decode(const std::vector<Completion>& r) {
        check(g6sch_complete_decode(&x_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(r.data()), r.size()));
    }
};

enum class Stall { None, Slots, Capacity };
struct AdmissionStatus { Stall why = Stall::None; uint32_t requested = 0, room = 0; };
// Rank0 ingress FIFO. Page choice stays in the pool (pool.h).
class AdmissionQueue {
    Config c_; std::deque<Request> q_;
    uint64_t last_revision_ = UINT64_MAX, last_head_ = UINT64_MAX, last_id_ = 0;
    bool have_id_ = false;
public:
    uint64_t stalls = 0;
    explicit AdmissionQueue(Config c) : c_(c) { c_.validate(); }
    void push(Request r) {
        (void)reserve_pages(c_, r);
        require(!have_id_ || r.id > last_id_, "native request ordinals must increase");
        for (auto x : q_) require(x.id != r.id, "duplicate queued id");
        q_.push_back(r); last_id_ = r.id; have_id_ = true;
    }
    bool cancel(uint64_t id) {
        auto it = std::find_if(q_.begin(), q_.end(), [&](const Request& r) { return r.id == id; });
        if (it == q_.end()) return false;
        q_.erase(it); return true;
    }
    const Request* front() const { return q_.empty() ? nullptr : &q_.front(); }
    size_t size() const { return q_.size(); }
    void admitted(uint64_t id) { require(!q_.empty() && q_.front().id == id, "FIFO admission violated"); q_.pop_front(); }
    // fits: the pool can admit the head (RunPool::fits); room: the pages one reservation can take now (RunPool::room)
    std::optional<AdmissionStatus> check(uint64_t revision, bool slot, bool fits, uint32_t room) {
        if (q_.empty() || (last_head_ == q_.front().id && last_revision_ == revision)) return {};
        require(room <= c_.pool_rows/PAGE, "invalid allocator view");
        last_head_ = q_.front().id; last_revision_ = revision;
        AdmissionStatus s; s.requested = grant_pages(c_, q_.front()); s.room = room;
        s.why = !slot ? Stall::Slots : !fits ? Stall::Capacity : Stall::None;
        if (s.why != Stall::None) ++stalls;
        return s;
    }
};
} // namespace g6s2
