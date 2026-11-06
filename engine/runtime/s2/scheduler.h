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
#include <memory>
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
constexpr unsigned PACK_MAX_MEMBERS = unsigned(g6mod::INFO.pack_members);
// The control link's version: the frame magic is the numeric base "G6S2E0" + this version (control_protocol.h), and the bootstrap agreement
// carries it (bootstrap.h), so ranks built with different commit frames refuse at bootstrap, not at the first event.
// 6 adds Commit::hold. 7 adds the Resize event (Commit::resize). 9 adds Ack::room (8 is the fail-soft store's, #159).
// 10 adds the steering profile slot. 11 adds Request::r_rows. 12 adds Commit::shrink. 13 adds 32 session slots and the
// DS41 prefill packs. 14 adds the image digest to a request's identity (DS41 image input).
constexpr unsigned CONTROL_VERSION = 14;
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
// The scheduler's settings: the scalars are g6sch_config (g6sched.h, which documents each), shared with scheduler-rs as
// they are; the steering receipt stays C++.
struct Config : g6sch_config {
    g6rt::steering::BootConfig steering{};
    Config() : g6sch_config{} {
        epoch = 1; pool_rows = 614400; max_rows = 24; max_sessions = 8; mtp = 2;
        guard_tokens = UINT32_MAX; prefill_rows = 4096; two_lanes = true;
        cp_after = UINT64_MAX; notice_max = uint32_t(NOTICE_MAX);
    }
    uint32_t guard() const { return g6sch_guard(this); }
    unsigned limit() const { return g6sch_limit(this); }
    void validate() const {
        check(g6sch_validate(this, MAX_SESSIONS, g6mod::has(G6_CAP_MTP0), g6mod::has(G6_CAP_PACK), g6mod::has(G6_CAP_PARK)));
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
G6S2_AT(g6rt::steering::Identity, g6sch_identity, slot) G6S2_AT(g6rt::steering::Identity, g6sch_identity, images)
G6S2_AT(g6rt::steering::Identity, g6sch_identity, image_digest)
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
inline g6sch_commit to_rs(const Commit& e) {
    g6sch_commit x{};
    x.epoch = e.epoch; x.event = e.event; x.boundary = e.boundary; x.before_hash = e.before_hash;
    if ((x.has_cancel = bool(e.cancel))) std::memcpy(&x.cancel, &*e.cancel, sizeof x.cancel);
    if ((x.has_admit = bool(e.admit))) std::memcpy(&x.admit, &*e.admit, sizeof x.admit);
    if ((x.has_shrink = bool(e.shrink))) std::memcpy(&x.shrink, &*e.shrink, sizeof x.shrink);
    return x;
}
// the Rust side agrees on every shared struct layout (g6sch_layout) and every constant both sides define
inline void check_abi() {
    static const bool ok = [] {
        if (g6sch_layout() != g6sch_layout_cpp()) return false;
        const std::pair<uint32_t, uint64_t> want[] = {
            {G6SCH_ABI_PAGE, PAGE},
            {G6SCH_ABI_PREFILL_DEFER_MAX, PREFILL_DEFER_MAX}, {G6SCH_ABI_NOTICE_MIN, NOTICE_MIN},
            {G6SCH_ABI_NOTICE_MAX, NOTICE_MAX}, {G6SCH_ABI_NOTICE_ROWS, NOTICE_ROWS},
            {G6SCH_ABI_MAX_DECODE_ROUNDS, MAX_DECODE_ROUNDS}, {G6SCH_ABI_PACK_MAX_MEMBERS, PACK_MAX_MEMBERS},
            {G6SCH_ABI_QMAX, G6SCH_QMAX}, {G6SCH_ABI_PARSE_FN, G6SCH_PARSE_FN}};
        for (const auto& [what, v] : want) if (g6sch_abi(what) != v) return false;
        return true;
    }();
    require(ok, "scheduler-rs: struct layouts or constants differ from g6sched.h and scheduler.h");
}
} // namespace sch
inline uint64_t reserve_rows(const Config& c, const Request& r) {
    return sch_out<uint64_t>([&](uint64_t* v) { return g6sch_reserve_rows(&c, sch::c<g6sch_request>(&r), v); });
}
inline uint32_t reserve_pages(const Config& c, const Request& r) {
    return sch_out<uint32_t>([&](uint32_t* v) { return g6sch_reserve_pages(&c, sch::c<g6sch_request>(&r), v); });
}
inline uint64_t round_rows(const Config& c, uint64_t length) { return g6sch_round_rows(&c, length); }
inline uint64_t grant_rows(const Config& c, const Request& r) {
    return sch_out<uint64_t>([&](uint64_t* v) { return g6sch_grant_rows(&c, sch::c<g6sch_request>(&r), v); });
}
inline uint32_t grant_pages(const Config& c, const Request& r) {
    return sch_out<uint32_t>([&](uint32_t* v) { return g6sch_grant_pages(&c, sch::c<g6sch_request>(&r), v); });
}
inline bool grant_pages_ok(const Config& c, const Request& r, uint32_t pages) {
    return sch_out<bool>([&](bool* v) { return g6sch_grant_pages_ok(&c, sch::c<g6sch_request>(&r), pages, v); });
}
inline uint32_t prefill_left(const Slot& s) { return g6sch_prefill_left(sch::c<g6sch_slot>(&s)); }
inline bool prefill_first(const Slot& a, const Slot& b) { return g6sch_prefill_first(sch::c<g6sch_slot>(&a), sch::c<g6sch_slot>(&b)); }

// The scheduler's state, read only: a Scheduler's own, or the Rust driver's (Driver::state). The rules are
// engine/scheduler-rs/src/scheduler.rs.
class SchedulerView {
    const Config* vc_ = nullptr;
    const g6sch_slot* vs_ = nullptr;
    const g6sch_state* vst_ = nullptr;
public:
    SchedulerView() = default;
    // s: MAX_SESSIONS slots
    SchedulerView(const Config* c, const g6sch_slot* s, const g6sch_state* st) : vc_(c), vs_(s), vst_(st) {}
    const Config& config() const { return *vc_; }
    const std::array<Slot, MAX_SESSIONS>& slots() const { return *reinterpret_cast<const std::array<Slot, MAX_SESSIONS>*>(vs_); }
    uint64_t boundary() const { return vst_->boundary; }
    uint64_t next_event() const { return vst_->event + 1; }
    uint64_t revision() const { return vst_->revision; }
    bool busy() const { return vst_->busy; }
    unsigned resident() const { return g6sch_resident(vs_, MAX_SESSIONS); }
    unsigned retiring() const { return g6sch_retiring(vs_, MAX_SESSIONS); }
    unsigned parked() const { return g6sch_parked(vs_, MAX_SESSIONS); }
    std::optional<Key> find(uint64_t id) const {
        g6sch_key k;
        if (!g6sch_find(vs_, MAX_SESSIONS, id, &k)) return {};
        return Key{k.epoch, k.slot, k.generation};
    }
    // park: a parked slot (or a Retiring one, which the commit parks) the grant takes for a follow-up that extends it
    Key next_key(std::optional<Key> cancel = {}, std::optional<unsigned> park = {}) const {
        g6sch_key k;
        check(g6sch_next_key(vc_, vs_, MAX_SESSIONS, cancel ? sch::c<g6sch_key>(&*cancel) : nullptr, bool(park),
                             park.value_or(0), &k));
        return Key{k.epoch, k.slot, k.generation};
    }
    std::vector<Run> reclaimable(std::optional<Key> cancel = {}) const {
        std::vector<Run> out(MAX_SESSIONS);
        out.resize(g6sch_reclaimable(vc_, vs_, MAX_SESSIONS, cancel ? sch::c<g6sch_key>(&*cancel) : nullptr, sch::m<g6sch_run>(out.data())));
        return out;
    }
    uint64_t digest() const { // event diagnostics only; NEVER called by next()/complete()
        uint64_t h = g6sch_digest_head(vc_, vst_);
        g6rt::steering::hash_boot(h, vc_->steering);
        return g6sch_digest_slots(vc_, vs_, MAX_SESSIONS, h);
    }
    // the decoding sessions in ascending request id, as next() orders them (tests only; the driver's is Rust's)
    std::vector<Key> decoding() const {
        std::vector<Key> out(MAX_SESSIONS);
        out.resize(g6sch_decoding(vs_, MAX_SESSIONS, sch::m<g6sch_key>(out.data())));
        return out;
    }
    // The boundary a notice anchored now should target (service.h event): the first boundary whose work after the one
    // in flight covers NOTICE_ROWS prefill rows, within [boundary + NOTICE_MIN, boundary + notice_max]. A pure function
    // of the slots and the alternation flag, so every rank names the same boundary.
    uint64_t horizon() const { return g6sch_horizon(vc_, vs_, MAX_SESSIONS, vst_); }
};

// A scheduler that owns its state (tests and tools; the service's is the Rust driver's).
class Scheduler : public SchedulerView {
    Config c_;
    std::array<Slot, MAX_SESSIONS> s_{};
    g6sch_state st_{};
    g6sch_slot* rs() { return sch::m<g6sch_slot>(s_.data()); }
    // the view points at this object's own state, again after a copy
    void point() { static_cast<SchedulerView&>(*this) = SchedulerView(&c_, sch::c<g6sch_slot>(s_.data()), &st_); }
public:
    explicit Scheduler(Config c) : c_(c) {
        sch::check_abi();
        c_.validate();
        st_.want_decode = true;
        g6sch_init(&c_, rs(), MAX_SESSIONS);
        point();
    }
    Scheduler(const Scheduler& o) : SchedulerView(), c_(o.c_), s_(o.s_), st_(o.st_) { point(); }
    Scheduler& operator=(const Scheduler& o) { c_ = o.c_; s_ = o.s_; st_ = o.st_; point(); return *this; }
    void idle_boundary(uint64_t target) { check(g6sch_idle_boundary(rs(), MAX_SESSIONS, &st_, target)); }   // event-only: no model work
    void fail() { st_.failed = true; }
    // the pool's digest (pool.h RunPool::digest) at this event boundary, so the ranks also agree on their pages
    void pool_digest(uint64_t d) { st_.pool = d; }
    Changes apply(const Commit& e) {
        const g6sch_commit x = sch::to_rs(e);
        Changes d;
        d.freed.resize(MAX_SESSIONS);
        size_t n = 0; bool admitted = false;
        check(g6sch_apply(&c_, rs(), MAX_SESSIONS, &st_, &x, digest(), sch::m<g6sch_run>(d.freed.data()), &n, &admitted));
        d.freed.resize(n);
        if (admitted) d.admitted = e.admit;
        return d; // allocator adapter now applies these canonical changes; no local allocation choice here
    }
    // a decoding session's run grew (Driver::step, every rank alike, between events)
    void grow(const Key& k, const Run& r) {
        check(g6sch_grow(&c_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_key>(&k), sch::c<g6sch_run>(&r)));
    }
    // The alternation flag after a completed quantum of kind k: a prefill (or tail) asks for a decode round next; a decode
    // round asks for the prefill unless fewer than decode_rounds rounds have run since the last prefill. horizon() runs
    // the same function forward, so the notice target agrees with the steps the ranks will take.
    static void alternate(const Config& c, Kind k, bool& want, uint32_t& run) {
        g6sch_alternate(&c, uint32_t(k), &want, &run);
    }
    // The next quantum's shape from slot state `s` and the alternation flag, without touching the scheduler. `stalled`:
    // decoding sessions left out of this round (the driver's); horizon() passes none.
    static Quantum plan(const Config& c, const std::array<Slot, MAX_SESSIONS>& s, bool want_decode,
                        const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        g6sch_quantum q;
        g6sch_plan(&c, sch::c<g6sch_slot>(s.data()), MAX_SESSIONS, want_decode, stalled.data(), &q);
        return sch::from_rs(q);
    }
    // stalled[slot]: a decoding session left out of this round (its run could not grow); it keeps its place
    Quantum next(const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        g6sch_quantum q;
        check(g6sch_next(&c_, rs(), MAX_SESSIONS, &st_, stalled.data(), &q));
        return sch::from_rs(q);
    }
    // Advance a slot copy the way the completions do (prefilled, phase, the deferral counters), for horizon()'s projection
    static void advance(std::array<Slot, MAX_SESSIONS>& sim, const Quantum& q) {
        const g6sch_quantum x = sch::to_rs(q);
        g6sch_advance(sch::m<g6sch_slot>(sim.data()), MAX_SESSIONS, &x);
    }
    // a just-admitted session that runs its whole prompt as decode steps (prompt logprobs): straight to PromptTail
    void skip_prefill(const Key& k) { check(g6sch_skip_prefill(rs(), MAX_SESSIONS, &st_, sch::c<g6sch_key>(&k))); }
    void complete_prefill() { check(g6sch_complete_prefill(&c_, rs(), MAX_SESSIONS, &st_)); }   // both lanes and draft prefill
    void complete_prefill_pack() { check(g6sch_complete_prefill_pack(&c_, rs(), MAX_SESSIONS, &st_)); }
    void complete_tail(const Completion& r) {
        check(g6sch_complete_tail(&c_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(&r)));
    }
    void complete_tail_pack(const std::vector<Completion>& r) {
        check(g6sch_complete_tail_pack(&c_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(r.data()), r.size()));
    }
    void complete_decode(const std::vector<Completion>& r) {
        check(g6sch_complete_decode(&c_, rs(), MAX_SESSIONS, &st_, sch::c<g6sch_completion>(r.data()), r.size()));
    }
};

enum class Stall { None, Slots, Capacity };
struct AdmissionStatus { Stall why = Stall::None; uint32_t requested = 0, room = 0; };
// Rank0 ingress FIFO, a Rust-owned queue (engine/scheduler-rs/src/admission.rs). Page choice stays in the pool (pool.h).
class AdmissionQueue {
    std::unique_ptr<g6sch_queue, void (*)(g6sch_queue*)> q_;
public:
    explicit AdmissionQueue(Config c) : q_(nullptr, g6sch_queue_free) {
        sch::check_abi();
        c.validate();
        q_.reset(g6sch_queue_new(&c));
    }
    void push(Request r) { g6s2::check(g6sch_queue_push(q_.get(), sch::c<g6sch_request>(&r))); }
    bool cancel(uint64_t id) { return g6sch_queue_cancel(q_.get(), id); }
    // valid until the queue changes
    const Request* front() const { return sch::c<Request>(g6sch_queue_front(q_.get())); }
    size_t size() const { return g6sch_queue_size(q_.get()); }
    uint64_t stalls() const { return g6sch_queue_stalls(q_.get()); }
    void admitted(uint64_t id) { g6s2::check(g6sch_queue_admitted(q_.get(), id)); }
    // fits: the pool can admit the head (RunPool::fits); room: the pages one reservation can take now (RunPool::room)
    std::optional<AdmissionStatus> check(uint64_t revision, bool slot, bool fits, uint32_t room) {
        bool has = false; g6sch_status st{};
        g6s2::check(g6sch_queue_check(q_.get(), revision, slot, fits, room, &has, &st));
        if (!has) return {};
        return AdmissionStatus{Stall(st.why), st.requested, st.room};
    }
};
} // namespace g6s2
