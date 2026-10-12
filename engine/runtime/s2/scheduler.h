// The S2 scheduler, host-only state. No CUDA, allocator implementation, sampler, or transport.
// A caller may apply Commit ONLY after event-boundary all-rank quiescence (event_gate.h).
// TP4 owns run allocation/claim/release; the batch driver owns tokens, MTP and hidden state.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "../steering.h"

namespace g6s2 {
inline void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
constexpr unsigned MAX_SESSIONS = 8, PAGE = 64;
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
    // max_rows bounds ONE step: a round is one main step of every session's mtp + 1 rows (s2_batch.h Batch::round)
    uint32_t guard() const { return guard_tokens == UINT32_MAX ? 2 * mtp : guard_tokens; }
    unsigned limit() const { return std::min(max_sessions, max_rows / (mtp + 1)); }
    void validate() const {
        require(epoch && pool_rows >= PAGE && pool_rows % PAGE == 0, "bad epoch/pool");
        require(mtp > 0 && mtp <= 7 && max_sessions && max_sessions <= MAX_SESSIONS && limit(), "bad session/row limit");
        require(guard() >= 2 * mtp && prefill_rows, "insufficient guard/prefill rows");
        require(!steering.loaded || (g6rt::steering::valid_profile_name(steering.profile_name) && steering.active_layers &&
                (steering.scope == g6rt::steering::Scope::Shared || steering.scope == g6rt::steering::Scope::Full)),
                "invalid loaded steering receipt");
        require(!steering.forced || steering.loaded, "forced steering needs a loaded profile");
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
};
inline uint64_t reserve_rows(const Config& c, const Request& r) {
    require(r.prompt_tokens && r.max_tokens, "empty request");
    return uint64_t(r.prompt_tokens) + r.max_tokens + c.guard();
}
inline uint32_t reserve_pages(const Config& c, const Request& r) {
    const auto rows = reserve_rows(c, r);
    require(rows <= c.pool_rows, "request exceeds pool including guard");
    return uint32_t((rows + PAGE - 1) / PAGE);
}
// rows a session holds from its admission: its prompt, the token the tail samples and the guard, which covers the prompt
// tail, its chain steps and the first round. A round at `length` tokens (prompt + output) writes rows below
// length + 2 * mtp - 1 (verify, first-form draft, chain), DFlash2 below length + 7, so round_rows = length + guard covers
// it (Config::validate keeps guard >= 2 * mtp, and DFlash2 builds set 14). Later rows come by growth (Driver::step).
inline uint64_t round_rows(const Config& c, uint64_t length) { return length + c.guard(); }
inline uint64_t grant_rows(const Config& c, const Request& r) {
    return std::min(reserve_rows(c, r), round_rows(c, uint64_t(r.prompt_tokens) + 1));
}
inline uint32_t grant_pages(const Config& c, const Request& r) { return uint32_t((grant_rows(c, r) + PAGE - 1) / PAGE); }
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
enum class Phase { Free, Prefill, Decode, Retiring };
struct Slot {
    Key key; Request req; Run run;
    Phase phase = Phase::Free;
    uint32_t prefilled = 0, emitted = 0;
    uint32_t inserted = 0;   // tokens the request's insert rule put in (s2_batch.h): in the digest, so the ranks agree it fired
    uint32_t deferred = 0;   // prefill quanta this session was passed over since it was last served (Scheduler::next): in the
                             // digest too, so a rank that counts differently fails at the grant rather than a step later
};
// prompt rows a session still has to prefill; 0 means only its prompt tail is left
inline uint32_t prefill_left(const Slot& s) { return s.req.prompt_tokens - 1 - s.prefilled; }
// the order this quantum's prefill candidates are served in. Owed (passed over PREFILL_DEFER_MAX times) outranks not
// owed; among the owed the largest count wins, and among equals the shortest prompt. Replicated state only: every rank
// holds the same slots and counters, so every rank names the same session, which the event digest then confirms.
inline bool prefill_first(const Slot& a, const Slot& b) {
    const bool owed_a = a.deferred >= PREFILL_DEFER_MAX, owed_b = b.deferred >= PREFILL_DEFER_MAX;
    if (owed_a != owed_b) return owed_a;                 // a session that has waited its turn goes first
    if (owed_a && a.deferred != b.deferred) return a.deferred > b.deferred;  // both owed: the one that waited longest
    const uint32_t left_a = prefill_left(a), left_b = prefill_left(b);
    if (left_a != left_b) return left_a < left_b;        // fewest remaining prompt rows
    return a.req.id < b.req.id;                          // immutable native ordinal, so the choice is total
}
// reuse: prompt rows the run already holds from a retained session (pool.h match), so prefill starts there
// reuse: rows the prefill skips. restored: 0, or 1 + the tier those rows come from (pool.h add_tier). counted: the
// prefixes the request counted toward (pool.h count)
struct Grant { Key key; Request req; Run run; uint32_t reuse = 0, restored = 0, counted = 0; };
struct Commit {
    uint64_t epoch = 0, event = 0, boundary = 0, before_hash = 0;
    std::optional<Key> cancel;
    std::optional<Grant> admit; // empty if FIFO head still cannot fit after reclaim
};
struct Changes {
    std::vector<Run> freed;
    std::optional<Grant> admitted;
};
enum class Kind { Idle, Prefill, PromptTail, Decode };
struct Quantum {
    Kind kind = Kind::Idle;
    uint64_t boundary = 0;
    std::array<Key, MAX_SESSIONS> keys{};
    unsigned count = 0;
    uint32_t first = 0, rows = 0;
    bool solo() const { return count == 1; }
};
struct Completion { Key key; uint32_t emitted = 0; bool finished = false; uint32_t inserted = 0; };   // inserted: in total

class Scheduler {
    Config c_;
    std::array<Slot, MAX_SESSIONS> s_{};
    uint64_t boundary_ = 0, event_ = 0, revision_ = 0, last_id_ = 0, pool_ = 0;
    bool have_id_ = false;
    bool want_decode_ = true, busy_ = false, failed_ = false;
    Quantum work_;
    Slot& slot(const Key& k) {
        require(k.slot < MAX_SESSIONS && s_[k.slot].key == k && s_[k.slot].phase != Phase::Free, "stale session key");
        return s_[k.slot];
    }
    void finish_slot(Slot& s, const Completion& r) {
        require(s.key == r.key && r.emitted <= s.req.max_tokens - s.emitted, "invalid completion");
        s.emitted += r.emitted;
        require(r.inserted >= s.inserted, "insert count went back");
        s.inserted = r.inserted;
        if (r.finished || s.emitted == s.req.max_tokens) { s.phase = Phase::Retiring; ++revision_; }
        else s.phase = Phase::Decode;
    }
    void completed() {
        boundary_ = work_.boundary; busy_ = false; want_decode_ = work_.kind != Kind::Decode;
        // The prefill choice below reads these counters, so they advance here on replicated state alone: the session this
        // quantum served has had its turn, every other prefilling session was passed over once. A PromptTail leaves
        // Phase::Prefill as it completes, so it is never reset here and its rivals are still counted as passed over.
        if (work_.kind == Kind::Prefill || work_.kind == Kind::PromptTail)
            for (auto& s : s_) if (s.phase == Phase::Prefill) s.deferred = s.key == work_.keys[0] ? 0 : s.deferred + 1;
    }
public:
    explicit Scheduler(Config c) : c_(c) {
        c_.validate();
        for (unsigned i = 0; i < MAX_SESSIONS; ++i) s_[i].key = Key{c_.epoch, i, 0};
    }
    const Config& config() const { return c_; }
    const auto& slots() const { return s_; }
    uint64_t boundary() const { return boundary_; }
    uint64_t next_event() const { return event_ + 1; }
    uint64_t revision() const { return revision_; }
    bool busy() const { return busy_; }
    void idle_boundary(uint64_t target) { // event-only: no model work, all sessions already quiescent
        require(!failed_ && !busy_ && !resident() && target >= boundary_, "cannot advance a live boundary");
        boundary_ = target;
    }
    void fail() { failed_ = true; }
    // the pool's digest (pool.h RunPool::digest) at this event boundary, so the ranks also agree on their pages
    void pool_digest(uint64_t d) { pool_ = d; }
    unsigned resident() const {
        unsigned n = 0; for (const auto& s : s_) n += s.phase != Phase::Free && s.phase != Phase::Retiring; return n;
    }
    unsigned retiring() const {
        unsigned n = 0; for (const auto& s : s_) n += s.phase == Phase::Retiring; return n;
    }
    std::optional<Key> find(uint64_t id) const {
        for (const auto& s : s_) if (s.phase != Phase::Free && s.req.id == id) return s.key;
        return {};
    }
    Key next_key(std::optional<Key> cancel = {}) const {
        for (const auto& s : s_) if (s.phase == Phase::Free || s.phase == Phase::Retiring || (cancel && s.key == *cancel)) {
            require(s.key.generation != UINT32_MAX, "session generation exhausted");
            return Key{c_.epoch, s.key.slot, s.key.generation + 1};
        }
        throw std::runtime_error("no session slot");
    }
    std::vector<Run> reclaimable(std::optional<Key> cancel = {}) const {
        std::vector<Run> out;
        for (const auto& s : s_) if (s.phase == Phase::Retiring || (cancel && s.phase != Phase::Free && s.key == *cancel)) out.push_back(s.run);
        return out;
    }
    uint64_t digest() const { // event diagnostics only; NEVER called by next()/complete()
        uint64_t h = 14695981039346656037ull;
        auto add = [&](uint64_t v) { for (unsigned i = 0; i < 8; ++i) { h ^= uint8_t(v >> (8*i)); h *= 1099511628211ull; } };
        add(c_.epoch); add(c_.pool_rows); add(c_.max_rows); add(c_.max_sessions); add(c_.mtp); add(c_.guard());
        add(c_.prefill_rows); add(c_.two_lanes); add(have_id_); add(last_id_); add(boundary_); add(event_); add(want_decode_); add(busy_); add(failed_);
        add(pool_); g6rt::steering::hash_boot(h, c_.steering);
        for (const auto& s : s_) {
            add(s.key.slot); add(s.key.generation); add(uint64_t(s.phase));
            if (s.phase == Phase::Free) continue;
            add(s.req.id); add(s.req.prompt_tokens); add(s.req.max_tokens);
            g6rt::steering::hash_identity(h, s.req.steering);
            add(s.run.first_page); add(s.run.pages); add(s.run.table_version); add(s.run.list_hash); add(s.prefilled); add(s.emitted);
            add(s.inserted); add(s.deferred);
        }
        return h;
    }
    Changes apply(const Commit& e) {
        require(!failed_ && !busy_, "commit requires quiescent live engine");
        require(e.epoch == c_.epoch && e.event == event_ + 1 && e.boundary == boundary_ && e.before_hash == digest(), "event/state mismatch");
        auto after = s_; // validate whole transaction before mutating scheduler state
        Changes d;
        for (auto& s : after) if (s.phase == Phase::Retiring || (e.cancel && s.phase != Phase::Free && s.key == *e.cancel)) {
            d.freed.push_back(s.run); s.phase = Phase::Free;
        }
        if (e.cancel) require(e.cancel->epoch == c_.epoch, "cancel epoch mismatch");
        if (e.admit) {
            const auto& g = *e.admit;
            require(!have_id_ || g.req.id > last_id_, "native request ordinals must increase");
            require(g.key.slot < MAX_SESSIONS && g.key.epoch == c_.epoch, "bad grant key");
            auto& target = after[g.key.slot];
            require(target.phase == Phase::Free && target.key.generation != UINT32_MAX && g.key.generation == target.key.generation + 1, "grant slot generation mismatch");
            // the pool rejects a page two runs hold. A run past R holds its worst case (reserve_pages) from the start
            require((g.run.pages == grant_pages(c_, g.req) || g.run.pages == reserve_pages(c_, g.req)) &&
                    g.run.first_page < c_.pool_rows/PAGE && g.run.table_version, "bad reservation");
            unsigned live = 0;
            for (const auto& s : after) if (s.phase != Phase::Free) {
                ++live;
                require(s.req.id != g.req.id, "duplicate native request id");
            }
            require(live < c_.limit(), "active session/row limit");
            require(g.reuse < g.req.prompt_tokens, "reused rows past the prompt");
            target = Slot{g.key, g.req, g.run, Phase::Prefill, g.reuse, 0}; d.admitted = g;
        }
        if (e.admit) { last_id_ = e.admit->req.id; have_id_ = true; }
        s_ = after; event_ = e.event; ++revision_;
        return d; // allocator adapter now applies these canonical changes; no local allocation choice here
    }
    // a decoding session's run grew (Driver::step, every rank alike, between events)
    void grow(const Key& k, const Run& r) {
        auto& s = slot(k);
        require(!busy_ && s.phase == Phase::Decode && r.table_version == s.run.table_version && r.first_page == s.run.first_page &&
                r.pages > s.run.pages && r.pages <= reserve_pages(c_, s.req), "bad run growth");
        s.run = r;
    }
    // the decoding sessions in ascending request id, as next() orders them
    std::vector<Key> decoding() const {
        std::vector<const Slot*> v;
        for (const auto& s : s_) if (s.phase == Phase::Decode) v.push_back(&s);
        std::sort(v.begin(), v.end(), [](const Slot* a, const Slot* b) { return a->req.id < b->req.id; });
        std::vector<Key> out;
        for (const Slot* s : v) out.push_back(s->key);
        return out;
    }
    // The next quantum's shape from slot state `s` and the alternation flag `want_decode`, without touching the
    // scheduler: next() runs it on the live state to do the work, horizon() runs it forward on a copy to size a notice.
    // `stalled` (decoding sessions left out of this round; the driver's) is passed as none by the projection, which can
    // only make its decode rounds bigger, never its prefill quanta: the safe direction for a coverage estimate.
    static Quantum plan(const Config& c, const std::array<Slot, MAX_SESSIONS>& s, bool want_decode,
                        const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        Quantum q;
        std::array<const Slot*, MAX_SESSIONS> ordered{}; unsigned n = 0;
        for (unsigned i = 0; i < MAX_SESSIONS; ++i)
            if (s[i].phase == Phase::Prefill || (s[i].phase == Phase::Decode && !stalled[i])) ordered[n++] = &s[i];
        // At most eight entries; avoid STL's 16-entry insertion-sort threshold
        // (and GCC's array-bounds diagnostic on a partially filled std::array).
        for (unsigned i = 1; i < n; ++i) {
            const Slot* x = ordered[i]; unsigned j = i;
            while (j && ordered[j-1]->req.id > x->req.id) { ordered[j] = ordered[j-1]; --j; }
            ordered[j] = x;
        }
        // One prefill quantum serves one session: every decoding session joins the round, and of the prefilling ones
        // prefill_first picks the shortest prompt that is not still owed a turn. Before, this was the lowest request id,
        // which let one long prompt hold every prefill quantum against a short request admitted behind it (#61).
        const Slot* pf = nullptr;
        for (unsigned i = 0; i < n; ++i) {
            if (ordered[i]->phase == Phase::Decode) q.keys[q.count++] = ordered[i]->key;
            else if (!pf || prefill_first(*ordered[i], *pf)) pf = ordered[i];
        }
        if (q.count && (want_decode || !pf)) q.kind = Kind::Decode;
        else if (pf) {
            q.count = 1; q.keys[0] = pf->key; q.first = pf->prefilled;
            const uint32_t left = pf->req.prompt_tokens - 1 - pf->prefilled;
            q.kind = left ? Kind::Prefill : Kind::PromptTail;
            const uint64_t cap = uint64_t(c.prefill_rows) * (c.two_lanes && left > c.prefill_rows ? 2 : 1);
            q.rows = left ? uint32_t(std::min(uint64_t(left), cap)) : 1;
        }
        return q;
    }
    // stalled[slot]: a decoding session left out of this round (its run could not grow); it keeps its place
    Quantum next(const std::array<bool, MAX_SESSIONS>& stalled = {}) {
        require(!failed_ && !busy_, "scheduler failed or work in flight");
        Quantum q = plan(c_, s_, want_decode_, stalled);
        q.boundary = boundary_ + 1;
        if (q.kind != Kind::Idle) { work_ = q; busy_ = true; }
        return q;
    }
    // Advance a slot copy the way the completions do (prefilled, phase, the deferral counters): the projection in
    // horizon() drives the same state machine next()/completed() drive, on a copy.
    static void advance(std::array<Slot, MAX_SESSIONS>& sim, const Quantum& q) {
        if (q.kind == Kind::Prefill) sim[q.keys[0].slot].prefilled += q.rows;
        else if (q.kind == Kind::PromptTail) sim[q.keys[0].slot].phase = Phase::Decode;
        if (q.kind == Kind::Prefill || q.kind == Kind::PromptTail)
            for (auto& t : sim)
                if (t.phase == Phase::Prefill) t.deferred = t.key == q.keys[0] ? 0 : t.deferred + 1;
    }
    // The boundary a notice anchored now should target (service.h event): the first boundary whose work after the one
    // in flight covers NOTICE_ROWS prefill rows, within [boundary + NOTICE_MIN, boundary + NOTICE_MAX]. The boundary in
    // flight may be nearly done, so its remaining time is unknowable and the projection advances past it before
    // counting; a tail's session is assumed to keep decoding, which if wrong only loses a decoder and so adds
    // coverage, never removes it. A pure function of the slots and the alternation flag, so every rank names the same
    // boundary; the head's compute thread publishes it for the control thread, which may not read the scheduler.
    uint64_t horizon() const {
        auto sim = s_;
        bool want = want_decode_;
        const Quantum inflight = plan(c_, sim, want);
        if (inflight.kind == Kind::Idle) return boundary_ + NOTICE_MIN;   // the engine blocks: no boundary departs
        advance(sim, inflight);
        want = inflight.kind != Kind::Decode;
        uint32_t covered = 0;
        for (uint64_t k = NOTICE_MIN; k < NOTICE_MAX; ++k) {
            const Quantum q = plan(c_, sim, want);
            if (q.kind == Kind::Idle) return boundary_ + k;   // no work left: the engine blocks before this boundary
            advance(sim, q);
            want = q.kind != Kind::Decode;
            covered += q.kind == Kind::Prefill ? q.rows : q.kind == Kind::PromptTail ? 1 : 0;
            if (covered >= NOTICE_ROWS) return boundary_ + k;
        }
        return boundary_ + NOTICE_MAX;
    }
    // a just-admitted session that runs its whole prompt as decode steps (prompt logprobs): straight to PromptTail
    void skip_prefill(const Key& k) {
        auto& s = slot(k); require(!busy_ && s.phase == Phase::Prefill && !s.prefilled, "skip_prefill outside admission");
        s.prefilled = s.req.prompt_tokens - 1;
    }
    void complete_prefill() { // full quantum incl. both lanes and draft prefill; caller already synchronized
        require(busy_ && work_.kind == Kind::Prefill, "not a prefill quantum");
        slot(work_.keys[0]).prefilled += work_.rows; completed();
    }
    void complete_tail(const Completion& r) {
        require(busy_ && work_.kind == Kind::PromptTail && work_.keys[0] == r.key, "not this prompt tail");
        auto copy = slot(r.key); finish_slot(copy, r); slot(r.key) = copy; completed();
    }
    void complete_decode(const std::vector<Completion>& r) {
        require(busy_ && work_.kind == Kind::Decode && r.size() == work_.count, "not this decode round");
        for (unsigned i = 0; i < r.size(); ++i) {
            require(r[i].key == work_.keys[i], "decode completion order differs");
            const auto& s = slot(r[i].key); require(r[i].emitted <= s.req.max_tokens - s.emitted, "too many output tokens");
        }
        for (const auto& x : r) finish_slot(slot(x.key), x);
        completed();
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
