// The C++ implementation of scheduler.h's Scheduler, its free functions and Config::validate before they moved to
// engine/scheduler-rs, kept only for the differential test (tests/scheduler_rs_diff_test.cpp). It uses g6s2's types.
// Namespace g6s2ref. Delete with R2's last step.
#pragma once
#include "../../scheduler.h"
#include <cmath>

namespace g6s2ref {
using g6s2::Changes; using g6s2::Commit; using g6s2::Completion; using g6s2::Config; using g6s2::Grant; using g6s2::Key;
using g6s2::Kind; using g6s2::Phase; using g6s2::Quantum; using g6s2::Request; using g6s2::Run; using g6s2::Shrink;
using g6s2::Slot; using g6s2::require; using g6s2::MAX_SESSIONS; using g6s2::PAGE; using g6s2::PACK_MAX_MEMBERS;
using g6s2::MAX_DECODE_ROUNDS; using g6s2::PREFILL_DEFER_MAX; using g6s2::NOTICE_MIN; using g6s2::NOTICE_MAX;
using g6s2::NOTICE_ROWS;
inline void validate(const Config& c) {
    using namespace g6rt::steering;
    require(c.epoch && c.pool_rows >= PAGE && c.pool_rows % PAGE == 0, "bad epoch/pool");
    require((c.mtp > 0 || g6mod::has(G6_CAP_MTP0)) && c.mtp <= 7 && c.max_sessions && c.max_sessions <= MAX_SESSIONS && c.limit(),
            "bad session/row limit");
    require(!c.c8_pack || g6mod::has(G6_CAP_PACK), "c8 prefill pack is DS41 only");
    require(!c.park || g6mod::has(G6_CAP_PARK), "parked slots are DS41 only");
    require(c.guard() >= 2 * c.mtp && c.prefill_rows, "insufficient guard/prefill rows");
    require(!c.decode_prefill_rows || (c.decode_prefill_rows % PAGE == 0 && c.decode_prefill_rows <= c.prefill_rows),
            "decode_prefill_rows must be 0 or a multiple of 64 no larger than prefill_rows");
    require(c.decode_rounds <= MAX_DECODE_ROUNDS, "decode_rounds must be at most 64");
    require(c.notice_max >= NOTICE_MIN && c.notice_max <= NOTICE_MAX, "notice_max must be 2..8");
    require(!c.cp_step_rows || c.cp_step_rows >= c.mtp + 1, "cp_step_rows must be 0 or at least one session's verify (mtp + 1)");
    require(!c.cp_seg || c.cp_step_rows, "cp_seg needs cp_step_rows");
    require(c.r2_rows % PAGE == 0 && (!c.r2_rows || c.window_pages), "r2_rows must be 0, or a multiple of 64 with window_pages");
    const auto& st = c.steering;
    require(st.loaded == !st.profiles.empty() && st.profiles.size() <= PROFILES_MAX &&
            (st.scope == Scope::Shared || st.scope == Scope::Full), "invalid loaded steering receipt");
    for (const auto& p : st.profiles) require(valid_profile_name(p.name) && p.active_layers, "invalid loaded steering receipt");
    require(!st.forced || st.profiles.size() == 1, "forced steering needs exactly one loaded profile");
    if (st.forced) {
        const float scale = bits_float(st.forced_scale_bits);
        require(std::isfinite(scale) && scale > 0.f && scale <= SCALE_MAX, "forced steering needs a finite scale in (0,4]");
    } else require(!st.forced_scale_bits, "unforced steering has a forced scale");
}
inline uint64_t reserve_rows(const Config& c, const Request& r) {
    require(r.prompt_tokens && r.max_tokens, "empty request");
    return uint64_t(r.prompt_tokens) + r.max_tokens + c.guard();
}
inline uint32_t reserve_pages(const Config& c, const Request& r) {
    const auto rows = g6s2ref::reserve_rows(c, r);
    require(rows <= c.pool_rows, "request exceeds pool including guard");
    return uint32_t((rows + PAGE - 1) / PAGE);
}
// rows a session holds from its admission: its prompt, the token the tail samples and the guard, which covers the prompt
// tail, its chain steps and the first round. A round at `length` tokens (prompt + output) writes rows below
// length + 2 * mtp - 1 (verify, first-form draft, chain), DFlash2 below length + 7, so round_rows = length + guard covers
// it (Config::validate keeps guard >= 2 * mtp, and DFlash2 builds set 14). Later rows come by growth (Driver::step).
inline uint64_t round_rows(const Config& c, uint64_t length) { return length + c.guard(); }
inline uint64_t grant_rows(const Config& c, const Request& r) {
    return std::min(g6s2ref::reserve_rows(c, r), g6s2ref::round_rows(c, uint64_t(r.prompt_tokens) + 1));
}
inline uint32_t grant_pages(const Config& c, const Request& r) { return uint32_t((g6s2ref::grant_rows(c, r) + PAGE - 1) / PAGE); }
// the pages a grant may list: what it needs now, its worst case, or (window runs past R) its worst case and the window
inline bool grant_pages_ok(const Config& c, const Request& r, uint32_t pages) {
    return pages == g6s2ref::grant_pages(c, r) || pages == g6s2ref::reserve_pages(c, r) || (c.window_pages && pages == g6s2ref::reserve_pages(c, r) + c.window_pages);
}
// prompt rows a session still has to prefill; 0 means only its prompt tail is left
inline uint32_t prefill_left(const Slot& s) { return s.req.prompt_tokens - 1 - s.prefilled; }
// the order this quantum's prefill candidates are served in. Owed (passed over PREFILL_DEFER_MAX times) outranks not
// owed; among the owed the largest count wins, and among equals the shortest prompt. Replicated state only: every rank
// holds the same slots and counters, so every rank names the same session, which the event digest then confirms.
inline bool prefill_first(const Slot& a, const Slot& b) {
    const bool owed_a = a.deferred >= PREFILL_DEFER_MAX, owed_b = b.deferred >= PREFILL_DEFER_MAX;
    if (owed_a != owed_b) return owed_a;                 // a session that has waited its turn goes first
    if (owed_a && a.deferred != b.deferred) return a.deferred > b.deferred;  // both owed: the one that waited longest
    const uint32_t left_a = g6s2ref::prefill_left(a), left_b = g6s2ref::prefill_left(b);
    if (left_a != left_b) return left_a < left_b;        // fewest remaining prompt rows
    return a.req.id < b.req.id;                          // immutable native ordinal, so the choice is total
}
class Scheduler {
    Config c_;
    std::array<Slot, MAX_SESSIONS> s_{};
    uint64_t boundary_ = 0, event_ = 0, revision_ = 0, last_id_ = 0, pool_ = 0;
    bool have_id_ = false;
    bool want_decode_ = true, busy_ = false, failed_ = false;
    uint64_t park_seq_ = 0;     // Config::park: slots parked so far (Slot::parked)
    uint32_t decode_run_ = 0;   // decode rounds completed since the last prefill quantum (alternate; read only when decode_rounds > 1)
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
        boundary_ = work_.boundary; busy_ = false; alternate(c_, work_.kind, want_decode_, decode_run_);
        // The prefill choice below reads these counters, so they advance here on replicated state alone: the session this
        // quantum served has had its turn, every other prefilling session was passed over once. A PromptTail leaves
        // Phase::Prefill as it completes, so it is never reset here and its rivals are still counted as passed over.
        if (!c_.c8_pack && (work_.kind == Kind::Prefill || work_.kind == Kind::PromptTail)) {
            for (auto& s : s_) if (s.phase == Phase::Prefill) s.deferred = s.key == work_.keys[0] ? 0 : s.deferred + 1;
        } else if (work_.kind == Kind::Prefill || work_.kind == Kind::PromptTail || work_.kind == Kind::PrefillPack || work_.kind == Kind::PromptTailPack)
            for (auto& s : s_) if (s.phase == Phase::Prefill) {
                bool served = false;
                for (unsigned i = 0; i < work_.count; ++i) served |= s.key == work_.keys[i];
                s.deferred = served ? 0 : s.deferred + 1;
            }
    }
public:
    explicit Scheduler(Config c) : c_(c) {
        validate(c_);
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
        unsigned n = 0; for (const auto& s : s_) n += s.phase != Phase::Free && s.phase != Phase::Retiring && s.phase != Phase::Parked; return n;
    }
    unsigned retiring() const {
        unsigned n = 0; for (const auto& s : s_) n += s.phase == Phase::Retiring; return n;
    }
    unsigned parked() const {
        unsigned n = 0; for (const auto& s : s_) n += s.phase == Phase::Parked; return n;
    }
    std::optional<Key> find(uint64_t id) const {
        for (const auto& s : s_) if (s.phase != Phase::Free && s.phase != Phase::Parked && s.req.id == id) return s.key;
        return {};
    }
    // park: a parked slot (or a Retiring one, which the commit parks) the grant takes for a follow-up that extends it
    Key next_key(std::optional<Key> cancel = {}, std::optional<unsigned> park = {}) const {
        auto key = [&](const Slot& s) {
            require(s.key.generation != UINT32_MAX, "session generation exhausted");
            return Key{c_.epoch, s.key.slot, s.key.generation + 1};
        };
        if (park) {
            require(c_.park && *park < MAX_SESSIONS && (s_[*park].phase == Phase::Parked || s_[*park].phase == Phase::Retiring),
                    "no parked session on that slot");
            return key(s_[*park]);
        }
        if (!c_.park) {
            for (const auto& s : s_) if (s.phase == Phase::Free || s.phase == Phase::Retiring || (cancel && s.key == *cancel)) return key(s);
            throw std::runtime_error("no session slot");
        }
        // park: a free slot of the first limit(), else the oldest parked one (a Retiring slot parks at this commit, so
        // it is younger than every parked slot; among those the lowest slot). Replicated state only: every rank names it
        for (unsigned i = 0; i < c_.limit(); ++i)
            if (s_[i].phase == Phase::Free || (cancel && s_[i].key == *cancel && s_[i].phase != Phase::Parked)) return key(s_[i]);
        const Slot* old = nullptr;
        auto order = [](const Slot& s) { return s.phase == Phase::Parked ? s.parked : UINT64_MAX; };
        for (unsigned i = 0; i < c_.limit(); ++i)
            if ((s_[i].phase == Phase::Parked || s_[i].phase == Phase::Retiring) && (!old || order(s_[i]) < order(*old))) old = &s_[i];
        if (old) return key(*old);
        throw std::runtime_error("no session slot");
    }
    std::vector<Run> reclaimable(std::optional<Key> cancel = {}) const {
        std::vector<Run> out;
        for (const auto& s : s_)   // park: a Retiring slot parks instead, and a parked one is never cancelled
            if (c_.park ? s.phase != Phase::Retiring && s.phase != Phase::Parked && cancel && s.phase != Phase::Free && s.key == *cancel
                        : s.phase == Phase::Retiring || (cancel && s.phase != Phase::Free && s.key == *cancel))
                out.push_back(s.run);
        return out;
    }
    uint64_t digest() const { // event diagnostics only; NEVER called by next()/complete()
        uint64_t h = 14695981039346656037ull;
        auto add = [&](uint64_t v) { for (unsigned i = 0; i < 8; ++i) { h ^= uint8_t(v >> (8*i)); h *= 1099511628211ull; } };
        add(c_.epoch); add(c_.pool_rows); add(c_.max_rows); add(c_.max_sessions); add(c_.mtp); add(c_.guard());
        add(c_.prefill_rows); add(c_.two_lanes); if (c_.decode_prefill_rows) add(c_.decode_prefill_rows); add(have_id_); add(last_id_); add(boundary_); add(event_); add(want_decode_); add(busy_); add(failed_);
        if (c_.decode_rounds > 1) { add(c_.decode_rounds); add(decode_run_); }
        if (c_.window_pages) add(c_.window_pages);
        if (c_.tail_first) add(0x7A11F1257ull);
        if (c_.c8_pack) add(0xC8A11F1257ull);
        if (c_.notice_max != NOTICE_MAX) add(c_.notice_max);
        if (c_.cp_step_rows) { add(c_.cp_step_rows); add(c_.cp_after); add(c_.cp_seg); }
        if (c_.r2_rows) add(c_.r2_rows);
        if (c_.park) { add(0x9A12CEDull); add(park_seq_); }
        add(pool_); g6rt::steering::hash_boot(h, c_.steering);
        for (const auto& s : s_) {
            add(s.key.slot); add(s.key.generation); add(uint64_t(s.phase));
            if (s.phase == Phase::Free) continue;
            add(s.req.id); add(s.req.prompt_tokens); add(s.req.max_tokens);
            g6rt::steering::hash_identity(h, s.req.steering);
            if (s.req.r_rows) add(s.req.r_rows);
            add(s.run.first_page); add(s.run.pages); add(s.run.table_version); add(s.run.list_hash); add(s.prefilled); add(s.emitted);
            add(s.inserted); add(s.deferred);
            if (c_.c8_pack) { add(s.packable); add(s.pack_tail); }
            if (c_.park) add(s.parked);
        }
        return h;
    }
    Changes apply(const Commit& e) {
        require(!failed_ && !busy_, "commit requires quiescent live engine");
        require(e.epoch == c_.epoch && e.event == event_ + 1 && e.boundary == boundary_ && e.before_hash == digest(), "event/state mismatch");
        auto after = s_; // validate whole transaction before mutating scheduler state
        Changes d;
        uint64_t park_seq = park_seq_;
        for (auto& s : after) {
            // park: a finished session parks (its pages went back to the pool when it finished, as before), and a cancel
            // never names a parked or finished one (find)
            if (c_.park && s.phase == Phase::Retiring) { s.phase = Phase::Parked; s.parked = ++park_seq; continue; }
            if (s.phase == Phase::Retiring || (e.cancel && s.phase != Phase::Free && s.phase != Phase::Parked && s.key == *e.cancel)) {
                d.freed.push_back(s.run); s.phase = Phase::Free;
            }
        }
        if (e.cancel) require(e.cancel->epoch == c_.epoch, "cancel epoch mismatch");
        if (e.shrink) {
            const Shrink& k = *e.shrink;
            require(k.key.epoch == c_.epoch && k.key.slot < MAX_SESSIONS, "bad shrink key");
            Slot& t = after[k.key.slot];
            require(t.phase == Phase::Decode && t.key == k.key && c_.r2_rows && k.r_rows == c_.r2_rows && !t.req.r_rows &&
                    k.run.table_version == t.run.table_version && k.run.pages == t.run.pages, "bad shrink");
            t.req.r_rows = k.r_rows; t.run = k.run;
        }
        if (e.admit) {
            const auto& g = *e.admit;
            require(!have_id_ || g.req.id > last_id_, "native request ordinals must increase");
            require(g.key.slot < MAX_SESSIONS && g.key.epoch == c_.epoch, "bad grant key");
            auto& target = after[g.key.slot];
            require((target.phase == Phase::Free || (c_.park && target.phase == Phase::Parked)) && target.key.generation != UINT32_MAX && g.key.generation == target.key.generation + 1, "grant slot generation mismatch");
            // the pool rejects a page two runs hold. A run past R holds its worst case (reserve_pages) from the start
            require(g6s2ref::grant_pages_ok(c_, g.req, g.run.pages) &&
                    g.run.first_page < c_.pool_rows/PAGE && g.run.table_version, "bad reservation");
            unsigned live = 0;
            for (const auto& s : after) if (s.phase != Phase::Free && s.phase != Phase::Parked) {
                ++live;
                require(s.req.id != g.req.id, "duplicate native request id");
            }
            require(live < c_.limit(), "active session/row limit");
            require(g.reuse < g.req.prompt_tokens, "reused rows past the prompt");
            require(!g.req.r_rows || g.req.r_rows == c_.r2_rows, "grant at an R the engine does not have");
            target = Slot{g.key, g.req, g.run, Phase::Prefill, g.reuse, 0};
            target.packable = c_.c8_pack && !g.reuse && !g.restored && !g.req.steering.enabled;
            d.admitted = g;
        }
        if (e.admit) { last_id_ = e.admit->req.id; have_id_ = true; }
        s_ = after; event_ = e.event; ++revision_; park_seq_ = park_seq;
        return d; // allocator adapter now applies these canonical changes; no local allocation choice here
    }
    // a decoding session's run grew (Driver::step, every rank alike, between events)
    void grow(const Key& k, const Run& r) {
        auto& s = slot(k);
        require(!busy_ && s.phase == Phase::Decode && r.table_version == s.run.table_version && r.first_page == s.run.first_page &&
                r.pages > s.run.pages && r.pages <= g6s2ref::reserve_pages(c_, s.req), "bad run growth");
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
    // The alternation flag after a completed quantum of kind k: a prefill (or tail) asks for a decode round next; a decode
    // round asks for the prefill unless fewer than decode_rounds rounds have run since the last prefill. With
    // decode_rounds <= 1 this is exactly `want = k != Decode`. Replicated state only (the kinds every rank completed), and
    // horizon() runs the same function forward, so the notice target agrees with the steps the ranks will take.
    static void alternate(const Config& c, Kind k, bool& want, uint32_t& run) {
        if (c.tail_first && k == Kind::PromptTail) return;   // a solo tail is one row: the decoders lose no round to it
        if (k != Kind::Decode) { want = true; run = 0; return; }
        if (run < MAX_DECODE_ROUNDS) ++run;
        want = c.decode_rounds > 1 && run < c.decode_rounds;
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
        // any decoding session, stalled or not: the projection (horizon) passes none stalled, so the cap must not
        // depend on `stalled`, or a projected quantum could be larger than the real one (the unsafe direction)
        bool decoding = false;
        for (unsigned i = 0; i < MAX_SESSIONS; ++i) decoding |= s[i].phase == Phase::Decode;
        for (unsigned i = 0; i < n; ++i) {
            if (ordered[i]->phase == Phase::Decode) q.keys[q.count++] = ordered[i]->key;
            else if (!pf) pf = ordered[i];
            else if (c.tail_first && (g6s2ref::prefill_left(*ordered[i]) == 0) != (g6s2ref::prefill_left(*pf) == 0)) {
                if (g6s2ref::prefill_left(*ordered[i]) == 0) pf = ordered[i];   // a ready tail outranks every prefill
            } else if (g6s2ref::prefill_first(*ordered[i], *pf)) pf = ordered[i];
        }
        // A held W pack owns its exact surviving membership until its tail. Cancellation may remove a key;
        // a later admission never joins it. Serve the tail immediately, before the next decoder round.
        if (c.c8_pack) {
            unsigned held = 0;
            for (unsigned i = 0; i < n; ++i) if (ordered[i]->phase == Phase::Prefill && ordered[i]->pack_tail) {
                q.keys[held] = ordered[i]->key;
                q.firsts[held] = ordered[i]->prefilled;
                q.row_counts[held] = 1;
                ++held;
            }
            if (held) {
                q.count = held; q.kind = held > 1 ? Kind::PromptTailPack : Kind::PromptTail;
                q.first = q.firsts[0]; q.rows = held;
                return q;
            }
        }
        if (q.count && (want_decode || !pf)) q.kind = Kind::Decode;
        else if (pf) {
            if (c.c8_pack && pf->packable && !pf->prefilled && g6s2ref::prefill_left(*pf) >= 1 && g6s2ref::prefill_left(*pf) <= 64) {
                const uint32_t cap = decoding && c.decode_prefill_rows ? std::min({512u, c.prefill_rows, c.decode_prefill_rows})
                    : std::min(512u, c.prefill_rows);
                std::array<const Slot*, MAX_SESSIONS> take{};
                unsigned m = 1; uint32_t total = g6s2ref::prefill_left(*pf); take[0] = pf;
                for (unsigned i = 0; i < n && m < PACK_MAX_MEMBERS; ++i) {
                    const Slot* x = ordered[i];
                    if (x == pf || x->phase != Phase::Prefill || !x->packable || x->prefilled) continue;
                    const uint32_t left = g6s2ref::prefill_left(*x);
                    if (!left || left > 64 || total + left > cap) continue;
                    take[m++] = x; total += left;
                }
                if (m > 1 && total <= cap) {
                    // The first choice honors prefill_first/owed priority; execution order is native request id.
                    for (unsigned i = 1; i < m; ++i) {
                        const Slot* x = take[i]; unsigned j = i;
                        while (j && take[j-1]->req.id > x->req.id) { take[j] = take[j-1]; --j; }
                        take[j] = x;
                    }
                    q.kind = Kind::PrefillPack; q.count = m; q.rows = total;
                    for (unsigned i = 0; i < m; ++i) {
                        q.keys[i] = take[i]->key; q.firsts[i] = take[i]->prefilled; q.row_counts[i] = g6s2ref::prefill_left(*take[i]);
                    }
                    return q;
                }
            }
            q.count = 1; q.keys[0] = pf->key; q.first = pf->prefilled;
            const uint32_t left = pf->req.prompt_tokens - 1 - pf->prefilled;
            q.kind = left ? Kind::Prefill : Kind::PromptTail;
            const uint64_t cap = decoding && c.decode_prefill_rows ? uint64_t(c.decode_prefill_rows)
                : uint64_t(c.prefill_rows) * (c.two_lanes && left > c.prefill_rows ? 2 : 1);
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
        else if (q.kind == Kind::PrefillPack) for (unsigned i = 0; i < q.count; ++i) {
            auto& t = sim[q.keys[i].slot]; t.prefilled += q.row_counts[i]; t.pack_tail = true;
        }
        else if (q.kind == Kind::PromptTailPack) for (unsigned i = 0; i < q.count; ++i) {
            auto& t = sim[q.keys[i].slot]; t.phase = Phase::Decode; t.pack_tail = false;
        }
        if (q.kind == Kind::Prefill || q.kind == Kind::PromptTail || q.kind == Kind::PrefillPack || q.kind == Kind::PromptTailPack)
            for (auto& t : sim) if (t.phase == Phase::Prefill) {
                bool served = false;
                for (unsigned i = 0; i < q.count; ++i) served |= t.key == q.keys[i];
                t.deferred = served ? 0 : t.deferred + 1;
            }
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
        uint32_t run = decode_run_;
        const Quantum inflight = plan(c_, sim, want);
        if (inflight.kind == Kind::Idle) return boundary_ + NOTICE_MIN;   // the engine blocks: no boundary departs
        advance(sim, inflight);
        alternate(c_, inflight.kind, want, run);
        uint32_t covered = 0;
        for (uint64_t k = NOTICE_MIN; k < c_.notice_max; ++k) {
            const Quantum q = plan(c_, sim, want);
            if (q.kind == Kind::Idle) return boundary_ + k;   // no work left: the engine blocks before this boundary
            advance(sim, q);
            alternate(c_, q.kind, want, run);
            covered += q.kind == Kind::Prefill || q.kind == Kind::PrefillPack ? q.rows
                : q.kind == Kind::PromptTail || q.kind == Kind::PromptTailPack ? q.rows : 0;
            if (covered >= NOTICE_ROWS) return boundary_ + k;
        }
        return boundary_ + c_.notice_max;
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
    void complete_prefill_pack() { // one synchronized W pass, all selected sessions advance together
        require(busy_ && work_.kind == Kind::PrefillPack && work_.count >= 2, "not a prefill pack quantum");
        for (unsigned i = 0; i < work_.count; ++i) {
            auto& s = slot(work_.keys[i]);
            require(s.phase == Phase::Prefill && !s.pack_tail && s.prefilled == work_.firsts[i] &&
                    work_.row_counts[i] && work_.row_counts[i] == g6s2ref::prefill_left(s), "prefill pack membership changed");
        }
        for (unsigned i = 0; i < work_.count; ++i) {
            auto& s = slot(work_.keys[i]); s.prefilled += work_.row_counts[i]; s.pack_tail = true;
        }
        completed();
    }
    void complete_tail(const Completion& r) {
        require(busy_ && work_.kind == Kind::PromptTail && work_.keys[0] == r.key, "not this prompt tail");
        auto copy = slot(r.key); finish_slot(copy, r); copy.pack_tail = false; slot(r.key) = copy; completed();
    }
    void complete_tail_pack(const std::vector<Completion>& r) {
        require(busy_ && work_.kind == Kind::PromptTailPack && r.size() == work_.count, "not this prompt tail pack");
        auto copy = s_; // reject the entire pack before publishing any session completion
        for (unsigned i = 0; i < r.size(); ++i) {
            require(work_.keys[i] == r[i].key && copy[r[i].key.slot].pack_tail &&
                    r[i].emitted <= copy[r[i].key.slot].req.max_tokens - copy[r[i].key.slot].emitted &&
                    r[i].inserted >= copy[r[i].key.slot].inserted, "tail pack completion invalid");
        }
        for (unsigned i = 0; i < r.size(); ++i) {
            finish_slot(copy[r[i].key.slot], r[i]); copy[r[i].key.slot].pack_tail = false;
        }
        s_ = copy; completed();
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
} // namespace g6s2ref
