// S2 component 5, host-only state. No CUDA, allocator implementation, sampler, or transport.
// A caller may apply Commit ONLY after event-boundary all-rank quiescence (event_gate.h).
// TP4 owns run allocation/claim/release; component 6 owns tokens, MTP and hidden state.
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

namespace g6s2 {
inline void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
constexpr unsigned MAX_SESSIONS = 8, PAGE = 64;
struct Config {
    uint64_t epoch = 1;
    uint32_t pool_rows = 614400, max_rows = 24, max_sessions = 8, mtp = 2;
    uint32_t guard_tokens = UINT32_MAX, prefill_rows = 4096;
    bool two_lanes = true;
    // max_rows bounds ONE step: a round is one main step of every session's mtp + 1 rows (s2_batch.h Batch::round)
    uint32_t guard() const { return guard_tokens == UINT32_MAX ? 2 * mtp : guard_tokens; }
    unsigned limit() const { return std::min(max_sessions, max_rows / (mtp + 1)); }
    void validate() const {
        require(epoch && pool_rows >= PAGE && pool_rows % PAGE == 0, "bad epoch/pool");
        require(mtp > 0 && mtp <= 7 && max_sessions && max_sessions <= MAX_SESSIONS && limit(), "bad session/row limit");
        require(guard() >= 2 * mtp && prefill_rows, "insufficient guard/prefill rows");
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
struct Run {
    uint32_t base_page = 0, pages = 0, table_version = 0;
    bool identity() const { return base_page == 0; }
};
enum class Phase { Free, Prefill, Decode, Retiring };
struct Slot {
    Key key; Request req; Run run;
    Phase phase = Phase::Free;
    uint32_t prefilled = 0, emitted = 0;
};
struct Grant { Key key; Request req; Run run; };
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
struct Completion { Key key; uint32_t emitted = 0; bool finished = false; };

class Scheduler {
    Config c_;
    std::array<Slot, MAX_SESSIONS> s_{};
    uint64_t boundary_ = 0, event_ = 0, revision_ = 0, last_id_ = 0;
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
        if (r.finished || s.emitted == s.req.max_tokens) { s.phase = Phase::Retiring; ++revision_; }
        else s.phase = Phase::Decode;
    }
    void completed() { boundary_ = work_.boundary; busy_ = false; want_decode_ = work_.kind != Kind::Decode; }
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
        for (const auto& s : s_) {
            add(s.key.slot); add(s.key.generation); add(uint64_t(s.phase));
            if (s.phase == Phase::Free) continue;
            add(s.req.id); add(s.req.prompt_tokens); add(s.req.max_tokens);
            add(s.run.base_page); add(s.run.pages); add(s.run.table_version); add(s.prefilled); add(s.emitted);
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
            require(g.run.pages == reserve_pages(c_, g.req) && uint64_t(g.run.base_page) + g.run.pages <= c_.pool_rows/PAGE && g.run.table_version, "bad contiguous reservation");
            unsigned live = 0;
            for (const auto& s : after) if (s.phase != Phase::Free) {
                ++live;
                require(s.req.id != g.req.id, "duplicate native request id");
                require(uint64_t(g.run.base_page) + g.run.pages <= s.run.base_page || uint64_t(s.run.base_page) + s.run.pages <= g.run.base_page, "overlapping runs");
            }
            require(live < c_.limit(), "active session/row limit");
            require(live || g.run.identity(), "empty pool admission must start at zero");
            target = Slot{g.key, g.req, g.run, Phase::Prefill, 0, 0}; d.admitted = g;
        }
        if (e.admit) { last_id_ = e.admit->req.id; have_id_ = true; }
        s_ = after; event_ = e.event; ++revision_;
        return d; // allocator adapter now applies these canonical changes; no local allocation choice here
    }
    Quantum next() {
        require(!failed_ && !busy_, "scheduler failed or work in flight");
        Quantum q; q.boundary = boundary_ + 1;
        std::array<const Slot*, MAX_SESSIONS> ordered{}; unsigned n = 0;
        for (const auto& s : s_) if (s.phase == Phase::Prefill || s.phase == Phase::Decode) ordered[n++] = &s;
        // At most eight entries; avoid STL's 16-entry insertion-sort threshold
        // (and GCC's array-bounds diagnostic on a partially filled std::array).
        for (unsigned i = 1; i < n; ++i) {
            const Slot* x = ordered[i]; unsigned j = i;
            while (j && ordered[j-1]->req.id > x->req.id) { ordered[j] = ordered[j-1]; --j; }
            ordered[j] = x;
        }
        const Slot* pf = nullptr;
        for (unsigned i = 0; i < n; ++i) {
            if (ordered[i]->phase == Phase::Decode) q.keys[q.count++] = ordered[i]->key;
            else if (!pf) pf = ordered[i];
        }
        if (q.count && (want_decode_ || !pf)) q.kind = Kind::Decode;
        else if (pf) {
            q.count = 1; q.keys[0] = pf->key; q.first = pf->prefilled;
            const uint32_t left = pf->req.prompt_tokens - 1 - pf->prefilled;
            q.kind = left ? Kind::Prefill : Kind::PromptTail;
            const uint64_t cap = uint64_t(c_.prefill_rows) * (c_.two_lanes && left > c_.prefill_rows ? 2 : 1);
            q.rows = left ? uint32_t(std::min(uint64_t(left), cap)) : 1;
        }
        if (q.kind != Kind::Idle) { work_ = q; busy_ = true; }
        return q;
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

enum class Stall { None, Slots, Capacity, Fragmentation };
struct AdmissionStatus { Stall why = Stall::None; uint32_t requested = 0, total_free = 0, largest_run = 0; };
// Rank0 ingress FIFO. reserve/first-fit remains exclusively in the TP4 allocator.
class AdmissionQueue {
    Config c_; std::deque<Request> q_;
    uint64_t last_revision_ = UINT64_MAX, last_head_ = UINT64_MAX, last_id_ = 0;
    bool have_id_ = false;
public:
    uint64_t stalls = 0, fragmentation_stalls = 0;
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
    std::optional<AdmissionStatus> check(uint64_t revision, bool slot, uint32_t total, uint32_t largest) {
        if (q_.empty() || (last_head_ == q_.front().id && last_revision_ == revision)) return {};
        require(largest <= total && total <= c_.pool_rows/PAGE, "invalid allocator view");
        last_head_ = q_.front().id; last_revision_ = revision;
        AdmissionStatus s; s.requested = reserve_pages(c_, q_.front()); s.total_free = total; s.largest_run = largest;
        s.why = !slot ? Stall::Slots : total < s.requested ? Stall::Capacity : largest < s.requested ? Stall::Fragmentation : Stall::None;
        if (s.why != Stall::None) { ++stalls; fragmentation_stalls += s.why == Stall::Fragmentation; }
        return s;
    }
};
} // namespace g6s2
