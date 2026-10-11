// L5 control-side accounting/replay, NOT wired into Service/native readiness.
// Quotes are frozen inputs, not measurements. KV allocation remains TP4-owned.
// Admission/release callers must already hold the existing all-rank event fence.
#pragma once
#include "scheduler.h"
#include <map>
#include <utility>

namespace g6s2::resources {
inline bool same_request(const Request& a, const Request& b) {
    return a.id == b.id && a.prompt_tokens == b.prompt_tokens && a.max_tokens == b.max_tokens;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    require(b <= UINT64_MAX - a, "resource accounting overflow"); return a + b;
}
struct Quote {
    Request request;
    uint64_t plan = 0;
    // One entry per rank. Never use estimated_ns as a hard memory reservation.
    std::vector<uint64_t> estimated_ns, auxiliary_bytes;
};
// Replay manifest: exact request/plan quotes, immutable after construction.
// table_id identifies the caller-verified frozen source/fixture, not its hash
// verification. Real layout/cost quote generation is not implemented here.
class FrozenQuotes {
    uint64_t id_; unsigned ranks_;
    std::map<uint64_t, Quote> quotes_;
public:
    FrozenQuotes(uint64_t id, unsigned ranks, std::vector<Quote> quotes) : id_(id), ranks_(ranks) {
        require(id && ranks, "missing frozen quote identity/ranks");
        for (auto& q : quotes) {
            require(q.request.prompt_tokens && q.request.max_tokens && q.plan &&
                    q.estimated_ns.size() == ranks && q.auxiliary_bytes.size() == ranks, "incomplete quote");
            const auto request_id = q.request.id;
            require(quotes_.emplace(request_id, std::move(q)).second, "duplicate frozen quote");
        }
    }
    uint64_t id() const { return id_; }
    unsigned ranks() const { return ranks_; }
    const Quote* lookup(const Request& request) const {
        auto it = quotes_.find(request.id);
        return it == quotes_.end() || !same_request(it->second.request, request) ? nullptr : &it->second;
    }
};
// Provisioned auxiliary arena ONLY: excludes weights/KV pool/other process RAM.
// shared_bytes is a fixed charge, even when shared scratch is momentarily idle.
struct Capacity { uint64_t auxiliary_bytes = 0, shared_bytes = 0; };
enum class Fit { Ready, MissingQuote, Slots, AuxiliaryCapacity };
enum class Storage { Main = 0, Draft = 1, Index = 2, Shared = 3 };
struct Usage {
    std::array<uint64_t, 3> extent_pages{}; // highest request-relative end page
    uint64_t auxiliary_peak = 0, actual_ns = 0, timed_slices = 0;
    std::array<bool, 3> observed{}; // absence is UNRUN, never a measured zero
};
struct Entry {
    Grant grant;
    Quote quote;
    uint64_t table_id = 0;
    std::vector<Usage> usage;
    bool retiring = false;
};
struct Violation {
    Key key; unsigned rank = 0; Storage storage = Storage::Main;
    uint64_t first_page = 0, pages = 0, auxiliary_bytes = 0;
    Run reserved_run; uint64_t reserved_auxiliary_bytes = 0;
};
class Ledger {
    Config config_; FrozenQuotes quotes_;
    std::vector<Capacity> capacity_;
    std::vector<uint64_t> reserved_, estimated_, shared_peak_;
    std::vector<bool> shared_observed_;
    std::array<std::optional<Entry>, MAX_SESSIONS> entries_;
    std::array<uint32_t, MAX_SESSIONS> generation_{};
    uint64_t event_ = 0, last_request_ = 0; bool have_request_ = false;
    std::optional<Violation> violation_;
    void healthy() const { require(!violation_, "L5 KILL: reservation previously exceeded"); }
    void event(uint64_t e) const { healthy(); require(e && e >= event_, "invalid resource event fence"); }
    const Entry& entry(Key k) const {
        require(k.epoch == config_.epoch && k.slot < MAX_SESSIONS && entries_[k.slot] &&
                entries_[k.slot]->grant.key == k, "stale resource key");
        return *entries_[k.slot];
    }
    Entry& entry(Key k) { return const_cast<Entry&>(static_cast<const Ledger&>(*this).entry(k)); }
    [[noreturn]] void kill(Violation v) {
        violation_ = v; throw std::runtime_error("L5 KILL: observed usage exceeds committed reservation");
    }
public:
    Ledger(Config c, FrozenQuotes q, std::vector<Capacity> capacity)
        : config_(c), quotes_(std::move(q)), capacity_(std::move(capacity)),
          reserved_(capacity_.size()), estimated_(capacity_.size()), shared_peak_(capacity_.size()),
          shared_observed_(capacity_.size(), false) {
        config_.validate(); require(capacity_.size() == quotes_.ranks(), "capacity rank count differs");
        for (unsigned r = 0; r < capacity_.size(); ++r) {
            require(capacity_[r].shared_bytes <= capacity_[r].auxiliary_bytes, "shared scratch exceeds arena");
            reserved_[r] = capacity_[r].shared_bytes;
        }
    }
    bool killed() const { return violation_.has_value(); }
    const auto& violation() const { return violation_; }
    const auto& reserved_auxiliary_bytes() const { return reserved_; }
    const auto& estimated_live_ns() const { return estimated_; }
    const auto& shared_peak_bytes() const { return shared_peak_; }
    const auto& shared_observed() const { return shared_observed_; }
    const Entry& snapshot(Key k) const { return entry(k); } // inspect even after KILL
    unsigned resident() const { unsigned n = 0; for (const auto& e : entries_) n += e.has_value(); return n; }
    Fit probe(const Request& request) const {
        healthy(); (void)reserve_pages(config_, request);
        const auto* q = quotes_.lookup(request); if (!q) return Fit::MissingQuote;
        if (resident() >= config_.limit()) return Fit::Slots;
        for (unsigned r = 0; r < capacity_.size(); ++r)
            if (q->auxiliary_bytes[r] > capacity_[r].auxiliary_bytes - reserved_[r]) return Fit::AuxiliaryCapacity;
        return Fit::Ready;
    }
    void admit_at_fence(const Grant& g, uint64_t fence_event) {
        event(fence_event);
        require(g.key.epoch == config_.epoch && g.key.slot < MAX_SESSIONS && !entries_[g.key.slot], "resource slot occupied/invalid");
        require(generation_[g.key.slot] != UINT32_MAX && g.key.generation == generation_[g.key.slot] + 1, "resource generation mismatch");
        require(!have_request_ || g.req.id > last_request_, "resource request ordinals must increase");
        require(g.run.table_version && g.run.pages == reserve_pages(config_, g.req) &&
                uint64_t(g.run.first_page) + g.run.pages <= config_.pool_rows / PAGE, "invalid canonical resource run");
        for (const auto& e : entries_) if (e) {
            const auto& run = e->grant.run;
            require(uint64_t(g.run.first_page) + g.run.pages <= run.first_page ||
                    uint64_t(run.first_page) + run.pages <= g.run.first_page, "resource runs overlap");
        }
        require(resident() || g.run.first_page == 0, "empty resource pool must start at zero");
        require(probe(g.req) == Fit::Ready, "resource admission without a fitting quote");
        const auto& q = *quotes_.lookup(g.req);
        // Check/allocate before mutation. Predicted time overflow must not leave a
        // partially admitted request. No device-ms admission threshold is assumed.
        auto next_estimated = estimated_;
        for (unsigned r = 0; r < capacity_.size(); ++r) next_estimated[r] = add(next_estimated[r], q.estimated_ns[r]);
        Entry next{g, q, quotes_.id(), std::vector<Usage>(capacity_.size()), false};
        entries_[g.key.slot] = std::move(next); estimated_.swap(next_estimated);
        for (unsigned r = 0; r < capacity_.size(); ++r) reserved_[r] += q.auxiliary_bytes[r];
        generation_[g.key.slot] = g.key.generation; last_request_ = g.req.id; have_request_ = true; event_ = fence_event;
    }
    // Absolute canonical page range, checked before any subtraction/addition.
    // Call for every native main/draft/index access/allocation observation, not
    // just from requested token counts. CPU fixtures cannot certify those calls.
    void observe(Key k, unsigned rank, Storage storage, uint64_t first_page, uint64_t pages, uint64_t auxiliary_bytes) {
        healthy(); auto& e = entry(k);
        require(rank < capacity_.size() && unsigned(storage) < 3, "invalid resource observation");
        const auto& run = e.grant.run;
        const bool range = first_page >= run.first_page && first_page - run.first_page <= run.pages &&
                           pages <= run.pages - (first_page - run.first_page);
        if (!range || auxiliary_bytes > e.quote.auxiliary_bytes[rank])
            kill({k, rank, storage, first_page, pages, auxiliary_bytes, run, e.quote.auxiliary_bytes[rank]});
        auto& u = e.usage[rank]; const unsigned s = unsigned(storage);
        if (pages) u.extent_pages[s] = std::max(u.extent_pages[s], first_page - run.first_page + pages);
        u.auxiliary_peak = std::max(u.auxiliary_peak, auxiliary_bytes); u.observed[s] = true;
    }
    void observe_shared(unsigned rank, uint64_t bytes) {
        healthy(); require(rank < capacity_.size(), "invalid shared rank");
        if (bytes > capacity_[rank].shared_bytes) kill({{}, rank, Storage::Shared, 0, 0, bytes, {}, capacity_[rank].shared_bytes});
        shared_peak_[rank] = std::max(shared_peak_[rank], bytes); shared_observed_[rank] = true;
    }
    void record_device_time(Key k, unsigned rank, uint64_t elapsed_ns) {
        healthy(); auto& e = entry(k); require(rank < capacity_.size(), "invalid timing rank");
        auto& u = e.usage[rank]; const auto total = add(u.actual_ns, elapsed_ns), count = add(u.timed_slices, 1);
        u.actual_ns = total; u.timed_slices = count; // exceeding forecast is NOT a memory KILL
    }
    void retire(Key k) { healthy(); entry(k).retiring = true; } // no credit returned
    Entry release_at_fence(Key k, uint64_t fence_event) {
        event(fence_event); const auto& e = entry(k); require(e.retiring, "resource release before retirement");
        Entry result = e; // retain evidence even when nothing was observed (UNRUN)
        for (unsigned r = 0; r < capacity_.size(); ++r) {
            reserved_[r] -= e.quote.auxiliary_bytes[r]; estimated_[r] -= e.quote.estimated_ns[r];
        }
        entries_[k.slot].reset(); event_ = fence_event; return result;
    }
};
} // namespace g6s2::resources
