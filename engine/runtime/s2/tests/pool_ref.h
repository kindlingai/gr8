// The C++ RunPool that engine/kvman-rs replaced, kept unchanged (namespace g6s2ref) as the reference for
// pool_diff_test.cpp. Do not change its behaviour: the Rust pool must match it call for call.
#pragma once
#include "../scheduler.h"
#include "../../prefix_reuse.h"
#include <cstdio>
#include <map>
#include <vector>

namespace g6s2ref {
using g6s2::PAGE; using g6s2::require; using g6s2::Run;
// match()'s result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for take()
struct PoolHit { int skip = 0, lcp = 0; const char* why = "none"; uint64_t age = 0; };
class RunPool {
    struct Held { std::vector<uint32_t> list; bool exclusive = false; };
    struct Kept { std::vector<uint32_t> list; g6rt::PrefixReuse st; uint64_t age = 0; };
    uint32_t pages_, place_, below_rows_, version_ = 0, live_ = 0;
    std::vector<bool> used_;                   // pages a live run holds
    std::map<uint32_t, Held> held_;            // live runs by table version
    bool exclusive_ = false;
    bool keep_ = false, log_ = false;
    g6rt::PrefixReuse proto_;                  // the reuse rules (strict, noend, grid) every retained state follows
    std::vector<Kept> kept_;                   // oldest retired first
    uint64_t age_ = 0;
    static uint64_t mix(uint64_t h, uint64_t v) { for (unsigned i = 0; i < 8; ++i) { h ^= uint8_t(v >> (8 * i)); h *= 1099511628211ull; } return h; }
    static uint64_t hash(const std::vector<uint32_t>& list, bool exclusive) {
        uint64_t h = mix(14695981039346656037ull, exclusive);
        for (uint32_t p : list) h = mix(h, p);
        return h;
    }
    static uint32_t pages_for(uint64_t rows) { return uint32_t((rows + PAGE - 1) / PAGE); }
    uint32_t room_below() const { return place_ - live_; }
    void mark(const std::vector<uint32_t>& list, bool on) {
        for (uint32_t p : list) {
            require(p < pages_ && used_[p] != on, on ? "pool page already held by a live run" : "pool release of a free page");
            used_[p] = on;
        }
        live_ = on ? live_ + uint32_t(list.size()) : live_ - uint32_t(list.size());
    }
    // `list` (adopted pages, if any) grows to n pages below place: the lowest pages no run holds or retains, after
    // evicting the least recently retired runs until there are enough of them
    void fill(std::vector<uint32_t>& list, uint32_t n, std::vector<Kept>& gone) {
        std::vector<bool> taken(used_.begin(), used_.begin() + place_);
        for (uint32_t p : list) taken[p] = true;
        auto spare = [&] {
            std::vector<bool> b = taken;
            for (const auto& k : kept_) for (uint32_t p : k.list) if (p < place_) b[p] = true;
            uint32_t f = 0;
            for (uint32_t p = 0; p < place_; ++p) f += !b[p];
            return std::make_pair(b, f);
        };
        auto s = spare();
        while (s.second < n - list.size()) {
            require(!kept_.empty(), "pool reserve without room");
            gone.push_back(std::move(kept_.front())); kept_.erase(kept_.begin());
            s = spare();
        }
        for (uint32_t p = 0; p < place_ && list.size() < n; ++p) if (!s.first[p]) list.push_back(p);
    }
    Run grant(std::vector<uint32_t> list, bool exclusive, uint64_t rows, const char* how, const std::vector<Kept>& gone) {
        require(version_ != UINT32_MAX, "pool table version exhausted");
        mark(list, true);
        const Run r{list[0], uint32_t(list.size()), ++version_, hash(list, exclusive)};
        if (log_) {
            std::fprintf(stderr, "# s2 pool claim %s first %u pages %u rows %llu live %u kept %u evicted %zu", how, r.first_page, r.pages,
                         (unsigned long long)rows, live_, (unsigned)kept_.size(), gone.size());
            for (const auto& k : gone) std::fprintf(stderr, " [%u+%zu age %llu valid %d]", k.list[0], k.list.size(), (unsigned long long)k.age, k.st.valid);
            std::fprintf(stderr, "\n");
        }
        held_[r.table_version] = Held{std::move(list), exclusive};
        exclusive_ = exclusive_ || exclusive;
        return r;
    }
    const Held& held(const Run& r) const {
        const auto it = held_.find(r.table_version);
        require(it != held_.end() && r == Run{it->second.list[0], uint32_t(it->second.list.size()), r.table_version,
                                               hash(it->second.list, it->second.exclusive)}, "pool run unknown or changed");
        return it->second;
    }
public:
    RunPool(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows)
        : pages_(pool_pages), place_(place_pages), below_rows_(below_rows), used_(pool_pages, false) {
        require(place_pages && place_pages <= pool_pages && below_rows && below_rows % PAGE == 0 && below_rows / PAGE <= place_pages,
                "bad S2 pool / placement / below-R size");
    }
    bool exclusive(uint64_t rows) const { return rows > below_rows_; }
    uint32_t free_pages() const { return exclusive_ ? 0 : pages_ - live_; }
    // the most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages
    uint32_t room() const { return exclusive_ ? 0 : live_ ? place_ - live_ : pages_; }
    // head, before reserve: a past-R request waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t rows) const { return !exclusive_ && (!exclusive(rows) || !live_); }
    Run reserve(uint64_t rows) {
        const uint32_t n = pages_for(rows);
        require(rows && n <= room() && admissible(rows), "pool reserve without room");
        std::vector<uint32_t> list;
        std::vector<Kept> gone;
        if (exclusive(rows)) {
            gone = std::move(kept_); kept_.clear();
            for (uint32_t p = 0; p < n; ++p) list.push_back(p);
            return grant(std::move(list), true, rows, "exclusive", gone);
        }
        fill(list, n, gone);
        return grant(std::move(list), false, rows, gone.empty() ? "free" : "lru", gone);
    }
    void release(const Run& r) {
        const Held& h = held(r);
        mark(h.list, false);
        if (h.exclusive) exclusive_ = false;
        held_.erase(r.table_version);
    }
    // a live run's pages, in table order, and whether it is exclusive (its table is the CP map as built)
    std::vector<int> list(const Run& r) const { const Held& h = held(r); return std::vector<int>(h.list.begin(), h.list.end()); }
    bool exclusive_run(const Run& r) const { return held(r).exclusive; }
    // prefix reuse on, with these rules (PrefixReuse strict / noend / grid)
    void keep(const g6rt::PrefixReuse& rules) { keep_ = true; proto_ = rules; proto_.clear(); }
    bool keeping() const { return keep_; }
    // a retiring session's run: retained with its finished reuse state (st.valid rows), or released when nothing is valid
    void retain(const Run& r, const g6rt::PrefixReuse& st) {
        std::vector<uint32_t> list = held(r).list;
        release(r);
        if (!keep_ || st.valid <= 0) return;
        Kept k{std::move(list), proto_, ++age_};
        k.st.history = st.history; k.st.valid = st.valid; k.st.main_end = st.main_end; k.st.draft_end = st.draft_end;
        k.st.written_end = st.written_end; k.st.dirty_end = st.dirty_end; k.st.pf_end = st.pf_end;
        kept_.push_back(std::move(k));
    }
    // head, at admission (workers at their grant): the retained run with the longest reusable prefix for `prompt` (limit:
    // rows the prefill may skip) that the request's `rows` can grow from. Ties go to the most recent.
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, int limit) const {
        PoolHit best;
        if (!keep_ || exclusive_) return best;
        const uint32_t n = pages_for(rows);
        for (const auto& k : kept_) {
            g6rt::PrefixReuse t = k.st;
            const int skip = t.begin(prompt, limit, true);
            if (skip <= 0) { if (!best.skip && t.lcp >= best.lcp) { best.lcp = t.lcp; best.why = t.why; } continue; }
            const uint32_t a = pages_for(uint64_t(skip));
            bool room = a <= k.list.size() && a <= n;
            if (exclusive(rows)) {       // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                room = room && !live_ && n <= pages_;
                for (uint32_t i = 0; room && i < a; ++i) room = k.list[i] == i;
            } else {
                room = room && n <= room_below();
                for (uint32_t i = 0; room && i < a; ++i) room = k.list[i] < place_;
            }
            if (!room) { if (!best.skip) best.why = "room"; continue; }
            if (skip > best.skip || (skip == best.skip && k.age > best.age)) best = PoolHit{skip, t.lcp, "hit", k.age};
        }
        return best;
    }
    // claim a match() hit as a new grant (a fresh table version, as reserve)
    Run take(const PoolHit& h, uint64_t rows) {
        require(h.skip > 0 && admissible(rows), "pool take without a hit");
        auto it = kept_.begin();
        while (it != kept_.end() && it->age != h.age) ++it;
        require(it != kept_.end(), "pool take of a run no longer retained");
        const uint32_t a = pages_for(uint64_t(h.skip)), n = pages_for(rows);
        std::vector<uint32_t> list(it->list.begin(), it->list.begin() + a);
        kept_.erase(it);
        std::vector<Kept> gone;
        if (exclusive(rows)) {
            gone = std::move(kept_); kept_.clear();
            for (uint32_t p = a; p < n; ++p) list.push_back(p);
            return grant(std::move(list), true, rows, "hit", gone);
        }
        fill(list, n, gone);
        return grant(std::move(list), false, rows, "hit", gone);
    }
    // every rank's pool must agree: folded into Scheduler::digest at each event boundary
    uint64_t digest() const {
        uint64_t h = mix(mix(mix(14695981039346656037ull, version_), age_), exclusive_);
        for (uint32_t p = 0; p < pages_; ++p) if (used_[p]) h = mix(h, p);
        for (const auto& x : held_) h = mix(mix(h, x.first), hash(x.second.list, x.second.exclusive));
        for (const auto& k : kept_) h = mix(mix(mix(h, k.age), hash(k.list, false)), uint64_t(k.st.valid));
        return h;
    }
    void log(bool on) { log_ = on; }
    uint32_t kept_runs() const { return uint32_t(kept_.size()); }
};
} // namespace g6s2ref
