// S2 below R v1 (S2-STATE.md) run allocator over the native pool pages. Host only, no CUDA. One instance per rank;
// every call is at an all-rank event fence (service.h). Head: reserve (first-fit, lowest base). Workers: claim the
// head's exact grant. A reservation of more than `below` rows (a session past R, CP decode) runs solo at base 0: it
// is admitted only into an EMPTY pool and, while resident, holds the whole pool (free 0, nothing else admits).
// Two sizes on the CP binding: `pages` = the logical capacity (Config::pool_rows / 64; a past-R session at base 0 reads
// the table as built, CP map included), `place` = the physical pages every rank has for runs at base B (S2Model::
// pool_pages, the fewest of main and draft). Runs at or below R are placed inside [0, place) only.
// Prefix reuse (keep()): a retired run's pages stay as they are, with the session's PrefixReuse state, until a claim
// overlaps them. match() finds the retained run whose cached prefix a new prompt can continue; the grant then takes that
// run's base (pages extended as needed) and starts its prefill at the retained row. Retained pages count as free.
// Eviction is placement only (no row of any session changes): a reservation lands in a gap free of retained runs when one
// is large enough; otherwise on the window of live-free pages whose overlapped retained runs are the least recently
// retired (lexicographic: newest age among them, then pages overlapped, then lowest base), so a miss destroys the oldest
// retained run and never a fresh one. Retained runs are never capped or aged out: they stay until their pages are needed.
// log(true) (g6m4: G6_POOL_LOG=1): one stderr line per claim, with the base, pages, and every retained run it evicted.
#pragma once
#include "scheduler.h"
#include "../prefix_reuse.h"
#include <cstdio>
#include <vector>

namespace g6s2 {
// match(): the retained run a prompt continues (skip = its retained rows; 0 = none)
struct PoolHit { Run run; int skip = 0, lcp = 0; const char* why = "none"; };
class RunPool {
    uint32_t pages_, place_, below_rows_, version_ = 0;
    std::vector<bool> used_;
    bool exclusive_ = false;
    Run held_{};                               // the exclusive run, while resident
    struct Kept { Run run; g6rt::PrefixReuse st; uint64_t age = 0; };
    bool keep_ = false, log_ = false;
    const char* how_ = "claim";                // how the next claim was placed (log only): gap / lru / fit / hit / exclusive
    g6rt::PrefixReuse proto_;                  // the reuse rules (strict, noend, grid) every retained state follows
    std::vector<Kept> kept_;
    uint64_t age_ = 0;
    uint32_t used_pages() const { uint32_t n = 0; for (bool u : used_) n += u; return n; }
    bool kept_at(uint32_t i) const {
        for (const auto& k : kept_) if (i >= k.run.base_page && i < k.run.base_page + k.run.pages) return true;
        return false;
    }
    // drop the retained runs that overlap [base, base + pages); all = every retained run; returns the dropped ones
    std::vector<Kept> evict(uint32_t base, uint32_t pages, bool all) {
        std::vector<Kept> left, gone;
        for (auto& k : kept_)
            if (!all && (k.run.base_page + k.run.pages <= base || base + pages <= k.run.base_page)) left.push_back(std::move(k));
            else gone.push_back(std::move(k));
        kept_ = std::move(left);
        return gone;
    }
    void mark(const Run& r, bool on) {
        require(r.pages && uint64_t(r.base_page) + r.pages <= pages_, "pool run outside the pool");
        for (uint32_t i = r.base_page; i < r.base_page + r.pages; ++i) {
            require(used_[i] != on, on ? "pool run overlaps a live run" : "pool release of a free page");
            used_[i] = on;
        }
    }
public:
    RunPool(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows)
        : pages_(pool_pages), place_(place_pages), below_rows_(below_rows), used_(pool_pages, false) {
        require(place_pages && place_pages <= pool_pages && below_rows && below_rows % PAGE == 0 && below_rows / PAGE <= place_pages,
                "bad S2 pool / placement / below-R size");
    }
    bool exclusive(uint64_t rows) const { return rows > below_rows_; }
    uint32_t free_pages() const { return exclusive_ ? 0 : pages_ - used_pages(); }
    uint32_t largest_run() const {
        if (exclusive_) return 0;
        if (!used_pages()) return pages_;      // empty: a past-R run takes [0, n) of the logical pool
        uint32_t best = 0, run = 0;
        for (uint32_t i = 0; i < place_; ++i) { run = used_[i] ? 0 : run + 1; best = std::max(best, run); }
        return best;
    }
    // head, before reserve: a past-R request waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t rows) const { return !exclusive_ && (!exclusive(rows) || !used_pages()); }
    // First fit, lowest base. With prefix reuse on: the largest gap free of live and retained runs, at its start when it
    // starts at page 0, else centred, so the run before it and this one both have room to grow into for a later turn
    // (a hit extends its run in place); without such a gap, first fit over retained pages (claim evicts them).
    // An exclusive run always takes base 0 (it holds the whole pool).
    Run reserve(uint64_t rows) {
        const uint32_t n = uint32_t((rows + PAGE - 1) / PAGE);
        require(rows && n <= pages_ && admissible(rows) && !exclusive_, "pool reserve without a fit");
        const uint32_t end = exclusive(rows) ? pages_ : place_;
        auto gap = [&](uint32_t& base) {
            uint32_t best = 0, bs = 0;
            for (uint32_t i = 0; i < end;) {
                if (used_[i] || kept_at(i)) { ++i; continue; }
                uint32_t j = i;
                while (j < end && !used_[j] && !kept_at(j)) ++j;
                if (j - i > best) { best = j - i; bs = i; }
                i = j;
            }
            if (best < n) return false;
            base = bs == 0 ? 0 : bs + (best - n) / 2;
            return true;
        };
        auto fit = [&](bool spare_kept, uint32_t& base) {
            uint32_t run = 0; base = 0;
            for (uint32_t i = 0; i < end && run < n; ++i) {
                if (used_[i] || (spare_kept && kept_at(i))) { run = 0; base = i + 1; } else ++run;
            }
            return run == n;
        };
        // no gap free of retained runs: the window of live-free pages that costs the least recently retired runs
        auto lru = [&](uint32_t& base) {
            bool found = false; uint64_t best_age = 0; uint32_t best_pages = 0;
            for (uint32_t b = 0; b + n <= end; ++b) {
                bool live = false;
                for (uint32_t i = b; i < b + n && !live; ++i) live = used_[i];
                if (live) continue;
                uint64_t newest = 0; uint32_t over = 0;
                for (const auto& k : kept_)
                    if (k.run.base_page < b + n && b < k.run.base_page + k.run.pages) { newest = std::max(newest, k.age); over += k.run.pages; }
                if (!found || newest < best_age || (newest == best_age && over < best_pages)) { found = true; best_age = newest; best_pages = over; base = b; }
            }
            return found;
        };
        uint32_t base = 0;
        const bool by_gap = keep_ && !exclusive(rows) && gap(base);
        const bool by_lru = !by_gap && keep_ && !exclusive(rows) && lru(base);
        const bool ok = by_gap || by_lru || fit(false, base);
        how_ = by_gap ? "gap" : by_lru ? "lru" : exclusive(rows) ? "exclusive" : "fit";
        require(ok, "pool reserve: no contiguous run");
        require(version_ != UINT32_MAX, "pool table version exhausted");
        Run r{base, n, ++version_};
        claim(r, rows);
        return r;
    }
    void claim(const Run& r, uint64_t rows) {
        require(r.pages == (rows + PAGE - 1) / PAGE && r.table_version >= version_ && r.table_version, "pool claim differs from the grant");
        require(!exclusive_, "pool claim while an exclusive run is resident");
        if (exclusive(rows)) require(r.base_page == 0 && !used_pages(), "exclusive run not at base 0 of an empty pool");
        else require(uint64_t(r.base_page) + r.pages <= place_, "run below R outside the physical pages");
        const auto gone = evict(r.base_page, r.pages, exclusive(rows));   // an exclusive run holds the whole pool
        if (log_) {
            std::fprintf(stderr, "# s2 pool claim %s base %u pages %u rows %llu live %u kept %u evicted %zu", how_, r.base_page, r.pages,
                         (unsigned long long)rows, used_pages(), (unsigned)kept_.size(), gone.size());
            for (const auto& k : gone) std::fprintf(stderr, " [%u+%u age %llu valid %d]", k.run.base_page, k.run.pages, (unsigned long long)k.age, k.st.valid);
            std::fprintf(stderr, "\n");
        }
        how_ = "claim";
        mark(r, true);
        version_ = r.table_version;
        if (exclusive(rows)) { exclusive_ = true; held_ = r; }
    }
    void release(const Run& r) {
        mark(r, false);
        if (exclusive_ && r.base_page == held_.base_page && r.pages == held_.pages && r.table_version == held_.table_version) {
            exclusive_ = false; held_ = {};
        }
    }
    // prefix reuse on, with these rules (PrefixReuse strict / noend / grid)
    void keep(const g6rt::PrefixReuse& rules) { keep_ = true; proto_ = rules; proto_.clear(); }
    bool keeping() const { return keep_; }
    // a retiring session's run: retained with its finished reuse state (st.valid rows), or released when nothing is valid
    void retain(const Run& r, const g6rt::PrefixReuse& st) {
        release(r);
        if (!keep_ || st.valid <= 0) return;
        Kept k{r, proto_, ++age_};
        k.st.history = st.history; k.st.valid = st.valid; k.st.main_end = st.main_end; k.st.draft_end = st.draft_end;
        k.st.written_end = st.written_end; k.st.dirty_end = st.dirty_end; k.st.pf_end = st.pf_end;
        kept_.push_back(std::move(k));
    }
    // head, at admission: the retained run with the longest reusable prefix for `prompt` (limit: rows the prefill may skip)
    // whose base can hold the request's `rows` without touching a live run; ties go to the most recent
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, int limit) const {
        PoolHit best;
        if (!keep_ || exclusive_) return best;
        const uint32_t n = uint32_t((rows + PAGE - 1) / PAGE);
        uint64_t best_age = 0;
        for (const auto& k : kept_) {
            g6rt::PrefixReuse t = k.st;
            const int skip = t.begin(prompt, limit, true);
            if (skip <= 0) { if (!best.skip && t.lcp >= best.lcp) { best.lcp = t.lcp; best.why = t.why; } continue; }
            const uint32_t b = k.run.base_page;
            bool room = exclusive(rows) ? b == 0 && !used_pages() && n <= pages_ : uint64_t(b) + n <= place_;
            for (uint32_t i = b; room && i < b + n && i < pages_; ++i) room = !used_[i];
            if (!room) { if (!best.skip) best.why = "room"; continue; }
            if (skip > best.skip || (skip == best.skip && k.age > best_age)) {
                best = PoolHit{Run{b, n, 0}, skip, t.lcp, "hit"}; best_age = k.age;
            }
        }
        return best;
    }
    // head: claim a match() hit as a new grant (a fresh table version, as reserve)
    Run take(const PoolHit& h, uint64_t rows) {
        require(h.skip > 0 && admissible(rows), "pool take without a hit");
        require(version_ != UINT32_MAX, "pool table version exhausted");
        Run r{h.run.base_page, h.run.pages, ++version_};
        how_ = "hit";
        claim(r, rows);
        return r;
    }
    void log(bool on) { log_ = on; }
    uint32_t kept_runs() const { return uint32_t(kept_.size()); }
};
} // namespace g6s2
