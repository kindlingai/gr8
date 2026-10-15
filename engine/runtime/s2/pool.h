// S2 below R page allocator over the native pool pages: a thin C++ face on the Rust pool (engine/kvman-rs, whose
// src/lib.rs has the rules: lowest free pages first, exclusive runs past R, prefix reuse with whole retained runs
// evicted least recently retired first, runs that grow on demand under a safe-state check). Host only, no CUDA. One
// instance per rank. Every rank makes the same calls in the same order: admissions at the event fences (service.h),
// growth and retirement in Driver::step. Calls take a lock: on the head, publish_status reads the pool from the control
// thread. Binaries that include it link libg6kvman.a. A pool error throws, with the message of require().
// log(true) (g6m4: G6_POOL_LOG=1): one stderr line per claim, with its pages and every retained run it evicted.
// stores() + add_tier(): block stores below the pool (kvman-rs src/stash.rs). The engine then takes io() after each
// reserve, take, grow or retain, runs the copies in order before anything writes those pages, and calls io_done().
#pragma once
#include "scheduler.h"
#include "../prefix_reuse.h"
#include "../../kvman-rs/g6kvman.h"
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace g6s2 {
// match()'s result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for take().
// stored: the run is in tier `tier`, and take() queues the copies that restore its pages.
struct PoolHit { int skip = 0, lcp = 0; const char* why = "none"; uint64_t age = 0; bool stored = false; uint32_t tier = 0; };
// status.s2: one live run's pages now and at worst
struct PoolUse { uint32_t table_version = 0, held = 0, worst = 0; };
class RunPool {
    g6kv_pool* p_ = nullptr;
    mutable std::mutex mu_;
    using Lock = std::lock_guard<std::mutex>;
    static void check(int rc) { require(rc == 0, g6kv_error(rc)); }
    static g6kv_run c(const Run& r) { return g6kv_run{r.first_page, r.pages, r.table_version, r.list_hash}; }
    static Run run(const g6kv_run& r) { return Run{r.first_page, r.pages, r.table_version, r.list_hash}; }
public:
    RunPool(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows) { check(g6kv_new(pool_pages, place_pages, below_rows, &p_)); }
    RunPool(RunPool&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    RunPool(const RunPool&) = delete;
    RunPool& operator=(const RunPool&) = delete;
    ~RunPool() { g6kv_free(p_); }
    bool exclusive(uint64_t rows) const { Lock l(mu_); return g6kv_exclusive(p_, rows); }
    uint32_t free_pages() const { Lock l(mu_); return g6kv_free_pages(p_); }
    // the most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages
    uint32_t room() const { Lock l(mu_); return g6kv_room(p_); }
    // head, before reserve: a past-R request (by its worst case) waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t worst) const { Lock l(mu_); return g6kv_admissible(p_, worst); }
    // a request needing `rows` now and `worst` at most: admissible, with room, and leaving every live run able to finish
    bool fits(uint64_t rows, uint64_t worst) const { Lock l(mu_); return g6kv_fits(p_, rows, worst); }
    // pages for `rows` now that grow to `worst` at most (a run past R takes its worst case at once)
    Run reserve(uint64_t rows, uint64_t worst) { Lock l(mu_); g6kv_run r; check(g6kv_reserve(p_, rows, worst, &r)); return run(r); }
    Run reserve(uint64_t rows) { return reserve(rows, rows); }
    // the run grown to the pages `rows` needs (same table version), or nothing: the pages would leave the pool unsafe
    std::optional<Run> grow(const Run& r, uint64_t rows) {
        Lock l(mu_); const g6kv_run x = c(r); g6kv_run out; int granted = 0;
        check(g6kv_grow(p_, &x, rows, &out, &granted));
        return granted ? std::optional<Run>(run(out)) : std::nullopt;
    }
    void release(const Run& r) { Lock l(mu_); const g6kv_run x = c(r); check(g6kv_release(p_, &x)); }
    // a live run's pages, in table order, and whether it is exclusive (its table is the CP map as built)
    std::vector<int> list(const Run& r) const {
        Lock l(mu_); const g6kv_run x = c(r); const uint32_t* pages; size_t n; int ex;
        check(g6kv_list(p_, &x, &pages, &n, &ex));
        return std::vector<int>(pages, pages + n);
    }
    bool exclusive_run(const Run& r) const {
        Lock l(mu_); const g6kv_run x = c(r); const uint32_t* pages; size_t n; int ex;
        check(g6kv_list(p_, &x, &pages, &n, &ex));
        return ex;
    }
    // prefix reuse on, with these rules (PrefixReuse strict / noend / grid)
    void keep(const g6rt::PrefixReuse& rules) { Lock l(mu_); g6kv_keep(p_, rules.strict, rules.noend, rules.grid); }
    bool keeping() const { Lock l(mu_); return g6kv_keeping(p_); }
    // a retiring session's run: retained with its finished reuse state (st.valid rows), or released when nothing is valid
    void retain(const Run& r, const g6rt::PrefixReuse& st) { retain(r, st, st.cache_namespace); }
    void retain(const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
        Lock l(mu_); const g6kv_run x = c(r);
        check(g6kv_retain_namespaced(p_, &x, st.history.data(), st.history.size(), st.valid,
                                    st.written_end, st.pf_end, ns.data(), ns.size()));
    }
    // head, at admission (workers at their grant): the retained run with the longest reusable prefix for `prompt` (limit:
    // rows the prefill may skip) that the request (`rows` now, `worst` at most) can grow from. Ties go to the most recent.
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit,
                  const g6rt::CacheNamespace& ns = {}) const {
        Lock l(mu_); g6kv_hit h;
        check(g6kv_match_namespaced(p_, prompt.data(), prompt.size(), rows, worst, limit, ns.data(), ns.size(), &h));
        return PoolHit{h.skip, h.lcp, h.why, h.age, h.stored != 0, h.tier};
    }
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, int limit, const g6rt::CacheNamespace& ns = {}) const {
        return match(prompt, rows, rows, limit, ns);
    }
    // claim a match() hit as a new grant (a fresh table version, as reserve)
    Run take(const PoolHit& h, uint64_t rows, uint64_t worst) {
        Lock l(mu_); g6kv_run r; check(g6kv_take(p_, h.skip, h.age, h.stored ? int32_t(h.tier) : -1, rows, worst, &r)); return run(r);
    }
    Run take(const PoolHit& h, uint64_t rows) { return take(h, rows, rows); }
    // every rank's pool must agree: folded into Scheduler::digest at each event boundary
    uint64_t digest() const { Lock l(mu_); return g6kv_digest(p_); }
    void log(bool on) { Lock l(mu_); g6kv_log(p_, on); }
    uint32_t kept_runs() const { Lock l(mu_); return g6kv_kept_runs(p_); }
    g6kv_totals stats() const { Lock l(mu_); g6kv_totals s; check(g6kv_stats(p_, &s)); return s; }
    void stores(uint32_t shared_after) { Lock l(mu_); g6kv_stores(p_, shared_after); }
    // keep: 0 evicted, 1 shared, 2 all. index_dir: empty, or the folder of the tier's index
    void add_tier(uint64_t slots, int keep, const std::string& index_dir, const std::string& compat) {
        Lock l(mu_);
        require(g6kv_add_tier(p_, slots, keep, index_dir.empty() ? nullptr : index_dir.c_str(), compat.c_str()) == 0,
                g6kv_tier_error());
    }
    // at admission, before the grant, on every rank. span: the prompt's system span in tokens, -1 = none
    uint32_t count(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace& ns = {}) {
        Lock l(mu_);
        uint32_t out;
        check(g6kv_count_namespaced(p_, prompt.data(), prompt.size(), span, ns.data(), ns.size(), &out));
        return out;
    }
    // tier t's policy: evicted runs need min_rows rows and min_hits continuations; block_bytes for the counters
    void policy(uint32_t t, uint64_t min_rows, uint32_t min_hits, uint64_t block_bytes) {
        Lock l(mu_);
        check(g6kv_tier_policy(p_, t, min_rows, min_hits, block_bytes));
    }
    // runs past R keep their tails: nr ranks share the attention caches past R, per_slot replicated parts a block.
    // nr > 0 also turns the pool's page classes on: a run past R takes its own pages (prefix and shard below place,
    // the drafter's tail copy above it) instead of the whole pool, which split() reports
    void tail(uint32_t nr, uint32_t per_slot) { Lock l(mu_); g6kv_tail(p_, nr, per_slot); }
    // whether tail() set the CP map's ranks: a run past R carries its own page list, and the engine builds its tables
    // from it (true), or it holds the pool as pages 0, 1, .. on the tables as built (false)
    bool split() const { Lock l(mu_); return g6kv_split(p_) != 0; }
    // the stores write at most milli / 1000 blocks per page of KV computed (0 = no limit)
    void write_limit(uint64_t milli) { Lock l(mu_); g6kv_write_limit(p_, milli); }
    // the copies queued since the last io(), in the order to run them. Then io_done()
    std::vector<g6kv_io> io() {
        Lock l(mu_);
        std::vector<g6kv_io> v(g6kv_io_pending(p_));
        check(g6kv_io_take(p_, v.data(), v.size()));
        return v;
    }
    // nullptr, or why an index record could not be written (the pool is unchanged)
    const char* io_done() { Lock l(mu_); return g6kv_io_done(p_) ? g6kv_tier_error() : nullptr; }
    size_t io_pending() const { Lock l(mu_); return g6kv_io_pending(p_); }
    std::vector<g6kv_tier_totals> tiers() const {
        Lock l(mu_);
        std::vector<g6kv_tier_totals> v;
        for (g6kv_tier_totals t; g6kv_tier_stats(p_, uint32_t(v.size()), &t) == 0;) v.push_back(t);
        return v;
    }
    std::string stored_summary() const {
        Lock l(mu_);
        std::string s(g6kv_stored_summary(p_, nullptr, 0), '\0');
        g6kv_stored_summary(p_, s.data(), s.size());
        s.pop_back();
        return s;
    }
    void keep_agreed(const std::string& agreed) { Lock l(mu_); g6kv_keep_agreed(p_, agreed.c_str()); }
    std::vector<PoolUse> runs() const {
        Lock l(mu_);
        std::vector<g6kv_use> u(g6kv_runs(p_, nullptr, 0));
        g6kv_runs(p_, u.data(), u.size());
        std::vector<PoolUse> v;
        for (const auto& x : u) v.push_back(PoolUse{x.table_version, x.held, x.worst});
        return v;
    }
};
} // namespace g6s2
