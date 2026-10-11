// S2 below R page allocator over the native pool pages: a thin C++ face on the Rust pool (engine/kvman-rs, whose
// src/lib.rs has the rules: lowest free pages first, exclusive runs past R, prefix reuse with whole retained runs
// evicted least recently retired first). Host only, no CUDA. One instance per rank, every call at an all-rank event
// fence (service.h). Binaries that include it link libg6kvman.a. A pool error throws, with the message of require().
// log(true) (g6m4: G6_POOL_LOG=1): one stderr line per claim, with its pages and every retained run it evicted.
#pragma once
#include "scheduler.h"
#include "../prefix_reuse.h"
#include "../../kvman-rs/g6kvman.h"
#include <utility>
#include <vector>

namespace g6s2 {
// match()'s result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for take()
struct PoolHit { int skip = 0, lcp = 0; const char* why = "none"; uint64_t age = 0; };
class RunPool {
    g6kv_pool* p_ = nullptr;
    static void check(int rc) { require(rc == 0, g6kv_error(rc)); }
    static g6kv_run c(const Run& r) { return g6kv_run{r.first_page, r.pages, r.table_version, r.list_hash}; }
    static Run run(const g6kv_run& r) { return Run{r.first_page, r.pages, r.table_version, r.list_hash}; }
public:
    RunPool(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows) { check(g6kv_new(pool_pages, place_pages, below_rows, &p_)); }
    RunPool(RunPool&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    RunPool(const RunPool&) = delete;
    RunPool& operator=(const RunPool&) = delete;
    ~RunPool() { g6kv_free(p_); }
    bool exclusive(uint64_t rows) const { return g6kv_exclusive(p_, rows); }
    uint32_t free_pages() const { return g6kv_free_pages(p_); }
    // the most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages
    uint32_t room() const { return g6kv_room(p_); }
    // head, before reserve: a past-R request waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t rows) const { return g6kv_admissible(p_, rows); }
    Run reserve(uint64_t rows) { g6kv_run r; check(g6kv_reserve(p_, rows, &r)); return run(r); }
    void release(const Run& r) { const g6kv_run x = c(r); check(g6kv_release(p_, &x)); }
    // a live run's pages, in table order, and whether it is exclusive (its table is the CP map as built)
    std::vector<int> list(const Run& r) const {
        const g6kv_run x = c(r); const uint32_t* pages; size_t n; int ex;
        check(g6kv_list(p_, &x, &pages, &n, &ex));
        return std::vector<int>(pages, pages + n);
    }
    bool exclusive_run(const Run& r) const {
        const g6kv_run x = c(r); const uint32_t* pages; size_t n; int ex;
        check(g6kv_list(p_, &x, &pages, &n, &ex));
        return ex;
    }
    // prefix reuse on, with these rules (PrefixReuse strict / noend / grid)
    void keep(const g6rt::PrefixReuse& rules) { g6kv_keep(p_, rules.strict, rules.noend, rules.grid); }
    bool keeping() const { return g6kv_keeping(p_); }
    // a retiring session's run: retained with its finished reuse state (st.valid rows), or released when nothing is valid
    void retain(const Run& r, const g6rt::PrefixReuse& st) {
        const g6kv_run x = c(r);
        check(g6kv_retain(p_, &x, st.history.data(), st.history.size(), st.valid, st.written_end, st.pf_end));
    }
    // head, at admission (workers at their grant): the retained run with the longest reusable prefix for `prompt` (limit:
    // rows the prefill may skip) that the request's `rows` can grow from. Ties go to the most recent.
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, int limit) const {
        g6kv_hit h;
        check(g6kv_match(p_, prompt.data(), prompt.size(), rows, limit, &h));
        return PoolHit{h.skip, h.lcp, h.why, h.age};
    }
    // claim a match() hit as a new grant (a fresh table version, as reserve)
    Run take(const PoolHit& h, uint64_t rows) { g6kv_run r; check(g6kv_take(p_, h.skip, h.age, rows, &r)); return run(r); }
    // every rank's pool must agree: folded into Scheduler::digest at each event boundary
    uint64_t digest() const { return g6kv_digest(p_); }
    void log(bool on) { g6kv_log(p_, on); }
    uint32_t kept_runs() const { return g6kv_kept_runs(p_); }
};
} // namespace g6s2
