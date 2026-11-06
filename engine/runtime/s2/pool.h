// S2 below R page allocator over the native pool pages: a thin C++ face on the Rust pool (engine/kvman-rs, whose
// src/lib.rs has the rules: lowest free pages first, exclusive runs past R, prefix reuse with whole retained runs
// evicted least recently retired first, runs that grow on demand under a safe-state check). Host only, no CUDA. One
// instance per rank. Every rank makes the same calls in the same order: admissions at the event fences (service.h),
// growth and retirement in Driver::step. The lock is in kvman-rs, one call at a time: on the head, publish_status reads
// the pool from the control thread. Binaries that include it link libg6host.a. A pool error throws, with the message
// of require().
// log(true) (g6m4: GR8_POOL_LOG=1): one stderr line per claim, with its pages and every retained run it evicted.
// stores() + add_tier(): block stores below the pool (kvman-rs src/stash.rs). The engine then takes io() after each
// reserve, take, grow or retain, runs the copies in order before anything writes those pages, and calls io_done().
#pragma once
#include "scheduler.h"
#include "../prefix_reuse.h"
#include "../../kvman-rs/g6kvman.h"
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace g6s2 {
// match()'s result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for take().
// stored: the run is in tier `tier`, and take() queues the copies that restore its pages. slot (Config::park): the parked
// session slot whose KV the prompt extends (skip = its position), granted that slot; -1 = none
struct PoolHit { int skip = 0, lcp = 0; const char* why = "none"; uint64_t age = 0; bool stored = false; uint32_t tier = 0; int slot = -1; };
// status.s2: one live run's pages now and at worst
struct PoolUse { uint32_t table_version = 0, held = 0, worst = 0; };
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
    // r: the R a grant takes, in rows (0 = the pool's). With window() on it may be a smaller multiple of 64
    bool exclusive(uint64_t rows, uint32_t r = 0) const {
        const int x = g6kv_exclusive_r(p_, rows, r); require(x >= 0, "pool R the pool cannot lay out"); return x;
    }
    uint32_t free_pages() const { return g6kv_free_pages(p_); }
    // the most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages
    uint32_t room() const { return g6kv_room(p_); }
    // head, before reserve: a past-R request (by its worst case) waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t worst, uint32_t r = 0) const { return g6kv_admissible_r(p_, worst, r); }
    // a request needing `rows` now and `worst` at most: admissible, with room, and leaving every live run able to finish
    bool fits(uint64_t rows, uint64_t worst, uint32_t r = 0) const { return g6kv_fits_r(p_, rows, worst, r); }
    // pages for `rows` now that grow to `worst` at most (a run past R takes its worst case at once)
    Run reserve(uint64_t rows, uint64_t worst, uint32_t r = 0) {
        g6kv_run x; check(g6kv_reserve_r(p_, rows, worst, r, &x)); return run(x);
    }
    Run reserve(uint64_t rows) { return reserve(rows, rows); }
    // the run grown to the pages `rows` needs (same table version), or nothing: the pages would leave the pool unsafe
    std::optional<Run> grow(const Run& r, uint64_t rows) {
        const g6kv_run x = c(r); g6kv_run out; int granted = 0;
        check(g6kv_grow(p_, &x, rows, &out, &granted));
        return granted ? std::optional<Run>(run(out)) : std::nullopt;
    }
    void release(const Run& r) { const g6kv_run x = c(r); check(g6kv_release(p_, &x)); }
    // a live run's pages (r.pages of them), in table order, and whether it is exclusive (its table is the CP map as built)
    std::vector<int> list(const Run& r) const {
        const g6kv_run x = c(r); std::vector<int> v(r.pages); size_t n; int ex;
        check(g6kv_list(p_, &x, reinterpret_cast<uint32_t*>(v.data()), v.size(), &n, &ex));
        require(n == v.size(), "pool list length");
        return v;
    }
    bool exclusive_run(const Run& r) const {
        const g6kv_run x = c(r); size_t n; int ex;
        check(g6kv_list(p_, &x, nullptr, 0, &n, &ex));
        return ex;
    }
    // prefix reuse on, with these rules (PrefixReuse strict / noend / grid)
    void keep(const g6rt::PrefixReuse& rules) { g6kv_keep(p_, rules.strict, rules.noend, rules.grid); }
    bool keeping() const { return g6kv_keeping(p_); }
    // a retiring session's run: retained with its finished reuse state (st.valid rows), or released when nothing is valid
    void retain(const Run& r, const g6rt::PrefixReuse& st) { retain(r, st, st.cache_namespace); }
    void retain(const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
        const g6kv_run x = c(r);
        check(g6kv_retain_namespaced(p_, &x, st.history.data(), st.history.size(), st.valid,
                                    st.written_end, st.pf_end, ns.data(), ns.size()));
    }
    // head, at admission (workers at their grant): the retained run with the longest reusable prefix for `prompt` (limit:
    // rows the prefill may skip) that the request (`rows` now, `worst` at most) can grow from. Ties go to the most recent.
    // A retained run past another R than r is no hit.
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit,
                  const g6rt::CacheNamespace& ns = {}, uint32_t r = 0) const {
        g6kv_hit h;
        check(g6kv_match_r(p_, prompt.data(), prompt.size(), rows, worst, limit, ns.data(), ns.size(), r, &h));
        return PoolHit{h.skip, h.lcp, h.why, h.age, h.stored != 0, h.tier};
    }
    PoolHit match(const std::vector<int>& prompt, uint64_t rows, int limit, const g6rt::CacheNamespace& ns = {}) const {
        return match(prompt, rows, rows, limit, ns);
    }
    // claim a match() hit as a new grant (a fresh table version, as reserve)
    Run take(const PoolHit& h, uint64_t rows, uint64_t worst, uint32_t r = 0) {
        g6kv_run x;
        check(g6kv_take_r(p_, h.skip, h.age, h.stored ? int32_t(h.tier) : -1, rows, worst, r, &x));
        return run(x);
    }
    Run take(const PoolHit& h, uint64_t rows) { return take(h, rows, rows); }
    // every rank's pool must agree: folded into Scheduler::digest at each event boundary
    uint64_t digest() const { return g6kv_digest(p_); }
    void log(bool on) { g6kv_log(p_, on); }
    uint32_t kept_runs() const { return g6kv_kept_runs(p_); }
    g6kv_totals stats() const { g6kv_totals s; check(g6kv_stats(p_, &s)); return s; }
    void stores(uint32_t shared_after) { g6kv_stores(p_, shared_after); }
    // keep: 0 evicted, 1 shared, 2 all. index_dir: empty, or the folder of the tier's index
    void add_tier(uint64_t slots, int keep, const std::string& index_dir, const std::string& compat) {
        require(g6kv_add_tier(p_, slots, keep, index_dir.empty() ? nullptr : index_dir.c_str(), compat.c_str()) == 0,
                g6kv_tier_error());
    }
    // at admission, before the grant, on every rank. span: the prompt's system span in tokens, -1 = none
    uint32_t count(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace& ns = {}) {
        uint32_t out;
        check(g6kv_count_namespaced(p_, prompt.data(), prompt.size(), span, ns.data(), ns.size(), &out));
        return out;
    }
    // tier t's policy: evicted runs need min_rows rows and min_hits continuations; block_bytes for the counters
    void policy(uint32_t t, uint64_t min_rows, uint32_t min_hits, uint64_t block_bytes) {
        check(g6kv_tier_policy(p_, t, min_rows, min_hits, block_bytes));
    }
    // runs past R keep their tails: nr ranks share the attention caches past R, per_slot replicated parts a block.
    // nr > 0 also turns the pool's page classes on: a run past R takes its own pages (prefix and shard below place,
    // the drafter's tail copy above it) instead of the whole pool, which split() reports
    void tail(uint32_t nr, uint32_t per_slot) { g6kv_tail(p_, nr, per_slot); }
    // whether tail() set the CP map's ranks: a run past R carries its own page list, and the engine builds its tables
    // from it (true), or it holds the pool as pages 0, 1, .. on the tables as built (false)
    bool split() const { return g6kv_split(p_) != 0; }
    // remote rows release (GR8_RT_LOCAL): a split run takes its prefix and `pages` main pages, its whole tail above place
    // ids: the records of each rank's rows store (0 = no limit): a request whose tail needs more waits
    // runs: the most window runs live at once, what the slot pool covers (0 = no limit)
    void window(uint32_t pages, uint32_t ids, uint32_t runs) { check(g6kv_window(p_, pages, ids, runs)); }
    uint32_t window_runs() const { return g6kv_window_runs(p_); }
    uint32_t window_runs(uint32_t r) const { return g6kv_window_runs_r(p_, r); }
    // a live window run takes the smaller R = r2 rows (kvman-rs Pool::shrink_run): same table version and length, a new list
    bool shrinkable(const Run& r, uint32_t r2) const { const g6kv_run x = c(r); return g6kv_shrinkable(p_, &x, r2) != 0; }
    Run shrink(const Run& r, uint32_t r2) { const g6kv_run x = c(r); g6kv_run out; check(g6kv_shrink_run(p_, &x, r2, &out)); return run(out); }
    // whether a retained run holds exactly this list (a window run's records stay while it does)
    bool is_kept(const std::vector<int>& list) const {
        std::vector<uint32_t> v(list.begin(), list.end()); return g6kv_is_kept(p_, v.data(), v.size()) != 0;
    }
    // a window request of `worst` rows with a record limit: {records per rank it needs, free, in all}, or nothing
    std::optional<std::array<uint64_t, 3>> ids_view(uint64_t worst, uint32_t r = 0) const {
        std::array<uint64_t, 3> v{};
        return g6kv_ids_view_r(p_, worst, r, v.data()) ? std::optional<std::array<uint64_t, 3>>(v) : std::nullopt;
    }
    // a past-R request of `worst` rows: {pages below place, above, free below, free above}, or nothing
    std::optional<std::array<uint64_t, 4>> split_view(uint64_t worst, uint32_t r = 0) const {
        std::array<uint64_t, 4> v{};
        return g6kv_split_view_r(p_, worst, r, v.data()) ? std::optional<std::array<uint64_t, 4>>(v) : std::nullopt;
    }
    // the stores write at most milli / 1000 blocks per page of KV computed (0 = no limit)
    void write_limit(uint64_t milli) { g6kv_write_limit(p_, milli); }
    // the copies queued since the last io(), in the order to run them. Then io_done()
    std::vector<g6kv_io> io() {
        std::vector<g6kv_io> v(g6kv_io_pending(p_));
        check(g6kv_io_take(p_, v.data(), v.size()));
        return v;
    }
    // nullptr, or why an index record could not be written (the pool is unchanged)
    const char* io_done() { return g6kv_io_done(p_) ? g6kv_tier_error() : nullptr; }
    size_t io_pending() const { return g6kv_io_pending(p_); }
    std::vector<g6kv_tier_totals> tiers() const {
        std::vector<g6kv_tier_totals> v;
        for (g6kv_tier_totals t; g6kv_tier_stats(p_, uint32_t(v.size()), &t) == 0;) v.push_back(t);
        return v;
    }
    std::string stored_summary() const {
        std::string s(g6kv_stored_summary(p_, nullptr, 0), '\0');
        g6kv_stored_summary(p_, s.data(), s.size());
        s.pop_back();
        return s;
    }
    void keep_agreed(const std::string& agreed) { g6kv_keep_agreed(p_, agreed.c_str()); }
    // tier t to `slots` blocks (its store, a group lease, grew or is about to shrink; s2/kv_share.h). false: the tier
    // refuses. A shrink queues moves into the slots that stay: run io() before the store gives its top back
    bool resize_tier(uint32_t t, uint64_t slots) {
        int32_t ok = 0; check(g6kv_tier_resize(p_, t, slots, &ok)); return ok != 0;
    }
    // the ages in ms a head reports for tier t: one per per_seg blocks, coldest first, at most `most`
    std::vector<uint64_t> coldest(uint32_t t, uint64_t per_seg, size_t most) {
        std::vector<uint64_t> v(most); v.resize(g6kv_tier_coldest(p_, t, per_seg, v.data(), v.size())); return v;
    }
    // resize (#153): chunks of pages below place (kvman-rs g6kvman.h). shrink: the chunks now offline, or nothing while
    // the pool must wait for sessions to finish. grow: the chunks back online, after their memory is mapped
    void chunks(uint32_t pages_per_chunk) { check(g6kv_chunks(p_, pages_per_chunk)); }
    uint32_t chunk_count() const { return g6kv_chunk_count(p_); }
    bool chunk_online(uint32_t k) const { return g6kv_chunk_online(p_, k) != 0; }
    uint32_t place_online() const { return g6kv_place_online(p_); }
    std::optional<std::vector<uint32_t>> shrink(uint32_t n) {
        std::vector<uint32_t> v(n); size_t got = 0; int done = 0;
        check(g6kv_shrink(p_, n, v.data(), v.size(), &got, &done));
        if (!done) return std::nullopt;
        v.resize(got);
        return v;
    }
    std::vector<uint32_t> grow_pool(uint32_t n) {
        std::vector<uint32_t> v(n); check(g6kv_grow_pool(p_, n, v.data(), v.size())); return v;
    }
    // publish_status calls this on the control thread while the compute thread changes the pool, so the count can
    // change between the two calls: fill again until the runs fit
    std::vector<PoolUse> runs() const {
        std::vector<g6kv_use> u(g6kv_runs(p_, nullptr, 0));
        for (size_t n; (n = g6kv_runs(p_, u.data(), u.size())) != u.size();) u.resize(n);
        std::vector<PoolUse> v;
        for (const auto& x : u) v.push_back(PoolUse{x.table_version, x.held, x.worst});
        return v;
    }
};
} // namespace g6s2
