// C interface of the Rust S2 page allocator (engine/kvman-rs). Built as libg6kvman.a with cargo. g6m4 links it, and
// engine/runtime/s2/pool.h (RunPool) wraps it. See src/lib.rs for the allocation rules.
// Calls that can fail return 0 or an error code, which g6kv_error names (the C++ pool's messages). An error leaves the
// pool unchanged.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct g6kv_pool;

// g6s2::Run: list_hash names the page list
struct g6kv_run
{
    uint32_t first_page, pages, table_version;
    uint64_t list_hash;
};

// g6kv_match's result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for g6kv_take.
// why is a static string: "hit", or why there is none (none, room, first, strict, no-lcp, short, grid).
struct g6kv_hit
{
    int32_t skip, lcp;
    const char* why;
    uint64_t age;
};

const char* g6kv_error(int code);

// pool_pages = the logical capacity, place_pages = the physical pages for runs below R, below_rows = R
int g6kv_new(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows, g6kv_pool** out);
void g6kv_free(g6kv_pool* p);

int g6kv_exclusive(const g6kv_pool* p, uint64_t rows);
uint32_t g6kv_free_pages(const g6kv_pool* p);
uint32_t g6kv_room(const g6kv_pool* p);
int g6kv_admissible(const g6kv_pool* p, uint64_t rows);
int g6kv_reserve(g6kv_pool* p, uint64_t rows, g6kv_run* out);
int g6kv_release(g6kv_pool* p, const g6kv_run* r);
// a live run's pages in table order, valid until the next call that changes the pool, and whether it is exclusive
int g6kv_list(const g6kv_pool* p, const g6kv_run* r, const uint32_t** pages, size_t* n, int* exclusive);

// prefix reuse on, with PrefixReuse's strict / noend / grid rules
void g6kv_keep(g6kv_pool* p, int strict, int noend, int32_t grid);
int g6kv_keeping(const g6kv_pool* p);
// retires a run with its session's finished PrefixReuse state, or releases it when valid <= 0 or reuse is off
int g6kv_retain(g6kv_pool* p, const g6kv_run* r, const int32_t* history, size_t n, int32_t valid, int32_t written_end,
                int32_t pf_end);
int g6kv_match(const g6kv_pool* p, const int32_t* prompt, size_t n, uint64_t rows, int32_t limit, g6kv_hit* out);
int g6kv_take(g6kv_pool* p, int32_t skip, uint64_t age, uint64_t rows, g6kv_run* out);

uint64_t g6kv_digest(const g6kv_pool* p);
// one stderr line per claim (g6m4: G6_POOL_LOG=1)
void g6kv_log(g6kv_pool* p, int on);
uint32_t g6kv_kept_runs(const g6kv_pool* p);
}
