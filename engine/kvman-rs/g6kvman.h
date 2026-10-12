// C interface of the Rust S2 page allocator (engine/kvman-rs). Built as libg6kvman.a with cargo. g6m4 links it, and
// engine/runtime/s2/pool.h (RunPool) wraps it. See src/lib.rs for the allocation rules.
// Calls that can fail return 0 or an error code, which g6kv_error names (the C++ pool's messages). An error leaves the
// pool unchanged.
#pragma once
#include <cstddef>
#include <cstdint>

#define G6KV_NAMESPACE_BYTES 70

extern "C" {

struct g6kv_pool;

// g6s2::Run: list_hash names the page list
struct g6kv_run
{
    uint32_t first_page, pages, table_version;
    uint64_t list_hash;
};

// g6kv_match's result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for g6kv_take.
// why is a static string: "hit", "stored", or why there is none (none, room, first, strict, no-lcp, short, grid).
// stored = 1: the run is in tier `tier` (g6kv_add_tier), and g6kv_take queues the copies that restore its pages.
struct g6kv_hit
{
    int32_t skip, lcp;
    const char* why;
    uint64_t age;
    int32_t stored;
    uint32_t tier;
};

// status.s2 totals: live runs, the pages they hold and could grow to, granted and refused growth
struct g6kv_totals
{
    uint32_t live_runs, held_pages, worst_pages;
    uint64_t grows, stalls;
};
// one live run: pages held now and at worst
struct g6kv_use
{
    uint32_t table_version, held, worst;
};

const char* g6kv_error(int code);

// pool_pages = the logical capacity, place_pages = the physical pages for runs below R, below_rows = R
int g6kv_new(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows, g6kv_pool** out);
void g6kv_free(g6kv_pool* p);

int g6kv_exclusive(const g6kv_pool* p, uint64_t rows);
uint32_t g6kv_free_pages(const g6kv_pool* p);
uint32_t g6kv_room(const g6kv_pool* p);
// a request that needs `rows` now and at most `worst` rows (exclusive past R, by worst): admissible = not blocked by
// an exclusive run, fits = admissible, with room, and leaving the pool safe (every live run can still reach its worst)
int g6kv_admissible(const g6kv_pool* p, uint64_t worst);
int g6kv_fits(const g6kv_pool* p, uint64_t rows, uint64_t worst);
// pages for `rows` now, growing to `worst` at most (an exclusive run takes its worst case at once)
int g6kv_reserve(g6kv_pool* p, uint64_t rows, uint64_t worst, g6kv_run* out);
// a live run grows to the pages `rows` needs: granted = 1 and the run (same table version) in out, or granted = 0 when
// the pages would leave the pool unsafe
int g6kv_grow(g6kv_pool* p, const g6kv_run* r, uint64_t rows, g6kv_run* out, int* granted);
int g6kv_release(g6kv_pool* p, const g6kv_run* r);
// a live run's pages in table order, valid until the next call that changes the pool, and whether it is exclusive
int g6kv_list(const g6kv_pool* p, const g6kv_run* r, const uint32_t** pages, size_t* n, int* exclusive);

// prefix reuse on, with PrefixReuse's strict / noend / grid rules
void g6kv_keep(g6kv_pool* p, int strict, int noend, int32_t grid);
int g6kv_keeping(const g6kv_pool* p);
// retires a run with its session's finished PrefixReuse state, or releases it when valid <= 0 or reuse is off
int g6kv_retain(g6kv_pool* p, const g6kv_run* r, const int32_t* history, size_t n, int32_t valid, int32_t written_end,
                int32_t pf_end);
int g6kv_match(const g6kv_pool* p, const int32_t* prompt, size_t n, uint64_t rows, uint64_t worst, int32_t limit,
               g6kv_hit* out);
// Opaque namespaces: exactly G6KV_NAMESPACE_BYTES bytes, copied during the call. Nonzero identities support
// resident reuse only. Null buffers or any other size return BadArgument without changing the pool/output.
// The existing retain/match/count entry points above/below retain all-zero namespace behavior and their ABI.
int g6kv_retain_namespaced(g6kv_pool* p, const g6kv_run* r, const int32_t* history, size_t n, int32_t valid,
                           int32_t written_end, int32_t pf_end, const uint8_t* namespace_, size_t namespace_bytes);
int g6kv_match_namespaced(const g6kv_pool* p, const int32_t* prompt, size_t n, uint64_t rows, uint64_t worst,
                          int32_t limit, const uint8_t* namespace_, size_t namespace_bytes, g6kv_hit* out);
// from: -1 for a pool hit, else the tier of a stored one
int g6kv_take(g6kv_pool* p, int32_t skip, uint64_t age, int32_t from, uint64_t rows, uint64_t worst, g6kv_run* out);

uint64_t g6kv_digest(const g6kv_pool* p);
// one stderr line per claim (g6m4: G6_POOL_LOG=1)
void g6kv_log(g6kv_pool* p, int on);
uint32_t g6kv_kept_runs(const g6kv_pool* p);
int g6kv_stats(const g6kv_pool* p, g6kv_totals* out);
// writes up to cap live runs into out, returns how many there are
size_t g6kv_runs(const g6kv_pool* p, g6kv_use* out, size_t cap);

// Block stores below the pool (src/stash.rs, --kv-tier): retained runs the pool evicts, and prefixes that started
// shared_after conversations, kept in tiers of blocks. After g6kv_keep: g6kv_stores, then g6kv_add_tier per tier in
// flag order. Each reserve, take, grow or retain may queue copies. The engine takes the copies (g6kv_io_take), runs
// them in order before anything writes those pages, then calls g6kv_io_done. Until then reserve, take and grow fail.
void g6kv_stores(g6kv_pool* p, uint32_t shared_after);
// keep: 0 = evicted runs, 1 = shared prefixes, 2 = both. index_dir (or NULL): the tier's shared prefixes survive
// restarts while `compat` stays the same. 0, or -1 with g6kv_tier_error.
int g6kv_add_tier(g6kv_pool* p, uint64_t slots, int32_t keep, const char* index_dir, const char* compat);
// at admission, before the grant, on every rank: counts the prompt toward its prefixes and returns how many
// span: the prompt's system span in tokens (the system message and tool definitions as templated), or -1 when the
// front end sent none
uint32_t g6kv_count(g6kv_pool* p, const int32_t* prompt, size_t n, int64_t span);
// Call before a namespace-aware admission's grant: nonzero identities count nothing and clear pending stock
// admission metadata. span follows g6kv_count. Returns an error code, with the count in *out on success.
int g6kv_count_namespaced(g6kv_pool* p, const int32_t* prompt, size_t n, int64_t span,
                          const uint8_t* namespace_, size_t namespace_bytes, uint32_t* out);
// tier t's policy (after g6kv_add_tier): evicted runs need min_rows rows and min_hits continuations to be stored there.
// block_bytes: what one block takes, for the counters
int g6kv_tier_policy(g6kv_pool* p, uint32_t t, uint64_t min_rows, uint32_t min_hits, uint64_t block_bytes);
// the write limit: all tiers write at most milli / 1000 blocks per page of KV computed (0 = none)
void g6kv_write_limit(g6kv_pool* p, uint64_t milli);
// one copy: kind 0 = spill (page -> tier's slot), 1 = move (tier's slot -> to's to_slot), 2 = restore (tier's slot -> page).
// part 0: the whole page block of pool page `page`. Past R (g6kv_tail): 1 = this rank's shard of the attention caches at
// physical page `page`; 2 = logical page `page`'s replicated parts (indexer keys, unsharded bindings), the sub-th of the
// block's per_slot. An earlier copy of the list can write the block a later one reads: an engine that runs copies
// together keeps that order.
struct g6kv_io
{
    uint32_t kind, page, tier, to;
    uint64_t slot, to_slot;
    uint32_t part, sub;
};
// runs past R keep their tails: nr ranks share the attention caches past R, a block holds per_slot logical pages'
// replicated parts. After g6kv_stores. nr > 0 also turns the page classes on for the pool itself: a run past R then
// takes its own pages (its prefix and shard below the pool's place pages, the drafter's tail copy above them)
// instead of the whole pool, and g6kv_split reports it.
void g6kv_tail(g6kv_pool* p, uint32_t nr, uint32_t per_slot);
// whether g6kv_tail set the CP map's ranks: a run past R carries its own page list, and the engine builds its tables
// from it (1), or it holds the pool as pages 0, 1, .. on the tables as built (0)
int g6kv_split(const g6kv_pool* p);
size_t g6kv_io_pending(const g6kv_pool* p);
// takes every queued copy into out. out holds cap entries, at least g6kv_io_pending.
int g6kv_io_take(g6kv_pool* p, g6kv_io* out, size_t cap);
// after the copies: commits the index records of the entries they wrote. 0, or -1 with g6kv_tier_error (the pool is
// unchanged)
int g6kv_io_done(g6kv_pool* p);
struct g6kv_tier_totals
{
    uint64_t slots, free_slots;
    uint32_t entries, shared;
    uint64_t spilled_pages, moved_in_pages, restored_pages, hits;
    uint64_t dropped, dropped_pages;     // entries that left with no tier to go to
    uint64_t moved_out, moved_out_pages; // entries passed on to the next tier
    uint64_t refused, refused_pages;     // entries the policy or the write limit refused
    uint64_t full;                       // times the tier had to make room
    uint64_t block_bytes;
    int64_t credit;                      // write limit credit left, blocks (0 without a limit)
};
// tier t's counters, or an error when there is no such tier
int g6kv_tier_stats(const g6kv_pool* p, uint32_t t, g6kv_tier_totals* out);
// writes into buf the entries loaded from indexes, one line each. The ranks compare these lines at boot. Returns the
// length of the whole text plus its NUL.
size_t g6kv_stored_summary(const g6kv_pool* p, char* buf, size_t cap);
// keeps the entries the agreed text lists (the head's intersection of every rank's summary), in its order
void g6kv_keep_agreed(g6kv_pool* p, const char* agreed);

// A block store for those copies (src/blocks.rs): "ssd:PATH", "remote:HOST:PORT" (a lease named `name` on a KV node,
// kvman-remote) or "mem:". Blocks are block_bytes rounded up to 4 KiB (g6kv_tier_block_bytes), and every read and
// write moves one whole block. Calls return 0, or -1 with the reason in g6kv_tier_error (this thread's last).
struct g6kv_tier;
int g6kv_tier_open(const char* spec, const char* name, uint64_t block_bytes, uint64_t blocks, g6kv_tier** out);
void g6kv_tier_free(g6kv_tier* t);
const char* g6kv_tier_error();
uint64_t g6kv_tier_block_bytes(const g6kv_tier* t);
void g6kv_tier_describe(const g6kv_tier* t, char* buf, size_t cap);
int g6kv_tier_write(const g6kv_tier* t, uint64_t block, const void* buf);
int g6kv_tier_read(const g6kv_tier* t, uint64_t block, void* buf);
// n blocks at once, blocks[i] from or into bufs[i], on up to 8 threads (RDMA lanes, O_DIRECT requests)
int g6kv_tier_write_many(const g6kv_tier* t, size_t n, const uint64_t* blocks, void* const* bufs);
int g6kv_tier_read_many(const g6kv_tier* t, size_t n, const uint64_t* blocks, void* const* bufs);
// staging memory a KV node's RDMA device may read and write directly (a no-op for the other stores)
int g6kv_tier_register(const g6kv_tier* t, void* buf, size_t len);
}
