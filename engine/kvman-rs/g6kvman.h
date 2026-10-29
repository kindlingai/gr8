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
// Resize (#153): pages below place come in chunks of pages_per_chunk (the last may be shorter), set while every chunk is
// online. An offline chunk holds no live or retained run, so its memory may be released. g6kv_shrink takes n chunks
// offline (the highest no live run holds a page of; retained runs in them are evicted and spill): *done = 1 and the
// chunks in out[0..*got], or *done = 0 when the pool must wait for sessions (nothing changed). It never evicts a live
// run, and refuses (*done = 0) a shrink that would leave a live run unable to reach its worst case. g6kv_grow_pool
// brings n chunks back, the lowest first, after the engine mapped their memory. Errors: before g6kv_chunks (both),
// a size no state allows (shrink), fewer than n chunks offline (grow). g6kv_chunk_online is 0 for k >= g6kv_chunk_count
// (and 1 for every k while chunks are off).
int g6kv_chunks(g6kv_pool* p, uint32_t pages_per_chunk);
uint32_t g6kv_chunk_count(const g6kv_pool* p);
int g6kv_chunk_online(const g6kv_pool* p, uint32_t k);
uint32_t g6kv_place_online(const g6kv_pool* p);
int g6kv_shrink(g6kv_pool* p, uint32_t n, uint32_t* out, size_t cap, size_t* got, int* done);
int g6kv_grow_pool(g6kv_pool* p, uint32_t n, uint32_t* out, size_t cap);
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
    uint64_t oversize, oversize_last;    // entries bigger than the whole tier, and the last one's pages
};
// tier t's counters, or an error when there is no such tier
int g6kv_tier_stats(const g6kv_pool* p, uint32_t t, g6kv_tier_totals* out);
// writes into buf the entries loaded from indexes, one line each. The ranks compare these lines at boot. Returns the
// length of the whole text plus its NUL.
size_t g6kv_stored_summary(const g6kv_pool* p, char* buf, size_t cap);
// A tier whose store is a group lease on a KV node (g6kv_tier_open_group) changes size at a point every rank sees:
// grow = g6kv_store_grow, then g6kv_tier_resize to its blocks, then run the copies; shrink = g6kv_tier_resize to the
// kept segments' blocks, run the copies (g6kv_io_take, g6kv_io_done), then g6kv_store_shrink. resized = 0: the tier
// refuses (an indexed tier, or entries it may not evict).
int g6kv_tier_resize(g6kv_pool* p, uint32_t t, uint64_t slots, int32_t* resized);
// the ages in ms of tier t's coldest blocks, one per per_seg blocks, coldest first (what a head reports). Returns how
// many it wrote, at most cap.
size_t g6kv_tier_coldest(g6kv_pool* p, uint32_t t, uint64_t per_seg, uint64_t* out, size_t cap);
// keeps the entries the agreed text lists (the head's intersection of every rank's summary), in its order
void g6kv_keep_agreed(g6kv_pool* p, const char* agreed);

// A block store for those copies (src/blocks.rs): "ssd:PATH", "remote:HOST:PORT" (a lease named `name` on a KV node,
// kvman-remote) or "mem:", any of them behind "ram:FLOOR+" (a RAM cache the kernel may take back, src/ram.rs).
// Blocks are block_bytes rounded up to 4 KiB (g6kv_tier_block_bytes), and every read and write moves one whole block.
// Calls return 0, or -1 with the reason in g6kv_tier_error (this thread's last).
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
// a RAM cache's counters ("ram:FLOOR+STORE"): out[0] reads from RAM, out[1] reads that went to the store, out[2]
// writes not cached for lack of free memory. 0, or -1 for a store without a RAM cache.
int g6kv_tier_ram_stats(const g6kv_tier* t, uint64_t* out);
// Remote attention rows (src/rows.rs, the engine's G6_RT_STORE tier): a store of `bytes` ("mem:" or "remote:HOST:PORT",
// a lease named `name`), and a helper thread that serves each group's request ring. Its first G6KV_ROWS_HEADER bytes
// hold the layout the records were written under: open writes `layout` into an empty store and refuses a store that
// names another one. The GPU fills a ring's entries, then stores seq (system-scope release); the thread reads each
// entry's record (rec bytes at off) into slots + slot * rec, then stores *done = seq (u64 max: a read failed or an
// entry was out of range). Errors: -1 with g6kv_tier_error.
struct g6kv_rows;
struct g6kv_rows_ent { uint64_t off; uint32_t slot, pad; };
struct g6kv_rows_req { uint64_t seq; uint32_t n, pad; };   // cap g6kv_rows_ent follow it
#define G6KV_ROWS_HEADER 4096
// (pool) remote rows release: a split run's main pages past its prefix (0 = off; needs g6kv_tail, before any grant),
// and the records of each rank's rows store (0 = no limit): a window request whose tail needs more waits
// runs: the most window runs live at once (0 = no limit): the engine's slot pool covers that many sessions past R
int g6kv_window(g6kv_pool* p, uint32_t pages, uint32_t ids, uint32_t runs);
uint32_t g6kv_window_runs(const g6kv_pool* p);
// (pool) whether a retained run holds exactly these pages (a window run's records live while it does)
int g6kv_is_kept(const g6kv_pool* p, const uint32_t* pages, size_t n);
// (pool) a window request of `worst` rows with a record limit: 1 and {records per rank it needs, free, in all} in out
int g6kv_ids_view(const g6kv_pool* p, uint64_t worst, uint64_t* out);
// (pool) a request of `worst` rows past R: 1 and {pages below place, pages above, free below, free above} in out
int g6kv_split_view(const g6kv_pool* p, uint64_t worst, uint64_t* out);
int g6kv_rows_open(const char* spec, const char* name, const char* layout, uint64_t bytes, g6kv_rows** out);
void g6kv_rows_free(g6kv_rows* r);
int g6kv_rows_write(const g6kv_rows* r, uint64_t off, const void* buf, size_t len);
// n pieces of len bytes, piece k at store byte offs[k], into dst + k * len (the engine's prefill staging of released
// pages; not while a decode request is in flight)
int g6kv_rows_gather(const g6kv_rows* r, const uint64_t* offs, size_t n, size_t len, void* dst);
int g6kv_rows_register(const g6kv_rows* r, void* p, size_t len);
// before g6kv_rows_start; returns the group's index
// slots: nslots of `stride` bytes; a record of `rec` bytes lands at byte `off` of its slot (an INT4 record at the tail of
// its FP8 slot, unpacked in place)
int g6kv_rows_group(g6kv_rows* r, g6kv_rows_req* req, uint64_t* done, void* slots, size_t rec, size_t stride, size_t off,
                    uint32_t nslots, uint32_t cap);
void g6kv_rows_start(g6kv_rows* r);
uint64_t g6kv_rows_fetched(const g6kv_rows* r);
void g6kv_rows_describe(const g6kv_rows* r, char* buf, size_t cap);
// This rank's lease in a group on a KV node (spec remote:HOST:PORT): clusters share the node and capacity moves
// between them by whole segments. Every rank of a cluster gives the same group, nr, floor, cap and weight (1000 = 1x,
// 0 = 1000). The lease starts at the floor.
int g6kv_tier_open_group(const char* spec, const char* name, const char* group, uint32_t nr, uint32_t rank,
                         uint64_t floor_bytes, uint64_t cap_bytes, uint32_t weight, uint64_t block_bytes, g6kv_tier** out);
uint64_t g6kv_store_blocks(const g6kv_tier* t);
// a group lease's blocks per segment and segments now; 0 for any other store
uint64_t g6kv_store_seg_blocks(const g6kv_tier* t);
uint64_t g6kv_store_segs(const g6kv_tier* t);
// the node's answer to a report, in segments per rank: every rank grows by grant, or shrinks by give, under txn
struct g6kv_report
{
    uint64_t grant, give, unclaimed;
    uint64_t txn;   // the transaction every rank names in g6kv_store_grow / g6kv_store_shrink; 0 = nothing to do
};
// rank 0 only: want = segments more per rank (0 = none), ages = g6kv_tier_coldest's
int g6kv_store_report(const g6kv_tier* t, uint64_t want, const uint64_t* ages, size_t n, g6kv_report* out);
// grow to segs segments / shrink to segs segments, under the report's transaction txn; the store's blocks after it in
// *blocks (or NULL). 0; G6KV_STORE_REFUSED: the node refused (a transaction no longer open, or no room), every rank
// alike, and the store keeps its size; -1 with g6kv_tier_error: a fault
#define G6KV_STORE_REFUSED 1
int g6kv_store_grow(const g6kv_tier* t, uint64_t segs, uint64_t txn, uint64_t* blocks);
int g6kv_store_shrink(const g6kv_tier* t, uint64_t segs, uint64_t txn, uint64_t* blocks);
// A session's KV in a form independent of the layout (src/canon.rs, #152): per binding and token, in order, the MLA
// latent row and, on full-index layers, the indexer key row (128 codes, then its FP32 scale). Host memory only: the
// caller copies the pages between the device and the buffers. flags[b]: 1 = binding b has index keys, 2 = its cache is
// sharded past R (cp_nr > 0); the blob keeps only bit 1, so a blob exported at one layout imports at any other.
// nr = CP ranks (0: every page on every rank), below_rows = R. list / place: the run's page list (the same on every
// rank) and the main model's pool pages, the bound s2_solo_cp_pages gives s2_list_place (UINT32_MAX: the whole list,
// the tables as built). The idx / idx_bytes arrays are never NULL when nb > 0.
// export: mla[r * nb + b] is rank r's MLA cache of binding b (nr ranks, or 1 without CP), idx likewise (NULL on
// bindings without index keys). *need = the blob's size; 1 when cap is too small (nothing written), 0 = done.
// import: one rank's caches mla[b], idx[b]; *written = rows written, summed over bindings.
// Both return -1 on an error, with the reason in g6kv_canon_error (this thread's last).
const char* g6kv_canon_error();
int g6kv_canon_export(uint32_t mla_row, uint32_t nb, const uint8_t* flags, uint32_t nr, uint32_t below_rows,
                      const uint32_t* list, size_t n_list, uint32_t place, const uint8_t* const* mla,
                      const size_t* mla_bytes, const uint8_t* const* idx, const size_t* idx_bytes, uint64_t first,
                      uint64_t rows, int check, uint8_t* out, size_t cap, size_t* need);
int g6kv_canon_import(const uint8_t* blob, size_t n, uint32_t mla_row, uint32_t nb, const uint8_t* flags, uint32_t nr,
                      uint32_t below_rows, const uint32_t* list, size_t n_list, uint32_t place, uint32_t rank,
                      uint8_t* const* mla, const size_t* mla_bytes, uint8_t* const* idx, const size_t* idx_bytes,
                      uint64_t* written);
// The host ledger of a box a Flash rank shares with a KV node (src/ledger.rs, kvman-ledger): bytes move between the
// GPU (side 0), the node (1) and free (2) as one transfer at a time, begin -> released (the giver let go) -> commit
// (the taker holds them) or abort. Each call is one locked, durable step. 0, or -1 with g6kv_tier_error.
int g6kv_ledger_begin(const char* path, uint32_t from, uint32_t to, uint64_t bytes, uint64_t* gen);
int g6kv_ledger_released(const char* path, uint64_t gen, uint64_t bytes);
int g6kv_ledger_commit(const char* path, uint64_t gen, uint64_t bytes);
int g6kv_ledger_abort(const char* path, uint64_t gen);
// out[0..6]: gen, total, gpu, node, free, floor; out[6..9]: the pending transfer's gen (0: none), phase (1 begun,
// 2 released), bytes
int g6kv_ledger_show(const char* path, uint64_t* out);
}
