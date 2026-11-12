//! The C entry points (g6kvman.h). Each returns 0 or an error code that g6kv_error names, and an error leaves the pool
//! unchanged. Release builds abort on a panic, so none unwinds into C++. Each call on a pool locks it for that call
//! only, so two threads may call at once. A call never calls out to C++ while it holds the lock.

use crate::pool::{Error, Namespace, Pool, Run, Stats, NAMESPACE_BYTES, STOCK_NAMESPACE};
use crate::reuse::{Finished, Rules};
use crate::stash::{Io, Keep, Part, Policy, TierStats};
use std::ffi::c_char;
use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};

/// The pool behind its lock: what g6kv_new hands out as a g6kv_pool.
pub struct Handle(Mutex<Pool>);

impl Handle {
    pub fn new(p: Pool) -> Self {
        Handle(Mutex::new(p))
    }

    /// The pool for one call. Where a panic unwinds (dev and test builds), it poisons the lock, and the next call takes
    /// the pool as it is. Release builds abort.
    pub fn lock(&self) -> MutexGuard<'_, Pool> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// g6kv_hit: match's result. `why` points at a static string. stored: the run is in the store.
#[repr(C)]
pub struct G6kvHit {
    pub skip: i32,
    pub lcp: i32,
    pub why: *const c_char,
    pub age: u64,
    pub stored: i32,
    pub tier: u32,
}

fn code(r: Result<(), Error>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> Option<&'a [T]> {
    if n == 0 {
        Some(&[])
    } else if p.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(p, n))
    }
}

unsafe fn namespace(p: *const u8, n: usize) -> Option<Namespace> {
    if n != NAMESPACE_BYTES || p.is_null() {
        return None;
    }
    slice(p, n)?.try_into().ok()
}

/// Names an error code with a static string: "" for 0, "unknown pool error" for a code no call returns.
#[no_mangle]
pub extern "C" fn g6kv_error(code: i32) -> *const c_char {
    match Error::from_code(code) {
        Some(e) => e.message().as_ptr(),
        None if code == 0 => c"".as_ptr(),
        None => c"unknown pool error".as_ptr(),
    }
}

/// Makes a pool in `*out`.
///
/// # Safety
/// `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_new(pool_pages: u32, place_pages: u32, below_rows: u32, out: *mut *mut Handle) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match Pool::new(pool_pages, place_pages, below_rows) {
        Ok(p) => {
            *out = Box::into_raw(Box::new(Handle::new(p)));
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// `p` must come from g6kv_new (or be NULL) and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6kv_free(p: *mut Handle) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// # Safety
/// `p` must be a live pool from g6kv_new (as for every call below).
#[no_mangle]
pub unsafe extern "C" fn g6kv_exclusive(p: *const Handle, rows: u64) -> i32 {
    (*p).lock().exclusive(rows) as i32
}

/// g6kv_exclusive at R = r rows (0 = the pool's R): -1 for an R the pool refuses.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_exclusive_r(p: *const Handle, rows: u64, r: u32) -> i32 {
    (*p).lock().exclusive_r(rows, r).map_or(-1, |x| x as i32)
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_free_pages(p: *const Handle) -> u32 {
    (*p).lock().free_pages()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_room(p: *const Handle) -> u32 {
    (*p).lock().room()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_admissible(p: *const Handle, worst: u64) -> i32 {
    g6kv_admissible_r(p, worst, 0)
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_admissible_r(p: *const Handle, worst: u64, r: u32) -> i32 {
    (*p).lock().admissible_r(worst, r) as i32
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_fits(p: *const Handle, rows: u64, worst: u64) -> i32 {
    g6kv_fits_r(p, rows, worst, 0)
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_fits_r(p: *const Handle, rows: u64, worst: u64, r: u32) -> i32 {
    (*p).lock().fits_r(rows, worst, r) as i32
}

/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_reserve(p: *mut Handle, rows: u64, worst: u64, out: *mut Run) -> i32 {
    g6kv_reserve_r(p, rows, worst, 0, out)
}

/// # Safety
/// As g6kv_reserve.
#[no_mangle]
pub unsafe extern "C" fn g6kv_reserve_r(p: *mut Handle, rows: u64, worst: u64, r: u32, out: *mut Run) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).lock().reserve_r(rows, worst, r) {
        Ok(r) => {
            *out = r;
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// As g6kv_exclusive, and `r` must be valid for a read.
#[no_mangle]
pub unsafe extern "C" fn g6kv_release(p: *mut Handle, r: *const Run) -> i32 {
    if r.is_null() {
        return Error::BadArgument.code();
    }
    code((*p).lock().release(&*r))
}

/// A live run's pages in table order and whether it is exclusive: copies up to `cap` pages into `pages` and sets `*n`
/// to the run's length, which may be more than cap.
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read, `pages` for cap writes (or NULL with cap 0), `n` and `exclusive`
/// for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_list(
    p: *const Handle,
    r: *const Run,
    pages: *mut u32,
    cap: usize,
    n: *mut usize,
    exclusive: *mut i32,
) -> i32 {
    if r.is_null() || (pages.is_null() && cap > 0) || n.is_null() || exclusive.is_null() {
        return Error::BadArgument.code();
    }
    let p = (*p).lock();
    match (p.list(&*r), p.exclusive_run(&*r)) {
        (Ok(l), Ok(x)) => {
            let k = l.len().min(cap);
            if k > 0 {
                std::ptr::copy_nonoverlapping(l.as_ptr(), pages, k);
            }
            *n = l.len();
            *exclusive = x as i32;
            0
        }
        (Err(e), _) | (_, Err(e)) => e.code(),
    }
}

/// Prefix reuse on, with PrefixReuse's strict / noend / grid rules.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_keep(p: *mut Handle, strict: i32, noend: i32, grid: i32) {
    (*p).lock().keep(Rules { strict: strict != 0, noend: noend != 0, grid });
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_keeping(p: *const Handle) -> i32 {
    (*p).lock().keeping() as i32
}

/// Retires a run with its session's finished PrefixReuse state (history, valid, written_end, pf_end).
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read and `history` must point at `n` ints.
#[no_mangle]
pub unsafe extern "C" fn g6kv_retain(
    p: *mut Handle,
    r: *const Run,
    history: *const i32,
    n: usize,
    valid: i32,
    written_end: i32,
    pf_end: i32,
) -> i32 {
    g6kv_retain_namespaced(p, r, history, n, valid, written_end, pf_end, STOCK_NAMESPACE.as_ptr(), NAMESPACE_BYTES)
}

/// Retire a run in an opaque namespace. Invalid sizes/pointers leave the pool unchanged.
///
/// # Safety
/// As g6kv_retain. A nonnull namespace of the required size must point to that many readable bytes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_retain_namespaced(
    p: *mut Handle,
    r: *const Run,
    history: *const i32,
    n: usize,
    valid: i32,
    written_end: i32,
    pf_end: i32,
    namespace_: *const u8,
    namespace_bytes: usize,
) -> i32 {
    let Some(namespace) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(history) = slice(history, n) else { return Error::BadArgument.code() };
    if p.is_null() || r.is_null() {
        return Error::BadArgument.code();
    }
    code((*p).lock().retain_namespaced(
        &*r,
        Finished { history: history.to_vec(), valid, written_end, pf_end },
        namespace,
    ))
}

/// # Safety
/// As g6kv_exclusive. `prompt` must point at `n` ints and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_match(
    p: *const Handle,
    prompt: *const i32,
    n: usize,
    rows: u64,
    worst: u64,
    limit: i32,
    out: *mut G6kvHit,
) -> i32 {
    g6kv_match_namespaced(p, prompt, n, rows, worst, limit, STOCK_NAMESPACE.as_ptr(), NAMESPACE_BYTES, out)
}

/// Find a prefix only in the exact namespace. Non-stock queries never search stored tiers.
///
/// # Safety
/// As g6kv_match. A nonnull namespace of the required size must point to that many readable bytes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_match_namespaced(
    p: *const Handle,
    prompt: *const i32,
    n: usize,
    rows: u64,
    worst: u64,
    limit: i32,
    namespace_: *const u8,
    namespace_bytes: usize,
    out: *mut G6kvHit,
) -> i32 {
    g6kv_match_r(p, prompt, n, rows, worst, limit, namespace_, namespace_bytes, 0, out)
}

/// g6kv_match_namespaced for a grant at R = r rows (0 = the pool's R). BadArgument for an R the pool refuses.
///
/// # Safety
/// As g6kv_match_namespaced.
#[no_mangle]
pub unsafe extern "C" fn g6kv_match_r(
    p: *const Handle,
    prompt: *const i32,
    n: usize,
    rows: u64,
    worst: u64,
    limit: i32,
    namespace_: *const u8,
    namespace_bytes: usize,
    r: u32,
    out: *mut G6kvHit,
) -> i32 {
    let Some(namespace) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(prompt) = slice(prompt, n) else { return Error::BadArgument.code() };
    if p.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    let p = (*p).lock();
    if p.exclusive_r(0, r).is_err() {
        return Error::BadArgument.code();
    }
    let h = p.match_prefix_r(prompt, rows, worst, limit, namespace, r);
    *out = G6kvHit {
        skip: h.skip,
        lcp: h.lcp,
        why: h.why.name().as_ptr(),
        age: h.age,
        stored: h.stored as i32,
        tier: h.tier,
    };
    0
}

/// Claims a g6kv_match hit: its skip and age, and `from` = -1 for the pool or the tier of a stored hit.
///
/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_take(
    p: *mut Handle,
    skip: i32,
    age: u64,
    from: i32,
    rows: u64,
    worst: u64,
    out: *mut Run,
) -> i32 {
    g6kv_take_r(p, skip, age, from, rows, worst, 0, out)
}

/// g6kv_take for a g6kv_match_r hit at R = r rows (0 = the pool's R).
///
/// # Safety
/// As g6kv_take.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn g6kv_take_r(
    p: *mut Handle,
    skip: i32,
    age: u64,
    from: i32,
    rows: u64,
    worst: u64,
    r: u32,
    out: *mut Run,
) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).lock().take_hit_r(skip, age, (from >= 0).then_some(from as u32), rows, worst, r) {
        Ok(r) => {
            *out = r;
            0
        }
        Err(e) => e.code(),
    }
}

/// Grows a live run to the pages `rows` needs: `*granted` = 1 and the run in `*out`, or 0 when the growth would leave
/// the pool unsafe (the session sits out the round).
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read, `out` and `granted` for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_grow(p: *mut Handle, r: *const Run, rows: u64, out: *mut Run, granted: *mut i32) -> i32 {
    if r.is_null() || out.is_null() || granted.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).lock().grow(&*r, rows) {
        Ok(g) => {
            *granted = g.is_some() as i32;
            *out = g.unwrap_or(*r);
            0
        }
        Err(e) => e.code(),
    }
}

/// Resize (#153): pages below place in chunks of `pages_per_chunk`. Only while every chunk is online.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_chunks(p: *mut Handle, pages_per_chunk: u32) -> i32 {
    code((*p).lock().chunks(pages_per_chunk))
}

/// Chunks (0 when resize is off), and whether chunk k is online.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_chunk_count(p: *const Handle) -> u32 {
    (*p).lock().chunk_count()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_chunk_online(p: *const Handle, k: u32) -> i32 {
    (*p).lock().chunk_online(k) as i32
}

/// The online pages below place.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_place_online(p: *const Handle) -> u32 {
    (*p).lock().place_online()
}

/// Takes `n` chunks offline (Pool::shrink): `*done` = 1 and the chunks in out[0..*got], ascending, or `*done` = 0 when
/// the pool must wait for sessions to finish (nothing changed). out holds cap entries, at least n.
///
/// # Safety
/// As g6kv_exclusive. `out` must be valid for cap writes, `got` and `done` for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_shrink(
    p: *mut Handle,
    n: u32,
    out: *mut u32,
    cap: usize,
    got: *mut usize,
    done: *mut i32,
) -> i32 {
    if (out.is_null() && n > 0) || got.is_null() || done.is_null() || cap < n as usize {
        return Error::BadArgument.code();
    }
    match (*p).lock().shrink(n) {
        Ok(Some(v)) => {
            for (i, &k) in v.iter().enumerate() {
                *out.add(i) = k;
            }
            *got = v.len();
            *done = 1;
            0
        }
        Ok(None) => {
            *got = 0;
            *done = 0;
            0
        }
        Err(e) => e.code(),
    }
}

/// Brings `n` offline chunks back, the lowest first (Pool::grow_pool), into out[0..n], ascending. The engine maps their
/// memory first.
///
/// # Safety
/// As g6kv_exclusive. `out` must be valid for cap writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_grow_pool(p: *mut Handle, n: u32, out: *mut u32, cap: usize) -> i32 {
    if (out.is_null() && n > 0) || cap < n as usize {
        return Error::BadArgument.code();
    }
    match (*p).lock().grow_pool(n) {
        Ok(v) => {
            for (i, &k) in v.iter().enumerate() {
                *out.add(i) = k;
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stats(p: *const Handle, out: *mut Stats) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    *out = (*p).lock().stats();
    0
}

/// g6kv_use: one live run's pages held now and at worst.
#[repr(C)]
pub struct G6kvUse {
    pub table_version: u32,
    pub held: u32,
    pub worst: u32,
}

/// Writes up to `cap` live runs (by table version) into `out` and returns how many there are.
///
/// # Safety
/// As g6kv_exclusive, and `out` must hold `cap` entries (or be NULL with cap 0).
#[no_mangle]
pub unsafe extern "C" fn g6kv_runs(p: *const Handle, out: *mut G6kvUse, cap: usize) -> usize {
    let mut n = 0;
    for (v, held, worst) in (*p).lock().runs() {
        if n < cap && !out.is_null() {
            *out.add(n) = G6kvUse { table_version: v, held, worst };
        }
        n += 1;
    }
    n
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_digest(p: *const Handle) -> u64 {
    (*p).lock().digest()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_log(p: *mut Handle, on: i32) {
    (*p).lock().log(on != 0);
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_kept_runs(p: *const Handle) -> u32 {
    (*p).lock().kept_runs()
}

/// Block stores below the pool (src/stash.rs), after g6kv_keep. Then g6kv_add_tier per tier, in flag order.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stores(p: *mut Handle, shared_after: u32) {
    (*p).lock().stores(shared_after);
}

/// A tier of `slots` blocks: keep 0 = evicted runs, 1 = shared prefixes, 2 = both. index_dir (or NULL) keeps its
/// shared prefixes across restarts, valid while `compat` is the same. 0, or -1 with g6kv_tier_error.
///
/// # Safety
/// As g6kv_exclusive. index_dir and compat are NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_add_tier(
    p: *mut Handle,
    slots: u64,
    keep: i32,
    index_dir: *const c_char,
    compat: *const c_char,
) -> i32 {
    // + 4: the index records every entry, evicted runs too (a persist=on ssd store with keep=sessions, whose runs the
    // quit stores so the next boot brings the sessions back)
    let whole = keep & 4 != 0 && !index_dir.is_null();
    let keep = match keep & !4 {
        0 => Keep::Evicted,
        1 => Keep::Shared,
        2 => Keep::All,
        _ => return crate::blocks::fail(std::io::Error::other("keep: 0, 1 or 2, + 4 for a whole index")),
    };
    let text = |c: *const c_char| (!c.is_null()).then(|| CStr::from_ptr(c).to_string_lossy().into_owned());
    let (dir, compat) = (text(index_dir), text(compat).unwrap_or_default());
    let mut pool = (*p).lock();
    let r = pool.add_tier(slots, keep, dir.as_deref().map(|d| (std::path::Path::new(d), compat.as_str())));
    if let (true, Ok(()), Some(st)) = (whole, &r, pool.stash_mut()) {
        st.index_whole(st.tiers() - 1);
    }
    match r {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// After the last g6kv_add_tier: drops the index runs whose base came back in no tier. Returns how many.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tiers_loaded(p: *mut Handle) -> u64 {
    (*p).lock().tiers_loaded() as u64
}

/// At admission, before the grant, on every rank: counts the prompt toward the prefixes it starts. Returns how many.
/// span: its system span in tokens, or -1 for none.
///
/// # Safety
/// As g6kv_exclusive; `prompt` points at `n` ints.
#[no_mangle]
pub unsafe extern "C" fn g6kv_count(p: *mut Handle, prompt: *const i32, n: usize, span: i64) -> u32 {
    let span = u32::try_from(span).ok();
    slice(prompt, n).map_or(0, |pr| (*p).lock().count(pr, span))
}

/// Prepare a namespace-aware admission. Non-stock prompts are excluded from stock counting/promotion.
///
/// # Safety
/// As g6kv_match, with `out` valid for a u32 write and a readable namespace buffer of the required size.
#[no_mangle]
pub unsafe extern "C" fn g6kv_count_namespaced(
    p: *mut Handle,
    prompt: *const i32,
    n: usize,
    span: i64,
    namespace_: *const u8,
    namespace_bytes: usize,
    out: *mut u32,
) -> i32 {
    let Some(namespace) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(prompt) = slice(prompt, n) else { return Error::BadArgument.code() };
    if p.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    *out = (*p).lock().count_namespaced(prompt, u32::try_from(span).ok(), namespace);
    0
}

/// Tier t's policy (stash.rs Policy). 0, or BadArgument when there is no such tier.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_policy(
    p: *mut Handle,
    t: u32,
    min_rows: u64,
    min_hits: u32,
    block_bytes: u64,
) -> i32 {
    match (*p).lock().set_policy(t as usize, Policy { min_rows, min_hits, block_bytes }) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// The write limit: all tiers write at most milli / 1000 blocks per page of KV computed (0 = none).
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_write_limit(p: *mut Handle, milli: u64) {
    (*p).lock().write_limit(milli);
}

/// g6kv_io: one copy. kind 0 = spill (page -> tier's slot), 1 = move (tier's slot -> to's to_slot), 2 = restore (tier's
/// slot -> page). part (stash.rs Part): 0 = the whole page block, 1 = past R, this rank's shard at physical page `page`,
/// 2 = past R, logical page `page`'s replicated parts, the sub-th in the block.
#[repr(C)]
pub struct G6kvIo {
    pub kind: u32,
    pub page: u32,
    pub tier: u32,
    pub to: u32,
    pub slot: u64,
    pub to_slot: u64,
    pub part: u32,
    pub sub: u32,
}

fn part_code(p: Part) -> (u32, u32) {
    match p {
        Part::Full => (0, 0),
        Part::Own => (1, 0),
        Part::Rep(i) => (2, i),
        Part::State(i) => (3, i),
    }
}

/// Runs past R keep their tails in the stores: nr ranks share the attention cache past R, and a block holds per_slot
/// logical pages' replicated parts (the engine's page block layout). After g6kv_stores. nr > 0 also turns the pool's
/// page classes on: a run past R takes its own pages instead of the whole pool.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tail(p: *mut Handle, nr: u32, per_slot: u32) {
    (*p).lock().tail(nr, per_slot);
}

/// Remote rows release: a split run's main pages past its prefix, the records of each rank's rows store and the most
/// window runs live at once (0 = no limit for either) (src/pool.rs Pool::window). Before any reservation.
///
/// # Safety
/// `p` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_window(p: *mut Handle, pages: u32, ids: u32, runs: u32) -> i32 {
    match (*p).lock().window(pages, ids, runs) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Whether a retained run holds exactly these `n` pages.
///
/// # Safety
/// `p` must be live, `pages` valid for `n` reads.
#[no_mangle]
pub unsafe extern "C" fn g6kv_is_kept(p: *const Handle, pages: *const u32, n: usize) -> i32 {
    (*p).lock().is_kept(std::slice::from_raw_parts(pages, n)) as i32
}

/// The window runs live now.
///
/// # Safety
/// `p` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_window_runs(p: *const Handle) -> u32 {
    (*p).lock().window_runs()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_window_runs_r(p: *const Handle, r: u32) -> u32 {
    (*p).lock().window_runs_r(r)
}

/// Whether live window run `*r` can shrink to R = r2 rows (Pool::shrinkable).
///
/// # Safety
/// As g6kv_exclusive, and `r` must be valid for a read.
#[no_mangle]
pub unsafe extern "C" fn g6kv_shrinkable(p: *const Handle, r: *const Run, r2: u32) -> i32 {
    (!r.is_null() && (*p).lock().shrinkable(&*r, r2)) as i32
}

/// Live window run `*r` takes R = r2 rows (Pool::shrink_run); the run it becomes in `*out`.
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read, `out` for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_shrink_run(p: *mut Handle, r: *const Run, r2: u32, out: *mut Run) -> i32 {
    if r.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).lock().shrink_run(&*r, r2) {
        Ok(x) => {
            *out = x;
            0
        }
        Err(e) => e.code(),
    }
}

/// For a window request of `worst` rows past R with a lease limit: 1 and {records its tail needs per rank, records
/// free, records in all} in out[0..3]; 0 otherwise.
///
/// # Safety
/// `p` must be live, `out` valid for 3 writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_ids_view(p: *const Handle, worst: u64, out: *mut u64) -> i32 {
    g6kv_ids_view_r(p, worst, 0, out)
}

/// g6kv_ids_view for a grant at R = r rows (0 = the pool's R): 0 for an R the pool refuses.
///
/// # Safety
/// As g6kv_ids_view.
#[no_mangle]
pub unsafe extern "C" fn g6kv_ids_view_r(p: *const Handle, worst: u64, r: u32, out: *mut u64) -> i32 {
    match (*p).lock().ids_view_r(worst, r) {
        Some((a, b, c)) => {
            for (k, v) in [a, b, c].into_iter().enumerate() {
                *out.add(k) = v;
            }
            1
        }
        None => 0,
    }
}

/// For a request of `worst` rows past R with the page classes on: 1 and its pages below / above place and what those
/// classes have free in out[0..4]; 0 otherwise.
///
/// # Safety
/// `p` must be live, `out` valid for 4 writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_split_view(p: *const Handle, worst: u64, out: *mut u64) -> i32 {
    g6kv_split_view_r(p, worst, 0, out)
}

/// g6kv_split_view for a grant at R = r rows (0 = the pool's R): 0 for an R the pool refuses.
///
/// # Safety
/// As g6kv_split_view.
#[no_mangle]
pub unsafe extern "C" fn g6kv_split_view_r(p: *const Handle, worst: u64, r: u32, out: *mut u64) -> i32 {
    match (*p).lock().split_view_r(worst, r) {
        Some((a, b, c, d)) => {
            for (k, v) in [a, b, c as u64, d as u64].into_iter().enumerate() {
                *out.add(k) = v;
            }
            1
        }
        None => 0,
    }
}

/// Whether g6kv_tail set the CP map's ranks: a run past R carries its own page list (1) or holds the pool as pages
/// 0, 1, .. (0).
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_split(p: *const Handle) -> i32 {
    (*p).lock().split() as i32
}

/// Copies queued since the last g6kv_io_take.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_io_pending(p: *const Handle) -> usize {
    (*p).lock().io_pending()
}

/// Takes the queued copies into `out`, in the order to run them. `cap` must hold them all (g6kv_io_pending). Then
/// g6kv_io_done once they ran.
///
/// # Safety
/// As g6kv_exclusive, and `out` must hold `cap` entries.
#[no_mangle]
pub unsafe extern "C" fn g6kv_io_take(p: *mut Handle, out: *mut G6kvIo, cap: usize) -> i32 {
    let mut p = (*p).lock();
    if p.io_pending() > cap || (cap > 0 && out.is_null()) {
        return Error::BadArgument.code();
    }
    for (i, x) in p.io().into_iter().enumerate() {
        *out.add(i) = match x {
            Io::Spill { tier, page, slot, part } => {
                let (part, sub) = part_code(part);
                G6kvIo { kind: 0, page, tier, to: 0, slot, to_slot: 0, part, sub }
            }
            Io::Move { tier, slot, to, to_slot } => {
                G6kvIo { kind: 1, page: 0, tier, to, slot, to_slot, part: 0, sub: 0 }
            }
            Io::Restore { tier, slot, page, part } => {
                let (part, sub) = part_code(part);
                G6kvIo { kind: 2, page, tier, to: 0, slot, to_slot: 0, part, sub }
            }
        };
    }
    0
}

/// The copies ran: commits what they wrote to the tiers' indexes. 0, or -1 with g6kv_tier_error (the pool is unchanged;
/// only a restart would miss those entries).
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_io_done(p: *mut Handle) -> i32 {
    let r = (*p).lock().io_done();
    match r {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Tier t's counters, or BadArgument when there is no such tier.
///
/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_stats(p: *const Handle, t: u32, out: *mut TierStats) -> i32 {
    match ((*p).lock().tier_stats(t as usize), out.is_null()) {
        (Some(s), false) => {
            *out = s;
            0
        }
        _ => Error::BadArgument.code(),
    }
}

/// Writes into `buf` (NUL-terminated, cut to `cap`) the stored entries the ranks compare at boot, one line each. Returns
/// the length of the whole text plus its NUL.
///
/// # Safety
/// As g6kv_exclusive; `buf` holds `cap` bytes (or is NULL with cap 0).
#[no_mangle]
pub unsafe extern "C" fn g6kv_stored_summary(p: *const Handle, buf: *mut c_char, cap: usize) -> usize {
    let s = (*p).lock().stored_summary();
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
    s.len() + 1
}

/// Keeps the stored entries that `agreed` names (the head's intersection of every rank's summary), in the text's order.
///
/// # Safety
/// As g6kv_exclusive; `agreed` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_keep_agreed(p: *mut Handle, agreed: *const c_char) {
    if !agreed.is_null() {
        (*p).lock().keep_agreed(&CStr::from_ptr(agreed).to_string_lossy());
    }
}

/// Remote attention rows (src/rows.rs): opens `spec` ("mem:" or "remote:HOST:PORT", a lease named `name`) of `bytes`
/// (the header included), and writes or checks its layout header. 0, or -1 with g6kv_tier_error.
///
/// # Safety
/// `spec`, `name` and `layout` must be NUL-terminated strings, `out` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_open(
    spec: *const c_char,
    name: *const c_char,
    layout: *const c_char,
    bytes: u64,
    out: *mut *mut crate::rows::Rows,
) -> i32 {
    if spec.is_null() || name.is_null() || layout.is_null() || out.is_null() {
        return crate::blocks::fail(std::io::Error::new(std::io::ErrorKind::InvalidInput, "null argument"));
    }
    let (spec, name) = (CStr::from_ptr(spec).to_string_lossy(), CStr::from_ptr(name).to_string_lossy());
    let layout = CStr::from_ptr(layout).to_string_lossy();
    match crate::rows::Store::open(&spec, &name, bytes).and_then(|s| s.layout(&layout).map(|()| s)) {
        Ok(s) => {
            *out = Box::into_raw(Box::new(crate::rows::Rows::new(s)));
            0
        }
        Err(e) => crate::blocks::fail(e),
    }
}

/// Stops the helper thread and closes the store.
///
/// # Safety
/// `r` must come from g6kv_rows_open (or be NULL) and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_free(r: *mut crate::rows::Rows) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

/// Writes `len` bytes at byte `off` of the store. Not while a request is in flight on the mem: store.
///
/// # Safety
/// `r` must be live, `buf` valid for `len` reads.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_write(r: *const crate::rows::Rows, off: u64, buf: *const u8, len: usize) -> i32 {
    match (*r).store.write(off, std::slice::from_raw_parts(buf, len)) {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Reads `n` pieces of `len` bytes, piece k at store byte `offs[k]`, into `dst + k * len`.
///
/// # Safety
/// `r` must be live, `offs` valid for `n` reads, `dst` for `n * len` writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_gather(
    r: *const crate::rows::Rows,
    offs: *const u64,
    n: usize,
    len: usize,
    dst: *mut u8,
) -> i32 {
    if n == 0 {
        return 0;
    }
    let ops: Vec<(u64, *mut u8, usize)> =
        std::slice::from_raw_parts(offs, n).iter().enumerate().map(|(k, &o)| (o, dst.add(k * len), len)).collect();
    match (*r).store.read_scatter(&ops) {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Memory the store's RDMA device may write into (the slots); a no-op without RDMA.
///
/// # Safety
/// `r` must be live, the memory must outlive it.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_register(r: *const crate::rows::Rows, p: *mut u8, len: usize) -> i32 {
    match (*r).register(p, len) {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Adds a group before g6kv_rows_start: its request ring (a g6kv_rows_req and `cap` entries), done word and `nslots`
/// slots of `stride` bytes each; a record of `rec` bytes goes at byte `off` of its slot. Returns the group's index, or
/// -1 with g6kv_tier_error.
///
/// # Safety
/// `r` must be live; the buffers must outlive it and stay mapped while its thread runs.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_group(
    r: *mut crate::rows::Rows,
    req: *mut crate::rows::Req,
    done: *mut u64,
    slots: *mut u8,
    rec: usize,
    stride: usize,
    off: usize,
    nslots: u32,
    cap: u32,
) -> i32 {
    match (*r).add_group(req, done, slots, rec, stride, off, nslots, cap) {
        Ok(i) => i as i32,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Starts the helper thread.
///
/// # Safety
/// `r` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_start(r: *mut crate::rows::Rows) {
    (*r).start();
}

/// Records fetched so far.
///
/// # Safety
/// `r` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_fetched(r: *const crate::rows::Rows) -> u64 {
    (*r).fetched.load(std::sync::atomic::Ordering::Relaxed)
}

/// Writes the store's description into `buf`, NUL-terminated and cut to `cap`.
///
/// # Safety
/// `r` must be live and `buf` hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rows_describe(r: *const crate::rows::Rows, buf: *mut c_char, cap: usize) {
    if buf.is_null() || cap == 0 {
        return;
    }
    let d = (*r).store.describe();
    let n = d.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(d.as_ptr() as *const c_char, buf, n);
    *buf.add(n) = 0;
}

/// Resizes tier t to `slots` (its store grew, or is about to shrink; stash.rs resize_tier) at a point every rank
/// sees: *resized = 1, or 0 when the tier refuses (an indexed tier, or entries it may not evict). A shrink evicts the
/// least recently used entries and queues the moves into the slots that stay: run them (g6kv_io_take, g6kv_io_done)
/// before the store gives its top back (g6kv_store_shrink).
///
/// # Safety
/// `p` must be a live pool, `resized` valid for a write (or NULL).
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_resize(p: *mut Handle, t: u32, slots: u64, resized: *mut i32) -> i32 {
    match (*p).lock().resize_tier(t as usize, slots) {
        Ok(ok) => {
            if !resized.is_null() {
                *resized = ok as i32;
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// The ages in ms a head reports (g6kv_store_report): ages[i] is that of the entry holding tier t's (i x per_seg)-th
/// coldest block, coldest first. Writes up to `cap` and returns how many.
///
/// # Safety
/// `p` must be a live pool and `out` hold `cap` values.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_coldest(p: *mut Handle, t: u32, per_seg: u64, out: *mut u64, cap: usize) -> usize {
    if out.is_null() || cap == 0 {
        return 0;
    }
    let a = (*p).lock().coldest_ages(t as usize, per_seg, cap);
    let n = a.len().min(cap);
    std::ptr::copy_nonoverlapping(a.as_ptr(), out, n);
    n
}

/// Standalone tiered store behind its lock (`struct g6kv_stash`), for slot-slab and hybrid boundary-state engines.
pub struct StashHandle(Mutex<crate::stash::Stash>);

impl StashHandle {
    pub fn new(s: crate::stash::Stash) -> Self {
        StashHandle(Mutex::new(s))
    }

    pub fn lock(&self) -> MutexGuard<'_, crate::stash::Stash> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Creates a standalone `g6kv_stash` in `*out`.
///
/// # Safety
/// `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_new(
    shared_after: u32,
    grid: u64,
    state_blocks: u32,
    mtp: i32,
    out: *mut *mut StashHandle,
) -> i32 {
    if out.is_null() || (state_blocks > 0 && grid != 0 && !grid.is_multiple_of(crate::pool::PAGE)) {
        return Error::BadArgument.code();
    }
    let mut s = crate::stash::Stash::new(shared_after, grid);
    s.set_state_mode(state_blocks, mtp != 0);
    *out = Box::into_raw(Box::new(StashHandle::new(s)));
    0
}

/// # Safety
/// `s` must come from `g6kv_stash_new` (or be NULL) and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_free(s: *mut StashHandle) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// # Safety
/// `s` must be a live stash from `g6kv_stash_new`.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_rules(s: *mut StashHandle, strict: i32, noend: i32, grid: i32) {
    if !s.is_null() {
        (*s).lock().set_rules(Rules { strict: strict != 0, noend: noend != 0, grid });
    }
}

/// # Safety
/// `s` must be live; `index_dir` and `compat` are NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_add_tier(
    s: *mut StashHandle,
    slots: u64,
    keep: i32,
    index_dir: *const c_char,
    compat: *const c_char,
) -> i32 {
    if s.is_null() || slots == 0 {
        return crate::blocks::fail(std::io::Error::other("stash add_tier: null handle or zero slots"));
    }
    let whole = keep & 4 != 0 && !index_dir.is_null();
    let keep = match keep & !4 {
        0 => Keep::Evicted,
        1 => Keep::Shared,
        2 => Keep::All,
        _ => return crate::blocks::fail(std::io::Error::other("keep: 0, 1 or 2, + 4 for a whole index")),
    };
    let text = |c: *const c_char| (!c.is_null()).then(|| CStr::from_ptr(c).to_string_lossy().into_owned());
    let (dir, compat) = (text(index_dir), text(compat).unwrap_or_default());
    let index = match dir {
        Some(d) => match crate::stash::Index::open(std::path::Path::new(&d), &compat) {
            Ok(ix) => Some(ix),
            Err(e) => return crate::blocks::fail(e),
        },
        None => None,
    };
    let mut st = (*s).lock();
    st.add_tier(slots, keep, index);
    if whole {
        let last = st.tiers() - 1;
        st.index_whole(last);
    }
    0
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_tiers_loaded(s: *mut StashHandle) -> u64 {
    if s.is_null() {
        return 0;
    }
    (*s).lock().loaded() as u64
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_tier_policy(
    s: *mut StashHandle,
    t: u32,
    min_rows: u64,
    min_hits: u32,
    block_bytes: u64,
) -> i32 {
    if s.is_null() {
        return Error::BadArgument.code();
    }
    let mut st = (*s).lock();
    if (t as usize) >= st.tiers() {
        return Error::BadArgument.code();
    }
    st.set_policy(t as usize, Policy { min_rows, min_hits, block_bytes });
    0
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_write_limit(s: *mut StashHandle, milli: u64) {
    if !s.is_null() {
        (*s).lock().set_limit(milli);
    }
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_earn(s: *mut StashHandle, pages: u64) {
    if !s.is_null() {
        (*s).lock().earn_pages(pages);
    }
}

/// # Safety
/// `s` must be live; `prompt` points at `n` ints, `namespace_` at `namespace_bytes` bytes, `out` valid for a write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn g6kv_stash_count(
    s: *mut StashHandle,
    prompt: *const i32,
    n: usize,
    span: i64,
    namespace_: *const u8,
    namespace_bytes: usize,
    resident_continuing: i32,
    out: *mut u32,
) -> i32 {
    let Some(ns) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(pr) = slice(prompt, n) else { return Error::BadArgument.code() };
    if s.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    *out = (*s).lock().count_admission(pr, u32::try_from(span).ok(), ns, resident_continuing != 0);
    0
}

/// # Safety
/// `s` must be live; `prompt` points at `n` ints, `namespace_` at `namespace_bytes` bytes, `out` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_match(
    s: *const StashHandle,
    prompt: *const i32,
    n: usize,
    limit: i32,
    namespace_: *const u8,
    namespace_bytes: usize,
    out: *mut G6kvHit,
) -> i32 {
    let Some(ns) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(pr) = slice(prompt, n) else { return Error::BadArgument.code() };
    if s.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    let h = (*s).lock().match_state(pr, limit, ns);
    *out = G6kvHit {
        skip: h.skip,
        lcp: h.lcp,
        why: h.why.name().as_ptr(),
        age: h.age,
        stored: h.stored as i32,
        tier: h.tier,
    };
    0
}

/// # Safety
/// `s` must be live; `pages` points at `n_pages` u32s. Output pointers may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn g6kv_stash_restore(
    s: *mut StashHandle,
    tier: u32,
    id: u64,
    pages: *const u32,
    n_pages: usize,
    state_handle: u32,
    out_valid: *mut i32,
    out_hits: *mut u32,
    out_pages: *mut usize,
) -> i32 {
    let Some(pg) = slice(pages, n_pages) else { return Error::BadArgument.code() };
    if s.is_null() {
        return Error::BadArgument.code();
    }
    match (*s).lock().restore_entry(tier as usize, id, pg, state_handle) {
        Ok((valid, hits, full)) => {
            if !out_valid.is_null() {
                *out_valid = valid;
            }
            if !out_hits.is_null() {
                *out_hits = hits;
            }
            if !out_pages.is_null() {
                *out_pages = full;
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// `s` must be live; `history`, `pages`, `namespace_`, `resident`/`resident_lens`, and `out_id` must be valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn g6kv_stash_capture(
    s: *mut StashHandle,
    history: *const i32,
    n_history: usize,
    valid: i32,
    written_end: i32,
    pf_end: i32,
    pages: *const u32,
    n_pages: usize,
    state_handle: u32,
    hits: u32,
    namespace_: *const u8,
    namespace_bytes: usize,
    from_id: u64,
    from_pages: usize,
    keep_from: i32,
    keep_id: u64,
    resident: *const *const i32,
    resident_lens: *const usize,
    n_resident: usize,
    out_id: *mut u64,
) -> i32 {
    let Some(ns) = namespace(namespace_, namespace_bytes) else { return Error::BadArgument.code() };
    let Some(hist) = slice(history, n_history) else { return Error::BadArgument.code() };
    let Some(pg) = slice(pages, n_pages) else { return Error::BadArgument.code() };
    if s.is_null() || out_id.is_null() {
        return Error::BadArgument.code();
    }
    let mut res_slices: Vec<&[i32]> = Vec::with_capacity(n_resident);
    if n_resident > 0 {
        if resident.is_null() || resident_lens.is_null() {
            return Error::BadArgument.code();
        }
        for i in 0..n_resident {
            let len = *resident_lens.add(i);
            let ptr = *resident.add(i);
            let Some(sl) = slice(ptr, len) else { return Error::BadArgument.code() };
            res_slices.push(sl);
        }
    }
    let from = (from_id != 0).then_some((from_id, from_pages));
    let keep = (keep_id != 0).then_some(keep_id);
    let fin = Finished { history: hist.to_vec(), valid, written_end, pf_end };
    match (*s).lock().capture_entry(fin, pg, state_handle, hits, ns, from, keep_from != 0, keep, &res_slices) {
        Ok(id) => {
            *out_id = id.unwrap_or(0);
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// `s` must be live; `history` and `pages` must point at `n_history` and `n_pages` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn g6kv_stash_promote(
    s: *mut StashHandle,
    history: *const i32,
    n_history: usize,
    valid: i32,
    written_end: i32,
    pf_end: i32,
    pages: *const u32,
    n_pages: usize,
    state_handle: u32,
    rows_max: u64,
    span: i64,
    keep_id: u64,
) -> i32 {
    let Some(hist) = slice(history, n_history) else { return Error::BadArgument.code() };
    let Some(pg) = slice(pages, n_pages) else { return Error::BadArgument.code() };
    if s.is_null() {
        return Error::BadArgument.code();
    }
    let fin = Finished { history: hist.to_vec(), valid, written_end, pf_end };
    code((*s).lock().promote_state(
        &fin,
        pg,
        state_handle,
        rows_max,
        u32::try_from(span).ok(),
        (keep_id != 0).then_some(keep_id),
    ))
}

/// # Safety
/// `s` must be live; `dropped` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_drop(s: *mut StashHandle, id: u64, dropped: *mut i32) -> i32 {
    if s.is_null() {
        return Error::BadArgument.code();
    }
    match (*s).lock().drop_entry(id) {
        Ok(ok) => {
            if !dropped.is_null() {
                *dropped = ok as i32;
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_has(s: *const StashHandle, id: u64) -> i32 {
    if s.is_null() || id == 0 {
        return 0;
    }
    (*s).lock().has_entry(id) as i32
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_io_pending(s: *const StashHandle) -> usize {
    if s.is_null() {
        return 0;
    }
    (*s).lock().io_pending_count()
}

/// # Safety
/// `s` must be live; `out` holds `cap` entries.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_io_take(s: *mut StashHandle, out: *mut G6kvIo, cap: usize) -> i32 {
    if s.is_null() {
        return Error::BadArgument.code();
    }
    let mut st = (*s).lock();
    if st.io_pending_count() > cap || (cap > 0 && out.is_null()) {
        return Error::BadArgument.code();
    }
    for (i, x) in st.take_pending_io().into_iter().enumerate() {
        *out.add(i) = match x {
            Io::Spill { tier, page, slot, part } => {
                let (part, sub) = part_code(part);
                G6kvIo { kind: 0, page, tier, to: 0, slot, to_slot: 0, part, sub }
            }
            Io::Move { tier, slot, to, to_slot } => {
                G6kvIo { kind: 1, page: 0, tier, to, slot, to_slot, part: 0, sub: 0 }
            }
            Io::Restore { tier, slot, page, part } => {
                let (part, sub) = part_code(part);
                G6kvIo { kind: 2, page, tier, to: 0, slot, to_slot: 0, part, sub }
            }
        };
    }
    0
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_io_done(s: *mut StashHandle) -> i32 {
    if s.is_null() {
        return Error::BadArgument.code();
    }
    let r = (*s).lock().finish_io();
    match r {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// # Safety
/// `s` must be live and `out` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_tier_stats(s: *const StashHandle, t: u32, out: *mut TierStats) -> i32 {
    if s.is_null() || out.is_null() {
        return Error::BadArgument.code();
    }
    match (*s).lock().stats(t as usize) {
        Some(st) => {
            *out = st;
            0
        }
        None => Error::BadArgument.code(),
    }
}

/// # Safety
/// `s` must be live; `buf` holds `cap` bytes (or is NULL with cap 0).
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_stored_summary(s: *const StashHandle, buf: *mut c_char, cap: usize) -> usize {
    if s.is_null() {
        return 0;
    }
    let text = (*s).lock().summary();
    if !buf.is_null() && cap > 0 {
        let n = text.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(text.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
    text.len() + 1
}

/// # Safety
/// `s` must be live; `agreed` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_keep_agreed(s: *mut StashHandle, agreed: *const c_char) {
    if !s.is_null() && !agreed.is_null() {
        (*s).lock().keep_agreed(&CStr::from_ptr(agreed).to_string_lossy());
    }
}

/// # Safety
/// `s` must be live; `resized` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_tier_resize(s: *mut StashHandle, t: u32, slots: u64, resized: *mut i32) -> i32 {
    if s.is_null() {
        return Error::BadArgument.code();
    }
    let mut st = (*s).lock();
    if st.pending() {
        return Error::PendingIo.code();
    }
    let ok = st.resize_tier(t as usize, slots);
    if !resized.is_null() {
        *resized = ok as i32;
    }
    0
}

/// # Safety
/// `s` must be live and `out` hold `cap` values.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_tier_coldest(
    s: *mut StashHandle,
    t: u32,
    per_seg: u64,
    out: *mut u64,
    cap: usize,
) -> usize {
    if s.is_null() || out.is_null() || cap == 0 {
        return 0;
    }
    let a = (*s).lock().coldest_ages(t as usize, per_seg, cap);
    let n = a.len().min(cap);
    std::ptr::copy_nonoverlapping(a.as_ptr(), out, n);
    n
}

/// # Safety
/// `s` must be live.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stash_digest(s: *const StashHandle) -> u64 {
    if s.is_null() {
        return 0;
    }
    (*s).lock().digest()
}
