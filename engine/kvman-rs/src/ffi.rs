//! The C entry points (g6kvman.h). Each returns 0 or an error code that g6kv_error names, and an error leaves the pool
//! unchanged. Release builds abort on a panic, so none unwinds into C++.

use crate::pool::{Error, Pool, Run, Stats};
use crate::reuse::{Finished, Rules};
use crate::stash::{Io, Keep, Part, Policy, TierStats};
use std::ffi::c_char;
use std::ffi::CStr;

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
pub unsafe extern "C" fn g6kv_new(pool_pages: u32, place_pages: u32, below_rows: u32, out: *mut *mut Pool) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match Pool::new(pool_pages, place_pages, below_rows) {
        Ok(p) => {
            *out = Box::into_raw(Box::new(p));
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// `p` must come from g6kv_new (or be NULL) and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6kv_free(p: *mut Pool) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// # Safety
/// `p` must be a live pool from g6kv_new (as for every call below).
#[no_mangle]
pub unsafe extern "C" fn g6kv_exclusive(p: *const Pool, rows: u64) -> i32 {
    (*p).exclusive(rows) as i32
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_free_pages(p: *const Pool) -> u32 {
    (*p).free_pages()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_room(p: *const Pool) -> u32 {
    (*p).room()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_admissible(p: *const Pool, worst: u64) -> i32 {
    (*p).admissible(worst) as i32
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_fits(p: *const Pool, rows: u64, worst: u64) -> i32 {
    (*p).fits(rows, worst) as i32
}

/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_reserve(p: *mut Pool, rows: u64, worst: u64, out: *mut Run) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).reserve(rows, worst) {
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
pub unsafe extern "C" fn g6kv_release(p: *mut Pool, r: *const Run) -> i32 {
    if r.is_null() {
        return Error::BadArgument.code();
    }
    code((*p).release(&*r))
}

/// A live run's pages in table order (borrowed: valid until the next call that changes the pool) and whether it is
/// exclusive.
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read, `pages`, `n` and `exclusive` for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_list(
    p: *const Pool,
    r: *const Run,
    pages: *mut *const u32,
    n: *mut usize,
    exclusive: *mut i32,
) -> i32 {
    if r.is_null() || pages.is_null() || n.is_null() || exclusive.is_null() {
        return Error::BadArgument.code();
    }
    let p = &*p;
    match (p.list(&*r), p.exclusive_run(&*r)) {
        (Ok(l), Ok(x)) => {
            *pages = l.as_ptr();
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
pub unsafe extern "C" fn g6kv_keep(p: *mut Pool, strict: i32, noend: i32, grid: i32) {
    (*p).keep(Rules { strict: strict != 0, noend: noend != 0, grid });
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_keeping(p: *const Pool) -> i32 {
    (*p).keeping() as i32
}

/// Retires a run with its session's finished PrefixReuse state (history, valid, written_end, pf_end).
///
/// # Safety
/// As g6kv_exclusive. `r` must be valid for a read and `history` must point at `n` ints.
#[no_mangle]
pub unsafe extern "C" fn g6kv_retain(
    p: *mut Pool,
    r: *const Run,
    history: *const i32,
    n: usize,
    valid: i32,
    written_end: i32,
    pf_end: i32,
) -> i32 {
    let Some(history) = slice(history, n) else { return Error::BadArgument.code() };
    if r.is_null() {
        return Error::BadArgument.code();
    }
    code((*p).retain(&*r, Finished { history: history.to_vec(), valid, written_end, pf_end }))
}

/// # Safety
/// As g6kv_exclusive. `prompt` must point at `n` ints and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_match(
    p: *const Pool,
    prompt: *const i32,
    n: usize,
    rows: u64,
    worst: u64,
    limit: i32,
    out: *mut G6kvHit,
) -> i32 {
    let Some(prompt) = slice(prompt, n) else { return Error::BadArgument.code() };
    if out.is_null() {
        return Error::BadArgument.code();
    }
    let h = (*p).match_prefix(prompt, rows, worst, limit);
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
    p: *mut Pool,
    skip: i32,
    age: u64,
    from: i32,
    rows: u64,
    worst: u64,
    out: *mut Run,
) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).take_hit(skip, age, (from >= 0).then_some(from as u32), rows, worst) {
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
pub unsafe extern "C" fn g6kv_grow(p: *mut Pool, r: *const Run, rows: u64, out: *mut Run, granted: *mut i32) -> i32 {
    if r.is_null() || out.is_null() || granted.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).grow(&*r, rows) {
        Ok(g) => {
            *granted = g.is_some() as i32;
            *out = g.unwrap_or(*r);
            0
        }
        Err(e) => e.code(),
    }
}

/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stats(p: *const Pool, out: *mut Stats) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    *out = (*p).stats();
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
pub unsafe extern "C" fn g6kv_runs(p: *const Pool, out: *mut G6kvUse, cap: usize) -> usize {
    let mut n = 0;
    for (v, held, worst) in (*p).runs() {
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
pub unsafe extern "C" fn g6kv_digest(p: *const Pool) -> u64 {
    (*p).digest()
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_log(p: *mut Pool, on: i32) {
    (*p).log(on != 0);
}

/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_kept_runs(p: *const Pool) -> u32 {
    (*p).kept_runs()
}

/// Block stores below the pool (src/stash.rs), after g6kv_keep. Then g6kv_add_tier per tier, in flag order.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_stores(p: *mut Pool, shared_after: u32) {
    (*p).stores(shared_after);
}

/// A tier of `slots` blocks: keep 0 = evicted runs, 1 = shared prefixes, 2 = both. index_dir (or NULL) keeps its
/// shared prefixes across restarts, valid while `compat` is the same. 0, or -1 with g6kv_tier_error.
///
/// # Safety
/// As g6kv_exclusive. index_dir and compat are NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_add_tier(
    p: *mut Pool,
    slots: u64,
    keep: i32,
    index_dir: *const c_char,
    compat: *const c_char,
) -> i32 {
    let keep = match keep {
        0 => Keep::Evicted,
        1 => Keep::Shared,
        2 => Keep::All,
        _ => return crate::blocks::fail(std::io::Error::other("keep: 0, 1 or 2")),
    };
    let text = |c: *const c_char| (!c.is_null()).then(|| CStr::from_ptr(c).to_string_lossy().into_owned());
    let (dir, compat) = (text(index_dir), text(compat).unwrap_or_default());
    match (*p).add_tier(slots, keep, dir.as_deref().map(|d| (std::path::Path::new(d), compat.as_str()))) {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// At admission, before the grant, on every rank: counts the prompt toward the prefixes it starts. Returns how many.
/// span: its system span in tokens, or -1 for none.
///
/// # Safety
/// As g6kv_exclusive; `prompt` points at `n` ints.
#[no_mangle]
pub unsafe extern "C" fn g6kv_count(p: *mut Pool, prompt: *const i32, n: usize, span: i64) -> u32 {
    let span = u32::try_from(span).ok();
    slice(prompt, n).map_or(0, |pr| (*p).count(pr, span))
}

/// Tier t's policy (stash.rs Policy). 0, or BadArgument when there is no such tier.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_policy(p: *mut Pool, t: u32, min_rows: u64, min_hits: u32, block_bytes: u64) -> i32 {
    match (*p).set_policy(t as usize, Policy { min_rows, min_hits, block_bytes }) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// The write limit: all tiers write at most milli / 1000 blocks per page of KV computed (0 = none).
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_write_limit(p: *mut Pool, milli: u64) {
    (*p).write_limit(milli);
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
    }
}

/// Runs past R keep their tails in the stores: nr ranks share the attention cache past R, and a block holds per_slot
/// logical pages' replicated parts (the engine's page block layout). After g6kv_stores.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tail(p: *mut Pool, nr: u32, per_slot: u32) {
    (*p).tail(nr, per_slot);
}

/// Copies queued since the last g6kv_io_take.
///
/// # Safety
/// As g6kv_exclusive.
#[no_mangle]
pub unsafe extern "C" fn g6kv_io_pending(p: *const Pool) -> usize {
    (*p).io_pending()
}

/// Takes the queued copies into `out`, in the order to run them. `cap` must hold them all (g6kv_io_pending). Then
/// g6kv_io_done once they ran.
///
/// # Safety
/// As g6kv_exclusive, and `out` must hold `cap` entries.
#[no_mangle]
pub unsafe extern "C" fn g6kv_io_take(p: *mut Pool, out: *mut G6kvIo, cap: usize) -> i32 {
    let p = &mut *p;
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
pub unsafe extern "C" fn g6kv_io_done(p: *mut Pool) -> i32 {
    match (*p).io_done() {
        Ok(()) => 0,
        Err(e) => crate::blocks::fail(e),
    }
}

/// Tier t's counters, or BadArgument when there is no such tier.
///
/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_stats(p: *const Pool, t: u32, out: *mut TierStats) -> i32 {
    match ((*p).tier_stats(t as usize), out.is_null()) {
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
pub unsafe extern "C" fn g6kv_stored_summary(p: *const Pool, buf: *mut c_char, cap: usize) -> usize {
    let s = (*p).stored_summary();
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
pub unsafe extern "C" fn g6kv_keep_agreed(p: *mut Pool, agreed: *const c_char) {
    if !agreed.is_null() {
        (*p).keep_agreed(&CStr::from_ptr(agreed).to_string_lossy());
    }
}
