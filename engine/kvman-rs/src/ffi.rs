//! The C entry points (g6kvman.h). Each returns 0 or an error code that g6kv_error names, and an error leaves the pool
//! unchanged. Release builds abort on a panic, so none unwinds into C++.

use crate::pool::{Error, Pool, Run};
use crate::reuse::{Finished, Rules};
use std::ffi::c_char;

/// g6kv_hit: match's result. `why` points at a static string.
#[repr(C)]
pub struct G6kvHit {
    pub skip: i32,
    pub lcp: i32,
    pub why: *const c_char,
    pub age: u64,
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
pub unsafe extern "C" fn g6kv_admissible(p: *const Pool, rows: u64) -> i32 {
    (*p).admissible(rows) as i32
}

/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_reserve(p: *mut Pool, rows: u64, out: *mut Run) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).reserve(rows) {
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
    limit: i32,
    out: *mut G6kvHit,
) -> i32 {
    let Some(prompt) = slice(prompt, n) else { return Error::BadArgument.code() };
    if out.is_null() {
        return Error::BadArgument.code();
    }
    let h = (*p).match_prefix(prompt, rows, limit);
    *out = G6kvHit { skip: h.skip, lcp: h.lcp, why: h.why.name().as_ptr(), age: h.age };
    0
}

/// Claims a g6kv_match hit (its skip and age).
///
/// # Safety
/// As g6kv_exclusive, and `out` must be valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_take(p: *mut Pool, skip: i32, age: u64, rows: u64, out: *mut Run) -> i32 {
    if out.is_null() {
        return Error::BadArgument.code();
    }
    match (*p).take(skip, age, rows) {
        Ok(r) => {
            *out = r;
            0
        }
        Err(e) => e.code(),
    }
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
