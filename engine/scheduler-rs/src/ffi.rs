//! The C entry points (g6sched.h). Each fallible call returns 0 or a code the C++ wrapper turns into its message; a
//! failed call leaves its state unchanged. Release builds abort on a panic, so none unwinds into C++.
//!
//! Safety, for every function here: pointers are valid for the counts passed with them (null only with a count of
//! 0), and state pointers point at the matching C++ struct.
#![allow(clippy::missing_safety_doc)]

use crate::cp_rows::{self, Cand};
use crate::flash_lend::Lend;
use crate::flash_ring::{self, Ring};
use crate::kv_plan::{self, Io, Item};

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    debug_assert!(n == 0 || !p.is_null());
    if n == 0 || p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

unsafe fn slice_mut<'a, T>(p: *mut T, n: usize) -> &'a mut [T] {
    debug_assert!(n == 0 || !p.is_null());
    if n == 0 || p.is_null() {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(p, n)
    }
}

/// g6sch_batch: next_batch's result, in buffers the caller owns (`items_cap` items, `slots_cap` slots).
#[repr(C)]
pub struct Batch {
    pub kind: u32,
    pub items: *mut Item,
    pub items_cap: i32,
    pub n_items: i32,
    pub from: *mut u32,
    pub to: *mut u32,
    pub slots: *mut u64,
    pub to_slots: *mut u64,
    pub slots_cap: i32,
    pub n_slots: i32,
}

/// 0, 1 = no copy fits, 2 = a buffer is too small.
#[no_mangle]
pub unsafe extern "C" fn g6sch_kv_next_batch(
    io: *const Io,
    n: usize,
    i: *mut usize,
    batch: i32,
    per_slot: u32,
    items_cap: i32,
    out: *mut Batch,
) -> i32 {
    let (io, out) = (slice(io, n), &mut *out);
    let mut at = *i;
    let Some(b) = kv_plan::next_batch(io, &mut at, batch, per_slot, items_cap) else { return 1 };
    if b.items.len() > out.items_cap as usize || b.slots.len() > out.slots_cap as usize {
        return 2;
    }
    out.kind = b.kind;
    slice_mut(out.items, b.items.len()).copy_from_slice(&b.items);
    out.n_items = b.items.len() as i32;
    slice_mut(out.from, b.from.len()).copy_from_slice(&b.from);
    slice_mut(out.to, b.to.len()).copy_from_slice(&b.to);
    slice_mut(out.slots, b.slots.len()).copy_from_slice(&b.slots);
    slice_mut(out.to_slots, b.to_slots.len()).copy_from_slice(&b.to_slots);
    out.n_slots = b.slots.len() as i32;
    *i = at;
    0
}

#[no_mangle]
pub unsafe extern "C" fn g6sch_lend_need(l: *const Lend, rows: u64) -> u32 {
    (*l).need(rows)
}

/// The chunk count a lend takes the ranks to, or -1 for none.
#[no_mangle]
pub unsafe extern "C" fn g6sch_lend_poll(l: *mut Lend, idle: bool, now_ns: i64) -> i64 {
    (*l).poll(idle, now_ns).map_or(-1, i64::from)
}

#[no_mangle]
pub unsafe extern "C" fn g6sch_lend_recall(l: *const Lend, rows: u64, room: u32) -> u32 {
    (*l).recall(rows, room)
}

/// The slot the ring's next step at `next` loads (relative to the session's base) into *src: 0, or 1 = outside the
/// previous step.
#[no_mangle]
pub unsafe extern "C" fn g6sch_ring_source(r: *const Ring, next: i32, src: *mut i32) -> i32 {
    match (*r).source(next) {
        Ok(s) => {
            *src = s;
            0
        }
        Err(_) => 1,
    }
}

/// 0, 1 = outside the previous step, 2 = a verify wider than the ring (or no rows). kslot holds n entries.
#[no_mangle]
pub unsafe extern "C" fn g6sch_ring_rows(
    r: *mut Ring,
    per: i32,
    q: i32,
    pos: i32,
    n: i32,
    verify: bool,
    kslot: *mut i32,
    src: *mut i32,
) -> i32 {
    if n < 1 || (verify && n > per) {
        return 2; // checked before kslot is read as n entries
    }
    match flash_ring::rows(&mut *r, per, q, pos, n, verify, slice_mut(kslot, n as usize)) {
        Ok(s) => {
            *src = s;
            0
        }
        Err(flash_ring::Error::OutsidePrevious) => 1,
        Err(flash_ring::Error::VerifyTooWide) => 2,
    }
}

/// 0, 1 = invalid arguments, 2 = the set exceeds mx. Each flag array holds mx + 1 entries, indexed by row count.
#[no_mangle]
pub unsafe extern "C" fn g6sch_flash_shapes(
    sessions: i32,
    mtp: i32,
    chunk: i32,
    adapt: bool,
    mx: i32,
    mains: *mut u8,
    drafts: *mut u8,
    chains: *mut u8,
) -> i32 {
    match flash_ring::shapes(sessions, mtp, chunk, adapt, mx) {
        Ok(f) => {
            for (dst, src) in [(mains, &f.mains), (drafts, &f.drafts), (chains, &f.chains)] {
                for (d, &s) in slice_mut(dst, (mx + 1) as usize).iter_mut().zip(src.iter()) {
                    *d = u8::from(s);
                }
            }
            0
        }
        Err(flash_ring::ShapesError::Invalid) => 1,
        Err(flash_ring::ShapesError::ExceedsMx) => 2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn g6sch_cp_after(bk: *const i32, nbk: i32, r: i32) -> u64 {
    cp_rows::cp_after(slice(bk, nbk.max(0) as usize), r)
}

/// 0, or 1 = below solo (refused).
#[no_mangle]
pub unsafe extern "C" fn g6sch_cp_step_rows(v: i64, solo: i32, full: i32, out: *mut i32) -> i32 {
    match cp_rows::cp_step_rows(v, solo, full) {
        Some(n) => {
            *out = n;
            0
        }
        None => 1,
    }
}

/// in_round holds n entries.
#[no_mangle]
pub unsafe extern "C" fn g6sch_cp_pick(
    cand: *const Cand,
    n: usize,
    served: *const u64,
    n_served: usize,
    rows: u32,
    per: u32,
    seg: bool,
    in_round: *mut bool,
    capped: *mut bool,
    fresh: *mut bool,
) {
    let p = cp_rows::pick(slice(cand, n), slice(served, n_served), rows, per, seg);
    slice_mut(in_round, n).copy_from_slice(&p.in_round);
    *capped = p.capped;
    *fresh = p.fresh;
}

// ---- the scheduler (scheduler.rs) --------------------------------------------------------------------------------
// Every call takes the slots as (pointer, count) and returns null or the C++ message of its refusal (a static string).

use crate::scheduler::{self as sch, Commit, Completion, Config, Key, Quantum, Request, Run, Slot, State};
use std::ffi::c_char;

fn err(r: Result<(), sch::Err>) -> *const c_char {
    match r {
        Ok(()) => std::ptr::null(),
        Err(m) => msg(m),
    }
}
// each refusal message as a NUL-terminated static string
fn msg(m: &'static str) -> *const c_char {
    macro_rules! table {
        ($($s:literal),* $(,)?) => {
            match m {
                $($s => concat!($s, "\0").as_ptr() as *const c_char,)*
                _ => "scheduler-rs: unknown refusal\0".as_ptr() as *const c_char,
            }
        };
    }
    table!(
        "empty request",
        "request exceeds pool including guard",
        "bad epoch/pool",
        "bad session/row limit",
        "c8 prefill pack is DS41 only",
        "parked slots are DS41 only",
        "insufficient guard/prefill rows",
        "decode_prefill_rows must be 0 or a multiple of 64 no larger than prefill_rows",
        "decode_rounds must be at most 64",
        "notice_max must be 2..8",
        "cp_step_rows must be 0 or at least one session's verify (mtp + 1)",
        "cp_seg needs cp_step_rows",
        "r2_rows must be 0, or a multiple of 64 with window_pages",
        "stale session key",
        "invalid completion",
        "insert count went back",
        "cannot advance a live boundary",
        "session generation exhausted",
        "no parked session on that slot",
        "no session slot",
        "commit requires quiescent live engine",
        "event/state mismatch",
        "cancel epoch mismatch",
        "bad shrink key",
        "bad shrink",
        "native request ordinals must increase",
        "bad grant key",
        "grant slot generation mismatch",
        "bad reservation",
        "duplicate native request id",
        "active session/row limit",
        "reused rows past the prompt",
        "grant at an R the engine does not have",
        "bad run growth",
        "scheduler failed or work in flight",
        "skip_prefill outside admission",
        "not a prefill quantum",
        "not a prefill pack quantum",
        "prefill pack membership changed",
        "not this prompt tail",
        "not this prompt tail pack",
        "tail pack completion invalid",
        "not this decode round",
        "decode completion order differs",
        "too many output tokens",
        "duplicate queued id",
        "FIFO admission violated",
        "invalid allocator view",
    )
}

#[no_mangle]
pub unsafe extern "C" fn g6sch_validate(
    c: *const Config,
    max_sessions: u32,
    mtp0: bool,
    pack: bool,
    park: bool,
) -> *const c_char {
    err(sch::validate(&*c, max_sessions, mtp0, pack, park))
}

macro_rules! out_fn {
    ($name:ident, $f:path, $t:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(c: *const Config, r: *const Request, out: *mut $t) -> *const c_char {
            match $f(&*c, &*r) {
                Ok(v) => {
                    *out = v;
                    std::ptr::null()
                }
                Err(m) => msg(m),
            }
        }
    };
}
out_fn!(g6sch_reserve_rows, sch::reserve_rows, u64);
out_fn!(g6sch_reserve_pages, sch::reserve_pages, u32);
out_fn!(g6sch_grant_rows, sch::grant_rows, u64);
out_fn!(g6sch_grant_pages, sch::grant_pages, u32);

#[no_mangle]
pub unsafe extern "C" fn g6sch_round_rows(c: *const Config, length: u64) -> u64 {
    sch::round_rows(&*c, length)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_grant_pages_ok(
    c: *const Config,
    r: *const Request,
    pages: u32,
    out: *mut bool,
) -> *const c_char {
    match sch::grant_pages_ok(&*c, &*r, pages) {
        Ok(v) => {
            *out = v;
            std::ptr::null()
        }
        Err(m) => msg(m),
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_prefill_left(s: *const Slot) -> u32 {
    sch::prefill_left(&*s)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_prefill_first(a: *const Slot, b: *const Slot) -> bool {
    sch::prefill_first(&*a, &*b)
}

#[no_mangle]
pub unsafe extern "C" fn g6sch_init(c: *const Config, s: *mut Slot, n: usize) {
    sch::init(&*c, slice_mut(s, n))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_idle_boundary(s: *const Slot, n: usize, st: *mut State, target: u64) -> *const c_char {
    err(sch::idle_boundary(slice(s, n), &mut *st, target))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_resident(s: *const Slot, n: usize) -> u32 {
    sch::resident(slice(s, n))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_retiring(s: *const Slot, n: usize) -> u32 {
    sch::retiring(slice(s, n))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_parked(s: *const Slot, n: usize) -> u32 {
    sch::parked(slice(s, n))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_find(s: *const Slot, n: usize, id: u64, out: *mut Key) -> bool {
    match sch::find(slice(s, n), id) {
        Some(k) => {
            *out = k;
            true
        }
        None => false,
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_next_key(
    c: *const Config,
    s: *const Slot,
    n: usize,
    cancel: *const Key,
    has_park: bool,
    park: u32,
    out: *mut Key,
) -> *const c_char {
    let cancel = if cancel.is_null() { None } else { Some(*cancel) };
    match sch::next_key(&*c, slice(s, n), cancel, has_park.then_some(park)) {
        Ok(k) => {
            *out = k;
            std::ptr::null()
        }
        Err(m) => msg(m),
    }
}
/// out holds n runs; the count written is returned.
#[no_mangle]
pub unsafe extern "C" fn g6sch_reclaimable(
    c: *const Config,
    s: *const Slot,
    n: usize,
    cancel: *const Key,
    out: *mut Run,
) -> usize {
    let cancel = if cancel.is_null() { None } else { Some(*cancel) };
    let v = sch::reclaimable(&*c, slice(s, n), cancel);
    slice_mut(out, v.len()).copy_from_slice(&v);
    v.len()
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_digest_head(c: *const Config, st: *const State) -> u64 {
    sch::digest_head(&*c, &*st)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_digest_slots(c: *const Config, s: *const Slot, n: usize, h: u64) -> u64 {
    sch::digest_slots(&*c, slice(s, n), h)
}
/// freed holds n runs; *n_freed and *admitted are written on success.
#[no_mangle]
pub unsafe extern "C" fn g6sch_apply(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
    e: *const Commit,
    digest: u64,
    freed: *mut Run,
    n_freed: *mut usize,
    admitted: *mut bool,
) -> *const c_char {
    match sch::apply(&*c, slice_mut(s, n), &mut *st, &*e, digest) {
        Ok(d) => {
            slice_mut(freed, d.freed.len()).copy_from_slice(&d.freed);
            *n_freed = d.freed.len();
            *admitted = d.admitted;
            std::ptr::null()
        }
        Err(m) => msg(m),
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_grow(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *const State,
    k: *const Key,
    r: *const Run,
) -> *const c_char {
    err(sch::grow(&*c, slice_mut(s, n), &*st, &*k, &*r))
}
/// out holds n keys; the count written is returned.
#[no_mangle]
pub unsafe extern "C" fn g6sch_decoding(s: *const Slot, n: usize, out: *mut Key) -> usize {
    let v = sch::decoding(slice(s, n));
    slice_mut(out, v.len()).copy_from_slice(&v);
    v.len()
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_alternate(c: *const Config, kind: u32, want: *mut bool, run: *mut u32) {
    sch::alternate(&*c, kind, &mut *want, &mut *run)
}
/// stalled holds n flags (null: none stalled).
#[no_mangle]
pub unsafe extern "C" fn g6sch_plan(
    c: *const Config,
    s: *const Slot,
    n: usize,
    want: bool,
    stalled: *const bool,
    out: *mut Quantum,
) {
    *out = sch::plan(&*c, slice(s, n), want, slice(stalled, if stalled.is_null() { 0 } else { n }))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_next(
    c: *const Config,
    s: *const Slot,
    n: usize,
    st: *mut State,
    stalled: *const bool,
    out: *mut Quantum,
) -> *const c_char {
    match sch::next(&*c, slice(s, n), &mut *st, slice(stalled, if stalled.is_null() { 0 } else { n })) {
        Ok(q) => {
            *out = q;
            std::ptr::null()
        }
        Err(m) => msg(m),
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_advance(s: *mut Slot, n: usize, q: *const Quantum) {
    sch::advance(slice_mut(s, n), &*q)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_horizon(c: *const Config, s: *const Slot, n: usize, st: *const State) -> u64 {
    sch::horizon(&*c, slice(s, n), &*st)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_skip_prefill(s: *mut Slot, n: usize, st: *const State, k: *const Key) -> *const c_char {
    err(sch::skip_prefill(slice_mut(s, n), &*st, &*k))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_complete_prefill(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
) -> *const c_char {
    err(sch::complete_prefill(&*c, slice_mut(s, n), &mut *st))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_complete_prefill_pack(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
) -> *const c_char {
    err(sch::complete_prefill_pack(&*c, slice_mut(s, n), &mut *st))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_complete_tail(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
    r: *const Completion,
) -> *const c_char {
    err(sch::complete_tail(&*c, slice_mut(s, n), &mut *st, &*r))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_complete_tail_pack(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
    r: *const Completion,
    nr: usize,
) -> *const c_char {
    err(sch::complete_tail_pack(&*c, slice_mut(s, n), &mut *st, slice(r, nr)))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_complete_decode(
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
    r: *const Completion,
    nr: usize,
) -> *const c_char {
    err(sch::complete_decode(&*c, slice_mut(s, n), &mut *st, slice(r, nr)))
}

/// The struct sizes and constants the C++ side duplicates, by index (g6sched.h G6SCH_ABI_*); 0 past the end.
#[no_mangle]
pub extern "C" fn g6sch_abi(what: u32) -> u64 {
    use std::mem::size_of;
    let v = [
        size_of::<Slot>(),
        size_of::<Config>(),
        size_of::<Quantum>(),
        size_of::<State>(),
        size_of::<Commit>(),
        size_of::<Completion>(),
        size_of::<sch::Grant>(),
        sch::PAGE as usize,
        sch::PREFILL_DEFER_MAX as usize,
        sch::NOTICE_MIN as usize,
        sch::NOTICE_MAX as usize,
        sch::NOTICE_ROWS as usize,
        sch::MAX_DECODE_ROUNDS as usize,
        sch::PACK_MAX_MEMBERS,
        sch::QMAX,
        size_of::<crate::kv_share::Totals>(),
        size_of::<crate::kv_share::Resize>(),
    ];
    v.get(what as usize).map_or(0, |&x| x as u64)
}

// ---- kv_share.rs ---------------------------------------------------------------------------------------------------

use crate::kv_share::{self as share, Resize, Totals};

#[no_mangle]
pub unsafe extern "C" fn g6sch_share_want(last: *const Totals, now: *const Totals, per: u64, busy: bool) -> u64 {
    share::want(&*last, &*now, per, busy)
}
/// true and *out written when the answer is a resize.
#[no_mangle]
pub unsafe extern "C" fn g6sch_share_resize(
    tier: u32,
    segs: u64,
    per: u64,
    grant: u64,
    give: u64,
    txn: u64,
    out: *mut Resize,
) -> bool {
    match share::resize(tier, segs, per, grant, give, txn) {
        Some(r) => {
            *out = r;
            true
        }
        None => false,
    }
}

// ---- admission.rs: the queue is Rust-owned, behind an opaque pointer ---------------------------------------------

use crate::admission::{Queue, Status};

/// A new empty queue for the (validated) config; free it with g6sch_queue_free.
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_new(c: *const Config) -> *mut Queue {
    Box::into_raw(Box::new(Queue::new(*c)))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_free(q: *mut Queue) {
    if !q.is_null() {
        drop(Box::from_raw(q));
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_push(q: *mut Queue, r: *const Request) -> *const c_char {
    err((*q).push(*r))
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_cancel(q: *mut Queue, id: u64) -> bool {
    (*q).cancel(id)
}
/// The front request, or null when the queue is empty; valid until the next call that changes the queue.
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_front(q: *const Queue) -> *const Request {
    (*q).front().map_or(std::ptr::null(), |r| r as *const Request)
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_size(q: *const Queue) -> usize {
    (*q).len()
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_stalls(q: *const Queue) -> u64 {
    (*q).stalls
}
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_admitted(q: *mut Queue, id: u64) -> *const c_char {
    err((*q).admitted(id))
}
/// *has: whether *out holds a status.
#[no_mangle]
pub unsafe extern "C" fn g6sch_queue_check(
    q: *mut Queue,
    revision: u64,
    slot: bool,
    fits: bool,
    room: u32,
    has: *mut bool,
    out: *mut Status,
) -> *const c_char {
    match (*q).check(revision, slot, fits, room) {
        Ok(s) => {
            *has = s.is_some();
            if let Some(s) = s {
                *out = s;
            }
            std::ptr::null()
        }
        Err(m) => msg(m),
    }
}
