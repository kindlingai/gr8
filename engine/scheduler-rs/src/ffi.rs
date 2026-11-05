//! The C entry points (g6sched.h). Each fallible call returns 0 or a code the C++ wrapper turns into its message; a
//! failed call leaves its state unchanged. Release builds abort on a panic, so none unwinds into C++.

use crate::cp_rows::{self, Cand};
use crate::flash_lend::Lend;
use crate::flash_ring::{self, Ring};
use crate::kv_plan::{self, Io, Item};

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 || p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

unsafe fn slice_mut<'a, T>(p: *mut T, n: usize) -> &'a mut [T] {
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
    match flash_ring::rows(&mut *r, per, q, pos, n, verify, slice_mut(kslot, n.max(0) as usize)) {
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
