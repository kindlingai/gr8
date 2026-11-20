//! A pool tier on a shared KV node's group lease (was the logic in engine/runtime/s2/kv_share.h): how many segments
//! the head's report asks for, the Resize a node's answer becomes, and every rank's change at the Resize commit
//! (`apply`). kvman-rs REMOTE-TIER.md "Sharing a node" describes the lease.
#![allow(clippy::missing_safety_doc)]

use g6kvman::blocks::{self, Blocks};
use g6kvman::ffi::Handle;
use std::ffi::{c_char, c_void};

/// A tier's counters (kvman-rs g6kv_tier_totals, the same layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub slots: u64,
    pub free_slots: u64,
    pub entries: u32,
    pub shared: u32,
    pub spilled_pages: u64,
    pub moved_in_pages: u64,
    pub restored_pages: u64,
    pub hits: u64,
    pub dropped: u64,
    pub dropped_pages: u64,
    pub moved_out: u64,
    pub moved_out_pages: u64,
    pub refused: u64,
    pub refused_pages: u64,
    pub full: u64,
    pub block_bytes: u64,
    pub credit: i64,
    pub oversize: u64,
    pub oversize_last: u64,
}

/// Segments more a rank the head asks for, from the tier's counters at the last report and now (`per` = blocks a
/// segment). It asks for room for an oversize entry offered since, the pages a filled tier pushed out, or, after a
/// spill (or while `busy`), room for one more average entry when fewer blocks than that are free. At least one segment
/// when any of these holds, and 0 for an idle tier that took no spill.
pub fn want(last: &Totals, now: &Totals, per: u64, busy: bool) -> u64 {
    if per == 0 {
        return 0;
    }
    // (blocks + per - 1) / per wraps as the C++ did
    let segs = |blocks: u64| (blocks.wrapping_add(per - 1) / per).max(1);
    let mut want = 0;
    if now.oversize > last.oversize && now.oversize_last > now.free_slots {
        want = want.max(segs(now.oversize_last - now.free_slots));
    }
    if now.full > last.full {
        let pushed = now
            .dropped_pages
            .wrapping_sub(last.dropped_pages)
            .wrapping_add(now.moved_out_pages.wrapping_sub(last.moved_out_pages));
        want = want.max(segs(pushed));
    } else if (now.spilled_pages > last.spilled_pages || busy) && now.entries > 0 {
        let avg = now.slots.wrapping_sub(now.free_slots) / u64::from(now.entries);
        if now.free_slots < avg {
            want = want.max(segs(avg - now.free_slots));
        }
    }
    want
}

/// A Resize event (scheduler.h Resize, the same layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resize {
    pub tier: u32,
    pub segs: u64,
    pub slots: u64,
    pub txn: u64,
}

/// The Resize a node's report answer becomes: a grant grows the lease, a give smaller than the lease shrinks it.
/// `segs`: the lease's segments now, `per`: blocks a segment.
pub fn resize(tier: u32, segs: u64, per: u64, grant: u64, give: u64, txn: u64) -> Option<Resize> {
    let to = if grant != 0 {
        segs.wrapping_add(grant)
    } else if give != 0 && give < segs {
        segs - give
    } else {
        return None;
    };
    Some(Resize { tier, segs: to, slots: to.wrapping_mul(per), txn })
}

/// A store call's result: the store's blocks after it, the node's refusal (every rank alike; the store keeps its
/// size), or a fault (the store's message in g6kv_tier_error).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    Blocks(u64),
    Refused,
    Fault,
}

/// The group lease below a tier: its segments now, and growing or shrinking it under a transaction.
pub trait Lease {
    /// segments now
    fn segs(&self) -> u64;
    /// grows to `segs` segments
    fn grow(&self, segs: u64, txn: u64) -> Store;
    /// gives back all but the first `segs` segments
    fn shrink(&self, segs: u64, txn: u64) -> Store;
}

fn store(rc: i32, blocks: u64) -> Store {
    match rc {
        0 => Store::Blocks(blocks),
        blocks::STORE_REFUSED => Store::Refused,
        _ => Store::Fault,
    }
}

impl Lease for Blocks {
    fn segs(&self) -> u64 {
        // SAFETY: a live store
        unsafe { blocks::g6kv_store_segs(self) }
    }
    fn grow(&self, segs: u64, txn: u64) -> Store {
        let mut b = 0;
        // SAFETY: as segs; b is valid for a write
        store(unsafe { blocks::g6kv_store_grow(self, segs, txn, &mut b) }, b)
    }
    fn shrink(&self, segs: u64, txn: u64) -> Store {
        let mut b = 0;
        // SAFETY: as grow
        store(unsafe { blocks::g6kv_store_shrink(self, segs, txn, &mut b) }, b)
    }
}

/// Why apply stopped short of a change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fail {
    /// the store faulted (its message in g6kv_tier_error)
    Store,
    /// a pool error or a size the event did not name
    Msg(&'static str),
    /// run_io failed (the caller holds why)
    Io,
}

/// Every rank, at the Resize commit, with no copies pending. Grow: the lease first, then the tier, then the copies the
/// pool had waiting for room. Shrink: the tier first (it evicts and queues moves into the slots that stay), the copies,
/// then the lease gives its top segments back. `run_io` runs the pool's queued copies in order and calls io_done; it
/// runs with the pool unlocked, since it calls the pool. Ok(false): the lease or tier keeps its size, on every rank
/// alike: the tier refused a shrink (an indexed tier, or entries it may not evict), or the node refused the transaction
/// (a grant no rank took in time). A shrink the node refuses after the tier shrank leaves the lease larger than the
/// tier, which is safe (the tier never addresses past its slots).
pub fn apply(pool: &Handle, lease: &impl Lease, r: &Resize, mut run_io: impl FnMut() -> bool) -> Result<bool, Fail> {
    let resize = |slots: u64| {
        pool.lock()
            .resize_tier(r.tier as usize, slots)
            .map_err(|e| Fail::Msg(e.message().to_str().unwrap_or("pool error")))
    };
    let segs = lease.segs();
    if r.segs > segs {
        let blocks = match lease.grow(r.segs, r.txn) {
            Store::Blocks(b) => b,
            Store::Refused => return Ok(false),
            Store::Fault => return Err(Fail::Store),
        };
        if blocks != r.slots {
            return Err(Fail::Msg("share_apply: the lease grew to a different size than the event's"));
        }
        if !resize(blocks)? {
            return Err(Fail::Msg("share_apply: the tier refused to grow"));
        }
        if !run_io() {
            return Err(Fail::Io);
        }
    } else if r.segs < segs {
        if !resize(r.slots)? {
            return Ok(false);
        }
        if !run_io() {
            return Err(Fail::Io);
        }
        match lease.shrink(r.segs, r.txn) {
            Store::Blocks(b) if b != r.slots => {
                return Err(Fail::Msg("share_apply: the lease shrank to a different size than the event's"))
            }
            Store::Blocks(_) => {}
            Store::Refused => return Ok(false),
            Store::Fault => return Err(Fail::Store),
        }
    }
    Ok(true)
}

/// run_io(ctx): 0, or nonzero when it failed
pub type RunIoFn = unsafe extern "C" fn(*mut c_void) -> i32;

/// apply over a pool and a store (kv_share.h share_apply): 0 applied, 1 the lease or tier keeps its size, 2 a store
/// fault (g6kv_tier_error), 3 a pool error or a size the event did not name (*msg), 4 run_io failed.
#[no_mangle]
pub unsafe extern "C" fn g6s2_share_apply(
    pool: *const Handle,
    lease: *const Blocks,
    r: *const Resize,
    run_io: RunIoFn,
    ctx: *mut c_void,
    msg: *mut *const c_char,
) -> i32 {
    match apply(&*pool, &*lease, &*r, || run_io(ctx) == 0) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(Fail::Store) => 2,
        Err(Fail::Msg(m)) => {
            *msg = crate::ffi::msg(m);
            3
        }
        Err(Fail::Io) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asks_for_what_pushed_it() {
        let last = Totals { slots: 100, free_slots: 50, entries: 5, ..Default::default() };
        assert_eq!(want(&last, &last, 8, false), 0);
        let filled = Totals { full: 1, dropped_pages: 20, moved_out_pages: 5, ..last };
        assert_eq!(want(&last, &filled, 8, false), 4);
        let tight = Totals { free_slots: 3, spilled_pages: 1, ..last };
        assert_eq!(want(&last, &tight, 8, false), 2); // avg 19 a entry, 3 free
        assert_eq!(want(&last, &Totals { spilled_pages: 0, ..tight }, 8, true), 2);
        assert_eq!(want(&last, &filled, 0, false), 0);
    }

    #[test]
    fn grant_and_give() {
        assert_eq!(resize(1, 4, 8, 2, 0, 9), Some(Resize { tier: 1, segs: 6, slots: 48, txn: 9 }));
        assert_eq!(resize(1, 4, 8, 0, 1, 9), Some(Resize { tier: 1, segs: 3, slots: 24, txn: 9 }));
        assert_eq!(resize(1, 4, 8, 0, 4, 9), None);
        assert_eq!(resize(1, 4, 8, 0, 0, 9), None);
    }

    // a lease that records its calls, with the answers a test sets
    struct Fake {
        segs: u64,
        per: u64,
        answer: Option<Store>,
        calls: std::cell::RefCell<Vec<String>>,
    }
    impl Lease for Fake {
        fn segs(&self) -> u64 {
            self.segs
        }
        fn grow(&self, segs: u64, txn: u64) -> Store {
            self.calls.borrow_mut().push(format!("grow {segs} {txn}"));
            self.answer.unwrap_or(Store::Blocks(segs * self.per))
        }
        fn shrink(&self, segs: u64, txn: u64) -> Store {
            self.calls.borrow_mut().push(format!("shrink {segs} {txn}"));
            self.answer.unwrap_or(Store::Blocks(segs * self.per))
        }
    }

    fn pool(slots: u64) -> Handle {
        let mut p = g6kvman::Pool::new(64, 64, 64 * 64).unwrap();
        p.keep(g6kvman::Rules { strict: false, noend: true, grid: 128 });
        p.stores(2);
        p.add_tier(slots, g6kvman::stash::Keep::All, None).unwrap();
        Handle::new(p)
    }
    fn slots(p: &Handle) -> u64 {
        p.lock().tier_stats(0).unwrap().slots
    }

    #[test]
    fn grows_and_shrinks_with_the_pool_unlocked_for_the_copies() {
        let p = pool(16);
        let f = Fake { segs: 2, per: 8, answer: None, calls: Default::default() };
        let mut io = 0;
        // run_io takes the pool itself: apply must not hold the lock around it
        let mut run_io = || {
            io += 1;
            p.lock().stats();
            f.calls.borrow_mut().push("io".into());
            true
        };
        let grow = Resize { tier: 0, segs: 4, slots: 32, txn: 7 };
        assert_eq!(apply(&p, &f, &grow, &mut run_io), Ok(true));
        assert_eq!(slots(&p), 32);
        let shrink = Resize { tier: 0, segs: 1, slots: 8, txn: 8 };
        assert_eq!(apply(&p, &f, &shrink, &mut run_io), Ok(true));
        assert_eq!(slots(&p), 8);
        assert_eq!(apply(&p, &f, &Resize { segs: 2, ..shrink }, &mut run_io), Ok(true)); // the size it has: nothing
        assert_eq!(io, 2);
        assert_eq!(*f.calls.borrow(), ["grow 4 7", "io", "io", "shrink 1 8"]);
    }

    #[test]
    fn refusals_and_faults() {
        let p = pool(16);
        let fake = |answer| Fake { segs: 2, per: 8, answer, calls: Default::default() };
        let grow = Resize { tier: 0, segs: 4, slots: 32, txn: 7 };
        let shrink = Resize { tier: 0, segs: 1, slots: 8, txn: 8 };
        let yes = || true;
        // the node refuses or faults before anything changed
        assert_eq!(apply(&p, &fake(Some(Store::Refused)), &grow, yes), Ok(false));
        assert_eq!(apply(&p, &fake(Some(Store::Fault)), &grow, yes), Err(Fail::Store));
        assert_eq!(slots(&p), 16);
        let odd = fake(Some(Store::Blocks(24)));
        let e = Err(Fail::Msg("share_apply: the lease grew to a different size than the event's"));
        assert_eq!(apply(&p, &odd, &grow, yes), e);
        // a shrink the node refuses after the tier shrank leaves the tier smaller
        assert_eq!(apply(&p, &fake(Some(Store::Refused)), &shrink, yes), Ok(false));
        assert_eq!(slots(&p), 8);
        let e = Err(Fail::Msg("share_apply: the lease shrank to a different size than the event's"));
        assert_eq!(apply(&p, &fake(Some(Store::Blocks(16))), &Resize { slots: 4, ..shrink }, yes), e);
        // copies that fail stop it; a tier the pool does not have is its error
        assert_eq!(apply(&p, &fake(None), &grow, || false), Err(Fail::Io));
        let e = Err(Fail::Msg("pool call with a null or invalid argument"));
        assert_eq!(apply(&p, &fake(None), &Resize { tier: 3, ..grow }, yes), e);
    }
}
