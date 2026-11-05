//! The head's side of a pool tier on a shared KV node's group lease (was the logic in engine/runtime/s2/kv_share.h):
//! how many segments a report asks for, and the Resize a node's answer becomes. kvman-rs REMOTE-TIER.md "Sharing a
//! node" describes the lease.

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
}
