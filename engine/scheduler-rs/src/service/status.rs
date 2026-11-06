//! status.s2: the head's view of its queue and pool, one JSON object, published on queue and lifecycle revisions.

use super::files::atomic_file;
use crate::control::{require, R};
use crate::kv_share::Totals;
use std::fmt::Write;

/// One live run's pages now and at worst (kvman-rs g6kv_use, the same layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunUse {
    pub table_version: u32,
    pub held: u32,
    pub worst: u32,
}

/// The pool's growth counters (kvman-rs g6kv_totals).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    pub held_pages: u32,
    pub worst_pages: u32,
    pub grows: u64,
    pub stalls: u64,
}

/// What status.s2 reports.
#[derive(Clone, Copy, Debug, Default)]
pub struct Status<'a> {
    pub epoch: u64,
    pub pool_pages: u32,
    pub resident: u32,
    pub queued: u64,
    pub retiring: u32,
    pub free: u32,
    pub room: u32,
    pub stalls: u64,
    pub growth: Option<Growth>,
    pub tiers: Option<&'a [Totals]>,
    pub runs: Option<&'a [RunUse]>,
}

/// The pool's free pages and largest free run must fit the pool.
pub fn pool_ok(free: u32, room: u32, pool_pages: u32) -> R<()> {
    require(room <= free && free <= pool_pages, "invalid pool status")
}

/// The JSON object, with its newline.
pub fn json(s: &Status) -> String {
    let mut o = String::with_capacity(512);
    // writes into a String do not fail
    let _ = write!(
        o,
        "{{\"epoch\":{},\"resident\":{},\"queued\":{},\"retiring\":{},\"reserved_pages\":{},\"free_pages\":{},\
         \"largest_free_run\":{},\"admission_stalls\":{}",
        s.epoch,
        s.resident,
        s.queued,
        s.retiring,
        s.pool_pages.wrapping_sub(s.free),
        s.free,
        s.room,
        s.stalls
    );
    if let Some(g) = s.growth {
        let _ = write!(
            o,
            ",\"held_pages\":{},\"worst_pages\":{},\"grown\":{},\"stalled_rounds\":{}",
            g.held_pages, g.worst_pages, g.grows, g.stalls
        );
    }
    if let Some(ts) = s.tiers {
        o.push_str(",\"tiers\":[");
        for (i, t) in ts.iter().enumerate() {
            // pages are page blocks; *_bytes = pages x block_bytes
            let bb = t.block_bytes;
            let _ = write!(
                o,
                "{}{{\"slots\":{},\"free_slots\":{},\"entries\":{},\"shared\":{},\"block_bytes\":{},\"spilled_pages\":{},\
                 \"spilled_bytes\":{},\"moved_in_pages\":{},\"moved_in_bytes\":{},\"moved_out\":{},\"moved_out_pages\":{},\
                 \"restored_pages\":{},\"restored_bytes\":{},\"hits\":{},\"dropped\":{},\"dropped_pages\":{},\"refused\":{},\
                 \"refused_pages\":{},\"full\":{},\"write_credit\":{}}}",
                if i > 0 { "," } else { "" },
                t.slots,
                t.free_slots,
                t.entries,
                t.shared,
                bb,
                t.spilled_pages,
                t.spilled_pages.wrapping_mul(bb),
                t.moved_in_pages,
                t.moved_in_pages.wrapping_mul(bb),
                t.moved_out,
                t.moved_out_pages,
                t.restored_pages,
                t.restored_pages.wrapping_mul(bb),
                t.hits,
                t.dropped,
                t.dropped_pages,
                t.refused,
                t.refused_pages,
                t.full,
                t.credit
            );
        }
        o.push(']');
    }
    if let Some(rs) = s.runs {
        o.push_str(",\"runs\":[");
        for (i, r) in rs.iter().enumerate() {
            let _ = write!(o, "{}{{\"held\":{},\"worst\":{}}}", if i > 0 { "," } else { "" }, r.held, r.worst);
        }
        o.push(']');
    }
    o.push_str("}\n");
    o
}

/// Writes `dir`/status.s2.
pub fn publish(dir: &[u8], s: &Status) -> R<()> {
    let mut p = dir.to_vec();
    p.extend_from_slice(b"/status.s2");
    atomic_file(&p, json(s).as_bytes())
}
