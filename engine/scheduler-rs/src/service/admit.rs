//! The head's admission decisions as functions of their inputs (was s2/admit.h, which is now a facade over these).
//! The control thread (`control`) calls them where service.h had the expressions. tests/admit_vectors_test.cpp checks
//! them against transcripts recorded from the C++.

use crate::admission::{STALL_CAPACITY, STALL_NONE};
use crate::scheduler::NOTICE_MIN;

/// The boundary a notice names. A held boundary names itself: every rank is stopped there with no work in flight.
/// Otherwise the horizon (the first boundary whose work after the one in flight covers NOTICE_ROWS rows), clamped to
/// boundary + NOTICE_MIN .. boundary + notice_max, so a torn read of the pair can only widen the target.
pub fn notice_target(boundary: u64, horizon: u64, held: Option<u64>, notice_max: u32) -> u64 {
    // std::clamp's order: below lo gives lo, else above hi gives hi (lo > hi cannot happen: notice_max >= NOTICE_MIN)
    held.unwrap_or_else(|| {
        let (lo, hi) = (boundary.wrapping_add(NOTICE_MIN), boundary.wrapping_add(u64::from(notice_max)));
        if horizon < lo {
            lo
        } else if hi < horizon {
            hi
        } else {
            horizon
        }
    })
}

/// A stalled head waits for a new revision rather than being checked again on every 1 ms pass: the queue's check
/// refuses a repeat at one revision.
pub fn retry(head: u64, attempted_head: u64, revision: u64, attempted_revision: u64) -> bool {
    head != attempted_head || revision != attempted_revision
}

/// The revision an attempt expects next: the commit that carries it moves the scheduler one revision on.
pub fn attempted_revision(revision: u64) -> u64 {
    revision.wrapping_add(1)
}

pub fn shrink(fits: bool, can_make_room: bool, live: u32, limit: u32) -> bool {
    !fits && can_make_room && live < limit
}

/// short_of: the chunks a recall of lent memory must bring back first (borrow-idle).
pub fn grant(why: u32, short_of: u32) -> bool {
    why == STALL_NONE && short_of == 0
}

/// GR8_S2_ADMIT_HOLD after a grant that leaves a slot: queue the ADMITs already written, then hold the boundary when a
/// request is queued behind the grant.
pub fn drain(admit_hold: bool, live: u32, limit: u32) -> bool {
    admit_hold && live.wrapping_add(1) < limit
}

pub fn hold(admit_hold: bool, queued: bool, live: u32, limit: u32) -> bool {
    drain(admit_hold, live, limit) && queued
}

/// The once-per-request lines: `last` is the request the line last named.
pub fn wait_line(why: u32, last: u64, head: u64) -> bool {
    why != STALL_NONE && last != head
}

pub fn refusal_line(why: u32, last: u64, head: u64) -> bool {
    why == STALL_CAPACITY && last != head
}

/// Borrow-idle's lines: Begin when the head starts waiting for a recall, Done when the head that waited is granted.
pub const RECALL_NONE: u32 = 0;
pub const RECALL_BEGIN: u32 = 1;
pub const RECALL_DONE: u32 = 2;

pub fn recall_note(why: u32, short_of: u32, waiting: u64, head: u64) -> u32 {
    if short_of != 0 {
        return if waiting != head { RECALL_BEGIN } else { RECALL_NONE };
    }
    if why == STALL_NONE && waiting == head {
        RECALL_DONE
    } else {
        RECALL_NONE
    }
}

/// The idle tests of the control loop's resizes. The KV share rebalances once nothing is resident or queued, and
/// borrow-idle lends chunks only when no session is retiring either.
pub fn share_idle(resident: u32, queued: usize) -> bool {
    resident == 0 && queued == 0
}

pub fn lend_idle(resident: u32, queued: usize, retiring: u32) -> bool {
    share_idle(resident, queued) && retiring == 0
}

/// The busy guard's grant decision (GR8_BUSY_GUARD=1, docs/ENV.md): a fresh grant whose eviction set (the pool's
/// busy_view) holds a live retained run — its session has no request in flight but its last response ended under the
/// idle bound ago — refuses the request with HTTP 429 + Retry-After instead of evicting the run. `hit`: the prompt
/// continues its own retained run, which the guard never refuses (the adoption is not an eviction).
pub fn busy_refuse(guard_on: bool, hit: bool, live_victims: u32) -> bool {
    guard_on && !hit && live_victims > 0
}

/// The busy guard's arrival decision: a request that would join a queue already `queued` deep is refused at once
/// (it would be left queued past the bound) instead of added. 0 = no bound.
pub fn busy_queue_full(guard_on: bool, bound: u32, queued: usize) -> bool {
    guard_on && bound > 0 && queued >= bound as usize
}

/// The busy refusal's once-per-request trace line, like wait_line.
pub fn busy_line(last: u64, head: u64) -> bool {
    last != head
}

/// The busy guard's intake decision (docs/ENV.md GR8_BUSY_*): with the guard on, a fresh arrival whose
/// worst-case grant would evict a live retained run — the pool's busy_view at arrival — is refused at intake, before
/// it queues, instead of being refused at grant time after its queue wait. The same rule as the grant path
/// (busy_refuse): a hit continuation is never refused, and no live victim never refuses; the grant path keeps the
/// check as the backstop, because the pool can change between intake and grant.
pub fn busy_intake_refuse(guard_on: bool, hit: bool, live_victims: u32) -> bool {
    busy_refuse(guard_on, hit, live_victims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::STALL_SLOTS;

    #[test]
    fn notice_clamps_and_holds() {
        assert_eq!(notice_target(10, 0, None, 8), 12);
        assert_eq!(notice_target(10, 15, None, 8), 15);
        assert_eq!(notice_target(10, 99, None, 8), 18);
        assert_eq!(notice_target(10, 99, Some(10), 8), 10);
    }

    #[test]
    fn busy_decisions() {
        // the guard off is upstream behaviour: never refuse
        assert!(!busy_refuse(false, false, 3) && !busy_refuse(false, true, 3));
        // on: a live victim refuses a fresh grant, never a hit continuation, never an all-idle eviction set
        assert!(busy_refuse(true, false, 1) && busy_refuse(true, false, 4));
        assert!(!busy_refuse(true, true, 2));
        assert!(!busy_refuse(true, false, 0));
        // the queue bound: full at the bound, off at 0, off with the guard off
        assert!(busy_queue_full(true, 8, 8) && busy_queue_full(true, 8, 9));
        assert!(!busy_queue_full(true, 8, 7) && !busy_queue_full(true, 0, 99) && !busy_queue_full(false, 8, 99));
        assert!(busy_line(1, 2) && !busy_line(2, 2));
    }

    #[test]
    fn busy_intake_decides_like_the_grant() {
        // the intake refusal is the grant rule applied at arrival: a fresh arrival with a live victim refuses, a hit
        // continuation and an all-idle eviction set never do, and the guard off is upstream behaviour
        assert!(busy_intake_refuse(true, false, 1) && busy_intake_refuse(true, false, 4));
        assert!(!busy_intake_refuse(true, true, 2) && !busy_intake_refuse(true, false, 0));
        assert!(!busy_intake_refuse(false, false, 3) && !busy_intake_refuse(false, true, 3));
        // and it stays the same decision the grant path makes on the same inputs
        for &(on, hit, v) in &[(true, false, 1), (true, false, 0), (true, true, 2), (false, false, 3)] {
            assert_eq!(busy_intake_refuse(on, hit, v), busy_refuse(on, hit, v), "intake and grant decide alike");
        }
    }

    #[test]
    fn gates() {
        assert!(!retry(3, 3, 7, 7) && retry(4, 3, 7, 7) && retry(3, 3, 8, 7));
        assert!(shrink(false, true, 1, 2) && !shrink(true, true, 1, 2) && !shrink(false, true, 2, 2));
        assert!(grant(STALL_NONE, 0) && !grant(STALL_NONE, 1) && !grant(STALL_SLOTS, 0));
        assert!(hold(true, true, 0, 2) && !hold(true, true, 1, 2) && !hold(false, true, 0, 2));
        assert!(wait_line(STALL_SLOTS, 1, 2) && !wait_line(STALL_SLOTS, 2, 2));
        assert!(refusal_line(STALL_CAPACITY, 1, 2) && !refusal_line(STALL_SLOTS, 1, 2));
        assert_eq!(recall_note(STALL_NONE, 3, 1, 2), RECALL_BEGIN);
        assert_eq!(recall_note(STALL_NONE, 3, 2, 2), RECALL_NONE);
        assert_eq!(recall_note(STALL_NONE, 0, 2, 2), RECALL_DONE);
        assert!(share_idle(0, 0) && !lend_idle(0, 0, 1));
    }
}
