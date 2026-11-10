//! The event boundary gate: the compute thread asks `before_work(b)` before it departs boundary b, and stops there
//! when an event is due. The control thread installs a notice for a future boundary (`announce`); the stopped rank
//! acknowledges it (`arrive`), the coordinator commits once every rank has (`commit`), and the rank applies the commit
//! (`applied`) before it moves on. While a rank waits here (`wait_commit`) its compute thread is paused, which is what
//! makes the head's multi-call pool sequences in the commit safe. The no-event path is atomics only: no lock, syscall
//! or allocation.

use super::wire::{Ack, Notice, ADMIT, RESIZE};
use super::{refused, require, R};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::SeqCst};
use std::sync::{Condvar, Mutex, MutexGuard};

struct State {
    n: Option<Notice>,
    next_event: u64,
    arrived: bool,
    committed: bool,
    // applied with hold: stopped at `due`, and the next notice must name it
    held: bool,
}

/// One rank's gate.
pub struct Gate {
    epoch: u64,
    rank: u32,
    passed: AtomicI64,
    due: AtomicU64,
    failed: AtomicBool,
    mu: Mutex<State>,
    cv: Condvar,
}

/// A mutex's guard whatever a panicking holder left (release builds abort on panic).
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Gate {
    /// A gate for `epoch` (nonzero) on `rank`.
    pub fn new(epoch: u64, rank: u32) -> R<Gate> {
        require(epoch != 0, "bad epoch")?;
        Ok(Gate {
            epoch,
            rank,
            passed: AtomicI64::new(-1),
            due: AtomicU64::new(u64::MAX),
            failed: AtomicBool::new(false),
            mu: Mutex::new(State { n: None, next_event: 1, arrived: false, committed: false, held: false }),
            cv: Condvar::new(),
        })
    }
    /// true: an event is due at `boundary`, which is not departed. false: departed.
    pub fn before_work(&self, boundary: u64) -> R<bool> {
        require(boundary <= i64::MAX as u64, "boundary overflow")?;
        require(!self.failed.load(SeqCst), "control epoch failed")?;
        require(boundary as i64 > self.passed.load(SeqCst), "boundary already departed or regressed")?;
        let due = self.due.load(SeqCst);
        require(due >= boundary, "missed event boundary")?;
        // a stopped boundary is not departed, so a second event may come at the same quiescent boundary
        if due == boundary {
            return Ok(true);
        }
        self.passed.store(boundary as i64, SeqCst);
        // pairs with announce's recheck: in the narrow race with a notice being installed, fail the epoch and do not
        // run past it
        if self.due.load(SeqCst) <= boundary {
            self.fail();
            return Err(refused("departure raced event notice"));
        }
        Ok(false)
    }
    /// The control thread installs the next event's notice.
    pub fn announce(&self, n: &Notice) -> R<()> {
        let mut s = lock(&self.mu);
        require(!self.failed.load(SeqCst) && s.n.is_none(), "pending event or failed epoch")?;
        require(
            n.epoch == self.epoch && n.event == s.next_event && (ADMIT..=RESIZE).contains(&n.kind),
            "notice epoch/sequence/kind mismatch",
        )?;
        require(n.boundary <= i64::MAX as u64 && n.boundary as i64 > self.passed.load(SeqCst), "late event notice")?;
        require(!s.held || n.boundary == self.due.load(SeqCst), "notice does not name the held boundary")?;
        s.n = Some(*n);
        s.arrived = false;
        s.committed = false;
        s.held = false;
        self.due.store(n.boundary, SeqCst);
        // compute departed while the notice went in: fail the epoch, since the event would apply a round late
        if self.passed.load(SeqCst) >= n.boundary as i64 {
            self.failed.store(true, SeqCst);
            self.cv.notify_all();
            return Err(refused("notice raced departed boundary"));
        }
        Ok(())
    }
    /// This rank stopped at the event's boundary: its ACK.
    pub fn arrive(&self, boundary: u64, state_hash: u64, room: u32) -> R<Ack> {
        let mut s = lock(&self.mu);
        let ok = !self.failed.load(SeqCst)
            && s.n.is_some_and(|n| n.boundary == boundary)
            && !s.arrived
            && self.due.load(SeqCst) == boundary;
        require(ok, "invalid event arrival")?;
        s.arrived = true;
        Ok(Ack { notice: s.n.unwrap_or_default(), rank: self.rank, state_hash, room })
    }
    /// The coordinator's commit, once every rank has acknowledged.
    pub fn commit(&self, n: &Notice) -> R<()> {
        let mut s = lock(&self.mu);
        let ok = !self.failed.load(SeqCst) && s.n == Some(*n) && s.arrived && !s.committed;
        require(ok, "commit before event ACK or mismatched notice")?;
        s.committed = true;
        self.cv.notify_all();
        Ok(())
    }
    /// At an installed event's boundary: wait for its commit.
    pub fn wait_commit(&self) -> R<()> {
        let s = lock(&self.mu);
        require(s.n.is_some() && s.arrived, "wait without event arrival")?;
        let s =
            self.cv.wait_while(s, |s| !(s.committed || self.failed.load(SeqCst))).unwrap_or_else(|e| e.into_inner());
        drop(s);
        require(!self.failed.load(SeqCst), "control epoch failed")
    }
    /// The commit is applied. hold: stay stopped at this boundary for the next event, whose notice must name it. The
    /// compute thread's next before_work(boundary) returns true.
    pub fn applied(&self, n: &Notice, hold: bool) -> R<()> {
        let mut s = lock(&self.mu);
        require(s.n == Some(*n) && s.committed && !self.failed.load(SeqCst), "invalid event application")?;
        require(s.next_event != u64::MAX, "event sequence exhausted")?;
        s.next_event += 1;
        s.n = None;
        s.arrived = false;
        s.committed = false;
        s.held = hold;
        self.due.store(if hold { n.boundary } else { u64::MAX }, SeqCst);
        Ok(())
    }
    /// Fails the epoch and wakes the waiter.
    pub fn fail(&self) {
        // under the waiter's mutex: an atomic alone can lose a wakeup between its predicate check and the wait
        {
            let _s = lock(&self.mu);
            self.failed.store(true, SeqCst);
        }
        self.cv.notify_all();
    }
}

/// The head's per-event ACK set, used only at events.
pub struct Agreement {
    nr: u32,
    n: Notice,
    seen: [bool; 6],
    hash: Option<u64>,
    room: u32,
}

impl Agreement {
    /// The ACKs of `nr` ranks (1 to 6) for `n`.
    pub fn new(nr: u32, n: Notice) -> R<Agreement> {
        require((1..=6).contains(&nr), "expected TP1 to TP6 ranks")?;
        Ok(Agreement { nr, n, seen: [false; 6], hash: None, room: u32::MAX })
    }
    /// One rank's ACK: its event, once, with the same state digest as the others.
    pub fn ack(&mut self, a: &Ack) -> R<()> {
        require(a.rank < self.nr && a.notice == self.n, "ACK rank/event mismatch")?;
        require(!self.seen[a.rank as usize], "duplicate event ACK")?;
        require(self.hash.is_none_or(|h| h == a.state_hash), "replicated state mismatch")?;
        self.hash = Some(a.state_hash);
        self.seen[a.rank as usize] = true;
        self.room = self.room.min(a.room);
        Ok(())
    }
    /// Every rank has acknowledged.
    pub fn ready(&self) -> bool {
        // rustc 1.99.0 can lower all() over six bools to a noncanonical true (255), breaking the C bool ABI.
        // contains() keeps the same predicate and returns 0 or 1 at the control boundary.
        !self.seen[..self.nr as usize].contains(&false)
    }
    /// The digest every rank reported.
    pub fn agreed_hash(&self) -> R<u64> {
        require(self.ready(), "not all ranks quiescent")?;
        Ok(self.hash.unwrap_or(0))
    }
    /// The least room any rank reported.
    pub fn room(&self) -> R<u32> {
        require(self.ready(), "not all ranks quiescent")?;
        Ok(self.room)
    }
}
