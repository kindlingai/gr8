//! The head's admission FIFO (was AdmissionQueue in engine/runtime/s2/scheduler.h): requests wait here in arrival
//! order until the scheduler has a slot and the pool has pages for the one at the front. Page choice stays in the pool.

use crate::scheduler::{grant_pages, reserve_pages, Config, Err, Request, PAGE};
use std::collections::VecDeque;

/// Why the front request cannot be admitted now.
pub const STALL_NONE: u32 = 0;
pub const STALL_SLOTS: u32 = 1;
pub const STALL_CAPACITY: u32 = 2;

/// The front request's admission status: why it waits, the pages its grant takes, and the pages one grant can take now.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub why: u32,
    pub requested: u32,
    pub room: u32,
}

pub struct Queue {
    c: Config,
    q: VecDeque<Request>,
    last_revision: u64,
    last_head: u64,
    last_id: u64,
    have_id: bool,
    pub stalls: u64,
}

impl Queue {
    /// `c` must already have passed scheduler::validate.
    pub fn new(c: Config) -> Queue {
        Queue {
            c,
            q: VecDeque::new(),
            last_revision: u64::MAX,
            last_head: u64::MAX,
            last_id: 0,
            have_id: false,
            stalls: 0,
        }
    }

    /// Queues `r`; it refuses a request the pool can never hold, an id at or below the last one, and a queued id.
    pub fn push(&mut self, r: Request) -> Result<(), Err> {
        reserve_pages(&self.c, &r)?;
        if self.have_id && r.id <= self.last_id {
            return Err("native request ordinals must increase");
        }
        if self.q.iter().any(|x| x.id == r.id) {
            return Err("duplicate queued id");
        }
        self.q.push_back(r);
        self.last_id = r.id;
        self.have_id = true;
        Ok(())
    }

    /// Removes a queued request; false when `id` is not queued.
    pub fn cancel(&mut self, id: u64) -> bool {
        match self.q.iter().position(|r| r.id == id) {
            Some(i) => {
                self.q.remove(i);
                true
            }
            None => false,
        }
    }

    pub fn front(&self) -> Option<&Request> {
        self.q.front()
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// The front request was granted; `id` must be its id.
    pub fn admitted(&mut self, id: u64) -> Result<(), Err> {
        if self.q.front().is_none_or(|r| r.id != id) {
            return Err("FIFO admission violated");
        }
        self.q.pop_front();
        Ok(())
    }

    /// The front request's status, once per (front, scheduler revision): None when the queue is empty or nothing changed
    /// since the last call. `slot`: the scheduler has a free slot; `fits`: the pool can admit it; `room`: the pages one
    /// grant can take now.
    pub fn check(&mut self, revision: u64, slot: bool, fits: bool, room: u32) -> Result<Option<Status>, Err> {
        let Some(&front) = self.q.front() else { return Ok(None) };
        if self.last_head == front.id && self.last_revision == revision {
            return Ok(None);
        }
        if room > self.c.pool_rows / PAGE {
            return Err("invalid allocator view");
        }
        self.last_head = front.id;
        self.last_revision = revision;
        let s = Status {
            why: if !slot {
                STALL_SLOTS
            } else if !fits {
                STALL_CAPACITY
            } else {
                STALL_NONE
            },
            requested: grant_pages(&self.c, &front)?,
            room,
        };
        if s.why != STALL_NONE {
            self.stalls += 1;
        }
        Ok(Some(s))
    }
}
