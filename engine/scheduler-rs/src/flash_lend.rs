//! FLASH_CAP_POLICY=borrow-idle (was engine/runtime/s2/flash_lend.h): the head's rule for lending the Flash rank's KV
//! chunks inside CAP (runtime/flash_kv.h) to the KV node beside it while the serve is idle, and recalling them before a
//! request that needs them is admitted. Both happen as Resize events on tier FLASH_KV_TIER, so every rank applies them
//! at the same commit. The head admits a request only when the least of the ranks' chunks online covers the request's
//! worst-case rows; a recall the node does not release in time leaves the request waiting, never running on offline
//! pages.

/// The rule's settings and its state (s2/flash_lend.h FlashLend, the same layout). `since_ns`: steady-clock
/// nanoseconds when the current idle stretch began.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lend {
    pub idle_ms: i64,    // idle this long before a lend; 0 = never lend (FLASH_CAP_POLICY=protected)
    pub keep: u32,       // chunks left online while lent
    pub chunk_rows: u64, // rows one chunk holds
    pub chunks: u32,     // chunks in the map
    pub idle: bool,
    pub lent: bool,
    pub since_ns: i64,
}

impl Lend {
    /// The chunks a run of `rows` rows needs online (at most all of them).
    pub fn need(&self, rows: u64) -> u32 {
        if self.chunk_rows == 0 {
            return 0;
        }
        rows.div_ceil(self.chunk_rows).min(u64::from(self.chunks)) as u32
    }

    /// Every poll of the controller: the engine is idle (no resident session, none queued). The chunk count a lend
    /// takes the ranks to, once per idle stretch of idle_ms, or None.
    pub fn poll(&mut self, idle: bool, now_ns: i64) -> Option<u32> {
        if self.idle_ms == 0 || self.chunks == 0 {
            return None;
        }
        if !idle {
            self.idle = false;
            self.lent = false;
            return None;
        }
        if !self.idle {
            self.idle = true;
            self.since_ns = now_ns;
        }
        if self.lent || now_ns - self.since_ns < self.idle_ms * 1_000_000 {
            return None;
        }
        self.lent = true;
        Some(self.keep.min(self.chunks))
    }

    /// Admission of a request whose run may reach `rows` rows, `room` the least chunks online over the ranks: 0 =
    /// admit, else the chunk count to recall to first (the request waits for it).
    pub fn recall(&self, rows: u64, room: u32) -> u32 {
        let n = self.need(rows);
        if room >= n {
            0
        } else {
            n
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lends_once_per_idle_stretch_and_recalls() {
        let mut l = Lend { idle_ms: 10, keep: 1, chunk_rows: 100, chunks: 4, ..Default::default() };
        assert_eq!(l.poll(true, 0), None);
        assert_eq!(l.poll(true, 9_999_999), None);
        assert_eq!(l.poll(true, 10_000_000), Some(1));
        assert_eq!(l.poll(true, 50_000_000), None); // once per stretch
        assert_eq!(l.poll(false, 60_000_000), None);
        assert_eq!(l.poll(true, 70_000_000), None); // a new stretch starts
        assert_eq!(l.poll(true, 80_000_000), Some(1));
        assert_eq!((l.need(1), l.need(301), l.need(10_000)), (1, 4, 4));
        assert_eq!((l.recall(250, 3), l.recall(250, 2)), (0, 3));
    }

    #[test]
    fn protected_never_lends() {
        let mut l = Lend { idle_ms: 0, keep: 1, chunk_rows: 100, chunks: 4, ..Default::default() };
        assert_eq!(l.poll(true, i64::MAX / 2), None);
        assert_eq!(Lend::default().need(5), 0);
    }
}
