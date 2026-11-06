//! Two threads on one pool through the C API, as on the head: the compute thread reserves, grows, lists and retires
//! runs while the control thread reads free pages, room, stats, runs and prefix matches. Each call locks the pool, so
//! every read sees a whole state.

use g6kvman::ffi::*;
use g6kvman::pool::Stats;
use g6kvman::{Run, PAGE};
use std::sync::atomic::{AtomicBool, Ordering};

const PAGES: u32 = 256;
const R_PAGES: u64 = 64;

struct Shared(*mut Handle);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    fn get(&self) -> *mut Handle {
        self.0
    }
}

fn prompt(k: u64) -> Vec<i32> {
    (0..300).map(|i| ((i as u64 * 7 + k) % 50) as i32).collect()
}

fn compute(p: *mut Handle, rounds: u64) {
    let mut live: Vec<(Run, u64)> = Vec::new();
    let mut list = vec![0u32; PAGES as usize];
    for i in 0..rounds {
        unsafe {
            if live.len() < 4 {
                let mut r = Run::default();
                if g6kv_fits(p, 4 * PAGE, 16 * PAGE) != 0 {
                    assert_eq!(g6kv_reserve(p, 4 * PAGE, 16 * PAGE, &mut r), 0);
                    live.push((r, i % 3));
                }
            }
            for (r, _) in live.iter_mut() {
                let rows = (r.pages as u64 + 1) * PAGE;
                let (mut out, mut granted) = (Run::default(), 0);
                assert_eq!(g6kv_grow(p, r, rows.min(16 * PAGE), &mut out, &mut granted), 0);
                *r = out;
                let (mut n, mut ex) = (0usize, 0);
                assert_eq!(g6kv_list(p, r, list.as_mut_ptr(), list.len(), &mut n, &mut ex), 0);
                assert_eq!(n, r.pages as usize);
                assert_eq!(list[0], r.first_page);
                assert_eq!(ex, 0);
            }
            if live.len() == 4 || i % 5 == 4 {
                let (r, k) = live.remove(0);
                if k == 0 {
                    assert_eq!(g6kv_release(p, &r), 0);
                } else {
                    let h = prompt(k);
                    let n = h.len() as i32;
                    assert_eq!(g6kv_retain(p, &r, h.as_ptr(), h.len(), n - 1, n, n - 1), 0);
                }
            }
        }
    }
    for (r, _) in live {
        unsafe { assert_eq!(g6kv_release(p, &r), 0) };
    }
}

fn control(p: *mut Handle, stop: &AtomicBool) -> u64 {
    let mut reads = 0;
    let mut uses: Vec<G6kvUse> = (0..8).map(|_| G6kvUse { table_version: 0, held: 0, worst: 0 }).collect();
    let q = prompt(1);
    while !stop.load(Ordering::Relaxed) {
        unsafe {
            assert!(g6kv_free_pages(p) <= PAGES);
            assert!(g6kv_room(p) <= PAGES);
            let mut s = Stats::default();
            assert_eq!(g6kv_stats(p, &mut s), 0);
            assert!(s.held_pages <= PAGES && s.held_pages <= s.worst_pages && s.live_runs <= 4);
            let n = g6kv_runs(p, uses.as_mut_ptr(), uses.len());
            assert!(n <= uses.len());
            let mut hit = std::mem::zeroed::<G6kvHit>();
            assert_eq!(g6kv_match(p, q.as_ptr(), q.len(), 4 * PAGE, 16 * PAGE, 1 << 20, &mut hit), 0);
            assert!(hit.skip >= 0 && hit.skip <= q.len() as i32);
            g6kv_digest(p);
        }
        reads += 1;
        // as publish_status polls: back to back reads would starve the compute thread of an unfair lock
        std::thread::sleep(std::time::Duration::from_micros(20));
    }
    reads
}

#[test]
fn compute_and_control_threads_share_a_pool() {
    let mut p: *mut Handle = std::ptr::null_mut();
    unsafe {
        assert_eq!(g6kv_new(PAGES, PAGES, (R_PAGES * PAGE) as u32, &mut p), 0);
        g6kv_keep(p, 0, 0, 0);
    }
    let (shared, stop) = (Shared(p), AtomicBool::new(false));
    let reads = std::thread::scope(|s| {
        let c = s.spawn(|| control(shared.get(), &stop));
        compute(shared.get(), 20_000);
        stop.store(true, Ordering::Relaxed);
        c.join().unwrap()
    });
    unsafe {
        let mut st = Stats::default();
        assert_eq!(g6kv_stats(p, &mut st), 0);
        assert_eq!(st.live_runs, 0);
        assert!(st.grows > 0);
        assert_eq!(g6kv_free_pages(p), PAGES);
        g6kv_free(p);
    }
    assert!(reads > 0);
}
