//! Remote rows release (window(): the engine's G6_RT_LOCAL): a split run takes its prefix and `window` pages below
//! place and its whole tail above it, so the drafter never holds one of the main model's pages; such a run is never
//! retained (its tail rows are records of its own rows store), so no later grant adopts, shares or spills them, and
//! the digest names the window.

use g6kvman::stash::{Io, Keep};
use g6kvman::{Error, Finished, Pool, Rules, Why, PAGE};

const RULES: Rules = Rules { strict: false, noend: true, grid: 0 };

fn pool(pages: u32, place: u32, rp: u32, nr: u32, window: u32) -> Pool {
    let mut p = Pool::new(pages, place, rp * 64).unwrap();
    p.keep(RULES);
    p.tail(nr, 0);
    if window > 0 {
        p.window(window, 0, 0).unwrap();
    }
    p
}

fn hist(rows: usize, salt: i32) -> Vec<i32> {
    (0..rows as i32).map(|i| i.wrapping_mul(7).wrapping_add(salt) % 1000).collect()
}

fn done(pages: usize, salt: i32) -> Finished {
    let n = pages * 64;
    Finished { history: hist(n - 1, salt), valid: (n - 2) as i32, written_end: (n - 1) as i32, pf_end: (n - 2) as i32 }
}

#[test]
fn a_window_run_takes_its_prefix_window_and_whole_tail() {
    // R/64 = 8, nr 4, place 64, window 4. A run of 40 pages: 8 + 4 below place, all 32 tail pages above
    let mut p = pool(160, 64, 8, 4, 4);
    let worst = 40 * PAGE;
    assert_eq!(p.split_view(worst), Some((12, 32, 64, 96)));
    let r = p.reserve(worst, worst).unwrap();
    let l = p.list(&r).unwrap().to_vec();
    assert_eq!(l.len(), 44);
    assert!(l[..12].iter().all(|&q| q < 64), "prefix and window below place: {l:?}");
    assert!(l[12..].iter().all(|&q| q >= 64), "the drafter's whole tail above place: {l:?}");
    // a run of the drafter's whole tail per class: two of them fill the 96 pages above place
    let _b = p.reserve(worst, worst).unwrap();
    let d = p.digest();
    assert!(p.fits(30 * PAGE, 30 * PAGE), "a tail of 22 fits the 32 left above place");
    assert!(!p.fits(44 * PAGE, 44 * PAGE) && p.split_view(44 * PAGE).unwrap().3 == 32);
    assert_eq!(p.reserve(44 * PAGE, 44 * PAGE), Err(Error::NoRoom));
    assert_eq!(p.digest(), d);
    // the window is fixed while runs are live
    assert_eq!(p.window(2, 0, 0), Err(Error::BadArgument));
}

#[test]
fn a_retained_window_run_is_adopted_with_its_pages() {
    // R/64 = 8, window 4, place 64: a 40-page run is 8 + 4 below place and 32 tail pages above it
    let mut p = pool(160, 64, 8, 4, 4);
    let worst = 40 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    let old = p.list(&a).unwrap().to_vec();
    p.retain(&a, done(40, 1)).unwrap();
    assert_eq!(p.kept_runs(), 1, "retained, with its list");
    assert!(p.is_kept(&old));
    // its next turn continues it: a hit that adopts the prefix, the window and the drafter pages below the hit
    let mut prompt = hist(40 * 64 - 1, 1);
    prompt.extend([5, 6, 7]);
    let w2 = 44 * PAGE;
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!(h.why, Why::Hit, "{h:?}");
    let r = p.take_hit(h.skip, h.age, None, w2, w2).unwrap();
    let l = p.list(&r).unwrap().to_vec();
    let d = (h.skip as u64).div_ceil(PAGE) as usize - 8;   // drafter pages below the hit
    assert_eq!(l.len(), 8 + 4 + 36);
    assert_eq!(&l[..12 + d], &old[..12 + d], "prefix, window and the drafter's pages below the hit adopted");
    assert!(l[12..].iter().all(|&q| q >= 64), "the drafter's tail stays above place: {l:?}");
    assert!(!p.is_kept(&old) && p.kept_runs() == 0);
    // the old run's other drafter pages are free again, none held twice
    let mut seen = std::collections::HashSet::new();
    assert!(l.iter().all(|q| seen.insert(*q)));
}

#[test]
fn a_retained_window_run_is_evicted_whole() {
    // the drafter class holds 48 pages, one 32-page tail and change: a second run needs the retained one's pages,
    // so the pool evicts it whole (and the engine frees its records when it sees it gone)
    let mut p = pool(112, 64, 8, 4, 4);
    let worst = 40 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    let old = p.list(&a).unwrap().to_vec();
    p.retain(&a, done(40, 2)).unwrap();
    assert!(p.is_kept(&old));
    let b = p.reserve(worst, worst).unwrap();   // a fresh conversation: no hit
    assert!(!p.is_kept(&old) && p.kept_runs() == 0);
    assert!(p.list(&b).unwrap().iter().any(|q| old.contains(q)));
}

#[test]
fn the_lease_counts_a_retained_window_runs_records() {
    let mut p = Pool::new(400, 200, 8 * 64).unwrap();
    p.keep(RULES);
    p.tail(4, 0);
    p.window(4, 12, 0).unwrap();
    let worst = 40 * PAGE;   // a tail of 32 pages: 8 records a rank
    let a = p.reserve(worst, worst).unwrap();
    p.retain(&a, done(40, 3)).unwrap();
    assert_eq!(p.ids_free(), Some(4), "a retained window run keeps its records");
    assert!(!p.fits(worst, worst), "a fresh 40-page run needs 8 more than the 4 free");
    // its own next turn takes them over: 8 of its own plus 1 of the free
    let mut prompt = hist(40 * 64 - 1, 3);
    prompt.extend([5, 6, 7]);
    let w2 = 44 * PAGE;
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!(h.why, Why::Hit, "{h:?}");
    p.take_hit(h.skip, h.age, None, w2, w2).unwrap();
    assert_eq!(p.ids_free(), Some(3));
}

#[test]
fn the_digest_names_the_window() {
    let a = pool(160, 64, 8, 4, 0);
    let b = pool(160, 64, 8, 4, 4);
    let c = pool(160, 64, 8, 4, 5);
    assert_ne!(a.digest(), b.digest());
    assert_ne!(b.digest(), c.digest());
    let mut d = Pool::new(160, 64, 8 * 64).unwrap();
    assert_eq!(d.window(4, 0, 0), Err(Error::BadArgument), "a window needs the page classes");
}

#[test]
fn the_cluster_admission_of_a_41k_prompt() {
    // the TP=4 boot of 2026-10-20: --cap 408832 (6388 pages), place 900 (G6_RT_PLACE), R = 32768 (512 pages), window
    // 40, prefix reuse on. A 41,399-token prompt with max_tokens 200 and its guard: an exclusive split run of 41,600 rows
    let mut p = pool(408_832 / 64, 900, 512, 4, 40);
    let worst = 41_600u64;
    assert!(p.exclusive(worst) && p.admissible(worst));
    assert_eq!(p.split_view(worst), Some((552, 650 - 512, 900, 6388 - 900)));
    assert!(p.fits(worst, worst), "fits() and reserve() must agree");
    let r = p.reserve(worst, worst).unwrap();
    // the grant lists its worst case and the window: the scheduler's check (scheduler.h grant_pages_ok) takes that
    assert_eq!(r.pages as u64, worst.div_ceil(PAGE) + 40);
    let l = p.list(&r).unwrap();
    assert!(l[..552].iter().all(|&q| q < 900) && l[552..].iter().all(|&q| q >= 900));
    // place 900 holds one prefix of 512 + 40 pages: a second run past R waits (fits says no, reserve agrees), and the
    // view names the class that is short
    assert!(!p.fits(worst, worst));
    assert_eq!(p.split_view(worst), Some((552, 138, 900 - 552, 6388 - 900 - 138)));
    assert_eq!(p.reserve(worst, worst), Err(Error::NoRoom));
    // the longest tail the boot line names (5488 pages, 351,232 rows) fits an empty pool, one page more does not
    let mut q = pool(408_832 / 64, 900, 512, 4, 40);
    assert!(q.fits((512 + 5488) * PAGE, (512 + 5488) * PAGE));
    assert!(!q.fits((512 + 5489) * PAGE, (512 + 5489) * PAGE));
    assert_eq!(q.reserve((512 + 5489) * PAGE, (512 + 5489) * PAGE), Err(Error::NoRoom));
}

#[test]
fn the_stores_never_read_a_window_runs_tail() {
    // block stores keeping everything, tails on: a window run's retire, its eviction pressure and a later turn queue
    // copies of its prefix pages at most, never of its window or drafter pages (where its tail sits, or sat)
    let mut p = Pool::new(160, 64, 8 * 64).unwrap();
    p.keep(RULES);
    p.stores(1);
    p.add_tier(256, Keep::All, None).unwrap();
    p.tail(4, 3);
    p.window(4, 0, 0).unwrap();
    let worst = 40 * PAGE;
    for turn in 0..3i32 {
        let a = p.reserve(worst, worst).unwrap();
        let prefix: Vec<u32> = p.list(&a).unwrap()[..8].to_vec();
        p.retain(&a, done(40, 9 + turn)).unwrap();
        assert!(p.kept_runs() <= turn as u32 + 1, "turn {turn}");
        for io in p.io() {
            match io {
                Io::Spill { page, .. } | Io::Restore { page, .. } => {
                    assert!(prefix.contains(&page), "turn {turn}: a copy of page {page} past the prefix {prefix:?}")
                }
                Io::Move { .. } => {}
            }
        }
        p.io_done().unwrap();
    }
    // a later turn of the first conversation: a pool hit while its run is retained, else a stored hit, which restores
    // at most the prefix
    let mut prompt = hist(40 * 64 - 1, 9);
    prompt.push(1);
    let w2 = 41 * PAGE;
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert!(!h.stored || h.skip <= 8 * 64, "{h:?}");
    if h.skip > 0 {
        let r = p.take_hit(h.skip, h.age, h.stored.then_some(h.tier), w2, w2).unwrap();
        let prefix: Vec<u32> = p.list(&r).unwrap()[..8].to_vec();
        for io in p.io() {
            if let Io::Restore { page, .. } = io {
                assert!(prefix.contains(&page), "restore into page {page} past the prefix");
            }
        }
        p.io_done().unwrap();
    }
}

#[test]
fn a_tail_waits_for_records_in_the_lease() {
    // each rank's lease holds 12 records; a 40-page run's tail of 32 pages needs 8 on the rank holding most of it
    let mut p = Pool::new(160, 64, 8 * 64).unwrap();
    p.keep(RULES);
    p.tail(4, 0);
    p.window(4, 12, 0).unwrap();
    let worst = 40 * PAGE;
    assert_eq!(p.ids_view(worst), Some((8, 12, 12)));
    let a = p.reserve(worst, worst).unwrap();
    assert_eq!(p.ids_free(), Some(4));
    // the classes have room for a second run, the lease does not: it waits, and reserve agrees
    assert!(p.split_view(worst).unwrap().2 >= 12 && !p.fits(worst, worst));
    assert_eq!(p.reserve(worst, worst), Err(Error::NoRoom));
    // a shorter tail fits the 4 left: 24 pages, a tail of 16, 4 a rank
    assert!(p.fits(24 * PAGE, 24 * PAGE));
    // a tail longer than the whole lease never fits
    assert_eq!(p.ids_view(60 * PAGE).unwrap().0, 13);
    p.release(&a).unwrap();
    assert!(p.fits(worst, worst) && !p.fits(60 * PAGE, 60 * PAGE));
    assert_eq!(p.ids_free(), Some(12));
}

#[test]
fn a_session_past_r_waits_for_its_share_of_the_slots() {
    // the slot pool covers two sessions past R at once: a third waits until one leaves; runs below R are not counted
    let mut p = Pool::new(400, 200, 8 * 64).unwrap();
    p.keep(RULES);
    p.tail(4, 0);
    p.window(4, 0, 2).unwrap();
    let worst = 20 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    let _b = p.reserve(worst, worst).unwrap();
    assert_eq!(p.window_runs(), 2);
    assert!(!p.fits(worst, worst), "a third session past R waits");
    assert_eq!(p.reserve(worst, worst), Err(Error::NoRoom));
    assert!(p.fits(3 * PAGE, 3 * PAGE), "a run below R does not take a share");
    p.release(&a).unwrap();
    assert!(p.fits(worst, worst));
    let d = pool(400, 200, 8, 4, 4).digest();
    let mut e = Pool::new(400, 200, 8 * 64).unwrap();
    e.keep(RULES);
    e.tail(4, 0);
    e.window(4, 0, 2).unwrap();
    assert_ne!(d, e.digest(), "the digest names the limit");
}
