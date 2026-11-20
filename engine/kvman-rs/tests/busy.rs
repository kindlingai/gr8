//! The busy guard's busy_view: a dry run of reserve's evictions. Before every grant it must name exactly the retained
//! runs the grant then evicts, in each pool shape (below R, split runs, split runs with a window and a record lease).

use g6kvman::{Finished, Pool, Rules, Run, PAGE};

const RULES: Rules = Rules { strict: false, noend: true, grid: 0 };

fn done(pages: u64, salt: i32) -> Finished {
    let n = pages as usize * 64;
    let history = (0..n as i32 - 1).map(|i| i.wrapping_mul(7).wrapping_add(salt * 131) % 1000).collect();
    Finished { history, valid: (n - 2) as i32, written_end: (n - 1) as i32, pf_end: (n - 2) as i32 }
}

fn ages(p: &Pool) -> Vec<u64> {
    p.kept().iter().map(|k| k.age).collect()
}

// Grants `sizes` (pages, worst case at once) one after another, retiring all but the newest live run before each.
// A grant that does not fit retires every live run first, and is skipped if it still does not fit. Returns how many
// retained runs the grants evicted.
fn grants_match_the_view(mut p: Pool, sizes: &[u64]) -> usize {
    let (mut live, mut salt, mut evicted): (Vec<(Run, u64)>, i32, usize) = (Vec::new(), 0, 0);
    for (i, &pages) in sizes.iter().enumerate() {
        let worst = pages * PAGE;
        while live.len() > 1 || (!p.fits(worst, worst) && !live.is_empty()) {
            let (r, n) = live.remove(0);
            salt += 1;
            p.retain(&r, done(n, salt)).unwrap();
        }
        if !p.fits(worst, worst) {
            continue;
        }
        let before = ages(&p);
        let view = p.busy_view(worst, worst, 0, p.now_ns(), 0);
        let r = p.reserve(worst, worst).unwrap();
        let after = ages(&p);
        let gone: Vec<u64> = before.into_iter().filter(|a| !after.contains(a)).collect();
        assert_eq!(view.victims, gone, "grant {i} of {pages} pages");
        assert_eq!(view.live_victims, 0, "idle_ns 0 counts no run as live");
        evicted += gone.len();
        live.push((r, pages));
    }
    evicted
}

const SIZES: [u64; 24] = [8, 6, 24, 5, 40, 7, 8, 30, 4, 44, 8, 20, 6, 36, 8, 8, 28, 5, 40, 7, 24, 8, 44, 6];

#[test]
fn the_view_names_what_a_grant_below_r_evicts() {
    let mut p = Pool::new(64, 64, 64 * 64).unwrap();
    p.keep(RULES);
    assert!(grants_match_the_view(p, &SIZES) > 4);
}

// R = 8 pages, 4 ranks, 64 pages below place. A small drafter class makes both fills of one grant evict.
fn split(pages: u32, window: u32) -> Pool {
    let mut p = Pool::new(pages, 64, 8 * 64).unwrap();
    p.keep(RULES);
    p.tail(4, 0);
    if window > 0 {
        p.window(window, 12, 0).unwrap();
    }
    p
}

#[test]
fn the_view_names_what_a_split_grant_evicts_from_both_classes() {
    for pages in [90, 110, 160] {
        assert!(grants_match_the_view(split(pages, 0), &SIZES) > 4, "{pages} pages");
    }
}

#[test]
fn the_view_names_what_a_window_grant_evicts_for_pages_and_records() {
    for pages in [90, 110, 160] {
        assert!(grants_match_the_view(split(pages, 4), &SIZES) > 4, "{pages} pages");
    }
}

#[test]
fn an_exclusive_grant_evicts_every_retained_run() {
    let mut p = Pool::new(64, 64, 8 * 64).unwrap();
    p.keep(RULES);
    for salt in 0..3 {
        let r = p.reserve(4 * PAGE, 4 * PAGE).unwrap();
        p.retain(&r, done(4, salt)).unwrap();
    }
    let view = p.busy_view(20 * PAGE, 20 * PAGE, 0, p.now_ns(), 0);
    assert_eq!(view.victims, ages(&p));
    p.reserve(20 * PAGE, 20 * PAGE).unwrap();
    assert_eq!(p.kept_runs(), 0);
}

#[test]
fn a_victim_is_live_until_idle_ns_after_its_retire() {
    let mut p = Pool::new(64, 64, 64 * 64).unwrap();
    p.keep(RULES);
    for salt in 0..2 {
        let r = p.reserve(32 * PAGE, 32 * PAGE).unwrap();
        p.retain(&r, done(32, salt)).unwrap();
    }
    // 32 pages evict only the older run
    let t = p.kept()[0].retired_ns;
    let live = |now, idle| p.busy_view(32 * PAGE, 32 * PAGE, 0, now, idle).live_victims;
    assert_eq!(p.busy_view(32 * PAGE, 32 * PAGE, 0, t, 0).victims, vec![p.kept()[0].age]);
    assert_eq!(live(t + 10, 10), 1);
    assert_eq!(live(t + 11, 10), 0);
    assert_eq!(live(t, 0), 0, "idle_ns 0: no run is live");
}
