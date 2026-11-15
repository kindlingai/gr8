//! The busy guard's busy_view: a dry run of reserve's evictions. Before every grant it must name exactly the retained
//! runs the grant then evicts, in each pool shape (below R, split runs, split runs with a window and a record lease).

use g6kvman::pool::{Idle, STOCK_NAMESPACE};
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
        let view = p.busy_view(worst, worst, 0, p.now_ns(), &Idle::default());
        let r = p.reserve(worst, worst).unwrap();
        let after = ages(&p);
        let gone: Vec<u64> = before.into_iter().filter(|a| !after.contains(a)).collect();
        assert_eq!(view.victims, gone, "grant {i} of {pages} pages");
        assert_eq!(view.live_victims, 0, "zero bounds count no run as live");
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
    let view = p.busy_view(20 * PAGE, 20 * PAGE, 0, p.now_ns(), &Idle::default());
    assert_eq!(view.victims, ages(&p));
    p.reserve(20 * PAGE, 20 * PAGE).unwrap();
    assert_eq!(p.kept_runs(), 0);
}

// Two 32-page runs retire, the older ending on a tool call when `tool`. A 32-page grant evicts only the older one.
// Returns the pool and the older run's retire time.
fn two_runs(tool: bool) -> (Pool, i64) {
    let mut p = Pool::new(64, 64, 64 * 64).unwrap();
    p.keep(RULES);
    for salt in 0..2 {
        let r = p.reserve(32 * PAGE, 32 * PAGE).unwrap();
        p.retain_ended(&r, done(32, salt), STOCK_NAMESPACE, tool && salt == 0).unwrap();
    }
    let t = p.kept()[0].retired_ns;
    assert_eq!(p.busy_view(32 * PAGE, 32 * PAGE, 0, t, &Idle::default()).victims, vec![p.kept()[0].age]);
    (p, t)
}

// a 32-page run holds 2046 valid rows
const BOUNDS: Idle = Idle { short_rows: 2000, short_ns: 5, tool_ns: 90, finish_ns: 30 };

fn live(p: &Pool, now: i64, idle: &Idle) -> (u32, i64) {
    let v = p.busy_view(32 * PAGE, 32 * PAGE, 0, now, idle);
    (v.live_victims, v.idle_in_ns)
}

#[test]
fn a_finished_run_is_live_for_the_finish_bound() {
    let (p, t) = two_runs(false);
    assert_eq!(live(&p, t + 10, &BOUNDS), (1, 20));
    assert_eq!(live(&p, t + 30, &BOUNDS), (1, 0));
    assert_eq!(live(&p, t + 31, &BOUNDS), (0, 0));
}

#[test]
fn a_run_that_ended_on_a_tool_call_is_live_for_the_tool_bound() {
    let (p, t) = two_runs(true);
    assert!(p.kept()[0].tool && !p.kept()[1].tool);
    assert_eq!(live(&p, t + 31, &BOUNDS), (1, 59));
    assert_eq!(live(&p, t + 91, &BOUNDS), (0, 0));
}

#[test]
fn a_short_run_takes_the_short_bound_however_it_ended() {
    let short = Idle { short_rows: 2047, ..BOUNDS };
    for tool in [false, true] {
        let (p, t) = two_runs(tool);
        assert_eq!(live(&p, t + 5, &short), (1, 0));
        assert_eq!(live(&p, t + 6, &short), (0, 0), "tool {tool}");
    }
}

#[test]
fn a_zero_bound_counts_the_run_idle_at_once() {
    let (p, t) = two_runs(false);
    assert_eq!(live(&p, t, &Idle { finish_ns: 0, ..BOUNDS }), (0, 0));
}
