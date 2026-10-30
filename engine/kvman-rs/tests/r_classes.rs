//! R per grant (reserve_r and the other _r calls): with the window on, a grant may take a smaller R than the pool's. A
//! run past it holds that R's prefix and window below place and its tail above, the window-run limit counts runs at
//! every R, a retained run past one R is no hit for a grant at another, and a pool whose grants all take its own R
//! hashes as before.

use g6kvman::{Error, Finished, Pool, Rules, Why, PAGE};

const RULES: Rules = Rules { strict: false, noend: true, grid: 0 };
// the pool's R is 8 pages, the small one 4
const BIG: u32 = 8 * 64;
const SMALL: u32 = 4 * 64;

// place 40, window 4, nr 4: a seat at R = 8 pages takes 12 pages below place, one at 4 takes 8
fn pool(runs: u32) -> Pool {
    let mut p = Pool::new(400, 40, BIG).unwrap();
    p.keep(RULES);
    p.tail(4, 0);
    p.window(4, 0, runs).unwrap();
    p
}

fn hist(rows: usize, salt: i32) -> Vec<i32> {
    (0..rows as i32).map(|i| i.wrapping_mul(7).wrapping_add(salt) % 1000).collect()
}

fn done(pages: usize, salt: i32) -> Finished {
    let n = pages * 64;
    Finished { history: hist(n - 1, salt), valid: (n - 2) as i32, written_end: (n - 1) as i32, pf_end: (n - 2) as i32 }
}

// a seat: a run of 20 pages
const WORST: u64 = 20 * PAGE;

fn seats(p: &mut Pool, r: u32) -> usize {
    let mut n = 0;
    while p.reserve_r(WORST, WORST, r).is_ok() {
        n += 1;
    }
    n
}

#[test]
fn a_smaller_r_takes_its_own_prefix_and_window() {
    let mut p = pool(0);
    assert_eq!(p.split_view_r(WORST, SMALL), Some((8, 16, 40, 360)));
    assert_eq!(p.split_view_r(WORST, 0), p.split_view(WORST));
    let r = p.reserve_r(WORST, WORST, SMALL).unwrap();
    let l = p.list(&r).unwrap().to_vec();
    assert_eq!(l.len(), 4 + 4 + 16);
    assert!(l[..8].iter().all(|&q| q < 40), "prefix and window below place: {l:?}");
    assert!(l[8..].iter().all(|&q| q >= 40), "the whole tail above place: {l:?}");
    // a request between the two Rs is past the small one only
    assert!(p.exclusive_r(6 * PAGE, SMALL).unwrap() && !p.exclusive_r(6 * PAGE, 0).unwrap());
}

#[test]
fn seats_follow_the_r_of_each_grant() {
    assert_eq!(seats(&mut pool(0), 0), 3, "3 x 12 of 40 pages");
    assert_eq!(seats(&mut pool(0), SMALL), 5, "5 x 8");
    let mut p = pool(0);
    for _ in 0..2 {
        p.reserve_r(WORST, WORST, 0).unwrap();
    }
    assert_eq!(seats(&mut p, SMALL), 2, "2 x 12 + 2 x 8");
    // the window-run limit counts runs at both Rs
    let mut p = pool(3);
    p.reserve_r(WORST, WORST, 0).unwrap();
    assert_eq!(seats(&mut p, SMALL), 2);
    assert_eq!(p.window_runs(), 3);
}

#[test]
fn a_pool_at_its_own_r_hashes_as_before() {
    let (mut a, mut b, mut c) = (pool(0), pool(0), pool(0));
    let ra = a.reserve(WORST, WORST).unwrap();
    let rb = b.reserve_r(WORST, WORST, 0).unwrap();
    let rc = c.reserve_r(WORST, WORST, BIG).unwrap();
    assert!(ra == rb && rb == rc);
    assert!(a.digest() == b.digest() && b.digest() == c.digest());
    a.retain(&ra, done(20, 1)).unwrap();
    b.retain(&rb, done(20, 1)).unwrap();
    assert_eq!(a.digest(), b.digest());
    // a run at the small R is named in the digest beside its list
    let (mut d, mut e) = (pool(0), pool(0));
    let rd = d.reserve_r(WORST, WORST, SMALL).unwrap();
    let re = e.reserve_r(WORST, WORST, SMALL).unwrap();
    assert_eq!(d.digest(), e.digest());
    assert_ne!(d.digest(), a.digest());
    d.retain(&rd, done(20, 1)).unwrap();
    e.retain(&re, done(20, 1)).unwrap();
    assert_eq!(d.digest(), e.digest());
}

// the next turn of a retained 20-page run, at R = r
fn next_turn(p: &Pool, r: u32) -> g6kvman::Hit {
    let mut prompt = hist(20 * 64 - 1, 1);
    prompt.extend([5, 6, 7]);
    let w = 24 * PAGE;
    p.match_prefix_r(&prompt, w, w, prompt.len() as i32 - 1, g6kvman::pool::STOCK_NAMESPACE, r)
}

#[test]
fn a_retained_run_past_one_r_is_no_hit_at_another() {
    for (kept, other) in [(SMALL, 0), (0, SMALL)] {
        let mut p = pool(0);
        let a = p.reserve_r(WORST, WORST, kept).unwrap();
        p.retain(&a, done(20, 1)).unwrap();
        assert_eq!(next_turn(&p, other).why, Why::Room, "kept at {kept}, asked at {other}");
        let h = next_turn(&p, kept);
        assert_eq!(h.why, Why::Hit, "kept at {kept}: {h:?}");
        let w = 24 * PAGE;
        assert_eq!(p.take_hit_r(h.skip, h.age, None, w, w, other), Err(Error::NoRoom));
        let r = p.take_hit_r(h.skip, h.age, None, w, w, kept).unwrap();
        assert_eq!(p.list(&r).unwrap().len(), 24 + 4);
    }
}

#[test]
fn a_run_below_r_gives_a_smaller_r_only_its_prefix() {
    // retained runs below the pool's R: 6 pages is past the small prefix, 3 inside it
    for (pages, small) in [(6, Why::Room), (3, Why::Hit)] {
        let mut p = pool(0);
        let rows = pages as u64 * PAGE;
        let a = p.reserve(rows, rows).unwrap();
        p.retain(&a, done(pages, 1)).unwrap();
        let mut prompt = hist(pages * 64 - 1, 1);
        prompt.extend([5, 6, 7]);
        let ask =
            |r| p.match_prefix_r(&prompt, WORST, WORST, prompt.len() as i32 - 1, g6kvman::pool::STOCK_NAMESPACE, r);
        assert_eq!(ask(0).why, Why::Hit, "{pages} pages at the pool's R");
        assert_eq!(ask(SMALL).why, small, "{pages} pages at the small R");
    }
}

#[test]
fn an_r_the_pool_cannot_lay_out_is_refused() {
    let mut p = pool(0);
    for r in [BIG + 64, SMALL + 1] {
        assert_eq!(p.reserve_r(WORST, WORST, r), Err(Error::BadArgument), "R {r}");
        assert!(!p.fits_r(WORST, WORST, r) && p.split_view_r(WORST, r).is_none());
    }
    // without the window a run past R reaches the stores, which hold tails at the pool's R only
    let mut q = Pool::new(400, 40, BIG).unwrap();
    q.tail(4, 0);
    assert_eq!(q.reserve_r(WORST, WORST, SMALL), Err(Error::BadArgument));
    assert!(q.reserve_r(WORST, WORST, BIG).is_ok());
}
