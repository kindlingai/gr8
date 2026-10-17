//! Several split runs (past R, page classes on: tail(nr > 0)) live at once, beside runs below R: the pool admits them
//! while others are live (admissible, fits, reserve), finds and takes prefix hits for them while others are live
//! (match_prefix, take_hit, hit_room, restore_room), keeps the Banker's state safe per page class (below place: every
//! live run's pages there, the runs below R growing to their worst cases; above place: the drafter's tail copies, which
//! never grow), and the digest follows the new state. nr = 0 keeps the exclusive rule.

use g6kvman::stash::Keep;
use g6kvman::{Error, Finished, Pool, Rules, Run, Why, PAGE};

const RULES: Rules = Rules { strict: false, noend: true, grid: 0 };

// a pool of `pages` with `place` below, R = rp pages, nr ranks, prefix reuse on
fn pool(pages: u32, place: u32, rp: u32, nr: u32) -> Pool {
    let mut p = Pool::new(pages, place, rp * 64).unwrap();
    p.keep(RULES);
    p.tail(nr, 0);
    p
}

fn hist(rows: usize, salt: i32) -> Vec<i32> {
    (0..rows as i32).map(|i| i.wrapping_mul(7).wrapping_add(salt) % 1000).collect()
}

// a finished conversation of `pages` pages whose history is hist(.., salt)
fn done(pages: usize, salt: i32) -> Finished {
    let n = pages * 64;
    Finished { history: hist(n - 1, salt), valid: (n - 2) as i32, written_end: (n - 1) as i32, pf_end: (n - 2) as i32 }
}

// the pages of a split run of `worst` rows: (below place, above place)
fn classes(worst: u64, rp: u64, nr: u64) -> (u64, u64) {
    let t = worst.div_ceil(PAGE).saturating_sub(rp);
    let own = t.div_ceil(nr).min(t);
    (rp + own, t - own)
}

fn low_high(p: &Pool, r: &Run) -> (u64, u64) {
    let l = p.list(r).unwrap();
    let low = l.iter().filter(|&&q| q < p.place_pages()).count() as u64;
    (low, l.len() as u64 - low)
}

// every live page once, the classes within their sizes, the Banker's state safe below place
struct Live {
    run: Run,
    worst: u64,
}
fn check(p: &Pool, live: &[Live]) {
    let mut seen = vec![false; p.pool_pages() as usize];
    let (mut low, mut high) = (0u64, 0u64);
    let mut need: Vec<(u64, u64)> = Vec::new();
    for l in live {
        let list = p.list(&l.run).unwrap();
        for &q in list {
            assert!(!seen[q as usize] && p.used(q), "page {q} held twice");
            seen[q as usize] = true;
        }
        let (lo, hi) = low_high(p, &l.run);
        low += lo;
        high += hi;
        if !p.exclusive_run(&l.run).unwrap() {
            assert_eq!(hi, 0, "a run below R above place");
        }
        need.push((l.worst.div_ceil(PAGE).saturating_sub(list.len() as u64), lo));
    }
    assert_eq!(low + high, p.live_pages() as u64);
    assert!(low <= p.place_pages() as u64 && high <= (p.pool_pages() - p.place_pages()) as u64);
    for k in p.kept() {
        for &q in &k.list {
            assert!(!seen[q as usize], "retained page {q} also live or retained twice");
            seen[q as usize] = true;
        }
    }
    // Banker's, by brute order: smallest need first
    let mut free = p.place_pages() as i64 - low as i64;
    need.sort_unstable();
    for (n, h) in need {
        assert!(n as i64 <= free, "pool left unsafe below place");
        free += h as i64;
    }
}

#[test]
fn four_split_runs_live_together() {
    // R/64 = 8, nr 4, place 64, drafter class 96. A run of 40 pages: 8 + 8 below place, 24 above
    let mut p = pool(160, 64, 8, 4);
    let worst = 40 * PAGE;
    assert_eq!(classes(worst, 8, 4), (16, 24));
    let mut live = Vec::new();
    for i in 0..4 {
        assert!(p.admissible(worst), "split run {i} admissible while {} live", live.len());
        assert!(p.fits(worst, worst), "split run {i} fits");
        let r = p.reserve(worst, worst).unwrap();
        assert!(p.exclusive_run(&r).unwrap());
        assert_eq!(low_high(&p, &r), (16, 24));
        live.push(Live { run: r, worst });
        check(&p, &live);
    }
    assert_eq!(p.stats().live_runs, 4);
    // a fifth: both classes are full; refused, and the pool is as it was
    let d = p.digest();
    assert!(p.admissible(worst) && !p.fits(worst, worst));
    assert_eq!(p.reserve(worst, worst), Err(Error::NoRoom));
    assert_eq!(p.digest(), d);
    // one leaves: the fifth fits on its pages
    let gone = live.remove(1);
    p.release(&gone.run).unwrap();
    let r = p.reserve(worst, worst).unwrap();
    live.push(Live { run: r, worst });
    check(&p, &live);
}

#[test]
fn the_drafter_class_bounds_split_runs() {
    // place 64 has room for 4 runs' 16 low pages, the drafter class only for 2 runs' 24 high pages
    let mut p = pool(112, 64, 8, 4);
    let worst = 40 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    let b = p.reserve(worst, worst).unwrap();
    assert!(p.room() >= 16, "low pages are left");
    assert!(!p.fits(worst, worst), "the drafter class is full");
    // a split run with a short tail still fits: 9 pages, 8 + 1 low, 0 high
    assert!(p.fits(9 * PAGE, 9 * PAGE));
    let c = p.reserve(9 * PAGE, 9 * PAGE).unwrap();
    assert_eq!(low_high(&p, &c), (9, 0));
    check(&p, &[Live { run: a, worst }, Live { run: b, worst }, Live { run: c, worst: 9 * PAGE }]);
}

#[test]
fn a_split_run_keeps_the_runs_below_r_safe() {
    // place 80, R/64 = 32, drafter class 220. A run below R holds 4 pages and may grow to 30. A split run holds its
    // worst case (need 0), so in the Banker's order it finishes first and gives its low pages back: it is admitted
    // when its low pages are free now, and the run below R waits for them (it sits rounds out, it never deadlocks)
    let mut p = pool(300, 80, 32, 4);
    let short = p.reserve(4 * PAGE, 30 * PAGE).unwrap();
    let big = 212 * PAGE; // 32 + 45 = 77 low > 76 free
    assert_eq!(classes(big, 32, 4), (77, 135));
    assert!(!p.fits(big, big), "77 low pages, 76 free");
    assert_eq!(p.reserve(big, big), Err(Error::NoRoom));
    let ok = 208 * PAGE; // 32 + 44 = 76 low, 132 high
    assert_eq!(classes(ok, 32, 4), (76, 132));
    assert!(p.fits(ok, ok));
    let s = p.reserve(ok, ok).unwrap();
    assert_eq!(low_high(&p, &s), (76, 132));
    check(&p, &[Live { run: short, worst: 30 * PAGE }, Live { run: s, worst: ok }]);
    // no low page left: the run below R cannot grow (a stall), nothing below R fits, but the state is safe
    assert_eq!(p.grow(&short, 5 * PAGE).unwrap(), None);
    assert!(!p.fits(PAGE, PAGE));
    // the split run leaves: the run below R grows to its worst case
    p.release(&s).unwrap();
    let g = p.grow(&short, 30 * PAGE).unwrap().expect("its pages are back");
    check(&p, &[Live { run: g, worst: 30 * PAGE }]);
}

#[test]
fn growth_below_r_counts_split_runs_low_pages() {
    let mut p = pool(200, 80, 32, 4);
    let a = p.reserve(2 * PAGE, 20 * PAGE).unwrap();
    let b = p.reserve(2 * PAGE, 20 * PAGE).unwrap();
    // a split run of 50 low pages: 26 low pages free
    let s = p.reserve(104 * PAGE, 104 * PAGE).unwrap();
    assert_eq!((low_high(&p, &s), p.room()), ((50, 54), 26));
    let a = p.grow(&a, 12 * PAGE).unwrap().unwrap(); // +10: 16 free
    let b = p.grow(&b, 12 * PAGE).unwrap().unwrap(); // +10: 6 free (safe: the split run finishes first)
    // +8 more for b: 8 > 6 low pages free: a stall. Without the split run's 50 low pages it would be granted
    assert_eq!(p.grow(&b, 20 * PAGE).unwrap(), None);
    assert_eq!(p.stats().stalls, 1);
    check(&p, &[Live { run: a, worst: 20 * PAGE }, Live { run: b, worst: 20 * PAGE }, Live { run: s, worst: 104 * PAGE }]);
    p.release(&s).unwrap();
    assert!(p.grow(&b, 20 * PAGE).unwrap().is_some(), "the split run's low pages are back");
}

#[test]
fn a_split_run_hits_its_prefix_while_others_run() {
    let mut p = pool(160, 64, 8, 4);
    let worst = 40 * PAGE;
    // agent A's first turn, retained
    let a = p.reserve(worst, worst).unwrap();
    p.retain(&a, done(40, 1)).unwrap();
    // agents B (split) and C (below R) are live
    let b = p.reserve(worst, worst).unwrap();
    let c = p.reserve(3 * PAGE, 6 * PAGE).unwrap();
    assert_eq!(p.kept_runs(), 1, "A's pages survive: the classes had room");
    // A's second turn continues its conversation: a hit, while two runs are live
    let mut prompt = hist(40 * 64 - 1, 1);
    prompt.extend([5, 6, 7]);
    let w2 = 44 * PAGE;
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!(h.why, Why::Hit, "{h:?}");
    assert!(h.skip as usize >= 39 * 64, "{h:?}");
    let kept_list = p.kept()[0].list.clone();
    let r = p.take_hit(h.skip, h.age, None, w2, w2).unwrap();
    assert!(p.exclusive_run(&r).unwrap());
    let a_pages = (h.skip as u64).div_ceil(PAGE) as usize;
    assert_eq!(&p.list(&r).unwrap()[..a_pages], &kept_list[..a_pages], "the hit adopts A's pages");
    let (lo, _) = low_high(&p, &r);
    assert!(lo >= classes(w2, 8, 4).0);
    check(&p, &[Live { run: b, worst }, Live { run: c, worst: 6 * PAGE }, Live { run: r, worst: w2 }]);
}

#[test]
fn a_split_hit_without_room_is_a_room_miss() {
    // place 64, drafter class 48. A (40 pages: 16 low, 24 high) retained, b (the same) live: A's hit for 41 pages
    // needs 17 low (16 adopted + 1 fresh) and 24 high (adopted)
    let mut p = pool(112, 64, 8, 4);
    let worst = 40 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    p.retain(&a, done(40, 2)).unwrap();
    let _b = p.reserve(worst, worst).unwrap();
    assert_eq!(p.kept_runs(), 1);
    let mut prompt = hist(40 * 64 - 1, 2);
    prompt.push(9);
    let w2 = 41 * PAGE;
    assert_eq!(classes(w2, 8, 4), (17, 24));
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!(h.why, Why::Hit, "room beside one live split run: {h:?}");
    // runs below R take the 32 spare low pages (A's stay retained): 16 low pages are left, A's hit needs 17
    let mut lows = Vec::new();
    while p.room() > 16 {
        lows.push(p.reserve(PAGE, PAGE).unwrap());
    }
    assert_eq!((p.room(), p.kept_runs(), lows.len()), (16, 1, 32));
    let h2 = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!((h2.skip, h2.why), (0, Why::Room), "{h2:?}");
    let d = p.digest();
    assert_eq!(p.take_hit(h.skip, h.age, None, w2, w2), Err(Error::NoRoom));
    assert_eq!(p.digest(), d, "a refused take changes nothing");
    // one run below R leaves: the hit has its room again
    p.release(&lows.pop().unwrap()).unwrap();
    let h3 = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert_eq!(h3.why, Why::Hit, "{h3:?}");
    assert!(p.take_hit(h3.skip, h3.age, None, w2, w2).is_ok());
}

#[test]
fn a_stored_split_run_restores_while_others_run() {
    // the stores hold A's tail; a later turn restores it into a fresh two-class grant while another run is live
    let mut p = pool(112, 64, 8, 4);
    p.stores(2);
    p.add_tier(256, Keep::All, None).unwrap();
    p.tail(4, 3);
    let worst = 40 * PAGE;
    let a = p.reserve(worst, worst).unwrap();
    p.retain(&a, done(40, 3)).unwrap();
    let _ = p.io();
    p.io_done().unwrap();
    // two live split runs need the drafter class: A's retained run is evicted to the store
    let b = p.reserve(worst, worst).unwrap();
    let c = p.reserve(worst, worst).unwrap();
    assert_eq!(p.kept_runs(), 0, "A spilled");
    let _ = p.io();
    p.io_done().unwrap();
    p.release(&c).unwrap();
    // A's next turn: a stored hit while b is live
    let mut prompt = hist(40 * 64 - 1, 3);
    prompt.push(4);
    let w2 = 41 * PAGE;
    let h = p.match_prefix(&prompt, w2, w2, prompt.len() as i32 - 1);
    assert!(h.stored && h.skip as usize >= 8 * 64, "A's rows are in the store: {h:?}");
    let r = p.take_hit(h.skip, h.age, Some(h.tier), w2, w2).unwrap();
    assert!(!p.io().is_empty(), "the restore is queued");
    p.io_done().unwrap();
    check(&p, &[Live { run: b, worst }, Live { run: r, worst: w2 }]);
}

#[test]
fn without_page_classes_a_past_r_run_stays_exclusive() {
    let mut p = Pool::new(100, 60, 8 * 64).unwrap();
    p.keep(RULES);
    let s = p.reserve(PAGE, PAGE).unwrap();
    assert!(!p.admissible(20 * PAGE), "nr = 0: a past-R run waits for an empty pool");
    p.release(&s).unwrap();
    let x = p.reserve(20 * PAGE, 20 * PAGE).unwrap();
    assert_eq!(p.list(&x).unwrap(), &(0..20).collect::<Vec<u32>>()[..]);
    assert_eq!((p.room(), p.free_pages(), p.admissible(PAGE), p.admissible(20 * PAGE)), (0, 0, false, false));
}

#[test]
fn the_digest_follows_split_runs() {
    let mut a = pool(160, 64, 8, 4);
    let mut b = pool(160, 64, 8, 4);
    let w = 40 * PAGE;
    for p in [&mut a, &mut b] {
        p.reserve(w, w).unwrap();
        p.reserve(3 * PAGE, 9 * PAGE).unwrap();
    }
    assert_eq!(a.digest(), b.digest());
    // one rank admits one more split run: another digest
    let extra = a.reserve(w, w).unwrap();
    assert_ne!(a.digest(), b.digest());
    // the same split run on both: equal again
    let same = b.reserve(w, w).unwrap();
    assert_eq!(extra, same);
    assert_eq!(a.digest(), b.digest());
    // one rank retires the split run, the other releases it: the retained state differs
    a.retain(&extra, done(40, 5)).unwrap();
    b.release(&same).unwrap();
    assert_ne!(a.digest(), b.digest());
}

// ---- random traces: split and below-R runs, growth, hits, retire, release, on two ranks
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

#[test]
fn random_traces_split_runs_share_the_pool() {
    let (mut splits_together, mut split_hits_live, mut grants) = (0u32, 0u32, 0u32);
    for seed in 1..=60u64 {
        let mut rng = Rng(seed * 104729);
        let nr = rng.range(2, 6) as u32;
        let rp = rng.range(1, 8) as u32;
        let place = rng.range(rp as u64 + 4, 200) as u32;
        let pages = place + rng.range(0, 300) as u32;
        let mut a = pool(pages, place, rp, nr);
        let mut b = pool(pages, place, rp, nr);
        let mut live: Vec<Live> = Vec::new();
        let mut convs: Vec<(Vec<i32>, i32)> = Vec::new(); // retired histories (prompt to continue, salt)
        for step in 0..1500 {
            match rng.range(0, 9) {
                0..=4 => {
                    // a new request or a continued conversation; past R or below it
                    let (mut prompt, salt) = if !convs.is_empty() && rng.range(0, 2) == 0 {
                        convs[rng.range(0, convs.len() as u64 - 1) as usize].clone()
                    } else {
                        (Vec::new(), step as i32)
                    };
                    let extra = rng.range(1, (place as u64 * PAGE / 3).max(2));
                    prompt.extend((0..extra).map(|i| ((i as i32).wrapping_mul(13) + salt) % 1000));
                    let rows = prompt.len() as u64 + rng.range(1, 64);
                    let worst = rows + rng.range(0, 6) * PAGE;
                    if !a.admissible(worst) || !a.fits(rows, worst) {
                        let d = a.digest();
                        assert!(a.reserve(rows, worst).is_err());
                        assert_eq!(a.digest(), d, "a refused reservation changes nothing");
                        continue;
                    }
                    let h = a.match_prefix(&prompt, rows, worst, prompt.len() as i32 - 1);
                    assert_eq!(h, b.match_prefix(&prompt, rows, worst, prompt.len() as i32 - 1));
                    let (ra, rb) = if h.skip > 0 {
                        if a.exclusive(worst) && !live.is_empty() {
                            split_hits_live += 1;
                        }
                        (a.take(h.skip, h.age, rows, worst).unwrap(), b.take(h.skip, h.age, rows, worst).unwrap())
                    } else {
                        (a.reserve(rows, worst).unwrap(), b.reserve(rows, worst).unwrap())
                    };
                    assert_eq!(ra, rb);
                    grants += 1;
                    if a.exclusive(worst) && live.iter().any(|l| a.exclusive_run(&l.run).unwrap()) {
                        splits_together += 1;
                    }
                    live.push(Live { run: ra, worst });
                }
                5 if !live.is_empty() => {
                    // a run below R grows
                    let i = rng.range(0, live.len() as u64 - 1) as usize;
                    if !a.exclusive_run(&live[i].run).unwrap() {
                        let have = a.list(&live[i].run).unwrap().len() as u64;
                        let to = (have * PAGE + rng.range(1, 3) * PAGE).min(live[i].worst);
                        let ga = a.grow(&live[i].run, to).unwrap();
                        assert_eq!(ga, b.grow(&live[i].run, to).unwrap());
                        if let Some(g) = ga {
                            live[i].run = g;
                        }
                    }
                }
                6..=7 if !live.is_empty() => {
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    let list_len = a.list(&l.run).unwrap().len();
                    let salt = step as i32;
                    let st = done(list_len, salt);
                    convs.push((st.history.clone(), salt));
                    a.retain(&l.run, st.clone()).unwrap();
                    b.retain(&l.run, st).unwrap();
                }
                8 if !live.is_empty() => {
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    a.release(&l.run).unwrap();
                    b.release(&l.run).unwrap();
                }
                _ => {}
            }
            assert_eq!(a.digest(), b.digest());
            check(&a, &live);
        }
    }
    assert!(grants > 10_000, "grants {grants}");
    assert!(splits_together > 500, "split runs admitted beside live split runs: {splits_together}");
    assert!(split_hits_live > 20, "split-run hits while others were live: {split_hits_live}");
}
