//! Properties of random traces: no page is held twice, no page is lost (live + retained + free = the pool), a live
//! run's list never changes until it leaves, runs below R stay below place, an exclusive run is pages 0.., and two
//! ranks given the same calls grant the same runs and agree on the digest. The bit-for-bit match with the C++ pool is
//! engine/runtime/s2/tests/pool_diff_test.cpp.

use g6kvman::{Finished, Pool, Rules, Run, PAGE};

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
    fn one_in(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }
}

struct Live {
    run: Run,
    list: Vec<u32>,
    prompt: Vec<i32>,
}

// a finished session: prompt prefilled, `gen` tokens decoded (as Service::retire_run hands it over)
fn finished(rng: &mut Rng, prompt: &[i32]) -> Finished {
    let gen = rng.range(0, 30) as usize;
    let mut history = prompt.to_vec();
    history.extend((0..gen).map(|_| rng.range(0, 20) as i32));
    let written_end = (prompt.len() + gen) as i32;
    let valid = if rng.one_in(10) { 0 } else { written_end.min(history.len() as i32 - 1) };
    Finished { history, valid, written_end, pf_end: prompt.len() as i32 - 1 }
}

fn check(p: &Pool, live: &[Live]) {
    let pages = p.pool_pages() as usize;
    let mut owner = vec![0u8; pages]; // 1 live, 2 retained
    let mut live_pages = 0;
    for l in live {
        assert_eq!(p.list(&l.run).unwrap(), &l.list[..], "a live run's list changed");
        let ex = p.exclusive_run(&l.run).unwrap();
        for (i, &q) in l.list.iter().enumerate() {
            assert!(owner[q as usize] == 0, "page {q} held twice");
            assert!(p.used(q));
            owner[q as usize] = 1;
            if ex {
                assert_eq!(q as usize, i, "an exclusive run is pages 0..");
            } else {
                assert!(q < p.place_pages(), "a run below R past place");
            }
        }
        live_pages += l.list.len();
    }
    assert_eq!(live_pages, p.live_pages() as usize);
    for k in p.kept() {
        for &q in &k.list {
            assert!(owner[q as usize] == 0, "retained page {q} also live or retained twice");
            owner[q as usize] = 2;
        }
    }
    let free = owner.iter().filter(|&&o| o == 0).count();
    let retained = owner.iter().filter(|&&o| o == 2).count();
    assert_eq!(live_pages + retained + free, pages);
    assert!((0..pages as u32).all(|q| p.used(q) == (owner[q as usize] == 1)));
}

#[test]
fn random_traces() {
    let (mut grants, mut hits, mut evicted) = (0, 0, 0);
    for seed in 1..=100u64 {
        let mut rng = Rng(seed * 7919);
        let pages = rng.range(8, 600) as u32;
        let place = if rng.one_in(3) { pages } else { rng.range(1, pages as u64) as u32 };
        let below = 64 * rng.range(1, place as u64) as u32;
        let (mut a, mut b) = (Pool::new(pages, place, below).unwrap(), Pool::new(pages, place, below).unwrap());
        let rules =
            Rules { strict: rng.one_in(5), noend: !rng.one_in(3), grid: [0, 64, 128, 1000][rng.range(0, 3) as usize] };
        a.keep(rules);
        b.keep(rules);
        let mut live: Vec<Live> = Vec::new();
        let mut past: Vec<Vec<i32>> = Vec::new();
        for _ in 0..2000 {
            match rng.range(0, 9) {
                0..=4 => {
                    let mut prompt = if !past.is_empty() && !rng.one_in(3) {
                        past[rng.range(0, past.len() as u64 - 1) as usize].clone()
                    } else {
                        Vec::new()
                    };
                    let extra = rng.range(1, (pages as u64 * PAGE / 4).max(1));
                    prompt.extend((0..extra).map(|_| rng.range(0, 20) as i32));
                    let rows = prompt.len() as u64 + rng.range(1, 80);
                    if !a.admissible(rows) || rows.div_ceil(PAGE) > a.room() as u64 {
                        assert!(a.reserve(rows).is_err());
                        check(&a, &live);
                        continue;
                    }
                    let kept = a.kept_runs();
                    let h = a.match_prefix(&prompt, rows, prompt.len() as i32 - 1);
                    assert_eq!(h, b.match_prefix(&prompt, rows, prompt.len() as i32 - 1));
                    let (ra, rb) = if h.skip > 0 {
                        hits += 1;
                        (a.take(h.skip, h.age, rows).unwrap(), b.take(h.skip, h.age, rows).unwrap())
                    } else {
                        (a.reserve(rows).unwrap(), b.reserve(rows).unwrap())
                    };
                    assert_eq!(ra, rb);
                    grants += 1;
                    evicted += kept - (h.skip > 0) as u32 - a.kept_runs();
                    live.push(Live { run: ra, list: a.list(&ra).unwrap().to_vec(), prompt });
                }
                5..=7 if !live.is_empty() => {
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    let st = finished(&mut rng, &l.prompt);
                    if st.valid > 0 {
                        past.push(st.history.clone());
                    }
                    a.retain(&l.run, st.clone()).unwrap();
                    b.retain(&l.run, st).unwrap();
                    assert!(a.list(&l.run).is_err());
                }
                8 if !live.is_empty() => {
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    a.release(&l.run).unwrap();
                    b.release(&l.run).unwrap();
                    assert!(a.release(&l.run).is_err(), "a run released twice");
                }
                _ => {}
            }
            assert_eq!(a.digest(), b.digest());
            check(&a, &live);
        }
        // a rank that makes one more call than the others has another digest
        if a.room() > 0 && a.admissible(PAGE) {
            a.reserve(PAGE).unwrap();
            assert_ne!(a.digest(), b.digest());
        }
    }
    assert!(grants > 5_000 && hits > 250 && evicted > 500, "grants {grants} hits {hits} evicted {evicted}");
}

#[test]
fn lowest_free_first_and_exclusive() {
    let mut p = Pool::new(100, 60, 60 * 64).unwrap();
    let runs: Vec<Run> = (0..6).map(|_| p.reserve(10 * PAGE).unwrap()).collect();
    assert_eq!(p.room(), 0);
    for r in runs.iter().step_by(2) {
        p.release(r).unwrap();
    }
    let big = p.reserve(25 * PAGE).unwrap();
    let want: Vec<u32> = (0..10).chain(20..30).chain(40..45).collect();
    assert_eq!(p.list(&big).unwrap(), &want[..]);
    for r in runs.iter().skip(1).step_by(2).chain([&big]) {
        p.release(r).unwrap();
    }
    assert_eq!(p.free_pages(), 100);
    let x = p.reserve(90 * PAGE).unwrap();
    assert!(p.exclusive_run(&x).unwrap() && p.list(&x).unwrap() == (0..90).collect::<Vec<u32>>());
    assert_eq!((p.room(), p.free_pages(), p.admissible(PAGE)), (0, 0, false));
}
