//! Runs that grow on demand (Pool::grow) under the safe-state check: random traces of admissions, growth, retirement
//! and prefix-reuse hits never hold a page twice or lose one, only extend a live run's list, refuse growth exactly when
//! granting it would leave some live run unable to reach its worst case, and never deadlock: some run can always go on. Two ranks given the same calls agree on every result and digest.

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
}

// Banker's safety the classic way (any run whose need fits finishes, repeat), independent of Pool::safe's sorted pass
fn safe(place: u32, live: u32, runs: &[(u32, u32)]) -> bool {
    let mut free = place as i64 - live as i64;
    if free < 0 {
        return false;
    }
    let mut left: Vec<(u32, u32)> = runs.to_vec();
    while !left.is_empty() {
        let Some(i) = left.iter().position(|&(held, worst)| (worst - held) as i64 <= free) else { return false };
        free += left[i].0 as i64;
        left.remove(i);
    }
    true
}

fn uses(p: &Pool) -> Vec<(u32, u32, u32)> {
    p.runs().collect()
}

struct Live {
    run: Run,
    list: Vec<u32>,
    prompt: Vec<i32>,
}

fn check(p: &Pool, live: &[Live]) {
    let mut owner = vec![0u8; p.pool_pages() as usize];
    for l in live {
        assert_eq!(p.list(&l.run).unwrap(), &l.list[..], "a live run's list changed");
        for &q in &l.list {
            assert!(owner[q as usize] == 0 && p.used(q) && q < p.place_pages(), "page {q} held twice or outside");
            owner[q as usize] = 1;
        }
    }
    for k in p.kept() {
        for &q in &k.list {
            assert!(owner[q as usize] == 0, "retained page {q} also live or retained twice");
            owner[q as usize] = 2;
        }
    }
    let held: usize = live.iter().map(|l| l.list.len()).sum();
    assert_eq!(held, p.live_pages() as usize);
    assert!((0..p.pool_pages()).all(|q| p.used(q) == (owner[q as usize] == 1)), "a page lost or double counted");
    let s = p.stats();
    assert_eq!((s.live_runs as usize, s.held_pages as usize), (live.len(), held));
    // the invariant growth keeps: the pool stays safe
    let runs: Vec<(u32, u32)> = uses(p).iter().map(|&(_, h, w)| (h, w)).collect();
    assert!(safe(p.place_pages(), p.live_pages(), &runs), "pool left unsafe");
}

#[test]
fn growth_traces() {
    let (mut grants, mut grows, mut stalls, mut hits, mut passes) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for seed in 1..=150u64 {
        let mut rng = Rng(seed * 104729);
        let place = rng.range(6, 300) as u32;
        let mut a = Pool::new(place, place, place * 64).unwrap();
        let mut b = Pool::new(place, place, place * 64).unwrap();
        let rules = Rules { strict: false, noend: true, grid: [0, 64, 128][rng.range(0, 2) as usize] };
        if rng.range(0, 3) > 0 {
            a.keep(rules);
            b.keep(rules);
        }
        let mut live: Vec<Live> = Vec::new();
        let mut past: Vec<Vec<i32>> = Vec::new();
        for _ in 0..1500 {
            match rng.range(0, 9) {
                // admit: pages for the prompt now, the worst case later
                0..=2 => {
                    let mut prompt = if !past.is_empty() && rng.range(0, 1) == 0 {
                        past[rng.range(0, past.len() as u64 - 1) as usize].clone()
                    } else {
                        Vec::new()
                    };
                    prompt.extend((0..rng.range(1, place as u64 * 16)).map(|_| rng.range(0, 9) as i32));
                    let worst = (prompt.len() as u64 + rng.range(1, place as u64 * 32)).min(place as u64 * 64);
                    let rows = (prompt.len() as u64 + 5).min(worst);
                    let fits = a.fits(rows, worst);
                    assert_eq!(fits, b.fits(rows, worst));
                    // fits is admissible + room + the safe state with the newcomer
                    let mut runs: Vec<(u32, u32)> = uses(&a).iter().map(|&(_, h, w)| (h, w)).collect();
                    let n = rows.div_ceil(PAGE) as u32;
                    runs.push((n, worst.div_ceil(PAGE) as u32));
                    let want = n <= a.room() && safe(place, a.live_pages() + n, &runs);
                    assert_eq!(fits, want, "fits disagrees with the safe state");
                    if !fits {
                        assert!(a.reserve(rows, worst).is_err());
                        continue;
                    }
                    let limit = prompt.len() as i32 - 1;
                    let h = a.match_prefix(&prompt, rows, worst, limit);
                    assert_eq!(h, b.match_prefix(&prompt, rows, worst, limit));
                    let (ra, rb) = if h.skip > 0 {
                        hits += 1;
                        (a.take(h.skip, h.age, rows, worst).unwrap(), b.take(h.skip, h.age, rows, worst).unwrap())
                    } else {
                        (a.reserve(rows, worst).unwrap(), b.reserve(rows, worst).unwrap())
                    };
                    assert_eq!(ra, rb);
                    grants += 1;
                    live.push(Live { run: ra, list: a.list(&ra).unwrap().to_vec(), prompt });
                }
                // a round: every live run asks for some of what it may still need, in order
                3..=7 => {
                    let before = uses(&a);
                    let wanting = before.iter().filter(|&&(_, h, w)| w > h).count();
                    let mut granted = 0;
                    for l in live.iter_mut() {
                        let (_, held, worst) = *uses(&a).iter().find(|u| u.0 == l.run.table_version).unwrap();
                        if worst == held {
                            continue;
                        }
                        let more = rng.range(1, (worst - held) as u64) as u32;
                        let rows = (held + more) as u64 * PAGE - rng.range(0, 63);
                        // would the growth leave the pool safe, by the independent check?
                        let runs: Vec<(u32, u32)> = uses(&a)
                            .iter()
                            .map(|&(v, h, w)| if v == l.run.table_version { (h + more, w) } else { (h, w) })
                            .collect();
                        let ok = more <= place - a.live_pages() && safe(place, a.live_pages() + more, &runs);
                        let ga = a.grow(&l.run, rows).unwrap();
                        assert_eq!(ga, b.grow(&l.run, rows).unwrap());
                        assert_eq!(ga.is_some(), ok, "a stall exactly when the growth is unsafe");
                        match ga {
                            None => stalls += 1,
                            Some(r) => {
                                let list = a.list(&r).unwrap().to_vec();
                                assert!(
                                    list.len() == (held + more) as usize && list.starts_with(&l.list),
                                    "growth only appends"
                                );
                                assert_eq!((r.table_version, r.first_page), (l.run.table_version, l.run.first_page));
                                l.run = r;
                                l.list = list;
                                grows += 1;
                                granted += 1;
                            }
                        }
                    }
                    // no deadlock: from a safe state either some run holds its worst case already (it runs on and
                    // finishes, giving its pages back) or some run that wants pages gets them
                    passes += 1;
                    let done = before.iter().any(|&(_, h, w)| h == w);
                    assert!(wanting == 0 || granted > 0 || done, "no run can go on: {before:?} place {place}");
                }
                // a session ends: retained (prefix reuse) or released
                _ => {
                    if live.is_empty() {
                        continue;
                    }
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    let mut history = l.prompt.clone();
                    history.extend((0..rng.range(0, 20)).map(|_| rng.range(0, 9) as i32));
                    let n = history.len() as i32;
                    let st = Finished { valid: n - 1, written_end: n, pf_end: l.prompt.len() as i32 - 1, history };
                    if st.valid > 0 && a.keeping() {
                        past.push(st.history.clone());
                    }
                    a.retain(&l.run, st.clone()).unwrap();
                    b.retain(&l.run, st).unwrap();
                }
            }
            assert_eq!(a.digest(), b.digest());
            check(&a, &live);
        }
    }
    assert!(
        grants > 10_000 && grows > 10_000 && stalls > 1_000 && hits > 100 && passes > 10_000,
        "grants {grants} grows {grows} stalls {stalls} hits {hits}"
    );
}

// sessions decode to their worst case a page at a time, finishing as they reach it: with admission and growth under
// the safe-state check every admitted session finishes, whatever the arrival order
#[test]
fn every_session_finishes() {
    for seed in 1..=300u64 {
        let mut rng = Rng(seed * 7919);
        let place = rng.range(4, 64) as u32;
        let mut p = Pool::new(place, place, place * 64).unwrap();
        let mut queue: Vec<(u64, u64)> = (0..rng.range(1, 40))
            .map(|_| {
                let worst = rng.range(1, place as u64 * 64);
                (rng.range(1, worst), worst)
            })
            .collect();
        let mut live: Vec<(Run, u64, u64)> = Vec::new(); // run, rows now, worst
        let mut rounds = 0;
        while !queue.is_empty() || !live.is_empty() {
            while let Some(&(rows, worst)) = queue.first() {
                if !p.fits(rows, worst) {
                    break;
                }
                live.push((p.reserve(rows, worst).unwrap(), rows, worst));
                queue.remove(0);
            }
            assert!(!live.is_empty(), "nothing admitted into an empty pool");
            let mut progressed = false;
            for (run, rows, worst) in live.iter_mut() {
                let want = (*rows + PAGE).min(*worst);
                if let Some(r) = p.grow(run, want).unwrap() {
                    *run = r;
                    *rows = want;
                    progressed = true;
                }
            }
            let (done, rest): (Vec<_>, Vec<_>) = live.into_iter().partition(|&(_, rows, worst)| rows == worst);
            for (run, _, _) in &done {
                p.release(run).unwrap();
            }
            assert!(progressed || !done.is_empty(), "deadlock");
            live = rest;
            rounds += 1;
            assert!(rounds < 100_000);
        }
        assert_eq!(p.live_pages(), 0);
    }
}
