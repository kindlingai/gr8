//! A pool that shrinks and grows at runtime (#153): random traces of admissions, growth, retirement, prefix-reuse hits,
//! shrinks and grows. Across every resize no page is held twice or lost (live + retained + free + offline = the pool),
//! no live run or retained run holds an offline page, a shrink never touches a live run (its list stays, it is never
//! evicted) and is refused exactly when it would take a chunk a live run holds or leave the pool unsafe, the pool is
//! always safe (growth only from a safe state), no session deadlocks, and two ranks given the same calls agree on every
//! result and digest. With every chunk online the pool behaves as one that never had chunks.

use g6kvman::stash::{Io, Keep};
use g6kvman::{Error, Finished, Pool, Rules, Run, PAGE};

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

// Banker's safety the classic way, independent of Pool::safe
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

struct Live {
    run: Run,
    list: Vec<u32>,
    prompt: Vec<i32>,
}

fn check(p: &Pool, live: &[Live]) {
    let pages = p.pool_pages() as usize;
    let mut owner = vec![0u8; pages]; // 1 live, 2 retained, 3 offline
    for q in 0..p.place_pages() {
        if !p.online(q) {
            owner[q as usize] = 3;
        }
    }
    let offline = owner.iter().filter(|&&o| o == 3).count();
    assert_eq!(offline, p.offline_pages() as usize);
    assert_eq!(p.place_online() as usize, p.place_pages() as usize - offline);
    for l in live {
        assert_eq!(p.list(&l.run).unwrap(), &l.list[..], "a live run's list changed");
        for &q in &l.list {
            assert!(owner[q as usize] != 3, "a live run holds offline page {q}");
            assert!(owner[q as usize] == 0 && p.used(q), "page {q} held twice");
            owner[q as usize] = 1;
        }
    }
    let held: usize = live.iter().map(|l| l.list.len()).sum();
    assert_eq!(held, p.live_pages() as usize);
    for k in p.kept() {
        for &q in &k.list {
            assert!(owner[q as usize] != 3, "a retained run holds offline page {q}");
            assert!(owner[q as usize] == 0, "retained page {q} also live or retained twice");
            owner[q as usize] = 2;
        }
    }
    let free = owner.iter().filter(|&&o| o == 0).count();
    let retained = owner.iter().filter(|&&o| o == 2).count();
    assert_eq!(held + retained + free + offline, pages, "a page lost");
    assert!((0..pages as u32).all(|q| p.used(q) == (owner[q as usize] == 1)));
    // a resident past-R run on the tables as built (nr 0) holds the pool: nothing is free, nothing admits, and it
    // already holds its worst case
    if live.iter().any(|l| p.exclusive_run(&l.run).unwrap()) && !p.admissible(PAGE) {
        assert_eq!((p.free_pages(), p.room()), (0, 0));
        return;
    }
    assert_eq!(p.free_pages() as usize, pages - offline - held);
    // the Banker's model below place: with the page classes on a split run shares the pool, holding its worst case
    // already (need 0); only its pages below place count, its draft-class pages above place are its own class
    let place = p.place_pages();
    let mut low = 0u32;
    let runs: Vec<(u32, u32)> = live
        .iter()
        .map(|l| {
            let (_, h, w) = p.runs().find(|u| u.0 == l.run.table_version).unwrap();
            let lo = l.list.iter().filter(|&&q| q < place).count() as u32;
            low += lo;
            if p.exclusive_run(&l.run).unwrap() {
                (lo, lo)
            } else {
                (h, w)
            }
        })
        .collect();
    assert!(safe(p.place_online(), low, &runs), "pool left unsafe");
}

fn finish(rng: &mut Rng, prompt: &[i32]) -> Finished {
    let mut history = prompt.to_vec();
    history.extend((0..rng.range(0, 20)).map(|_| rng.range(0, 9) as i32));
    let n = history.len() as i32;
    Finished { valid: n - 1, written_end: n, pf_end: prompt.len() as i32 - 1, history }
}

// what a shrink of n must do, from the pool's visible state (independent of Pool::shrink): Err(BadSize) when no state
// allows it, Ok(None) when it must wait, Ok(Some(chunks)) otherwise
fn shrink_model(
    a: &Pool,
    n: u32,
    chunk: u32,
    place: u32,
    below_pages: u32,
    resident: bool,
) -> Result<Option<Vec<u32>>, Error> {
    let on: Vec<u32> = (0..a.chunk_count()).filter(|&k| a.chunk_online(k)).collect();
    let size = |k: u32| (k * chunk..((k + 1) * chunk).min(place)).len() as u32;
    let mut sizes: Vec<u32> = on.iter().map(|&k| size(k)).collect();
    sizes.sort_unstable_by(|x, y| y.cmp(x));
    if sizes.len() < n as usize || a.place_online() - sizes.iter().take(n as usize).sum::<u32>() < below_pages {
        return Err(Error::BadSize);
    }
    if resident || !a.admissible(PAGE) {
        return Ok(None); // a past-R run is resident
    }
    let mut free_chunks: Vec<u32> = on
        .iter()
        .rev()
        .copied()
        .filter(|&k| !(k * chunk..((k + 1) * chunk).min(place)).any(|q| a.used(q)))
        .take(n as usize)
        .collect();
    let left = a.place_online() - free_chunks.iter().map(|&k| size(k)).sum::<u32>();
    let runs: Vec<(u32, u32)> = a.runs().map(|(_, h, w)| (h, w)).collect();
    if free_chunks.len() < n as usize || !safe(left, a.live_pages(), &runs) {
        return Ok(None);
    }
    free_chunks.sort_unstable();
    Ok(Some(free_chunks))
}

#[test]
fn resize_traces() {
    let (mut shrinks, mut waits, mut grows, mut evicted, mut hits, mut stalls) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for seed in 1..=200u64 {
        let mut rng = Rng(seed * 6151);
        let place = rng.range(8, 300) as u32;
        let below_pages = rng.range(1, (place as u64 / 2).max(1)) as u32;
        let chunk = rng.range(1, (place as u64 / 3).max(1)) as u32;
        let mut a = Pool::new(place, place, below_pages * 64).unwrap();
        let mut b = Pool::new(place, place, below_pages * 64).unwrap();
        a.chunks(chunk).unwrap();
        b.chunks(chunk).unwrap();
        let rules = Rules { strict: false, noend: true, grid: [0, 64, 128][rng.range(0, 2) as usize] };
        if !rng.one_in(4) {
            a.keep(rules);
            b.keep(rules);
        }
        let mut live: Vec<Live> = Vec::new();
        let mut past: Vec<Vec<i32>> = Vec::new();
        for _ in 0..1500 {
            match rng.range(0, 11) {
                0..=2 => {
                    let mut prompt = if !past.is_empty() && rng.one_in(2) {
                        past[rng.range(0, past.len() as u64 - 1) as usize].clone()
                    } else {
                        Vec::new()
                    };
                    prompt.extend((0..rng.range(1, below_pages as u64 * 32)).map(|_| rng.range(0, 9) as i32));
                    let worst =
                        (prompt.len() as u64 + rng.range(1, below_pages as u64 * 32)).min(below_pages as u64 * 64);
                    if prompt.len() as u64 >= worst {
                        continue;
                    }
                    let rows = (prompt.len() as u64 + 5).min(worst);
                    let fits = a.fits(rows, worst);
                    assert_eq!(fits, b.fits(rows, worst));
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
                    live.push(Live { run: ra, list: a.list(&ra).unwrap().to_vec(), prompt });
                }
                3..=5 => {
                    for l in live.iter_mut() {
                        let (_, held, worst) = a.runs().find(|u| u.0 == l.run.table_version).unwrap();
                        if worst == held {
                            continue;
                        }
                        let more = rng.range(1, (worst - held) as u64) as u32;
                        let rows = (held + more) as u64 * PAGE - rng.range(0, 63);
                        let ga = a.grow(&l.run, rows).unwrap();
                        assert_eq!(ga, b.grow(&l.run, rows).unwrap());
                        match ga {
                            None => stalls += 1,
                            Some(r) => {
                                let list = a.list(&r).unwrap().to_vec();
                                assert!(list.starts_with(&l.list), "growth only appends");
                                l.run = r;
                                l.list = list;
                            }
                        }
                    }
                }
                6..=7 => {
                    if live.is_empty() {
                        continue;
                    }
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    let st = finish(&mut rng, &l.prompt);
                    if st.valid > 0 && a.keeping() {
                        past.push(st.history.clone());
                    }
                    a.retain(&l.run, st.clone()).unwrap();
                    b.retain(&l.run, st).unwrap();
                }
                // shrink: what the pool must do, from an independent model of its state
                8..=9 => {
                    let n = rng.range(1, 3) as u32;
                    let on: Vec<u32> = (0..a.chunk_count()).filter(|&k| a.chunk_online(k)).collect();
                    let size = |k: u32| (k * chunk..((k + 1) * chunk).min(place)).len() as u32;
                    let mut sizes: Vec<u32> = on.iter().map(|&k| size(k)).collect();
                    sizes.sort_unstable_by(|x, y| y.cmp(x));
                    let impossible = sizes.len() < n as usize
                        || a.place_online() - sizes.iter().take(n as usize).sum::<u32>() < below_pages;
                    let free_chunks: Vec<u32> = on
                        .iter()
                        .rev()
                        .copied()
                        .filter(|&k| !(k * chunk..((k + 1) * chunk).min(place)).any(|q| a.used(q)))
                        .take(n as usize)
                        .collect();
                    let left = a.place_online() - free_chunks.iter().map(|&k| size(k)).sum::<u32>();
                    let runs: Vec<(u32, u32)> = a.runs().map(|(_, h, w)| (h, w)).collect();
                    let can = free_chunks.len() == n as usize && safe(left, a.live_pages(), &runs);
                    let kept_before = a.kept_runs();
                    let before = a.digest();
                    let ra = a.shrink(n);
                    assert_eq!(ra, b.shrink(n));
                    match ra {
                        Err(e) => {
                            assert!(impossible, "shrink {n} refused as {e:?}");
                            assert_eq!(e, Error::BadSize);
                            assert_eq!(a.digest(), before, "a refused shrink changed the pool");
                        }
                        Ok(None) => {
                            assert!(!impossible && !can, "shrink {n} waited though it could go");
                            assert_eq!(a.digest(), before, "a waiting shrink changed the pool");
                            waits += 1;
                        }
                        Ok(Some(ks)) => {
                            assert!(!impossible && can, "shrink {n} went though it must wait");
                            let mut want = free_chunks.clone();
                            want.sort_unstable();
                            assert_eq!(ks, want, "a shrink takes the highest chunks free of live runs");
                            assert!(ks.iter().all(|&k| !a.chunk_online(k)));
                            evicted += (kept_before - a.kept_runs()) as u64;
                            shrinks += 1;
                        }
                    }
                }
                _ => {
                    let off = (0..a.chunk_count()).filter(|&k| !a.chunk_online(k)).count() as u32;
                    let n = rng.range(1, 3) as u32;
                    let ra = a.grow_pool(n);
                    assert_eq!(ra, b.grow_pool(n));
                    if n > off {
                        assert_eq!(ra, Err(Error::BadArgument));
                    } else {
                        let ks = ra.unwrap();
                        assert_eq!(ks.len(), n as usize);
                        grows += 1;
                    }
                }
            }
            assert_eq!(a.digest(), b.digest());
            check(&a, &live);
        }
    }
    assert!(
        shrinks > 5_000 && waits > 500 && grows > 5_000 && evicted > 500 && hits > 100 && stalls > 100,
        "shrinks {shrinks} waits {waits} grows {grows} evicted {evicted} hits {hits} stalls {stalls}"
    );
}

// sessions decode to their worst case a page at a time while the pool shrinks and grows under them: a shrink that has
// to wait waits, and every admitted session still finishes with its list only ever appended to
#[test]
fn sessions_finish_through_resizes() {
    for seed in 1..=300u64 {
        let mut rng = Rng(seed * 104729);
        let place = rng.range(8, 64) as u32;
        let chunk = rng.range(1, (place as u64 / 4).max(1)) as u32;
        let below = rng.range(1, (place as u64 / 2).max(1)) as u32;
        let mut p = Pool::new(place, place, below * 64).unwrap();
        p.chunks(chunk).unwrap();
        let mut queue: Vec<(u64, u64)> = (0..rng.range(1, 40))
            .map(|_| {
                let worst = rng.range(1, below as u64 * 64);
                (rng.range(1, worst), worst)
            })
            .collect();
        let mut live: Vec<(Run, Vec<u32>, u64, u64)> = Vec::new();
        let mut rounds = 0;
        while !queue.is_empty() || !live.is_empty() {
            if rng.one_in(3) {
                let _ = p.shrink(rng.range(1, 2) as u32);
            } else if rng.one_in(3) {
                let off = (0..p.chunk_count()).filter(|&k| !p.chunk_online(k)).count() as u32;
                if off > 0 {
                    p.grow_pool(rng.range(1, off as u64) as u32).unwrap();
                }
            }
            while let Some(&(rows, worst)) = queue.first() {
                if !p.fits(rows, worst) {
                    break;
                }
                let r = p.reserve(rows, worst).unwrap();
                live.push((r, p.list(&r).unwrap().to_vec(), rows, worst));
                queue.remove(0);
            }
            let mut progressed = false;
            for (run, list, rows, worst) in live.iter_mut() {
                assert_eq!(p.list(run).unwrap(), &list[..], "a resize changed a live run");
                let want = (*rows + PAGE).min(*worst);
                if let Some(r) = p.grow(run, want).unwrap() {
                    let l = p.list(&r).unwrap().to_vec();
                    assert!(l.starts_with(list));
                    *run = r;
                    *list = l;
                    *rows = want;
                    progressed = true;
                }
            }
            let (done, rest): (Vec<_>, Vec<_>) = live.into_iter().partition(|x| x.2 == x.3);
            for x in &done {
                p.release(&x.0).unwrap();
            }
            // with nothing live and a shrunken pool too small for the queue head, the operator grows it back
            if rest.is_empty() && done.is_empty() && !progressed {
                if let Some(&(rows, worst)) = queue.first() {
                    if !p.fits(rows, worst) {
                        let off = (0..p.chunk_count()).filter(|&k| !p.chunk_online(k)).count() as u32;
                        assert!(off > 0, "an empty full-size pool refused {rows}/{worst}");
                        p.grow_pool(off).unwrap();
                    }
                }
            } else {
                assert!(progressed || !done.is_empty(), "deadlock");
            }
            live = rest;
            rounds += 1;
            assert!(rounds < 100_000);
        }
        assert_eq!(p.live_pages(), 0);
    }
}

// the gate's scenario: live sessions hold low pages, retained runs sit in high chunks; a shrink evicts the retained
// runs and leaves the live runs, a shrink into a live chunk waits, and a grow gives the pages back
#[test]
fn shrink_evicts_retained_only_and_waits_for_live() {
    let mut p = Pool::new(40, 40, 10 * 64).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 0 });
    p.chunks(10).unwrap();
    let a = p.reserve(5 * PAGE, 10 * PAGE).unwrap(); // pages 0..5
    let b = p.reserve(5 * PAGE, 10 * PAGE).unwrap(); // 5..10
    let c = p.reserve(10 * PAGE, 10 * PAGE).unwrap(); // 10..20
    let d = p.reserve(10 * PAGE, 10 * PAGE).unwrap(); // 20..30
    let hist: Vec<i32> = (0..10 * 64 - 1).collect();
    let st = Finished { history: hist.clone(), valid: 10 * 64 - 2, written_end: 10 * 64 - 1, pf_end: 10 * 64 - 2 };
    p.retain(&d, st.clone()).unwrap();
    let (la, lb, lc) = (p.list(&a).unwrap().to_vec(), p.list(&b).unwrap().to_vec(), p.list(&c).unwrap().to_vec());
    // chunks 3 (free) and 2 (retained): both go, the retained run with them
    assert_eq!(p.shrink(2), Ok(Some(vec![2, 3])));
    assert_eq!(p.kept_runs(), 0);
    assert_eq!((p.place_online(), p.free_pages()), (20, 0));
    assert_eq!((p.list(&a).unwrap(), p.list(&b).unwrap(), p.list(&c).unwrap()), (&la[..], &lb[..], &lc[..]));
    // 20 pages online, 20 held: a and b may each grow by 5 more, c holds its worst case and gives 10 back when it ends
    assert!(p.safe(None, None));
    assert_eq!(p.shrink(1), Ok(None), "every online chunk holds a live run");
    assert_eq!(p.grow(&a, 6 * PAGE).unwrap(), None, "no free page: a waits for c");
    p.release(&c).unwrap();
    // chunk 1 is free now, but taking it leaves 10 pages for a and b that could grow to 20: unsafe, so it waits
    assert_eq!(p.shrink(1), Ok(None), "a shrink that leaves a live run unable to finish waits");
    // with b gone, chunk 0 alone holds a's worst case
    p.release(&b).unwrap();
    assert_eq!(p.shrink(1), Ok(Some(vec![1])));
    assert_eq!(p.place_online(), 10);
    // R is 10 pages: no shrink may leave fewer online
    assert_eq!(p.shrink(1), Err(Error::BadSize));
    // the grow brings the lowest chunks back first, and a grows into them in order
    assert_eq!(p.grow_pool(2), Ok(vec![1, 2]));
    let g = p.grow(&a, 10 * PAGE).unwrap().unwrap();
    assert_eq!(p.list(&g).unwrap(), &(0..10).collect::<Vec<u32>>()[..]);
    assert_eq!(p.grow_pool(2), Err(Error::BadArgument), "only one chunk is offline");
    assert_eq!(p.grow_pool(1), Ok(vec![3]));
    assert_eq!(p.free_pages(), 30);
}

// an exclusive run on the tables as built takes pages 0, 1, ..: offline chunks cut that range, so a shrink takes the
// highest chunks and a grow brings back the lowest; a shrink waits while an exclusive run is resident
#[test]
fn exclusive_runs_and_resize() {
    let mut p = Pool::new(40, 40, 8 * 64).unwrap();
    p.chunks(10).unwrap();
    assert_eq!(p.shrink(2), Ok(Some(vec![2, 3])));
    assert_eq!(p.room(), 20);
    assert!(!p.fits(21 * PAGE, 21 * PAGE) && p.fits(20 * PAGE, 20 * PAGE));
    let x = p.reserve(20 * PAGE, 20 * PAGE).unwrap();
    assert_eq!(p.list(&x).unwrap(), &(0..20).collect::<Vec<u32>>()[..]);
    assert_eq!(p.shrink(1), Ok(None), "an exclusive run is resident");
    assert_eq!(p.grow_pool(1), Ok(vec![2]));
    p.release(&x).unwrap();
    assert_eq!(p.room(), 30);
    let y = p.reserve(30 * PAGE, 30 * PAGE).unwrap();
    assert_eq!(p.list(&y).unwrap(), &(0..30).collect::<Vec<u32>>()[..]);
}

// with every chunk online, a pool with chunks gives the same runs and digest as one without
#[test]
fn chunks_online_change_nothing() {
    for seed in 1..=50u64 {
        let mut rng = Rng(seed * 31337);
        let place = rng.range(8, 200) as u32;
        let below = rng.range(1, place as u64) as u32;
        let mut a = Pool::new(place, place, below * 64).unwrap();
        let mut b = Pool::new(place, place, below * 64).unwrap();
        b.chunks(rng.range(1, place as u64) as u32).unwrap();
        a.keep(Rules::default());
        b.keep(Rules::default());
        let mut live: Vec<Run> = Vec::new();
        for _ in 0..500 {
            if rng.one_in(2) || live.is_empty() {
                let worst = rng.range(1, place as u64 * 64);
                let rows = rng.range(1, worst);
                assert_eq!(a.fits(rows, worst), b.fits(rows, worst));
                assert_eq!(a.room(), b.room());
                if a.fits(rows, worst) {
                    let r = a.reserve(rows, worst).unwrap();
                    assert_eq!(r, b.reserve(rows, worst).unwrap());
                    live.push(r);
                }
            } else {
                let r = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                let st = Finished { history: vec![1, 2, 3], valid: 2, written_end: 3, pf_end: 2 };
                a.retain(&r, st.clone()).unwrap();
                b.retain(&r, st).unwrap();
            }
            assert_eq!(a.digest(), b.digest());
        }
    }
}

// the served configuration: the CP map's page classes on (tail(nr)), the drafter's class above place, and requests past
// R that take their own pages (prefix and shard below place, the drafter's tail above it). Across shrinks and grows no
// page is lost or held twice, no run holds an offline page, a past-R run keeps its prefix and shard below place, a
// shrink waits while a past-R run is resident, and two ranks agree.
#[test]
fn resize_traces_two_classes() {
    let (mut shrinks, mut grows, mut exclusive, mut hits, mut stalls_or_grows) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for seed in 1..=150u64 {
        let mut rng = Rng(seed * 92821);
        let place = rng.range(16, 300) as u32;
        let pages = place + rng.range(1, place as u64 * 2) as u32;
        let below_pages = rng.range(1, (place as u64 / 3).max(1)) as u32;
        let chunk = rng.range(1, (place as u64 / 4).max(1)) as u32;
        let nr = rng.range(2, 6) as u32;
        let mut a = Pool::new(pages, place, below_pages * 64).unwrap();
        let mut b = Pool::new(pages, place, below_pages * 64).unwrap();
        for p in [&mut a, &mut b] {
            p.tail(nr, 0);
            p.chunks(chunk).unwrap();
            p.keep(Rules { strict: false, noend: true, grid: 0 });
        }
        let mut live: Vec<Live> = Vec::new();
        let mut past: Vec<Vec<i32>> = Vec::new();
        for _ in 0..1200 {
            match rng.range(0, 10) {
                0..=3 => {
                    let mut prompt = if !past.is_empty() && rng.one_in(2) {
                        past[rng.range(0, past.len() as u64 - 1) as usize].clone()
                    } else {
                        Vec::new()
                    };
                    // past R (an exclusive run) one in two, the rest below it
                    let long = rng.one_in(2);
                    let most = if long { pages as u64 * 64 } else { below_pages as u64 * 64 };
                    prompt.extend((0..rng.range(1, most / 2)).map(|_| rng.range(0, 9) as i32));
                    let rows = prompt.len() as u64 + rng.range(1, 64);
                    if rows > most {
                        continue;
                    }
                    // below R a run may grow to its worst case; past R it takes it at once
                    let worst = if long { rows } else { (rows + rng.range(0, most / 2)).min(most) };
                    let fits = a.fits(rows, worst);
                    assert_eq!(fits, b.fits(rows, worst));
                    if !fits {
                        assert!(a.reserve(rows, worst).is_err());
                        continue;
                    }
                    let h = a.match_prefix(&prompt, rows, worst, prompt.len() as i32 - 1);
                    assert_eq!(h, b.match_prefix(&prompt, rows, worst, prompt.len() as i32 - 1));
                    let (ra, rb) = if h.skip > 0 {
                        hits += 1;
                        (a.take(h.skip, h.age, rows, worst).unwrap(), b.take(h.skip, h.age, rows, worst).unwrap())
                    } else {
                        (a.reserve(rows, worst).unwrap(), b.reserve(rows, worst).unwrap())
                    };
                    assert_eq!(ra, rb);
                    if a.exclusive_run(&ra).unwrap() {
                        exclusive += 1;
                        // its prefix and shard below place, all online
                        let l = a.list(&ra).unwrap();
                        let rp = below_pages as u64;
                        let n = l.len() as u64;
                        let main = rp + (n - rp).div_ceil(nr as u64).min(n - rp);
                        assert!(l.iter().filter(|&&q| q < place).count() as u64 >= main);
                        assert!(l.iter().take(rp as usize).all(|&q| q < place));
                    }
                    live.push(Live { run: ra, list: a.list(&ra).unwrap().to_vec(), prompt });
                }
                4..=5 => {
                    if live.is_empty() {
                        continue;
                    }
                    let l = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                    let st = finish(&mut rng, &l.prompt);
                    if st.valid > 0 {
                        past.push(st.history.clone());
                    }
                    a.retain(&l.run, st.clone()).unwrap();
                    b.retain(&l.run, st).unwrap();
                }
                6..=7 => {
                    let n = rng.range(1, 2) as u32;
                    let resident = live.iter().any(|l| a.exclusive_run(&l.run).unwrap());
                    let want = shrink_model(&a, n, chunk, place, below_pages, resident);
                    let before = a.digest();
                    let r = a.shrink(n);
                    assert_eq!(r, b.shrink(n));
                    assert_eq!(r, want, "shrink {n} against the model");
                    match r {
                        Ok(Some(_)) => {
                            assert!(!resident, "a shrink went while a past-R run was resident");
                            shrinks += 1;
                        }
                        _ => assert_eq!(a.digest(), before, "a shrink that did not go changed the pool"),
                    }
                }
                8 => {
                    // a round of growth for the runs below R
                    for l in live.iter_mut() {
                        let (_, held, worst) = a.runs().find(|u| u.0 == l.run.table_version).unwrap();
                        if worst == held || a.exclusive_run(&l.run).unwrap() {
                            continue;
                        }
                        let rows = (held + rng.range(1, (worst - held) as u64) as u32) as u64 * PAGE;
                        let ga = a.grow(&l.run, rows).unwrap();
                        assert_eq!(ga, b.grow(&l.run, rows).unwrap());
                        if let Some(r) = ga {
                            let list = a.list(&r).unwrap().to_vec();
                            assert!(list.starts_with(&l.list) && list.iter().all(|&q| q < place && a.online(q)));
                            l.run = r;
                            l.list = list;
                            stalls_or_grows += 1;
                        }
                    }
                }
                _ => {
                    let off = (0..a.chunk_count()).filter(|&k| !a.chunk_online(k)).count() as u64;
                    if off > 0 {
                        let n = rng.range(1, off) as u32;
                        assert_eq!(a.grow_pool(n), b.grow_pool(n));
                        grows += 1;
                    }
                }
            }
            assert_eq!(a.digest(), b.digest());
            check(&a, &live);
            // the drafter's class never goes offline
            assert!((place..pages).all(|q| a.online(q)));
        }
    }
    assert!(
        shrinks > 2_000 && grows > 1_000 && exclusive > 800 && hits > 100 && stalls_or_grows > 1_000,
        "shrinks {shrinks} grows {grows} exclusive {exclusive} hits {hits} run growths {stalls_or_grows}"
    );
}

// with block stores: a shrink that evicts retained runs queues their spills, and the next pool call that changes
// state (here shrink and grow_pool, as for reserve, take and grow: Pool::pending) is refused until the engine ran them
// (io(), io_done()), so the memory goes only after the copies. The spilled run then restores from the store
#[test]
fn a_shrink_spills_before_the_memory_goes() {
    let mut p = Pool::new(40, 40, 10 * 64).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 0 });
    p.stores(1);
    p.add_tier(64, Keep::All, None).unwrap();
    p.chunks(10).unwrap();
    let hist: Vec<i32> = (0..10 * 64 - 1).collect();
    let st = Finished { history: hist.clone(), valid: 10 * 64 - 2, written_end: 10 * 64 - 1, pf_end: 10 * 64 - 2 };
    let a = p.reserve(10 * PAGE, 10 * PAGE).unwrap();
    let b = p.reserve(10 * PAGE, 10 * PAGE).unwrap();
    let c = p.reserve(10 * PAGE, 10 * PAGE).unwrap();
    let d = p.reserve(10 * PAGE, 10 * PAGE).unwrap(); // pages 30..40
    p.retain(&d, st).unwrap();
    let _ = p.io();
    p.io_done().unwrap();
    assert_eq!(p.shrink(1), Ok(Some(vec![3])));
    assert_eq!(p.kept_runs(), 0);
    assert!(p.io_pending() > 0, "the evicted run's spills are queued");
    assert_eq!(p.grow_pool(1), Err(Error::PendingIo), "nothing goes on before the spills ran");
    assert_eq!(p.shrink(1), Err(Error::PendingIo));
    assert_eq!(p.reserve(PAGE, PAGE), Err(Error::PendingIo));
    let io = p.io();
    assert!(io.iter().all(|x| matches!(x, Io::Spill { page, .. } if (30..40).contains(page))), "{io:?}");
    p.io_done().unwrap();
    for r in [&a, &b, &c] {
        p.release(r).unwrap();
    }
    // the run spilled from the offline chunk comes back from the store into online pages: 11 pages is past R (10), so
    // this is an exclusive grant on the tables as built, into the 30 pages before the offline chunk (prefix_online)
    let h = p.match_prefix(&hist, 11 * PAGE, 11 * PAGE, hist.len() as i32 - 1);
    assert!(h.stored && h.skip > 0, "{h:?}");
    let r = p.take_hit(h.skip, h.age, Some(h.tier), 11 * PAGE, 11 * PAGE).unwrap();
    assert!(p.list(&r).unwrap().iter().all(|&q| q < 30 && p.online(q)));
    let io = p.io();
    assert!(io.iter().any(|x| matches!(x, Io::Restore { .. })));
    p.io_done().unwrap();
}
