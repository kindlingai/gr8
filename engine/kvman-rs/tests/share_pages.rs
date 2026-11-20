//! GR8_KV_SHARE_PAGES (c105): refcounted prompt pages below R. Live and retained runs share the full pages of a common
//! prefix; a hit that ends inside a page gets a fresh copy of that page (PageCopy) for its rows below the hit. A model of
//! the device pages (each row is the token it holds) checks that a grant's rows below its skip hold its prompt, that no
//! run ever writes a page another holder has, and that two ranks given the same calls agree on runs and the digest.

use g6kvman::pool::{PageCopy, LIVE_HIT};
use g6kvman::stash::{Io, Keep};
use g6kvman::{Error, Finished, Pool, Rules, Run, PAGE};

const ROWS: usize = PAGE as usize;
const R: u32 = 32768;

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

fn pool(pages: u32, share: bool) -> Pool {
    let mut p = Pool::new(pages, pages, R.min(pages * PAGE as u32)).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 0 });
    p.share_pages(share).unwrap();
    p
}

fn toks(seed: i32, n: usize) -> Vec<i32> {
    (0..n).map(|i| seed * 1_000_003 + i as i32).collect()
}

// admission as the engine does it: count, match, then take the hit or reserve
fn admit(p: &mut Pool, prompt: &[i32], rows: u64, worst: u64) -> (Result<Run, Error>, i32, u64) {
    p.count(prompt, None);
    let h = p.match_prefix(prompt, rows, worst, prompt.len() as i32 - 1);
    if h.skip > 0 {
        (p.take(h.skip, h.age, rows, worst), h.skip, h.age)
    } else {
        (p.reserve(rows, worst), 0, 0)
    }
}

fn live_pages(p: &Pool) -> u32 {
    (0..p.pool_pages()).filter(|&q| p.live_ref(q) > 0).count() as u32
}

// T1: 8 chats, one 30,000-token system prompt + 512 rows of their own each, place = 2,390 pages (gr8 int4 TP = 4)
#[test]
fn t1_eight_chats_on_one_30k_system_prompt_fit_in_place() {
    const PLACE: u32 = 2390;
    let system = toks(1, 30_000);
    let prompt = |c: i32| {
        let mut x = system.clone();
        x.extend(toks(100 + c, 200));
        x
    };
    let rows = 30_512u64; // 477 pages
    // stock: 5 * 477 = 2,385 pages, the 6th gets NoRoom
    let mut stock = pool(PLACE, false);
    let mut held = Vec::new();
    for c in 0..8 {
        let (r, skip, _) = admit(&mut stock, &prompt(c), rows, rows);
        assert_eq!(skip, 0, "stock: no live run is a hit");
        if c < 5 {
            held.push(r.unwrap());
        } else {
            assert_eq!(r, Err(Error::NoRoom), "stock chat {c}");
        }
    }
    assert_eq!(stock.live_pages(), 5 * 477);
    // shared: chat 0 prefills, then chats 1..7 share its 468 full pages and copy the 47 rows of page 468
    let mut sh = pool(PLACE, true);
    let (r0, _, _) = admit(&mut sh, &prompt(0), rows, rows);
    let r0 = r0.unwrap();
    sh.prefilled(&r0, 30_200).unwrap();
    let l0 = sh.list(&r0).unwrap().to_vec();
    let mut runs = vec![r0];
    for c in 1..8 {
        let (r, skip, age) = admit(&mut sh, &prompt(c), rows, rows);
        let r = r.unwrap();
        assert_eq!(skip, 29_999, "chat {c}: lcp 30,000, the last prompt row is prefilled");
        assert_ne!(age & LIVE_HIT, 0, "a hit on the live chat 0");
        let l = sh.list(&r).unwrap().to_vec();
        assert_eq!(l[..468], l0[..468], "chat {c} shares chat 0's full pages");
        assert!(!l0.contains(&l[468]), "the partial page is its own copy");
        assert_eq!(sh.take_copies(), vec![PageCopy { from: l0[468], to: l[468], rows: 29_999 % 64 }]);
        runs.push(r);
    }
    assert_eq!(live_pages(&sh), 468 + 8 * 9, "540 pages for 8 chats");
    assert_eq!(sh.live_pages(), 540);
    for q in &l0[..468] {
        assert_eq!(sh.live_ref(*q), 8);
    }
    // every chat retires: the pages stay counted once, then all come back
    for r in &runs {
        sh.release(r).unwrap();
    }
    assert_eq!(sh.live_pages(), 0);
    assert!((0..PLACE).all(|q| sh.live_ref(q) == 0 && !sh.used(q)));
}

// T2: a 30,001-row conversation forked 8 ways at row 30,000 = 468 * 64 + 48 (mid page), on 4 ranks
#[test]
fn t2_eight_forks_mid_page_share_468_pages_and_copy_48_rows() {
    let parent_prompt = toks(7, 30_001);
    let mut ranks: Vec<Pool> = (0..4).map(|_| pool(2390, true)).collect();
    let mut parent = Vec::new();
    for p in &mut ranks {
        let (r, _, _) = admit(p, &parent_prompt, 30_001, 30_001 + 128);
        let r = r.unwrap();
        p.prefilled(&r, 30_001).unwrap();
        parent.push(r);
    }
    let lp = ranks[0].list(&parent[0]).unwrap().to_vec();
    for f in 0..8 {
        let mut fork = parent_prompt.clone();
        fork.extend(toks(200 + f, 128));
        let mut got = Vec::new();
        for (i, p) in ranks.iter_mut().enumerate() {
            let (r, skip, _) = admit(p, &fork, fork.len() as u64, fork.len() as u64 + 128);
            let r = r.unwrap();
            assert_eq!(skip, 30_000);
            let l = p.list(&r).unwrap().to_vec();
            assert_eq!(l[..468], lp[..468], "fork {f} shares 468 pages zero-copy");
            let c = p.take_copies();
            assert_eq!(c, vec![PageCopy { from: lp[468], to: l[468], rows: 48 }], "fork {f}");
            assert_eq!(p.list(&parent[i]).unwrap()[468], lp[468], "the parent keeps its page 468");
            assert_eq!(p.live_ref(lp[468]), 1, "only the parent holds page 468");
            got.push((r, p.digest()));
        }
        assert!(got.windows(2).all(|w| w[0] == w[1]), "the 4 ranks agree: {got:?}");
    }
    assert_eq!(ranks[0].live_pages(), 469 + 8 * 3, "parent 469 + 8 forks * (the copy + 2 own)");
    // the parent retires: its pages stay for the forks, its retained run shares them too
    let d: Vec<u64> = ranks
        .iter_mut()
        .zip(&parent)
        .map(|(p, r)| {
            p.retain(r, Finished { history: parent_prompt.clone(), valid: 30_001, written_end: 30_001, pf_end: 30_000 })
                .unwrap();
            p.digest()
        })
        .collect();
    assert!(d.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(ranks[0].live_ref(lp[0]), 8);
    assert_eq!(ranks[0].kept_ref()[lp[0] as usize], 1);
}

#[test]
fn three_forks_hold_four_refs_and_survive_parent_release() {
    let mut p = pool(2048, true);
    let base = toks(7, 30_001);
    let (parent, _, _) = admit(&mut p, &base, 30_001, 30_129);
    let parent = parent.unwrap();
    p.prefilled(&parent, 30_001).unwrap();
    let parent_pages = p.list(&parent).unwrap().to_vec();
    let mut forks = Vec::new();
    for n in 0..3 {
        let mut prompt = base.clone();
        prompt.extend(toks(200 + n, 128));
        let (fork, skip, _) = admit(&mut p, &prompt, prompt.len() as u64, prompt.len() as u64 + 128);
        let fork = fork.unwrap();
        assert_eq!(skip, 30_000);
        let pages = p.list(&fork).unwrap().to_vec();
        assert_eq!(&pages[..468], &parent_pages[..468]);
        assert_eq!(p.take_copies(), vec![PageCopy { from: parent_pages[468], to: pages[468], rows: 48 }]);
        forks.push(fork);
    }
    for &page in &parent_pages[..468] { assert_eq!(p.live_ref(page), 4); }
    p.release(&parent).unwrap();
    for &page in &parent_pages[..468] { assert_eq!(p.live_ref(page), 3); }
    for fork in &forks { assert_eq!(&p.list(fork).unwrap()[..468], &parent_pages[..468]); }
    for fork in &forks { p.release(fork).unwrap(); }
    for &page in &parent_pages[..468] { assert_eq!(p.live_ref(page), 0); }
}

// a retained run hit by a shared prompt stays retained, and its retirement order is refreshed (the lease)
#[test]
fn a_shared_hit_keeps_the_retained_run_and_refreshes_it() {
    let mut p = pool(64, true);
    let a = toks(1, 300);
    let b = toks(2, 300);
    for x in [&a, &b] {
        let (r, _, _) = admit(&mut p, x, 300, 310);
        let r = r.unwrap();
        p.retain(&r, Finished { history: x.clone(), valid: 300, written_end: 300, pf_end: 299 }).unwrap();
    }
    let age_a = p.kept()[0].age;
    let mut q = a[..200].to_vec();
    q.extend(toks(3, 50));
    let (r, skip, age) = admit(&mut p, &q, 250, 260);
    let r = r.unwrap();
    assert_eq!((skip, age), (199, age_a));
    assert_eq!(p.kept_runs(), 2, "a stays retained");
    assert_eq!(p.kept().last().unwrap().st.history, a, "a is now the newest");
    assert!(p.kept().last().unwrap().age > age_a);
    let la = p.kept().last().unwrap().list.clone();
    assert_eq!(p.list(&r).unwrap()[..3], la[..3]);
    assert_eq!(p.take_copies(), vec![PageCopy { from: la[3], to: p.list(&r).unwrap()[3], rows: 199 % 64 }]);
}

// with stores: a retained run whose storable pages a newer retained run of its chat holds is let go, not written again
#[test]
fn a_covered_retained_run_is_not_spilled() {
    for share in [false, true] {
        let mut p = pool(16, share);
        p.stores(3);
        p.add_tier(64, Keep::All, None).unwrap();
        let mut t = toks(1, 300);
        let mut d = Vec::new();
        for c in 0..4 {
            let (r, _, _) = admit(&mut p, &t, t.len() as u64, t.len() as u64 + 20);
            let r = r.unwrap();
            run_io(&mut p);
            t.extend(toks(10 + c, 20));
            p.retain(&r, Finished { history: t.clone(), valid: t.len() as i32 - 1, written_end: t.len() as i32 - 1, pf_end: 0 })
                .unwrap();
            t.extend(toks(20 + c, 40));
            d.push(p.digest());
        }
        let spilled: usize = (0..p.kept_runs())
            .map(|_| {
                let (r, _, _) = admit(&mut p, &toks(99, 64 * 15), 64 * 15, 64 * 15);
                let n = run_io(&mut p).iter().filter(|x| matches!(x, Io::Spill { .. })).count();
                p.release(&r.unwrap()).unwrap();
                n
            })
            .sum();
        let e = p.stash().unwrap().entries(0).to_vec();
        assert_eq!(e.len(), 1, "share {share}: {e:?}");
        assert!(spilled <= 8, "share {share}: {spilled} blocks for 1 chat of 8 storable pages");
    }
}

fn run_io(p: &mut Pool) -> Vec<Io> {
    let io = p.io();
    p.io_done().unwrap();
    io
}

// ---- T3: seeded random traces -----------------------------------------------------------------------------------

struct Live {
    run: Run,
    tokens: Vec<i32>, // row i holds tokens[i] for i < written
    written: usize,
    worst: usize,
    prompt_len: usize,
}

struct Rank {
    pool: Pool,
    dev: Vec<[i32; ROWS]>,
}

impl Rank {
    fn copies(&mut self) {
        for c in self.pool.take_copies() {
            assert!(c.rows > 0 && c.rows <= PAGE as u32 && c.from != c.to);
            let src = self.dev[c.from as usize];
            self.dev[c.to as usize][..c.rows as usize].copy_from_slice(&src[..c.rows as usize]);
        }
    }
    // the run writes rows [from, to): only into pages it alone holds
    fn write(&mut self, l: &Live, from: usize, to: usize) {
        let list = self.pool.list(&l.run).unwrap().to_vec();
        let kept = self.pool.kept_ref();
        for row in from..to {
            let q = list[row / ROWS];
            assert_eq!(self.pool.live_ref(q), 1, "row {row}: page {q} has another live holder");
            assert_eq!(kept[q as usize], 0, "row {row}: page {q} is retained");
            self.dev[q as usize][row % ROWS] = l.tokens[row];
        }
    }
    fn check(&self, live: &[Live]) {
        let p = &self.pool;
        let mut lr = vec![0u32; p.pool_pages() as usize];
        for l in live {
            for &q in p.list(&l.run).unwrap() {
                lr[q as usize] += 1;
            }
        }
        let kr = p.kept_ref();
        let mut kr2 = vec![0u32; p.pool_pages() as usize];
        for k in p.kept() {
            for &q in &k.list {
                kr2[q as usize] += 1;
            }
        }
        for q in 0..p.pool_pages() {
            assert_eq!(p.live_ref(q), lr[q as usize], "live_ref of page {q}");
            assert_eq!(kr[q as usize], kr2[q as usize], "kept_ref of page {q}");
            assert_eq!(p.used(q), lr[q as usize] > 0, "used of page {q}");
        }
        assert_eq!(p.live_pages() as usize, lr.iter().filter(|&&x| x > 0).count());
        // the rows each live run wrote (or shares or copied) still hold its tokens
        for l in live {
            let list = p.list(&l.run).unwrap();
            for row in 0..l.written {
                assert_eq!(self.dev[list[row / ROWS] as usize][row % ROWS], l.tokens[row], "a live run's row {row}");
            }
        }
    }
}

// one random trace on two ranks; `drain_at`: stop after that many ops and prove every live run can still grow to its
// worst in some order (the Banker's guarantee), then release them all
fn trace(seed: u64, ops: usize, drain_at: Option<usize>) -> (u64, u64) {
    let mut rng = Rng(seed);
    let pages = rng.range(8, 32) as u32;
    let mut ranks: Vec<Rank> =
        (0..2).map(|_| Rank { pool: pool(pages, true), dev: vec![[-1; ROWS]; pages as usize] }).collect();
    let systems: Vec<Vec<i32>> = (0..3).map(|s| toks(s + 1, pages as usize * ROWS)).collect();
    let mut live: Vec<Live> = Vec::new();
    let (mut hits, mut copies) = (0u64, 0u64);
    let max = pages as usize * ROWS / 2;
    let n = drain_at.unwrap_or(ops);
    for op in 0..n {
        let k = rng.range(0, 9);
        if k < 4 || live.is_empty() {
            // a prompt: a system prefix of random length, maybe a live run's or a retained run's tokens, then its own
            let mut prompt: Vec<i32> = if !live.is_empty() && rng.one_in(3) {
                let l = &live[rng.range(0, live.len() as u64 - 1) as usize];
                l.tokens[..rng.range(1, l.written.max(1) as u64) as usize].to_vec()
            } else if !ranks[0].pool.kept().is_empty() && rng.one_in(3) {
                let ks = ranks[0].pool.kept();
                let h = &ks[rng.range(0, ks.len() as u64 - 1) as usize].st.history;
                h[..rng.range(1, h.len() as u64) as usize].to_vec()
            } else {
                let s = &systems[rng.range(0, 2) as usize];
                s[..rng.range(1, max as u64 - 1) as usize].to_vec()
            };
            let own = rng.range(1, 100) as usize;
            prompt.extend((0..own).map(|i| (seed as i32 & 0xffff) * 7 + op as i32 * 131 + i as i32 + 1_000_000_000));
            prompt.truncate(max);
            let worst = (prompt.len() + rng.range(0, 64) as usize).min(max);
            let mut got = Vec::new();
            for rk in &mut ranks {
                let (r, skip, _) = admit(&mut rk.pool, &prompt, prompt.len() as u64, worst as u64);
                let c = rk.pool.copies_pending() as u64;
                rk.copies();
                got.push((r, skip, c, rk.pool.digest()));
            }
            assert_eq!(got[0], got[1], "op {op}: the ranks agree");
            let (r, skip, c, _) = got[0];
            let Ok(run) = r else { continue };
            hits += (skip > 0) as u64;
            copies += c;
            let mut l = Live { run, tokens: prompt.clone(), written: skip.max(0) as usize, worst, prompt_len: prompt.len() };
            // rows below the skip hold the prompt already
            for rk in &ranks {
                let list = rk.pool.list(&l.run).unwrap();
                for row in 0..l.written {
                    assert_eq!(rk.dev[list[row / ROWS] as usize][row % ROWS], prompt[row], "op {op}: hit row {row}");
                }
            }
            for rk in &mut ranks {
                rk.write(&l, l.written, prompt.len());
                rk.pool.prefilled(&l.run, prompt.len() as u32).unwrap();
            }
            l.written = prompt.len();
            live.push(l);
        } else if k < 7 {
            // decode: grow by a few rows, toward the worst case
            let i = rng.range(0, live.len() as u64 - 1) as usize;
            let add = rng.range(1, 70) as usize;
            let to = (live[i].written + add).min(live[i].worst);
            if to == live[i].written {
                continue;
            }
            let mut got = Vec::new();
            for rk in &mut ranks {
                let g = rk.pool.grow(&live[i].run, to as u64);
                rk.copies();
                got.push((g, rk.pool.digest()));
            }
            assert_eq!(got[0], got[1], "op {op}: grow");
            if let Ok(Some(run)) = got[0].0 {
                live[i].run = run;
                let w = live[i].written;
                live[i].tokens.extend((w..to).map(|x| x as i32 + 2_000_000_000 - (op as i32 % 1000) * 70_000));
                for rk in &mut ranks {
                    rk.write(&live[i], w, to);
                }
                live[i].written = to;
            }
        } else {
            // retire (retain or release)
            let i = rng.range(0, live.len() as u64 - 1) as usize;
            let l = live.swap_remove(i);
            let keep = !rng.one_in(4);
            let d: Vec<u64> = ranks
                .iter_mut()
                .map(|rk| {
                    if keep {
                        let mut history = l.tokens[..l.written].to_vec();
                        history.push(-7);
                        let st = Finished {
                            history,
                            valid: l.written as i32,
                            written_end: l.written as i32,
                            pf_end: l.prompt_len as i32 - 1,
                        };
                        rk.pool.retain(&l.run, st).unwrap();
                    } else {
                        rk.pool.release(&l.run).unwrap();
                    }
                    rk.pool.digest()
                })
                .collect();
            assert_eq!(d[0], d[1]);
        }
        for rk in &ranks {
            rk.check(&live);
        }
    }
    if drain_at.is_some() {
        // the Banker's guarantee: grow some run to its worst, release it, repeat; each round must make progress
        while !live.is_empty() {
            let before = live.len();
            let mut i = 0;
            while i < live.len() {
                let w = live[i].worst as u64;
                let g: Vec<_> = ranks.iter_mut().map(|rk| rk.pool.grow(&live[i].run, w)).collect();
                assert_eq!(g[0], g[1]);
                if let Ok(Some(run)) = g[0] {
                    for rk in &mut ranks {
                        rk.copies();
                        rk.pool.release(&run).unwrap();
                    }
                    live.swap_remove(i);
                } else {
                    assert_eq!(g[0], Ok(None), "seed {seed}");
                    i += 1;
                }
            }
            assert!(live.len() < before, "seed {seed} op {n}: {} live runs can not grow to their worst", live.len());
        }
        for rk in &ranks {
            rk.check(&live);
            assert_eq!(rk.pool.live_pages(), 0);
        }
    }
    (hits, copies)
}

#[test]
fn t3_seeded_random_traces() {
    let (mut hits, mut copies) = (0, 0);
    for seed in 0..5u64 {
        let (h, c) = trace(0xC105_0000 + seed, 1000, None);
        hits += h;
        copies += c;
    }
    assert!(hits > 200 && copies > 50, "the traces share pages: {hits} hits, {copies} copies");
}

#[test]
fn t3_bankers_every_admitted_run_can_grow_to_its_worst() {
    for seed in 0..40u64 {
        trace(0xBA2C_0000 + seed, 0, Some(20 + (seed as usize * 37) % 400));
    }
}
