//! Multi-rank verification of CP-tail (past R) storable truncation, zero self-hit spills, and extend-in-place.

use g6kvman::stash::{Io, Keep, Kind, Part};
use g6kvman::{Finished, Pool, Rules, Run, PAGE};
use std::path::Path;

const ROWS: usize = PAGE as usize;

struct CpRank {
    pool: Pool,
    rank: usize,
    nr: usize,
    bp: usize,
    place: u32,
    attn: Vec<[i32; ROWS]>,
    idx: Vec<[i32; ROWS]>,
    tier: Vec<([i32; ROWS], [i32; ROWS], Vec<[i32; ROWS]>)>,
}

impl CpRank {
    fn new(
        rank: usize,
        nr: usize,
        per: u32,
        pages: u32,
        place: u32,
        bp: usize,
        grid: i32,
        slots: u64,
        dir: Option<&Path>,
    ) -> Self {
        let mut pool = Pool::new(pages, place, (bp * ROWS) as u32).unwrap();
        pool.keep(Rules { strict: false, noend: true, grid });
        pool.stores(2);
        let idx_arg = dir.map(|d| (d, "cp_tail"));
        pool.add_tier(slots, Keep::All, idx_arg).unwrap();
        if dir.is_some() {
            pool.stash_mut().unwrap().index_whole(0);
        }
        pool.tail(nr as u32, per);
        let blank = ([-2; ROWS], [-2; ROWS], vec![[-2; ROWS]; per as usize]);
        Self {
            pool,
            rank,
            nr,
            bp,
            place,
            attn: vec![[-1; ROWS]; pages as usize],
            idx: vec![[-1; ROWS]; pages as usize],
            tier: vec![blank; slots as usize],
        }
    }

    fn phys(&self, list: &[u32], lp: usize) -> Option<usize> {
        if lp < self.bp {
            return Some(list[lp] as usize);
        }
        if lp % self.nr != self.rank {
            return None;
        }
        let j = (self.bp..lp).filter(|q| q % self.nr == self.rank).count();
        let mut seen = 0;
        for &p in list {
            if p < self.place {
                seen += 1;
                if seen == self.bp + j + 1 {
                    return Some(p as usize);
                }
            }
        }
        None
    }

    fn write(&mut self, run: &Run, from: usize, tokens: &[i32]) {
        let list = self.pool.list(run).unwrap().to_vec();
        for (r, &tok) in tokens.iter().enumerate().skip(from) {
            let lp = r / ROWS;
            self.idx[list[lp] as usize][r % ROWS] = tok;
            if let Some(p) = self.phys(&list, lp) {
                self.attn[p][r % ROWS] = tok;
            }
        }
    }

    fn check_prefix(&self, run: &Run, prompt: &[i32], skip: usize) {
        let list = self.pool.list(run).unwrap().to_vec();
        for (row, &tok) in prompt.iter().enumerate().take(skip) {
            let lp = row / ROWS;
            assert_eq!(self.idx[list[lp] as usize][row % ROWS], tok, "rank {} idx row {row}", self.rank);
            if let Some(p) = self.phys(&list, lp) {
                assert_eq!(self.attn[p][row % ROWS], tok, "rank {} attn row {row}", self.rank);
            }
        }
    }

    fn run_io(&mut self) -> Vec<Io> {
        let io = self.pool.io();
        for x in &io {
            match *x {
                Io::Spill { page, slot, part, .. } => {
                    let b = &mut self.tier[slot as usize];
                    match part {
                        Part::Full => (b.0, b.1) = (self.attn[page as usize], self.idx[page as usize]),
                        Part::Own => b.0 = self.attn[page as usize],
                        Part::Rep(i) => b.2[i as usize] = self.idx[page as usize],
                        Part::State(_) => unreachable!(),
                    }
                }
                Io::Move { slot, to_slot, .. } => {
                    self.tier[to_slot as usize] = self.tier[slot as usize].clone();
                }
                Io::Restore { slot, page, part, .. } => {
                    let b = self.tier[slot as usize].clone();
                    match part {
                        Part::Full => (self.attn[page as usize], self.idx[page as usize]) = (b.0, b.1),
                        Part::Own => self.attn[page as usize] = b.0,
                        Part::Rep(i) => self.idx[page as usize] = b.2[i as usize],
                        Part::State(_) => unreachable!(),
                    }
                }
            }
        }
        self.pool.io_done().unwrap();
        io
    }
}

fn toks(seed: i32, n: usize) -> Vec<i32> {
    (0..n).map(|i| seed * 100_000 + i as i32 + 1).collect()
}

fn finished(tokens: &[i32], pf_end: usize) -> Finished {
    let valid = tokens.len() as i32 - 1;
    Finished { history: tokens.to_vec(), valid, written_end: valid, pf_end: pf_end as i32 }
}

fn all_ranks<T: PartialEq + std::fmt::Debug>(ranks: &mut [CpRank], mut f: impl FnMut(&mut CpRank) -> (T, u64)) -> T {
    let got: Vec<(T, u64)> = ranks.iter_mut().map(|r| f(r)).collect();
    assert!(got.iter().all(|g| *g == got[0]), "ranks diverged: {got:?}");
    got.into_iter().next().unwrap().0
}

#[test]
fn resident_cp_tail_continuation_spills_nothing_while_branch_spills() {
    let (nr, per, pages, bp) = (4, 3, 64u32, 8usize);
    let place = bp as u32 + (pages - bp as u32) / nr as u32;
    let mut ranks: Vec<CpRank> = (0..nr).map(|r| CpRank::new(r, nr, per, pages, place, bp, 256, 512, None)).collect();

    let mut conv = toks(1, 1700);
    let pf1 = conv.len() - 1;
    conv.extend(toks(2, 80));
    let w1 = (conv.len() + 16) as u64;
    all_ranks(&mut ranks, |r| {
        r.pool.count(&conv[..pf1 + 1], None);
        let run = r.pool.reserve(w1, w1).unwrap();
        let _ = r.run_io();
        r.write(&run, 0, &conv);
        r.pool.retain(&run, finished(&conv, pf1)).unwrap();
        let _ = r.run_io();
        ((), r.pool.digest())
    });

    for turn in 0..5 {
        let prompt_len = conv.len() + 90;
        conv.extend(toks(10 + turn, 90));
        let pf = prompt_len - 1;
        let prompt = conv[..prompt_len].to_vec();
        conv.extend(toks(50 + turn, 60));
        let w = (conv.len() + 16) as u64;
        let h = ranks[0].pool.match_prefix(&prompt, w, w, pf as i32);
        assert!(h.skip > (bp * ROWS) as i32 && !h.stored, "turn {turn}: expected resident hit past R: {h:?}");
        let io = all_ranks(&mut ranks, |r| {
            r.pool.count(&prompt, None);
            let run = r.pool.take_hit(h.skip, h.age, None, w, w).unwrap();
            let io = r.run_io();
            r.check_prefix(&run, &prompt, h.skip as usize);
            r.write(&run, h.skip as usize, &conv);
            r.pool.retain(&run, finished(&conv, pf)).unwrap();
            let io2 = r.run_io();
            assert!(io2.is_empty());
            (io, r.pool.digest())
        });
        assert!(io.is_empty(), "turn {turn}: self-hit past R must not spill to SSD: {io:?}");
        assert_eq!(ranks[0].pool.tier_stats(0).unwrap().spilled_pages, 0, "turn {turn}");
    }

    let mut branch = conv[..1080].to_vec();
    branch.extend(toks(99, 50));
    let w = (branch.len() + 16) as u64;
    let h = ranks[0].pool.match_prefix(&branch, w, w, branch.len() as i32 - 1);
    assert!(h.skip == 1024 && !h.stored, "expected branch hit at 1024: {h:?}");
    let io = all_ranks(&mut ranks, |r| {
        r.pool.count(&branch, None);
        let run = r.pool.take_hit(h.skip, h.age, None, w, w).unwrap();
        let io = r.run_io();
        r.check_prefix(&run, &branch, h.skip as usize);
        r.pool.release(&run).unwrap();
        (io, r.pool.digest())
    });
    assert!(!io.is_empty(), "branching strictly before storable must preserve abandoned run on SSD");
}

#[test]
fn cp_tail_extend_in_place_across_widths_grids_and_write_limit() {
    for &(nr, per, grid) in &[(4, 3, 256), (4, 4, 64), (6, 3, 64), (3, 2, 128), (4, 3, 0)] {
        let bp = 8usize;
        let pages = 80u32;
        let place = bp as u32 + (pages - bp as u32) / nr as u32;
        let mut ranks: Vec<CpRank> =
            (0..nr).map(|r| CpRank::new(r, nr, per, pages, place, bp, grid, 1024, None)).collect();
        for r in &mut ranks {
            r.pool.write_limit(500);
        }

        let mut a = toks(1, 1500);
        let mut total_blocks_after_first = 0u64;
        for turn in 0..8 {
            let pf = a.len() - 1;
            let prompt = a.clone();
            a.extend(toks(20 + turn, 40));
            let w = (a.len() + 32) as u64;
            let h = ranks[0].pool.match_prefix(&prompt, w, w, pf as i32);
            if turn > 0 {
                assert!(h.stored && h.skip > (bp * ROWS) as i32, "nr={nr} grid={grid} turn={turn}: {h:?}");
            }
            all_ranks(&mut ranks, |r| {
                r.pool.count(&prompt, None);
                let run = if h.skip > 0 {
                    r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), w, w).unwrap()
                } else {
                    r.pool.reserve(w, w).unwrap()
                };
                let _ = r.run_io();
                if h.skip > 0 {
                    r.check_prefix(&run, &prompt, h.skip as usize);
                }
                r.write(&run, h.skip as usize, &a);
                r.pool.retain(&run, finished(&a, pf)).unwrap();
                let _ = r.run_io();
                ((), r.pool.digest())
            });

            let before_spilled = ranks[0].pool.tier_stats(0).unwrap().spilled_pages;
            let b = toks(500 + turn, (pages as usize - 2) * ROWS);
            let wb = (b.len() + 1) as u64;
            all_ranks(&mut ranks, |r| {
                r.pool.count(&b, None);
                let rb = r.pool.reserve(wb, wb).unwrap();
                let _ = r.run_io();
                r.write(&rb, 0, &b);
                r.pool.release(&rb).unwrap();
                ((), r.pool.digest())
            });
            let st = ranks[0].pool.tier_stats(0).unwrap();
            assert_eq!(st.refused, 0, "nr={nr} grid={grid} turn={turn}: write limit must not starve with extend-in-place");
            assert_eq!(ranks[0].pool.stash().unwrap().entries(0).len(), 1, "extend replaces old entry in place");
            let delta_blocks = st.spilled_pages - before_spilled;
            if turn > 0 {
                total_blocks_after_first += delta_blocks;
                assert!(
                    delta_blocks <= 6,
                    "nr={nr} grid={grid} turn={turn}: expected small extend delta, got {delta_blocks} blocks"
                );
            }
            a.extend(toks(100 + turn, 90));
        }
        assert!(total_blocks_after_first > 0);
    }
}

#[test]
fn cp_tail_below_r_to_past_r_extend_and_persist_round_trip() {
    let dir = std::env::temp_dir().join(format!("g6kv-cp-tail-extend-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (nr, per, pages, bp, grid) = (4, 3, 64u32, 8usize, 128);
    let place = bp as u32 + (pages - bp as u32) / nr as u32;
    let rank_dirs: Vec<_> = (0..nr).map(|r| dir.join(format!("r{r}"))).collect();
    for d in &rank_dirs {
        std::fs::create_dir_all(d).unwrap();
    }
    let mut ranks: Vec<CpRank> =
        (0..nr).map(|r| CpRank::new(r, nr, per, pages, place, bp, grid, 512, Some(&rank_dirs[r]))).collect();

    let mut a = toks(1, 400);
    let pf1 = a.len() - 1;
    a.extend(toks(2, 20));
    let w1 = (a.len() + 16) as u64;
    all_ranks(&mut ranks, |r| {
        r.pool.count(&a[..pf1 + 1], None);
        let run = r.pool.reserve(w1, w1).unwrap();
        let _ = r.run_io();
        r.write(&run, 0, &a);
        r.pool.retain(&run, finished(&a, pf1)).unwrap();
        let _ = r.run_io();
        ((), r.pool.digest())
    });
    let flush = toks(77, (pages as usize - 2) * ROWS);
    all_ranks(&mut ranks, |r| {
        let rb = r.pool.reserve(flush.len() as u64, flush.len() as u64).unwrap();
        let _ = r.run_io();
        r.pool.release(&rb).unwrap();
        ((), r.pool.digest())
    });
    let e1 = ranks[0].pool.stash().unwrap().entries(0)[0].clone();
    assert!(e1.tail.is_none() && e1.slots.len() == 6);

    a.extend(toks(3, 780));
    let pf2 = a.len() - 1;
    let prompt2 = a.clone();
    a.extend(toks(4, 30));
    let w2 = (a.len() + 16) as u64;
    let h2 = ranks[0].pool.match_prefix(&prompt2, w2, w2, pf2 as i32);
    assert!(h2.stored && h2.skip == 384, "{h2:?}");
    all_ranks(&mut ranks, |r| {
        r.pool.count(&prompt2, None);
        let run = r.pool.take_hit(h2.skip, h2.age, Some(h2.tier), w2, w2).unwrap();
        let _ = r.run_io();
        r.check_prefix(&run, &prompt2, h2.skip as usize);
        r.write(&run, h2.skip as usize, &a);
        r.pool.retain(&run, finished(&a, pf2)).unwrap();
        let _ = r.run_io();
        ((), r.pool.digest())
    });
    let before = ranks[0].pool.tier_stats(0).unwrap().spilled_pages;
    all_ranks(&mut ranks, |r| {
        let rb = r.pool.reserve(flush.len() as u64, flush.len() as u64).unwrap();
        let _ = r.run_io();
        r.pool.release(&rb).unwrap();
        ((), r.pool.digest())
    });
    let e2 = ranks[0].pool.stash().unwrap().entries(0)[0].clone();
    assert_eq!(e2.slots[..6], e1.slots[..6], "first 6 below-R slots reused unchanged");
    assert_eq!(ranks[0].pool.tier_stats(0).unwrap().spilled_pages - before, 9);
    assert_eq!(e2.kind, Kind::Evicted);
    assert!(e2.tail.is_some());

    a.extend(toks(5, 20));
    let pf3 = a.len() - 1;
    let prompt3 = a.clone();
    let w3 = (a.len() + 16) as u64;
    let h3 = ranks[0].pool.match_prefix(&prompt3, w3, w3, pf3 as i32);
    assert!(h3.stored && h3.skip == 1152, "{h3:?}");
    let before3 = ranks[0].pool.tier_stats(0).unwrap().spilled_pages;
    all_ranks(&mut ranks, |r| {
        r.pool.count(&prompt3, None);
        let run = r.pool.take_hit(h3.skip, h3.age, Some(h3.tier), w3, w3).unwrap();
        let _ = r.run_io();
        r.check_prefix(&run, &prompt3, h3.skip as usize);
        r.write(&run, h3.skip as usize, &a);
        r.pool.retain(&run, finished(&a, pf3)).unwrap();
        let _ = r.run_io();
        let rb = r.pool.reserve(flush.len() as u64, flush.len() as u64).unwrap();
        let _ = r.run_io();
        r.pool.release(&rb).unwrap();
        ((), r.pool.digest())
    });
    assert_eq!(
        ranks[0].pool.tier_stats(0).unwrap().spilled_pages,
        before3,
        "no new grid cell crossed -> zero SSD writes on re-eviction"
    );

    let saved_tiers: Vec<_> = ranks.iter().map(|r| r.tier.clone()).collect();
    drop(ranks);
    let mut reloaded: Vec<CpRank> = (0..nr)
        .map(|r| {
            let mut cr = CpRank::new(r, nr, per, pages, place, bp, grid, 512, Some(&rank_dirs[r]));
            cr.tier = saved_tiers[r].clone();
            cr
        })
        .collect();
    let h4 = reloaded[0].pool.match_prefix(&prompt3, w3, w3, pf3 as i32);
    assert!(h4.stored && h4.skip == 1152, "reloaded hit: {h4:?}");
    all_ranks(&mut reloaded, |r| {
        let run = r.pool.take_hit(h4.skip, h4.age, Some(h4.tier), w3, w3).unwrap();
        let _ = r.run_io();
        r.check_prefix(&run, &prompt3, h4.skip as usize);
        r.pool.release(&run).unwrap();
        ((), r.pool.digest())
    });

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cp_tail_extend_after_resident_hit_chain_preserves_provenance() {
    let (nr, per, pages, bp, grid) = (4, 3, 80u32, 8usize, 128);
    let place = bp as u32 + (pages - bp as u32) / nr as u32;
    let mut ranks: Vec<CpRank> = (0..nr).map(|r| CpRank::new(r, nr, per, pages, place, bp, grid, 512, None)).collect();
    let flush = toks(888, (pages as usize - 2) * ROWS);

    let mut a = toks(1, 1500);
    let pf1 = a.len() - 1;
    a.extend(toks(2, 20));
    let w1 = (a.len() + 32) as u64;
    all_ranks(&mut ranks, |r| {
        r.pool.count(&a[..pf1 + 1], None);
        let run = r.pool.reserve(w1, w1).unwrap();
        let _ = r.run_io();
        r.write(&run, 0, &a);
        r.pool.retain(&run, finished(&a, pf1)).unwrap();
        let _ = r.run_io();
        let rb = r.pool.reserve(flush.len() as u64, flush.len() as u64).unwrap();
        let _ = r.run_io();
        r.pool.release(&rb).unwrap();
        ((), r.pool.digest())
    });
    let e1 = ranks[0].pool.stash().unwrap().entries(0)[0].clone();

    a.extend(toks(3, 150));
    let pf2 = a.len() - 1;
    let p2 = a.clone();
    a.extend(toks(4, 20));
    let w2 = (a.len() + 32) as u64;
    let h2 = ranks[0].pool.match_prefix(&p2, w2, w2, pf2 as i32);
    assert!(h2.stored && h2.skip == 1408, "{h2:?}");
    all_ranks(&mut ranks, |r| {
        r.pool.count(&p2, None);
        let run = r.pool.take_hit(h2.skip, h2.age, Some(h2.tier), w2, w2).unwrap();
        let _ = r.run_io();
        r.check_prefix(&run, &p2, h2.skip as usize);
        r.write(&run, h2.skip as usize, &a);
        r.pool.retain(&run, finished(&a, pf2)).unwrap();
        let _ = r.run_io();
        ((), r.pool.digest())
    });

    a.extend(toks(5, 150));
    let pf3 = a.len() - 1;
    let p3 = a.clone();
    a.extend(toks(6, 20));
    let w3 = (a.len() + 32) as u64;
    let h3 = ranks[0].pool.match_prefix(&p3, w3, w3, pf3 as i32);
    assert!(!h3.stored && h3.skip == 1664, "{h3:?}");
    let before_hit = ranks[0].pool.tier_stats(0).unwrap().spilled_pages;
    all_ranks(&mut ranks, |r| {
        r.pool.count(&p3, None);
        let run = r.pool.take_hit(h3.skip, h3.age, None, w3, w3).unwrap();
        let io = r.run_io();
        assert!(io.is_empty(), "resident hit after restore must not spill: {io:?}");
        r.check_prefix(&run, &p3, h3.skip as usize);
        r.write(&run, h3.skip as usize, &a);
        r.pool.retain(&run, finished(&a, pf3)).unwrap();
        let _ = r.run_io();
        ((), r.pool.digest())
    });
    assert_eq!(ranks[0].pool.tier_stats(0).unwrap().spilled_pages, before_hit);

    all_ranks(&mut ranks, |r| {
        let rb = r.pool.reserve(flush.len() as u64, flush.len() as u64).unwrap();
        let _ = r.run_io();
        r.pool.release(&rb).unwrap();
        ((), r.pool.digest())
    });
    let entries = ranks[0].pool.stash().unwrap().entries(0);
    assert_eq!(entries.len(), 1, "e1 replaced in place even across a resident hit");
    assert_eq!(entries[0].slots[..bp], e1.slots[..bp], "all below-R slots reused");
    let delta = ranks[0].pool.tier_stats(0).unwrap().spilled_pages - before_hit;
    assert!(delta <= 6, "expected delta <= 6 blocks, got {delta}");

    a.extend(toks(7, 50));
    let p4 = a.clone();
    let w4 = (p4.len() + 16) as u64;
    let h4 = ranks[0].pool.match_prefix(&p4, w4, w4, p4.len() as i32 - 1);
    assert!(h4.stored && h4.skip == 1792, "{h4:?}");
    all_ranks(&mut ranks, |r| {
        let run = r.pool.take_hit(h4.skip, h4.age, Some(h4.tier), w4, w4).unwrap();
        let _ = r.run_io();
        r.check_prefix(&run, &p4, h4.skip as usize);
        r.pool.release(&run).unwrap();
        ((), r.pool.digest())
    });
}

#[test]
fn cp_tail_randomized_multi_agent_churn_restores_exact_rows_on_every_rank() {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize % (hi - lo + 1))
        }
    }

    for &(nr, per, grid) in &[(4, 3, 64), (4, 4, 256), (6, 3, 0)] {
        let bp = 8usize;
        let pages = 96u32;
        let place = bp as u32 + (pages - bp as u32) / nr as u32;
        let mut ranks: Vec<CpRank> =
            (0..nr).map(|r| CpRank::new(r, nr, per, pages, place, bp, grid, 4096, None)).collect();
        let mut rng = Rng(0xC0FFEE ^ ((nr as u64) << 16) ^ (grid as u64));
        let sys = toks(7, 320); // shared system prompt below R (5 pages)
        let mut convs: Vec<Vec<i32>> = (0..5)
            .map(|c| {
                let mut v = sys.clone();
                v.extend(toks(10 + c as i32, rng.range(100, 900)));
                v
            })
            .collect();

        for step in 0..80 {
            let cid = rng.range(0, convs.len() - 1);
            // Occasionally branch back or grow past R
            if convs[cid].len() > 2800 || (convs[cid].len() > 900 && rng.range(0, 7) == 0) {
                let cut = rng.range(350, convs[cid].len() - 60);
                convs[cid].truncate(cut);
            }
            let delta = rng.range(20, 260);
            convs[cid].extend(toks(1000 + step as i32 * 2, delta));
            let pf = convs[cid].len() - 1;
            let prompt = convs[cid].clone();
            let gen = rng.range(5, 80);
            convs[cid].extend(toks(1000 + step as i32 * 2 + 1, gen));
            let w = (convs[cid].len() + 32) as u64;
            if w > pages as u64 * PAGE {
                convs[cid].truncate(600);
                continue;
            }
            let h = ranks[0].pool.match_prefix(&prompt, w, w, pf as i32);
            all_ranks(&mut ranks, |r| {
                r.pool.count(&prompt, Some(sys.len() as u32));
                let run = if h.skip > 0 {
                    r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), w, w).unwrap()
                } else {
                    r.pool.reserve(w, w).unwrap()
                };
                let _ = r.run_io();
                if h.skip > 0 {
                    r.check_prefix(&run, &prompt, h.skip as usize);
                }
                r.write(&run, h.skip as usize, &convs[cid]);
                r.pool.retain(&run, finished(&convs[cid], pf)).unwrap();
                let _ = r.run_io();
                ((), r.pool.digest())
            });
        }
    }
}
