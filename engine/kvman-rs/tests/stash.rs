//! The block stores below the pool (src/stash.rs) on random traces, with a model of the device pages and every tier's
//! blocks: each row is the token it holds. The engine's part is to run each call's copies in order. Then a grant's rows
//! below its skip must hold the new prompt's tokens, whether the pool kept them, a tier gave them back, or they moved
//! between tiers on the way. Two ranks given the same calls must queue the same copies and agree on the digest. Also:
//! what counts as a new conversation, promotion of a shared prefix, moves between tiers and the persistent index.

use g6kvman::stash::{prefix_key, Io, Keep};
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

const ROWS: usize = PAGE as usize;
const GRID: i32 = 128;
const BATCH: usize = 32;
type Page = [i32; ROWS];

/// One rank: its pool, its device pages and each tier's blocks.
struct Rank {
    pool: Pool,
    dev: Vec<Page>,
    tiers: Vec<Vec<Page>>,
}

fn pool(pages: u32, place: u32, below: u32, shared_after: u32) -> Pool {
    let mut pool = Pool::new(pages, place, below).unwrap();
    pool.keep(Rules { strict: false, noend: true, grid: GRID });
    pool.stores(shared_after);
    pool
}

impl Rank {
    fn new(pages: u32, place: u32, below: u32, tiers: &[(u64, Keep)], shared_after: u32) -> Rank {
        let mut p = pool(pages, place, below, shared_after);
        for &(slots, keep) in tiers {
            p.add_tier(slots, keep, None).unwrap();
        }
        Rank::with(p, tiers.iter().map(|t| t.0).collect())
    }
    fn with(pool: Pool, slots: Vec<u64>) -> Rank {
        let pages = pool.pool_pages() as usize;
        Rank {
            pool,
            dev: vec![[-1; ROWS]; pages],
            tiers: slots.iter().map(|&n| vec![[-2; ROWS]; n as usize]).collect(),
        }
    }
    // the engine: every copy in the order queued, then io_done
    fn run_io(&mut self) -> Vec<Io> {
        let io = self.pool.io();
        if !io.is_empty() {
            assert_eq!(self.pool.reserve(1, 1), Err(Error::PendingIo), "a claim before io_done");
        }
        let first_restore = io.iter().position(|x| matches!(x, Io::Restore { .. })).unwrap_or(io.len());
        assert!(io[first_restore..].iter().all(|x| matches!(x, Io::Restore { .. })), "a copy after a restore");
        // as the engine runs them (s2/kv_tier.cuh KvTiers::run): batches of up to 32 copies of one kind, a batch's reads
        // before its writes, and a batch of moves ends before a move that reads a block the batch writes
        let mut i = 0;
        while i < io.len() {
            let kind = std::mem::discriminant(&io[i]);
            let mut batch: Vec<Io> = Vec::new();
            while i < io.len() && std::mem::discriminant(&io[i]) == kind && batch.len() < BATCH {
                if let Io::Move { tier, slot, .. } = io[i] {
                    if batch.iter().any(|b| matches!(*b, Io::Move { to, to_slot, .. } if (to, to_slot) == (tier, slot)))
                    {
                        break;
                    }
                }
                batch.push(io[i]);
                i += 1;
            }
            let reads: Vec<Page> = batch
                .iter()
                .map(|x| match *x {
                    Io::Spill { page, .. } => self.dev[page as usize],
                    Io::Move { tier, slot, to, .. } => {
                        assert!(to > tier, "a move goes to a later tier");
                        self.tiers[tier as usize][slot as usize]
                    }
                    Io::Restore { tier, slot, .. } => self.tiers[tier as usize][slot as usize],
                })
                .collect();
            for (x, data) in batch.iter().zip(reads) {
                match *x {
                    Io::Spill { tier, slot, .. } => self.tiers[tier as usize][slot as usize] = data,
                    Io::Move { to, to_slot, .. } => self.tiers[to as usize][to_slot as usize] = data,
                    Io::Restore { page, .. } => self.dev[page as usize] = data,
                }
            }
        }
        self.pool.io_done().unwrap();
        io
    }
    fn write(&mut self, list: &[u32], from: usize, tokens: &[i32]) {
        for (r, &t) in tokens.iter().enumerate().skip(from) {
            self.dev[list[r / ROWS] as usize][r % ROWS] = t;
        }
    }
    fn check_slots(&self) {
        let s = self.pool.stash().unwrap();
        for t in 0..s.tiers() {
            let st = s.stats(t).unwrap();
            let mut seen = vec![false; st.slots as usize];
            for e in s.entries(t) {
                if let Some((base, n)) = e.base {
                    let found = (0..s.tiers()).flat_map(|u| s.entries(u)).find(|b| b.id == base);
                    assert!(found.is_some_and(|b| b.slots.len() == n), "entry {} lost its base {base}", e.id);
                }
                for &x in &e.slots {
                    assert!(!seen[x as usize], "slot {x} of tier {t} held twice");
                    seen[x as usize] = true;
                }
            }
            assert_eq!(seen.iter().filter(|&&x| x).count() as u64 + st.free_slots, st.slots, "a slot of tier {t} lost");
        }
    }
}

struct Live {
    run: Run,
    tokens: Vec<i32>,
}

fn finished(tokens: &[i32]) -> Finished {
    let valid = tokens.len() as i32 - 1;
    Finished { history: tokens.to_vec(), valid, written_end: valid, pf_end: valid }
}

#[test]
fn restored_rows_hold_the_prompt_and_ranks_agree() {
    let (mut evicted_hits, mut shared_hits, mut moves, mut spills, mut exclusive) = (0, 0, 0, 0, 0);
    for seed in 1..=60u64 {
        let mut rng = Rng(seed * 104_729);
        let pages = rng.range(12, 48) as u32;
        let place = if rng.one_in(2) { pages } else { rng.range(8, pages as u64) as u32 };
        let below = rng.range(4, place as u64) as u32 * PAGE as u32;
        // every second seed adds a middle tier. A full middle tier can pass on an entry in the call that brought it in
        let mut tiers = vec![(rng.range(2, 12), Keep::Evicted), (rng.range(4, 40), Keep::All)];
        if seed % 2 == 0 {
            tiers.insert(1, (rng.range(2, 10), Keep::Evicted));
        }
        let after = rng.range(2, 3) as u32;
        let mut ranks = [Rank::new(pages, place, below, &tiers, after), Rank::new(pages, place, below, &tiers, after)];
        // a few system prompts that many conversations start with
        let systems: Vec<Vec<i32>> =
            (0..3).map(|_| (0..rng.range(130, 400)).map(|_| rng.range(0, 1 << 20) as i32).collect()).collect();
        let mut live: Vec<Live> = Vec::new();
        let mut histories: Vec<Vec<i32>> = Vec::new();
        for _ in 0..400 {
            if !live.is_empty() && (rng.one_in(2) || live.len() > 3) {
                let s = live.remove(rng.range(0, live.len() as u64 - 1) as usize);
                let mut io = Vec::new();
                for r in &mut ranks {
                    r.pool.retain(&s.run, finished(&s.tokens)).unwrap();
                    io.push(r.run_io());
                }
                assert_eq!(io[0], io[1]);
                spills += io[0].len();
                histories.push(s.tokens);
                continue;
            }
            let mut prompt: Vec<i32> = match rng.range(0, 3) {
                0 if !histories.is_empty() => {
                    let h = &histories[rng.range(0, histories.len() as u64 - 1) as usize];
                    h[..rng.range(1, h.len() as u64) as usize].to_vec()
                }
                1 | 2 => systems[rng.range(0, 2) as usize].clone(),
                _ => Vec::new(),
            };
            prompt.extend((0..rng.range(1, 2 * PAGE)).map(|_| rng.range(0, 1 << 20) as i32));
            let gen = rng.range(1, PAGE) as usize;
            let worst = (prompt.len() + gen + 1) as u64;
            let now = if ranks[0].pool.exclusive(worst) { worst } else { prompt.len() as u64 + 1 };
            if worst > pages as u64 * PAGE || !ranks[0].pool.fits(now, worst) {
                continue;
            }
            let limit = prompt.len() as i32 - 1;
            let counted: Vec<u32> = ranks.iter_mut().map(|r| r.pool.count(&prompt)).collect();
            assert_eq!(counted[0], counted[1]);
            let h = ranks[0].pool.match_prefix(&prompt, now, worst, limit);
            assert_eq!(h, ranks[1].pool.match_prefix(&prompt, now, worst, limit));
            let shared = h.stored
                && ranks[0].pool.stash().unwrap().entries(h.tier as usize).iter().any(|e| e.id == h.age && e.key != 0);
            let mut tokens = prompt.clone();
            tokens.extend((0..gen).map(|_| rng.range(0, 1 << 20) as i32));
            let mut got = Vec::new();
            for r in &mut ranks {
                let from = h.stored.then_some(h.tier);
                let mut run = if h.skip > 0 {
                    r.pool.take_hit(h.skip, h.age, from, now, worst).unwrap()
                } else {
                    r.pool.reserve(now, worst).unwrap()
                };
                let mut io = r.run_io();
                let list = r.pool.list(&run).unwrap().to_vec();
                for (row, &t) in prompt.iter().enumerate().take(h.skip as usize) {
                    assert_eq!(
                        r.dev[list[row / ROWS] as usize][row % ROWS],
                        t,
                        "seed {seed}: reused row {row} ({h:?})"
                    );
                }
                r.write(&list, h.skip as usize, &prompt);
                let mut end = prompt.len();
                if let Some(g) = r.pool.grow(&run, worst).unwrap() {
                    run = g;
                    io.extend(r.run_io());
                    end = tokens.len();
                    r.write(r.pool.list(&run).unwrap().to_vec().as_slice(), prompt.len(), &tokens);
                }
                r.check_slots();
                got.push((run, io, end, r.pool.digest()));
            }
            assert_eq!(got[0], got[1], "the ranks differ");
            let (run, io, end, _) = got.remove(0);
            spills += io.iter().filter(|x| matches!(x, Io::Spill { .. })).count();
            moves += io.iter().filter(|x| matches!(x, Io::Move { .. })).count();
            exclusive += ranks[0].pool.exclusive_run(&run).unwrap() as u32;
            evicted_hits += (h.stored && !shared) as u32;
            shared_hits += shared as u32;
            tokens.truncate(end);
            live.push(Live { run, tokens });
        }
    }
    assert!(
        evicted_hits > 20 && shared_hits > 20 && moves > 50 && spills > 100 && exclusive > 0,
        "evicted hits {evicted_hits}, shared hits {shared_hits}, moves {moves}, spills {spills}, exclusive {exclusive}"
    );
}

fn toks(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| seed * 100_000 + i).collect()
}

// admits a prompt (count, match, take or reserve, copies) and retires it after `gen` more tokens
fn session(r: &mut Rank, prompt: &[i32], gen: usize) -> (u32, bool, Vec<i32>) {
    let n = r.pool.count(prompt);
    let rows = (prompt.len() + gen + 1) as u64;
    let h = r.pool.match_prefix(prompt, rows, rows, prompt.len() as i32 - 1);
    let run = if h.skip > 0 {
        r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), rows, rows).unwrap()
    } else {
        r.pool.reserve(rows, rows).unwrap()
    };
    r.run_io();
    let mut t = prompt.to_vec();
    t.extend(toks(-1 - prompt.len() as i32, gen));
    let list = r.pool.list(&run).unwrap().to_vec();
    r.write(&list, h.skip as usize, &t);
    r.pool.retain(&run, finished(&t)).unwrap();
    r.run_io();
    (n, h.stored, t)
}

// a reservation of the whole pool: every retained run is evicted
fn flush(r: &mut Rank) {
    let n = r.pool.pool_pages() as u64 * PAGE;
    let all = r.pool.reserve(n, n).unwrap();
    r.run_io();
    r.pool.release(&all).unwrap();
}

#[test]
fn a_chat_counts_once_and_a_shared_prefix_is_stored_at_the_third_conversation() {
    let mut r = Rank::new(12, 12, 12 * 64, &[(64, Keep::Shared)], 3);
    let system = toks(1, 300); // two grid cells of 128 rows
    let key = prefix_key(&system[..257]);
    let mut t = system.clone();
    t.extend(toks(2, 50));
    // a chat of five turns, each continuing the last
    let (first, _, mut t) = session(&mut r, &t, 20);
    assert_eq!(first, 2, "cells 128 and 256 of the first conversation");
    for _ in 0..4 {
        t.extend(toks(3, 10));
        let (n, _, t2) = session(&mut r, &t, 20);
        assert_eq!(n, 0, "a continued chat counts toward nothing");
        t = t2;
    }
    assert_eq!(r.pool.stash().unwrap().count_of(key), 1);
    assert!(r.pool.stash().unwrap().entries(0).is_empty());
    // two more conversations with the same system prompt: the third stores 256 rows as a shared prefix
    for c in 4..6 {
        let mut p = system.clone();
        p.extend(toks(c, 40));
        session(&mut r, &p, 10);
    }
    assert_eq!(r.pool.stash().unwrap().count_of(key), 3);
    let shared = r.pool.stash().unwrap().entries(0).to_vec();
    assert_eq!(shared.len(), 1, "stored once");
    assert_eq!((shared[0].key, shared[0].slots.len()), (key, 4));
    // with every retained run gone, a fourth conversation reads the prefix back, and the prefix stays
    flush(&mut r);
    let mut p = system.clone();
    p.extend(toks(9, 40));
    let (_, stored, _) = session(&mut r, &p, 10);
    assert!(stored);
    assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1);
    assert_eq!(r.pool.tier_stats(0).unwrap().hits, 1);
}

#[test]
fn a_full_tier_moves_its_oldest_entry_on() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(4, Keep::Evicted), (16, Keep::Evicted)], 3);
    let mut runs = Vec::new();
    for c in 0..4 {
        let (_, _, t) = session(&mut r, &toks(10 + c, 200), 30);
        runs.push(t);
        flush(&mut r);
    }
    let s = r.pool.stash().unwrap();
    assert_eq!(s.entries(0).len(), 1, "tier 0 holds the newest run");
    assert_eq!(s.entries(1).len(), 3, "the older ones moved on to tier 1");
    // the oldest comes back from tier 1, row for row
    let mut p = runs[0][..220].to_vec();
    p.push(7);
    let h = r.pool.match_prefix(&p, 300, 300, p.len() as i32 - 1);
    assert!(h.stored && h.tier == 1 && h.skip == 128, "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(1), 300, 300).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], p[row]);
    }
}

#[test]
fn shared_prefixes_survive_a_restart_and_a_torn_record_does_not() {
    let dir = std::env::temp_dir().join(format!("g6kv-index-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let open = |compat: &str| {
        let mut p = pool(12, 12, 12 * 64, 2);
        p.add_tier(32, Keep::Shared, Some((dir.as_path(), compat))).unwrap();
        Rank::with(p, vec![32])
    };
    let system = toks(1, 200);
    let key = prefix_key(&system[..129]);
    {
        let mut r = open("model-a tp4");
        for c in 2..4 {
            let mut p = system.clone();
            p.extend(toks(c, 30));
            session(&mut r, &p, 5);
        }
        assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1);
        // a second prefix whose copies were taken but never reported done: no record
        let other = toks(5, 200);
        for c in 6..8 {
            let mut p = other.clone();
            p.extend(toks(c, 30));
            assert!(r.pool.count(&p) > 0);
            let run = r.pool.reserve(p.len() as u64 + 10, p.len() as u64 + 10).unwrap();
            r.run_io();
            r.pool.retain(&run, finished(&p)).unwrap();
        }
        assert_eq!(r.pool.stash().unwrap().entries(0).len(), 2);
        let _ = r.pool.io(); // the engine died before io_done
    }
    // and a torn record at the end of the log
    let ix = dir.join("index");
    let mut raw = std::fs::read(&ix).unwrap();
    raw.extend_from_slice(&[40, 0, 0, 0, 1, 2, 3]);
    std::fs::write(&ix, raw).unwrap();
    let r = open("model-a tp4");
    let e = r.pool.stash().unwrap().entries(0);
    assert_eq!(e.len(), 1, "only the committed prefix comes back");
    assert_eq!(e[0].key, key);
    assert!(r.pool.stored_summary().lines().count() == 1);
    // the ranks keep the entries all of them loaded: here, none
    let mut r2 = open("model-a tp4");
    r2.pool.keep_agreed("");
    assert!(r2.pool.stash().unwrap().entries(0).is_empty());
    assert!(open("model-a tp4").pool.stash().unwrap().entries(0).is_empty(), "the drop was recorded");
    // another model's blocks are not this one's
    let mut r3 = open("model-a tp4");
    for c in 2..4 {
        let mut p = system.clone();
        p.extend(toks(c, 30));
        session(&mut r3, &p, 5);
    }
    drop(r3);
    assert_eq!(open("model-a tp4").pool.stash().unwrap().entries(0).len(), 1);
    assert!(open("model-b tp4").pool.stash().unwrap().entries(0).is_empty());
    assert!(open("model-a tp4").pool.stash().unwrap().entries(0).is_empty(), "the other model's index replaced it");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_hit_on_a_shared_prefix_keeps_the_rest_of_the_run_it_takes() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(32, Keep::Evicted)], 3);
    let system = toks(1, 200);
    let mut a = system.clone();
    a.extend(toks(2, 300));
    let (_, _, a_hist) = session(&mut r, &a, 20);
    // b shares only the system prompt: its hit takes a's run for 128 rows, and the rest of a's run goes to the tier
    let mut b = system.clone();
    b.extend(toks(3, 100));
    session(&mut r, &b, 20);
    assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1, "a's run is in the tier");
    // a's next turn carries its own prompt and an edited reply: it continues a, and a's rows come back
    let mut a2 = a.clone();
    a2.extend(toks(4, 30));
    assert_eq!(r.pool.count(&a2), 0);
    let h = r.pool.match_prefix(&a2, a2.len() as u64 + 10, a2.len() as u64 + 10, a2.len() as i32 - 1);
    assert!(h.stored && h.skip == 384, "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(0), a2.len() as u64 + 10, a2.len() as u64 + 10).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], a_hist[row]);
    }
}

#[test]
fn a_run_on_a_stored_shared_prefix_spills_only_its_own_pages() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    let system = toks(1, 300); // a shared prefix of 256 rows (4 pages) once three conversations start with it
    for c in 2..5 {
        let mut p = system.clone();
        p.extend(toks(c, 40));
        session(&mut r, &p, 10);
    }
    let shared = r.pool.stash().unwrap().entries(0).iter().find(|e| e.key != 0).map(|e| e.id).unwrap();
    // a fourth conversation: its run, evicted, spills only the pages past the shared prefix
    let mut p = system.clone();
    p.extend(toks(9, 140));
    let (_, _, hist) = session(&mut r, &p, 10);
    let before = r.pool.tier_stats(0).unwrap().spilled_pages;
    flush(&mut r);
    let e = r.pool.stash().unwrap().entries(0).iter().find(|e| e.st.history == hist).cloned().unwrap();
    assert_eq!(e.base, Some((shared, 4)));
    assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages - before, e.slots.len() as u64);
    assert_eq!(e.pages(), 8);
    // and it comes back whole: the shared prefix's blocks, then its own
    let mut p2 = hist.clone();
    p2.extend(toks(10, 5));
    let h = r.pool.match_prefix(&p2, p2.len() as u64 + 5, p2.len() as u64 + 5, p2.len() as i32 - 1);
    assert!(h.stored && h.age == e.id && h.skip == 384, "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), p2.len() as u64 + 5, p2.len() as u64 + 5).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], hist[row]);
    }
    assert!(r.pool.stash().unwrap().entries(0).iter().any(|e| e.id == shared), "the shared prefix stays");
}
