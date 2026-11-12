//! The block stores below the pool (src/stash.rs) on random traces, with a model of the device pages and every tier's
//! blocks: each row is the token it holds. The engine's part is to run each call's copies in order. Then a grant's rows
//! below its skip must hold the new prompt's tokens, whether the pool kept them, a tier gave them back, or they moved
//! between tiers on the way. Two ranks given the same calls must queue the same copies and agree on the digest. Also:
//! what counts as a new conversation, promotion of a shared prefix, moves between tiers and the persistent index.

use g6kvman::stash::{prefix_key, Io, Keep, Kind, Part, Policy};
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
                        // a later tier, or the same tier when it shrinks (resize_tier)
                        assert!(to >= tier, "a move goes to a later tier");
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
    random_traces(false);
}

// the same traces, where every so often a tier grows or shrinks on both ranks at once (a KV node lease that moved)
#[test]
fn restored_rows_hold_the_prompt_while_tiers_resize() {
    random_traces(true);
}

fn random_traces(resize: bool) {
    let mut resized = (0, 0, 0);
    let (mut evicted_hits, mut shared_hits, mut moves, mut spills, mut exclusive, mut refused) = (0, 0, 0, 0, 0, 0);
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
        // the policies the keep modes set (min_hits 1 for sessions), a length floor and the write limit
        let policy = Policy {
            min_rows: if seed % 5 == 0 { rng.range(64, 300) } else { 0 },
            min_hits: (seed % 3 == 0) as u32,
            block_bytes: 4096,
        };
        let limit = if seed % 4 == 1 { rng.range(100, 600) } else { 0 };
        for r in &mut ranks {
            for t in 0..tiers.len() {
                r.pool.set_policy(t, policy).unwrap();
            }
            r.pool.write_limit(limit);
        }
        // a few system prompts that many conversations start with
        let systems: Vec<Vec<i32>> =
            (0..3).map(|_| (0..rng.range(130, 400)).map(|_| rng.range(0, 1 << 20) as i32).collect()).collect();
        let mut live: Vec<Live> = Vec::new();
        let mut histories: Vec<Vec<i32>> = Vec::new();
        for _ in 0..400 {
            if resize && rng.one_in(12) {
                let t = rng.range(0, tiers.len() as u64 - 1) as usize;
                let old = ranks[0].pool.tier_stats(t).unwrap().slots;
                let to = rng.range(1, old * 2);
                let mut io = Vec::new();
                for r in &mut ranks {
                    if to > old {
                        r.tiers[t].resize(to as usize, [-2; ROWS]);
                    }
                    assert!(r.pool.resize_tier(t, to).unwrap());
                    io.push(r.run_io());
                    // the blocks past the new size go: poison them first, so a copy that still read one would show
                    for b in r.tiers[t].iter_mut().skip(to as usize) {
                        *b = [-3; ROWS];
                    }
                    r.tiers[t].truncate(to as usize);
                    r.check_slots();
                    assert_eq!(r.pool.tier_stats(t).unwrap().slots, to);
                }
                assert_eq!(io[0], io[1], "the ranks queue other copies for a resize");
                assert_eq!(ranks[0].pool.digest(), ranks[1].pool.digest());
                let inner = io[0].iter().filter(|x| matches!(x, Io::Move { tier, to, .. } if tier == to)).count();
                resized = (resized.0 + (to > old) as u32, resized.1 + (to < old) as u32, resized.2 + inner);
                continue;
            }
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
            // a prompt on a system prompt carries its span half the time, as a chat front end sends it
            let mut span = None;
            let mut prompt: Vec<i32> = match rng.range(0, 3) {
                0 if !histories.is_empty() => {
                    let h = &histories[rng.range(0, histories.len() as u64 - 1) as usize];
                    h[..rng.range(1, h.len() as u64) as usize].to_vec()
                }
                1 | 2 => {
                    let sys = systems[rng.range(0, 2) as usize].clone();
                    if rng.one_in(2) {
                        span = Some(sys.len() as u32);
                    }
                    sys
                }
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
            let counted: Vec<u32> = ranks.iter_mut().map(|r| r.pool.count(&prompt, span)).collect();
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
        let s = ranks[0].pool.stash().unwrap();
        refused += (0..s.tiers()).map(|t| s.stats(t).unwrap().refused).sum::<u64>();
    }
    if resize {
        assert!(resized.0 > 50 && resized.1 > 50 && resized.2 > 50, "grown, shrunk, moves inside a tier: {resized:?}");
    }
    assert!(
        evicted_hits > 20 && shared_hits > 20 && moves > 50 && spills > 100 && exclusive > 0 && refused > 20,
        "evicted hits {evicted_hits}, shared hits {shared_hits}, moves {moves}, spills {spills}, exclusive {exclusive}, \
         refused {refused}"
    );
}

fn toks(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| seed * 100_000 + i).collect()
}

// admits a prompt (count, match, take or reserve, copies) and retires it after `gen` more tokens
fn session(r: &mut Rank, prompt: &[i32], gen: usize) -> (u32, bool, Vec<i32>) {
    session_span(r, prompt, gen, None)
}

fn session_span(r: &mut Rank, prompt: &[i32], gen: usize, span: Option<u32>) -> (u32, bool, Vec<i32>) {
    let n = r.pool.count(prompt, span);
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
    // 240 + 30 tokens: a tier stores the 256 rows a hit can restore (two grid cells, 4 pages)
    for c in 0..4 {
        let (_, _, t) = session(&mut r, &toks(10 + c, 240), 30);
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
            assert!(r.pool.count(&p, None) > 0);
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
    assert_eq!(r.pool.stored_summary().lines().filter(|l| !l.starts_with("c ")).count(), 1);
    // the counts came back too: the two conversations, and the other prefix's two
    assert_eq!(r.pool.stash().unwrap().count_of(key), 2);
    assert_eq!(r.pool.stash().unwrap().count_of(prefix_key(&toks(5, 200)[..129])), 2);
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
    // a's run is in the tier as the system prompt they share (a prefix entry, stored once) and the rest of a's rows
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert_eq!(e.len(), 2, "{e:?}");
    assert!(e[0].kind == Kind::Prefix && e[0].st.valid == 128 && e[1].base == Some((e[0].id, 2)), "{e:?}");
    // a's next turn carries its own prompt and an edited reply: it continues a, and a's rows come back
    let mut a2 = a.clone();
    a2.extend(toks(4, 30));
    assert_eq!(r.pool.count(&a2, None), 0);
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
    // 449 valid rows: stored up to the 384 a hit can restore (grid 128), 6 pages
    assert_eq!(e.pages(), 6);
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

#[test]
fn a_shrinking_tier_keeps_the_prefix_its_runs_are_based_on_and_moves_them_down() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    let system = toks(1, 300);
    let mut hists = Vec::new();
    for c in 2..12 {
        let mut p = system.clone();
        p.extend(toks(c, 140));
        hists.push(session(&mut r, &p, 10).2);
    }
    flush(&mut r);
    let s = r.pool.stash().unwrap();
    let shared = s.entries(0).iter().find(|e| e.kind.prefix()).map(|e| e.id).unwrap();
    assert_eq!(s.lru(0)[0].0, shared, "the shared prefix is the least recently used");
    let used = 64 - r.pool.tier_stats(0).unwrap().free_slots;
    // shrink to a few slots more than half of what is used: some runs go, the base stays, the rest move down
    let to = used / 2 + 4;
    assert!(r.pool.resize_tier(0, to).unwrap());
    let io = r.run_io();
    assert!(io.iter().any(|x| matches!(x, Io::Move { tier: 0, to: 0, .. })), "{io:?}");
    r.tiers[0].truncate(to as usize);
    r.check_slots();
    let s = r.pool.stash().unwrap();
    assert!(s.entries(0).iter().any(|e| e.id == shared), "the base stays");
    let kept: Vec<_> = s.entries(0).iter().filter(|e| e.base.is_some()).map(|e| e.st.history.clone()).collect();
    assert!(!kept.is_empty() && kept.len() < hists.len(), "{} of {} runs kept", kept.len(), hists.len());
    // a kept run comes back whole from its new slots
    let mut p2 = kept[kept.len() - 1].clone();
    p2.extend(toks(50, 5));
    let n = p2.len() as u64 + 5;
    let h = r.pool.match_prefix(&p2, n, n, p2.len() as i32 - 1);
    assert!(h.stored, "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), n, n).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], p2[row]);
    }
}

#[test]
fn the_ages_a_head_reports_follow_the_shrink_and_never_rise() {
    // a shared prefix (the least recently used entry, the base of every run) and ten runs on it
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    let system = toks(1, 300);
    for c in 2..12 {
        let mut p = system.clone();
        p.extend(toks(c, 140));
        session(&mut r, &p, 10);
    }
    flush(&mut r);
    let s = r.pool.stash().unwrap();
    let shared = s.entries(0).iter().find(|e| e.kind.prefix()).map(|e| e.id).unwrap();
    assert_eq!(s.lru(0)[0].0, shared);
    let order = s.give_order(0);
    assert_eq!(order.len(), s.lru(0).len());
    assert_eq!(order.last().unwrap().0, shared, "a shrink takes the base last");
    let first = order[0].0;
    r.pool.coldest_ages(0, 4, 8); // the pool's clock learns every entry now
    std::thread::sleep(std::time::Duration::from_millis(60));
    // a new conversation starts with the prefix: the base is used now, every run stays cold
    let mut p = system.clone();
    p.extend(toks(90, 10));
    r.pool.count(&p, None);
    let ages = r.pool.coldest_ages(0, 4, 8);
    // before: ages[0] was the base's (~0 ms), so the node saw a hot group and never asked it to give
    assert!(ages[0] >= 60, "{ages:?}");
    assert!(ages.windows(2).all(|w| w[0] >= w[1]), "{ages:?}");
    // and the entry the report starts from is the one a shrink evicts first
    let used = 64 - r.pool.tier_stats(0).unwrap().free_slots;
    assert!(r.pool.resize_tier(0, used - 1).unwrap());
    let s = r.pool.stash().unwrap();
    assert!(!s.entries(0).iter().any(|e| e.id == first), "the first entry of give_order went first");
    assert!(s.entries(0).iter().any(|e| e.id == shared));
}

#[test]
fn a_system_span_counts_once_per_conversation_and_is_stored_cut_to_the_grid() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::Shared)], 2);
    let system = toks(1, 300); // the front end's span: 300 tokens, 256 rows on a grid of 128
    let key = prefix_key(&system[..257]);
    let mut p = system.clone();
    p.extend(toks(2, 50));
    let (n, _, mut t) = session_span(&mut r, &p, 20, Some(300));
    assert_eq!(n, 2, "the cells within the span: 128 and 256");
    t.extend(toks(3, 10));
    assert_eq!(session_span(&mut r, &t, 20, Some(300)).0, 0, "a continued chat counts toward nothing");
    assert!(r.pool.stash().unwrap().entries(0).is_empty(), "one conversation so far");
    let mut q = system.clone();
    q.extend(toks(4, 70));
    session_span(&mut r, &q, 10, Some(300));
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert_eq!(e.len(), 1, "stored at the second conversation");
    assert_eq!((e[0].key, e[0].st.valid, e[0].slots.len()), (key, 256, 4));
    // a span shorter than a grid cell counts nothing
    let mut x = toks(5, 100);
    x.extend(toks(6, 300));
    assert_eq!(session_span(&mut r, &x, 5, Some(100)).0, 0);
    // two system prompts that differ after 200 tokens share the first cell, which is stored at the second
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::Shared)], 2);
    for c in 0..2 {
        let mut p = toks(7, 200);
        p.extend(toks(20 + c, 100)); // a different working folder in each system prompt
        let span = p.len() as u32;
        p.extend(toks(30 + c, 50));
        session_span(&mut r, &p, 5, Some(span));
    }
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert!(e.len() == 1 && e[0].st.valid == 128, "{e:?}");
}

#[test]
fn min_hits_keeps_one_shots_out_and_min_rows_keeps_short_runs_out() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::Evicted)], 3);
    r.pool.set_policy(0, Policy { min_rows: 0, min_hits: 1, block_bytes: 4096 }).unwrap();
    // a one-shot conversation, evicted: not stored
    session(&mut r, &toks(1, 200), 20);
    flush(&mut r);
    assert!(r.pool.stash().unwrap().entries(0).is_empty());
    assert_eq!(r.pool.tier_stats(0).unwrap().refused, 1);
    // a conversation that continued once: stored
    let (_, _, mut t) = session(&mut r, &toks(2, 200), 20);
    t.extend(toks(3, 30));
    let (_, _, t2) = session(&mut r, &t, 20);
    flush(&mut r);
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert!(e.len() == 1 && e[0].st.history == t2 && e[0].hits == 1, "{e:?}");
    // min_rows: a run below it is refused, one above it stored
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::Evicted)], 3);
    r.pool.set_policy(0, Policy { min_rows: 300, min_hits: 0, block_bytes: 4096 }).unwrap();
    session(&mut r, &toks(4, 200), 20);
    flush(&mut r);
    assert!(r.pool.stash().unwrap().entries(0).is_empty());
    session(&mut r, &toks(5, 400), 20);
    flush(&mut r);
    assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1);
}

#[test]
fn the_write_limit_holds_writes_to_its_share_of_the_pages_computed() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(4096, Keep::Evicted)], 3);
    r.pool.write_limit(250); // a block per four pages computed
    let mut computed = 0u64;
    for c in 0..4000 {
        let p = toks(10 + c, 100 + (c as usize * 37) % 500);
        computed += (p.len() as u64).div_ceil(PAGE);
        session(&mut r, &p, 10);
    }
    flush(&mut r);
    let st = r.pool.tier_stats(0).unwrap();
    let burst = g6kvman::stash::BURST as u64;
    assert!(st.spilled_pages <= computed / 4 + burst + 16, "{} blocks written for {computed} pages", st.spilled_pages);
    assert!(st.refused > 0 && st.spilled_pages > computed / 8, "{st:?}");
}

#[test]
fn a_run_on_a_resident_prefix_comes_back_with_its_prefix_from_the_pool_or_the_tier() {
    for from_pool in [true, false] {
        let mut r = Rank::new(24, 24, 24 * 64, &[(64, Keep::Evicted)], 3);
        let system = toks(1, 300);
        let mut a = system.clone();
        a.extend(toks(2, 200));
        let (_, _, a_hist) = session(&mut r, &a, 20);
        let mut b = system.clone();
        b.extend(toks(3, 200));
        session(&mut r, &b, 20); // b takes a's system prompt rows: a's run is stored as a prefix and its own rows
        if !from_pool {
            flush(&mut r); // no run in the pool holds the system prompt now
        }
        let e = r.pool.stash().unwrap().entries(0).to_vec();
        let a_e = e.iter().find(|x| x.st.history == a_hist).unwrap().clone();
        assert!(a_e.base.is_some(), "{e:?}");
        let mut a2 = a_hist.clone();
        a2.extend(toks(4, 20));
        let rows = a2.len() as u64 + 5;
        let h = r.pool.match_prefix(&a2, rows, rows, a2.len() as i32 - 1);
        assert!(h.stored && h.age == a_e.id, "{h:?}");
        let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), rows, rows).unwrap();
        let io = r.run_io();
        let reads = io.iter().filter(|x| matches!(x, Io::Restore { .. })).count();
        let base = a_e.base.unwrap().1;
        assert_eq!(reads, (h.skip as usize).div_ceil(ROWS) - if from_pool { base } else { 0 }, "pool {from_pool}");
        let list = r.pool.list(&run).unwrap().to_vec();
        for row in 0..h.skip as usize {
            assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], a_hist[row], "pool {from_pool} row {row}");
        }
    }
}

/// A CP rank (stash.rs Tail): below R its pages hold every row; past R it holds the attention rows of every nr-th
/// logical page (its shard) and the drafter holds every page. A run past R carries its own page list (pool.rs): its
/// prefix and shard pages are the ones below the pool's place, the drafter's tail pages the ones above it, so this
/// rank's shard sits on its own pages of the shared list.
struct CpRank {
    pool: Pool,
    rank: usize,
    attn: Vec<Page>, // by physical page (this rank's attention cache)
    idx: Vec<Page>,  // by physical page (the drafter's, every logical page at its list page)
    tiers: Vec<Vec<(Page, Page, Vec<Page>)>>,
}

const NR: usize = 4;
const PER: u32 = 3;

impl CpRank {
    fn new(rank: usize, pages: u32, place: u32, below: u32, slots: u64) -> CpRank {
        let mut pool = pool(pages, place, below, 2);
        pool.add_tier(slots, Keep::All, None).unwrap();
        pool.tail(NR as u32, PER);
        let blank = ([-2; ROWS], [-2; ROWS], vec![[-2; ROWS]; PER as usize]);
        CpRank {
            pool,
            rank,
            attn: vec![[-1; ROWS]; pages as usize],
            idx: vec![[-1; ROWS]; pages as usize],
            tiers: vec![vec![blank; slots as usize]],
        }
    }
    // logical page -> this rank's physical page of the attention cache, when it holds it: the prefix at its list
    // page, an owned page at the (bp + j)-th page of the list below place (j its owned index)
    fn phys(&self, list: &[u32], lp: usize, bp: usize, place: u32) -> Option<usize> {
        if lp < bp {
            return Some(list[lp] as usize);
        }
        if lp % NR != self.rank {
            return None;
        }
        let j = (bp..lp).filter(|q| q % NR == self.rank).count();
        let mut seen = 0;
        for &p in list {
            if p < place {
                seen += 1;
                if seen == bp + j + 1 {
                    return Some(p as usize);
                }
            }
        }
        None
    }
    // a run's rows through its list: the attention cache at this rank's owned pages, the drafter at every page
    fn write(&mut self, run: &Run, from: usize, tokens: &[i32], bp: usize, place: u32) {
        let list = self.pool.list(run).unwrap().to_vec();
        for (r, &t) in tokens.iter().enumerate().skip(from) {
            let lp = r / ROWS;
            self.idx[list[lp] as usize][r % ROWS] = t;
            if let Some(p) = self.phys(&list, lp, bp, place) {
                self.attn[p][r % ROWS] = t;
            }
        }
    }
    fn run_io(&mut self) -> Vec<Io> {
        let io = self.pool.io();
        for x in &io {
            match *x {
                Io::Spill { tier, page, slot, part } => {
                    let b = &mut self.tiers[tier as usize][slot as usize];
                    match part {
                        Part::Full => (b.0, b.1) = (self.attn[page as usize], self.idx[page as usize]),
                        Part::Own => b.0 = self.attn[page as usize],
                        Part::Rep(i) => b.2[i as usize] = self.idx[page as usize],
                        Part::State(_) => unreachable!("Pool never emits Part::State"),
                    }
                }
                Io::Move { tier, slot, to, to_slot } => {
                    self.tiers[to as usize][to_slot as usize] = self.tiers[tier as usize][slot as usize].clone()
                }
                Io::Restore { tier, slot, page, part } => {
                    let b = self.tiers[tier as usize][slot as usize].clone();
                    match part {
                        Part::Full => (self.attn[page as usize], self.idx[page as usize]) = (b.0, b.1),
                        Part::Own => self.attn[page as usize] = b.0,
                        Part::Rep(i) => self.idx[page as usize] = b.2[i as usize],
                        Part::State(_) => unreachable!("Pool never emits Part::State"),
                    }
                }
            }
        }
        self.pool.io_done().unwrap();
        io
    }
}

#[test]
fn a_run_past_r_comes_back_with_every_ranks_shard_of_its_tail() {
    // 8 pages below R (512 rows), 32 logical pages past it shared by 4 ranks: 8 + 8 physical pages a rank
    let (pages, bp) = (40u32, 8usize);
    let place = bp as u32 + (pages - bp as u32) / NR as u32;
    let mut ranks: Vec<CpRank> = (0..NR).map(|r| CpRank::new(r, pages, place, bp as u32 * 64, 256)).collect();
    type Got = (Run, Vec<Io>, u64);
    let all = |ranks: &mut [CpRank], f: &mut dyn FnMut(&mut CpRank) -> Got| {
        let got: Vec<Got> = ranks.iter_mut().map(&mut *f).collect();
        assert!(got.iter().all(|g| *g == got[0]), "the ranks differ");
        got.into_iter().next().unwrap()
    };
    // a session past R: 1,500 prompt rows, 100 generated
    let a = toks(1, 1500);
    let mut a_hist = a.clone();
    a_hist.extend(toks(2, 100));
    let worst = (a.len() + 100 + 1) as u64;
    let (run, ..) = all(&mut ranks, &mut |r| {
        r.pool.count(&a, None);
        let run = r.pool.reserve(worst, worst).unwrap();
        let io = r.run_io();
        r.write(&run, 0, &a_hist, bp, place);
        (run, io, r.pool.digest())
    });
    all(&mut ranks, &mut |r| {
        r.pool.retain(&run, finished(&a_hist)).unwrap();
        let io = r.run_io();
        (run, io, r.pool.digest())
    });
    // another run past R takes the pool: a's run goes to the store, its tail as each rank's shard
    let b = toks(3, 2400);
    let wb = (b.len() + 1) as u64;
    let (runb, io, _) = all(&mut ranks, &mut |r| {
        r.pool.count(&b, None);
        let run = r.pool.reserve(wb, wb).unwrap();
        let io = r.run_io();
        r.write(&run, 0, &b, bp, place);
        (run, io, r.pool.digest())
    });
    let tail = (a_hist.len() - 1).div_ceil(ROWS) - bp; // logical pages past R the run held
    let owns = io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Own, .. })).count();
    let reps = io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Rep(_), .. })).count();
    assert_eq!((owns, reps), (tail.div_ceil(NR), tail), "{io:?}");
    all(&mut ranks, &mut |r| {
        r.pool.release(&runb).unwrap();
        (runb, Vec::new(), r.pool.digest())
    });
    // a's next turn: past R, its rows come back from the store on every rank, b's rows overwritten
    let mut a2 = a_hist.clone();
    a2.extend(toks(4, 30));
    let w2 = (a2.len() + 10) as u64;
    let h = ranks[0].pool.match_prefix(&a2, w2, w2, a2.len() as i32 - 1);
    assert!(h.stored && h.skip as usize > bp * ROWS, "{h:?}");
    let (run2, ..) = all(&mut ranks, &mut |r| {
        r.pool.count(&a2, None);
        let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), w2, w2).unwrap();
        let io = r.run_io();
        (run, io, r.pool.digest())
    });
    for r in &ranks {
        let list = r.pool.list(&run2).unwrap().to_vec();
        for (row, &t) in a2.iter().enumerate().take(h.skip as usize) {
            let lp = row / ROWS;
            assert_eq!(r.idx[list[lp] as usize][row % ROWS], t, "rank {} indexer row {row}", r.rank);
            if let Some(p) = r.phys(&list, lp, bp, place) {
                assert_eq!(r.attn[p][row % ROWS], t, "rank {} attention row {row}", r.rank);
            }
        }
    }
}

#[test]
fn a_long_run_spills_the_runs_it_evicts_for_either_class() {
    // a geometry whose drafter's class runs out of room first: place 22 in a pool of 56 pages, 4 below R, 4 ranks,
    // so a long run's drafter's tail (24 pages) is bigger than its low pages (12). A retained long run leaves 10
    // spare pages below place and 10 above; the next long run needs 9 below (no eviction) but 12 above: the
    // drafter's class is what evicts it, and it must spill — its tail as every rank's shard and the drafter's
    // copy — and reach the claim log, not vanish with its pages freed.
    let (pages, place, bp) = (56u32, 22u32, 4usize);
    let mut ranks: Vec<CpRank> = (0..NR).map(|r| CpRank::new(r, pages, place, bp as u32 * 64, 256)).collect();
    type Got = (Run, Vec<Io>, u64);
    let all = |ranks: &mut [CpRank], f: &mut dyn FnMut(&mut CpRank) -> Got| {
        let got: Vec<Got> = ranks.iter_mut().map(&mut *f).collect();
        assert!(got.iter().all(|g| *g == got[0]), "the ranks differ");
        got.into_iter().next().unwrap()
    };
    // session a: 2,300 prompt rows, 4 generated, its run 36 pages (4 prefix + 8 shard below place, 24 above)
    let a = toks(1, 2300);
    let mut a_hist = a.clone();
    a_hist.extend(toks(2, 4));
    let worst = a_hist.len() as u64;
    let (run, ..) = all(&mut ranks, &mut |r| {
        r.pool.count(&a, None);
        let run = r.pool.reserve(worst, worst).unwrap();
        let io = r.run_io();
        r.write(&run, 0, &a_hist, bp, place);
        (run, io, r.pool.digest())
    });
    all(&mut ranks, &mut |r| {
        r.pool.retain(&run, finished(&a_hist)).unwrap();
        (run, Vec::new(), r.pool.digest())
    });
    // session b: 1,280 rows, its run 21 pages (9 below place, 12 above): the low class has room, the drafter's
    // does not, so a is evicted for the high pages
    let b = toks(3, 1280);
    let wb = (b.len() + 1) as u64;
    let (runb, io, _) = all(&mut ranks, &mut |r| {
        r.pool.count(&b, None);
        let run = r.pool.reserve(wb, wb).unwrap();
        let io = r.run_io();
        r.write(&run, 0, &b, bp, place);
        (run, io, r.pool.digest())
    });
    // a's spill: its 4 prefix pages as full blocks, its 8 shard pages, the drafter's copy of its 32 tail pages
    let fulls = io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Full, .. })).count();
    let owns = io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Own, .. })).count();
    let reps = io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Rep(_), .. })).count();
    assert_eq!((fulls, owns, reps), (4, 8, 32), "the drafter's class eviction spilled nothing: {io:?}");
    all(&mut ranks, &mut |r| {
        r.pool.release(&runb).unwrap();
        (runb, Vec::new(), r.pool.digest())
    });
    // a's next turn: its rows come back from the store on every rank, b's rows overwritten
    let mut a2 = a_hist.clone();
    a2.extend(toks(4, 30));
    let w2 = (a2.len() + 10) as u64;
    let h = ranks[0].pool.match_prefix(&a2, w2, w2, a2.len() as i32 - 1);
    assert!(h.stored && h.skip as usize > bp * ROWS, "{h:?}");
    let (run2, ..) = all(&mut ranks, &mut |r| {
        r.pool.count(&a2, None);
        let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), w2, w2).unwrap();
        let io = r.run_io();
        (run, io, r.pool.digest())
    });
    for r in &ranks {
        let list = r.pool.list(&run2).unwrap().to_vec();
        for (row, &t) in a2.iter().enumerate().take(h.skip as usize) {
            let lp = row / ROWS;
            assert_eq!(r.idx[list[lp] as usize][row % ROWS], t, "rank {} indexer row {row}", r.rank);
            if let Some(p) = r.phys(&list, lp, bp, place) {
                assert_eq!(r.attn[p][row % ROWS], t, "rank {} attention row {row}", r.rank);
            }
        }
    }
}

#[test]
fn a_chat_counts_once_when_the_pool_lets_its_runs_go_unstored() {
    // a store of system prefixes only: the chat's runs are not stored when the pool evicts them
    let mut r = Rank::new(12, 12, 12 * 64, &[(64, Keep::Shared)], 2);
    let system = toks(1, 300);
    let mut t = system.clone();
    t.extend(toks(2, 50));
    let (first, _, mut t) = session_span(&mut r, &t, 20, Some(300));
    assert_eq!(first, 2);
    for c in 0..4 {
        flush(&mut r); // the chat's last run leaves the pool
        t.extend(toks(10 + c, 10));
        let (n, _, t2) = session_span(&mut r, &t, 20, Some(300));
        assert_eq!(n, 0, "turn {c} continues the chat");
        t = t2;
    }
    assert!(r.pool.stash().unwrap().entries(0).is_empty(), "one conversation never reaches N = 2");
    // a second conversation on the same system prompt does
    let mut q = system.clone();
    q.extend(toks(30, 40));
    session_span(&mut r, &q, 10, Some(300));
    assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1);
}

#[test]
fn non_stock_resident_rows_cannot_supply_a_stored_stock_base() {
    use g6kvman::pool::NAMESPACE_BYTES;
    let mut r = Rank::new(24, 24, 24 * 64, &[(64, Keep::Evicted)], 3);
    let system = toks(1, 300);
    let mut a = system.clone();
    a.extend(toks(2, 200));
    let (_, _, a_hist) = session(&mut r, &a, 20);
    let mut b = system.clone();
    b.extend(toks(3, 200));
    session(&mut r, &b, 20);
    flush(&mut r);
    let stored = r.pool.stash().unwrap().entries(0).iter().find(|e| e.st.history == a_hist).unwrap().clone();
    assert!(stored.base.is_some());
    // Same tokens, deliberately different KV bytes in the non-stock resident run.
    let ns = [1; NAMESPACE_BYTES];
    r.pool.count_namespaced(&b, None, ns);
    let run = r.pool.reserve(550, 550).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    r.write(&list, 0, &vec![-999; b.len()]);
    r.pool.retain_namespaced(&run, finished(&b), ns).unwrap();
    r.run_io();
    let mut next = a_hist.clone();
    next.extend(toks(4, 20));
    r.pool.count(&next, None);
    let rows = next.len() as u64 + 5;
    let h = r.pool.match_prefix(&next, rows, rows, next.len() as i32 - 1);
    assert!(h.stored && h.age == stored.id);
    let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), rows, rows).unwrap();
    let io = r.run_io();
    assert_eq!(io.iter().filter(|x| matches!(x, Io::Restore { .. })).count(), (h.skip as usize).div_ceil(ROWS));
    let list = r.pool.list(&run).unwrap();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], a_hist[row]);
    }
}

#[test]
fn non_stock_resident_history_cannot_create_a_stock_prefix_base() {
    let mut r = Rank::new(24, 24, 24 * 64, &[(64, Keep::Evicted)], 3);
    let system = toks(1, 300);
    let mut a = system.clone();
    a.extend(toks(2, 200));
    let (_, _, a_hist) = session(&mut r, &a, 20);
    let mut b = system;
    b.extend(toks(3, 200));
    let ns = [1; g6kvman::pool::NAMESPACE_BYTES];
    r.pool.count_namespaced(&b, None, ns);
    let run = r.pool.reserve(550, 550).unwrap();
    r.run_io();
    r.pool.retain_namespaced(&run, finished(&b), ns).unwrap();
    // Evict the older stock run while its only matching resident prefix is non-stock.
    let fresh = r.pool.reserve(550, 550).unwrap();
    r.run_io();
    let entries = r.pool.stash().unwrap().entries(0);
    let stored = entries.iter().find(|e| e.st.history == a_hist).unwrap();
    assert!(stored.base.is_none(), "a non-stock history cannot establish a stock base");
    assert!(entries.iter().all(|e| e.kind == Kind::Evicted));
    r.pool.release(&fresh).unwrap();
}

#[test]
fn an_entry_too_big_for_the_tier_is_counted_with_its_size() {
    // a 4-slot tier and a run of about 8 pages: the tier cannot hold it, and says so (the head asks for room for it,
    // s2/kv_share.h share_want)
    let mut r = Rank::new(16, 16, 16 * 64, &[(4, Keep::Evicted)], 3);
    session(&mut r, &toks(1, 480), 30);
    flush(&mut r);
    let st = r.pool.tier_stats(0).unwrap();
    assert_eq!((st.oversize, st.free_slots), (1, 4), "{st:?}");
    assert!(st.oversize_last > 4, "{st:?}");
}
// a persist=on ssd store with keep=sessions (the quit drain, g6m4.cu kv_whole): a whole index brings back the evicted
// runs the quit stored, with their base, so the next turn of a session restores instead of prefilling again
#[test]
fn a_whole_index_brings_back_evicted_runs_after_a_restart() {
    let dir = std::env::temp_dir().join(format!("g6kv-whole-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let open = |whole: bool| {
        let mut p = pool(12, 12, 12 * 64, 2);
        p.add_tier(64, Keep::All, Some((dir.as_path(), "whole-test"))).unwrap();
        if whole {
            p.stash_mut().unwrap().index_whole(0);
        }
        Rank::with(p, vec![64])
    };
    let system = toks(1, 200);
    let (blocks, turn) = {
        let mut r = open(true);
        let mut t = Vec::new();
        for c in 2..4 {
            let mut p = system.clone();
            p.extend(toks(c, 150));
            t = session(&mut r, &p, 20).2;
        }
        flush(&mut r);
        r.run_io();
        r.pool.io_done().unwrap();
        let e = r.pool.stash().unwrap().entries(0);
        assert!(e.iter().any(|e| e.kind == Kind::Evicted && e.base.is_some()), "{e:?}");
        (r.tiers.clone(), t)
    };
    let before: Vec<_> = {
        let r = open(true);
        r.pool.stash().unwrap().entries(0).to_vec()
    };
    assert!(before.iter().any(|e| e.kind == Kind::Evicted && e.base.is_some()), "the run came back with its base");
    // the next turn of the last conversation: restored from the store, row for row (session checks the rows)
    let mut r = open(true);
    r.tiers = blocks.clone();
    let mut next = turn.clone();
    next.extend(toks(9, 40));
    let h = r.pool.match_prefix(&next, next.len() as u64 + 6, next.len() as u64 + 6, next.len() as i32 - 1);
    assert!(h.stored && h.skip as usize > system.len(), "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), next.len() as u64 + 6, next.len() as u64 + 6).unwrap();
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for row in 0..h.skip as usize {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], next[row], "row {row}");
    }
    r.pool.io_done().unwrap();
    drop(r);
    // the hit leaves the run in the store (its next store extends it), and the index still has it
    assert_eq!(open(true).pool.stash().unwrap().entries(0).len(), before.len());
    // a prefixes-only index (persist=on) never kept the run
    let _ = std::fs::remove_dir_all(&dir);
    {
        let mut r = open(false);
        for c in 2..4 {
            let mut p = system.clone();
            p.extend(toks(c, 150));
            session(&mut r, &p, 20);
        }
        flush(&mut r);
        r.run_io();
        r.pool.io_done().unwrap();
    }
    assert!(open(false).pool.stash().unwrap().entries(0).iter().all(|e| e.kind.prefix()));
    let _ = std::fs::remove_dir_all(&dir);
}

// the quit drain turns the write limit off first: a store that refused writes while serving (no credit) still takes the
// pool's runs at the quit, and the next boot finds them
#[test]
fn the_quit_drain_stores_the_pool_past_the_write_limit() {
    let dir = std::env::temp_dir().join(format!("g6kv-quit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let open = || {
        let mut p = pool(12, 12, 12 * 64, 2);
        p.add_tier(64, Keep::All, Some((dir.as_path(), "quit-test"))).unwrap();
        p.stash_mut().unwrap().index_whole(0);
        Rank::with(p, vec![64])
    };
    let system = toks(1, 200);
    let stored = |limit: Option<u64>| {
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut r = open();
            r.pool.write_limit(1); // 0.001 blocks per page computed
            // serving spends the burst credit (BURST blocks), and the limit earns almost nothing back
            for c in 0..1500 {
                session(&mut r, &toks(100 + c, 300), 10);
            }
            assert!(r.pool.tier_stats(0).unwrap().refused > 0, "the credit is spent");
            for c in 2..4 {
                let mut p = system.clone();
                p.extend(toks(c, 150));
                session(&mut r, &p, 20);
            }
            if let Some(l) = limit {
                r.pool.write_limit(l);
            }
            flush(&mut r);
            r.run_io();
            r.pool.io_done().unwrap();
        }
        open().pool.stash().unwrap().entries(0).iter().filter(|e| e.kind == Kind::Evicted && e.st.history.starts_with(&system)).count()
    };
    assert_eq!(stored(None), 0, "the limit refuses the evicted runs");
    assert!(stored(Some(0)) > 0, "with the limit off, the drain stores them");
    let _ = std::fs::remove_dir_all(&dir);
}

// evict() moves an entry on to a later tier, so a run's base can be in a tier added after the run's: the run waits
// for it (slots held) and joins its tier when the base comes back; loaded() drops a run whose base came back nowhere,
// record and all, so no later boot reads its reused blocks
#[test]
fn a_run_whose_base_is_in_a_later_tier_comes_back_and_a_lost_base_drops_it() {
    use g6kvman::stash::{Entry, Index, Stash};
    let root = std::env::temp_dir().join(format!("g6kv-later-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (a, b) = (root.join("a"), root.join("b"));
    let entry = |id: u64, kind: Kind, slots: Vec<u64>, base: Option<(u64, usize)>| Entry {
        id,
        kind,
        key: id,
        st: Finished { history: toks(id as i32, 64), valid: 64, written_end: 64, pf_end: 64 },
        slots,
        base,
        hits: 1,
        tail: None,
        state: 0,
        namespace: g6kvman::pool::STOCK_NAMESPACE,
    };
    let write = |dir: &std::path::Path, e: &Entry| {
        let (mut ix, _) = Index::open(dir, "later").unwrap();
        ix.add(e).unwrap();
    };
    write(&a, &entry(7, Kind::Evicted, vec![0, 1], Some((3, 2))));
    write(&b, &entry(3, Kind::Prefix, vec![5, 6], None));
    let load = |with_b: bool| {
        let mut s = Stash::new(2, 64);
        s.add_tier(16, Keep::All, Some(Index::open(&a, "later").unwrap()));
        s.index_whole(0); // a run's record is only in a whole index
        assert!(s.entries(0).is_empty(), "the run waits for its base");
        if with_b {
            s.add_tier(16, Keep::All, Some(Index::open(&b, "later").unwrap()));
            s.index_whole(1);
        }
        let gone = s.loaded();
        (s.entries(0).iter().map(|e| e.id).collect::<Vec<_>>(), gone)
    };
    assert_eq!(load(true), (vec![7], 0), "the base in the later tier brings the run back");
    assert_eq!(load(false), (vec![], 1), "no tier brought the base: dropped");
    assert!(Index::open(&a, "later").unwrap().1.is_empty(), "and its record is gone");
    let _ = std::fs::remove_dir_all(&root);
}

// the quit sweep (g6m4.cu): reserve at --cap rows until refused (a live run still holds rows), then
// halving sizes down to one page, which is always tried last, so a retained one-page run is stored too (pool of 12
// pages: 768, 384, 192, 96, then 64 rows). A grid of 64 rows, so a one-page run is storable (one grid cell): 64 rows
// computed, and its history holds the next token
#[test]
fn the_quit_drains_fallback_ends_with_one_page_and_stores_the_last_retained_run() {
    let mut p = Pool::new(12, 12, 12 * 64).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 64 });
    p.stores(2);
    p.add_tier(64, Keep::All, None).unwrap();
    let mut r = Rank::with(p, vec![64]);
    let _live = r.pool.reserve(128, 128).unwrap();
    let run = r.pool.reserve(64, 64).unwrap();
    let tokens = toks(1, 65);
    let list = r.pool.list(&run).unwrap().to_vec();
    r.write(&list, 0, &tokens[..64]);
    r.pool.retain(&run, finished(&tokens)).unwrap();
    r.run_io();
    r.pool.write_limit(0);
    r.run_io();
    let cap = 12 * 64u64;
    let mut held = Vec::new();
    // the sweep starts at --cap rows (one session's; here the whole 12-page pool, which the live run refuses)
    let mut rows = cap.max(64);
    loop {
        loop {
            r.run_io();
            match r.pool.reserve(rows, rows) {
                Ok(x) => held.push(x),
                Err(_) => break,
            }
        }
        if rows == 64 {
            break;
        }
        rows = (rows / 2).max(64);
    }
    r.run_io();
    for x in held {
        r.pool.release(&x).unwrap();
    }
    r.run_io();
    assert_eq!(r.pool.kept_runs(), 0, "no retained run left in the pool");
    assert!(r.pool.stash().unwrap().entries(0).iter().any(|e| e.kind == Kind::Evicted && e.st.history == tokens));
}

// a store whose index was whole at the last boot and is keep=system now: the evicted runs it recorded are dropped at
// load (slots and records), since a prefixes-only index never journals their restore
#[test]
fn a_whole_index_reopened_with_keep_system_drops_its_runs() {
    let dir = std::env::temp_dir().join(format!("g6kv-tosystem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    {
        let mut p = pool(12, 12, 12 * 64, 2);
        p.add_tier(64, Keep::All, Some((dir.as_path(), "to-system"))).unwrap();
        p.stash_mut().unwrap().index_whole(0);
        let mut r = Rank::with(p, vec![64]);
        let system = toks(1, 200);
        for c in 2..4 {
            let mut p = system.clone();
            p.extend(toks(c, 150));
            session(&mut r, &p, 20);
        }
        flush(&mut r);
        r.run_io();
        r.pool.io_done().unwrap();
    }
    let recorded = g6kvman::stash::Index::open(&dir, "to-system").unwrap().1;
    assert!(recorded.iter().any(|e| !e.kind.prefix()), "the whole index recorded runs");
    let mut p = pool(12, 12, 12 * 64, 2);
    p.add_tier(64, Keep::Shared, Some((dir.as_path(), "to-system"))).unwrap();
    let runs = recorded.iter().filter(|e| !e.kind.prefix()).count();
    assert_eq!(p.tiers_loaded(), runs, "every dropped run is counted (the boot line prints it)");
    assert!(p.stash().unwrap().entries(0).iter().all(|e| e.kind.prefix()), "only prefixes load");
    drop(p);
    let left = g6kvman::stash::Index::open(&dir, "to-system").unwrap().1;
    assert!(left.iter().all(|e| e.kind.prefix()) && !left.is_empty(), "the runs' records are gone, the prefixes stay");
    let _ = std::fs::remove_dir_all(&dir);
}

// add_tier drops a record whose slots are past the tier or already claimed (record and all); loaded() counts it, so
// the boot line shows it
#[test]
fn records_with_slots_past_the_tier_or_taken_twice_are_dropped_and_counted() {
    use g6kvman::stash::{Entry, Index, Stash};
    let dir = std::env::temp_dir().join(format!("g6kv-badslots-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let entry = |id: u64, slots: Vec<u64>| Entry {
        id,
        kind: Kind::Prefix,
        key: id,
        st: Finished { history: toks(id as i32, 64), valid: 64, written_end: 64, pf_end: 64 },
        slots,
        base: None,
        hits: 1,
        tail: None,
        state: 0,
        namespace: g6kvman::pool::STOCK_NAMESPACE,
    };
    {
        let (mut ix, _) = Index::open(&dir, "bad").unwrap();
        ix.add(&entry(1, vec![0, 1])).unwrap();
        ix.add(&entry(2, vec![1, 2])).unwrap(); // slot 1 taken twice
        ix.add(&entry(3, vec![20, 21])).unwrap(); // past a 16-slot tier
    }
    let mut s = Stash::new(2, 64);
    s.add_tier(16, Keep::All, Some(Index::open(&dir, "bad").unwrap()));
    assert_eq!(s.loaded(), 2, "both bad records are counted");
    assert_eq!(s.entries(0).iter().map(|e| e.id).collect::<Vec<_>>(), vec![1]);
    assert_eq!(s.loaded(), 0, "counted once");
    let left: Vec<u64> = Index::open(&dir, "bad").unwrap().1.iter().map(|e| e.id).collect();
    assert_eq!(left, vec![1], "their records are gone");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- c21 step 2 (D2: extend in place): a store writes only new blocks ----

// a whole pool reservation that evicts every retained run: its copies
fn evict_all(r: &mut Rank) -> Vec<Io> {
    let n = r.pool.pool_pages() as u64 * PAGE;
    let all = r.pool.reserve(n, n).unwrap();
    let io = r.run_io();
    r.pool.release(&all).unwrap();
    io
}

fn spills(io: &[Io]) -> Vec<(u32, u64)> {
    io.iter().filter_map(|x| if let Io::Spill { page, slot, .. } = *x { Some((page, slot)) } else { None }).collect()
}

// one turn by hand: count, match, take, copies; checks the restored rows; retires the run after `gen` tokens
fn turn(r: &mut Rank, prompt: &[i32], gen: usize) -> (g6kvman::Hit, Vec<u32>, Vec<i32>) {
    r.pool.count(prompt, None);
    let rows = (prompt.len() + gen + 1) as u64;
    let h = r.pool.match_prefix(prompt, rows, rows, prompt.len() as i32 - 1);
    let run = if h.skip > 0 {
        r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), rows, rows).unwrap()
    } else {
        r.pool.reserve(rows, rows).unwrap()
    };
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    for (row, &t) in prompt.iter().enumerate().take(h.skip.max(0) as usize) {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], t, "restored row {row}");
    }
    let mut t = prompt.to_vec();
    t.extend(toks(-1 - prompt.len() as i32, gen));
    r.write(&list, h.skip.max(0) as usize, &t);
    r.pool.retain(&run, finished(&t)).unwrap();
    r.run_io();
    (h, list, t)
}

#[test]
fn a_stored_hit_keeps_its_entry_and_the_next_store_writes_only_new_pages() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    let (_, _, t1) = session(&mut r, &toks(1, 300), 20);
    evict_all(&mut r);
    // 319 valid rows: stored up to the 256 a hit restores (grid 128)
    let e1 = r.pool.stash().unwrap().entries(0)[0].clone();
    assert!(r.pool.stash().unwrap().entries(0).len() == 1 && e1.st.valid == 256 && e1.slots.len() == 4, "{e1:?}");
    let mut p2 = t1.clone();
    p2.extend(toks(2, 100));
    r.pool.count(&p2, None);
    let rows = p2.len() as u64 + 21;
    let h = r.pool.match_prefix(&p2, rows, rows, p2.len() as i32 - 1);
    assert!(h.stored && h.age == e1.id && h.skip == 256, "{h:?}");
    let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), rows, rows).unwrap();
    let io = r.run_io();
    assert!(spills(&io).is_empty());
    assert_eq!(r.pool.stash().unwrap().entries(0), std::slice::from_ref(&e1), "the hit keeps its entry");
    let list = r.pool.list(&run).unwrap().to_vec();
    let mut t2 = p2.clone();
    t2.extend(toks(3, 20));
    r.write(&list, 256, &t2);
    r.pool.retain(&run, finished(&t2)).unwrap();
    r.run_io();
    // evicted with 439 valid rows, 384 storable: only pages 4 and 5 are written, e1's blocks hold pages 0..4
    let sp = spills(&evict_all(&mut r));
    assert_eq!(sp.iter().map(|s| s.0).collect::<Vec<_>>(), list[4..6].to_vec());
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert_eq!(e.len(), 1, "the extended entry replaces e1");
    let e2 = e[0].clone();
    assert!(e2.id != e1.id && e2.st.valid == 384 && e2.st.history == t2 && e2.base == e1.base, "{e2:?}");
    assert_eq!(e2.slots[..4], e1.slots[..]);
    assert_eq!(e2.slots[4..], sp.iter().map(|s| s.1).collect::<Vec<_>>()[..]);
    assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 6, "4 blocks, then 2");
    r.check_slots();
    // the next turn gets all 384 rows back (turn checks them), and its store writes the next new page only
    let mut p3 = t2.clone();
    p3.extend(toks(4, 60));
    let (h, _, _) = turn(&mut r, &p3, 20);
    assert!(h.stored && h.skip == 384, "{h:?}");
    assert_eq!(spills(&evict_all(&mut r)).len(), 2, "pages 6 and 7 (527 valid rows, 512 storable)");
    assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1);
    r.check_slots();
}

#[test]
fn a_pool_hit_on_its_own_run_spills_nothing() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    let (_, _, mut t) = session(&mut r, &toks(1, 300), 20);
    for c in 2..5 {
        t.extend(toks(c, 60));
        let (h, _, t2) = turn(&mut r, &t, 20);
        assert!(h.skip > 0 && !h.stored, "turn {c} takes its own retained run: {h:?}");
        t = t2;
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 0, "turn {c}: the grant holds all a tier would store");
        assert!(r.pool.stash().unwrap().entries(0).is_empty());
    }
    // the last run, evicted: stored once, up to its storable rows (559 valid, 512)
    let n = spills(&evict_all(&mut r)).len();
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert!(e.len() == 1 && e[0].st.history == t && e[0].st.valid == 512 && e[0].slots.len() == 8 && n == 8, "{e:?}");
}

// a run a pool hit takes without spilling it (another conversation's hit holds all a tier would store of it) still
// leaves its head behind: its conversation's next prompt continues it, so min_hits takes the next run
#[test]
fn a_run_a_hit_takes_unspilled_still_continues_its_chat() {
    let mut r = Rank::new(16, 16, 16 * 64, &[(64, Keep::All)], 3);
    r.pool.set_policy(0, Policy { min_rows: 0, min_hits: 1, block_bytes: 4096 }).unwrap();
    let system = toks(1, 200);
    let mut x = system.clone();
    x.extend(toks(2, 20));
    let (_, _, xt) = turn(&mut r, &x, 5); // 224 valid rows: 128 storable
    let mut y = system.clone();
    y.extend(toks(3, 20));
    let (h, _, _) = turn(&mut r, &y, 5);
    assert!(h.skip == 128 && !h.stored, "y takes x's run for the system prompt: {h:?}");
    assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 0, "x's run holds nothing past it: not spilled");
    let mut x2 = xt.clone();
    x2.extend(toks(4, 100));
    assert_eq!(r.pool.count(&x2, None), 0, "x's next prompt continues x");
    let (_, _, x2t) = turn(&mut r, &x2, 20);
    evict_all(&mut r);
    let e = r.pool.stash().unwrap().entries(0).to_vec();
    assert!(e.iter().any(|e| e.kind == Kind::Evicted && e.st.history == x2t && e.hits == 1), "{e:?}");
}

fn whole(dir: &std::path::Path) -> Rank {
    let mut p = pool(16, 16, 16 * 64, 3);
    p.add_tier(64, Keep::All, Some((dir, "d2-test"))).unwrap();
    p.stash_mut().unwrap().index_whole(0);
    assert_eq!(p.tiers_loaded(), 0);
    Rank::with(p, vec![64])
}

// an extended entry's records: the old entry's drop (synced at once) and the new one's add (at io_done). After a
// restart the committed extension is back; an extension whose add never committed (the engine died before io_done)
// loses its session and only that one; a torn record at the end of the log changes nothing
#[test]
fn an_extended_entry_survives_a_restart_and_an_uncommitted_one_loses_only_its_session() {
    let dir = std::env::temp_dir().join(format!("g6kv-d2-ext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (a2, blocks) = {
        let mut r = whole(&dir);
        let (_, _, a) = session(&mut r, &toks(1, 300), 20);
        let (_, _, b) = session(&mut r, &toks(2, 300), 20);
        evict_all(&mut r);
        assert_eq!(r.pool.stash().unwrap().entries(0).len(), 2);
        // a's next turn, evicted: its entry extended in place, committed
        let mut p = a.clone();
        p.extend(toks(3, 100));
        let (h, _, a2) = turn(&mut r, &p, 20);
        assert!(h.stored && h.skip == 256, "{h:?}");
        assert_eq!(spills(&evict_all(&mut r)).len(), 2);
        // b's next turn: its extension is queued, and the engine dies before io_done
        let mut p = b.clone();
        p.extend(toks(4, 100));
        let (h, _, _) = turn(&mut r, &p, 20);
        assert!(h.stored && h.skip == 256, "{h:?}");
        let n = r.pool.pool_pages() as u64 * PAGE;
        r.pool.reserve(n, n).unwrap();
        let io = r.pool.io();
        assert_eq!(spills(&io).len(), 2, "b's extension writes its new pages");
        (a2, r.tiers.clone())
    };
    let check = |r: &Rank| {
        let e = r.pool.stash().unwrap().entries(0).to_vec();
        assert_eq!(e.len(), 1, "a's extension only: {e:?}");
        assert!(e[0].st.history == a2 && e[0].st.valid == 384 && e[0].slots.len() == 6, "{e:?}");
    };
    let mut r = whole(&dir);
    check(&r);
    r.check_slots();
    // a's blocks hold its rows: the next turn restores all 384 (turn checks them)
    r.tiers = blocks.clone();
    let mut p = a2.clone();
    p.extend(toks(5, 10));
    let (h, _, _) = turn(&mut r, &p, 5);
    assert!(h.stored && h.skip == 384, "{h:?}");
    drop(r);
    // a torn record at the end of the log
    let ix = dir.join("index");
    let mut raw = std::fs::read(&ix).unwrap();
    raw.extend_from_slice(&[40, 0, 0, 0, 1, 2, 3]);
    std::fs::write(&ix, raw).unwrap();
    check(&whole(&dir));
    let _ = std::fs::remove_dir_all(&dir);
}

// a hit writes no index record: an engine that dies after a hit, before the run is stored again, finds the entry at
// the next boot with the same blocks
#[test]
fn a_hit_kept_entry_survives_a_crash() {
    let dir = std::env::temp_dir().join(format!("g6kv-d2-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (a, e1, blocks) = {
        let mut r = whole(&dir);
        let (_, _, a) = session(&mut r, &toks(1, 300), 20);
        evict_all(&mut r);
        let e1 = r.pool.stash().unwrap().entries(0)[0].clone();
        (a, e1, r.tiers.clone())
    };
    let next = |a: &[i32]| {
        let mut p = a.to_vec();
        p.extend(toks(2, 100));
        p
    };
    {
        let mut r = whole(&dir);
        r.tiers = blocks.clone();
        let p = next(&a);
        r.pool.count(&p, None);
        let rows = p.len() as u64 + 21;
        let h = r.pool.match_prefix(&p, rows, rows, p.len() as i32 - 1);
        assert!(h.stored && h.skip == 256, "{h:?}");
        r.pool.take_hit(h.skip, h.age, Some(h.tier), rows, rows).unwrap();
        r.run_io();
        // the engine dies here, mid-session
    }
    let mut r = whole(&dir);
    assert_eq!(r.pool.stash().unwrap().entries(0), std::slice::from_ref(&e1), "the entry is back as it was");
    r.tiers = blocks;
    let (h, _, _) = turn(&mut r, &next(&a), 20);
    assert!(h.stored && h.age == e1.id && h.skip == 256, "{h:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_grid_off_partial_page_extension_restores_every_row() {
    let mut p = Pool::new(16, 16, 16 * PAGE as u32).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 0 });
    p.stores(2);
    p.add_tier(64, Keep::All, None).unwrap();
    p.set_policy(0, Policy { min_rows: 0, min_hits: 0, block_bytes: 1 }).unwrap();
    let mut r = Rank::with(p, vec![64]);
    let (_, _, mut chat) = turn(&mut r, &toks(1, 300), 20);
    evict_all(&mut r);
    for i in 2..7 {
        chat.extend(toks(i, 60));
        let (h, _, next) = turn(&mut r, &chat, 20);
        assert!(h.stored && h.skip > 0, "turn {i}: {h:?}");
        evict_all(&mut r);
        r.check_slots();
        assert_eq!(r.pool.stash().unwrap().entries(0).len(), 1, "turn {i}: one extended entry");
        chat = next;
    }
}

// A namespace-aware turn, with bytes stamped by the turn that computed them, not just by token identity.
fn namespaced_turn(
    r: &mut Rank,
    prompt: &[i32],
    namespace: g6kvman::pool::Namespace,
    stamp: i32,
    expected: &mut Vec<i32>,
) -> (g6kvman::Hit, Vec<i32>) {
    r.pool.count_namespaced(prompt, None, namespace);
    let rows = (prompt.len() + 21) as u64;
    let h = r.pool.match_prefix_namespaced(prompt, rows, rows, prompt.len() as i32 - 1, namespace);
    let run = if h.skip > 0 {
        r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), rows, rows).unwrap()
    } else {
        r.pool.reserve(rows, rows).unwrap()
    };
    r.run_io();
    let list = r.pool.list(&run).unwrap().to_vec();
    let skip = h.skip.max(0) as usize;
    for (row, &byte) in expected.iter().enumerate().take(skip) {
        assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], byte, "restored row {row}");
    }
    assert!(skip <= expected.len());
    let mut history = prompt.to_vec();
    history.extend(toks(-1 - prompt.len() as i32, 20));
    expected.truncate(skip);
    expected.extend(history[skip..].iter().map(|t| t + 10000 + stamp * 100000));
    r.write(&list, skip, expected);
    r.pool.retain_namespaced(&run, finished(&history), namespace).unwrap();
    r.run_io();
    (h, history)
}

#[test]
fn a_whole_index_brings_back_a_non_stock_run_only_for_its_own_namespace() {
    use g6kvman::pool::{NAMESPACE_BYTES, STOCK_NAMESPACE};
    use g6kvman::stash::Index;
    let dir = std::env::temp_dir().join(format!("g6kv-whole-namespace-{}", std::process::id()));
    let ns_a = [1; NAMESPACE_BYTES];
    let mut ns_b = ns_a;
    ns_b[NAMESPACE_BYTES - 1] = 2; // exact identity includes the last byte, not just a leading tag
    let open = |whole: bool| {
        let mut p = pool(16, 16, 16 * 64, 3);
        p.add_tier(64, if whole { Keep::Evicted } else { Keep::Shared }, Some((&dir, "namespace-whole"))).unwrap();
        if whole {
            p.stash_mut().unwrap().index_whole(0);
        }
        Rank::with(p, vec![64])
    };
    let mut expected = Vec::new();
    let (blocks, stock, steered) = {
        let mut r = open(true);
        assert_eq!(r.pool.tiers_loaded(), 0);
        let (_, stock) = namespaced_turn(&mut r, &toks(1, 300), STOCK_NAMESPACE, 0, &mut Vec::new());
        let (_, steered) = namespaced_turn(&mut r, &toks(2, 300), ns_a, 1, &mut expected);
        evict_all(&mut r);
        r.check_slots();
        (r.tiers.clone(), stock, steered)
    };
    let entries = Index::open(&dir, "namespace-whole").unwrap().1;
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().any(|e| e.namespace == ns_a && e.kind == Kind::Evicted));
    assert!(entries.iter().any(|e| e.namespace == STOCK_NAMESPACE && e.kind == Kind::Evicted));
    {
        let mut r = open(true);
        r.tiers = blocks;
        assert_eq!(r.pool.tiers_loaded(), 0);
        let rows = steered.len() as u64 + 1;
        let limit = steered.len() as i32 - 1;
        let h = r.pool.match_prefix_namespaced(&steered, rows, rows, limit, ns_a);
        assert!(h.stored && h.skip >= 128, "{h:?}");
        let other = r.pool.match_prefix_namespaced(&steered, rows, rows, limit, ns_b);
        let stock_miss = r.pool.match_prefix(&steered, rows, rows, limit);
        let steered_miss = r.pool.match_prefix_namespaced(&stock, rows, rows, limit, ns_a);
        for miss in [other, stock_miss, steered_miss] {
            assert!(!miss.stored && miss.skip == 0, "{miss:?}");
        }
        r.pool.count_namespaced(&steered, None, ns_a);
        let run = r.pool.take_hit(h.skip, h.age, Some(h.tier), rows, rows).unwrap();
        r.run_io();
        let list = r.pool.list(&run).unwrap();
        for row in 0..h.skip as usize {
            assert_eq!(r.dev[list[row / ROWS] as usize][row % ROWS], expected[row], "row {row}");
        }
        r.pool.release(&run).unwrap();
        r.check_slots();
    }
    // Prefix-only reopen drops both sessions, frees their blocks and durably removes their records.
    {
        let mut r = open(false);
        assert!(r.pool.tier_stats(0).unwrap().free_slots < 64);
        assert_eq!(r.pool.tiers_loaded(), 2);
        assert!(r.pool.stash().unwrap().entries(0).is_empty());
        assert_eq!(r.pool.tier_stats(0).unwrap().free_slots, 64);
        assert_eq!(r.pool.match_prefix_namespaced(&steered, 400, 400, 319, ns_a).skip, 0);
        r.check_slots();
    }
    assert!(Index::open(&dir, "namespace-whole").unwrap().1.is_empty());
    std::fs::remove_file(dir.join("index")).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn steered_session_extends_in_place_across_restarts_writing_only_new_blocks() {
    use g6kvman::pool::NAMESPACE_BYTES;
    use g6kvman::stash::Index;
    let dir = std::env::temp_dir().join(format!("g6kv-steered-extend-{}", std::process::id()));
    let ns_a = [1; NAMESPACE_BYTES];
    let mut ns_b = ns_a;
    ns_b[NAMESPACE_BYTES - 1] = 2;
    let open = || {
        let mut p = pool(16, 16, 16 * 64, 3);
        p.add_tier(64, Keep::Evicted, Some((&dir, "steered-extend"))).unwrap();
        p.set_policy(0, Policy { min_rows: 0, min_hits: 1, block_bytes: 0 }).unwrap();
        p.stash_mut().unwrap().index_whole(0);
        assert_eq!(p.tiers_loaded(), 0);
        Rank::with(p, vec![64])
    };
    let mut expected = Vec::new();
    let (blocks, mut chat, first) = {
        let mut r = open();
        let (_, mut chat) = namespaced_turn(&mut r, &toks(1, 300), ns_a, 1, &mut expected);
        chat.extend(toks(2, 100));
        let (hit, chat) = namespaced_turn(&mut r, &chat, ns_a, 2, &mut expected);
        assert!(!hit.stored && hit.skip >= 128);
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 0);
        let written = spills(&evict_all(&mut r)).len();
        let entries = Index::open(&dir, "steered-extend").unwrap().1;
        assert_eq!(entries.len(), 1);
        let first = entries[0].clone();
        assert_eq!(first.namespace, ns_a);
        assert_eq!(first.slots.len(), 6); // P2 = 384 grid-aligned rows / 64
        assert_eq!(written, first.slots.len());
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, written as u64);
        r.check_slots();
        (r.tiers.clone(), chat, first)
    };
    let (blocks, fourth) = {
        let mut r = open();
        r.tiers = blocks;
        chat.extend(toks(3, 100));
        let (hit, mut third) = namespaced_turn(&mut r, &chat, ns_a, 3, &mut expected);
        assert!(hit.stored && hit.skip == 384, "{hit:?}");
        third.extend(toks(4, 100));
        let (hit, fourth) = namespaced_turn(&mut r, &third, ns_a, 4, &mut expected);
        assert!(!hit.stored && hit.skip == 512, "{hit:?}");
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 0, "both hits leave the SSD untouched");
        let written = spills(&evict_all(&mut r)).len();
        let entries = Index::open(&dir, "steered-extend").unwrap().1;
        assert_eq!(entries.len(), 1);
        let extended = &entries[0];
        assert_eq!(extended.namespace, ns_a);
        assert_ne!(extended.id, first.id);
        assert_eq!(extended.slots.len(), 10); // P4 = 640 grid-aligned rows / 64
        assert_eq!(extended.slots[..first.slots.len()], first.slots);
        let new = extended.slots.len() - first.slots.len();
        assert_eq!(written, new, "only P4 - P2 new blocks, not P4");
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, new as u64);
        assert_eq!(written as f64 / new as f64, 1.0, "write amplification 1.00");
        r.check_slots();
        (r.tiers.clone(), fourth)
    };
    {
        let mut r = open();
        r.tiers = blocks;
        let mut fifth = fourth;
        fifth.extend(toks(5, 100));
        let rows = fifth.len() as u64 + 21;
        let stock = r.pool.match_prefix(&fifth, rows, rows, fifth.len() as i32 - 1);
        let other = r.pool.match_prefix_namespaced(&fifth, rows, rows, fifth.len() as i32 - 1, ns_b);
        for miss in [stock, other] {
            assert!(!miss.stored && miss.skip == 0, "{miss:?}");
        }
        // The row checks cover both the original six blocks and the four-block extension, with distinct turn stamps.
        let (hit, _) = namespaced_turn(&mut r, &fifth, ns_a, 5, &mut expected);
        assert!(hit.stored && hit.skip == 640, "{hit:?}");
        assert_eq!(r.pool.tier_stats(0).unwrap().spilled_pages, 0);
        r.check_slots();
    }
    std::fs::remove_file(dir.join("index")).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn non_stock_live_or_pending_prompt_cannot_split_a_stock_prefix_base() {
    for live in [false, true] {
        let mut r = Rank::new(24, 24, 24 * 64, &[(64, Keep::Evicted)], 3);
        let system = toks(1, 300);
        let mut stock = system.clone();
        stock.extend(toks(2, 200));
        let (_, _, history) = session(&mut r, &stock, 20);
        let mut selected = system;
        selected.extend(toks(3, 200));
        let ns = [1; g6kvman::pool::NAMESPACE_BYTES];
        assert_eq!(r.pool.count_namespaced(&selected, None, ns), 0);
        let held = if live {
            let run = r.pool.reserve(550, 550).unwrap();
            assert!(r.run_io().is_empty());
            Some(run)
        } else {
            None
        };
        // Evict the stock run while the matching non-stock prompt is either pending or held live.
        let rows = if live { 24 * PAGE - 576 } else { 24 * PAGE };
        let pressure = r.pool.reserve(rows, rows).unwrap();
        r.run_io();
        let entries = r.pool.stash().unwrap().entries(0);
        let stored = entries.iter().find(|e| e.st.history == history).unwrap();
        assert!(stored.base.is_none(), "non-stock tokens cannot establish a stock base (live={live})");
        assert!(entries.iter().all(|e| e.kind == Kind::Evicted));
        r.pool.release(&pressure).unwrap();
        if let Some(run) = held {
            r.pool.release(&run).unwrap();
        }
        r.check_slots();
    }
}
