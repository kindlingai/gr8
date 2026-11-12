//! c21: how many KV page blocks a keep=sessions SSD store writes per chat turn, and how many of them hold content the
//! store already wrote once (the same tokens through the page's end). Step 1 measured it on the engine as deployed;
//! step 2 ("D2: extend in place", stash.rs extend) writes only new blocks, and this test asserts it. Each scenario
//! prints a table per turn (run with --nocapture), and the last table puts the step-1 totals beside today's.
//!
//! The store is the deployed one: persist=on ssd with keep=sessions, i.e. one tier keeping all kinds (keep 2) with a
//! whole index (+ 4), policy min_hits 1 (g6m4.cu: pool.add_tier / pool.policy), --kv-system-after 2, the reuse rules
//! the engine sets by default (strict off, noend on, grid = --prefill 4096), no write limit. Every run stays below R
//! (below_rows = the whole pool), so no run has a tail past SHARE_LIMIT. The quit drain is g6m4.cu's sweep after
//! service.run(): write limit off, reserve at the pool's capacity until refused, halving down to one page, release.
//! Every scenario runs two ranks in lock step: the same calls, and they must queue the same copies and agree on the
//! digest after every call.
//!
//! The Rank harness (a model of the device pages and the tier's blocks, run_io in the engine's batch order) is copied
//! from tests/stash.rs (an integration test cannot import another one's items). A row holds its token and the turn
//! that computed it, so a restore that brings back the same tokens computed by another turn shows: the rows a turn
//! restores must equal the rows its conversation's last run held at its retirement.
//!
//! Content identity of a written block: the entry's token history through the last valid row of that page, plus the
//! next token (the MTP layer's row S - 1 depends on token S, as prefix_key counts it). A block is NEW when no block
//! with that identity was written before in the scenario, else REWRITTEN; "on_ssd" counts the rewrites whose identical
//! block is still stored in another slot when the copy is queued. The meter maps every Io::Spill by its slot to one
//! block of an entry the call stored; an entry's other blocks (extended in place) must be blocks the store already
//! held with the same content. After every call every stored block is checked row for row against its entry.

use g6kvman::stash::{Entry, Io, Keep, Kind, Policy};
use g6kvman::{Error, Finished, Pool, Rules, Run, PAGE};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const ROWS: usize = PAGE as usize;
const BATCH: usize = 32;
/// --prefill 4096: the reuse grid the engine sets (scripts/engine_args.sh)
const GRID: i32 = 4096;
const SYSTEM_AFTER: u32 = 2;
const BLOCK_BYTES: u64 = 3_506_176;
const SLOTS: u64 = 1 << 16;
/// a row: its token and the turn that computed it
type Row = (i32, u32);
type Page = [Row; ROWS];
const EMPTY: Row = (-1, 0);

// ---- the harness of tests/stash.rs ----

struct Rank {
    pool: Pool,
    dev: Vec<Page>,
    tiers: Vec<Vec<Page>>,
}

impl Rank {
    fn run_io(&mut self) -> Vec<Io> {
        let io = self.pool.io();
        if !io.is_empty() {
            assert_eq!(self.pool.reserve(1, 1), Err(Error::PendingIo), "a claim before io_done");
        }
        let first_restore = io.iter().position(|x| matches!(x, Io::Restore { .. })).unwrap_or(io.len());
        assert!(io[first_restore..].iter().all(|x| matches!(x, Io::Restore { .. })), "a copy after a restore");
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
                    Io::Move { tier, slot, .. } => self.tiers[tier as usize][slot as usize],
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
    // the prefill and decode of rows [from, tokens.len()), computed by turn `stamp`
    fn write(&mut self, list: &[u32], from: usize, tokens: &[i32], stamp: u32) {
        for (r, &t) in tokens.iter().enumerate().skip(from) {
            self.dev[list[r / ROWS] as usize][r % ROWS] = (t, stamp);
        }
    }
}

fn finished(tokens: &[i32]) -> Finished {
    let valid = tokens.len() as i32 - 1;
    Finished { history: tokens.to_vec(), valid, written_end: valid, pf_end: valid }
}

// the pool as g6m4 boots it: the tier, its whole index, the policy, then tiers_loaded
fn open(dir: &Path, pages: u32, slots: u64) -> Pool {
    let mut p = Pool::new(pages, pages, pages * PAGE as u32).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: GRID });
    p.stores(SYSTEM_AFTER);
    // keep=sessions on a persist=on ssd store: keep 2 (+ 4 whole index), min_hits 1
    p.add_tier(slots, Keep::All, Some((dir, "c21"))).unwrap();
    p.stash_mut().unwrap().index_whole(0);
    p.set_policy(0, Policy { min_rows: 0, min_hits: 1, block_bytes: BLOCK_BYTES }).unwrap();
    p.write_limit(0);
    assert_eq!(p.tiers_loaded(), 0, "every stored run came back with its base");
    p
}

// ---- the meter ----

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn hist_hash(h: &[i32]) -> u64 {
    h.iter().fold(0x6332_315f_6869_7374, |x, &t| mix(x ^ (t as u32 as u64).wrapping_add(0x9E3779B97F4A7C15)))
}

// the identity of each own block of a stored entry (its logical pages after its base)
fn idents(e: &Entry) -> Vec<u64> {
    let h = &e.st.history;
    let mut ph = Vec::with_capacity(h.len() + 1);
    let mut x = 0x6332_315f_7761_6d70u64;
    ph.push(x);
    for &t in h {
        x = mix(x ^ (t as u32 as u64).wrapping_add(0x9E3779B97F4A7C15));
        ph.push(x);
    }
    let base = e.base.map_or(0, |b| b.1);
    let valid = e.st.valid.max(0) as usize;
    (0..e.slots.len())
        .map(|k| {
            let end = (ROWS * (base + k + 1)).min(valid);
            ph[(end + 1).min(h.len())]
        })
        .collect()
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Tally {
    wr: u64,
    new: u64,
    rewr: u64,
    on_ssd: u64,
    other: u64, // of wr: blocks of the one-shot traffic (expected 0: min_hits refuses them)
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.wr += o.wr;
        self.new += o.new;
        self.rewr += o.rewr;
        self.on_ssd += o.on_ssd;
        self.other += o.other;
    }
    fn amp(&self) -> f64 {
        self.wr as f64 / self.new.max(1) as f64
    }
}

/// What a scenario measured: the totals of its serving and quit, the store after the quit, and the restart phase.
#[derive(Clone, Default, Debug)]
struct Totals {
    serve: Tally,
    entries: u32,
    blocks: u64,
    distinct: usize,
    restart: Option<Tally>,
    // rewritten blocks by the label of the run whose entry holds them
    rewr_by: Vec<(String, u64)>,
}

fn gb(blocks: u64) -> f64 {
    (blocks * BLOCK_BYTES) as f64 / 1e9
}

/// Tokens of one-shot pressure traffic start here (seed 20000 and up); a chat's tokens stay below.
const FILLER_SEED: i32 = 20_000;
/// The chat template's user-turn header: the token after the system span, the same in every chat (C8, D)
const USER_HEAD: i32 = 7;

struct Sim {
    ranks: Vec<Rank>, // rank 0 decides; every rank gets the same calls
    dirs: Vec<PathBuf>,
    pages: u32,
    slots: u64,
    ever: HashSet<u64>,
    live: HashMap<u64, Vec<u64>>, // the stored entries: id -> the identity of each own block
    slot_of: HashMap<u64, u64>,   // slot -> the identity of the block it holds
    win: Tally,
    stored: Vec<String>, // the entries stored in this window
    seed: i32,
    filler_seed: i32,
    stamp: u32,
    out: String,
    cum: Tally,
    snaps: HashMap<usize, Vec<Row>>, // conversation -> the rows its last run held at its retirement
    labels: HashMap<u64, String>,    // history hash -> the label of the turn whose run it is
    rewr_by: HashMap<String, u64>,
    head: bool, // user messages start with USER_HEAD
    shared: Option<u64>, // the shared system prefix, once stored (scenario D watches it)
}

fn kind_tag(k: Kind) -> &'static str {
    match k {
        Kind::Evicted => "E",
        Kind::Shared => "S",
        Kind::Prefix => "P",
    }
}

impl Sim {
    fn new(name: &str, pages: u32, slots: u64) -> Sim {
        let dirs: Vec<PathBuf> = (0..2)
            .map(|r| std::env::temp_dir().join(format!("c21-wamp-{name}-{}-r{r}", std::process::id())))
            .collect();
        let ranks = dirs
            .iter()
            .map(|d| {
                let _ = std::fs::remove_dir_all(d);
                Rank {
                    pool: open(d, pages, slots),
                    dev: vec![[EMPTY; ROWS]; pages as usize],
                    tiers: vec![vec![[(-2, 0); ROWS]; slots as usize]],
                }
            })
            .collect();
        Sim {
            ranks,
            dirs,
            pages,
            slots,
            ever: HashSet::new(),
            live: HashMap::new(),
            slot_of: HashMap::new(),
            win: Tally::default(),
            stored: Vec::new(),
            seed: 1,
            filler_seed: FILLER_SEED,
            stamp: 0,
            out: String::new(),
            cum: Tally::default(),
            snaps: HashMap::new(),
            labels: HashMap::new(),
            rewr_by: HashMap::new(),
            head: false,
            shared: None,
        }
    }

    // one call on every rank: the same result, and the same digest after it
    fn each<T: PartialEq + std::fmt::Debug>(&mut self, what: &str, f: impl Fn(&mut Rank) -> T) -> T {
        let mut it = self.ranks.iter_mut();
        let a = f(it.next().expect("a rank"));
        for r in it {
            assert_eq!(f(r), a, "{what}: the ranks differ");
        }
        let d = self.ranks[0].pool.digest();
        assert!(self.ranks.iter().all(|r| r.pool.digest() == d), "{what}: the ranks' digests differ");
        a
    }

    fn fresh(&mut self, n: usize) -> Vec<i32> {
        self.seed += 1;
        assert!(self.seed < FILLER_SEED);
        (0..n as i32).map(|i| self.seed * 100_000 + i).collect()
    }

    fn user(&mut self, n: usize) -> Vec<i32> {
        if !self.head {
            return self.fresh(n);
        }
        let mut u = vec![USER_HEAD];
        u.extend(self.fresh(n - 1));
        u
    }

    // the engine runs the queued copies (every rank); the meter classifies every block the store wrote and checks the
    // store row for row
    fn io(&mut self) {
        let io = self.each("io", |r| r.run_io());
        assert!(!io.iter().any(|x| matches!(x, Io::Move { .. })), "one tier: no moves");
        let r = &self.ranks[0];
        let s = r.pool.stash().unwrap();
        let entries: Vec<Entry> = s.entries(0).to_vec();
        let st = s.stats(0).unwrap();
        // no slot held twice, every base present, no slot lost
        let mut owner: HashMap<u64, (usize, usize)> = HashMap::new();
        for (i, e) in entries.iter().enumerate() {
            if let Some((b, n)) = e.base {
                assert!(entries.iter().any(|x| x.id == b && x.slots.len() == n), "entry {} lost its base {b}", e.id);
            }
            for (k, &slot) in e.slots.iter().enumerate() {
                assert!(owner.insert(slot, (i, k)).is_none(), "slot {slot} held twice");
            }
        }
        assert_eq!(owner.len() as u64 + st.free_slots, st.slots, "a slot lost");
        let ids: Vec<Vec<u64>> = entries.iter().map(idents).collect();
        let fresh: HashSet<usize> = (0..entries.len()).filter(|&i| !self.live.contains_key(&entries[i].id)).collect();
        let mut spilled: HashSet<u64> = HashSet::new();
        for x in &io {
            if let Io::Spill { tier, slot, .. } = *x {
                assert_eq!(tier, 0);
                assert!(spilled.insert(slot), "slot {slot} written twice in one call");
            }
        }
        // what the store holds apart from this call's writes: a rewrite of one of these is "on_ssd"
        let kept: HashSet<u64> =
            owner.iter().filter(|(slot, _)| !spilled.contains(slot)).map(|(_, &(i, k))| ids[i][k]).collect();
        for &slot in &spilled {
            let &(i, k) = owner.get(&slot).expect("a spill to a slot no entry holds");
            let e = &entries[i];
            assert!(fresh.contains(&i), "a spill into entry {} the call did not store", e.id);
            let id = ids[i][k];
            self.win.wr += 1;
            if e.st.history.first().is_some_and(|&t| t >= FILLER_SEED * 100_000) {
                self.win.other += 1;
            }
            if self.ever.insert(id) {
                self.win.new += 1;
            } else {
                self.win.rewr += 1;
                if kept.contains(&id) {
                    self.win.on_ssd += 1;
                }
                let label = self.labels.get(&hist_hash(&e.st.history)).cloned().unwrap_or_else(|| "?".into());
                *self.rewr_by.entry(label).or_default() += 1;
            }
        }
        // an entry's blocks this call did not write are blocks the store held before, with the same content
        for &i in &fresh {
            let e = &entries[i];
            let mut reused = 0;
            for (k, slot) in e.slots.iter().enumerate() {
                if !spilled.contains(slot) {
                    assert_eq!(self.slot_of.get(slot), Some(&ids[i][k]), "entry {} block {k} holds other content", e.id);
                    reused += 1;
                }
            }
            let base = e.base.map_or(String::new(), |b| format!("{}+", b.1));
            let reused = if reused > 0 { format!("^{reused}") } else { String::new() };
            self.stored.push(format!("{}[{}{}{}]", kind_tag(e.kind), base, e.slots.len(), reused));
        }
        // every stored block, row for row (on every rank: they must hold the same tokens)
        for r in &self.ranks {
            for e in &entries {
                let b = e.base.map_or(0, |b| b.1);
                for (k, &slot) in e.slots.iter().enumerate() {
                    let lp = b + k;
                    for row in lp * ROWS..((lp + 1) * ROWS).min(e.st.valid.max(0) as usize) {
                        assert_eq!(r.tiers[0][slot as usize][row % ROWS].0, e.st.history[row], "entry {} row {row}", e.id);
                    }
                }
            }
        }
        if let Some(sh) = self.shared {
            assert!(entries.iter().any(|e| e.id == sh), "the shared system prefix left the store");
        } else {
            self.shared = entries.iter().find(|e| e.kind == Kind::Shared).map(|e| e.id);
        }
        self.live = entries.iter().zip(&ids).map(|(e, v)| (e.id, v.clone())).collect();
        self.slot_of = owner.iter().map(|(&slot, &(i, k))| (slot, ids[i][k])).collect();
    }

    // one admission and retirement as g6m4 makes them: count, match, take or reserve, copies, prefill + decode
    // (`gen` tokens), retain. `conv`: the conversation (None: one-shot traffic). Returns the run's history, where its
    // prefix came from, and the hit.
    fn turn(
        &mut self,
        conv: Option<usize>,
        label: &str,
        prompt: &[i32],
        gen: usize,
        span: Option<u32>,
    ) -> (Vec<i32>, String, g6kvman::Hit) {
        let answer = if conv.is_none() {
            self.filler_seed += 1;
            (0..gen as i32).map(|i| self.filler_seed * 100_000 + i).collect()
        } else {
            self.fresh(gen)
        };
        self.stamp += 1;
        let stamp = self.stamp;
        self.each("count", |r| r.pool.count(prompt, span));
        let rows = (prompt.len() + gen + 1) as u64;
        let h = self.each("match", |r| r.pool.match_prefix(prompt, rows, rows, prompt.len() as i32 - 1));
        let how = if h.skip <= 0 {
            "-".to_string()
        } else if h.stored {
            let s = self.ranks[0].pool.stash().unwrap();
            let e = s.entries(h.tier as usize).iter().find(|e| e.id == h.age).unwrap();
            let k = match e.kind {
                Kind::Evicted if prompt.starts_with(&e.st.history[..e.st.valid as usize]) => "run",
                Kind::Evicted => "run*",
                Kind::Shared => "shared",
                Kind::Prefix => "prefix",
            };
            format!("ssd:{k} {}", h.skip)
        } else {
            let k = self.ranks[0].pool.kept().iter().find(|k| k.age == h.age).unwrap();
            let own = prompt.starts_with(&k.st.history[..k.st.valid as usize]);
            format!("pool:{} {}", if own { "own" } else { "other" }, h.skip)
        };
        let run: Run = self.each("take", |r| {
            if h.skip > 0 {
                r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), rows, rows).unwrap()
            } else {
                r.pool.reserve(rows, rows).unwrap()
            }
        });
        self.io();
        let list = self.each("list", |r| r.pool.list(&run).unwrap().to_vec());
        // the restored rows: the prompt's tokens, and the very rows the conversation's last run held
        let snap = conv.and_then(|c| self.snaps.get(&c));
        let dev = &self.ranks[0].dev;
        for row in 0..h.skip.max(0) as usize {
            let got = dev[list[row / ROWS] as usize][row % ROWS];
            assert_eq!(got.0, prompt[row], "{label}: restored row {row}");
            if let Some(s) = snap.filter(|s| row < s.len()) {
                assert_eq!(got, s[row], "{label}: restored row {row} is not the row its conversation's run held");
            }
        }
        let mut t = prompt.to_vec();
        t.extend(answer);
        let skip = h.skip.max(0) as usize;
        self.each("write", |r| r.write(&list, skip, &t, stamp));
        if let Some(c) = conv {
            let dev = &self.ranks[0].dev;
            self.snaps.insert(c, (0..t.len() - 1).map(|row| dev[list[row / ROWS] as usize][row % ROWS]).collect());
        }
        self.labels.insert(hist_hash(&t), label.to_string());
        let fin = finished(&t);
        self.each("retain", |r| r.pool.retain(&run, fin.clone()).unwrap());
        self.io();
        (t, how, h)
    }

    fn resident(&self, hist: &[i32]) -> bool {
        self.ranks[0].pool.kept().iter().any(|k| k.st.history == hist)
    }

    // one-shot requests (other users on a busy server) until the chat's retained run has left the pool
    fn press(&mut self, hist: &[i32]) -> usize {
        let mut n = 0;
        while self.resident(hist) {
            self.filler_seed += 1;
            let p: Vec<i32> = (0..24_000).map(|i| self.filler_seed * 100_000 + i).collect();
            self.turn(None, "filler", &p, 200, None);
            n += 1;
            assert!(n < 50, "the pressure never evicted the chat");
        }
        n
    }

    // g6m4.cu after service.run(): write limit off, then the sweep
    fn quit(&mut self) {
        self.each("limit", |r| r.pool.write_limit(0));
        self.io();
        let cap = self.ranks[0].pool.pool_pages() as u64 * PAGE;
        let mut held = Vec::new();
        let mut rows = cap.max(64);
        loop {
            loop {
                self.io();
                match self.each("reserve", |r| r.pool.reserve(rows, rows)) {
                    Ok(x) => held.push(x),
                    Err(_) => break,
                }
            }
            if rows == 64 {
                break;
            }
            rows = (rows / 2).max(64);
        }
        self.io();
        for x in held {
            self.each("release", |r| r.pool.release(&x).unwrap());
        }
        self.io();
        assert_eq!(self.ranks[0].pool.kept_runs(), 0, "the drain left a retained run");
    }

    // a new process on the same store: every rank's pool comes back from its index, the blocks stay
    fn restart(&mut self) {
        let summary = self.ranks[0].pool.stored_summary();
        for (r, dir) in self.ranks.iter_mut().zip(&self.dirs) {
            assert_eq!(r.pool.stored_summary(), summary, "the ranks' stores differ before the restart");
            let tiers = std::mem::take(&mut r.tiers);
            *r = Rank { pool: open(dir, self.pages, self.slots), dev: vec![[EMPTY; ROWS]; self.pages as usize], tiers };
            assert_eq!(r.pool.stored_summary(), summary, "the index brought back another store");
        }
        self.each("keep_agreed", |r| r.pool.keep_agreed(&summary));
        assert_eq!(self.ranks[0].pool.stored_summary(), summary);
        let _ = writeln!(self.out, "restart: {} lines of stored_summary agree", summary.lines().count());
    }

    fn header(&mut self, title: &str) {
        let _ = writeln!(self.out, "\n{title}");
        let _ = writeln!(
            self.out,
            "{:<9} {:>6} {:<17} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7} {:>8}  stored (kind[base+own^reused])",
            "turn", "prompt", "hit", "wr", "new", "rewr", "on_ssd", "cum_wr", "cum_new", "wr_GB", "cum_GB"
        );
    }

    // closes the window: one table row
    fn row(&mut self, label: &str, prompt: usize, how: &str) {
        let w = std::mem::take(&mut self.win);
        self.cum.add(&w);
        let stored = std::mem::take(&mut self.stored).join(" ");
        let _ = writeln!(
            self.out,
            "{:<9} {:>6} {:<17} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7.2} {:>8.2}  {}{}",
            label,
            if prompt > 0 { prompt.to_string() } else { "-".into() },
            how,
            w.wr,
            w.new,
            w.rewr,
            w.on_ssd,
            self.cum.wr,
            self.cum.new,
            gb(w.wr),
            gb(self.cum.wr),
            stored,
            if w.other > 0 { format!("  (other traffic: {} blocks)", w.other) } else { String::new() }
        );
    }

    // the totals of serving and the quit, and what the store holds for the next boot
    fn footer(&mut self) -> Totals {
        let c = std::mem::take(&mut self.cum);
        let _ = writeln!(
            self.out,
            "total: written {} blocks ({:.2} GB), new {} ({:.2} GB), rewritten {} (still on the SSD when rewritten: {}), \
             write amplification written/new = {:.2}",
            c.wr,
            gb(c.wr),
            c.new,
            gb(c.new),
            c.rewr,
            c.on_ssd,
            c.amp()
        );
        let st = self.ranks[0].pool.tier_stats(0).unwrap();
        let distinct: HashSet<u64> = self.live.values().flatten().copied().collect();
        let _ = writeln!(
            self.out,
            "on the SSD after the quit: {} entries, {} blocks, {} distinct contents",
            st.entries,
            st.slots - st.free_slots,
            distinct.len()
        );
        let mut rewr_by: Vec<(String, u64)> = std::mem::take(&mut self.rewr_by).into_iter().collect();
        rewr_by.sort();
        if !rewr_by.is_empty() {
            let _ = writeln!(self.out, "rewritten blocks by the run that holds them: {rewr_by:?}");
        }
        Totals {
            serve: c,
            entries: st.entries,
            blocks: st.slots - st.free_slots,
            distinct: distinct.len(),
            restart: None,
            rewr_by,
        }
    }

    // after the quit: a restart, one more turn of every session (each restores to its stored cut), and the quit
    fn restart_phase(&mut self, sessions: &[(Vec<i32>, Option<u32>)], names: &[String], t: &mut Totals) {
        self.restart();
        for (c, (hist, span)) in sessions.iter().enumerate() {
            let mut p = hist.clone();
            let u = self.user(USER);
            p.extend(u);
            let s = self.ranks[0].pool.stash().unwrap();
            let e = s.entries(0).iter().find(|e| e.kind == Kind::Evicted && p.starts_with(&e.st.history)).cloned();
            let e = e.expect("the session's run is stored");
            let label = format!("r:{}", names[c]);
            let (_, how, h) = self.turn(Some(c), &label, &p, ANSWER, *span);
            assert!(h.stored && h.age == e.id && h.skip == e.st.valid, "{label}: restores to its cut {}: {h:?}", e.st.valid);
            self.row(&label, p.len(), &how);
        }
        self.quit();
        self.row("r:quit", 0, "");
        let c = std::mem::take(&mut self.cum);
        let _ = writeln!(
            self.out,
            "restart phase: written {} blocks, new {}, rewritten {} (still on the SSD: {})",
            c.wr, c.new, c.rewr, c.on_ssd
        );
        t.restart = Some(c);
    }
}

impl Drop for Sim {
    fn drop(&mut self) {
        for d in &self.dirs {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

const SYSTEM: usize = 10_000; // an agent's system prompt and tool definitions (the front end's system span)
const USER: usize = 2_000;
const ANSWER: usize = 500;

// one chat of `turns` turns; `pressure`: one-shot traffic evicts the chat's run between turns
fn scenario_a(pressure: bool, pages: u32) -> (String, Totals) {
    let mut s = Sim::new(if pressure { "a" } else { "a0" }, pages, SLOTS);
    s.header(&format!(
        "Scenario {}: one chat, 10 turns (system {SYSTEM} + {USER} user per turn, {ANSWER} answer), {}; pool {pages} \
         pages, grid {GRID}",
        if pressure { "A" } else { "A0" },
        if pressure {
            "one-shot requests of 24,000 tokens evict its run between turns"
        } else {
            "no other traffic (its run stays in the pool)"
        }
    ));
    let mut hist = s.fresh(SYSTEM);
    let mut fillers = 0;
    for turn in 1..=10 {
        if pressure && turn > 1 {
            fillers += s.press(&hist);
        }
        let mut p = hist.clone();
        p.extend(s.user(USER));
        let (t, how, _) = s.turn(Some(0), &format!("{turn}"), &p, ANSWER, Some(SYSTEM as u32));
        hist = t;
        s.row(&format!("{turn}"), p.len(), &how);
    }
    s.quit();
    s.row("quit", 0, "");
    if pressure {
        let _ = writeln!(s.out, "({fillers} one-shot requests between the turns)");
    }
    let mut t = s.footer();
    s.restart_phase(&[(hist, Some(SYSTEM as u32))], &["1".into()], &mut t);
    (std::mem::take(&mut s.out), t)
}

// two long chats that evict each other: the pool holds one of them
fn scenario_b() -> (String, Totals) {
    let pages = 640;
    let mut s = Sim::new("b", pages, SLOTS);
    let first = 12_000;
    s.header(&format!(
        "Scenario B: two chats (system 8000 + a first message of {first}, then {USER} user per turn, {ANSWER} answer), 8 \
         turns each, alternating; pool {pages} pages ({} rows: one chat fits, two do not), grid {GRID}",
        pages * 64
    ));
    let mut hist: Vec<Vec<i32>> = (0..2).map(|_| s.fresh(8000)).collect();
    for turn in 1..=8 {
        for (c, name) in ["X", "Y"].iter().enumerate() {
            let mut p = hist[c].clone();
            p.extend(s.user(if turn == 1 { first } else { USER }));
            let (t, how, _) = s.turn(Some(c), &format!("{name}{turn}"), &p, ANSWER, Some(8000));
            hist[c] = t;
            s.row(&format!("{name}{turn}"), p.len(), &how);
        }
    }
    s.quit();
    s.row("quit", 0, "");
    let mut t = s.footer();
    let sessions: Vec<_> = hist.into_iter().map(|h| (h, Some(8000))).collect();
    s.restart_phase(&sessions, &["X".into(), "Y".into()], &mut t);
    (std::mem::take(&mut s.out), t)
}

// four chats on one system prompt, round robin: each chat's run is evicted before its next turn. `system`: its tokens
// (C: 10,000, past the 8,192-row grid cell; C8: 8,192, exactly two cells, and every user message starts with the chat
// template's user header, the same token)
fn scenario_c(system: usize) -> (String, Totals) {
    let pages = 640;
    let name = if system == SYSTEM { "C".to_string() } else { format!("C{}", system / 1024) };
    let mut s = Sim::new(&name.to_lowercase(), pages, SLOTS);
    s.head = system % GRID as usize == 0;
    s.header(&format!(
        "Scenario {name}: four chats sharing one system prompt ({system}{}), {USER} user per turn, {ANSWER} answer, 5 \
         turns each, round robin; pool {pages} pages ({} rows), grid {GRID}",
        if s.head { ", then the same user header token" } else { "" },
        pages * 64
    ));
    let sys = s.fresh(system);
    let mut hist: Vec<Vec<i32>> = vec![sys.clone(); 4];
    for turn in 1..=5 {
        for (c, h) in hist.iter_mut().enumerate() {
            let mut p = h.clone();
            p.extend(s.user(USER));
            let label = format!("c{}.{turn}", c + 1);
            let (t, how, _) = s.turn(Some(c), &label, &p, ANSWER, Some(system as u32));
            *h = t;
            s.row(&label, p.len(), &how);
        }
    }
    s.quit();
    s.row("quit", 0, "");
    let mut t = s.footer();
    let sessions: Vec<_> = hist.into_iter().map(|h| (h, Some(system as u32))).collect();
    let names: Vec<String> = (1..=4).map(|c| format!("c{c}")).collect();
    s.restart_phase(&sessions, &names, &mut t);
    (std::mem::take(&mut s.out), t)
}

// a full store: six chats on one 8,192-token system prompt, round robin, in a store of 700 blocks (the shared prefix
// and about two chats). The store makes room by its least recently used entries, leaves first
fn scenario_d() -> (String, Totals) {
    let (pages, slots, system) = (640, 700, 8192);
    let mut s = Sim::new("d", pages, slots);
    s.head = true;
    s.header(&format!(
        "Scenario D: six chats sharing one system prompt ({system}, then the same user header token), {USER} user per \
         turn, {ANSWER} answer, 8 turns each, round robin; pool {pages} pages, a store of {slots} blocks, grid {GRID}"
    ));
    let sys = s.fresh(system);
    let mut hist: Vec<Vec<i32>> = vec![sys.clone(); 6];
    for turn in 1..=8 {
        for (c, h) in hist.iter_mut().enumerate() {
            let mut p = h.clone();
            p.extend(s.user(USER));
            let label = format!("d{}.{turn}", c + 1);
            let (t, how, _) = s.turn(Some(c), &label, &p, ANSWER, Some(system as u32));
            *h = t;
            s.row(&label, p.len(), &how);
        }
    }
    s.quit();
    s.row("quit", 0, "");
    let t = s.footer();
    let st = s.ranks[0].pool.tier_stats(0).unwrap();
    let _ = writeln!(s.out, "store: full {} times, dropped {} entries ({} blocks)", st.full, st.dropped, st.dropped_pages);
    assert!(st.dropped > 0, "D: the store never had to make room");
    assert!(s.shared.is_some(), "D: the shared system prefix was stored");
    (std::mem::take(&mut s.out), t)
}

/// Step 1's totals (c21 step-1 report, 53340b9): written, new, rewritten, of them still on the SSD. C8 and D are
/// step 2's scenarios, measured on 53340b9 with this test file (the step-2 assertions removed); that run also gave the
/// step-1 report's A, A0, B and C totals exactly.
const STEP1: [(&str, u64, u64, u64, u64); 6] = [
    ("A", 3519, 555, 2964, 0),
    ("A0", 1983, 555, 1428, 1428),
    ("B", 6678, 1200, 5478, 0),
    ("C", 2776, 952, 1824, 420),
    ("C8", 2328, 924, 1404, 0),
    ("D", 8360, 2042, 6318, 0),
];

#[test]
fn c21_write_amplification_per_turn() {
    // the TP=4 pool (KNOBS.md: 2,280 pages of 64 tokens a rank)
    let runs: Vec<(&str, (String, Totals))> = vec![
        ("A", scenario_a(true, 2280)),
        ("A0", scenario_a(false, 2280)),
        ("B", scenario_b()),
        ("C", scenario_c(SYSTEM)),
        ("C8", scenario_c(8192)),
        ("D", scenario_d()),
    ];
    let mut out: String = runs.iter().map(|r| r.1 .0.as_str()).collect();
    let _ = writeln!(
        out,
        "\nblock = {BLOCK_BYTES} B; GB = 1e9 B. wr: blocks the store wrote (Io::Spill); new: content never written before \
         in the scenario; rewr: content written before; on_ssd: of rewr, an identical block still stored. A turn's row \
         counts the copies from the end of the previous row to its retirement (the pressure before it included). \
         hit: where the prompt's first rows came from (pool:own/other retained run, ssd:run/shared/prefix entry) and \
         the rows. stored: entries stored in the row, E evicted run, P prefix base, S shared system prefix, \
         [base pages + own blocks ^ of them the replaced entry's, extended in place, not written]. r: rows are the \
         restart phase (a new process on the same store, one more turn per session, the quit)."
    );
    let _ = writeln!(out, "\nc21 totals: step 1 (53340b9) beside step 2 (D2: extend in place)");
    let _ = writeln!(
        out,
        "{:<4} | {:>6} {:>6} {:>6} {:>6} {:>5} | {:>6} {:>6} {:>6} {:>6} {:>5} | {:>7} {:>6} {:>8} | {:>9}",
        "", "wr", "new", "rewr", "on_ssd", "amp", "wr", "new", "rewr", "on_ssd", "amp", "entries", "blocks", "distinct",
        "restart"
    );
    let _ = writeln!(out, "{:<4} | {:^34} | {:^34} | {:^24} | {:>9}", "", "step 1", "step 2", "SSD after the quit", "wr/new/rewr");
    for ((name, (_, t)), s1) in runs.iter().zip(STEP1) {
        assert_eq!(*name, s1.0);
        let c = t.serve;
        let s1s = if s1.1 > 0 {
            format!("{:>6} {:>6} {:>6} {:>6} {:>5.2}", s1.1, s1.2, s1.3, s1.4, s1.1 as f64 / s1.2 as f64)
        } else {
            format!("{:>34}", "-")
        };
        let rs = t.restart.map_or("-".to_string(), |r| format!("{}/{}/{}", r.wr, r.new, r.rewr));
        let _ = writeln!(
            out,
            "{name:<4} | {s1s} | {:>6} {:>6} {:>6} {:>6} {:>5.2} | {:>7} {:>6} {:>8} | {rs:>9}",
            c.wr,
            c.new,
            c.rewr,
            c.on_ssd,
            c.amp(),
            t.entries,
            t.blocks,
            t.distinct
        );
    }
    println!("{out}");

    let get = |n: &str| runs.iter().find(|r| r.0 == n).map(|r| r.1 .1.clone()).unwrap();
    // A, A0, B: every block written once, all of it still on the SSD, one entry per session
    for (n, sessions) in [("A", 1), ("A0", 1), ("B", 2)] {
        let t = get(n);
        let c = t.serve;
        assert_eq!((c.rewr, c.on_ssd, c.other), (0, 0, 0), "{n}: {c:?}");
        assert_eq!(c.wr, c.new, "{n}: write amplification 1.00");
        assert_eq!(c.wr, t.blocks, "{n}: every block written is still stored");
        assert_eq!(t.blocks, t.distinct as u64, "{n}: no block stored twice");
        assert_eq!(t.entries, sessions, "{n}: one entry per session");
    }
    // what the store must hold: the storable rows of each session's last run (grid cells of 4096 rows)
    assert_eq!(get("A").serve.wr, 512, "A: 32768 rows");
    assert_eq!(get("A0").serve.wr, 512, "A0: 32768 rows");
    assert_eq!(get("B").serve.wr, 2 * 576, "B: 36864 rows a chat");
    // C: the system rows past the 8192-row shared prefix (8192..9983, 28 blocks) are in every chat's own blocks; the
    // first store of each chat after the first writes them again (runs c2.2, c3.2, c4.2), nothing else
    let c = get("C");
    assert_eq!(c.serve.rewr, 84, "C: {:?}", c.serve);
    assert_eq!(c.serve.on_ssd, 84, "C: the 28 blocks are on the SSD (chat 1's)");
    assert_eq!(c.rewr_by, vec![("c2.2".into(), 28), ("c3.2".into(), 28), ("c4.2".into(), 28)], "C: who rewrites");
    assert_eq!(c.serve.wr, 128 + 4 * 192, "C: the shared prefix and 192 own blocks a chat");
    assert_eq!((c.blocks, c.distinct as u64 + 84), (c.serve.wr, c.serve.wr), "C: 84 duplicates stored");
    // C8: a system prompt of whole grid cells is shared whole: nothing is written twice
    let c8 = get("C8");
    assert_eq!((c8.serve.rewr, c8.serve.on_ssd), (0, 0), "C8: {:?}", c8.serve);
    assert_eq!(c8.serve.wr, c8.blocks, "C8: every block written is still stored");
    // the restart phase: every session restored to its cut, and wrote only new blocks
    for n in ["A", "A0", "B", "C", "C8"] {
        let r = get(n).restart.expect("a restart phase");
        assert_eq!((r.rewr, r.on_ssd), (0, 0), "{n} restart: {r:?}");
        assert_eq!(r.wr, r.new, "{n} restart");
    }
    assert_eq!(get("A").restart.unwrap().wr, 64, "A restart: 36864 rows now");
    // D: a full store. Every rewrite is of a block the store had dropped (none still on the SSD); slots, bases and the
    // shared prefix were checked after every call
    let d = get("D");
    assert_eq!(d.serve.on_ssd, 0, "D: {:?}", d.serve);
}
