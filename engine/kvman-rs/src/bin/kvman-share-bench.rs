//! kvman-share-bench: two clusters share one KV node, and capacity moves to whichever needs it (REMOTE-TIER.md,
//! "Sharing a node"). Each cluster is `--ranks` ranks, each with its own pool (src/pool.rs) and one store tier on its
//! own group lease of the node, all driven in lock step from one thread, as the engine's ranks are. The two clusters
//! use the block sizes of GLM 5.3 full (FP8 rows) and of the INT4-row build, so they share a node with blocks of two
//! sizes. Each cluster's head reports to the node every --poll-ms, and grows or shrinks its ranks' leases and tiers at
//! one point in the cluster's call sequence when the node says so.
//!
//!   kvman-share-bench [--server HOST:PORT] [--gb 4] [--seg-mb 128] [--margin-s 60] [--poll-ms 1000]
//!                     [--phase-s 150] [--ranks 2] [--agents 24] [--transport tcp|rdma|auto] [--json FILE]
//!                     [--clock pool|bench] [--seed N]
//!
//! Without --server it runs a node in this process (TCP on loopback). The run is four phases: A busy and B idle, B busy
//! and A idle, both busy, A busy and B idle. A busy cluster runs agent conversations whose KV does not fit its pool, so
//! retained runs spill to the store and come back from it. Every block a rank writes carries its cluster, its rank and
//! the tokens of its page, with a filler derived from them: every block read back is checked whole, and every reused row
//! must hold its prompt's token. The ranks of a cluster must queue the same copies and agree on the pool digest after
//! every call. One line of JSON per head poll (the timeline), and a summary at the end.

use g6kvman::remote::{coldest_ages, Config, GroupOpts, Node, Opts, RemoteStore, Share, Want};
use g6kvman::stash::{Io, Keep, Policy};
use g6kvman::tier::{page_slot, BlockStore, MLA_ROW, MLA_ROW_KV4};
use g6kvman::{Finished, Pool, Rules, Run, PAGE};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ROWS: usize = PAGE as usize;
const MAGIC: u64 = 0x5348_4152_4542_4e43; // "SHAREBNC"
const POOL_PAGES: u32 = 48;

fn usage() -> ! {
    eprintln!(
        "usage: kvman-share-bench [--server HOST:PORT] [--gb 4] [--seg-mb 128] [--margin-s 60] [--poll-ms 1000] \
         [--phase-s 150] [--ranks 2] [--agents 24] [--transport tcp|rdma|auto] [--json FILE] [--clock pool|bench] [--seed N] \
         [--only A|B] [--busy phases|always|idle]\n\
         --only: run that cluster alone (the other group is another process, e.g. vLLM, on --server);\n\
         --busy: phases (the 4 phases), always or idle"
    );
    std::process::exit(2)
}

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

fn mix(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x100000001b3).rotate_left(29)
}

/// A page block: magic, cluster, rank, the page's 64 tokens, then filler words from a hash of all that.
fn encode(cluster: u64, rank: u64, page: &[i32; ROWS], buf: &mut [u8]) {
    let mut h = mix(mix(MAGIC, cluster), rank);
    let words = buf.len() / 8;
    let mut w = |i: usize, x: u64| buf[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    w(0, MAGIC);
    w(1, cluster);
    w(2, rank);
    for (i, &t) in page.iter().enumerate() {
        w(3 + i, t as u32 as u64);
        h = mix(h, t as u32 as u64);
    }
    for i in 3 + ROWS..words {
        w(i, h ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
    }
}

/// The page a block holds, or what is wrong with it.
fn decode(cluster: u64, rank: u64, buf: &[u8]) -> Result<[i32; ROWS], String> {
    let r = |i: usize| u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    if (r(0), r(1), r(2)) != (MAGIC, cluster, rank) {
        return Err(format!("header {:x} cluster {} rank {}", r(0), r(1), r(2)));
    }
    let mut page = [0i32; ROWS];
    let mut h = mix(mix(MAGIC, cluster), rank);
    for (i, p) in page.iter_mut().enumerate() {
        *p = r(3 + i) as u32 as i32;
        h = mix(h, r(3 + i));
    }
    for i in 3 + ROWS..buf.len() / 8 {
        if r(i) != h ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15) {
            let n = buf.len() / 8;
            let zeros = (i..n).filter(|&j| r(j) == 0).count();
            let bad = (i..n).filter(|&j| r(j) != h ^ (j as u64).wrapping_mul(0x9E3779B97F4A7C15)).count();
            return Err(format!("filler word {i} of {n} is {:x}: {bad} words after it wrong, {zeros} zero", r(i)));
        }
    }
    Ok(page)
}

struct Rank {
    pool: Pool,
    dev: Vec<[i32; ROWS]>,
    store: RemoteStore,
    buf: Vec<u8>,
    buf2: Vec<u8>,
    // the last copies that touched each slot, and the lease's last resizes: named in the message of a wrong block
    hist: HashMap<u64, Vec<String>>,
    resizes: Vec<String>,
}

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn ms() -> u128 {
    START.get_or_init(Instant::now).elapsed().as_millis()
}

impl Rank {
    fn note(&mut self, slot: u64, what: String) {
        let h = self.hist.entry(slot).or_default();
        h.push(format!("{} ms {what}", ms()));
        if h.len() > 8 {
            h.remove(0);
        }
    }
    // a wrong block: what touched the slot, the lease's resizes, and whether a second read of it is right
    fn wrong(&mut self, cluster: u64, name: &str, ri: usize, slot: u64, e: String) -> ! {
        let mut again = vec![0u8; self.buf.len()];
        let second = match self.store.read_block(slot, &mut again) {
            Ok(()) => match decode(cluster, ri as u64, &again) {
                Ok(_) => "right on a second read (changed during the first)".to_string(),
                Err(e2) => format!("wrong again: {e2}"),
            },
            Err(x) => format!("second read failed: {x}"),
        };
        die(format!(
            "{name} rank {ri}: block {slot} is wrong at {} ms: {e}; {second}; lease {} segs {} blocks; slot history: {:?}; \
             resizes: {:?}; node: {}",
            ms(),
            self.store.segs(),
            self.store.blocks(),
            self.hist.get(&slot),
            self.resizes,
            g6kvman::remote::client::stats(&self.store.server.to_string()).unwrap_or_default()
        ))
    }
}

#[derive(Default, Clone)]
struct Counts {
    requests: u64,
    stored_hits: u64,
    restored_blocks: u64,
    spilled_blocks: u64,
    moved_blocks: u64,
    rows_checked: u64,
    grows: u64,
    shrinks: u64,
    give_moves: u64,
}

struct Cluster {
    id: u64,
    name: &'static str,
    ranks: Vec<Rank>,
    per_seg: u64,
    cap: u64,
    agents: Vec<Vec<i32>>,
    system: Vec<i32>,
    rng: Rng,
    // --clock bench: the head's own record of when each entry was last used; pool: the pool's (pool.rs coldest_ages)
    pool_clock: bool,
    last_use: HashMap<u64, Instant>,
    full_seen: u64,
    c: Counts,
}

fn die(m: String) -> ! {
    eprintln!("kvman-share-bench: {m}");
    std::process::exit(1)
}

impl Cluster {
    // every rank runs the call's copies in order through its own store, as the engine does; the ranks must have
    // queued the same copies
    fn run_io(&mut self) -> Vec<Io> {
        let mut first: Option<Vec<Io>> = None;
        for (ri, r) in self.ranks.iter_mut().enumerate() {
            let io = r.pool.io();
            for x in &io {
                match *x {
                    Io::Spill { page, slot, .. } => {
                        r.note(slot, format!("spill page {page}"));
                        encode(self.id, ri as u64, &r.dev[page as usize], &mut r.buf);
                        r.store.write_block(slot, &r.buf).unwrap_or_else(|e| die(format!("spill: {e}")));
                    }
                    Io::Move { slot, to_slot, .. } => {
                        // one tier: a move is the tier shrinking (resize_tier) or a no-op tier change
                        r.note(slot, format!("move out to {to_slot}"));
                        r.note(to_slot, format!("move in from {slot}"));
                        r.store.read_block(slot, &mut r.buf2).unwrap_or_else(|e| die(format!("move read: {e}")));
                        if let Err(e) = decode(self.id, ri as u64, &r.buf2) {
                            r.wrong(self.id, self.name, ri, slot, format!("moved: {e}"));
                        }
                        r.store.write_block(to_slot, &r.buf2).unwrap_or_else(|e| die(format!("move write: {e}")));
                    }
                    Io::Restore { slot, page, .. } => {
                        r.note(slot, format!("restore page {page}"));
                        r.store.read_block(slot, &mut r.buf).unwrap_or_else(|e| die(format!("restore: {e}")));
                        match decode(self.id, ri as u64, &r.buf) {
                            Ok(p) => r.dev[page as usize] = p,
                            Err(e) => r.wrong(self.id, self.name, ri, slot, format!("restored: {e}")),
                        }
                    }
                }
            }
            r.pool.io_done().unwrap();
            match &first {
                None => first = Some(io),
                Some(f) if *f != io => die(format!("{}: rank {ri} queued other copies than rank 0", self.name)),
                _ => {}
            }
        }
        let d = self.ranks[0].pool.digest();
        if self.ranks.iter().any(|r| r.pool.digest() != d) {
            die(format!("{}: the ranks' digests differ", self.name));
        }
        let io = first.unwrap_or_default();
        for x in &io {
            match x {
                Io::Spill { .. } => self.c.spilled_blocks += 1,
                Io::Move { .. } => self.c.moved_blocks += 1,
                Io::Restore { .. } => self.c.restored_blocks += 1,
            }
        }
        if !self.pool_clock {
            // entries new to the tier were used now
            let now = Instant::now();
            let s = self.ranks[0].pool.stash().unwrap();
            let ids: Vec<u64> = s.lru(0).iter().map(|e| e.0).collect();
            for id in &ids {
                self.last_use.entry(*id).or_insert(now);
            }
            self.last_use.retain(|id, _| ids.contains(id));
            // a prefix entry is as recently used as the newest run stored on it: hits on those runs read its blocks
            for e in s.entries(0) {
                if let Some((b, _)) = e.base {
                    let t = self.last_use[&e.id];
                    if let Some(x) = self.last_use.get_mut(&b) {
                        *x = (*x).max(t);
                    }
                }
            }
        }
        io
    }

    // one agent turn: its conversation so far plus a new message, reusing what the pool or the store holds
    fn request(&mut self) {
        let a = self.rng.range(0, self.agents.len() as u64 - 1) as usize;
        let max_rows = (POOL_PAGES as usize - 2) * ROWS;
        let mut prompt = self.agents[a].clone();
        if prompt.len() + 400 > max_rows || prompt.is_empty() {
            prompt = self.system.clone();
        }
        let n = self.rng.range(64, 320);
        prompt.extend((0..n).map(|_| self.rng.range(0, 1 << 30) as i32));
        let gen = self.rng.range(16, 64) as usize;
        let worst = (prompt.len() + gen + 1) as u64;
        let now_rows = prompt.len() as u64 + 1;
        let limit = prompt.len() as i32 - 1;
        for r in &mut self.ranks {
            r.pool.count(&prompt, Some(self.system.len() as u32));
        }
        let h = self.ranks[0].pool.match_prefix(&prompt, now_rows, worst, limit);
        let mut tokens = prompt.clone();
        tokens.extend((0..gen).map(|_| self.rng.range(0, 1 << 30) as i32));
        let mut runs: Vec<Run> = Vec::new();
        for r in &mut self.ranks {
            let hr = r.pool.match_prefix(&prompt, now_rows, worst, limit);
            if hr != h {
                die(format!("{}: the ranks match differently", self.name));
            }
            let run = if h.skip > 0 {
                r.pool.take_hit(h.skip, h.age, h.stored.then_some(h.tier), now_rows, worst)
            } else {
                r.pool.reserve(now_rows, worst)
            };
            match run {
                Ok(x) => runs.push(x),
                Err(e) => die(format!("{}: claim: {e:?}", self.name)),
            }
        }
        self.run_io();
        if h.stored {
            self.c.stored_hits += 1;
        }
        if !self.pool_clock {
            if h.stored {
                self.last_use.insert(h.age, Instant::now());
            }
            // a hit that reuses a stored prefix through a pool run still read it: every prefix entry the prompt starts with
            let now = Instant::now();
            let s = self.ranks[0].pool.stash().unwrap();
            for e in s.entries(0).iter().filter(|e| e.kind.prefix()) {
                if prompt.starts_with(&e.st.history[..(e.st.valid.max(0) as usize).min(e.st.history.len())]) {
                    self.last_use.insert(e.id, now);
                }
            }
        }
        // the reused rows hold the prompt's tokens, on every rank; the rest is computed
        for (r, run) in self.ranks.iter_mut().zip(&runs) {
            let list = r.pool.list(run).unwrap().to_vec();
            for (row, &t) in prompt.iter().enumerate().take(h.skip as usize) {
                if r.dev[list[row / ROWS] as usize][row % ROWS] != t {
                    die(format!("{}: reused row {row} does not hold its token ({h:?})", self.name));
                }
            }
            self.c.rows_checked += h.skip.max(0) as u64;
            for (row, &t) in prompt.iter().enumerate().skip(h.skip as usize) {
                r.dev[list[row / ROWS] as usize][row % ROWS] = t;
            }
        }
        let mut grown = Vec::new();
        for (r, run) in self.ranks.iter_mut().zip(&runs) {
            grown.push(r.pool.grow(run, worst).unwrap().unwrap_or(*run));
        }
        self.run_io();
        for (r, run) in self.ranks.iter_mut().zip(&grown) {
            let list = r.pool.list(run).unwrap().to_vec();
            for (row, &t) in tokens.iter().enumerate().skip(prompt.len()) {
                r.dev[list[row / ROWS] as usize][row % ROWS] = t;
            }
            let valid = tokens.len() as i32 - 1;
            r.pool.retain(run, Finished { history: tokens.clone(), valid, written_end: valid, pf_end: valid }).unwrap();
        }
        self.run_io();
        self.agents[a] = tokens;
        self.c.requests += 1;
    }

    // the head: report to the node, then grow or shrink every rank at this point of the call sequence
    fn head(&mut self, busy: bool) -> String {
        let s0 = self.ranks[0].pool.tier_stats(0).unwrap();
        let wants = busy && s0.full > self.full_seen && self.ranks[0].store.segs() < self.cap;
        self.full_seen = s0.full;
        let ages = if self.pool_clock {
            // the pool's own clock (pool.rs coldest_ages), as the engine's head reports it
            self.ranks[0].pool.coldest_ages(0, self.per_seg, 8)
        } else {
            let now = Instant::now();
            let lru = self.ranks[0].pool.stash().unwrap().give_order(0);
            let lu = &self.last_use;
            coldest_ages(
                &lru,
                |id| lu.get(&id).map_or(0, |t| now.duration_since(*t).as_millis() as u64),
                self.per_seg,
                8,
            )
        };
        let rep = self.ranks[0].store.report(if wants { 2 } else { 0 }, &ages).unwrap_or_else(|e| die(e.to_string()));
        let mut act = String::new();
        if rep.grant > 0 {
            for r in &mut self.ranks {
                let b = r.store.grow(rep.grant).unwrap_or_else(|e| die(format!("grow: {e}")));
                r.resizes.push(format!("{} ms grow +{} to {b} blocks", ms(), rep.grant));
                if !r.pool.resize_tier(0, b).unwrap() {
                    die("resize_tier refused a grow".into());
                }
            }
            self.run_io();
            self.c.grows += 1;
            act = format!("grow+{}", rep.grant);
        } else if rep.give > 0 {
            let to = self.ranks[0].store.segs() - rep.give;
            for r in &mut self.ranks {
                if !r.pool.resize_tier(0, to * self.per_seg).unwrap() {
                    die("resize_tier refused a shrink".into());
                }
            }
            // the moves into the blocks that stay, on every rank, before any rank gives its top segments back
            let io = self.run_io();
            self.c.give_moves += io.iter().filter(|x| matches!(x, Io::Move { .. })).count() as u64;
            for r in &mut self.ranks {
                r.store.shrink(to).unwrap_or_else(|e| die(format!("shrink: {e}")));
                r.resizes.push(format!("{} ms shrink to {to} segs", ms()));
            }
            self.c.shrinks += 1;
            act = format!("give-{}", rep.give);
        }
        let s = self.ranks[0].pool.tier_stats(0).unwrap();
        format!(
            "\"{}\":{{\"busy\":{},\"segs\":{},\"slots\":{},\"used\":{},\"entries\":{},\"coldest_ms\":{},\"want\":{},\
             \"act\":\"{}\",\"unclaimed\":{},\"requests\":{},\"stored_hits\":{},\"restored\":{},\"dropped\":{}}}",
            self.name,
            busy,
            self.ranks[0].store.segs(),
            s.slots,
            s.slots - s.free_slots,
            s.entries,
            ages.first().copied().unwrap_or(0),
            wants,
            act,
            rep.unclaimed,
            self.c.requests,
            self.c.stored_hits,
            self.c.restored_blocks,
            s.dropped
        )
    }
}

struct Args {
    server: Option<String>,
    gb: f64,
    seg_mb: usize,
    margin_s: f64,
    poll_ms: u64,
    phase_s: f64,
    ranks: u32,
    agents: usize,
    want: Want,
    json: Option<String>,
    pool_clock: bool,
    seed: u64,
    // run only cluster 0 (A) or 1 (B)
    only: Option<usize>,
    // None: the phases; Some(b): busy (or idle) all the time
    busy: Option<bool>,
}

fn cluster(a: &Args, addr: &str, id: u64, name: &'static str, mla_row: usize, floor: u64, cap: u64) -> Cluster {
    let block = page_slot(mla_row);
    let per = (a.seg_mb << 20) / block;
    let ranks = (0..a.ranks)
        .map(|r| {
            let o = Opts {
                want: a.want,
                lanes: 2,
                group: Some(GroupOpts {
                    group: format!("{name}-{}", std::process::id()),
                    nr: a.ranks,
                    rank: r,
                    floor_bytes: floor * (per * block) as u64,
                    cap_bytes: cap * (per * block) as u64,
                    weight: 1000,
                }),
                ..Opts::new(&format!("{name}-{}-r{r}", std::process::id()), block, 0)
            };
            let store = RemoteStore::open(addr, &o).unwrap_or_else(|e| die(format!("{name} rank {r}: {e}")));
            let mut pool = Pool::new(POOL_PAGES, POOL_PAGES, POOL_PAGES * PAGE as u32).unwrap();
            pool.keep(Rules { strict: false, noend: true, grid: 128 });
            pool.stores(2);
            pool.add_tier(store.blocks(), Keep::All, None).unwrap();
            pool.set_policy(0, Policy { block_bytes: block as u64, ..Policy::default() }).unwrap();
            Rank {
                pool,
                dev: vec![[-1; ROWS]; POOL_PAGES as usize],
                store,
                buf: vec![0; block],
                buf2: vec![0; block],
                hist: HashMap::new(),
                resizes: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    let mut rng = Rng(id * 7919 + 13 + a.seed * 104_729);
    let system: Vec<i32> = (0..300).map(|_| rng.range(0, 1 << 30) as i32).collect();
    Cluster {
        id,
        name,
        per_seg: ranks[0].store.per_seg,
        cap,
        ranks,
        agents: vec![Vec::new(); a.agents],
        system,
        rng,
        pool_clock: a.pool_clock,
        last_use: HashMap::new(),
        full_seen: 0,
        c: Counts::default(),
    }
}

fn main() {
    let mut a = Args {
        server: None,
        gb: 4.0,
        seg_mb: 128,
        margin_s: 60.0,
        poll_ms: 1000,
        phase_s: 150.0,
        ranks: 2,
        agents: 24,
        want: Want::Tcp,
        json: None,
        pool_clock: true,
        seed: 0,
        only: None,
        busy: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--server" => a.server = Some(v()),
            "--gb" => a.gb = v().parse().unwrap_or_else(|_| usage()),
            "--seg-mb" => a.seg_mb = v().parse().unwrap_or_else(|_| usage()),
            "--margin-s" => a.margin_s = v().parse().unwrap_or_else(|_| usage()),
            "--poll-ms" => a.poll_ms = v().parse().unwrap_or_else(|_| usage()),
            "--phase-s" => a.phase_s = v().parse().unwrap_or_else(|_| usage()),
            "--ranks" => a.ranks = v().parse().unwrap_or_else(|_| usage()),
            "--agents" => a.agents = v().parse().unwrap_or_else(|_| usage()),
            "--transport" => a.want = v().parse().unwrap_or_else(|_| usage()),
            "--json" => a.json = Some(v()),
            "--clock" => {
                a.pool_clock = match v().as_str() {
                    "pool" => true,
                    "bench" => false,
                    _ => usage(),
                }
            }
            "--seed" => a.seed = v().parse().unwrap_or_else(|_| usage()),
            "--only" => {
                a.only = Some(match v().as_str() {
                    "A" => 0,
                    "B" => 1,
                    _ => usage(),
                })
            }
            "--busy" => {
                a.busy = match v().as_str() {
                    "phases" => None,
                    "always" => Some(true),
                    "idle" => Some(false),
                    _ => usage(),
                }
            }
            _ => usage(),
        }
    }
    let margin = Duration::from_secs_f64(a.margin_s);
    let addr = match &a.server {
        Some(s) => s.clone(),
        None => {
            // the stall and cooldown scale with the margin, so a short run keeps the defaults' proportions
            let share = Share {
                seg_bytes: a.seg_mb << 20,
                margin,
                stall: margin / 6,
                cooldown: margin / 6,
                ..Share::default()
            };
            let n = Node::new(
                &Config {
                    bytes: (a.gb * (1u64 << 30) as f64) as usize,
                    lock: false,
                    rdma: false,
                    qp_timeout: 14,
                    share,
                },
                "127.0.0.1:0".parse().unwrap(),
            )
            .unwrap_or_else(|e| die(e.to_string()));
            n.spawn().to_string()
        }
    };
    let out: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(match &a.json {
        Some(p) => Box::new(std::fs::File::create(p).unwrap_or_else(|e| die(format!("{p}: {e}")))),
        None => Box::new(std::io::stdout()),
    }));
    // A: GLM 5.3 full, FP8 rows. B: the INT4-row build. Floor 2 segments a rank each, cap 24
    let specs = [(1, "A", MLA_ROW), (2, "B", MLA_ROW_KV4)];
    let clusters: Vec<Cluster> = specs
        .iter()
        .enumerate()
        .filter(|(i, _)| a.only.is_none_or(|o| o == *i))
        .map(|(_, &(id, name, row))| cluster(&a, &addr, id, name, row, 2, 24))
        .collect();
    let per: Vec<String> = clusters.iter().map(|c| format!("{} {}", c.name, c.per_seg)).collect();
    eprintln!(
        "# node {addr}: {}, segments of {} MiB; blocks a segment: {}; {} ranks each; margin {} s, poll {} ms, \
         phases of {} s, busy {}",
        if a.server.is_some() { "size: the node's".to_string() } else { format!("{} GiB", a.gb) },
        a.seg_mb,
        per.join(", "),
        a.ranks,
        a.margin_s,
        a.poll_ms,
        a.phase_s,
        a.busy.map_or("by phase", |b| if b { "always" } else { "never" })
    );
    // phase -> (A busy, B busy)
    let phases = [(true, false), (false, true), (true, true), (true, false)];
    let t0 = Instant::now();
    let phase_len = Duration::from_secs_f64(a.phase_s);
    let poll = Duration::from_millis(a.poll_ms);
    let end = phase_len * phases.len() as u32;
    let names: Vec<&str> = clusters.iter().map(|c| c.name).collect();
    let busy_all = a.busy;
    let results: Vec<Counts> = std::thread::scope(|sc| {
        let hs: Vec<_> = clusters
            .into_iter()
            .map(|mut c| {
                let i = (c.id - 1) as usize;
                let out = out.clone();
                sc.spawn(move || {
                    let mut next_poll = t0;
                    loop {
                        let t = t0.elapsed();
                        if t >= end {
                            break;
                        }
                        let p = (t.as_secs_f64() / phase_len.as_secs_f64()) as usize;
                        let busy = busy_all.unwrap_or(if i == 0 { phases[p].0 } else { phases[p].1 });
                        if Instant::now() >= next_poll {
                            let line = c.head(busy);
                            writeln!(
                                out.lock().unwrap(),
                                "{{\"t\":{:.2},\"phase\":{},{line}}}",
                                t0.elapsed().as_secs_f64(),
                                p + 1
                            )
                            .unwrap();
                            next_poll += poll;
                        }
                        if busy {
                            c.request();
                        } else {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                    }
                    c.c.clone()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (n, c) in names.iter().zip(&results) {
        eprintln!(
            "# {n}: {} requests, {} stored hits, {} blocks restored and checked whole, {} reused rows checked, {} spilled, \
             {} moved ({} in gives), {} grows, {} gives; no wrong block",
            c.requests,
            c.stored_hits,
            c.restored_blocks,
            c.rows_checked,
            c.spilled_blocks,
            c.moved_blocks,
            c.give_moves,
            c.grows,
            c.shrinks
        );
    }
    eprintln!("# node: {}", g6kvman::remote::client::stats(&addr).unwrap_or_default());
}
