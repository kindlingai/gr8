//! kvman-sim: replays agent session traces (sim/capture_trace.py) through the S2 pool with a time model, to compare
//! what happens to a retained run the pool needs back:
//!   none   no prefix reuse: every request prefills its whole prompt
//!   today  retained runs stay until a reservation evicts them, and a request that lost its run prefills again
//!   ssd    an evicted run is written to an SSD store first, and a request that misses the pool but continues a stored
//!          run reads its pages back instead of prefilling them
//!   grow   today's reuse, but a run holds pages as it needs them (Pool::grow): its prompt + 1 + guard rows at
//!          admission, a page more whenever its tokens reach the end of its pages, admission and growth only when the
//!          pool stays safe (every live run can still reach prompt + max_tokens + guard), else the session waits
//!   grow+ssd  both
//!   system[N], sessions, all, evicted: the engine's own store (src/stash.rs) under the --kv-keep policies, with
//!          growth: system prefixes stored after N conversations (default 2), plus sessions that continued once, or
//!          every eviction. evicted is #37's keep=evicted (every eviction, no prefix bases). Suffixes: -nodedup (no
//!          prefix bases), -dedupe (prefix bases), -nospan (no system span from the front end: grid-cell counting), @F
//!          (--kv-write-limit F)
//!   kvman-sim TRACE.csv [--policies none,today,ssd,grow,grow+ssd] [--repeat N] [flags]
//! --repeat N runs the trace N times over, each copy started --stagger seconds after the last, as N times the agents.
//! Time model (the engine at TP=4): one prefill at a time (--prefill-tps, new tokens only), and no decode while a prefill
//! runs. Decode shares a total rate by how many sessions decode (--decode-tps: 1, 2, 4 sessions, 3 interpolated, more than
//! 4 as 4). The SSD store moves page blocks (src/tier.rs PAGE_SLOT) at --ssd-write-gbs / --ssd-read-gbs, on the
//! admitting request's critical path (--spill evict), or writes every retained run at retirement off the critical path
//! (--spill retire). The pool, its reuse rules and its evictions are the engine's own code (src/pool.rs).
//! --rowcache W[,W..] adds an estimate for a sparse row cache: per decoded token, the indexer's selected rows that fall
//! outside a W-token resident window, read from SSD at --row-iops.

use g6kvman::reuse::{begin, Finished, Rules};
use g6kvman::stash::{Io, Keep};
use g6kvman::tier::PAGE_SLOT;
use g6kvman::{Pool, Run, PAGE};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone)]
struct Req {
    session: usize,
    prompt: usize,
    overlap: usize,
    completion: usize,
    think: f64,
    max_tokens: usize,
    ttft_measured: f64,
    system: usize,        // the session's system span, tokens (0 = unknown)
    system_shared: usize, // tokens of it that session system_from's span starts with
    system_from: Option<usize>,
}

#[derive(Clone)]
struct Opts {
    prefill_tps: f64,
    decode_tps: [f64; 3],
    pages: u32,
    place: u32,
    below: u32,
    sessions: usize,
    guard: usize,
    rules: Rules,
    write_gbs: f64,
    read_gbs: f64,
    store_gb: f64,
    spill_retire: bool,
    rowcache: Vec<usize>,
    row_iops: f64,
    groups: usize,
    store_slots: u64,
}

/// The engine's store under a --kv-keep policy.
#[derive(Clone, Copy, PartialEq)]
struct StoreCfg {
    keep: Keep,
    after: u32,
    min_hits: u32,
    dedupe: bool,
    span: bool,
    limit_milli: u64,
}

#[derive(Clone, PartialEq)]
enum Policy {
    None,
    Today,
    Ssd,
    Grow,
    GrowSsd,
    Store(String, StoreCfg),
}

impl Policy {
    fn name(&self) -> &str {
        match self {
            Policy::None => "none",
            Policy::Today => "today",
            Policy::Ssd => "ssd",
            Policy::Grow => "grow",
            Policy::GrowSsd => "grow+ssd",
            Policy::Store(n, _) => n,
        }
    }
    fn reuse(&self) -> bool {
        *self != Policy::None
    }
    fn ssd(&self) -> bool {
        matches!(self, Policy::Ssd | Policy::GrowSsd)
    }
    fn grow(&self) -> bool {
        matches!(self, Policy::Grow | Policy::GrowSsd | Policy::Store(..))
    }
    fn store(&self) -> Option<StoreCfg> {
        match self {
            Policy::Store(_, c) => Some(*c),
            _ => None,
        }
    }
    fn parse(p: &str) -> Option<Policy> {
        Some(match p {
            "none" => Policy::None,
            "today" => Policy::Today,
            "ssd" => Policy::Ssd,
            "grow" => Policy::Grow,
            "grow+ssd" => Policy::GrowSsd,
            _ => {
                let (base, limit) = match p.split_once('@') {
                    Some((b, f)) => (b, (f.parse::<f64>().ok()? * 1000.0).round() as u64),
                    None => (p, 0),
                };
                let mut c = StoreCfg {
                    keep: Keep::Shared,
                    after: 2,
                    min_hits: 0,
                    dedupe: true,
                    span: true,
                    limit_milli: limit,
                };
                let mut words = base.split('-');
                match words.next()? {
                    w if w.starts_with("system") => {
                        c.after = if w.len() > 6 { w[6..].parse().ok()? } else { 2 };
                    }
                    "sessions" => (c.keep, c.min_hits) = (Keep::All, 1),
                    "all" => c.keep = Keep::All,
                    "evicted" => (c.keep, c.dedupe) = (Keep::Evicted, false),
                    _ => return None,
                }
                for w in words {
                    match w {
                        "nodedup" => c.dedupe = false,
                        "dedupe" => c.dedupe = true,
                        "nospan" => c.span = false,
                        _ => return None,
                    }
                }
                Policy::Store(p.to_string(), c)
            }
        })
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: kvman-sim TRACE.csv [--policies none,today,ssd] [--prefill-tps 1300] [--decode-tps 30,51.5,64.4]\n\
         [--pages 4608] [--place 1920] [--share-limit 65536] [--sessions 8] [--guard 14] [--grid 4096] [--strict]\n\
         [--end-rule] [--ssd-write-gbs 5.7] [--ssd-read-gbs 7.0] [--ssd-gb 256] [--spill evict|retire]\n\
         [--rowcache 16384,32768] [--row-iops 832000] [--groups 21] [--max-tokens N] [--repeat N] [--stagger S]\n\
         policies also: system[N] sessions all evicted, -nodedup -nospan, @F (write limit)"
    );
    std::process::exit(2)
}

fn read_trace(path: &str, max_tokens: Option<usize>) -> Vec<Req> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let head: Vec<&str> = lines.next().unwrap_or_else(|| usage()).split(',').collect();
    let col =
        |name: &str| head.iter().position(|h| h.trim() == name).unwrap_or_else(|| panic!("trace has no {name} column"));
    let c = [
        col("session"),
        col("prompt_tokens"),
        col("overlap_tokens"),
        col("completion_tokens"),
        col("think_s"),
        col("max_tokens"),
        col("ttft_s"),
    ];
    let opt = |name: &str| head.iter().position(|h| h.trim() == name);
    let sys = [opt("system_tokens"), opt("system_shared"), opt("system_from")];
    lines
        .map(|l| {
            let f: Vec<f64> = l.split(',').map(|x| x.trim().parse().expect("a number")).collect();
            let max_tokens = max_tokens.unwrap_or(f[c[5]] as usize).max(1);
            Req {
                session: f[c[0]] as usize,
                prompt: f[c[1]] as usize,
                overlap: f[c[2]] as usize,
                // a request stops at its max_tokens (--max-tokens below what it wrote cuts it short)
                completion: (f[c[3]] as usize).clamp(1, max_tokens),
                think: f[c[4]],
                max_tokens,
                ttft_measured: f[c[6]],
                system: sys[0].map_or(0, |i| f[i] as usize),
                system_shared: sys[1].map_or(0, |i| f[i] as usize),
                system_from: sys[2].and_then(|i| (f[i] >= 0.0).then_some(f[i] as usize)),
            }
        })
        .collect()
}

// the trace n times over, each copy's sessions starting `stagger` seconds after the last copy's
// (agents in other folders: a copy's system span starts with the part every span in the trace shares)
fn repeat(reqs: Vec<Req>, n: usize, stagger: f64) -> Vec<Req> {
    let sessions = reqs.iter().map(|r| r.session + 1).max().unwrap_or(0);
    let common = reqs.iter().map(|r| r.system_shared).max().unwrap_or(0);
    let mut out = Vec::new();
    for k in 0..n {
        let mut first = vec![true; sessions];
        for r in &reqs {
            let mut x = r.clone();
            x.session += k * sessions;
            if k > 0 {
                x.system_from = Some(r.session);
                x.system_shared = common.min(r.system);
            }
            if first[r.session] {
                x.think += k as f64 * stagger;
                first[r.session] = false;
            }
            out.push(x);
        }
    }
    out
}

fn main() {
    let mut a = std::env::args().skip(1);
    let trace = a.next().unwrap_or_else(|| usage());
    let mut o = Opts {
        prefill_tps: 1300.0,
        decode_tps: [30.0, 51.5, 64.4],
        pages: 4608,
        place: 1920,
        below: 65536,
        sessions: 8,
        guard: 14,
        rules: Rules { strict: false, noend: true, grid: 4096 },
        write_gbs: 5.7,
        read_gbs: 7.0,
        store_gb: 256.0,
        spill_retire: false,
        rowcache: Vec::new(),
        row_iops: 832_000.0,
        groups: 21,
        store_slots: 0,
    };
    let mut policies = vec![Policy::None, Policy::Today, Policy::Ssd, Policy::Grow, Policy::GrowSsd];
    let mut max_tokens = None;
    let (mut copies, mut stagger) = (1usize, 600.0f64);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| usage());
        let num = |s: String| s.parse::<f64>().unwrap_or_else(|_| usage());
        match k.as_str() {
            "--policies" => policies = v().split(',').map(|p| Policy::parse(p).unwrap_or_else(|| usage())).collect(),
            "--prefill-tps" => o.prefill_tps = num(v()),
            "--decode-tps" => {
                let d: Vec<f64> = v().split(',').map(|x| x.parse().unwrap_or_else(|_| usage())).collect();
                o.decode_tps = d.try_into().unwrap_or_else(|_| usage());
            }
            "--pages" => o.pages = num(v()) as u32,
            "--place" => o.place = num(v()) as u32,
            "--share-limit" => o.below = num(v()) as u32,
            "--sessions" => o.sessions = num(v()) as usize,
            "--guard" => o.guard = num(v()) as usize,
            "--grid" => o.rules.grid = num(v()) as i32,
            "--strict" => o.rules.strict = true,
            "--end-rule" => o.rules.noend = false,
            "--ssd-write-gbs" => o.write_gbs = num(v()),
            "--ssd-read-gbs" => o.read_gbs = num(v()),
            "--ssd-gb" => o.store_gb = num(v()),
            "--spill" => {
                o.spill_retire = match v().as_str() {
                    "evict" => false,
                    "retire" => true,
                    _ => usage(),
                }
            }
            "--rowcache" => o.rowcache = v().split(',').map(|x| x.parse().unwrap_or_else(|_| usage())).collect(),
            "--row-iops" => o.row_iops = num(v()),
            "--groups" => o.groups = num(v()) as usize,
            "--max-tokens" => max_tokens = Some(num(v()) as usize),
            "--repeat" => copies = num(v()) as usize,
            "--stagger" => stagger = num(v()),
            _ => usage(),
        }
    }
    let reqs = repeat(read_trace(&trace, max_tokens), copies.max(1), stagger);
    o.store_slots = ((o.store_gb * 1e9) as u64 / PAGE_SLOT as u64).max(1);
    let sessions = reqs.iter().map(|r| r.session + 1).max().unwrap_or(0);
    let prompt_total: usize = reqs.iter().map(|r| r.prompt).sum();
    println!(
        "{trace}: {} requests in {sessions} sessions, {prompt_total} prompt tokens, {} completion tokens",
        reqs.len(),
        reqs.iter().map(|r| r.completion).sum::<usize>()
    );
    let mut meas: Vec<f64> = reqs.iter().map(|r| r.ttft_measured).collect();
    println!(
        "measured on the cluster: TTFT median {:.1} s, p90 {:.1} s",
        quantile(&mut meas, 0.5),
        quantile(&mut meas, 0.9)
    );
    println!(
        "{:<16} {:>8} {:>8} {:>9} {:>10} {:>9} {:>6} {:>6} {:>6} {:>7} {:>9} {:>9} {:>7} {:>8} {:>8}",
        "policy",
        "TTFT p50",
        "TTFT p90",
        "wall s",
        "prefilled",
        "avoided",
        "pool",
        "ssd",
        "lost",
        "evicted",
        "SSD wr GB",
        "SSD rd GB",
        "live",
        "waited s",
        "wr GB/h"
    );
    for p in policies {
        let r = simulate(&reqs, &o, &p);
        let mut t = r.ttft.clone();
        println!(
            "{:<16} {:>8.1} {:>8.1} {:>9.0} {:>10} {:>9} {:>6} {:>6} {:>6} {:>7} {:>9.1} {:>9.1} {:>7.2} {:>8.0} {:>8.1}",
            p.name(),
            quantile(&mut t, 0.5),
            quantile(&mut t, 0.9),
            r.wall,
            r.prefilled,
            prompt_total - r.prefilled,
            r.pool_hits,
            r.ssd_hits,
            r.lost,
            r.evicted,
            r.written as f64 / 1e9,
            r.read as f64 / 1e9,
            r.live_s / r.wall.max(1e-9),
            r.stall_s,
            r.written as f64 / 1e9 / r.wall.max(1e-9) * 3600.0
        );
    }
    for &w in &o.rowcache {
        let (reads, tokens) = rowcache(&reqs, w, o.groups);
        let per = reads / tokens.max(1) as f64;
        println!(
            "row cache, {w}-token window: {:.0} SSD row reads per decoded token on average ({:.1} MB), {:.1} ms at {:.0} IOPS",
            per,
            per * 4096.0 / 1e6,
            per / o.row_iops * 1e3,
            o.row_iops
        );
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

// Rows read from SSD per decoded token when the indexer selects 2,048 of the context's tokens per group and only the
// last w tokens are resident: assumes the selection is spread evenly over the context.
fn rowcache(reqs: &[Req], w: usize, groups: usize) -> (f64, usize) {
    let (mut reads, mut tokens) = (0.0, 0);
    for r in reqs {
        for c in r.prompt..r.prompt + r.completion {
            if c > w && c > 2048 {
                reads += groups as f64 * 2048.0 * (c - w) as f64 / c as f64;
            }
            tokens += 1;
        }
    }
    (reads, tokens)
}

#[derive(Default)]
struct Result {
    ttft: Vec<f64>,
    wall: f64,
    prefilled: usize,
    pool_hits: usize,
    ssd_hits: usize,
    lost: usize,
    evicted: usize,
    written: u64,
    read: u64,
    live_s: f64,  // the integral of admitted sessions over time: live_s / wall = sessions resident on average
    stall_s: f64, // session-seconds a decoding session waited for a page
}

struct Active {
    session: usize,
    run: Run,
    arrival: f64,
    prefill_s: f64,
    decode_left: f64,
    len: f64, // tokens so far, prompt included, once decoding
    worst: u64,
    stalled: bool,
    prompt: Vec<i32>,
    completion: usize,
}

struct Stored {
    st: Finished,
    pages: usize,
}

// the SSD store: stored runs, oldest first, dropped from the front past its capacity
struct Store {
    q: VecDeque<Stored>,
    pages: usize,
    cap: usize,
}

impl Store {
    fn push(&mut self, s: Stored) {
        self.pages += s.pages;
        self.q.push_back(s);
        while self.pages > self.cap {
            self.pages -= self.q.pop_front().map(|x| x.pages).unwrap_or(0);
        }
    }
}

fn decode_rate(o: &Opts, n: usize) -> f64 {
    let d = &o.decode_tps;
    let total = match n {
        0 => return 0.0,
        1 => d[0],
        2 => d[1],
        3 => (d[1] + d[2]) / 2.0,
        _ => d[2],
    };
    total / n as f64
}

fn simulate(reqs: &[Req], o: &Opts, policy: &Policy) -> Result {
    let mut pool = Pool::new(o.pages, o.place, o.below).expect("pool sizes");
    if policy.reuse() {
        pool.keep(o.rules);
    }
    let cfg = policy.store();
    if let Some(c) = cfg {
        pool.stores(c.after);
        pool.add_tier(o.store_slots, c.keep, None).expect("a tier");
        let p = g6kvman::stash::Policy { min_rows: 0, min_hits: c.min_hits, block_bytes: PAGE_SLOT as u64 };
        pool.set_policy(0, p).expect("tier 0");
        pool.write_limit(c.limit_milli);
        pool.stash_mut().expect("stores").dedupe = c.dedupe;
    }
    let mut res = Result::default();
    let nsess = reqs.iter().map(|r| r.session + 1).max().unwrap_or(0);
    let mut by_session: Vec<VecDeque<&Req>> = vec![VecDeque::new(); nsess];
    for r in reqs {
        by_session[r.session].push_back(r);
    }
    // next arrival per session (None while one of its requests is in flight or it is done)
    let mut next: Vec<Option<f64>> = by_session.iter().map(|q| q.front().map(|r| r.think)).collect();
    let mut last_prompt: Vec<Vec<i32>> = vec![Vec::new(); nsess];
    let mut token = 0i32;
    let mut fresh = || {
        token += 1;
        token
    };
    // each session's system span: the part another session's span starts with, then its own tokens
    let mut system: Vec<Vec<i32>> = vec![Vec::new(); nsess];
    for q in &by_session {
        if let Some(r) = q.front() {
            let mut t: Vec<i32> = match r.system_from {
                Some(f) if f < r.session => system[f][..r.system_shared.min(system[f].len())].to_vec(),
                _ => Vec::new(),
            };
            t.extend((t.len()..r.system).map(|_| fresh()));
            system[r.session] = t;
        }
    }
    let mut fifo: VecDeque<(usize, f64)> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let page_bytes = PAGE_SLOT as u64;
    let mut store = Store { q: VecDeque::new(), pages: 0, cap: ((o.store_gb * 1e9) as u64 / page_bytes) as usize };
    let mut io_block = 0.0f64; // SSD writes of runs that growth evicted: they hold up every session, as a prefill does
                               // the retained runs' contents by age: what the ssd policies write out when the pool evicts one
    let mut retained: BTreeMap<u64, Stored> = BTreeMap::new();
    let mut t = 0.0f64;
    let guard = o.guard as f64;
    // a run grows to `rows`: the retained runs it evicts are counted and, under the ssd policies, written out first
    let grow = |pool: &mut Pool,
                x: &mut Active,
                rows: u64,
                retained: &mut BTreeMap<u64, Stored>,
                res: &mut Result,
                store: &mut Store,
                io_block: &mut f64| {
        let before: Vec<u64> = pool.kept().iter().map(|k| k.age).collect();
        let r = pool.grow(&x.run, rows).expect("live run");
        *io_block += drain(pool, res, o);
        evicted(pool, &before, 0, retained, res, |y, res| {
            if policy.ssd() && !o.spill_retire {
                res.written += y.pages as u64 * page_bytes;
                *io_block += (y.pages as u64 * page_bytes) as f64 / (o.write_gbs * 1e9);
                store.push(y);
            }
        });
        match r {
            Some(r) => {
                x.run = r;
                x.stalled = false;
            }
            None => x.stalled = true,
        }
    };
    loop {
        for (s, a) in next.iter_mut().enumerate() {
            if a.is_some_and(|a| a <= t) {
                fifo.push_back((s, a.take().expect("checked")));
            }
        }
        // a waiting session takes its page as soon as the pool stays safe with it, oldest admission first
        for x in active.iter_mut().filter(|x| x.stalled) {
            let rows = (x.len.ceil() as u64 + o.guard as u64 + 1).min(x.worst);
            grow(&mut pool, x, rows, &mut retained, &mut res, &mut store, &mut io_block);
        }
        // admit the FIFO head while it fits: a session slot, and the pool (room, and a safe state for growth)
        while let Some(&(s, arrival)) = fifo.front() {
            let r = by_session[s].front().expect("a queued session has a request");
            let worst = (r.prompt + r.max_tokens + o.guard) as u64;
            let rows = if policy.grow() { worst.min((r.prompt + 1 + o.guard) as u64) } else { worst };
            if active.len() >= o.sessions || !pool.fits(rows, worst) {
                break;
            }
            fifo.pop_front();
            let mut prompt: Vec<i32> = last_prompt[s][..r.overlap.min(last_prompt[s].len())].to_vec();
            if prompt.is_empty() {
                prompt = system[s][..system[s].len().min(r.prompt)].to_vec();
            }
            prompt.extend((prompt.len()..r.prompt).map(|_| fresh()));
            let before: Vec<u64> = pool.kept().iter().map(|k| k.age).collect();
            let limit = r.prompt as i32 - 1;
            if let Some(c) = cfg {
                let span = (c.span && r.system > 0).then_some(r.system as u32);
                pool.count(&prompt, span);
            }
            let h = if policy.reuse() { pool.match_prefix(&prompt, rows, worst, limit) } else { Default::default() };
            let (mut io_s, mut skip) = (0.0, 0usize);
            let run = if h.stored {
                res.ssd_hits += 1;
                skip = h.skip as usize;
                let run = pool.take_hit(h.skip, h.age, Some(h.tier), rows, worst).expect("a stored hit from match");
                io_block += drain(&mut pool, &mut res, o);
                run
            } else if h.skip > 0 {
                res.pool_hits += 1;
                skip = h.skip as usize;
                retained.remove(&h.age);
                let run = pool.take(h.skip, h.age, rows, worst).expect("a hit from match");
                io_block += drain(&mut pool, &mut res, o);
                run
            } else {
                // a stored run this prompt continues, the longest first, then the most recent
                let best = policy
                    .ssd()
                    .then(|| {
                        store
                            .q
                            .iter()
                            .enumerate()
                            .map(|(i, x)| (i, begin(&o.rules, &x.st, &prompt, limit, true).skip))
                            .filter(|&(i, k)| k > 0 && (k as u64).div_ceil(PAGE) as usize <= store.q[i].pages)
                            .max_by_key(|&(i, k)| (k, i))
                    })
                    .flatten();
                let run = pool.reserve(rows, worst).expect("fits checked");
                io_block += drain(&mut pool, &mut res, o);
                if let Some((i, k)) = best {
                    let x = store.q.remove(i).expect("found");
                    store.pages -= x.pages;
                    let pages = (k as u64).div_ceil(PAGE);
                    res.read += pages * page_bytes;
                    io_s += (pages * page_bytes) as f64 / (o.read_gbs * 1e9);
                    res.ssd_hits += 1;
                    skip = k as usize;
                }
                run
            };
            // retained runs this grant evicted: the ssd policies write them out first (unless they went at retirement)
            evicted(&pool, &before, h.age, &mut retained, &mut res, |x, res| {
                if policy.ssd() && !o.spill_retire {
                    res.written += x.pages as u64 * page_bytes;
                    io_s += (x.pages as u64 * page_bytes) as f64 / (o.write_gbs * 1e9);
                    store.push(x);
                }
            });
            if r.overlap >= o.rules.grid.max(1) as usize && skip == 0 {
                res.lost += 1;
            }
            let new = r.prompt - skip;
            res.prefilled += new;
            last_prompt[s] = prompt.clone();
            active.push(Active {
                session: s,
                run,
                arrival,
                prefill_s: io_s + new as f64 / o.prefill_tps,
                decode_left: (r.completion - 1) as f64,
                len: (r.prompt + 1) as f64,
                worst,
                stalled: false,
                prompt,
                completion: r.completion,
            });
        }
        // the next event: the running prefill ends, a decoding session ends or reaches the end of its pages, or a
        // session sends its next request
        let prefilling = active.iter().position(|x| x.prefill_s > 0.0);
        let decoding = active.iter().filter(|x| x.prefill_s <= 0.0 && !x.stalled).count();
        let rate = decode_rate(o, decoding);
        let room = |x: &Active| x.run.pages as f64 * PAGE as f64 - guard - x.len; // tokens before it needs a page
        let mut dt = f64::INFINITY;
        if io_block > 0.0 {
            dt = io_block;
        } else if let Some(i) = prefilling {
            dt = active[i].prefill_s;
        } else if decoding > 0 {
            for x in active.iter().filter(|x| !x.stalled) {
                dt = dt.min(x.decode_left / rate).min(room(x).max(0.0) / rate);
            }
        }
        if let Some(a) = next.iter().flatten().copied().reduce(f64::min) {
            dt = dt.min((a - t).max(0.0));
        }
        if !dt.is_finite() {
            assert!(active.is_empty() && fifo.is_empty(), "{}: sessions left waiting: deadlock", policy.name());
            break;
        }
        t += dt;
        res.live_s += dt * active.len() as f64;
        res.stall_s += dt * active.iter().filter(|x| x.stalled).count() as f64;
        if io_block > 0.0 {
            io_block = (io_block - dt).max(0.0);
        } else if let Some(i) = prefilling {
            active[i].prefill_s -= dt;
            if active[i].prefill_s <= 1e-12 {
                active[i].prefill_s = 0.0;
                res.ttft.push(t - active[i].arrival);
            }
        } else {
            for x in active.iter_mut().filter(|x| !x.stalled) {
                x.decode_left -= rate * dt;
                x.len += rate * dt;
            }
        }
        // decoding sessions at the end of their pages grow by a page, or wait
        for x in active.iter_mut() {
            if x.prefill_s <= 0.0 && !x.stalled && x.decode_left > 1e-9 && room(x) <= 1e-9 {
                let rows = (x.len.ceil() as u64 + o.guard as u64 + 1).min(x.worst);
                grow(&mut pool, x, rows, &mut retained, &mut res, &mut store, &mut io_block);
            }
        }
        // sessions done decoding retire their runs
        let mut i = 0;
        while i < active.len() {
            let x = &active[i];
            if x.prefill_s > 0.0 || x.decode_left > 1e-9 {
                i += 1;
                continue;
            }
            let x = active.remove(i);
            let mut history = x.prompt.clone();
            history.extend((0..x.completion).map(|_| fresh()));
            let n = x.prompt.len() as i32;
            let st = Finished {
                valid: n + x.completion as i32 - 1,
                written_end: n + x.completion as i32,
                pf_end: n - 1,
                history,
            };
            let pages = x.run.pages as usize;
            if !policy.reuse() {
                pool.release(&x.run).expect("live run");
            } else {
                let kept = pool.kept_runs();
                pool.retain(&x.run, st.clone()).expect("live run");
                io_block += drain(&mut pool, &mut res, o);
                if pool.kept_runs() > kept {
                    let age = pool.kept().last().expect("just retained").age;
                    if policy.ssd() && o.spill_retire {
                        res.written += pages as u64 * page_bytes;
                        store.push(Stored { st: st.clone(), pages });
                    }
                    retained.insert(age, Stored { st, pages });
                }
            }
            res.wall = t;
            by_session[x.session].pop_front();
            next[x.session] = by_session[x.session].front().map(|r| t + r.think);
        }
    }
    if std::env::var_os("KVSIM_ENTRIES").is_some() {
        if let Some(s) = pool.stash() {
            for t in 0..s.tiers() {
                for e in s.entries(t) {
                    let key =
                        g6kvman::stash::prefix_key(&e.st.history[..(e.st.valid as usize + 1).min(e.st.history.len())]);
                    eprintln!(
                        "{}: entry {} {:?} rows {} blocks {} counted {}",
                        policy.name(),
                        e.id,
                        e.kind,
                        e.st.valid,
                        e.slots.len(),
                        s.count_of(key)
                    );
                }
            }
        }
    }
    res
}

// the store copies the last pool call queued (the engine's drain): bytes written and read, and the seconds the drain
// holds every session up
fn drain(pool: &mut Pool, res: &mut Result, o: &Opts) -> f64 {
    let io = pool.io();
    if io.is_empty() {
        let _ = pool.io_done();
        return 0.0;
    }
    // blocks, not copies: a tail's replicated parts share blocks
    let (mut w, mut r) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    for x in &io {
        match *x {
            Io::Spill { tier, slot, .. } | Io::Move { to: tier, to_slot: slot, .. } => w.insert((tier, slot)),
            Io::Restore { tier, slot, .. } => r.insert((tier, slot)),
        };
    }
    let (w, r) = (w.len() as u64 * PAGE_SLOT as u64, r.len() as u64 * PAGE_SLOT as u64);
    pool.io_done().expect("no index");
    res.written += w;
    res.read += r;
    w as f64 / (o.write_gbs * 1e9) + r as f64 / (o.read_gbs * 1e9)
}

// the retained runs a grant evicted (in `before`, not kept now, not the run a hit adopted): counted, and handed to
// `spill` with their contents
fn evicted<F: FnMut(Stored, &mut Result)>(
    pool: &Pool,
    before: &[u64],
    taken: u64,
    retained: &mut BTreeMap<u64, Stored>,
    res: &mut Result,
    mut spill: F,
) {
    let after: Vec<u64> = pool.kept().iter().map(|k| k.age).collect();
    for &age in before.iter().filter(|a| !after.contains(a) && **a != taken) {
        res.evicted += 1;
        if let Some(x) = retained.remove(&age) {
            spill(x, res);
        }
    }
}
