//! kvman-sim: replays agent session traces (sim/capture_trace.py) through the S2 pool with a time model, to compare
//! what happens to a retained run the pool needs back:
//!   none   no prefix reuse: every request prefills its whole prompt
//!   today  retained runs stay until a reservation evicts them, and a request that lost its run prefills again
//!   ssd    an evicted run is written to an SSD store first, and a request that misses the pool but continues a stored
//!          run reads its pages back instead of prefilling them
//!   kvman-sim TRACE.csv [--policies none,today,ssd] [flags]
//! Time model (the engine at TP=4): one prefill at a time (--prefill-tps, new tokens only), and no decode while a prefill
//! runs. Decode shares a total rate by how many sessions decode (--decode-tps: 1, 2, 4 sessions, 3 interpolated, more than
//! 4 as 4). The SSD store moves page blocks (src/tier.rs PAGE_SLOT) at --ssd-write-gbs / --ssd-read-gbs, on the
//! admitting request's critical path (--spill evict), or writes every retained run at retirement off the critical path
//! (--spill retire). The pool, its reuse rules and its evictions are the engine's own code (src/pool.rs).
//! --rowcache W[,W..] adds an estimate for a sparse row cache: per decoded token, the indexer's selected rows that fall
//! outside a W-token resident window, read from SSD at --row-iops.

use g6kvman::reuse::{begin, Finished, Rules};
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
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    None,
    Today,
    Ssd,
}

fn usage() -> ! {
    eprintln!(
        "usage: kvman-sim TRACE.csv [--policies none,today,ssd] [--prefill-tps 1300] [--decode-tps 30,51.5,64.4]\n\
         [--pages 4608] [--place 1920] [--share-limit 65536] [--sessions 8] [--guard 14] [--grid 4096] [--strict]\n\
         [--end-rule] [--ssd-write-gbs 5.7] [--ssd-read-gbs 7.0] [--ssd-gb 256] [--spill evict|retire]\n\
         [--rowcache 16384,32768] [--row-iops 832000] [--groups 21] [--max-tokens N]"
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
    lines
        .map(|l| {
            let f: Vec<f64> = l.split(',').map(|x| x.trim().parse().expect("a number")).collect();
            Req {
                session: f[c[0]] as usize,
                prompt: f[c[1]] as usize,
                overlap: f[c[2]] as usize,
                completion: (f[c[3]] as usize).max(1),
                think: f[c[4]],
                max_tokens: max_tokens.unwrap_or(f[c[5]] as usize),
                ttft_measured: f[c[6]],
            }
        })
        .collect()
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
    };
    let mut policies = vec![Policy::None, Policy::Today, Policy::Ssd];
    let mut max_tokens = None;
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| usage());
        let num = |s: String| s.parse::<f64>().unwrap_or_else(|_| usage());
        match k.as_str() {
            "--policies" => {
                policies = v()
                    .split(',')
                    .map(|p| match p {
                        "none" => Policy::None,
                        "today" => Policy::Today,
                        "ssd" => Policy::Ssd,
                        _ => usage(),
                    })
                    .collect()
            }
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
            _ => usage(),
        }
    }
    let reqs = read_trace(&trace, max_tokens);
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
        "{:<6} {:>8} {:>8} {:>9} {:>10} {:>9} {:>6} {:>6} {:>6} {:>7} {:>9} {:>9}",
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
        "SSD rd GB"
    );
    for p in policies {
        let r = simulate(&reqs, &o, p);
        let mut t = r.ttft.clone();
        println!(
            "{:<6} {:>8.1} {:>8.1} {:>9.0} {:>10} {:>9} {:>6} {:>6} {:>6} {:>7} {:>9.1} {:>9.1}",
            match p {
                Policy::None => "none",
                Policy::Today => "today",
                Policy::Ssd => "ssd",
            },
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
            r.read as f64 / 1e9
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
}

struct Active {
    session: usize,
    run: Run,
    arrival: f64,
    prefill_s: f64,
    decode_left: f64,
    prompt: Vec<i32>,
    completion: usize,
}

struct Stored {
    st: Finished,
    pages: usize,
}

fn simulate(reqs: &[Req], o: &Opts, policy: Policy) -> Result {
    let mut pool = Pool::new(o.pages, o.place, o.below).expect("pool sizes");
    if policy != Policy::None {
        pool.keep(o.rules);
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
    let mut fifo: VecDeque<(usize, f64)> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut store: VecDeque<Stored> = VecDeque::new();
    let page_bytes = PAGE_SLOT as u64;
    let store_cap = (o.store_gb * 1e9) as u64 / page_bytes;
    let mut stored_pages = 0usize;
    // the retained runs' contents by age: what the ssd policy writes out when the pool evicts one
    let mut retained: BTreeMap<u64, Stored> = BTreeMap::new();
    let mut t = 0.0f64;
    let store_push = |store: &mut VecDeque<Stored>, stored_pages: &mut usize, s: Stored| {
        *stored_pages += s.pages;
        store.push_back(s);
        while *stored_pages as u64 > store_cap {
            *stored_pages -= store.pop_front().map(|x| x.pages).unwrap_or(0);
        }
    };
    loop {
        for (s, a) in next.iter_mut().enumerate() {
            if a.is_some_and(|a| a <= t) {
                fifo.push_back((s, a.take().expect("checked")));
            }
        }
        // admit the FIFO head while it fits: a session slot, and the pool's room
        while let Some(&(s, arrival)) = fifo.front() {
            let r = by_session[s].front().expect("a queued session has a request");
            let rows = (r.prompt + r.max_tokens + o.guard) as u64;
            if active.len() >= o.sessions || !pool.admissible(rows) || rows.div_ceil(PAGE) > pool.room() as u64 {
                break;
            }
            fifo.pop_front();
            let mut prompt: Vec<i32> = last_prompt[s][..r.overlap.min(last_prompt[s].len())].to_vec();
            prompt.extend((prompt.len()..r.prompt).map(|_| fresh()));
            let before: Vec<u64> = pool.kept().iter().map(|k| k.age).collect();
            let limit = r.prompt as i32 - 1;
            let h = if policy == Policy::None { Default::default() } else { pool.match_prefix(&prompt, rows, limit) };
            let (mut io_s, mut skip) = (0.0, 0usize);
            let run = if h.skip > 0 {
                res.pool_hits += 1;
                skip = h.skip as usize;
                retained.remove(&h.age);
                pool.take(h.skip, h.age, rows).expect("a hit from match")
            } else {
                // a stored run this prompt continues, the longest first, then the most recent
                let n = rows.div_ceil(PAGE) as usize;
                let best = (policy == Policy::Ssd)
                    .then(|| {
                        store
                            .iter()
                            .enumerate()
                            .map(|(i, x)| (i, begin(&o.rules, &x.st, &prompt, limit, true).skip))
                            .filter(|&(i, k)| k > 0 && (k as u64).div_ceil(PAGE) as usize <= store[i].pages.min(n))
                            .max_by_key(|&(i, k)| (k, i))
                    })
                    .flatten();
                let run = pool.reserve(rows).expect("room checked");
                if let Some((i, k)) = best {
                    let x = store.remove(i).expect("found");
                    stored_pages -= x.pages;
                    let pages = (k as u64).div_ceil(PAGE);
                    res.read += pages * page_bytes;
                    io_s += (pages * page_bytes) as f64 / (o.read_gbs * 1e9);
                    res.ssd_hits += 1;
                    skip = k as usize;
                }
                run
            };
            // retained runs this grant evicted: the ssd policy writes them out first (unless they went at retirement)
            let after: Vec<u64> = pool.kept().iter().map(|k| k.age).collect();
            let gone: Vec<u64> = before.into_iter().filter(|a| !after.contains(a) && *a != h.age).collect();
            res.evicted += gone.len();
            for age in gone {
                let Some(x) = retained.remove(&age) else { continue };
                if policy == Policy::Ssd && !o.spill_retire {
                    res.written += x.pages as u64 * page_bytes;
                    io_s += (x.pages as u64 * page_bytes) as f64 / (o.write_gbs * 1e9);
                    store_push(&mut store, &mut stored_pages, x);
                }
            }
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
                prompt,
                completion: r.completion,
            });
        }
        // the next event: the running prefill ends, a decoding session ends, or a session sends its next request
        let prefilling = active.iter().position(|x| x.prefill_s > 0.0);
        let decoding = active.iter().filter(|x| x.prefill_s <= 0.0).count();
        let rate = if decoding == 0 {
            0.0
        } else {
            let d = &o.decode_tps;
            let total = match decoding {
                1 => d[0],
                2 => d[1],
                3 => (d[1] + d[2]) / 2.0,
                _ => d[2],
            };
            total / decoding as f64
        };
        let mut dt = f64::INFINITY;
        if let Some(i) = prefilling {
            dt = active[i].prefill_s;
        } else if decoding > 0 {
            dt = active.iter().map(|x| x.decode_left / rate).fold(f64::INFINITY, f64::min);
        }
        if let Some(a) = next.iter().flatten().copied().reduce(f64::min) {
            dt = dt.min((a - t).max(0.0));
        }
        if !dt.is_finite() {
            break;
        }
        t += dt;
        if let Some(i) = prefilling {
            active[i].prefill_s -= dt;
            if active[i].prefill_s <= 1e-12 {
                active[i].prefill_s = 0.0;
                res.ttft.push(t - active[i].arrival);
            }
        } else {
            for x in active.iter_mut() {
                x.decode_left -= rate * dt;
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
            if policy == Policy::None {
                pool.release(&x.run).expect("live run");
            } else {
                let kept = pool.kept_runs();
                pool.retain(&x.run, st.clone()).expect("live run");
                if pool.kept_runs() > kept {
                    let age = pool.kept().last().expect("just retained").age;
                    if policy == Policy::Ssd && o.spill_retire {
                        res.written += pages as u64 * page_bytes;
                        store_push(&mut store, &mut stored_pages, Stored { st: st.clone(), pages });
                    }
                    retained.insert(age, Stored { st, pages });
                }
            }
            res.wall = t;
            by_session[x.session].pop_front();
            next[x.session] = by_session[x.session].front().map(|r| t + r.think);
        }
    }
    res
}
