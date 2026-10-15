//! kvman-remote-bench: what a lease on a KV node (src/remote, kvman-remote) moves from this box, beside
//! kvman-ssd-bench for the box's own NVMe.
//!   kvman-remote-bench --server HOST[:PORT] [--transport auto|tcp|rdma] [--gb 2] [--threads 1,2,4,8,16]
//!       [--qd 1,2,4,8,16,32,64] [--gather 256,2048,4096] [--seconds 3] [--name NAME] [--copy] [--isolation]
//!   kvman-remote-bench --local GB ...   a node in this process on 127.0.0.1, unlocked (a smoke test)
//! Page blocks (one logical KV page of every layer, 3,506,176 B): write and read GB/s and per-block latency at each
//! thread count, every pass over the whole lease. Rows (2,624 B, one token's MLA rows of one indexer group) at random
//! places: reads/s and latency at each queue depth (one thread per read in flight). Gathers: one request for N random
//! rows, as a sparse-attention step would fetch them. Buffers are registered with the RDMA device unless --copy, which
//! sends them through each lane's staging buffer as an unregistered caller would. --isolation (RDMA) shows the NIC
//! refuse another lease's rkey and an address past the lease.

use g6kvman::remote::{Config, Node, Opts, RemoteStore, Transport, Want};
use g6kvman::tier::{par_for, AlignedBuf, BlockStore, PAGE_SLOT, ROW_BYTES};
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn usage() -> ! {
    eprintln!(
        "usage: kvman-remote-bench --server HOST[:PORT] | --local GB  [--transport auto|tcp|rdma] [--gb 2] \
         [--threads 1,2,4,8,16] [--qd 1,2,..,64] [--gather 256,2048,4096] [--seconds 3] [--name NAME] [--copy] \
         [--isolation]"
    );
    std::process::exit(2)
}

struct Args {
    server: String,
    want: Want,
    gb: f64,
    threads: Vec<usize>,
    qds: Vec<usize>,
    gathers: Vec<usize>,
    dur: Duration,
    name: String,
    copy: bool,
    isolation: bool,
}

fn main() {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_else(|_| "host".into());
    let mut o = Args {
        server: String::new(),
        want: Want::Auto,
        gb: 2.0,
        threads: vec![1, 2, 4, 8, 16],
        qds: vec![1, 2, 4, 8, 16, 32, 64],
        gathers: vec![256, 2048, 4096],
        dur: Duration::from_secs(3),
        name: format!("bench-{}-{}", host.trim(), std::process::id()),
        copy: false,
        isolation: false,
    };
    let mut local = None;
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--server" => o.server = v(),
            "--local" => local = Some(v().parse::<f64>().unwrap_or_else(|_| usage())),
            "--transport" => o.want = v().parse().unwrap_or_else(|_| usage()),
            "--gb" => o.gb = v().parse().unwrap_or_else(|_| usage()),
            "--threads" => o.threads = list(&v()),
            "--qd" => o.qds = list(&v()),
            "--gather" => o.gathers = list(&v()),
            "--seconds" => o.dur = Duration::from_secs_f64(v().parse().unwrap_or_else(|_| usage())),
            "--name" => o.name = v(),
            "--copy" => o.copy = true,
            "--isolation" => o.isolation = true,
            _ => usage(),
        }
    }
    if let Some(gb) = local {
        let n = Node::new(
            &Config { bytes: (gb * (1u64 << 30) as f64) as usize, lock: false, rdma: true, qp_timeout: 14 },
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap_or_else(|e| panic!("local node: {e}"));
        o.server = n.spawn().to_string();
    }
    if o.server.is_empty() {
        usage();
    }
    if !o.server.contains(':') {
        o.server += ":7350";
    }
    if let Err(e) = run(&o) {
        eprintln!("kvman-remote-bench: {e}");
        std::process::exit(1);
    }
}

fn gbs(bytes: u64, t: Duration) -> f64 {
    bytes as f64 / t.as_secs_f64() / 1e9
}

fn pct(lat: &[u64], p: f64) -> f64 {
    if lat.is_empty() {
        return 0.0;
    }
    lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)] as f64
}

fn run(o: &Args) -> std::io::Result<()> {
    let blocks = (o.gb * 1e9) as u64 / PAGE_SLOT as u64;
    let lanes = o.threads.iter().chain(&o.qds).copied().max().unwrap_or(1).max(1);
    let s = RemoteStore::open(&o.server, &Opts { want: o.want, lanes, ..Opts::new(&o.name, PAGE_SLOT, blocks) })?;
    println!(
        "lease {} on {} from {}: {} page blocks of {} B ({:.2} GB), {} lanes, {}{}{}",
        o.name,
        s.server,
        s.local.ip(),
        blocks,
        PAGE_SLOT,
        (blocks * PAGE_SLOT as u64) as f64 / 1e9,
        s.lanes(),
        match s.transport {
            Transport::Rdma => "RDMA",
            Transport::Tcp => "TCP",
        },
        s.device.as_ref().map_or(String::new(), |(d, g)| format!(" on {d} GID {g}")),
        s.fallback.as_ref().map_or(String::new(), |f| format!(" (RDMA unavailable: {f})")),
    );
    if o.isolation {
        isolation(o)?;
    }
    let mut bufs: Vec<Mutex<AlignedBuf>> = (0..lanes).map(|_| Mutex::new(AlignedBuf::new(PAGE_SLOT))).collect();
    for (i, b) in bufs.iter_mut().enumerate() {
        let b = b.get_mut().unwrap();
        b.as_mut_slice().iter_mut().enumerate().for_each(|(j, x)| *x = (i * 131 + j) as u8);
        if !o.copy {
            s.register(b.as_mut_slice().as_mut_ptr(), b.as_slice().len())?;
        }
    }
    // fill once, so reads see written blocks
    let t0 = Instant::now();
    par_for(lanes.min(8), blocks as usize, |w, i| {
        s.write_block(i as u64, &bufs[w].lock().unwrap().as_slice()[..PAGE_SLOT])
    })?;
    println!("fill  {:>3} threads  {:>6.2} GB/s", lanes.min(8), gbs(blocks * PAGE_SLOT as u64, t0.elapsed()));
    println!("page blocks: GB/s, and ms per block");
    println!("{:>8} {:>9} {:>8} {:>8} {:>9} {:>8} {:>8}", "threads", "write", "p50", "p99", "read", "p50", "p99");
    for &t in &o.threads {
        let mut row = Vec::new();
        for write in [true, false] {
            let lat = Mutex::new(Vec::new());
            let (t0, mut moved) = (Instant::now(), 0u64);
            while moved == 0 || t0.elapsed() < o.dur {
                par_for(t, blocks as usize, |w, i| {
                    let mut b = bufs[w].lock().unwrap();
                    let t1 = Instant::now();
                    if write {
                        s.write_block(i as u64, &b.as_slice()[..PAGE_SLOT])?;
                    } else {
                        s.read_block(i as u64, &mut b.as_mut_slice()[..PAGE_SLOT])?;
                    }
                    lat.lock().unwrap().push(t1.elapsed().as_nanos() as u64);
                    Ok(())
                })?;
                moved += blocks * PAGE_SLOT as u64;
            }
            let el = t0.elapsed();
            let mut lat = lat.into_inner().unwrap();
            lat.sort_unstable();
            row.push((gbs(moved, el), pct(&lat, 0.5) / 1e6, pct(&lat, 0.99) / 1e6));
        }
        println!(
            "{t:>8} {:>9.2} {:>8.3} {:>8.3} {:>9.2} {:>8.3} {:>8.3}",
            row[0].0, row[0].1, row[0].2, row[1].0, row[1].1, row[1].2
        );
    }
    let lease = blocks * PAGE_SLOT as u64;
    for (i, &qd) in o.qds.iter().enumerate() {
        if i == 0 {
            println!("random row reads ({ROW_BYTES} B), one thread per read in flight");
            println!(
                "{:>6} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "qd", "reads/s", "GB/s", "p50 us", "p90 us", "p99 us", "p999 us"
            );
        }
        let t0 = Instant::now();
        let lats: Vec<std::io::Result<Vec<u64>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..qd)
                .map(|w| {
                    let (s, bufs) = (&s, &bufs);
                    sc.spawn(move || {
                        let mut b = bufs[w].lock().unwrap();
                        let mut rng = Rng(w as u64 * 7919 + 17);
                        let mut lat = Vec::new();
                        while t0.elapsed() < o.dur {
                            let off = rng.next() % (lease - ROW_BYTES as u64);
                            let t = Instant::now();
                            s.read_at_off(off, &mut b.as_mut_slice()[..ROW_BYTES])?;
                            lat.push(t.elapsed().as_nanos() as u64);
                        }
                        Ok(lat)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(std::io::Error::other("reader panicked")))).collect()
        });
        let t = t0.elapsed();
        let mut all = Vec::new();
        for l in lats {
            all.extend(l?);
        }
        all.sort_unstable();
        println!(
            "{qd:>6} {:>10.0} {:>8.2} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            all.len() as f64 / t.as_secs_f64(),
            gbs(all.len() as u64 * ROW_BYTES as u64, t),
            pct(&all, 0.5) / 1e3,
            pct(&all, 0.9) / 1e3,
            pct(&all, 0.99) / 1e3,
            pct(&all, 0.999) / 1e3
        );
    }
    if !o.gathers.is_empty() {
        println!("gathers: N random rows in one request, one thread");
        println!("{:>6} {:>10} {:>8} {:>8} {:>8}", "rows", "gathers/s", "GB/s", "p50 ms", "p99 ms");
        let most = o.gathers.iter().copied().max().unwrap() * ROW_BYTES;
        let mut out = AlignedBuf::new(most);
        if !o.copy {
            s.register(out.as_mut_slice().as_mut_ptr(), out.as_slice().len())?;
        }
        let mut rng = Rng(99);
        for &n in &o.gathers {
            let mut lat = Vec::new();
            let t0 = Instant::now();
            while t0.elapsed() < o.dur {
                let ranges: Vec<(u64, usize)> =
                    (0..n).map(|_| (rng.next() % (lease - ROW_BYTES as u64), ROW_BYTES)).collect();
                let t = Instant::now();
                s.gather(&ranges, &mut out.as_mut_slice()[..n * ROW_BYTES])?;
                lat.push(t.elapsed().as_nanos() as u64);
            }
            let t = t0.elapsed();
            lat.sort_unstable();
            println!(
                "{n:>6} {:>10.0} {:>8.2} {:>8.3} {:>8.3}",
                lat.len() as f64 / t.as_secs_f64(),
                gbs(lat.len() as u64 * (n * ROW_BYTES) as u64, t),
                pct(&lat, 0.5) / 1e6,
                pct(&lat, 0.99) / 1e6
            );
        }
    }
    Ok(())
}

/// Two small leases. Lease A's QPs try lease B's region with B's own rkey and address, and A's region one byte past
/// its end: the NIC must refuse both. A read of A's own region is the control.
fn isolation(o: &Args) -> std::io::Result<()> {
    let open = |n: &str| {
        RemoteStore::open(
            &o.server,
            &Opts { want: Want::Rdma, lanes: 3, ..Opts::new(&format!("{}-{n}", o.name), 4096, 512) },
        )
    };
    let (a, b) = (open("iso-a")?, open("iso-b")?);
    b.write_block(0, &[0x5a; 4096])?;
    let (aa, ak) = a.rdma_target().expect("RDMA");
    let (ba, bk) = b.rdma_target().expect("RDMA");
    let say = |what: &str, r: std::io::Result<()>, want_ok: bool| {
        let verdict = match (&r, want_ok) {
            (Ok(()), true) | (Err(_), false) => "as it must",
            _ => "WRONG",
        };
        println!("isolation: {what}: {} ({verdict})", r.map_or_else(|e| format!("refused, {e}"), |_| "read".into()));
        verdict == "as it must"
    };
    let mut ok = say("lease A reads its own region", a.raw_read(aa, ak, 4096), true);
    ok &= say(&format!("lease A reads lease B's region (rkey {bk:#x})"), a.raw_read(ba, bk, 4096), false);
    ok &= say("lease A reads one block past its end", a.raw_read(aa + a.lease_bytes, ak, 4096), false);
    if !ok {
        return Err(std::io::Error::other("the isolation check failed"));
    }
    Ok(())
}
