//! kvman-ssd-bench: what the SSD tier (src/tier.rs) can move on this box, against a scratch file.
//!   kvman-ssd-bench --file PATH [--gb 8] [--threads 1,2,4,8,16] [--qd 1,2,4,8,16,32,64,128] [--seconds 3] [--keep]
//! Page blocks (one logical KV page of every layer, 3,506,176 B): write then read throughput, in slot order and in a
//! shuffled order, at each thread count. Rows (one 4 KiB slot each): random-read IOPS and latency percentiles at each
//! queue depth (one thread per outstanding read, synchronous pread), and 64 KiB random reads for scale. Every access is
//! O_DIRECT on Linux. The file is deleted at the end unless --keep.

use g6kvman::tier::{par_for, AlignedBuf, SlotFile, ALIGN, PAGE_SLOT};
use std::path::{Path, PathBuf};
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
        "usage: kvman-ssd-bench --file PATH [--gb 8] [--threads 1,2,4,8,16] [--qd 1,2,4,..,128] [--seconds 3] [--keep]"
    );
    std::process::exit(2)
}

fn main() {
    let mut file: Option<PathBuf> = None;
    let (mut gb, mut threads, mut qds, mut seconds, mut keep) =
        (8.0f64, vec![1, 2, 4, 8, 16], vec![1, 2, 4, 8, 16, 32, 64, 128], 3.0f64, false);
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--file" => file = Some(v().into()),
            "--gb" => gb = v().parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = list(&v()),
            "--qd" => qds = list(&v()),
            "--seconds" => seconds = v().parse().unwrap_or_else(|_| usage()),
            "--keep" => keep = true,
            _ => usage(),
        }
    }
    let Some(path) = file else { usage() };
    if let Err(e) = run(&path, gb, &threads, &qds, Duration::from_secs_f64(seconds)) {
        eprintln!("kvman-ssd-bench: {e}");
        let _ = std::fs::remove_file(&path);
        std::process::exit(1);
    }
    if !keep {
        let _ = std::fs::remove_file(&path);
    }
}

fn gbs(bytes: u64, t: Duration) -> f64 {
    bytes as f64 / t.as_secs_f64() / 1e9
}

fn run(path: &Path, gb: f64, threads: &[usize], qds: &[usize], dur: Duration) -> std::io::Result<()> {
    let slots = (gb * 1e9) as u64 / PAGE_SLOT as u64;
    let f = SlotFile::create(path, PAGE_SLOT, slots, true)?;
    println!(
        "file {} : {} page-block slots of {} B ({:.2} GB), O_DIRECT {}",
        path.display(),
        slots,
        PAGE_SLOT,
        (slots * PAGE_SLOT as u64) as f64 / 1e9,
        f.direct()
    );
    let tmax = threads.iter().copied().max().unwrap_or(8);
    let bufs: Vec<Mutex<AlignedBuf>> = (0..tmax.max(1)).map(|_| Mutex::new(AlignedBuf::new(PAGE_SLOT))).collect();
    for (i, b) in bufs.iter().enumerate() {
        b.lock().unwrap().as_mut_slice().iter_mut().enumerate().for_each(|(j, x)| *x = (i * 131 + j) as u8);
    }
    // fill the whole file once, so later reads hit written blocks
    let t0 = Instant::now();
    par_for(tmax, slots as usize, |w, i| f.write(i as u64, bufs[w].lock().unwrap().as_slice()))?;
    println!("fill  {tmax:>3} threads  {:>6.2} GB/s", gbs(slots * PAGE_SLOT as u64, t0.elapsed()));
    let mut shuffled: Vec<u64> = (0..slots).collect();
    let mut rng = Rng(1);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let seq: Vec<u64> = (0..slots).collect();
    println!("page blocks: GB/s (each pass moves the whole file)");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}", "threads", "write seq", "write rand", "read seq", "read rand");
    for &t in threads {
        let mut row = Vec::new();
        for write in [true, false] {
            for order in [&seq, &shuffled] {
                let t0 = Instant::now();
                par_for(t, order.len(), |w, i| {
                    let mut b = bufs[w].lock().unwrap();
                    if write {
                        f.write(order[i], b.as_slice())
                    } else {
                        f.read(order[i], b.as_mut_slice())
                    }
                })?;
                row.push(gbs(slots * PAGE_SLOT as u64, t0.elapsed()));
            }
        }
        println!("{t:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2}", row[0], row[1], row[2], row[3]);
    }
    println!("random reads (rows: one 4 KiB slot per token and indexer group)");
    println!(
        "{:>6} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "qd", "size", "IOPS", "GB/s", "p50 us", "p90 us", "p99 us", "p999 us"
    );
    for &(size, qd_list) in &[(ALIGN, qds), (16 * ALIGN, &[1usize, 8, 32, 64, 128][..])] {
        for &qd in qd_list {
            let (n, lat, t) = rand_reads(&f, size, qd, dur)?;
            let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)] as f64 / 1e3;
            println!(
                "{qd:>6} {:>6}K {:>10.0} {:>8.2} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
                size / 1024,
                n as f64 / t.as_secs_f64(),
                gbs(n * size as u64, t),
                pct(0.5),
                pct(0.9),
                pct(0.99),
                pct(0.999)
            );
        }
    }
    Ok(())
}

// qd threads, each with one synchronous read in flight, for `dur`: reads done, sorted latencies (ns), elapsed
fn rand_reads(f: &SlotFile, size: usize, qd: usize, dur: Duration) -> std::io::Result<(u64, Vec<u64>, Duration)> {
    let (slots, sb) = (f.slots(), f.slot_bytes() as u64);
    let places = (sb - size as u64) / ALIGN as u64 + 1;
    let t0 = Instant::now();
    let lats: Vec<std::io::Result<Vec<u64>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..qd)
            .map(|w| {
                s.spawn(move || {
                    let mut b = AlignedBuf::new(size);
                    let mut rng = Rng(w as u64 * 7919 + 17);
                    let mut lat = Vec::new();
                    while t0.elapsed() < dur {
                        let (slot, within) = (rng.next() % slots, rng.next() % places * ALIGN as u64);
                        let t = Instant::now();
                        f.read_at(slot, within, b.as_mut_slice())?;
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
    Ok((all.len() as u64, all, t))
}
