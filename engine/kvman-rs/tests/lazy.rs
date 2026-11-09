//! --lazy-lock and the node's memory guard (server.rs) over TCP on loopback: a lease's range is backed (and locked,
//! where RLIMIT_MEMLOCK allows) only while the lease lives, MemAvailable comes back when it ends, a new lease reads
//! zeros under load on the others, and a grant that would leave the box under --min-free-gb is refused with E_NOROOM.

use g6kvman::remote::{client, sys, Config, GroupOpts, Node, Opts, RemoteStore, Share, Want};
use g6kvman::tier::BlockStore;
use std::io::ErrorKind;
use std::time::{Duration, Instant};

const MIB: usize = 1 << 20;

// one test at a time: they measure this process's memory and the box's MemAvailable
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn one() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

fn lazy_node(mib: usize, min_free: u64, guard_every: Duration) -> String {
    // lock where this box allows it (outside containers); the backing is the same either way
    let cfg = Config { bytes: mib * MIB, lock: locks(mib), lazy: true, min_free, guard_every, ..Default::default() };
    Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).unwrap().spawn().to_string()
}

fn opts(name: &str, block: usize, blocks: u64, lanes: usize) -> Opts {
    Opts { want: Want::Tcp, lanes, ..Opts::new(name, block, blocks) }
}

fn field(stats: &str, k: &str) -> u64 {
    let i = stats.find(&format!("\"{k}\":")).unwrap_or_else(|| panic!("no {k} in {stats}")) + k.len() + 3;
    let v: String = stats[i..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    match v.as_str() {
        "true" => 1,
        "false" => 0,
        _ => v.parse().unwrap(),
    }
}

fn fill(tag: u8, block: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u64 * 7 + block * 131 + tag as u64 * 29) as u8).collect()
}

// a /proc/self/status line, bytes
fn status(k: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let l = s.lines().find(|l| l.starts_with(k)).unwrap();
    l.split_whitespace().nth(1).unwrap().parse::<u64>().unwrap() * 1024
}

fn rss_anon() -> u64 {
    status("RssAnon:")
}

// whether a node of this size locks its arena here
fn locks(mib: usize) -> bool {
    sys::is_root() || !sys::memlock_limit().is_some_and(|l| l < (mib * MIB) as u64 * 2)
}

fn until_held(addr: &str, held: u64) {
    let t = Instant::now();
    while field(&client::stats(addr).unwrap(), "held") != held {
        assert!(t.elapsed() < Duration::from_secs(2), "held stays {}", client::stats(addr).unwrap());
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_lease_is_backed_only_while_it_lives() {
    let _one = one();
    let seg = 512 * MIB;
    let addr = lazy_node(1024, 0, Duration::ZERO);
    let s = client::stats(&addr).unwrap();
    assert_eq!((field(&s, "lazy"), field(&s, "held"), field(&s, "locked")), (1, 0, 0), "{s}");
    let (r0, a0) = (rss_anon(), sys::mem_available().unwrap());
    let a = RemoteStore::open(&addr, &opts("lazy/a", MIB, (seg / MIB) as u64, 2)).unwrap();
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), seg as u64);
    assert!(rss_anon() >= r0 + seg as u64 * 9 / 10, "the lease's range is backed");
    if locks(1024) {
        assert!(status("VmLck:") >= seg as u64, "and locked");
    }
    let mut got = vec![1u8; MIB];
    a.read_block(511, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 0), "a new lease reads zeros");
    a.write_block(511, &fill(1, 511, MIB)).unwrap();
    let a1 = sys::mem_available().unwrap();
    drop(a);
    until_held(&addr, 0);
    // the box gets the bytes back within a second: the process's anonymous memory exactly, MemAvailable (which other
    // processes move too) by most of them
    let t = Instant::now();
    let mut best = 0;
    while t.elapsed() < Duration::from_secs(1) {
        best = best.max(sys::mem_available().unwrap());
        if best >= a1 + seg as u64 / 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(rss_anon() + seg as u64 * 9 / 10 <= r0 + seg as u64, "the range went back to the kernel");
    assert!(status("VmLck:") < seg as u64 / 10, "and is unlocked");
    assert!(best >= a1 + seg as u64 / 2, "MemAvailable {a0} -> {a1} with the lease, at most {best} a second after");
    // the next holder of the range reads zeros, not the last holder's bytes
    let b = RemoteStore::open(&addr, &opts("lazy/b", MIB, (seg / MIB) as u64, 2)).unwrap();
    b.read_block(511, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 0));
}

#[test]
fn leases_stay_apart_while_another_comes_and_goes_under_load() {
    let _one = one();
    let addr = lazy_node(256, 0, Duration::ZERO);
    let (bb, n) = (256 * 1024, 64u64);
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|sc| {
        // two leases of 4 lanes each, every block written and read back by its own thread
        let mut busy = Vec::new();
        for tag in 1..=2u8 {
            let addr = &addr;
            let stop = &stop;
            busy.push(sc.spawn(move || {
                let s = RemoteStore::open(addr, &opts(&format!("load/{tag}"), bb, n, 4)).unwrap();
                let mut got = vec![0u8; bb];
                let mut rounds = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) || rounds < 2 {
                    for blk in 0..n {
                        s.write_block(blk, &fill(tag + rounds as u8, blk, bb)).unwrap();
                    }
                    for blk in 0..n {
                        s.read_block(blk, &mut got).unwrap();
                        assert_eq!(got, fill(tag + rounds as u8, blk, bb), "lease {tag} block {blk}");
                    }
                    rounds += 1;
                }
                rounds
            }));
        }
        // a third lease granted, filled and ended over and over: each grant reads zeros. The load stops however this
        // ends (a failed check here must not leave the load threads running)
        struct Stop<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let halt = Stop(&stop);
        let mut got = vec![1u8; bb];
        for c in 0..20u64 {
            // the last cycle's lease ends on the node's side a moment after its connections close
            let t = Instant::now();
            let s = loop {
                match RemoteStore::open(&addr, &opts("load/cycle", bb, n, 2)) {
                    Err(e) if e.kind() == ErrorKind::AddrInUse && t.elapsed() < Duration::from_secs(2) => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    r => break r.unwrap(),
                }
            };
            for blk in [0, n / 2, n - 1] {
                s.read_block(blk, &mut got).unwrap();
                assert!(got.iter().all(|&x| x == 0), "cycle {c} block {blk}: a granted range holds old bytes");
                s.write_block(blk, &fill(9, blk, bb)).unwrap();
            }
            drop(s);
        }
        drop(halt);
        for b in busy {
            assert!(b.join().unwrap() >= 2);
        }
    });
    until_held(&addr, 0);
}

#[test]
fn a_grant_that_would_leave_the_box_short_is_refused() {
    let _one = one();
    let avail = sys::mem_available().unwrap();
    // the box is already short: every grant is refused, and the guard says so without one
    let addr = lazy_node(64, avail + (64u64 << 30), Duration::from_millis(20));
    let t = Instant::now();
    while field(&client::stats(&addr).unwrap(), "low") != 1 {
        assert!(t.elapsed() < Duration::from_secs(1), "the guard never ran");
        std::thread::sleep(Duration::from_millis(5));
    }
    let e = RemoteStore::open(&addr, &opts("short/a", MIB, 4, 1)).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::OutOfMemory);
    assert!(e.to_string().contains("--min-free-gb"), "{e}");
    let s = client::stats(&addr).unwrap();
    assert_eq!((field(&s, "refused"), field(&s, "held")), (1, 0), "{s}");
    // the box has room, but not for this lease: 1 GiB would cross the floor, 32 MiB does not
    let avail = sys::mem_available().unwrap();
    let addr = lazy_node(2048, avail - (256u64 << 20), Duration::from_secs(1));
    let e = RemoteStore::open(&addr, &opts("short/big", MIB, 1024, 1)).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::OutOfMemory, "{e}");
    let small = RemoteStore::open(&addr, &opts("short/small", MIB, 32, 1)).unwrap();
    small.write_block(0, &[3u8; MIB]).unwrap();
    let s = client::stats(&addr).unwrap();
    assert_eq!((field(&s, "refused"), field(&s, "held")), (1, 32 * MIB as u64), "{s}");
}

#[test]
fn without_lazy_lock_the_floor_is_only_the_start_check() {
    let _one = one();
    // today's node: the whole arena is backed from the start, and a running node grants whatever MemAvailable says
    let cfg =
        Config { bytes: 16 * MIB, min_free: u64::MAX, guard_every: Duration::from_millis(5), ..Default::default() };
    let addr = Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).unwrap().spawn().to_string();
    let s = RemoteStore::open(&addr, &opts("eager/a", MIB, 8, 1)).unwrap();
    s.write_block(7, &[5u8; MIB]).unwrap();
    let st = client::stats(&addr).unwrap();
    assert_eq!((field(&st, "lazy"), field(&st, "held"), field(&st, "refused")), (0, 16 * MIB as u64, 0), "{st}");
}

#[test]
fn with_a_ledger_the_node_holds_no_more_than_it_is_credited() {
    use g6kvman::ledger::{Book, Ledger, Side};
    let _one = one();
    let d = std::env::temp_dir().join(format!("kvman-lazy-ledger-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let file = d.join("ledger");
    let led = Ledger::at(&file);
    let m = MIB as u64;
    led.init(&Book::new(16 * m, 12 * m, 4 * m, 0).unwrap()).unwrap();
    let cfg = Config { bytes: 16 * MIB, lazy: true, ledger: Some(file.clone()), ..Default::default() };
    let addr = Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).unwrap().spawn().to_string();
    let a = RemoteStore::open(&addr, &opts("led/a", MIB, 4, 1)).unwrap();
    // the arena has room, the ledger does not
    let e = RemoteStore::open(&addr, &opts("led/b", MIB, 2, 1)).err().unwrap();
    assert!(e.to_string().contains("ledger credits the node"), "{e}");
    // the GPU gives 2 MiB to the node: the next lease fits
    let g = led.apply(|b| b.begin(Side::Gpu, Side::Node, 2 * m, 0)).unwrap();
    led.apply(|b| b.released(g, 2 * m)).unwrap();
    led.apply(|b| b.commit(g, 2 * m)).unwrap();
    let b = RemoteStore::open(&addr, &opts("led/b", MIB, 2, 1)).unwrap();
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 6 * m);
    drop((a, b));
    until_held(&addr, 0);
    // a node without --lazy-lock cannot take a ledger
    let cfg = Config { bytes: 16 * MIB, ledger: Some(file), ..Default::default() };
    assert_eq!(Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).err().unwrap().kind(), ErrorKind::InvalidInput);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_group_lease_backs_the_segments_it_grows_into_and_gives_back_the_ones_it_gives() {
    use g6kvman::ledger::{Book, Ledger, Side};
    let _one = one();
    let (seg, m) = (4 * MIB, MIB as u64);
    let d = std::env::temp_dir().join(format!("kvman-lazy-group-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let file = d.join("ledger");
    let led = Ledger::at(&file);
    led.init(&Book::new(64 * m, 40 * m, 24 * m, 0).unwrap()).unwrap();
    let share = Share { seg_bytes: seg, margin: Duration::from_secs(60), ..Share::default() };
    let cfg = Config { bytes: 64 * MIB, lazy: true, ledger: Some(file), share, ..Default::default() };
    let addr = Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).unwrap().spawn().to_string();
    let rank = |r: u32| {
        let group = Some(GroupOpts {
            group: "g".into(),
            nr: 2,
            rank: r,
            floor_bytes: seg as u64,
            cap_bytes: 8 * seg as u64,
            weight: 1000,
        });
        RemoteStore::open(&addr, &Opts { want: Want::Tcp, lanes: 2, group, ..Opts::new(&format!("g-r{r}"), MIB, 0) })
            .unwrap()
    };
    let c = [rank(0), rank(1)];
    let per = c[0].per_seg;
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 8 * m, "each rank's floor");
    let r = c[0].report(3, &[]).unwrap();
    assert_eq!(r.grant, 3);
    assert_eq!(c[0].grow(4, r.txn).unwrap(), 4 * per);
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 20 * m);
    // the ledger credits the node 24 MiB: the second rank's 12 MiB wait for the GPU to give
    let e = c[1].grow(4, r.txn).unwrap_err();
    assert!(e.to_string().contains("ledger credits the node"), "{e}");
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 20 * m, "a refused grow holds nothing more");
    let g = led.apply(|b| b.begin(Side::Gpu, Side::Node, 8 * m, 0)).unwrap();
    led.apply(|b| b.released(g, 8 * m)).unwrap();
    led.apply(|b| b.commit(g, 8 * m)).unwrap();
    assert_eq!(c[1].grow(4, r.txn).unwrap(), 4 * per);
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 32 * m);
    let mut got = vec![1u8; MIB];
    c[1].read_block(4 * per - 1, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 0), "a grown segment comes zeroed");
    for (t, s) in c.iter().enumerate() {
        for b in 0..s.blocks() {
            s.write_block(b, &fill(t as u8, b, MIB)).unwrap();
        }
    }
    // a give to the floor: each rank's 3 segments go back to the kernel as it gives them
    c[0].shrink(1, 0).unwrap();
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 20 * m);
    c[1].shrink(1, 0).unwrap();
    assert_eq!(field(&client::stats(&addr).unwrap(), "held"), 8 * m);
    for (t, s) in c.iter().enumerate() {
        for b in 0..per {
            s.read_block(b, &mut got).unwrap();
            assert!(got == fill(t as u8, b, MIB), "rank {t} block {b} kept");
        }
    }
    // grown again, the segments read zeros: what the group wrote there went back with them
    let r = c[0].report(1, &[]).unwrap();
    assert_eq!(r.grant, 1);
    for s in &c {
        assert_eq!(s.grow(2, r.txn).unwrap(), 2 * per);
        s.read_block(2 * per - 1, &mut got).unwrap();
        assert!(got.iter().all(|&x| x == 0), "a segment given and grown again comes zeroed");
    }
    drop(c);
    until_held(&addr, 0);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_transfer_from_the_node_waits_until_its_leases_fit_what_is_left() {
    use g6kvman::ledger::{Book, Ledger, Phase, Side};
    let _one = one();
    let m = MIB as u64;
    let d = std::env::temp_dir().join(format!("kvman-lazy-give-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let file = d.join("ledger");
    let led = Ledger::at(&file);
    led.init(&Book::new(16 * m, 8 * m, 8 * m, 0).unwrap()).unwrap();
    let cfg = Config {
        bytes: 16 * MIB,
        lazy: true,
        ledger: Some(file),
        guard_every: Duration::from_millis(10),
        ..Default::default()
    };
    let addr = Node::new(&cfg, "127.0.0.1:0".parse().unwrap()).unwrap().spawn().to_string();
    let a = RemoteStore::open(&addr, &opts("give/a", MIB, 4, 1)).unwrap();
    // the GPU asks 6 MiB of the node's 8: 4 are held, so the node cannot release yet, and grants no more
    let g = led.apply(|b| b.begin(Side::Node, Side::Gpu, 6 * m, 0)).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(led.show().unwrap().pending.unwrap().phase, Phase::Begun, "4 MiB held, 2 left to the node");
    let e = RemoteStore::open(&addr, &opts("give/b", MIB, 2, 1)).err().unwrap();
    assert!(e.to_string().contains("ledger credits the node 2097152"), "{e}");
    // the lease ends: the node releases the 6 MiB, and the GPU takes them
    drop(a);
    let t = Instant::now();
    while led.show().unwrap().pending.unwrap().phase != Phase::Released {
        assert!(t.elapsed() < Duration::from_secs(2), "not released: {:?}", led.show().unwrap());
        std::thread::sleep(Duration::from_millis(5));
    }
    led.apply(|b| b.commit(g, 6 * m)).unwrap();
    let b = led.show().unwrap();
    assert_eq!((b.gpu, b.node, b.free), (14 * m, 2 * m, 0));
    let _c = RemoteStore::open(&addr, &opts("give/c", MIB, 2, 1)).unwrap();
    assert!(RemoteStore::open(&addr, &opts("give/d", MIB, 1, 1)).is_err(), "2 MiB credited, 2 held");
    let _ = std::fs::remove_dir_all(&d);
}
