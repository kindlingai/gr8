//! The KV node and its client over TCP on loopback: leases keep clients apart, a name has one holder, the node checks
//! every offset, and a Tier spills to and restores from a remote store as from the SSD. RDMA needs the hardware:
//! kvman-remote-bench --isolation checks it there.

use g6kvman::pool::Kept;
use g6kvman::remote::proto::{self, Header};
use g6kvman::remote::{client, Config, Node, Opts, RemoteStore, Want};
use g6kvman::reuse::Finished;
use g6kvman::tier::{BlockStore, Tier};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

const MIB: usize = 1 << 20;

fn node(mib: usize) -> String {
    let n = Node::new(
        &Config { bytes: mib * MIB, lock: false, rdma: false, qp_timeout: 14 },
        "127.0.0.1:0".parse().unwrap(),
    )
    .unwrap();
    n.spawn().to_string()
}

fn opts(name: &str, block: usize, blocks: u64) -> Opts {
    Opts { want: Want::Tcp, lanes: 3, ..Opts::new(name, block, blocks) }
}

fn fill(tag: u8, block: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u64 * 7 + block * 131 + tag as u64 * 29) as u8).collect()
}

/// Waits until the node holds `leases` leases: a dropped store's connections close asynchronously.
fn until_free(addr: &str, leases: usize) {
    for _ in 0..200 {
        if client::stats(addr).unwrap().matches("\"id\":").count() == leases {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the node still holds leases: {}", client::stats(addr).unwrap());
}

#[test]
fn two_leases_never_see_each_other() {
    let addr = node(16);
    let (bb, n) = (100_000, 20);
    let a = RemoteStore::open(&addr, &opts("box-a/rank0", bb, n)).unwrap();
    let b = RemoteStore::open(&addr, &opts("box-a/rank1", bb, n)).unwrap();
    // a new lease reads zeros
    let mut got = vec![1u8; bb];
    b.read_block(n - 1, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 0));
    for blk in 0..n {
        a.write_block(blk, &fill(1, blk, bb)).unwrap();
        b.write_block(blk, &fill(2, blk, bb)).unwrap();
    }
    for blk in 0..n {
        a.read_block(blk, &mut got).unwrap();
        assert_eq!(got, fill(1, blk, bb));
        b.read_block(blk, &mut got).unwrap();
        assert_eq!(got, fill(2, blk, bb));
    }
    assert_eq!(a.write_block(n, &got).unwrap_err().kind(), ErrorKind::InvalidInput);
    let mut row = vec![0u8; 2624];
    b.read_at(3, 5000, &mut row).unwrap();
    assert_eq!(row[..], fill(2, 3, bb)[5000..5000 + 2624]);
}

#[test]
fn a_name_has_one_holder() {
    let addr = node(8);
    let a = RemoteStore::open(&addr, &opts("rank0", 4096, 16)).unwrap();
    a.write_block(0, &[9u8; 4096]).unwrap();
    let e = RemoteStore::open(&addr, &opts("rank0", 4096, 16)).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::AddrInUse);
    assert!(e.to_string().contains("held"), "{e}");
    drop(a);
    until_free(&addr, 0);
    // the name is free again, and the memory it held comes back zeroed
    let a = RemoteStore::open(&addr, &opts("rank0", 4096, 16)).unwrap();
    let mut got = vec![1u8; 4096];
    a.read_block(0, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 0));
}

#[test]
fn the_node_checks_every_offset() {
    let addr = node(8);
    let a = RemoteStore::open(&addr, &opts("checked", 4096, 4)).unwrap();
    let lease = a.lease_bytes;
    assert_eq!(lease, 2 * MIB as u64);
    // a raw connection: no lease, a wrong token, then reads past the lease
    let mut s = TcpStream::connect(&addr).unwrap();
    Header::new(proto::READ, 0, 16, 0).write(&mut s).unwrap();
    assert_eq!(proto::reply(&mut s).unwrap_err().kind(), ErrorKind::PermissionDenied);
    let mut s = TcpStream::connect(&addr).unwrap();
    Header::new(proto::ATTACH, a.lease_id, 0, 0).write(&mut s).unwrap();
    s.write_all(&[0u8; 16]).unwrap();
    assert_eq!(proto::reply(&mut s).unwrap_err().kind(), ErrorKind::PermissionDenied);
    for (off, len) in [(lease, 1), (lease - 8, 16), (u64::MAX - 4, 8)] {
        assert_eq!(
            a.read_at_off(off, &mut vec![0u8; len as usize]).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{off} {len}"
        );
        assert_eq!(
            a.gather(&[(0, 8), (off, len as usize)], &mut vec![0u8; 8 + len as usize]).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    // the last byte of the lease is in range
    a.read_at_off(lease - 1, &mut [0u8; 1]).unwrap();
}

#[test]
fn a_write_past_the_lease_ends_its_connection() {
    let addr = node(8);
    let a = RemoteStore::open(&addr, &opts("w", 4096, 4)).unwrap();
    let b = RemoteStore::open(&addr, &opts("neighbour", 4096, 4)).unwrap();
    b.write_block(0, &[5u8; 4096]).unwrap();
    let e = a.write_at(a.lease_bytes - 100, &[1u8; 4096]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    let mut got = vec![0u8; 4096];
    b.read_block(0, &mut got).unwrap();
    assert!(got.iter().all(|&x| x == 5));
}

#[test]
fn no_room_says_what_is_free() {
    let addr = node(8);
    let _a = RemoteStore::open(&addr, &opts("big", MIB, 6)).unwrap();
    let e = RemoteStore::open(&addr, &opts("bigger", MIB, 4)).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::OutOfMemory);
    assert!(e.to_string().contains("2097152 free"), "{e}");
    let mut s = TcpStream::connect(&addr).unwrap();
    Header::new(proto::STATS, 0, 0, 0).write(&mut s).unwrap();
    let h = proto::reply(&mut s).unwrap();
    let mut j = vec![0u8; h.a as usize];
    s.read_exact(&mut j).unwrap();
    let j = String::from_utf8(j).unwrap();
    assert!(j.contains("\"name\":\"big\"") && j.contains("\"free\":2097152"), "{j}");
}

#[test]
fn gather_returns_ranges_in_order() {
    let addr = node(8);
    let a = RemoteStore::open(&addr, &opts("g", 8192, 8)).unwrap();
    for blk in 0..8 {
        a.write_block(blk, &fill(3, blk, 8192)).unwrap();
    }
    let ranges: Vec<(u64, usize)> = (0..40u64).map(|i| ((i * 7919) % 60_000, 2624 - (i as usize % 3))).collect();
    let mut out = vec![0u8; ranges.iter().map(|r| r.1).sum()];
    a.gather(&ranges, &mut out).unwrap();
    let all: Vec<u8> = (0..8).flat_map(|b| fill(3, b, 8192)).collect();
    let mut at = 0;
    for &(o, n) in &ranges {
        assert_eq!(out[at..at + n], all[o as usize..o as usize + n]);
        at += n;
    }
}

#[test]
fn a_tier_spills_to_a_remote_store() {
    let addr = node(16);
    let store = RemoteStore::open(&addr, &opts("tier", 3 * 4096, 16)).unwrap();
    let mut t = Tier::new(store, 3);
    let block =
        |p: u32, buf: &mut [u8]| buf.iter_mut().enumerate().for_each(|(i, b)| *b = (p as usize * 31 + i * 7) as u8);
    let a = Kept {
        list: vec![5, 2, 9, 0],
        st: Finished { valid: 100, ..Default::default() },
        age: 1,
        exclusive: false,
        hits: 0,
        namespace: g6kvman::pool::STOCK_NAMESPACE,
    };
    let ha = t.spill(&a, block).unwrap().unwrap();
    assert_eq!(t.spilled()[&ha].slots, vec![0, 1, 2, 3]);
    let got = std::sync::Mutex::new(Vec::new());
    let st = t.restore(ha, &[40, 41, 42], |p, buf| got.lock().unwrap().push((p, buf.to_vec()))).unwrap();
    assert_eq!(st.valid, 100);
    let mut got = got.into_inner().unwrap();
    got.sort_by_key(|g| g.0);
    for (i, (p, buf)) in got.iter().enumerate() {
        let mut want = vec![0u8; 3 * 4096];
        block(a.list[i], &mut want);
        assert_eq!((*p, buf), (40 + i as u32, &want));
    }
    assert_eq!(t.free_slots(), 16);
    assert_eq!(t.store().blocks(), 16);
}

/// A forwarder between a client and the node that can freeze (stop moving bytes, the node SIGSTOPped) or cut every
/// connection (the node killed).
struct Proxy {
    addr: String,
    frozen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    conns: std::sync::Arc<std::sync::Mutex<Vec<TcpStream>>>,
}

impl Proxy {
    fn new(node: &str) -> Proxy {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let frozen = Arc::new(AtomicBool::new(false));
        let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
        let (node, f, cs) = (node.to_string(), frozen.clone(), conns.clone());
        std::thread::spawn(move || {
            for c in l.incoming() {
                let Ok(c) = c else { return };
                let n = TcpStream::connect(&node).unwrap();
                cs.lock().unwrap().extend([c.try_clone().unwrap(), n.try_clone().unwrap()]);
                for (mut from, mut to) in [(c.try_clone().unwrap(), n.try_clone().unwrap()), (n, c)] {
                    let f = f.clone();
                    std::thread::spawn(move || {
                        let mut b = vec![0u8; 1 << 16];
                        loop {
                            while f.load(Ordering::Relaxed) {
                                std::thread::sleep(std::time::Duration::from_millis(5));
                            }
                            match from.read(&mut b) {
                                Ok(0) | Err(_) => break,
                                Ok(k) => {
                                    while f.load(Ordering::Relaxed) {
                                        std::thread::sleep(std::time::Duration::from_millis(5));
                                    }
                                    if to.write_all(&b[..k]).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        let _ = to.shutdown(std::net::Shutdown::Both);
                    });
                }
            }
        });
        Proxy { addr, frozen, conns }
    }
    fn freeze(&self) {
        self.frozen.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    fn cut(&self) {
        for c in self.conns.lock().unwrap().iter() {
            let _ = c.shutdown(std::net::Shutdown::Both);
        }
    }
}

#[test]
fn a_frozen_node_times_out_and_the_lane_stays_failed() {
    let addr = node(4);
    let p = Proxy::new(&addr);
    let o = Opts { timeout: std::time::Duration::from_millis(500), ..opts("frozen", 8192, 8) };
    let s = RemoteStore::open(&p.addr, &o).unwrap();
    s.write_block(1, &fill(3, 1, 8192)).unwrap();
    p.freeze();
    let t0 = std::time::Instant::now();
    let mut got = vec![0u8; 8192];
    let e = s.read_block(1, &mut got).unwrap_err();
    let took = t0.elapsed();
    assert_eq!(e.kind(), ErrorKind::TimedOut, "{e}");
    assert!(e.to_string().contains("timed out"), "{e}");
    assert!(took >= std::time::Duration::from_millis(400) && took < std::time::Duration::from_secs(5), "{took:?}");
    // every lane fails within the timeout too, and a lane that timed out fails at once after
    for _ in 0..o.lanes + 2 {
        let t0 = std::time::Instant::now();
        assert!(s.write_block(2, &fill(3, 2, 8192)).is_err());
        assert!(t0.elapsed() < std::time::Duration::from_secs(2));
    }
}

#[test]
fn a_killed_node_fails_calls_at_once() {
    let addr = node(4);
    let p = Proxy::new(&addr);
    let s = RemoteStore::open(&p.addr, &opts("killed", 8192, 8)).unwrap();
    s.write_block(0, &fill(4, 0, 8192)).unwrap();
    p.cut();
    let t0 = std::time::Instant::now();
    let mut got = vec![0u8; 8192];
    assert!(s.read_block(0, &mut got).is_err());
    assert!(s.write_block(0, &got).is_err());
    assert!(t0.elapsed() < std::time::Duration::from_secs(2), "{:?}", t0.elapsed());
}
