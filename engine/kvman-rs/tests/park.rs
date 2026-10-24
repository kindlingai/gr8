//! Lease parking (server.rs Share::holdover, D-331 M4): a lease opened with persist outlives its client by the node's
//! holdover, so an engine that restarts (a resize, a reload) finds its blocks as it left them. Over TCP on loopback.

use g6kvman::remote::{client, Config, GroupOpts, Node, Opts, RemoteStore, Share, Want};
use g6kvman::tier::BlockStore;
use std::time::Duration;

const MIB: usize = 1 << 20;
const SEG: usize = 4 * MIB;
const BLOCK: usize = 100_000;

fn node(segs: usize, holdover_ms: u64) -> String {
    let share = Share { seg_bytes: SEG, holdover: Duration::from_millis(holdover_ms), ..Share::default() };
    let n = Node::new(
        &Config { bytes: segs * SEG, lock: false, rdma: false, qp_timeout: 14, share },
        "127.0.0.1:0".parse().unwrap(),
    )
    .unwrap();
    n.spawn().to_string()
}

fn fixed(addr: &str, name: &str, blocks: u64, persist: bool) -> RemoteStore {
    RemoteStore::open(addr, &Opts { want: Want::Tcp, lanes: 2, persist, ..Opts::new(name, BLOCK, blocks) }).unwrap()
}

fn rank(addr: &str, r: u32, persist: bool) -> RemoteStore {
    let per = (SEG / BLOCK * BLOCK) as u64;
    let g = GroupOpts { group: "g".into(), nr: 2, rank: r, floor_bytes: per, cap_bytes: 4 * per, weight: 1000 };
    let o = Opts { want: Want::Tcp, lanes: 2, persist, group: Some(g), ..Opts::new(&format!("g-r{r}"), BLOCK, 0) };
    RemoteStore::open(addr, &o).unwrap()
}

fn fill(tag: u64, b: u64) -> Vec<u8> {
    (0..BLOCK).map(|i| (i as u64 * 7 + b * 131 + tag * 29) as u8).collect()
}

fn write(s: &RemoteStore, tag: u64) {
    for b in 0..s.blocks() {
        s.write_block(b, &fill(tag, b)).unwrap();
    }
}

fn holds(s: &RemoteStore, tag: u64) -> bool {
    let mut got = vec![0u8; BLOCK];
    (0..s.blocks()).all(|b| {
        s.read_block(b, &mut got).unwrap();
        got == fill(tag, b)
    })
}

fn zeroed(s: &RemoteStore) -> bool {
    let mut got = vec![1u8; BLOCK];
    (0..s.blocks()).all(|b| {
        s.read_block(b, &mut got).unwrap();
        got.iter().all(|&x| x == 0)
    })
}

fn unclaimed(addr: &str) -> u64 {
    let st = client::stats(addr).unwrap();
    let i = st.find("\"unclaimed_segs\":").expect("unclaimed_segs") + 17;
    st[i..].split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()
}

fn wait_unclaimed(addr: &str, n: u64) {
    for _ in 0..200 {
        if unclaimed(addr) == n {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("unclaimed stays {} (want {n}): {}", unclaimed(addr), client::stats(addr).unwrap());
}

#[test]
fn a_persist_lease_comes_back_whole_within_the_holdover() {
    let addr = node(8, 2_000);
    let a = fixed(&addr, "a", 30, true);
    assert!(!a.reattached);
    write(&a, 5);
    drop(a);
    // parked: its memory stays taken
    std::thread::sleep(Duration::from_millis(50));
    assert!(unclaimed(&addr) < 8);
    let a = fixed(&addr, "a", 30, true);
    assert!(a.reattached);
    assert!(holds(&a, 5), "the blocks as the last holder left them");
}

#[test]
fn past_the_holdover_the_lease_is_freed_and_a_new_one_is_zeroed() {
    let addr = node(8, 200);
    let a = fixed(&addr, "a", 30, true);
    write(&a, 6);
    drop(a);
    wait_unclaimed(&addr, 8);
    let a = fixed(&addr, "a", 30, true);
    assert!(!a.reattached);
    assert!(zeroed(&a));
}

#[test]
fn without_persist_or_without_a_holdover_nothing_is_parked() {
    let addr = node(8, 2_000);
    let a = fixed(&addr, "a", 30, false);
    write(&a, 7);
    drop(a);
    wait_unclaimed(&addr, 8);
    let a = fixed(&addr, "a", 30, false);
    assert!(!a.reattached && zeroed(&a));
    let addr = node(8, 0);
    let b = fixed(&addr, "b", 30, true);
    write(&b, 7);
    drop(b);
    wait_unclaimed(&addr, 8);
}

#[test]
fn a_parked_lease_of_another_shape_is_freed_not_handed_out() {
    let addr = node(8, 2_000);
    let a = fixed(&addr, "a", 30, true);
    write(&a, 8);
    drop(a);
    std::thread::sleep(Duration::from_millis(50));
    // another block count: the old lease goes, this one starts zeroed
    let a = fixed(&addr, "a", 60, true);
    assert!(!a.reattached && zeroed(&a));
}

#[test]
fn a_group_comes_back_with_its_segments_and_blocks() {
    let addr = node(8, 2_000);
    let g = [rank(&addr, 0, true), rank(&addr, 1, true)];
    let r = g[0].report(2, &[]).unwrap();
    for s in &g {
        s.grow(3, r.txn).unwrap();
    }
    for (i, s) in g.iter().enumerate() {
        write(s, 10 + i as u64);
    }
    drop(g);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(unclaimed(&addr), 2, "6 of 8 segments still the group's");
    let g = [rank(&addr, 0, true), rank(&addr, 1, true)];
    for (i, s) in g.iter().enumerate() {
        assert!(s.reattached);
        assert_eq!(s.segs(), 3);
        assert!(holds(s, 10 + i as u64), "rank {i}");
    }
    // still one group: it grows and gives as before
    let r = g[0].report(1, &[]).unwrap();
    assert_eq!(r.grant, 1);
}
