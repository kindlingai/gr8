//! Group leases on one KV node over TCP on loopback (server.rs GOPEN .. SEGS): two clusters share the node, capacity
//! moves between them by whole segments at their own pace, the coldest group gives, and blocks never mix or get lost.
//! RDMA (one region per segment) needs the hardware.

use g6kvman::remote::{client, coldest_ages, Config, GroupOpts, Node, Opts, RemoteStore, Share, Want};
use g6kvman::tier::BlockStore;
use std::io::ErrorKind;
use std::time::Duration;

const MIB: usize = 1 << 20;
const SEG: usize = 4 * MIB;

fn node(segs: usize, margin_ms: u64) -> String {
    let share = Share { seg_bytes: SEG, margin: Duration::from_millis(margin_ms), ..Share::default() };
    let n = Node::new(
        &Config { bytes: segs * SEG, lock: false, rdma: false, qp_timeout: 14, share },
        "127.0.0.1:0".parse().unwrap(),
    )
    .unwrap();
    n.spawn().to_string()
}

fn rank(addr: &str, group: &str, nr: u32, r: u32, block: usize, floor: u64, cap: u64) -> std::io::Result<RemoteStore> {
    let per = (SEG / block * block) as u64;
    let o = Opts {
        want: Want::Tcp,
        lanes: 2,
        group: Some(GroupOpts {
            group: group.into(),
            nr,
            rank: r,
            floor_bytes: floor * per,
            cap_bytes: cap * per,
            weight: 1000,
        }),
        ..Opts::new(&format!("{group}-r{r}"), block, 0)
    };
    RemoteStore::open(addr, &o)
}

fn cluster(addr: &str, group: &str, nr: u32, block: usize, floor: u64, cap: u64) -> Vec<RemoteStore> {
    (0..nr).map(|r| rank(addr, group, nr, r, block, floor, cap).unwrap()).collect()
}

fn fill(tag: u64, block: u64, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u64 * 7 + block * 131 + tag * 29) as u8).collect()
}

fn write_all(c: &[RemoteStore], tag: u64) {
    for (r, s) in c.iter().enumerate() {
        for b in 0..s.blocks() {
            s.write_block(b, &fill(tag + r as u64, b, s.block_bytes)).unwrap();
        }
    }
}

fn check(c: &[RemoteStore], tag: u64, blocks: u64) {
    for (r, s) in c.iter().enumerate() {
        let mut got = vec![0u8; s.block_bytes];
        for b in 0..blocks {
            s.read_block(b, &mut got).unwrap();
            assert!(got == fill(tag + r as u64, b, s.block_bytes), "rank {r} block {b}");
        }
    }
}

fn stats(addr: &str) -> String {
    client::stats(addr).unwrap()
}

#[test]
fn a_group_grows_into_its_grant_and_shrinks_without_touching_what_it_keeps() {
    let addr = node(16, 60_000);
    let a = cluster(&addr, "a", 2, 100_000, 1, 6);
    let per = a[0].per_seg;
    assert_eq!(per, (SEG / 100_000) as u64);
    assert_eq!(a[0].blocks(), per);
    // a new segment reads zeros
    write_all(&a, 10);
    let r = a[0].report(3, &[]).unwrap();
    assert_eq!((r.grant, r.give), (3, 0));
    for s in &a {
        assert_eq!(s.grow(3).unwrap(), 4 * per);
    }
    let mut z = vec![1u8; 100_000];
    a[1].read_block(4 * per - 1, &mut z).unwrap();
    assert!(z.iter().all(|&x| x == 0), "a grown segment comes zeroed");
    check(&a, 10, per);
    write_all(&a, 20);
    // the grant is used up: growing again needs a new one or spare segments
    for s in &a {
        s.shrink(2).unwrap();
    }
    assert_eq!(a[0].blocks(), 2 * per);
    check(&a, 20, 2 * per);
    assert_eq!(a[0].read_block(2 * per, &mut z).unwrap_err().kind(), ErrorKind::InvalidInput);
    // past the cap, below the floor
    let e = a[0].grow(5).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::OutOfMemory, "{e}");
    assert!(a[0].shrink(0).is_err());
    assert!(stats(&addr).contains("\"name\":\"a\""), "{}", stats(&addr));
}

#[test]
fn the_ranks_of_a_group_must_agree() {
    let addr = node(16, 60_000);
    let _a0 = rank(&addr, "a", 2, 0, 100_000, 1, 4).unwrap();
    assert_eq!(rank(&addr, "a", 3, 1, 100_000, 1, 4).err().unwrap().kind(), ErrorKind::InvalidData, "other nr");
    assert_eq!(rank(&addr, "a", 2, 1, 90_000, 1, 4).err().unwrap().kind(), ErrorKind::InvalidData, "other block");
    assert_eq!(rank(&addr, "a", 2, 1, 100_000, 2, 4).err().unwrap().kind(), ErrorKind::InvalidData, "other floor");
    assert_eq!(rank(&addr, "a", 2, 2, 100_000, 1, 4).err().unwrap().kind(), ErrorKind::InvalidData, "rank 2 of 2");
    let o = Opts {
        want: Want::Tcp,
        group: Some(GroupOpts {
            group: "a".into(),
            nr: 2,
            rank: 0,
            floor_bytes: 1,
            cap_bytes: 4 * (SEG as u64 / 100_000) * 100_000,
            weight: 1000,
        }),
        ..Opts::new("a-r0-again", 100_000, 0)
    };
    assert_eq!(RemoteStore::open(&addr, &o).err().unwrap().kind(), ErrorKind::AddrInUse, "rank 0 twice");
    // a group the node cannot hold every rank's floor of is refused whole
    let e = rank(&addr, "big", 4, 0, 100_000, 4, 4).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::OutOfMemory, "{e}");
    // the first rank held the others' floors: other leases cannot take them
    let _b = cluster(&addr, "b", 2, 100_000, 6, 6);
    assert_eq!(rank(&addr, "c", 1, 0, 100_000, 3, 3).err().unwrap().kind(), ErrorKind::OutOfMemory);
    let a1 = rank(&addr, "a", 2, 1, 100_000, 1, 4).unwrap();
    assert_eq!(a1.segs(), 1);
}

#[test]
fn the_colder_group_gives_and_the_other_grows_into_it() {
    let addr = node(8, 1_000);
    let a = cluster(&addr, "glm", 2, 100_000, 1, 8);
    let b = cluster(&addr, "flash", 2, 70_000, 1, 8);
    // glm takes everything that is left: 2 more segments a rank
    assert_eq!(a[0].report(9, &[0]).unwrap().grant, 2);
    for s in &a {
        s.grow(2).unwrap();
    }
    write_all(&a, 1);
    write_all(&b, 5);
    // flash wants a segment. glm's coldest block is 0.5 s old, flash's 0 s: inside the 1 s margin, nobody gives
    let r = b[0].report(1, &[0]).unwrap();
    assert_eq!((r.grant, r.give, r.unclaimed), (0, 0, 0));
    assert_eq!(a[0].report(0, &[500, 400]).unwrap().give, 0);
    // glm's blocks get colder: 5 s. flash asks again, and glm is asked for one segment a rank
    a[0].report(0, &coldest_ages(&[(1, 1), (2, 2 * a[0].per_seg as usize)], |id| 6_000 - id * 500, a[0].per_seg, 4))
        .unwrap();
    assert_eq!(b[0].report(1, &[0]).unwrap().grant, 0);
    assert_eq!(a[0].report(0, &[5_500]).unwrap().give, 1);
    // glm shrinks every rank (its tier evicted and moved down first); its kept blocks are intact
    for s in &a {
        s.shrink(2).unwrap();
    }
    check(&a, 1, 2 * a[0].per_seg);
    assert_eq!(a[0].report(0, &[5_500]).unwrap().give, 0, "given");
    let r = b[0].report(1, &[0]).unwrap();
    assert_eq!(r.grant, 1);
    for s in &b {
        s.grow(1).unwrap();
    }
    check(&b, 5, b[0].per_seg);
    // glm just gave: it is not asked again inside the cooldown, even when colder
    assert_eq!(b[0].report(1, &[0]).unwrap().grant, 0);
    assert_eq!(a[0].report(0, &[60_000]).unwrap().give, 0);
    let st = stats(&addr);
    assert!(st.contains("\"gave\":1") && st.contains("\"granted\":1"), "{st}");
}

#[test]
fn a_dead_cluster_frees_everything_it_held() {
    let addr = node(8, 1_000);
    let a = cluster(&addr, "a", 2, 100_000, 2, 4);
    assert!(rank(&addr, "b", 1, 0, 100_000, 5, 5).is_err());
    drop(a);
    for _ in 0..200 {
        if stats(&addr).contains("\"unclaimed_segs\":8") {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let st = stats(&addr);
    assert!(st.contains("\"unclaimed_segs\":8") && st.contains("\"groups\":[]"), "{st}");
    let b = rank(&addr, "b", 1, 0, 100_000, 5, 5).unwrap();
    assert_eq!(b.segs(), 5);
}

#[test]
fn ages_are_one_per_segment_of_blocks_coldest_first() {
    // entries of 3, 1, 5 and 2 blocks, 4 blocks a segment: blocks 0, 4 and 8 sit in entries 10, 30 and 30
    let lru = [(10, 3), (20, 1), (30, 5), (40, 2)];
    assert_eq!(coldest_ages(&lru, |id| id * 100, 4, 8), vec![1000, 3000, 3000]);
    assert_eq!(coldest_ages(&lru, |id| id, 4, 1), vec![10]);
    assert!(coldest_ages(&[], |id| id, 4, 8).is_empty());
}

// Two clusters grow and shrink against each other as fast as they can, each from its own thread, and after every
// change write a fresh pattern into every block they hold and read it all back, twice. A segment the node handed to two
// leases at once (or zeroed while a rank still held it) shows up as a block that does not read back.
#[test]
fn segments_that_change_hands_under_load_never_mix() {
    let addr = node(12, 60_000);
    let run = |group: &'static str, block: usize, seed: u64| {
        let addr = addr.clone();
        std::thread::spawn(move || {
            let c = cluster(&addr, group, 2, block, 1, 8);
            let mut x = seed;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            for i in 0..120u64 {
                let segs = c[0].segs();
                if next() % 2 == 0 {
                    let r = c[0].report(1 + next() % 3, &[]).unwrap();
                    for s in &c {
                        s.grow(r.grant).unwrap();
                    }
                } else if segs > 1 {
                    let to = segs - 1 - next() % segs.min(2).max(1).min(segs - 1);
                    for s in &c {
                        s.shrink(to.max(1)).unwrap();
                    }
                }
                let tag = seed * 1000 + i * 4;
                write_all(&c, tag);
                check(&c, tag, c[0].blocks());
                std::thread::yield_now();
                check(&c, tag, c[0].blocks());
            }
        })
    };
    let a = run("a", 100_000, 0x9e3779b97f4a7c15);
    let b = run("b", 70_000, 0x2545f4914f6cdd1d);
    a.join().unwrap();
    b.join().unwrap();
}
