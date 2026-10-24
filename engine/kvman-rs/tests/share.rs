//! Group leases on one KV node over TCP on loopback (server.rs GOPEN .. SEGS): two clusters share the node, capacity
//! moves between them by whole segments at their own pace, the coldest group gives, and blocks never mix or get lost.
//! RDMA (one region per segment) needs the hardware.

use g6kvman::remote::{client, coldest_ages, parse_plan, Config, GroupOpts, Node, Opts, RemoteStore, Share, Want};
use g6kvman::tier::BlockStore;
use std::io::ErrorKind;
use std::time::Duration;

const MIB: usize = 1 << 20;
const SEG: usize = 4 * MIB;

fn node(segs: usize, margin_ms: u64) -> String {
    node_with(segs, Share { margin: Duration::from_millis(margin_ms), ..Share::default() })
}

fn node_with(segs: usize, share: Share) -> String {
    let share = Share { seg_bytes: SEG, ..share };
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
    (0..len).map(|i| (i as u64 * 7 + block * 131).wrapping_add(tag.wrapping_mul(29)) as u8).collect()
}

fn write_all(c: &[RemoteStore], tag: u64) {
    for (r, s) in c.iter().enumerate() {
        for b in 0..s.blocks() {
            s.write_block(b, &fill(tag.wrapping_add(r as u64), b, s.block_bytes)).unwrap();
        }
    }
}

fn check(c: &[RemoteStore], tag: u64, blocks: u64) {
    for (r, s) in c.iter().enumerate() {
        let mut got = vec![0u8; s.block_bytes];
        for b in 0..blocks {
            s.read_block(b, &mut got).unwrap();
            assert!(got == fill(tag.wrapping_add(r as u64), b, s.block_bytes), "rank {r} block {b}");
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
    let r = a[0].report(3, &[], &[]).unwrap();
    assert_eq!((r.grant, r.give), (3, 0));
    assert!(r.txn > 0);
    for s in &a {
        assert_eq!(s.grow(4, r.txn).unwrap(), 4 * per);
        // the same transaction again does nothing
        assert_eq!(s.grow(4, r.txn).unwrap(), 4 * per);
    }
    let mut z = vec![1u8; 100_000];
    a[1].read_block(4 * per - 1, &mut z).unwrap();
    assert!(z.iter().all(|&x| x == 0), "a grown segment comes zeroed");
    check(&a, 10, per);
    write_all(&a, 20);
    // a give the group makes on its own (txn 0): rank 0 opens it, rank 1 joins it
    for s in &a {
        s.shrink(2, 0).unwrap();
    }
    assert_eq!(a[0].blocks(), 2 * per);
    check(&a, 20, 2 * per);
    assert_eq!(a[0].read_block(2 * per, &mut z).unwrap_err().kind(), ErrorKind::InvalidInput);
    // a grow under no transaction, or one that ended, is stale; below the floor is refused
    let e = a[0].grow(5, 0).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::StaleNetworkFileHandle, "{e}");
    let e = a[0].grow(5, r.txn).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::StaleNetworkFileHandle, "{e}");
    assert!(a[0].shrink(0, 0).is_err());
    // past the cap: the grant stops at it
    let r = a[0].report(9, &[], &[]).unwrap();
    assert_eq!(r.grant, 4, "cap 6, at 2");
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
    let r = a[0].report(9, &[0], &[]).unwrap();
    assert_eq!(r.grant, 2);
    for s in &a {
        s.grow(3, r.txn).unwrap();
    }
    write_all(&a, 1);
    write_all(&b, 5);
    // flash wants a segment. glm's coldest block is 0.5 s old, flash's 0 s: inside the 1 s margin, nobody gives
    let r = b[0].report(1, &[0], &[]).unwrap();
    assert_eq!((r.grant, r.give, r.unclaimed), (0, 0, 0));
    assert_eq!(a[0].report(0, &[500, 400], &[]).unwrap().give, 0);
    // glm's blocks get colder: 5 s. flash asks again, and glm is asked for one segment a rank
    a[0].report(0, &coldest_ages(&[(1, 1), (2, 2 * a[0].per_seg as usize)], |id| 6_000 - id * 500, a[0].per_seg, 0, 4), &[])
        .unwrap();
    assert_eq!(b[0].report(1, &[0], &[]).unwrap().grant, 0);
    let r = a[0].report(0, &[5_500], &[]).unwrap();
    assert_eq!(r.give, 1);
    // glm shrinks every rank (its tier evicted and moved down first); its kept blocks are intact
    for s in &a {
        s.shrink(2, r.txn).unwrap();
    }
    check(&a, 1, 2 * a[0].per_seg);
    assert_eq!(a[0].report(0, &[5_500], &[]).unwrap().give, 0, "given");
    let r = b[0].report(1, &[0], &[]).unwrap();
    assert_eq!(r.grant, 1);
    for s in &b {
        s.grow(2, r.txn).unwrap();
    }
    check(&b, 5, b[0].per_seg);
    // glm just gave: it is not asked again inside the cooldown, even when colder
    assert_eq!(b[0].report(1, &[0], &[]).unwrap().grant, 0);
    assert_eq!(a[0].report(0, &[60_000], &[0, 700, u64::MAX]).unwrap().give, 0);
    let st = stats(&addr);
    assert!(st.contains("\"gave\":1") && st.contains("\"granted\":1"), "{st}");
    // the quotes a head sends show in STATS, beside its ages
    assert!(st.contains(&format!("\"ages_ms\":[60000],\"quote_rows\":[0,700,{}]", u64::MAX)), "{st}");
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
    assert_eq!(coldest_ages(&lru, |id| id * 100, 4, 0, 8), vec![1000, 1000, 1000]);
    // the head's clock disagrees with the order (entry 20 used last): the ages never rise
    assert_eq!(coldest_ages(&lru, |id| if id == 20 { 5 } else { 5000 - id }, 4, 0, 8), vec![4990, 5, 5]);
    assert_eq!(coldest_ages(&lru, |id| id, 4, 0, 1), vec![10]);
    assert!(coldest_ages(&[], |id| id, 4, 0, 8).is_empty());
    // 9 free blocks: two whole segments cost nothing; the third evicts 3 blocks (entry 10), the fourth 7 (to entry 30)
    let m = u64::MAX;
    assert_eq!(coldest_ages(&lru, |id| id * 100, 4, 9, 8), vec![m, m, 1000, 1000, 1000]);
    assert_eq!(coldest_ages(&[], |id| id, 4, 9, 8), vec![m, m]);
    assert_eq!(coldest_ages(&lru, |id| id, 4, 9, 1), vec![m]);
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
                    let r = c[0].report(1 + next() % 3, &[], &[]).unwrap();
                    for s in &c {
                        s.grow(segs + r.grant, r.txn).unwrap();
                    }
                } else if segs > 1 {
                    let to = segs - 1 - next() % segs.min(2).max(1).min(segs - 1);
                    for s in &c {
                        s.shrink(to.max(1), 0).unwrap();
                    }
                }
                let tag = seed.wrapping_mul(1000).wrapping_add(i * 4);
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

fn unclaimed(addr: &str) -> u64 {
    let st = stats(addr);
    let i = st.find("\"unclaimed_segs\":").expect("unclaimed_segs") + 17;
    st[i..].split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()
}

// every rank of c to `to` segments under txn
fn shrink_all(c: &[RemoteStore], to: u64, txn: u64) {
    for s in c {
        s.shrink(to, txn).unwrap();
    }
}

fn grow_all(c: &[RemoteStore], to: u64, txn: u64) {
    for s in c {
        s.grow(to, txn).unwrap();
    }
}

// M3 (c): a group that opens on a full node asks the others down toward their floors and waits for them, instead of
// being refused
#[test]
fn new_group_preempts_full_node_on_gopen() {
    let addr = node(16, 60_000);
    let a = cluster(&addr, "a", 2, 100_000, 1, 8);
    let r = a[0].report(9, &[0], &[]).unwrap();
    assert_eq!(r.grant, 7);
    grow_all(&a, 8, r.txn);
    write_all(&a, 3);
    assert_eq!(unclaimed(&addr), 0);
    let b_addr = addr.clone();
    let b = std::thread::spawn(move || rank(&b_addr, "b", 1, 0, 100_000, 2, 4));
    // a's head hears the give at its next report (a hot group: the margin does not apply to a floor)
    let mut gave = false;
    for _ in 0..250 {
        let r = a[0].report(0, &[0], &[]).unwrap();
        if r.give > 0 {
            assert_eq!(r.give, 1, "2 segments from 2 ranks");
            shrink_all(&a, 7, r.txn);
            gave = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(gave, "a was never asked");
    let b = b.join().unwrap().unwrap();
    assert_eq!(b.segs(), 2);
    check(&a, 3, 7 * a[0].per_seg);
}

// a group planned with --group-plan finds its floors kept for it, however full the node is
#[test]
fn a_planned_group_starts_on_a_full_node() {
    let plan = parse_plan("# group plan\np nr=1 floor=2\n").unwrap();
    assert_eq!(plan.get("p"), Some(&(1, 2)));
    assert!(parse_plan("p nr=1").is_err() && parse_plan("p nr=0 floor=1").is_err());
    assert!(parse_plan("p nr=1 floor=1\np nr=2 floor=1").is_err());
    let too_big = Share { seg_bytes: SEG, plan: parse_plan("p nr=3 floor=3").unwrap(), ..Share::default() };
    assert!(Node::new(
        &Config { bytes: 8 * SEG, lock: false, rdma: false, qp_timeout: 14, share: too_big },
        "127.0.0.1:0".parse().unwrap()
    )
    .is_err());
    let addr = node_with(8, Share { plan, ..Share::default() });
    let a = cluster(&addr, "a", 1, 100_000, 1, 8);
    let r = a[0].report(9, &[], &[]).unwrap();
    assert_eq!(r.grant, 5, "8 less a's floor less p's planned 2");
    grow_all(&a, 6, r.txn);
    let p = rank(&addr, "p", 1, 0, 100_000, 2, 2).unwrap();
    assert_eq!(p.segs(), 2);
}

// M3 (b): a group's empty segments report as free (u64::MAX), so the node takes them at once, not after the margin
#[test]
fn free_segments_given_without_margin() {
    let addr = node(8, 60_000);
    let a = cluster(&addr, "a", 1, 100_000, 1, 6);
    let r = a[0].report(5, &[], &[]).unwrap();
    grow_all(&a, 6, r.txn);
    let b = cluster(&addr, "b", 1, 100_000, 1, 8);
    let per = a[0].per_seg;
    // a holds 6 segments, 3 of them stored blocks used just now: a hot group, not asked
    let hot = coldest_ages(&[(1, 3 * per as usize)], |_| 0, per, 3 * per, 8);
    assert_eq!(hot, vec![u64::MAX, u64::MAX, u64::MAX, 0, 0, 0]);
    a[0].report(0, &[0], &[]).unwrap();
    let r = b[0].report(3, &[0], &[]).unwrap();
    assert_eq!(r.grant, 1, "the one segment nobody holds");
    grow_all(&b, 2, r.txn);
    assert_eq!(a[0].report(0, &[0], &[]).unwrap().give, 0, "inside the margin");
    // the same group reporting its free blocks: the node asks it for 2 at once
    a[0].report(0, &hot, &[]).unwrap();
    assert_eq!(b[0].report(2, &[0], &[]).unwrap().grant, 0);
    let r = a[0].report(0, &hot, &[]).unwrap();
    assert_eq!(r.give, 2);
    shrink_all(&a, 4, r.txn);
    let r = b[0].report(2, &[0], &[]).unwrap();
    assert_eq!(r.grant, 2);
}

// M3 (d): a giver of 4 ranks frees segments in fours; a taker of 2 ranks that needed 2 takes the other 2 with its
// next grant, so none are stranded
#[test]
fn nr4_vs_nr2_no_strand() {
    let addr = node(12, 1_000);
    let a = cluster(&addr, "a", 4, 100_000, 1, 2);
    let r = a[0].report(1, &[], &[]).unwrap();
    grow_all(&a, 2, r.txn);
    a[0].report(0, &[600_000], &[]).unwrap();
    let b = cluster(&addr, "b", 2, 100_000, 1, 4);
    let r = b[0].report(2, &[0], &[]).unwrap();
    assert_eq!(r.grant, 1, "2 free: one a rank, and a is asked for the rest");
    grow_all(&b, 2, r.txn);
    let r = a[0].report(0, &[600_000], &[]).unwrap();
    assert_eq!(r.give, 1, "1 a rank of 4 ranks: 4 freed, 2 needed");
    shrink_all(&a, 1, r.txn);
    assert_eq!(unclaimed(&addr), 4);
    let r = b[0].report(1, &[0], &[]).unwrap();
    assert_eq!(r.grant, 2, "the 1 it asked for and the 1 the give left over");
    grow_all(&b, 4, r.txn);
    assert_eq!(unclaimed(&addr), 0, "{}", stats(&addr));
}

// M3 (f): segments a rank gives back go free only once every rank of its group gave
#[test]
fn early_group_reuse() {
    let addr = node(8, 1_000);
    let a = cluster(&addr, "a", 2, 100_000, 1, 8);
    let r = a[0].report(2, &[], &[]).unwrap();
    grow_all(&a, 3, r.txn);
    let b = cluster(&addr, "b", 2, 100_000, 1, 8);
    a[0].report(0, &[600_000], &[]).unwrap();
    assert_eq!(b[0].report(1, &[0], &[]).unwrap().grant, 0);
    let r = a[0].report(0, &[600_000], &[]).unwrap();
    assert_eq!(r.give, 1);
    a[0].shrink(2, r.txn).unwrap();
    // rank 1 still holds its third segment and may write it: the one rank 0 gave is not free yet
    assert_eq!(unclaimed(&addr), 0);
    assert_eq!(b[0].report(1, &[0], &[]).unwrap().grant, 0);
    write_all(&a[1..], 9);
    a[1].shrink(2, r.txn).unwrap();
    assert_eq!(unclaimed(&addr), 2);
    let r = b[0].report(1, &[0], &[]).unwrap();
    assert_eq!(r.grant, 1);
    grow_all(&b, 2, r.txn);
}

// M3 (f): a grant one rank took stays held for the others past grant_ttl; one no rank took expires
#[test]
fn partial_grant_survives_ttl() {
    let addr = node_with(8, Share { grant_ttl: Duration::from_millis(200), ..Share::default() });
    let a = cluster(&addr, "a", 2, 100_000, 1, 8);
    let r = a[0].report(2, &[], &[]).unwrap();
    assert_eq!(r.grant, 2);
    a[0].grow(3, r.txn).unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(a[0].report(0, &[], &[]).unwrap().unclaimed, 2, "4 free, rank 1's part of 2 still held");
    a[1].grow(3, r.txn).unwrap();
    // nobody took this one: it lapses, and growing under it is stale
    let r = a[0].report(1, &[], &[]).unwrap();
    assert_eq!(r.grant, 1);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(a[0].report(0, &[], &[]).unwrap().unclaimed, 2);
    let e = a[0].grow(4, r.txn).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::StaleNetworkFileHandle, "{e}");
}

// M3 (f): a GROW or SHRINK under a transaction that is not the group's open one is refused, a repeat is a no-op
#[test]
fn stale_shrink_refused() {
    let addr = node(8, 1_000);
    let a = cluster(&addr, "a", 2, 100_000, 1, 8);
    let g = a[0].report(2, &[], &[]).unwrap();
    grow_all(&a, 3, g.txn);
    let b = cluster(&addr, "b", 2, 100_000, 1, 8);
    a[0].report(0, &[600_000], &[]).unwrap();
    b[0].report(1, &[0], &[]).unwrap();
    let r = a[0].report(0, &[600_000], &[]).unwrap();
    assert_eq!(r.give, 1);
    // under the grow's transaction, or past the give's target
    assert_eq!(a[0].shrink(2, g.txn).unwrap_err().kind(), ErrorKind::StaleNetworkFileHandle);
    assert_eq!(a[0].shrink(1, r.txn).unwrap_err().kind(), ErrorKind::InvalidData);
    shrink_all(&a, 2, r.txn);
    // the give is over: the same SHRINK again changes nothing, a lower one under it is stale
    shrink_all(&a, 2, r.txn);
    assert_eq!(a[0].segs(), 2);
    assert_eq!(a[0].shrink(1, r.txn).unwrap_err().kind(), ErrorKind::StaleNetworkFileHandle);
}
