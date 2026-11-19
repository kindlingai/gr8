//! GR8_KV_REWIDTH (src/rewidth.rs): a persisted store written at one width, rewritten for every rank of another over
//! loopback `kvman-peer` servers (src/peer.rs), against canon::table's ownership rule.

use g6kvman::canon::{self, Binding, Layout, View};
use g6kvman::peer::{self, Peer};
use g6kvman::pool::STOCK_NAMESPACE;
use g6kvman::rewidth::{rewidth, Spec};
use g6kvman::stash::{self, Entry, Kind, Tail};
use g6kvman::Finished;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const SB: u64 = 4096;
const BELOW: u64 = 128; // R: 2 pages
const RP: u64 = BELOW / 64;
const PER: u32 = 2; // replicated parts a block
const TOKEN: &str = "00ff12ab";

fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in [a, b, c] {
        h = (h ^ x).wrapping_mul(0x100000001b3);
    }
    h
}

fn block(v: u64) -> Vec<u8> {
    v.to_le_bytes().iter().copied().cycle().take(SB as usize).collect()
}

/// a width-neutral block (Full or Rep), the same on every rank
fn neutral(slot: u64) -> Vec<u8> {
    block(mix(1, slot, 0))
}

/// logical page lp's shard bytes of entry id
fn page(id: u64, lp: u64) -> Vec<u8> {
    block(mix(2, id, lp))
}

/// the store's compat line at width nr: the set id names the width, as the launcher's does (modelset.py describe_release)
fn compat(nr: u32, slots: u64, below: u64) -> String {
    format!(
        "set s{nr} nr {nr} below {below} place 900 grid 64 block {SB} bindings 4/2 slots {slots} format 3 engine abc kv int4"
    )
}

/// rank `rank` of `nr`'s own pages of a T-page tail, in block order (canon::table's rule)
fn owned(nr: u32, rank: u32, t: u32) -> Vec<u64> {
    let view = View { list: (0..RP as u32 + t).collect(), place: u32::MAX };
    let layout = Layout { nr, below_rows: BELOW as u32 };
    let tab = canon::table(Binding { index: true, cp: true }, &layout, &view, rank, (RP as usize) + t as usize).unwrap();
    let mut v: Vec<(i64, u64)> =
        tab.iter().enumerate().skip(RP as usize).filter(|(_, &p)| p >= 0).map(|(lp, &p)| (p, lp as u64)).collect();
    v.sort();
    v.into_iter().map(|(_, lp)| lp).collect()
}

fn entry(id: u64, kind: Kind, slots: Vec<u64>, base: Option<(u64, usize)>, tail: Option<Tail>) -> Entry {
    Entry {
        id,
        kind,
        key: id * 77,
        st: Finished { history: (0..10).collect(), valid: 9, written_end: 9, pf_end: 9 },
        slots,
        base,
        hits: 1,
        tail,
        state: 0,
        namespace: STOCK_NAMESPACE,
    }
}

/// The old store's entries at `nr` (the same index on every rank), with tails of `tails` pages, slots handed out from
/// `start` with a free gap after each entry.
fn entries(nr: u32, tails: &[u32], start: u64) -> Vec<Entry> {
    let mut next = start;
    let mut take = |n: u64| {
        let v: Vec<u64> = (next..next + n).collect();
        next += n + 1;
        v
    };
    let mut out = vec![entry(1, Kind::Prefix, take(RP), None, None), entry(2, Kind::Shared, take(1), None, None)];
    for (i, &t) in tails.iter().enumerate() {
        let tail = Tail { logical: t, own: t.div_ceil(nr), rep: t.div_ceil(PER) };
        let id = 10 + i as u64;
        let e = if i % 2 == 0 {
            entry(id, Kind::Evicted, take((tail.own + tail.rep) as u64), Some((1, RP as usize)), Some(tail))
        } else {
            entry(id, Kind::Evicted, take(RP + (tail.own + tail.rep) as u64), None, Some(tail))
        };
        out.push(e);
    }
    out
}

fn own_slots(e: &Entry) -> Vec<u64> {
    let t = e.tail.unwrap();
    let at = e.slots.len() - (t.own + t.rep) as usize;
    e.slots[at..at + t.own as usize].to_vec()
}

/// Writes old rank q's store in `dir`: its blocks, and the index as the launcher leaves it (index.prev).
fn write_old(dir: &Path, nr: u32, q: u32, slots: u64, es: &[Entry]) {
    std::fs::create_dir_all(dir).unwrap();
    let f = std::fs::File::create(dir.join("blocks")).unwrap();
    f.set_len(slots * SB).unwrap();
    for e in es {
        let own: BTreeSet<u64> = e.tail.map(|_| own_slots(e)).unwrap_or_default().into_iter().collect();
        for &s in &e.slots {
            if !own.contains(&s) {
                f.write_all_at(&neutral(s), s * SB).unwrap();
            }
        }
        if let Some(t) = e.tail {
            for (j, &s) in own_slots(e).iter().enumerate() {
                let lp = owned(nr, q, t.logical).get(j).copied();
                let b = lp.map_or_else(|| block(0xdead), |lp| page(e.id, lp));
                f.write_all_at(&b, s * SB).unwrap();
            }
        }
    }
    let counts: BTreeMap<u64, (u32, u64)> = [(5, (3, 9)), (6, (1, 2))].into_iter().collect();
    stash::write_index(dir, &compat(nr, slots, BELOW), es, &counts).unwrap();
    std::fs::rename(dir.join("index"), dir.join("index.prev")).unwrap();
}

struct Servers {
    addrs: Vec<String>,
    token_file: PathBuf,
}

fn serve(dirs: &[PathBuf], tmp: &Path) -> Servers {
    let token_file = tmp.join("token");
    std::fs::write(&token_file, format!("{TOKEN}\n")).unwrap();
    let addrs = dirs
        .iter()
        .map(|d| {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let a = l.local_addr().unwrap().to_string();
            let d = d.clone();
            std::thread::spawn(move || peer::serve(d, l, TOKEN.to_string()));
            a
        })
        .collect();
    Servers { addrs, token_file }
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("g6rewidth-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One scale: old ranks' stores in `old` (served), new rank r runs in `new[r]` with its old rank `from[r]` (None: a
/// joiner). Every new rank runs at once, as the boxes do. Returns each new rank's result.
fn scale(old: &[PathBuf], from: &[Option<u32>], new: &[PathBuf], nr: u32, slots: u64, tmp: &Path) -> Vec<Result<u64, String>> {
    let sv = serve(old, tmp);
    let hs: Vec<_> = (0..nr)
        .map(|r| {
            let spec = Spec {
                old_nr: old.len() as u32,
                old_rank: from[r as usize],
                token_file: sv.token_file.display().to_string(),
                old_set: format!("s{}", old.len()),
                peers: sv.addrs.clone(),
            };
            let dir = new[r as usize].clone();
            std::thread::spawn(move || {
                std::fs::create_dir_all(&dir).unwrap();
                rewidth(&dir, &compat(nr, slots, BELOW), nr, r, slots, SB, &spec).map(|r| r.map_err(|e| format!("skipped: {e}")))
            })
        })
        .collect();
    hs.into_iter().map(|h| h.join().unwrap().and_then(|r| r)).collect()
}

fn read_block(dir: &Path, s: u64) -> Vec<u8> {
    let mut b = vec![0u8; SB as usize];
    std::fs::File::open(dir.join("blocks")).unwrap().read_exact_at(&mut b, s * SB).unwrap();
    b
}

/// Every rank's index is the same bytes, and every block a rank holds is the one its old owner held.
fn check(new: &[PathBuf], nr: u32, joiners: &[bool]) -> Vec<Entry> {
    let raw: Vec<Vec<u8>> = new.iter().map(|d| std::fs::read(d.join("index")).unwrap()).collect();
    assert!(raw.iter().all(|r| *r == raw[0]), "every new rank writes the same index");
    let (c, es, counts) = stash::read_index(&raw[0]).unwrap();
    assert!(c.contains(&format!(" nr {nr} ")), "{c}");
    assert_eq!(counts.len(), 2);
    for (r, d) in new.iter().enumerate() {
        for e in &es {
            if let Some(t) = e.tail {
                assert_eq!(t.own, t.logical.div_ceil(nr));
                for (j, lp) in owned(nr, r as u32, t.logical).into_iter().enumerate() {
                    assert_eq!(read_block(d, own_slots(e)[j]), page(e.id, lp), "rank {r} entry {} page {lp}", e.id);
                }
            }
            if joiners[r] {
                let own: BTreeSet<u64> = e.tail.map(|_| own_slots(e)).unwrap_or_default().into_iter().collect();
                for &s in e.slots.iter().filter(|s| !own.contains(s)) {
                    assert_eq!(read_block(d, s), neutral(s), "joiner {r} slot {s}");
                }
            }
        }
    }
    es
}

fn prev(dir: &Path) {
    std::fs::copy(dir.join("index"), dir.join("index.prev")).unwrap();
}

#[test]
fn widths_4_to_3_3_to_4_and_4_to_6() {
    let tails = [5, 7, 1, 12];
    for (old_nr, nr) in [(4u32, 3u32), (3, 4), (4, 6), (3, 6), (6, 4)] {
        let tmp = tmpdir(&format!("w{old_nr}{nr}"));
        let es = entries(old_nr, &tails, 3);
        let old: Vec<PathBuf> = (0..old_nr).map(|q| tmp.join(format!("old{q}"))).collect();
        for (q, d) in old.iter().enumerate() {
            write_old(d, old_nr, q as u32, 256, &es);
        }
        // the first min(old, new) boxes stay (in place, same rank); the rest join with fresh folders
        let from: Vec<Option<u32>> = (0..nr).map(|r| (r < old_nr).then_some(r)).collect();
        let new: Vec<PathBuf> =
            (0..nr).map(|r| if r < old_nr { old[r as usize].clone() } else { tmp.join(format!("new{r}")) }).collect();
        let got = scale(&old, &from, &new, nr, 256, &tmp);
        assert!(got.iter().all(|g| *g == Ok(es.len() as u64)), "{old_nr}->{nr}: {got:?}");
        let joiners: Vec<bool> = from.iter().map(Option::is_none).collect();
        let after = check(&new, nr, &joiners);
        assert_eq!(after.iter().map(|e| e.id).collect::<Vec<_>>(), es.iter().map(|e| e.id).collect::<Vec<_>>());
        // the old index stays valid for a rollback: no stayer's old blocks were written over
        for r in 0..old_nr.min(nr) {
            for e in es.iter().filter(|e| e.tail.is_some()) {
                for (j, lp) in owned(old_nr, r, e.tail.unwrap().logical).into_iter().enumerate() {
                    assert_eq!(read_block(&old[r as usize], own_slots(e)[j]), page(e.id, lp));
                }
            }
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}

#[test]
fn a_box_whose_rank_changes_and_a_round_trip_4_to_3_to_4() {
    let tmp = tmpdir("trip");
    let es = entries(4, &[9, 4, 13], 0);
    let old: Vec<PathBuf> = (0..4).map(|q| tmp.join(format!("b{q}"))).collect();
    for (q, d) in old.iter().enumerate() {
        write_old(d, 4, q as u32, 128, &es);
    }
    // box 1 leaves: boxes 0, 2, 3 become ranks 0, 1, 2 (their folders renamed with them, rewritten in place)
    let new = vec![old[0].clone(), old[2].clone(), old[3].clone()];
    let got = scale(&old, &[Some(0), Some(2), Some(3)], &new, 3, 128, &tmp);
    assert!(got.iter().all(|g| *g == Ok(es.len() as u64)), "{got:?}");
    check(&new, 3, &[false; 3]);
    // and back to 4: the three stay, box 1 joins with its stale folder
    for d in &new {
        prev(d);
    }
    let back = vec![new[0].clone(), new[1].clone(), new[2].clone(), old[1].clone()];
    let got = scale(&new, &[Some(0), Some(1), Some(2), None], &back, 4, 128, &tmp);
    assert!(got.iter().all(|g| *g == Ok(es.len() as u64)), "{got:?}");
    check(&back, 4, &[false, false, false, true]);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_full_store_drops_the_least_recently_used_tails_on_every_rank() {
    let tmp = tmpdir("full");
    let tails = [6, 6, 6, 6, 6];
    let es = entries(4, &tails, 0);
    let used: BTreeSet<u64> = es.iter().flat_map(|e| e.slots.iter().copied()).collect();
    let top = *used.iter().max().unwrap() + 1;
    let old: Vec<PathBuf> = (0..4).map(|q| tmp.join(format!("o{q}"))).collect();
    for (q, d) in old.iter().enumerate() {
        write_old(d, 4, q as u32, top + 2, &es);
    }
    // at TP=3 each tail needs 2 own blocks; the gaps and 2 slots past the old entries leave room for only some
    let free = (0..top + 2).filter(|s| !used.contains(s)).count();
    let new = vec![old[0].clone(), old[1].clone(), old[2].clone()];
    let got = scale(&old, &[Some(0), Some(1), Some(2)], &new, 3, top + 2, &tmp);
    let after = check(&new, 3, &[false; 3]);
    let kept_tails = after.iter().filter(|e| e.tail.is_some()).count();
    assert_eq!(kept_tails, free / 2);
    assert!(kept_tails < tails.len());
    // the least recently used (first in order) went; the shared prefix and the plain entry stay
    let ids: Vec<u64> = after.iter().map(|e| e.id).collect();
    let want: Vec<u64> = [1, 2].into_iter().chain((10 + (tails.len() - kept_tails) as u64)..(10 + tails.len() as u64)).collect();
    assert_eq!(ids, want);
    assert!(got.iter().all(|g| *g == Ok(want.len() as u64)));
    // the new own blocks took only slots no old entry used
    for e in after.iter().filter(|e| e.tail.is_some()) {
        assert!(own_slots(e).iter().all(|s| !used.contains(s)));
    }
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn refusals() {
    let tmp = tmpdir("refuse");
    let es = entries(4, &[5], 0);
    let old: Vec<PathBuf> = (0..4).map(|q| tmp.join(format!("o{q}"))).collect();
    for (q, d) in old.iter().enumerate() {
        write_old(d, 4, q as u32, 64, &es);
    }
    let sv = serve(&old[..1], &tmp);
    // a wrong token: the server closes the connection
    let mut p = Peer::open(&sv.addrs[0], "0bad").unwrap();
    assert!(p.size("index.prev").is_err());
    // a name with `..`, or outside the folder: refused by the server too
    let mut p = Peer::open(&sv.addrs[0], TOKEN).unwrap();
    assert!(p.size("index.prev").unwrap() > 0);
    let mut s = std::net::TcpStream::connect(&sv.addrs[0]).unwrap();
    s.write_all(format!("G6PEER1 {TOKEN}\n").as_bytes()).unwrap();
    for name in ["../o1/blocks", "/etc/passwd", "a/b/c"] {
        let mut q = vec![2u8];
        q.extend((name.len() as u16).to_le_bytes());
        q.extend(name.as_bytes());
        q.extend([0u8; 16]);
        s.write_all(&q).unwrap();
        let mut head = [0u8; 9];
        s.read_exact(&mut head).unwrap();
        assert_eq!(head[0], 1, "{name}");
    }
    // a read past the end of a file is an error, never short
    assert!(p.read("index.prev", 0, &mut vec![0u8; 1 << 20]).is_err());
    // another R: skipped, the store starts empty
    let spec = Spec {
        old_nr: 4,
        old_rank: Some(0),
        token_file: sv.token_file.display().to_string(),
        old_set: "s4".into(),
        peers: vec![sv.addrs[0].clone(); 4],
    };
    let r = rewidth(&old[0], &compat(3, 64, BELOW * 2), 3, 0, 64, SB, &spec).unwrap();
    assert_eq!(r, Err("below differs".to_string()));
    assert!(!old[0].join("index").exists(), "nothing written");
    // another model set than the old width's: skipped
    let other = Spec { old_set: "s9".into(), ..spec.clone() };
    let r = rewidth(&old[0], &compat(3, 64, BELOW), 3, 0, 64, SB, &other).unwrap();
    assert_eq!(r, Err("set differs from the old width's".to_string()));
    // a peer that is gone: an error, and the index is untouched
    let spec = Spec { old_rank: None, peers: vec!["127.0.0.1:1".into(); 4], ..spec };
    let fresh = tmp.join("fresh");
    std::fs::create_dir_all(&fresh).unwrap();
    assert!(rewidth(&fresh, &compat(3, 64, BELOW), 3, 0, 64, SB, &spec).is_err());
    assert!(!fresh.join("index").exists());
    std::fs::remove_dir_all(&tmp).unwrap();
}
