//! The host ledger (src/ledger.rs, D-331 M7): a crash at every write of every ownership phase leaves a book that
//! balances and recovers; a stalled borrower is fenced off; many parties at once never credit a byte twice.

use g6kvman::ledger::{Book, Ledger, Phase, Side};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_kvman-ledger");

fn dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("kvman-ledger-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cli(file: &std::path::Path, args: &[&str], crash: Option<&str>) -> (bool, String) {
    let mut c = Command::new(BIN);
    c.arg(file).args(args);
    if let Some(p) = crash {
        c.env(gr8knobs::TEST_LEDGER_CRASH_AT.var(), p);
    }
    let o = c.output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).trim().to_string())
}

const GIB: u64 = 1 << 30;

fn fresh(file: &std::path::Path) -> Book {
    let b = Book::new(40 * GIB, 24 * GIB, 12 * GIB, 8 * GIB).unwrap();
    Ledger::at(file).init(&b).unwrap();
    Ledger::at(file).show().unwrap()
}

#[test]
fn a_crash_at_every_phase_leaves_a_book_that_balances_and_recovers() {
    let d = dir("crash");
    let file = d.join("ledger");
    // each step of both directions, from the state the step before it leaves
    for (from, to) in [("gpu", "node"), ("node", "gpu")] {
        for step in 0..4 {
            for point in ["write", "written"] {
                let b0 = fresh(&file);
                let n = 2 * GIB;
                let setup: Vec<Vec<String>> = vec![
                    vec!["begin".into(), from.into(), to.into(), n.to_string()],
                    vec!["released".into(), "1".into(), n.to_string()],
                    vec!["commit".into(), "1".into(), (n - GIB / 2).to_string()],
                ];
                // steps 0..2: begin, released, commit; step 3: abort after released
                let mut run: Vec<Vec<String>> = setup[..step.min(3)].to_vec();
                if step == 3 {
                    run = setup[..2].to_vec();
                    run.push(vec!["abort".into(), "1".into()]);
                }
                let (last, before) =
                    run.split_last().map(|(l, b)| (Some(l.clone()), b.to_vec())).unwrap_or((None, vec![]));
                for a in &before {
                    let a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
                    assert!(cli(&file, &a, None).0, "{a:?}");
                }
                let pre = Ledger::at(&file).show().unwrap();
                let Some(last) = last else { continue };
                let a: Vec<&str> = last.iter().map(|s| s.as_str()).collect();
                let (ok, _) = cli(&file, &a, Some(point));
                assert!(!ok, "{a:?} at {point} crashed");
                let post = Ledger::at(&file).show().unwrap_or_else(|e| panic!("{a:?} at {point}: {e}"));
                assert!(post.balanced() && post.total == b0.total, "{a:?} at {point}: {post:?}");
                if point == "write" {
                    assert_eq!(post, pre, "{a:?}: a crash before the rename keeps the old book");
                } else {
                    assert_ne!(post, pre, "{a:?}: a crash after the rename keeps the new book");
                }
                // the survivor recovers: no transfer pending, every byte counted once
                let (ok, out) = cli(&file, &["recover", "--stale-s", "0"], None);
                assert!(ok, "{out}");
                let r = Ledger::at(&file).show().unwrap();
                assert!(r.pending.is_none() && r.balanced(), "{r:?}");
                // a crashed transfer moves nothing to its destination it had not committed
                let dest = |b: &Book| if to == "node" { b.node } else { b.gpu };
                let src = |b: &Book| if from == "gpu" { b.gpu } else { b.node };
                assert!(dest(&r) <= dest(&b0) + n - GIB / 2 && src(&r) >= src(&b0) - n, "{r:?}");
                let _ = std::fs::remove_file(&file);
            }
        }
    }
    // a torn file is refused, never read as zeros
    std::fs::write(&file, "gen 3\ntotal 100\ngpu 60\n").unwrap();
    assert!(Ledger::at(&file).show().unwrap_err().to_string().contains("torn"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_stalled_borrower_is_fenced_off() {
    let d = dir("stall");
    let file = d.join("ledger");
    let l = Ledger::at(&file);
    let b0 = fresh(&file);
    // the node gives 4 GiB back to the GPU; the node releases them, the GPU side stalls before it commits
    let g = l.apply(|b| b.begin(Side::Node, Side::Gpu, 4 * GIB, 1000)).unwrap();
    l.apply(|b| b.released(g, 4 * GIB)).unwrap();
    // not stale yet: recover leaves it, and nobody else may start a transfer
    assert_eq!(l.apply(|b| Ok(b.recover(30, 1010))).unwrap(), None);
    assert!(l.apply(|b| b.begin(Side::Gpu, Side::Node, GIB, 1010)).unwrap_err().to_string().contains("busy"));
    // stale: aborted, its bytes go to free, not to the GPU and not back to the node
    assert_eq!(l.apply(|b| Ok(b.recover(30, 1031))).unwrap(), Some(g));
    let b = l.show().unwrap();
    assert_eq!((b.gpu, b.node, b.free), (b0.gpu, b0.node - 4 * GIB, b0.free + 4 * GIB));
    // the borrower wakes up: its commit is refused, so it gives back what it backed
    assert!(l.apply(|b| b.commit(g, 4 * GIB)).is_err());
    // a new transfer has a new generation; the old one stays refused
    let g2 = l.apply(|b| b.begin(Side::Free, Side::Gpu, 4 * GIB, 1040)).unwrap();
    assert!(g2 > g);
    assert!(l.apply(|b| b.released(g, GIB)).is_err());
    l.apply(|b| b.released(g2, 4 * GIB)).unwrap();
    l.apply(|b| b.commit(g2, 4 * GIB)).unwrap();
    let b = l.show().unwrap();
    assert_eq!((b.gpu, b.free), (b0.gpu + 4 * GIB, b0.free));
    assert!(b.balanced());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn many_parties_never_credit_a_byte_twice() {
    let d = dir("race");
    let file = d.join("ledger");
    let b0 = fresh(&file);
    // credits each thread got, by side; the book must end at b0 plus exactly these
    let (to_gpu, to_node, to_free) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let (from_gpu, from_node, from_free) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    std::thread::scope(|s| {
        for t in 0..8u64 {
            let (file, to_gpu, to_node, to_free) = (&file, &to_gpu, &to_node, &to_free);
            let (from_gpu, from_node, from_free) = (&from_gpu, &from_node, &from_free);
            s.spawn(move || {
                let l = Ledger::at(file);
                let mut x = t * 2654435761 + 1;
                let mut rnd = |m: u64| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x % m
                };
                for _ in 0..150 {
                    let sides = [Side::Gpu, Side::Node, Side::Free];
                    let (from, to) = (sides[rnd(3) as usize], sides[rnd(3) as usize]);
                    let n = (1 + rnd(4)) * GIB;
                    let Ok(g) = l.apply(|b| b.begin(from, to, n, 0)) else { continue };
                    // the recoverer below may abort it at any point: then this party's next call is refused
                    let m = n - rnd(2) * GIB / 2;
                    if rnd(5) == 0 {
                        let _ = l.apply(|b| b.abort(g));
                        continue;
                    }
                    if l.apply(|b| b.released(g, m)).is_err() {
                        continue; // aborted before it released: nothing moved
                    }
                    [from_gpu, from_node, from_free][from as usize].fetch_add(m, Ordering::Relaxed);
                    let k = m - rnd(2) * GIB / 4;
                    if rnd(6) == 0 {
                        let _ = l.apply(|b| b.abort(g));
                        to_free.fetch_add(m, Ordering::Relaxed);
                        // the late commit of an aborted transfer is refused
                        assert!(l.apply(|b| b.commit(g, k)).is_err());
                        continue;
                    }
                    if l.apply(|b| b.commit(g, k)).is_err() {
                        // aborted after it released: the bytes went to free, the destination gives back what it backed
                        to_free.fetch_add(m, Ordering::Relaxed);
                        continue;
                    }
                    [to_gpu, to_node, to_free][to as usize].fetch_add(k, Ordering::Relaxed);
                    to_free.fetch_add(m - k, Ordering::Relaxed);
                    let b = l.show().unwrap();
                    assert!(b.balanced(), "{b:?}");
                }
            });
        }
        // a recoverer that finds every transfer stale, now and then
        let f = &file;
        s.spawn(move || {
            let l = Ledger::at(f);
            let mut recovered = 0;
            for i in 0..400u64 {
                if i % 3 == 0 && l.apply(|b| Ok(b.recover(0, u64::MAX))).unwrap().is_some() {
                    recovered += 1;
                }
                std::thread::yield_now();
            }
            assert!(recovered > 0, "the recoverer aborted nothing");
        });
        // a reader the whole time: every book it sees balances
        let l = Ledger::at(&file);
        for _ in 0..300 {
            let b = l.show().unwrap();
            assert!(b.balanced() && b.total == b0.total, "{b:?}");
            assert!(b.pending.is_none_or(|p| p.phase == Phase::Begun || p.bytes <= 4 * GIB));
        }
    });
    let b = Ledger::at(&file).show().unwrap();
    let c = |a: &AtomicU64| a.load(Ordering::Relaxed);
    assert_eq!(b.gpu, b0.gpu + c(&to_gpu) - c(&from_gpu));
    assert_eq!(b.node, b0.node + c(&to_node) - c(&from_node));
    assert_eq!(b.free, b0.free + c(&to_free) - c(&from_free));
    assert!(b.balanced() && b.pending.is_none(), "{b:?}");
    assert!(b.gen > 0);
    let _ = std::fs::remove_dir_all(&d);
}
