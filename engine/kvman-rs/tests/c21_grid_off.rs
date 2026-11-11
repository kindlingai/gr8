//! c21 regression: with reuse alignment disabled, a hit can stop inside a page. That page may be
//! rewritten, but a retained run must not be spilled when it has no rows past the hit. A stored
//! run still extends its old entry, reusing its full pages instead of accumulating whole copies.
use g6kvman::stash::{Io, Keep, Policy};
use g6kvman::{Finished, Pool, Rules, PAGE};

fn fin(t: &[i32]) -> Finished {
    let v = t.len() as i32 - 1;
    Finished { history: t.to_vec(), valid: v, written_end: v, pf_end: v }
}
fn toks(seed: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| seed * 100_000 + i).collect()
}
fn pool(grid: i32) -> Pool {
    let mut p = Pool::new(64, 64, 64 * PAGE as u32).unwrap();
    p.keep(Rules { strict: false, noend: true, grid });
    p.stores(2);
    p.add_tier(512, Keep::All, None).unwrap();
    p.set_policy(0, Policy { min_rows: 0, min_hits: 0, block_bytes: 1 }).unwrap();
    p
}
fn drain(p: &mut Pool) -> usize {
    let io = p.io();
    p.io_done().unwrap();
    io.iter().filter(|x| matches!(x, Io::Spill { .. })).count()
}
fn turn(p: &mut Pool, prompt: &[i32], gen: usize, seed: i32) -> (Vec<i32>, usize, String) {
    p.count(prompt, None);
    let rows = (prompt.len() + gen + 1) as u64;
    let h = p.match_prefix(prompt, rows, rows, prompt.len() as i32 - 1);
    let run = if h.skip > 0 {
        p.take_hit(h.skip, h.age, h.stored.then_some(h.tier), rows, rows).unwrap()
    } else {
        p.reserve(rows, rows).unwrap()
    };
    let mut w = drain(p);
    let mut t = prompt.to_vec();
    t.extend(toks(seed, gen));
    p.retain(&run, fin(&t)).unwrap();
    w += drain(p);
    (t, w, format!("skip {} stored {}", h.skip, h.stored))
}
fn evict_all(p: &mut Pool) -> usize {
    let n = p.pool_pages() as u64 * PAGE;
    let all = p.reserve(n, n).unwrap();
    let w = drain(p);
    p.release(&all).unwrap();
    w + drain(p)
}

#[test]
fn a_grid_off_partial_page_hit_extends_without_whole_run_rewrites() {
    for grid in [0, 128] {
        // P1: one chat whose runs stay in the pool (pool:own hits every turn)
        let mut p = pool(grid);
        let (mut t, _, _) = turn(&mut p, &toks(1, 300), 20, 900);
        let mut w = 0;
        for c in 2..7 {
            t.extend(toks(c, 60));
            let (t2, wr, how) = turn(&mut p, &t, 20, 900 + c);
            assert!(how.contains("stored false"), "grid {grid} turn {c}: {how}");
            w += wr;
            t = t2;
        }
        assert_eq!(w, 0, "grid {grid}: pool hits of the same chat spill no blocks");
        // P2: the run is evicted between turns (stored hits)
        let mut p = pool(grid);
        let (mut t, _, _) = turn(&mut p, &toks(1, 300), 20, 900);
        let mut w = evict_all(&mut p);
        for c in 2..7 {
            t.extend(toks(c, 60));
            let (t2, wr, how) = turn(&mut p, &t, 20, 900 + c);
            let ev = evict_all(&mut p);
            let st = p.tier_stats(0).unwrap();
            assert!(how.contains("stored true"), "grid {grid} turn {c}: {how}");
            assert_eq!(st.entries, 1, "grid {grid} turn {c}: extend replaces old entry");
            assert!(wr + ev <= 3, "grid {grid} turn {c}: wrote {} blocks", wr + ev);
            w += wr + ev;
            t = t2;
        }
        assert_eq!(w, if grid == 0 { 17 } else { 10 }, "grid {grid}: only new pages and the changed partial page");
    }
}
