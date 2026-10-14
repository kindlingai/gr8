//! Session state independent of the layout (src/canon.rs, #152). Each rank's caches are filled by an independent
//! writer: every token's rows go to the page its rank's table names, the table written here as engine/runtime/s2.cuh
//! s2_solo_cp_pages writes it (prefix list[lp], owned tail pages through s2_list_place), not from canon::table. Runs
//! come from a real Pool (pool.rs: below R, and past R on the page classes) or are the tables as built. Then:
//! - the tables match that derivation, with and without CP;
//! - same layout: export then import into empty caches is byte-identical on every page the run holds;
//! - another layout: export at A, import at B gives B's caches as B's own writer fills them from the same tokens, and
//!   export at B gives A's canonical form back; chains of layouts (6 -> 4 -> 3 -> no CP -> 6) lose nothing;
//! - a bulk copy then the tail written since (a live migration's two parts) equals one whole copy;
//! - a replicated page two ranks hold differently is caught; the byte form round-trips and refuses a cut blob;
//! - head-split state gathers and splits back for every width.

use g6kvman::canon::{self, Binding, Cache, CacheMut, Canon, Layout, View, IDX_ROW, NONE, PAGE, REMOTE};
use g6kvman::Pool;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

// a token's rows: what every rank computes for it (the MLA row is the same on every head and rank)
struct Tokens {
    mla: Vec<Vec<u8>>, // [binding][pos * row ..]
    idx: Vec<Option<Vec<u8>>>,
}

fn tokens(rng: &mut Rng, bs: &[Binding], row: usize, n: usize) -> Tokens {
    Tokens {
        mla: bs.iter().map(|_| rng.bytes(n * row)).collect(),
        idx: bs.iter().map(|b| b.index.then(|| rng.bytes(n * IDX_ROW))).collect(),
    }
}

// the oracle, as s2_solo_cp_pages writes a CP binding's table: prefix pages lp < R/64 at list[lp], an owned tail page
// (lp % nr == rank) at s2_list_place(list, rp + j, place), every other page REMOTE; any other binding: list as built
fn oracle(b: Binding, l: &Layout, list: &[u32], place: u32, rank: u32, logical: usize) -> Vec<i64> {
    let mut t = vec![NONE; logical];
    if !b.cp || l.nr == 0 {
        for i in 0..logical.min(list.len()) {
            t[i] = list[i] as i64;
        }
        return t;
    }
    let rp = l.below_rows as usize / PAGE;
    // s2_list_place: the at-th page of the list below place, -1 when the list has none
    let list_place = |at: usize| -> i64 {
        let mut seen = 0;
        for &p in list {
            if p < place {
                if seen == at {
                    return p as i64;
                }
                seen += 1;
            }
        }
        -1
    };
    let mut j = 0;
    for (lp, e) in t.iter_mut().enumerate() {
        if lp < rp {
            *e = if lp < list.len() { list[lp] as i64 } else { NONE };
        } else if lp % l.nr as usize == rank as usize {
            *e = list_place(rp + j);
            j += 1;
        } else {
            *e = REMOTE;
        }
    }
    t
}

// one rank's caches: `pages` physical pages per binding, filled by the writer with tokens 0..n (the rest stays as
// `fill`), as the engine's append path would leave them
struct Rank {
    mla: Vec<Vec<u8>>,
    idx: Vec<Option<Vec<u8>>>,
}

#[allow(clippy::too_many_arguments)]
fn write(
    t: &Tokens,
    bs: &[Binding],
    row: usize,
    l: &Layout,
    v: &View,
    rank: u32,
    pages: usize,
    n: usize,
    fill: u8,
) -> Rank {
    let mut r = Rank {
        mla: bs.iter().map(|_| vec![fill; pages * PAGE * row]).collect(),
        idx: bs.iter().map(|b| b.index.then(|| vec![fill; pages * PAGE * IDX_ROW])).collect(),
    };
    for (bi, &b) in bs.iter().enumerate() {
        let tab = oracle(b, l, &v.list, v.place, rank, n.div_ceil(PAGE));
        for pos in 0..n {
            let p = tab[pos / PAGE];
            if p < 0 {
                assert_eq!(p, REMOTE, "the writer found no page for a token");
                continue;
            }
            let (p, o) = (p as usize, pos % PAGE);
            let at = (p * PAGE + o) * row;
            r.mla[bi][at..at + row].copy_from_slice(&t.mla[bi][pos * row..(pos + 1) * row]);
            if let (Some(ix), Some(ti)) = (r.idx[bi].as_mut(), t.idx[bi].as_ref()) {
                // the page: 64 x 128 codes, then 64 FP32 scales
                let base = p * PAGE * IDX_ROW;
                ix[base + o * 128..base + (o + 1) * 128].copy_from_slice(&ti[pos * IDX_ROW..pos * IDX_ROW + 128]);
                let s = base + PAGE * 128 + o * 4;
                ix[s..s + 4].copy_from_slice(&ti[pos * IDX_ROW + 128..(pos + 1) * IDX_ROW]);
            }
        }
    }
    r
}

fn caches(r: &Rank) -> Vec<Cache<'_>> {
    r.mla.iter().zip(&r.idx).map(|(m, i)| Cache { mla: m, idx: i.as_deref() }).collect()
}

fn caches_mut(r: &mut Rank) -> Vec<CacheMut<'_>> {
    r.mla.iter_mut().zip(r.idx.iter_mut()).map(|(m, i)| CacheMut { mla: m, idx: i.as_deref_mut() }).collect()
}

// a random run of n rows at a layout, from a real Pool where the layout has one: a run below R (every page in the
// prefix), or a run past R on the page classes (nr > 0: pool.rs grants the prefix and this rank's shard below place,
// the drafter's copy above). Else the tables as built: pages 0, 1, .. (place = the whole list).
struct Setup {
    views: Vec<View>,
    pages: usize,
    pooled: bool, // the list came from a Pool, not the tables as built
}

fn setup(rng: &mut Rng, layout: Layout, n: usize) -> Setup {
    let logical = n.div_ceil(PAGE);
    let ranks = layout.nr.max(1) as usize;
    let as_built = |pages: usize| Setup {
        views: (0..ranks).map(|_| View { list: (0..logical as u32).collect(), place: u32::MAX }).collect(),
        pages,
        pooled: false,
    };
    let rp = layout.below_rows as usize / PAGE;
    if rp == 0 || rng.range(0, 3) == 0 {
        return as_built(logical + rng.range(0, 4) as usize);
    }
    let nr = layout.nr.max(1) as usize;
    let tail = logical.saturating_sub(rp);
    let own = tail.div_ceil(nr);
    // the pool: room for the run plus some retained garbage, so its pages are not 0, 1, ..
    let place = (rp.max(logical.min(rp + own)) + rng.range(0, 6) as usize) as u32;
    let pages = place as usize + (tail - own.min(tail)) + rng.range(0, 6) as usize;
    let mut pool = Pool::new(pages as u32, place, layout.below_rows).unwrap();
    if layout.nr > 0 {
        pool.tail(layout.nr, 0);
    }
    // a few short runs first, some released, so the run's pages are scattered
    let mut held = Vec::new();
    for _ in 0..rng.range(0, 4) {
        let rows = PAGE as u64 * rng.range(1, 2);
        if pool.fits(rows, rows) {
            held.push(pool.reserve(rows, rows).unwrap());
        }
    }
    for r in held.iter().filter(|_| rng.range(0, 1) == 0) {
        pool.release(r).unwrap();
    }
    let rows = n as u64;
    if !pool.fits(rows, rows) {
        return as_built(logical);
    }
    let run = pool.reserve(rows, rows).unwrap();
    let list = pool.list(&run).unwrap().to_vec();
    assert_eq!(list.len(), logical);
    // s2_solo_cp_pages bounds s2_list_place with the main model's pool pages: the pool's place
    Setup { views: (0..ranks).map(|_| View { list: list.clone(), place }).collect(), pages, pooled: true }
}

const ROWS: [usize; 2] = [656, 416];

fn bindings(rng: &mut Rng) -> Vec<Binding> {
    // a full-index CP layer, shared-index CP layers, then the unsharded MTP layer and a draft layer
    let mut v = vec![Binding { index: true, cp: true }];
    for _ in 0..rng.range(0, 2) {
        v.push(Binding { index: false, cp: true });
    }
    v.push(Binding { index: rng.range(0, 1) == 1, cp: false });
    v.push(Binding { index: false, cp: false });
    v
}

fn layouts(rng: &mut Rng, n: usize) -> Vec<Layout> {
    let mut v = vec![Layout { nr: 0, below_rows: 0 }];
    for nr in [2u32, 3, 4, 6] {
        let r = 64 * rng.range(0, (n / PAGE) as u64) as u32;
        v.push(Layout { nr, below_rows: r });
    }
    v
}

#[test]
fn tables_follow_the_engine_rule() {
    let mut rng = Rng(11);
    let (mut pooled, mut scattered, mut split, mut total) = (0, 0, 0, 0);
    for _ in 0..400 {
        let n = rng.range(1, 3000) as usize;
        for l in layouts(&mut rng, n) {
            let s = setup(&mut rng, l, n);
            total += 1;
            if s.pooled {
                pooled += 1;
                let v = &s.views[0];
                scattered += v.list.iter().enumerate().any(|(i, &p)| p as usize != i) as u32;
                split += v.list.iter().any(|&p| p >= v.place) as u32;
            }
            for b in [Binding { index: true, cp: true }, Binding { index: false, cp: false }] {
                for r in 0..l.nr.max(1) {
                    let v = &s.views[r as usize];
                    let want = oracle(b, &l, &v.list, v.place, r, n.div_ceil(PAGE));
                    assert_eq!(canon::table(b, &l, v, r, n.div_ceil(PAGE)).unwrap(), want);
                }
            }
        }
    }
    // the traces reach the cases that matter: Pool runs, scattered lists, past-R runs with drafter pages above place
    assert!(
        pooled * 6 > total && scattered * 4 > pooled && split > 50,
        "pooled {pooled} of {total}, scattered {scattered}, split {split}"
    );
    // the CP map as built (bind.cu): pages 0, 1, .. -> the prefix, then this rank's pages past R, in order
    let l = Layout { nr: 4, below_rows: 128 };
    let v = View { list: (0..6).collect(), place: u32::MAX };
    let t = canon::table(Binding { index: true, cp: true }, &l, &v, 1, 10).unwrap();
    assert_eq!(t, vec![0, 1, REMOTE, REMOTE, REMOTE, 2, REMOTE, REMOTE, REMOTE, 3]);
    assert!(canon::table(Binding { index: true, cp: true }, &Layout { nr: 1, below_rows: 0 }, &v, 0, 4).is_err());
    assert!(canon::table(Binding { index: true, cp: true }, &Layout { nr: 4, below_rows: 100 }, &v, 0, 4).is_err());
}

#[test]
fn same_layout_is_byte_identical() {
    let mut rng = Rng(23);
    for _ in 0..60 {
        let row = ROWS[rng.range(0, 1) as usize];
        let bs = bindings(&mut rng);
        let n = rng.range(1, 2500) as usize;
        let t = tokens(&mut rng, &bs, row, n);
        for l in layouts(&mut rng, n) {
            let s = setup(&mut rng, l, n);
            let src: Vec<Rank> =
                (0..s.views.len()).map(|r| write(&t, &bs, row, &l, &s.views[r], r as u32, s.pages, n, 0)).collect();
            let cs: Vec<Vec<Cache>> = src.iter().map(caches).collect();
            let c = canon::export(row, &bs, &l, &s.views, &cs, 0, n as u64, true).unwrap();
            assert_eq!(c.rows, n as u64);
            for (r, want) in src.iter().enumerate() {
                let mut got = write(&t, &bs, row, &l, &s.views[r], r as u32, s.pages, 0, 0);
                canon::import(&c, &bs, row, &l, &s.views[r], r as u32, &mut caches_mut(&mut got)).unwrap();
                assert!(
                    got.mla == want.mla && got.idx == want.idx,
                    "rank {r} of {l:?}: import differs from the source"
                );
            }
        }
    }
}

#[test]
fn another_layout_equals_its_own_writer() {
    let mut rng = Rng(37);
    for _ in 0..40 {
        let row = ROWS[rng.range(0, 1) as usize];
        let bs = bindings(&mut rng);
        let n = rng.range(1, 2500) as usize;
        let t = tokens(&mut rng, &bs, row, n);
        let ls = layouts(&mut rng, n);
        let mut first: Option<Canon> = None;
        for a in &ls {
            let sa = setup(&mut rng, *a, n);
            let src: Vec<Rank> =
                (0..sa.views.len()).map(|r| write(&t, &bs, row, a, &sa.views[r], r as u32, sa.pages, n, 0)).collect();
            let cs: Vec<Vec<Cache>> = src.iter().map(caches).collect();
            let c = canon::export(row, &bs, a, &sa.views, &cs, 0, n as u64, true).unwrap();
            // every layout gives the same canonical form
            match &first {
                None => first = Some(c.clone()),
                Some(f) => assert_eq!(&c, f, "{a:?} exports another canonical form"),
            }
            for b in &ls {
                let sb = setup(&mut rng, *b, n);
                let mut got: Vec<Rank> = Vec::new();
                for r in 0..sb.views.len() {
                    // pages the run does not cover keep their bytes: 0xA5 in both
                    let want = write(&t, &bs, row, b, &sb.views[r], r as u32, sb.pages, n, 0xA5);
                    let mut g = write(&t, &bs, row, b, &sb.views[r], r as u32, sb.pages, 0, 0xA5);
                    canon::import(&c, &bs, row, b, &sb.views[r], r as u32, &mut caches_mut(&mut g)).unwrap();
                    assert!(g.mla == want.mla && g.idx == want.idx, "{a:?} -> {b:?} rank {r}");
                    got.push(g);
                }
                let gs: Vec<Vec<Cache>> = got.iter().map(caches).collect();
                assert_eq!(canon::export(row, &bs, b, &sb.views, &gs, 0, n as u64, true).unwrap(), c);
            }
        }
    }
}

#[test]
fn a_chain_of_layouts_loses_nothing() {
    let mut rng = Rng(41);
    let bs = vec![
        Binding { index: true, cp: true },
        Binding { index: false, cp: true },
        Binding { index: false, cp: false },
    ];
    let n = 5000;
    let row = 656;
    let t = tokens(&mut rng, &bs, row, n);
    let chain = [
        Layout { nr: 6, below_rows: 1024 },
        Layout { nr: 4, below_rows: 2048 },
        Layout { nr: 3, below_rows: 64 },
        Layout { nr: 0, below_rows: 0 },
        Layout { nr: 6, below_rows: 1024 },
    ];
    let s0 = setup(&mut rng, chain[0], n);
    let mut ranks: Vec<Rank> =
        (0..s0.views.len()).map(|r| write(&t, &bs, row, &chain[0], &s0.views[r], r as u32, s0.pages, n, 0)).collect();
    let mut views = s0.views.clone();
    let mut c0 = None;
    for w in chain.windows(2) {
        let cs: Vec<Vec<Cache>> = ranks.iter().map(caches).collect();
        let c = canon::export(row, &bs, &w[0], &views, &cs, 0, n as u64, true).unwrap();
        if let Some(f) = &c0 {
            assert_eq!(&c, f);
        }
        c0.get_or_insert(c.clone());
        // through bytes, as a migration over the fabric or a store would carry it
        let c = Canon::from_bytes(&c.to_bytes()).unwrap();
        let s = setup(&mut rng, w[1], n);
        ranks = (0..s.views.len())
            .map(|r| {
                let mut g = write(&t, &bs, row, &w[1], &s.views[r], r as u32, s.pages, 0, 0);
                canon::import(&c, &bs, row, &w[1], &s.views[r], r as u32, &mut caches_mut(&mut g)).unwrap();
                g
            })
            .collect();
        views = s.views;
    }
    // back at TP6 with R 1024, on another page list: the same rows on the pages that list names
    let cs: Vec<Vec<Cache>> = ranks.iter().map(caches).collect();
    assert_eq!(&canon::export(row, &bs, &chain[4], &views, &cs, 0, n as u64, true).unwrap(), c0.as_ref().unwrap());
    let fresh: Vec<Rank> = (0..views.len())
        .map(|r| write(&t, &bs, row, &chain[4], &views[r], r as u32, ranks[r].mla[0].len() / (PAGE * row), n, 0))
        .collect();
    for r in 0..views.len() {
        assert!(ranks[r].mla == fresh[r].mla && ranks[r].idx == fresh[r].idx, "rank {r} after the chain");
    }
}

#[test]
fn bulk_then_tail_equals_one_copy() {
    let mut rng = Rng(53);
    for _ in 0..40 {
        let row = ROWS[rng.range(0, 1) as usize];
        let bs = bindings(&mut rng);
        let total = rng.range(2, 3000) as usize;
        let bulk = rng.range(1, total as u64 - 1) as usize;
        let t = tokens(&mut rng, &bs, row, total);
        let ls = layouts(&mut rng, total);
        let a = ls[rng.range(0, ls.len() as u64 - 1) as usize];
        let b = ls[rng.range(0, ls.len() as u64 - 1) as usize];
        let sa = setup(&mut rng, a, total);
        let sb = setup(&mut rng, b, total);
        // the session decodes on at A while its first `bulk` rows copy: A's caches at the cutover hold every row
        let src: Vec<Rank> =
            (0..sa.views.len()).map(|r| write(&t, &bs, row, &a, &sa.views[r], r as u32, sa.pages, total, 0)).collect();
        let cs: Vec<Vec<Cache>> = src.iter().map(caches).collect();
        let c1 = canon::export(row, &bs, &a, &sa.views, &cs, 0, bulk as u64, true).unwrap();
        let c2 = canon::export(row, &bs, &a, &sa.views, &cs, bulk as u64, (total - bulk) as u64, true).unwrap();
        let whole = canon::export(row, &bs, &a, &sa.views, &cs, 0, total as u64, true).unwrap();
        // the two parts are the whole form's rows, split at `bulk`
        for (bi, d) in whole.data.iter().enumerate() {
            assert_eq!(d.mla[..bulk * row], c1.data[bi].mla[..]);
            assert_eq!(d.mla[bulk * row..], c2.data[bi].mla[..]);
            if let Some(ix) = &d.idx {
                assert_eq!(ix[..bulk * IDX_ROW], c1.data[bi].idx.as_ref().unwrap()[..]);
                assert_eq!(ix[bulk * IDX_ROW..], c2.data[bi].idx.as_ref().unwrap()[..]);
            }
        }
        for r in 0..sb.views.len() {
            let want = write(&t, &bs, row, &b, &sb.views[r], r as u32, sb.pages, total, 0);
            let mut g = write(&t, &bs, row, &b, &sb.views[r], r as u32, sb.pages, 0, 0);
            let w1 = canon::import(&c1, &bs, row, &b, &sb.views[r], r as u32, &mut caches_mut(&mut g)).unwrap();
            let w2 = canon::import(&c2, &bs, row, &b, &sb.views[r], r as u32, &mut caches_mut(&mut g)).unwrap();
            let mut g2 = write(&t, &bs, row, &b, &sb.views[r], r as u32, sb.pages, 0, 0);
            let w = canon::import(&whole, &bs, row, &b, &sb.views[r], r as u32, &mut caches_mut(&mut g2)).unwrap();
            assert_eq!(w1 + w2, w, "bulk + tail write the rows one copy writes");
            assert!(g.mla == want.mla && g.idx == want.idx, "{a:?} -> {b:?} rank {r}: bulk + tail");
            assert!(g2.mla == want.mla && g2.idx == want.idx, "{a:?} -> {b:?} rank {r}: one copy");
        }
    }
}

#[test]
fn a_replicated_page_that_differs_is_caught() {
    let mut rng = Rng(67);
    let bs = vec![Binding { index: true, cp: true }, Binding { index: false, cp: false }];
    let (row, n) = (656, 700);
    let t = tokens(&mut rng, &bs, row, n);
    let l = Layout { nr: 4, below_rows: 256 };
    let s = setup(&mut rng, l, n);
    let mut src: Vec<Rank> = (0..4).map(|r| write(&t, &bs, row, &l, &s.views[r], r as u32, s.pages, n, 0)).collect();
    // rank 2's copy of logical page 1 (replicated: below R) gets one byte wrong in its indexer scale
    let p = canon::table(bs[0], &l, &s.views[2], 2, 2).unwrap()[1] as usize;
    let at = p * PAGE * IDX_ROW + PAGE * 128 + 5 * 4;
    src[2].idx[0].as_mut().unwrap()[at] ^= 1;
    let cs: Vec<Vec<Cache>> = src.iter().map(caches).collect();
    assert_eq!(
        canon::export(row, &bs, &l, &s.views, &cs, 0, n as u64, true),
        Err(canon::Error::Disagree { binding: 0, page: 1, rank: 2 })
    );
    // without the check the lowest rank's copy wins
    assert!(canon::export(row, &bs, &l, &s.views, &cs, 0, n as u64, false).is_ok());
    // a list too short for the rows: no rank holds the page
    let short: Vec<View> = s.views.iter().map(|v| View { list: v.list[..2].to_vec(), place: v.place }).collect();
    assert_eq!(
        canon::export(row, &bs, &l, &short, &cs, 0, n as u64, false),
        Err(canon::Error::Bad("rows past the run's list"))
    );
}

#[test]
fn the_byte_form_round_trips_and_refuses_others() {
    let mut rng = Rng(71);
    let bs = bindings(&mut rng);
    let (row, n) = (416, 333);
    let t = tokens(&mut rng, &bs, row, n);
    let l = Layout { nr: 0, below_rows: 0 };
    let s = setup(&mut rng, l, n);
    let src = write(&t, &bs, row, &l, &s.views[0], 0, s.pages, n, 0);
    let c = canon::export(row, &bs, &l, &s.views, &[caches(&src)], 7, 300, true).unwrap();
    let b = c.to_bytes();
    assert_eq!(b.len(), c.byte_len());
    assert_eq!(Canon::from_bytes(&b).unwrap(), c);
    assert!(Canon::from_bytes(&b[..b.len() - 1]).is_err());
    let mut longer = b.clone();
    longer.push(0);
    assert!(Canon::from_bytes(&longer).is_err());
    let mut other = b.clone();
    other[0] ^= 1;
    assert!(Canon::from_bytes(&other).is_err());
    // sizes in the header that overflow or exceed the blob are refused before anything is sliced (a release build
    // aborts on a panic)
    for rows in [u64::MAX, u64::MAX / 2 + 1, 1 << 40, c.rows + 1] {
        let mut x = b.clone();
        x[24..32].copy_from_slice(&rows.to_le_bytes());
        assert!(Canon::from_bytes(&x).is_err(), "rows {rows}");
    }
    let mut x = b.clone();
    x[8..12].copy_from_slice(&1u32.to_le_bytes());
    assert!(Canon::from_bytes(&x).is_err());
    let mut x = b.clone();
    x[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Canon::from_bytes(&x).is_err());
    let mut x = b.clone();
    x[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Canon::from_bytes(&x).is_err());
    // a hand-built form whose data does not match its header is refused at import
    let mut bent = c.clone();
    bent.data[0].mla.pop();
    let mut dst = write(&t, &bs, row, &l, &s.views[0], 0, s.pages, 0, 0);
    assert!(canon::import(&bent, &bs, row, &l, &s.views[0], 0, &mut caches_mut(&mut dst)).is_err());
    // another row format or other bindings are refused at import
    let mut dst = write(&t, &bs, 656, &l, &s.views[0], 0, s.pages, 0, 0);
    assert_eq!(
        canon::import(&c, &bs, 656, &l, &s.views[0], 0, &mut caches_mut(&mut dst)),
        Err(canon::Error::Mismatch("MLA row format"))
    );
    let mut fewer = bs.clone();
    fewer.pop();
    let mut dst = write(&t, &fewer, row, &l, &s.views[0], 0, s.pages, 0, 0);
    assert_eq!(
        canon::import(&c, &fewer, row, &l, &s.views[0], 0, &mut caches_mut(&mut dst)),
        Err(canon::Error::Mismatch("layer bindings"))
    );
}

#[test]
fn head_split_state_regroups_for_every_width() {
    let mut rng = Rng(83);
    for heads in [8usize, 16, 32, 64] {
        let (outer, inner) = (rng.range(1, 5) as usize, rng.range(1, 300) as usize);
        let whole = rng.bytes(outer * heads * inner);
        for nr in 1..=heads.min(8) {
            let parts: Vec<Vec<u8>> =
                (0..nr).map(|j| canon::heads_in(&whole, heads, outer, inner, nr, j).unwrap()).collect();
            // the widths are the engine's (g6nr.h): larger chunks first, every head once
            let w: Vec<usize> = parts.iter().map(|p| p.len() / (outer * inner)).collect();
            assert_eq!(w.iter().sum::<usize>(), heads);
            assert!(w.windows(2).all(|x| x[0] >= x[1] && x[0] - x[1] <= 1));
            let refs: Vec<&[u8]> = parts.iter().map(|p| &p[..]).collect();
            assert_eq!(canon::heads_out(&refs, heads, outer, inner).unwrap(), whole);
            // and through another width
            for nr2 in [1, 2, 3, 4, 6] {
                if nr2 > heads {
                    continue;
                }
                let p2: Vec<Vec<u8>> =
                    (0..nr2).map(|j| canon::heads_in(&whole, heads, outer, inner, nr2, j).unwrap()).collect();
                let r2: Vec<&[u8]> = p2.iter().map(|p| &p[..]).collect();
                assert_eq!(canon::heads_out(&r2, heads, outer, inner).unwrap(), whole);
            }
        }
    }
    assert!(canon::heads_in(&[0u8; 10], 8, 1, 1, 2, 0).is_err());
    assert!(canon::heads_out(&[&[0u8; 3][..]], 8, 1, 1).is_err());
}
