//! Placement as data: which rank holds each piece of the model. A piece is one 128-column half of an expert (its gate
//! and up rows and its down columns), one attention head (its q_b rows, UK / UV and o_proj columns), one 128-channel
//! block of a dense MLP, or one 128-row tile of lm_head. Every rank holds everything else, so the map leaves it out.
//!
//! A layout is the member list (rank r is members[r]) and the owner of every piece. Today's formulas
//! (tools/convert/plan.py, the v2 owner in tools/convert/v2.py) are one way to make one: `Layout::formula`. `plan`
//! makes a layout for a new member list from an old layout and moves as few pieces as it can.
//!
//! The engine reads how many pieces a rank holds, not which (engine/attn/bind.cu, engine/kernels/bind.cuh), except
//! for vocab tiles. So every layout keeps the counts the build expects (`Layout::check`):
//!   heads   rank r holds the formula's count of layer l's heads: widths(64, nr) rotated by l (bind.cu checks UK, UV)
//!   dense   96 / nr blocks of each dense layer a rank (DENSE_CH)
//!   halves  16 / nr halves of each expert a rank, rounded down or up (at most EB, and at least one)
//!   vocab   the formula's tiles: the engine maps a logit to its token id by the rank's range (vocab_slice)
//! Every piece of a group costs the same compute, so these counts balance compute. Bytes are balanced over the
//! expert halves, whose widths differ.

use crate::json::{self, Json};
use crate::v2plan::{self, HALF_PER_BIT, NH, SHARED};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;

const FORMAT: &str = "g6layout-v1";
const HEADS: usize = 64;
const DENSE: usize = 96;
const VTILES: usize = 1210;
const L_MTP: usize = 78;

/// A set of interchangeable pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    /// the 16 halves of expert e (256: the shared expert) in layer l
    Halves(usize, usize),
    /// the 64 attention heads of layer l
    Heads(usize),
    /// the 96 128-channel blocks of layer l's dense MLP
    Dense(usize),
    /// the 1210 128-row tiles of lm_head
    Vocab,
}

impl Group {
    pub fn len(self) -> usize {
        match self {
            Group::Halves(..) => NH,
            Group::Heads(_) => HEADS,
            Group::Dense(_) => DENSE,
            Group::Vocab => VTILES,
        }
    }

    pub fn kind(self) -> &'static str {
        match self {
            Group::Halves(..) => "halves",
            Group::Heads(_) => "heads",
            Group::Dense(_) => "dense",
            Group::Vocab => "vocab",
        }
    }

    /// The group's key in a layout file: "halves L E", "heads L", "dense L" or "vocab".
    fn key(self) -> String {
        match self {
            Group::Halves(l, e) => format!("halves {l} {e}"),
            Group::Heads(l) => format!("heads {l}"),
            Group::Dense(l) => format!("dense {l}"),
            Group::Vocab => "vocab".into(),
        }
    }

    fn from_key(k: &str) -> Option<Group> {
        let w: Vec<&str> = k.split(' ').collect();
        let n = |i: usize| w.get(i).and_then(|s| s.parse::<usize>().ok());
        match (w[0], w.len()) {
            ("halves", 3) => Some(Group::Halves(n(1)?, n(2)?)),
            ("heads", 2) => Some(Group::Heads(n(1)?)),
            ("dense", 2) => Some(Group::Dense(n(1)?)),
            ("vocab", 1) => Some(Group::Vocab),
            _ => None,
        }
    }

    /// Today's owner of each piece at TP nr.
    fn formula(self, nr: usize) -> Vec<u8> {
        // chunk j of widths(n, nr), the larger first, goes to rank (j + off) % nr (plan.py _chunks)
        let chunks = |n: usize, off: usize| -> Vec<u8> {
            (0..nr).flat_map(|j| std::iter::repeat_n(((j + off) % nr) as u8, n / nr + usize::from(j < n % nr))).collect()
        };
        match self {
            Group::Halves(l, e) => (0..NH).map(|h| ((16 * e + h + l) % nr) as u8).collect(),
            Group::Heads(l) => chunks(HEADS, l),
            Group::Dense(l) => chunks(DENSE, l),
            Group::Vocab => chunks(VTILES, 0),
        }
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Group::Halves(l, e) if *e == SHARED => write!(f, "layer {l}'s shared expert halves"),
            Group::Halves(l, e) => write!(f, "layer {l} expert {e}'s halves"),
            Group::Heads(l) => write!(f, "layer {l}'s heads"),
            Group::Dense(l) => write!(f, "layer {l}'s dense blocks"),
            Group::Vocab => write!(f, "the vocab tiles"),
        }
    }
}

/// A model's pieces by group, with each piece's bytes: what a box fetches to hold it, the expert halves' bytes in
/// the width files. Heads, dense blocks and vocab tiles come from the dense and vocabulary files, which every box
/// holds whole, so they count 0.
pub struct Model {
    pub groups: Vec<(Group, Vec<u64>)>,
}

impl Model {
    /// GLM 5.3 (layers 0-78, 78 the MTP layer) at the widths of a width manifest.
    pub fn glm53(man: &Json) -> Result<Model, String> {
        let mut groups = vec![(Group::Vocab, vec![0; VTILES])];
        for l in 0..=L_MTP {
            groups.push((Group::Heads(l), vec![0; HEADS]));
            if l < 3 {
                groups.push((Group::Dense(l), vec![0; DENSE]));
                continue;
            }
            let w = v2plan::widths(man, l)?.ok_or(format!("the width manifest has no widths for MoE layer {l}"))?;
            let half = |g: &[u8; NH], u: &[u8; NH], d: u8| -> Vec<u64> {
                (0..NH).map(|h| HALF_PER_BIT * (g[h] as u64 + u[h] as u64 + d as u64)).collect()
            };
            for e in 0..w.down.len() {
                groups.push((Group::Halves(l, e), half(&w.gate[e], &w.up[e], w.down[e])));
            }
            if w.sdown > 0 {
                groups.push((Group::Halves(l, SHARED), half(&w.sgate, &w.sup, w.sdown)));
            }
        }
        Ok(Model { groups })
    }
}

/// gr8 builds TP 3, 6 or a divisor of 64, and the dense MLP's rank slices must fall on the release's 128-channel
/// blocks.
pub fn check_nr(nr: usize) -> Result<(), String> {
    if !(nr == 3 || nr == 6 || (nr > 0 && 64 % nr == 0)) {
        return Err(format!("TP {nr}: gr8 builds TP 3, 6 or a divisor of 64"));
    }
    if DENSE % nr != 0 {
        return Err(format!("TP {nr}: the dense MLP's rank slices do not fall on the release's 128-channel blocks"));
    }
    Ok(())
}

fn counts(o: &[u8], nr: usize) -> Vec<usize> {
    let mut c = vec![0; nr];
    for &r in o {
        c[r as usize] += 1;
    }
    c
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// rank r is members[r]
    pub members: Vec<String>,
    owner: BTreeMap<Group, Vec<u8>>,
}

impl Layout {
    pub fn nr(&self) -> usize {
        self.members.len()
    }

    /// Today's layout of `model` over `members`.
    pub fn formula(model: &Model, members: Vec<String>) -> Result<Layout, String> {
        check_nr(members.len())?;
        let owner = model.groups.iter().map(|(g, _)| (*g, g.formula(members.len()))).collect();
        Ok(Layout { members, owner })
    }

    /// The pieces of `g` rank `rank` holds, in ascending order.
    pub fn held(&self, g: Group, rank: usize) -> Result<Vec<usize>, String> {
        let o = self.owner.get(&g).ok_or(format!("the layout has no {g}"))?;
        Ok((0..o.len()).filter(|&i| o[i] as usize == rank).collect())
    }

    fn check_members(&self) -> Result<(), String> {
        check_nr(self.nr())?;
        match self.members.iter().find(|m| self.members.iter().filter(|x| x == m).count() > 1) {
            Some(m) => Err(format!("member {m} is listed twice")),
            None => Ok(()),
        }
    }

    /// Every piece of `model` has one owner, and each rank holds the counts this build's engine reads (module doc).
    pub fn check(&self, model: &Model) -> Result<(), String> {
        self.check_members()?;
        let nr = self.nr();
        for (g, _) in &model.groups {
            let o = self.owner.get(g).ok_or(format!("the layout has no {g}"))?;
            if o.len() != g.len() {
                return Err(format!("{g}: {} owners, {} pieces", o.len(), g.len()));
            }
            let want = g.formula(nr);
            let c = counts(o, nr);
            let ok = match g {
                Group::Vocab => *o == want,
                Group::Halves(..) => c.iter().all(|&n| n >= NH / nr && n <= NH.div_ceil(nr)),
                _ => c == counts(&want, nr),
            };
            if !ok {
                return Err(format!("{g}: rank counts {c:?} at TP {nr}, the engine reads {}", match g {
                    Group::Vocab => "the formula's tiles".to_string(),
                    Group::Halves(..) => format!("{} or {} halves a rank", NH / nr, NH.div_ceil(nr)),
                    _ => format!("{:?}", counts(&want, nr)),
                }));
            }
        }
        Ok(())
    }

    /// The layout file: one line per group, each piece's owner in piece order.
    pub fn to_json(&self) -> String {
        let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
        let members: Vec<String> = self.members.iter().map(|m| q(m)).collect();
        let mut out = format!("{{\"format\": \"{FORMAT}\", \"members\": [{}], \"groups\": {{", members.join(", "));
        for (i, (g, o)) in self.owner.iter().enumerate() {
            let o: Vec<String> = o.iter().map(u8::to_string).collect();
            out += &format!("{}\n\"{}\": [{}]", if i > 0 { "," } else { "" }, g.key(), o.join(", "));
        }
        out + "\n}}\n"
    }

    pub fn parse(text: &str) -> Result<Layout, String> {
        let j = json::parse(text)?;
        if j.get("format").and_then(Json::str) != Some(FORMAT) {
            return Err(format!("not a {FORMAT} layout"));
        }
        let members = j.get("members").and_then(Json::arr).ok_or("a layout without members")?
            .iter().map(|m| m.str().map(str::to_string).ok_or("a member that is not a string"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut owner = BTreeMap::new();
        for (k, v) in j.get("groups").and_then(Json::obj).ok_or("a layout without groups")? {
            let g = Group::from_key(k).ok_or(format!("layout group {k:?}"))?;
            let o = v.arr().ok_or(format!("{g}: not a list"))?.iter()
                .map(|r| r.int().filter(|&r| r >= 0 && (r as usize) < members.len()).map(|r| r as u8))
                .collect::<Option<Vec<u8>>>().ok_or(format!("{g}: an owner that is not a rank of {} members", members.len()))?;
            owner.insert(g, o);
        }
        Ok(Layout { members, owner })
    }

    pub fn read(path: &str) -> Result<Layout, String> {
        Layout::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))
    }
}

/// A layout of `model` for `members`, planned from `old`. Each group gets the formula's counts at the new width; an
/// expert's larger counts may go to other ranks than the formula's (the engine allows it), see `order`. Every member
/// keeps the pieces it held, in piece order, up to its count. The rest, largest first, go to the rank with the fewest
/// bytes that still has room. Pieces are interchangeable within a group, so the assignment problem (SpotServe's
/// Kuhn-Munkres) comes down to counts: a group moves n - sum over ranks of min(held, count) pieces, the fewest its
/// counts allow. Vocab tiles follow the formula. A member not in `old` starts empty; a member of `old` that is not in
/// `members` gives up every piece.
pub fn plan(model: &Model, old: &Layout, members: Vec<String>) -> Result<Layout, String> {
    let nr = members.len();
    let new = Layout { members, owner: BTreeMap::new() };
    new.check_members()?;
    let to_new: Vec<Option<usize>> = old.members.iter().map(|m| new.members.iter().position(|x| x == m)).collect();
    let mut bytes = vec![0u64; nr];
    let mut halves = vec![0usize; nr];
    let mut owner = BTreeMap::new();
    for (g, size) in &model.groups {
        let want = g.formula(nr);
        if *g == Group::Vocab {
            owner.insert(*g, want);
            continue;
        }
        let prev = old.owner.get(g).ok_or(format!("the old layout has no {g}"))?;
        let mut held: Vec<Vec<usize>> = vec![Vec::new(); nr];
        for (i, &r) in prev.iter().enumerate() {
            if let Some(n) = to_new.get(r as usize).copied().flatten() {
                held[n].push(i);
            }
        }
        let mut count = counts(&want, nr);
        if let Group::Halves(..) = g {
            count = order(&held, count, &halves);
            halves.iter_mut().zip(&count).for_each(|(h, c)| *h += c);
        }
        let mut o = vec![u8::MAX; g.len()];
        let mut have = vec![0usize; nr];
        for r in 0..nr {
            for &i in held[r].iter().take(count[r]) {
                o[i] = r as u8;
                have[r] += 1;
                bytes[r] += size[i];
            }
        }
        let mut free: Vec<usize> = (0..g.len()).filter(|&i| o[i] == u8::MAX).collect();
        free.sort_by_key(|&i| Reverse(size[i]));
        for i in free {
            let r = (0..nr).filter(|&r| have[r] < count[r]).min_by_key(|&r| bytes[r]).ok_or(format!("{g}: no room"))?;
            o[i] = r as u8;
            have[r] += 1;
            bytes[r] += size[i];
        }
        owner.insert(*g, o);
    }
    Ok(Layout { owner, ..new })
}

/// A rank may run this many halves ahead of the others to keep a half it holds.
const SLACK: usize = NH;

/// An expert's counts (16 / nr rounded down or up) in the rank order that balances the halves over the model and
/// keeps the most in place within that: the larger counts go to the ranks with the fewest halves so far, a rank that
/// holds more than the smaller count (it keeps the extra half) counting SLACK halves fewer. The formula's order breaks
/// ties. Putting the larger counts where most is held alone would load the members that stay: going 4 -> 6, the four
/// would take the larger count of every expert, and the two new members two thirds of their halves.
fn order(held: &[Vec<usize>], count: Vec<usize>, sofar: &[usize]) -> Vec<usize> {
    let small = *count.iter().min().unwrap();
    let mut by: Vec<usize> = (0..held.len()).collect();
    by.sort_by_key(|&r| (sofar[r] + SLACK * usize::from(held[r].len() <= small), Reverse(count[r])));
    let mut sorted = count.clone();
    sorted.sort_unstable_by_key(|&c| Reverse(c));
    let mut out = vec![0; held.len()];
    for (&r, &c) in by.iter().zip(&sorted) {
        out[r] = c;
    }
    out
}

/// What a re-plan moves: pieces by kind that change member, and per new rank the bytes it gains and holds.
#[derive(Debug, Default)]
pub struct Moves {
    pub pieces: BTreeMap<&'static str, usize>,
    pub gained: Vec<u64>,
    pub holds: Vec<u64>,
}

pub fn moves(model: &Model, old: &Layout, new: &Layout) -> Result<Moves, String> {
    let mut m = Moves { gained: vec![0; new.nr()], holds: vec![0; new.nr()], ..Moves::default() };
    for (g, size) in &model.groups {
        let (a, b) = (old.owner.get(g).ok_or(format!("the old layout has no {g}"))?, new.owner.get(g).ok_or(format!("the new layout has no {g}"))?);
        for i in 0..g.len() {
            let r = b[i] as usize;
            m.holds[r] += size[i];
            if old.members[a[i] as usize] != new.members[r] {
                *m.pieces.entry(g.kind()).or_default() += 1;
                m.gained[r] += size[i];
            }
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(s: &str) -> Vec<String> {
        s.split(',').map(str::to_string).collect()
    }

    /// GLM 5.3's groups with every half's bytes from a fixed pseudo-random width mix (K2-K4 per projection).
    fn model() -> Model {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut k = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            2 + x % 3
        };
        let mut groups = vec![(Group::Vocab, vec![0; VTILES])];
        for l in 0..=L_MTP {
            groups.push((Group::Heads(l), vec![0; HEADS]));
            if l < 3 {
                groups.push((Group::Dense(l), vec![0; DENSE]));
                continue;
            }
            for e in (0..256).chain([SHARED]) {
                let d = k();
                groups.push((Group::Halves(l, e), (0..NH).map(|_| HALF_PER_BIT * (k() + k() + d)).collect()));
            }
        }
        Model { groups }
    }

    fn range(l: &Layout, g: Group, r: usize) -> (usize, usize) {
        let h = l.held(g, r).unwrap();
        assert!(h.windows(2).all(|w| w[1] == w[0] + 1), "{g} of rank {r} is not contiguous");
        (h[0], h.len())
    }

    #[test]
    fn the_formula_is_todays_ownership() {
        // tools/convert/plan.py at TP4: heads 16 per rank rotating with the layer, dense 24 blocks (3072 channels),
        // vocab tiles 303, 303, 302, 302
        let m = model();
        let t4 = Layout::formula(&m, names("a,b,c,d")).unwrap();
        assert_eq!(range(&t4, Group::Heads(0), 1), (16, 16));
        assert_eq!(range(&t4, Group::Heads(1), 1), (0, 16));
        assert_eq!(range(&t4, Group::Heads(2), 1), (48, 16));
        assert_eq!(range(&t4, Group::Dense(0), 1), (24, 24));
        assert_eq!(range(&t4, Group::Vocab, 1), (303, 303));
        // TP3: 22, 21, 21 heads. Layer 4's rank 0 holds the last group (the TP3 pack's L004 heads [43, 64])
        let t3 = Layout::formula(&m, names("a,b,c")).unwrap();
        assert_eq!(range(&t3, Group::Heads(0), 0), (0, 22));
        assert_eq!(range(&t3, Group::Heads(4), 0), (43, 21));
        assert_eq!(range(&t3, Group::Dense(0), 0), (0, 32));
        assert_eq!(range(&t3, Group::Vocab, 0), (0, 404));
        // TP6: 11, 11, 11, 11, 10, 10
        let t6 = Layout::formula(&m, names("a,b,c,d,e,f")).unwrap();
        assert_eq!(range(&t6, Group::Heads(0), 5), (54, 10));
        for (t, nr) in [(&t3, 3), (&t4, 4), (&t6, 6)] {
            t.check(&m).unwrap();
            for r in 0..nr {
                assert_eq!(t.held(Group::Halves(7, 9), r).unwrap(), v2plan::formula_halves(9, 7, nr, r));
                assert_eq!(t.held(Group::Halves(5, SHARED), r).unwrap(), v2plan::formula_halves(SHARED, 5, nr, r));
            }
        }
        assert!(Layout::formula(&m, names("a,b,c,d,e")).is_err());
    }

    /// The fewest pieces that can move given the new layout's counts, worked out independently of the planner: in
    /// each group a member gains only what its count asks beyond what it held.
    fn fewest(m: &Model, old: &Layout, new: &Layout) -> usize {
        let mut n = 0;
        for (g, _) in m.groups.iter().filter(|(g, _)| *g != Group::Vocab) {
            for (r, x) in new.members.iter().enumerate() {
                let held = old.members.iter().position(|y| y == x).map_or(0, |o| old.held(*g, o).unwrap().len());
                let count = new.held(*g, r).unwrap().len();
                n += count - held.min(count);
            }
        }
        n
    }

    fn moved(m: &Model, a: &Layout, b: &Layout) -> usize {
        moves(m, a, b).unwrap().pieces.iter().filter(|(k, _)| **k != "vocab").map(|(_, n)| n).sum()
    }

    #[test]
    fn four_to_three_keeps_every_piece_and_spreads_the_fourth() {
        let m = model();
        let t4 = Layout::formula(&m, names("a,b,c,d")).unwrap();
        let p = plan(&m, &t4, names("a,b,d")).unwrap();
        p.check(&m).unwrap();
        for (g, _) in m.groups.iter().filter(|(g, _)| *g != Group::Vocab) {
            for (new, old) in [(0, 0), (1, 1), (2, 3)] {
                let (o, n) = (t4.held(*g, old).unwrap(), p.held(*g, new).unwrap());
                assert!(o.iter().all(|i| n.contains(i)), "{g}: rank {new} lost a piece");
            }
        }
        let mv = moves(&m, &t4, &p).unwrap();
        // what moves is c's quarter: a twelfth of the model to each of the three
        let halves = m.groups.iter().filter(|(g, _)| matches!(g, Group::Halves(..))).count() * NH;
        assert_eq!(mv.pieces["halves"], halves / 4);
        assert_eq!(mv.pieces["heads"], 79 * 16);
        assert_eq!(mv.pieces["dense"], 3 * 24);
        assert_eq!(moved(&m, &t4, &p), fewest(&m, &t4, &p));
        let total: u64 = mv.holds.iter().sum();
        for &g in &mv.gained {
            assert!((g as f64 / total as f64 - 1.0 / 12.0).abs() < 0.002, "gained {g} of {total}");
        }
        for &h in &mv.holds {
            assert!((h as f64 / total as f64 - 1.0 / 3.0).abs() < 0.002, "holds {h} of {total}");
        }
    }

    #[test]
    fn three_to_four_moves_the_new_members_quarter() {
        let m = model();
        let t3 = Layout::formula(&m, names("a,b,c")).unwrap();
        let p = plan(&m, &t3, names("a,b,c,d")).unwrap();
        p.check(&m).unwrap();
        let mv = moves(&m, &t3, &p).unwrap();
        let halves = m.groups.iter().filter(|(g, _)| matches!(g, Group::Halves(..))).count() * NH;
        assert_eq!(mv.pieces["halves"], halves / 4);
        assert_eq!(mv.pieces["heads"], 79 * 16);
        assert_eq!(mv.gained[..3], [0, 0, 0]);
        assert_eq!(moved(&m, &t3, &p), fewest(&m, &t3, &p));
        let total: u64 = mv.holds.iter().sum();
        for &h in &mv.holds {
            assert!((h as f64 / total as f64 - 0.25).abs() < 0.002, "holds {h} of {total}");
        }
    }

    #[test]
    fn a_chain_of_changes_moves_the_fewest_each_time() {
        // members join, leave and change rank: each step keeps the engine's counts, balances the halves and their
        // bytes, and moves no piece its counts do not need
        let m = model();
        let mut cur = Layout::formula(&m, names("a,b,c,d")).unwrap();
        for next in ["b,c,d", "e,b,c,d", "e,b,c,d,f,g", "g,e,b,c", "a,g,e"] {
            let p = plan(&m, &cur, names(next)).unwrap();
            p.check(&m).unwrap();
            assert_eq!(moved(&m, &cur, &p), fewest(&m, &cur, &p), "{next}");
            let mv = moves(&m, &cur, &p).unwrap();
            let total: u64 = mv.holds.iter().sum();
            let mean = total as f64 / p.nr() as f64;
            assert!(mv.holds.iter().all(|&h| (h as f64 / mean - 1.0).abs() < 0.005), "{next}: {:?}", mv.holds);
            let mut n = vec![0usize; p.nr()];
            for (g, _) in m.groups.iter().filter(|(g, _)| matches!(g, Group::Halves(..))) {
                n.iter_mut().zip(counts(&p.owner[g], p.nr())).for_each(|(a, b)| *a += b);
            }
            let mean = n.iter().sum::<usize>() as f64 / p.nr() as f64;
            assert!(n.iter().all(|&h| (h as f64 / mean - 1.0).abs() < 0.001), "{next}: {n:?}");
            for (g, _) in &m.groups {
                assert_eq!(counts(&p.owner[g], p.nr()).iter().sum::<usize>(), g.len());
            }
            cur = p;
        }
        // going 4 -> 6 the four keep their share and the two new members fetch theirs: halves move only to e and f...
        let t4 = Layout::formula(&m, names("a,b,c,d")).unwrap();
        let p6 = plan(&m, &t4, names("a,b,c,d,e,f")).unwrap();
        let mv = moves(&m, &t4, &p6).unwrap();
        assert_eq!(mv.gained[..4], [0, 0, 0, 0]);
        let halves: usize = m.groups.iter().filter(|(g, _)| matches!(g, Group::Halves(..))).count() * NH;
        assert!((mv.pieces["halves"] as f64 / halves as f64 - 2.0 / 6.0).abs() < 0.001, "{}", mv.pieces["halves"]);
        // ...and the same members in the same order: nothing moves
        let same = plan(&m, &cur, cur.members.clone()).unwrap();
        assert_eq!(moves(&m, &cur, &same).unwrap().pieces.values().sum::<usize>(), 0);
    }

    #[test]
    fn check_refuses_what_the_engine_cannot_read() {
        let m = model();
        let t4 = Layout::formula(&m, names("a,b,c,d")).unwrap();
        let bad = |f: &dyn Fn(&mut Layout)| {
            let mut x = t4.clone();
            f(&mut x);
            x.check(&m).unwrap_err()
        };
        // a head from rank 0 to rank 1: 15 and 17 heads
        let e = bad(&|x| x.owner.get_mut(&Group::Heads(0)).unwrap()[0] = 1);
        assert!(e.contains("layer 0's heads"), "{e}");
        // two vocab tiles swapped: the counts hold, the ranges do not
        let e = bad(&|x| x.owner.get_mut(&Group::Vocab).unwrap().swap(0, 400));
        assert!(e.contains("vocab"), "{e}");
        // 5 halves of an expert on one rank at TP4 (EB 4)
        let e = bad(&|x| {
            let o = x.owner.get_mut(&Group::Halves(3, 0)).unwrap();
            let i = o.iter().position(|&r| r == 1).unwrap();
            o[i] = 0;
        });
        assert!(e.contains("expert 0"), "{e}");
        let e = bad(&|x| { x.owner.remove(&Group::Dense(2)); });
        assert!(e.contains("no layer 2's dense"), "{e}");
        let e = bad(&|x| x.members[3] = "a".into());
        assert!(e.contains("twice"), "{e}");
        assert!(plan(&m, &t4, names("a,b,a")).is_err());
        assert!(plan(&m, &t4, names("a,b,c,d,e")).is_err());
    }

    #[test]
    fn the_file_reads_back() {
        let m = model();
        let p = plan(&m, &Layout::formula(&m, names("a,b,c,d")).unwrap(), names("a,b,\"d\"")).unwrap();
        let q = Layout::parse(&p.to_json()).unwrap();
        assert_eq!(p, q);
        assert!(Layout::parse(&p.to_json().replace("\"halves 3 0\": [", "\"halves 3 0\": [7, ")).is_err());
        assert!(Layout::parse(&p.to_json().replace(FORMAT, "x")).is_err());
    }
}
