//! A session's state in a form that does not depend on the layout that computed it (#152): export from the pages of
//! one layout, import into the pages of another. Host only; the engine (or a test) copies the pages between device
//! memory and the buffers here.
//!
//! What a rank holds of a run, per layer binding (engine/runtime/s2.cuh s2_solo_cp_pages, engine/attn/bind.cu):
//! - a binding that is not CP (the MTP layer, the draft model), or a layout without CP: logical page lp at list[lp];
//! - a CP binding: lp < R/64 (the replicated prefix) at list[lp]; past R, the pages with lp % nr == rank at the
//!   (R/64 + j)-th page of the list below `place` (j counts this rank's pages past R; s2_list_place), every other page
//!   on another rank (REMOTE). The pool grants a past-R run its low class first (pool.rs), so its prefix pages are
//!   below place too. A run below R (every page in the prefix), or a run past R on the tables as built (pages 0, 1, ..,
//!   place = the whole list), is the same rule.
//!
//! The canonical form: per binding and per token, in order, the MLA latent row (kvfmt.h: 656 B in FP8, 416 B in INT4)
//! and, on full-index layers, the indexer key row (128 E4M3 codes, then its FP32 scale: 132 B). Those rows are the
//! same bytes on every head and every rank, so a layout's pages and the canonical rows convert both ways without
//! arithmetic. A row range [first, first + rows) lets a live migration send the tail written since a bulk copy.
//!
//! State that splits by head (GQA KV heads such as the DFlash2 drafter's ring, linear-attention state) is kept per
//! head: `heads_out` gathers every rank's heads into head order, `heads_in` hands a rank its own, with the engine's
//! width rule (g6nr.h: n / nr each, the first n % nr ranks one more).
//!
//! What this module does not decide: whether a stored block may leave its layout or engine build (the persisted
//! store's compatibility line, engine/runtime/g6m4.cu, is unchanged). Another layout computes slightly different KV
//! bits from the same tokens, so that is for people (GRAFT-PLAN Rule 2).

use std::fmt;

/// Rows per page.
pub const PAGE: usize = 64;
/// The indexer key row: 128 E4M3 codes, then one FP32 scale (kvfmt.h g6kv::IDX_ROW).
pub const IDX_ROW: usize = 132;
const IDX_CODES: usize = 128;
/// A page table entry for a page another rank holds (attn.h G6_CP_REMOTE).
pub const REMOTE: i64 = -2;
/// A page table entry past the run's list.
pub const NONE: i64 = -1;

const MAGIC: &[u8; 8] = b"g6canon1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// a layout, list or argument the rules above do not allow
    Bad(&'static str),
    /// a logical page no rank holds
    Missing { binding: usize, page: usize },
    /// two ranks hold different bytes for a replicated page
    Disagree { binding: usize, page: usize, rank: usize },
    /// the canonical form was made for other bindings or another row format
    Mismatch(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bad(w) => write!(f, "canonical state: {w}"),
            Error::Missing { binding, page } => {
                write!(f, "canonical state: binding {binding} page {page} held by no rank")
            }
            Error::Disagree { binding, page, rank } => {
                write!(f, "canonical state: binding {binding} page {page} differs on rank {rank} from rank 0")
            }
            Error::Mismatch(w) => write!(f, "canonical state from another engine: {w}"),
        }
    }
}

impl std::error::Error for Error {}

/// One layer binding, in the order every rank shares (main model, then draft): whether it has indexer keys (a
/// full-index layer) and whether its cache is sharded past R (cp_nr > 0 in its G6Attn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub index: bool,
    pub cp: bool,
}

/// A layout's KV placement: `nr` CP ranks (0: every page on every rank) and R in rows (pages below R/64 replicated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub nr: u32,
    pub below_rows: u32,
}

impl Layout {
    fn check(&self) -> Result<(), Error> {
        if self.nr == 1 || self.nr > 64 {
            return Err(Error::Bad("CP ranks must be 0 or 2..64"));
        }
        if !(self.below_rows as usize).is_multiple_of(PAGE) {
            return Err(Error::Bad("R must be a multiple of 64 rows"));
        }
        Ok(())
    }
    fn rp(&self) -> usize {
        self.below_rows as usize / PAGE
    }
}

/// A rank's view of one run: its page list (the pool's, the same on every rank) and `place`, the bound s2_solo_cp_pages
/// passes to s2_list_place for a CP binding: the main model's pool pages (S2Model::pool_pages). The Pool's place pages
/// (g6m4.cu: the fewer of main and draft) equal it when the draft pool is not the smaller. u32::MAX: the whole list
/// (the tables as built).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub list: Vec<u32>,
    pub place: u32,
}

/// The rank's page table for `logical` pages of one binding: the physical page, REMOTE or NONE.
pub fn table(b: Binding, layout: &Layout, view: &View, rank: u32, logical: usize) -> Result<Vec<i64>, Error> {
    layout.check()?;
    if layout.nr > 0 && rank >= layout.nr {
        return Err(Error::Bad("rank outside the layout"));
    }
    let mut t = vec![NONE; logical];
    if !b.cp || layout.nr == 0 {
        for (lp, e) in t.iter_mut().enumerate() {
            if let Some(&p) = view.list.get(lp) {
                *e = p as i64;
            }
        }
        return Ok(t);
    }
    let low: Vec<u32> = view.list.iter().copied().filter(|&p| p < view.place).collect();
    let (rp, nr) = (layout.rp(), layout.nr as usize);
    let mut j = 0;
    for (lp, e) in t.iter_mut().enumerate() {
        *e = if lp < rp {
            view.list.get(lp).map_or(NONE, |&p| p as i64)
        } else if lp % nr == rank as usize {
            let at = rp + j;
            j += 1;
            low.get(at).map_or(NONE, |&p| p as i64)
        } else {
            REMOTE
        };
    }
    Ok(t)
}

/// One binding's canonical rows: `mla` rows x the MLA row, and `idx` rows x IDX_ROW on full-index layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rows {
    pub mla: Vec<u8>,
    pub idx: Option<Vec<u8>>,
}

/// A session's KV in canonical form: rows [first, first + rows) of every binding. `index` says which bindings carry
/// indexer rows; whether a binding is sharded is the layout's business, so it is not part of the form and an import
/// takes it from the destination's bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canon {
    pub mla_row: usize,
    pub index: Vec<bool>,
    pub first: u64,
    pub rows: u64,
    pub data: Vec<Rows>,
}

/// One rank's caches of one binding, page-major as bind.cu allocates them: `mla` pages x 64 x the MLA row, `idx`
/// pages x (64 x 128 codes, then 64 FP32 scales).
pub struct Cache<'a> {
    pub mla: &'a [u8],
    pub idx: Option<&'a [u8]>,
}

pub struct CacheMut<'a> {
    pub mla: &'a mut [u8],
    pub idx: Option<&'a mut [u8]>,
}

fn idx_get(page: &[u8], r: usize, out: &mut [u8]) {
    out[..IDX_CODES].copy_from_slice(&page[r * IDX_CODES..(r + 1) * IDX_CODES]);
    let s = PAGE * IDX_CODES + r * 4;
    out[IDX_CODES..IDX_ROW].copy_from_slice(&page[s..s + 4]);
}

fn idx_put(page: &mut [u8], r: usize, row: &[u8]) {
    page[r * IDX_CODES..(r + 1) * IDX_CODES].copy_from_slice(&row[..IDX_CODES]);
    let s = PAGE * IDX_CODES + r * 4;
    page[s..s + 4].copy_from_slice(&row[IDX_CODES..IDX_ROW]);
}

fn pages_of(first: u64, rows: u64) -> std::ops::Range<usize> {
    let a = first as usize / PAGE;
    let b = (first + rows) as usize;
    a..b.div_ceil(PAGE)
}

/// Exports rows [first, first + rows) of a run from every rank of a layout. caches[rank][binding]. A page on several
/// ranks (replicated) comes from the lowest rank, and with `check` every other rank must hold the same rows.
#[allow(clippy::too_many_arguments)]
pub fn export(
    mla_row: usize,
    bindings: &[Binding],
    layout: &Layout,
    views: &[View],
    caches: &[Vec<Cache<'_>>],
    first: u64,
    rows: u64,
    check: bool,
) -> Result<Canon, Error> {
    layout.check()?;
    let ranks = views.len();
    if ranks == 0 || caches.len() != ranks || (layout.nr > 0 && ranks != layout.nr as usize) {
        return Err(Error::Bad("one view and one set of caches per rank"));
    }
    if mla_row == 0 || !mla_row.is_multiple_of(16) {
        return Err(Error::Bad("MLA row bytes"));
    }
    // the rows must lie within the pages the run lists: that bounds every size below
    let end = first.checked_add(rows).ok_or(Error::Bad("row range"))?;
    if end.div_ceil(PAGE as u64) > views.iter().map(|v| v.list.len()).max().unwrap_or(0) as u64 {
        return Err(Error::Bad("rows past the run's list"));
    }
    let span = pages_of(first, rows);
    let mut data = Vec::with_capacity(bindings.len());
    for (bi, &b) in bindings.iter().enumerate() {
        let mut out =
            Rows { mla: vec![0; rows as usize * mla_row], idx: b.index.then(|| vec![0; rows as usize * IDX_ROW]) };
        let mut tabs = Vec::with_capacity(ranks);
        for (r, v) in views.iter().enumerate() {
            let c = caches[r].get(bi).ok_or(Error::Bad("a cache per binding"))?;
            if c.idx.is_some() != b.index {
                return Err(Error::Bad("indexer cache on a binding without index keys, or none on one with them"));
            }
            tabs.push(table(b, layout, v, r as u32, span.end)?);
        }
        let mut tmp = [0u8; IDX_ROW];
        for lp in span.clone() {
            let holders: Vec<(usize, usize)> =
                (0..ranks).filter_map(|r| (tabs[r][lp] >= 0).then_some((r, tabs[r][lp] as usize))).collect();
            let Some(&(r0, p0)) = holders.first() else { return Err(Error::Missing { binding: bi, page: lp }) };
            let lo = (lp * PAGE).max(first as usize);
            let hi = ((lp + 1) * PAGE).min((first + rows) as usize);
            let c = &caches[r0][bi];
            let mpage =
                c.mla.get(p0 * PAGE * mla_row..(p0 + 1) * PAGE * mla_row).ok_or(Error::Bad("page past a cache"))?;
            let ib = PAGE * IDX_ROW;
            let ipage = match c.idx {
                Some(ix) => Some(ix.get(p0 * ib..(p0 + 1) * ib).ok_or(Error::Bad("page past an indexer cache"))?),
                None => None,
            };
            for pos in lo..hi {
                let (r, o) = (pos % PAGE, pos - first as usize);
                out.mla[o * mla_row..(o + 1) * mla_row].copy_from_slice(&mpage[r * mla_row..(r + 1) * mla_row]);
                if let (Some(ip), Some(oi)) = (ipage, out.idx.as_mut()) {
                    idx_get(ip, r, &mut tmp);
                    oi[o * IDX_ROW..(o + 1) * IDX_ROW].copy_from_slice(&tmp);
                }
            }
            if check {
                for &(r, p) in &holders[1..] {
                    let c = &caches[r][bi];
                    let m2 = c
                        .mla
                        .get(p * PAGE * mla_row..(p + 1) * PAGE * mla_row)
                        .ok_or(Error::Bad("page past a cache"))?;
                    let i2 = match c.idx {
                        Some(ix) => Some(ix.get(p * ib..(p + 1) * ib).ok_or(Error::Bad("page past an indexer cache"))?),
                        None => None,
                    };
                    for pos in lo..hi {
                        let rr = pos % PAGE;
                        let same_m = m2[rr * mla_row..(rr + 1) * mla_row] == mpage[rr * mla_row..(rr + 1) * mla_row];
                        let same_i = match (i2, ipage) {
                            (Some(a), Some(b)) => {
                                let mut x = [0u8; IDX_ROW];
                                idx_get(a, rr, &mut x);
                                idx_get(b, rr, &mut tmp);
                                x == tmp
                            }
                            _ => true,
                        };
                        if !same_m || !same_i {
                            return Err(Error::Disagree { binding: bi, page: lp, rank: r });
                        }
                    }
                }
            }
        }
        data.push(out);
    }
    Ok(Canon { mla_row, index: bindings.iter().map(|b| b.index).collect(), first, rows, data })
}

/// Writes a canonical form's rows into one rank's caches at a layout (caches[binding]). Only the rows the canonical
/// form holds are written, and only on the pages this rank holds: the rest of each page stays as it was (the engine
/// zeroes a run's pages at its grant, s2_reset_pages). Returns the rows written per binding, summed.
pub fn import(
    canon: &Canon,
    bindings: &[Binding],
    mla_row: usize,
    layout: &Layout,
    view: &View,
    rank: u32,
    caches: &mut [CacheMut<'_>],
) -> Result<u64, Error> {
    if canon.mla_row != mla_row {
        return Err(Error::Mismatch("MLA row format"));
    }
    if canon.index.len() != bindings.len() || canon.index.iter().zip(bindings).any(|(&i, b)| i != b.index) {
        return Err(Error::Mismatch("layer bindings"));
    }
    canon.check_shape()?;
    if caches.len() != bindings.len() {
        return Err(Error::Bad("a cache per binding"));
    }
    let end = canon.first.checked_add(canon.rows).ok_or(Error::Bad("row range"))?;
    if end.div_ceil(PAGE as u64) > view.list.len() as u64 {
        return Err(Error::Bad("the run's list is shorter than the canonical rows"));
    }
    if canon.rows == 0 {
        return Ok(0);
    }
    let span = pages_of(canon.first, canon.rows);
    let mut written = 0;
    for (bi, &b) in bindings.iter().enumerate() {
        if caches[bi].idx.is_some() != b.index {
            return Err(Error::Bad("indexer cache on a binding without index keys, or none on one with them"));
        }
        let t = table(b, layout, view, rank, span.end)?;
        let src = &canon.data[bi];
        for lp in span.clone() {
            match t[lp] {
                REMOTE => continue,
                NONE => {
                    return Err(Error::Bad("no page for a row this rank holds (the list is too short below place)"))
                }
                _ => {}
            }
            let p = t[lp] as usize;
            let lo = (lp * PAGE).max(canon.first as usize);
            let hi = ((lp + 1) * PAGE).min((canon.first + canon.rows) as usize);
            let c = &mut caches[bi];
            let mpage =
                c.mla.get_mut(p * PAGE * mla_row..(p + 1) * PAGE * mla_row).ok_or(Error::Bad("page past a cache"))?;
            for pos in lo..hi {
                let (r, o) = (pos % PAGE, pos - canon.first as usize);
                mpage[r * mla_row..(r + 1) * mla_row].copy_from_slice(&src.mla[o * mla_row..(o + 1) * mla_row]);
            }
            if let (Some(ix), Some(si)) = (c.idx.as_deref_mut(), src.idx.as_ref()) {
                let ib = PAGE * IDX_ROW;
                let ipage = ix.get_mut(p * ib..(p + 1) * ib).ok_or(Error::Bad("page past an indexer cache"))?;
                for pos in lo..hi {
                    let (r, o) = (pos % PAGE, pos - canon.first as usize);
                    idx_put(ipage, r, &si[o * IDX_ROW..(o + 1) * IDX_ROW]);
                }
            }
            written += (hi - lo) as u64;
        }
    }
    Ok(written)
}

impl Canon {
    /// The canonical form as bytes: a header (magic, MLA row, rows, bindings and their flags), then each binding's
    /// MLA rows and indexer rows. Little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.byte_len());
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&(self.mla_row as u32).to_le_bytes());
        v.extend_from_slice(&(self.index.len() as u32).to_le_bytes());
        v.extend_from_slice(&self.first.to_le_bytes());
        v.extend_from_slice(&self.rows.to_le_bytes());
        for &i in &self.index {
            v.push(i as u8);
        }
        for d in &self.data {
            v.extend_from_slice(&d.mla);
            if let Some(i) = &d.idx {
                v.extend_from_slice(i);
            }
        }
        v
    }

    pub fn byte_len(&self) -> usize {
        MAGIC.len() + 24 + self.index.len() + self.rows as usize * self.row_bytes()
    }

    // bytes of one row over every binding
    fn row_bytes(&self) -> usize {
        self.index.iter().map(|&i| self.mla_row + if i { IDX_ROW } else { 0 }).sum()
    }

    // the data matches the header (a Canon built by hand through the Rust API may not)
    fn check_shape(&self) -> Result<(), Error> {
        let r = usize::try_from(self.rows).map_err(|_| Error::Mismatch("rows"))?;
        let ok = self.data.len() == self.index.len()
            && self.data.iter().zip(&self.index).all(|(d, &i)| {
                Some(d.mla.len()) == r.checked_mul(self.mla_row)
                    && d.idx.as_ref().map(|x| x.len()) == if i { r.checked_mul(IDX_ROW) } else { None }
            });
        if ok {
            Ok(())
        } else {
            Err(Error::Mismatch("rows of another shape than the header"))
        }
    }

    pub fn from_bytes(v: &[u8]) -> Result<Canon, Error> {
        let bad = Error::Bad("not a canonical state blob, or cut short");
        if v.len() < MAGIC.len() + 24 || &v[..8] != MAGIC {
            return Err(bad);
        }
        let u32_at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(v[o..o + 8].try_into().unwrap());
        let (mla_row, nb, first, rows) = (u32_at(8) as usize, u32_at(12) as usize, u64_at(16), u64_at(24));
        let mut at = 32;
        if mla_row == 0 || nb > v.len() - at || first.checked_add(rows).is_none() {
            return Err(bad);
        }
        if v[at..at + nb].iter().any(|&f| f > 1) {
            return Err(bad);
        }
        let index: Vec<bool> = v[at..at + nb].iter().map(|&f| f == 1).collect();
        at += nb;
        // every size from the header is checked against what is left before anything is sliced or allocated
        let per: usize = index.iter().map(|&i| mla_row + if i { IDX_ROW } else { 0 }).sum();
        let left = (v.len() - at) as u64;
        if per == 0 || rows.checked_mul(per as u64) != Some(left) {
            return Err(bad);
        }
        let r = rows as usize;
        let mut data = Vec::with_capacity(nb);
        for &i in &index {
            let (m, x) = (r * mla_row, if i { r * IDX_ROW } else { 0 });
            data.push(Rows { mla: v[at..at + m].to_vec(), idx: i.then(|| v[at + m..at + m + x].to_vec()) });
            at += m + x;
        }
        Ok(Canon { mla_row, index, first, rows, data })
    }
}

/// The engine's width rule (g6nr.h): rank j of nr holds n / nr units, the first n % nr ranks one more, in order.
pub fn width(n: usize, nr: usize, j: usize) -> usize {
    n / nr + usize::from(j < n % nr)
}

pub fn start(n: usize, nr: usize, j: usize) -> usize {
    j * (n / nr) + j.min(n % nr)
}

/// Gathers state split by head: parts[j] is head group j's buffer laid out [outer][its heads][inner bytes], group j
/// holding heads start(heads, nr, j) .. + width(heads, nr, j). Returns the canonical [outer][heads][inner]. The caller
/// maps ranks to groups: the MLA attention heads rotate by layer ((rank - layer) mod NR, bind.cu), the drafter's KV
/// heads do not.
pub fn heads_out(ranks: &[&[u8]], heads: usize, outer: usize, inner: usize) -> Result<Vec<u8>, Error> {
    let nr = ranks.len();
    if nr == 0 || nr > heads {
        return Err(Error::Bad("1..heads ranks"));
    }
    let mut out = vec![0u8; outer * heads * inner];
    for (j, buf) in ranks.iter().enumerate() {
        let (h0, w) = (start(heads, nr, j), width(heads, nr, j));
        if buf.len() != outer * w * inner {
            return Err(Error::Bad("a rank's head buffer has the wrong size"));
        }
        for o in 0..outer {
            let dst = (o * heads + h0) * inner;
            out[dst..dst + w * inner].copy_from_slice(&buf[o * w * inner..(o + 1) * w * inner]);
        }
    }
    Ok(out)
}

/// Head group j of nr's heads of a canonical [outer][heads][inner] buffer, laid out [outer][its heads][inner].
pub fn heads_in(canon: &[u8], heads: usize, outer: usize, inner: usize, nr: usize, j: usize) -> Result<Vec<u8>, Error> {
    if nr == 0 || nr > heads || j >= nr || canon.len() != outer * heads * inner {
        return Err(Error::Bad("head split arguments"));
    }
    let (h0, w) = (start(heads, nr, j), width(heads, nr, j));
    let mut out = Vec::with_capacity(outer * w * inner);
    for o in 0..outer {
        let src = (o * heads + h0) * inner;
        out.extend_from_slice(&canon[src..src + w * inner]);
    }
    Ok(out)
}

// ---- Draft (DFlash2 GQA ring) and linear-attention (KDA + conv) canonical state (#161) ----

const DFL_MAGIC: &[u8; 8] = b"g6cdfl1\0";
const KDA_MAGIC: &[u8; 8] = b"g6ckda1\0";
const SES_MAGIC: &[u8; 8] = b"g6cses1\0";

/// Geometry of a DFlash2 GQA drafter ring (engine/runtime/dflash.cuh: [L][NKR][RING][HD] fp16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflSpec {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub elem_bytes: u32,
}

impl DflSpec {
    fn row_bytes(&self) -> Result<usize, Error> {
        if self.layers == 0 || self.kv_heads == 0 || self.head_dim == 0 || self.elem_bytes == 0 {
            return Err(Error::Bad("DFlash2 geometry must be nonzero"));
        }
        (self.head_dim as usize)
            .checked_mul(self.elem_bytes as usize)
            .ok_or(Error::Bad("DFlash2 row size overflow"))
    }
}

/// One layout's DFlash2 ring distribution: either split across `nr` ranks (`kv_heads % nr == 0`, no layer rotation) or
/// replicated on every rank (`replicated == true`, e.g. TP=3 and TP=6 where `dfl::REP` holds all `kv_heads`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflLayout {
    pub nr: u32,
    pub ring: u32,
    pub replicated: bool,
}

impl DflLayout {
    fn nkr(&self, spec: &DflSpec) -> Result<usize, Error> {
        let _ = spec.row_bytes()?;
        if self.nr == 0 || self.nr > 64 || self.ring == 0 {
            return Err(Error::Bad("DFlash2 layout nr (1..=64) or ring (nonzero)"));
        }
        if self.replicated {
            Ok(spec.kv_heads as usize)
        } else if !spec.kv_heads.is_multiple_of(self.nr) {
            Err(Error::Bad("split DFlash2 layout requires kv_heads divisible by nr"))
        } else {
            Ok((spec.kv_heads / self.nr) as usize)
        }
    }

    fn rank_half_bytes(&self, spec: &DflSpec) -> Result<usize, Error> {
        let row = spec.row_bytes()?;
        let nkr = self.nkr(spec)?;
        (spec.layers as usize)
            .checked_mul(nkr)
            .and_then(|x| x.checked_mul(self.ring as usize))
            .and_then(|x| x.checked_mul(row))
            .ok_or(Error::Bad("DFlash2 ring buffer size overflow"))
    }
}

/// One rank's host copy of Dfl::kring and Dfl::vring: each [layers][nkr][ring][head_dim * elem_bytes].
pub struct DflRing<'a> {
    pub k: &'a [u8],
    pub v: &'a [u8],
}

pub struct DflRingMut<'a> {
    pub k: &'a mut [u8],
    pub v: &'a mut [u8],
}

/// Canonical DFlash2 GQA drafter ring for logical positions [lo, hi): k and v each [layers][kv_heads][rows][row_bytes],
/// independent of ring length and TP width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DflCanon {
    pub spec: DflSpec,
    pub lo: u64,
    pub hi: u64,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
}

impl DflCanon {
    pub fn rows(&self) -> Result<u64, Error> {
        self.hi.checked_sub(self.lo).ok_or(Error::Bad("DFlash2 hi < lo"))
    }

    fn half_bytes_for(spec: &DflSpec, rows: u64) -> Result<usize, Error> {
        let row = spec.row_bytes()?;
        let r = usize::try_from(rows).map_err(|_| Error::Bad("DFlash2 rows overflow"))?;
        (spec.layers as usize)
            .checked_mul(spec.kv_heads as usize)
            .and_then(|x| x.checked_mul(r))
            .and_then(|x| x.checked_mul(row))
            .ok_or(Error::Bad("DFlash2 canonical size overflow"))
    }

    fn check_shape(&self) -> Result<usize, Error> {
        let rows = self.rows()?;
        let half = Self::half_bytes_for(&self.spec, rows)?;
        if self.k.len() != half || self.v.len() != half {
            return Err(Error::Mismatch("DFlash2 data of another shape than the header"));
        }
        Ok(rows as usize)
    }

    pub fn byte_len(&self) -> usize {
        DFL_MAGIC.len() + 32 + self.k.len() + self.v.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(DFL_MAGIC);
        out.extend_from_slice(&self.spec.layers.to_le_bytes());
        out.extend_from_slice(&self.spec.kv_heads.to_le_bytes());
        out.extend_from_slice(&self.spec.head_dim.to_le_bytes());
        out.extend_from_slice(&self.spec.elem_bytes.to_le_bytes());
        out.extend_from_slice(&self.lo.to_le_bytes());
        out.extend_from_slice(&self.hi.to_le_bytes());
        out.extend_from_slice(&self.k);
        out.extend_from_slice(&self.v);
        out
    }

    pub fn from_bytes(v: &[u8]) -> Result<DflCanon, Error> {
        let bad = Error::Bad("not a canonical DFlash2 blob, or cut short");
        if v.len() < DFL_MAGIC.len() + 32 || &v[..8] != DFL_MAGIC {
            return Err(bad);
        }
        let u32_at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(v[o..o + 8].try_into().unwrap());
        let spec = DflSpec {
            layers: u32_at(8),
            kv_heads: u32_at(12),
            head_dim: u32_at(16),
            elem_bytes: u32_at(20),
        };
        let (lo, hi) = (u64_at(24), u64_at(32));
        let Some(rows) = hi.checked_sub(lo) else { return Err(bad) };
        let Ok(half) = Self::half_bytes_for(&spec, rows) else { return Err(bad) };
        let Some(total) = half.checked_mul(2) else { return Err(bad) };
        if v.len() - 40 != total {
            return Err(bad);
        }
        Ok(DflCanon {
            spec,
            lo,
            hi,
            k: v[40..40 + half].to_vec(),
            v: v[40 + half..40 + total].to_vec(),
        })
    }
}

/// Exports logical positions [lo, hi) from every rank's DFlash2 ring into canonical form.
/// With `check == true` on a replicated layout, verifies every rank holds identical live rows.
pub fn dfl_export(
    spec: DflSpec,
    layout: &DflLayout,
    rings: &[DflRing<'_>],
    lo: u64,
    hi: u64,
    check: bool,
) -> Result<DflCanon, Error> {
    let row = spec.row_bytes()?;
    let nkr = layout.nkr(&spec)?;
    let rank_half = layout.rank_half_bytes(&spec)?;
    let rows_u64 = hi.checked_sub(lo).ok_or(Error::Bad("DFlash2 hi < lo"))?;
    if rows_u64 > layout.ring as u64 {
        return Err(Error::Bad("DFlash2 row span exceeds ring capacity"));
    }
    if rings.len() != layout.nr as usize {
        return Err(Error::Bad("one DFlash2 ring per rank"));
    }
    for r in rings {
        if r.k.len() != rank_half || r.v.len() != rank_half {
            return Err(Error::Bad("a rank's DFlash2 ring has the wrong size"));
        }
    }
    let rows = rows_u64 as usize;
    let half = DflCanon::half_bytes_for(&spec, rows_u64)?;
    let mut k_out = vec![0u8; half];
    let mut v_out = vec![0u8; half];
    let (nl, nkv, ring) = (spec.layers as usize, spec.kv_heads as usize, layout.ring as usize);

    for l in 0..nl {
        if layout.replicated {
            for h in 0..nkv {
                let src_base = (l * nkr + h) * ring;
                let dst_base = (l * nkv + h) * rows;
                for o in 0..rows {
                    let pos = lo + o as u64;
                    let slot = (pos % layout.ring as u64) as usize;
                    let s = (src_base + slot) * row;
                    let d = (dst_base + o) * row;
                    k_out[d..d + row].copy_from_slice(&rings[0].k[s..s + row]);
                    v_out[d..d + row].copy_from_slice(&rings[0].v[s..s + row]);
                    if check {
                        for (r, rr) in rings.iter().enumerate().skip(1) {
                            if rr.k[s..s + row] != rings[0].k[s..s + row]
                                || rr.v[s..s + row] != rings[0].v[s..s + row]
                            {
                                return Err(Error::Disagree { binding: l, page: pos as usize, rank: r });
                            }
                        }
                    }
                }
            }
        } else {
            for (r, rr) in rings.iter().enumerate() {
                for hl in 0..nkr {
                    let h = r * nkr + hl;
                    let src_base = (l * nkr + hl) * ring;
                    let dst_base = (l * nkv + h) * rows;
                    for o in 0..rows {
                        let slot = ((lo + o as u64) % layout.ring as u64) as usize;
                        let s = (src_base + slot) * row;
                        let d = (dst_base + o) * row;
                        k_out[d..d + row].copy_from_slice(&rr.k[s..s + row]);
                        v_out[d..d + row].copy_from_slice(&rr.v[s..s + row]);
                    }
                }
            }
        }
    }
    Ok(DflCanon { spec, lo, hi, k: k_out, v: v_out })
}

/// Imports canonical DFlash2 ring rows [lo, hi) into one rank's ring at `layout`. Only ring slots in [lo, hi) mod ring
/// are written, so bulk + tail imports compose cleanly. Returns the number of logical positions written (`hi - lo`).
pub fn dfl_import(
    canon: &DflCanon,
    spec: DflSpec,
    layout: &DflLayout,
    rank: u32,
    dst: &mut DflRingMut<'_>,
) -> Result<u64, Error> {
    if canon.spec != spec {
        return Err(Error::Mismatch("DFlash2 geometry"));
    }
    let rows = canon.check_shape()?;
    let row = spec.row_bytes()?;
    let nkr = layout.nkr(&spec)?;
    let rank_half = layout.rank_half_bytes(&spec)?;
    if rank >= layout.nr {
        return Err(Error::Bad("rank outside the DFlash2 layout"));
    }
    if rows as u64 > layout.ring as u64 {
        return Err(Error::Bad("DFlash2 canonical span exceeds target ring capacity"));
    }
    if dst.k.len() != rank_half || dst.v.len() != rank_half {
        return Err(Error::Bad("destination DFlash2 ring has the wrong size"));
    }
    let (nl, nkv, ring) = (spec.layers as usize, spec.kv_heads as usize, layout.ring as usize);
    let h_base = if layout.replicated { 0 } else { rank as usize * nkr };

    for l in 0..nl {
        for hl in 0..nkr {
            let h = h_base + hl;
            let src_base = (l * nkv + h) * rows;
            let dst_base = (l * nkr + hl) * ring;
            for o in 0..rows {
                let slot = ((canon.lo + o as u64) % layout.ring as u64) as usize;
                let s = (src_base + o) * row;
                let d = (dst_base + slot) * row;
                dst.k[d..d + row].copy_from_slice(&canon.k[s..s + row]);
                dst.v[d..d + row].copy_from_slice(&canon.v[s..s + row]);
            }
        }
    }
    Ok(rows as u64)
}

/// Geometry of KDA linear-attention + short-convolution state (engine/kda/math.h, s2/flash_ckpt_rewidth.h).
/// Per KDA slot `li` (model layer `layer_ids[li]`):
/// - rank layout: `S [nhmax][dk][dv]` then `C [nhist * n_streams][nhmax][dk]` (element size `elem_bytes`),
///   active heads `0..nh`, padding slots `nh..nhmax` zeroed.
/// - canonical layout: `S [heads][dk][dv]` then `C [nhist * n_streams][heads][dk]`, head order `0..heads`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdaSpec {
    pub heads: u32,
    pub dk: u32,
    pub dv: u32,
    pub nhist: u32,
    pub n_streams: u32,
    pub elem_bytes: u32,
    pub layer_ids: Vec<u32>,
}

/// Rank `rank` of `nr`'s head range `(h0, nh)` at model layer `layer` (matches kda::heads in engine/kda/math.h).
pub fn kda_rank_heads(heads: usize, nr: usize, layer: usize, rank: usize) -> (usize, usize) {
    let j = (rank + nr - (layer % nr)) % nr;
    (start(heads, nr, j), width(heads, nr, j))
}

/// Maximum heads held by any rank at width `nr` (matches kda::nhmax in engine/kda/math.h).
pub fn kda_nhmax(heads: usize, nr: usize) -> usize {
    heads.div_ceil(nr)
}

impl KdaSpec {
    fn check(&self) -> Result<(usize, usize, usize), Error> {
        if self.heads == 0
            || self.dk == 0
            || self.dv == 0
            || self.nhist == 0
            || self.n_streams == 0
            || self.elem_bytes == 0
            || self.layer_ids.is_empty()
        {
            return Err(Error::Bad("KDA geometry must be nonzero"));
        }
        let s_head = (self.dk as usize)
            .checked_mul(self.dv as usize)
            .and_then(|x| x.checked_mul(self.elem_bytes as usize))
            .ok_or(Error::Bad("KDA S head size overflow"))?;
        let c_head = (self.dk as usize)
            .checked_mul(self.elem_bytes as usize)
            .ok_or(Error::Bad("KDA conv head size overflow"))?;
        let c_rows = (self.nhist as usize)
            .checked_mul(self.n_streams as usize)
            .ok_or(Error::Bad("KDA conv rows overflow"))?;
        Ok((s_head, c_head, c_rows))
    }

    /// Bytes of one rank's KDA buffer across all slots in `layer_ids` at width `nr`.
    pub fn rank_bytes(&self, nr: u32) -> Result<usize, Error> {
        let (s_head, c_head, c_rows) = self.check()?;
        let heads = self.heads as usize;
        if nr == 0 || nr as usize > heads {
            return Err(Error::Bad("KDA nr must be in 1..=heads"));
        }
        let nhm = kda_nhmax(heads, nr as usize);
        let s_layer = nhm.checked_mul(s_head).ok_or(Error::Bad("KDA rank S size overflow"))?;
        let c_layer = c_rows
            .checked_mul(nhm)
            .and_then(|x| x.checked_mul(c_head))
            .ok_or(Error::Bad("KDA rank conv size overflow"))?;
        let per_layer = s_layer.checked_add(c_layer).ok_or(Error::Bad("KDA rank layer size overflow"))?;
        self.layer_ids
            .len()
            .checked_mul(per_layer)
            .ok_or(Error::Bad("KDA rank total size overflow"))
    }

    /// Bytes of canonical KDA state across all slots in `layer_ids`.
    pub fn canon_bytes(&self) -> Result<usize, Error> {
        self.rank_bytes(1)
    }
}

/// Canonical KDA linear-attention + short-convolution state across all KDA layers, independent of TP width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdaCanon {
    pub spec: KdaSpec,
    pub data: Vec<u8>,
}

impl KdaCanon {
    fn check_shape(&self) -> Result<(), Error> {
        let want = self.spec.canon_bytes()?;
        if self.data.len() != want {
            return Err(Error::Mismatch("KDA data of another shape than the header"));
        }
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        KDA_MAGIC.len() + 28 + self.spec.layer_ids.len() * 4 + self.data.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(KDA_MAGIC);
        out.extend_from_slice(&self.spec.heads.to_le_bytes());
        out.extend_from_slice(&self.spec.dk.to_le_bytes());
        out.extend_from_slice(&self.spec.dv.to_le_bytes());
        out.extend_from_slice(&self.spec.nhist.to_le_bytes());
        out.extend_from_slice(&self.spec.n_streams.to_le_bytes());
        out.extend_from_slice(&self.spec.elem_bytes.to_le_bytes());
        out.extend_from_slice(&(self.spec.layer_ids.len() as u32).to_le_bytes());
        for &l in &self.spec.layer_ids {
            out.extend_from_slice(&l.to_le_bytes());
        }
        out.extend_from_slice(&self.data);
        out
    }

    pub fn from_bytes(v: &[u8]) -> Result<KdaCanon, Error> {
        let bad = Error::Bad("not a canonical KDA blob, or cut short");
        if v.len() < KDA_MAGIC.len() + 28 || &v[..8] != KDA_MAGIC {
            return Err(bad);
        }
        let u32_at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        let heads = u32_at(8);
        let dk = u32_at(12);
        let dv = u32_at(16);
        let nhist = u32_at(20);
        let n_streams = u32_at(24);
        let elem_bytes = u32_at(28);
        let n_layers = u32_at(32) as usize;
        let Some(ids_bytes) = n_layers.checked_mul(4) else { return Err(bad) };
        let at = 36;
        if ids_bytes > v.len() - at {
            return Err(bad);
        }
        let mut layer_ids = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layer_ids.push(u32_at(at + i * 4));
        }
        let spec = KdaSpec { heads, dk, dv, nhist, n_streams, elem_bytes, layer_ids };
        let Ok(need) = spec.canon_bytes() else { return Err(bad) };
        let data_at = at + ids_bytes;
        if v.len() - data_at != need {
            return Err(bad);
        }
        Ok(KdaCanon { spec, data: v[data_at..].to_vec() })
    }
}

/// Gathers KDA linear-attention + short-conv state from `nr` ranks into canonical head order.
pub fn kda_export(spec: &KdaSpec, nr: u32, ranks: &[&[u8]]) -> Result<KdaCanon, Error> {
    let (s_head, c_head, c_rows) = spec.check()?;
    let rank_need = spec.rank_bytes(nr)?;
    let canon_need = spec.canon_bytes()?;
    if ranks.len() != nr as usize {
        return Err(Error::Bad("one KDA buffer per rank"));
    }
    for buf in ranks {
        if buf.len() != rank_need {
            return Err(Error::Bad("a rank's KDA buffer has the wrong size"));
        }
    }
    let heads = spec.heads as usize;
    let nhm = kda_nhmax(heads, nr as usize);
    let r_s_layer = nhm * s_head;
    let r_c_layer = c_rows * nhm * c_head;
    let r_stride = r_s_layer + r_c_layer;
    let c_s_layer = heads * s_head;
    let c_c_layer = c_rows * heads * c_head;
    let c_stride = c_s_layer + c_c_layer;

    let mut data = vec![0u8; canon_need];
    for (li, &layer) in spec.layer_ids.iter().enumerate() {
        let dst_s = li * c_stride;
        let dst_c = dst_s + c_s_layer;
        for (r, buf) in ranks.iter().enumerate() {
            let (h0, nh) = kda_rank_heads(heads, nr as usize, layer as usize, r);
            let src_s = li * r_stride;
            let src_c = src_s + r_s_layer;
            data[dst_s + h0 * s_head..dst_s + (h0 + nh) * s_head]
                .copy_from_slice(&buf[src_s..src_s + nh * s_head]);
            for k in 0..c_rows {
                let s_off = src_c + k * nhm * c_head;
                let d_off = dst_c + (k * heads + h0) * c_head;
                data[d_off..d_off + nh * c_head].copy_from_slice(&buf[s_off..s_off + nh * c_head]);
            }
        }
    }
    Ok(KdaCanon { spec: spec.clone(), data })
}

/// Scatters canonical KDA state into rank `rank` of `nr`'s buffer, zeroing any padding heads `nh..nhmax`.
pub fn kda_import(canon: &KdaCanon, spec: &KdaSpec, nr: u32, rank: u32, dst: &mut [u8]) -> Result<usize, Error> {
    if canon.spec != *spec {
        return Err(Error::Mismatch("KDA geometry"));
    }
    canon.check_shape()?;
    let (s_head, c_head, c_rows) = spec.check()?;
    let rank_need = spec.rank_bytes(nr)?;
    if rank >= nr {
        return Err(Error::Bad("rank outside the KDA layout"));
    }
    if dst.len() != rank_need {
        return Err(Error::Bad("destination KDA buffer has the wrong size"));
    }
    dst.fill(0);
    let heads = spec.heads as usize;
    let nhm = kda_nhmax(heads, nr as usize);
    let r_s_layer = nhm * s_head;
    let r_c_layer = c_rows * nhm * c_head;
    let r_stride = r_s_layer + r_c_layer;
    let c_s_layer = heads * s_head;
    let c_c_layer = c_rows * heads * c_head;
    let c_stride = c_s_layer + c_c_layer;

    for (li, &layer) in spec.layer_ids.iter().enumerate() {
        let (h0, nh) = kda_rank_heads(heads, nr as usize, layer as usize, rank as usize);
        let src_s = li * c_stride;
        let src_c = src_s + c_s_layer;
        let dst_s = li * r_stride;
        let dst_c = dst_s + r_s_layer;
        dst[dst_s..dst_s + nh * s_head]
            .copy_from_slice(&canon.data[src_s + h0 * s_head..src_s + (h0 + nh) * s_head]);
        for k in 0..c_rows {
            let s_off = src_c + (k * heads + h0) * c_head;
            let d_off = dst_c + k * nhm * c_head;
            dst[d_off..d_off + nh * c_head].copy_from_slice(&canon.data[s_off..s_off + nh * c_head]);
        }
    }
    Ok(rank_need)
}

/// Checks and returns replicated opaque state (e.g. Flash raw indexer rings or MTP metadata) across ranks.
pub fn replicated_export(ranks: &[&[u8]], check: bool) -> Result<Vec<u8>, Error> {
    if ranks.is_empty() {
        return Err(Error::Bad("at least one rank required"));
    }
    let first = ranks[0];
    for (r, &b) in ranks.iter().enumerate().skip(1) {
        if b.len() != first.len() {
            return Err(Error::Bad("replicated state length differs across ranks"));
        }
        if check && b != first {
            return Err(Error::Disagree { binding: 0, page: 0, rank: r });
        }
    }
    Ok(first.to_vec())
}

/// Complete canonical session envelope: KV pages (main + unsharded MTP), optional DFlash2 ring, optional KDA linear
/// attention + conv state, and optional replicated opaque state (e.g. Flash raw index rings / draft scalars).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionCanon {
    pub kv: Option<Canon>,
    pub dfl: Option<DflCanon>,
    pub kda: Option<KdaCanon>,
    pub extra: Vec<u8>,
}

impl SessionCanon {
    pub fn to_bytes(&self) -> Vec<u8> {
        let kv_b = self.kv.as_ref().map(Canon::to_bytes).unwrap_or_default();
        let dfl_b = self.dfl.as_ref().map(DflCanon::to_bytes).unwrap_or_default();
        let kda_b = self.kda.as_ref().map(KdaCanon::to_bytes).unwrap_or_default();
        pack_session_parts(&kv_b, &dfl_b, &kda_b, &self.extra)
    }

    pub fn from_bytes(v: &[u8]) -> Result<SessionCanon, Error> {
        let s = unpack_session_slices(v)?;
        let kv = if s.kv.is_empty() { None } else { Some(Canon::from_bytes(s.kv)?) };
        let dfl = if s.dfl.is_empty() { None } else { Some(DflCanon::from_bytes(s.dfl)?) };
        let kda = if s.kda.is_empty() { None } else { Some(KdaCanon::from_bytes(s.kda)?) };
        Ok(SessionCanon { kv, dfl, kda, extra: s.extra.to_vec() })
    }
}

fn pack_session_parts(kv: &[u8], dfl: &[u8], kda: &[u8], extra: &[u8]) -> Vec<u8> {
    let total = SES_MAGIC.len() + 32 + kv.len() + dfl.len() + kda.len() + extra.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(SES_MAGIC);
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    out.extend_from_slice(&(dfl.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kda.len() as u64).to_le_bytes());
    out.extend_from_slice(&(extra.len() as u64).to_le_bytes());
    out.extend_from_slice(kv);
    out.extend_from_slice(dfl);
    out.extend_from_slice(kda);
    out.extend_from_slice(extra);
    out
}

struct SessionSlices<'a> {
    kv: &'a [u8],
    dfl: &'a [u8],
    kda: &'a [u8],
    extra: &'a [u8],
}

fn unpack_session_slices(v: &[u8]) -> Result<SessionSlices<'_>, Error> {
    let bad = Error::Bad("not a canonical session blob, or cut short");
    if v.len() < SES_MAGIC.len() + 32 || &v[..8] != SES_MAGIC {
        return Err(bad);
    }
    let u64_at = |o: usize| u64::from_le_bytes(v[o..o + 8].try_into().unwrap());
    let lens = [u64_at(8), u64_at(16), u64_at(24), u64_at(32)];
    let mut sum = 0usize;
    let mut parts = [0usize; 4];
    for (i, &l) in lens.iter().enumerate() {
        let Ok(n) = usize::try_from(l) else { return Err(bad) };
        let Some(next) = sum.checked_add(n) else { return Err(bad) };
        sum = next;
        parts[i] = n;
    }
    if v.len() - 40 != sum {
        return Err(bad);
    }
    let mut at = 40;
    let kv = &v[at..at + parts[0]];
    at += parts[0];
    let dfl = &v[at..at + parts[1]];
    at += parts[1];
    let kda = &v[at..at + parts[2]];
    at += parts[2];
    let extra = &v[at..at + parts[3]];
    Ok(SessionSlices { kv, dfl, kda, extra })
}

// ---- C interface (g6kvman.h g6kv_canon_*) ----

use std::cell::RefCell;
use std::ffi::{c_char, CString};

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(e: Error) -> i32 {
    LAST.with(|l| *l.borrow_mut() = CString::new(e.to_string()).unwrap_or_default());
    -1
}

/// The message of this thread's last failed g6kv_canon call.
#[no_mangle]
pub extern "C" fn g6kv_canon_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

unsafe fn c_bindings(nb: usize, flags: *const u8) -> Option<Vec<Binding>> {
    if nb > 0 && flags.is_null() {
        return None;
    }
    Some((0..nb).map(|i| *flags.add(i)).map(|f| Binding { index: f & 1 != 0, cp: f & 2 != 0 }).collect())
}

unsafe fn c_view(list: *const u32, n: usize, place: u32) -> Option<View> {
    if n > 0 && list.is_null() {
        return None;
    }
    Some(View { list: if n == 0 { Vec::new() } else { std::slice::from_raw_parts(list, n).to_vec() }, place })
}

/// Exports rows [first, first + rows) of a run from every rank (`ranks` = nr, or 1 without CP) into `out` (the byte
/// form, Canon::to_bytes). mla[r * nb + b] / mla_bytes: rank r's MLA cache of binding b (host memory), idx the same
/// for indexer caches (NULL entries on bindings without index keys; the arrays themselves are never NULL when nb > 0). flags[b]: 1 = index keys, 2 = CP. Writes the
/// blob's size to *need; with cap < *need nothing else is written and the call returns 1. 0 = done, -1 = error
/// (g6kv_canon_error).
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_export(
    mla_row: u32,
    nb: u32,
    flags: *const u8,
    nr: u32,
    below_rows: u32,
    list: *const u32,
    n_list: usize,
    place: u32,
    mla: *const *const u8,
    mla_bytes: *const usize,
    idx: *const *const u8,
    idx_bytes: *const usize,
    first: u64,
    rows: u64,
    check: i32,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    let (Some(bs), Some(v)) = (c_bindings(nb as usize, flags), c_view(list, n_list, place)) else {
        return fail(Error::Bad("null argument"));
    };
    let ranks = nr.max(1) as usize;
    let nb = nb as usize;
    if need.is_null() || (nb > 0 && (mla.is_null() || mla_bytes.is_null() || idx.is_null() || idx_bytes.is_null())) {
        return fail(Error::Bad("null argument"));
    }
    let mut caches: Vec<Vec<Cache>> = Vec::with_capacity(ranks);
    for r in 0..ranks {
        let mut v = Vec::with_capacity(nb);
        for b in 0..nb {
            let k = r * nb + b;
            let m = *mla.add(k);
            let i = *idx.add(k);
            if m.is_null() {
                return fail(Error::Bad("null MLA cache"));
            }
            v.push(Cache {
                mla: std::slice::from_raw_parts(m, *mla_bytes.add(k)),
                idx: (!i.is_null()).then(|| std::slice::from_raw_parts(i, *idx_bytes.add(k))),
            });
        }
        caches.push(v);
    }
    let views = vec![v; ranks];
    let layout = Layout { nr, below_rows };
    match export(mla_row as usize, &bs, &layout, &views, &caches, first, rows, check != 0) {
        Ok(c) => {
            *need = c.byte_len();
            if cap < *need || out.is_null() {
                return 1;
            }
            let b = c.to_bytes();
            std::ptr::copy_nonoverlapping(b.as_ptr(), out, b.len());
            0
        }
        Err(e) => fail(e),
    }
}

/// Imports a blob from g6kv_canon_export into one rank's caches (mla[b], idx[b], host memory) at a layout. Writes
/// the rows written (summed over bindings) to *written when not NULL. 0, or -1 (g6kv_canon_error).
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_import(
    blob: *const u8,
    n: usize,
    mla_row: u32,
    nb: u32,
    flags: *const u8,
    nr: u32,
    below_rows: u32,
    list: *const u32,
    n_list: usize,
    place: u32,
    rank: u32,
    mla: *const *mut u8,
    mla_bytes: *const usize,
    idx: *const *mut u8,
    idx_bytes: *const usize,
    written: *mut u64,
) -> i32 {
    let (Some(bs), Some(v)) = (c_bindings(nb as usize, flags), c_view(list, n_list, place)) else {
        return fail(Error::Bad("null argument"));
    };
    let nb = nb as usize;
    if blob.is_null() || (nb > 0 && (mla.is_null() || mla_bytes.is_null() || idx.is_null() || idx_bytes.is_null())) {
        return fail(Error::Bad("null argument"));
    }
    let c = match Canon::from_bytes(std::slice::from_raw_parts(blob, n)) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    let mut caches = Vec::with_capacity(nb);
    for b in 0..nb {
        let m = *mla.add(b);
        let i = *idx.add(b);
        if m.is_null() {
            return fail(Error::Bad("null MLA cache"));
        }
        caches.push(CacheMut {
            mla: std::slice::from_raw_parts_mut(m, *mla_bytes.add(b)),
            idx: (!i.is_null()).then(|| std::slice::from_raw_parts_mut(i, *idx_bytes.add(b))),
        });
    }
    match import(&c, &bs, mla_row as usize, &Layout { nr, below_rows }, &v, rank, &mut caches) {
        Ok(w) => {
            if !written.is_null() {
                *written = w;
            }
            0
        }
        Err(e) => fail(e),
    }
}

unsafe fn c_slice<'a>(p: *const u8, n: usize) -> Option<&'a [u8]> {
    if n == 0 {
        Some(&[])
    } else if p.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(p, n))
    }
}

unsafe fn c_slice_mut<'a>(p: *mut u8, n: usize) -> Option<&'a mut [u8]> {
    if n == 0 {
        Some(&mut [])
    } else if p.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(p, n))
    }
}

unsafe fn write_out(bytes: &[u8], out: *mut u8, cap: usize, need: *mut usize) -> i32 {
    if need.is_null() {
        return fail(Error::Bad("null need pointer"));
    }
    *need = bytes.len();
    if cap < bytes.len() || (!bytes.is_empty() && out.is_null()) {
        return 1;
    }
    if !bytes.is_empty() {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    }
    0
}

/// Gathers head-split state [outer][w][inner] from `nr` head groups into canonical [outer][heads][inner].
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_heads_out(
    ranks: *const *const u8,
    rank_bytes: *const usize,
    nr: u32,
    heads: u32,
    outer: usize,
    inner: usize,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    if need.is_null() || nr == 0 || ranks.is_null() || rank_bytes.is_null() {
        return fail(Error::Bad("null or zero argument"));
    }
    let mut parts = Vec::with_capacity(nr as usize);
    for r in 0..nr as usize {
        let Some(s) = c_slice(*ranks.add(r), *rank_bytes.add(r)) else {
            return fail(Error::Bad("null rank head buffer"));
        };
        parts.push(s);
    }
    match heads_out(&parts, heads as usize, outer, inner) {
        Ok(b) => write_out(&b, out, cap, need),
        Err(e) => fail(e),
    }
}

/// Scatters canonical [outer][heads][inner] into head group `group` of `nr`.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_heads_in(
    canon: *const u8,
    canon_bytes: usize,
    heads: u32,
    outer: usize,
    inner: usize,
    nr: u32,
    group: u32,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    if need.is_null() {
        return fail(Error::Bad("null argument"));
    }
    let Some(src) = c_slice(canon, canon_bytes) else {
        return fail(Error::Bad("null canonical head buffer"));
    };
    match heads_in(src, heads as usize, outer, inner, nr as usize, group as usize) {
        Ok(b) => write_out(&b, out, cap, need),
        Err(e) => fail(e),
    }
}

/// Exports DFlash2 GQA drafter ring positions [lo, hi) from `nr` ranks into a `g6cdfl1` blob.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_dfl_export(
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    elem_bytes: u32,
    nr: u32,
    ring: u32,
    replicated: i32,
    k_ranks: *const *const u8,
    k_bytes: *const usize,
    v_ranks: *const *const u8,
    v_bytes: *const usize,
    lo: u64,
    hi: u64,
    check: i32,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    if need.is_null() || nr == 0 || k_ranks.is_null() || k_bytes.is_null() || v_ranks.is_null() || v_bytes.is_null() {
        return fail(Error::Bad("null or zero argument"));
    }
    let mut rings = Vec::with_capacity(nr as usize);
    for r in 0..nr as usize {
        let (Some(k), Some(v)) = (c_slice(*k_ranks.add(r), *k_bytes.add(r)), c_slice(*v_ranks.add(r), *v_bytes.add(r)))
        else {
            return fail(Error::Bad("null DFlash2 ring buffer"));
        };
        rings.push(DflRing { k, v });
    }
    let spec = DflSpec { layers, kv_heads, head_dim, elem_bytes };
    let layout = DflLayout { nr, ring, replicated: replicated != 0 };
    match dfl_export(spec, &layout, &rings, lo, hi, check != 0) {
        Ok(c) => {
            *need = c.byte_len();
            if cap < *need || out.is_null() {
                return 1;
            }
            let b = c.to_bytes();
            std::ptr::copy_nonoverlapping(b.as_ptr(), out, b.len());
            0
        }
        Err(e) => fail(e),
    }
}

/// Imports a `g6cdfl1` blob into rank `rank` of `nr`'s DFlash2 K/V rings.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_dfl_import(
    blob: *const u8,
    n: usize,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    elem_bytes: u32,
    nr: u32,
    ring: u32,
    replicated: i32,
    rank: u32,
    k_dst: *mut u8,
    k_bytes: usize,
    v_dst: *mut u8,
    v_bytes: usize,
    written: *mut u64,
) -> i32 {
    let (Some(b), Some(k), Some(v)) = (c_slice(blob, n), c_slice_mut(k_dst, k_bytes), c_slice_mut(v_dst, v_bytes))
    else {
        return fail(Error::Bad("null argument"));
    };
    let c = match DflCanon::from_bytes(b) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    let spec = DflSpec { layers, kv_heads, head_dim, elem_bytes };
    let layout = DflLayout { nr, ring, replicated: replicated != 0 };
    let mut dst = DflRingMut { k, v };
    match dfl_import(&c, spec, &layout, rank, &mut dst) {
        Ok(w) => {
            if !written.is_null() {
                *written = w;
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Bytes of one rank's KDA linear-attention + short-conv buffer across `n_layers` slots at width `nr` (0 on error).
#[no_mangle]
pub extern "C" fn g6kv_canon_kda_rank_bytes(
    heads: u32,
    dk: u32,
    dv: u32,
    nhist: u32,
    n_streams: u32,
    elem_bytes: u32,
    n_layers: u32,
    nr: u32,
) -> usize {
    if n_layers == 0 {
        return 0;
    }
    let spec = KdaSpec { heads, dk, dv, nhist, n_streams, elem_bytes, layer_ids: vec![0; n_layers as usize] };
    spec.rank_bytes(nr).unwrap_or(0)
}

/// Exports KDA linear-attention + short-conv state from `nr` ranks into a `g6ckda1` blob.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_kda_export(
    heads: u32,
    dk: u32,
    dv: u32,
    nhist: u32,
    n_streams: u32,
    elem_bytes: u32,
    layer_ids: *const u32,
    n_layers: u32,
    nr: u32,
    ranks: *const *const u8,
    rank_bytes: *const usize,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    if need.is_null() || nr == 0 || n_layers == 0 || layer_ids.is_null() || ranks.is_null() || rank_bytes.is_null() {
        return fail(Error::Bad("null or zero argument"));
    }
    let ids = std::slice::from_raw_parts(layer_ids, n_layers as usize).to_vec();
    let spec = KdaSpec { heads, dk, dv, nhist, n_streams, elem_bytes, layer_ids: ids };
    let mut parts = Vec::with_capacity(nr as usize);
    for r in 0..nr as usize {
        let Some(s) = c_slice(*ranks.add(r), *rank_bytes.add(r)) else {
            return fail(Error::Bad("null KDA rank buffer"));
        };
        parts.push(s);
    }
    match kda_export(&spec, nr, &parts) {
        Ok(c) => {
            *need = c.byte_len();
            if cap < *need || out.is_null() {
                return 1;
            }
            let b = c.to_bytes();
            std::ptr::copy_nonoverlapping(b.as_ptr(), out, b.len());
            0
        }
        Err(e) => fail(e),
    }
}

/// Imports a `g6ckda1` blob into rank `rank` of `nr`'s KDA buffer.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_kda_import(
    blob: *const u8,
    n: usize,
    heads: u32,
    dk: u32,
    dv: u32,
    nhist: u32,
    n_streams: u32,
    elem_bytes: u32,
    layer_ids: *const u32,
    n_layers: u32,
    nr: u32,
    rank: u32,
    dst: *mut u8,
    dst_bytes: usize,
    written: *mut usize,
) -> i32 {
    if n_layers == 0 || layer_ids.is_null() {
        return fail(Error::Bad("null or zero layer_ids"));
    }
    let (Some(b), Some(out)) = (c_slice(blob, n), c_slice_mut(dst, dst_bytes)) else {
        return fail(Error::Bad("null argument"));
    };
    let c = match KdaCanon::from_bytes(b) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    let ids = std::slice::from_raw_parts(layer_ids, n_layers as usize).to_vec();
    let spec = KdaSpec { heads, dk, dv, nhist, n_streams, elem_bytes, layer_ids: ids };
    match kda_import(&c, &spec, nr, rank, out) {
        Ok(w) => {
            if !written.is_null() {
                *written = w;
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Packs validated optional KV, DFlash2, KDA and extra slices into one `g6cses1` session envelope.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_session_pack(
    kv: *const u8,
    kv_n: usize,
    dfl: *const u8,
    dfl_n: usize,
    kda: *const u8,
    kda_n: usize,
    extra: *const u8,
    extra_n: usize,
    out: *mut u8,
    cap: usize,
    need: *mut usize,
) -> i32 {
    if need.is_null() {
        return fail(Error::Bad("null need pointer"));
    }
    let (Some(kv_s), Some(dfl_s), Some(kda_s), Some(extra_s)) =
        (c_slice(kv, kv_n), c_slice(dfl, dfl_n), c_slice(kda, kda_n), c_slice(extra, extra_n))
    else {
        return fail(Error::Bad("null section with nonzero length"));
    };
    if !kv_s.is_empty() {
        if let Err(e) = Canon::from_bytes(kv_s) {
            return fail(e);
        }
    }
    if !dfl_s.is_empty() {
        if let Err(e) = DflCanon::from_bytes(dfl_s) {
            return fail(e);
        }
    }
    if !kda_s.is_empty() {
        if let Err(e) = KdaCanon::from_bytes(kda_s) {
            return fail(e);
        }
    }
    let total = SES_MAGIC.len() + 32 + kv_s.len() + dfl_s.len() + kda_s.len() + extra_s.len();
    *need = total;
    if cap < total || out.is_null() {
        return 1;
    }
    let packed = pack_session_parts(kv_s, dfl_s, kda_s, extra_s);
    std::ptr::copy_nonoverlapping(packed.as_ptr(), out, packed.len());
    0
}

/// Unpacks and validates a `g6cses1` session envelope, returning pointers and lengths into `blob`.
///
/// # Safety
/// Every pointer valid for the sizes given.
#[no_mangle]
pub unsafe extern "C" fn g6kv_canon_session_unpack(
    blob: *const u8,
    n: usize,
    kv: *mut *const u8,
    kv_n: *mut usize,
    dfl: *mut *const u8,
    dfl_n: *mut usize,
    kda: *mut *const u8,
    kda_n: *mut usize,
    extra: *mut *const u8,
    extra_n: *mut usize,
) -> i32 {
    let Some(b) = c_slice(blob, n) else {
        return fail(Error::Bad("null blob"));
    };
    let s = match unpack_session_slices(b) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    if !s.kv.is_empty() {
        if let Err(e) = Canon::from_bytes(s.kv) {
            return fail(e);
        }
    }
    if !s.dfl.is_empty() {
        if let Err(e) = DflCanon::from_bytes(s.dfl) {
            return fail(e);
        }
    }
    if !s.kda.is_empty() {
        if let Err(e) = KdaCanon::from_bytes(s.kda) {
            return fail(e);
        }
    }
    let put = |s: &[u8], pp: *mut *const u8, pn: *mut usize| {
        if !pp.is_null() {
            *pp = if s.is_empty() { std::ptr::null() } else { s.as_ptr() };
        }
        if !pn.is_null() {
            *pn = s.len();
        }
    };
    put(s.kv, kv, kv_n);
    put(s.dfl, dfl, dfl_n);
    put(s.kda, kda, kda_n);
    put(s.extra, extra, extra_n);
    0
}
