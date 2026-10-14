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
