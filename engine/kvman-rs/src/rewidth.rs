//! A persisted SSD store written at another width, rewritten for this rank before the engine opens it
//! (GR8_KV_REWIDTH, set by `./gr8 <model> scale` for one boot). Host only: no KV byte changes.
//!
//! A store's blocks hold row bytes that are the same at every width (canon.rs): a `Full` block (a page below R) and a
//! `Rep` block (the replicated parts of tail pages) are the same bytes on every rank. Only a tail's `Own` blocks depend
//! on the width: rank r of nr holds the logical pages lp >= R/64 with lp % nr == r (canon::table), its j-th own block
//! the j-th of them. So the rewrite keeps every entry, its ids and its Full and Rep slots, and builds each tail's own
//! blocks for this rank from the old ranks that held those pages: local blocks for this box's old rank, the others'
//! through their `kvman-peer` (peer.rs). Every new rank computes the same index from the same old index.
//!
//! New own blocks go only to slots no old entry uses, so the old index stays valid for a rollback (the launcher keeps
//! it as index.prev). When those slots run out, the least recently used entries with tails go first.

use crate::peer::{self, Peer};
use crate::stash::{self, Entry, Tail};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path;

const PAGE: u64 = 64;

/// GR8_KV_REWIDTH: `<old_nr>,<old_rank|x>,<token_file>,<old_set>,<peer0>,...,<peer{old_nr-1}>`. `x`: this box was not
/// a member. `old_set`: the model set id the old width booted with (the launcher's id names the TP too, so the old
/// store's `set` is checked against it, not against this boot's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub old_nr: u32,
    pub old_rank: Option<u32>,
    pub token_file: String,
    pub old_set: String,
    pub peers: Vec<String>,
}

impl Spec {
    pub fn parse(s: &str) -> Result<Spec, String> {
        let f: Vec<&str> = s.split(',').map(str::trim).collect();
        let bad = || format!("GR8_KV_REWIDTH {s:?}: want <old_nr>,<old_rank|x>,<token_file>,<old_set>,<peer0>,...");
        if f.len() < 5 {
            return Err(bad());
        }
        let old_nr: u32 = f[0].parse().map_err(|_| bad())?;
        let old_rank = if f[1] == "x" { None } else { Some(f[1].parse::<u32>().map_err(|_| bad())?) };
        let peers: Vec<String> = f[4..].iter().map(|p| p.to_string()).collect();
        if old_nr == 0 || peers.len() != old_nr as usize || old_rank.is_some_and(|r| r >= old_nr) || f[2].is_empty() ||
            f[3].is_empty()
        {
            return Err(bad());
        }
        Ok(Spec { old_nr, old_rank, token_file: f[2].to_string(), old_set: f[3].to_string(), peers })
    }
}

/// A compat line's `key value` pairs.
pub fn fields(compat: &str) -> BTreeMap<&str, &str> {
    let w: Vec<&str> = compat.split_whitespace().collect();
    w.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0], c[1])).collect()
}

/// What must be equal for the old store's blocks to be this engine's, besides the width and the model set (`set`, which
/// names the width too: checked against the spec's old_set).
const SAME: [&str; 7] = ["below", "grid", "block", "bindings", "format", "engine", "kv"];

/// The logical pages past R (from `rp`) rank `rank` of `nr` holds, in order: its own blocks, j-th to j-th.
fn owned(rp: u64, nr: u64, rank: u64) -> impl Iterator<Item = u64> {
    let first = rp + (rank + nr - rp % nr) % nr;
    (first..).step_by(nr as usize)
}

/// The old ranks' stores: this box's own (its old rank's, in `dir`) and the others' through their peers.
struct Src<'a> {
    spec: &'a Spec,
    token: String,
    peers: Vec<Option<Peer>>,
    blocks: File,
    slot_bytes: u64,
    buf: Vec<u8>,
}

impl Src<'_> {
    fn peer(&mut self, q: u32) -> Result<&mut Peer, String> {
        let p = &mut self.peers[q as usize];
        if p.is_none() {
            let a = &self.spec.peers[q as usize];
            *p = Some(Peer::open(a, &self.token).map_err(|e| format!("peer {q} ({a}): {e}"))?);
        }
        Ok(p.as_mut().expect("set"))
    }

    /// Old rank `q`'s block `src` into this store's slot `dst`.
    fn copy(&mut self, q: u32, src: u64, dst: u64) -> Result<(), String> {
        let mut buf = std::mem::take(&mut self.buf);
        let off = src * self.slot_bytes;
        let r = if self.spec.old_rank == Some(q) {
            self.blocks.read_exact_at(&mut buf, off).map_err(|e| format!("blocks slot {src}: {e}"))
        } else {
            self.peer(q).and_then(|p| p.read("blocks", off, &mut buf).map_err(|e| e.to_string()))
        };
        let r = r.and_then(|()| {
            self.blocks.write_all_at(&buf, dst * self.slot_bytes).map_err(|e| format!("blocks slot {dst}: {e}"))
        });
        self.buf = buf;
        r
    }
}

/// Rewrites `dir`/index (and `dir`/blocks) for rank `rank` of `nr` from the old width's store. `compat` is this
/// engine's compat line, `slots` and `slot_bytes` its store's. Ok(Err(why)): skipped (the engine starts the store
/// empty, as without the knob). Ok(Ok(n)): n entries carried. Err: failed; `dir`/index is as it was.
pub fn rewidth(
    dir: &Path,
    compat: &str,
    nr: u32,
    rank: u32,
    slots: u64,
    slot_bytes: u64,
    spec: &Spec,
) -> Result<Result<u64, String>, String> {
    let token = peer::read_token(spec.token_file.as_ref()).map_err(|e| e.to_string())?;
    let mut first: Option<Peer> = None;
    let raw = match spec.old_rank {
        Some(_) => match std::fs::read(dir.join("index.prev")) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Err("no index.prev".into())),
            Err(e) => return Err(format!("{}: {e}", dir.join("index.prev").display())),
        },
        None => {
            let a = &spec.peers[0];
            let p = first.insert(Peer::open(a, &token).map_err(|e| format!("peer 0 ({a}): {e}"))?);
            p.read_all("index.prev").map_err(|e| e.to_string())?
        }
    };
    let Some((old_compat, entries, counts)) = stash::read_index(&raw) else {
        return Ok(Err("index.prev has no index header".into()));
    };
    let (old, new) = (fields(&old_compat), fields(compat));
    if let Some(k) = SAME.iter().find(|k| old.get(*k) != new.get(*k)) {
        return Ok(Err(format!("{k} differs")));
    }
    if old.get("set") != Some(&spec.old_set.as_str()) {
        return Ok(Err("set differs from the old width's".into()));
    }
    if old.get("nr").and_then(|v| v.parse::<u32>().ok()) != Some(spec.old_nr) {
        return Ok(Err("nr differs from the old width".into()));
    }
    let num = |m: &BTreeMap<&str, &str>, k: &str| m.get(k).and_then(|v| v.parse::<u64>().ok());
    let (Some(below), Some(old_slots)) = (num(&old, "below"), num(&old, "slots")) else {
        return Ok(Err("below or slots missing".into()));
    };
    let (rp, onr, nnr) = (below / PAGE, spec.old_nr as u64, nr as u64);
    let lim = old_slots.min(slots);

    // the entries kept: none with boundary state (not a GLM store's), none with a slot past the new store
    let own_range = |e: &Entry| {
        let t = e.tail.map_or((0, 0), |t| (t.own as usize, t.rep as usize));
        let at = e.slots.len() - e.state as usize - t.0 - t.1;
        at..at + t.0
    };
    let mut kept: Vec<&Entry> = entries
        .iter()
        .filter(|e| {
            let own = own_range(e);
            e.state == 0 && e.slots.iter().enumerate().all(|(i, &s)| own.contains(&i) || s < slots)
        })
        .collect();
    let used: BTreeSet<u64> = entries.iter().flat_map(|e| e.slots.iter().copied()).collect();
    let free: Vec<u64> = (0..lim).filter(|s| !used.contains(s)).collect();
    let new_own = |e: &Entry| e.tail.map_or(0, |t| (t.logical as u64).div_ceil(nnr));
    loop {
        // an entry on a base that is gone goes too
        loop {
            let ids: HashSet<u64> = kept.iter().map(|e| e.id).collect();
            let n = kept.len();
            kept.retain(|e| e.base.is_none_or(|b| ids.contains(&b.0)));
            if kept.len() == n {
                break;
            }
        }
        if kept.iter().map(|e| new_own(e)).sum::<u64>() <= free.len() as u64 {
            break;
        }
        let lru = kept.iter().position(|e| e.tail.is_some()).expect("a tail needs the slots");
        kept.remove(lru);
    }

    let blocks = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("blocks"))
        .map_err(|e| format!("{}: {e}", dir.join("blocks").display()))?;
    let mut peers: Vec<Option<Peer>> = (0..spec.old_nr).map(|_| None).collect();
    peers[0] = first;
    let mut src = Src { spec, token, peers, blocks, slot_bytes, buf: vec![0u8; slot_bytes as usize] };
    let mut next_free = free.into_iter();
    let mut out: Vec<Entry> = Vec::with_capacity(kept.len());
    for e in kept {
        let own = own_range(e);
        let mut n = e.clone();
        // a box that held no old rank has none of the blocks: the width-neutral ones come from the old ranks in turn
        if spec.old_rank.is_none() {
            for (_, &s) in e.slots.iter().enumerate().filter(|(i, _)| !own.contains(i)) {
                src.copy((s % onr) as u32, s, s)?;
            }
        }
        if let Some(t) = e.tail {
            let k = new_own(e) as usize;
            let mine: Vec<u64> = (0..k).map(|_| next_free.next().expect("counted")).collect();
            for (j, lp) in owned(rp, nnr, rank as u64).take(k).enumerate() {
                if lp >= rp + t.logical as u64 {
                    break; // this rank holds fewer of the tail's pages: the block is never read
                }
                let q = lp % onr;
                let idx = ((lp - owned(rp, onr, q).next().expect("endless")) / onr) as usize;
                src.copy(q as u32, e.slots[own.start + idx], mine[j])?;
            }
            n.slots.splice(own, mine);
            n.tail = Some(Tail { own: k as u32, ..t });
        }
        out.push(n);
    }
    src.blocks.sync_all().map_err(|e| format!("blocks: {e}"))?;
    stash::write_index(dir, compat, &out, &counts).map_err(|e| format!("{}: {e}", dir.join("index").display()))?;
    Ok(Ok(out.len() as u64))
}

// ---- C interface (g6kvman.h g6kv_rewidth) ----

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

/// The reason of this thread's last g6kv_rewidth that carried nothing (returned 0 or -1).
#[no_mangle]
pub extern "C" fn g6kv_rewidth_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

/// Rewrites the store in `dir` for rank `rank` of `nr` from another width's (`spec`: GR8_KV_REWIDTH). >0: entries
/// carried; 0: nothing to do (g6kv_rewidth_error says why); -1: error (g6kv_rewidth_error), `dir`/index untouched.
///
/// # Safety
/// `dir`, `compat` and `spec` are NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn g6kv_rewidth(
    dir: *const c_char,
    compat: *const c_char,
    nr: u32,
    rank: u32,
    slots: u64,
    slot_bytes: u64,
    spec: *const c_char,
) -> i64 {
    let s = |p: *const c_char| (!p.is_null()).then(|| CStr::from_ptr(p).to_str().ok()).flatten();
    let set = |m: String| LAST.with(|l| *l.borrow_mut() = CString::new(m.replace('\0', " ")).unwrap_or_default());
    let (Some(dir), Some(compat), Some(spec)) = (s(dir), s(compat), s(spec)) else {
        set("g6kv_rewidth: bad arguments".into());
        return -1;
    };
    let r = Spec::parse(spec).and_then(|sp| rewidth(dir.as_ref(), compat, nr, rank, slots, slot_bytes, &sp));
    match r {
        Ok(Ok(n)) => {
            if n == 0 {
                set("no entries".into());
            }
            n as i64
        }
        Ok(Err(why)) => {
            set(why);
            0
        }
        Err(e) => {
            set(e);
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_pages_follow_the_width_rule() {
        // R = 2 pages, nr 3: rank 0 holds 3, 6, ..; rank 2 holds 2, 5, ..
        assert_eq!(owned(2, 3, 0).take(3).collect::<Vec<_>>(), [3, 6, 9]);
        assert_eq!(owned(2, 3, 2).take(3).collect::<Vec<_>>(), [2, 5, 8]);
        assert_eq!(owned(0, 4, 1).take(2).collect::<Vec<_>>(), [1, 5]);
    }

    #[test]
    fn spec_parse() {
        let s = Spec::parse("4,x,/t,s4,a:1,b:1,c:1,d:1").unwrap();
        assert_eq!((s.old_nr, s.old_rank, s.old_set.as_str(), s.peers.len()), (4, None, "s4", 4));
        assert!(Spec::parse("4,1,/t,s4,a:1,b:1").is_err());
        assert!(Spec::parse("2,2,/t,s4,a:1,b:1").is_err());
        assert!(Spec::parse("2,0,/t,,a:1,b:1").is_err());
    }
}
