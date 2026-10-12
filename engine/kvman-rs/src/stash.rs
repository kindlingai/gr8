//! Block stores below the pool (--kv-tier). Two kinds of entry live in their page blocks. One is a retained run the
//! pool evicts: without a tier, the pool loses it. The other is a prefix that many conversations start with. The pool
//! decides everything here, from calls every rank makes in the same order. It hands the engine the copies to make (Io),
//! and the engine runs the copies before anything writes the pages.
//!
//! - Tiers come in flag order, each a store of `slots` blocks with a keep rule: evicted runs, shared prefixes (those
//!   that started at least shared_after conversations), or all. Lookups and placement try the tiers in order. A full
//!   tier makes room by its least recently used entries. Each such entry goes on to the next tier whose keep takes it (a
//!   Move), or the tier drops it.
//! - The store holds an evicted run below R by its first ceil(valid / 64) pages. A run past R (exclusive) holds the CP
//!   map. That map's pages below R are the same logical pages on every rank, and the store keeps only those.
//!   A hit that takes a retained run for its first pages stores that run too, as an evicted one: the grant writes past
//!   the hit's skip, and the run's owner would keep only the shared prefix.
//! - Counting: count() at each admission. A request that continues a live, retained or stored run (its prompt holds
//!   all of that run's prompt) counts toward nothing, so a five-turn chat counts once. The next prompt may carry
//!   another reply than the run wrote: an agent client can drop the reasoning or edit the reply. Any other request
//!   starts a conversation and counts once toward every grid-cell prefix of its prompt. The key of a prefix of S rows is
//!   its first S + 1 tokens: the MTP layer's row S - 1 depends on token S. When a run retires, the longest counted
//!   prefix it holds goes to the first tier that keeps shared prefixes, unless a tier has it already. The tier writes
//!   that prefix once, every later hit reads it, and it stays until its tier needs the room.
//! - A hit on an evicted run takes the run out of its tier, a hit on a shared prefix leaves the prefix there. Either way
//!   the blocks go back into the grant's first pages.
//! - Each call's copies, in order: Moves and Spills as placement made them (a Move reads a slot before a later copy
//!   reuses the slot), then Restores (a restore writes a page a spill of the same call may still have to read). Until
//!   the engine has run the copies and called io_done(), the pool refuses reserve, take and grow (Error::PendingIo).
//! - A tier with an index (an SSD folder) keeps its shared prefixes across restarts. The engine writes the blocks,
//!   then io_done() commits their index records. A drop record reaches the disk before the entry's slots are reused.
//!   See Index.

use crate::pool::{Kept, PAGE};
use crate::reuse::Finished;
use crate::tier::SlotAlloc;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    Evicted,
    Shared,
    All,
}

impl Keep {
    fn takes(self, k: Kind) -> bool {
        matches!((self, k), (Keep::All, _) | (Keep::Evicted, Kind::Evicted) | (Keep::Shared, Kind::Shared))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Evicted,
    Shared,
}

/// One copy for the engine. A page is one pool page of every layer, a slot one block of a tier's store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Io {
    Spill { tier: u32, page: u32, slot: u64 },
    Move { tier: u32, slot: u64, to: u32, to_slot: u64 },
    Restore { tier: u32, slot: u64, page: u32 },
}

/// A stored run or prefix, and st is what match() runs begin() on. Its first pages are the blocks of `base` (a shared
/// prefix entry's id and its pages), when it has one, and its own slots hold the pages after them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: u64,
    pub kind: Kind,
    pub key: u64,
    pub st: Finished,
    pub slots: Vec<u64>,
    pub base: Option<(u64, usize)>,
}

impl Entry {
    /// The pages it can restore: its base's and its own.
    pub fn pages(&self) -> usize {
        self.base.map_or(0, |b| b.1) + self.slots.len()
    }
}

/// Counters for status.s2, per tier.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierStats {
    pub slots: u64,
    pub free_slots: u64,
    pub entries: u32,
    pub shared: u32,
    pub spilled_pages: u64,
    pub moved_in_pages: u64,
    pub restored_pages: u64,
    pub hits: u64,
    pub dropped: u64,
}

pub struct Tier {
    keep: Keep,
    alloc: SlotAlloc,
    slots: u64,
    entries: Vec<Entry>, // least recently used first
    index: Option<Index>,
    stats: TierStats,
}

/// Prefix counts kept at most, the least recently counted dropped first.
pub const COUNTS: usize = 4096;
/// The grid when prefix reuse has none: the served prefill chunk.
pub const GRID: u64 = 4096;

pub struct Stash {
    tiers: Vec<Tier>,
    pub(crate) io: Vec<Io>,
    taken: bool,
    commits: Vec<(usize, u64)>,
    counts: BTreeMap<u64, (u32, u64)>,
    clock: u64,
    shared_after: u32,
    grid: u64,
    next_id: u64,
    pin: Vec<u64>,
    pub index_errors: u64,
}

const FNV: u64 = 14695981039346656037;

pub fn prefix_key(tokens: &[i32]) -> u64 {
    tokens.iter().fold(FNV ^ tokens.len() as u64, |h, &t| (h ^ t as u32 as u64).wrapping_mul(1099511628211))
}

fn pages(rows: u64) -> usize {
    rows.div_ceil(PAGE) as usize
}

impl Stash {
    pub fn new(shared_after: u32, grid: u64) -> Stash {
        Stash {
            tiers: Vec::new(),
            io: Vec::new(),
            taken: false,
            commits: Vec::new(),
            counts: BTreeMap::new(),
            clock: 0,
            shared_after: shared_after.max(1),
            grid: if grid == 0 { GRID } else { grid },
            next_id: 1,
            pin: Vec::new(),
            index_errors: 0,
        }
    }

    /// A tier after the others. With an index, its committed shared prefixes come back (in commit order).
    pub fn add_tier(&mut self, slots: u64, keep: Keep, index: Option<(Index, Vec<Entry>)>) {
        let mut t = Tier {
            keep,
            alloc: SlotAlloc::new(slots),
            slots,
            entries: Vec::new(),
            index: None,
            stats: TierStats::default(),
        };
        if let Some((ix, loaded)) = index {
            for e in loaded {
                if e.slots.iter().all(|&s| s < slots) && t.alloc.claim(&e.slots) {
                    self.next_id = self.next_id.max(e.id + 1);
                    t.entries.push(e);
                }
            }
            t.index = Some(ix);
        }
        self.tiers.push(t);
    }

    pub fn tiers(&self) -> usize {
        self.tiers.len()
    }

    pub fn stats(&self, t: usize) -> Option<TierStats> {
        let x = self.tiers.get(t)?;
        Some(TierStats {
            slots: x.slots,
            free_slots: x.alloc.free(),
            entries: x.entries.len() as u32,
            shared: x.entries.iter().filter(|e| e.kind == Kind::Shared).count() as u32,
            ..x.stats
        })
    }

    pub fn entries(&self, t: usize) -> &[Entry] {
        &self.tiers[t].entries
    }

    /// Whether a shared tier exists, so counting has a purpose.
    pub fn counting(&self) -> bool {
        self.tiers.iter().any(|t| t.keep.takes(Kind::Shared))
    }

    pub fn count_of(&self, key: u64) -> u32 {
        self.counts.get(&key).map_or(0, |c| c.0)
    }

    pub(crate) fn pending(&self) -> bool {
        !self.io.is_empty() || self.taken
    }

    pub(crate) fn take_io(&mut self) -> Vec<Io> {
        self.taken = !self.io.is_empty() || !self.commits.is_empty();
        std::mem::take(&mut self.io)
    }

    /// The engine ran the copies io() gave it: commit the index records of the entries they wrote.
    pub(crate) fn io_done(&mut self) -> io::Result<()> {
        self.taken = false;
        let mut err = Ok(());
        for (t, id) in std::mem::take(&mut self.commits) {
            let tier = &mut self.tiers[t];
            if let (Some(ix), Some(e)) = (&mut tier.index, tier.entries.iter().find(|e| e.id == id)) {
                if let Err(e) = ix.add(e) {
                    self.index_errors += 1;
                    err = Err(e);
                }
            }
        }
        err
    }

    /// Whether `prompt` continues a stored evicted run.
    pub(crate) fn continues_stored(&self, prompt: &[i32]) -> bool {
        self.tiers
            .iter()
            .flat_map(|t| &t.entries)
            .any(|e| e.kind == Kind::Evicted && continues(prompt, &e.st.history, e.st.pf_end.max(0) as usize))
    }

    /// Counts a request that starts a conversation toward every grid-cell prefix of its prompt. Returns how many.
    pub(crate) fn count(&mut self, prompt: &[i32], continuing: bool) -> u32 {
        if !self.counting() || continuing || prompt.len() < 2 {
            return 0;
        }
        let g = self.grid as usize;
        let mut n = 0;
        let mut s = g;
        while s < prompt.len() {
            self.clock += 1;
            let c = self.counts.entry(prefix_key(&prompt[..s + 1])).or_insert((0, 0));
            c.0 += 1;
            c.1 = self.clock;
            n += 1;
            s += g;
        }
        while self.counts.len() > COUNTS {
            let (&k, _) = self.counts.iter().min_by_key(|(k, c)| (c.1, **k)).expect("not empty");
            self.counts.remove(&k);
        }
        n
    }

    /// A retiring run: the longest counted prefix it holds that no tier has yet goes to the first tier that keeps shared
    /// prefixes. `rows_max`: the rows whose pages are the same logical pages on every rank (R for a run past R).
    pub(crate) fn promote(&mut self, list: &[u32], st: &Finished, rows_max: u64) {
        if !self.counting() || st.valid <= 0 || st.history.is_empty() {
            return;
        }
        let g = self.grid;
        let top = (st.valid as u64).min(st.history.len() as u64 - 1).min(rows_max).min(list.len() as u64 * PAGE);
        let mut s = top / g * g;
        while s >= g {
            let key = prefix_key(&st.history[..s as usize + 1]);
            if self.count_of(key) >= self.shared_after {
                if self.tiers.iter().flat_map(|t| &t.entries).any(|e| e.kind == Kind::Shared && e.key == key) {
                    return;
                }
                let n = pages(s);
                let fin = Finished {
                    history: st.history[..s as usize + 1].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                self.store(Kind::Shared, key, fin, &list[..n]);
                return;
            }
            s -= g;
        }
    }

    /// Stores evicted retained runs, oldest first. below_pages: the pages of a run past R that are alike on every rank.
    pub(crate) fn put_evicted(&mut self, gone: &[Kept], below_pages: usize) {
        for k in gone {
            if k.st.valid <= 0 {
                continue;
            }
            let mut n = pages(k.st.valid as u64).min(k.list.len());
            let mut st = k.st.clone();
            if k.exclusive {
                n = n.min(below_pages);
                st.valid = st.valid.min((n as u64 * PAGE) as i32);
            }
            if n == 0 {
                continue;
            }
            self.store(Kind::Evicted, 0, st, &k.list[..n]);
        }
    }

    // the longest stored shared prefix that `history` starts with and whose pages fit in `max_pages`: (id, pages)
    fn shared_base(&self, history: &[i32], max_pages: usize) -> Option<(u64, usize)> {
        self.tiers
            .iter()
            .flat_map(|t| &t.entries)
            .filter(|e| e.kind == Kind::Shared && e.slots.len() <= max_pages && history.starts_with(&e.st.history))
            .max_by_key(|e| (e.slots.len(), e.id))
            .map(|e| (e.id, e.slots.len()))
    }

    fn store(&mut self, kind: Kind, key: u64, st: Finished, pages: &[u32]) {
        // a run that starts with a stored shared prefix stores only the pages after it
        let base = if kind == Kind::Evicted { self.shared_base(&st.history, pages.len()) } else { None };
        let own = &pages[base.map_or(0, |b| b.1)..];
        if own.is_empty() {
            return;
        }
        let pin = std::mem::take(&mut self.pin);
        self.pin = pin.iter().copied().chain(base.map(|b| b.0)).collect();
        let placed = self.place(0, kind, own.len());
        self.pin = pin;
        let Some((t, slots)) = placed else { return };
        self.io.extend(own.iter().zip(&slots).map(|(&page, &slot)| Io::Spill { tier: t as u32, page, slot }));
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[t];
        tier.stats.spilled_pages += own.len() as u64;
        tier.entries.push(Entry { id, kind, key, st, slots, base });
        if kind == Kind::Shared && tier.index.is_some() {
            self.commits.push((t, id));
        }
    }

    /// The first tier from `start` whose keep takes `kind` and which can make room for n blocks: (tier, slots).
    fn place(&mut self, start: usize, kind: Kind, n: usize) -> Option<(usize, Vec<u64>)> {
        for t in start..self.tiers.len() {
            if self.tiers[t].keep.takes(kind) && n as u64 <= self.tiers[t].slots && self.make_room(t, n) {
                return Some((t, self.tiers[t].alloc.alloc(n).expect("room made")));
            }
        }
        None
    }

    fn make_room(&mut self, t: usize, n: usize) -> bool {
        while self.tiers[t].alloc.free() < n as u64 {
            let Some(i) = self.tiers[t].entries.iter().position(|e| !self.pin.contains(&e.id)) else { return false };
            let e = self.tiers[t].entries.remove(i);
            self.evict(t, e);
        }
        true
    }

    // e leaves tier t for the next tier that keeps its kind, or is dropped. Tier t frees e's slots after queueing the move.
    fn evict(&mut self, t: usize, e: Entry) {
        if let Some(ix) = &mut self.tiers[t].index {
            if e.kind == Kind::Shared && ix.drop(e.id).is_err() {
                self.index_errors += 1;
            }
        }
        self.commits.retain(|c| *c != (t, e.id));
        // the base of a run on the move must stay while the run finds a place
        let pinned = self.pin.len();
        self.pin.extend(e.base.map(|b| b.0));
        let placed = self.place(t + 1, e.kind, e.slots.len());
        self.pin.truncate(pinned);
        match placed {
            Some((u, new)) => {
                self.io.extend(e.slots.iter().zip(&new).map(|(&slot, &to_slot)| Io::Move {
                    tier: t as u32,
                    slot,
                    to: u as u32,
                    to_slot,
                }));
                let to = &mut self.tiers[u];
                to.stats.moved_in_pages += new.len() as u64;
                if e.kind == Kind::Shared && to.index.is_some() {
                    self.commits.push((u, e.id));
                }
                self.tiers[u].entries.insert(0, Entry { slots: new, ..e.clone() });
            }
            None => {
                self.tiers[t].stats.dropped += 1;
                if e.kind == Kind::Shared {
                    self.drop_dependents(e.id);
                }
            }
        }
        self.tiers[t].alloc.release(&e.slots);
    }

    // a shared prefix left every tier: the runs stored on top of it go too
    fn drop_dependents(&mut self, id: u64) {
        for t in 0..self.tiers.len() {
            let all = std::mem::take(&mut self.tiers[t].entries);
            let (gone, keep): (Vec<Entry>, Vec<Entry>) =
                all.into_iter().partition(|e| e.base.is_some_and(|b| b.0 == id));
            self.tiers[t].entries = keep;
            for e in gone {
                self.tiers[t].alloc.release(&e.slots);
                self.tiers[t].stats.dropped += 1;
            }
        }
    }

    /// Where a shared entry is now: (tier, its slots).
    fn locate(&self, id: u64) -> Option<(usize, &[u64])> {
        self.tiers
            .iter()
            .enumerate()
            .find_map(|(t, x)| x.entries.iter().find(|e| e.id == id).map(|e| (t, &e.slots[..])))
    }

    pub(crate) fn find(&self, t: usize, id: u64) -> Option<&Entry> {
        self.tiers.get(t)?.entries.iter().find(|e| e.id == id)
    }

    /// A hit's entry, for a grant: an evicted run leaves its tier (its slots stay held until restore() queues the
    /// copies), a shared prefix stays, pinned against this call's placements, and becomes the most recently used.
    pub(crate) fn begin_restore(&mut self, t: usize, id: u64) -> Entry {
        let tier = &mut self.tiers[t];
        let i = tier.entries.iter().position(|e| e.id == id).expect("found by the caller");
        let e = tier.entries.remove(i);
        tier.stats.hits += 1;
        if e.kind == Kind::Shared {
            tier.entries.push(e.clone());
            self.pin.push(id);
        }
        self.pin.extend(e.base.map(|b| b.0));
        e
    }

    pub(crate) fn restore(&mut self, t: usize, e: &Entry, pages: &[u32]) {
        let b = e.base.map_or(0, |b| b.1).min(pages.len());
        if let Some((base, _)) = e.base {
            let (u, slots) = self.locate(base).expect("a base outlives the runs on it");
            let copies: Vec<Io> = slots
                .iter()
                .zip(&pages[..b])
                .map(|(&slot, &page)| Io::Restore { tier: u as u32, slot, page })
                .collect();
            self.io.extend(copies);
            self.tiers[u].stats.restored_pages += b as u64;
        }
        self.io.extend(e.slots.iter().zip(&pages[b..]).map(|(&slot, &page)| Io::Restore {
            tier: t as u32,
            slot,
            page,
        }));
        self.tiers[t].stats.restored_pages += (pages.len() - b) as u64;
        if e.kind == Kind::Evicted {
            self.tiers[t].alloc.release(&e.slots);
        }
        self.pin.clear();
    }

    /// The entries loaded from indexes, one line each. Every rank compares these lines at boot.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for (t, tier) in self.tiers.iter().enumerate() {
            for e in &tier.entries {
                s += &format!("{t} {} {:x} {} {:x}\n", e.id, e.key, e.st.valid, slots_hash(&e.slots));
            }
        }
        s
    }

    /// Keeps the entries on the lines of `agreed`, in their order, and drops the rest (with their index records).
    pub fn keep_agreed(&mut self, agreed: &str) {
        let want: Vec<&str> = agreed.lines().collect();
        for t in 0..self.tiers.len() {
            let all = std::mem::take(&mut self.tiers[t].entries);
            let line = |e: &Entry| format!("{t} {} {:x} {} {:x}", e.id, e.key, e.st.valid, slots_hash(&e.slots));
            let mut kept: Vec<(usize, Entry)> = Vec::new();
            for e in all {
                match want.iter().position(|w| *w == line(&e)) {
                    Some(i) => kept.push((i, e)),
                    None => {
                        if let Some(ix) = &mut self.tiers[t].index {
                            if ix.drop(e.id).is_err() {
                                self.index_errors += 1;
                            }
                        }
                        self.tiers[t].alloc.release(&e.slots);
                    }
                }
            }
            kept.sort_by_key(|k| k.0);
            self.tiers[t].entries = kept.into_iter().map(|k| k.1).collect();
        }
        self.next_id = self.tiers.iter().flat_map(|t| &t.entries).map(|e| e.id + 1).max().unwrap_or(1);
    }

    pub(crate) fn fold(&self, mut h: u64, mix: fn(u64, u64) -> u64) -> u64 {
        h = mix(h, self.next_id);
        for t in &self.tiers {
            h = mix(mix(h, t.slots), t.alloc.free());
            for e in &t.entries {
                h = mix(mix(mix(mix(h, e.id), e.key), e.st.valid as i64 as u64), slots_hash(&e.slots));
                if let Some((id, n)) = e.base {
                    h = mix(mix(h, id), n as u64);
                }
            }
        }
        for (k, c) in &self.counts {
            h = mix(mix(h, *k), c.0 as u64);
        }
        h
    }
}

/// Whether a prompt continues a run: it holds the run's first `n` tokens (its prompt, the rows the run prefilled).
pub(crate) fn continues(prompt: &[i32], run: &[i32], n: usize) -> bool {
    n > 0 && n <= run.len() && prompt.len() >= n && prompt[..n] == run[..n]
}

fn slots_hash(s: &[u64]) -> u64 {
    s.iter().fold(FNV, |h, &x| (h ^ x).wrapping_mul(1099511628211))
}

/// A tier's persistent index (PATH/index beside the blocks): a header naming what makes blocks compatible (model,
/// TP, R, page-block layout, size), then records, each [u32 length][u64 checksum][payload]: add (an entry) or drop (an
/// id). A record whose checksum fails ends the log, and the file is cut there. The engine writes an entry's blocks
/// before io_done() appends its add record, and a drop record is synced before the entry's slots are reused, so a torn
/// write never names blocks that hold anything else.
pub struct Index {
    path: PathBuf,
    file: File,
    compat: String,
    records: u64,
    live: BTreeMap<u64, Entry>,
}

const MAGIC: &str = "g6kv-index 1";

fn fnv_bytes(b: &[u8]) -> u64 {
    b.iter().fold(FNV, |h, &x| (h ^ x as u64).wrapping_mul(1099511628211))
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Some(a)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

fn encode_add(e: &Entry) -> Vec<u8> {
    let mut p = vec![1u8];
    put_u64(&mut p, e.id);
    put_u64(&mut p, e.key);
    for x in [e.st.valid, e.st.written_end, e.st.pf_end] {
        put_u32(&mut p, x as u32);
    }
    put_u32(&mut p, e.st.history.len() as u32);
    for &t in &e.st.history {
        put_u32(&mut p, t as u32);
    }
    put_u32(&mut p, e.slots.len() as u32);
    for &s in &e.slots {
        put_u64(&mut p, s);
    }
    p
}

fn decode(p: &[u8]) -> Option<Result<Entry, u64>> {
    let mut c = Cursor(p);
    match c.take(1)?[0] {
        1 => {
            let (id, key) = (c.u64()?, c.u64()?);
            let (valid, written_end, pf_end) = (c.u32()? as i32, c.u32()? as i32, c.u32()? as i32);
            let n = c.u32()? as usize;
            let history = (0..n).map(|_| c.u32().map(|t| t as i32)).collect::<Option<Vec<_>>>()?;
            let n = c.u32()? as usize;
            let slots = (0..n).map(|_| c.u64()).collect::<Option<Vec<_>>>()?;
            Some(Ok(Entry {
                id,
                kind: Kind::Shared,
                key,
                st: Finished { history, valid, written_end, pf_end },
                slots,
                base: None,
            }))
        }
        2 => Some(Err(c.u64()?)),
        _ => None,
    }
}

impl Index {
    /// Opens (or starts) the index in `dir`. A different compat string, or no header, starts it empty.
    pub fn open(dir: &Path, compat: &str) -> io::Result<(Index, Vec<Entry>)> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("index");
        let mut raw = Vec::new();
        if let Ok(mut f) = File::open(&path) {
            f.read_to_end(&mut raw)?;
        }
        let head = format!("{MAGIC} {compat}\n");
        let mut live: BTreeMap<u64, Entry> = BTreeMap::new();
        let mut order: Vec<u64> = Vec::new();
        let (mut good, mut records) = (0usize, 0u64);
        if raw.starts_with(head.as_bytes()) {
            good = head.len();
            let mut c = Cursor(&raw[good..]);
            while let Some(n) = c.u32() {
                let (Some(sum), Some(p)) = (c.u64(), c.take(n as usize)) else { break };
                if fnv_bytes(p) != sum {
                    break;
                }
                match decode(p) {
                    Some(Ok(e)) => {
                        order.retain(|&i| i != e.id);
                        order.push(e.id);
                        live.insert(e.id, e);
                    }
                    Some(Err(id)) => {
                        live.remove(&id);
                        order.retain(|&i| i != id);
                    }
                    None => break,
                }
                records += 1;
                good = raw.len() - c.0.len();
            }
        }
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        if good == 0 {
            file.set_len(0)?;
            file.write_all(head.as_bytes())?;
            good = head.len();
        }
        file.set_len(good as u64)?; // a torn last record goes
        file.seek(SeekFrom::End(0))?;
        file.sync_all()?;
        let loaded = order.iter().filter_map(|i| live.get(i).cloned()).collect();
        let ix = Index { path, file, compat: compat.to_string(), records, live };
        Ok((ix, loaded))
    }

    fn append(&mut self, p: &[u8]) -> io::Result<()> {
        let mut r = Vec::with_capacity(p.len() + 12);
        put_u32(&mut r, p.len() as u32);
        put_u64(&mut r, fnv_bytes(p));
        r.extend_from_slice(p);
        self.file.write_all(&r)?;
        self.file.sync_data()?;
        self.records += 1;
        if self.records > 4 * self.live.len() as u64 + 64 {
            self.compact()?;
        }
        Ok(())
    }

    pub fn add(&mut self, e: &Entry) -> io::Result<()> {
        self.live.insert(e.id, e.clone());
        self.append(&encode_add(e))
    }

    pub fn drop(&mut self, id: u64) -> io::Result<()> {
        if self.live.remove(&id).is_none() {
            return Ok(());
        }
        let mut p = vec![2u8];
        put_u64(&mut p, id);
        self.append(&p)
    }

    // rewrites the log with the live entries only, then renames it over the old one
    fn compact(&mut self) -> io::Result<()> {
        let tmp = self.path.with_extension("tmp");
        let mut f = File::create(&tmp)?;
        f.write_all(format!("{MAGIC} {}\n", self.compat).as_bytes())?;
        for e in self.live.values() {
            let p = encode_add(e);
            let mut r = Vec::new();
            put_u32(&mut r, p.len() as u32);
            put_u64(&mut r, fnv_bytes(&p));
            r.extend_from_slice(&p);
            f.write_all(&r)?;
        }
        f.sync_all()?;
        std::fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        self.records = self.live.len() as u64;
        Ok(())
    }
}
