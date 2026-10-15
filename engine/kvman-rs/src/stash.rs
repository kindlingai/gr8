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
//!   map. That map's pages below R are the same logical pages on every rank. Past R each rank holds every nr-th logical
//!   page of the attention cache and its indexer keys (its own shard), and every logical page of the bindings that are
//!   not sharded (the MTP layer). With set_tail(), a tail of T logical pages goes to each rank's own store as
//!   ceil(T / nr) blocks of its shard (Part::Own, the run's tail pages of its list below place) and the T pages' replicated
//!   parts, per_slot to a block (Part::Rep). Every rank queues the same copies; each block holds that rank's bytes.
//!   A hit that takes a retained run for its first pages stores that run too, as an evicted one: the grant writes past
//!   the hit's skip, and the run's owner would keep only the shared prefix.
//! - Counting: count() at each admission. A request that continues a live, retained or stored run (its prompt holds
//!   all of that run's prompt) counts toward nothing, so a five-turn chat counts once, also when the pool let its last
//!   run go unstored (the heads of the last 1,024 runs the pool let go are remembered). The next prompt may carry
//!   another reply than the run wrote: an agent client can drop the reasoning or edit the reply. Any other request
//!   starts a conversation and counts once toward every grid-cell prefix of its prompt, up to its system span (the
//!   system message and tool definitions as the chat template renders them) when the front end sent one. Conversations
//!   whose system prompts differ partway (an agent writes its working folder into it) still share the cells before the
//!   difference. The key of a prefix of S rows is its first S + 1 tokens: the MTP layer's row S - 1 depends on token S.
//!   When a run retires, the longest prefix it holds, within its span, that shared_after conversations started with
//!   goes to the first tier that keeps shared prefixes, unless a tier has it already. The tier writes that prefix once, every later hit reads it, and it stays until
//!   its tier needs the room. An SSD tier's index keeps the counts across restarts.
//! - A hit on an evicted run takes the run out of its tier, a hit on a shared prefix leaves the prefix there. Either way
//!   the blocks go back into the grant's first pages.
//! - Each call's copies, in order: Moves and Spills as placement made them (a Move reads a slot before a later copy
//!   reuses the slot), then Restores (a restore writes a page a spill of the same call may still have to read). Until
//!   the engine has run the copies and called io_done(), the pool refuses reserve, take and grow (Error::PendingIo).
//! - Each tier may refuse what it is offered (Policy): evicted runs of fewer than min_rows rows, and runs that
//!   continued their conversation fewer than min_hits times (a one-shot never). A refused entry tries the next tier.
//! - A write limit (--kv-write-limit F) caps the blocks all tiers write at F per page of KV the engine computes
//!   (earn(): prefill and decode pages, never the clock), so every rank refuses the same writes. A write needs credit
//!   above zero and may take it below; the credit holds at most BURST pages' worth. Without credit a spill is refused
//!   and a move drops its entry.
//! - A run's leading pages that another run in the pool also holds (live or retained, the same tokens) are a prefix
//!   conversations share. Spilling the run stores that prefix once as a Prefix entry, and the run as a reference to it
//!   plus its own pages. Later runs with that prefix store only their own pages, in every keep mode. A restore takes
//!   the prefix from a retained run in the pool when one holds it, else from the tier.
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
        matches!(
            (self, k),
            (Keep::All, _) | (_, Kind::Prefix) | (Keep::Evicted, Kind::Evicted) | (Keep::Shared, Kind::Shared)
        )
    }
}

/// Evicted: a retained run the pool let go. Shared: a prefix that started shared_after conversations. Prefix: a prefix
/// two runs in the pool held when one of them spilled, the base of the runs stored on it (every keep mode takes it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Evicted,
    Shared,
    Prefix,
}

impl Kind {
    /// A prefix entry: hits leave it in its tier, and runs may be stored on it.
    pub fn prefix(self) -> bool {
        self != Kind::Evicted
    }
}

/// What a tier refuses. 0 = no limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Policy {
    /// evicted runs need at least this many rows
    pub min_rows: u64,
    /// evicted runs need to have continued their conversation at least this many times
    pub min_hits: u32,
    /// bytes one block takes (status.s2 counts bytes)
    pub block_bytes: u64,
}

/// The write limit's credit starts at, and holds at most, this many blocks.
pub const BURST: i64 = 4096;

/// What of a page a copy moves. Full: one pool page of every layer (below R). Own: past R, physical page `page` of the
/// sharded attention caches and their indexer keys (this rank's shard). Rep(i): past R, logical page `page`'s
/// replicated parts (the unsharded bindings), the i-th of a block's per_slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Full,
    Own,
    Rep(u32),
}

/// One copy for the engine. A page is one pool page of every layer, a slot one block of a tier's store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Io {
    Spill { tier: u32, page: u32, slot: u64, part: Part },
    Move { tier: u32, slot: u64, to: u32, to_slot: u64 },
    Restore { tier: u32, slot: u64, page: u32, part: Part },
}

/// A stored run's tail past R (set_tail): `logical` pages, held in `own` blocks of each rank's shard (its tail pages
/// of its list below place) then `rep` blocks of
/// replicated parts, the last own + rep of the entry's slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tail {
    pub logical: u32,
    pub own: u32,
    pub rep: u32,
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
    /// times the conversation continued a run before this one was stored
    pub hits: u32,
    pub tail: Option<Tail>,
}

impl Entry {
    /// The (logical) pages it can restore: its base's, its own below R, and its tail's.
    pub fn pages(&self) -> usize {
        self.below() + self.tail.map_or(0, |t| t.logical as usize)
    }
    /// The pages it can restore below R.
    pub fn below(&self) -> usize {
        self.base.map_or(0, |b| b.1) + self.slots.len() - self.tail.map_or(0, |t| (t.own + t.rep) as usize)
    }
}

/// Counters for status.s2, per tier (g6kvman.h g6kv_tier_totals). Pages are blocks; block_bytes makes them bytes.
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
    /// entries that left the tier with no tier to go to, and their pages
    pub dropped: u64,
    pub dropped_pages: u64,
    /// entries passed on to the next tier, and their pages
    pub moved_out: u64,
    pub moved_out_pages: u64,
    /// entries the policy refused (min_rows, min_hits, rate), and their pages
    pub refused: u64,
    pub refused_pages: u64,
    /// times the tier had to make room (it was full)
    pub full: u64,
    pub block_bytes: u64,
    /// write limit credit left, blocks (0 without a limit; the same on every tier)
    pub credit: i64,
}

pub struct Tier {
    keep: Keep,
    alloc: SlotAlloc,
    slots: u64,
    entries: Vec<Entry>, // least recently used first
    index: Option<Index>,
    stats: TierStats,
    policy: Policy,
}

impl Tier {
    // whether the policy takes an evicted run of `rows` and `hits` (other kinds always)
    fn admits(&self, kind: Kind, rows: u64, hits: u32) -> bool {
        kind != Kind::Evicted || (rows >= self.policy.min_rows && hits >= self.policy.min_hits)
    }
}

/// Prefix counts kept at most, the least recently counted dropped first.
pub const COUNTS: usize = 4096;
/// Heads of runs the pool let go, kept at most for the continuation test.
pub const HEADS: usize = 1024;
/// The grid when prefix reuse has none: the served prefill chunk.
pub const GRID: u64 = 4096;

pub struct Stash {
    tiers: Vec<Tier>,
    pub(crate) io: Vec<Io>,
    taken: bool,
    commits: Vec<(usize, u64)>,
    counts: BTreeMap<u64, (u32, u64)>,
    counted: Vec<(u64, u32, u64)>, // counts since the last io_done, for the index
    // the heads of runs the pool let go (prompt rows, head_hash of them), oldest first, so a conversation whose last
    // run left the pool unstored still continues it
    heads: std::collections::VecDeque<(usize, u64)>,
    clock: u64,
    shared_after: u32,
    grid: u64,
    next_id: u64,
    pin: Vec<u64>,
    pub index_errors: u64,
    // the write limit: blocks per 1000 pages computed (0 = none), and the credit in thousandths of a block
    limit_milli: u64,
    credit: i64,
    /// prefix bases for evicted runs (on; kvman-sim turns it off to compare)
    pub dedupe: bool,
    // set_tail: (ranks, replicated parts per block); None = a run past R keeps only its pages below R.
    // tail_place: the pool's place with it, so a tail's shard pages are read out of a run's list (pool.rs page classes)
    tail: Option<(u32, u32)>,
    tail_place: u32,
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
            counted: Vec::new(),
            heads: std::collections::VecDeque::new(),
            clock: 0,
            shared_after: shared_after.max(1),
            grid: if grid == 0 { GRID } else { grid },
            next_id: 1,
            pin: Vec::new(),
            index_errors: 0,
            limit_milli: 0,
            credit: 0,
            dedupe: true,
            tail: None,
            tail_place: 0,
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
            policy: Policy::default(),
        };
        if let Some((ix, loaded)) = index {
            for e in loaded {
                if e.slots.iter().all(|&s| s < slots) && t.alloc.claim(&e.slots) {
                    self.next_id = self.next_id.max(e.id + 1);
                    t.entries.push(e);
                }
            }
            // the counts live in the first index
            if !self.tiers.iter().any(|x| x.index.is_some()) {
                for (&k, &c) in &ix.counts {
                    self.counts.insert(k, c);
                    self.clock = self.clock.max(c.1);
                }
            }
            t.index = Some(ix);
        }
        self.tiers.push(t);
    }

    pub fn tiers(&self) -> usize {
        self.tiers.len()
    }

    pub fn set_policy(&mut self, t: usize, p: Policy) {
        self.tiers[t].policy = p;
    }

    /// Runs past R keep their tails: nr ranks share the attention cache past R, and a block holds per_slot logical
    /// pages' replicated parts (the engine's page block layout decides it). A run past R holds its own pages
    /// (pool.rs): its shard tail pages are the ones of its list below `place`, so the store needs it.
    pub fn set_tail(&mut self, nr: u32, per_slot: u32, place: u32) {
        self.tail = (nr > 0 && per_slot > 0).then_some((nr, per_slot));
        self.tail_place = place;
    }

    /// The write limit: the tiers write at most `milli` / 1000 blocks per page of KV computed. 0 = none.
    pub fn set_limit(&mut self, milli: u64) {
        self.limit_milli = milli;
        self.credit = BURST * 1000;
    }

    /// `pages` pages of KV were computed (a grant's prefill, a growth): the write limit earns credit.
    pub(crate) fn earn(&mut self, pages: u64) {
        if self.limit_milli > 0 {
            self.credit = (self.credit + (self.limit_milli * pages).min(i64::MAX as u64 / 2) as i64).min(BURST * 1000);
        }
    }

    fn writable(&self) -> bool {
        self.limit_milli == 0 || self.credit > 0
    }

    fn charge(&mut self, n: usize) {
        if self.limit_milli > 0 {
            self.credit -= n as i64 * 1000;
        }
    }

    pub fn stats(&self, t: usize) -> Option<TierStats> {
        let x = self.tiers.get(t)?;
        Some(TierStats {
            slots: x.slots,
            free_slots: x.alloc.free(),
            entries: x.entries.len() as u32,
            shared: x.entries.iter().filter(|e| e.kind.prefix()).count() as u32,
            block_bytes: x.policy.block_bytes,
            credit: if self.limit_milli > 0 { self.credit / 1000 } else { 0 },
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
        let counted = std::mem::take(&mut self.counted);
        if let Some(ix) = self.tiers.iter_mut().find_map(|t| t.index.as_mut()) {
            if !counted.is_empty() {
                if let Err(e) = ix.add_counts(&counted) {
                    self.index_errors += 1;
                    err = Err(e);
                }
            }
        }
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

    /// Whether `prompt` continues a stored evicted run, or a run the pool let go (heads).
    pub(crate) fn continues_stored(&self, prompt: &[i32]) -> bool {
        self.tiers
            .iter()
            .flat_map(|t| &t.entries)
            .any(|e| e.kind == Kind::Evicted && continues(prompt, &e.st.history, e.st.pf_end.max(0) as usize))
            || self.continues_head(prompt)
    }

    // one pass over the prompt: does a remembered head's hash match the prompt's first rows?
    fn continues_head(&self, prompt: &[i32]) -> bool {
        if self.heads.is_empty() {
            return false;
        }
        let mut want: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        for &(n, h) in &self.heads {
            if n > 0 && n <= prompt.len() {
                want.entry(n).or_default().push(h);
            }
        }
        let mut h = FNV;
        let mut next = want.iter().peekable();
        for (i, &t) in prompt.iter().enumerate() {
            h = (h ^ t as u32 as u64).wrapping_mul(1099511628211);
            while let Some((&n, hs)) = next.peek() {
                if n > i + 1 {
                    break;
                }
                if n == i + 1 && hs.contains(&h) {
                    return true;
                }
                next.next();
            }
        }
        false
    }

    /// A run left the pool (evicted, stored or not): remember its prompt's head for continuations.
    pub(crate) fn forget(&mut self, st: &Finished) {
        let n = (st.pf_end.max(0) as usize).min(st.history.len());
        if !self.counting() || n == 0 {
            return;
        }
        self.heads.push_back((n, head_hash(&st.history[..n])));
        while self.heads.len() > HEADS {
            self.heads.pop_front();
        }
    }

    /// The rows of a system span of `span` tokens that a tier can hold: cut to the grid, 0 when shorter than a cell.
    pub fn span_rows(&self, span: u32, prompt_len: usize) -> u64 {
        let s = (span as u64).min(prompt_len.saturating_sub(1) as u64);
        s / self.grid * self.grid
    }

    /// Counts a request that starts a conversation toward every grid-cell prefix of its prompt, up to its system span
    /// (rows, from span_rows) when it has one. Returns how many prefixes it counted.
    pub(crate) fn count(&mut self, prompt: &[i32], continuing: bool, span: Option<u64>) -> u32 {
        if !self.counting() || continuing || prompt.len() < 2 {
            return 0;
        }
        let end = span.map_or(prompt.len() - 1, |s| s as usize);
        let cells: Vec<usize> = (1..).map(|i| i * self.grid as usize).take_while(|&s| s <= end).collect();
        for &s in &cells {
            self.clock += 1;
            let key = prefix_key(&prompt[..s + 1]);
            let c = self.counts.entry(key).or_insert((0, 0));
            c.0 += 1;
            c.1 = self.clock;
            self.counted.push((key, c.0, c.1));
        }
        while self.counts.len() > COUNTS {
            let (&k, _) = self.counts.iter().min_by_key(|(k, c)| (c.1, **k)).expect("not empty");
            self.counts.remove(&k);
        }
        cells.len() as u32
    }

    /// A retiring run: the longest prefix it holds, within its system span (rows, span_rows) when it has one, that
    /// shared_after conversations started with goes to the first tier that keeps shared prefixes, unless a tier has it.
    /// `rows_max`: the rows whose pages are the same logical pages on every rank (R for a run past R).
    pub(crate) fn promote(&mut self, list: &[u32], st: &Finished, rows_max: u64, span: Option<u64>) {
        if !self.counting() || st.valid <= 0 || st.history.is_empty() {
            return;
        }
        let g = self.grid;
        let top = (st.valid as u64).min(st.history.len() as u64 - 1).min(rows_max).min(list.len() as u64 * PAGE);
        let mut s = top.min(span.unwrap_or(u64::MAX)) / g * g;
        while s >= g {
            let key = prefix_key(&st.history[..s as usize + 1]);
            if self.count_of(key) >= self.shared_after {
                if self.tiers.iter().flat_map(|t| &t.entries).any(|e| e.kind.prefix() && e.key == key) {
                    return;
                }
                let n = pages(s);
                let fin = Finished {
                    history: st.history[..s as usize + 1].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                self.store(Kind::Shared, key, fin, &list[..n], 0, &[]);
                return;
            }
            s -= g;
        }
    }

    /// Stores evicted retained runs, oldest first. below_pages: the pages of a run past R that are alike on every rank.
    /// resident: the token histories of the runs still in the pool (live and retained), for prefix bases.
    pub(crate) fn put_evicted(&mut self, gone: &[Kept], below_pages: usize, resident: &[&[i32]]) {
        for k in gone {
            self.forget(&k.st);
            if k.st.valid <= 0 {
                continue;
            }
            let mut n = pages(k.st.valid as u64).min(k.list.len());
            let mut st = k.st.clone();
            let mut tail = 0;
            if k.exclusive {
                if self.tail.is_some() && n > below_pages {
                    tail = n - below_pages;
                } else {
                    st.valid = st.valid.min((n.min(below_pages) as u64 * PAGE) as i32);
                }
                n = n.min(below_pages);
            }
            if n == 0 {
                continue;
            }
            self.store_tail(Kind::Evicted, 0, st, &k.list[..n], &k.list, k.hits, resident, tail as u32);
        }
    }

    // the longest stored prefix entry that `history` starts with and whose pages fit in `max_pages`: (id, pages)
    fn shared_base(&self, history: &[i32], max_pages: usize) -> Option<(u64, usize)> {
        self.tiers
            .iter()
            .flat_map(|t| &t.entries)
            .filter(|e| e.kind.prefix() && e.slots.len() <= max_pages && history.starts_with(&e.st.history))
            .max_by_key(|e| (e.slots.len(), e.id))
            .map(|e| (e.id, e.slots.len()))
    }

    // the longest grid-cell prefix of `st`, within `max_pages`, that a run still in the pool also holds: its rows
    fn resident_prefix(&self, st: &Finished, max_pages: usize, resident: &[&[i32]]) -> u64 {
        let g = self.grid;
        let top = (st.valid.max(0) as u64).min(st.history.len() as u64 - 1).min(max_pages as u64 * PAGE);
        let mut s = top / g * g;
        while s >= g {
            let n = s as usize + 1;
            if resident.iter().any(|r| r.len() > n && r[..n] == st.history[..n]) {
                return s;
            }
            s -= g;
        }
        0
    }

    fn store(&mut self, kind: Kind, key: u64, st: Finished, pages: &[u32], hits: u32, resident: &[&[i32]]) {
        self.store_tail(kind, key, st, pages, pages, hits, resident, 0)
    }

    // store() for a run past R whose tail is `tail` logical pages after `pages` (the pages below R). `full` is the
    // run's whole list: its tail lives at the list positions past `pages` — the shard pages this rank holds (the
    // (below + j)-th page of the list below place) and the drafter's pages (one per logical tail page)
    #[allow(clippy::too_many_arguments)]
    fn store_tail(
        &mut self,
        kind: Kind,
        key: u64,
        st: Finished,
        pages: &[u32],
        full: &[u32],
        hits: u32,
        resident: &[&[i32]],
        tail: u32,
    ) {
        // a run that starts with a stored prefix stores only the pages after it. Else a prefix another run in the pool
        // holds becomes one, stored once, when a tier takes it
        let rows = st.valid.max(0) as u64;
        if !self.tiers.iter().any(|t| t.keep.takes(kind) && t.admits(kind, rows, hits)) {
            if let Some(t) = self.tiers.iter_mut().find(|t| t.keep.takes(kind)) {
                t.stats.refused += 1;
                t.stats.refused_pages += pages.len() as u64;
            }
            return;
        }
        let mut base = if kind == Kind::Evicted { self.shared_base(&st.history, pages.len()) } else { None };
        if kind == Kind::Evicted && base.is_none() && self.dedupe {
            let s = self.resident_prefix(&st, pages.len(), resident);
            if s > 0 {
                let fin = Finished {
                    history: st.history[..s as usize + 1].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                let key = prefix_key(&fin.history);
                if let Some(id) = self.put(Kind::Prefix, key, fin, &pages[..s.div_ceil(PAGE) as usize], 0) {
                    base = Some((id, s.div_ceil(PAGE) as usize));
                }
            }
        }
        let own = &pages[base.map_or(0, |b| b.1)..];
        let tail = self.tail.filter(|_| tail > 0).map(|(nr, per)| Tail {
            logical: tail,
            own: tail.div_ceil(nr),
            rep: tail.div_ceil(per),
        });
        let n = own.len() + tail.map_or(0, |t| (t.own + t.rep) as usize);
        if n == 0 {
            return;
        }
        let pin = std::mem::take(&mut self.pin);
        self.pin = pin.iter().copied().chain(base.map(|b| b.0)).collect();
        let placed = self.place(0, kind, n, rows, hits);
        self.pin = pin;
        let Some((t, slots)) = placed else { return };
        let tier = t as u32;
        self.io.extend(own.iter().zip(&slots).map(|(&page, &slot)| Io::Spill { tier, page, slot, part: Part::Full }));
        if let Some(x) = tail {
            // this rank's shard: the (below + j)-th page of the run's list below place, its tail pages in owned
            // order; then every logical tail page's replicated parts, at the list's tail positions (the drafter's)
            let (below, per, place) = (pages.len() as u32, self.tail.expect("set").1, self.tail_place);
            let own_slots = &slots[own.len()..own.len() + x.own as usize];
            let rep_slots = &slots[own.len() + x.own as usize..];
            let shard: Vec<u32> = full
                .iter()
                .filter(|&&p| p < place)
                .skip(below as usize)
                .take(x.own as usize)
                .copied()
                .collect();
            self.io.extend(shard.into_iter().zip(own_slots).map(|(page, &slot)| Io::Spill {
                tier,
                page,
                slot,
                part: Part::Own,
            }));
            self.io.extend((0..x.logical).map(|i| Io::Spill {
                tier,
                page: full[(below + i) as usize],
                slot: rep_slots[(i / per) as usize],
                part: Part::Rep(i % per),
            }));
        }
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[t];
        tier.stats.spilled_pages += n as u64;
        tier.entries.push(Entry { id, kind, key, st, slots, base, hits, tail });
        if kind.prefix() && tier.index.is_some() {
            self.commits.push((t, id));
        }
    }

    // a prefix entry of its own (no base): its id, or None when no tier takes it
    fn put(&mut self, kind: Kind, key: u64, st: Finished, pages: &[u32], hits: u32) -> Option<u64> {
        let (t, slots) = self.place(0, kind, pages.len(), st.valid.max(0) as u64, hits)?;
        let tier = t as u32;
        self.io.extend(pages.iter().zip(&slots).map(|(&page, &slot)| Io::Spill { tier, page, slot, part: Part::Full }));
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[t];
        tier.stats.spilled_pages += pages.len() as u64;
        tier.entries.push(Entry { id, kind, key, st, slots, base: None, hits, tail: None });
        if tier.index.is_some() {
            self.commits.push((t, id));
        }
        Some(id)
    }

    /// The first tier from `start` whose keep and policy take the entry (`rows`, `hits`) and which can make room for n
    /// blocks: (tier, slots). The tier's rate credit pays for the writes.
    fn place(&mut self, start: usize, kind: Kind, n: usize, rows: u64, hits: u32) -> Option<(usize, Vec<u64>)> {
        let mut refused = None;
        for t in start..self.tiers.len() {
            if !self.tiers[t].keep.takes(kind) || n as u64 > self.tiers[t].slots {
                continue;
            }
            if !self.tiers[t].admits(kind, rows, hits) || !self.writable() {
                refused.get_or_insert(t);
                continue;
            }
            if self.make_room(t, n) {
                self.charge(n);
                return Some((t, self.tiers[t].alloc.alloc(n).expect("room made")));
            }
        }
        if let Some(t) = refused {
            self.tiers[t].stats.refused += 1;
            self.tiers[t].stats.refused_pages += n as u64;
        }
        None
    }

    fn make_room(&mut self, t: usize, n: usize) -> bool {
        if self.tiers[t].alloc.free() < n as u64 {
            self.tiers[t].stats.full += 1;
        }
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
            if e.kind.prefix() && ix.drop(e.id).is_err() {
                self.index_errors += 1;
            }
        }
        self.commits.retain(|c| *c != (t, e.id));
        // the base of a run on the move must stay while the run finds a place
        let pinned = self.pin.len();
        self.pin.extend(e.base.map(|b| b.0));
        let placed = self.place(t + 1, e.kind, e.slots.len(), e.st.valid.max(0) as u64, e.hits);
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
                if e.kind.prefix() && to.index.is_some() {
                    self.commits.push((u, e.id));
                }
                self.tiers[u].entries.insert(0, Entry { slots: new, ..e.clone() });
                self.tiers[t].stats.moved_out += 1;
                self.tiers[t].stats.moved_out_pages += e.slots.len() as u64;
            }
            None => {
                self.tiers[t].stats.dropped += 1;
                self.tiers[t].stats.dropped_pages += e.slots.len() as u64;
                if e.kind.prefix() {
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
                self.tiers[t].stats.dropped_pages += e.slots.len() as u64;
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

    /// The history of a prefix entry, wherever it is.
    pub(crate) fn base_history(&self, id: u64) -> Option<&[i32]> {
        self.tiers.iter().flat_map(|t| &t.entries).find(|e| e.id == id).map(|e| &e.st.history[..])
    }

    /// A hit's entry, for a grant: an evicted run leaves its tier (its slots stay held until restore() queues the
    /// copies), a shared prefix stays, pinned against this call's placements, and becomes the most recently used.
    pub(crate) fn begin_restore(&mut self, t: usize, id: u64) -> Entry {
        let tier = &mut self.tiers[t];
        let i = tier.entries.iter().position(|e| e.id == id).expect("found by the caller");
        let e = tier.entries.remove(i);
        tier.stats.hits += 1;
        if e.kind.prefix() {
            tier.entries.push(e.clone());
            self.pin.push(id);
        }
        self.pin.extend(e.base.map(|b| b.0));
        e
    }

    /// Queues the copies of a hit's entry into `pages` (its first ones). pool_base: the grant's first base pages came
    /// from a retained run in the pool, so only the entry's own blocks are read. exclusive: the grant is a run past R
    /// (the CP map), the only kind a tail may go back into.
    pub(crate) fn restore(&mut self, t: usize, e: &Entry, pages: &[u32], pool_base: bool, exclusive: bool) {
        let b = e.base.map_or(0, |b| b.1).min(pages.len());
        if let Some((base, _)) = e.base.filter(|_| !pool_base) {
            let (u, slots) = self.locate(base).expect("a base outlives the runs on it");
            let copies: Vec<Io> = slots
                .iter()
                .zip(&pages[..b])
                .map(|(&slot, &page)| Io::Restore { tier: u as u32, slot, page, part: Part::Full })
                .collect();
            self.io.extend(copies);
            self.tiers[u].stats.restored_pages += b as u64;
        }
        let tier = t as u32;
        let below = e.slots.len() - e.tail.map_or(0, |x| (x.own + x.rep) as usize);
        let own = &e.slots[..below];
        let n = own.len().min(pages.len().saturating_sub(b));
        self.io.extend(own.iter().zip(&pages[b..b + n]).map(|(&slot, &page)| Io::Restore {
            tier,
            slot,
            page,
            part: Part::Full,
        }));
        let mut restored = n;
        // a tail comes back whole on every rank (its pages past the hit are written over by the prefill)
        if let (Some(x), Some((_, per)), true) = (e.tail, self.tail, exclusive) {
            let first = (b + below) as u32;
            let place = self.tail_place;
            let own_slots = &e.slots[below..below + x.own as usize];
            let rep_slots = &e.slots[below + x.own as usize..];
            // the grant's shard: the (first + j)-th page of its list below place; the drafter's pages at the list's
            // tail positions
            let shard: Vec<u32> = pages
                .iter()
                .filter(|&&p| p < place)
                .skip(first as usize)
                .take(x.own as usize)
                .copied()
                .collect();
            self.io.extend(shard.iter().zip(own_slots).map(|(&page, &slot)| Io::Restore {
                tier,
                slot,
                page,
                part: Part::Own,
            }));
            // a shorter new request grants fewer pages: the tail blocks past them are dropped (their rows sit past
            // the hit, rewritten by the prefill or never read). The pages below the hit always restore: the grant
            // covers skip's pages, and the entry's tail covers the rest of them.
            let cap = ((pages.len() as u32).saturating_sub(first)).min(x.logical);
            self.io.extend((0..cap).map(|i| Io::Restore {
                tier,
                slot: rep_slots[(i / per) as usize],
                page: pages[(first + i) as usize],
                part: Part::Rep(i % per),
            }));
            restored += shard.len() + cap as usize;
        }
        self.tiers[t].stats.restored_pages += restored as u64;
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
        for (k, c) in &self.counts {
            s += &format!("c {k:x} {} {}\n", c.0, c.1);
        }
        s
    }

    /// Keeps the entries on the lines of `agreed`, in their order, and drops the rest (with their index records).
    pub fn keep_agreed(&mut self, agreed: &str) {
        let want: Vec<&str> = agreed.lines().collect();
        self.counts.retain(|k, c| want.contains(&format!("c {k:x} {} {}", c.0, c.1).as_str()));
        self.clock = self.counts.values().map(|c| c.1).max().unwrap_or(0);
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
        if self.limit_milli > 0 {
            h = mix(h, self.credit as u64);
        }
        for t in &self.tiers {
            h = mix(mix(h, t.slots), t.alloc.free());
            for e in &t.entries {
                h = mix(mix(mix(mix(h, e.id), e.key), e.st.valid as i64 as u64), slots_hash(&e.slots));
                h = mix(mix(h, e.kind as u64), e.hits as u64);
                if let Some((id, n)) = e.base {
                    h = mix(mix(h, id), n as u64);
                }
            }
        }
        for (k, c) in &self.counts {
            h = mix(mix(h, *k), c.0 as u64);
        }
        for &(n, x) in &self.heads {
            h = mix(mix(h, n as u64), x);
        }
        h
    }
}

/// Whether a prompt continues a run: it holds the run's first `n` tokens (its prompt, the rows the run prefilled).
pub(crate) fn continues(prompt: &[i32], run: &[i32], n: usize) -> bool {
    n > 0 && n <= run.len() && prompt.len() >= n && prompt[..n] == run[..n]
}

// FNV-1a of the tokens, without prefix_key's length seed, so one pass gives every prefix's hash
fn head_hash(t: &[i32]) -> u64 {
    t.iter().fold(FNV, |h, &x| (h ^ x as u32 as u64).wrapping_mul(1099511628211))
}

fn slots_hash(s: &[u64]) -> u64 {
    s.iter().fold(FNV, |h, &x| (h ^ x).wrapping_mul(1099511628211))
}

/// A tier's persistent index (PATH/index beside the blocks): a header naming what makes blocks compatible (model,
/// TP, R, page-block layout, size), then records, each [u32 length][u64 checksum][payload]: add (an entry), drop (an
/// id) or count (a prefix count; the first index holds them). A record whose checksum fails ends the log, and the file is cut there. The engine writes an entry's blocks
/// before io_done() appends its add record, and a drop record is synced before the entry's slots are reused, so a torn
/// write never names blocks that hold anything else.
pub struct Index {
    path: PathBuf,
    file: File,
    compat: String,
    records: u64,
    live: BTreeMap<u64, Entry>,
    /// the prefix counts (key -> count, clock), loaded and kept
    pub counts: BTreeMap<u64, (u32, u64)>,
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

// one record: [u32 length][u64 checksum][payload]
fn frame(p: &[u8]) -> Vec<u8> {
    let mut r = Vec::with_capacity(p.len() + 12);
    put_u32(&mut r, p.len() as u32);
    put_u64(&mut r, fnv_bytes(p));
    r.extend_from_slice(p);
    r
}

fn encode_count(k: u64, n: u32, clock: u64) -> Vec<u8> {
    let mut p = vec![3u8];
    put_u64(&mut p, k);
    put_u32(&mut p, n);
    put_u64(&mut p, clock);
    p
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

enum Record {
    Add(Entry),
    Drop(u64),
    Count(u64, u32, u64),
}

fn decode(p: &[u8]) -> Option<Record> {
    let mut c = Cursor(p);
    match c.take(1)?[0] {
        1 => {
            let (id, key) = (c.u64()?, c.u64()?);
            let (valid, written_end, pf_end) = (c.u32()? as i32, c.u32()? as i32, c.u32()? as i32);
            let n = c.u32()? as usize;
            let history = (0..n).map(|_| c.u32().map(|t| t as i32)).collect::<Option<Vec<_>>>()?;
            let n = c.u32()? as usize;
            let slots = (0..n).map(|_| c.u64()).collect::<Option<Vec<_>>>()?;
            Some(Record::Add(Entry {
                id,
                kind: Kind::Shared,
                key,
                st: Finished { history, valid, written_end, pf_end },
                slots,
                base: None,
                hits: 0,
                tail: None,
            }))
        }
        2 => Some(Record::Drop(c.u64()?)),
        3 => Some(Record::Count(c.u64()?, c.u32()?, c.u64()?)),
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
        let mut counts: BTreeMap<u64, (u32, u64)> = BTreeMap::new();
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
                    Some(Record::Add(e)) => {
                        order.retain(|&i| i != e.id);
                        order.push(e.id);
                        live.insert(e.id, e);
                    }
                    Some(Record::Drop(id)) => {
                        live.remove(&id);
                        order.retain(|&i| i != id);
                    }
                    Some(Record::Count(k, n, clock)) => {
                        counts.insert(k, (n, clock));
                    }
                    None => break,
                }
                records += 1;
                good = raw.len() - c.0.len();
            }
        } else if !raw.is_empty() {
            // another model, layout, block format or engine build wrote it: its blocks are not this engine's
            let old = raw.split(|&b| b == b'\n').next().unwrap_or_default();
            eprintln!(
                "# s2 store {}: index is for another engine or layout, starting empty (was: {}; now: {})",
                dir.display(),
                String::from_utf8_lossy(&old[..old.len().min(400)]),
                head.trim_end()
            );
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
        while counts.len() > COUNTS {
            let (&k, _) = counts.iter().min_by_key(|(k, c)| (c.1, **k)).expect("not empty");
            counts.remove(&k);
        }
        let ix = Index { path, file, compat: compat.to_string(), records, live, counts };
        Ok((ix, loaded))
    }

    fn append(&mut self, p: &[u8]) -> io::Result<()> {
        self.file.write_all(&frame(p))?;
        self.file.sync_data()?;
        self.records += 1;
        self.maybe_compact()
    }

    fn maybe_compact(&mut self) -> io::Result<()> {
        if self.records > 4 * (self.live.len() + self.counts.len()) as u64 + 64 {
            self.compact()?;
        }
        Ok(())
    }

    /// Prefix counts (key, count, clock), in one write and one sync.
    pub fn add_counts(&mut self, c: &[(u64, u32, u64)]) -> io::Result<()> {
        let mut r = Vec::new();
        for &(k, n, clock) in c {
            self.counts.insert(k, (n, clock));
            r.extend(frame(&encode_count(k, n, clock)));
        }
        while self.counts.len() > COUNTS {
            let (&k, _) = self.counts.iter().min_by_key(|(k, c)| (c.1, **k)).expect("not empty");
            self.counts.remove(&k);
        }
        self.file.write_all(&r)?;
        self.file.sync_data()?;
        self.records += c.len() as u64;
        self.maybe_compact()
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
            f.write_all(&frame(&encode_add(e)))?;
        }
        for (&k, &(n, clock)) in &self.counts {
            f.write_all(&frame(&encode_count(k, n, clock)))?;
        }
        f.sync_all()?;
        std::fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        self.records = (self.live.len() + self.counts.len()) as u64;
        Ok(())
    }
}
