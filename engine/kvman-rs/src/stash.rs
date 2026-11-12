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
//! - A hit leaves its entry in its tier, the most recently used, and the blocks go back into the grant's first pages.
//!   The grant remembers it (pool.rs Kept::from): its pages below the hit's full pages are byte copies of the entry's.
//! - Write only new blocks (c21, "extend in place"): an evicted run below R is stored up to its storable rows (the
//!   grid cell below min(valid, pf_end), the most a later hit restores, reuse.rs), and a run that continues the stored
//!   run it came from, and holds all of it, replaces that entry: the new entry keeps the old one's base and blocks below
//!   the copied pages and writes only the pages past them. The old entry's drop record is synced first, the new one's
//!   add record commits at io_done.
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
use std::collections::{BTreeMap, HashSet};
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
/// replicated parts (the unsharded bindings), the i-th of a block's per_slot. State(i): the i-th boundary-state block
/// (`0 .. state_blocks`) at `st.valid`, with `page` naming the model's source or destination state handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Full,
    Own,
    Rep(u32),
    State(u32),
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
    /// boundary-state blocks at the end of `slots` (0 for page-only entries and Prefix bases)
    pub state: u32,
    /// the cache identity of the run (pool::Namespace). Prefix kinds and index entries are always stock; an evicted
    /// non-stock run is stored whole (no base) and matches only its own namespace
    pub namespace: crate::pool::Namespace,
}

impl Entry {
    /// The (logical) pages it can restore: its base's, its own below R, and its tail's.
    pub fn pages(&self) -> usize {
        self.below() + self.tail.map_or(0, |t| t.logical as usize)
    }
    /// The pages it can restore below R.
    pub fn below(&self) -> usize {
        self.base.map_or(0, |b| b.1) + self.slots.len()
            - self.state as usize
            - self.tail.map_or(0, |t| (t.own + t.rep) as usize)
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
    /// entries this tier keeps that were bigger than the whole tier (they went to a later tier, or nowhere), and the
    /// pages of the last one: a group lease's head asks to grow by that much (s2/kv_share.h share_want)
    pub oversize: u64,
    pub oversize_last: u64,
}

pub struct Tier {
    keep: Keep,
    alloc: SlotAlloc,
    slots: u64,
    entries: Vec<Entry>, // least recently used first
    index: Option<Index>,
    // the index records every entry, not only prefixes (a KV-node lease that outlives a restart, persist=lease)
    whole: bool,
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
    // index runs whose base is not loaded yet (tier, entry): their slots stay claimed until a later tier brings the
    // base (evict() moves an entry on to a later tier) or loaded() drops them
    waiting: Vec<(usize, Entry)>,
    // index records add_tier dropped (slots past the tier or taken twice); loaded() reports them
    bad_records: usize,
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
    // the pool's reuse grid (Rules::grid, 0 = none): a hit restores at most min(valid, pf_end) cut to it (storable)
    reuse_grid: u64,
    // standalone boundary-state mode (g6kv_stash): reuse rules, trailing state blocks per entry, MTP lookahead, and
    // per-entry access times when a remote group head queries coldest_ages
    rules: crate::reuse::Rules,
    state_blocks: u32,
    state_mtp: bool,
    ages_on: bool,
    seen: std::collections::HashMap<u64, std::time::Instant>,
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
            waiting: Vec::new(),
            bad_records: 0,
            index_errors: 0,
            limit_milli: 0,
            credit: 0,
            dedupe: true,
            tail: None,
            tail_place: 0,
            reuse_grid: grid,
            rules: crate::reuse::Rules::default(),
            state_blocks: 0,
            state_mtp: true,
            ages_on: false,
            seen: std::collections::HashMap::new(),
        }
    }

    /// The rows of a run's finished state a stored entry can give back to a hit other than an exact replay: with a
    /// reuse grid, min(valid, pf_end) cut to it (reuse.rs begin), else valid. The pages past them would be computed
    /// again by the next turn's prefill, so a tier stores no more.
    pub fn storable(&self, st: &Finished) -> u64 {
        let valid = st.valid.max(0) as u64;
        if self.reuse_grid == 0 {
            return valid;
        }
        valid.min(st.pf_end.max(0) as u64) / self.reuse_grid * self.reuse_grid
    }

    /// A tier after the others. With an index, its committed shared prefixes come back (in commit order).
    pub fn add_tier(&mut self, slots: u64, keep: Keep, index: Option<(Index, Vec<Entry>)>) {
        let mut t = Tier {
            keep,
            alloc: SlotAlloc::new(slots),
            slots,
            entries: Vec::new(),
            index: None,
            whole: false,
            stats: TierStats::default(),
            policy: Policy::default(),
        };
        if let Some((mut ix, loaded)) = index {
            // a run stored on a prefix comes back only with its prefix, in this tier or any other (evict() moves an
            // entry on to a LATER tier, so a base can be in a tier added after the run's). The records come in write
            // order, and a compaction or a move can write a base after the runs on it, so the runs wait for passes
            // until their base is in. A run whose base is in no tier so far keeps its slots claimed and waits for the
            // later tiers (retry_waiting); loaded() drops what never finds its base
            let mut ids: HashSet<u64> = self.tiers.iter().flat_map(|x| &x.entries).map(|e| e.id).collect();
            let mut wait: Vec<Entry> = Vec::new();
            for e in loaded {
                if e.slots.iter().all(|&s| s < slots) && t.alloc.claim(&e.slots) {
                    self.next_id = self.next_id.max(e.id + 1);
                    wait.push(e);
                } else {
                    // slots past the tier or taken twice: its record goes, so no later boot reads blocks reused since
                    self.bad_records += 1;
                    if ix.drop(e.id).is_err() {
                        self.index_errors += 1;
                    }
                }
            }
            loop {
                let before = wait.len();
                let mut rest = Vec::new();
                for e in wait {
                    if e.base.is_some_and(|b| !ids.contains(&b.0)) {
                        rest.push(e);
                    } else {
                        ids.insert(e.id);
                        t.entries.push(e);
                    }
                }
                wait = rest;
                if wait.is_empty() || wait.len() == before {
                    break;
                }
            }
            let ti = self.tiers.len();
            self.waiting.extend(wait.into_iter().map(|e| (ti, e)));
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
        self.retry_waiting();
    }

    // the waiting runs whose base is now in some tier join their tier, until a pass places none
    fn retry_waiting(&mut self) {
        let mut ids: HashSet<u64> = self.tiers.iter().flat_map(|x| &x.entries).map(|e| e.id).collect();
        loop {
            let before = self.waiting.len();
            for (ti, e) in std::mem::take(&mut self.waiting) {
                if e.base.is_some_and(|b| !ids.contains(&b.0)) {
                    self.waiting.push((ti, e));
                } else {
                    ids.insert(e.id);
                    self.tiers[ti].entries.push(e);
                }
            }
            if self.waiting.is_empty() || self.waiting.len() == before {
                break;
            }
        }
    }

    /// After the last add_tier: the index runs whose base came back in no tier are dropped (slots freed, records
    /// gone), so no later boot reads blocks reused since.
    /// A tier whose index is not whole (keep=system now) drops the runs an earlier whole index recorded: it never
    /// journals their lifecycle, so a kept one could later point at reused blocks. Returns how many records were
    /// dropped: by these two rules, plus the ones add_tier dropped for slots past the tier or taken twice.
    pub fn loaded(&mut self) -> usize {
        let mut unjournaled = 0;
        for t in self.tiers.iter_mut().filter(|t| t.index.is_some() && !t.whole) {
            let (gone, keep): (Vec<Entry>, Vec<Entry>) = std::mem::take(&mut t.entries).into_iter().partition(|e| !e.kind.prefix());
            t.entries = keep;
            unjournaled += gone.len();
            for e in gone {
                t.alloc.release(&e.slots);
                if t.index.as_mut().is_some_and(|ix| ix.drop(e.id).is_err()) {
                    self.index_errors += 1;
                }
            }
        }
        let gone = std::mem::take(&mut self.waiting);
        for (ti, e) in &gone {
            let t = &mut self.tiers[*ti];
            t.alloc.release(&e.slots);
            if let Some(ix) = &mut t.index {
                if ix.drop(e.id).is_err() {
                    self.index_errors += 1;
                }
            }
        }
        gone.len() + unjournaled + std::mem::take(&mut self.bad_records)
    }

    pub fn tiers(&self) -> usize {
        self.tiers.len()
    }

    /// Tier t's index records every entry it stores (evicted runs too, with their base and tail), so a KV-node lease
    /// that the node held over a restart comes back with its sessions. Prefixes only by default.
    pub fn index_whole(&mut self, t: usize) {
        if let Some(x) = self.tiers.get_mut(t) {
            x.whole = x.index.is_some();
        }
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

    /// Tier t to `slots` blocks, at an event every rank sees (a KV node lease that grew or shrank: REMOTE-TIER.md,
    /// "Sharing a node"). Growing adds free slots at the top. Shrinking evicts the tier's least recently used entries
    /// (each moves on to the next tier that takes it, or is dropped, as when the tier is full) until enough slots are
    /// free, then moves every entry block at or past `slots` into the lowest free slot below it (Io::Move within the
    /// tier). Entries that stored runs are based on go last. The engine runs those copies before the store gives the
    /// top blocks back. Every rank makes the same call,
    /// so every rank queues the same copies. A tier with an index keeps its size (its records name slots): false.
    pub fn resize_tier(&mut self, t: usize, slots: u64) -> bool {
        if t >= self.tiers.len() || self.tiers[t].index.is_some() {
            return false;
        }
        let old = self.tiers[t].slots;
        if slots < old {
            let need = old - slots;
            while self.tiers[t].alloc.free() < need {
                // the least recently used entry that no stored run is based on, before any base: a base's runs go
                // with it, and a base of many runs is a prefix many conversations still use
                let bases: Vec<u64> =
                    self.tiers.iter().flat_map(|x| &x.entries).filter_map(|e| e.base.map(|b| b.0)).collect();
                let free = |e: &Entry| !self.pin.contains(&e.id);
                let es = &self.tiers[t].entries;
                let Some(i) =
                    es.iter().position(|e| free(e) && !bases.contains(&e.id)).or_else(|| es.iter().position(free))
                else {
                    return false;
                };
                let e = self.tiers[t].entries.remove(i);
                self.evict(t, e);
            }
            let mut low = self.tiers[t].alloc.free_below(slots).into_iter();
            let tier = &mut self.tiers[t];
            for e in &mut tier.entries {
                for s in e.slots.iter_mut().filter(|s| **s >= slots) {
                    let to = low.next().expect("as many free slots below the new size as blocks above it");
                    self.io.push(Io::Move { tier: t as u32, slot: *s, to: t as u32, to_slot: to });
                    assert!(tier.alloc.claim(&[to]));
                    tier.alloc.release(&[*s]);
                    *s = to;
                }
            }
        }
        assert!(self.tiers[t].alloc.resize(slots), "the slots past the new size are free");
        self.tiers[t].slots = slots;
        true
    }

    /// Tier t's entries as (id, blocks) in the order a shrink (resize_tier) evicts them: unpinned entries no stored run
    /// is based on, least recently used first, then unpinned bases, then pinned ones (a shrink stops there). What a KV
    /// node's head reports its coldest from (pool.rs coldest_ages), so the ages it reports are those of the blocks a
    /// give costs. resize_tier takes the same rule again after each eviction (a base whose runs all went is no base).
    pub fn give_order(&self, t: usize) -> Vec<(u64, usize)> {
        let Some(x) = self.tiers.get(t) else { return Vec::new() };
        let bases: Vec<u64> = self.tiers.iter().flat_map(|x| &x.entries).filter_map(|e| e.base.map(|b| b.0)).collect();
        let rank = |e: &Entry| if self.pin.contains(&e.id) { 2 } else if bases.contains(&e.id) { 1 } else { 0 };
        let mut v: Vec<(u8, u64, usize)> = x.entries.iter().map(|e| (rank(e), e.id, e.slots.len())).collect();
        v.sort_by_key(|e| e.0); // stable: LRU order within each class
        v.into_iter().map(|(_, id, n)| (id, n)).collect()
    }

    /// Tier t's entries as (id, blocks), least recently used first.
    pub fn lru(&self, t: usize) -> Vec<(u64, usize)> {
        self.tiers.get(t).map_or_else(Vec::new, |x| x.entries.iter().map(|e| (e.id, e.slots.len())).collect())
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
        self.tiers.iter().flat_map(|t| &t.entries).any(|e| {
            e.kind == Kind::Evicted
                && e.namespace == crate::pool::STOCK_NAMESPACE
                && continues(prompt, &e.st.history, e.st.pf_end.max(0) as usize)
        }) || self.continues_head(prompt)
    }

    /// Whether `prompt` continues a stored evicted run of `namespace` (no heads: they are stock-only).
    pub(crate) fn continues_stored_in(&self, prompt: &[i32], namespace: crate::pool::Namespace) -> bool {
        self.tiers.iter().flat_map(|t| &t.entries).any(|e| {
            e.kind == Kind::Evicted
                && e.namespace == namespace
                && continues(prompt, &e.st.history, e.st.pf_end.max(0) as usize)
        })
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
        let mtp = usize::from(self.state_mtp);
        let s = (span as u64).min(prompt_len.saturating_sub(mtp) as u64);
        s / self.grid * self.grid
    }

    /// Counts a request that starts a conversation toward every grid-cell prefix of its prompt, up to its system span
    /// (rows, from span_rows) when it has one. Returns how many prefixes it counted.
    pub(crate) fn count(&mut self, prompt: &[i32], continuing: bool, span: Option<u64>) -> u32 {
        let mtp = usize::from(self.state_mtp);
        if !self.counting() || continuing || prompt.len() < 1 + mtp {
            return 0;
        }
        let max_s = prompt.len() - mtp;
        let end = span.map_or(max_s, |s| (s as usize).min(max_s));
        let cells: Vec<usize> = (1..).map(|i| i * self.grid as usize).take_while(|&s| s <= end).collect();
        for &s in &cells {
            self.clock += 1;
            let key = prefix_key(&prompt[..s + mtp]);
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
            // a non-stock run is stored whole under its namespace: no prefix base, no stock continuation head
            let stock = k.namespace == crate::pool::STOCK_NAMESPACE;
            if stock {
                self.forget(&k.st);
            }
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
            let mut from = None;
            if tail == 0 {
                // the rows a later hit can restore (storable): the pages past them are written again by its prefill
                let cut = self.storable(&st);
                st.valid = cut as i32;
                n = n.min(pages(cut));
                from = k.from;
            }
            if n == 0 {
                continue;
            }
            self.store_tail(Kind::Evicted, 0, st, &k.list[..n], &k.list, k.hits, resident, tail as u32, k.namespace, from);
        }
    }

    // the longest stored prefix entry that `history` starts with and whose pages fit in `max_pages`: (id, pages)
    fn shared_base(&self, history: &[i32], max_pages: usize) -> Option<(u64, usize)> {
        self.tiers
            .iter()
            .flat_map(|t| &t.entries)
            .filter(|e| e.kind.prefix() && e.below() > 0 && e.below() <= max_pages && history.starts_with(&e.st.history))
            .max_by_key(|e| (e.below(), e.id))
            .map(|e| (e.id, e.below()))
    }

    // the longest grid-cell prefix of `st`, within `max_pages`, that a run still in the pool also holds: its rows
    fn resident_prefix(&self, st: &Finished, max_pages: usize, resident: &[&[i32]]) -> u64 {
        let g = self.grid;
        let mtp = usize::from(self.state_mtp);
        let top = (st.valid.max(0) as u64).min(st.history.len().saturating_sub(mtp) as u64).min(max_pages as u64 * PAGE);
        let mut s = top / g * g;
        while s >= g {
            let n = s as usize + mtp;
            let shares = resident.iter().any(|r| {
                let enough = if self.state_mtp { r.len() > n } else { r.len() >= n };
                enough && r[..n] == st.history[..n]
            });
            if shares {
                return s;
            }
            s -= g;
        }
        0
    }

    fn store(&mut self, kind: Kind, key: u64, st: Finished, pages: &[u32], hits: u32, resident: &[&[i32]]) {
        self.store_tail(kind, key, st, pages, pages, hits, resident, 0, crate::pool::STOCK_NAMESPACE, None)
    }

    // store() for a run past R whose tail is `tail` logical pages after `pages` (the pages below R). `full` is the
    // run's whole list: its tail lives at the list positions past `pages` — the shard pages this rank holds (the
    // (below + j)-th page of the list below place) and the drafter's pages (one per logical tail page). `from`: the
    // run's provenance (pool.rs Kept::from), so an evicted run extends the entry it came from (extend)
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
        namespace: crate::pool::Namespace,
        from: Option<(u64, usize)>,
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
        if let (Kind::Evicted, Some((eid, p)), 0) = (kind, from, tail) {
            if self.extend(eid, p, key, &st, pages, hits, namespace) {
                return;
            }
        }
        let stock = namespace == crate::pool::STOCK_NAMESPACE;
        let mut base = if kind == Kind::Evicted && stock { self.shared_base(&st.history, pages.len()) } else { None };
        if kind == Kind::Evicted && stock && base.is_none() && self.dedupe {
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
            let shard: Vec<u32> =
                full.iter().filter(|&&p| p < place).skip(below as usize).take(x.own as usize).copied().collect();
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
        tier.entries.push(Entry { id, kind, key, st, slots, base, hits, tail, state: 0, namespace });
        if tier.index.is_some() && (kind.prefix() || tier.whole) {
            self.commits.push((t, id));
        }
    }

    // Extend in place: the run's pages [0, p) are byte copies of stored run eid's pages [0, p). The run must also
    // match eid's full history; if eid ends mid-page, the first page past p is rewritten. The run replaces eid in
    // eid's tier: eid's base and blocks below p, then new blocks for pages [p, n) only. Returns false for the usual way
    // (eid gone or moved on, pinned by this call's hit, another namespace, a tail, or not covered by the copies). True
    // also when nothing is written: no page past p (eid stays as it is), or its tier refused or had no room (eid stays).
    #[allow(clippy::too_many_arguments)]
    fn extend(
        &mut self,
        eid: u64,
        p: usize,
        key: u64,
        st: &Finished,
        pages: &[u32],
        hits: u32,
        namespace: crate::pool::Namespace,
    ) -> bool {
        let n = pages.len();
        let Some(te) = self.tiers.iter().position(|t| t.entries.iter().any(|e| e.id == eid)) else { return false };
        let e = self.tiers[te].entries.iter().find(|e| e.id == eid).expect("found");
        let b = e.base.map_or(0, |b| b.1);
        let cut = self.storable(&e.st) as usize;
        let m = (cut + 1).min(e.st.history.len());
        let ok = e.kind == Kind::Evicted
            && e.tail.is_none()
            && e.namespace == namespace
            && !self.pin.contains(&eid)
            // A grid-off hit may stop mid-page. Its last partial page must be rewritten,
            // while the full pages before it still come from eid unchanged.
            && p >= cut / PAGE as usize
            && p <= e.below()
            && p >= b
            && p <= n
            && st.history.len() >= m
            && st.history[..m] == e.st.history[..m];
        if !ok {
            return false;
        }
        if n == p {
            return true;
        }
        let base = e.base;
        let pinned = self.pin.len();
        self.pin.push(eid);
        self.pin.extend(base.map(|b| b.0));
        let placed = self.try_place(te, Kind::Evicted, n - p, st.valid.max(0) as u64, hits);
        self.pin.truncate(pinned);
        let new = match placed {
            Ok(slots) => slots,
            Err(refused) => {
                if refused {
                    self.tiers[te].stats.refused += 1;
                    self.tiers[te].stats.refused_pages += (n - p) as u64;
                }
                return true;
            }
        };
        // eid leaves: its drop record reaches the disk before any of its slots is reused (its blocks below p are the new
        // entry's, whose add record commits at io_done; a crash in between loses this session, never names reused blocks)
        let tier = &mut self.tiers[te];
        let i = tier.entries.iter().position(|e| e.id == eid).expect("pinned through the placement");
        let old = tier.entries.remove(i);
        if let (true, Some(ix)) = (tier.whole, &mut tier.index) {
            if ix.drop(eid).is_err() {
                self.index_errors += 1;
            }
        }
        self.commits.retain(|c| *c != (te, eid));
        let keep = p - b;
        // an entry stored past its storable rows (an older engine): those blocks go
        tier.alloc.release(&old.slots[keep..]);
        let t = te as u32;
        self.io.extend(pages[p..].iter().zip(&new).map(|(&page, &slot)| Io::Spill { tier: t, page, slot, part: Part::Full }));
        let mut slots = old.slots[..keep].to_vec();
        slots.extend(new);
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[te];
        tier.stats.spilled_pages += (n - p) as u64;
        tier.entries.push(Entry { id, kind: Kind::Evicted, key, st: st.clone(), slots, base, hits, tail: None, state: 0, namespace });
        if tier.index.is_some() && tier.whole {
            self.commits.push((te, id));
        }
        true
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
        tier.entries.push(Entry {
            id,
            kind,
            key,
            st,
            slots,
            base: None,
            hits,
            tail: None,
            state: 0,
            namespace: crate::pool::STOCK_NAMESPACE,
        });
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
            match self.try_place(t, kind, n, rows, hits) {
                Ok(slots) => return Some((t, slots)),
                Err(true) => {
                    refused.get_or_insert(t);
                }
                Err(false) => {}
            }
        }
        if let Some(t) = refused {
            self.tiers[t].stats.refused += 1;
            self.tiers[t].stats.refused_pages += n as u64;
        }
        None
    }

    // tier t's slots for n blocks, when its keep and policy take the entry and it can make room. Err(true): the policy or
    // the write limit refused it; Err(false): the tier does not keep the kind, is too small, or is full of pinned entries
    fn try_place(&mut self, t: usize, kind: Kind, n: usize, rows: u64, hits: u32) -> Result<Vec<u64>, bool> {
        if !self.tiers[t].keep.takes(kind) {
            return Err(false);
        }
        if n as u64 > self.tiers[t].slots {
            self.tiers[t].stats.oversize += 1;
            self.tiers[t].stats.oversize_last = n as u64;
            return Err(false);
        }
        if !self.tiers[t].admits(kind, rows, hits) || !self.writable() {
            return Err(true);
        }
        if self.make_room(t, n) {
            self.charge(n);
            return Ok(self.tiers[t].alloc.alloc(n).expect("room made"));
        }
        Err(false)
    }

    // evicts tier t's least recently used unpinned entries until n slots are free. Leaves first: the oldest entry no
    // entry uses as its base, so a shared base leaves after the runs on it; else the oldest unpinned one
    fn make_room(&mut self, t: usize, n: usize) -> bool {
        if self.tiers[t].alloc.free() < n as u64 {
            self.tiers[t].stats.full += 1;
        }
        while self.tiers[t].alloc.free() < n as u64 {
            let bases: HashSet<u64> = self.tiers.iter().flat_map(|x| &x.entries).filter_map(|e| e.base.map(|b| b.0)).collect();
            let entries = &self.tiers[t].entries;
            let unpinned = |e: &&Entry| !self.pin.contains(&e.id);
            let pick = entries
                .iter()
                .position(|e| unpinned(&e) && !bases.contains(&e.id))
                .or_else(|| entries.iter().position(|e| unpinned(&e)));
            let Some(i) = pick else { return false };
            let e = self.tiers[t].entries.remove(i);
            self.evict(t, e);
        }
        true
    }

    // e leaves tier t for the next tier that keeps its kind, or is dropped. Tier t frees e's slots after queueing the move.
    fn evict(&mut self, t: usize, e: Entry) {
        let whole = self.tiers[t].whole;
        if let Some(ix) = &mut self.tiers[t].index {
            if (e.kind.prefix() || whole) && ix.drop(e.id).is_err() {
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
                if to.index.is_some() && (e.kind.prefix() || to.whole) {
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
                if let (true, Some(ix)) = (self.tiers[t].whole, &mut self.tiers[t].index) {
                    if ix.drop(e.id).is_err() {
                        self.index_errors += 1;
                    }
                }
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

    /// A hit's entry, for a grant: it stays in its tier (an evicted run too: the grant's run, stored later, extends it
    /// in place), pinned with its base against this call's placements, and becomes the most recently used. No index
    /// record changes, so a crash before the run is stored again still finds the entry.
    pub(crate) fn begin_restore(&mut self, t: usize, id: u64) -> Entry {
        let tier = &mut self.tiers[t];
        let i = tier.entries.iter().position(|e| e.id == id).expect("found by the caller");
        let e = tier.entries.remove(i);
        tier.stats.hits += 1;
        tier.entries.push(e.clone());
        self.pin.push(id);
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
        let below = e.slots.len() - e.state as usize - e.tail.map_or(0, |x| (x.own + x.rep) as usize);
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
            let shard: Vec<u32> =
                pages.iter().filter(|&&p| p < place).skip(first as usize).take(x.own as usize).copied().collect();
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
        self.pin.clear();
    }

    /// Configures trailing boundary-state blocks (`Part::State(0..state_blocks)`) and MTP 1-token lookahead for
    /// standalone `g6kv_stash` engines (slot-slab or hybrid recurrent/rolling-state models).
    pub fn set_state_mode(&mut self, state_blocks: u32, mtp: bool) {
        self.state_blocks = state_blocks;
        self.state_mtp = mtp;
    }

    pub fn state_blocks(&self) -> u32 {
        self.state_blocks
    }

    pub fn state_mtp(&self) -> bool {
        self.state_mtp
    }

    pub fn set_rules(&mut self, rules: crate::reuse::Rules) {
        self.rules = rules;
        self.reuse_grid = rules.grid.max(0) as u64;
        if rules.grid > 0 {
            self.grid = rules.grid as u64;
        }
    }

    pub fn earn_pages(&mut self, pages: u64) {
        self.earn(pages);
    }

    pub fn io_pending_count(&self) -> usize {
        self.io.len()
    }

    pub fn take_pending_io(&mut self) -> Vec<Io> {
        self.take_io()
    }

    pub fn finish_io(&mut self) -> io::Result<()> {
        if self.ages_on {
            let now = std::time::Instant::now();
            let mut live = HashSet::new();
            for tier in &self.tiers {
                for e in &tier.entries {
                    self.seen.entry(e.id).or_insert(now);
                    live.insert(e.id);
                }
            }
            self.seen.retain(|id, _| live.contains(id));
        }
        self.io_done()
    }

    pub fn has_entry(&self, id: u64) -> bool {
        self.tiers.iter().any(|t| t.entries.iter().any(|e| e.id == id))
    }

    /// Drops stored entry `id` when no other entry depends on it as a prefix base.
    pub fn drop_entry(&mut self, id: u64) -> Result<bool, crate::pool::Error> {
        if self.pending() {
            return Err(crate::pool::Error::PendingIo);
        }
        if self.tiers.iter().flat_map(|t| &t.entries).any(|e| e.base.is_some_and(|b| b.0 == id)) {
            return Ok(false);
        }
        let Some(t) = self.tiers.iter().position(|t| t.entries.iter().any(|e| e.id == id)) else {
            return Ok(false);
        };
        let i = self.tiers[t].entries.iter().position(|e| e.id == id).expect("found");
        let e = self.tiers[t].entries.remove(i);
        let whole = self.tiers[t].whole;
        if let Some(ix) = &mut self.tiers[t].index {
            if (e.kind.prefix() || whole) && ix.drop(e.id).is_err() {
                self.index_errors += 1;
            }
        }
        self.commits.retain(|c| *c != (t, e.id));
        self.tiers[t].alloc.release(&e.slots);
        Ok(true)
    }

    /// Counts a stock prompt toward shared-prefix promotion unless it continues a resident or stored conversation.
    pub fn count_admission(
        &mut self,
        prompt: &[i32],
        span: Option<u32>,
        namespace: crate::pool::Namespace,
        resident_continuing: bool,
    ) -> u32 {
        if namespace != crate::pool::STOCK_NAMESPACE {
            return 0;
        }
        let continuing = resident_continuing || self.continues_stored(prompt);
        let s = span.map(|sp| self.span_rows(sp, prompt.len()));
        self.count(prompt, continuing, s)
    }

    /// Finds the best stored hit for `prompt` under `rules` and `namespace`.
    /// When `state_blocks > 0`, page-only `Kind::Prefix` bases (`e.state == 0`) are skipped as standalone hits, and
    /// entries with boundary state (`e.state > 0`) match only at their exact `e.st.valid` boundary.
    pub fn match_state(&self, prompt: &[i32], limit: i32, namespace: crate::pool::Namespace) -> crate::pool::Hit {
        let mut best = crate::pool::Hit::default();
        for t in 0..self.tiers() {
            for e in self.entries(t) {
                if self.state_blocks > 0 && e.state == 0 {
                    continue;
                }
                if e.namespace != namespace {
                    if best.skip == 0 {
                        best.why = crate::reuse::Why::Namespace;
                    }
                    continue;
                }
                let mut b = crate::reuse::begin(&self.rules, &e.st, prompt, limit, self.state_mtp);
                if e.state > 0 && b.skip > 0 && b.skip != e.st.valid {
                    b.skip = 0;
                    b.why = crate::reuse::Why::NoLcp;
                }
                if b.skip <= 0 {
                    if best.skip == 0 && b.lcp >= best.lcp {
                        best.lcp = b.lcp;
                        best.why = b.why;
                    }
                    continue;
                }
                let tier = t as u32;
                if b.skip > best.skip || (best.stored && b.skip == best.skip && tier == best.tier && e.id > best.age) {
                    best = crate::pool::Hit {
                        skip: b.skip,
                        lcp: b.lcp,
                        why: crate::reuse::Why::Stored,
                        age: e.id,
                        stored: true,
                        tier,
                    };
                }
            }
        }
        best
    }

    /// Queues `Io::Restore` copies for stored entry `id` in tier `t` into `pages[0..e.below())` (`Part::Full`) and
    /// `state_handle` (`Part::State(0..e.state)`). Returns `(valid, hits, full_pages_restored)`.
    pub fn restore_entry(
        &mut self,
        t: usize,
        id: u64,
        pages: &[u32],
        state_handle: u32,
    ) -> Result<(i32, u32, usize), crate::pool::Error> {
        if self.pending() {
            return Err(crate::pool::Error::PendingIo);
        }
        let Some(found) = self.find(t, id) else {
            return Err(crate::pool::Error::NotRetained);
        };
        if self.state_blocks > 0 && found.state == 0 {
            return Err(crate::pool::Error::NoHit);
        }
        if pages.len() < found.below() {
            return Err(crate::pool::Error::BadArgument);
        }
        let e = self.begin_restore(t, id);
        if self.ages_on {
            let now = std::time::Instant::now();
            self.seen.insert(id, now);
            if let Some((b, _)) = e.base {
                self.seen.insert(b, now);
            }
        }
        let full = e.below();
        let b = e.base.map_or(0, |b| b.1);
        if let Some((base, _)) = e.base {
            let (u, slots) = self.locate(base).expect("a base outlives the runs on it");
            let copies: Vec<Io> = slots[..b]
                .iter()
                .zip(&pages[..b])
                .map(|(&slot, &page)| Io::Restore { tier: u as u32, slot, page, part: Part::Full })
                .collect();
            self.io.extend(copies);
            self.tiers[u].stats.restored_pages += b as u64;
        }
        let own_pages = full - b;
        let tier = t as u32;
        self.io.extend(e.slots[..own_pages].iter().zip(&pages[b..full]).map(|(&slot, &page)| Io::Restore {
            tier,
            slot,
            page,
            part: Part::Full,
        }));
        let st_slots = &e.slots[own_pages..own_pages + e.state as usize];
        self.io.extend(st_slots.iter().enumerate().map(|(i, &slot)| Io::Restore {
            tier,
            slot,
            page: state_handle,
            part: Part::State(i as u32),
        }));
        self.tiers[t].stats.restored_pages += (own_pages + e.state as usize) as u64;
        self.pin.clear();
        Ok((e.st.valid, e.hits, full))
    }

    fn sole_prefix_in_tier(&self, te: usize, bid: u64, only_dep: u64, keep_id: Option<u64>) -> bool {
        if keep_id == Some(bid) || self.pin.contains(&bid) {
            return false;
        }
        let Some(be) = self.tiers[te].entries.iter().find(|e| e.id == bid) else {
            return false;
        };
        if be.kind != Kind::Prefix || be.state != 0 || be.base.is_some() {
            return false;
        }
        !self
            .tiers
            .iter()
            .flat_map(|t| &t.entries)
            .any(|e| e.id != only_dep && e.base.is_some_and(|b| b.0 == bid))
    }

    fn pin_with_keep(&mut self, extra: &[u64], keep_id: Option<u64>) -> Vec<u64> {
        let saved = std::mem::take(&mut self.pin);
        let mut next = saved.clone();
        next.extend_from_slice(extra);
        if let Some(kid) = keep_id {
            next.push(kid);
            if let Some(kb) = self.tiers.iter().flat_map(|t| &t.entries).find(|e| e.id == kid).and_then(|e| e.base) {
                next.push(kb.0);
            }
        }
        self.pin = next;
        saved
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_state(
        &mut self,
        eid: u64,
        p: usize,
        st: &Finished,
        pages: &[u32],
        state_handle: u32,
        hits: u32,
        namespace: crate::pool::Namespace,
        keep_id: Option<u64>,
    ) -> Option<Option<u64>> {
        let n = pages.len();
        let Some(te) = self.tiers.iter().position(|t| t.entries.iter().any(|e| e.id == eid)) else {
            return None;
        };
        let e = self.tiers[te].entries.iter().find(|e| e.id == eid).expect("found");
        let absorb = e.base.is_some_and(|(bid, b)| {
            self.sole_prefix_in_tier(te, bid, eid, keep_id)
                && self.tiers[te].entries.iter().find(|x| x.id == bid).is_some_and(|x| x.slots.len() == b)
        });
        let b = if absorb { 0 } else { e.base.map_or(0, |b| b.1) };
        let cut = if self.state_blocks > 0 { e.st.valid.max(0) as usize } else { self.storable(&e.st) as usize };
        let m = (cut + usize::from(self.state_mtp)).min(e.st.history.len());
        let ok = e.kind == Kind::Evicted
            && e.tail.is_none()
            && e.namespace == namespace
            && !self.pin.contains(&eid)
            && keep_id != Some(eid)
            && p >= cut / PAGE as usize
            && p <= e.below()
            && p >= b
            && p <= n
            && st.valid >= e.st.valid
            && st.history.len() >= m
            && st.history[..m] == e.st.history[..m];
        if !ok {
            return None;
        }
        let need = (n - p) + self.state_blocks as usize;
        if need == 0 && st.valid == e.st.valid {
            return Some(Some(eid));
        }
        let base = if absorb { None } else { e.base };
        let orig_base_id = e.base.map(|x| x.0);
        let mut extra = vec![eid];
        if let Some(bid) = orig_base_id {
            extra.push(bid);
        }
        let saved_pin = self.pin_with_keep(&extra, keep_id);
        let placed = self.try_place(te, Kind::Evicted, need, st.valid.max(0) as u64, hits);
        self.pin = saved_pin;
        let new = match placed {
            Ok(slots) => slots,
            Err(refused) => {
                if refused {
                    self.tiers[te].stats.refused += 1;
                    self.tiers[te].stats.refused_pages += need as u64;
                }
                return Some(Some(eid));
            }
        };
        let mut prefix_slots = Vec::new();
        if absorb {
            let bid = orig_base_id.expect("absorb has base");
            let tier = &mut self.tiers[te];
            let bi = tier.entries.iter().position(|x| x.id == bid).expect("pinned");
            let b_old = tier.entries.remove(bi);
            if let Some(ix) = &mut tier.index {
                if ix.drop(bid).is_err() {
                    self.index_errors += 1;
                }
            }
            self.commits.retain(|c| *c != (te, bid));
            prefix_slots = b_old.slots;
        }
        let tier = &mut self.tiers[te];
        let i = tier.entries.iter().position(|x| x.id == eid).expect("pinned");
        let old = tier.entries.remove(i);
        if let (true, Some(ix)) = (tier.whole, &mut tier.index) {
            if ix.drop(eid).is_err() {
                self.index_errors += 1;
            }
        }
        self.commits.retain(|c| *c != (te, eid));
        let old_b = old.base.map_or(0, |x| x.1);
        let keep_own = p - old_b;
        tier.alloc.release(&old.slots[keep_own..]);
        let t = te as u32;
        let new_pages = n - p;
        self.io.extend(pages[p..].iter().zip(&new[..new_pages]).map(|(&page, &slot)| Io::Spill {
            tier: t,
            page,
            slot,
            part: Part::Full,
        }));
        self.io.extend(new[new_pages..].iter().enumerate().map(|(k, &slot)| Io::Spill {
            tier: t,
            page: state_handle,
            slot,
            part: Part::State(k as u32),
        }));
        let mut slots = prefix_slots;
        slots.extend_from_slice(&old.slots[..keep_own]);
        slots.extend(new);
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[te];
        tier.stats.spilled_pages += need as u64;
        tier.entries.push(Entry {
            id,
            kind: Kind::Evicted,
            key: 0,
            st: st.clone(),
            slots,
            base,
            hits,
            tail: None,
            state: self.state_blocks,
            namespace,
        });
        if tier.index.is_some() && tier.whole {
            self.commits.push((te, id));
        }
        Some(Some(id))
    }

    #[allow(clippy::too_many_arguments)]
    fn sibling_share_state(
        &mut self,
        eid: u64,
        p: usize,
        st: &Finished,
        pages: &[u32],
        state_handle: u32,
        hits: u32,
        keep_id: Option<u64>,
    ) -> Option<Option<u64>> {
        let n = pages.len();
        if p == 0 {
            return None;
        }
        let te = self.tiers.iter().position(|t| t.entries.iter().any(|e| e.id == eid))?;
        let e = self.tiers[te].entries.iter().find(|e| e.id == eid).expect("found");
        let cut = e.st.valid.max(0) as usize;
        let m = (cut + usize::from(self.state_mtp)).min(e.st.history.len());
        let ok = e.kind == Kind::Evicted
            && e.tail.is_none()
            && e.namespace == crate::pool::STOCK_NAMESPACE
            && p == e.below()
            && p <= n
            && st.valid >= e.st.valid
            && st.history.len() >= m
            && st.history[..m] == e.st.history[..m];
        if !ok {
            return None;
        }
        let existing_base = e.base;
        // When eid already holds all p full pages in its base, or has no base (all p full pages in eid.slots[..p]),
        // share all p full pages with zero block copies. When eid has a partial base (b < p), reuse that base.
        let share_pages = match existing_base {
            Some((_, b)) if b < p => b,
            _ => p,
        };
        let need = (n - share_pages) + self.state_blocks as usize;
        let mut extra = vec![eid];
        if let Some((bid, _)) = existing_base {
            extra.push(bid);
        }
        let saved_pin = self.pin_with_keep(&extra, keep_id);
        let placed = self.try_place(te, Kind::Evicted, need, st.valid.max(0) as u64, hits);
        self.pin = saved_pin;
        let new = match placed {
            Ok(slots) => slots,
            Err(refused) => {
                if refused {
                    self.tiers[te].stats.refused += 1;
                    self.tiers[te].stats.refused_pages += need as u64;
                }
                return Some(None);
            }
        };
        let base = match existing_base {
            Some((bid, b)) => Some((bid, b)),
            None => {
                let s = p as u64 * PAGE;
                let hlen = (s as usize + usize::from(self.state_mtp)).min(st.history.len());
                let fin = Finished {
                    history: st.history[..hlen].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                let bkey = prefix_key(&fin.history);
                let bid = self.next_id;
                self.next_id += 1;
                let tier = &mut self.tiers[te];
                let ei = tier.entries.iter().position(|x| x.id == eid).expect("pinned");
                let b_slots = tier.entries[ei].slots[..p].to_vec();
                tier.entries[ei].slots.drain(..p);
                tier.entries[ei].base = Some((bid, p));
                let updated_e = tier.entries[ei].clone();
                let b_entry = Entry {
                    id: bid,
                    kind: Kind::Prefix,
                    key: bkey,
                    st: fin,
                    slots: b_slots,
                    base: None,
                    hits: 0,
                    tail: None,
                    state: 0,
                    namespace: crate::pool::STOCK_NAMESPACE,
                };
                tier.entries.insert(ei, b_entry.clone());
                if let Some(ix) = &mut tier.index {
                    if ix.add(&b_entry).is_err() {
                        self.index_errors += 1;
                    }
                    if tier.whole && ix.add(&updated_e).is_err() {
                        self.index_errors += 1;
                    }
                }
                Some((bid, p))
            }
        };
        let t = te as u32;
        let new_pages = n - share_pages;
        self.io.extend(pages[share_pages..].iter().zip(&new[..new_pages]).map(|(&page, &slot)| Io::Spill {
            tier: t,
            page,
            slot,
            part: Part::Full,
        }));
        self.io.extend(new[new_pages..].iter().enumerate().map(|(k, &slot)| Io::Spill {
            tier: t,
            page: state_handle,
            slot,
            part: Part::State(k as u32),
        }));
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[te];
        tier.stats.spilled_pages += need as u64;
        tier.entries.push(Entry {
            id,
            kind: Kind::Evicted,
            key: 0,
            st: st.clone(),
            slots: new,
            base,
            hits,
            tail: None,
            state: self.state_blocks,
            namespace: crate::pool::STOCK_NAMESPACE,
        });
        if tier.index.is_some() && tier.whole {
            self.commits.push((te, id));
        }
        Some(Some(id))
    }

    /// Captures a session snapshot (`Kind::Evicted`) into the tiers.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_entry(
        &mut self,
        mut st: Finished,
        pages: &[u32],
        state_handle: u32,
        hits: u32,
        namespace: crate::pool::Namespace,
        from: Option<(u64, usize)>,
        keep_from: bool,
        keep_id: Option<u64>,
        resident: &[&[i32]],
    ) -> Result<Option<u64>, crate::pool::Error> {
        if self.pending() {
            return Err(crate::pool::Error::PendingIo);
        }
        let stock = namespace == crate::pool::STOCK_NAMESPACE;
        if stock {
            self.forget(&st);
        }
        if st.valid <= 0 {
            return Ok(None);
        }
        if self.state_blocks == 0 {
            let cut = self.storable(&st);
            st.valid = cut as i32;
            if st.valid <= 0 {
                return Ok(None);
            }
        }
        let full_pages = if self.state_blocks > 0 {
            (st.valid as u64 / PAGE) as usize
        } else {
            self::pages(st.valid as u64)
        };
        if pages.len() < full_pages {
            return Err(crate::pool::Error::BadArgument);
        }
        let pages = &pages[..full_pages];
        let rows = st.valid as u64;
        if !self.tiers.iter().any(|t| t.keep.takes(Kind::Evicted) && t.admits(Kind::Evicted, rows, hits)) {
            if let Some(t) = self.tiers.iter_mut().find(|t| t.keep.takes(Kind::Evicted)) {
                t.stats.refused += 1;
                t.stats.refused_pages += (pages.len() + self.state_blocks as usize) as u64;
            }
            return Ok(None);
        }
        if let Some((eid, p)) = from {
            if keep_from && stock && self.state_blocks > 0 {
                if let Some(res) = self.sibling_share_state(eid, p, &st, pages, state_handle, hits, keep_id) {
                    return Ok(res);
                }
            } else if !keep_from {
                if let Some(res) = self.extend_state(eid, p, &st, pages, state_handle, hits, namespace, keep_id) {
                    return Ok(res);
                }
            }
        }
        let mut base = if stock { self.shared_base(&st.history, pages.len()) } else { None };
        if stock && base.is_none() && self.dedupe {
            let s = self.resident_prefix(&st, pages.len(), resident);
            if s > 0 && s.is_multiple_of(PAGE) {
                let hlen = (s as usize + usize::from(self.state_mtp)).min(st.history.len());
                let fin = Finished {
                    history: st.history[..hlen].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                let key = prefix_key(&fin.history);
                let b_pages = (s / PAGE) as usize;
                let saved_pin = self.pin_with_keep(&[], keep_id);
                let put_id = self.put(Kind::Prefix, key, fin, &pages[..b_pages], 0);
                self.pin = saved_pin;
                if let Some(id) = put_id {
                    base = Some((id, b_pages));
                }
            }
        }
        let own = &pages[base.map_or(0, |b| b.1)..];
        let need = own.len() + self.state_blocks as usize;
        if need == 0 {
            return Ok(None);
        }
        let mut extra = Vec::new();
        if let Some((bid, _)) = base {
            extra.push(bid);
        }
        let saved_pin = self.pin_with_keep(&extra, keep_id);
        let placed = self.place(0, Kind::Evicted, need, rows, hits);
        self.pin = saved_pin;
        let Some((t, slots)) = placed else { return Ok(None) };
        let tier = t as u32;
        self.io.extend(own.iter().zip(&slots[..own.len()]).map(|(&page, &slot)| Io::Spill {
            tier,
            page,
            slot,
            part: Part::Full,
        }));
        self.io.extend(slots[own.len()..].iter().enumerate().map(|(k, &slot)| Io::Spill {
            tier,
            page: state_handle,
            slot,
            part: Part::State(k as u32),
        }));
        let id = self.next_id;
        self.next_id += 1;
        let tier = &mut self.tiers[t];
        tier.stats.spilled_pages += need as u64;
        tier.entries.push(Entry {
            id,
            kind: Kind::Evicted,
            key: 0,
            st,
            slots,
            base,
            hits,
            tail: None,
            state: self.state_blocks,
            namespace,
        });
        if tier.index.is_some() && tier.whole {
            self.commits.push((t, id));
        }
        Ok(Some(id))
    }

    /// Promotes a shared system prefix (`shared_after` conversations) in standalone `g6kv_stash` mode.
    /// When `state_blocks > 0`, an exact boundary (`st.valid == s`) promotes `Kind::Shared` with boundary state so any
    /// matching prompt restores it standalone; a longer run (`st.valid > s`) promotes `Kind::Prefix` (`state = 0`, full
    /// pages only) so later conversations deduplicate those prefix pages without reading stale rolling state.
    pub fn promote_state(
        &mut self,
        st: &Finished,
        pages: &[u32],
        state_handle: u32,
        rows_max: u64,
        span: Option<u32>,
        keep_id: Option<u64>,
    ) -> Result<(), crate::pool::Error> {
        if self.pending() {
            return Err(crate::pool::Error::PendingIo);
        }
        let span_rows = span.map(|sp| self.span_rows(sp, st.history.len()));
        if self.state_blocks == 0 {
            let saved_pin = self.pin_with_keep(&[], keep_id);
            self.promote(pages, st, rows_max, span_rows);
            self.pin = saved_pin;
            return Ok(());
        }
        if !self.counting() || st.valid <= 0 || st.history.is_empty() {
            return Ok(());
        }
        let g = self.grid;
        let mtp = usize::from(self.state_mtp);
        let top = (st.valid as u64).min(st.history.len().saturating_sub(mtp) as u64).min(rows_max).min(pages.len() as u64 * PAGE);
        let mut s = top.min(span_rows.unwrap_or(u64::MAX)) / g * g;
        while s >= g {
            if !s.is_multiple_of(PAGE) {
                s -= g;
                continue;
            }
            let key = prefix_key(&st.history[..s as usize + mtp]);
            if self.count_of(key) >= self.shared_after {
                let exact = st.valid as u64 == s;
                let already = self.tiers.iter().flat_map(|t| &t.entries).any(|e| {
                    e.kind.prefix() && e.key == key && (!exact || e.state > 0)
                });
                if already {
                    return Ok(());
                }
                let n_pages = (s / PAGE) as usize;
                let fin = Finished {
                    history: st.history[..s as usize + mtp].to_vec(),
                    valid: s as i32,
                    written_end: s as i32,
                    pf_end: s as i32,
                };
                if exact {
                    let need = n_pages + self.state_blocks as usize;
                    let saved_pin = self.pin_with_keep(&[], keep_id);
                    let placed = self.place(0, Kind::Shared, need, s, 0);
                    self.pin = saved_pin;
                    if let Some((t, slots)) = placed {
                        let tier = t as u32;
                        self.io.extend(pages[..n_pages].iter().zip(&slots[..n_pages]).map(|(&page, &slot)| Io::Spill {
                            tier,
                            page,
                            slot,
                            part: Part::Full,
                        }));
                        self.io.extend(slots[n_pages..].iter().enumerate().map(|(k, &slot)| Io::Spill {
                            tier,
                            page: state_handle,
                            slot,
                            part: Part::State(k as u32),
                        }));
                        let id = self.next_id;
                        self.next_id += 1;
                        let tier = &mut self.tiers[t];
                        tier.stats.spilled_pages += need as u64;
                        tier.entries.push(Entry {
                            id,
                            kind: Kind::Shared,
                            key,
                            st: fin,
                            slots,
                            base: None,
                            hits: 0,
                            tail: None,
                            state: self.state_blocks,
                            namespace: crate::pool::STOCK_NAMESPACE,
                        });
                        if tier.index.is_some() {
                            self.commits.push((t, id));
                        }
                    }
                } else {
                    let saved_pin = self.pin_with_keep(&[], keep_id);
                    let _ = self.put(Kind::Prefix, key, fin, &pages[..n_pages], 0);
                    self.pin = saved_pin;
                }
                return Ok(());
            }
            s -= g;
        }
        Ok(())
    }

    pub fn digest(&self) -> u64 {
        fn mix64(mut h: u64, v: u64) -> u64 {
            for i in 0..8 {
                h ^= (v >> (8 * i)) & 0xff;
                h = h.wrapping_mul(1099511628211);
            }
            h
        }
        self.fold(FNV, mix64)
    }

    pub fn coldest_ages(&mut self, t: usize, per_seg: u64, most: usize) -> Vec<u64> {
        if t >= self.tiers() {
            return Vec::new();
        }
        self.ages_on = true;
        let now = std::time::Instant::now();
        let mut live = HashSet::new();
        for tier in &self.tiers {
            for e in &tier.entries {
                self.seen.entry(e.id).or_insert(now);
                live.insert(e.id);
            }
        }
        self.seen.retain(|id, _| live.contains(id));
        let mut at = self.seen.clone();
        for tier in &self.tiers {
            for e in &tier.entries {
                if let Some((b, _)) = e.base {
                    let u = self.seen[&e.id];
                    if let Some(v) = at.get_mut(&b) {
                        *v = (*v).max(u);
                    }
                }
            }
        }
        crate::remote::coldest_ages(
            &self.give_order(t),
            |id| at.get(&id).map_or(0, |u| now.duration_since(*u).as_millis() as u64),
            per_seg,
            self.stats(t).map_or(0, |x| x.free_slots),
            most,
        )
    }

    fn summary_line(t: usize, e: &Entry) -> String {
        if e.state == 0 {
            format!("{t} {} {:x} {} {:x}", e.id, e.key, e.st.valid, slots_hash(&e.slots))
        } else {
            format!("{t} {} {:x} {} {:x} s{}", e.id, e.key, e.st.valid, slots_hash(&e.slots), e.state)
        }
    }

    /// The entries loaded from indexes, one line each. Every rank compares these lines at boot.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for (t, tier) in self.tiers.iter().enumerate() {
            for e in &tier.entries {
                s += &Self::summary_line(t, e);
                s.push('\n');
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
            let mut kept: Vec<(usize, Entry)> = Vec::new();
            for e in all {
                let line = Self::summary_line(t, &e);
                match want.iter().position(|w| *w == line) {
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
        // Agreement on a dependent record alone is insufficient: its base must also survive.
        loop {
            let ids: HashSet<u64> = self.tiers.iter().flat_map(|t| &t.entries).map(|e| e.id).collect();
            let mut removed = false;
            for tier in &mut self.tiers {
                let all = std::mem::take(&mut tier.entries);
                for e in all {
                    if e.base.is_some_and(|b| !ids.contains(&b.0)) {
                        if let Some(ix) = &mut tier.index {
                            if ix.drop(e.id).is_err() { self.index_errors += 1; }
                        }
                        tier.alloc.release(&e.slots);
                        removed = true;
                    } else {
                        tier.entries.push(e);
                    }
                }
            }
            if !removed { break; }
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
                if e.state != 0 {
                    h = mix(h, e.state as u64);
                }
                // stock entries keep the pre-namespace digest; a non-stock one folds in every namespace byte
                if e.namespace != crate::pool::STOCK_NAMESPACE {
                    for c in e.namespace.chunks(8) {
                        h = mix(h, c.iter().fold(0u64, |x, &b| x << 8 | b as u64));
                    }
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

const MAGIC: &str = "g6kv-index 2";

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

// record 4: every field of an entry (an index that keeps evicted runs, Stash::index_whole).
// record 5: record 4 plus trailing u32 boundary-state block count (`e.state > 0`).
fn encode_entry(e: &Entry) -> Vec<u8> {
    let mut p = encode_add(e);
    p[0] = if e.state == 0 { 4 } else { 5 };
    p.push(match e.kind {
        Kind::Evicted => 0,
        Kind::Shared => 1,
        Kind::Prefix => 2,
    });
    let (b, bp) = e.base.map_or((u64::MAX, 0), |b| (b.0, b.1 as u64));
    put_u64(&mut p, b);
    put_u64(&mut p, bp);
    put_u32(&mut p, e.hits);
    let t = e.tail.unwrap_or(Tail { logical: 0, own: 0, rep: 0 });
    for x in [t.logical, t.own, t.rep] {
        put_u32(&mut p, x);
    }
    p.extend_from_slice(&e.namespace);
    if e.state != 0 {
        put_u32(&mut p, e.state);
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
    let tag = c.take(1)?[0];
    match tag {
        1 | 4 | 5 => {
            let (id, key) = (c.u64()?, c.u64()?);
            let (valid, written_end, pf_end) = (c.u32()? as i32, c.u32()? as i32, c.u32()? as i32);
            let n = c.u32()? as usize;
            let history = (0..n).map(|_| c.u32().map(|t| t as i32)).collect::<Option<Vec<_>>>()?;
            let n = c.u32()? as usize;
            let slots = (0..n).map(|_| c.u64()).collect::<Option<Vec<_>>>()?;
            let mut e = Entry {
                id,
                kind: Kind::Shared,
                key,
                st: Finished { history, valid, written_end, pf_end },
                slots,
                base: None,
                hits: 0,
                tail: None,
                state: 0,
                namespace: crate::pool::STOCK_NAMESPACE,
            };
            if tag == 4 || tag == 5 {
                e.kind = match c.take(1)?[0] {
                    0 => Kind::Evicted,
                    1 => Kind::Shared,
                    2 => Kind::Prefix,
                    _ => return None,
                };
                let (b, bp, hits) = (c.u64()?, c.u64()?, c.u32()?);
                e.base = (b != u64::MAX).then_some((b, bp as usize));
                e.hits = hits;
                let (logical, own, rep) = (c.u32()?, c.u32()?, c.u32()?);
                // a tail always has logical pages (own and rep count parts of them), so logical 0 is no tail
                e.tail = (logical > 0).then_some(Tail { logical, own, rep });
                e.namespace.copy_from_slice(c.take(crate::pool::NAMESPACE_BYTES)?);
                if tag == 5 {
                    let state = c.u32()?;
                    let tail_slots = e.tail.map_or(0, |t| (t.own + t.rep) as usize);
                    if state == 0 || e.slots.len() < state as usize + tail_slots {
                        return None;
                    }
                    e.state = state;
                }
            }
            Some(Record::Add(e))
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
        self.append(&encode_entry(e))
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
            f.write_all(&frame(&encode_entry(e)))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::STOCK_NAMESPACE;

    #[test]
    fn extend_journals_whole_index_entries_across_namespaces_and_skips_prefix_only() {
        for whole in [false, true] {
            for (case, namespace) in [STOCK_NAMESPACE, [1; crate::pool::NAMESPACE_BYTES]].into_iter().enumerate() {
                let dir = std::env::temp_dir().join(format!("g6kv-extend-namespace-{}-{case}-{whole}", std::process::id()));
                let mut stash = Stash::new(3, PAGE);
                stash.add_tier(8, Keep::All, Some(Index::open(&dir, "extend-namespace").unwrap()));
                if whole {
                    stash.index_whole(0);
                }
                let first = Finished { history: (0..65).collect(), valid: 64, written_end: 64, pf_end: 64 };
                stash.store_tail(Kind::Evicted, 0, first, &[0], &[0], 1, &[], 0, namespace, None);
                let old = stash.entries(0)[0].clone();
                assert_eq!(stash.commits, if whole { vec![(0, old.id)] } else { vec![] });
                assert_eq!(stash.take_io().len(), 1);
                stash.io_done().unwrap();
                let loaded = Index::open(&dir, "extend-namespace").unwrap().1;
                assert_eq!(loaded, if whole { vec![old.clone()] } else { vec![] });
                assert!(loaded.iter().all(|e| e.namespace == namespace));

                let next = Finished { history: (0..129).collect(), valid: 128, written_end: 128, pf_end: 128 };
                assert!(stash.extend(old.id, 1, 0, &next, &[0, 1], 1, namespace));
                let extended = stash.entries(0)[0].clone();
                assert_ne!(extended.id, old.id);
                assert_eq!(extended.namespace, namespace);
                assert_eq!(extended.slots.len(), 2);
                assert_eq!(extended.slots[0], old.slots[0]);
                assert_eq!(stash.commits, if whole { vec![(0, extended.id)] } else { vec![] });
                assert_eq!(stash.take_io(), vec![Io::Spill { tier: 0, page: 1, slot: extended.slots[1], part: Part::Full }]);
                stash.io_done().unwrap();
                drop(stash);
                let (index, loaded) = Index::open(&dir, "extend-namespace").unwrap();
                assert!(loaded.iter().all(|e| e.id != old.id && e.namespace == namespace && e.slots.len() == 2));
                assert_eq!(loaded, if whole { vec![extended] } else { vec![] });
                drop(index);
                std::fs::remove_file(dir.join("index")).unwrap();
                std::fs::remove_dir(dir).unwrap();
            }
        }
    }

    #[test]
    fn old_index_version_starts_empty() {
        let dir = std::env::temp_dir().join(format!("g6kv-old-index-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index"), b"g6kv-index 1 same-layout\n").unwrap();
        let (ix, entries) = Index::open(&dir, "same-layout").unwrap();
        assert!(entries.is_empty());
        assert!(std::fs::read(dir.join("index")).unwrap().starts_with(b"g6kv-index 2 "));
        drop(ix);
        std::fs::remove_file(dir.join("index")).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn boundary_state_exact_match_prompt_end_fallback_and_delta_extend() {
        let dir = std::env::temp_dir().join(format!("g6kv-state-exact-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = Stash::new(2, 128);
        s.set_state_mode(2, false);
        s.set_rules(crate::reuse::Rules { strict: false, noend: true, grid: 0 });
        s.add_tier(32, Keep::All, Some(Index::open(&dir, "ds41-v1").unwrap()));
        s.index_whole(0);

        // Sub-64 token prompt (37 tokens -> 0 full pages, 2 state blocks).
        let h37: Vec<i32> = (1000..1037).collect();
        let id37 = s
            .capture_entry(
                Finished { history: h37.clone(), valid: 37, written_end: 37, pf_end: 37 },
                &[],
                9,
                1,
                STOCK_NAMESPACE,
                None,
                false,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        let io = s.take_pending_io();
        assert_eq!(io.len(), 2);
        assert!(io.iter().enumerate().all(|(k, x)| matches!(x, Io::Spill { page: 9, part, .. } if *part == Part::State(k as u32))));
        s.finish_io().unwrap();

        // Partial prefix (20 < 37) must NOT match an exact-boundary entry.
        let mut p20 = h37[..20].to_vec();
        p20.extend(5000..5030);
        assert_eq!(s.match_state(&p20, 50, STOCK_NAMESPACE).skip, 0);

        // Exact continuation (lcp >= 37, limit >= 37) matches at 37.
        let mut turn1: Vec<i32> = h37.clone();
        turn1.extend(2000..2163); // 200 tokens total
        let hit = s.match_state(&turn1, turn1.len() as i32 - 1, STOCK_NAMESPACE);
        assert_eq!((hit.skip, hit.age, hit.stored), (37, id37, true));
        let (valid, hits, full) = s.restore_entry(hit.tier as usize, hit.age, &[0, 1, 2], 4).unwrap();
        assert_eq!((valid, hits, full), (37, 1, 0));
        assert_eq!(s.take_pending_io().len(), 2);
        s.finish_io().unwrap();

        // Turn 1 PromptEnd at 200 tokens (3 full pages + 2 state blocks) extends id37.
        let pf_id = s
            .capture_entry(
                Finished { history: turn1.clone(), valid: 200, written_end: 200, pf_end: 200 },
                &[10, 11, 12],
                4,
                2,
                STOCK_NAMESPACE,
                Some((id37, 0)),
                false,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        assert!(!s.has_entry(id37));
        assert_eq!(s.take_pending_io().len(), 5);
        s.finish_io().unwrap();

        // Turn 1 RequestEnd at 250 tokens (still 3 full pages + 2 state blocks) with keep_from=true:
        // shares the 3 full pages via zero-copy Prefix base and spills ONLY the 2 state blocks!
        let mut turn1_end = turn1.clone();
        turn1_end.extend(3000..3050);
        let req_id = s
            .capture_entry(
                Finished { history: turn1_end.clone(), valid: 250, written_end: 250, pf_end: 200 },
                &[10, 11, 12],
                4,
                2,
                STOCK_NAMESPACE,
                Some((pf_id, 3)),
                true,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        let req_io = s.take_pending_io();
        assert_eq!(req_io.len(), 2, "RequestEnd must write 0 duplicate full pages");
        assert!(req_io.iter().all(|x| matches!(x, Io::Spill { part: Part::State(_), .. })));
        s.finish_io().unwrap();
        assert!(s.has_entry(pf_id) && s.has_entry(req_id));

        // Restart from disk: tag-5 entries and their shared Prefix base reload intact.
        let summary_before = s.summary();
        drop(s);
        let mut s2 = Stash::new(2, 128);
        s2.set_state_mode(2, false);
        s2.set_rules(crate::reuse::Rules { strict: false, noend: true, grid: 0 });
        s2.add_tier(32, Keep::All, Some(Index::open(&dir, "ds41-v1").unwrap()));
        s2.index_whole(0);
        assert_eq!(s2.loaded(), 0);
        assert_eq!(s2.summary(), summary_before);
        s2.keep_agreed(&summary_before);
        assert!(s2.has_entry(pf_id) && s2.has_entry(req_id));

        // Reasoning-stripped Turn 2 (matches first 200 tokens, diverges at 200) falls back to PromptEnd (skip = 200)!
        let mut stripped_t2 = turn1.clone();
        stripped_t2.extend(9000..9080);
        let hit_stripped = s2.match_state(&stripped_t2, stripped_t2.len() as i32 - 1, STOCK_NAMESPACE);
        assert_eq!((hit_stripped.skip, hit_stripped.age), (200, pf_id));

        // Unstripped Turn 2 (matches all 250 tokens) hits RequestEnd (skip = 250)!
        let mut full_t2 = turn1_end.clone();
        full_t2.extend(8000..8090); // 340 tokens -> 5 full pages
        let hit_full = s2.match_state(&full_t2, full_t2.len() as i32 - 1, STOCK_NAMESPACE);
        assert_eq!((hit_full.skip, hit_full.age), (250, req_id));
        let (_, _, restored_pages) = s2.restore_entry(0, req_id, &[20, 21, 22], 7).unwrap();
        assert_eq!(restored_pages, 3);
        assert_eq!(s2.take_pending_io().len(), 5); // 3 base pages + 2 state blocks
        s2.finish_io().unwrap();

        // Drop superseded pf_id, then extend req_id to 340 tokens (5 full pages):
        // absorbs sole Prefix base (3 pages untouched!) and writes ONLY 2 new full pages + 2 state blocks!
        assert!(s2.drop_entry(pf_id).unwrap());
        let pf2_id = s2
            .capture_entry(
                Finished { history: full_t2.clone(), valid: 340, written_end: 340, pf_end: 340 },
                &[20, 21, 22, 23, 24],
                7,
                3,
                STOCK_NAMESPACE,
                Some((req_id, 3)),
                false,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        let ext_io = s2.take_pending_io();
        assert_eq!(
            ext_io.iter().filter(|x| matches!(x, Io::Spill { part: Part::Full, .. })).count(),
            2,
            "1.00x delta: only pages 3 and 4 written"
        );
        assert_eq!(ext_io.iter().filter(|x| matches!(x, Io::Spill { part: Part::State(_), .. })).count(), 2);
        s2.finish_io().unwrap();
        assert!(s2.has_entry(pf2_id) && !s2.has_entry(req_id));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn boundary_state_keep_id_pinning_and_namespace_isolation() {
        let mut s = Stash::new(2, 64);
        s.set_state_mode(1, false);
        s.set_rules(crate::reuse::Rules { strict: false, noend: true, grid: 0 });
        // Tight tier: 4 blocks total (each 64-token entry takes 1 full page + 1 state block = 2 blocks).
        s.add_tier(4, Keep::All, None);

        let a_hist: Vec<i32> = (100..170).collect();
        let b_hist: Vec<i32> = (200..270).collect();
        let c_hist: Vec<i32> = (300..370).collect();
        let ns_steered = [7u8; crate::pool::NAMESPACE_BYTES];

        // Store A (oldest, 2 blocks) and steered B (newest, 2 blocks) -> tier full (4/4 blocks).
        let id_a = s
            .capture_entry(
                Finished { history: a_hist.clone(), valid: 68, written_end: 68, pf_end: 68 },
                &[0],
                0,
                1,
                STOCK_NAMESPACE,
                None,
                false,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        let _ = s.take_pending_io();
        s.finish_io().unwrap();

        let id_b = s
            .capture_entry(
                Finished { history: b_hist.clone(), valid: 68, written_end: 68, pf_end: 68 },
                &[1],
                1,
                1,
                ns_steered,
                None,
                false,
                None,
                &[],
            )
            .unwrap()
            .unwrap();
        let _ = s.take_pending_io();
        s.finish_io().unwrap();

        // Namespace isolation: steered B never matches stock, stock A never matches steered.
        let mut b_cont = b_hist.clone();
        b_cont.push(999);
        assert_eq!(s.match_state(&b_cont, 69, STOCK_NAMESPACE).skip, 0);
        assert_eq!(s.match_state(&b_cont, 69, ns_steered).skip, 68);

        // Evict parked C while pinning oldest A (`keep_id = Some(id_a)`): B is evicted, A survives for restore!
        let id_c = s
            .capture_entry(
                Finished { history: c_hist, valid: 68, written_end: 68, pf_end: 68 },
                &[2],
                2,
                1,
                STOCK_NAMESPACE,
                None,
                false,
                Some(id_a),
                &[],
            )
            .unwrap()
            .unwrap();
        let _ = s.take_pending_io();
        s.finish_io().unwrap();
        assert!(s.has_entry(id_a), "keep_id must protect claimed hit from LRU eviction");
        assert!(!s.has_entry(id_b));
        assert!(s.has_entry(id_c));
    }
}
