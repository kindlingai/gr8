//! The S2 page allocator below R (the pool every rank runs). See lib.rs for the rules it follows.

use crate::reuse::{self, Finished, Rules, Why};
use crate::stash::{continues, Index, Io, Keep, Policy, Stash, TierStats};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

/// Rows per page.
pub const PAGE: u64 = 64;

/// A session's run as the control link carries it (g6s2::Run): list_hash names the page list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Run {
    pub first_page: u32,
    pub pages: u32,
    pub table_version: u32,
    pub list_hash: u64,
}

/// match()'s result: skip = the retained rows a prompt can reuse (0 = none), age = the retained run, for take().
/// stored: the run is in tier `tier` (stash.rs), and take() restores its pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hit {
    pub skip: i32,
    pub lcp: i32,
    pub why: Why,
    pub age: u64,
    pub stored: bool,
    pub tier: u32,
}

/// GR8_KV_SHARE_PAGES: rows [0, rows) of page `from` copied into page `to`, in every latent and index cache (main and
/// draft model), before anything writes `to` (copy-on-write of a shared partial page). The engine takes the queue
/// (take_copies) at each bind and growth, after the tier copies (io) and before any prefill.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageCopy {
    pub from: u32,
    pub to: u32,
    pub rows: u32,
}

/// Hit::age of a hit on a LIVE run (GR8_KV_SHARE_PAGES): this bit, plus the run's table version. Retained ages and
/// stored entry ids never reach it.
pub const LIVE_HIT: u64 = 1 << 63;

/// Every error leaves the pool as it was. The engine treats each one as fatal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BadSize = 1,
    NoRoom,
    UnknownRun,
    NoHit,
    NotRetained,
    VersionExhausted,
    BadArgument,
    PastWorst,
    PendingIo,
}

impl Error {
    pub fn code(self) -> i32 {
        self as i32
    }
    /// The C++ pool's require() messages, so the engine's failures read as before.
    pub fn message(self) -> &'static std::ffi::CStr {
        match self {
            Error::BadSize => c"bad S2 pool / placement / below-R size",
            Error::NoRoom => c"pool reserve without room",
            Error::UnknownRun => c"pool run unknown or changed",
            Error::NoHit => c"pool take without a hit",
            Error::NotRetained => c"pool take of a run no longer retained",
            Error::VersionExhausted => c"pool table version exhausted",
            Error::BadArgument => c"pool call with a null or invalid argument",
            Error::PastWorst => c"pool growth past the run's worst case",
            Error::PendingIo => c"pool call before the store copies of the last one ran",
        }
    }
    pub fn from_code(code: i32) -> Option<Error> {
        [
            Error::BadSize,
            Error::NoRoom,
            Error::UnknownRun,
            Error::NoHit,
            Error::NotRetained,
            Error::VersionExhausted,
            Error::BadArgument,
            Error::PastWorst,
            Error::PendingIo,
        ]
        .into_iter()
        .find(|e| e.code() == code)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message().to_string_lossy())
    }
}

impl std::error::Error for Error {}

struct Held {
    tokens: Vec<i32>,  // the prompt, kept only with tiers
    hits: u32,         // times the conversation continued a run to get here
    span: Option<u64>, // its system span's rows (stash.rs span_rows), when the front end sent one
    list: Vec<u32>,
    exclusive: bool,
    worst: u32, // the most pages the run may grow to
    rp: u32,    // its prefix pages, R / 64 for the R it was granted at (the pool's for a run below R)
    from: Option<(u64, usize)>, // Kept::from
    // GR8_KV_SHARE_PAGES: (namespace, prompt) from count(), and the rows prefill has written (prefilled()): a run with
    // pf > 0 shares its first pf rows with a new prompt that continues them
    sp: Option<(Namespace, Vec<i32>)>,
    pf: u32,
}

/// Totals for status.s2: live runs, the pages they hold and could grow to, granted and refused growth.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub live_runs: u32,
    pub held_pages: u32,
    pub worst_pages: u32,
    pub grows: u64,
    pub stalls: u64,
}

/// busy_view's answer: the retained runs (their ages, oldest first) a reserve on the same state would evict, and how
/// many of them the caller's clock still counts as live (their last response ended under the guard's idle bound ago).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BusyView {
    pub victims: Vec<u64>,
    pub live_victims: u32,
}

/// A retired run: its pages count as free until a reservation evicts it or a hit adopts its start.
#[derive(Clone)]
pub struct Kept {
    pub list: Vec<u32>,
    pub st: Finished,
    pub age: u64,
    pub exclusive: bool,
    /// times the conversation continued a run to get here (a tier's min_hits): a hit on its run, or a prompt that holds
    /// an earlier run's prompt
    pub hits: u32,
    pub namespace: Namespace,
    /// its prefix pages, as Held::rp
    pub rp: u32,
    /// (stored entry id, n): the run's pages [0, n) are byte copies of that entry's pages [0, n) (it was restored from
    /// it, or adopted a run that was), so storing the run writes only its pages past n (stash.rs extend)
    pub from: Option<(u64, usize)>,
    /// when this process's clock saw the retire, nanoseconds since the pool was built. Advice to the head's busy
    /// guard only (busy_view, called with the caller's clock): never in the digest, and no pool decision reads it.
    pub retired_ns: i64,
}

/// Opaque cache identity. STOCK_NAMESPACE (all zero) selects unsteered stock behavior.
pub const NAMESPACE_BYTES: usize = 70;
pub type Namespace = [u8; NAMESPACE_BYTES];
pub const STOCK_NAMESPACE: Namespace = [0; NAMESPACE_BYTES];

const FNV_OFFSET: u64 = 14695981039346656037;

fn mix(mut h: u64, v: u64) -> u64 {
    for i in 0..8 {
        h ^= (v >> (8 * i)) & 0xff;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn hash(list: &[u32], exclusive: bool) -> u64 {
    list.iter().fold(mix(FNV_OFFSET, exclusive as u64), |h, &p| mix(h, p as u64))
}

fn pages_for(rows: u64) -> u64 {
    rows.div_ceil(PAGE)
}

// whether a request of `rows` is past R = rp pages
fn past(rows: u64, rp: u64) -> bool {
    rows > rp * PAGE
}

// marks a run's R in the digest
const RP_TAG: u64 = 0x7270_0000_0000_0000;

pub struct Pool {
    pages: u32,
    place: u32,
    below_rows: u32,
    // ranks of the CP map past R (tail()): 0 = a run past R takes the whole pool as pages 0, 1, .. (the CP map as
    // built); above 0 it takes its own pages in two classes: the prefix (R/64 pages) and its shard (ceil of the tail
    // pages / nr) from [0, place), the rest (the drafter's copy of the tail) from [place, pages)
    nr: u32,
    // remote rows release (the engine's GR8_RT_LOCAL, with nr > 0): a split run's main class is its prefix and `window`
    // pages (the rows tier cycles its tail through them), and its drafter class the whole tail, so the drafter never
    // holds a page the main model releases. Retained, such a run keeps its list, and the engine keeps its rows store
    // records with it: a hit adopts its prefix, window and drafter pages below the hit (window_adopt), an eviction lets
    // them go, and the block stores never take one. 0 = off
    window: u32,
    // with window: the records each rank's rows store holds (its lease), 0 = no limit. A window run takes the most
    // records one rank can need for its tail, ceil(tail pages / nr), at its grant, as it takes its pages
    ids: u32,
    // with window: the most window runs live at once (the engine's slot pool covers that many sessions' rows of one
    // step), 0 = no limit
    runs: u32,
    // the most pages one run's worst case may take (one session's page table, --cap / 64), 0 = no limit. A pool larger
    // than one table (GR8_CAP_TOTAL) needs it: a longer run would get a table that maps only its first pages
    run_limit: u32,
    version: u32,
    live: u32,
    live_hi: u32, // of live: the pages at or above place (the drafter's tail copies of split runs)
    used: Vec<bool>, // used[p] == (live_ref[p] > 0): a live run holds p (retained runs count as free)
    // GR8_KV_SHARE_PAGES (share): live_ref[p] = the live runs whose list holds p. Off, it is 0 or 1 (= used). The
    // retained holders (kept_ref) are counted from `kept` when needed (kept_ref()), so no eviction path can miss them
    live_ref: Vec<u32>,
    share: bool,
    copies: Vec<PageCopy>,
    next_share: Option<(Namespace, Vec<i32>)>,
    // pages a grant is adopting while it fills (spill: still held, they do not leave HBM)
    adopting: Vec<u32>,
    held: BTreeMap<u32, Held>,
    exclusive: bool,
    keep: bool,
    log: bool,
    rules: Rules,
    kept: Vec<Kept>,
    age: u64,
    grows: u64,
    stalls: u64,
    stash: Option<Stash>,
    next_tokens: Option<Vec<i32>>,
    next_span: Option<u64>,
    next_continues: bool,    // count() found the next grant continues a conversation
    next_len: Option<usize>, // the next grant's prompt length (any namespace), for the write limit's credit
    next_prompt: Option<(Namespace, Vec<i32>)>, // the next grant's (namespace, prompt), for next_continues_run
    next_from: Option<(u64, usize)>, // the next grant's provenance (Kept::from), set by a hit just before its grant
    // resize (#153): pages [0, place) come in chunks of `chunk` pages (0 = off). An offline chunk's pages hold no live
    // or retained run and are not counted anywhere, so the engine may release their memory.
    chunk: u32,
    offline: Vec<bool>,
    // when each stored entry was last used, by this process's clock (coldest_ages: what a KV node's head reports).
    // Advice to the node only: never in the digest, and no pool decision reads it. Kept only once coldest_ages ran
    // (a group head): a pool that never reports does no work for it and holds nothing
    seen: HashMap<u64, Instant>,
    ages_on: bool,
    // the pool's build time: the base of the Kept::retired_ns stamps (advisory; see Kept)
    start: Instant,
}

impl Pool {
    /// pool_pages = the logical capacity, place_pages = the physical pages every rank has for runs below R,
    /// below_rows = R (a reservation of more rows is exclusive).
    pub fn new(pool_pages: u32, place_pages: u32, below_rows: u32) -> Result<Pool, Error> {
        if place_pages == 0
            || place_pages > pool_pages
            || below_rows == 0
            || !(below_rows as u64).is_multiple_of(PAGE)
            || below_rows as u64 / PAGE > place_pages as u64
        {
            return Err(Error::BadSize);
        }
        Ok(Pool {
            pages: pool_pages,
            place: place_pages,
            below_rows,
            nr: 0,
            window: 0,
            ids: 0,
            runs: 0,
            run_limit: 0,
            version: 0,
            live: 0,
            live_hi: 0,
            used: vec![false; pool_pages as usize],
            live_ref: vec![0; pool_pages as usize],
            share: cfg!(feature = "share_default"),
            copies: Vec::new(),
            next_share: None,
            adopting: Vec::new(),
            held: BTreeMap::new(),
            exclusive: false,
            keep: false,
            log: false,
            rules: Rules::default(),
            kept: Vec::new(),
            age: 0,
            grows: 0,
            stalls: 0,
            stash: None,
            next_tokens: None,
            next_span: None,
            next_continues: false,
            next_len: None,
            next_prompt: None,
            next_from: None,
            chunk: 0,
            offline: Vec::new(),
            seen: HashMap::new(),
            ages_on: false,
            start: Instant::now(),
        })
    }

    /// This process's clock, nanoseconds since the pool was built (the retire stamps and busy_view's caller base).
    pub fn now_ns(&self) -> i64 {
        self.start.elapsed().as_nanos() as i64
    }

    /// GR8_KV_SHARE_PAGES: runs below R share physical pages. A hit (on a retained run, or on a live run whose prefill
    /// passed the hit) takes the source's full pages below the hit by reference count instead of moving them, and a
    /// copy (PageCopy) of the source's partial page when the hit ends inside one. The retained source stays, its
    /// retirement order refreshed. Off (the default), every call behaves as before and the digest is unchanged.
    /// Only before any reservation. Err(BadArgument) otherwise.
    pub fn share_pages(&mut self, on: bool) -> Result<(), Error> {
        if self.live != 0 || !self.kept.is_empty() {
            return Err(Error::BadArgument);
        }
        self.share = on;
        Ok(())
    }

    pub fn sharing(&self) -> bool {
        self.share
    }

    /// The live runs that hold page p.
    pub fn live_ref(&self, p: u32) -> u32 {
        self.live_ref.get(p as usize).copied().unwrap_or(0)
    }

    /// The retained runs that hold each page (kept_ref), counted from the retained lists.
    pub fn kept_ref(&self) -> Vec<u32> {
        let mut v = vec![0u32; self.pages as usize];
        for k in &self.kept {
            for &p in &k.list {
                v[p as usize] += 1;
            }
        }
        v
    }

    /// The copies queued since the last take_copies(), in order. The engine runs them before anything writes their
    /// `to` pages (s2.cuh s2_copy_rows), on every rank.
    pub fn take_copies(&mut self) -> Vec<PageCopy> {
        std::mem::take(&mut self.copies)
    }

    pub fn copies_pending(&self) -> usize {
        self.copies.len()
    }

    /// GR8_KV_SHARE_PAGES: live run r's prefill has written rows [0, rows) (the driver, every rank, between rounds).
    /// From then on a prompt that continues those rows can share them. No-op with sharing off.
    pub fn prefilled(&mut self, r: &Run, rows: u32) -> Result<(), Error> {
        if !self.share {
            return Ok(());
        }
        self.held(r)?;
        let h = self.held.get_mut(&r.table_version).expect("held() found it");
        h.pf = h.pf.max(rows);
        Ok(())
    }

    /// Resize (#153): pages [0, place) in chunks of `pages_per_chunk` (the last one may be shorter). Only while every
    /// chunk is online. Chunk k is pages [k * pages_per_chunk, min((k + 1) * pages_per_chunk, place)).
    pub fn chunks(&mut self, pages_per_chunk: u32) -> Result<(), Error> {
        if pages_per_chunk == 0 || self.offline.iter().any(|&o| o) {
            return Err(Error::BadArgument);
        }
        self.chunk = pages_per_chunk;
        self.offline = vec![false; self.place.div_ceil(pages_per_chunk) as usize];
        Ok(())
    }

    /// Pages a chunk holds (0 = resize off).
    pub fn chunk_pages(&self) -> u32 {
        self.chunk
    }

    pub fn chunk_count(&self) -> u32 {
        self.offline.len() as u32
    }

    /// Whether chunk k is online: false for k >= chunk_count() once chunks are on, true for every k while they are off.
    pub fn chunk_online(&self, k: u32) -> bool {
        if self.chunk == 0 {
            return true;
        }
        self.offline.get(k as usize).is_some_and(|&o| !o)
    }

    fn chunk_range(&self, k: usize) -> std::ops::Range<u32> {
        let lo = k as u32 * self.chunk;
        lo..(lo + self.chunk).min(self.place)
    }

    /// Whether page p is online: past place always, below it when its chunk is.
    pub fn online(&self, p: u32) -> bool {
        self.chunk == 0 || p >= self.place || !self.offline[(p / self.chunk) as usize]
    }

    /// The online pages below place: what runs below R and the main class past R can hold.
    pub fn place_online(&self) -> u32 {
        self.place - self.offline_pages()
    }

    /// Pages of offline chunks.
    pub fn offline_pages(&self) -> u32 {
        (0..self.offline.len()).filter(|&k| self.offline[k]).map(|k| self.chunk_range(k).len() as u32).sum()
    }

    // the online pages from page 0 on, up to the first offline one: what an exclusive run on the tables as built
    // (pages 0, 1, ..) can take
    fn prefix_online(&self) -> u32 {
        match self.offline.iter().position(|&o| o) {
            Some(k) => self.chunk_range(k).start,
            None => self.pages,
        }
    }

    /// Takes `n` chunks offline: the highest online chunks no live run holds a page of, after evicting (and
    /// spilling) every retained run that holds one. Ok(None), with the pool unchanged, when fewer than n chunks are free
    /// of live runs, when the live runs could not all reach their worst cases in what stays online, or while an
    /// exclusive run is resident: the caller waits for sessions to finish and tries again. A shrink never evicts or
    /// moves a live run. Err(BadSize) when no state allows it: fewer than n chunks online, or the largest n of them
    /// would leave less than R / 64 pages online below place (a request of R rows could never fit). Err(BadArgument)
    /// before chunks(). Ok(Some(chunks)): the chunks now offline, ascending. The evicted runs' spills
    /// are queued (io()): the engine runs them before it releases the chunks' memory.
    pub fn shrink(&mut self, n: u32) -> Result<Option<Vec<u32>>, Error> {
        self.pending()?;
        if self.chunk == 0 {
            return Err(Error::BadArgument);
        }
        if n == 0 {
            return Ok(Some(Vec::new()));
        }
        // a size no state allows: fewer than n chunks online, or the largest n of them leave fewer than the floor
        // online below place. For CP pools (below_rows / PAGE < place) the floor is R / 64 so a request of R rows can
        // still fit; for flat non-CP pools (below_rows / PAGE == place, e.g. multi-session Flash borrow-idle) the floor
        // is 1 page (at least one chunk stays online).
        let mut sizes: Vec<u32> =
            (0..self.offline.len()).filter(|&k| !self.offline[k]).map(|k| self.chunk_range(k).len() as u32).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let floor_pages = if self.below_rows as u64 / PAGE == self.place as u64 { 1 } else { self.below_rows as u64 / PAGE };
        if sizes.len() < n as usize
            || ((self.place_online() - sizes.iter().take(n as usize).sum::<u32>()) as u64) < floor_pages
        {
            return Err(Error::BadSize);
        }
        // with the page classes on (nr > 0) a past-R run no longer holds the pool, but a shrink still waits for it
        if self.exclusive || self.held.values().any(|h| h.exclusive) {
            return Ok(None);
        }
        let mut pick = Vec::new();
        for k in (0..self.offline.len()).rev() {
            if pick.len() == n as usize {
                break;
            }
            if !self.offline[k] && !self.chunk_range(k).any(|p| self.used[p as usize]) {
                pick.push(k);
            }
        }
        if pick.len() < n as usize {
            return Ok(None); // live runs hold a page in too many chunks: wait for sessions to finish
        }
        let left = self.place_online() - pick.iter().map(|&k| self.chunk_range(k).len() as u32).sum::<u32>();
        if !self.safe_with(left, None, None, None) {
            return Ok(None);
        }
        let hit: Vec<bool> = {
            let mut h = vec![false; self.place as usize];
            for &k in &pick {
                for p in self.chunk_range(k) {
                    h[p as usize] = true;
                }
            }
            h
        };
        let (gone, stay): (Vec<Kept>, Vec<Kept>) = std::mem::take(&mut self.kept)
            .into_iter()
            .partition(|k| k.list.iter().any(|&p| p < self.place && hit[p as usize]));
        self.kept = stay;
        self.spill(&gone);
        for &k in &pick {
            self.offline[k] = true;
        }
        pick.sort_unstable();
        self.log_resize("shrink", &pick, gone.len());
        Ok(Some(pick.into_iter().map(|k| k as u32).collect()))
    }

    /// Brings `n` offline chunks back, the lowest first (so an exclusive run on the tables as built gets its pages 0, 1,
    /// .. back first). The engine maps their memory before this call. Growth starts from a safe state (every state
    /// the pool reaches is one) and adds only free pages, so it stays safe. Err(BadArgument) when fewer than n chunks
    /// are offline, or before chunks(). Returns the chunks now online, ascending.
    pub fn grow_pool(&mut self, n: u32) -> Result<Vec<u32>, Error> {
        self.pending()?;
        if self.chunk == 0 {
            return Err(Error::BadArgument);
        }
        let back: Vec<usize> = (0..self.offline.len()).filter(|&k| self.offline[k]).take(n as usize).collect();
        if back.len() < n as usize {
            return Err(Error::BadArgument);
        }
        debug_assert!(self.safe(None, None), "the pool left a safe state");
        for &k in &back {
            self.offline[k] = false;
        }
        self.log_resize("grow", &back, 0);
        Ok(back.into_iter().map(|k| k as u32).collect())
    }

    fn log_resize(&self, how: &str, chunks: &[usize], evicted: usize) {
        if !self.log {
            return;
        }
        let s = format!(
            "# s2 pool {how} chunks {chunks:?} online {} of {} live {} kept {} evicted {evicted}\n",
            self.place_online(),
            self.place,
            self.live,
            self.kept.len()
        );
        let _ = std::io::stderr().write_all(s.as_bytes());
    }

    /// Block stores below the pool (stash.rs), after keep(): a prefix that started `shared_after` conversations is
    /// shared. Then add_tier() for each, in flag order.
    pub fn stores(&mut self, shared_after: u32) {
        self.stash = Some(Stash::new(shared_after, self.rules.grid.max(0) as u64));
    }

    /// A tier of `slots` blocks with its keep rule. index: (its folder, what makes its blocks compatible) to keep shared
    /// prefixes across restarts.
    pub fn add_tier(&mut self, slots: u64, keep: Keep, index: Option<(&std::path::Path, &str)>) -> std::io::Result<()> {
        let ix = match index {
            Some((dir, compat)) => Some(Index::open(dir, compat)?),
            None => None,
        };
        self.stash.as_mut().ok_or_else(|| std::io::Error::other("add_tier before stores()"))?.add_tier(slots, keep, ix);
        Ok(())
    }

    /// After the last add_tier(): drops the index runs whose base came back in no tier, and the session runs of a
    /// whole index reopened with keep=system, plus the records add_tier dropped for slots past the tier or taken twice
    /// (stash.rs Stash::loaded). Returns how many.
    pub fn tiers_loaded(&mut self) -> usize {
        self.stash.as_mut().map_or(0, |s| s.loaded())
    }

    pub fn stash(&self) -> Option<&Stash> {
        self.stash.as_ref()
    }

    pub fn stash_mut(&mut self) -> Option<&mut Stash> {
        self.stash.as_mut()
    }

    pub fn tier_stats(&self, t: usize) -> Option<TierStats> {
        self.stash.as_ref()?.stats(t)
    }

    /// Tier t's policy (stash.rs Policy), after add_tier().
    pub fn set_policy(&mut self, t: usize, p: Policy) -> Result<(), Error> {
        match &mut self.stash {
            Some(s) if t < s.tiers() => {
                s.set_policy(t, p);
                Ok(())
            }
            _ => Err(Error::BadArgument),
        }
    }

    /// Tier t to `slots` blocks (stash.rs resize_tier): a store whose size changed at this event, on every rank. The
    /// copies it queues (moves out of the blocks that go) run like any call's, before the store shrinks. Ok(false):
    /// nothing changed (a tier with an index, or entries that cannot leave).
    pub fn resize_tier(&mut self, t: usize, slots: u64) -> Result<bool, Error> {
        self.pending()?;
        match &mut self.stash {
            Some(s) if t < s.tiers() => Ok(s.resize_tier(t, slots)),
            _ => Err(Error::BadArgument),
        }
    }

    /// Runs past R keep their tails in the stores (stash.rs set_tail): nr ranks, per_slot replicated parts a block.
    /// nr > 0 also turns the page classes on for the pool itself (see Pool::nr), so this is set whenever the pool's R
    /// is the CP map's, stores or not.
    pub fn tail(&mut self, nr: u32, per_slot: u32) {
        self.nr = nr;
        if let Some(s) = &mut self.stash {
            s.set_tail(nr, per_slot, self.place);
        }
    }

    /// Remote rows release: the main pages of a split run past its prefix (0 = off). Before any reservation.
    /// `ids`: the records of each rank's rows store (0 = no limit); a request whose tail needs more than are free waits.
    /// `runs`: the most window runs live at once (0 = no limit); one more waits.
    pub fn window(&mut self, pages: u32, ids: u32, runs: u32) -> Result<(), Error> {
        if self.nr == 0 || self.live != 0 || !self.kept.is_empty() || (pages == 0 && (ids != 0 || runs != 0)) {
            return Err(Error::BadArgument);
        }
        self.window = pages;
        self.ids = ids;
        self.runs = runs;
        Ok(())
    }

    /// The most pages one run's worst case may take (0 = no limit): one session's page table when the pool is larger.
    /// Before any reservation.
    pub fn run_limit(&mut self, pages: u32) -> Result<(), Error> {
        if self.live != 0 || !self.kept.is_empty() {
            return Err(Error::BadArgument);
        }
        self.run_limit = pages;
        Ok(())
    }

    /// The run limit (0 = none).
    pub fn run_limit_pages(&self) -> u32 {
        self.run_limit
    }

    fn over_run_limit(&self, worst: u64) -> bool {
        self.run_limit > 0 && pages_for(worst) > self.run_limit as u64
    }

    /// With window: the window runs live now.
    pub fn window_runs(&self) -> u32 {
        if self.window == 0 {
            return 0;
        }
        self.held.values().filter(|h| h.exclusive).count() as u32
    }

    /// With window: the window runs live now at R = r rows (0 = the pool's R); 0 for an R the pool refuses.
    /// Without window (in-RAM split pool, nr > 0): counts live split runs at R = r rows.
    pub fn window_runs_r(&self, r: u32) -> u32 {
        let Some(rp) = self.rp_of(r).filter(|_| self.window > 0 || self.nr > 0) else { return 0 };
        self.held.values().filter(|h| h.exclusive && h.rp as u64 == rp).count() as u32
    }

    // a live window run's records on the rank that holds the most of its tail
    fn ids_of(&self, h: &Held) -> u64 {
        if !h.exclusive || self.window == 0 {
            return 0;
        }
        let tail = (h.list.len() as u64).saturating_sub(h.rp as u64 + self.window as u64);
        tail.div_ceil(self.nr.max(1) as u64)
    }

    /// With window and a lease limit: the records no live window run holds.
    pub fn ids_free(&self) -> Option<u64> {
        let kept: u64 = self.kept.iter().map(|k| self.kept_ids(k)).sum();
        (self.ids > 0).then(|| (self.ids as u64).saturating_sub(self.held.values().map(|h| self.ids_of(h)).sum::<u64>() + kept))
    }

    /// With window and a lease limit: the records no live window run holds. Retained window runs' records count as
    /// free here, as their pages do: a grant that needs them evicts those runs, the least recently retired first.
    pub fn ids_room(&self) -> Option<u64> {
        (self.ids > 0).then(|| (self.ids as u64).saturating_sub(self.held.values().map(|h| self.ids_of(h)).sum::<u64>()))
    }

    // a grant whose tail needs `need` records a rank: the retained window runs to evict, oldest first, so they fit
    fn evict_for_ids(&mut self, need: u64) -> Vec<Kept> {
        let mut gone = Vec::new();
        while self.ids_free().is_some_and(|f| f < need) {
            let Some(i) = self.kept.iter().position(|k| self.kept_ids(k) > 0) else { break };
            gone.push(self.kept.remove(i));
        }
        gone
    }

    // a retained window run: the records it keeps, as ids_of counts a live one
    fn kept_ids(&self, k: &Kept) -> u64 {
        if !k.exclusive || self.window == 0 {
            return 0;
        }
        let tail = (k.list.len() as u64).saturating_sub(k.rp as u64 + self.window as u64);
        tail.div_ceil(self.nr.max(1) as u64)
    }

    /// Whether a retained run holds exactly this list (the engine asks at a retire: it keeps a window run's records
    /// only while the pool keeps the run).
    pub fn is_kept(&self, list: &[u32]) -> bool {
        self.kept.iter().any(|k| k.list == list)
    }

    // a past-R grant adopting retained window run k at a = ceil(skip / 64) logical pages: its prefix and window, and its
    // drafter pages of the logical pages below a (the drafter's tail is in list order past the window)
    fn window_adopt(&self, k: &Kept, a: u64) -> Option<Vec<u32>> {
        let (rp, w) = (k.rp as usize, self.window as usize);
        let d = (a as usize).saturating_sub(rp);
        if k.list.len() < rp + w + d {
            return None;
        }
        Some(k.list[..rp + w].iter().chain(&k.list[rp + w..rp + w + d]).copied().collect())
    }

    // whether that grant (worst rows) fits: the drafter's fresh pages above place, the adopted ones leaving the free
    // classes, its records (the kept run's own are counted free for it) and its share of the slots
    fn window_hit_room(&self, k: &Kept, a: u64, worst: u64, rp: u64) -> bool {
        let Some(adopted) = self.window_adopt(k, a) else { return false };
        let (main, draft) = self.exclusive_pages(worst, rp);
        let d = adopted.len() as u64 - main;
        if !past(worst, rp) || d > draft {
            return false;
        }
        let ids_ok = self.ids_room().map_or(true, |f| draft.div_ceil(self.nr as u64) <= f);
        let runs_ok = self.runs == 0 || self.window_runs() < self.runs;
        !self.exclusive
            && ids_ok
            && runs_ok
            && main <= self.lo_free() as u64
            && draft <= self.hi_free() as u64
            && self.safe_with(self.place_online(), None, None, Some(main))
    }

    /// For a request of `worst` rows past R: the pages it takes below place and above it, and what those classes have
    /// free now (retained runs count as free). None below R or without page classes.
    pub fn split_view(&self, worst: u64) -> Option<(u64, u64, u32, u32)> {
        self.split_view_r(worst, 0)
    }

    /// split_view() for a grant at R = r rows (0 = the pool's R). None also for an R the pool refuses (rp_of).
    pub fn split_view_r(&self, worst: u64, r: u32) -> Option<(u64, u64, u32, u32)> {
        let rp = self.rp_of(r)?;
        if self.nr == 0 || !past(worst, rp) {
            return None;
        }
        let (main, draft) = self.exclusive_pages(worst, rp);
        Some((main, draft, self.lo_free(), self.hi_free()))
    }

    /// For a window request of `worst` rows past R with a lease limit: the records its tail needs per rank, and the
    /// records it can have (free, or held by retained runs it would evict) and in all. None otherwise.
    pub fn ids_view(&self, worst: u64) -> Option<(u64, u64, u64)> {
        self.ids_view_r(worst, 0)
    }

    /// ids_view() for a grant at R = r rows (0 = the pool's R).
    pub fn ids_view_r(&self, worst: u64, r: u32) -> Option<(u64, u64, u64)> {
        let rp = self.rp_of(r)?;
        if self.window == 0 || self.ids == 0 || !past(worst, rp) {
            return None;
        }
        let (_, draft) = self.exclusive_pages(worst, rp);
        Some((draft.div_ceil(self.nr as u64), self.ids_room().unwrap_or(0), self.ids as u64))
    }

    /// Whether a run past R takes its own pages (tail() set the CP map's ranks): the engine builds its tables from
    /// its list instead of using the layers' tables as built.
    pub fn split(&self) -> bool {
        self.nr > 0
    }

    /// The write limit (stash.rs): the tiers write at most milli / 1000 blocks per page of KV computed. 0 = none.
    pub fn write_limit(&mut self, milli: u64) {
        if let Some(s) = &mut self.stash {
            s.set_limit(milli);
        }
    }

    // a grant of n pages whose first `reused` came from the pool or a tier: the rest of its prompt is computed
    fn earn(&mut self, n: u64, reused: u64) {
        if let Some(s) = &mut self.stash {
            let prompt = self.next_len.map_or(n, |t| pages_for(t as u64).min(n));
            s.earn(prompt.saturating_sub(reused));
        }
    }

    /// The entries loaded at boot, one line each, for the ranks to agree on (keep_agreed).
    pub fn stored_summary(&self) -> String {
        self.stash.as_ref().map_or_else(String::new, Stash::summary)
    }

    pub fn keep_agreed(&mut self, agreed: &str) {
        if let Some(s) = &mut self.stash {
            s.keep_agreed(agreed);
        }
    }

    /// At admission, before the grant (head at the event, workers at their grant): counts the prompt toward the
    /// prefixes it starts (stash.rs). Returns how many it counted, which the grant carries.
    /// `span`: the tokens of the prompt's system span (system message and tool definitions as templated), when the
    /// front end sent it.
    pub fn count(&mut self, prompt: &[i32], span: Option<u32>) -> u32 {
        self.next_prompt = None;
        self.next_share = self.share.then(|| (STOCK_NAMESPACE, prompt.to_vec()));
        let Some(s) = &self.stash else { return 0 };
        self.next_tokens = Some(prompt.to_vec());
        self.next_prompt = Some((STOCK_NAMESPACE, prompt.to_vec()));
        self.next_len = Some(prompt.len());
        self.next_span = span.map(|x| s.span_rows(x, prompt.len()));
        if !s.counting() {
            return 0;
        }
        let continuing = self.held.values().any(|h| continues(prompt, &h.tokens, h.tokens.len().saturating_sub(1)))
            || self.kept.iter().any(|k| {
                k.namespace == STOCK_NAMESPACE && continues(prompt, &k.st.history, k.st.pf_end.max(0) as usize)
            })
            || s.continues_stored(prompt);
        self.next_continues = continuing;
        let span = self.next_span;
        self.touch_prefixes(prompt);
        self.stash.as_mut().expect("checked").count(prompt, continuing, span)
    }

    // a prompt that starts with a stored prefix reads its blocks when it is admitted (or will, on a hit): used now
    fn touch_prefixes(&mut self, prompt: &[i32]) {
        let Some(s) = self.stash.as_ref().filter(|_| self.ages_on) else { return };
        let now = Instant::now();
        for t in 0..s.tiers() {
            for e in s.entries(t).iter().filter(|e| e.kind.prefix()) {
                if prompt.starts_with(&e.st.history[..(e.st.valid.max(0) as usize).min(e.st.history.len())]) {
                    self.seen.insert(e.id, now);
                }
            }
        }
    }

    /// The ages a KV node's head REPORTs for tier t (remote::coldest_ages): for each i, u64::MAX while the tier's free
    /// slots cover a give of i + 1 segments, then ms since the use of the most recently used entry that give would
    /// evict (the shrink's order, stash.rs give_order), at most `most`; they never rise. An entry was used when it was
    /// stored, restored, or a prompt started with it (a prefix); a prefix base counts as used as recently as the
    /// newest run stored on it.
    pub fn coldest_ages(&mut self, t: usize, per_seg: u64, most: usize) -> Vec<u64> {
        let Some(s) = &self.stash else { return Vec::new() };
        if t >= s.tiers() {
            return Vec::new();
        }
        self.ages_on = true;
        let now = Instant::now();
        let mut live = std::collections::HashSet::new();
        for x in 0..s.tiers() {
            for e in s.entries(x) {
                self.seen.entry(e.id).or_insert(now);
                live.insert(e.id);
            }
        }
        self.seen.retain(|id, _| live.contains(id));
        let mut at = self.seen.clone();
        for x in 0..s.tiers() {
            for e in s.entries(x) {
                if let Some((b, _)) = e.base {
                    let u = self.seen[&e.id];
                    if let Some(v) = at.get_mut(&b) {
                        *v = (*v).max(u);
                    }
                }
            }
        }
        crate::remote::coldest_ages(
            &s.give_order(t),
            |id| at.get(&id).map_or(0, |u| now.duration_since(*u).as_millis() as u64),
            per_seg,
            s.stats(t).map_or(0, |x| x.free_slots),
            most,
        )
    }

    /// Prepare an admission without counting non-stock prompts toward stock promotion. Call before its grant,
    /// including after an abandoned stock admission, so no pending stock metadata is inherited.
    pub fn count_namespaced(&mut self, prompt: &[i32], span: Option<u32>, namespace: Namespace) -> u32 {
        if namespace == STOCK_NAMESPACE {
            return self.count(prompt, span);
        }
        self.next_share = self.share.then(|| (namespace, prompt.to_vec()));
        self.next_tokens = None;
        self.next_prompt = self.stash.is_some().then(|| (namespace, prompt.to_vec()));
        self.next_len = Some(prompt.len());
        self.next_span = None;
        // a non-stock prompt that continues a run of its own namespace (retained or stored) counts as continued, so a
        // keep-sessions tier takes its run later; it never counts toward stock promotion
        self.next_continues = self.stash.as_ref().is_some_and(|s| {
            self.kept
                .iter()
                .any(|k| k.namespace == namespace && continues(prompt, &k.st.history, k.st.pf_end.max(0) as usize))
                || s.continues_stored_in(prompt, namespace)
        });
        0
    }

    /// The queued copies (stash.rs Io), in the order to run them. The engine calls io_done() after it ran them.
    pub fn io(&mut self) -> Vec<Io> {
        self.stash.as_mut().map_or_else(Vec::new, Stash::take_io)
    }

    /// After the copies from io(): commits the index records of the entries they wrote. An error leaves the pool as it
    /// is. The entries stay, and only a restart misses them.
    pub fn io_done(&mut self) -> std::io::Result<()> {
        if let Some(s) = self.stash.as_ref().filter(|_| self.ages_on) {
            // entries the copies stored were used now; the ones that left are forgotten
            let now = Instant::now();
            let mut live = std::collections::HashSet::new();
            for t in 0..s.tiers() {
                for e in s.entries(t) {
                    self.seen.entry(e.id).or_insert(now);
                    live.insert(e.id);
                }
            }
            self.seen.retain(|id, _| live.contains(id));
        }
        self.stash.as_mut().map_or(Ok(()), Stash::io_done)
    }

    /// Copies queued since the last io().
    pub fn io_pending(&self) -> usize {
        self.stash.as_ref().map_or(0, |s| s.io.len())
    }

    fn pending(&self) -> Result<(), Error> {
        match &self.stash {
            Some(s) if s.pending() => Err(Error::PendingIo),
            _ => Ok(()),
        }
    }

    // evicted retained runs go to the store, when there is one. The prompts of the live runs, the retained runs and the
    // request being granted are what a prefix base is shared with
    fn spill(&mut self, gone: &[Kept]) {
        // a window run's tail is in its rows store records, not in pool memory, and a run at a smaller R than the
        // pool's holds its tail in that R's layout (stores hold tails at the pool's R only): the stores never take them
        let rp0 = self.rp0();
        let skip_store = |k: &Kept| k.exclusive && (self.window > 0 || k.rp as u64 != rp0);
        let kept: Vec<Kept>;
        let gone = if gone.iter().any(skip_store) {
            kept = gone.iter().filter(|k| !skip_store(k)).cloned().collect();
            &kept[..]
        } else {
            gone
        };
        let below = (self.below_rows as u64 / PAGE) as usize;
        // GR8_KV_SHARE_PAGES: a retained run whose storable pages another holder keeps in HBM (a live run, a retained run
        // that stays, or the grant adopting them), or that another leaving run covers, adds nothing to a tier: it is
        // let go, not written.
        // Its prefix stays hit-able through those holders. A run that frees any page spills as before (the tier stores
        // a shared prefix once). Only pages with no holder left are ever freed (fill_range's claims).
        let shared_out: Vec<Kept>;
        let gone = if self.share && self.stash.is_some() && !gone.is_empty() {
            let mut held = vec![false; self.pages as usize];
            for k in &self.kept {
                for &p in &k.list {
                    held[p as usize] = true;
                }
            }
            for &p in &self.adopting {
                held[p as usize] = true;
            }
            // what a tier would store of a run: its full pages of valid rows (a tier's grid is a multiple of a page)
            let storable = |k: &Kept| ((k.st.valid.max(0) as u64 / PAGE) as usize).min(k.list.len());
            // a run another leaving run of its namespace covers (same physical pages over all its storable rows, more
            // valid rows, or as many and later in `gone`) is that run's prefix: the tier stores it once, with that run
            let covered = |i: usize, k: &Kept| {
                let s = storable(k);
                gone.iter().enumerate().any(|(j, k2)| {
                    j != i
                        && k2.namespace == k.namespace
                        && k2.list.len() >= s
                        && k2.list[..s] == k.list[..s]
                        && (k2.st.valid > k.st.valid || (k2.st.valid == k.st.valid && j > i))
                })
            };
            let mut stay = Vec::new();
            let mut leave = Vec::new();
            for (i, k) in gone.iter().enumerate() {
                let s = storable(k);
                if k.list[..s].iter().all(|&p| held[p as usize] || self.live_ref[p as usize] > 0) || covered(i, k) {
                    stay.push(k.clone());
                } else {
                    leave.push(k.clone());
                }
            }
            shared_out = leave;
            for k in &stay {
                self.let_go(k);
            }
            &shared_out[..]
        } else {
            gone
        };
        let Some(s) = &mut self.stash else { return };
        if gone.is_empty() {
            return;
        }
        let resident: Vec<&[i32]> = self
            .held
            .values()
            .map(|h| &h.tokens[..])
            .chain(self.kept.iter().filter(|k| k.namespace == STOCK_NAMESPACE).map(|k| &k.st.history[..]))
            .chain(self.next_tokens.as_deref())
            .filter(|t| !t.is_empty())
            .collect();
        s.put_evicted(gone, below, &resident);
    }

    // hits of a run granted from `src` (its finished state, hits and namespace): one more when the prompt continues it
    fn hits_from(&self, st: &Finished, hits: u32, namespace: Namespace) -> u32 {
        if self.next_continues_run(st, namespace) {
            hits.saturating_add(1)
        } else {
            0
        }
    }

    // whether the next grant's prompt continues the run or stored entry `st` in its own namespace
    fn next_continues_run(&self, st: &Finished, namespace: Namespace) -> bool {
        matches!(&self.next_prompt, Some((ns, p)) if *ns == namespace && continues(p, &st.history, st.pf_end.max(0) as usize))
    }

    pub fn exclusive(&self, rows: u64) -> bool {
        past(rows, self.rp0())
    }

    /// exclusive() for a grant at R = r rows (0 = the pool's R). Err(BadArgument) for an R the pool refuses.
    pub fn exclusive_r(&self, rows: u64, r: u32) -> Result<bool, Error> {
        self.rp_of(r).map(|rp| past(rows, rp)).ok_or(Error::BadArgument)
    }

    // R / 64 at the pool's R: its physical layout (the CP map as built, the stores' tails)
    fn rp0(&self) -> u64 {
        self.below_rows as u64 / PAGE
    }

    // The prefix pages of a grant at R = r rows: 0 or the pool's R gives the pool's. A smaller R, a multiple of 64,
    // needs the split page classes (nr > 0): runs at a smaller R never spill or restore tails through the stores,
    // whose tails are laid out at the pool's R. None otherwise.
    fn rp_of(&self, r: u32) -> Option<u64> {
        if r == 0 || r == self.below_rows {
            return Some(self.rp0());
        }
        (self.nr > 0 && r < self.below_rows && (r as u64).is_multiple_of(PAGE)).then_some(r as u64 / PAGE)
    }

    // Whether a grant at prefix pages rp may adopt the first a pages of retained run k. A run past R holds its pages
    // past its prefix (and a window run its records) in its own R's layout, so only a grant at the same R adopts from
    // it. A run below R holds replicated pages, which a grant past R adopts only as its prefix.
    fn same_layout(&self, k: &Kept, a: u64, worst: u64, rp: u64) -> bool {
        if k.exclusive {
            k.rp as u64 == rp
        } else {
            !past(worst, rp) || a <= rp
        }
    }

    // A run past R (nr > 0) takes its own pages: of its L = ceil(worst / 64) pages, the prefix (R / 64) and its shard
    // of the tail (ceil of the tail pages / nr) come from [0, place), the tail pages the drafter needs and this rank
    // does not hold from [place, pages). The lists are the same on every rank; what each page holds differs.
    fn exclusive_pages(&self, worst: u64, rp: u64) -> (u64, u64) {
        let t = pages_for(worst).saturating_sub(rp);
        if self.window > 0 {
            return (rp + self.window as u64, t);
        }
        let own = t.div_ceil(self.nr.max(1) as u64).min(t);
        (rp + own, t - own)
    }

    // (a window run's list is its prefix, window and tail: `window` pages longer than its worst case)
    pub fn free_pages(&self) -> u32 {
        if self.exclusive {
            0
        } else {
            self.pages - self.offline_pages() - self.live
        }
    }

    /// The most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages.
    pub fn room(&self) -> u32 {
        if self.exclusive {
            0
        } else if self.live != 0 {
            self.lo_free()
        } else if self.nr == 0 {
            // an exclusive run takes pages 0, 1, .. up to the first offline one; a run below R any online page
            self.prefix_online().max(self.place_online())
        } else {
            self.pages - self.offline_pages()
        }
    }

    // online pages below place no live run holds (retained runs count as free)
    fn lo_free(&self) -> u32 {
        self.place_online().wrapping_sub(self.live - self.live_hi)
    }

    // pages at or above place no live run holds: the drafter's class of split runs
    fn hi_free(&self) -> u32 {
        self.pages - self.place - self.live_hi
    }

    /// Head, before reserve: a past-R request (by its worst case) waits for an empty pool, unless the page classes are
    /// on (nr > 0): then split runs share the pool with each other and with runs below R (decoded in one CP step).
    pub fn admissible(&self, worst: u64) -> bool {
        self.admissible_at(worst, self.rp0())
    }

    /// admissible() for a grant at R = r rows (0 = the pool's R). False for an R the pool refuses.
    pub fn admissible_r(&self, worst: u64, r: u32) -> bool {
        self.rp_of(r).is_some_and(|rp| self.admissible_at(worst, rp))
    }

    fn admissible_at(&self, worst: u64, rp: u64) -> bool {
        !self.exclusive && (!past(worst, rp) || self.live == 0 || self.nr > 0)
    }

    /// Whether a request that needs `rows` now and at most `worst` rows can be admitted: an exclusive run takes its
    /// worst case at once from an empty pool, and a run below R takes `rows` and must leave the pool safe (safe()).
    pub fn fits(&self, rows: u64, worst: u64) -> bool {
        self.fits_at(rows, worst, self.rp0())
    }

    /// fits() for a grant at R = r rows (0 = the pool's R). False for an R the pool refuses.
    pub fn fits_r(&self, rows: u64, worst: u64, r: u32) -> bool {
        self.rp_of(r).is_some_and(|rp| self.fits_at(rows, worst, rp))
    }

    fn fits_at(&self, rows: u64, worst: u64, rp: u64) -> bool {
        let (n, w) = (pages_for(rows), pages_for(worst));
        if rows == 0 || n > w || !self.admissible_at(worst, rp) || self.over_run_limit(worst) {
            return false;
        }
        if past(worst, rp) {
            if self.nr == 0 {
                return w <= self.prefix_online() as u64;
            }
            // its own pages: both classes must have room beside the live runs (a reservation evicts retained runs as
            // it needs them, so every page of a class no live run holds is free or evictable), and the runs below R
            // must still reach their worst cases with its prefix and shard taken (it holds its worst case at once)
            let (main, draft) = self.exclusive_pages(worst, rp);
            return self.split_room(main, draft);
        }
        n <= self.lo_free() as u64 && w <= self.place_online() as u64 && self.safe(Some((n, w)), None)
    }

    /// The busy guard's view of a reserve_r(rows, worst, r) on this state: the retained runs it would evict, oldest
    /// first, and how many of them retired at most `idle_ns` before `now_ns` (`idle_ns` 0: none). It is a dry run of
    /// fill_range and evict_for_ids and must pick the same runs (tests/busy.rs). The pool itself never reads a clock,
    /// so every rank's pool stays identical. Where fits_at fails (the head's intake check), the view walks every
    /// retained run and still lacks room.
    pub fn busy_view(&self, rows: u64, worst: u64, r: u32, now_ns: i64, idle_ns: i64) -> BusyView {
        let mut victims = Vec::new();
        if let Some(rp) = self.rp_of(r).filter(|&rp| rows != 0 && self.admissible_at(worst, rp)) {
            if !past(worst, rp) {
                victims = self.evict_view(pages_for(rows) as usize, 0, self.place, 0);
            } else if self.nr == 0 {
                victims = self.kept.iter().map(|k| k.age).collect();
            } else {
                let (main, draft) = self.exclusive_pages(worst, rp);
                victims = self.evict_view(main as usize, 0, self.place, 0);
                // the drafter's fill walks the runs the main fill left
                let more = self.evict_view(draft as usize, self.place, self.pages, victims.len());
                victims.extend(more);
                if self.window > 0 {
                    let more = self.evict_ids_view(draft.div_ceil(self.nr as u64), &victims);
                    victims.extend(more);
                }
            }
        }
        let live = |k: &&Kept| idle_ns != 0 && now_ns - k.retired_ns <= idle_ns && victims.contains(&k.age);
        let live_victims = self.kept.iter().filter(live).count() as u32;
        BusyView { victims, live_victims }
    }

    // fill_range(need, lo, hi) as a dry run, with kept[..gone] already evicted: the ages of the runs it evicts
    fn evict_view(&self, need: usize, lo: u32, hi: u32, gone: usize) -> Vec<u64> {
        let (lo, hi) = (lo as usize, hi as usize);
        let left = &self.kept[gone..];
        let mut claims: Vec<u32> = (lo..hi).map(|p| (self.used[p] || !self.online(p as u32)) as u32).collect();
        for k in left {
            for &p in k.list.iter().filter(|&&p| (p as usize) >= lo && (p as usize) < hi) {
                claims[p as usize - lo] += 1;
            }
        }
        let mut spare = claims.iter().filter(|&&c| c == 0).count();
        let mut victims = Vec::new();
        for k in left {
            if spare >= need {
                break;
            }
            for &p in k.list.iter().filter(|&&p| (p as usize) >= lo && (p as usize) < hi) {
                claims[p as usize - lo] -= 1;
                if claims[p as usize - lo] == 0 {
                    spare += 1;
                }
            }
            victims.push(k.age);
        }
        victims
    }

    // evict_for_ids(need) as a dry run, with the runs in `gone` already evicted
    fn evict_ids_view(&self, need: u64, gone: &[u64]) -> Vec<u64> {
        let held: u64 = self.held.values().map(|h| self.ids_of(h)).sum();
        let mut gone = gone.to_vec();
        let mut victims = Vec::new();
        loop {
            let kept: u64 = self.kept.iter().filter(|k| !gone.contains(&k.age)).map(|k| self.kept_ids(k)).sum();
            if self.ids == 0 || (self.ids as u64).saturating_sub(held + kept) >= need {
                break;
            }
            let Some(k) = self.kept.iter().find(|k| !gone.contains(&k.age) && self.kept_ids(k) > 0) else { break };
            victims.push(k.age);
            gone.push(k.age);
        }
        victims
    }

    /// A run of pages for `rows` now that may grow (grow()) to `worst` rows. An exclusive run holds its worst case.
    pub fn reserve(&mut self, rows: u64, worst: u64) -> Result<Run, Error> {
        self.reserve_r(rows, worst, 0)
    }

    /// reserve() at R = r rows (0 = the pool's R): a run past r takes its prefix and window as r makes them.
    /// Err(BadArgument) for an R the pool refuses (rp_of).
    pub fn reserve_r(&mut self, rows: u64, worst: u64, r: u32) -> Result<Run, Error> {
        self.pending()?;
        let rp = self.rp_of(r).ok_or(Error::BadArgument)?;
        if !self.fits_at(rows, worst, rp) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let w = pages_for(worst) as u32;
        if past(worst, rp) {
            self.earn(w as u64, 0);
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                return Ok(self.grant((0..w).collect(), true, w, worst, "exclusive", &gone, 0, rp));
            }
            // its own pages: the prefix and its shard of the tail from below place, the drafter's copy of the
            // rest from above it. Retained runs give their pages back only as the classes need them, and every
            // one the classes evict spills (either class may be the one that runs out of room).
            let (main, draft) = self.exclusive_pages(worst, rp);
            let mut list = Vec::with_capacity(w as usize);
            let mut gone = self.fill_range(&mut list, main as usize, 0, self.place);
            gone.extend(self.fill_range(&mut list, draft as usize, self.place, self.pages));
            if self.window > 0 {
                gone.extend(self.evict_for_ids(draft.div_ceil(self.nr as u64)));
            }
            self.spill(&gone);
            return Ok(self.grant(list, true, w, worst, "exclusive", &gone, 0, rp));
        }
        let n = pages_for(rows) as usize;
        self.earn(n as u64, 0);
        let mut list = Vec::with_capacity(n);
        let gone = self.fill(&mut list, n);
        self.spill(&gone);
        Ok(self.grant(list, false, w, rows, if gone.is_empty() { "free" } else { "lru" }, &gone, 0, rp))
    }

    /// A live run below R grows to the pages `rows` needs, the lowest free ones (evicting retained runs as reserve()
    /// does). Ok(None) when the pages would leave the pool unsafe: the caller leaves the session out of this round.
    /// The run keeps its table version, and its list grows at the end.
    pub fn grow(&mut self, r: &Run, rows: u64) -> Result<Option<Run>, Error> {
        self.pending()?;
        let h = self.held(r)?;
        let (n, have) = (pages_for(rows), h.list.len() as u64);
        if n <= have {
            return Ok(Some(*r));
        }
        if h.exclusive || n > h.worst as u64 {
            return Err(Error::PastWorst);
        }
        let add = n - have;
        // GR8_KV_SHARE_PAGES copy-on-write guard: a run whose last page another run holds, not known to be full (its
        // prefilled rows end before it), gets that page as a fresh copy first, one page more. A hit never shares a
        // partial page, so this does not happen; the guard keeps a shared page from ever being written
        let cow = self.share && self.cow_last(h);
        let need = add + cow as u64;
        if need > self.lo_free() as u64 || !self.safe(None, Some((r.table_version, need))) {
            self.stalls += 1;
            return Ok(None);
        }
        let mut h = self.held.remove(&r.table_version).expect("held() found it");
        if let Some(s) = &mut self.stash {
            s.earn(add);
        }
        let mut have = have;
        if cow {
            let from = h.list.pop().expect("a run holds a page");
            let rows = if (h.pf as u64) > (have - 1) * PAGE { (h.pf as u64 - (have - 1) * PAGE) as u32 } else { PAGE as u32 };
            self.mark(&[from], false);
            have -= 1;
            let mut one = h.list.clone();
            let gone = self.fill(&mut one, h.list.len() + 1);
            self.spill(&gone);
            let to = *one.last().expect("filled");
            self.copies.push(PageCopy { from, to, rows });
            h.list.push(to);
        }
        let gone = self.fill(&mut h.list, n as usize);
        self.spill(&gone);
        self.mark(&h.list[have as usize..], true);
        self.grows += 1;
        let g = Run {
            first_page: h.list[0],
            pages: n as u32,
            table_version: r.table_version,
            list_hash: hash(&h.list, false),
        };
        self.log_claim("grow", &g, rows, &gone);
        self.held.insert(r.table_version, h);
        Ok(Some(g))
    }

    /// Whether live run r can shrink to R = r2 rows (shrink_run):
    /// - window mode (self.window > 0): an exclusive window run at a larger R, with room above place and in record ids.
    /// - in-RAM split mode (self.window == 0, self.nr > 0): either an exclusive split run at a larger R, or a below-R
    ///   run whose worst case exceeds r2 (and whose pages >= r2/64 are not shared with another live run), whenever
    ///   shrinking strictly reduces its low-class worst-case footprint and both classes have room.
    pub fn shrinkable(&self, r: &Run, r2: u32) -> bool {
        let Ok(h) = self.held(r) else { return false };
        let Some(rp) = self.rp_of(r2) else { return false };
        if rp >= h.rp as u64 || self.exclusive {
            return false;
        }
        if self.window > 0 {
            if !h.exclusive {
                return false;
            }
            let d = h.rp as u64 - rp;
            let tail = h.list.len() as u64 - h.rp as u64 - self.window as u64;
            let gained = (tail + d).div_ceil(self.nr as u64) - self.ids_of(h);
            return d <= self.hi_free() as u64 && self.ids_room().is_none_or(|f| gained <= f);
        }
        if self.nr == 0 {
            return false;
        }
        if h.exclusive {
            let w = h.list.len() as u64;
            let old_main = h.list.iter().filter(|&&p| p < self.place).count() as u64;
            let old_draft = w - old_main;
            let (new_main, new_draft) = self.exclusive_pages(w * PAGE, rp);
            let need_hi = new_draft.saturating_sub(old_draft);
            old_main > new_main && need_hi <= self.hi_free() as u64
        } else {
            let w = h.worst as u64;
            if w <= rp {
                return false;
            }
            // Pages >= rp will be compacted or freed below place: never touch pages shared with another live run.
            if self.share && h.list.len() > rp as usize && h.list[rp as usize..].iter().any(|&p| self.live_ref[p as usize] > 1) {
                return false;
            }
            let (new_main, new_draft) = self.exclusive_pages(w * PAGE, rp);
            if new_main >= w || new_draft > self.hi_free() as u64 {
                return false;
            }
            let n = h.list.len() as u64;
            if new_main <= n {
                true
            } else {
                let add_lo = new_main - n;
                add_lo <= self.lo_free() as u64 && self.safe_with(self.place_online(), None, Some((r.table_version, add_lo)), None)
            }
        }
    }

    /// Live run r takes R = r2 rows: its logical pages between the two Rs leave its replicated prefix and join its
    /// CP-sharded tail.
    /// - window mode: list becomes first r2/64 pages, window, fresh drafter pages above place, old drafter tail.
    /// - in-RAM split mode (exclusive): frees (old_main - new_main) pages below place identically on every rank,
    ///   keeping prefix [..rp] and [rp..rp+kept_shard] below place, filling [rp+kept_shard..old_rp] above place for
    ///   the drafter, and leaving [old_rp..] untouched.
    /// - in-RAM split mode (below-R run with worst > r2): converts to an exclusive split run at rp, keeping/growing
    ///   new_main pages below place (freeing excess if n > new_main) and allocating new_draft pages above place.
    /// Keeps table_version and worst case. Err(NoRoom) when shrinkable() is false, Err(BadArgument) for bad r2.
    pub fn shrink_run(&mut self, r: &Run, r2: u32) -> Result<Run, Error> {
        self.pending()?;
        let rp = self.rp_of(r2).ok_or(Error::BadArgument)?;
        if !self.shrinkable(r, r2) {
            return Err(Error::NoRoom);
        }
        let mut h = self.held.remove(&r.table_version).expect("shrinkable() found it");
        let gone: Vec<Kept>;
        if self.window > 0 {
            let (old, w) = (h.rp as usize, self.window as usize);
            let freed: Vec<u32> = h.list[rp as usize..old].to_vec();
            self.mark(&freed, false);
            let mut list: Vec<u32> = h.list[..rp as usize].iter().chain(&h.list[old..old + w]).copied().collect();
            let mut g = self.fill_range(&mut list, old - rp as usize, self.place, self.pages);
            let fresh = list[rp as usize + w..].to_vec();
            list.extend_from_slice(&h.list[old + w..]);
            g.extend(self.evict_for_ids(((list.len() - rp as usize - w) as u64).div_ceil(self.nr as u64)));
            self.spill(&g);
            self.mark(&fresh, true);
            h.list = list;
            h.rp = rp as u32;
            gone = g;
        } else if h.exclusive {
            let old = h.rp as usize;
            let w = h.list.len() as u64;
            let old_main = h.list.iter().filter(|&&p| p < self.place).count() as u64;
            let (new_main, _) = self.exclusive_pages(w * PAGE, rp);
            let d_lo = (old_main - new_main) as usize;
            let kept_shard = (old - rp as usize) - d_lo;
            let freed: Vec<u32> = h.list[rp as usize + kept_shard..old].to_vec();
            self.mark(&freed, false);
            let mut list: Vec<u32> = Vec::with_capacity(w as usize);
            list.extend_from_slice(&h.list[..rp as usize + kept_shard]);
            let g = self.fill_range(&mut list, d_lo, self.place, self.pages);
            let fresh = list[rp as usize + kept_shard..].to_vec();
            list.extend_from_slice(&h.list[old..]);
            self.spill(&g);
            self.mark(&fresh, true);
            h.list = list;
            h.rp = rp as u32;
            gone = g;
        } else {
            let w = h.worst as u64;
            let n = h.list.len() as u64;
            let (new_main, new_draft) = self.exclusive_pages(w * PAGE, rp);
            let mut list: Vec<u32> = Vec::with_capacity(w as usize);
            let mut fresh: Vec<u32> = Vec::new();
            let mut g: Vec<Kept> = Vec::new();
            if n >= new_main {
                let freed: Vec<u32> = h.list[new_main as usize..].to_vec();
                self.mark(&freed, false);
                list.extend_from_slice(&h.list[..new_main as usize]);
            } else {
                list.extend_from_slice(&h.list);
                let add_lo = (new_main - n) as usize;
                g.extend(self.fill_range(&mut list, add_lo, 0, self.place));
                fresh.extend_from_slice(&list[n as usize..]);
            }
            let hi_start = list.len();
            g.extend(self.fill_range(&mut list, new_draft as usize, self.place, self.pages));
            fresh.extend_from_slice(&list[hi_start..]);
            self.spill(&g);
            self.mark(&fresh, true);
            h.list = list;
            h.exclusive = true;
            h.rp = rp as u32;
            h.sp = None;
            gone = g;
        }
        let g = Run {
            first_page: h.list[0],
            pages: h.list.len() as u32,
            table_version: r.table_version,
            list_hash: hash(&h.list, true),
        };
        self.log_claim("shrink", &g, rp * PAGE, &gone);
        self.held.insert(r.table_version, h);
        Ok(g)
    }

    pub fn release(&mut self, r: &Run) -> Result<(), Error> {
        self.remove(r).map(|_| ())
    }

    // whether a grant adopting retained run k's first ceil(skip / 64) pages may write into the last of them: always
    // without sharing; with it, when the hit ends on a page boundary or no other run (live or retained) holds that page
    fn partial_own(&self, k: &Kept, skip: i32) -> bool {
        if !self.share || skip <= 0 || (skip as u64).is_multiple_of(PAGE) {
            return true;
        }
        let Some(&p) = k.list.get(pages_for(skip as u64) as usize - 1) else { return true };
        self.live_ref[p as usize] == 0 && self.kept.iter().filter(|x| x.list.contains(&p)).count() <= 1
    }

    // whether live run h's last page has another holder and may still be written (not known full)
    fn cow_last(&self, h: &Held) -> bool {
        let Some(&p) = h.list.last() else { return false };
        let full = (h.pf as u64) >= h.list.len() as u64 * PAGE;
        !full && (self.live_ref[p as usize] > 1 || self.kept.iter().any(|k| k.list.contains(&p)))
    }

    // GR8_KV_SHARE_PAGES: whether a grant of n pages (worst w) that shares `shared` (its first pages, held by another
    // run) fits: its fresh pages plus the shared ones no live run holds yet come out of the free pages below place,
    // and every live run can still reach its worst case (safe_share)
    fn share_room(&self, shared: &[u32], n: u64, w: u64) -> bool {
        let cost = (n - shared.len() as u64) + shared.iter().filter(|&&p| self.live_ref[p as usize] == 0).count() as u64;
        n <= w && cost <= self.lo_free() as u64 && self.safe_share(self.place_online(), Some((shared, n, w)), None, None)
    }

    // Banker's safety with shared pages (safe_with when sharing): the live runs in order of remaining need, smallest
    // first; when one finishes, each of its pages comes back only when no other live run in the simulation still holds
    // it (sim refs reach 0), so a shared page is charged once and credited once. `new` = (its shared first pages,
    // pages now, worst). Retained runs count as free, as in safe_with.
    fn safe_share(
        &self,
        online: u32,
        new: Option<(&[u32], u64, u64)>,
        grow: Option<(u32, u64)>,
        split: Option<u64>,
    ) -> bool {
        if self.exclusive {
            return new.is_none() && grow.is_none() && split.is_none();
        }
        let place = self.place as usize;
        let mut refs: Vec<u32> = self.live_ref[..place].to_vec();
        let mut free = online as i64 - (self.live - self.live_hi) as i64;
        // (remaining need, its pages below place in refs, pages it adds that no one else holds)
        let mut runs: Vec<(u64, u32, Vec<u32>, u64)> = Vec::with_capacity(self.held.len() + 2);
        for (v, h) in &self.held {
            let add = grow.filter(|g| g.0 == *v).map_or(0, |g| g.1);
            let low: Vec<u32> = h.list.iter().copied().filter(|&p| (p as usize) < place).collect();
            runs.push(((h.worst as u64).saturating_sub(h.list.len() as u64 + add), *v, low, add));
        }
        free -= grow.map_or(0, |g| g.1 as i64);
        if let Some((shared, n, w)) = new {
            for &p in shared {
                if refs[p as usize] == 0 {
                    free -= 1;
                }
                refs[p as usize] += 1;
            }
            let fresh = n - shared.len() as u64;
            free -= fresh as i64;
            runs.push((w.saturating_sub(n), u32::MAX, shared.to_vec(), fresh));
        }
        if let Some(m) = split {
            free -= m as i64;
            runs.push((0, u32::MAX - 1, Vec::new(), m));
        }
        if free < 0 {
            return false;
        }
        runs.sort_unstable_by_key(|r| (r.0, r.1));
        for (need, _, low, fresh) in runs {
            if need as i64 > free {
                return false;
            }
            free += fresh as i64;
            for p in low {
                let r = &mut refs[p as usize];
                *r -= 1;
                if *r == 0 {
                    free += 1;
                }
            }
        }
        true
    }

    /// A live run's pages, in table order.
    pub fn list(&self, r: &Run) -> Result<&[u32], Error> {
        Ok(&self.held(r)?.list)
    }

    /// Whether a live run is exclusive (its table is the CP map as built).
    pub fn exclusive_run(&self, r: &Run) -> Result<bool, Error> {
        Ok(self.held(r)?.exclusive)
    }

    /// Prefix reuse on, with these rules.
    pub fn keep(&mut self, rules: Rules) {
        self.keep = true;
        self.rules = rules;
    }

    pub fn keeping(&self) -> bool {
        self.keep
    }

    /// A retiring session's run: retained with its finished reuse state, or released when no row is valid.
    pub fn retain(&mut self, r: &Run, st: Finished) -> Result<(), Error> {
        self.retain_namespaced(r, st, STOCK_NAMESPACE)
    }

    pub fn retain_namespaced(&mut self, r: &Run, st: Finished, namespace: Namespace) -> Result<(), Error> {
        let (exclusive, hits, span, rp, from) = self.held(r).map(|h| (h.exclusive, h.hits, h.span, h.rp, h.from))?;
        let list = self.remove(r)?;
        if !self.keep || st.valid <= 0 {
            return Ok(());
        }
        let rows_max = if exclusive { rp as u64 * PAGE } else { u64::MAX };
        if namespace == STOCK_NAMESPACE {
            if let Some(s) = &mut self.stash {
                // a run past R promotes at most its prefix (rows_max), which is on pool pages even in a window run
                s.promote(&list, &st, rows_max, span);
            }
        }
        // a window run is retained with its list (prefix, window, drafter tail): the engine keeps its rows store records
        // for it, and a hit adopts them with the pages (take_hit), or they go when the pool evicts it
        self.age += 1;
        let retired_ns = self.now_ns();
        self.kept.push(Kept { list, st, age: self.age, exclusive, hits, namespace, rp, from, retired_ns });
        Ok(())
    }

    /// The retained run with the longest reusable prefix for `prompt` (limit: rows the prefill may skip) that a
    /// reservation of `rows` now and `worst` at most can grow from. Ties go to the most recent.
    pub fn match_prefix(&self, prompt: &[i32], rows: u64, worst: u64, limit: i32) -> Hit {
        self.match_prefix_namespaced(prompt, rows, worst, limit, STOCK_NAMESPACE)
    }

    pub fn match_prefix_namespaced(
        &self,
        prompt: &[i32],
        rows: u64,
        worst: u64,
        limit: i32,
        namespace: Namespace,
    ) -> Hit {
        self.match_prefix_r(prompt, rows, worst, limit, namespace, 0)
    }

    /// match_prefix_namespaced() for a grant at R = r rows (0 = the pool's R). A retained run past another R is no hit
    /// (Why::Room), and an R the pool refuses finds none.
    pub fn match_prefix_r(
        &self,
        prompt: &[i32],
        rows: u64,
        worst: u64,
        limit: i32,
        namespace: Namespace,
        r: u32,
    ) -> Hit {
        let mut best = Hit::default();
        let Some(rp) = self.rp_of(r) else { return best };
        if !self.keep || self.exclusive {
            return best;
        }
        let w = pages_for(worst);
        let n = if past(worst, rp) { w + self.window as u64 } else { pages_for(rows) };
        for k in &self.kept {
            if k.namespace != namespace {
                if best.skip == 0 {
                    best.why = Why::Namespace;
                }
                continue;
            }
            let shared_hit = self.share && !k.exclusive && !past(worst, rp);
            let t = if shared_hit {
                reuse::begin_pool(&self.rules, &k.st, prompt, limit, true)
            } else {
                reuse::begin(&self.rules, &k.st, prompt, limit, true)
            };
            if t.skip <= 0 {
                if best.skip == 0 && t.lcp >= best.lcp {
                    best.lcp = t.lcp;
                    best.why = t.why;
                }
                continue;
            }
            let a = pages_for(t.skip as u64);
            let adopted = &k.list[..(a as usize).min(k.list.len())];
            let mut room = a <= k.list.len() as u64 && a <= n && self.same_layout(k, a, worst, rp);
            if past(worst, rp) {
                room = room && (self.live == 0 || self.nr > 0) && n <= self.pages as u64;
                if self.nr == 0 {
                    room = room && n <= self.prefix_online() as u64;
                    // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                    room = room && adopted.iter().enumerate().all(|(i, &p)| p as usize == i);
                } else if k.exclusive && self.window > 0 {
                    // a retained window run: its prefix, window and drafter pages below the hit, with its records
                    room = room && self.window_hit_room(k, a, worst, rp);
                } else {
                    // its own pages: the adopted ones and fresh ones of both classes must fit
                    room = room && self.hit_room(adopted, a, n, worst, rp);
                }
            } else if shared_hit {
                let full = (t.skip as u64 / PAGE) as usize;
                room = room && adopted.iter().all(|&p| p < self.place) && self.share_room(&k.list[..full], n, w);
            } else {
                room = room && n <= self.lo_free() as u64 && n <= w;
                room = room && adopted.iter().all(|&p| p < self.place) && self.safe(Some((n, w)), None);
            }
            // GR8_KV_SHARE_PAGES: a grant that adopts (moves) a run's pages writes past the hit into its last adopted
            // page: only when no other run holds that page
            room = room && (shared_hit || self.partial_own(k, t.skip));
            if !room {
                if best.skip == 0 {
                    best.why = Why::Room;
                }
                continue;
            }
            if t.skip > best.skip || (t.skip == best.skip && k.age > best.age) {
                best = Hit { skip: t.skip, lcp: t.lcp, why: Why::Hit, age: k.age, stored: false, tier: 0 };
            }
        }
        // GR8_KV_SHARE_PAGES: a live run below R whose prefill passed the hit (prefilled()) shares its pages too. It
        // wins only with more rows than every retained run (a retained hit refreshes its run; a live one has no
        // lease to refresh). Its age is LIVE_HIT | its table version
        if self.share && !past(worst, rp) {
            for (v, h) in &self.held {
                let Some((ns, toks)) = h.sp.as_ref().filter(|_| h.pf > 0 && !h.exclusive) else { continue };
                if *ns != namespace {
                    continue;
                }
                let pf = (h.pf as usize).min(toks.len()) as i32;
                let t = reuse::begin_on(&self.rules, toks, pf, pf, pf, prompt, limit, true, false);
                if t.skip <= best.skip {
                    if best.skip == 0 && t.skip <= 0 && t.lcp >= best.lcp {
                        best.lcp = t.lcp;
                        best.why = t.why;
                    }
                    continue;
                }
                let (a, full) = (pages_for(t.skip as u64), (t.skip as u64 / PAGE) as usize);
                let room = a <= h.list.len() as u64
                    && a <= n
                    && h.list[..a as usize].iter().all(|&p| p < self.place)
                    && self.share_room(&h.list[..full], n, w);
                if !room {
                    if best.skip == 0 {
                        best.why = Why::Room;
                    }
                    continue;
                }
                best = Hit { skip: t.skip, lcp: t.lcp, why: Why::Hit, age: LIVE_HIT | *v as u64, stored: false, tier: 0 };
            }
        }
        // the tiers in order. A stored hit must reuse more rows than the pool's best, since a pool hit is free of copies.
        // A stored entry matches only its own namespace
        let Some(stash) = &self.stash else { return best };
        for t in 0..stash.tiers() {
            for e in stash.entries(t) {
                if e.namespace != namespace {
                    if best.skip == 0 {
                        best.why = Why::Namespace;
                    }
                    continue;
                }
                let b = reuse::begin(&self.rules, &e.st, prompt, limit, true);
                if b.skip <= 0 {
                    if best.skip == 0 && b.lcp >= best.lcp {
                        best.lcp = b.lcp;
                        best.why = b.why;
                    }
                    continue;
                }
                let a = pages_for(b.skip as u64);
                if a > self.restorable(e, worst, rp) || !self.restore_room(a, n, w, worst, e.tail.is_some(), rp) {
                    if best.skip == 0 {
                        best.why = Why::Room;
                    }
                    continue;
                }
                let tier = t as u32;
                if b.skip > best.skip || (best.stored && b.skip == best.skip && tier == best.tier && e.id > best.age) {
                    best = Hit { skip: b.skip, lcp: b.lcp, why: Why::Stored, age: e.id, stored: true, tier };
                }
            }
        }
        best
    }

    // the pages of stored entry e a grant of `worst` rows can restore: a tail only goes back into a run past R
    fn restorable(&self, e: &crate::stash::Entry, worst: u64, rp: u64) -> u64 {
        (if past(worst, rp) { e.pages() } else { e.below() }) as u64
    }

    // whether a grant of n pages (worst w) can restore its first a pages. Below R it is as reserve() makes it. Past R it
    // needs an empty pool, and the restored pages must lie inside the CP map's replicated start unless the entry has
    // its tail
    fn restore_room(&self, a: u64, n: u64, w: u64, worst: u64, tail: bool, rp: u64) -> bool {
        if a > n {
            return false;
        }
        if past(worst, rp) {
            let room = (self.live == 0 || self.nr > 0)
                && !self.exclusive
                && n <= self.pages as u64
                // a window run's list past its prefix is its window, not its logical tail, and stored tails are laid
                // out at the pool's R only: only the prefix restores otherwise
                && ((tail && self.window == 0 && rp == self.rp0()) || a <= rp);
            if self.nr == 0 || !room {
                return room && n <= self.prefix_online() as u64;
            }
            // its own pages: the two classes must fit beside the live runs (a fresh grant restores into them)
            let (main, draft) = self.exclusive_pages(worst, rp);
            return self.split_room(main, draft);
        }
        n <= self.lo_free() as u64 && n <= w && self.safe(Some((n, w)), None)
    }

    /// Claims a stored hit: a fresh grant as reserve() makes it, with the entry's first ceil(skip / 64) blocks queued
    /// for restore into its first pages.
    fn take_stored(&mut self, skip: i32, id: u64, tier: usize, rows: u64, worst: u64, rp: u64) -> Result<Run, Error> {
        let e = self.stash.as_ref().and_then(|s| s.find(tier, id)).ok_or(Error::NotRetained)?;
        let w = pages_for(worst);
        let (a, n) = (pages_for(skip as u64), if past(worst, rp) { w + self.window as u64 } else { pages_for(rows) });
        if a > self.restorable(e, worst, rp) || !self.restore_room(a, n, w, worst, e.tail.is_some(), rp) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let e = self.stash.as_mut().expect("found in it").begin_restore(tier, id);
        let hits = self.hits_from(&e.st, e.hits, e.namespace);
        let cont = self.next_continues_run(&e.st, e.namespace);
        self.earn(n, a);
        let (list, gone, exclusive, pool_base) = if past(worst, rp) {
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                ((0..n as u32).collect::<Vec<u32>>(), gone, true, false)
            } else {
                // its own pages, a fresh two-class grant; the entry's blocks (its tail included) restore into them.
                // Every run the classes evict spills, from either class.
                let (main, draft) = self.exclusive_pages(worst, rp);
                let mut list = Vec::with_capacity(n as usize);
                let mut gone = self.fill_range(&mut list, main as usize, 0, self.place);
                gone.extend(self.fill_range(&mut list, draft as usize, self.place, self.pages));
                if self.window > 0 {
                    gone.extend(self.evict_for_ids(draft.div_ceil(self.nr as u64)));
                }
                self.spill(&gone);
                (list, gone, true, false)
            }
        } else {
            // the entry's base from a retained run that holds it, when one does: those pages need no copy
            let mut list = Vec::with_capacity(n as usize);
            let from_pool = self.base_in_pool(&e, a);
            if let Some(i) = from_pool {
                let k = self.kept.remove(i);
                let b = e.base.expect("a pool base is a base").1;
                // what of k a tier would store (stash.rs storable) past the pages the grant adopts
                if self.stored_rows(&k) > b as u64 * PAGE {
                    self.spill(std::slice::from_ref(&k));
                } else {
                    self.let_go(&k);
                }
                list.extend_from_slice(&k.list[..b]);
            }
            let gone = self.fill(&mut list, n as usize);
            self.spill(&gone);
            (list, gone, false, from_pool.is_some())
        };
        // a past-R grant restores its tail into pages past the hit's, so it passes its whole list
        let into = if exclusive { &list[..] } else { &list[..a as usize] };
        self.stash.as_mut().expect("found in it").restore(tier, &e, into, pool_base, exclusive);
        // the grant's pages below the hit's full pages are byte copies of e's (D2: its store extends e in place).
        // Only for a run that continues e: it then holds all e holds, and replaces it
        if !exclusive && cont {
            self.next_from = Some((e.id, ((skip as u64 / PAGE) as usize).min(e.below())));
        }
        let (rows, how) = (if exclusive { worst } else { rows }, if pool_base { "stored+pool" } else { "stored" });
        Ok(self.grant(list, exclusive, w as u32, rows, how, &gone, hits, rp))
    }

    // the retained run (its index in kept) whose first pages hold the base of stored entry e, when e restores past it
    fn base_in_pool(&self, e: &crate::stash::Entry, a: u64) -> Option<usize> {
        let (id, b) = e.base?;
        if (b as u64) >= a {
            return None;
        }
        let hist = self.stash.as_ref()?.base_history(id)?;
        let rows = (b as u64 * PAGE) as i32;
        self.kept.iter().rposition(|k| {
            k.namespace == STOCK_NAMESPACE
                && k.st.valid >= rows
                && k.st.history.len() >= hist.len()
                && k.st.history[..hist.len()] == *hist
                && k.list.len() >= b
                && k.list[..b].iter().all(|&p| p < self.place)
        })
    }

    // a retained run leaves the pool unspilled (a hit holds all a tier would store of it): its conversation's head is
    // remembered, as put_evicted does for a spilled one, so the conversation's next prompt still counts as continuing
    fn let_go(&mut self, k: &Kept) {
        if let Some(s) = &mut self.stash {
            if k.namespace == STOCK_NAMESPACE {
                s.forget(&k.st);
            }
        }
    }

    // the rows of retained run k a tier would store when the pool evicts it (stash.rs storable; a run past R keeps
    // its valid rows, its tail is not cut)
    fn stored_rows(&self, k: &Kept) -> u64 {
        match &self.stash {
            Some(s) if !k.exclusive => s.storable(&k.st),
            _ => k.st.valid.max(0) as u64,
        }
    }

    /// Claims a match_prefix() hit (skip, age) as a new grant: the retained run's first ceil(skip / 64) pages, then
    /// pages as reserve() takes them.
    pub fn take(&mut self, skip: i32, age: u64, rows: u64, worst: u64) -> Result<Run, Error> {
        self.take_hit(skip, age, None, rows, worst)
    }

    /// take() for a hit that may be stored (from = its tier).
    pub fn take_hit(&mut self, skip: i32, age: u64, from: Option<u32>, rows: u64, worst: u64) -> Result<Run, Error> {
        self.take_hit_r(skip, age, from, rows, worst, 0)
    }

    /// take_hit() for a match_prefix_r() hit at R = r rows (0 = the pool's R). Err(BadArgument) for an R the pool
    /// refuses.
    pub fn take_hit_r(
        &mut self,
        skip: i32,
        age: u64,
        from: Option<u32>,
        rows: u64,
        worst: u64,
        r: u32,
    ) -> Result<Run, Error> {
        self.pending()?;
        let rp = self.rp_of(r).ok_or(Error::BadArgument)?;
        if skip <= 0 || !self.admissible_at(worst, rp) {
            return Err(Error::NoHit);
        }
        if self.over_run_limit(worst) {
            return Err(Error::NoRoom);
        }
        if let Some(t) = from {
            let r = self.take_stored(skip, age, t as usize, rows, worst, rp);
            if r.is_ok() && self.ages_on {
                self.seen.insert(age, Instant::now());
            }
            return r;
        }
        if self.share && age & LIVE_HIT != 0 {
            return self.take_live(skip, (age & !LIVE_HIT) as u32, rows, worst, rp);
        }
        let i = self.kept.iter().position(|k| k.age == age).ok_or(Error::NotRetained)?;
        if self.share && !self.kept[i].exclusive && !past(worst, rp) {
            return self.take_kept_shared(i, skip, rows, worst, rp);
        }
        let w = pages_for(worst);
        let (a, n) = (pages_for(skip as u64), if past(worst, rp) { w + self.window as u64 } else { pages_for(rows) });
        // checks the C++ pool left to match(): a hit from match() on this state always passes them
        let k = &self.kept[i];
        if a > k.list.len() as u64 || a > n {
            return Err(Error::NoHit);
        }
        if !self.same_layout(k, a, worst, rp) || !self.partial_own(k, skip) {
            return Err(Error::NoRoom);
        }
        let adopted = &k.list[..a as usize];
        if past(worst, rp) {
            if n > self.pages as u64 || (self.nr == 0 && n > self.prefix_online() as u64) {
                return Err(Error::NoRoom);
            }
            if self.nr == 0 {
                // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                if adopted.iter().enumerate().any(|(i, &p)| p as usize != i) {
                    return Err(Error::NoRoom);
                }
            } else if k.exclusive && self.window > 0 {
                if !self.window_hit_room(k, a, worst, rp) {
                    return Err(Error::NoRoom);
                }
            } else if !self.hit_room(adopted, a, n, worst, rp) {
                return Err(Error::NoRoom);
            }
        } else if n > self.lo_free() as u64
            || n > w
            || adopted.iter().any(|&p| p >= self.place)
            || !self.safe(Some((n, w)), None)
        {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let k = self.kept.remove(i);
        let hits = self.hits_from(&k.st, k.hits, k.namespace);
        let full = (skip as u64 / PAGE) as usize;
        // the grant's full pages below the hit are k's: k's provenance carries over to them (Kept::from)
        let from = k.from.filter(|_| self.next_continues_run(&k.st, k.namespace)).map(|(id, n)| (id, n.min(full)));
        self.earn(n, a);
        // the grant keeps the run's first a pages and writes past its skip: what the run held beyond that goes to a
        // tier, read before the grant writes. Below R, only what a tier would store of it (stash.rs storable) counts;
        // a hit on the run's own conversation has no rows past skip and spills nothing.
        // With no reuse grid, `skip` may end inside a page. The grant will rewrite that page,
        // but the retained run has nothing *past* the hit when its valid rows end at `skip`.
        let spills_past_hit = if k.exclusive { k.st.valid > skip } else { self.stored_rows(&k) > skip as u64 };
        if spills_past_hit {
            self.spill(std::slice::from_ref(&k));
        } else {
            self.let_go(&k);
        }
        if past(worst, rp) && k.exclusive && self.window > 0 {
            // a retained window run: its prefix, window and drafter pages below the hit, then the drafter's fresh ones
            let mut list = self.window_adopt(&k, a).expect("window_hit_room checked it");
            let (main, draft) = self.exclusive_pages(worst, rp);
            let fresh = (draft - (list.len() as u64 - main)) as usize;
            let mut gone = self.fill_range(&mut list, fresh, self.place, self.pages);
            gone.extend(self.evict_for_ids(draft.div_ceil(self.nr as u64)));
            self.spill(&gone);
            return Ok(self.grant(list, true, w as u32, worst, "hit", &gone, hits, rp));
        }
        let mut list = k.list;
        list.truncate(a as usize);
        if past(worst, rp) {
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                list.extend(a as u32..n as u32);
                return Ok(self.grant(list, true, w as u32, worst, "hit", &gone, hits, rp));
            }
            // its own pages: the adopted ones, then the two classes fill the rest (the low ones first, so the run's
            // pages below place stay a prefix of its low pages in list order)
            let gone = self.fill_hit(&mut list, a, n, worst, rp);
            self.spill(&gone);
            return Ok(self.grant(list, true, w as u32, worst, "hit", &gone, hits, rp));
        }
        let gone = self.fill(&mut list, n as usize);
        self.spill(&gone);
        self.next_from = from;
        Ok(self.grant(list, false, w as u32, rows, "hit", &gone, hits, rp))
    }

    // GR8_KV_SHARE_PAGES hit on retained run i (below R): the run stays retained, its retirement refreshed (it is
    // retired again now, so the oldest-first eviction reaches it last); the grant shares its full pages below the hit
    // and copies its partial page (PageCopy) into a fresh one
    fn take_kept_shared(&mut self, i: usize, skip: i32, rows: u64, worst: u64, rp: u64) -> Result<Run, Error> {
        let w = pages_for(worst);
        let (a, n, full) = (pages_for(skip as u64), pages_for(rows), (skip as u64 / PAGE) as usize);
        let k = &self.kept[i];
        if a > k.list.len() as u64 || a > n {
            return Err(Error::NoHit);
        }
        if k.list[..a as usize].iter().any(|&p| p >= self.place) || !self.share_room(&k.list[..full], n, w) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let mut k = self.kept.remove(i);
        let hits = self.hits_from(&k.st, k.hits, k.namespace);
        let from = k.from.filter(|_| self.next_continues_run(&k.st, k.namespace)).map(|(id, m)| (id, m.min(full)));
        self.earn(n, a);
        let src = (a as usize > full).then(|| k.list[a as usize - 1]);
        let list = k.list[..full].to_vec();
        self.age += 1;
        k.age = self.age;
        k.retired_ns = self.now_ns();
        self.kept.push(k);
        Ok(self.grant_shared(list, src, skip, n, w, rows, hits, from, rp, "share"))
    }

    // GR8_KV_SHARE_PAGES hit on live run v (below R, prefilled past the hit): the grant shares its full pages below the
    // hit and copies its partial page, which the live run goes on writing
    fn take_live(&mut self, skip: i32, v: u32, rows: u64, worst: u64, rp: u64) -> Result<Run, Error> {
        let w = pages_for(worst);
        let (a, n, full) = (pages_for(skip as u64), pages_for(rows), (skip as u64 / PAGE) as usize);
        let h = self.held.get(&v).ok_or(Error::NotRetained)?;
        if h.exclusive || past(worst, rp) || skip as u64 > h.pf as u64 || a > h.list.len() as u64 || a > n {
            return Err(Error::NoHit);
        }
        if h.list[..a as usize].iter().any(|&p| p >= self.place) || !self.share_room(&h.list[..full], n, w) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let src = (a as usize > full).then(|| h.list[a as usize - 1]);
        let list = h.list[..full].to_vec();
        self.earn(n, a);
        Ok(self.grant_shared(list, src, skip, n, w, rows, 0, None, rp, "share-live"))
    }

    // the shared grant: `list` = the shared full pages, then fresh pages to n (the first of them the copy of `src`,
    // the source's partial page, when the hit ends inside it)
    #[allow(clippy::too_many_arguments)]
    fn grant_shared(
        &mut self,
        mut list: Vec<u32>,
        src: Option<u32>,
        skip: i32,
        n: u64,
        w: u64,
        rows: u64,
        hits: u32,
        from: Option<(u64, usize)>,
        rp: u64,
        how: &str,
    ) -> Run {
        let full = list.len();
        self.adopting = list.clone();
        let gone = self.fill(&mut list, n as usize);
        self.spill(&gone);
        self.adopting.clear();
        if let Some(from) = src {
            let to = list[full];
            // the source left in this fill (its run evicted): its page may come back as the copy itself
            if to != from {
                self.copies.push(PageCopy { from, to, rows: (skip as u64 % PAGE) as u32 });
            }
        }
        self.next_from = from;
        self.grant(list, false, w as u32, rows, how, &gone, hits, rp)
    }

    /// Every rank's pool must agree: folded into Scheduler::digest at each event boundary.
    pub fn digest(&self) -> u64 {
        let mut h = mix(mix(mix(FNV_OFFSET, self.version as u64), self.age), self.exclusive as u64);
        if self.window > 0 {
            h = mix(h, 0x7274_0000_0000_0000 | self.window as u64);   // a pool without the window hashes as before
            h = mix(h, self.ids as u64);
            h = mix(h, self.runs as u64);
        }
        if self.run_limit > 0 {
            h = mix(h, 0x726c_0000_0000_0000 | self.run_limit as u64);   // a pool without the limit hashes as before
        }
        for (p, _) in self.used.iter().enumerate().filter(|(_, &u)| u) {
            h = mix(h, p as u64);
        }
        for (v, x) in &self.held {
            h = mix(mix(h, *v as u64), hash(&x.list, x.exclusive));
            // a run that may still grow: its worst case is state every rank must agree on (a run granted its worst case
            // at once hashes as it did before growth existed)
            if x.worst as usize > x.list.len() {
                h = mix(h, x.worst as u64);
            }
            // a run at another R than the pool's (a pool of one R hashes as before per-grant R existed)
            if x.rp as u64 != self.rp0() {
                h = mix(h, RP_TAG | x.rp as u64);
            }
        }
        for k in &self.kept {
            h = mix(mix(mix(h, k.age), hash(&k.list, false)), k.st.valid as i64 as u64);
            if k.rp as u64 != self.rp0() {
                h = mix(h, RP_TAG | k.rp as u64);
            }
            // Keep the pre-namespace digest bit-for-bit stable for stock runs. A non-stock identity contributes every
            // namespace byte so ranks still fail closed if their retained cache identities diverge.
            if k.namespace != STOCK_NAMESPACE {
                for chunk in k.namespace.chunks(8) {
                    let mut v = 0u64;
                    for (i, &b) in chunk.iter().enumerate() {
                        v |= (b as u64) << (8 * i);
                    }
                    h = mix(h, v);
                }
            }
        }
        // GR8_KV_SHARE_PAGES: every page's live and retained holders, each live run's prefilled rows and shared
        // prompt, and the queued copies (without sharing the digest is what it was before)
        if self.share {
            h = mix(h, 0x7368_0000_0000_0000);
            let kr = self.kept_ref();
            for p in 0..self.pages as usize {
                if self.live_ref[p] != 0 || kr[p] != 0 {
                    h = mix(h, (p as u64) << 32 | (self.live_ref[p] as u64) << 16 | kr[p] as u64);
                }
            }
            for (v, x) in &self.held {
                h = mix(mix(h, *v as u64), x.pf as u64);
                if let Some((ns, t)) = &x.sp {
                    h = mix(h, t.len() as u64);
                    h = t.iter().fold(h, |h, &x| mix(h, x as u32 as u64));
                    h = ns.iter().fold(h, |h, &b| mix(h, b as u64));
                }
            }
            for c in &self.copies {
                h = mix(mix(mix(h, c.from as u64), c.to as u64), c.rows as u64);
            }
        }
        // offline chunks (resize): with every chunk online the digest is what it was before resize existed
        if self.offline.iter().any(|&o| o) {
            h = mix(h, self.chunk as u64);
            for k in (0..self.offline.len()).filter(|&k| self.offline[k]) {
                h = mix(h, k as u64);
            }
        }
        // without a store the digest is what it was before stores existed
        match &self.stash {
            Some(s) => {
                for x in self.held.values() {
                    h = mix(h, x.hits as u64);
                    if let Some((id, n)) = x.from {
                        h = mix(mix(h, id), n as u64);
                    }
                }
                for k in &self.kept {
                    h = mix(h, k.hits as u64);
                    if let Some((id, n)) = k.from {
                        h = mix(mix(h, id), n as u64);
                    }
                }
                s.fold(h, mix)
            }
            None => h,
        }
    }

    /// One stderr line per claim (g6m4: GR8_POOL_LOG=1).
    pub fn log(&mut self, on: bool) {
        self.log = on;
    }

    pub fn kept_runs(&self) -> u32 {
        self.kept.len() as u32
    }

    pub fn stats(&self) -> Stats {
        Stats {
            live_runs: self.held.len() as u32,
            held_pages: self.live,
            worst_pages: self.held.values().map(|h| h.worst).sum(),
            grows: self.grows,
            stalls: self.stalls,
        }
    }

    /// Each live run, by table version: (table_version, pages held, worst-case pages).
    pub fn runs(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        self.held.iter().map(|(v, h)| (*v, h.list.len() as u32, h.worst))
    }

    /// Banker's safety (one resource, no preemption): with `new` = (pages now, worst) admitted and `grow` = (table
    /// version, pages) added to a live run, can every live run still reach its worst case? Runs finish in order of
    /// their remaining need, smallest first, each returning what it holds. Retained runs count as free.
    pub fn safe(&self, new: Option<(u64, u64)>, grow: Option<(u32, u64)>) -> bool {
        self.safe_with(self.place_online(), new, grow, None)
    }

    // safe() per page class. Only runs below R ever need more pages, all below place, so the drafter's class is safe
    // as soon as it fits; below place the Banker's order runs over every live run's pages there. `split` = the pages
    // below place of a new split run (its prefix and shard: held at once, need 0). `online` = the online pages below
    // place.
    fn safe_with(&self, online: u32, new: Option<(u64, u64)>, grow: Option<(u32, u64)>, split: Option<u64>) -> bool {
        if self.share {
            return self.safe_share(online, new.map(|(n, w)| (&[][..], n, w)), grow, split);
        }
        if self.exclusive {
            return new.is_none() && grow.is_none() && split.is_none();
        }
        let mut free = online as i64 - (self.live - self.live_hi) as i64;
        let place = self.place;
        let mut runs: Vec<(u64, u64)> = self
            .held
            .iter()
            .map(|(v, h)| {
                let add = grow.filter(|g| g.0 == *v).map_or(0, |g| g.1);
                let low = h.list.iter().filter(|&&p| p < place).count() as u64 + add;
                ((h.worst as u64).saturating_sub(h.list.len() as u64 + add), low)
            })
            .collect();
        free -= grow.map_or(0, |g| g.1 as i64);
        if let Some((n, w)) = new {
            free -= n as i64;
            runs.push((w.saturating_sub(n), n));
        }
        if let Some(m) = split {
            free -= m as i64;
            runs.push((0, m));
        }
        if free < 0 {
            return false;
        }
        runs.sort_unstable();
        for (need, held) in runs {
            if need as i64 > free {
                return false;
            }
            free += held as i64;
        }
        true
    }

    /// The retained runs, oldest retired first.
    pub fn kept(&self) -> &[Kept] {
        &self.kept
    }

    /// Pages the live runs hold.
    pub fn live_pages(&self) -> u32 {
        self.live
    }

    pub fn pool_pages(&self) -> u32 {
        self.pages
    }

    pub fn place_pages(&self) -> u32 {
        self.place
    }

    /// Whether page p is held by a live run.
    pub fn used(&self, p: u32) -> bool {
        self.used.get(p as usize).copied().unwrap_or(false)
    }

    fn check_version(&self) -> Result<(), Error> {
        if self.version == u32::MAX {
            Err(Error::VersionExhausted)
        } else {
            Ok(())
        }
    }

    fn held(&self, r: &Run) -> Result<&Held, Error> {
        let h = self.held.get(&r.table_version).ok_or(Error::UnknownRun)?;
        let now = Run {
            first_page: h.list[0],
            pages: h.list.len() as u32,
            table_version: r.table_version,
            list_hash: hash(&h.list, h.exclusive),
        };
        if now == *r {
            Ok(h)
        } else {
            Err(Error::UnknownRun)
        }
    }

    // a live run leaves: its pages are free again
    fn remove(&mut self, r: &Run) -> Result<Vec<u32>, Error> {
        self.held(r)?;
        let h = self.held.remove(&r.table_version).expect("held() found it");
        self.mark(&h.list, false);
        if h.exclusive {
            self.exclusive = false;
        }
        Ok(h.list)
    }

    // Callers have checked every page: free, below pages, and no page twice.
    // With sharing a page may be in several live lists: live_ref counts them, and live / live_hi count each page once
    // (while live_ref > 0). Without, live_ref is 0 or 1 and this is the single-owner mark it was.
    fn mark(&mut self, list: &[u32], on: bool) {
        for &p in list {
            let i = p as usize;
            debug_assert!(self.share || self.used[i] != on);
            debug_assert!(on || self.live_ref[i] > 0);
            let edge = if on {
                self.live_ref[i] += 1;
                self.live_ref[i] == 1
            } else {
                self.live_ref[i] -= 1;
                self.live_ref[i] == 0
            };
            if edge {
                self.used[i] = on;
                let hi = (p >= self.place) as u32;
                self.live = if on { self.live + 1 } else { self.live - 1 };
                self.live_hi = if on { self.live_hi + hi } else { self.live_hi - hi };
            }
        }
    }

    // `list` (adopted pages, if any) grows by `need` pages out of [lo, hi): the lowest pages no run holds or retains,
    // after evicting the least recently retired runs until there are enough of them. Returns the evicted runs.
    fn fill_range(&mut self, list: &mut Vec<u32>, need: usize, lo: u32, hi: u32) -> Vec<Kept> {
        let (lo, hi) = (lo as usize, hi as usize);
        // claims[p]: 1 when a live run or the adopted list holds p, plus one per retained run that holds it. A page
        // is spare at 0, and evicting a run only ever takes back its own claims.
        // an offline page is claimed for good: no retained run holds one, so no eviction frees it
        let mut claims: Vec<u32> = (lo..hi).map(|p| (self.used[p] || !self.online(p as u32)) as u32).collect();
        for &p in list.iter() {
            if (p as usize) >= lo && (p as usize) < hi {
                claims[p as usize - lo] = 1;
            }
        }
        for k in &self.kept {
            for &p in k.list.iter().filter(|&&p| (p as usize) >= lo && (p as usize) < hi) {
                claims[p as usize - lo] += 1;
            }
        }
        let mut spare = claims.iter().filter(|&&c| c == 0).count();
        let mut evict = 0;
        while spare < need {
            // the room checks before fill_range() guarantee enough pages once every retained run is gone
            let k = &self.kept[evict];
            for &p in k.list.iter().filter(|&&p| (p as usize) >= lo && (p as usize) < hi) {
                claims[p as usize - lo] -= 1;
                if claims[p as usize - lo] == 0 {
                    spare += 1;
                }
            }
            evict += 1;
        }
        let gone: Vec<Kept> = self.kept.drain(..evict).collect();
        list.extend((lo as u32..hi as u32).filter(|&p| claims[p as usize - lo] == 0).take(need));
        gone
    }

    // `list` (adopted pages, if any) grows to n pages below place: the lowest pages no run holds or retains, after
    // evicting the least recently retired runs until there are enough of them. Returns the evicted runs.
    fn fill(&mut self, list: &mut Vec<u32>, n: usize) -> Vec<Kept> {
        let need = n - list.len();
        let place = self.place;
        self.fill_range(list, need, 0, place)
    }

    // Whether a past-R hit's grant works with its own pages (nr > 0): the adopted low pages plus fresh ones of both
    // classes must fit, the fresh low pages within the positions the grant still has (hit_room's caller checked
    // a <= n and n <= pages).
    // The adopted pages are retained (no live run holds them), so they count among each class's free pages.
    fn hit_room(&self, adopted: &[u32], a: u64, n: u64, worst: u64, rp: u64) -> bool {
        let (main, _draft) = self.exclusive_pages(worst, rp);
        let alt = adopted.iter().filter(|&&p| p < self.place).count() as u64;
        let fresh_lt = main.saturating_sub(alt);
        fresh_lt <= n - a && self.split_room(alt + fresh_lt, n - alt - fresh_lt)
    }

    // whether a split run holding `low` pages below place and `high` above it fits beside the live runs, with every
    // run below R still able to reach its worst case (safe_with)
    fn split_room(&self, low: u64, high: u64) -> bool {
        // a window run's tail also needs its records in every rank's lease (high = its tail pages)
        let ids_ok = self.window == 0 || self.ids_room().map_or(true, |f| high.div_ceil(self.nr as u64) <= f);
        // and a share of the slot pool: the engine sized it for `runs` sessions past R in one step
        let runs_ok = self.window == 0 || self.runs == 0 || self.window_runs() < self.runs;
        !self.exclusive
            && ids_ok
            && runs_ok
            && low <= self.lo_free() as u64
            && high <= self.hi_free() as u64
            && self.safe_with(self.place_online(), None, None, Some(low))
    }

    // take_hit's past-R grant (nr > 0): the list grows from a to n pages, the low class first so the run's pages
    // below place stay, in list order, its prefix, its shard and then the drafter's tail pages. Returns the evicted.
    fn fill_hit(&mut self, list: &mut Vec<u32>, a: u64, n: u64, worst: u64, rp: u64) -> Vec<Kept> {
        let (main, _draft) = self.exclusive_pages(worst, rp);
        let alt = list.iter().filter(|&&p| p < self.place).count() as u64;
        let fresh_lt = main.saturating_sub(alt).min(n - a);
        let fresh_gt = n - a - fresh_lt;
        let mut gone = self.fill_range(list, fresh_lt as usize, 0, self.place);
        if fresh_gt > 0 {
            gone.extend(self.fill_range(list, fresh_gt as usize, self.place, self.pages));
        }
        gone
    }

    #[allow(clippy::too_many_arguments)]
    fn grant(
        &mut self,
        list: Vec<u32>,
        exclusive: bool,
        worst: u32,
        rows: u64,
        how: &str,
        gone: &[Kept],
        hits: u32,
        rp: u64,
    ) -> Run {
        self.mark(&list, true);
        self.version += 1;
        let r = Run {
            first_page: list[0],
            pages: list.len() as u32,
            table_version: self.version,
            list_hash: hash(&list, exclusive),
        };
        self.log_claim(how, &r, rows, gone);
        let tokens = self.next_tokens.take().unwrap_or_default();
        self.next_len = None;
        self.next_prompt = None;
        let span = self.next_span.take();
        // a run that continues its conversation counts as continued once, even when it found nothing to reuse
        let hits = hits.max(std::mem::take(&mut self.next_continues) as u32);
        let from = self.next_from.take();
        let rp = if exclusive { rp } else { self.rp0() } as u32;
        let sp = self.next_share.take();
        self.held.insert(r.table_version, Held { tokens, hits, span, list, exclusive, worst, rp, from, sp, pf: 0 });
        // with the page classes on (nr > 0) a split run holds only its own pages; without, the whole pool
        self.exclusive = self.exclusive || (exclusive && self.nr == 0);
        r
    }

    fn log_claim(&self, how: &str, r: &Run, rows: u64, gone: &[Kept]) {
        if !self.log {
            return;
        }
        let mut s = format!(
            "# s2 pool claim {how} first {} pages {} rows {rows} live {} kept {} evicted {}",
            r.first_page,
            r.pages,
            self.live,
            self.kept.len(),
            gone.len()
        );
        for k in gone {
            let _ = write!(s, " [{}+{} age {} valid {}]", k.list[0], k.list.len(), k.age, k.st.valid);
        }
        s.push('\n');
        let _ = std::io::stderr().write_all(s.as_bytes());
    }
}
