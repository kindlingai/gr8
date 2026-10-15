//! The S2 page allocator below R (the pool every rank runs). See lib.rs for the rules it follows.

use crate::reuse::{self, Finished, Rules, Why};
use crate::stash::{continues, Index, Io, Keep, Policy, Stash, TierStats};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;

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

/// A retired run: its pages count as free until a reservation evicts it or a hit adopts its start.
pub struct Kept {
    pub list: Vec<u32>,
    pub st: Finished,
    pub age: u64,
    pub exclusive: bool,
    /// times the conversation continued a run to get here (a tier's min_hits): a hit on its run, or a prompt that holds
    /// an earlier run's prompt
    pub hits: u32,
}

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

pub struct Pool {
    pages: u32,
    place: u32,
    below_rows: u32,
    // ranks of the CP map past R (tail()): 0 = a run past R takes the whole pool as pages 0, 1, .. (the CP map as
    // built); above 0 it takes its own pages in two classes: the prefix (R/64 pages) and its shard (ceil of the tail
    // pages / nr) from [0, place), the rest (the drafter's copy of the tail) from [place, pages)
    nr: u32,
    version: u32,
    live: u32,
    used: Vec<bool>,
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
    next_continues: bool, // count() found the next grant continues a conversation
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
            version: 0,
            live: 0,
            used: vec![false; pool_pages as usize],
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
        })
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

    /// Runs past R keep their tails in the stores (stash.rs set_tail): nr ranks, per_slot replicated parts a block.
    /// nr > 0 also turns the page classes on for the pool itself (see Pool::nr), so this is set whenever the pool's R
    /// is the CP map's, stores or not.
    pub fn tail(&mut self, nr: u32, per_slot: u32) {
        self.nr = nr;
        if let Some(s) = &mut self.stash {
            s.set_tail(nr, per_slot, self.place);
        }
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
            let prompt = self.next_tokens.as_ref().map_or(n, |t| pages_for(t.len() as u64).min(n));
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
        let Some(s) = &self.stash else { return 0 };
        self.next_tokens = Some(prompt.to_vec());
        self.next_span = span.map(|x| s.span_rows(x, prompt.len()));
        if !s.counting() {
            return 0;
        }
        let continuing = self.held.values().any(|h| continues(prompt, &h.tokens, h.tokens.len().saturating_sub(1)))
            || self.kept.iter().any(|k| continues(prompt, &k.st.history, k.st.pf_end.max(0) as usize))
            || s.continues_stored(prompt);
        self.next_continues = continuing;
        let span = self.next_span;
        self.stash.as_mut().expect("checked").count(prompt, continuing, span)
    }

    /// The queued copies (stash.rs Io), in the order to run them. The engine calls io_done() after it ran them.
    pub fn io(&mut self) -> Vec<Io> {
        self.stash.as_mut().map_or_else(Vec::new, Stash::take_io)
    }

    /// After the copies from io(): commits the index records of the entries they wrote. An error leaves the pool as it
    /// is. The entries stay, and only a restart misses them.
    pub fn io_done(&mut self) -> std::io::Result<()> {
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
        let below = (self.below_rows as u64 / PAGE) as usize;
        let Some(s) = &mut self.stash else { return };
        if gone.is_empty() {
            return;
        }
        let resident: Vec<&[i32]> = self
            .held
            .values()
            .map(|h| &h.tokens[..])
            .chain(self.kept.iter().map(|k| &k.st.history[..]))
            .chain(self.next_tokens.as_deref())
            .filter(|t| !t.is_empty())
            .collect();
        s.put_evicted(gone, below, &resident);
    }

    // hits of a run granted from `src` (its finished state and hits): one more when the prompt continues it
    fn hits_from(&self, st: &Finished, hits: u32) -> u32 {
        match &self.next_tokens {
            Some(p) if continues(p, &st.history, st.pf_end.max(0) as usize) => hits.saturating_add(1),
            _ => 0,
        }
    }

    pub fn exclusive(&self, rows: u64) -> bool {
        rows > self.below_rows as u64
    }

    // A run past R (nr > 0) takes its own pages: of its L = ceil(worst / 64) pages, the prefix (R / 64) and its shard
    // of the tail (ceil of the tail pages / nr) come from [0, place), the tail pages the drafter needs and this rank
    // does not hold from [place, pages). The lists are the same on every rank; what each page holds differs.
    fn exclusive_pages(&self, worst: u64) -> (u64, u64) {
        let rp = self.below_rows as u64 / PAGE;
        let t = pages_for(worst).saturating_sub(rp);
        let own = t.div_ceil(self.nr.max(1) as u64).min(t);
        (rp + own, t - own)
    }

    pub fn free_pages(&self) -> u32 {
        if self.exclusive {
            0
        } else {
            self.pages - self.live
        }
    }

    /// The most pages one reservation can take now: an empty pool takes an exclusive run of up to all its pages.
    pub fn room(&self) -> u32 {
        if self.exclusive {
            0
        } else if self.live != 0 {
            self.place.wrapping_sub(self.live)
        } else {
            self.pages
        }
    }

    /// Head, before reserve: a past-R request (by its worst case) waits for an empty pool.
    pub fn admissible(&self, worst: u64) -> bool {
        !self.exclusive && (!self.exclusive(worst) || self.live == 0)
    }

    /// Whether a request that needs `rows` now and at most `worst` rows can be admitted: an exclusive run takes its
    /// worst case at once from an empty pool, and a run below R takes `rows` and must leave the pool safe (safe()).
    pub fn fits(&self, rows: u64, worst: u64) -> bool {
        let (n, w) = (pages_for(rows), pages_for(worst));
        if rows == 0 || n > w || !self.admissible(worst) {
            return false;
        }
        if self.exclusive(worst) {
            if self.nr == 0 {
                return w <= self.room() as u64;
            }
            // its own pages: both classes must have room in an empty pool (a reservation evicts retained runs as it
            // needs them, so every page of a class is free or evictable)
            let (main, draft) = self.exclusive_pages(worst);
            return main <= self.place as u64 && draft <= (self.pages - self.place) as u64;
        }
        n <= self.room() as u64 && w <= self.place as u64 && self.safe(Some((n, w)), None)
    }

    /// A run of pages for `rows` now that may grow (grow()) to `worst` rows. An exclusive run holds its worst case.
    pub fn reserve(&mut self, rows: u64, worst: u64) -> Result<Run, Error> {
        self.pending()?;
        if !self.fits(rows, worst) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let w = pages_for(worst) as u32;
        if self.exclusive(worst) {
            self.earn(w as u64, 0);
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                return Ok(self.grant((0..w).collect(), true, w, worst, "exclusive", &gone, 0));
            }
            // its own pages: the prefix and its shard of the tail from below place, the drafter's copy of the
            // rest from above it. Retained runs give their pages back only as the classes need them, and every
            // one the classes evict spills (either class may be the one that runs out of room).
            let (main, draft) = self.exclusive_pages(worst);
            let mut list = Vec::with_capacity(w as usize);
            let mut gone = self.fill_range(&mut list, main as usize, 0, self.place);
            gone.extend(self.fill_range(&mut list, draft as usize, self.place, self.pages));
            self.spill(&gone);
            return Ok(self.grant(list, true, w, worst, "exclusive", &gone, 0));
        }
        let n = pages_for(rows) as usize;
        self.earn(n as u64, 0);
        let mut list = Vec::with_capacity(n);
        let gone = self.fill(&mut list, n);
        self.spill(&gone);
        Ok(self.grant(list, false, w, rows, if gone.is_empty() { "free" } else { "lru" }, &gone, 0))
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
        if add > self.place.wrapping_sub(self.live) as u64 || !self.safe(None, Some((r.table_version, add))) {
            self.stalls += 1;
            return Ok(None);
        }
        let mut h = self.held.remove(&r.table_version).expect("held() found it");
        if let Some(s) = &mut self.stash {
            s.earn(add);
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

    pub fn release(&mut self, r: &Run) -> Result<(), Error> {
        self.remove(r).map(|_| ())
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
        let (exclusive, hits, span) = self.held(r).map(|h| (h.exclusive, h.hits, h.span))?;
        let list = self.remove(r)?;
        if !self.keep || st.valid <= 0 {
            return Ok(());
        }
        let rows_max = if exclusive { self.below_rows as u64 } else { u64::MAX };
        if let Some(s) = &mut self.stash {
            s.promote(&list, &st, rows_max, span);
        }
        self.age += 1;
        self.kept.push(Kept { list, st, age: self.age, exclusive, hits });
        Ok(())
    }

    /// The retained run with the longest reusable prefix for `prompt` (limit: rows the prefill may skip) that a
    /// reservation of `rows` now and `worst` at most can grow from. Ties go to the most recent.
    pub fn match_prefix(&self, prompt: &[i32], rows: u64, worst: u64, limit: i32) -> Hit {
        let mut best = Hit::default();
        if !self.keep || self.exclusive {
            return best;
        }
        let w = pages_for(worst);
        let n = if self.exclusive(worst) { w } else { pages_for(rows) };
        for k in &self.kept {
            let t = reuse::begin(&self.rules, &k.st, prompt, limit, true);
            if t.skip <= 0 {
                if best.skip == 0 && t.lcp >= best.lcp {
                    best.lcp = t.lcp;
                    best.why = t.why;
                }
                continue;
            }
            let a = pages_for(t.skip as u64);
            let adopted = &k.list[..(a as usize).min(k.list.len())];
            let mut room = a <= k.list.len() as u64 && a <= n;
            if self.exclusive(worst) {
                room = room && self.live == 0 && n <= self.pages as u64;
                if self.nr == 0 {
                    // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                    room = room && adopted.iter().enumerate().all(|(i, &p)| p as usize == i);
                } else {
                    // its own pages: the adopted ones and fresh ones of both classes must fit
                    room = room && self.hit_room(adopted, a, n, worst);
                }
            } else {
                room = room && n <= (self.place.wrapping_sub(self.live)) as u64 && n <= w;
                room = room && adopted.iter().all(|&p| p < self.place) && self.safe(Some((n, w)), None);
            }
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
        // the tiers in order. A stored hit must reuse more rows than the pool's best, since a pool hit is free of copies
        let Some(stash) = &self.stash else { return best };
        for t in 0..stash.tiers() {
            for e in stash.entries(t) {
                let b = reuse::begin(&self.rules, &e.st, prompt, limit, true);
                if b.skip <= 0 {
                    if best.skip == 0 && b.lcp >= best.lcp {
                        best.lcp = b.lcp;
                        best.why = b.why;
                    }
                    continue;
                }
                let a = pages_for(b.skip as u64);
                if a > self.restorable(e, worst) || !self.restore_room(a, n, w, worst, e.tail.is_some()) {
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
    fn restorable(&self, e: &crate::stash::Entry, worst: u64) -> u64 {
        (if self.exclusive(worst) { e.pages() } else { e.below() }) as u64
    }

    // whether a grant of n pages (worst w) can restore its first a pages. Below R it is as reserve() makes it. Past R it
    // needs an empty pool, and the restored pages must lie inside the CP map's replicated start unless the entry has
    // its tail
    fn restore_room(&self, a: u64, n: u64, w: u64, worst: u64, tail: bool) -> bool {
        if a > n {
            return false;
        }
        if self.exclusive(worst) {
            let room = self.live == 0
                && !self.exclusive
                && n <= self.pages as u64
                && (tail || a * PAGE <= self.below_rows as u64);
            if self.nr == 0 || !room {
                return room;
            }
            // its own pages: the two classes must fit (a fresh grant restores into them)
            let (main, draft) = self.exclusive_pages(worst);
            return main <= self.place as u64 && draft <= (self.pages - self.place) as u64;
        }
        n <= self.place.wrapping_sub(self.live) as u64 && n <= w && self.safe(Some((n, w)), None)
    }

    /// Claims a stored hit: a fresh grant as reserve() makes it, with the entry's first ceil(skip / 64) blocks queued
    /// for restore into its first pages.
    fn take_stored(&mut self, skip: i32, id: u64, tier: usize, rows: u64, worst: u64) -> Result<Run, Error> {
        let e = self.stash.as_ref().and_then(|s| s.find(tier, id)).ok_or(Error::NotRetained)?;
        let w = pages_for(worst);
        let (a, n) = (pages_for(skip as u64), if self.exclusive(worst) { w } else { pages_for(rows) });
        if a > self.restorable(e, worst) || !self.restore_room(a, n, w, worst, e.tail.is_some()) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let e = self.stash.as_mut().expect("found in it").begin_restore(tier, id);
        let hits = self.hits_from(&e.st, e.hits);
        self.earn(n, a);
        let (list, gone, exclusive, pool_base) = if self.exclusive(worst) {
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                ((0..n as u32).collect::<Vec<u32>>(), gone, true, false)
            } else {
                // its own pages, a fresh two-class grant; the entry's blocks (its tail included) restore into them.
                // Every run the classes evict spills, from either class.
                let (main, draft) = self.exclusive_pages(worst);
                let mut list = Vec::with_capacity(n as usize);
                let mut gone = self.fill_range(&mut list, main as usize, 0, self.place);
                gone.extend(self.fill_range(&mut list, draft as usize, self.place, self.pages));
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
                if k.st.valid > (b as u64 * PAGE) as i32 {
                    self.spill(std::slice::from_ref(&k));
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
        let rows = if exclusive { worst } else { rows };
        Ok(self.grant(list, exclusive, w as u32, rows, if pool_base { "stored+pool" } else { "stored" }, &gone, hits))
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
            k.st.valid >= rows
                && k.st.history.len() >= hist.len()
                && k.st.history[..hist.len()] == *hist
                && k.list.len() >= b
                && k.list[..b].iter().all(|&p| p < self.place)
        })
    }

    /// Claims a match_prefix() hit (skip, age) as a new grant: the retained run's first ceil(skip / 64) pages, then
    /// pages as reserve() takes them.
    pub fn take(&mut self, skip: i32, age: u64, rows: u64, worst: u64) -> Result<Run, Error> {
        self.take_hit(skip, age, None, rows, worst)
    }

    /// take() for a hit that may be stored (from = its tier).
    pub fn take_hit(&mut self, skip: i32, age: u64, from: Option<u32>, rows: u64, worst: u64) -> Result<Run, Error> {
        self.pending()?;
        if skip <= 0 || !self.admissible(worst) {
            return Err(Error::NoHit);
        }
        if let Some(t) = from {
            return self.take_stored(skip, age, t as usize, rows, worst);
        }
        let i = self.kept.iter().position(|k| k.age == age).ok_or(Error::NotRetained)?;
        let w = pages_for(worst);
        let (a, n) = (pages_for(skip as u64), if self.exclusive(worst) { w } else { pages_for(rows) });
        // checks the C++ pool left to match(): a hit from match() on this state always passes them
        let k = &self.kept[i];
        if a > k.list.len() as u64 || a > n {
            return Err(Error::NoHit);
        }
        let adopted = &k.list[..a as usize];
        if self.exclusive(worst) {
            if n > self.pages as u64 {
                return Err(Error::NoRoom);
            }
            if self.nr == 0 {
                // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                if adopted.iter().enumerate().any(|(i, &p)| p as usize != i) {
                    return Err(Error::NoRoom);
                }
            } else if !self.hit_room(adopted, a, n, worst) {
                return Err(Error::NoRoom);
            }
        } else if n > (self.place.wrapping_sub(self.live)) as u64
            || n > w
            || adopted.iter().any(|&p| p >= self.place)
            || !self.safe(Some((n, w)), None)
        {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let k = self.kept.remove(i);
        let hits = self.hits_from(&k.st, k.hits);
        self.earn(n, a);
        // the grant keeps the run's first a pages and writes past its skip: what the run held beyond that goes to a
        // tier, read before the grant writes
        if k.st.valid > skip {
            self.spill(std::slice::from_ref(&k));
        }
        let mut list = k.list;
        list.truncate(a as usize);
        if self.exclusive(worst) {
            if self.nr == 0 {
                let gone = std::mem::take(&mut self.kept);
                self.spill(&gone);
                list.extend(a as u32..n as u32);
                return Ok(self.grant(list, true, w as u32, worst, "hit", &gone, hits));
            }
            // its own pages: the adopted ones, then the two classes fill the rest (the low ones first, so the run's
            // pages below place stay a prefix of its low pages in list order)
            let gone = self.fill_hit(&mut list, a, n, worst);
            self.spill(&gone);
            return Ok(self.grant(list, true, w as u32, worst, "hit", &gone, hits));
        }
        let gone = self.fill(&mut list, n as usize);
        self.spill(&gone);
        Ok(self.grant(list, false, w as u32, rows, "hit", &gone, hits))
    }

    /// Every rank's pool must agree: folded into Scheduler::digest at each event boundary.
    pub fn digest(&self) -> u64 {
        let mut h = mix(mix(mix(FNV_OFFSET, self.version as u64), self.age), self.exclusive as u64);
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
        }
        for k in &self.kept {
            h = mix(mix(mix(h, k.age), hash(&k.list, false)), k.st.valid as i64 as u64);
        }
        // without a store the digest is what it was before stores existed
        match &self.stash {
            Some(s) => {
                for x in self.held.values() {
                    h = mix(h, x.hits as u64);
                }
                for k in &self.kept {
                    h = mix(h, k.hits as u64);
                }
                s.fold(h, mix)
            }
            None => h,
        }
    }

    /// One stderr line per claim (g6m4: G6_POOL_LOG=1).
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
        if self.exclusive {
            return new.is_none() && grow.is_none();
        }
        let mut free = self.place as i64 - self.live as i64;
        let mut runs: Vec<(u64, u64)> = self
            .held
            .iter()
            .map(|(v, h)| {
                let held = h.list.len() as u64 + grow.filter(|g| g.0 == *v).map_or(0, |g| g.1);
                ((h.worst as u64).saturating_sub(held), held)
            })
            .collect();
        free -= grow.map_or(0, |g| g.1 as i64);
        if let Some((n, w)) = new {
            free -= n as i64;
            runs.push((w.saturating_sub(n), n));
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
    fn mark(&mut self, list: &[u32], on: bool) {
        for &p in list {
            debug_assert!(self.used[p as usize] != on);
            self.used[p as usize] = on;
        }
        let n = list.len() as u32;
        self.live = if on { self.live + n } else { self.live - n };
    }

    // `list` (adopted pages, if any) grows by `need` pages out of [lo, hi): the lowest pages no run holds or retains,
    // after evicting the least recently retired runs until there are enough of them. Returns the evicted runs.
    fn fill_range(&mut self, list: &mut Vec<u32>, need: usize, lo: u32, hi: u32) -> Vec<Kept> {
        let (lo, hi) = (lo as usize, hi as usize);
        // claims[p]: 1 when a live run or the adopted list holds p, plus one per retained run that holds it. A page
        // is spare at 0, and evicting a run only ever takes back its own claims.
        let mut claims: Vec<u32> = self.used[lo..hi].iter().map(|&u| u as u32).collect();
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
    fn hit_room(&self, adopted: &[u32], a: u64, n: u64, worst: u64) -> bool {
        let (main, _draft) = self.exclusive_pages(worst);
        let alt = adopted.iter().filter(|&&p| p < self.place).count() as u64;
        let fresh_lt = main.saturating_sub(alt);
        fresh_lt <= n - a
            && alt + fresh_lt <= self.place as u64
            && n - alt - fresh_lt <= (self.pages - self.place) as u64
    }

    // take_hit's past-R grant (nr > 0): the list grows from a to n pages, the low class first so the run's pages
    // below place stay, in list order, its prefix, its shard and then the drafter's tail pages. Returns the evicted.
    fn fill_hit(&mut self, list: &mut Vec<u32>, a: u64, n: u64, worst: u64) -> Vec<Kept> {
        let (main, _draft) = self.exclusive_pages(worst);
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
        let span = self.next_span.take();
        // a run that continues its conversation counts as continued once, even when it found nothing to reuse
        let hits = hits.max(std::mem::take(&mut self.next_continues) as u32);
        self.held.insert(r.table_version, Held { tokens, hits, span, list, exclusive, worst });
        self.exclusive = self.exclusive || exclusive;
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
