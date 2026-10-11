//! The S2 page allocator below R (the pool every rank runs). See lib.rs for the rules it follows.

use crate::reuse::{self, Finished, Rules, Why};
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hit {
    pub skip: i32,
    pub lcp: i32,
    pub why: Why,
    pub age: u64,
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
    list: Vec<u32>,
    exclusive: bool,
}

/// A retired run: its pages count as free until a reservation evicts it or a hit adopts its start.
pub struct Kept {
    pub list: Vec<u32>,
    pub st: Finished,
    pub age: u64,
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
        })
    }

    pub fn exclusive(&self, rows: u64) -> bool {
        rows > self.below_rows as u64
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

    /// Head, before reserve: a past-R request waits for an empty pool.
    pub fn admissible(&self, rows: u64) -> bool {
        !self.exclusive && (!self.exclusive(rows) || self.live == 0)
    }

    pub fn reserve(&mut self, rows: u64) -> Result<Run, Error> {
        let n = pages_for(rows);
        if rows == 0 || n > self.room() as u64 || !self.admissible(rows) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        if self.exclusive(rows) {
            let gone = std::mem::take(&mut self.kept);
            return Ok(self.grant((0..n as u32).collect(), true, rows, "exclusive", &gone));
        }
        let mut list = Vec::with_capacity(n as usize);
        let gone = self.fill(&mut list, n as usize);
        Ok(self.grant(list, false, rows, if gone.is_empty() { "free" } else { "lru" }, &gone))
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
        let list = self.remove(r)?;
        if !self.keep || st.valid <= 0 {
            return Ok(());
        }
        self.age += 1;
        self.kept.push(Kept { list, st, age: self.age });
        Ok(())
    }

    /// The retained run with the longest reusable prefix for `prompt` (limit: rows the prefill may skip) that a
    /// reservation of `rows` can grow from. Ties go to the most recent.
    pub fn match_prefix(&self, prompt: &[i32], rows: u64, limit: i32) -> Hit {
        let mut best = Hit::default();
        if !self.keep || self.exclusive {
            return best;
        }
        let n = pages_for(rows);
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
            if self.exclusive(rows) {
                // an exclusive run is pages 0, 1, ..: the adopted ones must be its start
                room = room && self.live == 0 && n <= self.pages as u64;
                room = room && adopted.iter().enumerate().all(|(i, &p)| p as usize == i);
            } else {
                room = room && n <= (self.place.wrapping_sub(self.live)) as u64;
                room = room && adopted.iter().all(|&p| p < self.place);
            }
            if !room {
                if best.skip == 0 {
                    best.why = Why::Room;
                }
                continue;
            }
            if t.skip > best.skip || (t.skip == best.skip && k.age > best.age) {
                best = Hit { skip: t.skip, lcp: t.lcp, why: Why::Hit, age: k.age };
            }
        }
        best
    }

    /// Claims a match_prefix() hit (skip, age) as a new grant: the retained run's first ceil(skip / 64) pages, then
    /// pages as reserve() takes them.
    pub fn take(&mut self, skip: i32, age: u64, rows: u64) -> Result<Run, Error> {
        if skip <= 0 || !self.admissible(rows) {
            return Err(Error::NoHit);
        }
        let i = self.kept.iter().position(|k| k.age == age).ok_or(Error::NotRetained)?;
        let (a, n) = (pages_for(skip as u64), pages_for(rows));
        // checks the C++ pool left to match(): a hit from match() on this state always passes them
        let k = &self.kept[i];
        if a > k.list.len() as u64 || a > n {
            return Err(Error::NoHit);
        }
        let adopted = &k.list[..a as usize];
        if self.exclusive(rows) {
            if n > self.pages as u64 || adopted.iter().enumerate().any(|(i, &p)| p as usize != i) {
                return Err(Error::NoRoom);
            }
        } else if n > (self.place.wrapping_sub(self.live)) as u64 || adopted.iter().any(|&p| p >= self.place) {
            return Err(Error::NoRoom);
        }
        self.check_version()?;
        let mut list = self.kept.remove(i).list;
        list.truncate(a as usize);
        if self.exclusive(rows) {
            let gone = std::mem::take(&mut self.kept);
            list.extend(a as u32..n as u32);
            return Ok(self.grant(list, true, rows, "hit", &gone));
        }
        let gone = self.fill(&mut list, n as usize);
        Ok(self.grant(list, false, rows, "hit", &gone))
    }

    /// Every rank's pool must agree: folded into Scheduler::digest at each event boundary.
    pub fn digest(&self) -> u64 {
        let mut h = mix(mix(mix(FNV_OFFSET, self.version as u64), self.age), self.exclusive as u64);
        for (p, _) in self.used.iter().enumerate().filter(|(_, &u)| u) {
            h = mix(h, p as u64);
        }
        for (v, x) in &self.held {
            h = mix(mix(h, *v as u64), hash(&x.list, x.exclusive));
        }
        for k in &self.kept {
            h = mix(mix(mix(h, k.age), hash(&k.list, false)), k.st.valid as i64 as u64);
        }
        h
    }

    /// One stderr line per claim (g6m4: G6_POOL_LOG=1).
    pub fn log(&mut self, on: bool) {
        self.log = on;
    }

    pub fn kept_runs(&self) -> u32 {
        self.kept.len() as u32
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

    // `list` (adopted pages, if any) grows to n pages below place: the lowest pages no run holds or retains, after
    // evicting the least recently retired runs until there are enough of them. Returns the evicted runs.
    fn fill(&mut self, list: &mut Vec<u32>, n: usize) -> Vec<Kept> {
        let place = self.place as usize;
        // claims[p]: 1 when a live run or the adopted list holds p, plus one per retained run that holds it. A page
        // is spare at 0, and evicting a run only ever takes back its own claims.
        let mut claims: Vec<u32> = self.used[..place].iter().map(|&u| u as u32).collect();
        for &p in list.iter() {
            claims[p as usize] = 1;
        }
        for k in &self.kept {
            for &p in k.list.iter().filter(|&&p| (p as usize) < place) {
                claims[p as usize] += 1;
            }
        }
        let mut spare = claims.iter().filter(|&&c| c == 0).count();
        let mut evict = 0;
        while spare < n - list.len() {
            // the room checks before fill() guarantee enough pages once every retained run is gone
            let k = &self.kept[evict];
            for &p in k.list.iter().filter(|&&p| (p as usize) < place) {
                claims[p as usize] -= 1;
                if claims[p as usize] == 0 {
                    spare += 1;
                }
            }
            evict += 1;
        }
        let gone: Vec<Kept> = self.kept.drain(..evict).collect();
        let more = n - list.len();
        list.extend((0..place as u32).filter(|&p| claims[p as usize] == 0).take(more));
        gone
    }

    fn grant(&mut self, list: Vec<u32>, exclusive: bool, rows: u64, how: &str, gone: &[Kept]) -> Run {
        self.mark(&list, true);
        self.version += 1;
        let r = Run {
            first_page: list[0],
            pages: list.len() as u32,
            table_version: self.version,
            list_hash: hash(&list, exclusive),
        };
        if self.log {
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
        self.held.insert(r.table_version, Held { list, exclusive });
        self.exclusive = self.exclusive || exclusive;
        r
    }
}
