//! The host ledger of a box that runs a Flash rank and a co-located KV node (D-331 M7): who may count each exchangeable
//! byte of the box's memory. One small file per box, read and written under flock, every change one atomic rename.
//!
//! `total` bytes are exchangeable. Each is counted in exactly one place: `gpu` (backing the rank's KV pool above what
//! FLASH_CAP_POLICY protects), `node` (the KV node may lock it), `free` (neither may count it), or the pending
//! transfer's `inflight` (released by its source, not yet credited to its destination). So
//! gpu + node + free + inflight == total, always.
//!
//! A transfer is a transaction with one generation:
//!   begin(from, to, n)  -> gen     nothing moves; `from` still counts the bytes. One transfer at a time.
//!   released(gen, m)               the source has physically let go of m <= n bytes: from -= m, inflight = m
//!   commit(gen, k)                 the destination has backed k <= m bytes: to += k, the rest goes to free
//!   abort(gen)                     before released: nothing moved; after: the inflight bytes go to free
//! Order on each side: the source releases its memory BEFORE it records released; the destination backs memory only
//! AFTER released is recorded, and records commit after. A finished transfer (commit or abort) moves `gen` on, so a
//! late call from a stalled party (its gen is no longer the pending one) is refused: a stalled borrower can never be
//! credited after its transfer was aborted, and it must give back what it backed when its commit is refused.
//! recover(stale) aborts a transfer pending longer than `stale`: a party that crashed frees its memory with its process,
//! so its bytes are counted to free, never to both sides.
//! `floor`: gpu never goes below it through a transfer (FLASH_CAP_POLICY=protected keeps the configured CAP's backing
//! out of the ledger entirely, so its floor is 0 over the chunks above CAP; the field is for a partial protection).

use std::ffi::c_int;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

extern "C" {
    fn flock(fd: c_int, op: c_int) -> c_int;
}
const LOCK_EX: c_int = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Gpu,
    Node,
    Free,
}

impl Side {
    pub fn parse(s: &str) -> Option<Side> {
        match s {
            "gpu" => Some(Side::Gpu),
            "node" => Some(Side::Node),
            "free" => Some(Side::Free),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Side::Gpu => "gpu",
            Side::Node => "node",
            Side::Free => "free",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Begun,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transfer {
    pub gen: u64,
    pub from: Side,
    pub to: Side,
    /// asked at begin; after released, the bytes in flight
    pub bytes: u64,
    pub phase: Phase,
    /// unix seconds at begin
    pub since: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Book {
    /// the last finished transfer's generation
    pub gen: u64,
    pub total: u64,
    pub gpu: u64,
    pub node: u64,
    pub free: u64,
    pub floor: u64,
    pub pending: Option<Transfer>,
}

impl Book {
    pub fn new(total: u64, gpu: u64, node: u64, floor: u64) -> Result<Book, String> {
        let used = gpu.checked_add(node).filter(|&u| u <= total).ok_or("gpu + node is more than total")?;
        if floor > gpu {
            return Err("floor is more than gpu".into());
        }
        Ok(Book { gen: 0, total, gpu, node, free: total - used, floor, pending: None })
    }
    pub fn inflight(&self) -> u64 {
        match self.pending {
            Some(t) if t.phase == Phase::Released => t.bytes,
            _ => 0,
        }
    }
    /// gpu + node + free + inflight == total
    pub fn balanced(&self) -> bool {
        [self.gpu, self.node, self.free, self.inflight()].iter().try_fold(0u64, |a, &b| a.checked_add(b))
            == Some(self.total)
            && self.gpu >= self.floor
    }
    fn side(&mut self, s: Side) -> &mut u64 {
        match s {
            Side::Gpu => &mut self.gpu,
            Side::Node => &mut self.node,
            Side::Free => &mut self.free,
        }
    }
    fn pending_is(&self, gen: u64, phase: Phase) -> Result<Transfer, String> {
        match self.pending {
            Some(t) if t.gen == gen && t.phase == phase => Ok(t),
            Some(t) if t.gen == gen => Err(format!("transfer {gen} is {:?}, not {phase:?}", t.phase)),
            Some(t) => Err(format!("transfer {gen} is not pending (transfer {} is)", t.gen)),
            None => Err(format!("transfer {gen} is not pending (it finished or was aborted; gen is {})", self.gen)),
        }
    }
    pub fn begin(&mut self, from: Side, to: Side, bytes: u64, now: u64) -> Result<u64, String> {
        if let Some(t) = self.pending {
            return Err(format!(
                "busy: transfer {} ({} -> {}, {} bytes) pending since {}",
                t.gen,
                t.from.name(),
                t.to.name(),
                t.bytes,
                t.since
            ));
        }
        if from == to || bytes == 0 {
            return Err("a transfer moves bytes between two sides".into());
        }
        let have = *self.side(from);
        if have < bytes {
            return Err(format!("{} counts {have} bytes, fewer than {bytes}", from.name()));
        }
        if from == Side::Gpu && have - bytes < self.floor {
            return Err(format!("gpu would go below its floor of {} bytes", self.floor));
        }
        let gen = self.gen + 1;
        self.pending = Some(Transfer { gen, from, to, bytes, phase: Phase::Begun, since: now });
        Ok(gen)
    }
    pub fn released(&mut self, gen: u64, bytes: u64) -> Result<(), String> {
        let mut t = self.pending_is(gen, Phase::Begun)?;
        if bytes > t.bytes {
            return Err(format!("released {bytes} bytes, more than the {} begun", t.bytes));
        }
        *self.side(t.from) -= bytes;
        t.bytes = bytes;
        t.phase = Phase::Released;
        self.pending = Some(t);
        Ok(())
    }
    pub fn commit(&mut self, gen: u64, bytes: u64) -> Result<(), String> {
        let t = self.pending_is(gen, Phase::Released)?;
        if bytes > t.bytes {
            return Err(format!("committed {bytes} bytes, more than the {} in flight", t.bytes));
        }
        *self.side(t.to) += bytes;
        self.free += t.bytes - bytes;
        self.gen = gen;
        self.pending = None;
        Ok(())
    }
    pub fn abort(&mut self, gen: u64) -> Result<(), String> {
        let t = self.pending.filter(|t| t.gen == gen).ok_or_else(|| format!("transfer {gen} is not pending"))?;
        if t.phase == Phase::Released {
            self.free += t.bytes;
        }
        self.gen = gen;
        self.pending = None;
        Ok(())
    }
    /// Aborts the pending transfer when it began more than `stale_s` seconds before `now`. Returns its generation.
    pub fn recover(&mut self, stale_s: u64, now: u64) -> Option<u64> {
        let t = self.pending?;
        if now.saturating_sub(t.since) < stale_s {
            return None;
        }
        self.abort(t.gen).ok().map(|_| t.gen)
    }

    pub fn to_text(&self) -> String {
        let mut s = format!(
            "gen {}\ntotal {}\ngpu {}\nnode {}\nfree {}\nfloor {}\n",
            self.gen, self.total, self.gpu, self.node, self.free, self.floor
        );
        if let Some(t) = self.pending {
            let ph = if t.phase == Phase::Begun { "begun" } else { "released" };
            s += &format!("pending {} {} {} {} {ph} {}\n", t.gen, t.from.name(), t.to.name(), t.bytes, t.since);
        }
        s + "end\n"
    }
    pub fn from_text(s: &str) -> Result<Book, String> {
        if !s.ends_with("end\n") {
            return Err("torn ledger (no end line)".into());
        }
        let mut b = Book { gen: 0, total: 0, gpu: 0, node: 0, free: 0, floor: 0, pending: None };
        let num = |v: Option<&str>| v.and_then(|v| v.parse::<u64>().ok()).ok_or("bad number");
        for line in s.lines() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("gen") => b.gen = num(w.next())?,
                Some("total") => b.total = num(w.next())?,
                Some("gpu") => b.gpu = num(w.next())?,
                Some("node") => b.node = num(w.next())?,
                Some("free") => b.free = num(w.next())?,
                Some("floor") => b.floor = num(w.next())?,
                Some("pending") => {
                    let gen = num(w.next())?;
                    let side = |v: Option<&str>| v.and_then(Side::parse).ok_or("bad side");
                    let (from, to, bytes) = (side(w.next())?, side(w.next())?, num(w.next())?);
                    let phase = match w.next() {
                        Some("begun") => Phase::Begun,
                        Some("released") => Phase::Released,
                        _ => return Err("bad phase".into()),
                    };
                    b.pending = Some(Transfer { gen, from, to, bytes, phase, since: num(w.next())? });
                }
                Some("end") | None => {}
                Some(k) => return Err(format!("unknown ledger line {k}")),
            }
        }
        if !b.balanced() {
            return Err(format!("the ledger does not balance: {}", b.to_text().replace('\n', "; ")));
        }
        Ok(b)
    }
    pub fn to_json(&self) -> String {
        let p = match self.pending {
            Some(t) => format!(
                "{{\"gen\":{},\"from\":\"{}\",\"to\":\"{}\",\"bytes\":{},\"phase\":\"{}\",\"since\":{}}}",
                t.gen,
                t.from.name(),
                t.to.name(),
                t.bytes,
                if t.phase == Phase::Begun { "begun" } else { "released" },
                t.since
            ),
            None => "null".into(),
        };
        format!(
            "{{\"gen\":{},\"total\":{},\"gpu\":{},\"node\":{},\"free\":{},\"inflight\":{},\"floor\":{},\"pending\":{p}}}",
            self.gen,
            self.total,
            self.gpu,
            self.node,
            self.free,
            self.inflight(),
            self.floor
        )
    }
}

pub fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The ledger file. Every call takes the lock file (PATH.lock), reads PATH, applies one change and renames a new PATH
/// into place (fsynced), so a crash leaves the old book or the new one, never a torn one.
pub struct Ledger {
    path: PathBuf,
}

/// Tests only: G6_LEDGER_CRASH_AT=write (before the rename) or written (after it) aborts the process there.
fn crash_at(point: &str) {
    if std::env::var("G6_LEDGER_CRASH_AT").is_ok_and(|v| v == point) {
        std::process::abort();
    }
}

impl Ledger {
    pub fn at(path: impl AsRef<Path>) -> Ledger {
        Ledger { path: path.as_ref().to_path_buf() }
    }
    fn lock(&self) -> io::Result<File> {
        let mut p = self.path.clone().into_os_string();
        p.push(".lock");
        let f = OpenOptions::new().create(true).truncate(false).write(true).open(p)?;
        if unsafe { flock(f.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(f)
    }
    fn read(&self) -> io::Result<Book> {
        let s = std::fs::read_to_string(&self.path)?;
        Book::from_text(&s)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", self.path.display())))
    }
    fn write(&self, b: &Book) -> io::Result<()> {
        let mut tmp = self.path.clone().into_os_string();
        tmp.push(".tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(b.to_text().as_bytes())?;
            f.sync_all()?;
        }
        crash_at("write");
        std::fs::rename(&tmp, &self.path)?;
        if let Some(d) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            File::open(d)?.sync_all()?;
        }
        crash_at("written");
        Ok(())
    }
    /// Creates the book. Refuses when one exists with a transfer pending (recover or abort it first).
    pub fn init(&self, b: &Book) -> io::Result<()> {
        let _l = self.lock()?;
        if let Ok(old) = self.read() {
            if let Some(t) = old.pending {
                return Err(io::Error::other(format!("transfer {} is pending: abort or recover it first", t.gen)));
            }
        }
        let mut b = b.clone();
        b.gen = self.read().map_or(0, |o| o.gen);
        self.write(&b)
    }
    pub fn show(&self) -> io::Result<Book> {
        let _l = self.lock()?;
        self.read()
    }
    /// Applies f to the book under the lock, and writes the book when f succeeds.
    pub fn apply<T>(&self, f: impl FnOnce(&mut Book) -> Result<T, String>) -> io::Result<T> {
        let _l = self.lock()?;
        let mut b = self.read()?;
        let r = f(&mut b).map_err(io::Error::other)?;
        self.write(&b)?;
        Ok(r)
    }
}
