//! Remote attention rows (the engine's G6_RT_STORE tier): one rank's latent rows past L live in a store, and a helper
//! thread fetches the rows each decode step selects into the engine's resident slots.
//!
//! The store is one byte range per rank: a header (HEADER bytes), then per indexer group one record per position past
//! L that the rank holds, each record the group's layers' latent rows side by side (engine/runtime/rt_rows.cuh has the
//! layout). The engine writes records itself (write) between steps.
//!
//! The header names the layout the records were written under (the engine's layout string: R, ranks, row format,
//! model set). Records are only read under that layout: open writes the header into an empty store and refuses a store
//! whose header names another layout.
//!
//! Each group has a request ring the GPU writes and this thread reads, both in mapped pinned memory (g6kv_rows_req):
//! the GPU fills n entries {record offset, slot}, then stores seq with a system-scope release. The thread sees the new
//! seq, reads every record into slots + slot * rec_bytes (the group's resident slots, also mapped pinned memory), then
//! stores done = seq, which the GPU's wait kernel spins on. On a failed read it stores done = u64::MAX and the wait
//! reports an error. The thread makes no CUDA calls.
//!
//! Stores: "mem:" (this process's memory, for m4sim on one box) and "remote:HOST:PORT" (a lease on a KV node, RDMA
//! where both ends can, one chain of one-sided reads per request, straight into the registered slots).

use crate::remote::{Opts, RemoteStore};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// One entry of a request ring (g6kv_rows_ent).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ent {
    pub off: u64,
    pub slot: u32,
    pub pad: u32,
}

/// The ring's header (g6kv_rows_req); `cap` entries follow it.
#[repr(C)]
pub struct Req {
    pub seq: AtomicU64,
    pub n: u32,
    pub pad: u32,
}

/// Bytes at the start of a store that hold its layout header.
pub const HEADER: usize = 4096;
const MAGIC: &[u8; 8] = b"G6RT0001";

pub enum Store {
    Mem(Mutex<Vec<u8>>),
    Remote(Box<RemoteStore>),
}

impl Store {
    pub fn open(spec: &str, name: &str, bytes: u64) -> io::Result<Store> {
        let bad =
            || io::Error::new(io::ErrorKind::InvalidInput, format!("rows store {spec}: mem: or remote:HOST:PORT"));
        let (kind, rest) = spec.split_once(':').ok_or_else(bad)?;
        match kind {
            "mem" => Ok(Store::Mem(Mutex::new(vec![0u8; bytes as usize]))),
            "remote" if !rest.is_empty() => {
                let addr = if rest.contains(':') {
                    rest.to_string()
                } else {
                    format!("{rest}:{}", crate::remote::proto::DEFAULT_PORT)
                };
                // the lease in 1 MiB blocks: the block size only sizes each lane's staging buffer here
                const BLOCK: usize = 1 << 20;
                let o =
                    Opts { lanes: 2, qp_timeout: 17, ..Opts::new(name, BLOCK, bytes.div_ceil(BLOCK as u64).max(1)) };
                Ok(Store::Remote(Box::new(RemoteStore::open(&addr, &o)?)))
            }
            _ => Err(bad()),
        }
    }

    /// Writes the layout header into an empty store, or checks the one there names `layout`.
    pub fn layout(&self, layout: &str) -> io::Result<()> {
        if layout.len() > HEADER - MAGIC.len() || layout.contains('\0') {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "rows layout string too long"));
        }
        let mut want = vec![0u8; HEADER];
        want[..MAGIC.len()].copy_from_slice(MAGIC);
        want[MAGIC.len()..MAGIC.len() + layout.len()].copy_from_slice(layout.as_bytes());
        let mut have = vec![0u8; HEADER];
        unsafe { self.read_scatter(&[(0, have.as_mut_ptr(), HEADER)])? };
        if have.iter().all(|&b| b == 0) {
            return self.write(0, &want);
        }
        if have != want {
            let end = have[MAGIC.len()..].iter().position(|&b| b == 0).map_or(HEADER, |e| e + MAGIC.len());
            let theirs = String::from_utf8_lossy(&have[MAGIC.len().min(end)..end]).into_owned();
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("rows store holds records of another layout ({theirs}); this engine reads {layout}"),
            ));
        }
        Ok(())
    }

    pub fn write(&self, off: u64, buf: &[u8]) -> io::Result<()> {
        match self {
            Store::Mem(m) => {
                let mut v = m.lock().unwrap();
                let end = off as usize + buf.len();
                if end > v.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "rows write past the store"));
                }
                v[off as usize..end].copy_from_slice(buf);
                Ok(())
            }
            Store::Remote(r) => r.write_at(off, buf),
        }
    }

    /// # Safety
    /// As RemoteStore::read_scatter.
    pub unsafe fn read_scatter(&self, ops: &[(u64, *mut u8, usize)]) -> io::Result<()> {
        match self {
            Store::Mem(m) => {
                let v = m.lock().unwrap();
                for &(o, p, n) in ops {
                    if o as usize + n > v.len() {
                        return Err(io::Error::new(io::ErrorKind::InvalidInput, "rows read past the store"));
                    }
                    std::ptr::copy_nonoverlapping(v.as_ptr().add(o as usize), p, n);
                }
                Ok(())
            }
            Store::Remote(r) => r.read_scatter(ops),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Store::Mem(m) => format!("memory, {} B", m.lock().unwrap().len()),
            Store::Remote(r) => format!(
                "KV node {} from {}, lease {} of {} B, {}",
                r.server,
                r.local.ip(),
                r.lease_id,
                r.lease_bytes,
                match r.transport {
                    crate::remote::Transport::Rdma => "RDMA",
                    crate::remote::Transport::Tcp => "TCP",
                }
            ),
        }
    }
}

/// A group's ring, done word and slots, as addresses the helper thread owns while it runs.
#[derive(Clone, Copy)]
struct Group {
    req: usize,
    done: usize,
    slots: usize,
    rec: usize,
    cap: u32,
    nslots: u32,
}

// an entry names a slot past the group's slots
fn ents_bad(g: &Group, n: u32) -> bool {
    if n > g.cap {
        return true;
    }
    let ents = unsafe { std::slice::from_raw_parts((g.req + std::mem::size_of::<Req>()) as *const Ent, n as usize) };
    ents.iter().any(|e| unsafe { std::ptr::read_volatile(e) }.slot >= g.nslots)
}

pub struct Rows {
    pub store: Arc<Store>,
    groups: Vec<Group>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub fetched: Arc<AtomicU64>,
}

impl Rows {
    pub fn new(store: Store) -> Rows {
        Rows {
            store: Arc::new(store),
            groups: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            fetched: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Adds a group before start. `req` holds a Req and `cap` entries, `done` a u64, `slots` `nslots` slots of `rec`
    /// bytes.
    pub fn add_group(
        &mut self,
        req: *mut Req,
        done: *mut u64,
        slots: *mut u8,
        rec: usize,
        nslots: u32,
        cap: u32,
    ) -> io::Result<usize> {
        if self.thread.is_some() || req.is_null() || done.is_null() || slots.is_null() || rec == 0 || nslots == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "rows group after start or with a null buffer"));
        }
        self.groups.push(Group { req: req as usize, done: done as usize, slots: slots as usize, rec, cap, nslots });
        Ok(self.groups.len() - 1)
    }

    /// Registers memory the store's RDMA device may write (the slots), a no-op without RDMA.
    pub fn register(&self, p: *mut u8, len: usize) -> io::Result<()> {
        match &*self.store {
            Store::Remote(r) => r.register(p, len),
            Store::Mem(_) => Ok(()),
        }
    }

    pub fn start(&mut self) {
        if self.thread.is_some() {
            return;
        }
        let (groups, store, stop, fetched) =
            (self.groups.clone(), self.store.clone(), self.stop.clone(), self.fetched.clone());
        // read here, not on the thread: a request rung before the thread runs must still be served
        let seen: Vec<u64> =
            groups.iter().map(|g| unsafe { (*(g.req as *const Req)).seq.load(Ordering::Acquire) }).collect();
        self.thread = Some(std::thread::spawn(move || serve(&groups, seen, &store, &stop, &fetched)));
    }
}

impl Drop for Rows {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// How long the helper spins after its last request: longer than the gaps between one decode step's groups.
const SPIN: Duration = Duration::from_millis(5);
/// After this long without a request it sleeps 1 ms at a time instead of 50 us.
const NAP_AFTER: Duration = Duration::from_secs(1);

fn serve(groups: &[Group], mut seen: Vec<u64>, store: &Store, stop: &AtomicBool, fetched: &AtomicU64) {
    let mut ops: Vec<(u64, *mut u8, usize)> = Vec::new();
    let mut last = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let mut any = false;
        for (g, s) in groups.iter().zip(seen.iter_mut()) {
            let req = unsafe { &*(g.req as *const Req) };
            let seq = req.seq.load(Ordering::Acquire);
            if seq == *s {
                continue;
            }
            any = true;
            *s = seq;
            let n = unsafe { std::ptr::read_volatile(&req.n) };
            let done = unsafe { &*(g.done as *const AtomicU64) };
            if n > g.cap || ents_bad(g, n) {
                eprintln!(
                    "g6kv rows: a request of {n} entries (the ring holds {}) or one past the {} slots",
                    g.cap, g.nslots
                );
                done.store(u64::MAX, Ordering::Release);
                continue;
            }
            let ents =
                unsafe { std::slice::from_raw_parts((g.req + std::mem::size_of::<Req>()) as *const Ent, n as usize) };
            ops.clear();
            ops.extend(ents.iter().map(|e| {
                let e = unsafe { std::ptr::read_volatile(e) };
                (e.off, (g.slots + e.slot as usize * g.rec) as *mut u8, g.rec)
            }));
            match unsafe { store.read_scatter(&ops) } {
                Ok(()) => {
                    fetched.fetch_add(n as u64, Ordering::Relaxed);
                    done.store(seq, Ordering::Release);
                }
                Err(e) => {
                    eprintln!("g6kv rows: read failed: {e}");
                    done.store(u64::MAX, Ordering::Release);
                }
            }
        }
        // spin through the gaps between one step's groups, then sleep: an idle engine costs no core
        if any {
            last = Instant::now();
        } else {
            let quiet = last.elapsed();
            if quiet < SPIN {
                std::hint::spin_loop();
            } else if quiet < NAP_AFTER {
                std::thread::sleep(Duration::from_micros(50));
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct Ring {
        h: Req,
        e: [Ent; 4],
    }

    #[test]
    fn refuses_another_layout() {
        let s = Store::open("mem:", "t", 2 * HEADER as u64).unwrap();
        s.layout("R=1024 nr=4 row=fp8 model=a").unwrap();
        s.layout("R=1024 nr=4 row=fp8 model=a").unwrap();
        let e = s.layout("R=2048 nr=4 row=fp8 model=a").unwrap_err();
        assert!(e.to_string().contains("R=1024 nr=4 row=fp8 model=a"), "{e}");
        assert!(s.layout("R=1024 nr=3 row=fp8 model=a").is_err());
        assert!(s.layout("R=1024 nr=4 row=int4g32h model=a").is_err());
    }

    #[test]
    fn fetches_records_into_slots() {
        let rec = 40;
        let mut rows = Rows::new(Store::open("mem:", "t", 10 * rec as u64).unwrap());
        for p in 0..10u8 {
            rows.store.write(p as u64 * rec as u64, &vec![p + 1; rec]).unwrap();
        }
        let mut ring =
            Box::new(Ring { h: Req { seq: AtomicU64::new(0), n: 0, pad: 0 }, e: [Ent { off: 0, slot: 0, pad: 0 }; 4] });
        let done = Box::new(AtomicU64::new(0));
        let mut slots = vec![0u8; 3 * rec];
        rows.add_group(&mut ring.h, done.as_ptr(), slots.as_mut_ptr(), rec, 3, 4).unwrap();
        rows.start();
        ring.e[0] = Ent { off: 7 * rec as u64, slot: 2, pad: 0 };
        ring.e[1] = Ent { off: 3 * rec as u64, slot: 0, pad: 0 };
        ring.h.n = 2;
        ring.h.seq.store(1, Ordering::Release);
        while done.load(Ordering::Acquire) != 1 {
            std::hint::spin_loop();
        }
        assert!(slots[..rec].iter().all(|&x| x == 4));
        assert!(slots[rec..2 * rec].iter().all(|&x| x == 0));
        assert!(slots[2 * rec..].iter().all(|&x| x == 8));
        ring.e[0] = Ent { off: 100 * rec as u64, slot: 1, pad: 0 };
        ring.h.n = 1;
        ring.h.seq.store(2, Ordering::Release);
        while done.load(Ordering::Acquire) == 1 {
            std::hint::spin_loop();
        }
        assert_eq!(done.load(Ordering::Acquire), u64::MAX);
        drop(rows);
    }
}
