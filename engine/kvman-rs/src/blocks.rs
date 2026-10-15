//! The block stores behind one C handle (g6kvman.h g6kv_tier_*), for the engine to run the pool's copies through:
//!   ssd:PATH          a file of slots on this box (tier::SlotFile, O_DIRECT where the filesystem allows it)
//!   remote:HOST:PORT  a lease on a KV node (remote::RemoteStore, RDMA where both ends can, else TCP)
//!   mem:              this process's memory (tests)
//! A block's bytes are rounded up to 4 KiB, the slot an SSD needs, so every store takes the same buffers.

use crate::remote::{Opts, RemoteStore};
use crate::tier::{par_for, BlockStore, SlotFile, ALIGN};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::io;
use std::sync::Mutex;

pub enum Blocks {
    Ssd(SlotFile),
    Remote(Box<RemoteStore>),
    Mem(Mutex<Vec<u8>>, usize, u64),
}

impl Blocks {
    /// `spec` as above, `name` the lease name on a KV node, `block_bytes` before rounding.
    pub fn open(spec: &str, name: &str, block_bytes: usize, blocks: u64) -> io::Result<Blocks> {
        let slot = block_bytes.div_ceil(ALIGN) * ALIGN;
        let bad =
            || io::Error::new(io::ErrorKind::InvalidInput, format!("store {spec}: ssd:PATH, remote:HOST:PORT or mem:"));
        let (kind, rest) = spec.split_once(':').ok_or_else(bad)?;
        match kind {
            "ssd" if !rest.is_empty() => {
                let path = std::path::Path::new(rest);
                // tmpfs and some others refuse O_DIRECT: the page cache is then the price of running there
                let f = SlotFile::create(path, slot, blocks, true).or_else(|e| match e.raw_os_error() {
                    Some(22) => SlotFile::create(path, slot, blocks, false),
                    _ => Err(e),
                })?;
                Ok(Blocks::Ssd(f))
            }
            "remote" if !rest.is_empty() => {
                let addr = if rest.contains(':') {
                    rest.to_string()
                } else {
                    format!("{rest}:{}", crate::remote::proto::DEFAULT_PORT)
                };
                // four ranks share one switch port without PFC: a dropped packet costs one ack timeout, and 4.096 us *
                // 2^17 (0.5 s) leaves room for the retries before a lane fails
                let o = Opts { lanes: THREADS, qp_timeout: 17, ..Opts::new(name, slot, blocks) };
                Ok(Blocks::Remote(Box::new(RemoteStore::open(&addr, &o)?)))
            }
            "mem" => Ok(Blocks::Mem(Mutex::new(vec![0u8; slot * blocks as usize]), slot, blocks)),
            _ => Err(bad()),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Blocks::Ssd(f) => format!("ssd, {} slots of {} B, O_DIRECT {}", f.slots(), f.slot_bytes(), f.direct()),
            Blocks::Remote(r) => format!(
                "KV node {} from {}, lease {} of {} B, {}{}",
                r.server,
                r.local.ip(),
                r.lease_id,
                r.lease_bytes,
                match r.transport {
                    crate::remote::Transport::Rdma => "RDMA",
                    crate::remote::Transport::Tcp => "TCP",
                },
                r.fallback.as_ref().map_or(String::new(), |f| format!(" (no RDMA: {f})"))
            ),
            Blocks::Mem(_, s, n) => format!("memory, {n} blocks of {s} B"),
        }
    }
}

impl BlockStore for Blocks {
    fn block_bytes(&self) -> usize {
        match self {
            Blocks::Ssd(f) => f.slot_bytes(),
            Blocks::Remote(r) => r.block_bytes,
            Blocks::Mem(_, s, _) => *s,
        }
    }
    fn blocks(&self) -> u64 {
        match self {
            Blocks::Ssd(f) => f.slots(),
            Blocks::Remote(r) => r.blocks,
            Blocks::Mem(_, _, n) => *n,
        }
    }
    fn write_block(&self, block: u64, buf: &[u8]) -> io::Result<()> {
        match self {
            Blocks::Ssd(f) => f.write_block(block, buf),
            Blocks::Remote(r) => r.write_block(block, buf),
            Blocks::Mem(m, s, n) => {
                if block >= *n || buf.len() > *s {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "block out of range"));
                }
                let at = block as usize * s;
                m.lock().unwrap()[at..at + buf.len()].copy_from_slice(buf);
                Ok(())
            }
        }
    }
    fn read_block(&self, block: u64, buf: &mut [u8]) -> io::Result<()> {
        match self {
            Blocks::Ssd(f) => f.read_block(block, buf),
            Blocks::Remote(r) => r.read_block(block, buf),
            Blocks::Mem(m, s, n) => {
                if block >= *n || buf.len() > *s {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "block out of range"));
                }
                let at = block as usize * s;
                buf.copy_from_slice(&m.lock().unwrap()[at..at + buf.len()]);
                Ok(())
            }
        }
    }
}

/// Blocks a batch call moves at once: an RDMA lane or an O_DIRECT request each.
const THREADS: usize = 8;

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

pub(crate) fn fail(e: io::Error) -> i32 {
    LAST.with(|l| *l.borrow_mut() = CString::new(e.to_string().replace('\0', " ")).unwrap_or_default());
    -1
}

/// The message of this thread's last failed g6kv_tier call.
#[no_mangle]
pub extern "C" fn g6kv_tier_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

/// Opens a store of `blocks` blocks of `block_bytes` (rounded up to 4 KiB) into `*out`. 0, or -1 with
/// g6kv_tier_error.
///
/// # Safety
/// `spec` and `name` must be NUL-terminated strings, `out` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_open(
    spec: *const c_char,
    name: *const c_char,
    block_bytes: u64,
    blocks: u64,
    out: *mut *mut Blocks,
) -> i32 {
    if spec.is_null() || name.is_null() || out.is_null() {
        return fail(io::Error::new(io::ErrorKind::InvalidInput, "null argument"));
    }
    let (spec, name) = (CStr::from_ptr(spec).to_string_lossy(), CStr::from_ptr(name).to_string_lossy());
    match Blocks::open(&spec, &name, block_bytes as usize, blocks) {
        Ok(b) => {
            *out = Box::into_raw(Box::new(b));
            0
        }
        Err(e) => fail(e),
    }
}

/// # Safety
/// `t` must come from g6kv_tier_open (or be NULL) and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_free(t: *mut Blocks) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// A block's bytes as the store keeps them (rounded up to 4 KiB): every read and write moves this many.
///
/// # Safety
/// `t` must be a live store.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_block_bytes(t: *const Blocks) -> u64 {
    (*t).block_bytes() as u64
}

/// Writes the store's description into `buf`, NUL-terminated and cut to `cap`.
///
/// # Safety
/// `t` must be a live store and `buf` hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_describe(t: *const Blocks, buf: *mut c_char, cap: usize) {
    if buf.is_null() || cap == 0 {
        return;
    }
    let d = (*t).describe();
    let n = d.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(d.as_ptr() as *const c_char, buf, n);
    *buf.add(n) = 0;
}

/// Writes block `block` from `buf` (g6kv_tier_block_bytes bytes, 4 KiB-aligned for an SSD).
///
/// # Safety
/// `t` must be a live store, `buf` valid for g6kv_tier_block_bytes reads.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_write(t: *const Blocks, block: u64, buf: *const c_void) -> i32 {
    let n = (*t).block_bytes();
    match (*t).write_block(block, std::slice::from_raw_parts(buf as *const u8, n)) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Reads block `block` into `buf`.
///
/// # Safety
/// `t` must be a live store, `buf` valid for g6kv_tier_block_bytes writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_read(t: *const Blocks, block: u64, buf: *mut c_void) -> i32 {
    let n = (*t).block_bytes();
    match (*t).read_block(block, std::slice::from_raw_parts_mut(buf as *mut u8, n)) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Registers the caller's staging memory with a KV node's RDMA device, so blocks inside it skip a copy. A no-op for
/// the other stores.
///
/// # Safety
/// `t` must be a live store, and the memory must outlive it.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_register(t: *const Blocks, buf: *mut c_void, len: usize) -> i32 {
    match &*t {
        Blocks::Remote(r) => match r.register(buf as *mut u8, len) {
            Ok(()) => 0,
            Err(e) => fail(e),
        },
        _ => 0,
    }
}

// a batch call: block blocks[i] from or into bufs[i], for i < n
unsafe fn batch(
    t: *const Blocks,
    n: usize,
    blocks: *const u64,
    bufs: *const *mut c_void,
    write: bool,
) -> io::Result<()> {
    if n == 0 {
        return Ok(());
    }
    if blocks.is_null() || bufs.is_null() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "null argument"));
    }
    let (t, len) = (&*t, (*t).block_bytes());
    let blocks = std::slice::from_raw_parts(blocks, n);
    // the workers share the buffers as plain addresses
    let bufs: Vec<usize> = std::slice::from_raw_parts(bufs, n).iter().map(|&p| p as usize).collect();
    par_for(THREADS.min(n), n, |_, i| {
        if write {
            t.write_block(blocks[i], std::slice::from_raw_parts(bufs[i] as *const u8, len))
        } else {
            t.read_block(blocks[i], std::slice::from_raw_parts_mut(bufs[i] as *mut u8, len))
        }
    })
}

/// Writes `n` blocks at once, block `blocks[i]` from `bufs[i]`, on up to 8 threads.
///
/// # Safety
/// `t` must be a live store; `blocks` and `bufs` hold `n` entries, each buffer valid for g6kv_tier_block_bytes reads.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_write_many(
    t: *const Blocks,
    n: usize,
    blocks: *const u64,
    bufs: *const *mut c_void,
) -> i32 {
    match batch(t, n, blocks, bufs, true) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

/// Reads `n` blocks at once, block `blocks[i]` into `bufs[i]`, on up to 8 threads.
///
/// # Safety
/// As g6kv_tier_write_many, each buffer valid for g6kv_tier_block_bytes writes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_tier_read_many(
    t: *const Blocks,
    n: usize,
    blocks: *const u64,
    bufs: *const *mut c_void,
) -> i32 {
    match batch(t, n, blocks, bufs, false) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}
