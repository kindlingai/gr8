//! SSD tier, a proof of concept: an NVMe store for KV pages that the pool would otherwise drop. Nothing in the engine
//! calls it and it has no C entry points. kvman-ssd-bench measures it, kvman-sim replays agent traces through it.
//!
//! The store is one file of fixed-size slots, each a multiple of 4 KiB, read and written with O_DIRECT (Linux) from
//! 4 KiB-aligned buffers, so the page cache holds nothing. Two shapes of slot:
//!
//! - A page block: one logical KV page (64 rows) of every layer, the unit a retained run spills and restores in. Its
//!   size follows the engine's caches (engine/attn/bind.cu create(), attn.cu KV_BYTES / IDX_BYTES): per layer and page
//!   the MLA cache holds 64 rows x 656 B (512 FP8 latent + scales + 64 BF16 rope), and the full-index layers also hold
//!   the indexer's keys, 64 rows x 132 B (128 FP8 + scale). GLM 5.3 has 78 layers plus the MTP layer (79 MLA caches),
//!   and the full-index layers are 0, 1, 2, 6, 10, .., 74 (rank.cuh cp_full_index) and the MTP layer: 22 index caches.
//!   One page: 79 x 41,984 + 22 x 8,448 = 3,502,592 B, 856 x 4 KiB = 3,506,176 B on disk. Below R every rank holds a
//!   session's whole KV, so every rank would spill and restore its own copy.
//! - A row: one token's MLA rows of one indexer group, 4 layers x 656 B = 2,624 B in one 4 KiB slot. A full-index
//!   layer's top-2,048 list serves the three layers after it, so a sparse-attention step reads, per selected token and
//!   group, one row slot.
//!
//! Slots are allocated lowest free first, and spill / restore order is the run's page order, so two ranks given the
//! same calls use the same slots, as the pool does.

use crate::pool::Kept;
use crate::reuse::Finished;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const ALIGN: usize = 4096;
pub const PAGE_ROWS: usize = 64;
pub const MLA_ROW: usize = 656;
pub const IDX_ROW: usize = 132;
pub const MLA_LAYERS: usize = 79;
pub const INDEX_LAYERS: usize = 22;
pub const GROUP_LAYERS: usize = 4;

/// Bytes of one logical page across every layer's caches, before rounding to 4 KiB.
pub const PAGE_BLOCK_BYTES: usize = MLA_LAYERS * PAGE_ROWS * MLA_ROW + INDEX_LAYERS * PAGE_ROWS * IDX_ROW;
/// One page block's slot.
pub const PAGE_SLOT: usize = PAGE_BLOCK_BYTES.div_ceil(ALIGN) * ALIGN;
/// One token's rows of one indexer group, and its slot.
pub const ROW_BYTES: usize = GROUP_LAYERS * MLA_ROW;
pub const ROW_SLOT: usize = ALIGN;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_DIRECT: i32 = 0o200000;
#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
const O_DIRECT: i32 = 0o40000;

/// A zeroed buffer aligned to 4 KiB, as O_DIRECT needs.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// The buffer is owned memory like a Vec<u8>.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> AlignedBuf {
        let len = len.div_ceil(ALIGN).max(1) * ALIGN;
        let layout = std::alloc::Layout::from_size_align(len, ALIGN).expect("buffer layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBuf { ptr, len }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN).expect("buffer layout");
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

/// A file of `slots` slots of `slot_bytes` each. Reads and writes are positional, so threads share one handle.
pub struct SlotFile {
    file: File,
    slot_bytes: usize,
    slots: u64,
    direct: bool,
}

impl SlotFile {
    /// Creates (or reuses) the file at its full size. `direct` asks for O_DIRECT, which exists on Linux only.
    pub fn create(path: &Path, slot_bytes: usize, slots: u64, direct: bool) -> io::Result<SlotFile> {
        if slot_bytes == 0 || !slot_bytes.is_multiple_of(ALIGN) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "slot size is not a multiple of 4 KiB"));
        }
        let mut o = OpenOptions::new();
        o.read(true).write(true).create(true).truncate(false);
        #[cfg(target_os = "linux")]
        let direct = {
            use std::os::unix::fs::OpenOptionsExt;
            if direct {
                o.custom_flags(O_DIRECT);
            }
            direct
        };
        #[cfg(not(target_os = "linux"))]
        let direct = {
            let _ = direct;
            false
        };
        let file = o.open(path)?;
        file.set_len(slot_bytes as u64 * slots)?;
        Ok(SlotFile { file, slot_bytes, slots, direct })
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    pub fn slots(&self) -> u64 {
        self.slots
    }
    /// Whether the page cache is bypassed (O_DIRECT).
    pub fn direct(&self) -> bool {
        self.direct
    }
    /// Writes `buf` (aligned, whole slots) at slot `slot`.
    pub fn write(&self, slot: u64, buf: &[u8]) -> io::Result<()> {
        self.check(slot, buf.len())?;
        self.file.write_all_at(buf, slot * self.slot_bytes as u64)
    }
    /// Reads `buf.len()` bytes (aligned, a multiple of 4 KiB) from the start of slot `slot`.
    pub fn read(&self, slot: u64, buf: &mut [u8]) -> io::Result<()> {
        self.read_at(slot, 0, buf)
    }
    /// Reads `buf.len()` bytes at byte `off` (a multiple of 4 KiB) of slot `slot`: one row of a wider slot.
    pub fn read_at(&self, slot: u64, off: u64, buf: &mut [u8]) -> io::Result<()> {
        if !off.is_multiple_of(ALIGN as u64) || off > self.slot_bytes as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "unaligned offset"));
        }
        self.check(slot, off as usize + buf.len())?;
        self.file.read_exact_at(buf, slot * self.slot_bytes as u64 + off)
    }
    fn check(&self, slot: u64, len: usize) -> io::Result<()> {
        if slot >= self.slots || !len.is_multiple_of(ALIGN) || len > self.slot_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "slot out of range or unaligned length"));
        }
        Ok(())
    }
}

/// Runs `f(worker, i)` for i in 0..n on `threads` threads, each i once. Workers take the next index in turn, so the
/// order work completes in varies, but each i's work is the same.
pub fn par_for<F: Fn(usize, usize) -> io::Result<()> + Sync>(threads: usize, n: usize, f: F) -> io::Result<()> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|w| {
                let (next, f) = (&next, &f);
                s.spawn(move || -> io::Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return Ok(());
                        }
                        f(w, i)?;
                    }
                })
            })
            .collect();
        workers.into_iter().try_for_each(|h| h.join().unwrap_or_else(|_| Err(io::Error::other("io worker panicked"))))
    })
}

/// Lowest-free-first slot allocator.
pub struct SlotAlloc {
    used: Vec<bool>,
    free: u64,
}

impl SlotAlloc {
    pub fn new(slots: u64) -> SlotAlloc {
        SlotAlloc { used: vec![false; slots as usize], free: slots }
    }
    pub fn free(&self) -> u64 {
        self.free
    }
    /// `n` slots, the lowest free first, or None (nothing taken) when fewer are free.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<u64>> {
        if (n as u64) > self.free {
            return None;
        }
        let out: Vec<u64> = (0..self.used.len()).filter(|&s| !self.used[s]).take(n).map(|s| s as u64).collect();
        for &s in &out {
            self.used[s as usize] = true;
        }
        self.free -= n as u64;
        Some(out)
    }
    pub fn release(&mut self, slots: &[u64]) {
        for &s in slots {
            debug_assert!(self.used[s as usize]);
            self.used[s as usize] = false;
        }
        self.free += slots.len() as u64;
    }
}

/// A spilled retained run: its page blocks' slots in the run's page order, and its finished reuse state.
pub struct Spilled {
    pub slots: Vec<u64>,
    pub st: Finished,
    pub age: u64,
}

/// Spilled retained runs, by handle (the order they were spilled in).
pub struct Tier {
    file: SlotFile,
    alloc: SlotAlloc,
    spilled: BTreeMap<u64, Spilled>,
    next: u64,
    threads: usize,
    pub bytes_written: u64,
    pub bytes_read: u64,
}

impl Tier {
    /// A store of `slots` page blocks of `block_bytes` each (PAGE_BLOCK_BYTES for the engine's layout).
    pub fn create(path: &Path, block_bytes: usize, slots: u64, threads: usize, direct: bool) -> io::Result<Tier> {
        let file = SlotFile::create(path, block_bytes.div_ceil(ALIGN) * ALIGN, slots, direct)?;
        Ok(Tier {
            alloc: SlotAlloc::new(slots),
            file,
            spilled: BTreeMap::new(),
            next: 0,
            threads,
            bytes_written: 0,
            bytes_read: 0,
        })
    }

    pub fn free_slots(&self) -> u64 {
        self.alloc.free()
    }

    pub fn spilled(&self) -> &BTreeMap<u64, Spilled> {
        &self.spilled
    }

    /// Writes a retained run's pages to the store: `page(p, buf)` fills buf with page p's block (from device memory in
    /// an engine). Returns the run's handle, or None when the store has too few free slots (nothing written).
    pub fn spill<F>(&mut self, k: &Kept, page: F) -> io::Result<Option<u64>>
    where
        F: Fn(u32, &mut [u8]) + Sync,
    {
        let Some(slots) = self.alloc.alloc(k.list.len()) else { return Ok(None) };
        let sb = self.file.slot_bytes();
        let bufs: Vec<std::sync::Mutex<AlignedBuf>> =
            (0..self.threads.max(1)).map(|_| std::sync::Mutex::new(AlignedBuf::new(sb))).collect();
        let r = par_for(self.threads, slots.len(), |w, i| {
            let mut b = bufs[w].lock().expect("one worker per buffer");
            page(k.list[i], b.as_mut_slice());
            self.file.write(slots[i], b.as_slice())
        });
        if let Err(e) = r {
            self.alloc.release(&slots);
            return Err(e);
        }
        self.bytes_written += (slots.len() * sb) as u64;
        let h = self.next;
        self.next += 1;
        self.spilled.insert(h, Spilled { slots, st: k.st.clone(), age: k.age });
        Ok(Some(h))
    }

    /// Reads a spilled run's first `pages.len()` page blocks back into `pages` (the pages a fresh grant took):
    /// `put(p, buf)` copies buf into page p. The run leaves the store either way. Returns its reuse state.
    pub fn restore<F>(&mut self, h: u64, pages: &[u32], put: F) -> io::Result<Finished>
    where
        F: Fn(u32, &[u8]) + Sync,
    {
        let s =
            self.spilled.remove(&h).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such spilled run"))?;
        let n = pages.len().min(s.slots.len());
        let sb = self.file.slot_bytes();
        let bufs: Vec<std::sync::Mutex<AlignedBuf>> =
            (0..self.threads.max(1)).map(|_| std::sync::Mutex::new(AlignedBuf::new(sb))).collect();
        let r = par_for(self.threads, n, |w, i| {
            let mut b = bufs[w].lock().expect("one worker per buffer");
            self.file.read(s.slots[i], b.as_mut_slice())?;
            put(pages[i], b.as_slice());
            Ok(())
        });
        self.alloc.release(&s.slots);
        r?;
        self.bytes_read += (n * sb) as u64;
        Ok(s.st)
    }

    /// Drops a spilled run without reading it.
    pub fn drop_run(&mut self, h: u64) {
        if let Some(s) = self.spilled.remove(&h) {
            self.alloc.release(&s.slots);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(p: u32, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (p as usize * 31 + i * 7) as u8;
        }
    }

    #[test]
    fn layout_sizes() {
        assert_eq!(PAGE_BLOCK_BYTES, 3_502_592);
        assert_eq!(PAGE_SLOT, 856 * 4096);
        assert_eq!(ROW_BYTES, 2_624);
    }

    #[test]
    fn spill_and_restore_round_trip() {
        let dir = std::env::temp_dir().join(format!("g6kvman-tier-{}", std::process::id()));
        let mut t = Tier::create(&dir, 3 * ALIGN, 16, 3, false).unwrap();
        let a = Kept { list: vec![5, 2, 9, 0], st: Finished { valid: 100, ..Default::default() }, age: 1 };
        let b = Kept { list: vec![1, 3], st: Finished { valid: 7, ..Default::default() }, age: 2 };
        let ha = t.spill(&a, block).unwrap().unwrap();
        let hb = t.spill(&b, block).unwrap().unwrap();
        assert_eq!(t.spilled()[&ha].slots, vec![0, 1, 2, 3]);
        assert_eq!(t.spilled()[&hb].slots, vec![4, 5]);
        assert!(t.spill(&Kept { list: vec![0; 11], st: Finished::default(), age: 3 }, block).unwrap().is_none());
        // restore a's first three pages into other pages: each gets a's page block in list order
        let got = std::sync::Mutex::new(Vec::new());
        let st = t.restore(ha, &[40, 41, 42], |p, buf| got.lock().unwrap().push((p, buf.to_vec()))).unwrap();
        assert_eq!(st.valid, 100);
        let mut got = got.into_inner().unwrap();
        got.sort_by_key(|g| g.0);
        for (i, (p, buf)) in got.iter().enumerate() {
            let mut want = vec![0u8; 3 * ALIGN];
            block(a.list[i], &mut want);
            assert_eq!((*p, buf), (40 + i as u32, &want));
        }
        // a's slots are free again, and the next spill takes the lowest
        assert_eq!(t.free_slots(), 14);
        let hc = t.spill(&Kept { list: vec![7], st: Finished::default(), age: 4 }, block).unwrap().unwrap();
        assert_eq!(t.spilled()[&hc].slots, vec![0]);
        assert_eq!(t.bytes_read, 3 * 3 * ALIGN as u64);
        drop(t);
        std::fs::remove_file(&dir).unwrap();
    }
}
