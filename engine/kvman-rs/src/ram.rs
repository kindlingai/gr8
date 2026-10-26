//! A RAM cache in front of a block store ("ram:FLOOR+STORE", the store's ram=on): the box's free memory holds copies
//! of the store's blocks, and the kernel takes that memory back whenever it needs it.
//!
//! - Writes go to the store first, then into the cache, whose pages are then given MADV_FREE. Until the kernel needs
//!   the memory, a read copies the block from RAM. A page the kernel took reads as zeros.
//! - Each cached block keeps a hash of its bytes. A read copies the block, then hashes the copy: a mismatch (a page
//!   taken before or during the copy) reads the store instead. Every read returns the store's bytes, so ranks whose
//!   kernels took different pages still agree on everything the pool decides.
//! - A block is cached only while MemAvailable stays above FLOOR bytes (read at most every 100 ms). Below it, filling
//!   the cache would push the engine's own memory into swap.
//! - Slots are whole pages (64 KiB on a 64K kernel), so the kernel's reclaim of one block never touches another.

use crate::blocks::Blocks;
use crate::remote::sys;
use crate::tier::BlockStore;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The hash of a block not in the cache. A block whose hash comes out 0 is stored as 1.
const ABSENT: u64 = 0;
const CHECK_MS: u64 = 100;

pub struct RamCache {
    pub back: Blocks,
    base: *mut u8,
    slot: usize,
    blocks: u64,
    floor: u64,
    sums: Vec<AtomicU64>,
    start: Instant,
    checked_ms: AtomicU64,
    above: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub skipped: AtomicU64,
}

unsafe impl Send for RamCache {}
unsafe impl Sync for RamCache {}

/// Four independent lanes, so the hash runs near copy speed.
fn hash(b: &[u8]) -> u64 {
    let mut h = [0x9e3779b97f4a7c15u64, 0xc2b2ae3d27d4eb4f, 0x165667b19e3779f9, 0x27d4eb2f165667c5];
    let whole = b.len() / 32 * 32;
    for at in (0..whole).step_by(32) {
        for (l, h) in h.iter_mut().enumerate() {
            let w = u64::from_le_bytes(b[at + 8 * l..at + 8 * l + 8].try_into().unwrap());
            *h = (*h ^ w).wrapping_mul(0xff51afd7ed558ccd);
            *h ^= *h >> 29;
        }
    }
    for (i, &x) in b[whole..].iter().enumerate() {
        h[i % 4] = (h[i % 4] ^ x as u64).wrapping_mul(0xff51afd7ed558ccd);
    }
    let v = h[0] ^ h[1].wrapping_mul(3) ^ h[2].wrapping_mul(5) ^ h[3].wrapping_mul(7);
    if v == ABSENT {
        1
    } else {
        v
    }
}

impl RamCache {
    /// Maps one page-rounded slot per block of `back`. No memory is backed until a block is cached.
    pub fn new(back: Blocks, floor: u64) -> io::Result<RamCache> {
        let page = sys::page_size();
        let slot = back.block_bytes().div_ceil(page) * page;
        let blocks = back.blocks();
        let base = sys::map_noreserve(slot * blocks as usize)?;
        Ok(RamCache {
            back,
            base,
            slot,
            blocks,
            floor,
            sums: (0..blocks).map(|_| AtomicU64::new(ABSENT)).collect(),
            start: Instant::now(),
            checked_ms: AtomicU64::new(u64::MAX),
            above: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        })
    }

    fn slot_ptr(&self, block: u64) -> *mut u8 {
        unsafe { self.base.add(block as usize * self.slot) }
    }

    // MemAvailable above the floor, read at most every CHECK_MS; a box without /proc/meminfo caches nothing
    fn room(&self) -> bool {
        let now = self.start.elapsed().as_millis() as u64;
        let last = self.checked_ms.load(Ordering::Relaxed);
        if last == u64::MAX || now >= last + CHECK_MS {
            self.checked_ms.store(now, Ordering::Relaxed);
            let ok = sys::mem_available().is_some_and(|a| a > self.floor);
            self.above.store(ok as u64, Ordering::Relaxed);
        }
        self.above.load(Ordering::Relaxed) == 1
    }

    pub fn describe(&self) -> String {
        format!(
            "RAM cache over {}: {} slots of {} B, floor {:.1} GiB, {} hits {} misses {} skipped",
            self.back.describe(),
            self.blocks,
            self.slot,
            self.floor as f64 / (1u64 << 30) as f64,
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.skipped.load(Ordering::Relaxed)
        )
    }

    /// Drops a cached block's pages now, as the kernel's reclaim would (tests).
    pub fn drop_block(&self, block: u64, pages: std::ops::Range<usize>) {
        let page = sys::page_size();
        let p = unsafe { self.slot_ptr(block).add(pages.start * page) };
        unsafe { sys::drop_now(p, pages.len() * page) };
    }
}

impl Drop for RamCache {
    fn drop(&mut self) {
        unsafe { sys::unmap(self.base, self.slot * self.blocks as usize) };
    }
}

impl BlockStore for RamCache {
    fn block_bytes(&self) -> usize {
        self.back.block_bytes()
    }
    fn blocks(&self) -> u64 {
        self.blocks
    }
    fn write_block(&self, block: u64, buf: &[u8]) -> io::Result<()> {
        self.back.write_block(block, buf)?;
        let sum = &self.sums[block as usize];
        sum.store(ABSENT, Ordering::Release);
        if !self.room() {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let p = self.slot_ptr(block);
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), p, buf.len());
            sys::free_lazily(p, self.slot)?;
        }
        sum.store(hash(buf), Ordering::Release);
        Ok(())
    }
    fn read_block(&self, block: u64, buf: &mut [u8]) -> io::Result<()> {
        let want = self.sums[block as usize].load(Ordering::Acquire);
        if want != ABSENT {
            unsafe { std::ptr::copy_nonoverlapping(self.slot_ptr(block), buf.as_mut_ptr(), buf.len()) };
            if hash(buf) == want {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            self.sums[block as usize].store(ABSENT, Ordering::Release);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.back.read_block(block, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(blocks: u64, floor: u64) -> RamCache {
        RamCache::new(Blocks::open("mem:", "t", 3 * 65536 + 4096, blocks).unwrap(), floor).unwrap()
    }

    fn pattern(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x2545f4914f6cdd1d) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x | 1) as u8
            })
            .collect()
    }

    fn read(c: &RamCache, b: u64) -> Vec<u8> {
        let mut v = vec![0u8; c.block_bytes()];
        c.read_block(b, &mut v).unwrap();
        v
    }

    #[test]
    fn a_kept_block_reads_from_ram() {
        let c = cache(4, 0);
        let n = c.block_bytes();
        for b in 0..4 {
            c.write_block(b, &pattern(b, n)).unwrap();
        }
        for b in 0..4 {
            assert_eq!(read(&c, b), pattern(b, n));
        }
        assert_eq!((c.hits.load(Ordering::Relaxed), c.misses.load(Ordering::Relaxed)), (4, 0));
    }

    #[test]
    fn a_dropped_block_reads_the_store() {
        let c = cache(4, 0);
        let n = c.block_bytes();
        let pages = c.slot / sys::page_size();
        for b in 0..4 {
            c.write_block(b, &pattern(b, n)).unwrap();
        }
        c.drop_block(1, 0..pages); // all of it
        c.drop_block(2, pages - 1..pages); // only its last page
        for b in 0..4 {
            assert_eq!(read(&c, b), pattern(b, n));
        }
        assert_eq!((c.hits.load(Ordering::Relaxed), c.misses.load(Ordering::Relaxed)), (2, 2));
        // a miss forgets the block: the next read goes straight to the store
        assert_eq!(read(&c, 1), pattern(1, n));
        assert_eq!(c.misses.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn below_the_floor_blocks_are_not_cached() {
        let c = cache(2, u64::MAX);
        let n = c.block_bytes();
        c.write_block(0, &pattern(0, n)).unwrap();
        assert_eq!(read(&c, 0), pattern(0, n));
        assert_eq!((c.skipped.load(Ordering::Relaxed), c.hits.load(Ordering::Relaxed)), (1, 0));
    }

    #[test]
    fn a_rewrite_below_the_floor_never_reads_the_old_copy() {
        let mut c = cache(2, 0);
        let n = c.block_bytes();
        c.write_block(0, &pattern(0, n)).unwrap();
        c.floor = u64::MAX;
        c.checked_ms.store(u64::MAX, Ordering::Relaxed);
        c.write_block(0, &pattern(9, n)).unwrap();
        assert_eq!(read(&c, 0), pattern(9, n));
    }

    #[test]
    fn a_block_of_zeros_is_correct_either_way() {
        let c = cache(1, 0);
        let z = vec![0u8; c.block_bytes()];
        c.write_block(0, &z).unwrap();
        c.drop_block(0, 0..1);
        assert_eq!(read(&c, 0), z);
    }
}
