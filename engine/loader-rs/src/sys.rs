//! The few libc calls std does not wrap: mmap / madvise / munmap and posix_fadvise (Linux, aarch64 and x86_64).

use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;

#[cfg(target_arch = "aarch64")]
pub const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))]
pub const O_DIRECT: i32 = 0o40000;

const PROT_READ: i32 = 1;
const MAP_SHARED: i32 = 1;
const MADV_DONTNEED: i32 = 4;
const MADV_POPULATE_READ: i32 = 22;
const POSIX_FADV_DONTNEED: i32 = 4;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
    fn sysconf(name: i32) -> i64;
}

const SC_PAGESIZE: i32 = 30;

pub fn fadvise_dontneed(f: &File, off: u64, len: u64) {
    unsafe { posix_fadvise(f.as_raw_fd(), off as i64, len as i64, POSIX_FADV_DONTNEED) };
}

/// A whole file mapped read-only. copy_out prefaults the range (MADV_POPULATE_READ), copies it, then drops the
/// pages again, so a loaded image does not stay in the page cache next to its device copy.
pub struct Map {
    base: *mut u8,
    len: usize,
    page: u64,
    fd: i32,
}

unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    pub fn new(f: &File) -> std::io::Result<Map> {
        let len = f.metadata()?.len() as usize;
        let page = unsafe { sysconf(SC_PAGESIZE) }.max(4096) as u64;
        if len == 0 {
            return Ok(Map { base: std::ptr::null_mut(), len, page, fd: f.as_raw_fd() });
        }
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, f.as_raw_fd(), 0) };
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Map { base: p as *mut u8, len, page, fd: f.as_raw_fd() })
    }

    pub fn copy_out(&self, off: u64, dst: &mut [u8]) -> Result<(), String> {
        let end = off + dst.len() as u64;
        if end > self.len as u64 {
            return Err(format!("range {off}..{end} past the end of the file ({} bytes)", self.len));
        }
        let (a, b) = (off / self.page * self.page, end.div_ceil(self.page) * self.page).min((u64::MAX, self.len as u64));
        let (pa, n) = (unsafe { self.base.add(a as usize) } as *mut c_void, (b.min(self.len as u64) - a) as usize);
        if unsafe { madvise(pa, n, MADV_POPULATE_READ) } != 0 {
            return Err(format!("MADV_POPULATE_READ: {}", std::io::Error::last_os_error()));
        }
        dst.copy_from_slice(unsafe { std::slice::from_raw_parts(self.base.add(off as usize), dst.len()) });
        unsafe {
            madvise(pa, n, MADV_DONTNEED);
            posix_fadvise(self.fd, a as i64, n as i64, POSIX_FADV_DONTNEED);
        }
        Ok(())
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        if !self.base.is_null() {
            unsafe { munmap(self.base as *mut c_void, self.len) };
        }
    }
}

/// A 4096-aligned scratch buffer (O_DIRECT reads of whole v2 down halves).
pub struct AlignedBuf {
    p: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    pub fn new(n: usize) -> AlignedBuf {
        let layout = std::alloc::Layout::from_size_align(n.div_ceil(4096) * 4096, 4096).unwrap();
        AlignedBuf { p: unsafe { std::alloc::alloc(layout) }, layout }
    }

    pub fn slice(&mut self, n: usize) -> &mut [u8] {
        assert!(n <= self.layout.size());
        unsafe { std::slice::from_raw_parts_mut(self.p, n) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.p, self.layout) };
    }
}
