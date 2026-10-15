//! libc calls for the remote store, declared by hand to keep the crate free of dependencies, and the node's memory:
//! one anonymous mapping, mlock()ed. The kernel must never swap or move it under a registered RDMA region.

use std::ffi::{c_char, c_int, c_void};
use std::io;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn mlock(addr: *const c_void, len: usize) -> c_int;
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn getrlimit(resource: c_int, rlim: *mut [u64; 2]) -> c_int;
    fn geteuid() -> u32;
    fn recv(fd: c_int, buf: *mut c_void, len: usize, flags: c_int) -> isize;
    fn send(fd: c_int, buf: *const c_void, len: usize, flags: c_int) -> isize;
    pub fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    pub fn dlerror() -> *const c_char;
}

pub const RTLD_NOW: c_int = 2;
const PROT_RW: c_int = 3;
#[cfg(target_os = "linux")]
const MAP_PRIVATE_ANON: c_int = 0x02 | 0x20;
#[cfg(not(target_os = "linux"))]
const MAP_PRIVATE_ANON: c_int = 0x02 | 0x1000;
const MADV_HUGEPAGE: c_int = 14;
#[cfg(target_os = "linux")]
const RLIMIT_MEMLOCK: c_int = 8;
#[cfg(not(target_os = "linux"))]
const RLIMIT_MEMLOCK: c_int = 6;
#[cfg(target_os = "linux")]
const MSG_NOSIGNAL: c_int = 0x4000;
#[cfg(not(target_os = "linux"))]
const MSG_NOSIGNAL: c_int = 0;
const RLIM_INFINITY: u64 = u64::MAX;

/// RLIMIT_MEMLOCK's soft limit in bytes, None for unlimited.
pub fn memlock_limit() -> Option<u64> {
    let mut r = [0u64; 2];
    if unsafe { getrlimit(RLIMIT_MEMLOCK, &mut r) } != 0 || r[0] == RLIM_INFINITY {
        return None;
    }
    Some(r[0])
}

pub fn is_root() -> bool {
    unsafe { geteuid() == 0 }
}

/// MemAvailable from /proc/meminfo, in bytes.
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
}

const GIB: f64 = (1u64 << 30) as f64;

/// The node's memory. `lock` false is for tests, where RLIMIT_MEMLOCK is usually 8 MiB.
pub struct Arena {
    ptr: *mut u8,
    len: usize,
    pub locked: bool,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(len: usize, lock: bool) -> io::Result<Arena> {
        if lock && !is_root() {
            if let Some(lim) = memlock_limit() {
                if lim < len as u64 {
                    return Err(io::Error::other(format!(
                        "RLIMIT_MEMLOCK is {} bytes ({:.2} GiB), and locking {:.2} GiB needs more. Run it under \
                         `systemd-run -p LimitMEMLOCK=infinity ...`, or raise memlock in /etc/security/limits.conf",
                        lim,
                        lim as f64 / GIB,
                        len as f64 / GIB
                    )));
                }
            }
        }
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_RW, MAP_PRIVATE_ANON, -1, 0) };
        if p as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        // Huge pages cut page-table and NIC translation entries. A kernel without THP refuses, and that is fine.
        unsafe { madvise(p, len, MADV_HUGEPAGE) };
        let mut a = Arena { ptr: p as *mut u8, len, locked: false };
        if lock {
            if unsafe { mlock(p, len) } != 0 {
                let e = io::Error::last_os_error();
                return Err(io::Error::other(format!(
                    "mlock of {:.2} GiB failed: {e} (RLIMIT_MEMLOCK {})",
                    len as f64 / GIB,
                    memlock_limit().map_or("unlimited".to_string(), |l| format!("{l} bytes"))
                )));
            }
            a.locked = true;
        }
        Ok(a)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn base(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr as *mut c_void, self.len) };
    }
}

/// recv() until `len` bytes land at `p`.
///
/// # Safety
/// `p` must be valid for `len` bytes of writes.
pub unsafe fn recv_exact(fd: c_int, p: *mut u8, len: usize) -> io::Result<()> {
    let mut got = 0;
    while got < len {
        let r = recv(fd, p.add(got) as *mut c_void, len - got, 0);
        if r < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if r == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        got += r as usize;
    }
    Ok(())
}

/// send() all `len` bytes at `p`.
///
/// # Safety
/// `p` must be valid for `len` bytes of reads.
pub unsafe fn send_all(fd: c_int, p: *const u8, len: usize) -> io::Result<()> {
    let mut put = 0;
    while put < len {
        let r = send(fd, p.add(put) as *const c_void, len - put, MSG_NOSIGNAL);
        if r < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        put += r as usize;
    }
    Ok(())
}

/// 16 random bytes from the kernel.
pub fn random16() -> io::Result<[u8; 16]> {
    use std::io::Read;
    let mut b = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_locked_arena_stays_mapped() {
        let len = 1 << 20;
        if memlock_limit().is_some_and(|l| l < len as u64) && !is_root() {
            return;
        }
        let a = Arena::new(len, true).unwrap();
        assert!(a.locked);
        unsafe {
            std::ptr::write_bytes(a.base(), 7, len);
            assert_eq!(*a.base().add(len - 1), 7);
        }
    }

    #[test]
    fn too_little_memlock_is_named() {
        let Some(lim) = memlock_limit() else { return };
        if is_root() {
            return;
        }
        let e = Arena::new((lim as usize + (1 << 30)).next_multiple_of(1 << 20), true).err().unwrap();
        assert!(e.to_string().contains("RLIMIT_MEMLOCK"), "{e}");
    }
}
