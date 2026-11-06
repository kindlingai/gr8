//! The socket calls and constants the control links use, declared here (std has no keepalive, poll or raw socket
//! options). Linux and macOS only. The C++ compares every constant and struct layout here with its system headers
//! once (control_io.h check_sys, through `g6ctl_sys`), so a wrong value fails at startup.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("the control links know Linux's and macOS's socket constants only");

pub type socklen_t = u32;
#[cfg(target_os = "linux")]
pub type nfds_t = std::ffi::c_ulong;
#[cfg(target_os = "macos")]
pub type nfds_t = std::ffi::c_uint;

#[repr(C)]
pub struct pollfd {
    pub fd: c_int,
    pub events: i16,
    pub revents: i16,
}

#[repr(C)]
pub struct timeval {
    pub tv_sec: i64,
    #[cfg(target_os = "linux")]
    pub tv_usec: i64,
    #[cfg(target_os = "macos")]
    pub tv_usec: i32,
}

#[repr(C)]
pub struct sockaddr_in {
    #[cfg(target_os = "macos")]
    pub sin_len: u8,
    #[cfg(target_os = "macos")]
    pub sin_family: u8,
    #[cfg(target_os = "linux")]
    pub sin_family: u16,
    pub sin_port: u16,
    pub sin_addr: u32,
    pub sin_zero: [u8; 8],
}

extern "C" {
    pub fn socket(domain: c_int, ty: c_int, protocol: c_int) -> c_int;
    pub fn setsockopt(fd: c_int, level: c_int, name: c_int, value: *const c_void, len: socklen_t) -> c_int;
    pub fn getsockopt(fd: c_int, level: c_int, name: c_int, value: *mut c_void, len: *mut socklen_t) -> c_int;
    pub fn bind(fd: c_int, addr: *const sockaddr_in, len: socklen_t) -> c_int;
    pub fn listen(fd: c_int, backlog: c_int) -> c_int;
    #[cfg(target_os = "macos")]
    pub fn accept(fd: c_int, addr: *mut c_void, len: *mut socklen_t) -> c_int;
    #[cfg(target_os = "linux")]
    pub fn accept4(fd: c_int, addr: *mut c_void, len: *mut socklen_t, flags: c_int) -> c_int;
    pub fn connect(fd: c_int, addr: *const sockaddr_in, len: socklen_t) -> c_int;
    pub fn poll(fds: *mut pollfd, n: nfds_t, timeout_ms: c_int) -> c_int;
    pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    pub fn inet_pton(af: c_int, src: *const c_char, dst: *mut c_void) -> c_int;
    pub fn send(fd: c_int, buf: *const c_void, n: usize, flags: c_int) -> isize;
    pub fn recv(fd: c_int, buf: *mut c_void, n: usize, flags: c_int) -> isize;
    pub fn shutdown(fd: c_int, how: c_int) -> c_int;
    pub fn close(fd: c_int) -> c_int;
    pub fn strerror(e: c_int) -> *const c_char;
}

pub const AF_INET: c_int = 2;
pub const SOCK_STREAM: c_int = 1;
pub const IPPROTO_TCP: c_int = 6;
pub const TCP_NODELAY: c_int = 1;
pub const SHUT_RDWR: c_int = 2;
pub const F_GETFL: c_int = 3;
pub const F_SETFL: c_int = 4;
pub const F_SETFD: c_int = 2;
pub const FD_CLOEXEC: c_int = 1;
pub const EINTR: c_int = 4;
pub const POLLIN: i16 = 1;
pub const POLLOUT: i16 = 4;
pub const POLLNVAL: i16 = 0x20;

#[cfg(target_os = "linux")]
mod os {
    use std::ffi::c_int;
    pub const SOL_SOCKET: c_int = 1;
    pub const SO_REUSEADDR: c_int = 2;
    pub const SO_ERROR: c_int = 4;
    pub const SO_KEEPALIVE: c_int = 9;
    pub const SO_RCVTIMEO: c_int = 20;
    pub const SO_SNDTIMEO: c_int = 21;
    pub const SO_NOSIGPIPE: c_int = -1;
    pub const TCP_KEEPALIVE_IDLE: c_int = 4; // TCP_KEEPIDLE
    pub const TCP_KEEPINTVL: c_int = 5;
    pub const TCP_KEEPCNT: c_int = 6;
    pub const TCP_USER_TIMEOUT: c_int = 18;
    pub const MSG_NOSIGNAL: c_int = 0x4000;
    pub const SOCK_CLOEXEC: c_int = 0x80000;
    pub const SOCK_NONBLOCK: c_int = 0x800;
    pub const O_NONBLOCK: c_int = 0x800;
    pub const EAGAIN: c_int = 11;
    pub const EWOULDBLOCK: c_int = 11;
    pub const EINPROGRESS: c_int = 115;
    pub const ECONNREFUSED: c_int = 111;
}
#[cfg(target_os = "macos")]
mod os {
    use std::ffi::c_int;
    pub const SOL_SOCKET: c_int = 0xffff;
    pub const SO_REUSEADDR: c_int = 0x4;
    pub const SO_ERROR: c_int = 0x1007;
    pub const SO_KEEPALIVE: c_int = 0x8;
    pub const SO_RCVTIMEO: c_int = 0x1006;
    pub const SO_SNDTIMEO: c_int = 0x1005;
    pub const SO_NOSIGPIPE: c_int = 0x1022;
    pub const TCP_KEEPALIVE_IDLE: c_int = 0x10; // TCP_KEEPALIVE
    pub const TCP_KEEPINTVL: c_int = 0x101;
    pub const TCP_KEEPCNT: c_int = 0x102;
    pub const TCP_USER_TIMEOUT: c_int = -1;
    pub const MSG_NOSIGNAL: c_int = 0;
    pub const SOCK_CLOEXEC: c_int = 0;
    pub const SOCK_NONBLOCK: c_int = 0;
    pub const O_NONBLOCK: c_int = 0x4;
    pub const EAGAIN: c_int = 35;
    pub const EWOULDBLOCK: c_int = 35;
    pub const EINPROGRESS: c_int = 36;
    pub const ECONNREFUSED: c_int = 61;
}
pub use os::*;

/// The flags every send passes: no SIGPIPE from a closed peer (macOS sets SO_NOSIGPIPE on the socket instead).
pub const SEND_FLAGS: c_int = MSG_NOSIGNAL;

/// errno after the call that just failed.
pub fn errno() -> c_int {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// strerror's text for `e`.
pub fn strerror_text(e: c_int) -> String {
    // SAFETY: strerror returns a NUL-terminated static or thread-local string, read here before any other libc call
    unsafe { std::ffi::CStr::from_ptr(strerror(e)) }.to_string_lossy().into_owned()
}

#[cfg(target_os = "linux")]
pub fn cloexec_socket(domain: c_int, ty: c_int, nonblock: bool) -> c_int {
    // SAFETY: plain syscall
    unsafe { socket(domain, ty | SOCK_CLOEXEC | if nonblock { SOCK_NONBLOCK } else { 0 }, 0) }
}
#[cfg(target_os = "linux")]
pub fn cloexec_accept(listener: c_int) -> c_int {
    // SAFETY: no peer address is asked for
    unsafe { accept4(listener, std::ptr::null_mut(), std::ptr::null_mut(), SOCK_CLOEXEC) }
}

// macOS: close-on-exec, the blocking mode and SO_NOSIGPIPE by fcntl and setsockopt, and an accepted socket loses the
// O_NONBLOCK it inherits from its listener, as on Linux
#[cfg(target_os = "macos")]
fn set_socket_flags(fd: c_int, nonblock: bool) -> c_int {
    let on: c_int = 1;
    // SAFETY: fcntl and setsockopt on a descriptor this function owns; `on` outlives the call
    unsafe {
        let fl = fcntl(fd, F_GETFL);
        if fl < 0
            || fcntl(fd, F_SETFD, FD_CLOEXEC) != 0
            || fcntl(fd, F_SETFL, if nonblock { fl | O_NONBLOCK } else { fl & !O_NONBLOCK }) != 0
            || setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, (&on as *const c_int).cast(), 4) != 0
        {
            close(fd);
            return -1;
        }
    }
    fd
}
#[cfg(target_os = "macos")]
pub fn cloexec_socket(domain: c_int, ty: c_int, nonblock: bool) -> c_int {
    // SAFETY: plain syscall
    let fd = unsafe { socket(domain, ty, 0) };
    if fd < 0 {
        fd
    } else {
        set_socket_flags(fd, nonblock)
    }
}
#[cfg(target_os = "macos")]
pub fn cloexec_accept(listener: c_int) -> c_int {
    // SAFETY: no peer address is asked for
    let fd = unsafe { accept(listener, std::ptr::null_mut(), std::ptr::null_mut()) };
    if fd < 0 {
        fd
    } else {
        set_socket_flags(fd, false)
    }
}

/// setsockopt with an int value.
pub fn set_int(fd: c_int, level: c_int, name: c_int, v: c_int) -> bool {
    // SAFETY: `v` outlives the call
    unsafe { setsockopt(fd, level, name, (&v as *const c_int).cast(), 4) == 0 }
}

/// TCP_USER_TIMEOUT where the system has it; macOS has none, where only the keepalive probes see a dead peer.
pub fn set_user_timeout(fd: c_int, ms: u32) -> bool {
    if TCP_USER_TIMEOUT < 0 {
        return true;
    }
    // SAFETY: `ms` outlives the call
    unsafe { setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, (&ms as *const u32).cast(), 4) == 0 }
}

/// The constants and layouts above, by index (g6sched.h G6CTL_SYS_*), for the C++ to compare with its headers.
pub fn constant(what: u32) -> i64 {
    use std::mem::{offset_of, size_of};
    let v: [i64; 43] = [
        AF_INET.into(),
        SOCK_STREAM.into(),
        SOL_SOCKET.into(),
        SO_REUSEADDR.into(),
        SO_ERROR.into(),
        SO_RCVTIMEO.into(),
        SO_SNDTIMEO.into(),
        SO_KEEPALIVE.into(),
        IPPROTO_TCP.into(),
        TCP_NODELAY.into(),
        TCP_KEEPALIVE_IDLE.into(),
        TCP_KEEPINTVL.into(),
        TCP_KEEPCNT.into(),
        TCP_USER_TIMEOUT.into(),
        SEND_FLAGS.into(),
        SOCK_CLOEXEC.into(),
        SOCK_NONBLOCK.into(),
        SO_NOSIGPIPE.into(),
        O_NONBLOCK.into(),
        F_GETFL.into(),
        F_SETFL.into(),
        F_SETFD.into(),
        FD_CLOEXEC.into(),
        EINTR.into(),
        EAGAIN.into(),
        EWOULDBLOCK.into(),
        EINPROGRESS.into(),
        ECONNREFUSED.into(),
        POLLIN.into(),
        POLLOUT.into(),
        POLLNVAL.into(),
        SHUT_RDWR.into(),
        size_of::<sockaddr_in>() as i64,
        offset_of!(sockaddr_in, sin_family) as i64,
        offset_of!(sockaddr_in, sin_port) as i64,
        offset_of!(sockaddr_in, sin_addr) as i64,
        size_of::<timeval>() as i64,
        offset_of!(timeval, tv_usec) as i64,
        size_of::<pollfd>() as i64,
        offset_of!(pollfd, events) as i64,
        offset_of!(pollfd, revents) as i64,
        size_of::<socklen_t>() as i64,
        size_of::<nfds_t>() as i64,
    ];
    v.get(what as usize).copied().unwrap_or(i64::MIN)
}
