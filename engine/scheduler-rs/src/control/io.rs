//! One control link: an already connected stream socket (TCP between ranks, a socket pair in the host tests) carrying
//! length-prefixed frames. The link owns the descriptor and closes it when dropped. Its calls take `&self`: one thread
//! may receive while another sends, as the head's and workers' channels do. The channels keep each direction to one
//! thread at a time.

use super::sys;
use super::{require, R};
use std::ffi::c_int;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// The largest frame: event metadata and request payloads (a 393K-token prompt, DS41 images up to 48 MiB of base64),
/// never model tensors.
pub const MAX_BYTES: usize = 64 << 20;

/// A connected control socket.
pub struct Io {
    fd: c_int,
    sent: AtomicU64,
    received: AtomicU64,
}

impl Io {
    /// Takes `fd`; None when it is negative.
    pub fn new(fd: c_int) -> Option<Io> {
        (fd >= 0).then_some(Io { fd, sent: AtomicU64::new(0), received: AtomicU64::new(0) })
    }
    /// Frames sent and received so far.
    pub fn counts(&self) -> (u64, u64) {
        (self.sent.load(Ordering::Relaxed), self.received.load(Ordering::Relaxed))
    }
    // One greppable line in the rank's stderr before the link's failure is thrown: a crash log that showed only
    // "control disconnected/truncated" could not say why the link dropped.
    #[cold]
    fn log_disconnect(&self, dir: &str, k: isize, e: c_int, remaining: usize, total: usize) {
        let (sent, received) = self.counts();
        let line = format!(
            "g6ctl: control {dir} failed: ret={k} errno={e} ({}) remaining={remaining} of {total} bytes, frames sent={sent} recv={received}\n",
            sys::strerror_text(e)
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    fn send_all(&self, mut p: &[u8]) -> R<()> {
        let total = p.len();
        while !p.is_empty() {
            // SAFETY: p is a live slice of p.len() bytes
            let k = unsafe { sys::send(self.fd, p.as_ptr().cast(), p.len(), sys::SEND_FLAGS) };
            let e = if k < 0 { sys::errno() } else { 0 };
            if k < 0 && e == sys::EINTR {
                continue;
            }
            if k <= 0 {
                self.log_disconnect("send", k, if k == 0 { sys::errno() } else { e }, p.len(), total);
            }
            require(k > 0, "control send failed")?;
            p = &p[k as usize..];
        }
        Ok(())
    }
    // idle: the link may stay silent indefinitely before the first byte (a worker's event reader between events), so a
    // receive timeout with nothing read yet is retried. Once a frame has started the timeout applies again. A peer's
    // close or shutdown() still ends the wait.
    fn read_all(&self, p: &mut [u8], idle: bool) -> R<()> {
        let total = p.len();
        let mut at = 0;
        while at < total {
            let left = &mut p[at..];
            // SAFETY: left is a live slice of left.len() bytes
            let k = unsafe { sys::recv(self.fd, left.as_mut_ptr().cast(), left.len(), 0) };
            let e = if k <= 0 { sys::errno() } else { 0 };
            if k < 0 && e == sys::EINTR {
                continue;
            }
            if k < 0 && idle && at == 0 && (e == sys::EAGAIN || e == sys::EWOULDBLOCK) {
                continue;
            }
            if k <= 0 {
                self.log_disconnect("recv", k, e, total - at, total);
            }
            require(k > 0, "control disconnected/truncated")?;
            at += k as usize;
        }
        Ok(())
    }
    /// Wakes both directions on an epoch's failure.
    pub fn shutdown(&self) {
        // SAFETY: plain syscall on the owned descriptor
        unsafe { sys::shutdown(self.fd, sys::SHUT_RDWR) };
    }
    /// The send and receive timeouts. A failure ends the epoch.
    pub fn timeout_seconds(&self, seconds: u32) -> R<()> {
        require(seconds > 0, "control timeout must be positive")?;
        let tv = sys::timeval { tv_sec: i64::from(seconds), tv_usec: 0 };
        let len = std::mem::size_of::<sys::timeval>() as sys::socklen_t;
        // SAFETY: tv outlives both calls
        let ok = unsafe {
            sys::setsockopt(self.fd, sys::SOL_SOCKET, sys::SO_RCVTIMEO, (&tv as *const sys::timeval).cast(), len) == 0
                && sys::setsockopt(self.fd, sys::SOL_SOCKET, sys::SO_SNDTIMEO, (&tv as *const sys::timeval).cast(), len)
                    == 0
        };
        require(ok, "control timeout setup failed")
    }
    /// TCP peer liveness for the idle event wait, which has no timeout: a peer lost without a FIN or RST (power loss,
    /// a pulled cable) is seen by the kernel's keepalive probes after `idle_s` of silence, every `intvl_s`, `cnt`
    /// times, and TCP_USER_TIMEOUT bounds unacknowledged sends the same way. recv or send then fail with ETIMEDOUT.
    pub fn keepalive(&self, idle_s: i32, intvl_s: i32, cnt: i32) -> R<()> {
        require(idle_s > 0 && intvl_s > 0 && cnt > 0, "invalid keepalive")?;
        let user_ms = (idle_s.wrapping_add(intvl_s.wrapping_mul(cnt)) as u32).wrapping_mul(1000);
        let ok = sys::set_int(self.fd, sys::SOL_SOCKET, sys::SO_KEEPALIVE, 1)
            && sys::set_int(self.fd, sys::IPPROTO_TCP, sys::TCP_KEEPALIVE_IDLE, idle_s)
            && sys::set_int(self.fd, sys::IPPROTO_TCP, sys::TCP_KEEPINTVL, intvl_s)
            && sys::set_int(self.fd, sys::IPPROTO_TCP, sys::TCP_KEEPCNT, cnt)
            && sys::set_user_timeout(self.fd, user_ms);
        require(ok, "control keepalive setup failed")
    }
    /// TCP_NODELAY: a frame goes out as two sends (its length, then its body); with Nagle the body waits for the
    /// length's ACK, which the peer, blocked in recv for the body, delays by up to 40 ms.
    pub fn nodelay(&self) -> R<()> {
        require(sys::set_int(self.fd, sys::IPPROTO_TCP, sys::TCP_NODELAY, 1), "control nodelay setup failed")
    }
    /// One frame: its length (4 bytes, big-endian), then its body.
    pub fn send(&self, body: &[u8]) -> R<()> {
        require(!body.is_empty() && body.len() <= MAX_BYTES, "control frame size invalid")?;
        let n = body.len() as u32;
        self.send_all(&n.to_be_bytes())?;
        self.send_all(body)?;
        self.sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// The next frame's body. idle: wait for its length with no timeout (event readers only).
    pub fn receive(&self, idle: bool) -> R<Vec<u8>> {
        let mut h = [0u8; 4];
        self.read_all(&mut h, idle)?;
        let n = u32::from_be_bytes(h) as usize;
        require(n != 0 && n <= MAX_BYTES, "control frame size invalid")?;
        let mut body = vec![0u8; n];
        self.read_all(&mut body, false)?;
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(body)
    }
}

impl Drop for Io {
    fn drop(&mut self) {
        // SAFETY: the descriptor is this link's, closed once
        unsafe { sys::close(self.fd) };
    }
}
