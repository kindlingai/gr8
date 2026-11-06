//! The ranks' bootstrap over TCP, after each rank loaded its model: the head accepts one link per worker, each worker
//! says its rank and agreement (settings and runtime signature), the head checks them against its own and releases
//! every worker, and each acknowledges. Then the store agreement (`agree_lines`) and, after the graph capture, the
//! capture barrier. A failed bootstrap never publishes ready.s2. The worker's connect retries while the head is still
//! loading, which only the bootstrap may do.

use super::io::Io;
use super::sys;
use super::{require, Fail, R};
use std::borrow::Cow;
use std::ffi::{c_int, CString};
use std::time::{Duration, Instant};

/// The bootstrap's settings, as the C++ BootstrapConfig holds them; `agreement` is its agreement() string.
pub struct Boot<'a> {
    pub rank: u32,
    pub ranks: u32,
    pub timeout: u32,
    pub port: u16,
    pub head_ip: &'a [u8],
    pub directory: &'a [u8],
    pub agreement: &'a [u8],
}

// a descriptor closed when dropped, unless released
struct Socket(c_int);
impl Socket {
    fn release(mut self) -> c_int {
        std::mem::replace(&mut self.0, -1)
    }
}
impl Drop for Socket {
    fn drop(&mut self) {
        if self.0 >= 0 {
            // SAFETY: the descriptor is this guard's
            unsafe { sys::close(self.0) };
        }
    }
}

fn deadline(seconds: u32) -> R<Instant> {
    require(seconds != 0 && seconds <= 86400, "invalid control timeout")?;
    Ok(Instant::now() + Duration::from_secs(u64::from(seconds)))
}
fn remaining_ms(end: Instant) -> R<c_int> {
    let n = end.saturating_duration_since(Instant::now()).as_millis();
    require(n > 0, "control bootstrap timed out")?;
    Ok(n.min(i32::MAX as u128) as c_int)
}
fn socket_wait(fd: c_int, events: i16, end: Instant) -> R<()> {
    loop {
        let mut p = sys::pollfd { fd, events, revents: 0 };
        // SAFETY: one live pollfd
        let n = unsafe { sys::poll(&mut p, 1, remaining_ms(end)?) };
        if n < 0 && sys::errno() == sys::EINTR {
            continue;
        }
        // the caller's accept, SO_ERROR or recv tells a hangup from readiness
        return require(n > 0 && p.revents & sys::POLLNVAL == 0, "control socket wait failed");
    }
}
// up to the first NUL, as the C++ passed c_str()
fn c_text(s: &[u8]) -> CString {
    let n = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    CString::new(&s[..n]).unwrap_or_default()
}
fn address(ip: &[u8], port: u16) -> R<sys::sockaddr_in> {
    let mut a = sys::sockaddr_in {
        #[cfg(target_os = "macos")]
        sin_len: 0,
        sin_family: sys::AF_INET as _,
        sin_port: port.to_be(),
        sin_addr: 0,
        sin_zero: [0; 8],
    };
    let ip = c_text(ip);
    // SAFETY: ip is NUL-terminated, and sin_addr holds an in_addr
    let ok =
        port != 0 && unsafe { sys::inet_pton(sys::AF_INET, ip.as_ptr(), (&mut a.sin_addr as *mut u32).cast()) } == 1;
    require(ok, "control requires IPv4 and nonzero port")?;
    Ok(a)
}
const SOCKADDR_LEN: sys::socklen_t = std::mem::size_of::<sys::sockaddr_in>() as sys::socklen_t;

/// BootstrapConfig::validate's address and timeout checks.
pub fn check(ip: &[u8], port: u16, timeout: u32) -> R<()> {
    address(ip, port)?;
    deadline(timeout)?;
    Ok(())
}

fn listen_control(ip: &[u8], port: u16) -> R<Socket> {
    let fd = Socket(sys::cloexec_socket(sys::AF_INET, sys::SOCK_STREAM, true));
    require(fd.0 >= 0, "control socket failed")?;
    require(sys::set_int(fd.0, sys::SOL_SOCKET, sys::SO_REUSEADDR, 1), "control reuseaddr failed")?;
    let a = address(ip, port)?;
    // SAFETY: a is a live sockaddr_in
    let ok = unsafe { sys::bind(fd.0, &a, SOCKADDR_LEN) == 0 && sys::listen(fd.0, 8) == 0 };
    require(ok, "control listen failed")?;
    Ok(fd)
}
// a dead peer is seen in at most 30 s
fn configure(fd: Socket, timeout: u32) -> R<Io> {
    let io = Io::new(fd.release()).ok_or_else(|| super::refused("bad control fd"))?;
    io.timeout_seconds(timeout)?;
    io.keepalive(10, 5, 4)?;
    io.nodelay()?;
    Ok(io)
}
fn accept_control(listener: c_int, timeout: u32) -> R<Io> {
    let end = deadline(timeout)?;
    loop {
        socket_wait(listener, sys::POLLIN, end)?;
        let fd = Socket(sys::cloexec_accept(listener));
        if fd.0 < 0 {
            let e = sys::errno();
            if e == sys::EINTR || e == sys::EAGAIN || e == sys::EWOULDBLOCK {
                continue;
            }
        }
        require(fd.0 >= 0, "control accept failed")?;
        return configure(fd, timeout);
    }
}
fn connect_control(ip: &[u8], port: u16, timeout: u32) -> R<Io> {
    let a = address(ip, port)?;
    let end = deadline(timeout)?;
    loop {
        let fd = Socket(sys::cloexec_socket(sys::AF_INET, sys::SOCK_STREAM, true));
        require(fd.0 >= 0, "control socket failed")?;
        // SAFETY: a is a live sockaddr_in
        let rc = unsafe { sys::connect(fd.0, &a, SOCKADDR_LEN) };
        let mut error: c_int = 0;
        if rc != 0 {
            let e = sys::errno();
            if e == sys::EINPROGRESS {
                socket_wait(fd.0, sys::POLLOUT, end)?;
                let mut n = std::mem::size_of::<c_int>() as sys::socklen_t;
                // SAFETY: error and n outlive the call
                let ok = unsafe {
                    sys::getsockopt(fd.0, sys::SOL_SOCKET, sys::SO_ERROR, (&mut error as *mut c_int).cast(), &mut n)
                } == 0;
                require(ok, "control connect status failed")?;
            } else {
                error = e;
            }
        }
        if error == 0 {
            // SAFETY: plain fcntl on the owned descriptor
            require(unsafe { sys::fcntl(fd.0, sys::F_SETFL, 0 as c_int) } == 0, "control blocking mode failed")?;
            return configure(fd, timeout);
        }
        require(error == sys::ECONNREFUSED || error == sys::EINTR, "control connect failed")?;
        // a worker may finish loading before the head
        std::thread::sleep(Duration::from_millis(20.min(remaining_ms(end)?) as u64));
    }
}

fn number(s: &[u8]) -> R<u64> {
    require(!s.is_empty(), "empty number")?;
    let mut n: u64 = 0;
    for &c in s {
        require(c.is_ascii_digit() && n <= (u64::MAX - u64::from(c - b'0')) / 10, "bad/overflow number")?;
        n = n * 10 + u64::from(c - b'0');
    }
    Ok(n)
}

/// The first word where a worker's agreement differs from the head's, for the refusal: "rank 2 has 'G6_X=1', the head
/// has 'G6_X=2'". Empty when they are equal.
pub fn agreement_difference(rank: u32, head: &[u8], worker: &[u8]) -> Vec<u8> {
    let words = |s: &[u8]| -> Vec<Vec<u8>> {
        s.split(|&c| c == b' ' || c == b'\n').filter(|w| !w.is_empty()).map(<[u8]>::to_vec).collect()
    };
    let (h, w) = (words(head), words(worker));
    let empty = Vec::new();
    for i in 0..h.len().max(w.len()) {
        let (a, b) = (h.get(i).unwrap_or(&empty), w.get(i).unwrap_or(&empty));
        if i >= h.len() || i >= w.len() || a != b {
            let mut out = format!("rank {rank} has '").into_bytes();
            out.extend_from_slice(b);
            out.extend_from_slice(b"', the head has '");
            out.extend_from_slice(a);
            out.push(b'\'');
            return out;
        }
    }
    Vec::new()
}

fn exists(dir: &[u8], name: &str) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let mut p = dir[..dir.iter().position(|&c| c == 0).unwrap_or(dir.len())].to_vec();
    p.push(b'/');
    p.extend_from_slice(name.as_bytes());
    std::fs::metadata(std::ffi::OsStr::from_bytes(&p)).is_ok()
}
fn cat(a: &[u8], b: &[u8]) -> Vec<u8> {
    [a, b].concat()
}

/// The links of this rank: the head's to ranks 1.. in rank order, a worker's one to the head. The C++ validated the
/// settings. The directory must be fresh.
pub fn bootstrap(c: &Boot) -> R<Vec<Io>> {
    for f in ["ready", "ready.s2", "abort.s2", "quit", "cmd.0", "req.0"] {
        require(!exists(c.directory, f), "S2 directory is not fresh")?;
    }
    let release = cat(b"G6S2 START\n", c.agreement);
    let ack = cat(b"G6S2 STARTED\n", c.agreement);
    let mut links: Vec<Option<Io>> = Vec::new();
    let r = (|| {
        if c.rank == 0 {
            let listener = listen_control(c.head_ip, c.port)?;
            links.resize_with(c.ranks as usize - 1, || None);
            for _ in 1..c.ranks {
                let io = accept_control(listener.0, c.timeout)?;
                let hello = io.receive(false)?;
                let eol = hello.iter().position(|&x| x == b'\n');
                require(eol.is_some() && hello.starts_with(b"G6S2 "), "bad bootstrap hello")?;
                let eol = eol.unwrap_or(0);
                let nr = number(&hello[5..eol])?;
                require(nr > 0 && nr < u64::from(c.ranks), "bad bootstrap worker rank")?;
                let rank = nr as usize;
                require(links[rank - 1].is_none(), "duplicate rank")?;
                let diff = agreement_difference(nr as u32, c.agreement, &hello[eol + 1..]);
                if !diff.is_empty() {
                    return Err(Fail::Refused(Cow::Owned(cat(b"bootstrap config mismatch: ", &diff))));
                }
                links[rank - 1] = Some(io);
            }
            for io in links.iter().flatten() {
                io.send(&release)?;
            }
            for io in links.iter().flatten() {
                require(io.receive(false)? == ack, "worker bootstrap ACK mismatch")?;
            }
            drop(listener);
        } else {
            let io = connect_control(c.head_ip, c.port, c.timeout)?;
            io.send(&cat(format!("G6S2 {}\n", c.rank).as_bytes(), c.agreement))?;
            require(io.receive(false)? == release, "head bootstrap config mismatch")?;
            io.send(&ack)?;
            links.push(Some(io));
        }
        Ok(())
    })();
    if let Err(e) = r {
        links.iter().flatten().for_each(Io::shutdown);
        return Err(e);
    }
    Ok(links.into_iter().flatten().collect())
}

fn lines(s: &[u8]) -> Vec<&[u8]> {
    s.split(|&c| c == b'\n').filter(|l| !l.is_empty()).collect()
}

/// After bootstrap every rank sends its lines (one stored entry each); the head keeps the lines every rank has, in its
/// own order, and sends them back, so every rank starts from the same stored entries. idle: wait for the other ranks
/// with no receive timeout, as the capture barrier does. A worker's lines past the socket buffers wait for the head
/// under the send timeout: a head busy for longer first fails the worker with "control send failed".
pub fn agree_lines(rank: u32, links: &[Option<&Io>], mine: &[u8], idle: bool) -> R<Vec<u8>> {
    const ASK: &[u8] = b"G6S2 STORED\n";
    const AGREED: &[u8] = b"G6S2 AGREED\n";
    if rank != 0 {
        let io = match links {
            [Some(io)] => *io,
            _ => return Err(super::refused("store agreement without head link")),
        };
        io.send(&cat(ASK, mine))?;
        let r = io.receive(idle)?;
        require(r.starts_with(AGREED), "store agreement: unexpected frame")?;
        return Ok(r[AGREED.len()..].to_vec());
    }
    let mut keep = lines(mine);
    for io in links {
        let io = io.ok_or_else(|| super::refused("store agreement with missing rank"))?;
        let r = io.receive(idle)?;
        require(r.starts_with(ASK), "store agreement: unexpected frame")?;
        let theirs = lines(&r[ASK.len()..]);
        keep.retain(|l| theirs.contains(l));
    }
    let mut out = Vec::new();
    for l in keep {
        out.extend_from_slice(l);
        out.push(b'\n');
    }
    let frame = cat(AGREED, &out);
    for io in links.iter().flatten() {
        io.send(&frame)?;
    }
    Ok(out)
}

/// The all-rank barrier after the boot graph capture: every worker reports to the head. The head waits for every
/// worker before it publishes ready.s2. Idle (no receive timeout): capture times may differ by more than the bootstrap
/// timeout, and a rank that dies mid-capture closes its link or fails the keepalive.
pub fn capture_barrier(c: &Boot, links: &[Option<&Io>]) -> R<()> {
    let done = cat(b"G6S2 CAPTURED\n", c.agreement);
    if c.rank != 0 {
        let io = match links {
            [Some(io)] => *io,
            _ => return Err(super::refused("capture barrier without head link")),
        };
        return io.send(&done);
    }
    require(links.len() as u64 == u64::from(c.ranks) - 1, "capture barrier before all-rank bootstrap")?;
    for io in links {
        let io = io.ok_or_else(|| super::refused("capture barrier with missing rank"))?;
        require(io.receive(true)? == done, "worker capture barrier mismatch")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difference_names_the_first_word() {
        assert!(agreement_difference(1, b"a b\nenv G6_X=1", b"a b\nenv G6_X=1").is_empty());
        assert_eq!(
            agreement_difference(1, b"a b\nenv G6_X=1", b"a b\nenv G6_X=2"),
            b"rank 1 has 'G6_X=2', the head has 'G6_X=1'"
        );
        assert_eq!(agreement_difference(3, b"a b", b"a b c"), b"rank 3 has 'c', the head has ''");
    }

    #[test]
    fn numbers() {
        assert_eq!(number(b"12").unwrap(), 12);
        assert!(number(b"").is_err() && number(b"1x").is_err() && number(b"18446744073709551616").is_err());
        assert_eq!(number(b"18446744073709551615").unwrap(), u64::MAX);
    }
}
