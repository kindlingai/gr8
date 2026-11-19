//! A read-only file service over TCP: a box serves its persisted KV store's files to the other boxes of a scale
//! (`./gr8 <model> scale`, rewidth.rs, engine/runtime/s2/flash_ckpt_rewidth.h), so a rank of the new width can read
//! the blocks another box's old rank wrote. `kvman-peer` is the server; `Peer` and g6kv_peer_* the client.
//!
//! Wire: the client sends `G6PEER1 <token hex>\n`; the server closes the connection unless the token matches. Then
//! requests, each `u8 op` (1 read, 2 size), `u16 name length`, the name, `u64 offset`, `u64 length` (little endian),
//! answered by `u8 status` (0 ok, 1 error), `u64 n`, then n bytes for a read (a size answers its size as n, no
//! bytes). A name is one or two path parts of `[A-Za-z0-9._-]`, never `..`, under the served folder. A read past the
//! end of a file is an error, never a short answer.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const HELLO: &str = "G6PEER1 ";
const OP_READ: u8 = 1;
const OP_SIZE: u8 = 2;
/// the most one read may ask for
pub const MAX_READ: u64 = 1 << 30;
/// How long `Peer::open` tries again while a peer refuses or times out: an old rank's box starts its kvman-peer only
/// after its engine's orderly quit, which a new rank's engine can reach the store open before.
pub const PEER_WAIT: Duration = Duration::from_secs(60);

/// Whether `name` may be served: one or two path parts of [A-Za-z0-9._-], none of them `.` or `..`.
pub fn name_ok(name: &str) -> bool {
    let parts: Vec<&str> = name.split('/').collect();
    parts.len() <= 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        })
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// The token in a token file: its first line, trimmed.
pub fn read_token(path: &Path) -> io::Result<String> {
    let t = std::fs::read_to_string(path)?;
    let t = t.lines().next().unwrap_or("").trim().to_string();
    if t.is_empty() || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: not a hex token", path.display())));
    }
    Ok(t)
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Serves `dir`'s files on `listener` until the process ends, one thread per connection.
pub fn serve(dir: PathBuf, listener: TcpListener, token: String) {
    for conn in listener.incoming() {
        let Ok(s) = conn else { continue };
        let (dir, token) = (dir.clone(), token.clone());
        std::thread::spawn(move || {
            if let Err(e) = serve_one(&dir, s, &token) {
                if e.kind() != io::ErrorKind::UnexpectedEof {
                    eprintln!("kvman-peer: {e}");
                }
            }
        });
    }
}

fn serve_one(dir: &Path, s: TcpStream, token: &str) -> io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(600)))?;
    let mut rd = BufReader::new(s.try_clone()?);
    let mut line = Vec::new();
    rd.by_ref().take(HELLO.len() as u64 + 130).read_until(b'\n', &mut line)?;
    let got = line.strip_prefix(HELLO.as_bytes()).and_then(|t| t.strip_suffix(b"\n")).unwrap_or_default();
    if !ct_eq(got, token.as_bytes()) {
        return Err(bad("refused a connection: bad token"));
    }
    let mut wr = s;
    loop {
        let mut op = [0u8; 1];
        if rd.read(&mut op)? == 0 {
            return Ok(());
        }
        let mut n2 = [0u8; 2];
        rd.read_exact(&mut n2)?;
        let mut name = vec![0u8; u16::from_le_bytes(n2) as usize];
        rd.read_exact(&mut name)?;
        let mut w = [0u8; 16];
        rd.read_exact(&mut w)?;
        let off = u64::from_le_bytes(w[..8].try_into().expect("8"));
        let len = u64::from_le_bytes(w[8..].try_into().expect("8"));
        let name = String::from_utf8(name).unwrap_or_default();
        let r: io::Result<(u64, Option<File>)> = (|| {
            if !name_ok(&name) {
                return Err(bad(format!("refused name {name:?}")));
            }
            let f = File::open(dir.join(&name))?;
            let size = f.metadata()?.len();
            match op[0] {
                OP_SIZE => Ok((size, None)),
                OP_READ if len <= MAX_READ && off.checked_add(len).is_some_and(|e| e <= size) => Ok((len, Some(f))),
                OP_READ => Err(bad(format!("{name}: {len} B at {off} is past its {size} B"))),
                o => Err(bad(format!("unknown op {o}"))),
            }
        })();
        match r {
            Ok((n, f)) => {
                let mut head = [0u8; 9];
                head[1..].copy_from_slice(&n.to_le_bytes());
                wr.write_all(&head)?;
                if let Some(f) = f {
                    let mut buf = vec![0u8; (1 << 20).min(n as usize)];
                    let (mut at, end) = (off, off + n);
                    while at < end {
                        let k = buf.len().min((end - at) as usize);
                        f.read_exact_at(&mut buf[..k], at)?;
                        wr.write_all(&buf[..k])?;
                        at += k as u64;
                    }
                }
            }
            Err(e) => {
                eprintln!("kvman-peer: {e}");
                let mut head = [0u8; 9];
                head[0] = 1;
                wr.write_all(&head)?;
            }
        }
    }
}

/// A connect error a peer that is not listening yet gives.
fn transient(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::TimedOut
            | io::ErrorKind::WouldBlock
            | io::ErrorKind::Interrupted
    )
}

/// One connection to a peer's file service.
pub struct Peer {
    s: TcpStream,
    addr: String,
}

impl Peer {
    pub fn open(addr: &str, token: &str) -> io::Result<Peer> {
        Peer::open_wait(addr, token, PEER_WAIT)
    }

    /// `open`, trying the connection again every 200 ms for up to `wait` while it is refused or times out.
    pub fn open_wait(addr: &str, token: &str, wait: Duration) -> io::Result<Peer> {
        let sa = addr.to_socket_addrs()?.next().ok_or_else(|| bad(format!("{addr}: no address")))?;
        let end = Instant::now() + wait;
        let mut s = loop {
            match TcpStream::connect_timeout(&sa, Duration::from_secs(10)) {
                Ok(s) => break s,
                Err(e) if transient(&e) && Instant::now() < end => std::thread::sleep(Duration::from_millis(200)),
                Err(e) => return Err(e),
            }
        };
        s.set_read_timeout(Some(Duration::from_secs(120)))?;
        s.set_write_timeout(Some(Duration::from_secs(120)))?;
        s.set_nodelay(true)?;
        s.write_all(format!("{HELLO}{token}\n").as_bytes())?;
        Ok(Peer { s, addr: addr.to_string() })
    }

    fn ask(&mut self, op: u8, name: &str, off: u64, len: u64) -> io::Result<u64> {
        if !name_ok(name) {
            return Err(bad(format!("refused name {name:?}")));
        }
        let mut q = vec![op];
        q.extend_from_slice(&(name.len() as u16).to_le_bytes());
        q.extend_from_slice(name.as_bytes());
        q.extend_from_slice(&off.to_le_bytes());
        q.extend_from_slice(&len.to_le_bytes());
        self.s.write_all(&q)?;
        let mut head = [0u8; 9];
        self.s.read_exact(&mut head).map_err(|e| bad(format!("{}: {e} (bad token?)", self.addr)))?;
        if head[0] != 0 {
            return Err(bad(format!("{}: {name}: refused or unreadable", self.addr)));
        }
        Ok(u64::from_le_bytes(head[1..].try_into().expect("8")))
    }

    /// Reads `dst.len()` bytes of `name` at `off`.
    pub fn read(&mut self, name: &str, off: u64, dst: &mut [u8]) -> io::Result<()> {
        let n = self.ask(OP_READ, name, off, dst.len() as u64)?;
        if n != dst.len() as u64 {
            return Err(bad(format!("{}: {name}: short answer", self.addr)));
        }
        self.s.read_exact(dst)
    }

    pub fn size(&mut self, name: &str) -> io::Result<u64> {
        self.ask(OP_SIZE, name, 0, 0)
    }

    /// The whole of `name`.
    pub fn read_all(&mut self, name: &str) -> io::Result<Vec<u8>> {
        let n = self.size(name)?;
        let mut v = vec![0u8; n as usize];
        self.read(name, 0, &mut v)?;
        Ok(v)
    }
}

// ---- C interface (g6kvman.h g6kv_peer_*) ----

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(e: impl std::fmt::Display) {
    LAST.with(|l| *l.borrow_mut() = CString::new(e.to_string().replace('\0', " ")).unwrap_or_default());
}

/// The message of this thread's last failed g6kv_peer call.
#[no_mangle]
pub extern "C" fn g6kv_peer_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

unsafe fn c_str<'a>(p: *const c_char) -> Option<&'a str> {
    (!p.is_null()).then(|| CStr::from_ptr(p).to_str().ok()).flatten()
}

/// Connects to `addr` (HOST:PORT) with `token` (hex). NULL on failure (g6kv_peer_error).
///
/// # Safety
/// `addr` and `token` are NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn g6kv_peer_open(addr: *const c_char, token: *const c_char) -> *mut Peer {
    let (Some(a), Some(t)) = (c_str(addr), c_str(token)) else {
        fail("g6kv_peer_open: bad arguments");
        return std::ptr::null_mut();
    };
    match Peer::open(a, t) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(e) => {
            fail(format!("{a}: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Reads `len` bytes of `name` at `off` into `dst`. 0, or -1 (g6kv_peer_error).
///
/// # Safety
/// `p` from g6kv_peer_open, `name` NUL-terminated, `dst` valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6kv_peer_read(p: *mut Peer, name: *const c_char, off: u64, len: u64, dst: *mut u8) -> i32 {
    let (Some(p), Some(n)) = (p.as_mut(), c_str(name)) else {
        fail("g6kv_peer_read: bad arguments");
        return -1;
    };
    if dst.is_null() && len > 0 {
        fail("g6kv_peer_read: bad arguments");
        return -1;
    }
    let d = if len == 0 { &mut [][..] } else { std::slice::from_raw_parts_mut(dst, len as usize) };
    match p.read(n, off, d) {
        Ok(()) => 0,
        Err(e) => {
            fail(e);
            -1
        }
    }
}

/// The size of `name`, or -1 (g6kv_peer_error).
///
/// # Safety
/// `p` from g6kv_peer_open, `name` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn g6kv_peer_size(p: *mut Peer, name: *const c_char) -> i64 {
    let (Some(p), Some(n)) = (p.as_mut(), c_str(name)) else {
        fail("g6kv_peer_size: bad arguments");
        return -1;
    };
    match p.size(n) {
        Ok(s) => s as i64,
        Err(e) => {
            fail(e);
            -1
        }
    }
}

/// # Safety
/// `p` from g6kv_peer_open (or NULL), not used after.
#[no_mangle]
pub unsafe extern "C" fn g6kv_peer_close(p: *mut Peer) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}
