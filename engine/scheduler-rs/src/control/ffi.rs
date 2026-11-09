//! The control links' C entry points (g6sched.h `g6ctl_*`). Fallible calls return 0, 1 when a callback returned
//! nonzero (the facade holds its exception), or 2 when refused; `g6ctl_error` then gives the refusal's message, kept
//! per thread, since the head's control and compute threads call one endpoint at once. Bytes go back to C++ through a
//! grow callback that appends n bytes to the caller's buffer and returns them.
#![allow(clippy::missing_safety_doc)]

use super::bootstrap::{self, Boot};
use super::channel::{Endpoint, Link};
use super::gate::{Agreement, Gate};
use super::io::Io;
use super::wire::{self, Ack, Commit, Dec, Enc, Notice};
use super::{refused, Fail, R};
use crate::scheduler::{Identity, Key};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};

/// g6ctl_frame: any frame's fields. tag picks which are read (encode) or written (decode): notice for NOTICE and
/// COMMIT, ack for ACK, commit for COMMIT, epoch and request and bytes for REQUEST, epoch for QUIT, epoch and rank and
/// seq for SYNC, epoch and seq for GO. A decoded REQUEST's bytes point into the decoded buffer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub tag: u32,
    pub notice: Notice,
    pub ack: Ack,
    pub commit: Commit,
    pub epoch: u64,
    pub request: u64,
    pub seq: u64,
    pub rank: u32,
    pub bytes: *const u8,
    pub n: usize,
}

/// g6ctl_boot: BootstrapConfig's fields the bootstrap reads, and its agreement() string.
#[repr(C)]
pub struct CBoot {
    pub rank: u32,
    pub ranks: u32,
    pub timeout: u32,
    pub port: u16,
    pub head_ip: *const u8,
    pub head_ip_n: usize,
    pub directory: *const u8,
    pub directory_n: usize,
    pub agreement: *const u8,
    pub agreement_n: usize,
}

/// Appends n bytes to the caller's buffer and returns them.
pub type GrowFn = unsafe extern "C" fn(*mut c_void, usize) -> *mut u8;
/// A rank's state digest and its room, at an event's boundary.
pub type DigestFn = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u32) -> i32;
/// The head's commit for the agreed digest.
pub type CommitFn = unsafe extern "C" fn(*mut c_void, u64, *mut Commit) -> i32;
/// The head quit in order.
pub type QuitFn = unsafe extern "C" fn(*mut c_void) -> i32;

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub(crate) fn code<T>(r: R<T>) -> i32 {
    match r {
        Ok(_) => 0,
        Err(Fail::Hook) => 1,
        Err(Fail::Refused(m)) => {
            let n = m.iter().position(|&c| c == 0).unwrap_or(m.len());
            let c = CString::new(&m[..n]).unwrap_or_default();
            ERROR.with(|e| *e.borrow_mut() = c);
            2
        }
    }
}
pub(crate) unsafe fn bytes<'a>(p: *const u8, n: usize) -> &'a [u8] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}
pub(crate) unsafe fn give(b: &[u8], grow: GrowFn, ctx: *mut c_void) {
    let p = grow(ctx, b.len());
    if !b.is_empty() {
        std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len());
    }
}
fn hook(rc: i32) -> R<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Fail::Hook)
    }
}
unsafe fn links<'a>(p: *const *const Io, n: usize) -> Vec<Option<&'a Io>> {
    (0..n).map(|i| (*p.add(i)).as_ref()).collect()
}

/// The calling thread's last refusal, valid until its next refused call.
#[no_mangle]
pub extern "C" fn g6ctl_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

/// A socket constant or layout the Rust declares (G6CTL_SYS_*), for the C++ to compare with its headers.
#[no_mangle]
pub extern "C" fn g6ctl_sys(what: u32) -> i64 {
    super::sys::constant(what)
}

// --- frames

#[no_mangle]
pub unsafe extern "C" fn g6ctl_encode(f: *const Frame, grow: GrowFn, ctx: *mut c_void) -> i32 {
    let f = &*f;
    let b = match f.tag {
        wire::NOTICE => Ok(wire::encode_notice(&f.notice)),
        wire::ACK => Ok(wire::encode_ack(&f.ack)),
        wire::COMMIT => wire::encode_commit(&f.notice, &f.commit),
        wire::REQUEST => Ok(wire::encode_request(f.epoch, f.request, bytes(f.bytes, f.n))),
        wire::QUIT => Ok(wire::encode_quit(f.epoch)),
        wire::SYNC => Ok(wire::encode_sync(f.epoch, f.rank, f.seq)),
        wire::GO => Ok(wire::encode_go(f.epoch, f.seq)),
        _ => Err(refused("unknown control tag")),
    };
    code(b.map(|b| give(&b, grow, ctx)))
}

/// Decodes b as a frame of `tag`, refused as that frame's decoder refuses.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_decode(b: *const u8, n: usize, tag: u32, out: *mut Frame) -> i32 {
    let b = bytes(b, n);
    let f = &mut *out;
    f.tag = tag;
    code((|| {
        match tag {
            wire::NOTICE => f.notice = wire::decode_notice(b)?,
            wire::ACK => f.ack = wire::decode_ack(b)?,
            wire::COMMIT => (f.notice, f.commit) = wire::decode_commit(b)?,
            wire::REQUEST => {
                let (epoch, request, body) = wire::decode_request(b)?;
                (f.epoch, f.request, f.bytes, f.n) = (epoch, request, body.as_ptr(), body.len());
            }
            wire::QUIT => f.epoch = wire::decode_quit(b)?,
            wire::SYNC => (f.epoch, f.rank, f.seq) = wire::decode_sync(b)?,
            wire::GO => (f.epoch, f.seq) = wire::decode_go(b)?,
            _ => return Err(refused("unknown control tag")),
        }
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn g6ctl_match(n: *const Notice, c: *const Commit) -> i32 {
    code(wire::check_match(&*n, &*c))
}

// the C++ Encoder's and Decoder's pieces; `put` appends to the buffer, `get` reads at *i and advances it
fn put(grow: GrowFn, ctx: *mut c_void, f: impl FnOnce(&mut Enc)) {
    let mut e = Enc { b: Vec::new() };
    f(&mut e);
    // SAFETY: the caller's grow callback
    unsafe { give(&e.b, grow, ctx) };
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_put_header(tag: u32, grow: GrowFn, ctx: *mut c_void) {
    give(&Enc::new(tag).b, grow, ctx);
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_put_u64(x: u64, grow: GrowFn, ctx: *mut c_void) {
    put(grow, ctx, |e| e.u64(x));
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_put_notice(n: *const Notice, grow: GrowFn, ctx: *mut c_void) {
    put(grow, ctx, |e| e.notice(&*n));
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_put_key(k: *const Key, grow: GrowFn, ctx: *mut c_void) {
    put(grow, ctx, |e| e.key(&*k));
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_put_steering(s: *const Identity, grow: GrowFn, ctx: *mut c_void) {
    put(grow, ctx, |e| e.steering(&*s));
}
unsafe fn get<T>(b: *const u8, n: usize, i: *mut usize, out: *mut T, f: impl FnOnce(&mut Dec) -> R<T>) -> i32 {
    let mut d = Dec { b: bytes(b, n), i: *i, tag: 0 };
    let r = f(&mut d);
    *i = d.i;
    code(r.map(|v| *out = v))
}
/// The magic and the tag (the C++ Decoder's constructor).
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_header(b: *const u8, n: usize, i: *mut usize, tag: *mut u32) -> i32 {
    get(b, n, i, tag, |d| {
        let x = Dec::new(d.b)?;
        d.i = x.i;
        Ok(x.tag)
    })
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_u64(b: *const u8, n: usize, i: *mut usize, out: *mut u64) -> i32 {
    get(b, n, i, out, |d| d.u64())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_u32(b: *const u8, n: usize, i: *mut usize, out: *mut u32) -> i32 {
    get(b, n, i, out, |d| d.u32())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_notice(b: *const u8, n: usize, i: *mut usize, out: *mut Notice) -> i32 {
    get(b, n, i, out, |d| d.notice())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_key(b: *const u8, n: usize, i: *mut usize, out: *mut Key) -> i32 {
    get(b, n, i, out, |d| d.key())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_steering(b: *const u8, n: usize, i: *mut usize, out: *mut Identity) -> i32 {
    get(b, n, i, out, |d| d.steering())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_get_end(b: *const u8, n: usize, i: usize) -> i32 {
    code(Dec { b: bytes(b, n), i, tag: 0 }.end())
}

// --- links

/// Takes `fd`; null when it is negative.
#[no_mangle]
pub extern "C" fn g6ctl_io_new(fd: i32) -> *mut Io {
    Io::new(fd).map_or(std::ptr::null_mut(), |io| Box::into_raw(Box::new(io)))
}
/// Closes the link.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_free(io: *mut Io) {
    if !io.is_null() {
        drop(Box::from_raw(io));
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_shutdown(io: *const Io) {
    (*io).shutdown();
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_timeout(io: *const Io, seconds: u32) -> i32 {
    code((*io).timeout_seconds(seconds))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_keepalive(io: *const Io, idle_s: i32, intvl_s: i32, cnt: i32) -> i32 {
    code((*io).keepalive(idle_s, intvl_s, cnt))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_nodelay(io: *const Io) -> i32 {
    code((*io).nodelay())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_send(io: *const Io, b: *const u8, n: usize) -> i32 {
    code((*io).send(bytes(b, n)))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_receive(io: *const Io, idle: bool, grow: GrowFn, ctx: *mut c_void) -> i32 {
    code((*io).receive(idle).map(|b| give(&b, grow, ctx)))
}
/// Frames sent and received.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_io_counts(io: *const Io, sent: *mut u64, received: *mut u64) {
    (*sent, *received) = (*io).counts();
}

// --- bootstrap

unsafe fn boot<'a>(c: *const CBoot) -> Boot<'a> {
    let c = &*c;
    Boot {
        rank: c.rank,
        ranks: c.ranks,
        timeout: c.timeout,
        port: c.port,
        head_ip: bytes(c.head_ip, c.head_ip_n),
        directory: bytes(c.directory, c.directory_n),
        agreement: bytes(c.agreement, c.agreement_n),
    }
}
/// BootstrapConfig::validate's address and timeout checks.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_boot_check(ip: *const u8, ip_n: usize, port: u16, timeout: u32) -> i32 {
    code(bootstrap::check(bytes(ip, ip_n), port, timeout))
}
/// This rank's links: out has room for ranks - 1 (the head's, in rank order) or 1 (a worker's); *n is how many.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_bootstrap(c: *const CBoot, out: *mut *mut Io, n: *mut usize) -> i32 {
    *n = 0;
    code(bootstrap::bootstrap(&boot(c)).map(|v| {
        for (i, io) in v.into_iter().enumerate() {
            *out.add(i) = Box::into_raw(Box::new(io));
            *n = i + 1;
        }
    }))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agree_lines(
    rank: u32,
    l: *const *const Io,
    n: usize,
    mine: *const u8,
    mine_n: usize,
    idle: bool,
    grow: GrowFn,
    ctx: *mut c_void,
) -> i32 {
    code(bootstrap::agree_lines(rank, &links(l, n), bytes(mine, mine_n), idle).map(|b| give(&b, grow, ctx)))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_capture_barrier(c: *const CBoot, l: *const *const Io, n: usize) -> i32 {
    code(bootstrap::capture_barrier(&boot(c), &links(l, n)))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_difference(
    rank: u32,
    h: *const u8,
    hn: usize,
    w: *const u8,
    wn: usize,
    grow: GrowFn,
    ctx: *mut c_void,
) {
    give(&bootstrap::agreement_difference(rank, bytes(h, hn), bytes(w, wn)), grow, ctx);
}

// --- the gate and the ACK set, for the C++ classes of the same names

/// Null when refused (g6ctl_error).
#[no_mangle]
pub extern "C" fn g6ctl_gate_new(epoch: u64, rank: u32) -> *mut Gate {
    let r = Gate::new(epoch, rank);
    match r {
        Ok(g) => Box::into_raw(Box::new(g)),
        Err(e) => {
            code::<()>(Err(e));
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_free(g: *mut Gate) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_before_work(g: *const Gate, b: u64, due: *mut bool) -> i32 {
    code((*g).before_work(b).map(|d| *due = d))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_announce(g: *const Gate, n: *const Notice) -> i32 {
    code((*g).announce(&*n))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_arrive(g: *const Gate, b: u64, hash: u64, room: u32, out: *mut Ack) -> i32 {
    code((*g).arrive(b, hash, room).map(|a| *out = a))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_commit(g: *const Gate, n: *const Notice) -> i32 {
    code((*g).commit(&*n))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_wait_commit(g: *const Gate) -> i32 {
    code((*g).wait_commit())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_applied(g: *const Gate, n: *const Notice, hold: bool) -> i32 {
    code((*g).applied(&*n, hold))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_gate_fail(g: *const Gate) {
    (*g).fail();
}
/// Null when refused (g6ctl_error).
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_new(nr: u32, n: *const Notice) -> *mut Agreement {
    match Agreement::new(nr, *n) {
        Ok(a) => Box::into_raw(Box::new(a)),
        Err(e) => {
            code::<()>(Err(e));
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_free(a: *mut Agreement) {
    if !a.is_null() {
        drop(Box::from_raw(a));
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_ack(a: *mut Agreement, k: *const Ack) -> i32 {
    code((*a).ack(&*k))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_ready(a: *const Agreement) -> bool {
    (*a).ready()
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_hash(a: *const Agreement, out: *mut u64) -> i32 {
    code((*a).agreed_hash().map(|h| *out = h))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreement_room(a: *const Agreement, out: *mut u32) -> i32 {
    code((*a).room().map(|r| *out = r))
}

// --- the event channels

fn endpoint(r: R<Endpoint>) -> *mut Endpoint {
    match r {
        Ok(e) => Box::into_raw(Box::new(e)),
        Err(e) => {
            code::<()>(Err(e));
            std::ptr::null_mut()
        }
    }
}
/// The head over its n links (each must outlive it); null when refused (g6ctl_error).
#[no_mangle]
pub unsafe extern "C" fn g6ctl_head_new(epoch: u64, peers: *const *const Io, n: usize) -> *mut Endpoint {
    let ios: Vec<*const Io> = (0..n).map(|i| *peers.add(i)).collect();
    if ios.iter().any(|p| p.is_null()) {
        code::<()>(Err(refused("bad control fd")));
        return std::ptr::null_mut();
    }
    endpoint(Endpoint::head(epoch, ios.into_iter().map(Link::new).collect()))
}
/// Worker `rank` over its link to the head (which must outlive it); null when refused (g6ctl_error).
#[no_mangle]
pub unsafe extern "C" fn g6ctl_worker_new(epoch: u64, rank: u32, io: *const Io) -> *mut Endpoint {
    if io.is_null() {
        code::<()>(Err(refused("bad control fd")));
        return std::ptr::null_mut();
    }
    endpoint(Endpoint::worker(epoch, rank, Link::new(io)))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_endpoint_free(e: *mut Endpoint) {
    if !e.is_null() {
        drop(Box::from_raw(e));
    }
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_fail(e: *const Endpoint) {
    (*e).fail();
}
/// *due: an event's commit is in out.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_boundary(
    e: *const Endpoint,
    b: u64,
    digest: DigestFn,
    ctx: *mut c_void,
    due: *mut bool,
    out: *mut Commit,
) -> i32 {
    *due = false;
    let r = (*e).boundary(b, || {
        let (mut d, mut room) = (0u64, u32::MAX);
        hook(digest(ctx, &mut d, &mut room))?;
        Ok((d, room))
    });
    code(r.map(|c| {
        if let Some(c) = c {
            *out = c;
            *due = true;
        }
    }))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_wait_notice(e: *const Endpoint, out: *mut Notice) -> i32 {
    code((*e).wait_notice().map(|n| *out = n))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_stamps(e: *const Endpoint, installed: *mut i64, arrived: *mut i64) {
    (*installed, *arrived) = (*e).stamps();
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_applied(e: *const Endpoint) -> i32 {
    code((*e).applied())
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_sync(e: *const Endpoint, seq: u64) -> i32 {
    code((*e).sync(seq))
}
/// payload: an ADMIT's request bytes (has_payload), else none.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_event(
    e: *const Endpoint,
    n: *const Notice,
    make_commit: CommitFn,
    ctx: *mut c_void,
    has_payload: bool,
    payload: *const u8,
    payload_n: usize,
) -> i32 {
    let payload = has_payload.then(|| bytes(payload, payload_n));
    let make = |hash: u64| {
        let mut c = Commit::zero();
        hook(make_commit(ctx, hash, &mut c))?;
        Ok(c)
    };
    code((*e).event(&*n, make, payload))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_agreed_room(e: *const Endpoint) -> u32 {
    (*e).agreed_room()
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_quit(e: *const Endpoint) {
    (*e).quit();
}
/// on_quit: null for none.
#[no_mangle]
pub unsafe extern "C" fn g6ctl_receive_events(e: *const Endpoint, on_quit: Option<QuitFn>, ctx: *mut c_void) -> i32 {
    code((*e).receive_events(|| match on_quit {
        Some(f) => hook(f(ctx)),
        None => Ok(()),
    }))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_request(e: *const Endpoint, id: u64, grow: GrowFn, ctx: *mut c_void) -> i32 {
    code((*e).request(id).map(|b| give(&b, grow, ctx)))
}
#[no_mangle]
pub unsafe extern "C" fn g6ctl_admitted(e: *const Endpoint, id: u64) {
    (*e).admitted(id);
}
