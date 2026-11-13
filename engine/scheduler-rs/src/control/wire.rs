//! The control link's frames: big-endian 64-bit words after a header of the magic (the numeric base "G6S2E0" plus the
//! link's version) and a tag. Event metadata and request bytes only, never model tensors. A decoder reads the fields
//! in the encoder's order and refuses at the first one that is short, out of range or inconsistent, with the message
//! the C++ codec threw.

use super::{require, R};
use crate::kv_share::Resize;
use crate::scheduler::{self as sch, Grant, Identity, Key, Run, Shrink, PAGE};

/// The link's version: a frame of another version is refused, and the bootstrap agreement carries it, so ranks of
/// different builds refuse each other there (scheduler.h CONTROL_VERSION).
pub const CONTROL_VERSION: u64 = 15;
/// The first word of every frame.
pub const MAGIC: u64 = 0x473653324530 + CONTROL_VERSION;

/// Frame tags (control_protocol.h Tag).
pub const NOTICE: u32 = 1;
/// An event's ACK, worker to head.
pub const ACK: u32 = 2;
/// An event's commit, head to workers.
pub const COMMIT: u32 = 3;
/// A request's bytes, head to workers before its ADMIT notice.
pub const REQUEST: u32 = 4;
/// The head's orderly quit.
pub const QUIT: u32 = 5;
/// A worker's barrier arrival.
pub const SYNC: u32 = 6;
/// The head's barrier release.
pub const GO: u32 = 7;

/// Event kinds (event_gate.h EventKind).
pub const ADMIT: u32 = 1;
/// A session leaves.
pub const CANCEL: u32 = 2;
/// A tier changes size.
pub const RESIZE: u32 = 3;

/// The steering scopes a decoder accepts (steering.h Scope) and the identity's limits.
const SCOPE_SHARED: u8 = 1;
const SCOPE_FULL: u8 = 2;
const SCALE_MAX: f32 = 4.0;
const PROFILES_MAX: u32 = 8;

/// event_gate.h Notice: an event the head announces for a future boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Notice {
    pub epoch: u64,
    pub event: u64,
    pub boundary: u64,
    pub request: u64,
    pub payload_hash: u64,
    pub kind: u32,
}

/// event_gate.h Ack: a rank stopped at the event's boundary, its state digest and its Flash KV room.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ack {
    pub notice: Notice,
    pub rank: u32,
    pub state_hash: u64,
    pub room: u32,
}

/// scheduler.h Commit whole: the scheduler's part, the hold flag and a Resize event's change.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Commit {
    pub c: sch::Commit,
    pub hold: bool,
    pub has_resize: bool,
    pub resize: Resize,
}

/// steering.h Identity's defaults: off, scope Full.
pub fn identity_off() -> Identity {
    Identity {
        enabled: false,
        profile_digest: [0; 32],
        scope: SCOPE_FULL,
        schedule_digest: [0; 32],
        scale_bits: 0,
        slot: 0,
        images: false,
        image_digest: [0; 32],
    }
}

fn grant_zero() -> Grant {
    Grant {
        key: Key::default(),
        req: sch::Request { id: 0, prompt_tokens: 0, max_tokens: 0, steering: identity_off(), r_rows: 0, ckpt_at: 0 },
        run: Run::default(),
        reuse: 0,
        restored: 0,
        counted: 0,
    }
}

impl Commit {
    /// A commit with every field zero and no optional part.
    pub fn zero() -> Commit {
        Commit {
            c: sch::Commit {
                epoch: 0,
                event: 0,
                boundary: 0,
                before_hash: 0,
                has_cancel: false,
                cancel: Key::default(),
                has_admit: false,
                admit: grant_zero(),
                has_shrink: false,
                shrink: Shrink { key: Key::default(), r_rows: 0, run: Run::default() },
            },
            hold: false,
            has_resize: false,
            resize: Resize::default(),
        }
    }
}

/// A frame being written.
pub struct Enc {
    pub b: Vec<u8>,
}

impl Enc {
    /// The magic and the tag.
    pub fn new(tag: u32) -> Enc {
        let mut e = Enc { b: Vec::new() };
        e.u64(MAGIC);
        e.u64(u64::from(tag));
        e
    }
    /// One big-endian word.
    pub fn u64(&mut self, x: u64) {
        self.b.extend_from_slice(&x.to_be_bytes());
    }
    /// A notice's six words.
    pub fn notice(&mut self, n: &Notice) {
        for x in [n.epoch, n.event, n.boundary, n.request, n.payload_hash, u64::from(n.kind)] {
            self.u64(x);
        }
    }
    /// A key's three words.
    pub fn key(&mut self, k: &Key) {
        self.u64(k.epoch);
        self.u64(u64::from(k.slot));
        self.u64(u64::from(k.generation));
    }
    /// A steering identity: its flags, the image digest with images, the rest only when enabled. A digest's 32 bytes
    /// are four words, each its eight bytes in order.
    pub fn steering(&mut self, s: &Identity) {
        self.u64(u64::from(s.enabled) | if s.images { 2 } else { 0 });
        if s.images {
            self.b.extend_from_slice(&s.image_digest);
        }
        if !s.enabled {
            return;
        }
        self.b.extend_from_slice(&s.profile_digest);
        self.u64(u64::from(s.scope));
        self.b.extend_from_slice(&s.schedule_digest);
        self.u64(u64::from(s.scale_bits));
        self.u64(u64::from(s.slot));
    }
    fn run(&mut self, r: &Run) {
        self.u64(u64::from(r.first_page));
        self.u64(u64::from(r.pages));
        self.u64(u64::from(r.table_version));
        self.u64(r.list_hash);
    }
}

/// A frame being read.
pub struct Dec<'a> {
    pub b: &'a [u8],
    pub i: usize,
    pub tag: u32,
}

impl<'a> Dec<'a> {
    /// Reads the magic and the tag.
    pub fn new(b: &'a [u8]) -> R<Dec<'a>> {
        let mut d = Dec { b, i: 0, tag: 0 };
        let magic = d.u64()?;
        require(magic == MAGIC, "control version mismatch")?;
        let t = d.u64()?;
        require((1..=7).contains(&t), "unknown control tag")?;
        d.tag = t as u32;
        Ok(d)
    }
    /// One word.
    pub fn u64(&mut self) -> R<u64> {
        require(self.b.len().saturating_sub(self.i) >= 8, "truncated control metadata")?;
        let mut w = [0u8; 8];
        w.copy_from_slice(&self.b[self.i..self.i + 8]);
        self.i += 8;
        Ok(u64::from_be_bytes(w))
    }
    /// One word that must fit 32 bits.
    pub fn u32(&mut self) -> R<u32> {
        let x = self.u64()?;
        require(x <= u64::from(u32::MAX), "control integer overflow")?;
        Ok(x as u32)
    }
    /// A notice; its kind must be one of the three.
    pub fn notice(&mut self) -> R<Notice> {
        let mut n = Notice { epoch: self.u64()?, ..Notice::default() };
        n.event = self.u64()?;
        n.boundary = self.u64()?;
        n.request = self.u64()?;
        n.payload_hash = self.u64()?;
        let k = self.u64()?;
        require((1..=3).contains(&k), "unknown event kind")?;
        n.kind = k as u32;
        Ok(n)
    }
    /// A key.
    pub fn key(&mut self) -> R<Key> {
        let epoch = self.u64()?;
        let slot = self.u32()?;
        let generation = self.u32()?;
        Ok(Key { epoch, slot, generation })
    }
    fn digest(&mut self) -> R<[u8; 32]> {
        let mut d = [0u8; 32];
        for i in (0..32).step_by(8) {
            d[i..i + 8].copy_from_slice(&self.u64()?.to_be_bytes());
        }
        Ok(d)
    }
    /// A steering identity, checked as steering.h checks a loaded one.
    pub fn steering(&mut self) -> R<Identity> {
        let mut s = identity_off();
        let on = self.u64()?;
        require(on <= 3, "invalid steering enabled bits")?;
        s.enabled = on & 1 != 0;
        s.images = on & 2 != 0;
        if s.images {
            s.image_digest = self.digest()?;
        }
        if !s.enabled {
            return Ok(s);
        }
        s.profile_digest = self.digest()?;
        let scope = self.u64()?;
        require(scope == u64::from(SCOPE_SHARED) || scope == u64::from(SCOPE_FULL), "invalid steering scope")?;
        s.scope = scope as u8;
        s.schedule_digest = self.digest()?;
        s.scale_bits = self.u32()?;
        let scale = f32::from_bits(s.scale_bits);
        require(
            s.scale_bits != 0 && scale.is_finite() && scale > 0.0 && scale <= SCALE_MAX,
            "invalid steering scale bits",
        )?;
        s.slot = self.u32()?;
        require(s.slot < PROFILES_MAX, "invalid steering profile slot")?;
        Ok(s)
    }
    fn run(&mut self) -> R<Run> {
        let first_page = self.u32()?;
        let pages = self.u32()?;
        let table_version = self.u32()?;
        let list_hash = self.u64()?;
        Ok(Run { first_page, pages, table_version, list_hash })
    }
    /// Nothing may follow the last field.
    pub fn end(&self) -> R<()> {
        require(self.i == self.b.len(), "trailing control metadata")
    }
    fn expect(b: &'a [u8], tag: u32, what: &'static str) -> R<Dec<'a>> {
        let d = Dec::new(b)?;
        require(d.tag == tag, what)?;
        Ok(d)
    }
}

/// A commit agrees with its notice: same event, a grant only for its ADMIT, hold only with a grant, a resize exactly
/// for a RESIZE notice and alone, a shrink only with an admission.
pub fn check_match(n: &Notice, x: &Commit) -> R<()> {
    let c = &x.c;
    require(n.epoch == c.epoch && n.event == c.event && n.boundary == c.boundary, "commit/notice mismatch")?;
    if c.has_admit {
        require(n.kind == ADMIT && c.admit.req.id == n.request, "grant does not match announced request")?;
    }
    require(!x.hold || c.has_admit, "a held boundary needs a grant")?;
    require((n.kind == RESIZE) == x.has_resize, "a resize commit needs a RESIZE notice and back")?;
    if x.has_resize {
        require(!c.has_admit && !c.has_cancel && !x.hold && !c.has_shrink, "a resize commit changes nothing else")?;
    }
    if c.has_shrink {
        require(n.kind == ADMIT, "a shrink comes with an admission")?;
    }
    Ok(())
}

/// A NOTICE frame.
pub fn encode_notice(n: &Notice) -> Vec<u8> {
    let mut e = Enc::new(NOTICE);
    e.notice(n);
    e.b
}

/// An ACK frame.
pub fn encode_ack(a: &Ack) -> Vec<u8> {
    let mut e = Enc::new(ACK);
    e.notice(&a.notice);
    e.u64(u64::from(a.rank));
    e.u64(a.state_hash);
    e.u64(u64::from(a.room));
    e.b
}

/// A COMMIT frame; refused when the commit does not match its notice.
pub fn encode_commit(n: &Notice, x: &Commit) -> R<Vec<u8>> {
    check_match(n, x)?;
    let mut e = Enc::new(COMMIT);
    e.notice(n);
    let c = &x.c;
    e.u64(c.before_hash);
    e.u64(u64::from(c.has_cancel));
    if c.has_cancel {
        e.key(&c.cancel);
    }
    e.u64(u64::from(c.has_admit));
    if c.has_admit {
        let g = &c.admit;
        e.key(&g.key);
        e.u64(g.req.id);
        e.u64(u64::from(g.req.prompt_tokens));
        e.u64(u64::from(g.req.max_tokens));
        e.steering(&g.req.steering);
        e.run(&g.run);
        e.u64(u64::from(g.reuse));
        e.u64(u64::from(g.restored));
        e.u64(u64::from(g.counted));
        e.u64(u64::from(g.req.r_rows));
        e.u64(u64::from(g.req.ckpt_at));
    }
    e.u64(u64::from(x.hold));
    e.u64(u64::from(x.has_resize));
    if x.has_resize {
        let r = &x.resize;
        e.u64(u64::from(r.tier));
        e.u64(r.segs);
        e.u64(r.slots);
        e.u64(r.txn);
    }
    e.u64(u64::from(c.has_shrink));
    if c.has_shrink {
        let k = &c.shrink;
        e.key(&k.key);
        e.u64(u64::from(k.r_rows));
        e.run(&k.run);
    }
    Ok(e.b)
}

/// A REQUEST frame: the request's bytes after its length.
pub fn encode_request(epoch: u64, request: u64, bytes: &[u8]) -> Vec<u8> {
    let mut e = Enc::new(REQUEST);
    e.u64(epoch);
    e.u64(request);
    e.u64(bytes.len() as u64);
    e.b.extend_from_slice(bytes);
    e.b
}

/// A QUIT frame.
pub fn encode_quit(epoch: u64) -> Vec<u8> {
    let mut e = Enc::new(QUIT);
    e.u64(epoch);
    e.b
}

/// A SYNC frame: a worker reached barrier `seq`.
pub fn encode_sync(epoch: u64, rank: u32, seq: u64) -> Vec<u8> {
    let mut e = Enc::new(SYNC);
    e.u64(epoch);
    e.u64(u64::from(rank));
    e.u64(seq);
    e.b
}

/// A GO frame: every rank reached barrier `seq`.
pub fn encode_go(epoch: u64, seq: u64) -> Vec<u8> {
    let mut e = Enc::new(GO);
    e.u64(epoch);
    e.u64(seq);
    e.b
}

/// A NOTICE frame's notice.
pub fn decode_notice(b: &[u8]) -> R<Notice> {
    let mut d = Dec::expect(b, NOTICE, "expected notice")?;
    let n = d.notice()?;
    d.end()?;
    Ok(n)
}

/// An ACK frame's ACK.
pub fn decode_ack(b: &[u8]) -> R<Ack> {
    let mut d = Dec::expect(b, ACK, "expected ACK")?;
    let notice = d.notice()?;
    let rank = d.u32()?;
    let state_hash = d.u64()?;
    let room = d.u32()?;
    d.end()?;
    Ok(Ack { notice, rank, state_hash, room })
}

/// A COMMIT frame's notice and commit, refused as the C++ refused it.
pub fn decode_commit(b: &[u8]) -> R<(Notice, Commit)> {
    let mut d = Dec::expect(b, COMMIT, "expected commit")?;
    let n = d.notice()?;
    let mut x = Commit::zero();
    let c = &mut x.c;
    c.epoch = n.epoch;
    c.event = n.event;
    c.boundary = n.boundary;
    c.before_hash = d.u64()?;
    let has = d.u64()?;
    require(has <= 1, "invalid optional cancel")?;
    if has != 0 {
        c.cancel = d.key()?;
        c.has_cancel = true;
    }
    let has = d.u64()?;
    require(has <= 1, "invalid optional grant")?;
    if has != 0 {
        let g = &mut c.admit;
        g.key = d.key()?;
        g.req.id = d.u64()?;
        g.req.prompt_tokens = d.u32()?;
        g.req.max_tokens = d.u32()?;
        g.req.steering = d.steering()?;
        g.run = d.run()?;
        g.reuse = d.u32()?;
        g.restored = d.u32()?;
        g.counted = d.u32()?;
        g.req.r_rows = d.u32()?;
        g.req.ckpt_at = d.u32()?;
        require(g.req.ckpt_at == 0 || g.req.ckpt_at < g.req.prompt_tokens, "invalid checkpoint boundary")?;
        require(g.restored == 0 || g.reuse != 0, "invalid restored grant")?;
        require(g.req.r_rows.is_multiple_of(PAGE), "invalid grant R")?;
        c.has_admit = true;
    }
    let has = d.u64()?;
    require(has <= 1, "invalid hold flag")?;
    x.hold = has != 0;
    let has = d.u64()?;
    require(has <= 1, "invalid optional resize")?;
    if has != 0 {
        x.resize.tier = d.u32()?;
        x.resize.segs = d.u64()?;
        x.resize.slots = d.u64()?;
        x.resize.txn = d.u64()?;
        x.has_resize = true;
    }
    let has = d.u64()?;
    require(has <= 1, "invalid optional shrink")?;
    if has != 0 {
        let k = &mut x.c.shrink;
        k.key = d.key()?;
        k.r_rows = d.u32()?;
        k.run = d.run()?;
        require(k.r_rows != 0 && k.r_rows.is_multiple_of(PAGE), "invalid shrink R")?;
        x.c.has_shrink = true;
    }
    check_match(&n, &x)?;
    d.end()?;
    Ok((n, x))
}

/// A REQUEST frame's epoch, request and bytes (borrowed from `b`).
pub fn decode_request(b: &[u8]) -> R<(u64, u64, &[u8])> {
    let mut d = Dec::expect(b, REQUEST, "expected request")?;
    let epoch = d.u64()?;
    let request = d.u64()?;
    let n = d.u64()?;
    require(n == (b.len() - d.i) as u64, "request length mismatch")?;
    let bytes = &b[d.i..];
    d.i = b.len();
    d.end()?;
    Ok((epoch, request, bytes))
}

/// A QUIT frame's epoch.
pub fn decode_quit(b: &[u8]) -> R<u64> {
    let mut d = Dec::expect(b, QUIT, "expected quit")?;
    let e = d.u64()?;
    d.end()?;
    Ok(e)
}

/// A SYNC frame's epoch, rank and barrier.
pub fn decode_sync(b: &[u8]) -> R<(u64, u32, u64)> {
    let mut d = Dec::expect(b, SYNC, "expected sync")?;
    let epoch = d.u64()?;
    let rank = d.u32()?;
    let seq = d.u64()?;
    d.end()?;
    Ok((epoch, rank, seq))
}

/// A GO frame's epoch and barrier.
pub fn decode_go(b: &[u8]) -> R<(u64, u64)> {
    let mut d = Dec::expect(b, GO, "expected go")?;
    let epoch = d.u64()?;
    let seq = d.u64()?;
    d.end()?;
    Ok((epoch, seq))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(r: R<impl std::fmt::Debug>) -> String {
        String::from_utf8_lossy(r.unwrap_err().message()).into_owned()
    }

    #[test]
    fn notice_round_trip_and_refusals() {
        let n = Notice { epoch: 9, event: 1, boundary: 5, request: 11, payload_hash: 7, kind: ADMIT };
        let b = encode_notice(&n);
        assert_eq!(b.len(), 8 * 8);
        assert_eq!(&b[..8], &MAGIC.to_be_bytes());
        assert_eq!(decode_notice(&b).unwrap(), n);
        let mut old = b.clone();
        old[7] -= 1;
        assert_eq!(msg(decode_notice(&old)), "control version mismatch");
        assert_eq!(msg(decode_ack(&b)), "expected ACK");
        assert_eq!(msg(decode_notice(&b[..20])), "truncated control metadata");
        let mut more = b.clone();
        more.push(0);
        assert_eq!(msg(decode_notice(&more)), "trailing control metadata");
    }

    #[test]
    fn commit_checks_its_notice() {
        let n = Notice { epoch: 3, event: 1, boundary: 5, request: 11, payload_hash: 0, kind: ADMIT };
        let mut x = Commit::zero();
        x.c.epoch = 3;
        x.c.event = 1;
        x.c.boundary = 5;
        x.hold = true;
        assert_eq!(msg(encode_commit(&n, &x)), "a held boundary needs a grant");
        x.c.has_admit = true;
        x.c.admit.req.id = 11;
        x.c.admit.req.r_rows = PAGE;
        let b = encode_commit(&n, &x).unwrap();
        let (m, y) = decode_commit(&b).unwrap();
        assert_eq!(m, n);
        assert!(y.hold && y.c.has_admit && y.c.admit.req.id == 11);
        assert_eq!(y.c.admit.req.steering, identity_off());
    }
}
