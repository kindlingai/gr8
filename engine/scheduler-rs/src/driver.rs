//! The S2 driver (s2/driver.h `g6s2::Driver` is its C++ facade): it applies commits, grows runs before a round, picks
//! the capped rounds past R, runs one scheduler quantum on the planner and reports completions. It owns the per-slot
//! emitted counts, the initialised flags, what it reads of each payload and the capped-round cycle. The scheduler's
//! state is still held by the C++ `Scheduler` and passed in per call. The planner is reached through the C++
//! `s2::Batch`, which writes the round log.
//!
//! Hooks may read the driver back (the service's retire reads a slot's emitted tokens), so every field is a `Cell`
//! and every method takes `&self`; apply and step refuse to be entered again from a hook.
#![allow(clippy::missing_safety_doc)]

use crate::cp_rows::{self, Cand};
use crate::scheduler::{
    self as sch, Commit, Completion, Config, Grant, Key, Quantum, Run, Shrink, Slot, State, DECODE, FREE, K_DECODE,
    K_PREFILL, PARKED, PREFILL, PREFILL_PACK, PROMPT_TAIL, PROMPT_TAIL_PACK, QMAX, RETIRING,
};
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void};
use std::io::Write;

/// g6s2_drv_sched: the scheduler's state, which the C++ `Scheduler` holds
#[repr(C)]
pub struct CSched {
    pub c: *const Config,
    pub s: *mut Slot,
    pub n: usize,
    pub st: *mut State,
}

/// g6s2_drv_payload: what the driver reads of an admitted request's payload. steering_ok: its steering equals the
/// grant's (the C++ compares them)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPayload {
    pub prompt_len: usize,
    pub chunk: i32,
    pub decode_prompt: bool,
    pub steering_ok: bool,
}

/// bind(grant)
pub type BindFn = unsafe extern "C" fn(*mut c_void, *const Grant) -> i32;
/// retire or finished(slot)
pub type SlotFn = unsafe extern "C" fn(*mut c_void, *const Slot) -> i32;
/// prefill(slot, first, rows)
pub type PrefillFn = unsafe extern "C" fn(*mut c_void, *const Slot, u32, u32) -> i32;
/// prefill_pack(slots, firsts, rows, n)
pub type PrefillPackFn = unsafe extern "C" fn(*mut c_void, *const u32, *const u32, *const u32, u32) -> i32;
/// dfl_ok(key, *ok)
pub type KeyBoolFn = unsafe extern "C" fn(*mut c_void, *const Key, *mut bool) -> i32;
/// terminal(key)
pub type KeyFn = unsafe extern "C" fn(*mut c_void, *const Key) -> i32;
/// grow(slot, rows, *run, *some)
pub type GrowFn = unsafe extern "C" fn(*mut c_void, *const Slot, u64, *mut Run, *mut bool) -> i32;
/// resize, settle or demote()
pub type VoidFn = unsafe extern "C" fn(*mut c_void) -> i32;
/// shrink(shrink, run before, length)
pub type ShrinkFn = unsafe extern "C" fn(*mut c_void, *const Shrink, *const Run, u64) -> i32;
/// payload(slot, set)
pub type PayloadFn = unsafe extern "C" fn(*mut c_void, u32, bool) -> i32;

/// g6s2_drv_hooks: the engine's callbacks (s2/driver.h DriverHooks less the planner's). Each returns 0, or nonzero
/// when the C++ hook threw. has_*: the C++ member is set. payload: the C++ copies apply's payload into the slot (set)
/// or clears it
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CHooks {
    pub ctx: *mut c_void,
    pub bind: Option<BindFn>,
    pub retire: Option<SlotFn>,
    pub prefill: Option<PrefillFn>,
    pub prefill_pack: Option<PrefillPackFn>,
    pub dfl_ok: Option<KeyBoolFn>,
    pub terminal: Option<KeyFn>,
    pub grow: Option<GrowFn>,
    pub finished: Option<SlotFn>,
    pub resize: Option<VoidFn>,
    pub shrink: Option<ShrinkFn>,
    pub settle: Option<VoidFn>,
    pub payload: Option<PayloadFn>,
    pub has_prefill_pack: bool,
    pub has_dfl_ok: bool,
    pub has_terminal: bool,
    pub has_grow: bool,
    pub has_finished: bool,
    pub has_resize: bool,
    pub has_shrink: bool,
    pub has_settle: bool,
}

/// admit(slot, first, n_gen, dfl)
pub type AdmitFn = unsafe extern "C" fn(*mut c_void, u32, i32, i32, bool) -> i32;
/// admit_pack(slots, firsts, n_gens, n)
pub type AdmitPackFn = unsafe extern "C" fn(*mut c_void, *const u32, *const i32, *const i32, u32) -> i32;
/// round(order, n, *ran)
pub type RoundFn = unsafe extern "C" fn(*mut c_void, *const i32, u32, *mut bool) -> i32;
/// promote or release(q)
pub type QFn = unsafe extern "C" fn(*mut c_void, i32) -> i32;
/// length or inserted(q, *out)
pub type QIntFn = unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> i32;
/// finished or active(q, *out)
pub type QBoolFn = unsafe extern "C" fn(*mut c_void, i32, *mut bool) -> i32;

/// g6s2_drv_planner: the planner's calls, through the C++ s2::Batch (which logs them). admit builds the session from
/// the slot's payload; inserted: Stats::inserted
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPlanner {
    pub ctx: *mut c_void,
    pub admit: Option<AdmitFn>,
    pub admit_pack: Option<AdmitPackFn>,
    pub round: Option<RoundFn>,
    pub demote: Option<VoidFn>,
    pub promote: Option<QFn>,
    pub release: Option<QFn>,
    pub length: Option<QIntFn>,
    pub finished: Option<QBoolFn>,
    pub active: Option<QBoolFn>,
    pub inserted: Option<QIntFn>,
}

/// why a call stopped: a hook threw (the facade holds the exception), or a refusal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Hook,
    Refused(&'static str),
}
type Res<T> = Result<T, Stop>;

fn require(ok: bool, what: &'static str) -> Res<()> {
    if ok {
        Ok(())
    } else {
        Err(Stop::Refused(what))
    }
}
fn ok(rc: i32) -> Res<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Stop::Hook)
    }
}
fn missing() -> Stop {
    Stop::Refused("s2: a driver callback is not wired")
}
fn sch_err(e: sch::Err) -> Stop {
    Stop::Refused(e)
}

/// One driver: its hooks and the per-slot state it keeps between calls.
pub struct Driver {
    h: CHooks,
    p: CPlanner,
    emitted: [Cell<u32>; QMAX],
    initialized: [Cell<bool>; QMAX],
    prompt_len: [Cell<usize>; QMAX],
    decode_prompt: [Cell<bool>; QMAX],
    served: RefCell<Vec<u64>>, // GR8_CP_STEP_ROWS: ids served in the current cycle
    cp_capped: Cell<bool>,
    cp_rounds: Cell<u64>,
    active_decode: Cell<bool>, // the quantum in flight is a decode round
    busy: Cell<bool>,          // in apply or step
    err: Cell<&'static str>,
}

/// the scheduler's state for one call; slices are formed only around scheduler calls, since hooks read the slots
pub(crate) struct S {
    c: *const Config,
    s: *mut Slot,
    n: usize,
    st: *mut State,
}
impl S {
    fn c(&self) -> &Config {
        // SAFETY: the facade's Config outlives the call and nothing writes it
        unsafe { &*self.c }
    }
    fn slot(&self, i: usize) -> Slot {
        assert!(i < self.n);
        // SAFETY: in range; a copy, so no reference outlives a hook
        unsafe { *self.s.add(i) }
    }
    fn slots(&self) -> &[Slot] {
        // SAFETY: the facade's array of n slots; read-only while hooks may read it too
        unsafe { std::slice::from_raw_parts(self.s, self.n) }
    }
    fn with<T>(&self, f: impl FnOnce(&Config, &mut [Slot], &mut State) -> T) -> T {
        // SAFETY: no hook runs inside f, so nothing else touches the slots or the state meanwhile
        unsafe { f(&*self.c, std::slice::from_raw_parts_mut(self.s, self.n), &mut *self.st) }
    }
    fn fail(&self) {
        // SAFETY: as with()
        unsafe { (*self.st).failed = true }
    }
}

/// the decoding sessions in ascending request id (scheduler.rs decoding, without its allocation)
fn decoding(s: &[Slot], out: &mut [Key; QMAX]) -> usize {
    let mut n = 0;
    for x in s.iter().filter(|x| x.phase == DECODE) {
        let mut i = n;
        while i > 0 && s[out[i - 1].slot as usize].req.id > x.req.id {
            out[i] = out[i - 1];
            i -= 1;
        }
        out[i] = x.key;
        n += 1;
    }
    n
}

/// a fixed buffer for one log line, written to stderr in one call
struct Line {
    b: [u8; 256],
    n: usize,
}
impl std::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let m = s.len().min(self.b.len() - self.n);
        self.b[self.n..self.n + m].copy_from_slice(&s.as_bytes()[..m]);
        self.n += m;
        Ok(())
    }
}

impl Driver {
    /// A driver over these hooks; None when a required one (bind, retire, prefill, payload) is missing.
    pub fn new(h: CHooks, p: CPlanner) -> Option<Driver> {
        if h.bind.is_none() || h.retire.is_none() || h.prefill.is_none() || h.payload.is_none() {
            return None;
        }
        Some(Driver {
            h,
            p,
            emitted: Default::default(),
            initialized: Default::default(),
            prompt_len: Default::default(),
            decode_prompt: Default::default(),
            served: RefCell::new(Vec::with_capacity(QMAX)),
            cp_capped: Cell::new(false),
            cp_rounds: Cell::new(0),
            active_decode: Cell::new(false),
            busy: Cell::new(false),
            err: Cell::new(""),
        })
    }

    fn enter(&self) -> Res<Busy<'_>> {
        require(!self.busy.get(), "s2: a driver hook called the driver back")?;
        self.busy.set(true);
        Ok(Busy(&self.busy))
    }

    // ---- the planner, through the facade
    fn length(&self, q: u32) -> Res<i32> {
        let mut v = 0;
        // SAFETY: the facade's trampoline, ctx its own
        ok(unsafe { self.p.length.ok_or_else(missing)?(self.p.ctx, q as i32, &mut v) })?;
        Ok(v)
    }
    fn q_bool(&self, f: Option<QBoolFn>, q: u32) -> Res<bool> {
        let mut v = false;
        // SAFETY: as length
        ok(unsafe { f.ok_or_else(missing)?(self.p.ctx, q as i32, &mut v) })?;
        Ok(v)
    }
    fn q_call(&self, f: Option<QFn>, q: u32) -> Res<()> {
        // SAFETY: as length
        ok(unsafe { f.ok_or_else(missing)?(self.p.ctx, q as i32) })
    }
    fn completion(&self, s: &S, k: Key) -> Res<Completion> {
        let i = k.slot as usize;
        let emitted = self.emitted[i].get().wrapping_sub(s.slot(i).emitted);
        let finished = self.q_bool(self.p.finished, k.slot)?;
        let mut inserted = 0;
        // SAFETY: as length
        ok(unsafe { self.p.inserted.ok_or_else(missing)?(self.p.ctx, k.slot as i32, &mut inserted) })?;
        Ok(Completion { key: k, emitted, finished, inserted: inserted as u32 })
    }

    // ---- the engine's hooks
    fn payload(&self, slot: u32, set: bool) -> Res<()> {
        // SAFETY: the facade's trampoline, ctx its own
        ok(unsafe { self.h.payload.ok_or_else(missing)?(self.h.ctx, slot, set) })
    }

    /// sessions in Prefill or Decode (Dispatch::synchronize_mode's count: 1 = the solo path)
    fn live(s: &S) -> u32 {
        s.slots().iter().filter(|x| x.phase == PREFILL || x.phase == DECODE).count() as u32
    }

    /// Driver::apply: validate the payload, apply the commit to the scheduler, then the hooks it implies. `digest` is
    /// the scheduler's digest before it. Refusals before the scheduler changes leave it as it was; a failure after it
    /// fails the scheduler.
    pub(crate) fn apply(
        &self,
        s: &S,
        e: &Commit,
        digest: u64,
        resize: bool,
        pay: Option<&CPayload>,
    ) -> Res<sch::Changes> {
        let _b = self.enter()?;
        if e.has_admit {
            let r = &e.admit.req;
            let p = pay.filter(|p| p.prompt_len == r.prompt_tokens as usize && p.chunk > 0 && p.steering_ok);
            let p = p.ok_or(Stop::Refused("admit payload mismatch"))?;
            require(r.max_tokens <= i32::MAX as u32 && p.prompt_len <= i32::MAX as usize, "Batch integer range")?;
        } else {
            require(pay.is_none(), "payload without grant")?;
        }
        let mut before = [s.slot(0); QMAX];
        before[..s.n].copy_from_slice(s.slots());
        let changes = s.with(|c, sl, st| sch::apply(c, sl, st, e, digest)).map_err(sch_err)?;
        if let Err(r) = self.apply_hooks(s, e, resize, pay, &before[..s.n]) {
            s.fail();
            return Err(r);
        }
        Ok(changes)
    }

    fn apply_hooks(&self, s: &S, e: &Commit, resize: bool, pay: Option<&CPayload>, before: &[Slot]) -> Res<()> {
        for old in before {
            let i = old.key.slot as usize;
            let now = s.slot(i);
            // park (Config::park): a finished session that parks leaves like a freed one, and a parked slot's session
            // already left when it parked
            if old.phase == FREE
                || old.phase == PARKED
                || (now.phase != FREE && now.phase != PARKED && old.key == now.key)
            {
                continue;
            }
            // SAFETY: the facade's trampoline; old outlives the call
            ok(unsafe { self.h.retire.ok_or_else(missing)?(self.h.ctx, old) })?;
            if self.initialized[i].get() {
                self.q_call(self.p.release, old.key.slot)?;
            }
            self.initialized[i].set(false);
            self.payload(old.key.slot, false)?;
            self.prompt_len[i].set(0);
            self.decode_prompt[i].set(false);
            self.emitted[i].set(0);
        }
        if e.has_shrink {
            require(self.h.has_shrink, "a shrink commit without the shrink hook")?;
            let k = e.shrink.key.slot;
            let len = self.length(k)?;
            // SAFETY: as retire
            ok(unsafe {
                self.h.shrink.ok_or_else(missing)?(self.h.ctx, &e.shrink, &before[k as usize].run, len as u64)
            })?;
        }
        if e.has_admit {
            let (g, p) = (&e.admit, pay.expect("validated"));
            let i = g.key.slot as usize;
            self.payload(g.key.slot, true)?;
            self.prompt_len[i].set(p.prompt_len);
            self.decode_prompt[i].set(p.decode_prompt);
            // SAFETY: as retire
            ok(unsafe { self.h.bind.ok_or_else(missing)?(self.h.ctx, g) })?;
            if p.decode_prompt {
                s.with(|_, sl, st| sch::skip_prefill(sl, st, &g.key)).map_err(sch_err)?;
            }
        }
        if resize {
            require(self.h.has_resize, "a resize commit without the resize hook")?;
            // SAFETY: as retire
            ok(unsafe { self.h.resize.ok_or_else(missing)?(self.h.ctx) })?;
        }
        Ok(())
    }

    /// Driver::step: grow, pick the capped round, take the scheduler's next quantum and run it.
    pub(crate) fn step(&self, s: &S, out: &mut Quantum) -> Res<()> {
        let _b = self.enter()?;
        let mut stalled = [false; QMAX];
        if let Err(e) = self.grow(s, &mut stalled) {
            s.fail();
            return Err(e);
        }
        // GR8_CP_STEP_ROWS: of the sessions that can decode, the ones this round takes (cp_rows.rs); the rest sit it out
        // like a grow stall. A session's verify context is at most its length + mtp, so `past` holds for every session
        // whose main step can run past R. The pick is recorded only if the quantum is a decode round
        let cc = *s.c();
        let mut cand = [Cand { id: 0, past: false }; QMAX];
        let mut cand_slot = [0u32; QMAX];
        let mut in_round = [true; QMAX];
        let (mut nc, mut capped, mut fresh) = (0, false, false);
        if cc.cp_step_rows != 0 {
            let mut keys = [Key::default(); QMAX];
            let nd = decoding(s.slots(), &mut keys);
            for k in &keys[..nd] {
                if !stalled[k.slot as usize] {
                    let ctx = (self.length(k.slot)? as u64).wrapping_add(u64::from(cc.mtp));
                    cand[nc] = Cand { id: s.slot(k.slot as usize).req.id, past: ctx > cc.cp_after };
                    cand_slot[nc] = k.slot;
                    nc += 1;
                }
            }
            let served = self.served.borrow();
            (capped, fresh) = cp_rows::pick_into(
                &cand[..nc],
                &served,
                cc.cp_step_rows,
                cc.mtp.wrapping_add(1),
                cc.cp_seg,
                &mut in_round[..nc],
            );
            for i in 0..nc {
                if !in_round[i] {
                    stalled[cand_slot[i] as usize] = true;
                }
            }
        }
        let q = s.with(|c, sl, st| sch::next(c, sl, st, &stalled[..sl.len()])).map_err(sch_err)?;
        if cc.cp_step_rows != 0 && q.kind == K_DECODE {
            let mut served = self.served.borrow_mut();
            if !capped || fresh {
                served.clear();
            }
            if capped {
                served.extend((0..nc).filter(|&i| in_round[i]).map(|i| cand[i].id));
            }
            if capped != self.cp_capped.get() {
                // one line per change, every rank (the fleet check reads it)
                let past = cand[..nc].iter().filter(|c| c.past).count();
                let inr = in_round[..nc].iter().filter(|&&x| x).count();
                let mut l = Line { b: [0; 256], n: 0 };
                let _ = std::fmt::Write::write_fmt(
                    &mut l,
                    format_args!(
                        "# s2 GR8_CP_STEP_ROWS={}: rounds {} at boundary {} ({} decoding, {} past R, {} in this round)\n",
                        cc.cp_step_rows,
                        if capped { "capped" } else { "uncapped" },
                        q.boundary,
                        nc,
                        past,
                        inr
                    ),
                );
                let _ = std::io::stderr().write_all(&l.b[..l.n]);
                self.cp_capped.set(capped);
            }
            self.cp_rounds.set(self.cp_rounds.get() + u64::from(capped));
        }
        *out = q;
        let r = self.run(s, &q);
        if r.is_err() {
            s.fail();
        }
        r
    }

    fn grow(&self, s: &S, stalled: &mut [bool; QMAX]) -> Res<()> {
        if self.h.has_grow {
            let mut keys = [Key::default(); QMAX];
            let nd = decoding(s.slots(), &mut keys);
            for k in &keys[..nd] {
                let x = s.slot(k.slot as usize);
                let rows = sch::round_rows(s.c(), self.length(k.slot)? as u64);
                let (mut r, mut some) = (Run::default(), false);
                // SAFETY: the facade's trampoline; x outlives the call
                ok(unsafe { self.h.grow.ok_or_else(missing)?(self.h.ctx, &x, rows, &mut r, &mut some) })?;
                if !some {
                    stalled[k.slot as usize] = true;
                } else if r != x.run {
                    s.with(|c, sl, st| sch::grow(c, sl, st, k, &r)).map_err(sch_err)?;
                }
            }
        }
        if self.h.has_settle {
            // SAFETY: as grow
            ok(unsafe { self.h.settle.ok_or_else(missing)?(self.h.ctx) })?;
        }
        Ok(())
    }

    fn run(&self, s: &S, q: &Quantum) -> Res<()> {
        self.active_decode.set(q.kind == K_DECODE);
        let count = q.count as usize;
        let mut results = [Completion { key: Key::default(), emitted: 0, finished: false, inserted: 0 }; QMAX];
        match q.kind {
            K_PREFILL => {
                let x = s.slot(q.keys[0].slot as usize);
                // SAFETY: the facade's trampoline, which reads the slot's payload; x outlives the call
                ok(unsafe { self.h.prefill.ok_or_else(missing)?(self.h.ctx, &x, q.first, q.rows) })?;
                s.with(sch::complete_prefill).map_err(sch_err)?;
            }
            PREFILL_PACK => {
                require(self.h.has_prefill_pack, "prefill pack native hook not wired")?;
                let mut slots = [0u32; QMAX];
                for (x, k) in slots.iter_mut().zip(&q.keys[..count]) {
                    *x = k.slot;
                }
                let f = self.h.prefill_pack.ok_or_else(missing)?;
                // SAFETY: as prefill; the arrays hold count entries
                ok(unsafe { f(self.h.ctx, slots.as_ptr(), q.firsts.as_ptr(), q.row_counts.as_ptr(), q.count) })?;
                s.with(sch::complete_prefill_pack).map_err(sch_err)?;
            }
            PROMPT_TAIL => {
                let k = q.keys[0];
                let i = k.slot as usize;
                require(!self.initialized[i].get(), "duplicate prompt tail")?;
                let dp = self.decode_prompt[i].get();
                let mut dfl = false;
                if self.h.has_dfl_ok && Self::live(s) == 1 && !dp {
                    // SAFETY: as prefill
                    ok(unsafe { self.h.dfl_ok.ok_or_else(missing)?(self.h.ctx, &k, &mut dfl) })?;
                }
                let first = if dp { 0 } else { q.first as i32 };
                let n_gen = s.slot(i).req.max_tokens as i32;
                // SAFETY: the facade's trampoline
                ok(unsafe { self.p.admit.ok_or_else(missing)?(self.p.ctx, k.slot, first, n_gen, dfl) })?;
                self.initialized[i].set(true);
                let r = self.completion(s, k)?;
                s.with(|c, sl, st| sch::complete_tail(c, sl, st, &r)).map_err(sch_err)?;
            }
            PROMPT_TAIL_PACK => {
                let (mut slots, mut firsts, mut n_gens) = ([0u32; QMAX], [0i32; QMAX], [0i32; QMAX]);
                for i in 0..count {
                    let k = q.keys[i];
                    let j = k.slot as usize;
                    let x = s.slot(j);
                    let plen = self.prompt_len[j].get();
                    require(
                        !self.initialized[j].get()
                            && x.pack_tail
                            && !self.decode_prompt[j].get()
                            && plen >= 2
                            && q.row_counts[i] == 1
                            && q.firsts[i] as usize == plen - 1,
                        "packed prompt tail membership invalid",
                    )?;
                    slots[i] = k.slot;
                    firsts[i] = q.firsts[i] as i32;
                    n_gens[i] = x.req.max_tokens as i32;
                }
                // one MAIN, one existing DSpark DRAFT if K>0, then batched CHAIN
                let f = self.p.admit_pack.ok_or_else(missing)?;
                // SAFETY: the facade's trampoline; the arrays hold count entries
                ok(unsafe { f(self.p.ctx, slots.as_ptr(), firsts.as_ptr(), n_gens.as_ptr(), q.count) })?;
                for (r, &k) in results.iter_mut().zip(&q.keys[..count]) {
                    self.initialized[k.slot as usize].set(true);
                    *r = self.completion(s, k)?;
                }
                s.with(|c, sl, st| sch::complete_tail_pack(c, sl, st, &results[..count])).map_err(sch_err)?;
            }
            K_DECODE => {
                let mut order = [0i32; QMAX];
                for (o, key) in order.iter_mut().zip(&q.keys[..count]) {
                    let k = key.slot;
                    let on = self.initialized[k as usize].get() && self.q_bool(self.p.active, k)?;
                    require(on, "scheduler/Batch active mismatch")?;
                    *o = k as i32;
                }
                if Self::live(s) > 1 {
                    // another session is resident (perhaps still prefilling): the shared path
                    // SAFETY: the facade's trampoline
                    ok(unsafe { self.p.demote.ok_or_else(missing)?(self.p.ctx) })?;
                } else if count == 1 && !self.decode_prompt[q.keys[0].slot as usize].get() {
                    self.q_call(self.p.promote, q.keys[0].slot)?; // GR8_DFL_REUSE=2
                }
                let mut ran = false;
                // SAFETY: the facade's trampoline; order holds count entries
                ok(unsafe { self.p.round.ok_or_else(missing)?(self.p.ctx, order.as_ptr(), q.count, &mut ran) })?;
                require(ran, "empty scheduled decode round")?;
                for (r, &k) in results.iter_mut().zip(&q.keys[..count]) {
                    *r = self.completion(s, k)?;
                }
                s.with(|c, sl, st| sch::complete_decode(c, sl, st, &results[..count])).map_err(sch_err)?;
            }
            _ => {}
        }
        if q.kind == PROMPT_TAIL || q.kind == PROMPT_TAIL_PACK || q.kind == K_DECODE {
            for k in &q.keys[..count] {
                let x = s.slot(k.slot as usize);
                if x.phase != RETIRING {
                    continue;
                }
                if self.h.has_terminal {
                    // SAFETY: as prefill
                    ok(unsafe { self.h.terminal.ok_or_else(missing)?(self.h.ctx, k) })?;
                }
                if self.h.has_finished {
                    // SAFETY: as prefill
                    ok(unsafe { self.h.finished.ok_or_else(missing)?(self.h.ctx, &x) })?;
                }
            }
        }
        Ok(())
    }

    /// The emit hook's accounting: `count` of the `n` tokens offered went out, `finished` the emitter's stop. Returns
    /// whether the session is finished (its stop, or max_tokens emitted).
    pub fn emit(&self, q: usize, n: i32, count: u32, finished: bool, max_tokens: u32) -> Res<bool> {
        let e = self.emitted[q].get();
        require(count <= n as u32 && count <= max_tokens.wrapping_sub(e), "bad emitter count")?;
        require(finished || count == n as u32, "partial emission without stop")?;
        self.emitted[q].set(e.wrapping_add(count));
        Ok(finished || e.wrapping_add(count) == max_tokens)
    }
}

/// resets the busy flag when apply or step returns
struct Busy<'a>(&'a Cell<bool>);
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

// ---- the C entry points (g6sched.h g6s2_driver_*). Safety, for each: `d` is a handle from g6s2_driver_new not yet
// freed, and pointers are valid for the counts passed with them.

fn rc(d: &Driver, r: Res<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(Stop::Hook) => 1,
        Err(Stop::Refused(m)) => {
            d.err.set(m);
            2
        }
    }
}
fn sched(p: *const CSched) -> S {
    // SAFETY: the caller's struct, valid during the call
    let x = unsafe { &*p };
    S { c: x.c, s: x.s, n: x.n.min(QMAX), st: x.st }
}

/// a new driver, or null when bind, retire, prefill or payload is missing
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_new(h: *const CHooks, p: *const CPlanner) -> *mut Driver {
    match Driver::new(*h, *p) {
        Some(d) => Box::into_raw(Box::new(d)),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_free(d: *mut Driver) {
    if !d.is_null() {
        drop(Box::from_raw(d));
    }
}

/// the last refusal's message
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_error(d: *const Driver) -> *const c_char {
    crate::ffi::msg((*d).err.get())
}

/// Driver::apply. pay: null without an admit. freed: room for the scheduler's slot count
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_apply(
    d: *const Driver,
    s: *const CSched,
    e: *const Commit,
    digest: u64,
    resize: bool,
    pay: *const CPayload,
    freed: *mut Run,
    n_freed: *mut usize,
    admitted: *mut bool,
) -> i32 {
    let d = &*d;
    let s = sched(s);
    let r = d.apply(&s, &*e, digest, resize, pay.as_ref()).map(|c| {
        let n = c.freed.len().min(s.n);
        std::slice::from_raw_parts_mut(freed, n).copy_from_slice(&c.freed[..n]);
        *n_freed = n;
        *admitted = c.admitted;
    });
    rc(d, r)
}

/// Driver::step: the quantum it ran in *out
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_step(d: *const Driver, s: *const CSched, out: *mut Quantum) -> i32 {
    let d = &*d;
    let r = d.step(&sched(s), &mut *out);
    rc(d, r)
}

/// the emit hook's accounting (Driver::emit); a slot out of range is refused
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_emit(
    d: *const Driver,
    q: u32,
    n: i32,
    count: u32,
    finished: bool,
    max_tokens: u32,
    done: *mut bool,
) -> i32 {
    let d = &*d;
    let r = if (q as usize) < QMAX {
        d.emit(q as usize, n, count, finished, max_tokens).map(|v| *done = v)
    } else {
        Err(Stop::Refused("s2: driver slot out of range"))
    };
    rc(d, r)
}

/// whether the quantum in flight is a decode round (Step::scheduler_decode)
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_decoding(d: *const Driver) -> bool {
    (*d).active_decode.get()
}

/// a slot's emitted tokens; 0 out of range
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_emitted(d: *const Driver, q: u32) -> u32 {
    (*d).emitted.get(q as usize).map_or(0, Cell::get)
}

/// whether a slot's session was admitted to the planner; false out of range
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_initialized(d: *const Driver, q: u32) -> bool {
    (*d).initialized.get(q as usize).is_some_and(Cell::get)
}

/// capped rounds so far (GR8_CP_STEP_ROWS)
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_cp_rounds(d: *const Driver) -> u64 {
    (*d).cp_rounds.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(i: u32, id: u64, phase: u32) -> Slot {
        let mut x: Slot = unsafe { std::mem::zeroed() };
        x.key.slot = i;
        x.req.id = id;
        x.phase = phase;
        x
    }

    #[test]
    fn decoding_matches_the_scheduler() {
        let s = [slot(0, 9, DECODE), slot(1, 3, PREFILL), slot(2, 5, DECODE), slot(3, 1, DECODE), slot(4, 7, FREE)];
        let mut k = [Key::default(); QMAX];
        let n = decoding(&s, &mut k);
        assert_eq!(&k[..n], &sch::decoding(&s)[..]);
    }

    #[test]
    fn emit_accounting() {
        let h: CHooks = unsafe { std::mem::zeroed() };
        let p: CPlanner = unsafe { std::mem::zeroed() };
        assert!(Driver::new(h, p).is_none());
        extern "C" fn nop_b(_: *mut c_void, _: *const Grant) -> i32 {
            0
        }
        extern "C" fn nop_s(_: *mut c_void, _: *const Slot) -> i32 {
            0
        }
        extern "C" fn nop_p(_: *mut c_void, _: *const Slot, _: u32, _: u32) -> i32 {
            0
        }
        extern "C" fn nop_y(_: *mut c_void, _: u32, _: bool) -> i32 {
            0
        }
        let h = CHooks { bind: Some(nop_b), retire: Some(nop_s), prefill: Some(nop_p), payload: Some(nop_y), ..h };
        let d = Driver::new(h, p).unwrap();
        assert_eq!(d.emit(1, 3, 3, false, 5), Ok(false));
        assert_eq!(d.emit(1, 3, 2, false, 5), Err(Stop::Refused("partial emission without stop")));
        assert_eq!(d.emit(1, 3, 3, false, 5), Err(Stop::Refused("bad emitter count")));
        assert_eq!(d.emit(1, 2, 2, false, 5), Ok(true));
        assert_eq!(d.emitted[1].get(), 5);
    }

    // a fake engine and planner: every session decodes forever, its run never moves
    extern "C" fn b_(_: *mut c_void, _: *const Grant) -> i32 {
        0
    }
    extern "C" fn s_(_: *mut c_void, _: *const Slot) -> i32 {
        0
    }
    extern "C" fn p_(_: *mut c_void, _: *const Slot, _: u32, _: u32) -> i32 {
        0
    }
    extern "C" fn y_(_: *mut c_void, _: u32, _: bool) -> i32 {
        0
    }
    unsafe extern "C" fn grow_(_: *mut c_void, s: *const Slot, _: u64, out: *mut Run, some: *mut bool) -> i32 {
        *out = (*s).run;
        *some = true;
        0
    }
    unsafe extern "C" fn round_(c: *mut c_void, _: *const i32, _: u32, ran: *mut bool) -> i32 {
        *(c as *mut u64) += 1;
        *ran = true;
        0
    }
    extern "C" fn v_(_: *mut c_void) -> i32 {
        0
    }
    extern "C" fn q_(_: *mut c_void, _: i32) -> i32 {
        0
    }
    unsafe extern "C" fn len_(_: *mut c_void, _: i32, out: *mut i32) -> i32 {
        *out = 100;
        0
    }
    unsafe extern "C" fn yes_(_: *mut c_void, _: i32, out: *mut bool) -> i32 {
        *out = true;
        0
    }
    unsafe extern "C" fn no_(_: *mut c_void, _: i32, out: *mut bool) -> i32 {
        *out = false;
        0
    }
    unsafe extern "C" fn zero_(_: *mut c_void, _: i32, out: *mut i32) -> i32 {
        *out = 0;
        0
    }

    #[test]
    fn rounds_do_not_allocate() {
        use crate::planner::tests::counting::{ALLOCS, COUNTING};
        let mut c: Config = unsafe { std::mem::zeroed() };
        c.epoch = 1;
        c.pool_rows = 614400;
        c.max_rows = 24;
        c.max_sessions = 8;
        c.mtp = 2;
        c.guard_tokens = u32::MAX;
        c.prefill_rows = 4096;
        c.two_lanes = true;
        c.notice_max = 8;
        c.cp_step_rows = 6; // three sessions past R: capped rounds of two
        c.cp_after = 0;
        let mut sl = [slot(0, 0, FREE); 8];
        sch::init(&c, &mut sl);
        for (n, i) in [0usize, 2, 5].into_iter().enumerate() {
            sl[i].key.epoch = 1;
            sl[i].phase = DECODE;
            sl[i].req.id = 10 + n as u64;
            sl[i].req.max_tokens = 1_000_000;
        }
        let mut st: State = unsafe { std::mem::zeroed() };
        st.want_decode = true;
        let mut rounds = 0u64;
        let z: CHooks = unsafe { std::mem::zeroed() };
        let h = CHooks {
            bind: Some(b_),
            retire: Some(s_),
            prefill: Some(p_),
            payload: Some(y_),
            grow: Some(grow_),
            has_grow: true,
            ..z
        };
        let p = CPlanner {
            ctx: &mut rounds as *mut u64 as *mut c_void,
            admit: None,
            admit_pack: None,
            round: Some(round_),
            demote: Some(v_),
            promote: Some(q_),
            release: Some(q_),
            length: Some(len_),
            finished: Some(no_),
            active: Some(yes_),
            inserted: Some(zero_),
        };
        let d = Driver::new(h, p).unwrap();
        for i in [0, 2, 5] {
            d.initialized[i].set(true);
        }
        let s = S { c: &c, s: sl.as_mut_ptr(), n: 8, st: &mut st };
        let mut q = Quantum::default();
        for _ in 0..10 {
            d.step(&s, &mut q).unwrap();
        }
        ALLOCS.with(|a| a.set(0));
        COUNTING.with(|x| x.set(true));
        for _ in 0..500 {
            d.step(&s, &mut q).unwrap();
        }
        COUNTING.with(|x| x.set(false));
        assert_eq!(ALLOCS.with(|a| a.get()), 0);
        assert_eq!(rounds, 510);
        assert_eq!(q.kind, K_DECODE);
        assert_eq!(d.cp_rounds.get(), 510);
    }
}
