//! The S2 driver (s2/driver.h `g6s2::Driver` is its C++ facade): it applies commits, grows runs before a round, picks
//! the capped rounds past R, runs one scheduler quantum on the planner and reports completions. It owns the
//! scheduler's state (config, slots, counters), the per-slot emitted counts, the initialised flags, what it reads of
//! each payload and the capped-round cycle. The C++ reads the slots and counters through one view
//! (`g6s2_driver_view`). The planner is reached through the C++ `s2::Batch`, which writes the round log.
//!
//! Given the rank's pool, the driver makes the pure pool calls itself (`CPool`): growth before a round, a worker's
//! half of a shrink commit, and a leaving session's release when nothing is retained. Each locks the pool for that call
//! only, and no hook runs while it is held. Binds, retains, parks and tier resizes stay in the engine's hooks, which
//! wrap their pool calls in model work.
//!
//! Hooks may read the driver back (the service's retire reads a slot's emitted tokens, the view's slots), so every
//! field is a `Cell` or behind an `UnsafeCell` and every method takes `&self`; apply and step refuse to be entered
//! again from a hook.
#![allow(clippy::missing_safety_doc)]

use crate::cp_rows::{self, Cand};
use crate::scheduler::{
    self as sch, Commit, Completion, Config, Grant, Key, Quantum, Run, Shrink, Slot, State, DECODE, FREE, K_DECODE,
    K_PREFILL, PARKED, PREFILL, PREFILL_PACK, PROMPT_TAIL, PROMPT_TAIL_PACK, QMAX, RETIRING,
};
use g6kvman::ffi::Handle;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::{c_char, c_void};
use std::io::Write;

/// g6s2_drv_view: the scheduler's slots (n of them) and counters, as the driver holds them; valid while the driver
/// lives
#[repr(C)]
pub struct CView {
    pub s: *const Slot,
    pub n: usize,
    pub st: *const State,
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

/// g6s2_drv_pool: the pool calls the driver makes itself on the pool passed to g6s2_driver_new. grow: runs grow before
/// a round (then the grown hook, required). shrink: a worker's half of a shrink commit, before the shrink hook.
/// release: a leaving session's run goes back to the pool (no prefix reuse; retain and park stay in the hooks). head:
/// this rank made the commit's cancel and shrink calls when it built the commit, so the driver skips them
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CPool {
    pub grow: bool,
    pub shrink: bool,
    pub release: bool,
    pub head: bool,
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
/// grown(slot as it was, its new run)
pub type GrownFn = unsafe extern "C" fn(*mut c_void, *const Slot, *const Run) -> i32;

/// g6s2_drv_hooks: the engine's callbacks (s2/driver.h DriverHooks less the planner's). Each returns 0, or nonzero
/// when the C++ hook threw. has_*: the C++ member is set. payload: the C++ copies apply's payload into the slot (set)
/// or clears it. grow decides growth itself; with the driver's own pool growth (CPool::grow), grown follows a run
/// that grew
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
    pub grown: Option<GrownFn>,
    pub has_prefill_pack: bool,
    pub has_dfl_ok: bool,
    pub has_terminal: bool,
    pub has_grow: bool,
    pub has_finished: bool,
    pub has_resize: bool,
    pub has_shrink: bool,
    pub has_settle: bool,
    pub has_grown: bool,
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
fn pool_err(e: g6kvman::Error) -> Stop {
    Stop::Refused(e.message().to_str().unwrap_or("pool error"))
}
fn kv(r: &Run) -> g6kvman::Run {
    g6kvman::Run { first_page: r.first_page, pages: r.pages, table_version: r.table_version, list_hash: r.list_hash }
}
fn run_of(r: &g6kvman::Run) -> Run {
    Run { first_page: r.first_page, pages: r.pages, table_version: r.table_version, list_hash: r.list_hash }
}

/// the scheduler's state; the C++ view points into it
struct Sched {
    c: Config,
    s: [Slot; QMAX],
    n: usize,
    st: State,
}

/// One driver: its hooks, the scheduler's state and the per-slot state it keeps between calls.
pub struct Driver {
    h: CHooks,
    p: CPlanner,
    pool: *const Handle,
    u: CPool,
    sch: UnsafeCell<Sched>,
    emitted: [Cell<u32>; QMAX],
    initialized: [Cell<bool>; QMAX],
    prompt_len: [Cell<usize>; QMAX],
    decode_prompt: [Cell<bool>; QMAX],
    reused: [Cell<u32>; QMAX],
    served: RefCell<Vec<u64>>, // GR8_CP_STEP_ROWS: ids served in the current cycle
    cp_capped: Cell<bool>,
    cp_rounds: Cell<u64>,
    active_decode: Cell<bool>, // the quantum in flight is a decode round
    busy: Cell<bool>,          // in apply or step
    err: Cell<&'static str>,
}

/// the scheduler's state for one call; slices are formed only around scheduler calls, since hooks read the slots
struct S {
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
    /// A driver over these hooks with `n` slots, each `empty` (the C++ default slot) before `init` keys it, making
    /// the pool calls `u` names on `pool`, which must outlive it. None when a required hook (bind, retire, prefill,
    /// payload) is missing, n is not in 1..=QMAX, `u` names a call without a pool, or both the grow hook and the
    /// driver would grow runs.
    pub fn new(
        h: CHooks,
        p: CPlanner,
        c: Config,
        n: usize,
        empty: Slot,
        pool: *const Handle,
        u: CPool,
    ) -> Option<Driver> {
        if h.bind.is_none() || h.retire.is_none() || h.prefill.is_none() || h.payload.is_none() || n == 0 || n > QMAX {
            return None;
        }
        if ((u.grow || u.shrink || u.release) && pool.is_null()) || (u.grow && (h.has_grow || !h.has_grown)) {
            return None;
        }
        let mut s = [empty; QMAX];
        sch::init(&c, &mut s[..n]);
        // SAFETY: State is integers and bools; all zero is the C++ value-initialised state
        let mut st: State = unsafe { std::mem::zeroed() };
        st.want_decode = true;
        Some(Driver {
            h,
            p,
            pool,
            u,
            sch: UnsafeCell::new(Sched { c, s, n, st }),
            emitted: Default::default(),
            initialized: Default::default(),
            prompt_len: Default::default(),
            decode_prompt: Default::default(),
            reused: Default::default(),
            served: RefCell::new(Vec::with_capacity(QMAX)),
            cp_capped: Cell::new(false),
            cp_rounds: Cell::new(0),
            active_decode: Cell::new(false),
            busy: Cell::new(false),
            err: Cell::new(""),
        })
    }

    fn s(&self) -> S {
        let x = self.sch.get();
        // SAFETY: field addresses only; S forms references for the length of one scheduler call
        unsafe { S { c: &raw const (*x).c, s: (&raw mut (*x).s).cast(), n: (*x).n, st: &raw mut (*x).st } }
    }

    /// the pool for one call; the guard must not outlive the statement that calls the pool
    fn pool(&self) -> Res<std::sync::MutexGuard<'_, g6kvman::Pool>> {
        require(!self.pool.is_null(), "s2: the driver has no pool")?;
        // SAFETY: the caller of new() keeps the pool alive while the driver lives
        Ok(unsafe { &*self.pool }.lock())
    }
    fn release(&self, r: &Run) -> Res<()> {
        self.pool()?.release(&kv(r)).map_err(pool_err)
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
    pub fn apply(&self, e: &Commit, digest: u64, resize: bool, pay: Option<&CPayload>) -> Res<sch::Changes> {
        let _b = self.enter()?;
        let s = &self.s();
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
            // a cancelled live session's run; the head gave it back when it built the commit, and a finished one went
            // back at its step
            if self.u.release && !self.u.head && old.phase != RETIRING {
                self.release(&old.run)?;
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
            self.reused[i].set(0);
            self.emitted[i].set(0);
        }
        if e.has_shrink {
            require(self.h.has_shrink, "a shrink commit without the shrink hook")?;
            let k = e.shrink.key.slot;
            let len = self.length(k)?;
            let was = &before[k as usize].run;
            if !self.u.head {
                // the head's change, on the same pool state (a worker without the pool's shrink refuses)
                require(self.u.shrink, "a shrink commit without the pool's shrink")?;
                let r = self.pool()?.shrink_run(&kv(was), e.shrink.r_rows).map_err(pool_err)?;
                require(run_of(&r) == e.shrink.run, "worker shrink differs from the head's")?;
            }
            // SAFETY: as retire
            ok(unsafe { self.h.shrink.ok_or_else(missing)?(self.h.ctx, &e.shrink, was, len as u64) })?;
        }
        if e.has_admit {
            let (g, p) = (&e.admit, pay.expect("validated"));
            let i = g.key.slot as usize;
            self.payload(g.key.slot, true)?;
            self.prompt_len[i].set(p.prompt_len);
            self.decode_prompt[i].set(p.decode_prompt);
            self.reused[i].set(g.reuse);
            // SAFETY: as retire
            ok(unsafe { self.h.bind.ok_or_else(missing)?(self.h.ctx, g) })?;
            if p.decode_prompt {
                s.with(|_, sl, st| sch::skip_prefill_from(sl, st, &g.key, g.reuse)).map_err(sch_err)?;
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
    pub fn step(&self, out: &mut Quantum) -> Res<()> {
        let _b = self.enter()?;
        let s = &self.s();
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
            let nd = sch::decoding_into(s.slots(), &mut keys);
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
        if self.h.has_grow || self.u.grow {
            let mut keys = [Key::default(); QMAX];
            let nd = sch::decoding_into(s.slots(), &mut keys);
            for k in &keys[..nd] {
                let x = s.slot(k.slot as usize);
                let rows = sch::round_rows(s.c(), self.length(k.slot)? as u64);
                let got = if self.h.has_grow {
                    let (mut r, mut some) = (Run::default(), false);
                    // SAFETY: the facade's trampoline; x outlives the call
                    ok(unsafe { self.h.grow.ok_or_else(missing)?(self.h.ctx, &x, rows, &mut r, &mut some) })?;
                    some.then_some(r)
                } else {
                    let g = self.pool()?.grow(&kv(&x.run), rows).map_err(pool_err)?;
                    g.map(|r| run_of(&r))
                };
                match got {
                    None => stalled[k.slot as usize] = true,
                    Some(r) if r != x.run => {
                        if !self.h.has_grow {
                            // SAFETY: as above; r outlives the call
                            ok(unsafe { self.h.grown.ok_or_else(missing)?(self.h.ctx, &x, &r) })?;
                        }
                        s.with(|c, sl, st| sch::grow(c, sl, st, k, &r)).map_err(sch_err)?;
                    }
                    Some(_) => {}
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
                let first = if dp { self.reused[i].get() as i32 } else { q.first as i32 };
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
                // its pages go back now, on every rank, so the sessions still decoding can grow before the next event
                if self.u.release {
                    self.release(&x.run)?;
                }
                if self.h.has_finished {
                    // SAFETY: as prefill
                    ok(unsafe { self.h.finished.ok_or_else(missing)?(self.h.ctx, &x) })?;
                }
            }
        }
        Ok(())
    }

    /// Scheduler::idle_boundary: an event-only boundary, no model work.
    pub fn idle_boundary(&self, target: u64) -> Res<()> {
        let _b = self.enter()?;
        self.s().with(|_, sl, st| sch::idle_boundary(sl, st, target)).map_err(sch_err)
    }

    /// The pool's digest at this event boundary, folded into the scheduler's.
    pub fn pool_digest(&self, d: u64) -> Res<()> {
        let _b = self.enter()?;
        self.s().with(|_, _, st| st.pool = d);
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
/// a new driver with `n` slots over the scheduler config `c`, each slot `empty` (the C++ default) before it is keyed,
/// making the pool calls `u` names on `pool` (null: none; it must outlive the driver). Null as Driver::new refuses
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_new(
    h: *const CHooks,
    p: *const CPlanner,
    c: *const Config,
    n: usize,
    empty: *const Slot,
    pool: *const Handle,
    u: *const CPool,
) -> *mut Driver {
    match Driver::new(*h, *p, *c, n, *empty, pool, *u) {
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

/// the scheduler's slots and counters, where the driver keeps them for its life
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_view(d: *const Driver, out: *mut CView) {
    let s = (*d).s();
    *out = CView { s: s.s, n: s.n, st: s.st };
}

/// Driver::apply. pay: null without an admit. freed: room for the driver's slot count
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_apply(
    d: *const Driver,
    e: *const Commit,
    digest: u64,
    resize: bool,
    pay: *const CPayload,
    freed: *mut Run,
    n_freed: *mut usize,
    admitted: *mut bool,
) -> i32 {
    let d = &*d;
    let n_slots = d.s().n;
    let r = d.apply(&*e, digest, resize, pay.as_ref()).map(|c| {
        let n = c.freed.len().min(n_slots);
        std::slice::from_raw_parts_mut(freed, n).copy_from_slice(&c.freed[..n]);
        *n_freed = n;
        *admitted = c.admitted;
    });
    rc(d, r)
}

/// Driver::step: the quantum it ran in *out
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_step(d: *const Driver, out: *mut Quantum) -> i32 {
    let d = &*d;
    let r = d.step(&mut *out);
    rc(d, r)
}

/// fails the scheduler: no later commit or step runs
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_fail(d: *const Driver) {
    (*d).s().fail();
}

/// Scheduler::idle_boundary
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_idle_boundary(d: *const Driver, target: u64) -> i32 {
    let d = &*d;
    rc(d, d.idle_boundary(target))
}

/// the pool's digest at this event boundary (Scheduler::pool_digest)
#[no_mangle]
pub unsafe extern "C" fn g6s2_driver_pool_digest(d: *const Driver, v: u64) -> i32 {
    let d = &*d;
    rc(d, d.pool_digest(v))
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
    fn decoding_in_request_order() {
        let s = [slot(0, 9, DECODE), slot(1, 3, PREFILL), slot(2, 5, DECODE), slot(3, 1, DECODE), slot(4, 5, DECODE)];
        let slots: Vec<u32> = sch::decoding(&s).iter().map(|k| k.slot).collect();
        assert_eq!(slots, [3, 2, 4, 0]); // request 5 twice: slot order
    }

    #[test]
    fn emit_accounting() {
        let h: CHooks = unsafe { std::mem::zeroed() };
        let p: CPlanner = unsafe { std::mem::zeroed() };
        let c: Config = unsafe { std::mem::zeroed() };
        let e = slot(0, 0, FREE);
        assert!(Driver::new(h, p, c, 8, e, std::ptr::null(), CPool::default()).is_none());
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
        assert!(Driver::new(h, p, c, 0, e, std::ptr::null(), CPool::default()).is_none());
        assert!(Driver::new(h, p, c, QMAX + 1, e, std::ptr::null(), CPool::default()).is_none());
        let d = Driver::new(h, p, c, 8, e, std::ptr::null(), CPool::default()).unwrap();
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
        let d = Driver::new(h, p, c, 8, slot(0, 0, FREE), std::ptr::null(), CPool::default()).unwrap();
        d.s().with(|_, sl, _| {
            for (n, i) in [0usize, 2, 5].into_iter().enumerate() {
                sl[i].key.epoch = 1;
                sl[i].phase = DECODE;
                sl[i].req.id = 10 + n as u64;
                sl[i].req.max_tokens = 1_000_000;
            }
        });
        for i in [0, 2, 5] {
            d.initialized[i].set(true);
        }
        let mut q = Quantum::default();
        for _ in 0..10 {
            d.step(&mut q).unwrap();
        }
        ALLOCS.with(|a| a.set(0));
        COUNTING.with(|x| x.set(true));
        for _ in 0..500 {
            d.step(&mut q).unwrap();
        }
        COUNTING.with(|x| x.set(false));
        assert_eq!(ALLOCS.with(|a| a.get()), 0);
        assert_eq!(rounds, 510);
        assert_eq!(q.kind, K_DECODE);
        assert_eq!(d.cp_rounds.get(), 510);
    }

    // the driver's own pool growth: a fake planner whose sessions grow by `step` rows a round
    unsafe extern "C" fn len_grow(c: *mut c_void, _: i32, out: *mut i32) -> i32 {
        let t = &*(c as *const PoolTest);
        *out = t.len.get();
        0
    }
    unsafe extern "C" fn round_grow(c: *mut c_void, _: *const i32, _: u32, ran: *mut bool) -> i32 {
        let t = &*(c as *const PoolTest);
        t.len.set(t.len.get() + t.step);
        *ran = true;
        0
    }
    // grown: counts, and checks that another thread can take the pool meanwhile (the driver holds no lock here)
    unsafe extern "C" fn grown_(c: *mut c_void, s: *const Slot, r: *const Run) -> i32 {
        let t = &*(c as *const PoolTest);
        assert!((*r).pages > (*s).run.pages && (*r).table_version == (*s).run.table_version);
        t.grown.set(t.grown.get() + 1);
        let pool = t.pool as usize;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let p = unsafe { &*(pool as *const Handle) };
            tx.send(p.lock().stats().live_runs).ok();
        });
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(10)), Ok(3));
        0
    }
    struct PoolTest {
        len: Cell<i32>,
        step: i32,
        grown: Cell<u32>,
        pool: *const Handle,
    }

    fn pool_driver(t: &PoolTest, c: Config, u: CPool) -> Driver {
        let z: CHooks = unsafe { std::mem::zeroed() };
        let h = CHooks {
            ctx: t as *const PoolTest as *mut c_void,
            bind: Some(b_),
            retire: Some(s_),
            prefill: Some(p_),
            payload: Some(y_),
            grown: Some(grown_),
            has_grown: true,
            ..z
        };
        let p = CPlanner {
            ctx: t as *const PoolTest as *mut c_void,
            admit: None,
            admit_pack: None,
            round: Some(round_grow),
            demote: Some(v_),
            promote: Some(q_),
            release: Some(q_),
            length: Some(len_grow),
            finished: Some(no_),
            active: Some(yes_),
            inserted: Some(zero_),
        };
        Driver::new(h, p, c, 8, slot(0, 0, FREE), t.pool, u).unwrap()
    }

    fn decode_config() -> Config {
        let mut c: Config = unsafe { std::mem::zeroed() };
        c.epoch = 1;
        c.pool_rows = 64 * 64;
        c.max_rows = 24;
        c.max_sessions = 8;
        c.mtp = 2;
        c.guard_tokens = u32::MAX;
        c.prefill_rows = 4096;
        c.two_lanes = true;
        c.notice_max = 8;
        c
    }

    // three decoding sessions on runs of the pool
    fn seat(d: &Driver, pool: &Handle, rows: u64, worst: u64) {
        let runs: Vec<Run> = (0..3).map(|_| run_of(&pool.lock().reserve(rows, worst).unwrap())).collect();
        d.s().with(|_, sl, _| {
            for (n, i) in [0usize, 2, 5].into_iter().enumerate() {
                sl[i].key.epoch = 1;
                sl[i].phase = DECODE;
                sl[i].req.id = 10 + n as u64;
                sl[i].req.prompt_tokens = 4;
                sl[i].req.max_tokens = 1200;
                sl[i].run = runs[n];
            }
        });
        for i in [0, 2, 5] {
            d.initialized[i].set(true);
        }
    }

    #[test]
    fn grows_runs_in_the_pool() {
        let pool = Handle::new(g6kvman::Pool::new(64, 64, 64 * 64).unwrap());
        let t = PoolTest { len: Cell::new(10), step: 16, grown: Cell::new(0), pool: &pool };
        let u = CPool { grow: true, ..Default::default() };
        // the grow hook and the driver's growth together, the driver's growth without the grown hook, or pool calls
        // without a pool, are refused
        let z: CHooks = unsafe { std::mem::zeroed() };
        let p: CPlanner = unsafe { std::mem::zeroed() };
        let h = CHooks { bind: Some(b_), retire: Some(s_), prefill: Some(p_), payload: Some(y_), has_grow: true, ..z };
        assert!(Driver::new(h, p, decode_config(), 8, slot(0, 0, FREE), &pool, u).is_none());
        let h = CHooks { has_grow: false, ..h };
        assert!(Driver::new(h, p, decode_config(), 8, slot(0, 0, FREE), &pool, u).is_none());
        let h = CHooks { grown: Some(grown_), has_grown: true, ..h };
        assert!(Driver::new(h, p, decode_config(), 8, slot(0, 0, FREE), std::ptr::null(), u).is_none());
        assert!(Driver::new(h, p, decode_config(), 8, slot(0, 0, FREE), &pool, u).is_some());

        let d = pool_driver(&t, decode_config(), u);
        seat(&d, &pool, 64, 64 * 20);
        let mut q = Quantum::default();
        for _ in 0..20 {
            d.step(&mut q).unwrap();
        }
        // each run grew to cover its length + guard at the last round; the slots hold the pool's runs
        let st = pool.lock().stats();
        assert_eq!((st.live_runs, st.grows as u32), (3, t.grown.get()));
        assert!(t.grown.get() >= 3 * 3);
        let rows = sch::round_rows(&decode_config(), (t.len.get() - t.step) as u64);
        for i in [0, 2, 5] {
            let r = d.s().slot(i).run;
            assert_eq!(r.pages as u64, rows.div_ceil(64));
            assert_eq!(pool.lock().list(&kv(&r)).unwrap().len(), r.pages as usize);
        }
        // past a run's worst case the pool refuses, and the driver fails the scheduler with its message
        t.len.set(64 * 20);
        assert_eq!(d.step(&mut q), Err(Stop::Refused("pool growth past the run's worst case")));
        assert!(d.s().with(|_, _, st| st.failed));
    }

    #[test]
    fn pool_rounds_do_not_allocate() {
        use crate::planner::tests::counting::{ALLOCS, COUNTING};
        let pool = Handle::new(g6kvman::Pool::new(64, 64, 64 * 64).unwrap());
        // the runs already cover every round: each grow is the pool's lookup, no new pages
        let t = PoolTest { len: Cell::new(10), step: 0, grown: Cell::new(0), pool: &pool };
        let d = pool_driver(&t, decode_config(), CPool { grow: true, release: true, ..Default::default() });
        seat(&d, &pool, 64 * 4, 64 * 4);
        let mut q = Quantum::default();
        for _ in 0..10 {
            d.step(&mut q).unwrap();
        }
        ALLOCS.with(|a| a.set(0));
        COUNTING.with(|x| x.set(true));
        for _ in 0..500 {
            d.step(&mut q).unwrap();
        }
        COUNTING.with(|x| x.set(false));
        assert_eq!(ALLOCS.with(|a| a.get()), 0);
        assert_eq!((q.kind, t.grown.get()), (K_DECODE, 0));
    }

    unsafe extern "C" fn shrink_(_: *mut c_void, _: *const Shrink, _: *const Run, _: u64) -> i32 {
        0
    }
    fn commit(d: &Driver) -> Commit {
        let st = d.s().with(|_, _, st| *st);
        let mut e: Commit = unsafe { std::mem::zeroed() };
        (e.epoch, e.event, e.boundary) = (1, st.event + 1, st.boundary);
        e
    }

    #[test]
    fn a_worker_shrinks_in_the_pool_and_the_head_does_not() {
        // place 40, window 4, nr 4: the pool's R is 8 pages, the small one 4; seats of 20 pages
        let mk = || {
            let mut p = g6kvman::Pool::new(400, 40, 8 * 64).unwrap();
            p.tail(4, 0);
            p.window(4, 0, 0).unwrap();
            Handle::new(p)
        };
        let mut c = decode_config();
        (c.pool_rows, c.window_pages, c.r2_rows) = (400 * 64, 4, 4 * 64);
        // slot 2 takes the small R with this run
        // own: the driver has the pool (seated either way)
        let shrink_own = |pool: &Handle, u: CPool, run: Run, own: bool| {
            let t = PoolTest { len: Cell::new(10), step: 0, grown: Cell::new(0), pool };
            let mut d = pool_driver(&t, c, u);
            (d.h.shrink, d.h.has_shrink) = (Some(shrink_ as ShrinkFn), true);
            seat(&d, pool, 20 * 64, 20 * 64);
            if !own {
                d.pool = std::ptr::null();
            }
            let mut e = commit(&d);
            e.has_shrink = true;
            e.shrink = Shrink { key: d.s().slot(2).key, r_rows: c.r2_rows, run };
            d.apply(&e, 0, false, None).map(|_| (d.s().slot(2).run, pool.lock().digest()))
        };
        let shrink = |pool: &Handle, u: CPool, run: Run| shrink_own(pool, u, run, true);
        // the head's change, made when it built the commit, on a pool seated alike
        let mirror = mk();
        let runs: Vec<_> = (0..3).map(|_| mirror.lock().reserve(20 * 64, 20 * 64).unwrap()).collect();
        let want = run_of(&mirror.lock().shrink_run(&runs[1], c.r2_rows).unwrap());
        let shrunk = mirror.lock().digest();
        // the head's driver leaves its pool alone; a worker's makes the change and then agrees with the head
        let (r, d) = shrink(&mk(), CPool { shrink: true, head: true, ..Default::default() }, want).unwrap();
        assert!(r == want && d != shrunk);
        assert_eq!(shrink(&mk(), CPool { shrink: true, ..Default::default() }, want), Ok((want, shrunk)));
        // a worker without the pool's shrink (or without a pool), or whose change differs, refuses and fails the
        // scheduler
        let no = CPool { release: true, ..Default::default() };
        assert_eq!(shrink(&mk(), no, want), Err(Stop::Refused("a shrink commit without the pool's shrink")));
        let none = shrink_own(&mk(), CPool::default(), want, false);
        assert_eq!(none, Err(Stop::Refused("a shrink commit without the pool's shrink")));
        let bad = Run { list_hash: want.list_hash ^ 1, ..want };
        let yes = CPool { shrink: true, ..Default::default() };
        assert_eq!(shrink(&mk(), yes, bad), Err(Stop::Refused("worker shrink differs from the head's")));
    }

    #[test]
    fn leaving_sessions_release_their_runs() {
        let pool = Handle::new(g6kvman::Pool::new(64, 64, 64 * 64).unwrap());
        let live = |p: &Handle| p.lock().stats().live_runs;
        for head in [false, true] {
            let t = PoolTest { len: Cell::new(10), step: 0, grown: Cell::new(0), pool: &pool };
            let mut d = pool_driver(&t, decode_config(), CPool { release: true, head, ..Default::default() });
            seat(&d, &pool, 64, 64);
            // a cancelled live session: a worker releases its run, the head released it when it built the commit
            let x = d.s().slot(0);
            let mut e = commit(&d);
            (e.has_cancel, e.cancel) = (true, x.key);
            d.apply(&e, 0, false, None).unwrap();
            assert_eq!(live(&pool), if head { 3 } else { 2 });
            if head {
                pool.lock().release(&kv(&x.run)).unwrap();
            }
            // a session that finishes gives its run back at that step, on every rank
            let mut q = Quantum::default();
            d.step(&mut q).unwrap();
            assert_eq!(live(&pool), 2);
            d.p.finished = Some(yes_);
            d.step(&mut q).unwrap();
            assert_eq!((q.kind, live(&pool)), (K_DECODE, 0));
        }
    }
}
