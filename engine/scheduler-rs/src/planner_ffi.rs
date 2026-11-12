//! The planner's C entry points (g6sched.h, `g6s2_*`). The planner's state is Rust-owned behind an opaque
//! `g6s2_batch`. The C++ facade (s2_batch.h `s2::Batch`) passes settings in as the plain structs below, answers the
//! hooks through C function pointers with a context pointer, and reads results and the learned state back out.
//!
//! A call that runs steps returns 0, 1 when a hook failed (the facade holds that hook's exception and rethrows it) or
//! 2 when the planner refused (`g6s2_batch_error` has the message). Either way the call stops where it was, as the
//! C++ planner's exception did, and the state is left as far as it got.
//!
//! Safety, for every function here: `b` is a handle from `g6s2_batch_new` not yet freed, pointers are valid for the
//! counts passed with them (null only with a count of 0), and a slot index is below the planner's slot count.
#![allow(clippy::missing_safety_doc)]

use crate::ngram::{NgramTable, NB};
use crate::planner::{
    Batch, Cost, DSel, DSelTable, DflDrafts, DflReq, DflReuse, Engine, KStop, KStopB, NgramSrc, Res, Ring, Row, SessIn,
    St, Stats, StepBuf, Stop, DFL_B, MAX_SLOTS, VX,
};
use crate::spec_len::{SpecLen, KX, LEVELS_MAX, NB as SNB};
use std::ffi::{c_char, c_void, CString};

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 || p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// g6s2_step: one step as the run hook sees it, its arrays valid during the call
#[repr(C)]
pub struct CStep {
    pub kind: i32,
    pub event: i32,
    pub cls: i32,
    pub round_active: i32,
    pub round: i64,
    pub n_sess: i32,
    pub n_rows: i32,
    pub sess: *const i32,
    pub row0: *const i32,
    pub rows: *const i32,
    pub ctx: *const i32,
    pub ev: *const i32,
    pub r: *const Row,
}

pub type RunFn = unsafe extern "C" fn(*mut c_void, *const CStep, *mut i32) -> i32;
pub type RoundFn = unsafe extern "C" fn(*mut c_void, i32, i64, i32, i32, i32) -> i32;
pub type PromptFn = unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> i32;
pub type EmitFn = unsafe extern "C" fn(*mut c_void, i32, *const i32, i32, i32, *mut bool) -> i32;
pub type DflFn = unsafe extern "C" fn(*mut c_void, *const DflReq, *mut DflDrafts) -> i32;
pub type DflStartFn = unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> i32;
pub type LpFn = unsafe extern "C" fn(*mut c_void, i32, *mut f32) -> i32;

/// g6s2_hooks: each function returns 0, or nonzero when the C++ hook threw. has_*: the C++ Hooks member is set. The
/// planner reads has_* as the C++ read the member. Where the C++ called an empty member, the function runs and fails
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CHooks {
    pub ctx: *mut c_void,
    pub run: Option<RunFn>,
    pub round: Option<RoundFn>,
    pub prompt: Option<PromptFn>,
    pub emit: Option<EmitFn>,
    pub dfl: Option<DflFn>,
    pub dfl_start: Option<DflStartFn>,
    pub lp: Option<LpFn>,
    pub has_round: bool,
    pub has_prompt: bool,
    pub has_emit: bool,
    pub has_dfl: bool,
    pub has_dfl_start: bool,
    pub has_lp: bool,
}

fn ok(rc: i32) -> Res<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Stop::Hook)
    }
}

fn missing() -> Stop {
    Stop::Refused("s2: a hook the planner calls has no function".into())
}

impl Engine for CHooks {
    fn run(&self, s: &StepBuf, out: &mut [i32]) -> Res<()> {
        let f = self.run.ok_or_else(missing)?;
        let c = CStep {
            kind: s.kind,
            event: s.event,
            cls: s.cls,
            round_active: s.round_active,
            round: s.round,
            n_sess: s.sess.len() as i32,
            n_rows: s.r.len() as i32,
            sess: s.sess.as_ptr(),
            row0: s.row0.as_ptr(),
            rows: s.rows.as_ptr(),
            ctx: s.ctx.as_ptr(),
            ev: s.ev.as_ptr(),
            r: s.r.as_ptr(),
        };
        // SAFETY: the C++ facade's trampoline reads the step's arrays and writes n_rows tokens
        ok(unsafe { f(self.ctx, &c, out.as_mut_ptr()) })
    }
    fn has_round(&self) -> bool {
        self.has_round
    }
    fn round(&self, q: i32, n: i64, base: i32, a: i32, k: i32) -> Res<()> {
        let f = self.round.ok_or_else(missing)?;
        ok(unsafe { f(self.ctx, q, n, base, a, k) })
    }
    fn has_prompt(&self) -> bool {
        self.has_prompt
    }
    fn prompt(&self, q: i32, m: i32, pos0: i32) -> Res<()> {
        let f = self.prompt.ok_or_else(missing)?;
        ok(unsafe { f(self.ctx, q, m, pos0) })
    }
    fn has_emit(&self) -> bool {
        self.has_emit
    }
    fn emit(&self, q: i32, toks: &[i32], row: i32) -> Res<bool> {
        let f = self.emit.ok_or_else(missing)?;
        let mut done = false;
        ok(unsafe { f(self.ctx, q, toks.as_ptr(), toks.len() as i32, row, &mut done) })?;
        Ok(done)
    }
    fn has_dfl(&self) -> bool {
        self.has_dfl
    }
    fn dfl(&self, r: &DflReq) -> Res<DflDrafts> {
        let f = self.dfl.ok_or_else(missing)?;
        let mut o = DflDrafts::default();
        ok(unsafe { f(self.ctx, r, &mut o) })?;
        Ok(o)
    }
    fn has_dfl_start(&self) -> bool {
        self.has_dfl_start
    }
    fn dfl_start(&self, q: i32) -> Res<i32> {
        let f = self.dfl_start.ok_or_else(missing)?;
        let mut lo = 0;
        ok(unsafe { f(self.ctx, q, &mut lo) })?;
        Ok(lo)
    }
    fn has_lp(&self) -> bool {
        self.has_lp
    }
    fn lp(&self, row: i32) -> Res<f32> {
        let f = self.lp.ok_or_else(missing)?;
        let mut v = 0.0;
        ok(unsafe { f(self.ctx, row, &mut v) })?;
        Ok(v)
    }
}

/// g6s2_spec_len: spec_len.h SpecLen, its levels inline (at most LEVELS_MAX)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CSpecLen {
    pub on: bool,
    pub n_levels: i32,
    pub levels: [i32; LEVELS_MAX],
    pub draft_round: f64,
    pub draft_each: f64,
    pub table: [f64; KX + 2],
    pub seed: [f64; KX + 1],
    pub pn: [f64; KX + 1],
    pub pk: [f64; KX + 1],
    pub bn: [[f64; SNB]; KX + 1],
    pub bk: [[f64; SNB]; KX + 1],
    pub pos_decay: f64,
    pub bucket_decay: f64,
    pub prior: f64,
    pub rounds: i64,
    pub chosen: [i64; KX + 1],
    pub pred_sum: f64,
    pub got_sum: f64,
}

impl From<&CSpecLen> for SpecLen {
    fn from(c: &CSpecLen) -> SpecLen {
        SpecLen {
            on: c.on,
            levels: c.levels,
            n_levels: (c.n_levels.max(0) as usize).min(LEVELS_MAX),
            draft_round: c.draft_round,
            draft_each: c.draft_each,
            table: c.table,
            seed: c.seed,
            pn: c.pn,
            pk: c.pk,
            bn: c.bn,
            bk: c.bk,
            pos_decay: c.pos_decay,
            bucket_decay: c.bucket_decay,
            prior: c.prior,
            rounds: c.rounds,
            chosen: c.chosen,
            pred_sum: c.pred_sum,
            got_sum: c.got_sum,
        }
    }
}

impl From<&SpecLen> for CSpecLen {
    fn from(s: &SpecLen) -> CSpecLen {
        // every field named: a new one fails to compile here until the learn record exports it
        let SpecLen {
            on,
            levels,
            n_levels,
            draft_round,
            draft_each,
            table,
            seed,
            pn,
            pk,
            bn,
            bk,
            pos_decay,
            bucket_decay,
            prior,
            rounds,
            chosen,
            pred_sum,
            got_sum,
        } = *s;
        CSpecLen {
            on,
            n_levels: n_levels as i32,
            levels,
            draft_round,
            draft_each,
            table,
            seed,
            pn,
            pk,
            bn,
            bk,
            pos_decay,
            bucket_decay,
            prior,
            rounds,
            chosen,
            pred_sum,
            got_sum,
        }
    }
}

/// g6s2_dsel_table: s2_batch.h DSelTable's values
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CDSelTable {
    pub v: [f64; VX + 1],
    pub d: f64,
    pub m: f64,
    pub e: f64,
    pub p: f64,
    pub g: f64,
    pub k: f64,
    pub n: f64,
    pub w: f64,
    pub r: f64,
}

/// g6s2_dsel: s2_batch.h DSel
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CDSel {
    pub on: bool,
    pub dwell: i32,
    pub margin: f64,
    pub probe: i32,
    pub model: i32,
    pub d: CSpecLen,
    pub m: CSpecLen,
    pub tab: CDSelTable,
    pub rn: f64,
    pub rc: f64,
    pub an: f64,
    pub ag: f64,
    pub prior: f64,
    pub dn: f64,
    pub dt: f64,
    pub dc: f64,
}

impl From<&CDSel> for DSel {
    fn from(c: &CDSel) -> DSel {
        let t = &c.tab;
        DSel {
            on: c.on,
            dwell: c.dwell,
            margin: c.margin,
            probe: c.probe,
            d: (&c.d).into(),
            m: (&c.m).into(),
            model: c.model,
            tab: DSelTable { v: t.v, d: t.d, m: t.m, e: t.e, p: t.p, g: t.g, k: t.k, n: t.n, w: t.w, r: t.r },
            rn: c.rn,
            rc: c.rc,
            an: c.an,
            ag: c.ag,
            prior: c.prior,
            dn: c.dn,
            dt: c.dt,
            dc: c.dc,
        }
    }
}

impl From<&DSel> for CDSel {
    fn from(s: &DSel) -> CDSel {
        // every field named: a new one fails to compile here until the learn record exports it
        let DSel { on, dwell, margin, probe, ref d, ref m, model, tab, rn, rc, an, ag, prior, dn, dt, dc } = *s;
        let DSelTable { v, d: td, m: tm, e, p, g, k, n, w, r } = tab;
        CDSel {
            on,
            dwell,
            margin,
            probe,
            model,
            d: d.into(),
            m: m.into(),
            tab: CDSelTable { v, d: td, m: tm, e, p, g, k, n, w, r },
            rn,
            rc,
            an,
            ag,
            prior,
            dn,
            dt,
            dc,
        }
    }
}

/// g6s2_kstop: s2_batch.h KStop with KStopB's arrays by pointer: b_tau holds b_tau_rows rows of 8
#[repr(C)]
pub struct CKStop {
    pub mode: i32,
    pub kmax: i32,
    pub tau: [f64; 4],
    pub seed3: f64,
    pub log: bool,
    pub solo: bool,
    pub b_mode: i32,
    pub b_kmin: i32,
    pub b_explore: i32,
    pub b_tau_rows: i32,
    pub b_tau: *const f64,
    pub b_widths: *const i32,
    pub b_n_widths: i32,
}

/// g6s2_ngram: s2_batch.h NgramSrc and, when `table`, its table's entries (ngram_lookup.h, parsed by the C++)
#[repr(C)]
pub struct CNgram {
    pub table: bool,
    pub n_ns: i32,
    pub ns: *const i32,
    pub has_cal: bool,
    pub cal: [f64; NB],
    pub n: usize,
    pub keys: *const u64,
    pub toks: *const i32,
    pub ps: *const f64,
    pub kmax: i32,
    pub kmin: i32,
    pub margin: f64,
    pub log: bool,
}

/// g6s2_session: s2_batch.h Session
#[repr(C)]
pub struct CSession {
    pub prompt: *const i32,
    pub n_prompt: usize,
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub dflash: bool,
    pub ins_after: i32,
    pub ins_unless: i32,
    pub ins_tokens: *const i32,
    pub n_ins: usize,
}

/// g6s2_cost: s2_batch.h S2Cost, with Flash's solo table when `flash` (spec_len.h flash_cost)
#[repr(C)]
pub struct CCost {
    pub main0: f64,
    pub main_row: f64,
    pub draft0: f64,
    pub draft_sess: f64,
    pub chain0: f64,
    pub chain_sess: f64,
    pub flash: bool,
    pub flash_table: [f64; KX + 2],
    pub flash_draft_round: f64,
    pub flash_draft_each: f64,
}

/// g6s2_stats: s2_batch.h Stats. acc is valid until the planner's next call
#[repr(C)]
pub struct CStats {
    pub rounds: i64,
    pub tokens: i64,
    pub drafted: i64,
    pub mtp_rounds: i64,
    pub switches: i64,
    pub inserted: i64,
    pub ngram_rounds: i64,
    pub ngram_drafted: i64,
    pub ngram_acc: i64,
    pub dfl_rounds: i64,
    pub dfl_from: i32,
    pub dfl_lo: i32,
    pub dfl_warm: i32,
    pub promoted: i32,
    pub acc: *const i64,
    pub n_acc: usize,
}

/// g6s2_slot: one slot's session and draft state, for the round log's learn record. The arrays are valid until the
/// next call
#[repr(C)]
pub struct CSlot {
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub dflash: bool,
    pub ins_after: i32,
    pub ins_unless: i32,
    pub n_ins: usize,
    pub ins_tokens: *const i32,
    pub n_tokens: usize,
    pub tokens: *const i32,
    pub n_d: usize,
    pub d: *const i32,
    pub n_dlm: usize,
    pub dlm: *const f32,
    pub n: i32,
    pub on: bool,
    pub done: bool,
    pub rounds: i32,
    pub k: i32,
    pub dfl: bool,
    pub mtp: bool,
    pub pb: i32,
    pub pd: [i32; DFL_B],
    pub pl: [f32; DFL_B],
    pub since: i32,
    pub mn: i32,
    pub lo: i32,
    pub warm: i32,
    pub armed: bool,
    pub force: i32,
    pub dd: [i32; DFL_B],
    pub dl: [f32; DFL_B],
}

/// g6s2_learn: the planner's shared state besides its estimates (the drafter's ring and the round counters)
#[repr(C)]
pub struct CLearn {
    pub round_no: i64,
    pub cur_round: i64,
    pub cur_active: i32,
    pub ring_q: i32,
    pub ring_lo: i32,
    pub ring_end: i32,
    pub ring_top: i32,
    pub n_ring_tok: usize,
    pub ring_tok: *const i32,
}

fn in_range<E: Engine>(b: &Batch<E>, q: i32) -> bool {
    q >= 0 && (q as usize) < b.slots()
}

/// q, or a refusal when it is not one of the planner's slots (a corrupt round log)
fn slot<E: Engine>(b: &Batch<E>, q: i32) -> Res<i32> {
    if in_range(b, q) {
        Ok(q)
    } else {
        Err(Stop::Refused(format!("s2: slot {q} is out of range")))
    }
}

/// the handle: the planner and its last refusal
pub struct Planner {
    b: Batch<CHooks>,
    err: CString,
}

fn rc(p: &mut Planner, r: Res<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(Stop::Hook) => 1,
        Err(Stop::Refused(m)) => {
            p.err = CString::new(m).unwrap_or_default();
            2
        }
    }
}

/// GR8_DS_SESS_DRAFTS's rule (s2/ds41_sess_drafts.h calls these)
#[no_mangle]
pub extern "C" fn g6s2_sess_drafts_kv(k: i32, max_rows: i32, n: i32) -> i32 {
    crate::planner::sess_drafts_kv(k, max_rows, n)
}

#[no_mangle]
pub extern "C" fn g6s2_sess_drafts_max(k: i32, max_rows: i32, slots: i32) -> i32 {
    crate::planner::sess_drafts_max(k, max_rows, slots)
}

/// the n-gram table's key of a context (ngram_lookup.h NgramTable::key calls it)
#[no_mangle]
pub unsafe extern "C" fn g6s2_ngram_key(ctx: *const i32, n: i32) -> u64 {
    NgramTable::key(slice(ctx, n.max(0) as usize))
}

/// a new SpecLen's settings and seeds, for the C++'s check against g6rt::SpecLen
#[no_mangle]
pub unsafe extern "C" fn g6s2_spec_len_default(out: *mut CSpecLen) {
    *out = (&SpecLen::default()).into();
}

/// SpecLen::bucket, for the same check
#[no_mangle]
pub extern "C" fn g6s2_spec_len_bucket(lp: f32) -> i32 {
    SpecLen::bucket(lp) as i32
}

/// a new planner, or null when slots exceeds the most it holds
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_new(
    k: i32,
    max_rows: i32,
    slots: i32,
    par: bool,
    fresh_spec: bool,
    cost: *const CCost,
    hooks: *const CHooks,
) -> *mut Planner {
    if slots < 0 || slots as usize > MAX_SLOTS {
        return std::ptr::null_mut();
    }
    let c = &*cost;
    let mut flash_cost = SpecLen { table: c.flash_table, ..SpecLen::default() };
    flash_cost.draft_round = c.flash_draft_round;
    flash_cost.draft_each = c.flash_draft_each;
    let cost = Cost {
        main0: c.main0,
        main_row: c.main_row,
        draft0: c.draft0,
        draft_sess: c.draft_sess,
        chain0: c.chain0,
        chain_sess: c.chain_sess,
        flash: c.flash,
        flash_cost,
    };
    let b = Batch::new(k, max_rows, slots as usize, par, fresh_spec, cost, *hooks);
    Box::into_raw(Box::new(Planner { b, err: CString::default() }))
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_free(b: *mut Planner) {
    if !b.is_null() {
        drop(Box::from_raw(b));
    }
}

/// the last refusal's message (valid until the next call that refuses)
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_error(b: *const Planner) -> *const c_char {
    (*b).err.as_ptr()
}

unsafe fn sessions<'a>(ss: *const CSession, n: usize) -> Vec<SessIn<'a>> {
    slice(ss, n)
        .iter()
        .map(|s| SessIn {
            prompt: slice(s.prompt, s.n_prompt),
            first: s.first,
            chunk: s.chunk,
            n_gen: s.n_gen,
            dflash: s.dflash,
            ins_after: s.ins_after,
            ins_unless: s.ins_unless,
            ins_tokens: slice(s.ins_tokens, s.n_ins),
        })
        .collect()
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_admit(
    b: *mut Planner,
    qs: *const i32,
    n: usize,
    ss: *const CSession,
    n_ss: usize,
) -> i32 {
    let p = &mut *b;
    let ss = sessions(ss, n_ss);
    let r = p.b.admit(slice(qs, n), &ss);
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_admit_packed_tail(
    b: *mut Planner,
    qs: *const i32,
    n: usize,
    ss: *const CSession,
    n_ss: usize,
) -> i32 {
    let p = &mut *b;
    let ss = sessions(ss, n_ss);
    let r = p.b.admit_packed_tail(slice(qs, n), &ss);
    rc(p, r)
}

/// one round over the active sessions of order; *ran = false when none was active
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_round(b: *mut Planner, order: *const i32, n: usize, ran: *mut bool) -> i32 {
    let p = &mut *b;
    let r = p.b.round(slice(order, n)).map(|x| *ran = x);
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_demote(b: *mut Planner) -> i32 {
    let p = &mut *b;
    let r = p.b.demote();
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_promote(b: *mut Planner, q: i32) -> i32 {
    let p = &mut *b;
    let r = slot(&p.b, q).map(|q| p.b.promote(q));
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_cancel(b: *mut Planner, q: i32) -> i32 {
    let p = &mut *b;
    let r = slot(&p.b, q).map(|q| p.b.cancel(q));
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_release(b: *mut Planner, q: i32) -> i32 {
    let p = &mut *b;
    let r = slot(&p.b, q).map(|q| p.b.release(q));
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_set_mtp_sessions(b: *mut Planner, n: i32) {
    (*b).b.set_mtp_sessions(n)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_set_sess_drafts(b: *mut Planner, on: bool) {
    (*b).b.set_sess_drafts(on)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_adapt(b: *mut Planner, on: bool) {
    (*b).b.set_adapt(on)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_draft_par(b: *mut Planner, on: bool) {
    (*b).b.set_draft_par(on)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_fresh_spec(b: *mut Planner, on: bool) {
    (*b).b.set_fresh_spec(on)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_dflash(b: *mut Planner, kv: i32, sl: *const CSpecLen) {
    (*b).b.dflash(kv, (&*sl).into())
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_dsel(b: *mut Planner, d: *const CDSel) {
    (*b).b.dsel((&*d).into())
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_kstop(b: *mut Planner, k: *const CKStop) {
    let k = &*k;
    let tau = slice(k.b_tau, k.b_tau_rows.max(0) as usize * 8);
    let ks = KStop {
        mode: k.mode,
        kmax: k.kmax,
        tau: k.tau,
        seed3: k.seed3,
        log: k.log,
        solo: k.solo,
        b: KStopB {
            mode: k.b_mode,
            kmin: k.b_kmin,
            tau: tau.as_chunks::<8>().0.to_vec(),
            widths: slice(k.b_widths, k.b_n_widths.max(0) as usize).to_vec(),
            explore: k.b_explore,
        },
    };
    (*b).b.kstop(ks)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_dfl_reuse(b: *mut Planner, mode: i32, min: i32) {
    (*b).b.dfl_reuse(DflReuse { mode, min })
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_ngram(b: *mut Planner, n: *const CNgram) -> i32 {
    let p = &mut *b;
    let c = &*n;
    let t = if c.table {
        let mut t = NgramTable {
            ns: slice(c.ns, c.n_ns.max(0) as usize).to_vec(),
            has_cal: c.has_cal,
            cal: c.cal,
            ..Default::default()
        };
        let (keys, toks, ps) = (slice(c.keys, c.n), slice(c.toks, c.n), slice(c.ps, c.n));
        t.t.reserve(c.n);
        for i in 0..c.n {
            t.t.insert(keys[i], (toks[i], ps[i]));
        }
        Some(t)
    } else {
        None
    };
    let r = p.b.ngram(NgramSrc { t, kmax: c.kmax, kmin: c.kmin, margin: c.margin, log: c.log });
    rc(p, r)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_mtp_max_sess(b: *const Planner) -> i32 {
    (*b).b.mtp_max_sess()
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_active(b: *const Planner, q: i32) -> bool {
    in_range(&(*b).b, q) && (*b).b.active(q)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_finished(b: *const Planner, q: i32) -> bool {
    in_range(&(*b).b, q) && (*b).b.finished(q)
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_length(b: *const Planner, q: i32) -> i32 {
    if in_range(&(*b).b, q) {
        (*b).b.length(q)
    } else {
        0
    }
}

/// the session's generated tokens, valid until the next call
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_result(b: *const Planner, q: i32, p: *mut *const i32, n: *mut usize) {
    let r = if in_range(&(*b).b, q) { (*b).b.result(q) } else { &[] };
    *p = r.as_ptr();
    *n = r.len();
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_stats(b: *const Planner, q: i32, out: *mut CStats) {
    let p = &*b;
    let Some(s) = p.b.stats.get(q as usize) else {
        *out = std::mem::zeroed();
        return;
    };
    // every field named: a new one fails to compile here until the learn record exports it
    let Stats {
        rounds,
        tokens,
        drafted,
        mtp_rounds,
        switches,
        inserted,
        ngram_rounds,
        ngram_drafted,
        ngram_acc,
        ref acc,
        dfl_rounds,
        dfl_from,
        dfl_lo,
        dfl_warm,
        promoted,
    } = *s;
    *out = CStats {
        rounds,
        tokens,
        drafted,
        mtp_rounds,
        switches,
        inserted,
        ngram_rounds,
        ngram_drafted,
        ngram_acc,
        dfl_rounds,
        dfl_from,
        dfl_lo,
        dfl_warm,
        promoted,
        acc: acc.as_ptr(),
        n_acc: acc.len(),
    };
}

/// the round log's state record: 6 values a slot (out holds 6 x slots)
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_state(b: *const Planner, out: *mut u64) {
    let p = &*b;
    p.b.state(std::slice::from_raw_parts_mut(out, 6 * p.b.slots()));
}

/// the drafter choice's settings and estimates
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_dsel_state(b: *const Planner, out: *mut CDSel) {
    *out = (&(*b).b.sel).into();
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_dsel_rates(b: *const Planner, rm: *mut f64, rd: *mut f64) {
    let (m, d) = (*b).b.dsel_rates();
    *rm = m;
    *rd = d;
}

/// an estimate: 0 = the DFlash2 rounds' (dflash), 1 / 2 = the n-gram choice's DFlash2 / MTP, 3 + q = slot q's
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_spec_len(b: *const Planner, which: i32, out: *mut CSpecLen) {
    let p = &(*b).b;
    let s = match which {
        0 => &p.dsl,
        1 => &p.ngd,
        2 => &p.ngm,
        w => match p.sl.get((w as usize).wrapping_sub(3)) {
            Some(s) => s,
            None => {
                *out = std::mem::zeroed();
                return;
            }
        },
    };
    *out = s.into();
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_slot(b: *const Planner, q: i32, out: *mut CSlot) {
    let p = &*b;
    let Some(x) = p.b.st.get(q as usize) else {
        *out = std::mem::zeroed();
        return;
    };
    // every field named: a new one fails to compile here until the learn record exports it
    let St {
        first,
        chunk,
        n_gen,
        dflash,
        ins_after,
        ins_unless,
        ref ins_tokens,
        ref tokens,
        ref d,
        ref dlm,
        n,
        on,
        done,
        rounds,
        k,
        dfl,
        mtp,
        pb,
        pd,
        pl,
        since,
        mn,
        lo,
        warm,
        armed,
        force,
        dd,
        dl,
    } = *x;
    *out = CSlot {
        first,
        chunk,
        n_gen,
        dflash,
        ins_after,
        ins_unless,
        n_ins: ins_tokens.len(),
        ins_tokens: ins_tokens.as_ptr(),
        n_tokens: tokens.len(),
        tokens: tokens.as_ptr(),
        n_d: d.len(),
        d: d.as_ptr(),
        n_dlm: dlm.len(),
        dlm: dlm.as_ptr(),
        n,
        on,
        done,
        rounds,
        k,
        dfl,
        mtp,
        pb,
        pd,
        pl,
        since,
        mn,
        lo,
        warm,
        armed,
        force,
        dd,
        dl,
    };
}

#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_learn(b: *const Planner, out: *mut CLearn) {
    let p = &(*b).b;
    let Ring { q, ref tok, lo, end, top } = p.ring; // every field named, as above
    *out = CLearn {
        round_no: p.round_no,
        cur_round: p.cur_round,
        cur_active: p.cur_active,
        ring_q: q,
        ring_lo: lo,
        ring_end: end,
        ring_top: top,
        n_ring_tok: tok.len(),
        ring_tok: tok.as_ptr(),
    };
}

/// Configure deterministic shared allocation before admitting sessions.
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_waterfill(b: *mut Planner, on: bool, margin: f64, shared_only: bool) -> i32 {
    let p = &mut *b;
    let result = p.b.set_waterfill(on, margin);
    if result.is_ok() { p.b.wf_ng_shared_only = shared_only; }
    rc(p, result)
}

/// Copy the shared-only confidence state into the round log's learn record.
#[no_mangle]
pub unsafe extern "C" fn g6s2_batch_waterfill_lp(b: *const Planner, out: *mut f32) {
    for (q, row) in (*b).b.wf_lp.iter().enumerate() {
        std::ptr::copy_nonoverlapping(row.as_ptr(), out.add(q * 8), 8);
    }
}
