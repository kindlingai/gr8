//! The S2 batch planner (s2_batch.h `s2::Batch`, docs/HOST-CORE.md R3b): which engine steps each round runs, their
//! rows, and the acceptance and drafter state between rounds. Hooks must not call back into the planner: it holds its
//! state mutably while it waits for them. The state lives here. The C++ `s2::Batch` is a facade over it: it passes the
//! settings in as plain data, answers the hooks, and writes the round log.
//!
//! Per session the step sequence is the one rank.cuh mtp_sequence runs for that session alone. A round is one main
//! step (every active session's verify rows), one first-form draft step, and K-1 chain steps. A solo DFlash2 session
//! runs dflash_sequence's rounds instead. The drafter choice, the confidence stops, the insert rule and the n-gram
//! source decide within that, and s2_batch.h's header describes each. The functions here keep the C++'s names, order
//! of operations and floating-point expressions: every rank, and a replay of a round log, must decide the same.

use crate::ngram::NgramTable;
use crate::spec_len::{dmax, dmin, SpecLen, KX};
use std::time::Instant;

/// DFlash2 block rows: the bonus row and 7 drafts (dflash.cuh dfl::B)
pub const DFL_B: usize = 8;
/// the drafter's window and its KV ring, positions (dfl::WIN, dfl::RING)
pub const DFL_WIN: i32 = 2048;
pub const DFL_RING: i32 = 4096;
/// the most session slots a planner holds (g6s2::MAX_SESSIONS is 8 or 32)
pub const MAX_SLOTS: usize = 32;
/// verify rows of the drafter choice's cost table: 1..VX
pub const VX: usize = KX + 1;

pub const MAIN: i32 = 0;
pub const DRAFT: i32 = 1;
pub const CHAIN: i32 = 2;
pub const TAIL: i32 = 0;
pub const DECODE1: i32 = 1;
pub const ROUND: i32 = 2;

/// the k1 split classes of bind::k1_split: 1 row / 2-3 rows / 4+ rows
pub fn row_class(rows: i32) -> i32 {
    if rows == 1 {
        1
    } else if rows <= 3 {
        3
    } else {
        5
    }
}

/// GR8_DS_SESS_DRAFTS (s2/ds41_sess_drafts.h): the drafts a session of a shared round of n verifies
pub fn sess_drafts_kv(k: i32, max_rows: i32, n: i32) -> i32 {
    if k <= 0 || n < 1 || max_rows < 1 {
        return 0;
    }
    k.min(max_rows / n - 1).max(0)
}

/// the most sessions that still draft, at most slots
pub fn sess_drafts_max(k: i32, max_rows: i32, slots: i32) -> i32 {
    let mut n = 0;
    while n < slots && sess_drafts_kv(k, max_rows, n + 1) >= 1 {
        n += 1;
    }
    n
}

/// One row of a step (s2_batch.h Row). spos: the position the row's sample is keyed to.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub sess: i32,
    pub tok: i32,
    pub pos: i32,
    pub spos: i32,
    pub tgt: i32,
}

/// One drafter step (s2_batch.h DflReq).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflReq {
    pub sess: i32,
    pub rows: i32,
    pub pos0: i32,
    pub tok0: i32,
    pub lo: i32,
    pub query: bool,
    pub phase: i32,
}

/// The drafter's answer (s2_batch.h DflDrafts).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DflDrafts {
    pub tok: [i32; DFL_B],
    pub lp: [f32; DFL_B],
}

/// One engine step as the planner builds it (s2_batch.h Step, less the driver's scheduler_decode). The planner keeps
/// one per step kind and refills it, so a round allocates nothing once the buffers have grown.
#[derive(Clone, Debug, Default)]
pub struct StepBuf {
    pub kind: i32,
    pub event: i32,
    pub cls: i32,
    pub round: i64,
    pub round_active: i32,
    pub sess: Vec<i32>,
    pub row0: Vec<i32>,
    pub rows: Vec<i32>,
    pub ctx: Vec<i32>,
    pub ev: Vec<i32>,
    pub r: Vec<Row>,
}

impl StepBuf {
    /// a new step: the C++ Step's defaults (event ROUND, class 1) with the given kind, round and active count
    fn begin(&mut self, kind: i32, round: i64, round_active: i32) -> &mut Self {
        self.kind = kind;
        self.event = ROUND;
        self.cls = 1;
        self.round = round;
        self.round_active = round_active;
        self.sess.clear();
        self.row0.clear();
        self.rows.clear();
        self.ctx.clear();
        self.ev.clear();
        self.r.clear();
        self
    }
    pub fn m(&self) -> i32 {
        self.r.len() as i32
    }
    /// add_rows: session q's rows (pushed after this call into r), its event class ev
    fn open(&mut self, q: i32, ev: i32) {
        self.sess.push(q);
        self.row0.push(self.m());
        self.ev.push(ev);
    }
    /// closes the session open() started: its row count and context
    fn close(&mut self) {
        let r0 = *self.row0.last().unwrap();
        self.rows.push(self.m() - r0);
        self.ctx.push(self.r.last().unwrap().pos + 1);
    }
    /// m draft rows of session q at positions pos0.. with next tokens toks[from..from + m] (rank.cuh draft_inputs). A
    /// token past the end of toks is 0: when an insert rule fires exactly at n_gen, a verify's draft step asks for the
    /// token after the last committed one. The C++ planner read it from past the end of its vector. The session is
    /// over and the row's output unused
    fn draft_rows(&mut self, q: i32, ev: i32, pos0: i32, toks: &[i32], from: usize, m: usize) {
        self.open(q, ev);
        for r in 0..m {
            let p = pos0 + r as i32;
            let tok = toks.get(from + r).copied().unwrap_or(0);
            self.r.push(Row { sess: q, tok, pos: p, spos: p + 2, tgt: -1 });
        }
        self.close();
    }
}

/// Why a call stopped: a hook failed (the C++ side holds its exception), or the planner refused (its message).
#[derive(Debug, PartialEq)]
pub enum Stop {
    Hook,
    Refused(String),
}
pub type Res<T> = Result<T, Stop>;

/// The engine's side of the planner (s2_batch.h Hooks). `has_*` is false for a member the C++ Hooks left empty.
pub trait Engine {
    /// the sampled token of every row of the step, into out (s.m() entries)
    fn run(&self, s: &StepBuf, out: &mut [i32]) -> Res<()>;
    fn has_round(&self) -> bool;
    fn round(&self, q: i32, n: i64, base: i32, a: i32, k: i32) -> Res<()>;
    fn has_prompt(&self) -> bool;
    fn prompt(&self, q: i32, m: i32, pos0: i32) -> Res<()>;
    fn has_emit(&self) -> bool;
    /// true: the session is finished
    fn emit(&self, q: i32, toks: &[i32], row: i32) -> Res<bool>;
    fn has_dfl(&self) -> bool;
    fn dfl(&self, r: &DflReq) -> Res<DflDrafts>;
    fn has_dfl_start(&self) -> bool;
    fn dfl_start(&self, q: i32) -> Res<i32>;
    fn has_lp(&self) -> bool;
    fn lp(&self, row: i32) -> Res<f32>;
}

/// A session as admit() takes it (s2_batch.h Session, its insert rule flattened).
#[derive(Clone, Copy, Debug)]
pub struct SessIn<'a> {
    pub prompt: &'a [i32],
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub dflash: bool,
    pub ins_after: i32,
    pub ins_unless: i32,
    pub ins_tokens: &'a [i32],
}

/// Per-session statistics (s2_batch.h Stats).
#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub rounds: i64,
    pub tokens: i64,
    pub drafted: i64,
    pub mtp_rounds: i64,
    pub switches: i64,
    pub inserted: i64,
    pub ngram_rounds: i64,
    pub ngram_drafted: i64,
    pub ngram_acc: i64,
    pub acc: Vec<i64>,
    pub dfl_rounds: i64,
    pub dfl_from: i32,
    pub dfl_lo: i32,
    pub dfl_warm: i32,
    pub promoted: i32,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            rounds: 0,
            tokens: 0,
            drafted: 0,
            mtp_rounds: 0,
            switches: 0,
            inserted: 0,
            ngram_rounds: 0,
            ngram_drafted: 0,
            ngram_acc: 0,
            acc: Vec::new(),
            dfl_rounds: 0,
            dfl_from: -1,
            dfl_lo: -1,
            dfl_warm: 0,
            promoted: -1,
        }
    }
}

impl Stats {
    /// back to the defaults, keeping acc's buffer
    fn reset(&mut self) {
        let mut acc = std::mem::take(&mut self.acc);
        acc.clear();
        *self = Stats { acc, ..Stats::default() };
    }
}

/// A slot's session and its draft state (s2_batch.h Batch::St).
#[derive(Clone, Debug)]
pub struct St {
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub dflash: bool,
    pub ins_after: i32,
    pub ins_unless: i32,
    pub ins_tokens: Vec<i32>,
    /// the prompt (tokens[..n]) and the committed tokens
    pub tokens: Vec<i32>,
    /// d[1..k]: drafts held for the next verify. dlm: their log-probabilities under the MTP head (NaN: unknown)
    pub d: Vec<i32>,
    pub dlm: Vec<f32>,
    pub n: i32,
    pub on: bool,
    pub done: bool,
    /// decode rounds so far (round 1's main step is DECODE1)
    pub rounds: i32,
    pub k: i32,
    /// on the DFlash2 drafter (until demote)
    pub dfl: bool,
    /// drafter choice: this round drafts with MTP
    pub mtp: bool,
    /// a pending probe's base position (-1: none); pd / pl its drafts
    pub pb: i32,
    pub pd: [i32; DFL_B],
    pub pl: [f32; DFL_B],
    pub since: i32,
    pub mn: i32,
    /// the lowest position the drafter holds context for
    pub lo: i32,
    /// GR8_DFL_REUSE: MTP rounds until the drafter's context reaches this position (0: none)
    pub warm: i32,
    /// the insert rule may still fire
    pub armed: bool,
    /// inserted tokens from this position on wait for their steps (-1: none)
    pub force: i32,
    /// DFlash2 drafts d1..d7 and their log-probabilities under the drafter
    pub dd: [i32; DFL_B],
    pub dl: [f32; DFL_B],
}

impl Default for St {
    fn default() -> Self {
        St {
            first: 0,
            chunk: 4,
            n_gen: 16,
            dflash: false,
            ins_after: -1,
            ins_unless: -1,
            ins_tokens: Vec::new(),
            tokens: Vec::new(),
            d: Vec::new(),
            dlm: Vec::new(),
            n: 0,
            on: false,
            done: false,
            rounds: 0,
            k: 0,
            dfl: false,
            mtp: false,
            pb: -1,
            pd: [0; DFL_B],
            pl: [0.0; DFL_B],
            since: 0,
            mn: 0,
            lo: 0,
            warm: 0,
            armed: false,
            force: -1,
            dd: [0; DFL_B],
            dl: [0.0; DFL_B],
        }
    }
}

impl St {
    /// `x = St()` keeping the vectors' buffers
    fn reset(&mut self) {
        let mut ins = std::mem::take(&mut self.ins_tokens);
        let mut tokens = std::mem::take(&mut self.tokens);
        let mut d = std::mem::take(&mut self.d);
        let mut dlm = std::mem::take(&mut self.dlm);
        ins.clear();
        tokens.clear();
        d.clear();
        dlm.clear();
        *self = St { ins_tokens: ins, tokens, d, dlm, ..St::default() };
    }
    fn active(&self) -> bool {
        self.on && !self.done && (self.tokens.len() as i64 - self.n as i64) < self.n_gen as i64
    }
    fn base(&self) -> i32 {
        self.tokens.len() as i32 - 1
    }
}

/// GR8_DFL_REUSE: what the drafter's KV ring holds (s2_batch.h Batch::Ring).
#[derive(Clone, Debug, PartialEq)]
pub struct Ring {
    pub q: i32,
    pub tok: Vec<i32>,
    pub lo: i32,
    pub end: i32,
    pub top: i32,
}

impl Default for Ring {
    fn default() -> Self {
        Ring { q: -1, tok: Vec::new(), lo: 0, end: 0, top: -1 }
    }
}

impl Ring {
    fn set(&mut self, q: i32, lo: i32, end: i32, top: i32) {
        self.q = q;
        self.tok.clear();
        self.lo = lo;
        self.end = end;
        self.top = top;
    }
}

/// GR8_S2_DSEL_TABLE (s2_batch.h DSelTable): the drafter choice's frozen cost table. Parsing stays C++.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DSelTable {
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

impl Default for DSelTable {
    fn default() -> Self {
        DSelTable { v: [0.0; VX + 1], d: 0.0, m: 0.0, e: 0.0, p: 0.0, g: -1.0, k: 0.0, n: 0.0, w: 0.0, r: 0.0 }
    }
}

impl DSelTable {
    /// the verify ms of a fractional row count, linear between rows
    pub fn verify(&self, rows: f64) -> f64 {
        let rows = dmin(dmax(rows, 1.0), VX as f64);
        let a = (rows as i32).min(VX as i32 - 1);
        let au = a as usize;
        self.v[au] + (rows - a as f64) * (self.v[au + 1] - self.v[au])
    }
    /// a SpecLen's verify table, drafting ms and prior weight
    fn price(&self, s: &mut SpecLen, round: f64, each: f64) {
        s.table[1..=VX].copy_from_slice(&self.v[1..=VX]);
        s.draft_round = round;
        s.draft_each = each;
        if self.w > 0.0 {
            s.prior = self.w;
        }
    }
}

/// The drafter choice (s2_batch.h DSel): settings and the request's estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DSel {
    pub on: bool,
    pub dwell: i32,
    pub margin: f64,
    pub probe: i32,
    pub d: SpecLen,
    pub m: SpecLen,
    pub model: i32,
    pub tab: DSelTable,
    pub rn: f64,
    pub rc: f64,
    pub an: f64,
    pub ag: f64,
    pub prior: f64,
    pub dn: f64,
    pub dt: f64,
    pub dc: f64,
}

impl Default for DSel {
    fn default() -> Self {
        DSel {
            on: false,
            dwell: 8,
            margin: 0.15,
            probe: 4,
            d: SpecLen::default(),
            m: SpecLen::default(),
            model: 0,
            tab: DSelTable::default(),
            rn: 0.0,
            rc: 0.0,
            an: 0.0,
            ag: 0.0,
            prior: 1.0,
            dn: 0.0,
            dt: 0.0,
            dc: 0.0,
        }
    }
}

/// GR8_MTP_KSTOPB (s2_batch.h KStopB): each session's verified drafts in a shared round.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KStopB {
    pub mode: i32,
    pub kmin: i32,
    /// [S][j], S 0..=slots, j 1..7
    pub tau: Vec<[f64; 8]>,
    pub widths: Vec<i32>,
    pub explore: i32,
}

impl KStopB {
    /// whether a shared round of s sessions may launch a main step of m rows (k: the MTP drafts)
    pub fn width(&self, s: i32, k: i32, m: i32) -> bool {
        if m < s * (self.kmin + 1) || m > s * (k + 1) {
            return false;
        }
        self.widths.is_empty() || m % (k + 1) == 0 || self.widths.contains(&m)
    }
}

/// GR8_MTP_KSTOP (s2_batch.h KStop).
#[derive(Clone, Debug, PartialEq)]
pub struct KStop {
    pub mode: i32,
    pub kmax: i32,
    pub tau: [f64; 4],
    pub seed3: f64,
    pub log: bool,
    pub solo: bool,
    pub b: KStopB,
}

impl Default for KStop {
    fn default() -> Self {
        KStop {
            mode: 0,
            kmax: 0,
            tau: [0.0, 0.0, 0.2, 0.5],
            seed3: 0.55,
            log: false,
            solo: false,
            b: KStopB::default(),
        }
    }
}

/// S2Cost: the step times that price a round for the adaptive draft count. flash: a session alone takes Flash's own
/// table, which the C++ fills from spec_len.h flash_cost
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub main0: f64,
    pub main_row: f64,
    pub draft0: f64,
    pub draft_sess: f64,
    pub chain0: f64,
    pub chain_sess: f64,
    pub flash: bool,
    pub flash_cost: SpecLen,
}

impl Default for Cost {
    fn default() -> Self {
        Cost {
            main0: 44.0,
            main_row: 7.0,
            draft0: 2.1,
            draft_sess: 0.25,
            chain0: 2.0,
            chain_sess: 0.15,
            flash: false,
            flash_cost: SpecLen::default(),
        }
    }
}

impl Cost {
    pub fn round(&self, s: i32, k: i32) -> f64 {
        if self.flash && s == 1 {
            return self.flash_cost.verify_ms(k + 1) + self.flash_cost.draft_ms(k);
        }
        self.main0
            + self.main_row * s as f64 * (k + 1) as f64
            + self.draft0
            + self.draft_sess * s as f64
            + (k - 1) as f64 * (self.chain0 + self.chain_sess * s as f64)
    }
}

/// GR8_DFL_REUSE (s2_batch.h DflReuse)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflReuse {
    pub mode: i32,
    pub min: i32,
}

impl Default for DflReuse {
    fn default() -> Self {
        DflReuse { mode: 0, min: 256 }
    }
}

/// GR8_NGRAM (s2_batch.h NgramSrc): the table (None: off) and its round choice
pub struct NgramSrc {
    pub t: Option<NgramTable>,
    pub kmax: i32,
    pub kmin: i32,
    pub margin: f64,
    pub log: bool,
}

impl Default for NgramSrc {
    fn default() -> Self {
        NgramSrc { t: None, kmax: 7, kmin: 2, margin: 0.1, log: false }
    }
}

/// a slot list on the stack (sessions of a round, at most MAX_SLOTS)
#[derive(Clone, Copy)]
struct Slots {
    v: [i32; MAX_SLOTS],
    n: usize,
}

impl Slots {
    fn new() -> Self {
        Slots { v: [0; MAX_SLOTS], n: 0 }
    }
    fn push(&mut self, q: i32) {
        self.v[self.n] = q;
        self.n += 1;
    }
    fn as_slice(&self) -> &[i32] {
        &self.v[..self.n]
    }
    fn len(&self) -> usize {
        self.n
    }
    fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// libc snprintf for the GR8_MTP_LPLOG lines, so they keep the C++'s %g / %f / %a text (tools read them)
mod cfmt {
    use std::os::raw::{c_char, c_int};
    extern "C" {
        pub fn snprintf(s: *mut c_char, n: usize, fmt: *const c_char, ...) -> c_int;
    }
    macro_rules! cfmt {
        ($out:expr, $fmt:literal $(, $a:expr)*) => {{
            let mut b = [0u8; 256];
            let f = concat!($fmt, "\0").as_ptr() as *const std::os::raw::c_char;
            // SAFETY: the format is NUL-terminated and its conversions match the argument types
            let n = unsafe { $crate::planner::cfmt::snprintf(b.as_mut_ptr() as *mut _, b.len(), f $(, $a)*) };
            if n >= 0 && (n as usize) < b.len() {
                $out.push_str(std::str::from_utf8(&b[..n as usize]).unwrap_or(""));
            } else if n >= 0 {
                let mut v = vec![0u8; n as usize + 1];
                // SAFETY: as above, into a buffer of the length the first call asked for
                let m = unsafe { $crate::planner::cfmt::snprintf(v.as_mut_ptr() as *mut _, v.len(), f $(, $a)*) };
                v.truncate(m.max(0) as usize);
                $out.push_str(std::str::from_utf8(&v).unwrap_or(""));
            }
        }};
    }
    pub(crate) use cfmt;
}
use cfmt::cfmt;

/// a NUL-terminated copy for a %s
fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap_or_default()
}

fn stderr_line(s: &str) {
    use std::io::Write;
    let _ = std::io::stderr().write_all(s.as_bytes());
}

fn run_step<E: Engine>(e: &E, s: &StepBuf, out: &mut Vec<i32>) -> Res<()> {
    out.clear();
    out.resize(s.r.len(), 0);
    e.run(s, out)
}

/// the c tokens t a step sampled for the session (the verify's accepted ones and its bonus), committed with the insert
/// rule applied. Returns the tokens committed with the inserted ones, and how many of t
fn commit(x: &mut St, stats: &mut Stats, t: &[i32]) -> (i32, i32) {
    let c = t.len() as i32;
    let len = x.tokens.len() as i32 - x.n;
    let mut i = 0;
    while i < c {
        if x.armed && len + i == x.ins_after {
            break;
        }
        if t[i as usize] == x.ins_unless {
            x.armed = false;
        }
        x.tokens.push(t[i as usize]);
        i += 1;
    }
    let kept = i;
    if i == c {
        return (c, kept);
    }
    x.armed = false;
    let m = (x.ins_tokens.len() as i32).min(x.n_gen - (len + i)).max(0);
    let from = x.tokens.len() as i32;
    x.tokens.extend_from_slice(&x.ins_tokens[..m as usize]);
    stats.inserted += m as i64;
    if (x.tokens.len() as i32 - x.n) < x.n_gen {
        x.force = from;
    }
    (i + m, kept)
}

/// KStop mode 1: the probability that drafts 1..j are all right, as the MTP head sees them (NaN: 1)
fn kprob(x: &St, j: i32) -> f64 {
    let mut p = 1.0;
    for i in 1..=j.max(0) as usize {
        p *= if x.dlm[i].is_nan() { 1.0 } else { (x.dlm[i] as f64).exp() };
    }
    p
}

/// the drafts' log-probabilities the choice's estimates read: DFlash2's always; MTP's only under KStop
fn ng_lp(x: &St, m: bool, ks_mode: i32) -> Option<&[f32]> {
    if m {
        if ks_mode != 0 {
            Some(&x.dlm)
        } else {
            None
        }
    } else {
        Some(&x.dl)
    }
}

pub struct Batch<E: Engine> {
    pub e: E,
    k: i32,
    max_rows: i32,
    slots: usize,
    mtp_sessions: i32,
    sess_drafts: bool,
    adapt: bool,
    dkv: i32,
    par: bool,
    fresh_spec: bool,
    pub dsl: SpecLen,
    pub sel: DSel,
    ks: KStop,
    reuse: DflReuse,
    ng: NgramSrc,
    pub ngd: SpecLen,
    pub ngm: SpecLen,
    pub ring: Ring,
    cost: Cost,
    pub sl: Vec<SpecLen>,
    pub round_no: i64,
    pub cur_round: i64,
    pub cur_active: i32,
    pub st: Vec<St>,
    pub stats: Vec<Stats>,
    // one step buffer and one output buffer per step kind: a round has a main, a draft and chain steps live at once
    sm: StepBuf,
    sd: StepBuf,
    sc: StepBuf,
    tm: Vec<i32>,
    td: Vec<i32>,
    tc: Vec<i32>,
}

impl<E: Engine> Batch<E> {
    /// K: the MTP drafts. slots: the planner's session slots, at most MAX_SLOTS. par and fresh_spec: GR8_DRAFT_PAR and
    /// GR8_S2_FRESH_SPEC as the C++ read them
    pub fn new(k: i32, max_rows: i32, slots: usize, par: bool, fresh_spec: bool, cost: Cost, e: E) -> Self {
        assert!(slots <= MAX_SLOTS, "s2: more slots than the planner holds");
        Batch {
            e,
            k,
            max_rows,
            slots,
            mtp_sessions: 0,
            sess_drafts: false,
            adapt: false,
            dkv: DFL_B as i32 - 1,
            par,
            fresh_spec,
            dsl: SpecLen::default(),
            sel: DSel::default(),
            ks: KStop::default(),
            reuse: DflReuse::default(),
            ng: NgramSrc::default(),
            ngd: SpecLen::default(),
            ngm: SpecLen::default(),
            ring: Ring::default(),
            cost,
            sl: vec![SpecLen::default(); slots],
            round_no: 0,
            cur_round: -1,
            cur_active: 0,
            st: vec![St::default(); slots],
            stats: vec![Stats::default(); slots],
            sm: StepBuf::default(),
            sd: StepBuf::default(),
            sc: StepBuf::default(),
            tm: Vec::new(),
            td: Vec::new(),
            tc: Vec::new(),
        }
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn set_mtp_sessions(&mut self, n: i32) {
        self.mtp_sessions = n;
    }
    pub fn set_sess_drafts(&mut self, on: bool) {
        self.sess_drafts = on;
    }
    pub fn set_adapt(&mut self, on: bool) {
        self.adapt = on;
    }
    pub fn set_draft_par(&mut self, on: bool) {
        self.par = on;
    }
    pub fn set_fresh_spec(&mut self, on: bool) {
        self.fresh_spec = on;
    }
    pub fn dflash(&mut self, kv: i32, sl: SpecLen) {
        self.dkv = kv;
        self.dsl = sl;
    }
    pub fn dsel(&mut self, d: DSel) {
        self.sel = d;
        self.price();
        let ks = self.ks.clone();
        self.kstop(ks);
    }
    /// solo MTP rounds' draft count (KStop). Mode 2 sets DSel::m's levels and third-position seed
    pub fn kstop(&mut self, k: KStop) {
        self.ks = k;
        if self.ks.mode != 2 {
            return;
        }
        let lv: Vec<i32> = (1..=self.kmax()).collect();
        self.sel.m.set_levels(&lv);
        self.sel.m.seed[3] = self.ks.seed3;
    }
    pub fn dfl_reuse(&mut self, r: DflReuse) {
        self.reuse = r;
    }
    /// the n-gram source. With a table it refuses kmin and kmax outside 1 <= kmin <= kmax <= 7
    pub fn ngram(&mut self, n: NgramSrc) -> Res<()> {
        if n.t.is_some() && (n.kmin < 1 || n.kmax < n.kmin || n.kmax > DFL_B as i32 - 1) {
            return Err(Stop::Refused("s2: ngram: 1 <= kmin <= kmax <= 7".into()));
        }
        self.ng = n;
        self.ng_fresh();
        Ok(())
    }

    pub fn mtp_max_sess(&self) -> i32 {
        if self.k <= 0 {
            return 0;
        }
        if self.sess_drafts {
            return sess_drafts_max(self.k, self.max_rows, self.slots as i32);
        }
        if self.mtp_sessions > 0 {
            (self.max_rows / (self.k + 1)).min(self.mtp_sessions)
        } else {
            self.max_rows / (self.k + 1)
        }
    }

    pub fn active(&self, q: i32) -> bool {
        self.st[q as usize].active()
    }
    pub fn finished(&self, q: i32) -> bool {
        self.st[q as usize].on && !self.active(q)
    }
    pub fn length(&self, q: i32) -> i32 {
        self.st[q as usize].tokens.len() as i32
    }
    /// the session's generated tokens (as mtp_sequence returns them)
    pub fn result(&self, q: i32) -> &[i32] {
        let x = &self.st[q as usize];
        let len = x.tokens.len();
        let e = x.n as i64 + x.n_gen as i64;
        let end = if e < 0 { len } else { len.min(e as usize) };
        &x.tokens[(x.n as usize).min(end)..end]
    }
    pub fn cancel(&mut self, q: i32) {
        let x = &mut self.st[q as usize];
        x.done = true;
        x.on = false;
    }
    pub fn release(&mut self, q: i32) {
        self.st[q as usize].on = false;
    }
    /// the round log's state record. Per slot: on, active, finished, length, the committed tokens' FNV-1a and the
    /// inserted count
    pub fn state(&self, out: &mut [u64]) {
        for q in 0..self.slots {
            let x = &self.st[q];
            let mut h: u64 = 1469598103934665603;
            for &t in &x.tokens {
                h ^= t as u32 as u64;
                h = h.wrapping_mul(1099511628211);
            }
            let o = &mut out[6 * q..6 * q + 6];
            o[0] = x.on as u64;
            o[1] = self.active(q as i32) as u64;
            o[2] = self.finished(q as i32) as u64;
            o[3] = x.tokens.len() as u64;
            o[4] = h;
            o[5] = self.stats[q].inserted as u64;
        }
    }

    fn all_active(&self) -> Slots {
        let mut all = Slots::new();
        for q in 0..self.slots as i32 {
            if self.active(q) {
                all.push(q);
            }
        }
        all
    }

    pub fn admit(&mut self, qs: &[i32], ss: &[SessIn]) -> Res<()> {
        if qs.len() != ss.len() {
            return Err(Stop::Refused("s2: admit: slots and sessions differ in count".into()));
        }
        let existing = self.all_active().len();
        let allow_mtp = self.k > 0 && (existing + qs.len()) as i64 <= self.mtp_max_sess() as i64;
        let mut live = Slots::new();
        for (k, &q) in qs.iter().enumerate() {
            if q < 0 || q as usize >= self.slots || self.st[q as usize].on {
                return Err(Stop::Refused(format!("s2: admit: slot {q} is out of range or in use")));
            }
            if !self.prompt(q, &ss[k], allow_mtp)? || self.st[q as usize].force >= 0 {
                continue;
            }
            if !self.st[q as usize].dfl {
                live.push(q);
            } else if self.st[q as usize].mtp {
                self.kchain(q, TAIL)?; // GR8_DFL_REUSE warm-up: d[2..K] for its first round
            }
        }
        let all = self.all_active();
        if !allow_mtp || all.len() as i64 > self.mtp_max_sess() as i64 {
            for &q in live.as_slice() {
                self.st[q as usize].k = 0;
            }
        } else if self.ksolo(all.len()) && live.len() == 1 {
            self.kchain(live.v[0], TAIL)?;
        } else {
            let evs = [TAIL; MAX_SLOTS];
            let kc = self.choose(all.as_slice());
            self.chain(live.as_slice(), &evs[..live.len()], kc, 1)?;
        }
        Ok(())
    }

    /// DS41 prefill pack: one main step takes every session's one owed prompt row, then one draft and the chain
    pub fn admit_packed_tail(&mut self, qs: &[i32], ss: &[SessIn]) -> Res<()> {
        if qs.len() < 2 || qs.len() > self.slots || qs.len() != ss.len() || qs.len() as i64 > self.max_rows as i64 {
            return Err(Stop::Refused("s2: packed tail count/width invalid".into()));
        }
        let mut seen = [false; MAX_SLOTS];
        for (i, &q) in qs.iter().enumerate() {
            let p = &ss[i];
            if q < 0
                || q as usize >= self.slots
                || seen[q as usize]
                || self.st[q as usize].on
                || p.prompt.len() < 2
                || p.first as i64 != p.prompt.len() as i64 - 1
                || p.chunk < 1
                || p.n_gen < 1
                || p.dflash
            {
                return Err(Stop::Refused("s2: packed tail slot or prompt invalid".into()));
            }
            seen[q as usize] = true;
        }
        self.sm.begin(MAIN, -1, 0).event = TAIL;
        for (i, &q) in qs.iter().enumerate() {
            let p = &ss[i];
            self.start_prompt(q, p)?;
            let pos = p.prompt.len() as i32 - 1;
            self.sm.open(q, TAIL);
            self.sm.r.push(Row { sess: q, tok: p.prompt[pos as usize], pos, spos: pos + 1, tgt: -1 });
            self.sm.close();
        }
        self.sm.cls = row_class(self.sm.m());
        self.check(&self.sm)?;
        run_step(&self.e, &self.sm, &mut self.tm)?;
        let mut drafted = Slots::new();
        for (i, &q) in qs.iter().enumerate() {
            let row = self.sm.row0[i];
            let qu = q as usize;
            if self.e.has_prompt() {
                self.e.prompt(q, 1, self.st[qu].n - 1)?;
            }
            let (c, _) = commit(&mut self.st[qu], &mut self.stats[qu], &self.tm[row as usize..row as usize + 1]);
            let x = &self.st[qu];
            if self.e.has_emit() && self.e.emit(q, &x.tokens[x.n as usize..(x.n + c) as usize], row)? {
                self.st[qu].done = true;
                continue;
            }
            drafted.push(q);
        }
        let all = self.all_active();
        let allow_mtp = self.k > 0 && !drafted.is_empty() && all.len() as i64 <= self.mtp_max_sess() as i64;
        if allow_mtp {
            self.sd.begin(DRAFT, -1, 0).event = TAIL;
            for &q in drafted.as_slice() {
                let x = &self.st[q as usize];
                self.sd.draft_rows(q, TAIL, x.n - 1, &x.tokens, x.n as usize, 1);
            }
            self.sd.cls = row_class(self.sd.m());
            self.check(&self.sd)?;
            run_step(&self.e, &self.sd, &mut self.td)?;
            for (i, &q) in drafted.as_slice().iter().enumerate() {
                let row = self.sd.row0[i];
                let lp = self.dlp(row)?;
                let x = &mut self.st[q as usize];
                x.d[1] = self.td[row as usize];
                x.dlm[1] = lp;
            }
        }
        let mut live = Slots::new();
        for &q in drafted.as_slice() {
            if self.st[q as usize].force < 0 {
                live.push(q);
            }
        }
        if !allow_mtp {
            for &q in live.as_slice() {
                self.st[q as usize].k = 0;
            }
        } else if self.ksolo(all.len()) && live.len() == 1 {
            self.kchain(live.v[0], TAIL)?;
        } else {
            let evs = [TAIL; MAX_SLOTS];
            let kc = self.choose(all.as_slice());
            self.chain(live.as_slice(), &evs[..live.len()], kc, 1)?;
        }
        Ok(())
    }

    /// GR8_DFL_REUSE=2: session q, the only active one and drafting with MTP, takes the drafter before the next round
    pub fn promote(&mut self, q: i32) {
        let qu = q as usize;
        {
            let x = &self.st[qu];
            if self.reuse.mode < 2 || !self.e.has_dfl() || !self.active(q) || x.dfl || x.force >= 0 {
                return;
            }
        }
        for o in 0..self.slots as i32 {
            if o != q && self.active(o) {
                return;
            }
        }
        let x = &mut self.st[qu];
        x.dfl = true;
        x.mtp = true;
        x.lo = x.tokens.len() as i32 - 1;
        x.warm = x.lo + self.reuse.min.max(1);
        x.pb = -1;
        x.since = 0;
        x.mn = 0;
        let (lo, warm) = (x.lo, x.warm);
        if self.sel.on {
            self.restart();
        }
        if self.ng.t.is_some() {
            self.ng_fresh();
        }
        let s = &mut self.stats[qu];
        if s.acc.len() < DFL_B {
            s.acc.resize(DFL_B, 0);
        }
        s.promoted = lo;
        s.dfl_lo = lo;
        s.dfl_warm = warm;
        self.ring.set(q, lo, lo, lo - 1);
    }

    /// a second session is resident: every DFlash2 session goes to MTP (the K-1 chain steps), before the next round
    pub fn demote(&mut self) -> Res<()> {
        for q in 0..self.slots as i32 {
            let qu = q as usize;
            if self.st[qu].dfl && self.active(q) {
                self.st[qu].dfl = false;
                if !self.st[qu].mtp && self.st[qu].force < 0 {
                    self.chain(&[q], &[ROUND], self.k, 1)?;
                } else if self.st[qu].k < self.k && self.st[qu].force < 0 {
                    let j0 = self.st[qu].k;
                    self.chain(&[q], &[ROUND], self.k, j0)?;
                }
                self.st[qu].mtp = false;
            }
        }
        Ok(())
    }

    /// one decode round of the active sessions of `order`. false: none was active and nothing ran
    pub fn round(&mut self, order: &[i32]) -> Res<bool> {
        let mut act = Slots::new();
        for &q in order {
            if self.active(q) {
                if act.len() == MAX_SLOTS {
                    return Err(Stop::Refused("s2: round: more sessions than slots".into()));
                }
                act.push(q);
            }
        }
        if act.is_empty() {
            return Ok(false);
        }
        // With a session cap a round past mtp_max_sess() drafts nothing, so it needs one row a session
        let rows = if self.mtp_sessions > 0 { act.len() as i64 } else { act.len() as i64 * (self.k as i64 + 1) };
        if rows > self.max_rows as i64 {
            return Err(Stop::Refused(format!(
                "s2: {} sessions need {} rows > max rows {}",
                act.len(),
                rows,
                self.max_rows
            )));
        }
        self.cur_round = self.round_no;
        self.round_no += 1;
        self.cur_active = act.len() as i32;
        for &q in act.as_slice() {
            if self.st[q as usize].force >= 0 {
                self.forced(q)?;
            }
        }
        let mut live = Slots::new();
        for &q in act.as_slice() {
            if self.active(q) {
                live.push(q);
            }
        }
        if live.is_empty() {
            self.cur_round = -1;
            self.cur_active = 0;
            return Ok(true);
        }
        if live.len() == 1 && self.st[live.v[0] as usize].dfl {
            self.round_dfl(live.v[0])?;
        } else {
            self.demote()?;
            if self.ks.solo && live.len() > 1 {
                self.complete(live.as_slice())?;
            }
            self.round_all(live.as_slice())?;
        }
        self.cur_round = -1;
        self.cur_active = 0;
        Ok(true)
    }

    fn round_all(&mut self, act: &[i32]) -> Res<()> {
        let (cur_round, cur_active) = (self.cur_round, self.cur_active);
        self.sm.begin(MAIN, cur_round, cur_active);
        // the first decode-round MAIN of a session is DECODE1; the step takes the lowest class among its sessions
        for &q in act {
            if self.st[q as usize].rounds == 0 {
                self.sm.event = DECODE1;
                break;
            }
        }
        let mut base = [0i32; MAX_SLOTS];
        let mtp_on = self.k > 0 && act.len() as i64 <= self.mtp_max_sess() as i64;
        let mut k = if mtp_on { self.k } else { 0 };
        if mtp_on && self.sess_drafts {
            k = k.min(sess_drafts_kv(self.k, self.max_rows, act.len() as i32));
        }
        if mtp_on {
            for &q in act {
                k = k.min(self.st[q as usize].k);
            }
        }
        let mut kq = [k; MAX_SLOTS];
        let mut aq = [-1i32; MAX_SLOTS];
        let mut nd = [0i32; MAX_SLOTS];
        let xr = mtp_on && self.ks.b.mode != 0 && self.ks.b.explore != 0 && cur_round % self.ks.b.explore as i64 == 0;
        let solo = mtp_on && self.ksolo(act.len());
        let mut pe = 0.0;
        if solo {
            kq[act[0] as usize] = self.kverify(act[0], &mut pe);
        } else if mtp_on && self.ks.b.mode != 0 {
            self.kshared(act, k, xr, &mut kq);
        }
        let t0 = if self.ks.log { Some(Instant::now()) } else { None };
        let mut ms = [0.0f64; 3];
        for &q in act {
            let qu = q as usize;
            let x = &self.st[qu];
            base[qu] = x.base();
            nd[qu] = x.k;
            self.sm.open(q, if x.rounds == 0 { DECODE1 } else { ROUND });
            for r in 0..=kq[qu] {
                let tok = if r > 0 { x.d[r as usize] } else { x.tokens[base[qu] as usize] };
                self.sm.r.push(Row { sess: q, tok, pos: base[qu] + r, spos: base[qu] + r + 1, tgt: -1 });
            }
            self.sm.close();
        }
        // a session's own rows (K+1), as its solo step, so a row's bits do not depend on the batch (KStop::solo: k+1)
        self.sm.cls = row_class(if solo { kq[act[0] as usize] + 1 } else { (if mtp_on { self.k } else { 0 }) + 1 });
        self.check(&self.sm)?;
        run_step(&self.e, &self.sm, &mut self.tm)?;
        if let Some(t) = t0 {
            ms[0] = t.elapsed().as_secs_f64() * 1e3;
        }
        let mut on = Slots::new();
        let mut acc = [0i32; MAX_SLOTS];
        for i in 0..self.sm.sess.len() {
            let q = self.sm.sess[i];
            let qu = q as usize;
            let r0 = self.sm.row0[i] as usize;
            let kv = kq[qu];
            let mut a = 0;
            {
                let x = &self.st[qu];
                let so = &self.tm[r0..];
                while a < kv && x.d[(a + 1) as usize] == so[a as usize] {
                    a += 1;
                }
            }
            aq[qu] = a;
            let (c, kept) = commit(&mut self.st[qu], &mut self.stats[qu], &self.tm[r0..r0 + (a + 1) as usize]);
            self.sl[qu].observe(kv, a, None, 0.0);
            if self.e.has_round() {
                self.e.round(q, self.stats[qu].rounds, base[qu], a, kv)?;
            }
            let s = &mut self.stats[qu];
            s.rounds += 1;
            s.tokens += (a + 1) as i64;
            s.acc[a as usize] += 1;
            s.drafted += kv as i64;
            self.st[qu].rounds += 1;
            acc[qu] = kept.min(a); // the draft step's rows: those whose input the session committed
            let x = &self.st[qu];
            let b1 = (base[qu] + 1) as usize;
            if self.e.has_emit() && self.e.emit(q, &x.tokens[b1..b1 + c as usize], self.sm.row0[i])? {
                self.st[qu].done = true;
                continue;
            }
            if self.st[qu].force < 0 {
                on.push(q);
            }
        }
        if self.ks.log {
            self.klog(act, k, xr, &kq, &aq, &nd, &base); // the drafts' lp before the draft step replaces them
        }
        let t1 = if self.ks.log { Some(Instant::now()) } else { None };
        let mut dm = 0;
        let ndraft = self.sm.sess.iter().filter(|&&q| !self.st[q as usize].done).count() as i64;
        let mms = self.mtp_max_sess() as i64;
        let draft_on = self.k > 0 && ndraft > 0 && ndraft <= mms && on.len() as i64 <= mms;
        if draft_on {
            // first-form draft: one step, rows a+1 of each survivor
            self.sd.begin(DRAFT, cur_round, cur_active).event = self.sm.event;
            for i in 0..self.sm.sess.len() {
                let q = self.sm.sess[i];
                let qu = q as usize;
                let x = &self.st[qu];
                if x.done {
                    continue;
                }
                let b1 = (base[qu] + 1) as usize;
                self.sd.draft_rows(q, self.sm.ev[i], base[qu], &x.tokens, b1, (acc[qu] + 1) as usize);
            }
            if !self.sd.sess.is_empty() {
                self.sd.cls = row_class(self.sd.m());
                self.check(&self.sd)?;
                dm = self.sd.m();
                run_step(&self.e, &self.sd, &mut self.td)?;
                for i in 0..self.sd.sess.len() {
                    // dlm[1]: KStop's input once GR8_DFL_REUSE=2 promotes the session to the solo path
                    let row = self.sd.row0[i] + self.sd.rows[i] - 1;
                    let lp = self.dlp(row)?;
                    let x = &mut self.st[self.sd.sess[i] as usize];
                    x.d[1] = self.td[row as usize];
                    x.dlm[1] = lp;
                }
            }
        }
        if draft_on {
            // the chain carries each survivor's own round class
            let mut evs = [0i32; MAX_SLOTS];
            for (j, &q) in on.as_slice().iter().enumerate() {
                evs[j] = match self.sm.sess.iter().position(|&s| s == q) {
                    Some(i) => self.sm.ev[i],
                    None => self.sm.event,
                };
            }
            let t2 = if self.ks.log { Some(Instant::now()) } else { None };
            let kc = if solo && on.len() == 1 { 0 } else { self.choose(on.as_slice()) };
            if solo && on.len() == 1 {
                self.kchain(on.v[0], evs[0])?;
            } else {
                self.chain(on.as_slice(), &evs[..on.len()], kc, 1)?;
            }
            if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                let t3 = Instant::now();
                ms[1] = (t2 - t1).as_secs_f64() * 1e3;
                ms[2] = (t3 - t2).as_secs_f64() * 1e3;
                let nch = if on.is_empty() {
                    0
                } else {
                    (if solo && on.len() == 1 { self.st[on.v[0] as usize].k } else { kc }) - 1
                };
                let mut l = String::new();
                cfmt!(
                    l,
                    "# kstopb_ms round %ld main %.3f draft %.3f dM %d chain %.3f nch %d cM %d round %.3f\n",
                    cur_round,
                    ms[0],
                    ms[1],
                    dm,
                    ms[2],
                    nch,
                    on.len() as i32,
                    (t3 - t0).as_secs_f64() * 1e3
                );
                stderr_line(&l);
            }
        } else {
            for &q in on.as_slice() {
                self.st[q as usize].k = 0;
            }
        }
        Ok(())
    }

    /// KStopB: each session's verified drafts kq[q] <= k in a shared round (rank-invariant: the drafts'
    /// log-probabilities, the round's sessions and its id)
    fn kshared(&self, act: &[i32], k: i32, xr: bool, kq: &mut [i32; MAX_SLOTS]) {
        let s = act.len() as i32;
        let lo = self.ks.b.kmin.min(k);
        let mut m = 0;
        for &q in act {
            let mut j = 0;
            if xr {
                let h = (self.cur_round as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add((q as u64).wrapping_mul(0xBF58476D1CE4E5B9));
                j = lo + ((h >> 33) as i32) % (k - lo + 1);
            } else {
                while j < k && kprob(&self.st[q as usize], j + 1) >= self.ks.b.tau[s as usize][(j + 1) as usize] {
                    j += 1;
                }
            }
            kq[q as usize] = j.max(lo);
            m += kq[q as usize] + 1;
        }
        while m != s * (k + 1) && !self.ks.b.width(s, self.k, m) {
            // a width the boot did not capture: the unverified draft most likely right joins (ties: the first in act)
            let (mut bq, mut bp) = (-1i32, -1.0f64);
            for &q in act {
                if kq[q as usize] < k {
                    let p = kprob(&self.st[q as usize], kq[q as usize] + 1);
                    if p > bp {
                        bp = p;
                        bq = q;
                    }
                }
            }
            kq[bq as usize] += 1;
            m += 1;
        }
    }

    /// GR8_MTP_LPLOG: one "# kstopb" line per shared round, after its verify
    #[allow(clippy::too_many_arguments)]
    fn klog(
        &self,
        act: &[i32],
        k: i32,
        xr: bool,
        kq: &[i32; MAX_SLOTS],
        aq: &[i32; MAX_SLOTS],
        nd: &[i32; MAX_SLOTS],
        base: &[i32; MAX_SLOTS],
    ) {
        let mut l = String::new();
        let mut hs: u32 = 0;
        for &q in act {
            let qu = q as usize;
            let x = &self.st[qu];
            cfmt!(l, " %d:%d:%d:%d:", q, kq[qu], aq[qu], nd[qu]);
            for j in 1..=nd[qu].max(0) as usize {
                let sep = cstr(if j > 1 { "," } else { "" });
                cfmt!(l, "%s%.6g", sep.as_ptr(), x.dlm[j] as f64);
            }
            cfmt!(l, "@%d", base[qu]);
            hs = hs.wrapping_mul(0x01000193) ^ SpecLen::hash(Some(&x.dlm), nd[qu], kq[qu]);
        }
        let ls = cstr(&l);
        let mut o = String::new();
        cfmt!(
            o,
            "# kstopb round %ld S %zu M %d k %d x %d sess%s hash %08x\n",
            self.cur_round,
            act.len(),
            self.sm.m(),
            k,
            xr as i32,
            ls.as_ptr(),
            hs
        );
        stderr_line(&o);
    }

    /// one round of solo DFlash2 session q (dflash.cuh dflash_sequence's round): verify k + 1 rows, accept, emit. Then
    /// the MTP first-form pass and the drafter step run over the a + 1 accepted rows. On MTP (x.mtp) the round verifies the
    /// MTP drafts, the drafter step takes the rows as context only, and the chain steps follow.
    fn round_dfl(&mut self, q: i32) -> Res<()> {
        let qu = q as usize;
        let (cur_round, cur_active) = (self.cur_round, self.cur_active);
        let base = self.st[qu].base();
        let m = self.st[qu].mtp;
        let mut pe = 0.0;
        let mut k = if m {
            if self.ks.mode != 0 {
                self.kverify(q, &mut pe)
            } else {
                self.st[qu].k
            }
        } else if self.dsl.on {
            self.dsl.choose(Some(&self.st[qu].dl), self.dkv, self.dkv, Some(&mut pe))
        } else {
            self.dkv
        };
        // the drafts this round verifies: MTP's, DFlash2's, or the n-gram table's chain (GR8_NGRAM)
        let mut dr = [0i32; 16];
        {
            let x = &self.st[qu];
            let src: &[i32] = if m { &x.d } else { &x.dd };
            let n = src.len().min(dr.len());
            dr[..n].copy_from_slice(&src[..n]);
        }
        let mut nd = [0i32; DFL_B];
        let (mut ne, mut de) = (0.0, 0.0);
        let nk = if self.ng.t.is_some() { self.ngram_pick(q, base, m, k, &mut nd, &mut ne, &mut de) } else { 0 };
        let kd = k;
        if nk != 0 {
            dr[..DFL_B].copy_from_slice(&nd);
            k = nk;
        }
        assert!((k as usize) < dr.len(), "s2: a round of more than 15 drafts");
        let ev = if self.st[qu].rounds == 0 { DECODE1 } else { ROUND };
        self.sm.begin(MAIN, cur_round, cur_active).event = ev;
        {
            let x = &self.st[qu];
            self.sm.open(q, ev);
            for r in 0..=k {
                let tok = if r > 0 { dr[r as usize] } else { x.tokens[base as usize] };
                self.sm.r.push(Row { sess: q, tok, pos: base + r, spos: base + r + 1, tgt: -1 });
            }
            self.sm.close();
        }
        self.sm.cls = row_class(if m && self.ks.mode == 0 && nk == 0 { self.k + 1 } else { k + 1 });
        self.check(&self.sm)?;
        run_step(&self.e, &self.sm, &mut self.tm)?;
        let mut a = 0;
        while a < k && dr[(a + 1) as usize] == self.tm[a as usize] {
            a += 1;
        }
        let (c, kept) = commit(&mut self.st[qu], &mut self.stats[qu], &self.tm[..(a + 1) as usize]);
        if nk == 0 && !m && self.dsl.on {
            self.dsl.observe(k, a, Some(&self.st[qu].dl), pe);
        }
        if nk == 0 && self.sel.on {
            self.score(q, k, a, pe);
        }
        if nk == 0 && self.ng.t.is_some() {
            let lp = ng_lp(&self.st[qu], m, self.ks.mode);
            if m {
                self.ngm.observe(k, a, lp, 0.0);
            } else {
                self.ngd.observe(k, a, lp, 0.0);
            }
        }
        if nk != 0 {
            self.ngram_score(q, m, kd, a, de);
            let s = &mut self.stats[qu];
            s.ngram_rounds += 1;
            s.ngram_drafted += k as i64;
            s.ngram_acc += a as i64;
            if self.ng.log {
                // hash: the chain's tokens and its length, then the drafter's log-probabilities (the decision's inputs)
                let mut tk = [0f32; DFL_B];
                for j in 1..=k as usize {
                    tk[j] = dr[j] as f32;
                }
                let lp = ng_lp(&self.st[qu], m, self.ks.mode);
                let h = SpecLen::hash(Some(&tk), k, k) ^ lp.map_or(0, |l| SpecLen::hash(Some(l), kd, kd));
                let dname = cstr(if m { "mtp" } else { "dfl" });
                let mut l = String::new();
                cfmt!(
                    l,
                    "# ngram round %ld slot %d base %d k %d a %d expected %.3f drafter %s k %d expected %.3f hash %08x\n",
                    cur_round,
                    q,
                    base,
                    k,
                    a,
                    ne,
                    dname.as_ptr(),
                    kd,
                    de,
                    h
                );
                stderr_line(&l);
            }
        }
        if self.ks.log && nk == 0 {
            // GR8_MTP_LPLOG: the verified drafts' log-probabilities (MTP: every drafted one), and MTP's d[1] with
            // whether the target took it (t[0])
            let x = &self.st[qu];
            let n = if m { x.k } else { k };
            let lp: &[f32] = if m { &x.dlm } else { &x.dl };
            let mut ls = String::new();
            for (j, &v) in lp.iter().enumerate().take(n.max(0) as usize + 1).skip(1) {
                let sep = cstr(if j > 1 { "," } else { "" });
                cfmt!(ls, "%s%.6g", sep.as_ptr(), v as f64);
            }
            let (lsc, dname) = (cstr(&ls), cstr(if m { "mtp" } else { "dfl" }));
            let mut l = String::new();
            cfmt!(
                l,
                "# kstop round %ld slot %d n %ld %s base %d k %d nd %d a %d lp %s m1 %.6g m1ok %d pe %.3f hash %08x\n",
                cur_round,
                q,
                self.stats[qu].rounds,
                dname.as_ptr(),
                base,
                k,
                n,
                a,
                lsc.as_ptr(),
                x.dlm[1] as f64,
                (x.d[1] == self.tm[0]) as i32,
                pe,
                SpecLen::hash(Some(lp), n, k)
            );
            stderr_line(&l);
        }
        if self.e.has_round() {
            self.e.round(q, self.stats[qu].rounds, base, a, k)?;
        }
        {
            let s = &mut self.stats[qu];
            s.rounds += 1;
            s.tokens += (a + 1) as i64;
            s.acc[a as usize] += 1;
            s.drafted += k as i64;
            if nk != 0 {
            } else if m {
                s.mtp_rounds += 1;
            } else {
                s.dfl_rounds += 1;
            }
        }
        self.st[qu].rounds += 1;
        {
            let x = &self.st[qu];
            let b1 = (base + 1) as usize;
            if self.e.has_emit() && self.e.emit(q, &x.tokens[b1..b1 + c as usize], 0)? {
                self.st[qu].done = true;
                return Ok(());
            }
        }
        // GR8_DRAFT_PAR: the drafter choice reads nothing the first form writes, so it is made first and the drafter
        // step runs under the first form
        let par = self.par && self.st[qu].force < 0;
        let (mut nm, mut probe) = (false, false);
        let tok_a = self.tm[a as usize];
        let mut r = DflReq::default();
        if par {
            r = self.choose_drafter(q, base, a, tok_a, &mut nm, &mut probe);
            r.phase = 1;
            self.drafts(q, &r)?;
        }
        let dn = kept.min(a) + 1; // rows whose input the session committed
        self.sd.begin(DRAFT, cur_round, cur_active).event = ev;
        {
            let x = &self.st[qu];
            let b1 = (base + 1) as usize;
            self.sd.draft_rows(q, ev, base, &x.tokens, b1, dn as usize);
        }
        self.sd.cls = row_class(self.sd.m());
        self.check(&self.sd)?;
        run_step(&self.e, &self.sd, &mut self.td)?;
        self.st[qu].d[1] = self.td[(dn - 1) as usize];
        let lp = self.dlp(dn - 1)?;
        self.st[qu].dlm[1] = lp;
        if self.st[qu].force >= 0 {
            // the insert cut this round: the drafter takes the rows as context, forced() drafts
            let lo = self.st[qu].lo;
            self.drafts(q, &DflReq { sess: q, rows: dn, pos0: base, tok0: 0, lo, query: false, phase: 0 })?;
            self.st[qu].pb = -1;
            return Ok(());
        }
        if !par {
            r = self.choose_drafter(q, base, a, tok_a, &mut nm, &mut probe);
            self.drafts(q, &r)?;
        } else {
            // the chain needs only d[1]: it runs before the wait, under the drafter step too
            if nm {
                self.kchain(q, ev)?;
            }
            r.phase = 2;
            self.drafts(q, &r)?;
        }
        let x = &mut self.st[qu];
        if probe {
            x.pb = base + a + 1;
            x.pd = x.dd;
            x.pl = x.dl;
        }
        if !nm {
            x.pb = -1;
        }
        if nm && !par {
            self.kchain(q, ev)?;
        }
        Ok(())
    }

    /// the drafter of the next round (drafter choice, GR8_DFL_REUSE warm-up) and this round's drafter step: the a + 1
    /// committed rows as context, drafting when the next round is on DFlash2 or probes
    fn choose_drafter(&mut self, q: i32, base: i32, a: i32, tok: i32, nm: &mut bool, probe: &mut bool) -> DflReq {
        let qu = q as usize;
        *nm = self.sel.on && self.next_mtp(qu);
        if self.ks.log && self.sel.on && self.sel.model != 0 {
            // GR8_MTP_LPLOG with DSel::model: the tokens per ms the choice compares after this round, and its MTP means
            let (rm, rd) = self.dsel_rates();
            let s = &self.sel;
            let next = cstr(if *nm { "mtp" } else { "dfl" });
            let mut l = String::new();
            cfmt!(
                l,
                "# dsel round %ld slot %d rm %.6f rd %.6f c1 %.4f mtp_rounds %.3f ms %.3f g %.4f next %s\n",
                self.cur_round,
                q,
                rm,
                rd,
                s.m.rate(1),
                s.rn,
                if s.rn > 0.0 { s.rc / s.rn } else { 0.0 },
                if s.an > 0.0 { s.ag / s.an } else { 0.0 },
                next.as_ptr()
            );
            stderr_line(&l);
        }
        let x = &mut self.st[qu];
        if x.warm != 0 {
            // GR8_DFL_REUSE warm-up: DFlash2 once it is over
            *nm = base + a + 1 < x.warm;
            if !*nm {
                x.warm = 0;
            }
        }
        if *nm != x.mtp {
            x.since = 0;
            self.stats[qu].switches += 1;
        }
        x.since += 1;
        x.mtp = *nm;
        *probe = *nm && x.warm == 0 && x.pb < 0 && {
            x.mn += 1;
            x.mn % self.sel.probe == 0
        };
        DflReq { sess: q, rows: a + 1, pos0: base, tok0: tok, lo: x.lo, query: !*nm || *probe, phase: 0 }
    }

    fn drafts(&mut self, q: i32, r: &DflReq) -> Res<()> {
        let o = self.e.dfl(r)?;
        if self.reuse.mode != 0 && r.phase != 2 {
            self.ring_rows(q, r);
        }
        if !r.query || r.phase == 1 {
            return Ok(());
        }
        let x = &mut self.st[q as usize];
        x.dd[1..DFL_B].copy_from_slice(&o.tok[1..DFL_B]);
        x.dl[1..DFL_B].copy_from_slice(&o.lp[1..DFL_B]);
        Ok(())
    }

    fn ring_rows(&mut self, q: i32, r: &DflReq) {
        if self.ring.q != q {
            self.ring.set(q, r.pos0, r.pos0, r.pos0 - 1);
        }
        if r.pos0 > self.ring.end {
            self.ring.lo = r.pos0; // a gap: the rows below are not this context's
        }
        self.ring.end = r.pos0 + r.rows;
        self.ring.top = self.ring.top.max(self.ring.end - 1 + if r.query { DFL_B as i32 } else { 0 });
    }

    /// GR8_NGRAM: the table's chain after tokens[0..=base] when it promises more tokens per verify ms than the
    /// drafter's k drafts: its length (0: the drafter's), the chain in nd[1..], the expected tokens of both
    #[allow(clippy::too_many_arguments)]
    fn ngram_pick(&self, q: i32, base: i32, m: bool, k: i32, nd: &mut [i32; DFL_B], ne: &mut f64, de: &mut f64) -> i32 {
        let x = &self.st[q as usize];
        let Some(t) = &self.ng.t else { return 0 };
        let mut c = [0.0f64; DFL_B];
        let n = t.chain(&x.tokens[..(base + 1) as usize], self.ng.kmax as usize, nd, &mut c) as i32;
        if n < self.ng.kmin {
            return 0;
        }
        let (mut bk, mut e, mut s, mut br) = (0, 1.0, 1.0, -1.0);
        for j in 1..=n {
            s *= c[j as usize];
            e += s;
            let r = e / self.dsl.verify_ms(j + 1);
            if j >= self.ng.kmin && r > br * (1.0 + 1e-9) {
                br = r;
                bk = j;
                *ne = e;
            }
        }
        let mut ed = 0.0;
        let est = if m { &self.ngm } else { &self.ngd };
        est.ratio(k, ng_lp(x, m, self.ks.mode), k, Some(&mut ed));
        *de = ed;
        if br > ed / self.dsl.verify_ms(k + 1) * (1.0 + self.ng.margin) {
            bk
        } else {
            0
        }
    }

    /// GR8_NGRAM: a round that verified the table's chain still teaches the drafter's estimates: its kd drafts against
    /// the a + 1 target tokens the verify produced (tm)
    fn ngram_score(&mut self, q: i32, m: bool, kd: i32, a: i32, de: f64) {
        let qu = q as usize;
        let x = &self.st[qu];
        let t = &self.tm;
        let dr: &[i32] = if m { &x.d } else { &x.dd };
        let lp = ng_lp(x, m, self.ks.mode);
        let mut b = 0;
        while b < kd && b < a + 1 && dr[(b + 1) as usize] == t[b as usize] {
            b += 1;
        }
        let ko = if b == a + 1 { b } else { (b + 1).min(kd) }; // known right ones, and the first miss
        if ko == 0 {
            return;
        }
        if m {
            self.ngm.observe(ko, b, lp, 0.0);
        } else {
            self.ngd.observe(ko, b, lp, 0.0);
        }
        if m {
            // as score() on an MTP round, with what the verify showed of MTP's drafts
            if self.sel.on || self.ks.mode == 2 {
                self.sel.m.observe(ko, b, lp, de);
            }
            if self.sel.on {
                self.probe_score(qu);
            }
            return;
        }
        if self.dsl.on {
            self.dsl.observe(ko, b, lp, de);
        }
        if self.sel.on {
            // as score() on a DFlash2 round: MTP's first draft (t[0] is always known) and DFlash2's drafts
            self.sel.d.observe(ko, b, lp, 0.0);
            let ml = if self.ks.mode != 0 { Some(&x.dlm[..]) } else { None };
            self.sel.m.observe(1, (x.d[1] == t[0]) as i32, ml, 0.0);
            self.dfl_obs(b.min(kd) + 1, kd);
        }
    }

    /// the chain of a solo session about to verify MTP drafts: K-1 steps, or with KStop while kmore() allows
    fn kchain(&mut self, q: i32, ev: i32) -> Res<()> {
        if self.ks.mode == 0 {
            return self.chain(&[q], &[ev], self.k, 1);
        }
        let mut n = 1;
        while n < self.kmax() && self.kmore(q, n) {
            self.chain(&[q], &[ev], n + 1, n)?;
            n += 1;
        }
        self.st[q as usize].k = n;
        Ok(())
    }
    fn kmax(&self) -> i32 {
        if self.ks.kmax != 0 {
            self.ks.kmax
        } else {
            self.k
        }
    }
    /// KStop::solo: a round_all of n sessions is a solo MTP-only round (mode 1)
    fn ksolo(&self, n: usize) -> bool {
        self.ks.solo && self.ks.mode == 1 && n == 1
    }
    /// KStop::solo, before a round of several sessions: a session whose chain stopped short drafts up to K
    fn complete(&mut self, act: &[i32]) -> Res<()> {
        for &q in act {
            let x = &self.st[q as usize];
            if x.k < self.k && x.force < 0 {
                let j0 = x.k;
                self.chain(&[q], &[ROUND], self.k, j0)?;
            }
        }
        Ok(())
    }
    /// KStop: one more chain step after n drafts (mode 1: draft n + 1 could still pass, and drafts 1..n do)
    fn kmore(&self, q: i32, n: i32) -> bool {
        let x = &self.st[q as usize];
        if self.ks.mode == 2 {
            return self.sel.m.more(&x.dlm, n);
        }
        for j in 1..=n {
            if kprob(x, j) < self.ks.tau[j as usize] {
                return false;
            }
        }
        kprob(x, n) >= self.ks.tau[(n + 1) as usize]
    }
    /// KStop: the drafts of x.d[1..x.k] to verify (at least one, at most kmax)
    fn kverify(&self, q: i32, pe: &mut f64) -> i32 {
        let x = &self.st[q as usize];
        let n = x.k.min(self.kmax());
        if self.ks.mode == 2 {
            return self.sel.m.choose(Some(&x.dlm), n, n, Some(pe));
        }
        let mut k = 0;
        while k < n && kprob(x, k + 1) >= self.ks.tau[(k + 1) as usize] {
            k += 1;
        }
        k.max(1)
    }
    fn dlp(&self, row: i32) -> Res<f32> {
        if self.e.has_lp() && (self.ks.mode != 0 || self.ks.b.mode != 0 || self.ks.log) {
            self.e.lp(row)
        } else {
            Ok(f32::NAN)
        }
    }

    /// GR8_DFL_REUSE, a DFlash2 session's start: the drafter's context starts at the lowest position the ring still
    /// holds for this prompt. With fewer than reuse.min rows the session drafts with MTP until the drafter holds that
    /// many
    fn reuse_start(&mut self, q: i32) {
        let qu = q as usize;
        let (n, from) = (self.st[qu].n, self.st[qu].lo);
        let wlo = (n - 1 - DFL_WIN).max(0);
        if from > wlo {
            let t: &[i32] = if self.ring.q >= 0 { &self.st[self.ring.q as usize].tokens } else { &self.ring.tok };
            let a = wlo.max(self.ring.lo).max(self.ring.top - DFL_RING + 1);
            let xt = &self.st[qu].tokens;
            let fu = from as usize;
            if a < from && self.ring.end >= from && t.len() >= fu && xt.len() >= fu && t[..fu] == xt[..fu] {
                self.st[qu].lo = a;
            }
        }
        let lo = self.st[qu].lo;
        self.ring.set(q, lo, n - 1, n - 2);
        let x = &mut self.st[qu];
        if self.reuse.min != 0 && x.lo > 0 && n - x.lo < self.reuse.min {
            x.warm = x.lo + self.reuse.min;
            x.mtp = true;
        }
        let s = &mut self.stats[qu];
        s.dfl_from = from;
        s.dfl_lo = x.lo;
        s.dfl_warm = x.warm;
    }

    /// drafter choice: teach the estimates what this round's tokens tm[0..=a] say about each drafter
    fn score(&mut self, q: i32, k: i32, a: i32, pe: f64) {
        let qu = q as usize;
        let x = &self.st[qu];
        let ml = if self.ks.mode != 0 { Some(&x.dlm[..]) } else { None }; // KStop: MTP's confidence buckets
        if !x.mtp {
            self.sel.d.observe(k, a, Some(&x.dl), 0.0);
            self.sel.m.observe(1, (x.d[1] == self.tm[0]) as i32, ml, 0.0);
            self.dfl_obs(a + 1, k);
            return;
        }
        self.sel.m.observe(k, a, ml, pe);
        if self.sel.model != 0 {
            // the round as drafted (x.k) and verified (k), at the table's cost: each rank's clock differs
            let f = self.sel.m.pos_decay;
            let cost = self.mtp_cost(k, x.k);
            let s = &mut self.sel;
            s.rn = s.rn * f + 1.0;
            s.rc = s.rc * f + cost;
            s.an *= f;
            s.ag *= f;
            if a >= 1 {
                s.an += 1.0;
                s.ag += (a - 1) as f64;
            }
            s.dn *= s.tab.r;
            s.dt *= s.tab.r;
            s.dc *= s.tab.r;
        }
        self.probe_score(qu);
    }

    /// tab.r: a DFlash2 round (or a scored probe) that committed c tokens verifying k drafts, at the table's cost
    fn dfl_obs(&mut self, c: i32, k: i32) {
        let s = &mut self.sel;
        if s.model == 0 || s.tab.r.is_nan() || s.tab.r <= 0.0 {
            return;
        }
        let f = s.d.pos_decay;
        s.dn = s.dn * f + 1.0;
        s.dt = s.dt * f + c as f64;
        s.dc = s.dc * f + s.tab.verify((k + 1) as f64) + s.tab.d;
    }

    /// a probe's drafts pd[j] are for positions pb + j. It scores them once a mismatch or all DFL_B - 1 are committed
    fn probe_score(&mut self, qu: usize) {
        let x = &self.st[qu];
        if x.pb < 0 {
            return;
        }
        let have = x.tokens.len() as i32 - 1 - x.pb;
        let n = DFL_B as i32 - 1;
        let mut b = 0;
        while b < n && b < have && x.pd[(b + 1) as usize] == x.tokens[(x.pb + b + 1) as usize] {
            b += 1;
        }
        if b < n && b == have {
            return; // every committed position matched so far: wait for more
        }
        self.sel.d.observe((b + 1).min(n), b, Some(&x.pl), 0.0);
        let dkv = self.dkv;
        self.dfl_obs(b.min(dkv) + 1, dkv); // as a round verifying K drafts would have
        self.st[qu].pb = -1;
    }

    /// the drafter for the next round: switch after dwell rounds when the other one's tokens per ms is margin higher
    fn next_mtp(&self, qu: usize) -> bool {
        let x = &self.st[qu];
        if x.since < self.sel.dwell {
            return x.mtp;
        }
        let (rm, rd) = self.dsel_rates();
        if x.mtp {
            rm * (1.0 + self.sel.margin) >= rd
        } else {
            rm > rd * (1.0 + self.sel.margin)
        }
    }

    /// the drafter choice's tokens per ms of an MTP round and of a DFlash2 round, as next_mtp compares them
    pub fn dsel_rates(&self) -> (f64, f64) {
        let s = &self.sel;
        let mut rd = 0.0;
        if s.model != 0 && s.tab.r > 0.0 && s.dc > 0.0 {
            rd = s.dt / s.dc; // tab.r: the rounds as they ran
        } else if self.dsl.on {
            for &k in self.dsl.levels() {
                rd = dmax(rd, s.d.ratio(k, None, k, None));
            }
        } else {
            rd = s.d.ratio(self.dkv, None, self.dkv, None);
        }
        (if s.model != 0 { self.mtp_rate() } else { s.m.ratio(self.k, None, self.k, None) }, rd)
    }

    /// DSel::model: an MTP round's tokens per ms under the confidence stop (s2_batch.h mtp_rate)
    fn mtp_rate(&self) -> f64 {
        let t = &self.sel.tab;
        let w = self.sel.prior;
        let mut g0 = t.g;
        if g0 < 0.0 {
            g0 = 0.0;
            let mut s = 1.0;
            for j in 2..=self.k.max(0) as usize {
                s *= self.sel.m.seed[j];
                g0 += s;
            }
        }
        let k0 = if t.k > 0.0 { t.k } else { self.k as f64 };
        let n0 = if t.n > 0.0 { t.n } else { dmax(self.k as f64, k0) };
        let c0 = t.verify(k0 + 1.0) + t.m + t.e * (n0 - 1.0);
        let g = (self.sel.ag + w * g0) / (self.sel.an + w);
        let c = (self.sel.rc + w * c0) / (self.sel.rn + w);
        (1.0 + self.sel.m.rate(1) * (1.0 + g)) / (c + t.p / self.sel.probe as f64)
    }
    fn mtp_cost(&self, k: i32, nd: i32) -> f64 {
        let t = &self.sel.tab;
        t.verify((k + 1) as f64) + t.m + t.e * (nd - 1).max(0) as f64
    }
    /// DSel::model: the table's costs in the drafter choice's estimates
    fn price(&mut self) {
        if self.sel.model == 0 {
            return;
        }
        let t = self.sel.tab;
        t.price(&mut self.sel.d, t.d, 0.0);
        t.price(&mut self.sel.m, t.m, t.e);
    }
    /// the drafter choice learns each request afresh. Its settings, the verify table included, stay
    fn restart(&mut self) {
        let s = &mut self.sel;
        s.d = s.d.fresh();
        s.m = s.m.fresh();
        s.rn = 0.0;
        s.rc = 0.0;
        s.an = 0.0;
        s.ag = 0.0;
        s.dn = 0.0;
        s.dt = 0.0;
        s.dc = 0.0;
    }
    /// each request starts from the seeds, with dsl's verify table
    fn ng_fresh(&mut self) {
        self.ngd = self.dsl.fresh();
        self.ngm = self.dsl.fresh();
        self.ngd.on = false;
        self.ngm.on = false;
    }
    /// the draft count for the next round of sessions qs: the k with the most expected tokens per ms of the round
    fn choose(&self, qs: &[i32]) -> i32 {
        if !self.adapt || qs.is_empty() {
            return self.k;
        }
        let (mut best, mut br) = (self.k, -1.0);
        for k in 1..=self.k {
            let mut e = 0.0;
            for &q in qs {
                let mut s = 1.0;
                e += 1.0;
                for j in 1..=k as usize {
                    s *= self.sl[q as usize].rate(j);
                    e += s;
                }
            }
            let r = e / self.cost.round(qs.len() as i32, k);
            if r > br * (1.0 + 1e-9) {
                br = r;
                best = k;
            }
        }
        best
    }

    fn check(&self, s: &StepBuf) -> Res<()> {
        if s.m() > self.max_rows {
            return Err(Stop::Refused(format!("s2: step of {} rows > max rows {}", s.m(), self.max_rows)));
        }
        Ok(())
    }

    /// chain steps for the sessions in qs, each holding d[1..=j0]. They fill d[j0+1..=k]
    fn chain(&mut self, qs: &[i32], evs: &[i32], k: i32, j0: i32) -> Res<()> {
        for &q in qs {
            self.st[q as usize].k = k;
        }
        let (cur_round, cur_active) = (self.cur_round, self.cur_active);
        let mut j = j0;
        while j < k && !qs.is_empty() {
            self.sc.begin(CHAIN, cur_round, cur_active);
            for (i, &q) in qs.iter().enumerate() {
                let x = &self.st[q as usize];
                let base = x.base();
                let ju = j as usize;
                self.sc.draft_rows(q, evs[i], base - 1 + j, &x.d, ju, 1);
            }
            self.sc.cls = row_class(self.sc.m());
            self.check(&self.sc)?;
            run_step(&self.e, &self.sc, &mut self.tc)?;
            for i in 0..self.sc.sess.len() {
                let row = self.sc.row0[i];
                let lp = self.dlp(row)?;
                let x = &mut self.st[self.sc.sess[i] as usize];
                x.d[(j + 1) as usize] = self.tc[row as usize];
                x.dlm[(j + 1) as usize] = lp;
            }
            j += 1;
        }
        Ok(())
    }

    /// the state setup shared by solo prompt() and the packed final row
    fn start_prompt(&mut self, q: i32, p: &SessIn) -> Res<()> {
        let qu = q as usize;
        if self.ring.q == q {
            // the slot's last session wrote the drafter's ring: keep its tokens
            let x = &self.st[qu];
            let n = (x.tokens.len() as i64).min(self.ring.end as i64).max(0) as usize;
            self.ring.tok.clear();
            self.ring.tok.extend_from_slice(&x.tokens[..n]);
            self.ring.q = -1;
        }
        let dsize = self.k.max(self.kmax()).max(1) as usize + 1; // K = 0 (DS41): d[1] still exists and stays unwritten
        let x = &mut self.st[qu];
        x.reset();
        x.first = p.first;
        x.chunk = p.chunk;
        x.n_gen = p.n_gen;
        x.dflash = p.dflash;
        x.ins_after = p.ins_after;
        x.ins_unless = p.ins_unless;
        x.ins_tokens.extend_from_slice(p.ins_tokens);
        x.on = true;
        x.tokens.extend_from_slice(p.prompt);
        x.n = p.prompt.len() as i32;
        x.d.resize(dsize, 0);
        x.dlm.resize(dsize, f32::NAN);
        self.sl[qu] = SpecLen::default();
        if p.dflash && self.fresh_spec {
            self.dsl = self.dsl.fresh(); // GR8_S2_FRESH_SPEC
        }
        self.stats[qu].reset();
        let x = &mut self.st[qu];
        x.dfl = p.dflash;
        x.armed = p.ins_after >= 0;
        let dfl = x.dfl;
        if dfl && self.sel.on {
            self.restart(); // the drafter choice learns each request afresh
        }
        if dfl && self.ng.t.is_some() {
            self.ng_fresh();
        }
        if dfl && !(self.e.has_dfl() && self.e.has_dfl_start()) {
            return Err(Stop::Refused("s2: DFlash2 session without drafter hooks".into()));
        }
        let na = self.k.max(if dfl { DFL_B as i32 - 1 } else { 0 }) as usize + 1;
        self.stats[qu].acc.resize(na, 0);
        self.st[qu].dl = [f32::NAN; DFL_B];
        let lo = if dfl { self.e.dfl_start(q)? } else { 0 };
        self.st[qu].lo = lo;
        if dfl && self.reuse.mode != 0 {
            self.reuse_start(q);
        }
        Ok(())
    }

    /// the prompt steps of session q as its own steps (rank.cuh mtp_sequence, prompt loop); false = finished by emit
    fn prompt(&mut self, q: i32, p: &SessIn, allow_mtp: bool) -> Res<bool> {
        let qu = q as usize;
        self.start_prompt(q, p)?;
        let n = self.st[qu].n;
        let ps = (n - p.first + p.chunk - 1) / p.chunk;
        for s in 0..ps {
            let left = n - p.first - s * p.chunk;
            let m = if left < p.chunk { left } else { p.chunk };
            let event = if s == ps - 1 { TAIL } else { ROUND };
            let pos0 = p.first + s * p.chunk;
            self.sm.begin(MAIN, -1, 0).event = event;
            self.sm.cls = row_class(m);
            self.sm.open(q, event);
            for r in 0..m {
                let pos = pos0 + r;
                let tgt = if pos + 1 < n { p.prompt[(pos + 1) as usize] } else { -1 };
                self.sm.r.push(Row { sess: q, tok: p.prompt[pos as usize], pos, spos: pos + 1, tgt });
            }
            self.sm.close();
            self.check(&self.sm)?;
            run_step(&self.e, &self.sm, &mut self.tm)?;
            if self.e.has_prompt() {
                self.e.prompt(q, m, pos0)?;
            }
            if s == ps - 1 {
                let mu = m as usize;
                let (c, _) = commit(&mut self.st[qu], &mut self.stats[qu], &self.tm[mu - 1..mu]);
                let x = &self.st[qu];
                if self.e.has_emit() && self.e.emit(q, &x.tokens[n as usize..(n + c) as usize], m - 1)? {
                    self.st[qu].done = true;
                    return Ok(false);
                }
            }
            if allow_mtp && self.k > 0 {
                // K = 0 or S > mtp_max_sess(): no draft step
                self.sd.begin(DRAFT, -1, 0).event = event;
                self.sd.cls = row_class(m);
                let x = &self.st[qu];
                let p1 = (pos0 + 1) as usize;
                self.sd.draft_rows(q, event, pos0, &x.tokens, p1, m as usize);
                self.check(&self.sd)?;
                run_step(&self.e, &self.sd, &mut self.td)?;
                self.st[qu].d[1] = self.td[(m - 1) as usize];
                let lp = self.dlp(m - 1)?;
                self.st[qu].dlm[1] = lp;
            }
            let x = &self.st[qu];
            let query = s == ps - 1 && x.force < 0 && x.warm == 0;
            if x.dfl {
                let r = DflReq {
                    sess: q,
                    rows: m,
                    pos0,
                    tok0: if query { self.tm[(m - 1) as usize] } else { 0 },
                    lo: x.lo,
                    query,
                    phase: 0,
                };
                self.drafts(q, &r)?;
            }
        }
        Ok(true)
    }

    /// the inserted tokens (positions x.force on) as session q's own steps, as its prompt steps run. Then the chain
    /// runs, unless the session drafts with DFlash2. The emit may finish the session
    fn forced(&mut self, q: i32) -> Res<()> {
        let qu = q as usize;
        let (cur_round, cur_active) = (self.cur_round, self.cur_active);
        let (from, end, ev, chunk) = {
            let x = &mut self.st[qu];
            let r = (x.force, x.tokens.len() as i32, if x.rounds == 0 { DECODE1 } else { ROUND }, x.chunk);
            x.force = -1;
            r
        };
        let mut p0 = from;
        while p0 < end {
            let m = chunk.min(end - p0);
            let last = p0 + m == end;
            self.sm.begin(MAIN, cur_round, cur_active).event = ev;
            self.sm.cls = row_class(m);
            {
                let x = &self.st[qu];
                self.sm.open(q, ev);
                for r in 0..m {
                    let p = p0 + r;
                    self.sm.r.push(Row { sess: q, tok: x.tokens[p as usize], pos: p, spos: p + 1, tgt: -1 });
                }
                self.sm.close();
            }
            self.check(&self.sm)?;
            run_step(&self.e, &self.sm, &mut self.tm)?;
            let mu = m as usize;
            if last {
                let t = self.tm[mu - 1];
                self.st[qu].tokens.push(t);
                if self.e.has_emit() && self.e.emit(q, &self.tm[mu - 1..mu], m - 1)? {
                    self.st[qu].done = true;
                    return Ok(());
                }
            }
            if self.k > 0 {
                // K = 0 (DS41): no draft head, no draft step
                self.sd.begin(DRAFT, cur_round, cur_active).event = ev;
                self.sd.cls = row_class(m);
                let x = &self.st[qu];
                let p1 = (p0 + 1) as usize;
                self.sd.draft_rows(q, ev, p0, &x.tokens, p1, mu);
                self.check(&self.sd)?;
                run_step(&self.e, &self.sd, &mut self.td)?;
                self.st[qu].d[1] = self.td[mu - 1];
                let lp = self.dlp(m - 1)?;
                self.st[qu].dlm[1] = lp;
            }
            if self.st[qu].dfl {
                let r = DflReq {
                    sess: q,
                    rows: m,
                    pos0: p0,
                    tok0: if last { self.tm[mu - 1] } else { 0 },
                    lo: self.st[qu].lo,
                    query: last,
                    phase: 0,
                };
                self.drafts(q, &r)?;
            }
            p0 += chunk;
        }
        if !self.st[qu].dfl {
            self.chain(&[q], &[ev], self.k, 1)?;
        } else if self.st[qu].mtp {
            self.kchain(q, ev)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// the token at position pos + 1; MTP drafts are always right
    fn target(pos: i32) -> i32 {
        (pos * 37 + 11) % 997
    }

    #[derive(Default)]
    struct Fake {
        steps: RefCell<Vec<(i32, i32, i32)>>, // kind, rows, class
        got: RefCell<Vec<i32>>,
    }

    impl Engine for Fake {
        fn run(&self, s: &StepBuf, out: &mut [i32]) -> Res<()> {
            self.steps.borrow_mut().push((s.kind, s.m(), s.cls));
            for (o, r) in out.iter_mut().zip(&s.r) {
                *o = target(if s.kind == MAIN { r.pos } else { r.spos - 1 });
            }
            Ok(())
        }
        fn has_round(&self) -> bool {
            false
        }
        fn round(&self, _: i32, _: i64, _: i32, _: i32, _: i32) -> Res<()> {
            Ok(())
        }
        fn has_prompt(&self) -> bool {
            false
        }
        fn prompt(&self, _: i32, _: i32, _: i32) -> Res<()> {
            Ok(())
        }
        fn has_emit(&self) -> bool {
            true
        }
        fn emit(&self, _: i32, toks: &[i32], _: i32) -> Res<bool> {
            self.got.borrow_mut().extend_from_slice(toks);
            Ok(false)
        }
        fn has_dfl(&self) -> bool {
            false
        }
        fn dfl(&self, _: &DflReq) -> Res<DflDrafts> {
            Err(Stop::Hook)
        }
        fn has_dfl_start(&self) -> bool {
            false
        }
        fn dfl_start(&self, _: i32) -> Res<i32> {
            Err(Stop::Hook)
        }
        fn has_lp(&self) -> bool {
            false
        }
        fn lp(&self, _: i32) -> Res<f32> {
            Ok(f32::NAN)
        }
    }

    fn session(prompt: &[i32], n_gen: i32) -> SessIn<'_> {
        SessIn {
            prompt,
            first: prompt.len() as i32 - 1,
            chunk: 4,
            n_gen,
            dflash: false,
            ins_after: -1,
            ins_unless: -1,
            ins_tokens: &[],
        }
    }

    #[test]
    fn a_round_is_one_main_one_draft_and_the_chain() {
        let prompt: Vec<i32> = (0..8).map(|p| target(p - 1)).collect();
        let mut b = Batch::new(3, 32, 8, false, false, Cost::default(), Fake::default());
        b.admit(&[2], &[session(&prompt, 12)]).unwrap();
        // the prompt tail's main step and first-form draft, then the chain to K drafts
        assert_eq!(*b.e.steps.borrow(), vec![(MAIN, 1, 1), (DRAFT, 1, 1), (CHAIN, 1, 1), (CHAIN, 1, 1)]);
        b.e.steps.borrow_mut().clear();
        assert!(b.round(&[0, 2]).unwrap());
        // every draft is right: verify K + 1 rows, draft the a + 1 = 4 accepted rows, chain K - 1
        assert_eq!(*b.e.steps.borrow(), vec![(MAIN, 4, 5), (DRAFT, 4, 5), (CHAIN, 1, 1), (CHAIN, 1, 1)]);
        assert_eq!(b.length(2), 8 + 1 + 4);
        while b.round(&[2]).unwrap() {}
        let want: Vec<i32> = (7..19).map(target).collect();
        assert_eq!(b.result(2), &want[..]);
        assert!(b.finished(2) && !b.active(2));
        assert_eq!(b.stats[2].acc[3], b.stats[2].rounds);
        assert!(!b.round(&[2]).unwrap());
    }

    #[test]
    fn refusals_leave_a_message() {
        let prompt = [1, 2, 3];
        let mut b = Batch::new(3, 4, 8, false, false, Cost::default(), Fake::default());
        assert_eq!(
            b.admit(&[9], &[session(&prompt, 4)]),
            Err(Stop::Refused("s2: admit: slot 9 is out of range or in use".into()))
        );
        b.admit(&[0], &[session(&prompt, 4)]).unwrap();
        b.admit(&[1], &[session(&prompt, 4)]).unwrap();
        assert_eq!(b.round(&[0, 1]), Err(Stop::Refused("s2: 2 sessions need 8 rows > max rows 4".into())));
    }

    #[test]
    fn an_insert_at_the_last_token_drafts_past_the_reply() {
        // the rule fires exactly at n_gen: nothing is inserted, and the verify's draft step reads one token past the
        // committed ones: 0 here. The C++ planner read past its vector
        let prompt: Vec<i32> = (0..4).map(|p| target(p - 1)).collect();
        let mut b = Batch::new(3, 32, 8, false, false, Cost::default(), Fake::default());
        let mut s = session(&prompt, 3);
        s.ins_after = 3;
        s.ins_tokens = &[5, 6];
        b.admit(&[0], &[s]).unwrap();
        assert!(b.round(&[0]).unwrap());
        assert_eq!(b.result(0).len(), 3);
        assert_eq!(b.stats[0].inserted, 0);
    }

    /// counts the heap allocations of the thread that set COUNTING
    mod counting {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;
        thread_local! {
            pub static COUNTING: Cell<bool> = const { Cell::new(false) };
            pub static ALLOCS: Cell<u64> = const { Cell::new(0) };
        }
        pub struct Counter;
        unsafe impl GlobalAlloc for Counter {
            unsafe fn alloc(&self, l: Layout) -> *mut u8 {
                if COUNTING.with(|c| c.get()) {
                    ALLOCS.with(|a| a.set(a.get() + 1));
                }
                System.alloc(l)
            }
            unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
                System.dealloc(p, l)
            }
        }
        #[global_allocator]
        static A: Counter = Counter;
    }

    /// an engine that allocates nothing: every MTP draft right
    struct Quiet;
    impl Engine for Quiet {
        fn run(&self, s: &StepBuf, out: &mut [i32]) -> Res<()> {
            for (o, r) in out.iter_mut().zip(&s.r) {
                *o = target(if s.kind == MAIN { r.pos } else { r.spos - 1 });
            }
            Ok(())
        }
        fn has_round(&self) -> bool {
            false
        }
        fn round(&self, _: i32, _: i64, _: i32, _: i32, _: i32) -> Res<()> {
            Ok(())
        }
        fn has_prompt(&self) -> bool {
            false
        }
        fn prompt(&self, _: i32, _: i32, _: i32) -> Res<()> {
            Ok(())
        }
        fn has_emit(&self) -> bool {
            false
        }
        fn emit(&self, _: i32, _: &[i32], _: i32) -> Res<bool> {
            Ok(false)
        }
        fn has_dfl(&self) -> bool {
            false
        }
        fn dfl(&self, _: &DflReq) -> Res<DflDrafts> {
            Err(Stop::Hook)
        }
        fn has_dfl_start(&self) -> bool {
            false
        }
        fn dfl_start(&self, _: i32) -> Res<i32> {
            Err(Stop::Hook)
        }
        fn has_lp(&self) -> bool {
            false
        }
        fn lp(&self, _: i32) -> Res<f32> {
            Ok(f32::NAN)
        }
    }

    #[test]
    fn rounds_do_not_allocate() {
        let prompt: Vec<i32> = (0..8).map(|p| target(p - 1)).collect();
        let mut b = Batch::new(3, 32, 8, false, false, Cost::default(), Quiet);
        b.set_adapt(true);
        let ss = [session(&prompt, 100_000), session(&prompt, 100_000), session(&prompt, 100_000)];
        b.admit(&[0, 3, 5], &ss).unwrap();
        for q in [0, 3, 5] {
            b.st[q].tokens.reserve(100_000); // the committed tokens grow by doubling: leave that out
        }
        let order = [0, 3, 5];
        for _ in 0..50 {
            b.round(&order).unwrap(); // the step buffers reach their size (the adaptive count widens rounds as it learns)
        }
        counting::ALLOCS.with(|a| a.set(0));
        counting::COUNTING.with(|c| c.set(true));
        for _ in 0..500 {
            b.round(&order).unwrap();
        }
        counting::COUNTING.with(|c| c.set(false));
        assert_eq!(counting::ALLOCS.with(|a| a.get()), 0);
        assert_eq!(b.stats[3].rounds, 550);
    }
}
