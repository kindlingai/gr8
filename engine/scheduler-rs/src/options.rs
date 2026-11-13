//! g6m4's option parse (engine_options.h, docs/HOST-CORE.md R6): argv, the knob table and the build's description
//! (module_info.h) in; the refusal main dies with, or every option main reads afterwards. Each check, its message,
//! their order and the lines printed to stderr on the way are the C++'s; s2/tests/options_vectors_test.cpp holds them
//! to transcripts of it. A refusal is a message; the helpers that die (an unreadable @file, the Engram knobs) still
//! exit the process as the boot does.
//!
//! Knobs come through the knob table's C API (gr8_knobs.h) as function pointers, so this library needs nothing from
//! libgr8knobs.a at link time. Two pieces stay C++ and arrive in `Host`: the DSpark knob parse (ds41_dspark.h
//! parse_knobs, which DS41's serve path and tests also call) and the Engram collective count (ds41_engram_boot.h).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::io::Write;

use crate::cstd::{c_atof, c_atoi, c_strtol10, c_strtoul10, c_strtoull10, cs, read_ints};
use crate::ffi::Ints;
use crate::knob_ids as kid;
use crate::module_info::{ModuleInfo, MODEL_DS41, MODEL_GLM53F, MODULE_ABI};
use crate::spec_len::SpecLen;

pub type FlagFn = unsafe extern "C" fn(u32, u32, *mut c_int) -> c_int;
pub type IntFn = unsafe extern "C" fn(u32, u32, *mut i64) -> c_int;
pub type FloatFn = unsafe extern "C" fn(u32, u32, *mut f64) -> c_int;
pub type TextFn = unsafe extern "C" fn(u32, u32) -> *const c_char;
pub type IntsFn = unsafe extern "C" fn(u32, u32, *mut *const i64) -> isize;
pub type FloatsFn = unsafe extern "C" fn(u32, u32, *mut *const f64) -> isize;
pub type StrFn = unsafe extern "C" fn() -> *const c_char;
pub type EngramFn = unsafe extern "C" fn(*mut c_void, *const i32, usize) -> i32;

/// g6opt_knobs: gr8_knobs.h's C API and the hash of the table its header was written from
#[repr(C)]
pub struct KnobApi {
    pub table_hash: u64,
    pub flag: Option<FlagFn>,
    pub int_: Option<IntFn>,
    pub float_: Option<FloatFn>,
    pub text: Option<TextFn>,
    pub ints: Option<IntsFn>,
    pub floats: Option<FloatsFn>,
    pub tests_set: Option<StrFn>,
}

/// g6opt_host: what stays C++. dsk_*: ds41_dspark.h parse_knobs's result (DS41; it reads knobs only, so the C++
/// runs it first). engram_colls: ds41_engram_colls(layers), which dies on a bad Engram config.
#[repr(C)]
pub struct Host {
    pub ctx: *mut c_void,
    pub engram_colls: Option<EngramFn>,
    pub dsk_mtp: i32,
    pub dsk_relaxed: bool,
    pub dsk_relax_topp: f64,
    pub dsk_err: *const c_char,
}

/// g6opt_strs
#[repr(C)]
pub struct Strs {
    pub p: *const *const c_char,
    pub n: usize,
}

/// g6opt_plan: rank_host.h Plan's fields the parse sets (the steering ones stay their defaults)
#[repr(C)]
pub struct CPlan {
    pub prompt: Ints,
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub inv_temp: f32,
    pub seed: u32,
    pub top_p: f32,
    pub lp: i32,
    pub plp: i32,
    pub rs: bool,
}

/// g6opt_options: engine_options.h EngineOptions as scalars, its strings and lists in the parse's storage. The
/// FabricConfig, CommonOpts, V2Src, SpecLen and KStop fields the parse leaves at their defaults are not here.
#[repr(C)]
pub struct Options {
    pub rank: i32,
    pub ips: Strs,
    pub ips2: Strs,
    pub ring: bool,
    pub ring_devs: Strs,
    pub proxy_cpu: i32,
    pub pf_ips2: Strs,
    pub manifest: *const c_char,
    pub packfile: *const c_char,
    pub v2_dir: *const c_char,
    pub v2_slices: *const c_char,
    pub v2_widths: *const c_char,
    pub v2_dense: *const c_char,
    pub v2_layout: *const c_char,
    pub v2_trellis: bool,
    pub qo_man: *const c_char,
    pub qo_file: *const c_char,
    pub layers_s: *const c_char,
    pub levels_s: *const c_char,
    pub prompts: Strs,
    pub co_cap: i32,
    pub co_mp: i32,
    pub co_top: i32,
    pub co_pfrows: i32,
    pub co_mtp: i32,
    pub co_lanes: i32,
    pub co_nc: i32,
    pub co_temp: f32,
    pub co_graph: bool,
    pub co_wide: bool,
    pub co_pfv2: bool,
    pub co_adapt: bool,
    pub has_n_gen: bool,
    pub n_gen: i32,
    pub has_chunk: bool,
    pub chunk: i32,
    pub has_seed: bool,
    pub seed: u32,
    pub has_top_p: bool,
    pub top_p: f32,
    pub has_rs: bool,
    pub rs: bool,
    pub qp_man: *const c_char,
    pub qp_file: *const c_char,
    pub dfl_man: *const c_char,
    pub dfl_file: *const c_char,
    pub dfl_sel: i32,
    pub dfl_k: i32,
    pub spec_reset_per_request: bool,
    pub spec_fixed: bool,
    pub plan: CPlan,
    pub image: bool,
    pub hash: bool,
    pub reuse_prefix: bool,
    pub check_args: bool,
    pub lanes_explicit: bool,
    pub s2dir: *const c_char,
    pub grammar_dir: *const c_char,
    pub kv_tiers: Strs,
    pub kv_keep: *const c_char,
    pub kv_system_after: i32,
    pub kv_write_limit: f64,
    pub kv_format: *const c_char,
    pub kv_old: bool,
    pub kv_old_all: bool,
    pub set_id: *const c_char,
    pub steering_dir: *const c_char,
    pub steering_force_s: *const c_char,
    pub steering_model_revision: *const c_char,
    pub steering_tokenizer_revision: *const c_char,
    pub steering_template_digest: *const c_char,
    pub steering_weight_set: *const c_char,
    pub steering_template_file: *const c_char,
    pub s2port: i32,
    pub s2sess: i32,
    pub s2epoch: u64,
    pub cap: i32,
    pub top: i32,
    pub pfrows: i32,
    pub nc: i32,
    pub mtp: i32,
    pub mp: i32,
    pub lanes: i32,
    pub temp: f32,
    pub graph: bool,
    pub wide: bool,
    pub pfv2: bool,
    pub adapt: bool,
    pub release: bool,
    pub s2mode: bool,
    pub dfl: bool,
    pub k: i32,
    pub layers: Ints,
    pub progress_isolate: bool,
    pub route_split_mode: i32,
    pub c8_pack_env: i32,
    pub spec_on: bool,
    pub spec_levels: Ints,
    pub spec_draft_round: f64,
    pub spec_draft_each: f64,
    pub spec_table: [f64; 9],
    pub rt_win: i32,
    pub kstop_mode: i32,
    pub kstop_kmax: i32,
    pub kstop_tau: [f64; 4],
    pub kstop_seed3: f64,
    pub kstop_log: bool,
    pub kstop_solo: bool,
    pub kstopb_mode: i32,
    pub kstopb_kmin: i32,
    /// [S][j] for S in 0..=s2n, j in 0..8
    pub kstopb_tau: *const f64,
    pub kstopb_tau_n: usize,
    pub kstopb_widths: Ints,
    pub kstopb_explore: i32,
    /// s2/flash_checkpoint.h FlashCache: 0 off, 1 aligned, 2 terminal
    pub flash_cache: i32,
    pub flash_budget_mib: i64,
    pub ds_moe_main_topk: i32,
    pub ds_moe_decode_topk: i32,
    pub ds_idx_pipe_b: i32,
    pub ds_idx_q_trig: i32,
    pub ds_idx_pipe_qr: i32,
    pub ds_admit_keep_scratch: i32,
    pub ds_sess_drafts: bool,
    pub ds_draft_cap: i32,
    pub plans: *const CPlan,
    pub n_plans: usize,
    pub row_counts: Ints,
    pub draft_rows: Ints,
    pub nl: i32,
    pub nx: i32,
    pub ncoll: i32,
    pub r: i32,
    pub own4: bool,
    pub qp_roll_s: i32,
    pub qp_roll_ser: bool,
    pub qp_wire_e: i32,
}

/// An option value or message: bytes, as the C++ kept them, whatever their encoding.
pub type Str = Vec<u8>;

// ---------------------------------------------------------------------------------------------------- the knob table

struct Knobs<'a>(&'a KnobApi);

impl Knobs<'_> {
    fn flag_opt(&self, id: u32) -> Option<bool> {
        let f = self.0.flag?;
        let mut v: c_int = 0;
        // SAFETY: the C API writes one int
        (unsafe { f(id, 0, &mut v) } != 0).then_some(v != 0)
    }
    fn int_opt(&self, id: u32) -> Option<i64> {
        let f = self.0.int_?;
        let mut v: i64 = 0;
        // SAFETY: as above
        (unsafe { f(id, 0, &mut v) } != 0).then_some(v)
    }
    fn float_opt(&self, id: u32) -> Option<f64> {
        let f = self.0.float_?;
        let mut v: f64 = 0.0;
        // SAFETY: as above
        (unsafe { f(id, 0, &mut v) } != 0).then_some(v)
    }
    fn text_opt(&self, id: u32) -> Option<Str> {
        let f = self.0.text?;
        // SAFETY: the C API returns null or a string that lives for the process
        let p = unsafe { f(id, 0) };
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_bytes().to_vec())
    }
    fn ints_opt(&self, id: u32) -> Option<Vec<i64>> {
        let f = self.0.ints?;
        let mut p: *const i64 = std::ptr::null();
        // SAFETY: the C API sets p to n values that live for the process
        let n = unsafe { f(id, 0, &mut p) };
        (n >= 0).then(|| if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec() })
    }
    fn floats_n(&self, id: u32, n: u32) -> Option<Vec<f64>> {
        let f = self.0.floats?;
        let mut p: *const f64 = std::ptr::null();
        // SAFETY: as above
        let len = unsafe { f(id, n, &mut p) };
        (len >= 0).then(|| if len == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, len as usize) }.to_vec() })
    }
    // a knob with a default always has a value
    fn flag(&self, id: u32) -> bool {
        self.flag_opt(id).unwrap_or(false)
    }
    fn int(&self, id: u32) -> i64 {
        self.int_opt(id).unwrap_or(0)
    }
    fn float(&self, id: u32) -> f64 {
        self.float_opt(id).unwrap_or(0.0)
    }
    fn text(&self, id: u32) -> Str {
        self.text_opt(id).unwrap_or_default()
    }
    fn floats(&self, id: u32) -> Vec<f64> {
        self.floats_n(id, 0).unwrap_or_default()
    }
    fn tests_set(&self) -> Str {
        let Some(f) = self.0.tests_set else { return Str::new() };
        // SAFETY: as text_opt
        let p = unsafe { f() };
        if p.is_null() {
            Str::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_bytes().to_vec()
        }
    }
}

// ------------------------------------------------------------------------------------------------------- the options

#[derive(Clone, Debug)]
pub struct Plan {
    pub prompt: Vec<i32>,
    pub first: i32,
    pub chunk: i32,
    pub n_gen: i32,
    pub inv_temp: f32,
    pub seed: u32,
    pub top_p: f32,
    pub lp: i32,
    pub plp: i32,
    pub rs: bool,
}

impl Default for Plan {
    fn default() -> Self {
        Plan { prompt: Vec::new(), first: 0, chunk: 4, n_gen: 16, inv_temp: 0.0, seed: 1234, top_p: 0.0, lp: -1, plp: -1, rs: false }
    }
}

impl Plan {
    fn prompt_steps(&self) -> i32 {
        (self.prompt.len() as i32 - self.first + self.chunk - 1) / self.chunk
    }
    fn step_rows(&self, s: i32) -> i32 {
        let left = self.prompt.len() as i32 - self.first - s * self.chunk;
        if s >= self.prompt_steps() {
            1
        } else if left < self.chunk {
            left
        } else {
            self.chunk
        }
    }
    /// the distinct row counts of the plan, in step order
    fn rows(&self) -> Vec<i32> {
        let mut v = Vec::new();
        for s in 0..self.prompt_steps().wrapping_add(self.n_gen) {
            add(&mut v, self.step_rows(s));
        }
        v
    }
}

/// engine_opts.h CommonOpts: the flags m4sim takes too
#[derive(Clone, Debug)]
pub struct Common {
    pub v2_dir: Str,
    pub v2_slices: Str,
    pub v2_widths: Str,
    pub v2_dense: Str,
    pub v2_layout: Str,
    pub v2_trellis: bool,
    pub qo_man: Str,
    pub qo_file: Str,
    pub layers_s: Str,
    pub levels_s: Str,
    pub prompts: Vec<Str>,
    pub cap: i32,
    pub mp: i32,
    pub top: i32,
    pub pfrows: i32,
    pub mtp: i32,
    pub lanes: i32,
    pub nc: i32,
    pub temp: f32,
    pub graph: bool,
    pub wide: bool,
    pub pfv2: bool,
    pub adapt: bool,
    pub n_gen: Option<i32>,
    pub chunk: Option<i32>,
    pub seed: Option<u32>,
    pub top_p: Option<f32>,
    pub rs: Option<bool>,
}

impl Default for Common {
    fn default() -> Self {
        Common {
            v2_dir: Str::new(),
            v2_slices: Str::new(),
            v2_widths: Str::new(),
            v2_dense: Str::new(),
            v2_layout: Str::new(),
            v2_trellis: true,
            qo_man: Str::new(),
            qo_file: Str::new(),
            layers_s: "0,3,40,77".into(),
            levels_s: Str::new(),
            prompts: Vec::new(),
            cap: 64,
            mp: 0,
            top: 0,
            pfrows: 0,
            mtp: 0,
            lanes: 1,
            nc: 0,
            temp: 0.0,
            graph: false,
            wide: false,
            pfv2: false,
            adapt: false,
            n_gen: None,
            chunk: None,
            seed: None,
            top_p: None,
            rs: None,
        }
    }
}

/// s2_batch.h KStopB; tau is [S][j] for S in 0..=s2n
#[derive(Clone, Debug)]
pub struct KStopB {
    pub mode: i32,
    pub kmin: i32,
    pub tau: Vec<[f64; 8]>,
    pub widths: Vec<i32>,
    pub explore: i32,
}

impl KStopB {
    fn new(s2n: i32) -> Self {
        KStopB { mode: 0, kmin: 0, tau: vec![[0.0; 8]; s2n.max(0) as usize + 1], widths: Vec::new(), explore: 0 }
    }
    fn width(&self, s: i32, k: i32, m: i32) -> bool {
        if m < s * (self.kmin + 1) || m > s * (k + 1) {
            return false;
        }
        self.widths.is_empty() || m % (k + 1) == 0 || self.widths.contains(&m)
    }
    /// s2_batch.h kstopb_widths: the main widths of shared rounds of 1..n sessions
    fn widths_of(&self, n: i32, k: i32) -> Vec<i32> {
        let mut v = Vec::new();
        if self.mode == 0 {
            return v;
        }
        for m in 1..=n * (k + 1) {
            if (1..=n).any(|s| self.width(s, k, m)) {
                v.push(m);
            }
        }
        v
    }
}

/// s2_batch.h KStop
#[derive(Clone, Debug)]
pub struct KStop {
    pub mode: i32,
    pub kmax: i32,
    pub tau: [f64; 4],
    pub seed3: f64,
    pub log: bool,
    pub solo: bool,
    pub b: KStopB,
}

#[derive(Clone, Debug, Default)]
pub struct Opts {
    pub rank: i32,
    pub ips: Vec<Str>,
    pub ips2: Vec<Str>,
    pub ring: bool,
    pub ring_devs: Vec<Str>,
    pub proxy_cpu: i32,
    pub pf_ips2: Vec<Str>,
    pub manifest: Str,
    pub packfile: Str,
    pub co: Common,
    pub qp_man: Str,
    pub qp_file: Str,
    pub dfl_man: Str,
    pub dfl_file: Str,
    pub dfl_sel: i32,
    pub dfl_k: i32,
    pub spec_reset_per_request: bool,
    pub spec_fixed: bool,
    pub plan: Plan,
    pub image: bool,
    pub hash: bool,
    pub reuse_prefix: bool,
    pub check_args: bool,
    pub lanes_explicit: bool,
    pub s2dir: Str,
    pub grammar_dir: Str,
    pub kv_tiers: Vec<Str>,
    pub kv_keep: Str,
    pub kv_system_after: i32,
    pub kv_write_limit: f64,
    pub kv_format: Str,
    pub kv_old: bool,
    pub kv_old_all: bool,
    pub set_id: Str,
    pub steering_dir: Str,
    pub steering_force_s: Str,
    pub steering_model_revision: Str,
    pub steering_tokenizer_revision: Str,
    pub steering_template_digest: Str,
    pub steering_weight_set: Str,
    pub steering_template_file: Str,
    pub s2port: i32,
    pub s2sess: i32,
    pub s2epoch: u64,
    pub cap: i32,
    pub top: i32,
    pub pfrows: i32,
    pub nc: i32,
    pub mtp: i32,
    pub mp: i32,
    pub lanes: i32,
    pub temp: f32,
    pub graph: bool,
    pub wide: bool,
    pub pfv2: bool,
    pub adapt: bool,
    pub release: bool,
    pub s2mode: bool,
    pub dfl: bool,
    pub k: i32,
    pub layers: Vec<i32>,
    pub progress_isolate: bool,
    pub route_split_mode: i32,
    pub c8_pack_env: i32,
    pub spec: SpecLen,
    pub rt_win: i32,
    pub kstop: Option<KStop>,
    pub flash_cache: i32,
    pub flash_budget_mib: i64,
    pub ds_moe_main_topk: i32,
    pub ds_moe_decode_topk: i32,
    pub ds_idx_pipe_b: i32,
    pub ds_idx_q_trig: i32,
    pub ds_idx_pipe_qr: i32,
    pub ds_admit_keep_scratch: i32,
    pub ds_sess_drafts: bool,
    pub ds_draft_cap: i32,
    pub plans: Vec<Plan>,
    pub row_counts: Vec<i32>,
    pub draft_rows: Vec<i32>,
    pub nl: i32,
    pub nx: i32,
    pub ncoll: i32,
    pub r: i32,
    pub own4: bool,
    pub qp_roll_s: i32,
    pub qp_roll_ser: bool,
    pub qp_wire_e: i32,
}

/// The prompt both mains run when none is given (engine_opts.h DEFAULT_PROMPT).
pub const DEFAULT_PROMPT: &[u8] = b"154822,154824,154826,25062";

fn add(v: &mut Vec<i32>, m: i32) {
    if !v.contains(&m) {
        v.push(m);
    }
}

/// engine_options.h eopt::split: "a,,b" keeps the empty piece, a trailing separator adds none
fn split(s: &[u8], c: u8) -> Vec<Str> {
    let mut v: Vec<Str> = s.split(|&b| b == c).map(<[u8]>::to_vec).collect();
    if v.last().is_some_and(|x| x.is_empty()) {
        v.pop();
    }
    v
}

fn cat(parts: &[&[u8]]) -> Str {
    parts.concat()
}

fn has(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// a line to stderr, its bytes as they are
fn err_line(b: &[u8]) {
    let _ = std::io::stderr().write_all(b);
}

/// rank_host.h die
fn die(m: &[u8]) -> ! {
    err_line(&cat(&[b"g6: ", m, b"\n"]));
    std::process::exit(2)
}

/// rank_host.h parse_ints: comma-separated integers, "@file" read from a file, "all" every main layer
fn parse_ints(s: &[u8], layers: i32) -> Vec<i32> {
    if s == b"all" {
        return (0..layers).collect();
    }
    if let Some(path) = s.strip_prefix(b"@") {
        return read_ints(path).unwrap_or_else(|| die(&cat(&[b"cannot read ", path])));
    }
    let mut v = Vec::new();
    let mut p = 0;
    while p < s.len() {
        let q = s[p..].iter().position(|&c| c == b',').map_or(s.len(), |i| p + i);
        v.push(c_atoi(&s[p..q]));
        p = q + 1;
    }
    v
}

/// engine_opts.h common_arg: Ok(true) when `a` is a shared flag
fn common_arg(a: &[u8], next: &mut dyn FnMut() -> Result<Str, Str>, o: &mut Common) -> Result<bool, Str> {
    match a {
        b"--v2" => o.v2_dir = next()?,
        b"--v2-slices" => o.v2_slices = next()?,
        b"--v2-widths" => o.v2_widths = next()?,
        b"--v2-dense" => o.v2_dense = next()?,
        b"--layout" => o.v2_layout = next()?,
        b"--v2-dense-form" => {
            let f = next()?;
            if f != b"trellis" && f != b"rounded" {
                return Err("--v2-dense-form: trellis or rounded".into());
            }
            o.v2_trellis = f == b"trellis";
        }
        b"--qo-manifest" => o.qo_man = next()?,
        b"--qo-pack" => o.qo_file = next()?,
        b"--spec-adaptive" => o.adapt = true,
        b"--spec-levels" => o.levels_s = next()?,
        b"--layers" => o.layers_s = next()?,
        b"--prompt" => o.prompts.push(next()?),
        b"--gen" => o.n_gen = Some(c_atoi(&next()?)),
        b"--temp" => o.temp = c_atof(&next()?) as f32,
        b"--seed" => o.seed = Some(c_strtoul10(&next()?) as u32),
        b"--cap" => o.cap = c_atoi(&next()?),
        b"--chunk" => o.chunk = Some(c_atoi(&next()?)),
        b"--graph" => o.graph = true,
        b"--prefill" => o.mp = c_atoi(&next()?),
        b"--top" => o.top = c_atoi(&next()?),
        b"--wide" => o.wide = true,
        b"--pfv2" => o.pfv2 = true,
        b"--pf-rows" => o.pfrows = c_atoi(&next()?),
        b"--mtp" => o.mtp = c_atoi(&next()?),
        b"--pf-lanes" => o.lanes = c_atoi(&next()?),
        b"--top-p" => o.top_p = Some(c_atof(&next()?) as f32),
        b"--top-p-cands" => o.nc = c_atoi(&next()?),
        b"--spec-accept" => {
            let v = next()?;
            o.rs = Some(match v.as_slice() {
                b"coupled" => false,
                b"rejection" => true,
                _ => return Err("--spec-accept must be coupled or rejection".into()),
            });
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// engine_opts.h common_check
fn common_check(o: &Common, mx: i32, chunk: i32) -> Result<(), Str> {
    if chunk < 1 || chunk > mx {
        return Err(format!("--chunk must be 1..{mx}").into());
    }
    if o.mtp < 0 || o.mtp > (mx - 1).min(7) {
        return Err("--mtp must be 1..7 drafts".into());
    }
    Ok(())
}

/// GR8_PF_OWN=4 with GR8_PF_QP_ROLL: the rolling pack's env refusals (qp_wire.h's protocol), "" when they pass
fn roll_env_refusal(pf_qp: bool, roll: i32, wire_e_set: bool, wire_e: i32, qp_ctx: &str, qp_pack: bool) -> Str {
    const MAX_ENTRIES: i32 = 4;
    if wire_e_set && roll <= 0 {
        return "GR8_PF_QP_WIRE_E is set but GR8_PF_QP_ROLL is not (the wire ring exists only with ROLL)".into();
    }
    if roll <= 0 {
        return Str::new();
    }
    if !pf_qp {
        return "GR8_PF_QP_ROLL needs GR8_PF_OWN=4 and --prefill".into();
    }
    if !(1..=MAX_ENTRIES).contains(&wire_e) {
        return format!("GR8_PF_QP_WIRE_E: 1..{MAX_ENTRIES} (unset = 1)").into();
    }
    if qp_pack {
        return "GR8_PF_QP_ROLL: no --qp-pack (the rolling pack is built from this rank's pack + the partner's half by RDMA)"
            .into();
    }
    // without it the QO key table would be sized at the cap, and every request would run OWN=4
    if qp_ctx.is_empty() {
        return "GR8_PF_QP_ROLL needs GR8_PF_QP_CTX (the QO key / RoPE table size, in positions)".into();
    }
    let (v, whole) = c_strtol10(qp_ctx.as_bytes());
    let first = qp_ctx.as_bytes()[0];
    if !whole || !(b'1'..=b'9').contains(&first) || v <= 0 || v > 0x7fffffff {
        return format!("GR8_PF_QP_CTX='{qp_ctx}': a positive decimal position count").into();
    }
    Str::new()
}

/// s2/flash_checkpoint.h flash_cache_mode
fn flash_cache_mode(s: &[u8]) -> Result<i32, Str> {
    match s {
        b"off" | b"0" => Ok(0),
        b"aligned" => Ok(1),
        b"terminal" => Ok(2),
        _ => Err(cat(&[b"GR8_GLM53F_CACHE=", s, b": off, aligned or terminal"])),
    }
}

/// spec_len.h SpecLen::flash_cost: Flash's verify table and MTP draft costs for the build's width
fn flash_cost(s: &mut SpecLen, tp: i32) {
    const T2: [f64; 9] = [0.0, 28.7, 37.9, 45.7, 52.3, 59.0, 65.6, 71.7, 77.5];
    const T3: [f64; 9] = [0.0, 24.8, 30.4, 35.8, 39.6, 46.4, 51.7, 56.2, 60.7];
    const T4: [f64; 9] = [0.0, 20.5, 21.9, 25.4, 27.5, 33.0, 36.9, 40.7, 44.7];
    s.table = if tp == 4 { T4 } else if tp == 3 { T3 } else { T2 };
    s.draft_round = if tp == 4 { 2.6 } else if tp == 3 { 2.39 } else { 2.8 };
    s.draft_each = if tp == 4 { 1.25 } else if tp == 3 { 1.94 } else { 2.2 };
}

/// printf's %#x
fn alt_hex(v: u32) -> String {
    if v == 0 {
        "0".into()
    } else {
        format!("{v:#x}")
    }
}

/// The parse. Err is the message main dies with.
pub fn parse(m: &ModuleInfo, api: &KnobApi, host: &Host, argv: &[Str]) -> Result<Opts, Str> {
    let k = Knobs(api);
    let waterfill = k.flag(kid::S2_SHARED_WATERFILL);
    let wf_margin = k.float(kid::S2_SHARED_WF_MARGIN);
    if !wf_margin.is_finite() || wf_margin < 0.0 {
        return Err("GR8_S2_SHARED_WF_MARGIN: want a finite nonnegative value".into());
    }
    let (ds41, flash) = (m.model == MODEL_DS41, m.model == MODEL_GLM53F);
    if waterfill && ds41 {
        return Err("GR8_S2_SHARED_WATERFILL: DS41 shared steps require uniform rows; ragged rows are unsupported".into());
    }
    let (nr, mx, s2n) = (m.nr, m.mx, m.s2n);
    let dfl_drafts = m.dfl_drafts;
    let mut o = Opts {
        rank: -1,
        proxy_cpu: -1,
        dfl_sel: -1,
        dfl_k: dfl_drafts,
        hash: true,
        kv_system_after: 2,
        s2sess: if flash { 1 } else { 8 },
        s2epoch: 1,
        lanes: 1,
        r: -1,
        qp_wire_e: 1,
        ..Default::default()
    };
    let mut kstop = KStop {
        mode: 0,
        kmax: 0,
        tau: [0.0, 0.0, 0.2, 0.5],
        seed3: 0.55,
        log: false,
        solo: false,
        b: KStopB::new(s2n),
    };

    // GR8_MOE_TOPK (model change, screen arm): unset = the model's top-k
    if ds41 {
        o.c8_pack_env = k.flag(kid::DS_C8PACK) as i32;
    }
    if flash {
        let tk = k.int(kid::MOE_TOPK) as i32;
        if tk < m.topk {
            eprint!("# GR8_MOE_TOPK={tk}: routed experts per token cut {} -> {tk}, weights renormalized (screen arm)\n", m.topk);
        }
    } else if ds41 {
        if k.int(kid::MOE_TOPK) != 8 {
            return Err("GR8_MOE_TOPK is a Flash-model option: DeepSeek-V4.1 reads GR8_DS_MOE_TOPK".into());
        }
        if k.flag(kid::PF_ROUTER_SPLIT) {
            return Err("GR8_PF_ROUTER_SPLIT is a Flash-model option".into());
        }
    } else {
        if k.int(kid::MOE_TOPK) != m.topk as i64 {
            return Err("GR8_MOE_TOPK is a Flash-model option (this build is the GLM model)".into());
        }
        if k.flag(kid::PF_ROUTER_SPLIT) {
            return Err("GR8_PF_ROUTER_SPLIT is a Flash-model option (this build is the GLM model)".into());
        }
    }
    if flash {
        o.route_split_mode = k.flag(kid::PF_ROUTER_SPLIT) as i32;
    }
    o.progress_isolate = k.flag(kid::PROGRESS_ISOLATE);

    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].clone();
        let mut next = || -> Result<Str, Str> {
            if i + 1 >= argv.len() {
                return Err(cat(&[b"missing value for ", &a]));
            }
            i += 1;
            Ok(argv[i].clone())
        };
        if a == b"--pf-lanes" {
            o.lanes_explicit = true;
        }
        if common_arg(&a, &mut next, &mut o.co)? {
            i += 1;
            continue;
        }
        match a.as_slice() {
            b"--rank" => o.rank = c_atoi(&next()?),
            b"--ips" => o.ips = split(&next()?, b','),
            b"--ips2" => o.ips2 = split(&next()?, b','),
            b"--pf-ips2" => o.pf_ips2 = split(&next()?, b','),
            b"--ring" => o.ring = true,
            b"--ring-devs" => o.ring_devs = split(&next()?, b','),
            b"--proxy-cpu" => o.proxy_cpu = c_atoi(&next()?),
            b"--manifest" => o.manifest = next()?,
            b"--pack" => o.packfile = next()?,
            b"--qp-manifest" => o.qp_man = next()?,
            b"--qp-pack" => o.qp_file = next()?,
            b"--dflash-manifest" => o.dfl_man = next()?,
            b"--dflash-pack" => o.dfl_file = next()?,
            b"--dflash-sel" => o.dfl_sel = c_atoi(&next()?),
            b"--dflash-k" => o.dfl_k = c_atoi(&next()?),
            b"--spec-fixed" if flash => o.spec_fixed = true,
            b"--spec-reset-per-request" if flash => o.spec_reset_per_request = true,
            b"--image" => o.image = true,
            b"--no-hash" => o.hash = false,
            b"--reuse-prefix" => o.reuse_prefix = true,
            b"--check-args" => o.check_args = true,
            b"--kv-store" => o.kv_tiers.push(next()?),
            b"--kv-keep" => o.kv_keep = next()?,
            b"--kv-system-after" => o.kv_system_after = c_atoi(&next()?),
            b"--kv-write-limit" => o.kv_write_limit = c_atof(&next()?),
            b"--kv-format" => o.kv_format = next()?,
            b"--kv-tier" => {
                let v = next()?;
                o.kv_old = true;
                o.kv_old_all = o.kv_old_all || has(&v, b"keep=evicted") || has(&v, b"keep=all") || !has(&v, b"keep=");
                o.kv_tiers.push(v);
            }
            b"--kv-shared-after" => {
                o.kv_old = true;
                o.kv_system_after = c_atoi(&next()?);
            }
            b"--serve" => o.s2dir = next()?,
            b"--grammar-dir" => o.grammar_dir = next()?,
            b"--s2-port" => o.s2port = c_atoi(&next()?),
            b"--s2-sessions" => o.s2sess = c_atoi(&next()?),
            b"--s2-epoch" => o.s2epoch = c_strtoull10(&next()?),
            b"--set-id" => {
                o.set_id = next()?;
                let s = &o.set_id;
                if s.is_empty() || s.len() > 64 || s.iter().any(|c| !matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
                    return Err("--set-id must be 1..64 lowercase hex digits".into());
                }
            }
            b"--steering-profile" => o.steering_dir = next()?,
            b"--steering-weight-set" => o.steering_weight_set = next()?,
            b"--steering-force" => o.steering_force_s = next()?,
            b"--steering-model-revision" => o.steering_model_revision = next()?,
            b"--steering-tokenizer-revision" => o.steering_tokenizer_revision = next()?,
            b"--steering-chat-template" => o.steering_template_file = next()?,
            b"--steering-chat-template-sha256" => o.steering_template_digest = next()?,
            _ => return Err(cat(&[b"unknown argument ", &a])),
        }
        i += 1;
    }
    let co = &mut o.co;
    o.cap = co.cap;
    o.top = co.top;
    o.pfrows = co.pfrows;
    o.nc = co.nc;
    o.mtp = co.mtp;
    o.mp = co.mp;
    o.lanes = co.lanes;
    o.temp = co.temp;
    o.graph = co.graph;
    o.wide = co.wide;
    o.pfv2 = co.pfv2;
    o.adapt = co.adapt;
    let (cap, pfrows, nc, mtp, mp) = (o.cap, o.pfrows, o.nc, o.mtp, o.mp);
    let plan = &mut o.plan;
    if let Some(v) = co.n_gen {
        plan.n_gen = v;
    }
    if let Some(v) = co.seed {
        plan.seed = v;
    }
    if let Some(v) = co.chunk {
        plan.chunk = v;
    }
    if let Some(v) = co.top_p {
        plan.top_p = v;
    }
    if let Some(v) = co.rs {
        plan.rs = v;
    }
    let release = !co.v2_dense.is_empty();
    o.release = release;
    if k.text(kid::KVB) == b"fp8" && !release {
        return Err("GR8_KVB=fp8 needs --v2-dense: packs hold kv_b in BF16".into());
    }
    if o.rank < 0 || o.ips.len() != nr as usize || (!release && (o.manifest.is_empty() || o.packfile.is_empty())) {
        return Err(format!(
            "usage: g6m4 --rank R --ips NR,addresses (this build: NR = {nr}) [--ips2 ..] --manifest M --pack P|- [--qo-manifest QM --qo-pack QP] [--qp-manifest PM --qp-pack PP] [--layers ..] [--gen N] [--temp T]"
        ).into());
    }
    if ds41 {
        if release && (!o.manifest.is_empty() || !o.packfile.is_empty() || !o.image) {
            return Err("--v2-dense CHECKPOINT_DIR replaces --manifest / --pack and needs --image (DS41: no --v2 / --v2-slices / --v2-widths)".into());
        }
    } else if release && (!o.manifest.is_empty() || !o.packfile.is_empty() || co.v2_dir.is_empty()) {
        return Err("--v2-dense replaces --manifest / --pack and needs --v2 (the image comes from the release alone)".into());
    }
    if !release && !co.v2_layout.is_empty() {
        return Err("--layout needs --v2-dense (it places the release's pieces)".into());
    }
    if k.flag(kid::DEC_SHARED_SCRATCH) && !o.pfv2 {
        return Err("GR8_DEC_SHARED_SCRATCH=1 needs --pfv2".into());
    }
    o.layers = parse_ints(&co.layers_s, m.layers);
    if flash {
        if co.layers_s == b"all" {
            if o.layers.len() != m.layers as usize {
                return Err("Flash --layers all did not select every main layer".into());
            }
            if o.layers.iter().enumerate().any(|(j, &l)| l != j as i32) {
                return Err("Flash --layers all selected an unexpected layer".into());
            }
        }
        eprint!(
            "# rank {}: Flash actual main layers {} first {} last {}\n",
            o.rank,
            o.layers.len(),
            o.layers.first().copied().unwrap_or(-1),
            o.layers.last().copied().unwrap_or(-1)
        );
    }
    if co.prompts.is_empty() {
        co.prompts.push(DEFAULT_PROMPT.to_vec());
    }
    plan.inv_temp = if o.temp > 0.0 { 1.0 / o.temp } else { 0.0 };
    common_check(co, mx, plan.chunk)?;
    if nc < 0 || nc.wrapping_mul(nr) > m.tp_sort {
        return Err(format!("--top-p-cands must be 0..{}", m.tp_sort / nr).into());
    }
    // Flash lane default: the MP8192 memory gate passed; larger MPs remain opt-in
    if flash && !o.lanes_explicit && mp > 0 && mp <= 8192 {
        o.lanes = 2;
    }
    let s2mode = !o.s2dir.is_empty();
    let dfl = !o.dfl_man.is_empty();
    o.s2mode = s2mode;
    o.dfl = dfl;
    if s2mode && (k.text_opt(kid::RESID_DUMP).is_some() || k.text_opt(kid::LOGIT_DUMP).is_some()) {
        return Err("GR8_RESID_DUMP / GR8_LOGIT_DUMP support offline main steps, not --serve".into());
    }
    // S2 keeps --mtp next to the drafter: a session alone at its prompt tail runs DFlash2, the others MTP
    let (dfl_sel, dfl_k) = (o.dfl_sel, o.dfl_k);
    if dfl && (o.dfl_file.is_empty() || (mtp != 0 && !s2mode) || !(-1..=1).contains(&dfl_sel) || dfl_k < 1 || dfl_k > dfl_drafts) {
        return Err("--dflash-manifest needs --dflash-pack, and replaces --mtp (7 drafts per round) except under S2; --dflash-sel 0|1; --dflash-k 1..7".into());
    }
    // TP=6 runs the drafter body replicated, and so does Flash at TP=3; GLM-5.3's TP=3 would too, but is not measured
    if dfl && m.dfl_rep != 0 && nr != 6 && !(flash && nr == 3) {
        return Err(format!(
            "--dflash: TP={nr} has no DFlash2 drafter (TP=4 splits it, TP=6 and Flash's TP=3 replicate it)"
        ).into());
    }
    if plan.rs && mtp == 0 && !dfl {
        return Err("--spec-accept rejection needs --mtp or --dflash-manifest".into());
    }
    o.k = if dfl { dfl_drafts } else { mtp };
    let kk = o.k;
    let levels_s = co.levels_s.clone();
    if flash {
        if o.spec_reset_per_request && s2mode {
            return Err("--spec-reset-per-request is offline only (no --serve)".into());
        }
        // Flash: --mtp K is adaptive speculation (up to K drafts) unless --spec-fixed
        if o.spec_fixed && (o.adapt || !levels_s.is_empty() || mtp == 0) {
            return Err("--spec-fixed needs --mtp, and not --spec-adaptive / --spec-levels".into());
        }
        if mtp != 0 && !o.spec_fixed && !s2mode && !dfl {
            o.adapt = true;
        }
    }
    let adapt = o.adapt;
    if (adapt || !levels_s.is_empty()) && ((s2mode && !dfl) || (mtp == 0 && !dfl)) {
        return Err("--spec-adaptive needs --mtp or --dflash-manifest; under S2 it is the DFlash2 sessions' policy".into());
    }
    if !levels_s.is_empty() && !adapt {
        return Err("--spec-levels needs --spec-adaptive".into());
    }
    if adapt {
        let kmax = if dfl { dfl_k } else { mtp };
        let lv: Vec<i32> = if levels_s.is_empty() { (1..=kmax).collect() } else { parse_ints(&levels_s, m.layers) };
        if lv.iter().any(|&l| l < 1 || l > kmax) {
            return Err(format!("--spec-levels: each level must be 1..{kmax}").into());
        }
        o.spec.on = true;
        if !o.spec.set_levels(&lv) {
            return Err("--spec-levels: no level".into());
        }
        o.spec.draft_round = if dfl { 5.6 } else { 2.5 };
        o.spec.draft_each = if dfl { 0.0 } else { 2.1 };
        if flash {
            flash_cost(&mut o.spec, nr);
            if dfl {
                o.spec.draft_round = 5.3;
                o.spec.draft_each = 0.0;
            }
        }
    }
    let (s2port, s2sess, s2epoch) = (o.s2port, o.s2sess, o.s2epoch);
    if ds41 {
        // DS41 serve: no --prefill tensor prefill and no --mtp draft (the draft is DSpark, GR8_DS_MTP); --graph is optional
        if s2mode && (!(1..=65535).contains(&s2port) || !(1..=32).contains(&s2sess) || s2epoch == 0) {
            return Err("DS41 serving (--serve) needs --s2-port 1..65535, --s2-sessions 1..32 and --s2-epoch > 0 (no --prefill, --mtp 0; GR8_DS_MTP 1..5 is the DSpark draft); --graph is optional (without it each step runs eagerly)".into());
        }
    } else if s2mode
        && (!o.graph || mtp == 0 || mp == 0 || !(1..=65535).contains(&s2port) || !(1..=8).contains(&s2sess) || s2epoch == 0)
    {
        return Err("serving (--serve) needs --graph, --mtp, --prefill, --s2-port 1..65535, --s2-sessions 1..8 and --s2-epoch > 0".into());
    }
    if s2mode {
        let t = k.tests_set();
        if !t.is_empty() {
            return Err(cat(&[b"serving (--serve) refuses test knobs (knobs/test.yaml): ", &t]));
        }
    }
    let rt_store = k.text_opt(kid::RT_STORE).is_some();
    if rt_store && !s2mode {
        return Err("GR8_RT_STORE needs --serve (S2 below R with GR8_CP on)".into());
    }
    if k.flag(kid::TEST_RT_POISON) {
        return Err("GR8_TEST_RT_POISON is an m4sim test: g6m4 reads the local copy of a row without a slot".into());
    }
    // GR8_RT_LOCAL: each split run's tail cycles through that many window pages per rank; GR8_RT_PLACE: the main
    // model's physical pages per rank, so --cap may pass what the local pages hold
    let rt_win = k.int(kid::RT_LOCAL) as i32;
    o.rt_win = rt_win;
    if rt_win < 0 || (rt_win != 0 && !rt_store) {
        return Err("GR8_RT_LOCAL needs GR8_RT_STORE and a page count >= 1".into());
    }
    if k.int_opt(kid::RT_PLACE).is_some() && rt_win == 0 {
        return Err("GR8_RT_PLACE needs GR8_RT_LOCAL (the pages past it are reached through the rows tier only)".into());
    }
    if rt_win != 0 && k.int_opt(kid::RT_L).is_some_and(|l| l != k.int(kid::CP_R)) {
        return Err("GR8_RT_LOCAL needs GR8_RT_L = GR8_CP_R (the window holds every tail page past R)".into());
    }
    // the profile itself is loaded by main (steering::prepare_boot), after the parse
    if !o.steering_dir.is_empty() && !o.hash {
        return Err("--steering-profile requires weight hash checks".into());
    }
    if !o.steering_dir.is_empty() && (o.set_id.is_empty() || o.steering_weight_set.is_empty()) {
        return Err("--steering-profile needs --set-id and --steering-weight-set".into());
    }
    // GR8_MTP_KSTOP (s2_batch.h KStop) and GR8_MTP_KSTOPB (KStopB): their policies join the S2 runtime signature
    {
        kstop.mode = k.int_opt(kid::MTP_KSTOP).unwrap_or(if s2mode && dfl && mtp >= 2 { 2 } else { 0 }) as i32;
        let lplog = k.int(kid::MTP_LPLOG) as i32;
        kstop.log = lplog == 2 || (lplog == 1 && o.rank == 0);
        kstop.kmax = k.int_opt(kid::MTP_KSTOP_KMAX).unwrap_or(mtp as i64) as i32;
        let tau = k.floats(kid::MTP_KSTOP_TAU);
        if tau.len() > 3 {
            return Err("GR8_MTP_KSTOP_TAU: at most 3 thresholds".into());
        }
        kstop.tau[1..1 + tau.len()].copy_from_slice(&tau);
        kstop.seed3 = k.float(kid::MTP_KSTOP_SEED3);
        let kmax = kstop.kmax;
        if kstop.mode != 0
            && (!s2mode
                || (!dfl && (kstop.mode != 1 || kmax > mtp))
                || mtp < 2
                || !(1..=3).contains(&kmax)
                || !(kstop.seed3 > 0.0 && kstop.seed3 < 1.0)
                || kmax + 1 > s2sess.wrapping_mul(mtp + 1).max(plan.chunk))
        {
            return Err("GR8_MTP_KSTOP needs --serve and --mtp 2+ (without --dflash-manifest: mode 1 and GR8_MTP_KSTOP_KMAX <= --mtp); GR8_MTP_KSTOP_KMAX 1..3 (verify rows within a step's), GR8_MTP_KSTOP_SEED3 in (0, 1)".into());
        }
        if kstop.mode == 0 {
            kstop.kmax = 0;
        }
        kstop.solo = kstop.mode == 1 && !dfl;
        let b = &mut kstop.b;
        b.mode = k.flag(kid::MTP_KSTOPB) as i32;
        b.kmin = k.int(kid::MTP_KSTOPB_KMIN) as i32;
        let mut bt = [0.0f64; 8];
        let btau = k.floats(kid::MTP_KSTOPB_TAU);
        if btau.len() > 7 {
            return Err("GR8_MTP_KSTOPB_TAU: at most 7 comma-separated numbers".into());
        }
        bt[1..1 + btau.len()].copy_from_slice(&btau);
        for s in 1..=s2n {
            let row = &mut b.tau[s as usize];
            *row = bt;
            if let Some(t) = k.floats_n(kid::MTP_KSTOPB_TAU_S, s as u32) {
                if t.len() > 7 {
                    return Err(format!("GR8_MTP_KSTOPB_TAU_S{s}: at most 7 comma-separated numbers").into());
                }
                row[1..1 + t.len()].copy_from_slice(&t);
            }
        }
        if let Some(widths) = k.ints_opt(kid::MTP_KSTOPB_WIDTHS) {
            if widths.len() > 64 {
                return Err("GR8_MTP_KSTOPB_WIDTHS: at most 64 comma-separated numbers".into());
            }
            for w in widths {
                if w < 1 || w > s2sess.wrapping_mul(mtp + 1) as i64 {
                    return Err("GR8_MTP_KSTOPB_WIDTHS: whole widths 1..sessions x (mtp + 1)".into());
                }
                b.widths.push(w as i32);
            }
        }
        b.explore = k.int(kid::MTP_KSTOPB_EXPLORE) as i32;
        // the draft log-probabilities come from every MTP draft step, with or without the DFlash2 drafter
        if b.mode != 0 && (!s2mode || b.explore < 0) {
            return Err("GR8_MTP_KSTOPB needs --serve; GR8_MTP_KSTOPB_EXPLORE >= 0".into());
        }
        if b.mode == 0 && !waterfill {
            *b = KStopB::new(s2n);
        }
    }
    // GR8_GLM53F_CACHE: Flash session checkpoints under --serve, in place of prefix reuse. Its store is the one
    // --kv-store, else this process's memory, GR8_GLM53F_STATE_BUDGET_MIB a rank
    let flash_cache = flash_cache_mode(&k.text(kid::GLM53F_CACHE))?;
    o.flash_cache = flash_cache;
    let flash_budget_mib = k.int(kid::GLM53F_STATE_BUDGET_MIB);
    o.flash_budget_mib = flash_budget_mib;
    let reuse_prefix = o.reuse_prefix;
    if flash {
        // Flash: --s2-sessions 1 (the default) is the solo path; 2..8 share the --cap pool
        if s2mode && (reuse_prefix || plan.rs) {
            return Err("Flash serve supports no prefix reuse or rejection sampling".into());
        }
        // DFlash2 with GR8_GLM53F_CACHE: a checkpoint hit restores none of the drafter's context
        if s2mode && dfl && mtp == 0 {
            return Err("Flash serve with DFlash2 needs --mtp (a second session moves the DFlash2 one to MTP)".into());
        }
        // a terminal checkpoint's KDA state comes from the verify ring, not a cold prefill: aligned checkpoints only
        if s2mode && dfl && flash_cache == 2 {
            return Err("Flash serve with DFlash2 needs GR8_GLM53F_CACHE=aligned or off (a terminal checkpoint's KDA state is not cold-exact)".into());
        }
        if s2mode && s2sess > 1 && s2sess.wrapping_mul(mtp + 1) > mx {
            return Err("Flash serve: --s2-sessions x (--mtp + 1) rows exceed G6_MX (every session's round is one step)".into());
        }
        if flash_cache != 0 {
            if !s2mode || mp == 0 {
                return Err("GR8_GLM53F_CACHE needs --serve and tensor prefill (--prefill N)".into());
            }
            if o.kv_tiers.len() > 1 {
                return Err("GR8_GLM53F_CACHE: one --kv-store at most (its checkpoint store)".into());
            }
            if o.kv_tiers.is_empty() && flash_budget_mib < 1 {
                return Err("GR8_GLM53F_STATE_BUDGET_MIB: >= 1".into());
            }
        }
        if !o.kv_tiers.is_empty() && flash_cache == 0 {
            return Err("--kv-store on Flash needs GR8_GLM53F_CACHE (its checkpoint store)".into());
        }
        if s2mode && s2sess != 1 && k.flag(kid::GLM53F_KV_CHUNKS) {
            return Err("GR8_GLM53F_KV_CHUNKS needs --s2-sessions 1".into());
        }
    } else {
        if flash_cache != 0 {
            return Err("GR8_GLM53F_CACHE is GLM-5.3-Flash only (bin/g6m4_flash)".into());
        }
        // --reuse-prefix: one conversation's caches, or under S2 every retired session's run
        if !o.kv_tiers.is_empty() && (m.caps & crate::module_info::CAP_KV_STORE == 0 || !(s2mode && reuse_prefix)) {
            return Err("--kv-store needs --serve and --reuse-prefix (it keeps the runs prefix reuse retains)".into());
        }
    }
    if o.kv_old {
        if o.kv_keep.is_empty() {
            o.kv_keep = if o.kv_old_all { "all" } else { "system" }.into();
        }
        if o.rank == 0 {
            err_line(&cat(&[
                b"# warning: --kv-tier and --kv-shared-after are deprecated: --kv-store STORE,size=N, --kv-keep ",
                &o.kv_keep,
                format!(", --kv-system-after {}\n", o.kv_system_after).as_bytes(),
            ]));
        }
    }
    if o.kv_keep.is_empty() {
        o.kv_keep = "system".into();
    }
    if !matches!(o.kv_keep.as_slice(), b"system" | b"sessions" | b"all") {
        return Err("--kv-keep system, sessions or all".into());
    }
    if o.kv_system_after < 1 {
        return Err("--kv-system-after N: N >= 1".into());
    }
    if !(o.kv_write_limit >= 0.0) || o.kv_write_limit > 1000.0 {
        return Err("--kv-write-limit F: 0 (off) to 1000 blocks per page computed".into());
    }
    {
        // the KV row is fixed at build time: the launcher runs bin/g6m4 for fp8 and bin/g6m4-int4 for int4, and each
        // binary refuses the other's format. The int4 build needs the flag, so it never starts by accident.
        let built = if m.kv4 != 0 { "int4" } else { "fp8" };
        let bin = if flash {
            "bin/g6m4_flash"
        } else if ds41 {
            "bin/g6m4_ds41"
        } else {
            "bin/g6m4"
        };
        if o.kv_format.is_empty() && built == "int4" {
            return Err(format!("this engine is the int4 KV build ({bin}-int4): it needs --kv-format int4").into());
        }
        if o.kv_format.is_empty() {
            o.kv_format = built.into();
        }
        if o.kv_format != b"fp8" && o.kv_format != b"int4" {
            return Err("--kv-format fp8 or int4".into());
        }
        if o.kv_format != built.as_bytes() {
            return Err(cat(&[
                b"--kv-format ",
                &o.kv_format,
                format!(": this engine is the {built} build ({bin} is fp8, {bin}-int4 is int4)").as_bytes(),
            ]));
        }
        if o.rank == 0 {
            eprint!("# rank 0: KV format {built} (latent row {} B, {})\n", m.kv_row_bytes, m.kv_format());
            eprint!(
                "# rank 0: module {} abi {}: TP{}, {} rows a step, {} session slots, KV {} {} B a row, caps {}\n",
                m.name(),
                m.abi,
                m.nr,
                m.mx,
                m.max_sessions,
                m.kv_format(),
                m.kv_row_bytes,
                alt_hex(m.caps)
            );
        }
    }
    let rank = o.rank;
    let (dsk_on, dsk_mtp) = (host.dsk_mtp > 0, host.dsk_mtp);
    let engram_colls = |layers: &[i32]| -> i32 {
        // SAFETY: the C++ passes its own context and a function that reads layers[0..n]
        host.engram_colls.map_or(0, |f| unsafe { f(host.ctx, layers.as_ptr(), layers.len()) })
    };
    let ds_block = m.ds_block;
    let ds_ms_rps = |ns: i32, sess_drafts: bool| -> i32 {
        if !dsk_on {
            1
        } else if sess_drafts {
            crate::planner::sess_drafts_kv(dsk_mtp, mx, ns) + 1
        } else if ns <= 8 {
            dsk_mtp + 1
        } else {
            1
        }
    };
    if ds41 {
        // before CUDA, fabric or weights; unset decode cut inherits the main cut
        let main_cut = k.int(kid::DS_MOE_TOPK) as i32;
        o.ds_moe_main_topk = main_cut;
        let decode = k.int_opt(kid::DS_MOE_TOPK_DECODE).unwrap_or(main_cut as i64) as i32;
        if decode < 1 || decode > m.topk || decode > main_cut {
            return Err(format!("GR8_DS_MOE_TOPK_DECODE must be 1..{}", main_cut.min(m.topk)).into());
        }
        o.ds_moe_decode_topk = decode;
        eprint!("# rank {rank} ds41 moe topk main {main_cut} decode {decode}\n");
        o.ds_idx_pipe_b = k.flag(kid::DS41_IDX_PIPE_B) as i32;
        eprint!("# rank {rank} ds41 idx pipe B {}\n", o.ds_idx_pipe_b);
        o.ds_idx_q_trig = k.flag(kid::DS41_IDX_Q_TRIG) as i32;
        o.ds_idx_pipe_qr = k.flag(kid::DS41_IDX_PIPE_QR) as i32;
        eprint!("# rank {rank} ds41 idx q trig {} pipe QR {}\n", o.ds_idx_q_trig, o.ds_idx_pipe_qr);
        o.ds_admit_keep_scratch = k.flag(kid::DS41_ADMIT_KEEP_SCRATCH) as i32;
        eprint!("# rank {rank} ds41 idx admit keep scratch {}\n", o.ds_admit_keep_scratch);
        // Prefix reuse uses device snapshots and optional shared tiered stores.
        if reuse_prefix && !s2mode {
            return Err("DS41: --reuse-prefix needs --serve".into());
        }
        if mp > 0 {
            return Err("DS41: --prefill is not supported (there is no DS41 tensor prefill; the prompt runs as decode steps)".into());
        }
        if mtp > 0 {
            return Err("DS41: --mtp N > 0 is not supported (the draft is DSpark: GR8_DS_MTP=1..5 with --mtp 0)".into());
        }
        // DSpark (off unless GR8_DS_MTP = 1..5): served only
        if !host.dsk_err.is_null() {
            // SAFETY: the C++ passes a NUL-terminated string that outlives the call
            let err = unsafe { CStr::from_ptr(host.dsk_err) }.to_bytes().to_vec();
            if !err.is_empty() {
                return Err(err);
            }
        }
        if dsk_on && !s2mode {
            return Err("DS41: GR8_DS_MTP is served only (it needs --serve)".into());
        }
        // GR8_DS_SESS_DRAFTS=1 (s2/ds41_sess_drafts.h): a session's drafts in a round go by its active sessions
        let sess_drafts = k.flag(kid::DS_SESS_DRAFTS);
        o.ds_sess_drafts = sess_drafts;
        if sess_drafts && !dsk_on {
            return Err("DS41: GR8_DS_SESS_DRAFTS=1 needs --serve and GR8_DS_MTP = 1..5".into());
        }
        o.ds_draft_cap = if sess_drafts { (mx / ds_block).min(m.ds_markov_maxs).max(1) } else { m.ds_markov_maxs };
        if host.dsk_relaxed && !dsk_on {
            return Err("DS41: GR8_DS_RELAXED_ACCEPT needs GR8_DS_MTP = 1..5 (there are no drafts to accept)".into());
        }
        if host.dsk_relax_topp > 0.0 && nc == 0 {
            return Err("DS41: GR8_DS_RELAXED_ACCEPT topp:P needs --top-p-cands > 0 (the candidates carry the mass above the draft)".into());
        }
        if dfl {
            return Err("DS41: --dflash-manifest is not supported".into());
        }
        if !o.grammar_dir.is_empty() {
            return Err("DS41: --grammar-dir is not supported".into());
        }
        // the DS41 serve runs the whole step as one graph (GR8_DEC_SPLIT=0)
        if s2mode && k.int(kid::DEC_SPLIT) != 0 {
            return Err("DS41: serving runs GR8_DEC_SPLIT=0 (a split serve graph is not built for DS41 yet)".into());
        }
        // Engram under --serve with DSpark is not checked against the token ring; engram_colls dies on a bad config
        if engram_colls(&o.layers) != 0 && dsk_on && !k.flag(kid::DS_ENGRAM_MTP) {
            return Err("DS41: GR8_DS_ENGRAM=1 with GR8_DS_MTP is not validated yet (Engram on serves at MTP 0)".into());
        }
        // GR8_DS_C8PACK=1: a pack needs 2+ real S2 slots and the W-prefill buffers
        if o.c8_pack_env != 0 && (!s2mode || s2sess < 2 || k.int(kid::DS_PREFILL_MIN) < 2) {
            return Err("GR8_DS_C8PACK=1 needs --serve, --s2-sessions >= 2 and GR8_DS_PREFILL_MIN >= 2".into());
        }
        // SESS > 1: one attention / index / Engram slab per session slot, every prompt of 2+ tokens on the W-row prefill
        if s2mode && s2sess != 1 && k.int(kid::DS_PREFILL_MIN) < 2 {
            return Err("DS41: --s2-sessions > 1 needs GR8_DS_PREFILL_MIN >= 2 (with GR8_DS_PREFILL_W)".into());
        }
        if s2mode
            && s2sess != 1
            && dsk_on
            && !sess_drafts
            && (s2sess.min(8).wrapping_mul(dsk_mtp + 1) > mx || s2sess.min(8).wrapping_mul(ds_block) > mx)
        {
            return Err(format!(
                "DS41: --s2-sessions x (GR8_DS_MTP + 1) verify rows and --s2-sessions x 5 draft rows must fit G6_MX {mx}"
            ).into());
        }
        // GR8_DS_PARK=1: a parked slot is evicted through the prefix cache; one slot has nothing to park beside
        if k.flag(kid::DS_PARK) && (!s2mode || s2sess < 2 || !reuse_prefix) {
            return Err("DS41: GR8_DS_PARK=1 needs --serve, --s2-sessions >= 2 and --reuse-prefix".into());
        }
    }
    if reuse_prefix && dfl && !s2mode {
        return Err("--reuse-prefix is not supported with --dflash-manifest (the drafter's prompt-context taps come from the prefill)".into());
    }
    if waterfill && (!s2mode || mtp < 1 || mx > 64) {
        return Err("GR8_S2_SHARED_WATERFILL needs --serve, --mtp >= 1 and max rows <= 64".into());
    }
    let mut row_counts: Vec<i32> = Vec::new();
    for ps in &o.co.prompts {
        let mut p = o.plan.clone();
        p.prompt = parse_ints(ps, m.layers);
        if p.prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if (p.prompt.len() as i32).wrapping_add(p.n_gen).wrapping_add(2 * kk) > cap {
            return Err("prompt + gen (+ 2 x drafts) exceeds --cap".into());
        }
        if mp != 0 {
            // --pf-rows -N: all rows but the last N; the rest is teacher-forced through decode steps
            let n = p.prompt.len() as i32;
            let pr = if pfrows < 0 { n.wrapping_add(pfrows) } else { pfrows };
            p.first = if pr > 0 && pr < n - 1 { pr } else { n - 1 };
        }
        for r in p.rows() {
            add(&mut row_counts, r);
        }
        o.plans.push(p);
    }
    let chunk = o.plan.chunk;
    if s2mode || reuse_prefix {
        for r in 1..=chunk {
            add(&mut row_counts, r);
        }
    }
    let mut draft_rows: Vec<i32> = Vec::new();
    if mtp != 0 {
        draft_rows = row_counts.clone();
        for r in 1..=mtp + 1 {
            add(&mut draft_rows, r);
        }
        add(&mut row_counts, mtp + 1);
    }
    if dfl {
        add(&mut row_counts, dfl_k + 1);
    }
    // a DFlash2 round's MTP first form runs on the accepted prefix + 1 rows (1 .. dfl_k + 1)
    if dfl && mtp != 0 {
        for r in 1..=dfl_k + 1 {
            add(&mut draft_rows, r);
        }
    }
    for r in 2..=kstop.kmax + 1 {
        add(&mut row_counts, r);
        add(&mut draft_rows, r);
    }
    for r in kstop.b.widths_of(1, mtp) {
        add(&mut row_counts, r);
    }
    if waterfill {
        let depth = if k.text(kid::NGRAM).is_empty() { mtp } else { mtp.max(7) };
        // First-form drafts may accept any subset/prefix; chain steps have one row per survivor.
        // Every width the allocator prices is therefore captured, without padding fictitious rows.
        for r in 1..=mx.min(s2sess * (depth + 1)) {
            add(&mut row_counts, r);
            add(&mut draft_rows, r);
        }
    }
    // GR8_S2_ADAPT: a session alone may verify any k <= mtp
    if s2mode && k.flag(kid::S2_ADAPT) {
        for r in 2..=mtp + 1 {
            add(&mut row_counts, r);
        }
    }
    for &l in o.spec.levels() {
        add(&mut row_counts, l + 1);
    }
    if flash && s2mode {
        for r in 1..=mtp + 1 {
            add(&mut row_counts, r);
        }
    }
    o.nl = o.layers.len() as i32;
    o.r = rank;
    if ds41 {
        o.nx = engram_colls(&o.layers);
        o.ncoll = 2 * o.nl + 1 + o.nx;
        if dsk_on {
            let sess_drafts = o.ds_sess_drafts;
            let top = (if s2mode { s2sess } else { 1 }).min(if sess_drafts { 32 } else { 8 });
            for ns in 1..=top {
                let mut r = ns;
                while r <= ns.wrapping_mul(ds_ms_rps(ns, sess_drafts)) && r <= mx {
                    add(&mut row_counts, r);
                    r += ns;
                }
            }
        }
    } else {
        o.nx = 0;
        o.ncoll = 2 * o.nl + 1;
    }
    o.row_counts = row_counts;
    o.draft_rows = draft_rows;
    o.kstop = Some(kstop);
    if !o.grammar_dir.is_empty() {
        if !s2mode {
            return Err("--grammar-dir needs --serve".into());
        }
        if m.xgrammar == 0 {
            // a build without XGrammar serves without grammars: its hello leaves out `grammar`
            eprint!("# rank {}: --grammar-dir ignored: this engine was built without XGrammar (GR8_XGRAMMAR)\n", o.r);
            o.grammar_dir.clear();
        }
    }
    // GR8_PF_OWN=4 (TP4) env refusals before any CUDA or fabric work
    let own4 = mp != 0 && k.int(kid::PF_OWN) == 4;
    o.own4 = own4;
    o.qp_roll_s = k.int(kid::PF_QP_ROLL) as i32;
    o.qp_roll_ser = k.flag(kid::PF_QP_ROLL_SER);
    let wire_e = k.int_opt(kid::PF_QP_WIRE_E);
    o.qp_wire_e = wire_e.unwrap_or(1) as i32;
    let qp_ctx = k.int_opt(kid::PF_QP_CTX).map_or(String::new(), |v| v.to_string());
    let why = roll_env_refusal(own4, o.qp_roll_s, wire_e.is_some(), o.qp_wire_e, &qp_ctx, !o.qp_file.is_empty());
    if !why.is_empty() {
        return Err(why);
    }
    if own4 && nr != 4 {
        return Err("GR8_PF_OWN=4 is TP4 only (G6_NR=4)".into());
    }
    if own4 && o.qp_roll_s == 0 && (o.qp_man.is_empty() || o.qp_file.is_empty()) {
        return Err("GR8_PF_OWN=4 needs --qp-manifest and --qp-pack (or GR8_PF_QP_ROLL)".into());
    }
    if !own4 && (!o.qp_man.is_empty() || !o.qp_file.is_empty()) {
        return Err("--qp-manifest / --qp-pack need GR8_PF_OWN=4 and --prefill".into());
    }
    Ok(o)
}

// ------------------------------------------------------------------------------------------------------ the C ABI
// scripts/check_options_coverage.py reads the refusals above this banner and finds it by its text

/// The options' strings and lists, which the view points into.
#[derive(Default)]
struct Store {
    strs: Vec<std::ffi::CString>,
    lists: Vec<Vec<*const c_char>>,
    ints: Vec<Vec<i32>>,
    f64s: Vec<Vec<f64>>,
    plans: Vec<CPlan>,
}

impl Store {
    fn s(&mut self, v: &[u8]) -> *const c_char {
        let c = cs(v);
        let p = c.as_ptr();
        self.strs.push(c);
        p
    }
    fn strs(&mut self, v: &[Str]) -> Strs {
        let l: Vec<*const c_char> = v.iter().map(|x| self.s(x)).collect();
        let s = Strs { p: l.as_ptr(), n: l.len() };
        self.lists.push(l);
        s
    }
    fn ints(&mut self, v: &[i32]) -> Ints {
        let l = v.to_vec();
        let s = Ints { p: l.as_ptr(), n: l.len() };
        self.ints.push(l);
        s
    }
    fn plan(&mut self, p: &Plan) -> CPlan {
        CPlan {
            prompt: self.ints(&p.prompt),
            first: p.first,
            chunk: p.chunk,
            n_gen: p.n_gen,
            inv_temp: p.inv_temp,
            seed: p.seed,
            top_p: p.top_p,
            lp: p.lp,
            plp: p.plp,
            rs: p.rs,
        }
    }
}

fn view(o: &Opts, st: &mut Store) -> Options {
    let co = &o.co;
    let ks = o.kstop.as_ref().expect("a parse that passed sets kstop");
    let tau: Vec<f64> = ks.b.tau.iter().flatten().copied().collect();
    let (tau_p, tau_n) = (tau.as_ptr(), tau.len());
    st.f64s.push(tau);
    let plans: Vec<CPlan> = o.plans.iter().map(|p| st.plan(p)).collect();
    st.plans = plans;
    Options {
        rank: o.rank,
        ips: st.strs(&o.ips),
        ips2: st.strs(&o.ips2),
        ring: o.ring,
        ring_devs: st.strs(&o.ring_devs),
        proxy_cpu: o.proxy_cpu,
        pf_ips2: st.strs(&o.pf_ips2),
        manifest: st.s(&o.manifest),
        packfile: st.s(&o.packfile),
        v2_dir: st.s(&co.v2_dir),
        v2_slices: st.s(&co.v2_slices),
        v2_widths: st.s(&co.v2_widths),
        v2_dense: st.s(&co.v2_dense),
        v2_layout: st.s(&co.v2_layout),
        v2_trellis: co.v2_trellis,
        qo_man: st.s(&co.qo_man),
        qo_file: st.s(&co.qo_file),
        layers_s: st.s(&co.layers_s),
        levels_s: st.s(&co.levels_s),
        prompts: st.strs(&co.prompts),
        co_cap: co.cap,
        co_mp: co.mp,
        co_top: co.top,
        co_pfrows: co.pfrows,
        co_mtp: co.mtp,
        co_lanes: co.lanes,
        co_nc: co.nc,
        co_temp: co.temp,
        co_graph: co.graph,
        co_wide: co.wide,
        co_pfv2: co.pfv2,
        co_adapt: co.adapt,
        has_n_gen: co.n_gen.is_some(),
        n_gen: co.n_gen.unwrap_or(0),
        has_chunk: co.chunk.is_some(),
        chunk: co.chunk.unwrap_or(0),
        has_seed: co.seed.is_some(),
        seed: co.seed.unwrap_or(0),
        has_top_p: co.top_p.is_some(),
        top_p: co.top_p.unwrap_or(0.0),
        has_rs: co.rs.is_some(),
        rs: co.rs.unwrap_or(false),
        qp_man: st.s(&o.qp_man),
        qp_file: st.s(&o.qp_file),
        dfl_man: st.s(&o.dfl_man),
        dfl_file: st.s(&o.dfl_file),
        dfl_sel: o.dfl_sel,
        dfl_k: o.dfl_k,
        spec_reset_per_request: o.spec_reset_per_request,
        spec_fixed: o.spec_fixed,
        plan: st.plan(&o.plan),
        image: o.image,
        hash: o.hash,
        reuse_prefix: o.reuse_prefix,
        check_args: o.check_args,
        lanes_explicit: o.lanes_explicit,
        s2dir: st.s(&o.s2dir),
        grammar_dir: st.s(&o.grammar_dir),
        kv_tiers: st.strs(&o.kv_tiers),
        kv_keep: st.s(&o.kv_keep),
        kv_system_after: o.kv_system_after,
        kv_write_limit: o.kv_write_limit,
        kv_format: st.s(&o.kv_format),
        kv_old: o.kv_old,
        kv_old_all: o.kv_old_all,
        set_id: st.s(&o.set_id),
        steering_dir: st.s(&o.steering_dir),
        steering_force_s: st.s(&o.steering_force_s),
        steering_model_revision: st.s(&o.steering_model_revision),
        steering_tokenizer_revision: st.s(&o.steering_tokenizer_revision),
        steering_template_digest: st.s(&o.steering_template_digest),
        steering_weight_set: st.s(&o.steering_weight_set),
        steering_template_file: st.s(&o.steering_template_file),
        s2port: o.s2port,
        s2sess: o.s2sess,
        s2epoch: o.s2epoch,
        cap: o.cap,
        top: o.top,
        pfrows: o.pfrows,
        nc: o.nc,
        mtp: o.mtp,
        mp: o.mp,
        lanes: o.lanes,
        temp: o.temp,
        graph: o.graph,
        wide: o.wide,
        pfv2: o.pfv2,
        adapt: o.adapt,
        release: o.release,
        s2mode: o.s2mode,
        dfl: o.dfl,
        k: o.k,
        layers: st.ints(&o.layers),
        progress_isolate: o.progress_isolate,
        route_split_mode: o.route_split_mode,
        c8_pack_env: o.c8_pack_env,
        spec_on: o.spec.on,
        spec_levels: st.ints(o.spec.levels()),
        spec_draft_round: o.spec.draft_round,
        spec_draft_each: o.spec.draft_each,
        spec_table: o.spec.table,
        rt_win: o.rt_win,
        kstop_mode: ks.mode,
        kstop_kmax: ks.kmax,
        kstop_tau: ks.tau,
        kstop_seed3: ks.seed3,
        kstop_log: ks.log,
        kstop_solo: ks.solo,
        kstopb_mode: ks.b.mode,
        kstopb_kmin: ks.b.kmin,
        kstopb_tau: tau_p,
        kstopb_tau_n: tau_n,
        kstopb_widths: st.ints(&ks.b.widths),
        kstopb_explore: ks.b.explore,
        flash_cache: o.flash_cache,
        flash_budget_mib: o.flash_budget_mib,
        ds_moe_main_topk: o.ds_moe_main_topk,
        ds_moe_decode_topk: o.ds_moe_decode_topk,
        ds_idx_pipe_b: o.ds_idx_pipe_b,
        ds_idx_q_trig: o.ds_idx_q_trig,
        ds_idx_pipe_qr: o.ds_idx_pipe_qr,
        ds_admit_keep_scratch: o.ds_admit_keep_scratch,
        ds_sess_drafts: o.ds_sess_drafts,
        ds_draft_cap: o.ds_draft_cap,
        plans: st.plans.as_ptr(),
        n_plans: st.plans.len(),
        row_counts: st.ints(&o.row_counts),
        draft_rows: st.ints(&o.draft_rows),
        nl: o.nl,
        nx: o.nx,
        ncoll: o.ncoll,
        r: o.r,
        own4: o.own4,
        qp_roll_s: o.qp_roll_s,
        qp_roll_ser: o.qp_roll_ser,
        qp_wire_e: o.qp_wire_e,
    }
}

/// g6opt_parsed: the refusal ("" when the options pass) and, when they pass, the view and its storage
pub struct Parsed {
    refusal: std::ffi::CString,
    view: Option<Box<Options>>,
    _store: Store,
}

/// Parses argv (argv[0] is the program). Free the result with g6opt_free.
///
/// # Safety
/// `m`, `k` and `h` must be valid; argv must hold argc NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn g6opt_parse(
    m: *const ModuleInfo,
    k: *const KnobApi,
    h: *const Host,
    argc: c_int,
    argv: *const *const c_char,
) -> *mut Parsed {
    let (m, k, h) = (&*m, &*k, &*h);
    let args: Vec<Str> = (0..argc.max(0) as usize).map(|i| CStr::from_ptr(*argv.add(i)).to_bytes().to_vec()).collect();
    let mut store = Store::default();
    let r = if m.abi != MODULE_ABI || k.table_hash != kid::TABLE_HASH {
        Err("scheduler-rs: the module description or the knob table differs from libg6host.a's (rebuild it)".into())
    } else {
        parse(m, k, h, &args)
    };
    let (refusal, view) = match r {
        Ok(o) => (Str::new(), Some(Box::new(view(&o, &mut store)))),
        Err(e) => (e, None),
    };
    Box::into_raw(Box::new(Parsed {
        refusal: cs(&refusal),
        view,
        _store: store,
    }))
}

/// "" when the options pass, else the message main dies with
///
/// # Safety
/// `p` must come from g6opt_parse.
#[no_mangle]
pub unsafe extern "C" fn g6opt_refusal(p: *const Parsed) -> *const c_char {
    (*p).refusal.as_ptr()
}

/// The options, or null after a refusal. They live until g6opt_free.
///
/// # Safety
/// `p` must come from g6opt_parse.
#[no_mangle]
pub unsafe extern "C" fn g6opt_options_of(p: *const Parsed) -> *const Options {
    (*p).view.as_deref().map_or(std::ptr::null(), |v| v as *const Options)
}

/// # Safety
/// `p` must come from g6opt_parse, once.
#[no_mangle]
pub unsafe extern "C" fn g6opt_free(p: *mut Parsed) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    static ENV: Mutex<Option<HashMap<u32, String>>> = Mutex::new(None);

    fn get(id: u32) -> Option<String> {
        ENV.lock().unwrap().as_ref().and_then(|e| e.get(&id).cloned())
    }
    unsafe extern "C" fn flag(id: u32, _: u32, out: *mut c_int) -> c_int {
        *out = get(id).is_some_and(|v| v == "1") as c_int;
        1
    }
    unsafe extern "C" fn int_(id: u32, _: u32, out: *mut i64) -> c_int {
        match get(id) {
            Some(v) => {
                *out = v.parse().unwrap();
                1
            }
            // the defaults the parse depends on in these cases
            None if id == kid::MOE_TOPK => {
                *out = 8;
                1
            }
            None if id == kid::MTP_KSTOPB_KMIN || id == kid::MTP_KSTOPB_EXPLORE || id == kid::MTP_LPLOG => {
                *out = 0;
                1
            }
            None if id == kid::RT_LOCAL || id == kid::PF_OWN || id == kid::PF_QP_ROLL || id == kid::GLM53F_STATE_BUDGET_MIB => {
                *out = 0;
                1
            }
            None => 0,
        }
    }
    unsafe extern "C" fn float_(_: u32, _: u32, out: *mut f64) -> c_int {
        *out = 0.55;
        1
    }
    unsafe extern "C" fn text(id: u32, _: u32) -> *const c_char {
        if id == kid::GLM53F_CACHE {
            c"off".as_ptr()
        } else if id == kid::KVB {
            c"bf16".as_ptr()
        } else {
            std::ptr::null()
        }
    }
    unsafe extern "C" fn ints(_: u32, _: u32, _: *mut *const i64) -> isize {
        -1
    }
    unsafe extern "C" fn floats(_: u32, _: u32, _: *mut *const f64) -> isize {
        -1
    }
    unsafe extern "C" fn tests_set() -> *const c_char {
        c"".as_ptr()
    }

    fn glm() -> ModuleInfo {
        let mut m = ModuleInfo {
            abi: MODULE_ABI,
            model: crate::module_info::MODEL_GLM53,
            name: [0; 16],
            nr: 4,
            mx: 32,
            s2n: 8,
            max_sessions: 8,
            hidden: 6144,
            layers: 78,
            mtp_layers: 1,
            vocab: 154880,
            experts: 256,
            topk: 8,
            kv_row_bytes: 656,
            kv_page: 64,
            kv_format: [0; 16],
            caps: 0x320,
            dfl_block: 8,
            pack_members: 8,
            dfl_drafts: 7,
            dfl_rep: 0,
            tp_sort: 2048,
            kv4: 0,
            xgrammar: 0,
            ds_markov_maxs: 0,
            ds_block: 0,
        };
        crate::module_info::set_name(&mut m.name, c"glm-5.3");
        crate::module_info::set_name(&mut m.kv_format, c"fp8");
        m
    }

    fn run_bytes(args: &[&[u8]]) -> Result<Opts, Str> {
        *ENV.lock().unwrap() = Some(HashMap::new());
        let api = KnobApi {
            table_hash: kid::TABLE_HASH,
            flag: Some(flag),
            int_: Some(int_),
            float_: Some(float_),
            text: Some(text),
            ints: Some(ints),
            floats: Some(floats),
            tests_set: Some(tests_set),
        };
        let host = Host {
            ctx: std::ptr::null_mut(),
            engram_colls: None,
            dsk_mtp: 0,
            dsk_relaxed: false,
            dsk_relax_topp: 0.0,
            dsk_err: std::ptr::null(),
        };
        let argv: Vec<Str> = std::iter::once(&b"g6m4"[..]).chain(args.iter().copied()).map(<[u8]>::to_vec).collect();
        parse(&glm(), &api, &host, &argv)
    }

    fn run(args: &[&str]) -> Result<Opts, Str> {
        run_bytes(&args.iter().map(|a| a.as_bytes()).collect::<Vec<_>>())
    }

    const BASE: [&str; 8] = ["--rank", "1", "--ips", "a,b,c,d", "--manifest", "m", "--pack", "p"];

    #[test]
    fn refusals_and_rows() {
        assert_eq!(run(&["--bogus"]).unwrap_err(), b"unknown argument --bogus");
        assert_eq!(run(&["--rank"]).unwrap_err(), b"missing value for --rank");
        assert!(run(&["--rank", "1"]).unwrap_err().starts_with(b"usage: g6m4 --rank R --ips NR,addresses (this build: NR = 4)"));
        let o = run(&BASE).unwrap();
        assert_eq!(o.plans.len(), 1);
        assert_eq!(o.plans[0].prompt, vec![154822, 154824, 154826, 25062]);
        assert_eq!(o.row_counts, vec![4, 1]);
        assert_eq!((o.nl, o.ncoll, o.kv_keep.as_slice(), o.kv_format.as_slice()), (4, 9, &b"system"[..], &b"fp8"[..]));
        let mut a = BASE.to_vec();
        a.extend(["--chunk", "0"]);
        assert_eq!(run(&a).unwrap_err(), b"--chunk must be 1..32");
        let mut a = BASE.to_vec();
        a.extend(["--set-id", "AB"]);
        assert_eq!(run(&a).unwrap_err(), b"--set-id must be 1..64 lowercase hex digits");
    }

    // a Latin-1 path and argument reach the options and the refusal byte for byte, as in the C++
    #[test]
    fn values_keep_their_bytes() {
        let path: &[u8] = b"/models/caf\xe9.pack";
        let mut a: Vec<&[u8]> = BASE.iter().map(|x| x.as_bytes()).collect();
        a[7] = path;
        let o = run_bytes(&a).unwrap();
        assert_eq!(o.packfile, path);
        let mut st = Store::default();
        let v = view(&o, &mut st);
        // SAFETY: the view's strings live in st
        assert_eq!(unsafe { CStr::from_ptr(v.packfile) }.to_bytes(), path);
        assert_eq!(run_bytes(&[b"--x\xff"]).unwrap_err(), b"unknown argument --x\xff");
    }

    #[test]
    fn waterfill_boot_widths_margin_and_model_refusal() {
        use std::cell::Cell;
        thread_local! {
            static NG: Cell<bool> = const { Cell::new(false) };
            static MARGIN: Cell<f64> = const { Cell::new(0.0) };
        }
        unsafe extern "C" fn wf_flag(id: u32, _: u32, out: *mut c_int) -> c_int {
            *out = (id == kid::S2_SHARED_WATERFILL) as c_int; 1
        }
        unsafe extern "C" fn wf_float(id: u32, n: u32, out: *mut f64) -> c_int {
            if id == kid::S2_SHARED_WF_MARGIN { *out = MARGIN.with(Cell::get); 1 } else { float_(id, n, out) }
        }
        unsafe extern "C" fn wf_text(id: u32, n: u32) -> *const c_char {
            if id == kid::NGRAM && NG.with(Cell::get) { c"table".as_ptr() } else { text(id, n) }
        }
        let api = KnobApi { table_hash: kid::TABLE_HASH, flag: Some(wf_flag), int_: Some(int_),
            float_: Some(wf_float), text: Some(wf_text), ints: Some(ints), floats: Some(floats), tests_set: Some(tests_set) };
        let host = Host { ctx: std::ptr::null_mut(), engram_colls: None, dsk_mtp: 0,
            dsk_relaxed: false, dsk_relax_topp: 0.0, dsk_err: std::ptr::null() };
        for model in [crate::module_info::MODEL_GLM53, MODEL_GLM53F] {
            let mut m = glm(); m.model = model;
            for ng in [false, true] {
                NG.with(|v| v.set(ng));
                for n in [2, 4, 8] {
                    let mut args: Vec<Str> = std::iter::once("g6m4").chain(BASE).map(|s| s.as_bytes().to_vec()).collect();
                    args.extend(["--serve", "session", "--graph", "--mtp", "3", "--prefill", "64", "--s2-port", "1", "--s2-sessions"].map(|s| s.as_bytes().to_vec()));
                    args.push(n.to_string().into_bytes());
                    let o = parse(&m, &api, &host, &args).unwrap();
                    let max_m = 32.min(n * if ng { 8 } else { 4 });
                    for width in 1..=max_m {
                        assert!(o.row_counts.contains(&width), "MAIN model={model} c={n} width={width}");
                        assert!(o.draft_rows.contains(&width), "DRAFT model={model} c={n} width={width}");
                    }
                }
            }
        }
        let mut ds = glm(); ds.model = MODEL_DS41;
        assert!(String::from_utf8(parse(&ds, &api, &host, &[]).unwrap_err()).unwrap().contains("uniform rows"));
        for margin in [-1.0, f64::NAN, f64::INFINITY] {
            MARGIN.with(|v| v.set(margin));
            assert_eq!(parse(&glm(), &api, &host, &[]).unwrap_err(),
                b"GR8_S2_SHARED_WF_MARGIN: want a finite nonnegative value");
        }
    }

    #[test]
    fn helpers() {
        assert_eq!(split(b"a,,b,", b','), vec![b"a".to_vec(), b"".to_vec(), b"b".to_vec()]);
        assert!(split(b"", b',').is_empty());
        assert_eq!(parse_ints(b"1,,2,", 78), vec![1, 0, 2]);
        assert_eq!(parse_ints(b"all", 3), vec![0, 1, 2]);
        assert_eq!(roll_env_refusal(true, 2, false, 1, "0", false), b"GR8_PF_QP_CTX='0': a positive decimal position count");
        assert_eq!(alt_hex(0x120), "0x120");
        assert_eq!(alt_hex(0), "0");
    }
}
