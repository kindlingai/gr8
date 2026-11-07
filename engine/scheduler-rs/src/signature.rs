//! The S2 runtime signature (s2/signature.h): the string every rank sends to the bootstrap agreement, so ranks whose
//! options, knobs or loaded files would plan differently refuse to start instead of diverging. One format per model.
//! The inputs include values main has only after the load and strings other builders make (the knob signature, the
//! n-gram and KStopB parts, the drafter choice's table), which arrive here finished.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

use crate::cstd::to_string_f;

pub const SIG_GLM: u32 = 0;
pub const SIG_FLASH: u32 = 1;
pub const SIG_DS41: u32 = 2;

/// g6sig_inputs: s2/signature.h SigInputs with its strings as C strings (null reads as empty)
#[repr(C)]
pub struct SigInputs {
    pub model: u32,
    pub cap: i32,
    pub layers_s: *const c_char,
    pub chunk: i32,
    pub nc: i32,
    pub set_id: *const c_char,
    pub dfl: bool,
    pub dfl_k: i32,
    pub s2sess: i32,
    pub knobs: *const c_char,
    pub mp: i32,
    pub mtp: i32,
    pub dfl_reuse: i32,
    pub dfl_reuse_min: i32,
    pub lanes: i32,
    pub gsplit: i32,
    pub ckpt: *const c_char,
    pub ckpt_blocks: u64,
    pub ckpt_persist: bool,
    pub s2_below: i32,
    pub two: bool,
    pub pf_own: i32,
    pub pf_qp: bool,
    pub qp_ctx: i32,
    pub qp_min: i32,
    pub grammar: bool,
    pub ngram: *const c_char,
    pub kstop_mode: i32,
    pub kstop_kmax: i32,
    pub kstop_tau: [f64; 4],
    pub kstop_seed3: f64,
    pub draft_par: i32,
    pub qs_mode: i32,
    pub iq_split: bool,
    pub mtp_split: bool,
    pub dhead_on: i32,
    pub dhead_tpc: i32,
    pub dhead_ks: i32,
    pub mtp_last: bool,
    pub dhead2: i32,
    pub dfl_i4: bool,
    pub dfl_body_i4: bool,
    pub dfl_alpha: f32,
    pub dsel_model: i32,
    pub dsel_tab: *const c_char,
    pub kstopb: *const c_char,
    pub cp_step: i32,
    pub cp_seg: bool,
    pub ctx_sp: i32,
    pub posidx: i32,
    pub steer_shared: bool,
    pub ds_mtp: i32,
    pub sh_tc: bool,
    pub l2hint: i32,
    pub fc2stg: i32,
    pub fc1bstg: i32,
    pub c8_pack: bool,
    pub sess_drafts: bool,
    pub pf_lanes: i32,
    pub tk_main: i32,
    pub tk_decode: i32,
    pub idx_pipe_b: i32,
    pub idx_q_trig: i32,
    pub idx_pipe_qr: i32,
    pub reuse: bool,
    pub vis_on: bool,
    pub vis_id: *const c_char,
}

fn text(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the caller passes NUL-terminated strings that outlive the call
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn when(c: bool, s: impl FnOnce() -> String) -> String {
    if c {
        s()
    } else {
        String::new()
    }
}

pub fn runtime_signature(i: &SigInputs) -> String {
    let (layers, set, knobs) = (text(i.layers_s), text(i.set_id), text(i.knobs));
    let dflreuse = || format!(" dflreuse {}/{}", i.dfl_reuse, i.dfl_reuse_min);
    if i.model == SIG_FLASH {
        let mut s = format!(
            "flash solo cap {} layers {layers} mp {} lanes {} mtp {} chunk {} cands {} set {set}",
            i.cap, i.mp, i.lanes, i.mtp, i.chunk, i.nc
        );
        s += &when(i.dfl, || format!(" dflash k {}", i.dfl_k));
        s += &when(i.dfl_reuse != 0, dflreuse);
        s += &when(i.s2sess > 1, || format!(" sessions {} split {}", i.s2sess, i.gsplit));
        s += &knobs;
        // every rank must keep the same checkpoint entries: the mode, the store's blocks and whether they persist
        let ckpt = text(i.ckpt);
        if !ckpt.is_empty() {
            s += &format!(" ckpt {ckpt} blocks {}{}", i.ckpt_blocks, if i.ckpt_persist { " persist" } else { "" });
        }
        return s;
    }
    if i.model == SIG_DS41 {
        let mut s = if i.s2sess > 1 { format!("ds41 sessions {}", i.s2sess) } else { "ds41 solo".to_string() };
        s += &format!(" cap {} layers {layers} mtp {} chunk {} cands {} set {set}", i.cap, i.ds_mtp, i.chunk, i.nc);
        s += &when(i.dfl, || format!(" dflash k {}", i.dfl_k));
        s += &format!(" shtc {}", i.sh_tc as i32);
        s += &when(i.l2hint != 0, || format!(" l2h {}", i.l2hint));
        s += &when(i.fc2stg != 0, || " stg 1".into());
        s += &when(i.fc1bstg != 0, || " fc1bstg 1".into());
        s += &when(i.c8_pack, || " c8pack 1".into());
        s += &when(i.sess_drafts, || " sessdrafts 1".into());
        s += &when(i.pf_lanes == 2, || " pflanes 2".into());
        s += &when(i.tk_decode != i.tk_main, || format!(" tkmain {} tkdecode {}", i.tk_main, i.tk_decode));
        s += &when(i.idx_pipe_b != 0, || " idxpipeb 1".into());
        s += &when(i.idx_q_trig != 0, || " idxqtrig 1".into());
        s += &when(i.idx_pipe_qr != 0, || " idxpipeqr 1".into());
        s += &when(i.reuse, || " reuse 1".into());
        // with the tower, " vision <file id>" (header and size agreement, not tensor data); a rank without the file
        // boots text-only and refuses beside ranks with it
        s += &when(i.vis_on, || format!(" vision {}", text(i.vis_id)));
        return s + &knobs;
    }
    let mut s = format!(
        "g6m4 s2 below {} cap {} layers {layers} mp {} lanes {} mtp {} chunk {} cands {} split 3 own {} qpctx {} qpmin {}",
        i.s2_below,
        i.cap,
        i.mp,
        if i.two { 2 } else { 1 },
        i.mtp,
        i.chunk,
        i.nc,
        i.pf_own,
        if i.pf_qp { i.qp_ctx } else { 0 },
        if i.pf_qp { i.qp_min } else { 0 }
    );
    s += &when(!set.is_empty(), || format!(" set {set}"));
    s += &when(i.grammar, || " grammar".into());
    s += &when(i.dfl_reuse != 0, dflreuse);
    s += &text(i.ngram);
    s += &when(i.kstop_mode != 0, || {
        format!(
            " kstop {} kmax {} tau {},{},{} seed3 {}",
            i.kstop_mode,
            i.kstop_kmax,
            to_string_f(i.kstop_tau[1]),
            to_string_f(i.kstop_tau[2]),
            to_string_f(i.kstop_tau[3]),
            to_string_f(i.kstop_seed3)
        )
    });
    s += &when(i.draft_par != 0, || format!(" draftpar {}", i.draft_par));
    s += &when(i.qs_mode != 0, || format!(" qasplit {}", i.qs_mode));
    s += &when(i.iq_split, || " idxqsplit 1".into());
    s += &when(i.mtp_split, || " mtpsplit 1".into());
    s += &when(i.dhead_on != 0, || format!(" dheadi4 {}x{}", i.dhead_tpc, i.dhead_ks));
    s += &when(i.mtp_last, || " mtplast 1".into());
    s += &when(i.dhead2 != 0, || format!(" dhead2 {}", i.dhead2));
    s += &when(i.dfl_i4, || format!(" dflheadi4 {}x{}", i.dhead_tpc, i.dhead_ks));
    s += &when(i.dfl_body_i4, || " dflbodyi4 1".into());
    s += &when(i.dfl_alpha != 1.0, || format!(" dflalpha {}", to_string_f(i.dfl_alpha as f64)));
    s += &when(i.dsel_model != 0, || format!(" dselmodel {} {}", i.dsel_model, text(i.dsel_tab)));
    s += &text(i.kstopb);
    s += &when(i.cp_step != 0, || format!(" cpsr {}{}", i.cp_step, if i.cp_seg { " seg" } else { "" }));
    s += &when(i.ctx_sp != 0, || format!(" ctxsp {}", i.ctx_sp));
    s += &when(i.posidx != 0, || " posidx 1".into());
    s += &when(i.steer_shared, || " s2steer 1".into());
    s + &knobs
}

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

/// The signature, valid until this thread's next call.
///
/// # Safety
/// `i` must point to a valid g6sig_inputs whose strings are NUL-terminated or null.
#[no_mangle]
pub unsafe extern "C" fn g6sig_runtime(i: *const SigInputs) -> *const c_char {
    let s = runtime_signature(&*i);
    LAST.with(|l| {
        *l.borrow_mut() = CString::new(s).unwrap_or_default();
        l.borrow().as_ptr()
    })
}
