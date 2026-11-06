//! `g6s2_module_ops` (g6sched.h, s2/module.h): the service's calls on a model module, as one table of function pointers
//! and the module's context. No Rust calls the table yet. These types fix its layout (layout.rs) for the Rust service.
//! Each call returns 0, or 1 when the module threw (the C++ caller rethrows). A `None` member is a call the module does
//! not have.

use crate::kv_share::Resize;
use crate::planner::{DflDrafts, DflReq};
use crate::scheduler::{Run, Slot};
use std::ffi::c_void;

/// C++ `g6s2::Launch`, passed through.
#[repr(C)]
pub struct Launch {
    _p: [u8; 0],
}
/// C++ `g6s2::Payload`, passed through.
#[repr(C)]
pub struct Payload {
    _p: [u8; 0],
}
/// C++ `std::vector<g6s2::PrefillPart>`, passed through.
#[repr(C)]
pub struct Parts {
    _p: [u8; 0],
}
/// C++ `std::vector<int>`: launch assigns its output there.
#[repr(C)]
pub struct Ids {
    _p: [u8; 0],
}
/// C++ `g6rt::ServeReq`, passed through.
#[repr(C)]
pub struct ServeReq {
    _p: [u8; 0],
}
/// C++ `g6rt::PrefixReuse`, passed through.
#[repr(C)]
pub struct Reuse {
    _p: [u8; 0],
}

/// A row's sampling (sessions.h Sampling).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sampling {
    pub inv_temp: f32,
    pub top_p: f32,
    pub seed: u32,
    pub lp: i32,
    pub plp: i32,
}

/// A row's logprob entry (serve.h LpRow; LP_MAX = 20).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LpRow {
    pub lp: f32,
    pub rank: i32,
    pub n: i32,
    pub id: [i32; 20],
    pub tlp: [f32; 20],
}

/// bind_request(slot, request)
pub type BindRequestFn = unsafe extern "C" fn(*mut c_void, u32, *const ServeReq) -> i32;
/// grown(slot, run)
pub type SlotRunFn = unsafe extern "C" fn(*mut c_void, u32, *const Run) -> i32;
/// retain(run, reuse state, namespace)
pub type RetainFn = unsafe extern "C" fn(*mut c_void, *const Run, *const Reuse, *const u8) -> i32;
/// park(slot, run, reuse state, namespace)
pub type ParkFn = unsafe extern "C" fn(*mut c_void, u32, *const Run, *const Reuse, *const u8) -> i32;
/// drained(*out)
pub type BoolOutFn = unsafe extern "C" fn(*mut c_void, *mut bool) -> i32;
/// resize(resize)
pub type ResizeFn = unsafe extern "C" fn(*mut c_void, *const Resize) -> i32;
/// flash_room(*out)
pub type U32OutFn = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
/// bucket(draft, context, *out)
pub type BucketFn = unsafe extern "C" fn(*mut c_void, bool, i32, *mut i32) -> i32;
/// mode(old solo, new solo)
pub type ModeFn = unsafe extern "C" fn(*mut c_void, i32, i32) -> i32;
/// launch(launch, *ids)
pub type LaunchFn = unsafe extern "C" fn(*mut c_void, *const Launch, *mut Ids) -> i32;
/// prefill(slot, payload, first, rows)
pub type ModPrefillFn = unsafe extern "C" fn(*mut c_void, *const Slot, *const Payload, u32, u32) -> i32;
/// prefill_seq(slot, payload, first, rows, sequence)
pub type PrefillSeqFn = unsafe extern "C" fn(*mut c_void, *const Slot, *const Payload, u32, u32, u64) -> i32;
/// prefill_pack(parts, sequence)
pub type PackSeqFn = unsafe extern "C" fn(*mut c_void, *const Parts, u64) -> i32;
/// logprob(row, target, k, *out)
pub type LogprobFn = unsafe extern "C" fn(*mut c_void, i32, bool, i32, *mut LpRow) -> i32;
/// dfl(request, sampling, *drafts)
pub type ModDflFn = unsafe extern "C" fn(*mut c_void, *const DflReq, *const Sampling, *mut DflDrafts) -> i32;

/// g6s2_module_ops: one module's calls. Calls on one module never overlap.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModuleOps {
    pub ctx: *mut c_void,
    pub bind: Option<crate::driver::BindFn>,
    pub bind_request: Option<BindRequestFn>,
    pub grown: Option<SlotRunFn>,
    pub shrink: Option<crate::driver::ShrinkFn>,
    pub retire: Option<crate::driver::SlotFn>,
    pub retain: Option<RetainFn>,
    pub park: Option<ParkFn>,
    pub fence: Option<crate::driver::VoidFn>,
    pub before_work: Option<crate::driver::VoidFn>,
    pub drained: Option<BoolOutFn>,
    pub resize: Option<ResizeFn>,
    pub flash_room: Option<U32OutFn>,
    pub bucket: Option<BucketFn>,
    pub mode: Option<ModeFn>,
    pub launch: Option<LaunchFn>,
    pub prefill: Option<ModPrefillFn>,
    pub prefill_seq: Option<PrefillSeqFn>,
    pub prefill_pack: Option<PackSeqFn>,
    pub logprob: Option<LogprobFn>,
    pub draft_lp: Option<crate::planner_ffi::LpFn>,
    pub dfl_ok: Option<crate::driver::QBoolFn>,
    pub dfl_start: Option<crate::driver::QIntFn>,
    pub dfl: Option<ModDflFn>,
    pub gm_on: *mut i32,
    pub gm_bits: *mut u32,
    pub gm_words: i32,
    pub pen_rows: *mut c_void,
    pub pen_sess: *mut c_void,
}
