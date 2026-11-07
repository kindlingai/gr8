//! gr8's host-side S2 scheduling logic (docs/HOST-CORE.md, R2 and R3), moved from C++ headers one piece at a time. The
//! C++ headers keep their types and call these functions through `g6sched.h`. For the R2 pieces the state they act on
//! stays in C++ structs with the same layout (`#[repr(C)]`), so C++ copies and owns it as before. The batch planner
//! (R3, `planner`) owns its state here, behind an opaque handle. The C++ passes settings in and reads results out.
//! The driver (`driver`) owns the scheduler's state and its per-slot state here too; the C++ reads the slots through a
//! view. It reaches the planner through the C++ `s2::Batch`, which writes the round log.
//!
//! Every function here is deterministic in its inputs: each rank runs the same calls on the same replicated state and
//! must reach the same answer. The exceptions are `control`, the links between the ranks (bootstrap, frames, the event
//! gate and channels) and their socket I/O, and `service`, the S2 files and the trace.

pub mod admission;
pub mod control;
pub mod cp_rows;
pub mod cstd;
pub mod driver;
pub mod fabric_plan;
pub mod ffi;
pub mod flash_lend;
pub mod flash_ring;
pub mod kv_plan;
pub mod kv_share;
/// gr8_knobs.h's ids, which `knobs gen` writes
#[allow(dead_code)]
pub mod knob_ids {
    include!("../../knobs-rs/gr8_knob_ids.rs");
}
pub mod layout;
pub mod module;
pub mod module_info;
pub mod ngram;
pub mod options;
pub mod plan_sizes;
pub mod planner;
pub mod planner_ffi;
pub mod pool_ops;
pub mod request;
pub mod scheduler;
pub mod service;
pub mod signature;
pub mod spec_len;
