//! gr8's host-side S2 scheduling logic (docs/HOST-CORE.md, R2 and R3), moved from C++ headers one piece at a time. The
//! C++ headers keep their types and call these functions through `g6sched.h`. For the R2 pieces the state they act on
//! stays in C++ structs with the same layout (`#[repr(C)]`), so C++ copies and owns it as before. The batch planner
//! (R3, `planner`) owns its state here, behind an opaque handle. The C++ passes settings in and reads results out.
//! The driver (`driver`) owns the scheduler's state and its per-slot state here too; the C++ reads the slots through a
//! view. It reaches the planner through the C++ `s2::Batch`, which writes the round log.
//!
//! Every function here is deterministic in its inputs: each rank runs the same calls on the same replicated state and
//! must reach the same answer.

pub mod admission;
pub mod cp_rows;
pub mod driver;
pub mod ffi;
pub mod flash_lend;
pub mod flash_ring;
pub mod kv_plan;
pub mod kv_share;
pub mod layout;
pub mod ngram;
pub mod planner;
pub mod planner_ffi;
pub mod request;
pub mod scheduler;
pub mod spec_len;
