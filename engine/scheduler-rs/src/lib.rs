//! gr8's host-side S2 scheduling logic (docs/HOST-CORE.md, R2), moved from C++ headers one piece at a time. The C++
//! headers keep their types and call these functions through `g6sched.h`; the state they act on stays in C++ structs
//! with the same layout (`#[repr(C)]`), so C++ copies and owns it as before.
//!
//! Every function here is deterministic in its inputs: each rank runs the same calls on the same replicated state and
//! must reach the same answer.

pub mod admission;
pub mod cp_rows;
pub mod ffi;
pub mod flash_lend;
pub mod flash_ring;
pub mod kv_plan;
pub mod kv_share;
pub mod layout;
pub mod request;
pub mod scheduler;
