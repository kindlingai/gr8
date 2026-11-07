//! `g6s2_pool_ops` (g6sched.h, s2/pool_ops.h): a module's policy over its rank's pool, as one table of function
//! pointers and the module's context. No Rust calls the table yet. These types fix its layout (layout.rs) for the Rust
//! service. Each call returns 0, or 1 when the module threw (the C++ caller rethrows). A `None` member is a call the
//! module does not have. The service makes the plain pool calls on `pool` itself.

use crate::scheduler::{Request, Run, Shrink, Slot};
use std::ffi::{c_char, c_void};

/// pool.h `PoolHit`: what a prompt can reuse. why: a static string.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PoolHit {
    pub skip: i32,
    pub lcp: i32,
    pub why: *const c_char,
    pub age: u64,
    pub stored: bool,
    pub tier: u32,
    pub slot: i32,
}

/// C++ `std::string`: refusal assigns its output there.
#[repr(C)]
pub struct Str {
    _p: [u8; 0],
}
/// C++ `std::vector<g6kv_tier_totals>`: tiers assigns its output there.
#[repr(C)]
pub struct Tiers {
    _p: [u8; 0],
}

/// match(prompt, n, rows, worst, limit, namespace, r, *hit)
pub type MatchFn =
    unsafe extern "C" fn(*mut c_void, *const i32, usize, u64, u64, i32, *const u8, u32, *mut PoolHit) -> i32;
/// take(hit, rows, worst, r, *run)
pub type TakeFn = unsafe extern "C" fn(*mut c_void, *const PoolHit, u64, u64, u32, *mut Run) -> i32;
/// digest(*out)
pub type U64OutFn = unsafe extern "C" fn(*mut c_void, *mut u64) -> i32;
/// tiers(*tiers)
pub type TiersFn = unsafe extern "C" fn(*mut c_void, *mut Tiers) -> i32;
/// refusal(worst, r, *why)
pub type RefusalFn = unsafe extern "C" fn(*mut c_void, u64, u32, *mut Str) -> i32;
/// r_rows(request, rows, worst, prompt, n, has_prompt, *out)
pub type RRowsFn =
    unsafe extern "C" fn(*mut c_void, *const Request, u64, u64, *const i32, usize, bool, *mut u32) -> i32;
/// make_room(slots, request, rows, worst, *made, *shrink)
pub type MakeRoomFn =
    unsafe extern "C" fn(*mut c_void, *const Slot, *const Request, u64, u64, *mut bool, *mut Shrink) -> i32;

/// g6s2_pool_ops: one module's pool policy. pool: the rank's g6kv_pool (kvman-rs `Handle`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolOps {
    pub ctx: *mut c_void,
    pub r#match: Option<MatchFn>,
    pub take: Option<TakeFn>,
    pub digest: Option<U64OutFn>,
    pub tiers: Option<TiersFn>,
    pub refusal: Option<RefusalFn>,
    pub r_rows: Option<RRowsFn>,
    pub make_room: Option<MakeRoomFn>,
    pub pool: *mut c_void,
    pub max_run_pages: u32,
    pub grow: bool,
    pub count: bool,
    pub status: bool,
}
