//! The service's C entry points (g6sched.h `g6svc_*`). Fallible calls return 0, or 2 when refused, with the message in
//! `g6ctl_error`, as the control links' calls do. Bytes go back through a grow callback.
#![allow(clippy::missing_safety_doc)]

use super::admit;
use super::control::{self, CControl, Control, ObserveFn};
use super::files::{self, Boot, Command, Ingress, Profile, Shape, Steering};
use super::status::{self, Growth, RunUse, Status};
use super::trace::{self, Trace};
use crate::control::ffi::{bytes, code, give, GrowFn};
use crate::kv_share::Totals;
use crate::scheduler::{Config, Identity, Request};
use std::ffi::c_void;

/// g6svc_profile: one loaded steering profile.
#[repr(C)]
pub struct CProfile {
    pub name: *const u8,
    pub name_n: usize,
    pub digest: [u8; 32],
    pub active_layers: u32,
}

/// g6svc_steering: the boot's steering receipt (steering.h BootConfig).
#[repr(C)]
pub struct CSteering {
    pub profiles: *const CProfile,
    pub n: usize,
    pub scope: u32,
    pub forced_scale_bits: u32,
    pub loaded: bool,
    pub forced: bool,
}

/// g6svc_boot: BootstrapConfig's fields that ready.s2 and the agreement read, and the module's limits (module_info.h)
/// its config is checked against.
#[repr(C)]
pub struct CBoot {
    pub scheduler: *const Config,
    pub steering: CSteering,
    pub max_sessions: u32,
    pub rank: u32,
    pub ranks: u32,
    pub prefill_min: u32,
    pub control_version: u32,
    pub lp_max: u32,
    pub directory: *const u8,
    pub directory_n: usize,
    pub signature: *const u8,
    pub signature_n: usize,
    pub mtp0: bool,
    pub pack: bool,
    pub park: bool,
}

/// g6svc_status: what status.s2 reports. The growth counters, tiers and runs are read only with their has_ flag.
#[repr(C)]
pub struct CStatus {
    pub epoch: u64,
    pub queued: u64,
    pub stalls: u64,
    pub grows: u64,
    pub stalled: u64,
    pub tiers: *const Totals,
    pub n_tiers: usize,
    pub runs: *const RunUse,
    pub n_runs: usize,
    pub pool_pages: u32,
    pub resident: u32,
    pub retiring: u32,
    pub free: u32,
    pub room: u32,
    pub held_pages: u32,
    pub worst_pages: u32,
    pub has_growth: bool,
    pub has_tiers: bool,
    pub has_runs: bool,
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

unsafe fn profiles(s: &CSteering) -> Vec<Profile<'_>> {
    slice(s.profiles, s.n)
        .iter()
        .map(|p| {
            let CProfile { name, name_n, digest, active_layers } = *p;
            Profile { name: bytes(name, name_n), digest, active_layers }
        })
        .collect()
}
fn steering<'a>(s: &CSteering, profiles: &'a [Profile<'a>]) -> Steering<'a> {
    let CSteering { profiles: _, n: _, scope, forced_scale_bits, loaded, forced } = *s;
    Steering { loaded, forced, scope, forced_scale_bits, profiles }
}
fn boot<'a>(b: &'a CBoot, profiles: &'a [Profile<'a>]) -> Boot<'a> {
    let CBoot {
        scheduler,
        steering: ref s,
        max_sessions,
        rank,
        ranks,
        prefill_min,
        control_version,
        lp_max,
        directory,
        directory_n,
        signature,
        signature_n,
        mtp0,
        pack,
        park,
    } = *b;
    // SAFETY: the C++ passes live pointers for the call
    unsafe {
        Boot {
            scheduler: &*scheduler,
            steering: steering(s, profiles),
            shape: Shape { max_sessions, mtp0, pack, park },
            rank,
            ranks,
            prefill_min,
            control_version,
            lp_max,
            directory: bytes(directory, directory_n),
            signature: bytes(signature, signature_n),
        }
    }
}

// --- files

/// FNV-1a over the bytes (the ADMIT's request hash).
#[no_mangle]
pub unsafe extern "C" fn g6svc_payload_hash(b: *const u8, n: usize) -> u64 {
    files::payload_hash(bytes(b, n))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_exists(p: *const u8, n: usize) -> bool {
    files::exists(bytes(p, n))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_read_file(p: *const u8, n: usize, grow: GrowFn, ctx: *mut c_void) -> i32 {
    code(files::read_file(bytes(p, n)).map(|b| give(&b, grow, ctx)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_atomic_file(p: *const u8, n: usize, b: *const u8, bn: usize) -> i32 {
    code(files::atomic_file(bytes(p, n), bytes(b, bn)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_number(s: *const u8, n: usize, out: *mut u64) -> i32 {
    code(files::number(bytes(s, n)).map(|v| *out = v))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_parse_command(t: *const u8, n: usize, epoch: u64, out: *mut Command) -> i32 {
    code(files::parse_command(bytes(t, n), epoch).map(|c| *out = c))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_ingress_new(dir: *const u8, n: usize, epoch: u64, out: *mut *mut Ingress) -> i32 {
    *out = std::ptr::null_mut();
    code(Ingress::new(bytes(dir, n), epoch).map(|i| *out = Box::into_raw(Box::new(i))))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_ingress_free(i: *mut Ingress) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_ingress_next(i: *const Ingress, has: *mut bool, out: *mut Command) -> i32 {
    *has = false;
    code((*i).next().map(|c| {
        if let Some(c) = c {
            *has = true;
            *out = c;
        }
    }))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_ingress_peek(i: *const Ingress, has: *mut bool, out: *mut Command) -> i32 {
    *has = false;
    code((*i).peek().map(|c| {
        if let Some(c) = c {
            *has = true;
            *out = c;
        }
    }))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_ingress_request(
    i: *const Ingress,
    c: *const Command,
    grow: GrowFn,
    ctx: *mut c_void,
) -> i32 {
    code((*i).request(&*c).map(|b| give(&b, grow, ctx)))
}

// --- ready.s2 and the agreement

#[no_mangle]
pub unsafe extern "C" fn g6svc_steering_check(s: *const CSteering) -> i32 {
    let p = profiles(&*s);
    code(files::steering_ok(&steering(&*s, &p)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_capability(b: *const CBoot, grow: GrowFn, ctx: *mut c_void) -> i32 {
    let p = profiles(&(*b).steering);
    let b = boot(&*b, &p);
    code(files::capability(b.scheduler, &b.steering, &b.shape, b.ranks, b.lp_max).map(|v| give(&v, grow, ctx)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_agreement(b: *const CBoot, grow: GrowFn, ctx: *mut c_void) -> i32 {
    let p = profiles(&(*b).steering);
    code(files::agreement(&boot(&*b, &p)).map(|v| give(&v, grow, ctx)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_publish_ready(b: *const CBoot, links: usize, missing: bool) -> i32 {
    let p = profiles(&(*b).steering);
    code(files::publish_ready(&boot(&*b, &p), links, missing))
}

// --- status.s2

#[no_mangle]
pub extern "C" fn g6svc_status_pool(free: u32, room: u32, pool_pages: u32) -> i32 {
    code(status::pool_ok(free, room, pool_pages))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_status_publish(s: *const CStatus, dir: *const u8, n: usize) -> i32 {
    let CStatus {
        epoch,
        queued,
        stalls,
        grows,
        stalled,
        tiers,
        n_tiers,
        runs,
        n_runs,
        pool_pages,
        resident,
        retiring,
        free,
        room,
        held_pages,
        worst_pages,
        has_growth,
        has_tiers,
        has_runs,
    } = *s;
    let st = Status {
        epoch,
        pool_pages,
        resident,
        queued,
        retiring,
        free,
        room,
        stalls,
        growth: has_growth.then_some(Growth { held_pages, worst_pages, grows, stalls: stalled }),
        tiers: has_tiers.then(|| slice(tiers, n_tiers)),
        runs: has_runs.then(|| slice(runs, n_runs)),
    };
    code(status::publish(bytes(dir, n), &st))
}

// --- the trace

/// PATH.r<rank>.json, or null (an empty path, or one that cannot be written: a line on stderr).
#[no_mangle]
pub unsafe extern "C" fn g6svc_trace_open(p: *const u8, n: usize, rank: u32) -> *mut Trace {
    Trace::open(bytes(p, n), rank).map_or(std::ptr::null_mut(), |t| Box::into_raw(Box::new(t)))
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_trace_free(t: *mut Trace) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_trace_span(
    t: *const Trace,
    name: *const u8,
    name_n: usize,
    tid: i32,
    t0: u64,
    t1: u64,
    args: *const u8,
    args_n: usize,
) {
    (*t).span(bytes(name, name_n), tid, t0, t1, bytes(args, args_n));
}

#[no_mangle]
pub unsafe extern "C" fn g6svc_trace_instant(
    t: *const Trace,
    name: *const u8,
    name_n: usize,
    tid: i32,
    args: *const u8,
    args_n: usize,
) {
    (*t).instant(bytes(name, name_n), tid, bytes(args, args_n));
}

#[no_mangle]
pub extern "C" fn g6svc_trace_now() -> u64 {
    trace::now()
}

// --- the admission decisions (admit.h)

/// held: the notice names `held_at`.
#[no_mangle]
pub extern "C" fn g6svc_notice_target(boundary: u64, horizon: u64, held: bool, held_at: u64, notice_max: u32) -> u64 {
    admit::notice_target(boundary, horizon, held.then_some(held_at), notice_max)
}
#[no_mangle]
pub extern "C" fn g6svc_admit_retry(head: u64, attempted_head: u64, revision: u64, attempted_revision: u64) -> bool {
    admit::retry(head, attempted_head, revision, attempted_revision)
}
#[no_mangle]
pub extern "C" fn g6svc_attempted_revision(revision: u64) -> u64 {
    admit::attempted_revision(revision)
}
#[no_mangle]
pub extern "C" fn g6svc_admit_shrink(fits: bool, can_make_room: bool, live: u32, limit: u32) -> bool {
    admit::shrink(fits, can_make_room, live, limit)
}
#[no_mangle]
pub extern "C" fn g6svc_admit_grant(why: u32, short_of: u32) -> bool {
    admit::grant(why, short_of)
}
#[no_mangle]
pub extern "C" fn g6svc_admit_drain(admit_hold: bool, live: u32, limit: u32) -> bool {
    admit::drain(admit_hold, live, limit)
}
#[no_mangle]
pub extern "C" fn g6svc_admit_hold(admit_hold: bool, queued: bool, live: u32, limit: u32) -> bool {
    admit::hold(admit_hold, queued, live, limit)
}
#[no_mangle]
pub extern "C" fn g6svc_wait_line(why: u32, last: u64, head: u64) -> bool {
    admit::wait_line(why, last, head)
}
#[no_mangle]
pub extern "C" fn g6svc_refusal_line(why: u32, last: u64, head: u64) -> bool {
    admit::refusal_line(why, last, head)
}
#[no_mangle]
pub extern "C" fn g6svc_recall_note(why: u32, short_of: u32, waiting: u64, head: u64) -> u32 {
    admit::recall_note(why, short_of, waiting, head)
}
#[no_mangle]
pub extern "C" fn g6svc_share_idle(resident: u32, queued: usize) -> bool {
    admit::share_idle(resident, queued)
}
#[no_mangle]
pub extern "C" fn g6svc_lend_idle(resident: u32, queued: usize, retiring: u32) -> bool {
    admit::lend_idle(resident, queued, retiring)
}
/// Sessions::request
#[no_mangle]
pub unsafe extern "C" fn g6svc_request(
    id: u64,
    prompt_tokens: usize,
    n_gen: i32,
    steering: *const Identity,
    out: *mut Request,
) {
    *out = control::request(id, prompt_tokens, n_gen, *steering);
}
/// steering.h cache_namespace: NAMESPACE_BYTES into out
#[no_mangle]
pub unsafe extern "C" fn g6svc_cache_namespace(s: *const Identity, out: *mut u8) {
    let n = control::cache_namespace(&*s);
    std::ptr::copy_nonoverlapping(n.as_ptr(), out, n.len());
}

// --- the head's control thread (control.rs) and the state the compute thread shares with it

#[no_mangle]
pub unsafe extern "C" fn g6svc_control_new(c: *const CControl, out: *mut *mut Control) -> i32 {
    *out = std::ptr::null_mut();
    code(Control::new(&*c).map(|x| *out = Box::into_raw(Box::new(x))))
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_free(c: *mut Control) {
    if !c.is_null() {
        drop(Box::from_raw(c));
    }
}
/// The head's control thread: returns once the service stops, or when something failed.
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_run(c: *const Control) -> i32 {
    code((*c).run())
}
/// Every rank's watcher (abort.s2 and quit).
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_monitor(c: *const Control, linger_ms: u32) -> i32 {
    code((*c).monitor(linger_ms))
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_publish(
    c: *const Control,
    boundary: u64,
    revision: u64,
    horizon: u64,
    resident: u32,
    retiring: u32,
) {
    (*c).publish(boundary, revision, horizon, resident, retiring);
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_resized(c: *const Control) {
    (*c).resized();
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_resident(c: *const Control) -> u32 {
    (*c).resident()
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_stopping(c: *const Control) -> bool {
    (*c).stopping()
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_stop(c: *const Control) {
    (*c).stop();
}
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_shutdown(c: *const Control) {
    (*c).shutdown();
}
/// status.s2 from the pool's values in s (free, room, and the growth, tiers and runs with their has_ flags); the
/// service's own fields in s are not read.
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_write_status(c: *const Control, s: *const CStatus) -> i32 {
    let s = &*s;
    let growth = s.has_growth.then_some(Growth {
        held_pages: s.held_pages,
        worst_pages: s.worst_pages,
        grows: s.grows,
        stalls: s.stalled,
    });
    let tiers = s.has_tiers.then(|| slice(s.tiers, s.n_tiers));
    let runs = s.has_runs.then(|| slice(s.runs, s.n_runs));
    code((*c).write_status(s.free, s.room, growth, tiers, runs))
}
/// A test's observer of the control thread's plain pool calls; set before the control thread starts.
#[no_mangle]
pub unsafe extern "C" fn g6svc_control_observe(c: *const Control, f: ObserveFn, ctx: *mut c_void) {
    (*c).observe(f, ctx);
}
