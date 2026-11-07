//! The head's control thread (was service.h's controller, event, intake, drain_admits and monitor). It reads the
//! shim's commands, keeps the admission FIFO and the queued requests, announces every event and builds its commit, and
//! publishes status.s2. Every rank's watcher runs `monitor`. No thread is started here: the C++ Service's threads call
//! in, and a worker's control thread stays in the channel's receive_events.
//!
//! What both threads touch is atomics here: the scheduler's progress, which the compute thread (compute.rs) publishes
//! after each commit and step, whether a resize is in flight, and whether the service is stopping. Everything else is
//! the control thread's, apart from the plain pool and policy calls a worker's bind makes through it. What stays C++
//! is reached through `Host`: the request checks (the parser's ServeReq, the grammar, the images), a cancelled
//! session's retire_run (it reads Dispatch and the driver facade), the KV share report, and the pool policy's two calls
//! that return C++ objects (refusal, tiers).
//!
//! The commit's pool calls have an order every worker replays at its grant: the cancel's retire, then `r_rows`,
//! `match`, `fits`, an optional `make_room` and a second fit, the queue's check, and in a grant `count`, `match`,
//! then `take` or `reserve`.

use super::admit;
use super::files::{atomic_file, exists, Command, Ingress};
use super::status::{self, Growth, RunUse, Status};
use super::trace::{Trace, CONTROL};
use crate::admission::{Queue, Status as Fit, STALL_NONE, STALL_SLOTS};
use crate::control::channel::{wall_us, Endpoint};
use crate::control::wire::{self, Commit, Notice};
use crate::control::{refused, require, Fail, R};
use crate::flash_lend::Lend;
use crate::kv_share::{Resize, Totals};
use crate::pool_ops::{PoolHit, PoolOps};
use crate::scheduler::{
    self as sch, Config, Grant, Identity, Key, Request, Run, Shrink, Slot, State, FREE, NOTICE_MAX, PAGE, RETIRING,
};
use g6kvman::ffi::Handle;
use g6kvman::pool::{Namespace, NAMESPACE_BYTES};
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Resize events on this tier lend and recall the Flash rank's KV chunks (scheduler.h FLASH_KV_TIER).
pub const FLASH_KV_TIER: u32 = u32::MAX;

/// g6svc_checked: a request file the host parsed and checked. ok false: refused and answered (resp written), nothing
/// queued. prompt: the parsed prompt, valid until the next host call.
#[repr(C)]
pub struct Checked {
    pub prompt: *const i32,
    pub n: usize,
    pub sys: i64,
    pub plp: i32,
    pub n_gen: i32,
    pub steering: Identity,
    pub ok: bool,
}

/// g6svc_pool_call: one plain pool call of the service (the head's control thread, or a worker's bind), for a test's
/// observer. out: room's or count's result, or fits' as 0 or 1; run: reserve's.
#[repr(C)]
pub struct PoolCall {
    pub kind: u32,
    pub rows: u64,
    pub worst: u64,
    pub r: u32,
    pub out: u32,
    pub run: Run,
    pub prompt: *const i32,
    pub n: usize,
    pub span: i64,
    pub ns: *const u8,
}
pub const CALL_ROOM: u32 = 0;
pub const CALL_FITS: u32 = 1;
pub const CALL_RESERVE: u32 = 2;
pub const CALL_COUNT: u32 = 3;

/// check(id, bytes, n, *checked)
pub type CheckFn = unsafe extern "C" fn(*mut c_void, u64, *const u8, usize, *mut Checked) -> i32;
/// retire_run(slot)
pub type RetireRunFn = crate::driver::SlotFn;
/// share(idle, *has, *resize)
pub type ShareFn = unsafe extern "C" fn(*mut c_void, bool, *mut bool, *mut Resize) -> i32;
/// refusal(worst, r, *bytes, *n): the pool policy's reason, valid until the next host call
pub type ReasonFn = unsafe extern "C" fn(*mut c_void, u64, u32, *mut *const u8, *mut usize) -> i32;
/// tiers(*totals, *n): the pool policy's tier counters, valid until the next host call
pub type TotalsFn = unsafe extern "C" fn(*mut c_void, *mut *const Totals, *mut usize) -> i32;
/// a test's observer of the plain pool calls
pub type ObserveFn = unsafe extern "C" fn(*mut c_void, *const PoolCall);
pub(super) type Observer = Option<(ObserveFn, *mut c_void)>;

/// g6svc_staged: the request the compute thread staged for an ADMIT (req: the parser's ServeReq, kept by the host;
/// void for the layout digest), and what the worker's bind reads of it. Valid until the host's next stage or unstage.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Staged {
    pub req: *const c_void,
    pub prompt: *const i32,
    pub n: usize,
    pub sys: i64,
    pub plp: i32,
}

/// idle_boundary(boundary)
pub type BoundaryFn = unsafe extern "C" fn(*mut c_void, u64) -> i32;
/// stage(bytes, n, *staged)
pub type StageFn = unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut Staged) -> i32;
/// digest(has_pool, pool, *out)
pub type DigestFn = unsafe extern "C" fn(*mut c_void, bool, u64, *mut u64) -> i32;
/// apply(commit, admit)
pub type ApplyFn = unsafe extern "C" fn(*mut c_void, *const Commit, bool) -> i32;
/// step(*quantum)
pub type StepFn = unsafe extern "C" fn(*mut c_void, *mut crate::scheduler::Quantum) -> i32;
/// counters(keys, n)
pub type CountersFn = unsafe extern "C" fn(*mut c_void, *const Key, u32) -> i32;

/// g6svc_host: the C++ the service's threads call. Each call returns 0, or 1 when the C++ threw (the C++ caller
/// rethrows it). The control thread calls check, retire_run, share, refusal and tiers (share: null without a KV
/// share); the compute thread retire_run and the rest, each thread with its own ctx. The compute thread's are the
/// driver facade (idle_boundary, digest with the pool's, apply, step), the staged request (stage, unstage) and Sessions
/// and Dispatch: abort (a cancel's terminal lines), mode (Dispatch::synchronize_mode), counters (the "m" lines due),
/// begin (Dispatch::begin), bound (Sessions::bind) and release (Sessions::retire, Dispatch::release).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Host {
    pub ctx: *mut c_void,
    pub check: Option<CheckFn>,
    pub retire_run: Option<RetireRunFn>,
    pub share: Option<ShareFn>,
    pub refusal: Option<ReasonFn>,
    pub tiers: Option<TotalsFn>,
    pub idle_boundary: Option<BoundaryFn>,
    pub stage: Option<StageFn>,
    pub unstage: Option<crate::driver::VoidFn>,
    pub digest: Option<DigestFn>,
    pub abort: Option<crate::driver::KeyFn>,
    pub apply: Option<ApplyFn>,
    pub mode: Option<crate::driver::VoidFn>,
    pub step: Option<StepFn>,
    pub counters: Option<CountersFn>,
    pub begin: Option<crate::driver::BindFn>,
    pub bound: Option<crate::driver::BindFn>,
    pub release: Option<crate::driver::SlotFn>,
}

/// g6svc_control_cfg: the control thread's settings, and what it borrows from the C++ Service for its life: the
/// channel endpoint, the ingress, the admission queue, the trace (null: none), the driver's slots and counters and the
/// rank's pool.
#[repr(C)]
pub struct CControl {
    pub scheduler: *const Config,
    pub directory: *const u8,
    pub directory_n: usize,
    pub endpoint: *const c_void,
    pub ingress: *const c_void,
    pub queue: *mut c_void,
    pub trace: *const c_void,
    pub slots: *const Slot,
    pub n_slots: usize,
    pub state: *const State,
    pub pool: PoolOps,
    pub host: Host,
    pub lend: Lend,
    pub share_ms: i64,
    pub rank: u32,
    pub head: bool,
    pub admit_hold: bool,
    pub elog: bool,
}

/// steering.h cache_namespace: the prefix cache's namespace of a request's identity.
pub fn cache_namespace(s: &Identity) -> Namespace {
    let mut out = [0u8; NAMESPACE_BYTES];
    if s.images {
        out[0] = 2;
        out[1..33].copy_from_slice(&s.image_digest);
        return out;
    }
    if !s.enabled {
        return out;
    }
    out[0] = 1;
    out[1..33].copy_from_slice(&s.profile_digest);
    out[33] = s.scope;
    out[34..66].copy_from_slice(&s.schedule_digest);
    out[66..70].copy_from_slice(&s.scale_bits.to_le_bytes());
    out
}

/// Sessions::request: what the scheduler and the FIFO keep of a request.
pub fn request(id: u64, prompt_tokens: usize, n_gen: i32, steering: Identity) -> Request {
    Request { id, prompt_tokens: prompt_tokens as u32, max_tokens: n_gen as u32, steering, r_rows: 0 }
}

// an event to announce: a shim command, or a Resize the head makes itself
#[derive(Clone, Copy)]
struct Cmd {
    c: Command,
    resize: Option<Resize>,
}

fn resize_cmd(epoch: u64, r: Resize) -> Cmd {
    Cmd { c: Command { epoch, ..Command::default() }, resize: Some(r) }
}

pub(super) fn hook(rc: i32) -> R<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Fail::Hook)
    }
}

fn no_hit() -> PoolHit {
    PoolHit { skip: 0, lcp: 0, why: c"none".as_ptr(), age: 0, stored: false, tier: 0, slot: -1 }
}

fn why(h: &PoolHit) -> String {
    // SAFETY: the pool policy's why is a static C string
    unsafe { CStr::from_ptr(h.why) }.to_string_lossy().into_owned()
}

fn run_of(r: &g6kvman::Run) -> Run {
    Run { first_page: r.first_page, pages: r.pages, table_version: r.table_version, list_hash: r.list_hash }
}

// "key":value pairs for the trace's args
pub(super) fn targ(k: &str, v: u64) -> String {
    format!("\"{k}\":{v}")
}
pub(super) fn tstr(k: &str, v: &str) -> String {
    format!("\"{k}\":\"{v}\"")
}

// one write per line, as fprintf to the unbuffered stderr makes it, so lines from other threads do not cut into it
pub(super) fn eline(s: String) {
    let _ = std::io::stderr().lock().write_all(s.as_bytes());
}

struct Inner {
    commands: BTreeMap<u64, Command>,
    // queued requests' bytes, sent to the workers with each ADMIT notice
    payloads: BTreeMap<u64, Arc<Vec<u8>>>,
    // prefix reuse: queued requests' prompts (none for prompt logprobs) and system spans
    prompts: BTreeMap<u64, Vec<i32>>,
    spans: BTreeMap<u64, i64>,
    attempted_head: u64,
    attempted_revision: u64,
    event: u64,
    waited: u64,  // the request whose wait the trace has
    refused: u64, // the last request the refusal line named
    // borrow-idle: the chunks the FIFO head waits for (0 = none), when its first recall went out, the request the
    // wait line named
    recall: u32,
    recall_t0: Instant,
    recall_for: u64,
    lend: Lend,
}

/// The head's control thread and the state the compute thread shares with it (every rank has one: the watcher and
/// the compute thread's stop flag are here).
pub struct Control {
    c: Config,
    dir: Vec<u8>,
    rank: u32,
    head: bool,
    endpoint: *const Endpoint,
    ingress: *const Ingress,
    queue: *mut Queue,
    trace: *const Trace,
    slots: *const Slot,
    n_slots: usize,
    state: *const State,
    pool: *const Handle,
    ops: PoolOps,
    host: Host,
    admit_hold: bool,
    elog: bool,
    share_ms: u64,
    start: Instant,
    // the compute thread's: published after each commit and step. horizon is stored before boundary and read after it
    boundary: AtomicU64,
    revision: AtomicU64,
    horizon: AtomicU64,
    resident: AtomicU32,
    retiring: AtomicU32,
    // a Resize announced and not yet applied on this rank (no second report until then: it would ask again for
    // segments already granted)
    resizing: AtomicBool,
    stopping: AtomicBool,
    inner: Mutex<Inner>,
    // set before the threads start; the compute thread's plain pool calls (a worker's bind) report here too
    observe: Mutex<Observer>,
}

impl Control {
    /// # Safety
    /// Every pointer in `c` must stay valid for the Control's life, the endpoint, ingress, queue, trace and pool
    /// handles must be the Rust objects the C++ facades hold, and the slots and state the driver's.
    pub unsafe fn new(c: &CControl) -> R<Control> {
        require(
            !c.scheduler.is_null() && !c.endpoint.is_null() && !c.ingress.is_null() && !c.queue.is_null(),
            "control handles",
        )?;
        require(!c.pool.pool.is_null() && !c.slots.is_null() && !c.state.is_null(), "control handles")?;
        require(
            !c.head || (c.host.check.is_some() && c.host.retire_run.is_some()),
            "the head's control needs check and retire_run",
        )?;
        require(c.pool.refusal.is_none() || c.host.refusal.is_some(), "a pool refusal without the host's")?;
        require(c.pool.tiers.is_none() || c.host.tiers.is_some(), "pool tiers without the host's")?;
        let dir = if c.directory_n == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(c.directory, c.directory_n).to_vec()
        };
        let now = Instant::now();
        Ok(Control {
            c: *c.scheduler,
            dir,
            rank: c.rank,
            head: c.head,
            endpoint: c.endpoint.cast(),
            ingress: c.ingress.cast(),
            queue: c.queue.cast(),
            trace: c.trace.cast(),
            slots: c.slots,
            n_slots: c.n_slots,
            state: c.state,
            pool: c.pool.pool.cast(),
            ops: c.pool,
            host: c.host,
            admit_hold: c.admit_hold,
            elog: c.elog,
            share_ms: c.share_ms.max(1) as u64,
            start: now,
            boundary: AtomicU64::new(0),
            revision: AtomicU64::new(0),
            horizon: AtomicU64::new(0),
            resident: AtomicU32::new(0),
            retiring: AtomicU32::new(0),
            resizing: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                commands: BTreeMap::new(),
                payloads: BTreeMap::new(),
                prompts: BTreeMap::new(),
                spans: BTreeMap::new(),
                attempted_head: u64::MAX,
                attempted_revision: u64::MAX,
                event: 1,
                waited: 0,
                refused: u64::MAX,
                recall: 0,
                recall_t0: now,
                recall_for: u64::MAX,
                lend: c.lend,
            }),
            observe: Mutex::new(None),
        })
    }

    // --- what both threads touch

    /// The compute thread after each commit and step.
    pub fn publish(&self, boundary: u64, revision: u64, horizon: u64, resident: u32, retiring: u32) {
        // horizon before boundary: a reader that loads boundary first and horizon second (event) sees a horizon
        // computed for that boundary or a later one, never an earlier one. Later only widens the margin.
        self.horizon.store(horizon, SeqCst);
        self.boundary.store(boundary, SeqCst);
        self.resident.store(resident, SeqCst);
        self.retiring.store(retiring, SeqCst);
        self.revision.store(revision, SeqCst);
    }
    /// The compute thread applied a Resize commit.
    pub fn resized(&self) {
        self.resizing.store(false, SeqCst);
    }
    pub fn resident(&self) -> u32 {
        self.resident.load(SeqCst)
    }
    pub fn stopping(&self) -> bool {
        self.stopping.load(SeqCst)
    }
    pub fn stop(&self) {
        self.stopping.store(true, SeqCst);
    }
    /// Stops and fails the epoch's channel, which wakes every thread waiting in it.
    pub fn shutdown(&self) {
        self.stop();
        self.endpoint().fail();
    }
    /// A test's observer of the plain pool calls (the head's control thread and a worker's binds); set before the threads
    /// start.
    pub fn observe(&self, f: ObserveFn, ctx: *mut c_void) {
        *self.observe.lock().unwrap_or_else(|e| e.into_inner()) = Some((f, ctx));
    }
    pub(super) fn observer(&self) -> Observer {
        *self.observe.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(super) fn config(&self) -> &Config {
        &self.c
    }

    // --- borrowed objects

    fn endpoint(&self) -> &Endpoint {
        // SAFETY: the C++ Service's channel outlives this Control (new's contract); its calls are safe from any thread
        unsafe { &*self.endpoint }
    }
    fn ingress(&self) -> &Ingress {
        // SAFETY: as endpoint; the compute thread may read a request file at the same time, which Ingress allows
        unsafe { &*self.ingress }
    }
    fn trace(&self) -> Option<&Trace> {
        // SAFETY: as endpoint; the trace locks each write
        unsafe { self.trace.as_ref() }
    }
    fn pool(&self) -> &Handle {
        // SAFETY: as endpoint; the handle locks the pool for each call
        unsafe { &*self.pool }
    }
    // The queue is the control thread's alone (the C++ Service no longer calls it; tests touch it only while no control
    // thread runs). Each call takes the reference for that call only.
    fn queue_len(&self) -> usize {
        // SAFETY: see above
        unsafe { (*self.queue).len() }
    }
    fn queue_stalls(&self) -> u64 {
        // SAFETY: see above
        unsafe { (*self.queue).stalls }
    }
    fn queue_front(&self) -> Option<Request> {
        // SAFETY: see above; copied, so nothing points into the queue once it changes
        unsafe { (*self.queue).front().copied() }
    }
    fn queue_push(&self, r: Request) -> R<()> {
        // SAFETY: see above
        unsafe { (*self.queue).push(r) }.map_err(refused)
    }
    fn queue_cancel(&self, id: u64) -> bool {
        // SAFETY: see above
        unsafe { (*self.queue).cancel(id) }
    }
    fn queue_admitted(&self, id: u64) -> R<()> {
        // SAFETY: see above
        unsafe { (*self.queue).admitted(id) }.map_err(refused)
    }
    fn queue_check(&self, revision: u64, slot: bool, fits: bool, room: u32) -> R<Option<Fit>> {
        // SAFETY: see above
        unsafe { (*self.queue).check(revision, slot, fits, room) }.map_err(refused)
    }
    /// The driver's slots and counters.
    ///
    /// # Safety
    /// Only inside a commit: the compute thread waits in the event boundary until the commit is applied, so the
    /// driver does not change them while they are read, and nothing on this thread changes them.
    unsafe fn view(&self) -> (&[Slot], &State) {
        (std::slice::from_raw_parts(self.slots, self.n_slots), &*self.state)
    }

    pub(super) fn reuse(&self) -> bool {
        self.ops.r#match.is_some()
    }
    pub(super) fn counts(&self) -> bool {
        self.ops.count
    }
    fn path(&self, name: &str) -> Vec<u8> {
        let mut p = self.dir.clone();
        p.push(b'/');
        p.extend_from_slice(name.as_bytes());
        p
    }
    fn now_ns(&self) -> i64 {
        self.start.elapsed().as_nanos() as i64
    }

    // --- the plain pool calls, each run held to the module's max_run_pages

    fn seen(obs: Observer, call: PoolCall) {
        if let Some((f, ctx)) = obs {
            // SAFETY: the observer and its context outlive the run (observe's contract)
            unsafe { f(ctx, &call) }
        }
    }
    fn call(kind: u32) -> PoolCall {
        PoolCall {
            kind,
            rows: 0,
            worst: 0,
            r: 0,
            out: 0,
            run: Run::default(),
            prompt: std::ptr::null(),
            n: 0,
            span: 0,
            ns: std::ptr::null(),
        }
    }
    fn within(&self, worst: u64) -> bool {
        self.ops.max_run_pages == 0 || worst <= u64::from(self.ops.max_run_pages) * u64::from(PAGE)
    }
    fn room(&self, obs: Observer) -> u32 {
        let n = self.pool().lock().room();
        Self::seen(obs, PoolCall { out: n, ..Self::call(CALL_ROOM) });
        if self.ops.max_run_pages != 0 {
            n.min(self.ops.max_run_pages)
        } else {
            n
        }
    }
    fn fits(&self, rows: u64, worst: u64, r: u32, obs: Observer) -> bool {
        if !self.within(worst) {
            return false;
        }
        let f = self.pool().lock().fits_r(rows, worst, r);
        Self::seen(obs, PoolCall { rows, worst, r, out: u32::from(f), ..Self::call(CALL_FITS) });
        f
    }
    // the refusal's message on failure: the pool said it fits and then refused, one request's bug
    pub(super) fn reserve(&self, rows: u64, worst: u64, r: u32, obs: Observer) -> Result<Run, String> {
        if !self.within(worst) {
            return Err("a run larger than the module's max_run_pages (DS41: one session's slab)".into());
        }
        let x = self.pool().lock().reserve_r(rows, worst, r).map_err(|e| e.message().to_string_lossy().into_owned())?;
        let run = run_of(&x);
        Self::seen(obs, PoolCall { rows, worst, r, run, ..Self::call(CALL_RESERVE) });
        Ok(run)
    }
    pub(super) fn count(&self, prompt: &[i32], span: i64, ns: &Namespace, obs: Observer) -> u32 {
        let n = self.pool().lock().count_namespaced(prompt, u32::try_from(span).ok(), *ns);
        let c = PoolCall {
            prompt: prompt.as_ptr(),
            n: prompt.len(),
            span,
            ns: ns.as_ptr(),
            out: n,
            ..Self::call(CALL_COUNT)
        };
        Self::seen(obs, c);
        n
    }

    // --- the pool policy (PoolOps): on this thread only inside a commit, on the compute thread at its own bind and
    // event boundary

    fn r_rows(&self, q: &Request, rows: u64, worst: u64, prompt: Option<&[i32]>) -> R<u32> {
        let f = self.ops.r_rows.ok_or_else(|| refused("bad_function_call"))?;
        let (p, n) = prompt.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len()));
        let mut out = 0u32;
        // SAFETY: the module's table and context outlive the service
        hook(unsafe { f(self.ops.ctx, q, rows, worst, p, n, prompt.is_some(), &mut out) })?;
        Ok(out)
    }
    pub(super) fn pool_match(
        &self,
        prompt: &[i32],
        rows: u64,
        worst: u64,
        limit: i32,
        ns: &Namespace,
        r: u32,
    ) -> R<PoolHit> {
        let f = self.ops.r#match.ok_or_else(|| refused("bad_function_call"))?;
        let mut h = no_hit();
        // SAFETY: as r_rows
        hook(unsafe { f(self.ops.ctx, prompt.as_ptr(), prompt.len(), rows, worst, limit, ns.as_ptr(), r, &mut h) })?;
        Ok(h)
    }
    pub(super) fn take(&self, h: &PoolHit, rows: u64, worst: u64, r: u32) -> R<Run> {
        let f = self.ops.take.ok_or_else(|| refused("bad_function_call"))?;
        let mut run = Run::default();
        // SAFETY: as r_rows
        hook(unsafe { f(self.ops.ctx, h, rows, worst, r, &mut run) })?;
        Ok(run)
    }
    /// The pool's digest, when the policy has one.
    pub(super) fn digest(&self) -> R<Option<u64>> {
        let Some(f) = self.ops.digest else { return Ok(None) };
        let mut d = 0u64;
        // SAFETY: as r_rows
        hook(unsafe { f(self.ops.ctx, &mut d) })?;
        Ok(Some(d))
    }
    fn make_room(&self, slots: &[Slot], q: &Request, rows: u64, worst: u64) -> R<Option<Shrink>> {
        let f = self.ops.make_room.ok_or_else(|| refused("bad_function_call"))?;
        let (mut made, mut k) = (false, Shrink { key: Key::default(), r_rows: 0, run: Run::default() });
        // SAFETY: as r_rows; slots: the driver's, every slot of them
        hook(unsafe { f(self.ops.ctx, slots.as_ptr(), q, rows, worst, &mut made, &mut k) })?;
        Ok(made.then_some(k))
    }

    // --- the host

    // the request's prompt, sys, plp, n_gen and identity, or none when the checks refused it
    fn check(&self, id: u64, bytes: &[u8]) -> R<Option<Checked>> {
        let f = self.host.check.ok_or_else(|| refused("bad_function_call"))?;
        let mut ck = Checked {
            prompt: std::ptr::null(),
            n: 0,
            sys: 0,
            plp: 0,
            n_gen: 0,
            steering: wire::identity_off(),
            ok: false,
        };
        // SAFETY: the host's context outlives the Control
        hook(unsafe { f(self.host.ctx, id, bytes.as_ptr(), bytes.len(), &mut ck) })?;
        Ok(ck.ok.then_some(ck))
    }
    fn retire_run(&self, s: &Slot) -> R<()> {
        let f = self.host.retire_run.ok_or_else(|| refused("bad_function_call"))?;
        // SAFETY: as check
        hook(unsafe { f(self.host.ctx, s) })
    }
    fn share(&self, idle: bool) -> R<Option<Resize>> {
        let Some(f) = self.host.share else { return Ok(None) };
        let (mut has, mut r) = (false, Resize::default());
        // SAFETY: as check
        hook(unsafe { f(self.host.ctx, idle, &mut has, &mut r) })?;
        Ok(has.then_some(r))
    }
    fn refusal(&self, worst: u64, r: u32) -> R<Vec<u8>> {
        let f = self.host.refusal.ok_or_else(|| refused("bad_function_call"))?;
        let (mut p, mut n) = (std::ptr::null(), 0usize);
        // SAFETY: as check; the bytes stay valid until the next host call
        hook(unsafe { f(self.host.ctx, worst, r, &mut p, &mut n) })?;
        Ok(if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, n) }.to_vec() })
    }
    fn tiers(&self) -> R<Vec<Totals>> {
        let f = self.host.tiers.ok_or_else(|| refused("bad_function_call"))?;
        let (mut p, mut n) = (std::ptr::null(), 0usize);
        // SAFETY: as refusal
        hook(unsafe { f(self.host.ctx, &mut p, &mut n) })?;
        Ok(if n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, n) }.to_vec() })
    }

    // --- status.s2

    fn publish_status(&self) -> R<()> {
        let free = self.pool().lock().free_pages();
        let room = self.room(None);
        let growth = self.ops.status.then(|| {
            let g = self.pool().lock().stats();
            Growth { held_pages: g.held_pages, worst_pages: g.worst_pages, grows: g.grows, stalls: g.stalls }
        });
        let tiers = if self.ops.tiers.is_some() { Some(self.tiers()?) } else { None };
        let runs: Option<Vec<RunUse>> = self.ops.status.then(|| {
            let p = self.pool().lock();
            p.runs().map(|(table_version, held, worst)| RunUse { table_version, held, worst }).collect()
        });
        self.write_status(free, room, growth, tiers.as_deref(), runs.as_deref())
    }
    /// status.s2 from the pool's values and this service's queue and progress.
    pub fn write_status(
        &self,
        free: u32,
        room: u32,
        growth: Option<Growth>,
        tiers: Option<&[Totals]>,
        runs: Option<&[RunUse]>,
    ) -> R<()> {
        let pool_pages = self.c.pool_rows / PAGE;
        status::pool_ok(free, room, pool_pages)?;
        let st = Status {
            epoch: self.c.epoch,
            pool_pages,
            resident: self.resident.load(SeqCst),
            queued: self.queue_len() as u64,
            retiring: self.retiring.load(SeqCst),
            free,
            room,
            stalls: self.queue_stalls(),
            growth,
            tiers,
            runs,
        };
        status::publish(&self.dir, &st)
    }

    // --- the control loop

    /// The head's control thread, until the service stops (Ok) or something fails.
    pub fn run(&self) -> R<()> {
        let mut inn = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.controller(&mut inn)
    }

    fn controller(&self, inn: &mut Inner) -> R<()> {
        let mut status_revision = u64::MAX;
        let mut next_share = Instant::now();
        while !self.stopping() {
            let mut changed = false;
            if self.host.share.is_some() && !self.resizing.load(SeqCst) && Instant::now() >= next_share {
                next_share = Instant::now() + Duration::from_millis(self.share_ms);
                let resident = self.resident.load(SeqCst);
                if let Some(r) = self.share(admit::share_idle(resident, self.queue_len()))? {
                    eline(format!("# s2 share: tier {} to {} segments ({} blocks)\n", r.tier, r.segs, r.slots));
                    self.resizing.store(true, SeqCst);
                    self.event(inn, &resize_cmd(self.c.epoch, r), None)?;
                    changed = true;
                }
            }
            // borrow-idle: lend the chunks above flash.keep once the engine has been idle flash.idle_ms
            if inn.lend.chunk_rows != 0 && !self.resizing.load(SeqCst) {
                let resident = self.resident.load(SeqCst);
                let queued = self.queue_len();
                let retiring = self.retiring.load(SeqCst);
                if let Some(n) = inn.lend.poll(admit::lend_idle(resident, queued, retiring), self.now_ns()) {
                    eline(format!("# s2 flash kv: idle, lending down to {n} chunks\n"));
                    self.resizing.store(true, SeqCst);
                    let r = Resize { tier: FLASH_KV_TIER, segs: u64::from(n), slots: 0, txn: 0 };
                    self.event(inn, &resize_cmd(self.c.epoch, r), None)?;
                    changed = true;
                }
            }
            while let Some(command) = self.ingress().next()? {
                changed = true;
                let id = command.request;
                if command.admit {
                    self.intake(inn, &command)?;
                } else if self.queue_cancel(id) {
                    inn.commands.remove(&id);
                    inn.prompts.remove(&id);
                    inn.spans.remove(&id);
                    inn.payloads.remove(&id);
                    atomic_file(&self.path(&format!("resp.{id}")), b"end abort -1 0\n")?;
                } else {
                    // a resident or retiring session: found only at the commit
                    self.event(inn, &Cmd { c: command, resize: None }, None)?;
                }
                if self.stopping() {
                    return Ok(());
                }
            }
            if let Some(front) = self.queue_front() {
                if admit::retry(front.id, inn.attempted_head, self.revision.load(SeqCst), inn.attempted_revision) {
                    let command = *inn.commands.get(&front.id).ok_or_else(|| refused("map::at"))?;
                    self.event(inn, &Cmd { c: command, resize: None }, None)?;
                    changed = true;
                }
            }
            let rev = self.revision.load(SeqCst);
            if changed || rev != status_revision {
                self.publish_status()?;
                status_revision = rev;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    // an ADMIT command into the FIFO, its request file read and checked; a refused request is answered and not queued
    fn intake(&self, inn: &mut Inner, command: &Command) -> R<()> {
        let bytes = self.ingress().request(command)?;
        let Some(ck) = self.check(command.request, &bytes)? else { return Ok(()) };
        let queued = request(command.request, ck.n, ck.n_gen, ck.steering);
        if let Some(t) = self.trace() {
            let args = format!(
                "{},{},{},{}",
                targ("request", queued.id),
                targ("prompt_tokens", u64::from(queued.prompt_tokens)),
                targ("max_tokens", u64::from(queued.max_tokens)),
                targ("queued", self.queue_len() as u64)
            );
            t.instant(b"arrive", CONTROL, args.as_bytes());
        }
        self.queue_push(queued)?;
        inn.commands.entry(command.request).or_insert(*command);
        inn.payloads.entry(command.request).or_insert_with(|| Arc::new(bytes));
        if self.reuse() && ck.plp < 0 {
            inn.spans.entry(command.request).or_insert(ck.sys);
            // SAFETY: the host's prompt stays valid until the next host call
            let prompt =
                if ck.n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(ck.prompt, ck.n) }.to_vec() };
            inn.prompts.entry(command.request).or_insert(prompt);
        }
        Ok(())
    }

    // inside an admission's commit: queue the ADMITs already written, in cmd order; stop at the first other command (a
    // CANCEL may need its own event, which only the loop may start), so the order of commands is unchanged
    fn drain_admits(&self, inn: &mut Inner) -> R<()> {
        while let Some(command) = self.ingress().peek()? {
            if !command.admit {
                return Ok(());
            }
            self.ingress().next()?;
            self.intake(inn, &command)?;
        }
        Ok(())
    }

    // held: the boundary every rank is stopped at after a commit with hold; the notice names it, not a future one.
    // Only event itself passes it, at once after the held commit, so nothing else is announced between.
    fn event(&self, inn: &mut Inner, cmd: &Cmd, held: Option<u64>) -> R<()> {
        // boundary first: the horizon below is for this boundary or a later one
        let boundary = match held {
            Some(b) => b,
            None => self.boundary.load(SeqCst),
        };
        let horizon = self.horizon.load(SeqCst);
        require(boundary <= i64::MAX as u64 - NOTICE_MAX, "event boundary overflow")?;
        let target = admit::notice_target(boundary, horizon, held, self.c.notice_max);
        let (kind, wkind) = if cmd.resize.is_some() {
            ("resize", wire::RESIZE)
        } else if cmd.c.admit {
            ("admit", wire::ADMIT)
        } else {
            ("cancel", wire::CANCEL)
        };
        if let Some(t) = self.trace() {
            let args = format!(
                "{},{},{},{}",
                targ("request", cmd.c.request),
                tstr("kind", kind),
                targ("anchored", boundary),
                targ("boundary", target)
            );
            t.instant(b"notice", CONTROL, args.as_bytes());
        }
        if self.elog {
            eline(format!(
                "# s2_event notice rank {} event {} {} request {} anchored {} -> boundary {} (lookahead {}) t_us {}\n",
                self.rank,
                inn.event,
                kind,
                cmd.c.request,
                boundary,
                target,
                target.wrapping_sub(boundary),
                wall_us()
            ));
        }
        let n = Notice {
            epoch: self.c.epoch,
            event: inn.event,
            boundary: target,
            request: cmd.c.request,
            payload_hash: cmd.c.hash,
            kind: wkind,
        };
        // the channel sends these bytes before make_commit, and the commit erases the request's entry: this reference
        // keeps them alive for the whole event
        let payload = if cmd.c.admit {
            Some(inn.payloads.get(&cmd.c.request).cloned().ok_or_else(|| refused("map::at"))?)
        } else {
            None
        };
        let mut hold = false;
        self.endpoint().event(
            &n,
            |hash| self.commit(inn, &n, cmd, hash, &mut hold),
            payload.as_deref().map(Vec::as_slice),
        )?;
        require(inn.event != u64::MAX, "event sequence exhausted")?;
        inn.event += 1;
        self.publish_status()?;
        // Every rank is stopped at n.boundary and waits for a notice that names it: announce the new FIFO head there
        // now. Its commit holds again or releases (no grant, or nothing left to chain), so the chain ends within
        // limit() admissions.
        if hold {
            let front = self.queue_front().ok_or_else(|| refused("a held commit with nothing queued"))?;
            let next = *inn.commands.get(&front.id).ok_or_else(|| refused("map::at"))?;
            self.event(inn, &Cmd { c: next, resize: None }, Some(n.boundary))?;
        }
        // the FIFO head waits for lent chunks: recall them (every rank grows at the commit), then it is tried again
        if inn.recall != 0 && !self.resizing.load(SeqCst) {
            let r = Resize { tier: FLASH_KV_TIER, segs: u64::from(inn.recall), slots: 0, txn: 0 };
            inn.recall = 0;
            self.resizing.store(true, SeqCst);
            self.event(inn, &resize_cmd(self.c.epoch, r), None)?;
        }
        Ok(())
    }

    // The commit, after every rank's ACK, while every compute thread waits at the event boundary.
    fn commit(&self, inn: &mut Inner, n: &Notice, cmd: &Cmd, hash: u64, hold: &mut bool) -> R<Commit> {
        // SAFETY: inside the commit (view's contract): compute is paused at the boundary until this commit is applied
        let (s, st) = unsafe { self.view() };
        let mut cm = Commit::zero();
        cm.c.epoch = n.epoch;
        cm.c.event = n.event;
        cm.c.boundary = n.boundary;
        cm.c.before_hash = hash;
        // every rank applies a resize at the commit (the driver's resize hook), the head too
        if let Some(r) = cmd.resize {
            cm.has_resize = true;
            cm.resize = r;
        }
        if !cmd.c.admit && cmd.resize.is_none() {
            if let Some(k) = sch::find(s, cmd.c.request) {
                cm.c.has_cancel = true;
                cm.c.cancel = k;
            }
        }
        // The head makes the pool's change once, here; the driver skips it. A Retiring slot gave its run back when it
        // finished (the finished hook); a cancelled live one gives it back now.
        for sl in s {
            if cm.c.has_cancel && sl.phase != FREE && sl.phase != RETIRING && sl.key == cm.c.cancel {
                self.retire_run(sl)?;
            }
        }
        if cmd.c.admit {
            self.admission(inn, s, st, cmd, &mut cm, hold)?;
        }
        Ok(cm)
    }

    // A grant's rows at admission: grant_rows when runs grow on demand, else its worst case.
    pub(super) fn admit_rows(&self, r: &Request) -> R<u64> {
        if self.ops.grow { sch::grant_rows(&self.c, r) } else { sch::reserve_rows(&self.c, r) }.map_err(refused)
    }

    // r_rows, match and fits: the R the run takes, the hit on a retained run, and whether it fits
    fn try_fit(
        &self,
        req: &mut Request,
        rows: u64,
        worst: u64,
        prompt: Option<&[i32]>,
        ns: &Namespace,
        obs: Observer,
    ) -> R<(u32, PoolHit, bool)> {
        if self.ops.r_rows.is_some() {
            req.r_rows = self.r_rows(req, rows, worst, prompt)?;
        }
        let r = req.r_rows;
        // a prompt that continues a retained run fits when the hit does (match checks the room the hit needs, with the
        // run's own pages and records counted as its own), even when a fresh run of it would not
        let pre = match prompt {
            Some(p) => self.pool_match(p, rows, worst, req.prompt_tokens as i32 - 1, ns, r)?,
            None => no_hit(),
        };
        let fits = pre.skip > 0 || self.fits(rows, worst, r, obs);
        Ok((r, pre, fits))
    }

    fn admission(&self, inn: &mut Inner, s: &[Slot], st: &State, cmd: &Cmd, cm: &mut Commit, hold: &mut bool) -> R<()> {
        // copied at once: the queue's front is valid only until the queue changes (drain_admits pushes)
        let mut req = match self.queue_front() {
            Some(f) if f.id == cmd.c.request => f,
            _ => return Err(refused("admission FIFO changed during event")),
        };
        let id = req.id;
        let live = sch::resident(s);
        let limit = self.c.limit();
        let worst = sch::reserve_rows(&self.c, &req).map_err(refused)?;
        let rows = self.admit_rows(&req)?;
        let ns = cache_namespace(&req.steering);
        let obs = self.observer();
        {
            let Inner { prompts, spans, waited, refused: refused_for, recall, recall_t0, recall_for, lend, .. } =
                &mut *inn;
            let prompt: Option<&[i32]> = if self.reuse() { prompts.get(&id).map(Vec::as_slice) } else { None };
            let (mut r, mut pre, mut fits) = self.try_fit(&mut req, rows, worst, prompt, &ns, obs)?;
            // no room: a decoding session may take the smaller R, and this request then tries the room it gave up
            if admit::shrink(fits, self.ops.make_room.is_some(), live, limit) {
                if let Some(k) = self.make_room(s, &req, rows, worst)? {
                    cm.c.has_shrink = true;
                    cm.c.shrink = k;
                    if let Some(t) = self.trace() {
                        let args = format!(
                            "{},{},{}",
                            targ("request", id),
                            targ("slot", u64::from(k.key.slot)),
                            targ("r", u64::from(k.r_rows))
                        );
                        t.instant(b"shrink", CONTROL, args.as_bytes());
                    }
                    (r, pre, fits) = self.try_fit(&mut req, rows, worst, prompt, &ns, obs)?;
                }
            }
            let room = self.room(obs);
            let fit = self
                .queue_check(st.revision, live < limit, fits, room)?
                .ok_or_else(|| refused("duplicate admission attempt without revision"))?;
            if let Some(t) = self.trace() {
                if admit::wait_line(fit.why, *waited, id) {
                    *waited = id;
                    let args = format!(
                        "{},{},{},{},{}",
                        targ("request", id),
                        tstr("why", if fit.why == STALL_SLOTS { "slots" } else { "capacity" }),
                        targ("requested", u64::from(fit.requested)),
                        targ("room", u64::from(fit.room)),
                        targ("live", u64::from(live))
                    );
                    t.instant(b"wait", CONTROL, args.as_bytes());
                }
            }
            if admit::refusal_line(fit.why, *refused_for, id) {
                *refused_for = id; // once per request
                if prompt.is_some() {
                    eline(format!(
                        "# s2 reuse request {} waits: {} retained {} lcp {} prompt {}\n",
                        id,
                        why(&pre),
                        pre.skip,
                        pre.lcp,
                        req.prompt_tokens
                    ));
                }
                if self.ops.refusal.is_some() {
                    let w = self.refusal(worst, r)?;
                    if !w.is_empty() {
                        let mut line = format!("# s2 rt: request {id} ({worst} rows) waits: ").into_bytes();
                        // printed as %s prints it: up to a NUL
                        line.extend_from_slice(w.split(|&c| c == 0).next().unwrap_or(&[]));
                        line.push(b'\n');
                        let _ = std::io::stderr().lock().write_all(&line);
                    }
                }
            }
            // borrow-idle: a request whose worst case passes the chunks every rank has online waits for a recall
            let short_of = if fit.why == STALL_NONE && lend.chunk_rows != 0 {
                lend.recall(worst, self.endpoint().agreed_room())
            } else {
                0
            };
            let note = admit::recall_note(fit.why, short_of, *recall_for, id);
            if note == admit::RECALL_BEGIN {
                *recall_for = id;
                *recall_t0 = Instant::now();
                eline(format!(
                    "# s2 flash kv: request {} ({} rows) waits for a recall to {} chunks ({} online)\n",
                    id,
                    worst,
                    short_of,
                    self.endpoint().agreed_room()
                ));
            } else if note == admit::RECALL_DONE {
                eline(format!(
                    "# s2 flash kv: request {} admitted {} ms after its recall began\n",
                    id,
                    recall_t0.elapsed().as_millis() as i64
                ));
                *recall_for = u64::MAX;
            }
            if short_of != 0 {
                *recall = short_of;
            }
            if admit::grant(fit.why, short_of) {
                let (mut run, mut skip, mut restored, mut counted) = (Run::default(), 0u32, 0u32, 0u32);
                let mut parked = -1i32; // park: the parked slot the prompt extends, which the grant takes
                if let Some(p) = prompt {
                    if self.ops.count {
                        counted = self.count(p, spans.get(&id).copied().unwrap_or(-1), &ns, obs);
                    }
                    let h = self.pool_match(p, rows, worst, req.prompt_tokens as i32 - 1, &ns, r)?;
                    if h.skip > 0 {
                        run = self.take(&h, rows, worst, r)?;
                        skip = h.skip as u32;
                        restored = if h.stored { h.tier + 1 } else { 0 };
                        parked = h.slot;
                    }
                    eline(format!(
                        "# s2 reuse request {} {} retained {} lcp {} prompt {} first page {}\n",
                        id,
                        why(&h),
                        h.skip,
                        h.lcp,
                        req.prompt_tokens,
                        if skip != 0 { run.first_page } else { 0 }
                    ));
                }
                // a pool that said it fits and then refuses is a bug, but one request's: it waits, the epoch goes on
                let mut got = true;
                if skip == 0 {
                    match self.reserve(rows, worst, r, obs) {
                        Ok(x) => run = x,
                        Err(e) => {
                            got = false;
                            eline(format!(
                                "# s2: request {id} ({worst} rows): the pool refused its reservation after fits(): {e}; it waits\n"
                            ));
                        }
                    }
                }
                if got {
                    let park = (parked >= 0).then_some(parked as u32);
                    let key = sch::next_key(&self.c, s, None, park).map_err(refused)?;
                    cm.c.has_admit = true;
                    cm.c.admit = Grant { key, req, run, reuse: skip, restored, counted };
                }
            }
        }
        inn.attempted_head = id;
        inn.attempted_revision = admit::attempted_revision(st.revision);
        if cm.c.has_admit {
            if let Some(t) = self.trace() {
                let args = format!(
                    "{},{},{},{}",
                    targ("request", id),
                    targ("reused", u64::from(cm.c.admit.reuse)),
                    targ("rows", worst),
                    targ("live", u64::from(live))
                );
                t.instant(b"admit", CONTROL, args.as_bytes());
            }
            self.queue_admitted(id)?;
            inn.commands.remove(&cmd.c.request);
            inn.prompts.remove(&cmd.c.request);
            inn.spans.remove(&cmd.c.request);
            inn.payloads.remove(&cmd.c.request);
            // another request queued and a slot left after this one: keep the ranks at this boundary for it. First
            // queue the ADMIT files that landed while this thread was inside the event chain (each event waits for
            // every rank's ACK, ~8 ms); the loop reads cmd files only between events, so without this a burst written
            // within 1 ms loses the hold after its 2nd grant and waits NOTICE_MAX boundaries
            if admit::drain(self.admit_hold, live, limit) {
                self.drain_admits(inn)?;
            }
            *hold = admit::hold(self.admit_hold, self.queue_front().is_some(), live, limit);
            cm.hold = *hold;
        }
        Ok(())
    }

    // --- every rank's watcher

    /// Fails on the shim's abort.s2. On DIR/quit stops at once, the head passes the quit to the workers, and the links
    /// stay open for linger_ms (so a worker reads EOF only after it is stopping too) before the channel fails. Quit is
    /// for an idle engine: a rank stopping mid-round leaves its peers' step.
    pub fn monitor(&self, linger_ms: u32) -> R<()> {
        let (abort, quit) = (self.path("abort.s2"), self.path("quit"));
        while !self.stopping() {
            require(!exists(&abort), "shim aborted S2 epoch")?;
            if exists(&quit) {
                if self.head {
                    self.endpoint().quit();
                }
                self.stop();
                let end = Instant::now() + Duration::from_millis(u64::from(linger_ms));
                while Instant::now() < end && !exists(&abort) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                self.shutdown();
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_layout() {
        let mut s = wire::identity_off();
        assert_eq!(cache_namespace(&s), [0; NAMESPACE_BYTES]);
        s.enabled = true;
        s.profile_digest = [7; 32];
        s.scope = 3;
        s.schedule_digest = [9; 32];
        s.scale_bits = 0x0403_0201;
        let n = cache_namespace(&s);
        assert_eq!((n[0], n[1], n[32], n[33], n[34], n[65]), (1, 7, 7, 3, 9, 9));
        assert_eq!(&n[66..70], &[1, 2, 3, 4]);
        s.images = true;
        s.image_digest = [5; 32];
        let n = cache_namespace(&s);
        assert_eq!((n[0], n[1], n[32], n[33]), (2, 5, 5, 0));
    }
}
