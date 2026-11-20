//! Every rank's compute thread and the driver hooks that carry service logic. It waits for a notice
//! while idle, stops at each event boundary for the commit (the module's fence, the staged request, the pool's and the
//! scheduler's digests), applies the commit, and runs the driver's steps between events. No thread is started here: the
//! C++ Service's compute thread calls `run`, and the driver's hooks call back in on that thread.
//!
//! What stays C++ is reached through the host table (`Host`, this thread's ctx): the driver facade (it keeps the
//! payloads and the planner), the parsed request, `Sessions` and `Dispatch`. The module's calls go through its table
//! and the pool's through `Control`, in the order the C++ made them: the workers' digests, the round logs and
//! module_calls_test's transcripts depend on it.

use super::control::{eline, hook, targ, Control, Host, Staged};
use super::files::{payload_hash, Command, Ingress};
use super::trace::{self, Trace, STEPS};
use crate::control::channel::{wall_us, Endpoint};
use crate::control::wire::{self, Commit, Notice};
use crate::control::{refused, require, Fail, R};
use crate::module::ModuleOps;
use crate::planner_ffi::CStats;
use crate::scheduler::{self as sch, Config, Grant, Key, Quantum, Slot, State, FREE, RETIRING};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::time::{SystemTime, UNIX_EPOCH};

/// g6svc_compute_cfg: the compute thread's settings, and what it borrows from the C++ Service for its life: the
/// control (its stop flag, progress and pool calls), the channel endpoint, the ingress (head), the trace (null: none)
/// and the driver's slots and counters.
#[repr(C)]
pub struct CCompute {
    pub control: *const c_void,
    pub endpoint: *const c_void,
    pub ingress: *const c_void,
    pub trace: *const c_void,
    pub slots: *const Slot,
    pub n_slots: usize,
    pub state: *const State,
    pub module: ModuleOps,
    pub host: Host,
    pub rank: u32,
    pub elog: bool,
    pub dfl_reuse: bool,
    pub ngram: bool,
}

// a step's trace span, open from its launch to the next boundary
struct Span {
    open: bool,
    kind: &'static str,
    t0: u64,
    args: String,
}

/// The compute thread's state. Only the compute thread calls in, from `run` and from the driver's hooks inside it.
pub struct Compute {
    ctl: *const Control,
    endpoint: *const Endpoint,
    ingress: *const Ingress,
    trace: *const Trace,
    slots: *const Slot,
    n_slots: usize,
    state: *const State,
    module: ModuleOps,
    host: Host,
    rank: u32,
    elog: bool,
    dfl_reuse: bool,
    ngram: bool,
    // the ADMIT's request, parsed before the event ACK and checked against the notice's hash
    staged: Cell<Option<Staged>>,
    step: RefCell<Span>,
    first: RefCell<BTreeSet<u64>>, // requests whose first token the trace has
    sync: Cell<u64>,               // barriers passed (the module's drained)
}

fn wall_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

fn kind_name(k: u32) -> &'static str {
    match k {
        wire::RESIZE => "resize",
        wire::CANCEL => "cancel",
        _ => "admit",
    }
}

impl Compute {
    /// # Safety
    /// Every pointer in `c` must stay valid for the Compute's life: the control, endpoint, ingress and trace handles the
    /// Rust objects the C++ Service holds, the slots and state the driver's.
    pub unsafe fn new(c: &CCompute) -> R<Compute> {
        require(
            !c.control.is_null() && !c.endpoint.is_null() && !c.slots.is_null() && !c.state.is_null(),
            "compute handles",
        )?;
        require(c.rank != 0 || !c.ingress.is_null(), "compute handles")?;
        let h = &c.host;
        require(
            h.retire_run.is_some()
                && h.idle_boundary.is_some()
                && h.stage.is_some()
                && h.unstage.is_some()
                && h.digest.is_some()
                && h.abort.is_some()
                && h.apply.is_some()
                && h.mode.is_some()
                && h.step.is_some()
                && h.counters.is_some()
                && h.begin.is_some()
                && h.bound.is_some()
                && h.release.is_some(),
            "the compute thread's host calls",
        )?;
        require(
            c.module.fence.is_some() && c.module.bind.is_some() && c.module.retire.is_some(),
            "incomplete native service hooks",
        )?;
        Ok(Compute {
            ctl: c.control.cast(),
            endpoint: c.endpoint.cast(),
            ingress: c.ingress.cast(),
            trace: c.trace.cast(),
            slots: c.slots,
            n_slots: c.n_slots,
            state: c.state,
            module: c.module,
            host: c.host,
            rank: c.rank,
            elog: c.elog,
            dfl_reuse: c.dfl_reuse,
            ngram: c.ngram,
            staged: Cell::new(None),
            step: RefCell::new(Span { open: false, kind: "", t0: 0, args: String::new() }),
            first: RefCell::new(BTreeSet::new()),
            sync: Cell::new(0),
        })
    }

    // --- borrowed objects

    fn ctl(&self) -> &Control {
        // SAFETY: the C++ Service frees this Compute before its Control (new's contract)
        unsafe { &*self.ctl }
    }
    fn endpoint(&self) -> &Endpoint {
        // SAFETY: as ctl; the endpoint's calls are safe from any thread
        unsafe { &*self.endpoint }
    }
    fn trace(&self) -> Option<&Trace> {
        // SAFETY: as ctl; the trace locks each write
        unsafe { self.trace.as_ref() }
    }
    fn c(&self) -> &Config {
        self.ctl().config()
    }
    /// The driver's slots and counters. The driver changes them only inside the host's apply, idle_boundary and step,
    /// on this thread, so a read between those calls sees them whole.
    fn view(&self) -> (&[Slot], &State) {
        // SAFETY: the driver's view, valid for the Service's life (new's contract)
        unsafe { (std::slice::from_raw_parts(self.slots, self.n_slots), &*self.state) }
    }
    fn head(&self) -> bool {
        self.rank == 0
    }

    // --- the host

    fn idle_boundary(&self, b: u64) -> R<()> {
        // SAFETY: new checked the member; the host's context outlives the Compute
        hook(unsafe { self.host.idle_boundary.unwrap_unchecked()(self.host.ctx, b) })
    }
    fn stage(&self, bytes: &[u8]) -> R<()> {
        let mut st = Staged { req: std::ptr::null(), prompt: std::ptr::null(), n: 0, sys: 0, plp: 0 };
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.stage.unwrap_unchecked()(self.host.ctx, bytes.as_ptr(), bytes.len(), &mut st) })?;
        self.staged.set(Some(st));
        Ok(())
    }
    fn unstage(&self) -> R<()> {
        self.staged.set(None);
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.unstage.unwrap_unchecked()(self.host.ctx) })
    }
    // the pool's digest into the driver, then the scheduler's: once per event, never per round
    fn digest(&self) -> R<u64> {
        let pool = self.ctl().digest()?;
        let mut d = 0u64;
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.digest.unwrap_unchecked()(self.host.ctx, pool.is_some(), pool.unwrap_or(0), &mut d) })?;
        Ok(d)
    }
    fn abort(&self, k: &Key) -> R<()> {
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.abort.unwrap_unchecked()(self.host.ctx, k) })
    }
    fn apply(&self, cm: &Commit) -> R<()> {
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.apply.unwrap_unchecked()(self.host.ctx, cm, cm.c.has_admit) })
    }
    fn mode(&self) -> R<()> {
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.mode.unwrap_unchecked()(self.host.ctx) })
    }
    fn step_host(&self) -> R<Quantum> {
        let mut q = Quantum::default();
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.step.unwrap_unchecked()(self.host.ctx, &mut q) })?;
        Ok(q)
    }
    fn counters(&self, keys: &[Key]) -> R<()> {
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.counters.unwrap_unchecked()(self.host.ctx, keys.as_ptr(), keys.len() as u32) })
    }
    fn retire_run(&self, s: &Slot) -> R<()> {
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.retire_run.unwrap_unchecked()(self.host.ctx, s) })
    }

    // --- the module

    fn fence(&self) -> R<()> {
        // SAFETY: new checked the member; the module's table and context outlive the service
        hook(unsafe { self.module.fence.unwrap_unchecked()(self.module.ctx) })
    }
    fn room(&self) -> R<u32> {
        let Some(f) = self.module.flash_room else { return Ok(u32::MAX) };
        let mut n = 0u32;
        // SAFETY: as fence
        hook(unsafe { f(self.module.ctx, &mut n) })?;
        Ok(n)
    }
    fn drained(&self) -> R<bool> {
        let f = self.module.drained.ok_or_else(|| refused("bad_function_call"))?;
        let mut d = false;
        // SAFETY: as fence
        hook(unsafe { f(self.module.ctx, &mut d) })?;
        Ok(d)
    }

    // --- progress

    fn publish(&self) {
        let (s, st) = self.view();
        self.ctl().publish(st.boundary, st.revision, sch::horizon(self.c(), s, st), sch::resident(s), sch::retiring(s));
    }
    fn end_step(&self) {
        let mut sp = self.step.borrow_mut();
        if !sp.open {
            return;
        }
        if let Some(t) = self.trace() {
            t.span(sp.kind.as_bytes(), STEPS, sp.t0, trace::now(), sp.args.as_bytes());
        }
        sp.open = false;
    }

    // --- the loop

    /// The compute loop, until the service stops (Ok) or something fails.
    pub fn run(&self) -> R<()> {
        while !self.ctl().stopping() {
            self.end_step();
            let mut noticed = Notice::default();
            if sch::resident(self.view().0) == 0 {
                let n = self.endpoint().wait_notice()?;
                self.idle_boundary(n.boundary)?;
            }
            let cm = self.endpoint().boundary(self.view().1.boundary, || {
                self.fence()?;
                let n = self.endpoint().wait_notice()?;
                noticed = n;
                self.unstage()?;
                if n.kind == wire::ADMIT {
                    // the head reads its own req.<id>; a worker has the bytes from the head
                    let bytes = if self.head() {
                        let c = Command { epoch: n.epoch, request: n.request, hash: n.payload_hash, admit: true };
                        // SAFETY: the head's ingress outlives the Compute; it allows the control thread's reads too
                        unsafe { (*self.ingress).request(&c)? }
                    } else {
                        self.endpoint().request(n.request)?
                    };
                    require(payload_hash(&bytes) == n.payload_hash, "request bytes do not match the ADMIT notice")?;
                    self.stage(&bytes)?;
                }
                let d = self.digest()?;
                Ok((d, self.room()?))
            })?;
            if let Some(cm) = cm {
                self.commit(&cm, &noticed)?;
                continue;
            }
            if self.module.before_work.is_some() {
                // SAFETY: as fence
                hook(unsafe { self.module.before_work.unwrap_unchecked()(self.module.ctx) })?;
            }
            let t0 = if self.trace.is_null() { 0 } else { trace::now() };
            let work = self.step_host()?;
            require(work.kind != sch::IDLE, "departed idle boundary")?;
            // rank 0: each session's spec counters, once a second
            if work.count != 0 {
                self.counters(&work.keys[..work.count as usize])?;
            }
            let (s, _) = self.view();
            let id = |i: usize| s[work.keys[i].slot as usize].req.id;
            if self.trace().is_some() {
                let ids: Vec<String> = (0..work.count as usize).map(|i| id(i).to_string()).collect();
                let kind = match work.kind {
                    sch::K_PREFILL => "prefill",
                    sch::PROMPT_TAIL => "prompt tail",
                    _ => "decode",
                };
                let args = format!(
                    "{},{},{},\"requests\":[{}]",
                    targ("boundary", work.boundary),
                    targ("first", u64::from(work.first)),
                    targ("rows", u64::from(work.rows)),
                    ids.join(",")
                );
                *self.step.borrow_mut() = Span { open: true, kind, t0, args };
            }
            if self.elog && work.kind != sch::K_DECODE {
                let kind = match work.kind {
                    sch::K_PREFILL => "prefill",
                    sch::PREFILL_PACK => "prefill_pack",
                    sch::PROMPT_TAIL_PACK => "tail_pack",
                    _ => "tail",
                };
                eline(format!(
                    "# s2_event step rank {} boundary {} request {} {} first {} rows {}\n",
                    self.rank,
                    work.boundary,
                    id(0),
                    kind,
                    work.first,
                    work.rows
                ));
            }
            self.mode()?;
            self.publish();
        }
        Ok(())
    }

    // The commit's application, every rank at the same boundary.
    fn commit(&self, cm: &Commit, noticed: &Notice) -> R<()> {
        // a CANCEL may arrive before the session's first round, or after several: its counters before the driver
        // releases it
        if cm.c.has_cancel {
            let k = self
                .view()
                .0
                .iter()
                .find(|o| o.phase != FREE && o.phase != RETIRING && o.key == cm.c.cancel)
                .map(|o| o.key);
            if let Some(k) = k {
                self.abort(&k)?;
            }
        }
        if cm.c.has_admit {
            require(self.staged.get().is_some(), "commit without verified request")?;
        }
        self.apply(cm)?;
        if let Some(t) = self.trace() {
            let args = format!(
                "{},\"kind\":\"{}\",{}",
                targ("request", noticed.request),
                kind_name(noticed.kind),
                targ("boundary", cm.c.boundary)
            );
            t.instant(b"commit", STEPS, args.as_bytes());
        }
        if self.elog {
            // t_us: now (applied); installed_us / arrived_us: this rank's notice install and boundary arrival, so
            // arrived - installed is the notice's margin on this rank
            let (installed, arrived) = self.endpoint().stamps();
            eline(format!(
                "# s2_event commit rank {} event {} {} request {} boundary {} t_us {} installed_us {} arrived_us {}\n",
                self.rank,
                noticed.event,
                kind_name(noticed.kind),
                noticed.request,
                cm.c.boundary,
                wall_us(),
                installed,
                arrived
            ));
        }
        if !self.head() && cm.c.has_admit {
            self.endpoint().admitted(cm.c.admit.req.id);
        }
        if cm.has_resize {
            self.ctl().resized();
        }
        self.mode()?;
        self.publish();
        self.unstage()?;
        self.endpoint().applied()
    }

    // --- the driver's hooks (DriverHooks), from the C++ Driver inside the host's apply and step

    /// A grant, every rank: a worker first makes the head's pool calls on its own pool and checks it got the head's
    /// run; then Dispatch::begin, the module's bind_request and bind, and Sessions::bind.
    pub fn bind(&self, g: &Grant) -> R<()> {
        let st = self.staged.get().ok_or_else(|| refused("binding without verified payload"))?;
        let ctl = self.ctl();
        require(g.reuse == 0 || ctl.reuse(), "reused rows without prefix reuse")?;
        if !self.head() {
            // the head's calls, on the same pool state
            let worst = sch::reserve_rows(self.c(), &g.req).map_err(refused)?;
            let rows = ctl.admit_rows(&g.req)?;
            let ns = super::control::cache_namespace(&g.req.steering);
            let obs = ctl.observer();
            // SAFETY: the staged prompt stays valid until the next stage or unstage, which run makes only after apply
            let prompt = if st.n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(st.prompt, st.n) } };
            if ctl.counts() && ctl.reuse() && st.plp < 0 {
                require(ctl.count(prompt, st.sys, &ns, obs) == g.counted, "worker prefix counts differ from the head")?;
            } else {
                require(g.counted == 0, "a counted grant without counting")?;
            }
            let mine = if g.reuse != 0 {
                let h = ctl.pool_match(prompt, rows, worst, g.req.prompt_tokens as i32 - 1, &ns, g.req.r_rows)?;
                let restored = if h.stored { h.tier + 1 } else { 0 };
                require(
                    h.skip == g.reuse as i32 && restored == g.restored,
                    "worker prefix reuse differs from the head",
                )?;
                require(h.slot < 0 || h.slot == g.key.slot as i32, "worker parked slot differs from the head's grant")?;
                ctl.take(&h, rows, worst, g.req.r_rows)?
            } else {
                ctl.reserve(rows, worst, g.req.r_rows, obs).map_err(|e| Fail::Refused(Cow::Owned(e.into_bytes())))?
            };
            require(mine == g.run, "worker page list differs from the head's grant")?;
        }
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.begin.unwrap_unchecked()(self.host.ctx, g) })?;
        if let Some(f) = self.module.bind_request {
            // SAFETY: as fence; the staged request as above
            hook(unsafe { f(self.module.ctx, g.key.slot, st.req.cast()) })?;
        }
        // SAFETY: as fence
        hook(unsafe { self.module.bind.unwrap_unchecked()(self.module.ctx, g) })?;
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.bound.unwrap_unchecked()(self.host.ctx, g) })
    }

    /// A slot leaves at a commit. A worker's cancelled live session gives its run back here (the head's went back when
    /// it built the commit); a finished one gave it back at its step.
    pub fn retire(&self, old: &Slot) -> R<()> {
        if !self.head() && old.phase != RETIRING && self.ctl().reuse() {
            self.retire_run(old)?;
        }
        // SAFETY: as fence
        hook(unsafe { self.module.retire.unwrap_unchecked()(self.module.ctx, old) })?;
        // SAFETY: as idle_boundary
        hook(unsafe { self.host.release.unwrap_unchecked()(self.host.ctx, old) })
    }

    /// A session finished at this step: its run goes back now, so the sessions still decoding can grow before the next
    /// commit frees the slot. Without prefix reuse the driver releases it.
    pub fn finished(&self, s: &Slot) -> R<()> {
        if let Some(t) = self.trace() {
            let args = format!("{},{}", targ("request", s.req.id), targ("tokens", u64::from(s.emitted)));
            t.instant(b"finish", STEPS, args.as_bytes());
            self.first.borrow_mut().remove(&s.req.id);
        }
        if self.ctl().reuse() {
            self.retire_run(s)?;
        }
        Ok(())
    }

    /// A session's first emission, when tracing.
    pub fn first_token(&self, k: &Key) {
        let id = self.view().0[k.slot as usize].req.id;
        if self.first.borrow_mut().insert(id) {
            if let Some(t) = self.trace() {
                t.instant(b"first token", STEPS, targ("request", id).as_bytes());
            }
        }
    }

    /// After the growth and before a round: when this rank ran tier copies, every rank waits here for the slowest
    /// rank's (the tier io lines carry the same drains).
    pub fn settle(&self) -> R<()> {
        if !self.drained()? {
            return Ok(());
        }
        let b0 = wall_ms();
        let seq = self.sync.get() + 1;
        self.sync.set(seq);
        self.endpoint().sync(seq)?;
        eline(format!("# rank {}: s2 drain barrier {}: in {} ms, out {} ms\n", self.rank, seq, b0, wall_ms()));
        Ok(())
    }

    /// A session's drafter lines at its completion (Sessions writes its terminal lines after).
    pub fn terminal(&self, slot: u32, st: &CStats) {
        if st.switches != 0 || st.mtp_rounds != 0 {
            eline(format!(
                "# s2 drafter choice slot {}: {} of {} rounds on MTP, {} switches, {} tokens\n",
                slot, st.mtp_rounds, st.rounds, st.switches, st.tokens
            ));
        }
        // every request, so one that never had the drafter shows too
        if self.dfl_reuse && self.head() {
            eline(format!(
                "# s2 dflash reuse slot {}: {} of {} rounds on DFlash2, {} on MTP; context from {} (taps from {}), \
                 warm-up to {}, promoted at {}, {} tokens\n",
                slot,
                st.dfl_rounds,
                st.rounds,
                st.mtp_rounds,
                st.dfl_lo,
                st.dfl_from,
                st.dfl_warm,
                st.promoted,
                st.tokens
            ));
        }
        if self.ngram && self.head() {
            eline(format!(
                "# s2 ngram slot {}: {} of {} rounds verified a table chain, {} of {} of its tokens accepted, {} tokens\n",
                slot, st.ngram_rounds, st.rounds, st.ngram_acc, st.ngram_drafted, st.tokens
            ));
        }
    }
}
