// Host orchestration for g6m4's OPT-IN S2 adapter. NativeHooks must be backed by
// the real TP4 pool and fc2 CUDA APIs; test pools never enable native readiness.
#pragma once
#include "bootstrap.h"
#include "control_channel.h"
#include "dispatch.h"
#include "flash_lend.h"
#include "pool.h"
#include "pool_ops.h"
#include "trace.h"
#include "../../knobs-rs/gr8_knobs.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace g6s2 {
// rows: what a request needs at admission (grant_rows), worst: all it may grow to (reserve_rows), r: the R its run
// takes (Request::r_rows, 0 = the pool's)
struct PoolHooks {
    // the rank's pool itself. The service makes the plain calls on it (free pages, room, reserve, fits, count, stats,
    // runs; held to ops.max_run_pages), and the driver its own (driver.h DriverPool): growth, a worker's half of a
    // shrink, and a leaving session's release when nothing is retained. Required
    RunPool* pool = nullptr;
    // the module's pool policy (PoolOps, pool_ops.h). When it is set, the members below stay empty. Unset: FnPool builds
    // a table over them
    g6s2_pool_ops ops{};
    // optional (remote rows release): why a stalled request past R does not fit its page classes; empty = no reason
    std::function<std::string(uint64_t worst, uint32_t r)> refusal;
    // optional, head only: the R a request's run takes (Request::r_rows: 0 or Config::r2_rows). prompt: null without
    // prefix reuse. Unset = 0
    std::function<uint32_t(const Request&, uint64_t rows, uint64_t worst, const std::vector<int>* prompt)> r_rows;
    // optional (with r_rows): a request that does not fit. make_room, head only: a decoding session to take the smaller
    // R, its run shrunk on the head's pool (nothing = none to shrink). The driver makes the same change on a worker's pool
    std::function<std::optional<Shrink>(const std::array<Slot, MAX_SESSIONS>& slots, const Request&, uint64_t rows,
                                        uint64_t worst)> make_room;
    // growth on demand (pool.h grow): the driver grows each decoding run before a round, every rank; a run the pool
    // cannot grow sits the round out
    bool grow = false;
    // the most pages one run may hold (0 = no limit; DS41: one session's slab): fits refuses a larger worst case, reserve
    // throws, room is at most this
    uint32_t max_run_pages = 0;
    // status.s2 shows the pool's growth totals and each live run's pages now and at worst (pool.h stats, runs)
    bool status = false;
    // optional, prefix reuse (pool.h keep): every rank retires a run with its session's reuse state; the head matches a
    // new prompt against the retained runs (limit: rows its prefill may skip) and takes a hit. retain and park are module
    // calls (ModuleOps); they are here for a table built over these std::functions (NativeHooks::module unset)
    std::function<void(const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)> retain;
    std::function<PoolHit(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit,
                          const g6rt::CacheNamespace&, uint32_t r)> match;
    std::function<Run(const PoolHit&, uint64_t rows, uint64_t worst, uint32_t r)> take;
    // optional, Config::park (DS41): every rank, a finished session's slot parks with its reuse state instead of retain;
    // match may then name the parked slot (PoolHit::slot) and the grant takes it
    std::function<void(unsigned slot, const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)> park;
    // block stores (pool.h add_tier): count = admissions count toward shared prefixes (pool.h count, head and workers,
    // before the grant, including nonstock calls that clear pending stock metadata), and each tier's counters for status.s2
    bool count = false;
    std::function<std::vector<g6kv_tier_totals>()> tiers;
    // optional: the pool's state, compared across ranks at every event (Scheduler::pool_digest)
    std::function<uint64_t()> digest;
};
struct NativeHooks {
    PoolHooks pool;
    // the module (ModuleOps). Unset: a table over the std::function members below and pool.retain and pool.park, which
    // must then be empty when it is set
    g6s2_module_ops module{};
    std::function<void(const Grant&)> bind; // native main AND draft binding, no pool mutation
    // optional, before bind on every rank: the bound request itself (DS41: its image rows, for the Engram ring of the
    // slot's snapshots and parks, s2/ds41_engram_ring.h)
    std::function<void(unsigned slot, const g6rt::ServeReq&)> bind_request;
    // with pool.grow: slot's run grew (the driver's pool call); its new pages go into the main and draft tables before
    // the round
    std::function<void(unsigned slot, const Run&)> grown;
    std::function<void(const Slot&)> retire; // native per-slot graph/view cleanup only
    std::function<void()> fence; // all native streams/drafts quiescent before event ACK
    // optional: every rank, at each boundary after its last commit and before its work (compute thread). Flash takes a
    // prefill's checkpoint here (flash_native.cuh), so a Resize that lands at the boundary the quantum ended on is in
    // place before the capture
    std::function<void()> before_work;
    // optional: did this rank run tier copies since the last call (the same answer on every rank: the copies are). Then
    // the ranks line up before the next round: one rank's copies can take seconds longer than another's, and a GPU wait
    // for a peer's rows gives up after ~10 s (G6_SPIN_TIMEOUT_CYC)
    std::function<bool()> drained;
    // optional, a tier on a group lease of a KV node (s2/kv_share.h): share = the head's report to the node, on its
    // control thread every GR8_S2_SHARE_MS (default 1000) while no resize is in flight, told whether the engine is idle
    // (no resident session, none queued); the resize to announce, or nothing. resize = every rank's change at the
    // Resize commit (share_apply with the tier drain)
    std::function<std::optional<Resize>(bool idle)> share;
    std::function<void(const Resize&)> resize;
    // with pool.make_room: a session takes the smaller R, every rank at the commit (its run as it was): the main and
    // draft tables, and the pages between the two Rs that leave its prefix
    std::function<void(const Shrink&, const Run& was, uint64_t length)> shrink;
    // optional, GR8_GLM53F_KV_CHUNKS: flash_room = this rank's Flash KV chunks online (its ACKs carry it, Ack::room);
    // flash = the head's borrow-idle rule (s2/flash_lend.h; chunk_rows 0 = no gate, idle_ms 0 = never lend). Its lends
    // and recalls are Resize events on FLASH_KV_TIER, so they need the resize hook
    std::function<uint32_t()> flash_room;
    FlashLend flash;
    DispatchHooks dispatch;
    // DFlash2 (dispatch.dfl set): drafts a round verifies and the --spec-adaptive policy (s2_batch.h Batch::dflash)
    int dfl_kv = 7;
    g6rt::SpecLen dfl_spec;
    s2::DSel dsel;                    // drafter choice (GR8_S2_DSEL)
    s2::KStop kstop;                  // solo MTP rounds' draft count (GR8_MTP_KSTOP, GR8_MTP_LPLOG)
    s2::DflReuse dfl_reuse;           // GR8_DFL_REUSE
    s2::NgramSrc ngram;               // GR8_NGRAM: the n-gram table as a draft source (S5, #56)
};
// what status.s2 shows of the pool (Service::write_status)
struct PoolStatus {
    uint32_t free = 0, room = 0;
    std::optional<g6kv_totals> stats;
    std::optional<std::vector<g6kv_tier_totals>> tiers;
    std::optional<std::vector<PoolUse>> runs;
};
// the Rust control thread (engine/scheduler-rs src/service/control.rs)
struct ControlFree { void operator()(g6svc_control* c) const { g6svc_control_free(c); } };
using ControlHandle = std::unique_ptr<g6svc_control, ControlFree>;
class Service {
    BootstrapConfig c_; ControlLinks& links_; NativeHooks native_;
    std::unique_ptr<FnModule<NativeHooks>> fn_;   // the table over native_'s hooks, when native_.module is unset
    Module mod_;
    std::unique_ptr<FnPool<PoolHooks>> fp_;       // the table over native_.pool's hooks, when native_.pool.ops is unset
    PoolPolicy pol_;
    Sessions::Logprob logprob_;                   // the module's logprob, for Sessions
    Sessions sessions_; Dispatch dispatch_; std::unique_ptr<Driver> driver_;
    std::unique_ptr<LeaderChannel> leader_; std::unique_ptr<WorkerChannel> worker_;
    EventEndpoint* endpoint_ = nullptr;
    Ingress ingress_; AdmissionQueue queue_;   // the head's: the Rust control thread reads and keeps them
    bool reuse() const { return pol_.o.match != nullptr; }
    // rows a request's run holds at admission: grant_rows when runs grow on demand, else its worst case
    uint64_t admit_rows(const Request& r) const { return pol_.o.grow ? grant_rows(c_.scheduler, r) : reserve_rows(c_.scheduler, r); }
    // the plain pool calls a worker's bind makes, each run held to the module's max_run_pages. reserve: the head after
    // all event ACKs, a worker at the grant, so the same call on the same pool state gives the same Run
    RunPool& pool() const { return *native_.pool.pool; }
    bool within(uint64_t worst) const { return !pol_.o.max_run_pages || worst <= uint64_t(pol_.o.max_run_pages) * PAGE; }
    Run pool_reserve(uint64_t rows, uint64_t worst, uint32_t r) const {
        require(within(worst), "a run larger than the module's max_run_pages (DS41: one session's slab)");
        return pool().reserve(rows, worst, r);
    }
    std::optional<g6rt::ServeReq> staged_; // COMPUTE thread only, parsed before event ACK
    std::mutex failure_mu_; std::exception_ptr failure_;
    uint64_t sync_ = 0;   // compute thread: barriers passed (the module's drained)
    // GR8_S2_EVENTLOG=1: the event-latency diagnostic #61 needs. One "notice" line when the head anchors a notice (the
    // boundary it read and the future boundary it chose), one "commit" line when that notice lands on each rank, and one
    // "step" line per prefill quantum. Together they say whether a cancel was installed while a long prefill was still
    // running -- the lookahead binding -- or merely waited for it, which is the scheduler's admission. Logs only: no
    // state, no digest and no behaviour change, so it can run on a serving build.
    const bool elog_ = knob::s2_eventlog();
    // GR8_S2_TRACE=PATH: this rank's steps, and on the head the arrivals, waits and admissions, as a Chrome trace
    // (trace.h). A step's span runs from its launch to the next boundary.
    Trace trace_;
    struct { bool open = false; const char* kind = ""; uint64_t t0 = 0; std::string args; } step_;
    std::set<uint64_t> first_;    // requests whose first token the trace has
    void end_step() {
        if (!step_.open) return;
        trace_.span(step_.kind, Trace::STEPS, step_.t0, Trace::now(), step_.args);
        step_.open = false;
    }
    // the Rust control thread and the state the threads share: the progress the compute thread publishes, a resize in
    // flight, and the stop flag. ctl_ outlives the threads (run joins them) and is freed before what it borrows
    ControlHandle ctl_;
    bool stopping() const { return g6svc_control_stopping(ctl_.get()); }
    void publish(const SchedulerView& s) {
        g6svc_control_publish(ctl_.get(), s.boundary(), s.revision(), s.horizon(), s.resident(), s.retiring());
    }
    void fail(std::exception_ptr error) {
        { std::lock_guard<std::mutex> lock(failure_mu_); if (!failure_) failure_ = error; }
        g6svc_control_shutdown(ctl_.get());
    }
    void shutdown() { g6svc_control_shutdown(ctl_.get()); }
    // What the Rust control thread calls in C++ (g6svc_host). The pool policy's exception, if one threw inside the
    // Rust, stays in its own context (PoolOps) and is rethrown from there.
    struct Host : ModuleCtx {
        Service& s;
        std::vector<int> prompt;                  // the last checked request's, until the next check
        std::string why;                          // the pool policy's last refusal
        std::vector<g6kv_tier_totals> tiers;      // and its last tier counters
        explicit Host(Service& x) : s(x) {}
        static int32_t check(void* x, uint64_t id, const uint8_t* b, size_t n, g6svc_checked* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                auto request = parse_request(std::string(reinterpret_cast<const char*>(b), n), h.s.c_.scheduler.steering);
                out->ok = h.s.grammar_ok(id, request) && h.s.images_ok(id, request);
                if (!out->ok) return;
                h.prompt = std::move(request.prompt);
                out->prompt = h.prompt.data(); out->n = h.prompt.size();
                out->sys = request.sys; out->plp = request.plp; out->n_gen = request.n_gen;
                std::memcpy(&out->steering, &request.steering, sizeof out->steering);
            });
        }
        static int32_t retire_run(void* x, const g6sch_slot* sl) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { h.s.retire_run(*mod::as<const Slot>(sl)); });
        }
        static int32_t share(void* x, bool idle, bool* has, g6sch_resize* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                const auto r = h.s.native_.share(idle);
                if ((*has = r.has_value())) *out = g6sch_resize{r->tier, r->segs, r->slots, r->txn};
            });
        }
        static int32_t refusal(void* x, uint64_t worst, uint32_t r, const uint8_t** b, size_t* n) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                h.why = h.s.pol_.refusal(worst, r);
                *b = reinterpret_cast<const uint8_t*>(h.why.data()); *n = h.why.size();
            });
        }
        static int32_t tiers_of(void* x, const g6kv_tier_totals** t, size_t* n) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { h.tiers = h.s.pol_.tiers(); *t = h.tiers.data(); *n = h.tiers.size(); });
        }
    };
    Host host_{*this};
    // a control call's result: what the host or the pool policy threw (rc 1) rethrown, or the refusal (rc 2)
    void control_check(int32_t rc) {
        if (rc == 1 && !host_.err && pol_.o.ctx) {
            std::exception_ptr& e = static_cast<ModuleCtx*>(pol_.o.ctx)->err;
            if (e) { std::exception_ptr x = e; e = nullptr; std::rethrow_exception(x); }
        }
        ctl::check(rc, &host_.err);
    }
    // a slot leaves: its run back to the pool, retained with the session's reuse state when prefix reuse is on. A
    // finished session at the step it finished (Driver finished hook, every rank), a cancelled one at the event: head in
    // event(), workers at apply, the same point of the same commit. So every rank keeps the same runs. Without prefix
    // reuse the driver releases the run itself (DriverPool::release), except the head's cancelled one: that goes back
    // here, when the head builds the commit
    void retire_run(const Slot& s) {
        if (!reuse()) { pool().release(s.run); return; }
        g6rt::PrefixReuse st = dispatch_.rows(s.key.slot);
        const auto gen = driver_->emitted(s.key.slot);
        st.finish(driver_->prompt(s.key.slot), gen, int(gen.size()), true);
        // park: a finished session (not a cancelled one) keeps its slot; the scheduler parks it at the next commit
        if (c_.scheduler.park && s.phase == Phase::Retiring) {
            mod_.park(s.key.slot, s.run, st, g6rt::steering::cache_namespace(s.req.steering));
            return;
        }
        mod_.retain(s.run, st, g6rt::steering::cache_namespace(s.req.steering));
    }
    // status.s2 from these pool values. Only service_vectors_test calls it: the control thread's publish reads the pool itself
    void write_status(const PoolStatus& p) {
        g6svc_status st{};
        st.free = p.free; st.room = p.room;
        if ((st.has_growth = p.stats.has_value())) {
            const auto& g = *p.stats;
            st.held_pages = g.held_pages; st.worst_pages = g.worst_pages; st.grows = g.grows; st.stalled = g.stalls;
        }
        if ((st.has_tiers = p.tiers.has_value())) { st.tiers = p.tiers->data(); st.n_tiers = p.tiers->size(); }
        std::vector<g6svc_run_use> runs;
        if ((st.has_runs = p.runs.has_value())) {
            for (const PoolUse& u : *p.runs) runs.push_back({u.table_version, u.held, u.worst});
            st.runs = runs.data(); st.n_runs = runs.size();
        }
        ctl::check(g6svc_control_write_status(ctl_.get(), &st));
    }
    // A grammar that does not compile ends its request with an error before it is queued, so no request that asks
    // for a grammar runs without one. The front end compiles each schema first and returns a 400 for one that fails.
    // This catches a schema that compiles there but not with the engine's fixed layout (grammar.cc).
    bool grammar_ok(uint64_t id, const g6rt::ServeReq& r) {
        if (!r.gram.on()) return true;
        try { g6rt::grammar_open(r.gram); return true; }
        catch (const std::exception& e) {
            fprintf(stderr, "# s2: request %llu: the grammar does not compile: %s\n", (unsigned long long)id, e.what());
            atomic_file(c_.directory + "/resp." + std::to_string(id), "end error -1 0\n");
            return false;
        }
    }
    // An image request the engine cannot run (ds41_image.h img::refusal) ends with an error before it is queued, as a
    // grammar that does not compile does. The front end refuses the same cases first.
    bool images_ok(uint64_t id, const g6rt::ServeReq& r) {
        if (r.images.empty()) return true;
#if G6_MODEL_DS41
        const bool decode_prompt = ds41_decode_prompt(c_.prefill_min, r.prompt.size(), r.plp, r.steering.enabled);
#else
        const bool decode_prompt = true;
#endif
        const char* why = g6rt::img::refusal(r.prompt, r.images, decode_prompt);
        if (!*why) return true;
        fprintf(stderr, "# s2: request %llu: images refused: %s\n", (unsigned long long)id, why);
        atomic_file(c_.directory + "/resp." + std::to_string(id), "end error -1 0\n");
        return false;
    }
    void compute() {
        while (!stopping()) {
            end_step();
            Notice noticed{};
            if (!driver_->state().resident()) {
                const auto n = endpoint_->wait_notice(); driver_->idle_boundary(n.boundary);
            }
            auto cm = endpoint_->boundary(driver_->state().boundary(), [&] {
                mod_.fence(); const auto n = endpoint_->wait_notice(); noticed = n; staged_.reset();
                if (n.kind == EventKind::Admit) {   // the head reads its own req.<id>; a worker has the bytes from the head
                    const auto bytes = worker_ ? worker_->request(n.request) : ingress_.request({true,n.epoch,n.request,n.payload_hash});
                    require(payload_hash(bytes) == n.payload_hash, "request bytes do not match the ADMIT notice");
                    staged_ = parse_request(bytes, c_.scheduler.steering);
                }
                if (pol_.o.digest) driver_->pool_digest(pol_.digest());
                return driver_->state().digest(); // once per event, NEVER per round
            });
            if (cm) {
                // CANCEL may arrive before Batch::admit, or after several rounds.
                // Capture its actual request counters before Driver releases Batch.
                for (const auto& old : driver_->state().slots())
                    if (old.phase != Phase::Free && old.phase != Phase::Retiring && cm->cancel && old.key == *cm->cancel) {
                        const auto* stats = driver_->stats_if_initialized(old.key);
                        sessions_.terminal(old.key, stats ? *stats : s2::Stats{}, "abort");
                    }
                std::optional<Payload> payload;
                if (cm->admit) { require(staged_.has_value(), "commit without verified request"); payload = sessions_.payload(*staged_); }
                driver_->apply(*cm, payload ? &*payload : nullptr);
                if (trace_)
                    trace_.instant("commit", Trace::STEPS, targ("request", noticed.request) + "," +
                                   targ("kind", noticed.kind == EventKind::Resize ? "resize"
                                                : noticed.kind == EventKind::Cancel ? "cancel" : "admit") + "," +
                                   targ("boundary", cm->boundary));
                if (elog_) {
                    // t_us: now (applied); installed_us / arrived_us: this rank's notice install and boundary arrival
                    // (control_channel.h stamps), so arrived - installed is the notice's margin on this rank
                    const auto st = endpoint_->stamps();
                    fprintf(stderr, "# s2_event commit rank %u event %llu %s request %llu boundary %llu t_us %lld installed_us %lld arrived_us %lld\n", c_.rank,
                            (unsigned long long)noticed.event,
                            noticed.kind == EventKind::Resize ? "resize" : noticed.kind == EventKind::Cancel ? "cancel" : "admit",
                            (unsigned long long)noticed.request, (unsigned long long)cm->boundary, wall_us(), st.first, st.second);
                }
                if (worker_ && cm->admit) worker_->admitted(cm->admit->req.id);
                if (cm->resize) g6svc_control_resized(ctl_.get());
                dispatch_.synchronize_mode(driver_->state()); publish(driver_->state());
                staged_.reset(); endpoint_->applied(); continue;
            }
            if (mod_.o.before_work) mod_.before_work();
            const auto t0 = trace_ ? Trace::now() : 0;
            const auto work = driver_->step(); require(work.kind != Kind::Idle, "departed idle boundary");
            for (unsigned i = 0; i < work.count; ++i) {   // rank 0: each session's spec counters, once a second
                const Key k = work.keys[i];
                if (!sessions_.counters_due(k) || driver_->state().slots()[k.slot].key != k) continue;
                if (const auto* st = driver_->stats_if_initialized(k)) sessions_.counters(k, *st);
            }
            if (trace_) {
                std::string ids;
                for (unsigned i = 0; i < work.count; ++i)
                    ids += (i ? "," : "") + std::to_string(driver_->state().slots()[work.keys[i].slot].req.id);
                step_ = {true, work.kind == Kind::Prefill ? "prefill" : work.kind == Kind::PromptTail ? "prompt tail" : "decode",
                         t0, targ("boundary", work.boundary) + "," + targ("first", work.first) + "," + targ("rows", work.rows) +
                                 ",\"requests\":[" + ids + "]"};
            }
            if (elog_ && work.kind != Kind::Decode)
                fprintf(stderr, "# s2_event step rank %u boundary %llu request %llu %s first %u rows %u\n", c_.rank,
                        (unsigned long long)work.boundary, (unsigned long long)driver_->state().slots()[work.keys[0].slot].req.id,
                        work.kind == Kind::Prefill ? "prefill" : work.kind == Kind::PrefillPack ? "prefill_pack" :
                        work.kind == Kind::PromptTailPack ? "tail_pack" : "tail", work.first, work.rows);
            dispatch_.synchronize_mode(driver_->state()); publish(driver_->state());
        }
    }
public:
    unsigned quit_linger_ms = 15000; // orderly-quit linger (monitor); > the time the workers take to see the head's quit
    Service(BootstrapConfig c, ControlLinks& links, NativeHooks hooks, int chunk)
        : c_(std::move(c)), links_(links), native_(std::move(hooks)),
          fn_(native_.module.ctx ? nullptr : new FnModule<NativeHooks>(native_)), mod_(fn_ ? fn_->ops() : native_.module),
          fp_(native_.pool.ops.ctx ? nullptr : new FnPool<PoolHooks>(native_.pool)), pol_(fp_ ? fp_->ops() : native_.pool.ops),
          sessions_(c_.scheduler,c_.directory,c_.rank == 0,chunk,c_.prefill_min),
          dispatch_(sessions_,c_.scheduler.max_rows,mod_.o), ingress_(c_.directory,c_.scheduler.epoch), queue_(c_.scheduler) {
        c_.validate();
        trace_.open(knob::s2_trace().value_or(""), c_.rank);
        require(native_.pool.pool, "the service needs the rank's pool");
        require(!native_.module.ctx || !(native_.bind || native_.bind_request || native_.grown || native_.retire || native_.fence ||
                                         native_.before_work || native_.drained || native_.resize || native_.shrink ||
                                         native_.flash_room || native_.pool.retain || native_.pool.park || native_.dispatch.launch ||
                                         native_.dispatch.bucket || native_.dispatch.mode || native_.dispatch.prefill ||
                                         native_.dispatch.logprob || native_.dispatch.dfl_ok || native_.dispatch.dfl_start ||
                                         native_.dispatch.dfl || native_.dispatch.draft_lp || native_.dispatch.prefill_seq ||
                                         native_.dispatch.prefill_pack_seq || native_.dispatch.gm_on || native_.dispatch.pen_rows),
                "a module table and module hooks together");
        const PoolHooks& ph = native_.pool;
        require(!ph.ops.ctx || !(ph.refusal || ph.r_rows || ph.make_room || ph.grow || ph.max_run_pages || ph.status || ph.match ||
                                 ph.take || ph.count || ph.tiers || ph.digest),
                "a pool table and pool hooks together");
        require(pol_.o.pool == pool().handle(), "the pool table is for another pool");
        const g6s2_module_ops& m = mod_.o;
        require(m.bind && m.retire && m.fence, "incomplete native service hooks");
        require(!pol_.o.grow || m.grown, "pool growth needs the native table hook");
        require(!pol_.o.make_room || m.shrink, "a shrink needs the native hook");
        require(!native_.share || m.resize, "a share report needs the resize hook");
        require(!native_.flash.chunk_rows || m.resize, "borrow-idle needs the resize hook");
        if (m.logprob) logprob_ = [this](int row, bool target, int k) { return mod_.logprob(row, target, k); };
        DriverHooks h;
        if (m.resize) h.resize = [&](const Resize& r) { mod_.resize(r); };
        h.shrink = [&](const Shrink& k, const Run& was, uint64_t length) {   // a worker's pool changed in the driver first
            require(mod_.o.shrink != nullptr, "a shrink commit without the native shrink hook");
            mod_.shrink(k, was, length);
        };
        h.bind = [&](const Grant& g) {
            require(staged_.has_value(), "binding without verified payload");
            require(!g.reuse || reuse(), "reused rows without prefix reuse");
            if (c_.rank) {   // the head's call, on the same pool state
                const auto worst = reserve_rows(c_.scheduler,g.req), rows = admit_rows(g.req);
                Run mine;
                if (pol_.o.count && reuse() && staged_->plp < 0)
                    require(pool().count(staged_->prompt, staged_->sys,
                        g6rt::steering::cache_namespace(g.req.steering)) == g.counted, "worker prefix counts differ from the head");
                else require(!g.counted, "a counted grant without counting");
                if (g.reuse) {
                    const PoolHit h = pol_.match(staged_->prompt, rows, worst, int(g.req.prompt_tokens) - 1,
                                                 g6rt::steering::cache_namespace(g.req.steering), g.req.r_rows);
                    require(h.skip == int(g.reuse) && (h.stored ? h.tier + 1 : 0) == g.restored, "worker prefix reuse differs from the head");
                    require(h.slot < 0 || h.slot == int(g.key.slot), "worker parked slot differs from the head's grant");
                    mine = pol_.take(h, rows, worst, g.req.r_rows);
                } else mine = pool_reserve(rows, worst, g.req.r_rows);
                require(mine == g.run, "worker page list differs from the head's grant");
            }
            dispatch_.begin(g.key.slot, int(g.reuse), g6rt::steering::cache_namespace(g.req.steering));
            if (mod_.o.bind_request) mod_.bind_request(g.key.slot, *staged_);
            mod_.bind(g); sessions_.bind(g,*staged_);
        };
        h.retire = [&](const Slot& old) {
            // a cancelled live session (finished ones: below)
            if (c_.rank && old.phase != Phase::Retiring && reuse()) retire_run(old);
            mod_.retire(old); sessions_.retire(old); dispatch_.release(old.key.slot);
        };
        h.prefill = [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows) {
            dispatch_.prefill(s,p,first,rows);
            sessions_.progress(s.key,first + rows);   // rank 0: a long prefill tells the front end it is alive
        };
        h.prefill_pack = [&](const std::vector<PrefillPart>& parts) {
            dispatch_.prefill_pack(parts);
            for (const auto& p : parts) sessions_.progress(p.slot->key, p.first + p.rows);
        };
        h.run = [&](const s2::Step& step) { return dispatch_.run(step); };
        h.emit = [&](Key k,const int* ids,int n,int row) {
            if (trace_ && n > 0) {
                const auto id = driver_->state().slots()[k.slot].req.id;
                if (first_.insert(id).second) trace_.instant("first token", Trace::STEPS, targ("request", id));
            }
            return sessions_.emit(k,ids,n,row,logprob_);
        };
        h.prompt = [&](Key k,int M,int pos0) { sessions_.prompt_rows(k,M,pos0,logprob_); };
        h.finished = [&](const Slot& s) {
            if (trace_) {
                trace_.instant("finish", Trace::STEPS, targ("request", s.req.id) + "," + targ("tokens", s.emitted));
                first_.erase(s.req.id);
            }
            if (reuse()) retire_run(s);
        };
        if (m.drained)   // the copies run at a bind, a growth or a retire: all before the next step's round
            h.settle = [&] {
                if (!mod_.drained()) return;
                // the barrier after a drain: every rank waits here for the slowest rank's drain (the drain trace's other
                // half: the tier io lines carry the same ranks' drains)
                auto wall = [] { return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); };
                const long long b0 = wall();
                endpoint_->sync(++sync_);
                fprintf(stderr, "# rank %u: s2 drain barrier %llu: in %lld ms, out %lld ms\n", (unsigned)c_.rank, (unsigned long long)sync_, b0, wall());
            };
        if (pol_.o.grow)
            h.grown = [&](const Slot& s, const Run& r) { mod_.grown(s.key.slot, r); sessions_.grow(s.key.slot, r); };
        h.terminal = [&](Key k,const s2::Stats& stats) {
            if (stats.switches || stats.mtp_rounds)
                fprintf(stderr,"# s2 drafter choice slot %u: %ld of %ld rounds on MTP, %ld switches, %ld tokens\n",unsigned(k.slot),
                        stats.mtp_rounds,stats.rounds,stats.switches,stats.tokens);
            if (native_.dfl_reuse.mode && !c_.rank)   // every request, so one that never had the drafter shows too
                fprintf(stderr,"# s2 dflash reuse slot %u: %ld of %ld rounds on DFlash2, %ld on MTP; context from %d (taps from %d), "
                        "warm-up to %d, promoted at %d, %ld tokens\n",unsigned(k.slot),stats.dfl_rounds,stats.rounds,stats.mtp_rounds,
                        stats.dfl_lo,stats.dfl_from,stats.dfl_warm,stats.promoted,stats.tokens);
            if (native_.ngram.t && !c_.rank)
                fprintf(stderr,"# s2 ngram slot %u: %ld of %ld rounds verified a table chain, %ld of %ld of its tokens accepted, %ld tokens\n",
                        unsigned(k.slot),stats.ngram_rounds,stats.rounds,stats.ngram_acc,stats.ngram_drafted,stats.tokens);
            sessions_.terminal(k,stats);
        };
        if (m.dfl) {
            require(m.dfl_ok && m.dfl_start, "DFlash2 needs dfl_ok, dfl_start and dfl");
            h.dfl_ok = [&](Key k) { return dispatch_.dfl_ok(k.slot); };
            h.dfl_start = [&](Key k) { return dispatch_.dfl_start(k.slot); };
            h.dfl = [&](Key, const s2::DflReq& r) { return dispatch_.dfl(r); };
            if (m.draft_lp) h.draft_lp = [&](int row) { return mod_.draft_lp(row); };
        }
        // MTP only (no drafter): the confidence stops of round_all read the draft log-probabilities too (g6m4 sets kstop)
        const bool kstop_mtp = !m.dfl && (native_.kstop.mode || native_.kstop.b.mode || native_.kstop.log);
        if (kstop_mtp && m.draft_lp) h.draft_lp = [&](int row) { return mod_.draft_lp(row); };
        const DriverPool dp{pool().handle(), pol_.o.grow, pol_.o.make_room != nullptr, !reuse(),
                            c_.rank == 0};
        driver_ = std::make_unique<Driver>(c_.scheduler,std::move(h),dp);
        if (m.dfl) {
            driver_->dflash(native_.dfl_kv, native_.dfl_spec); driver_->dsel(native_.dsel); driver_->kstop(native_.kstop);
            driver_->dfl_reuse(native_.dfl_reuse);
            driver_->ngram(native_.ngram);
        }
        else if (kstop_mtp) driver_->kstop(native_.kstop);
        require(!c_.scheduler.park || (reuse() && m.park), "parked slots need prefix reuse and the park hook");
        if (reuse()) {
            require(m.retain && pol_.o.take, "prefix reuse needs retain, match and take");
            dispatch_.track_rows();
        }
        if (!c_.rank) {
            leader_ = std::make_unique<LeaderChannel>(c_.scheduler.epoch,links_.peers()); endpoint_ = leader_.get();
        } else {
            require(links_.links.size() == 1, "worker needs one head link");
            worker_ = std::make_unique<WorkerChannel>(c_.scheduler.epoch,c_.rank,*links_.links[0]); endpoint_ = worker_.get();
            worker_->on_quit = [this] { atomic_file(c_.directory + "/quit", ""); };
        }
        if (m.flash_room) endpoint_->room = [this] { return mod_.flash_room(); };
        g6svc_control_cfg cc{};
        cc.scheduler = &c_.scheduler; cc.directory = ctl::bytes(c_.directory); cc.directory_n = path_n(c_.directory);
        cc.endpoint = endpoint_->handle(); cc.ingress = ingress_.handle(); cc.queue = queue_.handle(); cc.trace = trace_.handle();
        cc.slots = driver_->state().slot_data(); cc.n_slots = MAX_SESSIONS; cc.state = driver_->state().counters();
        cc.pool = pol_.o;
        cc.host.ctx = static_cast<ModuleCtx*>(&host_);
        cc.host.check = Host::check; cc.host.retire_run = Host::retire_run;
        if (native_.share) cc.host.share = Host::share;
        if (pol_.o.refusal) cc.host.refusal = Host::refusal;
        if (pol_.o.tiers) cc.host.tiers = Host::tiers_of;
        const FlashLend& fl = native_.flash;
        cc.lend = g6sch_lend{int64_t(fl.idle_ms), fl.keep, fl.chunk_rows, fl.chunks, false, false, 0};
        cc.share_ms = std::max<long>(1, knob::s2_share_ms());
        cc.rank = c_.rank; cc.head = c_.rank == 0;
        // GR8_S2_ADMIT_HOLD=1 (head only; the workers follow Commit::hold): an admission with another request queued
        // behind it and a slot left holds its boundary, and the next admission is announced at that same boundary, so
        // a burst of queued requests is granted in milliseconds rather than one per NOTICE_MIN boundaries (~15 s each
        // behind 8,192-row prefill quanta). Default off: one admission per notice
        cc.admit_hold = knob::s2_admit_hold();
        cc.elog = elog_;
        g6svc_control* ch = nullptr;
        const int32_t rc = g6svc_control_new(&cc, &ch);
        ctl_.reset(ch);
        ctl::check(rc);
    }
    // Blocking compute loop, with persistent control + abort/quit monitor threads.
    // Caller must use the runtime's all-rank failure path if this throws.
    void run() {
        std::thread control, watcher;
        auto guarded = [&](const std::function<void()>& f) {
            try { f(); } catch (...) { if (!stopping()) fail(std::current_exception()); }
        };
        try {
            control = std::thread([&] {
                guarded([&] { if (leader_) control_check(g6svc_control_run(ctl_.get())); else worker_->receive_events(); });
            });
            watcher = std::thread([&] { guarded([&] { control_check(g6svc_control_monitor(ctl_.get(), quit_linger_ms)); }); });
            if (!c_.rank) publish_ready(c_,links_);
            guarded([&] { compute(); });
        } catch (...) { fail(std::current_exception()); }
        // Orderly quit: compute leaves its loop as soon as the service is stopping, possibly mid-work (a straggler or a
        // descheduled head); the links stay open until the monitor has finished its bounded linger. On a
        // failure fail() has already woken/closed the endpoints and the monitor returns within one poll.
        if (watcher.joinable()) watcher.join();
        shutdown(); if (control.joinable()) control.join();
        if (failure_) {
            driver_->fail();
            try { if (!exists(c_.directory + "/abort.s2")) atomic_file(c_.directory + "/abort.s2", "native S2 epoch failed\n"); } catch (...) {}
            std::rethrow_exception(failure_);
        }
    }
};
} // namespace g6s2
