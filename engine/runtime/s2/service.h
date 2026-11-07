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
    // optional, head: why the engine cannot run this request at all (nullptr = it can); a refused request ends with an
    // error before it is queued, as a grammar that does not compile
    std::function<const char*(const g6rt::ServeReq&)> refuse;
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
// the Rust control and compute threads (engine/scheduler-rs src/service/control.rs, compute.rs)
struct ControlFree { void operator()(g6svc_control* c) const { g6svc_control_free(c); } };
using ControlHandle = std::unique_ptr<g6svc_control, ControlFree>;
struct ComputeFree { void operator()(g6svc_compute* c) const { g6svc_compute_free(c); } };
using ComputeHandle = std::unique_ptr<g6svc_compute, ComputeFree>;
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
    RunPool& pool() const { return *native_.pool.pool; }
    std::mutex failure_mu_; std::exception_ptr failure_;
    // GR8_S2_EVENTLOG=1: the event-latency diagnostic #61 needs. One "notice" line when the head anchors a notice (the
    // boundary it read and the future boundary it chose), one "commit" line when that notice lands on each rank, and one
    // "step" line per prefill quantum. Together they say whether a cancel was installed while a long prefill was still
    // running -- the lookahead binding -- or merely waited for it, which is the scheduler's admission. Logs only: no
    // state, no digest and no behaviour change, so it can run on a serving build.
    const bool elog_ = knob::s2_eventlog();
    // GR8_S2_TRACE=PATH: this rank's steps, and on the head the arrivals, waits and admissions, as a Chrome trace
    // (trace.h). A step's span runs from its launch to the next boundary.
    Trace trace_;
    // the Rust control thread and the state the threads share: the progress the compute thread publishes, a resize in
    // flight, and the stop flag. ctl_ outlives the threads (run joins them) and is freed before what it borrows
    ControlHandle ctl_;
    bool stopping() const { return g6svc_control_stopping(ctl_.get()); }
    void fail(std::exception_ptr error) {
        { std::lock_guard<std::mutex> lock(failure_mu_); if (!failure_) failure_ = error; }
        g6svc_control_shutdown(ctl_.get());
    }
    void shutdown() { g6svc_control_shutdown(ctl_.get()); }
    // What the Rust threads call in C++ (g6svc_host), one Host each, so each holds its own thread's exception. The
    // module's and the pool policy's exceptions, if one threw inside the Rust, stay in their own contexts and are
    // rethrown from there.
    struct Host : ModuleCtx {
        Service& s;
        std::vector<int> prompt;                  // control: the last checked request's, until the next check
        std::string why;                          // control: the pool policy's last refusal
        std::vector<g6kv_tier_totals> tiers;      // control: and its last tier counters
        std::optional<g6rt::ServeReq> staged;     // compute: the ADMIT's request, parsed before the event ACK
        explicit Host(Service& x) : s(x) {}
        static int32_t check(void* x, uint64_t id, const uint8_t* b, size_t n, g6svc_checked* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                auto request = parse_request(std::string(reinterpret_cast<const char*>(b), n), h.s.c_.scheduler.steering);
                out->ok = h.s.grammar_ok(id, request) && h.s.images_ok(id, request) && h.s.engine_ok(id, request);
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
        static int32_t idle_boundary(void* x, uint64_t b) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { h.s.driver_->idle_boundary(b); });
        }
        static int32_t stage(void* x, const uint8_t* b, size_t n, g6svc_staged* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                h.staged = parse_request(std::string(reinterpret_cast<const char*>(b), n), h.s.c_.scheduler.steering);
                const g6rt::ServeReq& r = *h.staged;
                *out = g6svc_staged{&r, r.prompt.data(), r.prompt.size(), r.sys, r.plp};
            });
        }
        static int32_t unstage(void* x) noexcept { return mod::guard<Host>(x, [&](Host& h) { h.staged.reset(); }); }
        static int32_t digest(void* x, bool has_pool, uint64_t pool, uint64_t* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                if (has_pool) h.s.driver_->pool_digest(pool);
                *out = h.s.driver_->state().digest();
            });
        }
        static int32_t abort(void* x, const g6sch_key* k) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                const Key& key = *mod::as<const Key>(k);
                const auto* stats = h.s.driver_->stats_if_initialized(key);
                h.s.sessions_.terminal(key, stats ? *stats : s2::Stats{}, "abort");
            });
        }
        static int32_t apply(void* x, const g6ctl_commit* c, bool admit) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                std::optional<Payload> payload;
                if (admit) payload = h.s.sessions_.payload(h.staged.value());
                h.s.driver_->apply(ctl::from_ctl(*c), payload ? &*payload : nullptr);
            });
        }
        static int32_t mode(void* x) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { h.s.dispatch_.synchronize_mode(h.s.driver_->state()); });
        }
        static int32_t step(void* x, g6sch_quantum* out) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { *out = sch::to_rs(h.s.driver_->step()); });
        }
        static int32_t counters(void* x, const g6sch_key* keys, uint32_t n) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                for (uint32_t i = 0; i < n; ++i) {
                    const Key k = *mod::as<const Key>(&keys[i]);
                    if (!h.s.sessions_.counters_due(k) || h.s.driver_->state().slots()[k.slot].key != k) continue;
                    if (const auto* st = h.s.driver_->stats_if_initialized(k)) h.s.sessions_.counters(k, *st);
                }
            });
        }
        static int32_t begin(void* x, const g6sch_grant* gr) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                const Grant& g = *mod::as<const Grant>(gr);
                h.s.dispatch_.begin(g.key.slot, int(g.reuse), g6rt::steering::cache_namespace(g.req.steering));
            });
        }
        static int32_t bound(void* x, const g6sch_grant* g) noexcept {
            return mod::guard<Host>(x, [&](Host& h) { h.s.sessions_.bind(*mod::as<const Grant>(g), h.staged.value()); });
        }
        static int32_t release(void* x, const g6sch_slot* sl) noexcept {
            return mod::guard<Host>(x, [&](Host& h) {
                const Slot& old = *mod::as<const Slot>(sl);
                h.s.sessions_.retire(old); h.s.dispatch_.release(old.key.slot);
            });
        }
    };
    Host host_{*this}, chost_{*this};   // the control thread's, the compute thread's
    // a control call's result: what the host or the pool policy threw (rc 1) rethrown, or the refusal (rc 2)
    void control_check(int32_t rc) {
        if (rc == 1 && !host_.err && pol_.o.ctx) {
            std::exception_ptr& e = static_cast<ModuleCtx*>(pol_.o.ctx)->err;
            if (e) { std::exception_ptr x = e; e = nullptr; std::rethrow_exception(x); }
        }
        ctl::check(rc, &host_.err);
    }
    // the Rust compute thread and the driver's hooks it answers. Freed before ctl_, which it borrows
    ComputeHandle cmp_;
    // a compute call's result: what the host, the module or the pool policy threw (rc 1) rethrown, or the refusal (rc 2)
    void compute_check(int32_t rc) {
        if (rc == 1 && !chost_.err)
            for (void* c : {mod_.o.ctx, pol_.o.ctx}) {
                if (!c) continue;
                std::exception_ptr& e = static_cast<ModuleCtx*>(c)->err;
                if (e) { std::exception_ptr x = e; e = nullptr; std::rethrow_exception(x); }
            }
        ctl::check(rc, &chost_.err);
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
    bool engine_ok(uint64_t id, const g6rt::ServeReq& r) {
        const char* why = native_.refuse ? native_.refuse(r) : nullptr;
        if (!why) return true;
        fprintf(stderr, "# s2: request %llu refused: %s\n", (unsigned long long)id, why);
        atomic_file(c_.directory + "/resp." + std::to_string(id), "end error -1 0\n");
        return false;
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
        // the hooks whose logic is the service's are Rust (compute.rs); the rest forward to Dispatch and Sessions
        DriverHooks h;
        if (m.resize) h.resize = [&](const Resize& r) { mod_.resize(r); };
        h.shrink = [&](const Shrink& k, const Run& was, uint64_t length) {   // a worker's pool changed in the driver first
            require(mod_.o.shrink != nullptr, "a shrink commit without the native shrink hook");
            mod_.shrink(k, was, length);
        };
        h.bind = [&](const Grant& g) { compute_check(g6svc_compute_bind(cmp_.get(), sch::c<g6sch_grant>(&g))); };
        h.retire = [&](const Slot& old) { compute_check(g6svc_compute_retire(cmp_.get(), sch::c<g6sch_slot>(&old))); };
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
            if (trace_ && n > 0) g6svc_compute_first_token(cmp_.get(), sch::c<g6sch_key>(&k));
            return sessions_.emit(k,ids,n,row,logprob_);
        };
        h.prompt = [&](Key k,int M,int pos0) { sessions_.prompt_rows(k,M,pos0,logprob_); };
        h.finished = [&](const Slot& s) { compute_check(g6svc_compute_finished(cmp_.get(), sch::c<g6sch_slot>(&s))); };
        if (m.drained)   // the copies run at a bind, a growth or a retire: all before the next step's round
            h.settle = [&] { compute_check(g6svc_compute_settle(cmp_.get())); };
        if (pol_.o.grow)
            h.grown = [&](const Slot& s, const Run& r) { mod_.grown(s.key.slot, r); sessions_.grow(s.key.slot, r); };
        h.terminal = [&](Key k,const s2::Stats& st) {
            const g6s2_stats x{st.rounds, st.tokens, st.drafted, st.mtp_rounds, st.switches, st.inserted, st.ngram_rounds,
                               st.ngram_drafted, st.ngram_acc, st.dfl_rounds, st.dfl_from, st.dfl_lo, st.dfl_warm, st.promoted,
                               nullptr, 0};
            g6svc_compute_terminal(cmp_.get(), k.slot, &x);
            sessions_.terminal(k,st);
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
        g6svc_compute_cfg pc{};
        pc.control = ctl_.get(); pc.endpoint = endpoint_->handle(); pc.ingress = ingress_.handle(); pc.trace = trace_.handle();
        pc.slots = driver_->state().slot_data(); pc.n_slots = MAX_SESSIONS; pc.state = driver_->state().counters();
        pc.module = mod_.o;
        g6svc_host& ct = pc.host;
        ct.ctx = static_cast<ModuleCtx*>(&chost_);
        ct.retire_run = Host::retire_run; ct.idle_boundary = Host::idle_boundary; ct.stage = Host::stage; ct.unstage = Host::unstage;
        ct.digest = Host::digest; ct.abort = Host::abort; ct.apply = Host::apply; ct.mode = Host::mode; ct.step = Host::step;
        ct.counters = Host::counters; ct.begin = Host::begin; ct.bound = Host::bound; ct.release = Host::release;
        pc.rank = c_.rank; pc.elog = elog_;
        pc.dfl_reuse = native_.dfl_reuse.mode != 0; pc.ngram = native_.ngram.t != nullptr;
        g6svc_compute* mh = nullptr;
        const int32_t mrc = g6svc_compute_new(&pc, &mh);
        cmp_.reset(mh);
        ctl::check(mrc);
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
            guarded([&] { compute_check(g6svc_compute_run(cmp_.get())); });
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
