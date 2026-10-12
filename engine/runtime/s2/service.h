// Host orchestration for g6m4's OPT-IN S2 adapter. NativeHooks must be backed by
// the real TP4 pool and fc2 CUDA APIs; test pools never enable native readiness.
#pragma once
#include "bootstrap.h"
#include "control_channel.h"
#include "dispatch.h"
#include "pool.h"
#include <map>
#include <thread>

namespace g6s2 {
// rows: what a request needs at admission (grant_rows), worst: all it may grow to (reserve_rows)
struct PoolHooks {
    std::function<uint32_t()> free_pages, room;
    // head after ALL event ACKs, workers at the grant (the same call on the same pool state must give the same Run)
    std::function<Run(uint64_t rows, uint64_t worst)> reserve;
    std::function<void(const Run&)> release;
    // can the FIFO head be admitted now: room, no exclusive run in the way, and every live run still able to reach its
    // worst case (pool.h fits). false is counted as an admission stall
    std::function<bool(uint64_t rows, uint64_t worst)> fits;
    // optional (growth on demand, pool.h grow): before a round, every rank; nothing = the session sits the round out
    std::function<std::optional<Run>(const Run&, uint64_t rows)> grow;
    // optional, status.s2: each live run's pages now and at worst, granted growth and stalled rounds
    std::function<std::vector<PoolUse>()> runs;
    std::function<g6kv_totals()> stats;
    // optional, prefix reuse (pool.h keep): every rank retires a run with its session's reuse state; the head matches a
    // new prompt against the retained runs (limit: rows its prefill may skip) and takes a hit
    std::function<void(const Run&, const g6rt::PrefixReuse&)> retain;
    std::function<PoolHit(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit)> match;
    std::function<Run(const PoolHit&, uint64_t rows, uint64_t worst)> take;
    // optional, block stores (pool.h add_tier): counting toward shared prefixes at admission (head and workers, before
    // the grant), and each tier's counters for status.s2
    std::function<uint32_t(const std::vector<int>& prompt)> count;
    std::function<std::vector<g6kv_tier_totals>()> tiers;
    // optional: the pool's state, compared across ranks at every event (Scheduler::pool_digest)
    std::function<uint64_t()> digest;
};
struct NativeHooks {
    PoolHooks pool;
    std::function<void(const Grant&)> bind; // native main AND draft binding, no pool mutation
    // with pool.grow: slot's run grew; its new pages go into the main and draft tables before the round
    std::function<void(unsigned slot, const Run&)> grown;
    std::function<void(const Slot&)> retire; // native per-slot graph/view cleanup only
    std::function<void()> fence; // all native streams/drafts quiescent before event ACK
    // optional: did this rank run tier copies since the last call (the same answer on every rank: the copies are). Then
    // the ranks line up before the next round: one rank's copies can take seconds longer than another's, and a GPU wait
    // for a peer's rows gives up after ~10 s (G6_SPIN_TIMEOUT_CYC)
    std::function<bool()> drained;
    DispatchHooks dispatch;
    // DFlash2 (dispatch.dfl set): drafts a round verifies and the --spec-adaptive policy (s2_batch.h Batch::dflash)
    int dfl_kv = 7;
    g6rt::SpecLen dfl_spec;
    s2::DSel dsel;                    // drafter choice (G6_S2_DSEL)
};
// Only atomic scalar stores/loads on the compute hot path. No request-file
// checks, digest, allocation, or lock in an event-free boundary.
struct Progress {
    std::atomic<uint64_t> boundary{0}, revision{0};
    std::atomic<unsigned> resident{0}, retiring{0};
    void publish(const Scheduler& s) {
        boundary.store(s.boundary()); resident.store(s.resident()); retiring.store(s.retiring()); revision.store(s.revision());
    }
};
class Service {
    BootstrapConfig c_; ControlLinks& links_; NativeHooks native_;
    Sessions sessions_; Dispatch dispatch_; std::unique_ptr<Driver> driver_;
    std::unique_ptr<LeaderChannel> leader_; std::unique_ptr<WorkerChannel> worker_;
    EventEndpoint* endpoint_ = nullptr;
    Ingress ingress_; AdmissionQueue queue_;
    std::map<uint64_t, Command> commands_;
    std::map<uint64_t, std::string> payloads_;   // head: queued requests' bytes, sent to the workers with each ADMIT notice
    std::map<uint64_t, std::vector<int>> prompts_;   // head, prefix reuse: queued requests' prompts (none for prompt logprobs)
    bool reuse() const { return bool(native_.pool.match); }
    // rows a request's run holds at admission: grant_rows when runs grow on demand, else its worst case
    uint64_t admit_rows(const Request& r) const { return native_.pool.grow ? grant_rows(c_.scheduler, r) : reserve_rows(c_.scheduler, r); }
    std::optional<g6rt::ServeReq> staged_; // COMPUTE thread only, parsed before event ACK
    Progress progress_;
    std::atomic<bool> stopping_{false};
    std::mutex failure_mu_; std::exception_ptr failure_;
    uint64_t attempted_head_ = UINT64_MAX, attempted_revision_ = UINT64_MAX;
    uint64_t event_ = 1;
    uint64_t sync_ = 0;   // compute thread: barriers passed (native_.drained)
    static constexpr uint64_t LOOKAHEAD = 8; // future safe boundary; late notices remain fatal
    void fail(std::exception_ptr error) {
        { std::lock_guard<std::mutex> lock(failure_mu_); if (!failure_) failure_ = error; }
        stopping_.store(true); endpoint_->fail();
    }
    void shutdown() { stopping_.store(true); endpoint_->fail(); }
    // a slot leaves: its run back to the pool, retained with the session's reuse state when prefix reuse is on. A
    // finished session at the step it finished (Driver finished hook, every rank), a cancelled one at the event: head in
    // event(), workers at apply, the same point of the same commit. So every rank keeps the same runs.
    void retire_run(const Slot& s) {
        if (!reuse()) { native_.pool.release(s.run); return; }
        g6rt::PrefixReuse st = dispatch_.rows(s.key.slot);
        const auto gen = driver_->emitted(s.key.slot);
        st.finish(driver_->prompt(s.key.slot), gen, int(gen.size()), true);
        native_.pool.retain(s.run, st);
    }
    void publish_status() { // head CONTROL thread only, on queue/lifecycle revisions
        const auto free = native_.pool.free_pages(), room = native_.pool.room();
        require(room <= free && free <= c_.scheduler.pool_rows / PAGE, "invalid pool status");
        std::ostringstream o;
        o << "{\"epoch\":" << c_.scheduler.epoch << ",\"resident\":" << progress_.resident.load()
          << ",\"queued\":" << queue_.size() << ",\"retiring\":" << progress_.retiring.load()
          << ",\"reserved_pages\":" << c_.scheduler.pool_rows / PAGE - free << ",\"free_pages\":" << free
          << ",\"largest_free_run\":" << room << ",\"admission_stalls\":" << queue_.stalls;
        // growth on demand: pages the live runs hold and could grow to, each run's pair, and rounds a session sat out
        if (native_.pool.stats) {
            const auto st = native_.pool.stats();
            o << ",\"held_pages\":" << st.held_pages << ",\"worst_pages\":" << st.worst_pages << ",\"grown\":" << st.grows
              << ",\"stalled_rounds\":" << st.stalls;
        }
        if (native_.pool.tiers) {
            o << ",\"tiers\":[";
            const auto ts = native_.pool.tiers();
            for (size_t i = 0; i < ts.size(); ++i) {
                const auto& t = ts[i];
                o << (i ? "," : "") << "{\"slots\":" << t.slots << ",\"free_slots\":" << t.free_slots << ",\"entries\":" << t.entries
                  << ",\"shared\":" << t.shared << ",\"spilled_pages\":" << t.spilled_pages << ",\"moved_in_pages\":" << t.moved_in_pages
                  << ",\"restored_pages\":" << t.restored_pages << ",\"hits\":" << t.hits << ",\"dropped\":" << t.dropped << "}";
            }
            o << "]";
        }
        if (native_.pool.runs) {
            o << ",\"runs\":[";
            const auto runs = native_.pool.runs();
            for (size_t i = 0; i < runs.size(); ++i)
                o << (i ? "," : "") << "{\"held\":" << runs[i].held << ",\"worst\":" << runs[i].worst << "}";
            o << "]";
        }
        o << "}\n";
        atomic_file(c_.directory + "/status.s2", o.str());
    }
    void event(const Command& command) { // CONTROL thread: never read Driver until ALL ACKs
        const auto boundary = progress_.boundary.load();
        require(boundary <= uint64_t(INT64_MAX) - LOOKAHEAD, "event boundary overflow");
        Notice n{c_.scheduler.epoch, event_, boundary + LOOKAHEAD, command.request, command.hash,
                 command.admit ? EventKind::Admit : EventKind::Cancel};
        leader_->event(n, [&](uint64_t hash) {
            const auto& s = driver_->state(); // compute paused in boundary() until this commit
            Commit cm{n.epoch, n.event, n.boundary, hash, {}, {}};
            if (!command.admit) cm.cancel = s.find(command.request);
            // Head owns the real pool delta once; Driver callbacks skip it here. A Retiring slot gave its run back when it
            // finished (Driver finished hook); a cancelled live one gives it back now.
            for (const auto& sl : s.slots())
                if (cm.cancel && sl.phase != Phase::Free && sl.phase != Phase::Retiring && sl.key == *cm.cancel) retire_run(sl);
            if (command.admit) {
                const auto* request = queue_.front(); require(request && request->id == command.request, "admission FIFO changed during event");
                unsigned live = s.resident();
                const auto worst = reserve_rows(c_.scheduler, *request), rows = admit_rows(*request);
                const auto fit = queue_.check(s.revision(), live < c_.scheduler.limit(), native_.pool.fits(rows, worst), native_.pool.room());
                require(fit.has_value(), "duplicate admission attempt without revision");
                if (fit->why == Stall::None) {
                    Run run{}; uint32_t skip = 0, restored = 0, counted = 0;
                    const auto pr = prompts_.find(request->id);
                    if (reuse() && pr != prompts_.end()) {
                        if (native_.pool.count) counted = native_.pool.count(pr->second);
                        const PoolHit h = native_.pool.match(pr->second, rows, worst, int(request->prompt_tokens) - 1);
                        if (h.skip > 0) { run = native_.pool.take(h, rows, worst); skip = uint32_t(h.skip); restored = h.stored ? h.tier + 1 : 0; }
                        fprintf(stderr, "# s2 reuse request %llu %s retained %d lcp %d prompt %u first page %u\n", (unsigned long long)request->id,
                                h.why, h.skip, h.lcp, request->prompt_tokens, skip ? run.first_page : 0u);
                    }
                    if (!skip) run = native_.pool.reserve(rows, worst);
                    cm.admit = Grant{s.next_key(), *request, run, skip, restored, counted};
                }
                attempted_head_ = request->id; attempted_revision_ = s.revision() + 1;
                if (cm.admit) { queue_.admitted(request->id); commands_.erase(command.request); prompts_.erase(command.request); payloads_.erase(command.request); }
            }
            return cm;
        }, command.admit ? &payloads_.at(command.request) : nullptr);
        require(event_ != UINT64_MAX, "event sequence exhausted"); ++event_; publish_status();
    }
    void controller() {
        uint64_t status_revision = UINT64_MAX;
        while (!stopping_.load()) {
            bool changed = false;
            while (auto command = ingress_.next()) {
                changed = true;
                if (command->admit) {
                    auto bytes = ingress_.request(*command);
                    auto request = parse_request(bytes);
                    queue_.push(Sessions::request(command->request, request)); commands_.emplace(command->request, *command);
                    payloads_.emplace(command->request, std::move(bytes));
                    if (reuse() && request.plp < 0) prompts_.emplace(command->request, std::move(request.prompt));
                } else if (queue_.cancel(command->request)) {
                    commands_.erase(command->request); prompts_.erase(command->request); payloads_.erase(command->request);
                    atomic_file(c_.directory + "/resp." + std::to_string(command->request), "end abort -1 0\n");
                } else event(*command); // resident/retiring lookup only at the fence
                if (stopping_.load()) return;
            }
            if (const auto* front = queue_.front()) {
                if (front->id != attempted_head_ || progress_.revision.load() != attempted_revision_) {
                    // Copy: event() may erase the map entry after all ACKs.
                    auto command = commands_.at(front->id); event(command); changed = true;
                }
            }
            const auto rev = progress_.revision.load();
            if (changed || rev != status_revision) { publish_status(); status_revision = rev; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // control only
        }
    }
    void monitor() {
        while (!stopping_.load()) {
            require(!exists(c_.directory + "/abort.s2"), "shim aborted S2 epoch");
            if (exists(c_.directory + "/quit")) {
                // Orderly quit: stop at once (controller/compute loops end, later control errors are ignored) but keep
                // this rank's control sockets open for quit_linger_ms before closing them, so a worker reads EOF only
                // after it is stopping too. The head passes its quit to the workers over the control links, and each
                // worker writes its own DIR/quit. Closing at once made the workers fail with 'control disconnected'
                // (g6m4 exit 3) on a clean quit.
                // Quit is for an idle engine (no resident session): a rank stopping mid-round leaves its peers' step.
                if (leader_) leader_->quit();
                stopping_.store(true);
                const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(quit_linger_ms);
                while (std::chrono::steady_clock::now() < end && !exists(c_.directory + "/abort.s2"))
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                shutdown(); return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    void compute() {
        while (!stopping_.load()) {
            if (!driver_->state().resident()) {
                const auto n = endpoint_->wait_notice(); driver_->idle_boundary(n.boundary);
            }
            auto cm = endpoint_->boundary(driver_->state().boundary(), [&] {
                native_.fence(); const auto n = endpoint_->wait_notice(); staged_.reset();
                if (n.kind == EventKind::Admit) {   // the head reads its own req.<id>; a worker has the bytes from the head
                    const auto bytes = worker_ ? worker_->request(n.request) : ingress_.request({true,n.epoch,n.request,n.payload_hash});
                    require(payload_hash(bytes) == n.payload_hash, "request bytes do not match the ADMIT notice");
                    staged_ = parse_request(bytes);
                }
                if (native_.pool.digest) driver_->pool_digest(native_.pool.digest());
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
                if (worker_ && cm->admit) worker_->admitted(cm->admit->req.id);
                dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
                staged_.reset(); endpoint_->applied(); continue;
            }
            const auto work = driver_->step(); require(work.kind != Kind::Idle, "departed idle boundary");
            dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
        }
    }
public:
    unsigned quit_linger_ms = 15000; // orderly-quit linger (monitor); > the time the workers take to see the head's quit
    Service(BootstrapConfig c, ControlLinks& links, NativeHooks hooks, int chunk)
        : c_(std::move(c)), links_(links), native_(std::move(hooks)), sessions_(c_.scheduler,c_.directory,c_.rank == 0,chunk),
          dispatch_(sessions_,c_.scheduler.max_rows,native_.dispatch), ingress_(c_.directory,c_.scheduler.epoch), queue_(c_.scheduler) {
        c_.validate();
        require(bool(native_.pool.free_pages) && bool(native_.pool.room) && bool(native_.pool.reserve) && bool(native_.pool.fits) &&
                bool(native_.pool.release) && bool(native_.bind) && bool(native_.retire) && bool(native_.fence), "incomplete native service hooks");
        require(!native_.pool.grow || bool(native_.grown), "pool growth needs the native table hook");
        DriverHooks h;
        h.bind = [&](const Grant& g) {
            require(staged_.has_value(), "binding without verified payload");
            require(!g.reuse || reuse(), "reused rows without prefix reuse");
            if (c_.rank) {   // the head's call, on the same pool state
                const auto worst = reserve_rows(c_.scheduler,g.req), rows = admit_rows(g.req);
                Run mine;
                if (native_.pool.count && reuse() && staged_->plp < 0)
                    require(native_.pool.count(staged_->prompt) == g.counted, "worker prefix counts differ from the head");
                else require(!g.counted, "a counted grant without counting");
                if (g.reuse) {
                    const PoolHit h = native_.pool.match(staged_->prompt, rows, worst, int(g.req.prompt_tokens) - 1);
                    require(h.skip == int(g.reuse) && (h.stored ? h.tier + 1 : 0) == g.restored, "worker prefix reuse differs from the head");
                    mine = native_.pool.take(h, rows, worst);
                } else mine = native_.pool.reserve(rows, worst);
                require(mine == g.run, "worker page list differs from the head's grant");
            }
            dispatch_.begin(g.key.slot, int(g.reuse));
            native_.bind(g); sessions_.bind(g,*staged_);
        };
        h.retire = [&](const Slot& old) {
            if (c_.rank && old.phase != Phase::Retiring) retire_run(old);   // a cancelled live session (finished ones: below)
            native_.retire(old); sessions_.retire(old); dispatch_.release(old.key.slot);
        };
        h.prefill = [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows) { dispatch_.prefill(s,p,first,rows); };
        h.run = [&](const s2::Step& step) { return dispatch_.run(step); };
        h.emit = [&](Key k,const int* ids,int n,int row) { return sessions_.emit(k,ids,n,row,native_.dispatch.logprob); };
        h.prompt = [&](Key k,int M,int pos0) { sessions_.prompt_rows(k,M,pos0,native_.dispatch.logprob); };
        h.finished = [&](const Slot& s) { retire_run(s); };
        if (native_.drained)   // the copies run at a bind, a growth or a retire: all before the next step's round
            h.settle = [&] { if (native_.drained()) endpoint_->sync(++sync_); };
        if (native_.pool.grow)
            h.grow = [&](const Slot& s, uint64_t rows) {
                const auto r = native_.pool.grow(s.run, rows);
                if (r && *r != s.run) { native_.grown(s.key.slot, *r); sessions_.grow(s.key.slot, *r); }
                return r;
            };
        h.terminal = [&](Key k,const s2::Stats& stats) {
            if (stats.switches || stats.mtp_rounds)
                fprintf(stderr,"# s2 drafter choice slot %u: %ld of %ld rounds on MTP, %ld switches, %ld tokens\n",unsigned(k.slot),
                        stats.mtp_rounds,stats.rounds,stats.switches,stats.tokens);
            sessions_.terminal(k,stats);
        };
        if (native_.dispatch.dfl) {
            require(bool(native_.dispatch.dfl_ok) && bool(native_.dispatch.dfl_start), "DFlash2 needs dfl_ok, dfl_start and dfl");
            h.dfl_ok = [&](Key k) { return dispatch_.dfl_ok(k.slot); };
            h.dfl_start = [&](Key k) { return dispatch_.dfl_start(k.slot); };
            h.dfl = [&](Key, const s2::DflReq& r) { return dispatch_.dfl(r); };
        }
        driver_ = std::make_unique<Driver>(c_.scheduler,std::move(h));
        if (native_.dispatch.dfl) { driver_->dflash(native_.dfl_kv, native_.dfl_spec); driver_->dsel(native_.dsel); }
        if (reuse()) {
            require(bool(native_.pool.retain) && bool(native_.pool.take), "prefix reuse needs retain, match and take");
            dispatch_.track_rows();
        }
        if (!c_.rank) {
            leader_ = std::make_unique<LeaderChannel>(c_.scheduler.epoch,links_.peers()); endpoint_ = leader_.get();
        } else {
            require(links_.links.size() == 1, "worker needs one head link");
            worker_ = std::make_unique<WorkerChannel>(c_.scheduler.epoch,c_.rank,*links_.links[0]); endpoint_ = worker_.get();
            worker_->on_quit = [this] { atomic_file(c_.directory + "/quit", ""); };
        }
    }
    // Blocking compute loop, with persistent control + abort/quit monitor threads.
    // Caller must use the runtime's all-rank failure path if this throws.
    void run() {
        std::thread control, watcher;
        auto guarded = [&](const std::function<void()>& f) {
            try { f(); } catch (...) { if (!stopping_.load()) fail(std::current_exception()); }
        };
        try {
            control = std::thread([&] { guarded([&] { if (leader_) controller(); else worker_->receive_events(); }); });
            watcher = std::thread([&] { guarded([&] { monitor(); }); });
            if (!c_.rank) publish_ready(c_,links_);
            guarded([&] { compute(); });
        } catch (...) { fail(std::current_exception()); }
        // Orderly quit: compute leaves its loop as soon as stopping_ is set, possibly mid-work (a straggler or a
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
