// Host orchestration for g6m4's OPT-IN S2 adapter. NativeHooks must be backed by
// the real TP4 pool and fc2 CUDA APIs; test pools never enable native readiness.
#pragma once
#include "bootstrap.h"
#include "control_channel.h"
#include "dispatch.h"
#include <map>
#include <thread>

namespace g6s2 {
struct PoolHooks {
    std::function<uint32_t()> free_pages, largest_run;
    std::function<Run(uint64_t rows)> reserve; // head only, after ALL event ACKs
    std::function<void(const Run&, uint64_t rows)> claim; // workers: exact head grant
    std::function<void(const Run&)> release;
    // optional (S2 below R, pool.h): false = the FIFO head may not take a run yet even though one fits (a past-R
    // request waits for an empty pool); counted as an admission stall
    std::function<bool(uint64_t rows)> admissible;
};
struct NativeHooks {
    PoolHooks pool;
    std::function<void(const Grant&)> bind; // native main AND draft binding, no pool mutation
    std::function<void(const Slot&)> retire; // native per-slot graph/view cleanup only
    std::function<void()> fence; // all native streams/drafts quiescent before event ACK
    DispatchHooks dispatch;
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
    std::optional<g6rt::ServeReq> staged_; // COMPUTE thread only, parsed before event ACK
    Progress progress_;
    std::atomic<bool> stopping_{false};
    std::mutex failure_mu_; std::exception_ptr failure_;
    uint64_t attempted_head_ = UINT64_MAX, attempted_revision_ = UINT64_MAX;
    uint64_t event_ = 1;
    static constexpr uint64_t LOOKAHEAD = 8; // future safe boundary; late notices remain fatal
    void fail(std::exception_ptr error) {
        { std::lock_guard<std::mutex> lock(failure_mu_); if (!failure_) failure_ = error; }
        stopping_.store(true); endpoint_->fail();
    }
    void shutdown() { stopping_.store(true); endpoint_->fail(); }
    void publish_status() { // head CONTROL thread only, on queue/lifecycle revisions
        const auto free = native_.pool.free_pages(), largest = native_.pool.largest_run();
        require(largest <= free && free <= c_.scheduler.pool_rows / PAGE, "invalid pool status");
        std::ostringstream o;
        o << "{\"epoch\":" << c_.scheduler.epoch << ",\"resident\":" << progress_.resident.load()
          << ",\"queued\":" << queue_.size() << ",\"retiring\":" << progress_.retiring.load()
          << ",\"reserved_pages\":" << c_.scheduler.pool_rows / PAGE - free << ",\"free_pages\":" << free
          << ",\"largest_free_run\":" << largest << ",\"admission_stalls\":" << queue_.stalls
          << ",\"fragmentation_stalls\":" << queue_.fragmentation_stalls << "}\n";
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
            // Head owns the real pool delta once; Driver callbacks skip it here.
            for (const auto& run : s.reclaimable(cm.cancel)) native_.pool.release(run);
            if (command.admit) {
                const auto* request = queue_.front(); require(request && request->id == command.request, "admission FIFO changed during event");
                unsigned live = s.resident();
                const auto fit = queue_.check(s.revision(), live < c_.scheduler.limit(), native_.pool.free_pages(), native_.pool.largest_run());
                require(fit.has_value(), "duplicate admission attempt without revision");
                const auto rows = reserve_rows(c_.scheduler, *request);
                const bool ok = fit->why == Stall::None && (!native_.pool.admissible || native_.pool.admissible(rows));
                if (fit->why == Stall::None && !ok) ++queue_.stalls;
                if (ok) {
                    Run run = native_.pool.reserve(rows);
                    cm.admit = Grant{s.next_key(), *request, run};
                }
                attempted_head_ = request->id; attempted_revision_ = s.revision() + 1;
                if (cm.admit) { queue_.admitted(request->id); commands_.erase(command.request); }
            }
            return cm;
        });
        require(event_ != UINT64_MAX, "event sequence exhausted"); ++event_; publish_status();
    }
    void controller() {
        uint64_t status_revision = UINT64_MAX;
        while (!stopping_.load()) {
            bool changed = false;
            while (auto command = ingress_.next()) {
                changed = true;
                if (command->admit) {
                    auto request = parse_request(ingress_.request(*command));
                    queue_.push(Sessions::request(command->request, request)); commands_.emplace(command->request, *command);
                } else if (queue_.cancel(command->request)) {
                    commands_.erase(command->request);
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
                // this rank's control sockets open for quit_linger_ms before closing them, so a peer whose DIR/quit lands
                // later (sequential ssh touches) reads EOF only after it is stopping too. Closing at once made the
                // other ranks fail with 'control disconnected' -> g6m4 exit 3 on a clean quit (P5 CPU repro).
                // Quit is for an idle engine (no resident session): a rank stopping mid-round leaves its peers' step.
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
                if (n.kind == EventKind::Admit) staged_ = parse_request(ingress_.request({true,n.epoch,n.request,n.payload_hash}));
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
                dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
                staged_.reset(); endpoint_->applied(); continue;
            }
            const auto work = driver_->step(); require(work.kind != Kind::Idle, "departed idle boundary");
            dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
        }
    }
public:
    unsigned quit_linger_ms = 15000; // orderly-quit linger (monitor); > the spread of the quit touches over all ranks
    Service(BootstrapConfig c, ControlLinks& links, NativeHooks hooks, int chunk)
        : c_(std::move(c)), links_(links), native_(std::move(hooks)), sessions_(c_.scheduler,c_.directory,c_.rank == 0,chunk),
          dispatch_(sessions_,c_.scheduler.max_rows,native_.dispatch), ingress_(c_.directory,c_.scheduler.epoch), queue_(c_.scheduler) {
        c_.validate();
        require(bool(native_.pool.free_pages) && bool(native_.pool.largest_run) && bool(native_.pool.reserve) &&
                bool(native_.pool.claim) && bool(native_.pool.release) && bool(native_.bind) && bool(native_.retire) && bool(native_.fence), "incomplete native service hooks");
        DriverHooks h;
        h.bind = [&](const Grant& g) {
            require(staged_.has_value(), "binding without verified payload");
            if (c_.rank) native_.pool.claim(g.run, reserve_rows(c_.scheduler,g.req));
            native_.bind(g); sessions_.bind(g,*staged_);
        };
        h.retire = [&](const Slot& old) {
            native_.retire(old); sessions_.retire(old); dispatch_.release(old.key.slot);
            if (c_.rank) native_.pool.release(old.run);
        };
        h.prefill = [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows) { dispatch_.prefill(s,p,first,rows); };
        h.run = [&](const s2::Step& step) { return dispatch_.run(step); };
        h.emit = [&](Key k,const int* ids,int n,int row) { return sessions_.emit(k,ids,n,row,native_.dispatch.logprob); };
        h.prompt = [&](Key k,int M,int pos0) { sessions_.prompt_rows(k,M,pos0,native_.dispatch.logprob); };
        h.terminal = [&](Key k,const s2::Stats& stats) { sessions_.terminal(k,stats); };
        driver_ = std::make_unique<Driver>(c_.scheduler,std::move(h));
        if (!c_.rank) {
            leader_ = std::make_unique<LeaderChannel>(c_.scheduler.epoch,links_.peers()); endpoint_ = leader_.get();
        } else {
            require(links_.links.size() == 1, "worker needs one head link");
            worker_ = std::make_unique<WorkerChannel>(c_.scheduler.epoch,c_.rank,*links_.links[0]); endpoint_ = worker_.get();
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
