// Host orchestration for g6m4's OPT-IN S2 adapter. NativeHooks must be backed by
// the real TP4 pool and fc2 CUDA APIs; test pools never enable native readiness.
#pragma once
#include "bootstrap.h"
#include "control_channel.h"
#include "dispatch.h"
#include "flash_lend.h"
#include "pool.h"
#include "trace.h"
#include "../../knobs-rs/gr8_knobs.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <thread>

namespace g6s2 {
// rows: what a request needs at admission (grant_rows), worst: all it may grow to (reserve_rows)
struct PoolHooks {
    std::function<uint32_t()> free_pages, room;
    // head after ALL event ACKs, workers at the grant (the same call on the same pool state must give the same Run)
    std::function<Run(uint64_t rows, uint64_t worst)> reserve;
    std::function<void(const Run&)> release;
    // can the FIFO head be admitted now: room, no exclusive run in the way (on the page classes a split run past R
    // shares the pool, kvman-rs pool.rs), and every live run still able to reach its worst case (pool.h fits). false is
    // counted as an admission stall
    std::function<bool(uint64_t rows, uint64_t worst)> fits;
    // optional (remote rows release): why a stalled request past R does not fit its page classes; empty = no reason
    std::function<std::string(uint64_t worst)> refusal;
    // optional (growth on demand, pool.h grow): before a round, every rank; nothing = the session sits the round out
    std::function<std::optional<Run>(const Run&, uint64_t rows)> grow;
    // optional, status.s2: each live run's pages now and at worst, granted growth and stalled rounds
    std::function<std::vector<PoolUse>()> runs;
    std::function<g6kv_totals()> stats;
    // optional, prefix reuse (pool.h keep): every rank retires a run with its session's reuse state; the head matches a
    // new prompt against the retained runs (limit: rows its prefill may skip) and takes a hit
    std::function<void(const Run&, const g6rt::PrefixReuse&, const g6rt::CacheNamespace&)> retain;
    std::function<PoolHit(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit,
                          const g6rt::CacheNamespace&)> match;
    std::function<Run(const PoolHit&, uint64_t rows, uint64_t worst)> take;
    // optional, block stores (pool.h add_tier): counting toward shared prefixes at admission (head and workers, before
    // the grant, including nonstock calls that clear pending stock metadata; span: the request's system span, -1 = none), and each tier's counters for status.s2
    std::function<uint32_t(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace&)> count;
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
// Only atomic scalar stores/loads on the compute hot path. No request-file
// checks, digest, allocation, or lock in an event-free boundary. horizon() is
// a bounded six-step projection over the eight slots: no allocation, no lock,
// a few hundred nanoseconds against a >= 20 ms round.
struct Progress {
    std::atomic<uint64_t> boundary{0}, revision{0}, horizon{0};
    std::atomic<unsigned> resident{0}, retiring{0};
    void publish(const Scheduler& s) {
        // horizon before boundary: a reader that loads boundary first and horizon second (event) sees a horizon
        // computed for that boundary or a later one, never an earlier one. Later only widens the margin.
        horizon.store(s.horizon());
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
    std::map<uint64_t, int64_t> spans_;              // head: their system spans (ServeReq::sys)
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
    // head: a Resize announced and not yet applied on this rank (no second report until then: it would ask again for
    // segments already granted)
    std::atomic<bool> resizing_{false};
    // head, borrow-idle: the chunks the FIFO head waits for (0 = none), when its first recall went out, the last
    // request the wait line named
    uint32_t recall_ = 0;
    std::chrono::steady_clock::time_point recall_t0_{};
    uint64_t recall_for_ = UINT64_MAX;
    const long share_ms_ = std::max<long>(1, knob::s2_share_ms());
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
    uint64_t waited_ = 0;         // the request whose wait the trace has
    void end_step() {
        if (!step_.open) return;
        trace_.span(step_.kind, Trace::STEPS, step_.t0, Trace::now(), step_.args);
        step_.open = false;
    }
    // GR8_S2_ADMIT_HOLD=1 (head only; the workers follow Commit::hold): an admission with another request queued behind it
    // and a slot left holds its boundary, and the next admission is announced at that same boundary (event below), so a
    // burst of queued requests is granted in milliseconds rather than one per NOTICE_MIN boundaries (~15 s each behind
    // 8,192-row prefill quanta). Default off: one admission per notice, as before.
    const bool admit_hold_ = knob::s2_admit_hold();
    uint64_t refused_ = UINT64_MAX;   // the last request the refusal line named (head)
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
        native_.pool.retain(s.run, st, g6rt::steering::cache_namespace(s.req.steering));
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
                // pages are page blocks; *_bytes = pages x block_bytes
                const uint64_t bb = t.block_bytes;
                o << (i ? "," : "") << "{\"slots\":" << t.slots << ",\"free_slots\":" << t.free_slots << ",\"entries\":" << t.entries
                  << ",\"shared\":" << t.shared << ",\"block_bytes\":" << bb << ",\"spilled_pages\":" << t.spilled_pages
                  << ",\"spilled_bytes\":" << t.spilled_pages * bb << ",\"moved_in_pages\":" << t.moved_in_pages
                  << ",\"moved_in_bytes\":" << t.moved_in_pages * bb << ",\"moved_out\":" << t.moved_out
                  << ",\"moved_out_pages\":" << t.moved_out_pages << ",\"restored_pages\":" << t.restored_pages
                  << ",\"restored_bytes\":" << t.restored_pages * bb << ",\"hits\":" << t.hits << ",\"dropped\":" << t.dropped
                  << ",\"dropped_pages\":" << t.dropped_pages << ",\"refused\":" << t.refused << ",\"refused_pages\":" << t.refused_pages
                  << ",\"full\":" << t.full << ",\"write_credit\":" << t.credit << "}";
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
    // held: the boundary every rank is stopped at after a commit with hold (Commit::hold); the notice names it, not a
    // future one. Only event() itself passes it, at once after the held commit, so nothing else is announced between.
    void event(const Command& command, std::optional<uint64_t> held = {}) { // CONTROL thread: never read Driver until ALL ACKs
        const auto boundary = held ? *held : progress_.boundary.load(); // first: the horizon below is for this boundary or a later one
        const auto horizon = progress_.horizon.load();
        require(boundary <= uint64_t(INT64_MAX) - NOTICE_MAX, "event boundary overflow");
        // The notice lands at the first boundary whose work after the one in flight covers NOTICE_ROWS rows, so a
        // prefill no longer waits out eight ~6 s quanta for a propagation that takes milliseconds (#61); late notices
        // remain fatal. Clamped, so a torn read of the pair can only widen the target, never narrow it past the floor.
        // A held boundary has no work in flight: every rank is stopped at it (event_gate.h before_work returns true).
        const auto target = held ? *held : std::clamp(horizon, boundary + NOTICE_MIN, boundary + NOTICE_MAX);
        if (trace_)
            trace_.instant("notice", Trace::CONTROL, targ("request", command.request) + "," +
                           targ("kind", command.resize ? "resize" : command.admit ? "admit" : "cancel") + "," +
                           targ("anchored", boundary) + "," + targ("boundary", target));
        if (elog_)
            fprintf(stderr, "# s2_event notice rank %u event %llu %s request %llu anchored %llu -> boundary %llu (lookahead %llu)\n",
                    c_.rank, (unsigned long long)event_, command.resize ? "resize" : command.admit ? "admit" : "cancel",
                    (unsigned long long)command.request,
                    (unsigned long long)boundary, (unsigned long long)target, (unsigned long long)(target - boundary));
        Notice n{c_.scheduler.epoch, event_, target, command.request, command.hash,
                 command.resize ? EventKind::Resize : command.admit ? EventKind::Admit : EventKind::Cancel};
        bool hold = false;
        leader_->event(n, [&](uint64_t hash) {
            const auto& s = driver_->state(); // compute paused in boundary() until this commit
            Commit cm{n.epoch, n.event, n.boundary, hash, {}, {}};
            cm.resize = command.resize;   // every rank applies it at the commit (Driver resize hook), the head too
            if (!command.admit && !command.resize) cm.cancel = s.find(command.request);
            // Head owns the real pool delta once; Driver callbacks skip it here. A Retiring slot gave its run back when it
            // finished (Driver finished hook); a cancelled live one gives it back now.
            for (const auto& sl : s.slots())
                if (cm.cancel && sl.phase != Phase::Free && sl.phase != Phase::Retiring && sl.key == *cm.cancel) retire_run(sl);
            if (command.admit) {
                const auto* request = queue_.front(); require(request && request->id == command.request, "admission FIFO changed during event");
                unsigned live = s.resident();
                const auto worst = reserve_rows(c_.scheduler, *request), rows = admit_rows(*request);
                // a prompt that continues a retained run fits when the hit does (match checks the room the hit needs, with
                // the run's own pages and records counted as its own), even when a fresh run of it would not
                const auto pr0 = prompts_.find(request->id);
                PoolHit pre{};
                if (reuse() && pr0 != prompts_.end())
                    pre = native_.pool.match(pr0->second, rows, worst, int(request->prompt_tokens) - 1,
                                             g6rt::steering::cache_namespace(request->steering));
                const bool fits = pre.skip > 0 || native_.pool.fits(rows, worst);
                const auto fit = queue_.check(s.revision(), live < c_.scheduler.limit(), fits, native_.pool.room());
                require(fit.has_value(), "duplicate admission attempt without revision");
                if (trace_ && fit->why != Stall::None && waited_ != request->id) {
                    waited_ = request->id;
                    trace_.instant("wait", Trace::CONTROL, targ("request", request->id) + "," +
                                   targ("why", fit->why == Stall::Slots ? "slots" : "capacity") + "," +
                                   targ("requested", fit->requested) + "," + targ("room", fit->room) + "," +
                                   targ("live", live));
                }
                if (fit->why == Stall::Capacity && refused_ != request->id) {
                    refused_ = request->id;   // once per request
                    if (reuse() && pr0 != prompts_.end())
                        fprintf(stderr, "# s2 reuse request %llu waits: %s retained %d lcp %d prompt %u\n", (unsigned long long)request->id,
                                pre.why, pre.skip, pre.lcp, request->prompt_tokens);
                    if (native_.pool.refusal)
                        if (const std::string why = native_.pool.refusal(worst); !why.empty())
                            fprintf(stderr, "# s2 rt: request %llu (%llu rows) waits: %s\n", (unsigned long long)request->id,
                                    (unsigned long long)worst, why.c_str());
                }
                // borrow-idle: a request whose worst case passes the chunks every rank has online waits for a recall
                const uint32_t short_of = fit->why == Stall::None && native_.flash.chunk_rows
                                              ? native_.flash.recall(worst, leader_->agreed_room()) : 0;
                if (short_of) {
                    if (recall_for_ != request->id) {
                        recall_for_ = request->id; recall_t0_ = std::chrono::steady_clock::now();
                        fprintf(stderr, "# s2 flash kv: request %llu (%llu rows) waits for a recall to %u chunks (%u online)\n",
                                (unsigned long long)request->id, (unsigned long long)worst, short_of, leader_->agreed_room());
                    }
                    recall_ = short_of;
                } else if (fit->why == Stall::None && recall_for_ == request->id) {
                    fprintf(stderr, "# s2 flash kv: request %llu admitted %lld ms after its recall began\n",
                            (unsigned long long)request->id, (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - recall_t0_).count());
                    recall_for_ = UINT64_MAX;
                }
                if (fit->why == Stall::None && !short_of) {
                    Run run{}; uint32_t skip = 0, restored = 0, counted = 0;
                    const auto pr = prompts_.find(request->id);
                    if (reuse() && pr != prompts_.end()) {
                        if (native_.pool.count)
                            counted = native_.pool.count(pr->second, spans_.count(request->id) ? spans_.at(request->id) : -1,
                                g6rt::steering::cache_namespace(request->steering));
                        const PoolHit h = native_.pool.match(pr->second, rows, worst, int(request->prompt_tokens) - 1,
                                                            g6rt::steering::cache_namespace(request->steering));
                        if (h.skip > 0) { run = native_.pool.take(h, rows, worst); skip = uint32_t(h.skip); restored = h.stored ? h.tier + 1 : 0; }
                        fprintf(stderr, "# s2 reuse request %llu %s retained %d lcp %d prompt %u first page %u\n", (unsigned long long)request->id,
                                h.why, h.skip, h.lcp, request->prompt_tokens, skip ? run.first_page : 0u);
                    }
                    // a pool that said it fits and then refuses is a bug, but one request's: it waits, the epoch goes on
                    bool got = true;
                    if (!skip) {
                        try { run = native_.pool.reserve(rows, worst); }
                        catch (const std::exception& e) {
                            got = false;
                            fprintf(stderr, "# s2: request %llu (%llu rows): the pool refused its reservation after fits(): %s; it waits\n",
                                    (unsigned long long)request->id, (unsigned long long)worst, e.what());
                        }
                    }
                    if (got) cm.admit = Grant{s.next_key(), *request, run, skip, restored, counted};
                }
                attempted_head_ = request->id; attempted_revision_ = s.revision() + 1;
                if (cm.admit) {
                    if (trace_)
                        trace_.instant("admit", Trace::CONTROL, targ("request", request->id) + "," +
                                       targ("reused", cm.admit->reuse) + "," + targ("rows", worst) + "," + targ("live", live));
                    queue_.admitted(request->id); commands_.erase(command.request); prompts_.erase(command.request);
                    spans_.erase(command.request); payloads_.erase(command.request);
                    // another request queued and a slot left after this one: keep the ranks at this boundary for it
                    hold = cm.hold = admit_hold_ && queue_.front() && live + 1 < c_.scheduler.limit();
                }
            }
            return cm;
        }, command.admit ? &payloads_.at(command.request) : nullptr);
        require(event_ != UINT64_MAX, "event sequence exhausted"); ++event_; publish_status();
        // Every rank is stopped at n.boundary and waits for a notice that names it (event_gate.h held_): announce the new
        // FIFO head there now. Its commit holds again or releases (no grant, or nothing left to chain), so the chain
        // ends within limit() admissions.
        if (hold) { const auto next = commands_.at(queue_.front()->id); event(next, n.boundary); }
        // the FIFO head waits for lent chunks: recall them (every rank grows at the commit), then it is tried again
        if (recall_ && !resizing_.load()) {
            Command c; c.epoch = c_.scheduler.epoch; c.resize = Resize{FLASH_KV_TIER, recall_, 0, 0};
            recall_ = 0; resizing_.store(true); event(c);
        }
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
    void controller() {
        uint64_t status_revision = UINT64_MAX;
        auto next_share = std::chrono::steady_clock::now();
        while (!stopping_.load()) {
            bool changed = false;
            if (native_.share && !resizing_.load() && std::chrono::steady_clock::now() >= next_share) {
                next_share = std::chrono::steady_clock::now() + std::chrono::milliseconds(share_ms_);
                if (const auto r = native_.share(progress_.resident.load() == 0 && queue_.size() == 0)) {
                    fprintf(stderr, "# s2 share: tier %u to %llu segments (%llu blocks)\n", r->tier, (unsigned long long)r->segs,
                            (unsigned long long)r->slots);
                    resizing_.store(true);
                    Command c; c.epoch = c_.scheduler.epoch; c.resize = *r;
                    event(c); changed = true;
                }
            }
            // borrow-idle: lend the chunks above flash.keep once the engine has been idle flash.idle_ms
            if (native_.flash.chunk_rows && !resizing_.load())
                if (const auto n = native_.flash.poll(progress_.resident.load() == 0 && queue_.size() == 0 &&
                                                      progress_.retiring.load() == 0, std::chrono::steady_clock::now())) {
                    fprintf(stderr, "# s2 flash kv: idle, lending down to %u chunks\n", *n);
                    resizing_.store(true);
                    Command c; c.epoch = c_.scheduler.epoch; c.resize = Resize{FLASH_KV_TIER, *n, 0, 0};
                    event(c); changed = true;
                }
            while (auto command = ingress_.next()) {
                changed = true;
                if (command->admit) {
                    auto bytes = ingress_.request(*command);
                    auto request = parse_request(bytes, c_.scheduler.steering);
                    if (!grammar_ok(command->request, request)) continue;
                    const auto queued = Sessions::request(command->request, request);
                    if (trace_)
                        trace_.instant("arrive", Trace::CONTROL, targ("request", queued.id) + "," +
                                       targ("prompt_tokens", queued.prompt_tokens) + "," + targ("max_tokens", queued.max_tokens) +
                                       "," + targ("queued", queue_.size()));
                    queue_.push(queued); commands_.emplace(command->request, *command);
                    payloads_.emplace(command->request, std::move(bytes));
                    if (reuse() && request.plp < 0) {
                        spans_.emplace(command->request, request.sys);
                        prompts_.emplace(command->request, std::move(request.prompt));
                    }
                } else if (queue_.cancel(command->request)) {
                    commands_.erase(command->request); prompts_.erase(command->request); spans_.erase(command->request);
                    payloads_.erase(command->request);
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
            end_step();
            Notice noticed{};
            if (!driver_->state().resident()) {
                const auto n = endpoint_->wait_notice(); driver_->idle_boundary(n.boundary);
            }
            auto cm = endpoint_->boundary(driver_->state().boundary(), [&] {
                native_.fence(); const auto n = endpoint_->wait_notice(); noticed = n; staged_.reset();
                if (n.kind == EventKind::Admit) {   // the head reads its own req.<id>; a worker has the bytes from the head
                    const auto bytes = worker_ ? worker_->request(n.request) : ingress_.request({true,n.epoch,n.request,n.payload_hash});
                    require(payload_hash(bytes) == n.payload_hash, "request bytes do not match the ADMIT notice");
                    staged_ = parse_request(bytes, c_.scheduler.steering);
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
                if (trace_)
                    trace_.instant("commit", Trace::STEPS, targ("request", noticed.request) + "," +
                                   targ("kind", noticed.kind == EventKind::Resize ? "resize"
                                                : noticed.kind == EventKind::Cancel ? "cancel" : "admit") + "," +
                                   targ("boundary", cm->boundary));
                if (elog_)
                    fprintf(stderr, "# s2_event commit rank %u event %llu %s request %llu boundary %llu\n", c_.rank,
                            (unsigned long long)noticed.event,
                            noticed.kind == EventKind::Resize ? "resize" : noticed.kind == EventKind::Cancel ? "cancel" : "admit",
                            (unsigned long long)noticed.request, (unsigned long long)cm->boundary);
                if (worker_ && cm->admit) worker_->admitted(cm->admit->req.id);
                if (cm->resize) resizing_.store(false);
                dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
                staged_.reset(); endpoint_->applied(); continue;
            }
            if (native_.before_work) native_.before_work();
            const auto t0 = trace_ ? Trace::now() : 0;
            const auto work = driver_->step(); require(work.kind != Kind::Idle, "departed idle boundary");
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
                        work.kind == Kind::Prefill ? "prefill" : "tail", work.first, work.rows);
            dispatch_.synchronize_mode(driver_->state()); progress_.publish(driver_->state());
        }
    }
public:
    unsigned quit_linger_ms = 15000; // orderly-quit linger (monitor); > the time the workers take to see the head's quit
    Service(BootstrapConfig c, ControlLinks& links, NativeHooks hooks, int chunk)
        : c_(std::move(c)), links_(links), native_(std::move(hooks)), sessions_(c_.scheduler,c_.directory,c_.rank == 0,chunk),
          dispatch_(sessions_,c_.scheduler.max_rows,native_.dispatch), ingress_(c_.directory,c_.scheduler.epoch), queue_(c_.scheduler) {
        c_.validate();
        trace_.open(knob::s2_trace().value_or(""), c_.rank);
        require(bool(native_.pool.free_pages) && bool(native_.pool.room) && bool(native_.pool.reserve) && bool(native_.pool.fits) &&
                bool(native_.pool.release) && bool(native_.bind) && bool(native_.retire) && bool(native_.fence), "incomplete native service hooks");
        require(!native_.pool.grow || bool(native_.grown), "pool growth needs the native table hook");
        require(!native_.share || bool(native_.resize), "a share report needs the resize hook");
        require(!native_.flash.chunk_rows || bool(native_.resize), "borrow-idle needs the resize hook");
        DriverHooks h;
        h.resize = native_.resize;
        h.bind = [&](const Grant& g) {
            require(staged_.has_value(), "binding without verified payload");
            require(!g.reuse || reuse(), "reused rows without prefix reuse");
            if (c_.rank) {   // the head's call, on the same pool state
                const auto worst = reserve_rows(c_.scheduler,g.req), rows = admit_rows(g.req);
                Run mine;
                if (native_.pool.count && reuse() && staged_->plp < 0)
                    require(native_.pool.count(staged_->prompt, staged_->sys,
                        g6rt::steering::cache_namespace(g.req.steering)) == g.counted, "worker prefix counts differ from the head");
                else require(!g.counted, "a counted grant without counting");
                if (g.reuse) {
                    const PoolHit h = native_.pool.match(staged_->prompt, rows, worst, int(g.req.prompt_tokens) - 1,
                                                         g6rt::steering::cache_namespace(g.req.steering));
                    require(h.skip == int(g.reuse) && (h.stored ? h.tier + 1 : 0) == g.restored, "worker prefix reuse differs from the head");
                    mine = native_.pool.take(h, rows, worst);
                } else mine = native_.pool.reserve(rows, worst);
                require(mine == g.run, "worker page list differs from the head's grant");
            }
            dispatch_.begin(g.key.slot, int(g.reuse), g6rt::steering::cache_namespace(g.req.steering));
            native_.bind(g); sessions_.bind(g,*staged_);
        };
        h.retire = [&](const Slot& old) {
            if (c_.rank && old.phase != Phase::Retiring) retire_run(old);   // a cancelled live session (finished ones: below)
            native_.retire(old); sessions_.retire(old); dispatch_.release(old.key.slot);
        };
        h.prefill = [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows) {
            dispatch_.prefill(s,p,first,rows);
            sessions_.progress(s.key,first + rows);   // rank 0: a long prefill tells the front end it is alive
        };
        h.run = [&](const s2::Step& step) { return dispatch_.run(step); };
        h.emit = [&](Key k,const int* ids,int n,int row) {
            if (trace_ && n > 0) {
                const auto id = driver_->state().slots()[k.slot].req.id;
                if (first_.insert(id).second) trace_.instant("first token", Trace::STEPS, targ("request", id));
            }
            return sessions_.emit(k,ids,n,row,native_.dispatch.logprob);
        };
        h.prompt = [&](Key k,int M,int pos0) { sessions_.prompt_rows(k,M,pos0,native_.dispatch.logprob); };
        h.finished = [&](const Slot& s) {
            if (trace_) {
                trace_.instant("finish", Trace::STEPS, targ("request", s.req.id) + "," + targ("tokens", s.emitted));
                first_.erase(s.req.id);
            }
            retire_run(s);
        };
        if (native_.drained)   // the copies run at a bind, a growth or a retire: all before the next step's round
            h.settle = [&] {
                if (!native_.drained()) return;
                // the barrier after a drain: every rank waits here for the slowest rank's drain (the drain trace's other
                // half: the tier io lines carry the same ranks' drains)
                auto wall = [] { return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); };
                const long long b0 = wall();
                endpoint_->sync(++sync_);
                fprintf(stderr, "# rank %u: s2 drain barrier %llu: in %lld ms, out %lld ms\n", (unsigned)c_.rank, (unsigned long long)sync_, b0, wall());
            };
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
            if (native_.dfl_reuse.mode && !c_.rank)   // every request, so one that never had the drafter shows too
                fprintf(stderr,"# s2 dflash reuse slot %u: %ld of %ld rounds on DFlash2, %ld on MTP; context from %d (taps from %d), "
                        "warm-up to %d, promoted at %d, %ld tokens\n",unsigned(k.slot),stats.dfl_rounds,stats.rounds,stats.mtp_rounds,
                        stats.dfl_lo,stats.dfl_from,stats.dfl_warm,stats.promoted,stats.tokens);
            if (native_.ngram.t && !c_.rank)
                fprintf(stderr,"# s2 ngram slot %u: %ld of %ld rounds verified a table chain, %ld of %ld of its tokens accepted, %ld tokens\n",
                        unsigned(k.slot),stats.ngram_rounds,stats.rounds,stats.ngram_acc,stats.ngram_drafted,stats.tokens);
            sessions_.terminal(k,stats);
        };
        if (native_.dispatch.dfl) {
            require(bool(native_.dispatch.dfl_ok) && bool(native_.dispatch.dfl_start), "DFlash2 needs dfl_ok, dfl_start and dfl");
            h.dfl_ok = [&](Key k) { return dispatch_.dfl_ok(k.slot); };
            h.dfl_start = [&](Key k) { return dispatch_.dfl_start(k.slot); };
            h.dfl = [&](Key, const s2::DflReq& r) { return dispatch_.dfl(r); };
            h.draft_lp = native_.dispatch.draft_lp;
        }
        // MTP only (no drafter): the confidence stops of round_all read the draft log-probabilities too (g6m4 sets kstop)
        const bool kstop_mtp = !native_.dispatch.dfl && (native_.kstop.mode || native_.kstop.b.mode || native_.kstop.log);
        if (kstop_mtp) h.draft_lp = native_.dispatch.draft_lp;
        driver_ = std::make_unique<Driver>(c_.scheduler,std::move(h));
        if (native_.dispatch.dfl) {
            driver_->dflash(native_.dfl_kv, native_.dfl_spec); driver_->dsel(native_.dsel); driver_->kstop(native_.kstop);
            driver_->dfl_reuse(native_.dfl_reuse);
            driver_->ngram(native_.ngram);
        }
        else if (kstop_mtp) driver_->kstop(native_.kstop);
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
        endpoint_->room = native_.flash_room;
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
