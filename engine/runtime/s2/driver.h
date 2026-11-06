// Bridge from the scheduler to the persistent Batch. No CUDA or allocator.
// All calls run on the compute thread at safe boundaries. apply() requires an
// EventGate all-rank fence; callbacks must fail the whole epoch on any error.
// The driver's logic is Rust (engine/scheduler-rs src/driver.rs), and so is the scheduler's state, which state() reads.
// Driver is a facade: it keeps the payloads and the Batch (which writes the round log), and answers the Rust's hooks
// and planner calls from them. An exception a hook throws is held until the Rust returns, then rethrown. The Batch's hooks (run, emit, prompt, round,
// dfl, dfl_start) must not call stats() or emitted(), which read the Batch; the other hooks may call any getter. No hook
// may call apply() or step(): the Rust refuses them.
#pragma once
#include "../ds41_image.h"
#include "scheduler.h"
#include "cp_rows.h"
#include "../s2_batch.h"
#include "../grammar.h"

namespace g6s2 {
// decode_prompt: the whole prompt runs as decode steps from position 0, no tensor prefill (prompt logprobs)
struct Payload {
    std::vector<int> prompt; int chunk = 4; bool decode_prompt = false;
    g6rt::InsertRule ins;
    g6rt::steering::Identity steering{};
    g6rt::GrammarSpec gram;
    std::vector<g6rt::ImageIn> images;   // DS41 image spans (ds41_image.h); only on prefilled prompts (Service::images_ok)
};
// One member of a single bounded W pass. Pointers remain valid for Driver::step's synchronous callback.
struct PrefillPart { const Slot* slot = nullptr; const Payload* payload = nullptr; uint32_t first = 0, rows = 0; };
struct Emission { uint32_t count = 0; bool finished = false; };
struct DriverHooks {
    // Real TP4 pool/binding adapter: canonical base, never a per-rank first-fit.
    std::function<void(const Grant&)> bind;
    std::function<void(const Slot&)> retire;
    // Full prefill quantum: both lanes, main AND draft consumption, synchronized.
    std::function<void(const Slot&, const Payload&, uint32_t, uint32_t)> prefill;
    std::function<std::vector<int>(const s2::Step&)> run;
    // Exact native Emitter policy (stop/min/max) on EVERY rank, not just rank0.
    std::function<Emission(Key, const int*, int, int)> emit;   // (key, ids, count, row of ids[0] in the last main step)
    // optional: after each prompt main step of a session, (key, rows, position of row 0)
    std::function<void(Key, int, int)> prompt;
    std::function<void(Key, long, int, int, int)> round;
    // optional, DFlash2: may a session alone at its prompt tail use the drafter (it owns the prefill's taps); the
    // drafter's prompt context (returns the lowest position it holds); one drafter step
    std::function<bool(Key)> dfl_ok;
    std::function<int(Key)> dfl_start;
    std::function<s2::DflDrafts(Key, const s2::DflReq&)> dfl;
    // optional: row r's log-probability in the last draft or chain step (s2::Hooks::lp, GR8_MTP_KSTOP / GR8_MTP_LPLOG)
    std::function<float(int)> draft_lp;
    // One completion notification.
    std::function<void(Key, const s2::Stats&)> terminal;
    // optional: before a round, a decoding session's run grows to cover `rows` (round_rows). The run, perhaps the same,
    // or nothing: the pages would leave the pool unsafe and the session sits this round out. Every rank, same order.
    // Not with DriverPool::grow, where the driver grows the run in the pool itself
    std::function<std::optional<Run>(const Slot&, uint64_t rows)> grow;
    // optional, with DriverPool::grow: a decoding session's run grew before a round (the slot as it was, its new run);
    // the engine's tables take the new pages. Every rank, same order
    std::function<void(const Slot&, const Run&)> grown;
    // optional: a session finished at this step (now Retiring): its pages go back to the pool now, on every rank, so
    // the sessions still decoding can grow before the next event fence frees the slot
    std::function<void(const Slot&)> finished;
    // optional: a Resize event's change (Commit::resize), every rank at the commit: the tier and its group lease change
    // size (s2/kv_share.h share_apply). Required when a resize commit arrives
    std::function<void(const Resize&)> resize;
    // optional: a session takes the smaller R (Commit::shrink), every rank at the commit before any bind, with its run as
    // it was and its length (tokens; rows below length - 1 are written): the pool's change (workers) and the engine's
    // tables and pages. Required when a shrink commit arrives
    std::function<void(const Shrink&, const Run& was, uint64_t length)> shrink;
    // optional: after the growth and before the round, every rank (the host work since the last round may have taken a
    // different time on each rank: the round's fabric waits time out)
    std::function<void()> settle;
    // DS41 prefill pack: one native pass for the ordered segments, never N solo hooks.
    std::function<void(const std::vector<PrefillPart>&)> prefill_pack;
};
// The pool calls the Rust driver makes itself on the rank's pool (pool.h RunPool::handle; the pool must outlive the
// Driver), each under the pool's lock for that call alone and with no hook running. grow: runs grow before a round
// (then DriverHooks::grown, required). shrink: a worker's half of a shrink commit, checked against the head's, before
// the shrink hook. release: a session's run goes back when it leaves (no prefix reuse: retain and park stay in the
// retire and finished hooks). head: this rank made the commit's cancel and shrink calls itself when it built the commit
struct DriverPool { g6kv_pool* pool = nullptr; bool grow = false, shrink = false, release = false, head = false; };
class Driver {
    struct Impl {
        Config cfg;                        // the C++ copy: the Rust's scalars and the steering receipt the digest reads
        SchedulerView view;                // the Rust's slots and counters, set once the Rust driver exists
        DriverHooks hooks;
        std::array<Payload, MAX_SESSIONS> payload;
        s2::Batch batch;
        struct Free { void operator()(g6s2_driver* d) const { g6s2_driver_free(d); } };
        std::unique_ptr<g6s2_driver, Free> rs;
        std::exception_ptr err;            // the exception a hook threw, rethrown when the Rust returns
        const Payload* admit = nullptr;    // apply's payload and commit, for the payload and resize hooks
        const Commit* commit = nullptr;
        // apply's admit and commit as they were, back on any exit: a hook's own apply() (refused) leaves the outer call's
        struct Restore {
            Impl& m; const Payload* admit; const Commit* commit;
            explicit Restore(Impl& i) : m(i), admit(i.admit), commit(i.commit) {}
            ~Restore() { m.admit = admit; m.commit = commit; }
        };
        s2::Step work;                     // reused buffers: a round allocates nothing here after warm-up
        std::vector<int> order, ids;
        std::vector<s2::Session> sessions;
        std::vector<PrefillPart> parts;

        static const Config& checked(const Config& c) { sch::check_abi(); c.validate(); return c; }
        Impl(const Config& c, DriverHooks h) : cfg(checked(c)), hooks(std::move(h)), batch(int(c.mtp), int(c.max_rows), batch_hooks()) {}
        s2::Hooks batch_hooks() {
            return {
                [this](const s2::Step& s) {
                    work = s;
                    work.scheduler_decode = g6s2_driver_decoding(rs.get());
                    auto ids = hooks.run(work);
                    require(ids.size() == s.r.size(), "engine returned wrong row count");
                    return ids;
                },
                [this](int q, long round, int base, int accepted, int K) {
                    if (hooks.round) hooks.round(view.slots()[q].key, round, base, accepted, K);
                },
                [this](int q, int M, int pos0) {
                    if (hooks.prompt) hooks.prompt(view.slots()[q].key, M, pos0);
                },
                [this](int q, const int* ids, int n, int row) {
                    const auto& s = view.slots()[q];
                    const auto e = hooks.emit(s.key, ids, n, row);
                    bool done = false;
                    check(g6s2_driver_emit(rs.get(), unsigned(q), n, e.count, e.finished, s.req.max_tokens, &done));
                    return done;
                },
                hooks.dfl ? std::function<s2::DflDrafts(const s2::DflReq&)>([this](const s2::DflReq& r) {
                    return hooks.dfl(view.slots()[r.sess].key, r);
                }) : nullptr,
                hooks.dfl_start ? std::function<int(int)>([this](int q) { return hooks.dfl_start(view.slots()[q].key); }) : nullptr,
                hooks.draft_lp
            };
        }
        // a call's result: 1 = a hook threw (rethrown here), 2 = the Rust refused
        void check(int32_t rc) {
            if (rc == 1) {
                if (!err) throw std::runtime_error("s2: a driver hook failed without an exception");
                std::exception_ptr e = err; err = nullptr; std::rethrow_exception(e);
            }
            if (rc) throw std::runtime_error(g6s2_driver_error(rs.get()));
        }
        template <class F> int32_t guard(F&& f) noexcept {
            try { f(); return 0; } catch (...) { err = std::current_exception(); return 1; }
        }
        static Impl& of(void* c) { return *static_cast<Impl*>(c); }
        // the live slot a Rust slot copy names (the hooks have always seen the scheduler's own)
        const Slot& live(const g6sch_slot* s) const { return view.slots()[s->key.slot]; }

        static int32_t t_bind(void* c, const g6sch_grant* g) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.bind(*sch::c<Grant>(g)); });
        }
        static int32_t t_retire(void* c, const g6sch_slot* s) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.retire(*sch::c<Slot>(s)); });
        }
        static int32_t t_prefill(void* c, const g6sch_slot* s, uint32_t first, uint32_t rows) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.prefill(m.live(s), m.payload[s->key.slot], first, rows); });
        }
        static int32_t t_prefill_pack(void* c, const uint32_t* slots, const uint32_t* firsts, const uint32_t* rows, uint32_t n) noexcept {
            Impl& m = of(c);
            return m.guard([&] {
                m.parts.clear();
                for (uint32_t i = 0; i < n; ++i) m.parts.push_back({&m.view.slots()[slots[i]], &m.payload[slots[i]], firsts[i], rows[i]});
                m.hooks.prefill_pack(m.parts);
            });
        }
        static int32_t t_dfl_ok(void* c, const g6sch_key* k, bool* ok) noexcept {
            Impl& m = of(c); return m.guard([&] { *ok = m.hooks.dfl_ok(*sch::c<Key>(k)); });
        }
        static int32_t t_terminal(void* c, const g6sch_key* k) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.terminal(*sch::c<Key>(k), m.batch.stats(int(k->slot))); });
        }
        static int32_t t_grow(void* c, const g6sch_slot* s, uint64_t rows, g6sch_run* out, bool* some) noexcept {
            Impl& m = of(c);
            return m.guard([&] {
                const auto r = m.hooks.grow(m.live(s), rows);
                if ((*some = bool(r))) std::memcpy(out, &*r, sizeof *out);
            });
        }
        static int32_t t_grown(void* c, const g6sch_slot* s, const g6sch_run* r) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.grown(m.live(s), *sch::c<Run>(r)); });
        }
        static int32_t t_finished(void* c, const g6sch_slot* s) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.finished(m.live(s)); });
        }
        static int32_t t_resize(void* c) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.resize(*m.commit->resize); });
        }
        static int32_t t_shrink(void* c, const g6sch_shrink*, const g6sch_run* was, uint64_t length) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.shrink(*m.commit->shrink, *sch::c<Run>(was), length); });
        }
        static int32_t t_settle(void* c) noexcept {
            Impl& m = of(c); return m.guard([&] { m.hooks.settle(); });
        }
        static int32_t t_payload(void* c, uint32_t slot, bool set) noexcept {
            Impl& m = of(c); return m.guard([&] { m.payload[slot] = set ? *m.admit : Payload{}; });
        }
        static int32_t t_admit(void* c, uint32_t slot, int32_t first, int32_t n_gen, bool dfl) noexcept {
            Impl& m = of(c);
            return m.guard([&] {
                const Payload& p = m.payload[slot];
                m.batch.admit({int(slot)}, {{p.prompt, first, p.chunk, n_gen, dfl, p.ins}});
            });
        }
        static int32_t t_admit_pack(void* c, const uint32_t* slots, const int32_t* firsts, const int32_t* n_gens, uint32_t n) noexcept {
            Impl& m = of(c);
            return m.guard([&] {
                m.ids.clear(); m.sessions.clear();
                for (uint32_t i = 0; i < n; ++i) {
                    const Payload& p = m.payload[slots[i]];
                    m.ids.push_back(int(slots[i]));
                    m.sessions.push_back({p.prompt, firsts[i], p.chunk, n_gens[i], false, p.ins});
                }
                m.batch.admit_packed_tail(m.ids, m.sessions);
            });
        }
        static int32_t t_round(void* c, const int32_t* order, uint32_t n, bool* ran) noexcept {
            Impl& m = of(c); return m.guard([&] { m.order.assign(order, order + n); *ran = m.batch.round(m.order); });
        }
        static int32_t t_demote(void* c) noexcept { Impl& m = of(c); return m.guard([&] { m.batch.demote(); }); }
        static int32_t t_promote(void* c, int32_t q) noexcept { Impl& m = of(c); return m.guard([&] { m.batch.promote(q); }); }
        static int32_t t_release(void* c, int32_t q) noexcept { Impl& m = of(c); return m.guard([&] { m.batch.release(q); }); }
        static int32_t t_length(void* c, int32_t q, int32_t* out) noexcept {
            Impl& m = of(c); return m.guard([&] { *out = m.batch.length(q); });
        }
        static int32_t t_finished_q(void* c, int32_t q, bool* out) noexcept {
            Impl& m = of(c); return m.guard([&] { *out = m.batch.finished(q); });
        }
        static int32_t t_active(void* c, int32_t q, bool* out) noexcept {
            Impl& m = of(c); return m.guard([&] { *out = m.batch.active(q); });
        }
        static int32_t t_inserted(void* c, int32_t q, int32_t* out) noexcept {
            Impl& m = of(c); return m.guard([&] { *out = int32_t(m.batch.stats(q).inserted); });
        }
    };
    std::unique_ptr<Impl> m_;   // on the heap: the Rust holds its address, so a moved Driver keeps working
    bool initialized(unsigned slot) const { return g6s2_driver_initialized(m_->rs.get(), slot); }
public:
    Driver(Config c, DriverHooks h, DriverPool pool = {}) : m_(new Impl(c, std::move(h))) {
        Impl& m = *m_;
        m.batch.adapt(s2::adapt_env());
        m.batch.set_mtp_sessions(int(c.mtp_sessions));
        m.batch.set_sess_drafts(c.sess_drafts);
        const DriverHooks& k = m.hooks;
        require(bool(k.bind) && bool(k.retire) && bool(k.prefill) && bool(k.run) && bool(k.emit), "missing driver callback");
        require(pool.pool || !(pool.grow || pool.shrink || pool.release), "driver pool calls without a pool");
        require(!(pool.grow && k.grow), "the grow hook and the driver's pool growth together");
        require(!pool.grow || bool(k.grown), "the driver's pool growth without the grown hook");
        const g6s2_drv_hooks hooks{&m, Impl::t_bind, Impl::t_retire, Impl::t_prefill, Impl::t_prefill_pack, Impl::t_dfl_ok,
                                   Impl::t_terminal, Impl::t_grow, Impl::t_finished, Impl::t_resize, Impl::t_shrink,
                                   Impl::t_settle, Impl::t_payload, Impl::t_grown, bool(k.prefill_pack), bool(k.dfl_ok),
                                   bool(k.terminal), bool(k.grow), bool(k.finished), bool(k.resize), bool(k.shrink),
                                   bool(k.settle), bool(k.grown)};
        const g6s2_drv_planner planner{&m, Impl::t_admit, Impl::t_admit_pack, Impl::t_round, Impl::t_demote, Impl::t_promote,
                                       Impl::t_release, Impl::t_length, Impl::t_finished_q, Impl::t_active, Impl::t_inserted};
        const Slot empty{};
        const g6s2_drv_pool use{pool.grow, pool.shrink, pool.release, pool.head};
        m.rs.reset(g6s2_driver_new(&hooks, &planner, &m.cfg, MAX_SESSIONS, sch::c<g6sch_slot>(&empty), pool.pool, &use));
        require(bool(m.rs), "missing driver callback");
        g6s2_drv_view v;
        g6s2_driver_view(m.rs.get(), &v);
        require(v.n == MAX_SESSIONS, "driver slot count");
        m.view = SchedulerView(&m.cfg, v.s, v.st);
    }
    // the scheduler's slots and counters. Valid for the Driver's life; while apply() or step() runs, only its hooks
    // may read them
    const SchedulerView& state() const { return m_->view; }
    uint64_t cp_capped_rounds() const { return g6s2_driver_cp_rounds(m_->rs.get()); }
    const s2::Stats& stats(Key k) const {
        require(k.slot < MAX_SESSIONS && state().slots()[k.slot].key == k && initialized(k.slot), "stats for stale/uninitialized slot");
        return m_->batch.stats(int(k.slot));
    }
    const s2::Stats* stats_if_initialized(Key k) const {
        require(k.slot < MAX_SESSIONS && state().slots()[k.slot].key == k, "stats for stale slot");
        return initialized(k.slot) ? &m_->batch.stats(int(k.slot)) : nullptr;
    }
    // a live or retiring slot's prompt and the tokens it emitted (prefix reuse history)
    const std::vector<int>& prompt(unsigned slot) const { require(slot < MAX_SESSIONS, "invalid slot"); return m_->payload[slot].prompt; }
    std::vector<int> emitted(unsigned slot) const {
        require(slot < MAX_SESSIONS, "invalid slot");
        if (!initialized(slot)) return {};
        auto t = m_->batch.result(int(slot));
        t.resize(std::min<size_t>(t.size(), g6s2_driver_emitted(m_->rs.get(), slot)));
        return t;
    }
    // DFlash2 sessions verify up to kv drafts with the --spec-adaptive policy sl (s2_batch.h Batch::dflash)
    void dflash(int kv, const g6rt::SpecLen& sl) { m_->batch.dflash(kv, sl); }
    void dsel(const s2::DSel& d) { m_->batch.dsel(d); }
    void kstop(const s2::KStop& k) { m_->batch.kstop(k); }
    void dfl_reuse(const s2::DflReuse& r) { m_->batch.dfl_reuse(r); }
    void ngram(const s2::NgramSrc& n) { m_->batch.ngram(n); }
    void fail() { g6s2_driver_fail(m_->rs.get()); }
    void idle_boundary(uint64_t b) { m_->check(g6s2_driver_idle_boundary(m_->rs.get(), b)); }   // event-only: no model work
    // the pool's digest (pool.h RunPool::digest) at this event boundary, so the ranks also agree on their pages
    void pool_digest(uint64_t d) { m_->check(g6s2_driver_pool_digest(m_->rs.get(), d)); }
    Changes apply(const Commit& c, const Payload* p = nullptr) {
        Impl& m = *m_;
        const g6sch_commit x = sch::to_rs(c);
        g6s2_drv_payload pay{};
        if (p) pay = {p->prompt.size(), p->chunk, p->decode_prompt, c.admit && p->steering == c.admit->req.steering};
        const uint64_t digest = m.view.digest();
        Changes d;
        d.freed.resize(MAX_SESSIONS);
        size_t n = 0; bool admitted = false;
        int32_t rc;
        {
            const Impl::Restore keep(m);
            m.admit = p; m.commit = &c;
            rc = g6s2_driver_apply(m.rs.get(), &x, digest, bool(c.resize), p ? &pay : nullptr,
                                   sch::m<g6sch_run>(d.freed.data()), &n, &admitted);
        }
        m.check(rc);
        d.freed.resize(n);
        if (admitted) d.admitted = c.admit;
        return d;
    }
    Quantum step() {
        Impl& m = *m_;
        g6sch_quantum q;
        m.check(g6s2_driver_step(m.rs.get(), &q));
        return sch::from_rs(q);
    }
};
} // namespace g6s2
