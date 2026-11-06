// The seeded command streams of scheduler_rs_vectors_test.cpp. Each section drives the S2 host logic (the Rust in
// engine/scheduler-rs, through its C++ headers) with a portable random stream and writes every result, refusal and
// state to a transcript; the test compares each transcript's hash with the one recorded from the C++ the Rust replaced.
// Included inside a namespace with an `impl` namespace that holds the implementation; shared types are g6s2's.
// The generator is mt19937_64 with modulo ranges, never <random>'s distributions, so a seed gives the same streams
// with every standard library. A call never takes two draws as arguments, since compilers order arguments differently.

struct Rng
{
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    int U(int lo, int hi) { return lo + int(g() % uint64_t(int64_t(hi) - lo + 1)); }
    uint64_t raw() { return g(); }
};

// the transcript: an FNV-1a hash of its lines, and the lines themselves when `lines` is set (to find a divergence)
struct Tx
{
    uint64_t h = 14695981039346656037ull;
    std::vector<std::string>* lines = nullptr;
    void operator()(const std::string& s)
    {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
        h = (h ^ '\n') * 1099511628211ull;
        if (lines) lines->push_back(s);
    }
};

// a call's outcome: its value printed, or the exception's type and message (out_of_range's message is the standard
// library's own, which differs between them)
template <class F> std::string outcome(F f)
{
    try { return "ok " + f(); }
    catch (const std::out_of_range&) { return "out_of_range"; }
    catch (const std::runtime_error& e) { return std::string("runtime_error ") + e.what(); }
    catch (const std::exception& e) { return std::string("exception ") + e.what(); }
}
template <class T> std::string str(const std::vector<T>& v)
{
    std::string s;
    for (const T& x : v) s += std::to_string(x) + ",";
    return s;
}
inline std::string str(const std::set<int>& v) { return str(std::vector<int>(v.begin(), v.end())); }
inline std::string str(const g6s2::Key& k) { return std::to_string(k.epoch) + "." + std::to_string(k.slot) + "." + std::to_string(k.generation); }
inline std::string str(const g6s2::Run& r)
{
    return std::to_string(r.first_page) + "+" + std::to_string(r.pages) + "v" + std::to_string(r.table_version) + "#" + std::to_string(r.list_hash);
}
inline std::string str(const g6s2::Quantum& q)
{
    std::string s = std::to_string(int(q.kind)) + "@" + std::to_string(q.boundary) + " n" + std::to_string(q.count) + " " +
                    std::to_string(q.first) + "+" + std::to_string(q.rows) + ":";
    for (unsigned i = 0; i < g6s2::MAX_SESSIONS; ++i)
        s += str(q.keys[i]) + "/" + std::to_string(q.firsts[i]) + "/" + std::to_string(q.row_counts[i]) + ",";
    return s;
}
inline std::string str(const g6rt::ServeReq& r)
{
    auto bits = [](float f) { uint32_t b; std::memcpy(&b, &f, 4); return std::to_string(b); };
    std::string s = str(r.prompt) + "|" + str(r.stop) + "|" + std::to_string(r.n_gen) + " " + std::to_string(r.min_tokens) + " " +
                    bits(r.temp) + " " + bits(r.top_p) + " " + std::to_string(r.seed) + " " + std::to_string(r.lp) + " " +
                    std::to_string(r.plp) + " " + std::to_string(r.sys) + "|" + std::to_string(r.ins.after) + " " +
                    std::to_string(r.ins.unless) + " " + str(r.ins.tokens) + "|" + std::to_string(int(r.gram.kind)) +
                    std::to_string(r.gram.strict) + " " + std::to_string(r.gram.arm) + " " + r.gram.schema + "|" +
                    std::to_string(r.steering.enabled) + " " + std::to_string(int(r.steering.scope)) + " " +
                    std::to_string(r.steering.scale_bits) + " " + std::to_string(r.steering.slot) + " ";
    for (auto b : r.steering.profile_digest) s += std::to_string(b) + ".";
    for (auto b : r.steering.schedule_digest) s += std::to_string(b) + ".";
    return s;
}
// the scheduler's state, field by field (padding bytes differ)
template <class S> std::string state(const S& x)
{
    std::string s = std::to_string(x.boundary()) + " " + std::to_string(x.next_event()) + " " + std::to_string(x.revision()) + " " +
                    std::to_string(x.busy()) + " " + std::to_string(x.resident()) + " " + std::to_string(x.retiring()) + " " +
                    std::to_string(x.parked()) + " h" + std::to_string(x.horizon()) + " d" + std::to_string(x.digest()) + "|";
    for (const auto& t : x.slots())
        s += str(t.key) + " " + std::to_string(t.req.id) + " " + std::to_string(t.req.prompt_tokens) + " " + std::to_string(t.req.max_tokens) +
             " " + std::to_string(t.req.steering.enabled) + std::to_string(t.req.steering.slot) + std::to_string(t.req.steering.scale_bits) +
             " " + std::to_string(t.req.r_rows) + " " + str(t.run) + " " + std::to_string(int(t.phase)) + " " + std::to_string(t.prefilled) +
             " " + std::to_string(t.emitted) + " " + std::to_string(t.inserted) + " " + std::to_string(t.deferred) + " " +
             std::to_string(t.packable) + std::to_string(t.pack_tail) + " " + std::to_string(t.parked) + ";";
    return s;
}

// kv_plan: copy lists in runs of one kind, a few tiers and slots, parts and subs, and every batch to the end
inline uint64_t kv_plan(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    for (int t = 0; t < 4000; ++t)
    {
        std::vector<g6kv_io> io;
        const int n = R.U(0, 40);
        uint32_t kind = uint32_t(R.U(0, 2));
        for (int k = 0; k < n; ++k)
        {
            if (!R.U(0, 6)) kind = uint32_t(R.U(0, 2));
            io.push_back({kind, uint32_t(R.U(0, 30)), uint32_t(R.U(0, 2)), uint32_t(R.U(0, 2)), uint64_t(R.U(0, 6)), uint64_t(R.U(0, 6)),
                          uint32_t(R.U(0, 2)), uint32_t(R.U(0, 3))});
        }
        const int batch = R.U(-1, 8), items_cap = R.U(-1, 12);
        const uint32_t per_slot = uint32_t(R.U(0, 4));
        size_t i = 0;
        for (int guard = 0; guard < 64 && (i < io.size() || guard == 0); ++guard)
        {
            const std::string a = outcome([&] {
                const auto b = impl::kvt::next_batch(io, i, batch, per_slot, items_cap);
                std::string s = std::to_string(b.kind) + "|" + str(b.from) + "|" + str(b.to) + "|" + str(b.slots) + "|" + str(b.to_slots) + "|";
                for (const auto& x : b.items) s += std::to_string(x.page) + ":" + std::to_string(x.part) + ":" + std::to_string(x.sub) + ":" + std::to_string(x.hslot) + ",";
                return s;
            });
            tx(a + " @" + std::to_string(i));
            if (a.rfind("ok", 0) != 0) break;
        }
    }
    return tx.h;
}

// flash_lend: settings, then polls at rising times with idle on and off, and admissions
inline uint64_t flash_lend(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    for (int t = 0; t < 2000; ++t)
    {
        impl::FlashLend a;
        a.idle_ms = R.U(0, 3) ? R.U(0, 50) : 0;
        a.keep = uint32_t(R.U(0, 5));
        a.chunk_rows = uint64_t(R.U(0, 3) ? R.U(1, 4096) : 0);
        a.chunks = uint32_t(R.U(0, 8));
        auto now = std::chrono::steady_clock::now();   // the rule reads only differences
        for (int k = 0; k < 60; ++k)
        {
            now += std::chrono::microseconds(R.U(0, 20000));
            const bool idle = R.U(0, 4) != 0;
            const auto p = a.poll(idle, now);
            const uint64_t rows = uint64_t(R.U(0, 40000));
            const uint32_t room = uint32_t(R.U(0, 9));
            tx((p ? std::to_string(*p) : "-") + " " + std::to_string(a.need(rows)) + " " + std::to_string(a.recall(rows, room)));
        }
    }
    return tx.h;
}

// flash_ring: steps that follow the accepted rows, and steps that do not; the capture set over its arguments
inline uint64_t flash_ring(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    for (int t = 0; t < 3000; ++t)
    {
        const int sessions = R.U(1, 4), per = R.U(1, 5);
        impl::FlashSlots a(sessions, per);
        std::vector<int> pos(size_t(sessions), 100);
        for (int k = 0; k < 40; ++k)
        {
            const int q = R.U(-1, sessions), n = R.U(0, per + 1);
            const bool verify = R.U(0, 1);
            if (q >= 0 && q < sessions && R.U(0, 3)) pos[size_t(q)] += R.U(0, per + 1);
            else if (q >= 0 && q < sessions) pos[size_t(q)] += R.U(-3, 9);
            const int p = q >= 0 && q < sessions ? pos[size_t(q)] : 0;
            int ks[16] = {};
            const std::string o = outcome([&] { return std::to_string(a.rows(q, p, n, verify, ks)); });
            tx(o + " " + str(std::vector<int>(ks, ks + 16)));
            if (!R.U(0, 9) && q >= 0 && q < sessions) a.reset(q);
        }
        const int s2 = R.U(0, 9), mtp = R.U(0, 7), chunk = R.U(0, 40), mx = R.U(8, 48);
        const bool adapt = R.U(0, 1);
        tx(outcome([&] { const auto f = impl::flash_shapes(s2, mtp, chunk, adapt, mx); return str(f.mains) + "|" + str(f.drafts) + "|" + str(f.chains); }));
    }
    return tx.h;
}

// cp_rows: the CP boundary, the boot value, and rounds over cycles with sessions past R and below it
inline uint64_t cp_rows(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    for (int t = 0; t < 3000; ++t)
    {
        std::vector<int> bk;
        int v = 0;
        for (int k = R.U(1, 8); k > 0; --k) bk.push_back(v += R.U(1, 4096));
        const int r = R.U(-2, 20000);
        tx(std::to_string(impl::cp_after(bk.data(), int(bk.size()), r)));
        const int64_t val = R.U(0, 3) ? R.U(-5, 40) : 0;
        const int solo = R.U(1, 12), full = R.U(1, 40);
        std::string e;
        tx(std::to_string(impl::cp_step_rows(val, solo, full, &e)) + " " + e);
        impl::CpRound a;
        const uint32_t rows = uint32_t(R.U(0, 3) ? R.U(1, 30) : 0), per = uint32_t(R.U(0, 8));
        const bool seg = R.U(0, 1);
        for (int k = 0; k < 30; ++k)
        {
            std::vector<impl::CpCand> c;
            for (int j = R.U(0, 8), id = 0; j > 0; --j) { id += R.U(1, 3); c.push_back({uint64_t(id), R.U(0, 2) == 0}); }
            const auto p = a.pick(c, rows, per, seg);
            std::string s;
            for (bool x : p.in) s += x ? '1' : '0';
            tx(s + " " + std::to_string(p.capped) + std::to_string(p.fresh));
            if (R.U(0, 4)) a.commit(p, c);
            tx(str(a.cycle()));
        }
    }
    return tx.h;
}

// the scheduler through random command streams: every call's outcome and the whole state after it
inline uint64_t scheduler(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    using namespace g6s2;
    Rng R(seed);
    Tx tx{{}, lines};
    auto U = [&](int lo, int hi) { return R.U(lo, hi); };
    for (int t = 0; t < 600; ++t)
    {
        Config c;
        c.epoch = uint64_t(U(0, 20) ? U(1, 3) : 0);
        c.pool_rows = uint32_t(U(0, 20) ? PAGE * U(8, 400) : U(0, 9000));
        c.max_rows = uint32_t(U(1, 40));
        c.max_sessions = uint32_t(U(0, int(MAX_SESSIONS) + 1));
        c.mtp = uint32_t(U(0, 4));
        c.guard_tokens = U(0, 3) ? UINT32_MAX : uint32_t(U(0, 16));
        c.prefill_rows = uint32_t(U(0, 20) ? U(1, 16) * 64 : 0);
        c.two_lanes = U(0, 1);
        c.decode_prefill_rows = uint32_t(U(0, 2) ? 0 : U(0, 3) ? 64 * U(1, 2) : U(1, 100));
        c.decode_rounds = uint32_t(U(0, 2) ? 0 : U(0, 70));
        c.window_pages = uint32_t(U(0, 2) ? 0 : U(1, 4));
        c.cp_step_rows = uint32_t(U(0, 2) ? 0 : U(1, 12));
        c.cp_after = U(0, 1) ? UINT64_MAX : uint64_t(U(0, 4096));
        c.cp_seg = !U(0, 3);
        c.r2_rows = uint32_t(U(0, 2) ? 0 : U(0, 4) ? 64 * U(1, 4) : U(1, 100));
        c.tail_first = U(0, 1);
        c.notice_max = uint32_t(U(0, 4) ? 8 : U(0, 10));
        c.c8_pack = U(0, 1) && (U(0, 6) || g6mod::has(G6_CAP_PACK));
        c.mtp_sessions = uint32_t(U(0, 2) ? 0 : U(1, 4));
        c.park = U(0, 1) && (U(0, 6) || g6mod::has(G6_CAP_PARK));
        const std::string v = outcome([&] { impl::validate(c); return std::string(); });
        tx(v);
        if (v != "ok ") continue;
        impl::Scheduler a(c);
        uint64_t id = 0;
        Quantum work;   // the last quantum next() returned
        for (int k = 0; k < 300; ++k)
        {
            const int op = U(0, 9);
            std::string o;
            if (op <= 2)
            {
                // a commit: maybe a cancel, a grant at the next key, a shrink; the digest right or wrong
                Commit e;
                e.epoch = U(0, 30) ? c.epoch : c.epoch + 1;
                e.event = a.next_event() - uint64_t(!U(0, 30));
                e.boundary = a.boundary() + uint64_t(!U(0, 30));
                e.before_hash = U(0, 30) ? a.digest() : 7;
                const auto& sl = a.slots();
                if (!U(0, 3))
                {
                    e.cancel = sl[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                    if (!U(0, 30)) e.cancel->epoch++;
                }
                if (U(0, 3))
                {
                    std::optional<unsigned> park;
                    if ((c.park || !U(0, 20)) && !U(0, 2)) park = unsigned(U(0, int(MAX_SESSIONS)));
                    Key key{};
                    const std::string ko = outcome([&] { key = a.next_key(e.cancel, park); return str(key); });
                    tx(ko);
                    if (ko.rfind("ok", 0) == 0)
                    {
                        Grant g;
                        g.key = key;
                        if (!U(0, 40)) g.key.epoch++;
                        if (!U(0, 40)) g.key.slot = MAX_SESSIONS + uint32_t(U(0, 2));
                        if (!U(0, 20)) g.key.generation++;
                        g.req.id = U(0, 20) ? (id += uint64_t(U(1, 3))) : id;
                        g.req.prompt_tokens = uint32_t(U(0, 30) ? (U(0, 3) ? U(1, 400) : U(1, 6000)) : 0);
                        g.req.max_tokens = uint32_t(U(0, 30) ? U(1, 60) : 0);
                        g.req.steering.enabled = !U(0, 8);
                        if (g.req.steering.enabled)
                        {
                            auto& st = g.req.steering;
                            st.profile_digest[size_t(U(0, 31))] = uint8_t(U(0, 255));
                            st.schedule_digest[size_t(U(0, 31))] = uint8_t(U(0, 255));
                            st.scope = U(0, 1) ? g6rt::steering::Scope::Shared : g6rt::steering::Scope::Full;
                            st.scale_bits = uint32_t(U(0, 1 << 30));
                            st.slot = uint32_t(U(0, 7));
                        }
                        g.req.r_rows = U(0, 4) ? 0 : c.r2_rows;
                        tx(outcome([&] { return std::to_string(impl::grant_pages(c, g.req)) + "/" + std::to_string(impl::reserve_pages(c, g.req)); }));
                        uint32_t pages = 1;
                        try { pages = U(0, 1) ? impl::grant_pages(c, g.req) : impl::reserve_pages(c, g.req) + (U(0, 2) ? 0 : c.window_pages); } catch (...) {}
                        if (!U(0, 20)) pages += 1;
                        g.run = Run{uint32_t(U(0, int(c.pool_rows / PAGE))), pages, uint32_t(U(0, 20) ? U(1, 9) : 0), uint64_t(U(0, 1 << 20))};
                        g.reuse = uint32_t(U(0, 2) ? 0 : U(0, int(g.req.prompt_tokens) + 1));
                        g.restored = uint32_t(U(0, 3) ? 0 : U(1, 3));
                        g.counted = uint32_t(U(0, 2));
                        tx(outcome([&] { return std::to_string(impl::grant_pages_ok(c, g.req, pages)); }));
                        e.admit = g;
                    }
                }
                if (c.r2_rows && !U(0, 3))
                {
                    const auto& x = sl[size_t(U(0, int(MAX_SESSIONS) - 1))];
                    e.shrink = Shrink{x.key, U(0, 8) ? c.r2_rows : 0, Run{x.run.first_page + uint32_t(!U(0, 4)), x.run.pages, x.run.table_version, 99}};
                    if (!U(0, 20)) e.shrink->key.epoch++;
                    if (!U(0, 20)) e.shrink->key.slot = MAX_SESSIONS;
                }
                e.hold = U(0, 1);
                o = outcome([&] {
                    const Changes d = a.apply(e);
                    std::string s;
                    for (const auto& x : d.freed) s += str(x) + ",";
                    return s + "|" + (d.admitted ? str(d.admitted->key) : "-");
                });
            }
            else if (op == 3)
            {
                std::array<bool, MAX_SESSIONS> stalled{};
                for (auto& x : stalled) x = !U(0, 5);
                o = outcome([&] { const auto q = a.next(stalled); if (q.kind != Kind::Idle) work = q; return str(q); });
                for (int w = 0; w < 2; ++w)
                {
                    const bool want = U(0, 1);
                    tx(str(impl::Scheduler::plan(c, a.slots(), want, stalled)));
                    auto sim = a.slots();
                    const Quantum q = impl::Scheduler::plan(c, sim, want);
                    impl::Scheduler::advance(sim, q);
                    bool wa = want;
                    uint32_t run = uint32_t(U(0, 70));
                    impl::Scheduler::alternate(c, q.kind, wa, run);
                    std::string s = std::to_string(wa) + " " + std::to_string(run) + "|";
                    for (unsigned i = 0; i < MAX_SESSIONS; ++i)
                    {
                        s += std::to_string(sim[i].prefilled) + std::to_string(int(sim[i].phase)) + std::to_string(sim[i].deferred) +
                             std::to_string(sim[i].pack_tail) + " " + std::to_string(impl::prefill_left(sim[i])) + ":";
                        for (unsigned j = 0; j < MAX_SESSIONS; ++j) s += impl::prefill_first(sim[i], sim[j]) ? '1' : '0';
                        s += ",";
                    }
                    tx(s);
                }
            }
            else if (op <= 6)
            {
                // complete the quantum in flight, mostly with valid completions
                const auto& sl = a.slots();
                auto comps = [&](unsigned n, const auto& keys) {
                    std::vector<Completion> v;
                    for (unsigned i = 0; i < n + unsigned(!U(0, 30)); ++i)
                    {
                        const Key key = U(0, 30) ? keys[i % MAX_SESSIONS] : sl[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                        const auto& x = sl[key.slot % MAX_SESSIONS];
                        const uint32_t left = x.req.max_tokens - x.emitted;
                        v.push_back({key, uint32_t(U(0, 30) ? U(0, int(std::min(left, 4u))) : left + 1), !U(0, 6),
                                     x.inserted + uint32_t(U(0, 30) ? U(0, 2) : 0) - uint32_t(!U(0, 40))});
                    }
                    return v;
                };
                const int pick = U(0, 7) ? int(work.kind) : U(1, 5);
                if (pick == int(Kind::Prefill)) o = outcome([&] { a.complete_prefill(); return std::string(); });
                else if (pick == int(Kind::PrefillPack)) o = outcome([&] { a.complete_prefill_pack(); return std::string(); });
                else
                {
                    const auto v = comps(work.count, work.keys);
                    if (pick == int(Kind::PromptTail) && !v.empty()) o = outcome([&] { a.complete_tail(v[0]); return std::string(); });
                    else if (pick == int(Kind::PromptTailPack)) o = outcome([&] { a.complete_tail_pack(v); return std::string(); });
                    else o = outcome([&] { a.complete_decode(v); return std::string(); });
                }
            }
            else if (op == 7)
            {
                const auto x = a.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))];
                const Run g{x.run.first_page + uint32_t(!U(0, 10)), x.run.pages + uint32_t(U(0, 3)), x.run.table_version, uint64_t(U(0, 99))};
                o = outcome([&] { a.grow(x.key, g); return std::string(); });
            }
            else if (op == 8)
            {
                const auto x = a.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))];
                o = outcome([&] { a.skip_prefill(x.key); return std::string(); });
            }
            else
            {
                const uint64_t to = a.boundary() + uint64_t(U(0, 3)) - uint64_t(!U(0, 6));
                o = outcome([&] { a.idle_boundary(to); return std::string(); });
                a.pool_digest(uint64_t(U(0, 9)));
                const auto f = a.find(uint64_t(U(0, 40)));
                std::optional<Key> cancel;
                if (U(0, 1)) cancel = a.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                std::string s = (f ? str(*f) : "-") + "|";
                for (const auto& x : a.reclaimable(cancel)) s += str(x) + ",";
                s += "|";
                for (const auto& x : a.decoding()) s += str(x) + ",";
                tx(s);
                if (!U(0, 200)) a.fail();
            }
            tx(std::to_string(op) + " " + o);
            tx(state(a));
        }
    }
    return tx.h;
}

// share_want over random counter pairs
inline uint64_t share(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    auto W = [&] { return R.U(0, 9) ? uint64_t(R.U(0, 200)) : R.U(0, 1) ? UINT64_MAX - uint64_t(R.U(0, 3)) : R.raw(); };
    for (int t = 0; t < 200000; ++t)
    {
        g6kv_tier_totals a{}, b{};
        for (auto* x : {&a, &b})
        {
            x->slots = W(); x->free_slots = W(); x->entries = uint32_t(R.U(0, 9) ? R.U(0, 20) : int(R.raw() & 0x7fffffff));
            x->spilled_pages = W(); x->dropped_pages = W(); x->moved_out_pages = W(); x->full = W();
            x->oversize = W(); x->oversize_last = W();
        }
        if (R.U(0, 1)) b = a, b.full += uint64_t(R.U(0, 1)), b.spilled_pages += uint64_t(R.U(0, 1)), b.oversize += uint64_t(R.U(0, 1));
        const uint64_t per = R.U(0, 9) ? uint64_t(R.U(0, 16)) : W();
        const bool busy = R.U(0, 1);
        tx(std::to_string(impl::share_want(a, b, per, busy)));
    }
    return tx.h;
}

// the node's answer to a lease report (grant, give) as the Resize it becomes
inline uint64_t share_resize(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    Rng R(seed);
    Tx tx{{}, lines};
    auto W = [&] { return R.U(0, 9) ? uint64_t(R.U(0, 40)) : R.U(0, 1) ? UINT64_MAX - uint64_t(R.U(0, 3)) : R.raw(); };
    for (int t = 0; t < 50000; ++t)
    {
        const uint32_t tier = uint32_t(R.U(0, 3));
        const uint64_t segs = W(), per = W(), grant = R.U(0, 2) ? 0 : W(), give = R.U(0, 2) ? 0 : W(), txn = R.raw();
        g6sch_resize r{};
        const bool on = g6sch_share_resize(tier, segs, per, grant, give, txn, &r);
        tx(on ? std::to_string(r.tier) + " " + std::to_string(r.segs) + " " + std::to_string(r.slots) + " " + std::to_string(r.txn) : "-");
    }
    return tx.h;
}

// the admission queue through random streams
inline uint64_t queue(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    using namespace g6s2;
    Rng R(seed);
    Tx tx{{}, lines};
    for (int t = 0; t < 400; ++t)
    {
        Config c;
        c.pool_rows = PAGE * uint32_t(R.U(1, 64));
        c.mtp = uint32_t(R.U(1, 3));
        impl::AdmissionQueue a(c);
        uint64_t id = 0, rev = 0;
        for (int k = 0; k < 400; ++k)
        {
            const int op = R.U(0, 5);
            std::string o;
            if (op <= 1)
            {
                Request q;
                q.id = R.U(0, 15) ? (id += uint64_t(R.U(1, 3))) : id - uint64_t(R.U(0, 1));
                q.prompt_tokens = uint32_t(R.U(0, 30) ? R.U(1, 3000) : 0);
                q.max_tokens = uint32_t(R.U(0, 30) ? R.U(1, 900) : 0);
                o = outcome([&] { a.push(q); return std::string(); });
            }
            else if (op == 2) o = std::to_string(a.cancel(uint64_t(R.U(0, int(id) + 2))));
            else if (op == 3)
            {
                const uint64_t x = a.front() && R.U(0, 9) ? a.front()->id : uint64_t(R.U(0, int(id) + 2));
                o = outcome([&] { a.admitted(x); return std::string(); });
            }
            else
            {
                if (R.U(0, 2)) rev += uint64_t(R.U(0, 1));
                const bool slot = R.U(0, 3), fits = R.U(0, 3);
                const uint32_t room = uint32_t(R.U(0, 9) ? R.U(0, int(c.pool_rows / PAGE)) : R.U(0, 200));
                o = outcome([&] {
                    const auto x = a.check(rev, slot, fits, room);
                    return x ? std::to_string(int(x->why)) + "/" + std::to_string(x->requested) + "/" + std::to_string(x->room) : std::string("-");
                });
            }
            const auto* f = a.front();
            tx(o + " " + std::to_string(a.size()) + " " + std::to_string(a.stalls()) + " " +
               (f ? std::to_string(f->id) + "/" + std::to_string(f->prompt_tokens) + "/" + std::to_string(f->max_tokens) : "-"));
        }
    }
    return tx.h;
}

// the request parser over generated request files, well formed and mutated; ready: the engine has a grammar loaded
// (otherwise every gram request stops at that refusal)
inline uint64_t request(uint64_t seed, std::vector<std::string>* lines = nullptr, bool ready = false)
{
    Rng R(seed);
    Tx tx{{}, lines};
    auto U = [&](int lo, int hi) { return R.U(lo, hi); };
    auto pick = [&](const std::vector<std::string>& v) { return v[size_t(U(0, int(v.size()) - 1))]; };
    namespace st = g6rt::steering;
    st::BootConfig off, one, forced;
    one.loaded = true;
    one.profiles.push_back({"calm", {}, {}, 2});
    for (unsigned i = 0; i < 32; ++i) { one.profiles[0].profile_digest[i] = uint8_t(i); one.profiles[0].schedule_digest[i] = uint8_t(63 - i); }
    forced = one;
    forced.forced = true;
    forced.forced_scale_bits = 0x3f800000u;
    const st::BootConfig* boots[] = {&off, &one, &forced};
    const std::vector<std::string> nums = {"0", "1", "7", "42", "4096", "2147483647", "2147483648", "4294967295", "4294967296",
                                           "18446744073709551615", "18446744073709551616", "", "-1", "+3", "1a", "007"};
    // the C++ read these with libstdc++'s num_get: a number ends at the first character it cannot take
    const std::vector<std::string> floats = {"0", "0.5", "1", "1.0", "0.25", "2", "1e-3", "0.9999999", "1.00000001", "-0", ".5", "5.",
                                             "+0.5", "1e", "1e+", "1e40", "1e-50", "-1", "nan", "inf", "0x1p3", "0.5seed", ".", "-",
                                             "1.2.3", "1e5e5", "00000.5", "3E2", "1,5", "1e+5", "1e-5", "1.e5", "0e5", "00e5", ".e5",
                                             "+-5", "1e-", "-.5", "+.", "0.30000000000000000000000001", "1e-40", "9e-46",
                                             "3.40282356e38", "3.4028236e38", "0." + std::string(70000, '0') + "1e70001",
                                             "1" + std::string(70000, '0') + "e-70001"};
    const std::vector<std::string> ints = {"0", "1", "5", "20", "21", "-1", "2147483647", "2147483648", "+4", "-0", "0x5", "5x", "",
                                           "99999999999999999999", "-9223372036854775809", "1.5", "-2147483648", "-2147483649",
                                           "9223372036854775807", "-9223372036854775808", "0x", "00", "+0", "1e5"};
    auto ones = [](int n) { std::string s = "1"; while (--n > 0) s += ",1"; return s; };   // one past INSERT_MAX at 65
    const std::vector<std::string> idlists = {"1", "1,2,3", "0,2147483647", "2147483648", "1,,2", ",1", "1,", "", "x", "5,6,7,8"};
    for (int t = 0; t < 60000; ++t)
    {
        std::vector<std::string> w = {"n_gen", pick(nums), "temp", pick(floats), "seed", pick(nums), "min", pick(nums), "stop",
                                      U(0, 1) ? "-" : pick(idlists)};
        if (U(0, 2)) w.insert(w.end(), {"top_p", pick(floats)});
        if (!U(0, 2)) w.insert(w.end(), {"lp", pick(ints)});
        if (!U(0, 2)) w.insert(w.end(), {"plp", pick(ints)});
        if (!U(0, 2)) w.insert(w.end(), {"sys", pick(ints)});
        if (!U(0, 3)) w.insert(w.end(), {"ins", pick({"3:-1:8,9", "3:7:8", "3::8", "3:-1:", "x:-1:1", "2147483648:-1:1", "3:-1",
                                                     "3:99999999999:1", "1:2:" + ones(65)})});
        if (!U(0, 3)) w.insert(w.end(), {"gram", pick({"1:0:-1:", "2:1:5:7b7d", "2:0:-1:7B22", "2:0:-1:7", "2:0:-1:zz", "3:0:-1:", "1:2:-1:",
                                                       "1:0:-1", "1:0:x:", "2:1:99999999999:7b7d", "1:0:-1:7b"})});
        if (!U(0, 3))
        {
            w.push_back("steer");
            w.push_back(pick({"calm", "other", "-bad", "a.b_c-d", "", std::string(128, 'a'), std::string(129, 'a'), "caf\xc3\xa9"}));
            w.push_back(pick({"3f800000", "00000000", "80000000", "40800000", "40800001", "7f800000", "3F800000", "3f8000", "bf800000"}));
        }
        // mutations: a dropped, swapped, doubled or glued word, a stray word at the end
        for (int m = U(0, 3) ? 0 : U(1, 3); m > 0 && !w.empty(); --m)
        {
            const size_t i = size_t(U(0, int(w.size()) - 1));
            switch (U(0, 4))
            {
            case 0: w.erase(w.begin() + long(i)); break;
            case 1: std::swap(w[i], w[size_t(U(0, int(w.size()) - 1))]); break;
            case 2: w.insert(w.begin() + long(i), w[i]); break;
            case 3: if (i + 1 < w.size()) { w[i] += w[i + 1]; w.erase(w.begin() + long(i) + 1); } break;
            default: w.push_back(pick({"extra", "top_p", "lp", "0"})); break;
            }
        }
        std::string head;
        for (const auto& x : w) head += (head.empty() ? "" : pick({" ", " ", " ", "  ", "\t", "\r", "\v", "\f"})) + x;
        if (!U(0, 20)) head = pick({" ", "\t"}) + head;
        std::string bytes = head;
        if (U(0, 30)) bytes += "\n" + pick({"1,2,3", "1", "", "1 2", "1,2\n", " 7 ", "2147483648", "1,,2", "1\n\n"});
        if (!U(0, 4)) bytes += "\n";
        if (!U(0, 40))
        {
            // one draw per statement: the order of a call's arguments is the compiler's choice
            const size_t at = size_t(U(0, int(bytes.size())));
            bytes.insert(at, 1, pick({std::string(1, '\0'), "\xff", "\xa0", "\x85"})[0]);
        }
        const st::BootConfig& b = *boots[U(0, 2)];
        tx(outcome([&] { return str(impl::parse_request(bytes, b, ready)); }));
    }
    for (const char* g : {"1:0:-1:", "2:1:5:7b7d", "2:0:-1:7", "x", "1:0", "2:0:-1:zZ", "2:1:-1:7B7d"})
        tx(outcome([&] {
            const auto x = impl::parse_grammar(g);
            return std::to_string(int(x.kind)) + std::to_string(x.strict) + std::to_string(x.arm) + x.schema;
        }));
    return tx.h;
}

// the driver (driver.h): Drivers over a fake engine that answers from a hash of each row, taking grants (decode-prompt
// requests and insert rules among them), cancels and Resize commits between steps, with runs that grow, stall or stay,
// rounds capped past R, emitters that stop early or miscount, and hooks that throw. Every hook call, step, refusal and
// digest goes in. The planner's own choices are the planner tests'; this pins the driver's calls into it.
inline uint64_t driver(uint64_t seed, std::vector<std::string>* lines = nullptr)
{
    using namespace g6s2;
    Rng R(seed);
    Tx tx{{}, lines};
    auto U = [&](int lo, int hi) { return R.U(lo, hi); };
    auto mix = [](uint64_t a, uint64_t b, uint64_t c) {
        uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 7) * 0x94D049BB133111EBull;
        x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
        return x;
    };
    for (int t = 0; t < 60; ++t)
    {
        Config c;
        c.max_sessions = uint32_t(U(1, int(std::min(MAX_SESSIONS, 8u))));
        c.mtp = uint32_t(U(1, 3));
        c.max_rows = uint32_t(U(int(c.mtp) + 1, 32));
        c.pool_rows = PAGE * 4096;
        c.prefill_rows = uint32_t(64 * U(1, 4));
        c.two_lanes = U(0, 1);
        c.decode_rounds = uint32_t(U(0, 2) ? 0 : U(1, 4));
        c.cp_step_rows = uint32_t(U(0, 1) ? 0 : U(int(c.mtp) + 1, 12));
        c.cp_after = U(0, 3) ? uint64_t(U(0, 300)) : UINT64_MAX;
        c.cp_seg = !U(0, 3);
        c.tail_first = U(0, 1);
        c.mtp_sessions = uint32_t(U(0, 2) ? 0 : U(1, 3));
        const std::string v = outcome([&] { impl::validate(c); return std::string(); });
        tx(v);
        if (v != "ok ") continue;
        const uint64_t salt = R.raw();
        const int throw_rate = U(0, 1) ? 0 : 400;   // per hook call, 1 in throw_rate throws
        const bool with_grow = U(0, 3), with_resize = U(0, 3);
        std::unique_ptr<impl::Driver> d;
        std::map<uint64_t, uint32_t> sent;   // request id -> tokens the emitter let out
        auto hook = [&](const std::string& s) {
            tx(s);
            if (throw_rate && !U(0, throw_rate - 1)) throw std::runtime_error("hook threw at " + s);
        };
        DriverHooks h;
        h.bind = [&](const Grant& g) { hook("bind " + str(g.key) + " " + str(g.run) + " id " + std::to_string(g.req.id)); };
        h.retire = [&](const Slot& s) {
            hook("retire " + str(s.key) + " phase " + std::to_string(int(s.phase)) + " emitted " + std::to_string(d->emitted(s.key.slot).size()) +
                 " prompt " + std::to_string(d->prompt(s.key.slot).size()));
        };
        h.prefill = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
            hook("prefill " + str(s.key) + " " + std::to_string(p.prompt.size()) + " " + std::to_string(first) + "+" + std::to_string(rows));
        };
        h.run = [&](const s2::Step& st) {
            std::string s = "run " + st.key() + " d" + std::to_string(st.scheduler_decode) + " e" + std::to_string(int(st.event));
            std::vector<int> ids;
            for (const auto& r : st.r)
            {
                ids.push_back(int(mix(salt + uint64_t(r.sess), uint64_t(r.pos) + (st.kind != s2::Step::MAIN), uint64_t(r.tok)) % 4));
                s += " " + std::to_string(r.sess) + ":" + std::to_string(r.pos) + ":" + std::to_string(r.tok);
            }
            hook(s);
            return ids;
        };
        h.emit = [&](Key k, const int* ids, int n, int row) {
            const Slot& s = d->state().slots()[k.slot];
            uint32_t& out = sent[s.req.id];
            const uint32_t left = s.req.max_tokens - out;
            Emission e{uint32_t(n), false};
            const int r = U(0, 40);
            if (r == 0) e = {uint32_t(U(0, n)), true};   // a stop
            if (e.count > left) e = {left, true};
            if (r == 1 && !throw_rate) e.count = uint32_t(n) + 1;   // a miscount: refused
            out += std::min(e.count, left);
            hook("emit " + str(k) + " " + std::to_string(n) + " row " + std::to_string(row) + " first " + std::to_string(n ? ids[0] : -1) +
                 " -> " + std::to_string(e.count) + (e.finished ? " stop" : ""));
            return e;
        };
        h.prompt = [&](Key k, int M, int pos0) { hook("prompt " + str(k) + " " + std::to_string(M) + " " + std::to_string(pos0)); };
        h.round = [&](Key k, long n, int base, int a, int K) {
            hook("round " + str(k) + " " + std::to_string(n) + " " + std::to_string(base) + " " + std::to_string(a) + "/" + std::to_string(K));
        };
        h.dfl_ok = [&](Key k) { hook("dfl_ok " + str(k)); return false; };
        h.terminal = [&](Key k, const s2::Stats& st) {
            hook("terminal " + str(k) + " " + std::to_string(st.rounds) + " " + std::to_string(st.tokens) + " " + std::to_string(st.inserted));
        };
        if (with_grow)
            h.grow = [&](const Slot& s, uint64_t rows) -> std::optional<Run> {
                hook("grow " + str(s.key) + " " + std::to_string(rows) + " " + str(s.run));
                const int r = U(0, 5);
                if (r == 0) return std::nullopt;
                const uint32_t most = reserve_pages(c, s.req);
                if (r == 1 || s.run.pages >= most) return s.run;
                return Run{s.run.first_page, std::min(most, s.run.pages + uint32_t(U(1, 3))), s.run.table_version, R.raw()};
            };
        h.finished = [&](const Slot& s) { hook("finished " + str(s.key) + " " + std::to_string(s.emitted)); };
        if (with_resize) h.resize = [&](const Resize& z) { hook("resize " + std::to_string(z.tier) + " " + std::to_string(z.segs)); };
        h.settle = [&] { hook("settle"); };
        const std::string made = outcome([&] { d = std::make_unique<impl::Driver>(c, h); return std::string(); });
        tx(made);
        if (!d) continue;
        uint64_t id = 0;
        int refused = 0;
        for (int k = 0; k < 400 && refused < 3; ++k)
        {
            const int op = U(0, 9);
            std::string o;
            if (op <= 1)
            {
                const auto& sl = d->state().slots();
                Commit e;
                e.epoch = c.epoch;
                e.event = d->state().next_event();
                e.boundary = d->state().boundary();
                e.before_hash = d->state().digest();
                std::optional<Payload> p;
                const int kind = U(0, 9);
                if (kind <= 2)
                {   // cancel a live session, or a key that is not one
                    const auto& s = sl[unsigned(U(0, int(MAX_SESSIONS) - 1))];
                    if (s.phase != Phase::Free || !U(0, 3)) e.cancel = s.key;
                }
                if (kind == 3 && U(0, 1)) e.resize = Resize{uint32_t(U(0, 3)), uint64_t(U(0, 9)), 0, 0};
                else if (kind >= 3 && d->state().resident() < c.limit())
                {
                    Request r;
                    r.id = ++id;
                    r.prompt_tokens = uint32_t(U(0, 4) ? U(2, 200) : U(201, 600));
                    r.max_tokens = uint32_t(U(1, 80));
                    Payload q;
                    for (uint32_t i = 0; i < r.prompt_tokens; ++i) q.prompt.push_back(U(0, 3));
                    q.chunk = U(0, 20) ? U(1, 8) : 0;   // 0: refused
                    q.decode_prompt = !U(0, 6);
                    if (!U(0, 3))
                    {
                        q.ins.after = U(0, 20);
                        q.ins.unless = U(0, 1) ? -1 : U(0, 3);
                        for (int i = U(1, 6); i > 0; --i) q.ins.tokens.push_back(U(0, 3));
                    }
                    const std::string key = outcome([&] { e.admit = Grant{d->state().next_key(e.cancel), r, {}, 0}; return std::string(); });
                    if (e.admit)
                    {
                        e.admit->run = Run{uint32_t(U(0, 1000)), grant_pages(c, r), 1, R.raw()};
                        if (!U(0, 30)) q.prompt.pop_back();   // a payload that does not match its grant
                        p = q;
                    }
                    else tx("next_key " + key);
                }
                o = "apply " + outcome([&] {
                    const Changes ch = d->apply(e, p ? &*p : nullptr);
                    std::string s;
                    for (const auto& r : ch.freed) s += str(r) + ",";
                    return s + (ch.admitted ? " admitted " + str(ch.admitted->key) : "");
                });
            }
            else o = "step " + outcome([&] { return str(d->step()); });
            const bool dead = o.find("failed or work in flight") != std::string::npos || o.find("quiescent") != std::string::npos;
            refused = dead ? refused + 1 : 0;
            tx(o);
            tx("digest " + std::to_string(d->state().digest()) + " cp " + std::to_string(d->cp_capped_rounds()));
        }
    }
    return tx.h;
}
