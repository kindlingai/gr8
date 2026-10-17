// Config::decode_rounds (s2/scheduler.h, Scheduler::alternate; G6_S2_DECODE_ROUNDS): while a prefill waits and sessions
// decode, K decode rounds run per prefill quantum instead of one. The properties here: the step pattern is exactly K
// rounds between quanta (one with the knob 0 or 1, as before); pure decode never idles and a lone prefill is never
// delayed; the notice target (horizon) matches an independent re-derivation at every step and two schedulers driven by
// the same events agree on it and on the digest; 0 and 1 are the build before the knob (same steps, digest, horizon,
// bootstrap agreement); a knob past MAX_DECODE_ROUNDS is refused; and the agreement names the knob so ranks booted with
// different K refuse each other. Host only: no GPU, no CUDA, no pool. Build and run: see scripts/host_tests.sh.
#include "s2/bootstrap.h"
#include "s2/scheduler.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static Config cfg(uint32_t rounds)
{
    Config c;
    c.pool_rows = 614400; c.max_rows = 24; c.max_sessions = 8; c.mtp = 2;
    c.prefill_rows = 4096; c.two_lanes = true; c.decode_rounds = rounds;
    return c;
}

static void admit(Scheduler& s, uint64_t id, uint32_t prompt, uint32_t max_tokens)
{
    const Config& c = s.config();
    const Request r{id, prompt, max_tokens};
    Grant g;
    g.key = s.next_key();
    g.req = r;
    g.run = Run{0, grant_pages(c, r), 1, id * 0x9E3779B97F4A7C15ull + 1};
    Commit cm;
    cm.epoch = c.epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest(); cm.admit = g;
    s.apply(cm);
}

static Quantum step(Scheduler& s)
{
    const Quantum q = s.next();
    if (q.kind == Kind::Prefill) s.complete_prefill();
    else if (q.kind == Kind::PromptTail) s.complete_tail(Completion{q.keys[0], 1, false, 0});
    else if (q.kind == Kind::Decode) {
        std::vector<Completion> out;
        for (unsigned i = 0; i < q.count; ++i) out.push_back(Completion{q.keys[i], 1, false, 0});
        s.complete_decode(out);
    }
    return q;
}

// The alternation as the rule states it, written here apart from Scheduler::alternate so a slip there shows up
struct Alt {
    bool want = true; uint32_t run = 0;
    void after(const Config& c, Kind k) {
        if (k == Kind::Decode) { ++run; want = run < std::max<uint32_t>(c.decode_rounds, 1); }
        else { run = 0; want = true; }
    }
};

static uint64_t expected_horizon(const Scheduler& s, Alt a)
{
    auto sim = s.slots();
    const Config& c = s.config();
    const Quantum inflight = Scheduler::plan(c, sim, a.want);
    if (inflight.kind == Kind::Idle) return s.boundary() + NOTICE_MIN;
    Scheduler::advance(sim, inflight);
    a.after(c, inflight.kind);
    uint32_t covered = 0;
    for (uint64_t k = NOTICE_MIN; k < NOTICE_MAX; ++k) {
        const Quantum q = Scheduler::plan(c, sim, a.want);
        if (q.kind == Kind::Idle) return s.boundary() + k;
        Scheduler::advance(sim, q);
        a.after(c, q.kind);
        covered += q.kind == Kind::Prefill ? q.rows : q.kind == Kind::PromptTail ? 1 : 0;
        if (covered >= NOTICE_ROWS) return s.boundary() + k;
    }
    return s.boundary() + NOTICE_MAX;
}

// A short chat decoding while a long prompt prefills: between two of the long prompt's quanta run exactly
// max(K, 1) decode rounds. Before the short one decodes, the long prompt's quanta run back to back (no decoder).
static void pattern()
{
    for (uint32_t K : {0u, 1u, 2u, 4u, 8u, 16u}) {
        Scheduler s(cfg(K));
        admit(s, 1, 98304, 16);                                 // 12 full quanta
        const Quantum q0 = step(s);
        CHECK(q0.kind == Kind::Prefill && q0.keys[0].slot == 0);
        CHECK(step(s).kind == Kind::Prefill);                  // still alone: no decode round between its quanta
        admit(s, 2, 100, 100000);                               // fewest rows first: it prefills, tails, then decodes
        Alt a; a.after(s.config(), Kind::Prefill);
        unsigned run = 0, gaps = 0; bool seen_decode = false, ok = true;
        for (unsigned i = 0; i < 400 && s.slots()[0].phase == Phase::Prefill; ++i) {
            CHECK(s.horizon() == expected_horizon(s, a));
            const Quantum q = step(s);
            a.after(s.config(), q.kind);
            if (q.kind == Kind::Decode) { ++run; seen_decode = true; }
            else if (q.kind == Kind::Prefill && q.keys[0].slot == 0 && seen_decode) {
                ok &= run == std::max(K, 1u); ++gaps; run = 0;
            }
        }
        CHECK(ok);
        CHECK(gaps >= 8);
        std::printf("scheduler_rounds_test: K=%u: %u quanta of the long prompt, %u decode round(s) before each\n",
                    K, gaps, std::max(K, 1u));
    }
}

// pure decode with the knob on never idles; a prefill admitted after many rounds goes at once (the count is past K)
static void pure_decode_then_prefill()
{
    Scheduler s(cfg(4));
    admit(s, 1, 64, 100000);
    for (unsigned i = 0; i < 3; ++i) (void)step(s);
    for (unsigned i = 0; i < 20; ++i) CHECK(step(s).kind == Kind::Decode);
    admit(s, 2, 32768, 16);
    const Quantum q = step(s);
    CHECK(q.kind == Kind::Prefill && q.keys[0].slot == 1);
    for (unsigned i = 0; i < 4; ++i) CHECK(step(s).kind == Kind::Decode);
    CHECK(step(s).kind == Kind::Prefill);
}

struct Rng {
    uint64_t s = 1;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; }
    uint32_t range(uint32_t lo, uint32_t hi) { return lo + uint32_t(next() % (uint64_t(hi) - lo + 1)); }
};

// Randomized: horizon == the independent re-derivation at every step; two schedulers with the same events agree on the
// horizon and the digest; and a K=0 and a K=1 scheduler (the build before the knob) take the same steps with the same
// digest and horizon throughout.
static void randomized_agrees()
{
    for (uint64_t seed = 1; seed <= 60; ++seed) {
        Rng rng{seed * 0x9E3779B97F4A7C15ull + 7};
        Config c = cfg(rng.range(0, 16));
        c.prefill_rows = rng.range(16, 64) * PAGE;
        c.two_lanes = (rng.next() & 1) != 0;
        if (rng.next() & 1) c.decode_prefill_rows = rng.range(1, c.prefill_rows / PAGE) * PAGE;
        Config off0 = c, off1 = c; off0.decode_rounds = 0; off1.decode_rounds = 1;
        Scheduler a(c), b(c), z(off0), o(off1);
        Alt alt;
        uint64_t next_id = 1;
        auto admit_all = [&](uint32_t prompt, uint32_t tokens) {
            for (Scheduler* x : {&a, &b, &z, &o}) admit(*x, next_id, prompt, tokens);
            ++next_id;
        };
        for (unsigned i = 0, n = 1 + unsigned(rng.range(0, 3)); i < n; ++i) admit_all(rng.range(1, 40000), rng.range(16, 300));
        const unsigned steps = rng.range(1, 400);
        for (unsigned i = 0; i < steps; ++i) {
            CHECK(a.horizon() == expected_horizon(a, alt));
            CHECK(a.horizon() == b.horizon() && a.digest() == b.digest());
            CHECK(z.horizon() == o.horizon() && z.digest() == o.digest());
            const Quantum qa = step(a);
            CHECK(step(b).kind == qa.kind);
            const Quantum qz = step(z), qo = step(o);
            CHECK(qz.kind == qo.kind && qz.rows == qo.rows);
            alt.after(c, qa.kind);
            if (!a.resident() && !z.resident()) break;
            if (rng.range(0, 30) == 0 && next_id <= 8 && a.resident() < 4 && z.resident() < 4)
                admit_all(rng.range(1, 40000), rng.range(16, 300));
        }
    }
}

static void validate_and_agreement()
{
    for (uint32_t K : {0u, 1u, 16u, MAX_DECODE_ROUNDS}) { bool threw = false; try { cfg(K).validate(); } catch (...) { threw = true; } CHECK(!threw); }
    { bool threw = false; try { cfg(MAX_DECODE_ROUNDS + 1).validate(); } catch (...) { threw = true; } CHECK(threw); }
    // 0 and 1: the agreement and the boot digest of the build before the knob; K > 1 names it and changes the digest
    auto agreement = [](uint32_t K) { BootstrapConfig b; b.scheduler = cfg(K); b.runtime_signature = "x"; return b.agreement(); };
    CHECK(agreement(0) == agreement(1));
    CHECK(agreement(0).find(" dr ") == std::string::npos);
    CHECK(agreement(4).find(" dr 4 ctl ") != std::string::npos);
    CHECK(agreement(4) != agreement(8));
    CHECK(Scheduler(cfg(0)).digest() == Scheduler(cfg(1)).digest());
    CHECK(Scheduler(cfg(0)).digest() != Scheduler(cfg(4)).digest());
}

int main()
{
    pattern();
    pure_decode_then_prefill();
    randomized_agrees();
    validate_and_agreement();
    if (fails) { std::fprintf(stderr, "scheduler_rounds_test: %d failure(s)\n", fails); return 1; }
    std::printf("scheduler_rounds_test: PASS\n");
    return 0;
}
