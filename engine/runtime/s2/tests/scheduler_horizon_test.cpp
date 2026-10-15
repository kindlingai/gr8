// The notice target (s2/scheduler.h, Scheduler::horizon; service.h event): the first boundary whose work after the one
// in flight covers NOTICE_ROWS prefill rows, within [boundary + NOTICE_MIN, boundary + NOTICE_MAX]. The constant it
// replaces (LOOKAHEAD = 8 boundaries) was the margin a decode round needs, and during a prefill eight ~6 s quanta for a
// propagation that takes milliseconds -- which is why a cancel could never land inside a below-R prefill: R is exactly
// eight 8,192-row quanta at the cluster's line (#61). The properties here: pure decode keeps the constant's behaviour
// exactly (the cap); a prefill in flight targets the floor; the alternation with a decoding session and a partial last
// quantum are both accounted; a quantum the size of NOTICE_ROWS covers; and the target two schedulers driven by the
// same events name is the same boundary at every step of a randomized run. Host only (scheduler.h is plain C++): no
// GPU, no CUDA, no pool. Build and run: see scripts/host_tests.sh.
#include "s2/scheduler.h"
#include <cstdio>
#include <utility>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static Config cfg()
{
    Config c;
    c.pool_rows = 614400; c.max_rows = 24; c.max_sessions = 8; c.mtp = 2;
    c.prefill_rows = 4096; c.two_lanes = true;
    return c;
}

// A grant the scheduler accepts: the pages the request needs at admission (runs grow on demand, driver.h), on a slot
// next_key picks, which is the same slot every rank picks. No pool is involved: scheduler.h owns the choice under test.
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

// one model step: the quantum next() names, completed as the driver would (one token per round, so sessions stay live)
static Kind step(Scheduler& s)
{
    const Quantum q = s.next();
    if (q.kind == Kind::Prefill) s.complete_prefill();
    else if (q.kind == Kind::PromptTail) s.complete_tail(Completion{q.keys[0], 1, false, 0});
    else if (q.kind == Kind::Decode) {
        std::vector<Completion> out;
        for (unsigned i = 0; i < q.count; ++i) out.push_back(Completion{q.keys[i], 1, false, 0});
        s.complete_decode(out);
    }
    return q.kind;
}

// the alternation flag horizon() starts from: true at boot, and each completed step sets it (Scheduler::completed).
// apply() does not touch it, so a caller that drives the steps knows it exactly.
static bool want_after(Kind k) { return k != Kind::Decode; }

// The target the rule names, re-derived from the slots through the public plan()/advance(): the same projection
// horizon() runs, driven independently, so a wrong slot index or a missed deferral update in horizon() shows up as a
// mismatch. The hand-computed cases below are what pin the rule itself.
static uint64_t expected_horizon(const Scheduler& s, bool want)
{
    auto sim = s.slots();
    const Config& c = s.config();
    const Quantum inflight = Scheduler::plan(c, sim, want);
    if (inflight.kind == Kind::Idle) return s.boundary() + NOTICE_MIN;
    Scheduler::advance(sim, inflight);
    want = inflight.kind != Kind::Decode;
    uint32_t covered = 0;
    for (uint64_t k = NOTICE_MIN; k < NOTICE_MAX; ++k) {
        const Quantum q = Scheduler::plan(c, sim, want);
        if (q.kind == Kind::Idle) return s.boundary() + k;
        Scheduler::advance(sim, q);
        want = q.kind != Kind::Decode;
        covered += q.kind == Kind::Prefill ? q.rows : q.kind == Kind::PromptTail ? 1 : 0;
        if (covered >= NOTICE_ROWS) return s.boundary() + k;
    }
    return s.boundary() + NOTICE_MAX;
}

static void check_invariants(const Scheduler& s, bool want, const char* where)
{
    const uint64_t delta = s.horizon() - s.boundary();
    CHECK(delta >= NOTICE_MIN && delta <= NOTICE_MAX);          // the clamp the head applies, before the clamp
    CHECK(s.horizon() == expected_horizon(s, want));            // the rule, re-derived through plan()
    (void)where;
}

// An idle engine cannot depart any boundary (it blocks in wait_notice), so the floor is safe and the target is minimal.
static void idle_targets_the_floor()
{
    Scheduler s(cfg());
    CHECK(s.horizon() == s.boundary() + NOTICE_MIN);
}

// Pure decode is the case the constant 8 was calibrated for: every boundary is a tens-of-milliseconds round, a round
// adds no coverage (stalls can shrink one to almost nothing), so the target is the cap -- exactly LOOKAHEAD's behaviour.
static void decode_keeps_the_constant()
{
    Scheduler s(cfg());
    admit(s, 1, 64, 512);
    bool want = true;
    for (unsigned i = 0; i < 6; ++i) want = want_after(step(s));   // its quantum, tail, then decode rounds
    for (unsigned i = 0; i < 20; ++i) {
        check_invariants(s, want, "decode");
        CHECK(s.horizon() == s.boundary() + NOTICE_MAX);
        want = want_after(step(s));
    }
}

// #61's prompt: 31,990 tokens, four quanta (8,192 x 3 + 7,413) and a tail. While a full quantum sits beyond the one in
// flight the target is the floor; the trace's cancel, anchored at the admit's commit boundary, targets boundary + 2 --
// the boundary the measured review named as where it should have applied -- and once only the tail and decode rounds
// remain the target opens to the cap, because that work is fast.
static void long_prefill_targets_the_floor()
{
    Scheduler s(cfg());
    admit(s, 1, 31990, 512);
    bool want = true;
    for (const uint64_t floor : {2ull, 2ull, 2ull}) {
        check_invariants(s, want, "prefill");
        CHECK(s.horizon() == s.boundary() + floor);
        want = want_after(step(s));
    }
    for (unsigned i = 0; i < 30; ++i) {                            // the last quantum, the tail, then decode
        check_invariants(s, want, "prefill end");
        CHECK(s.horizon() == s.boundary() + NOTICE_MAX);
        want = want_after(step(s));
    }
}

// A decoding session between quanta: the alternation inserts a decode round, which adds no coverage, so the target
// moves one boundary out after a round and back to the floor after a quantum. Counting that round as coverage would
// name a boundary the round's stalls could shrink past.
static void alternation_is_counted()
{
    Scheduler s(cfg());
    admit(s, 1, 31990, 512);
    bool want = true;
    while (true) {                                                 // the long prompt to decode
        const Kind k = step(s);
        want = want_after(k);
        if (k == Kind::PromptTail) break;
    }
    admit(s, 2, 31990, 16);                                        // a second long prompt behind it
    for (unsigned i = 0; i < 6; ++i) {                             // D,Q,D,Q,D,Q: the round between quanta is counted
        check_invariants(s, want, "alternation");
        const Kind k = step(s);
        CHECK(s.horizon() == s.boundary() + (k == Kind::Prefill ? NOTICE_MIN : NOTICE_MIN + 1));
        want = want_after(k);
    }
    for (unsigned i = 0; i < 3; ++i) {   // steps 7-9: the last quantum is partial and the tail one row, so the work
        const Kind k = step(s);          // left beyond the one in flight is fast, and the target opens to the cap
        want = want_after(k);
        check_invariants(s, want, "alternation end");
        CHECK(s.horizon() == s.boundary() + NOTICE_MAX);
    }
}

// The coverage threshold is NOTICE_ROWS exactly: a prompt whose second quantum is 2,048 rows covers, one whose is
// 2,047 needs the tail's row as well, and a prompt whose remainder is 599 rows does not cover at all -- that work is
// fast, and the target correctly opens to the cap.
static void threshold_is_notice_rows()
{
    for (const auto& [prompt, delta] : std::vector<std::pair<uint32_t, uint64_t>>{
             {10241, NOTICE_MIN}, {10240, NOTICE_MIN + 1}, {8792, NOTICE_MAX}}) {
        Scheduler s(cfg());
        admit(s, 1, prompt, 16);
        check_invariants(s, true, "threshold");
        CHECK(s.horizon() == s.boundary() + delta);
    }
}

// The configuration the discrimination run shrank to (--prefill 2048 --pf-lanes 1): a quantum is exactly NOTICE_ROWS
// and still covers, so the same prompt that could not be interrupted at 8,192 rows can be at 2,048.
static void small_quantum_config()
{
    Config c = cfg();
    c.prefill_rows = 2048; c.two_lanes = false;
    Scheduler s(c);
    admit(s, 1, 32768, 16);
    for (unsigned i = 0; i < 14; ++i) {
        check_invariants(s, true, "small quantum");
        CHECK(s.horizon() == s.boundary() + NOTICE_MIN);
        (void)step(s);
    }
}

// The smallest RNG the battery needs, so the run is reproducible from the seed alone.
struct Rng {
    uint64_t s = 1;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; }
    uint32_t range(uint32_t lo, uint32_t hi) { return lo + uint32_t(next() % (uint64_t(hi) - lo + 1)); }
};

// Randomized: the clamp holds and the rule matches its re-derivation at every boundary, over varied configs, prompts
// and mid-run admissions; and two schedulers driven by the same events name the same target throughout -- the property
// the head relies on, since it publishes one horizon for the notice every rank applies.
static void randomized_agrees()
{
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        Rng rng{seed * 0x9E3779B97F4A7C15ull + 1};
        Config c = cfg();
        c.prefill_rows = rng.range(1024, 4096);
        c.two_lanes = (rng.next() & 1) != 0;
        Scheduler a(c), b(c);
        bool want = true;
        uint64_t next_id = 1;
        for (unsigned i = 0, admits = 1 + unsigned(rng.range(0, 2)); i < 3 && admits; ++i, --admits) {
            const uint32_t prompt = rng.range(1, 40000), tokens = rng.range(16, 96);
            admit(a, next_id, prompt, tokens); admit(b, next_id, prompt, tokens); ++next_id;
        }
        const unsigned steps = rng.range(1, 250);
        for (unsigned i = 0; i < steps; ++i) {
            check_invariants(a, want, "randomized");
            CHECK(a.horizon() == b.horizon());
            const Kind k = step(a);
            CHECK(step(b) == k);
            want = want_after(k);
            if (!a.resident() && rng.range(0, 8) == 0 && next_id <= 6) {   // refill an emptied engine
                const uint32_t prompt = rng.range(1, 40000), tokens = rng.range(16, 96);
                admit(a, next_id, prompt, tokens); admit(b, next_id, prompt, tokens); ++next_id;
            }
        }
        check_invariants(a, want, "randomized end");
        CHECK(a.horizon() == b.horizon());
    }
}

int main()
{
    idle_targets_the_floor();
    decode_keeps_the_constant();
    long_prefill_targets_the_floor();
    alternation_is_counted();
    threshold_is_notice_rows();
    small_quantum_config();
    randomized_agrees();
    if (fails) { std::fprintf(stderr, "scheduler_horizon_test: %d failure(s)\n", fails); return 1; }
    std::printf("scheduler_horizon_test: notice target floored at %llu, capped at %llu, %u rows of cover\n",
                (unsigned long long)NOTICE_MIN, (unsigned long long)NOTICE_MAX, NOTICE_ROWS);
    return 0;
}
