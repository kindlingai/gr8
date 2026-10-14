// The scheduler's prefill choice (s2/scheduler.h, Scheduler::next): of the sessions with prompt rows left, the shortest
// goes next, so a short request admitted behind a long prefill is not stuck behind every quantum of it; and a session
// passed over PREFILL_DEFER_MAX times is served whatever its size, so a stream of short requests cannot defer a long
// prompt without bound. The choice must be a pure function of the slots, because every rank recomputes it and the event
// digest compares the result: two schedulers driven by the same events must pick the same session at every boundary.
// Host only (scheduler.h is plain C++): no GPU, no CUDA, no pool. Build and run: see scripts/host_tests.sh.
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
static std::pair<Kind, uint64_t> step(Scheduler& s)
{
    const Quantum q = s.next();
    uint64_t id = 0;
    if (q.count) id = s.slots()[q.keys[0].slot].req.id;
    if (q.kind == Kind::Prefill) s.complete_prefill();
    else if (q.kind == Kind::PromptTail) s.complete_tail(Completion{q.keys[0], 1, false, 0});
    else if (q.kind == Kind::Decode) {
        std::vector<Completion> out;
        for (unsigned i = 0; i < q.count; ++i) out.push_back(Completion{q.keys[i], 1, false, 0});
        s.complete_decode(out);
    }
    return {q.kind, id};
}

static bool is_prefill_quantum(Kind k) { return k == Kind::Prefill || k == Kind::PromptTail; }
static Phase phase_of(const Scheduler& s, uint64_t id)
{
    for (const auto& sl : s.slots()) if (sl.req.id == id && sl.phase != Phase::Free) return sl.phase;
    return Phase::Free;
}
static uint32_t deferred_of(const Scheduler& s, uint64_t id)
{
    for (const auto& sl : s.slots()) if (sl.req.id == id) return sl.deferred;
    return 0;
}
// the sessions already owed a turn, read BEFORE the choice: prefill_first must serve one of them if any exist
static std::vector<uint64_t> owed(const Scheduler& s)
{
    std::vector<uint64_t> v;
    for (const auto& sl : s.slots()) if (sl.phase == Phase::Prefill && sl.deferred >= PREFILL_DEFER_MAX) v.push_back(sl.req.id);
    return v;
}
static bool holds(const std::vector<uint64_t>& v, uint64_t id)
{
    for (uint64_t x : v) if (x == id) return true;
    return false;
}

// #61: a 31,984-token prompt holds the engine while a short follow-up waits for a step. The short request must get the
// first prefill quantum, not the long one, and reach decode in a couple of boundaries rather than after five quanta.
static void short_behind_long()
{
    Scheduler s(cfg());
    admit(s, 1, 31984, 64);          // the long prompt, admitted first, so the old rule (lowest id) always chose it
    admit(s, 2, 200, 64);            // the short follow-up

    const auto first = step(s);      // the first step with both resident must prefill the short one
    CHECK(is_prefill_quantum(first.first));
    CHECK(first.second == 2);

    unsigned boundaries = 1;
    while (phase_of(s, 2) == Phase::Prefill && boundaries < 8) { step(s); ++boundaries; }
    CHECK(phase_of(s, 2) == Phase::Decode);          // it prefills and tails out without waiting for the long prompt
    CHECK(phase_of(s, 1) == Phase::Prefill);         // which is still prefilling: this is fairness, not a size effect
    CHECK(boundaries <= 4);
    std::printf("short behind long: short reached decode in %u boundaries, long deferred %u\n",
                boundaries, deferred_of(s, 1));
}

// SRPT alone would let a stream of short requests defer a long prefill indefinitely. PREFILL_DEFER_MAX bounds that: a
// session owed a turn is served before any session that is not, and no continuously-prefilling session is ever passed
// over more than PREFILL_DEFER_MAX + MAX_SESSIONS prefill quanta.
static void long_never_starved()
{
    Scheduler s(cfg());
    admit(s, 1, 200000, 64);         // 25 quanta of work: it cannot finish while the short ones keep arriving
    for (uint64_t id = 2; id <= 8; ++id) admit(s, id, 5000, 64);   // one prefill quantum each, then a tail

    unsigned gap = 0, worst_gap = 0, checked = 0;
    for (unsigned round = 0; round < 600; ++round) {
        const auto before = owed(s);
        const auto st = step(s);
        if (!is_prefill_quantum(st.first)) continue;
        if (!before.empty()) { CHECK(holds(before, st.second)); ++checked; }   // owed always wins, whoever it is
        if (phase_of(s, 1) != Phase::Prefill) break;
        if (st.second == 1) { if (gap > worst_gap) worst_gap = gap; gap = 0; }
        else ++gap;
    }
    CHECK(checked > 0);                                      // the owed-first branch was exercised, not just SRPT
    CHECK(worst_gap <= PREFILL_DEFER_MAX + MAX_SESSIONS);
    std::printf("long never starved: worst gap %u prefill quanta, %u owed-first choices\n", worst_gap, checked);
}

// One session is the whole point of the old rule and must not change: same quanta, same sizes, same order.
static void single_session_unchanged()
{
    Scheduler s(cfg());
    admit(s, 1, 31984, 64);
    std::vector<uint32_t> rows;
    while (phase_of(s, 1) == Phase::Prefill) {
        const Quantum q = s.next();
        CHECK(q.count == 1 && q.keys[0].slot == 0);
        rows.push_back(q.rows);
        if (q.kind == Kind::Prefill) s.complete_prefill();
        else { CHECK(q.kind == Kind::PromptTail && q.rows == 1); s.complete_tail(Completion{q.keys[0], 1, false, 0}); }
        CHECK(rows.size() <= 16);                    // bounded: a stuck scheduler must not hang the test
    }
    CHECK(phase_of(s, 1) == Phase::Decode);
    // 31,983 rows to prefill, two lanes: 8192 + 8192 + 8192 + 7407, then the one-row tail
    const std::vector<uint32_t> want{8192, 8192, 8192, 7407, 1};
    CHECK(rows == want);
    std::printf("single session: %zu quanta of 8192/7407/1 rows, unchanged\n", rows.size());
}

// The choice is replicated state: two schedulers fed the same events must choose the same sessions and agree on the
// event digest, which is what a rank compares at the grant. A divergence shows up here as a key or digest mismatch.
static void ranks_agree()
{
    Scheduler a(cfg()), b(cfg());
    admit(a, 1, 24000, 64); admit(b, 1, 24000, 64);
    admit(a, 2, 900, 64);   admit(b, 2, 900, 64);
    admit(a, 3, 6000, 64);  admit(b, 3, 6000, 64);
    CHECK(a.digest() == b.digest());
    for (unsigned i = 0; i < 300; ++i) {
        const auto qa = a.next();
        const auto qb = b.next();
        CHECK(qa.kind == qb.kind && qa.count == qb.count && qa.rows == qb.rows && qa.first == qb.first);
        for (unsigned k = 0; k < qa.count; ++k) CHECK(qa.keys[k] == qb.keys[k]);
        CHECK(a.digest() == b.digest());
        if (qa.kind == Kind::Prefill) { a.complete_prefill(); b.complete_prefill(); }
        else if (qa.kind == Kind::PromptTail) {
            a.complete_tail(Completion{qa.keys[0], 1, false, 0});
            b.complete_tail(Completion{qb.keys[0], 1, false, 0});
        } else if (qa.kind == Kind::Decode) {
            std::vector<Completion> ca, cb;
            for (unsigned k = 0; k < qa.count; ++k) {
                ca.push_back(Completion{qa.keys[k], 1, false, 0});
                cb.push_back(Completion{qb.keys[k], 1, false, 0});
            }
            a.complete_decode(ca); b.complete_decode(cb);
        }
        CHECK(a.digest() == b.digest());
    }
    std::printf("two ranks: identical quanta and digests over 300 boundaries\n");
}

// Decode must still get its round between prefill quanta wherever a session is decoding: fairness among prefills must
// not run prefills back to back and delay every decoder.
static void decode_still_alternates()
{
    Scheduler s(cfg());
    admit(s, 1, 31984, 64);
    admit(s, 2, 31984, 64);
    unsigned rounds = 0, prefills = 0;
    bool prev_prefill = false;
    for (unsigned i = 0; i < 80; ++i) {
        unsigned decoders = 0;
        for (const auto& sl : s.slots()) decoders += sl.phase == Phase::Decode;
        const auto st = step(s);
        if (st.first == Kind::Decode) ++rounds;
        else if (is_prefill_quantum(st.first)) { if (decoders) CHECK(!prev_prefill); ++prefills; }
        prev_prefill = is_prefill_quantum(st.first);
    }
    CHECK(prefills > 0);
    std::printf("alternation: %u prefill quanta, %u decode rounds in 80 boundaries\n", prefills, rounds);
}

int main()
{
    short_behind_long();
    long_never_starved();
    single_session_unchanged();
    ranks_agree();
    decode_still_alternates();
    std::printf("scheduler_fair_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
