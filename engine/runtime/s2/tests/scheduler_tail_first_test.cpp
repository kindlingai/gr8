// Config::tail_first (s2/scheduler.h, Scheduler::plan + Scheduler::alternate; GR8_S2_TAIL_FIRST=1): a session whose prompt
// is prefilled is served before any prefill, and its tail asks for no decode round. The properties here: the c8 burst
// (one session decoding, seven short prompts admitted at one boundary) reaches every first token as P D T per session and
// sooner than with the knob off; between two prefills a decoder still gets its round; a lone session takes the same steps
// either way; the notice target (horizon) matches an independent re-derivation and two schedulers driven by the same
// events agree on it and on the digest; a long prompt still finishes behind a stream of short ones; and the agreement and
// digest name the knob so ranks booted differently refuse each other. Host only. Build and run: see scripts/host_tests.sh.
#include "s2/bootstrap.h"
#include "s2/scheduler.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static Config cfg(bool tf)
{
    Config c;
    c.pool_rows = 614400; c.max_rows = 48; c.max_sessions = 8; c.mtp = 5;
    c.prefill_rows = 1024; c.two_lanes = true; c.tail_first = tf;
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

static char letter(Kind k) { return k == Kind::Prefill ? 'P' : k == Kind::PromptTail ? 'T' : k == Kind::Decode ? 'D' : '-'; }

// The alternation as the rule states it, written apart from Scheduler::alternate (decode_rounds 0 here)
struct Alt {
    bool want = true;
    void after(const Config& c, Kind k) {
        if (c.tail_first && k == Kind::PromptTail) return;
        want = k != Kind::Decode;
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

// The measured c8 shape: session 1 (39 tokens) decoding, then 7 prompts of 39 tokens admitted at one held boundary.
// Returns the step pattern until every session decodes and fills `first` with each new session's first-token step.
static std::string burst(bool tf, std::vector<unsigned>& first)
{
    Scheduler s(cfg(tf));
    admit(s, 1, 39, 256);
    for (unsigned i = 0; i < 6; ++i) (void)step(s);             // P T then decode rounds
    CHECK(s.slots()[0].phase == Phase::Decode);
    for (uint64_t id = 2; id <= 8; ++id) admit(s, id, 39, 256);
    std::string pat; first.assign(8, 0);
    for (unsigned i = 1; i <= 200; ++i) {
        const Quantum q = step(s);
        pat += letter(q.kind);
        if (q.kind == Kind::PromptTail) first[q.keys[0].slot] = i;
        bool all = true;
        for (const auto& x : s.slots()) all &= x.phase == Phase::Decode || x.phase == Phase::Free;   // MAX_SESSIONS 32: skip Free slots
        if (all) break;
    }
    return pat;
}

static void c8_burst()
{
    std::vector<unsigned> off, on;
    const std::string p_off = burst(false, off), p_on = burst(true, on);
    std::printf("scheduler_tail_first_test: c8 burst off: %s\n", p_off.c_str());
    std::printf("scheduler_tail_first_test: c8 burst on : %s\n", p_on.c_str());
    CHECK(p_on == "PDTPDTPDTPDTPDTPDTPDT");                    // one tail per prefill, at once, no round after it
    const unsigned last_off = *std::max_element(off.begin(), off.end()), last_on = *std::max_element(on.begin(), on.end());
    unsigned sum_off = 0, sum_on = 0;
    for (unsigned i = 1; i < 8; ++i) { sum_off += off[i]; sum_on += on[i]; CHECK(on[i] <= off[i]); }
    CHECK(last_on < last_off);
    std::printf("scheduler_tail_first_test: c8 burst first-token step, sum/last: off %u/%u, on %u/%u\n",
                sum_off, last_off, sum_on, last_on);
}

// Replay of a measured eight-request prose burst (ADMIT_HOLD=1, TAIL_FIRST off), rank 0 s2_event lines: req 37
// (39 tokens) committed alone at 1183 on an idle
// engine, prefill 1184, tail 1185; reqs 38..44 (39 tokens each) committed at 1191 (38 by notice, 39..44 chained by the
// hold); then 38 prefill 1192 tail 1194; 39..44 prefill 1196..1206; tails 1208..1218. Boundaries are relative to 1183.
// With the knob off the scheduler must reproduce that order exactly (owed counters included); with it on, P D T each.
struct Ev { unsigned rel; uint64_t req; Kind kind; };
static std::vector<Ev> replay(bool tf)
{
    Config c = cfg(tf);
    c.prefill_rows = 1024; c.two_lanes = false;                 // ds41_prefill_single_lane(W = 1024)
    Scheduler s(c);
    const uint64_t b0 = s.boundary();
    std::vector<Ev> ev;
    auto run_to = [&](uint64_t until) {
        while (s.boundary() < until) {
            const Quantum q = step(s);
            if (q.kind == Kind::Prefill || q.kind == Kind::PromptTail)
                ev.push_back(Ev{unsigned(q.boundary - b0), s.slots()[q.keys[0].slot].req.id, q.kind});
            bool all = s.resident() > 0;
            for (const auto& x : s.slots()) all &= x.phase == Phase::Decode || x.phase == Phase::Free;
            if (until == UINT64_MAX && all) return;
        }
    };
    admit(s, 37, 39, 256);
    run_to(b0 + 8);                                            // 1191
    CHECK(s.boundary() == b0 + 8);
    for (uint64_t id = 38; id <= 44; ++id) admit(s, id, 39, 256);
    run_to(UINT64_MAX);
    return ev;
}

static void f3h_replay()
{
    std::vector<Ev> off{{1, 37, Kind::Prefill}, {2, 37, Kind::PromptTail}, {9, 38, Kind::Prefill}, {11, 38, Kind::PromptTail}};
    for (unsigned i = 0; i < 6; ++i) off.push_back(Ev{13 + 2 * i, 39 + i, Kind::Prefill});
    for (unsigned i = 0; i < 6; ++i) off.push_back(Ev{25 + 2 * i, 39 + i, Kind::PromptTail});
    std::vector<Ev> on{{1, 37, Kind::Prefill}, {2, 37, Kind::PromptTail}};
    for (unsigned i = 0; i < 7; ++i) { on.push_back(Ev{9 + 3 * i, 38 + i, Kind::Prefill}); on.push_back(Ev{11 + 3 * i, 38 + i, Kind::PromptTail}); }
    auto same = [](const std::vector<Ev>& a, const std::vector<Ev>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (a[i].rel != b[i].rel || a[i].req != b[i].req || a[i].kind != b[i].kind) return false;
        return true;
    };
    const auto r_off = replay(false), r_on = replay(true);
    auto show = [](const char* n, const std::vector<Ev>& v) {
        std::printf("scheduler_tail_first_test: measured replay %s:", n);
        for (const auto& e : v) std::printf(" %llu%c@%u", (unsigned long long)e.req, letter(e.kind), 1183 + e.rel);
        std::printf("\n");
    };
    show("off", r_off); show("on ", r_on);
    CHECK(same(r_off, off));                                   // the logged order, boundary for boundary
    CHECK(same(r_on, on));
}

// with a decoder present, two prefill quanta never run back to back (the decoders' round between prefills stays)
static bool rounds_kept(const std::string& pat)
{
    for (size_t i = 0; i + 1 < pat.size(); ++i) if (pat[i] == 'P' && pat[i + 1] == 'P') return false;
    return true;
}

// one session at a time: the knob changes nothing
static void lone_session_same()
{
    for (uint32_t prompt : {1u, 2u, 39u, 1500u, 5000u}) {
        Scheduler a(cfg(false)), b(cfg(true));
        admit(a, 1, prompt, 32); admit(b, 1, prompt, 32);
        for (unsigned i = 0; i < 60; ++i) {
            const Quantum qa = step(a), qb = step(b);
            CHECK(qa.kind == qb.kind && qa.rows == qb.rows && qa.first == qb.first);
            if (!a.resident()) break;
        }
    }
}

// a long prompt behind short ones still finishes: its prefill gets served under the owed rule
static void long_prompt_progresses()
{
    Scheduler s(cfg(true));
    admit(s, 1, 39, 100000);
    for (unsigned i = 0; i < 4; ++i) (void)step(s);
    admit(s, 2, 20000, 16);
    uint64_t id = 3; unsigned i = 0;
    std::string pat;
    for (; i < 2000 && s.slots()[1].phase == Phase::Prefill; ++i) {
        if (s.resident() < 8 && (i % 3) == 0) admit(s, id++, 39, 4);   // short requests keep arriving
        pat += letter(step(s).kind);
    }
    CHECK(s.slots()[1].phase != Phase::Prefill);
    CHECK(rounds_kept(pat));
    std::printf("scheduler_tail_first_test: 20000-row prompt behind %llu short ones done in %u steps\n",
                (unsigned long long)(id - 3), i);
}

struct Rng {
    uint64_t s = 1;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; }
    uint32_t range(uint32_t lo, uint32_t hi) { return lo + uint32_t(next() % (uint64_t(hi) - lo + 1)); }
};

// Randomized, knob on: horizon == the re-derivation at every step; two schedulers with the same events agree on the
// horizon and the digest; prefills never run back to back while a session decodes
static void randomized_agrees()
{
    for (uint64_t seed = 1; seed <= 80; ++seed) {
        Rng rng{seed * 0x9E3779B97F4A7C15ull + 11};
        Config c = cfg(true);
        c.prefill_rows = rng.range(1, 64) * PAGE;
        c.two_lanes = (rng.next() & 1) != 0;
        if (rng.next() & 1) c.decode_prefill_rows = rng.range(1, c.prefill_rows / PAGE) * PAGE;
        Scheduler a(c), b(c);
        Alt alt;
        uint64_t next_id = 1;
        auto admit_all = [&](uint32_t prompt, uint32_t tokens) {
            for (Scheduler* x : {&a, &b}) admit(*x, next_id, prompt, tokens);
            ++next_id;
        };
        for (unsigned i = 0, n = 1 + unsigned(rng.range(0, 4)); i < n; ++i) admit_all(rng.range(1, 6000), rng.range(4, 300));
        bool ok = true; Kind prev = Kind::Idle;
        for (unsigned i = 0, steps = rng.range(1, 500); i < steps; ++i) {
            CHECK(a.horizon() == expected_horizon(a, alt));
            CHECK(a.horizon() == b.horizon() && a.digest() == b.digest());
            bool decoding = false;
            for (const auto& x : a.slots()) decoding |= x.phase == Phase::Decode;
            const Quantum qa = step(a);
            CHECK(step(b).kind == qa.kind);
            if (decoding && qa.kind == Kind::Prefill && prev == Kind::Prefill) ok = false;   // a decoder lost its round
            prev = qa.kind;
            alt.after(c, qa.kind);
            if (!a.resident()) break;
            if (rng.range(0, 12) == 0 && next_id <= 40 && a.resident() < 8) admit_all(rng.range(1, 6000), rng.range(4, 300));
        }
        CHECK(ok);
    }
}

static void agreement_and_digest()
{
    auto agreement = [](bool tf) { BootstrapConfig b; b.scheduler = cfg(tf); b.runtime_signature = "x"; return b.agreement(); };
    CHECK(agreement(false).find(" tf") == std::string::npos);
    CHECK(agreement(true).find(" tf ctl ") != std::string::npos);
    CHECK(Scheduler(cfg(false)).digest() != Scheduler(cfg(true)).digest());
    // Config::notice_max (GR8_S2_NOTICE_MAX): named only when not the default 8
    auto nmax = [](uint32_t k) { BootstrapConfig b; b.scheduler = cfg(false); b.scheduler.notice_max = k; b.runtime_signature = "x"; return b.agreement(); };
    CHECK(nmax(8) == agreement(false));
    CHECK(nmax(4).find(" nmax 4 ctl ") != std::string::npos);
    CHECK(nmax(4) != nmax(3));
}

// Starvation bound, knob on: while a session is owed (deferred >= PREFILL_DEFER_MAX), at most MAX_SESSIONS - 1 owed
// rivals' prefill quanta and MAX_SESSIONS - 1 one-row tails run before it is served (each rival is served ahead of it
// at most once and every session has exactly one tail): <= 2 x 7 = 14 quanta, <= 7 of them tails. Knob off: <= 7.
static void owed_bound()
{
    for (bool tf : {false, true}) {
        unsigned worst = 0, worst_tails = 0;
        for (uint64_t seed = 1; seed <= 200; ++seed) {
            Rng rng{seed * 0xD1B54A32D192ED03ull + 3};
            Config c = cfg(tf);
            c.prefill_rows = rng.range(1, 16) * PAGE;
            Scheduler s(c);
            uint64_t id = 1;
            std::map<uint64_t, std::pair<unsigned, unsigned>> wait;   // owed request -> (quanta to others, tails) since owed
            for (unsigned i = 0; i < 3000; ++i) {
                if (s.resident() < 8 && rng.range(0, 2) == 0)
                    admit(s, id++, rng.range(0, 3) ? rng.range(2, 80) : rng.range(2000, 20000), rng.range(1, 6));
                if (!s.resident()) continue;
                std::vector<uint64_t> owed;
                for (const auto& x : s.slots())
                    if (x.phase == Phase::Prefill && x.deferred >= PREFILL_DEFER_MAX) owed.push_back(x.req.id);
                const Quantum q = step(s);
                if (q.kind != Kind::Prefill && q.kind != Kind::PromptTail) continue;
                const uint64_t served = s.slots()[q.keys[0].slot].req.id;
                for (uint64_t o : owed) {
                    if (o == served) { wait.erase(o); continue; }
                    auto& w = wait[o];
                    ++w.first; w.second += q.kind == Kind::PromptTail;
                    worst = std::max(worst, w.first); worst_tails = std::max(worst_tails, w.second);
                }
            }
        }
        std::printf("scheduler_tail_first_test: owed bound %s: worst %u quanta (%u tails) ahead of an owed prefill\n",
                    tf ? "on " : "off", worst, worst_tails);
        CHECK(worst <= (tf ? 2 : 1) * (MAX_SESSIONS - 1));
        CHECK(worst_tails <= MAX_SESSIONS - 1);
    }
}

int main()
{
    f3h_replay();
    owed_bound();
    c8_burst();
    lone_session_same();
    long_prompt_progresses();
    randomized_agrees();
    agreement_and_digest();
    if (fails) { std::fprintf(stderr, "scheduler_tail_first_test: %d failure(s)\n", fails); return 1; }
    std::printf("scheduler_tail_first_test: PASS\n");
    return 0;
}
