// Host-side scheduler, batch, dispatch & prefix-reuse verification for the 32-session 64K config:
//   SESS = 32, CAP = 65536 (32,768 pages / 2,097,152 pool rows), MX = 48,
//   dynamic MTP=5 for S <= 8 and MTP=0 (rps=1) for S in 9..32,
//   including S = 1, 8, 9, 16, 32 waves and live S: 8 -> 16 -> 8 crossing + PrefixReuse.
#include "s2_batch.h"
#include "s2/scheduler.h"
#include "s2/dispatch.h"
#include "s2/ds41_dspark_round_logic.h"
#include "prefix_reuse.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace s2;
using g6s2::Ds41RoundLogic;

static int fails = 0;
static std::string ctx;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s [%s]\n", __FILE__, __LINE__, #c, ctx.c_str()); ++fails; } } while (0)

constexpr int V = 1000;
constexpr uint32_t SEED = 0x1234567u;
static uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16; return x; }
static uint32_t step_state(uint32_t prev, int tok) { return mix(prev * 0x9E3779B1u + (uint32_t)tok * 0x85ebca77u + 0x165667b1u); }
static int sample(uint32_t st) { return (int)(mix(st ^ 0xA5A5A5A5u) % V); }

static long accepted_of(const s2::Stats& st) {
    long acc = 0;
    for (size_t a = 1; a < st.acc.size(); ++a) acc += (long)a * st.acc[a];
    return acc;
}

struct Ref {
    std::vector<int> tok;
    std::vector<uint32_t> st;
    void push(int t) { st.push_back(step_state(st.empty() ? SEED : st.back(), t)); tok.push_back(t); }
};

static Ref reference(const Session& s, int total) {
    Ref r;
    for (int t : s.prompt) r.push(t);
    for (int gen = 0; gen < total; ++gen) r.push(sample(r.st.back()));
    return r;
}

struct FakeS32 {
    int K;
    std::vector<Ds41RoundLogic> logic;
    std::vector<const Ref*> ref = std::vector<const Ref*>(S2N_SLOTS, nullptr);
    std::vector<std::vector<uint32_t>> h = std::vector<std::vector<uint32_t>>(S2N_SLOTS, std::vector<uint32_t>(4096, 0xDEADBEEFu));
    std::vector<g6rt::PrefixReuse> slot_state = std::vector<g6rt::PrefixReuse>(S2N_SLOTS);
    std::vector<uint32_t> taps;
    std::vector<int> src = std::vector<int>(S2N_SLOTS, 0), pos0 = std::vector<int>(S2N_SLOTS, -1);
    long mains = 0, drafts = 0, chains = 0;
    long drafts_above_8 = 0, chains_above_8 = 0;
    long rps6_rounds = 0, rps1_rounds = 0;
    int max_rows = 0, max_ns = 0, max_draft_rows = 0;

    explicit FakeS32(int k) : K(k), logic(size_t(S2N_SLOTS), Ds41RoundLogic(k)) {
        for (auto& st : slot_state) st.noend = true;
    }

    Hooks hooks() {
        Hooks hk;
        hk.run = [this](const Step& s) -> std::vector<int> {
            return s.kind == Step::MAIN ? main_step(s) : s.kind == Step::DRAFT ? draft_step(s) : chain_step(s);
        };
        hk.emit = [](int, const int*, int, int) { return false; };
        return hk;
    }

    void bind_slot(int q, int reuse) {
        logic[size_t(q)].reset();
        slot_state[size_t(q)].main_end = slot_state[size_t(q)].draft_end =
            slot_state[size_t(q)].written_end = slot_state[size_t(q)].pf_end = reuse;
    }

    void retire_slot(int q, const std::vector<int>& prompt, const std::vector<int>& gen) {
        slot_state[size_t(q)].finish(prompt, gen, int(gen.size()), true);
    }

    std::vector<int> main_step(const Step& s) {
        ++mains;
        const size_t ns = s.sess.size();
        max_rows = std::max(max_rows, s.M());
        max_ns = std::max(max_ns, (int)ns);
        CHECK(s.M() <= 48);
        if (ns > 1) {
            for (size_t j = 0; j < ns; ++j) {
                CHECK(s.rows[j] == s.rows[0] && s.row0[j] == (int)j * s.rows[0]);
            }
        }
        if (s.round >= 0) {
            if (ns > 8) {
                CHECK(s.rows[0] == 1);
                ++rps1_rounds;
            } else if (s.rows[0] == K + 1) {
                ++rps6_rounds;
            } else {
                CHECK(s.rows[0] == 1);
                ++rps1_rounds;
            }
        }
        std::vector<int> out(size_t(s.M()), 0);
        taps.assign(size_t(s.M()), 0);
        for (size_t j = 0; j < ns; ++j) {
            const int q = s.sess[j], r0 = s.row0[j], n = s.rows[j];
            const Ref& rf = *ref[size_t(q)];
            auto& hq = h[size_t(q)];
            const int p0 = s.r[size_t(r0)].pos;
            CHECK(s.r[size_t(r0)].tok == rf.tok[size_t(p0)]);
            CHECK(p0 == 0 || hq[size_t(p0 - 1)] == rf.st[size_t(p0 - 1)]);
            for (int i = 0; i < n; ++i) {
                const Row& r = s.r[size_t(r0 + i)];
                CHECK(r.sess == q && r.pos == p0 + i);
                hq[size_t(r.pos)] = step_state(r.pos == 0 ? SEED : hq[size_t(r.pos - 1)], r.tok);
                taps[size_t(r0 + i)] = hq[size_t(r.pos)];
                out[size_t(r0 + i)] = sample(hq[size_t(r.pos)]);
            }
            src[size_t(q)] = r0;
            pos0[size_t(q)] = p0;
            // Mirror Dispatch::run (engine/runtime/s2/dispatch.h):
            slot_state[size_t(q)].main_rows(p0, n);
            if (s.round < 0 || n == 1) slot_state[size_t(q)].draft_rows(p0, n, false);
        }
        return out;
    }

    std::vector<int> draft_step(const Step& s) {
        ++drafts;
        const size_t ns = s.sess.size();
        if (ns > 8) ++drafts_above_8;
        CHECK(ns <= 8);
        std::vector<int> u(size_t(s.M()), 0);
        int S = 0;
        for (size_t j = 0; j < ns; ++j) {
            const int q = s.sess[j], r0 = s.row0[j], n = s.rows[j];
            const int p0 = s.r[size_t(r0)].pos;
            slot_state[size_t(q)].draft_rows(p0, n, false);
            Ds41RoundLogic::DraftIn in;
            if (!logic[size_t(q)].draft_in_sess(s, j, in)) continue;
            ++S;
            const Ref& rf = *ref[size_t(q)];
            const int crow0 = src[size_t(q)];
            CHECK(in.acc == in.q - pos0[size_t(q)] && in.acc + 1 == n);
            CHECK(in.golden == rf.tok[size_t(in.q + 1)]);
            const int tr = crow0 + in.acc;
            CHECK(tr >= 0 && tr < (int)taps.size() && taps[size_t(tr)] == rf.st[size_t(in.q)]);
            const uint32_t s0 = taps[size_t(tr)];
            int c[Ds41RoundLogic::BLOCK], dr[Ds41RoundLogic::BLOCK];
            uint32_t st = step_state(s0, in.golden);
            for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i) { c[i] = sample(st); st = step_state(st, c[i]); }
            const int w = 1 + (int)(mix((uint32_t)in.q * 2654435761u + (uint32_t)q * 977u) % 5);
            for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i) dr[i] = i < w ? c[i] : (c[i] + 1 + i) % V;
            u[size_t(r0 + n - 1)] = logic[size_t(q)].draft_set(in.q, dr);
        }
        max_draft_rows = std::max(max_draft_rows, S * Ds41RoundLogic::BLOCK);
        CHECK(S * Ds41RoundLogic::BLOCK <= 48);
        return u;
    }

    std::vector<int> chain_step(const Step& s) {
        ++chains;
        const size_t ns = s.sess.size();
        if (ns > 8) ++chains_above_8;
        CHECK(ns <= 8);
        std::vector<int> u(size_t(s.M()), 0);
        for (size_t j = 0; j < ns; ++j) {
            const int q = s.sess[j], r0 = s.row0[j];
            CHECK(s.rows[j] == 1);
            slot_state[size_t(q)].draft_rows(s.r[size_t(r0)].pos, 1, true);
            u[size_t(r0)] = logic[size_t(q)].chain_row(s.r[size_t(r0)]);
        }
        return u;
    }
};

static Session make_sess(int q, int wave, int plen, int n_gen) {
    Session s;
    s.n_gen = n_gen;
    for (int i = 0; i < plen; ++i) {
        s.prompt.push_back((int)(mix((uint32_t)(i * 13 + q * 257 + wave * 4099 + 19)) % V));
    }
    s.first = 0;
    s.chunk = 8;
    return s;
}

static void test_scheduler_s32_64k() {
    ctx = "scheduler_s32_64k";
    CHECK(g6s2::MAX_SESSIONS == 32);
    CHECK(S2N_SLOTS == 32);

    g6s2::Config sc;
    sc.pool_rows = 32u * 65536u; // 2,097,152 rows = 32,768 pages of 64
    sc.max_rows = 48;
    sc.max_sessions = 32;
    sc.mtp = 5;
    sc.prefill_rows = 4096;
    sc.mtp_sessions = 0;
    CHECK(sc.limit() == 8u); // every session drafts: 48 / 6 rows
    sc.mtp_sessions = 8;
    CHECK(sc.limit() == 32u); // drafting stops past 8 sessions, one row each

    g6s2::Scheduler sched(sc);
    uint64_t next_id = 1;
    uint32_t used_pages = 0;
    const uint32_t total_pages = sc.pool_rows / g6s2::PAGE;
    CHECK(total_pages == 32768u);

    auto admit_req = [&](uint32_t prompt_tok, uint32_t max_tok) -> g6s2::Key {
        g6s2::Request rq{next_id++, prompt_tok, max_tok, false, {}};
        const uint32_t pages = g6s2::reserve_pages(sc, rq);
        g6s2::Key k = sched.next_key();
        g6s2::Commit cm;
        cm.epoch = sc.epoch;
        cm.event = sched.next_event();
        cm.boundary = sched.boundary();
        cm.before_hash = sched.digest();
        cm.admit = g6s2::Grant{k, rq, g6s2::Run{used_pages, pages, 1, 7}, 0};
        sched.apply(cm);
        used_pages += pages;
        return k;
    };

    auto cancel_req = [&](g6s2::Key k) {
        g6s2::Commit cm;
        cm.epoch = sc.epoch;
        cm.event = sched.next_event();
        cm.boundary = sched.boundary();
        cm.before_hash = sched.digest();
        cm.cancel = k;
        auto d = sched.apply(cm);
        for (const auto& r : d.freed) used_pages -= r.pages;
    };

    // 1. Single-slot 64K context allocation check (65,000 prompt + 526 max_tok + 10 guard = 65,536 rows = 1,024 pages)
    g6s2::Key k64 = admit_req(65000, 526);
    CHECK(sched.resident() == 1u);
    CHECK(used_pages == 1024u && total_pages - used_pages == 31744u);
    cancel_req(k64);
    CHECK(sched.resident() == 0u && used_pages == 0u);

    // 2. Concurrent wave admissions for S = 1, 8, 9, 16, 32
    for (unsigned S : {1u, 8u, 9u, 16u, 32u}) {
        ctx = "scheduler_wave_S=" + std::to_string(S);
        std::vector<g6s2::Key> keys;
        for (unsigned i = 0; i < S; ++i) keys.push_back(admit_req(128, 128));
        CHECK(sched.resident() == S);
        if (S == 32u) {
            bool threw = false;
            try { (void)sched.next_key(); } catch (const std::exception&) { threw = true; }
            CHECK(threw); // 33rd session refused when all 32 slots are occupied
        }
        for (auto k : keys) cancel_req(k);
        CHECK(sched.resident() == 0u && used_pages == 0u);
    }
}

static void test_batch_wave(int S) {
    ctx = "batch_wave_S=" + std::to_string(S);
    FakeS32 f(5);
    Hooks hk = f.hooks();
    Batch b(5, 48, hk);
    b.set_mtp_sessions(8);   // the DS41 engine's Config::mtp_sessions
    std::vector<Ref> refs(static_cast<size_t>(S));
    std::vector<Session> sess(static_cast<size_t>(S));
    std::vector<int> order;
    for (int q = 0; q < S; ++q) {
        sess[size_t(q)] = make_sess(q, 0, 12 + (q % 5), 32);
        refs[size_t(q)] = reference(sess[size_t(q)], sess[size_t(q)].n_gen + 16);
        f.ref[size_t(q)] = &refs[size_t(q)];
        f.bind_slot(q, 0);
        order.push_back(q);
    }
    b.admit(order, sess);
    while (b.round(order)) {}
    CHECK(f.drafts_above_8 == 0 && f.chains_above_8 == 0);
    CHECK(f.max_ns == S);
    for (int q = 0; q < S; ++q) {
        const std::vector<int> got = b.result(q);
        const int plen = (int)sess[size_t(q)].prompt.size();
        const std::vector<int> want(refs[size_t(q)].tok.begin() + plen, refs[size_t(q)].tok.begin() + plen + sess[size_t(q)].n_gen);
        CHECK(got == want);
        const s2::Stats st = b.stats(q);
        const long acc = accepted_of(st);
        if (S <= 8) {
            CHECK(st.rounds > 0);
            CHECK(st.drafted == 5 * st.rounds);
            CHECK(acc > 0);
        } else {
            CHECK(st.rounds == sess[size_t(q)].n_gen - 1);
            CHECK(st.drafted == 0 && acc == 0);
        }
        f.retire_slot(q, sess[size_t(q)].prompt, got);
        CHECK(f.slot_state[size_t(q)].valid == plen + (int)got.size() - 1);
        b.release(q);
    }
    if (S <= 8) {
        CHECK(f.rps6_rounds > 0 && f.drafts > 0 && f.chains > 0);
    } else {
        CHECK(f.rps6_rounds == 0 && f.rps1_rounds > 0 && f.drafts == 0 && f.chains == 0);
    }
    std::printf("wave S=%2d: max_ns=%2d max_rows=%2d rps6_rounds=%3ld rps1_rounds=%3ld drafts=%3ld chains=%3ld -> PASS\n",
                S, f.max_ns, f.max_rows, f.rps6_rounds, f.rps1_rounds, f.drafts, f.chains);
}

static void test_crossing_8_16_8_and_reuse() {
    ctx = "crossing_8_16_8";
    FakeS32 f(5);
    Hooks hk = f.hooks();
    Batch b(5, 48, hk);
    b.set_mtp_sessions(8);   // the DS41 engine's Config::mtp_sessions
    std::vector<Ref> refs(32);
    std::vector<Session> sess(16);
    std::vector<int> order;

    // 1. Start 8 long sessions (slots 0..7, n_gen = 90) in S = 8 (MTP=5)
    for (int q = 0; q < 8; ++q) {
        sess[size_t(q)] = make_sess(q, 10, 14, 90);
        refs[size_t(q)] = reference(sess[size_t(q)], sess[size_t(q)].n_gen + 16);
        f.ref[size_t(q)] = &refs[size_t(q)];
        f.bind_slot(q, 0);
        order.push_back(q);
    }
    b.admit(order, std::vector<Session>(sess.begin(), sess.begin() + 8));

    // Run 4 rounds in S = 8 (MTP=5)
    for (int r = 0; r < 4; ++r) {
        CHECK(b.round(order));
    }
    const long rps6_before_cross = f.rps6_rounds;
    CHECK(rps6_before_cross == 4);
    for (int q = 0; q < 8; ++q) {
        CHECK(b.stats(q).rounds == 4 && b.stats(q).drafted == 20);
    }

    // 2. Inject 8 short sessions (slots 8..15, n_gen = 8) so active concurrency crosses 8 -> 16 (MTP=5 -> MTP=0)
    std::vector<int> short_q;
    std::vector<Session> short_s;
    for (int q = 8; q < 16; ++q) {
        sess[size_t(q)] = make_sess(q, 20, 10, 8);
        refs[size_t(q)] = reference(sess[size_t(q)], sess[size_t(q)].n_gen + 16);
        f.ref[size_t(q)] = &refs[size_t(q)];
        f.bind_slot(q, 0);
        short_q.push_back(q);
        short_s.push_back(sess[size_t(q)]);
        order.push_back(q);
    }
    b.admit(short_q, short_s);

    const long drafts_at_16_start = f.drafts;
    bool retired_short = false;
    while (b.round(order)) {
        if (!retired_short && !b.active(8)) {
            retired_short = true;
            for (int q = 8; q < 16; ++q) {
                CHECK(!b.active(q));
                const s2::Stats st = b.stats(q);
                CHECK(st.drafted == 0 && accepted_of(st) == 0);
                const std::vector<int> got = b.result(q);
                const int plen = (int)sess[size_t(q)].prompt.size();
                const std::vector<int> want(refs[size_t(q)].tok.begin() + plen, refs[size_t(q)].tok.begin() + plen + sess[size_t(q)].n_gen);
                CHECK(got == want);
                f.retire_slot(q, sess[size_t(q)].prompt, got);
                CHECK(f.slot_state[size_t(q)].valid == plen + (int)got.size() - 1);
                b.release(q);
            }
            order.resize(8); // slots 0..7 remain
        }
    }
    CHECK(retired_short);
    CHECK(f.max_ns == 16);
    CHECK(f.drafts_above_8 == 0 && f.chains_above_8 == 0);
    CHECK(f.rps1_rounds >= 7); // 7 decode rounds at S=16 plus 1 transition round at S=8 before MTP=5 resumes
    CHECK(f.rps6_rounds > rps6_before_cross); // MTP=5 resumed after dropping back to S=8
    CHECK(f.drafts > drafts_at_16_start);

    // 3. Verify all 8 long sessions (0..7) executed in both MTP=5 and MTP=0 regimes and match reference
    for (int q = 0; q < 8; ++q) {
        const s2::Stats st = b.stats(q);
        CHECK(st.drafted > 0 && st.drafted < 5 * st.rounds);
        CHECK(accepted_of(st) > 0);
        const std::vector<int> got = b.result(q);
        const int plen = (int)sess[size_t(q)].prompt.size();
        const std::vector<int> want(refs[size_t(q)].tok.begin() + plen, refs[size_t(q)].tok.begin() + plen + sess[size_t(q)].n_gen);
        CHECK(got == want);
        f.retire_slot(q, sess[size_t(q)].prompt, got);
        CHECK(f.slot_state[size_t(q)].valid == plen + (int)got.size() - 1);
        b.release(q);
    }

    // 4. REUSE=1 check: re-send 4 repeat prompts (slots 8, 9 from MTP=0 wave and slots 0, 1 from crossed wave) with new suffix
    ctx = "crossing_reuse_4_repeats";
    const int rep_slots[4] = {8, 9, 0, 1};
    for (int i = 0; i < 4; ++i) {
        const int q = rep_slots[i];
        Session s2 = sess[size_t(q)];
        const int orig_len = (int)s2.prompt.size();
        s2.prompt.push_back(777 + i);
        s2.prompt.push_back(888 + i);
        s2.n_gen = 16;
        const int skip = f.slot_state[size_t(q)].begin(s2.prompt, (int)s2.prompt.size() - 1, true);
        CHECK(skip == orig_len - 1 && skip > 0);
        s2.first = skip;
        refs[size_t(16 + i)] = reference(s2, s2.n_gen + 16);
        f.ref[size_t(q)] = &refs[size_t(16 + i)];
        f.bind_slot(q, skip);
        b.admit({q}, {s2});
        while (b.round({q})) {}
        const std::vector<int> got = b.result(q);
        const int plen = (int)s2.prompt.size();
        const std::vector<int> want(refs[size_t(16 + i)].tok.begin() + plen, refs[size_t(16 + i)].tok.begin() + plen + s2.n_gen);
        CHECK(got == want);
        b.release(q);
    }
    std::printf("crossing 8->16->8 + 4 repeat REUSE=1 prompts: rps6_rounds=%ld rps1_rounds=%ld drafts=%ld -> PASS\n",
                f.rps6_rounds, f.rps1_rounds, f.drafts);
}

int main() {
    test_scheduler_s32_64k();
    for (int S : {1, 8, 9, 16, 32}) {
        test_batch_wave(S);
    }
    test_crossing_8_16_8_and_reuse();
    std::printf("s32_64k_mtp8_test: %s (%d failures)\n", fails ? "FAILED" : "all passed", fails);
    return fails ? 1 : 0;
}
