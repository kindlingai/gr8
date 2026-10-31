// DSpark round (B12) on the REAL s2::Batch with a fake engine: the host side of the served MTP=1..5 flow for DeepSeek-V4.1-Flash.
// The fake engine is a recurrent model: the state of position p is H(state of p - 1, input token), written by EVERY main row (like the KV
// cache: rejected draft rows leave wrong state above the accepted prefix, which the next step must rewrite), and the token after p is
// sample(state of p). The DRAFT hook is the round runner's host logic (s2/ds41_dspark_round_logic.h, the same class
// ds41_dspark_round.cuh uses) around a fake draft device that reads ONLY what the real device reads: the taps (the states of the latest main
// step's rows), q, golden and acc. Error injection (all right / first wrong / last wrong / mixed) decides which drafts the fake device gets
// right, so every accept count 0..K occurs.
// Checks (each also run as a mutation that must fail, so the test has teeth):
//   (a) canary: the committed tokens equal the direct reference (= MTP=0, temperature 0) for every K 1..5, chunk size, prompt length, pattern
//   (b) every executed DRAFT: acc == q - (pos of row 0 of the latest MAIN step), acc + 1 rows, golden == the true token at q + 1, the state
//       at q is the true one (accepted rows only)
//   (c) every CHAIN step continues the cached draft (pos = q + j, tok = draft[j - 1]), no step throws, K - 1 chain steps per round
//   (d) every verify step is K + 1 consecutive positions of one session starting at q + 1 (B5's rule: the first rejected position), the
//       state below its row 0 is the true one
//   (e) a prompt draft before the tail is skipped and nothing changes when it is not (the fake device depends on main-step state only)
//   (f) back-to-back requests (reset in between, stale state left above the new prompt); the insert rule's forced chunks (draft with M > 6)
//   (g) the Engram token ring (engine/engram/engram_ring.h, the code k_engram_win runs) fed with the (pos, tok) rows of EVERY main step the
//       real Batch emits (prompt chunks, tail, DECODE1, verify rounds at K 1..5, accepts and rejects, the ring never cleared between
//       requests): per step all rows are put, then every row reads pos-0..3, and each window equals the true sequence's (positions below the
//       step's row 0: the committed token; rows of the step: the fed token; before position 0: -1). A rejected draft's entry sits in the
//       ring above the accepted prefix; the next verify step starts at the first rejected position and rewrites it before any read.
// Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -I engine/runtime engine/runtime/s2/tests/batch_ds41_dspark_test.cpp -o /tmp/batch_ds41_dspark_test
#include "s2_batch.h"
#include "s2/ds41_dspark_round_logic.h"
#include "../../../engram/engram_ring.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
using namespace s2;
using g6s2::Ds41RoundLogic;

static int fails = 0;
static bool quiet = false;
static std::string ctx;
#define CHECK(c) do { if (!(c)) { if (!quiet && fails < 25) std::printf("FAIL %s:%d %s  [%s]\n", __FILE__, __LINE__, #c, ctx.c_str()); ++fails; } } while (0)

constexpr int V = 1000;
constexpr uint32_t SEED = 0x1234567u;
static uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16; return x; }
static uint32_t step_state(uint32_t prev, int tok) { return mix(prev * 0x9E3779B1u + (uint32_t)tok * 0x85ebca77u + 0x165667b1u); }
static int sample(uint32_t st) { return (int)(mix(st ^ 0xA5A5A5A5u) % V); }

// the true stream: prompt, then n total generated tokens (MTP=0 at temperature 0), with the insert rule as Batch::commit applies it
struct Ref
{
    std::vector<int> tok;
    std::vector<uint32_t> st;
    void push(int t) { st.push_back(step_state(st.empty() ? SEED : st.back(), t)); tok.push_back(t); }
};
static Ref reference(const Session& s, int total)
{
    Ref r;
    for (int t : s.prompt) r.push(t);
    bool armed = s.ins.on();
    int gen = 0;
    while (gen < total)
    {
        if (armed && gen == s.ins.after)
        {
            armed = false;
            const int m = std::min<int>((int)s.ins.tokens.size(), total - gen);
            for (int i = 0; i < m; ++i) r.push(s.ins.tokens[i]);
            gen += m;
            continue;
        }
        const int t = sample(r.st.back());
        if (t == s.ins.unless) armed = false;
        r.push(t);
        ++gen;
    }
    return r;
}

enum Pattern { ALL, FIRST, LAST, MIX, NPAT };
static const char* PN[] = {"all", "first", "last", "mix"};
using DraftRec = std::array<int, 8>;     // q, golden, acc, draft[0..4]

struct Fake
{
    int K;
    Pattern pat;
    bool insert_mode, skip;
    int mut = 0;                          // mutation (the test must fail): 1 acc + 1, 2 tail draft skipped, 3 golden + 1,
                                          // 4 ring reads before the writes, 5 ring put at pos + 1, 6 verify row 0 not put (stale draft stays)
    Ds41RoundLogic logic;
    const Ref* ref = nullptr;
    std::vector<uint32_t> h = std::vector<uint32_t>(4096, 0xDEADBEEFu);   // state by position, never cleared (stale across requests)
    std::vector<uint32_t> taps;           // the states of the rows of the latest MAIN step
    int tap_pos0 = -1, plen = 0;           // plen: the request's prompt length
    int last_q = -1, chains_since = 0, last_draft[Ds41RoundLogic::BLOCK] = {};
    bool drafted = false;
    long executed = 0, skipped = 0, chains = 0, mains = 0;
    int max_draft_M = 0;
    std::vector<DraftRec> dlog;           // executed drafts that are not a skippable prompt chunk's
    int ring[ds41::engram::kRing];        // (g) the Engram token ring: garbage at the start, never cleared between requests (like the device buffer)
    std::vector<int> fed = std::vector<int>(4096, -1);   // the token last fed for each position of this request (to see a rejected draft come back)
    long ring_windows = 0, ring_steps = 0, ring_split = 0, ring_valid_rows = 0, ring_rejected_rows = 0, ring_refed_rejected = 0;

    Fake(int k, Pattern p, bool ins, bool sk) : K(k), pat(p), insert_mode(ins), skip(sk), logic(k, sk) { std::fill(ring, ring + ds41::engram::kRing, 7777); }

    void new_request()
    {
        logic.reset();
        drafted = false; last_q = -1; chains_since = 0;
        std::fill(fed.begin(), fed.end(), -1);
    }
    Hooks hooks()
    {
        Hooks hk;
        hk.run = [this](const Step& s) -> std::vector<int> {
            return s.kind == Step::MAIN ? main(s) : s.kind == Step::DRAFT ? draft(s) : chain(s);
        };
        hk.emit = [](int, const int*, int, int) { return false; };
        return hk;
    }

    std::vector<int> main(const Step& s)
    {
        ++mains;
        CHECK(s.sess.size() == 1);
        for (int i = 0; i < s.M(); ++i) CHECK(s.r[i].pos == s.r[0].pos + i);       // (d) consecutive positions
        const int p0 = s.r[0].pos;
        CHECK(p0 >= 0 && s.r.back().pos < (int)h.size());
        if (p0 < 0 || s.r.back().pos >= (int)h.size()) return std::vector<int>(size_t(s.M()), 0);
        CHECK(s.r[0].tok == ref->tok[p0]);                                          // row 0 is a committed token
        CHECK(p0 == 0 || h[p0 - 1] == ref->st[p0 - 1]);                             // the state below row 0 is the true one
        if (s.round >= 0 && !insert_mode)
        {
            CHECK(s.M() == K + 1);
            CHECK(drafted && p0 == last_q + 1);                                     // the first rejected position (B5)
            CHECK(chains_since == K - 1);
        }
        std::vector<int> out;
        taps.assign(size_t(s.M()), 0);
        for (int i = 0; i < s.M(); ++i)
        {
            const Row& r = s.r[i];
            h[r.pos] = step_state(r.pos == 0 ? SEED : h[r.pos - 1], r.tok);        // every row writes its state, rejected ones too
            taps[i] = h[r.pos];
            out.push_back(sample(h[r.pos]));
        }
        tap_pos0 = p0;
        ring_step(s);
        return out;
    }

    // (g) one main step through the ring, in the order of k_engram_win. The engine runs at most MX = G6_MX = 32 rows a step (engine/g6mx.h,
    // s2/native.cuh:418 rejects more): a longer step of this test (chunk 64, a forced insert chunk) is fed as consecutive sub-steps of <= 32 rows.
    void ring_step(const Step& s)
    {
        constexpr int MXR = 32;
        const int n = s.M();
        if (n > MXR) ++ring_split;
        for (int a = 0; a < n; a += MXR)
        {
            const int m = std::min(MXR, n - a), first = s.r[a].pos;
            int win[MXR][4];
            auto put_all = [&] {
                for (int i = 0; i < m; ++i)
                {
                    if (mut == 6 && s.round >= 0 && i == 0) continue;
                    ds41::engram::ring_put(ring, s.r[a + i].pos + (mut == 5), s.r[a + i].tok);
                }
            };
            auto get_all = [&] {
                for (int i = 0; i < m; ++i)
                    for (int k = 0; k < 4; ++k) win[i][k] = ds41::engram::ring_get(ring, s.r[a + i].pos - k);
            };
            for (int i = 0; i < m; ++i)
            {
                const int pos = s.r[a + i].pos;
                if (pos < (int)fed.size() && fed[pos] >= 0 && fed[pos] != s.r[a + i].tok) ++ring_refed_rejected;   // a rejected draft's position is fed again
                if (pos < (int)fed.size()) fed[pos] = s.r[a + i].tok;
            }
            if (mut == 4) { get_all(); put_all(); } else { put_all(); get_all(); }
            ++ring_steps;
            bool valid = true;                                                      // rows 0..i all equal the true sequence: a row the round can accept
            for (int i = 0; i < m; ++i)
            {
                const int pos = s.r[a + i].pos;
                valid = valid && pos < (int)ref->tok.size() && s.r[a + i].tok == ref->tok[pos];
                ring_valid_rows += valid; ring_rejected_rows += !valid;
                for (int k = 0; k < 4; ++k)
                {
                    const int q = pos - k;
                    const int want = q < 0 ? -1 : q >= first ? s.r[a + q - first].tok : q < (int)ref->tok.size() ? ref->tok[q] : -3;
                    ++ring_windows;
                    CHECK(win[i][k] == want);                                       // the window of this step: committed below row 0, fed rows from row 0
                    if (valid) CHECK(win[i][k] == (q < 0 ? -1 : ref->tok[q]));      // an acceptable row sees exactly the true sequence's window
                }
            }
        }
    }

    std::vector<int> draft(const Step& s)
    {
        Ds41RoundLogic::DraftIn in;
        bool run = logic.draft_in(s, in);
        if (run && mut == 2 && s.event == Step::TAIL && s.round < 0) run = false;
        if (!run)
        {
            ++skipped;
            CHECK(mut == 2 || (s.round < 0 && s.event != Step::TAIL));
            return logic.skipped(s);
        }
        if (mut == 1) ++in.acc;
        if (mut == 3) ++in.golden;
        ++executed;
        max_draft_M = std::max(max_draft_M, s.M());
        const bool skippable = s.round < 0 && s.event != Step::TAIL;
        CHECK(!skip || !skippable);
        CHECK(in.acc == in.q - tap_pos0);                                           // (b)
        CHECK(in.acc >= 0 && in.acc < (int)taps.size());
        CHECK(in.acc + 1 == s.M() || mut);
        CHECK(in.q + 1 < (int)ref->tok.size() && in.golden == ref->tok[in.q + 1]);
        CHECK(h[in.q] == ref->st[in.q]);
        const uint32_t s0 = in.acc >= 0 && in.acc < (int)taps.size() ? taps[in.acc] : 0;
        CHECK(s0 == ref->st[in.q]);
        // the fake draft device: the true continuation from the tap state and golden, then the injected error
        int c[Ds41RoundLogic::BLOCK], dr[Ds41RoundLogic::BLOCK];
        uint32_t st = step_state(s0, in.golden);
        for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i) { c[i] = sample(st); st = step_state(st, c[i]); }
        const int w = pat == ALL ? 5 : pat == FIRST ? 0 : pat == LAST ? K - 1 : (int)(mix((uint32_t)in.q * 2654435761u + 17) % 6);
        for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i)
            dr[i] = i < w ? c[i] : i == w ? (c[i] + 1 + i) % V : (mix((uint32_t)in.q * 31u + i) & 1 ? c[i] : (c[i] + 2 + i) % V);
        if (!insert_mode && in.q + 1 >= plen)                                       // from the tail on the stream is the model's own
            for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i)
                if (in.q + 2 + i < (int)ref->tok.size()) CHECK(c[i] == ref->tok[in.q + 2 + i]);
        if (!skippable) dlog.push_back({in.q, in.golden, in.acc, dr[0], dr[1], dr[2], dr[3], dr[4]});
        last_q = in.q; drafted = true; chains_since = 0;
        std::copy(dr, dr + Ds41RoundLogic::BLOCK, last_draft);
        return logic.draft_out(s, in.q, dr);
    }

    std::vector<int> chain(const Step& s)
    {
        ++chains; ++chains_since;
        const std::vector<int> out = logic.chain(s);                                // throws if it does not continue the cached draft
        CHECK(s.M() == 1 && chains_since < Ds41RoundLogic::BLOCK && s.r[0].pos == last_q + chains_since);
        CHECK(chains_since < Ds41RoundLogic::BLOCK && s.r[0].tok == last_draft[chains_since - 1]);
        CHECK(out.size() == 1 && chains_since < Ds41RoundLogic::BLOCK && out[0] == last_draft[chains_since]);
        return out;
    }
};

struct Outcome
{
    std::vector<int> res[2];
    std::vector<long> acc[2];
    std::vector<DraftRec> dlog;
    long executed = 0, skipped = 0;
    int max_draft_M = 0;
};

static void run_request(Fake& f, Batch& b, const Session& s, int req, Outcome& o)
{
    const int n = (int)s.prompt.size(), K = f.K;
    const Ref ref = reference(s, s.n_gen + 2 * (K + 1) + 8);
    f.ref = &ref; f.plen = n;
    f.new_request();
    const long e0 = f.executed, sk0 = f.skipped, c0 = f.chains;
    try
    {
        b.admit({0}, {s});
        while (b.round({0})) {}
    }
    catch (const std::exception& e)
    {
        if (!quiet) std::printf("THROW [%s]: %s\n", ctx.c_str(), e.what());
        ++fails;
        b.release(0);
        return;
    }
    const std::vector<int> got = b.result(0);
    const std::vector<int> want(ref.tok.begin() + n, ref.tok.begin() + n + s.n_gen);
    CHECK(got == want);                                                             // (a) the canary: equal to MTP=0
    const Stats& st = b.stats(0);
    CHECK((int)st.acc.size() == K + 1);
    o.res[req] = got;
    o.acc[req] = st.acc;
    const int PS = (n + s.chunk - 1) / s.chunk;
    if (!f.insert_mode)
    {
        const long rounds = st.rounds;
        CHECK(f.executed - e0 == (f.skip ? 1 : PS) + rounds);                       // the tail (or every chunk's) draft, one per round
        CHECK(f.skipped - sk0 == (f.skip ? PS - 1 : 0));
        CHECK(f.chains - c0 == (rounds + 1) * (K - 1));                             // (c) K - 1 chain steps after the tail and every round
        if (f.pat == ALL) { CHECK(st.acc[K] == rounds && rounds == (s.n_gen - 1 + K) / (K + 1)); }
        if (f.pat == FIRST) CHECK(st.acc[0] == rounds);
        if (f.pat == LAST) CHECK(st.acc[K - 1] == rounds);
        if (f.pat == MIX) for (int a = 0; a <= K; ++a) CHECK(st.acc[a] > 0);        // every accept count occurs
    }
    b.release(0);
}

struct RingTotals { long windows = 0, steps = 0, split = 0, valid_rows = 0, rejected_rows = 0, refed_rejected = 0; } g_ring;

static Outcome run_cfg(int K, int chunk, int plen, Pattern pat, bool skip, const Session* ins, int mut = 0)
{
    Outcome o;
    Fake f(K, pat, ins != nullptr, skip);
    f.mut = mut;
    Hooks hk = f.hooks();
    Batch b(K, 64, hk);
    for (int req = 0; req < 2; ++req)                                               // two requests back to back on the same engine state
    {
        Session s;
        if (ins) s = *ins;
        else s.n_gen = 300;
        const int len = req ? std::max(1, plen / 2 + 1) : plen;
        s.prompt.clear();
        for (int i = 0; i < len; ++i) s.prompt.push_back((int)(mix((uint32_t)(i * 7 + req * 1000 + 3)) % V));
        s.first = 0; s.chunk = chunk;
        ctx = std::string("K ") + std::to_string(K) + " chunk " + std::to_string(chunk) + " len " + std::to_string(len) + " " + PN[pat] +
              (skip ? "" : " noskip") + (ins ? " insert" : "") + " req " + std::to_string(req) + (mut ? " MUTATION" : "");
        run_request(f, b, s, req, o);
    }
    o.dlog = f.dlog; o.executed = f.executed; o.skipped = f.skipped; o.max_draft_M = f.max_draft_M;
    if (!mut)
    {
        g_ring.windows += f.ring_windows; g_ring.steps += f.ring_steps; g_ring.split += f.ring_split;
        g_ring.valid_rows += f.ring_valid_rows; g_ring.rejected_rows += f.ring_rejected_rows; g_ring.refed_rejected += f.ring_refed_rejected;
    }
    return o;
}

// ---- the logic class alone: every fail-closed rule
static Step mkstep(Step::Kind kind, int round, Step::Event ev, int pos0, int M, int nsess = 1)
{
    Step s;
    s.kind = kind; s.round = round; s.event = ev;
    for (int q = 0; q < nsess; ++q) { s.sess.push_back(q); s.row0.push_back(q * M); s.rows.push_back(M); }
    for (int q = 0; q < nsess; ++q)
        for (int i = 0; i < M; ++i) s.r.push_back(Row{q, 100 + i, pos0 + i, pos0 + i + 1, -1});
    return s;
}
template <class F> static bool throws(F f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; }

static void logic_tests()
{
    ctx = "logic";
    CHECK(throws([] { Ds41RoundLogic l(0); }));
    CHECK(throws([] { Ds41RoundLogic l(6); }));
    Ds41RoundLogic l(5);
    const int dr[5] = {11, 12, 13, 14, 15};
    Ds41RoundLogic::DraftIn in;
    CHECK(throws([&] { l.chain(mkstep(Step::CHAIN, 0, Step::ROUND, 11, 1)); }));                   // no draft yet
    CHECK(!l.draft_in(mkstep(Step::DRAFT, -1, Step::ROUND, 0, 8), in));                           // prompt chunk before the tail: skipped
    CHECK(l.draft_in(mkstep(Step::DRAFT, -1, Step::TAIL, 8, 3), in) && in.q == 10 && in.golden == 102 && in.acc == 2);
    CHECK(l.draft_in(mkstep(Step::DRAFT, 7, Step::ROUND, 20, 6), in) && in.q == 25 && in.golden == 105 && in.acc == 5);
    CHECK(l.draft_in(mkstep(Step::DRAFT, 7, Step::DECODE1, 20, 1), in) && in.q == 20 && in.acc == 0);
    CHECK(l.draft_in(mkstep(Step::DRAFT, 0, Step::ROUND, 20, 64), in) && in.acc == 63);          // a forced chunk of 64 rows
    CHECK(throws([&] { l.draft_in(mkstep(Step::DRAFT, 0, Step::ROUND, 20, 3, 2), in); }));        // two sessions
    CHECK(throws([&] { l.draft_in(mkstep(Step::MAIN, 0, Step::ROUND, 20, 3), in); }));
    Step gap = mkstep(Step::DRAFT, 0, Step::ROUND, 20, 3);
    gap.r[2].pos += 1;
    CHECK(throws([&] { l.draft_in(gap, in); }));                                                  // rows not consecutive
    const Step d = mkstep(Step::DRAFT, 0, Step::ROUND, 6, 5);                                     // q = 10
    const std::vector<int> u = l.draft_out(d, 10, dr);
    CHECK(u.size() == 5 && u[4] == 11 && u[0] == 0 && u[3] == 0);
    for (int j = 1; j < 5; ++j)
    {
        Step c = mkstep(Step::CHAIN, 0, Step::ROUND, 10 + j, 1);
        c.r[0].tok = dr[j - 1];
        const std::vector<int> o = l.chain(c);
        CHECK(o.size() == 1 && o[0] == dr[j]);
    }
    Step c = mkstep(Step::CHAIN, 0, Step::ROUND, 15, 1);                                          // j = 5 = k: past the chain
    c.r[0].tok = dr[4];
    CHECK(throws([&] { l.chain(c); }));
    Step wrongtok = mkstep(Step::CHAIN, 0, Step::ROUND, 12, 1);
    wrongtok.r[0].tok = 99;
    CHECK(throws([&] { l.chain(wrongtok); }));
    CHECK(throws([&] { l.chain(mkstep(Step::CHAIN, 0, Step::ROUND, 10, 1)); }));                  // j = 0
    CHECK(throws([&] { l.chain(mkstep(Step::CHAIN, 0, Step::ROUND, 12, 2)); }));                  // two rows
    CHECK(throws([&] { l.chain(mkstep(Step::CHAIN, 0, Step::ROUND, 12, 1, 2)); }));               // two sessions
    Ds41RoundLogic l3(3);
    l3.draft_out(d, 10, dr);
    Step j3 = mkstep(Step::CHAIN, 0, Step::ROUND, 13, 1);                                         // j = 3 = k: past the chain of k = 3
    j3.r[0].tok = dr[2];
    CHECK(throws([&] { l3.chain(j3); }));
    l.reset();
    Step again = mkstep(Step::CHAIN, 0, Step::ROUND, 11, 1);
    again.r[0].tok = dr[0];
    CHECK(throws([&] { l.chain(again); }));                                                       // reset: the cached draft is stale
    Ds41RoundLogic ns(5, false);                                                                  // the no-skip variant drafts every chunk
    CHECK(ns.draft_in(mkstep(Step::DRAFT, -1, Step::ROUND, 0, 8), in) && in.acc == 7);
}


// ---- SESS > 1 (ds41_dspark_round.cuh draft_step_sessions / chain_step, one Ds41RoundLogic per S2 slot): the real Batch with S = 2..8 sessions
// at once, sessions of different lengths (they finish at different rounds) and a second wave admitted on freed slots (the bind's per-slot reset).
// Checks: (a) every session's stream equals its reference; (h) every MAIN step of several sessions has the same rows per session (s2/ds41_native.cuh
// refuses anything else: the captured [ns][rps] graphs) and at most S x (K + 1) <= 48 rows; (b) per drafted session acc == q - its row 0 position
// in the latest MAIN step, crow0 = its row0 in that step (Launch::source_rows), the tap at crow0 + acc is the true state at q; (c) every CHAIN row
// continues its own slot's cached draft. A mutation (session j's crow0 taken from session 0) must fail.
struct FakeMS
{
    int K, mut = 0;
    std::vector<Ds41RoundLogic> logic;
    std::vector<const Ref*> ref = std::vector<const Ref*>(S2N_SLOTS, nullptr);
    std::vector<std::vector<uint32_t>> h = std::vector<std::vector<uint32_t>>(S2N_SLOTS, std::vector<uint32_t>(4096, 0xDEADBEEFu));
    std::vector<uint32_t> taps;                        // the states of every row of the latest MAIN step (all sessions)
    std::vector<int> src = std::vector<int>(S2N_SLOTS, 0), pos0 = std::vector<int>(S2N_SLOTS, -1);   // per slot: row0 / row-0 position in its last MAIN
    long mains = 0, multi_mains = 0, drafts = 0, multi_drafts = 0, nonuniform_drafts = 0, chains = 0, multi_chains = 0, drafted_sessions = 0;
    int max_rows = 0, max_ns = 0, max_draft_rows = 0;
    explicit FakeMS(int k) : K(k), logic(size_t(S2N_SLOTS), Ds41RoundLogic(k)) {}
    Hooks hooks()
    {
        Hooks hk;
        hk.run = [this](const Step& s) -> std::vector<int> { return s.kind == Step::MAIN ? main(s) : s.kind == Step::DRAFT ? draft(s) : chain(s); };
        hk.emit = [](int, const int*, int, int) { return false; };
        return hk;
    }
    std::vector<int> main(const Step& s)
    {
        ++mains;
        const size_t ns = s.sess.size();
        max_rows = std::max(max_rows, s.M()); max_ns = std::max(max_ns, (int)ns);
        CHECK(s.M() <= 48);
        if (ns > 1) { ++multi_mains; for (size_t j = 0; j < ns; ++j) CHECK(s.rows[j] == s.rows[0] && s.row0[j] == (int)j * s.rows[0]); }   // (h)
        std::vector<int> out(size_t(s.M()), 0);
        taps.assign(size_t(s.M()), 0);
        for (size_t j = 0; j < ns; ++j)
        {
            const int q = s.sess[j], r0 = s.row0[j];
            const Ref& rf = *ref[size_t(q)];
            auto& hq = h[size_t(q)];
            const int p0 = s.r[size_t(r0)].pos;
            CHECK(s.r[size_t(r0)].tok == rf.tok[size_t(p0)]);
            CHECK(p0 == 0 || hq[size_t(p0 - 1)] == rf.st[size_t(p0 - 1)]);
            for (int i = 0; i < s.rows[j]; ++i)
            {
                const Row& r = s.r[size_t(r0 + i)];
                CHECK(r.sess == q && r.pos == p0 + i);
                hq[size_t(r.pos)] = step_state(r.pos == 0 ? SEED : hq[size_t(r.pos - 1)], r.tok);
                taps[size_t(r0 + i)] = hq[size_t(r.pos)];
                out[size_t(r0 + i)] = sample(hq[size_t(r.pos)]);
            }
            src[size_t(q)] = r0; pos0[size_t(q)] = p0;
        }
        return out;
    }
    std::vector<int> draft(const Step& s)
    {
        ++drafts;
        const size_t ns = s.sess.size();
        if (ns > 1) ++multi_drafts;
        for (size_t j = 1; j < ns; ++j) if (s.rows[j] != s.rows[0]) { ++nonuniform_drafts; break; }
        std::vector<int> u(size_t(s.M()), 0);
        int S = 0;
        for (size_t j = 0; j < ns; ++j)
        {
            const int q = s.sess[j];
            Ds41RoundLogic::DraftIn in;
            if (!logic[size_t(q)].draft_in_sess(s, j, in)) continue;
            ++S; ++drafted_sessions;
            const Ref& rf = *ref[size_t(q)];
            const int crow0 = mut == 1 ? src[size_t(s.sess[0])] : src[size_t(q)];           // the runner's crow0 (Launch::source_rows)
            CHECK(in.acc == in.q - pos0[size_t(q)] && in.acc + 1 == s.rows[j]);              // (b)
            CHECK(in.golden == rf.tok[size_t(in.q + 1)]);
            const int tr = crow0 + in.acc;
            CHECK(tr >= 0 && tr < (int)taps.size() && taps[size_t(tr)] == rf.st[size_t(in.q)]);
            const uint32_t s0 = tr >= 0 && tr < (int)taps.size() ? taps[size_t(tr)] : 0;
            int c[Ds41RoundLogic::BLOCK], dr[Ds41RoundLogic::BLOCK];
            uint32_t st = step_state(s0, in.golden);
            for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i) { c[i] = sample(st); st = step_state(st, c[i]); }
            const int w = (int)(mix((uint32_t)in.q * 2654435761u + 17u + (uint32_t)q * 977u) % 6);
            for (int i = 0; i < Ds41RoundLogic::BLOCK; ++i) dr[i] = i < w ? c[i] : (c[i] + 1 + i) % V;
            u[size_t(s.row0[j] + s.rows[j] - 1)] = logic[size_t(q)].draft_set(in.q, dr);
        }
        max_draft_rows = std::max(max_draft_rows, S * Ds41RoundLogic::BLOCK);
        CHECK(S * Ds41RoundLogic::BLOCK <= 48);
        return u;
    }
    std::vector<int> chain(const Step& s)
    {
        ++chains;
        if (s.sess.size() > 1) ++multi_chains;
        std::vector<int> u(size_t(s.M()), 0);
        for (size_t j = 0; j < s.sess.size(); ++j)
        {
            CHECK(s.rows[j] == 1);
            u[size_t(s.row0[j])] = logic[size_t(s.sess[j])].chain_row(s.r[size_t(s.row0[j])]);   // (c) throws if not its own slot's draft
        }
        return u;
    }
};

static Session ms_session(int q, int wave, int K)
{
    Session s;
    s.n_gen = 40 + 23 * q + 11 * wave + 3 * K;
    const int len = 1 + (q * 5 + wave * 3) % 9;
    for (int i = 0; i < len; ++i) s.prompt.push_back((int)(mix((uint32_t)(i * 7 + q * 131 + wave * 1000 + 5)) % V));
    s.first = 0; s.chunk = 8;
    return s;
}

// returns the failed checks of one configuration (S sessions, draft length K)
static int run_ms(int S, int K, int mut, FakeMS** keep = nullptr)
{
    const int before = fails;
    static FakeMS* last = nullptr;
    delete last;
    last = new FakeMS(K);
    FakeMS& f = *last;
    f.mut = mut;
    if (keep) *keep = last;
    Hooks hk = f.hooks();
    Batch b(K, 48, hk);
    std::vector<Ref> refs(size_t(2 * S2N_SLOTS));
    std::vector<int> wave(size_t(S2N_SLOTS), 0), order;
    std::vector<Session> sess;
    for (int q = 0; q < S; ++q)
    {
        sess.push_back(ms_session(q, 0, K));
        refs[size_t(q)] = reference(sess.back(), sess.back().n_gen + 2 * (K + 1) + 8);
        f.ref[size_t(q)] = &refs[size_t(q)];
        f.logic[size_t(q)].reset();
        order.push_back(q);
    }
    ctx = "SESS " + std::to_string(S) + " K " + std::to_string(K) + (mut ? " MUTATION" : "");
    try
    {
        b.admit(order, sess);
        for (;;)
        {
            const bool more = b.round(order);
            for (int q = 0; q < S; ++q)                // a finished session: check it, free the slot, and (first wave only) admit the next
            {
                if (wave[size_t(q)] > 1 || b.active(q)) continue;
                const Session s = ms_session(q, wave[size_t(q)], K);
                const Ref& rf = refs[size_t(wave[size_t(q)] * S2N_SLOTS + q)];
                const std::vector<int> got = b.result(q);
                const int n = (int)s.prompt.size();
                CHECK(got == std::vector<int>(rf.tok.begin() + n, rf.tok.begin() + n + s.n_gen));   // (a)
                b.release(q);
                if (++wave[size_t(q)] == 2) continue;
                const Session s2 = ms_session(q, 1, K);
                refs[size_t(S2N_SLOTS + q)] = reference(s2, s2.n_gen + 2 * (K + 1) + 8);
                f.ref[size_t(q)] = &refs[size_t(S2N_SLOTS + q)];
                f.logic[size_t(q)].reset();            // the bind of a fresh run on slot q (ds41_native.cuh: round->reset(q))
                b.admit({q}, {s2});
            }
            if (!more && std::none_of(order.begin(), order.end(), [&](int q) { return b.active(q); })) break;
        }
    }
    catch (const std::exception& e)
    {
        if (!quiet) std::printf("THROW [%s]: %s\n", ctx.c_str(), e.what());
        ++fails;
    }
    for (int q = 0; q < S; ++q) CHECK(wave[size_t(q)] == 2);
    if (S > 1 && !mut) { CHECK(f.multi_mains > 0 && f.multi_drafts > 0 && (K == 1 || f.multi_chains > 0)); }   // K = 1: no chain steps
    return fails - before;
}

int main()
{
    logic_tests();
    // ---- the sweep: every K, chunk size, prompt length (one chunk, several, remainder, one token), error pattern; skip vs no skip
    long runs = 0, big_tail = 0;
    for (int K = 1; K <= 5; ++K)
        for (int chunk : {1, 3, 8, 64})
            for (int plen : {1, chunk, chunk + 1, 2 * chunk + 3, 17})
                for (int pat = 0; pat < NPAT; ++pat)
                {
                    const Outcome a = run_cfg(K, chunk, plen, (Pattern)pat, true, nullptr);
                    const Outcome b = run_cfg(K, chunk, plen, (Pattern)pat, false, nullptr);
                    ctx = std::string("K ") + std::to_string(K) + " chunk " + std::to_string(chunk) + " len " + std::to_string(plen) + " " + PN[pat] + " skip vs noskip";
                    CHECK(a.res[0] == b.res[0] && a.res[1] == b.res[1]);            // (e) the skipped drafts change nothing
                    CHECK(a.acc[0] == b.acc[0] && a.acc[1] == b.acc[1]);
                    CHECK(a.dlog == b.dlog);
                    CHECK(a.skipped >= 0 && b.skipped == 0 && b.executed >= a.executed);
                    runs += 2;
                    big_tail += a.max_draft_M > Ds41RoundLogic::BLOCK + 1;
                }
    CHECK(big_tail > 0);                                                            // a tail draft of more than 6 rows ran
    // ---- the insert rule: forced chunks draft with their own M (up to the chunk, well over 6) and chain; the stream is still the reference
    long forced_big = 0;
    for (int K : {1, 2, 5})
        for (int chunk : {3, 8, 64})
            for (int ntok : {1, 5, 20})
                for (int pat = 0; pat < NPAT; ++pat)
                    for (int unless : {-1, 0})
                    {
                        Session s;
                        s.n_gen = 120; s.ins.after = 6 + ntok % 4; s.ins.unless = unless;
                        for (int i = 0; i < ntok; ++i) s.ins.tokens.push_back(500 + 7 * i);
                        const Outcome a = run_cfg(K, chunk, 10, (Pattern)pat, true, &s);
                        const Outcome b = run_cfg(K, chunk, 10, (Pattern)pat, false, &s);
                        ctx = "insert skip vs noskip";
                        CHECK(a.res[0] == b.res[0] && a.res[1] == b.res[1] && a.dlog == b.dlog);
                        runs += 2;
                        forced_big += a.max_draft_M > Ds41RoundLogic::BLOCK + 1;
                    }
    CHECK(forced_big > 0);
    {   // one line of what the sweep exercised: K = 5, chunk 8, 20 prompt tokens, mixed errors
        const Outcome o = run_cfg(5, 8, 20, MIX, true, nullptr);
        std::printf("K=5 chunk 8 len 20 mix, request 0: accept counts 0..5 =");
        for (long a : o.acc[0]) std::printf(" %ld", a);
        std::printf("; %ld drafts run, %ld prompt drafts skipped (2 requests); largest draft step %d rows\n", o.executed, o.skipped, o.max_draft_M);
    }
    CHECK(g_ring.windows > 0 && g_ring.rejected_rows > 0 && g_ring.refed_rejected > 0 && g_ring.split > 0);   // (g) rejects, re-fed positions and long steps occurred
    std::printf("Engram ring fed from the Batch's main steps: %ld steps (%ld long steps split at MX = 32), %ld windows all equal the true sequence's; "
                "%ld rows acceptable, %ld rows with a rejected draft token, %ld rejected-draft positions rewritten by the next step\n",
                g_ring.steps, g_ring.split, g_ring.windows, g_ring.valid_rows, g_ring.rejected_rows, g_ring.refed_rejected);
    // ---- teeth: each mutation of the host side must be caught
    const int clean = fails;
    quiet = true;
    int caught = 0;
    for (int mut = 1; mut <= 6; ++mut)
    {
        const int before = fails;
        run_cfg(5, 8, 20, MIX, true, nullptr, mut);
        caught += fails > before;
        std::printf("mutation %d: %d checks failed\n", mut, fails - before);
    }
    fails = clean;
    quiet = false;
    ctx = "mutations";
    CHECK(caught == 6);
    {   // the draft fabric error text names the collective and the peer (first-light failure of 10-25: no error word reached the log)
        ctx = "fabric error text";
        const std::string a = g6s2::draft_fabric_error_text(0x62000601u), b = g6s2::draft_fabric_error_text(0x61000503u);
        CHECK(a == "DSpark draft fabric error 0x62000601: head merge wait timed out, collective 6, peer rank 1");
        CHECK(b == "DSpark draft fabric error 0x61000503: layer exchange wait timed out, collective 5, peer rank 3");
        CHECK(g6s2::draft_fabric_error_text(0x7000000au) == "DSpark draft fabric error 0x7000000a: proxy doorbell timeout, collective 10");
        CHECK(g6s2::draft_fabric_error_text(0x71000000u) == "DSpark draft fabric error 0x71000000: work completion error");
        CHECK(g6s2::draft_fabric_error_text(0x1u) == "DSpark draft fabric error 0x00000001");
    }
    {   // ---- SESS > 1: 1..8 sessions x K 1..5 (8 x 6 = 48 verify rows), two waves per slot
        long nonuni = 0, multi = 0;
        for (int S = 1; S <= 8; ++S)
            for (int K = 1; K <= 5; ++K)
            {
                if (S * (K + 1) > 48) continue;
                FakeMS* f = nullptr;
                run_ms(S, K, 0, &f);
                nonuni += f->nonuniform_drafts; multi += f->multi_mains;
                if (S == 8 && K == 5)
                    std::printf("SESS 8 K 5: %ld main steps (%ld of several sessions, widest %d rows, %d sessions), %ld draft steps (%ld of several, "
                                "%ld with uneven rows per session, widest block %d rows), %ld chain steps (%ld of several), %ld session drafts\n",
                                f->mains, f->multi_mains, f->max_rows, f->max_ns, f->drafts, f->multi_drafts, f->nonuniform_drafts, f->max_draft_rows,
                                f->chains, f->multi_chains, f->drafted_sessions);
                runs += 1;
            }
        ctx = "SESS sweep";
        CHECK(nonuni > 0 && multi > 0);                                              // uneven draft rows (acc + 1 per session) occurred
        quiet = true;
        const int c0 = fails;
        const int bad = run_ms(4, 5, 1);                                             // crow0 of session 0 for every session: must fail
        fails = c0;
        quiet = false;
        std::printf("SESS mutation (crow0 of session 0): %d checks failed\n", bad);
        ctx = "SESS mutation";
        CHECK(bad > 0);
    }
    std::printf("batch_ds41_dspark_test: %ld Batch runs (2 requests each), logic tests, ring windows, 6/6 mutations caught, SESS 1..8: %s\n", runs, fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
