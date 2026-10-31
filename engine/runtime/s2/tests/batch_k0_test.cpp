// K = 0: a Batch without a draft head (DS41 with no drafter, --mtp 0). One round = ONE main step, one row per session; no DRAFT
// step and no CHAIN step is ever run (the DS41 build has no MTP layer, so a draft step would run GLM's), and St::d is never
// written past its end. Also pins the K >= 1 behaviour: the digest lines (DIGEST K=...) of this test are the same before
// and after the K = 0 edit of s2_batch.h / scheduler.h / sessions.h (compare the stdout of this file built against the
// two trees: step counts by kind, row count, St::d / St::dlm sizes, an order-sensitive hash of every step, the tokens).
// Host only. Build (ASAN + UBSAN + container checks catch a write past St::d):
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined -D_GLIBCXX_ASSERTIONS
//       -I engine/runtime engine/runtime/s2/tests/batch_k0_test.cpp -o /tmp/batch_k0_test
// and with -DG6_MODEL_DS41=1 for the DS41 parts (Config::validate accepts mtp 0, Sessions::payload decodes every prompt, the
// Driver serves a request at mtp 0 end to end); without it mtp 0 is refused and plp < 0 still means a tensor prefill.
// The fake model is a pure function of (session salt, row position, row token), as batch_round_test's.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#define private public   // the test reads Batch::st_ (the size of St::d); every standard header it needs is included above
#include "s2_batch.h"
#undef private
#include "s2/sessions.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
static uint64_t fnv(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001B3ull; }

struct Trace
{
    struct Rec { int kind, tok, pos, spos; };
    std::vector<std::vector<Rec>> rows;                    // per session (slot): its rows in step order
    long steps[4] = {};                                    // by Step::Kind
    long nrows = 0;
    uint64_t hash = 0xCBF29CE484222325ull;                 // order-sensitive over every step: kind, cls, round, every row
    std::vector<int> main_m;                               // the row count of each MAIN step
};

// salt[q] = the identity of the session in slot q; V = the vocabulary (2: drafts match often enough to give every accept count)
static s2::Hooks make_hooks(const std::vector<uint32_t>& salt, Trace& tr, int V)
{
    s2::Hooks h;
    tr.rows.assign(salt.size(), {});
    h.run = [&salt, &tr, V](const s2::Step& s) {
        const uint32_t k = (uint32_t)s.kind;
        ++tr.steps[k & 3];
        tr.hash = fnv(fnv(fnv(tr.hash, k), (uint64_t)s.cls), (uint64_t)(s.round + 7));
        if (s.kind == s2::Step::MAIN) tr.main_m.push_back(s.M());
        std::vector<int> out;
        for (const s2::Row& r : s.r)
        {
            int t = (int)(mix(salt[r.sess], k == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
            if (k != s2::Step::MAIN && mix(salt[r.sess], r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;   // a wrong guess one time in three
            out.push_back(t);
            tr.rows[r.sess].push_back({(int)k, r.tok, r.pos, r.spos});
            tr.hash = fnv(fnv(fnv(fnv(tr.hash, (uint64_t)r.sess), (uint64_t)(r.tok + 1)), (uint64_t)r.pos), (uint64_t)r.spos);
            ++tr.nrows;
        }
        return out;
    };
    return h;
}

static s2::Session session(int salt, int plen, int n_gen, int chunk, int V)
{
    s2::Session s;
    for (int i = 0; i < plen; ++i) s.prompt.push_back((int)(mix(salt, i, 77) % V));
    s.n_gen = n_gen; s.chunk = chunk;
    return s;
}

// the tokens a session of K = 0 produces: each is the model at (its last position, its last token)
static std::vector<int> reference(const s2::Session& s, uint32_t salt, int V)
{
    std::vector<int> t = s.prompt, out;
    for (int i = 0; i < s.n_gen; ++i)
    {
        const int x = (int)(mix(salt, (uint32_t)t.size() - 1, t.back()) % V);
        t.push_back(x); out.push_back(x);
    }
    return out;
}

struct RunOut { std::vector<std::vector<int>> out; std::vector<s2::Stats> st; Trace tr; std::vector<size_t> dsize, dlmsize; };

// all sessions in one Batch, the way mtp_batch runs them, keeping the Batch to read St::d
static RunOut run(const std::vector<s2::Session>& S, const std::vector<uint32_t>& salt, int K, int max_rows, int V, const s2::KStop* ks = nullptr)
{
    RunOut r;
    s2::Batch b(K, max_rows, make_hooks(salt, r.tr, V));
    if (ks) b.kstop(*ks);
    const int NS = (int)S.size();
    std::vector<int> qs(NS);
    for (int q = 0; q < NS; ++q) qs[q] = q;
    b.admit(qs, S);
    while (b.round(qs)) {}
    r.out.resize(NS); r.st.resize(NS);
    for (int q = 0; q < NS; ++q)
    {
        r.out[q] = b.result(q); r.st[q] = b.stats(q);
        r.dsize.push_back(b.st_[q].d.size()); r.dlmsize.push_back(b.st_[q].dlm.size());
    }
    return r;
}

static bool same_rows(const Trace& a, int qa, const Trace& b, int qb)
{
    if (a.rows[qa].size() != b.rows[qb].size()) return false;
    for (size_t i = 0; i < a.rows[qa].size(); ++i)
    {
        const auto &x = a.rows[qa][i], &y = b.rows[qb][i];
        if (x.kind != y.kind || x.tok != y.tok || x.pos != y.pos || x.spos != y.spos) return false;
    }
    return true;
}

// K = 0 through a Batch: tokens are the model's own chain, one MAIN step per round, no draft or chain step, St::d intact
static void k0_batch()
{
    const int V = 7, K = 0;
    // sessions differ in prompt length, chunk and n_gen, so they finish in different rounds; n_gen 1 ends at the prompt tail
    std::vector<s2::Session> S = {session(11, 5, 14, 4, V), session(12, 9, 6, 1, V), session(13, 3, 20, 2, V), session(14, 1, 9, 4, V),
                                  session(15, 8, 1, 8, V), session(16, 4, 11, 3, V)};
    const int NS = (int)S.size();
    std::vector<uint32_t> salt;
    for (int q = 0; q < NS; ++q) salt.push_back(11 + q);
    const RunOut bt = run(S, salt, K, 8, V);
    CHECK(bt.tr.steps[s2::Step::DRAFT] == 0 && bt.tr.steps[s2::Step::CHAIN] == 0);
    long prompt_steps = 0;
    for (const auto& s : S) prompt_steps += (long)((s.prompt.size() + s.chunk - 1) / s.chunk);
    long round_steps = 0;
    {   // a round runs while any session still has tokens to make: the longest session's n_gen - 1 rounds, fewer sessions as they finish
        int mx = 0;
        for (const auto& s : S) mx = std::max(mx, s.n_gen - 1);
        round_steps = mx;
    }
    CHECK(bt.tr.steps[s2::Step::MAIN] == prompt_steps + round_steps);
    for (int q = 0; q < NS; ++q)
    {
        CHECK(bt.out[q] == reference(S[q], salt[q], V));
        CHECK(bt.st[q].rounds == S[q].n_gen - 1 && bt.st[q].tokens == S[q].n_gen - 1 && bt.st[q].drafted == 0);
        CHECK(bt.st[q].acc.size() == 1 && bt.st[q].acc[0] == S[q].n_gen - 1);
        CHECK(bt.dsize[q] == 2 && bt.dlmsize[q] == 2);                 // d[1] exists at K = 0, and is never written
        for (const auto& r : bt.tr.rows[q]) CHECK(r.kind == s2::Step::MAIN);
        // row invariance: the session alone gives the same rows in the same order
        std::vector<uint32_t> one = {salt[q]};
        const RunOut so = run({S[q]}, one, K, 8, V);
        CHECK(so.out[0] == bt.out[q] && same_rows(so.tr, 0, bt.tr, q));
    }
    // every decode round is one MAIN step with one row per active session
    {
        // the steps after the prompts: a MAIN step of rows == active sessions; they shrink as sessions finish
        const std::vector<int>& m = bt.tr.main_m;
        CHECK((long)m.size() == prompt_steps + round_steps);
        int prev = NS + 1;
        for (size_t i = (size_t)prompt_steps; i < m.size(); ++i) { CHECK(m[i] >= 1 && m[i] <= prev); prev = m[i]; }
    }
    // an insert rule (forced steps) at K = 0: tokens of the rule go in at `after`, no draft or chain step runs for them
    {
        std::vector<s2::Session> T = {session(21, 6, 12, 4, V), session(22, 4, 10, 2, V)};
        T[0].ins.after = 2; T[0].ins.unless = -1; T[0].ins.tokens = {3, 5, 1};
        T[1].ins.after = 0; T[1].ins.unless = -1; T[1].ins.tokens = {2, 2};
        std::vector<uint32_t> sl = {21, 22};
        const RunOut t = run(T, sl, K, 8, V);
        CHECK(t.tr.steps[s2::Step::DRAFT] == 0 && t.tr.steps[s2::Step::CHAIN] == 0);
        for (int q = 0; q < 2; ++q)
        {
            CHECK((int)t.out[q].size() == T[q].n_gen);
            const auto& o = t.out[q];
            CHECK(std::equal(T[q].ins.tokens.begin(), T[q].ins.tokens.end(), o.begin() + T[q].ins.after));
            std::vector<uint32_t> one = {sl[q]};
            const RunOut so = run({T[q]}, one, K, 8, V);
            CHECK(so.out[0] == o && same_rows(so.tr, 0, t.tr, q));
            CHECK(t.dsize[q] == 2);
        }
    }
    // KStop's kmax cannot enlarge St::d at K = 0 either (it applies only with mode on): d stays K + 1 .. kmax + 1 wide, never below 2
    {
        s2::KStop ks; ks.mode = 1; ks.kmax = 3;
        const RunOut t = run({S[0]}, {salt[0]}, K, 8, V, &ks);
        CHECK(t.tr.steps[s2::Step::DRAFT] == 0 || t.dsize[0] >= 2);
    }
}

// K >= 1: the DIGEST lines (the same text before and after the edit)
static void k_digest()
{
    const int V = 2;
    for (int K = 1; K <= 3; ++K)
    {
        std::vector<s2::Session> S = {session(11, 5, 14, 4, V), session(12, 9, 6, 4, V), session(13, 3, 20, 2, V), session(14, 6, 9, 4, V),
                                      session(15, 4, 11, 4, V), session(16, 7, 13, 3, V)};
        S[1].ins.after = 1; S[1].ins.unless = 1; S[1].ins.tokens = {0, 1, 0, 0};   // forced steps too
        S[4].ins.after = 3; S[4].ins.unless = -1; S[4].ins.tokens = {1, 1};
        const int NS = (int)S.size();
        std::vector<uint32_t> salt;
        for (int q = 0; q < NS; ++q) salt.push_back(11 + q);
        for (int pass = 0; pass < 3; ++pass)
        {
            s2::KStop ks;
            if (pass == 1) { ks.mode = 1; ks.kmax = K + 2; }       // St::d is kmax + 1 wide
            if (pass == 2) { ks.mode = 1; ks.kmax = K; ks.solo = true; }
            const RunOut r = run(S, salt, K, NS * (K + 1), V, pass ? &ks : nullptr);
            uint64_t th = 0xCBF29CE484222325ull;
            for (const auto& o : r.out) { for (int t : o) th = fnv(th, (uint64_t)(t + 1)); th = fnv(th, 0xFFFF); }
            std::string ds;
            for (size_t q = 0; q < r.dsize.size(); ++q) ds += (q ? "," : "") + std::to_string(r.dsize[q]) + "/" + std::to_string(r.dlmsize[q]);
            uint64_t acc = 0;
            for (const auto& s : r.st) for (long a : s.acc) acc = acc * 31 + (uint64_t)a;
            std::printf("DIGEST K=%d pass=%d main=%ld draft=%ld chain=%ld rows=%ld dsize=%s hash=%016llx tok=%016llx acc=%llu\n", K, pass,
                        r.tr.steps[s2::Step::MAIN], r.tr.steps[s2::Step::DRAFT], r.tr.steps[s2::Step::CHAIN], r.tr.nrows, ds.c_str(),
                        (unsigned long long)r.tr.hash, (unsigned long long)th, (unsigned long long)acc);
            CHECK(r.tr.steps[s2::Step::DRAFT] > 0);                // K >= 1 still drafts
            for (size_t q = 0; q < r.dsize.size(); ++q) CHECK(r.dsize[q] == (size_t)std::max(K, pass == 1 ? K + 2 : K) + 1);
        }
    }
}

using namespace g6s2;

#if G6_MODEL_DS41
// the DS41 serve path on the host: Config::validate with mtp 0, Sessions::payload (decode_prompt), a Driver serving three
// requests (A, B, A again) one after the other through one slot, every prompt as decode steps from position 0
struct Req { uint64_t id; std::vector<int> prompt; uint32_t max_tokens; };
static void driver_k0()
{
    const int V = 7;
    Config c;
    c.epoch = 1; c.mtp = 0; c.max_rows = 8; c.max_sessions = 1; c.prefill_rows = 64; c.pool_rows = 64 * 4; c.two_lanes = false;
    c.validate();
    Sessions ss(c, "/nonexistent-ds41-k0-test", false, 4);
    std::map<uint64_t, std::vector<int>> out;
    long steps[4] = {}, prefills = 0, main_rows = 0;
    std::vector<int> main_m;
    std::unique_ptr<Driver> d;
    DriverHooks h;
    h.bind = [](const Grant&) {};
    h.retire = [](const Slot&) {};
    h.prefill = [&](const Slot&, const Payload&, uint32_t, uint32_t) { ++prefills; };
    h.run = [&](const s2::Step& st) {
        std::vector<int> ids;
        ++steps[(unsigned)st.kind & 3];
        if (st.kind == s2::Step::MAIN) main_m.push_back(st.M());
        for (const auto& r : st.r)
        {
            const Slot& s = d->state().slots()[r.sess];
            ids.push_back((int)(mix((uint32_t)s.req.id, st.kind == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V));
            ++main_rows;
        }
        return ids;
    };
    h.emit = [&](Key k, const int* ids, int n, int) {
        const Slot& s = d->state().slots()[k.slot];
        auto& o = out[s.req.id];
        const int cnt = std::min<int>(n, int(s.req.max_tokens - o.size()));
        o.insert(o.end(), ids, ids + cnt);
        return Emission{uint32_t(cnt), o.size() == s.req.max_tokens};
    };
    d = std::make_unique<Driver>(c, h);
    std::deque<Req> queue;
    auto mk = [&](uint64_t id, const std::vector<int>& prompt, uint32_t mt) { Req r; r.id = id; r.max_tokens = mt; r.prompt = prompt; return r; };
    auto prompt_of = [&](uint32_t seed, int plen) { std::vector<int> v; for (int k = 0; k < plen; ++k) v.push_back((int)(mix(seed, k, 7) % V)); return v; };
    // request ids increase (the scheduler checks); the model's salt is the id. A, B, then A's prompt again (a second request),
    // then a one-token prompt
    const std::vector<int> pa = prompt_of(1, 9), pb = prompt_of(2, 5), pc = prompt_of(3, 1);
    queue = {mk(1, pa, 6), mk(2, pb, 4), mk(3, pa, 6), mk(4, pc, 3)};
    const std::vector<Req> all(queue.begin(), queue.end());
    long expect_main = 0;
    long guard = 0;
    while (!queue.empty() || d->state().resident())
    {
        const Scheduler& s = d->state();
        Commit cm;
        cm.epoch = 1; cm.event = s.next_event(); cm.boundary = s.boundary();
        Payload p;
        if (!queue.empty() && s.resident() < c.limit())
        {
            const Req& r = queue.front();
            const Request rq{r.id, uint32_t(r.prompt.size()), r.max_tokens};
            Run run;
            run.first_page = 0; run.pages = grant_pages(c, rq); run.table_version = 1;
            cm.admit = Grant{s.next_key(), rq, run, 0};
            g6rt::ServeReq sr; sr.prompt = r.prompt; sr.n_gen = int(r.max_tokens);   // plp -1: a plain request, no prompt logprobs asked
            p = ss.payload(sr);
            CHECK(p.decode_prompt && p.chunk == 4 && p.prompt == r.prompt);
            expect_main += (long)((r.prompt.size() + 3) / 4) + (long)r.max_tokens - 1;
        }
        d->pool_digest(0);
        cm.before_hash = d->state().digest();
        d->apply(cm, cm.admit ? &p : nullptr);
        if (cm.admit) queue.pop_front();
        // steps until the slot retires (the next event frees it)
        for (int k = 0; k < 64 && d->state().resident(); ++k)
        {
            bool all_retiring = true;
            for (const auto& sl : d->state().slots()) if (sl.phase != Phase::Free && sl.phase != Phase::Retiring) all_retiring = false;
            if (all_retiring) break;
            const auto q = d->step();
            CHECK(q.kind != Kind::Idle);
            if (q.kind == Kind::Idle) break;
        }
        if (++guard > 1000 || fails > 20) { CHECK(!"no progress"); break; }
    }
    CHECK(prefills == 0);                                  // decode_prompt: no tensor prefill quantum, ever
    CHECK(steps[s2::Step::DRAFT] == 0 && steps[s2::Step::CHAIN] == 0);
    CHECK(steps[s2::Step::MAIN] == expect_main);
    for (int m : main_m) CHECK(m >= 1 && m <= 4);          // a prompt chunk (4) or one decode row
    for (const Req& r : all)
    {
        s2::Session q; q.prompt = r.prompt; q.n_gen = int(r.max_tokens);
        CHECK(out[r.id] == reference(q, (uint32_t)r.id, V));
    }
    std::printf("driver mtp0: %ld MAIN steps (expected %ld), 0 DRAFT, 0 CHAIN, %ld prefill quanta, %zu requests\n", steps[s2::Step::MAIN], expect_main, prefills, out.size());
}
#endif

// Config::validate and Sessions::payload: mtp 0 is refused unless DS41; plp < 0 is a decode-only prompt only on DS41
static void config_payload()
{
    Config z;
    z.epoch = 1; z.mtp = 0; z.max_rows = 8; z.max_sessions = 1; z.prefill_rows = 64; z.pool_rows = 64 * 4; z.two_lanes = false;
    bool threw = false;
    try { z.validate(); } catch (const std::exception&) { threw = true; }
#if G6_MODEL_DS41
    CHECK(!threw);
#else
    CHECK(threw);                                         // GLM / Flash: mtp must be 1..7, as before
#endif
    Config c = z;
    c.mtp = 2;
    c.validate();
    Sessions ss(c, "/nonexistent-ds41-k0-test", false, 4);
    g6rt::ServeReq sr; sr.prompt = {1, 2, 3}; sr.n_gen = 4;
    sr.plp = -1;
#if G6_MODEL_DS41
    CHECK(ss.payload(sr).decode_prompt);
#else
    CHECK(!ss.payload(sr).decode_prompt);                 // a plain request prefills with tensors
#endif
    sr.plp = 2;
    CHECK(ss.payload(sr).decode_prompt);                  // prompt logprobs always decode the prompt
    // mtp 8 and mtp > 7 stay refused in both builds
    c.mtp = 8;
    threw = false;
    try { c.validate(); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

int main(int argc, char** argv)
{
    // `batch_k0_test digest`: only the K >= 1 DIGEST lines (built against the pre-edit headers to compare them)
    if (argc > 1 && !std::strcmp(argv[1], "digest")) { k_digest(); return fails ? 1 : 0; }
    config_payload();
    k_digest();
    k0_batch();
#if G6_MODEL_DS41
    driver_k0();
#endif
    if (fails) { std::fprintf(stderr, "batch_k0_test: %d FAILED\n", fails); return 1; }
    std::printf("batch_k0_test ok (DS41=%d)\n", (int)(
#if G6_MODEL_DS41
        1
#else
        0
#endif
    ));
    return 0;
}
