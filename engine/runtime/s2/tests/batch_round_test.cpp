// s2_batch.h Batch: one round = ONE main step (every active session's K+1 rows), ONE first-form draft step, K-1 chain
// steps; the k1 class follows the step's row count; a session's rows and tokens do not depend on the other sessions.
// Host only. Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_round_test.cpp -o /tmp/batch_round_test
// The fake model is a pure function of (session salt, row position, row token): row invariance holds by construction,
// so any difference between a solo run and a batched run comes from the step planner.
#include "s2_batch.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>

using namespace s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
constexpr int V = 2;   // tiny vocabulary: drafts match the main sample often enough to give every accept count 0..K

struct Trace
{
    struct Rec { int kind, tok, pos, spos; };
    std::vector<std::vector<Rec>> rows;                    // per session (slot): its rows in step order
    std::vector<std::vector<int>> sample;                  // per session: the sampled token of each of its rows, in order
    struct StepInfo { int kind, M, cls, S; long round; };
    std::vector<StepInfo> steps;
};

// salt[q] = the identity of the session in slot q (so the same session gives the same rows whatever slot it is in)
static Hooks make_hooks(const std::vector<uint32_t>& salt, Trace& tr, int stop_slot_salt = -1)
{
    Hooks h;
    tr.rows.assign(salt.size(), {}); tr.sample.assign(salt.size(), {});
    h.run = [&salt, &tr](const Step& s) {
        tr.steps.push_back({(int)s.kind, s.M(), s.cls, (int)s.sess.size(), s.round});
        std::vector<int> out;
        for (const Row& r : s.r)
        {
            const uint32_t k = (uint32_t)s.kind;
            // main sample depends on (salt, pos, tok); a draft row's guess on (salt, pos + 1, tok): the same function, one
            // position on, so a guess matches the main sample unless it is flipped
            int t = (int)(mix(salt[r.sess], k == Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
            if (k != Step::MAIN && mix(salt[r.sess], r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;   // a wrong guess one time in three
            out.push_back(t);
            tr.rows[r.sess].push_back({(int)k, r.tok, r.pos, r.spos});
            tr.sample[r.sess].push_back(t);
        }
        return out;
    };
    h.emit = [stop_slot_salt, &salt](int q, const int* toks, int k, int) {
        if (stop_slot_salt < 0 || salt[q] != (uint32_t)stop_slot_salt) return false;
        for (int i = 0; i < k; ++i) if (toks[i] == 1) return true;
        return false;
    };
    return h;
}

static Session session(int salt, int plen, int n_gen, int chunk)
{
    Session s;
    for (int i = 0; i < plen; ++i) s.prompt.push_back((int)(mix(salt, i, 77) % V));
    s.n_gen = n_gen; s.chunk = chunk;
    return s;
}

int main()
{
    const int K = 2;
    // sessions differ in prompt length, chunk and n_gen, so they finish in different rounds
    const std::vector<Session> S = {session(11, 5, 14, 4), session(12, 9, 6, 4), session(13, 3, 20, 2), session(14, 6, 9, 4), session(15, 4, 11, 4)};
    const int NS = (int)S.size();
    std::vector<uint32_t> salt;
    for (int q = 0; q < NS; ++q) salt.push_back(11 + q);

    // batched run: all sessions through one Batch
    Trace tb;
    std::vector<Stats> stb;
    const auto outb = mtp_batch(S, K, NS * (K + 1), make_hooks(salt, tb, 14), stb);

    // 1. row invariance: every session alone gives the same tokens, stats and the same rows in the same order
    for (int q = 0; q < NS; ++q)
    {
        Trace ts;
        std::vector<Stats> sts;
        const std::vector<uint32_t> one = {salt[q]};
        const auto outs = mtp_batch({S[q]}, K, 8, make_hooks(one, ts, 14), sts);
        CHECK(outs[0] == outb[q]);
        CHECK(sts[0].rounds == stb[q].rounds && sts[0].tokens == stb[q].tokens && sts[0].acc == stb[q].acc);
        bool same = ts.rows[0].size() == tb.rows[q].size();
        for (size_t i = 0; same && i < ts.rows[0].size(); ++i)
        {
            const auto &a = ts.rows[0][i], &b = tb.rows[q][i];
            same = a.kind == b.kind && a.tok == b.tok && a.pos == b.pos && a.spos == b.spos;
        }
        CHECK(same);
        CHECK(ts.sample[0] == tb.sample[q]);
        CHECK((int)outb[q].size() <= S[q].n_gen && !outb[q].empty());
    }
    long long accs = 0;
    for (int a = 0; a <= K; ++a) { long long n = 0; for (int q = 0; q < NS; ++q) n += stb[q].acc[a]; CHECK(n > 0); accs += n; }   // every accept count is exercised
    CHECK(accs > 0);

    // 2. step structure of every decode round: one MAIN of S*(K+1) rows (k1 class of one session, K+1), at most one DRAFT, then K-1 CHAIN steps
    std::map<long, std::vector<Trace::StepInfo>> rounds;
    for (const auto& s : tb.steps) if (s.round >= 0) rounds[s.round].push_back(s);
    CHECK(!rounds.empty());
    bool shrank = false;
    int prev_active = NS + 1;
    for (const auto& kv : rounds)
    {
        const auto& v = kv.second;
        const int S_ = v[0].S;                       // active sessions of the round = sessions of its main step
        CHECK(v[0].kind == Step::MAIN && v[0].M == S_ * (K + 1) && v[0].cls == row_class(K + 1));
        size_t i = 1, drafts = 0, chains = 0;
        for (; i < v.size() && v[i].kind == Step::DRAFT; ++i) { ++drafts; CHECK(v[i].cls == row_class(v[i].M) && v[i].S <= S_); }
        for (; i < v.size() && v[i].kind == Step::CHAIN; ++i) { ++chains; CHECK(v[i].M == v[i].S && v[i].cls == row_class(v[i].M) && v[i].S <= S_); }
        CHECK(i == v.size());
        CHECK(drafts <= 1);
        CHECK(drafts == 1 ? chains == (size_t)(K - 1) || chains == 0 : chains == 0);
        if (S_ < prev_active && prev_active <= NS) shrank = true;
        prev_active = S_;
    }
    CHECK(shrank);                                   // the sessions finish at different rounds
    CHECK(rounds.begin()->second[0].S == NS);

    // 3. the graph key is the step shape: kind and M, whatever the sessions
    {
        Step a, b;
        a.kind = b.kind = Step::MAIN;
        a.r.assign(6, Row{0, 0, 0, 0, -1}); b.r.assign(6, Row{5, 1, 9, 9, -1});
        a.sess = {0, 1}; b.sess = {3, 6};
        CHECK(a.key() == b.key());
        b.r.push_back(Row{});
        CHECK(a.key() != b.key());
        b.r.pop_back(); b.kind = Step::DRAFT;
        CHECK(a.key() != b.key());
    }

    // 4. max rows: a round of S sessions needs S*(K+1) rows, otherwise it throws before running anything
    {
        Trace t;
        const std::vector<uint32_t> sl = {salt[0], salt[1], salt[2]};
        Batch b(K, 3 * (K + 1) - 1, make_hooks(sl, t));
        b.admit({0, 1, 2}, {S[0], S[1], S[2]});
        const size_t before = t.steps.size();
        bool threw = false;
        try { b.round({0, 1, 2}); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw && t.steps.size() == before);
        CHECK(b.round({0, 1}));                      // two sessions fit
    }
    // prompt steps: one session per step, each followed by its first-form draft (unchanged)
    {
        int prompt_main = 0;
        for (const auto& s : tb.steps) if (s.round < 0 && s.kind == Step::MAIN) { ++prompt_main; CHECK(s.S == 1); }
        int expect = 0;
        for (const auto& s : S) expect += ((int)s.prompt.size() + s.chunk - 1) / s.chunk;
        CHECK(prompt_main == expect);
    }

    std::printf("batch_round_test: %s (%zu steps, %zu rounds)\n", fails ? "FAIL" : "PASS", tb.steps.size(), rounds.size());
    return fails ? 1 : 0;
}
