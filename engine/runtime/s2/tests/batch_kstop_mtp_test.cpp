// MTP confidence stop in an MTP-only engine (s2_batch.h KStop::solo: G6_MTP_KSTOP=1 without --dflash-manifest, TP=6 /
// TP=3). No drafter hooks; every session drafts with MTP and runs round_all. A session alone verifies kverify of the
// drafts its chain held (main step k + 1 rows, class row_class(k + 1)), its chain stops where kmore says; a round of
// several sessions first chains every short session to K, so its main step is S * (K + 1) rows of class row_class(K + 1)
// as without the stop; every committed token is the target's. Off (solo false, or mode 0), the steps are the same as
// before. KStopB (G6_MTP_KSTOPB) without the drafter: the rule's widths and exact tokens.
// Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_kstop_mtp_test.cpp -o /tmp/batch_kstop_mtp_test
#include "s2_batch.h"
#include <cmath>
#include <cstdio>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
static bool good(int spos) { return (spos * 2654435761u >> 7) % 4 != 0; }   // the MTP draft for sample position spos is right
static float prob(int spos) { return !good(spos) ? 0.15f : spos % 4 ? 0.97f : spos % 8 ? 0.7f : 0.92f; }
static double P(int base, int j)  // p_1 * .. * p_j of the drafts after base, as Batch::kprob computes it
{
    double p = 1;
    for (int i = 1; i <= j; ++i) p *= std::exp((double)std::log(prob(base + i)));
    return p;
}

struct Fake
{
    std::vector<float> lp;
    std::vector<std::tuple<int, int, int, long>> trace;   // (kind, M, first row token, round) per step
    std::vector<Step> mains;          // decode-round main steps
    std::vector<int> chain_after;     // per decode-round main step: the chain steps of the previous round (solo runs)
    int chains = 0;
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out;
            if (s.kind != Step::MAIN) lp.clear();
            for (const Row& r : s.r)
            {
                if (s.kind == Step::MAIN) out.push_back(target(r.spos - 1));
                else { out.push_back(good(r.spos) ? target(r.spos - 1) : -2); lp.push_back(std::log(prob(r.spos))); }
            }
            trace.emplace_back((int)s.kind, s.M(), s.r[0].tok, s.round);
            if (s.kind == Step::CHAIN) ++chains;
            if (s.kind == Step::MAIN && s.round >= 0) { mains.push_back(s); chain_after.push_back(chains); chains = 0; }
            return out;
        };
        h.emit = [](int, const int*, int, int) { return false; };
        h.lp = [this](int r) { return lp.at(r); };   // no h.dfl / h.dfl_start: an MTP-only engine
        return h;
    }
};
static Session session(int q, int n_gen)
{
    Session s; s.prompt.assign(10 + 3 * q, 5); s.first = 9 + 3 * q; s.chunk = 8; s.n_gen = n_gen;
    return s;
}
static bool exact(const Batch& b, int q, const Session& s)
{
    const std::vector<int> r = b.result(q);
    const int n = (int)s.prompt.size();
    bool ok = (int)r.size() == s.n_gen;
    for (size_t i = 0; i < r.size(); ++i) ok &= r[i] == target(n - 1 + (int)i);
    return ok;
}
// KStop mode 1 as Batch::kchain / kverify compute it for the drafts after base: drafts held (n), drafts verified (k)
static int held(const KStop& ks, int K, int base)
{
    const int kmax = ks.kmax ? ks.kmax : K;
    auto more = [&](int n) {
        for (int j = 1; j <= n; ++j) if (P(base, j) < ks.tau[j]) return false;
        return P(base, n) >= ks.tau[n + 1];
    };
    int n = 1;
    while (n < kmax && more(n)) ++n;
    return n;
}
static int verified(const KStop& ks, int K, int base)
{
    const int n = std::min(held(ks, K, base), ks.kmax ? ks.kmax : K);
    int k = 0;
    while (k < n && P(base, k + 1) >= ks.tau[k + 1]) ++k;
    return std::max(k, 1);
}
static KStop solo_on(int kmax)
{
    KStop ks; ks.mode = 1; ks.kmax = kmax; ks.solo = true;
    ks.tau[1] = 0; ks.tau[2] = 0.5; ks.tau[3] = 0.75;   // the recipe's G6_MTP_KSTOP_TAU=0,0.5,0.75
    return ks;
}

int main()
{
    const int K = 3;
    // 1. off: solo false (the DFlash2 engine's KStop) and mode 0 plan the same steps as no stop at all
    {
        std::vector<std::tuple<int, int, int, long>> ref;
        for (int arm = 0; arm < 3; ++arm)
        {
            Fake f;
            Batch b(K, 32, f.hooks());
            KStop ks;
            if (arm == 1) { ks = solo_on(3); ks.solo = false; }   // as the DFlash2 engine sets it: round_all untouched
            if (arm == 2) { ks = solo_on(3); ks.mode = 0; }
            b.kstop(ks);
            const std::vector<Session> ss = {session(0, 120), session(1, 60)};
            b.admit({0, 1}, ss);
            while (b.round({0, 1})) {}
            if (arm == 0) ref = f.trace;
            else CHECK(f.trace == ref);
            for (int q = 0; q < 2; ++q) CHECK(exact(b, q, ss[q]));
            for (const Step& v : f.mains) CHECK(v.M() == (int)v.sess.size() * (K + 1) && v.cls == row_class(K + 1));
        }
    }
    // 2. one session alone, kmax 3 and 2: every main step verifies the rule's k of the drafts its chain held
    for (int kmax : {3, 2})
    {
        Fake f;
        Batch b(K, 32, f.hooks());
        const KStop ks = solo_on(kmax);
        b.kstop(ks);
        const Session s = session(0, 600);
        b.admit({0}, {s});
        while (b.round({0})) {}
        CHECK(exact(b, 0, s));
        int bad = 0, short_k = 0, full_k = 0, chain_bad = 0;
        for (size_t i = 0; i < f.mains.size(); ++i)
        {
            const Step& v = f.mains[i];
            const int base = v.r[0].pos, k = verified(ks, K, base);
            bad += v.sess.size() != 1 || v.M() != k + 1 || v.cls != row_class(k + 1);
            chain_bad += f.chain_after[i] != held(ks, K, base) - 1;   // the chain that drafted this round's drafts
            short_k += k < kmax;
            full_k += k == kmax;
        }
        CHECK(bad == 0 && chain_bad == 0 && f.mains.size() > 100);
        CHECK(short_k > 0 && full_k > 0);
        const Stats& st = b.stats(0);
        CHECK(st.drafted < st.rounds * kmax && st.tokens > st.rounds);
    }
    // 3. two sessions, then one: shared rounds verify K each (short chains completed first), the survivor stops alone
    {
        Fake f;
        Batch b(K, 32, f.hooks());
        const KStop ks = solo_on(3);
        b.kstop(ks);
        const std::vector<Session> ss = {session(0, 400), session(1, 80)};
        b.admit({0}, {ss[0]});
        for (int i = 0; i < 30; ++i) CHECK(b.round({0, 1}));   // alone: solo rounds, some chains short
        b.admit({1}, {ss[1]});
        while (b.round({0, 1})) {}
        for (int q = 0; q < 2; ++q) CHECK(exact(b, q, ss[q]));
        int shared = 0, solo = 0, bad = 0;
        for (const Step& v : f.mains)
        {
            if (v.sess.size() == 2) { ++shared; bad += v.M() != 2 * (K + 1) || v.cls != row_class(K + 1); }
            else { ++solo; const int k = verified(ks, K, v.r[0].pos); bad += v.M() != k + 1 || v.cls != row_class(k + 1); }
        }
        CHECK(bad == 0 && shared > 10 && solo > 100);
    }
    // 4. KStopB without the drafter (MTP-only engine, G6_MTP_KSTOPB=1): each session verifies the rule's k_q, exact tokens
    {
        Fake f;
        Batch b(K, 32, f.hooks());
        KStop ks;
        ks.b.mode = 1;
        for (int S = 0; S <= S2N_SLOTS; ++S) { ks.b.tau[S][1] = 0.6; ks.b.tau[S][2] = 0.88; ks.b.tau[S][3] = 0.95; }
        b.kstop(ks);
        const std::vector<Session> ss = {session(0, 300), session(1, 150)};
        b.admit({0, 1}, ss);
        while (b.round({0, 1})) {}
        for (int q = 0; q < 2; ++q) CHECK(exact(b, q, ss[q]));
        int narrow = 0;
        for (const Step& v : f.mains) narrow += v.M() < (int)v.sess.size() * (K + 1);
        CHECK(narrow > 0);
    }
    std::printf("batch_kstop_mtp_test: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
