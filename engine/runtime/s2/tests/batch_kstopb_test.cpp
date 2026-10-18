// MTP confidence stop in shared rounds (s2_batch.h KStopB, G6_MTP_KSTOPB): every shared verify takes each session's
// k_q from the rule at the round's session count (k = 0 allowed unless kmin 1), a width outside the captured ones is
// filled with the most likely unverified drafts, explore rounds stay within 0..k, every committed token is the target's,
// and the stop off (or on with zero thresholds) plans the same steps with or without the log. Around it: a DFlash2
// session demoted into shared rounds and promoted back (G6_DFL_REUSE=2, with G6_MTP_KSTOP), and the insert rule.
// Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_kstopb_test.cpp -o /tmp/batch_kstopb_test
#include "s2_batch.h"
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
static bool good(int spos) { return (spos * 2654435761u >> 7) % 4 != 0; }   // the MTP draft for sample position spos is right
// the MTP head's probability of its draft for spos: wrong drafts unsure, right ones mostly sure
static float prob(int spos) { return !good(spos) ? 0.15f : spos % 4 ? 0.97f : spos % 8 ? 0.7f : 0.92f; }
static double P(int base, int j)  // p_1 * .. * p_j of the drafts after base, as Batch::kprob computes it
{
    double p = 1;
    for (int i = 1; i <= j; ++i) p *= std::exp((double)std::log(prob(base + i)));
    return p;
}

struct Fake
{
    std::vector<float> lp;            // the last draft or chain step's rows
    std::vector<std::tuple<int, int, int, long>> trace;   // (kind, M, first row token, round) per step
    std::vector<Step> mains;          // decode-round main steps (round >= 0)
    Hooks hooks(bool with_lp)
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
            if (s.kind == Step::MAIN && s.round >= 0) mains.push_back(s);
            return out;
        };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [](const DflReq& r) {        // the drafter's drafts are the target's (DFlash2 rounds accept all 7)
            DflDrafts o{};
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = target(r.pos0 + r.rows + i - 1); o.lp[i] = -0.1f; }
            return o;
        };
        h.emit = [](int, const int*, int, int) { return false; };
        if (with_lp) h.lp = [this](int r) { return lp.at(r); };
        return h;
    }
};
static Session session(int q, int n_gen, bool dflash = false)
{
    Session s; s.prompt.assign(10 + 3 * q, 5); s.first = 9 + 3 * q; s.chunk = 8; s.n_gen = n_gen; s.dflash = dflash;
    return s;
}
// every token session q committed is the target's (inserted ones: the rule's)
static bool exact(const Batch& b, int q, const Session& s)
{
    const std::vector<int> r = b.result(q);
    const int n = (int)s.prompt.size();
    bool ok = !r.empty();
    for (size_t i = 0; i < r.size(); ++i)
    {
        const int at = (int)i - s.ins.after;
        if (s.ins.on() && at >= 0 && at < (int)s.ins.tokens.size()) ok &= r[i] == s.ins.tokens[at];
        else ok &= r[i] == target(n - 1 + (int)i);
    }
    return ok;
}
static KStopB rule_on(double t1, double t2)
{
    KStopB kb; kb.mode = 1;
    for (int S = 0; S <= S2N_SLOTS; ++S) { kb.tau[S][1] = t1; kb.tau[S][2] = t2; kb.tau[S][3] = 2; }
    return kb;
}
// the rule's k_q of a session at base in a round of S sessions holding k drafts
static int rule(const KStopB& kb, int S, int base, int k)
{
    int j = 0;
    while (j < k && P(base, j + 1) >= kb.tau[S][j + 1]) ++j;
    return std::max(j, std::min(kb.kmin, k));
}
// the rule with the width fill: kq per session of main step v (bases from its rows)
static std::vector<int> expected(const KStopB& kb, const Step& v, int K, int k)
{
    const int S = (int)v.sess.size();
    std::vector<int> kq(S), base(S);
    int M = 0;
    for (int i = 0; i < S; ++i) { base[i] = v.r[v.row0[i]].pos; kq[i] = rule(kb, S, base[i], k); M += kq[i] + 1; }
    while (M != S * (k + 1) && !kb.width(S, K, M))
    {
        int bi = -1;
        double bp = -1;
        for (int i = 0; i < S; ++i) if (kq[i] < k && P(base[i], kq[i] + 1) > bp) { bp = P(base[i], kq[i] + 1); bi = i; }
        ++kq[bi]; ++M;
    }
    return kq;
}
static void run_all(Batch& b, const std::vector<int>& qs) { while (b.round(qs)) {} }

int main()
{
    const int N = 4, K = 2;
    // 0. the captured widths
    {
        KStopB kb = rule_on(0.6, 0.88);
        std::vector<int> all;
        for (int M = 1; M <= 12; ++M) all.push_back(M);
        CHECK(kstopb_widths(kb, N, K) == all);
        kb.kmin = 1;
        std::vector<int> k1(all.begin() + 1, all.end());
        CHECK(kstopb_widths(kb, N, K) == k1);
        kb.kmin = 0; kb.widths = {8};
        CHECK(kstopb_widths(kb, N, K) == std::vector<int>({3, 6, 8, 9, 12}));
        CHECK(kstopb_widths(KStopB(), N, K).empty() && kstopb_sig(KStopB()).empty());
        CHECK(kstopb_sig(kb) != kstopb_sig(rule_on(0.6, 0.88)));
    }
    // 1. off: the same steps without the hook, with it, with the log; the stop on with zero thresholds too
    {
        std::vector<std::tuple<int, int, int, long>> ref;
        for (int arm = 0; arm < 4; ++arm)
        {
            Fake f;
            Batch b(K, 32, f.hooks(arm > 0));
            KStop ks; ks.log = arm == 2;
            if (arm == 3) ks.b = rule_on(0, 0);
            b.kstop(ks);
            std::vector<Session> ss;
            for (int q = 0; q < N; ++q) ss.push_back(session(q, 40 + 25 * q));
            b.admit({0, 1, 2, 3}, ss);
            if (arm == 2) for (int i = 0; i < 4; ++i) CHECK(b.round({0, 1, 2, 3}));
            else run_all(b, {0, 1, 2, 3});
            if (arm == 0) ref = f.trace;
            else CHECK(std::equal(f.trace.begin(), f.trace.end(), ref.begin()));
            for (int q = 0; arm != 2 && q < N; ++q) CHECK(exact(b, q, ss[q]));
        }
    }
    // 2. the rule at every session count (thresholds by S): each session's rows are its k_q + 1, widths S..3S
    for (int kmin : {0, 1})
    {
        Fake f;
        Batch b(K, 32, f.hooks(true));
        KStop ks; ks.b = rule_on(0.6, 0.88); ks.b.kmin = kmin;
        ks.b.tau[3][1] = 0.5; ks.b.tau[3][2] = 0.8;
        ks.b.tau[2][1] = 0.3; ks.b.tau[2][2] = 0.66;
        ks.b.tau[1][1] = 0; ks.b.tau[1][2] = 0.5;
        b.kstop(ks);
        std::vector<Session> ss;
        for (int q = 0; q < N; ++q) ss.push_back(session(q, 300 + 200 * q));   // they leave one by one: S 4, 3, 2, 1
        b.admit({0, 1, 2, 3}, ss);
        run_all(b, {0, 1, 2, 3});
        int bad = 0, zero = 0;
        std::set<std::pair<int, int>> widths;   // (S, M)
        for (const Step& v : f.mains)
        {
            const int S = (int)v.sess.size();
            for (int i = 0; i < S; ++i)
            {
                const int kq = v.rows[i] - 1;
                bad += kq != rule(ks.b, S, v.r[v.row0[i]].pos, K);
                zero += kq == 0;
            }
            bad += v.M() < S * (kmin + 1) || v.M() > S * (K + 1) || v.cls != row_class(K + 1);
            widths.insert({S, v.M()});
        }
        CHECK(bad == 0 && f.mains.size() > 300);
        CHECK(kmin ? zero == 0 : zero > 0);
        for (int S = 1; S <= N; ++S)          // several widths at every session count
        {
            int n = 0;
            for (const auto& w : widths) n += w.first == S;
            CHECK(n >= (S == 1 ? 2 : 3));
        }
        for (int q = 0; q < N; ++q) CHECK(exact(b, q, ss[q]));
        long tok = 0, rounds = 0, drafted = 0;
        for (int q = 0; q < N; ++q) { tok += b.stats(q).tokens; rounds += b.stats(q).rounds; drafted += b.stats(q).drafted; }
        CHECK(drafted < rounds * K && tok > rounds);
    }
    // 3. captured widths {8} at K 2 (and the multiples of 3): S 4 launches 6, 8, 9 or 12 rows; the fill takes the most likely drafts
    {
        Fake f;
        Batch b(K, 32, f.hooks(true));
        KStop ks; ks.b = rule_on(0.6, 0.88); ks.b.widths = {8};
        b.kstop(ks);
        std::vector<Session> ss;
        for (int q = 0; q < N; ++q) ss.push_back(session(q, 200 + 100 * q));
        b.admit({0, 1, 2, 3}, ss);
        run_all(b, {0, 1, 2, 3});
        const std::vector<int> wv = kstopb_widths(ks.b, N, K);
        int bad = 0, filled = 0, m8 = 0;
        for (const Step& v : f.mains)
        {
            const int S = (int)v.sess.size();
            const std::vector<int> kq = expected(ks.b, v, K, K);
            for (int i = 0; i < S; ++i) bad += v.rows[i] - 1 != kq[i];
            bad += std::find(wv.begin(), wv.end(), v.M()) == wv.end() || !ks.b.width(S, K, v.M());
            int M0 = 0;
            for (int i = 0; i < S; ++i) M0 += rule(ks.b, S, v.r[v.row0[i]].pos, K) + 1;
            filled += M0 != v.M();
            m8 += v.M() == 8;
        }
        CHECK(bad == 0 && filled > 0 && m8 > 0);
        for (int q = 0; q < N; ++q) CHECK(exact(b, q, ss[q]));
    }
    // 4. explore every 3rd round: those verify hashed counts in kmin..k, the rest the rule; tokens exact
    {
        Fake f;
        Batch b(K, 32, f.hooks(true));
        KStop ks; ks.b = rule_on(0.6, 0.88); ks.b.explore = 3; ks.log = true;
        b.kstop(ks);
        std::vector<Session> ss;
        for (int q = 0; q < N; ++q) ss.push_back(session(q, 150));
        b.admit({0, 1, 2, 3}, ss);
        run_all(b, {0, 1, 2, 3});
        int bad = 0, xr = 0;
        std::set<int> xw;
        for (const Step& v : f.mains)
        {
            const int S = (int)v.sess.size();
            if (v.round % 3 == 0) { ++xr; xw.insert(v.M()); bad += v.M() < S || v.M() > S * (K + 1); continue; }
            for (int i = 0; i < S; ++i) bad += v.rows[i] - 1 != rule(ks.b, S, v.r[v.row0[i]].pos, K);
        }
        CHECK(bad == 0 && xr > 10 && xw.size() >= 4);
        for (int q = 0; q < N; ++q) CHECK(exact(b, q, ss[q]));
    }
    // 5. a DFlash2 session (with G6_MTP_KSTOP, kmax 3, or without) demoted into shared rounds with the stop, then
    // promoted back when alone (G6_DFL_REUSE=2): shared rounds follow the rule, every token is the target's
    for (int solo_kstop : {0, 1})
    {
        Fake f;
        Batch b(K, 32, f.hooks(true));
        b.dflash(7, g6rt::SpecLen());
        DSel sel;
        sel.on = true;
        sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
        b.dsel(sel);
        KStop ks; ks.b = rule_on(0.6, 0.88);
        if (solo_kstop) { ks.mode = 1; ks.kmax = 3; ks.tau[2] = 0.95; ks.tau[3] = 0.99; }   // solo MTP rounds: one draft held
        b.kstop(ks);
        DflReuse r; r.mode = 2; r.min = 16;
        b.dfl_reuse(r);
        const Session s0 = session(0, 600, true), s1 = session(1, 80);
        b.admit({0}, {s0});
        for (int i = 0; i < 6; ++i) CHECK(b.round({0}));
        const size_t m0 = f.mains.size();
        b.admit({1}, {s1});
        while (b.active(1)) CHECK(b.round({0, 1}));
        int bad = 0, shared = 0;
        for (size_t i = m0; i < f.mains.size(); ++i)
        {
            const Step& v = f.mains[i];
            if (v.sess.size() < 2) continue;
            ++shared;
            for (size_t j = 0; j < v.sess.size(); ++j) bad += v.rows[j] - 1 != rule(ks.b, 2, v.r[v.row0[j]].pos, K);
        }
        CHECK(bad == 0 && shared > 10);
        CHECK(exact(b, 1, s1));
        const int at = b.length(0) - 1;
        b.promote(0);
        CHECK(b.stats(0).promoted == at);
        run_all(b, {0});
        CHECK(exact(b, 0, s0) && b.stats(0).dfl_rounds > 0);
    }
    // 6. the insert rule in shared rounds with the stop: the inserted tokens' steps, then the rule again
    {
        Fake f;
        Batch b(K, 32, f.hooks(true));
        KStop ks; ks.b = rule_on(0.6, 0.88);
        b.kstop(ks);
        std::vector<Session> ss;
        for (int q = 0; q < 3; ++q)
        {
            ss.push_back(session(q, 120));
            ss.back().ins.after = 20 + 7 * q;
            ss.back().ins.tokens = {900 + q, 901, 902};
        }
        b.admit({0, 1, 2}, ss);
        run_all(b, {0, 1, 2});
        for (int q = 0; q < 3; ++q) CHECK(exact(b, q, ss[q]) && b.stats(q).inserted == 3);
    }
    std::printf("batch_kstopb_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
