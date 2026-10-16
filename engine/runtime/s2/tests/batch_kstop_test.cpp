// MTP confidence stop (s2_batch.h Batch::kstop, G6_MTP_KSTOP): a solo DFlash2 session's MTP rounds chain and verify the
// drafts the rule allows (mode 1: p_1 * .. * p_j >= tau[j]; mode 2: the drafter choice's MTP estimates), a demoted
// session's chain is completed to K before its first shared round, every committed token is the target's, and with the
// stop off the steps are the same with or without the log-probability hook and the log, and a session promoted with
// more drafts than kmax (G6_DFL_REUSE=2) verifies at most kmax. Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_kstop_test.cpp -o /tmp/batch_kstop_test
#include "s2_batch.h"
#include <cmath>
#include <cstdio>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
static bool good(int spos) { return (spos * 2654435761u >> 7) % 4 != 0; }   // the MTP draft for sample position spos is right

struct Fake
{
    int regime = 0;           // 0: drafts right where good(); 1: always right, confident; 2: always wrong, unsure
    std::vector<float> lp;    // the last draft or chain step's rows
    std::vector<std::tuple<int, int, int, long>> trace;   // (kind, M, first row token, round) per step
    std::vector<int> got;
    bool right(int sp) const { return regime == 1 || (regime == 0 && good(sp)); }
    float p(int sp) const { return regime == 1 ? 0.95f : regime == 2 ? 0.05f : !good(sp) ? 0.15f : sp % 4 ? 0.97f : 0.8f; }
    Hooks hooks(bool with_lp)
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out;
            if (s.kind != Step::MAIN) lp.clear();
            for (const Row& r : s.r)
            {
                if (s.kind == Step::MAIN) out.push_back(target(r.spos - 1));
                else { out.push_back(right(r.spos) ? target(r.spos - 1) : -2); lp.push_back(std::log(p(r.spos))); }
            }
            trace.emplace_back((int)s.kind, s.M(), s.r[0].tok, s.round);
            return out;
        };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [](const DflReq&) { DflDrafts o{}; for (int i = 1; i < DFL_B; ++i) { o.tok[i] = -1; o.lp[i] = -3.f; } return o; };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        if (with_lp) h.lp = [this](int r) { return lp.at(r); };
        return h;
    }
};
static DSel dsel()
{
    DSel sel;
    sel.on = true; sel.dwell = 0;           // DFlash2 never pays (its drafts are all wrong and its round costs 1e6 ms):
    sel.d.draft_round = 1e6; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;   // MTP from the second round on
    return sel;
}
static Session session(int n_gen)
{
    Session s; s.prompt.assign(10, 5); s.first = 9; s.chunk = 8; s.n_gen = n_gen; s.dflash = true;
    return s;
}
static bool exact(const std::vector<int>& got) { bool e = true; for (size_t i = 0; i < got.size(); ++i) e &= got[i] == target(9 + (int)i); return e; }

// mode 1 rule, from the fake's probabilities: drafts chained and verified for a round at base
static std::pair<int, int> rule(const Fake& f, const KStop& ks, int base)
{
    auto P = [&](int j) { double p = 1; for (int i = 1; i <= j; ++i) p *= std::exp((double)std::log(f.p(base + i))); return p; };
    int nd = 1;
    for (;;)
    {
        if (nd >= ks.kmax) break;
        bool ok = true;
        for (int j = 1; j <= nd; ++j) ok &= P(j) >= ks.tau[j];
        if (!ok || P(nd) < ks.tau[nd + 1]) break;
        ++nd;
    }
    int k = 0;
    while (k < nd && P(k + 1) >= ks.tau[k + 1]) ++k;
    return {nd, std::max(k, 1)};
}

int main()
{
    // 1. stop off: the same steps and tokens without the hook, with it, and with the log on
    {
        std::vector<std::tuple<int, int, int, long>> ref;
        for (int arm = 0; arm < 3; ++arm)
        {
            Fake f;
            Batch b(2, 32, f.hooks(arm > 0));
            b.dflash(7, g6rt::SpecLen());
            b.dsel(dsel());
            KStop ks; ks.log = arm == 2;
            b.kstop(ks);
            b.admit({0}, {session(4000)});
            for (int i = 0; i < (arm == 2 ? 6 : 60); ++i) CHECK(b.round({0}));
            if (arm == 0) ref = f.trace;
            else CHECK(std::equal(f.trace.begin(), f.trace.end(), ref.begin()));
            CHECK(exact(f.got));
        }
    }
    // 2. mode 1, kmax 3 (K 2): every MTP verify takes the rule's k, after the rule's chain steps
    {
        Fake f;
        Hooks h = f.hooks(true);
        auto run = h.run;
        int nchain = 0, bad = 0, rounds = 0, by[5] = {};
        KStop ks; ks.mode = 1; ks.kmax = 3; ks.tau[1] = 0; ks.tau[2] = 0.3; ks.tau[3] = 0.8;
        h.run = [&](const Step& s) {
            if (s.kind == Step::CHAIN) ++nchain;
            if (s.kind == Step::MAIN && s.round >= 0 && s.M() <= 4)
            {
                const int base = s.r[0].pos;
                const auto r = rule(f, ks, base);
                bad += s.M() - 1 != r.second || nchain != r.first - 1;
                ++rounds; ++by[s.M()];
            }
            if (s.kind == Step::MAIN) nchain = 0;
            return run(s);
        };
        Batch b(2, 32, h);
        b.dflash(7, g6rt::SpecLen());
        b.dsel(dsel());
        b.kstop(ks);
        b.admit({0}, {session(4000)});
        for (int i = 0; i < 300; ++i) CHECK(b.round({0}));
        CHECK(rounds > 250 && bad == 0);
        CHECK(by[2] > 0 && by[3] > 0 && by[4] > 0);   // every draft count is used
        CHECK(exact(f.got));
        CHECK(b.stats(0).acc[3] > 0);                  // a third draft accepted
    }
    // 3. demote after a one-draft round: the chain is completed to K, and the shared round verifies K + 1 rows each
    {
        Fake f;
        f.regime = 1;
        Batch b(2, 32, f.hooks(true));
        b.dflash(7, g6rt::SpecLen());
        b.dsel(dsel());
        KStop ks; ks.mode = 1; ks.kmax = 3; ks.tau[2] = 0.95;   // p 0.95: one draft (0.9025 < 0.95)
        b.kstop(ks);
        b.admit({0}, {session(4000)});
        for (int i = 0; i < 6; ++i) CHECK(b.round({0}));   // one DFlash2 round, then five MTP rounds
        int m2 = 0;
        for (const auto& t : f.trace) m2 += std::get<0>(t) == Step::MAIN && std::get<1>(t) == 2;
        CHECK(m2 == 5);
        const std::vector<int> mine = f.got;
        Session s2 = session(4000); s2.dflash = false;
        b.admit({1}, {s2});
        f.trace.clear();
        CHECK(b.round({0, 1}));                        // demote, then one shared round
        int i = 0;
        while (i < (int)f.trace.size() && std::get<0>(f.trace[i]) != Step::MAIN) ++i;
        CHECK(i >= 1 && std::get<0>(f.trace[i - 1]) == Step::CHAIN);   // the top-up chain step before the shared verify
        CHECK(i < (int)f.trace.size() && std::get<1>(f.trace[i]) == 2 * 3);
        for (int r = 0; r < 20; ++r) CHECK(b.round({0, 1}));
        CHECK(b.stats(0).acc[2] > 0);
    }
    // 4. mode 2: confident right drafts go to three, unsure wrong ones to one
    for (int regime : {1, 2})
    {
        Fake f;
        f.regime = regime;
        Batch b(2, 32, f.hooks(true));
        b.dflash(7, g6rt::SpecLen());
        b.dsel(dsel());
        KStop ks; ks.mode = 2; ks.kmax = 3;
        b.kstop(ks);
        b.admit({0}, {session(4000)});
        for (int i = 0; i < 100; ++i) CHECK(b.round({0}));
        int by[9] = {};
        for (const auto& t : f.trace) if (std::get<0>(t) == Step::MAIN && std::get<3>(t) >= 50) ++by[std::get<1>(t)];
        if (regime == 1) CHECK(by[4] > 40);
        else CHECK(by[2] > 40);
        CHECK(exact(f.got));
    }
    // 5. G6_DFL_REUSE=2 promotes a session that ran shared rounds, so it holds the shared chain's K drafts. With K above
    //    G6_MTP_KSTOP_KMAX no solo MTP round may verify more than kmax drafts (K 2 / kmax 1, and K 4 / kmax 3, where a
    //    verify past kmax would also read tau past its end), in either mode
    for (int mode : {1, 2})
        for (const auto& kk : {std::make_pair(2, 1), std::make_pair(4, 3)})
        {
            Fake f;
            f.regime = 1;                              // every MTP draft right and confident: the stop would take them all
            Batch b(kk.first, 32, f.hooks(true));
            b.dflash(7, g6rt::SpecLen());
            b.dsel(dsel());
            KStop ks; ks.mode = mode; ks.kmax = kk.second; ks.tau[1] = ks.tau[2] = ks.tau[3] = 0;
            b.kstop(ks);
            DflReuse r; r.mode = 2; r.min = 64;
            b.dfl_reuse(r);
            Session s = session(4000); s.dflash = false;
            b.admit({0}, {s});
            for (int i = 0; i < 4; ++i) CHECK(b.round({0}));
            b.promote(0);
            CHECK(b.stats(0).promoted >= 0);
            f.trace.clear();
            for (int i = 0; i < 40; ++i) CHECK(b.round({0}));
            int mtp = 0, over = 0;
            for (const auto& t : f.trace)
                if (std::get<0>(t) == Step::MAIN && std::get<1>(t) < 8) { ++mtp; over += std::get<1>(t) - 1 > kk.second; }
            CHECK(mtp > 10 && over == 0);
            CHECK(exact(f.got));
        }
    std::printf("batch_kstop_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
