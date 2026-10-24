// CPU checks of engine/runtime/spec_len.h, no GPU:
//   1. choose(): every draft certain -> the largest level, every draft unlikely -> the smallest
//   2. a simulated drafter whose probability is calibrated (a draft with probability q is accepted with chance q):
//      DFlash2-like rounds (7 drafts, the drafting cost fixed) on reasoning-like text (q around 0.6), JSON-like text
//      (q near 1) and the two mixed in blocks. Tokens per ms of adaptive vs every fixed level: adaptive must reach the
//      best fixed level minus 2% on each text and beat every fixed level on the mix
//      and with an overconfident drafter that reports q^0.2 (the buckets must learn the calibration)
//   3. gapped levels: the MTP chain reaches the next allowed level, and choose() returns only levels
//   4. MTP-like rounds (chain steps cost 2.1 ms, more() decides after each): the same bounds for K = 4
// build and run:  g++ -O2 -std=c++17 engine/tests/test_spec_len.cpp -o /tmp/test_spec_len && /tmp/test_spec_len
#include <cstdio>
#include <random>
#include "../runtime/spec_len.h"

using namespace g6rt;

static int fails = 0, checks = 0;
static void check(bool ok, const char* what)
{
    ++checks;
    if (!ok) ++fails;
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

// one text: draft j's acceptance chance q_j drawn per round from Beta(a, b); the drafter reports q_j^g (g = 1: calibrated,
// g < 1: overconfident)
struct Text { double a, b, g = 1; };

// simulated tokens per ms. fixed > 0: always that k; 0: adaptive. mtp: drafts come one chain step at a time
static double run(const std::vector<int>& levels, int fixed, bool mtp, const std::vector<Text>& texts, int rounds_each, unsigned seed)
{
    SpecLen sl;
    sl.on = true;
    sl.set_levels(levels);
    sl.draft_round = mtp ? 2.5 : 5.6;
    sl.draft_each = mtp ? 2.1 : 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    double tok = 0, ms = 0;
    for (const Text& tx : texts)
    {
        std::gamma_distribution<double> ga(tx.a, 1), gb(tx.b, 1);
        for (int r = 0; r < rounds_each; ++r)
        {
            float lp[SpecLen::KX + 1];
            double q[SpecLen::KX + 1];
            const int kmax = levels.back();
            for (int j = 1; j <= kmax; ++j) { const double x = ga(rng), y = gb(rng); q[j] = x / (x + y); lp[j] = (float)(tx.g * std::log(q[j])); }
            int n = kmax;
            if (mtp && !fixed)
            {
                n = 1;
                while (sl.more(lp, n)) ++n;
            }
            if (mtp && fixed) n = fixed;
            double e = 0;
            const int k = fixed ? fixed : sl.choose(lp, n, n, &e);
            int a = 0;
            while (a < k && U(rng) < q[a + 1]) ++a;
            const double step = sl.table[k + 1];
            tok += a + 1;
            ms += step + sl.draft_ms(mtp ? n : k);
            sl.observe(k, a, lp, e);
        }
    }
    return tok / ms;
}

int main()
{
    {
        double values[8] = {};
        check(parse_spec_costs("43,52.5,59.7,68.9,77.5,86.5,95.2,104.9", values, 8)
              && values[0] == 43 && values[7] == 104.9, "parse the verify override");
        check(parse_spec_costs(" 8.1, 4.3, 2.0 ", values, 3, true)
              && values[0] == 8.1 && values[2] == 2.0, "parse drafting costs with whitespace");
        check(parse_spec_costs("0,0,0", values, 3, true), "zero drafting costs are allowed");
        check(!parse_spec_costs(nullptr, values, 3), "reject a missing cost list");
        for (const char* bad : {"", "1,2", "1,2,3,4", "1,,3", ",1,2", "1,2,3,", "1,2,3junk",
                                "1 2 3", "nan,2,3", "inf,2,3", "-1,2,3", "0,2,3", "1e999,2,3", "1e-999,2,3"})
            check(!parse_spec_costs(bad, values, 3), bad);
        check(!parse_spec_costs("-1,2,3", values, 3, true), "negative drafting cost is rejected");
        check(!parse_spec_costs("nan,2,3", values, 3, true), "NaN drafting cost is rejected");
    }
    {
        SpecLen sl;
        sl.set_levels({2, 3, 5, 7});
        sl.draft_round = 5.6;
        float hi[8], lo[8];
        for (int j = 0; j < 8; ++j) { hi[j] = std::log(0.999f); lo[j] = std::log(0.1f); }
        // the bucket table is empty, so seed the position rates the way a run of such drafts would
        for (int r = 0; r < 200; ++r) sl.observe(7, 7, hi, 8);
        check(sl.choose(hi, 7, 7) == 7, "certain drafts -> k = 7");
        SpecLen s2 = sl;
        for (int r = 0; r < 200; ++r) s2.observe(7, 0, lo, 1);
        check(s2.choose(lo, 7, 7) == 2, "unlikely drafts -> the smallest level");
    }
    for (const std::vector<int>& lv : {std::vector<int>{1, 2, 4}, std::vector<int>{3, 4}, std::vector<int>{2, 5, 7}})
    {
        // gapped levels: certain drafts take the MTP chain to the top level, and every choice is a level
        SpecLen sl;
        sl.set_levels(lv);
        sl.draft_round = 2.5;
        sl.draft_each = 2.1;
        float hi[8];
        for (float& x : hi) x = std::log(0.999f);
        for (int r = 0; r < 200; ++r) sl.observe(lv.back(), lv.back(), hi, lv.back() + 1);
        int n = 1;
        while (sl.more(hi, n)) ++n;
        const int k = sl.choose(hi, n, n);
        char b[96];
        snprintf(b, sizeof b, "levels {%d..%d}, %zu of them: the chain reaches %d, k = %d", lv.front(), lv.back(), lv.size(), n, k);
        check(n == lv.back() && k == lv.back(), b);
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> U(0.02, 1);
        bool in = true;
        for (int r = 0; r < 2000; ++r)
        {
            float lp[8];
            for (float& x : lp) x = (float)std::log(U(rng));
            int m = 1;
            while (sl.more(lp, m)) ++m;
            const int c = sl.choose(lp, m, m);
            in &= std::find(lv.begin(), lv.end(), c) != lv.end();
            int a = 0;
            while (a < c && U(rng) < std::exp(lp[a + 1])) ++a;
            sl.observe(c, a, lp, 0);
        }
        snprintf(b, sizeof b, "levels {%d..%d}: 2000 random rounds choose only levels", lv.front(), lv.back());
        check(in, b);
    }
    const Text reasoning{3, 2}, json{40, 0.6}, prose{1.5, 1.5}, over{3, 2, 0.2};
    struct Case { const char* name; std::vector<Text> texts; };
    const std::vector<Case> cases = {{"reasoning", {reasoning}}, {"json", {json}}, {"prose", {prose}},
                                     {"mixed", {reasoning, json, prose, json, reasoning, prose}},
                                     {"json after reasoning", {reasoning, reasoning, reasoning, json}},
                                     {"overconfident", {over}}};
    for (bool mtp : {false, true})
    {
        const std::vector<int> levels = mtp ? std::vector<int>{1, 2, 3, 4} : std::vector<int>{1, 2, 3, 4, 5, 6, 7};
        for (const Case& c : cases)
        {
            const int each = c.texts.size() > 1 ? 150 : 3000;   // short blocks: the estimates must follow the text
            const double ad = run(levels, 0, mtp, c.texts, each, 11);
            double best = 0;
            int bk = 0;
            printf("%s %-9s adaptive %.4f tok/ms; fixed", mtp ? "mtp  " : "dfl  ", c.name, ad);
            for (int k : levels)
            {
                const double f = run(levels, k, mtp, c.texts, each, 11);
                printf(" %d:%.4f", k, f);
                if (f > best) { best = f; bk = k; }
            }
            printf("\n");
            char b[128];
            snprintf(b, sizeof b, "%s %s: adaptive %.1f%% of the best fixed k (%d)", mtp ? "mtp" : "dfl", c.name, 100 * ad / best, bk);
            check(ad >= 0.98 * best, b);
            if (c.texts.size() > 1)
            {
                snprintf(b, sizeof b, "%s %s: adaptive beats every fixed k", mtp ? "mtp" : "dfl", c.name);
                check(ad > best, b);
            }
        }
    }
    printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
