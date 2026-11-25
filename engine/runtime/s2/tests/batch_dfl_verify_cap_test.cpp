// Adaptive dynamic K* for solo DFlash2 rounds: GR8_DFL_VERIFY_CAP (survival truncation 0..K) and GR8_SPEC_K0
// (SpecLen::allow_k0 selecting K*=0 plain decode on low-acceptance stretches). Host only, fake engine.
#include "s2_batch.h"
#include <cmath>
#include <cstdio>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }

struct SimRes {
    std::vector<int> emitted;
    std::vector<int> verify_m;
};

static SimRes run_cap(double cap, const float (&lp_vals)[8], int n_right, bool spec_k0 = false)
{
    SimRes res;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r) out.push_back(target(r.pos));
        if (s.kind == Step::MAIN && s.r[0].pos >= 300) res.verify_m.push_back(s.M());
        return out;
    };
    h.dfl_start = [&](int) { return 100; };
    h.dfl = [&](const DflReq& r) {
        DflDrafts o{};
        const int q0 = r.pos0 + r.rows;
        for (int i = 1; i < DFL_B; ++i) {
            o.tok[i] = i <= n_right ? target(q0 + i - 1) : -1;
            o.lp[i] = lp_vals[i];
        }
        return o;
    };
    h.emit = [&](int, const int* t, int k, int) { res.emitted.insert(res.emitted.end(), t, t + k); return false; };

    Batch b(2, 24, h);
    g6rt::SpecLen sl;
    if (spec_k0) {
        sl.on = true;
        sl.set_levels({1, 2, 3, 4, 5, 6, 7});
        sl.allow_k0();
        const double t2[9] = {0.0, 62.0, 72.0, 81.0, 89.5, 97.5, 105.0, 112.0, 118.5};
        std::copy(t2, t2 + 9, sl.table);
        sl.draft_round = 5.6;
        sl.draft_each = 0.0;
    }
    b.dflash(7, sl);
    if (cap > 0.0) b.dfl_verify_cap(cap);
    Session s;
    s.prompt.assign(300, 5);
    s.first = 299;
    s.chunk = 8;
    s.n_gen = 32;
    s.dflash = true;
    b.admit({0}, {s});
    while (b.round({0})) {}
    return res;
}

int main()
{
    const float lo_lp[8]  = {0.f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f};
    const float mid_lp[8] = {0.f, std::log(0.9f), std::log(0.8f), std::log(0.2f), -2.f, -2.f, -2.f, -2.f};
    const float hi_lp[8]  = {0.f, std::log(0.99f), std::log(0.99f), std::log(0.99f), std::log(0.99f), std::log(0.99f), std::log(0.99f), std::log(0.99f)};

    // 1. Low-confidence survival cap -> M=1 (K*=0), 100% bit-exact emitted tokens
    const SimRes lo_off = run_cap(0.0, lo_lp, 0);
    const SimRes lo_on  = run_cap(0.4, lo_lp, 0);
    CHECK(lo_off.emitted == lo_on.emitted);
    CHECK(lo_off.emitted.size() == 32);
    CHECK(!lo_on.verify_m.empty());
    for (int m : lo_on.verify_m) CHECK(m == 1);

    // 2. Medium-confidence survival cap (0.9 * 0.8 = 0.72 >= 0.5, * 0.2 = 0.144 < 0.5) -> M=3 (K*=2), bit-exact
    const SimRes mid_off = run_cap(0.0, mid_lp, 2);
    const SimRes mid_on  = run_cap(0.5, mid_lp, 2);
    CHECK(mid_off.emitted == mid_on.emitted);
    for (int m : mid_on.verify_m) CHECK(m == 3);

    // 3. High-confidence survival cap (0.99^7 = 0.932 >= 0.5) -> M=8 (K*=7), bit-exact
    const SimRes hi_off = run_cap(0.0, hi_lp, 7);
    const SimRes hi_on  = run_cap(0.5, hi_lp, 7);
    CHECK(hi_off.emitted == hi_on.emitted);
    for (int m : hi_on.verify_m) CHECK(m == 8);

    // 4. SpecLen with allow_k0() (GR8_SPEC_K0=1): learns low acceptance -> drops to M=1 (K*=0) cleanly, bit-exact
    const SimRes k0_lo = run_cap(0.0, lo_lp, 0, true);
    CHECK(k0_lo.emitted == lo_off.emitted);
    bool saw_k0 = false;
    for (int m : k0_lo.verify_m) if (m == 1) saw_k0 = true;
    CHECK(saw_k0);

    // 5. SpecLen with allow_k0() on high acceptance -> reaches M=8 (K*=7), bit-exact over all n_gen tokens
    const SimRes k0_hi = run_cap(0.0, hi_lp, 7, true);
    CHECK(k0_hi.emitted.size() >= 32 && hi_off.emitted.size() >= 32);
    CHECK(std::equal(k0_hi.emitted.begin(), k0_hi.emitted.begin() + 32, hi_off.emitted.begin()));
    CHECK(!k0_hi.verify_m.empty() && k0_hi.verify_m.back() == 8);

    std::printf("batch_dfl_verify_cap_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
