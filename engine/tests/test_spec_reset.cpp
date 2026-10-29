// CPU-only policy snapshot regression for g6m4 --spec-reset-per-request.
// g++ -O2 -std=c++17 -DG6_MODEL_FLASH=1 engine/tests/test_spec_reset.cpp -o /tmp/test_spec_reset
#include "../runtime/spec_len.h"
#include <cassert>
#include <cstdio>
using g6rt::SpecLen;
static void equal(const SpecLen& a, const SpecLen& b) {
    assert(a.on == b.on && a.levels == b.levels);
    assert(a.draft_round == b.draft_round && a.draft_each == b.draft_each);
    assert(a.pos_decay == b.pos_decay && a.bucket_decay == b.bucket_decay && a.prior == b.prior);
    assert(a.rounds == b.rounds && a.pred_sum == b.pred_sum && a.got_sum == b.got_sum);
    for (int j = 0; j <= SpecLen::KX + 1; ++j) assert(a.table[j] == b.table[j]);
    for (int j = 0; j <= SpecLen::KX; ++j) {
        assert(a.seed[j] == b.seed[j] && a.pn[j] == b.pn[j] && a.pk[j] == b.pk[j] && a.chosen[j] == b.chosen[j]);
        for (int k = 0; k < SpecLen::NB; ++k) assert(a.bn[j][k] == b.bn[j][k] && a.bk[j][k] == b.bk[j][k]);
    }
}
static void test(std::vector<int> levels) {
    SpecLen spec; spec.on = true; assert(spec.set_levels(levels)); spec.flash_cost(2);
    spec.pos_decay = .85; spec.bucket_decay = .99; spec.prior = 3; spec.seed[2] = .61;
    const SpecLen spec_initial = spec;
    float lp[8] = {}; for (int j = 1; j < 8; ++j) lp[j] = std::log(.75f);
    SpecLen old_path = spec;
    // OFF preserves the exact carried estimates and choices across five requests.
    for (int q = 0; q < 5; ++q) {
        const bool spec_reset_per_request = false;
        if (spec_reset_per_request) spec = spec_initial;
        for (int n = 0; n < 100; ++n) {
            double a = 0, b = 0;
            const int k = spec.choose(lp, 7, 7, &a);
            assert(k == old_path.choose(lp, 7, 7, &b) && a == b);
            spec.observe(k, n % (k + 1), lp, a);
            old_path.observe(k, n % (k + 1), lp, b);
        }
        equal(spec, old_path);
    }
    assert(spec.rounds == 500 && spec.pn[1] > 0);
    SpecLen first_end;
    // ON restores all fields and gives the same synthetic request decisions five times.
    for (int q = 0; q < 5; ++q) {
        const bool spec_reset_per_request = true;
        if (spec_reset_per_request) spec = spec_initial;
        equal(spec, spec_initial);
        for (int n = 0; n < 100; ++n) {
            double e = 0; const int k = spec.choose(lp, 7, 7, &e);
            spec.observe(k, n % (k + 1), lp, e);
        }
        if (!q) first_end = spec;
        equal(spec, first_end);
    }
    // Poison every mutable entry (including unused index 0) to check the entire reset list.
    for (int j = 0; j < 8; ++j) {
        spec.pn[j] = 1; spec.pk[j] = 2; spec.chosen[j] = 3;
        for (int b = 0; b < 8; ++b) { spec.bn[j][b] = 4; spec.bk[j][b] = 5; }
    }
    spec.rounds = 6; spec.pred_sum = 7; spec.got_sum = 8;
    spec = spec_initial; equal(spec, spec_initial);
}
int main() {
    test({1,2,3,4,5,6,7}); test({2,4,7});
    SpecLen fixed; const auto initial = fixed; fixed = initial; equal(fixed, initial);
    puts("PASS: OFF carried policy exact; ON five synthetic requests exact; all learned fields reset; configured costs/levels/priors preserved.");
}
