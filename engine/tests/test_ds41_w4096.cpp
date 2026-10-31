// Host-only checks of the 4096-row read-once bound.
//   dsr::RO_MAX == 4096 == the largest GR8_DS_PREFILL_W (ds41_prefill_rowcap, TAIL_MAX); 4097 refused.
//   The RoScratch tile capacity of ds41_moe_ops.cuh ro_alloc, (ROMAX * TOPK + 7 * NEXP) / 8 = 3408, holds the host plan (dsr::plan_ref, the
//   specification of the device planner) of 4096 rows at the worst case (every expert ends on a 1-token tile) and at uniform and skewed routes;
//   every route in exactly one live column.
//   The Engram fabric payload at W4096 (ds41_step.cuh:142 check) = 4096 x 6144 x 2.
// g++ -std=c++17 -O2 -Wall -Wextra -I engine -I engine/kernels engine/tests/test_ds41_w4096.cpp -o /tmp/test_ds41_w4096 && /tmp/test_ds41_w4096
#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "runtime/ds41_prefill_capacity.h"
#include "runtime/ds41_pf_split.h"
#include "kernels/ds_ro.h"

constexpr int TK = 6, NE = 384, W = 4096;
constexpr int TILES = (W * TK + 7 * NE) / 8;      // ds41_moe_ops.cuh ro_alloc at ROMAX 4096

static void check(const std::vector<int>& ex, const char* what)
{
    std::vector<float> wt(static_cast<size_t>(W) * 8, 1.f / TK);
    const auto P = std::make_unique<dsr::PlanMT<TK, W>>();
    dsr::plan_ref<TK, W>(W, reinterpret_cast<const int (*)[8]>(ex.data()), reinterpret_cast<const float (*)[8]>(wt.data()), *P, TK);
    long live = 0;
    for (int t = 0; t < P->n_tiles; ++t)
        for (int c = 0; c < 8; ++c) live += (P->tile_mask[t] >> c & 1) && P->tile_tok[t][c] >= 0 && P->tile_tok[t][c] < W;
    std::printf("  %s: slots %d tiles %d (capacity %d) live columns %ld\n", what, P->n_slots, P->n_tiles, TILES, live);
    assert(P->n_tiles <= TILES && P->n_tiles <= dsr::pw_tile_bound(W * TK, NE));
    assert(live == static_cast<long>(W) * TK);
}

int main()
{
    static_assert(dsr::RO_MAX == 4096, "W4096: dsr::RO_MAX");
    static_assert(dsr::RO_MAX == ds41::br::TAIL_MAX, "the read-once bound is the largest PREFILL_W");
    assert(ds41_prefill_rowcap("4096", 32) == 4096 && ds41_prefill_rowcap("4097", 32) == 0 && ds41_prefill_rowcap("2048", 32) == 2048);
    assert(dsr::pw_tile_bound(W * TK, NE) == 3408 && TILES == 3408);
    assert(ds41_prefill_fabric_payload(4096, 5120, 6144) == size_t(4096) * 6144 * 2);
    // routes: expert ids per token, 6 distinct per token, 8 slots per row (plan_ref reads [8])
    std::vector<int> ex(static_cast<size_t>(W) * 8, -1);
    auto put = [&](auto f) { for (int m = 0; m < W; ++m) for (int k = 0; k < TK; ++k) ex[static_cast<size_t>(m) * 8 + k] = f(m, k); };
    put([](int m, int k) { return (m * TK + k) % NE; });                          // uniform: 64 routes an expert, full tiles
    check(ex, "uniform");
    // worst case for tiles: 383 experts get 57 = 8 x 7 + 1 routes (each last tile holds 1 token), expert 383 the other 2745. Routes are assigned
    // k-major (route r = k * W + m), in runs of at most W, so a token never holds one expert twice (plan_ref's precondition).
    for (int r = 0; r < W * TK; ++r) ex[static_cast<size_t>(r % W) * 8 + r / W] = r / 57 < NE - 1 ? r / 57 : NE - 1;
    check(ex, "skewed 8q+1");
    put([](int m, int k) { return (k * 61 + (m % 7)) % NE; });                    // hot: 42 experts take every route
    check(ex, "hot 42 experts");
    std::printf("DS41 W4096 host PASS: RO_MAX %d = max PREFILL_W, 4097 refused, RoScratch %d tiles holds the 4096-row plan, Engram payload %zu B\n",
                dsr::RO_MAX, TILES, ds41_prefill_fabric_payload(4096, 5120, 6144));
}
