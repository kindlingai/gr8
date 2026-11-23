// Host unit test for dual-rail symmetry/balance (dual_rail_of) and rank-0 incast avoidance (stagger_peer).
#include "fabric.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace g6;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d %s: ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); g_fail++; } } while (0)

int main() {
  for (int nr = 2; nr <= 8; ++nr) {
    int total[2] = {0, 0};
    std::vector<int> deg0(nr, 0), deg1(nr, 0);
    for (int me = 0; me < nr; ++me) {
      for (int p = 0; p < nr; ++p) {
        if (p == me) continue;
        const int r_me = dual_rail_of(p, me, nr);
        const int r_p = dual_rail_of(me, p, nr);
        CHECK(r_me == 0 || r_me == 1, "nr=%d (%d,%d): rail %d out of range", nr, me, p, r_me);
        CHECK(r_me == r_p, "nr=%d (%d,%d): asymmetric rails %d vs %d", nr, me, p, r_me, r_p);
        total[r_me]++;
        (r_me == 0 ? deg0[me] : deg1[me])++;
      }
    }
    const int nedges = nr * (nr - 1) / 2;
    const int e0 = total[0] / 2, e1 = total[1] / 2;
    CHECK(e0 + e1 == nedges, "nr=%d: edge sum %d != %d", nr, e0 + e1, nedges);
    CHECK(std::abs(e0 - e1) <= 1, "nr=%d: total rail skew %d vs %d edges (%d vs %d QPs)", nr, e0, e1, total[0], total[1]);
    if (nr == 4) CHECK(total[0] == 6 && total[1] == 6, "nr=4: expected exact 6/6 QPs, got %d/%d", total[0], total[1]);
    if (nr == 8) CHECK(total[0] == 28 && total[1] == 28, "nr=8: expected exact 28/28 QPs, got %d/%d", total[0], total[1]);
    for (int me = 0; me < nr; ++me) {
      const int max_diff = (nr & 1) ? 2 : 1;
      CHECK(std::abs(deg0[me] - deg1[me]) <= max_diff, "nr=%d rank %d: degree skew %d vs %d", nr, me, deg0[me], deg1[me]);
    }
    // Staggered send schedule: in every round k=1..nr-1, every sender hits a distinct receiver (zero incast).
    for (int k = 1; k < nr; ++k) {
      std::vector<int> rx(nr, 0);
      for (int me = 0; me < nr; ++me) {
        const int p = stagger_peer(me, k, nr);
        CHECK(p >= 0 && p < nr && p != me, "nr=%d me=%d k=%d: bad peer %d", nr, me, k, p);
        rx[p]++;
      }
      for (int p = 0; p < nr; ++p) CHECK(rx[p] == 1, "nr=%d round %d: receiver %d fan-in %d != 1", nr, k, p, rx[p]);
    }
  }
  if (g_fail) {
    std::printf("fabric_balance_test: %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("fabric_balance_test: PASS (symmetry, balance 6/6@TP4..28/28@TP8, zero-incast stagger for nr=2..8)\n");
  return 0;
}
