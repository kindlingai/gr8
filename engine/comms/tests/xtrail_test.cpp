// xtrail.h: every proxy stamp index of every rail (0..3, as a ring or hybrid fabric has) stays inside its own record and
// inside the vector, the last record included; the dump reads at most two rails, never another record.
//   g++ -std=c++17 -O2 -Wall -I engine/comms engine/comms/tests/xtrail_test.cpp -o xtrail_test && ./xtrail_test
#include "xtrail.h"

#include <cstdio>
#include <vector>

using namespace g6;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

int main() {
  for (size_t records : {1ul, 7ul, 64ul * 157}) {          // 157 = TP8 decode collectives (2 * 78 + 1)
    std::vector<unsigned> owner(xt_stamps(records), ~0u);
    const size_t last = records - 1;
    for (size_t k = 0; k < records; k++)
      for (int rail = 0; rail < 4; rail++)
        for (int t = 0; t < XT_TICKS; t++) {
          const size_t i = xt_at(k, rail, t);
          CHECK(i < owner.size());
          if (i < owner.size()) { CHECK(owner[i] == ~0u || owner[i] == k); owner[i] = (unsigned)k; }
          CHECK(i >= xt_at(k, 0, 0) && i < xt_at(k, 0, 0) + XT_REC);   // inside its own record
        }
    CHECK(xt_at(last, 3, XT_TICKS - 1) == owner.size() - 1);          // rail 3 of the last record: the last element
    for (int nrails = 1; nrails <= 4; nrails++) {
      CHECK(xt_dump_rails(nrails) <= XT_RAILS);
      for (int r = 0; r < xt_dump_rails(nrails); r++) CHECK(xt_at(last, r, XT_TICKS - 1) < owner.size());
    }
  }
  // the old unfolded layout overran: rail 3 of the last record at stride 8 is 4 past the end
  const size_t records = 4096; CHECK((records - 1) * 8 + 3 * 4 >= records * 8);
  CHECK(xt_rail(0) == 0 && xt_rail(1) == 1 && xt_rail(2) == 1 && xt_rail(3) == 1);
  if (fails) { fprintf(stderr, "xtrail_test: %d failures\n", fails); return 1; }
  printf("xtrail_test: OK (rails 0..3 in bounds, last record included; dump capped at %d rails)\n", XT_RAILS);
  return 0;
}
