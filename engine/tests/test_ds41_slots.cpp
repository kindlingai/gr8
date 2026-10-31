// Host test of the SESS slot checks (engine/runtime/ds41_slots.h) that build_ds41 and ds41dsp_ops::build call: no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -I engine engine/tests/test_ds41_slots.cpp
// F3 set B boot 1 (--sess 8 + DSpark) died on every rank: build_ds41 refused ds_slots > 1 for the draft Model.
#include <cstdio>
#include "runtime/ds41_slots.h"

static int fails = 0;
static void expect(bool ok, const char* what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } }

int main()
{
    using namespace ds41slots;
    for (int s = 1; s <= 32; ++s) expect(slots_error(s) == nullptr, "slots 1..32 accepted (main and draft Model: one check)");
    expect(slots_error(0) != nullptr, "slots 0 refused");
    expect(slots_error(33) != nullptr, "slots 33 refused");
    const int MAXS = 8, BLOCK = 5;
    expect(dsp_nsess_error(8, 8, 8, 5, 48, MAXS, BLOCK) == nullptr, "SESS 8 + MTP5 at MX 48 (48 verify rows, 40 draft rows)");
    expect(dsp_nsess_error(8, 32, 32, 5, 48, MAXS, BLOCK) == nullptr, "SESS 32 + MTP5 (capped at 8 in DSpark) at MX 48");
    expect(dsp_nsess_error(1, 1, 1, 5, 32, MAXS, BLOCK) == nullptr, "SESS 1 + MTP5 at MX 32 (F2)");
    expect(dsp_nsess_error(8, 8, 8, 5, 32, MAXS, BLOCK) != nullptr, "SESS 8 + MTP5 at MX 32 refused");
    expect(dsp_nsess_error(8, 8, 1, 5, 48, MAXS, BLOCK) != nullptr, "draft slots != main slots refused");
    expect(dsp_nsess_error(4, 8, 8, 5, 48, MAXS, BLOCK) != nullptr, "nsess != slots refused");
    expect(dsp_nsess_error(9, 9, 9, 1, 96, MAXS, BLOCK) != nullptr, "nsess > MARKOV_MAXS refused");
    expect(dsp_nsess_error(8, 8, 8, 6, 64, MAXS, BLOCK) != nullptr, "k 6 refused");
    expect(dsp_nsess_error(8, 8, 8, 0, 48, MAXS, BLOCK) != nullptr, "k 0 refused");
    if (fails) return 1;
    std::printf("DS41_SLOTS_OK\n");
    return 0;
}
