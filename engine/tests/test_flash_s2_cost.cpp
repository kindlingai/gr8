#ifndef G6_MODEL_FLASH
#define G6_MODEL_FLASH 1
#endif
#include "../runtime/s2_batch.h"
#include <cassert>
#include <cstdio>
int main() {
    unsetenv("G6_FLASH_S2_COST"); s2::S2Cost old;
    setenv("G6_FLASH_S2_COST", "1", 1); s2::S2Cost flash;
#if G6_MODEL_FLASH
    // the width's table: TP3's in a -DG6_NR=3 build, TP2's otherwise (no G6_NR here = a host build)
    g6rt::SpecLen ref; ref.flash_cost(s2::S2Cost::flash_tp);
#ifdef G6_NR
    assert(s2::S2Cost::flash_tp == G6_NR);
#else
    assert(s2::S2Cost::flash_tp == 2);
#endif
    g6rt::SpecLen t2, t3; t2.flash_cost(2); t3.flash_cost(3);
    assert(t2.verify_ms(4) == 52.3 && t2.draft_round == 2.8 && t2.draft_each == 2.2);
    assert(t3.verify_ms(4) == 39.6 && t3.verify_ms(8) == 60.7 && t3.draft_round == 2.39 && t3.draft_each == 1.94);
    g6rt::SpecLen t4; t4.flash_cost(4);   // a width without its own table keeps TP2's
    for (int r = 1; r <= 8; ++r) assert(t4.verify_ms(r) == t2.verify_ms(r));
#endif
    setenv("G6_FLASH_S2_COST", "0", 1); s2::S2Cost disabled;
    for (int k = 1; k <= 7; ++k) {
        for (int sessions = 1; sessions <= 8; ++sessions) {
            const double baseline = 44 + 7 * sessions * (k + 1) + 2.1 + 0.25 * sessions +
                                    (k - 1) * (2.0 + 0.15 * sessions);
            assert(old.round(sessions, k) == baseline);
            assert(disabled.round(sessions, k) == baseline);
#if !G6_MODEL_FLASH
            assert(flash.round(sessions, k) == baseline);
#endif
        }
#if G6_MODEL_FLASH
        assert(flash.round(1, k) == ref.verify_ms(k + 1) + ref.draft_ms(k));
        assert(flash.round(2, k) == old.round(2, k));
#endif
    }
    puts("flash_s2_cost: PASS (default/OFF baseline exact; ON Flash-only solo timing)");
}
