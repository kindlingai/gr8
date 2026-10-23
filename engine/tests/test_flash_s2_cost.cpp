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
    g6rt::SpecLen ref; ref.flash_tp2();
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
