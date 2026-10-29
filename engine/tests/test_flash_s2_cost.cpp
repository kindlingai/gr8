#ifndef G6_MODEL_FLASH
#define G6_MODEL_FLASH 1
#endif
#include "../runtime/s2_batch.h"
#include <sys/wait.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#if G6_MODEL_FLASH
// S2Cost::flash in a child process with GR8_FLASH_S2_COST=val (null: unset): knobs are read once per process
static bool flash_from_env(const char* val) {
    const pid_t p = fork();
    if (!p) {
        if (val) setenv("GR8_FLASH_S2_COST", val, 1); else unsetenv("GR8_FLASH_S2_COST");
        _exit(s2::S2Cost().flash ? 1 : 0);
    }
    int st = 0;
    waitpid(p, &st, 0);
    assert(WIFEXITED(st));
    return WEXITSTATUS(st) == 1;
}
#endif
int main() {
#if G6_MODEL_FLASH
    assert(!flash_from_env(nullptr) && flash_from_env("1") && !flash_from_env("0"));
    s2::S2Cost old, flash, disabled;
    old.flash = false; flash.flash = true; disabled.flash = false;
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
#else
    setenv("GR8_FLASH_S2_COST", "1", 1);   // the full-GLM cost model ignores it
    s2::S2Cost old, flash, disabled;
#endif
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
