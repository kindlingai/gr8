// GR8_SPIN_TIMEOUT_S -> SM cycles (fabric.h g6::spin_timeout_cyc): seconds at the ~2 GHz GB10 estimate the
// compile-time G6_SPIN_TIMEOUT_CYC documents; unset (or <= 0) keeps the compile value, so a -D override
// (tests) is untouched. The accessor itself (knob::spin_timeout_s, an optional: unset -> nullopt) is covered
// by the knobs table tests.
#include "../fabric.h"
#include <cassert>
#include <climits>
#include <cstdio>

int main()
{
    const long long compile_cyc = G6_SPIN_TIMEOUT_CYC;
    assert(g6::spin_timeout_cyc(0, compile_cyc) == compile_cyc);          // unset keeps the compile value
    assert(g6::spin_timeout_cyc(-5, compile_cyc) == compile_cyc);         // so does a nonsense value
    assert(g6::spin_timeout_cyc(10, 123) == 20000000000LL);               // 10 s -> the documented 20e9 cycles
    assert(g6::spin_timeout_cyc(60, 123) == 120000000000LL);              // defensive raise for long-prefill lanes: 60 s
    assert(g6::spin_timeout_cyc(1, 123) == 2000000000LL);                 // smallest allowed knob
    // Huge values clamp so 2 * spin_cyc (ds41_step.cuh relay wait) never overflows signed 64-bit cycles.
    const long long clamped = g6::spin_timeout_cyc(LLONG_MAX, 123);
    assert(clamped == 2000000000000000000LL);
    assert(clamped <= LLONG_MAX / 2);
    assert(g6::spin_timeout_cyc(1000000001LL, 123) == clamped);
    std::puts("spin_timeout_test ok");
}
