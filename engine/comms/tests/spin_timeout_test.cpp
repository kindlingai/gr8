// GR8_SPIN_TIMEOUT_S -> SM cycles (fabric.h g6::spin_timeout_cyc): seconds at the ~2 GHz GB10 estimate the
// compile-time G6_SPIN_TIMEOUT_CYC documents; unset (or <= 0) keeps the compile value, so a -D override
// (tests) is untouched. The accessor itself (knob::spin_timeout_s, an optional: unset -> nullopt) is covered
// by the knobs table tests.
#include "../fabric.h"
#include <cassert>
#include <cstdio>

int main()
{
    const unsigned long long compile_cyc = (unsigned long long)G6_SPIN_TIMEOUT_CYC;
    assert(g6::spin_timeout_cyc(0, compile_cyc) == compile_cyc);          // unset keeps the compile value
    assert(g6::spin_timeout_cyc(-5, compile_cyc) == compile_cyc);         // so does a nonsense value
    assert(g6::spin_timeout_cyc(10, 123) == 20000000000ull);              // 10 s -> the documented 20e9 cycles
    assert(g6::spin_timeout_cyc(60, 123) == 120000000000ull);             // the defensive raise for long-prefill lanes: 60 s
    assert(g6::spin_timeout_cyc(1, 123) == 2000000000ull);                // the smallest allowed knob
    std::puts("spin_timeout_test ok");
}
