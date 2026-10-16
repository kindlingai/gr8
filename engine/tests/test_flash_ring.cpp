#include "../runtime/s2/flash_ring.h"
#include <cassert>
#include <cstdio>
int main() {
    g6s2::FlashRing r;
    assert(r.source(31) == 0);
    r.record(31, 1, false); assert(r.source(32) == 0);
    // Each accepted length, including rejection of every draft and full acceptance.
    for (int k = 1; k <= 7; ++k) for (int a = 0; a <= k; ++a) {
        r.record(32, k + 1, true); assert(r.source(33 + a) == a);
    }
    // A forced-token step after partial acceptance, then back to verify.
    r.record(32, 5, true); assert(r.source(35) == 2);
    r.record(35, 1, false); assert(r.source(36) == 0);
    for (int p : {35, 37}) {
        bool failed = false;
        try { r.source(p); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    r.reset(); assert(r.source(0) == 0);
    puts("flash_ring: PASS (all accepted lengths, insertion, reset, invalid positions)");
}
