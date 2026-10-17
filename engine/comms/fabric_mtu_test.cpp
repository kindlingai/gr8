#include "fabric_mtu.h"
#include <cassert>
#include <cstdio>
int main() {
    using namespace g6;
    for (uint32_t a = 1; a <= 5; ++a) for (uint32_t b = 1; b <= 5; ++b) {
        assert(valid_fabric_mtu(a) && valid_fabric_mtu(b));
        const auto mtu = fabric_mtu_min(a, b);
        assert(mtu <= a && mtu <= b && valid_fabric_mtu(mtu));
        assert(mtu == fabric_mtu_min(b, a));
    }
    assert(!valid_fabric_mtu(0) && !valid_fabric_mtu(6));
    assert(!valid_fabric_mtu(1024) && !valid_fabric_mtu(UINT32_MAX));
    assert(fabric_mtu_min(3, 5) == 3 && fabric_mtu_min(5, 3) == 3);
    std::puts("fabric MTU: 25 endpoint pairs, invalid wire values and 1024/4096 regression PASS");
}
