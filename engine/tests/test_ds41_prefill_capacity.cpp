#include "../runtime/ds41_prefill_capacity.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <initializer_list>

int main() {
    assert(ds41_prefill_rowcap(nullptr, 32) == 32);
    for (int w : {32, 64, 128, 256, 1024, 2048, 4096}) {
        const auto s = std::to_string(w);
        assert(ds41_prefill_rowcap(s.c_str(), 32) == w);
    }
    for (const char* bad : {"", "0", "31", "4097", "x", "128x", "-1", "999999999999999999999"})
        assert(ds41_prefill_rowcap(bad, 32) == 0);
    assert(ds41_prefill_fabric_payload(32, 5120, 0) == size_t(32) * 5120 * 2);
    assert(ds41_prefill_fabric_payload(32, 5120, 6144) == size_t(32) * 6144 * 2);
    assert(ds41_prefill_fabric_payload(1024, 5120, 0) == 10485760);
    assert(ds41_prefill_fabric_payload(1024, 5120, 6144) == 12582912);
    assert(ds41_prefill_fabric_payload(2048, 5120, 6144) == 25165824);
    std::puts("DS41 prefill capacity PASS decode32 W32..4096, invalid refused, fabric OFF/ON W1024/2048");
}
