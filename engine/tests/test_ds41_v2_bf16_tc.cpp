#include "../runtime/ds41_v2_bf16_tc.h"
#include <cassert>
#include <string>
using ds41::v2::Bf16Tc;
int main() {
    std::vector<int> out{9};
    for (const auto& bad : std::vector<std::vector<int64_t>>{{}, {-1}, {8}, {0, 8}, {7, -1}}) {
        assert(!Bf16Tc::decode(bad, out));
        Bf16Tc s; assert(s.init({7})); s.select(1, 0, 0, nullptr);
        assert(!s.init(bad) && !s.on() && s.seen.empty());
    }
    assert(Bf16Tc::decode({0,1,2,3,4,5,6,7}, out) && out == std::vector<int>({0,1,2,3,4,5,6,7}));
    for (int mask = 0; mask < 8; ++mask) {
        Bf16Tc s; assert(s.init({mask}) && s.on() == (mask != 0));
        for (int rows : {0, 1, 15, 16, 17, 3072}) {
            assert(s.gate(rows) == ((mask & 1) != 0 && rows >= 16));
            assert(s.comp(rows) == ((mask & 2) != 0 && rows >= 16));
            assert(s.idx(rows) == ((mask & 4) != 0 && rows >= 16));
        }
    }
    FILE* log = std::tmpfile(); assert(log); Bf16Tc s;
    s.select(1, 0, 2, log); assert(s.mask == 0);
    assert(s.init({0})); s.select(0, 0, 2, log); s.select(7, 1, 2, log);
    assert(std::ftell(log) == 0 && s.seen.empty());
    assert(s.init({0,7})); s.select(0, 0, 2, log); assert(!s.on() && std::ftell(log) == 0);
    s.select(1, 0, 2, log); s.select(1, 0, 2, log); s.select(1, 1, 2, log); s.select(1, 0, 2, log);
    s.select(2, 0, 2, log); assert(s.mask == 0); s.select(1, 1, 2, log);
    s.select(3, 0, 2, log); assert(s.mask == 7);
    std::rewind(log); char buffer[512]{}; size_t n = std::fread(buffer, 1, sizeof(buffer), log);
    assert(std::string(buffer, n) ==
        "# ds41_v2_bf16_tc rank=2 request=1 slot=0 on=1 mask=7\n"
        "# ds41_v2_bf16_tc rank=2 request=1 slot=1 on=1 mask=7\n"
        "# ds41_v2_bf16_tc rank=2 request=3 slot=0 on=1 mask=7\n");
    std::fclose(log);
    assert(s.init({1,2,4,7}) && s.seen.empty());
    const int cycle[] = {1,2,4,7};
    for (uint64_t r = 0; r < 16; ++r) { s.select(r, 0, 1, nullptr); assert(s.mask == cycle[r % 4]); }
    s.select(UINT64_MAX, 1, 1, nullptr); assert(s.mask == 7);
    puts("PASS test_ds41_v2_bf16_tc: masks, row boundaries, cyclic requests, deduplicated logs, default silent");
}
