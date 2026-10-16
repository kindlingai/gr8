// CPU-only TP4 route-table all-gather. Four ranks own disjoint pf_r0 rows and each receives all three peer slices.
// Build: g++ -std=c++17 -O1 -Wall -o /tmp/t_rs4 engine/tests/test_flash_route_split_tp4.cpp && /tmp/t_rs4
#include "../runtime/flash_route_split.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
namespace rs = flash_route_split;
int main() {
    constexpr int NR = 4;
    int cases = 0;
    for (int M : {8, 9, 63, 64, 65, 4095, 4096, 4097, 8191, 8192}) for (int K : {8, 4}) {
        std::vector<int32_t> ref_e(size_t(M) * 8);
        std::vector<float> ref_w(size_t(M) * 8);
        for (int r = 0; r < M; ++r) for (int k = 0; k < 8; ++k) {
            ref_e[size_t(r) * 8 + k] = k < K ? (r * 17 + k * 31) % 288 : -1;
            ref_w[size_t(r) * 8 + k] = k < K ? float(r + k + 1) / 8192.f : 0.f;
        }
        std::vector<std::vector<int32_t>> got_e(NR, std::vector<int32_t>(size_t(M) * 8, -999));
        std::vector<std::vector<float>> got_w(NR, std::vector<float>(size_t(M) * 8, -999.f));
        std::vector<std::vector<uint8_t>> slots(NR, std::vector<uint8_t>(rs::payload(M, NR) + 8, 0x5a));
        for (int p = 0; p < NR; ++p) {
            const int lo = rs::r0(M, p, NR), n = rs::rows(M, p, NR);
            assert(lo == p * M / NR && n >= 2);
            std::memcpy(got_e[p].data() + size_t(lo) * 8, ref_e.data() + size_t(lo) * 8, size_t(n) * 8 * 4);
            std::memcpy(got_w[p].data() + size_t(lo) * 8, ref_w.data() + size_t(lo) * 8, size_t(n) * 8 * 4);
            rs::pack(got_e[p].data(), got_w[p].data(), M, NR, p, slots[p].data());
            for (size_t i = rs::payload(M, NR); i < slots[p].size(); ++i) assert(slots[p][i] == 0x5a);
        }
        for (int q = 0; q < NR; ++q) {
            for (int p = 0; p < NR; ++p) if (p != q)
                rs::unpack(slots[p].data(), M, NR, p, got_e[q].data(), got_w[q].data());
            assert(std::memcmp(got_e[q].data(), ref_e.data(), ref_e.size() * 4) == 0);
            assert(std::memcmp(got_w[q].data(), ref_w.data(), ref_w.size() * 4) == 0);
        }
        ++cases;
    }
    std::printf("test_flash_route_split_tp4: %d cases (4 ranks, K8/K4), byte-equal\n", cases);
}
