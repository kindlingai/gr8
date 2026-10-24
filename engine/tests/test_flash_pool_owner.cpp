// CPU-only independent construction of complete selected-token rows, then TP2 wire roundtrip.
#include "../attn/flash_pool_owner.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
namespace po = flash_pool_owner;
static std::vector<int> reference(int pos) {
    const int visible = (pos + 1) / 4, keep = std::min(visible, 512);
    std::vector<int> ids(2304, -1);
    // Sorted, nontrivial visible pools, not just a dense prefix.
    for (int p = 0; p < keep; ++p) {
        const int id = visible <= 512 ? p : int(int64_t(p) * (visible - 1) / 511);
        for (int j = 0; j < 4; ++j) ids[p * 4 + j] = id * 4 + j;
    }
    for (int j = 0; j < (pos + 1) % 4; ++j) ids[keep * 4 + j] = (pos + 1) / 4 * 4 + j;
    return ids;
}
int main() {
    int roundtrips = 0, partitions = 0;
    po::Flow lanes[2];
    for (int step = 0; step < 1000; ++step) {
        for (auto& lane : lanes) {
            const int m = 1 + step * 17 % 8192;
            assert(!lane.received(m));
            assert(!lane.publish(0));
            assert(lane.publish(m));
            assert(lane.current == uint64_t(step));
            assert(!lane.publish(m) && lane.next == uint64_t(step + 1));
            assert(!lane.received(m + 1) && lane.awaiting);
            assert(lane.received(m) && !lane.awaiting);
            assert(!lane.received(m));
        }
    }
    po::Flow broken;
    assert(broken.publish(1));
    ++broken.current;
    assert(!broken.received(1));
    for (int m = 1; m <= 16385; ++m) {
        const int mid = po::end(m, 0);
        assert(po::begin(m, 0) == 0 && mid == po::begin(m, 1) && po::end(m, 1) == m);
        assert(mid <= po::padded_rows(m) && m - mid == po::padded_rows(m));
        assert(po::payload(m) == size_t((m + 1) / 2) * 2048);
        assert(po::payload(m) % 8 == 0);
        ++partitions;
    }
    for (int m : {1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 5087, 8191, 8192, 16384}) {
        for (int context : {m, 32768, 131072, 300000}) {
            if (context < m) continue;
            std::vector<int> wire[2];
            for (int rank = 0; rank < 2; ++rank) {
                wire[rank].assign(po::payload(m) / sizeof(int), -1);
                for (int r = po::begin(m, rank); r < po::end(m, rank); ++r) {
                    const int pos = context - m + r;
                    const auto ids = reference(pos);
                    for (int j = 0; j < 512; ++j)
                        wire[rank][size_t(r - po::begin(m, rank)) * 512 + j] = po::pack(ids.data(), pos, j);
                }
                if (rank == 0 && m % 2)
                    for (int j = 0; j < 512; ++j) assert(wire[rank][size_t(m / 2) * 512 + j] == -1);
            }
            // Every receiver reconstructs every peer row; own rows remain unchanged.
            for (int receiver = 0; receiver < 2; ++receiver) {
                const int peer = 1 - receiver;
                for (int r = po::begin(m, peer); r < po::end(m, peer); ++r) {
                    const int pos = context - m + r;
                    const auto expected = reference(pos);
                    const int* pools = wire[peer].data() + size_t(r - po::begin(m, peer)) * 512;
                    for (int j = 0; j < 2304; ++j) assert(po::unpack(pools, pos, j) == expected[j]);
                }
            }
            ++roundtrips;
        }
    }
    // Dense/sparse and incomplete-pool boundaries independent of chunk shape.
    for (int pos : {0,1,2,3,4,2046,2047,2048,2049,2050,2051,131070,299998}) {
        const auto ids = reference(pos);
        int pools[512];
        for (int j = 0; j < 512; ++j) pools[j] = po::pack(ids.data(), pos, j);
        for (int j = 0; j < 2304; ++j) assert(po::unpack(pools, pos, j) == ids[j]);
        ++roundtrips;
    }
    assert(po::payload(8192) == 8388608 && po::payload(5087) == 5210112);
    std::cout << "PASS partitions=" << partitions << " lane sequence transitions=2000 TP2 roundtrip cases=" << roundtrips << "\n";
}
