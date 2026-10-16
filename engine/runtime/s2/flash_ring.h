// Single-session KDA rollback, independent of CUDA and the acceptance policy.
#pragma once
#include <stdexcept>
namespace g6s2 {
struct FlashRing {
    int first = -1, rows = 0;
    bool ring = false;
    void reset() { first = -1; rows = 0; ring = false; }
    int source(int next) const {
        if (first < 0) return 0;
        const int accepted = next - first;
        if (accepted < 1 || accepted > rows)
            throw std::runtime_error("Flash next main position outside previous step");
        return ring ? accepted - 1 : 0;
    }
    void record(int pos, int n, bool verify) { first = pos; rows = n; ring = verify; }
};
}
