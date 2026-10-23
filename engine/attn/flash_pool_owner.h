// Flash TP2 owner layout and exact pool/token conversion; shared by CPU tests and CUDA.
#pragma once
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#define G6_POOL_HD __host__ __device__
#else
#define G6_POOL_HD
#endif
namespace flash_pool_owner {
constexpr int POOLS = 512, KP = 4, TOKENS = 2304;
G6_POOL_HD inline int begin(int rows, int rank) { return rank == 0 ? 0 : rows / 2; }
G6_POOL_HD inline int end(int rows, int rank) { return rank == 0 ? rows / 2 : rows; }
G6_POOL_HD inline int padded_rows(int rows) { return rows / 2 + rows % 2; }
G6_POOL_HD inline size_t payload(int rows) { return size_t(padded_rows(rows)) * POOLS * sizeof(int); }
G6_POOL_HD inline int keep(int pos) { const int n = (pos + 1) / KP; return n < POOLS ? n : POOLS; }
G6_POOL_HD inline int pack(const int* ids, int pos, int j) {
    return j < keep(pos) ? ids[j * KP] / KP : -1;
}
G6_POOL_HD inline int unpack(const int* pools, int pos, int j) {
    const int n = keep(pos) * KP;
    if (j < n) return pools[j / KP] * KP + j % KP;
    const int count = pos + 1, tail = count - count % KP + j - n;
    return tail < count ? tail : -1;
}
// Host-side per-lane protocol. A failed transition does not consume a sequence.
struct Flow {
    uint64_t next = 0, current = 0;
    int rows = 0;
    bool awaiting = false;
    bool publish(int m) {
        if (awaiting || m < 1 || next == UINT64_MAX) return false;
        current = next++; rows = m; awaiting = true; return true;
    }
    bool can_receive(int m) const {
        return awaiting && next > 0 && current == next - 1 && rows == m;
    }
    bool received(int m) {
        if (!can_receive(m)) return false;
        awaiting = false; return true;
    }
};
} // namespace flash_pool_owner
#undef G6_POOL_HD
