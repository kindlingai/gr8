// CPU-only: G6_PF_ROUTER_SPLIT row partition, payload and 64-byte record layout, then the exchange of every rank's table.
//   g++ -std=c++17 -O1 -Wall -o /tmp/t_rs engine/tests/test_flash_route_split.cpp && /tmp/t_rs
// The model of the kernels (flash_route_split.h pack / unpack) is applied for every rank count 2..MAX_NR: each rank fills only its
// own rows with a per-row route rule (stand-in for k_pf_route<K>: it writes all 8 slots, dropped slots as -1 / 0.0f), packs,
// "sends"; every rank must end byte-equal to the replicated table, for K = 8, 6, 4, 1 (G6_MOE_TOPK). A byte compare, not a float
// compare. The 16-byte index maps of k_rs_put and k_rs_unpack (flash_route_split.cuh, one instance for every rank count, grid y =
// peer) are run as written, for several grid strides, and must write the same bytes as pack / unpack.
#include "../runtime/flash_route_split.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
namespace rs = flash_route_split;
// per-row rule: depends on the row index only (as k_pf_route depends on its logits row only); slots >= K are -1 / 0
static void route_row(int row, int K, int32_t* e, float* w) {
    uint32_t s = 2654435761u * (row + 1);
    for (int k = 0; k < rs::SLOTS; ++k) {
        s = s * 1664525u + 1013904223u;
        if (k < K) { e[k] = int32_t(s >> 8) % 288; if (e[k] < 0) e[k] = -e[k]; float f = float(s >> 9) / float(1u << 23); std::memcpy(&w[k], &f, 4); }
        else { e[k] = -1; w[k] = 0.f; }
    }
    // a NaN payload bit pattern and -0.0f must travel as raw bits, not as values
    if (row % 7 == 0) { uint32_t nan = 0x7fc00123u; std::memcpy(&w[0], &nan, 4); }
    if (row % 11 == 0) { uint32_t nz = 0x80000000u; std::memcpy(&w[K - 1], &nz, 4); }
}
// k_rs_put's loop over 16-byte words, for one (block, thread) stride: word i of the slot <- tok_e / tok_w word of row lo + i / 4
struct U4 { uint32_t v[4]; };
static void put_kernel(const int32_t* tok_e, const float* tok_w, int M, int nr, int rank, uint8_t* slot, int stride) {
    const int lo = rs::r0(M, rank, nr), n = rs::rows(M, rank, nr);
    U4* out = reinterpret_cast<U4*>(slot);
    const U4* e = reinterpret_cast<const U4*>(tok_e + size_t(lo) * rs::SLOTS);
    const U4* w = reinterpret_cast<const U4*>(tok_w + size_t(lo) * rs::SLOTS);
    for (int t = 0; t < stride; ++t)
        for (int i = t; i < n * 4; i += stride)
            out[i] = (i & 3) < 2 ? e[(i >> 2) * 2 + (i & 3)] : w[(i >> 2) * 2 + (i & 3) - 2];
}
// k_rs_unpack over its whole grid: blockIdx.y = peer p (own rank and p >= nr return), then the same word map back
static void unpack_kernel(const std::vector<std::vector<uint8_t>>& slots, int M, int nr, int rank, int grid_y, int32_t* tok_e,
                          float* tok_w, int stride) {
    for (int p = 0; p < grid_y; ++p) {
        if (p >= nr || p == rank) continue;
        const U4* src = reinterpret_cast<const U4*>(slots[p].data());
        const int lo = rs::r0(M, p, nr), n = rs::rows(M, p, nr);
        U4* e = reinterpret_cast<U4*>(tok_e + size_t(lo) * rs::SLOTS);
        U4* w = reinterpret_cast<U4*>(tok_w + size_t(lo) * rs::SLOTS);
        for (int t = 0; t < stride; ++t)
            for (int i = t; i < n * 4; i += stride) {
                const U4 v = src[i];
                if ((i & 3) < 2) e[(i >> 2) * 2 + (i & 3)] = v; else w[(i >> 2) * 2 + (i & 3) - 2] = v;
            }
    }
}
int main() {
    int cases = 0, kernel_cases = 0;
    static_assert(rs::ROW_BYTES == 64, "record is 64 bytes");
    static_assert(rs::ROW_BYTES == 4 * sizeof(U4), "a record is the four 16-byte words of the kernels");
    for (int nr = 2; nr <= rs::MAX_NR; ++nr) {
        for (int M : {1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 127, 255, 256, 257, 1000, 4096, 4097, 8191, 8192, 32767}) {
            // partition: contiguous, exact cover, same formula as prefill.cuh pf_r0 (p * M / nr), padded >= every rank's rows
            int sum = 0;
            for (int p = 0; p < nr; ++p) {
                assert(rs::r0(M, p, nr) == p * M / nr);
                assert(rs::rows(M, p, nr) >= 0 && rs::rows(M, p, nr) <= rs::padded_rows(M, nr));
                if (p) assert(rs::r0(M, p, nr) == rs::r0(M, p - 1, nr) + rs::rows(M, p - 1, nr));
                sum += rs::rows(M, p, nr);
            }
            assert(rs::r0(M, 0, nr) == 0 && rs::r0(M, nr, nr) == M && sum == M);
            assert(rs::payload(M, nr) % 8 == 0 && rs::payload(M, nr) >= size_t(rs::rows(M, 0, nr)) * rs::ROW_BYTES);
            assert(rs::active(M, nr) == (M >= 2 * nr));
            if (rs::active(M, nr)) for (int p = 0; p < nr; ++p) assert(rs::rows(M, p, nr) >= 2);
            for (int K : {8, 6, 4, 1}) {
                std::vector<int32_t> ref_e(size_t(M) * 8); std::vector<float> ref_w(size_t(M) * 8);
                for (int r = 0; r < M; ++r) route_row(r, K, &ref_e[size_t(r) * 8], &ref_w[size_t(r) * 8]);
                // each rank: garbage table, own rows routed, own slice packed into a slot of exactly `payload` bytes + tail guard
                std::vector<std::vector<int32_t>> te(nr, std::vector<int32_t>(size_t(M) * 8, 0x5a5a5a5a));
                std::vector<std::vector<float>> tw(nr);
                std::vector<std::vector<uint8_t>> slot(nr, std::vector<uint8_t>(rs::payload(M, nr) + 8, 0xcd));
                for (int q = 0; q < nr; ++q) {
                    tw[q].assign(size_t(M) * 8, 0.f); std::memset(tw[q].data(), 0xa5, tw[q].size() * 4);
                    for (int r = rs::r0(M, q, nr); r < rs::r0(M, q + 1, nr); ++r) route_row(r, K, &te[q][size_t(r) * 8], &tw[q][size_t(r) * 8]);
                    rs::pack(te[q].data(), tw[q].data(), M, nr, q, slot[q].data());
                    for (size_t i = rs::payload(M, nr); i < slot[q].size(); ++i) assert(slot[q][i] == 0xcd);   // pack stays inside the payload
                }
                for (int q = 0; q < nr; ++q)
                    for (int p = 0; p < nr; ++p)
                        if (p != q) rs::unpack(slot[p].data(), M, nr, p, te[q].data(), tw[q].data());
                // the kernels' word maps: same slot bytes as pack, same tables as unpack (grid strides: one CTA, the launch's)
                if (K == 8 || M <= 1000) for (int stride : {1, 128, 32 * 128 + 1}) {
                    std::vector<std::vector<int32_t>> ke(nr, std::vector<int32_t>(size_t(M) * 8, 0x5a5a5a5a));
                    std::vector<std::vector<float>> kw(nr, std::vector<float>(size_t(M) * 8));
                    std::vector<std::vector<uint8_t>> kslot(nr, std::vector<uint8_t>(rs::payload(M, nr) + 8, 0xcd));
                    for (int q = 0; q < nr; ++q) {
                        std::memset(kw[q].data(), 0xa5, kw[q].size() * 4);
                        for (int r = rs::r0(M, q, nr); r < rs::r0(M, q + 1, nr); ++r) route_row(r, K, &ke[q][size_t(r) * 8], &kw[q][size_t(r) * 8]);
                        put_kernel(ke[q].data(), kw[q].data(), M, nr, q, kslot[q].data(), stride);
                        assert(std::memcmp(kslot[q].data(), slot[q].data(), kslot[q].size()) == 0);
                    }
                    for (int q = 0; q < nr; ++q) {
                        unpack_kernel(kslot, M, nr, q, nr, ke[q].data(), kw[q].data(), stride);
                        assert(std::memcmp(ke[q].data(), ref_e.data(), ref_e.size() * 4) == 0);
                        assert(std::memcmp(kw[q].data(), ref_w.data(), ref_w.size() * 4) == 0);
                    }
                    ++kernel_cases;
                }
                for (int q = 0; q < nr; ++q) {
                    assert(std::memcmp(te[q].data(), ref_e.data(), ref_e.size() * 4) == 0);
                    assert(std::memcmp(tw[q].data(), ref_w.data(), ref_w.size() * 4) == 0);
                    for (int r = 0; r < M; ++r) for (int k = K; k < rs::SLOTS; ++k) {
                        assert(te[q][size_t(r) * 8 + k] == -1);
                        assert(tw[q][size_t(r) * 8 + k] == 0.f);
                    }
                }
                ++cases;
            }
        }
    }
    printf("test_flash_route_split: %d exchange cases (nr 2..%d x 21 M x K 8/6/4/1) and %d kernel word-map cases byte-equal, "
           "partition / payload / active ok\n", cases, rs::MAX_NR, kernel_cases);
    return 0;
}
