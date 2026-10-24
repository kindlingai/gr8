// G6_L2PF / G6_MTP_L2PF queue order, CPU only (no GPU, no launch): l2pf::queue over a main layer's plan gives the queue of
// before the MTP draft targets existed (targets T_XIN .. T_SHARED in enum order, same chunks), the MTP draft's plan reads
// T_EH first and T_HEAD last, and the largest draft plan (trellis q_b / o_proj, FP8 kv_b with scales, two shared ranges,
// eh_proj, the two-stage head's four ranges) fits MAXR with no range dropped.
// build + run (scripts/build.sh does): nvcc -O2 -std=c++17 -I engine/kernels -o l2pf_queue_test engine/kernels/tests/l2pf_queue_test.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_bf16.h>
#include "../l2pf.cuh"

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const uint8_t* at(size_t off) { return reinterpret_cast<const uint8_t*>(uintptr_t(0x100000000ull) + off); }

// the queue of before (targets in enum order 0..7, no READ_ORDER)
static l2pf::Queue old_queue(const l2pf::Plan& P, unsigned tmask)
{
    l2pf::Queue q{};
    uint32_t c = 0;
    for (int t = 0; t <= l2pf::T_SHARED; ++t)
    {
        q.from[t] = c;
        if (!(tmask >> t & 1)) continue;
        for (int k = 0; k < 4 && P.n[t][k]; ++k)
        {
            const uintptr_t b = reinterpret_cast<uintptr_t>(P.p[t][k]), lo = (b + 15) & ~uintptr_t(15), hi = (b + P.n[t][k]) & ~uintptr_t(15);
            if (hi <= lo) continue;
            q.p[q.count] = reinterpret_cast<const uint8_t*>(lo);
            q.n[q.count] = uint32_t(hi - lo);
            c += (q.n[q.count] + l2pf::CHUNK - 1) / l2pf::CHUNK;
            q.end[q.count++] = c;
        }
    }
    return q;
}
static bool same(const l2pf::Queue& a, const l2pf::Queue& b)
{
    if (a.count != b.count) return false;
    for (int i = 0; i < a.count; ++i)
        if (a.p[i] != b.p[i] || a.n[i] != b.n[i] || a.end[i] != b.end[i]) return false;
    for (int t = 0; t <= l2pf::T_SHARED; ++t)
        if (a.from[t] != b.from[t]) return false;
    return true;
}
// a main layer as plan_layer gives it: x_in 4, idx_q, trellis q_b 3, UK + scales, UV + scales, trellis o_proj 3, router,
// shared gate + up (18 ranges)
static l2pf::Plan main_plan()
{
    l2pf::Plan P;
    size_t o = 0;
    auto add = [&](int t, size_t n) { P.add(t, at(o), n); o += (n + 4095) & ~size_t(4095); };
    add(l2pf::T_XIN, 2228224); add(l2pf::T_XIN, 608256); add(l2pf::T_XIN, 135168); add(l2pf::T_XIN, 33792);
    add(l2pf::T_IDXQ, 1441792);
    add(l2pf::T_QB, 4096); add(l2pf::T_QB, 5632); add(l2pf::T_QB, 2883584);
    add(l2pf::T_UK, 1081344); add(l2pf::T_UK, 33792);
    add(l2pf::T_UV, 1441792); add(l2pf::T_UV, 45056);
    add(l2pf::T_OPROJ, 5632); add(l2pf::T_OPROJ, 12288); add(l2pf::T_OPROJ, 8000000);
    add(l2pf::T_ROUTER, 3145728);
    add(l2pf::T_SHARED, 2097152); add(l2pf::T_SHARED, 2097152);
    return P;
}

int main()
{
    const unsigned all = (1u << l2pf::NTG) - 1;
    // 1. a main layer's queue is the queue of before, for every target mask of the old 8 bits (and with the new bits set)
    {
        const l2pf::Plan P = main_plan();
        for (unsigned m = 0; m < 256; ++m)
        {
            EXPECT(same(l2pf::queue(P, m), old_queue(P, m)));
            EXPECT(same(l2pf::queue(P, m | 0x300u), old_queue(P, m)));
        }
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 18);
        EXPECT(q.from[l2pf::T_EH] == 0 && q.from[l2pf::T_XIN] == 0);
        EXPECT(q.from[l2pf::T_HEAD] == q.end[q.count - 1]);     // empty: a window from it has nothing to issue
    }
    // 2. the MTP draft's plan: T_EH first, T_HEAD last (the two-stage head: codes / scales of an eighth, then the rest)
    {
        l2pf::Plan P = main_plan();
        const size_t eh = 13 * 1048576, cb = 25813ull * 96 * 16, sb = 25813ull * 192 * 2, r1 = 25813 / 8;
        P.add(l2pf::T_EH, at(size_t(64) << 30), eh);
        const uint8_t* codes = at(size_t(65) << 30);
        const uint8_t* scales = at(size_t(66) << 30);
        P.add(l2pf::T_HEAD, codes, r1 * 96 * 16);
        P.add(l2pf::T_HEAD, scales, r1 * 192 * 2);
        P.add(l2pf::T_HEAD, codes + r1 * 96 * 16, cb - r1 * 96 * 16);
        P.add(l2pf::T_HEAD, scales + r1 * 192 * 2, sb - r1 * 192 * 2);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 23 && q.count <= l2pf::MAXR);          // nothing dropped
        EXPECT(q.p[0] == at(size_t(64) << 30) && q.from[l2pf::T_EH] == 0);
        EXPECT(q.from[l2pf::T_XIN] == (eh + l2pf::CHUNK - 1) / l2pf::CHUNK);
        EXPECT(q.p[q.count - 4] == codes && q.p[q.count - 3] == scales);
        EXPECT(q.from[l2pf::T_HEAD] == q.end[q.count - 5]);
        // every target after T_EH keeps its order and its chunk counts (shifted by T_EH's chunks)
        const l2pf::Queue m = old_queue(main_plan(), all);
        for (int i = 0; i < m.count; ++i) EXPECT(q.p[i + 1] == m.p[i] && q.n[i + 1] == m.n[i] && q.end[i + 1] == m.end[i] + q.end[0]);
        // G6_L2PF_T without bit 8 / 9 (a mask set before these targets existed) drops them, nothing else
        EXPECT(same(l2pf::queue(P, 0xffu), m));
    }
    // 3. the front of the head-merge tails: T_EH then x_in
    {
        l2pf::Plan P;
        P.add(l2pf::T_EH, at(0), 13 * 1048576);
        P.add(l2pf::T_XIN, at(size_t(1) << 30), 2228224);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 2 && q.p[0] == at(0) && q.from[l2pf::T_EH] == 0 && q.from[l2pf::T_XIN] == q.end[0]);
    }
    printf("l2pf_queue_test: %s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
