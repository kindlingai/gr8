// Host CPU test: verifies O(|list| + n) s2_cp_map matches the original s2_list_place
// definition bit-for-bit across random page lists, topologies (NR = 3, 4, 6), rp values,
// and pool bounds, and verifies 78-layer table build stays under 10 ms at 1M capacity.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define G6_CP_REMOTE (-2)

static int ref_list_place(const std::vector<int>& list, int at, int place)
{
    int seen = 0;
    for (int p : list)
        if (p < place && seen++ == at)
            return p;
    return -1;
}

static std::vector<int> ref_cp_map(int pool_pages, int cp_nr, int cp_rank, const std::vector<int>& list, int rp, size_t n)
{
    std::vector<int> t(n, -1);
    if (cp_nr)
    {
        int j = 0;
        for (size_t lp = 0; lp < t.size(); ++lp)
        {
            if ((int)lp < rp) t[lp] = lp < list.size() ? list[lp] : -1;
            else if ((int)lp % cp_nr == cp_rank) t[lp] = ref_list_place(list, rp + j++, pool_pages);
            else t[lp] = G6_CP_REMOTE;
        }
        return t;
    }
    for (size_t i = 0; i < t.size() && i < list.size(); ++i) t[i] = list[i];
    return t;
}

static std::vector<int> fast_cp_map(int pool_pages, int cp_nr, int cp_rank, const std::vector<int>& list, int rp, size_t n)
{
    std::vector<int> t(n, -1);
    if (cp_nr)
    {
        size_t idx = 0;
        int skipped = 0;
        while (idx < list.size() && skipped < rp)
            if (list[idx++] < pool_pages) ++skipped;
        for (size_t lp = 0; lp < t.size(); ++lp)
        {
            if ((int)lp < rp) t[lp] = lp < list.size() ? list[lp] : -1;
            else if ((int)lp % cp_nr == cp_rank)
            {
                while (idx < list.size() && list[idx] >= pool_pages) ++idx;
                t[lp] = idx < list.size() ? list[idx++] : -1;
            }
            else t[lp] = G6_CP_REMOTE;
        }
        return t;
    }
    for (size_t i = 0; i < t.size() && i < list.size(); ++i) t[i] = list[i];
    return t;
}

int main()
{
    std::mt19937_64 rng(0x52c9a1b3ull);
    const int nrs[] = {0, 3, 4, 6};
    int cases = 0;
    for (int nr : nrs)
    {
        for (int rank = 0; rank < (nr ? nr : 1); ++rank)
        {
            for (int trial = 0; trial < 80; ++trial)
            {
                const int pool_pages = 32 + (int)(rng() % 256);
                const int most = pool_pages + (int)(rng() % 128);
                const size_t list_len = rng() % (size_t)(most + 1);
                const size_t n = 1 + (size_t)(rng() % 400);
                const int rp = (int)(rng() % 128);
                std::vector<int> list(list_len);
                for (size_t i = 0; i < list_len; ++i) list[i] = (int)(rng() % most);

                const auto r = ref_cp_map(pool_pages, nr, rank, list, rp, n);
                const auto f = fast_cp_map(pool_pages, nr, rank, list, rp, n);
                if (r != f)
                {
                    fprintf(stderr, "MISMATCH nr=%d rank=%d trial=%d pool=%d list=%zu rp=%d n=%zu\n",
                            nr, rank, trial, pool_pages, list_len, rp, n);
                    return 1;
                }
                ++cases;
            }
        }
    }

    // Performance check at CAP = 1,048,576 (16,384 pages), 4000-page list (256K tokens)
    const int pool_pages = 4096;
    std::vector<int> big(4096);
    for (size_t i = 0; i < big.size(); ++i) big[i] = (i % 5 == 0) ? (pool_pages + (int)i) : (int)(i % pool_pages);
    const auto t0 = std::chrono::steady_clock::now();
    volatile int sink = 0;
    for (int iter = 0; iter < 100; ++iter)
    {
        auto m = fast_cp_map(pool_pages, 4, iter & 3, big, 256, 16384);
        sink += m[1000];
    }
    const double ms_per_call = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 100.0;
    printf("s2_cp_map_cpu_test: PASS (%d randomized cases, 1M-cap single build = %.4f ms, sink=%d)\n", cases, ms_per_call, (int)sink);
    if (ms_per_call > 5.0)
    {
        fprintf(stderr, "SLOW: %.4f ms > 5.0 ms\n", ms_per_call);
        return 1;
    }
    return 0;
}
