#include "context_buckets.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

static std::vector<int> old_buckets(int base, int capacity)
{
    std::vector<int> out;
    if (base > 0) for (; base < capacity && out.size() < 7; base *= 2) out.push_back(base);
    out.push_back(capacity);
    return out;
}

static std::vector<int> buckets(int base, int capacity, int r)
{
    int out[8] = {};
    int n = context_buckets(out, base, capacity, r);
    assert(n >= 1 && n <= 8);
    assert(out[n - 1] == capacity);
    assert(std::is_sorted(out, out + n));
    return {out, out + n};
}

// the engine's list (rank.cuh: NBK = 9 slots, 7 doublings)
static std::vector<int> buckets9(int base, int capacity, int r)
{
    int out[9] = {};
    int n = context_buckets(out, base, capacity, r, 7);
    assert(n >= 1 && n <= 9);
    assert(out[n - 1] == capacity);
    assert(std::is_sorted(out, out + n));
    return {out, out + n};
}

int main()
{
    // With CP off or the usual R=32768, every graph capacity matches the old schedule.
    for (int cap : {65536, 131072, 262144, 614400})
        for (int base : {0, 512, 1024, 2048, 4096, 8192})
            for (int r : {0, 32768})
            {
                if (r && !base) continue;
                int actual[8] = {}, expected[8] = {};
                const auto old = old_buckets(base, cap);
                std::copy(old.begin(), old.end(), expected);
                assert(context_buckets(actual, base, cap, r) == int(old.size()));
                assert(std::memcmp(actual, expected, sizeof actual) == 0);
            }

    // An extra R boundary fits even when all eight slots were previously used.
    assert((buckets(2048, 614400, 98304) ==
            std::vector<int>{4096, 8192, 16384, 32768, 65536, 98304, 131072, 614400}));
    assert((buckets(4096, 614400, 98304) ==
            std::vector<int>{8192, 16384, 32768, 65536, 98304, 131072, 262144, 614400}));
    assert((buckets(4096, 131072, 98304) ==
            std::vector<int>{4096, 8192, 16384, 32768, 65536, 98304, 131072}));
    assert((buckets(0, 131072, 98304) == std::vector<int>{98304, 131072}));
    assert(buckets(2048, 614400, 131072) == old_buckets(2048, 614400));

    // Nine slots, seven doublings: the eight-slot list whenever R is 0 or one of the seven doublings (at the default
    // base 2048: every power of two from 4096 to 131072), and an R off them is added without merging the smallest bucket.
    for (int cap : {65536, 131072, 212224, 262144, 310528, 408832, 614400})
        for (int base : {0, 512, 1024, 2048, 4096, 8192})
            for (int r : {0, 4096, 32768, 65536, 131072, 262144})
            {
                const auto old = old_buckets(base, cap);
                if (r && (!base || std::find(old.begin(), old.end(), r) == old.end())) continue;
                assert(buckets9(base, cap, r) == buckets(base, cap, r));
            }
    assert((buckets9(2048, 212224, 98304) ==
            std::vector<int>{2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 212224}));
    assert((buckets9(2048, 614400, 98304) ==
            std::vector<int>{2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 614400}));
    assert((buckets9(2048, 310528, 65536) ==
            std::vector<int>{2048, 4096, 8192, 16384, 32768, 65536, 131072, 310528}));
    std::cout << "context_buckets: PASS (R=0/32768 schedule byte-exact; R=98304; BK=4096; 9 slots = 8 for R on the doubling)\n";
}
