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
    std::cout << "context_buckets: PASS (R=0/32768 schedule byte-exact; R=98304; BK=4096)\n";
}
