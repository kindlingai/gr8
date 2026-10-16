#pragma once
#include <algorithm>

// Decode graph capacities. R must be a boundary so a graph never mixes the replicated and CP paths.
// At most `doublings` buckets below the cap (default N - 1: every slot). If all slots are in use, merge the smallest
// bucket into the next one; preserve the cap. doublings = N - 2 keeps a slot for an R off the doubling (no merge).
template <int N>
inline int context_buckets(int (&bk)[N], int base, int capacity, int R, int doublings = N - 1)
{
    static_assert(N >= 2, "context buckets need room for the cap");
    int n = 0;
    if (base > 0) for (; base < capacity && n < doublings && n < N - 1; base *= 2) bk[n++] = base;
    bk[n++] = capacity;
    if (R > 0 && R < capacity && std::find(bk, bk + n, R) == bk + n)
    {
        if (n == N) { std::move(bk + 1, bk + n, bk); --n; }
        bk[n++] = R;
        std::sort(bk, bk + n);
    }
    return n;
}
