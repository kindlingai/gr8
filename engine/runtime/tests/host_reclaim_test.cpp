// host_reclaim.h (#4): the /proc parser, and that malloc_trim gives freed heap back to the kernel.
//   g++ -std=c++17 -O2 -Wall -I engine/runtime engine/runtime/tests/host_reclaim_test.cpp && ./a.out
#include "host_reclaim.h"
#include <cstdlib>
#include <vector>

using namespace g6rt;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main()
{
    const char* mi = "MemTotal:       125000000 kB\nMemFree:          1000 kB\nMemAvailable:   118000000 kB\nBuffers: 5 kB\n"
                     "Cached:          4194304 kB\nSwapCached:            7 kB\n";
    CHECK(proc_kb(mi, "MemAvailable") == 118000000);
    CHECK(proc_kb(mi, "Cached") == 4194304);           // not SwapCached
    CHECK(proc_kb(mi, "SwapCached") == 7);
    CHECK(proc_kb("Shmem:  1024 kB\nShmemHugePages: 0 kB\n", "Shmem") == 1024);
    CHECK(proc_kb(mi, "MemTotal") == 125000000);
    CHECK(proc_kb(mi, "Mem") == -1);                   // a prefix of a key is not the key
    CHECK(proc_kb(mi, "VmRSS") == -1);
    CHECK(proc_kb("", "Cached") == -1);
    CHECK(proc_kb("Cached:\n", "Cached") == -1);
    CHECK(kb_gib(1048576) == 1.0 && kb_gib(-1) == -1.0);

    const HostMem m = host_mem();                       // Linux: all three are there
    CHECK(m.avail_kb > 0 && m.cached_kb >= 0 && m.rss_kb > 0);

#if !defined(__GLIBC__)
    printf("host_reclaim_test: parser ok; trim not tested (no glibc)\n");
    return fails ? 1 : 0;
#else
    // 256 MiB in 1 MiB blocks kept in the heap (mmap threshold raised), each followed by a small block that stays live:
    // the freed blocks cannot merge into the heap's top, so free() keeps them resident, as a loader's freed plans and
    // vectors stay between live allocations. malloc_trim must give most of them back (whole pages inside free chunks;
    // the Sparks run 64 KiB pages).
    mallopt(M_MMAP_THRESHOLD, 64 << 20);
    std::vector<char*> v, keep;
    for (int i = 0; i < 256; ++i)
    {
        char* p = (char*)malloc(1 << 20);
        memset(p, 1, 1 << 20);
        v.push_back(p);
        keep.push_back((char*)malloc(64));
    }
    for (char* p : v) free(p);
    const long freed = host_reclaim(0, "test");
    fprintf(stderr, "malloc_trim gave back %ld MiB of 256\n", freed >> 10);
    CHECK(freed >= 192 << 10);
    for (char* p : keep) free(p);

    if (fails) { fprintf(stderr, "host_reclaim_test: %d FAILED\n", fails); return 1; }
    printf("host_reclaim_test: ok (trim gave back %ld MiB)\n", freed >> 10);
    return 0;
#endif
}
