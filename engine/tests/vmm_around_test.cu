// engine/vmm.h alloc_around with a cuMemCreate'd handle standing in for
// atticd's slice: the mappings it makes, every byte written and read back through the range, the ordinary memory it
// costs beyond the bytes, the memory back after release, and the refusals.
//   nvcc -O2 -std=c++17 -I engine -o vmm_around_test engine/tests/vmm_around_test.cu && ./vmm_around_test   (a GB10, ~1 min)
#include "vmm.h"
#include <fstream>

static std::map<std::string, long> meminfo()
{
    std::map<std::string, long> m;
    std::ifstream f("/proc/meminfo");
    std::string k, unit;
    long v;
    while (f >> k >> v)
    {
        m[k.substr(0, k.size() - 1)] = v;
        if (f.peek() != '\n') f >> unit;
    }
    return m;
}
static double gib(long kb) { return kb / 1048576.0; }

__global__ void fill(uint64_t* p, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = i * 0x9E3779B97F4A7C15ull;
}
__global__ void check(const uint64_t* p, size_t n, unsigned long long* bad)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != i * 0x9E3779B97F4A7C15ull) atomicAdd(bad, 1ull);
}

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } } while (0)

static CUmemGenericAllocationHandle slice(size_t n)
{
    int dev = 0;
    cudaGetDevice(&dev);
    const CUmemAllocationProp p = g6vmm::prop(dev);
    CUmemGenericAllocationHandle h = 0;
    if (g6vmm::api().create(&h, n, &p, 0)) { printf("FAIL cuMemCreate of the stand-in slice\n"); exit(1); }
    return h;
}

// one range: `bytes` with [lo, hi) MiB lent, cut at `piece`; expect `want` mappings
static void one(size_t bytes, size_t lo_mb, size_t hi_mb, size_t piece, size_t want)
{
    const size_t lo = lo_mb << 20, hi = hi_mb << 20;
    const auto m0 = meminfo();
    std::string why;
    void* d = g6vmm::alloc_around(bytes, lo, hi, slice(hi - lo), piece, why);
    EXPECT(d, "alloc_around(%zu, [%zu, %zu) MiB, piece %zu MiB): %s", bytes, lo_mb, hi_mb, piece >> 20, why.c_str());
    if (!d) return;
    const size_t got = g6vmm::mappings(d);
    EXPECT(got == want, "%zu mappings, want %zu", got, want);
    const size_t n = bytes / 8;
    unsigned long long* bad = nullptr;
    cudaMalloc(&bad, sizeof *bad);
    cudaMemset(bad, 0, sizeof *bad);
    fill<<<1024, 256>>>((uint64_t*)d, n);
    check<<<1024, 256>>>((const uint64_t*)d, n, bad);
    unsigned long long b = ~0ull;
    cudaMemcpy(&b, bad, sizeof b, cudaMemcpyDeviceToHost);
    EXPECT(cudaGetLastError() == cudaSuccess && b == 0, "%llu words read back wrong", b);
    cudaFree(bad);
    const auto m1 = meminfo();
    EXPECT(g6vmm::release(d), "release");
    EXPECT(!g6vmm::release(d), "a second release finds nothing");
    EXPECT(g6vmm::mappings(d) == 0, "mappings after release");
    cudaDeviceSynchronize();
    const auto m2 = meminfo();
    printf("%.2f GiB, lent [%zu, %zu) MiB, pieces %zu MiB: %zu mappings, every word read back; MemFree -%.3f GiB "
           "(%+.3f over the bytes), Slab %+.3f GiB; after release MemFree %+.3f GiB of the start\n",
           bytes / 1073741824.0, lo_mb, hi_mb, piece >> 20, got, gib(m0.at("MemFree") - m1.at("MemFree")),
           gib(m0.at("MemFree") - m1.at("MemFree")) - bytes / 1073741824.0, gib(m1.at("Slab") - m0.at("Slab")),
           gib(m2.at("MemFree") - m0.at("MemFree")));
}

int main()
{
    cudaFree(0);
    size_t gmin = 0, grec = 0;
    g6vmm::granularity(0, gmin, grec);
    printf("granularity min %zu KB, recommended %zu KB\n", gmin >> 10, grec >> 10);
    const size_t G = size_t(1) << 30, M = size_t(1) << 20;
    // the image's shape: G.embed's slice at offset 0, the rest cut at 1 GiB multiples: [1816, 2048) MiB, then whole GiB,
    // then the tail (16 GiB + 5 MiB rounds up to 16390 MiB: [16384, 16390) MiB)
    one(16 * G + 5 * M, 0, 1816, G, 1 + 1 + 14 + 1);
    // a slice inside the range: [0, 1024) and [1024, 1026) below it, [2842, 3072) and [3072, 4096) above
    one(4 * G, 1024 + 2, 1024 + 2 + 1816, G, 2 + 1 + 2);
    // the slice at the very end: [0, 1024) and [1024, 1792) below it
    one(1792 * M + 1816 * M, 1792, 1792 + 1816, G, 2 + 1);
    // refusals give the handle back and map nothing
    std::string why;
    EXPECT(!g6vmm::alloc_around(4 * G, 0, 1816 * M, slice(1816 * M), 3 * G, why), "pieces of 3 GiB accepted");
    printf("pieces of 3 GiB: %s\n", why.c_str());
    why.clear();
    EXPECT(!g6vmm::alloc_around(4 * G, 0, 1816 * M, slice(1816 * M), M, why), "pieces of 1 MiB accepted");
    printf("pieces of 1 MiB: %s\n", why.c_str());
    why.clear();
    EXPECT(!g6vmm::alloc_around(4 * G, M, M + 1816 * M, slice(1816 * M), G, why), "a slice at 1 MiB accepted");
    printf("a slice at 1 MiB: %s\n", why.c_str());
    printf(fails ? "%d FAILED\n" : "all OK\n", fails);
    return fails ? 1 : 0;
}
