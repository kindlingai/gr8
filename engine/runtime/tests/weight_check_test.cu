// The weight check's sums (weight_check.cuh): the GPU's equal the CPU's for whole and partial chunks and a partial last
// word, and one changed byte anywhere changes its chunk's sum and no other.
//   TEST=weight_check_test scripts/c1_tests.sh
#include <cstdio>
#include <random>
#include <vector>
#include "../weight_check.cuh"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

static std::vector<uint64_t> host_sums(const std::vector<uint8_t>& h, uint64_t chunk)
{
    std::vector<uint64_t> v;
    for (uint64_t off = 0; off < h.size(); off += chunk)
        v.push_back(wcheck::host_sum(h.data() + off, off, std::min<uint64_t>(chunk, h.size() - off)));
    return v;
}

int main()
{
    std::mt19937_64 rng(7);
    cudaStream_t st;
    cudaStreamCreate(&st);
    // whole chunks; a partial last chunk; a partial last word; a chunk that is not a multiple of the block stride
    const struct { uint64_t bytes, chunk; } cases[] = {{8 << 20, 1 << 20}, {(5 << 20) + 4096, 1 << 20},
                                                       {(3 << 20) + 13, 1 << 20}, {100003, 4096}, {7, 4096}};
    for (const auto& c : cases)
    {
        std::vector<uint8_t> h(c.bytes);
        for (auto& b : h) b = uint8_t(rng());
        uint8_t* d = nullptr;
        cudaMalloc(&d, c.bytes);
        cudaMemcpy(d, h.data(), c.bytes, cudaMemcpyHostToDevice);
        const std::vector<uint64_t> want = host_sums(h, c.chunk);
        const std::vector<uint64_t> got = wcheck::device_sums(d, c.bytes, c.chunk, st);
        CHECK(got == want, "%llu bytes in chunks of %llu: GPU sums differ from the CPU's", (unsigned long long)c.bytes,
              (unsigned long long)c.chunk);
        for (int k = 0; k < 4; ++k)      // one byte, anywhere, the last byte included
        {
            const uint64_t at = k == 3 ? c.bytes - 1 : rng() % c.bytes;
            uint8_t x = h[at] ^ uint8_t(1 + rng() % 255);
            cudaMemcpy(d + at, &x, 1, cudaMemcpyHostToDevice);
            const std::vector<uint64_t> now = wcheck::device_sums(d, c.bytes, c.chunk, st);
            for (size_t j = 0; j < now.size(); ++j)
                CHECK((now[j] != want[j]) == (j == at / c.chunk), "byte %llu changed: chunk %zu %s",
                      (unsigned long long)at, j, now[j] != want[j] ? "changed" : "did not change");
            cudaMemcpy(d + at, &h[at], 1, cudaMemcpyHostToDevice);
        }
        // two words swapped: an order-free sum of the words alone would miss it
        if (c.bytes >= 16)
        {
            std::vector<uint8_t> s = h;
            for (int b = 0; b < 8; ++b) std::swap(s[b], s[8 + b]);
            CHECK(s == h || host_sums(s, c.chunk)[0] != want[0], "swapped words: same sum");
        }
        cudaFree(d);
        printf("%s %llu bytes, chunks of %llu\n", fails ? "..." : "PASS", (unsigned long long)c.bytes,
               (unsigned long long)c.chunk);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "weight_check_test: FAIL\n" : "weight_check_test: PASS\n");
    return fails ? 1 : 0;
}
