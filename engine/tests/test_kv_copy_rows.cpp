// GR8_KV_SHARE_PAGES (T4): s2.cuh k_kv_copy_page_rows's per-binding routine (runtime/s2/kv_copy_rows.h) on the host, over the
// GLM-5.3 layout: 79 latent caches (attn/kvfmt.h g6kv::ROW B a row: 416 with -DG6_KV4=1, 656 without) and 22 index caches
// (g6kv::IDX_ROW = 132: a page is 64 x 128 B of codes, then 64 FP32 scales). A copy of rows [0, r) of page `from` to page
// `to` must make those rows of both planes bit-identical and leave rows [r, 64) of `to`, all of `from` and every other page
// as they were, for r = 1, 17, 48, 63, 64, with the kernel's 16 B words over 256 threads and with 4 B words.
// build and run:  g++ -O2 -std=c++17 [-DG6_KV4=1] -I engine/attn -I engine/runtime engine/tests/test_kv_copy_rows.cpp \
//                 -o /tmp/test_kv_copy_rows && /tmp/test_kv_copy_rows
#include "kvfmt.h"
#include "s2/kv_copy_rows.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

struct W16 { uint32_t x, y, z, w; };   // the device's uint4
constexpr int MLA = 79, IDX = 22, PAGES = 9;

static uint64_t s_ = 0x6a09e667f3bcc909ull;
static uint8_t rnd() { s_ ^= s_ << 13; s_ ^= s_ >> 7; s_ ^= s_ << 17; return uint8_t(s_ >> 24); }

template <class W>
static void run(unsigned r, unsigned from, unsigned to)
{
    const size_t mp = 64 * (size_t)g6kv::ROW, ip = 64 * (size_t)g6kv::IDX_ROW, codes = g6kv::IDX_ROW - 4;
    std::vector<std::vector<uint8_t>> mla(MLA, std::vector<uint8_t>(PAGES * mp)), idx(IDX, std::vector<uint8_t>(PAGES * ip));
    for (auto& b : mla) for (auto& x : b) x = rnd();
    for (auto& b : idx) for (auto& x : b) x = rnd();
    const auto mla0 = mla, idx0 = idx;
    const g6kvcopy::Copy c{from, to, r};
    // grid (bindings, copies): binding l holds latent cache l and, for the first 22, an index cache
    for (int l = 0; l < MLA; ++l)
        for (unsigned t = 0; t < 256; ++t)
            g6kvcopy::copy_page_rows<W>(reinterpret_cast<W*>(mla[l].data()), l < IDX ? reinterpret_cast<W*>(idx[l].data()) : nullptr,
                                        g6kv::ROW, g6kv::IDX_ROW, c, t, 256);
    for (int l = 0; l < MLA; ++l)
        for (unsigned p = 0; p < PAGES; ++p)
            for (unsigned row = 0; row < 64; ++row)
            {
                const uint8_t* got = mla[l].data() + p * mp + row * g6kv::ROW;
                const uint8_t* want = (p == to && row < r) ? mla0[l].data() + from * mp + row * g6kv::ROW : mla0[l].data() + p * mp + row * g6kv::ROW;
                CHECK(memcmp(got, want, g6kv::ROW) == 0);
            }
    for (int l = 0; l < IDX; ++l)
        for (unsigned p = 0; p < PAGES; ++p)
            for (unsigned row = 0; row < 64; ++row)
            {
                const unsigned src = (p == to && row < r) ? from : p;
                CHECK(memcmp(idx[l].data() + p * ip + row * codes, idx0[l].data() + src * ip + row * codes, codes) == 0);
                CHECK(memcmp(idx[l].data() + p * ip + 64 * codes + row * 4, idx0[l].data() + src * ip + 64 * codes + row * 4, 4) == 0);
            }
}

int main()
{
    static_assert(g6kv::IDX_ROW == 132, "the GLM-5.3 index row");
    static_assert(g6kv::ROW % 16 == 0 && (g6kv::IDX_ROW - 4) % 16 == 0, "the kernel's 16 B words cover any row count");
    int n = 0;
    for (unsigned r : {1u, 17u, 48u, 63u, 64u})
        for (auto [from, to] : {std::pair<unsigned, unsigned>{2, 5}, {7, 0}, {0, 8}})
        {
            run<W16>(r, from, to);
            run<uint32_t>(r, from, to);
            n += 2;
        }
    printf("test_kv_copy_rows %s (row %d B, index row %d B): %d cases, %d failures\n", g6kv::NAME, g6kv::ROW, g6kv::IDX_ROW, n, fails);
    return fails != 0;
}
