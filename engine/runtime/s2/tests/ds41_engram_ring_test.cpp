// DS41 Engram ring of a snapshot or a park after an image (s2/ds41_engram_ring.h): the ring a prefix-cache entry stores
// (Ds41PrefixCache::capture, then restore) and the one a GR8_DS_PARK resume writes (ds41_native.cuh) must equal the ring
// the live steps left: W-row prefill chunks with -1 at image rows (ds41_engram_ops.cuh k_engram_dead_tok, then the window's
// ring_put_tail), then 1-row decode steps of the generated tokens. An image that ends 0, 1, 2 or 3 text tokens before the
// snapshot position, at the prompt end (a cache hit) and at a request end (a cache hit of the request end, or a park).
#include "s2/ds41_engram_ring.h"
#include <cstdio>
#include <cstring>

using namespace g6s2;
namespace E = ds41::engram;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the live ring after positions [0, end) of `tok`: prefill chunks of W rows over [0, pend) (image rows dead), then 1-row decode
// steps over [pend, end)
static void live(int* ring, const std::vector<int>& tok, int pend, int end, const Ds41ImageRows& im, int W) {
    std::fill_n(ring, E::kRing, -1);
    for (int a = 0; a < pend; a += W) {
        const int rows = std::min(W, pend - a);
        std::vector<int> t(static_cast<size_t>(rows)), p(static_cast<size_t>(rows));
        for (int r = 0; r < rows; ++r) { p[size_t(r)] = a + r; t[size_t(r)] = ds41_image_row(im, a + r) ? -1 : tok[size_t(a + r)]; }
        for (int r = 0; r < rows; ++r) E::ring_put_tail(ring, t.data(), p.data(), rows, r);
    }
    for (int q = pend; q < end; ++q) E::ring_put(ring, q, tok[size_t(q)]);
}
// the reconstruction before the fix: raw tokens, the image token at image rows
static void raw(int* ring, const std::vector<int>& tok, int end) {
    std::fill_n(ring, E::kRing, -1);
    for (int t = std::max(0, end - E::kRing); t < end; ++t) E::ring_put(ring, t, tok[size_t(t)]);
}
static bool same(const int* a, const int* b) { return std::memcmp(a, b, sizeof(int) * E::kRing) == 0; }
// the 3 n-gram predecessors of position q (the window's reads) differ between the two rings
static bool window_differs(const int* a, const int* b, int q) {
    for (int s = 1; s <= 3; ++s) if (E::ring_get(a, q - s) != E::ring_get(b, q - s)) return true;
    return false;
}

int main() {
    constexpr int T = g6rt::TOK_IMAGE;
    int pe_cases = 0, pe_ok = 0, req_cases = 0, req_ok = 0;
    for (int W : {64, 4096}) {
        for (int span : {4, 1599}) {
            for (int gap = 0; gap <= 3; ++gap) {
                const int lead = 300;   // text before the image (an arbitrarily long prefix makes the prompt prefill)
                // prompt end: the snapshot position is prompt.size() - 1 (the last token is a MAIN row, never prefilled)
                std::vector<int> prompt;
                for (int i = 0; i < lead; ++i) prompt.push_back(1000 + i);
                g6rt::ImageIn im; im.pos = lead; im.span = span;
                for (int i = 0; i < span; ++i) prompt.push_back(T);
                for (int i = 0; i < gap + 1; ++i) prompt.push_back(2000 + i);
                const Ds41ImageRows rows = ds41_image_rows({im});
                const int pos = int(prompt.size()) - 1;   // = lead + span + gap
                int want[E::kRing], got[E::kRing], old[E::kRing];
                live(want, prompt, pos, pos, rows, W);
                ds41_engram_ring(got, prompt, uint32_t(pos), rows);
                raw(old, prompt, pos);
                const bool ok = same(want, got);
                CHECK(ok);
                CHECK(!same(want, old));                                     // the raw rebuild is what the fix replaces
                CHECK(window_differs(want, old, pos) == (gap < 3));          // the n-grams of the next row: only gap < 3 reach the image
                CHECK(!window_differs(want, got, pos));
                if (W == 4096 && span == 1599)
                    std::printf("prompt end, image %d tokens before the snapshot: ring %s the live ring (raw rebuild: next n-gram %s)\n", gap,
                                ok ? "equals" : "DIFFERS from", window_differs(want, old, pos) ? "differs" : "same");
                ++pe_cases; pe_ok += ok;
                // request end (a park, or a request-end entry): the prompt ends gap + 1 text tokens after the image, then
                // generated tokens; the snapshot is at the request end p, the image's end 0..3 tokens before it when the
                // reply is short
                for (int gen = 0; gen <= 3; ++gen) {
                    std::vector<int> h(prompt);
                    for (int i = 0; i < gen; ++i) h.push_back(3000 + i);
                    const int p = pos + gen;   // the request end, as retain/park compute it: at most history.size() - 1
                    int lw[E::kRing], lg[E::kRing];
                    live(lw, h, pos, p, rows, W);
                    ds41_engram_ring(lg, h, uint32_t(p), rows);
                    CHECK(same(lw, lg));
                    ++req_cases; req_ok += same(lw, lg);
                }
            }
        }
    }
    std::printf("prompt end (cache hit): %d of %d rings equal the live ring (W 64/4096, span 4/1599, gap 0..3)\n", pe_ok, pe_cases);
    std::printf("request end (cache hit of the request end, PARK resume): %d of %d rings equal the live ring (+0..3 generated)\n", req_ok, req_cases);
    // two images, the second inside the ring; text-only: the ring is the raw one (no change for text requests)
    {
        std::vector<int> t;
        g6rt::ImageIn a, b; a.pos = 10; a.span = 4; b.pos = 20; b.span = 4;
        for (int i = 0; i < 30; ++i) t.push_back((i >= 10 && i < 14) || (i >= 20 && i < 24) ? T : 500 + i);
        const Ds41ImageRows rows = ds41_image_rows({a, b});
        int want[E::kRing], got[E::kRing];
        live(want, t, 29, 29, rows, 64);
        ds41_engram_ring(got, t, 29, rows);
        CHECK(same(want, got));
        int r0[E::kRing], r1[E::kRing];
        std::vector<int> txt(200);
        for (int i = 0; i < 200; ++i) txt[size_t(i)] = 7 + i;
        raw(r0, txt, 199);
        ds41_engram_ring(r1, txt, 199, {});
        CHECK(same(r0, r1));
        live(want, txt, 199, 199, {}, 64);
        CHECK(same(want, r1));
    }
    if (fails) { std::printf("ds41_engram_ring_test: %d FAILED\n", fails); return 1; }
    std::printf("ds41_engram_ring_test: all passed\n");
    return 0;
}
