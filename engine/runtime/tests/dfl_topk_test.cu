// The DFlash2 selector's top-16 ids, CPU only (no GPU, no launch): dfl_topk.h's top_id / walk_id in a host model of
// dflash.cuh's k_top_local (each rank's top-16 of its vocabulary slice: 16 rounds of an argmax under cand_before, the
// taken ids excluded) and of k_walk's merge (the global top-16 of the ranks' sorted lists). cand_before is a strict
// total order on non-NaN logits and a NaN never wins it, so the kernel's block reduction picks what this loop picks.
//   finite rows (random, with ties, NR = 1..8): the ids and logits are the bits of main's (vlo + bi, no guard), no error
//   -inf rows, and a rank whose slice is all NaN beside finite ranks: still main's bits, no error (-inf orders by id)
//   rows with fewer than 16 orderable logits (all NaN, or 10 finite and NaN): main sends an id outside the vocabulary
//   (vlo + 0x7fffffff: the k_walk read at succ + id x SELR x 2 bytes that compute-sanitizer reported); here every id
//   the walk keeps is in the vocabulary and the error flag is set
// build + run: nvcc -O2 -std=c++17 $GEN -o dfl_topk_test engine/runtime/tests/dfl_topk_test.cu && ./dfl_topk_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "../dfl_topk.h"

using namespace g6rt::dfl;
constexpr int TOPK = 16, VOCAB = 4096;   // the test's vocabulary (the walk's bound is glue::VOCAB)

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { if (fails < 10) fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

struct Cand { float logit; int id; };
static bool cand_before(float la, int ia, float lb, int ib) { return la > lb || (la == lb && ia < ib); }
static bool same_bits(float a, float b) { uint32_t i, j; memcpy(&i, &a, 4); memcpy(&j, &b, 4); return i == j; }
static int main_id(int vlo, int bi) { return (int)((unsigned)vlo + (unsigned)bi); }   // main's vlo + bi (wraps as the GPU does)

// k_top_local on one rank's slice row[vlo .. vlo + vn): 16 rounds; guard = this change's top_id, else main's
static std::vector<Cand> top_local(const float* row, int vlo, int vn, bool guard)
{
    std::vector<Cand> out;
    std::vector<int> taken;
    for (int k = 0; k < TOPK; ++k)
    {
        float bl = -INFINITY;
        int bi = TOP_NONE;
        for (int i = 0; i < vn; ++i)
        {
            bool t = false;
            for (int q : taken) t |= q == i;
            if (!t && cand_before(row[vlo + i], i, bl, bi)) { bl = row[vlo + i]; bi = i; }
        }
        taken.push_back(bi);
        out.push_back(Cand{bl, guard ? top_id(vlo, bi) : main_id(vlo, bi)});
    }
    return out;
}
// k_walk's merge of the ranks' lists, then (guard) walk_id on each kept id
static std::vector<Cand> walk_merge(const std::vector<std::vector<Cand>>& src, bool guard, bool& bad)
{
    float bl[TOPK];
    int bi[TOPK];
    int n = 0;
    for (const auto& s : src)
        for (int k = 0; k < TOPK; ++k)
        {
            const Cand x = s[k];
            if (n == TOPK && !cand_before(x.logit, x.id, bl[TOPK - 1], bi[TOPK - 1])) break;
            int at = n < TOPK ? n++ : TOPK - 1;
            while (at > 0 && cand_before(x.logit, x.id, bl[at - 1], bi[at - 1])) { bl[at] = bl[at - 1]; bi[at] = bi[at - 1]; --at; }
            bl[at] = x.logit; bi[at] = x.id;
        }
    std::vector<Cand> out;
    bad = false;
    for (int k = 0; k < TOPK; ++k) out.push_back(Cand{bl[k], guard ? walk_id(bi[k], VOCAB, bad) : bi[k]});
    return out;
}
struct Run { std::vector<Cand> c; bool bad; };
static Run run(const std::vector<float>& row, int nr, bool guard)
{
    std::vector<std::vector<Cand>> src;
    const int vn = VOCAB / nr;
    for (int r = 0; r < nr; ++r) src.push_back(top_local(row.data(), r * vn, vn, guard));
    Run o;
    o.c = walk_merge(src, guard, o.bad);
    return o;
}
static bool in_vocab(const std::vector<Cand>& c) { for (const Cand& x : c) if (x.id < 0 || x.id >= VOCAB) return false; return true; }
static bool same(const Run& a, const Run& b)
{
    for (int k = 0; k < TOPK; ++k) if (a.c[k].id != b.c[k].id || !same_bits(a.c[k].logit, b.c[k].logit)) return false;
    return true;
}

static uint32_t rng = 12345u;
static uint32_t next() { rng = rng * 1664525u + 1013904223u; return rng; }

int main()
{
    const int widths[] = {1, 2, 4, 8};
    int finite_rows = 0;
    for (int nr : widths)
        for (int rep = 0; rep < 64; ++rep)
        {   // finite: random logits, every 8th drawn from 32 values so ties occur
            std::vector<float> row(VOCAB);
            for (int i = 0; i < VOCAB; ++i)
                row[i] = rep % 8 == 0 ? (float)(next() % 32) * .25f : ((float)(next() >> 8) / 16777216.f - .5f) * 40.f;
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(same(a, b));
            EXPECT(!b.bad);
            EXPECT(in_vocab(b.c));
            ++finite_rows;
        }
    for (int nr : widths)
    {
        {   // all -inf: ordered by id, ids 0..15, unchanged, no error
            const std::vector<float> row(VOCAB, -INFINITY);
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(same(a, b) && !b.bad && in_vocab(b.c));
            for (int k = 0; k < TOPK; ++k) EXPECT(b.c[k].id == k);
        }
        if (nr > 1)
        {   // the last rank's slice all NaN, the others finite: the merge drops its entries, unchanged, no error
            std::vector<float> row(VOCAB);
            for (int i = 0; i < VOCAB; ++i) row[i] = i >= VOCAB - VOCAB / nr ? NAN : (float)(next() % 1000) * .01f;
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(same(a, b) && !b.bad && in_vocab(b.c));
        }
        {   // 10 finite, the rest -inf, every 3rd of those NaN: the -inf ids fill the list, unchanged, no error
            std::vector<float> row(VOCAB);
            for (int i = 0; i < VOCAB; ++i) row[i] = i % 3 == 0 ? NAN : -INFINITY;
            for (int j = 0; j < 10; ++j) row[(j * 401 + 7) % VOCAB] = (float)j;
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(same(a, b) && !b.bad && in_vocab(b.c));
        }
        {   // all NaN (the compute-sanitizer case): main sends ids outside the vocabulary; the guard keeps id 0 and flags
            const std::vector<float> row(VOCAB, NAN);
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(!in_vocab(a.c));
            if (nr == 1) EXPECT(a.c[0].id == 0x7fffffff);   // rank 0 (vlo 0): succ + 0x7fffffff x 256 x 2 bytes
            EXPECT(b.bad && in_vocab(b.c));
            for (int k = 0; k < TOPK; ++k) EXPECT(b.c[k].id == 0);
        }
        {   // 10 finite and NaN elsewhere: 6 of the 16 have no logit; the 10 keep their ids, the rest 0, flagged
            std::vector<float> row(VOCAB, NAN);
            for (int j = 0; j < 10; ++j) row[(j * 401 + 7) % VOCAB] = (float)j;
            const Run a = run(row, nr, false), b = run(row, nr, true);
            EXPECT(!in_vocab(a.c));
            EXPECT(b.bad && in_vocab(b.c));
            for (int k = 0; k < 10; ++k) EXPECT(b.c[k].id == a.c[k].id && same_bits(b.c[k].logit, a.c[k].logit));
            for (int k = 10; k < TOPK; ++k) EXPECT(b.c[k].id == 0 && b.c[k].logit == -INFINITY);
        }
    }
    {   // the functions alone
        bool bad = false;
        EXPECT(top_id(1024, 5) == 1029 && top_id(1024, TOP_NONE) == TOP_NONE);
        EXPECT(walk_id(0, VOCAB, bad) == 0 && walk_id(VOCAB - 1, VOCAB, bad) == VOCAB - 1 && !bad);
        EXPECT(walk_id(VOCAB, VOCAB, bad) == 0 && bad);
        bad = false;
        EXPECT(walk_id(-1, VOCAB, bad) == 0 && bad);
        bad = false;
        EXPECT(walk_id(TOP_NONE, VOCAB, bad) == 0 && bad);
    }
    if (fails) { fprintf(stderr, "dfl_topk_test: %d FAILED\n", fails); return 1; }
    printf("dfl_topk_test: PASS (%d finite rows at NR 1/2/4/8: main's ids and logit bits, no error; -inf and a NaN slice: "
           "unchanged; rows with < 16 orderable logits: every id in the vocabulary, error set)\n", finite_rows);
    return 0;
}
