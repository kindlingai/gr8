// CPU checks of engine/kernels/topk.cuh, no GPU:
//   1. log-sum-exp: the per-slice (max, sum) pairs combined as k_head_merge does vs a float64 reference, for the TP4 and
//      TP6 vocabulary splits (unequal slice widths); logprob of every top-20 token vs the float64 value
//   2. top list: per-slice select (LP_K rounds of "best entry strictly after the previous pick", as head_local_topk)
//      + tk_merge vs a brute-force sort of the whole row; slices narrower than LP_K are padded
//   3. tk_rank vs the brute-force rank (1 + number of larger logits), -1 beyond the top LP_K
// build and run:  g++ -O2 -std=c++17 -I engine/kernels engine/tests/test_topk.cpp -o /tmp/test_topk && /tmp/test_topk
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include "topk.cuh"

using namespace glue;

static int fails = 0, checks = 0;
static void check(bool ok, const char* what, const char* info)
{
    ++checks;
    if (!ok) { ++fails; printf("FAIL %s  [%s]\n", what, info); }
}

constexpr int V = 154880;

// slice boundaries: nr slices of unequal width (multiples of 64, the last takes the rest)
static std::vector<int> split(int nr, int skew)
{
    std::vector<int> lo(nr + 1, 0);
    const int base = (V / nr) / 64 * 64;
    for (int p = 1; p < nr; ++p) lo[p] = p * base + (p % 2 ? skew : -skew) * 64;
    lo[nr] = V;
    return lo;
}

// one slice: fp32 (mx, sm) as the kernel (1024 lanes, strided, then summed), top LP_K by strictly-after rounds
static void slice_local(const float* row, int lo, int hi, float& mx, float& sm, TkEnt* top)
{
    mx = -INFINITY;
    for (int i = lo; i < hi; ++i) mx = fmaxf(mx, row[i]);
    float lane[1024];
    for (int t = 0; t < 1024; ++t)
    {
        float s = 0.f;
        for (int i = lo + t; i < hi; i += 1024) s += row[i] == -INFINITY ? 0.f : expf(row[i] - mx);
        lane[t] = s;
    }
    sm = 0.f;
    for (int t = 0; t < 1024; ++t) sm += lane[t];
    float pl = INFINITY;
    int pi = -1;
    for (int k = 0; k < LP_K; ++k)
    {
        float bl = -INFINITY;
        int bi = TK_PAD_ID;
        for (int i = lo; i < hi; ++i)
            if (tk_before(pl, pi, row[i], i) && tk_before(row[i], i, bl, bi)) { bl = row[i]; bi = i; }
        top[k] = TkEnt{bl, bi};
        pl = bl; pi = bi;
    }
}

static void run(const char* name, const std::vector<float>& row, int nr, int skew)
{
    char info[200];
    const std::vector<int> lo = split(nr, skew);
    std::vector<float> mxs(nr), sms(nr);
    std::vector<TkEnt> tops(nr * LP_K);
    for (int p = 0; p < nr; ++p) slice_local(row.data(), lo[p], lo[p + 1], mxs[p], sms[p], &tops[p * LP_K]);
    float mx = -INFINITY, sm = 0.f;
    for (int p = 0; p < nr; ++p) lse_combine(mx, sm, mxs[p], sms[p]);
    TkEnt top[LP_K];
    tk_merge(nr, [&](int p, int i) { return tops[p * LP_K + i]; }, top);

    // float64 reference
    double rmx = -INFINITY;
    for (float x : row) rmx = std::max(rmx, (double)x);
    double rs = 0;
    for (float x : row) rs += x == -INFINITY ? 0.0 : std::exp((double)x - rmx);
    const double rlse = rmx + std::log(rs), lse = (double)mx + std::log((double)sm);
    snprintf(info, sizeof info, "%s nr=%d lse %.9g ref %.9g", name, nr, lse, rlse);
    check(std::fabs(lse - rlse) < 2e-5 * std::max(1.0, std::fabs(rlse)), "lse vs float64", info);

    // brute-force order
    std::vector<int> ord(V);
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::partial_sort(ord.begin(), ord.begin() + LP_K, ord.end(), [&](int a, int b) { return tk_before(row[a], a, row[b], b); });
    bool same = true;
    double worst = 0;
    TkRes res;
    tk_result(&res, 0, top, mx, sm, row[ord[0]], -INFINITY);
    for (int k = 0; k < LP_K; ++k)
    {
        same &= top[k].id == ord[k] && top[k].logit == row[ord[k]] && res.id[0][k] == ord[k];
        const double ref = (double)row[ord[k]] - rlse;
        if (row[ord[k]] != -INFINITY) worst = std::max(worst, std::fabs((double)res.lp[0][k] - ref));
    }
    snprintf(info, sizeof info, "%s nr=%d", name, nr);
    check(same, "merged top list == brute-force top list", info);
    snprintf(info, sizeof info, "%s nr=%d max |lp - ref| %.3g", name, nr, worst);
    check(worst < 2e-5, "top-20 logprobs vs float64", info);
    printf("  %-12s nr=%d  lse err %.2e  top-20 lp err %.2e\n", name, nr, std::fabs(lse - rlse), worst);

    // ranks: tokens of the top list, a few beyond it, ties
    std::mt19937 g(7);
    std::vector<int> ids(ord.begin(), ord.begin() + LP_K);
    for (int k = 0; k < 40; ++k) ids.push_back((int)(g() % V));
    int bad = 0, beyond = 0;
    for (int id : ids)
    {
        int larger = 0;
        for (float x : row) larger += x > row[id];
        const int want = larger + 1 <= LP_K ? larger + 1 : -1;
        beyond += want < 0;
        bad += tk_rank(top, row[id]) != want;
    }
    snprintf(info, sizeof info, "%s nr=%d %d of %zu differ (%d beyond top 20)", name, nr, bad, ids.size(), beyond);
    check(bad == 0, "tk_rank vs brute force", info);
}

int main()
{
    std::mt19937 g(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](float scale, float off) {
        std::vector<float> r(V);
        for (float& x : r) x = off + scale * nd(g);
        return r;
    };
    std::vector<std::pair<const char*, std::vector<float>>> rows;
    rows.push_back({"normal", fill(2.f, 0.f)});
    rows.push_back({"peaked", [&] { auto r = fill(1.f, -5.f); r[777] = 30.f; r[91234] = 29.5f; return r; }()});
    rows.push_back({"wide", fill(12.f, 3.f)});
    rows.push_back({"ties", [&] { auto r = fill(1.f, 0.f); for (float& x : r) x = std::floor(x * 2.f) / 2.f; return r; }()});
    rows.push_back({"all-equal", std::vector<float>(V, 1.5f)});
    rows.push_back({"neg-inf", [&] { auto r = fill(2.f, 0.f); for (int i = 0; i < V; ++i) if (i % 3 != 0) r[i] = -INFINITY; return r; }()});
    rows.push_back({"sparse", [&] { std::vector<float> r(V, -INFINITY); for (int i : {5, 6, 70000, 154879, 100}) r[i] = (float)i * 1e-4f; return r; }()});
    rows.push_back({"big-offset", fill(1.f, 80.f)});
    for (auto& [name, row] : rows)
        for (int nr : {4, 6}) run(name, row, nr, nr == 4 ? 3 : 5);

    // slices narrower than LP_K are padded: 4 slices of width 7 + one holding the rest
    {
        std::vector<float> row = fill(2.f, 0.f);
        const int cut[6] = {0, 7, 14, 21, 28, V};
        std::vector<TkEnt> tops(5 * LP_K);
        for (int p = 0; p < 5; ++p) { float a, b; slice_local(row.data(), cut[p], cut[p + 1], a, b, &tops[p * LP_K]); }
        int pads = 0;
        for (int k = 0; k < LP_K; ++k) pads += tops[k].id == TK_PAD_ID;
        TkEnt top[LP_K];
        tk_merge(5, [&](int p, int i) { return tops[p * LP_K + i]; }, top);
        std::vector<int> ord(V);
        for (int i = 0; i < V; ++i) ord[i] = i;
        std::partial_sort(ord.begin(), ord.begin() + LP_K, ord.end(), [&](int a, int b) { return tk_before(row[a], a, row[b], b); });
        bool same = true;
        for (int k = 0; k < LP_K; ++k) same &= top[k].id == ord[k];
        check(pads == LP_K - 7 && same, "narrow slices are padded, merge unaffected", "5 slices, widths 7,7,7,7,rest");
        // a vocabulary smaller than LP_K: pad entries come out as id -1, lp -inf, rank by count
        TkEnt small[LP_K];
        for (int k = 0; k < LP_K; ++k) small[k] = k < 3 ? TkEnt{3.f - k, k} : tk_pad();
        TkRes r;
        tk_result(&r, 2, small, 3.f, 1.5f, 2.f, 1.f);
        check(r.id[2][3] == -1 && r.lp[2][3] == -INFINITY && r.rank[2] == 2 && r.rank_t[2] == 3, "pad entries", "3-token vocabulary");
        tk_result(&r, 2, small, 3.f, 1.5f, 2.f, -INFINITY);
        check(r.rank_t[2] == -1, "no target -> rank_t -1", "");
    }
    printf("%d checks, %d failed\n%s\n", checks, fails, fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
