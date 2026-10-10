// Top-k logprobs of the head: the order, the cross-rank merge and the rank rule, as plain functions that
// compile for the device (glue.cuh k_head_local / k_head_merge / k_head_merge_tp) and for g++ (engine/tests/test_topk.cpp).
//
// A token's place in a top list: larger logit first, ties to the lower vocab id. Each rank sends its slice's LP_K best
// (logit, id) pairs in that order (unused entries: logit -inf, id TK_PAD_ID, which sort after every real token). The
// merge of nr sorted lists is the global top LP_K: a token of the global top LP_K is in the top LP_K of its own slice.
// lp = logit - mx - log(sm) with (mx, sm) the merged log-sum-exp of the head records (the same value k_head_merge gives
// the sampled token).
#pragma once
#include <climits>
#include <cmath>

#include "../g6mx.h"

#ifdef __CUDACC__
#define TK_HD __host__ __device__ __forceinline__
#else
#define TK_HD inline
#endif

namespace glue {

constexpr int LP_K = 20;                  // largest k of logprobs / prompt_logprobs (PROTOCOL.md max_logprobs)
constexpr int TK_ROWS = G6_MX;            // rows of a decode step (rank.cuh MX)
constexpr int TK_PAD_ID = INT_MAX;

struct TkEnt
{
    float logit;
    int id;
};
static_assert(sizeof(TkEnt) == 8, "TkEnt is 8 bytes on the wire");

// a before b
TK_HD bool tk_before(float la, int ia, float lb, int ib) { return la > lb || (la == lb && ia < ib); }
TK_HD bool tk_before(const TkEnt& a, const TkEnt& b) { return tk_before(a.logit, a.id, b.logit, b.id); }
TK_HD TkEnt tk_pad() { return TkEnt{-INFINITY, TK_PAD_ID}; }

// What one step hands back for the rows that asked for top lists (device buffer, host copy after the step).
struct TkRes
{
    int id[TK_ROWS][LP_K];                // global top LP_K of the row; id -1 and lp -inf where the vocabulary has fewer
    float lp[TK_ROWS][LP_K];
    int rank[TK_ROWS];                    // rank of the row's sampled token (1 = best), -1 = not within the top LP_K
    int rank_t[TK_ROWS];                  // same for the teacher-forced target token
};

// Merge nr lists of LP_K entries, each sorted by tk_before, into out[LP_K]. get(p, i) -> TkEnt.
template <class Get>
TK_HD void tk_merge(int nr, Get get, TkEnt* out)
{
    int at[8];
    for (int p = 0; p < nr; ++p) at[p] = 0;
    for (int k = 0; k < LP_K; ++k)
    {
        int bp = -1;
        TkEnt best = tk_pad();
        for (int p = 0; p < nr; ++p)
        {
            if (at[p] >= LP_K) continue;
            const TkEnt e = get(p, at[p]);
            if (bp < 0 || tk_before(e, best)) { best = e; bp = p; }
        }
        out[k] = best;
        if (bp >= 0) ++at[bp];
    }
}

// Rank (vLLM's: 1 + the number of tokens with a larger logit) of a token with logit L, from the merged list; -1 when
// L is below the list's last entry (the rank is above LP_K and not known).
TK_HD int tk_rank(const TkEnt* top, float L)
{
    const TkEnt last = top[LP_K - 1];
    if (last.id != TK_PAD_ID && L < last.logit) return -1;
    int r = 1;
    for (int i = 0; i < LP_K; ++i) r += top[i].id != TK_PAD_ID && top[i].logit > L;
    return r;
}

// log-sum-exp pairs of two vocabulary slices (mx = max logit, sm = sum exp(logit - mx)), the combine of the head kernels
TK_HD void lse_combine(float& mx, float& sm, float omx, float osm)
{
    const float nmx = fmaxf(mx, omx);
    sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (omx == -INFINITY ? 0.f : osm * expf(omx - nmx));
    mx = nmx;
}

// out[row] for a merged list: ids, logprobs, ranks of the sampled logit ls and the target logit lt (-inf = none)
TK_HD void tk_result(TkRes* o, int m, const TkEnt* top, float mx, float sm, float ls, float lt)
{
    for (int i = 0; i < LP_K; ++i)
    {
        const bool ok = top[i].id != TK_PAD_ID;
        o->id[m][i] = ok ? top[i].id : -1;
        o->lp[m][i] = ok ? top[i].logit - mx - logf(sm) : -INFINITY;
    }
    o->rank[m] = tk_rank(top, ls);
    o->rank_t[m] = lt == -INFINITY ? -1 : tk_rank(top, lt);
}

}   // namespace glue
