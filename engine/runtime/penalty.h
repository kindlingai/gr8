// Presence / frequency penalties (OpenAI's presence_penalty and frequency_penalty, vLLM's semantics). Before a row
// samples, the logit of every id the request has generated so far drops by presence + frequency * count. The prompt
// does not count. s2/dispatch.h writes this layout before each main step. rank.cuh k_penalty and k_penalty_commit read
// it. The header compiles without CUDA, as s2_in.h does, and s2/tests/penalty_test.cpp runs the kernel's row logic.
//
// Exact with speculative decoding. The row that checks draft j counts the tokens committed so far plus the drafts of
// the rows before it in the same step, as a step alone at its position would. Every penalized logit is one subtraction
// from the raw logit, raw - (presence + frequency * count), whichever way the count was reached. The sample therefore
// does not depend on the draft length.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define G6_PEN_HD __host__ __device__
#else
#define G6_PEN_HD
#endif

namespace g6rt
{

// tokens a session commits between two of its main steps: a verify's K + 1 (at most 8) plus an insert rule's 64
constexpr int PEN_NEW_MAX = 96;

// one row of a main step
struct PenRow
{
    int slot;                     // the session's slot, or -1: the row keeps its logits
    int first;                    // the session's row that samples its next reply token, or the step's row count when none
                                  // does. Rows before it are teacher-forced and keep their logits. Row first + k also
                                  // counts the drafts of rows first + 1 .. first + k (the step's input tokens)
    int commit;                   // 1 on one row of the session: after the head, the session's counts take its new tokens
    float presence, frequency;
};
// one session slot
struct PenSess
{
    int reset;                    // 1: the slot's counts are another request's: read as zero, cleared before the new tokens go in
    int nnew;                     // tokens committed since the counts were last updated
    int news[PEN_NEW_MAX];
};

// what a count of c takes off a logit. The caller skips c = 0, so an id never generated keeps its exact logit
G6_PEN_HD inline float pen_drop(int c, float presence, float frequency) { return presence + frequency * float(c); }

// the pending ids p[0..n) of a row: for p[t], its total count (base plus its occurrences in p), or -1 when p[t] already
// appeared in p[0..t) (that entry writes the logit)
G6_PEN_HD inline int pen_pending_total(const int* p, int n, int t, int base)
{
    for (int i = 0; i < t; ++i)
        if (p[i] == p[t]) return -1;
    int c = base;
    for (int i = t; i < n; ++i) c += p[i] == p[t];
    return c;
}

// The row logic, one thread's share (t, nthreads) of each pass. The kernel runs the three passes with a block barrier
// between them, the host test runs them in order. row: the row's logits of the slice [vlo, vlo + vn). counts: the slot's
// counts of the slice, or null when they read as zero. pend / n: the row's pending ids (new tokens, then the drafts
// before the row). s_raw: n floats, the pending ids' raw logits kept across pass 1.
G6_PEN_HD inline void pen_pass0(const float* row, int vlo, int vn, const int* pend, int n, float* s_raw, int t, int nthreads)
{
    for (int i = t; i < n; i += nthreads)
    {
        const int v = pend[i];
        s_raw[i] = v >= vlo && v < vlo + vn ? row[v - vlo] : 0.f;
    }
}
G6_PEN_HD inline void pen_pass1(float* row, int vn, const int* counts, float presence, float frequency, int t, int nthreads)
{
    if (!counts) return;
    for (int i = t; i < vn; i += nthreads)
    {
        const int c = counts[i];
        if (c > 0) row[i] = row[i] - pen_drop(c, presence, frequency);
    }
}
G6_PEN_HD inline void pen_pass2(float* row, int vlo, int vn, const int* counts, const int* pend, int n, const float* s_raw,
                                float presence, float frequency, int t, int nthreads)
{
    for (int i = t; i < n; i += nthreads)
    {
        const int v = pend[i];
        if (v < vlo || v >= vlo + vn) continue;
        const int c = pen_pending_total(pend, n, i, counts ? counts[v - vlo] : 0);
        if (c > 0) row[v - vlo] = s_raw[i] - pen_drop(c, presence, frequency);
    }
}

}   // namespace g6rt
