// KDA chunked prefill math (block CH = 64 rows), shared by the CUDA kernels (kda_prefill.cuh) and the CPU model
// (tests/kda_cpu.h), in the style of math.h: each phase is one function a thread runs between barriers.
//
// It replaces the step kernel for M rows of ONE slot (prep and out are unchanged): from C (conv + silu), G and the
// b_proj column of P it writes O and leaves S of the slot after the M rows. Algorithm = flash_ref.FlashRef.kda_chunk
// (the transformers chunk_kimi_delta_attention form), per head, chunks of CH rows from row 0 of the call:
//   gc = cumsum_chunk(g);  qn, kn = l2norm(q) * 128^-0.5, l2norm(k)
//   A[i][j] = -beta_i sum_d kn_i kn_j exp(gc_i - gc_j)  (j < i);  B = A solved by forward substitution, Ai = B + I
//   W = Ai (kn beta exp(gc));  U = Ai (v beta);  AQ[i][j] = sum_d qn_i kn_j exp(gc_i - gc_j)  (j <= i)
//   QE = qn exp(gc);  KD = kn exp(gc_last - gc)                                      -- intra kernel, no S, parallel
//   per chunk: VN = U - W S;  O = QE S + AQ VN;  S = S exp(gc_last)[k] + KD^T VN     -- state kernel, sequential
// The pairwise decays exp(gc_i - gc_j) are formed per channel (never exp(gc_i) / exp(gc_j): gc reaches -320 in a
// chunk). The state is fp32 always.
#pragma once
#include "math.h"

namespace kda
{
constexpr int CH = 64;                     // rows per chunk
constexpr int INTRA_T = 256;               // intra kernel threads, CTA (head h, chunk c)
constexpr int ST_T = 256;                  // state kernel threads, CTA (head h, v block vb of VB columns)
constexpr int ST_R = CH * VB / ST_T;       // VN / O cells per state-kernel thread: 8
constexpr int ST_K = DK * VB / ST_T;       // S cells per state-kernel thread: 16

// prefill scratch, [rows][dl] like C / G (head h at column h * DK), AQ [rows][nh][CH]
struct PfArgs
{
    Args a;                             // a.M rows of slot a.row_slot[0]
    float *QE = nullptr, *KD = nullptr, *W = nullptr, *U = nullptr, *GC = nullptr, *AQ = nullptr;
    KDA_HD int nchunk() const { return (a.M + CH - 1) / CH; }
    KDA_HD int rows(int c) const { return a.M - c * CH < CH ? a.M - c * CH : CH; }
    KDA_HD size_t e(int m, int h, int d) const { return static_cast<size_t>(m) * a.dl() + h * DK + d; }
    KDA_HD size_t aq(int m, int h, int j) const { return (static_cast<size_t>(m) * a.nh + h) * CH + j; }
};

// ---------------------------------------------------------------------------------------------- intra kernel
struct IntraSmem
{
    float qd[CH], kd[CH], beta[CH];
    float A[CH][CH + 1], B[CH][CH + 1];
};
// phase 0: q, k norm denominators (same order as step_row_1), beta; cumsum of g per channel (row order)
KDA_HD void intra_0(const PfArgs& p, IntraSmem& s, int h, int c, int t)
{
    const Args& a = p.a;
    const int r0 = c * CH, n = p.rows(c), dl = a.dl();
    if (t < 2 * CH)
    {
        const int i = t % CH;
        if (i < n)
        {
            const float* x = a.C + static_cast<size_t>(r0 + i) * 3 * dl + (t / CH) * dl + h * DK;
            float ss = 0.f;
            for (int j = 0; j < DK; ++j) ss += x[j] * x[j];
            (t < CH ? s.qd : s.kd)[i] = sqrtf(ss + L2_EPS);
        }
    }
    else if (t < 3 * CH)
    {
        const int i = t - 2 * CH;
        if (i < n) s.beta[i] = sigm(a.P[static_cast<size_t>(r0 + i) * a.pl().ld() + a.pl().b() + a.h0 + h]);
    }
    if (t < DK)
    {
        float g = 0.f;
        for (int i = 0; i < n; ++i)
        {
            g += a.G[p.e(r0 + i, h, t)];
            p.GC[p.e(r0 + i, h, t)] = g;
        }
    }
}
// phase 1: normalized q (scaled) into QE, normalized k into KD (finished in phase 4)
KDA_HD void intra_1(const PfArgs& p, const IntraSmem& s, int h, int c, int t)
{
    const Args& a = p.a;
    const int r0 = c * CH, n = p.rows(c), dl = a.dl();
    for (int x = t; x < n * DK; x += INTRA_T)
    {
        const int i = x / DK, d = x % DK, m = r0 + i;
        const float* cr = a.C + static_cast<size_t>(m) * 3 * dl + h * DK + d;
        p.QE[p.e(m, h, d)] = cr[0] / s.qd[i] * QSCALE;
        p.KD[p.e(m, h, d)] = cr[dl] / s.kd[i];
    }
}
// phase 2: A (strict lower) into smem, AQ (lower with diagonal; zero above) into scratch
KDA_HD void intra_2(const PfArgs& p, IntraSmem& s, int h, int c, int t)
{
    const int r0 = c * CH, n = p.rows(c);
    for (int x = t; x < CH * CH; x += INTRA_T)
    {
        const int i = x / CH, j = x % CH;
        float aa = 0.f, aq = 0.f;
        if (j <= i && i < n)
        {
            const float *qi = p.QE + p.e(r0 + i, h, 0), *ki = p.KD + p.e(r0 + i, h, 0), *kj = p.KD + p.e(r0 + j, h, 0);
            const float *gi = p.GC + p.e(r0 + i, h, 0), *gj = p.GC + p.e(r0 + j, h, 0);
            for (int d = 0; d < DK; ++d)
            {
                const float dec = expf(gi[d] - gj[d]);
                aq += qi[d] * kj[d] * dec;
                aa += ki[d] * kj[d] * dec;
            }
            aa = j < i ? -s.beta[i] * aa : 0.f;
        }
        s.A[i][j] = aa;
        if (i < n) p.AQ[p.aq(r0 + i, h, j)] = aq;
    }
}
// phase 3: forward substitution, one thread per column j: B[i][j] = A[i][j] + sum_{j<m<i} A[i][m] B[m][j]
KDA_HD void intra_3(IntraSmem& s, int t)
{
    if (t >= CH) return;
    const int j = t;
    for (int i = 0; i < CH; ++i)
    {
        float b = 0.f;
        if (i > j)
        {
            b = s.A[i][j];
            for (int m = j + 1; m < i; ++m) b += s.A[i][m] * s.B[m][j];
        }
        s.B[i][j] = b;
    }
}
// phase 4: W = Ai (kn beta exp(gc)), U = Ai (v beta), Ai = B + I
KDA_HD void intra_4(const PfArgs& p, const IntraSmem& s, int h, int c, int t)
{
    const Args& a = p.a;
    const int r0 = c * CH, n = p.rows(c), dl = a.dl();
    for (int x = t; x < n * DK; x += INTRA_T)
    {
        const int i = x / DK, d = x % DK;
        float w = 0.f, u = 0.f;
        for (int j = 0; j <= i; ++j)
        {
            const float aij = j == i ? 1.f : s.B[i][j], bj = s.beta[j];
            const size_t ej = p.e(r0 + j, h, d);
            w += aij * (p.KD[ej] * bj * expf(p.GC[ej]));
            u += aij * (a.C[static_cast<size_t>(r0 + j) * 3 * dl + 2 * dl + h * DK + d] * bj);
        }
        p.W[p.e(r0 + i, h, d)] = w;
        p.U[p.e(r0 + i, h, d)] = u;
    }
}
// phase 5: QE = qn exp(gc), KD = kn exp(gc_last - gc) (elementwise, same cells as phase 1)
KDA_HD void intra_5(const PfArgs& p, int h, int c, int t)
{
    const int r0 = c * CH, n = p.rows(c);
    for (int x = t; x < n * DK; x += INTRA_T)
    {
        const int i = x / DK, d = x % DK;
        const size_t ei = p.e(r0 + i, h, d);
        const float g = p.GC[ei], gl = p.GC[p.e(r0 + n - 1, h, d)];
        p.QE[ei] *= expf(g);
        p.KD[ei] *= expf(gl - g);
    }
}

// ---------------------------------------------------------------------------------------------- state kernel
// CTA (head h, v block vb): S columns vb * VB .. + VB of head h in smem for all chunks of the call.
// VN / O cell r of thread t: row i = t / VB + r * (ST_T / VB), column v = t % VB. S cell r: k = (t / VB) * ST_K + r, v.
struct StSmem
{
    float S[DK][VB + 1];
    float vn[CH][VB + 1];
};
KDA_HD float* st_cell(const Args& a, int h, int vb, int k, int v)
{
    return a.S + a.s_off(a.row_slot[0], h) + static_cast<size_t>(k) * DV + vb * VB + v;
}
KDA_HD void st_load(const PfArgs& p, StSmem& s, int h, int vb, int t)
{
    for (int x = t; x < DK * VB; x += ST_T) s.S[x / VB][x % VB] = *st_cell(p.a, h, vb, x / VB, x % VB);
}
// phase A of chunk c: VN = U - W S into smem; the inter part QE S of O into o[]
KDA_HD void st_a(const PfArgs& p, StSmem& s, int h, int vb, int c, int t, float* o)
{
    const int r0 = c * CH, n = p.rows(c), v = t % VB;
    for (int r = 0; r < ST_R; ++r)
    {
        const int i = t / VB + r * (ST_T / VB);
        if (i >= n) { s.vn[i][v] = 0.f; o[r] = 0.f; continue; }
        const float *w = p.W + p.e(r0 + i, h, 0), *qe = p.QE + p.e(r0 + i, h, 0);
        float ws = 0.f, qs = 0.f;
        for (int k = 0; k < DK; ++k)
        {
            ws += w[k] * s.S[k][v];
            qs += qe[k] * s.S[k][v];
        }
        s.vn[i][v] = p.U[p.e(r0 + i, h, vb * VB + v)] - ws;
        o[r] = qs;
    }
}
// phase B of chunk c: O = o + AQ VN (written to a.O); S = S exp(gc_last) + KD^T VN
KDA_HD void st_b(const PfArgs& p, StSmem& s, int h, int vb, int c, int t, const float* o)
{
    const Args& a = p.a;
    const int r0 = c * CH, n = p.rows(c), v = t % VB;
    for (int r = 0; r < ST_R; ++r)
    {
        const int i = t / VB + r * (ST_T / VB);
        if (i >= n) continue;
        const float* aq = p.AQ + p.aq(r0 + i, h, 0);
        float x = o[r];
        for (int j = 0; j <= i; ++j) x += aq[j] * s.vn[j][v];
        a.O[p.e(r0 + i, h, vb * VB + v)] = x;
    }
    for (int r = 0; r < ST_K; ++r)
    {
        const int k = (t / VB) * ST_K + r;
        float x = s.S[k][v] * expf(p.GC[p.e(r0 + n - 1, h, k)]);
        for (int i = 0; i < n; ++i) x += p.KD[p.e(r0 + i, h, k)] * s.vn[i][v];
        s.S[k][v] = x;
    }
}
KDA_HD void st_store(const PfArgs& p, const StSmem& s, int h, int vb, int t)
{
    for (int x = t; x < DK * VB; x += ST_T) *st_cell(p.a, h, vb, x / VB, x % VB) = s.S[x / VB][x % VB];
}
}   // namespace kda
