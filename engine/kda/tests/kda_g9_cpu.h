// CPU emulation of the G9 ON path (docs/FLASH-G9-KDA-SCAN.md, CH16 bf16-MMA scan) for ONE head, plus an fp64
// reference of the exact gated delta rule. The ON emulation rounds to bf16 at the doc's rounding points (RNE),
// accumulates in fp32 in a fixed order (MMA sum order is unspecified on the GPU; this is one valid order), keeps the
// state fp32, and flushes exp results / products below FLT_MIN to zero (ex2.approx.ftz).
//
// Inputs of ON per call of M rows (one slot, chunks of 16 from the call's row 0):
//   C   [M][3][128]  q | k | v after conv + silu (fp32, before l2norm), as ksc.C holds them
//   G16 [M][128]     f_b logit rounded to bf16 (caller models gz16: bf16 f_a operand, fp32 accumulate, bf16 out)
//   Z16 [M][128]     g_b logit rounded to bf16
//   braw[M]          b_proj column (fp32); beta = sigm(braw)
//   A_log, dt_bias[128], onorm[128] (bf16 values as float)
// State S [128 d][128 v] fp32, in and out. Outputs: O16 (bf16-valued fp32, pre-gate), y (fp32 gated), OH (fp16 bits).
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>
#include "../math.h"

namespace g9cpu
{
using namespace kda;
constexpr int C16 = 16;

inline float rb(float x) { return bf2f(f2bf_rn(x)); }
inline float ftz(float x) { return std::fabs(x) < FLT_MIN ? 0.f : x; }
inline float h2f(uint16_t h)
{
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(static_cast<float>(m | 1024u), static_cast<int>(e) - 25);
    return s ? -v : v;
}

struct OnIn
{
    int M = 0;
    const float *C = nullptr, *G16 = nullptr, *Z16 = nullptr, *braw = nullptr;
    float A_log = 0.f;
    const float *dt_bias = nullptr, *onorm = nullptr;
};
struct OnOut
{
    float *O16 = nullptr, *y = nullptr;   // [M][128]
    uint16_t* OH = nullptr;               // [M][128] fp16 bits
};

// ON for one call, one head. S [128][128] in/out.
inline void on_call(const OnIn& in, float* S, const OnOut& out)
{
    constexpr int D = 128;
    const int M = in.M;
    const float ea = std::exp(in.A_log);
    // conv16 tail: l2norm in fp32, one bf16 round
    std::vector<float> Qn(static_cast<size_t>(M) * D), Kn(Qn.size()), V(Qn.size());
    for (int m = 0; m < M; ++m)
    {
        const float* c = in.C + static_cast<size_t>(m) * 3 * D;
        float sq = 0.f, sk = 0.f;
        for (int d = 0; d < D; ++d) { sq += c[d] * c[d]; sk += c[D + d] * c[D + d]; }
        const float dq = std::sqrt(sq + L2_EPS), dk = std::sqrt(sk + L2_EPS);
        for (int d = 0; d < D; ++d)
        {
            Qn[static_cast<size_t>(m) * D + d] = rb(c[d] / dq);
            Kn[static_cast<size_t>(m) * D + d] = rb(c[D + d] / dk);
            V[static_cast<size_t>(m) * D + d] = rb(c[2 * D + d]);
        }
    }
    std::vector<float> Sbf(static_cast<size_t>(D) * D);
    static thread_local float gc[C16][D], kdp[C16][D], qdp[C16][D], kip[C16][D], kd[C16][D], qd[C16][D], kr[C16][D];
    for (int c0 = 0; c0 < M; c0 += C16)
    {
        const int n = M - c0 < C16 ? M - c0 : C16;
        float beta[C16] = {};
        for (int i = 0; i < n; ++i) beta[i] = sigm(in.braw[c0 + i]);
        // K1: gate, cumsum (fp32, rows in order), pivot p = gc[7], factorized decays, bf16 operands
        for (int d = 0; d < D; ++d)
        {
            float acc = 0.f;
            for (int i = 0; i < C16; ++i)
            {
                const float g = i < n ? GATE_LO * sigm(ea * (in.G16[static_cast<size_t>(c0 + i) * D + d] + in.dt_bias[d])) : 0.f;
                acc += g;
                gc[i][d] = acc;
            }
            const float gt = gc[C16 - 1][d], p = gc[7][d];
            for (int i = 0; i < C16; ++i)
            {
                const float kn = i < n ? Kn[static_cast<size_t>(c0 + i) * D + d] : 0.f;
                const float qn = (i < n ? Qn[static_cast<size_t>(c0 + i) * D + d] : 0.f) * QSCALE;
                const float g = gc[i][d];
                const float ep = ftz(std::exp(g - p)), en = ftz(std::exp(p - g)), eg = ftz(std::exp(g)), er = ftz(std::exp(gt - g));
                kdp[i][d] = rb(ftz(kn * ep)); qdp[i][d] = rb(ftz(qn * ep)); kip[i][d] = rb(ftz(kn * en));
                kd[i][d] = rb(ftz(kn * eg)); qd[i][d] = rb(ftz(qn * eg)); kr[i][d] = rb(ftz(kn * er));
            }
        }
        float L[C16][C16] = {}, Mq[C16][C16] = {};
        for (int i = 0; i < C16; ++i)
            for (int j = 0; j <= i; ++j)
            {
                float sl = 0.f, sm = 0.f;
                for (int d = 0; d < D; ++d)
                {
                    sl += kdp[i][d] * kip[j][d];
                    sm += qdp[i][d] * kip[j][d];
                }
                if (j < i) L[i][j] = beta[i] * sl;
                Mq[i][j] = rb(sm);
            }
        // INV = (I + L)^-1 by forward substitution (fp32), INVb = INV diag(beta), bf16
        float X[C16][C16] = {}, INVb[C16][C16] = {};
        for (int i = 0; i < C16; ++i)
        {
            for (int j = 0; j < C16; ++j) X[i][j] = i == j ? 1.f : 0.f;
            for (int k = 0; k < i; ++k)
                for (int j = 0; j <= k; ++j) X[i][j] -= L[i][k] * X[k][j];
            for (int j = 0; j <= i; ++j) INVb[i][j] = rb(X[i][j] * beta[j]);
        }
        // K2: state S fp32, bf16 copy for the MMAs
        for (size_t i = 0; i < Sbf.size(); ++i) Sbf[i] = rb(S[i]);
        static thread_local float Xs[C16][D], Ys[C16][D], r[C16][D], u[C16][D];
        for (int i = 0; i < C16; ++i)
        {
            for (int v = 0; v < D; ++v) { Xs[i][v] = 0.f; Ys[i][v] = 0.f; }
            for (int d = 0; d < D; ++d)
            {
                const float a = kd[i][d], b = qd[i][d];
                for (int v = 0; v < D; ++v)
                {
                    Xs[i][v] += a * Sbf[static_cast<size_t>(d) * D + v];
                    Ys[i][v] += b * Sbf[static_cast<size_t>(d) * D + v];
                }
            }
            for (int v = 0; v < D; ++v) r[i][v] = rb((i < n ? V[static_cast<size_t>(c0 + i) * D + v] : 0.f) - Xs[i][v]);
        }
        for (int i = 0; i < C16; ++i)
            for (int v = 0; v < D; ++v)
            {
                float a = 0.f;
                for (int j = 0; j <= i; ++j) a += INVb[i][j] * r[j][v];
                u[i][v] = rb(a);   // U^T rounded to bf16 as MMA operand
            }
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < D; ++v)
            {
                float a = Ys[i][v];
                for (int j = 0; j <= i; ++j) a += Mq[i][j] * u[j][v];
                out.O16[static_cast<size_t>(c0 + i) * D + v] = rb(a);
            }
        for (int d = 0; d < D; ++d)
        {
            const float dec = std::exp(gc[C16 - 1][d]);
            float* Sr = S + static_cast<size_t>(d) * D;
            for (int v = 0; v < D; ++v) Sr[v] *= dec;
            for (int i = 0; i < C16; ++i)
            {
                const float k = kr[i][d];
                for (int v = 0; v < D; ++v) Sr[v] += k * u[i][v];
            }
        }
    }
    // out16: fp32 math, OH fp16
    for (int m = 0; m < M; ++m)
    {
        const float* o = out.O16 + static_cast<size_t>(m) * D;
        float ss = 0.f;
        for (int v = 0; v < D; ++v) ss += o[v] * o[v];
        const float rr = 1.f / std::sqrt(ss / D + RMS_EPS);
        for (int v = 0; v < D; ++v)
        {
            const float y = o[v] * rr * in.onorm[v] * sigm(in.Z16[static_cast<size_t>(m) * D + v]);
            out.y[static_cast<size_t>(m) * D + v] = y;
            out.OH[static_cast<size_t>(m) * D + v] = f2h_rn(y);
        }
    }
}

// exact gated delta rule in fp64, token by token, from the same fp32 inputs: q/k l2norm, gate g, beta in double.
//   S = diag(e^g) S;  u = beta (v - S^T kn);  S += kn u^T;  o = S^T (qn * QSCALE)
// g [M][128] is the (fp32) gate-applied log decay, braw the b_proj logits. S [128][128] double in/out.
inline void ref64_call(int M, const float* C, const float* g, const float* Zraw, const float* braw, const float* onorm,
                       double* S, double* O, double* y)
{
    constexpr int D = 128;
    std::vector<double> qn(D), kn(D), u(D);
    for (int m = 0; m < M; ++m)
    {
        const float* c = C + static_cast<size_t>(m) * 3 * D;
        double sq = 0, sk = 0;
        for (int d = 0; d < D; ++d) { sq += double(c[d]) * c[d]; sk += double(c[D + d]) * c[D + d]; }
        const double dq = std::sqrt(sq + double(L2_EPS)), dk = std::sqrt(sk + double(L2_EPS));
        for (int d = 0; d < D; ++d) { qn[d] = c[d] / dq * double(QSCALE); kn[d] = c[D + d] / dk; }
        const double beta = 1.0 / (1.0 + std::exp(-double(braw[m])));
        for (int d = 0; d < D; ++d)
        {
            const double dec = std::exp(double(g[static_cast<size_t>(m) * D + d]));
            for (int v = 0; v < D; ++v) S[static_cast<size_t>(d) * D + v] *= dec;
        }
        for (int v = 0; v < D; ++v) u[v] = c[2 * D + v];
        for (int d = 0; d < D; ++d)
            for (int v = 0; v < D; ++v) u[v] -= kn[d] * S[static_cast<size_t>(d) * D + v];
        for (int v = 0; v < D; ++v) u[v] *= beta;
        for (int d = 0; d < D; ++d)
            for (int v = 0; v < D; ++v) S[static_cast<size_t>(d) * D + v] += kn[d] * u[v];
        double* o = O + static_cast<size_t>(m) * D;
        for (int v = 0; v < D; ++v) o[v] = 0;
        for (int d = 0; d < D; ++d)
            for (int v = 0; v < D; ++v) o[v] += qn[d] * S[static_cast<size_t>(d) * D + v];
        double ss = 0;
        for (int v = 0; v < D; ++v) ss += o[v] * o[v];
        const double rr = 1.0 / std::sqrt(ss / D + double(RMS_EPS));
        for (int v = 0; v < D; ++v)
            y[static_cast<size_t>(m) * D + v] =
                o[v] * rr * double(onorm[v]) / (1.0 + std::exp(-double(Zraw[static_cast<size_t>(m) * D + v])));
    }
}
}   // namespace g9cpu
