// W4: KDA chunked prefill (block 64, fp32 state), the prompt path of the KDA block. Math and phases: prefill.h.
//
//   kda::build_prefill_scratch(ps, rows, nr)        build: QE, KD, W, U, GC [rows][nhmax*128], AQ [rows][nhmax][64]
//   kda::launch_prefill(w, state, scratch, ps, xb, row_slot, M, partial, st)
//       as kda::launch, for M rows of ONE slot in position order (row_slot[0] .. all equal; scratch sc and ps of at
//       least M rows): projections -> prep -> intra (CTA per head and chunk) -> state (CTA per head and v block,
//       chunks in order) -> out -> o_proj. Projections and o_proj run in launches of g6mx::MX rows (the decode GEMVs).
//   kda::launch_core_prefill(p, st)                 prep, intra, state, out on projections already in P (tests)
#pragma once
#include "kda.cuh"
#include "prefill.h"

namespace kda
{
struct KdaPrefillScratch
{
    float *QE = nullptr, *KD = nullptr, *W = nullptr, *U = nullptr, *GC = nullptr, *AQ = nullptr;
    int rows = 0;
};

inline const char* build_prefill_scratch(KdaPrefillScratch& s, int rows, int nr)
{
    s = KdaPrefillScratch();
    s.rows = rows;
    const size_t n = static_cast<size_t>(rows) * nhmax(nr) * DK, naq = static_cast<size_t>(rows) * nhmax(nr) * CH;
    float** f[] = {&s.QE, &s.KD, &s.W, &s.U, &s.GC, &s.AQ};
    for (int i = 0; i < 6; ++i)
        if (cudaMalloc(f[i], (i == 5 ? naq : n) * sizeof(float)) != cudaSuccess) return "KDA prefill scratch: cudaMalloc";
    return nullptr;
}

// ---------------------------------------------------------------------------------------------- row-parallel prep
// The prep of M rows of ONE slot (row_slot[0]) without the row loop of k_kda_prep, same sums in the same order:
//   k_kda_conv_pf     C = silu(conv) per (row, stream, channel), CTA (channel block, row); rows < 3 take the slot's
//                     conv history
//   k_kda_gz_pf       G, Z = f_b (f_a x), g_b (g_a x) as a 64 x 64 tiled product over K = 128, CTA (channel tile,
//                     row tile, mat); each output sums j = 0 .. 127 in order like prep_row_b
//   k_kda_conv_end_pf the slot's conv history = the last 3 rows (after k_kda_conv_pf has read the old one)
constexpr int GZ_T = 256, GZ_TILE = 64, GZ_KC = 64;
__global__ void __launch_bounds__(256) k_kda_conv_pf(Args a)
{
    const int dl = a.dl(), slot = a.row_slot[0], m = blockIdx.y, rem = blockIdx.x * blockDim.x + threadIdx.x;
    if (rem >= 3 * dl) return;
    const int stream = rem / dl, ch = rem % dl;
    const int ld = a.pl().ld();
    const uint16_t* w = a.conv_w[stream] + static_cast<size_t>(ch) * CONV;
    auto x = [&](int r) { return r >= 0 ? a.P[static_cast<size_t>(r) * ld + rem] : a.cst[a.c_off(slot, NHIST + r, stream, ch)]; };
    float acc = x(m - NHIST) * bf2f(w[0]);
    for (int j = 1; j < NHIST; ++j) acc += x(m - NHIST + j) * bf2f(w[j]);
    acc += x(m) * bf2f(w[NHIST]);
    a.C[static_cast<size_t>(m) * 3 * dl + rem] = silu(acc);
}
__global__ void __launch_bounds__(256) k_kda_conv_end_pf(Args a)
{
    const int dl = a.dl(), slot = a.row_slot[0], rem = blockIdx.x * blockDim.x + threadIdx.x;
    if (rem >= 3 * dl) return;
    const int stream = rem / dl, ch = rem % dl, ld = a.pl().ld();
    float h[NHIST];
    for (int j = 0; j < NHIST; ++j)
    {
        const int r = a.M - NHIST + j;
        h[j] = r >= 0 ? a.P[static_cast<size_t>(r) * ld + rem] : a.cst[a.c_off(slot, NHIST + r, stream, ch)];
    }
    for (int j = 0; j < NHIST; ++j) a.cst[a.c_off(slot, j, stream, ch)] = h[j];
}
__global__ void __launch_bounds__(GZ_T) k_kda_gz_pf(Args a)
{
    __shared__ float xs[GZ_TILE][GZ_KC + 1];
    __shared__ float ws[GZ_KC][GZ_TILE + 1];
    const int c0 = blockIdx.x * GZ_TILE, m0 = blockIdx.y * GZ_TILE, mat = blockIdx.z, t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int dl = a.dl(), ld = a.pl().ld(), col = mat ? a.pl().ga() : a.pl().fa();
    const uint16_t* wsrc = mat ? a.gb : a.fb;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < DK; k0 += GZ_KC)
    {
        for (int i = t; i < GZ_TILE * GZ_KC; i += GZ_T)
        {
            const int r = i / GZ_KC, k = i % GZ_KC;
            xs[r][k] = m0 + r < a.M ? a.P[static_cast<size_t>(m0 + r) * ld + col + k0 + k] : 0.f;
            ws[k][r] = bf2f(wsrc[static_cast<size_t>(c0 + r) * DK + k0 + k]);
        }
        __syncthreads();
        for (int k = 0; k < GZ_KC; ++k)
        {
            float xv[4], wv[4];
            for (int r = 0; r < 4; ++r) xv[r] = xs[ty + 16 * r][k];
            for (int c = 0; c < 4; ++c) wv[c] = ws[k][tx + 16 * c];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) acc[r][c] += wv[c] * xv[r];
        }
        __syncthreads();
    }
    for (int r = 0; r < 4; ++r)
    {
        const int m = m0 + ty + 16 * r;
        if (m >= a.M) continue;
        for (int c = 0; c < 4; ++c)
        {
            const int ch = c0 + tx + 16 * c, h = ch / DK;
            const size_t o = static_cast<size_t>(m) * dl + ch;
            if (mat == 0) a.G[o] = GATE_LO * sigm(expf(a.A_log[h]) * (acc[r][c] + a.dt_bias[ch]));
            else a.Z[o] = acc[r][c];
        }
    }
}
inline void launch_prep_pf(const Args& a, cudaStream_t st)
{
    k_kda_conv_pf<<<dim3((3 * a.dl() + 255) / 256, a.M), 256, 0, st>>>(a);
    k_kda_gz_pf<<<dim3(a.dl() / GZ_TILE, (a.M + GZ_TILE - 1) / GZ_TILE, 2), GZ_T, 0, st>>>(a);
    k_kda_conv_end_pf<<<(3 * a.dl() + 255) / 256, 256, 0, st>>>(a);
}

__global__ void __launch_bounds__(INTRA_T) k_kda_intra(PfArgs p)
{
    __shared__ IntraSmem s;
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x;
    intra_0(p, s, h, c, t);
    __syncthreads();
    intra_1(p, s, h, c, t);
    __syncthreads();
    intra_2(p, s, h, c, t);
    __syncthreads();
    intra_3(s, t);
    __syncthreads();
    intra_4(p, s, h, c, t);
    __syncthreads();
    intra_5(p, h, c, t);
}

// Intra kernel v2, same CTA (head h, chunk c) and the same sums in the same order as intra_0 .. intra_5, with operands
// staged through shared memory:
//   0  intra_0 (norms, beta, cumsum -> GC)
//   1  A / AQ in d tiles of I2_DT: q, k, gc staged [d][row]; thread t < 136 owns lower 4 x 4 tile t, t >= 136 writes
//      the zero upper tile t - 136; A (times -beta, strict lower) into smem M, AQ to global
//   2  forward substitution in place on M, one row per step (the row's A is read before any of it is overwritten)
//   3  W, U = Ai (kn beta exp(gc)), Ai (v beta) in d tiles; thread rows ty, ty + 32, d 4 tx .. + 4
//   4  QE = qn exp(gc), KD = kn exp(gc_last - gc)
constexpr int I2_DT = 32;
struct Intra2Smem
{
    float qd[CH], kd[CH], beta[CH];
    float M[CH][CH + 1];
    union
    {
        float4 a[3][I2_DT][CH / 4];          // phase 1: q, k, gc  [d][row]
        float4 w[2][CH][I2_DT / 4];          // phase 3: kn beta exp(gc), v beta  [row][d]
    } u;
};
KDA_HD void i2_tile(int x, bool upper, int& ti, int& tj)   // x-th lower (tj <= ti) or strict upper (tj > ti) 4 x 4 tile
{
    int r = upper ? 1 : 0, base = 0;
    while (base + r + (upper ? 0 : 1) <= x) { base += r + (upper ? 0 : 1); ++r; }
    const int a = r, b = x - base;      // lower: row a, col b <= a; upper: rows b < a
    if (upper) { ti = b; tj = a; }
    else { ti = a; tj = b; }
}
__global__ void __launch_bounds__(INTRA_T, 2) k_kda_intra2(PfArgs p)
{
    __shared__ Intra2Smem s;
    const Args& a = p.a;
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x, r0 = c * CH, n = p.rows(c), dl = a.dl();
    // 0 (intra_0)
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
    __syncthreads();
    // 1
    constexpr int NLOW = 136;
    int ti, tj;
    i2_tile(t < NLOW ? t : t - NLOW, t >= NLOW, ti, tj);
    float aa[4][4] = {}, aq[4][4] = {};
    float* af = reinterpret_cast<float*>(s.u.a);
    for (int d0 = 0; d0 < DK; d0 += I2_DT)
    {
        for (int x = t; x < I2_DT * CH; x += INTRA_T)
        {
            const int i = x / I2_DT, d = x % I2_DT;
            float q = 0.f, k = 0.f, g = 0.f;
            if (i < n)
            {
                const float* cr = a.C + static_cast<size_t>(r0 + i) * 3 * dl + h * DK + d0 + d;
                q = cr[0] / s.qd[i] * QSCALE;
                k = cr[dl] / s.kd[i];
                g = p.GC[p.e(r0 + i, h, d0 + d)];
            }
            af[(0 * I2_DT + d) * CH + i] = q;
            af[(1 * I2_DT + d) * CH + i] = k;
            af[(2 * I2_DT + d) * CH + i] = g;
        }
        __syncthreads();
        if (t < NLOW)
        {
            for (int d = 0; d < I2_DT; ++d)
            {
                const float4 q4 = s.u.a[0][d][ti], ki4 = s.u.a[1][d][ti], gi4 = s.u.a[2][d][ti];
                const float4 kj4 = s.u.a[1][d][tj], gj4 = s.u.a[2][d][tj];
                const float qi[4] = {q4.x, q4.y, q4.z, q4.w}, ki[4] = {ki4.x, ki4.y, ki4.z, ki4.w};
                const float gi[4] = {gi4.x, gi4.y, gi4.z, gi4.w}, kj[4] = {kj4.x, kj4.y, kj4.z, kj4.w};
                const float gj[4] = {gj4.x, gj4.y, gj4.z, gj4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y)
                    {
                        if (ti == tj && y > x) continue;
                        const float dec = expf(gi[x] - gj[y]);
                        aq[x][y] += qi[x] * kj[y] * dec;
                        aa[x][y] += ki[x] * kj[y] * dec;
                    }
            }
        }
        __syncthreads();
    }
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
        {
            const int i = 4 * ti + x, j = 4 * tj + y;
            const bool on = t < NLOW && j <= i && i < n;
            s.M[i][j] = on && j < i ? -s.beta[i] * aa[x][y] : 0.f;
            if (i < n) p.AQ[p.aq(r0 + i, h, j)] = on ? aq[x][y] : 0.f;
        }
    __syncthreads();
    // 2: B[i][j] = A[i][j] + sum_{j<m<i} A[i][m] B[m][j]; rows below i already hold B
    for (int i = 2; i < CH; ++i)
    {
        float b = 0.f;
        const int j = t;
        if (j < i - 1)
        {
            b = s.M[i][j];
            for (int m = j + 1; m < i; ++m) b += s.M[i][m] * s.M[m][j];
        }
        __syncthreads();
        if (j < i - 1) s.M[i][j] = b;
        __syncthreads();
    }
    // 3
    const int tx = t % 8, ty = t / 8;
    for (int d0 = 0; d0 < DK; d0 += I2_DT)
    {
        for (int x = t; x < CH * I2_DT; x += INTRA_T)
        {
            const int j = x / I2_DT, d = x % I2_DT;
            float kb = 0.f, vb = 0.f;
            if (j < n)
            {
                const float* cr = a.C + static_cast<size_t>(r0 + j) * 3 * dl + h * DK + d0 + d;
                const float bj = s.beta[j];
                kb = cr[dl] / s.kd[j] * bj * expf(p.GC[p.e(r0 + j, h, d0 + d)]);
                vb = cr[2 * dl] * bj;
            }
            reinterpret_cast<float*>(s.u.w[0])[j * I2_DT + d] = kb;
            reinterpret_cast<float*>(s.u.w[1])[j * I2_DT + d] = vb;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int i = ty + 32 * r;
            if (i >= n) continue;
            float w[4] = {}, u[4] = {};
            for (int j = 0; j <= i; ++j)
            {
                const float aij = j == i ? 1.f : s.M[i][j];
                const float4 kb = s.u.w[0][j][tx], vb = s.u.w[1][j][tx];
                w[0] += aij * kb.x; w[1] += aij * kb.y; w[2] += aij * kb.z; w[3] += aij * kb.w;
                u[0] += aij * vb.x; u[1] += aij * vb.y; u[2] += aij * vb.z; u[3] += aij * vb.w;
            }
            *reinterpret_cast<float4*>(p.W + p.e(r0 + i, h, d0 + 4 * tx)) = make_float4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<float4*>(p.U + p.e(r0 + i, h, d0 + 4 * tx)) = make_float4(u[0], u[1], u[2], u[3]);
        }
        __syncthreads();
    }
    // 4
    for (int x = t; x < n * DK; x += INTRA_T)
    {
        const int i = x / DK, d = x % DK;
        const size_t ei = p.e(r0 + i, h, d);
        const float* cr = a.C + static_cast<size_t>(r0 + i) * 3 * dl + h * DK + d;
        const float g = p.GC[ei], gl = p.GC[p.e(r0 + n - 1, h, d)];
        p.QE[ei] = cr[0] / s.qd[i] * QSCALE * expf(g);
        p.KD[ei] = cr[dl] / s.kd[i] * expf(gl - g);
    }
}

// Intra kernel v3: v2 with two passes over d tiles instead of five sweeps of the chunk:
//   1  per d tile: stage raw q, k, g [d][row]; cumsum g down the rows (-> GC), sums of squares of q, k (the same
//      per-row order as intra_0), raw A / AQ dots with the decay; after the last tile scale by the norms
//   2  forward substitution as v2
//   3  per d tile: stage kn beta exp(gc), v beta; write QE, KD of the tile; W, U as v2
constexpr int I3_DT = 32;
struct Intra3Smem
{
    float qss[CH], kss[CH], beta[CH], gl[DK];
    float M[CH][CH + 1];
    union
    {
        float4 a[3][I3_DT][CH / 4];          // pass 1: raw q, k, then g -> gc  [d][row]
        float4 w[2][CH][I3_DT / 4];          // pass 3: kn beta exp(gc), v beta  [row][d]
    } u;
};
__global__ void __launch_bounds__(INTRA_T, 2) k_kda_intra3(PfArgs p)
{
    __shared__ Intra3Smem s;
    const Args& a = p.a;
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x, r0 = c * CH, n = p.rows(c), dl = a.dl();
    if (t < CH)
    {
        s.qss[t] = 0.f; s.kss[t] = 0.f;
        s.beta[t] = t < n ? sigm(a.P[static_cast<size_t>(r0 + t) * a.pl().ld() + a.pl().b() + a.h0 + h]) : 0.f;
    }
    constexpr int NLOW = 136;
    int ti, tj;
    i2_tile(t < NLOW ? t : t - NLOW, t >= NLOW, ti, tj);
    float aa[4][4] = {}, aq[4][4] = {};
    float* af = reinterpret_cast<float*>(s.u.a);
    for (int d0 = 0; d0 < DK; d0 += I3_DT)
    {
        for (int x = t; x < I3_DT * CH; x += INTRA_T)
        {
            const int i = x / I3_DT, d = x % I3_DT;
            float q = 0.f, k = 0.f, g = 0.f;
            if (i < n)
            {
                const float* cr = a.C + static_cast<size_t>(r0 + i) * 3 * dl + h * DK + d0 + d;
                q = cr[0];
                k = cr[dl];
                g = a.G[p.e(r0 + i, h, d0 + d)];
            }
            af[(0 * I3_DT + d) * CH + i] = q;
            af[(1 * I3_DT + d) * CH + i] = k;
            af[(2 * I3_DT + d) * CH + i] = g;
        }
        __syncthreads();
        if (t < I3_DT)
        {   // cumsum of channel d0 + t down the rows
            float g = 0.f;
            for (int i = 0; i < n; ++i)
            {
                g += af[(2 * I3_DT + t) * CH + i];
                af[(2 * I3_DT + t) * CH + i] = g;
                p.GC[p.e(r0 + i, h, d0 + t)] = g;
            }
            s.gl[d0 + t] = g;
        }
        else if (t >= 64 && t < 64 + 2 * CH)
        {   // sums of squares, d in order
            const int i = (t - 64) % CH, mat = (t - 64) / CH;
            float ss = mat ? s.kss[i] : s.qss[i];
            for (int d = 0; d < I3_DT; ++d)
            {
                const float x = af[(mat * I3_DT + d) * CH + i];
                ss += x * x;
            }
            (mat ? s.kss : s.qss)[i] = ss;
        }
        __syncthreads();
        if (t < NLOW)
        {
            for (int d = 0; d < I3_DT; ++d)
            {
                const float4 q4 = s.u.a[0][d][ti], ki4 = s.u.a[1][d][ti], gi4 = s.u.a[2][d][ti];
                const float4 kj4 = s.u.a[1][d][tj], gj4 = s.u.a[2][d][tj];
                const float qi[4] = {q4.x, q4.y, q4.z, q4.w}, ki[4] = {ki4.x, ki4.y, ki4.z, ki4.w};
                const float gi[4] = {gi4.x, gi4.y, gi4.z, gi4.w}, kj[4] = {kj4.x, kj4.y, kj4.z, kj4.w};
                const float gj[4] = {gj4.x, gj4.y, gj4.z, gj4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y)
                    {
                        if (ti == tj && y > x) continue;
                        const float kd = kj[y] * expf(gi[x] - gj[y]);
                        aq[x][y] += qi[x] * kd;
                        aa[x][y] += ki[x] * kd;
                    }
            }
        }
        __syncthreads();
    }
    if (t < CH)
    {   // sums of squares -> norm denominators (in place)
        s.qss[t] = sqrtf(s.qss[t] + L2_EPS);
        s.kss[t] = sqrtf(s.kss[t] + L2_EPS);
    }
    __syncthreads();
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
        {
            const int i = 4 * ti + x, j = 4 * tj + y;
            const bool on = t < NLOW && j <= i && i < n;
            const float kij = on ? s.kss[i] * s.kss[j] : 1.f;
            s.M[i][j] = on && j < i ? -s.beta[i] * (aa[x][y] / kij) : 0.f;
            if (i < n) p.AQ[p.aq(r0 + i, h, j)] = on ? aq[x][y] / (s.qss[i] * s.kss[j]) * QSCALE : 0.f;
        }
    __syncthreads();
    // 2
    for (int i = 2; i < CH; ++i)
    {
        float b = 0.f;
        const int j = t;
        if (j < i - 1)
        {
            b = s.M[i][j];
            for (int m = j + 1; m < i; ++m) b += s.M[i][m] * s.M[m][j];
        }
        __syncthreads();
        if (j < i - 1) s.M[i][j] = b;
        __syncthreads();
    }
    // 3
    const int tx = t % 8, ty = t / 8;
    for (int d0 = 0; d0 < DK; d0 += I3_DT)
    {
        for (int x = t; x < CH * I3_DT; x += INTRA_T)
        {
            const int j = x / I3_DT, d = x % I3_DT;
            float kb = 0.f, vb = 0.f;
            if (j < n)
            {
                const float* cr = a.C + static_cast<size_t>(r0 + j) * 3 * dl + h * DK + d0 + d;
                const size_t ej = p.e(r0 + j, h, d0 + d);
                const float bj = s.beta[j], g = p.GC[ej], kn = cr[dl] / s.kss[j];
                kb = kn * bj * expf(g);
                vb = cr[2 * dl] * bj;
                p.QE[ej] = cr[0] / s.qss[j] * QSCALE * expf(g);
                p.KD[ej] = kn * expf(s.gl[d0 + d] - g);
            }
            reinterpret_cast<float*>(s.u.w[0])[j * I3_DT + d] = kb;
            reinterpret_cast<float*>(s.u.w[1])[j * I3_DT + d] = vb;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int i = ty + 32 * r;
            if (i >= n) continue;
            float w[4] = {}, u[4] = {};
            for (int j = 0; j <= i; ++j)
            {
                const float aij = j == i ? 1.f : s.M[i][j];
                const float4 kb = s.u.w[0][j][tx], vb = s.u.w[1][j][tx];
                w[0] += aij * kb.x; w[1] += aij * kb.y; w[2] += aij * kb.z; w[3] += aij * kb.w;
                u[0] += aij * vb.x; u[1] += aij * vb.y; u[2] += aij * vb.z; u[3] += aij * vb.w;
            }
            *reinterpret_cast<float4*>(p.W + p.e(r0 + i, h, d0 + 4 * tx)) = make_float4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<float4*>(p.U + p.e(r0 + i, h, d0 + 4 * tx)) = make_float4(u[0], u[1], u[2], u[3]);
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(ST_T) k_kda_state(PfArgs p)
{
    __shared__ StSmem s;
    float o[ST_R];
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x;
    st_load(p, s, h, vb, t);
    __syncthreads();
    for (int c = 0; c < p.nchunk(); ++c)
    {
        st_a(p, s, h, vb, c, t, o);
        __syncthreads();
        st_b(p, s, h, vb, c, t, o);
        __syncthreads();
    }
    st_store(p, s, h, vb, t);
}

// State kernel v2, same CTAs (head h, v block vb of VB = 32) and the same sums in the same order as st_a / st_b, with the
// chunk operands staged through shared memory tiles and register tiles of 2 rows x 4 v (VN, O) and 4 k x 4 v (S):
//   A  VN = U - W S, o = QE S       W / QE in k tiles of S2_KT columns; thread rows ty, ty + 32, v 4 tx .. + 4
//   O  O = o + AQ VN                AQ rows from global (L1), VN from smem
//   B  S = S exp(gc_last) + KD^T VN KD in row tiles of S2_IT rows; thread k rows 4 ty .. + 4, v 4 tx .. + 4
// smem ~30 KB so 3 CTAs fit an SM (nh * 4 = 128 CTAs on 48 SMs run in one wave).
constexpr int S2_KT = 8, S2_KTP = 12, S2_IT = 8;
struct St2Smem
{
    float4 S[DK][VB / 4];
    float4 vn[CH][VB / 4];
    union
    {
        float wq[2][CH][S2_KTP];
        float4 kd[S2_IT][DK / 4];
    } t;
};
__global__ void __launch_bounds__(ST_T, 3) k_kda_state2(PfArgs p)
{
    __shared__ St2Smem s;
    const Args& a = p.a;
    const int h = blockIdx.x, vb = blockIdx.y, t = threadIdx.x, tx = t % 8, ty = t / 8, v0 = vb * VB + 4 * tx;
    float* Sf = reinterpret_cast<float*>(s.S);
    for (int x = t; x < DK * VB; x += ST_T) Sf[x] = *st_cell(a, h, vb, x / VB, x % VB);
    __syncthreads();
    for (int c = 0; c < p.nchunk(); ++c)
    {
        const int r0 = c * CH, n = p.rows(c);
        // A
        float vn[2][4] = {}, o[2][4] = {};
        for (int k0 = 0; k0 < DK; k0 += S2_KT)
        {
            {   // 2 matrices x 64 rows x 8 columns = 256 float4 halves: thread t loads 4 floats of one row
                const int mat = t / 128, i = (t % 128) / 2, half = t % 2;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i < n) v = *reinterpret_cast<const float4*>((mat ? p.QE : p.W) + p.e(r0 + i, h, k0 + 4 * half));
                *reinterpret_cast<float4*>(&s.t.wq[mat][i][4 * half]) = v;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < S2_KT; ++kk)
            {
                const float4 sv = s.S[k0 + kk][tx];
#pragma unroll
                for (int r = 0; r < 2; ++r)
                {
                    const float w = s.t.wq[0][ty + 32 * r][kk], q = s.t.wq[1][ty + 32 * r][kk];
                    vn[r][0] += w * sv.x; vn[r][1] += w * sv.y; vn[r][2] += w * sv.z; vn[r][3] += w * sv.w;
                    o[r][0] += q * sv.x; o[r][1] += q * sv.y; o[r][2] += q * sv.z; o[r][3] += q * sv.w;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int i = ty + 32 * r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < n)
            {
                const float4 u = *reinterpret_cast<const float4*>(p.U + p.e(r0 + i, h, v0));
                v = make_float4(u.x - vn[r][0], u.y - vn[r][1], u.z - vn[r][2], u.w - vn[r][3]);
            }
            s.vn[i][tx] = v;
        }
        __syncthreads();
        // O
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int i = ty + 32 * r;
            if (i >= n) continue;
            const float* aq = p.AQ + p.aq(r0 + i, h, 0);
            float x0 = o[r][0], x1 = o[r][1], x2 = o[r][2], x3 = o[r][3];
            for (int j = 0; j <= i; ++j)
            {
                const float q = aq[j];
                const float4 w = s.vn[j][tx];
                x0 += q * w.x; x1 += q * w.y; x2 += q * w.z; x3 += q * w.w;
            }
            *reinterpret_cast<float4*>(a.O + p.e(r0 + i, h, v0)) = make_float4(x0, x1, x2, x3);
        }
        // B
        float sc[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            const int k = 4 * ty + r;
            const float eg = expf(p.GC[p.e(r0 + n - 1, h, k)]);
            const float4 sv = s.S[k][tx];
            sc[r][0] = sv.x * eg; sc[r][1] = sv.y * eg; sc[r][2] = sv.z * eg; sc[r][3] = sv.w * eg;
        }
        for (int i0 = 0; i0 < n; i0 += S2_IT)
        {
            {   // 8 rows x 32 float4
                const int i = t / 32, q4 = t % 32;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i0 + i < n) v = *reinterpret_cast<const float4*>(p.KD + p.e(r0 + i0 + i, h, 4 * q4));
                s.t.kd[i][q4] = v;
            }
            __syncthreads();
#pragma unroll
            for (int ii = 0; ii < S2_IT; ++ii)
            {
                const float4 kd = s.t.kd[ii][ty], w = s.vn[i0 + ii][tx];
                const float kv[4] = {kd.x, kd.y, kd.z, kd.w};
#pragma unroll
                for (int r = 0; r < 4; ++r)
                {
                    sc[r][0] += kv[r] * w.x; sc[r][1] += kv[r] * w.y; sc[r][2] += kv[r] * w.z; sc[r][3] += kv[r] * w.w;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) s.S[4 * ty + r][tx] = make_float4(sc[r][0], sc[r][1], sc[r][2], sc[r][3]);
        __syncthreads();
    }
    for (int x = t; x < DK * VB; x += ST_T) *st_cell(a, h, vb, x / VB, x % VB) = Sf[x];
}

inline PfArgs make_pf_args(const Args& a, const KdaPrefillScratch& ps)
{
    PfArgs p;
    p.a = a;
    p.QE = ps.QE; p.KD = ps.KD; p.W = ps.W; p.U = ps.U; p.GC = ps.GC; p.AQ = ps.AQ;
    return p;
}

inline void launch_core_prefill(const PfArgs& p, cudaStream_t st)
{
    const Args& a = p.a;
    static const bool p1 = getenv("G6_KDA_PF_P1") && atoi(getenv("G6_KDA_PF_P1")) == 1;   // 1: row-loop prep (old)
    if (p1) k_kda_prep<<<a.nh * 2, PREP_T, 0, st>>>(a);
    else launch_prep_pf(a, st);
    static const bool in1 = getenv("G6_KDA_PF_IN1") && atoi(getenv("G6_KDA_PF_IN1")) == 1;   // 1: intra kernel v1 (old)
    if (in1) k_kda_intra<<<dim3(a.nh, p.nchunk()), INTRA_T, 0, st>>>(p);
    else k_kda_intra3<<<dim3(a.nh, p.nchunk()), INTRA_T, 0, st>>>(p);
    static const bool st1 = getenv("G6_KDA_PF_ST1") && atoi(getenv("G6_KDA_PF_ST1")) == 1;   // 1: state kernel v1 (old)
    if (st1) k_kda_state<<<dim3(a.nh, DV / VB), ST_T, 0, st>>>(p);
    else k_kda_state2<<<dim3(a.nh, DV / VB), ST_T, 0, st>>>(p);
    k_kda_out<<<dim3(a.nh, a.M), OUT_T, 0, st>>>(a);
}

// o_proj of any M rows, in launches of g6mx::MX rows
inline void launch_oproj_rows(const KdaLayer& w, const KdaScratch& sc, int M, void* partial, int odt, cudaStream_t st)
{
    const size_t ob = odt == gv::O_BF16 ? 2 : 4, dl = static_cast<size_t>(w.nh) * DK;
    for (int m0 = 0; m0 < M; m0 += g6mx::MX)
        launch_oproj(w, sc.OF + (m0 / 8) * dl * 8, std::min(g6mx::MX, M - m0), static_cast<uint8_t*>(partial) + static_cast<size_t>(m0) * HID * ob,
                     odt, st);
}

inline void launch_prefill(const KdaLayer& w, const KdaState& s, const KdaScratch& sc, const KdaPrefillScratch& ps, const u32* xb,
                           const int* row_slot, int M, void* partial, cudaStream_t st, int odt = gv::O_BF16)
{
    const Args a = make_args(w, s, sc, row_slot, M);
    const int ld = a.pl().ld(), mx = g6mx::MX;
    for (int m0 = 0; m0 < M; m0 += mx)   // the projections run g6mx::MX rows per launch (fragment tiles of 8 rows)
        launch_proj_in(w, a, sc.P + static_cast<size_t>(m0) * ld, xb + static_cast<size_t>(m0 / 8) * HID * 4, std::min(mx, M - m0), st);
    launch_core_prefill(make_pf_args(a, ps), st);
    launch_oproj_rows(w, sc, M, partial, odt, st);
}
}   // namespace kda
