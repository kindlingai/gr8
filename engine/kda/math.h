// KDA (Kimi delta attention) decode math for GLM 5.3 Flash, shared by the CUDA kernels (kda.cuh) and the CPU model
// (tests/kda_cpu.h): every kernel is a sequence of phases separated by __syncthreads, and each phase is one function
// here that a thread runs. The CPU model runs the same functions, thread by thread and phase by phase, so the CPU
// test checks the kernels' arithmetic, order and layouts; only the launch and smem plumbing are GPU-only.
//
// Math per row (INTERFACES.md 4.2; flash_ref.FlashRef.kda): mixed = x @ [q;k;v]^T; conv = silu(sum_j w[c][j] in[t-3+j]);
// q,k = l2norm (eps 1e-6, sqrt(sum + eps)); q *= 128^-0.5; g = -5 sigmoid(exp(A_log[h]) (f_b(f_a x) + dt_bias));
// beta = sigmoid(b_proj x); S = S exp(g)[k]; delta = (v - S^T k) beta; S += k delta^T; o = S^T q;
// o = RMSNorm_head(o, o_norm, 1e-5) sigmoid(g_b(g_a x)); partial = o @ o_proj_shard^T.
// The state S is fp32 always (never rounded to bf16).
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define KDA_HD __host__ __device__ __forceinline__
#define KDA_UNROLL _Pragma("unroll")
#else
#define KDA_HD inline
#define KDA_UNROLL
#endif

namespace kda
{
// geometry (INTERFACES.md section 1 names KDA_HEADS, KDA_DK, KDA_DV, KDA_CONV, KDA_GATE_LO, N_KDA, kda_slot; switch to
// geom:: once model_geom.h carries them)
constexpr int HEADS = 64, DK = 128, DV = 128, D = HEADS * DK, CONV = 4, NHIST = CONV - 1;
constexpr int N_KDA = 34;
constexpr float GATE_LO = -5.f, L2_EPS = 1e-6f, RMS_EPS = 1e-5f;
constexpr float QSCALE = 0.08838834764831845f;   // DK^-0.5
KDA_HD constexpr bool is_kda(int l) { return l < 45 && l % 4 != 3; }
KDA_HD constexpr int kda_slot(int l) { return l - l / 4; }

// decode kernel shapes
constexpr int PREP_CH = 64;                // channels per prep CTA (half a head)
constexpr int PREP_T = 2 * PREP_CH;        // threads: f_b dots, then g_b dots; conv items 3 x PREP_CH
constexpr int WST = DK + 2;                // smem row stride of the staged f_b / g_b rows (bf16; odd word stride)
constexpr int VB = 32;                     // v columns per step CTA
constexpr int KP = 8;                      // k parts per step CTA
constexpr int KW = DK / KP;                // k rows per thread: 16
constexpr int STEP_T = VB * KP;            // 256 threads
constexpr int OUT_T = DV;                  // out kernel: one thread per v of one head and row

KDA_HD float bf2f(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}
KDA_HD uint16_t f2bf_rn(float f)   // round to nearest even (finite input)
{
#ifdef __CUDA_ARCH__
    uint32_t u = __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
#endif
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
KDA_HD uint16_t f2h_rn(float f)    // IEEE-754 binary16 round to nearest even
{
#ifdef __CUDA_ARCH__
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int exp = static_cast<int>((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0)
    {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int sh = 14 - exp;
        const uint32_t half = 1u << (sh - 1), mask = (1u << sh) - 1u;
        const uint32_t q = (mant >> sh) + ((mant & mask) > half || ((mant & mask) == half && ((mant >> sh) & 1u)));
        return static_cast<uint16_t>(sign | q);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    const uint32_t r = mant & 0x1fffu;
    mant = (mant >> 13) + (r > 0x1000u || (r == 0x1000u && ((mant >> 13) & 1u)));
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) + mant);
#endif
}
KDA_HD float sigm(float x) { return 1.f / (1.f + expf(-x)); }
KDA_HD float silu(float x) { return x / (1.f + expf(-x)); }

// mma B-operand fragment table (bind.cuh Xfrag layout): element (row m, k) of an [M][K] activation, as a u16 index
// into the tables (tile m / 8 at K * 4 u32; inside: [K/16][32 lanes][2 regs], lane = (m % 8) * 4 + c).
KDA_HD size_t frag_u16(int m, int k, int K)
{
    const int ks = k >> 4, kk = k & 15, reg = kk >> 3, c = (kk & 7) >> 1, half = kk & 1;
    const size_t u32i = static_cast<size_t>(m >> 3) * K * 4 + (static_cast<size_t>(ks) * 32 + (m & 7) * 4 + c) * 2 + reg;
    return u32i * 2 + half;
}

// ---------------------------------------------------------------------------------------------- layouts
// projections P [rows][pld]: q | k | v (own channels, nh*128 each) | f_a 128 | g_a 128 | b 64 (all heads), fp32
struct PLayout
{
    int dl = 0;                         // own channels nh * 128
    KDA_HD int q() const { return 0; }
    KDA_HD int fa() const { return 3 * dl; }
    KDA_HD int ga() const { return 3 * dl + DK; }
    KDA_HD int b() const { return 3 * dl + 2 * DK; }
    KDA_HD int ld() const { return 3 * dl + 2 * DK + HEADS; }
};

// what the phases read and write (device pointers on the GPU, host pointers in the CPU model)
struct Args
{
    int nh = 0, h0 = 0, M = 0;
    // weights, own heads (pointers already at h0 where the source is replicated)
    const uint16_t *conv_w[3] = {};     // q/k/v conv taps [own ch][4] bf16
    const uint16_t *fb = nullptr, *gb = nullptr;   // [own ch][128] bf16
    const float *A_log = nullptr;       // [own heads]
    const float *dt_bias = nullptr;     // [own ch]
    const uint16_t *onorm = nullptr;    // [128] bf16
    // state of this layer: S [slots][nhmax][DK][DV], conv [slots][NHIST][3][nhmax*DK]
    float *S = nullptr, *cst = nullptr;
    int nhmax = 0;
    const int* row_slot = nullptr;      // [M]
    // carry mode (MTP verify ring; k_kda_core only): row 0 loads the state of slot *src; a slot change between rows
    // stores the state to the old slot and keeps it in registers, so slot row_slot[m] ends with the state after rows
    // 0..m. nullptr: a slot change stores the old slot and loads the new one (independent sessions).
    const int* src = nullptr;
    // scratch
    const float* P = nullptr;           // [M][pld]
    float *C = nullptr;                 // [M][3 dl] conv + silu (q | k | v, before l2norm)
    float *G = nullptr;                 // [M][dl]   log decay g
    float *Z = nullptr;                 // [M][dl]   output gate before sigmoid
    float *O = nullptr;                 // [M][dl]   o = S^T q
    float *OG = nullptr;                // [M][dl]   gated o fp32 (optional, tests)
    uint16_t *OF = nullptr;             // BF16 fragment table of gated o, K = dl (BF16W o_proj input)
    uint16_t *OH = nullptr;             // FP16 row-major [M][dl] of gated o (trellis o_proj input)
    KDA_HD int dl() const { return nh * DK; }
    KDA_HD PLayout pl() const { PLayout p; p.dl = nh * DK; return p; }
    KDA_HD size_t s_off(int slot, int h) const { return (static_cast<size_t>(slot) * nhmax + h) * DK * DV; }
    KDA_HD size_t c_off(int slot, int hist, int stream, int ch) const
    {
        return ((static_cast<size_t>(slot) * NHIST + hist) * 3 + stream) * nhmax * DK + ch;
    }
};

// ---------------------------------------------------------------------------------------------- prep kernel
// CTA (head h, half hf): channels c0 = h * 128 + hf * 64 .. + 64. Thread t < 64: f_b row c0 + t; t >= 64: g_b row c0 + t - 64.
// Conv items i = t and t + 128 (< 192): stream i / 64, channel c0 + i % 64. Rows run in order; a slot's 3-token
// history is loaded when the slot changes and stored back when it changes again (rows of one slot are consecutive).
struct PrepSmem
{
    uint16_t w[2][PREP_CH][WST];        // staged f_b and g_b rows
    float vec[2][DK];                   // f_a x and g_a x of the current row
};
struct PrepThread
{
    float hist[2][NHIST];               // conv history of the thread's two items
    int cur = -1;                       // slot whose history is in hist
};
KDA_HD int prep_c0(int h, int hf) { return h * DK + hf * PREP_CH; }

// phase 0 (once): stage the CTA's f_b / g_b rows
KDA_HD void prep_stage(const Args& a, PrepSmem& s, int h, int hf, int t)
{
    const int c0 = prep_c0(h, hf);
    for (int i = t; i < 2 * PREP_CH * DK; i += PREP_T)
    {
        const int mat = i / (PREP_CH * DK), r = (i / DK) % PREP_CH, j = i % DK;
        s.w[mat][r][j] = (mat ? a.gb : a.fb)[static_cast<size_t>(c0 + r) * DK + j];
    }
}
// conv of one item for row m (slot history handled here); writes C
KDA_HD void prep_conv_item(const Args& a, PrepThread& th, int e, int item, int m, int h, int hf)
{
    const int stream = item / PREP_CH, ch = prep_c0(h, hf) + item % PREP_CH, dl = a.dl();
    const int slot = a.row_slot[m];
    if (slot != th.cur)
    {
        if (th.cur >= 0)
            for (int j = 0; j < NHIST; ++j) a.cst[a.c_off(th.cur, j, stream, ch)] = th.hist[e][j];
        for (int j = 0; j < NHIST; ++j) th.hist[e][j] = a.cst[a.c_off(slot, j, stream, ch)];
    }
    const float x = a.P[static_cast<size_t>(m) * a.pl().ld() + stream * dl + ch];
    const uint16_t* w = a.conv_w[stream] + static_cast<size_t>(ch) * CONV;
    float acc = th.hist[e][0] * bf2f(w[0]);
    for (int j = 1; j < NHIST; ++j) acc += th.hist[e][j] * bf2f(w[j]);
    acc += x * bf2f(w[NHIST]);
    for (int j = 0; j + 1 < NHIST; ++j) th.hist[e][j] = th.hist[e][j + 1];
    th.hist[e][NHIST - 1] = x;
    a.C[static_cast<size_t>(m) * 3 * dl + stream * dl + ch] = silu(acc);
}
// phase A of row m: load f_a x, g_a x; conv items
KDA_HD void prep_row_a(const Args& a, PrepSmem& s, PrepThread& th, int m, int h, int hf, int t)
{
    const float* p = a.P + static_cast<size_t>(m) * a.pl().ld();
    if (t < DK)
    {
        s.vec[0][t] = p[a.pl().fa() + t];
        s.vec[1][t] = p[a.pl().ga() + t];
    }
    prep_conv_item(a, th, 0, t, m, h, hf);
    if (t + PREP_T < 3 * PREP_CH) prep_conv_item(a, th, 1, t + PREP_T, m, h, hf);
    th.cur = a.row_slot[m];             // after both items: they share the slot
}
// phase B of row m: f_b / g_b dots -> G, Z
KDA_HD void prep_row_b(const Args& a, const PrepSmem& s, int m, int h, int hf, int t)
{
    const int mat = t / PREP_CH, r = t % PREP_CH, ch = prep_c0(h, hf) + r;
    const float* v = s.vec[mat];
    float d = 0.f;
    for (int j = 0; j < DK; ++j) d += bf2f(s.w[mat][r][j]) * v[j];
    const size_t o = static_cast<size_t>(m) * a.dl() + ch;
    if (mat == 0) a.G[o] = GATE_LO * sigm(expf(a.A_log[h]) * (d + a.dt_bias[ch]));
    else a.Z[o] = d;
}
// end: store the conv history of the last slot
KDA_HD void prep_end(const Args& a, const PrepThread& th, int h, int hf, int t)
{
    for (int e = 0; e < 2; ++e)
    {
        const int item = t + e * PREP_T;
        if (item >= 3 * PREP_CH || th.cur < 0) continue;
        const int stream = item / PREP_CH, ch = prep_c0(h, hf) + item % PREP_CH;
        for (int j = 0; j < NHIST; ++j) a.cst[a.c_off(th.cur, j, stream, ch)] = th.hist[e][j];
    }
}

// ---------------------------------------------------------------------------------------------- step kernel
// CTA (head h, v block vb): v columns vb * 32 .. + 32 of head h. Thread t: v = t % 32, k part kp = t / 32 (k rows
// kp * 16 .. + 16), its 16 state cells in registers across the rows of one slot.
struct StepSmem
{
    float q[DK], k[DK], eg[DK];
    float qd, kd, beta;                 // q, k norm denominators; beta of the row
    float red[KP][VB];
};
struct StepThread
{
    float s[KW];
    int cur = -1;
};
KDA_HD float* step_cell(const Args& a, int slot, int h, int vb, int t, int i)
{
    const int v = vb * VB + t % VB, k = (t / VB) * KW + i;
    return a.S + a.s_off(slot, h) + static_cast<size_t>(k) * DV + v;
}
// phase 0 of row m: swap the slot's state in, stage q, k, exp(g)
KDA_HD void step_row_0(const Args& a, StepSmem& s, StepThread& th, int m, int h, int vb, int t)
{
    const int slot = a.row_slot[m];
    if (slot != th.cur)
    {
        if (th.cur >= 0)
        {
            KDA_UNROLL
            for (int i = 0; i < KW; ++i) *step_cell(a, th.cur, h, vb, t, i) = th.s[i];
        }
        KDA_UNROLL
        for (int i = 0; i < KW; ++i) th.s[i] = *step_cell(a, slot, h, vb, t, i);
        th.cur = slot;
    }
    const int dl = a.dl();
    const float* c = a.C + static_cast<size_t>(m) * 3 * dl + h * DK;
    if (t < DK)
    {
        s.q[t] = c[t];
        s.eg[t] = expf(a.G[static_cast<size_t>(m) * dl + h * DK + t]);
    }
    else if (t < 2 * DK) s.k[t - DK] = c[dl + t - DK];
}
// phase 1: norms (one thread each, fixed order) and beta
KDA_HD void step_row_1(const Args& a, StepSmem& s, int m, int h, int t)
{
    if (t == 0 || t == 32)
    {
        const float* x = t ? s.k : s.q;
        float ss = 0.f;
        for (int j = 0; j < DK; ++j) ss += x[j] * x[j];
        (t ? s.kd : s.qd) = sqrtf(ss + L2_EPS);
    }
    else if (t == 64) s.beta = sigm(a.P[static_cast<size_t>(m) * a.pl().ld() + a.pl().b() + a.h0 + h]);
}
// phase 2: decay, partial S^T k
KDA_HD void step_row_2(StepSmem& s, StepThread& th, int t)
{
    const int k0 = (t / VB) * KW;
    float p = 0.f;
    KDA_UNROLL
    for (int i = 0; i < KW; ++i)
    {
        th.s[i] *= s.eg[k0 + i];
        p += th.s[i] * (s.k[k0 + i] / s.kd);
    }
    s.red[t / VB][t % VB] = p;
}
// phase 3: delta, update, partial S^T q (the partials of phase 2 are read before any thread overwrites them: the
// q partials go to red only after phase 3's barrier, in phase 3b)
KDA_HD float step_row_3(const Args& a, const StepSmem& s, StepThread& th, int m, int h, int vb, int t)
{
    const int k0 = (t / VB) * KW, v = t % VB;
    float kv = 0.f;
    for (int p = 0; p < KP; ++p) kv += s.red[p][v];
    const float vv = a.C[static_cast<size_t>(m) * 3 * a.dl() + 2 * a.dl() + h * DK + vb * VB + v];
    const float delta = (vv - kv) * s.beta;
    float o = 0.f;
    KDA_UNROLL
    for (int i = 0; i < KW; ++i)
    {
        th.s[i] += (s.k[k0 + i] / s.kd) * delta;
        o += th.s[i] * (s.q[k0 + i] / s.qd * QSCALE);
    }
    return o;
}
KDA_HD void step_row_3b(StepSmem& s, int t, float o) { s.red[t / VB][t % VB] = o; }
// phase 4: o of the row
KDA_HD void step_row_4(const Args& a, const StepSmem& s, int m, int h, int vb, int t)
{
    if (t >= VB) return;
    float o = 0.f;
    for (int p = 0; p < KP; ++p) o += s.red[p][t];
    a.O[static_cast<size_t>(m) * a.dl() + h * DK + vb * VB + t] = o;
}
KDA_HD void step_end(const Args& a, const StepThread& th, int h, int vb, int t)
{
    if (th.cur < 0) return;
    KDA_UNROLL
    for (int i = 0; i < KW; ++i) *step_cell(a, th.cur, h, vb, t, i) = th.s[i];
}

// ---------------------------------------------------------------------------------------------- out kernel
// CTA (head h, row m), thread v: gated o = RMSNorm(o_h) * o_norm * sigmoid(z); BF16 into the o_proj fragment table
struct OutSmem
{
    float o[DV];
    float r;
};
KDA_HD void out_0(const Args& a, OutSmem& s, int m, int h, int t) { s.o[t] = a.O[static_cast<size_t>(m) * a.dl() + h * DK + t]; }
KDA_HD void out_1(OutSmem& s, int t)
{
    if (t) return;
    float ss = 0.f;
    for (int j = 0; j < DV; ++j) ss += s.o[j] * s.o[j];
    s.r = 1.f / sqrtf(ss / DV + RMS_EPS);
}
KDA_HD void out_2(const Args& a, const OutSmem& s, int m, int h, int t)
{
    const size_t i = static_cast<size_t>(m) * a.dl() + h * DK + t;
    const float y = s.o[t] * s.r * bf2f(a.onorm[t]) * sigm(a.Z[i]);
    if (a.OG) a.OG[i] = y;
    if (a.OF) a.OF[frag_u16(m, h * DK + t, a.dl())] = f2bf_rn(y);
    if (a.OH) a.OH[i] = f2h_rn(y);
}

// ---------------------------------------------------------------------------------------------- fused core kernel
// CTA (head h): all 128 channels and all 128 v columns of head h in one launch (CORE_T = KP * DV = 1024 threads).
// Thread t: in step, v = t & 127, kp = t >> 7 (k rows kp * 16 .. + 16); in prep, t < 256 computes f_b / g_b dots
// (mat = t >> 7, r = t & 127) and 256 <= t < 640 computes conv + silu (stream = (t - 256) >> 7, r = (t - 256) & 127).
constexpr int CORE_T = KP * DV;

struct CoreSmem
{
    float vec[4][2][DK];
    float q[4][DK], k[4][DK], v[4][DK];
    float eg[4][DK], z[4][DK];
    float o[DV];
    float qd[4], kd[4], beta[4], r;
    union
    {
        uint32_t w[8][32][33];
        struct
        {
            float red[KP][DV];
            float red_q[KP][DV];
        };
    } u;
};

struct CoreThread
{
    float s[KW];
    int cur_s = -1;
};

KDA_HD float* core_step_cell(const Args& a, int slot, int h, int t, int i)
{
    const int v = t & (DV - 1), k = (t >> 7) * KW + i;
    return a.S + a.s_off(slot, h) + static_cast<size_t>(k) * DV + v;
}

// phase 0 of row m: stage f_a/g_a and beta (hoisted at m==0 for M<=4)
template <bool WRITE_DBG, int M_MODE = 0>
KDA_HD void core_row_0(const Args& a, CoreSmem& s, CoreThread&, int m, int h, int t)
{
    const size_t pld = a.pl().ld();
    if constexpr (M_MODE == 1)
    {
        if (t < 2 * DK)
        {
            const int mat = t >> 7, c = t & (DK - 1);
            s.vec[0][mat][c] = a.P[(mat ? a.pl().ga() : a.pl().fa()) + c];
        }
        else if (t == 2 * DK)
        {
            s.beta[0] = sigm(a.P[a.pl().b() + a.h0 + h]);
        }
        return;
    }
    if constexpr (M_MODE == 4)
    {
        if (t < a.M * 2 * DK)
        {
            const int mi = t >> 8, mat = (t >> 7) & 1, c = t & (DK - 1);
            const float* pm = a.P + static_cast<size_t>(mi) * pld;
            s.vec[mi][mat][c] = pm[(mat ? a.pl().ga() : a.pl().fa()) + c];
        }
        if (t < a.M)
        {
            s.beta[t] = sigm(a.P[static_cast<size_t>(t) * pld + a.pl().b() + a.h0 + h]);
        }
        return;
    }
    const float* p = a.P + static_cast<size_t>(m) * pld;
    if (t < DK) s.vec[0][0][t] = p[a.pl().fa() + t];
    else if (t < 2 * DK) s.vec[0][1][t - DK] = p[a.pl().ga() + t - DK];
    else if (t == 2 * DK) s.beta[0] = sigm(p[a.pl().b() + a.h0 + h]);
}

// phase 1 of row m: f_b / g_b dots (warps 0..7, loaded ONCE across M<=4), overlapped with conv+silu + q/k L2 norms
// (warps 8..19) and state S prefetch (warps 8..31)
template <bool WRITE_DBG, int M_MODE = 0>
KDA_HD void core_row_1(const Args& a, CoreSmem& s, CoreThread& th, int m, int h, int t)
{
    const int dl = a.dl();
    const size_t pld = a.pl().ld();
    if constexpr (M_MODE == 1)
    {
        const int slot0 = a.row_slot[0];
        const int from0 = a.src ? *a.src : slot0;
        if (t >= 2 * DK)
        {
            if (t < 5 * DK)
            {
                const int item = t - 2 * DK, stream = item >> 7, r_head = item & (DK - 1), ch = h * DK + r_head;
                const uint16_t* cw = a.conv_w[stream] + static_cast<size_t>(ch) * CONV;
                const float h0 = a.cst[a.c_off(from0, 0, stream, ch)];
                const float h1 = a.cst[a.c_off(from0, 1, stream, ch)];
                const float h2 = a.cst[a.c_off(from0, 2, stream, ch)];
                const float x  = a.P[stream * dl + ch];
                float acc = h0 * bf2f(cw[0]);
                acc += h1 * bf2f(cw[1]);
                acc += h2 * bf2f(cw[2]);
                acc += x * bf2f(cw[3]);
                a.cst[a.c_off(slot0, 0, stream, ch)] = h1;
                a.cst[a.c_off(slot0, 1, stream, ch)] = h2;
                a.cst[a.c_off(slot0, 2, stream, ch)] = x;
                const float val = silu(acc);
                if (stream == 0) s.q[0][r_head] = val;
                else if (stream == 1) s.k[0][r_head] = val;
                else s.v[0][r_head] = val;
                if (WRITE_DBG && a.C) a.C[stream * dl + ch] = val;
#ifdef __CUDA_ARCH__
                if (t < 4 * DK)
                {
                    asm volatile("bar.sync 2, 256;");
                    const int idx = t - 2 * DK;
                    if ((idx & 31) == 0 && (idx >> 5) < 2)
                    {
                        const bool isk = ((idx >> 5) & 1) != 0;
                        const float* xv = isk ? s.k[0] : s.q[0];
                        float ss = 0.f;
                        KDA_UNROLL
                        for (int j = 0; j < DK; ++j) ss += xv[j] * xv[j];
                        (isk ? s.kd[0] : s.qd[0]) = sqrtf(ss + L2_EPS);
                    }
                }
#endif
            }
#ifndef __CUDA_ARCH__
            if (t >= 5 * DK && t < 5 * DK + 2)
            {
                const bool isk = ((t - 5 * DK) & 1) != 0;
                const float* xv = isk ? s.k[0] : s.q[0];
                float ss = 0.f;
                for (int j = 0; j < DK; ++j) ss += xv[j] * xv[j];
                (isk ? s.kd[0] : s.qd[0]) = sqrtf(ss + L2_EPS);
            }
#endif
            KDA_UNROLL
            for (int i = 0; i < KW; ++i)
            {
#ifdef __CUDA_ARCH__
                th.s[i] = __ldg(core_step_cell(a, from0, h, t, i));
#else
                th.s[i] = *core_step_cell(a, from0, h, t, i);
#endif
            }
            return;
        }
        const int mat = t >> 7, r_head = t & (DK - 1), ch = h * DK + r_head;
        float d0 = 0.f;
#ifdef __CUDA_ARCH__
        const int w = t >> 5, lane = t & 31, q = lane >> 3, p = lane & 7, c32 = p * 4;
        const int c_warp = h * DK + ((w & 3) << 5);
        const uint4* wm4 = reinterpret_cast<const uint4*>((mat ? a.gb : a.fb) + static_cast<size_t>(c_warp) * DK);
        KDA_UNROLL
        for (int half = 0; half < 2; ++half)
        {
            KDA_UNROLL
            for (int k = 0; k < 4; ++k)
            {
                const int r0 = k * 8 + q, r1 = r0 + 4;
                const uint4 ld0 = __ldg(wm4 + r0 * 16 + half * 8 + p);
                const uint4 ld1 = __ldg(wm4 + r1 * 16 + half * 8 + p);
                s.u.w[w][r0][c32 + 0] = ld0.x;
                s.u.w[w][r0][c32 + 1] = ld0.y;
                s.u.w[w][r0][c32 + 2] = ld0.z;
                s.u.w[w][r0][c32 + 3] = ld0.w;
                s.u.w[w][r1][c32 + 0] = ld1.x;
                s.u.w[w][r1][c32 + 1] = ld1.y;
                s.u.w[w][r1][c32 + 2] = ld1.z;
                s.u.w[w][r1][c32 + 3] = ld1.w;
            }
            __syncwarp();
            const int c_base = half * 64;
            KDA_UNROLL
            for (int col32 = 0; col32 < 32; ++col32)
            {
                const uint32_t u = s.u.w[w][lane][col32];
                const int c = c_base + 2 * col32;
                d0 += bf2f(static_cast<uint16_t>(u)) * s.vec[0][mat][c];
                d0 += bf2f(static_cast<uint16_t>(u >> 16)) * s.vec[0][mat][c + 1];
            }
            if (half == 0) __syncwarp();
        }
#else
        const uint16_t* wm = (mat ? a.gb : a.fb) + static_cast<size_t>(ch) * DK;
        for (int j = 0; j < DK; ++j) d0 += bf2f(wm[j]) * s.vec[0][mat][j];
#endif
        if (mat == 0)
        {
            const float g = GATE_LO * sigm(expf(a.A_log[h]) * (d0 + a.dt_bias[ch]));
            s.eg[0][r_head] = expf(g);
            if (WRITE_DBG && a.G) a.G[ch] = g;
        }
        else
        {
            s.z[0][r_head] = d0;
            if (WRITE_DBG && a.Z) a.Z[ch] = d0;
        }
        return;
    }
    if (t >= 2 * DK)
    {
        const int nm = (M_MODE == 4) ? a.M : 1;
        if (t < 5 * DK)
        {
            const int item = t - 2 * DK, stream = item >> 7, r_head = item & (DK - 1), ch = h * DK + r_head;
            const uint16_t* cw = a.conv_w[stream] + static_cast<size_t>(ch) * CONV;
            if constexpr (M_MODE == 4)
            {
                const float w0 = bf2f(cw[0]), w1 = bf2f(cw[1]), w2 = bf2f(cw[2]), w3 = bf2f(cw[3]);
                int cur_c = a.row_slot[0];
                const int from0 = a.src ? *a.src : cur_c;
                float h0 = a.cst[a.c_off(from0, 0, stream, ch)];
                float h1 = a.cst[a.c_off(from0, 1, stream, ch)];
                float h2 = a.cst[a.c_off(from0, 2, stream, ch)];
                #pragma unroll 1
                for (int mi = 0; mi < nm; ++mi)
                {
                    const int sl = a.row_slot[mi];
                    if (sl != cur_c)
                    {
                        a.cst[a.c_off(cur_c, 0, stream, ch)] = h0;
                        a.cst[a.c_off(cur_c, 1, stream, ch)] = h1;
                        a.cst[a.c_off(cur_c, 2, stream, ch)] = h2;
                        if (!a.src)
                        {
                            h0 = a.cst[a.c_off(sl, 0, stream, ch)];
                            h1 = a.cst[a.c_off(sl, 1, stream, ch)];
                            h2 = a.cst[a.c_off(sl, 2, stream, ch)];
                        }
                        cur_c = sl;
                    }
                    const float x = a.P[static_cast<size_t>(mi) * pld + stream * dl + ch];
                    float acc = h0 * w0;
                    acc += h1 * w1;
                    acc += h2 * w2;
                    acc += x * w3;
                    h0 = h1;
                    h1 = h2;
                    h2 = x;
                    const float val = silu(acc);
                    if (stream == 0) s.q[mi][r_head] = val;
                    else if (stream == 1) s.k[mi][r_head] = val;
                    else s.v[mi][r_head] = val;
                    if (WRITE_DBG && a.C) a.C[static_cast<size_t>(mi) * 3 * dl + stream * dl + ch] = val;
                }
                a.cst[a.c_off(cur_c, 0, stream, ch)] = h0;
                a.cst[a.c_off(cur_c, 1, stream, ch)] = h1;
                a.cst[a.c_off(cur_c, 2, stream, ch)] = h2;
            }
            else
            {
                float* sh = (stream == 0 ? s.q[1] : (stream == 1 ? s.k[1] : s.v[1])) + r_head;
                const int slot = a.row_slot[m];
                const int prev_c = (m == 0) ? -1 : a.row_slot[m - 1];
                float h0, h1, h2;
                if (m == 0)
                {
                    const int from = a.src ? *a.src : slot;
                    h0 = a.cst[a.c_off(from, 0, stream, ch)];
                    h1 = a.cst[a.c_off(from, 1, stream, ch)];
                    h2 = a.cst[a.c_off(from, 2, stream, ch)];
                }
                else
                {
                    h0 = sh[0 * DK];
                    h1 = sh[1 * DK];
                    h2 = sh[2 * DK];
                    if (slot != prev_c)
                    {
                        a.cst[a.c_off(prev_c, 0, stream, ch)] = h0;
                        a.cst[a.c_off(prev_c, 1, stream, ch)] = h1;
                        a.cst[a.c_off(prev_c, 2, stream, ch)] = h2;
                        if (!a.src)
                        {
                            h0 = a.cst[a.c_off(slot, 0, stream, ch)];
                            h1 = a.cst[a.c_off(slot, 1, stream, ch)];
                            h2 = a.cst[a.c_off(slot, 2, stream, ch)];
                        }
                    }
                }
                const float x = a.P[static_cast<size_t>(m) * pld + stream * dl + ch];
                float acc = h0 * bf2f(cw[0]);
                acc += h1 * bf2f(cw[1]);
                acc += h2 * bf2f(cw[2]);
                acc += x * bf2f(cw[3]);
                if (m + 1 == a.M)
                {
                    a.cst[a.c_off(slot, 0, stream, ch)] = h1;
                    a.cst[a.c_off(slot, 1, stream, ch)] = h2;
                    a.cst[a.c_off(slot, 2, stream, ch)] = x;
                }
                else
                {
                    sh[0 * DK] = h1;
                    sh[1 * DK] = h2;
                    sh[2 * DK] = x;
                }
                const float val = silu(acc);
                if (stream == 0) s.q[0][r_head] = val;
                else if (stream == 1) s.k[0][r_head] = val;
                else s.v[0][r_head] = val;
                if (WRITE_DBG && a.C) a.C[static_cast<size_t>(m) * 3 * dl + stream * dl + ch] = val;
            }
#ifdef __CUDA_ARCH__
            if (t < 4 * DK)
            {
                asm volatile("bar.sync 2, 256;");
                const int idx = t - 2 * DK;
                if ((idx & 31) == 0 && (idx >> 5) < 2 * nm)
                {
                    const int pair = idx >> 5;
                    const int mi = pair >> 1;
                    const bool isk = (pair & 1) != 0;
                    const float* x = isk ? s.k[mi] : s.q[mi];
                    float ss = 0.f;
                    KDA_UNROLL
                    for (int j = 0; j < DK; ++j) ss += x[j] * x[j];
                    (isk ? s.kd[mi] : s.qd[mi]) = sqrtf(ss + L2_EPS);
                }
            }
#endif
        }
#ifndef __CUDA_ARCH__
        if (t >= 5 * DK && t < 5 * DK + 2 * nm)
        {
            const int pair = t - 5 * DK;
            const int mi = pair >> 1;
            const bool isk = (pair & 1) != 0;
            const float* x = isk ? s.k[mi] : s.q[mi];
            float ss = 0.f;
            for (int j = 0; j < DK; ++j) ss += x[j] * x[j];
            (isk ? s.kd[mi] : s.qd[mi]) = sqrtf(ss + L2_EPS);
        }
#endif
        // Warps 8..31 prefetch slot 0's state S concurrently with warps 0..7 loading/computing f_b/g_b!
        if (M_MODE == 4 || m == 0)
        {
            const int slot0 = a.row_slot[0];
            const int from0 = a.src ? *a.src : slot0;
            KDA_UNROLL
            for (int i = 0; i < KW; ++i)
            {
#ifdef __CUDA_ARCH__
                th.s[i] = __ldg(core_step_cell(a, from0, h, t, i));
#else
                th.s[i] = *core_step_cell(a, from0, h, t, i);
#endif
            }
            th.cur_s = slot0;
        }
        return;
    }

    const int mat = t >> 7, r_head = t & (DK - 1), ch = h * DK + r_head;
    if constexpr (M_MODE == 4)
    {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#ifdef __CUDA_ARCH__
        const int w = t >> 5, lane = t & 31, q = lane >> 3, p = lane & 7, c32 = p * 4;
        const int c_warp = h * DK + ((w & 3) << 5);
        const uint4* wm4 = reinterpret_cast<const uint4*>((mat ? a.gb : a.fb) + static_cast<size_t>(c_warp) * DK);
        KDA_UNROLL
        for (int half = 0; half < 2; ++half)
        {
            KDA_UNROLL
            for (int k = 0; k < 4; ++k)
            {
                const int r0 = k * 8 + q, r1 = r0 + 4;
                const uint4 ld0 = __ldg(wm4 + r0 * 16 + half * 8 + p);
                const uint4 ld1 = __ldg(wm4 + r1 * 16 + half * 8 + p);
                s.u.w[w][r0][c32 + 0] = ld0.x;
                s.u.w[w][r0][c32 + 1] = ld0.y;
                s.u.w[w][r0][c32 + 2] = ld0.z;
                s.u.w[w][r0][c32 + 3] = ld0.w;
                s.u.w[w][r1][c32 + 0] = ld1.x;
                s.u.w[w][r1][c32 + 1] = ld1.y;
                s.u.w[w][r1][c32 + 2] = ld1.z;
                s.u.w[w][r1][c32 + 3] = ld1.w;
            }
            __syncwarp();
            const int c_base = half * 64;
            KDA_UNROLL
            for (int col32 = 0; col32 < 32; ++col32)
            {
                const uint32_t u = s.u.w[w][lane][col32];
                const float w_lo = bf2f(static_cast<uint16_t>(u));
                const float w_hi = bf2f(static_cast<uint16_t>(u >> 16));
                const int c = c_base + 2 * col32;
                d0 += w_lo * s.vec[0][mat][c]; d0 += w_hi * s.vec[0][mat][c + 1];
                d1 += w_lo * s.vec[1][mat][c]; d1 += w_hi * s.vec[1][mat][c + 1];
                if (a.M > 2) { d2 += w_lo * s.vec[2][mat][c]; d2 += w_hi * s.vec[2][mat][c + 1]; }
                if (a.M > 3) { d3 += w_lo * s.vec[3][mat][c]; d3 += w_hi * s.vec[3][mat][c + 1]; }
            }
            if (half == 0) __syncwarp();
        }
#else
        const uint16_t* wm = (mat ? a.gb : a.fb) + static_cast<size_t>(ch) * DK;
        for (int j = 0; j < DK; ++j)
        {
            const float wv = bf2f(wm[j]);
            d0 += wv * s.vec[0][mat][j];
            if (a.M > 1) d1 += wv * s.vec[1][mat][j];
            if (a.M > 2) d2 += wv * s.vec[2][mat][j];
            if (a.M > 3) d3 += wv * s.vec[3][mat][j];
        }
#endif
        const float alog = (mat == 0) ? expf(a.A_log[h]) : 0.f;
        const float dtb = (mat == 0) ? a.dt_bias[ch] : 0.f;
        auto store_out = [&](int mi, float dval) {
            const size_t o = static_cast<size_t>(mi) * dl + ch;
            if (mat == 0)
            {
                const float g = GATE_LO * sigm(alog * (dval + dtb));
                s.eg[mi][r_head] = expf(g);
                if (WRITE_DBG && a.G) a.G[o] = g;
            }
            else
            {
                s.z[mi][r_head] = dval;
                if (WRITE_DBG && a.Z) a.Z[o] = dval;
            }
        };
        store_out(0, d0);
        if (a.M > 1) store_out(1, d1);
        if (a.M > 2) store_out(2, d2);
        if (a.M > 3) store_out(3, d3);
    }
    else
    {
        float d = 0.f;
#ifdef __CUDA_ARCH__
        const int w = t >> 5, lane = t & 31, q = lane >> 3, p = lane & 7, c32 = p * 4;
        const int c_warp = h * DK + ((w & 3) << 5);
        const uint4* wm4 = reinterpret_cast<const uint4*>((mat ? a.gb : a.fb) + static_cast<size_t>(c_warp) * DK);
        #pragma unroll 1
        for (int half = 0; half < 2; ++half)
        {
            KDA_UNROLL
            for (int k = 0; k < 4; ++k)
            {
                const int r0 = k * 8 + q, r1 = r0 + 4;
                const uint4 ld0 = __ldg(wm4 + r0 * 16 + half * 8 + p);
                const uint4 ld1 = __ldg(wm4 + r1 * 16 + half * 8 + p);
                s.u.w[w][r0][c32 + 0] = ld0.x;
                s.u.w[w][r0][c32 + 1] = ld0.y;
                s.u.w[w][r0][c32 + 2] = ld0.z;
                s.u.w[w][r0][c32 + 3] = ld0.w;
                s.u.w[w][r1][c32 + 0] = ld1.x;
                s.u.w[w][r1][c32 + 1] = ld1.y;
                s.u.w[w][r1][c32 + 2] = ld1.z;
                s.u.w[w][r1][c32 + 3] = ld1.w;
            }
            __syncwarp();
            const int c_base = half * 64;
            KDA_UNROLL
            for (int col32 = 0; col32 < 32; ++col32)
            {
                const uint32_t u = s.u.w[w][lane][col32];
                const int c = c_base + 2 * col32;
                d += bf2f(static_cast<uint16_t>(u)) * s.vec[0][mat][c];
                d += bf2f(static_cast<uint16_t>(u >> 16)) * s.vec[0][mat][c + 1];
            }
            if (half == 0) __syncwarp();
        }
#else
        const uint16_t* wm = (mat ? a.gb : a.fb) + static_cast<size_t>(ch) * DK;
        for (int j = 0; j < DK; ++j) d += bf2f(wm[j]) * s.vec[0][mat][j];
#endif
        const size_t o = static_cast<size_t>(m) * dl + ch;
        if (mat == 0)
        {
            const float g = GATE_LO * sigm(expf(a.A_log[h]) * (d + a.dt_bias[ch]));
            s.eg[0][r_head] = expf(g);
            if (WRITE_DBG && a.G) a.G[o] = g;
        }
        else
        {
            s.z[0][r_head] = d;
            if (WRITE_DBG && a.Z) a.Z[o] = d;
        }
    }
}

// phase 2+3 fused: ensure slot state S is loaded (warps 0..7 or slot change), apply decay, and compute partial S^T k
template <int M_MODE = 0>
KDA_HD void core_row_23(const Args& a, CoreSmem& s, CoreThread& th, int m, int h, int t)
{
    if constexpr (M_MODE == 1)
    {
        if (t < 2 * DK)
        {
            const int from = a.src ? *a.src : a.row_slot[0];
            KDA_UNROLL
            for (int i = 0; i < KW; ++i)
            {
#ifdef __CUDA_ARCH__
                th.s[i] = __ldg(core_step_cell(a, from, h, t, i));
#else
                th.s[i] = *core_step_cell(a, from, h, t, i);
#endif
            }
        }
    }
    else
    {
        const int slot = a.row_slot[m];
        if (slot != th.cur_s)
        {
            if (th.cur_s >= 0)
            {
                KDA_UNROLL
                for (int i = 0; i < KW; ++i) *core_step_cell(a, th.cur_s, h, t, i) = th.s[i];
            }
            if (th.cur_s < 0 || !a.src)
            {
                const int from = th.cur_s < 0 && a.src ? *a.src : slot;
                KDA_UNROLL
                for (int i = 0; i < KW; ++i)
                {
#ifdef __CUDA_ARCH__
                    th.s[i] = th.cur_s < 0 ? __ldg(core_step_cell(a, from, h, t, i)) : *core_step_cell(a, from, h, t, i);
#else
                    th.s[i] = *core_step_cell(a, from, h, t, i);
#endif
                }
            }
            th.cur_s = slot;
        }
    }
    const int mi = (M_MODE == 4) ? m : 0;
    const int v = t & (DV - 1), kp = t >> 7, k0 = kp * KW;
    const float kd = s.kd[mi];
    const float* eg = s.eg[mi] + k0;
    const float* k = s.k[mi] + k0;
    float p = 0.f;
    KDA_UNROLL
    for (int i = 0; i < KW; ++i)
    {
        th.s[i] *= eg[i];
        p += th.s[i] * (k[i] / kd);
    }
    s.u.red[kp][v] = p;
}

// phase 4+4b fused: delta, state update (and final writeback on last row), partial S^T q into red_q
template <int M_MODE = 0>
KDA_HD void core_row_4(const Args& a, CoreSmem& s, CoreThread& th, int m, int h, int t)
{
    const int mi = (M_MODE == 4) ? m : 0;
    const int v = t & (DV - 1), kp = t >> 7, k0 = kp * KW;
    float kv = 0.f;
    KDA_UNROLL
    for (int p = 0; p < KP; ++p) kv += s.u.red[p][v];
    const float delta = (s.v[mi][v] - kv) * s.beta[mi];
    const float kd = s.kd[mi], qd = s.qd[mi];
    const float* k = s.k[mi] + k0;
    const float* q = s.q[mi] + k0;
    float o = 0.f;
    KDA_UNROLL
    for (int i = 0; i < KW; ++i)
    {
        th.s[i] += (k[i] / kd) * delta;
        o += th.s[i] * (q[i] / qd * QSCALE);
    }
    s.u.red_q[kp][v] = o;
    if constexpr (M_MODE == 1)
    {
        const int slot = a.row_slot[0];
        KDA_UNROLL
        for (int i = 0; i < KW; ++i) *core_step_cell(a, slot, h, t, i) = th.s[i];
    }
    else if (m + 1 == a.M)
    {
        const int slot = th.cur_s;
        KDA_UNROLL
        for (int i = 0; i < KW; ++i) *core_step_cell(a, slot, h, t, i) = th.s[i];
        th.cur_s = -1;
    }
}

// phase 5+6+7: reduce S^T q partials into s.o[0..127], head RMSNorm, gated o -> OG / OF / OH
template <bool WRITE_DBG, int M_MODE = 0>
KDA_HD void core_row_5(const Args& a, CoreSmem& s, int m, int h, int t)
{
    if (t >= DV) return;
    const int mi = (M_MODE == 4) ? m : 0;
    const int row = (M_MODE == 1) ? 0 : m;
    float o = 0.f;
    KDA_UNROLL
    for (int p = 0; p < KP; ++p) o += s.u.red_q[p][t];
    s.o[t] = o;
    if (WRITE_DBG && a.O) a.O[static_cast<size_t>(row) * a.dl() + h * DK + t] = o;
#ifdef __CUDA_ARCH__
    asm volatile("bar.sync 1, 128;");
    float r = 0.f;
    if ((t & 31) == 0)
    {
        float ss = 0.f;
        KDA_UNROLL
        for (int j = 0; j < DV; ++j) ss += s.o[j] * s.o[j];
        r = 1.f / sqrtf(ss / DV + RMS_EPS);
    }
    r = __shfl_sync(0xffffffff, r, 0);
    const size_t i = static_cast<size_t>(row) * a.dl() + h * DK + t;
    const float y = o * r * bf2f(a.onorm[t]) * sigm(s.z[mi][t]);
    if (WRITE_DBG && a.OG) a.OG[i] = y;
    if (a.OF) a.OF[frag_u16(row, h * DK + t, a.dl())] = f2bf_rn(y);
    if (a.OH) a.OH[i] = f2h_rn(y);
#endif
}

template <bool WRITE_DBG>
KDA_HD void core_row_67(const Args& a, CoreSmem& s, int m, int h, int t)
{
#ifndef __CUDA_ARCH__
    if (t >= DV) return;
    const int mi = (a.M <= 4) ? m : 0;
    if (t == 0)
    {
        float ss = 0.f;
        for (int j = 0; j < DV; ++j) ss += s.o[j] * s.o[j];
        s.r = 1.f / sqrtf(ss / DV + RMS_EPS);
    }
    const float r = s.r;
    const size_t i = static_cast<size_t>(m) * a.dl() + h * DK + t;
    const float y = s.o[t] * r * bf2f(a.onorm[t]) * sigm(s.z[mi][t]);
    if (WRITE_DBG && a.OG) a.OG[i] = y;
    if (a.OF) a.OF[frag_u16(m, h * DK + t, a.dl())] = f2bf_rn(y);
    if (a.OH) a.OH[i] = f2h_rn(y);
#endif
}
}   // namespace kda
