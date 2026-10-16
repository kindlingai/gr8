// T1 of docs/FLASH-G9-KDA-SCAN.md: the G9 ON algorithm on the CPU (kda_g9_cpu.h) against the current CPU model (the
// OFF path: prefill.h through kda_cpu.h, fp32) and an fp64 gated-delta-rule reference. Reports, per case, the KDA
// state S error and the output error at depth 2K / 4K / 8K / 16K / 32K of ONE slot (the engine runs 4 calls of 8192
// rows; state carried between calls). 2048-row calls are chunk-aligned (CH 64 and 16 divide 2048), so they give the
// same bits as longer calls and expose the checkpoints; the self-check below verifies that for the OFF model.
//
//   g++ -O3 -march=native -fopenmp -std=c++17 -DG6_MODEL_FLASH=1 -o kda_g9_cpu_test kda_g9_cpu_test.cpp
//   kda_g9_cpu_test [--dense FLASH/full-dense/dense --golden golden/kda-L00-4096.safetensors] [--depth 32768]
//                   [--cases weak,strong,...] [--heads 0,17,33,60] [--small]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <omp.h>
#include "common.h"
#include "kda_cpu.h"
#include "kda_g9_cpu.h"

using namespace kda;
using namespace g9cpu;

constexpr int HD = 128, CALL = 2048;
static const int CKPT[] = {2048, 4096, 8192, 16384, 32768};

struct Err
{
    double d2 = 0, r2 = 0, mx = 0;
    void add(double a, double ref)
    {
        const double e = a - ref;
        d2 += e * e; r2 += ref * ref; mx = std::max(mx, std::fabs(e));
    }
    double rel() const { return std::sqrt(d2 / (r2 > 0 ? r2 : 1)); }
};

struct Case
{
    std::string name;
    int golden_head = -1;                 // >= 0: golden L00 head tiled over the depth
    float A_log = 0.f;
    float dt_lo = 0.f, dt_hi = 0.f;       // dt_bias per channel uniform in [lo, hi]
    float beta_mu = 0.f, beta_sd = 1.f;   // b_proj logit ~ N(mu, sd)
    bool corr = false;                    // highly correlated keys
    unsigned seed = 1;
};

struct Data   // one head, T rows
{
    int T = 0;
    std::vector<float> C, g, Graw16, Z, Z16, braw, dt, onorm;
    float A_log = 0.f;
    std::vector<float> S0;   // initial state [128][128]
};

static void gen_synth(const Case& cs, int T, Data& d)
{
    std::mt19937 rng(cs.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    d.T = T; d.A_log = cs.A_log;
    d.dt.resize(HD); d.onorm.resize(HD);
    for (int c = 0; c < HD; ++c)
    {
        d.dt[c] = cs.dt_lo + (cs.dt_hi - cs.dt_lo) * ud(rng);
        d.onorm[c] = rb(1.f + 0.2f * nd(rng));
    }
    std::vector<float> sc(3 * HD), mu(HD), fb(static_cast<size_t>(HD) * HD);
    for (auto& s : sc) s = std::exp(0.5f * nd(rng));
    for (auto& m : mu) m = nd(rng);
    for (auto& w : fb) w = rb(nd(rng) / std::sqrt(static_cast<float>(HD)));
    d.C.resize(static_cast<size_t>(T) * 3 * HD); d.g.resize(static_cast<size_t>(T) * HD); d.Graw16 = d.g;
    d.Z.resize(static_cast<size_t>(T) * HD); d.Z16 = d.Z; d.braw.resize(T);
    std::vector<float> fa(HD), fab(HD);
    const float ea = std::exp(cs.A_log);
    for (int t = 0; t < T; ++t)
    {
        float* c = &d.C[static_cast<size_t>(t) * 3 * HD];
        for (int j = 0; j < HD; ++j)
        {
            const float qx = nd(rng) * sc[j], kx = cs.corr ? mu[j] + 0.25f * nd(rng) : nd(rng) * sc[HD + j], vx = nd(rng) * sc[2 * HD + j];
            c[j] = silu(qx); c[HD + j] = silu(kx); c[2 * HD + j] = silu(vx);
            fa[j] = nd(rng); fab[j] = rb(fa[j]);
        }
        d.braw[t] = cs.beta_mu + cs.beta_sd * nd(rng);
        for (int ch = 0; ch < HD; ++ch)
        {
            float s32 = 0.f, s16 = 0.f;
            for (int j = 0; j < HD; ++j) { s32 += fb[static_cast<size_t>(ch) * HD + j] * fa[j]; s16 += fb[static_cast<size_t>(ch) * HD + j] * fab[j]; }
            d.g[static_cast<size_t>(t) * HD + ch] = GATE_LO * sigm(ea * (s32 + d.dt[ch]));
            d.Graw16[static_cast<size_t>(t) * HD + ch] = rb(s16);
            const float z = 1.5f * nd(rng);
            d.Z[static_cast<size_t>(t) * HD + ch] = z; d.Z16[static_cast<size_t>(t) * HD + ch] = rb(z);
        }
    }
    d.S0.assign(static_cast<size_t>(HD) * HD, 0.f);
}

// golden L00 head h, tiled to T rows. Graw is rebuilt from the golden g (inverse of the gate) and rounded to bf16.
struct Golden
{
    st::File f;
    const float *q, *k, *v, *g, *beta, *S4096;
    kda_test::W w;
};
static void gen_golden(const Case& cs, Golden& G, int T, Data& d)
{
    std::mt19937 rng(cs.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int h = cs.golden_head;
    d.T = T; d.A_log = G.w.A_log[h];
    d.dt.assign(G.w.dt_bias + h * HD, G.w.dt_bias + (h + 1) * HD);
    d.onorm.resize(HD);
    for (int c = 0; c < HD; ++c) d.onorm[c] = bf2f(G.w.onorm[c]);
    d.C.resize(static_cast<size_t>(T) * 3 * HD); d.g.resize(static_cast<size_t>(T) * HD); d.Graw16 = d.g;
    d.Z.resize(static_cast<size_t>(T) * HD); d.Z16 = d.Z; d.braw.resize(T);
    const double ea = std::exp(double(d.A_log));
    for (int t = 0; t < T; ++t)
    {
        const size_t gt = static_cast<size_t>(t % 4096) * 64 + h;
        float* c = &d.C[static_cast<size_t>(t) * 3 * HD];
        for (int j = 0; j < HD; ++j)
        {
            c[j] = G.q[gt * HD + j] / QSCALE; c[HD + j] = G.k[gt * HD + j]; c[2 * HD + j] = G.v[gt * HD + j];
        }
        const double b = std::min(std::max(double(G.beta[gt]), 1e-7), 1.0 - 1e-7);
        d.braw[t] = static_cast<float>(std::log(b / (1.0 - b)));
        for (int ch = 0; ch < HD; ++ch)
        {
            const float gv = G.g[gt * HD + ch];
            d.g[static_cast<size_t>(t) * HD + ch] = gv;
            const double s = std::min(std::max(double(gv) / GATE_LO, 1e-12), 1.0 - 1e-12);
            const double graw = std::log(s / (1.0 - s)) / ea - double(d.dt[ch]);
            d.Graw16[static_cast<size_t>(t) * HD + ch] = rb(static_cast<float>(graw));
            const float z = 1.5f * nd(rng);
            d.Z[static_cast<size_t>(t) * HD + ch] = z; d.Z16[static_cast<size_t>(t) * HD + ch] = rb(z);
        }
    }
    d.S0.assign(static_cast<size_t>(HD) * HD, 0.f);
}

// OFF: the current CPU model on rows [r0, r0 + M) of d, state S (fp32, in/out); O, OG fp32, OH fp16 bits.
static void off_call(const Data& d, int r0, int M, std::vector<float>& S, std::vector<float>& O, std::vector<float>& OG,
                     std::vector<uint16_t>& OH, std::vector<float>& P, std::vector<float>& Zc, std::vector<float>& Ocs,
                     std::vector<float>& QE, std::vector<float>& KD, std::vector<float>& PW, std::vector<float>& PU,
                     std::vector<float>& GC, std::vector<float>& AQ, std::vector<float>& Gc)
{
    PLayout pl; pl.dl = HD;
    const int ld = pl.ld();
    P.assign(static_cast<size_t>(M) * ld, 0.f);
    for (int m = 0; m < M; ++m) P[static_cast<size_t>(m) * ld + pl.b()] = d.braw[r0 + m];
    std::vector<int> rs(M, 0);
    std::vector<uint16_t> onb(HD);
    for (int c = 0; c < HD; ++c) onb[c] = f2bf_rn(d.onorm[c]);
    Args a;
    a.nh = 1; a.h0 = 0; a.M = M; a.nhmax = 1;
    a.onorm = onb.data(); a.S = S.data(); a.row_slot = rs.data(); a.P = P.data();
    a.C = const_cast<float*>(&d.C[static_cast<size_t>(r0) * 3 * HD]);
    Gc.assign(d.g.begin() + static_cast<size_t>(r0) * HD, d.g.begin() + static_cast<size_t>(r0 + M) * HD);
    Zc.assign(d.Z.begin() + static_cast<size_t>(r0) * HD, d.Z.begin() + static_cast<size_t>(r0 + M) * HD);
    a.G = Gc.data(); a.Z = Zc.data();
    O.assign(static_cast<size_t>(M) * HD, 0.f); OG = O; OH.assign(O.size(), 0);
    a.O = O.data(); a.OG = OG.data(); a.OH = OH.data();
    QE.assign(O.size(), 0.f); KD = QE; PW = QE; PU = QE; GC = QE; AQ.assign(static_cast<size_t>(M) * CH, 0.f);
    PfArgs p;
    p.a = a; p.QE = QE.data(); p.KD = KD.data(); p.W = PW.data(); p.U = PU.data(); p.GC = GC.data(); p.AQ = AQ.data();
    kda_cpu::prefill(p);
    kda_cpu::out(a);
}

struct Row { double s_on_off_mx, s_on_off, s_on_64_mx, s_on_64, s_off_64_mx, s_off_64; double o_on_off, o_on_64, o_off_64, y_on_off, y_on_64, y_off_64, oh_on_off, oh_on_off_mx, oh_on_64, oh_off_64; double s_on_gold = -1, s_off_gold = -1; };

static int g_bad = 0;

static std::vector<Row> run_case(const Case& cs, Golden* G, int T, bool selfcheck)
{
    Data d;
    if (cs.golden_head >= 0) gen_golden(cs, *G, T, d); else gen_synth(cs, T, d);
    std::vector<float> S_off(d.S0), S_on(d.S0);
    std::vector<double> S_64(d.S0.begin(), d.S0.end());
    std::vector<float> O, OG, Zc, Pb, Ocs, QE, KD, PW, PU, GC, AQ, Gc;
    std::vector<uint16_t> OH;
    std::vector<float> o16(static_cast<size_t>(CALL) * HD), yon(o16.size());
    std::vector<uint16_t> ohon(o16.size());
    std::vector<double> o64(o16.size()), y64(o16.size());
    std::vector<Row> rows;
    Err so[3], oo[3], yy[3], hh[3]; double ohmx = 0;   // window accumulators: on-off, on-64, off-64
    int next = 0;
    for (int r0 = 0; r0 < T; r0 += CALL)
    {
        const int M = std::min(CALL, T - r0);
        off_call(d, r0, M, S_off, O, OG, OH, Pb, Zc, Ocs, QE, KD, PW, PU, GC, AQ, Gc);
        OnIn in;
        in.M = M; in.C = &d.C[static_cast<size_t>(r0) * 3 * HD]; in.G16 = &d.Graw16[static_cast<size_t>(r0) * HD];
        in.Z16 = &d.Z16[static_cast<size_t>(r0) * HD]; in.braw = &d.braw[r0]; in.A_log = d.A_log; in.dt_bias = d.dt.data(); in.onorm = d.onorm.data();
        OnOut out; out.O16 = o16.data(); out.y = yon.data(); out.OH = ohon.data();
        on_call(in, S_on.data(), out);
        ref64_call(M, in.C, &d.g[static_cast<size_t>(r0) * HD], &d.Z[static_cast<size_t>(r0) * HD], in.braw, d.onorm.data(), S_64.data(), o64.data(), y64.data());
        for (size_t i = 0; i < static_cast<size_t>(M) * HD; ++i)
        {
            oo[0].add(o16[i], O[i]); oo[1].add(o16[i], o64[i]); oo[2].add(O[i], o64[i]);
            yy[0].add(yon[i], OG[i]); yy[1].add(yon[i], y64[i]); yy[2].add(OG[i], y64[i]);
            const double h_on = h2f(ohon[i]), h_off = h2f(OH[i]);
            hh[0].add(h_on, h_off); hh[1].add(h_on, y64[i]); hh[2].add(h_off, y64[i]);
        }
        if (next < 5 && (r0 + M == CKPT[next] || (r0 + M == T && next < 5 && CKPT[next] > T)))
        {
            Row R{};
            for (size_t i = 0; i < S_on.size(); ++i)
            {
                so[0].add(S_on[i], S_off[i]); so[1].add(S_on[i], S_64[i]); so[2].add(S_off[i], S_64[i]);
            }
            R.s_on_off = so[0].rel(); R.s_on_off_mx = so[0].mx; R.s_on_64 = so[1].rel(); R.s_on_64_mx = so[1].mx;
            R.s_off_64 = so[2].rel(); R.s_off_64_mx = so[2].mx;
            R.o_on_off = oo[0].rel(); R.o_on_64 = oo[1].rel(); R.o_off_64 = oo[2].rel();
            R.y_on_off = yy[0].rel(); R.y_on_64 = yy[1].rel(); R.y_off_64 = yy[2].rel();
            R.oh_on_off = hh[0].rel(); R.oh_on_off_mx = hh[0].mx; R.oh_on_64 = hh[1].rel(); R.oh_off_64 = hh[2].rel();
            if (cs.golden_head >= 0 && r0 + M == 4096)
            {
                const float* gs = G->S4096 + static_cast<size_t>(cs.golden_head) * HD * HD;
                Err a, b;
                for (size_t i = 0; i < S_on.size(); ++i) { a.add(S_on[i], gs[i]); b.add(S_off[i], gs[i]); }
                R.s_on_gold = a.rel(); R.s_off_gold = b.rel();
            }
            rows.push_back(R);
            for (auto* e : {&so[0], &so[1], &so[2], &oo[0], &oo[1], &oo[2], &yy[0], &yy[1], &yy[2], &hh[0], &hh[1], &hh[2]}) *e = Err();
            ++next;
        }
        (void)ohmx;
        if (selfcheck && r0 + M == 8192)
        {
            // OFF 4 x 2048 vs one call of 8192 rows: bit-identical state (chunk-aligned calls)
            std::vector<float> S1(d.S0);
            off_call(d, 0, 8192, S1, O, OG, OH, Pb, Zc, Ocs, QE, KD, PW, PU, GC, AQ, Gc);
            const bool ok = !memcmp(S1.data(), S_off.data(), S1.size() * 4);
            printf("  [%s] OFF state after 8192 rows: 4 x 2048-row calls vs one 8192-row call bit-identical: %s\n", cs.name.c_str(), ok ? "PASS" : "FAIL");
            if (!ok) ++g_bad;
        }
    }
    return rows;
}

// small calls from a nonzero state: ON and OFF vs fp64
static void small_cases()
{
    printf("\nsmall calls from nonzero state (random S0 ~ N(0, 0.3)), mixed decay; rel-L2 vs fp64\n");
    printf("  %-5s %-12s %-12s %-12s %-12s\n", "rows", "S ON", "S OFF", "O ON", "O OFF");
    const int Ms[] = {1, 15, 16, 17, 63, 64, 65, 4096};
    for (int M : Ms)
    {
        Case cs; cs.name = "small"; cs.A_log = 0.f; cs.dt_lo = -6.f; cs.dt_hi = 6.f; cs.seed = 100 + M;
        Data d; gen_synth(cs, M, d);
        std::mt19937 rng(7 + M);
        std::normal_distribution<float> nd(0.f, 0.3f);
        for (auto& s : d.S0) s = nd(rng);
        std::vector<float> S_off(d.S0), S_on(d.S0), O, OG, Zc, Pb, Ocs, QE, KD, PW, PU, GC, AQ, Gc;
        std::vector<double> S_64(d.S0.begin(), d.S0.end());
        std::vector<uint16_t> OH;
        off_call(d, 0, M, S_off, O, OG, OH, Pb, Zc, Ocs, QE, KD, PW, PU, GC, AQ, Gc);
        std::vector<float> o16(static_cast<size_t>(M) * HD), yon(o16.size());
        std::vector<uint16_t> ohon(o16.size());
        OnIn in; in.M = M; in.C = d.C.data(); in.G16 = d.Graw16.data(); in.Z16 = d.Z16.data(); in.braw = d.braw.data(); in.A_log = d.A_log;
        in.dt_bias = d.dt.data(); in.onorm = d.onorm.data();
        OnOut out; out.O16 = o16.data(); out.y = yon.data(); out.OH = ohon.data();
        on_call(in, S_on.data(), out);
        std::vector<double> o64(o16.size()), y64(o16.size());
        ref64_call(M, d.C.data(), d.g.data(), d.Z.data(), d.braw.data(), d.onorm.data(), S_64.data(), o64.data(), y64.data());
        Err s_on, s_off, o_on, o_off;
        for (size_t i = 0; i < S_on.size(); ++i) { s_on.add(S_on[i], S_64[i]); s_off.add(S_off[i], S_64[i]); }
        for (size_t i = 0; i < o16.size(); ++i) { o_on.add(o16[i], o64[i]); o_off.add(O[i], o64[i]); }
        const bool ok = s_on.rel() < 2e-2 && o_on.rel() < 2e-2 && s_off.rel() < 1e-4 && o_off.rel() < 1e-4;
        if (!ok) ++g_bad;
        printf("  %-5d %-12.3e %-12.3e %-12.3e %-12.3e %s\n", M, s_on.rel(), s_off.rel(), o_on.rel(), o_off.rel(), ok ? "PASS" : "FAIL");
    }
}

int main(int argc, char** argv)
{
    std::string dense, golden, only;
    std::vector<int> heads = {0, 17, 33, 60};
    int depth = 32768;
    bool small = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--dense" && i + 1 < argc) dense = argv[++i];
        else if (a == "--golden" && i + 1 < argc) golden = argv[++i];
        else if (a == "--depth" && i + 1 < argc) depth = atoi(argv[++i]);
        else if (a == "--cases" && i + 1 < argc) only = argv[++i];
        else if (a == "--small") small = true;
        else if (a == "--heads" && i + 1 < argc)
        {
            heads.clear();
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) heads.push_back(atoi(t));
        }
    }
    std::vector<Case> cases;
    auto add = [&](const char* n, float al, float lo, float hi, float bm, float bs, bool corr, unsigned seed) {
        Case c; c.name = n; c.A_log = al; c.dt_lo = lo; c.dt_hi = hi; c.beta_mu = bm; c.beta_sd = bs; c.corr = corr; c.seed = seed;
        cases.push_back(c);
    };
    add("weak",         0.f, -5.f, -5.f, 0.f, 1.f, false, 11);
    add("strong",       0.f,  5.f,  5.f, 0.f, 1.f, false, 12);
    add("mid",          0.f,  0.f,  0.f, 0.f, 1.f, false, 13);
    add("mixed",        0.f, -6.f,  6.f, 0.f, 1.f, false, 14);
    add("mixed-hibeta", 0.f, -6.f,  6.f, 3.f, 0.5f, false, 15);
    add("nodecay",      0.f, -12.f, -12.f, 0.f, 1.f, false, 16);
    add("corr-weak",    0.f, -5.f, -5.f, 3.f, 0.5f, true, 17);
    add("corr-nodecay", 0.f, -12.f, -12.f, 3.f, 0.5f, true, 18);
    Golden* G = nullptr;
    if (!golden.empty() && !dense.empty())
    {
        G = new Golden;
        if (!G->f.open(golden)) { fprintf(stderr, "cannot open %s\n", golden.c_str()); return 2; }
        G->q = G->f["L00.kda.q"].f32(); G->k = G->f["L00.kda.k"].f32(); G->v = G->f["L00.kda.v"].f32();
        G->g = G->f["L00.kda.g"].f32(); G->beta = G->f["L00.kda.beta"].f32(); G->S4096 = G->f["L00.S.4096"].f32();
        G->w.load(dense, 0);
        for (int h : heads)
        {
            Case c; c.name = "golden-L00-h" + std::to_string(h); c.golden_head = h; c.seed = 200 + h;
            cases.push_back(c);
        }
    }
    if (!only.empty())
    {
        std::vector<Case> keep;
        for (auto& c : cases)
            if (("," + only + ",").find("," + c.name + ",") != std::string::npos) keep.push_back(c);
        cases = keep;
    }
    printf("G9 T1 CPU: ON emulation (CH16, bf16 RNE at the doc rounding points, fp32 state) vs OFF CPU model (fp32, prefill.h)\n");
    printf("and fp64 gated delta rule; depth %d rows of one slot, calls of %d rows, state carried. 'on-off' = ON vs OFF,\n", depth, CALL);
    printf("'on-64' = ON vs fp64, 'off-64' = OFF vs fp64. S = state [128][128]; O pre-gate fp32; y gated fp32; OH fp16.\n");
    printf("Output errors are rel-L2 over the window since the previous checkpoint.\n");
    std::vector<std::vector<Row>> res(cases.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < cases.size(); ++i)
        res[i] = run_case(cases[i], G, depth, i == 0);
    for (size_t i = 0; i < cases.size(); ++i)
    {
        printf("\ncase %s\n", cases[i].name.c_str());
        printf("  %-6s | %-23s | %-23s | %-23s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-9s %-9s %-9s\n", "depth", "S on-off  maxabs relL2",
               "S on-64   maxabs relL2", "S off-64  maxabs relL2", "O on-off", "O on-64", "O off-64", "y on-off", "y on-64", "y off-64", "OH on-off", "OH on-64", "OH off-64");
        for (size_t k = 0; k < res[i].size(); ++k)
        {
            const Row& R = res[i][k];
            printf("  %-6d | %-10.3e %-12.3e | %-10.3e %-12.3e | %-10.3e %-12.3e | %-9.2e %-9.2e %-9.2e | %-9.2e %-9.2e %-9.2e | %-9.2e %-9.2e %-9.2e\n",
                   CKPT[k] > depth ? depth : CKPT[k], R.s_on_off_mx, R.s_on_off, R.s_on_64_mx, R.s_on_64, R.s_off_64_mx, R.s_off_64, R.o_on_off,
                   R.o_on_64, R.o_off_64, R.y_on_off, R.y_on_64, R.y_off_64, R.oh_on_off, R.oh_on_64, R.oh_off_64);
            if (R.s_on_gold >= 0)
                printf("         vs golden fp32 S.4096 (python): ON %.3e  OFF %.3e\n", R.s_on_gold, R.s_off_gold);
            // bug detector, not a quality gate: ON must track fp64 within 5e-2 rel-L2 on S and O, OFF within 1e-3
            if (!(R.s_on_64 < 5e-2 && R.o_on_64 < 5e-2 && R.s_off_64 < 1e-3)) { ++g_bad; printf("         FAIL: outside bug-detector bounds\n"); }
        }
    }
    if (small) small_cases();
    printf("\n%s (%d failing checks)\n", g_bad ? "FAIL" : "PASS", g_bad);
    return g_bad ? 1 : 0;
}
