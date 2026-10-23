// GPU-free test of the glue that kda_g9_test.cu uses to feed the CPU ON-emulation (kda_g9_glue.h, the SAME function), on nh = 2 heads in the
// GPU test's layouts, M in {63, 200} (chunk tails). Checks:
//   1. head independence: the nh = 2 result of head h is BIT-identical to the nh = 1 glue run on head h's own data (catches wrong head slice,
//      s_off, b_proj column, plane offset).
//   2. ON-emulation vs CPU-OFF (kda_cpu, the base kernels' CPU model, same inputs): S and OH rel-L2 in the expected band (nonzero, <= 1e-2;
//      T1 range for 2048-row calls is 1.4e-3..8.8e-3).
//   3. ON-emulation vs fp64 reference (ref64_call) per head: S rel-L2 <= 1e-2.
//   4. negative control: head 1's G16 plane fed to head 0 must push the glue-vs-OFF error of head 0 above the 2e-3 bug gate, so a wrong
//      slice would be caught by that gate.
//   g++ -O3 -march=native -fopenmp -std=c++17 -DG6_MODEL_FLASH=1 kda_g9_glue_test.cpp
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "kda_cpu.h"
#include "kda_g9_glue.h"

using namespace kda;

static double rel2(const std::vector<float>& a, const std::vector<float>& b)
{
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = double(a[i]) - b[i]; d += e * e; r += double(b[i]) * b[i]; }
    return std::sqrt(d / (r > 0 ? r : 1));
}
static int g_bad = 0;
static void check(bool ok, const char* what) { printf("  %-4s %s\n", ok ? "PASS" : "FAIL", what); g_bad += !ok; }

struct In
{
    int M, nh, dl, ld;
    std::vector<float> Cm, fraw, z, P, hAlog, hdt, hS;
    std::vector<uint16_t> pG, pZ, hon;
};

static In gen(int M, int nh, unsigned seed)
{
    In d; d.M = M; d.nh = nh; d.dl = nh * DK;
    PLayout pl; pl.dl = d.dl; d.ld = pl.ld();
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    d.Cm.resize(static_cast<size_t>(M) * 3 * d.dl);
    for (auto& x : d.Cm) x = silu(nd(rng));
    d.fraw.resize(static_cast<size_t>(M) * d.dl); d.z = d.fraw;
    for (auto& x : d.fraw) x = nd(rng);
    for (auto& x : d.z) x = 1.5f * nd(rng);
    d.pG.resize(d.fraw.size()); d.pZ.resize(d.fraw.size());
    for (size_t i = 0; i < d.fraw.size(); ++i) { d.pG[i] = f2bf_rn(d.fraw[i]); d.pZ[i] = f2bf_rn(d.z[i]); }
    d.P.assign(static_cast<size_t>(M) * d.ld, 0.f);
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < nh; ++h) d.P[static_cast<size_t>(m) * d.ld + pl.b() + h] = nd(rng);
    d.hAlog.resize(nh); d.hdt.resize(d.dl); d.hS.resize(static_cast<size_t>(nh) * DK * DV); d.hon.resize(DK);
    for (auto& x : d.hAlog) x = 0.5f * nd(rng);
    for (auto& x : d.hdt) x = -2.f + nd(rng);
    for (auto& x : d.hS) x = 0.3f * nd(rng);
    for (auto& x : d.hon) x = f2bf_rn(1.f + 0.1f * nd(rng));
    return d;
}

// the nh = 1 input of head h of d, in the nh = 1 layouts
static In slice(const In& d, int h)
{
    In s; s.M = d.M; s.nh = 1; s.dl = DK;
    PLayout pl; pl.dl = DK; s.ld = pl.ld();
    s.Cm.resize(static_cast<size_t>(d.M) * 3 * DK);
    s.pG.resize(static_cast<size_t>(d.M) * DK); s.pZ = s.pG;
    s.P.assign(static_cast<size_t>(d.M) * s.ld, 0.f);
    PLayout pd; pd.dl = d.dl;
    for (int m = 0; m < d.M; ++m)
    {
        for (int s3 = 0; s3 < 3; ++s3)
            for (int k = 0; k < DK; ++k) s.Cm[(static_cast<size_t>(m) * 3 + s3) * DK + k] = d.Cm[static_cast<size_t>(m) * 3 * d.dl + s3 * d.dl + h * DK + k];
        for (int k = 0; k < DK; ++k)
        {
            s.pG[static_cast<size_t>(m) * DK + k] = d.pG[static_cast<size_t>(m) * d.dl + h * DK + k];
            s.pZ[static_cast<size_t>(m) * DK + k] = d.pZ[static_cast<size_t>(m) * d.dl + h * DK + k];
        }
        s.P[static_cast<size_t>(m) * s.ld + pl.b()] = d.P[static_cast<size_t>(m) * d.ld + pd.b() + h];
    }
    s.hAlog = {d.hAlog[h]};
    s.hdt.assign(d.hdt.begin() + h * DK, d.hdt.begin() + (h + 1) * DK);
    s.hS.assign(d.hS.begin() + static_cast<size_t>(h) * DK * DV, d.hS.begin() + static_cast<size_t>(h + 1) * DK * DV);
    s.hon = d.hon;
    return s;
}

static void glue(const In& d, std::vector<float>& eS, std::vector<float>& eH)
{
    Args a; a.nh = d.nh; a.h0 = 0; a.M = d.M; a.nhmax = d.nh;
    PLayout pl; pl.dl = d.dl;
    g9glue::emu_on_heads(a, d.M, d.nh, d.Cm, d.pG, d.pZ, d.P.data(), d.ld, pl.b(), d.hAlog, d.hdt, d.hon, d.hS, eS, eH);
}

int main()
{
    for (int M : {63, 200})
    {
        printf("M %d nh 2\n", M);
        const In d = gen(M, 2, 100 + M);
        std::vector<float> eS, eH;
        glue(d, eS, eH);

        // 1. head independence, bitwise
        bool same = true;
        std::vector<In> sl;
        for (int h = 0; h < 2; ++h)
        {
            sl.push_back(slice(d, h));
            std::vector<float> s1, h1;
            glue(sl[h], s1, h1);
            same = same && !memcmp(s1.data(), &eS[static_cast<size_t>(h) * DK * DV], s1.size() * 4);
            for (int m = 0; m < M && same; ++m)
                same = !memcmp(&h1[static_cast<size_t>(m) * DK], &eH[static_cast<size_t>(m) * d.dl + h * DK], DK * 4);
        }
        check(same, "head h of the nh=2 glue == nh=1 glue on head h's own data (bit-identical S and OH)");

        // 2. CPU OFF on the same inputs (nh = 2)
        std::vector<float> g(d.fraw.size());
        for (int m = 0; m < M; ++m)
            for (int h = 0; h < 2; ++h)
                for (int k = 0; k < DK; ++k)
                {
                    const size_t i = static_cast<size_t>(m) * d.dl + h * DK + k;
                    g[i] = GATE_LO * sigm(std::exp(d.hAlog[h]) * (d.fraw[i] + d.hdt[h * DK + k]));
                }
        std::vector<float> S = d.hS, Cc = d.Cm, zz = d.z, P = d.P, O(g.size()), OG(g.size()), QE(g.size()), KD(g.size()), PW(g.size()), PU(g.size()), GC(g.size());
        std::vector<float> AQ(static_cast<size_t>(M) * 2 * CH);
        std::vector<uint16_t> OH(g.size());
        std::vector<int> rs(M, 0);
        Args a; a.nh = 2; a.h0 = 0; a.M = M; a.nhmax = 2;
        a.onorm = d.hon.data(); a.S = S.data(); a.row_slot = rs.data(); a.P = P.data(); a.C = Cc.data(); a.G = g.data(); a.Z = zz.data();
        a.O = O.data(); a.OG = OG.data(); a.OH = OH.data();
        PfArgs p; p.a = a; p.QE = QE.data(); p.KD = KD.data(); p.W = PW.data(); p.U = PU.data(); p.GC = GC.data(); p.AQ = AQ.data();
        kda_cpu::prefill(p);
        kda_cpu::out(a);
        std::vector<float> ohOff(OH.size());
        for (size_t i = 0; i < OH.size(); ++i) ohOff[i] = g9cpu::h2f(OH[i]);
        const double es = rel2(eS, S), eh = rel2(eH, ohOff);
        printf("  info: ON-emulation vs CPU-OFF: S rel %.3e  OH rel %.3e\n", es, eh);
        check(es > 1e-6 && es <= 1e-2 && eh > 1e-6 && eh <= 1e-2, "ON-emulation vs CPU-OFF in the expected band (1e-6 < rel <= 1e-2)");

        // 3. fp64 reference per head
        bool ok64 = true;
        for (int h = 0; h < 2; ++h)
        {
            const In& s = sl[h];
            std::vector<float> gh(static_cast<size_t>(M) * DK), zh(gh.size()), onf(DK), braw(M);
            for (int m = 0; m < M; ++m)
            {
                for (int k = 0; k < DK; ++k)
                {
                    gh[static_cast<size_t>(m) * DK + k] = g[static_cast<size_t>(m) * d.dl + h * DK + k];
                    zh[static_cast<size_t>(m) * DK + k] = d.z[static_cast<size_t>(m) * d.dl + h * DK + k];
                }
                PLayout pl1; pl1.dl = DK;
                braw[m] = s.P[static_cast<size_t>(m) * s.ld + pl1.b()];
            }
            for (int k = 0; k < DK; ++k) onf[k] = bf2f(d.hon[k]);
            std::vector<double> S64(s.hS.begin(), s.hS.end()), o64(gh.size()), y64(gh.size());
            g9cpu::ref64_call(M, s.Cm.data(), gh.data(), zh.data(), braw.data(), onf.data(), S64.data(), o64.data(), y64.data());
            double dd = 0, rr = 0;
            for (size_t i = 0; i < S64.size(); ++i) { const double e = eS[static_cast<size_t>(h) * DK * DV + i] - S64[i]; dd += e * e; rr += S64[i] * S64[i]; }
            const double r = std::sqrt(dd / rr);
            printf("  info: head %d ON-emulation vs fp64: S rel %.3e\n", h, r);
            ok64 = ok64 && r <= 1e-2;
        }
        check(ok64, "ON-emulation vs fp64 reference per head, S rel <= 1e-2");

        // 4. negative control: wrong slice (head 1's G16 into head 0) must exceed the 2e-3 bug gate vs CPU-OFF on head 0
        In bad = d;
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < DK; ++k) bad.pG[static_cast<size_t>(m) * d.dl + k] = d.pG[static_cast<size_t>(m) * d.dl + DK + k];
        std::vector<float> bS, bH;
        glue(bad, bS, bH);
        const std::vector<float> b0(bS.begin(), bS.begin() + DK * DV), o0(S.begin(), S.begin() + DK * DV);
        const double good0 = rel2(std::vector<float>(eS.begin(), eS.begin() + DK * DV), o0), bad0 = rel2(b0, o0);
        printf("  info: head 0 S rel vs CPU-OFF: correct slice %.3e, wrong G16 slice %.3e\n", good0, bad0);
        check(bad0 > 2e-3 && bad0 > 3 * good0, "negative control: a wrong G16 slice is flagged by the 2e-3 gate (>= 3x the correct-slice error)");
    }
    printf("%s\n", g_bad ? "FAIL" : "PASS");
    return g_bad ? 1 : 0;
}
