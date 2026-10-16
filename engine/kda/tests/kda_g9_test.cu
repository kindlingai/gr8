// T2 of docs/FLASH-G9-KDA-SCAN.md: the G9 ON kernels (kda_g9.cuh) against the OFF kernels / host math on synthetic input.
// Stage 1: conv16 and out16 against host math. Stage 2 (run_e2e): the whole ON path (kda::g9::launch, VS 1/2/4) against the
// OFF path (launch_core_prefill) from the same inputs and start state, with the scratch beyond M filled with NaN patterns.
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -o kda_g9_test kda_g9_test.cu
//   kda_g9_test [nh=2]          cases M in {1, 15, 17, 63, 8191}
// P is allocated with exactly M rows (as kda_pf_bench does) so a load past row M is an out-of-bounds read for compute-sanitizer.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../kda_g9.cuh"
#include "kda_g9_glue.h"

using namespace kda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

template <class T> static T* dalloc(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <class T> static T* up(const std::vector<T>& h) { T* d = dalloc<T>(h.size()); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static double rel2(const std::vector<float>& a, const std::vector<float>& b)
{
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = static_cast<double>(a[i]) - b[i]; d += e * e; r += static_cast<double>(b[i]) * b[i]; }
    return std::sqrt(d / (r > 0 ? r : 1));
}
static float h2f_host(uint16_t h)
{
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : e == 31 ? INFINITY : std::ldexp(static_cast<float>(1024 + m), e - 25);
    return s ? -v : v;
}

static int ulp_bf(uint16_t a, uint16_t b) { return std::abs(static_cast<int>(a) - static_cast<int>(b)); }   // same sign, finite

// Precision gate (KA): |got - ref| <= 1 ulp(ref) + 2^-9 * max|ref| of the row, ref in fp64 host math, ulp = spacing of the output
// format (mant = 7 bf16, 10 fp16). Raw ulp counts near zero are poor (cancellation), so they are printed as info only.
static double ulp_of(double ref, int mant)
{
    const double a = std::fabs(ref);
    return a > 0 ? std::ldexp(1.0, static_cast<int>(std::floor(std::log2(a))) - mant) : 0.0;
}
struct Gate
{
    long long bad = 0, n = 0; double worst = 0;   // worst = max err / allowed
    void add(double err, double ref, int mant, double rowmax)
    {
        const double allow = ulp_of(ref, mant) + std::ldexp(rowmax, -9);
        const double r = allow > 0 ? err / allow : (err > 0 ? 1e30 : 0.0);
        worst = std::max(worst, r); bad += r > 1.0; ++n;
    }
};

static bool run_case(int M, int nh)
{
    const int dl = nh * DK;
    PLayout pl; pl.dl = dl;
    const int ld = pl.ld();
    std::mt19937 rng(77 + M);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hP(static_cast<size_t>(M) * ld), hcst(static_cast<size_t>(NHIST) * 3 * dl);
    for (auto& x : hP) x = nd(rng);
    for (auto& x : hcst) x = nd(rng);
    std::vector<uint16_t> hconv[3], hon(DK);
    for (int s = 0; s < 3; ++s) { hconv[s].resize(static_cast<size_t>(dl) * CONV); for (auto& x : hconv[s]) x = f2bf_rn(0.5f * nd(rng)); }
    for (auto& x : hon) x = f2bf_rn(1.f + 0.1f * nd(rng));
    std::vector<int> hslot(M, 0);

    Args a;
    a.nh = nh; a.h0 = 0; a.M = M; a.nhmax = nh;
    const uint16_t* dconv[3];
    for (int s = 0; s < 3; ++s) { dconv[s] = up(hconv[s]); a.conv_w[s] = dconv[s]; }
    a.onorm = up(hon);
    a.row_slot = up(hslot);
    a.P = up(hP);
    float* cst0 = up(hcst);
    a.cst = dalloc<float>(hcst.size());
    CK(cudaMemcpy(a.cst, cst0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
    const int MP = (M + 63) / 64 * 64;   // the engine's MP % 64 == 0 scratch; the planes stay M-exact in what they are checked on
    a.C = dalloc<float>(static_cast<size_t>(MP) * 3 * dl);
    float* Cref = dalloc<float>(static_cast<size_t>(M) * 3 * dl);

    // ---- conv16 vs k_kda_conv_pf + host l2norm
    Args ar = a;
    ar.C = Cref;
    k_kda_conv_pf<<<dim3((3 * dl + 255) / 256, M), 256>>>(ar);
    CK(cudaDeviceSynchronize());
    kda::g9::G9Args g{a, kda::g9::planes(a, MP)};
    g.dbg = dalloc<float>(static_cast<size_t>(M) * 3 * dl);
    kda::g9::launch_conv16(g, 0);
    CK(cudaDeviceSynchronize());
    const auto C = down(Cref, static_cast<size_t>(M) * 3 * dl);
    const auto Pq = down(g.pl.p[kda::g9::QN], static_cast<size_t>(MP) * dl), Pk = down(g.pl.p[kda::g9::KN], static_cast<size_t>(MP) * dl),
               Pv = down(g.pl.p[kda::g9::VV], static_cast<size_t>(MP) * dl);
    // pre-rounding fp32 c of conv16 vs conv_pf's C, bitwise: first 16 mismatches, histograms by m%8 and by block (first / middle / last)
    long long cbad = 0;
    {
        const auto cd = down(g.dbg, static_cast<size_t>(M) * 3 * dl);
        long long h8[8] = {}, hb3[3] = {}, hs3[3] = {};
        const int lm0 = (M - 1) / 8 * 8;
        for (int m = 0; m < M; ++m)
            for (int s3 = 0; s3 < 3; ++s3)
                for (int ch = 0; ch < dl; ++ch)
                {
                    const size_t i = static_cast<size_t>(m) * 3 * dl + s3 * dl + ch;
                    if (std::memcmp(&cd[i], &C[i], 4) == 0) continue;
                    if (cbad++ < 16)
                        printf("  c32 mismatch m %d stream %d ch %d: got %08x ref %08x (%.9g vs %.9g) m%%8 %d m0 %d nh %d\n", m, s3, ch, *reinterpret_cast<const unsigned*>(&cd[i]),
                               *reinterpret_cast<const unsigned*>(&C[i]), cd[i], C[i], m % 8, m / 8 * 8, nh);
                    ++h8[m % 8]; ++hb3[m < 8 ? 0 : m >= lm0 ? 2 : 1]; ++hs3[s3];
                }
        printf("M %5d conv16 [gate] fp32 c bit mismatches vs conv_pf %lld  (hist m%%8: %lld %lld %lld %lld %lld %lld %lld %lld | first/middle/last block %lld %lld %lld | stream q/k/v %lld %lld %lld)\n",
               M, cbad, h8[0], h8[1], h8[2], h8[3], h8[4], h8[5], h8[6], h8[7], hb3[0], hb3[1], hb3[2], hs3[0], hs3[1], hs3[2]);
    }
    long long vbad = 0, qinfo = 0, qmax = 0, vshown = 0, qshown = 0, vhist8[8] = {}, vblk[3] = {};   // vblk: block m0==0, middle, last block
    Gate qg;
    const int lastm0 = (M - 1) / 8 * 8;
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < nh; ++h)
        {
            for (int s = 0; s < 2; ++s)
            {
                double ss = 0;
                for (int d = 0; d < DK; ++d) { const double x = C[static_cast<size_t>(m) * 3 * dl + s * dl + h * DK + d]; ss += x * x; }
                const double inv = 1.0 / std::sqrt(ss + static_cast<double>(L2_EPS));
                double rowmax = 0;
                for (int d = 0; d < DK; ++d) rowmax = std::max(rowmax, std::fabs(C[static_cast<size_t>(m) * 3 * dl + s * dl + h * DK + d] * inv));
                for (int d = 0; d < DK; ++d)
                {
                    const double ref = C[static_cast<size_t>(m) * 3 * dl + s * dl + h * DK + d] * inv;
                    const uint16_t got = (s ? Pk : Pq)[static_cast<size_t>(m) * dl + h * DK + d];
                    qg.add(std::fabs(bf2f(got) - ref), ref, 7, rowmax);
                    if (std::fabs(bf2f(got) - ref) > ulp_of(ref, 7) + std::ldexp(rowmax, -9) && qshown++ < 8)
                        printf("  q/k gate viol m %d %c ch %d: ref %.9g got %.9g |ref|/rowmax %.3e\n", m, s ? 'k' : 'q', h * DK + d, ref, bf2f(got), rowmax > 0 ? std::fabs(ref) / rowmax : 0.0);
                    const int u = ulp_bf(f2bf_rn(static_cast<float>(ref)), got);   // info only
                    qmax = std::max<long long>(qmax, u);
                    qinfo += u > 1 && std::fabs(ref) > std::ldexp(rowmax, -6);
                }
            }
            for (int d = 0; d < DK; ++d)
            {
                const size_t i = static_cast<size_t>(m) * dl + h * DK + d;
                const uint16_t want = f2bf_rn(C[static_cast<size_t>(m) * 3 * dl + 2 * dl + h * DK + d]);
                if (Pv[i] != want)
                {
                    ++vbad; ++vhist8[m % 8]; ++vblk[m < 8 ? 0 : m >= lastm0 ? 2 : 1];
                    if (vshown++ < 8)
                        printf("  V mismatch m %d (m%%8 %d, m0 %d) ch %d: C %.9g (bits %08x) want bf16 %04x got %04x\n", m, m % 8, m / 8 * 8, h * DK + d, C[static_cast<size_t>(m) * 3 * dl + 2 * dl + h * DK + d],
                               *reinterpret_cast<const unsigned*>(&C[static_cast<size_t>(m) * 3 * dl + 2 * dl + h * DK + d]), want, Pv[i]);
                }
            }
        }
    // history written by k_kda_conv_end_pf = last 3 rows of P (rows < 3 of M: older history)
    const auto cst = down(a.cst, hcst.size());
    long long hbad = 0;
    for (int j = 0; j < NHIST; ++j)
        for (int s = 0; s < 3; ++s)
            for (int c = 0; c < dl; ++c)
            {
                const int r = M - NHIST + j;
                const float ref = r >= 0 ? hP[static_cast<size_t>(r) * ld + s * dl + c] : hcst[a.c_off(0, NHIST + r, s, c)];
                hbad += cst[a.c_off(0, j, s, c)] != ref;
            }
    printf("M %5d conv16 [gate]: q/k fp64 gate viol %lld/%lld (worst %.3f of allowed), V bit mismatches vs conv_pf %lld, history %lld\n", M, qg.bad, qg.n, qg.worst, vbad, hbad);
    if (vbad)
        printf("M %5d conv16 V mismatch histogram: by m%%8 %lld %lld %lld %lld %lld %lld %lld %lld | by block first/middle/last %lld %lld %lld  (M %d nh %d)\n", M, vhist8[0], vhist8[1],
               vhist8[2], vhist8[3], vhist8[4], vhist8[5], vhist8[6], vhist8[7], vblk[0], vblk[1], vblk[2], M, nh);
    printf("M %5d conv16 [info]: q/k raw >1ulp %lld where |ref| > 2^-6 rowmax (max %lld ulp over all elements, vs bf16(fp64 ref))\n", M, qinfo, qmax);

    // ---- out16 vs host math of out_2, OH and OF
    std::vector<uint16_t> hO(static_cast<size_t>(MP) * dl), hZ(hO.size());
    for (auto& x : hO) x = f2bf_rn(nd(rng));
    for (auto& x : hZ) x = f2bf_rn(2.f * nd(rng));
    CK(cudaMemcpy(g.pl.p[kda::g9::O16], hO.data(), hO.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.pl.p[kda::g9::Z16], hZ.data(), hZ.size() * 2, cudaMemcpyHostToDevice));
    const size_t nof = static_cast<size_t>((M + 7) / 8) * dl * 8;
    Args ao = a;
    ao.OH = dalloc<uint16_t>(static_cast<size_t>(M) * dl);      // exactly M rows
    ao.OF = dalloc<uint16_t>(nof);
    kda::g9::launch_out16(kda::g9::G9Args{ao, g.pl}, 0);
    CK(cudaDeviceSynchronize());
    const auto OH = down(ao.OH, static_cast<size_t>(M) * dl), OF = down(ao.OF, nof);
    Gate hg, fg;
    long long hinfo = 0, finfo = 0, hmax = 0;
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < nh; ++h)
        {
            double ss = 0;
            for (int d = 0; d < DK; ++d) { const double o = bf2f(hO[static_cast<size_t>(m) * dl + h * DK + d]); ss += o * o; }
            const double r = 1.0 / std::sqrt(ss / DV + static_cast<double>(RMS_EPS));
            std::vector<double> y(DK);
            double rowmax = 0;
            for (int d = 0; d < DK; ++d)
            {
                const size_t i = static_cast<size_t>(m) * dl + h * DK + d;
                const double z = bf2f(hZ[i]);
                y[d] = bf2f(hO[i]) * r * bf2f(hon[d]) * (1.0 / (1.0 + std::exp(-z)));
                rowmax = std::max(rowmax, std::fabs(y[d]));
            }
            for (int d = 0; d < DK; ++d)
            {
                const size_t i = static_cast<size_t>(m) * dl + h * DK + d;
                const float of = bf2f(OF[frag_u16(m, h * DK + d, dl)]);
                hg.add(std::fabs(h2f_host(OH[i]) - y[d]), y[d], 10, rowmax);
                fg.add(std::fabs(of - y[d]), y[d], 7, rowmax);
                const int uh = ulp_bf(f2h_rn(static_cast<float>(y[d])), OH[i]), uf = ulp_bf(f2bf_rn(static_cast<float>(y[d])), OF[frag_u16(m, h * DK + d, dl)]);
                hmax = std::max<long long>(hmax, std::max(uh, uf));
                hinfo += uh > 1; finfo += uf > 1;
            }
        }
    printf("M %5d out16 [gate]: OH fp64 gate viol %lld/%lld (worst %.3f), OF viol %lld/%lld (worst %.3f)\n", M, hg.bad, hg.n, hg.worst, fg.bad, fg.n, fg.worst);
    printf("M %5d out16 [info]: OH raw >1ulp %lld, OF raw >1ulp %lld (max %lld ulp)\n", M, hinfo, finfo, hmax);
    return cbad == 0 && vbad == 0 && qg.bad == 0 && hbad == 0 && hg.bad == 0 && fg.bad == 0;
}

// ON (kda::g9::launch) vs OFF (launch_core_prefill) on one slot of M rows, synthetic input like kda_pf_bench. OFF is the reference:
// S, OG (fp32 gated o), and the conv history. ON writes OH (fp16) and OF (bf16 fragments). Every ON scratch byte is first set to
// poison (0xFF: NaN as bf16 and fp32); a second ON run with zero-filled scratch must be bit-identical (no read of rows >= M).
static bool run_e2e(int M, int nh, int VS)
{
    const int dl = nh * DK;
    PLayout pl; pl.dl = dl;
    const int ld = pl.ld();
    std::mt19937 rng(1234 + M);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [](float f) { return f2bf_rn(f); };
    std::vector<float> hP(static_cast<size_t>(M) * ld);
    for (auto& x : hP) x = nd(rng);
    std::vector<uint16_t> hconv[3], hfb(static_cast<size_t>(dl) * DK), hgb(hfb.size()), hon(DK);
    for (int s = 0; s < 3; ++s) { hconv[s].resize(static_cast<size_t>(dl) * CONV); for (auto& x : hconv[s]) x = bf(0.5f * nd(rng)); }
    for (auto& x : hfb) x = bf(0.15f * nd(rng));
    for (auto& x : hgb) x = bf(0.1f * nd(rng));
    for (auto& x : hon) x = bf(1.f + 0.1f * nd(rng));
    std::vector<float> hAlog(nh), hdt(dl), hS(static_cast<size_t>(nh) * DK * DV), hcst(static_cast<size_t>(NHIST) * 3 * dl);
    for (auto& x : hAlog) x = 0.5f * nd(rng);
    for (auto& x : hdt) x = -2.f + nd(rng);
    for (auto& x : hS) x = 0.05f * nd(rng);
    for (auto& x : hcst) x = nd(rng);
    std::vector<int> hslot(M, 0);

    Args a;
    a.nh = nh; a.h0 = 0; a.M = M; a.nhmax = nh;
    const uint16_t* dconv[3];
    for (int s = 0; s < 3; ++s) { dconv[s] = up(hconv[s]); a.conv_w[s] = dconv[s]; }
    a.fb = up(hfb); a.gb = up(hgb); a.onorm = up(hon); a.A_log = up(hAlog); a.dt_bias = up(hdt);
    a.row_slot = up(hslot);
    a.P = up(hP);
    float* S0 = up(hS);
    float* c0 = up(hcst);
    a.S = dalloc<float>(hS.size()); a.cst = dalloc<float>(hcst.size());
    auto reset = [&] {
        CK(cudaMemcpy(a.S, S0, hS.size() * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(a.cst, c0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
    };
    const int MP = (M + 63) / 64 * 64;
    KdaPrefillScratch ps;
    if (const char* e = build_prefill_scratch(ps, MP, 1)) { fprintf(stderr, "%s\n", e); exit(3); }

    // OFF
    Args ar = a;
    ar.C = dalloc<float>(static_cast<size_t>(M) * 3 * dl); ar.G = dalloc<float>(static_cast<size_t>(M) * dl);
    ar.Z = dalloc<float>(static_cast<size_t>(M) * dl); ar.O = dalloc<float>(static_cast<size_t>(M) * dl);
    ar.OG = dalloc<float>(static_cast<size_t>(M) * dl);
    reset();
    launch_core_prefill(make_pf_args(ar, ps), 0);
    CK(cudaDeviceSynchronize());
    const auto rOG = down(ar.OG, static_cast<size_t>(M) * dl), rS = down(a.S, hS.size()), rc = down(a.cst, hcst.size());

    // ON, twice: poisoned scratch, zero scratch
    std::vector<float> oS[2], oH[2], oF[2], oc[2];
    const size_t nof = static_cast<size_t>((M + 7) / 8) * dl * 8;
    a.C = dalloc<float>(static_cast<size_t>(MP) * 3 * dl);
    a.OH = dalloc<uint16_t>(static_cast<size_t>(M) * dl);   // exactly M rows
    a.OF = dalloc<uint16_t>(nof);
    std::vector<uint16_t> bitsH[2], bitsF[2];
    for (int pass = 0; pass < 2; ++pass)
    {
        const int fill = pass == 0 ? 0xFF : 0x00;
        CK(cudaMemset(a.C, fill, static_cast<size_t>(MP) * 3 * dl * 4));
        float* sc[] = {ps.QE, ps.KD, ps.W, ps.U};
        for (float* f : sc) CK(cudaMemset(f, fill, static_cast<size_t>(MP) * nhmax(1) * DK * 4));
        CK(cudaMemset(a.OH, 0, static_cast<size_t>(M) * dl * 2));
        CK(cudaMemset(a.OF, 0, nof * 2));
        reset();
        if (const char* e = kda::g9::launch(a, ps, VS, 0)) { fprintf(stderr, "%s\n", e); exit(3); }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        oS[pass] = down(a.S, hS.size()); oc[pass] = down(a.cst, hcst.size());
        bitsH[pass] = down(a.OH, static_cast<size_t>(M) * dl); bitsF[pass] = down(a.OF, nof);
    }
    std::vector<float> H(static_cast<size_t>(M) * dl), F(H.size());
    long long nonfinite = 0;
    for (int m = 0; m < M; ++m)
        for (int c = 0; c < dl; ++c)
        {
            const size_t i = static_cast<size_t>(m) * dl + c;
            H[i] = h2f_host(bitsH[0][i]); F[i] = bf2f(bitsF[0][frag_u16(m, c, dl)]);
            nonfinite += !std::isfinite(H[i]) || !std::isfinite(F[i]);
        }
    for (float x : oS[0]) nonfinite += !std::isfinite(x);
    const bool same = oS[0] == oS[1] && bitsH[0] == bitsH[1] && bitsF[0] == bitsF[1] && oc[0] == oc[1];
    const double es = rel2(oS[0], rS), eh = rel2(H, rOG), ef = rel2(F, rOG), ec = rel2(oc[0], rc);

    // BUG GATE (a): GPU ON vs the CPU emulation of ON (kda_g9_cpu.h, doc 6/T2 <= 2e-3), per head. The emulation takes the GPU's own G16/Z16 planes
    // (gz16 output), the OFF conv output C (bitwise what conv16 should round from), and b_proj from P; it models K1 + K2 + out16.
    const auto pls = kda::g9::planes(a, MP);
    const auto pG = down(pls.p[kda::g9::G16], static_cast<size_t>(MP) * dl), pZ = down(pls.p[kda::g9::Z16], static_cast<size_t>(MP) * dl);
    const auto Cm = down(ar.C, static_cast<size_t>(M) * 3 * dl);
    std::vector<float> eS, eH, gH(static_cast<size_t>(M) * dl);
    g9glue::emu_on_heads(a, M, nh, Cm, pG, pZ, hP.data(), ld, pl.b(), hAlog, hdt, hon, hS, eS, eH);
    for (size_t i = 0; i < gH.size(); ++i) gH[i] = H[i];
    const double ems = rel2(oS[0], eS), emh = rel2(gH, eH);
    printf("M %5d nh %d VS %d [report] GPU ON vs OFF (FP32 path): S rel %.3e  OH rel %.3e  OF rel %.3e  conv hist rel %.3e  nonfinite %lld  poison-vs-zero %s\n", M, nh, VS, es, eh, ef,
           ec, nonfinite, same ? "identical" : "DIFFERENT");
    printf("M %5d nh %d VS %d [bug gate <= 2e-3] GPU ON vs CPU ON-emulation: S rel %.3e  OH rel %.3e\n", M, nh, VS, ems, emh);
    return es < 2e-2 && eh < 2e-2 && ef < 2e-2 && ec < 1e-6 && nonfinite == 0 && same && ems <= 2e-3 && emh <= 2e-3;
}

// Carried depth (T2 record): the same synthetic row stream (16 sub-batches of 2048 rows, fresh rows each, 32768 in all) through
// OFF (FP32 path) and ON, each carrying its own S and conv history across sub-batches. Prints S rel-L2 ON vs OFF at 2K / 8K / 32K rows.
// Synthetic N(0,1) input, so the numbers are a GPU cross-check of the CPU 32K-depth test (real KDA-L00 input), not a replacement.
static bool run_depth(int nh, int VS)
{
    const int dl = nh * DK, MS = 2048, NB = 16;
    PLayout pl; pl.dl = dl;
    const int ld = pl.ld();
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [](float f) { return f2bf_rn(f); };
    std::vector<uint16_t> hconv[3], hfb(static_cast<size_t>(dl) * DK), hgb(hfb.size()), hon(DK);
    for (int s = 0; s < 3; ++s) { hconv[s].resize(static_cast<size_t>(dl) * CONV); for (auto& x : hconv[s]) x = bf(0.5f * nd(rng)); }
    for (auto& x : hfb) x = bf(0.15f * nd(rng));
    for (auto& x : hgb) x = bf(0.1f * nd(rng));
    for (auto& x : hon) x = bf(1.f + 0.1f * nd(rng));
    std::vector<float> hAlog(nh), hdt(dl), hS(static_cast<size_t>(nh) * DK * DV), hcst(static_cast<size_t>(NHIST) * 3 * dl);
    for (auto& x : hAlog) x = 0.5f * nd(rng);
    for (auto& x : hdt) x = -2.f + nd(rng);
    for (auto& x : hS) x = 0.05f * nd(rng);
    for (auto& x : hcst) x = nd(rng);
    std::vector<int> hslot(MS, 0);
    std::vector<float> hP(static_cast<size_t>(MS) * ld);

    Args a;
    a.nh = nh; a.h0 = 0; a.M = MS; a.nhmax = nh;
    const uint16_t* dconv[3];
    for (int s = 0; s < 3; ++s) { dconv[s] = up(hconv[s]); a.conv_w[s] = dconv[s]; }
    a.fb = up(hfb); a.gb = up(hgb); a.onorm = up(hon); a.A_log = up(hAlog); a.dt_bias = up(hdt);
    a.row_slot = up(hslot);
    float* dP = dalloc<float>(hP.size());
    a.P = dP;
    a.S = up(hS); a.cst = up(hcst);
    a.C = dalloc<float>(static_cast<size_t>(MS) * 3 * dl);
    a.OH = dalloc<uint16_t>(static_cast<size_t>(MS) * dl);
    a.OF = dalloc<uint16_t>(static_cast<size_t>((MS + 7) / 8) * dl * 8);
    Args ar = a;
    ar.S = up(hS); ar.cst = up(hcst);
    ar.C = dalloc<float>(static_cast<size_t>(MS) * 3 * dl); ar.G = dalloc<float>(static_cast<size_t>(MS) * dl);
    ar.Z = dalloc<float>(static_cast<size_t>(MS) * dl); ar.O = dalloc<float>(static_cast<size_t>(MS) * dl);
    ar.OG = dalloc<float>(static_cast<size_t>(MS) * dl);
    KdaPrefillScratch ps;
    if (const char* e = build_prefill_scratch(ps, MS, 1)) { fprintf(stderr, "%s\n", e); exit(3); }
    bool ok = true;
    for (int b = 0; b < NB; ++b)
    {
        for (auto& x : hP) x = nd(rng);
        CK(cudaMemcpy(dP, hP.data(), hP.size() * 4, cudaMemcpyHostToDevice));
        launch_core_prefill(make_pf_args(ar, ps), 0);
        if (const char* e = kda::g9::launch(a, ps, VS, 0)) { fprintf(stderr, "%s\n", e); exit(3); }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const int rows = (b + 1) * MS;
        if (rows == 2048 || rows == 8192 || rows == 32768)
        {
            const auto sOn = down(a.S, hS.size()), sOff = down(ar.S, hS.size());
            long long nf = 0;
            for (float x : sOn) nf += !std::isfinite(x);
            const double e = rel2(sOn, sOff);
            printf("[depth] nh %d VS %d carried %5d rows: S rel-L2 ON vs FP32 path %.3e  nonfinite %lld\n", nh, VS, rows, e, nf);
            ok = ok && e < 2e-2 && nf == 0;
        }
    }
    return ok;
}

int main(int argc, char** argv)
{
    const int nh = argc > 1 ? atoi(argv[1]) : 2;
    bool ok = true;
    int npass = 0, ncase = 0;
    auto tally = [&](bool r) { ++ncase; npass += r; ok = r && ok; };
    for (int M : {1, 15, 17, 63, 8191}) tally(run_case(M, nh));
    for (int M : {1, 15, 17, 63, 8191})
        for (int VS : {1, 2, 4}) tally(run_e2e(M, nh, VS));
    tally(run_depth(4, 2));
    printf("%s %d/%d\n", ok ? "PASS" : "FAIL", npass, ncase);
    return ok ? 0 : 1;
}
