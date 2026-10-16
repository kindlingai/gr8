// Determinism / sanitizer gate for the G9 ON kernels (order after the first T2: conv16 M=8191 failed once, then was bit-exact,
// cause unknown). Same inputs as kda_g9_test (seeds 77+M conv16 case, 1234+M e2e case).
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG9_CONV_DBG=1 -o kda_g9_det kda_g9_det.cu
//   Built a second time WITHOUT -DG9_CONV_DBG (the engine kernel has no dbg store): then the fp32 c compare is skipped, everything else is kept.
//   kda_g9_det [nh=2] [reps=20] [arm=both|conv|e2e] [san=0|1] [neg=0|1]
// det (san=0): REPS runs of conv16 (M=8191) from identical inputs, planes + dbg buffer re-poisoned with a different pattern each rep
//   (0xFF, 0x00, 0xA5, 0x5A, then LCG noise), conv history reset each rep. Every rep must (a) have fp32 c bits == k_kda_conv_pf, (b) V == bf16(c),
//   (c) leave rows >= M of the planes at the poison, (d) hash equal to rep 0 (q, k, v, c, new history; FNV-1a 64 + memcmp).
//   Then a stress set: 200 back-to-back conv16 launches (stream-ordered poison memset + history memcpy only, no host work), result hashed after every 10th.
//   Then REPS runs of the whole ON path (VS 2) with scratch re-poisoned each rep: OH, OF, S, history, planes (rows < M) hash equal to rep 0.
// san (san=1): one rep, planes and dbg allocated WITHOUT memset (initcheck sees reads of unwritten memory; only rows < M are copied back);
//   run under compute-sanitizer --tool racecheck / initcheck / memcheck. Exit code = result.
// Last line: "DET PASS k/k" or "DET FAIL ...".
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../kda_g9.cuh"
#include "kda_g9_glue.h"

using namespace kda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

static bool g_san = false, g_neg = false;   // g_neg: negative control, flips one q bit in conv rep 3 on the host; the run MUST report DET FAIL
template <class T> static T* dalloc(size_t n, bool zero = true) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); if (zero) CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <class T> static T* up(const std::vector<T>& h) { T* d = dalloc<T>(h.size()); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull)
{
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
template <class T> static uint64_t fnv(const std::vector<T>& v, uint64_t h = 1469598103934665603ull) { return fnv(v.data(), v.size() * sizeof(T), h); }
template <class T> static bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * sizeof(T)); }

// poison byte pattern for rep r: 0xFF, 0x00, 0xA5, 0x5A, then a per-rep byte from an LCG (host fill via memset of one byte)
static int poison_byte(int r) { static const int f[4] = {0xFF, 0x00, 0xA5, 0x5A}; return r < 4 ? f[r] : static_cast<int>((1103515245u * (r + 7) + 12345u) >> 16 & 0xFF); }

static int nfail = 0, ntest = 0;
static void check(bool ok, const char* what, int rep)
{
    ++ntest;
    if (!ok) { ++nfail; printf("  FAIL rep %d: %s\n", rep, what); }
}

static void conv_arm(int M, int nh, int reps)
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
    a.P = up(hP);                                  // exactly M rows
    float* cst0 = up(hcst);
    a.cst = dalloc<float>(hcst.size());
    const int MP = (M + 63) / 64 * 64;
    const size_t nC = static_cast<size_t>(MP) * 3 * dl;   // floats; six bf16 planes of MP rows alias it
    a.C = dalloc<float>(nC, !g_san);
    float* Cref = dalloc<float>(static_cast<size_t>(M) * 3 * dl);
    float* dbg = dalloc<float>(static_cast<size_t>(M) * 3 * dl, !g_san);
    kda::g9::G9Args g{a, kda::g9::planes(a, MP)};
    g.dbg = dbg;

    // reference (OFF kernel), twice: it must itself be deterministic
    CK(cudaMemcpy(a.cst, cst0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
    Args ar = a; ar.C = Cref;
    k_kda_conv_pf<<<dim3((3 * dl + 255) / 256, M), 256>>>(ar);
    CK(cudaDeviceSynchronize());
    const auto C0 = down(Cref, static_cast<size_t>(M) * 3 * dl);
    CK(cudaMemset(Cref, 0xFF, static_cast<size_t>(M) * 3 * dl * 4));
    k_kda_conv_pf<<<dim3((3 * dl + 255) / 256, M), 256>>>(ar);
    CK(cudaDeviceSynchronize());
    check(same(C0, down(Cref, static_cast<size_t>(M) * 3 * dl)), "OFF conv_pf not repeatable", -1);
    #ifdef G9_CONV_DBG
    printf("conv arm: M %d nh %d reps %d san %d  dbg build (fp32 c compared)  OFF conv_pf c hash %016llx\n", M, nh, reps, g_san, (unsigned long long)fnv(C0));
#else
    printf("conv arm: M %d nh %d reps %d san %d  NO-DBG build (engine kernel, fp32 c not compared)  OFF conv_pf c hash %016llx\n", M, nh, reps, g_san, (unsigned long long)fnv(C0));
#endif

    std::vector<uint16_t> rq, rk, rv; std::vector<float> rc, rh; uint64_t h0 = 0;
    for (int r = 0; r < reps; ++r)
    {
        if (!g_san)
        {
            const int pb = poison_byte(r);
            CK(cudaMemset(a.C, pb, nC * 4));
            CK(cudaMemset(dbg, pb, static_cast<size_t>(M) * 3 * dl * 4));
        }
        CK(cudaMemcpy(a.cst, cst0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
        kda::g9::launch_conv16(g, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        // rows < M of each plane (the planes are MP rows apart, the engine reads rows < M only)
        std::vector<uint16_t> q = down(g.pl.p[kda::g9::QN], static_cast<size_t>(M) * dl), k = down(g.pl.p[kda::g9::KN], static_cast<size_t>(M) * dl),
                              v = down(g.pl.p[kda::g9::VV], static_cast<size_t>(M) * dl);
        if (g_neg && r == 3) q[12345] ^= 1;
#ifdef G9_CONV_DBG
        const std::vector<float> c = down(dbg, static_cast<size_t>(M) * 3 * dl);
#else
        const std::vector<float> c;   // no dbg store in this build
#endif
        const std::vector<float> hist = down(a.cst, hcst.size());
        long long cbad = 0, vbad = 0;
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < 3; ++j)
                for (int ch = 0; ch < dl; ++ch)
                {
                    const size_t i = static_cast<size_t>(m) * 3 * dl + j * dl + ch;
#ifdef G9_CONV_DBG
                    uint32_t x, y; memcpy(&x, &c[i], 4); memcpy(&y, &C0[i], 4);
                    cbad += x != y;
#endif
                    if (j == 2) vbad += v[static_cast<size_t>(m) * dl + ch] != f2bf_rn(C0[i]);
                }
        check(cbad == 0, "fp32 c bits differ from k_kda_conv_pf", r);   // cbad is 0 by construction without G9_CONV_DBG
        check(vbad == 0, "V bits differ from bf16(c)", r);
        if (cbad || vbad) printf("  rep %d: c bit mismatches %lld, V mismatches %lld\n", r, cbad, vbad);
        if (!g_san)
        {
            // rows >= M of every plane must still hold the poison (no write past M)
            long long oob = 0;
            const int pb = poison_byte(r);
            const uint16_t want = static_cast<uint16_t>(pb | pb << 8);
            for (int p = 0; p < 3; ++p)
            {
                const std::vector<uint16_t> tail = down(g.pl.p[p] + static_cast<size_t>(M) * dl, static_cast<size_t>(MP - M) * dl);
                for (uint16_t t : tail) oob += t != want;
            }
            check(oob == 0, "conv16 wrote past row M", r);
        }
        uint64_t h = fnv(q); h = fnv(k, h); h = fnv(v, h); h = fnv(c, h); h = fnv(hist, h);
        if (r == 0) { rq = q; rk = k; rv = v; rc = c; rh = hist; h0 = h; }
        else check(same(q, rq) && same(k, rk) && same(v, rv) && same(c, rc) && same(hist, rh), "output differs from rep 0", r);
        printf("conv rep %2d poison %02x hash %016llx%s\n", r, poison_byte(r), (unsigned long long)h, h == h0 ? "" : "  <-- DIFFERS");
    }
    if (g_san || reps < 2) return;
    // stress: 200 launches with no host work between them (poison memset + history memcpy are stream-ordered), the result of every 10th hashed
    const int nstress = 200;
    for (int r = 0; r < nstress; ++r)
    {
        CK(cudaMemsetAsync(a.C, poison_byte(r), nC * 4, 0));
        CK(cudaMemcpyAsync(a.cst, cst0, hcst.size() * 4, cudaMemcpyDeviceToDevice, 0));
        kda::g9::launch_conv16(g, 0);
        if (r % 10 == 9)
        {
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const std::vector<uint16_t> q = down(g.pl.p[kda::g9::QN], static_cast<size_t>(M) * dl), k = down(g.pl.p[kda::g9::KN], static_cast<size_t>(M) * dl),
                                        v = down(g.pl.p[kda::g9::VV], static_cast<size_t>(M) * dl);
            const std::vector<float> hist = down(a.cst, hcst.size());
            const bool eq = same(q, rq) && same(k, rk) && same(v, rv) && same(hist, rh);
            check(eq, "stress launch output differs from rep 0", r);
            printf("conv stress launch %3d %s\n", r, eq ? "equal to rep 0" : "DIFFERS from rep 0");
        }
    }
}

static void e2e_arm(int M, int nh, int VS, int reps)
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
    const int MP = (M + 63) / 64 * 64;
    KdaPrefillScratch ps;
    if (const char* e = build_prefill_scratch(ps, MP, 1)) { fprintf(stderr, "%s\n", e); exit(3); }
    const size_t nC = static_cast<size_t>(MP) * 3 * dl, nof = static_cast<size_t>((M + 7) / 8) * dl * 8;
    a.C = dalloc<float>(nC, !g_san);
    a.OH = dalloc<uint16_t>(static_cast<size_t>(M) * dl);
    a.OF = dalloc<uint16_t>(nof);
    printf("e2e arm: M %d nh %d VS %d reps %d san %d\n", M, nh, VS, reps, g_san);
    std::vector<std::vector<uint16_t>> r0; std::vector<std::vector<float>> f0; uint64_t h0 = 0;
    for (int r = 0; r < reps; ++r)
    {
        const int pb = poison_byte(r);
        if (!g_san)
        {
            CK(cudaMemset(a.C, pb, nC * 4));
            float* sc[] = {ps.QE, ps.KD, ps.W, ps.U};
            for (float* f : sc) CK(cudaMemset(f, pb, static_cast<size_t>(MP) * nhmax(1) * DK * 4));
        }
        CK(cudaMemset(a.OH, pb, static_cast<size_t>(M) * dl * 2));
        CK(cudaMemset(a.OF, pb, nof * 2));
        CK(cudaMemcpy(a.S, S0, hS.size() * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(a.cst, c0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
        if (const char* e = kda::g9::launch(a, ps, VS, 0)) { fprintf(stderr, "%s\n", e); exit(3); }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const kda::g9::Planes P = kda::g9::planes(a, MP);
        std::vector<uint16_t> of = down(a.OF, nof);
        // out16 never writes the pad rows of the last 8-row tile (m >= M): mask those frag slots, they keep the per-rep poison by design
        long long masked = 0;
        for (int m = M; m < (M + 7) / 8 * 8; ++m)
            for (int k = 0; k < dl; ++k) { of[frag_u16(m, k, dl)] = 0; ++masked; }
        if (r == 0) printf("e2e: %lld OF pad slots masked (rows %d..%d)\n", masked, M, (M + 7) / 8 * 8 - 1);
        std::vector<std::vector<uint16_t>> u16 = {down(a.OH, static_cast<size_t>(M) * dl), of};
        for (int p = 0; p < kda::g9::NPLANE; ++p) u16.push_back(down(P.p[p], static_cast<size_t>(M) * dl));
        std::vector<std::vector<float>> f32 = {down(a.S, hS.size()), down(a.cst, hcst.size())};
        uint64_t h = 1469598103934665603ull;
        for (auto& v : u16) h = fnv(v, h);
        for (auto& v : f32) h = fnv(v, h);
        if (r == 0) { r0 = u16; f0 = f32; h0 = h; }
        else
        {
            bool eq = true;
            for (size_t i = 0; i < u16.size(); ++i) eq = eq && same(u16[i], r0[i]);
            for (size_t i = 0; i < f32.size(); ++i) eq = eq && same(f32[i], f0[i]);
            check(eq, "ON-path output differs from rep 0", r);
        }
        long long nonfinite = 0;
        for (float x : f32[0]) nonfinite += !std::isfinite(x);
        check(nonfinite == 0, "nonfinite S", r);
        printf("e2e  rep %2d poison %02x hash %016llx%s\n", r, pb, (unsigned long long)h, h == h0 ? "" : "  <-- DIFFERS");
    }
}

int main(int argc, char** argv)
{
    const int nh = argc > 1 ? atoi(argv[1]) : 2;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    const std::string arm = argc > 3 ? argv[3] : "both";
    g_san = argc > 4 && atoi(argv[4]) != 0;
    g_neg = argc > 5 && atoi(argv[5]) != 0;
    if (g_san) reps = 1;
    if (arm == "both" || arm == "conv") conv_arm(8191, nh, reps);
    if (arm == "both" || arm == "e2e") e2e_arm(8191, nh, 2, reps);
    printf("%s %d/%d\n", nfail ? "DET FAIL" : "DET PASS", ntest - nfail, ntest);
    return nfail ? 1 : 0;
}
