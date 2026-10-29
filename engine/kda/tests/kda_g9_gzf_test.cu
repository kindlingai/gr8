// GR8_GLM53F_PF_KDA_GZF gate: the G9 ON path with gz16 + the G16 / Z16 planes (gzf 0) and with the tiles built in K1 / k_out16z
// (gzf 1), same inputs (kda_g9_det's e2e setup) and same poison, compared byte for byte: OH, OF (pad slots masked), S,
// the conv history, the planes QN / KN / V / O16 (rows < M) and the whole K1 -> K2 tape (QE, KD, W, U). G16 / Z16 are
// not compared (gzf 1 does not write them). Then both arms timed at M 4096 / nh 21 (the TP3 Flash lane).
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -o kda_g9_gzf_test kda_g9_gzf_test.cu
// Last line: "GZF PASS k/k" or "GZF FAIL ...".
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

template <class T> static T* dalloc(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <class T> static T* up(const std::vector<T>& h) { T* d = dalloc<T>(h.size()); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
template <class T> static long long ndiff(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size()) return -1;
    long long n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}

static int nfail = 0, ntest = 0;

static void run_case(int M, int nh, int VS, bool timed)
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
    for (int s = 0; s < 3; ++s) a.conv_w[s] = up(hconv[s]);
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
    const size_t nsc = static_cast<size_t>(MP) * nhmax(1) * DK;
    a.C = dalloc<float>(nC);
    a.OH = dalloc<uint16_t>(static_cast<size_t>(M) * dl);
    a.OF = dalloc<uint16_t>(nof);
    float* sc[] = {ps.QE, ps.KD, ps.W, ps.U};
    auto reset = [&] {
        CK(cudaMemset(a.C, 0xA5, nC * 4));
        for (float* f : sc) CK(cudaMemset(f, 0xA5, nsc * 4));
        CK(cudaMemset(a.OH, 0xA5, static_cast<size_t>(M) * dl * 2));
        CK(cudaMemset(a.OF, 0xA5, nof * 2));
        CK(cudaMemcpy(a.S, S0, hS.size() * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(a.cst, c0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
    };
    struct Out { std::vector<std::vector<uint16_t>> u16; std::vector<std::vector<float>> f32; };
    auto run = [&](int gzf) {
        reset();
        if (const char* e = kda::g9::launch(a, ps, VS, 0, gzf)) { fprintf(stderr, "%s\n", e); exit(3); }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const kda::g9::Planes P = kda::g9::planes(a, MP);
        Out o;
        o.u16.push_back(down(a.OH, static_cast<size_t>(M) * dl));
        o.u16.push_back(down(a.OF, nof));
        for (int p : {kda::g9::QN, kda::g9::KN, kda::g9::VV, kda::g9::O16}) o.u16.push_back(down(P.p[p], static_cast<size_t>(M) * dl));
        o.f32.push_back(down(a.S, hS.size()));
        o.f32.push_back(down(a.cst, hcst.size()));
        for (float* f : sc) o.f32.push_back(down(f, nsc));
        return o;
    };
    const Out r0 = run(0), r1 = run(1);
    static const char* un[] = {"OH", "OF", "QN", "KN", "V", "O16"};
    static const char* fn[] = {"S", "conv hist", "tape QE", "tape KD", "tape W", "tape U"};
    long long tot = 0;
    std::string bad;
    for (size_t i = 0; i < r0.u16.size(); ++i) { const long long d = ndiff(r0.u16[i], r1.u16[i]); tot += d; if (d) bad += std::string(" ") + un[i] + "=" + std::to_string(d); }
    for (size_t i = 0; i < r0.f32.size(); ++i) { const long long d = ndiff(r0.f32[i], r1.f32[i]); tot += d; if (d) bad += std::string(" ") + fn[i] + "=" + std::to_string(d); }
    ++ntest;
    if (tot) ++nfail;
    printf("M %5d nh %2d VS %d: %s%s\n", M, nh, VS, tot ? "DIFFER" : "equal (OH OF QN KN V O16 S hist tape)", bad.c_str());
    if (timed)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int gzf : {0, 1, 0, 1})
        {
            for (int w = 0; w < 3; ++w) kda::g9::launch(a, ps, VS, 0, gzf);
            CK(cudaEventRecord(e0));
            const int it = 20;
            for (int i = 0; i < it; ++i) kda::g9::launch(a, ps, VS, 0, gzf);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("  time M %d nh %d gzf %d: %.1f us per ON path\n", M, nh, gzf, 1000.f * ms / it);
        }
    }
    for (int s = 0; s < 3; ++s) CK(cudaFree(const_cast<uint16_t*>(a.conv_w[s])));
    for (const void* p : {(const void*)a.fb, (const void*)a.gb, (const void*)a.onorm, (const void*)a.A_log, (const void*)a.dt_bias,
                          (const void*)a.row_slot, (const void*)a.P, (const void*)S0, (const void*)c0, (const void*)a.S, (const void*)a.cst,
                          (const void*)a.C, (const void*)a.OH, (const void*)a.OF})
        CK(cudaFree(const_cast<void*>(p)));
}

int main()
{
    const int cases[][2] = {{1, 2}, {2, 2}, {15, 2}, {16, 3}, {17, 3}, {100, 2}, {1365, 21}, {1366, 22}, {4095, 21}, {8191, 2}};
    for (auto& c : cases) run_case(c[0], c[1], 2, false);
    run_case(4096, 21, 1, false);
    run_case(4096, 21, 2, true);
    printf("%s %d/%d\n", nfail ? "GZF FAIL" : "GZF PASS", ntest - nfail, ntest);
    return nfail ? 1 : 0;
}
