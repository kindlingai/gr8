// MXFP4 routed-expert decode GEMV (mxfp4.cuh): layout check, correctness vs a CPU double reference, and a bench against the
// DRAM byte ceiling.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o mxfp4_gemv_test mxfp4_gemv_test.cu
//   ./mxfp4_gemv_test layout        host only: packed record stream decodes back to the checkpoint weights (no GPU)
//   ./mxfp4_gemv_test check         GPU: fc1 + fc2 vs the CPU reference, M = 1 and M = 5, masks, determinism
//   ./mxfp4_gemv_test bench [sets]  GPU: 6 cold experts per step (a pool rotated so L2 never holds the next set); fc1, fc2,
//                                   fc1 + fc2 vs the probe (same loads, no dequant/mma) and a plain stream read of the same bytes
// Geometry: DS-V4.1-Flash-1017 at TP4 (I = 2304 / 4 = 576). -DMX4_I=384 for TP6, 768 for TP3, 1152 for TP2.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include "../mxfp4.cuh"
#include "../mxfp4_pack.h"

#ifndef MX4_I
#define MX4_I 576
#endif
constexpr int K1 = 5120, I = MX4_I, NOUT = 5120, TOPK = 6;
constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR;     // fc1 bytes per expert
constexpr size_t EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;       // fc2 bytes per expert

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

// ------------------------------------------------------------------------------------------ host data
struct Matrix { std::vector<uint8_t> packed, e8; int R, K; };
static Matrix rand_matrix(std::mt19937_64& g, int R, int K, int e_lo, int e_hi)
{
    Matrix m; m.R = R; m.K = K;
    m.packed.resize(static_cast<size_t>(R) * K / 2);
    m.e8.resize(static_cast<size_t>(R) * K / 32);
    for (auto& b : m.packed) b = static_cast<uint8_t>(g());
    for (auto& b : m.e8) b = static_cast<uint8_t>(e_lo + g() % (e_hi - e_lo + 1));
    return m;
}
static float wval(const Matrix& m, int r, int k)   // checkpoint decode
{
    const uint8_t b = m.packed[static_cast<size_t>(r) * m.K / 2 + k / 2];
    return mx4pack::e2m1((k & 1) ? (b >> 4) : (b & 15)) * static_cast<float>(mx4pack::e8m0(m.e8[static_cast<size_t>(r) * m.K / 32 + k / 32]));
}
static std::vector<float> dequant(const Matrix& m)
{
    std::vector<float> w(static_cast<size_t>(m.R) * m.K);
    for (int r = 0; r < m.R; ++r)
        for (int k = 0; k < m.K; ++k) w[static_cast<size_t>(r) * m.K + k] = wval(m, r, k);
    return w;
}

static int layout_check()
{
    std::mt19937_64 g(1);
    int bad = 0;
    for (int R : {16, 48}) for (int K : {64, 192, 576})
    {
        const Matrix m = rand_matrix(g, R, K, 100, 140);
        const std::vector<uint8_t> s = mx4pack::pack_matrix(m.packed.data(), m.e8.data(), R, K);
        for (int ot = 0; ot < R / 16; ++ot) for (int gi = 0; gi < K / 64; ++gi)
        {
            double d[16][64];
            mx4pack::decode_group(s.data() + (static_cast<size_t>(ot) * (K / 64) + gi) * mx4pack::GR, d);
            for (int r = 0; r < 16; ++r) for (int k = 0; k < 64; ++k)
                bad += d[r][k] != static_cast<double>(wval(m, ot * 16 + r, gi * 64 + k));
        }
    }
    printf("layout: pack_group -> decode_group mismatches %d   record %d B = %.4f bits/weight\n", bad, mx4pack::GR, mx4pack::GR * 8.0 / (16 * 64));
    return bad != 0;
}

// ------------------------------------------------------------------------------------------ check
struct Cfg { int M; std::vector<std::vector<int>> route; std::vector<std::vector<float>> w; };   // route[m] = TOPK experts

static int check()
{
    constexpr int NE = 8;
    std::mt19937_64 g(7);
    struct Expert { Matrix gate, up, down; };
    std::vector<Expert> ex(NE);
    for (auto& e : ex) { e.gate = rand_matrix(g, I, K1, 119, 124); e.up = rand_matrix(g, I, K1, 119, 124); e.down = rand_matrix(g, NOUT, I, 117, 122); }
    std::vector<uint8_t> h1, h2;
    for (auto& e : ex)
    {
        auto a = mx4pack::pack_fc1(e.gate.packed.data(), e.gate.e8.data(), e.up.packed.data(), e.up.e8.data(), I, K1);
        auto b = mx4pack::pack_matrix(e.down.packed.data(), e.down.e8.data(), NOUT, I);
        h1.insert(h1.end(), a.begin(), a.end());
        h2.insert(h2.end(), b.begin(), b.end());
    }
    uint8_t *d1, *d2;
    CK(cudaMalloc(&d1, h1.size())); CK(cudaMalloc(&d2, h2.size()));
    CK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    std::vector<std::vector<float>> deq_g(NE), deq_u(NE), deq_d(NE);
    for (int e = 0; e < NE; ++e) { deq_g[e] = dequant(ex[e].gate); deq_u[e] = dequant(ex[e].up); deq_d[e] = dequant(ex[e].down); }

    int fails = 0;
    for (int M : {1, 5, 8})
    {
        // routes: token m takes TOPK distinct experts of NE
        std::vector<std::vector<int>> route(M);
        std::vector<std::vector<float>> rw(M, std::vector<float>(TOPK));
        for (int m = 0; m < M; ++m)
        {
            std::vector<int> p(NE); for (int i = 0; i < NE; ++i) p[i] = i;
            std::shuffle(p.begin(), p.end(), g);
            route[m].assign(p.begin(), p.begin() + TOPK);
            for (auto& v : rw[m]) v = 0.1f + (g() % 1000) / 1000.f * 1.5f;
        }
        // slots: distinct experts ascending, mask bit c = token c routed there
        std::vector<int> se, sm; std::vector<float> sw;
        for (int e = 0; e < NE; ++e)
        {
            int mask = 0; float w[8] = {};
            for (int m = 0; m < M; ++m) for (int k = 0; k < TOPK; ++k) if (route[m][k] == e) { mask |= 1 << m; w[m] = rw[m][k]; }
            if (!mask) continue;
            se.push_back(e); sm.push_back(mask); sw.insert(sw.end(), w, w + 8);
        }
        const int NS = static_cast<int>(se.size());
        // x: bf16 N(0, .5)
        std::normal_distribution<float> nd(0.f, 0.5f);
        std::vector<__nv_bfloat16> xb(static_cast<size_t>(M) * K1);
        std::vector<float> xr(static_cast<size_t>(M) * K1);
        for (size_t i = 0; i < xb.size(); ++i)
        {
            xb[i] = __float2bfloat16_rn(nd(g));
            xr[i] = __half2float(__float2half_rn(__bfloat162float(xb[i])));
        }
        __nv_bfloat16* dx; u32* dxf; int *dse, *dsm; float* dsw; __half *dact, *dy;
        CK(cudaMalloc(&dx, xb.size() * 2)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4));
        CK(cudaMalloc(&dse, NS * 4)); CK(cudaMalloc(&dsm, NS * 4)); CK(cudaMalloc(&dsw, NS * 8 * 4));
        CK(cudaMalloc(&dact, static_cast<size_t>(NS) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(NS) * 8 * NOUT * 2));
        CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dse, se.data(), NS * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsm, sm.data(), NS * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsw, sw.data(), NS * 8 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dact, 0xff, static_cast<size_t>(NS) * 8 * I * 2)); CK(cudaMemset(dy, 0xff, static_cast<size_t>(NS) * 8 * NOUT * 2));   // NaN sentinel
        mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, M, K1, dxf);
        mx4::Args A{d1, d2, dxf, dse, dsm, dsw, nullptr, NS, 10.f, dact, dy};
        constexpr int KS = 8, WARPS = 8, DEPTH = 2;
        for (int rep = 0; rep < 2; ++rep)
        {
            mx4::k_fc1<K1, I, KS, 1, DEPTH, false><<<NS * (I / 16), KS * 32>>>(A);
            mx4::k_fc2<I, NOUT, WARPS, DEPTH, false><<<NS * (NOUT / 16 / WARPS), WARPS * 32>>>(A);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            if (rep == 0) {   // keep the first run for the determinism compare
                std::vector<__half> a0(static_cast<size_t>(NS) * 8 * I), y0(static_cast<size_t>(NS) * 8 * NOUT);
                CK(cudaMemcpy(a0.data(), dact, a0.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y0.data(), dy, y0.size() * 2, cudaMemcpyDeviceToHost));
                std::vector<__half> a1, y1;
                // second run overwrites; compare after it
                mx4::k_fc1<K1, I, KS, 1, DEPTH, false><<<NS * (I / 16), KS * 32>>>(A);
                mx4::k_fc2<I, NOUT, WARPS, DEPTH, false><<<NS * (NOUT / 16 / WARPS), WARPS * 32>>>(A);
                CK(cudaDeviceSynchronize());
                a1.resize(a0.size()); y1.resize(y0.size());
                CK(cudaMemcpy(a1.data(), dact, a1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y1.data(), dy, y1.size() * 2, cudaMemcpyDeviceToHost));
                const bool same = !memcmp(a0.data(), a1.data(), a0.size() * 2) && !memcmp(y0.data(), y1.data(), y0.size() * 2);
                printf("M=%d slots=%d determinism (2 runs bitwise equal): %s\n", M, NS, same ? "OK" : "FAIL");
                fails += !same;
                // compare with CPU reference
                double max_e_act = 0, max_e_y = 0; long n_act = 0, n_y = 0, nan_unmasked = 0, unwritten = 0;
                for (int s = 0; s < NS; ++s)
                {
                    const int e = se[s];
                    for (int c = 0; c < 8; ++c)
                    {
                        const bool used = (sm[s] >> c) & 1;
                        if (!used)
                        {   // fc2 must not have touched unused columns of y (sentinel = NaN)
                            for (int o = 0; o < NOUT; ++o) nan_unmasked += !std::isnan(__half2float(y0[(static_cast<size_t>(s) * 8 + c) * NOUT + o]));
                            continue;
                        }
                        std::vector<double> aref(I), tolA(I);
                        for (int i = 0; i < I; ++i)
                        {
                            double gt = 0, ut = 0, S = 0;
                            for (int k = 0; k < K1; ++k)
                            {
                                const double xv = xr[static_cast<size_t>(c) * K1 + k];
                                const double tg = deq_g[e][static_cast<size_t>(i) * K1 + k] * xv, tu = deq_u[e][static_cast<size_t>(i) * K1 + k] * xv;
                                gt += tg; ut += tu; S += std::fabs(tg) + std::fabs(tu);
                            }
                            const double gc = std::min(gt, 10.0), uc = std::min(std::max(ut, -10.0), 10.0);
                            aref[i] = gc / (1.0 + std::exp(-gc)) * uc;
                            tolA[i] = std::ldexp(std::fabs(aref[i]), -9) + 1e-5 * S;
                            const double got = __half2float(a0[(static_cast<size_t>(s) * 8 + c) * I + i]);
                            const double err = std::fabs(got - aref[i]) / tolA[i];
                            max_e_act = std::max(max_e_act, err); ++n_act; unwritten += std::isnan(got);
                        }
                        // fc2 from the GPU's own activations (isolates fc2 from fc1's fp16 rounding)
                        for (int o = 0; o < NOUT; ++o)
                        {
                            double acc = 0, S = 0;
                            for (int i = 0; i < I; ++i)
                            {
                                const double t = deq_d[e][static_cast<size_t>(o) * I + i] * __half2float(a0[(static_cast<size_t>(s) * 8 + c) * I + i]);
                                acc += t; S += std::fabs(t);
                            }
                            const double w = sw[static_cast<size_t>(s) * 8 + c];
                            const double ref = w * acc;
                            const double tol = std::ldexp(std::fabs(ref), -9) + 1e-5 * w * S;
                            const double got = __half2float(y0[(static_cast<size_t>(s) * 8 + c) * NOUT + o]);
                            max_e_y = std::max(max_e_y, std::fabs(got - ref) / tol); ++n_y; unwritten += std::isnan(got);
                        }
                    }
                }
                const bool ok = max_e_act <= 1.0 && max_e_y <= 1.0 && !nan_unmasked && !unwritten;
                printf("M=%d slots=%d  fc1 act: %ld values, max err/tol %.3f | fc2 ypart: %ld values, max err/tol %.3f | written-unmasked %ld, NaN-in-masked %ld  -> %s\n",
                       M, NS, n_act, max_e_act, n_y, max_e_y, nan_unmasked, unwritten, ok ? "OK" : "FAIL");
                fails += !ok;
            }
        }
        cudaFree(dx); cudaFree(dxf); cudaFree(dse); cudaFree(dsm); cudaFree(dsw); cudaFree(dact); cudaFree(dy);
    }
    cudaFree(d1); cudaFree(d2);
    return fails != 0;
}

// ------------------------------------------------------------------------------------------ bench
__global__ void k_fill(uint8_t* p, size_t n, u32 seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        u32 h = static_cast<u32>(i) * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = static_cast<uint8_t>(h);
    }
}
__global__ void k_fix_scales(uint8_t* p, size_t ngroups)   // scale bytes of every record into 2^-7 .. 2^-3
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < ngroups * 32; i += 256ull * gridDim.x)
    {
        uint8_t* b = p + (i / 32) * mx4::GR + mx4::SOFF + (i % 32);
        *b = static_cast<uint8_t>(120 + (*b & 3));
    }
}
__global__ void k_stream(const uint4* p, size_t n, u32* sink)
{
    u32 x = 0;
    size_t i = blockIdx.x * 256ull + threadIdx.x;
    const size_t st = 256ull * gridDim.x;
    for (; i + 3 * st < n; i += 4 * st)
    {
        u32 a[4][4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) mx4::ld4(p + i + u * st, a[u]);
        #pragma unroll
        for (int u = 0; u < 4; ++u) x ^= a[u][0] ^ a[u][1] ^ a[u][2] ^ a[u][3];
    }
    for (; i < n; i += st) { u32 a[4]; mx4::ld4(p + i, a); x ^= a[0] ^ a[1] ^ a[2] ^ a[3]; }
    if (x == 0x9e3779b9u) sink[0] = x;
}

struct Var { std::string name; std::function<void(const mx4::Args&, cudaStream_t)> run; std::function<void(const mx4::Args&, cudaStream_t)> run_pdl; };

template <int KS, int D, bool PR, int CW = 1> static Var v_fc1()
{
    return {"fc1 KS=" + std::to_string(KS) + (CW > 1 ? " CW=" + std::to_string(CW) : std::string()) + " D=" + std::to_string(D) + (PR ? " probe" : ""),
            [](const mx4::Args& A, cudaStream_t s) { mx4::k_fc1<K1, I, KS, CW, D, PR><<<A.n_slots * (I / 16 / CW), KS * CW * 32, 0, s>>>(A); }, nullptr};
}
template <int W, int D, bool PR> static Var v_fc2()
{
    return {"fc2 W=" + std::to_string(W) + " D=" + std::to_string(D) + (PR ? " probe" : ""),
            [](const mx4::Args& A, cudaStream_t s) { mx4::k_fc2<I, NOUT, W, D, PR><<<A.n_slots * (NOUT / 16 / W), W * 32, 0, s>>>(A); },
            [](const mx4::Args& A, cudaStream_t s) {   // launched as a programmatic dependent of the kernel before it in the stream
                cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(A.n_slots * (NOUT / 16 / W)); cfg.blockDim = dim3(W * 32); cfg.stream = s;
                cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
                cfg.attrs = at; cfg.numAttrs = 1;
                CK(cudaLaunchKernelEx(&cfg, mx4::k_fc2<I, NOUT, W, D, PR>, A));
            }};
}

static int bench(int sets)
{
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int PE = sets * TOPK;
    uint8_t *p1, *p2; u32* sink;
    CK(cudaMalloc(&p1, EB1 * PE)); CK(cudaMalloc(&p2, EB2 * PE)); CK(cudaMalloc(&sink, 64));
    k_fill<<<nsm * 4, 256>>>(p1, EB1 * PE, 1u); k_fill<<<nsm * 4, 256>>>(p2, EB2 * PE, 2u);
    k_fix_scales<<<nsm * 4, 256>>>(p1, EB1 * PE / mx4::GR); k_fix_scales<<<nsm * 4, 256>>>(p2, EB2 * PE / mx4::GR);
    std::vector<int> se(PE); for (int i = 0; i < PE; ++i) se[i] = i;
    std::vector<int> sm(TOPK, 1); std::vector<float> sw(TOPK * 8, 0.f); for (int s = 0; s < TOPK; ++s) sw[s * 8] = 1.f + 0.1f * s;
    int *dse, *dsm; float* dsw; u32* dxf; __nv_bfloat16* dx; __half *dact, *dy;
    CK(cudaMalloc(&dse, PE * 4)); CK(cudaMalloc(&dsm, TOPK * 4)); CK(cudaMalloc(&dsw, TOPK * 8 * 4));
    CK(cudaMalloc(&dx, K1 * 2)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4));
    CK(cudaMalloc(&dact, static_cast<size_t>(TOPK) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(TOPK) * 8 * NOUT * 2));
    CK(cudaMemcpy(dse, se.data(), PE * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsm, sm.data(), TOPK * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsw, sw.data(), TOPK * 8 * 4, cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> xb(K1); std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 0.5f);
    for (auto& v : xb) v = __float2bfloat16_rn(nd(g));
    CK(cudaMemcpy(dx, xb.data(), K1 * 2, cudaMemcpyHostToDevice));
    mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, 1, K1, dxf);
    CK(cudaMemset(dact, 0, static_cast<size_t>(TOPK) * 8 * I * 2));
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const int ITERS = 20 * sets, WARM = 2 * sets;
    auto args = [&](int it) { return mx4::Args{p1, p2, dxf, dse + (it % sets) * TOPK, dsm, dsw, nullptr, TOPK, 10.f, dact, dy}; };
    auto time_us = [&](const std::function<void(int)>& f) {   // min of 3 repetitions (a rep can be perturbed by other work on the GPU)
        for (int i = 0; i < WARM; ++i) f(i);
        CK(cudaStreamSynchronize(st));
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < ITERS; ++i) f(i);
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms * 1000.0 / ITERS);
        }
        return best;
    };
    const double BY1 = 6.0 * EB1, BY2 = 6.0 * EB2;   // bytes per step
    const double SPEC = 273.0, FLEET = 255.8;
    auto line = [&](const std::string& n, double us, double bytes) {
        const double gbs = bytes / us / 1e3;
        printf("  %-26s %8.1f us  %7.1f GB/s  %5.1f%% of 273  %5.1f%% of 255.8\n", n.c_str(), us, gbs, 100 * gbs / SPEC, 100 * gbs / FLEET);
    };
    printf("device %s, %d SMs, %zu MB L2; geometry K1=%d I=%d NOUT=%d; per step 6 experts: fc1 %.2f MB + fc2 %.2f MB = %.2f MB (%.4f bits/weight)\n",
           pr.name, nsm, static_cast<size_t>(pr.l2CacheSize) >> 20, K1, I, NOUT, BY1 / 1e6, BY2 / 1e6, (BY1 + BY2) / 1e6, (BY1 + BY2) * 8 / (6.0 * (3.0 * I * K1)));
    printf("pool %d experts (%.0f MB), %d rotating sets of 6, %d iterations\n", PE, (EB1 + EB2) * PE / 1e6, sets, ITERS);

    // plain stream read of the same bytes: the achievable ceiling
    const size_t n1 = static_cast<size_t>(BY1 / 16), n2 = static_cast<size_t>(BY2 / 16);
    auto stream = [&](int it, int mult) {
        k_stream<<<nsm * mult, 256, 0, st>>>(reinterpret_cast<const uint4*>(p1 + EB1 * TOPK * (it % sets)), n1, sink);
        k_stream<<<nsm * mult, 256, 0, st>>>(reinterpret_cast<const uint4*>(p2 + EB2 * TOPK * (it % sets)), n2, sink);
    };
    printf("ceiling: plain read of the same bytes (fc1 range, fc2 range)\n");
    double best_stream = 1e30;
    for (int mult : {8, 16, 32, 64})
    {
        const double us = time_us([&](int it) { stream(it, mult); });
        printf("  stream, %d CTAs/SM            ", mult); line("fc1+fc2 bytes", us, BY1 + BY2);
        best_stream = std::min(best_stream, us);
    }

    std::vector<Var> f1 = {v_fc1<4, 1, false>(), v_fc1<4, 2, false>(), v_fc1<8, 1, false>(), v_fc1<8, 2, false>(), v_fc1<8, 3, false>(),
                           v_fc1<10, 2, false>(), v_fc1<16, 1, false>(), v_fc1<16, 2, false>(), v_fc1<20, 2, false>(),
                           v_fc1<5, 2, false>(), v_fc1<2, 3, false>(), v_fc1<8, 2, false, 2>(), v_fc1<4, 2, false, 4>(), v_fc1<10, 2, false, 2>()};
    std::vector<Var> p1v = {v_fc1<4, 2, true>(), v_fc1<8, 2, true>(), v_fc1<10, 2, true>(), v_fc1<16, 2, true>()};
    std::vector<Var> f2 = {v_fc2<4, 1, false>(), v_fc2<4, 3, false>(), v_fc2<8, 1, false>(), v_fc2<8, 3, false>(), v_fc2<16, 1, false>(), v_fc2<16, 3, false>()};
    std::vector<Var> p2v = {v_fc2<4, 3, true>(), v_fc2<8, 3, true>(), v_fc2<16, 3, true>()};
    Var* b1 = nullptr; Var* b2 = nullptr; double t1 = 1e30, t2 = 1e30;
    printf("fc1 (gate+up, swiglu) variants:\n");
    for (auto& v : f1) { const double us = time_us([&](int it) { v.run(args(it), st); }); line(v.name, us, BY1); if (us < t1) { t1 = us; b1 = &v; } }
    printf("fc1 probe (loads only):\n");
    for (auto& v : p1v) line(v.name, time_us([&](int it) { v.run(args(it), st); }), BY1);
    printf("fc2 (down) variants:\n");
    for (auto& v : f2) { const double us = time_us([&](int it) { v.run(args(it), st); }); line(v.name, us, BY2); if (us < t2) { t2 = us; b2 = &v; } }
    printf("fc2 probe (loads only):\n");
    for (auto& v : p2v) line(v.name, time_us([&](int it) { v.run(args(it), st); }), BY2);
    printf("best: %s %.1f us, %s %.1f us\n", b1->name.c_str(), t1, b2->name.c_str(), t2);
    const double us = time_us([&](int it) { b1->run(args(it), st); b2->run(args(it), st); });
    printf("one routed layer (fc1 then fc2, back to back):\n");
    line("fc1+fc2 best", us, BY1 + BY2);
    line("stream ceiling (best)", best_stream, BY1 + BY2);
    printf("  routed-expert time = %.1f us = %.1f%% of the stream ceiling's rate\n", us, 100 * best_stream / us);
    printf("fc1 -> fc2 with programmatic dependent launch (fc2 prefetches its weights during fc1's tail):\n");
    double best_pdl = 1e30; std::string best_pdl_name;
    for (auto& v : f2)
    {
        const double upd = time_us([&](int it) { b1->run(args(it), st); v.run_pdl(args(it), st); });
        line(b1->name + " + " + v.name + " PDL", upd, BY1 + BY2);
        if (upd < best_pdl) { best_pdl = upd; best_pdl_name = b1->name + " + " + v.name; }
    }
    printf("summary (min of 3 reps each):\n");
    line("stream ceiling", best_stream, BY1 + BY2);
    line("fc1+fc2 stream-ordered", us, BY1 + BY2);
    line("fc1+fc2 PDL: " + best_pdl_name, best_pdl, BY1 + BY2);
    printf("  routed layer: %.1f us stream-ordered, %.1f us with PDL; ceiling %.1f us (PDL = %.1f%% of the ceiling's rate)\n", us, best_pdl, best_stream, 100 * best_stream / best_pdl);
    return 0;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "layout";
    if (!strcmp(mode, "layout")) return layout_check();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { printf("no GPU visible: only `layout` runs here\n"); return 3; }
    if (!strcmp(mode, "check")) return layout_check() | check();
    if (!strcmp(mode, "bench")) return bench(argc > 2 ? atoi(argv[2]) : 16);
    printf("usage: %s layout|check|bench [sets]\n", argv[0]);
    return 1;
}
