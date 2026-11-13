// Inkling-2-Small router + routed experts on the GPU against the CPU reference (sm_121a). Our own code.
//
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -I engine/kernels -I engine/ref/ink2 \
//        engine/kernels/tests/ink2_moe_test.cu -o /tmp/ink2_moe_test
//
//   /tmp/ink2_moe_test                          synthetic accuracy suite: M in {1, 2, 8, 9, 33, 128} on the shapes
//                                               (H 256, I 128, 32 experts, top 4) and (H 4608, I 384, 32 experts, top 8),
//                                               an NVFP4 layer and an MXFP8 layer each, against ink2ref::Model
//   /tmp/ink2_moe_test --bench                  latency, weight GB/s and TFLOP/s at H 4608, 384 experts, top 8, intermediate
//                                               slices 384 and 576, M = 1, 8 (decode) and 1024 (prefill), both formats
//   /tmp/ink2_moe_test --checkpoint-dir <dir>   layer 0 (NVFP4) and layer 40 (MXFP8) of a checkpoint, the first
//                                               --slice (384) intermediate rows, M = 1 and 8, against ink2ref::Model
//
// Checks of the suite: the router's picks equal the reference on every token whose keys are separated by more than
// 1e-4; the experts' output (run on the reference's routes, so a near tie cannot change the sum) has cosine > 0.9995 and
// relative L2 error < 0.02; decode and prefill agree (cosine > 0.99999) for M <= 8.
#include "nvfp4.cuh"
#include "ink2_moe.cuh"
#include "ink2_cpu_ref.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <fcntl.h>
#include <unistd.h>

using nv4::u32;
using nv4::u64;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } \
    } while (0)

template <class T>
static T* to_dev(const std::vector<T>& v)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, (v.size() ? v.size() : 1) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
static T* dev_alloc(size_t n)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    return p;
}
template <class T>
static std::vector<T> to_host(const T* p, size_t n)
{
    std::vector<T> v(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

struct Rng
{
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9e3779b97f4a7c15ull + 0x1234567ull) {}
    u32 next()
    {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return static_cast<u32>((s * 0x2545f4914f6cdd1dull) >> 32);
    }
    float uni() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
    float nrm() { return (uni() + uni() + uni() + uni() - 2.0f) * 1.7320508f; }   // variance 1
};

static int g_fail = 0;
static void expect(bool ok, const char* what)
{
    if (ok) return;
    printf("    FAIL: %s\n", what);
    ++g_fail;
}

struct Stat { double cos, rel; bool finite; };
// a = BF16 cells, b = reference values
static Stat compare(const uint16_t* a, const float* b, size_t n)
{
    double ab = 0, aa = 0, bb = 0, dd = 0;
    bool fin = true;
    for (size_t i = 0; i < n; ++i)
    {
        const double x = ink2ref::bf16_to_f32(a[i]), y = b[i];
        fin = fin && std::isfinite(x);
        ab += x * y; aa += x * x; bb += y * y; dd += (x - y) * (x - y);
    }
    return Stat{ab / std::sqrt(aa * bb + 1e-300), std::sqrt(dd / (bb + 1e-300)), fin};
}
static std::vector<float> cells_f32(const std::vector<uint16_t>& a)
{
    std::vector<float> f(a.size());
    for (size_t i = 0; i < a.size(); ++i) f[i] = ink2ref::bf16_to_f32(a[i]);
    return f;
}

template <class Fn>
static double time_ms(Fn&& f, int warm, int iters)
{
    for (int i = 0; i < warm; ++i) f();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) f();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return static_cast<double>(ms) / iters;
}

// Weight bytes one pass over the plan reads: every slot's records once per group of NT tiles.
template <class F>
static double plan_bytes(const ink2::Work& wk, int NT, int* slots_out)
{
    const std::vector<int> counts = to_host(wk.plan.counts, 2);
    const std::vector<int> nt = to_host(wk.plan.slot_nt, static_cast<size_t>(counts[0]));
    size_t groups = 0;
    for (int v : nt) groups += static_cast<size_t>((v + NT - 1) / NT);
    if (slots_out) *slots_out = counts[0];
    const double per = static_cast<double>(wk.I / 16) * (wk.H / F::KG) * 2 * F::GR + static_cast<double>(wk.H / 16) * (wk.I / F::KG) * F::GR;
    return static_cast<double>(groups) * per;
}

// ---------------------------------------------------------------------------------------------- one layer, GPU vs CPU
struct Ref
{
    std::vector<int> idx;            // [M][TK]
    std::vector<float> wt, out, gap; // [M][TK], [M][H], [M] = the smallest key distance among the picks and the next
};

static Ref cpu_ref(const ink2ref::Config& c, const ink2ref::Weights& w, const std::vector<float>& x, int M)
{
    const ink2ref::Model model(c, w);
    const int H = c.hidden, NE = c.n_exp, TK = c.top_k;
    Ref r;
    r.out.resize(static_cast<size_t>(M) * H);
    std::vector<int> idx;
    std::vector<float> wt, key(static_cast<size_t>(NE));
    for (int m = 0; m < M; ++m)
    {
        const float* xm = &x[static_cast<size_t>(m) * H];
        model.route(0, xm, idx, wt);
        r.idx.insert(r.idx.end(), idx.begin(), idx.end());
        r.wt.insert(r.wt.end(), wt.begin(), wt.end());
        model.moe(0, xm, &r.out[static_cast<size_t>(m) * H]);
        for (int e = 0; e < NE; ++e)
            key[e] = ink2ref::sigmoidf(ink2ref::dot(xm, &w.layers[0].router_w[static_cast<size_t>(e) * H], H)) + w.layers[0].router_bias[e];
        std::sort(key.begin(), key.end(), [](float a, float b) { return a > b; });
        float gap = 1e30f;
        for (int k = 0; k < TK && k + 1 < NE; ++k) gap = std::fmin(gap, key[k] - key[k + 1]);
        r.gap.push_back(gap);
    }
    return r;
}

template <class F>
static void eval_layer(const char* tag, const ink2ref::Config& c, const ink2ref::Weights& w, const std::vector<float>& x, const std::vector<int>& Ms,
                       bool timing)
{
    const int H = c.hidden, I = c.inter, NE = c.n_exp, TK = c.top_k, Mmax = *std::max_element(Ms.begin(), Ms.end());
    const ink2ref::Layer& L = w.layers[0];
    printf("%s: H %d, I %d, %d experts, top %d; CPU reference of %d tokens ...\n", tag, H, I, NE, TK, Mmax);
    fflush(stdout);
    const Ref ref = cpu_ref(c, w, x, Mmax);

    std::vector<uint16_t> xb(x.size()), gwb(L.router_w.size());
    for (size_t i = 0; i < x.size(); ++i) xb[i] = ink2ref::f32_to_bf16(x[i]);
    for (size_t i = 0; i < gwb.size(); ++i) gwb[i] = ink2ref::f32_to_bf16(L.router_w[i]);
    uint8_t* d_fc1 = to_dev(L.fc1);
    uint8_t* d_fc2 = to_dev(L.fc2);
    float* d_s1 = to_dev(L.s1);
    float* d_s2 = to_dev(L.s2);
    uint16_t* d_gw = to_dev(gwb);
    float* d_bias = to_dev(L.router_bias);
    uint16_t* d_x = to_dev(xb);
    uint16_t* d_out = dev_alloc<uint16_t>(static_cast<size_t>(Mmax) * H);
    ink2::Work wk, wkb;
    wk.alloc(Mmax, H, I, NE, TK, false);
    wkb.alloc(Mmax, H, I, NE, TK, true);

    for (int M : Ms)
    {
        const size_t n = static_cast<size_t>(M) * H, nr = static_cast<size_t>(M) * TK;
        const ink2::MoeArgs a = wk.args(d_fc1, d_fc2, d_s1, d_s2, M, d_out), ab = wkb.args(d_fc1, d_fc2, d_s1, d_s2, M, d_out);
        // the GPU router, and the experts on its routes
        ink2::route(d_x, d_gw, d_bias, M, H, NE, TK, c.route_scale, wk.score, wk.idx, wk.wt, wk.plan, 0);
        CK(cudaGetLastError());
        const std::vector<int> gi = to_host(wk.idx, nr);
        const std::vector<float> gw = to_host(wk.wt, nr);
        int sep = 0, sep_ok = 0, all_ok = 0;
        float wdiff = 0.0f;
        for (int m = 0; m < M; ++m)
        {
            const bool same = std::equal(gi.begin() + m * TK, gi.begin() + (m + 1) * TK, ref.idx.begin() + m * TK);
            all_ok += same;
            if (ref.gap[m] > 1e-4f) { ++sep; sep_ok += same; }
            if (same)
                for (int k = 0; k < TK; ++k) wdiff = std::fmax(wdiff, std::fabs(gw[m * TK + k] - ref.wt[m * TK + k]));
        }
        ink2::moe_prefill<F>(a, d_x, NE, 0);
        CK(cudaGetLastError());
        const Stat e2e = compare(to_host(d_out, n).data(), ref.out.data(), n);
        // the experts on the reference's routes
        for (ink2::Work* k : {&wk, &wkb})
        {
            CK(cudaMemcpy(k->idx, ref.idx.data(), nr * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(k->wt, ref.wt.data(), nr * sizeof(float), cudaMemcpyHostToDevice));
            ink2::plan(k->plan, k->idx, k->wt, M, NE, TK, 0);
        }
        ink2::moe_prefill<F>(a, d_x, NE, 0);
        const std::vector<uint16_t> pf = to_host(d_out, n);
        ink2::moe_prefill<F>(ab, d_x, NE, 0);
        const std::vector<uint16_t> pfb = to_host(d_out, n);
        CK(cudaGetLastError());
        const Stat s_pf = compare(pf.data(), ref.out.data(), n), s_pfb = compare(pfb.data(), ref.out.data(), n);
        printf("  M %3d: router picks equal on %d / %d separated tokens (%d / %d of all), max weight difference %.2g\n", M, sep_ok, sep, all_ok, M,
               static_cast<double>(wdiff));
        printf("         prefill cosine %.7f rel L2 %.2e | BF16 parts cosine %.7f rel L2 %.2e | on the GPU's routes cosine %.7f\n", s_pf.cos, s_pf.rel,
               s_pfb.cos, s_pfb.rel, e2e.cos);
        expect(sep_ok == sep, "router picks differ on a token with separated keys");
        expect(wdiff < 1e-4f, "route weights differ");
        expect(s_pf.finite && s_pf.cos > 0.9995 && s_pf.rel < 0.02, "prefill output against the reference");
        expect(s_pfb.finite && s_pfb.cos > 0.9995 && s_pfb.rel < 0.02, "prefill output with BF16 parts against the reference");
        expect(e2e.finite && (all_ok < M || (e2e.cos > 0.9995 && e2e.rel < 0.02)), "output on the GPU's routes");
        if (M <= 8)
        {
            ink2::moe_decode<F>(a, d_x, NE, 0);
            CK(cudaGetLastError());
            const std::vector<uint16_t> dec = to_host(d_out, n);
            const Stat s_dec = compare(dec.data(), ref.out.data(), n);
            const std::vector<float> pff = cells_f32(pf);
            const Stat s_dp = compare(dec.data(), pff.data(), n);
            printf("         decode  cosine %.7f rel L2 %.2e | decode against prefill cosine %.8f, bit equal: %s\n", s_dec.cos, s_dec.rel, s_dp.cos,
                   dec == pf ? "yes" : "no");
            expect(s_dec.finite && s_dec.cos > 0.9995 && s_dec.rel < 0.02, "decode output against the reference");
            expect(s_dp.cos > 0.99999, "decode against prefill");
            if (timing)
            {
                const double ms = time_ms([&] { ink2::moe_decode<F>(a, d_x, NE, 0); }, 3, 20);
                int slots = 0;
                const double bytes = plan_bytes<F>(wk, 1, &slots);
                printf("         decode %.3f ms, %d expert slots, %.1f MB of weights, %.1f GB/s (%.0f%% of 273)\n", ms, slots, bytes / 1e6,
                       bytes / ms / 1e6, bytes / ms / 1e6 / 273.0 * 1e2);
            }
        }
        fflush(stdout);
    }
    wk.release();
    wkb.release();
    cudaFree(d_fc1); cudaFree(d_fc2); cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_gw); cudaFree(d_bias); cudaFree(d_x); cudaFree(d_out);
}

static ink2ref::Config layer_config(int H, int I, int NE, int TK, bool mx)
{
    ink2ref::Config c;
    c.hidden = H; c.n_layers = 1; c.n_heads = 1; c.n_kv_heads = 1; c.head_dim = 16;
    c.inter = I; c.n_exp = NE; c.top_k = TK; c.tp = 1;
    c.global = {0};
    c.mxfp8 = {static_cast<uint8_t>(mx ? 1 : 0)};
    return c;
}
static std::vector<float> random_tokens(int M, int H, u64 seed)
{
    Rng g(seed);
    std::vector<float> x(static_cast<size_t>(M) * H);
    for (float& v : x) v = ink2ref::rbf(g.nrm());
    return x;
}

// ---------------------------------------------------------------------------------------------- synthetic suite
static ink2ref::Weights synth_layer(int H, int I, int NE, bool mx, u64 seed)
{
    Rng g(seed);
    ink2ref::Weights w;
    w.layers.resize(1);
    ink2ref::Layer& L = w.layers[0];
    const float sr = 1.5f / std::sqrt(static_cast<float>(H)), s13 = 1.5f / std::sqrt(static_cast<float>(H)), sd = 1.0f / std::sqrt(static_cast<float>(I));
    L.router_w.resize(static_cast<size_t>(NE) * H);
    for (float& v : L.router_w) v = ink2ref::rbf(sr * g.nrm());
    for (int e = 0; e < NE; ++e) L.router_bias.push_back(0.1f * g.nrm());
    const size_t n13 = static_cast<size_t>(2) * I * H, n2 = static_cast<size_t>(H) * I, half = static_cast<size_t>(I) * H;
    std::vector<float> w13(n13), w2(n2);
    std::vector<uint8_t> c13(n13), sc13(n13 / 16), c2(n2), sc2(n2 / 16);
    for (int e = 0; e < NE; ++e)
    {
        const float mag = 0.5f + g.uni();   // experts of different magnitude, so the outer scales differ
        for (float& v : w13) v = mag * s13 * g.nrm();
        for (float& v : w2) v = mag * sd * g.nrm();
        std::vector<uint8_t> r1, r2;
        if (mx)
        {
            ink2ref::mx8::quantize(w13.data(), n13 / 32, c13.data(), sc13.data());
            ink2ref::mx8::quantize(w2.data(), n2 / 32, c2.data(), sc2.data());
            r1 = ink2ref::mx8::pack_fc1(c13.data(), sc13.data(), c13.data() + half, sc13.data() + half / 32, I, H);
            r2 = ink2ref::mx8::pack_fc2(c2.data(), sc2.data(), H, I, 0, I);
        }
        else
        {
            L.s1.push_back(nv4pack::quantize(w13.data(), static_cast<size_t>(2) * I, H, nv4pack::ScaleRule::Static6, nullptr, c13.data(), sc13.data()));
            L.s2.push_back(nv4pack::quantize(w2.data(), static_cast<size_t>(H), I, nv4pack::ScaleRule::Static6, nullptr, c2.data(), sc2.data()));
            r1 = nv4pack::pack_fc1(c13.data(), sc13.data(), c13.data() + half / 2, sc13.data() + half / 16, I, H);
            r2 = nv4pack::pack_matrix(c2.data(), sc2.data(), H, I);
        }
        L.fc1.insert(L.fc1.end(), r1.begin(), r1.end());
        L.fc2.insert(L.fc2.end(), r2.begin(), r2.end());
    }
    return w;
}

static void suite()
{
    const std::vector<int> Ms = {1, 2, 8, 9, 33, 128};
    const int shapes[2][4] = {{256, 128, 32, 4}, {4608, 384, 32, 8}};
    for (const auto& s : shapes)
    {
        const int H = s[0], I = s[1], NE = s[2], TK = s[3];
        const std::vector<float> x = random_tokens(128, H, 11u + static_cast<u64>(H));
        {
            const ink2ref::Weights w = synth_layer(H, I, NE, false, 100u + static_cast<u64>(H));
            eval_layer<nv4::Frag>("NVFP4 layer", layer_config(H, I, NE, TK, false), w, x, Ms, false);
        }
        {
            const ink2ref::Weights w = synth_layer(H, I, NE, true, 200u + static_cast<u64>(H));
            eval_layer<ink2::Mx8Frag>("MXFP8 layer", layer_config(H, I, NE, TK, true), w, x, Ms, false);
        }
    }
}

// ---------------------------------------------------------------------------------------------- bench
// Random valid record bytes: any code byte (MXFP8: no NaN code), scales in a narrow finite range.
__global__ void k_fill(uint8_t* p, size_t n, u32 seed, int GR, int mx)
{
    const size_t i0 = (static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x) * 64;
    for (size_t i = i0; i < i0 + 64 && i < n; ++i)
    {
        u32 h = static_cast<u32>(i) * 2654435761u ^ static_cast<u32>(i >> 32) ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        u32 b = h & 255u;
        if (static_cast<int>(i % GR) < 512) { if (mx && (b & 0x7fu) == 0x7fu) b &= 0xfeu; }
        else b = mx ? 120u + (b & 7u) : 0x30u + (b & 15u);
        p[i] = static_cast<uint8_t>(b);
    }
}
static uint8_t* random_records(size_t n, u32 seed, int GR, bool mx)
{
    uint8_t* p = dev_alloc<uint8_t>(n);
    k_fill<<<static_cast<unsigned>((n + 16383) / 16384), 256>>>(p, n, seed, GR, mx ? 1 : 0);
    CK(cudaGetLastError());
    return p;
}

template <class F>
static void bench_format(const char* tag, int H, int I, int NE, int TK, const uint16_t* d_x, const uint16_t* d_gw, const float* d_bias, int Mmax)
{
    const bool mx = !F::OUTER;
    const size_t n1 = static_cast<size_t>(NE) * (I / 16) * (H / F::KG) * 2 * F::GR, n2 = static_cast<size_t>(NE) * (H / 16) * (I / F::KG) * F::GR;
    uint8_t* d_fc1 = random_records(n1, 1u, F::GR, mx);
    uint8_t* d_fc2 = random_records(n2, 2u, F::GR, mx);
    const std::vector<float> s1(static_cast<size_t>(NE), 0.4f / std::sqrt(static_cast<float>(H))), s2(static_cast<size_t>(NE), 0.4f / std::sqrt(static_cast<float>(I)));
    float* d_s1 = to_dev(s1);
    float* d_s2 = to_dev(s2);
    uint16_t* d_out = dev_alloc<uint16_t>(static_cast<size_t>(Mmax) * H);
    printf("%s, intermediate slice %d: fc1 %.0f MB, fc2 %.0f MB\n", tag, I, n1 / 1e6, n2 / 1e6);
    for (int M : {1, 8, Mmax})
    {
        const bool dec = M <= 8;
        ink2::Work wk;
        wk.alloc(M, H, I, NE, TK, !dec);
        const ink2::MoeArgs a = wk.args(d_fc1, d_fc2, d_s1, d_s2, M, d_out);
        const double t_route = time_ms([&] { ink2::route(d_x, d_gw, d_bias, M, H, NE, TK, 8.0f, wk.score, wk.idx, wk.wt, wk.plan, 0); }, 2, dec ? 20 : 5);
        const double ms = time_ms([&] { if (dec) ink2::moe_decode<F>(a, d_x, NE, 0); else ink2::moe_prefill<F>(a, d_x, NE, 0); }, 2, dec ? 20 : 5);
        CK(cudaGetLastError());
        int slots = 0;
        const double bytes = plan_bytes<F>(wk, dec ? 1 : 4, &slots);
        const double flop = 2.0 * M * TK * 3.0 * I * H;
        const std::vector<uint16_t> out = to_host(d_out, static_cast<size_t>(M) * H);
        bool fin = true;
        for (uint16_t v : out) fin = fin && (v & 0x7f80u) != 0x7f80u;
        printf("  M %4d %s: router %.3f ms, experts %.3f ms, %3d slots, %7.1f MB read, %6.1f GB/s (%3.0f%% of 273), %.3f TFLOP/s%s\n", M,
               dec ? "decode " : "prefill", t_route, ms, slots, bytes / 1e6, bytes / ms / 1e6, bytes / ms / 1e6 / 273.0 * 1e2, flop / ms / 1e9,
               fin ? "" : "  (non-finite outputs)");
        fflush(stdout);
        wk.release();
    }
    cudaFree(d_fc1); cudaFree(d_fc2); cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_out);
}

static void bench()
{
    const int H = 4608, NE = 384, TK = 8, Mmax = 1024;
    Rng g(3);
    std::vector<uint16_t> xb(static_cast<size_t>(Mmax) * H), gw(static_cast<size_t>(NE) * H);
    for (uint16_t& v : xb) v = ink2ref::f32_to_bf16(g.nrm());
    for (uint16_t& v : gw) v = ink2ref::f32_to_bf16(g.nrm() * 1.5f / std::sqrt(static_cast<float>(H)));
    const std::vector<float> bias(static_cast<size_t>(NE), 0.0f);
    uint16_t* d_x = to_dev(xb);
    uint16_t* d_gw = to_dev(gw);
    float* d_bias = to_dev(bias);
    for (int I : {384, 576})
    {
        bench_format<nv4::Frag>("NVFP4 (576 B records)", H, I, NE, TK, d_x, d_gw, d_bias, Mmax);
        bench_format<ink2::Mx8Frag>("MXFP8 (528 B records)", H, I, NE, TK, d_x, d_gw, d_bias, Mmax);
    }
    cudaFree(d_x); cudaFree(d_gw); cudaFree(d_bias);
}

// ---------------------------------------------------------------------------------------------- checkpoint
// A small safetensors reader: the tensor table of model.safetensors.index.json + shards, or of one model.safetensors.
struct Tensor
{
    std::string name, dtype;
    std::vector<long long> shape;
    int fd = -1;
    u64 start = 0, len = 0;
    void read(u64 off, size_t n, void* dst) const
    {
        if (off + n > len) throw std::runtime_error(name + ": read past the tensor");
        size_t done = 0;
        while (done < n)
        {
            const ssize_t r = pread(fd, static_cast<char*>(dst) + done, n - done, static_cast<off_t>(start + off + done));
            if (r <= 0) throw std::runtime_error(name + ": pread failed");
            done += static_cast<size_t>(r);
        }
    }
    std::vector<float> f32s() const
    {
        if (dtype != "F32") throw std::runtime_error(name + ": stored as " + dtype + ", expected F32");
        std::vector<float> v(len / 4);
        read(0, len, v.data());
        return v;
    }
};

struct Checkpoint
{
    std::string dir, index;                                   // index = text of the weight map, or empty
    std::vector<std::pair<std::string, std::string>> shards;  // file name -> header text
    std::vector<int> fds;
    std::vector<u64> data0;

    static std::string slurp(const std::string& path)
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return std::string();
        std::string s;
        char buf[1 << 16];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
        fclose(f);
        return s;
    }
    explicit Checkpoint(const std::string& d) : dir(d)
    {
        index = slurp(dir + "/model.safetensors.index.json");
        if (index.empty()) shard("model.safetensors");
    }
    size_t shard(const std::string& file)
    {
        for (size_t i = 0; i < shards.size(); ++i)
            if (shards[i].first == file) return i;
        const std::string path = dir + "/" + file;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        u64 n = 0;
        if (pread(fd, &n, 8, 0) != 8 || n > (1ull << 30)) throw std::runtime_error(path + ": bad header length");
        std::string h(n, '\0');
        if (pread(fd, &h[0], n, 8) != static_cast<ssize_t>(n)) throw std::runtime_error(path + ": short header");
        shards.emplace_back(file, h);
        fds.push_back(fd);
        data0.push_back(8 + n);
        return shards.size() - 1;
    }
    // The key of `text` that is `suffix` or ends with "." + suffix: position of its opening quote, or npos.
    static size_t find_key(const std::string& text, const std::string& suffix, std::string& key)
    {
        const std::string pat = suffix + "\"";
        for (size_t p = text.find(pat); p != std::string::npos; p = text.find(pat, p + 1))
        {
            if (p == 0 || (text[p - 1] != '.' && text[p - 1] != '"')) continue;
            const size_t q = text.rfind('"', p - 1);
            if (q == std::string::npos) continue;
            size_t c = p + pat.size();
            while (c < text.size() && isspace(static_cast<unsigned char>(text[c]))) ++c;
            if (c >= text.size() || text[c] != ':') continue;   // a value, not a key
            key = text.substr(q + 1, p + suffix.size() - q - 1);
            return q;
        }
        return std::string::npos;
    }
    static std::string field(const std::string& obj, const char* name)
    {
        const size_t p = obj.find(std::string("\"") + name + "\"");
        if (p == std::string::npos) throw std::runtime_error(std::string("safetensors header: no field ") + name);
        size_t c = obj.find(':', p) + 1;
        while (isspace(static_cast<unsigned char>(obj[c]))) ++c;
        const char close = obj[c] == '[' ? ']' : '"';
        const size_t e = obj.find(close, c + 1);
        return obj.substr(c + 1, e - c - 1);
    }
    bool get(const std::string& suffix, Tensor& t)
    {
        std::string key;
        size_t si = 0;
        if (!index.empty())
        {
            const size_t q = find_key(index, suffix, key);
            if (q == std::string::npos) return false;
            const size_t a = index.find('"', index.find(':', q + key.size() + 2)), b = index.find('"', a + 1);
            si = shard(index.substr(a + 1, b - a - 1));
        }
        const std::string& h = shards[si].second;
        std::string k2;
        const size_t q = find_key(h, index.empty() ? suffix : key, k2);
        if (q == std::string::npos) return false;
        const size_t o = h.find('{', q), e = h.find('}', o);
        const std::string obj = h.substr(o, e - o + 1);
        t.name = k2;
        t.dtype = field(obj, "dtype");
        t.shape.clear();
        const std::string sh = field(obj, "shape"), offs = field(obj, "data_offsets");
        for (const char* p = sh.c_str(); *p;)
        {
            char* end;
            const long long v = strtoll(p, &end, 10);
            if (end == p) { ++p; continue; }
            t.shape.push_back(v);
            p = end;
        }
        char* end;
        const u64 b0 = strtoull(offs.c_str(), &end, 10);
        while (*end && (*end < '0' || *end > '9')) ++end;
        const u64 b1 = strtoull(end, nullptr, 10);
        t.fd = fds[si];
        t.start = data0[si] + b0;
        t.len = b1 - b0;
        return true;
    }
    Tensor need(const std::string& suffix)
    {
        Tensor t;
        if (!get(suffix, t)) throw std::runtime_error("checkpoint: no tensor *" + suffix);
        return t;
    }
};

static std::vector<float> outer_scales(Checkpoint& ck, const std::string& w, int NE)
{
    Tensor t;
    std::vector<float> v;
    if (ck.get(w + "_amax", t))
    {
        v = t.f32s();
        const std::vector<float> den = ck.need(w + "_outer_scale_denominator").f32s();
        if (den.size() != 1 || den[0] == 0.0f) throw std::runtime_error(w + "_outer_scale_denominator: expected one nonzero value");
        for (float& x : v) x /= den[0];
    }
    else v = ck.need(w + "_scale_2").f32s();
    if (v.size() != 1 && v.size() != static_cast<size_t>(NE)) throw std::runtime_error(w + ": outer scale count");
    if (v.size() == 1) v.assign(static_cast<size_t>(NE), v[0]);
    return v;
}

// Layer l of the checkpoint, the first Is intermediate rows (gate rows 0 .., up rows I_full .., fc2 columns 0 ..).
static void checkpoint_layer(Checkpoint& ck, int l, int Is, int TK)
{
    const std::string pre = "layers." + std::to_string(l) + ".mlp.";
    const Tensor w13 = ck.need(pre + "experts.w13_weight"), w13s = ck.need(pre + "experts.w13_weight_scale");
    const Tensor w2 = ck.need(pre + "experts.w2_weight"), w2s = ck.need(pre + "experts.w2_weight_scale");
    const bool nv = w13.dtype == "U8" && w13s.dtype == "F8_E4M3", mx = w13.dtype == "F8_E4M3" && w13s.dtype == "U8";
    if ((!nv && !mx) || w13.shape.size() != 3 || w2.shape.size() != 3 || w2.dtype != w13.dtype || w2s.dtype != w13s.dtype)
        throw std::runtime_error(w13.name + ": neither NVFP4 nor MXFP8");
    const int NE = static_cast<int>(w13.shape[0]), If = static_cast<int>(w13.shape[1] / 2), H = static_cast<int>(w2.shape[1]);
    const size_t cb = nv ? static_cast<size_t>(H) / 2 : static_cast<size_t>(H), sb = static_cast<size_t>(H) / (nv ? 16 : 32);   // bytes per fc1 row
    const size_t cb2 = nv ? static_cast<size_t>(If) / 2 : static_cast<size_t>(If), sb2 = static_cast<size_t>(If) / (nv ? 16 : 32);
    if (static_cast<size_t>(w13.shape[2]) != cb || static_cast<size_t>(w13s.shape[2]) != sb || static_cast<size_t>(w2.shape[2]) != cb2 ||
        static_cast<size_t>(w2s.shape[2]) != sb2 || w2.shape[0] != NE || Is > If || Is % 64 || H % 64)
        throw std::runtime_error(w13.name + ": unexpected shapes");
    printf("layer %d: %s, %d experts, hidden %d, intermediate %d (slice %d)\n", l, nv ? "NVFP4" : "MXFP8", NE, H, If, Is);
    fflush(stdout);
    ink2ref::Weights w;
    w.layers.resize(1);
    ink2ref::Layer& L = w.layers[0];
    std::vector<uint8_t> gc(Is * cb), gs(Is * sb), uc(Is * cb), us(Is * sb), dc(H * cb2), ds(H * sb2);
    for (int e = 0; e < NE; ++e)
    {
        const u64 r0 = static_cast<u64>(e) * 2 * If;
        w13.read(r0 * cb, gc.size(), gc.data());
        w13.read((r0 + If) * cb, uc.size(), uc.data());
        w13s.read(r0 * sb, gs.size(), gs.data());
        w13s.read((r0 + If) * sb, us.size(), us.data());
        w2.read(static_cast<u64>(e) * H * cb2, dc.size(), dc.data());
        w2s.read(static_cast<u64>(e) * H * sb2, ds.size(), ds.data());
        const std::vector<uint8_t> r1 = nv ? nv4pack::pack_fc1(gc.data(), gs.data(), uc.data(), us.data(), Is, H)
                                           : ink2ref::mx8::pack_fc1(gc.data(), gs.data(), uc.data(), us.data(), Is, H);
        const std::vector<uint8_t> r2 = nv ? nv4pack::pack_fc2(dc.data(), ds.data(), H, If, 0, Is) : ink2ref::mx8::pack_fc2(dc.data(), ds.data(), H, If, 0, Is);
        L.fc1.insert(L.fc1.end(), r1.begin(), r1.end());
        L.fc2.insert(L.fc2.end(), r2.begin(), r2.end());
    }
    if (nv)
    {
        L.s1 = outer_scales(ck, pre + "experts.w13_weight", NE);
        L.s2 = outer_scales(ck, pre + "experts.w2_weight", NE);
    }
    const Tensor gw = ck.need(pre + "gate.weight");
    if (gw.dtype != "BF16" || gw.len != static_cast<u64>(NE) * H * 2) throw std::runtime_error(gw.name + ": expected BF16 [experts][hidden]");
    std::vector<uint16_t> gwb(static_cast<size_t>(NE) * H);
    gw.read(0, gw.len, gwb.data());
    L.router_w = cells_f32(gwb);
    L.router_bias = ck.need(pre + "gate.bias").f32s();
    if (L.router_bias.size() != static_cast<size_t>(NE)) throw std::runtime_error("gate.bias: expected one value per expert");
    const std::vector<float> x = random_tokens(8, H, 500u + static_cast<u64>(l));   // unit-variance random rows, not model activations
    const ink2ref::Config c = layer_config(H, Is, NE, TK, mx);
    if (nv) eval_layer<nv4::Frag>("checkpoint NVFP4 layer", c, w, x, {1, 8}, true);
    else eval_layer<ink2::Mx8Frag>("checkpoint MXFP8 layer", c, w, x, {1, 8}, true);
}

int main(int argc, char** argv)
{
    std::string dir;
    bool do_bench = false;
    int slice = 384, top_k = 8;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--bench") do_bench = true;
        else if (a == "--checkpoint-dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--slice" && i + 1 < argc) slice = atoi(argv[++i]);
        else if (a == "--top-k" && i + 1 < argc) top_k = atoi(argv[++i]);
        else
        {
            printf("usage: %s [--bench | --checkpoint-dir <dir> [--slice 384] [--top-k 8]]\n", argv[0]);
            return 2;
        }
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    try
    {
        if (do_bench) bench();
        else if (!dir.empty())
        {
            Checkpoint ck(dir);
            checkpoint_layer(ck, 0, slice, top_k);
            checkpoint_layer(ck, 40, slice, top_k);
        }
        else suite();
    }
    catch (const std::exception& e)
    {
        printf("error: %s\n", e.what());
        return 2;
    }
    if (do_bench) return 0;
    printf(g_fail ? "ink2_moe_test: FAIL (%d)\n" : "ink2_moe_test: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
