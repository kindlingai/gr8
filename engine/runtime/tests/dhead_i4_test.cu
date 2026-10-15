// G6_DRAFT_HEAD_I4 unit test: one GB10, synthetic records, no pack.
//   dhead_i4_test [--time [reps]]
// 1. kernels/head_i4.cuh k_mx_to_i4 against a host derivation: random MXFP8 head records (the layout of
//    engine/ref/g6ref.py mxfp8_record) -> decoded values -> INT4 g32 (engine/loader-rs records::int4_quantize) -> the
//    rt_record layout. Byte equality, every byte of the output written.
// 2. The INT4 head GEMV on those records (bind::make_gv INT4, O_F32, the draft head's tpc / ks; DEC_I4PAIR builds repack
//    them first, as build_draft_head_i4 does) against a host fp64 dot of the same INT4 weights and fp16 activations:
//    tolerance 1e-5 of the row's sum of |terms| (fp32 accumulate), and the same argmax per row unless the host top two
//    are within that tolerance.
// --time: the lm_head GEMV of one TP4 vocab slice (38784 x 6144): today's MXFP8 head (tpc 1, ks 32) against the INT4 head
//    over (tpc, ks), M = 1 and 3, mean of reps launches (CUDA events), and the bytes per launch.
// Build: scripts/c1_tests.sh <bin_dir> (TEST=dhead_i4_test)      Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../kernels/glue.cuh"
#include "../../kernels/bind.cuh"
#include "../../kernels/head_i4.cuh"

#define CK(x)                                                                                                   \
    do                                                                                                          \
    {                                                                                                           \
        const cudaError_t e_ = (x);                                                                             \
        if (e_ != cudaSuccess)                                                                                  \
        {                                                                                                       \
            printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));             \
            exit(2);                                                                                            \
        }                                                                                                       \
    } while (0)

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const std::string& s)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", s.c_str());
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

static uint32_t rnd(uint64_t i, uint32_t seed)
{
    uint32_t x = (uint32_t)i ^ (((uint32_t)(i >> 32) + seed) * 0x9e3779b9u) ^ (seed * 0x85ebca6bu);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static float e4m3_host(uint8_t b)
{
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp((float)(8 + m), e - 10) : std::ldexp((float)m, -9);
    return (b & 0x80) ? -v : v;
}
static float e8m0_host(uint8_t c) { return c ? std::ldexp(1.f, (int)c - 127) : std::ldexp(1.f, -127); }

// canonical codes [n][K] (E4M3), scales [n][K/32] (E8M0) -> MXFP8 GEMV record [n/16][K/32][528] (g6ref.mxfp8_record)
static std::vector<uint8_t> mx_record(const std::vector<uint8_t>& c, const std::vector<uint8_t>& s, int n, int K)
{
    const int ng = K / 32;
    std::vector<uint8_t> out((size_t)n / 16 * ng * 528);
    for (int t = 0; t < n / 16; ++t)
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* rec = &out[((size_t)t * ng + g) * 528];
            for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 32; ++l)
                {
                    const int r = l / 4, cc = l % 4;
                    const int rows[8] = {r, r, r + 8, r + 8, r, r, r + 8, r + 8};
                    const int ks[8] = {2 * cc, 2 * cc + 1, 2 * cc, 2 * cc + 1, 2 * cc + 8, 2 * cc + 9, 2 * cc + 8, 2 * cc + 9};
                    for (int i = 0; i < 8; ++i) rec[h * 256 + l * 8 + i] = c[(size_t)(16 * t + rows[i]) * K + 32 * g + 16 * h + ks[i]];
                }
            for (int i = 0; i < 8; ++i)
            {
                rec[512 + 2 * i] = s[(size_t)(16 * t + i) * ng + g];
                rec[512 + 2 * i + 1] = s[(size_t)(16 * t + i + 8) * ng + g];
            }
        }
    return out;
}

// INT4 g32 of the decoded MXFP8 values: q [n][K] (0..15), fp16 scales [n][K/32]; (records::int4_quantize)
static void i4_quantize(const std::vector<uint8_t>& c, const std::vector<uint8_t>& s, int n, int K, std::vector<uint8_t>& q,
                        std::vector<__half>& sc)
{
    const int ng = K / 32;
    q.assign((size_t)n * K, 8);
    sc.assign((size_t)n * ng, __float2half_rn(0.f));
    for (int i = 0; i < n; ++i)
        for (int g = 0; g < ng; ++g)
        {
            float v[32], amax = 0.f;
            for (int j = 0; j < 32; ++j)
            {
                v[j] = e4m3_host(c[(size_t)i * K + 32 * g + j]) * e8m0_host(s[(size_t)i * ng + g]);
                amax = std::max(amax, std::fabs(v[j]));
            }
            const __half h = __float2half_rn(amax / 7.f);
            sc[(size_t)i * ng + g] = h;
            const float sf = __half2float(h);
            if (sf == 0.f) continue;
            for (int j = 0; j < 32; ++j)
                q[(size_t)i * K + 32 * g + j] = (uint8_t)(std::min(std::max((int)std::nearbyint(v[j] / sf), -8), 7) + 8);
        }
}

// q [n][K], scales [n][K/32] -> INT4 GEMV record [n/16][K/32][288] (g6ref.rt_record with rt_tiles)
static std::vector<uint8_t> i4_record(const std::vector<uint8_t>& q, const std::vector<__half>& sc, int n, int K)
{
    const int ng = K / 32;
    std::vector<uint8_t> out((size_t)n / 16 * ng * 288);
    auto byte = [&](int row, int k0) { return (uint8_t)(q[(size_t)row * K + k0] | (q[(size_t)row * K + k0 + 1] << 4)); };
    for (int t = 0; t < n / 16; ++t)
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* rec = &out[((size_t)t * ng + g) * 288];
            for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 32; ++l)
                {
                    const int r = l / 4, cc = l % 4, k = 32 * g + 16 * h;
                    rec[h * 128 + l * 4 + 0] = byte(16 * t + r, k + 2 * cc);
                    rec[h * 128 + l * 4 + 1] = byte(16 * t + r + 8, k + 2 * cc);
                    rec[h * 128 + l * 4 + 2] = byte(16 * t + r, k + 2 * cc + 8);
                    rec[h * 128 + l * 4 + 3] = byte(16 * t + r + 8, k + 2 * cc + 8);
                }
            for (int i = 0; i < 8; ++i)
            {
                memcpy(rec + 256 + 4 * i, &sc[(size_t)(16 * t + i) * ng + g], 2);
                memcpy(rec + 256 + 4 * i + 2, &sc[(size_t)(16 * t + i + 8) * ng + g], 2);
            }
        }
    return out;
}

// random head-like MXFP8 rows: no NaN codes, scales 2^-11 .. 2^-6, a few all-zero groups (scale 0 path)
static void random_mx(int n, int K, uint32_t seed, std::vector<uint8_t>& c, std::vector<uint8_t>& s)
{
    c.resize((size_t)n * K);
    s.resize((size_t)n * K / 32);
    for (size_t i = 0; i < c.size(); ++i)
    {
        uint8_t b = (uint8_t)rnd(i, seed);
        if ((b & 0x7f) == 0x7f) b &= 0xbf;
        c[i] = b;
    }
    for (size_t i = 0; i < s.size(); ++i) s[i] = (uint8_t)(116 + rnd(i, seed + 1) % 6);
    for (int i = 0; i < n; i += 37)
        for (int j = 0; j < 32; ++j) c[(size_t)i * K + j] = 0;
}

static bind::Tensor mk(const void* p, long long a, long long b, long long c)
{
    bind::Tensor t;
    t.p = p; t.nd = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    return t;
}

// y [M][O] f32 = x [M][K] fp16 @ W^T through bind (fragment table + gv launch)
static void run_head(const void* rec, int codec, int O, int K, int tpc, int ks, int M, const __half* x, u32* xf, float* y)
{
    bind::Xfrag f;
    f.on = 1; f.src = x; f.ld = K; f.K = K; f.M = M; f.f16 = xf;
    bind::launch(f, nullptr);
    bind::Gv g;
    const int GR = codec == bind::INT4 ? 288 : 528;
    if (const char* e = bind::make_gv(mk(rec, O / 16, K / 32, GR), codec, tpc, ks, M, xf, y, gv::O_F32, g))
    {
        printf("make_gv: %s\n", e);
        exit(2);
    }
    bind::launch(g, nullptr);
}

static void test_convert_and_gemv()
{
    const int n = 1024, K = 6144, ng = K / 32;
    std::vector<uint8_t> c, s, q;
    std::vector<__half> sc;
    random_mx(n, K, 11u, c, s);
    const std::vector<uint8_t> mx = mx_record(c, s, n, K);
    i4_quantize(c, s, n, K, q, sc);
    const std::vector<uint8_t> want = i4_record(q, sc, n, K);
    uint8_t *dmx, *di4;
    CK(cudaMalloc(&dmx, mx.size()));
    CK(cudaMalloc(&di4, want.size()));
    CK(cudaMemcpy(dmx, mx.data(), mx.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(di4, 0xa5, want.size()));
    head_i4::launch(dmx, di4, n / 16, ng, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(want.size());
    CK(cudaMemcpy(got.data(), di4, got.size(), cudaMemcpyDeviceToHost));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i] && !bad++) first = i;
    char b[256];
    snprintf(b, sizeof b, "k_mx_to_i4 %d x %d: %zu of %zu bytes differ from the host derivation%s", n, K, bad, got.size(),
             bad ? (" (first at " + std::to_string(first) + ")").c_str() : "");
    report(bad == 0, b);

    // GEMV on the converted records
    const int MS[2] = {1, 3};
    const int SPLIT[3][2] = {{1, 32}, {1, 16}, {2, 8}};
    std::vector<__half> hx((size_t)8 * K);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half_rn(((rnd(i, 99u) >> 8) * (1.f / 16777216.f) * 2.f - 1.f));
    __half* dx;
    u32* xf;
    float* dy;
    CK(cudaMalloc(&dx, hx.size() * 2));
    CK(cudaMalloc(&xf, (size_t)K / 16 * 64 * 4 * g6mx::NT));
    CK(cudaMalloc(&dy, (size_t)8 * n * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    uint8_t* dp;
    CK(cudaMalloc(&dp, want.size()));
    for (const auto& sp : SPLIT)
    {
        CK(cudaMemcpy(dp, di4, want.size(), cudaMemcpyDeviceToDevice));
#if DEC_I4PAIR
        launch_i4pair(dp, n / 16, ng, sp[1], nullptr);
#endif
        for (int M : MS)
        {
            run_head(dp, bind::INT4, n, K, sp[0], sp[1], M, dx, xf, dy);
            CK(cudaDeviceSynchronize());
            std::vector<float> y((size_t)M * n);
            CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0;
            int argmax_bad = 0;
            for (int m = 0; m < M; ++m)
            {
                std::vector<double> ref(n), mag(n);
                for (int o = 0; o < n; ++o)
                {
                    double a = 0, am = 0;
                    for (int k = 0; k < K; ++k)
                    {
                        const double w = ((int)q[(size_t)o * K + k] - 8) * (double)__half2float(sc[(size_t)o * ng + k / 32]);
                        const double t = w * (double)__half2float(hx[(size_t)m * K + k]);
                        a += t; am += std::fabs(t);
                    }
                    ref[o] = a; mag[o] = am;
                    worst = std::max(worst, std::fabs(a - y[(size_t)m * n + o]) / (am + 1e-30));
                }
                int rg = 0, gg = 0;
                for (int o = 1; o < n; ++o)
                {
                    if (ref[o] > ref[rg]) rg = o;
                    if (y[(size_t)m * n + o] > y[(size_t)m * n + gg]) gg = o;
                }
                if (rg != gg && std::fabs(ref[rg] - ref[gg]) > 1e-5 * (mag[rg] + mag[gg])) ++argmax_bad;
            }
            snprintf(b, sizeof b, "INT4 head GEMV O=%d K=%d tpc=%d ks=%d M=%d: worst |gpu - host| / sum|terms| %.2e, argmax differs in %d rows",
                     n, K, sp[0], sp[1], M, worst, argmax_bad);
            report(worst < 1e-5 && argmax_bad == 0, b);
        }
    }
    cudaFree(dp); cudaFree(dmx); cudaFree(di4); cudaFree(dx); cudaFree(xf); cudaFree(dy);
}

static void time_heads(int reps)
{
    const int O = 38784, K = 6144, ng = K / 32;      // TP4 ranks 0 / 1 vocab slice
    const size_t mxb = (size_t)O / 16 * ng * 528, i4b = head_i4::bytes(O / 16, ng);
    std::vector<uint8_t> h(mxb);
    for (size_t i = 0; i < mxb; ++i)
    {
        uint8_t v = (uint8_t)rnd(i, 5u);
        if (i % 528 >= 512) v = (uint8_t)(116 + v % 6);
        else if ((v & 0x7f) == 0x7f) v &= 0xbf;
        h[i] = v;
    }
    uint8_t *dmx, *di4, *dp;
    CK(cudaMalloc(&dmx, mxb));
    CK(cudaMalloc(&di4, i4b));
    CK(cudaMalloc(&dp, i4b));
    CK(cudaMemcpy(dmx, h.data(), mxb, cudaMemcpyHostToDevice));
    head_i4::launch(dmx, di4, O / 16, ng, nullptr);
    __half* dx;
    u32* xf;
    float* dy;
    CK(cudaMalloc(&dx, (size_t)8 * K * 2));
    CK(cudaMemset(dx, 0x31, (size_t)8 * K * 2));
    CK(cudaMalloc(&xf, (size_t)K / 16 * 64 * 4 * g6mx::NT));
    CK(cudaMalloc(&dy, (size_t)8 * O * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timeit = [&](const void* rec, int codec, int tpc, int ks, int M) {
        for (int i = 0; i < 10; ++i) run_head(rec, codec, O, K, tpc, ks, M, dx, xf, dy);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run_head(rec, codec, O, K, tpc, ks, M, dx, xf, dy);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return 1e3 * ms / reps;
    };
    for (int M : {1, 3})
    {
        const double t = timeit(dmx, bind::MXFP8, 1, 32, M);
        printf("TIME head MXFP8 O=%d M=%d tpc=1 ks=32: %.1f us, %.1f MB, %.1f GB/s\n", O, M, t, mxb / 1e6, mxb / t / 1e3);
        for (int ks : {4, 8, 16, 32})
            for (int tpc : {1, 2, 4})
            {
                if (tpc * ks > 32) continue;
                CK(cudaMemcpy(dp, di4, i4b, cudaMemcpyDeviceToDevice));
#if DEC_I4PAIR
                launch_i4pair(dp, O / 16, ng, ks, nullptr);
#endif
                const double u = timeit(dp, bind::INT4, tpc, ks, M);
                printf("TIME head INT4  O=%d M=%d tpc=%d ks=%d: %.1f us, %.1f MB, %.1f GB/s, %.2fx MXFP8\n", O, M, tpc, ks, u, i4b / 1e6,
                       i4b / u / 1e3, t / u);
            }
    }
    fflush(stdout);
}

int main(int argc, char** argv)
{
    int reps = 0;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--time")) reps = i + 1 < argc ? atoi(argv[++i]) : 200;
    test_convert_and_gemv();
    if (reps > 0) time_heads(reps);
    printf("%d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
