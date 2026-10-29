// MXFP8 prefill GEMM bench: random fp16 rows -> k_xq8 records, random e4m3 codes + E8M0 scales, then bindp::launch
// (the served kernel choice) timed with events. --cand runs the candidate kernel too and compares its output bytes
// with the served kernel's.
//   gemm_bench [--M 4096] [--N 2688] [--K 4096] [--out f32|f16] [--iters 50] [--cand 1] [--seed 1] [--flush 1]
// Default shape: the Flash KDA q / k / v projection at TP=3 (M = 4096 chunk rows, N = 21 heads x 128, K = HID).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../bind_prefill.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

#ifdef GEMM_CAND
#include GEMM_CAND
#endif

static void* g_flush = nullptr;   // --flush 1: 256 MB memset before each timed launch (cold L2, as in the engine)
template <class F> static float time_ms(F f, int iters, cudaStream_t st)
{
    for (int i = 0; i < 3; ++i) f();
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) { if (g_flush) CK(cudaMemsetAsync(g_flush, i & 0xff, 256u << 20, st)); CK(cudaEventRecord(a, st)); f(); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&t[i], a, b)); }
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

int main(int argc, char** argv)
{
    int M = 4096, N = 2688, K = 4096, iters = 50, cand = 0, flush = 0, rgrp = 0; uint64_t seed = 1; std::string out = "f32";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i]; const char* v = argv[i + 1];
        if (a == "--M") M = atoi(v); else if (a == "--N") N = atoi(v); else if (a == "--K") K = atoi(v);
        else if (a == "--iters") iters = atoi(v); else if (a == "--flush") flush = atoi(v); else if (a == "--rgrp") rgrp = atoi(v); else if (a == "--cand") cand = atoi(v);
        else if (a == "--seed") seed = strtoull(v, nullptr, 10); else if (a == "--out") out = v;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (K % 64 || N % 16) { fprintf(stderr, "K %% 64 and N %% 16\n"); return 2; }
    const int bf16 = out == "f32" ? 2 : 0;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> hx(static_cast<size_t>(M) * K);
    for (auto& v : hx) v = __float2half(nd(rng));
    std::vector<uint8_t> hw(static_cast<size_t>(N) * K), hs(static_cast<size_t>(N) * (K / 32));
    for (auto& v : hw) { uint8_t c = static_cast<uint8_t>(rng()); if ((c & 0x7f) == 0x7f) c ^= 1; v = c; }   // no e4m3 NaN
    for (auto& v : hs) v = static_cast<uint8_t>(118 + rng() % 6);
    __half* dx; uint8_t *dx8, *dw, *dws; void *dy, *dy2;
    const size_t x8b = static_cast<size_t>((M + 15) / 16) * (K / 32) * bindp::REC, yb = static_cast<size_t>(M) * N * (bf16 == 2 ? 4 : 2);
    CK(cudaMalloc(&dx, hx.size() * 2)); CK(cudaMalloc(&dx8, x8b)); CK(cudaMalloc(&dw, hw.size())); CK(cudaMalloc(&dws, hs.size()));
    CK(cudaMalloc(&dy, yb)); CK(cudaMalloc(&dy2, yb));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dws, hs.data(), hs.size(), cudaMemcpyHostToDevice));
    if (flush) CK(cudaMalloc(&g_flush, 256u << 20));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    bindp::Xq xq; xq.on = 1; xq.src = dx; xq.ld = K; xq.K = K; xq.M = M; xq.dst = dx8;
    bindp::launch(xq, st);
    bindp::Gemm g; g.on = 1; g.x8 = dx8; g.w = dw; g.ws = dws; g.K = K; g.N = N; g.M = M; g.y = dy; g.ldy = N; g.bf16 = bf16;
    CK(cudaMemsetAsync(dy, 0xcd, yb, st));
    const float ms = time_ms([&] { bindp::launch(g, st); }, iters, st);
    CK(cudaGetLastError());
    const double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("served M %d N %d K %d out %s: median %.3f ms = %.1f TFLOP/s\n", M, N, K, out.c_str(), ms, tf);
    if (rgrp)   // --rgrp G: the same launch with Gemm::rgrp = G (grouped raster), output bytes compared to the plain run
    {
        bindp::Gemm c = g; c.y = dy2; c.rgrp = rgrp;
        CK(cudaMemsetAsync(dy2, 0xab, yb, st));
        const float ms2 = time_ms([&] { bindp::launch(c, st); }, iters, st);
        CK(cudaGetLastError());
        std::vector<uint8_t> y1(yb), y2(yb);
        CK(cudaMemcpy(y1.data(), dy, yb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y2.data(), dy2, yb, cudaMemcpyDeviceToHost));
        size_t diff = 0; for (size_t i = 0; i < yb; ++i) diff += y1[i] != y2[i];
        printf("rgrp %2d: median %.3f ms = %.1f TFLOP/s, %.2fx, differing bytes %zu of %zu\n", rgrp, ms2,
               2.0 * M * N * K / (ms2 * 1e-3) / 1e12, ms / ms2, diff, yb);
        return diff ? 1 : 0;
    }
#ifdef GEMM_CAND
    if (cand)
    {
        bindp::Gemm c = g; c.y = dy2;
        CK(cudaMemsetAsync(dy2, 0xab, yb, st));
        const float ms2 = time_ms([&] { gemm_cand(c, st); }, iters, st);
        CK(cudaGetLastError());
        std::vector<uint8_t> y1(yb), y2(yb);
        CK(cudaMemcpy(y1.data(), dy, yb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y2.data(), dy2, yb, cudaMemcpyDeviceToHost));
        size_t diff = 0; for (size_t i = 0; i < yb; ++i) diff += y1[i] != y2[i];
        printf("cand   %-34s: median %.3f ms = %.1f TFLOP/s, %.2fx, differing bytes %zu of %zu\n", GEMM_CAND_NAME, ms2,
               2.0 * M * N * K / (ms2 * 1e-3) / 1e12, ms / ms2, diff, yb);
        return diff ? 1 : 0;
    }
#endif
    return 0;
}
