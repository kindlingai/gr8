// Router GEMM bench: flash_mlp_pf::k_router_tile variants and k_linear_mr vs flash_mlp::k_linear (bitwise) on random
// FP32 x [M][4096] and BF16 w [288][4096].   router_pf_bench M [REPS]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "runtime/rank.cuh"
#include "runtime/flash_mlp_pf.cuh"

using namespace g6rt;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

template <class F> float timeit(F f, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char** argv)
{
    const int M = argc > 1 ? atoi(argv[1]) : 9216, reps = argc > 2 ? atoi(argv[2]) : 10, K = 4096, N = 288;
    std::mt19937 g(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx(static_cast<size_t>(M) * K);
    std::vector<uint16_t> hw(static_cast<size_t>(N) * K);
    for (auto& v : hx) v = nd(g);
    for (auto& v : hw) { float f = nd(g) * 0.02f; uint32_t u; memcpy(&u, &f, 4); v = static_cast<uint16_t>(u >> 16); }
    float *x, *y0, *y1;
    uint16_t* w;
    const size_t ny = static_cast<size_t>(M) * N;
    CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMalloc(&w, hw.size() * 2));
    CK(cudaMalloc(&y0, ny * 4)); CK(cudaMalloc(&y1, ny * 4));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> r0(ny), r1(ny);
    const float t0 = timeit([&] { flash_mlp::k_linear<<<dim3((N + 7) / 8, M), 256>>>(x, w, y0, K, N); }, reps);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(r0.data(), y0, ny * 4, cudaMemcpyDeviceToHost));
    printf("M %d: k_linear %.3f ms\n", M, t0);
    auto check = [&](const char* name, auto f) {
        CK(cudaMemset(y1, 0xff, ny * 4));
        const float t = timeit(f, reps);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(r1.data(), y1, ny * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < ny; ++i) diff += memcmp(&r0[i], &r1[i], 4) != 0;
        printf("  %-28s %8.3f ms  %s (%zu of %zu differ)\n", name, t, diff ? "NOT bit-identical" : "bit-identical", diff, ny);
    };
    check("k_linear_mr<4>", [&] { flash_mlp_pf::k_linear_mr<4><<<dim3((N + 7) / 8, (M + 3) / 4), 256>>>(x, w, y1, K, N, M); });
    check("launch_router (default)", [&] { flash_mlp_pf::launch_router(x, w, y1, K, N, M, 0); });
    check("tile<8,8,2,2,8>", [&] { flash_mlp_pf::launch_router<8, 8, 2, 2, 8>(x, w, y1, K, N, M, 0); });
    return 0;
}
