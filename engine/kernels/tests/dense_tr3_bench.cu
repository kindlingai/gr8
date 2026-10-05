// #65 dense trellis GEMV (tr3_dense.cuh) against g6's int4 g32 and MXFP8 decode GEMVs (densev.cuh gv_gemv, as
// bind::make_layer splits them), on one rank's slice of q_b and o_proj. Random weights (decode cost does not depend on
// them), a pool of copies so the matrix is read from DRAM each step, M = 1..24 (the gv kernels take 8 rows per launch,
// so M > 8 runs them once per group of 8, as the engine would have to). The trellis kernel is checked against a host
// decode first (E4M3-rounded codebook, the rounding points of tr3_dense.cuh).
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DDEC_I4PAIR=1 -o dense_tr3_bench dense_tr3_bench.cu
//   dense_tr3_bench [--tp 4] [--bits 4] [--ks 8] [--iters 200]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_fp8.h>
#include "../bind.cuh"
#include "../tr3_dense.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

template <class T> static T* dev(const std::vector<T>& v)
{
    T* p;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// host decode of weight (k, n) of a tile, the tr3_e4m3_test reference with E4M3 rounding
static float ref_w(const uint8_t* tile, int K, int t)
{
    const int nb = 256 * K;
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i)
    {
        const int bit = (((t + 1) * K - 16 + i) % nb + nb) % nb;
        uint32_t word;
        memcpy(&word, tile + (bit / 32) * 4, 4);
        s = (s << 1) | ((word >> (31 - bit % 32)) & 1u);
    }
    uint32_t x = s * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    __half_raw lo, hi;
    lo.x = static_cast<unsigned short>(x & 0xffff);
    hi.x = static_cast<unsigned short>(x >> 16);
    const __half_raw v = __half_raw(__hadd(__half(lo), __half(hi)));
    const __nv_fp8_storage_t q = __nv_cvt_halfraw_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

static void had_host(float* v)          // 128-point Sylvester Hadamard, 1/sqrt(128)
{
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
    for (int i = 0; i < 128; ++i) v[i] *= SQ128;
}
static float f16r(float v) { return __half2float(__float2half_rn(v)); }

int main(int argc, char** argv)
{
    int tp = 4, bits = 4, ks = 8, iters = 200;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--tp") tp = atoi(argv[i + 1]);
        else if (a == "--bits") bits = atoi(argv[i + 1]);
        else if (a == "--ks") ks = atoi(argv[i + 1]);
        else if (a == "--iters") iters = atoi(argv[i + 1]);
    }
    const int heads = 64 / tp + (64 % tp ? 1 : 0);
    struct Shape { const char* name; int K, N; } shapes[2] = {{"q_b", 2048, heads * 256}, {"o_proj", heads * 256, 6144}};
    std::mt19937_64 rng(5);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (const Shape& S : shapes)
    {
        const int K = S.K, N = S.N, K16 = K / 16;
        const size_t tb = static_cast<size_t>(32) * bits, mat = static_cast<size_t>(K16) * (N / 16) * tb;
        const int P = std::max<size_t>(2, std::min<size_t>(64, (size_t)600e6 / mat));
        std::vector<uint8_t> wh(mat * P);
        for (auto& b : wh) b = static_cast<uint8_t>(rng() >> 24);
        std::vector<__half> suh(K), svh(N), xh(static_cast<size_t>(24) * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : suh) v = __float2half((rng() & 1 ? 1.f : -1.f) * (0.01f + 0.02f * (rng() % 1000) / 1000.f));
        for (auto& v : svh) v = __float2half((rng() & 1 ? 1.f : -1.f) * (0.5f + (rng() % 1000) / 1000.f));
        for (auto& v : xh) v = __float2half(nd(rng));
        uint8_t* dw = dev(wh);
        __half *dsuh = dev(suh), *dsvh = dev(svh), *dx = dev(xh), *dy;
        CK(cudaMalloc(&dy, static_cast<size_t>(24) * N * 2));
        tr3d::Dense D;
        D.w = dw; D.suh = dsuh; D.svh = dsvh; D.K = K; D.N = N; D.bits = bits; D.ks = ks;
        while (D.ks > 1 && K16 % D.ks) D.ks >>= 1;
        CK(cudaMalloc(&D.xa, static_cast<size_t>(3) * K16 * 64 * 4));
        CK(cudaMemset(D.xa, 0, static_cast<size_t>(3) * K16 * 64 * 4));
        CK(cudaMalloc(&D.part, static_cast<size_t>(D.ks) * 24 * N * 4));

        // correctness at M = 5 and 20 on copy 0, against the host decode
        for (int M : {5, 20})
        {
            CK(cudaMemset(D.xa, 0, static_cast<size_t>(3) * K16 * 64 * 4));
            tr3d::launch(D, dx, K, M, dy, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            std::vector<__half> yh(static_cast<size_t>(M) * N);
            CK(cudaMemcpy(yh.data(), dy, yh.size() * 2, cudaMemcpyDeviceToHost));
            // W (k, n): tile (k16, nb, j) at ((nb * K16 + k16) * 8 + j) * tb; weight t at (k16*16 + kk, (nb*8 + j)*16 + nn)
            std::vector<float> W(static_cast<size_t>(K) * N);
            for (int nb = 0; nb < N / 128; ++nb)
                for (int k16 = 0; k16 < K16; ++k16)
                    for (int j = 0; j < 8; ++j)
                    {
                        const uint8_t* tile = wh.data() + ((static_cast<size_t>(nb) * K16 + k16) * 8 + j) * tb;
                        for (int t = 0; t < 256; ++t)
                        {
                            const int lane = t / 8, q = t % 8;
                            const int kk = 2 * (lane % 4) + (q & 1) + 8 * ((q >> 1) & 1), nn = lane / 4 + 8 * (q >> 2);
                            W[static_cast<size_t>(k16 * 16 + kk) * N + (nb * 8 + j) * 16 + nn] = ref_w(tile, bits, t);
                        }
                    }
            double maxrel = 0;
            for (int m = 0; m < M; ++m)
            {
                std::vector<float> xr(K), z(N, 0.f);
                for (int k = 0; k < K; ++k) xr[k] = __half2float(xh[static_cast<size_t>(m) * K + k]) * __half2float(suh[k]);
                for (int b = 0; b < K / 128; ++b) had_host(&xr[b * 128]);
                for (int k = 0; k < K; ++k) xr[k] = f16r(xr[k]);
                for (int k = 0; k < K; ++k)
                    for (int n = 0; n < N; ++n) z[n] += xr[k] * W[static_cast<size_t>(k) * N + n];
                for (int n = 0; n < N; ++n) z[n] = f16r(z[n]);
                for (int b = 0; b < N / 128; ++b) had_host(&z[b * 128]);
                double mx = 0, err = 0;
                for (int n = 0; n < N; ++n)
                {
                    const double r = z[n] * __half2float(svh[n]);
                    mx = std::max(mx, std::fabs(r));
                    err = std::max(err, std::fabs(r - __half2float(yh[static_cast<size_t>(m) * N + n])));
                }
                maxrel = std::max(maxrel, err / mx);
            }
            printf("%s TP%d K%d: M=%d trellis GEMV vs host decode: max rel err %.2e %s\n", S.name, tp, bits, M, maxrel, maxrel < 5e-3 ? "ok" : "FAIL");
        }

        // int4 g32 and MXFP8 records of the same shape (bind::make_gv splits of today's paths)
        const int OT = N / 16;
        const size_t mi4 = static_cast<size_t>(OT) * (K / 32) * 288, mmx = static_cast<size_t>(OT) * (K / 32) * 528;
        const int Pi = std::max<size_t>(2, std::min<size_t>(64, (size_t)600e6 / mmx));
        std::vector<uint8_t> i4(mi4 * Pi), mx(mmx * Pi);
        for (auto& b : i4) b = static_cast<uint8_t>(rng() >> 24);
        for (size_t g = 0; g < i4.size() / 288; ++g)
            for (int i = 0; i < 16; ++i) { const __half h = __float2half(0.002f + 0.004f * (rng() % 1000) / 1000.f); memcpy(&i4[g * 288 + 256 + i * 2], &h, 2); }
        for (auto& b : mx) { b = static_cast<uint8_t>(rng() >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; }
        for (size_t g = 0; g < mx.size() / 528; ++g) for (int i = 0; i < 16; ++i) mx[g * 528 + 512 + i] = static_cast<uint8_t>(118 + rng() % 5);
        uint8_t *di4 = dev(i4), *dmx = dev(mx);
#if DEC_I4PAIR
        for (int i = 0; i < Pi; ++i) launch_i4pair(di4 + i * mi4, OT, K / 32, 8, st);
        CK(cudaStreamSynchronize(st));
#endif
        u32* xf;
        CK(cudaMalloc(&xf, static_cast<size_t>(3) * K16 * 64 * 4));
        bind::Gv gi, gm;
        auto ten = [&](const void* p, int ng, int gr) { bind::Tensor t; t.p = p; t.nd = 3; t.shape[0] = OT; t.shape[1] = ng; t.shape[2] = gr; return t; };

        // timed as a CUDA graph of 32 steps (decode runs captured), replayed iters / 32 times
        auto time = [&](auto f) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < 32; ++i) f(i);
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st));
            const int reps = std::max(1, iters / 32);
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            CK(cudaGetLastError());
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
            return 1e3 * ms / (reps * 32);
        };
        printf("%s TP%d (K %d, N %d): bytes trellis K%d %.2f MB, int4 g32 %.2f MB, MXFP8 %.2f MB\n", S.name, tp, K, N, bits,
               mat / 1e6, mi4 / 1e6, mmx / 1e6);
        for (int M : {1, 5, 8, 16, 20})
        {
            const int groups = (M + 7) / 8;
            CK(cudaMemset(D.xa, 0, static_cast<size_t>(3) * K16 * 64 * 4));
            const double tt = time([&](int i) { tr3d::Dense d = D; d.w = dw + (i % P) * mat; tr3d::launch(d, dx, K, M, dy, st); });
            // gv: one fragment table + launch per group of 8 rows
            const double ti = time([&](int i) {
                for (int g = 0; g < groups; ++g)
                {
                    bind::Xfrag a;
                    a.on = 1; a.src = dx + static_cast<size_t>(g) * 8 * K; a.ld = K; a.K = K; a.M = std::min(8, M - 8 * g); a.f16 = xf;
                    bind::launch(a, st);
                    bind::make_gv(ten(di4 + (i % Pi) * mi4, K / 32, 288), bind::INT4, 2, 8, a.M, xf, dy, gv::O_F16, gi);
                    bind::launch(gi, st);
                }
            });
            const double tm = time([&](int i) {
                for (int g = 0; g < groups; ++g)
                {
                    bind::Xfrag a;
                    a.on = 1; a.src = dx + static_cast<size_t>(g) * 8 * K; a.ld = K; a.K = K; a.M = std::min(8, M - 8 * g); a.f16 = xf;
                    bind::launch(a, st);
                    bind::make_gv(ten(dmx + (i % Pi) * mmx, K / 32, 528), bind::MXFP8, S.K == 2048 ? 1 : 2, S.K == 2048 ? 16 : 8, a.M, xf, dy, gv::O_F16, gm);
                    bind::launch(gm, st);
                }
            });
            printf("  M=%2d: trellis K%d %7.1f us (%5.1f GB/s) | int4 g32 %7.1f us (%5.1f GB/s) | MXFP8 %7.1f us | trellis/int4 %.2f\n", M, bits, tt,
                   mat / tt / 1e3, ti, mi4 / ti / 1e3, tm, tt / ti);
        }
        CK(cudaFree(dw)); CK(cudaFree(di4)); CK(cudaFree(dmx)); CK(cudaFree(xf)); CK(cudaFree(D.xa)); CK(cudaFree(D.part));
    }
    return 0;
}
