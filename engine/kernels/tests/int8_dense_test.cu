// Native exlr8 int8 gG CUDA kernel parity & sm_121a microbenchmark test:
//   1. Decode GEMV (gv_gemv<Int8v<GRP>>) across GRP in {1,2,4,8}, M in {1,2,4,8,16,32}, O_F16/O_BF16/O_F32 vs FP32 CPU ref
//   2. Prefill GEMM (i8gemm::launch, paired w2, Rs scatter) across GRP in {1,2,4,8}, M in {1,7,32,63,64,129,256} vs FP32 CPU ref & decode GEMV
//   3. K1 Dense MLP (k1_shared_fc1 + k1_shared_fc2 with s_i8_fc1/fc2) vs FP32 CPU SwiGLU ref
//   4. Direct microbenchmarks on sm_121a:
//        - Decode GEMV: Int8v<1/2/4/8> vs Mxfp8 vs Int4v vs Bf16w at GLM-5.3 shapes (M=1,4,8,16,32)
//        - Prefill GEMM: k_gemm_i8g (g16/g32/g64/g128) vs k_gemm_mx8b (Mxfp8) vs cuBLAS BF16 at GLM-5.3 shapes (M=64,256,1024,4096)
//
// Build (sm_121a):
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_NR=4 -DDEC_S2 -DDEC_I4PAIR=1 \
//        -Iengine/kernels -o int8_dense_test engine/kernels/tests/int8_dense_test.cu \
//        engine/knobs-rs/target/release/libgr8knobs.a -lcublas -lpthread -ldl -lm
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../bind_prefill.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CUBLAS_CK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)s_); exit(1); } } while (0)

static void die(const char* msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

static float bf16_to_f(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Host packer matching engine/loader-rs/src/records.rs::int8_record exactly:
// [O/16][K/G][GR] where GR = GRP*256 + 32, each 16x16 sub-tile in GV lane order, followed by 16 f16 scales (s[0..7], s[8..15]).
static std::vector<uint8_t> pack_int8_record(const std::vector<int8_t>& q, const std::vector<__half>& s, int O, int K, int G)
{
    const int grp = G / 16;
    const int gr = grp * 256 + 32;
    const int ng = K / G;
    std::vector<uint8_t> out(static_cast<size_t>(O / 16) * ng * gr, 0);
    for (int t = 0; t < O / 16; ++t)
    {
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* rec = out.data() + (static_cast<size_t>(t) * ng + g) * gr;
            for (int h = 0; h < grp; ++h)
            {
                uint8_t* sub = rec + h * 256;
                for (int l = 0; l < 32; ++l)
                {
                    const int r = l >> 2, c = l & 3;
                    const int rows[8] = {r, r, r + 8, r + 8, r, r, r + 8, r + 8};
                    const int cols[8] = {2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c + 8, 2 * c + 9, 2 * c + 8, 2 * c + 9};
                    for (int e = 0; e < 8; ++e)
                        sub[l * 8 + e] = static_cast<uint8_t>(q[static_cast<size_t>(16 * t + rows[e]) * K + g * G + h * 16 + cols[e]]);
                }
            }
            uint16_t* sc = reinterpret_cast<uint16_t*>(rec + grp * 256);
            for (int i = 0; i < 8; ++i)
            {
                sc[2 * i] = __half_as_ushort(s[static_cast<size_t>(16 * t + i) * ng + g]);
                sc[2 * i + 1] = __half_as_ushort(s[static_cast<size_t>(16 * t + i + 8) * ng + g]);
            }
        }
    }
    return out;
}

static void cpu_int8_matmul(const std::vector<__half>& x, const std::vector<int8_t>& q, const std::vector<__half>& s,
                            int M, int O, int K, int G, std::vector<float>& y)
{
    const int ng = K / G;
    y.assign(static_cast<size_t>(M) * O, 0.f);
    for (int m = 0; m < M; ++m)
    {
        for (int o = 0; o < O; ++o)
        {
            double total = 0.0;
            for (int g = 0; g < ng; ++g)
            {
                double acc = 0.0;
                const int k0 = g * G;
                for (int j = 0; j < G; ++j)
                    acc += static_cast<double>(__half2float(x[static_cast<size_t>(m) * K + k0 + j])) *
                           static_cast<double>(q[static_cast<size_t>(o) * K + k0 + j]);
                total += acc * static_cast<double>(__half2float(s[static_cast<size_t>(o) * ng + g]));
            }
            y[static_cast<size_t>(m) * O + o] = static_cast<float>(total);
        }
    }
}

struct ErrStats { float max_abs = 0.f, rel_l2 = 0.f; double cos = 1.0; };

static ErrStats compare_f32(const std::vector<float>& got, const std::vector<float>& ref)
{
    ErrStats st;
    double dot = 0.0, ng = 0.0, nr = 0.0, nd = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        const double a = got[i], b = ref[i], d = a - b;
        if (fabs(d) > st.max_abs) st.max_abs = static_cast<float>(fabs(d));
        nd += d * d;
        dot += a * b;
        ng += a * a;
        nr += b * b;
    }
    st.rel_l2 = (nr > 0.0) ? static_cast<float>(sqrt(nd / nr)) : 0.f;
    st.cos = (ng > 0.0 && nr > 0.0) ? dot / sqrt(ng * nr) : 1.0;
    return st;
}

static void make_random_data(int M, int O, int K, int G, uint32_t seed,
                             std::vector<__half>& x, std::vector<int8_t>& q, std::vector<__half>& s)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nx(0.f, 0.5f);
    std::uniform_int_distribution<int> nq(-127, 127);
    std::uniform_real_distribution<float> ns(0.002f, 0.05f);
    x.resize(static_cast<size_t>(M) * K);
    for (auto& v : x) v = __float2half_rn(nx(rng));
    q.resize(static_cast<size_t>(O) * K);
    for (auto& v : q) v = static_cast<int8_t>(nq(rng));
    // Boundary stress values
    q[0] = -127; q[1] = 127; q[2] = 0; q[3] = -1; q[4] = 1;
    s.resize(static_cast<size_t>(O) * (K / G));
    for (auto& v : s) v = __float2half_rn(ns(rng));
    s[0] = __float2half_rn(0.f);
}

static void test_decode_gemv()
{
    printf("=== 1. Decode GEMV (gv_gemv<Int8v<GRP>>) FP32 Parity ===\n");
    const int O = 256, K = 512;
    const int Ms[] = {1, 2, 4, 8, 16, 32};
    const int Gs[] = {16, 32, 64, 128};
    const int odts[] = {gv::O_F32, gv::O_F16, gv::O_BF16};

    for (int G : Gs)
    {
        const int grp = G / 16;
        const int codec = bind::int8_codec_of(grp);
        std::vector<__half> x_max;
        std::vector<int8_t> q;
        std::vector<__half> s;
        make_random_data(32, O, K, G, 1000u + G, x_max, q, s);
        auto rec = pack_int8_record(q, s, O, K, G);

        uint8_t* dw = nullptr;
        __half* dx = nullptr;
        u32* dxf = nullptr;
        void* dy = nullptr;
        CK(cudaMalloc(&dw, rec.size()));
        CK(cudaMemcpy(dw, rec.data(), rec.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x_max.size() * sizeof(__half)));
        CK(cudaMemcpy(dx, x_max.data(), x_max.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxf, 4 * (K / 16) * 64 * sizeof(u32)));
        CK(cudaMalloc(&dy, static_cast<size_t>(32) * O * sizeof(float)));

        bind::Tensor wt;
        wt.p = dw; wt.nd = 3;
        wt.shape[0] = O / 16; wt.shape[1] = K / G; wt.shape[2] = grp * 256 + 32;

        for (int M : Ms)
        {
            std::vector<__half> xm(x_max.begin(), x_max.begin() + static_cast<size_t>(M) * K);
            std::vector<float> yref;
            cpu_int8_matmul(xm, q, s, M, O, K, G, yref);

            bind::Xfrag xf{};
            xf.on = 1; xf.src = dx; xf.ld = K; xf.K = K; xf.M = M; xf.f16 = dxf;
            bind::launch(xf, 0);

            for (int odt : odts)
            {
                bind::Gv g;
                const char* err = bind::make_gv(wt, codec, 2, 4, M, dxf, dy, odt, g);
                if (err) die(err);
                bind::launch(g, 0);
                CK(cudaDeviceSynchronize());

                std::vector<float> ygot(static_cast<size_t>(M) * O);
                if (odt == gv::O_F32)
                {
                    CK(cudaMemcpy(ygot.data(), dy, ygot.size() * 4, cudaMemcpyDeviceToHost));
                }
                else if (odt == gv::O_F16)
                {
                    std::vector<__half> tmp(ygot.size());
                    CK(cudaMemcpy(tmp.data(), dy, tmp.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < tmp.size(); ++i) ygot[i] = __half2float(tmp[i]);
                }
                else
                {
                    std::vector<uint16_t> tmp(ygot.size());
                    CK(cudaMemcpy(tmp.data(), dy, tmp.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < tmp.size(); ++i) ygot[i] = bf16_to_f(tmp[i]);
                }
                ErrStats st = compare_f32(ygot, yref);
                const float tol = (odt == gv::O_F32) ? 1e-5f : (odt == gv::O_F16 ? 1e-3f : 6e-3f);
                if (st.rel_l2 > tol || st.cos < 0.99998)
                {
                    fprintf(stderr, "FAIL decode g%d M=%d odt=%d max_abs=%.6g rel_l2=%.6g cos=%.8f\n",
                            G, M, odt, st.max_abs, st.rel_l2, st.cos);
                    exit(1);
                }
            }
        }
        printf("  PASS g%-3d: all M in {1,2,4,8,16,32} x {F32,F16,BF16} matched FP32 CPU ref\n", G);
        fflush(stdout);
        CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(dxf)); CK(cudaFree(dy));
    }
}

static void test_prefill_gemm()
{
    printf("=== 2. Prefill GEMM (i8gemm::launch, pair w2, Rs scatter) FP32 & Decode Parity ===\n");
    const int Gs[] = {16, 32, 64, 128};
    const int Ms[] = {1, 7, 32, 63, 64, 129, 256};
    const int O = 384, K = 512;

    for (int G : Gs)
    {
        const int grp = G / 16;
        const int codec = bind::int8_codec_of(grp);
        std::vector<__half> x_max;
        std::vector<int8_t> q, q2;
        std::vector<__half> s, s2;
        make_random_data(256, O, K, G, 2000u + G, x_max, q, s);
        std::vector<__half> dummy_x;
        make_random_data(1, O, K, G, 3000u + G, dummy_x, q2, s2);
        auto rec = pack_int8_record(q, s, O, K, G);
        auto rec2 = pack_int8_record(q2, s2, O, K, G);

        uint8_t *dw = nullptr, *dw2 = nullptr;
        __half* dx = nullptr;
        void *dy = nullptr, *dy2 = nullptr;
        u32* dxf = nullptr;
        float* dy_gv = nullptr;
        CK(cudaMalloc(&dw, rec.size())); CK(cudaMemcpy(dw, rec.data(), rec.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dw2, rec2.size())); CK(cudaMemcpy(dw2, rec2.data(), rec2.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x_max.size() * sizeof(__half))); CK(cudaMemcpy(dx, x_max.data(), x_max.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dy, static_cast<size_t>(256) * O * sizeof(float)));
        CK(cudaMalloc(&dy2, static_cast<size_t>(256) * O * sizeof(float)));
        CK(cudaMalloc(&dxf, 4 * (K / 16) * 64 * sizeof(u32)));
        CK(cudaMalloc(&dy_gv, static_cast<size_t>(32) * O * sizeof(float)));

        for (int M : Ms)
        {
            std::vector<__half> xm(x_max.begin(), x_max.begin() + static_cast<size_t>(M) * K);
            std::vector<float> yref, yref2;
            cpu_int8_matmul(xm, q, s, M, O, K, G, yref);
            cpu_int8_matmul(xm, q2, s2, M, O, K, G, yref2);

            // Single F32, F16, BF16
            for (int bf16 : {2, 0, 1})
            {
                i8gemm::Args a{};
                a.x = dx; a.w = dw; a.y = dy;
                a.M = M; a.N = O; a.K = K; a.ldx = K; a.ldy = O;
                a.grp = grp; a.bf16 = bf16;
                i8gemm::launch(a, 0);
                CK(cudaDeviceSynchronize());

                std::vector<float> ygot(static_cast<size_t>(M) * O);
                if (bf16 == 2) CK(cudaMemcpy(ygot.data(), dy, ygot.size() * 4, cudaMemcpyDeviceToHost));
                else if (bf16 == 0)
                {
                    std::vector<__half> tmp(ygot.size());
                    CK(cudaMemcpy(tmp.data(), dy, tmp.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < tmp.size(); ++i) ygot[i] = __half2float(tmp[i]);
                }
                else
                {
                    std::vector<uint16_t> tmp(ygot.size());
                    CK(cudaMemcpy(tmp.data(), dy, tmp.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < tmp.size(); ++i) ygot[i] = bf16_to_f(tmp[i]);
                }
                ErrStats st = compare_f32(ygot, yref);
                const float tol = (bf16 == 2) ? 5e-4f : (bf16 == 0 ? 1.5e-3f : 6e-3f);
                if (st.rel_l2 > tol || st.cos < 0.99998)
                {
                    fprintf(stderr, "FAIL prefill g%d M=%d bf16=%d max_abs=%.6g rel_l2=%.6g cos=%.8f\n",
                            G, M, bf16, st.max_abs, st.rel_l2, st.cos);
                    exit(1);
                }

                // Direct decode-vs-prefill comparison at M <= 32, F32
                if (bf16 == 2 && M <= 32)
                {
                    bind::Tensor wt; wt.p = dw; wt.nd = 3; wt.shape[0] = O / 16; wt.shape[1] = K / G; wt.shape[2] = grp * 256 + 32;
                    bind::Xfrag xf{}; xf.on = 1; xf.src = dx; xf.ld = K; xf.K = K; xf.M = M; xf.f16 = dxf;
                    bind::launch(xf, 0);
                    bind::Gv gv_op;
                    const char* e = bind::make_gv(wt, codec, 2, 4, M, dxf, dy_gv, gv::O_F32, gv_op);
                    if (e) die(e);
                    bind::launch(gv_op, 0);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> ygv(static_cast<size_t>(M) * O);
                    CK(cudaMemcpy(ygv.data(), dy_gv, ygv.size() * 4, cudaMemcpyDeviceToHost));
                    ErrStats dp = compare_f32(ygot, ygv);
                    if (dp.rel_l2 > 5e-4f || dp.cos < 0.999999)
                    {
                        fprintf(stderr, "FAIL decode-vs-prefill g%d M=%d max_abs=%.6g rel_l2=%.6g cos=%.9f\n", G, M, dp.max_abs, dp.rel_l2, dp.cos);
                        exit(1);
                    }
                }
            }

            // Paired w2 launch (gate + up)
            {
                i8gemm::Args a{};
                a.x = dx; a.w = dw; a.y = dy; a.w2 = dw2; a.y2 = dy2;
                a.M = M; a.N = O; a.N2 = O; a.K = K; a.ldx = K; a.ldy = O; a.ldy2 = O;
                a.grp = grp; a.bf16 = 2;
                i8gemm::launch(a, 0);
                CK(cudaDeviceSynchronize());
                std::vector<float> g1(static_cast<size_t>(M) * O), g2(static_cast<size_t>(M) * O);
                CK(cudaMemcpy(g1.data(), dy, g1.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(g2.data(), dy2, g2.size() * 4, cudaMemcpyDeviceToHost));
                ErrStats s1 = compare_f32(g1, yref), s2 = compare_f32(g2, yref2);
                if (s1.rel_l2 > 5e-4f || s2.rel_l2 > 5e-4f)
                    die("paired w2 prefill mismatch");
            }

            // Rs scatter check at M = 64
            if (M == 64)
            {
                const size_t sbytes = static_cast<size_t>(20) * O * 2;
                char* drs = nullptr;
                uint64_t* dstep = nullptr;
                CK(cudaMalloc(&drs, 4 * sbytes));
                CK(cudaMemset(drs, 0, 4 * sbytes));
                CK(cudaMalloc(&dstep, sizeof(uint64_t)));
                CK(cudaMemset(dstep, 0, sizeof(uint64_t)));
                i8gemm::Args a{};
                a.x = dx; a.w = dw; a.y = dy;
                a.M = M; a.N = O; a.K = K; a.ldx = K; a.ldy = O;
                a.grp = grp; a.bf16 = 1;
                i8gemm::Rs rs{drs, 0, sbytes, dstep, M, 4};
                i8gemm::launch(a, 0, rs);
                CK(cudaDeviceSynchronize());
                std::vector<uint8_t> hrs(4 * sbytes);
                CK(cudaMemcpy(hrs.data(), drs, 4 * sbytes, cudaMemcpyDeviceToHost));
                std::vector<float> flat(static_cast<size_t>(M) * O);
                for (int r = 0; r < M; ++r)
                {
                    const int slot = r / 16, local = r % 16;
                    const uint16_t* rowp = reinterpret_cast<const uint16_t*>(hrs.data() + slot * sbytes) + static_cast<size_t>(local) * O;
                    for (int o = 0; o < O; ++o) flat[static_cast<size_t>(r) * O + o] = bf16_to_f(rowp[o]);
                }
                ErrStats st = compare_f32(flat, yref);
                if (st.rel_l2 > 6e-3f || st.cos < 0.99998) die("Rs scatter mismatch");
                CK(cudaFree(drs));
                CK(cudaFree(dstep));
            }
        }
        printf("  PASS g%-3d: all M in {1,7,32,63,64,129,256} x {F32,F16,BF16,pair,Rs,decode-parity} passed\n", G);
        fflush(stdout);
        CK(cudaFree(dw)); CK(cudaFree(dw2)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dy2)); CK(cudaFree(dxf)); CK(cudaFree(dy_gv));
    }
}

static void test_k1_dense_mlp()
{
    printf("=== 3. K1 Dense MLP (k1_shared_fc1 + k1_shared_fc2 native int8 gG) FP32 Parity ===\n");
    fflush(stdout);
    const int K = HID, I = 1024;
    for (int G : {16, 32, 64, 128})
    {
        const int grp = G / 16;
        std::vector<__half> x_max, sg, su, sd, d1, d2;
        std::vector<int8_t> qg, qu, qd;
        make_random_data(8, I, K, G, 4000u + G, x_max, qg, sg);
        make_random_data(1, I, K, G, 5000u + G, d1, qu, su);
        make_random_data(1, K, I, G, 6000u + G, d2, qd, sd);
        auto rg = pack_int8_record(qg, sg, I, K, G);
        auto ru = pack_int8_record(qu, su, I, K, G);
        auto rd = pack_int8_record(qd, sd, K, I, G);

        uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr;
        __half* dx = nullptr;
        uint16_t* dy = nullptr;
        CK(cudaMalloc(&dg, rg.size())); CK(cudaMemcpy(dg, rg.data(), rg.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&du, ru.size())); CK(cudaMemcpy(du, ru.data(), ru.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dd, rd.size())); CK(cudaMemcpy(dd, rd.data(), rd.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x_max.size() * sizeof(__half))); CK(cudaMemcpy(dx, x_max.data(), x_max.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dy, static_cast<size_t>(8) * K * sizeof(uint16_t)));

        const size_t sb = bind::scratch_bytes(8);
        char* smem = nullptr;
        CK(cudaMalloc(&smem, sb));
        CK(cudaMemset(smem, 0, sb));
        bind::Scratch sc = bind::scratch_at(smem, 8);

        bind::LayerTensors T;
        T.t[bind::T_D_GATE_I8] = bind::Tensor{dg, {I / 16, K / G, grp * 256 + 32, 0}, 3, 0};
        T.t[bind::T_D_UP_I8]   = bind::Tensor{du, {I / 16, K / G, grp * 256 + 32, 0}, 3, 0};
        T.t[bind::T_D_DOWN_I8] = bind::Tensor{dd, {K / 16, I / G, grp * 256 + 32, 0}, 3, 0};

        for (int M : {1, 4, 8})
        {
            bind::Xfrag xf{};
            xf.on = 1; xf.src = dx; xf.ld = K; xf.K = K; xf.M = M; xf.f16 = sc.xf_x2;
            bind::launch(xf, 0);

            bind::K1 k1;
            bind::Tensor sh[3];
            const char* e = bind::make_k1(T, M, sc, dx, sc.xf_x2, bind::PlanPtrs{}, dy, 1, k1, sh);
            if (e) die(e);
            if (k1.a.s_i8_fc1 != grp || k1.a.s_i8_fc2 != grp) die("k1 did not select native int8 grp");
            bind::launch_k1(k1, 0);
            CK(cudaDeviceSynchronize());

            std::vector<uint16_t> hbf(static_cast<size_t>(M) * K);
            CK(cudaMemcpy(hbf.data(), dy, hbf.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<float> ygot(hbf.size());
            for (size_t i = 0; i < hbf.size(); ++i) ygot[i] = bf16_to_f(hbf[i]);

            // CPU SwiGLU reference
            std::vector<__half> xm(x_max.begin(), x_max.begin() + static_cast<size_t>(M) * K);
            std::vector<float> g_ref, u_ref, y_ref;
            cpu_int8_matmul(xm, qg, sg, M, I, K, G, g_ref);
            cpu_int8_matmul(xm, qu, su, M, I, K, G, u_ref);
            std::vector<__half> mid(static_cast<size_t>(M) * I);
            for (size_t i = 0; i < mid.size(); ++i)
            {
                const float gv = g_ref[i];
                const float act = (gv / (1.f + expf(-gv))) * u_ref[i];
                mid[i] = __float2half_rn(act);
            }
            cpu_int8_matmul(mid, qd, sd, M, K, I, G, y_ref);
            ErrStats st = compare_f32(ygot, y_ref);
            if (st.cos < 0.9999 || st.rel_l2 > 1e-2f)
            {
                fprintf(stderr, "FAIL K1 dense g%d M=%d max_abs=%.6g rel_l2=%.6g cos=%.8f\n",
                        G, M, st.max_abs, st.rel_l2, st.cos);
                exit(1);
            }
        }
        printf("  PASS g%-3d: K1 dense MLP M in {1,4,8} matched FP32 SwiGLU CPU ref\n", G);
        fflush(stdout);
        CK(cudaFree(dg)); CK(cudaFree(du)); CK(cudaFree(dd)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(smem));
    }
}

template <class Fn>
static float bench_us(Fn fn, int warmup = 25, int iters = 120)
{
    for (int i = 0; i < warmup; ++i) fn();
    CK(cudaDeviceSynchronize());
    std::vector<float> times(iters);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < iters; ++i)
    {
        CK(cudaEventRecord(a, 0));
        fn();
        CK(cudaEventRecord(b, 0));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        times[i] = ms * 1000.f;
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

static void bench_decode_gemv()
{
    printf("\n=== 4A. sm_121a Decode GEMV Microbenchmark (GLM-5.3 TP4 shapes, median us & GB/s) ===\n");
    struct Shape { const char* name; int N, K, tpc, ks; };
    const Shape shapes[] = {
        {"q_a",        2048, 6144, 1, 16},
        {"q_b",        4096, 2048, 2, 8},
        {"o_proj",     6144, 4096, 2, 16},
        {"dense.gate", 4608, 6144, 2, 16},
    };
    // Allocate a pool of 4 copies (> 96 MB L2) so decode reads DRAM cleanly
    const int POOL = 4;
    const int Ms[] = {1, 4, 8, 16, 32};

    for (const auto& sh : shapes)
    {
        printf("\nShape %s [N=%d, K=%d]:\n", sh.name, sh.N, sh.K);
        printf("  %-10s %8s %8s %8s %8s %8s %10s\n", "Format", "M=1 us", "M=4 us", "M=8 us", "M=16 us", "M=32 us", "M=1 GB/s");
        struct Fmt { const char* name; int codec; size_t rec_b; int gk; };
        const Fmt fmts[] = {
            {"Int4 g32",  bind::INT4,      288,  32},
            {"MXFP8",     bind::MXFP8,     528,  32},
            {"Int8 g128", bind::INT8_G128, 2080, 128},
            {"Int8 g64",  bind::INT8_G64,  1056, 64},
            {"Int8 g32",  bind::INT8_G32,  544,  32},
            {"Int8 g16",  bind::INT8_G16,  288,  16},
            {"BF16W",     bind::BF16W,     512,  16},
        };
        u32* dxf = nullptr;
        void* dy = nullptr;
        CK(cudaMalloc(&dxf, 4 * (sh.K / 16) * 64 * sizeof(u32)));
        CK(cudaMemset(dxf, 0x3c, 4 * (sh.K / 16) * 64 * sizeof(u32)));
        CK(cudaMalloc(&dy, static_cast<size_t>(32) * sh.N * 2));

        for (const auto& f : fmts)
        {
            const size_t nbytes = static_cast<size_t>(sh.N / 16) * (sh.K / f.gk) * f.rec_b;
            uint8_t* dw = nullptr;
            CK(cudaMalloc(&dw, nbytes * POOL));
            CK(cudaMemset(dw, 0x11, nbytes * POOL));
            int ks = sh.ks;
            while (ks > 1 && (sh.K / f.gk) % ks) ks >>= 1;
            float res[5] = {};
            for (int mi = 0; mi < 5; ++mi)
            {
                bind::Tensor wt;
                wt.p = dw; wt.nd = 3;
                wt.shape[0] = sh.N / 16; wt.shape[1] = sh.K / f.gk; wt.shape[2] = f.rec_b;
                bind::Gv g[POOL];
                for (int p = 0; p < POOL; ++p)
                {
                    wt.p = dw + p * nbytes;
                    const char* e = bind::make_gv(wt, f.codec, sh.tpc, ks, Ms[mi], dxf, dy, gv::O_F16, g[p]);
                    if (e) die(e);
                }
                int idx = 0;
                res[mi] = bench_us([&]() { bind::launch(g[idx++ % POOL], 0); }, 20, 80);
            }
            const double gbps = (static_cast<double>(nbytes) / 1e9) / (res[0] / 1e6);
            printf("  %-10s %8.1f %8.1f %8.1f %8.1f %8.1f %10.1f\n",
                   f.name, res[0], res[1], res[2], res[3], res[4], gbps);
            fflush(stdout);
            CK(cudaFree(dw));
        }
        CK(cudaFree(dxf)); CK(cudaFree(dy));
    }
}

static void bench_prefill_gemm()
{
    printf("\n=== 4B. sm_121a Prefill GEMM Microbenchmark (k_gemm_i8g vs MXFP8 k_gemm_mx8b vs cuBLAS BF16) ===\n");
    cublasHandle_t blas = nullptr;
    CUBLAS_CK(cublasCreate(&blas));
    CUBLAS_CK(cublasSetStream(blas, 0));

    struct Shape { const char* name; int N, K; };
    const Shape shapes[] = {
        {"q_b",        4096, 2048},
        {"o_proj",     6144, 4096},
        {"dense.gate", 4608, 6144},
        {"dense.down", 6144, 4608},
    };
    const int Ms[] = {64, 256, 1024, 4096};

    for (const auto& sh : shapes)
    {
        printf("\nShape %s [N=%d, K=%d]:\n", sh.name, sh.N, sh.K);
        printf("  %-12s %12s %12s %12s %12s\n", "Format", "M=64 (TF/s)", "M=256 (TF/s)", "M=1024 (TF/s)", "M=4096 (TF/s)");

        __half* dx16 = nullptr;
        uint16_t *dxbf = nullptr, *dwbf = nullptr, *dy = nullptr;
        uint8_t *dx8 = nullptr, *dmx_c = nullptr, *dmx_s = nullptr;
        CK(cudaMalloc(&dx16, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMemset(dx16, 0x3c, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMalloc(&dxbf, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMemset(dxbf, 0x3f, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMalloc(&dwbf, static_cast<size_t>(sh.N) * sh.K * 2));
        CK(cudaMemset(dwbf, 0x3f, static_cast<size_t>(sh.N) * sh.K * 2));
        CK(cudaMalloc(&dy, static_cast<size_t>(4096) * sh.N * 2));
        CK(cudaMalloc(&dx8, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMemset(dx8, 0x38, static_cast<size_t>(4096) * sh.K * 2));
        CK(cudaMalloc(&dmx_c, static_cast<size_t>(sh.N) * sh.K));
        CK(cudaMemset(dmx_c, 0x38, static_cast<size_t>(sh.N) * sh.K));
        CK(cudaMalloc(&dmx_s, static_cast<size_t>(sh.N) * (sh.K / 32)));
        CK(cudaMemset(dmx_s, 127, static_cast<size_t>(sh.N) * (sh.K / 32)));

        auto print_row = [&](const char* label, float us[4]) {
            printf("  %-12s", label);
            for (int i = 0; i < 4; ++i)
            {
                const double tf = (2.0 * Ms[i] * sh.N * sh.K) / (us[i] * 1e6);
                printf(" %5.0fus(%4.1f)", us[i], tf);
            }
            printf("\n");
            fflush(stdout);
        };

        // 1. cuBLAS BF16 baseline
        {
            float us[4];
            const float one = 1.f, zero = 0.f;
            for (int i = 0; i < 4; ++i)
                us[i] = bench_us([&]() {
                    CUBLAS_CK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, Ms[i], sh.K,
                                           &one, dwbf, CUDA_R_16BF, sh.K,
                                           dxbf, CUDA_R_16BF, sh.K,
                                           &zero, dy, CUDA_R_16BF, sh.N,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }, 10, 40);
            print_row("cuBLAS BF16", us);
        }

        // 2. Existing gr8 MXFP8 prefill GEMM (bindp::launch / k_gemm_mx8b)
        {
            bind::Tensor tc{dmx_c, {sh.N, sh.K, 0, 0}, 2, 0};
            bind::Tensor ts{dmx_s, {sh.N, sh.K / 32, 0, 0}, 2, 0};
            float us[4];
            for (int i = 0; i < 4; ++i)
            {
                bindp::Gemm g;
                const char* e = bindp::make_gemm(tc, ts, Ms[i], dx8, dy, 1, g);
                if (e) die(e);
                us[i] = bench_us([&]() { bindp::launch(g, 0); }, 10, 40);
            }
            print_row("gr8 MXFP8", us);
        }

        // 3. Native k_gemm_i8g across g128, g64, g32, g16
        for (int grp : {8, 4, 2, 1})
        {
            const int G = grp * 16;
            const size_t nbytes = static_cast<size_t>(sh.N / 16) * (sh.K / G) * (grp * 256 + 32);
            uint8_t* dw = nullptr;
            CK(cudaMalloc(&dw, nbytes));
            CK(cudaMemset(dw, 0x11, nbytes));
            float us[4];
            for (int i = 0; i < 4; ++i)
            {
                i8gemm::Args a{};
                a.x = dx16; a.w = dw; a.y = dy;
                a.M = Ms[i]; a.N = sh.N; a.K = sh.K; a.ldx = sh.K; a.ldy = sh.N;
                a.grp = grp; a.bf16 = 1;
                us[i] = bench_us([&]() { i8gemm::launch(a, 0); }, 10, 40);
            }
            char lbl[32];
            snprintf(lbl, sizeof lbl, "i8gemm g%d", G);
            print_row(lbl, us);
            CK(cudaFree(dw));
        }

        CK(cudaFree(dx16)); CK(cudaFree(dxbf)); CK(cudaFree(dwbf)); CK(cudaFree(dy));
        CK(cudaFree(dx8)); CK(cudaFree(dmx_c)); CK(cudaFree(dmx_s));
    }
    CUBLAS_CK(cublasDestroy(blas));
}

int main(int argc, char** argv)
{
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--bench") == 0) bench = true;

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);

    test_decode_gemv();
    test_prefill_gemm();
    test_k1_dense_mlp();
    if (bench)
    {
        bench_decode_gemv();
        bench_prefill_gemm();
    }
    printf("\nALL INT8 DENSE TESTS PASSED\n");
    return 0;
}
