// DS41 Attention B test of the C entry points the step ops call (engine/attn/ds41_idx_api.h, compiled in ds41_bind.cu): device-position
// paths (dpos), klen, the bf16 GEMV, the block-scaled fp8 GEMM, the ca/ica row producers (ratio 1 and 2), and the dense -> candidate-blocks ->
// candidate-consumer chain on the idx_fixture.py data. Needs a GPU that runs sm_121a (a Spark).
//   python3 engine/ref/ds41/idx_fixture.py /tmp/ds41_idx_fx
//   nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I. -Iengine engine/attn/tests/ds41_idx_api_test.cu engine/attn/ds41_bind.cu engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm -o /tmp/ds41_idx_api_test
//   /tmp/ds41_idx_api_test /tmp/ds41_idx_fx
// This TU must not include ds41_idx.cuh (ds41_bind.cu has the kernels). EXACT = 0 mismatches; NEAR = last-bf16-bit differences of rope /
// matmul order on under 0.5% of the elements. Exit code = number of failed checks.
#include "../ds41_idx_api.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

typedef __nv_bfloat16 bf16;
static std::string g_dir;
static int g_fail = 0;

static std::vector<uint8_t> rd(const char* name)
{
    std::string p = g_dir + "/" + name + ".bin";
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f);
    return v;
}
template <class T> static T* dmem(size_t bytes, int fill = 0xAB)
{
    void* d;
    CK(cudaMalloc(&d, bytes ? bytes : 1));
    CK(cudaMemset(d, fill, bytes ? bytes : 1));
    return (T*)d;
}
template <class T> static T* up(const void* h, size_t bytes)
{
    T* d = dmem<T>(bytes);
    CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    return d;
}
template <class T> static T* up(const std::vector<uint8_t>& v) { return up<T>(v.data(), v.size()); }
static std::vector<uint8_t> down(const void* d, size_t bytes)
{
    std::vector<uint8_t> v(bytes);
    CK(cudaMemcpy(v.data(), d, bytes, cudaMemcpyDeviceToHost));
    return v;
}
static float bf2f(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t f2bf(float f) { bf16 b = __float2bfloat16_rn(f); uint16_t u; memcpy(&u, &b, 2); return u; }

// elem 1 / 2 / 4 bytes; exact: 0 mismatches; else under 0.5% of the elements
static void check(const char* name, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, size_t elem, bool exact)
{
    size_t n = want.size() / elem, bad = 0;
    if (got.size() != want.size()) { printf("%-34s FAIL size %zu vs %zu\n", name, got.size(), want.size()); g_fail++; return; }
    for (size_t i = 0; i < n; i++)
        if (memcmp(got.data() + i * elem, want.data() + i * elem, elem)) bad++;
    const bool ok = exact ? bad == 0 : (double)bad <= 0.005 * (double)n;
    printf("%-34s %s  %zu / %zu mismatch%s\n", name, ok ? "PASS" : "FAIL", bad, n, exact ? "" : " (near)");
    if (!ok) g_fail++;
}
static void verdict(const char* name, size_t bad, size_t n)
{
    printf("%-34s %s  %zu / %zu mismatch\n", name, bad ? "FAIL" : "PASS", bad, n);
    if (bad) g_fail++;
}

static const float MAG[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: ds41_idx_api_test <fixture dir>\n"); return 2; }
    g_dir = argv[1];
    int T, POS0, N, R;
    {
        FILE* f = fopen((g_dir + "/dims.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %d %d", &T, &POS0, &N, &R) != 4) { fprintf(stderr, "dims.txt\n"); return 2; }
        fclose(f);
    }
    void* st = nullptr;                          // the default stream, as Model::stream 0 would be
    int* page0 = dmem<int>(4, 0);                // page table {0}: flat pools

    // ---------------------------------------------------------------- klen = (pos + 1) / ratio
    {
        const int hp[10] = {0, 1, 2, 3, 4, 5, 7, 8, 40000, 131071};
        int* dp = up<int>(hp, sizeof hp);
        int* kl = dmem<int>(sizeof hp);
        size_t bad = 0;
        for (int ratio = 1; ratio <= 2; ratio++)
        {
            ds41_idx_klen(dp, ratio, kl, 10, st);
            CK(cudaDeviceSynchronize());
            std::vector<uint8_t> h = down(kl, sizeof hp);
            for (int i = 0; i < 10; i++)
                if (((int*)h.data())[i] != (hp[i] + 1) / ratio) bad++;
        }
        verdict("klen (pos+1)/ratio", bad, 20);
    }

    // ---------------------------------------------------------------- bf16 GEMV (cmp.wkv N=512, idx.weights_proj N=32; K=5120)
    {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.f, 1.f);
        const int K = 5120, Rg = 5, ldx = 5120;
        for (int N : {512, 32})
        {
            std::vector<uint16_t> X((size_t)Rg * ldx), W((size_t)N * K);
            for (auto& v : X) v = f2bf(nd(rng));
            for (auto& v : W) v = f2bf(0.05f * nd(rng));
            uint16_t* dx = up<uint16_t>(X.data(), X.size() * 2);
            uint16_t* dw = up<uint16_t>(W.data(), W.size() * 2);
            uint16_t* dy = dmem<uint16_t>((size_t)Rg * N * 2);
            ds41_idx_gemv(dx, ldx, dw, N, K, dy, N, Rg, st);
            CK(cudaDeviceSynchronize());
            std::vector<uint8_t> h = down(dy, (size_t)Rg * N * 2);
            size_t bad = 0;
            for (int r = 0; r < Rg; r++)
                for (int n = 0; n < N; n++)
                {
                    double acc = 0;
                    for (int k = 0; k < K; k++) acc += (double)bf2f(X[(size_t)r * ldx + k]) * (double)bf2f(W[(size_t)n * K + k]);
                    const double got = bf2f(((uint16_t*)h.data())[(size_t)r * N + n]);
                    if (std::fabs(got - acc) > std::fabs(acc) * 0.0079 + 2e-3) bad++;   // one bf16 ulp (2^-7) of the exact value + abs slack
                }
            char nm[64];
            snprintf(nm, sizeof nm, "gemv bf16 N=%d K=%d", N, K);
            verdict(nm, bad, (size_t)Rg * N);
            cudaFree(dx); cudaFree(dw); cudaFree(dy);
        }
    }

    // ---------------------------------------------------------------- block-scaled fp8 GEMM (idx.wq_b: N=4096 rows of K=1280)
    {
        std::mt19937 rng(12);
        const int M = 6, Nn = 4096, K = 1280, KB = K / 32;
        std::vector<uint8_t> A((size_t)M * K), SA((size_t)M * KB), W((size_t)Nn * K), SW((size_t)Nn * KB);
        auto code = [&]() { uint8_t c = (uint8_t)(rng() % 0x58); return (uint8_t)(c | ((rng() & 1) ? 0x80 : 0)); };   // finite e4m3 values
        for (auto& v : A) v = code();
        for (auto& v : W) v = code();
        for (auto& v : SA) v = (uint8_t)(122 + rng() % 8);
        for (auto& v : SW) v = (uint8_t)(122 + rng() % 8);
        uint8_t *dA = up<uint8_t>(A.data(), A.size()), *dSA = up<uint8_t>(SA.data(), SA.size());
        uint8_t *dW = up<uint8_t>(W.data(), W.size()), *dSW = up<uint8_t>(SW.data(), SW.size());
        uint16_t* dy = dmem<uint16_t>((size_t)M * Nn * 2);
        ds41_idx_gemm_mx(dA, K, dSA, KB, dW, dSW, Nn, K, dy, Nn, M, st);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> h = down(dy, (size_t)M * Nn * 2);
        auto e4 = [](uint8_t c) { __nv_fp8_e4m3 f; f.__x = c; return (double)(float)f; };
        auto e8 = [](uint8_t b) { return std::ldexp(1.0, (int)b - 127); };
        size_t bad = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < Nn; n += (n < 64 || n >= Nn - 64) ? 1 : 37)    // all of the first and last tile, a stride through the rest
            {
                double acc = 0;
                for (int b = 0; b < KB; b++)
                {
                    double part = 0;
                    for (int k = 0; k < 32; k++) part += e4(A[(size_t)m * K + b * 32 + k]) * e4(W[(size_t)n * K + b * 32 + k]);
                    acc += part * e8(SA[(size_t)m * KB + b]) * e8(SW[(size_t)n * KB + b]);
                }
                const double got = bf2f(((uint16_t*)h.data())[(size_t)m * Nn + n]);
                if (std::fabs(got - acc) > std::fabs(acc) * 0.0079 + 1e-30) bad++;
            }
        verdict("gemm_mx fp8 M=6 N=4096 K=1280", bad, (size_t)M * 200);
    }

    // ---------------------------------------------------------------- ratio-1 ca row from device positions (+ the bf16 shadow)
    const int RPP = POS0 + T;
    uint8_t* d_pre1 = nullptr;
    {
        std::vector<int> hp(T);
        for (int t = 0; t < T; t++) hp[t] = POS0 + t;
        int* dp = up<int>(hp.data(), T * 4);
        uint8_t* ca = dmem<uint8_t>((size_t)RPP * 288);
        uint16_t* shadow = dmem<uint16_t>((size_t)RPP * 512 * 2);
        d_pre1 = dmem<uint8_t>((size_t)T * 512 * 2);
        uint8_t* y = up<uint8_t>(rd("ca_y"));
        uint8_t* nw = up<uint8_t>(rd("ca_normw"));
        ds41_idx_ca_row_r1(y, nw, dp, ca, page0, RPP, d_pre1, shadow, T, st);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> rows = down(ca + (size_t)POS0 * 288, (size_t)T * 288);
        check("ca pre (rmsnorm), dpos", down(d_pre1, (size_t)T * 1024), rd("ca_pre"), 2, false);
        check("ca row 288 B, dpos", rows, rd("ca_rec"), 1, false);
        std::vector<uint8_t> sh = down(shadow + (size_t)POS0 * 512, (size_t)T * 1024);
        size_t bad = 0;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 512; c++)
            {
                const uint32_t cd = (rows[(size_t)t * 288 + (c >> 1)] >> ((c & 1) * 4)) & 0xF;
                float v = MAG[cd & 7] * ((cd & 8) && (cd & 7) ? -1.f : 1.f);
                __nv_fp8_e4m3 e; e.__x = rows[(size_t)t * 288 + 256 + (c >> 4)];
                v *= (float)e;
                if (bf2f(((uint16_t*)sh.data())[(size_t)t * 512 + c]) != v) bad++;
            }
        verdict("ca shadow = dequant(row)", bad, (size_t)T * 512);
        // rows below POS0 must be untouched
        std::vector<uint8_t> below = down(ca + (size_t)(POS0 - 1) * 288, 288);
        size_t nb = 0;
        for (uint8_t b : below) nb += b != 0xAB;
        verdict("ca row POS0-1 untouched", nb, 288);
    }

    // ---------------------------------------------------------------- ica row (index key) from device positions
    uint8_t *ic_wk = up<uint8_t>(rd("ica_wk")), *ic_kn = up<uint8_t>(rd("ica_knormw"));
    auto ica_rows = [&](const uint8_t* ica, int rpp, int row0, int n, std::vector<uint8_t>& codes, std::vector<uint8_t>& sf)
    {
        codes = down(ica + (size_t)row0 * 64, (size_t)n * 64);
        sf = down(ica + (size_t)rpp * 64 + (size_t)row0 * 4, (size_t)n * 4);
    };
    {
        std::vector<int> hp(T);
        for (int t = 0; t < T; t++) hp[t] = POS0 + t;
        int* dp = up<int>(hp.data(), T * 4);
        uint8_t* ica = dmem<uint8_t>((size_t)RPP * 68);
        ds41_idx_ica_row(d_pre1, ic_wk, ic_kn, dp, 1, ica, page0, RPP, T, st);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> c, s;
        ica_rows(ica, RPP, POS0, T, c, s);
        check("ica codes ratio 1, dpos", c, rd("ica_pk"), 1, false);
        check("ica e8m0 ratio 1, dpos", s, rd("ica_ex"), 1, false);
    }
    {
        // ratio 2: only odd positions make a row; row = pos / 2; cnorm slot k = k-th completion of the call
        const int RP2 = 20008;
        uint8_t* cn = up<uint8_t>(rd("ica2_cn"));
        for (int variant = 0; variant < 2; variant++)
        {
            const int n = variant == 0 ? 6 : 4, start = variant == 0 ? 40000 : 40001, rows = variant == 0 ? 3 : 2;
            std::vector<int> hp(n);
            for (int t = 0; t < n; t++) hp[t] = start + t;
            int* dp = up<int>(hp.data(), n * 4);
            uint8_t* ica = dmem<uint8_t>((size_t)RP2 * 68);
            ds41_idx_ica_row(cn, ic_wk, ic_kn, dp, 2, ica, page0, RP2, n, st);
            CK(cudaDeviceSynchronize());
            std::vector<uint8_t> c, s;
            ica_rows(ica, RP2, 20000, rows, c, s);
            check(variant == 0 ? "ica codes ratio 2, even start" : "ica codes ratio 2, odd start", c, rd(variant == 0 ? "ica2_pk" : "ica2b_pk"), 1, false);
            check(variant == 0 ? "ica e8m0 ratio 2, even start" : "ica e8m0 ratio 2, odd start", s, rd(variant == 0 ? "ica2_ex" : "ica2b_ex"), 1, false);
            std::vector<uint8_t> a, b, a2, b2;
            ica_rows(ica, RP2, 19999, 1, a, a2);
            ica_rows(ica, RP2, 20000 + rows, 1, b, b2);
            size_t nb = 0;
            for (uint8_t v : a) nb += v != 0xAB;
            for (uint8_t v : b) nb += v != 0xAB;
            verdict("ica ratio 2: neighbours untouched", nb, 128);
        }
    }

    // ---------------------------------------------------------------- dense top-512 -> candidate blocks -> candidate consumer (fixture data)
    {
        const std::vector<uint8_t> kpk = rd("k_pk"), kex = rd("k_ex");
        std::vector<uint8_t> flat((size_t)N * 68);
        memcpy(flat.data(), kpk.data(), (size_t)N * 64);
        memcpy(flat.data() + (size_t)N * 64, kex.data(), (size_t)N * 4);
        uint8_t* ica = up<uint8_t>(flat.data(), flat.size());
        std::vector<int> hp(R);
        for (int r = 0; r < R; r++) hp[r] = POS0 + r;
        int* dp = up<int>(hp.data(), R * 4);
        int* kl = up<int>(rd("klen"));
        const int Rcap = 8;
        void* scratch = dmem<uint8_t>(ds41_idx_scratch_bytes(Rcap, N));
        int* out = dmem<int>((size_t)R * 512 * 4);
        Ds41IdxStage s{};
        s.q_b = up<uint8_t>(rd("q_b")); s.wproj = up<uint8_t>(rd("q_wproj")); s.dpos = dp; s.klen = kl; s.max_klen = N; s.R = R;
        s.ica = ica; s.pages = page0; s.rpp = N; s.scratch = scratch; s.Rcap = Rcap; s.max_rows = N; s.use_cand = 0; s.out = out; s.ldo = 512;
        ds41_idx_topk(&s, st);
        ds41_idx_cand_blocks(&s, st);
        CK(cudaDeviceSynchronize());
        check("dense top-512 (rope+logits+topk)", down(out, (size_t)R * 512 * 4), rd("topk"), 4, false);
        Ds41IdxStage c = s;
        c.q_b = up<uint8_t>(rd("q2_b")); c.wproj = up<uint8_t>(rd("q2_wproj")); c.use_cand = 1;
        int* out2 = dmem<int>((size_t)R * 512 * 4);
        c.out = out2;
        ds41_idx_topk(&c, st);
        CK(cudaDeviceSynchronize());
        check("candidate consumer top-512", down(out2, (size_t)R * 512 * 4), rd("cons_topk"), 4, false);
    }
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail;
}
