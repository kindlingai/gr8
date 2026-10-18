// #50: kv_b (UK / UV) in FP8. One GPU, synthetic weights in the release's form: kv_b of 64 heads as E4M3 [64 x 448][512]
// with one f32 scale per 128 x 128 block, sliced to one rank's heads as the loader slices it (engine/loader-rs/src/release.rs
// kv_b_fp8: the E4M3 rows of UK / UV and each row's 4 block scales). The BF16 form of the same weights is BF16(e4m3 x
// scale), what the BF16 loader stores.
//   1. g6_attn_kvb_bf16 (the QO / pair-owner staging) against the host's BF16(e4m3 x scale): every byte.
//   2. Two bindings of one layer, BF16 and FP8, on the same KV rows and queries: decode (g6_attn_bind_run) under each
//      absorb / value_project variant (absorb_m / value_project_m, G6_AVP1, G6_ATTN_QT) and prefill
//      (g6_attn_bind_prefill_run) must give byte-identical outputs.
//   3. --bench: absorb_m and value_project_m of one layer (their launchers: the BF16 or FP8 instantiation), BF16 against FP8, at M rows with a cold L2 (a 96 MB write
//      between calls, as the experts' stream between two layers' attention when serving): us per call.
//   kvb8_test [--bench] [--layer L] [--seed S]
// Build and run: TEST=kvb8_test G6_NR=4 scripts/c1_tests.sh <bin_dir> [args]  (the K1H2 flags; attn.cu included, bind.cu)
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../attn/attn.cu"
#include "../../attn/bind.h"

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define CKI(x) do { const int e_ = (x); if (e_) { printf("call failed (%d) %s at %s:%d\n", e_, #x, __FILE__, __LINE__); exit(2); } } while (0)

template<class T> static T* up(const std::vector<T>& v) { T* d = nullptr; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template<class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

static float e4m3f(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
    return s ? -v : v;
}
static uint16_t bf16_rne(float x)   // finite x
{
    uint32_t b; memcpy(&b, &x, 4);
    return (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

static int fails = 0, passes = 0;
static void check(const char* what, bool ok) { ok ? ++passes : ++fails; printf("%s %s\n", ok ? "PASS" : "FAIL", what); }

int main(int argc, char** argv)
{
    bool bench = false;
    int L = 7;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    }
    if ((L - 2) % 4 == 0 || L <= 2 || L == 78) { printf("--layer: a shared-index layer (not 0-2, 78 or 2 + 4k)\n"); return 2; }
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    CK((cudaError_t)g6_attn_rope_init(8192, nullptr));
    const int group = (0 - L % g6nr::NR + g6nr::NR) % g6nr::NR, H = g6nr::width(g6nr::HEADS, group), h0 = g6nr::start(g6nr::HEADS, group);
    printf("kvb8_test: NR %d, layer %d, heads %d (from %d), seed %llu\n", g6nr::NR, L, H, h0, seed);

    // the release's kv_b: E4M3 [64 x 448][512], block scales [224][4]; this rank's rows h0 * 448 ..
    const int RB = 64 * 448 / 128;
    std::vector<float> bs((size_t)RB * 4);
    for (auto& s : bs) s = 0.002f * std::exp2(g(rng) * 0.5f);          // weights ~ N(0, 0.1) once scaled
    std::vector<uint8_t> uk8((size_t)H * 192 * 512), uv8((size_t)H * 256 * 512);
    std::vector<float> uks((size_t)H * 192 * 4), uvs((size_t)H * 256 * 4);
    std::vector<__nv_bfloat16> ukb(uk8.size()), uvb(uv8.size());
    auto byte = [&]() { uint8_t b; do b = (uint8_t)(__nv_fp8_e4m3(g(rng) * 50.f).__x); while ((b & 0x7f) == 0x7f); return b; };
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < 448; ++i)
        {
            const int row = (h0 + h) * 448 + i;
            const bool k = i < 192;
            uint8_t* w = k ? &uk8[((size_t)h * 192 + i) * 512] : &uv8[((size_t)h * 256 + i - 192) * 512];
            float* s = k ? &uks[((size_t)h * 192 + i) * 4] : &uvs[((size_t)h * 256 + i - 192) * 4];
            __nv_bfloat16* b = k ? &ukb[((size_t)h * 192 + i) * 512] : &uvb[((size_t)h * 256 + i - 192) * 512];
            for (int c = 0; c < 4; ++c) s[c] = bs[(size_t)(row / 128) * 4 + c];
            for (int j = 0; j < 512; ++j)
            {
                w[j] = byte();
                const uint16_t v = bf16_rne(e4m3f(w[j]) * s[j / 128]);
                memcpy(&b[j], &v, 2);
            }
        }
    uint8_t *duk8 = up(uk8), *duv8 = up(uv8);
    float *duks = up(uks), *duvs = up(uvs);
    __nv_bfloat16 *dukb = up(ukb), *duvb = up(uvb);

    // 1. the BF16 expansion of the staging paths
    {
        __nv_bfloat16* x = nullptr;
        CK(cudaMalloc(&x, uv8.size() * 2));
        CKI(g6_attn_kvb_bf16(duk8, duks, x, H * 192, nullptr));
        CK(cudaDeviceSynchronize());
        check("g6_attn_kvb_bf16 UK = BF16(e4m3 x scale)", !memcmp(down(x, uk8.size()).data(), ukb.data(), uk8.size() * 2));
        CKI(g6_attn_kvb_bf16(duv8, duvs, x, H * 256, nullptr));
        CK(cudaDeviceSynchronize());
        check("g6_attn_kvb_bf16 UV = BF16(e4m3 x scale)", !memcmp(down(x, uv8.size()).data(), uvb.data(), uv8.size() * 2));
        CK(cudaFree(x));
    }

    // 2. two bindings, the same KV and queries
    std::vector<__nv_bfloat16> kvn(512);
    for (auto& x : kvn) x = __float2bfloat16(1.f + 0.1f * g(rng));
    __nv_bfloat16* dkvn = up(kvn);
    G6AttnPack pb{}, p8{};
    pb.uk = G6AttnTensor{dukb, 3, {H, 192, 512}}; pb.uv = G6AttnTensor{duvb, 3, {H, 256, 512}};
    pb.kv_norm = G6AttnTensor{dkvn, 1, {512, 0, 0}};
    p8 = pb;
    p8.uk.data = duk8; p8.uv.data = duv8;
    p8.uks = G6AttnTensor{duks, 3, {H, 192, 4}}; p8.uvs = G6AttnTensor{duvs, 3, {H, 256, 4}};
    {
        G6AttnPack bad = p8; bad.uvs = G6AttnTensor{};
        G6AttnBinding x{};
        check("FP8 pack without UV scales is refused", g6_attn_bind_create(&x, &bad, 0, L, 4096, 8, 8, nullptr) != 0);
    }
    for (int T : {700, 3000})
        for (int M : {1, 3, 8, 24})
        {
            const int N = T + M, CAP = (N + 63) / 64 * 64 + 64, QT = std::min(M, 64);
            G6AttnBinding bb{}, b8{};
            CKI(g6_attn_bind_create(&bb, &pb, 0, L, CAP, M, QT, nullptr));
            CKI(g6_attn_bind_create(&b8, &p8, 0, L, CAP, M, QT, nullptr));
            check("FP8 binding holds less than the BF16 one", b8.storage_bytes < bb.storage_bytes);
            std::vector<__half> kva((size_t)N * 576), qb((size_t)M * H * 256);
            for (auto& x : kva) x = __float2half(g(rng));
            for (auto& x : qb) x = __float2half(1.5f * g(rng));
            std::vector<int> pos(N), ids((size_t)M * 2048, -1);
            for (int t = 0; t < N; ++t) pos[t] = t;
            for (int i = 0; i < M; ++i)    // a 2048-key list over 0 .. T + i (dead tails at short contexts)
                for (int j = 0; j < 2048; ++j) ids[(size_t)i * 2048 + j] = T + i < 2048 ? (j <= T + i ? j : -1) : (int)(rng() % (T + i + 1));
            int *dpos = up(pos), *dids = up(ids);
            __half *dkva = up(kva), *dqb = up(qb);
            for (G6AttnBinding* b : {&bb, &b8})
            {
                G6Attn a = b->args;
                a.rows = N; a.context = N; a.positions = dpos; a.kv_a = dkva; a.reuse_indices = dids; a.reuse_rows = M;
                CKI(g6_attn_append(&a, nullptr));
            }
            G6AttnStep s{};
            s.rows = M; s.context = N; s.q_b = dqb; s.kv_a = dkva + (size_t)T * 576; s.positions = dpos + T;
            s.reuse_indices = dids; s.reuse_rows = M;
            const size_t no = (size_t)M * H * 256;
            char what[160];
            auto both = [&](const char* path, auto run) {
                CK(cudaMemset(bb.args.output, 0x55, no * 2)); CK(cudaMemset(b8.args.output, 0x33, no * 2));
                run(&bb); run(&b8);
                CK(cudaDeviceSynchronize());
                const bool e0 = down(bb.args.error, 1)[0] == 0 && down(b8.args.error, 1)[0] == 0;
                const std::vector<__half> ob = down((const __half*)bb.args.output, no), o8 = down((const __half*)b8.args.output, no);
                snprintf(what, sizeof what, "%s T %d M %d: FP8 output = BF16 output", path, T, M);
                check(what, e0 && !memcmp(ob.data(), o8.data(), no * 2));
            };
            auto decode = [&](G6AttnBinding* b) { CKI(g6_attn_bind_run(b, &s, nullptr)); };
            both("decode", decode);
            ga::avp1_on() = 1; both("decode G6_AVP1", decode); ga::avp1_on() = 0;
            ga::attn_qt() = 1; ga::attn_sc2() = 4; both("decode G6_ATTN_QT", decode); ga::attn_qt() = 0; ga::attn_sc2() = 0;
            G6AttnPrefill *fb = nullptr, *f8 = nullptr;
            CKI(g6_attn_bind_prefill_create(&fb, &bb, nullptr));
            CKI(g6_attn_bind_prefill_create(&f8, &b8, nullptr));
            both("prefill", [&](G6AttnBinding* b) { CKI(g6_attn_bind_prefill_run(b == &bb ? fb : f8, b, &s, nullptr)); });
            CKI(g6_attn_bind_prefill_release(fb)); CKI(g6_attn_bind_prefill_release(f8));
            CKI(g6_attn_bind_release(&bb)); CKI(g6_attn_bind_release(&b8));
            for (void* p : {(void*)dpos, (void*)dids, (void*)dkva, (void*)dqb}) CK(cudaFree(p));
        }

    // 3. absorb_m / value_project_m, cold L2
    if (bench)
    {
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, 0));
        int clk = 0;
        CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
        printf("BENCH device %s, L2 %d MB, reported max clock %d MHz\n", prop.name, prop.l2CacheSize >> 20, clk / 1000);
        const size_t FL = (size_t)96 << 20;
        void* flush = nullptr;
        CK(cudaMalloc(&flush, FL));
        G6AttnBinding bb{}, b8{};
        const int MX = 32;
        CKI(g6_attn_bind_create(&bb, &pb, 0, L, 4096, MX, MX, nullptr));
        CKI(g6_attn_bind_create(&b8, &p8, 0, L, 4096, MX, MX, nullptr));
        std::vector<int> pos(MX);
        for (int i = 0; i < MX; ++i) pos[i] = 1000 + i;
        int* dpos = up(pos);
        std::vector<__half> qb((size_t)MX * H * 256);
        for (auto& x : qb) x = __float2half(g(rng));
        __half* dqb = up(qb);
        float *q = nullptr, *rope = nullptr, *ctx = nullptr;
        CK(cudaMalloc(&q, (size_t)MX * H * 512 * 4)); CK(cudaMalloc(&rope, (size_t)MX * H * 64 * 4)); CK(cudaMalloc(&ctx, (size_t)MX * H * 512 * 4));
        CK(cudaMemset(ctx, 0, (size_t)MX * H * 512 * 4));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int REPS = 200;
        auto timed = [&](auto launch) {   // median us of REPS cold calls
            std::vector<float> t(REPS);
            for (int i = 0; i < REPS; ++i)
            {
                CK(cudaMemsetAsync(flush, i & 0xff, FL));
                CK(cudaEventRecord(e0)); launch(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&t[i], e0, e1));
            }
            std::sort(t.begin(), t.end());
            return t[REPS / 2] * 1000.f;
        };
        printf("BENCH heads %d: UK %.2f MB / UV %.2f MB a layer in BF16, %.2f / %.2f MB in FP8 (+ scales)\n", H,
               H * 192 * 512 * 2 / 1e6, H * 256 * 512 * 2 / 1e6, H * 192 * 512 / 1e6, H * 256 * 512 / 1e6);
        printf("BENCH rows | absorb_m BF16 FP8 us | value_project_m BF16 FP8 us | sum saved us | x 78 layers ms\n");
        for (int M : {1, 3, 4, 8, 16, 24})
        {
            float t[2][2];
            for (int k = 0; k < 2; ++k)
            {
                G6Attn a = (k ? b8 : bb).args;
                a.rows = M; a.q_b = dqb; a.positions = dpos; a.output = (k ? b8 : bb).args.output;
                ga::Work w{}; w.q = q; w.rope = rope; w.context = ctx;
                t[k][0] = timed([&] { ga::launch_absorb_m(a, a, w, 0, M, nullptr); });
                t[k][1] = timed([&] { ga::launch_value_project_m(a, a, w, 0, M, nullptr); });
                CK(cudaGetLastError());
            }
            const float saved = t[0][0] + t[0][1] - t[1][0] - t[1][1];
            printf("BENCH %4d | %8.1f %8.1f | %8.1f %8.1f | %8.1f | %6.3f\n", M, t[0][0], t[1][0], t[0][1], t[1][1], saved, saved * 78 / 1000);
        }
        for (void* p : {(void*)dpos, (void*)dqb, (void*)q, (void*)rope, (void*)ctx, flush}) CK(cudaFree(p));
        CKI(g6_attn_bind_release(&bb)); CKI(g6_attn_bind_release(&b8));
    }
    printf("kvb8_test: %d PASS, %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
