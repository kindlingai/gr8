// KV latent row test, for the FP8 row (default) and the G6_KV4 INT4 row (kvfmt.h). One GPU, synthetic weights.
//   kv4_test [--tokens T] [--rows M] [--seed S]
// 1. Round trip: T rows of kv_a go through g6_attn_append. On the host, each cached row is decoded (and rotated back by
//    H under G6_KV4) and compared with the exact RMS-normalized latent: SQNR in dB, plus the row's invariants (codes,
//    scales, the BF16 rope key).
// 2. Attention: M decode rows at positions T..T+M-1 attend to every earlier key (reuse lists 0..pos). Three paths read
//    the cache: decode (g6_attn_bind_run), prefill (g6_attn_bind_prefill_run: the FP16 key table from expand) and the
//    row-owner prefill (g6_attn_qo_run, 64 heads: prepass to the FP16 table). Each output is compared with a host
//    reference in double precision on (a) the cache's own decoded latent: the kernel error, which must be small, and
//    (b) the exact latent: the cost of the row format, which this test reports.
// Thresholds: kernel error < 1e-2 (fp16 tables and outputs); round trip >= 28 dB (FP8) / 17 dB (INT4); attention cost
// < 0.05 (FP8) / 0.25 (INT4) relative L2.
// Build and run: TEST=kv4_test G6_NR=4 [EXTRA=-DG6_KV4=1] scripts/c1_tests.sh <bin_dir> (the K1H2 flags; attn.cu, bind.cu)
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../attn/bind.h"
#include "../../attn/prefill.h"
#include "../../attn/prefill-qo-bind.h"
#include "../../attn/kvfmt.h"
#include "../../g6nr.h"

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define CKI(x) do { const int e_ = (x); if (e_) { printf("call failed (%d) %s at %s:%d\n", e_, #x, __FILE__, __LINE__); exit(2); } } while (0)

static std::mt19937_64 rng;
static float gauss(float s) { return std::normal_distribution<float>(0.f, s)(rng); }
template<class T> static T* up(const std::vector<T>& v) { T* d = nullptr; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template<class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
static float h2f(__half x) { return __half2float(x); }
static float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

static double e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}
static void fwht(double* x)   // unnormalized Sylvester Hadamard, 512 points
{
    for (int h = 1; h < 512; h <<= 1)
        for (int i = 0; i < 512; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const double a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
}
// the cached row -> the latent in the model's basis
static void decode_row(const uint8_t* r, double* c)
{
#if G6_KV4
    const __half* sc = reinterpret_cast<const __half*>(r + g6kv::SCALE_OFF);
    for (int d = 0; d < 512; ++d) c[d] = (double)(((r[d >> 1] >> ((d & 1) * 4)) & 15) - 8) * h2f(sc[d >> 5]);
    fwht(c);
    for (int d = 0; d < 512; ++d) c[d] /= std::sqrt(512.0);
#else
    const float* sc = reinterpret_cast<const float*>(r + g6kv::SCALE_OFF);
    for (int d = 0; d < 512; ++d) c[d] = e4m3(r[d]) * sc[d / 128];
#endif
}
static void rope(double x, double y, int pos, int j, double& ox, double& oy)
{
    const double t = pos * std::exp((-2.0 * j / 64) * std::log(8000000.0)), c = std::cos(t), s = std::sin(t);
    ox = x * c - y * s; oy = y * c + x * s;
}
static double rel(const std::vector<double>& a, const std::vector<double>& ref)
{
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - ref[i]) * (a[i] - ref[i]); d += ref[i] * ref[i]; }
    return std::sqrt(n / std::max(d, 1e-300));
}

// host attention: rows M at pos[i], keys 0..pos[i], heads H; out [M][H][256]
static std::vector<double> attend(const std::vector<std::vector<double>>& lat, const std::vector<std::vector<double>>& kr,
    const std::vector<__half>& qb, const std::vector<__nv_bfloat16>& uk, const std::vector<__nv_bfloat16>& uv,
    const std::vector<int>& pos, int H, double* score_sd)
{
    const int M = (int)pos.size();
    std::vector<double> out((size_t)M * H * 256);
    double s1 = 0, s2 = 0; size_t sn = 0;
    for (int i = 0; i < M; ++i)
        for (int h = 0; h < H; ++h)
        {
            const __half* q = &qb[((size_t)i * H + h) * 256];
            std::vector<double> qa(512, 0.0), qr(64);
            for (int k = 0; k < 192; ++k)
                for (int d = 0; d < 512; ++d) qa[d] += (double)h2f(q[k]) * b2f(uk[((size_t)h * 192 + k) * 512 + d]);
            for (int j = 0; j < 32; ++j) rope(h2f(q[192 + 2 * j]), h2f(q[193 + 2 * j]), pos[i], j, qr[2 * j], qr[2 * j + 1]);
            const int n = pos[i] + 1;
            std::vector<double> s(n);
            double mx = -1e300;
            for (int t = 0; t < n; ++t)
            {
                double v = 0;
                for (int d = 0; d < 512; ++d) v += qa[d] * lat[t][d];
                for (int d = 0; d < 64; ++d) v += qr[d] * kr[t][d];
                s[t] = v / 16;
                mx = std::max(mx, s[t]);
                s1 += s[t]; s2 += s[t] * s[t]; ++sn;
            }
            double den = 0;
            for (double& v : s) { v = std::exp(v - mx); den += v; }
            std::vector<double> ctx(512, 0.0);
            for (int t = 0; t < n; ++t)
                for (int d = 0; d < 512; ++d) ctx[d] += s[t] / den * lat[t][d];
            for (int e = 0; e < 256; ++e)
            {
                double v = 0;
                for (int d = 0; d < 512; ++d) v += (double)b2f(uv[((size_t)h * 256 + e) * 512 + d]) * ctx[d];
                out[((size_t)i * H + h) * 256 + e] = v;
            }
        }
    if (score_sd) *score_sd = std::sqrt(s2 / sn - (s1 / sn) * (s1 / sn));
    return out;
}
static std::vector<double> as_double(const std::vector<__half>& v) { std::vector<double> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = h2f(v[i]); return o; }

int main(int argc, char** argv)
{
    int T = 1000, M = 4;
    unsigned long long seed = 1;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (!strcmp(argv[i], "--tokens")) T = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--rows")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], nullptr, 10);
    }
    rng.seed(seed);
    const int N = T + M, CAP = (N + 63) / 64 * 64 + 64, L = 7;       // layer 7: a shared-index layer (no index cache)
    const int group = (0 - L % g6nr::NR + g6nr::NR) % g6nr::NR, H = g6nr::width(g6nr::HEADS, group);
    printf("kv4_test: row %s %d B, NR %d, heads %d, tokens %d, rows %d, seed %llu\n", g6kv::NAME, g6kv::ROW, g6nr::NR, H, T, M, seed);
    bool ok = true;
    auto need = [&](bool c, const char* what) { if (!c) { printf("FAIL %s\n", what); ok = false; } };

    // synthetic weights; the latent gets 16 outlier channels at 8x (layers 2-3 of the real model have such channels)
    std::vector<float> chs(512, 1.f);
    for (int k = 0; k < 16; ++k) chs[(k * 97 + 5) % 512] = 8.f;
    std::vector<__nv_bfloat16> uk((size_t)H * 192 * 512), uv((size_t)H * 256 * 512), kvn(512);
    for (auto& x : uk) x = __float2bfloat16(gauss(0.1f));
    for (auto& x : uv) x = __float2bfloat16(gauss(0.05f));
    for (auto& x : kvn) x = __float2bfloat16(1.f + gauss(0.1f));
    std::vector<__half> kva((size_t)N * 576);
    for (int t = 0; t < N; ++t)
        for (int d = 0; d < 576; ++d) kva[(size_t)t * 576 + d] = __float2half(gauss(d < 512 ? chs[d] : 1.f));
    __nv_bfloat16 *duk = up(uk), *duv = up(uv), *dkvn = up(kvn);
    G6AttnPack pack{};
    pack.uk = G6AttnTensor{duk, 3, {H, 192, 512}};
    pack.uv = G6AttnTensor{duv, 3, {H, 256, 512}};
    pack.kv_norm = G6AttnTensor{dkvn, 1, {512, 0, 0}};
    G6AttnBinding b{};
    CKI(g6_attn_bind_create(&b, &pack, 0, L, CAP, M, std::min(M, 64), nullptr));

    // 1. append every row (the reuse list pointer stands in for a shared layer's: no index keys are appended)
    std::vector<int> pos(N);
    for (int t = 0; t < N; ++t) pos[t] = t;
    int* dpos = up(pos);
    __half* dkva = up(kva);
    std::vector<int> ids((size_t)M * 2048, -1);
    for (int i = 0; i < M; ++i) for (int t = 0; t <= T + i; ++t) ids[(size_t)i * 2048 + t] = t;
    int* dids = up(ids);
    {
        G6Attn a = b.args;
        a.rows = N; a.context = N; a.positions = dpos; a.kv_a = dkva; a.reuse_indices = dids; a.reuse_rows = M;
        CKI(g6_attn_append(&a, nullptr));
        CK(cudaDeviceSynchronize());
    }
    const std::vector<uint8_t> cache = down((const uint8_t*)b.args.mla_cache, (size_t)N * g6kv::ROW);   // identity table
    std::vector<std::vector<double>> exact(N, std::vector<double>(512)), deq(N, std::vector<double>(512)),
        kr_exact(N, std::vector<double>(64)), kr_cache(N, std::vector<double>(64));
    double sig = 0, err = 0;
    int bad_rope = 0, bad_code = 0;
    for (int t = 0; t < N; ++t)
    {
        const __half* x = &kva[(size_t)t * 576];
        double ss = 0;
        for (int d = 0; d < 512; ++d) ss += (double)h2f(x[d]) * h2f(x[d]);
        const double inv = 1.0 / std::sqrt(ss / 512 + 1e-5);
        for (int d = 0; d < 512; ++d) exact[t][d] = h2f(x[d]) * inv * b2f(kvn[d]);
        const uint8_t* r = &cache[(size_t)t * g6kv::ROW];
        decode_row(r, deq[t].data());
        for (int d = 0; d < 512; ++d) { sig += exact[t][d] * exact[t][d]; err += (deq[t][d] - exact[t][d]) * (deq[t][d] - exact[t][d]); }
        const __nv_bfloat16* rk = reinterpret_cast<const __nv_bfloat16*>(r + g6kv::ROPE_OFF);
        for (int j = 0; j < 32; ++j)
        {
            rope(h2f(x[512 + 2 * j]), h2f(x[513 + 2 * j]), t, j, kr_exact[t][2 * j], kr_exact[t][2 * j + 1]);
            for (int u = 0; u < 2; ++u)
            {
                kr_cache[t][2 * j + u] = b2f(rk[2 * j + u]);
                bad_rope += std::fabs(kr_cache[t][2 * j + u] - kr_exact[t][2 * j + u]) > 1e-2 * std::fabs(kr_exact[t][2 * j + u]) + 1e-6;
            }
        }
#if G6_KV4
        for (int i = 0; i < 256; ++i) bad_code += (r[i] & 15) == 0 || (r[i] >> 4) == 0;   // codes are q + 8, q in -7..7
#endif
    }
    const double sqnr = 10 * std::log10(sig / err);
    printf("round trip: latent SQNR %.2f dB over %d rows; rope keys off by > 1%%: %d; codes outside -7..7: %d\n", sqnr, N, bad_rope, bad_code);
    need(sqnr >= (G6_KV4 ? 17.0 : 28.0), "round-trip SQNR");
    need(bad_rope == 0, "rope key");
    need(bad_code == 0, "INT4 codes");

    // 2. attention: q_b scaled so the scores have a standard deviation of about 3
    std::vector<__half> qb((size_t)M * H * 256);
    for (auto& x : qb) x = __float2half(gauss(1.5f));
    __half* dqb = up(qb);
    G6AttnStep s{};
    s.rows = M; s.context = N; s.q_b = dqb; s.kv_a = dkva + (size_t)T * 576; s.positions = dpos + T;
    s.reuse_indices = dids; s.reuse_rows = M;
    const std::vector<int> qpos(pos.begin() + T, pos.end());
    double sd = 0;
    const std::vector<double> ref_q = attend(deq, kr_cache, qb, uk, uv, qpos, H, &sd);
    const std::vector<double> ref_x = attend(exact, kr_exact, qb, uk, uv, qpos, H, nullptr);
    const double cost = rel(ref_q, ref_x);
    printf("attention (host, %d heads): score sd %.2f; the row format's cost |ref(cache) - ref(exact)| / |ref(exact)| = %.4f\n", H, sd, cost);
    need(cost < (G6_KV4 ? 0.25 : 0.05), "attention cost of the row format");

    CKI(g6_attn_bind_run(&b, &s, nullptr));
    CK(cudaDeviceSynchronize());
    need(down(b.args.error, 1)[0] == 0, "decode error word");
    const double kd = rel(as_double(down((const __half*)b.args.output, (size_t)M * H * 256)), ref_q);
    printf("decode   (g6_attn_bind_run):          kernel error %.2e, vs exact %.4f\n", kd,
           rel(as_double(down((const __half*)b.args.output, (size_t)M * H * 256)), ref_x));
    need(kd < 1e-2, "decode kernel error");
    const std::vector<uint8_t> after = down((const uint8_t*)b.args.mla_cache, (size_t)N * g6kv::ROW);
    need(after == cache, "decode re-append changed the cache");

    G6AttnPrefill* pf = nullptr;
    CKI(g6_attn_bind_prefill_create(&pf, &b, nullptr));
    CKI(g6_attn_bind_prefill_run(pf, &b, &s, nullptr));
    CK(cudaDeviceSynchronize());
    need(down(b.args.error, 1)[0] == 0, "prefill error word");
    const std::vector<double> po = as_double(down((const __half*)b.args.output, (size_t)M * H * 256));
    const double kp = rel(po, ref_q);
    printf("prefill  (g6_attn_bind_prefill_run):  kernel error %.2e, vs exact %.4f\n", kp, rel(po, ref_x));
    need(kp < 1e-2, "prefill kernel error");
    CKI(g6_attn_bind_prefill_release(pf));

    // row-owner prefill: its own 64-head UK / UV, the binding's cache
    std::vector<__nv_bfloat16> uk64((size_t)64 * 192 * 512), uv64((size_t)64 * 256 * 512);
    for (auto& x : uk64) x = __float2bfloat16(gauss(0.1f));
    for (auto& x : uv64) x = __float2bfloat16(gauss(0.05f));
    std::vector<__half> qb64((size_t)M * 64 * 256);
    for (auto& x : qb64) x = __float2half(gauss(1.5f));
    const G6AttnTensor tuk{up(uk64), 3, {64, 192, 512}}, tuv{up(uv64), 3, {64, 256, 512}};
    G6AttnQo* qo = nullptr;
    CKI(g6_attn_qo_create(&qo, &b, &tuk, &tuv, M, nullptr, nullptr));
    __half* qout = nullptr;
    CK(cudaMalloc(&qout, (size_t)M * 64 * 256 * 2));
    G6AttnQoStep qs{M, N, up(qb64), 64 * 256, dpos + T, dids, qout};
    CKI(g6_attn_qo_run(qo, &b, &qs, nullptr));
    CK(cudaDeviceSynchronize());
    need(down(b.args.error, 1)[0] == 0, "row-owner error word");
    const std::vector<double> qo_out = as_double(down(qout, (size_t)M * 64 * 256));
    const std::vector<double> qref_q = attend(deq, kr_cache, qb64, uk64, uv64, qpos, 64, nullptr);
    const std::vector<double> qref_x = attend(exact, kr_exact, qb64, uk64, uv64, qpos, 64, nullptr);
    const double kq = rel(qo_out, qref_q);
    printf("row-owner (g6_attn_qo_run, 64 heads): kernel error %.2e, vs exact %.4f\n", kq, rel(qo_out, qref_x));
    need(kq < 1e-2, "row-owner kernel error");
    CKI(g6_attn_qo_release(qo));
    CKI(g6_attn_bind_release(&b));
    printf("%s kv4_test (%s)\n", ok ? "PASS" : "FAIL", g6kv::NAME);
    return ok ? 0 : 1;
}
