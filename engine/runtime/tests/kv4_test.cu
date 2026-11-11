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
// Build and run: TEST=kv4_test GR8_NR=4 [EXTRA=-DG6_KV4=1] scripts/c1_tests.sh <bin_dir> (the K1H2 flags; attn.cu, bind.cu)
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../attn/bind.h"
#include "../../attn/prefill.h"
#include "../../attn/prefill-qo-bind.h"
#include "../../attn/ops.h"
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

#if !G6_KV4
static double e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}
#endif
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

#if GR8_KV4_TEST && G6_KV4
extern "C" void g6_kv4_test_set(int, int);
extern "C" unsigned long long g6_kv4_test_fwht_count(int);
extern "C" int g6_kv4_test_scores(G6Attn, int, int, const float*, const float*, float*, int, cudaStream_t);
extern "C" int g6_kv4_test_parts(G6Attn, int, int, const float*, float*, int, cudaStream_t);
extern "C" int g6_kv4_test_rotate(float*, int, cudaStream_t);
static double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    double aa=0, bb=0, ab=0;
    for(size_t i=0;i<a.size();++i){aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];}
    return aa==0&&bb==0 ? 1 : ab/std::sqrt(aa*bb);
}
static void max_errors(const std::vector<double>& a,const std::vector<double>& b) {
    double absolute=0,relative=0;
    for(size_t i=0;i<a.size();++i){double e=std::fabs(a[i]-b[i]);absolute=std::max(absolute,e);relative=std::max(relative,b[i]!=0?e/std::fabs(b[i]):e==0?0:INFINITY);}
    printf("  max_abs=%.9g max_pointwise_rel=%.9g (zero reference: nonzero error -> inf)\n",absolute,relative);
}
static std::vector<__nv_bfloat16> folded(const std::vector<__nv_bfloat16>& in) {
    auto out=in;
    for(size_t i=0;i<in.size();i+=512){
        double x[512];for(int d=0;d<512;++d)x[d]=b2f(in[i+d]);fwht(x);
        for(int d=0;d<512;++d)out[i+d]=__float2bfloat16_rn(float(x[d]/std::sqrt(512.0)));
    }
    return out;
}
// Decode the stored latent in the folded basis; no inverse Hadamard here.
static void decode_row_rot(const uint8_t* r, double* c_rot) {
    const __half* sc = reinterpret_cast<const __half*>(r + g6kv::SCALE_OFF);
    for (int d = 0; d < 512; ++d)
        c_rot[d] = double(int((r[d >> 1] >> ((d & 1) * 4)) & 15) - 8) * h2f(sc[d >> 5]);
}
static bool raw_tests(bool bench);
#endif

int main(int argc, char** argv)
{
    int T = 1000, M = 4; bool bench = false;
    unsigned long long seed = 1;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (!strcmp(argv[i], "--bench")) bench = atoi(argv[i + 1]) != 0;
        else if (!strcmp(argv[i], "--tokens")) T = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--rows")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], nullptr, 10);
    }
    rng.seed(seed);
#if GR8_KV4_TEST && G6_KV4
    // Exercise fold coexistence with both exact-path rotation knobs requested, before their first read.
    setenv("GR8_KV4_FUSE_ROT", "1", 1);
    setenv("GR8_KV4_QT", "1", 1);
    g6_kv4_test_set(0, 1);
#endif
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
    const auto decode_out = as_double(down((const __half*)b.args.output, (size_t)M * H * 256));
    const double kd = rel(decode_out, ref_q);
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
    CKI(g6_attn_bind_prefill_release(pf)); pf=nullptr;

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
    double sd64 = 0;
    const std::vector<double> qref_q = attend(deq, kr_cache, qb64, uk64, uv64, qpos, 64, &sd64);
    const std::vector<double> qref_x = attend(exact, kr_exact, qb64, uk64, uv64, qpos, 64, nullptr);
    const double kq = rel(qo_out, qref_q);
    printf("row-owner (g6_attn_qo_run, 64 heads): kernel error %.2e, vs exact %.4f\n", kq, rel(qo_out, qref_x));
    need(kq < 1e-2, "row-owner kernel error");
    CKI(g6_attn_qo_release(qo)); qo=nullptr;
    CKI(g6_attn_bind_release(&b));
#if GR8_KV4_TEST && G6_KV4
    std::vector<std::vector<double>> deq_rot(N, std::vector<double>(512));
    for (int t = 0; t < N; ++t)
        decode_row_rot(&cache[(size_t)t * g6kv::ROW], deq_rot[t].data());
    // Pre-registered synthetic BF16-fold drift envelope, separate from kernel error.
    constexpr double EPS_BF16 = 1.533e-3; // (2^-8) / sqrt(3 * (3 / (2 * ln 2)))
    const double drift_bound = 1.20 * EPS_BF16 * std::sqrt(1.0 + sd * sd);
    const double drift_bound64 = 1.20 * EPS_BF16 * std::sqrt(1.0 + sd64 * sd64);
    // Recreate every weight view at each toggle: the binder transposes UK and prefill owns converted copies.
    for(int fold : {0, 1, 0}) {
        auto ku=fold?folded(uk):uk, vu=fold?folded(uv):uv;
        auto k64=fold?folded(uk64):uk64, v64=fold?folded(uv64):uv64;
        const auto ref_pipe = fold ? attend(deq_rot, kr_cache, qb, ku, vu, qpos, H, nullptr) : ref_q;
        const auto qref_pipe = fold ? attend(deq_rot, kr_cache, qb64, k64, v64, qpos, 64, nullptr) : qref_q;
        if (fold) {
            printf("fold=1 host_bf16_drift decode/pf=%.8g (bound=%.8g) qo64=%.8g (bound=%.8g)\n",
                   rel(ref_pipe, ref_q), drift_bound, rel(qref_pipe, qref_q), drift_bound64);
            need(rel(ref_pipe, ref_q) < drift_bound && cosine(ref_pipe, ref_q) > 0.99998, "fold host BF16 drift bound");
            need(rel(qref_pipe, qref_q) < drift_bound64 && cosine(qref_pipe, qref_q) > 0.99998, "fold qo64 host BF16 drift bound");
        }
        CK(cudaMemcpy(duk,ku.data(),ku.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(duv,vu.data(),vu.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy((void*)tuk.data,k64.data(),k64.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy((void*)tuv.data,v64.data(),v64.size()*2,cudaMemcpyHostToDevice));
        g6_kv4_test_set(fold,1);
        CKI(g6_attn_bind_create(&b,&pack,0,L,CAP,M,std::min(M,64),nullptr));
        // Publish this binding's live RoPE table before append/decode can read PF_ROPE2.
        CKI(g6_attn_qo_create(&qo,&b,&tuk,&tuv,M,nullptr,nullptr));
        G6Attn a=b.args; a.rows=N;a.context=N;a.positions=dpos;a.kv_a=dkva;a.reuse_indices=dids;a.reuse_rows=M;
        CKI(g6_attn_append(&a,nullptr));
        need(down((const uint8_t*)b.args.mla_cache,(size_t)N*g6kv::ROW)==cache,"fold append cache byte equality");
        g6_kv4_test_fwht_count(1);
        std::vector<double> scalar;
        for(int tc : {0,1}) {
            g6_kv4_test_set(fold,tc);
            CKI(g6_attn_bind_run(&b,&s,nullptr));CK(cudaDeviceSynchronize());
            auto out=as_double(down((const __half*)b.args.output,(size_t)M*H*256));
            printf("fold=%d tc=%d decode same_host=%.8g same_cos=%.10g vs_unfold_dev=%.8g vs_unfold_host=%.8g\n",
                   fold,tc,rel(out,ref_pipe),cosine(out,ref_pipe),rel(out,decode_out),rel(out,ref_q));
            max_errors(out,decode_out);
            need(rel(out,ref_pipe)<1.5e-3&&cosine(out,ref_pipe)>0.999998,"fold same-pipeline decode tolerance");
            if(!fold&&tc==1)need(out==decode_out,"unfolded tc=1 exact replay");
            if(fold)need(rel(out,decode_out)<drift_bound&&cosine(out,decode_out)>0.99998&&rel(out,ref_q)<drift_bound,"fold decode BF16 drift bound");
            if(tc==0)scalar=out;
            else need(rel(out,scalar)<1.5e-3&&cosine(out,scalar)>0.999998,"TC end-to-end tolerance");
        }
        CKI(g6_attn_bind_prefill_create(&pf,&b,nullptr));
        CKI(g6_attn_bind_prefill_run(pf,&b,&s,nullptr));CK(cudaDeviceSynchronize());
        auto out=as_double(down((const __half*)b.args.output,(size_t)M*H*256));
        printf("fold=%d prefill same_host=%.8g same_cos=%.10g vs_unfold_dev=%.8g vs_unfold_host=%.8g\n",
               fold,rel(out,ref_pipe),cosine(out,ref_pipe),rel(out,po),rel(out,ref_q));
        max_errors(out,po);
        need(rel(out,ref_pipe)<1.5e-3&&cosine(out,ref_pipe)>0.999998,"fold same-pipeline prefill tolerance");
        if(!fold)need(out==po,"unfolded prefill exact replay");
        if(fold)need(rel(out,po)<drift_bound&&cosine(out,po)>0.99998&&rel(out,ref_q)<drift_bound,"fold prefill BF16 drift bound");
        CKI(g6_attn_bind_prefill_release(pf)); pf=nullptr;
        CKI(g6_attn_qo_run(qo,&b,&qs,nullptr));CK(cudaDeviceSynchronize());
        out=as_double(down(qout,(size_t)M*64*256));
        printf("fold=%d qo64 same_host=%.8g same_cos=%.10g vs_unfold_dev=%.8g vs_unfold_host=%.8g\n",
               fold,rel(out,qref_pipe),cosine(out,qref_pipe),rel(out,qo_out),rel(out,qref_q));
        max_errors(out,qo_out);
        need(rel(out,qref_pipe)<1.5e-3&&cosine(out,qref_pipe)>0.999998,"fold same-pipeline qo64 tolerance");
        if(!fold)need(out==qo_out,"unfolded qo64 exact replay");
        if(fold)need(rel(out,qo_out)<drift_bound64&&cosine(out,qo_out)>0.99998&&rel(out,qref_q)<drift_bound64,"fold qo64 BF16 drift bound");
        auto count=g6_kv4_test_fwht_count(0);
        printf("fold=%d fwht_launches=%llu\n",fold,count);
        need(fold?count==0:count>0,"fold launch skip");
        if(fold){
            need(count==0,"fold with FUSE_ROT=1 and QT=1 launches no standalone FWHT");
            printf("fold coexistence: FUSE_ROT=1 QT=1 standalone_fwht=%llu\n",count);
        }
        need(down(b.args.error,1)[0]==0,"fold error word");
        need(down((const uint8_t*)b.args.mla_cache,(size_t)N*g6kv::ROW)==cache,"fold final cache byte equality");
        CKI(g6_attn_qo_release(qo)); qo=nullptr;CKI(g6_attn_bind_release(&b));
    }
    g6_kv4_test_set(0,1);
#endif
    for(void* p : { (void*)duk,(void*)duv,(void*)dkvn,(void*)dpos,(void*)dkva,(void*)dids,(void*)dqb,
                   (void*)tuk.data,(void*)tuv.data,(void*)qs.q_b,(void*)qout }) CK(cudaFree(p));
#if GR8_KV4_TEST && G6_KV4
    ok=raw_tests(bench)&&ok;
#endif
    printf("%s kv4_test (%s)\n", ok ? "PASS" : "FAIL", g6kv::NAME);
    return ok ? 0 : 1;
}

#if GR8_KV4_TEST && G6_KV4
// Allocation-local red zones catch writers even when the address is still CUDA-accessible.
template<class T> struct Guarded {
    uint8_t* base=nullptr; T* p=nullptr; size_t n;
    explicit Guarded(size_t count):n(count){
        CK(cudaMalloc(&base,n*sizeof(T)+512));p=reinterpret_cast<T*>(base+256);
        CK(cudaMemset(base,0xA7,256));CK(cudaMemset(base+256+n*sizeof(T),0xA7,256));
    }
    explicit Guarded(const std::vector<T>& v):Guarded(v.size()){set(v);}
    ~Guarded(){CK(cudaFree(base));}
    Guarded(const Guarded&)=delete;Guarded& operator=(const Guarded&)=delete;
    void set(const std::vector<T>& v){if(v.size()!=n)exit(2);CK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    void poison(){CK(cudaMemsetAsync(p,0xFF,n*sizeof(T)));}
    std::vector<T> get() const{return down(p,n);}
    bool guards() const {
        auto a=down(base,256),b=down(base+256+n*sizeof(T),256);
        return std::all_of(a.begin(),a.end(),[](uint8_t v){return v==0xA7;})&&
               std::all_of(b.begin(),b.end(),[](uint8_t v){return v==0xA7;});
    }
    uint64_t hash() const {
        auto bytes=down(reinterpret_cast<const uint8_t*>(p),n*sizeof(T));uint64_t h=14695981039346656037ull;
        for(auto b:bytes){h^=b;h*=1099511628211ull;}return h;
    }
};
static bool raw_close(const std::vector<float>& out,const std::vector<float>& ref,bool score,bool print=true){
    std::vector<double> a,b;double maxabs=0;int bad=0;
    for(size_t i=0;i<out.size();++i){
        if(score&&ref[i]==-INFINITY){bad+=out[i]!=-INFINITY;continue;}
        if(!std::isfinite(out[i])||!std::isfinite(ref[i])){++bad;continue;}
        a.push_back(out[i]);b.push_back(ref[i]);maxabs=std::max(maxabs,std::fabs(double(out[i])-ref[i]));
    }
    const double rl=rel(a,b),cs=cosine(a,b);
    if(print)printf("%s rel=%.9g cos=%.12g max_abs=%.9g invalid=%d\n",score?"scores":"parts",rl,cs,maxabs,bad);
    return bad==0&&rl<1.5e-3&&cs>0.999995&&(score||maxabs<2e-3);
}
static std::vector<float> heads_slice(const std::vector<float>& in,int M,int H,int h0,int width){
    std::vector<float> out((size_t)M*16*width);
    for(int m=0;m<M;++m)std::copy_n(in.data()+((size_t)m*H+h0)*width,16*width,out.data()+(size_t)m*16*width);
    return out;
}
static bool raw_case(int H,int M,int mode,bool bench){
    constexpr int TOP=2048,PART=8,START=2,N=2048;
    bool ok=true;auto need=[&](bool c,const char* s){if(!c){printf("FAIL raw H=%d M=%d mode=%d %s\n",H,M,mode,s);ok=false;}};
    printf("raw H=%d M=%d start=%d mode=%d\n",H,M,START,mode);
    // mode 0 dense; 1 sparse with entire empty warp/part; 2 CP ownership + tiny probabilities;
    // mode 3 broken physical page; 4 context-bound error. None read a remote page in the context kernel.
    std::vector<uint8_t> cache((size_t)N*g6kv::ROW);
    for(int row=0;row<N;++row){
        auto r=cache.data()+(size_t)row*g6kv::ROW;
        for(int d=0;d<256;++d)r[d]=uint8_t((1+rng()%15)|((1+rng()%15)<<4));
        for(int g=0;g<16;++g){__half v=__float2half_rn(0.05f+0.01f*(rng()%20));memcpy(r+g6kv::SCALE_OFF+g*2,&v,2);}
        for(int d=0;d<64;++d){auto v=__float2bfloat16_rn(gauss(0.2f));memcpy(r+g6kv::ROPE_OFF+d*2,&v,2);}
    }
    std::vector<int> ids((START+M)*TOP,-1),pos(START+M,N-1),table(N/64);
    for(int i=0;i<N/64;++i)table[i]=mode==2&&i%4!=1?G6_CP_REMOTE:i;
    if(mode==3)table[0]=-1;
    for(int m=0;m<M;++m){
        for(int j=0;j<TOP;++j)ids[(START+m)*TOP+j]=j;
        if(mode==1){
            for(int j=0;j<TOP;++j)if(j<256||j%7==0)ids[(START+m)*TOP+j]=-1;
            pos[START+m]=N-64-m;
        }
    }
    std::vector<float> q((size_t)M*H*512),ropev((size_t)M*H*64),prob((size_t)M*H*TOP);
    for(auto& v:q)v=gauss(0.4f);for(auto& v:ropev)v=gauss(0.3f);
    for(int m=0;m<M;++m)for(int h=0;h<H;++h)for(int j=0;j<TOP;++j){
        int id=ids[(START+m)*TOP+j];bool live=id>=0&&id<=pos[START+m]&&(mode!=2||(id/64)%4==1);
        prob[((size_t)m*H+h)*TOP+j]=live?std::ldexp(float(1+rng()%31)/32768.f,mode==2?-60:0):0.f;
    }
    Guarded<uint8_t> dc(cache);Guarded<int> di(ids),dp(pos),dt(table);Guarded<unsigned> de(1);
    Guarded<float> dq(q),dr(ropev),pr(prob),s0((size_t)M*H*TOP),s1(s0.n),p0((size_t)M*H*PART*512),p1(p0.n);
    const uint64_t hashes[]={dc.hash(),di.hash(),dp.hash(),dt.hash(),dq.hash(),dr.hash(),pr.hash()};
    G6Attn a{};a.rows=START+M;a.heads=H;a.context=mode==4?N-64:N;a.page_size=64;a.page_count=N/64;a.table_size=N/64;
    a.positions=dp.p;a.block_table=dt.p;a.mla_cache=dc.p;a.indices=di.p;a.error=de.p;
    if(mode==2){a.cp_nr=4;a.cp_rank=1;}
    unsigned errors[2]{};
    for(int tc=0;tc<2;++tc){
        auto& so=tc?s1:s0;auto& po=tc?p1:p0;so.poison();po.poison();CK(cudaMemset(de.p,0,4));
        CKI(g6_kv4_test_scores(a,START,M,dq.p,dr.p,so.p,tc,nullptr));
        CKI(g6_kv4_test_parts(a,START,M,pr.p,po.p,tc,nullptr));CK(cudaDeviceSynchronize());errors[tc]=de.get()[0];
    }
    auto sr=s0.get(),st=s1.get(),cr=p0.get(),ct=p1.get();
    need(raw_close(st,sr,true),"score tolerance and masks");need(raw_close(ct,cr,false),"context tolerance");
    unsigned expected=mode==3?2:mode==4?1:0;need(errors[0]==expected&&errors[1]==expected,"error-bit equivalence");
    if(mode==1)for(int m=0;m<M;++m)for(int h=0;h<H;++h)for(int d=0;d<512;++d)
        need(ct[(((size_t)m*H+h)*PART)*512+d]==0.f,"empty part not zero");
    if(H==48){
        for(int h0=0;h0<48;h0+=16){
            Guarded<float> q16(heads_slice(q,M,H,h0,512)),r16(heads_slice(ropev,M,H,h0,64)),
                pr16(heads_slice(prob,M,H,h0,TOP)),so((size_t)M*16*TOP),po((size_t)M*16*PART*512);
            auto b=a;b.heads=16;so.poison();po.poison();
            CKI(g6_kv4_test_scores(b,START,M,q16.p,r16.p,so.p,1,nullptr));
            CKI(g6_kv4_test_parts(b,START,M,pr16.p,po.p,1,nullptr));CK(cudaDeviceSynchronize());
            auto ss=so.get(),pp=po.get(),want_s=heads_slice(st,M,H,h0,TOP),want_p=heads_slice(ct,M,H,h0,PART*512);
            need(memcmp(ss.data(),want_s.data(),ss.size()*4)==0&&memcmp(pp.data(),want_p.data(),pp.size()*4)==0,"ONECHAIN 0-ULP");
            need(q16.guards()&&r16.guards()&&pr16.guards()&&so.guards()&&po.guards(),"ONECHAIN red zones");
        }
        printf("ONECHAIN 48 vs 3x16 checked byte-for-byte\n");
    }
    if(bench){
        printf("SYNTHETIC kernel-only, NOT TTFT; keep_drop=DISABLED_pending_capture; calibration=unavailable\n");
        cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
        auto call=[&](int tc){CKI(g6_kv4_test_scores(a,START,M,dq.p,dr.p,s1.p,tc,nullptr));CKI(g6_kv4_test_parts(a,START,M,pr.p,p1.p,tc,nullptr));};
        // Poison outside each event pair; reference/candidate alternate and reverse on every other iteration.
        auto timed=[&](int tc){s1.poison();p1.poison();CK(cudaEventRecord(begin));call(tc);CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));return double(ms)*1000;};
        s1.poison();p1.poison();
        need(!raw_close(s1.get(),sr,true,false)&&!raw_close(p1.get(),cr,false,false),"compute-once negative control did not fail");
        printf("negative_control compute-once rejected\n");
        // A deliberately wrong-stream consumer must not accept an event from the idle stream as completion.
        struct Gate {std::mutex mutex;std::condition_variable cv;bool open=false;} gate;
        cudaStream_t wrong;CK(cudaStreamCreateWithFlags(&wrong,cudaStreamNonBlocking));
        s1.poison();CK(cudaStreamSynchronize(nullptr));
        CK(cudaLaunchHostFunc(wrong,[](void* ptr){auto& g=*static_cast<Gate*>(ptr);std::unique_lock<std::mutex> lock(g.mutex);g.cv.wait(lock,[&]{return g.open;});},&gate));
        CKI(g6_kv4_test_scores(a,START,M,dq.p,dr.p,s1.p,1,wrong));
        float* snapshot=nullptr;CK(cudaMallocHost(&snapshot,s1.n*sizeof(float)));
        CK(cudaMemcpyAsync(snapshot,s1.p,s1.n*sizeof(float),cudaMemcpyDeviceToHost,nullptr));
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
        const bool rejected=!raw_close(std::vector<float>(snapshot,snapshot+s1.n),st,true,false);
        {std::lock_guard<std::mutex> lock(gate.mutex);gate.open=true;}gate.cv.notify_one();
        CK(cudaStreamSynchronize(wrong));CK(cudaStreamDestroy(wrong));CK(cudaFreeHost(snapshot));
        need(rejected,"wrong-stream negative control did not fail");printf("negative_control wrong-stream %s\n",rejected?"rejected":"FAILED");
        for(int i=0;i<200;++i){s1.poison();p1.poison();call(i&1);}CK(cudaDeviceSynchronize());
        double sums[2][2]{},nulls[2]{};
        for(int i=0;i<1000;++i){
            int order=i&1;for(int j=0;j<2;++j){int tc=j^order;sums[order][tc]+=timed(tc);
                if(i%250==0||i==999){need(raw_close(s1.get(),tc?st:sr,true,false),"timed score sample");need(raw_close(p1.get(),tc?ct:cr,false,false),"timed parts sample");}}
        }
        for(int i=0;i<1000;++i)for(int j=0;j<2;++j)nulls[j]+=timed(0);
        for(int o=0;o<2;++o)printf("BENCH H=%d M=%d order=%s n=500 scalar_us=%.6f tc_us=%.6f speedup=%.6f\n",H,M,o?"BA":"AB",sums[o][0]/500,sums[o][1]/500,sums[o][0]/sums[o][1]);
        printf("NULL H=%d M=%d n=1000 off0_us=%.6f off1_us=%.6f delta_us=%.6f\n",H,M,nulls[0]/1000,nulls[1]/1000,(nulls[1]-nulls[0])/1000);
        // Time the actual rotation hook, retaining fresh input because the transform is in-place.
        Guarded<float> rot(q);double ft[2]{};
        auto rotate=[&](int fold){g6_kv4_test_set(fold,1);CK(cudaMemcpyAsync(rot.p,dq.p,q.size()*4,cudaMemcpyDeviceToDevice));CK(cudaEventRecord(begin));CKI(g6_kv4_test_rotate(rot.p,M*H,nullptr));CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));return double(ms)*1000;};
        for(int i=0;i<200;++i)rotate(i&1);
        for(int i=0;i<1000;++i)for(int j=0;j<2;++j){int fold=j^(i&1);ft[fold]+=rotate(fold);}
        rotate(0);
        auto rotated=rot.get();std::vector<float> rotated_ref(q.size());
        for(size_t row=0;row<q.size();row+=512){double x[512];for(int d=0;d<512;++d)x[d]=q[row+d];fwht(x);for(int d=0;d<512;++d)rotated_ref[row+d]=float(x[d]/std::sqrt(512.0));}
        need(raw_close(rotated,rotated_ref,false,false),"rotation reference");
        rotate(1);need(rot.hash()==dq.hash(),"fold skip changed input");need(rot.guards(),"rotation red zones");
        printf("FWHT H=%d M=%d n=1000 rotate_us=%.6f skip_us=%.6f\n",H,M,ft[0]/1000,ft[1]/1000);
        g6_kv4_test_set(0,1);CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
    }
    need(dc.guards()&&di.guards()&&dp.guards()&&dt.guards()&&de.guards()&&dq.guards()&&dr.guards()&&pr.guards()&&s0.guards()&&s1.guards()&&p0.guards()&&p1.guards(),"allocation red zones");
    need(hashes[0]==dc.hash()&&hashes[1]==di.hash()&&hashes[2]==dp.hash()&&hashes[3]==dt.hash()&&hashes[4]==dq.hash()&&hashes[5]==dr.hash()&&hashes[6]==pr.hash(),"immutable input hashes");
    printf("raw guards/input hashes %s\n",ok?"PASS":"FAIL");return ok;
}
static bool raw_tests(bool bench){
    bool ok=true;
    for(int H:{16,11,48})for(int M:{1,3,8})for(int mode:{0,1,2})ok=raw_case(H,M,mode,false)&&ok;
    for(int mode:{3,4})ok=raw_case(11,3,mode,false)&&ok;
    if(bench)for(int H:{16,48})for(int M:{1,4})ok=raw_case(H,M,0,true)&&ok;
    return ok;
}
#endif
