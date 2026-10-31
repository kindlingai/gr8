// DS41 attention B (indexer) perf-path test: the multi-CTA top-k (launch_topk_mc) and the MXFP4 tensor-core dense logits (launch_logits_dense_mma)
// against the reference kernels they replace (k_topk, k_logits dp4a), then timing of the dense stage with each.
// Needs a GPU that runs sm_121a (a Spark). Build on the Spark:
//   nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I. -Iengine engine/attn/tests/ds41_idx_perf_test.cu -o /tmp/ds41_idx_perf_test
//   /tmp/ds41_idx_perf_test [probe] [topk] [logits] [timing] [small]    (no args = all; small = N <= 131072, for compute-sanitizer)
// Exit code = number of failed checks. The timing section never fails.
#include "../ds41_idx.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ds41i;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static int g_fail = 0;
static void verdict(const char* name, bool ok, const char* extra = "")
{
    printf("%-58s %s %s\n", name, ok ? "PASS" : "FAIL", extra);
    if (!ok) g_fail++;
    fflush(stdout);
}
static uint16_t bfr(float f)   // fp32 -> bf16 RNE (host)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}
static float bff(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
static const float E2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static double e2m1v(int c) { return (c & 8 ? -1.0 : 1.0) * E2M1[c & 7]; }

// ------------------------------------------------------------------------------------------------ (a) MMA layout probe
__global__ void k_probe(const uint4* a, const uint32_t* b, const uint32_t* sfa, const uint32_t* sfb, float* d)
{
    const int l = threadIdx.x;
    const float c[4] = {0.f, 0.f, 0.f, 0.f};
    float o[4];
    mma_mxf4(o, a[l], b[2 * l], b[2 * l + 1], c, sfa[l], sfb[l]);
    for (int i = 0; i < 4; i++) d[l * 4 + i] = o[i];
}
static void test_probe()
{
    std::mt19937 rng(123);
    for (int trial = 0; trial < 3; trial++)
    {
        uint8_t A[16][32], B[8][32];   // bytes: dims 2j (low nibble), 2j+1 (high)
        int ea[16][2], eb[8][2];
        for (auto& r : A) for (auto& x : r) x = (uint8_t)rng();
        for (auto& r : B) for (auto& x : r) x = (uint8_t)rng();
        for (auto& r : ea) for (auto& x : r) x = 120 + rng() % 11;
        for (auto& r : eb) for (auto& x : r) x = 120 + rng() % 11;
        double ref[16][8];
        auto nib = [](const uint8_t* row, int k) { return (row[k >> 1] >> (4 * (k & 1))) & 15; };
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 8; n++)
            {
                double s = 0;
                for (int k = 0; k < 64; k++)
                    s += e2m1v(nib(A[m], k)) * e2m1v(nib(B[n], k)) * std::ldexp(1.0, ea[m][k / 32] - 127) * std::ldexp(1.0, eb[n][k / 32] - 127);
                ref[m][n] = s;
            }
        std::vector<uint4> ha(32);
        std::vector<uint32_t> hb(64), hsa(32), hsb(32);
        auto w32 = [](const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; };
        for (int l = 0; l < 32; l++)
        {
            const int m = l >> 2, t = l & 3;
            ha[l] = make_uint4(w32(&A[m][4 * t]), w32(&A[m + 8][4 * t]), w32(&A[m][16 + 4 * t]), w32(&A[m + 8][16 + 4 * t]));
            hb[2 * l] = w32(&B[m][4 * t]);
            hb[2 * l + 1] = w32(&B[m][16 + 4 * t]);
            const int hs = m + 8 * (l & 1);
            hsa[l] = (uint32_t)ea[hs][0] | ((uint32_t)ea[hs][1] << 8);
            hsb[l] = (uint32_t)eb[m][0] | ((uint32_t)eb[m][1] << 8);
        }
        uint4* da; uint32_t *db, *dsa, *dsb; float* dd;
        CK(cudaMalloc(&da, 512)); CK(cudaMalloc(&db, 256)); CK(cudaMalloc(&dsa, 128)); CK(cudaMalloc(&dsb, 128)); CK(cudaMalloc(&dd, 512));
        CK(cudaMemcpy(da, ha.data(), 512, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, hb.data(), 256, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsa, hsa.data(), 128, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsb, hsb.data(), 128, cudaMemcpyHostToDevice));
        k_probe<<<1, 32>>>(da, db, dsa, dsb, dd);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> hd(128);
        CK(cudaMemcpy(hd.data(), dd, 512, cudaMemcpyDeviceToHost));
        int bad = 0;
        double maxrel = 0;
        for (int l = 0; l < 32; l++)
            for (int i = 0; i < 4; i++)
            {
                const int m = (l >> 2) + 8 * (i >> 1), n = 2 * (l & 3) + (i & 1);
                const double want = ref[m][n], got = hd[l * 4 + i];
                const double rel = std::fabs(got - want) / std::max(1e-30, std::fabs(want));
                maxrel = std::max(maxrel, rel);
                if (rel > 1e-5 && std::fabs(got - want) > 1e-6) { if (bad < 8) printf("   probe diff lane %d d%d (row %d key %d): got %.9g want %.9g\n", l, i, m, n, got, want); bad++; }
            }
        char ex[96];
        snprintf(ex, sizeof ex, "%d / 128 mismatch, max rel %.2e", bad, maxrel);
        char nm[64];
        snprintf(nm, sizeof nm, "mma m16n8k64 mxf4 layout probe, trial %d", trial);
        verdict(nm, bad == 0, ex);
        cudaFree(da); cudaFree(db); cudaFree(dsa); cudaFree(dsb); cudaFree(dd);
    }
}

// ------------------------------------------------------------------------------------------------ (b) top-k
template <int K>
static std::vector<int> topk_host(const std::vector<uint16_t>& row, int n)
{
    auto key = [](uint16_t b) -> uint32_t {
        if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;
        return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
    };
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min(K, n);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key(row[a]) > key(row[b]); });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    idx.resize(K, -1);
    return idx;
}
enum Gen { G_RAND, G_FEW, G_EQUAL, G_SPECIAL, G_COARSE };
static const char* gen_name[] = {"random", "8 distinct", "all equal", "nan/inf/zero mix", "coarse (ties at cut)"};
static uint16_t gen_val(Gen g, std::mt19937& rng)
{
    static std::normal_distribution<float> nd(0.f, 1.f);
    switch (g)
    {
    case G_RAND: return bfr(nd(rng));
    case G_FEW: return bfr((float)(int)(rng() % 8) - 3.5f);
    case G_EQUAL: return bfr(0.25f);
    case G_COARSE: return bfr((float)(int)(rng() % 400) * 0.01f);
    default:
    {
        const uint32_t c = rng() % 12;
        switch (c)
        {
        case 0: return 0x7FC0;                  // NaN
        case 1: return 0xFFC1;                  // -NaN
        case 2: return 0x7F80;                  // +inf
        case 3: return 0xFF80;                  // -inf
        case 4: return 0x0000;                  // +0
        case 5: return 0x8000;                  // -0
        default: return bfr(nd(rng));
        }
    }
    }
}
template <int K>
static void test_topk_case(Gen g, int N, const std::vector<int>& ends, int ctas, const char* tag, uint32_t seed, bool verbose_pass)
{
    const int R = (int)ends.size();
    const long ldv = (N + 127) / 128 * 128;
    std::mt19937 rng(seed);
    std::vector<uint16_t> h((size_t)R * ldv, 0xFF80);   // -inf past the data
    for (int r = 0; r < R; r++)
        for (int i = 0; i < N; i++) h[(size_t)r * ldv + i] = i < ends[r] ? gen_val(g, rng) : 0xFF80;
    bf16* dv; int *dend, *o1, *o2; int32_t* ws;
    CK(cudaMalloc(&dv, h.size() * 2));
    CK(cudaMemcpy(dv, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dend, R * 4));
    CK(cudaMemcpy(dend, ends.data(), R * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&o1, (size_t)R * K * 4)); CK(cudaMalloc(&o2, (size_t)R * K * 4));
    CK(cudaMalloc(&ws, topk_mc_ws_ints(R, ldv) * 4));
    CK(cudaMemset(o1, 0x55, (size_t)R * K * 4)); CK(cudaMemset(o2, 0x55, (size_t)R * K * 4));
    launch_topk<K>(dv, ldv, dend, N, o1, K, R, 0);
    launch_topk_mc<K>(dv, ldv, dend, N, o2, K, R, ws, ctas, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int> g1((size_t)R * K), g2((size_t)R * K);
    CK(cudaMemcpy(g1.data(), o1, g1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g2.data(), o2, g2.size() * 4, cudaMemcpyDeviceToHost));
    int bad1 = 0, bad2 = 0, bad12 = 0, firstr = -1;
    for (int r = 0; r < R; r++)
    {
        std::vector<uint16_t> row(h.begin() + (size_t)r * ldv, h.begin() + (size_t)r * ldv + N);
        const std::vector<int> ref = topk_host<K>(row, std::max(0, std::min(ends[r], N)));
        for (int i = 0; i < K; i++)
        {
            bad1 += g1[(size_t)r * K + i] != ref[i];
            bad2 += g2[(size_t)r * K + i] != ref[i];
            bad12 += g1[(size_t)r * K + i] != g2[(size_t)r * K + i];
            if (g2[(size_t)r * K + i] != ref[i] && firstr < 0) { firstr = r; printf("   first mismatch row %d slot %d: mc %d host %d k_topk %d\n", r, i, g2[(size_t)r * K + i], ref[i], g1[(size_t)r * K + i]); }
        }
    }
    char nm[160], ex[96];
    snprintf(nm, sizeof nm, "topk K=%d %s N=%d R=%d ends[0]=%d ctas=%d", K, gen_name[g], N, R, ends[0], ctas);
    snprintf(ex, sizeof ex, "k_topk-vs-host %d, mc-vs-host %d, mc-vs-k_topk %d mismatch %s", bad1, bad2, bad12, tag);
    (void)verbose_pass;
    verdict(nm, !(bad1 || bad2 || bad12), ex);
    cudaFree(dv); cudaFree(dend); cudaFree(o1); cudaFree(o2); cudaFree(ws);
}
template <int K> static void test_topk(bool small)
{
    const std::vector<int> Ns = small ? std::vector<int>{8192, 24576, 100003, 131072} : std::vector<int>{8192, 24576, 100003, 131072, 524288, 1048575, 1048576};
    for (int N : Ns)
        for (int gi = 0; gi < 5; gi++)
        {
            const Gen g = (Gen)gi;
            // R = 6 with varied per-row ends (n > K, == K + 1, == K, partial, tiny) and R = 1 full
            std::vector<int> e6 = {N, N - 1, std::min(N, K + 1), std::min(N, K), N / 2, 3};
            test_topk_case<K>(g, N, e6, 288, "(R=6 varied ends)", 1000 + N + gi, false);
            test_topk_case<K>(g, N, {N}, 288, "(R=1)", 2000 + N + gi, false);
        }
    // CTA-count sweep on one shape (different chunking must give the same result)
    for (int ctas : {1, 7, 96, 192, 576, 2000})
        test_topk_case<K>(G_COARSE, small ? 100003 : 524288, {small ? 100003 : 524288, 100000, 99999, 70000, 65536, 30000}, ctas, "(cta sweep)", 77 + ctas, false);
}

// ------------------------------------------------------------------------------------------------ (c) logits + (d) timing setup
struct Fx
{
    int NMAX, RMAX;
    uint8_t* ica;
    int* page0;
    bf16 *dq_unused;
    uint8_t *qpack, *qexp;
    bf16* w;
    void* scratch;
    int* klen;
    int *out1, *out2;
    Scratch sc;
    IcaMap km;
};
// host fp4 quantizer (E8M0 scale per 32 dims, e2m1 nearest): x[32] -> 16 code bytes + scale byte
static uint8_t quant_group(const float* x, uint8_t* codes16)
{
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 6.f * std::ldexp(1.f, -126));
    int e = (int)std::ceil(std::log2(amax * (1.f / 6.f)));
    e = std::max(-127, std::min(127, e));
    const float inv = std::ldexp(1.f, -e);
    uint8_t c[32];
    for (int i = 0; i < 32; i++)
    {
        const float v = std::fabs(x[i] * inv);
        int best = 0;
        float bd = 1e30f;
        for (int k = 0; k < 8; k++)
        {
            const float d = std::fabs(v - E2M1[k]);
            if (d < bd) { bd = d; best = k; }
        }
        c[i] = (uint8_t)(best | (x[i] < 0.f ? 8 : 0));
    }
    for (int i = 0; i < 16; i++) codes16[i] = (uint8_t)(c[2 * i] | (c[2 * i + 1] << 4));
    return (uint8_t)(e + 127);
}
// realistic row: normal dims, a row gain exp(N(0,.7)) and a per-group gain exp(N(0,.3)), quantized like the engine's index rows
static void real_row(std::mt19937& rng, uint8_t* codes64, uint8_t* sc4)
{
    std::normal_distribution<float> nd(0.f, 1.f);
    const float rg = std::exp(0.7f * nd(rng));
    for (int g = 0; g < 4; g++)
    {
        float x[32];
        const float gg = rg * std::exp(0.3f * nd(rng));
        for (int i = 0; i < 32; i++) x[i] = nd(rng) * gg;
        sc4[g] = quant_group(x, codes64 + g * 16);
    }
}
static Fx make_fx(int NMAX, int RMAX, uint32_t seed, bool realistic = false)
{
    Fx f{};
    f.NMAX = NMAX; f.RMAX = RMAX;
    std::mt19937 rng(seed);
    std::vector<uint8_t> flat((size_t)NMAX * 68);
    if (realistic)
        for (int j = 0; j < NMAX; j++) real_row(rng, &flat[(size_t)j * 64], &flat[(size_t)NMAX * 64 + (size_t)j * 4]);
    else
    {
        for (size_t i = 0; i < (size_t)NMAX * 64; i++) flat[i] = (uint8_t)rng();
        for (size_t i = 0; i < (size_t)NMAX * 4; i++) flat[(size_t)NMAX * 64 + i] = (uint8_t)(120 + rng() % 11);
    }
    CK(cudaMalloc(&f.ica, flat.size()));
    CK(cudaMemcpy(f.ica, flat.data(), flat.size(), cudaMemcpyHostToDevice));
    int zero = 0;
    CK(cudaMalloc(&f.page0, 4));
    CK(cudaMemcpy(f.page0, &zero, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&f.scratch, Scratch::bytes(RMAX, NMAX)));
    f.sc = Scratch::carve(f.scratch, RMAX, NMAX);
    std::vector<uint8_t> qp((size_t)RMAX * 32 * 64), qe((size_t)RMAX * 32 * 4);
    if (realistic)
        for (int r = 0; r < RMAX * 32; r++) real_row(rng, &qp[(size_t)r * 64], &qe[(size_t)r * 4]);
    else
    {
        for (auto& x : qp) x = (uint8_t)rng();
        for (auto& x : qe) x = (uint8_t)(120 + rng() % 11);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hw((size_t)RMAX * 32);
    for (auto& x : hw) x = bfr(nd(rng) * 0.02f);
    CK(cudaMemcpy(f.sc.q_pack, qp.data(), qp.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(f.sc.q_exp, qe.data(), qe.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(f.sc.w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&f.klen, RMAX * 4));
    CK(cudaMalloc(&f.out1, (size_t)RMAX * 512 * 4));
    CK(cudaMalloc(&f.out2, (size_t)RMAX * 512 * 4));
    f.km = ica_map(f.ica, f.page0, NMAX, (size_t)NMAX * 68);
    return f;
}
static LogitsP mk_logits(const Fx& f, int N, int R)
{
    LogitsP l{};
    l.q_pack = f.sc.q_pack; l.q_exp = f.sc.q_exp; l.w = f.sc.w; l.k = f.km; l.klen = f.klen; l.p = f.sc.p; l.ldp = f.sc.ldp; l.ncols = N; l.R = R;
    return l;
}
// host logits of (row r, token j), float math in the dp4a kernel's order (reference for the dp4a kernel), and the exact double value
static void host_logit(const std::vector<uint8_t>& qp, const std::vector<uint8_t>& qe, const std::vector<uint16_t>& hw, const std::vector<uint8_t>& flat,
                       int NMAX, int r, int j, float* f_order, double* exact)
{
    const uint8_t* kc = &flat[(size_t)j * 64];
    const uint8_t* ks = &flat[(size_t)NMAX * 64 + (size_t)j * 4];
    float s = 0.f;
    double sx = 0;
    for (int h = 0; h < 32; h++)
    {
        const uint8_t* qc = &qp[((size_t)r * 32 + h) * 64];
        const uint8_t* qs = &qe[((size_t)r * 32 + h) * 4];
        float dot = 0.f;
        double dx = 0;
        for (int g = 0; g < 4; g++)
        {
            long acc = 0;   // codes doubled twice -> exact ints
            double ad = 0;
            for (int d = 0; d < 32; d++)
            {
                const int dim = g * 32 + d;
                const int cq = (qc[dim >> 1] >> (4 * (dim & 1))) & 15, ck = (kc[dim >> 1] >> (4 * (dim & 1))) & 15;
                acc += (long)(2 * e2m1v(cq)) * (long)(2 * e2m1v(ck));
                ad += e2m1v(cq) * e2m1v(ck);
            }
            const float fq = std::ldexp(1.f, qs[g] - 127), fk = std::ldexp(1.f, ks[g] - 127) * 0.25f;
            dot = dot + (float)acc * (fq * fk);
            dx += ad * std::ldexp(1.0, qs[g] - 127) * std::ldexp(1.0, ks[g] - 127);
        }
        const float db = bff(bfr(dot));
        s += bff(bfr((db > 0.f ? db : 0.f) * bff(hw[(size_t)r * 32 + h])));
        const double dxb = (double)bff(bfr((float)dx));
        sx += (double)bff(bfr((float)((dxb > 0 ? dxb : 0.0) * (double)bff(hw[(size_t)r * 32 + h]))));
    }
    *f_order = bff(bfr(s));
    *exact = (double)bff(bfr((float)sx));
}
static void test_logits(bool small, bool realistic, bool exact)
{
    const int NMAX = small ? 131072 : 1048576, RMAX = 6;
    printf("\n-- logits data: %s, mma variant: %s\n", realistic ? "REALISTIC (quantized normals, row/group gains)" : "WIDE (random codes, scale bytes 120..130)", exact ? "EXACT (must be bit-identical to dp4a)" : "FAST (rounding may differ on wide-scale data)");
    Fx f = make_fx(NMAX, RMAX, 99, realistic);
    std::vector<uint8_t> qp((size_t)RMAX * 2048), qe((size_t)RMAX * 128), flat((size_t)NMAX * 68);
    std::vector<uint16_t> hw((size_t)RMAX * 32);
    CK(cudaMemcpy(qp.data(), f.sc.q_pack, qp.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qe.data(), f.sc.q_exp, qe.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hw.data(), f.sc.w, hw.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(flat.data(), f.ica, flat.size(), cudaMemcpyDeviceToHost));
    std::vector<int> Ns = {8192, 100003, 131072};
    if (!small) { Ns.push_back(524288); Ns.push_back(1048576); }
    for (int N : Ns)
        for (int R : {1, 6})
        {
            std::vector<int> hk(R);
            for (int r = 0; r < R; r++) hk[r] = std::max(1, N - r * 997 - (r == 3 ? 5 : 0));   // varied visible lengths
            CK(cudaMemcpy(f.klen, hk.data(), R * 4, cudaMemcpyHostToDevice));
            const LogitsP l = mk_logits(f, N, R);
            const size_t cnt = (size_t)R * f.sc.ldp;
            CK(cudaMemset(f.sc.p, 0x77, cnt * 2));
            launch_logits_dense(l, 0);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<uint16_t> p1(cnt), p2(cnt);
            CK(cudaMemcpy(p1.data(), f.sc.p, cnt * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemset(f.sc.p, 0x77, cnt * 2));
            launch_logits_dense_mma(l, 0, exact);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(p2.data(), f.sc.p, cnt * 2, cudaMemcpyDeviceToHost));
            // mma vs dp4a over all columns < N
            long mism = 0, ulp1 = 0, ulpbig = 0, badmask = 0;
            for (int r = 0; r < R; r++)
                for (int c = 0; c < N; c++)
                {
                    const uint16_t a = p1[(size_t)r * f.sc.ldp + c], b = p2[(size_t)r * f.sc.ldp + c];
                    if (c >= hk[r] && b != 0xFF80) badmask++;
                    if (c >= hk[r] && a != 0xFF80) badmask++;
                    if (a == b) continue;
                    mism++;
                    const int ia = (a & 0x8000) ? -(int)(a & 0x7FFF) : (int)a, ib = (b & 0x8000) ? -(int)(b & 0x7FFF) : (int)b;
                    if (std::abs(ia - ib) == 1) ulp1++;
                    else ulpbig++;
                }
            // top-512 set identity on the two logit buffers (the single-CTA kernel on both)
            CK(cudaMemcpy(f.sc.p, p1.data(), cnt * 2, cudaMemcpyHostToDevice));
            launch_topk<TOPK>(f.sc.p, f.sc.ldp, f.klen, N, f.out1, 512, R, 0);
            CK(cudaMemcpy(f.sc.p, p2.data(), cnt * 2, cudaMemcpyHostToDevice));
            launch_topk<TOPK>(f.sc.p, f.sc.ldp, f.klen, N, f.out2, 512, R, 0);
            CK(cudaDeviceSynchronize());
            std::vector<int> t1((size_t)R * 512), t2((size_t)R * 512);
            CK(cudaMemcpy(t1.data(), f.out1, t1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(t2.data(), f.out2, t2.size() * 4, cudaMemcpyDeviceToHost));
            long tdiff = 0;
            for (size_t i = 0; i < t1.size(); i++) tdiff += t1[i] != t2[i];
            // dp4a kernel vs host float-order reference, mma vs host exact value, on a sample of columns
            std::mt19937 rng(5 + N + R);
            long s_dp = 0, s_mma_f = 0, s_mma_x = 0, ns = 0;
            for (int r = 0; r < R; r++)
                for (int k = 0; k < (small ? 1500 : 800); k++)
                {
                    const int c = k < 8 ? (hk[r] - 1 - k) : (int)(rng() % (uint32_t)hk[r]);
                    if (c < 0 || c >= hk[r]) continue;
                    float hf; double hx;
                    host_logit(qp, qe, hw, flat, NMAX, r, c, &hf, &hx);
                    const float a = bff(p1[(size_t)r * f.sc.ldp + c]), b = bff(p2[(size_t)r * f.sc.ldp + c]);
                    s_dp += a != hf;
                    s_mma_f += b != hf;
                    s_mma_x += (double)b != hx;
                    ns++;
                }
            char nm[96], ex[220];
            snprintf(nm, sizeof nm, "logits mma-%s vs dp4a %s N=%d R=%d", exact ? "exact" : "fast", realistic ? "real" : "wide", N, R);
            snprintf(ex, sizeof ex, "%ld/%ld bit-diff (1ulp %ld, >1ulp %ld), top512 idx diff %ld, bad mask %ld | host sample %ld: dp4a!=host %ld, mma!=hostfp32order %ld, mma!=hostexact %ld",
                     mism, (long)R * N, ulp1, ulpbig, tdiff, badmask, ns, s_dp, s_mma_f, s_mma_x);
            verdict(nm, ((realistic || exact) ? mism == 0 : true) && badmask == 0 && s_dp == 0 && (!exact || (s_mma_f == 0 && tdiff == 0)), ex);
            if (mism)
            {
                printf("   NOTE: %ld mma outputs differ from dp4a (selection-relevant only through ties; top512 idx diff above); first ones vs host:\n", mism);
                int shown = 0;
                double e_dp = 0, e_mma = 0;
                for (int r = 0; r < R && shown < 12; r++)
                    for (int c = 0; c < N && shown < 12; c++)
                    {
                        const uint16_t a = p1[(size_t)r * f.sc.ldp + c], b = p2[(size_t)r * f.sc.ldp + c];
                        if (a == b || c >= hk[r]) continue;
                        float hf; double hx;
                        host_logit(qp, qe, hw, flat, NMAX, r, c, &hf, &hx);
                        printf("     r%d c%d: dp4a %.9g  mma %.9g  host(fp32 order) %.9g  host(exact dot) %.9g\n", r, c, bff(a), bff(b), hf, hx);
                        e_dp += std::fabs(bff(a) - hx); e_mma += std::fabs(bff(b) - hx);
                        shown++;
                    }
                printf("     sum |err vs exact| over these: dp4a %.4g, mma %.4g\n", e_dp, e_mma);
            }
        }
}

// ------------------------------------------------------------------------------------------------ (d) timing
static cudaEvent_t e0, e1;
template <class F> static double span_us(F fn, int rep = 10)
{
    for (int i = 0; i < 3; i++) fn();
    CK(cudaEventRecord(e0));
    for (int i = 0; i < rep; i++) fn();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / rep;
}
static void timing(bool small)
{
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int NMAX = small ? 131072 : 1048576, RMAX = 6;
    Fx f = make_fx(NMAX, RMAX, 11);
    std::vector<int> Ns = {8192, 131072};
    if (!small) { Ns.push_back(262144); Ns.push_back(524288); Ns.push_back(1048576); }
    printf("\ndense index stage (logits + top-512; idx_q is a constant ~few us on top), us, cudaEvent mean of 10 launches\n");
    printf("%-8s %-2s | %-9s %-9s %-9s | %-9s %-9s | %-9s %-10s %-10s | %-8s %-8s\n", "N", "R", "lg_dp4a", "lg_mmaEx", "lg_mmaFast", "tk_1cta", "tk_mc288", "OLD", "NEW(ex)", "NEW(fast)", "spdup ex", "spdup fast");
    for (int N : Ns)
        for (int R : {1, 6})
        {
            std::vector<int> hk(R, N);
            CK(cudaMemcpy(f.klen, hk.data(), R * 4, cudaMemcpyHostToDevice));
            const LogitsP l = mk_logits(f, N, R);
            const double lg_d = span_us([&] { launch_logits_dense(l, 0); });
            const double lg_x = span_us([&] { launch_logits_dense_mma(l, 0, true); });
            const double lg_f = span_us([&] { launch_logits_dense_mma(l, 0, false); });
            launch_logits_dense_mma(l, 0, true);   // realistic values for the top-k
            const double t1 = span_us([&] { launch_topk<TOPK>(f.sc.p, f.sc.ldp, f.klen, N, f.out1, 512, R, 0); });
            const double tm = span_us([&] { launch_topk_mc<TOPK>(f.sc.p, f.sc.ldp, f.klen, N, f.out2, 512, R, f.sc.tkw, MC_CTAS, 0); });
            CK(cudaGetLastError());
            printf("%-8d %-2d | %-9.1f %-9.1f %-9.1f | %-9.1f %-9.1f | %-9.1f %-10.1f %-10.1f | %.2fx    %.2fx\n", N, R, lg_d, lg_x, lg_f, t1, tm, lg_d + t1, lg_x + tm, lg_f + tm,
                   (lg_d + t1) / (lg_x + tm), (lg_d + t1) / (lg_f + tm));
            fflush(stdout);
        }
    printf("\nmulti-CTA top-512 CTA-count sweep (target total CTAs), us\n%-8s %-2s |", "N", "R");
    const int sweep[] = {48, 96, 144, 192, 288, 384, 576, 1152};
    for (int c : sweep) printf(" %-8d", c);
    printf("\n");
    for (int N : Ns)
        for (int R : {1, 6})
        {
            std::vector<int> hk(R, N);
            CK(cudaMemcpy(f.klen, hk.data(), R * 4, cudaMemcpyHostToDevice));
            launch_logits_dense_mma(mk_logits(f, N, R), 0);
            printf("%-8d %-2d |", N, R);
            for (int c : sweep) printf(" %-8.1f", span_us([&] { launch_topk_mc<TOPK>(f.sc.p, f.sc.ldp, f.klen, N, f.out2, 512, R, f.sc.tkw, c, 0); }));
            printf("\n");
            fflush(stdout);
        }
    printf("\nblock-score top-2048 (candidate stage, layer 20) 1-CTA vs multi-CTA, us\n%-8s %-2s | %-9s %-9s\n", "N", "R", "tk2048_1", "tk2048_mc");
    for (int N : Ns)
        for (int R : {1, 6})
        {
            std::vector<int> hk(R, N);
            CK(cudaMemcpy(f.klen, hk.data(), R * 4, cudaMemcpyHostToDevice));
            const int nb = (int)f.sc.ldsc, nbv = (N + 7) / 8;
            launch_logits_dense_mma(mk_logits(f, N, R), 0);
            launch_block_scores(f.sc.p, f.sc.ldp, f.klen, nb, f.sc.sc, f.sc.ldsc, f.sc.nvalid, R, 0);
            CK(cudaDeviceSynchronize());
            const double a = span_us([&] { launch_topk<CAND_BLOCKS>(f.sc.sc, f.sc.ldsc, f.sc.nvalid, nb, f.sc.cand, CAND_BLOCKS, R, 0); });
            const double b = span_us([&] { launch_topk_mc<CAND_BLOCKS>(f.sc.sc, f.sc.ldsc, f.sc.nvalid, std::min(nb, nbv), f.sc.cand, CAND_BLOCKS, R, f.sc.tkw, MC_CTAS, 0); });
            printf("%-8d %-2d | %-9.1f %-9.1f\n", N, R, a, b);
        }
}

int main(int argc, char** argv)
{
    bool want[4] = {false, false, false, false}, small = false, any = false;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "probe")) want[0] = any = true;
        else if (!strcmp(argv[i], "topk")) want[1] = any = true;
        else if (!strcmp(argv[i], "logits")) want[2] = any = true;
        else if (!strcmp(argv[i], "timing")) want[3] = any = true;
        else if (!strcmp(argv[i], "small")) small = true;
    }
    if (!any) want[0] = want[1] = want[2] = want[3] = true;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s sm_%d%d, %d SMs%s\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount, small ? " (small)" : "");
    if (want[0]) test_probe();
    if (want[1]) { test_topk<TOPK>(small); test_topk<CAND_BLOCKS>(small); }
    if (want[2]) { test_logits(small, true, true); test_logits(small, false, true); test_logits(small, true, false); test_logits(small, false, false); }
    if (want[3]) timing(small);
    printf("\n%s (%d failed)\n", g_fail ? "SOME FAILED" : "ALL PASS", g_fail);
    return g_fail;
}
