// GR8_DRAFT_HEAD2 (kernels/head2.cuh, two-stage draft head) unit test and bench: one GB10, synthetic records, no pack.
//   dhead2_test [--time [reps]] [--rows O]
// Check (O = 4096 vocab rows of K = 6144, M = 1, 3, 8; random head-like MXFP8 records as dhead_i4_test):
//   1. k_mx_to_i2 against a host derivation from the MXFP8 values (head2.cuh header), byte for byte, every byte written
//   2. k_screen against a host fp64 dot of the same 2-bit values: |error| <= 2e-3 of the row's sum of |terms| (fp16
//      products in groups of 32, the screen's own arithmetic)
//   3. k_screen_top against the host rule on the GPU's screen logits: the same candidate list, in the same order
//   4. k_rescore against a host fp64 dot of the MXFP8 values (1e-5 of the sum of |terms|) and against the MXFP8 GEMV the
//      target's head runs (bind::make_gv MXFP8): the same logit within 1e-5 of the sum of |terms|
//   5. k_head_local_sl against glue::k_head_local on the same row logits (a fake fabric slot): byte-equal records on
//      every row whose best logit is a candidate; the draft = the MXFP8 GEMV's argmax there
//   Reported, not checked: the two-stage drafts against the MXFP8 GEMV's and the INT4 head's argmax, on random hidden
//   rows (flat logits: the worst case for a screen) and on planted ones (x = a w_t + b w_u + noise: one clear token).
//   Real drafts: an offline replay of 24k recorded draft-head inputs had no miss at SL >= 32.
// --time: one TP4 vocab slice (O = 38784): the MXFP8 GEMV, the INT4 head (tpc 1, ks 32 = the engine's), and the
//   two-stage head per part (screen, top, rescore) and in all, M = 1, 3, 8, mean of reps (CUDA events), bytes per launch.
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a <K1H2 flags> -DG6_NR=4 -DDEC_S2 -o dhead2_test dhead2_test.cu
// Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../kernels/glue.cuh"
#include "../../kernels/bind.cuh"
#include "../../kernels/head_i4.cuh"
#include "../../kernels/head2.cuh"

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

// canonical codes [n][K] (E4M3), scales [n][K/32] (E8M0) -> MXFP8 GEMV record [n/16][K/32][528] (as dhead_i4_test)
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
static float mxv(const std::vector<uint8_t>& c, const std::vector<uint8_t>& s, int K, int row, int k)
{
    return e4m3_host(c[(size_t)row * K + k]) * e8m0_host(s[(size_t)row * (K / 32) + k / 32]);
}
// the 2-bit records of head2.cuh's header, from the MXFP8 values: codes [n][96] x 16 bytes, scales [n][192] fp16; and
// the dequantized values [n][K]
static void i2_host(const std::vector<uint8_t>& c, const std::vector<uint8_t>& s, int n, int K, std::vector<uint8_t>& codes,
                    std::vector<__half>& sc, std::vector<float>& val)
{
    const int ng = K / 32, nch = K / 64;
    codes.assign((size_t)n * nch * 16, 0);
    sc.assign((size_t)n * ng, __float2half_rn(0.f));
    val.assign((size_t)n * K, 0.f);
    for (int i = 0; i < n; ++i)
        for (int g = 0; g < ng; ++g)
        {
            float v[32], amax = 0.f;
            for (int j = 0; j < 32; ++j) { v[j] = mxv(c, s, K, i, 32 * g + j); amax = std::max(amax, std::fabs(v[j])); }
            const __half h = __float2half_rn(amax / 1.5f);
            sc[(size_t)i * ng + g] = h;
            const float sf = __half2float(h);
            for (int j = 0; j < 32; ++j)
            {
                const int q = sf != 0.f ? std::min(std::max((int)std::floor(v[j] / sf + 2.f), 0), 3) : 2;
                const int e = (g & 1) * 32 + j, word = e >> 4, ii = e & 15;
                uint32_t* w = reinterpret_cast<uint32_t*>(&codes[((size_t)i * nch + g / 2) * 16]) + word;
                *w |= (uint32_t)q << (2 * (ii >> 1) + 16 * (ii & 1));
                val[(size_t)i * K + 32 * g + j] = (q - 1.5f) * sf;
            }
        }
}

static bind::Tensor mk(const void* p, long long a, long long b, long long c)
{
    bind::Tensor t;
    t.p = p; t.nd = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    return t;
}
// y [M][O] f32 = x [M][K] fp16 @ W^T through bind (fragment table + gv launch), as the engine's head
struct Gemv
{
    bind::Xfrag f;
    bind::Gv g;
    void make(const void* rec, int codec, int O, int K, int tpc, int ks, int M, const __half* x, u32* xf, float* y)
    {
        f.on = 1; f.src = x; f.ld = K; f.K = K; f.M = M; f.f16 = xf;
        const int GR = codec == bind::INT4 ? 288 : 528;
        if (const char* e = bind::make_gv(mk(rec, O / 16, K / 32, GR), codec, tpc, ks, M, xf, y, gv::O_F32, g)) { printf("make_gv: %s\n", e); exit(2); }
    }
    void run(cudaStream_t st) { bind::launch(f, st); bind::launch(g, st); }
};

// fake fabric for the head-local kernels: one collective, send slot only
struct FakeFab
{
    g6::FabricDev f{};
    u64* stepp = nullptr;
    unsigned* ticket = nullptr;
    void init(size_t slot)
    {
        f.ncoll = 1; f.nr = 1; f.rank = 0;
        f.stride = (slot + 8 + 255) & ~(size_t)255;
        CK(cudaMalloc(&f.send, 2 * f.stride));
        CK(cudaMalloc(&f.doorbell, 8));
        CK(cudaMalloc(&stepp, 8));
        CK(cudaMemset(stepp, 0, 8));
        CK(cudaMalloc(&ticket, 4));
        CK(cudaMemset(ticket, 0, 4));
    }
};

static void check(int O, uint32_t seed)
{
    const int K = HID, ng = K / 32;
    std::vector<uint8_t> c, s;
    random_mx(O, K, seed, c, s);
    const std::vector<uint8_t> rec = mx_record(c, s, O, K);
    std::vector<uint8_t> hcodes;
    std::vector<__half> hsc;
    std::vector<float> hval;
    i2_host(c, s, O, K, hcodes, hsc, hval);
    uint8_t *dmx, *dcodes;
    __half2* dsc;
    CK(cudaMalloc(&dmx, rec.size()));
    CK(cudaMemcpy(dmx, rec.data(), rec.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dcodes, head2::code_bytes(O)));
    CK(cudaMalloc(&dsc, head2::scale_bytes(O)));
    CK(cudaMemset(dcodes, 0xa5, head2::code_bytes(O)));
    CK(cudaMemset(dsc, 0xa5, head2::scale_bytes(O)));
    head2::convert(dmx, reinterpret_cast<uint4*>(dcodes), dsc, O, nullptr);
    CK(cudaDeviceSynchronize());
    {
        std::vector<uint8_t> gc(head2::code_bytes(O));
        std::vector<__half> gs((size_t)O * ng);
        CK(cudaMemcpy(gc.data(), dcodes, gc.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gs.data(), dsc, gs.size() * 2, cudaMemcpyDeviceToHost));
        size_t bc = 0, bs = 0;
        for (size_t i = 0; i < gc.size(); ++i) bc += gc[i] != hcodes[i];
        for (size_t i = 0; i < gs.size(); ++i) bs += memcmp(&gs[i], &hsc[i], 2) != 0;
        report(bc == 0 && bs == 0, "k_mx_to_i2 O=" + std::to_string(O) + ": codes " + std::to_string(bc) + " bytes and scales " +
                                       std::to_string(bs) + " differ from the host derivation");
    }
    for (int planted = 0; planted < 2; ++planted)
        for (int M : {1, 3, 8})
        {
            // hidden rows: random fp16 in [-2, 2], or planted x = 2.5 w_t / |w_t|^2-ish + 1.5 w_u + noise
            std::vector<__half> hx((size_t)M * K);
            for (int m = 0; m < M; ++m)
            {
                const int tt = (int)(rnd(m, seed + 9) % O), uu = (int)(rnd(m, seed + 10) % O);
                double nt = 0, nu = 0;
                for (int k = 0; k < K; ++k) { const double a = mxv(c, s, K, tt, k), b = mxv(c, s, K, uu, k); nt += a * a; nu += b * b; }
                for (int k = 0; k < K; ++k)
                {
                    float v = 4.f * ((rnd((uint64_t)m * K + k, seed + 11) >> 8) * (1.f / 16777216.f)) - 2.f;
                    if (planted) v = (float)(0.25 * v + 6.0 * mxv(c, s, K, tt, k) / std::sqrt(nt) * std::sqrt(K) * 0.1 +
                                             4.0 * mxv(c, s, K, uu, k) / std::sqrt(nu) * std::sqrt(K) * 0.1);
                    hx[(size_t)m * K + k] = __float2half_rn(v);
                }
            }
            __half* dx;
            float *dy, *dmxy, *di4y, *dcl;
            int* dcand;
            u32* xf;
            CK(cudaMalloc(&dx, hx.size() * 2));
            CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dy, (size_t)M * O * 4));
            CK(cudaMalloc(&dmxy, (size_t)M * O * 4));
            CK(cudaMalloc(&di4y, (size_t)M * O * 4));
            CK(cudaMalloc(&dcl, (size_t)M * head2::SL * 4));
            CK(cudaMalloc(&dcand, (size_t)M * head2::SL * 4));
            CK(cudaMalloc(&xf, (size_t)K / 16 * 64 * 4 * g6mx::NT));
            // the engine's heads: MXFP8 (the exact reference) and INT4 (today's draft head)
            uint8_t* di4;
            CK(cudaMalloc(&di4, head_i4::bytes(O / 16, ng)));
            head_i4::launch(dmx, di4, O / 16, ng, nullptr);
#if DEC_I4PAIR
            launch_i4pair(di4, O / 16, ng, 32, nullptr);
#endif
            Gemv gm, gi;
            gm.make(dmx, bind::MXFP8, O, K, 1, 32, M, dx, xf, dmxy);
            gm.run(nullptr);
            gi.make(di4, bind::INT4, O, K, 1, 32, M, dx, xf, di4y);
            gi.run(nullptr);
            head2::Bufs b;
            b.codes = reinterpret_cast<const uint4*>(dcodes); b.scales = dsc; b.mx = dmx; b.cand = dcand; b.cl = dcl;
            head2::launch_screen(b.codes, b.scales, dx, K, dy, O, O, M, nullptr);
            CK(cudaDeviceSynchronize());
            const std::string tag = std::string(planted ? "planted" : "random") + " M=" + std::to_string(M) + " O=" + std::to_string(O);
            std::vector<float> ys((size_t)M * O), ym((size_t)M * O), yi((size_t)M * O);
            CK(cudaMemcpy(ys.data(), dy, ys.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ym.data(), dmxy, ym.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(yi.data(), di4y, yi.size() * 4, cudaMemcpyDeviceToHost));
            {   // 2. the screen against fp64
                double worst = 0;
                for (int m = 0; m < M; ++m)
                    for (int v = 0; v < O; ++v)
                    {
                        double d = 0, a = 0;
                        for (int k = 0; k < K; ++k)
                        {
                            const double t = (double)hval[(size_t)v * K + k] * __half2float(hx[(size_t)m * K + k]);
                            d += t; a += std::fabs(t);
                        }
                        worst = std::max(worst, std::fabs(ys[(size_t)m * O + v] - d) / (a + 1e-30));
                    }
                char buf[160];
                snprintf(buf, sizeof buf, "k_screen %s: worst |error| / sum|terms| %.2e (limit 2e-3)", tag.c_str(), worst);
                report(worst <= 2e-3, buf);
            }
            // 3. the candidates: host rule on the GPU's screen logits
            head2::k_screen_top<<<M, 1024>>>(dy, O, O, dcand);
            CK(cudaDeviceSynchronize());
            std::vector<int> gcand((size_t)M * head2::SL);
            CK(cudaMemcpy(gcand.data(), dcand, gcand.size() * 4, cudaMemcpyDeviceToHost));
            {
                size_t bad = 0;
                for (int m = 0; m < M; ++m)
                {
                    const float* row = &ys[(size_t)m * O];
                    float mx = -INFINITY;
                    for (int v = 0; v < O; ++v) mx = std::max(mx, row[v]);
                    auto bin = [&](float v) { const float bb = (mx - v) * (head2::BINS / head2::RANGE); return bb < (float)(head2::BINS - 1) ? (int)bb : head2::BINS - 1; };
                    std::vector<int> hist(head2::BINS, 0);
                    for (int v = 0; v < O; ++v) ++hist[bin(row[v])];
                    int cut = 0, lt = 0;
                    while (lt + hist[cut] < head2::SL) lt += hist[cut++];
                    std::vector<int> want;
                    for (int v = 0; v < O; ++v) if (bin(row[v]) < cut) want.push_back(v);
                    for (int v = 0; v < O && (int)want.size() < head2::SL; ++v) if (bin(row[v]) == cut) want.push_back(v);
                    for (int j = 0; j < head2::SL; ++j) bad += want[j] != gcand[(size_t)m * head2::SL + j];
                }
                report(bad == 0, "k_screen_top " + tag + ": " + std::to_string(bad) + " candidate entries differ from the host rule");
            }
            // 4. rescoring against fp64 and the MXFP8 GEMV
            head2::k_rescore<<<dim3(M, head2::SL / 8), 256>>>(dmx, dx, K, dcand, dy, O, dcl);
            CK(cudaDeviceSynchronize());
            std::vector<float> cl((size_t)M * head2::SL), y2((size_t)M * O);
            CK(cudaMemcpy(cl.data(), dcl, cl.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y2.data(), dy, y2.size() * 4, cudaMemcpyDeviceToHost));
            {
                double w64 = 0, wgv = 0;
                size_t ybad = 0;
                for (int m = 0; m < M; ++m)
                    for (int j = 0; j < head2::SL; ++j)
                    {
                        const int v = gcand[(size_t)m * head2::SL + j];
                        double d = 0, a = 0;
                        for (int k = 0; k < K; ++k)
                        {
                            const double t = (double)mxv(c, s, K, v, k) * __half2float(hx[(size_t)m * K + k]);
                            d += t; a += std::fabs(t);
                        }
                        const float e = cl[(size_t)m * head2::SL + j];
                        w64 = std::max(w64, std::fabs(e - d) / (a + 1e-30));
                        wgv = std::max(wgv, std::fabs((double)e - ym[(size_t)m * O + v]) / (a + 1e-30));
                        ybad += memcmp(&y2[(size_t)m * O + v], &e, 4) != 0;
                    }
                char buf[200];
                snprintf(buf, sizeof buf, "k_rescore %s: worst |error| / sum|terms| %.2e vs fp64, %.2e vs the MXFP8 GEMV (limit 1e-5); %zu row logits not written",
                         tag.c_str(), w64, wgv, ybad);
                report(w64 <= 1e-5 && wgv <= 1e-5 && ybad == 0, buf);
            }
            // 5. head-local records: k_head_local_sl vs k_head_local on the same rows (greedy, no top_p)
            {
                FakeFab F;
                F.init(glue::head_bytes(M, 0));
                glue::SampleRow* rows;
                int *tok, *tok2;
                float *lp, *lp2;
                CK(cudaMalloc(&rows, M * sizeof(glue::SampleRow)));
                std::vector<glue::SampleRow> hr(M);
                for (int m = 0; m < M; ++m) hr[m] = glue::SampleRow{0.f, 1u, 0u, 100 + m, 0, 0.f, 0, 0};
                CK(cudaMemcpy(rows, hr.data(), M * sizeof(glue::SampleRow), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&tok, M * 4)); CK(cudaMalloc(&tok2, M * 4)); CK(cudaMalloc(&lp, M * 4)); CK(cudaMalloc(&lp2, M * 4));
                glue::HeadArgs a{F.f, F.stepp, 0, M, dy, O, 1000, O, rows, F.ticket, tok, lp};
                std::vector<glue::HeadRec> r1(M), r2(M);
                glue::k_head_local<<<M, 1024>>>(a);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(r1.data(), F.f.send, M * sizeof(glue::HeadRec), cudaMemcpyDeviceToHost));
                head2::k_head_local_sl<<<M, 1024>>>(a, dcand, dcl);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(r2.data(), F.f.send, M * sizeof(glue::HeadRec), cudaMemcpyDeviceToHost));
                int eq = 0, comparable = 0, bad = 0, agree_mx = 0, agree_i4 = 0, cand_has = 0;
                for (int m = 0; m < M; ++m)
                {
                    const float* row = &y2[(size_t)m * O];
                    const int best = (int)(std::max_element(row, row + O) - row);
                    bool in = false;
                    for (int j = 0; j < head2::SL; ++j) in |= gcand[(size_t)m * head2::SL + j] == best;
                    if (in) { ++comparable; if (!memcmp(&r1[m], &r2[m], sizeof(glue::HeadRec))) ++eq; else ++bad; }
                    const int amx = (int)(std::max_element(&ym[(size_t)m * O], &ym[(size_t)m * O] + O) - &ym[(size_t)m * O]);
                    const int ai4 = (int)(std::max_element(&yi[(size_t)m * O], &yi[(size_t)m * O] + O) - &yi[(size_t)m * O]);
                    bool has = false;
                    for (int j = 0; j < head2::SL; ++j) has |= gcand[(size_t)m * head2::SL + j] == amx;
                    cand_has += has;
                    agree_mx += r2[m].id - 1000 == amx;
                    agree_i4 += ai4 == amx;
                }
                report(bad == 0, "k_head_local_sl " + tag + ": " + std::to_string(eq) + " of " + std::to_string(comparable) +
                                     " rows (best logit a candidate) byte-equal to k_head_local");
                printf("INFO %s: MXFP8 argmax among the candidates %d/%d; two-stage draft == MXFP8 GEMV argmax %d/%d; INT4 head == MXFP8 %d/%d\n",
                       tag.c_str(), cand_has, M, agree_mx, M, agree_i4, M);
                cudaFree(rows); cudaFree(tok); cudaFree(tok2); cudaFree(lp); cudaFree(lp2);
                cudaFree(F.f.send); cudaFree(F.f.doorbell); cudaFree(F.stepp); cudaFree(F.ticket);
            }
            cudaFree(dx); cudaFree(dy); cudaFree(dmxy); cudaFree(di4y); cudaFree(dcl); cudaFree(dcand); cudaFree(xf); cudaFree(di4);
        }
    cudaFree(dmx); cudaFree(dcodes); cudaFree(dsc);
}

// one TP4 vocab slice: MXFP8 GEMV, INT4 head, two-stage head (parts and all)
static void time_heads(int O, int reps)
{
    const int K = HID, ng = K / 32;
    const size_t mxb = (size_t)O / 16 * ng * 528, i4b = head_i4::bytes(O / 16, ng), i2b = head2::bytes(O);
    std::vector<uint8_t> h(mxb);
    for (size_t i = 0; i < mxb; ++i)
    {
        uint8_t v = (uint8_t)rnd(i, 7u);
        if (i % 528 >= 512) v = (uint8_t)(116 + v % 6);
        else if ((v & 0x7f) == 0x7f) v &= 0xbf;
        h[i] = v;
    }
    uint8_t *dmx, *di4, *dcodes;
    __half2* dsc;
    CK(cudaMalloc(&dmx, mxb));
    CK(cudaMemcpy(dmx, h.data(), mxb, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&di4, i4b));
    head_i4::launch(dmx, di4, O / 16, ng, nullptr);
#if DEC_I4PAIR
    launch_i4pair(di4, O / 16, ng, 32, nullptr);
#endif
    CK(cudaMalloc(&dcodes, head2::code_bytes(O)));
    CK(cudaMalloc(&dsc, head2::scale_bytes(O)));
    head2::convert(dmx, reinterpret_cast<uint4*>(dcodes), dsc, O, nullptr);
    __half* dx;
    float *dy, *dcl;
    int* dcand;
    u32* xf;
    CK(cudaMalloc(&dx, (size_t)8 * K * 2));
    CK(cudaMemset(dx, 0x31, (size_t)8 * K * 2));
    CK(cudaMalloc(&dy, (size_t)8 * O * 4));
    CK(cudaMalloc(&dcl, (size_t)8 * head2::SL * 4));
    CK(cudaMalloc(&dcand, (size_t)8 * head2::SL * 4));
    CK(cudaMalloc(&xf, (size_t)K / 16 * 64 * 4 * g6mx::NT));
    CK(cudaDeviceSynchronize());
    head2::Bufs b;
    b.codes = reinterpret_cast<const uint4*>(dcodes); b.scales = dsc; b.mx = dmx; b.cand = dcand; b.cl = dcl;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timeit = [&](const std::function<void()>& f) {
        for (int i = 0; i < 10; ++i) f();
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return 1e3 * ms / reps;
    };
    for (int M : {1, 3, 8})
    {
        Gemv gm, gi;
        gm.make(dmx, bind::MXFP8, O, K, 1, 32, M, dx, xf, dy);
        gi.make(di4, bind::INT4, O, K, 1, 32, M, dx, xf, dy);
        const double tm = timeit([&] { gm.run(nullptr); });
        const double ti = timeit([&] { gi.run(nullptr); });
        const double ts = timeit([&] { head2::launch_screen(b.codes, b.scales, dx, K, dy, O, O, M, nullptr); });
        const double tt = timeit([&] { head2::k_screen_top<<<M, 1024>>>(dy, O, O, dcand); });
        const double tr = timeit([&] { head2::k_rescore<<<dim3(M, head2::SL / 8), 256>>>(dmx, dx, K, dcand, dy, O, dcl); });
        const double ta = timeit([&] { head2::launch_front(b, dx, K, dy, O, O, M, nullptr); });
        printf("TIME O=%d M=%d: MXFP8 GEMV %.1f us (%.1f MB, %.0f GB/s) | INT4 head %.1f us (%.1f MB, %.0f GB/s) | two-stage %.1f us = screen %.1f "
               "(%.1f MB, %.0f GB/s) + top %.1f + rescore %.1f | vs INT4 %+.1f us\n",
               O, M, tm, mxb / 1e6, mxb / tm / 1e3, ti, i4b / 1e6, i4b / ti / 1e3, ta, ts, i2b / 1e6, i2b / ts / 1e3, tt, tr, ta - ti);
        fflush(stdout);
    }
    cudaFree(dmx); cudaFree(di4); cudaFree(dcodes); cudaFree(dsc); cudaFree(dx); cudaFree(dy); cudaFree(dcl); cudaFree(dcand); cudaFree(xf);
}

int main(int argc, char** argv)
{
    int reps = 0, O = 38784;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--time")) reps = i + 1 < argc && argv[i + 1][0] != '-' ? atoi(argv[++i]) : 200;
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) O = atoi(argv[++i]);
        else { printf("dhead2_test [--time [reps]] [--rows O]\n"); return 2; }
    }
    CK(cudaSetDevice(0));
    CK(bind::init());
    if (const char* e = head2::init()) { printf("%s\n", e); return 2; }
    check(4096, 1234u);
    if (reps > 0) time_heads(O, reps);
    printf("%d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
