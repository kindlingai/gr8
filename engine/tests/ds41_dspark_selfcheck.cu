// Device self-check of the DSpark kernels: every kernel in engine/runtime/ds41_dspark.cuh and engine/attn/ds41_attn_dsp.cuh is launched
// directly (no fabric, no Model, no weights file) on fixed-seed synthetic inputs and compared with a host reference. Run it first on a new
// box, before any canary; a failure here means the kernel is wrong, not the engine around it.
//
//   build: /usr/local/cuda/bin/nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I. -Iengine -Xcompiler -Wall
//              engine/tests/ds41_dspark_selfcheck.cu -o ds41_dspark_selfcheck
//   run:   ./ds41_dspark_selfcheck [name]        name = one test (tap, accept, markov, conf, rope, kvprep, attn, main); default = all in order,
//                                                the first failure stops the run. Exit 0 = all pass, 1 = a check failed, 2 = a CUDA error.
//
// The tolerances below are taken from the numerics of each kernel, not from measured values; a run may need one of them loosened,
// which is not a kernel bug. Host references reuse the same rule headers the kernels use (ds41_dspark.h, ds41_dspark_dev.h) for the integer logic, and double-precision
// maths for the float logic.
//
// Not covered here (needs the Model / fabric, so it runs in the step driver): the local argmax + head merge (tie rule: lowest index), the
// attn_proj_blk / attn_out_blk composites (they chain the attention gemm + act-quant kernels, which have their own checks), the draft layers' MoE.
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <vector>
#include <sys/resource.h>
#include "attn/ds41_attn_dsp.cuh"
#include "runtime/ds41_dspark.cuh"
#include "runtime/ds41_dspark.h"

#define CK(x)                                                                                                                  \
    do                                                                                                                         \
    {                                                                                                                          \
        const cudaError_t e_ = (x);                                                                                            \
        if (e_ != cudaSuccess)                                                                                                 \
        {                                                                                                                      \
            std::fprintf(stderr, "%s:%d: CUDA error from %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));            \
            std::exit(2);                                                                                                      \
        }                                                                                                                      \
    } while (0)

namespace
{
using bf16 = ds41a::bf16;
constexpr int HID = ds41a::HID, HD = ds41a::HD, NOPE = ds41a::NOPE, ROPE = ds41a::ROPE;
constexpr int RING = ds41::dsp::RING;   // 136
constexpr int BLK = ds41::dsp::B;       // 5

// ---------------------------------------------------------------------------------------------------------------- tiny helpers
struct Rng   // xorshift64*, fixed seed
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint32_t u32()
    {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    float uni() { return (float)(u32() >> 8) * (1.0f / 16777216.0f); }   // [0, 1)
    float sym(float a) { return (uni() * 2.0f - 1.0f) * a; }
    int below(int n) { return (int)(u32() % (uint32_t)n); }
};

size_t g_live = 0, g_peak = 0;   // device bytes held by Dev<> now / at most (printed at the end: the Marshal's memory cap)

template <class T> struct Dev
{
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t n_) : n(n_)
    {
        CK(cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)));
        g_live += (n ? n : 1) * sizeof(T);
        g_peak = std::max(g_peak, g_live);
    }
    ~Dev() { cudaFree(p); g_live -= (n ? n : 1) * sizeof(T); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h)
    {
        if (h.size() != n) { std::fprintf(stderr, "Dev::up size %zu != %zu\n", h.size(), n); std::exit(2); }
        CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> down() const
    {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void bytes(int v) { CK(cudaMemset(p, v, n * sizeof(T))); }
};

void sync() { CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
bf16 hb(float f) { return __float2bfloat16_rn(f); }
float hf(bf16 b) { return __bfloat162float(b); }
float bfr(float f) { return hf(hb(f)); }
uint16_t bits(bf16 b) { uint16_t u; std::memcpy(&u, &b, 2); return u; }
std::vector<bf16> rand_bf16(Rng& r, size_t n, float amp)
{
    std::vector<bf16> v(n);
    for (auto& x : v) x = hb(r.sym(amp));
    return v;
}
std::vector<float> plain_inv()   // plain rope, theta 10000, 64 rope dims: inv[i] = 10000^(-i/32)
{
    std::vector<float> v(ROPE / 2);
    for (int i = 0; i < ROPE / 2; i++) v[i] = (float)std::pow(10000.0, -(double)i / 32.0);
    return v;
}
// |got - ref| <= ulps * 2^-7 * |ref| + abs_tol  (2^-7 |ref| >= one BF16 ulp of ref)
bool close_bf16(float got, float ref, double ulps, double abs_tol) { return std::fabs((double)got - (double)ref) <= ulps * 0.0078125 * std::fabs((double)ref) + abs_tol; }

// ---------------------------------------------------------------------------------------------------------------- host fp8 / act-quant (twin of ds41a::act_quant_lane)
float f8dec(uint8_t c)   // e4m3 (bias 7, no inf; exp 15 / mantissa 7 = NaN)
{
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = std::ldexp((float)m, -9);
    else v = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}
float f8e8m0(uint8_t b) { return b ? std::ldexp(1.0f, (int)b - 127) : std::ldexp(1.0f, -127); }
int log2ceil_host(float x)
{
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0);
}
// one 32-block: codes, the exponent byte, and the dequantized values
void quant32(const float* x, uint8_t* codes, uint8_t& ebyte, float* deq)
{
    float a = 0.0f;
    for (int i = 0; i < 32; i++) a = std::max(a, std::fabs(x[i]));
    a = std::max(a, 1e-4f);
    const int e = log2ceil_host(a * ds41a::INV448);
    for (int i = 0; i < 32; i++)
    {
        const uint8_t c = (uint8_t)__nv_cvt_float_to_fp8(x[i] * std::ldexp(1.0f, -e), __NV_SATFINITE, __NV_E4M3);
        if (codes) codes[i] = c;
        if (deq) deq[i] = f8dec(c) * std::ldexp(1.0f, e);
    }
    ebyte = (uint8_t)(e + 127);
}
// act_quant of one bf16 row of K values: codes [K], scale bytes [K/32]
void act_quant_row(const bf16* x, int K, uint8_t* codes, uint8_t* sc)
{
    std::vector<float> f(32);
    for (int b = 0; b < K / 32; b++)
    {
        for (int i = 0; i < 32; i++) f[i] = hf(x[b * 32 + i]);
        quant32(f.data(), codes + b * 32, sc[b], nullptr);
    }
}
// the window-KV prep of one row (twin of k_kv_prep_pos): rms-norm (eps 1e-20) with weight nw, rope on the last 64 dims at position p, fp8 round trip over 512
std::vector<float> ref_kv_prep(const bf16* kv0, const bf16* nw, const float* inv, int p)
{
    double ss = 0.0;
    for (int c = 0; c < HD; c++) ss += (double)hf(kv0[c]) * hf(kv0[c]);
    const double r = 1.0 / std::sqrt(ss / (double)HD + (double)ds41a::EPS);
    std::vector<float> x(HD);
    for (int c = 0; c < HD; c++) x[c] = bfr((float)((double)hf(nw[c]) * ((double)hf(kv0[c]) * r)));
    for (int c = NOPE; c < HD; c += 2)
    {
        const int i = (c - NOPE) >> 1;
        const double ang = (double)((float)p * inv[i]);
        const double s = std::sin(ang), co = std::cos(ang);
        const double a = x[c], b = x[c + 1];
        x[c] = bfr((float)(a * co - b * s));
        x[c + 1] = bfr((float)(a * s + b * co));
    }
    std::vector<float> o(HD);
    uint8_t e;
    for (int b = 0; b < HD / 32; b++) quant32(x.data() + b * 32, nullptr, e, o.data() + b * 32);
    for (auto& v : o) v = bfr(v);
    return o;
}
// the fp8 quantizer can flip one code when the FP32 sum order differs: allow <= 1% of the elements to differ by <= 15% (one e4m3 step is 12.5%)
struct FlipStat
{
    long n = 0, diff = 0;
    double worst = 0.0;
    void add(float got, float ref)
    {
        n++;
        if (bits(hb(got)) == bits(hb(ref))) return;
        diff++;
        const double rel = std::fabs((double)got - ref) / (std::max(std::fabs((double)got), std::fabs((double)ref)) + 1e-3);
        worst = std::max(worst, rel);
    }
    bool ok() const { return diff * 100 <= n && worst <= 0.15; }
    std::string str() const
    {
        char b[256];
        std::snprintf(b, sizeof b, "%ld/%ld elements differ, worst rel %.3f", diff, n, worst);
        return b;
    }
};

bool fail(const char* name, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
bool fail(const char* name, const char* fmt, ...)
{
    char b[400];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    std::printf("FAIL %-7s %s\n", name, b);
    return false;
}
bool pass(const char* name, const std::string& msg)
{
    std::printf("PASS %-7s %s\n", name, msg.c_str());
    return true;
}

// ---------------------------------------------------------------------------------------------------------------- 1. ds_tap
bool t_tap()
{
    const int rows = 7;
    Rng r(11);
    std::vector<float> st((size_t)rows * 4 * HID);
    for (auto& v : st) v = r.sym(4.0f);
    Dev<float> ds(st.size());
    ds.up(st);
    Dev<bf16> tap((size_t)rows * 3 * HID);
    tap.bytes(0x7f);
    ds41dsp::ds_tap(ds.p, tap.p, 1, rows, 0);   // slot 1 alone first: slots 0 and 2 must stay untouched
    sync();
    {
        const std::vector<bf16> t = tap.down();
        for (int row = 0; row < rows; row++)
            for (int slot : {0, 2})
                for (int e = 0; e < HID; e += 97)
                    if (bits(t[((size_t)row * 3 + slot) * HID + e]) != 0x7f7f) return fail("tap", "slot %d of row %d was written by a slot-1 launch", slot, row);
    }
    ds41dsp::ds_tap(ds.p, tap.p, 0, rows, 0);
    ds41dsp::ds_tap(ds.p, tap.p, 2, rows, 0);
    sync();
    const std::vector<bf16> t = tap.down();
    long bad = 0;
    for (int row = 0; row < rows; row++)
        for (int slot = 0; slot < 3; slot++)
            for (int e = 0; e < HID; e++)
            {
                // every slot reads the same streams in this test (a real step reads the trunk at three different layers), so the reference is slot independent
                const float s[4] = {st[((size_t)row * 4 + 0) * HID + e], st[((size_t)row * 4 + 1) * HID + e], st[((size_t)row * 4 + 2) * HID + e], st[((size_t)row * 4 + 3) * HID + e]};
                const float ref = ds41::dsp::tap_mean(s);
                if (hf(t[((size_t)row * 3 + slot) * HID + e]) != ref && ++bad <= 3)
                    std::printf("  row %d slot %d e %d: got %.9g ref %.9g\n", row, slot, e, hf(t[((size_t)row * 3 + slot) * HID + e]), ref);
            }
    if (bad) return fail("tap", "%ld of %d BF16 values differ (the rule is bit exact)", bad, rows * 3 * HID);
    return pass("tap", "bit exact over 7 rows x 3 slots x 5120");
}

// ---------------------------------------------------------------------------------------------------------------- 2. accept + draft input + verify input
bool t_accept()
{
    Rng r(22);
    long cases = 0;
    for (int it = 0; it < 40; it++)
    {
        const int nsess = 1 + it % 4, k = 1 + (it / 4) % 5, rps = k + 1, M = nsess * rps;
        std::vector<int> vtok(M), vtgt(M), vpos(M), acc_ref(nsess), P(nsess);
        for (int s = 0; s < nsess; s++)
        {
            P[s] = r.below(100000);
            const int a = (it < 20 && nsess == 1) ? it % (k + 1) : r.below(k + 1);   // the first cases walk a = 0..k
            acc_ref[s] = a;
            vtok[s * rps] = r.below(129280);
            for (int j = 0; j < k; j++) vtok[s * rps + 1 + j] = r.below(129280);
            for (int j = 0; j < rps; j++) vtgt[s * rps + j] = r.below(129280);
            for (int j = 0; j < a; j++) vtgt[s * rps + j] = vtok[s * rps + 1 + j];                 // drafts 0 .. a-1 are accepted
            if (a < k) { int v = r.below(129280); if (v == vtok[s * rps + 1 + a]) v ^= 1; vtgt[s * rps + a] = v; }   // draft a is rejected
            for (int j = 0; j < rps; j++) vpos[s * rps + j] = P[s] + j;
        }
        Dev<int> dvtok(M), dvtgt(M), dvpos(M), dq(nsess), dg(nsess), da(nsess), dtok(nsess * BLK), dpos(nsess * BLK);
        dvtok.up(vtok); dvtgt.up(vtgt); dvpos.up(vpos);
        ds41dsp::ds_draft_in(dvtok.p, dvtgt.p, dvpos.p, rps, nsess, dq.p, dg.p, da.p, dtok.p, dpos.p, 0);
        sync();
        const std::vector<int> q = dq.down(), g = dg.down(), a = da.down(), tok = dtok.down(), pos = dpos.down();
        for (int s = 0; s < nsess; s++)
        {
            const ds41::dsp::Accept ref = ds41::dsp::accept_greedy(vtok.data() + s * rps + 1, vtgt.data() + s * rps, k, P[s]);
            if (ref.a != acc_ref[s]) return fail("accept", "internal: fixture accepted %d, host rule says %d", acc_ref[s], ref.a);
            if (a[s] != ref.a || q[s] != ref.q || g[s] != ref.golden)
                return fail("accept", "case %d session %d k %d: device a/q/golden %d/%d/%d, host %d/%d/%d", it, s, k, a[s], q[s], g[s], ref.a, ref.q, ref.golden);
            const ds41::dsp::Block blk = ds41::dsp::make_block(ref.golden, ref.q);
            for (int j = 0; j < BLK; j++)
                if (tok[s * BLK + j] != blk.tok[j] || pos[s * BLK + j] != blk.pos[j])
                    return fail("accept", "case %d session %d block row %d: device tok/pos %d/%d, host %d/%d", it, s, j, tok[s * BLK + j], pos[s * BLK + j], blk.tok[j], blk.pos[j]);
        }
        // the next verify step from this draft block
        std::vector<int> draft(nsess * BLK);
        for (auto& v : draft) v = r.below(129280);
        Dev<int> ddraft(nsess * BLK), vt(M), vp(M);
        ddraft.up(draft);
        ds41dsp::ds_verify_in(dg.p, ddraft.p, dq.p, k, nsess, vt.p, vp.p, 0);
        sync();
        const std::vector<int> ntok = vt.down(), npos = vp.down();
        for (int s = 0; s < nsess; s++)
        {
            const ds41::dsp::Verify ref = ds41::dsp::verify_rows(g[s], draft.data() + s * BLK, k, q[s] + 1);
            for (int j = 0; j < rps; j++)
                if (ntok[s * rps + j] != ref.tok[j] || npos[s * rps + j] != ref.pos[j])
                    return fail("accept", "case %d verify_in session %d row %d: device tok/pos %d/%d, host %d/%d", it, s, j, ntok[s * rps + j], npos[s * rps + j], ref.tok[j], ref.pos[j]);
        }
        cases++;
    }
    return pass("accept", std::to_string(cases) + " cases (k 1..5, 1..4 sessions, accepted 0..k): integer exact");
}

// ---------------------------------------------------------------------------------------------------------------- 3. markov head
bool t_markov()
{
    const int V = 2048, vn = 1021, ld = vn + 3;   // vn is not a multiple of 8 (the tail block), ld > vn (a padded logits row)
    Rng r(33);
    const std::vector<bf16> membed = rand_bf16(r, (size_t)V * 256, 1.0f), mhead = rand_bf16(r, (size_t)vn * 256, 1.0f);
    Dev<bf16> dmem((size_t)V * 256), dmh((size_t)vn * 256);
    dmem.up(membed);
    dmh.up(mhead);
    double worst = 0.0;
    for (const int nsess : {1, 3, 8})
    {
        std::vector<int> golden(nsess), dtok(BLK * nsess);
        for (auto& v : golden) v = r.below(V);
        for (auto& v : dtok) v = r.below(V);
        std::vector<float> hlog((size_t)nsess * BLK * ld);
        for (auto& v : hlog) v = r.sym(2.0f);
        Dev<int> dg(nsess), dd(BLK * nsess);
        Dev<float> dl(hlog.size());
        dg.up(golden); dd.up(dtok); dl.up(hlog);
        std::vector<float> ref = hlog;
        for (int i = 0; i < BLK; i++)
        {
            ds41dsp::markov_add(dmem.p, dmh.p, dg.p, dd.p, i, nsess, dl.p, ld, vn, 0);   // step i uses the merge output of step i - 1: dtok is read as given here
            for (int s = 0; s < nsess; s++)
            {
                const int prev = ds41::dspd::markov_prev(golden.data(), dtok.data(), nsess, s, i);
                for (int v = 0; v < vn; v++)
                {
                    double dot = 0.0;
                    for (int c = 0; c < 256; c++) dot += (double)hf(mhead[(size_t)v * 256 + c]) * hf(membed[(size_t)prev * 256 + c]);
                    ref[(size_t)(s * BLK + i) * ld + v] = (float)((double)ref[(size_t)(s * BLK + i) * ld + v] + dot);
                }
            }
        }
        sync();
        const std::vector<float> got = dl.down();
        for (size_t e = 0; e < got.size(); e++)
        {
            const double err = std::fabs((double)got[e] - ref[e]) / (1.0 + std::fabs((double)ref[e]));
            worst = std::max(worst, err);
            if (err > 1e-4) return fail("markov", "nsess %d element %zu (row %zu v %zu): got %.7g ref %.7g", nsess, e, e / ld, e % ld, got[e], ref[e]);
        }
        // the padding columns vn .. ld-1 must not be touched
        for (int row = 0; row < nsess * BLK; row++)
            for (int v = vn; v < ld; v++)
                if (got[(size_t)row * ld + v] != hlog[(size_t)row * ld + v]) return fail("markov", "padding column %d of row %d was written", v, row);
    }
    char b[256];
    std::snprintf(b, sizeof b, "nsess 1/3/8, 5 steps, shard %d: worst relative error %.2e (limit 1e-4)", vn, worst);
    return pass("markov", b);
}

// ---------------------------------------------------------------------------------------------------------------- 4. confidence head
bool t_conf()
{
    const int V = 2048, nsess = 2, rows = nsess * BLK;
    Rng r(44);
    const std::vector<bf16> x = rand_bf16(r, (size_t)rows * HID, 1.0f), membed = rand_bf16(r, (size_t)V * 256, 1.0f), proj = rand_bf16(r, 5376, 0.2f);
    std::vector<int> golden(nsess), dtok(BLK * nsess);
    for (auto& v : golden) v = r.below(V);
    for (auto& v : dtok) v = r.below(V);
    Dev<bf16> dx(x.size()), dm(membed.size()), dp(proj.size());
    Dev<int> dg(nsess), dd(dtok.size()), ddraft(rows);
    Dev<float> dc(rows);
    dx.up(x); dm.up(membed); dp.up(proj); dg.up(golden); dd.up(dtok);
    ddraft.bytes(0xff);
    ds41dsp::conf_head(dx.p, dm.p, dp.p, dg.p, dd.p, nsess, dc.p, ddraft.p, 0);
    sync();
    const std::vector<float> conf = dc.down();
    const std::vector<int> draft = ddraft.down();
    double worst = 0.0;
    for (int s = 0; s < nsess; s++)
        for (int i = 0; i < BLK; i++)
        {
            const int row = s * BLK + i, prev = ds41::dspd::markov_prev(golden.data(), dtok.data(), nsess, s, i);
            double dot = 0.0, mag = 0.0;
            for (int c = 0; c < HID; c++) { const double t = (double)hf(x[(size_t)row * HID + c]) * hf(proj[c]); dot += t; mag += std::fabs(t); }
            for (int c = 0; c < 256; c++) { const double t = (double)hf(membed[(size_t)prev * 256 + c]) * hf(proj[HID + c]); dot += t; mag += std::fabs(t); }
            const double err = std::fabs((double)conf[row] - dot) / (mag * 1e-3 + 1e-6);   // limit 1: error under 0.1% of the sum of the term magnitudes
            worst = std::max(worst, err);
            if (err > 1.0) return fail("conf", "row %d: raw dot got %.7g ref %.7g (term magnitude %.3g)", row, conf[row], dot, mag);
            if (draft[row] != dtok[i * nsess + s]) return fail("conf", "draft[%d] = %d, expected dtok[%d * %d + %d] = %d", row, draft[row], i, nsess, s, dtok[i * nsess + s]);
        }
    char b[256];
    std::snprintf(b, sizeof b, "%d rows: raw FP32 dot within %.2f of the 1e-3 limit; draft vector exact", rows, worst);
    return pass("conf", b);
}

// ---------------------------------------------------------------------------------------------------------------- 5. rope on the last 64 dims at explicit positions
bool t_rope()
{
    const int rows = 6, heads = 2;
    const int posl[rows] = {0, 1, 127, 128, 4000, 120000};
    Rng r(55);
    const std::vector<float> inv = plain_inv();
    const std::vector<bf16> x0 = rand_bf16(r, (size_t)rows * heads * HD, 2.0f);
    Dev<bf16> dx(x0.size());
    Dev<int> dpos(rows);
    Dev<float> dinv(ROPE / 2);
    dx.up(x0);
    dpos.up(std::vector<int>(posl, posl + rows));
    dinv.up(inv);
    const long tot = (long)rows * heads * (ROPE / 2);
    ds41dsp::k_rope_tail_pos<<<(unsigned)((tot + 255) / 256), 256>>>(dx.p, (long)heads * HD, rows, heads, dpos.p, dinv.p, 0);
    sync();
    const std::vector<bf16> fwd = dx.down();
    for (int row = 0; row < rows; row++)
        for (int h = 0; h < heads; h++)
            for (int c = 0; c < HD; c++)
            {
                const size_t e = ((size_t)row * heads + h) * HD + c;
                float ref;
                if (c < NOPE) ref = hf(x0[e]);
                else
                {
                    const int i = (c - NOPE) >> 1;
                    const double ang = (double)((float)posl[row] * inv[i]), s = std::sin(ang), co = std::cos(ang);
                    const double a = hf(x0[e - ((c - NOPE) & 1)]), b = hf(x0[e - ((c - NOPE) & 1) + 1]);
                    ref = (c & 1) == 0 ? (float)(a * co - b * s) : (float)(a * s + b * co);
                    const double tol = 0.0078125 * std::fabs((double)ref) + 0.0009765625 * (std::fabs(a) + std::fabs(b));
                    if (std::fabs((double)hf(fwd[e]) - ref) > tol) return fail("rope", "row %d head %d c %d pos %d: got %.6g ref %.6g", row, h, c, posl[row], hf(fwd[e]), ref);
                    continue;
                }
                if (bits(fwd[e]) != bits(x0[e])) return fail("rope", "row %d head %d c %d (non-rope dim) changed", row, h, c);
            }
    ds41dsp::k_rope_tail_pos<<<(unsigned)((tot + 255) / 256), 256>>>(dx.p, (long)heads * HD, rows, heads, dpos.p, dinv.p, 1);   // inverse undoes it
    sync();
    const std::vector<bf16> back = dx.down();
    double worst = 0.0;
    for (size_t e = 0; e < back.size(); e++)
    {
        const double err = std::fabs((double)hf(back[e]) - hf(x0[e])), lim = 0.0234375 * std::fabs((double)hf(x0[e])) + 0.004;   // two BF16 round trips: 3 x 2^-7
        worst = std::max(worst, err / lim);
        if (err > lim) return fail("rope", "inverse(forward(x)) differs at element %zu: %.6g vs %.6g", e, hf(back[e]), hf(x0[e]));
    }
    char b[256];
    std::snprintf(b, sizeof b, "6 positions up to 120000, forward within BF16 ulp, non-rope dims exact, round trip at %.2f of limit", worst);
    return pass("rope", b);
}

// ---------------------------------------------------------------------------------------------------------------- 6. window-KV prep into the ring
bool t_kvprep()
{
    const int nsess = 2, rps = 6, rows = nsess * rps;
    Rng r(66);
    const std::vector<float> inv = plain_inv();
    const std::vector<bf16> kv0 = rand_bf16(r, (size_t)rows * HD, 2.0f);
    std::vector<bf16> nw(HD);
    for (auto& v : nw) v = hb(1.0f + r.sym(0.1f));
    std::vector<int> pos(rows);
    for (int i = 0; i < rps; i++) { pos[i] = i; pos[rps + i] = 300 + i; }   // session 0 starts the sequence, session 1 wraps the 136-row ring (300 % 136 = 28)
    pos[rps + 5] = -1;                                                        // a skipped row: negative position writes nothing
    const size_t stride = (size_t)RING * HD;
    Dev<bf16> dkv(kv0.size()), dnw(HD), ring(stride * nsess);
    Dev<float> dinv(ROPE / 2);
    Dev<int> dpos(rows);
    dkv.up(kv0); dnw.up(nw); dinv.up(inv); dpos.up(pos);
    ring.bytes(0x5a);
    ds41dsp::k_kv_prep_pos<<<rows, 512>>>(dkv.p, dnw.p, dinv.p, dpos.p, rps, ring.p, stride, RING);
    sync();
    const std::vector<bf16> got = ring.down();
    std::vector<char> written((size_t)nsess * RING, 0);
    FlipStat fs;
    for (int row = 0; row < rows; row++)
    {
        if (pos[row] < 0) continue;
        const int s = row / rps, slot = pos[row] % RING;
        written[(size_t)s * RING + slot] = 1;
        const std::vector<float> ref = ref_kv_prep(kv0.data() + (size_t)row * HD, nw.data(), inv.data(), pos[row]);
        for (int c = 0; c < HD; c++) fs.add(hf(got[s * stride + (size_t)slot * HD + c]), ref[c]);
    }
    for (int s = 0; s < nsess; s++)
        for (int slot = 0; slot < RING; slot++)
            if (!written[(size_t)s * RING + slot])
                for (int c = 0; c < HD; c += 61)
                    if (bits(got[s * stride + (size_t)slot * HD + c]) != 0x5a5a) return fail("kvprep", "session %d slot %d was written but no row maps to it (slot = pos %% %d)", s, slot, RING);
    if (!fs.ok()) return fail("kvprep", "%s", fs.str().c_str());
    return pass("kvprep", fs.str() + "; slot = pos % 136, negative position skipped");
}

// ---------------------------------------------------------------------------------------------------------------- 7. block attention
bool t_attn()
{
    const int heads = 16, nsess = 2, rows = nsess * BLK;
    const int cases[3][2] = {{0, 5}, {127, 128}, {300, 129}};   // q of the two sessions: first window, exactly full, ring wrapped (300 % 136 = 28)
    const double scale = ds41a::SOFTMAX_SCALE;
    Rng r(77);
    double worst_all = 0.0;
    for (int cs = 0; cs < 3; cs++)
    {
        const size_t stride = (size_t)RING * HD;
        std::vector<bf16> ring(stride * nsess);
        for (auto& v : ring) v = hb(r.sym(40.0f));   // garbage: a kernel that reads outside the key list changes the result
        std::vector<std::vector<std::vector<float>>> keys(nsess);
        std::vector<std::vector<int>> kpos(nsess);
        for (int s = 0; s < nsess; s++)
        {
            kpos[s] = ds41::dsp::key_positions(cases[cs][s], BLK);
            for (const int p : kpos[s])
            {
                std::vector<float> k(HD);
                for (int c = 0; c < HD; c++) { const bf16 b = hb(r.sym(1.0f)); k[c] = hf(b); ring[s * stride + (size_t)(p % RING) * HD + c] = b; }
                keys[s].push_back(k);
            }
        }
        const std::vector<bf16> q = rand_bf16(r, (size_t)rows * heads * HD, 3.0f);
        std::vector<float> sink(heads);
        for (auto& v : sink) v = r.sym(1.0f);
        Dev<bf16> dring(ring.size()), dq(q.size()), dout(q.size());
        Dev<float> dsink(heads);
        Dev<int> dqpos(nsess);
        dring.up(ring); dq.up(q); dsink.up(sink); dqpos.up(std::vector<int>(cases[cs], cases[cs] + nsess));
        dout.bytes(0);
        ds41a::Weights W{};
        W.sink = dsink.p;
        ds41a::Scratch S{};
        S.q = dq.p;
        S.o = dout.p;
        ds41dsp::Ring R;
        R.base = dring.p; R.sess_stride = stride; R.ring_rows = RING;
        ds41dsp::attn_core_blk(W, R, S, rows, BLK, dqpos.p, heads, 0);
        sync();
        const std::vector<bf16> got = dout.down();
        double worst = 0.0;
        for (int t = 0; t < rows; t++)
        {
            const int s = t / BLK;
            const size_t nk = keys[s].size();
            for (int h = 0; h < heads; h++)
            {
                const bf16* qv = q.data() + ((size_t)t * heads + h) * HD;
                std::vector<double> sc(nk);
                double m = -1e300;
                for (size_t i = 0; i < nk; i++)
                {
                    double d = 0.0;
                    for (int c = 0; c < HD; c++) d += (double)hf(qv[c]) * keys[s][i][c];
                    sc[i] = d * scale;
                    m = std::max(m, sc[i]);
                }
                double den = std::exp((double)sink[h] - m);
                for (size_t i = 0; i < nk; i++) { sc[i] = std::exp(sc[i] - m); den += sc[i]; }
                double refmax = 0.0, err = 0.0;
                std::vector<double> o(HD, 0.0);
                for (size_t i = 0; i < nk; i++)
                    for (int c = 0; c < HD; c++) o[c] += sc[i] * keys[s][i][c];
                for (int c = 0; c < HD; c++) { o[c] /= den; refmax = std::max(refmax, std::fabs(o[c])); }
                for (int c = 0; c < HD; c++) err = std::max(err, std::fabs(o[c] - hf(got[((size_t)t * heads + h) * HD + c])));
                const double rel = err / (refmax + 1e-3);
                worst = std::max(worst, rel);
                if (rel > 2e-2) return fail("attn", "case q=%d/%d row %d head %d: max abs error %.4g, ref max %.4g (rel %.3g > 2e-2)", cases[cs][0], cases[cs][1], t, h, err, refmax, rel);
            }
        }
        worst_all = std::max(worst_all, worst);
    }
    char b[256];
    std::snprintf(b, sizeof b, "q 0/5, 127/128, 300/129, 16 heads, learned sink, no causal mask: worst relative error %.3g (limit 2e-2)", worst_all);
    return pass("attn", b);
}

// ---------------------------------------------------------------------------------------------------------------- 8. ds_main (tap -> main_proj -> norm -> wkv x3 -> kv prep -> rings)
// Staged: each stage's host reference takes the device output of the stage before it, so one wrong kernel does not hide behind another.
bool t_main()
{
    const int M = 6, rps = 6, N1 = HID, K1 = 3 * HID;
    const int P0 = 1000;   // slots 1000 % 136 = 48 .. 53
    Rng r(88);
    const std::vector<float> inv = plain_inv();
    const std::vector<bf16> tap = rand_bf16(r, (size_t)M * K1, 1.0f);
    auto make_fp8 = [&](int N, int K, std::vector<uint8_t>& w, std::vector<uint8_t>& s) {
        w.resize((size_t)N * K);
        s.resize((size_t)N * (K / 32));
        for (auto& c : w) { c = (uint8_t)r.u32(); if ((c & 0x7f) == 0x7f) c ^= 1; }   // no NaN codes
        for (auto& e : s) e = (uint8_t)(112 + r.below(5));                              // scales 2^-15 .. 2^-11: weights of order 0.01
    };
    std::vector<uint8_t> w0, s0, wk[3], sk[3];
    make_fp8(N1, K1, w0, s0);
    for (int j = 0; j < 3; j++) make_fp8(HD, HID, wk[j], sk[j]);
    std::vector<bf16> mnorm(HID), knorm[3];
    for (auto& v : mnorm) v = hb(1.0f + r.sym(0.1f));
    for (int j = 0; j < 3; j++) { knorm[j].resize(HD); for (auto& v : knorm[j]) v = hb(1.0f + r.sym(0.1f)); }

    Dev<uint8_t> dw0(w0.size()), ds0(s0.size()), dwk0(wk[0].size()), dsk0(sk[0].size()), dwk1(wk[1].size()), dsk1(sk[1].size()), dwk2(wk[2].size()), dsk2(sk[2].size());
    dw0.up(w0); ds0.up(s0); dwk0.up(wk[0]); dsk0.up(sk[0]); dwk1.up(wk[1]); dsk1.up(sk[1]); dwk2.up(wk[2]); dsk2.up(sk[2]);
    Dev<bf16> dtap(tap.size()), dmn(HID), dkn0(HD), dkn1(HD), dkn2(HD);
    dtap.up(tap); dmn.up(mnorm); dkn0.up(knorm[0]); dkn1.up(knorm[1]); dkn2.up(knorm[2]);
    Dev<float> dinv(ROPE / 2);
    dinv.up(inv);
    const size_t stride = (size_t)RING * HD;
    Dev<bf16> dring0(stride), dring1(stride), dring2(stride);
    dring0.bytes(0x5a); dring1.bytes(0x5a); dring2.bytes(0x5a);
    std::vector<int> pos(M);
    for (int i = 0; i < M; i++) pos[i] = P0 + i;
    Dev<int> dpos(M);
    dpos.up(pos);
    Dev<uint8_t> scratch(ds41dsp::MainS::bytes(M));

    ds41dsp::MainW W;
    W.main_proj = ds41a::Fp8Mat{dw0.p, ds0.p, N1, K1};
    W.main_norm = dmn.p;
    W.wkv[0] = ds41a::Fp8Mat{dwk0.p, dsk0.p, HD, HID};
    W.wkv[1] = ds41a::Fp8Mat{dwk1.p, dsk1.p, HD, HID};
    W.wkv[2] = ds41a::Fp8Mat{dwk2.p, dsk2.p, HD, HID};
    W.kv_norm[0] = dkn0.p; W.kv_norm[1] = dkn1.p; W.kv_norm[2] = dkn2.p;
    W.inv = dinv.p;
    bf16* rings[3] = {dring0.p, dring1.p, dring2.p};
    for (int j = 0; j < 3; j++) { W.ring[j].base = rings[j]; W.ring[j].sess_stride = stride; W.ring[j].ring_rows = RING; }
    ds41dsp::MainS S;
    S.carve(scratch.p, M);
    ds41dsp::ds_main(W, S, dtap.p, M, rps, dpos.p, 0);
    sync();

    // A. act-quant of the tap: bit exact
    std::vector<uint8_t> tq((size_t)M * K1), ts((size_t)M * K1 / 32);
    CK(cudaMemcpy(tq.data(), S.tq, tq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ts.data(), S.ts, ts.size(), cudaMemcpyDeviceToHost));
    {
        std::vector<uint8_t> rq(K1), rs(K1 / 32);
        for (int m = 0; m < M; m++)
        {
            act_quant_row(tap.data() + (size_t)m * K1, K1, rq.data(), rs.data());
            if (std::memcmp(rq.data(), tq.data() + (size_t)m * K1, K1) || std::memcmp(rs.data(), ts.data() + (size_t)m * (K1 / 32), K1 / 32))
                return fail("main", "stage A (act-quant of the 3 taps): codes or scale bytes differ in row %d", m);
        }
    }
    // B. main_proj: every 41st output column (full N is 5120 x 15360 per row, too slow for the unoptimized host loop), against the device codes
    std::vector<bf16> mh0((size_t)M * N1);
    CK(cudaMemcpy(mh0.data(), S.mh0, mh0.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> lut(256);
    for (int c = 0; c < 256; c++) lut[c] = f8dec((uint8_t)c);
    auto gemm_ref = [&](const uint8_t* A, const uint8_t* SA, int K, const uint8_t* Wc, const uint8_t* Ws, int n) {   // one output element, double accumulation over 32-blocks
        double acc = 0.0;
        for (int kb = 0; kb < K / 32; kb++)
        {
            double part = 0.0;
            for (int k = 0; k < 32; k++) part += (double)lut[A[kb * 32 + k]] * lut[Wc[(size_t)n * K + kb * 32 + k]];
            acc += part * ((double)f8e8m0(SA[kb]) * f8e8m0(Ws[(size_t)n * (K / 32) + kb]));
        }
        return acc;
    };
    long nb = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N1; n += 41)
        {
            const double ref = gemm_ref(tq.data() + (size_t)m * K1, ts.data() + (size_t)m * (K1 / 32), K1, w0.data(), s0.data(), n);
            const float got = hf(mh0[(size_t)m * N1 + n]);
            nb++;
            if (!close_bf16(got, (float)ref, 1.0, 1e-4)) return fail("main", "stage B (main_proj) row %d col %d: got %.6g ref %.6g", m, n, got, ref);
        }
    // C. main_norm (rms-norm of the device main_proj output): within 2 BF16 ulps
    std::vector<bf16> mh((size_t)M * N1);
    CK(cudaMemcpy(mh.data(), S.mh, mh.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    for (int m = 0; m < M; m++)
    {
        double ss = 0.0;
        for (int c = 0; c < N1; c++) ss += (double)hf(mh0[(size_t)m * N1 + c]) * hf(mh0[(size_t)m * N1 + c]);
        const double rr = 1.0 / std::sqrt(ss / N1 + (double)ds41a::EPS);
        for (int c = 0; c < N1; c++)
        {
            const double ref = (double)hf(mnorm[c]) * ((double)hf(mh0[(size_t)m * N1 + c]) * rr);
            if (!close_bf16(hf(mh[(size_t)m * N1 + c]), (float)ref, 2.0, 1e-6)) return fail("main", "stage C (main_norm) row %d col %d: got %.6g ref %.6g", m, c, hf(mh[(size_t)m * N1 + c]), ref);
        }
    }
    // D. act-quant of main_x (the device norm output): bit exact
    std::vector<uint8_t> hq((size_t)M * HID), hs((size_t)M * HID / 32);
    CK(cudaMemcpy(hq.data(), S.hq, hq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), S.hs, hs.size(), cudaMemcpyDeviceToHost));
    {
        std::vector<uint8_t> rq(HID), rs(HID / 32);
        for (int m = 0; m < M; m++)
        {
            act_quant_row(mh.data() + (size_t)m * HID, HID, rq.data(), rs.data());
            if (std::memcmp(rq.data(), hq.data() + (size_t)m * HID, HID) || std::memcmp(rs.data(), hs.data() + (size_t)m * (HID / 32), HID / 32))
                return fail("main", "stage D (act-quant of main_x): codes or scale bytes differ in row %d", m);
        }
    }
    // E. wkv j -> kv prep -> ring slot (pos % 136), j = 0..2 (full N = 512 per layer)
    FlipStat fs;
    for (int j = 0; j < 3; j++)
    {
        std::vector<bf16> ring(stride);
        CK(cudaMemcpy(ring.data(), rings[j], stride * sizeof(bf16), cudaMemcpyDeviceToHost));
        std::vector<char> written(RING, 0);
        for (int m = 0; m < M; m++)
        {
            std::vector<bf16> mk(HD);
            for (int n = 0; n < HD; n++) mk[n] = hb((float)gemm_ref(hq.data() + (size_t)m * HID, hs.data() + (size_t)m * (HID / 32), HID, wk[j].data(), sk[j].data(), n));
            const std::vector<float> ref = ref_kv_prep(mk.data(), knorm[j].data(), inv.data(), pos[m]);
            const int slot = pos[m] % RING;
            written[slot] = 1;
            for (int c = 0; c < HD; c++) fs.add(hf(ring[(size_t)slot * HD + c]), ref[c]);
        }
        for (int slot = 0; slot < RING; slot++)
            if (!written[slot] && bits(ring[(size_t)slot * HD]) != 0x5a5a) return fail("main", "stage E layer %d: slot %d written but no verify row maps to it", 40 + j, slot);
    }
    if (!fs.ok()) return fail("main", "stage E (wkv + kv prep + ring, 3 layers x 6 rows): %s", fs.str().c_str());
    char b[320];
    std::snprintf(b, sizeof b, "A-D exact / within BF16 (%ld main_proj columns checked), E: %s", nb, fs.str().c_str());
    return pass("main", b);
}

// ---------------------------------------------------------------------------------------------------------------- 9. ds_main_rows (prefill hook)
// M = 40 prompt rows from position 120 (the ring wraps at slot 136) through ds_main_rows with cap 8 / 16 / 32, against the decode route: ds_main on
// the same taps, 8 rows per call. Ring bytes of the 3 layers. cap 8 repeats the decode launches, so it must be byte equal; cap 16 / 32 change the GEMM
// tile only (diagnostic: the number of differing elements is printed, the quality bar for the hook is teacher-forced top-1 >= 90 %, not this).
bool t_main_rows()
{
    const int M = 304, N1 = HID, K1 = 3 * HID, P0 = 120;
    Rng r(89);
    const std::vector<float> inv = plain_inv();
    const std::vector<bf16> tap = rand_bf16(r, (size_t)M * K1, 1.0f);
    auto make_fp8 = [&](int N, int K, std::vector<uint8_t>& w, std::vector<uint8_t>& s) {
        w.resize((size_t)N * K);
        s.resize((size_t)N * (K / 32));
        for (auto& c : w) { c = (uint8_t)r.u32(); if ((c & 0x7f) == 0x7f) c ^= 1; }
        for (auto& e : s) e = (uint8_t)(112 + r.below(5));
    };
    std::vector<uint8_t> w0, s0, wk[3], sk[3];
    make_fp8(N1, K1, w0, s0);
    for (int j = 0; j < 3; j++) make_fp8(HD, HID, wk[j], sk[j]);
    std::vector<bf16> mnorm(HID), knorm[3];
    for (auto& v : mnorm) v = hb(1.0f + r.sym(0.1f));
    for (int j = 0; j < 3; j++) { knorm[j].resize(HD); for (auto& v : knorm[j]) v = hb(1.0f + r.sym(0.1f)); }
    Dev<uint8_t> dw0(w0.size()), ds0(s0.size()), dwk0(wk[0].size()), dsk0(sk[0].size()), dwk1(wk[1].size()), dsk1(sk[1].size()), dwk2(wk[2].size()), dsk2(sk[2].size());
    dw0.up(w0); ds0.up(s0); dwk0.up(wk[0]); dsk0.up(sk[0]); dwk1.up(wk[1]); dsk1.up(sk[1]); dwk2.up(wk[2]); dsk2.up(sk[2]);
    Dev<bf16> dtap(tap.size()), dmn(HID), dkn0(HD), dkn1(HD), dkn2(HD);
    dtap.up(tap); dmn.up(mnorm); dkn0.up(knorm[0]); dkn1.up(knorm[1]); dkn2.up(knorm[2]);
    Dev<float> dinv(ROPE / 2);
    dinv.up(inv);
    std::vector<int> pos(M);
    for (int i = 0; i < M; i++) pos[i] = P0 + i;
    Dev<int> dpos(M);
    dpos.up(pos);
    const size_t stride = (size_t)RING * HD;
    Dev<uint8_t> scratch(ds41dsp::MainS::bytes(128));
    ds41dsp::MainS S;
    S.carve(scratch.p, 128);
    std::unique_ptr<Dev<bf16>> rr[5][3];   // 0 = decode route (8 rows per ds_main), 1..4 = ds_main_rows cap 8 / 16 / 32 / 128
    for (auto& v : rr) for (auto& d : v) d.reset(new Dev<bf16>(stride));
    auto weights = [&](int route) {
        ds41dsp::MainW W;
        W.main_proj = ds41a::Fp8Mat{dw0.p, ds0.p, N1, K1};
        W.main_norm = dmn.p;
        W.wkv[0] = ds41a::Fp8Mat{dwk0.p, dsk0.p, HD, HID};
        W.wkv[1] = ds41a::Fp8Mat{dwk1.p, dsk1.p, HD, HID};
        W.wkv[2] = ds41a::Fp8Mat{dwk2.p, dsk2.p, HD, HID};
        W.kv_norm[0] = dkn0.p; W.kv_norm[1] = dkn1.p; W.kv_norm[2] = dkn2.p;
        W.inv = dinv.p;
        for (int j = 0; j < 3; j++) { rr[route][j]->bytes(0x5a); W.ring[j].base = rr[route][j]->p; W.ring[j].sess_stride = stride; W.ring[j].ring_rows = RING; }
        return W;
    };
    {
        ds41dsp::MainW W = weights(0);
        for (int r0 = 0; r0 < M; r0 += 8) ds41dsp::ds_main(W, S, dtap.p + (size_t)r0 * K1, 8, 8, dpos.p + r0, 0);
    }
    const int caps[4] = {8, 16, 32, 128};
    for (int c = 0; c < 4; c++)
    {
        ds41dsp::MainW W = weights(1 + c);
        ds41dsp::ds_main_rows(W, S, dtap.p, M, dpos.p, caps[c], 0);
    }
    sync();
    std::vector<bf16> ref[3];
    for (int j = 0; j < 3; j++) { ref[j].resize(stride); CK(cudaMemcpy(ref[j].data(), rr[0][j]->p, stride * sizeof(bf16), cudaMemcpyDeviceToHost)); }
    long diff[4] = {0, 0, 0, 0};
    for (int c = 0; c < 4; c++)
        for (int j = 0; j < 3; j++)
        {
            std::vector<bf16> g(stride);
            CK(cudaMemcpy(g.data(), rr[1 + c][j]->p, stride * sizeof(bf16), cudaMemcpyDeviceToHost));
            for (size_t e = 0; e < stride; e++) diff[c] += bits(g[e]) != bits(ref[j][e]);
        }
    long written = 0;
    for (int j = 0; j < 3; j++)
        for (size_t e = 0; e < stride; e++) written += bits(ref[j][e]) != 0x5a5a;
    if (written < (long)RING * HD * 3 * 99 / 100) return fail("main_rows", "the decode route wrote only %ld elements, expected the whole ring (%d slots x 512 x 3 layers)", written, RING);
    if (diff[0]) return fail("main_rows", "cap 8 differs from the decode route in %ld ring elements (same launches, must be byte equal)", diff[0]);
    char b[240];
    std::snprintf(b, sizeof b, "%d rows from pos %d (ring wraps twice): cap 8 byte equal to 8-row ds_main; diag: cap 16 differs in %ld, cap 32 in %ld, cap 128 in %ld of %zu ring elements x3", M, P0, diff[1], diff[2], diff[3], stride);
    return pass("main_rows", b);
}

struct Test { const char* name; bool (*fn)(); };
const Test TESTS[] = {{"tap", t_tap}, {"accept", t_accept}, {"markov", t_markov}, {"conf", t_conf}, {"rope", t_rope}, {"kvprep", t_kvprep}, {"attn", t_attn}, {"main", t_main}, {"main_rows", t_main_rows}};
}   // namespace

int main(int argc, char** argv)
{
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    std::printf("ds41_dspark_selfcheck on %s (sm_%d%d)\n", pr.name, pr.major, pr.minor);
    size_t free0 = 0, total0 = 0;
    CK(cudaFree(0));   // create the context first, so the context size is measured apart from the tensors
    CK(cudaMemGetInfo(&free0, &total0));
    const char* only = argc > 1 ? argv[1] : nullptr;
    int ran = 0;
    for (const Test& t : TESTS)
    {
        if (only && std::strcmp(only, t.name)) continue;
        ran++;
        if (!t.fn()) { std::printf("STOP at the first failure: %s\n", t.name); return 1; }
    }
    if (!ran) { std::fprintf(stderr, "no test named '%s'\n", only); return 2; }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::printf("memory: Dev<> tensors peak %.1f MB; device free at start %.1f MB of %.1f MB (context already counted); host max RSS %.1f MB\n",
                g_peak / 1048576.0, free0 / 1048576.0, total0 / 1048576.0, ru.ru_maxrss / 1024.0);
    std::printf("ALL %d CHECKS PASSED\n", ran);
    return 0;
}
