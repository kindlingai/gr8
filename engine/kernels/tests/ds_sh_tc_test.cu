// GPU test of the shared expert on the tensor-core path (GR8_DS_SH_TC=1).
// The TC chain is ds41_moe_ops.cuh shared_ro_tc, replicated with the same C entries and the same arguments: ds41_sh_act_quant(hc_y) ->
// ds41_gemm_mx_tc fc1 (M x 1152 x 5120) -> ds41_sh_swiglu_q -> ds41_gemm_mx_tc fc2 (M x 5120 x 640) -> Y bf16, then dsh::k_combine_mt_shb.
// The weights are bind_moe's repack: ds41_sh_unrecord of random MXFP8 records into W13 [1152][5120] and W2p [5120][640] (zeroed pad).
// The GL path (the default, DS41_SH_GL) is dsh::launch_shared_gl with the ds_ro_cfg.h shape, then dsh::k_combine_mt (fp32 shared term).
//   A  device repack == host repack (the index map attn/ds41_sh_tc_map.h on the host), bytewise, all 4 layer copies: W13, SW13, W2p, SW2p (pad 0).
//   B  TC chain vs a host fp64 model of the same W8A8 semantics, on 64 sampled rows of a 4096-row call, each GEMM fed the GPU's own operands:
//      B.x x codes + E8M0 bytes == the host act_quant (engram/act_quant.h, the formulas of act_quant_lane), bytewise (gate: 0 mismatches);
//      B.fc1 fc1 Y13 bf16 vs fp64 dot(x codes, W13) rounded to bf16; B.fc2 fc2 Y bf16 vs fp64 dot(act codes, W2p) rounded to bf16 (k < 576 only).
//      Gate: within 1 bf16 ulp on >= 99.9 % of elements, for B.fc1 and for B.fc2. Max / mean abs error reported.
//      B.act (info): act codes vs host quant of the host swiglu (expf, not __expf) of the GPU's Y13: mismatches counted, not gated.
//   C  TC chain vs the GL path (W8A16), M 2048 rows: rel-RMS and max abs of Y vs sh_out. Reported, not a gate.
//   D  row independence: Y of the 4096-row call == two 2048-row calls == 64 single-row calls (sampled rows), bitwise.
//   E  determinism: the 4096-row call 3 times, bitwise.
//   F  timing per layer (cudaEvent, 4 layer copies round robin so the weights are not L2-resident, warm-up + median of 7 reps x 10 layers) at M 2048 and
//      4096: GL (launch_shared_gl + k_combine_mt) vs TC (4 launches + k_combine_mt_shb); the combine has no routes (tok_slot -1: only the shared term
//      is read, the routed reads are the same in both). Per chunk = 40 x per layer; the saving per chunk at W2048 is printed.
//   CONTROL  act pad and W2p pad set nonzero (code 0x38 = 1.0, byte 127): B.fc2 must FAIL (proves B.fc2 sees the pad).
// Exit 0 = "SH_TC_ALL_PASS" (A, B.x, B.fc1, B.fc2, D, E, CONTROL; C and F are reported).
//   (no argument)  usage + memory need; no CUDA call.    gpu   run (spark only: aarch64 + sm_121; not next to a serve).
// Build on a spark:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -diag-suppress 20011,20014 -c -o ds41_attn.o engine/attn/ds41_attn.cu
//   nvcc -std=c++17 -O2 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -I engine/kernels -Xcompiler -Wall \
//        -o ds_sh_tc_test engine/kernels/tests/ds_sh_tc_test.cu ds41_attn.o
// This file does not compile off aarch64. -DDS_SH_TC_CC_ONLY=1 (an x86 compile check) lets it compile there; main then refuses before any
// CUDA call.
#if !defined(__aarch64__) && !defined(DS_SH_TC_CC_ONLY)
#error "ds_sh_tc_test.cu: spark (aarch64) only; -DDS_SH_TC_CC_ONLY=1 for an x86 compile check"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../ds_shared_gl.cuh"
#include "../ds_ro_cfg.h"
#include "../../attn/ds41_sh_tc_map.h"
#include "../../engram/act_quant.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define CE(x) do { const int r_ = (x); if (r_) { printf("entry %s returned %d at %s:%d\n", #x, r_, __FILE__, __LINE__); exit(2); } } while (0)

// the attention TU's C entries (attn/ds41_attn.cu), linked from ds41_attn.o
extern "C" int ds41_gemm_mx_tc(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const uint8_t* W, const uint8_t* SW, void* Y, int ldy,
                               int M, int N, int K, void* st);
extern "C" int ds41_sh_act_quant(const void* x, int ldx, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int K, void* st);
extern "C" int ds41_sh_unrecord(const uint8_t* rec, int T, int K, uint8_t* codes, int ldc, uint8_t* scales, int lds, void* st);
extern "C" int ds41_sh_swiglu_q(const void* y13, int ldy, int I, float lim, uint8_t* codes, int ldc, uint8_t* scales, int lds, int M, int KP, void* st);

constexpr int K1 = 5120, I = 576, KP = (I + 127) / 128 * 128, N13 = 2 * I, NOUT = 5120, TK = 6;   // ds41_moe_ops.cuh HID, IW, SH_KP at TP4
constexpr int MBIG = 4096, MW = 2048, NL = 4, NS = 64, CHUNK_LAYERS = 40;
constexpr float LIM = 10.f;                                                                       // geom::SWIGLU_LIMIT
constexpr size_t SH1 = static_cast<size_t>(I / 16) * (K1 / 32) * dsh::GR;                         // shared gate / up record bytes each
constexpr size_t SH2 = static_cast<size_t>(NOUT / 16) * (I / 32) * dsh::GR;                       // shared down record bytes
constexpr size_t XFW = static_cast<size_t>(MBIG / 32 * 4) * (K1 / 16 * 64);                       // xfrag words (ro_alloc)
static_assert(KP == 640, "fc2 K pad");

__device__ __forceinline__ uint32_t hash32(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
__global__ void k_fill_mx8(uint8_t* p, size_t nrec, uint32_t seed)                // ds_ro_w4096_test.cu: 512 E4M3 codes |x| < 2 (no NaN) + 16 scales 2^-10..2^-3
{
    const size_t tot = nrec * dsh::GR;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += 256ull * gridDim.x)
    {
        const uint32_t h = hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) * 0x9e3779b1u, seed);
        p[i] = static_cast<int>(i % dsh::GR) < 512 ? static_cast<uint8_t>((h & 0x80) | (h & 0x3f)) : static_cast<uint8_t>(117 + (h >> 24 & 7));
    }
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed)            // finite bf16 in [-4, 4)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        p[i] = __float2bfloat16_rn((static_cast<float>(hash32(static_cast<uint32_t>(i), seed) & 0xffffu) / 65535.f - 0.5f) * 8.f);
}
template <class T> static void dal(T*& p, size_t n) { CK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
template <class T> static std::vector<T> get(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

struct Layer { uint8_t *r1, *r3, *r2, *w13, *sw13, *w2, *sw2; };
struct Bufs
{
    Layer L[NL];
    __nv_bfloat16* x; uint32_t* xf; __half* sh_act; float* sh_out;                                  // GL
    uint8_t *xc, *xs, *ac, *as; __nv_bfloat16 *y13, *y;                                              // TC (y = the bf16 Y, aliased in sh_out in production)
    char* plan; __nv_bfloat16* out; __half* ypart;                                                   // combine (no routes)
};
// shared_ro_tc, split at the swiglu (the CONTROL edits the act pad between the two halves)
static void tc_fc1(const Bufs& b, const Layer& l, const __nv_bfloat16* x, int M)
{
    CE(ds41_sh_act_quant(x, K1, b.xc, K1, b.xs, K1 / 32, M, K1, nullptr));
    CE(ds41_gemm_mx_tc(b.xc, K1, b.xs, K1 / 32, l.w13, l.sw13, b.y13, N13, M, N13, K1, nullptr));
    CE(ds41_sh_swiglu_q(b.y13, N13, I, LIM, b.ac, KP, b.as, KP / 32, M, KP, nullptr));
}
static void tc_fc2(const Bufs& b, const uint8_t* w2, const uint8_t* sw2, __nv_bfloat16* y, int M)
{
    CE(ds41_gemm_mx_tc(b.ac, KP, b.as, KP / 32, w2, sw2, y, NOUT, M, NOUT, KP, nullptr));
}
static void tc_chain(const Bufs& b, const Layer& l, const __nv_bfloat16* x, int M, __nv_bfloat16* y) { tc_fc1(b, l, x, M); tc_fc2(b, l.w2, l.sw2, y, M); }
static void gl_chain(const Bufs& b, const Layer& l, int M)
{
    dsh::ArgsGL GA{l.r1, l.r3, l.r2, b.xf, M, LIM, b.sh_act, b.sh_out};
    CK(static_cast<cudaError_t>(dsh::launch_shared_gl<K1, I, NOUT, 8, 1, 2, 8, 3, DS41_SH_NT>(GA, DS41_SH_RG, 0)));
}
static void combine(const Bufs& b, const void* shared, bool shb, int M)
{
    using P = dsr::PlanMT<TK, MBIG>;
    dsh::CombineArgsMT<TK, MBIG> CA{reinterpret_cast<const P*>(b.plan), b.ypart, reinterpret_cast<const float*>(shared), b.out, M};
    if (shb) dsh::k_combine_mt_shb<NOUT, 128, gv::O_BF16, TK, MBIG><<<M * NOUT / (8 * 128), 128>>>(CA);
    else dsh::k_combine_mt<NOUT, 128, gv::O_BF16, TK, MBIG><<<M * NOUT / (8 * 128), 128>>>(CA);
}

// ---- host model
static float h_e4m3(uint8_t c) { __half_raw h = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3); return __half2float(__half(h)); }
static double h_e8m0(uint8_t b) { return b ? std::ldexp(1.0, (int)b - 127) : std::ldexp(1.0, -127); }
static float h_bf(uint16_t u) { uint32_t v = (uint32_t)u << 16; float f; memcpy(&f, &v, 4); return f; }
static uint16_t h_tobf(double d) { __nv_bfloat16 b = __float2bfloat16_rn((float)d); uint16_t u; memcpy(&u, &b, 2); return u; }
static int ord(uint16_t u) { return (u & 0x8000) ? -(int)(u & 0x7fff) : (int)u; }
// y[n] = sum_k code_a[k] * 2^ea[k/32] * code_w[n][k] * sw[n][k/32] over k < kl (fp64), for one row
static void h_dot_row(const uint8_t* ac, const uint8_t* as, const std::vector<uint8_t>& w, const std::vector<uint8_t>& sw, int N, int K, int kl, double* y)
{
    std::vector<double> a(kl);
    for (int k = 0; k < kl; ++k) a[k] = h_e4m3(ac[k]) * h_e8m0(as[k / 32]);
    static float lut[256];
    static bool init = false;
    if (!init) { for (int c = 0; c < 256; ++c) lut[c] = h_e4m3((uint8_t)c); init = true; }
    for (int n = 0; n < N; ++n)
    {
        const uint8_t* wr = w.data() + (size_t)n * K;
        const uint8_t* sr = sw.data() + (size_t)n * (K / 32);
        double s = 0;
        for (int g = 0; g < kl / 32; ++g)
        {
            double p = 0;
            for (int k = g * 32; k < g * 32 + 32; ++k) p += a[k] * lut[wr[k]];
            s += p * h_e8m0(sr[g]);
        }
        y[n] = s;
    }
}
struct Gate { long n = 0, in1 = 0; double maxe = 0, sume = 0; };
static void gate_add(Gate& g, uint16_t got, double ref)
{
    const uint16_t r = h_tobf(ref);
    const int d = std::abs(ord(got) - ord(r));
    ++g.n; g.in1 += d <= 1;
    const double e = std::fabs((double)h_bf(got) - ref);
    g.maxe = std::max(g.maxe, e); g.sume += e;
}
static bool gate_ok(const Gate& g) { return g.n > 0 && (double)g.in1 >= 0.999 * (double)g.n; }
static void gate_print(const char* name, const Gate& g)
{
    printf("%s: %ld elements, %ld within 1 bf16 ulp (%.4f %%), max abs err %.4g, mean abs err %.4g -> %s\n", name, g.n, g.in1, 100.0 * g.in1 / std::max(1L, g.n),
           g.maxe, g.sume / std::max(1L, g.n), gate_ok(g) ? "PASS" : "FAIL");
}
// host repack (the index map of k_mx_unrecord): records [T][K/32][528] -> codes [16T][ldc] + bytes [16T][lds]
static void h_unrecord(const std::vector<uint8_t>& rec, int T, int K, uint8_t* codes, int ldc, uint8_t* sc, int lds)
{
    const int ng = K / 32;
    for (int bi = 0; bi < T * ng; ++bi)
    {
        const int t = bi / ng, g = bi % ng;
        const uint8_t* r = rec.data() + (size_t)bi * ds41a::MX_REC_B;
        for (int i = 0; i < 256; ++i)
        {
            int row, col;
            ds41a::mx_rec_rc(i, row, col);
            for (int h = 0; h < 2; ++h) codes[(size_t)(16 * t + row) * ldc + 32 * g + 16 * h + col] = r[h * 256 + i];
            if (i < 8) { sc[(size_t)(16 * t + i) * lds + g] = r[512 + 2 * i]; sc[(size_t)(16 * t + i + 8) * lds + g] = r[512 + 2 * i + 1]; }
        }
    }
}
static float h_swiglu(float g, float u, float lim) { g = std::fmin(g, lim); u = std::fmin(std::fmax(u, -lim), lim); return g / (1.f + std::exp(-g)) * u; }

int main(int argc, char** argv)
{
    const size_t per_layer = 2 * SH1 + SH2 + (size_t)N13 * K1 + (size_t)N13 * (K1 / 32) + (size_t)NOUT * KP + (size_t)NOUT * (KP / 32);
    const size_t dev = per_layer * NL + (size_t)MBIG * K1 * 2 + XFW * 4 + (size_t)MBIG * I * 2 + (size_t)MBIG * NOUT * 4 + (size_t)MBIG * (K1 + K1 / 32 + KP + KP / 32)
                       + (size_t)MBIG * N13 * 2 + 4 * (size_t)MBIG * NOUT * 2 + sizeof(dsr::PlanMT<TK, MBIG>) + (size_t)NOUT * (KP + KP / 32);
    if (argc < 2 || strcmp(argv[1], "gpu"))
    {
        printf("usage: ds_sh_tc_test gpu   (spark only, not next to a serve)\nneed: device %.0f MB, host ~%.0f MB\n", dev / 1048576.0,
               (3.0 * MBIG * NOUT * 2 + per_layer * 2) / 1048576.0);
        return argc < 2 ? 0 : 2;
    }
#if !defined(__aarch64__)
    printf("REFUSED: gpu mode runs on a spark (aarch64) only (this is the DS_SH_TC_CC_ONLY compile check)\n");
    return 3;
#endif
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major * 10 + pr.minor != 121) { printf("REFUSED: device sm_%d%d is not sm_121\n", pr.major, pr.minor); return 3; }
    printf("device %s, %d SMs; need %.0f MB device\n", pr.name, pr.multiProcessorCount, dev / 1048576.0);
    const int g = pr.multiProcessorCount * 4;
    Bufs b;
    for (int l = 0; l < NL; ++l)
    {
        Layer& L = b.L[l];
        dal(L.r1, SH1); dal(L.r3, SH1); dal(L.r2, SH2);
        dal(L.w13, (size_t)N13 * K1); dal(L.sw13, (size_t)N13 * (K1 / 32)); dal(L.w2, (size_t)NOUT * KP); dal(L.sw2, (size_t)NOUT * (KP / 32));
        CK(cudaMemset(L.w13, 0xA5, (size_t)N13 * K1)); CK(cudaMemset(L.sw13, 0xA5, (size_t)N13 * (K1 / 32)));   // poison: every byte of W13 must be written
        CK(cudaMemset(L.w2, 0, (size_t)NOUT * KP)); CK(cudaMemset(L.sw2, 0, (size_t)NOUT * (KP / 32)));          // zeroed as dmalloc does (the pad)
        k_fill_mx8<<<g, 256>>>(L.r1, SH1 / dsh::GR, 21u + 10 * l); k_fill_mx8<<<g, 256>>>(L.r3, SH1 / dsh::GR, 22u + 10 * l);
        k_fill_mx8<<<g, 256>>>(L.r2, SH2 / dsh::GR, 23u + 10 * l);
        CK(cudaGetLastError());
        // bind_moe's repack, the same three calls
        CE(ds41_sh_unrecord(L.r1, I / 16, K1, L.w13, K1, L.sw13, K1 / 32, nullptr));
        CE(ds41_sh_unrecord(L.r3, I / 16, K1, L.w13 + (size_t)I * K1, K1, L.sw13 + (size_t)I * (K1 / 32), K1 / 32, nullptr));
        CE(ds41_sh_unrecord(L.r2, NOUT / 16, I, L.w2, KP, L.sw2, KP / 32, nullptr));
    }
    dal(b.x, (size_t)MBIG * K1); dal(b.xf, XFW); dal(b.sh_act, (size_t)MBIG * I); dal(b.sh_out, (size_t)MBIG * NOUT);
    dal(b.xc, (size_t)MBIG * K1); dal(b.xs, (size_t)MBIG * (K1 / 32)); dal(b.ac, (size_t)MBIG * KP); dal(b.as, (size_t)MBIG * (KP / 32));
    dal(b.y13, (size_t)MBIG * N13); dal(b.y, (size_t)MBIG * NOUT);
    dal(b.plan, sizeof(dsr::PlanMT<TK, MBIG>)); dal(b.out, (size_t)MBIG * NOUT); dal(b.ypart, 8 * NOUT);
    CK(cudaMemset(b.plan, 0xff, sizeof(dsr::PlanMT<TK, MBIG>)));                                       // tok_slot -1: the combine reads the shared term only
    k_fill_bf16<<<g, 256>>>(b.x, (size_t)MBIG * K1, 31u);
    CK(cudaMemset(b.xf, 0, XFW * 4));
    mx4::build_xfrag_mt<0><<<(MBIG / 8 * K1 * 4 + 255) / 256, 256>>>(b.x, MBIG, K1, MBIG / 8, b.xf);     // the router's xfrag (GL path input)
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    bool ok = true;

    // ---- A: device repack == host repack
    std::vector<std::vector<uint8_t>> hw13(NL), hsw13(NL), hw2(NL), hsw2(NL);
    {
        long bad = 0;
        for (int l = 0; l < NL; ++l)
        {
            const Layer& L = b.L[l];
            const auto r1 = get(L.r1, SH1), r3 = get(L.r3, SH1), r2 = get(L.r2, SH2);
            hw13[l].assign((size_t)N13 * K1, 0); hsw13[l].assign((size_t)N13 * (K1 / 32), 0);
            hw2[l].assign((size_t)NOUT * KP, 0); hsw2[l].assign((size_t)NOUT * (KP / 32), 0);
            h_unrecord(r1, I / 16, K1, hw13[l].data(), K1, hsw13[l].data(), K1 / 32);
            h_unrecord(r3, I / 16, K1, hw13[l].data() + (size_t)I * K1, K1, hsw13[l].data() + (size_t)I * (K1 / 32), K1 / 32);
            h_unrecord(r2, NOUT / 16, I, hw2[l].data(), KP, hsw2[l].data(), KP / 32);
            const auto d13 = get(L.w13, hw13[l].size()), ds13 = get(L.sw13, hsw13[l].size()), d2 = get(L.w2, hw2[l].size()), ds2 = get(L.sw2, hsw2[l].size());
            bad += (d13 != hw13[l]) + (ds13 != hsw13[l]) + (d2 != hw2[l]) + (ds2 != hsw2[l]);
        }
        printf("A device repack vs host repack, %d layers x 4 arrays (W13 %d x %d, SW13, W2p %d x %d with pad, SW2p): %ld arrays differ -> %s\n", NL, N13, K1, NOUT, KP, bad,
               bad ? "FAIL" : "PASS");
        ok &= bad == 0;
    }

    // ---- 4096-row call (B, D, E)
    tc_chain(b, b.L[0], b.x, MBIG, b.y); CK(cudaDeviceSynchronize());
    const auto ybig = get(reinterpret_cast<const uint16_t*>(b.y), (size_t)MBIG * NOUT);
    std::vector<int> rows;
    for (int i = 0; i < NS; ++i) rows.push_back(i == 0 ? 0 : i == NS - 1 ? MBIG - 1 : (int)((i * 2654435761u) % MBIG));
    // ---- B
    {
        const auto x = get(reinterpret_cast<const uint16_t*>(b.x), (size_t)MBIG * K1);
        const auto xc = get(b.xc, (size_t)MBIG * K1), xs = get(b.xs, (size_t)MBIG * (K1 / 32));
        const auto y13 = get(reinterpret_cast<const uint16_t*>(b.y13), (size_t)MBIG * N13);
        const auto ac = get(b.ac, (size_t)MBIG * KP), as = get(b.as, (size_t)MBIG * (KP / 32));
        long b0 = 0, b2 = 0, padnz = 0;
        Gate g1, g3;
        std::vector<double> ref(std::max(N13, NOUT));
        for (int r : rows)
        {
            for (int blk = 0; blk < K1 / 32; ++blk)                                                  // B.x: host act_quant of x
            {
                float am = 0;
                for (int k = 0; k < 32; ++k) am = std::fmax(am, std::fabs(h_bf(x[(size_t)r * K1 + blk * 32 + k])));
                const int e = ds41::engram::aq_exp(am);
                b0 += xs[(size_t)r * (K1 / 32) + blk] != ds41::engram::aq_scale_byte(e);
                for (int k = 0; k < 32; ++k) b0 += xc[(size_t)r * K1 + blk * 32 + k] != ds41::engram::aq_code(h_bf(x[(size_t)r * K1 + blk * 32 + k]), e);
            }
            h_dot_row(xc.data() + (size_t)r * K1, xs.data() + (size_t)r * (K1 / 32), hw13[0], hsw13[0], N13, K1, K1, ref.data());   // B.fc1
            for (int n = 0; n < N13; ++n) gate_add(g1, y13[(size_t)r * N13 + n], ref[n]);
            for (int blk = 0; blk < KP / 32; ++blk)                                                   // B.act (info) + the pad
            {
                if (blk * 32 >= I)
                {
                    padnz += as[(size_t)r * (KP / 32) + blk] != 0;
                    for (int k = 0; k < 32; ++k) padnz += ac[(size_t)r * KP + blk * 32 + k] != 0;
                    continue;
                }
                float v[32], am = 0;
                for (int k = 0; k < 32; ++k)
                {
                    v[k] = h_swiglu(h_bf(y13[(size_t)r * N13 + blk * 32 + k]), h_bf(y13[(size_t)r * N13 + I + blk * 32 + k]), LIM);
                    am = std::fmax(am, std::fabs(v[k]));
                }
                const int e = ds41::engram::aq_exp(am);
                b2 += as[(size_t)r * (KP / 32) + blk] != ds41::engram::aq_scale_byte(e);
                for (int k = 0; k < 32; ++k) b2 += ac[(size_t)r * KP + blk * 32 + k] != ds41::engram::aq_code(v[k], e);
            }
            h_dot_row(ac.data() + (size_t)r * KP, as.data() + (size_t)r * (KP / 32), hw2[0], hsw2[0], NOUT, KP, I, ref.data());          // B.fc2 (k < 576)
            for (int n = 0; n < NOUT; ++n) gate_add(g3, ybig[(size_t)r * NOUT + n], ref[n]);
        }
        printf("B.x x act_quant vs host (engram/act_quant.h), %d rows: %ld byte mismatches -> %s\n", NS, b0, b0 ? "FAIL" : "PASS");
        gate_print("B.fc1 fc1 Y13 vs fp64", g1);
        printf("B.act (info) act codes + bytes vs host swiglu (expf) + act_quant of the GPU Y13: %ld byte mismatches; pad nonzero bytes %ld (expect 0)\n", b2, padnz);
        gate_print("B.fc2 fc2 Y vs fp64 (k < 576)", g3);
        ok &= b0 == 0 && gate_ok(g1) && gate_ok(g3) && padnz == 0;
    }
    // ---- D: row independence
    {
        tc_chain(b, b.L[0], b.x, MW, b.y); tc_chain(b, b.L[0], b.x + (size_t)MW * K1, MW, b.y + (size_t)MW * NOUT); CK(cudaDeviceSynchronize());
        const auto yh = get(reinterpret_cast<const uint16_t*>(b.y), (size_t)MBIG * NOUT);
        long dh = 0, d1 = 0;
        for (int r = 0; r < MBIG; ++r) dh += memcmp(ybig.data() + (size_t)r * NOUT, yh.data() + (size_t)r * NOUT, NOUT * 2) != 0;
        for (int r : rows)
        {
            tc_chain(b, b.L[0], b.x + (size_t)r * K1, 1, b.y); CK(cudaDeviceSynchronize());
            const auto y1 = get(reinterpret_cast<const uint16_t*>(b.y), (size_t)NOUT);
            d1 += memcmp(ybig.data() + (size_t)r * NOUT, y1.data(), NOUT * 2) != 0;
        }
        printf("D row independence: 4096-row call vs two 2048-row calls: %ld rows differ; vs M = 1 calls (%d sampled rows): %ld differ -> %s\n", dh, NS, d1,
               dh || d1 ? "FAIL" : "PASS");
        ok &= dh == 0 && d1 == 0;
    }
    // ---- E: determinism
    {
        long de = 0;
        for (int rep = 0; rep < 3; ++rep)
        {
            CK(cudaMemset(b.y, 0xff, (size_t)MBIG * NOUT * 2));
            tc_chain(b, b.L[0], b.x, MBIG, b.y); CK(cudaDeviceSynchronize());
            de += get(reinterpret_cast<const uint16_t*>(b.y), (size_t)MBIG * NOUT) != ybig;
        }
        printf("E determinism: 3 runs of the 4096-row call vs the first: %ld differ -> %s\n", de, de ? "FAIL" : "PASS");
        ok &= de == 0;
    }
    // ---- C: TC vs GL (M 2048)
    {
        gl_chain(b, b.L[0], MW); tc_chain(b, b.L[0], b.x, MW, b.y); CK(cudaDeviceSynchronize());
        const auto so = get(b.sh_out, (size_t)MW * NOUT);
        const auto yt = get(reinterpret_cast<const uint16_t*>(b.y), (size_t)MW * NOUT);
        double se = 0, sr = 0, mx = 0, mref = 0;
        for (size_t i = 0; i < so.size(); ++i)
        {
            const double d = (double)h_bf(yt[i]) - so[i];
            se += d * d; sr += (double)so[i] * so[i]; mx = std::max(mx, std::fabs(d)); mref = std::max(mref, (double)std::fabs(so[i]));
        }
        printf("C TC (W8A8) vs GL (W8A16), %d rows: rel-RMS %.4g, max abs %.4g (max |GL| %.4g) [report only]\n", MW, std::sqrt(se / std::max(sr, 1e-300)), mx, mref);
    }
    // ---- CONTROL: act pad and W2p pad nonzero -> B.fc2 must fail
    {
        uint8_t *w2x, *sw2x;
        dal(w2x, (size_t)NOUT * KP); dal(sw2x, (size_t)NOUT * (KP / 32));
        CK(cudaMemcpy(w2x, b.L[0].w2, (size_t)NOUT * KP, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(sw2x, b.L[0].sw2, (size_t)NOUT * (KP / 32), cudaMemcpyDeviceToDevice));
        CK(cudaMemset2D(w2x + I, KP, 0x38, KP - I, NOUT)); CK(cudaMemset2D(sw2x + I / 32, KP / 32, 127, (KP - I) / 32, NOUT));
        tc_fc1(b, b.L[0], b.x, MBIG);
        CK(cudaMemset2D(b.ac + I, KP, 0x38, KP - I, MBIG)); CK(cudaMemset2D(b.as + I / 32, KP / 32, 127, (KP - I) / 32, MBIG));
        tc_fc2(b, w2x, sw2x, b.y, MBIG); CK(cudaDeviceSynchronize());
        const auto yc = get(reinterpret_cast<const uint16_t*>(b.y), (size_t)MBIG * NOUT);
        const auto ac = get(b.ac, (size_t)MBIG * KP), as = get(b.as, (size_t)MBIG * (KP / 32));
        Gate gc;
        std::vector<double> ref(NOUT);
        for (int r : rows)
        {
            h_dot_row(ac.data() + (size_t)r * KP, as.data() + (size_t)r * (KP / 32), hw2[0], hsw2[0], NOUT, KP, I, ref.data());
            for (int n = 0; n < NOUT; ++n) gate_add(gc, yc[(size_t)r * NOUT + n], ref[n]);
        }
        gate_print("CONTROL B.fc2 with nonzero pads (must FAIL)", gc);
        printf("CONTROL -> %s\n", gate_ok(gc) ? "FAIL (the gate did not see the pad)" : "PASS (B.fc2 failed as it must)");
        ok &= !gate_ok(gc);
        CK(cudaFree(w2x)); CK(cudaFree(sw2x));
    }
    // ---- F: timing per layer
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        constexpr int REPS = 7, LPR = 10;
        auto time = [&](bool tc, int M) {
            auto one = [&](int l) {
                if (tc) { tc_chain(b, b.L[l % NL], b.x, M, b.y); combine(b, b.y, true, M); }
                else { gl_chain(b, b.L[l % NL], M); combine(b, b.sh_out, false, M); }
            };
            for (int l = 0; l < 2 * NL; ++l) one(l);                                                   // warm-up
            CK(cudaDeviceSynchronize());
            std::vector<float> ms;
            for (int rep = 0; rep < REPS; ++rep)
            {
                CK(cudaEventRecord(e0));
                for (int l = 0; l < LPR; ++l) one(l);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float t = 0;
                CK(cudaEventElapsedTime(&t, e0, e1));
                ms.push_back(t / LPR);
            }
            CK(cudaGetLastError());
            std::sort(ms.begin(), ms.end());
            return ms[REPS / 2];
        };
        double save2048 = 0;
        for (int M : {MW, MBIG})
        {
            const float gl = time(false, M), tc = time(true, M);
            printf("F M %d: per layer GL %.3f ms, TC %.3f ms, saved %.3f ms; per chunk (x %d layers) GL %.2f ms, TC %.2f ms, saved %.2f ms\n", M, gl, tc, gl - tc,
                   CHUNK_LAYERS, gl * CHUNK_LAYERS, tc * CHUNK_LAYERS, (gl - tc) * CHUNK_LAYERS);
            if (M == MW) save2048 = (gl - tc) * CHUNK_LAYERS;
        }
        printf("F W2048 saved %.2f ms per chunk\n", save2048);
    }
    printf(ok ? "SH_TC_ALL_PASS\n" : "SH_TC_RED\n");
    return ok ? 0 : 1;
}
