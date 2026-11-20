// Probe of the block-scaled tensor-core operand layouts and of the device quantizers (sm_121a). Our own code.
//
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -I engine/kernels \
//        engine/kernels/tests/nvfp4_mma_probe.cu -o /tmp/nvfp4_mma_probe && /tmp/nvfp4_mma_probe
//
// 1. NVFP4: every basis coordinate (row 0..15, column 0..7, k 0..63) of a record nv4g through nv4::Frag (load, loadB,
//    mma kind::mxf4nvf4 scale_vec::4X m16n8k64) against nv4pack::decode_group x nv4pack::dequantize. One-hot cases have
//    one product per output, so they must be bit exact; every block has its own scale.
// 2. MXFP8: the same for the 528-byte record through ink2::Mx8Frag (mma kind::mxf8f6f4 scale_vec::1X m16n8k32)
//    against ink2ref::mx8::decode_group x mx8::dequantize.
// 3. The device quantizers against nv4pack::quantize / ink2ref::mx8::quantize on edge rows.
// 4. nv4::dense_gemv and nv4::dense_mt on three shapes against a host sum.
// A failure of 1 or 2 prints the first wrong coordinates; the fix is in the constants of nv4::Frag / ink2::Mx8Frag.
#include "../nvfp4.cuh"
#include "../ink2_moe.cuh"
#include "../../ref/ink2/ink2_cpu_ref.h"
#include <cstdio>
#include <functional>
#include <random>
#include <string>

using nv4::u32;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } \
    } while (0)

template <class T>
static T* to_dev(const std::vector<T>& v)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, (v.size() ? v.size() : 1) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
static T* dev_alloc(size_t n)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return p;
}
template <class T>
static std::vector<T> to_host(const T* p, size_t n)
{
    std::vector<T> v(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

static int g_fail = 0;
static void verdict(const char* name, bool ok)
{
    printf("%-58s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------------------------------------- 1, 2: the operand maps
// One warp per case: out [case][16 rows][8 columns] = record (16 x KG) @ activations (KG x 8).
template <class F>
__global__ void k_basis(const uint8_t* rec, const uint8_t* xq, const u32* xs, float* out)
{
    const int cs = blockIdx.x, lane = threadIdx.x, r = lane >> 2, c = lane & 3;
    const typename F::A a = F::load(rec + static_cast<size_t>(cs) * F::GR, lane);
    u32 b0, b1;
    F::loadB(xq + (static_cast<size_t>(cs) * 8 + r) * F::XREC, c, b0, b1);
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    F::mma(d, a, b0, b1, xs[cs * 8 + r]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) out[(static_cast<size_t>(cs) * 16 + r + 8 * (j >> 1)) * 8 + 2 * c + (j & 1)] = d[j];
}

struct Batch
{
    int KG = 0;
    std::vector<uint8_t> rec, xq;   // [case][GR], [case][8][32]
    std::vector<u32> xs;            // [case][8]
    std::vector<float> W, X;        // [case][16][KG], [case][8][KG]
    std::vector<int> mode, pa, pk;  // 0 = x one-hot (column pa, k pk), 1 = W one-hot (row pa, k pk), 2 = dense
    int n() const { return static_cast<int>(mode.size()); }
};

static void nv4_case(std::mt19937& g, int mode, int a, int k, Batch& b)
{
    uint8_t wc[16][32] = {}, ws[16][4], xc[8][32] = {}, xsb[8][4];
    const auto nib = [&] { return static_cast<uint8_t>((g() % 7u + 1u) | ((g() & 1u) << 3)); };
    const auto scales4 = [&](uint8_t* s) {   // four distinct finite nonzero E4M3 bytes
        for (int t = 0; t < 4; ++t)
        {
            bool again = true;
            while (again)
            {
                s[t] = static_cast<uint8_t>(g() % 0x7eu + 1u);
                again = false;
                for (int j = 0; j < t; ++j) again = again || s[j] == s[t];
            }
        }
    };
    const auto put = [](uint8_t* row, int kk, uint8_t n) { row[kk >> 1] = static_cast<uint8_t>(row[kk >> 1] | (n << (4 * (kk & 1)))); };
    for (int r = 0; r < 16; ++r) scales4(ws[r]);
    for (int c = 0; c < 8; ++c) scales4(xsb[c]);
    if (mode == 1) put(wc[a], k, nib());
    else
        for (int r = 0; r < 16; ++r)
            for (int kk = 0; kk < 64; ++kk) put(wc[r], kk, nib());
    if (mode == 0) put(xc[a], k, nib());
    else
        for (int c = 0; c < 8; ++c)
            for (int kk = 0; kk < 64; ++kk) put(xc[c], kk, nib());
    uint8_t rec[nv4pack::GR];
    nv4pack::pack_group(rec, &wc[0][0], &ws[0][0], 64, 0, 0);
    float W[16][64];
    nv4pack::decode_group(rec, 1.0f, W);
    b.rec.insert(b.rec.end(), rec, rec + nv4pack::GR);
    b.W.insert(b.W.end(), &W[0][0], &W[0][0] + 16 * 64);
    for (int c = 0; c < 8; ++c)
    {
        float X[64];
        nv4pack::dequantize(xc[c], xsb[c], 4, 1.0f, X);
        b.X.insert(b.X.end(), X, X + 64);
        uint8_t xr[32] = {};
        for (int kk = 0; kk < 64; ++kk)
        {
            const unsigned n = (xc[c][kk >> 1] >> (4 * (kk & 1))) & 15u;
            xr[nv4::Frag::xrec_byte(kk)] = static_cast<uint8_t>(xr[nv4::Frag::xrec_byte(kk)] | (n << (4 * (kk & 1))));
        }
        b.xq.insert(b.xq.end(), xr, xr + 32);
        u32 s;
        memcpy(&s, xsb[c], 4);
        b.xs.push_back(s);
    }
    b.mode.push_back(mode);
    b.pa.push_back(a);
    b.pk.push_back(k);
}

static void mx8_case(std::mt19937& g, int mode, int a, int k, Batch& b)
{
    uint8_t wc[16][32] = {}, ws[16], xc[8][32] = {}, xsb[8];
    const auto code = [&] {   // a finite nonzero E4M3 byte
        uint8_t v = 0;
        while ((v & 0x7fu) == 0u || (v & 0x7fu) == 0x7fu) v = static_cast<uint8_t>(g());
        return v;
    };
    for (int r = 0; r < 16; ++r) ws[r] = static_cast<uint8_t>(119u + g() % 17u);
    for (int c = 0; c < 8; ++c) xsb[c] = static_cast<uint8_t>(119u + g() % 17u);
    if (mode == 1) wc[a][k] = code();
    else
        for (int r = 0; r < 16; ++r)
            for (int kk = 0; kk < 32; ++kk) wc[r][kk] = code();
    if (mode == 0) xc[a][k] = code();
    else
        for (int c = 0; c < 8; ++c)
            for (int kk = 0; kk < 32; ++kk) xc[c][kk] = code();
    uint8_t rec[ink2ref::mx8::GR];
    ink2ref::mx8::pack_group(rec, &wc[0][0], ws, 32, 1, 0, 0, 0);
    float W[16][32];
    ink2ref::mx8::decode_group(rec, W);
    b.rec.insert(b.rec.end(), rec, rec + ink2ref::mx8::GR);
    b.W.insert(b.W.end(), &W[0][0], &W[0][0] + 16 * 32);
    for (int c = 0; c < 8; ++c)
    {
        float X[32];
        ink2ref::mx8::dequantize(xc[c], &xsb[c], 1, X);
        b.X.insert(b.X.end(), X, X + 32);
        uint8_t xr[32];
        for (int kk = 0; kk < 32; ++kk) xr[ink2::Mx8Frag::xrec_byte(kk)] = xc[c][kk];
        b.xq.insert(b.xq.end(), xr, xr + 32);
        b.xs.push_back(xsb[c]);
    }
    b.mode.push_back(mode);
    b.pa.push_back(a);
    b.pk.push_back(k);
}

template <class F>
static void run_basis(const char* name, void (*make)(std::mt19937&, int, int, int, Batch&))
{
    const int KG = F::KG;
    std::mt19937 g(1234u + static_cast<unsigned>(KG));
    Batch b;
    b.KG = KG;
    for (int c = 0; c < 8; ++c)
        for (int k = 0; k < KG; ++k) make(g, 0, c, k, b);
    for (int r = 0; r < 16; ++r)
        for (int k = 0; k < KG; ++k) make(g, 1, r, k, b);
    for (int i = 0; i < 64; ++i) make(g, 2, 0, 0, b);
    const int n = b.n();
    uint8_t* d_rec = to_dev(b.rec);
    uint8_t* d_xq = to_dev(b.xq);
    u32* d_xs = to_dev(b.xs);
    float* d_out = dev_alloc<float>(static_cast<size_t>(n) * 128);
    k_basis<F><<<n, 32>>>(d_rec, d_xq, d_xs, d_out);
    CK(cudaGetLastError());
    const std::vector<float> out = to_host(d_out, static_cast<size_t>(n) * 128);
    int bad[3] = {0, 0, 0}, shown = 0;
    for (int cs = 0; cs < n; ++cs)
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 8; ++c)
            {
                const float* w = &b.W[(static_cast<size_t>(cs) * 16 + r) * KG];
                const float* x = &b.X[(static_cast<size_t>(cs) * 8 + c) * KG];
                float e = 0.0f, mag = 0.0f;
                for (int k = 0; k < KG; ++k)
                {
                    e += w[k] * x[k];
                    mag += std::fabs(w[k] * x[k]);
                }
                const float got = out[(static_cast<size_t>(cs) * 16 + r) * 8 + c];
                const bool ok = b.mode[cs] == 2 ? std::fabs(got - e) <= 2e-5f * mag : got == e;
                if (ok) continue;
                ++bad[b.mode[cs]];
                if (shown++ < 12)
                    printf("  %s mismatch: mode %d (hot %s %d, k %d) out row %d col %d: got %.9g want %.9g\n", name, b.mode[cs],
                           b.mode[cs] == 0 ? "col" : "row", b.pa[cs], b.pk[cs], r, c, static_cast<double>(got), static_cast<double>(e));
            }
    printf("%s: %d cases; wrong outputs: activation one-hot %d, weight one-hot %d, dense %d\n", name, n, bad[0], bad[1], bad[2]);
    verdict((std::string(name) + " basis, bit exact").c_str(), bad[0] == 0 && bad[1] == 0);
    verdict((std::string(name) + " dense tiles").c_str(), bad[2] == 0);
    cudaFree(d_rec); cudaFree(d_xq); cudaFree(d_xs); cudaFree(d_out);
}

// ---------------------------------------------------------------------------------------------- 3: quantizers
static float bfv(float f) { return ink2ref::rbf(f); }

// Edge rows of K BF16-representable values.
static std::vector<std::vector<float>> edge_rows(int K)
{
    std::mt19937 g(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<float>> rows;
    const auto add = [&](const std::function<float(int)>& f) {
        std::vector<float> r(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k) r[k] = bfv(f(k));
        rows.push_back(r);
    };
    add([&](int) { return 0.0f; });
    add([&](int k) { return k == 37 ? 1e2f : 0.0f; });
    add([&](int k) { return k == K - 1 ? -3.0f : 0.0f; });
    add([&](int) { return nd(g); });
    add([&](int) { return 1e-3f * nd(g); });
    add([&](int) { return 300.0f * nd(g); });
    add([&](int) { return -std::fabs(nd(g)); });
    add([&](int k) { return (k % 16 == 5 ? 8.0f : 1.0f) * 0.1f * nd(g); });          // one outlier per block
    add([&](int k) { return (k % 16 == 11 ? 1.4f : 0.3f) * (g() & 1u ? 1.0f : -1.0f); });
    add([&](int) { return nv4::u2f((g() % 0x7fu + 1u) << 16); });                      // fp32 subnormals
    add([&](int k) { return k == 100 ? 1.0f : nv4::u2f((g() % 0x7fu + 1u) << 16); });
    add([&](int) { return 1e30f * nd(g); });
    add([&](int k) { return k < 16 ? 1e4f * nd(g) : 1e-4f * nd(g); });                 // block scales that round to 0
    add([&](int) { return g() & 1u ? 0.0f : -0.0f; });
    add([&](int k) { return (k >> 4) & 1 ? 6.0f : 4.0f; });
    add([&](int) { return 0.25f * static_cast<float>(static_cast<int>(g() % 49u) - 24); });   // rounding ties
    add([&](int k) { return std::ldexp(nd(g), static_cast<int>((static_cast<unsigned>(k >> 4) * 2654435761u >> 28)) - 8); });
    add([&](int k) { return std::ldexp(nd(g), static_cast<int>((static_cast<unsigned>(k >> 5) * 40503u >> 4) % 40u) - 20); });
    for (int i = 0; i < 6; ++i)
    {
        const float sc = std::ldexp(1.0f, i * 5 - 12);
        add([&](int) { return sc * nd(g) * (g() % 9u == 0u ? 5.0f : 1.0f); });
    }
    return rows;
}

// Device NVFP4 quantizer against the host one. cells = the 16-bit cells, f = their fp32 values.
static void nv4_quant_check(const char* name, const std::vector<uint16_t>& cells, const std::vector<std::vector<float>>& f, int K, bool in_bf16,
                            nv4::ScaleRule rule, bool fixed)
{
    const int rows = static_cast<int>(f.size()), NG = K / 64;
    std::vector<float> fo(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r)
    {
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(f[r][k]));
        fo[r] = amax / nv4pack::outer_denominator(rule) * (r & 1 ? 0.01f : 3.0f);   // saturating and coarse scales
    }
    uint16_t* d_x = to_dev(cells);
    float* d_fo = to_dev(fo);
    uint8_t* d_xq = dev_alloc<uint8_t>(static_cast<size_t>(rows) * NG * 32);
    u32* d_xs = dev_alloc<u32>(static_cast<size_t>(rows) * NG);
    float* d_outer = dev_alloc<float>(static_cast<size_t>(rows));
    nv4::quant(d_x, rows, K, in_bf16, rule, fixed ? d_fo : nullptr, nullptr, 0, d_xq, d_xs, d_outer, 0);
    CK(cudaGetLastError());
    const std::vector<uint8_t> xq = to_host(d_xq, static_cast<size_t>(rows) * NG * 32);
    const std::vector<u32> xs = to_host(d_xs, static_cast<size_t>(rows) * NG);
    const std::vector<float> outer = to_host(d_outer, static_cast<size_t>(rows));
    int bad_outer = 0, bad_blocks = 0, tie_blocks = 0, shown = 0;
    double min_cos = 1.0;
    std::vector<uint8_t> codes(static_cast<size_t>(K) / 2), scales(static_cast<size_t>(K) / 16), dc(codes.size()), ds(scales.size());
    std::vector<float> hq(static_cast<size_t>(K)), dq(static_cast<size_t>(K));
    for (int r = 0; r < rows; ++r)
    {
        const float ho = nv4pack::quantize(f[r].data(), 1, K, rule, fixed ? &fo[r] : nullptr, codes.data(), scales.data());
        if (nv4::f2u(ho) != nv4::f2u(outer[r])) ++bad_outer;
        std::fill(dc.begin(), dc.end(), uint8_t(0));
        for (int k = 0; k < K; ++k)
        {
            const unsigned n = (xq[(static_cast<size_t>(r) * NG + k / 64) * 32 + nv4::Frag::xrec_byte(k & 63)] >> (4 * (k & 1))) & 15u;
            dc[k >> 1] = static_cast<uint8_t>(dc[k >> 1] | (n << (4 * (k & 1))));
        }
        for (int b = 0; b < K / 16; ++b) ds[b] = static_cast<uint8_t>(xs[static_cast<size_t>(r) * NG + b / 4] >> (8 * (b & 3)));
        for (int b = 0; b < K / 16; ++b)
        {
            if (ds[b] == scales[b] && !memcmp(&dc[b * 8], &codes[b * 8], 8)) continue;
            bool tie = false;
            if (rule == nv4::ScaleRule::FourOverSixMae)
            {
                const float* blk = &f[r][b * 16];
                float b_max = 0.0f;
                for (int j = 0; j < 16; ++j) b_max = std::fmax(b_max, std::fabs(blk[j]));
                uint8_t s, q[16];
                tie = nv4pack::block_candidate(blk, b_max, ho, 6.0f, s, q) == nv4pack::block_candidate(blk, b_max, ho, 4.0f, s, q);
            }
            if (tie) { ++tie_blocks; continue; }
            ++bad_blocks;
            if (shown++ < 6) printf("  %s: row %d block %d: scale %02x want %02x\n", name, r, b, ds[b], scales[b]);
        }
        nv4pack::dequantize(codes.data(), scales.data(), static_cast<size_t>(K) / 16, ho, hq.data());
        nv4pack::dequantize(dc.data(), ds.data(), static_cast<size_t>(K) / 16, outer[r], dq.data());
        double ab = 0, aa = 0, bb = 0;
        for (int k = 0; k < K; ++k)
        {
            ab += static_cast<double>(hq[k]) * dq[k];
            aa += static_cast<double>(hq[k]) * hq[k];
            bb += static_cast<double>(dq[k]) * dq[k];
        }
        const double cs = aa == 0.0 && bb == 0.0 ? 1.0 : ab / std::sqrt(aa * bb + 1e-300);
        if (!(cs >= min_cos)) min_cos = cs;
    }
    printf("%s: %d rows x %d; outer mismatches %d, wrong blocks %d (+ %d on exact error ties), min cosine %.9f\n", name, rows, K, bad_outer,
           bad_blocks, tie_blocks, min_cos);
    verdict(name, bad_outer == 0 && bad_blocks == 0 && min_cos > 0.999999);
    cudaFree(d_x); cudaFree(d_fo); cudaFree(d_xq); cudaFree(d_xs); cudaFree(d_outer);
}

static void mx8_quant_check(const std::vector<uint16_t>& cells, const std::vector<std::vector<float>>& f, int K)
{
    const int rows = static_cast<int>(f.size()), NG = K / 32;
    uint16_t* d_x = to_dev(cells);
    uint8_t* d_xq = dev_alloc<uint8_t>(static_cast<size_t>(rows) * NG * 32);
    u32* d_xs = dev_alloc<u32>(static_cast<size_t>(rows) * NG);
    ink2::mx8_quant(d_x, rows, K, true, nullptr, 0, d_xq, d_xs, 0);
    CK(cudaGetLastError());
    const std::vector<uint8_t> xq = to_host(d_xq, static_cast<size_t>(rows) * NG * 32);
    const std::vector<u32> xs = to_host(d_xs, static_cast<size_t>(rows) * NG);
    int bad_scale = 0, bad_code = 0, shown = 0;
    std::vector<uint8_t> codes(static_cast<size_t>(K)), scales(static_cast<size_t>(NG));
    for (int r = 0; r < rows; ++r)
    {
        ink2ref::mx8::quantize(f[r].data(), static_cast<size_t>(NG), codes.data(), scales.data());
        for (int b = 0; b < NG; ++b)
        {
            const size_t blk = static_cast<size_t>(r) * NG + b;
            if (xs[blk] != scales[b])
            {
                ++bad_scale;
                if (shown++ < 6) printf("  mxfp8 quant: row %d block %d: scale %u want %u\n", r, b, xs[blk], scales[b]);
            }
            for (int k = 0; k < 32; ++k) bad_code += xq[blk * 32 + ink2::Mx8Frag::xrec_byte(k)] != codes[b * 32 + k];
        }
    }
    printf("mxfp8 quant: %d rows x %d; wrong scales %d, wrong codes %d\n", rows, K, bad_scale, bad_code);
    verdict("device MXFP8 quantizer, exact", bad_scale == 0 && bad_code == 0);
    cudaFree(d_x); cudaFree(d_xq); cudaFree(d_xs);
}

static void quant_tests()
{
    const int K = 256;
    const std::vector<std::vector<float>> rows = edge_rows(K);
    std::vector<uint16_t> cells;
    for (const auto& r : rows)
        for (float v : r) cells.push_back(ink2ref::f32_to_bf16(v));
    nv4_quant_check("device NVFP4 quantizer, Static6", cells, rows, K, true, nv4::ScaleRule::Static6, false);
    nv4_quant_check("device NVFP4 quantizer, FourOverSixMae", cells, rows, K, true, nv4::ScaleRule::FourOverSixMae, false);
    nv4_quant_check("device NVFP4 quantizer, Static6, fixed outer", cells, rows, K, true, nv4::ScaleRule::Static6, true);
    nv4_quant_check("device NVFP4 quantizer, FourOverSixMae, fixed outer", cells, rows, K, true, nv4::ScaleRule::FourOverSixMae, true);
    mx8_quant_check(cells, rows, K);
    // FP16 cells: random finite bit patterns
    std::mt19937 g(5);
    std::vector<uint16_t> hc;
    std::vector<std::vector<float>> hf;
    for (int r = 0; r < 8; ++r)
    {
        std::vector<float> row(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k)
        {
            u32 bits = g() & 0xffffu;
            if (((bits >> 10) & 31u) == 31u) bits &= 0xbfffu;
            if (r >= 4) bits = (bits & 0x83ffu) | ((13u + g() % 5u) << 10);   // a narrow range of exponents
            hc.push_back(static_cast<uint16_t>(bits));
            row[k] = nv4::cell16(bits, 0);
        }
        hf.push_back(row);
    }
    nv4_quant_check("device NVFP4 quantizer, FP16 cells, FourOverSixMae", hc, hf, K, false, nv4::ScaleRule::FourOverSixMae, false);
}

// ---------------------------------------------------------------------------------------------- 4: dense
static double cosine(const float* a, const float* b, size_t n, double* rel)
{
    double ab = 0, aa = 0, bb = 0, dd = 0;
    for (size_t i = 0; i < n; ++i)
    {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        dd += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    }
    if (rel) *rel = std::sqrt(dd / (bb + 1e-300));
    return ab / std::sqrt(aa * bb + 1e-300);
}

static void dense_test(int N, int K)
{
    std::mt19937 g(static_cast<unsigned>(N * 31 + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int MG = 8, MT = 40, M = MT;
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (float& v : w) v = 0.05f * nd(g);
    std::vector<uint8_t> wc(w.size() / 2), ws(w.size() / 16);
    const float w_outer = nv4pack::quantize(w.data(), static_cast<size_t>(N), K, nv4pack::ScaleRule::Static6, nullptr, wc.data(), ws.data());
    const std::vector<uint8_t> rec = nv4pack::pack_matrix(wc.data(), ws.data(), N, K);
    std::vector<float> wd(w.size());
    nv4pack::dequantize(wc.data(), ws.data(), w.size() / 16, w_outer, wd.data());
    std::vector<uint16_t> xb(static_cast<size_t>(M) * K);
    std::vector<float> xd(static_cast<size_t>(M) * K), row(static_cast<size_t>(K));
    std::vector<uint8_t> xc(static_cast<size_t>(K) / 2), xsc(static_cast<size_t>(K) / 16);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            xb[static_cast<size_t>(m) * K + k] = ink2ref::f32_to_bf16((1.0f + 0.2f * static_cast<float>(m)) * nd(g));
            row[k] = ink2ref::bf16_to_f32(xb[static_cast<size_t>(m) * K + k]);
        }
        const float o = nv4pack::quantize_row(row.data(), K, nv4pack::ScaleRule::FourOverSixMae, xc.data(), xsc.data());
        nv4pack::dequantize(xc.data(), xsc.data(), static_cast<size_t>(K) / 16, o, &xd[static_cast<size_t>(m) * K]);
    }
    std::vector<float> ref(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) ref[static_cast<size_t>(m) * N + n] = ink2ref::dot(&wd[static_cast<size_t>(n) * K], &xd[static_cast<size_t>(m) * K], K);

    const int NG = K / 64;
    uint8_t* d_w = to_dev(rec);
    uint16_t* d_x = to_dev(xb);
    uint8_t* d_xq = dev_alloc<uint8_t>(static_cast<size_t>(M) * NG * 32);
    u32* d_xs = dev_alloc<u32>(static_cast<size_t>(M) * NG);
    float* d_outer = dev_alloc<float>(static_cast<size_t>(M));
    float* d_y = dev_alloc<float>(static_cast<size_t>(M) * N);
    uint16_t* d_y16 = dev_alloc<uint16_t>(static_cast<size_t>(M) * N);
    nv4::quant(d_x, M, K, true, nv4::ScaleRule::FourOverSixMae, nullptr, nullptr, 0, d_xq, d_xs, d_outer, 0);
    char name[96];
    bool ok = true;
    double rel = 0;
    // multi-tile, fp32
    nv4::dense_mt(d_w, N, K, d_xq, d_xs, d_outer, w_outer, MT, d_y, nv4::O_F32, 0);
    CK(cudaGetLastError());
    const std::vector<float> ymt = to_host(d_y, static_cast<size_t>(MT) * N);
    const double c_mt = cosine(ymt.data(), ref.data(), static_cast<size_t>(MT) * N, &rel);
    ok = ok && c_mt > 0.999999 && rel < 1e-4;
    printf("dense N %d K %d: dense_mt M %d cosine %.9f rel L2 %.3g\n", N, K, MT, c_mt, rel);
    // gemv, fp32: the same sums as the first 8 tokens of the multi-tile kernel
    CK(cudaMemset(d_y, 0, static_cast<size_t>(M) * N * sizeof(float)));
    nv4::dense_gemv(d_w, N, K, d_xq, d_xs, d_outer, w_outer, MG, d_y, nv4::O_F32, 0);
    CK(cudaGetLastError());
    const std::vector<float> ygv = to_host(d_y, static_cast<size_t>(MG) * N);
    const double c_gv = cosine(ygv.data(), ref.data(), static_cast<size_t>(MG) * N, &rel);
    const bool same = !memcmp(ygv.data(), ymt.data(), static_cast<size_t>(MG) * N * sizeof(float));
    ok = ok && c_gv > 0.999999 && rel < 1e-4 && same;
    printf("dense N %d K %d: dense_gemv M %d cosine %.9f rel L2 %.3g, equal to dense_mt: %s\n", N, K, MG, c_gv, rel, same ? "yes" : "NO");
    // BF16 and FP16 outputs = the fp32 output rounded once
    nv4::dense_mt(d_w, N, K, d_xq, d_xs, d_outer, w_outer, MT, d_y16, nv4::O_BF16, 0);
    const std::vector<uint16_t> yb = to_host(d_y16, static_cast<size_t>(MT) * N);
    nv4::dense_mt(d_w, N, K, d_xq, d_xs, d_outer, w_outer, MT, d_y16, nv4::O_F16, 0);
    CK(cudaGetLastError());
    const std::vector<uint16_t> yh = to_host(d_y16, static_cast<size_t>(MT) * N);
    size_t bad_b = 0, bad_h = 0;
    for (size_t i = 0; i < ymt.size(); ++i)
    {
        bad_b += yb[i] != ink2ref::f32_to_bf16(ymt[i]);
        const float h = nv4::cell16(yh[i], 0);
        bad_h += !(std::fabs(h - ymt[i]) <= std::fabs(ymt[i]) * (1.0f / 1024.0f) + 1e-7f);
    }
    ok = ok && bad_b == 0 && bad_h == 0;
    printf("dense N %d K %d: BF16 output cells not the rounded fp32: %zu, FP16 cells off: %zu\n", N, K, bad_b, bad_h);
    snprintf(name, sizeof name, "nv4::dense_gemv / dense_mt  N %d K %d", N, K);
    verdict(name, ok);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_xs); cudaFree(d_outer); cudaFree(d_y); cudaFree(d_y16);
}

int main()
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    run_basis<nv4::Frag>("NVFP4 nv4g (mxf4nvf4 4X m16n8k64)", nv4_case);
    run_basis<ink2::Mx8Frag>("MXFP8 528 B (mxf8f6f4 1X m16n8k32)", mx8_case);
    quant_tests();
    dense_test(64, 128);
    dense_test(768, 4608);
    dense_test(4608, 384);
    printf(g_fail ? "nvfp4_mma_probe: FAIL (%d)\n" : "nvfp4_mma_probe: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
