// Shared expert (MXFP8) + MoE combine (ds_shared.cuh): layout check, correctness vs a CPU double reference, and a bench against the
// DRAM byte ceiling.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o ds_shared_test ds_shared_test.cu
//   ./ds_shared_test layout        host only: the MXFP8 record packer (the loader's records::mxfp8_record layout) decodes back
//   ./ds_shared_test check         GPU: shared fc1/fc2 vs the CPU, combine bit-exact vs a fp32 emulation (fp16/bf16/fp32 out),
//                                  and a whole MoE layer (routed fc1/fc2 + shared + combine) vs a CPU double reference
//   ./ds_shared_test chain         GPU: router -> shared -> routed (n_slots_dev gate) -> bf16 combine, the launch sequence of ds41_moe_ops.cuh, vs a CPU double reference
//   ./ds_shared_test multitile     GPU: M 1..32 through the multi-tile kernels: device plan == plan_ref, M <= 8 == one-tile path, token m of M 9..32 == the token alone, CPU reference, skewed routes
//   ./ds_shared_test pdl [rounds]  GPU: 8 back-to-back layers through the PDL chain (stream and CUDA graph; scratch poisoned) == the plain chain, bitwise, M 1..32, 3 bias modes
//   ./ds_shared_test fixture IN LAYER OUT  GPU: one real layer (checkpoint weights, oracle ffn_in) -> router_idx/router_w/moe_routed/shared_out/ffn_out dumps for ds41_compare.py
//   ./ds_shared_test bench [sets]  GPU: shared fc1 / fc2 / combine timing, cold weights, vs a plain stream read of the same bytes
//   ./ds_shared_test mtbench [ROUTES_DIR]  GPU: the MoE layer at M = 9, 16, 24, 32 (multi-tile) and M = 1, 8 (one tile), per stage and whole, cold weights; ROUTES_DIR = reference-model L<n>.router_idx.bin
// Geometry: DS-V4.1-Flash-1017 at TP4 (I = 2304 / 4 = 576). -DSH_I=384 for TP6, 768 for TP3, 1152 for TP2 (any multiple of 64).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <type_traits>
#include <string>
#include <cuda_runtime.h>
#include "../ds_shared.cuh"
#include "../mxfp4_pack.h"

#ifndef SH_I
#define SH_I 576
#endif
constexpr int K1 = 5120, I = SH_I, NOUT = 5120, TOPK = 6;
constexpr size_t EBG = static_cast<size_t>(I / 16) * (K1 / 32) * dsh::GR;     // bytes of the gate (and of the up) stream
constexpr size_t EBD = static_cast<size_t>(NOUT / 16) * (I / 32) * dsh::GR;   // bytes of the down stream

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

// ------------------------------------------------------------------------------------------ MXFP8 host side
struct M8 { std::vector<uint8_t> codes, sc; int R, K; };    // codes [R][K] E4M3, sc [R][K/32] E8M0
static M8 rand_m8(std::mt19937_64& g, int R, int K, int e_lo, int e_hi, int big_every = 0, int big_lo = 0, int big_hi = 0)
{
    M8 m; m.R = R; m.K = K;
    m.codes.resize(static_cast<size_t>(R) * K);
    m.sc.resize(static_cast<size_t>(R) * K / 32);
    for (auto& b : m.codes)    // E4M3 without NaN: exponent field 3..10 (2^-4 .. 2^3), 1 in 16 a subnormal / zero
    {
        const int e = (g() % 16 == 0) ? 0 : 3 + static_cast<int>(g() % 8);
        b = static_cast<uint8_t>(((g() & 1) << 7) | (e << 3) | (g() & 7));
    }
    for (int r = 0; r < R; ++r)
    {
        const bool big = big_every && r % big_every == 0;
        for (int j = 0; j < K / 32; ++j)
            m.sc[static_cast<size_t>(r) * (K / 32) + j] = static_cast<uint8_t>(big ? big_lo + g() % (big_hi - big_lo + 1) : e_lo + g() % (e_hi - e_lo + 1));
    }
    return m;
}
static double e4m3d(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}
static std::vector<float> dequant8(const M8& m)
{
    std::vector<float> w(static_cast<size_t>(m.R) * m.K);
    for (int r = 0; r < m.R; ++r)
        for (int k = 0; k < m.K; ++k)
            w[static_cast<size_t>(r) * m.K + k] = static_cast<float>(e4m3d(m.codes[static_cast<size_t>(r) * m.K + k]) * std::ldexp(1.0, m.sc[static_cast<size_t>(r) * (m.K / 32) + k / 32] - 127));
    return w;
}
// [R/16][K/32][528]: per 32-k group two 16x16 tiles in lane order (lane l = (r = l/4, c = l%4) holds (r,2c) (r,2c+1) (r+8,2c) (r+8,2c+1)
// (r,2c+8) (r,2c+9) (r+8,2c+8) (r+8,2c+9)), then the 8 scale pairs {s[i], s[i+8]}. Same bytes as loader-rs records::mxfp8_record.
static void lane_elem(int l, int i, int& r, int& c)
{
    const int rr = l / 4, cc = l % 4;
    static const int dr[8] = {0, 0, 8, 8, 0, 0, 8, 8}, dc[8] = {0, 1, 0, 1, 8, 9, 8, 9};
    r = rr + dr[i]; c = 2 * cc + dc[i];
}
static std::vector<uint8_t> pack8(const M8& m)
{
    const int ng = m.K / 32;
    std::vector<uint8_t> out(static_cast<size_t>(m.R / 16) * ng * dsh::GR, 0);
    for (int t = 0; t < m.R / 16; ++t)
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* rec = out.data() + (static_cast<size_t>(t) * ng + g) * dsh::GR;
            for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 32; ++l)
                    for (int i = 0; i < 8; ++i)
                    {
                        int r, c; lane_elem(l, i, r, c);
                        rec[h * 256 + l * 8 + i] = m.codes[static_cast<size_t>(16 * t + r) * m.K + 32 * g + 16 * h + c];
                    }
            for (int i = 0; i < 8; ++i)
            {
                rec[512 + 2 * i] = m.sc[static_cast<size_t>(16 * t + i) * ng + g];
                rec[512 + 2 * i + 1] = m.sc[static_cast<size_t>(16 * t + i + 8) * ng + g];
            }
        }
    return out;
}

static int layout_check()
{
    std::mt19937_64 g(1);
    int bad = 0;
    for (int R : {16, 48}) for (int K : {32, 96, 576})
    {
        const M8 m = rand_m8(g, R, K, 100, 140);
        const std::vector<uint8_t> s = pack8(m);
        // decode the way the kernel reads a record: lane l's 8 bytes of tile h, scales at 512 + 2 (l / 4) {row r, row r + 8}
        for (int t = 0; t < R / 16; ++t) for (int gi = 0; gi < K / 32; ++gi)
        {
            const uint8_t* rec = s.data() + (static_cast<size_t>(t) * (K / 32) + gi) * dsh::GR;
            for (int l = 0; l < 32; ++l) for (int h = 0; h < 2; ++h)
            {
                const int r = l / 4, c = l % 4;
                const uint8_t* w = rec + h * 256 + l * 8;
                const int k0 = gi * 32 + h * 16;
                bad += w[0] != m.codes[static_cast<size_t>(16 * t + r) * K + k0 + 2 * c] || w[1] != m.codes[static_cast<size_t>(16 * t + r) * K + k0 + 2 * c + 1];
                bad += w[2] != m.codes[static_cast<size_t>(16 * t + r + 8) * K + k0 + 2 * c] || w[3] != m.codes[static_cast<size_t>(16 * t + r + 8) * K + k0 + 2 * c + 1];
                bad += w[4] != m.codes[static_cast<size_t>(16 * t + r) * K + k0 + 2 * c + 8] || w[5] != m.codes[static_cast<size_t>(16 * t + r) * K + k0 + 2 * c + 9];
                bad += w[6] != m.codes[static_cast<size_t>(16 * t + r + 8) * K + k0 + 2 * c + 8] || w[7] != m.codes[static_cast<size_t>(16 * t + r + 8) * K + k0 + 2 * c + 9];
            }
            for (int l = 0; l < 32; ++l)
            {
                const int r = l / 4;
                bad += rec[512 + 2 * r] != m.sc[static_cast<size_t>(16 * t + r) * (K / 32) + gi] || rec[512 + 2 * r + 1] != m.sc[static_cast<size_t>(16 * t + r + 8) * (K / 32) + gi];
            }
        }
    }
    printf("layout: mxfp8 pack -> kernel-order decode mismatches %d   record %d B = %.4f bits/weight\n", bad, dsh::GR, dsh::GR * 8.0 / (16 * 32));
    return bad != 0;
}

// ------------------------------------------------------------------------------------------ MXFP4 host side (routed experts, as mxfp4_gemv_test.cu)
struct Matrix { std::vector<uint8_t> packed, e8; int R, K; };
static Matrix rand_matrix(std::mt19937_64& g, int R, int K, int e_lo, int e_hi)
{
    Matrix m; m.R = R; m.K = K;
    m.packed.resize(static_cast<size_t>(R) * K / 2);
    m.e8.resize(static_cast<size_t>(R) * K / 32);
    for (auto& b : m.packed) b = static_cast<uint8_t>(g());
    for (auto& b : m.e8) b = static_cast<uint8_t>(e_lo + g() % (e_hi - e_lo + 1));
    return m;
}
static std::vector<float> dequant(const Matrix& m)
{
    std::vector<float> w(static_cast<size_t>(m.R) * m.K);
    for (int r = 0; r < m.R; ++r)
        for (int k = 0; k < m.K; ++k)
        {
            const uint8_t b = m.packed[static_cast<size_t>(r) * m.K / 2 + k / 2];
            w[static_cast<size_t>(r) * m.K + k] = mx4pack::e2m1((k & 1) ? (b >> 4) : (b & 15)) * static_cast<float>(mx4pack::e8m0(m.e8[static_cast<size_t>(r) * m.K / 32 + k / 32]));
        }
    return w;
}

// ------------------------------------------------------------------------------------------ CPU reference of one expert on one token
struct Ref { std::vector<double> act, actS; long hits = 0; };     // act [I], the abs-sum of each pre-activation's products (tolerance scale), rows past the swiglu limit
static Ref ref_fc1(const std::vector<float>& wg, const std::vector<float>& wu, const float* x, double limit)
{
    Ref r; r.act.resize(I); r.actS.resize(I);
    for (int i = 0; i < I; ++i)
    {
        double gt = 0, ut = 0, S = 0;
        for (int k = 0; k < K1; ++k)
        {
            const double tg = static_cast<double>(wg[static_cast<size_t>(i) * K1 + k]) * x[k], tu = static_cast<double>(wu[static_cast<size_t>(i) * K1 + k]) * x[k];
            gt += tg; ut += tu; S += std::fabs(tg) + std::fabs(tu);
        }
        const double gc = std::min(gt, limit), uc = std::min(std::max(ut, -limit), limit);
        r.hits += gt > limit || ut > limit || ut < -limit;
        r.act[i] = gc / (1.0 + std::exp(-gc)) * uc;
        r.actS[i] = S;
    }
    return r;
}
static void ref_fc2(const std::vector<float>& wd, const std::vector<double>& act, std::vector<double>& y, std::vector<double>& S)
{
    y.assign(NOUT, 0.0); S.assign(NOUT, 0.0);
    for (int o = 0; o < NOUT; ++o)
    {
        double a = 0, s = 0;
        for (int i = 0; i < I; ++i) { const double t = static_cast<double>(wd[static_cast<size_t>(o) * I + i]) * act[i]; a += t; s += std::fabs(t); }
        y[o] = a; S[o] = s;
    }
}

// ------------------------------------------------------------------------------------------ combine, bit-exact
static float bf16_rn(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }

template <int TK> static int check_combine()
{
    std::mt19937_64 g(11);
    constexpr int NS = 30;
    int fails = 0;
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<__half> yp(static_cast<size_t>(NS) * 8 * NOUT);
    for (auto& v : yp) v = __float2half_rn(nd(g));
    std::vector<float> sh(8 * NOUT);
    for (auto& v : sh) v = nd(g);
    __half* dyp; float* dsh_; void* dout; dsr::PlanT<TK>* dplan;
    CK(cudaMalloc(&dyp, yp.size() * 2)); CK(cudaMalloc(&dsh_, sh.size() * 4)); CK(cudaMalloc(&dout, 8 * NOUT * 4)); CK(cudaMalloc(&dplan, sizeof(dsr::PlanT<TK>)));
    CK(cudaMemcpy(dyp, yp.data(), yp.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsh_, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice));
    for (int M : {1, 3, 8})
        for (int variant = 0; variant < 2; ++variant)     // 1: token 1 has fewer routes (tok_slot -1 beyond 2) and token 0 none
        {
            dsr::PlanT<TK> P; memset(&P, 0, sizeof P);
            for (int m = 0; m < dsr::MMAX; ++m) for (int k = 0; k < 8; ++k) { P.tok_slot[m][k] = -1; P.tok_col[m][k] = 0; }
            for (int m = 0; m < M; ++m)
            {
                const int nr = variant == 0 ? TK : (m == 0 ? 0 : m == 1 ? 2 : TK);
                for (int k = 0; k < nr; ++k) { P.tok_slot[m][k] = static_cast<int>(g() % NS); P.tok_col[m][k] = m; }
            }
            CK(cudaMemcpy(dplan, &P, sizeof P, cudaMemcpyHostToDevice));
            for (int odt = 0; odt < 3; ++odt)
                for (int use_sh = 0; use_sh < 2; ++use_sh)
                {
                    CK(cudaMemset(dout, 0xff, 8 * NOUT * 4));
                    dsh::CombineArgsT<TK> A{dplan, dyp, use_sh ? dsh_ : nullptr, dout, M};
                    const int grid = M * NOUT / (8 * 128);
                    if (odt == 0) dsh::k_combine<NOUT, 128, gv::O_F16, TK><<<grid, 128>>>(A);
                    else if (odt == 1) dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<grid, 128>>>(A);
                    else dsh::k_combine<NOUT, 128, gv::O_F32, TK><<<grid, 128>>>(A);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    std::vector<uint8_t> got(static_cast<size_t>(M) * NOUT * (odt == 2 ? 4 : 2));
                    CK(cudaMemcpy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost));
                    long bad = 0;
                    for (int m = 0; m < M; ++m) for (int n = 0; n < NOUT; ++n)
                    {
                        float a = 0.f;     // the kernel's fp32 sequence: routes in order 0..TK-1, then the shared term
                        for (int k = 0; k < TK; ++k)
                            if (P.tok_slot[m][k] >= 0) a += __half2float(yp[(static_cast<size_t>(P.tok_slot[m][k]) * 8 + P.tok_col[m][k]) * NOUT + n]);
                        if (use_sh) a += sh[static_cast<size_t>(m) * NOUT + n];
                        const size_t i = static_cast<size_t>(m) * NOUT + n;
                        if (odt == 0) { __half h; memcpy(&h, &got[i * 2], 2); bad += __half2float(h) != __half2float(__float2half_rn(a)); }
                        else if (odt == 1) { uint16_t b; memcpy(&b, &got[i * 2], 2); const uint32_t u = static_cast<uint32_t>(b) << 16; float f; memcpy(&f, &u, 4); bad += f != bf16_rn(a); }
                        else { float f; memcpy(&f, &got[i * 4], 4); bad += f != a; }
                    }
                    printf("combine TK=%d M=%d routes=%s out=%s shared=%s: %ld mismatches of %d -> %s\n", TK, M, variant ? "ragged" : "TK", odt == 0 ? "f16" : odt == 1 ? "bf16" : "f32",
                           use_sh ? "yes" : "no", bad, M * NOUT, bad ? "FAIL" : "OK");
                    fails += bad != 0;
                }
        }
    cudaFree(dyp); cudaFree(dsh_); cudaFree(dout); cudaFree(dplan);
    return fails;
}

// ------------------------------------------------------------------------------------------ shared expert + whole layer
template <int TK> static int check_layer()
{
    constexpr int NE = 8;
    std::mt19937_64 g(7);
    struct Expert { Matrix gate, up, down; };
    std::vector<Expert> ex(NE);
    for (auto& e : ex) { e.gate = rand_matrix(g, I, K1, 119, 124); e.up = rand_matrix(g, I, K1, 119, 124); e.down = rand_matrix(g, NOUT, I, 117, 122); }
    // shared expert: every 7th gate / up row has scales ~2^1..2^3 so its pre-activations pass the swiglu limit (clamp exercised)
    const M8 sg = rand_m8(g, I, K1, 118, 123, 7, 126, 130), su = rand_m8(g, I, K1, 118, 123, 5, 126, 130), sd = rand_m8(g, NOUT, I, 117, 122);
    std::vector<uint8_t> h1, h2;
    for (auto& e : ex)
    {
        auto a = mx4pack::pack_fc1(e.gate.packed.data(), e.gate.e8.data(), e.up.packed.data(), e.up.e8.data(), I, K1);
        auto b = mx4pack::pack_matrix(e.down.packed.data(), e.down.e8.data(), NOUT, I);
        h1.insert(h1.end(), a.begin(), a.end());
        h2.insert(h2.end(), b.begin(), b.end());
    }
    const std::vector<uint8_t> pg = pack8(sg), pu = pack8(su), pd = pack8(sd);
    uint8_t *d1, *d2, *dg, *du, *dd;
    CK(cudaMalloc(&d1, h1.size())); CK(cudaMalloc(&d2, h2.size())); CK(cudaMalloc(&dg, pg.size())); CK(cudaMalloc(&du, pu.size())); CK(cudaMalloc(&dd, pd.size()));
    CK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, pg.data(), pg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, pu.data(), pu.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, pd.data(), pd.size(), cudaMemcpyHostToDevice));
    std::vector<std::vector<float>> deq_g(NE), deq_u(NE), deq_d(NE);
    for (int e = 0; e < NE; ++e) { deq_g[e] = dequant(ex[e].gate); deq_u[e] = dequant(ex[e].up); deq_d[e] = dequant(ex[e].down); }
    const std::vector<float> sgw = dequant8(sg), suw = dequant8(su), sdw = dequant8(sd);

    int fails = 0;
    for (int M : {1, 5, 8})
    {
        std::vector<std::vector<int>> route(M);
        std::vector<std::vector<float>> rw(M, std::vector<float>(TK));
        for (int m = 0; m < M; ++m)
        {
            std::vector<int> p(NE); for (int i = 0; i < NE; ++i) p[i] = i;
            std::shuffle(p.begin(), p.end(), g);
            route[m].assign(p.begin(), p.begin() + TK);
            for (auto& v : rw[m]) v = 0.1f + (g() % 1000) / 1000.f * 1.5f;
        }
        std::vector<int> se, sm; std::vector<float> sw;
        dsr::PlanT<TK> P; memset(&P, 0, sizeof P);
        for (int m = 0; m < dsr::MMAX; ++m) for (int k = 0; k < 8; ++k) { P.tok_slot[m][k] = -1; P.tok_col[m][k] = 0; }
        for (int e = 0; e < NE; ++e)
        {
            int mask = 0; float w[8] = {};
            const int slot = static_cast<int>(se.size());
            for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) if (route[m][k] == e) { mask |= 1 << m; w[m] = rw[m][k]; P.tok_slot[m][k] = slot; P.tok_col[m][k] = m; }
            if (!mask) continue;
            se.push_back(e); sm.push_back(mask); sw.insert(sw.end(), w, w + 8);
        }
        const int NS = static_cast<int>(se.size());
        std::normal_distribution<float> nd(0.f, 0.5f);
        std::vector<__nv_bfloat16> xb(static_cast<size_t>(M) * K1);
        std::vector<float> xr(static_cast<size_t>(M) * K1);
        for (size_t i = 0; i < xb.size(); ++i) { xb[i] = __float2bfloat16_rn(nd(g)); xr[i] = __half2float(__float2half_rn(__bfloat162float(xb[i]))); }

        __nv_bfloat16* dx; u32* dxf; int *dse, *dsm; float* dsw; __half *dact, *dy, *dsact; float* dsout; dsr::PlanT<TK>* dplan; void* dfin;
        CK(cudaMalloc(&dx, xb.size() * 2)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4));
        CK(cudaMalloc(&dse, NS * 4)); CK(cudaMalloc(&dsm, NS * 4)); CK(cudaMalloc(&dsw, NS * 8 * 4));
        CK(cudaMalloc(&dact, static_cast<size_t>(NS) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(NS) * 8 * NOUT * 2));
        CK(cudaMalloc(&dsact, 8 * I * 2)); CK(cudaMalloc(&dsout, 8 * NOUT * 4)); CK(cudaMalloc(&dplan, sizeof P)); CK(cudaMalloc(&dfin, 8 * NOUT * 4));
        CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dse, se.data(), NS * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsm, sm.data(), NS * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsw, sw.data(), NS * 8 * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dplan, &P, sizeof P, cudaMemcpyHostToDevice));
        CK(cudaMemset(dsact, 0xff, 8 * I * 2)); CK(cudaMemset(dsout, 0xff, 8 * NOUT * 4));    // NaN sentinels
        mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, M, K1, dxf);
        mx4::Args RA{d1, d2, dxf, dse, dsm, dsw, nullptr, NS, 10.f, dact, dy};
        const int live = (1 << M) - 1;
        dsh::Args SA{dg, du, dd, dxf, live, 10.f, dsact, dsout};
        constexpr int KS = 8, WARPS = 8;
        std::vector<__half> a0(static_cast<size_t>(8) * I);
        std::vector<float> o0(static_cast<size_t>(8) * NOUT);
        for (int rep = 0; rep < 2; ++rep)
        {
            mx4::k_fc1<K1, I, KS, 1, 2, false><<<NS * (I / 16), KS * 32>>>(RA);
            mx4::k_fc2<I, NOUT, WARPS, 2, false><<<NS * (NOUT / 16 / WARPS), WARPS * 32>>>(RA);
            dsh::k_sh_fc1<K1, I, KS, 1, 2><<<I / 16, KS * 32>>>(SA);
            dsh::k_sh_fc2<I, NOUT, WARPS, 3><<<NOUT / 16 / WARPS, WARPS * 32>>>(SA);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            if (rep == 0)
            {
                CK(cudaMemcpy(a0.data(), dsact, a0.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o0.data(), dsout, o0.size() * 4, cudaMemcpyDeviceToHost));
            }
        }
        std::vector<__half> a1(a0.size()); std::vector<float> o1(o0.size());
        CK(cudaMemcpy(a1.data(), dsact, a1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o1.data(), dsout, o1.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = !memcmp(a0.data(), a1.data(), a0.size() * 2) && !memcmp(o0.data(), o1.data(), o0.size() * 4);
        printf("M=%d shared determinism (2 runs bitwise equal): %s\n", M, same ? "OK" : "FAIL");
        fails += !same;

        // ---- shared expert vs the CPU
        double max_a = 0, max_o = 0; long n_a = 0, n_o = 0, bad_nan = 0, clamped = 0;
        std::vector<std::vector<double>> sref(M, std::vector<double>(NOUT)), sS(M, std::vector<double>(NOUT));
        for (int c = 0; c < 8; ++c)
        {
            if (c >= M)
            {   // fc2 must not have touched unused columns of out
                for (int o = 0; o < NOUT; ++o) bad_nan += !std::isnan(o0[static_cast<size_t>(c) * NOUT + o]);
                continue;
            }
            const Ref r = ref_fc1(sgw, suw, &xr[static_cast<size_t>(c) * K1], 10.0);
            std::vector<double> act(I);
            for (int i = 0; i < I; ++i)
            {
                const double got = __half2float(a0[static_cast<size_t>(c) * I + i]);
                const double tol = std::ldexp(std::fabs(r.act[i]), -9) + 1e-5 * r.actS[i];
                max_a = std::max(max_a, std::fabs(got - r.act[i]) / tol); ++n_a; bad_nan += std::isnan(got);
                act[i] = got;
            }
            clamped += r.hits;
            ref_fc2(sdw, act, sref[c], sS[c]);   // from the GPU's own activations: isolates fc2 from fc1's fp16 rounding
            for (int o = 0; o < NOUT; ++o)
            {
                const double got = o0[static_cast<size_t>(c) * NOUT + o];
                const double tol = 1e-5 * sS[c][o] + 1e-30;
                max_o = std::max(max_o, std::fabs(got - sref[c][o]) / tol); ++n_o; bad_nan += std::isnan(got);
            }
        }
        const bool ok = max_a <= 1.0 && max_o <= 1.0 && !bad_nan;
        printf("M=%d shared: fc1 act %ld values, max err/tol %.3f | fc2 out %ld values, max err/tol %.3f | unmasked-written/NaN-in-live %ld | rows past the limit (clamp hit) %ld -> %s\n",
               M, n_a, max_a, n_o, max_o, bad_nan, clamped, ok ? "OK" : "FAIL");
        fails += !ok;

        // ---- the whole layer: routed (6 experts, route weights) + shared, combined
        std::vector<std::vector<std::vector<double>>> eact(NE, std::vector<std::vector<double>>(M));
        for (int odt = 0; odt < 3; ++odt)
        {
            dsh::CombineArgsT<TK> CA{dplan, dy, dsout, dfin, M};
            const int grid = M * NOUT / (8 * 128);
            if (odt == 0) dsh::k_combine<NOUT, 128, gv::O_F16, TK><<<grid, 128>>>(CA);
            else if (odt == 1) dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<grid, 128>>>(CA);
            else dsh::k_combine<NOUT, 128, gv::O_F32, TK><<<grid, 128>>>(CA);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<uint8_t> got(static_cast<size_t>(M) * NOUT * (odt == 2 ? 4 : 2));
            CK(cudaMemcpy(got.data(), dfin, got.size(), cudaMemcpyDeviceToHost));
            double max_y = 0; long n_y = 0, neg_sh = 0, neg_rt = 0;      // neg_*: negative controls, see below
            for (int m = 0; m < M; ++m)
            {
                std::vector<double> ysum(NOUT, 0.0), yS(NOUT, 0.0), lastr(NOUT, 0.0);
                for (int k = 0; k < TK; ++k)
                {
                    const int e = route[m][k];
                    if (eact[e][m].empty())
                    {   // routed expert e on token m, double reference
                        const Ref r = ref_fc1(deq_g[e], deq_u[e], &xr[static_cast<size_t>(m) * K1], 10.0);
                        eact[e][m] = r.act;
                    }
                    std::vector<double> y, S;
                    ref_fc2(deq_d[e], eact[e][m], y, S);
                    for (int o = 0; o < NOUT; ++o) { ysum[o] += rw[m][k] * y[o]; yS[o] += rw[m][k] * S[o]; if (k == TK - 1) lastr[o] = rw[m][k] * y[o]; }
                }
                for (int o = 0; o < NOUT; ++o)
                {
                    const double ref = ysum[o] + sref[m][o];
                    const double S = yS[o] + sS[m][o];
                    const size_t i = static_cast<size_t>(m) * NOUT + o;
                    double v;
                    if (odt == 0) { __half h; memcpy(&h, &got[i * 2], 2); v = __half2float(h); }
                    else if (odt == 1) { uint16_t b; memcpy(&b, &got[i * 2], 2); const uint32_t u = static_cast<uint32_t>(b) << 16; float f; memcpy(&f, &u, 4); v = f; }
                    else { float f; memcpy(&f, &got[i * 4], 4); v = f; }
                    // fp16 acts and fp16 route outputs: each term is good to 2^-10 of its abs-sum; the final cast adds 2^-9 (bf16: 2^-8) of |y|
                    const double tol = std::ldexp(S, -10) + std::ldexp(std::fabs(ref), odt == 1 ? -8 : -9);
                    max_y = std::max(max_y, std::fabs(v - ref) / tol); ++n_y;
                    neg_sh += std::fabs(v - ysum[o]) > tol;            // the reference without the shared term must be rejected
                    neg_rt += std::fabs(v - (ref - lastr[o])) > tol;   // so must the reference without the last route
                }
            }
            const bool oky = max_y <= 1.0;
            printf("TK=%d M=%d layer (%d routed x weight + shared, out=%s): %ld values, max err/tol %.3f -> %s\n", TK, M, TK, odt == 0 ? "f16" : odt == 1 ? "bf16" : "f32", n_y, max_y, oky ? "OK" : "FAIL");
            const bool okn = neg_sh > n_y / 2 && neg_rt > n_y / 2;
            printf("   negative control: of %ld values, %ld fall outside the tolerance of a reference MISSING the shared term, %ld of one missing the last route -> %s\n", n_y, neg_sh, neg_rt, okn ? "tolerance is sensitive" : "TOLERANCE TOO LOOSE");
            fails += !oky + !okn;
        }
        cudaFree(dx); cudaFree(dxf); cudaFree(dse); cudaFree(dsm); cudaFree(dsw); cudaFree(dact); cudaFree(dy); cudaFree(dsact); cudaFree(dsout); cudaFree(dplan); cudaFree(dfin);
    }
    cudaFree(d1); cudaFree(d2); cudaFree(dg); cudaFree(du); cudaFree(dd);
    return fails;
}

// ------------------------------------------------------------------------------------------ the whole MoE chain, as ds41_moe_ops.cuh launches it
// router (k_plan on the device) -> shared -> routed over the plan with the n_slots_dev gate -> combine to bf16. The experts the router may
// pick are the first NREAL (the gate bias of the others is -1e4, so the top-TK never reaches them); the reference takes its routes and weights
// from the device plan (the router itself is ds_router_test's job; here: same set of routes as a CPU top-TK over the NREAL, weights sum 1.5).
template <int NE, int TK> static int check_chain()
{
    constexpr int NREAL = 8;
    static_assert(NE >= NREAL && TK <= NREAL, "chain test");
    std::mt19937_64 g(11);
    struct Expert { Matrix gate, up, down; };
    std::vector<Expert> ex(NREAL);
    for (auto& e : ex) { e.gate = rand_matrix(g, I, K1, 119, 124); e.up = rand_matrix(g, I, K1, 119, 124); e.down = rand_matrix(g, NOUT, I, 117, 122); }
    const M8 sg = rand_m8(g, I, K1, 118, 123, 7, 126, 130), su = rand_m8(g, I, K1, 118, 123, 5, 126, 130), sd = rand_m8(g, NOUT, I, 117, 122);
    std::vector<uint8_t> h1, h2;
    for (auto& e : ex)
    {
        auto a = mx4pack::pack_fc1(e.gate.packed.data(), e.gate.e8.data(), e.up.packed.data(), e.up.e8.data(), I, K1);
        auto b = mx4pack::pack_matrix(e.down.packed.data(), e.down.e8.data(), NOUT, I);
        h1.insert(h1.end(), a.begin(), a.end());
        h2.insert(h2.end(), b.begin(), b.end());
    }
    const std::vector<uint8_t> pg = pack8(sg), pu = pack8(su), pd = pack8(sd);
    std::vector<__nv_bfloat16> hgate(static_cast<size_t>(NE) * K1);
    std::normal_distribution<float> gn(0.f, 0.05f);
    for (auto& v : hgate) v = __float2bfloat16_rn(gn(g));
    std::vector<float> hbias(NE, -1e4f);
    for (int e = 0; e < NREAL; ++e) hbias[e] = 0.f;
    uint8_t *d1, *d2, *dg, *du, *dd; __nv_bfloat16* dgate; float* dbias;
    CK(cudaMalloc(&d1, h1.size())); CK(cudaMalloc(&d2, h2.size())); CK(cudaMalloc(&dg, pg.size())); CK(cudaMalloc(&du, pu.size())); CK(cudaMalloc(&dd, pd.size()));
    CK(cudaMalloc(&dgate, hgate.size() * 2)); CK(cudaMalloc(&dbias, NE * 4));
    CK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, pg.data(), pg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, pu.data(), pu.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, pd.data(), pd.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgate, hgate.data(), hgate.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dbias, hbias.data(), NE * 4, cudaMemcpyHostToDevice));
    std::vector<std::vector<float>> deq_g(NREAL), deq_u(NREAL), deq_d(NREAL);
    for (int e = 0; e < NREAL; ++e) { deq_g[e] = dequant(ex[e].gate); deq_u[e] = dequant(ex[e].up); deq_d[e] = dequant(ex[e].down); }
    const std::vector<float> sgw = dequant8(sg), suw = dequant8(su), sdw = dequant8(sd);

    constexpr int SM = dsr::MMAX * TK;
    int fails = 0;
    for (int M : {1, 5, 8})
    {
        std::normal_distribution<float> nd(0.f, 0.5f);
        std::vector<__nv_bfloat16> xb(static_cast<size_t>(M) * K1);
        std::vector<float> xr(static_cast<size_t>(M) * K1);
        for (size_t i = 0; i < xb.size(); ++i) { xb[i] = __float2bfloat16_rn(nd(g)); xr[i] = __half2float(__float2half_rn(__bfloat162float(xb[i]))); }
        __nv_bfloat16* dx; u32* dxf; float* dlog; char* dplan; __half *dact, *dy, *dsact; float* dsout; void* dfin;
        CK(cudaMalloc(&dx, xb.size() * 2)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4)); CK(cudaMalloc(&dlog, 8 * NE * 4)); CK(cudaMalloc(&dplan, sizeof(dsr::PlanT<TK>)));
        CK(cudaMalloc(&dact, static_cast<size_t>(SM) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(SM) * 8 * NOUT * 2));
        CK(cudaMalloc(&dsact, 8 * I * 2)); CK(cudaMalloc(&dsout, 8 * NOUT * 4)); CK(cudaMalloc(&dfin, 8 * NOUT * 2));
        CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(dact, 0xff, static_cast<size_t>(SM) * 8 * I * 2)); CK(cudaMemset(dy, 0xff, static_cast<size_t>(SM) * 8 * NOUT * 2));    // NaN sentinels: an unwritten slot poisons the output
        CK(cudaMemset(dsact, 0xff, 8 * I * 2)); CK(cudaMemset(dsout, 0xff, 8 * NOUT * 4)); CK(cudaMemset(dfin, 0xff, 8 * NOUT * 2));
        using P = dsr::PlanT<TK>;
        auto run = [&]()
        {
            mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, M, K1, dxf);
            dsr::k_gate_logits<NE><<<NE / 8, 256>>>(dgate, dx, M, dlog);
            dsr::k_plan<NE, TK><<<1, dsr::PLAN_THREADS>>>(dlog, dbias, M, reinterpret_cast<P*>(dplan));
            dsh::Args SA{dg, du, dd, dxf, (1 << M) - 1, 10.f, dsact, dsout};
            dsh::k_sh_fc1<K1, I, 8, 1, 2><<<I / 16, 8 * 32>>>(SA);
            dsh::k_sh_fc2<I, NOUT, 8, 3><<<NOUT / 16 / 8, 8 * 32>>>(SA);
            mx4::Args RA{d1, d2, dxf, reinterpret_cast<const int*>(dplan + offsetof(P, slot_expert)), reinterpret_cast<const int*>(dplan + offsetof(P, slot_mask)),
                         reinterpret_cast<const float*>(dplan + offsetof(P, slot_w)), reinterpret_cast<const int*>(dplan + offsetof(P, n_slots)), M * TK, 10.f, dact, dy};
            mx4::k_fc1<K1, I, 8, 1, 2, false><<<M * TK * (I / 16), 8 * 32>>>(RA);
            mx4::k_fc2<I, NOUT, 8, 2, false><<<M * TK * (NOUT / 16 / 8), 8 * 32>>>(RA);
            dsh::CombineArgsT<TK> CA{reinterpret_cast<const P*>(dplan), dy, dsout, dfin, M};
            dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<M * NOUT / (8 * 128), 128>>>(CA);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        };
        run();
        std::vector<uint16_t> o0(static_cast<size_t>(M) * NOUT);
        CK(cudaMemcpy(o0.data(), dfin, o0.size() * 2, cudaMemcpyDeviceToHost));
        run();
        std::vector<uint16_t> o1(o0.size());
        CK(cudaMemcpy(o1.data(), dfin, o1.size() * 2, cudaMemcpyDeviceToHost));
        const bool same = o0 == o1;
        std::vector<uint8_t> hp(sizeof(P)); CK(cudaMemcpy(hp.data(), dplan, sizeof(P), cudaMemcpyDeviceToHost));
        const P& pl = *reinterpret_cast<const P*>(hp.data());
        // routes: the CPU top-TK over the NREAL real experts (key = sqrt(softplus(logit)), bias 0 there) must be the plan's set
        int bad_route = 0, bad_w = 0, bad_slot = 0;
        std::vector<std::vector<int>> route(M);
        std::vector<std::vector<float>> rw(M);
        for (int m = 0; m < M; ++m)
        {
            std::vector<std::pair<double, int>> key;
            for (int e = 0; e < NREAL; ++e)
            {
                double lg = 0; for (int k = 0; k < K1; ++k) lg += static_cast<double>(__bfloat162float(hgate[static_cast<size_t>(e) * K1 + k])) * __bfloat162float(xb[static_cast<size_t>(m) * K1 + k]);
                key.push_back({std::sqrt(lg > 20 ? lg : std::log1p(std::exp(lg))), e});
            }
            std::sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            std::vector<int> cpu, got; float wsum = 0;
            for (int k = 0; k < TK; ++k) { cpu.push_back(key[k].second); got.push_back(pl.tok_expert[m][k]); route[m].push_back(pl.tok_expert[m][k]); rw[m].push_back(pl.tok_w[m][k]); wsum += pl.tok_w[m][k]; }
            std::sort(cpu.begin(), cpu.end()); std::sort(got.begin(), got.end());
            const bool near_tie = key[TK - 1].first - key[TK].first < 1e-6;
            bad_route += cpu != got && !near_tie;
            bad_w += std::fabs(wsum - 1.5f) > 1e-4f;
            for (int k = 0; k < TK; ++k) bad_slot += pl.tok_slot[m][k] < 0 || pl.tok_slot[m][k] >= pl.n_slots || pl.slot_expert[pl.tok_slot[m][k]] != pl.tok_expert[m][k] || pl.tok_expert[m][k] >= NREAL;
        }
        // the layer: sum over the plan's routes of weight * expert(x) + shared, from double references
        double max_y = 0; long n_y = 0, bad_nan = 0;
        std::vector<std::vector<std::vector<double>>> eact(NREAL, std::vector<std::vector<double>>(M));
        std::vector<std::vector<double>> sref(M), sS(M);
        for (int m = 0; m < M; ++m)
        {
            const Ref r = ref_fc1(sgw, suw, &xr[static_cast<size_t>(m) * K1], 10.0);
            ref_fc2(sdw, r.act, sref[m], sS[m]);
        }
        for (int m = 0; m < M; ++m)
        {
            std::vector<double> ysum(NOUT, 0.0), yS(NOUT, 0.0);
            for (int k = 0; k < TK; ++k)
            {
                const int e = route[m][k];
                if (e < 0 || e >= NREAL) continue;
                if (eact[e][m].empty()) eact[e][m] = ref_fc1(deq_g[e], deq_u[e], &xr[static_cast<size_t>(m) * K1], 10.0).act;
                std::vector<double> y, S;
                ref_fc2(deq_d[e], eact[e][m], y, S);
                for (int o = 0; o < NOUT; ++o) { ysum[o] += rw[m][k] * y[o]; yS[o] += rw[m][k] * S[o]; }
            }
            for (int o = 0; o < NOUT; ++o)
            {
                const double ref = ysum[o] + sref[m][o], S = yS[o] + sS[m][o];
                const uint32_t u = static_cast<uint32_t>(o0[static_cast<size_t>(m) * NOUT + o]) << 16; float v; memcpy(&v, &u, 4);
                bad_nan += std::isnan(v);
                const double tol = std::ldexp(S, -10) + std::ldexp(std::fabs(ref), -8);
                max_y = std::max(max_y, std::fabs(v - ref) / tol); ++n_y;
            }
        }
        const bool ok = max_y <= 1.0 && !bad_nan && !bad_route && !bad_w && !bad_slot && same;
        printf("chain NE=%d TK=%d M=%d: n_slots %d, routes wrong %d, weight sums wrong %d, plan entries wrong %d, determinism (2 runs bitwise) %s, bf16 out %ld values: NaN %ld, max err/tol %.3f -> %s\n",
               NE, TK, M, pl.n_slots, bad_route, bad_w, bad_slot, same ? "OK" : "FAIL", n_y, bad_nan, max_y, ok ? "OK" : "FAIL");
        fails += !ok;
        cudaFree(dx); cudaFree(dxf); cudaFree(dlog); cudaFree(dplan); cudaFree(dact); cudaFree(dy); cudaFree(dsact); cudaFree(dsout); cudaFree(dfin);
    }
    cudaFree(d1); cudaFree(d2); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dgate); cudaFree(dbias);
    return fails;
}

// ------------------------------------------------------------------------------------------ multi-tile chain, M 1..32
// The launch sequence of ds41_moe_ops.cuh router_mt / shared_mt_nt / experts_mt_nt, replicated here (the test does not include the adapter).
struct MtBuf
{
    const __nv_bfloat16* gate; const float* bias; const __nv_bfloat16* x; u32* xf; float* log; char* plan;
    __half *act, *y, *sact; float* sout; void* fin; const uint8_t *w1, *w2, *sg, *su, *sd;
};
template <int TK> static void mt_combine(const MtBuf& b, int M, cudaStream_t st = 0)
{
    using P = dsr::PlanMT<TK, dsr::MT_MAX>;
    dsh::CombineArgsMT<TK, dsr::MT_MAX> CA{reinterpret_cast<const P*>(b.plan), b.y, b.sout, b.fin, M};
    dsh::k_combine_mt<NOUT, 128, gv::O_BF16, TK, dsr::MT_MAX><<<M * NOUT / (8 * 128), 128, 0, st>>>(CA);
}
template <int NE, int TK> static void mt_router(const MtBuf& b, int M, cudaStream_t st = 0)
{
    using P = dsr::PlanMT<TK, dsr::MT_MAX>;
    const int nt = (M + 7) / 8;
    mx4::build_xfrag_mt<<<(nt * K1 * 4 + 255) / 256, 256, 0, st>>>(b.x, M, K1, nt, b.xf);
    dsr::k_gate_logits_mt<NE><<<NE / 8, 256, 0, st>>>(b.gate, b.x, M, b.log);
    dsr::k_plan_mt<NE, TK, dsr::MT_MAX><<<1, dsr::PLAN_THREADS, 0, st>>>(b.log, b.bias, M, reinterpret_cast<P*>(b.plan));
}
template <int NT> static void mt_shared(const MtBuf& b, int M, cudaStream_t st = 0)
{
    dsh::ArgsMT SA{b.sg, b.su, b.sd, b.xf, M, 10.f, b.sact, b.sout};
    dsh::k_sh_fc1_mt<K1, I, 8, 1, 2, NT><<<I / 16, 8 * 32, 0, st>>>(SA);
    dsh::k_sh_fc2_mt<I, NOUT, 8, 3, NT><<<NOUT / 16 / 8, 8 * 32, 0, st>>>(SA);
}
template <int TK> static mx4::ArgsMT mt_routed_args(const MtBuf& b, int M)
{
    using P = dsr::PlanMT<TK, dsr::MT_MAX>;
    const char* p = b.plan;
    return mx4::ArgsMT{b.w1, b.w2, b.xf,
                       reinterpret_cast<const int*>(p + offsetof(P, slot_expert)), reinterpret_cast<const int*>(p + offsetof(P, slot_tile0)), reinterpret_cast<const int*>(p + offsetof(P, slot_nt)),
                       reinterpret_cast<const int*>(p + offsetof(P, tile_mask)), reinterpret_cast<const float*>(p + offsetof(P, tile_w)), reinterpret_cast<const int*>(p + offsetof(P, tile_tok)),
                       reinterpret_cast<const int*>(p + offsetof(P, n_slots)), M * TK, 10.f, b.act, b.y};
}
template <int TK, int NT> static void mt_fc1(const MtBuf& b, int M, cudaStream_t st = 0) { mx4::k_fc1_mt<K1, I, 8, 1, mx4::FC1_MT_DEPTH<NT>, NT><<<M * TK * (I / 16), 8 * 32, 0, st>>>(mt_routed_args<TK>(b, M)); }
// fc1 variants for the mtbench NT=4 study: v 0 = D2/minB1 (the shipped instance), 1 = D2/minB2 (<= 128 regs, 2 CTAs/SM), 2 = D1/minB1, 3 = D1/minB2
template <int TK, int NT, int DEPTH, int MINB> static void mt_fc1v(const MtBuf& b, int M) { mx4::k_fc1_mt<K1, I, 8, 1, DEPTH, NT, MINB><<<M * TK * (I / 16), 8 * 32>>>(mt_routed_args<TK>(b, M)); }
template <int TK, int NT> static void mt_fc1_var(int v, const MtBuf& b, int M)
{
    if (v == 0) mt_fc1v<TK, NT, 2, 0>(b, M); else if (v == 1) mt_fc1v<TK, NT, 2, 2>(b, M); else if (v == 2) mt_fc1v<TK, NT, 1, 0>(b, M); else mt_fc1v<TK, NT, 1, 2>(b, M);
}
template <int TK> static void mt_fc1_any(int nt, int v, const MtBuf& b, int M)
{
    if (nt == 2) mt_fc1_var<TK, 2>(v, b, M); else if (nt == 3) mt_fc1_var<TK, 3>(v, b, M); else mt_fc1_var<TK, 4>(v, b, M);
}
template <int TK, int NT> static void mt_fc2(const MtBuf& b, int M, cudaStream_t st = 0) { mx4::k_fc2_mt<I, NOUT, 8, 2, NT><<<M * TK * (NOUT / 16 / 8), 8 * 32, 0, st>>>(mt_routed_args<TK>(b, M)); }
template <int NE, int TK, int NT> static void mt_chain(const MtBuf& b, int M, cudaStream_t st = 0)
{
    mt_router<NE, TK>(b, M, st);
    mt_shared<NT>(b, M, st);
    mt_fc1<TK, NT>(b, M, st);
    mt_fc2<TK, NT>(b, M, st);
    mt_combine<TK>(b, M, st);
}
template <int NE, int TK> static void mt_run(const MtBuf& b, int M, int NT)
{
    if (NT == 2) mt_chain<NE, TK, 2>(b, M); else if (NT == 3) mt_chain<NE, TK, 3>(b, M); else mt_chain<NE, TK, 4>(b, M);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------------------ the PDL chain (ds41_moe_ops.cuh, PD = 1) next to the plain one
// layer_plain = the shipped kernels with plain launches on a stream (the PD = 0 adapter's sequence); layer_pdl = the adapter's PD = 1 sequence launch for launch
// (tl_launch is its launch(): cudaLaunchKernelEx + the programmatic-stream-serialization attribute). The test does not include the adapter, so keep these in step with it.
template <typename... KP, typename... AP>
static void tl_launch(bool pdl, void (*k)(KP...), dim3 grid, dim3 block, cudaStream_t st, AP... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, k, static_cast<KP>(args)...));
}
__global__ void k_spin(long long cycles) { const long long t0 = clock64(); while (clock64() - t0 < cycles) {} }

template <int TK> static mx4::Args old_routed_args(const MtBuf& b, int M)
{
    using P = dsr::PlanT<TK>;
    const char* p = b.plan;
    return mx4::Args{b.w1, b.w2, b.xf, reinterpret_cast<const int*>(p + offsetof(P, slot_expert)), reinterpret_cast<const int*>(p + offsetof(P, slot_mask)),
                     reinterpret_cast<const float*>(p + offsetof(P, slot_w)), reinterpret_cast<const int*>(p + offsetof(P, n_slots)), M * TK, 10.f, b.act, b.y};
}
template <int NE, int TK> static void old_plain(const MtBuf& b, int M, cudaStream_t st)       // M <= 8, the original one-tile kernels
{
    using P = dsr::PlanT<TK>;
    mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256, 0, st>>>(b.x, M, K1, b.xf);
    dsr::k_gate_logits<NE><<<NE / 8, 256, 0, st>>>(b.gate, b.x, M, b.log);
    dsr::k_plan<NE, TK><<<1, dsr::PLAN_THREADS, 0, st>>>(b.log, b.bias, M, reinterpret_cast<P*>(b.plan));
    dsh::Args SA{b.sg, b.su, b.sd, b.xf, (1 << M) - 1, 10.f, b.sact, b.sout};
    dsh::k_sh_fc1<K1, I, 8, 1, 2><<<I / 16, 8 * 32, 0, st>>>(SA);
    dsh::k_sh_fc2<I, NOUT, 8, 3><<<NOUT / 16 / 8, 8 * 32, 0, st>>>(SA);
    const mx4::Args RA = old_routed_args<TK>(b, M);
    mx4::k_fc1<K1, I, 8, 1, 2, false><<<M * TK * (I / 16), 8 * 32, 0, st>>>(RA);
    mx4::k_fc2<I, NOUT, 8, 2, false><<<M * TK * (NOUT / 16 / 8), 8 * 32, 0, st>>>(RA);
    dsh::CombineArgsT<TK> CA{reinterpret_cast<const P*>(b.plan), b.y, b.sout, b.fin, M};
    dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<M * NOUT / (8 * 128), 128, 0, st>>>(CA);
}
template <int NE, int TK> static void old_pdl(const MtBuf& b, int M, cudaStream_t st)
{
    using P = dsr::PlanT<TK>;
    mx4::build_xfrag_p<<<(K1 / 16 * 64 + 255) / 256, 256, 0, st>>>(b.x, M, K1, b.xf);       // plain launch, the first of the chain
    tl_launch(true, dsr::k_gate_logits<NE, 1>, dim3(NE / 8), dim3(256), st, b.gate, b.x, M, b.log);
    tl_launch(true, dsr::k_plan_p<NE, TK>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)b.log, b.bias, M, reinterpret_cast<P*>(b.plan));
    dsh::Args SA{b.sg, b.su, b.sd, b.xf, (1 << M) - 1, 10.f, b.sact, b.sout};
    tl_launch(true, dsh::k_sh_fc1<K1, I, 8, 1, 2, 1>, dim3(I / 16), dim3(8 * 32), st, SA);
    tl_launch(true, dsh::k_sh_fc2<I, NOUT, 8, 3, 1>, dim3(NOUT / 16 / 8), dim3(8 * 32), st, SA);
    const mx4::Args RA = old_routed_args<TK>(b, M);
    tl_launch(true, mx4::k_fc1<K1, I, 8, 1, 2, false, 1>, dim3(M * TK * (I / 16)), dim3(8 * 32), st, RA);
    tl_launch(true, mx4::k_fc2<I, NOUT, 8, 2, false, 1>, dim3(M * TK * (NOUT / 16 / 8)), dim3(8 * 32), st, RA);
    dsh::CombineArgsT<TK> CA{reinterpret_cast<const P*>(b.plan), b.y, b.sout, b.fin, M};
    tl_launch(true, dsh::k_combine<NOUT, 128, gv::O_BF16, TK>, dim3(M * NOUT / (8 * 128)), dim3(128), st, CA);
}
template <int NE, int TK, int NT> static void mt_pdl(const MtBuf& b, int M, cudaStream_t st)    // M 9..32
{
    using P = dsr::PlanMT<TK, dsr::MT_MAX>;
    const int nt = (M + 7) / 8;
    mx4::build_xfrag_mt<1><<<(nt * K1 * 4 + 255) / 256, 256, 0, st>>>(b.x, M, K1, nt, b.xf);        // plain launch, the first of the chain
    tl_launch(true, dsr::k_gate_logits_mt<NE, 1>, dim3(NE / 8), dim3(256), st, b.gate, b.x, M, b.log);
    tl_launch(true, dsr::k_plan_mt<NE, TK, dsr::MT_MAX, 1>, dim3(1), dim3(dsr::PLAN_THREADS), st, (const float*)b.log, b.bias, M, reinterpret_cast<P*>(b.plan));
    dsh::ArgsMT SA{b.sg, b.su, b.sd, b.xf, M, 10.f, b.sact, b.sout};
    tl_launch(true, dsh::k_sh_fc1_mt<K1, I, 8, 1, 2, NT, 1>, dim3(I / 16), dim3(8 * 32), st, SA);
    tl_launch(true, dsh::k_sh_fc2_mt<I, NOUT, 8, 3, NT, 1>, dim3(NOUT / 16 / 8), dim3(8 * 32), st, SA);
    const mx4::ArgsMT RA = mt_routed_args<TK>(b, M);
    tl_launch(true, mx4::k_fc1_mt<K1, I, 8, 1, mx4::FC1_MT_DEPTH<NT>, NT, 0, 1>, dim3(M * TK * (I / 16)), dim3(8 * 32), st, RA);
    tl_launch(true, mx4::k_fc2_mt<I, NOUT, 8, 2, NT, 1>, dim3(M * TK * (NOUT / 16 / 8)), dim3(8 * 32), st, RA);
    dsh::CombineArgsMT<TK, dsr::MT_MAX> CA{reinterpret_cast<const P*>(b.plan), b.y, b.sout, b.fin, M};
    tl_launch(true, dsh::k_combine_mt<NOUT, 128, gv::O_BF16, TK, dsr::MT_MAX>, dim3(M * NOUT / (8 * 128)), dim3(128), st, CA);
}
template <int NE, int TK> static void layer_plain(const MtBuf& b, int M, cudaStream_t st)
{
    if (M <= 8) { old_plain<NE, TK>(b, M, st); return; }
    const int nt = (M + 7) / 8;
    if (nt == 2) mt_chain<NE, TK, 2>(b, M, st); else if (nt == 3) mt_chain<NE, TK, 3>(b, M, st); else mt_chain<NE, TK, 4>(b, M, st);
}
template <int NE, int TK> static void layer_pdl(const MtBuf& b, int M, cudaStream_t st)
{
    if (M <= 8) { old_pdl<NE, TK>(b, M, st); return; }
    const int nt = (M + 7) / 8;
    if (nt == 2) mt_pdl<NE, TK, 2>(b, M, st); else if (nt == 3) mt_pdl<NE, TK, 3>(b, M, st); else mt_pdl<NE, TK, 4>(b, M, st);
}

// The device plan against dsr::plan_ref built from the device's own routes (tok_expert / tok_w). Returns the first differing field, or nullptr.
template <int TK> static const char* mt_plan_diff(const dsr::PlanMT<TK, dsr::MT_MAX>& d, int M)
{
    int ex[dsr::MT_MAX][8] = {}; float wt[dsr::MT_MAX][8] = {};
    for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) { ex[m][k] = d.tok_expert[m][k]; wt[m][k] = d.tok_w[m][k]; }
    auto r = std::make_unique<dsr::PlanMT<TK, dsr::MT_MAX>>();
    dsr::plan_ref<TK, dsr::MT_MAX>(M, ex, wt, *r);
#define MT_DIFF(f) if (memcmp(&d.f, &r->f, sizeof d.f)) return #f;
    MT_DIFF(n_slots) MT_DIFF(n_tiles) MT_DIFF(slot_expert) MT_DIFF(slot_tile0) MT_DIFF(slot_nt) MT_DIFF(tile_mask) MT_DIFF(tile_w) MT_DIFF(tile_tok)
    MT_DIFF(tok_slot) MT_DIFF(tok_col) MT_DIFF(tok_w) MT_DIFF(tok_expert)
#undef MT_DIFF
    return nullptr;
}

template <class F> static void par_for(int n, F f)
{
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(n, static_cast<int>(std::thread::hardware_concurrency())));
    for (int t = 0; t < nth; ++t) th.emplace_back([&] { for (int i; (i = next++) < n;) f(i); });
    for (auto& t : th) t.join();
}

// Checks per (bias mode, M): (a) device plan == plan_ref; routes = CPU top-TK over the NREAL real experts, weights sum 1.5;
// (b) M <= 8 through the mt kernels (NT 2 and 4) == the one-tile path, bf16 bit for bit; (c) M 9..32: token m == the same token alone through the one-tile path (M = 1),
// bit for bit; (d) CPU double reference within the chain test's tolerance; (e) two runs bit-identical, NT = 4 (and 2) == NT = ceil(M/8); (f) NaN sentinels in
// act / ypart / sact / sout / fin / plan and NaN in the x rows >= M. bias "skew": experts 0 and 1 get +8, so every token routes to both (n_e = M, nt = 4 at M = 32).
// Controls: a NaN-poisoned ypart tile must reach the output; a corrupted plan copy must be reported by mt_plan_diff.
template <int NE, int TK> static int check_mt()
{
    constexpr int NREAL = 8, MT = dsr::MT_MAX;
    static_assert(NE >= NREAL && TK <= NREAL, "multitile test");
    using PM = dsr::PlanMT<TK, MT>; using PO = dsr::PlanT<TK>;
    std::mt19937_64 g(11);
    struct Expert { Matrix gate, up, down; };
    std::vector<Expert> ex(NREAL);
    for (auto& e : ex) { e.gate = rand_matrix(g, I, K1, 119, 124); e.up = rand_matrix(g, I, K1, 119, 124); e.down = rand_matrix(g, NOUT, I, 117, 122); }
    const M8 sg = rand_m8(g, I, K1, 118, 123, 7, 126, 130), su = rand_m8(g, I, K1, 118, 123, 5, 126, 130), sd = rand_m8(g, NOUT, I, 117, 122);
    std::vector<uint8_t> h1, h2;
    for (auto& e : ex)
    {
        auto a = mx4pack::pack_fc1(e.gate.packed.data(), e.gate.e8.data(), e.up.packed.data(), e.up.e8.data(), I, K1);
        auto b = mx4pack::pack_matrix(e.down.packed.data(), e.down.e8.data(), NOUT, I);
        h1.insert(h1.end(), a.begin(), a.end());
        h2.insert(h2.end(), b.begin(), b.end());
    }
    const std::vector<uint8_t> pg = pack8(sg), pu = pack8(su), pd = pack8(sd);
    std::vector<__nv_bfloat16> hgate(static_cast<size_t>(NE) * K1);
    std::normal_distribution<float> gn(0.f, 0.05f);
    for (auto& v : hgate) v = __float2bfloat16_rn(gn(g));
    uint8_t *d1, *d2, *dg, *du, *dd; __nv_bfloat16* dgate; float* dbias;
    CK(cudaMalloc(&d1, h1.size())); CK(cudaMalloc(&d2, h2.size())); CK(cudaMalloc(&dg, pg.size())); CK(cudaMalloc(&du, pu.size())); CK(cudaMalloc(&dd, pd.size()));
    CK(cudaMalloc(&dgate, hgate.size() * 2)); CK(cudaMalloc(&dbias, NE * 4));
    CK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, pg.data(), pg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, pu.data(), pu.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, pd.data(), pd.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgate, hgate.data(), hgate.size() * 2, cudaMemcpyHostToDevice));
    std::vector<std::vector<float>> deq_g(NREAL), deq_u(NREAL), deq_d(NREAL);
    for (int e = 0; e < NREAL; ++e) { deq_g[e] = dequant(ex[e].gate); deq_u[e] = dequant(ex[e].up); deq_d[e] = dequant(ex[e].down); }
    ex.clear(); ex.shrink_to_fit(); h1.clear(); h1.shrink_to_fit(); h2.clear(); h2.shrink_to_fit();       // host memory: the dequantized copies are what the reference needs
    const std::vector<float> sgw = dequant8(sg), suw = dequant8(su), sdw = dequant8(sd);

    // multi-tile buffers (one set, sized for MT = 32) and the one-tile path's (8 rows)
    constexpr size_t PLAN_B = sizeof(PM) > sizeof(PO) ? sizeof(PM) : sizeof(PO);
    constexpr int SMT = MT * TK, SMO = dsr::MMAX * TK;
    __nv_bfloat16* dx; u32 *dxf, *dxo; float *dlog, *dsout, *dsouto; char *dplan, *dplo; __half *dact, *dy, *dsact, *dacto, *dyo, *dsacto; void *dfin, *dfino; __nv_bfloat16* dxs;
    CK(cudaMalloc(&dx, static_cast<size_t>(MT) * K1 * 2)); CK(cudaMalloc(&dxs, static_cast<size_t>(MT) * K1 * 2)); CK(cudaMalloc(&dxf, static_cast<size_t>(MT / 8) * K1 * 4 * 4)); CK(cudaMalloc(&dxo, static_cast<size_t>(K1) * 4 * 4));
    CK(cudaMalloc(&dlog, static_cast<size_t>(MT) * NE * 4)); CK(cudaMalloc(&dplan, PLAN_B)); CK(cudaMalloc(&dplo, sizeof(PO)));
    CK(cudaMalloc(&dact, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(SMT) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsact, static_cast<size_t>(MT) * I * 2)); CK(cudaMalloc(&dsout, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMalloc(&dfin, static_cast<size_t>(MT) * NOUT * 2));
    CK(cudaMalloc(&dacto, static_cast<size_t>(SMO) * 8 * I * 2)); CK(cudaMalloc(&dyo, static_cast<size_t>(SMO) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsacto, 8 * I * 2)); CK(cudaMalloc(&dsouto, 8 * NOUT * 4)); CK(cudaMalloc(&dfino, 8 * NOUT * 2));
    const MtBuf B{dgate, dbias, dx, dxf, dlog, dplan, dact, dy, dsact, dsout, dfin, d1, d2, dg, du, dd};
    auto poison = [&]()
    {
        CK(cudaMemset(dact, 0xff, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMemset(dy, 0xff, static_cast<size_t>(SMT) * 8 * NOUT * 2));
        CK(cudaMemset(dsact, 0xff, static_cast<size_t>(MT) * I * 2)); CK(cudaMemset(dsout, 0xff, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMemset(dfin, 0xff, static_cast<size_t>(MT) * NOUT * 2));
        CK(cudaMemset(dplan, 0xff, PLAN_B)); CK(cudaMemset(dxf, 0xff, static_cast<size_t>(MT / 8) * K1 * 4 * 4));
    };
    // the one-tile path (check_chain's launch sequence) on M <= 8 rows of dsrc
    auto run_old = [&](const __nv_bfloat16* dsrc, int M, std::vector<uint16_t>& out, std::vector<u32>* xf0)
    {
        CK(cudaMemset(dacto, 0xff, static_cast<size_t>(SMO) * 8 * I * 2)); CK(cudaMemset(dyo, 0xff, static_cast<size_t>(SMO) * 8 * NOUT * 2));
        CK(cudaMemset(dsacto, 0xff, 8 * I * 2)); CK(cudaMemset(dsouto, 0xff, 8 * NOUT * 4)); CK(cudaMemset(dfino, 0xff, 8 * NOUT * 2));
        mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dsrc, M, K1, dxo);
        dsr::k_gate_logits<NE><<<NE / 8, 256>>>(dgate, dsrc, M, dlog);
        dsr::k_plan<NE, TK><<<1, dsr::PLAN_THREADS>>>(dlog, dbias, M, reinterpret_cast<PO*>(dplo));
        dsh::Args SA{dg, du, dd, dxo, (1 << M) - 1, 10.f, dsacto, dsouto};
        dsh::k_sh_fc1<K1, I, 8, 1, 2><<<I / 16, 8 * 32>>>(SA);
        dsh::k_sh_fc2<I, NOUT, 8, 3><<<NOUT / 16 / 8, 8 * 32>>>(SA);
        mx4::Args RA{d1, d2, dxo, reinterpret_cast<const int*>(dplo + offsetof(PO, slot_expert)), reinterpret_cast<const int*>(dplo + offsetof(PO, slot_mask)),
                     reinterpret_cast<const float*>(dplo + offsetof(PO, slot_w)), reinterpret_cast<const int*>(dplo + offsetof(PO, n_slots)), M * TK, 10.f, dacto, dyo};
        mx4::k_fc1<K1, I, 8, 1, 2, false><<<M * TK * (I / 16), 8 * 32>>>(RA);
        mx4::k_fc2<I, NOUT, 8, 2, false><<<M * TK * (NOUT / 16 / 8), 8 * 32>>>(RA);
        dsh::CombineArgsT<TK> CA{reinterpret_cast<const PO*>(dplo), dyo, dsouto, dfino, M};
        dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<M * NOUT / (8 * 128), 128>>>(CA);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        out.resize(static_cast<size_t>(M) * NOUT);
        CK(cudaMemcpy(out.data(), dfino, out.size() * 2, cudaMemcpyDeviceToHost));
        if (xf0) { xf0->resize(static_cast<size_t>(K1) * 4); CK(cudaMemcpy(xf0->data(), dxo, xf0->size() * 4, cudaMemcpyDeviceToHost)); }
    };

    int fails = 0;
    long last_M = 0;
    for (int skew = 0; skew < 2; ++skew)
    {
        std::vector<float> hbias(NE, -1e4f);
        for (int e = 0; e < NREAL; ++e) hbias[e] = skew && e < 2 ? 8.f : 0.f;
        CK(cudaMemcpy(dbias, hbias.data(), NE * 4, cudaMemcpyHostToDevice));
        for (int M : {1, 5, 8, 9, 12, 16, 17, 24, 25, 31, 32})
        {
            std::normal_distribution<float> nd(0.f, 0.5f);
            std::vector<__nv_bfloat16> xb(static_cast<size_t>(MT) * K1, __float2bfloat16_rn(NAN));       // rows >= M stay NaN: a read of them poisons the result
            std::vector<float> xr(static_cast<size_t>(M) * K1);
            for (size_t i = 0; i < xr.size(); ++i) { xb[i] = __float2bfloat16_rn(nd(g)); xr[i] = __half2float(__float2half_rn(__bfloat162float(xb[i]))); }
            CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
            const int NT0 = M <= 8 ? 2 : (M + 7) / 8, NT1 = NT0 == 4 ? 0 : 4;      // NT must be >= ceil(M/8): the other template is NT = 4 (none for M > 24)
            poison();
            mt_run<NE, TK>(B, M, NT0);
            std::vector<uint16_t> o0(static_cast<size_t>(M) * NOUT);
            CK(cudaMemcpy(o0.data(), dfin, o0.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<uint8_t> hp(PLAN_B); CK(cudaMemcpy(hp.data(), dplan, PLAN_B, cudaMemcpyDeviceToHost));
            const PM& pl = *reinterpret_cast<const PM*>(hp.data());
            std::vector<u32> xfm(static_cast<size_t>(K1) * 4); CK(cudaMemcpy(xfm.data(), dxf, xfm.size() * 4, cudaMemcpyDeviceToHost));
            // (e) the second run (same NT) and the other NT, bitwise
            poison(); mt_run<NE, TK>(B, M, NT0);
            std::vector<uint16_t> o1(o0.size()); CK(cudaMemcpy(o1.data(), dfin, o1.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<uint16_t> o2 = o0;
            if (NT1) { poison(); mt_run<NE, TK>(B, M, NT1); CK(cudaMemcpy(o2.data(), dfin, o2.size() * 2, cudaMemcpyDeviceToHost)); }
            const bool same_run = o0 == o1, same_nt = o0 == o2;
            // (a) plan == plan_ref, routes, weights
            const char* pdiff = mt_plan_diff<TK>(pl, M);
            int bad_route = 0, bad_w = 0, max_tiles = 0;
            std::vector<std::vector<int>> route(M);
            std::vector<std::vector<float>> rw(M);
            for (int m = 0; m < M; ++m)
            {
                std::vector<std::pair<double, int>> key;
                for (int e = 0; e < NREAL; ++e)
                {
                    double lg = 0; for (int k = 0; k < K1; ++k) lg += static_cast<double>(__bfloat162float(hgate[static_cast<size_t>(e) * K1 + k])) * __bfloat162float(xb[static_cast<size_t>(m) * K1 + k]);
                    key.push_back({std::sqrt(lg > 20 ? lg : std::log1p(std::exp(lg))) + hbias[e], e});
                }
                std::sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                std::vector<int> cpu, got; float wsum = 0;
                for (int k = 0; k < TK; ++k) { cpu.push_back(key[k].second); got.push_back(pl.tok_expert[m][k]); route[m].push_back(pl.tok_expert[m][k]); rw[m].push_back(pl.tok_w[m][k]); wsum += pl.tok_w[m][k]; }
                std::sort(cpu.begin(), cpu.end()); std::sort(got.begin(), got.end());
                const bool near_tie = key[TK - 1].first - key[TK].first < 1e-6;
                bad_route += cpu != got && !near_tie;
                bad_w += std::fabs(wsum - 1.5f) > 1e-4f;
            }
            for (int s = 0; s < pl.n_slots; ++s) max_tiles = std::max(max_tiles, pl.slot_nt[s]);
            // (b) / (c) against the one-tile path
            int bad_old = 0, bad_xf = 0; const char* old_kind = "";
            {
                std::vector<u32> xf8;
                std::vector<uint16_t> oo;
                if (M <= 8)
                {
                    old_kind = "one-tile path";
                    run_old(dx, M, oo, &xf8);
                    bad_old += oo != o0;
                }
                else
                {
                    old_kind = "token alone";
                    CK(cudaMemcpy(dxs, dx, static_cast<size_t>(8) * K1 * 2, cudaMemcpyDeviceToDevice));
                    run_old(dxs, 8, oo, &xf8);                                                  // the 8-token table is tile 0 of the multi-tile table
                    for (int m = 0; m < M; ++m)
                    {
                        std::vector<uint16_t> os;
                        CK(cudaMemcpy(dxs, dx + static_cast<size_t>(m) * K1, static_cast<size_t>(K1) * 2, cudaMemcpyDeviceToDevice));
                        run_old(dxs, 1, os, nullptr);
                        bad_old += memcmp(os.data(), &o0[static_cast<size_t>(m) * NOUT], NOUT * 2) != 0;
                    }
                }
                bad_xf = xf8 != xfm;
            }
            // (d) the CPU double reference over the plan's routes
            std::vector<double> max_err(M, 0.0); std::vector<long> nan_cnt(M, 0);
            par_for(M, [&](int m)
            {
                const Ref r = ref_fc1(sgw, suw, &xr[static_cast<size_t>(m) * K1], 10.0);
                std::vector<double> ysum(NOUT, 0.0), yS(NOUT, 0.0), y, S, sref, sS;
                ref_fc2(sdw, r.act, sref, sS);
                for (int k = 0; k < TK; ++k)
                {
                    const int e = route[m][k];
                    if (e < 0 || e >= NREAL) { max_err[m] = 1e9; continue; }
                    const Ref re = ref_fc1(deq_g[e], deq_u[e], &xr[static_cast<size_t>(m) * K1], 10.0);
                    ref_fc2(deq_d[e], re.act, y, S);
                    for (int o = 0; o < NOUT; ++o) { ysum[o] += rw[m][k] * y[o]; yS[o] += rw[m][k] * S[o]; }
                }
                for (int o = 0; o < NOUT; ++o)
                {
                    const double ref = ysum[o] + sref[o], Ssum = yS[o] + sS[o];
                    const uint32_t u = static_cast<uint32_t>(o0[static_cast<size_t>(m) * NOUT + o]) << 16; float v; memcpy(&v, &u, 4);
                    nan_cnt[m] += std::isnan(v);
                    const double tol = std::ldexp(Ssum, -10) + std::ldexp(std::fabs(ref), -8);
                    max_err[m] = std::max(max_err[m], std::fabs(v - ref) / tol);
                }
            });
            double max_y = 0; long bad_nan = 0;
            for (int m = 0; m < M; ++m) { max_y = std::max(max_y, max_err[m]); bad_nan += nan_cnt[m]; }
            const bool ok = max_y <= 1.0 && !bad_nan && !bad_route && !bad_w && !pdiff && !bad_old && !bad_xf && same_run && same_nt;
            printf("multitile NE=%d TK=%d %s M=%2d NT=%d: slots %3d tiles %3d (max %d/slot), plan==plan_ref %s, routes wrong %d, weight sums wrong %d, tile-0 table==build_xfrag %s, "
                   "%s mismatches %d, 2 runs %s, NT=%d %s, out NaN %ld, max err/tol %.3f -> %s\n",
                   NE, TK, skew ? "skew" : "even", M, NT0, pl.n_slots, pl.n_tiles, max_tiles, pdiff ? pdiff : "OK", bad_route, bad_w, bad_xf ? "FAIL" : "OK",
                   old_kind, bad_old, same_run ? "OK" : "FAIL", NT1, NT1 ? (same_nt ? "OK" : "FAIL") : "n/a", bad_nan, max_y, ok ? "OK" : "FAIL");
            fails += !ok;
            last_M = M;
        }
    }
    // controls on the last state (M = 32, skew): a NaN ypart tile must poison the output; a corrupted plan copy must be reported
    {
        std::vector<uint8_t> hp(PLAN_B); CK(cudaMemcpy(hp.data(), dplan, PLAN_B, cudaMemcpyDeviceToHost));
        PM& pl = *reinterpret_cast<PM*>(hp.data());
        const bool clean = mt_plan_diff<TK>(pl, static_cast<int>(last_M)) == nullptr;
        pl.tok_slot[3][1] ^= 1;
        const bool c_plan = mt_plan_diff<TK>(pl, static_cast<int>(last_M)) != nullptr;
        pl.tok_slot[3][1] ^= 1; pl.tile_w[0][0] += 0.25f;
        const bool c_plan2 = mt_plan_diff<TK>(pl, static_cast<int>(last_M)) != nullptr;
        poison(); mt_run<NE, TK>(B, static_cast<int>(last_M), dsr::MT_MAX / 8);                         // a clean run, then the same combine with ypart tile 0 poisoned
        std::vector<uint16_t> ob(static_cast<size_t>(last_M) * NOUT); CK(cudaMemcpy(ob.data(), dfin, ob.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemset(dy, 0xff, static_cast<size_t>(8) * NOUT * 2));                                    // tile 0 of ypart (fp16 NaN; the combine's f2bf maps a NaN sum to a wrong finite value)
        mt_combine<TK>(B, static_cast<int>(last_M)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> oc(ob.size()); CK(cudaMemcpy(oc.data(), dfin, oc.size() * 2, cudaMemcpyDeviceToHost));
        long nn = 0; for (size_t i = 0; i < oc.size(); ++i) nn += oc[i] != ob[i];
        printf("multitile NE=%d TK=%d controls: clean plan diff-free %s, corrupted tok_slot reported %s, corrupted tile_w reported %s, poisoned ypart tile 0 changes the output (%ld of %zu values) %s\n",
               NE, TK, clean ? "OK" : "FAIL", c_plan ? "OK" : "FAIL", c_plan2 ? "OK" : "FAIL", nn, oc.size(), nn ? "OK" : "FAIL");
        fails += !(clean && c_plan && c_plan2 && nn);
    }
    cudaFree(d1); cudaFree(d2); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dgate); cudaFree(dbias);
    cudaFree(dx); cudaFree(dxs); cudaFree(dxf); cudaFree(dxo); cudaFree(dlog); cudaFree(dplan); cudaFree(dplo); cudaFree(dact); cudaFree(dy); cudaFree(dsact); cudaFree(dsout); cudaFree(dfin);
    cudaFree(dacto); cudaFree(dyo); cudaFree(dsacto); cudaFree(dsouto); cudaFree(dfino);
    return fails;
}

// k_plan_mt alone, on logits written straight to the device, with ALL experts in play (the chain test only has 8 real experts, ids 0..7 = warp 0 of the
// plan's scans): spread routes (up to M*TK distinct experts over every warp), clustered routes (a shared logit pattern + noise: few experts, many tokens each,
// nt up to 4), and routes of one expert that straddle tile edges. The plan must equal plan_ref of its own routes; the routes the CPU top-TK of the same logits.
template <int NE, int TK> static int check_plan_mt()
{
    constexpr int MT = dsr::MT_MAX;
    using PM = dsr::PlanMT<TK, MT>;
    std::mt19937_64 g(5);
    float *dlog, *dbias; PM* dplan;
    CK(cudaMalloc(&dlog, static_cast<size_t>(MT) * NE * 4)); CK(cudaMalloc(&dbias, NE * 4)); CK(cudaMalloc(&dplan, sizeof(PM)));
    std::vector<float> zero(NE, 0.f);
    CK(cudaMemcpy(dbias, zero.data(), NE * 4, cudaMemcpyHostToDevice));
    int fails = 0, trials = 0, bad_plan = 0, bad_route = 0, bad_w = 0, max_slots = 0, max_nt = 0, spread_warps = 0;
    std::normal_distribution<float> nd(0.f, 1.5f), noise(0.f, 0.4f);
    for (int kind = 0; kind < 3; ++kind)                      // 0 spread, 1 clustered (pattern + noise), 2 two hot experts per token + the rest spread
        for (int M : {1, 7, 8, 9, 16, 17, 24, 31, 32})
            for (int rep = 0; rep < 40; ++rep)
            {
                std::vector<float> lg(static_cast<size_t>(M) * NE), pat(NE);
                for (auto& v : pat) v = nd(g);
                for (int m = 0; m < M; ++m)
                    for (int e = 0; e < NE; ++e)
                    {
                        float v = kind == 0 ? nd(g) : pat[e] + noise(g);
                        if (kind == 2) v = (e == 3 || e == NE - 5) ? 6.f : nd(g);
                        lg[static_cast<size_t>(m) * NE + e] = v;
                    }
                CK(cudaMemset(dplan, 0xff, sizeof(PM)));
                CK(cudaMemcpy(dlog, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
                dsr::k_plan_mt<NE, TK, MT><<<1, dsr::PLAN_THREADS>>>(dlog, dbias, M, dplan);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                std::vector<uint8_t> hp(sizeof(PM)); CK(cudaMemcpy(hp.data(), dplan, sizeof(PM), cudaMemcpyDeviceToHost));
                const PM& pl = *reinterpret_cast<const PM*>(hp.data());
                ++trials;
                if (mt_plan_diff<TK>(pl, M)) ++bad_plan;
                for (int m = 0; m < M; ++m)
                {
                    std::vector<std::pair<double, int>> key;
                    for (int e = 0; e < NE; ++e) { const double x = lg[static_cast<size_t>(m) * NE + e]; key.push_back({std::sqrt(x > 20 ? x : std::log1p(std::exp(x))), e}); }
                    std::sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                    std::vector<int> cpu, got; float wsum = 0;
                    for (int k = 0; k < TK; ++k) { cpu.push_back(key[k].second); got.push_back(pl.tok_expert[m][k]); wsum += pl.tok_w[m][k]; }
                    std::sort(cpu.begin(), cpu.end()); std::sort(got.begin(), got.end());
                    bad_route += cpu != got && !(key[TK - 1].first - key[TK].first < 1e-5);
                    bad_w += std::fabs(wsum - 1.5f) > 1e-4f;
                }
                max_slots = std::max(max_slots, pl.n_slots);
                for (int s = 0; s < pl.n_slots; ++s) max_nt = std::max(max_nt, pl.slot_nt[s]);
                unsigned wm = 0; for (int s = 0; s < pl.n_slots; ++s) wm |= 1u << (pl.slot_expert[s] / 32);
                spread_warps = std::max(spread_warps, __builtin_popcount(wm));
            }
    fails = bad_plan + bad_route + bad_w;
    printf("multitile plan NE=%d TK=%d: %d trials (spread / clustered / hot-expert logits, M 1..32, up to %d distinct experts in %d warps' id ranges, max %d tiles per expert): "
           "plan != plan_ref %d, routes wrong %d, weight sums wrong %d -> %s\n", NE, TK, trials, max_slots, spread_warps, max_nt, bad_plan, bad_route, bad_w, fails ? "FAIL" : "OK");
    cudaFree(dlog); cudaFree(dbias); cudaFree(dplan);
    return fails;
}

// ------------------------------------------------------------------------------------------ PDL chain vs the plain chain
// NL = 8 back-to-back "layers" of one random 8-expert model, each with its own x (fresh routes) and its own fin, one scratch set shared by all (the adapter's Scratch).
// Reference = the plain launches (layer_plain, default stream). Then the scratch is poisoned (0xff = NaN: a PDL kernel that reads an unwritten value shows) and the PDL
// chain (layer_pdl) runs the 8 layers on a non-blocking stream, and once more as a replayed CUDA graph; all fin must equal the reference bit for bit.
// Per (bias mode, M) over `rounds` rounds, each with new random x for every layer. Bias modes: normal / skew (experts 0 and 1 +8: every token routes to both) / alternate (layers
// alternate between the two, so the slot count and the plan change from layer to layer). In odd rounds a ~60 us spin kernel runs first, so the host enqueues the whole chain
// before the GPU gets to it (the PDL launches then sit resident behind their predecessors).
template <int NE, int TK> static int check_pdl(int rounds)
{
    constexpr int NREAL = 8, MT = dsr::MT_MAX, NL = 8;
    static_assert(NE >= NREAL && TK <= NREAL, "pdl test");
    using PM = dsr::PlanMT<TK, MT>; using PO = dsr::PlanT<TK>;
    std::mt19937_64 g(31);
    std::vector<uint8_t> h1, h2;
    for (int e = 0; e < NREAL; ++e)
    {
        const Matrix gt = rand_matrix(g, I, K1, 119, 124), up = rand_matrix(g, I, K1, 119, 124), dn = rand_matrix(g, NOUT, I, 117, 122);
        auto a = mx4pack::pack_fc1(gt.packed.data(), gt.e8.data(), up.packed.data(), up.e8.data(), I, K1);
        auto b = mx4pack::pack_matrix(dn.packed.data(), dn.e8.data(), NOUT, I);
        h1.insert(h1.end(), a.begin(), a.end());
        h2.insert(h2.end(), b.begin(), b.end());
    }
    const M8 sg = rand_m8(g, I, K1, 118, 123, 7, 126, 130), su = rand_m8(g, I, K1, 118, 123, 5, 126, 130), sd = rand_m8(g, NOUT, I, 117, 122);
    const std::vector<uint8_t> pg = pack8(sg), pu = pack8(su), pd = pack8(sd);
    std::vector<__nv_bfloat16> hgate(static_cast<size_t>(NE) * K1);
    std::normal_distribution<float> gn(0.f, 0.05f);
    for (auto& v : hgate) v = __float2bfloat16_rn(gn(g));
    uint8_t *d1, *d2, *dg, *du, *dd; __nv_bfloat16* dgate; float *dbias0, *dbias1;
    CK(cudaMalloc(&d1, h1.size())); CK(cudaMalloc(&d2, h2.size())); CK(cudaMalloc(&dg, pg.size())); CK(cudaMalloc(&du, pu.size())); CK(cudaMalloc(&dd, pd.size()));
    CK(cudaMalloc(&dgate, hgate.size() * 2)); CK(cudaMalloc(&dbias0, NE * 4)); CK(cudaMalloc(&dbias1, NE * 4));
    CK(cudaMemcpy(d1, h1.data(), h1.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, pg.data(), pg.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, pu.data(), pu.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, pd.data(), pd.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgate, hgate.data(), hgate.size() * 2, cudaMemcpyHostToDevice));
    for (int skew = 0; skew < 2; ++skew)
    {
        std::vector<float> hb(NE, -1e4f);
        for (int e = 0; e < NREAL; ++e) hb[e] = skew && e < 2 ? 8.f : 0.f;
        CK(cudaMemcpy(skew ? dbias1 : dbias0, hb.data(), NE * 4, cudaMemcpyHostToDevice));
    }
    constexpr size_t PLAN_B = sizeof(PM) > sizeof(PO) ? sizeof(PM) : sizeof(PO);
    constexpr int SMT = MT * TK;
    constexpr size_t XL = static_cast<size_t>(MT) * K1, FL = static_cast<size_t>(MT) * NOUT;       // elements per layer of x and fin
    __nv_bfloat16* dx; u32* dxf; float *dlog, *dsout; char* dplan; __half *dact, *dy, *dsact; uint16_t* dfin;
    CK(cudaMalloc(&dx, XL * 2 * NL)); CK(cudaMalloc(&dxf, static_cast<size_t>(MT / 8) * K1 * 4 * 4)); CK(cudaMalloc(&dlog, static_cast<size_t>(MT) * NE * 4)); CK(cudaMalloc(&dplan, PLAN_B));
    CK(cudaMalloc(&dact, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(SMT) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsact, static_cast<size_t>(MT) * I * 2)); CK(cudaMalloc(&dsout, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMalloc(&dfin, FL * 2 * NL));
    auto poison = [&]()
    {
        CK(cudaMemset(dact, 0xff, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMemset(dy, 0xff, static_cast<size_t>(SMT) * 8 * NOUT * 2));
        CK(cudaMemset(dsact, 0xff, static_cast<size_t>(MT) * I * 2)); CK(cudaMemset(dsout, 0xff, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMemset(dfin, 0xff, FL * 2 * NL));
        CK(cudaMemset(dplan, 0xff, PLAN_B)); CK(cudaMemset(dxf, 0xff, static_cast<size_t>(MT / 8) * K1 * 4 * 4)); CK(cudaMemset(dlog, 0xff, static_cast<size_t>(MT) * NE * 4));
        CK(cudaDeviceSynchronize());
    };
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const char* mode_name[3] = {"normal", "skew", "alternate"};
    int fails = 0;
    for (int mode = 0; mode < 3; ++mode)
    {
        auto layer_buf = [&](int l)
        {
            const float* bias = mode == 1 || (mode == 2 && (l & 1)) ? dbias1 : dbias0;
            return MtBuf{dgate, bias, dx + XL * l, dxf, dlog, dplan, dact, dy, dsact, dsout, dfin + FL * l, d1, d2, dg, du, dd};
        };
        for (int M : {1, 4, 8, 9, 16, 24, 32})
        {
            // the CUDA graph of the 8 PDL layers, captured once (x and fin addresses are fixed; the content is refilled per round)
            cudaGraph_t gr; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int l = 0; l < NL; ++l) layer_pdl<NE, TK>(layer_buf(l), M, st);
            CK(cudaStreamEndCapture(st, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            std::vector<uint16_t> ref(FL * NL), got(FL * NL);
            long bad_stream = 0, bad_graph = 0, nan_ref = 0, n_cmp = 0;
            const size_t N = static_cast<size_t>(M) * NOUT;
            for (int r = 0; r < rounds; ++r)
            {
                for (int l = 0; l < NL; ++l)
                {
                    std::vector<__nv_bfloat16> xb(XL, __float2bfloat16_rn(NAN));       // rows >= M stay NaN: a read of them poisons the result
                    for (size_t i = 0; i < static_cast<size_t>(M) * K1; ++i)
                    {
                        const uint64_t u = g();     // sum of three uniforms (Irwin-Hall), mean 0, sd ~0.45
                        const float a = static_cast<float>(u & 0x1fffff), b = static_cast<float>((u >> 21) & 0x1fffff), c = static_cast<float>((u >> 42) & 0x1fffff);
                        xb[i] = __float2bfloat16_rn((a + b + c) * (1.f / 2097152) - 1.5f);
                    }
                    CK(cudaMemcpy(dx + XL * l, xb.data(), XL * 2, cudaMemcpyHostToDevice));
                }
                poison();
                for (int l = 0; l < NL; ++l) layer_plain<NE, TK>(layer_buf(l), M, 0);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(ref.data(), dfin, FL * 2 * NL, cudaMemcpyDeviceToHost));
                for (int l = 0; l < NL; ++l) for (size_t i = 0; i < N; ++i) nan_ref += (ref[FL * l + i] & 0x7fff) > 0x7f80;
                // PDL chain on the stream
                poison();
                if (r & 1) k_spin<<<1, 32, 0, st>>>(100000);
                for (int l = 0; l < NL; ++l) layer_pdl<NE, TK>(layer_buf(l), M, st);
                CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(got.data(), dfin, FL * 2 * NL, cudaMemcpyDeviceToHost));
                for (int l = 0; l < NL; ++l) bad_stream += memcmp(&got[FL * l], &ref[FL * l], N * 2) != 0;
                // the same chain as a graph replay
                poison();
                CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(got.data(), dfin, FL * 2 * NL, cudaMemcpyDeviceToHost));
                for (int l = 0; l < NL; ++l) bad_graph += memcmp(&got[FL * l], &ref[FL * l], N * 2) != 0;
                n_cmp += NL;
            }
            const bool ok = !bad_stream && !bad_graph && !nan_ref;
            printf("pdl NE=%d TK=%d bias=%-9s M=%2d: %ld layers (%d rounds x %d), fin vs plain chain bitwise: stream mismatching layers %ld, graph %ld, NaN in the reference %ld -> %s\n",
                   NE, TK, mode_name[mode], M, n_cmp, rounds, NL, bad_stream, bad_graph, nan_ref, ok ? "OK" : "FAIL");
            fails += !ok;
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
        }
    }
    CK(cudaStreamDestroy(st));
    cudaFree(d1); cudaFree(d2); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dgate); cudaFree(dbias0); cudaFree(dbias1);
    cudaFree(dx); cudaFree(dxf); cudaFree(dlog); cudaFree(dplan); cudaFree(dact); cudaFree(dy); cudaFree(dsact); cudaFree(dsout); cudaFree(dfin);
    return fails;
}

// ------------------------------------------------------------------------------------------ real-checkpoint fixture
// One main layer with the checkpoint's own weights (ds_moe_fixture_prep.py) on the oracle's ffn_in: the launch sequence of ds41_moe_ops.cuh run
// for each of the 4 TP slices (I = 576), the partials summed on the host the way the all-reduce does. Experts: only those the oracle routed to
// are shipped, as a compact table; k_remap rewrites the plan's slot_expert (the only thing the routed kernels read from it) to table indices.
// Writes <out>/L<layer>.{router_idx,router_w,moe_routed,shared_out,ffn_out,ffn_out_bf16partials}.bin (dtypes: ds_moe_fixture_meta.py), for ds41_compare.py.
static std::vector<uint8_t> slurp(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (fread(v.data(), 1, v.size(), f) != v.size()) { printf("short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    return v;
}
static void spit(const std::string& p, const void* d, size_t n)
{
    FILE* f = fopen(p.c_str(), "wb");
    if (!f || fwrite(d, 1, n, f) != n) { printf("cannot write %s\n", p.c_str()); exit(2); }
    fclose(f);
}
__global__ void k_remap(int* slot_expert, const int* n_slots, const int* map)
{
    const int s = threadIdx.x;
    if (s < *n_slots) slot_expert[s] = map[slot_expert[s]];
}
static float bf2f(uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float f; memcpy(&f, &u, 4); return f; }

static int fixture(const std::string& in, int layer, const std::string& out)
{
    constexpr int NE = 384, TK = 6, NSL = 4, RW = 2304;
    static_assert(I * NSL == RW, "fixture: the 4 TP slices of the 2304 intermediate");
    using P = dsr::PlanT<TK>;
    const std::vector<uint8_t> xv = slurp(in + "/x.bf16");
    const int M = static_cast<int>(xv.size() / (K1 * 2));
    if (M < 1 || M > 8) { printf("fixture: S = %d rows, only 1..8 here\n", M); return 1; }
    const std::vector<uint8_t> gv_ = slurp(in + "/gate.bf16"), bv = slurp(in + "/bias.f32");
    std::vector<int> ids;
    { const std::vector<uint8_t> tv = slurp(in + "/experts.txt"); const std::string t(tv.begin(), tv.end());
      for (size_t p = 0; p < t.size(); p = t.find('\n', p) + 1) { ids.push_back(atoi(t.c_str() + p)); if (t.find('\n', p) == std::string::npos) break; } }
    const int U = static_cast<int>(ids.size());
    std::vector<int> hmap(NE, -1), dmap_h(NE, 0);      // -1 = not in the table (counted); the device map sends those to table entry 0
    for (int j = 0; j < U; ++j) hmap[ids[j]] = dmap_h[ids[j]] = j;
    const std::vector<uint8_t> s1c = slurp(in + "/sh_w1.c"), s1s = slurp(in + "/sh_w1.s"), s3c = slurp(in + "/sh_w3.c"), s3s = slurp(in + "/sh_w3.s"), s2c = slurp(in + "/sh_w2.c"), s2s = slurp(in + "/sh_w2.s");
    constexpr size_t PK = 5898240, SCB = 368640, EXB = 3 * (PK + SCB);
    const size_t B1E = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4pack::GR, B2E = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4pack::GR;
    printf("fixture L%d: S=%d rows, %d experts in the table (fc1 %zu B + fc2 %zu B each, device %.0f MB)\n", layer, M, U, B1E, B2E, U * (B1E + B2E) / 1e6);

    __nv_bfloat16 *dx, *dgate; float *dbias, *dlog, *dsout, *dr; u32* dxf; char* dplan; __half *dact, *dy, *dsact; void* dfin; int* dmap;
    uint8_t *d1, *d2, *dg, *du, *dd;
    const int SM = dsr::MMAX * TK;
    CK(cudaMalloc(&dx, xv.size())); CK(cudaMalloc(&dgate, gv_.size())); CK(cudaMalloc(&dbias, bv.size())); CK(cudaMalloc(&dlog, 8 * NE * 4)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4));
    CK(cudaMalloc(&dplan, sizeof(P))); CK(cudaMalloc(&dact, static_cast<size_t>(SM) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(SM) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsact, 8 * I * 2)); CK(cudaMalloc(&dsout, 8 * NOUT * 4)); CK(cudaMalloc(&dr, 8 * NOUT * 4)); CK(cudaMalloc(&dfin, 8 * NOUT * 2)); CK(cudaMalloc(&dmap, NE * 4));
    CK(cudaMalloc(&d1, U * B1E)); CK(cudaMalloc(&d2, U * B2E)); CK(cudaMalloc(&dg, EBG)); CK(cudaMalloc(&du, EBG)); CK(cudaMalloc(&dd, EBD));
    CK(cudaMemcpy(dx, xv.data(), xv.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dgate, gv_.data(), gv_.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbias, bv.data(), bv.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmap, dmap_h.data(), NE * 4, cudaMemcpyHostToDevice));

    // router once (the same for every slice)
    mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, M, K1, dxf);
    dsr::k_gate_logits<NE><<<NE / 8, 256>>>(dgate, dx, M, dlog);
    dsr::k_plan<NE, TK><<<1, dsr::PLAN_THREADS>>>(dlog, dbias, M, reinterpret_cast<P*>(dplan));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hp(sizeof(P)); CK(cudaMemcpy(hp.data(), dplan, sizeof(P), cudaMemcpyDeviceToHost));
    const P pl = *reinterpret_cast<const P*>(hp.data());
    int outside = 0;
    for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) outside += hmap[pl.tok_expert[m][k]] < 0;
    k_remap<<<1, 64>>>(reinterpret_cast<int*>(dplan + offsetof(P, slot_expert)), reinterpret_cast<const int*>(dplan + offsetof(P, n_slots)), dmap);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if (outside) { printf("  %d routes of this run go to experts the oracle did not pick (their rows will differ)\n", outside); }

    std::vector<double> routed(static_cast<size_t>(M) * NOUT, 0.0), shared(routed), ffn(routed);
    std::vector<uint8_t> raw(EXB);
    FILE* fe = fopen((in + "/experts.bin").c_str(), "rb");
    if (!fe) { printf("cannot open experts.bin\n"); return 2; }
    std::vector<uint8_t> hold1(B1E), hold2(B2E), cp, ce;
    long nan_out = 0;
    for (int r = 0; r < NSL; ++r)
    {
        // shared slice: w1/w3 rows [r*I, (r+1)*I); w2 columns [r*I, (r+1)*I). The scales are one per row and 32 k.
        M8 a{std::vector<uint8_t>(s1c.begin() + static_cast<size_t>(r) * I * K1, s1c.begin() + static_cast<size_t>(r + 1) * I * K1), std::vector<uint8_t>(s1s.begin() + static_cast<size_t>(r) * I * (K1 / 32), s1s.begin() + static_cast<size_t>(r + 1) * I * (K1 / 32)), I, K1};
        M8 b{std::vector<uint8_t>(s3c.begin() + static_cast<size_t>(r) * I * K1, s3c.begin() + static_cast<size_t>(r + 1) * I * K1), std::vector<uint8_t>(s3s.begin() + static_cast<size_t>(r) * I * (K1 / 32), s3s.begin() + static_cast<size_t>(r + 1) * I * (K1 / 32)), I, K1};
        M8 c{std::vector<uint8_t>(static_cast<size_t>(NOUT) * I), std::vector<uint8_t>(static_cast<size_t>(NOUT) * (I / 32)), NOUT, I};
        for (int o = 0; o < NOUT; ++o)
        {
            memcpy(c.codes.data() + static_cast<size_t>(o) * I, s2c.data() + static_cast<size_t>(o) * RW + r * I, I);
            memcpy(c.sc.data() + static_cast<size_t>(o) * (I / 32), s2s.data() + static_cast<size_t>(o) * (RW / 32) + r * (I / 32), I / 32);
        }
        { const auto pa = pack8(a), pb = pack8(b), pc = pack8(c);
          CK(cudaMemcpy(dg, pa.data(), pa.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, pb.data(), pb.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, pc.data(), pc.size(), cudaMemcpyHostToDevice)); }
        for (int j = 0; j < U; ++j)
        {
            fseek(fe, static_cast<long>(j * EXB), SEEK_SET);
            if (fread(raw.data(), 1, EXB, fe) != EXB) { printf("short read experts.bin\n"); return 2; }
            const uint8_t *w1p = raw.data(), *w1s = w1p + PK, *w3p = w1s + SCB, *w3s = w3p + PK, *w2p = w3s + SCB, *w2s = w2p + PK;
            const auto f1 = mx4pack::pack_fc1(w1p + static_cast<size_t>(r) * I * (K1 / 2), w1s + static_cast<size_t>(r) * I * (K1 / 32),
                                              w3p + static_cast<size_t>(r) * I * (K1 / 2), w3s + static_cast<size_t>(r) * I * (K1 / 32), I, K1);
            mx4pack::slice_cols(cp, ce, w2p, w2s, NOUT, RW, r * I, I);
            const auto f2 = mx4pack::pack_matrix(cp.data(), ce.data(), NOUT, I);
            if (f1.size() != B1E || f2.size() != B2E) { printf("pack size mismatch %zu/%zu %zu/%zu\n", f1.size(), B1E, f2.size(), B2E); return 2; }
            CK(cudaMemcpy(d1 + j * B1E, f1.data(), B1E, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2 + j * B2E, f2.data(), B2E, cudaMemcpyHostToDevice));
        }
        CK(cudaMemset(dact, 0xff, static_cast<size_t>(SM) * 8 * I * 2)); CK(cudaMemset(dy, 0xff, static_cast<size_t>(SM) * 8 * NOUT * 2));    // NaN sentinels
        CK(cudaMemset(dsact, 0xff, 8 * I * 2)); CK(cudaMemset(dsout, 0xff, 8 * NOUT * 4)); CK(cudaMemset(dfin, 0xff, 8 * NOUT * 2)); CK(cudaMemset(dr, 0xff, 8 * NOUT * 4));
        dsh::Args SA{dg, du, dd, dxf, (1 << M) - 1, 10.f, dsact, dsout};
        dsh::k_sh_fc1<K1, I, 8, 1, 2><<<I / 16, 8 * 32>>>(SA);
        dsh::k_sh_fc2<I, NOUT, 8, 3><<<NOUT / 16 / 8, 8 * 32>>>(SA);
        mx4::Args RA{d1, d2, dxf, reinterpret_cast<const int*>(dplan + offsetof(P, slot_expert)), reinterpret_cast<const int*>(dplan + offsetof(P, slot_mask)),
                     reinterpret_cast<const float*>(dplan + offsetof(P, slot_w)), reinterpret_cast<const int*>(dplan + offsetof(P, n_slots)), M * TK, 10.f, dact, dy};
        mx4::k_fc1<K1, I, 8, 1, 2, false><<<M * TK * (I / 16), 8 * 32>>>(RA);
        mx4::k_fc2<I, NOUT, 8, 2, false><<<M * TK * (NOUT / 16 / 8), 8 * 32>>>(RA);
        dsh::CombineArgsT<TK> CR{reinterpret_cast<const P*>(dplan), dy, nullptr, dr, M};          // routed alone, fp32
        dsh::k_combine<NOUT, 128, gv::O_F32, TK><<<M * NOUT / (8 * 128), 128>>>(CR);
        dsh::CombineArgsT<TK> CA{reinterpret_cast<const P*>(dplan), dy, dsout, dfin, M};          // routed + shared, bf16: what the rank hands to the all-reduce
        dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<M * NOUT / (8 * 128), 128>>>(CA);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> hr(static_cast<size_t>(M) * NOUT), hs(hr.size());
        std::vector<uint16_t> hf(hr.size());
        CK(cudaMemcpy(hr.data(), dr, hr.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), dsout, hs.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hf.data(), dfin, hf.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hr.size(); ++i)
        {
            const float f = bf2f(hf[i]);
            if (!std::isfinite(hr[i]) || !std::isfinite(hs[i]) || !std::isfinite(f)) ++nan_out;
            routed[i] += hr[i]; shared[i] += hs[i]; ffn[i] += f;
        }
        printf("  slice %d done (n_slots %d)\n", r, pl.n_slots);
    }
    fclose(fe);
    // moe_routed / shared_out are fp32 (the engine keeps both in fp32 until the combine); ffn_out = one bf16 round of routed + shared (what an fp32
    // all-reduce of the rank partials would give); ffn_out_bf16partials = the sum of the 4 ranks' bf16 partials, rounded once more (a bf16 all-reduce)
    std::vector<float> o_r(routed.size()), o_s(routed.size());
    std::vector<uint16_t> o_f(routed.size()), o_p(routed.size());
    auto bf = [](float v) { const __nv_bfloat16 h = __float2bfloat16_rn(v); uint16_t u; memcpy(&u, &h, 2); return u; };
    for (size_t i = 0; i < routed.size(); ++i) { o_r[i] = static_cast<float>(routed[i]); o_s[i] = static_cast<float>(shared[i]); o_f[i] = bf(static_cast<float>(routed[i] + shared[i])); o_p[i] = bf(static_cast<float>(ffn[i])); }
    std::vector<int> ridx(static_cast<size_t>(M) * TK);
    std::vector<float> rw(ridx.size());
    for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) { ridx[m * TK + k] = pl.tok_expert[m][k]; rw[m * TK + k] = pl.tok_w[m][k]; }
    const std::string L = out + "/L" + std::to_string(layer) + ".";
    spit(L + "router_idx.bin", ridx.data(), ridx.size() * 4); spit(L + "router_w.bin", rw.data(), rw.size() * 4);
    spit(L + "moe_routed.bin", o_r.data(), o_r.size() * 4); spit(L + "shared_out.bin", o_s.data(), o_s.size() * 4); spit(L + "ffn_out.bin", o_f.data(), o_f.size() * 2);
    spit(L + "ffn_out_bf16partials.bin", o_p.data(), o_p.size() * 2);
    printf("fixture L%d: non-finite values %ld, routes outside the oracle's expert set %d, outputs in %s\n", layer, nan_out, outside, out.c_str());
    return nan_out != 0;
}

// ------------------------------------------------------------------------------------------ bench
__global__ void k_fill(uint8_t* p, size_t n, u32 seed)
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
    {
        u32 h = static_cast<u32>(i) * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = static_cast<uint8_t>(h & 0x77u);     // sign clear, exponent field < 15: no NaN codes
    }
}
__global__ void k_fix_scales(uint8_t* p, size_t nrec)   // scale bytes of every record into 2^-7 .. 2^-4
{
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < nrec * 16; i += 256ull * gridDim.x)
    {
        uint8_t* b = p + (i / 16) * dsh::GR + 512 + (i % 16);
        *b = static_cast<uint8_t>(120 + (*b & 3));
    }
}
__global__ void k_stream(const uint4* p, size_t n, u32* sink)
{
    u32 x = 0;
    size_t i = blockIdx.x * 256ull + threadIdx.x;
    const size_t st = 256ull * gridDim.x;
    for (; i + 3 * st < n; i += 4 * st)
    {
        u32 a[4][4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) mx4::ld4(p + i + u * st, a[u]);
        #pragma unroll
        for (int u = 0; u < 4; ++u) x ^= a[u][0] ^ a[u][1] ^ a[u][2] ^ a[u][3];
    }
    for (; i < n; i += st) { u32 a[4]; mx4::ld4(p + i, a); x ^= a[0] ^ a[1] ^ a[2] ^ a[3]; }
    if (x == 0x9e3779b9u) sink[0] = x;
}

__global__ void k_stream3(const uint4* p0, const uint4* p1, const uint4* p2, size_t n0, size_t n1, size_t n2, u32* sink)   // one launch over three regions
{
    u32 x = 0;
    const size_t tot = n0 + n1 + n2, st = 256ull * gridDim.x;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < tot; i += st)
    {
        u32 a[4];
        mx4::ld4(i < n0 ? p0 + i : i < n0 + n1 ? p1 + (i - n0) : p2 + (i - n0 - n1), a);
        x ^= a[0] ^ a[1] ^ a[2] ^ a[3];
    }
    if (x == 0x9e3779b9u) sink[0] = x;
}

struct Var { std::string name; std::function<void(const dsh::Args&, cudaStream_t)> run; std::function<void(const dsh::Args&, cudaStream_t)> run_pdl; };
template <int KS, int D, int CW = 1> static Var v_fc1()
{
    return {"sh_fc1 KS=" + std::to_string(KS) + (CW > 1 ? " CW=" + std::to_string(CW) : std::string()) + " D=" + std::to_string(D),
            [](const dsh::Args& A, cudaStream_t s) { dsh::k_sh_fc1<K1, I, KS, CW, D><<<I / 16 / CW, KS * CW * 32, 0, s>>>(A); }, nullptr};
}
template <int W, int D> static Var v_fc2()
{
    return {"sh_fc2 W=" + std::to_string(W) + " D=" + std::to_string(D),
            [](const dsh::Args& A, cudaStream_t s) { dsh::k_sh_fc2<I, NOUT, W, D><<<NOUT / 16 / W, W * 32, 0, s>>>(A); },
            [](const dsh::Args& A, cudaStream_t s) {
                cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(NOUT / 16 / W); cfg.blockDim = dim3(W * 32); cfg.stream = s;
                cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
                cfg.attrs = at; cfg.numAttrs = 1;
                CK(cudaLaunchKernelEx(&cfg, dsh::k_sh_fc2<I, NOUT, W, D>, A));
            }};
}

static int bench(int sets)
{
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    uint8_t *pg, *pu, *pd; u32* sink;
    CK(cudaMalloc(&pg, EBG * sets)); CK(cudaMalloc(&pu, EBG * sets)); CK(cudaMalloc(&pd, EBD * sets)); CK(cudaMalloc(&sink, 64));
    k_fill<<<nsm * 4, 256>>>(pg, EBG * sets, 1u); k_fill<<<nsm * 4, 256>>>(pu, EBG * sets, 2u); k_fill<<<nsm * 4, 256>>>(pd, EBD * sets, 3u);
    k_fix_scales<<<nsm * 4, 256>>>(pg, EBG * sets / dsh::GR); k_fix_scales<<<nsm * 4, 256>>>(pu, EBG * sets / dsh::GR); k_fix_scales<<<nsm * 4, 256>>>(pd, EBD * sets / dsh::GR);
    u32* dxf; __nv_bfloat16* dx; __half* dact; float* dout;
    CK(cudaMalloc(&dx, K1 * 2)); CK(cudaMalloc(&dxf, K1 / 16 * 64 * 4)); CK(cudaMalloc(&dact, 8 * I * 2)); CK(cudaMalloc(&dout, 8 * NOUT * 4));
    std::vector<__nv_bfloat16> xb(K1); std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 0.5f);
    for (auto& v : xb) v = __float2bfloat16_rn(nd(g));
    CK(cudaMemcpy(dx, xb.data(), K1 * 2, cudaMemcpyHostToDevice));
    mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(dx, 1, K1, dxf);
    CK(cudaMemset(dact, 0, 8 * I * 2));
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const int ITERS = 20 * sets, WARM = 2 * sets;
    auto args = [&](int it) { const int s = it % sets; return dsh::Args{pg + EBG * s, pu + EBG * s, pd + EBD * s, dxf, 1, 10.f, dact, dout}; };
    auto time_us = [&](const std::function<void(int)>& f) {   // min of 3 repetitions (a rep can be perturbed by other work on the GPU)
        for (int i = 0; i < WARM; ++i) f(i);
        CK(cudaStreamSynchronize(st));
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < ITERS; ++i) f(i);
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms * 1000.0 / ITERS);
        }
        return best;
    };
    const double BY1 = 2.0 * EBG, BY2 = static_cast<double>(EBD);
    const double SPEC = 273.0, FLEET = 255.8;
    auto line = [&](const std::string& n, double us, double bytes) {
        const double gbs = bytes / us / 1e3;
        printf("  %-26s %8.1f us  %7.1f GB/s  %5.1f%% of 273  %5.1f%% of 255.8\n", n.c_str(), us, gbs, 100 * gbs / SPEC, 100 * gbs / FLEET);
    };
    printf("device %s, %d SMs, %zu MB L2; geometry K1=%d I=%d NOUT=%d; shared expert per layer: gate+up %.2f MB + down %.2f MB = %.2f MB (%.4f bits/weight)\n",
           pr.name, nsm, static_cast<size_t>(pr.l2CacheSize) >> 20, K1, I, NOUT, BY1 / 1e6, BY2 / 1e6, (BY1 + BY2) / 1e6, (BY1 + BY2) * 8 / (3.0 * I * K1));
    printf("pool %d layers (%.0f MB), %d iterations\n", sets, (2 * EBG + EBD) * sets / 1e6, ITERS);

    auto stream = [&](int it, int mult) {     // all three streams in one launch (three ~3 MB launches are dominated by launch and ramp)
        const int s = it % sets;
        k_stream3<<<nsm * mult, 256, 0, st>>>(reinterpret_cast<const uint4*>(pg + EBG * s), reinterpret_cast<const uint4*>(pu + EBG * s),
                                                reinterpret_cast<const uint4*>(pd + EBD * s), EBG / 16, EBG / 16, EBD / 16, sink);
    };
    double best_stream = 1e30;
    printf("plain stream read of the same bytes in one launch (the ceiling):\n");
    for (int mult : {2, 4, 8, 16}) { const double us = time_us([&](int it) { stream(it, mult); }); line("stream x" + std::to_string(mult), us, BY1 + BY2); best_stream = std::min(best_stream, us); }
    printf("the same bytes as two launches (gate+up streams are 6.08 MB, down 3.04 MB): fc1-sized and fc2-sized plain reads:\n");
    {
        double b1s = 1e30, b2s = 1e30;
        for (int mult : {4, 8, 16})
        {
            b1s = std::min(b1s, time_us([&](int it) { const int s = it % sets; k_stream3<<<nsm * mult, 256, 0, st>>>(reinterpret_cast<const uint4*>(pg + EBG * s), reinterpret_cast<const uint4*>(pu + EBG * s), reinterpret_cast<const uint4*>(pu), EBG / 16, EBG / 16, 0, sink); }));
            b2s = std::min(b2s, time_us([&](int it) { const int s = it % sets; k_stream3<<<nsm * mult, 256, 0, st>>>(reinterpret_cast<const uint4*>(pd + EBD * s), reinterpret_cast<const uint4*>(pd), reinterpret_cast<const uint4*>(pd), EBD / 16, 0, 0, sink); }));
        }
        line("stream ceiling, fc1 bytes", b1s, BY1); line("stream ceiling, fc2 bytes", b2s, BY2);
    }

    std::vector<Var> f1 = {v_fc1<4, 1>(), v_fc1<4, 2>(), v_fc1<8, 1>(), v_fc1<8, 2>(), v_fc1<8, 4>(), v_fc1<10, 2>(), v_fc1<16, 1>(), v_fc1<16, 2>(), v_fc1<20, 1>(), v_fc1<20, 2>()};
    std::vector<Var> f2 = {v_fc2<4, 2>(), v_fc2<4, 3>(), v_fc2<8, 1>(), v_fc2<8, 2>(), v_fc2<8, 3>(), v_fc2<8, 6>(), v_fc2<5, 3>(), v_fc2<10, 3>(), v_fc2<10, 6>()};
    printf("shared fc1 (gate + up, %.2f MB):\n", BY1 / 1e6);
    double t1 = 1e30; Var* b1 = nullptr;
    for (auto& v : f1) { const double us = time_us([&](int it) { v.run(args(it), st); }); line(v.name, us, BY1); if (us < t1) { t1 = us; b1 = &v; } }
    printf("shared fc2 (down, %.2f MB):\n", BY2 / 1e6);
    double t2 = 1e30; Var* b2 = nullptr;
    for (auto& v : f2) { const double us = time_us([&](int it) { v.run(args(it), st); }); line(v.name, us, BY2); if (us < t2) { t2 = us; b2 = &v; } }
    printf("best pair stream-ordered: %s + %s\n", b1->name.c_str(), b2->name.c_str());
    const double us = time_us([&](int it) { b1->run(args(it), st); b2->run(args(it), st); });
    line("shared fc1+fc2", us, BY1 + BY2);
    double best_pdl = 1e30; std::string best_pdl_name;
    for (auto& v : f2)
    {
        const double upd = time_us([&](int it) { b1->run(args(it), st); v.run_pdl(args(it), st); });
        line(b1->name + " + " + v.name + " PDL", upd, BY1 + BY2);
        if (upd < best_pdl) { best_pdl = upd; best_pdl_name = b1->name + " + " + v.name; }
    }
    // combine: 6 routes + shared, M = 1 and 8 (ypart / shared are L2-resident here: the routed fc2 just wrote them)
    {
        dsr::Plan P; memset(&P, 0, sizeof P);
        for (int m = 0; m < dsr::MMAX; ++m) for (int k = 0; k < 8; ++k) { P.tok_slot[m][k] = k < TOPK ? (m * TOPK + k) % 30 : -1; P.tok_col[m][k] = m; }
        dsr::Plan* dplan; __half* dyp; void* dfin;
        CK(cudaMalloc(&dplan, sizeof P)); CK(cudaMemcpy(dplan, &P, sizeof P, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dyp, 30ull * 8 * NOUT * 2)); CK(cudaMemset(dyp, 0, 30ull * 8 * NOUT * 2)); CK(cudaMalloc(&dfin, 8 * NOUT * 4));
        for (int M : {1, 8})
        {
            dsh::CombineArgs CA{dplan, dyp, dout, dfin, M};
            const double c = time_us([&](int) { dsh::k_combine<NOUT, 128, gv::O_BF16><<<M * NOUT / (8 * 128), 128, 0, st>>>(CA); });
            const double c64 = time_us([&](int) { dsh::k_combine<NOUT, 64, gv::O_BF16><<<M * NOUT / (8 * 64), 64, 0, st>>>(CA); });
            printf("combine M=%d bf16 out: %.1f us (128 thr/CTA), %.1f us (64 thr/CTA)\n", M, c, c64);
        }
    }
    printf("summary (min of 3 reps each):\n");
    line("stream ceiling", best_stream, BY1 + BY2);
    line("shared fc1+fc2 stream-ordered", us, BY1 + BY2);
    line("shared fc1+fc2 PDL: " + best_pdl_name, best_pdl, BY1 + BY2);
    printf("  shared expert: %.1f us stream-ordered, %.1f us with PDL; ceiling %.1f us (PDL = %.1f%% of the ceiling's rate)\n", us, best_pdl, best_stream, 100 * best_stream / best_pdl);
    return 0;
}

// ------------------------------------------------------------------------------------------ multi-tile bench
// ds_shared_test mtbench [ROUTES_DIR]: the MoE layer at M = 9, 16, 24, 32 through the multi-tile kernels, and the one-tile path at M = 1 and 8, on random
// weights, cold (every pool is far larger than L2 and is rotated per iteration), stream-ordered launches, min of 3 reps. Two route sets:
//   live  the router runs on random logits restricted to NLIVE = 48 experts (uniform routes: about the most distinct experts a step touches)
//   real  routes of a 128-token prompt from the reference model (ROUTES_DIR/L{0,2,20,39}.router_idx.bin, int32 [128][6]), tokens 0..M-1, planned on the host (plan_ref); the plan's experts are
//         folded onto the POOL experts that exist on the device, another fold offset per timed iteration. The router and the shared expert do not depend on the
//         routes, so layer(real) = chain(live) - routed(live) + routed(real), routed = fc1 + fc2 + combine.
// Numbers taken while another engine shares the GPU are not comparable with a quiet GPU.
struct MtTimes { double xf = 0, gate = 0, plan = 0, router = 0, shared = 0, fc1 = 0, fc2 = 0, routed = 0, chain = 0; int slots = 0, tiles = 0; };

template <int NE, int TK> static int mtbench(const std::string& dir)
{
    constexpr int MT = dsr::MT_MAX, POOL = 56, NLIVE = 48, GSETS = 8, SSETS = 6, NV = 8, ITERS = 120;
    using PM = dsr::PlanMT<TK, MT>; using PO = dsr::PlanT<TK>;
    constexpr size_t EB1 = static_cast<size_t>(I / 16) * (K1 / 64) * 2 * mx4::GR;      // one expert's fc1 bytes [I/16][K1/64][2][GR]
    constexpr size_t EB2 = static_cast<size_t>(NOUT / 16) * (I / 64) * mx4::GR;        // its fc2 bytes [NOUT/16][I/64][GR]
    constexpr size_t PLAN_B = sizeof(PM) > sizeof(PO) ? sizeof(PM) : sizeof(PO);
    constexpr int SMT = MT * TK, SMO = dsr::MMAX * TK;
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    uint8_t *w1, *w2, *sg, *su, *sd;
    CK(cudaMalloc(&w1, EB1 * POOL)); CK(cudaMalloc(&w2, EB2 * POOL));
    CK(cudaMalloc(&sg, EBG * SSETS)); CK(cudaMalloc(&su, EBG * SSETS)); CK(cudaMalloc(&sd, EBD * SSETS));
    k_fill<<<nsm * 4, 256>>>(w1, EB1 * POOL, 11u); k_fill<<<nsm * 4, 256>>>(w2, EB2 * POOL, 12u);
    k_fill<<<nsm * 4, 256>>>(sg, EBG * SSETS, 1u); k_fill<<<nsm * 4, 256>>>(su, EBG * SSETS, 2u); k_fill<<<nsm * 4, 256>>>(sd, EBD * SSETS, 3u);
    k_fix_scales<<<nsm * 4, 256>>>(sg, EBG * SSETS / dsh::GR); k_fix_scales<<<nsm * 4, 256>>>(su, EBG * SSETS / dsh::GR); k_fix_scales<<<nsm * 4, 256>>>(sd, EBD * SSETS / dsh::GR);
    std::mt19937_64 g(21);
    std::vector<__nv_bfloat16> hgate(static_cast<size_t>(NE) * K1), hx(static_cast<size_t>(MT) * K1);
    { std::normal_distribution<float> gn(0.f, 0.05f), xn(0.f, 0.5f);
      for (auto& v : hgate) v = __float2bfloat16_rn(gn(g));
      for (auto& v : hx) v = __float2bfloat16_rn(xn(g)); }
    __nv_bfloat16 *dgate, *dx; float *dbias, *dlog, *dsout, *dsouto; u32 *dxf, *dxo; char *dplan, *dplanv, *dplo; __half *dact, *dy, *dsact, *dacto, *dyo, *dsacto; void *dfin, *dfino;
    CK(cudaMalloc(&dgate, hgate.size() * 2 * GSETS)); CK(cudaMalloc(&dx, hx.size() * 2)); CK(cudaMalloc(&dbias, NE * 4));
    for (int s = 0; s < GSETS; ++s) CK(cudaMemcpy(dgate + static_cast<size_t>(s) * hgate.size(), hgate.data(), hgate.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxf, static_cast<size_t>(MT / 8) * K1 * 4 * 4)); CK(cudaMalloc(&dxo, static_cast<size_t>(K1) * 4 * 4));
    CK(cudaMalloc(&dlog, static_cast<size_t>(MT) * NE * 4)); CK(cudaMalloc(&dplan, PLAN_B)); CK(cudaMalloc(&dplanv, PLAN_B * NV)); CK(cudaMalloc(&dplo, PLAN_B));
    CK(cudaMalloc(&dact, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMalloc(&dy, static_cast<size_t>(SMT) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsact, static_cast<size_t>(MT) * I * 2)); CK(cudaMalloc(&dsout, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMalloc(&dfin, static_cast<size_t>(MT) * NOUT * 2));
    CK(cudaMalloc(&dacto, static_cast<size_t>(SMO) * 8 * I * 2)); CK(cudaMalloc(&dyo, static_cast<size_t>(SMO) * 8 * NOUT * 2));
    CK(cudaMalloc(&dsacto, 8 * I * 2)); CK(cudaMalloc(&dsouto, 8 * NOUT * 4)); CK(cudaMalloc(&dfino, 8 * NOUT * 2));
    CK(cudaMemset(dact, 0, static_cast<size_t>(SMT) * 8 * I * 2)); CK(cudaMemset(dy, 0, static_cast<size_t>(SMT) * 8 * NOUT * 2)); CK(cudaMemset(dsact, 0, static_cast<size_t>(MT) * I * 2));
    CK(cudaMemset(dsout, 0, static_cast<size_t>(MT) * NOUT * 4)); CK(cudaMemset(dfin, 0, static_cast<size_t>(MT) * NOUT * 2));
    CK(cudaMemset(dacto, 0, static_cast<size_t>(SMO) * 8 * I * 2)); CK(cudaMemset(dyo, 0, static_cast<size_t>(SMO) * 8 * NOUT * 2));
    CK(cudaMemset(dsacto, 0, 8 * I * 2)); CK(cudaMemset(dsouto, 0, 8 * NOUT * 4)); CK(cudaMemset(dfino, 0, 8 * NOUT * 2));
    std::vector<float> bias_live(NE, -1e4f);
    for (int e = 0; e < NLIVE; ++e) bias_live[e] = 0.f;
    CK(cudaMemcpy(dbias, bias_live.data(), NE * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    printf("mtbench (beside GLM) device %s, %d SMs, L2 %zu MB; NE=%d TK=%d; routed pool %d experts (%.0f MB: fc1 %.2f MB + fc2 %.2f MB each), gate %d sets x %.1f MB, shared %d sets x %.1f MB; free device mem %zu MB of %zu MB\n",
           pr.name, nsm, static_cast<size_t>(pr.l2CacheSize) >> 20, NE, TK, POOL, (EB1 + EB2) * POOL / 1e6, EB1 / 1e6, EB2 / 1e6, GSETS, hgate.size() * 2 / 1e6, SSETS, (2 * EBG + EBD) / 1e6, fr >> 20, tot >> 20);

    const MtBuf B{dgate, dbias, dx, dxf, dlog, dplan, dact, dy, dsact, dsout, dfin, w1, w2, sg, su, sd};
    auto bufs = [&](int it, char* plan)
    {
        MtBuf b = B;
        b.gate = dgate + static_cast<size_t>(it % GSETS) * hgate.size();
        b.sg = sg + EBG * (it % SSETS); b.su = su + EBG * (it % SSETS); b.sd = sd + EBD * (it % SSETS);
        b.plan = plan;
        return b;
    };
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_us = [&](int iters, const std::function<void(int)>& f)     // min of 3 repetitions (GLM runs beside this test)
    {
        for (int i = 0; i < 16; ++i) f(i);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            CK(cudaEventRecord(e0, 0));
            for (int i = 0; i < iters; ++i) f(i);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms * 1000.0 / iters);
        }
        return best;
    };
    auto with_nt = [](int nt, auto&& f)
    {
        if (nt == 2) f(std::integral_constant<int, 2>{}); else if (nt == 3) f(std::integral_constant<int, 3>{}); else f(std::integral_constant<int, 4>{});
    };
    const double SPEC = 273.0, FLEET = 255.8;
    auto gbs = [&](int slots, double us) { return slots * static_cast<double>(EB1 + EB2) / us / 1e3; };

    // ---- live routes
    const int Ms[4] = {9, 16, 24, 32};
    MtTimes live[4];
    printf("\n[live: random logits, %d experts in play] us per call, stream-ordered, cold weights; router = xfrag + gate + plan; shared = fc1 + fc2; routed = fc1 + fc2 + combine\n", NLIVE);
    for (int mi = 0; mi < 4; ++mi)
    {
        const int M = Ms[mi], nt = (M + 7) / 8;
        MtTimes& T = live[mi];
        with_nt(nt, [&](auto c)
        {
            constexpr int NT = decltype(c)::value;
            mt_chain<NE, TK, NT>(B, M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<uint8_t> hp(PLAN_B); CK(cudaMemcpy(hp.data(), dplan, PLAN_B, cudaMemcpyDeviceToHost));
            const PM& pl = *reinterpret_cast<const PM*>(hp.data());
            T.slots = pl.n_slots; T.tiles = pl.n_tiles;
            T.xf = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplan); mx4::build_xfrag_mt<<<(nt * K1 * 4 + 255) / 256, 256>>>(b.x, M, K1, nt, b.xf); });
            T.gate = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplan); dsr::k_gate_logits_mt<NE><<<NE / 8, 256>>>(b.gate, b.x, M, b.log); });
            T.plan = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplan); dsr::k_plan_mt<NE, TK, MT><<<1, dsr::PLAN_THREADS>>>(b.log, b.bias, M, reinterpret_cast<PM*>(b.plan)); });
            T.router = time_us(ITERS, [&](int it) { mt_router<NE, TK>(bufs(it, dplan), M); });
            T.shared = time_us(ITERS, [&](int it) { mt_shared<NT>(bufs(it, dplan), M); });
            T.fc1 = time_us(ITERS, [&](int it) { mt_fc1<TK, NT>(bufs(it, dplan), M); });
            T.fc2 = time_us(ITERS, [&](int it) { mt_fc2<TK, NT>(bufs(it, dplan), M); });
            T.routed = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplan); mt_fc1<TK, NT>(b, M); mt_fc2<TK, NT>(b, M); mt_combine<TK>(b, M); });
            T.chain = time_us(ITERS, [&](int it) { mt_chain<NE, TK, NT>(bufs(it, dplan), M); });
        });
        const double rb = T.fc1 + T.fc2;
        printf("  M=%2d NT=%d slots %3d tiles %3d | xfrag %5.1f gate %6.1f plan %6.1f = router %6.1f | shared %6.1f | fc1 %6.1f fc2 %6.1f (%5.1f GB/s = %4.1f%% of %.1f) | routed %6.1f | layer chain %7.1f us = %5.2f us/token\n",
               M, nt, T.slots, T.tiles, T.xf, T.gate, T.plan, T.router, T.shared, T.fc1, T.fc2, gbs(T.slots, rb), 100 * gbs(T.slots, rb) / FLEET, FLEET, T.routed, T.chain, T.chain / M);
        (void)SPEC;
    }

    // ---- fc1 NT=3/4 study: occupancy (minB 2 = at most 128 regs) and unroll depth (D1 halves the loop body), live routes; the instance with NT smaller than the plan's
    //      tile count is timing-only (it computes the first NT tiles of each slot; live plans have almost only one-tile slots)
    printf("\n[fc1 variants, live routes, cold weights] fc1 us (per-slot us; %% of the %.1f GB/s fleet stream), instance NT, variant D<depth>/minB<CTAs per SM>; bit-identity = act vs the shipped instance\n", FLEET);
    for (int M : {16, 24, 32})
    {
        const int nt = (M + 7) / 8;
        mt_router<NE, TK>(B, M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint8_t> hp(PLAN_B); CK(cudaMemcpy(hp.data(), dplan, PLAN_B, cudaMemcpyDeviceToHost));
        const PM& pl = *reinterpret_cast<const PM*>(hp.data());
        const size_t ACT_B = static_cast<size_t>(SMT) * 8 * I * 2;
        std::vector<uint8_t> base(ACT_B), cur(ACT_B);
        const char* vn[4] = {"D2/b1", "D2/b2", "D1/b1", "D1/b2"};
        printf("  M=%2d plan: slots %d tiles %d\n", M, pl.n_slots, pl.n_tiles);
        for (int inst = nt; inst >= 2; --inst)
        {
            for (int v = 0; v < 4; ++v)
            {
                CK(cudaMemset(dact, 0, ACT_B));
                mt_fc1_any<TK>(inst, v, B, M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(cur.data(), dact, ACT_B, cudaMemcpyDeviceToHost));
                const char* same = "";
                if (inst == nt && v == 0) base = cur; else if (inst == nt) same = memcmp(base.data(), cur.data(), ACT_B) ? " act DIFFERS" : " act bit-identical";
                const double us = time_us(ITERS, [&](int it) { mt_fc1_any<TK>(inst, v, bufs(it, dplan), M); });
                printf("    instance NT=%d %s : fc1 %7.1f us  (%5.2f us/slot, %5.1f GB/s = %4.1f%%)%s%s\n", inst, vn[v], us, us / pl.n_slots, pl.n_slots * static_cast<double>(EB1) / us / 1e3,
                       100 * pl.n_slots * static_cast<double>(EB1) / us / 1e3 / FLEET, inst < nt ? " [timing only]" : "", same);
            }
        }
    }

    // ---- the one-tile path (what rows <= 8 launch), live routes, same pools
    printf("\n[one-tile path, live routes]\n");
    for (int M : {1, 8})
    {
        auto o_router = [&](const MtBuf& b)
        {
            mx4::build_xfrag<<<(K1 / 16 * 64 + 255) / 256, 256>>>(b.x, M, K1, dxo);
            dsr::k_gate_logits<NE><<<NE / 8, 256>>>(b.gate, b.x, M, b.log);
            dsr::k_plan<NE, TK><<<1, dsr::PLAN_THREADS>>>(b.log, b.bias, M, reinterpret_cast<PO*>(dplo));
        };
        auto o_shared = [&](const MtBuf& b)
        {
            dsh::Args SA{b.sg, b.su, b.sd, dxo, (1 << M) - 1, 10.f, dsacto, dsouto};
            dsh::k_sh_fc1<K1, I, 8, 1, 2><<<I / 16, 8 * 32>>>(SA);
            dsh::k_sh_fc2<I, NOUT, 8, 3><<<NOUT / 16 / 8, 8 * 32>>>(SA);
        };
        auto o_fc = [&](const MtBuf& b, bool fc1, bool fc2, bool comb)
        {
            mx4::Args RA{b.w1, b.w2, dxo, reinterpret_cast<const int*>(dplo + offsetof(PO, slot_expert)), reinterpret_cast<const int*>(dplo + offsetof(PO, slot_mask)),
                         reinterpret_cast<const float*>(dplo + offsetof(PO, slot_w)), reinterpret_cast<const int*>(dplo + offsetof(PO, n_slots)), M * TK, 10.f, dacto, dyo};
            if (fc1) mx4::k_fc1<K1, I, 8, 1, 2, false><<<M * TK * (I / 16), 8 * 32>>>(RA);
            if (fc2) mx4::k_fc2<I, NOUT, 8, 2, false><<<M * TK * (NOUT / 16 / 8), 8 * 32>>>(RA);
            if (comb) { dsh::CombineArgsT<TK> CA{reinterpret_cast<const PO*>(dplo), dyo, dsouto, dfino, M}; dsh::k_combine<NOUT, 128, gv::O_BF16, TK><<<M * NOUT / (8 * 128), 128>>>(CA); }
        };
        MtTimes T;
        o_router(B); o_shared(B); o_fc(B, true, true, true); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        { PO pl; CK(cudaMemcpy(&pl, dplo, sizeof pl, cudaMemcpyDeviceToHost)); T.slots = pl.n_slots; }
        T.router = time_us(ITERS, [&](int it) { o_router(bufs(it, dplan)); });
        T.shared = time_us(ITERS, [&](int it) { o_shared(bufs(it, dplan)); });
        T.fc1 = time_us(ITERS, [&](int it) { o_fc(bufs(it, dplan), true, false, false); });
        T.fc2 = time_us(ITERS, [&](int it) { o_fc(bufs(it, dplan), false, true, false); });
        T.routed = time_us(ITERS, [&](int it) { o_fc(bufs(it, dplan), true, true, true); });
        T.chain = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplan); o_router(b); o_shared(b); o_fc(b, true, true, true); });
        const double rb = T.fc1 + T.fc2;
        printf("  M=%2d one tile slots %3d | router %6.1f | shared %6.1f | fc1 %6.1f fc2 %6.1f (%5.1f GB/s = %4.1f%% of %.1f) | routed %6.1f | layer chain %7.1f us = %5.2f us/token\n",
               M, T.slots, T.router, T.shared, T.fc1, T.fc2, gbs(T.slots, rb), 100 * gbs(T.slots, rb) / FLEET, FLEET, T.routed, T.chain, T.chain / M);
    }

    // ---- the same layer chains, plain launches vs the PDL chain (ds41_moe_ops.cuh PD = 0 / 1), live routes, same pools. Both through layer_plain / layer_pdl on one
    //      non-blocking stream. "stream" = async launches timed with events (the host enqueues ahead); "graph" = NG layers captured once (each with its own gate / shared set,
    //      cold weights) and replayed, the per-layer time of a captured decode step.
    {
        constexpr int NG = 24;
        printf("\n[plain launches vs PDL chain, live routes, cold weights] us per layer; stream = %d async layers, graph = %d layers captured and replayed\n", ITERS, NG);
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        cudaEvent_t s0, s1; CK(cudaEventCreate(&s0)); CK(cudaEventCreate(&s1));
        auto time_st = [&](const std::function<void(int)>& f)
        {
            for (int i = 0; i < 16; ++i) f(i);
            CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep)
            {
                CK(cudaEventRecord(s0, st));
                for (int i = 0; i < ITERS; ++i) f(i);
                CK(cudaEventRecord(s1, st)); CK(cudaEventSynchronize(s1)); CK(cudaGetLastError());
                float ms; CK(cudaEventElapsedTime(&ms, s0, s1));
                best = std::min(best, ms * 1000.0 / ITERS);
            }
            return best;
        };
        auto time_graph = [&](const std::function<void(int)>& f)
        {
            cudaGraph_t gr; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < NG; ++i) f(i);
            CK(cudaStreamEndCapture(st, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep)
            {
                CK(cudaEventRecord(s0, st));
                for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(ge, st));
                CK(cudaEventRecord(s1, st)); CK(cudaEventSynchronize(s1)); CK(cudaGetLastError());
                float ms; CK(cudaEventElapsedTime(&ms, s0, s1));
                best = std::min(best, ms * 1000.0 / (4 * NG));
            }
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
            return best;
        };
        for (int M : {1, 4, 8, 9, 16, 24, 32})
        {
            layer_plain<NE, TK>(B, M, st); CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
            int slots = 0;
            if (M <= 8) { PO pl; CK(cudaMemcpy(&pl, dplan, sizeof pl, cudaMemcpyDeviceToHost)); slots = pl.n_slots; }
            else { PM pl; CK(cudaMemcpy(&pl, dplan, sizeof pl, cudaMemcpyDeviceToHost)); slots = pl.n_slots; }
            const double sp = time_st([&](int it) { layer_plain<NE, TK>(bufs(it, dplan), M, st); });
            const double sq = time_st([&](int it) { layer_pdl<NE, TK>(bufs(it, dplan), M, st); });
            const double gp = time_graph([&](int it) { layer_plain<NE, TK>(bufs(it, dplan), M, st); });
            const double gq = time_graph([&](int it) { layer_pdl<NE, TK>(bufs(it, dplan), M, st); });
            printf("  M=%2d %s slots %3d | stream: plain %7.1f  pdl %7.1f  (%+6.1f us, %+5.1f%%) | graph: plain %7.1f  pdl %7.1f  (%+6.1f us, %+5.1f%%)\n",
                   M, M <= 8 ? "one tile " : "multi-tile", slots, sp, sq, sq - sp, 100 * (sq - sp) / sp, gp, gq, gq - gp, 100 * (gq - gp) / gp);
        }
        CK(cudaStreamDestroy(st)); CK(cudaEventDestroy(s0)); CK(cudaEventDestroy(s1));
    }

    // ---- real routes (p128)
    if (!dir.empty())
    {
        printf("\n[real p128 routes, tokens 0..M-1; slots > %d are folded onto the %d-expert pool (reuse distance >> L2)] fc1/fc2/routed in us; layer = chain(live) - routed(live) + routed(real)\n", POOL, POOL);
        for (int layer : {0, 2, 20, 39})
        {
            const std::string path = dir + "/L" + std::to_string(layer) + ".router_idx.bin";
            const std::vector<uint8_t> raw = slurp(path);
            if (raw.size() != 128 * 6 * 4) { printf("  L%d: %s missing or wrong size (%zu bytes)\n", layer, path.c_str(), raw.size()); continue; }
            std::vector<int> idx(128 * 6); memcpy(idx.data(), raw.data(), raw.size());
            for (int mi = 0; mi < 4; ++mi)
            {
                const int M = Ms[mi], nt = (M + 7) / 8;
                int ex[MT][8] = {}; float wt[MT][8] = {};
                for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) { ex[m][k] = idx[m * 6 + k]; wt[m][k] = 1.5f / TK; }
                auto P0 = std::make_unique<PM>();
                dsr::plan_ref<TK, MT>(M, ex, wt, *P0);
                for (int v = 0; v < NV; ++v)
                {
                    auto P = std::make_unique<PM>(*P0);
                    for (int s = 0; s < P->n_slots; ++s) P->slot_expert[s] = (v * P0->n_slots + s) % POOL;
                    CK(cudaMemcpy(dplanv + v * PLAN_B, P.get(), sizeof(PM), cudaMemcpyHostToDevice));
                }
                MtTimes T;
                with_nt(nt, [&](auto c)
                {
                    constexpr int NT = decltype(c)::value;
                    T.fc1 = time_us(ITERS, [&](int it) { mt_fc1<TK, NT>(bufs(it, dplanv + (it % NV) * PLAN_B), M); });
                    T.fc2 = time_us(ITERS, [&](int it) { mt_fc2<TK, NT>(bufs(it, dplanv + (it % NV) * PLAN_B), M); });
                    T.routed = time_us(ITERS, [&](int it) { const MtBuf b = bufs(it, dplanv + (it % NV) * PLAN_B); mt_fc1<TK, NT>(b, M); mt_fc2<TK, NT>(b, M); mt_combine<TK>(b, M); });
                });
                CK(cudaGetLastError());
                const double rb = T.fc1 + T.fc2, est = live[mi].chain - live[mi].routed + T.routed;
                printf("  L%-2d M=%2d NT=%d slots %3d tiles %3d%s | fc1 %6.1f fc2 %6.1f (%5.1f GB/s = %4.1f%% of %.1f) | routed %6.1f | layer %7.1f us = %5.2f us/token\n",
                       layer, M, nt, P0->n_slots, P0->n_tiles, P0->n_slots > POOL ? " (folded)" : "", T.fc1, T.fc2, gbs(P0->n_slots, rb), 100 * gbs(P0->n_slots, rb) / FLEET, FLEET, T.routed, est, est / M);
                if (nt >= 3)
                {
                    printf("        fc1 variants (us):");
                    const char* vn[4] = {"D2/b1", "D2/b2", "D1/b1", "D1/b2"};
                    for (int v = 0; v < 4; ++v)
                        printf("  %s %7.1f", vn[v], time_us(ITERS, [&](int it) { mt_fc1_any<TK>(nt, v, bufs(it, dplanv + (it % NV) * PLAN_B), M); }));
                    printf("\n");
                }
            }
        }
    }
    cudaFree(w1); cudaFree(w2); cudaFree(sg); cudaFree(su); cudaFree(sd); cudaFree(dgate); cudaFree(dx); cudaFree(dbias); cudaFree(dxf); cudaFree(dxo); cudaFree(dlog);
    cudaFree(dplan); cudaFree(dplanv); cudaFree(dplo); cudaFree(dact); cudaFree(dy); cudaFree(dsact); cudaFree(dsout); cudaFree(dfin);
    cudaFree(dacto); cudaFree(dyo); cudaFree(dsacto); cudaFree(dsouto); cudaFree(dfino);
    return 0;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "layout";
    if (!strcmp(mode, "layout")) return layout_check();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { printf("no GPU visible: only `layout` runs here\n"); return 3; }
    if (!strcmp(mode, "check")) { // the main model (top-6) and the DSpark draft layers 40..42 (top-3): the combine and the whole layer are generic in the route count
        const int a = layout_check(), b = check_combine<6>() + check_combine<3>(), c = check_layer<6>() + check_layer<3>();
        printf("%s\n", a || b || c ? "FAILED" : "ALL OK"); return a || b || c; }
    if (!strcmp(mode, "chain")) {   // the adapter's launch sequence end to end: router plan on the device -> shared -> routed (n_slots_dev gate) -> bf16 combine
        const int c = check_chain<384, 6>() + check_chain<128, 3>();
        printf("%s\n", c ? "FAILED" : "ALL OK"); return c; }
    if (!strcmp(mode, "multitile")) {   // M 1..32 through the multi-tile kernels (ds41_moe_ops.cuh router_mt / shared_mt / experts_mt), design G3
        const int c = check_plan_mt<384, 6>() + check_plan_mt<128, 3>() + check_mt<384, 6>() + check_mt<128, 3>();
        printf("%s\n", c ? "FAILED" : "ALL OK"); return c; }
    if (!strcmp(mode, "fixture")) {   // fixture IN_DIR LAYER OUT_DIR: one real layer on the oracle's ffn_in (ds_moe_fixture_prep.py makes IN_DIR)
        if (argc < 5) { printf("usage: %s fixture IN_DIR LAYER OUT_DIR\n", argv[0]); return 1; }
        return fixture(argv[2], atoi(argv[3]), argv[4]); }
    if (!strcmp(mode, "pdl")) return check_pdl<384, 6>(argc > 2 ? atoi(argv[2]) : 100) ? 1 : 0;   // pdl [ROUNDS]: the PDL chain (stream and graph) vs the plain chain, bitwise
    if (!strcmp(mode, "bench")) return bench(argc > 2 ? atoi(argv[2]) : 16);
    if (!strcmp(mode, "mtbench")) return mtbench<384, 6>(argc > 2 ? argv[2] : "");   // mtbench [ROUTES_DIR]: M 9..32 layer timings, cold weights
    printf("usage: %s layout|check|chain|multitile|pdl [rounds]|fixture|bench [sets]|mtbench [ROUTES_DIR]\n", argv[0]);
    return 1;
}
