// GPU parity test for Inkling-2-Mini looped kernels & exlr8 pmcgx MoE on sm_121a (run on DGX Spark):
//   1. ink2::attn_res_f32 + ink2::msg_gate_norm at exact Inkling-2-Mini width (H=2304, R=64, M=1,5,16) vs ink2ref.
//   2. ink2::tr3::moe_tr3 (pmcgx E4M3-rounded, K=2/3/4 mixed per-expert widths, H=2304, eb=9 -> I=1152, M=1,4,13)
//      vs exact CPU pmcgx + H128 + Inkling-2 softcapped SwiGLU reference.
//   3. End-to-end looped ink2rt::Engine (8 phys -> 16 virtual layers, core=2..6, passes=3, gate_rank=16, NVFP4+MXFP8,
//      3-layer MTP with virtual taps [1, 7, 13, 16]) planned via Rust g6l_ink2_plan_ex vs ink2ref::Model.
//
// Build (from repo root):
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_INK2=1 -DG6_NR=1 -DDEC_S2 -DG6_RS_LOADER \
//     -Xcompiler -ffp-contract=off engine/tests/ink2_loop_gpu_test.cu engine/loader/g6pack.cpp \
//     engine/loader-rs/target/release/libg6load.a engine/knobs-rs/target/release/libgr8knobs.a \
//     engine/target/release/libg6host.a -lcublas -lcublasLt -lpthread -ldl -lm -o bin/ink2_loop_gpu_test
#include "../ref/ink2/ink2_cpu_ref.h"
#include "../runtime/ink2_step.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace
{
using namespace ink2ref;

struct Rng
{
    uint64_t s;
    uint32_t next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 40);
    }
    float uni() { return static_cast<float>(next()) / 8388608.0f - 1.0f; }
    std::vector<float> bf(size_t n, float scale, float bias = 0.0f)
    {
        std::vector<float> v(n);
        for (float& x : v) x = rbf(bias + scale * uni());
        return v;
    }
    std::vector<float> f32(size_t n, float scale)
    {
        std::vector<float> v(n);
        for (float& x : v) x = scale * uni();
        return v;
    }
    std::vector<uint8_t> bytes(size_t n)
    {
        std::vector<uint8_t> v(n);
        for (uint8_t& b : v) b = static_cast<uint8_t>(next() & 0xffu);
        return v;
    }
};

void ck(cudaError_t e, const char* w)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA FAIL %s: %s\n", w, cudaGetErrorString(e));
        exit(1);
    }
}
#define CK(x) ck((x), #x)

template <class T> T* to_dev(const std::vector<T>& h)
{
    T* d = nullptr;
    CK(cudaMalloc(&d, (h.empty() ? 1 : h.size()) * sizeof(T)));
    if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

std::vector<uint16_t> to_bf16(const std::vector<float>& v)
{
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = f32_to_bf16(v[i]);
    return o;
}

std::vector<float> from_bf16(const std::vector<uint16_t>& v)
{
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f32(v[i]);
    return o;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b, float& max_abs)
{
    double ab = 0.0, aa = 0.0, bb = 0.0;
    max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        const float d = std::fabs(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
    }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

void require(bool cond, const char* msg)
{
    if (!cond)
    {
        fprintf(stderr, "FAIL: %s\n", msg);
        exit(1);
    }
}

// ============================================================================
// Test 1: attn_res_f32 + msg_gate_norm at Inkling-2-Mini H=2304, R=64
// ============================================================================
void test_loop_kernels()
{
    constexpr int H = 2304, R = 64, D = 3, Q = 4;
    Rng rng{0xA11CE01ULL};
    for (int M : {1, 5, 16})
    {
        std::vector<std::vector<float>> rows(D);
        std::vector<uint16_t*> d_rows(D);
        for (int d = 0; d < D; ++d)
        {
            rows[d] = rng.bf(static_cast<size_t>(M) * H, 1.0f);
            d_rows[d] = to_dev(to_bf16(rows[d]));
        }
        const std::vector<float> query = rng.f32(static_cast<size_t>(Q) * H, 0.5f);
        float* d_query = to_dev(query);
        std::vector<float*> d_out(Q);
        for (int q = 0; q < Q; ++q) CK(cudaMalloc(&d_out[q], static_cast<size_t>(M) * H * sizeof(float)));

        ink2::AttnResF32Args a{};
        for (int d = 0; d < D; ++d) a.rows[d] = d_rows[d];
        a.query = d_query;
        for (int q = 0; q < Q; ++q) a.out[q] = d_out[q];
        a.D = D;
        a.Q = Q;
        a.H = H;
        ink2::attn_res_f32(a, M, 0);
        CK(cudaDeviceSynchronize());

        // CPU reference
        Config cfg;
        cfg.hidden = H;
        cfg.n_layers = 1;
        cfg.n_heads = 24;
        cfg.n_kv_heads = 8;
        cfg.head_dim = 128;
        cfg.inter = 1152;
        cfg.n_exp = 16;
        cfg.top_k = 4;
        cfg.vocab = 64;
        cfg.valid_vocab = 64;
        cfg.global = {0};
        cfg.mxfp8 = {0};
        Weights w;
        w.layers.resize(1);
        Model ref(cfg, w);

        std::vector<std::vector<float>> states(rows.begin(), rows.begin() + D - 1);
        const std::vector<float> ref_ar = ref.attn_res_f32(states, rows[D - 1], query, Q, M);
        for (int q = 0; q < Q; ++q)
        {
            std::vector<float> got(static_cast<size_t>(M) * H);
            CK(cudaMemcpy(got.data(), d_out[q], got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> exp(static_cast<size_t>(M) * H);
            for (int m = 0; m < M; ++m)
                memcpy(&exp[static_cast<size_t>(m) * H], &ref_ar[(static_cast<size_t>(m) * Q + q) * H], H * sizeof(float));
            float md = 0.0f;
            const double cos = cosine(got, exp, md);
            if (cos < 0.999999 || md > 1e-4f)
            {
                fprintf(stderr, "attn_res_f32 M=%d q=%d cos=%.8f max_abs=%.6f\n", M, q, cos, md);
                exit(1);
            }
        }

        // msg_gate_norm on (d_out[0], d_out[1])
        Layer::MsgGate g;
        g.down = rng.bf(static_cast<size_t>(R) * 2 * H, 1.0f / std::sqrt(static_cast<float>(2 * H)));
        g.forget = rng.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
        g.gain = rng.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
        const std::vector<float> nw = rng.bf(H, 0.2f, 1.0f);
        uint16_t* d_down = to_dev(to_bf16(g.down));
        uint16_t* d_forget = to_dev(to_bf16(g.forget));
        uint16_t* d_gain = to_dev(to_bf16(g.gain));
        uint16_t* d_nw = to_dev(to_bf16(nw));
        uint16_t *d_cat = nullptr, *d_g = nullptr, *d_y = nullptr;
        CK(cudaMalloc(&d_cat, static_cast<size_t>(M) * 2 * H * 2));
        CK(cudaMalloc(&d_g, static_cast<size_t>(M) * R * 2));
        CK(cudaMalloc(&d_y, static_cast<size_t>(M) * H * 2));
        ink2::msg_gate_norm(d_out[0], d_out[1], d_down, d_forget, d_gain, d_nw, d_cat, d_g, d_y, M, H, R, 0);
        CK(cudaDeviceSynchronize());

        std::vector<uint16_t> got_u16(static_cast<size_t>(M) * H);
        CK(cudaMemcpy(got_u16.data(), d_y, got_u16.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<float> got_y = from_bf16(got_u16);
        std::vector<float> exp_y(static_cast<size_t>(M) * H), tmp(H);
        for (int m = 0; m < M; ++m)
        {
            ref.msg_gate(&ref_ar[(static_cast<size_t>(m) * Q + 0) * H], &ref_ar[(static_cast<size_t>(m) * Q + 1) * H], g, tmp.data());
            ref.wnorm(tmp.data(), nw, &exp_y[static_cast<size_t>(m) * H]);
        }
        float md = 0.0f;
        const double cos = cosine(got_y, exp_y, md);
        printf("[test_loop_kernels] H=%d R=%d M=%2d: cos=%.8f max_abs=%.6f\n", H, R, M, cos, md);
        require(cos > 0.99999 && md < 0.02f, "msg_gate_norm parity");

        for (uint16_t* p : d_rows) cudaFree(p);
        for (float* p : d_out) cudaFree(p);
        cudaFree(d_query);
        cudaFree(d_down);
        cudaFree(d_forget);
        cudaFree(d_gain);
        cudaFree(d_nw);
        cudaFree(d_cat);
        cudaFree(d_g);
        cudaFree(d_y);
    }
}

// ============================================================================
// Test 2: exlr8 tr3 pmcgx MoE (K=2,3,4 mixed per-expert, H=2304, eb=9, I=1152)
// ============================================================================
float f16_to_f32(uint16_t h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }
uint16_t f32_to_f16(float x)
{
    __half h = __float2half_rn(x);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

float pmcgx_e4m3_weight(const uint8_t* tile, int t, int bits)
{
    const int nb = 256 * bits;
    const int end = ((t | 1) + 1) * bits;
    uint32_t st = 0;
    for (int q = 0; q < 20; ++q)
    {
        const int bit = ((end - 20 + q) % nb + nb) % nb;
        uint32_t word;
        memcpy(&word, tile + (bit >> 5) * 4, 4);
        st = (st << 1) | ((word >> (31 - (bit & 31))) & 1u);
    }
    const uint32_t x0 = ((st * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint32_t x1 = ((st * 0xBCA14C51u) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint16_t a_lo = static_cast<uint16_t>(x0 & 0xffffu), a_hi = static_cast<uint16_t>(x0 >> 16);
    const uint16_t b_lo = static_cast<uint16_t>(x1 & 0xffffu), b_hi = static_cast<uint16_t>(x1 >> 16);
    const float sum = (t & 1) == 0 ? (f16_to_f32(a_lo) + f16_to_f32(b_hi)) : (f16_to_f32(a_hi) + f16_to_f32(b_lo));
    const float h = f16_to_f32(f32_to_f16(sum));
    return nv4pack::e4m3_to_f32(nv4pack::f32_to_e4m3_rne(h));
}

void decode_tile_16x16(const uint8_t* tile, int bits, float (&w)[16][16])
{
    // Lane l (0..31): rows r = l/4, r+8; cols c = (l%4)*2, +1, +8, +9; stream weights t = 8*l + j:
    //   j=0,1 -> (r, 2c, 2c+1); j=2,3 -> (r, 2c+8, 2c+9); j=4,5 -> (r+8, 2c, 2c+1); j=6,7 -> (r+8, 2c+8, 2c+9)
    for (int lane = 0; lane < 32; ++lane)
    {
        const int r = lane >> 2, c = lane & 3;
        const int rr[8] = {r, r, r, r, r + 8, r + 8, r + 8, r + 8};
        const int cc[8] = {2 * c, 2 * c + 1, 2 * c + 8, 2 * c + 9, 2 * c, 2 * c + 1, 2 * c + 8, 2 * c + 9};
        for (int j = 0; j < 8; ++j) w[rr[j]][cc[j]] = pmcgx_e4m3_weight(tile, lane * 8 + j, bits);
    }
}

void had128_cpu(float* x)
{
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += 2 * len)
            for (int j = 0; j < len; ++j)
            {
                const float a = x[i + j], b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
    const float sc = 1.0f / std::sqrt(128.0f);
    for (int i = 0; i < 128; ++i) x[i] *= sc;
}

void test_tr3_pmcgx_moe()
{
    constexpr int H = 2304, EB = 9, I = EB * 128, NE = 16, TK = 4, HB = H / 128;
    const int widths[NE] = {2, 3, 4, 3, 2, 4, 3, 2, 4, 3, 2, 4, 3, 2, 3, 4};   // mixed per-expert widths, kmix = 0
    Rng rng{0xE4156C28ULL};

    const size_t mb = static_cast<size_t>(H) * 16;   // (H/16)*8*32 bytes per bit
    std::vector<long long> table(NE * 6, 0), kmix(NE, 0);
    size_t fc1_tot = 0, fc2_tot = 0;
    for (int e = 0; e < NE; ++e)
    {
        const int k = widths[e];
        table[e * 6 + 0] = static_cast<long long>(fc1_tot);
        table[e * 6 + 1] = static_cast<long long>(fc2_tot);
        table[e * 6 + 2] = static_cast<long long>(e * EB);
        table[e * 6 + 3] = EB;
        table[e * 6 + 4] = k;
        table[e * 6 + 5] = 0;
        fc1_tot += static_cast<size_t>(EB) * 2 * k * mb;
        fc2_tot += static_cast<size_t>(EB) * k * mb;
    }
    const std::vector<uint8_t> fc1 = rng.bytes(fc1_tot);
    const std::vector<uint8_t> fc2 = rng.bytes(fc2_tot);
    const size_t nblk = static_cast<size_t>(NE) * EB;
    std::vector<__half> v1(nblk * 2 * H), v2(nblk * 3 * 128), v3(nblk * H);
    for (__half& h : v1) h = __float2half_rn(0.05f * rng.uni());
    for (__half& h : v2) h = __float2half_rn(0.15f * rng.uni());
    for (__half& h : v3) h = __float2half_rn(0.05f * rng.uni());

    uint8_t* d_fc1 = to_dev(fc1);
    uint8_t* d_fc2 = to_dev(fc2);
    __half* d_v1 = to_dev(v1);
    __half* d_v2 = to_dev(v2);
    __half* d_v3 = to_dev(v3);
    long long* d_tab = to_dev(table);
    long long* d_kmix = to_dev(kmix);

    for (int M : {1, 4, 13})
    {
        const std::vector<float> x = rng.bf(static_cast<size_t>(M) * H, 0.5f);
        std::vector<int> idx(static_cast<size_t>(M) * TK);
        std::vector<float> wt(static_cast<size_t>(M) * TK);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < TK; ++k)
            {
                idx[m * TK + k] = (m * 3 + k * 4) % NE;
                wt[m * TK + k] = 0.5f + 0.25f * k;
            }
        ink2::PlanHost ph;
        ink2::plan_ref(idx.data(), wt.data(), M, TK, NE, ph);

        ink2::Work wk;
        wk.alloc(M, H, I, NE, TK, false);
        CK(cudaMemcpy(wk.idx, idx.data(), idx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(wk.wt, wt.data(), wt.size() * sizeof(float), cudaMemcpyHostToDevice));
        ink2::plan(wk.plan, wk.idx, wk.wt, M, NE, TK, 0);

        const size_t mt = static_cast<size_t>(ink2::plan_max_tiles(M, TK, NE));
        __half* d_arot = nullptr;
        uint16_t* d_x = to_dev(to_bf16(x));
        uint16_t* d_out = nullptr;
        CK(cudaMalloc(&d_arot, mt * 8 * EB * 128 * sizeof(__half)));
        CK(cudaMalloc(&d_out, static_cast<size_t>(M) * H * 2));

        ink2::tr3::Tr3Args ta{};
        ta.fc1 = d_fc1;
        ta.fc2 = d_fc2;
        ta.v1 = d_v1;
        ta.v2 = d_v2;
        ta.v3 = d_v3;
        ta.table = d_tab;
        ta.kmix = d_kmix;
        ta.x = d_x;
        ta.plan = wk.plan;
        ta.a_rot = d_arot;
        ta.ypart = wk.ypart;
        ta.out = d_out;
        ta.H = H;
        ta.eb = EB;
        ta.M = M;
        ta.TK = TK;
        ta.y_bf16 = 0;
        ink2::tr3::moe_tr3(ta, ink2::tr3::CB_PMCGX, NE, 0);
        CK(cudaDeviceSynchronize());

        std::vector<uint16_t> got_u16(static_cast<size_t>(M) * H);
        CK(cudaMemcpy(got_u16.data(), d_out, got_u16.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<float> got = from_bf16(got_u16);

        // Exact CPU reference for token 0 and last token (full M=1,4 and tokens 0,M-1 for M=13 to stay fast)
        std::vector<int> check_m;
        if (M <= 4)
            for (int m = 0; m < M; ++m) check_m.push_back(m);
        else
            check_m = {0, 7, 8, M - 1};   // checks tile 0, boundary 7/8, and final partial tile!

        std::vector<float> ref_sub, got_sub;
        float w16[16][16];
        for (int m : check_m)
        {
            std::vector<float> acc(H, 0.0f);
            for (int kk = 0; kk < TK; ++kk)
            {
                const int e = idx[m * TK + kk];
                const float rw = wt[m * TK + kk];
                const int bits = widths[e];
                const size_t tb = static_cast<size_t>(32 * bits);
                for (int b = 0; b < EB; ++b)
                {
                    const size_t brow = static_cast<size_t>(e) * EB + b;
                    float yg[128] = {}, yu[128] = {};
                    for (int mat = 0; mat < 2; ++mat)
                    {
                        const uint8_t* mw = fc1.data() + static_cast<size_t>(table[e * 6 + 0]) + static_cast<size_t>(2 * b + mat) * bits * mb;
                        float* yd = mat == 0 ? yg : yu;
                        for (int hb = 0; hb < HB; ++hb)
                        {
                            float xr[128];
                            for (int c = 0; c < 128; ++c) xr[c] = x[static_cast<size_t>(m) * H + hb * 128 + c] * __half2float(v1[(brow * 2 + mat) * H + hb * 128 + c]);
                            had128_cpu(xr);
                            for (int c = 0; c < 128; ++c) xr[c] = __half2float(__float2half_rn(xr[c]));
                            for (int p = 0; p < 8; ++p)
                                for (int warp = 0; warp < 8; ++warp)
                                {
                                    decode_tile_16x16(mw + (static_cast<size_t>(hb * 8 + p) * 8 + warp) * tb, bits, w16);
                                    for (int r = 0; r < 16; ++r)
                                    {
                                        float s = 0.0f;
                                        for (int k = 0; k < 16; ++k) s += w16[r][k] * xr[p * 16 + k];
                                        yd[warp * 16 + r] += s;
                                    }
                                }
                        }
                        for (int c = 0; c < 128; ++c) yd[c] = __half2float(__float2half_rn(yd[c]));
                    }
                    had128_cpu(yg);
                    had128_cpu(yu);
                    float va[128];
                    for (int c = 0; c < 128; ++c)
                    {
                        const float g = __half2float(__float2half_rn(yg[c] * __half2float(v2[(brow * 3 + 0) * 128 + c])));
                        const float u = __half2float(__float2half_rn(yu[c] * __half2float(v2[(brow * 3 + 1) * 128 + c])));
                        va[c] = Model::act(g, u) * __half2float(v2[(brow * 3 + 2) * 128 + c]);
                    }
                    had128_cpu(va);
                    for (int c = 0; c < 128; ++c) va[c] = __half2float(__float2half_rn(va[c]));
                    const uint8_t* mw2 = fc2.data() + static_cast<size_t>(table[e * 6 + 1]) + static_cast<size_t>(b) * bits * mb;
                    for (int hb = 0; hb < HB; ++hb)
                    {
                        float yr[128] = {};
                        for (int warp = 0; warp < 8; ++warp)
                        {
                            const int nt = hb * 8 + warp;
                            for (int ks = 0; ks < 8; ++ks)
                            {
                                decode_tile_16x16(mw2 + (static_cast<size_t>(nt) * 8 + ks) * tb, bits, w16);
                                for (int r = 0; r < 16; ++r)
                                {
                                    float s = 0.0f;
                                    for (int k = 0; k < 16; ++k) s += w16[r][k] * va[ks * 16 + k];
                                    yr[warp * 16 + r] += s;
                                }
                            }
                        }
                        for (int c = 0; c < 128; ++c) yr[c] = __half2float(__float2half_rn(yr[c]));
                        had128_cpu(yr);
                        for (int c = 0; c < 128; ++c) acc[hb * 128 + c] += rw * (yr[c] * __half2float(v3[brow * H + hb * 128 + c]));
                    }
                }
            }
            for (int c = 0; c < H; ++c)
            {
                ref_sub.push_back(rbf(acc[c]));
                got_sub.push_back(got[static_cast<size_t>(m) * H + c]);
            }
        }
        float md = 0.0f;
        const double cos = cosine(got_sub, ref_sub, md);
        printf("[test_tr3_pmcgx_moe] H=%d eb=%d M=%2d (checked %zu toks): cos=%.8f max_abs=%.6f\n", H, EB, M, check_m.size(), cos, md);
        require(cos > 0.9999 && md < 0.05f, "tr3 pmcgx MoE parity");

        wk.release();
        cudaFree(d_arot);
        cudaFree(d_x);
        cudaFree(d_out);
    }

    cudaFree(d_fc1);
    cudaFree(d_fc2);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_tab);
    cudaFree(d_kmix);
}

// ============================================================================
// Test 3: End-to-end looped ink2rt::Engine (native NVFP4 + MXFP8 + MTP) vs ink2ref
// ============================================================================
struct StWriter
{
    struct Entry { std::string name, dtype; std::vector<int64_t> shape; std::vector<uint8_t> data; };
    std::vector<Entry> items;
    void add_raw(const std::string& name, const std::string& dtype, std::vector<int64_t> shape, std::vector<uint8_t> data)
    {
        items.push_back({name, dtype, std::move(shape), std::move(data)});
    }
    void add_bf16(const std::string& name, std::vector<int64_t> shape, const std::vector<float>& v)
    {
        auto b = to_bf16(v);
        std::vector<uint8_t> raw(b.size() * 2);
        memcpy(raw.data(), b.data(), raw.size());
        add_raw(name, "BF16", std::move(shape), std::move(raw));
    }
    void add_f32(const std::string& name, std::vector<int64_t> shape, const std::vector<float>& v)
    {
        std::vector<uint8_t> raw(v.size() * 4);
        memcpy(raw.data(), v.data(), raw.size());
        add_raw(name, "F32", std::move(shape), std::move(raw));
    }
    void write(const std::string& path) const
    {
        std::string hdr = "{";
        size_t off = 0;
        for (size_t i = 0; i < items.size(); ++i)
        {
            const auto& e = items[i];
            if (i) hdr += ",";
            hdr += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
            for (size_t j = 0; j < e.shape.size(); ++j)
            {
                if (j) hdr += ",";
                hdr += std::to_string(e.shape[j]);
            }
            hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + e.data.size()) + "]}";
            off += e.data.size();
        }
        hdr += "}";
        while ((8 + hdr.size()) % 8 != 0) hdr += " ";
        const uint64_t hlen = hdr.size();
        std::ofstream f(path, std::ios::binary);
        require(static_cast<bool>(f), "write safetensors");
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        for (const auto& e : items)
            if (!e.data.empty()) f.write(reinterpret_cast<const char*>(e.data.data()), static_cast<std::streamsize>(e.data.size()));
    }
};

void test_e2e_looped_engine()
{
    constexpr int H = 256, NL = 8, B = 2, NH = 24, NKV = 8, HD = 16, RD = 16, RW = 8, SW = 8, BASE = 6;
    constexpr int I = 256, NE = 16, TK = 4, V = 256, VV = 240, R = 16, LS = 2, LE = 6, PASSES = 3;
    constexpr int ML = 3, MI = 256, MW = 5, MB = 8, MD = 4, MR = 16;

    Config c;
    c.hidden = H; c.n_layers = NL; c.block = B; c.n_heads = NH; c.n_kv_heads = NKV; c.head_dim = HD;
    c.rel_dim = RD; c.rel_win = RW; c.sliding_window = SW; c.log_scale_base = BASE;
    c.inter = I; c.n_exp = NE; c.top_k = TK; c.route_scale = 4.0f; c.vocab = V; c.valid_vocab = VV; c.tp = 1;
    c.global = {0, 1, 0, 1, 0, 1, 0, 1};
    c.mxfp8 = {0, 0, 0, 0, 0, 0, 1, 1};
    c.loop_start = LS; c.loop_end = LE; c.loop_passes = PASSES; c.loop_gate_rank = R;
    c.mtp_layers = ML; c.mtp_inter = MI; c.mtp_window = MW; c.mtp_block = MB; c.mtp_drafts = MD; c.markov_rank = MR;
    c.mtp_global = {0, 1, 0};
    c.mtp_target_layers = {1, 7, 13, 16};

    Rng rng{0x100B2026ULL};
    Weights w;
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    w.embed = rng.bf(static_cast<size_t>(V) * H, 0.8f);
    w.embed_norm = rng.bf(H, 0.2f, 1.0f);
    w.final_norm = rng.bf(H, 0.2f, 1.0f);
    w.output_query = rng.f32(H, 0.5f);
    w.lm_head = rng.bf(static_cast<size_t>(V) * H, 0.25f);
    w.msg_queries = rng.f32(static_cast<size_t>(LE - LS) * 4 * H, 0.5f);

    char tmpl[] = "/tmp/ink2_loop_gpu_XXXXXX";
    require(mkdtemp(tmpl) != nullptr, "mkdtemp");
    const std::string dir = tmpl;

    StWriter st;
    st.add_bf16("model.llm.tok_embeddings.weight", {V, H}, w.embed);
    st.add_bf16("model.llm.embedding_norm.weight", {H}, w.embed_norm);
    st.add_bf16("model.llm.final_norm.weight", {H}, w.final_norm);
    st.add_f32("model.llm.final_attn_res_query", {H}, w.output_query);
    st.add_bf16("model.llm.unembed.weight", {V, H}, w.lm_head);
    // Split msg_queries [4, 4, H] (q, k, v, mlp) into checkpoint's two [8, H] tensors: (q, mlp) and (k, v)
    std::vector<float> qm_q_mlp((LE - LS) * 2 * H), qm_k_v((LE - LS) * 2 * H);
    for (int j = 0; j < LE - LS; ++j)
    {
        memcpy(&qm_q_mlp[(j * 2 + 0) * H], &w.msg_queries[(j * 4 + 0) * H], H * 4);
        memcpy(&qm_q_mlp[(j * 2 + 1) * H], &w.msg_queries[(j * 4 + 3) * H], H * 4);
        memcpy(&qm_k_v[(j * 2 + 0) * H], &w.msg_queries[(j * 4 + 1) * H], H * 4);
        memcpy(&qm_k_v[(j * 2 + 1) * H], &w.msg_queries[(j * 4 + 2) * H], H * 4);
    }
    st.add_f32("model.llm.middle_cycle_cross_pass_queries", {(LE - LS) * 2, H}, qm_q_mlp);
    st.add_f32("model.llm.middle_cycle_cross_pass_kv_queries", {(LE - LS) * 2, H}, qm_k_v);

    w.layers.resize(NL);
    for (int l = 0; l < NL; ++l)
    {
        Layer& y = w.layers[l];
        const bool g = c.global[l] != 0;
        const std::string p = "model.llm.layers." + std::to_string(l) + ".";
        if (l)
        {
            y.attn_query = rng.f32(static_cast<size_t>(g ? 3 : 1) * H, 0.5f);
            if (g) st.add_f32(p + "attn_res.attn_query", {3, H}, y.attn_query);
            else st.add_f32(p + "attn_res.attn_query", {H}, y.attn_query);
        }
        y.mlp_query = rng.f32(H, 0.5f);
        st.add_f32(p + "attn_res.mlp_query", {H}, y.mlp_query);
        if (g)
        {
            y.q_norm = rng.bf(H, 0.2f, 1.0f);
            y.k_norm = rng.bf(H, 0.2f, 1.0f);
            y.v_norm = rng.bf(H, 0.2f, 1.0f);
            y.k_static = rng.bf(4 * nk, 0.3f);
            y.v_static = rng.bf(4 * nk, 0.3f);
            y.k_wproj = rng.bf(static_cast<size_t>(NKV) * 4 * H, ih);
            y.v_wproj = rng.bf(static_cast<size_t>(NKV) * 4 * H, ih);
            st.add_bf16(p + "q_input_norm.weight", {H}, y.q_norm);
            st.add_bf16(p + "k_input_norm.weight", {H}, y.k_norm);
            st.add_bf16(p + "v_input_norm.weight", {H}, y.v_norm);
            st.add_bf16(p + "attn.k_sconv.static_w", {4, static_cast<int64_t>(nk)}, y.k_static);
            st.add_bf16(p + "attn.v_sconv.static_w", {4, static_cast<int64_t>(nk)}, y.v_static);
            st.add_bf16(p + "attn.k_sconv.weight_proj.weight", {NKV * 4, H}, y.k_wproj);
            st.add_bf16(p + "attn.v_sconv.weight_proj.weight", {NKV * 4, H}, y.v_wproj);
        }
        else
        {
            y.attn_norm = rng.bf(H, 0.2f, 1.0f);
            y.k_sconv = rng.bf(nk * 4, 0.3f);
            y.v_sconv = rng.bf(nk * 4, 0.3f);
            st.add_bf16(p + "attn_norm.weight", {H}, y.attn_norm);
            st.add_bf16(p + "attn.k_sconv.conv.weight", {static_cast<int64_t>(nk), 1, 4}, y.k_sconv);
            st.add_bf16(p + "attn.v_sconv.conv.weight", {static_cast<int64_t>(nk), 1, 4}, y.v_sconv);
        }
        y.wq = rng.bf(nq * H, ih);
        y.wr = rng.bf(static_cast<size_t>(NH) * RD * H, ih);
        y.gate = rng.bf(nq * H, ih);
        y.sink = rng.bf(static_cast<size_t>(NH) * H, ih);
        y.wk = rng.bf(nk * H, ih);
        y.wv = rng.bf(nk * H, ih);
        y.wo = rng.bf(static_cast<size_t>(H) * nq, 1.0f / std::sqrt(static_cast<float>(nq)));
        y.q_clip = rng.bf(nq, 0.2f, 1.0f);
        y.rel_clip = rng.bf(static_cast<size_t>(NH) * RD, 0.2f, 1.0f);
        y.rel_proj = rng.bf(static_cast<size_t>(RD) * RW, 0.2f);
        y.attn_sconv = rng.bf(static_cast<size_t>(H) * 4, 0.3f);
        y.mlp_sconv = rng.bf(static_cast<size_t>(H) * 4, 0.3f);
        y.mlp_norm = rng.bf(H, 0.2f, 1.0f);
        y.router_w = rng.bf(static_cast<size_t>(NE) * H, ih);
        y.router_bias = rng.f32(NE, 0.1f);
        for (int k = 0; k < TK; ++k) y.router_bias[(3 * l + 5 * k) % NE] += 2.0f * static_cast<float>(TK - k);

        st.add_bf16(p + "attn.wq.weight", {static_cast<int64_t>(nq), H}, y.wq);
        st.add_bf16(p + "attn.wr.weight", {NH * RD, H}, y.wr);
        st.add_bf16(p + "attn.gate_attn.weight", {static_cast<int64_t>(nq), H}, y.gate);
        st.add_bf16(p + "attn.attn_sink_proj.weight", {NH, H}, y.sink);
        st.add_bf16(p + "attn.wk.weight", {static_cast<int64_t>(nk), H}, y.wk);
        st.add_bf16(p + "attn.wv.weight", {static_cast<int64_t>(nk), H}, y.wv);
        st.add_bf16(p + "attn.wo.weight", {H, static_cast<int64_t>(nq)}, y.wo);
        st.add_bf16(p + "attn.q_l2_clip.scale", {NH, HD}, y.q_clip);
        st.add_bf16(p + "attn.rel_l2_clip.scale", {NH, RD}, y.rel_clip);
        st.add_bf16(p + "attn.rel_logits_proj.weight", {RD, RW}, y.rel_proj);
        st.add_bf16(p + "attn_sconv.conv.weight", {H, 1, 4}, y.attn_sconv);
        st.add_bf16(p + "mlp_sconv.conv.weight", {H, 1, 4}, y.mlp_sconv);
        st.add_bf16(p + "mlp_norm.weight", {H}, y.mlp_norm);
        st.add_bf16(p + "mlp.router.gate.weight", {NE, H}, y.router_w);
        st.add_f32(p + "mlp.router.e_score_correction_bias", {NE}, y.router_bias);

        if (l >= LS && l < LE)
        {
            const char* cnames[] = {"attn", "k", "v", "mlp"};
            Layer::MsgGate* gs[] = {&y.q_gate, &y.k_gate, &y.v_gate, &y.mlp_gate};
            for (int gi = 0; gi < 4; ++gi)
            {
                gs[gi]->down = rng.bf(static_cast<size_t>(R) * 2 * H, ih * 0.5f);
                gs[gi]->forget = rng.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
                gs[gi]->gain = rng.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
                const std::string cp = p + "controllers." + cnames[gi] + ".";
                st.add_bf16(cp + "down_weight", {R, 2 * H}, gs[gi]->down);
                st.add_bf16(cp + "forget_weight", {H, R}, gs[gi]->forget);
                st.add_bf16(cp + "gain_weight", {H, R}, gs[gi]->gain);
            }
        }

        // Experts
        if (c.mxfp8[l])
        {
            std::vector<uint8_t> w13c, w13s, w2c, w2s;
            for (int e = 0; e < NE; ++e)
            {
                auto wg = rng.bf(static_cast<size_t>(I) * H, ih), wu = rng.bf(static_cast<size_t>(I) * H, ih);
                auto wd = rng.bf(static_cast<size_t>(H) * I, 1.0f / std::sqrt(static_cast<float>(I)));
                std::vector<uint8_t> gc(wg.size()), gs(wg.size() / 32), uc(wu.size()), us(wu.size() / 32), dc(wd.size()), ds(wd.size() / 32);
                mx8::quantize(wg.data(), wg.size() / 32, gc.data(), gs.data());
                mx8::quantize(wu.data(), wu.size() / 32, uc.data(), us.data());
                mx8::quantize(wd.data(), wd.size() / 32, dc.data(), ds.data());
                auto f1 = mx8::pack_fc1(gc.data(), gs.data(), uc.data(), us.data(), I, H);
                auto f2 = mx8::pack_fc2(dc.data(), ds.data(), H, I, 0, I);
                y.fc1.insert(y.fc1.end(), f1.begin(), f1.end());
                y.fc2.insert(y.fc2.end(), f2.begin(), f2.end());
                // Checkpoint stores w13 with interleaved gate/up rows [2*I, H]
                for (int r = 0; r < I; ++r)
                {
                    w13c.insert(w13c.end(), gc.begin() + r * H, gc.begin() + (r + 1) * H);
                    w13c.insert(w13c.end(), uc.begin() + r * H, uc.begin() + (r + 1) * H);
                    w13s.insert(w13s.end(), gs.begin() + r * (H / 32), gs.begin() + (r + 1) * (H / 32));
                    w13s.insert(w13s.end(), us.begin() + r * (H / 32), us.begin() + (r + 1) * (H / 32));
                }
                w2c.insert(w2c.end(), dc.begin(), dc.end());
                w2s.insert(w2s.end(), ds.begin(), ds.end());
            }
            st.add_raw(p + "mlp.experts.w13_weight", "F8_E4M3", {NE, 2 * I, H}, std::move(w13c));
            st.add_raw(p + "mlp.experts.w13_weight_scale", "U8", {NE, 2 * I, H / 32}, std::move(w13s));
            st.add_raw(p + "mlp.experts.w2_weight", "F8_E4M3", {NE, H, I}, std::move(w2c));
            st.add_raw(p + "mlp.experts.w2_weight_scale", "U8", {NE, H, I / 32}, std::move(w2s));
        }
        else
        {
            std::vector<uint8_t> w13c, w13s, w2c, w2s;
            std::vector<float> a13(NE), a2(NE);
            for (int e = 0; e < NE; ++e)
            {
                auto wg = rng.bf(static_cast<size_t>(I) * H, ih), wu = rng.bf(static_cast<size_t>(I) * H, ih);
                auto wd = rng.bf(static_cast<size_t>(H) * I, 1.0f / std::sqrt(static_cast<float>(I)));
                std::vector<float> g_u(wg.size() + wu.size());
                memcpy(g_u.data(), wg.data(), wg.size() * 4);
                memcpy(g_u.data() + wg.size(), wu.data(), wu.size() * 4);
                std::vector<uint8_t> c13(g_u.size() / 2), s13(g_u.size() / 16), c2(wd.size() / 2), s2(wd.size() / 16);
                const float o1 = nv4pack::quantize_row(g_u.data(), g_u.size(), nv4pack::ScaleRule::FourOverSixMae, c13.data(), s13.data());
                const float o2 = nv4pack::quantize_row(wd.data(), wd.size(), nv4pack::ScaleRule::FourOverSixMae, c2.data(), s2.data());
                const size_t n1 = wg.size();
                auto f1 = nv4pack::pack_fc1(c13.data(), s13.data(), c13.data() + n1 / 2, s13.data() + n1 / 16, I, H);
                auto f2 = nv4pack::pack_fc2(c2.data(), s2.data(), H, I, 0, I);
                y.fc1.insert(y.fc1.end(), f1.begin(), f1.end());
                y.fc2.insert(y.fc2.end(), f2.begin(), f2.end());
                y.s1.push_back(o1);
                y.s2.push_back(o2);
                a13[e] = o1 * (448.0f * 6.0f);
                a2[e] = o2 * (448.0f * 6.0f);
                // Interleave gate/up rows for checkpoint w13
                for (int r = 0; r < I; ++r)
                {
                    w13c.insert(w13c.end(), c13.begin() + r * (H / 2), c13.begin() + (r + 1) * (H / 2));
                    w13c.insert(w13c.end(), c13.begin() + n1 / 2 + r * (H / 2), c13.begin() + n1 / 2 + (r + 1) * (H / 2));
                    w13s.insert(w13s.end(), s13.begin() + r * (H / 16), s13.begin() + (r + 1) * (H / 16));
                    w13s.insert(w13s.end(), s13.begin() + n1 / 16 + r * (H / 16), s13.begin() + n1 / 16 + (r + 1) * (H / 16));
                }
                w2c.insert(w2c.end(), c2.begin(), c2.end());
                w2s.insert(w2s.end(), s2.begin(), s2.end());
            }
            const std::vector<float> den = {448.0f * 6.0f};
            st.add_raw(p + "mlp.experts.w13_weight", "U8", {NE, 2 * I, H / 2}, std::move(w13c));
            st.add_raw(p + "mlp.experts.w13_weight_scale", "F8_E4M3", {NE, 2 * I, H / 16}, std::move(w13s));
            st.add_f32(p + "mlp.experts.w13_weight_amax", {NE}, a13);
            st.add_f32(p + "mlp.experts.w13_outer_scale_denominator", {1}, den);
            st.add_raw(p + "mlp.experts.w2_weight", "U8", {NE, H, I / 2}, std::move(w2c));
            st.add_raw(p + "mlp.experts.w2_weight_scale", "F8_E4M3", {NE, H, I / 16}, std::move(w2s));
            st.add_f32(p + "mlp.experts.w2_weight_amax", {NE}, a2);
            st.add_f32(p + "mlp.experts.w2_outer_scale_denominator", {1}, den);
        }
    }
    st.write(dir + "/model.safetensors");

    // MTP drafter
    StWriter mtp_st;
    MtpWeights& mtp = w.mtp;
    const int NT = static_cast<int>(c.mtp_target_layers.size());
    mtp.fuse_weight = rng.bf(static_cast<size_t>(ML + 1) * NT, 0.5f);
    mtp.mask_embed = rng.bf(H, 0.5f);
    mtp.norm = rng.bf(H, 0.2f, 1.0f);
    mtp.head_dflare_norm = rng.bf(H, 0.2f, 1.0f);
    mtp.markov_w1 = rng.bf(static_cast<size_t>(V) * MR, 0.2f);
    mtp.rnn_w = rng.bf(static_cast<size_t>(3) * MR * (2 * MR + 2 * H), ih);
    mtp.rnn_b = rng.bf(static_cast<size_t>(3) * MR, 0.1f);
    mtp.hidden_out_w = rng.bf(static_cast<size_t>(H) * MR, 0.2f);
    mtp.hidden_out_b = rng.bf(H, 0.1f);
    mtp_st.add_bf16("fuse_weight", {ML + 1, NT}, mtp.fuse_weight);
    mtp_st.add_bf16("mask_embed", {H}, mtp.mask_embed);
    mtp_st.add_bf16("norm.weight", {H}, mtp.norm);
    mtp_st.add_bf16("markov_head.dflare_norm.weight", {H}, mtp.head_dflare_norm);
    mtp_st.add_bf16("markov_head.markov_w1", {V, MR}, mtp.markov_w1);
    mtp_st.add_bf16("markov_head.rnn_proj.weight", {3 * MR, 2 * MR + 2 * H}, mtp.rnn_w);
    mtp_st.add_bf16("markov_head.rnn_proj.bias", {3 * MR}, mtp.rnn_b);
    mtp_st.add_bf16("markov_head.hidden_out.weight", {H, MR}, mtp.hidden_out_w);
    mtp_st.add_bf16("markov_head.hidden_out.bias", {H}, mtp.hidden_out_b);
    mtp.layers.resize(ML);
    for (int l = 0; l < ML; ++l)
    {
        MtpLayer& ml = mtp.layers[l];
        const std::string p = "layers." + std::to_string(l) + ".";
        ml.dflare_norm = rng.bf(H, 0.2f, 1.0f);
        ml.input_norm = rng.bf(H, 0.2f, 1.0f);
        ml.post_norm = rng.bf(H, 0.2f, 1.0f);
        ml.q_norm = rng.bf(HD, 0.2f, 1.0f);
        ml.k_norm = rng.bf(HD, 0.2f, 1.0f);
        ml.k_norm_ctx = rng.bf(HD, 0.2f, 1.0f);
        ml.wq = rng.bf(nq * H, ih);
        ml.wk = rng.bf(nk * H, ih);
        ml.wv = rng.bf(nk * H, ih);
        ml.k_ctx = rng.bf(nk * H, ih);
        ml.v_ctx = rng.bf(nk * H, ih);
        ml.wo = rng.bf(static_cast<size_t>(H) * nq, 1.0f / std::sqrt(static_cast<float>(nq)));
        ml.gate = rng.bf(static_cast<size_t>(MI) * H, ih);
        ml.up = rng.bf(static_cast<size_t>(MI) * H, ih);
        ml.down = rng.bf(static_cast<size_t>(H) * MI, 1.0f / std::sqrt(static_cast<float>(MI)));
        mtp_st.add_bf16(p + "dflare_norm.weight", {H}, ml.dflare_norm);
        mtp_st.add_bf16(p + "input_layernorm.weight", {H}, ml.input_norm);
        mtp_st.add_bf16(p + "post_attention_layernorm.weight", {H}, ml.post_norm);
        mtp_st.add_bf16(p + "self_attn.q_norm.weight", {HD}, ml.q_norm);
        mtp_st.add_bf16(p + "self_attn.k_norm.weight", {HD}, ml.k_norm);
        mtp_st.add_bf16(p + "self_attn.k_norm_ctx.weight", {HD}, ml.k_norm_ctx);
        mtp_st.add_bf16(p + "self_attn.q_proj.weight", {static_cast<int64_t>(nq), H}, ml.wq);
        mtp_st.add_bf16(p + "self_attn.k_proj.weight", {static_cast<int64_t>(nk), H}, ml.wk);
        mtp_st.add_bf16(p + "self_attn.v_proj.weight", {static_cast<int64_t>(nk), H}, ml.wv);
        mtp_st.add_bf16(p + "self_attn.k_proj_ctx.weight", {static_cast<int64_t>(nk), H}, ml.k_ctx);
        mtp_st.add_bf16(p + "self_attn.v_proj_ctx.weight", {static_cast<int64_t>(nk), H}, ml.v_ctx);
        mtp_st.add_bf16(p + "self_attn.o_proj.weight", {H, static_cast<int64_t>(nq)}, ml.wo);
        mtp_st.add_bf16(p + "mlp.gate_proj.weight", {MI, H}, ml.gate);
        mtp_st.add_bf16(p + "mlp.up_proj.weight", {MI, H}, ml.up);
        mtp_st.add_bf16(p + "mlp.down_proj.weight", {H, MI}, ml.down);
    }
    mtp_st.write(dir + "/mtp.safetensors");

    {
        std::ofstream f(dir + "/config.json");
        f << "{\n"
          << "  \"architectures\": [\"Inkling2LoopForConditionalGeneration\"],\n"
          << "  \"text_config\": {\n"
          << "    \"hidden_size\": " << H << ", \"num_hidden_layers\": " << NL << ",\n"
          << "    \"num_attention_heads\": " << NH << ", \"num_key_value_heads\": " << NKV << ", \"head_dim\": " << HD << ",\n"
          << "    \"n_routed_experts\": " << NE << ", \"num_experts_per_tok\": " << TK << ", \"intermediate_size\": " << I << ",\n"
          << "    \"vocab_size\": " << V << ", \"unpadded_vocab_size\": " << VV << ",\n"
          << "    \"sliding_window_size\": " << SW << ", \"log_scaling_n_floor\": " << BASE << ",\n"
          << "    \"n_layers_per_attn_res_block\": " << B << ", \"route_scale\": 4.0,\n"
          << "    \"local_layer_ids\": [0, 2, 4, 6],\n"
          << "    \"loop_config\": {\"core_start\": " << LS << ", \"core_end\": " << LE
          << ", \"num_passes\": " << PASSES << ", \"gate_rank\": " << R << ", \"is_varloop\": false}\n"
          << "  },\n"
          << "  \"mtp_config\": {\n"
          << "    \"hidden_size\": " << H << ", \"num_hidden_layers\": " << ML << ",\n"
          << "    \"num_attention_heads\": " << NH << ", \"num_key_value_heads\": " << NKV << ", \"head_dim\": " << HD << ",\n"
          << "    \"intermediate_size\": " << MI << ", \"sliding_window_size\": " << MW << ",\n"
          << "    \"block_size\": " << MB << ", \"num_speculative_tokens\": " << MD << ", \"markov_rank\": " << MR << ",\n"
          << "    \"local_layer_ids\": [0, 2], \"target_layer_ids\": [1, 7, 13, 16]\n"
          << "  }\n"
          << "}\n";
    }
    {
        std::ofstream f(dir + "/hf_quant_config.json");
        f << "{\"quantization\": {\"quant_algo\": \"NVFP4\"}}\n";
    }

    const std::vector<int> toks = {3, 17, 42, 99, 5, 128, 7, 200, 11, 64, 19, 88};
    const int M = static_cast<int>(toks.size());

    // CPU reference
    Model ref(c, w);
    const std::vector<float> ref_lg = ref.forward(toks.data(), M);
    const std::vector<int> ref_draft = ref.mtp_draft(42, MD);

    // GPU engine
    ink2rt::Options opt;
    opt.nr = 1;
    opt.nloc = 1;
    opt.slots = 1;
    opt.max_rows = 16;
    opt.cap = 128;
    opt.nc = 240;
    opt.mtp = MD;
    opt.whole_image = true;
    ink2rt::Engine eng;
    eng.init(dir, opt);
    eng.reset_slot(0);

    // Run in 5 + 4 + 1 + 1 + 1 chunks (testing prefill + decode across all 16 virtual layers!)
    const int chunks[] = {5, 4, 1, 1, 1};
    int off = 0, top1_match = 0;
    std::vector<float> gpu_all, ref_all;
    for (int k : chunks)
    {
        ink2::RowsHost rh;
        rh.add(0, off, k);
        std::vector<ink2rt::RowSample> sm(k);
        std::vector<ink2rt::RowOut> out;
        eng.forward(rh, &toks[off], sm.data(), out);
        eng.commit();
        for (int i = 0; i < k; ++i)
        {
            const float* rl = &ref_lg[static_cast<size_t>(off + i) * V];
            int ref_top = 0;
            for (int v = 1; v < VV; ++v)
                if (rl[v] > rl[ref_top]) ref_top = v;
            if (out[i].tok == ref_top) ++top1_match;
            std::vector<float> gl(VV, -1e9f);
            for (const auto& cd : out[i].top)
                if (cd.id >= 0 && cd.id < VV) gl[cd.id] = cd.logit;
            for (int v = 0; v < VV; ++v)
            {
                gpu_all.push_back(gl[v]);
                ref_all.push_back(rl[v]);
            }
        }
        off += k;
    }
    float md = 0.0f;
    const double cos = cosine(gpu_all, ref_all, md);
    printf("[test_e2e_looped_engine] 8 phys -> 16 virtual (5+4+1+1+1): cos=%.8f max_abs=%.6f top1=%d/%d\n",
           cos, md, top1_match, M);
    require(cos > 0.9999 && top1_match == M, "e2e looped engine logit parity");

    std::vector<int> gpu_draft;
    eng.mtp_draft({0}, {M - 1}, {42}, {ink2rt::RowSample{}}, MD, gpu_draft);
    printf("[test_e2e_looped_engine] MTP drafts: ref=[%d,%d,%d,%d] gpu=[%d,%d,%d,%d]\n",
           ref_draft[0], ref_draft[1], ref_draft[2], ref_draft[3],
           gpu_draft[0], gpu_draft[1], gpu_draft[2], gpu_draft[3]);
    require(gpu_draft == ref_draft, "e2e looped engine MTP draft parity");

    // Cleanup temp dir
    unlink((dir + "/model.safetensors").c_str());
    unlink((dir + "/mtp.safetensors").c_str());
    unlink((dir + "/config.json").c_str());
    unlink((dir + "/hf_quant_config.json").c_str());
    rmdir(dir.c_str());
}
}   // namespace

int main()
{
    test_loop_kernels();
    test_tr3_pmcgx_moe();
    test_e2e_looped_engine();
    printf("ink2_loop_gpu_test: ALL PASSED\n");
    return 0;
}
