// #65 prefill MoE (bindp::launch_experts: route, fc1, mid, fc2, final sum) on one rank's expert pack. The test dumps output for
// the NumPy reference (engine/ref/k1_v2_ref.py) and times against the GB10 read floor. Prefill quantizes activations to
// MXFP8, so compare the error level across widths. Same pack and flags as k1_v2_test.cu, plus:
//   pf_v2_test --pack DIR/rank0 --layer 40 [--nr 4] [--M 20] [--dump FILE] [--bench ITERS] [--wide 1] [--seed 1]
//              [--int4-shared] [--shared-pass] [--shared-ab]
// Routing: random router logits (bias 0), the prefill path's own top-8. A pack with tr3.table row 256 also runs the
// trellis shared expert pass (bindp::launch_shared) unless --int4-shared (then ysh is zero: this test has no MXFP8
// shared weights). The shared expert runs from its FP8 copy (bind::make_shared_fp8) unless --shared-pass (the trellis
// pass). --shared-ab: time today's MXFP8 shared GEMMs (random weights at the TP size's width: xq, gate + up,
// act, xq, down) against the trellis shared pass, both alone.
// Dump: as k1_v2_test (out converted from BF16 to fp16; flags bit 1 = trellis shared expert in out).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../bind_prefill.cuh"
#include "../../h2d.h"
#include "../../loader/devimage.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static void die(const std::string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }

template <class T> static T* dev(const std::vector<T>& v)
{
    T* p;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(g6h2d::copy(p, v.data(), v.size() * sizeof(T)));     // pageable source: pinned staging (64 KiB-page kernels)
    return p;
}

int main(int argc, char** argv)
{
    std::string pack_base, dump;
    int layer = 40, M = 20, iters = 0, wide = 1, int4sh = 0, shab = 0, shpass = 0, nr = g6nr::NR;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--pack") pack_base = nx();
        else if (a == "--layer") layer = std::stoi(nx());
        else if (a == "--nr") nr = std::stoi(nx());
        else if (a == "--int4-shared") int4sh = 1;
        else if (a == "--shared-ab") shab = 1;
        else if (a == "--shared-pass") shpass = 1;
        else if (a == "--M") M = std::stoi(nx());
        else if (a == "--dump") dump = nx();
        else if (a == "--bench") iters = std::stoi(nx());
        else if (a == "--wide") wide = std::stoi(nx());
        else if (a == "--seed") seed = std::stoull(nx());
        else die("unknown arg " + a);
    }
    if (pack_base.empty() || M < 1) die("usage: see the file header");

    g6::Pack pack;
    g6::DevImage img;
    std::string err;
    if (!pack.open_manifest(pack_base + ".manifest.json", err)) die(err);
    g6::LoadOpts lo;
    g6::LoadStats ls;
    if (!img.load(pack, pack_base + ".pack", lo, ls, err)) die(err);
    auto find = [&](const char* name) {
        bind::Tensor t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        t.p = img.ptr(*e);
        t.nd = static_cast<int>(e->shape.size());
        for (int i = 0; i < t.nd && i < 4; ++i) t.shape[i] = e->shape[i];
        t.fmt = e->expert_format;
        return t;
    };
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    bind::LayerTensors D = bind::layer_tensors(layer, find);          // decode view: widen in place, scan the widths
    if (!D.t[bind::T_TR3_TAB].p) die("pack has no tr3.table for the layer");
#ifndef K1V2_MAIN   // -DK1V2_MAIN: build against a tree before #65, for the one-width A/B
    if (const char* e = bind::scan_widths(D, true)) die(e);
    if (D.sh_trellis && !int4sh && !shpass)      // the runtime's order: the FP8 copy before the wide reorder
        if (const char* e = bind::make_shared_fp8(D, st)) die(e);
#endif
    if (wide) { bind::widen_fc1(D, st); bind::widen_fc2(D, st); CK(cudaStreamSynchronize(st)); }
    bindp::Tensors T = bindp::layer_tensors(layer, find);

    // scratch, inputs
    void* blk;
    CK(cudaMalloc(&blk, bindp::scratch_bytes(M)));
    CK(cudaMemset(blk, 0, bindp::scratch_bytes(M)));
    const bindp::Scratch sc = bindp::scratch_at(blk, M);
    std::mt19937_64 rng(seed * 7919 + 3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> xh(static_cast<size_t>(M) * HID);
    for (auto& v : xh) v = __float2half(nd(rng));
    std::vector<float> lg(static_cast<size_t>(M) * 256), bias(256, 0.f);
    for (auto& v : lg) v = nd(rng);
    __half* x = dev(xh);
    float* logits = dev(lg);
    float* bd = dev(bias);
    uint16_t* out;
    CK(cudaMalloc(&out, static_cast<size_t>(M) * HID * 2));

    const bind::Tensor &tab = T.t[bindp::T_TR3_TAB], &v1 = T.t[bindp::T_TR3_V1], &v2 = T.t[bindp::T_TR3_V2], &v3 = T.t[bindp::T_TR3_V3];
    bindp::Expert A;
    A.on = 1; A.M = M; A.maxseg = bindp::max_segments(M);
    A.logits = logits; A.bias = bd;
    A.tok_e = sc.tok_e; A.tok_w = sc.tok_w; A.cnt = sc.cnt;
    A.rr_row = sc.rr_row; A.rr_e = sc.rr_e; A.rr_sp = sc.rr_sp; A.rr_w = sc.rr_w; A.pos = sc.pos; A.segs = sc.segs; A.n_seg = sc.n_seg;
    A.t_tab = static_cast<const long long*>(tab.p);
#ifndef K1V2_MAIN
    A.t_kmix = static_cast<const long long*>(T.t[bindp::T_TR3_KMIX].p);
    A.k2 = D.k2;
#endif
    A.t_fc1 = static_cast<const uint8_t*>(T.t[bindp::T_TR3_FC1].p); A.t_wide = D.fc1_wide;
    A.t_fc2 = static_cast<const uint8_t*>(T.t[bindp::T_TR3_FC2].p); A.t_wide2 = D.fc2_wide;
    A.t_v1 = static_cast<const __half*>(v1.p); A.t_v2 = static_cast<const __half*>(v2.p); A.t_v3 = static_cast<const __half*>(v3.p);
    A.x = x; A.ldx = HID;
    A.z1 = sc.z1; A.x2 = sc.x2; A.yw = sc.yw; A.ysh = sc.ysh; A.out = out;
    CK(cudaMemset(sc.ysh, 0, static_cast<size_t>(M) * HID * 2));
    // the shared expert as bindp::make_layer binds it: Ls.exs = the trellis pass, Lf = the FP8 copy on the GEMMs
    bindp::Layer Ls, Lf;
    int sh = 0;
#ifndef K1V2_MAIN
    sh = D.sh_trellis && !int4sh;
    if (sh)
    {
        bindp::Expert& S = Ls.exs;
        S = A;
        S.nroutes = M; S.maxseg = (M + bindp::RS - 1) / bindp::RS;
        S.rr_row = sc.sh_rr_row; S.rr_e = sc.sh_rr_e; S.rr_sp = sc.sh_rr_sp; S.rr_w = sc.sh_rr_w;
        S.segs = sc.sh_segs; S.n_seg = sc.sh_n_seg; S.pos = nullptr; S.yw = sc.ysh;
    }
    if (sh && D.shfp8.codes[0])
    {
        const bind::LayerTensors::ShFp8& F = D.shfp8;
        const int I = F.I;
        auto ten = [](const void* p, long long a, long long b) { bind::Tensor t; t.p = p; t.nd = 2; t.shape[0] = a; t.shape[1] = b; return t; };
        const char* e = bindp::make_gemm(ten(F.codes[0], I, HID), ten(F.scales[0], I, HID / 32), M, sc.x8_big, sc.g, 0, Lf.gate);
        if (!e) e = bindp::make_gemm(ten(F.codes[1], I, HID), ten(F.scales[1], I, HID / 32), M, sc.x8_big, sc.u, 0, Lf.up);
        if (!e) e = bindp::make_gemm(ten(F.codes[2], HID, I), ten(F.scales[2], HID, I / 32), M, sc.x8_small, sc.ysh, 0, Lf.down);
        if (e) die(e);
        bindp::ShX& X = Lf.shx;
        X.on = 1; X.M = M; X.I = I; X.x = x; X.x8g = sc.x8_big; X.x8u = sc.x8_up; X.x8a = sc.x8_small;
        const size_t row = static_cast<size_t>(D.sh_tr[2]);
        X.suh_g = A.t_v1 + row * 2 * HID; X.suh_u = X.suh_g + HID; X.v2 = A.t_v2 + row * 3 * 128;
        X.g = sc.g; X.u = sc.u;
        Lf.up.x8 = sc.x8_up;
        A.sh_svh = A.t_v3 + row * HID;
    }
#endif
    auto step = [&]() {
#ifndef K1V2_MAIN
        bindp::launch_shared(Lf.shx.on ? Lf : Ls, st);
#endif
        bindp::launch_experts(A, st);
    };
    step();
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());

    // bytes this rank reads: every expert routed at least once, fc1 + fc2 trellis
    std::vector<int> te(static_cast<size_t>(M) * 8);
    std::vector<float> tw(static_cast<size_t>(M) * 8);
    CK(cudaMemcpy(te.data(), sc.tok_e, te.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tw.data(), sc.tok_w, tw.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int64_t> tabh(tab.shape[0] * 6), kmh(tab.shape[0], 0);
    CK(cudaMemcpy(tabh.data(), tab.p, tabh.size() * 8, cudaMemcpyDeviceToHost));
#ifndef K1V2_MAIN
    if (A.t_kmix) CK(cudaMemcpy(kmh.data(), A.t_kmix, kmh.size() * 8, cudaMemcpyDeviceToHost));
#endif
    std::vector<int> ex(te);
    std::sort(ex.begin(), ex.end());
    ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
    double tre = 0;
    for (int e : ex)
    {
        const int64_t* tr = &tabh[e * 6];
        for (int b = 0; b < tr[3]; ++b)
            for (int mat = 0; mat < 2; ++mat) tre += 384.0 * 8 * 32 * (kmh[e] ? (kmh[e] >> (4 * (2 * b + mat))) & 15 : tr[4]);
        tre += tr[3] * 384.0 * 8 * 32 * tr[4];
    }

    if (!dump.empty())
    {
        std::vector<uint16_t> ob(static_cast<size_t>(M) * HID);
        CK(cudaMemcpy(ob.data(), out, ob.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<__half> oh(ob.size());
        for (size_t i = 0; i < ob.size(); ++i)
        {
            uint32_t u = static_cast<uint32_t>(ob[i]) << 16;
            float f;
            memcpy(&f, &u, 4);
            oh[i] = __float2half(f);
        }
        FILE* f = fopen(dump.c_str(), "wb");
        if (!f) die("cannot write " + dump);
        int hdr[2] = {M, 1 + 2 * sh};
        fwrite(hdr, 4, 2, f);
        fwrite(xh.data(), 2, xh.size(), f);
        fwrite(te.data(), 4, te.size(), f);
        fwrite(tw.data(), 4, tw.size(), f);
        fwrite(oh.data(), 2, oh.size(), f);
        fclose(f);
        printf("dump %s: M=%d experts=%zu\n", dump.c_str(), M, ex.size());
    }
    if (iters > 0)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int i = 0; i < 5; ++i) step();
        std::vector<float> ms(iters);
        for (int r = 0; r < iters; ++r)
        {
            CK(cudaEventRecord(e0, st));
            step();
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms[r], e0, e1));
        }
        CK(cudaGetLastError());
        std::sort(ms.begin(), ms.end());
        const double us = 1e3 * ms[iters / 2];
        printf("prefill M=%d wide=%d sh=%d: median %.1f us (best %.1f), %zu experts, trellis %.1f MB -> %.1f GB/s = %.1f%% of 220\n",
               M, wide, sh, us, 1e3 * ms[0], ex.size(), tre / 1e6, tre / us / 1e3, 100 * tre / us / 1e3 / 220);
    }
#ifndef K1V2_MAIN
    if (shab && iters > 0)
    {
        // today's path: MXFP8 gate, up [I][6144] and down [6144][I], I = the TP size's widest shared slice
        const int I = (64 / nr + (64 % nr ? 1 : 0)) * 32;
        std::mt19937_64 r2(99);
        auto codes = [&](size_t n) { std::vector<uint8_t> v(n); for (auto& b : v) b = static_cast<uint8_t>(r2() & 0x77); return dev(v); };
        auto scales = [&](size_t n) { std::vector<uint8_t> v(n); for (auto& b : v) b = static_cast<uint8_t>(124 + r2() % 4); return dev(v); };
        auto ten = [](const void* p, long long a, long long b) { bind::Tensor t; t.p = p; t.nd = 2; t.shape[0] = a; t.shape[1] = b; return t; };
        bindp::Gemm gg, gu, gd;
        const char* e = bindp::make_gemm(ten(codes((size_t)I * HID), I, HID), ten(scales((size_t)I * HID / 32), I, HID / 32), M, sc.x8_big, sc.g, 0, gg);
        if (!e) e = bindp::make_gemm(ten(codes((size_t)I * HID), I, HID), ten(scales((size_t)I * HID / 32), I, HID / 32), M, sc.x8_big, sc.u, 0, gu);
        if (!e) e = bindp::make_gemm(ten(codes((size_t)HID * I), HID, I), ten(scales((size_t)HID * I / 32), HID, I / 32), M, sc.x8_small, sc.ysh, 0, gd);
        if (e) die(e);
        bindp::Xq q1, q2;
        q1.on = 1; q1.src = x; q1.ld = HID; q1.K = HID; q1.M = M; q1.dst = sc.x8_big;
        q2.on = 1; q2.src = sc.act; q2.ld = I; q2.K = I; q2.M = M; q2.dst = sc.x8_small;
        bindp::Act ac;
        ac.on = 1; ac.g = sc.g; ac.u = sc.u; ac.act = sc.act; ac.I = I; ac.M = M;
        auto mx = [&]() { bindp::launch(q1, st); bindp::launch_pair(gg, gu, st); bindp::launch(ac, st); bindp::launch(q2, st); bindp::launch(gd, st); };
        auto tr = [&]() { bindp::launch_shared(Ls, st); };
        auto f8 = [&]() { bindp::launch_shared(Lf, st); };
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        auto time = [&](auto f) {
            for (int i = 0; i < 5; ++i) f();
            std::vector<float> ms(iters);
            for (int r = 0; r < iters; ++r)
            {
                CK(cudaEventRecord(e0, st));
                f();
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[r], e0, e1));
            }
            std::sort(ms.begin(), ms.end());
            return 1e3 * ms[iters / 2];
        };
        const double tmx = time(mx), ttr = Ls.exs.on ? time(tr) : 0, tf8 = Lf.shx.on ? time(f8) : 0;
        CK(cudaGetLastError());
        printf("shared expert, prefill M=%d: MXFP8 GEMMs (I=%d) %.1f us; trellis pass %.1f us; FP8 copy on the GEMMs %.1f us\n",
               M, I, tmx, ttr, tf8);
        if (Lf.shx.on)              // the FP8 path's stages, each alone
        {
            const bindp::ShX& X = Lf.shx;
            const int rows = (X.M + 15) / 16 * 16;
            const double t_rot = time([&]() { bindp::k_shx_rotq<<<(rows * 48 + 7) / 8, 256, 0, st>>>(X); });
            const double t_g = time([&]() { bindp::launch_pair(Lf.gate, Lf.up, st); });
            const double t_mid = time([&]() { bindp::k_shx_mid<<<(rows * (X.I / 128) + 7) / 8, 256, 0, st>>>(X); });
            const double t_d = time([&]() { bindp::launch(Lf.down, st); });
            bindp::Expert B = A;
            B.sh_svh = nullptr;
            const double t_fin1 = time([&]() { bindp::launch_fin(A, st); }), t_fin0 = time([&]() { bindp::launch_fin(B, st); });
            const double t_pair = time([&]() { bindp::launch(q1, st); bindp::launch_pair(gg, gu, st); });
            const double t_act = time([&]() { bindp::launch(ac, st); bindp::launch(q2, st); });
            printf("  FP8 stages: rotate+quantize %.1f, gate/up pair %.1f, mid %.1f, down GEMM %.1f, fin +%.1f us;"
                   " today: xq + gate/up pair %.1f, act + xq %.1f\n", t_rot, t_g, t_mid, t_d, t_fin1 - t_fin0, t_pair, t_act);
        }
    }
#endif
    return 0;
}
