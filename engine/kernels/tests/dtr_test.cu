// #116 dense trellis matrices (dtr.cuh, dtr_prefill.cuh) on real release entries: output dumps for the reference check
// (engine/ref/dtr_ref.py, which also writes the inputs), and decode timing against today's int4 g32 / MXFP8 GEMVs and
// the int4 dense MLP (K1).
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_NR=4 -DDEC_S2 -DDEC_I4PAIR=1 -I.. \
//        -o dtr_test dtr_test.cu -lcublas
//   dtr_test DIR            check: every matrix of DIR/meta.txt through the decode GEMV at M = 1..8 (and 16, 32 under
//                           DEC_S2), the FP8 copy and the prefill path. Outputs go into DIR
//   dtr_test DIR --bench [--iters 640] [--sweep 1]    decode timing at M = 1, 3, 8 (CUDA graphs of 32 steps, a pool of
//                           copies so every step reads DRAM); --sweep 1 prints every (ks, sub)
//
// DIR/meta.txt, one line each:
//   mat NAME SIDE K N R       NAME.codes.bin u8 [E][entry bytes], NAME.suh.bin f16 [K], NAME.svh.bin f16 [N],
//                             NAME.x.bin f16 [R][K] input rows
//   glu GATE UP DOWN          the MLP chain on GATE.x.bin
// Outputs: NAME.gemv.M<m>.bin f32 [m][N], NAME.fp8.bin u8 [N][K], NAME.pf.bin f16 [R][N]; glu.gemv.M<m>.bin, glu.pf.bin.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "../dtr_prefill.cuh"
#include "../../h2d.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
static void die(const std::string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }
static void ok(const char* e) { if (e) die(e); }

static std::vector<uint8_t> slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) die("cannot read " + p);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void spit(const std::string& p, const void* d, size_t n)
{
    FILE* f = fopen(p.c_str(), "wb");
    if (!f || fwrite(d, 1, n, f) != n) die("cannot write " + p);
    fclose(f);
}
template <class T> static T* dev(const void* h, size_t n)
{
    T* p;
    CK(cudaMalloc(&p, n));
    CK(g6h2d::copy(p, h, n));
    return p;
}
static void dump_dev(const std::string& p, const void* d, size_t n)
{
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
    spit(p, h.data(), n);
}

struct HostMat
{
    std::string name;
    int side = 0, K = 0, N = 0, R = 0, E = 0;
    long long eb = 0;
    std::vector<uint8_t> codes, suh, svh, x;
    uint8_t* dcodes = nullptr;
    __half *dsuh = nullptr, *dsvh = nullptr, *dx = nullptr;
    dtr::Mat m;
};

static bind::Tensor ten(const void* p, std::initializer_list<long long> s)
{
    bind::Tensor t;
    t.p = p;
    for (long long v : s) t.shape[t.nd++] = v;
    return t;
}

static void load(HostMat& h, const std::string& dir)
{
    h.codes = slurp(dir + "/" + h.name + ".codes.bin");
    h.suh = slurp(dir + "/" + h.name + ".suh.bin");
    h.svh = slurp(dir + "/" + h.name + ".svh.bin");
    h.x = slurp(dir + "/" + h.name + ".x.bin");
    h.E = (h.side == dtr::SIDE_OUT ? h.N : h.K) / 128;
    h.eb = static_cast<long long>(h.codes.size()) / h.E;
    if (h.suh.size() != static_cast<size_t>(h.K) * 2 || h.svh.size() != static_cast<size_t>(h.N) * 2 ||
        h.x.size() != static_cast<size_t>(h.R) * h.K * 2 || static_cast<long long>(h.codes.size()) != h.eb * h.E)
        die(h.name + ": file sizes do not match meta.txt");
    h.dcodes = dev<uint8_t>(h.codes.data(), h.codes.size());
    h.dsuh = dev<__half>(h.suh.data(), h.suh.size());
    h.dsvh = dev<__half>(h.svh.data(), h.svh.size());
    h.dx = dev<__half>(h.x.data(), h.x.size());
    ok(dtr::make_mat(ten(h.dcodes, {h.E, h.eb}), ten(h.dsuh, {h.K}), ten(h.dsvh, {h.N}), h.side, h.m));
}

static dtr::Scratch make_scratch()
{
    dtr::Scratch s;
    s.part_n = dtr::part_floats(16384, 2, 64, G6_MX);
    s.cnt_n = 1024;
    CK(cudaMalloc(&s.part, s.part_n * 4));
    CK(cudaMalloc(&s.cnt, s.cnt_n * 4));
    CK(cudaMemset(s.cnt, 0, s.cnt_n * 4));
    return s;
}

static std::vector<int> rows_list()
{
    std::vector<int> r = {1, 2, 3, 4, 5, 6, 7, 8};
    if (G6_MX >= 16) r.push_back(16);
    if (G6_MX >= 32) { r.push_back(13); r.push_back(32); }
    return r;
}

static int check(const std::string& dir, std::vector<HostMat>& mats, const std::vector<std::array<std::string, 3>>& glus)
{
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    const dtr::Scratch sc = make_scratch();
    float* y;
    CK(cudaMalloc(&y, static_cast<size_t>(G6_MX) * 16384 * 4));
    std::map<std::string, HostMat*> by;
    int Rmax = 0;
    for (HostMat& h : mats) { by[h.name] = &h; Rmax = std::max(Rmax, h.R); }
    // prefill scratch
    uint8_t *x8, *x8u, *x8d;
    __half *g, *u, *yp, *pout;
    const size_t recb = static_cast<size_t>((Rmax + 15) / 16) * (16384 / 32) * bindp::REC;
    CK(cudaMalloc(&x8, recb)); CK(cudaMalloc(&x8u, recb)); CK(cudaMalloc(&x8d, recb));
    CK(cudaMalloc(&g, static_cast<size_t>(Rmax) * 16384 * 2)); CK(cudaMalloc(&u, static_cast<size_t>(Rmax) * 16384 * 2));
    CK(cudaMalloc(&yp, static_cast<size_t>(Rmax) * 16384 * 2)); CK(cudaMalloc(&pout, static_cast<size_t>(Rmax) * 16384 * 2));
    std::map<std::string, dtr::Fp8> fp8;

    for (HostMat& h : mats)
    {
        for (int M : rows_list())
        {
            if (M > h.R) continue;
            dtr::Gemv gv;
            ok(dtr::make_out(h.m, M, h.dx, h.K, nullptr, y, h.N, dtr::O_F32, sc, 0, 0, 1, gv));
            dtr::launch(gv, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            dump_dev(dir + "/" + h.name + ".gemv.M" + std::to_string(M) + ".bin", y, static_cast<size_t>(M) * h.N * 4);
        }
        dtr::Fp8& F = fp8[h.name];
        ok(dtr::make_fp8(h.m, st, F));
        dump_dev(dir + "/" + h.name + ".fp8.bin", F.codes, static_cast<size_t>(h.N) * h.K);
        dtr::PfMat P;
        ok(dtr::make_pf(h.m, F, h.R, h.dx, h.K, x8, yp, pout, h.N, 0, P));
        dtr::launch(P, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        dump_dev(dir + "/" + h.name + ".pf.bin", pout, static_cast<size_t>(h.R) * h.N * 2);
        printf("%s: side %d K %d N %d entries %d x %lld B: decode, FP8 copy, prefill (%d rows) dumped\n", h.name.c_str(), h.side, h.K,
               h.N, h.E, h.eb, h.R);
    }
    for (const auto& gl : glus)
    {
        HostMat &G = *by.at(gl[0]), &U = *by.at(gl[1]), &D = *by.at(gl[2]);
        u32* xd;
        CK(cudaMalloc(&xd, static_cast<size_t>(g6mx::NT) * (G.N / 16) * 64 * 4));
        for (int M : rows_list())
        {
            if (M > G.R) continue;
            dtr::Gemv a, b;
            ok(dtr::make_glu(G.m, U.m, D.m, M, G.dx, G.K, xd, sc, 0, 0, 1, a));
            ok(dtr::make_out(D.m, M, nullptr, 0, xd, y, D.N, dtr::O_F32, sc, 512, 0, 1, b));
            dtr::launch(a, st);
            dtr::launch(b, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            dump_dev(dir + "/glu.gemv.M" + std::to_string(M) + ".bin", y, static_cast<size_t>(M) * D.N * 4);
        }
        const bool same = G.suh == U.suh;
        dtr::PfMlp P;
        ok(dtr::make_pf_mlp(G.m, U.m, D.m, fp8[G.name], fp8[U.name], fp8[D.name], G.R, G.dx, G.K, x8, x8u, x8d, g, u, yp, pout, D.N,
                            0, same, P));
        dtr::launch(P, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        dump_dev(dir + "/glu.pf.bin", pout, static_cast<size_t>(G.R) * D.N * 2);
        printf("glu %s %s %s: decode chain and prefill dumped (gate and up share their rotation: %s)\n", gl[0].c_str(), gl[1].c_str(),
               gl[2].c_str(), same ? "yes" : "no");
        CK(cudaFree(xd));
    }
    return 0;
}

// ------------------------------------------------------------------------------------------ timing
struct Timer
{
    cudaStream_t st;
    cudaEvent_t e0, e1;
    int iters;
    Timer(cudaStream_t s, int it) : st(s), iters(it) { CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); }
    // us per step, from replays of a CUDA graph of 32 steps (decode runs as captured graphs)
    template <class F> double operator()(F f)
    {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 32; ++i) f(i);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        const int reps = std::max(1, iters / 32);
        std::vector<double> t;
        for (int r = 0; r < 5; ++r)
        {
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            t.push_back(1e3 * ms / (reps * 32));
        }
        CK(cudaGetLastError());
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
};

// P copies of a device buffer. Steps cycle through them and read DRAM, past the 24 MB L2.
static uint8_t* pool_of(const void* src, size_t n, int P)
{
    uint8_t* p;
    CK(cudaMalloc(&p, n * P));
    for (int i = 0; i < P; ++i) CK(cudaMemcpy(p + n * i, src, n, cudaMemcpyDeviceToDevice));
    return p;
}
static int pool_n(size_t n) { return static_cast<int>(std::max<size_t>(2, std::min<size_t>(64, static_cast<size_t>(800e6) / n))); }

static std::vector<uint8_t> rand_bytes(size_t n, std::mt19937_64& rng)
{
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i += 8) { const uint64_t r = rng(); memcpy(&v[i], &r, std::min<size_t>(8, n - i)); }
    return v;
}
// int4 g32 records [O/16][K/32][288] with sane fp16 scales; MXFP8 records [O/16][K/32][528] without NaN codes
static std::vector<uint8_t> i4_records(int O, int K, std::mt19937_64& rng)
{
    std::vector<uint8_t> v = rand_bytes(static_cast<size_t>(O / 16) * (K / 32) * 288, rng);
    for (size_t gi = 0; gi < v.size() / 288; ++gi)
        for (int i = 0; i < 16; ++i) { const __half h = __float2half(0.002f + 0.004f * (rng() % 1000) / 1000.f); memcpy(&v[gi * 288 + 256 + i * 2], &h, 2); }
    return v;
}
static std::vector<uint8_t> mx_records(int O, int K, std::mt19937_64& rng)
{
    std::vector<uint8_t> v = rand_bytes(static_cast<size_t>(O / 16) * (K / 32) * 528, rng);
    for (size_t gi = 0; gi < v.size() / 528; ++gi)
    {
        for (int i = 0; i < 512; ++i) if ((v[gi * 528 + i] & 0x7f) == 0x7f) v[gi * 528 + i] ^= 1;
        for (int i = 0; i < 16; ++i) v[gi * 528 + 512 + i] = static_cast<uint8_t>(118 + rng() % 5);
    }
    return v;
}

static int bench(std::vector<HostMat>& mats, const std::vector<std::array<std::string, 3>>& glus, int iters, int sweep)
{
    CK(bind::init());
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    Timer time(st, iters);
    const dtr::Scratch sc = make_scratch();
    std::mt19937_64 rng(7);
    float* y;
    CK(cudaMalloc(&y, static_cast<size_t>(G6_MX) * 16384 * 4));
    u32* xf;
    CK(cudaMalloc(&xf, static_cast<size_t>(g6mx::NT) * 16384 / 16 * 64 * 4));
    const int Ms[3] = {1, 3, 8};
    std::map<std::string, HostMat*> by;
    for (HostMat& h : mats) by[h.name] = &h;

    // the trellis GEMV at the binder's split (dtr::split_for), or with --sweep 1 the best of every (ks, sub), on a pool of
    // copies of the real entries
    auto run_tr = [&](const dtr::Gemv& g, const uint8_t* pool, size_t mb, int P) {
        return time([&](int i) { dtr::Gemv h = g; h.a.w[0] = pool + mb * (i % P); dtr::launch(h, st); });
    };
    auto best_tr = [&](const dtr::Mat& m0, const uint8_t* pool, size_t mb, int P, int M, const __half* x, int ldx, const u32* xa, int odt,
                       int& bks, int& bsub) {
        dtr::Gemv g;
        ok(dtr::make_out(m0, M, x, ldx, xa, y, m0.N, odt, sc, 0, 0, 1, g));
        double best = run_tr(g, pool, mb, P);
        bks = g.a.ks; bsub = g.sub;
        for (int sub = 1; sub <= 2 && sweep; ++sub)
            for (int ks : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32})
            {
                if (dtr::make_out(m0, M, x, ldx, xa, y, m0.N, odt, sc, 0, ks, sub, g) || g.a.ks != ks) continue;
                const double t = run_tr(g, pool, mb, P);
                printf("    ks %2d sub %d: %7.1f us\n", ks, sub, t);
                if (t < best) { best = t; bks = ks; bsub = sub; }
            }
        return best;
    };

    for (HostMat& h : mats)
    {
        const size_t mb = h.codes.size();
        const int P = pool_n(mb);
        uint8_t* pool = pool_of(h.dcodes, mb, P);
        // today's records of the same shape: int4 g32 (o_proj, the dense MLP) and MXFP8 (q_b)
        const size_t mi = static_cast<size_t>(h.N / 16) * (h.K / 32) * 288, mm = static_cast<size_t>(h.N / 16) * (h.K / 32) * 528;
        const int Pi = pool_n(mi), Pm = pool_n(mm);
        std::vector<uint8_t> hi = i4_records(h.N, h.K, rng), hm = mx_records(h.N, h.K, rng);
        uint8_t* di = pool_of(dev<uint8_t>(hi.data(), mi), mi, Pi);
        uint8_t* dm = pool_of(dev<uint8_t>(hm.data(), mm), mm, Pm);
#if DEC_I4PAIR
        for (int i = 0; i < Pi; ++i) launch_i4pair(di + mi * i, h.N / 16, h.K / 32, 8, st);
        CK(cudaStreamSynchronize(st));
#endif
        const bool qb = h.K == 2048;     // today's splits (bind::make_layer): q_b MXFP8 tpc 1 ks 16, o_proj tpc 2 ks 8
        printf("%s (K %d, N %d): trellis K4 %.2f MB, int4 g32 %.2f MB, MXFP8 %.2f MB\n", h.name.c_str(), h.K, h.N, mb / 1e6, mi / 1e6, mm / 1e6);
        for (int M : Ms)
        {
            int ks = 0, sub = 0;
            const double tt = best_tr(h.m, pool, mb, P, M, h.dx, h.K, nullptr, dtr::O_F16, ks, sub);
            bind::Xfrag xa;
            xa.on = 1; xa.src = h.dx; xa.ld = h.K; xa.K = h.K; xa.M = M; xa.f16 = xf;
            bind::Gv gi, gm;
            ok(bind::make_gv(ten(di, {h.N / 16, h.K / 32, 288}), bind::INT4, 2, 8, M, xf, y, gv::O_F16, gi));
            ok(bind::make_gv(ten(dm, {h.N / 16, h.K / 32, 528}), bind::MXFP8, qb ? 1 : 2, qb ? 16 : 8, M, xf, y, gv::O_F16, gm));
            const double ti = time([&](int i) { bind::launch(xa, st); bind::Gv g = gi; g.w = di + mi * (i % Pi); bind::launch(g, st); });
            const double tm = time([&](int i) { bind::launch(xa, st); bind::Gv g = gm; g.w = dm + mm * (i % Pm); bind::launch(g, st); });
            printf("  M=%d: trellis %6.1f us (ks %d sub %d, %5.1f GB/s) | int4 g32 %6.1f us | MXFP8 %6.1f us | trellis/int4 %.2f, trellis/MXFP8 %.2f\n",
                   M, tt, ks, sub, mb / tt / 1e3, ti, tm, tt / ti, tt / tm);
        }
        CK(cudaFree(pool)); CK(cudaFree(di)); CK(cudaFree(dm));
    }

    // prefill: the FP8 copy on the MXFP8 GEMM inside its rotations (rotq, GEMM, post) against today's form of the same
    // GEMM (k_xq8 on unrotated rows, GEMM; MXFP8 weights of the same shape), at 512 and 4096 rows
    {
        const int PR[2] = {512, 4096};
        __half *px, *pyp, *pout;
        uint8_t* px8;
        CK(cudaMalloc(&px, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&pyp, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&pout, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&px8, static_cast<size_t>(4096 / 16) * (16384 / 32) * bindp::REC));
        {
            std::vector<__half> hx(static_cast<size_t>(4096) * 16384);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = __float2half(nd(rng));
            CK(g6h2d::copy(px, hx.data(), hx.size() * 2));
        }
        for (HostMat& h : mats)
        {
            dtr::Fp8 F;
            ok(dtr::make_fp8(h.m, st, F));
            for (int R : PR)
            {
                dtr::PfMat P;
                ok(dtr::make_pf(h.m, F, R, px, h.K, px8, pyp, pout, h.N, 1, P));
                bindp::Xq q;
                q.on = 1; q.src = px; q.ld = h.K; q.K = h.K; q.M = R; q.dst = px8;
                bindp::Gemm g = P.gemm;
                g.y = pout; g.bf16 = 1;
                const double tt = time([&](int) { dtr::launch(P, st); });
                const double t0 = time([&](int) { bindp::launch(q, st); bindp::launch(g, st); });
                printf("%s prefill %4d rows: trellis FP8 copy %7.1f us | today (xq8 + GEMM) %7.1f us | ratio %.2f\n", h.name.c_str(), R, tt, t0, tt / t0);
            }
            dtr::free_fp8(F);
        }
        CK(cudaFree(px)); CK(cudaFree(pyp)); CK(cudaFree(pout)); CK(cudaFree(px8));
    }

    // the dense MLP: gate + up (one launch) + down, against today's int4 dense MLP in K1 (fused fc1 + fc2, the default)
    for (const auto& gl : glus)
    {
        HostMat &G = *by.at(gl[0]), &U = *by.at(gl[1]), &D = *by.at(gl[2]);
        const size_t mg = G.codes.size(), mu = U.codes.size(), md = D.codes.size(), all = mg + mu + md;
        const int P = pool_n(all);
        uint8_t* pool;
        CK(cudaMalloc(&pool, all * P));
        for (int i = 0; i < P; ++i)
        {
            CK(cudaMemcpy(pool + all * i, G.dcodes, mg, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(pool + all * i + mg, U.dcodes, mu, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(pool + all * i + mg + mu, D.dcodes, md, cudaMemcpyDeviceToDevice));
        }
        u32* xd;
        CK(cudaMalloc(&xd, static_cast<size_t>(g6mx::NT) * (G.N / 16) * 64 * 4));
        const int I = G.N;
        // int4 dense records: fc1 [I/16][192][576] (gate and up interleaved), fc2 [384][I/32][288]
        const size_t m1 = static_cast<size_t>(I / 16) * 192 * 576, m2 = static_cast<size_t>(384) * (I / 32) * 288, mall = m1 + m2;
        const int Pi = pool_n(mall);
        std::vector<uint8_t> h1 = i4_records(I * 2, HID, rng), h2 = i4_records(HID, I, rng);
        uint8_t* ipool;
        CK(cudaMalloc(&ipool, mall * Pi));
        for (int i = 0; i < Pi; ++i)
        {
            CK(g6h2d::copy(ipool + mall * i, h1.data(), m1));
            CK(g6h2d::copy(ipool + mall * i + m1, h2.data(), m2));
        }
        void* blk;
        CK(cudaMalloc(&blk, bind::scratch_bytes()));
        CK(cudaMemset(blk, 0, bind::scratch_bytes()));
        const bind::Scratch bsc = bind::scratch_at(blk);
        uint16_t* out;
        CK(cudaMalloc(&out, static_cast<size_t>(G6_MX) * HID * 2));
        printf("dense MLP (I %d per rank): trellis K4 %.2f MB, int4 g32 %.2f MB\n", I, all / 1e6, mall / 1e6);
        for (int M : Ms)
        {
            double best = 1e30;
            int b1 = 0, s1 = 0, b2 = 0, s2 = 0;
            // down alone first (on tables the GLU launch wrote), then gate + up and down together
            dtr::Gemv a0;
            ok(dtr::make_glu(G.m, U.m, D.m, M, G.dx, G.K, xd, sc, 0, 0, 1, a0));
            dtr::launch(a0, st);
            CK(cudaStreamSynchronize(st));
            const dtr::Mat dm0 = D.m;
            const double tdown = best_tr(dm0, pool + mg + mu, all, P, M, nullptr, 0, xd, dtr::O_BF16, b2, s2);
            auto glu_time = [&](int ks, int sub, int dks, int dsub, int& gks, int& gsub) {
                dtr::Gemv a, b;
                if (dtr::make_glu(G.m, U.m, D.m, M, G.dx, G.K, xd, sc, 0, ks, sub, a) || (ks && a.a.ks != ks)) return 1e30;
                ok(dtr::make_out(D.m, M, nullptr, 0, xd, out, HID, dtr::O_BF16, sc, 512, dks, dsub, b));
                gks = a.a.ks; gsub = a.sub;
                return time([&](int i) {
                    dtr::Gemv x = a, z = b;
                    x.a.w[0] = pool + all * (i % P); x.a.w[1] = x.a.w[0] + mg; z.a.w[0] = x.a.w[0] + mg + mu;
                    dtr::launch(x, st);
                    dtr::launch(z, st);
                });
            };
            best = glu_time(0, 1, 0, 1, b1, s1);                 // the binder's splits for both launches
            for (int sub = 1; sub <= 2 && sweep; ++sub)
                for (int ks : {1, 2, 3, 4, 6, 8, 12, 16})
                {
                    int gk = 0, gs = 0;
                    const double t = glu_time(ks, sub, b2, s2, gk, gs);
                    if (t > 1e29) continue;
                    printf("    glu ks %2d sub %d + down: %7.1f us\n", ks, sub, t);
                    if (t < best) { best = t; b1 = ks; s1 = sub; }
                }
            bind::LayerTensors T;
            T.t[bind::T_D_FC1] = ten(ipool, {I / 16, 192, 576});
            T.t[bind::T_D_FC2] = ten(ipool + m1, {384, I / 32, 288});
            bind::K1 k;
            bind::Tensor sh[3];
            static const int fuse = getenv("G6_FUSE") ? atoi(getenv("G6_FUSE")) : 1;
            ok(bind::make_k1(T, M, bsc, G.dx, bsc.xf_x2, bind::PlanPtrs(), out, 1, k, sh, fuse));
            bind::Xfrag xa;
            xa.on = 1; xa.src = G.dx; xa.ld = HID; xa.K = HID; xa.M = M; xa.f16 = bsc.xf_x2;
            const double ti = time([&](int i) {
                bind::K1 kk = k;
                kk.a.s_fc1 = ipool + mall * (i % Pi); kk.a.s_fc2 = kk.a.s_fc1 + m1;
                bind::launch(xa, st);
                bind::launch_k1(kk, st);
            });
            printf("  M=%d: trellis %6.1f us (gate+up ks %d sub %d; down ks %d sub %d, alone %.1f us) | int4 K1 dense %6.1f us | trellis/int4 %.2f\n",
                   M, best, b1, s1, b2, s2, tdown, ti, best / ti);
        }
        CK(cudaFree(pool)); CK(cudaFree(ipool)); CK(cudaFree(blk)); CK(cudaFree(xd)); CK(cudaFree(out));
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) die("usage: dtr_test DIR [--bench] [--iters N] [--sweep 1]");
    const std::string dir = argv[1];
    bool b = false;
    int iters = 640, sweep = 0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--bench") b = true;
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--sweep" && i + 1 < argc) sweep = atoi(argv[++i]);
        else die("unknown argument " + a);
    }
    CK(dtr::init());
    std::vector<HostMat> mats;
    std::vector<std::array<std::string, 3>> glus;
    std::ifstream f(dir + "/meta.txt");
    if (!f) die("no " + dir + "/meta.txt");
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream s(line);
        std::string kind;
        s >> kind;
        if (kind == "mat")
        {
            HostMat h;
            s >> h.name >> h.side >> h.K >> h.N >> h.R;
            mats.push_back(std::move(h));
        }
        else if (kind == "glu")
        {
            std::array<std::string, 3> g;
            s >> g[0] >> g[1] >> g[2];
            glus.push_back(g);
        }
    }
    for (HostMat& h : mats) load(h, dir);
    return b ? bench(mats, glus, iters, sweep) : check(dir, mats, glus);
}
