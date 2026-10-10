// #116 dense trellis matrices (dtr.cuh, and the prefill GEMMs of bind_prefill.cuh) on real release entries: output dumps for the reference check
// (engine/ref/dtr_ref.py, which also writes the inputs), and decode timing against today's int4 g32 / MXFP8 GEMVs and
// the int4 dense MLP (K1).
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_NR=4 -DDEC_S2 -DDEC_I4PAIR=1 -I.. \
//        -o dtr_test dtr_test.cu -lcublas
//   dtr_test DIR            check: every matrix of DIR/meta.txt through the decode GEMV at M = 1..8 (and 16, 32 under
//                           DEC_S2), the FP8 copy and the prefill path. Outputs go into DIR
//   dtr_test DIR --bind     the binders the engine uses on one dense layer (bind_check below)
//   dtr_test DIR --bench [--iters 640] [--sweep 1] [--pf-only]    decode timing at M = 1, 3, 8 (CUDA graphs of 32 steps, a pool of
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
#include "../bind_prefill.cuh"
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
static bool same_dev(const void* a, const void* b, size_t n)
{
    std::vector<uint8_t> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n, cudaMemcpyDeviceToHost));
    return x == y;
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

static bind::Tensor fp8_codes(const dtr::Fp8& F) { return ten(F.codes, {F.N, F.K}); }
static bind::Tensor fp8_scales(const dtr::Fp8& F) { return ten(F.scales, {F.N, F.K / 32}); }
// one matrix on R prefill rows as bindp::make_layer binds it: rotated records (Gemm::pre, Xq::suh), the GEMM on the FP8
// copy or (direct) on the tiles (Gemm::tw), the output rotation in its epilogue (Gemm::osvh; out = 0: plain fp16 output)
static bindp::Gemm pf_gemm(const dtr::Mat& m, const dtr::Fp8& F, int R, const __half* x, int ldx, uint8_t* x8, void* y, int bf16, bool out,
                           bool direct = false)
{
    bindp::Gemm g;
    if (direct) ok(bindp::make_gemm_tr(m, R, x8, y, bf16, g));
    else ok(bindp::make_gemm(fp8_codes(F), fp8_scales(F), R, x8, y, bf16, g));
    if (x) { g.pre.on = 1; g.pre.src = x; g.pre.ld = ldx; g.pre.K = m.K; g.pre.M = R; g.pre.dst = x8; g.pre.suh = m.suh; }
    if (out) g.osvh = m.svh;
    return g;
}
// the dense MLP on R prefill rows: gate and up (one paired launch), the mid pass (Act::svh_g) into down's records, down
struct PfMlp { bindp::Gemm gate, up, down; bindp::Act act; };
static PfMlp pf_mlp(const dtr::Mat& G, const dtr::Mat& U, const dtr::Mat& D, const dtr::Fp8& Fg, const dtr::Fp8& Fu, const dtr::Fp8& Fd, int R,
                    const __half* x, uint8_t* x8, uint8_t* x8u, uint8_t* x8d, __half* g, __half* u, void* y, int bf16, bool direct = false)
{
    PfMlp P;
    P.gate = pf_gemm(G, Fg, R, x, G.K, x8, g, 0, false, direct);
    P.up = pf_gemm(U, Fu, R, x, U.K, x8u, u, 0, false, direct);
    P.down = pf_gemm(D, Fd, R, nullptr, 0, x8d, y, bf16, true, direct);
    P.act.on = 1; P.act.g = g; P.act.u = u; P.act.I = G.N; P.act.M = R;
    P.act.svh_g = G.svh; P.act.svh_u = U.svh; P.act.suh_d = D.suh; P.act.x8 = x8d;
    return P;
}
static void launch(const PfMlp& P, cudaStream_t st)
{
    bindp::launch_pair(P.gate, P.up, st);
    bindp::launch(P.act, st);
    bindp::launch(P.down, st);
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
    __half *g, *u, *pout, *pout2;
    int bad = 0;
    const size_t recb = static_cast<size_t>((Rmax + 15) / 16) * (16384 / 32) * bindp::REC;
    CK(cudaMalloc(&x8, recb)); CK(cudaMalloc(&x8u, recb)); CK(cudaMalloc(&x8d, recb));
    CK(cudaMalloc(&g, static_cast<size_t>(Rmax) * 16384 * 2)); CK(cudaMalloc(&u, static_cast<size_t>(Rmax) * 16384 * 2));
    CK(cudaMalloc(&pout, static_cast<size_t>(Rmax) * 16384 * 2));
    CK(cudaMalloc(&pout2, static_cast<size_t>(Rmax) * 16384 * 2));
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
        bindp::launch(pf_gemm(h.m, F, h.R, h.dx, h.K, x8, pout, 0, true), st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        dump_dev(dir + "/" + h.name + ".pf.bin", pout, static_cast<size_t>(h.R) * h.N * 2);
        bindp::launch(pf_gemm(h.m, F, h.R, h.dx, h.K, x8, pout2, 0, true, true), st);
        const bool same = same_dev(pout, pout2, static_cast<size_t>(h.R) * h.N * 2);
        printf("%s: side %d K %d N %d entries %d x %lld B: decode, FP8 copy, prefill (%d rows) dumped; prefill from the tiles %s the copy's\n",
               h.name.c_str(), h.side, h.K, h.N, h.E, h.eb, h.R, same ? "equals" : "DIFFERS FROM");
        if (!same) bad = 1;
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
        launch(pf_mlp(G.m, U.m, D.m, fp8[G.name], fp8[U.name], fp8[D.name], G.R, G.dx, x8, x8u, x8d, g, u, pout, 0), st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        dump_dev(dir + "/glu.pf.bin", pout, static_cast<size_t>(G.R) * D.N * 2);
        launch(pf_mlp(G.m, U.m, D.m, fp8[G.name], fp8[U.name], fp8[D.name], G.R, G.dx, x8, x8u, x8d, g, u, pout2, 0, true), st);
        const bool eq = same_dev(pout, pout2, static_cast<size_t>(G.R) * D.N * 2);
        if (!eq) bad = 1;
        printf("glu %s %s %s: decode chain and prefill dumped (gate and up share their rotation: %s); prefill from the tiles %s the copy's\n",
               gl[0].c_str(), gl[1].c_str(), gl[2].c_str(), same ? "yes" : "no", eq ? "equals" : "DIFFERS FROM");
        CK(cudaFree(xd));
    }
    return bad;
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

static int bench(std::vector<HostMat>& mats, const std::vector<std::array<std::string, 3>>& glus, int iters, int sweep, int pfonly)
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
        if (pfonly) break;
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

    // prefill: the FP8 copy on the MXFP8 GEMM inside its rotations (rotated records, GEMM with the output rotation in its
    // epilogue) against today's form of the same GEMM (k_xq8 on unrotated rows, GEMM), BF16 out, at 512 and 4096 rows
    {
        const int PR[3] = {512, 2048, 4096};
        __half *px, *pout, *pout2;
        uint8_t* px8;
        CK(cudaMalloc(&px, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&pout, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&pout2, static_cast<size_t>(4096) * 16384 * 2));
        CK(cudaMalloc(&px8, static_cast<size_t>(4096 / 16) * (16384 / 32) * bindp::REC));
        {
            std::vector<__half> hx(static_cast<size_t>(4096) * 16384);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = __float2half(nd(rng));
            CK(g6h2d::copy(px, hx.data(), hx.size() * 2));
        }
        // the pair-owner o_proj shape (G6_PF_OWN=4: two members' entries back to back, K = 2 x 4096), from o_proj.r0 twice
        std::vector<HostMat> pair;
        for (HostMat& h : mats)
            if (h.name.rfind("o_proj", 0) == 0 && h.K == 4096)
            {
                HostMat p = h;
                p.name = h.name + ".pairK8192";
                p.K = 2 * h.K; p.E = 2 * h.E;
                std::vector<uint8_t> c2(h.codes), s2(h.suh);
                c2.insert(c2.end(), h.codes.begin(), h.codes.end());
                s2.insert(s2.end(), h.suh.begin(), h.suh.end());
                p.dcodes = dev<uint8_t>(c2.data(), c2.size());
                p.dsuh = dev<__half>(s2.data(), s2.size());
                ok(dtr::make_mat(ten(p.dcodes, {p.E, h.eb}), ten(p.dsuh, {p.K}), ten(h.dsvh, {h.N}), h.side, p.m));
                pair.push_back(p);
                break;
            }
        std::vector<HostMat*> pm;
        for (HostMat& h : mats) pm.push_back(&h);
        for (HostMat& h : pair) pm.push_back(&h);
        for (HostMat* hp : pm)
        {
            HostMat& h = *hp;
            dtr::Fp8 F;
            ok(dtr::make_fp8(h.m, st, F));
            for (int R : PR)
            {
                const bindp::Gemm g = pf_gemm(h.m, F, R, px, h.K, px8, pout, 1, true);
                const bindp::Gemm gd = pf_gemm(h.m, F, R, px, h.K, px8, pout, 1, true, true);
                bindp::Gemm g0 = g;
                g0.pre.suh = nullptr;                 // today: k_xq8, no output rotation
                g0.osvh = nullptr;
                const double tt = time([&](int) { bindp::launch(g, st); });
                const double td = time([&](int) { bindp::launch(gd, st); });
                const double t0 = time([&](int) { bindp::launch(g0, st); });
                bindp::Gemm gd2 = gd;                 // the tiles' output against the copy's, bit for bit
                gd2.y = pout2;
                bindp::launch(g, st);
                bindp::launch(gd2, st);
                CK(cudaStreamSynchronize(st));
                const bool eq = same_dev(pout, pout2, static_cast<size_t>(R) * h.N * 2);
                printf("%s prefill %4d rows: tiles %7.1f us | FP8 copy %7.1f us | today (xq8 + GEMM) %7.1f us | tiles/copy %.2f, tiles/today %.2f | %s\n",
                       h.name.c_str(), R, td, tt, t0, td / tt, td / t0, eq ? "bits equal" : "BITS DIFFER");
            }
            dtr::free_fp8(F);
        }
        CK(cudaFree(px)); CK(cudaFree(pout)); CK(cudaFree(pout2)); CK(cudaFree(px8));
    }

    // the dense MLP: gate + up (one launch) + down, against today's int4 dense MLP in K1 (fused fc1 + fc2, the default)
    for (const auto& gl : glus)
    {
        if (pfonly) break;
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

// ------------------------------------------------------------------------------------------ bindings
// --bind: one dense layer's q_b, o_proj and MLP (DIR's q_b.r0, o_proj.r0, gate.r0, up.r0, down.r0) through the binders the
// engine uses: bind::make_layer with launch_q_b / launch_o_proj / launch_mlp at every row count (DEC_SLOT: outputs in a
// send slot, its doorbell checked), and bindp::make_layer's q_b, o_proj and dense MLP GEMMs on R prefill rows. The other
// tensors make_layer needs (q_a, kv_a) are random. Outputs: bind.q_b.M<m>.bin f16 [m][N], bind.o_proj.M<m>.bin and
// bind.mlp.M<m>.bin BF16 [m][6144]; bind.pf.q_b.bin f16 [R][N], bind.pf.o_proj.bin, bind.pf.mlp.bin BF16 [R][6144].
template <class T> static T* dzero(size_t n)
{
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
static uint8_t* dev_bytes(const std::vector<uint8_t>& v) { return dev<uint8_t>(v.data(), v.size()); }
static std::vector<uint8_t> e4m3_codes(size_t n, std::mt19937_64& rng)
{
    std::vector<uint8_t> v = rand_bytes(n, rng);
    for (auto& b : v) if ((b & 0x7f) == 0x7f) b ^= 1;
    return v;
}
static int bind_check(const std::string& dir, std::vector<HostMat>& mats)
{
    std::map<std::string, HostMat*> by;
    for (HostMat& h : mats) by[h.name] = &h;
    for (const char* n : {"q_b.r0", "o_proj.r0", "gate.r0", "up.r0", "down.r0"})
        if (!by.count(n)) die(std::string("--bind needs ") + n + " in meta.txt (a dense layer, rank 0)");
    HostMat &Q = *by["q_b.r0"], &O = *by["o_proj.r0"], &G = *by["gate.r0"], &U = *by["up.r0"], &D = *by["down.r0"];
    const int R = Q.R;
    if (O.R != R || G.R != R || D.R != R) die("--bind: every matrix needs the same row count");
    CK(bind::init());
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    std::mt19937_64 rng(11);

    // decode: bind::LayerTensors of a dense layer
    bind::LayerTensors T;
    T.t[bind::T_Q_A] = ten(dev_bytes(i4_records(2048, HID, rng)), {2048 / 16, HID / 32, 288});
    T.t[bind::T_KV_A] = ten(dev_bytes(mx_records(576, HID, rng)), {576 / 16, HID / 32, 528});
    T.mats.q_b = Q.m; T.mats.o_proj = O.m; T.mats.gate = G.m; T.mats.up = U.m; T.mats.down = D.m;
#if DEC_I4PAIR
    bind::i4pair(T, st);
    CK(cudaStreamSynchronize(st));
#endif
    void* blk = dzero<char>(bind::scratch_bytes());
    const bind::Scratch sc = bind::scratch_at(blk);
    bind::Io io;
    std::vector<__half> hx(static_cast<size_t>(G6_MX) * HID);
    for (auto& v : hx) v = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.f);
    io.x = dev<__half>(hx.data(), hx.size() * 2);
    io.q_a = dzero<__half>(static_cast<size_t>(G6_MX) * 2048);
    io.q_a_n = Q.dx;
    io.kv_a = dzero<__half>(static_cast<size_t>(G6_MX) * 576);
    io.q_b = dzero<__half>(static_cast<size_t>(G6_MX) * Q.N);
    io.attn = O.dx;
    io.attn_part = dzero<uint16_t>(static_cast<size_t>(G6_MX) * HID);
    io.x2 = G.dx;
    io.mlp_part = dzero<uint16_t>(static_cast<size_t>(G6_MX) * HID);
#ifdef DEC_SLOT
    const size_t stride = static_cast<size_t>(G6_MX) * HID * 2 + 256;
    char* send = dzero<char>(2 * 2 * stride);
    unsigned long long* bell = dzero<unsigned long long>(2);
    u64* stepp = dzero<u64>(1);
    unsigned* ticket = dzero<unsigned>(2);
    auto slot = [&](int c) {
        dslot::Slot q;
        q.send = send; q.bell = bell; q.stepp = stepp; q.stride = stride; q.ncoll = 2; q.c = c; q.ticket = ticket + c;
        return q;
    };
#endif
    for (int M : rows_list())
    {
        if (M > R) continue;
        bind::Layer L;
        ok(bind::make_layer(T, M, sc, io, L, 1));
        if (!L.q_b_t.on || !L.o_proj_t.on || !L.glu_t.on || !L.down_t.on || L.q_b.on || L.o_proj.on || L.k1.on || L.xf_attn.on)
            die("--bind: make_layer did not bind the trellis matrices");
#ifdef DEC_SLOT
        bind::set_slots(L, slot(0), slot(1));
        CK(cudaMemset(bell, 0, 2 * sizeof(unsigned long long)));
#endif
        bind::launch_q_b(L, st);
        bind::launch_o_proj(L, st);
        bind::launch_mlp(L, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        const std::string m = std::to_string(M);
        dump_dev(dir + "/bind.q_b.M" + m + ".bin", io.q_b, static_cast<size_t>(M) * Q.N * 2);
#ifdef DEC_SLOT
        dump_dev(dir + "/bind.o_proj.M" + m + ".bin", send, static_cast<size_t>(M) * HID * 2);
        dump_dev(dir + "/bind.mlp.M" + m + ".bin", send + stride, static_cast<size_t>(M) * HID * 2);
#if DEC_SLOT >= 2
        unsigned long long hb[2];
        CK(cudaMemcpy(hb, bell, sizeof hb, cudaMemcpyDeviceToHost));
        if (hb[0] != 1 || hb[1] != 2) die("--bind: send-slot doorbells " + std::to_string(hb[0]) + ", " + std::to_string(hb[1]) + " (want 1, 2)");
#endif
#else
        dump_dev(dir + "/bind.o_proj.M" + m + ".bin", io.attn_part, static_cast<size_t>(M) * HID * 2);
        dump_dev(dir + "/bind.mlp.M" + m + ".bin", io.mlp_part, static_cast<size_t>(M) * HID * 2);
#endif
    }

    // prefill: bindp::Tensors of the same layer with the FP8 copies and the vectors
    bindp::Tensors P;
    auto mx = [&](int id, int N, int K) {
        P.t[id] = ten(dev_bytes(e4m3_codes(static_cast<size_t>(N) * K, rng)), {N, K});
        std::vector<uint8_t> sc8(static_cast<size_t>(N) * K / 32);
        for (auto& b : sc8) b = static_cast<uint8_t>(118 + rng() % 5);
        P.s[id] = ten(dev_bytes(sc8), {N, K / 32});
    };
    mx(bindp::T_Q_A, 2048, HID);
    mx(bindp::T_KV_A, 576, HID);
    const HostMat* hm[dtr::NMAT] = {&Q, &O, &G, &U, &D};
    const int id[dtr::NMAT] = {bindp::T_Q_B, bindp::T_O_PROJ, bindp::T_D_GATE, bindp::T_D_UP, bindp::T_D_DOWN};
    dtr::Fp8 F[dtr::NMAT];
    for (int i = 0; i < dtr::NMAT; ++i)
    {
        ok(dtr::make_fp8(hm[i]->m, st, F[i]));
        P.t[id[i]] = fp8_codes(F[i]);
        P.s[id[i]] = fp8_scales(F[i]);
        P.tsuh[i] = hm[i]->m.suh;
        P.tsvh[i] = hm[i]->m.svh;
    }
    void* pblk = dzero<char>(bindp::scratch_bytes(R));
    const bindp::Scratch psc = bindp::scratch_at(pblk, R);
    bindp::Io pio;
    pio.x = dzero<__half>(static_cast<size_t>(R) * HID);
    pio.q_a = dzero<__half>(static_cast<size_t>(R) * 2048);
    pio.q_a_n = Q.dx;
    pio.kv_a = dzero<__half>(static_cast<size_t>(R) * 576);
    pio.q_b = dzero<__half>(static_cast<size_t>(R) * Q.N);
    pio.attn = O.dx;
    pio.attn_part = dzero<uint16_t>(static_cast<size_t>(R) * HID);
    pio.x2 = G.dx;
    pio.mlp_part = dzero<uint16_t>(static_cast<size_t>(R) * HID);
    cublasHandle_t blas;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) die("cublasCreate");
    bindp::Layer B;
    ok(bindp::make_layer(P, R, psc, pio, blas, B));
    if (!B.q_b.osvh || !B.o_proj.osvh || !B.down.osvh || !B.xq_attn.suh || !B.act.svh_g || B.xq_act.on) die("--bind: bindp::make_layer did not wrap the trellis GEMMs");
    bindp::launch(B.q_b, st);
    bindp::launch(B.xq_attn, st);
    bindp::launch(B.o_proj, st);
    bindp::launch(B.xq_x2, st);
    bindp::launch_pair(B.gate, B.up, st);
    bindp::launch(B.act, st);
    bindp::launch(B.xq_act, st);
    bindp::launch(B.down, st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    dump_dev(dir + "/bind.pf.q_b.bin", pio.q_b, static_cast<size_t>(R) * Q.N * 2);
    dump_dev(dir + "/bind.pf.o_proj.bin", pio.attn_part, static_cast<size_t>(R) * HID * 2);
    // o_proj as the engine binds it by default: no FP8 copy, the GEMM on the tiles (make_gemm_tr). Bit-equal to the copy.
    bindp::Tensors P2 = P;
    P2.t[bindp::T_O_PROJ] = bind::Tensor();
    P2.s[bindp::T_O_PROJ] = bind::Tensor();
    P2.tm.o_proj = O.m;
    std::vector<uint8_t> o1(static_cast<size_t>(R) * HID * 2);
    CK(cudaMemcpy(o1.data(), pio.attn_part, o1.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemset(pio.attn_part, 0, o1.size()));
    bindp::Layer B2;
    ok(bindp::make_layer(P2, R, psc, pio, blas, B2));
    if (!B2.o_proj.tw || !B2.o_proj.osvh) die("--bind: bindp::make_layer did not bind o_proj on its tiles");
    bindp::launch(B2.xq_attn, st);
    bindp::launch(B2.o_proj, st);
    CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> o2(o1.size());
    CK(cudaMemcpy(o2.data(), pio.attn_part, o2.size(), cudaMemcpyDeviceToHost));
    if (o1 != o2) die("--bind: prefill o_proj on the tiles differs from the FP8 copy's");
    dump_dev(dir + "/bind.pf.mlp.bin", pio.mlp_part, static_cast<size_t>(R) * HID * 2);
    printf("bind: decode at %zu row counts%s and prefill (%d rows, o_proj on its tiles bit-equal to the copy) through make_layer dumped\n", rows_list().size(),
#if DEC_SLOT >= 2
           ", send slots and doorbells checked",
#else
           "",
#endif
           R);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) die("usage: dtr_test DIR [--bind | --bench [--iters N] [--sweep 1]]");
    const std::string dir = argv[1];
    bool b = false, bnd = false;
    int iters = 640, sweep = 0, pfonly = 0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--bench") b = true;
        else if (a == "--bind") bnd = true;
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--sweep" && i + 1 < argc) sweep = atoi(argv[++i]);
        else if (a == "--pf-only") pfonly = 1;
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
    if (bnd) return bind_check(dir, mats);
    return b ? bench(mats, glus, iters, sweep, pfonly) : check(dir, mats, glus);
}
