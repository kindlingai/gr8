// exlr8 v2 decode MoE (K1) on one rank's expert pack: output dump for the NumPy reference (engine/ref/k1_v2_ref.py) and
// step timing against the GB10 read floor. The pack needs a layer's tr3.* tensors only (tools/convert/v2.py writes
// expert-only packs, and a full rank pack works too). The shared expert is a zero int4 record of the TP size's width
// (its bytes are read, its output is zero), so the dumped output is the routed experts' rank partial.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I.. -o k1_v2_test k1_v2_test.cu \
//        ../../loader/g6pack.cpp ../../loader/devimage.cpp -lcublas -lpthread           (add -DG6_NR=4 for TP4)
//   k1_v2_test --pack DIR/rank0 --layer 40 [--nr 4] [--M 5] [--dump FILE] [--bench ITERS] [--plans 32] [--seed 1]
//              [--fuse 1] [--wide 1] [--hot N] [--nokmix] [--int4-shared] [--flush] [--warm] [--split]
// --pack DIR/rank0 reads DIR/rank0.manifest.json + DIR/rank0.pack. --nr = the pack's TP size (sets the
// shared expert's width: 64 g32 groups split as g6nr::width). Routing: each token picks 8 distinct experts uniformly
// (--hot N: from experts 0..N-1 only), weights random and summing to 2.5, one plan per step from --plans rotating
// plans (so the weights do not stay in the 24 MB L2 across steps). --wide 1 reorders K3 fc1/fc2 into the wide order as
// the runtime does (bind::widen_fc1/fc2).
// A pack with tr3.table row 256 runs its shared expert as trellis (bind::scan_widths) unless --int4-shared.
// Dump (binary, little endian): int32 M, int32 flags (1 | 2 if the trellis shared expert is in out), then x fp16 [M][6144], tok_e int32 [M][8], tok_w f32 [M][8],
// out fp16 [M][6144].
// Bench line: per-step time, routed trellis bytes per step (mean over the plans), all bytes (trellis + side vectors +
// shared records), and the read rate of each against the 220 GB/s floor.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../bind.cuh"
#include "../../h2d.h"
#include "../../loader/devimage.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static void die(const std::string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }

struct Plans                          // [NP] plans in the K1Args array layout (M <= 8: column c of a slot = token c)
{
    std::vector<int> n_slots, slot_expert, slot_mask, slot_tok, tok_slot, tok_col, tok_e;
    std::vector<float> slot_w, tok_w;
};

static Plans make_plans(int NP, int M, int hot, uint64_t seed)
{
    Plans P;
    P.n_slots.assign(NP, 0);
    P.slot_expert.assign(NP * SMAX, 0);
    P.slot_mask.assign(NP * SMAX, 0);
    P.slot_w.assign(NP * SMAX * 8, 0.f);
    P.slot_tok.assign(NP * SMAX * 8, -1);
    P.tok_slot.assign(NP * G6_MX * 8, -1);
    P.tok_col.assign(NP * G6_MX * 8, 0);
    P.tok_w.assign(NP * G6_MX * 8, 0.f);
    P.tok_e.assign(NP * G6_MX * 8, -1);
    std::mt19937_64 rng(seed);
    for (int p = 0; p < NP; ++p)
    {
        std::vector<int> te(M * 8);
        std::vector<float> tw(M * 8);
        for (int m = 0; m < M; ++m)
        {
            std::vector<int> pool(hot);
            for (int i = 0; i < hot; ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            float sum = 0;
            for (int k = 0; k < 8; ++k) { te[m * 8 + k] = pool[k]; tw[m * 8 + k] = 0.2f + (rng() % 1000) / 1000.f; sum += tw[m * 8 + k]; }
            for (int k = 0; k < 8; ++k) tw[m * 8 + k] *= 2.5f / sum;
        }
        std::vector<int> ex(te);
        std::sort(ex.begin(), ex.end());
        ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
        P.n_slots[p] = static_cast<int>(ex.size());
        for (size_t s = 0; s < ex.size(); ++s) P.slot_expert[p * SMAX + s] = ex[s];
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < 8; ++k)
            {
                const int s = static_cast<int>(std::lower_bound(ex.begin(), ex.end(), te[m * 8 + k]) - ex.begin());
                P.slot_mask[p * SMAX + s] |= 1 << m;
                P.slot_w[(p * SMAX + s) * 8 + m] = tw[m * 8 + k];
                P.slot_tok[(p * SMAX + s) * 8 + m] = m;
                P.tok_slot[(p * G6_MX + m) * 8 + k] = s;
                P.tok_col[(p * G6_MX + m) * 8 + k] = m;
                P.tok_w[(p * G6_MX + m) * 8 + k] = tw[m * 8 + k];
                P.tok_e[(p * G6_MX + m) * 8 + k] = te[m * 8 + k];
            }
    }
    return P;
}

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
    int layer = 40, nr = g6nr::NR, M = 5, iters = 0, NP = 32, fuse = 1, wide = 1, hot = 256, nokmix = 0, int4sh = 0, flush = 0, warmsh = 0, split = 0;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--pack") pack_base = nx();
        else if (a == "--layer") layer = std::stoi(nx());
        else if (a == "--nr") nr = std::stoi(nx());
        else if (a == "--M") M = std::stoi(nx());
        else if (a == "--dump") dump = nx();
        else if (a == "--bench") iters = std::stoi(nx());
        else if (a == "--plans") NP = std::stoi(nx());
        else if (a == "--seed") seed = std::stoull(nx());
        else if (a == "--fuse") fuse = std::stoi(nx());
        else if (a == "--wide") wide = std::stoi(nx());
        else if (a == "--hot") hot = std::stoi(nx());
        else if (a == "--nokmix") nokmix = 1;
        else if (a == "--int4-shared") int4sh = 1;
        else if (a == "--flush") flush = 1;
        else if (a == "--warm") warmsh = 1;
        else if (a == "--split") split = 1;
        else die("unknown arg " + a);
    }
    if (pack_base.empty() || M < 1 || M > 8 || hot < 8 || hot > 256 || NP < 1) die("usage: see the file header");
    if (!dump.empty()) NP = 1;

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
    bind::LayerTensors T = bind::layer_tensors(layer, find);
#ifndef K1V2_MAIN
    if (nokmix) T.t[bind::T_TR3_KMIX] = bind::Tensor();     // --nokmix: bind as a pack without tr3.kmix
#endif
    const bind::Tensor& tab = T.t[bind::T_TR3_TAB];
    if (!tab.p) die("pack has no tr3.table for the layer");
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    CK(bind::init());
    if (wide) { bind::widen_fc1(T, st); bind::widen_fc2(T, st); CK(cudaStreamSynchronize(st)); }
    if (const char* e = bind::check_shared_signs(T, st)) die(e);
#ifndef K1V2_MAIN
    if (const char* e = bind::scan_widths(T, true)) die(e);
    if (int4sh) T.sh_trellis = 0;          // --int4-shared: run the s_* records (zeros here)
#endif

    // zero shared expert records at the TP size's widest rank width (64 g32 groups over nr ranks)
    std::vector<int64_t> tabh(tab.shape[0] * 6), kmh(tab.shape[0], 0);
    CK(cudaMemcpy(tabh.data(), tab.p, tabh.size() * 8, cudaMemcpyDeviceToHost));
#ifndef K1V2_MAIN   // -DK1V2_MAIN: build against a tree before #65 (no tr3.kmix), for the one-width A/B
    if (T.t[bind::T_TR3_KMIX].p) CK(cudaMemcpy(kmh.data(), T.t[bind::T_TR3_KMIX].p, kmh.size() * 8, cudaMemcpyDeviceToHost));
#endif
    const int sch = (64 / nr + (64 % nr ? 1 : 0)) * 32;
    if (sch > SACTW) die("shared width above SACTW for this build");
    std::vector<uint8_t> z1(static_cast<size_t>(sch / 16) * 192 * 576, 0), z2(static_cast<size_t>(384) * (sch / 32) * 288, 0);
    uint8_t* s1 = dev(z1);
    uint8_t* s2 = dev(z2);
    T.t[bind::T_S_FC1].p = s1; T.t[bind::T_S_FC1].nd = 3;
    T.t[bind::T_S_FC1].shape[0] = sch / 16; T.t[bind::T_S_FC1].shape[1] = 192; T.t[bind::T_S_FC1].shape[2] = 576;
    T.t[bind::T_S_FC2].p = s2; T.t[bind::T_S_FC2].nd = 3;
    T.t[bind::T_S_FC2].shape[0] = 384; T.t[bind::T_S_FC2].shape[1] = sch / 32; T.t[bind::T_S_FC2].shape[2] = 288;

    // inputs
    std::mt19937_64 rng(seed * 7919 + 3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> xh(8 * HID, __float2half(0.f));
    for (int i = 0; i < M * HID; ++i) xh[i] = __float2half(nd(rng));
    __half* x = dev(xh);
    void* blk;
    CK(cudaMalloc(&blk, bind::scratch_bytes()));
    CK(cudaMemset(blk, 0, bind::scratch_bytes()));
    const bind::Scratch sc = bind::scratch_at(blk);
    __half* out;
    CK(cudaMalloc(&out, 8 * HID * 2));
    CK(cudaMemset(out, 0, 8 * HID * 2));
    Plans P = make_plans(NP, M, hot, seed);
    bind::PlanPtrs pp;
    pp.n_slots = dev(P.n_slots); pp.slot_expert = dev(P.slot_expert); pp.slot_mask = dev(P.slot_mask);
    pp.slot_w = dev(P.slot_w); pp.slot_tok = dev(P.slot_tok); pp.tok_slot = dev(P.tok_slot); pp.tok_col = dev(P.tok_col); pp.tok_w = dev(P.tok_w);
    bind::Xfrag xf;
    xf.on = 1; xf.src = x; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2;
    bind::launch(xf, st);
    bind::K1 k;
    bind::Tensor shrec[3];
    if (const char* e = bind::make_k1(T, M, sc, x, sc.xf_x2, pp, out, 0, k, shrec, fuse)) die(e);
    k.a.n_plans = NP;

    // bytes per step on this rank
    double tre = 0, side = 0;
    for (int p = 0; p < NP; ++p)
        for (int s = 0; s < P.n_slots[p]; ++s)
        {
            const int e = P.slot_expert[p * SMAX + s];
            const int64_t* tr = &tabh[e * 6];
            const int nb = static_cast<int>(tr[3]);
            for (int b = 0; b < nb; ++b)
                for (int mat = 0; mat < 2; ++mat)
                    tre += 384.0 * 8 * 32 * (kmh[e] ? (kmh[e] >> (4 * (2 * b + mat))) & 15 : tr[4]);
            tre += nb * 384.0 * 8 * 32 * tr[4];
            side += nb * (2.0 * HID * 2 + 3 * 128 * 2 + HID * 2);
        }
    tre /= NP; side /= NP;
    int sh = 0;
#ifndef K1V2_MAIN
    sh = k.sh;
#endif
    double shared = static_cast<double>(z1.size() + z2.size());
    if (sh)
    {
        const int64_t* tr = &tabh[256 * 6];
        shared = tr[3] * 384.0 * 8 * 32 * (2 * (kmh[256] ? 0 : tr[4]) + tr[4]);
        for (int u = 0; kmh[256] && u < 2 * tr[3]; ++u) shared += 384.0 * 8 * 32 * ((kmh[256] >> (4 * u)) & 15);
    }

    if (!dump.empty())
    {
        bind::launch_k1(k, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        std::vector<__half> oh(M * HID);
        CK(cudaMemcpy(oh.data(), out, oh.size() * 2, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb");
        if (!f) die("cannot write " + dump);
        int hdr[2] = {M, 1 + 2 * sh};      // [1]: 1 plan, bit 1 = the trellis shared expert is in the output
        fwrite(hdr, 4, 2, f);
        fwrite(xh.data(), 2, M * HID, f);
        for (int m = 0; m < M; ++m) fwrite(&P.tok_e[m * 8], 4, 8, f);
        for (int m = 0; m < M; ++m) fwrite(&P.tok_w[m * 8], 4, 8, f);
        fwrite(oh.data(), 2, M * HID, f);
        fclose(f);
        printf("dump %s: M=%d slots=%d rank trellis bytes %.1f MB\n", dump.c_str(), M, P.n_slots[0], tre / 1e6);
    }
    if (iters > 0)
    {
        for (int i = 0; i < 2 * NP; ++i) bind::launch_k1(k, st);      // warm-up (plans rotate through ctl)
        CK(cudaStreamSynchronize(st));
        // --flush: before each step, read 64 MB of another buffer through L2 (launch_warm), so nothing of the last step
        // stays there (in the engine, a layer's own shared expert never does); the flush alone is timed and subtracted
        uint8_t* fb = nullptr;
        if (flush) CK(cudaMalloc(&fb, 64u << 20));
        // --warm: after the flush, warm the shared expert as bind::warm_mlp does in the collective wait (trellis row 256's
        // fc1 bytes, or the s_* fc1 records); timed with the flush and subtracted
        bind::Layer wl;
        wl.sh[0] = shrec[0]; wl.sh[1] = shrec[1];
#ifndef K1V2_MAIN
        if (k.sh)
        {
            size_t o0 = 0, o1 = 0;
            bind::sh_unit_off(T, 0, o0);
            bind::sh_unit_off(T, 2 * static_cast<int>(T.sh_tr[3]), o1);
            wl.sh[0] = wl.sh[1] = bind::Tensor();
            wl.shw = static_cast<const uint8_t*>(T.t[bind::T_TR3_FC1].p) + o0; wl.shw_bytes = o1 - o0;
        }
#endif
        // what: 0 = the K1 launches, 1 = fc1 only, 2 = fc2 only, 3 = finalize only, -1 = none (flush / warm alone)
        auto graph = [&](int what) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < NP; ++i)
            {
                if (flush) launch_warm(fb, 64u << 20, st);
                if (warmsh) bind::warm_mlp(wl, st);
                if (what == 0) bind::launch_k1(k, st);
                else if (what == 1) bind::launch_k1_fc1(k, st);
                else if (what == 2) bind::launch_k1_fc2(k, st);
                else if (what == 3) bind::launch_k1_fin(k, st);
            }
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            return ge;
        };
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        const int reps = std::max(1, iters / NP);
        auto run = [&](cudaGraphExec_t ge) {
            CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st));
            std::vector<float> ms(reps);
            for (int r = 0; r < reps; ++r)
            {
                CK(cudaEventRecord(e0, st));
                CK(cudaGraphLaunch(ge, st));
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[r], e0, e1));
            }
            CK(cudaGetLastError());
            std::sort(ms.begin(), ms.end());
            return std::make_pair(1e3 * ms[reps / 2] / NP, 1e3 * ms[0] / NP);
        };
        auto [us, best] = run(graph(0));
        double base = 0;
        if (flush || warmsh)
        {
            const auto [fu, fbest] = run(graph(-1));
            base = fu;
            us -= fu; best -= fbest;
        }
        if (split)
        {
            if (k.fuse) die("--split needs --fuse 0");
            const double a1 = run(graph(1)).first - base, a2 = run(graph(2)).first - base, a3 = run(graph(3)).first - base;
            printf("split M=%d: fc1 %.1f us, fc2 %.1f us, finalize %.1f us (each alone)\n", M, a1, a2, a3);
        }
        printf("bench M=%d fuse=%d wide=%d sh=%d flush=%d plans=%d: median %.1f us/step (best %.1f), trellis %.2f MB/step -> %.1f GB/s = %.1f%% "
               "of 220; all bytes %.2f MB -> %.1f GB/s = %.1f%%; slots/step %.1f\n",
               M, k.fuse, wide, sh, flush, NP, us, best, tre / 1e6, tre / us / 1e3, 100 * tre / us / 1e3 / 220,
               (tre + side + shared) / 1e6, (tre + side + shared) / us / 1e3, 100 * (tre + side + shared) / us / 1e3 / 220,
               [&] { double s = 0; for (int p = 0; p < NP; ++p) s += P.n_slots[p]; return s / NP; }());
    }
    return 0;
}
