// exlr8 v2 prefill MoE (bindp::launch_experts: route, fc1, mid, fc2, final sum) on one rank's expert pack. The test dumps output for
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
//
// Without a pack (a -DG6_RS_LOADER build): the rank's image planned from the exlr8 release, as m4sim --v2-dense plans it,
// reading only the layer's tensors:
//   pf_v2_test --release DENSE_DIR --v2 EXPERTS_DIR --v2-slices SLICES_DIR --v2-widths MANIFEST [--rank 0] --layer 3 ...
// Routing from a production route-count dump instead of random logits: --rcdump FILE [--rcline N] takes line N (default
// 0) of a GR8_PF_RCDUMP routes.r<rank>.jsonl, scales its per-expert counts to M x 8 routes and hands the routes to the
// pass pre-routed (tok_e / tok_w), so the segment and tile fill are production's.
// --ab [--bench ITERS] [--repeat N]: the segment kernels (GR8_PF_WT=0) against the expert-tile kernels (GR8_PF_WT=1) on
// the same inputs: per-kernel median times, the relative RMS of z1 / yw / out between them (and of yw with the segment fc1
// feeding the tile fc2, which isolates fc2), and N repeat-run hashes of each. --wt 0|1 picks the path of the run (and of
// --dump) regardless of the knob.
// --writepack BASE (with --release): writes the layer's tr3.* tensors of the planned image as BASE.manifest.json + BASE.pack
// in pack order, which engine/ref/k1_v2_ref.py reads, so a --dump of this rank can be checked against the NumPy reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include "../bind_prefill.cuh"
#include "../../knobs-rs/gr8_knobs.h"
#include "../../h2d.h"
#include "../../loader/devimage.h"
#ifdef G6_RS_LOADER
#include "../../loader-rs/g6load.h"
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static void die(const std::string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }

template <class T> static T* dev(const std::vector<T>& v)
{
    T* p;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(g6h2d::copy(p, v.data(), v.size() * sizeof(T)));     // pageable source: pinned staging (64 KiB-page kernels)
    return p;
}

// FNV-1a of a device buffer, for the repeat-run hashes
static uint64_t dev_hash(const void* d, size_t n)
{
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
    uint64_t x = 0xcbf29ce484222325ull;
    for (uint8_t b : h) x = (x ^ b) * 0x100000001b3ull;
    return x;
}

// the "cnt" array of line `line` of a GR8_PF_RCDUMP jsonl file
static std::vector<long long> rcdump_counts(const std::string& file, int line)
{
    FILE* f = fopen(file.c_str(), "r");
    if (!f) die("cannot read " + file);
    std::string s;
    char buf[1 << 16];
    for (int l = 0; l <= line; ++l)
    {
        s.clear();
        bool got = false;
        while (fgets(buf, sizeof buf, f)) { s += buf; got = true; if (s.back() == '\n') break; }
        if (!got) die(file + ": fewer than " + std::to_string(line + 1) + " lines");
    }
    fclose(f);
    const size_t p = s.find("\"cnt\": [");
    if (p == std::string::npos) die(file + ": no \"cnt\" array on line " + std::to_string(line));
    std::vector<long long> c;
    const char* q = s.c_str() + p + 8;
    while (*q && *q != ']') { c.push_back(strtoll(q, const_cast<char**>(&q), 10)); while (*q == ',' || *q == ' ') ++q; }
    if (c.size() != static_cast<size_t>(bindp::NEXP)) die(file + ": cnt has " + std::to_string(c.size()) + " entries");
    return c;
}

int main(int argc, char** argv)
{
    std::string pack_base, dump, release, v2dir, v2slices, v2widths, rcdump, writepack;
    int layer = 40, M = 20, iters = 0, wide = 1, int4sh = 0, shab = 0, shpass = 0, nr = g6nr::NR, rank = 0, rcline = 0, ab = 0, repeat = 0;
    int wt = -1, fuse = -1, gmajor = -1;   // -1: the knobs (GR8_PF_WT, GR8_PF_WT_FUSE, GR8_PF_WT_GMAJOR); --wt / --fuse / --gmajor 0|1 force
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) die("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--pack") pack_base = nx();
        else if (a == "--release") release = nx();
        else if (a == "--v2") v2dir = nx();
        else if (a == "--v2-slices") v2slices = nx();
        else if (a == "--v2-widths") v2widths = nx();
        else if (a == "--rank") rank = std::stoi(nx());
        else if (a == "--rcdump") rcdump = nx();
        else if (a == "--rcline") rcline = std::stoi(nx());
        else if (a == "--ab") ab = 1;
        else if (a == "--repeat") repeat = std::stoi(nx());
        else if (a == "--wt") wt = std::stoi(nx());
        else if (a == "--fuse") fuse = std::stoi(nx());
        else if (a == "--gmajor") gmajor = std::stoi(nx());
        else if (a == "--writepack") writepack = nx();
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
    if ((pack_base.empty() && release.empty()) || M < 1) die("usage: see the file header");

    g6::Pack pack;
    g6::DevImage img;
    std::string err;
    void* plan = nullptr;
    std::map<std::string, void*> planned;        // --release: device copies of the planned image's tensors, by name
    std::vector<std::pair<const g6::Entry*, std::vector<uint8_t>>> kept;   // --writepack: the layer's tr3.* bytes in pack order
    char lpfx[16];
    snprintf(lpfx, sizeof lpfx, "L%03d.tr3.", layer);
    if (!release.empty())
    {
#ifdef G6_RS_LOADER
        if (v2dir.empty() || v2slices.empty() || v2widths.empty()) die("--release needs --v2, --v2-slices and --v2-widths");
        char tmp[] = "/tmp/pf_v2_test.XXXXXX.manifest.json";
        const int fd = mkstemps(tmp, 14);
        if (fd < 0) die("mkstemps");
        close(fd);
        char perr[1024] = "";
        plan = g6l_release_plan(release.c_str(), v2dir.c_str(), v2slices.c_str(), v2widths.c_str(), nr, rank, 0, 1, "", tmp, perr, sizeof perr);
        if (!plan) { unlink(tmp); die(std::string("release plan: ") + perr); }
        const bool ok = pack.open_manifest(tmp, err);
        unlink(tmp);
        if (!ok) die(err);
#else
        die("--release needs the Rust loader (-DG6_RS_LOADER, loader-rs/target/release/libg6load.a)");
#endif
    }
    else
    {
        if (!pack.open_manifest(pack_base + ".manifest.json", err)) die(err);
        g6::LoadOpts lo;
        g6::LoadStats ls;
        if (!img.load(pack, pack_base + ".pack", lo, ls, err)) die(err);
    }
    auto find = [&](const char* name) {
        bind::Tensor t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (plan)
        {
#ifdef G6_RS_LOADER
            void*& d = planned[name];
            if (!d)
            {
                std::vector<uint8_t> h(e->nbytes);
                char perr[1024] = "";
                if (e->nbytes && g6l_plan_read(plan, e->offset, e->nbytes, h.data(), perr, sizeof perr)) die(std::string(name) + ": " + perr);
                CK(cudaMalloc(&d, e->nbytes ? e->nbytes : 1));
                CK(g6h2d::copy(d, h.data(), e->nbytes));
                if (!writepack.empty() && strncmp(name, lpfx, strlen(lpfx)) == 0) kept.emplace_back(e, std::move(h));
            }
            t.p = d;
#endif
        }
        else t.p = img.ptr(*e);
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
    if (!writepack.empty())
    {
        if (!plan) die("--writepack needs --release");
        static const char* const DT[] = {"u8", "i16", "f16", "bf16", "f32", "i32", "i64"};
        static const char* const XF[] = {"tr3", "exlr8-e4m3", "exlr8-e4m3-mul1", "exlr8-e4m3-pmcgx", "exlr8-e4m3-unchecked"};
        FILE* fp = fopen((writepack + ".pack").c_str(), "wb");
        FILE* fm = fopen((writepack + ".manifest.json").c_str(), "w");
        if (!fp || !fm) die("cannot write " + writepack + ".pack / .manifest.json");
        fprintf(fm, "{\"rank\": %d, \"tensors\": [", rank);
        uint64_t off = 0;
        for (size_t i = 0; i < kept.size(); ++i)
        {
            const g6::Entry& e = *kept[i].first;
            fwrite(kept[i].second.data(), 1, e.nbytes, fp);
            fprintf(fm, "%s\n {\"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [", i ? "," : "", e.name.c_str(), DT[static_cast<int>(e.dtype)]);
            for (size_t j = 0; j < e.shape.size(); ++j) fprintf(fm, "%s%lld", j ? ", " : "", static_cast<long long>(e.shape[j]));
            fprintf(fm, "], \"offset\": %llu, \"nbytes\": %llu, \"expert_format\": \"%s\"}", static_cast<unsigned long long>(off),
                    static_cast<unsigned long long>(e.nbytes), XF[e.expert_format < 5 ? e.expert_format : 4]);
            off += e.nbytes;
        }
        fprintf(fm, "\n]}\n");
        fclose(fp); fclose(fm);
        printf("writepack %s: %zu tensors, %.1f MB\n", writepack.c_str(), kept.size(), off / 1e6);
        kept.clear();
    }

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
    // --rcdump: the dump's per-expert counts scaled to M x 8 routes, dealt to the tokens (8 distinct experts each) in a
    // shuffled order, random weights summing to the route scale; the pass takes them pre-routed
    std::vector<int> pre_e;
    std::vector<float> pre_w;
    if (!rcdump.empty())
    {
        const std::vector<long long> c = rcdump_counts(rcdump, rcline);
        long long tot = 0;
        for (long long v : c) tot += v;
        if (tot < 1) die("--rcdump: no routes on that line");
        std::vector<long long> n(bindp::NEXP);
        long long got = 0;
        for (int e = 0; e < bindp::NEXP; ++e) { n[e] = (c[e] * static_cast<long long>(M) * bindp::TOPK + tot / 2) / tot; got += n[e]; }
        for (long long d = static_cast<long long>(M) * bindp::TOPK - got; d != 0;)     // the rounding remainder onto the largest experts
            for (int e = 0; e < bindp::NEXP && d != 0; ++e)
                if (n[e] > 0 || d > 0) { n[e] += d > 0 ? 1 : -1; d += d > 0 ? -1 : 1; }
        std::vector<int> L;
        for (int e = 0; e < bindp::NEXP; ++e) for (long long j = 0; j < n[e]; ++j) L.push_back(e);
        std::shuffle(L.begin(), L.end(), rng);
        pre_e.assign(static_cast<size_t>(M) * bindp::TOPK, 0);
        pre_w.assign(pre_e.size(), 0.f);
        for (int m = 0; m < M; ++m)
        {
            int* te = &pre_e[static_cast<size_t>(m) * bindp::TOPK];
            for (int k = 0; k < bindp::TOPK; ++k)
            {
                size_t i = static_cast<size_t>(m) * bindp::TOPK + k;
                for (size_t j = i; j < L.size(); ++j)            // the first later slot whose expert the token does not have yet
                {
                    bool dup = false;
                    for (int kk = 0; kk < k; ++kk) dup |= te[kk] == L[j];
                    if (!dup) { std::swap(L[i], L[j]); break; }
                }
                te[k] = L[i];
            }
            float s = 0.f, w[bindp::TOPK];
            for (int k = 0; k < bindp::TOPK; ++k) { w[k] = 0.2f + std::uniform_real_distribution<float>(0.f, 1.f)(rng); s += w[k]; }
            for (int k = 0; k < bindp::TOPK; ++k) pre_w[static_cast<size_t>(m) * bindp::TOPK + k] = w[k] / s * 2.5f;
        }
        bool bad = false;
        for (int m = 0; m < M && !bad; ++m)
            for (int k = 0; k < bindp::TOPK; ++k)
                for (int kk = 0; kk < k; ++kk) bad |= pre_e[m * bindp::TOPK + k] == pre_e[m * bindp::TOPK + kk];
        if (bad) die("--rcdump: could not deal 8 distinct experts to every token (counts too concentrated)");
    }

    const bind::Tensor &tab = T.t[bindp::T_TR3_TAB], &v1 = T.t[bindp::T_TR3_V1], &v2 = T.t[bindp::T_TR3_V2], &v3 = T.t[bindp::T_TR3_V3];
    bindp::Expert A;
    A.on = 1; A.M = M; A.maxseg = bindp::max_segments(M);
    A.logits = logits; A.bias = bd;
    A.tok_e = sc.tok_e; A.tok_w = sc.tok_w; A.cnt = sc.cnt;
    if (!pre_e.empty())
    {
        CK(g6h2d::copy(sc.tok_e, pre_e.data(), pre_e.size() * 4));
        CK(g6h2d::copy(sc.tok_w, pre_w.data(), pre_w.size() * 4));
        A.pre_routed = 1;
    }
    A.rr_row = sc.rr_row; A.rr_e = sc.rr_e; A.rr_sp = sc.rr_sp; A.rr_w = sc.rr_w; A.pos = sc.pos; A.segs = sc.segs; A.n_seg = sc.n_seg;
    A.t_tab = static_cast<const long long*>(tab.p);
#ifndef K1V2_MAIN
    A.t_kmix = static_cast<const long long*>(T.t[bindp::T_TR3_KMIX].p);
    A.k2 = D.k2;
    A.cb = T.cb;                                    // the codebook (tr3.table's expert format); make_layer's checks:
    if (A.cb != D.cb || !tr3_cb_built(A.cb) || (A.cb != TR3_MCG && !T.t[bindp::T_TR3_V1B].p))
        die("the layer's codebook does not run here (mul1 and pmcgx need the trellis shared expert)");
    A.bs = T.t[bindp::T_TR3_V1B].p != nullptr;      // block-scale hidden-side vectors (v2 layers with the trellis shared expert)
    A.t_sg = static_cast<const __half*>(T.t[bindp::T_TR3_SG].p);
    A.t_v1b = static_cast<const __half*>(T.t[bindp::T_TR3_V1B].p);
    A.t_v3b = static_cast<const __half*>(T.t[bindp::T_TR3_V3B].p);
#endif
    A.t_fc1 = static_cast<const uint8_t*>(T.t[bindp::T_TR3_FC1].p); A.t_wide = D.fc1_wide;
    A.t_fc2 = static_cast<const uint8_t*>(T.t[bindp::T_TR3_FC2].p); A.t_wide2 = D.fc2_wide;
    A.t_v1 = static_cast<const __half*>(v1.p); A.t_v2 = static_cast<const __half*>(v2.p); A.t_v3 = static_cast<const __half*>(v3.p);
    A.x = x; A.ldx = HID;
    A.z1 = sc.z1; A.x2 = sc.x2; A.yw = sc.yw; A.ysh = sc.ysh; A.out = out;
#ifdef PF_AOP
    A.aop = knob::pf_aop().value_or(true) ? sc.aop : nullptr;
#endif
#ifdef PF_F12
    A.f12 = knob::pf_f12().value_or(true) ? 1 : 0;
#endif
    // GR8_PF_WT (--wt): the expert-tile kernels, under make_layer's conditions
    const bool wt_ok = A.bs && ((!D.k2 && !D.k3) || (D.fc1_wide && D.fc2_wide));
    const bool wt_on = wt_ok && (wt < 0 ? knob::pf_wt() : wt != 0);
    if ((wt > 0 || ab) && !wt_ok) die("the expert-tile kernels need a block-scale layer with wide-order K2 / K3 units (--wide 1)");
    const bool fuse_on = fuse < 0 ? knob::pf_wt_fuse() : fuse != 0, gmajor_on = gmajor < 0 ? knob::pf_wt_gmajor() : gmajor != 0;
    auto set_wt = [&](bindp::Expert& E, bool on, bool fused = false) {
        E.wtk = on ? 1 : 0;
        E.wtiles = on ? sc.wtiles : nullptr; E.n_wtiles = on ? sc.n_wtiles : nullptr; E.rr_wp = on ? sc.rr_wp : nullptr;
        E.hq8 = on ? sc.hq8 : nullptr; E.hs8 = on ? sc.hs8 : nullptr;
        E.wtc = on && fused ? 1 : 0; E.cnt_tg = on ? sc.cnt_tg : nullptr; E.wt_gmajor = gmajor_on ? 1 : 0;
    };
    set_wt(A, wt_on, fuse_on);
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
        set_wt(S, false);                       // the trellis shared pass keeps the segment kernels (as make_layer)
#ifdef PF_AOP
        S.aop = nullptr;
#endif
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
        if (A.bs)
        {
            X.suh_g = X.suh_u = A.t_sg; X.sc_g = A.t_v1b + row * 2 * (HID / 128); X.sc_u = X.sc_g + HID / 128;
            A.sh_svh = A.t_sg + HID; A.sh_sc = A.t_v3b + row * (HID / 128);
        }
        else
        {
            X.suh_g = A.t_v1 + row * 2 * HID; X.suh_u = X.suh_g + HID;
            A.sh_svh = A.t_v3 + row * HID;
        }
        X.v2 = A.t_v2 + row * 3 * 128;
        X.g = sc.g; X.u = sc.u;
        Lf.up.x8 = sc.x8_up;
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

    if (ab)
    {
        // the two paths on the same routes: per-kernel medians (ITERS = --bench, at least 3), out's relative RMS, repeat hashes
        const int it = std::max(iters, 3);
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        auto med = [&](auto f) {
            for (int i = 0; i < 2; ++i) f();
            std::vector<float> ms(it);
            for (int r = 0; r < it; ++r)
            {
                CK(cudaEventRecord(e0, st));
                f();
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[r], e0, e1));
            }
            CK(cudaGetLastError());
            std::sort(ms.begin(), ms.end());
            return 1e3 * ms[it / 2];
        };
        bindp::Expert O = A, N = A, F = A;
        set_wt(O, false);
        set_wt(N, true);                                     // the tile kernels, partials through yw and k_pf_fin
        set_wt(F, true, true);                               // the tile kernels with the fused combine
        // the route list of the pass (the same for both paths): once, outside the timing
        bindp::launch_experts(O, st);
        CK(cudaStreamSynchronize(st));
        const size_t ob = static_cast<size_t>(M) * HID * 2, nz1 = static_cast<size_t>(M) * 8 * 2 * g6nr::EB * 128, nyw = static_cast<size_t>(M) * 8 * HID;
        auto fetch16 = [&](const void* d, size_t n) { std::vector<uint16_t> h(n); CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost)); return h; };
        auto rms16 = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {       // fp16: |a - b| / |b|
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                const double fa = __half2float(__ushort_as_half(a[i])), fb = __half2float(__ushort_as_half(b[i]));
                if (!std::isfinite(fa) || !std::isfinite(fb)) continue;
                num += (fa - fb) * (fa - fb); den += fb * fb;
            }
            return std::sqrt(num / std::max(den, 1e-30));
        };
        std::vector<uint16_t> o_old(static_cast<size_t>(M) * HID), o_new(o_old.size());
        CK(cudaMemcpy(o_old.data(), out, ob, cudaMemcpyDeviceToHost));
        const std::vector<uint16_t> z1_old = fetch16(sc.z1, nz1), yw_old = fetch16(sc.yw, nyw);
        bindp::launch_experts(N, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(o_new.data(), out, ob, cudaMemcpyDeviceToHost));
        const std::vector<uint16_t> z1_new = fetch16(sc.z1, nz1), yw_new = fetch16(sc.yw, nyw);
        bindp::launch_route(N, st);                          // the segment fc1 feeding the tile mid + fc2: fc2's own deviation
        bindp::launch_fc1(O, st);
        bindp::launch_mid(N, st);
        bindp::launch_fc2(N, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<uint16_t> yw_mix = fetch16(sc.yw, nyw);
        printf("ab stages, relative RMS new vs old: z1 (fc1) %.3e, yw (fc1 + fc2) %.3e, yw with the old fc1 (fc2 alone) %.3e\n",
               rms16(z1_new, z1_old), rms16(yw_new, yw_old), rms16(yw_mix, yw_old));
        std::vector<uint16_t> o_fused(o_old.size());
        CK(cudaMemset(out, 0, ob));
        bindp::launch_experts(F, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(o_fused.data(), out, ob, cudaMemcpyDeviceToHost));
        size_t nfd = 0;
        for (size_t i = 0; i < o_new.size(); ++i) nfd += o_fused[i] != o_new[i];
        printf("ab fused combine: out differs from the unfused tile path in %zu of %zu elements (0 = k_pf_fin's bytes)\n", nfd, o_new.size());
        double num = 0, den = 0, mx = 0, mref = 0;
        size_t ndiff = 0;
        for (size_t i = 0; i < o_old.size(); ++i)
        {
            uint32_t ua = static_cast<uint32_t>(o_old[i]) << 16, ub = static_cast<uint32_t>(o_new[i]) << 16;
            float fa, fb;
            memcpy(&fa, &ua, 4); memcpy(&fb, &ub, 4);
            const double d = static_cast<double>(fb) - fa;
            num += d * d; den += static_cast<double>(fa) * fa;
            mx = std::max(mx, std::fabs(d)); mref = std::max(mref, std::fabs(static_cast<double>(fa)));
            ndiff += o_old[i] != o_new[i];
        }
        int nseg = 0, nwt = 0, cnt[bindp::NEXP];
        CK(cudaMemcpy(&nseg, sc.n_seg, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&nwt, sc.n_wtiles, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cnt, sc.cnt, sizeof cnt, cudaMemcpyDeviceToHost));
        long long routes = 0, rb16 = 0;
        for (int e = 0; e < bindp::NEXP; ++e) { routes += cnt[e]; rb16 += (cnt[e] + 15) / 16; }
        printf("ab M=%d layer=%d routes=%lld: segments %d (fill %.3f), tiles %d (fill %.3f), 16-row blocks %lld (fill %.3f)\n", M, layer, routes,
               nseg, routes / (static_cast<double>(nseg) * bindp::RS), nwt, routes / (static_cast<double>(nwt) * bindp::WT_R), rb16,
               routes / (16.0 * rb16));
        printf("ab out: new vs old relative RMS %.3e, max |diff| %.3e (max |old| %.3e), %zu of %zu elements differ\n",
               std::sqrt(num / std::max(den, 1e-30)), mx, mref, ndiff, o_old.size());
        const double t_route = med([&]() { bindp::launch_route(O, st); });
        const double t_fc1o = med([&]() { bindp::launch_fc1(O, st); }), t_mido = med([&]() { bindp::launch_mid(O, st); });
        const double t_fc2o = med([&]() { bindp::launch_fc2(O, st); }), t_fino = med([&]() { bindp::launch_fin(O, st); });
        const double t_hq8 = med([&]() { bindp::k_pf_hq8<<<((N.M + 7) / 8 * (HID / 128) + 7) / 8, 256, 0, st>>>(N); });
        bindp::Expert N1 = N;
        N1.hq8_ready = 1;
        bindp::launch_route(N, st);                         // the tile list (k_pf_fill writes it when wtiles is set)
        const double t_fc1n = med([&]() { bindp::launch_fc1(N1, st); }), t_midn = med([&]() { bindp::launch_mid(N, st); });
        const double t_fc2n = med([&]() { bindp::launch_fc2(N, st); }), t_finn = med([&]() { bindp::launch_fin(N, st); });
        const double t_fc2f = med([&]() { bindp::launch_fc2(F, st); });
        const double t_old = med([&]() { bindp::launch_experts(O, st); }), t_new = med([&]() { bindp::launch_experts(N, st); });
        const double t_fused = med([&]() { bindp::launch_experts(F, st); });
        printf("ab us       route   hq8     fc1      mid     fc2      fin      pass\n");
        printf("old      %8.1f %7s %8.1f %8.1f %8.1f %8.1f %8.1f\n", t_route, "-", t_fc1o, t_mido, t_fc2o, t_fino, t_old);
        printf("tiles    %8.1f %7.1f %8.1f %8.1f %8.1f %8.1f %8.1f   (%.2fx)\n", t_route, t_hq8, t_fc1n, t_midn, t_fc2n, t_finn, t_new, t_old / t_new);
        printf("fused    %8.1f %7.1f %8.1f %8.1f %8.1f %8s %8.1f   (%.2fx)\n", t_route, t_hq8, t_fc1n, t_midn, t_fc2f, "-", t_fused, t_old / t_fused);
        for (int r = 0; r < repeat; ++r)
        {
            CK(cudaMemset(out, 0, ob));
            bindp::launch_experts(O, st);
            CK(cudaStreamSynchronize(st));
            const uint64_t ho = dev_hash(out, ob);
            CK(cudaMemset(out, 0, ob));
            bindp::launch_experts(N, st);
            CK(cudaStreamSynchronize(st));
            const uint64_t hn = dev_hash(out, ob);
            CK(cudaMemset(out, 0, ob));
            bindp::launch_experts(F, st);
            CK(cudaStreamSynchronize(st));
            printf("repeat %d: old %016llx tiles %016llx fused %016llx\n", r, static_cast<unsigned long long>(ho), static_cast<unsigned long long>(hn),
                   static_cast<unsigned long long>(dev_hash(out, ob)));
        }
        CK(cudaGetLastError());
    }

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
