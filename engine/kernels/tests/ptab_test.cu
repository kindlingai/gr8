// ptab_test: EL3 (#151) stage 1, routed-expert pieces read through a per-layer device piece table (ptab.cuh) against
// the pack layout (one contiguous tr3.fc1 / tr3.fc2 per layer). One GB10, no pack files, no fabric. Synthetic exlr8 v2
// layers in this build's rank shape (g6nr::EB blocks per expert: 4 at TP4, 3 at TP6, 6 at TP3): 256 routed experts,
// widths 3 / 4 per expert (every K4MOD-th at 4), every 5th expert with mixed fc1 widths through tr3.kmix (gate 2, up 3,
// down 3: the K2 kernels), tr3.table row 256 = the trellis shared expert; block-scale vectors; K3 / K2 tiles in the wide
// order (as the engine runs them). Routing comes from a router GEMV over x2 sets, as k1_ovl_test.
//
//   ptab_test [--M 1,2,3,4,5,6,7,8,20] [--pf-M 1024,4096] [--sets 24] [--layers 2] [--reps 200] [--k4mod 4]
//             [--check-only] [--bench-only] [--no-prefill] [--sabotage]
//
// Check (the claim: same bytes out, and the table is read every launch):
//   decode  for every M and fuse 1 / 0: out (BF16 [M][6144]) of the table pointing into the image (mode 1) and of every
//           piece copied into its own scattered slab (mode 2) == the pack layout, byte for byte, as plain launches and as
//           one CUDA graph over all sets; the pack layout run twice; the output finite and not all zero.
//   graph   a graph captured over the scatter table: replayed with the entries pointed at a SECOND scattered copy of the
//           same bytes == the same output; with the entries of same-size rows swapped pairwise (other bytes, same table
//           address) the output differs and matches a plain run on the swapped table; restored == the original again.
//   prefill for every --pf-M: the prefill MoE (bindp::launch_shared + launch_experts, mlp_part BF16 [M][6144]) through
//           mode 1 and mode 2 == the pack layout, byte for byte.
//   --sabotage: builds the scatter table with same-size rows swapped pairwise and expects the decode and prefill checks
//           to FAIL (the proof that the comparisons bite); exit 0 if they did fail, 1 if one passed.
//   --dump FILE / --cmp FILE: write the pack layout's outputs (every M, fuse, prefill size) / check them against a file
//           another build wrote (-DPTAB_MAIN against main: the knob-off path is byte-identical to main).
// Bench (idle GPU, after the check): per M a CUDA graph of reps x the layer call (sets and layers in turn), median of 5
// interleaved timings, pack layout vs mode 1 vs mode 2; prefill the same with plain launches. us per MoE layer.
// Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error / usage.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../glue.cuh"
#include "../bind.cuh"
#ifndef PTAB_MAIN
#include "../ptab.cuh"
#endif
#ifndef PTAB_NO_PREFILL
#include "../bind_prefill.cuh"
#endif
// -DPTAB_MAIN: the same program against a tree without the piece table (main before #151): only the pack layout runs, for
// --dump. Then --cmp FILE on the new build checks that its pack layout gives main's bytes (the knob-off gate).

#define CK(x)                                                                                                   \
    do                                                                                                          \
    {                                                                                                           \
        const cudaError_t e_ = (x);                                                                             \
        if (e_ != cudaSuccess)                                                                                  \
        {                                                                                                       \
            printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));             \
            exit(2);                                                                                            \
        }                                                                                                       \
    } while (0)

static int g_pass = 0, g_fail = 0;

__device__ __forceinline__ uint32_t trnd(size_t i, uint32_t seed)
{
    uint32_t x = static_cast<uint32_t>(i) ^ ((static_cast<uint32_t>(i >> 32) + seed) * 0x9e3779b9u) ^ (seed * 0x85ebca6bu);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float trnd_u(size_t i, uint32_t seed) { return 2.f * ((trnd(i, seed) >> 8) * (1.f / 16777216.f)) - 1.f; }
__global__ void k_fill_words(uint32_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = trnd(i, seed);
}
__global__ void k_fill_f16(__half* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = __float2half(trnd_u(i, seed) * amp);
}
__global__ void k_fill_f32(float* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = trnd_u(i, seed) * amp;
}
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = __float2bfloat16(trnd_u(i, seed) * amp);
}
// x2 sets [sets][rows][HID]: row 0 noise, row m = corr * row 0 + (1 - corr) * noise
__global__ void k_fill_x2(__half* p, int sets, int rows, float corr, uint32_t seed)
{
    const size_t n = static_cast<size_t>(sets) * rows * HID;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const size_t s = i / (static_cast<size_t>(rows) * HID), m = (i / HID) % rows, k = i % HID;
        const float r0 = trnd_u(s * rows * HID + k, seed);
        p[i] = __float2half(m == 0 ? r0 : corr * r0 + (1.f - corr) * trnd_u(i, seed + 1));
    }
}
static void fill_words(void* p, size_t bytes, uint32_t seed)
{
    k_fill_words<<<4096, 256>>>(static_cast<uint32_t*>(p), bytes / 4, seed);
    CK(cudaGetLastError());
}
static void fill_f16(__half* p, size_t n, float amp, uint32_t seed)
{
    k_fill_f16<<<1024, 256>>>(p, n, amp, seed);
    CK(cudaGetLastError());
}

static bind::Tensor mk(const void* p, int nd, long long a, long long b = 0, long long c = 0)
{
    bind::Tensor t;
    t.p = p; t.nd = nd; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    return t;
}

struct SynLayer
{
    bind::LayerTensors T;             // the pack layout (ptab = nullptr)
    std::vector<long long> tab, kmix; // host copies [257][6], [257]
    double ebytes[257];
    bind::Tensor router;              // BF16 records [16][384][512]
    float* bias = nullptr;            // [256]
#ifndef PTAB_MAIN
    ptab::Table img, sct, sct2;       // mode 1, mode 2, and a second scattered copy (the graph content check)
#endif
};

// widths: expert e: e % 5 == 1 -> kmix (gate 2, up 3 per block, down 3: the K2 kernels); else e % k4mod == 0 -> 4, else 3
static void build_layer(SynLayer& L, int k4mod, uint32_t seed, std::vector<void*>& mem)
{
    L.tab.assign(257 * 6, 0);
    L.kmix.assign(257, 0);
    size_t o1 = 0, o2 = 0;
    int row = 0;
    for (int e = 0; e < 257; ++e)
    {
        const int nb = g6nr::EB;
        const bool mixed = e < 256 && e % 5 == 1;
        const int bits = e == 256 ? 4 : mixed ? 3 : e % k4mod == 0 ? 4 : 3;
        long long* r = &L.tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = bits; r[5] = 0;
        long long km = 0;
        size_t f1 = 0;
        for (int b = 0; b < nb; ++b)
        {
            const int g = mixed ? 2 : bits, u = mixed ? 3 : bits;
            if (mixed) km |= (static_cast<long long>(g) << (8 * b)) | (static_cast<long long>(u) << (8 * b + 4));
            f1 += static_cast<size_t>(g + u) * KSH * 8 * 32;
        }
        L.kmix[e] = km;
        o1 += f1;
        o2 += static_cast<size_t>(nb) * KSH * 8 * 32 * bits;
        L.ebytes[e] = static_cast<double>(f1) + static_cast<double>(nb) * KSH * 8 * 32 * bits;
        row += nb;
    }
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    long long* tab = static_cast<long long*>(alloc(L.tab.size() * 8));
    CK(cudaMemcpy(tab, L.tab.data(), L.tab.size() * 8, cudaMemcpyHostToDevice));
    long long* km = static_cast<long long*>(alloc(L.kmix.size() * 8));
    CK(cudaMemcpy(km, L.kmix.data(), L.kmix.size() * 8, cudaMemcpyHostToDevice));
    uint8_t* fc1 = static_cast<uint8_t*>(alloc(o1));
    uint8_t* fc2 = static_cast<uint8_t*>(alloc(o2));
    fill_words(fc1, o1, seed + 1);
    fill_words(fc2, o2, seed + 2);
    __half* v2 = static_cast<__half*>(alloc(static_cast<size_t>(row) * 3 * 128 * 2));
    fill_f16(v2, static_cast<size_t>(row) * 3 * 128, 0.5f, seed + 3);
    std::vector<__half> sgh(2 * HID);
    for (int i = 0; i < 2 * HID; ++i) sgh[i] = __float2half(((i * 2654435761u + seed) >> 13) & 1 ? 1.f : -1.f);
    __half* sg = static_cast<__half*>(alloc(sgh.size() * 2));
    CK(cudaMemcpy(sg, sgh.data(), sgh.size() * 2, cudaMemcpyHostToDevice));
    __half* v1b = static_cast<__half*>(alloc(static_cast<size_t>(row) * 2 * (HID / 128) * 2));
    fill_f16(v1b, static_cast<size_t>(row) * 2 * (HID / 128), 0.05f, seed + 4);
    __half* v3b = static_cast<__half*>(alloc(static_cast<size_t>(row) * (HID / 128) * 2));
    fill_f16(v3b, static_cast<size_t>(row) * (HID / 128), 0.5f, seed + 5);
    bind::LayerTensors& T = L.T;
    T.t[bind::T_TR3_TAB] = mk(tab, 2, 257, 6);
    T.t[bind::T_TR3_KMIX] = mk(km, 1, 257);
    T.t[bind::T_TR3_FC1] = mk(fc1, 1, static_cast<long long>(o1));
    T.t[bind::T_TR3_FC2] = mk(fc2, 1, static_cast<long long>(o2));
    T.t[bind::T_TR3_V2] = mk(v2, 3, row, 3, 128);
    T.t[bind::T_TR3_SG] = mk(sg, 2, 2, HID);
    T.t[bind::T_TR3_V1B] = mk(v1b, 3, row, 2, HID / 128);
    T.t[bind::T_TR3_V3B] = mk(v3b, 2, row, HID / 128);
    for (int i : {bind::T_TR3_TAB, bind::T_TR3_FC1, bind::T_TR3_FC2, bind::T_TR3_V2, bind::T_TR3_KMIX}) T.t[i].fmt = bind::FMT_EXLR8_E4M3;
    T.e4m3 = 1;
    if (const char* e = bind::scan_widths(T)) { printf("scan_widths: %s\n", e); exit(2); }
    if (!T.sh_trellis) { printf("no trellis shared expert after scan_widths\n"); exit(2); }
    if (!T.k2) { printf("expected 2-bit units (tr3.kmix) in the synthetic layer\n"); exit(2); }
    bind::widen_fc1(T, nullptr);
    bind::widen_fc2(T, nullptr);
    const size_t rn = static_cast<size_t>(16) * 384 * 256;
    __nv_bfloat16* rw = static_cast<__nv_bfloat16*>(alloc(rn * 2));
    k_fill_bf16<<<1024, 256>>>(rw, rn, 0.02f, seed + 6);
    CK(cudaGetLastError());
    L.router = mk(rw, 3, 16, 384, 512);
    std::vector<float> bh(256);
    for (int e = 0; e < 256; ++e) bh[e] = 0.01f * static_cast<float>(((e * 2654435761u + seed) >> 9) % 201) / 100.f - 0.01f;
    L.bias = static_cast<float*>(alloc(256 * 4));
    CK(cudaMemcpy(L.bias, bh.data(), 256 * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
}

#ifndef PTAB_MAIN
static void build_tables(SynLayer& L, ptab::Pool& pool, cudaStream_t st)
{
    const bind::LayerTensors& T = L.T;
    auto bld = [&](ptab::Table& t, ptab::Mode m) {
        if (const char* e = ptab::build(t, T.t[bind::T_TR3_TAB].p, T.t[bind::T_TR3_KMIX].p, static_cast<const uint8_t*>(T.t[bind::T_TR3_FC1].p),
                                        static_cast<const uint8_t*>(T.t[bind::T_TR3_FC2].p), 257, g6nr::EB, m, pool, st))
        { printf("ptab::build: %s\n", e); exit(2); }
    };
    bld(L.img, ptab::IMAGE);
    bld(L.sct, ptab::SCATTER);
    bld(L.sct2, ptab::SCATTER);
    // the image table points into the image; the scatter tables into slabs; no scatter entry equals its image entry
    for (size_t i = 0; i < L.img.n(); ++i)
        if (L.img.bytes[i] && (L.img.host[i] == L.sct.host[i] || L.sct.host[i] == L.sct2.host[i])) { printf("scatter table entry %zu not scattered\n", i); exit(2); }
    // adjacent pieces of the image are not adjacent in the slabs (what the kernels must not rely on)
    size_t adj = 0;
    for (size_t i = 2; i < L.sct.n(); ++i)
        if (L.sct.bytes[i] && L.sct.bytes[i - 2] && L.sct.host[i] == L.sct.host[i - 2] + ptab::Pool::round(L.sct.bytes[i - 2])) ++adj;
    printf("tables: %zu entries, scatter: %zu adjacent-piece pairs kept adjacent (of %zu), %.1f MB of slabs per table\n",
           L.sct.n(), adj, L.sct.n() - 2, L.sct.slab_bytes / 1e6);
}
#endif

#ifndef PTAB_MAIN
// Swaps the entries of every same-size pair of rows (consecutive rows of one width class without tr3.kmix, row 256
// among the width-4 ones): other bytes at the same table address for every routed expert and the shared expert, with
// no read past a slab's end (the pair's slabs have the same bytes). Its own inverse: apply twice to restore.
static void swap_pairs(SynLayer& L, ptab::Table& t, cudaStream_t st)
{
    std::vector<int> cls[2];          // width 3, width 4
    for (int e = 0; e < 257; ++e)
    {
        if (L.kmix[e]) continue;
        cls[L.tab[static_cast<size_t>(e) * 6 + 4] == 4].push_back(e);
    }
    int pairs = 0;
    for (const std::vector<int>& c : cls)
        for (size_t i = 0; i + 1 < c.size(); i += 2, ++pairs)
            for (int b = 0; b < g6nr::EB; ++b)
            {
                const int e1 = c[i], e2 = c[i + 1];
                const uint8_t *f1 = t.host[t.slot(e1, b)], *f2 = t.host[t.slot(e1, b) + 1];
                if (const char* e = ptab::set(t, e1, b, t.host[t.slot(e2, b)], t.host[t.slot(e2, b) + 1], st)) { printf("%s\n", e); exit(2); }
                if (const char* e = ptab::set(t, e2, b, f1, f2, st)) { printf("%s\n", e); exit(2); }
            }
    CK(cudaStreamSynchronize(st));
    printf("swap_pairs: %d pairs of rows swapped in the scatter table (every expert without tr3.kmix, and row 256)\n", pairs);
}
#endif

static std::vector<int> parse_list(const char* s)
{
    std::vector<int> v;
    for (const char* p = s; *p;)
    {
        v.push_back(atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

static bool cmp_bytes(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t rowb, bool expect_equal = true)
{
    size_t bad = 0, nbad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (!nbad) bad = i; ++nbad; }
    const bool equal = nbad == 0, ok = equal == expect_equal;
    printf("%s %s: %s", ok ? "PASS" : "FAIL", what, equal ? "identical" : "differ");
    if (!equal) printf(" (%zu bytes, first at row %zu col %zu)", nbad, bad / rowb, (bad % rowb) / 2);
    if (!expect_equal) printf(" [expected to differ]");
    printf("\n");
    (ok ? g_pass : g_fail)++;
    return ok;
}
static bool sane(const char* what, const std::vector<uint8_t>& ref)
{
    size_t nz = 0, nonfin = 0;
    for (size_t i = 0; i + 1 < ref.size(); i += 2)
    {
        const uint16_t b = static_cast<uint16_t>(ref[i] | (ref[i + 1] << 8));
        nz += (b & 0x7fff) != 0;
        nonfin += (b & 0x7f80) == 0x7f80;
    }
    const bool ok = nonfin == 0 && nz > ref.size() / 4;
    printf("%s %s reference output: %zu non-finite, %.1f%% nonzero\n", ok ? "PASS" : "FAIL", what, nonfin, 200.0 * nz / ref.size());
    (ok ? g_pass : g_fail)++;
    return ok;
}

int main(int argc, char** argv)
{
    std::vector<int> Ms = {1, 2, 3, 4, 5, 6, 7, 8}, PMs = {1024, 4096};
    if (G6_MX >= 20) Ms.push_back(20);
    int sets = 24, nlay = 2, reps = 200, k4mod = 4, check = 1, bench = 1, prefill = 1, sabotage = 0;
    std::string dump, cmp;
    const float corr = 0.6f;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { printf("missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--M") Ms = parse_list(nx());
        else if (a == "--pf-M") PMs = parse_list(nx());
        else if (a == "--sets") sets = atoi(nx());
        else if (a == "--layers") nlay = atoi(nx());
        else if (a == "--reps") reps = atoi(nx());
        else if (a == "--k4mod") k4mod = atoi(nx());
        else if (a == "--check-only") bench = 0;
        else if (a == "--bench-only") check = 0;
        else if (a == "--no-prefill") prefill = 0;
        else if (a == "--sabotage") sabotage = 1;
        else if (a == "--dump") dump = nx();
        else if (a == "--cmp") cmp = nx();
        else { printf("unknown arg %s (see the file header)\n", a.c_str()); return 2; }
    }
    for (int M : Ms) if (M < 1 || M > G6_MX) { printf("M=%d outside 1..%d\n", M, G6_MX); return 2; }
    if (sets < 1 || nlay < 1 || reps < 1 || k4mod < 1) { printf("bad --sets / --layers / --reps / --k4mod\n"); return 2; }
#ifdef PTAB_NO_PREFILL
    prefill = 0;
#endif
#ifdef PTAB_MAIN
    bench = 0; sabotage = 0;          // main's build: the pack layout only, for --dump
    if (dump.empty()) { printf("PTAB_MAIN build: --dump FILE\n"); return 2; }
#endif
    // --dump / --cmp: the pack layout's bytes of every check, in order (decode per M and fuse, then prefill per M and layer)
    FILE* fdump = nullptr;
    std::vector<uint8_t> cmpdata;
    size_t cmppos = 0;
    if (!dump.empty() && !(fdump = fopen(dump.c_str(), "wb"))) { printf("cannot write %s\n", dump.c_str()); return 2; }
    if (!cmp.empty())
    {
        FILE* f = fopen(cmp.c_str(), "rb");
        if (!f) { printf("cannot read %s\n", cmp.c_str()); return 2; }
        fseek(f, 0, SEEK_END);
        cmpdata.resize(static_cast<size_t>(ftell(f)));
        fseek(f, 0, SEEK_SET);
        if (fread(cmpdata.data(), 1, cmpdata.size(), f) != cmpdata.size()) { printf("short read of %s\n", cmp.c_str()); return 2; }
        fclose(f);
        printf("--cmp %s: %.1f MB of reference bytes\n", cmp.c_str(), cmpdata.size() / 1e6);
    }
    auto dumpcmp = [&](const char* what, const std::vector<uint8_t>& ref, size_t rowb) {
        if (fdump) fwrite(ref.data(), 1, ref.size(), fdump);
        if (cmp.empty()) return;
        if (cmppos + ref.size() > cmpdata.size()) { printf("FAIL %s == main: reference file too short\n", what); ++g_fail; cmppos = cmpdata.size(); return; }
        std::vector<uint8_t> m(cmpdata.begin() + cmppos, cmpdata.begin() + cmppos + ref.size());
        cmppos += ref.size();
        cmp_bytes((std::string(what) + " pack layout == main's build").c_str(), m, ref, rowb);
    };

    CK(cudaSetDevice(0));
    CK(bind::init());
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("ptab_test: %s sm_%d%d %d SMs, clock attr %d MHz, G6_NR=%d EB=%d G6_MX=%d, layers %d, sets %d, K4 = experts %% %d == 0, kmix on e %% 5 == 1%s\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount, clk / 1000, g6nr::NR, g6nr::EB, G6_MX, nlay, sets, k4mod, sabotage ? ", SABOTAGE" : "");

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::vector<void*> mem;
    std::vector<SynLayer> Ls(nlay);
#ifndef PTAB_MAIN
    ptab::Pool pool;
#endif
    for (int l = 0; l < nlay; ++l)
    {
        build_layer(Ls[l], k4mod, 1000u + 17u * l, mem);
#ifndef PTAB_MAIN
        build_tables(Ls[l], pool, st);
#endif
    }
#ifndef PTAB_MAIN
    printf("pool: %.1f MB held in %zu chunks, %.1f MB in slabs\n", pool.held / 1e6, pool.chunks.size(), pool.used / 1e6);
    if (sabotage)      // same-size rows swapped pairwise in every layer's scatter table: the checks must then FAIL
        for (SynLayer& L : Ls) swap_pairs(L, L.sct, st);
#endif
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    const size_t xb = static_cast<size_t>(G6_MX) * HID * 2;
    __half* x2 = static_cast<__half*>(alloc(xb));
    __half* x2sets = static_cast<__half*>(alloc(xb * sets));
    k_fill_x2<<<1024, 256>>>(x2sets, sets, G6_MX, corr, 41u);
    CK(cudaGetLastError());
    float* logits = static_cast<float*>(alloc(static_cast<size_t>(G6_MX) * 256 * 4));
    glue::RoutePlan* dplan = static_cast<glue::RoutePlan*>(alloc(sizeof(glue::RoutePlan)));
    CK(cudaMemset(dplan, 0, sizeof(glue::RoutePlan)));
    const size_t sb = bind::scratch_bytes();
    void* blk = alloc(sb);
    CK(cudaMemset(blk, 0, sb));
    const bind::Scratch sc = bind::scratch_at(blk);
    const size_t obytes = static_cast<size_t>(G6_MX) * HID * 2;
    uint8_t* out = static_cast<uint8_t*>(alloc(obytes));
    uint8_t* outs = static_cast<uint8_t*>(alloc(obytes * sets));
    CK(cudaDeviceSynchronize());
    {
        size_t fr = 0, to = 0;
        CK(cudaMemGetInfo(&fr, &to));
        printf("device memory in use %.2f GB (weights %d layers x 3 tables + scratch %.0f MB)\n", (to - fr) / 1e9, nlay, sb / 1e6);
    }

    // the layer's tensors with the table of variant v: 0 = pack layout, 1 = image table, 2 = scatter table, 3 = scatter copy 2
    auto tens = [&](int l, int v) {
        bind::LayerTensors T = Ls[l].T;
#ifndef PTAB_MAIN
        T.ptab = v == 0 ? nullptr : v == 1 ? Ls[l].img.dev : v == 2 ? Ls[l].sct.dev : Ls[l].sct2.dev;
#else
        (void)v;
#endif
        return T;
    };
    struct Bound { bind::K1 k1; bind::Gv router; };
    auto make = [&](int M, int fuse, int v, std::vector<Bound>& bs) -> const char* {
        bs.assign(nlay, Bound());
        for (int l = 0; l < nlay; ++l)
        {
            bind::Tensor sh[3];
            const bind::LayerTensors T = tens(l, v);
            if (const char* e = bind::make_k1(T, M, sc, x2, sc.xf_x2, bind::plan_ptrs(dplan), out, 1, bs[l].k1, sh, fuse, 0)) return e;
            if (const char* e = bind::make_gv(Ls[l].router, bind::BF16W, 2, 16, M, sc.xb_x2, logits, gv::O_F32, bs[l].router)) return e;
        }
        return nullptr;
    };
    auto call = [&](const std::vector<Bound>& bs, int M, int s, int l) {
        CK(cudaMemcpyAsync(x2, x2sets + static_cast<size_t>(s) * G6_MX * HID, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
        bind::Xfrag xf;
        xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;
        bind::launch(xf, st);
        bind::launch(bs[l].router, st);
        glue::k_router_plan<<<1, 256, 0, st>>>(logits, Ls[l].bias, M, dplan);
        bind::launch_k1(bs[l].k1, st);
    };
    auto plain = [&](const std::vector<Bound>& bs, int M, std::vector<uint8_t>& h) {
        const size_t ob = static_cast<size_t>(M) * HID * 2;
        h.assign(ob * sets, 0);
        for (int s = 0; s < sets; ++s)
        {
            CK(cudaMemsetAsync(out, 0xff, obytes, st));
            call(bs, M, s, s % nlay);
            CK(cudaMemcpyAsync(h.data() + ob * s, out, ob, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
    };
    auto capture = [&](const std::vector<Bound>& bs, int M) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));   // as g6m4
        for (int s = 0; s < sets; ++s)
        {
            call(bs, M, s, s % nlay);
            CK(cudaMemcpyAsync(outs + static_cast<size_t>(M) * HID * 2 * s, out, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphDestroy(g));
        return ge;
    };
    auto replay = [&](cudaGraphExec_t ge, int M, std::vector<uint8_t>& h) {
        const size_t ob = static_cast<size_t>(M) * HID * 2;
        h.assign(ob * sets, 0);
        CK(cudaMemsetAsync(outs, 0xff, obytes * sets, st));
        for (int r = 0; r < 2; ++r) CK(cudaGraphLaunch(ge, st));    // twice: the second replay is compared
        CK(cudaMemcpyAsync(h.data(), outs, ob * sets, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
    };

    // ---------------------------------------------------------------- decode exactness
    bool sab_decode = false, sab_prefill = false;
    if (check)
    {
        const int sab_fail0 = g_fail;
        for (int M : Ms)
            for (int fuse : {1, 0})
            {
                std::vector<Bound> b0, b1, b2;
                const char* e = make(M, fuse, 0, b0);
#ifndef PTAB_MAIN
                if (!e) e = make(M, fuse, 1, b1);
                if (!e) e = make(M, fuse, 2, b2);
#endif
                if (e) { printf("FAIL M=%d fuse=%d make: %s\n", M, fuse, e); ++g_fail; continue; }
                const size_t ob = static_cast<size_t>(M) * HID * 2;
                char tag[96];
                snprintf(tag, sizeof tag, "decode M=%d fuse=%d(%s)", M, fuse, b0[0].k1.fuse ? "fused" : "separate");
                std::vector<uint8_t> ref, got, tmp;
                plain(b0, M, ref);
                sane(tag, ref);
                plain(b0, M, tmp); cmp_bytes((std::string(tag) + " pack layout run twice").c_str(), ref, tmp, ob);
                dumpcmp(tag, ref, ob);
#ifdef PTAB_MAIN
                continue;
#endif
                plain(b1, M, got); cmp_bytes((std::string(tag) + " image table == pack layout (plain)").c_str(), ref, got, ob);
                plain(b2, M, got); cmp_bytes((std::string(tag) + " scattered slabs == pack layout (plain)").c_str(), ref, got, ob);
                cudaGraphExec_t g1 = capture(b1, M), g2 = capture(b2, M);
                replay(g1, M, got); cmp_bytes((std::string(tag) + " image table == pack layout (graph)").c_str(), ref, got, ob);
                replay(g2, M, got); cmp_bytes((std::string(tag) + " scattered slabs == pack layout (graph)").c_str(), ref, got, ob);
                CK(cudaGraphExecDestroy(g1));
                CK(cudaGraphExecDestroy(g2));
            }
        if (sabotage)
        {
            const bool bit = g_fail > sab_fail0;
            printf("%s: sabotage (swapped table entries) %s the decode check\n", bit ? "OK" : "FAILED", bit ? "FAILED, as it must" : "PASSED, the check does not bite");
            sab_decode = bit;
        }
    }

    // ---------------------------------------------------------------- the graph reads the table every replay
#ifndef PTAB_MAIN
    if (check && !sabotage)
    {
        const int M = Ms[0];
        std::vector<Bound> b2;
        if (const char* e = make(M, 1, 2, b2)) { printf("FAIL graph-content make: %s\n", e); ++g_fail; }
        else
        {
            const size_t ob = static_cast<size_t>(M) * HID * 2;
            std::vector<uint8_t> ref, got, swp;
            cudaGraphExec_t g2 = capture(b2, M);
            replay(g2, M, ref);
            // 1. every entry re-pointed at the second scattered copy (other addresses, the same bytes)
            std::vector<std::vector<const uint8_t*>> saved(nlay);
            for (int l = 0; l < nlay; ++l)
            {
                saved[l] = Ls[l].sct.host;
                Ls[l].sct.host = Ls[l].sct2.host;
                if (const char* e = ptab::upload(Ls[l].sct, st)) { printf("%s\n", e); return 2; }
            }
            replay(g2, M, got);
            cmp_bytes("graph M=1: replay after the table was re-pointed at the second scattered copy == before", ref, got, ob);
            // 2. same-size rows swapped pairwise (other bytes at the same table address): the graph's output changes, and equals a plain run on the swapped table
            for (int l = 0; l < nlay; ++l)
            {
                Ls[l].sct.host = saved[l];
                if (const char* e = ptab::upload(Ls[l].sct, st)) { printf("%s\n", e); return 2; }
                swap_pairs(Ls[l], Ls[l].sct, st);
            }
            replay(g2, M, got);
            cmp_bytes("graph M=1: replay after the table's rows were swapped pairwise differs from before", ref, got, ob, false);
            plain(b2, M, swp);
            cmp_bytes("graph M=1: that replay == a plain run on the swapped table", swp, got, ob);
            // 3. restored
            for (int l = 0; l < nlay; ++l)
            {
                Ls[l].sct.host = saved[l];
                if (const char* e = ptab::upload(Ls[l].sct, st)) { printf("%s\n", e); return 2; }
            }
            replay(g2, M, got);
            cmp_bytes("graph M=1: replay after the table was restored == before", ref, got, ob);
            CK(cudaGraphExecDestroy(g2));
        }
    }
#endif

#ifndef PTAB_NO_PREFILL
    // ---------------------------------------------------------------- prefill exactness and bench
    struct Pf
    {
        int M = 0;
        void* blk = nullptr;
        bindp::Scratch sc;
        __half* x = nullptr;
        float* logits = nullptr;
        uint16_t* out = nullptr;
    };
    auto pf_setup = [&](int M) {
        Pf p;
        p.M = M;
        CK(cudaMalloc(&p.blk, bindp::scratch_bytes(M)));
        CK(cudaMemset(p.blk, 0, bindp::scratch_bytes(M)));
        p.sc = bindp::scratch_at(p.blk, M);
        p.x = static_cast<__half*>(alloc(static_cast<size_t>(M) * HID * 2));
        fill_f16(p.x, static_cast<size_t>(M) * HID, 1.f, 77u + M);
        p.logits = static_cast<float*>(alloc(static_cast<size_t>(M) * 256 * 4));
        k_fill_f32<<<1024, 256>>>(p.logits, static_cast<size_t>(M) * 256, 1.f, 78u + M);
        CK(cudaGetLastError());
        p.out = static_cast<uint16_t*>(alloc(static_cast<size_t>(M) * HID * 2));
        return p;
    };
    // the routed pass and the trellis shared pass of layer l through variant v, as pf_v2_test binds them
    auto pf_make = [&](const Pf& p, int l, int v, bindp::Expert& A, bindp::Expert& S) {
        const bind::LayerTensors D = tens(l, v);
        const bind::Tensor &tab = D.t[bind::T_TR3_TAB], &v2 = D.t[bind::T_TR3_V2];
        A = bindp::Expert();
        A.on = 1; A.M = p.M; A.maxseg = bindp::max_segments(p.M);
        A.logits = p.logits; A.bias = Ls[l].bias;
        A.tok_e = p.sc.tok_e; A.tok_w = p.sc.tok_w; A.cnt = p.sc.cnt;
        A.rr_row = p.sc.rr_row; A.rr_e = p.sc.rr_e; A.rr_sp = p.sc.rr_sp; A.rr_w = p.sc.rr_w; A.pos = p.sc.pos; A.segs = p.sc.segs; A.n_seg = p.sc.n_seg;
        A.t_tab = static_cast<const long long*>(tab.p);
        A.t_kmix = static_cast<const long long*>(D.t[bind::T_TR3_KMIX].p);
        A.k2 = D.k2; A.cb = D.cb; A.bs = 1;
        A.t_sg = static_cast<const __half*>(D.t[bind::T_TR3_SG].p);
        A.t_v1b = static_cast<const __half*>(D.t[bind::T_TR3_V1B].p);
        A.t_v3b = static_cast<const __half*>(D.t[bind::T_TR3_V3B].p);
        A.t_fc1 = static_cast<const uint8_t*>(D.t[bind::T_TR3_FC1].p); A.t_wide = D.fc1_wide;
        A.t_fc2 = static_cast<const uint8_t*>(D.t[bind::T_TR3_FC2].p); A.t_wide2 = D.fc2_wide;
#ifndef PTAB_MAIN
        A.t_ptab = D.ptab;
#endif
        A.t_v2 = static_cast<const __half*>(v2.p);
        A.x = p.x; A.ldx = HID;
        A.z1 = p.sc.z1; A.x2 = p.sc.x2; A.yw = p.sc.yw; A.ysh = p.sc.ysh; A.out = p.out;
#ifdef PF_F12
        A.f12 = 1;
#endif
        S = A;
        S.nroutes = p.M; S.maxseg = (p.M + bindp::RS - 1) / bindp::RS;
        S.rr_row = p.sc.sh_rr_row; S.rr_e = p.sc.sh_rr_e; S.rr_sp = p.sc.sh_rr_sp; S.rr_w = p.sc.sh_rr_w;
        S.segs = p.sc.sh_segs; S.n_seg = p.sc.sh_n_seg; S.pos = nullptr; S.yw = p.sc.ysh;
#ifdef PF_F12
        S.f12 = 0;                    // as bindp::make_layer: the shared pass takes the stock fc1
#endif
        const size_t row = static_cast<size_t>(D.sh_tr[2]);
        A.sh_svh = A.t_sg + HID; A.sh_sc = A.t_v3b + row * (HID / 128);
    };
    auto pf_run = [&](const Pf& p, int l, int v, std::vector<uint8_t>* h) {
        bindp::Expert A, S;
        pf_make(p, l, v, A, S);
        CK(cudaMemsetAsync(p.out, 0xff, static_cast<size_t>(p.M) * HID * 2, st));
        bindp::launch_shared(S, st);
        bindp::launch_experts(A, st);
        if (h)
        {
            h->assign(static_cast<size_t>(p.M) * HID * 2, 0);
            CK(cudaMemcpyAsync(h->data(), p.out, h->size(), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
        }
    };
    if (prefill)
    {
        const int pf_fail0 = g_fail;
        for (int M : PMs)
        {
            Pf p = pf_setup(M);
            const size_t ob = static_cast<size_t>(M) * HID * 2;
            if (check)
                for (int l = 0; l < nlay; ++l)
                {
                    char tag[96];
                    snprintf(tag, sizeof tag, "prefill M=%d layer %d", M, l);
                    std::vector<uint8_t> ref, got;
                    pf_run(p, l, 0, &ref);
                    sane(tag, ref);
                    pf_run(p, l, 0, &got); cmp_bytes((std::string(tag) + " pack layout run twice").c_str(), ref, got, ob);
                    dumpcmp(tag, ref, ob);
#ifndef PTAB_MAIN
                    pf_run(p, l, 1, &got); cmp_bytes((std::string(tag) + " image table == pack layout").c_str(), ref, got, ob);
                    pf_run(p, l, 2, &got); cmp_bytes((std::string(tag) + " scattered slabs == pack layout").c_str(), ref, got, ob);
#endif
                }
            if (bench)
            {
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0));
                CK(cudaEventCreate(&e1));
                const int NV = 3;
                const char* name[NV] = {"pack layout", "image table", "scattered"};
                std::vector<double> t[NV];
                const int preps = std::max(5, reps / 10);
                for (int rep = 0; rep < 5; ++rep)
                    for (int v = 0; v < NV; ++v)
                    {
                        for (int i = 0; i < 2; ++i) pf_run(p, i % nlay, v, nullptr);
                        CK(cudaStreamSynchronize(st));
                        CK(cudaEventRecord(e0, st));
                        for (int i = 0; i < preps; ++i) pf_run(p, i % nlay, v, nullptr);
                        CK(cudaEventRecord(e1, st));
                        CK(cudaEventSynchronize(e1));
                        float ms = 0;
                        CK(cudaEventElapsedTime(&ms, e0, e1));
                        t[v].push_back(1000.0 * ms / preps);
                    }
                CK(cudaGetLastError());
                double med[NV];
                for (int v = 0; v < NV; ++v) { std::sort(t[v].begin(), t[v].end()); med[v] = t[v][2]; }
                for (int v = 0; v < NV; ++v)
                    printf("bench prefill M=%d %-12s us/layer=%8.1f (min %8.1f max %8.1f)  vs pack %+7.1f us (%+5.2f%%)\n", M, name[v], med[v], t[v][0], t[v][4],
                           med[v] - med[0], 100 * (med[v] - med[0]) / med[0]);
                fflush(stdout);
            }
            CK(cudaFree(p.blk));
        }
        if (sabotage && check)
        {
            sab_prefill = g_fail > pf_fail0;
            printf("%s: sabotage (swapped table entries) %s the prefill check\n", sab_prefill ? "OK" : "FAILED", sab_prefill ? "FAILED, as it must" : "PASSED, the check does not bite");
        }
    }
#else
    if (sabotage) sab_prefill = true;
#endif
    if (sabotage)
    {
        const bool bit = sab_decode && (sab_prefill || !prefill);
        printf("%s: sabotage %s\n", bit ? "OK" : "FAILED", bit ? "bit every check" : "did not bite every check");
        return bit ? 0 : 1;
    }

    // ---------------------------------------------------------------- decode bench
    if (bench)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        printf("bench decode: us per MoE layer call = graph of reps x [x2 copy, xfrag, router GEMV, plan, K1] / reps, median of 5 interleaved\n");
        for (int M : Ms)
        {
            const int NV = 3;
            const char* name[NV] = {"pack layout", "image table", "scattered"};
            std::vector<Bound> bs[NV];
            bool ok = true;
            for (int v = 0; v < NV && ok; ++v)
                if (const char* e = make(M, 1, v, bs[v])) { printf("M=%d make: %s\n", M, e); ok = false; }
            if (!ok) continue;
            double nex = 0, bytes = 0;
            for (int s = 0; s < sets; ++s)
            {
                CK(cudaMemcpyAsync(x2, x2sets + static_cast<size_t>(s) * G6_MX * HID, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
                bind::Xfrag xf;
                xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;
                bind::launch(xf, st);
                bind::launch(bs[0][0].router, st);
                glue::k_router_plan<<<1, 256, 0, st>>>(logits, Ls[0].bias, M, dplan);
                glue::RoutePlan P;
                CK(cudaMemcpyAsync(&P, dplan, sizeof P, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::vector<int> ex(P.slot_expert, P.slot_expert + P.n_slots);
                ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
                nex += ex.size();
                bytes += Ls[0].ebytes[256];
                for (int e : ex) bytes += Ls[0].ebytes[e];
            }
            nex /= sets; bytes /= sets;
            auto cap = [&](int v) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < reps; ++i) call(bs[v], M, i % sets, (i / sets + i) % nlay);
                CK(cudaStreamEndCapture(st, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                CK(cudaGraphDestroy(g));
                return ge;
            };
            auto time = [&](cudaGraphExec_t ge) {
                CK(cudaGraphLaunch(ge, st));
                CK(cudaStreamSynchronize(st));
                float ms = 0;
                CK(cudaEventRecord(e0, st));
                CK(cudaGraphLaunch(ge, st));
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                return 1000.0 * ms / reps;
            };
            cudaGraphExec_t g[NV];
            for (int v = 0; v < NV; ++v) g[v] = cap(v);
            std::vector<double> t[NV];
            for (int rep = 0; rep < 5; ++rep)
                for (int v = 0; v < NV; ++v) t[v].push_back(time(g[v]));
            double med[NV];
            for (int v = 0; v < NV; ++v)
            {
                std::sort(t[v].begin(), t[v].end());
                med[v] = t[v][2];
                CK(cudaGraphExecDestroy(g[v]));
            }
            CK(cudaGetLastError());
            for (int v = 0; v < NV; ++v)
                printf("bench decode M=%2d %-12s form=%s experts=%.1f MB=%.2f us/layer=%7.1f (min %7.1f max %7.1f)  vs pack %+6.1f us (%+5.2f%%) = %+.2f ms/step @75\n",
                       M, name[v], bs[v][0].k1.fuse ? "fused" : "separate", nex, bytes / 1e6, med[v], t[v][0], t[v][4], med[v] - med[0],
                       100 * (med[v] - med[0]) / med[0], 75 * (med[v] - med[0]) / 1e3);
            fflush(stdout);
        }
    }
    if (fdump) { fclose(fdump); printf("dump: %s written\n", dump.c_str()); }
    if (!cmp.empty() && cmppos != cmpdata.size()) { printf("FAIL --cmp: %zu of %zu reference bytes consumed\n", cmppos, cmpdata.size()); ++g_fail; }
    for (void* p : mem) cudaFree(p);
    if (check) printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
