// k1_ovl_test: GR8_ROUTE_OVL (the trellis shared expert on a forked stream while the router GEMV and the plan kernel
// run; bind.cuh k1_ovl_fork / launch_k1_ovl, k1.cuh k1_tr3_fc1_ovl / _fc2_ovl) against the default MoE layer. One GB10,
// no pack files, no fabric. One MoE layer call as rank.cuh launch_mlp_block runs it: x2 -> fragment tables (k_xfrag),
// router GEMV (BF16 records [16][384][512], 3.15 MB, tpc 2 x ks 16 = 8 CTAs, as make_layer binds it) -> k_router_plan
// -> K1 (fc1, fc2, finalize). Synthetic exlr8 v2 layer in the GLM-5.3 TP4 rank shape: 256 routed experts x 4 blocks of
// 128 columns (g6nr::EB), width 4 for experts e % K4MOD == 0 and 3 for the rest; tr3.table row 256 = the trellis shared
// expert at width SHBITS (default 4; 3.15 / 4.72 MB at width 3 / 4); block-scale vectors; K3 tiles in the wide order.
// Routing comes from the router itself: x2 row m = corr * row 0 + (1 - corr) * noise (MTP verify rows share experts).
//
//   k1_ovl_test [--M 1,2,3,4,5,8,12] [--bench-M 1,3,12] [--sets 48] [--layers 2] [--reps 300] [--corr 0.6]
//               [--k4mod 4] [--shbits 4] [--check-only] [--bench-only]
//
// Check (the claim: same bytes out): for every M, fuse 1 (the engine's policy: fused at M = 1, separate above; the
// fork applies only to separate launches, so M = 1 checks the fallback) and fuse 0, every routing set: out (BF16
// [M][6144], the engine's mlp_part) of mode 1 and mode 2 == default, byte for byte, as plain stream launches and as
// one CUDA graph over all sets (fork / join as graph edges). The default runs twice (the reference is deterministic),
// and its output is checked finite and not all zero.
// Bench (idle GPU, after the check): per M, a CUDA graph of reps x the layer call (sets and layers in turn, so nothing
// stays in L2), median of 5 interleaved timings: default, mode 1, mode 2, and for reference the call without K1
// (router bucket: xfrag + router + plan) and the default without the router GEMV. ms/step = us x 75 MoE layers.
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
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = __float2bfloat16(trnd_u(i, seed) * amp);
}
// x2 sets [sets][G6_MX][HID]: row 0 noise, row m = corr * row 0 + (1 - corr) * noise
__global__ void k_fill_x2(__half* p, int sets, float corr, uint32_t seed)
{
    const size_t n = static_cast<size_t>(sets) * G6_MX * HID;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const size_t s = i / (static_cast<size_t>(G6_MX) * HID), m = (i / HID) % G6_MX, k = i % HID;
        const float r0 = trnd_u(s * G6_MX * HID + k, seed);
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
    bind::LayerTensors T;
    std::vector<long long> tab;      // host copy [257][6]
    double ebytes[257];              // trellis bytes of each expert on this rank (fc1 + fc2)
    bind::Tensor router;             // BF16 records [16][384][512]
    float* bias = nullptr;           // [256]
};

static void build_layer(SynLayer& L, int k4mod, int shbits, uint32_t seed, std::vector<void*>& mem)
{
    L.tab.assign(257 * 6, 0);
    size_t o1 = 0, o2 = 0;
    int row = 0;
    for (int e = 0; e < 257; ++e)
    {
        const int bits = e == 256 ? shbits : e % k4mod == 0 ? 4 : 3, nb = g6nr::EB;
        long long* r = &L.tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = bits; r[5] = 0;
        o1 += static_cast<size_t>(nb) * 2 * KSH * 8 * 32 * bits;
        o2 += static_cast<size_t>(nb) * KSH * 8 * 32 * bits;
        L.ebytes[e] = 3.0 * nb * KSH * 8 * 32 * bits;
        row += nb;
    }
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    long long* tab = static_cast<long long*>(alloc(L.tab.size() * 8));
    CK(cudaMemcpy(tab, L.tab.data(), L.tab.size() * 8, cudaMemcpyHostToDevice));
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
    T.t[bind::T_TR3_FC1] = mk(fc1, 1, static_cast<long long>(o1));
    T.t[bind::T_TR3_FC2] = mk(fc2, 1, static_cast<long long>(o2));
    T.t[bind::T_TR3_V2] = mk(v2, 3, row, 3, 128);
    T.t[bind::T_TR3_SG] = mk(sg, 2, 2, HID);
    T.t[bind::T_TR3_V1B] = mk(v1b, 3, row, 2, HID / 128);
    T.t[bind::T_TR3_V3B] = mk(v3b, 2, row, HID / 128);
    for (int i : {bind::T_TR3_TAB, bind::T_TR3_FC1, bind::T_TR3_FC2, bind::T_TR3_V2}) T.t[i].fmt = bind::FMT_EXLR8_E4M3;
    T.e4m3 = 1;
    if (const char* e = bind::scan_widths(T)) { printf("scan_widths: %s\n", e); exit(2); }
    if (!T.sh_trellis) { printf("no trellis shared expert after scan_widths\n"); exit(2); }
    bind::widen_fc1(T, nullptr);
    bind::widen_fc2(T, nullptr);
    // router: 256 outputs x 6144 in BF16 records (16 output tiles x 384 k16 groups x 512 B), uniform +-0.02
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

int main(int argc, char** argv)
{
    std::vector<int> Ms = {1, 2, 3, 4, 5, 8}, BMs = {1, 3};
    if (G6_MX >= 12) { Ms.push_back(12); BMs.push_back(12); }
    int sets = 48, nlay = 2, reps = 300, k4mod = 4, shbits = 4, check = 1, bench = 1;
    float corr = 0.6f;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { printf("missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--M") Ms = parse_list(nx());
        else if (a == "--bench-M") BMs = parse_list(nx());
        else if (a == "--sets") sets = atoi(nx());
        else if (a == "--layers") nlay = atoi(nx());
        else if (a == "--reps") reps = atoi(nx());
        else if (a == "--k4mod") k4mod = atoi(nx());
        else if (a == "--shbits") shbits = atoi(nx());
        else if (a == "--corr") corr = static_cast<float>(atof(nx()));
        else if (a == "--check-only") bench = 0;
        else if (a == "--bench-only") check = 0;
        else { printf("unknown arg %s (see the file header)\n", a.c_str()); return 2; }
    }
    for (int M : Ms) if (M < 1 || M > G6_MX) { printf("M=%d outside 1..%d\n", M, G6_MX); return 2; }
    for (int M : BMs) if (M < 1 || M > G6_MX) { printf("M=%d outside 1..%d\n", M, G6_MX); return 2; }
    if (sets < 1 || nlay < 1 || reps < 1 || k4mod < 1 || (shbits != 3 && shbits != 4)) { printf("bad --sets / --layers / --reps / --k4mod / --shbits\n"); return 2; }

    CK(cudaSetDevice(0));
    CK(bind::init());
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("k1_ovl_test: %s sm_%d%d %d SMs, G6_NR=%d EB=%d G6_MX=%d, layers %d, sets %d, corr %.2f, K4 = experts %% %d == 0, shared width %d\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount, g6nr::NR, g6nr::EB, G6_MX, nlay, sets, corr, k4mod, shbits);

    std::vector<void*> mem;
    std::vector<SynLayer> Ls(nlay);
    for (int l = 0; l < nlay; ++l) build_layer(Ls[l], k4mod, shbits, 1000u + 17u * l, mem);
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    const size_t xb = static_cast<size_t>(G6_MX) * HID * 2;
    __half* x2 = static_cast<__half*>(alloc(xb));
    __half* x2sets = static_cast<__half*>(alloc(xb * sets));
    k_fill_x2<<<1024, 256>>>(x2sets, sets, corr, 41u);
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
    uint8_t* outs = static_cast<uint8_t*>(alloc(obytes * sets));      // per routing set (graph check)
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    bind::K1Ovl O[3];                                                // [variant]: 0 = off, 1 / 2 = GR8_ROUTE_OVL mode
    for (int v = 1; v < 3; ++v)
    {
        O[v].mode = v;
        CK(cudaStreamCreateWithFlags(&O[v].side, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&O[v].fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&O[v].join, cudaEventDisableTiming));
    }
    CK(cudaDeviceSynchronize());
    {
        size_t fr = 0, to = 0;
        CK(cudaMemGetInfo(&fr, &to));
        printf("device memory in use %.2f GB (weights %d layers + scratch %.0f MB)\n", (to - fr) / 1e9, nlay, sb / 1e6);
    }

    struct Bound { bind::K1 k1; bind::Gv router; };
    auto make = [&](int M, int fuse, std::vector<Bound>& bs) -> const char* {
        bs.assign(nlay, Bound());
        for (int l = 0; l < nlay; ++l)
        {
            bind::Tensor sh[3];
            if (const char* e = bind::make_k1(Ls[l].T, M, sc, x2, sc.xf_x2, bind::plan_ptrs(dplan), out, 1, bs[l].k1, sh, fuse, 0)) return e;
            if (const char* e = bind::make_gv(Ls[l].router, bind::BF16W, 2, 16, M, sc.xb_x2, logits, gv::O_F32, bs[l].router)) return e;
        }
        return nullptr;
    };
    // one layer call (rank.cuh launch_mlp_block): set s, layer l, variant v; router / k1 = false leave that part out
    auto call = [&](const std::vector<Bound>& bs, int v, int M, int s, int l, bool router, bool k1) {
        CK(cudaMemcpyAsync(x2, x2sets + static_cast<size_t>(s) * G6_MX * HID, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
        bind::Xfrag xf;
        xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;
        bind::launch(xf, st);
        const bool ovl = k1 && bind::k1_ovl_on(bs[l].k1, O[v]);
        if (ovl) bind::k1_ovl_fork(bs[l].k1, O[v], st);
        if (router) bind::launch(bs[l].router, st);
        glue::k_router_plan<<<1, 256, 0, st>>>(logits, Ls[l].bias, M, dplan);
        if (!k1) return;
        if (ovl) bind::launch_k1_ovl(bs[l].k1, O[v], st);
        else bind::launch_k1(bs[l].k1, st);
    };

    // ---------------------------------------------------------------- exactness
    if (check)
    {
        for (int M : Ms)
            for (int fuse : {1, 0})
            {
                std::vector<Bound> bs;
                if (const char* e = make(M, fuse, bs)) { printf("FAIL M=%d fuse=%d make: %s\n", M, fuse, e); ++g_fail; continue; }
                const bool on = bind::k1_ovl_on(bs[0].k1, O[1]);
                const size_t ob = static_cast<size_t>(M) * HID * 2;
                std::vector<uint8_t> ref(ob * sets), got(ob * sets), tmp(ob * sets);
                auto plain = [&](int v, std::vector<uint8_t>& h) {
                    for (int s = 0; s < sets; ++s)
                    {
                        CK(cudaMemsetAsync(out, 0xff, obytes, st));
                        call(bs, v, M, s, s % nlay, true, true);
                        CK(cudaMemcpyAsync(h.data() + ob * s, out, ob, cudaMemcpyDeviceToHost, st));
                    }
                    CK(cudaStreamSynchronize(st));
                    CK(cudaGetLastError());
                };
                auto graph = [&](int v, std::vector<uint8_t>& h) {
                    cudaGraph_t g;
                    cudaGraphExec_t ge;
                    CK(cudaMemsetAsync(outs, 0xff, obytes * sets, st));
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));   // as g6m4
                    for (int s = 0; s < sets; ++s)
                    {
                        call(bs, v, M, s, s % nlay, true, true);
                        CK(cudaMemcpyAsync(outs + ob * s, out, ob, cudaMemcpyDeviceToDevice, st));
                    }
                    CK(cudaStreamEndCapture(st, &g));
                    CK(cudaGraphInstantiate(&ge, g, 0));
                    for (int r = 0; r < 2; ++r) CK(cudaGraphLaunch(ge, st));    // twice: the second replay is compared
                    CK(cudaMemcpyAsync(h.data(), outs, ob * sets, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    CK(cudaGetLastError());
                    CK(cudaGraphExecDestroy(ge));
                    CK(cudaGraphDestroy(g));
                };
                auto cmp = [&](const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
                    size_t bad = 0, nbad = 0;
                    for (size_t i = 0; i < a.size(); ++i)
                        if (a[i] != b[i]) { if (!nbad) bad = i; ++nbad; }
                    const bool ok = nbad == 0;
                    printf("%s M=%d fuse=%d(%s, fork %s) %s: %d sets x %d rows%s", ok ? "PASS" : "FAIL", M, fuse,
                           bs[0].k1.fuse ? "fused" : "separate", on ? "on" : "n/a", what, sets, M, ok ? "\n" : "");
                    if (!ok) printf(", %zu bytes differ, first at set %zu row %zu col %zu\n", nbad, bad / ob, (bad % ob) / (HID * 2), (bad % (HID * 2)) / 2);
                    (ok ? g_pass : g_fail)++;
                };
                plain(0, ref);
                {   // sanity of the reference: finite, not all zero; and the routing spread
                    size_t nz = 0, nonfin = 0;
                    for (size_t i = 0; i < ref.size(); i += 2)
                    {
                        const uint16_t b = static_cast<uint16_t>(ref[i] | (ref[i + 1] << 8));
                        nz += (b & 0x7fff) != 0;
                        nonfin += (b & 0x7f80) == 0x7f80;
                    }
                    const bool ok = nonfin == 0 && nz > ref.size() / 4;
                    printf("%s M=%d fuse=%d reference output: %zu non-finite, %.1f%% nonzero\n", ok ? "PASS" : "FAIL", M, fuse, nonfin,
                           200.0 * nz / ref.size());
                    (ok ? g_pass : g_fail)++;
                }
                plain(0, tmp);  cmp("default run twice (plain)", ref, tmp);
                plain(1, got);  cmp("ovl1 == default (plain launches)", ref, got);
                plain(2, got);  cmp("ovl2 == default (plain launches)", ref, got);
                graph(0, tmp);  cmp("default (graph) == default (plain)", ref, tmp);
                graph(1, got);  cmp("ovl1 (graph) == default", ref, got);
                graph(2, got);  cmp("ovl2 (graph) == default", ref, got);
            }
    }

    // ---------------------------------------------------------------- bench
    if (bench)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        printf("bench: us per MoE layer call = graph of reps x [x2 copy, xfrag, (fork), router GEMV, plan, K1] / reps, median of 5\n");
        for (int M : BMs)
        {
            std::vector<Bound> bs;
            if (const char* e = make(M, 1, bs)) { printf("M=%d make: %s\n", M, e); continue; }
            // mean distinct experts and this rank's trellis bytes per call
            double nex = 0, bytes = 0;
            for (int s = 0; s < sets; ++s)
            {
                call(bs, 0, M, s, 0, true, false);
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
            // rows: 0 default, 1 ovl mode 1, 2 ovl mode 2, 3 no K1 (router bucket), 4 default without the router GEMV
            const int NV = 5;
            const char* name[NV] = {"default", "ovl1", "ovl2", "router+plan", "k1+plan"};
            auto capture = [&](int r) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < reps; ++i) call(bs, r < 3 ? r : 0, M, i % sets, (i / sets + i) % nlay, r != 4, r != 3);
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
            for (int r = 0; r < NV; ++r) g[r] = capture(r);
            std::vector<double> t[NV];
            for (int rep = 0; rep < 5; ++rep)                       // interleaved, so drift hits every row alike
                for (int r = 0; r < NV; ++r) t[r].push_back(time(g[r]));
            double med[NV];
            for (int r = 0; r < NV; ++r)
            {
                std::sort(t[r].begin(), t[r].end());
                med[r] = t[r][2];
                CK(cudaGraphExecDestroy(g[r]));
            }
            CK(cudaGetLastError());
            const bool on = bind::k1_ovl_on(bs[0].k1, O[1]);
            for (int r = 0; r < NV; ++r)
                printf("bench M=%d %-11s form=%s%s experts=%.1f MB=%.2f us/layer=%7.1f (min %7.1f max %7.1f)  vs default %+6.1f us (%+5.1f%%) = %+.2f ms/step @75\n",
                       M, name[r], bs[0].k1.fuse ? "fused" : "separate", r == 1 || r == 2 ? (on ? "" : " (fork n/a)") : "", nex,
                       bytes / 1e6, med[r], t[r][0], t[r][4], med[r] - med[0], 100 * (med[r] - med[0]) / med[0], 75 * (med[r] - med[0]) / 1e3);
            fflush(stdout);
        }
    }
    for (void* p : mem) cudaFree(p);
    if (check) printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
