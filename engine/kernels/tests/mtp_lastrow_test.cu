// mtp_lastrow_test: G6_MTP_LASTROW (rank.cuh launch_mlp_block; glue.cuh k_router_plan keep / keep_dev) against the full
// MoE layer call. One GB10, no pack files, no fabric. The MTP draft's MoE layer call as launch_mlp_block runs it: x2 ->
// fragment tables (k_xfrag), router GEMV (BF16 records, tpc 2 x ks 16, as make_layer binds it) -> k_router_plan -> K1
// (fc1, fc2, finalize), on the synthetic exlr8 v2 layer of k1_ovl_test (GLM-5.3 TP4 rank shape: 256 routed experts x 4
// blocks of 128 columns, widths 4 / 3, trellis shared expert, block-scale vectors, K3 tiles in the wide order).
// Routing comes from the router itself: x2 row m = corr * row 0 + (1 - corr) * noise (the rows of a first form share
// experts).
//
//   mtp_lastrow_test [--M 1,2,3,4,5,6,7,8,12,16,24,32] [--bench-M 2,3,4,5,8] [--sets 48] [--layers 2] [--reps 300]
//                    [--corr 0.6] [--k4mod 4] [--shbits 4] [--check-only] [--bench-only]
//
// Check (the claim: the rows a draft step's readers take are the same bytes): for every M, fuse 1 (the engine's policy:
// fused at M = 1, separate above) and fuse 0, G6_ROUTE_OVL off / 1 / 2, every routing set:
//   last   keep = 1 << (M - 1) (today's path: the step's last row), by value as the S=1 draft graphs capture it
//   s2     keep_dev = the last rows of a session layout (S2: every session's last row; one layout per set, 1..min(M, 8)
//          sessions of random sizes), read from device memory at launch as k_s2_last_rows writes it
// against the full plan: the kept rows of out (BF16 [M][6144], the engine's mlp_part) byte for byte, as plain stream
// launches and as one CUDA graph over all sets (replayed twice, the second replay compared). The other rows (unread by
// the engine) must be finite. keep = 0 and keep = all ones must be the full plan byte for byte (flag off = today). The
// plan itself: the kept rows' tok_slot / tok_col / tok_w / tok_expert as the full plan's (tok_slot / tok_col through
// slot_expert: the same expert and column), the other rows -1.
// Bench (idle GPU, after the check): per M, a CUDA graph of reps x the layer call (sets and layers in turn, so nothing
// stays in L2), median of 5 interleaved timings: full plan, last row only, M = 1 (for reference), with the mean distinct
// experts and this rank's trellis bytes per call. The saving is the draft step's: layer 78 is its one MoE layer.
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

// as k1_ovl_test build_layer
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

// a session layout of M rows (1..min(M, 8) sessions, random sizes, in order): bit = each session's last row
static unsigned layout_mask(int M, std::mt19937& g)
{
    const int S = 1 + static_cast<int>(g() % static_cast<unsigned>(std::min(M, 8)));
    std::vector<int> cut;                     // S - 1 distinct cut points in 1..M-1
    std::vector<int> all;
    for (int i = 1; i < M; ++i) all.push_back(i);
    std::shuffle(all.begin(), all.end(), g);
    cut.assign(all.begin(), all.begin() + (S - 1));
    cut.push_back(M);
    unsigned b = 0;
    for (int c : cut) b |= 1u << (c - 1);
    return b;
}

int main(int argc, char** argv)
{
    std::vector<int> Ms = {1, 2, 3, 4, 5, 6, 7, 8}, BMs = {2, 3, 4, 5, 8};
    for (int M : {12, 16, 24, 32}) if (M <= G6_MX) Ms.push_back(M);
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
    printf("mtp_lastrow_test: %s sm_%d%d %d SMs, G6_NR=%d EB=%d G6_MX=%d, layers %d, sets %d, corr %.2f, K4 = experts %% %d == 0, shared width %d\n",
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
    unsigned* keeps = static_cast<unsigned*>(alloc(static_cast<size_t>(sets) * 4));    // per set: the S2 layout's last rows
    const size_t sb = bind::scratch_bytes();
    void* blk = alloc(sb);
    CK(cudaMemset(blk, 0, sb));
    const bind::Scratch sc = bind::scratch_at(blk);
    const size_t obytes = static_cast<size_t>(G6_MX) * HID * 2;
    uint8_t* out = static_cast<uint8_t*>(alloc(obytes));
    uint8_t* outs = static_cast<uint8_t*>(alloc(obytes * sets));      // per routing set (graph check)
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    bind::K1Ovl O[3];                                                // [variant]: 0 = off, 1 / 2 = G6_ROUTE_OVL mode
    for (int v = 1; v < 3; ++v)
    {
        O[v].mode = v;
        CK(cudaStreamCreateWithFlags(&O[v].side, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&O[v].fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&O[v].join, cudaEventDisableTiming));
    }
    CK(cudaDeviceSynchronize());

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
    // one layer call (rank.cuh launch_mlp_block): set s, layer l, ovl variant v; keep / keep_dev as k_router_plan takes them
    auto call = [&](const std::vector<Bound>& bs, int v, int M, int s, int l, unsigned keep, const unsigned* keep_dev, bool k1 = true) {
        CK(cudaMemcpyAsync(x2, x2sets + static_cast<size_t>(s) * G6_MX * HID, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
        bind::Xfrag xf;
        xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;
        bind::launch(xf, st);
        const bool ovl = k1 && bind::k1_ovl_on(bs[l].k1, O[v]);
        if (ovl) bind::k1_ovl_fork(bs[l].k1, O[v], st);
        bind::launch(bs[l].router, st);
        glue::k_router_plan<<<1, 256, 0, st>>>(logits, Ls[l].bias, M, dplan, keep, keep_dev);
        if (!k1) return;
        if (ovl) bind::launch_k1_ovl(bs[l].k1, O[v], st);
        else bind::launch_k1(bs[l].k1, st);
    };

    // ---------------------------------------------------------------- exactness
    if (check)
    {
        std::mt19937 g(7);
        std::vector<unsigned> hkeep(sets);
        for (int M : Ms)
            for (int fuse : {1, 0})
            {
                std::vector<Bound> bs;
                if (const char* e = make(M, fuse, bs)) { printf("FAIL M=%d fuse=%d make: %s\n", M, fuse, e); ++g_fail; continue; }
                for (int s = 0; s < sets; ++s) hkeep[s] = layout_mask(M, g);
                CK(cudaMemcpy(keeps, hkeep.data(), sets * 4, cudaMemcpyHostToDevice));
                const unsigned last = 1u << (M - 1), all = M == 32 ? 0xffffffffu : (1u << M) - 1u;
                const size_t ob = static_cast<size_t>(M) * HID * 2, rb = HID * 2;
                std::vector<uint8_t> ref(ob * sets), got(ob * sets);
                // mode: 0 full (keep 0), 1 keep = all rows, 2 last row, 3 S2 layout (keep_dev)
                auto kp = [&](int mode, int s) { return mode == 1 ? all : mode == 2 ? last : 0u; };
                auto kd = [&](int mode, int s) { return mode == 3 ? keeps + s : nullptr; };
                auto plain = [&](int v, int mode, std::vector<uint8_t>& h) {
                    for (int s = 0; s < sets; ++s)
                    {
                        CK(cudaMemsetAsync(out, 0xff, obytes, st));
                        call(bs, v, M, s, s % nlay, kp(mode, s), kd(mode, s));
                        CK(cudaMemcpyAsync(h.data() + ob * s, out, ob, cudaMemcpyDeviceToHost, st));
                    }
                    CK(cudaStreamSynchronize(st));
                    CK(cudaGetLastError());
                };
                auto graph = [&](int v, int mode, std::vector<uint8_t>& h) {
                    cudaGraph_t gr;
                    cudaGraphExec_t ge;
                    CK(cudaMemsetAsync(outs, 0xff, obytes * sets, st));
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));   // as g6m4
                    for (int s = 0; s < sets; ++s)
                    {
                        call(bs, v, M, s, s % nlay, kp(mode, s), kd(mode, s));
                        CK(cudaMemcpyAsync(outs + ob * s, out, ob, cudaMemcpyDeviceToDevice, st));
                    }
                    CK(cudaStreamEndCapture(st, &gr));
                    CK(cudaGraphInstantiate(&ge, gr, 0));
                    for (int r = 0; r < 2; ++r) CK(cudaGraphLaunch(ge, st));    // twice: the second replay is compared
                    CK(cudaMemcpyAsync(h.data(), outs, ob * sets, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    CK(cudaGetLastError());
                    CK(cudaGraphExecDestroy(ge));
                    CK(cudaGraphDestroy(gr));
                };
                // the rows of mask b in every set equal ref's; the other rows finite (BF16)
                auto cmp = [&](const char* what, int v, int mode, const std::vector<uint8_t>& b) {
                    size_t nbad = 0, nonfin = 0, rows = 0;
                    long first = -1;
                    for (int s = 0; s < sets; ++s)
                    {
                        const unsigned want = mode == 0 || mode == 1 ? all : mode == 2 ? last : hkeep[s];
                        for (int m = 0; m < M; ++m)
                        {
                            const uint8_t* x = b.data() + ob * s + rb * m;
                            if (want >> m & 1u)
                            {
                                ++rows;
                                if (memcmp(x, ref.data() + ob * s + rb * m, rb)) { if (first < 0) first = s * 64 + m; ++nbad; }
                            }
                            else
                                for (int i = 0; i < HID; ++i) nonfin += ((x[2 * i] | (x[2 * i + 1] << 8)) & 0x7f80) == 0x7f80;
                        }
                    }
                    const bool ok = nbad == 0 && nonfin == 0;
                    printf("%s M=%d fuse=%d(%s) ovl=%d %s: %zu kept rows over %d sets%s", ok ? "PASS" : "FAIL", M, fuse,
                           bs[0].k1.fuse ? "fused" : "separate", v, what, rows, sets, ok ? "\n" : "");
                    if (!ok) printf(", %zu rows differ (first set %ld row %ld), %zu non-finite values in the other rows\n", nbad,
                                    first < 0 ? -1 : first / 64, first < 0 ? -1 : first % 64, nonfin);
                    (ok ? g_pass : g_fail)++;
                };
                plain(0, 0, ref);
                {   // sanity of the reference: finite, not all zero
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
                {   // the plan: kept rows as the full plan's, the others empty (last-row and S2 masks, set 0..sets-1, layer 0)
                    size_t bad = 0;
                    for (int s = 0; s < sets; ++s)
                        for (int mode : {2, 3})
                        {
                            glue::RoutePlan F, P;
                            call(bs, 0, M, s, 0, 0u, nullptr, false);
                            CK(cudaMemcpyAsync(&F, dplan, sizeof F, cudaMemcpyDeviceToHost, st));
                            call(bs, 0, M, s, 0, kp(mode, s), kd(mode, s), false);
                            CK(cudaMemcpyAsync(&P, dplan, sizeof P, cudaMemcpyDeviceToHost, st));
                            CK(cudaStreamSynchronize(st));
                            const unsigned want = mode == 2 ? last : hkeep[s];
                            for (int m = 0; m < M; ++m)
                                for (int k = 0; k < 8; ++k)
                                {
                                    if (want >> m & 1u)
                                    {
                                        const int fs = F.tok_slot[m][k], ps = P.tok_slot[m][k];
                                        bad += ps < 0 || P.slot_expert[ps] != F.slot_expert[fs] || P.tok_expert[m][k] != F.tok_expert[m][k] ||
                                               memcmp(&P.tok_w[m][k], &F.tok_w[m][k], 4) || P.slot_tok[ps][P.tok_col[m][k]] != m ||
                                               (M <= 8 && P.tok_col[m][k] != F.tok_col[m][k]);
                                    }
                                    else bad += P.tok_slot[m][k] != -1 || P.tok_expert[m][k] != -1;
                                }
                        }
                    printf("%s M=%d fuse=%d plan: kept rows as the full plan, the others without routes (%zu entries differ)\n", bad ? "FAIL" : "PASS",
                           M, fuse, bad);
                    (bad ? g_fail : g_pass)++;
                }
                plain(0, 0, got);  cmp("full run twice (plain)", 0, 0, got);
                plain(0, 1, got);  cmp("keep = all rows == full (plain)", 0, 1, got);
                for (int v = 0; v < 3; ++v)
                {
                    if (v && !bind::k1_ovl_on(bs[0].k1, O[v])) continue;   // the fork applies only to separate launches
                    plain(v, 2, got);  cmp("last row (plain)", v, 2, got);
                    plain(v, 3, got);  cmp("S2 layout's last rows, keep_dev (plain)", v, 3, got);
                    graph(v, 2, got);  cmp("last row (graph)", v, 2, got);
                    graph(v, 3, got);  cmp("S2 layout's last rows, keep_dev (graph)", v, 3, got);
                }
                graph(0, 0, got);  cmp("full (graph) == full (plain)", 0, 0, got);
            }
    }

    // ---------------------------------------------------------------- bench
    if (bench)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        printf("bench: us per MoE layer call = graph of reps x [x2 copy, xfrag, router GEMV, plan, K1] / reps, median of 5\n");
        for (int M : BMs)
        {
            std::vector<Bound> bs, b1;
            if (const char* e = make(M, 1, bs)) { printf("M=%d make: %s\n", M, e); continue; }
            if (const char* e = make(1, 1, b1)) { printf("M=1 make: %s\n", e); continue; }
            double nex[2] = {0, 0}, bytes[2] = {0, 0};      // full, last row
            for (int s = 0; s < sets; ++s)
                for (int k = 0; k < 2; ++k)
                {
                    call(bs, 0, M, s, 0, k ? 1u << (M - 1) : 0u, nullptr, false);
                    glue::RoutePlan P;
                    CK(cudaMemcpyAsync(&P, dplan, sizeof P, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    std::vector<int> ex(P.slot_expert, P.slot_expert + P.n_slots);
                    ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
                    nex[k] += ex.size();
                    bytes[k] += Ls[0].ebytes[256];
                    for (int e : ex) bytes[k] += Ls[0].ebytes[e];
                }
            for (int k = 0; k < 2; ++k) { nex[k] /= sets; bytes[k] /= sets; }
            const int NV = 3;
            const char* name[NV] = {"full", "last row", "M=1 step"};
            auto capture = [&](int r) {
                cudaGraph_t gr;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < reps; ++i)
                {
                    const int s = i % sets, l = (i / sets + i) % nlay;
                    if (r == 2) call(b1, 0, 1, s, l, 0u, nullptr);
                    else call(bs, 0, M, s, l, r ? 1u << (M - 1) : 0u, nullptr);
                }
                CK(cudaStreamEndCapture(st, &gr));
                CK(cudaGraphInstantiate(&ge, gr, 0));
                CK(cudaGraphDestroy(gr));
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
            cudaGraphExec_t ge[NV];
            for (int r = 0; r < NV; ++r) ge[r] = capture(r);
            std::vector<double> t[NV];
            for (int rep = 0; rep < 5; ++rep)
                for (int r = 0; r < NV; ++r) t[r].push_back(time(ge[r]));
            double med[NV];
            for (int r = 0; r < NV; ++r)
            {
                std::sort(t[r].begin(), t[r].end());
                med[r] = t[r][2];
                CK(cudaGraphExecDestroy(ge[r]));
            }
            CK(cudaGetLastError());
            for (int r = 0; r < NV; ++r)
                printf("bench M=%d %-9s form=%s experts=%.1f MB=%.2f us/call=%7.1f (min %7.1f max %7.1f)  vs full %+6.1f us (%+5.1f%%)\n", M, name[r],
                       bs[0].k1.fuse ? "fused" : "separate", r == 2 ? 8.0 : nex[r], (r == 2 ? bytes[1] : bytes[r]) / 1e6, med[r], t[r][0], t[r][4],
                       med[r] - med[0], 100 * (med[r] - med[0]) / med[0]);
            fflush(stdout);
        }
    }
    for (void* p : mem) cudaFree(p);
    if (check) printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
