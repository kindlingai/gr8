// GR8_L2PF microbenchmark + exactness check, one GPU: NL distinct layers of the decode step's per-layer sequence at the TP
// rank shapes, in one CUDA graph, with the engine's prefetch calls (l2pf::arm / window / join, kernels/l2pf.cuh) at the
// engine's places (rank.cuh launch_consume, launch_layer_in_qs, launch_attn_block):
//   [W_M] wait stand-in (the MLP all-reduce of layer l-1)          the rank waits, DRAM idle
//   x_in: the rank's q_a + kv_a (+ idx_k on full layers) tiles, MXFP8 / BF16 gv_multi as under GR8_QA_SPLIT=1
//   [W_Q] wait stand-in (the q_a gather)
//   q_b: dense trellis, K 2048 x N H*256
//   [W_T] the real decode attention (g6_attn_replay_run, S2 batched cell; reads UK / UV / KV / index keys)
//   o_proj: dense trellis, K H*256 x N 6144
//   [W_A] wait stand-in (the attention all-reduce)
//   router (BF16 gv, 256 x 6144) on the stream, || a stand-in for the trellis shared expert's fc1 on a forked stream (a
//   dense trellis K 6144 x N 1024, about the bytes of the rank's gate + up), as GR8_ROUTE_OVL=1
//   MoE stand-in: --moe MB of streaming nc loads at the DRAM rate (the routed experts; evicts L2), then the join
// A wait stand-in is one warp that idles for its time, with the peer's tail landing --arr us before its end (the
// window's stop sees the collective arrive there, as in the engine). Every layer's weights are distinct (each layer
// starts cold, as after the routed experts). The attention runs the serving flags (GR8_ATTN_SC2=4 ... GR8_ATTN_FRONT=1),
// which the environment must set (the bench stops and names them otherwise); --stock-attn 1 runs the environment's
// attention flags as they are.
//   EXACT  per arm: every layer's x_in, q_b, attention (output + indices), o_proj, router and shared outputs after one
//          graph launch, against off, byte for byte (exit 3 otherwise)
//   TIME   per arm: us per layer (graph time / NL, median of TR alternating trials), against off, x 78 in ms per step;
//          and per segment (one stamp kernel between segments, a separate graph): wm x_in wq q_b attn o_proj wa
//          router moe, us per layer
// Arms: --arms "win[:tmask[:gbps[:opmb[:wave]]]],..." (GR8_L2PF windows 1 W_M 2 W_Q 4 W_T 8 W_A, GR8_L2PF_T targets
// 1 x_in 2 idx_q 4 q_b 8 UK 16 UV 32 o_proj 64 router 128 shared; empty field = the --gbps / --opmb / --wave default),
// budgets --us wm,wq,wt,wa and --mb MiB per window for all arms.
// build (scripts/build.sh builds it): from the repo root, with the engine's K1H2 flags, -DG6_NR=4 -DDEC_S2 -DPF_F12:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $FLAGS -I engine/kernels -o l2pf_bench \
//        engine/kernels/tests/l2pf_bench.cu engine/attn/attn.cu engine/attn/bind.cu -lcublas
// run: l2pf_bench [--M 3,12] [--T 2048] [--L 1000] [--layers 78] [--arms ...] [--gbps 175] [--wave 8] [--us 30,30,100,30]
//                 [--mb 10] [--opmb 8] [--wm 17] [--wq 20] [--wa 17] [--arr 2] [--moe 80] [--reps 5] [--tr 7] [--seg 1]
//   e.g. ./l2pf_bench; ./l2pf_bench --T 32768 --L 32000; ./l2pf_bench --M 3 --arms 4,4:32,4:32:125,15
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include "../l2pf.cuh"
#include "../../knobs-rs/gr8_knobs.h"
#include "../../attn/replay.h"
#include "../../attn/ops.h"
#include "../../attn/kvfmt.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CKI(x) do { int e_ = (x); if (e_) { fprintf(stderr, "%s:%d status %d (%s)\n", __FILE__, __LINE__, e_, cudaGetErrorString((cudaError_t)e_)); exit(1); } } while (0)
static void die(const std::string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }
constexpr int TOP = 2048;             // attn.cu ga::TOP
constexpr int QA_T = 2048 / 16 / 4, KVA_T = 576 / 16 / 4, IDXK_T = 128 / 16 / 4;   // the rank's tiles under GR8_QA_SPLIT (TP4)
constexpr int SH_N = 1024;            // shared expert fc1 stand-in outputs (gate + up, 512 each at TP4)

__device__ uint32_t hsh(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return (uint32_t)x; }
__device__ float unif(uint64_t seed, uint64_t i) { return (float)(hsh(seed * 0x9e3779b97f4a7c15ull + i) >> 8) * (2.f / 16777216.f) - 1.f; }   // [-1, 1)
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, uint64_t seed, float s, float c)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(c + s * unif(seed, i));
}
__global__ void fill_half(__half* p, size_t n, uint64_t seed, float s, float c)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half(c + s * unif(seed, i));
}
__global__ void fill_u32(uint32_t* p, size_t n, uint64_t seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = hsh(seed * 0x100000001B3ull + i);
}
__device__ uint8_t fin8(uint32_t v) { uint8_t b = (uint8_t)v; if ((b & 0x7f) == 0x7f) b ^= 1; return b; }   // finite e4m3 code
// MXFP8 gv records [O/16][NG][528]: finite e4m3 codes, E8M0 scales near 2^-7 (the last 16 bytes of a record)
__global__ void fill_mx(uint8_t* p, size_t recs, uint64_t seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < recs * 528; i += (size_t)gridDim.x * blockDim.x)
        p[i] = i % 528 < 512 ? fin8(hsh(seed + i)) : (uint8_t)(120 + hsh(seed + 7 * i) % 4);
}
// latent rows (kvfmt.h) with finite codes, positive scales and small rope
__global__ void fill_kv(uint8_t* c, size_t rows, uint64_t seed)
{
    for (size_t r = blockIdx.x; r < rows; r += gridDim.x)
    {
        uint8_t* v = c + r * g6kv::ROW;
        const int t = threadIdx.x;
#if G6_KV4
        if (t < 256) v[t] = (uint8_t)hsh(seed + r * 1000 + t);
        if (t < 16) ((__half*)(v + g6kv::SCALE_OFF))[t] = __float2half(0.02f * (1.5f + unif(seed + 7, r * 16 + t)));
#else
        for (int d = t; d < 512; d += blockDim.x) v[d] = fin8(hsh(seed + r * 1000 + d));
        if (t < 4) ((float*)(v + g6kv::SCALE_OFF))[t] = 0.004f * (1.5f + unif(seed + 7, r * 4 + t));
#endif
        if (t < 64) ((__nv_bfloat16*)(v + g6kv::ROPE_OFF))[t] = __float2bfloat16(0.5f * unif(seed + 9, r * 64 + t));
    }
}
__global__ void fill_idx(uint8_t* c, size_t pages, int ps, uint64_t seed)
{
    for (size_t p = blockIdx.x; p < pages; p += gridDim.x)
    {
        uint8_t* b = c + p * ps * g6kv::IDX_ROW;
        for (int i = threadIdx.x; i < ps * 128; i += blockDim.x) b[i] = fin8(hsh(seed + p * 100000 + i));
        for (int i = threadIdx.x; i < ps; i += blockDim.x) ((float*)(b + ps * 128))[i] = ldexpf(1.f, -6 - (int)(hsh(seed + 3 + p * 64 + i) % 3));
    }
}
// the routed experts' phase: n x 16 B read once with the GEMVs' streaming loads (DRAM rate, evicts L2)
__global__ void k_moe(const uint4* p, size_t n, float* sink)
{
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) *sink = 1.f;   // never (the buffer is zero); keeps the loads
}
// a collective wait: one warp idles ns; the peer's tail of collective c lands arr ns before the end
__global__ void __launch_bounds__(32) k_wait_standin(g6::FabricDev f, const u64* stepp, int c, unsigned ns, unsigned arr)
{
    if (threadIdx.x) return;
    const u64 step = *stepp, t0 = l2pf::gtimer();
    while (l2pf::gtimer() < t0 + (ns > arr ? ns - arr : 0)) {}
    glue::st_release_sys(reinterpret_cast<u64*>(const_cast<char*>(glue::peer_slot(f, step, c, 1 - f.rank))), g6::coll_seq(step, c, f.ncoll));
    while (l2pf::gtimer() < t0 + ns) {}
}
__global__ void k_bump(u64* stepp) { *stepp += 1; }
__global__ void k_stamp(u64* d) { *d = l2pf::gtimer(); }

template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> static std::vector<T> host(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
static size_t ndiff(const void* x, const void* y, size_t bytes) { size_t k = 0; for (size_t i = 0; i < bytes; ++i) k += ((const uint8_t*)x)[i] != ((const uint8_t*)y)[i]; return k; }
static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
template <class F> static cudaGraphExec_t inst(cudaStream_t st, F body)
{
    cudaGraph_t g;
    cudaGraphExec_t x;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphDestroy(g));
    return x;
}
static float run_us(cudaGraphExec_t x, cudaStream_t st, int reps)   // us per graph launch
{
    for (int i = 0; i < 2; ++i) CK(cudaGraphLaunch(x, st));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(x, st));
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.f / reps;
}
struct TV { const void* p = nullptr; int nd = 0; long long shape[4] = {}; };
static TV tv(const void* p, std::initializer_list<long long> s) { TV t; t.p = p; for (long long v : s) t.shape[t.nd++] = v; return t; }
static bind::Tensor bt(const void* p, std::initializer_list<long long> s) { bind::Tensor t; t.p = p; for (long long v : s) t.shape[t.nd++] = v; return t; }
static std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> v;
    for (size_t p = 0;;) { size_t q = s.find(d, p); v.push_back(s.substr(p, q == std::string::npos ? q : q - p)); if (q == std::string::npos) break; p = q + 1; }
    return v;
}
static std::vector<int> ints(const std::string& s) { std::vector<int> v; for (auto& x : split(s, ',')) v.push_back(atoi(x.c_str())); return v; }

enum { S_WM, S_XIN, S_WQ, S_QB, S_ATTN, S_OPROJ, S_WA, S_ROUTER, S_MOE, NSEG };
static const char* const SEG[NSEG] = {"wm", "x_in", "wq", "q_b", "attn", "o_proj", "wa", "router", "moe"};

int main(int argc, char** argv)
{
    std::vector<int> Ms = {3, 12};
    std::string arms_s = "15,15:0xf9,15:0x39,15::125,15::250,15:::0,1,2,4,8,4:32,4:48";
    int T = 2048, L = 1000, NL = 78, REPS = 5, TR = 7, S2 = 1, SEGT = 1, STOCK = 0, wave = 8;
    float gbps = 175, mb = 10, opmb = 8, wm = 17, wq = 20, wa = 17, arr = 2, moe = 80;
    float us[l2pf::NW] = {30, 30, 100, 30};
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { if (i + 1 >= argc) die("missing value after " + a); return argv[++i]; };
        if (a == "--M") Ms = ints(nx());
        else if (a == "--T") T = atoi(nx());
        else if (a == "--L") L = atoi(nx());
        else if (a == "--layers") NL = atoi(nx());
        else if (a == "--arms") arms_s = nx();
        else if (a == "--gbps") gbps = atof(nx());
        else if (a == "--wave") wave = atoi(nx());
        else if (a == "--us") { const char* e = nx(); if (sscanf(e, "%f,%f,%f,%f", &us[0], &us[1], &us[2], &us[3]) != 4) die("--us wm,wq,wt,wa"); }
        else if (a == "--mb") mb = atof(nx());
        else if (a == "--opmb") opmb = atof(nx());
        else if (a == "--wm") wm = atof(nx());
        else if (a == "--wq") wq = atof(nx());
        else if (a == "--wa") wa = atof(nx());
        else if (a == "--arr") arr = atof(nx());
        else if (a == "--moe") moe = atof(nx());
        else if (a == "--reps") REPS = atoi(nx());
        else if (a == "--tr") TR = atoi(nx());
        else if (a == "--seg") SEGT = atoi(nx());
        else if (a == "--s2") S2 = atoi(nx());
        else if (a == "--stock-attn") STOCK = atoi(nx());
        else die("usage: l2pf_bench [--M 3,12] [--T 2048] [--L 1000] [--layers 78] [--arms win[:tmask[:gbps[:opmb[:wave]]]],...] [--gbps 175] "
                 "[--wave 8] [--us 30,30,100,30] [--mb 10] [--opmb 8] [--wm 17] [--wq 20] [--wa 17] [--arr 2] [--moe 80] [--reps 5] [--tr 7] "
                 "[--seg 1] [--s2 1] [--stock-attn 0]");
    }
    // the arms: off first
    std::vector<l2pf::Cfg> arms(1);
    std::vector<std::string> names = {"off"};
    for (const std::string& s : split(arms_s, ','))
    {
        if (s.empty()) continue;
        const std::vector<std::string> f = split(s, ':');
        l2pf::Cfg c;
        c.win = atoi(f[0].c_str()) & 15;
        if (!c.win) die("--arms: window mask 1..15");
        c.tmask = f.size() > 1 && !f[1].empty() ? (unsigned)strtoul(f[1].c_str(), nullptr, 0) : (1u << l2pf::NTG) - 1;
        c.gbps = f.size() > 2 && !f[2].empty() ? atof(f[2].c_str()) : gbps;
        c.opmb = f.size() > 3 && !f[3].empty() ? atof(f[3].c_str()) : opmb;
        c.wave = f.size() > 4 && !f[4].empty() ? atoi(f[4].c_str()) : wave;
        c.mb = mb;
        for (int w = 0; w < l2pf::NW; ++w) c.us[w] = us[w];
        arms.push_back(c);
        names.push_back(s);
    }
    int Mmax = 1;
    for (int M : Ms) { if (M < 1 || M > G6_MX || M > L) die("--M: rows must be 1..G6_MX and <= L"); Mmax = std::max(Mmax, M); }
    if (L > T || NL < 2) die("need L <= T and at least 2 layers");
#ifndef DEC_S2
    if (S2) die("--s2 1 needs -DDEC_S2");
#endif
    if (!STOCK && !(knob::attn_sc2() == 4 && knob::attn_sc2_tr() == 2 && knob::attn_ctx2() == 4 && knob::attn_ctx3() == 2 &&
                    knob::attn_sel2() == 16 && knob::idx_grp() == 16 && knob::attn_qt() && knob::attn_front()))
        die("the serving attention flags: GR8_ATTN_SC2=4 GR8_ATTN_SC2_TR=2 GR8_ATTN_CTX2=4 GR8_ATTN_CTX3=2 GR8_ATTN_SEL2=16 "
            "GR8_IDX_GRP=16 GR8_ATTN_QT=1 GR8_ATTN_FRONT=1 in the environment, or --stock-attn 1 for its own");
    if (knob::l2pf()) fprintf(stderr, "note: GR8_L2PF is set; the bench sets the arms itself\n");
    CK(bind::init());
    const int H = g6nr::HMAX, K = g6nr::QBW, N = HID, pages = (T + 63) / 64;
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    int dev = 0, sms = 0, clk = 0, l2 = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    printf("GPU %d SMs, clock %d MHz, L2 %.2f MB | TP%d: H %d, q_b 2048 x %d, o_proj %d x %d | T=%d L=%d layers %d s2=%d | stand-ins: "
           "wm %.0f wq %.0f wa %.0f us (arrival %.1f us before the end), moe %.0f MB | budgets %.0f/%.0f/%.0f/%.0f us, <= %.1f MiB a window | "
           "attention %s | reps %d TR %d\n", sms, clk / 1000, l2 / 1e6, g6nr::NR, H, K, K, N, T, L, NL, S2, wm, wq, wa, arr, moe, us[0], us[1],
           us[2], us[3], mb, STOCK ? "the environment's flags" : "serving flags", REPS, TR);
    CKI(g6_attn_rope_init(T, st));

    // ---- layers: full-index ones spread as 22 of 78 (the others reuse the last full layer's indices)
    std::vector<int> full(NL), src(NL);
    int nfull = 0;
    for (int l = 0; l < NL; ++l)
    {
        full[l] = l == 0 || (l * 22) / 78 != ((l - 1) * 22) / 78;
        src[l] = full[l] ? l : src[l - 1];
        nfull += full[l];
    }
    __nv_bfloat16 *kvn = dalloc<__nv_bfloat16>(512), *inw = dalloc<__nv_bfloat16>(128), *inb = dalloc<__nv_bfloat16>(128);
    fill_bf16<<<2, 256, 0, st>>>(kvn, 512, 3, 0.2f, 1); fill_bf16<<<1, 128, 0, st>>>(inw, 128, 4, 0.2f, 1); fill_bf16<<<1, 128, 0, st>>>(inb, 128, 5, 0.1f, 0);
    __half *qb = dalloc<__half>((size_t)Mmax * H * 256), *kva = dalloc<__half>((size_t)Mmax * 576), *iq = dalloc<__half>((size_t)Mmax * 32 * 128),
           *ik = dalloc<__half>((size_t)Mmax * 128), *iw = dalloc<__half>((size_t)Mmax * 32);
    fill_half<<<64, 256, 0, st>>>(qb, (size_t)Mmax * H * 256, 6, 1.f, 0); fill_half<<<8, 256, 0, st>>>(kva, (size_t)Mmax * 576, 7, 1.f, 0);
    fill_half<<<64, 256, 0, st>>>(iq, (size_t)Mmax * 32 * 128, 8, 1.f, 0); fill_half<<<1, 256, 0, st>>>(ik, (size_t)Mmax * 128, 9, 1.f, 0);
    fill_half<<<1, 256, 0, st>>>(iw, (size_t)Mmax * 32, 10, 0.1f, 0);
    // the GEMVs' inputs: x (fragment tables, fp16 and BF16), q_a_n rows (q_b), x2 rows (shared stand-in)
    const size_t xft = (size_t)g6mx::tiles(Mmax) * (HID / 16) * 64;
    u32 *xf = dalloc<u32>(xft), *xb = dalloc<u32>(xft);
    fill_half<<<64, 256, 0, st>>>(reinterpret_cast<__half*>(xf), xft * 2, 11, 1.f, 0);
    fill_bf16<<<64, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(xb), xft * 2, 12, 1.f, 0);
    __half *qan = dalloc<__half>((size_t)Mmax * 2048), *x2 = dalloc<__half>((size_t)Mmax * HID);
    fill_half<<<64, 256, 0, st>>>(qan, (size_t)Mmax * 2048, 13, 1.f, 0); fill_half<<<64, 256, 0, st>>>(x2, (size_t)Mmax * HID, 14, 1.f, 0);
    int *dpos = dalloc<int>(Mmax), *dtab = dalloc<int>(pages);
    {
        std::vector<int> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = i;
        CK(cudaMemcpy(dtab, table.data(), pages * 4, cudaMemcpyHostToDevice));
    }
    const size_t ukn = (size_t)H * 512 * 192, uvn = (size_t)H * 256 * 512, opb = (size_t)K * N * dtr::BITS / 8, qbb = (size_t)2048 * K * dtr::BITS / 8,
                 shb = (size_t)HID * SH_N * dtr::BITS / 8;
    struct Lyr
    {
        __nv_bfloat16 *uk, *uv;
        uint8_t *mla, *ixc = nullptr, *wo, *wq, *ws, *gqa, *gkva, *gik = nullptr, *grt;
        __half *suo, *svo, *suq, *svq, *sus, *svs, *att, *qay, *kvay, *iky, *qby, *shy;
        float* rty;
        int* ind;
        unsigned* err;
        uint16_t* y;
        G6AttnStepCell* cell;
        G6AttnReplay rp;
        dtr::Mat mo, mq, ms;
        bind::Layer B;                // q_a, kv_a, idx_k, x_in, q_b_t, o_proj_t, router, shw (the plan's view)
        dtr::Gemv sh;                 // the shared stand-in
    };
    std::vector<Lyr> Y(NL);
    for (int l = 0; l < NL; ++l)
    {
        Lyr& y = Y[l];
        y.uk = dalloc<__nv_bfloat16>(ukn); y.uv = dalloc<__nv_bfloat16>(uvn);
        fill_bf16<<<512, 256, 0, st>>>(y.uk, ukn, 1000 + l, 0.05f, 0); fill_bf16<<<512, 256, 0, st>>>(y.uv, uvn, 2000 + l, 0.05f, 0);
        y.mla = dalloc<uint8_t>((size_t)pages * 64 * g6kv::ROW);
        fill_kv<<<1024, 256, 0, st>>>(y.mla, (size_t)pages * 64, 3000 + l);
        if (full[l]) { y.ixc = dalloc<uint8_t>((size_t)pages * 64 * g6kv::IDX_ROW); fill_idx<<<512, 256, 0, st>>>(y.ixc, pages, 64, 4000 + l); }
        y.ind = dalloc<int>((size_t)Mmax * TOP);
        y.err = dalloc<unsigned>(1);
        y.att = dalloc<__half>((size_t)Mmax * K);
        y.y = dalloc<uint16_t>((size_t)Mmax * N);
        CK(cudaMalloc(&y.cell, sizeof(G6AttnStepCell)));
        // o_proj (side 2), q_b and the shared stand-in (side 1): K4 entries + suh / svh
        auto mat = [&](uint8_t*& w, __half*& su, __half*& sv, size_t bytes, int k, int n, int side, uint64_t seed, dtr::Mat& m) {
            w = dalloc<uint8_t>(bytes);
            fill_u32<<<1024, 256, 0, st>>>(reinterpret_cast<uint32_t*>(w), bytes / 4, seed);
            su = dalloc<__half>(k); sv = dalloc<__half>(n);
            fill_half<<<16, 256, 0, st>>>(su, k, seed + 1, 0.5f, 1.f); fill_half<<<16, 256, 0, st>>>(sv, n, seed + 2, 0.01f, 0.f);
            const int E = (side == dtr::SIDE_OUT ? n : k) / 128;
            const long long eb = (long long)(side == dtr::SIDE_OUT ? k : n) / 16 * 8 * 32 * dtr::BITS;
            if (const char* e = dtr::make_mat(tv(w, {E, eb}), tv(su, {k}), tv(sv, {n}), side, m)) die(e);
        };
        mat(y.wo, y.suo, y.svo, opb, K, N, dtr::SIDE_IN, 5000 + 10 * l, y.mo);
        mat(y.wq, y.suq, y.svq, qbb, 2048, K, dtr::SIDE_OUT, 900000 + 10 * l, y.mq);
        mat(y.ws, y.sus, y.svs, shb, HID, SH_N, dtr::SIDE_OUT, 1900000 + 10 * l, y.ms);
        // gv records: the rank's q_a / kv_a tiles (MXFP8), idx_k tiles and the router (BF16)
        y.gqa = dalloc<uint8_t>((size_t)QA_T * 192 * 528); fill_mx<<<512, 256, 0, st>>>(y.gqa, (size_t)QA_T * 192, 7000000 + l);
        y.gkva = dalloc<uint8_t>((size_t)KVA_T * 192 * 528); fill_mx<<<512, 256, 0, st>>>(y.gkva, (size_t)KVA_T * 192, 8000000 + l);
        if (full[l]) { y.gik = dalloc<uint8_t>((size_t)IDXK_T * 384 * 512); fill_bf16<<<256, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(y.gik), (size_t)IDXK_T * 384 * 256, 9000000 + l, 0.05f, 0); }
        y.grt = dalloc<uint8_t>((size_t)16 * 384 * 512); fill_bf16<<<512, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(y.grt), (size_t)16 * 384 * 256, 9500000 + l, 0.05f, 0);
        y.qay = dalloc<__half>((size_t)Mmax * QA_T * 16); y.kvay = dalloc<__half>((size_t)Mmax * KVA_T * 16); y.iky = dalloc<__half>((size_t)Mmax * IDXK_T * 16);
        y.qby = dalloc<__half>((size_t)Mmax * K); y.shy = dalloc<__half>((size_t)Mmax * SH_N); y.rty = dalloc<float>((size_t)Mmax * 256);
    }
#ifdef DEC_S2
    G6AttnRow* drows = nullptr;
    if (S2)
    {
        std::vector<G6AttnRow> rr(Mmax);
        for (auto& x : rr) { x.table = dtab; x.context = L; x.page_count = pages; x.table_size = pages; x.ibase = 0; x.sess = 0; }
        drows = dalloc<G6AttnRow>(Mmax);
        CK(cudaMemcpy(drows, rr.data(), Mmax * sizeof(G6AttnRow), cudaMemcpyHostToDevice));
    }
#endif
    dtr::Scratch sc;
    sc.part_n = dtr::part_floats(N, 1, 64, Mmax);
    sc.cnt_n = 256;
    sc.part = dalloc<float>(sc.part_n);
    sc.cnt = dalloc<int>(sc.cnt_n);
    // the stand-ins' fabric: rank 0 of 2, three collectives a layer (W_M, W_Q, W_A), payload 0 (the tail at the slot start)
    g6::FabricDev fab{};
    fab.rank = 0; fab.nr = 2; fab.ncoll = 3 * NL; fab.stride = 256;
    fab.send = dalloc<char>((size_t)2 * fab.ncoll * fab.stride);
    fab.recv = dalloc<char>((size_t)2 * fab.ncoll * fab.nr * fab.stride);
    fab.doorbell = dalloc<unsigned long long>(fab.ncoll);
    fab.error = dalloc<unsigned>(1);
    u64* stepp = dalloc<u64>(1);
    const int NMOE = 4;
    const size_t moen = (size_t)(moe * 1e6) / 16;
    std::vector<uint4*> mbuf(NMOE);
    for (auto& b : mbuf) b = moen ? dalloc<uint4>(moen) : nullptr;
    float* sink = dalloc<float>(1);
    u64* stamps = dalloc<u64>((size_t)NL * (NSEG + 1));
    cudaStream_t ovl;
    cudaEvent_t ef, ej;
    CK(cudaStreamCreateWithFlags(&ovl, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ef, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ej, cudaEventDisableTiming));
    void* scr = nullptr;
    size_t sbmax = 0;
    CK(cudaStreamSynchronize(st));

    for (const int M : Ms)
    {
        {
            std::vector<int> pos(M);
            for (int i = 0; i < M; ++i) pos[i] = L - M + i;
            CK(cudaMemcpy(dpos, pos.data(), M * 4, cudaMemcpyHostToDevice));
        }
        size_t sb = 0;
        for (int l = 0; l < NL; ++l)
        {
            Lyr& y = Y[l];
            G6Attn x{};
            x.rows = M; x.heads = H; x.context = T; x.page_size = 64; x.page_count = x.table_size = pages; x.query_tile = std::max(M, 32);
            x.q_b = qb; x.kv_a = kva; x.kv_norm = kvn; x.uk = y.uk; x.uv = y.uv; x.positions = dpos; x.block_table = dtab; x.mla_cache = y.mla;
            x.indices = y.ind; x.output = y.att; x.error = y.err;
#ifdef DEC_S2
            x.s2_batched = S2;
#endif
            if (full[l]) { x.index_q = iq; x.index_k = ik; x.index_weight = iw; x.index_norm = inw; x.index_bias = inb; x.index_cache = y.ixc; }
            else { x.reuse_indices = Y[src[l]].ind; x.reuse_rows = M; }
            G6AttnStepCell c{};
            c.context = L; c.page_count = pages; c.table_size = pages; c.reuse_rows = full[l] ? 0 : M;
#ifdef DEC_S2
            c.rows = drows;
#endif
            CK(cudaMemcpy(y.cell, &c, sizeof c, cudaMemcpyHostToDevice));
            y.rp = G6AttnReplay{x, y.cell, nullptr, 0};
            sb = std::max(sb, g6_attn_scratch_bytes(&x));
            bind::Layer& B = y.B;
            B = bind::Layer();
            B.M = M; B.heads = H; B.indexer = full[l];
            const char* e = nullptr;
            if ((e = bind::make_gv(bt(y.gqa, {QA_T, 192, 528}), bind::MXFP8, 2, 16, M, xf, y.qay, gv::O_F16, B.q_a))) die(e);
            if ((e = bind::make_gv(bt(y.gkva, {KVA_T, 192, 528}), bind::MXFP8, 2, 16, M, xf, y.kvay, gv::O_F16, B.kv_a))) die(e);
            if (full[l] && (e = bind::make_gv(bt(y.gik, {IDXK_T, 384, 512}), bind::BF16W, 1, 32, M, xb, y.iky, gv::O_F16, B.idx_k))) die(e);
            if ((e = bind::make_gv(bt(y.grt, {16, 384, 512}), bind::BF16W, 2, 16, M, xb, y.rty, gv::O_F32, B.router))) die(e);
            B.q_a.tpc = B.kv_a.tpc = B.idx_k.tpc = 1;          // qs_narrow's GR8_QA_SPLIT_TPC default
            bind::make_gv_multi({&B.q_a, &B.kv_a, &B.idx_k}, M, B.x_in);
            if ((e = dtr::make_out(y.mq, M, qan, 2048, nullptr, y.qby, K, dtr::O_F16, sc, 0, 0, 1, B.q_b_t))) die(e);
            if ((e = dtr::make_out(y.mo, M, y.att, K, nullptr, y.y, N, dtr::O_BF16, sc, 64, 0, 1, B.o_proj_t))) die(e);
            if ((e = dtr::make_out(y.ms, M, x2, HID, nullptr, y.shy, SH_N, dtr::O_F16, sc, 128, 0, 1, y.sh))) die(e);
            B.shw = y.ws; B.shw_bytes = shb;
        }
        if (sb > sbmax) { if (scr) CK(cudaFree(scr)); scr = dalloc<uint8_t>(sb); sbmax = sb; }
        for (auto& y : Y) { y.rp.scratch = scr; y.rp.scratch_bytes = sbmax; }
        {
            const double kvb = (double)std::min(L, TOP) * g6kv::ROW, ixb = (double)L * g6kv::IDX_ROW;
            const double xin = (double)(QA_T + KVA_T) * 192 * 528 + IDXK_T * 384 * 512;
            printf("\n== M=%d: per layer: x_in %.2f MB, q_b %.2f, UK+UV %.2f, o_proj %.2f, router %.2f, shared stand-in %.2f, KV %.2f, index keys %.2f "
                   "(full layers), attention scratch <= %.2f | q_b ks %d, o_proj ks %d, shared ks %d | x_in %s\n", M, xin / 1e6, qbb / 1e6,
                   (ukn + uvn) * 2 / 1e6, opb / 1e6, 16 * 384 * 512 / 1e6, shb / 1e6, kvb / 1e6, ixb / 1e6, sb / 1e6, Y[0].B.q_b_t.a.ks,
                   Y[0].B.o_proj_t.a.ks, Y[0].sh.a.ks, Y[0].B.x_in.n ? "one gv_multi" : "three launches");
        }

        // one graph of NL layers; stamps (seg) between the segments
        auto body = [&](bool seg) {
            l2pf::arm();
            k_bump<<<1, 1, 0, st>>>(stepp);
            const unsigned na = (unsigned)(arr * 1000);
            for (int l = 0; l < NL; ++l)
            {
                Lyr& y = Y[l];
                const l2pf::Plan P = l2pf::plan_layer(y.B, y.uk, y.uv);
                auto stop = [&](int c) { l2pf::Stop s; s.f = fab; s.stepp = stepp; s.c = c; return s; };
                u64* sp = stamps + (size_t)l * (NSEG + 1);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_WM);
                if (l > 0)
                {
                    l2pf::window(l2pf::W_M, l2pf::T_XIN, l, P, stop(3 * l), st);
                    k_wait_standin<<<1, 32, 0, st>>>(fab, stepp, 3 * l, (unsigned)(wm * 1000), na);
                }
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_XIN);
                if (y.B.x_in.n) bind::launch(y.B.x_in, st);
                else { bind::launch(y.B.q_a, st); bind::launch(y.B.kv_a, st); bind::launch(y.B.idx_k, st); }
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_WQ);
                l2pf::window(l2pf::W_Q, l2pf::T_IDXQ, l, P, stop(3 * l + 1), st);
                k_wait_standin<<<1, 32, 0, st>>>(fab, stepp, 3 * l + 1, (unsigned)(wq * 1000), na);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_QB);
                bind::launch_q_b(y.B, st);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_ATTN);
                l2pf::Stop none;
                none.stepp = stepp;
                l2pf::window(l2pf::W_T, l2pf::T_UK, l, P, none, st);
                CKI(g6_attn_replay_run(&y.rp, st));
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_OPROJ);
                bind::launch_o_proj(y.B, st);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_WA);
                l2pf::window(l2pf::W_A, l2pf::T_ROUTER, l, P, stop(3 * l + 2), st);
                k_wait_standin<<<1, 32, 0, st>>>(fab, stepp, 3 * l + 2, (unsigned)(wa * 1000), na);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_ROUTER);
                CK(cudaEventRecord(ef, st));
                CK(cudaStreamWaitEvent(ovl, ef, 0));
                dtr::launch(y.sh, ovl);
                CK(cudaEventRecord(ej, ovl));
                bind::launch(y.B.router, st);
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + S_MOE);
                if (moen) k_moe<<<sms * 4, 256, 0, st>>>(mbuf[l % NMOE], moen, sink);
                CK(cudaStreamWaitEvent(st, ej, 0));
                if (seg) k_stamp<<<1, 1, 0, st>>>(sp + NSEG);
            }
            l2pf::join(st);
        };
        auto set_arm = [&](int a) { CK(l2pf::setup(arms[a])); };

        // ---- exactness: every layer's outputs after one launch of the graph, arm vs off
        auto snap = [&](int a) {
            set_arm(a);
            cudaGraphExec_t x = inst(st, [&] { body(false); });
            for (auto& y : Y)
            {
                CK(cudaMemsetAsync(y.y, 0xff, (size_t)M * N * 2, st)); CK(cudaMemsetAsync(y.att, 0xff, (size_t)M * K * 2, st));
                CK(cudaMemsetAsync(y.ind, 0x5a, (size_t)M * TOP * 4, st)); CK(cudaMemsetAsync(y.qay, 0xff, (size_t)M * QA_T * 32, st));
                CK(cudaMemsetAsync(y.kvay, 0xff, (size_t)M * KVA_T * 32, st)); CK(cudaMemsetAsync(y.iky, 0xff, (size_t)M * IDXK_T * 32, st));
                CK(cudaMemsetAsync(y.qby, 0xff, (size_t)M * K * 2, st)); CK(cudaMemsetAsync(y.shy, 0xff, (size_t)M * SH_N * 2, st));
                CK(cudaMemsetAsync(y.rty, 0xff, (size_t)M * 256 * 4, st));
            }
            CK(cudaGraphLaunch(x, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaGraphExecDestroy(x));
            std::vector<uint8_t> s;
            auto put = [&](const void* p, size_t n) { auto h = host((const uint8_t*)p, n); s.insert(s.end(), h.begin(), h.end()); };
            for (int l = 0; l < NL; ++l)
            {
                const Lyr& y = Y[l];
                put(y.qay, (size_t)M * QA_T * 32); put(y.kvay, (size_t)M * KVA_T * 32); if (full[l]) put(y.iky, (size_t)M * IDXK_T * 32);
                put(y.qby, (size_t)M * K * 2); put(y.att, (size_t)M * K * 2); if (full[l]) put(y.ind, (size_t)M * TOP * 4);
                put(y.y, (size_t)M * N * 2); put(y.rty, (size_t)M * 256 * 4); put(y.shy, (size_t)M * SH_N * 2);
                if (host(y.err, 1)[0]) die("attention error flag set");
                G6AttnStepCell c;
                CK(cudaMemcpy(&c, y.cell, sizeof c, cudaMemcpyDeviceToHost));
                if (!c.valid) die("attention cell invalid");
            }
            return s;
        };
        const std::vector<uint8_t> ref = snap(0);
        {
            const std::vector<uint8_t> again = snap(0);
            printf("EXACT M=%d off vs off: %zu bytes differ (%s)\n", M, ndiff(ref.data(), again.data(), ref.size()), ref == again ? "deterministic" : "NOT DETERMINISTIC");
            if (ref != again) return 3;
        }
        int bad = 0;
        for (size_t a = 1; a < arms.size(); ++a)
        {
            const std::vector<uint8_t> s = snap((int)a);
            const size_t d = ndiff(ref.data(), s.data(), ref.size());
            printf("EXACT M=%d arm %-14s x_in, q_b, attention (+ indices), o_proj, router, shared of %d layers (%zu bytes): %zu differ | %s\n",
                   M, names[a].c_str(), NL, ref.size(), d, d ? "DIFFER" : "IDENTICAL");
            bad += d != 0;
        }
        if (bad) return 3;

        // ---- timing: the arms' graphs alternating over TR trials
        struct Run { cudaGraphExec_t g, s; std::vector<float> t; std::vector<float> seg[NSEG]; };
        std::vector<Run> R(arms.size());
        for (size_t a = 0; a < arms.size(); ++a)
        {
            set_arm((int)a);
            R[a].g = inst(st, [&] { body(false); });
            R[a].s = SEGT ? inst(st, [&] { body(true); }) : nullptr;
        }
        for (int t = 0; t < TR; ++t)
            for (size_t a = 0; a < arms.size(); ++a)
            {
                R[a].t.push_back(run_us(R[a].g, st, REPS) / NL);
                if (!R[a].s) continue;
                CK(cudaGraphLaunch(R[a].s, st));
                CK(cudaGraphLaunch(R[a].s, st));
                CK(cudaStreamSynchronize(st));
                const std::vector<u64> h = host(stamps, (size_t)NL * (NSEG + 1));
                for (int k = 0; k < NSEG; ++k)
                {
                    double sum = 0;
                    for (int l = 0; l < NL; ++l) sum += (double)(h[(size_t)l * (NSEG + 1) + k + 1] - h[(size_t)l * (NSEG + 1) + k]);
                    R[a].seg[k].push_back((float)(sum / NL / 1000.0));
                }
            }
        const float t0 = med(R[0].t);
        for (size_t a = 0; a < arms.size(); ++a)
        {
            const float ta = med(R[a].t);
            const l2pf::Cfg& c = arms[a];
            printf("TIME M=%d arm %-14s", M, names[a].c_str());
            if (a) printf(" (win %2d targets 0x%02x %3.0f GB/s wave %2d o_proj <= %4.1f MB)", c.win, c.tmask, c.gbps, c.wave, c.opmb);
            else printf(" %51s", "");
            printf(": layer %.1f us (%+.1f) = %+.3f ms over 78 layers%s\n", ta, ta - t0, (ta - t0) * 78 / 1e3, a ? "" : "   <- off");
        }
        if (SEGT)
        {
            printf("SEG M=%d us per layer (stamped graph; the stamps add ~1 us a segment) %-6s", M, "arm");
            for (int k = 0; k < NSEG; ++k) printf(" %7s", SEG[k]);
            printf("\n");
            for (size_t a = 0; a < arms.size(); ++a)
            {
                printf("SEG M=%d %-14s", M, names[a].c_str());
                for (int k = 0; k < NSEG; ++k) printf(" %7.1f", med(R[a].seg[k]));
                if (a)
                {
                    printf("  | vs off:");
                    for (int k = 0; k < NSEG; ++k) printf(" %+6.1f", med(R[a].seg[k]) - med(R[0].seg[k]));
                }
                printf("\n");
            }
        }
        for (auto& r : R) { CK(cudaGraphExecDestroy(r.g)); if (r.s) CK(cudaGraphExecDestroy(r.s)); }
        set_arm(0);
    }
    printf("\nfull-index layers %d of %d; kill criterion: < 15 us per layer saved at M=3, 1K context\n", nfull, NL);
    return 0;
}
