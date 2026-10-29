// k1_audit: does the routed-expert MoE phase (router plan -> fc1 -> fc2 -> finalize) read its weights at the DRAM floor?
// One GB10, no pack files, no fabric. The layers are synthetic exlr8 v2 layers in the GLM-5.3 TP4 rank shape (as
// k1_ovl_test: 256 routed experts x 4 blocks of 128 columns + the trellis shared expert as tr3.table row 256, block-scale
// vectors, exlr8-e4m3, K3 tiles in the wide order) with random trellis words, but every expert has the width the EXLR8
// width manifest gives it on this rank (--manifest; home = 64 K4 + 192 K3 routed experts per layer, shared K4), and the
// layers are as many as the model has (75 MoE layers, 3..77, ~0.99 GB each on a rank: ~74 GB), each in its own
// allocation, so a step's calls stream cold weights exactly as the engine's do (the L2 holds 24 MB of one call's ~75 MB).
//
// Routing: the plan of every (step, layer) call comes from the engine's k_router_plan on logits that select given
// experts per row: --routes FILE (GR8_ROUTE_LOG lines: the distinct expert ids of every MoE layer of every MTP main
// step; offline --prompt runs only, rank.cuh route_log_line; the lines with M = the bench M are used), else --exlog FILE
// (the "# s2_round" lines of a GR8_S2_ROUNDLOG=1 rank-0 log: distinct counts per layer per main step, and the ids
// themselves when the lines carry them; without ids they are drawn uniformly), else the served means
// (kstop-1015 meas.r0.err, 5823 main steps: distinct per layer 8.0 / 13.5 / 18.6 / 23.9 / 28.6 / 33.7 / 38.6 / 44.3 at
// M = 1..8; above 8 rows +5.2 per row). Rows are then given 8 experts each so that every distinct id is used (the
// log has no row->expert map; it changes the slot masks, not the bytes).
//
// Bytes per call (the numerator): the routed experts' fc1 + fc2 tiles at their widths + the shared expert's + their
// block-scale vectors; every one is read exactly once per call at M <= 8 (an expert with more than 8 tokens takes
// two slots above 8 rows and its tiles are fetched twice, the second time mostly from L2: the "slots" column).
//
//   k1_audit [--M 1,3,4,8,12] [--manifest exlr8-manifest-home.json] [--rank 0] [--widths home|k2|k3|k4|mix|mod:N]
//            [--routes route.jsonl | --exlog rank0.err] [--steps 8] [--layers 75] [--reps 5] [--arms all|base|ns2|tlbw]
//            [--nowide] [--nowide2] [--seed 1] [--check-calls 75] [--ncu N] [--split 0|2|3|5] [--sess R]
//            [--alloc each|one|vmm[:MB]] [--hold N] [--window O] [--tab GR8_K1_SPLIT-table] [--tlb 0|1|only]
//            [--w2all 0|1] [--check-only]
//   The v2 release form (a boot from the exlr8 release, --v2-dense, TP=4): --manifest = the release's own width manifest
//   (manifests/exlr8-manifest-home.json of the model set, mixed widths per unit -> tr3.kmix) and --w2all 1 (the launcher's
//   TP=4 default, TR3_W2ALL=1). The layers are the release's MoE layer form: the trellis shared expert as row 256 with
//   block-scale hidden-side vectors (tr3.sg / v1b / v3b: the BS fc1 bodies and finalize), exlr8-e4m3, K3 tiles widened.
//   --widths k2 / mix cover the K2 bodies too (every routed unit K2; or K2 / K3 / K4 per unit and per expert down width).
//   --split S   bind every arm as an S2 step of row class S (make_k1 split_m = S; the engine's batched main step is
//               class row_class(K+1) = 3 at MTP K = 2, M = sessions x 3). 0 = the solo split of M rows (class 5 at M >= 4:
//               DFlash2 verify at 8 rows). The "bind" line prints the split the k1 arm got (a GR8_K1_SPLIT table: --tab).
//   --sess R    also check: the k1 arm's M rows == M / R solo calls of R rows each (split 0, the session's solo step),
//               row for row (a batched row's bits do not depend on the batch; with every ns2 arm == k1, for them too).
//               Give --split the class of R rows (3 for R = 2, 3): another class sums fc1 k partials in another order.
//   --arms ns2  k1, the plan/fc1/fc2 split, the ns2 arms and read1 only (the NS2 sweep).
//   --arms tlbw k1, the split, the tlbw arms (and tab with --tab) and read1 only.
//   --alloc     the weights' device memory: each = one cudaMalloc per tensor (the default; the runs before
//               k1gap); one = ONE cudaMalloc for every held layer's tensors, 4 KB-aligned, back to back in layer order,
//               as the engine's device image (runtime/packfind.h rs_fill); vmm[:MB] = that range from cuMemCreate chunks
//               of MB (default: the recommended granularity), as GR8_IMG_VMM (engine/vmm.h). Same bytes, other pages.
//   --hold N    allocate N layers (default --layers) and bench --layers of them from layer --window O (default 0):
//               --layers 8 --hold 75 --window 0 / 67 tells the footprint (TLB reach, page walks) from the allocation
//               order (late layers on more fragmented memory: smaller pages).
//   --tab T     arms "tab", "tab <other form>", "tab chain-ovl1", "tab chain-ovl2": the k1 arm bound with GR8_K1_SPLIT=T
//               (bind.cuh; e.g. 1:/4,3:2x1,4-32:/4) in the K1 settings make_k1 binds with (bind::K1Vars), as the
//               launcher's environment gives it, so both readers of the variable see it (k1_split's triple form skips a
//               table, #155; before that it read 1:... as KS1 = 1: the #107 report), in the engine's launch form, the
//               other one (fused / separate) and the route-overlap forms. The k1 arm itself is bound without
//               GR8_K1_SPLIT / GR8_K1_SPLIT_M* (the environment's are ignored).
//   --w2all V   GR8_TR3_W2ALL=V for every binding (the launcher sets 1 at TP=4). Default: the environment's value
//               (unset = 0). The header line prints the value in force.
//   --check-only  the exactness checks only (no stream floor, no bench): for a matrix of ranks / widths / M.
//   --tlb 1     after the benches: the GPU address-translation probe (one thread chasing pointers stored in the held
//               weights at stride S over N granules from the first held layer, and inside the last one): latency per
//               access against S x N, the knee per S, and the page size it implies; plus the system's page size,
//               THP mode, /proc/buddyinfo and the VMM granularity (read only). --tlb only: the probe without benches.
//
// Arms (all inside one CUDA graph over steps x layers calls, median of --reps interleaved timings, us per call):
//   k1            the engine's K1 at M rows as rank.cuh binds it (GR8_FUSE=1 + the fuse policy of bind.cuh make_k1 for
//                 these layers: fused at M = 1, separate fc1 / fc2 launches above), k_router_plan in front. THE number:
//                 GB/s = call bytes / its time.
//   split         plan alone, + fc1 (or the fused kernel), + fc2, + finalize: per-kernel marginal time incl. boundaries.
//   chain-ovlN    the full served MoE block: x2 fragment tables, router GEMV (BF16, 3.15 MB, written to a dummy:
//                 the plan still reads the served logits), plan, K1; ovl1 = GR8_ROUTE_OVL=1 (the serving flag), ovl0 off.
//   fused/sep     the other launch form (GR8_K1_FUSE_POLICY=0): exact (same bytes out), 3-11 % either way in the table
//                 bind.cuh quotes; the M >= 2 rule for trellis-shared layers was set for 16+ rows.
//   fc1 KSxWG     fc1 CTA shape with the same k ranges (KS1 * WG fixed: partial p = h * WG + q covers the same k range,
//                 fc2 sums the same partials in the same order): exact. 256 threads/CTA = 2 CTAs per SM at M 2-3.
//   ns2=N         fc2 n-split (every z tile is still one warp's sum over its block's 8 k-tiles): exact. N a power of two
//                 that divides 384 (the kernels index fc2 units by shifts, #136): 1 2 4 8 16 32. The "ns2 pick" line names the fastest exact one (k1 = the bound NS2) per M.
//   tab ...       with --tab: the k1 arm bound with that GR8_K1_SPLIT table, in each launch form (exact by construction;
//                 checked, the engine form and chain-ovl1 also as a graph).
//   tlbwN         GR8_K1_TLBW=N KB (N = 64, 2048): fc1 prefetches every routed slot's fc1 + fc2 translations at the
//                 start of the launch (k1.cuh k1_tlbw); chain-ovl1 tlbw2048 = the same in the served route-overlap form.
//   read1/read2   the floor for the same bytes: a plain 16 B ld.global.nc read of exactly the call's tile ranges in one
//                 launch (read2: fc1 ranges, then fc2 ranges), same graph, same cold rotation.
//   stream        (once) a persistent read over 4 GB of cold weights: the GPU's plain-read rate.
// Compile-time load-order arms are separate builds of this file (add the flag to its nvcc line): -DTR3_UNR=4,
// -DTR3_PU=1|4, -DTR3_EARLY=0, -DTR3_XCOMP=1. Each prints the same "digest" line as the default build when its bytes are
// the same. (TR3_UNR1 / TR3_EARLY4 are printed for comparison with the audit's development builds; this tree's tr3.cuh
// does not read them.)
//
// Check (before timing): every arm's outputs (BF16 [M][6144] per call) over the first --check-calls calls, plain
// launches, == the k1 arm's, byte for byte; the k1 arm and chain-ovl1 also as one CUDA graph. FAIL = not exact.
// --ncu N: after the setup, N k1 calls as plain launches (for ncu: lts__t_sectors_lookup_miss.sum x 32 B = DRAM bytes
// fetched, gb10-no-dram-counters), no check, no bench.
// Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error / usage.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include "../glue.cuh"
#include "../bind.cuh"
#include "../../knobs-rs/gr8_knobs.h"
#include "../../vmm.h"

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

#ifndef TR3_EARLY4
#define TR3_EARLY4 0
#endif
#ifndef TR3_UNR1
#define TR3_UNR1 TR3_UNR     // a load-order knob of the audit's development builds; not in this tree's tr3.cuh
#endif

static int g_pass = 0, g_fail = 0;
static const int NL_MODEL = 75, L0 = 3;            // MoE layers 3..77 of the main model

// ------------------------------------------------------------------------------------------------ fills
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

// ------------------------------------------------------------------------------------------------ plain reads (floors)
__device__ __forceinline__ uint4 ld_nc128(const void* p)
{
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
struct Chunk { const uint8_t* p; unsigned bytes, pad; };           // bytes <= 32768, a multiple of 4096
// one CTA = one chunk: 256 threads x 8 loads of 16 B in flight
__global__ void __launch_bounds__(256) k_seg_read(const Chunk* c, unsigned* sink)
{
    const Chunk ch = c[blockIdx.x];
    const uint4* q = reinterpret_cast<const uint4*>(ch.p);
    const int n16 = static_cast<int>(ch.bytes >> 4);
    uint4 v[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = threadIdx.x + i * 256 < n16 ? ld_nc128(q + threadIdx.x + i * 256) : make_uint4(0, 0, 0, 0);
    unsigned a = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) a ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    if (a == 0x9e3779b9u) sink[blockIdx.x & 255] = a;               // keeps the loads; practically never stores
}
// persistent grid-stride read of n16 x 16 B, 4 loads in flight per thread per iteration
__global__ void k_stream(const uint4* p, size_t n16, unsigned* sink)
{
    const size_t st = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned a = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n16; i += 4 * st)
    {
        uint4 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = i + u * st < n16 ? ld_nc128(p + i + u * st) : make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) a ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (a == 0x9e3779b9u) sink[threadIdx.x] = a;
}

// ------------------------------------------------------------------------------------------------ tiny JSON (manifest, route log)
struct J
{
    enum { NUL, NUM, STR, ARR, OBJ } t = NUL;
    double n = 0;
    std::string s;
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J* get(const std::string& k) const
    {
        for (const auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
static void jws(const char*& p) { while (*p && isspace(static_cast<unsigned char>(*p))) ++p; }
static bool jstr(const char*& p, std::string& s)
{
    if (*p != '"') return false;
    ++p;
    s.clear();
    while (*p && *p != '"') { if (*p == '\\' && p[1]) ++p; s += *p++; }
    if (*p != '"') return false;
    ++p;
    return true;
}
static bool jparse(const char*& p, J& v)
{
    jws(p);
    if (*p == '{')
    {
        v.t = J::OBJ; ++p; jws(p);
        if (*p == '}') { ++p; return true; }
        for (;;)
        {
            std::string k;
            jws(p);
            if (!jstr(p, k)) return false;
            jws(p);
            if (*p++ != ':') return false;
            v.o.emplace_back(k, J());
            if (!jparse(p, v.o.back().second)) return false;
            jws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; return true; }
            return false;
        }
    }
    if (*p == '[')
    {
        v.t = J::ARR; ++p; jws(p);
        if (*p == ']') { ++p; return true; }
        for (;;)
        {
            v.a.emplace_back();
            if (!jparse(p, v.a.back())) return false;
            jws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; return true; }
            return false;
        }
    }
    if (*p == '"') { v.t = J::STR; return jstr(p, v.s); }
    if (!strncmp(p, "null", 4) || !strncmp(p, "true", 4)) { p += 4; return true; }
    if (!strncmp(p, "false", 5)) { p += 5; return true; }
    char* e;
    v.n = strtod(p, &e);
    if (e == p) return false;
    v.t = J::NUM;
    p = e;
    return true;
}
static std::string slurp(const char* f)
{
    FILE* fp = fopen(f, "rb");
    if (!fp) { printf("cannot open %s\n", f); exit(2); }
    std::string s;
    char b[1 << 16];
    size_t n;
    while ((n = fread(b, 1, sizeof b, fp)) > 0) s.append(b, n);
    fclose(fp);
    return s;
}

// ------------------------------------------------------------------------------------------------ widths
// per bench layer: [257] fc1 unit widths (8 = 2 per block, u = 2b + mat) and the down width, on this rank
struct Widths { int u[257][8]; int down[257]; };
// half h of expert e in layer l lives on rank (16 e + h + l) mod NR (tools/tune/route_manifest.py, the v2 sharding
// rule); a rank's blocks are its halves in ascending h
static void widths_from_manifest(const J& man, int l, int rank, Widths& W)
{
    const J* L = man.get("layers");
    const J* y = L ? L->get(std::to_string(l)) : nullptr;
    if (!y) { printf("manifest has no layer %d\n", l); exit(2); }
    for (int e = 0; e < 257; ++e)
    {
        if (e == 256 && !y->get("shared")) { printf("manifest layer %d has no shared expert widths\n", l); exit(2); }
        const J* g = e < 256 ? y->get("gate") : y->get("shared")->get("gate");
        const J* up = e < 256 ? y->get("up") : y->get("shared")->get("up");
        const J* d = e < 256 ? y->get("down") : y->get("shared")->get("down");
        const J& ge = e < 256 ? g->a[e] : *g;
        const J& ue = e < 256 ? up->a[e] : *up;
        int b = 0;
        for (int h = 0; h < 16 && b < g6nr::EB; ++h)
            if ((16 * e + h + l) % g6nr::NR == rank)
            {
                W.u[e][2 * b] = static_cast<int>(ge.a[h].n);
                W.u[e][2 * b + 1] = static_cast<int>(ue.a[h].n);
                ++b;
            }
        if (b != g6nr::EB) { printf("layer %d expert %d: %d halves on rank %d, not %d\n", l, e, b, rank, g6nr::EB); exit(2); }
        W.down[e] = static_cast<int>(e < 256 ? d->a[e].n : d->n);
    }
}
// mode 2 / 3 / 4: all routed experts at that width; 0: e % mod == 0 -> K4, the rest K3; -1 (mix): routed unit u of
// expert e at K(2 + (e + u) % 3) (mixed widths per unit: tr3.kmix), down at K(2 + e % 3). The shared expert is K4.
static void widths_fixed(int mode, int mod, Widths& W)
{
    for (int e = 0; e < 257; ++e)
    {
        const int k = e == 256 ? 4 : mode > 0 ? mode : mode < 0 ? 2 + e % 3 : e % mod == 0 ? 4 : 3;
        for (int u = 0; u < 2 * g6nr::EB; ++u) W.u[e][u] = e < 256 && mode < 0 ? 2 + (e + u) % 3 : k;
        W.down[e] = k;
    }
}

// ------------------------------------------------------------------------------------------------ layer
static bind::Tensor mk(const void* p, int nd, long long a, long long b = 0, long long c = 0)
{
    bind::Tensor t;
    t.p = p; t.nd = nd; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    return t;
}
constexpr size_t MB1 = static_cast<size_t>(KSH) * 8 * 32;        // bytes of one fc1 matrix (or fc2 block) per bit of width
constexpr size_t VEC_ROW = (3 * 128 + 2 * (HID / 128) + HID / 128) * 2;   // v2 + v1b + v3b bytes of one block row

struct SynLayer
{
    bind::LayerTensors T;
    std::vector<long long> tab;      // host [257][6]
    size_t f1b[257], f2b[257];       // this rank's fc1 / fc2 bytes of each expert
    int k4[257];                     // the expert is K4 (down width 4)
    const uint8_t* fc1 = nullptr;
    const uint8_t* fc2 = nullptr;
    size_t o1 = 0, o2 = 0;           // tensor bytes
    bind::Tensor router;
};

// --alloc: where build_layer puts a layer's tensors. each = cudaMalloc per tensor; one / vmm = bump allocation (4 KB
// alignment, as the image's g6pack-v1 align) from one range sized by layer_need over the held layers.
struct Arena
{
    std::string mode = "each";
    size_t chunk = 0;                // vmm: chunk bytes (0 = recommended granularity)
    uint8_t* base = nullptr;
    size_t cap = 0, used = 0;
    void begin(size_t need)
    {
        cap = need;
        if (mode == "one")
        {
            if (cudaMalloc(&base, need) != cudaSuccess) { printf("--alloc one: cudaMalloc of %.2f GB failed\n", need / 1e9); exit(2); }
        }
        else if (mode == "vmm")
        {
            std::string why;
            base = static_cast<uint8_t*>(g6vmm::alloc(need, chunk, why));
            if (!base) { printf("--alloc vmm: %s\n", why.c_str()); exit(2); }
        }
    }
    void* get(size_t n, std::vector<void*>& mem)
    {
        if (mode == "each") { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; }
        const size_t at = (used + 4095) / 4096 * 4096;
        if (at + n > cap) { printf("--alloc %s: range of %zu bytes exhausted\n", mode.c_str(), cap); exit(2); }
        used = at + n;
        return base + at;
    }
    void end()
    {
        if (mode == "one") cudaFree(base);
        else if (mode == "vmm") g6vmm::release(base);
        base = nullptr;
    }
};
// bytes build_layer takes from a bump range for widths W (each tensor rounded up to 4 KB)
static size_t layer_need(const Widths& W)
{
    auto r4 = [](size_t n) { return (n + 4095) / 4096 * 4096; };
    size_t o1 = 0, o2 = 0;
    const size_t rows = static_cast<size_t>(257) * g6nr::EB;
    for (int e = 0; e < 257; ++e)
    {
        for (int u = 0; u < 2 * g6nr::EB; ++u) o1 += static_cast<size_t>(W.u[e][u]) * MB1;
        o2 += static_cast<size_t>(g6nr::EB) * W.down[e] * MB1;
    }
    return r4(257 * 6 * 8) + r4(257 * 8) + r4(o1) + r4(o2) + r4(rows * 3 * 128 * 2) + r4(2 * HID * 2) + r4(rows * 2 * (HID / 128) * 2) +
           r4(rows * (HID / 128) * 2) + r4(static_cast<size_t>(16) * 384 * 256 * 2);
}

static void build_layer(SynLayer& L, const Widths& W, uint32_t seed, bool wide1, bool wide2, std::vector<void*>& mem, Arena& ar)
{
    L.tab.assign(257 * 6, 0);
    std::vector<long long> km(257, 0);
    bool any_mix = false;
    size_t o1 = 0, o2 = 0;
    int row = 0;
    for (int e = 0; e < 257; ++e)
    {
        const int nb = g6nr::EB, kd = W.down[e];
        bool uni = true;
        long long kmix = 0;
        size_t b1 = 0;
        for (int u = 0; u < 2 * nb; ++u)
        {
            uni = uni && W.u[e][u] == kd;
            kmix |= static_cast<long long>(W.u[e][u]) << (4 * u);
            b1 += static_cast<size_t>(W.u[e][u]) * MB1;
        }
        if (!uni) { km[e] = kmix; any_mix = true; }
        long long* r = &L.tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = kd; r[5] = 0;
        L.f1b[e] = b1;
        L.f2b[e] = static_cast<size_t>(nb) * kd * MB1;
        L.k4[e] = kd == 4;
        o1 += b1;
        o2 += L.f2b[e];
        row += nb;
    }
    auto alloc = [&](size_t n) { return ar.get(n, mem); };
    long long* tab = static_cast<long long*>(alloc(L.tab.size() * 8));
    CK(cudaMemcpy(tab, L.tab.data(), L.tab.size() * 8, cudaMemcpyHostToDevice));
    long long* kmd = nullptr;
    if (any_mix)
    {
        kmd = static_cast<long long*>(alloc(257 * 8));
        CK(cudaMemcpy(kmd, km.data(), 257 * 8, cudaMemcpyHostToDevice));
    }
    uint8_t* fc1 = static_cast<uint8_t*>(alloc(o1));
    uint8_t* fc2 = static_cast<uint8_t*>(alloc(o2));
    fill_words(fc1, o1, seed + 1);
    fill_words(fc2, o2, seed + 2);
    L.fc1 = fc1; L.fc2 = fc2; L.o1 = o1; L.o2 = o2;
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
    if (kmd) T.t[bind::T_TR3_KMIX] = mk(kmd, 1, 257);
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
    if (wide1) bind::widen_fc1(T, nullptr);            // as rank.cuh with --wide (scripts/engine_args.sh), GR8_NOWIDE2 unset
    if (wide2) bind::widen_fc2(T, nullptr);
    const size_t rn = static_cast<size_t>(16) * 384 * 256;      // router: BF16 records [16][384][512], as k1_ovl_test
    __nv_bfloat16* rw = static_cast<__nv_bfloat16*>(alloc(rn * 2));
    k_fill_bf16<<<1024, 256>>>(rw, rn, 0.02f, seed + 6);
    CK(cudaGetLastError());
    L.router = mk(rw, 3, 16, 384, 512);
    CK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------------------------ routing
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
// distinct ids D -> M rows of 8 distinct experts that use every id: row 0 takes 8, the rest of D is spread over the
// other rows, and each row is topped up from ids already used
static void rows_from_distinct(std::vector<int> D, int M, std::mt19937& rng, std::vector<int>& rows)
{
    std::shuffle(D.begin(), D.end(), rng);
    std::vector<char> in(256, 0);
    for (int e : D) in[e] = 1;
    for (int e = 0; static_cast<int>(D.size()) < 8 && e < 256; ++e) if (!in[e]) { D.push_back(e); in[e] = 1; }
    if (static_cast<int>(D.size()) > 8 * M) D.resize(8 * M);
    const int d = static_cast<int>(D.size());
    rows.assign(static_cast<size_t>(M) * 8, 0);
    for (int k = 0; k < 8; ++k) rows[k] = D[k];
    int next = 8;
    for (int m = 1; m < M; ++m)
    {
        const int left = M - m, k = std::min(8, (d - next + left - 1) / left);
        std::vector<int> r(D.begin() + next, D.begin() + next + k);
        next += k;
        std::vector<int> pool(D.begin(), D.begin() + next - k);
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int e : pool) { if (static_cast<int>(r.size()) == 8) break; r.push_back(e); }
        for (int e = 0; static_cast<int>(r.size()) < 8 && e < 256; ++e)
            if (std::find(r.begin(), r.end(), e) == r.end()) r.push_back(e);
        std::copy(r.begin(), r.end(), rows.begin() + static_cast<size_t>(m) * 8);
    }
}
static std::vector<int> random_distinct(int d, std::mt19937& rng)
{
    std::vector<int> all(256);
    for (int e = 0; e < 256; ++e) all[e] = e;
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(std::max(8, std::min(256, d)));
    return all;
}
static double served_mean(int M)    // kstop-1015 meas.r0.err, distinct routed experts per MoE layer per main step
{
    static const double t[9] = {0, 8.0, 13.474, 18.626, 23.875, 28.636, 33.702, 38.556, 44.312};
    return M <= 8 ? t[M] : t[8] + 5.2 * (M - 8);
}

// ------------------------------------------------------------------------------------------------ translation probe (--tlb)
static std::string first_line(const char* f)
{
    std::string s;
    if (FILE* fp = fopen(f, "r"))
    {
        char b[512];
        if (fgets(b, sizeof b, fp)) s = b;
        fclose(fp);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.empty() ? "unknown" : s;
}
// read only: the host page size, THP mode, free blocks per order (fragmentation) and the GPU's memory attributes
static void sys_info()
{
    printf("sys: page size %ld B, THP enabled %s, defrag %s\n", sysconf(_SC_PAGESIZE), first_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str(),
           first_line("/sys/kernel/mm/transparent_hugepage/defrag").c_str());
    if (FILE* fp = fopen("/proc/buddyinfo", "r"))
    {
        char b[512];
        while (fgets(b, sizeof b, fp)) printf("sys: buddyinfo %s", b);
        fclose(fp);
    }
    int integ = -1, pma = -1, hpt = -1;
    cudaDeviceGetAttribute(&integ, cudaDevAttrIntegrated, 0);
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, 0);
    cudaDeviceGetAttribute(&hpt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    size_t gmin = 0, grec = 0;
    g6vmm::granularity(0, gmin, grec);
    printf("sys: GPU integrated %d, pageable memory access %d (host page tables %d), VMM granularity min %zu KB rec %zu KB (0: no VMM)\n",
           integ, pma, hpt, gmin >> 10, grec >> 10);
}
__global__ void k_chain_write(const unsigned long long* a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        *reinterpret_cast<unsigned long long*>(a[i]) = a[i + 1 < n ? i + 1 : 0];
}
// one thread: `warm` dependent loads (a lap: L2 lines and translations in), then `reps` timed ones
__global__ void k_chase(unsigned long long p, int warm, int reps, long long* cyc, unsigned long long* sink)
{
    for (int i = 0; i < warm; ++i) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
    const long long t0 = clock64();
    for (int i = 0; i < reps; ++i) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
    const long long t1 = clock64();
    cyc[0] = t1 - t0;
    sink[0] = p;
}
struct Region { uintptr_t a, b; };
// The chain is N lines, one per S-aligned granule of the regions in order (a line offset inside the granule's first
// 4 KB spreads the L2 sets), so N lines always fit the L2 and a slower load is a translation miss: at stride S >= the
// page size P the latency rises past N = E (the entries of a TLB level), at S < P past E x P / S. The page size is
// the smallest S whose knee equals the largest S's. Writes the chain words INTO the weights: after the benches only.
static void tlb_probe(const char* what, const std::vector<Region>& R, cudaStream_t st)
{
    const size_t Ss[4] = {4096, 65536, size_t(2) << 20, size_t(32) << 20};
    const char* sn[4] = {"4K", "64K", "2M", "32M"};
    const int Ns[11] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    unsigned long long *da, *sink;
    long long* dc;
    CK(cudaMalloc(&da, 16384 * 8));
    CK(cudaMalloc(&dc, 8));
    CK(cudaMalloc(&sink, 8));
    printf("tlb %s: cycles per dependent L2-hit load (ld.global.cg, one thread), stride S over N granules ('-': fewer granules)\n", what);
    printf("tlb %s: S \\ N", what);
    for (int n : Ns) printf(" %6d", n);
    printf("\n");
    int knee[4];
    for (int k = 0; k < 4; ++k)
    {
        std::vector<unsigned long long> v;
        for (const Region& r : R)
        {
            for (uintptr_t a = (r.a + Ss[k] - 1) / Ss[k] * Ss[k]; a + 4096 <= r.b && v.size() < 16384; a += Ss[k]) v.push_back(a + ((v.size() * 7) & 31) * 128);
            if (v.size() >= 16384) break;
        }
        double lat[11], lo = 1e30;
        knee[k] = 0;
        printf("tlb %s: %5s", what, sn[k]);
        for (int j = 0; j < 11; ++j)
        {
            lat[j] = -1;
            if (Ns[j] > static_cast<int>(v.size())) { printf(" %6s", "-"); continue; }
            CK(cudaMemcpy(da, v.data(), Ns[j] * 8, cudaMemcpyHostToDevice));
            k_chain_write<<<64, 256, 0, st>>>(da, Ns[j]);
            const int reps = std::max(4096, 2 * Ns[j]);
            k_chase<<<1, 1, 0, st>>>(v[0], Ns[j], reps, dc, sink);
            long long cyc = 0;
            CK(cudaMemcpyAsync(&cyc, dc, 8, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            lat[j] = static_cast<double>(cyc) / reps;
            lo = std::min(lo, lat[j]);
            if (!knee[k] && lat[j] > 1.2 * lo) knee[k] = Ns[j];
            printf(" %6.0f", lat[j]);
        }
        printf("   knee %s\n", knee[k] ? std::to_string(knee[k]).c_str() : "none");
    }
    int big = -1;
    for (int k = 3; k >= 0 && big < 0; --k) if (knee[k]) big = k;
    int pg = -1;
    for (int k = 0; k <= big && pg < 0; ++k) if (knee[k] && knee[k] <= 2 * knee[big]) pg = k;
    if (pg < 0) printf("tlb %s: no knee at any stride (every chain within the first TLB level's reach)\n", what);
    else printf("tlb %s: page size ~%s (smallest stride whose knee, N=%d, matches the %s stride's; N ~ the entries of the first TLB level that misses)\n",
                what, sn[pg], knee[pg], sn[big]);
    CK(cudaFree(da));
    CK(cudaFree(dc));
    CK(cudaFree(sink));
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv)
{
    std::vector<int> Ms = {1, 3, 4, 8, 12};
    int steps = 8, nlay = NL_MODEL, reps = 5, rank = 0, seed = 1, all_arms = 1, nowide = 0, nowide2 = 0, ncheck = NL_MODEL, ncu = 0, split = 0, sess = 0;
    int hold = 0, window = 0, tlb = 0, check_only = 0, w2all = -1;
    std::string manifest, routes, exlog, widths = "home", tab;
    Arena arena;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { printf("missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--M") Ms = parse_list(nx());
        else if (a == "--manifest") manifest = nx();
        else if (a == "--rank") rank = atoi(nx());
        else if (a == "--widths") widths = nx();
        else if (a == "--routes") routes = nx();
        else if (a == "--exlog") exlog = nx();
        else if (a == "--steps") steps = atoi(nx());
        else if (a == "--layers") nlay = atoi(nx());
        else if (a == "--reps") reps = atoi(nx());
        else if (a == "--arms") { const char* v = nx(); all_arms = !strcmp(v, "base") ? 0 : !strcmp(v, "ns2") ? 2 : !strcmp(v, "tlbw") ? 3 : 1; }
        else if (a == "--alloc")
        {
            const std::string v = nx();
            arena.mode = v.substr(0, v.find(':'));
            if (v.find(':') != std::string::npos) arena.chunk = static_cast<size_t>(atol(v.c_str() + v.find(':') + 1)) << 20;
            if (arena.mode != "each" && arena.mode != "one" && arena.mode != "vmm") { printf("--alloc each|one|vmm[:MB]\n"); return 2; }
        }
        else if (a == "--hold") hold = atoi(nx());
        else if (a == "--window") window = atoi(nx());
        else if (a == "--tab") tab = nx();
        else if (a == "--w2all") w2all = atoi(nx());
        else if (a == "--check-only") check_only = 1;
        else if (a == "--tlb") { const char* v = nx(); tlb = !strcmp(v, "only") ? 2 : atoi(v); }
        else if (a == "--split") split = atoi(nx());
        else if (a == "--sess") sess = atoi(nx());
        else if (a == "--nowide") nowide = 1;
        else if (a == "--nowide2") nowide2 = 1;
        else if (a == "--seed") seed = atoi(nx());
        else if (a == "--check-calls") ncheck = atoi(nx());
        else if (a == "--ncu") ncu = atoi(nx());
        else { printf("unknown arg %s (see the file header)\n", a.c_str()); return 2; }
    }
    for (int M : Ms) if (M < 1 || M > G6_MX) { printf("M=%d outside 1..%d\n", M, G6_MX); return 2; }
    if (steps < 1 || nlay < 1 || nlay > NL_MODEL || reps < 1 || rank < 0 || rank >= g6nr::NR) { printf("bad --steps / --layers / --reps / --rank\n"); return 2; }
    if (split < 0 || split > 8 || sess < 0) { printf("bad --split / --sess\n"); return 2; }
    if (!hold) hold = nlay;
    if (hold < nlay || hold > NL_MODEL || window < 0 || window + nlay > hold) { printf("bad --hold / --window (window + layers <= hold <= %d)\n", NL_MODEL); return 2; }
    // the K1 settings of every binding: the release split (no GR8_K1_SPLIT*), no GR8_K1_TLBW (the tlbw arms set it per
    // binding), GR8_K1_FUSE_POLICY=0 (the bench passes the engine's choice itself, fuse_engine), --w2all or the environment's
    bind::K1Vars k1v;
    k1v.policy = false;
    k1v.w2all = w2all >= 0 ? w2all != 0 : knob::tr3_w2all();
    const char* ignored[] = {knob::k1_split() ? "GR8_K1_SPLIT" : nullptr, knob::k1_split_m1() ? "GR8_K1_SPLIT_M1" : nullptr,
                             knob::k1_split_m2() ? "GR8_K1_SPLIT_M2" : nullptr, knob::k1_split_m4() ? "GR8_K1_SPLIT_M4" : nullptr,
                             knob::k1_tlbw() ? "GR8_K1_TLBW" : nullptr};
    for (const char* v : ignored)
        if (v) printf("env %s ignored: the k1 arm is the release split (use --tab / the tlbw arms)\n", v);
    if (const char* e = bind::k1_split_env(tab.c_str(), k1v).err) { printf("--tab %s: %s\n", tab.c_str(), e); return 2; }
    bind::k1_split_env("", k1v);             // the k1 arm: no table

    CK(cudaSetDevice(0));
    CK(bind::init());
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("k1_audit: %s sm_%d%d %d SMs, G6_NR=%d EB=%d G6_MX=%d rank %d | build TR3_UNR=%d TR3_UNR1=%d TR3_PU=%d TR3_EARLY=%d TR3_EARLY4=%d TR3_XCOMP=%d\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount, g6nr::NR, g6nr::EB, G6_MX, rank, TR3_UNR, TR3_UNR1, TR3_PU, TR3_EARLY,
           TR3_EARLY4, TR3_XCOMP);
    printf("k1_audit: GR8_TR3_W2ALL=%d (release split per M:", k1v.w2all ? 1 : 0);
    for (int M : Ms) { const bind::K1Split r = bind::k1_split(M, 0, k1v); printf(" M%d %d,%d,%d", M, r.KS1, r.WG, r.NS2); }
    printf(")%s\n", check_only ? ", --check-only" : "");

    // widths
    J man;
    std::string mtext;
    if (!manifest.empty() && widths == "home")
    {
        mtext = slurp(manifest.c_str());
        const char* p = mtext.c_str();
        if (!jparse(p, man) || man.t != J::OBJ) { printf("%s: not JSON\n", manifest.c_str()); return 2; }
        const J* nm = man.get("name");
        printf("widths: manifest %s (%s)\n", manifest.c_str(), nm ? nm->s.c_str() : "?");
    }
    else if (widths == "home") { widths = "mod:4"; printf("widths: NO --manifest: synthetic home (expert e %% 4 == 0 at K4, the rest K3, shared K4)\n"); }
    else printf("widths: %s (diagnostic; not the served widths)\n", widths.c_str());

    // memory: each layer ~0.99 GB at home widths; fewer layers if the device has not the room
    {
        size_t fr = 0, to = 0;
        CK(cudaMemGetInfo(&fr, &to));
        const double per = 1.0e9, keep = 6.0e9;
        const int fit = static_cast<int>((static_cast<double>(fr) - keep) / per);
        if (fit < hold)
        {
            const int h = std::max(fit, 2);
            printf("WARNING: %.1f GB free: %d layers held instead of %d (a smaller cold set than the engine's)\n", fr / 1e9, h, hold);
            hold = h;
            nlay = std::min(nlay, hold);
            window = std::min(window, hold - nlay);
        }
    }
    if (tlb) sys_info();                   // before the weights: the memory state their pages come from
    std::vector<void*> mem;
    std::vector<SynLayer> All(hold);
    {
        std::vector<Widths> Ws(hold);
        size_t need = 0;
        for (int l = 0; l < hold; ++l)
        {
            Widths& W = Ws[l];
            if (man.t == J::OBJ) widths_from_manifest(man, L0 + l, rank, W);
            else if (widths == "k2" || widths == "k3" || widths == "k4") widths_fixed(widths[1] - '0', 1, W);
            else if (widths == "mix") widths_fixed(-1, 1, W);
            else if (!widths.compare(0, 4, "mod:")) widths_fixed(0, std::max(1, atoi(widths.c_str() + 4)), W);
            else { printf("--widths home|k2|k3|k4|mix|mod:N\n"); return 2; }
            need += layer_need(W);
        }
        arena.begin(need);
        for (int l = 0; l < hold; ++l) build_layer(All[l], Ws[l], 1000u + 17u * l, !nowide, !nowide && !nowide2, mem, arena);
    }
    std::vector<SynLayer> Ls(All.begin() + window, All.begin() + window + nlay);   // the benched window (same tensors)
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    {
        size_t fr = 0, to = 0;
        CK(cudaMemGetInfo(&fr, &to));
        double wb = 0, hb = 0;
        for (auto& L : Ls) wb += L.o1 + L.o2;
        for (auto& L : All) hb += L.o1 + L.o2;
        printf("layers %d (model layers %d..%d), trellis weights %.2f GB, fc1 %s fc2 %s, device memory in use %.1f GB\n", nlay, L0 + window,
               L0 + window + nlay - 1, wb / 1e9, nowide ? "narrow" : "wide (K3)", nowide || nowide2 ? "narrow" : "wide (K3)", (to - fr) / 1e9);
        size_t gmin = 0, grec = 0;
        g6vmm::granularity(0, gmin, grec);
        printf("alloc %s%s: %d layers held (%.2f GB of trellis weights), benched from held layer %d; VMM granularity min %zu KB rec %zu KB\n",
               arena.mode.c_str(), arena.mode == "vmm" ? (":" + std::to_string((arena.chunk ? arena.chunk : grec) >> 20) + " MB chunks").c_str() : "",
               hold, hb / 1e9, window, gmin >> 10, grec >> 10);
    }

    // route sources
    std::vector<J> rlines;           // GR8_ROUTE_LOG lines
    if (!routes.empty())
    {
        const std::string t = slurp(routes.c_str());
        size_t s = 0;
        while (s < t.size())
        {
            size_t e = t.find('\n', s);
            if (e == std::string::npos) e = t.size();
            const std::string line = t.substr(s, e - s);
            s = e + 1;
            J v;
            const char* p = line.c_str();
            if (jparse(p, v) && v.t == J::OBJ && v.get("L") && v.get("M")) rlines.push_back(std::move(v));
        }
        std::map<int, int> byM;
        for (auto& v : rlines) byM[static_cast<int>(v.get("M")->n)]++;
        printf("routes: %s, %zu main steps (", routes.c_str(), rlines.size());
        for (auto& kv : byM) printf(" M%d:%d", kv.first, kv.second);
        printf(" )\n");
    }
    struct XLine { int M; std::vector<int> n; std::vector<std::vector<int>> ids; };   // "# s2_round ... main" lines
    std::vector<XLine> xlines;
    if (!exlog.empty())
    {
        const std::string t = slurp(exlog.c_str());
        size_t s = 0;
        while (s < t.size())
        {
            size_t e = t.find('\n', s);
            if (e == std::string::npos) e = t.size();
            const std::string line = t.substr(s, e - s);
            s = e + 1;
            if (line.find("# s2_round") == std::string::npos || line.find(" main ") == std::string::npos) continue;
            const size_t pm = line.find(" M "), px = line.find(" ex "), pi = line.find(" ids ");
            if (pm == std::string::npos || px == std::string::npos) continue;
            XLine x;
            x.M = atoi(line.c_str() + pm + 3);
            x.n = parse_list(line.substr(px + 4, pi == std::string::npos ? std::string::npos : pi - px - 4).c_str());
            if (pi != std::string::npos)     // a line with route ids: "ids a.b.c,d.e,..." per MoE layer
                for (const char* p = line.c_str() + pi + 5; *p && *p != ' ' && *p != '\n';)
                {
                    x.ids.emplace_back();
                    while (*p && *p != ',' && *p != ' ')
                    {
                        x.ids.back().push_back(atoi(p));
                        while (isdigit(static_cast<unsigned char>(*p))) ++p;
                        if (*p == '.') ++p;
                    }
                    if (*p == ',') ++p;
                }
            xlines.push_back(std::move(x));
        }
        size_t nid = 0;
        for (auto& x : xlines) nid += !x.ids.empty();
        printf("exlog: %s, %zu main steps (%zu with routed ids)\n", exlog.c_str(), xlines.size(), nid);
    }

    const size_t sb = bind::scratch_bytes();
    void* blk = alloc(sb);
    CK(cudaMemset(blk, 0, sb));
    const bind::Scratch sc = bind::scratch_at(blk);
    __half* x2 = static_cast<__half*>(alloc(static_cast<size_t>(G6_MX) * HID * 2));
    fill_f16(x2, static_cast<size_t>(G6_MX) * HID, 1.0f, 41u);
    float* dlogit_dummy = static_cast<float*>(alloc(static_cast<size_t>(G6_MX) * 256 * 4));
    float* bias = static_cast<float*>(alloc(256 * 4));
    CK(cudaMemset(bias, 0, 256 * 4));
    glue::RoutePlan* dplan = static_cast<glue::RoutePlan*>(alloc(sizeof(glue::RoutePlan)));
    CK(cudaMemset(dplan, 0, sizeof(glue::RoutePlan)));
    const size_t obytes = static_cast<size_t>(G6_MX) * HID * 2;
    uint8_t* out = static_cast<uint8_t*>(alloc(obytes));
    unsigned* sink = static_cast<unsigned*>(alloc(4096));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    bind::K1Ovl O[3];
    for (int v = 1; v < 3; ++v)
    {
        O[v].mode = v;
        CK(cudaStreamCreateWithFlags(&O[v].side, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&O[v].fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&O[v].join, cudaEventDisableTiming));
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    // ---------------------------------------------------------------- stream floor (once)
    double stream_gbs = 0;
    if (tlb == 2) Ms.clear();               // --tlb only
    if (!ncu && !check_only && !Ms.empty())
    {
        // up to 4 GB of the layers' fc1 tensors, one persistent read per tensor, cold (each > the L2)
        std::vector<std::pair<const uint4*, size_t>> bufs;
        double tot = 0;
        for (int l = 0; l < nlay && tot < 4e9; ++l) { bufs.emplace_back(reinterpret_cast<const uint4*>(Ls[l].fc1), Ls[l].o1 / 16); tot += Ls[l].o1; }
        const int nsm = pr.multiProcessorCount;
        const int grids[3] = {4, 8, 16}, blocks[2] = {256, 512};
        std::string best;
        for (int g : grids)
            for (int b : blocks)
            {
                std::vector<double> t;
                for (int r = 0; r < 3; ++r)
                {
                    CK(cudaEventRecord(e0, st));
                    for (auto& bu : bufs) k_stream<<<nsm * g, b, 0, st>>>(bu.first, bu.second, sink);
                    CK(cudaEventRecord(e1, st));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    t.push_back(ms);
                }
                std::sort(t.begin(), t.end());
                const double gbs = tot / (t[1] * 1e-3) / 1e9;
                if (gbs > stream_gbs) { stream_gbs = gbs; best = std::to_string(nsm * g) + "x" + std::to_string(b); }
            }
        CK(cudaGetLastError());
        printf("stream floor: %.1f GB/s (plain 16 B ld.global.nc over %.2f GB of cold weights, best grid %s)\n", stream_gbs, tot / 1e9, best.c_str());
    }

    for (int M : Ms)
    {
        // ---------------------------------------------------------------- plans
        const int ncall = steps * nlay;
        std::vector<std::vector<int>> rowsel(ncall);        // [call] rows [M][8]
        std::vector<std::vector<int>> distinct(ncall);
        std::vector<int> rsel, xsel;
        for (int i = 0; i < static_cast<int>(rlines.size()); ++i) if (static_cast<int>(rlines[i].get("M")->n) == M) rsel.push_back(i);
        for (int i = 0; i < static_cast<int>(xlines.size()); ++i) if (xlines[i].M == M) xsel.push_back(i);
        const bool xids = !xsel.empty() && !xlines[xsel[0]].ids.empty();
        const char* src = !rsel.empty() ? "routes (served ids)" : xids ? "exlog (served ids)" : !xsel.empty() ? "exlog (served counts, uniform ids)"
                                                                                                         : "served means (uniform ids)";
        if (!routes.empty() && rsel.empty()) printf("M=%d: no route-log lines with M=%d: %s\n", M, M, src);
        for (int s = 0; s < steps; ++s)
            for (int l = 0; l < nlay; ++l)
            {
                const int c = s * nlay + l;
                std::mt19937 rng(static_cast<uint32_t>(seed * 1000003 + M * 7919 + c));
                std::vector<int> D;
                if (!rsel.empty())
                {
                    const J& v = rlines[rsel[static_cast<size_t>(s) * rsel.size() / steps]];
                    const J* ids = v.get("L")->get(std::to_string(L0 + l));
                    if (ids) for (const J& x : ids->a) D.push_back(static_cast<int>(x.n));
                }
                const XLine* xl0 = xsel.empty() ? nullptr : &xlines[xsel[static_cast<size_t>(s) * xsel.size() / steps]];
                if (D.empty() && xl0 && l < static_cast<int>(xl0->ids.size())) D = xl0->ids[l];
                if (D.empty())
                {
                    int d;
                    if (xl0)
                    {
                        const auto& xl = xl0->n;
                        d = l < static_cast<int>(xl.size()) ? xl[l] : static_cast<int>(lround(served_mean(M)));
                    }
                    else
                    {
                        const double m = served_mean(M);
                        d = static_cast<int>(m) + (std::uniform_real_distribution<double>(0, 1)(rng) < m - floor(m));
                    }
                    D = random_distinct(std::min(d, 8 * M), rng);
                }
                rows_from_distinct(D, M, rng, rowsel[c]);
                std::vector<int> u(rowsel[c]);
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                distinct[c] = u;
            }
        // logits [call][M][256]: the row's 8 experts at 3.0 - 0.25 k, the rest at -4 (sigmoid 0.78..0.95 vs 0.018)
        std::vector<float> hl(static_cast<size_t>(ncall) * M * 256, -4.f);
        for (int c = 0; c < ncall; ++c)
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < 8; ++k) hl[(static_cast<size_t>(c) * M + m) * 256 + rowsel[c][m * 8 + k]] = 3.f - 0.25f * k;
        float* dlog = static_cast<float*>(alloc(hl.size() * 4));
        CK(cudaMemcpy(dlog, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
        auto logits_of = [&](int c) { return dlog + static_cast<size_t>(c) * M * 256; };

        // bytes per call, and the read-floor chunk lists (fc1 ranges, then fc2 ranges)
        double bf1 = 0, bf2 = 0, bvec = 0, bk4 = 0, nd = 0, slots = 0;
        int dmin = 1 << 30, dmax = 0;
        std::vector<Chunk> ch;
        std::vector<int> ch_off(ncall + 1), ch_mid(ncall);
        for (int c = 0; c < ncall; ++c)
        {
            const SynLayer& L = Ls[c % nlay];
            ch_off[c] = static_cast<int>(ch.size());
            std::vector<int> ex(distinct[c]);
            ex.push_back(256);
            auto add = [&](const uint8_t* p, size_t n) {
                for (size_t o = 0; o < n; o += 32768) ch.push_back(Chunk{p + o, static_cast<unsigned>(std::min<size_t>(32768, n - o)), 0});
            };
            for (int e : ex) add(L.fc1 + L.tab[e * 6 + 0], L.f1b[e]);
            ch_mid[c] = static_cast<int>(ch.size());
            for (int e : ex) add(L.fc2 + L.tab[e * 6 + 1], L.f2b[e]);
            for (int e : ex)
            {
                bf1 += L.f1b[e]; bf2 += L.f2b[e]; bvec += g6nr::EB * VEC_ROW;
                if (e < 256 && L.k4[e]) bk4 += L.f1b[e] + L.f2b[e];
            }
            const int d = static_cast<int>(distinct[c].size());
            nd += d; dmin = std::min(dmin, d); dmax = std::max(dmax, d);
            std::vector<int> cnt(256, 0);
            for (int e : rowsel[c]) cnt[e]++;
            for (int e : distinct[c]) slots += (cnt[e] + 7) / 8;
        }
        ch_off[ncall] = static_cast<int>(ch.size());
        Chunk* dch = static_cast<Chunk*>(alloc(ch.size() * sizeof(Chunk)));
        CK(cudaMemcpy(dch, ch.data(), ch.size() * sizeof(Chunk), cudaMemcpyHostToDevice));
        bf1 /= ncall; bf2 /= ncall; bvec /= ncall; bk4 /= ncall; nd /= ncall; slots /= ncall;
        double bsh = 0;
        for (int c = 0; c < ncall; ++c) bsh += Ls[c % nlay].f1b[256] + Ls[c % nlay].f2b[256];
        bsh /= ncall;
        const double bcall = bf1 + bf2 + bvec, brouter = 16.0 * 384 * 512;
        printf("\nM=%d: %d calls (%d steps x %d layers), routing from %s\n", M, ncall, steps, nlay, src);
        printf("M=%d: distinct routed/layer %.2f (min %d max %d), slots %.2f, K4 share of routed bytes %.1f%%\n", M, nd, dmin, dmax, slots,
               100 * bk4 / (bf1 + bf2 - bsh));
        printf("M=%d: MB/call %.2f = fc1 %.2f + fc2 %.2f (shared %.2f incl.) + vectors %.2f; router GEMV %.2f (chain only)\n", M, bcall / 1e6,
               bf1 / 1e6, bf2 / 1e6, bsh / 1e6, bvec / 1e6, brouter / 1e6);
        {
            // distinct GPU pages the calls' weight tiles touch, at 4 KB / 64 KB / 2 MB pages (the layers of a step are
            // disjoint, so a step touches nlay x the per-call mean), against the held weights' pages
            const int lg[3] = {12, 16, 21};
            double pc[3] = {0, 0, 0}, ph[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k)
            {
                for (int c = 0; c < ncall; ++c)
                {
                    std::vector<uintptr_t> pg;
                    for (int i = ch_off[c]; i < ch_off[c + 1]; ++i)
                    {
                        const uintptr_t a = reinterpret_cast<uintptr_t>(ch[i].p);
                        for (uintptr_t q = a >> lg[k]; q <= (a + ch[i].bytes - 1) >> lg[k]; ++q) pg.push_back(q);
                    }
                    std::sort(pg.begin(), pg.end());
                    pc[k] += std::unique(pg.begin(), pg.end()) - pg.begin();
                }
                pc[k] /= ncall;
                for (auto& L : All)
                    for (auto r : {std::make_pair(reinterpret_cast<uintptr_t>(L.fc1), L.o1), std::make_pair(reinterpret_cast<uintptr_t>(L.fc2), L.o2)})
                        ph[k] += ((r.first + r.second - 1) >> lg[k]) - (r.first >> lg[k]) + 1;
            }
            printf("M=%d: pages touched per call 4K %.0f / 64K %.0f / 2M %.1f, per step (x%d layers) %.0f / %.0f / %.0f; held weights %.0f / %.0f / %.0f pages\n",
                   M, pc[0], pc[1], pc[2], nlay, pc[0] * nlay, pc[1] * nlay, pc[2] * nlay, ph[0], ph[1], ph[2]);
        }

        // ---------------------------------------------------------------- bindings
        const int fuse_engine = M == 1 ? 1 : 0;    // bind.cuh make_k1 policy for these layers (no 2-bit units, trellis shared)
        auto bind_all = [&](int fuse, std::vector<bind::K1>& ks) {
            ks.assign(nlay, bind::K1());
            for (int l = 0; l < nlay; ++l)
            {
                bind::Tensor sh[3];
                if (const char* e = bind::make_k1(Ls[l].T, M, sc, x2, sc.xf_x2, bind::plan_ptrs(dplan), out, 1, ks[l], sh, fuse, split, k1v))
                { printf("M=%d fuse=%d split=%d make_k1: %s\n", M, fuse, split, e); exit(2); }
            }
        };
        std::vector<bind::Gv> rgv(nlay);
        for (int l = 0; l < nlay; ++l)
            if (const char* e = bind::make_gv(Ls[l].router, bind::BF16W, 2, 16, M, sc.xb_x2, dlogit_dummy, gv::O_F32, rgv[l]))
            { printf("make_gv: %s\n", e); return 2; }
        bind::Xfrag xf;
        xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;

        enum Kind { K_K1, K_SPLIT, K_CHAIN, K_READ1, K_READ2 };
        struct Arm { std::string name; Kind kind; std::vector<bind::K1> ks; int stage = 3, ovl = 0; bool exact_arm = false; };
        std::vector<Arm> arms;
        std::vector<bind::K1> keng, kother;
        bind_all(fuse_engine, keng);
        bind_all(1 - fuse_engine, kother);
        arms.push_back(Arm{"k1", K_K1, keng});
        printf("M=%d bind: split_m %d fuse %d KS1 %d WG %d NS2 %d (fc2 grid %d)\n", M, split, keng[0].fuse, keng[0].a.KS1, keng[0].a.WG, keng[0].a.NS2, keng[0].g2);
        if (!ncu)
        {
            for (int sg = 0; sg < 3; ++sg)
            {
                Arm a{sg == 0 ? "split:plan" : sg == 1 ? (fuse_engine ? "split:+fused" : "split:+fc1") : (fuse_engine ? "split:+fused" : "split:+fc2"),
                      K_SPLIT, keng};
                a.stage = sg;
                if (!(fuse_engine && sg == 2)) arms.push_back(a);
            }
            for (int v : {1, 0, 2})
            {
                Arm a{"chain-ovl" + std::to_string(v), K_CHAIN, keng};
                a.ovl = v;
                a.exact_arm = true;
                if (v == 1 || all_arms == 1) arms.push_back(a);
            }
            if (all_arms == 1)
            {
                arms.push_back(Arm{fuse_engine ? "separate" : "fused", K_K1, kother, 3, 0, true});
                // fc1 CTA shapes with the same KS1 * WG (separate fc1 launches only)
                const std::vector<bind::K1>& ksep = fuse_engine ? kother : keng;
                const int P = ksep[0].a.KS1 * ksep[0].a.WG;
                for (int wg : {1, 2, 4})
                {
                    if (P % wg || (P / wg) * wg != P || wg == ksep[0].a.WG) continue;
                    const int ks1 = P / wg, KSU = KSH / ks1;
                    if (KSU % (8 * wg) || KSU % 8) continue;
                    Arm a{"fc1 " + std::to_string(ks1) + "x" + std::to_string(wg) + (fuse_engine ? " (sep)" : ""), K_K1, ksep, 3, 0, true};
                    size_t sm_max = 0;
                    for (auto& k : a.ks)
                    {
                        k.a.KS1 = ks1; k.a.WG = wg;
                        const int nsh = g6mx::tiles(M);
                        k.g1 = (k.a.n_slots + nsh) * k.a.B * 2 * ks1;          // trellis shared expert: no s_* CTAs
                        k.b1 = 256 * wg;
                        k.sm1 = k.a.xc ? static_cast<size_t>(M) * (KSH / ks1 * 16 + 8) * 2 : static_cast<size_t>(KSH / ks1) * 256;
                        const size_t w1 = static_cast<size_t>(8 * wg) * 1024 * g6mx::tiles(M);
                        if (k.sm1 < w1) k.sm1 = w1;
                        if (k.sm1 > sm_max) sm_max = k.sm1;
                    }
                    // the fc1 kernels' dynamic smem attribute is KSH * 256 (bind.cuh); the engine refuses larger shapes
                    if (sm_max > static_cast<size_t>(KSH) * 256)
                    {
                        printf("  M=%d: skip arm '%s': fc1 smem %zu B > %d B (KSH * 256, the engine refuses this shape)\n", M,
                               a.name.c_str(), sm_max, KSH * 256);
                        continue;
                    }
                    arms.push_back(a);
                }
            }
            if (!tab.empty())
            {
                // as on a launch: the table in the settings (both readers of GR8_K1_SPLIT see it), read by make_k1
                std::vector<bind::K1> kt, kto;
                k1v.split = tab;
                bind::k1_split_env(tab.c_str(), k1v);
                bind_all(fuse_engine, kt);
                bind_all(1 - fuse_engine, kto);
                k1v.split.reset();
                bind::k1_split_env("", k1v);
                arms.push_back(Arm{"tab", K_K1, kt, 3, 0, true});
                arms.push_back(Arm{fuse_engine ? "tab separate" : "tab fused", K_K1, kto, 3, 0, true});
                const std::vector<bind::K1>& ksep = fuse_engine ? kto : kt;      // the route overlap runs separate launches
                for (int v : {1, 2})
                {
                    Arm a{"tab chain-ovl" + std::to_string(v), K_CHAIN, ksep, 3, v, true};
                    arms.push_back(a);
                }
                printf("M=%d tab %s: fuse %d KS1 %d WG %d NS2 %d | other form fuse %d KS1 %d WG %d NS2 %d\n", M, tab.c_str(), kt[0].fuse,
                       kt[0].a.KS1, kt[0].a.WG, kt[0].a.NS2, kto[0].fuse, kto[0].a.KS1, kto[0].a.WG, kto[0].a.NS2);
            }
            if (all_arms == 1 || all_arms == 3)
            {
                // GR8_K1_TLBW: fc1 prefetches the routed slots' translations (64 KB and 2 MB granules); and in the served
                // route-overlap form (GR8_ROUTE_OVL=1, chain-ovl1)
                for (int kb : {64, 2048})
                {
                    Arm a{"tlbw" + std::to_string(kb), K_K1, keng, 3, 0, true};
                    for (auto& k : a.ks) k.a.tlbw = 10 + __builtin_ctz(kb);
                    arms.push_back(a);
                }
                Arm a{"chain-ovl1 tlbw2048", K_CHAIN, keng, 3, 1, true};
                for (auto& k : a.ks) k.a.tlbw = 21;
                arms.push_back(a);
            }
            if (all_arms == 1 || all_arms == 2)
            {
                // fc2 n-splits (divisors of KSH = 384), on the engine's form
                for (int ns : {1, 2, 4, 8, 16, 32})
                {
                    if (ns == keng[0].a.NS2) continue;
                    Arm a{"ns2=" + std::to_string(ns), K_K1, keng, 3, 0, true};
                    for (auto& k : a.ks)
                    {
                        k.a.NS2 = ns;
                        k.g2 = (k.a.n_slots + g6mx::tiles(M)) * k.a.B * ns;
                        if (k.fuse) k.gf = bind::k1_fused_grid(k.g1 + k.g2, k.smf, k.e4, k.k2, k.sh);
                    }
                    arms.push_back(a);
                }
            }
            arms.push_back(Arm{"read1", K_READ1, {}});   // (all arm sets)
            if (all_arms == 1) arms.push_back(Arm{"read2", K_READ2, {}});
        }

        auto emit = [&](const Arm& a, int c) {
            const int l = c % nlay;
            switch (a.kind)
            {
            case K_K1:
                glue::k_router_plan<<<1, 256, 0, st>>>(logits_of(c), bias, M, dplan);
                bind::launch_k1(a.ks[l], st);
                break;
            case K_SPLIT:
                glue::k_router_plan<<<1, 256, 0, st>>>(logits_of(c), bias, M, dplan);
                if (a.stage >= 1) { if (a.ks[l].fuse) bind::launch_k1_fused(a.ks[l], st); else bind::launch_k1_fc1(a.ks[l], st); }
                if (a.stage >= 2 && !a.ks[l].fuse) bind::launch_k1_fc2(a.ks[l], st);
                break;
            case K_CHAIN:
            {
                bind::launch(xf, st);
                const bool ovl = bind::k1_ovl_on(a.ks[l], O[a.ovl]);
                if (ovl) bind::k1_ovl_fork(a.ks[l], O[a.ovl], st);
                bind::launch(rgv[l], st);
                glue::k_router_plan<<<1, 256, 0, st>>>(logits_of(c), bias, M, dplan);
                if (ovl) bind::launch_k1_ovl(a.ks[l], O[a.ovl], st);
                else bind::launch_k1(a.ks[l], st);
                break;
            }
            case K_READ1:
                k_seg_read<<<ch_off[c + 1] - ch_off[c], 256, 0, st>>>(dch + ch_off[c], sink);
                break;
            case K_READ2:
                k_seg_read<<<ch_mid[c] - ch_off[c], 256, 0, st>>>(dch + ch_off[c], sink);
                k_seg_read<<<ch_off[c + 1] - ch_mid[c], 256, 0, st>>>(dch + ch_mid[c], sink);
                break;
            }
        };

        if (ncu)
        {
            for (int c = 0; c < std::min(ncu, ncall); ++c) emit(arms[0], c);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            printf("ncu: %d k1 calls launched (MB/call %.2f: sum lts__t_sectors_lookup_miss.sum x 32 B over k1_tr3_* / calls)\n", std::min(ncu, ncall), bcall / 1e6);
            continue;
        }

        // ---------------------------------------------------------------- exactness
        const int nchk = std::min(ncheck, ncall);
        const size_t ob = static_cast<size_t>(M) * HID * 2;
        std::vector<uint8_t> ref(ob * nchk), got(ob * nchk);
        uint8_t* outs = static_cast<uint8_t*>(alloc(ob * nchk));
        auto plain = [&](const Arm& a, std::vector<uint8_t>& h) {
            for (int c = 0; c < nchk; ++c)
            {
                CK(cudaMemsetAsync(out, 0xff, obytes, st));
                emit(a, c);
                CK(cudaMemcpyAsync(h.data() + ob * c, out, ob, cudaMemcpyDeviceToHost, st));
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
        };
        auto graphed = [&](const Arm& a, std::vector<uint8_t>& h) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaMemsetAsync(outs, 0xff, ob * nchk, st));
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            for (int c = 0; c < nchk; ++c)
            {
                emit(a, c);
                CK(cudaMemcpyAsync(outs + ob * c, out, ob, cudaMemcpyDeviceToDevice, st));
            }
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            for (int r = 0; r < 2; ++r) CK(cudaGraphLaunch(ge, st));
            CK(cudaMemcpyAsync(h.data(), outs, ob * nchk, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
        };
        auto cmp = [&](const std::string& what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
            size_t nbad = 0, bad = 0;
            for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) { if (!nbad) bad = i; ++nbad; }
            printf("%s M=%d %s: %d calls x %d rows", nbad ? "FAIL" : "PASS", M, what.c_str(), nchk, M);
            if (nbad) printf(", %zu bytes differ, first at call %zu row %zu col %zu", nbad, bad / ob, (bad % ob) / (HID * 2), (bad % (HID * 2)) / 2);
            printf("\n");
            (nbad ? g_fail : g_pass)++;
            return nbad == 0;
        };
        plain(arms[0], ref);
        {
            size_t nz = 0, nonfin = 0;
            for (size_t i = 0; i < ref.size(); i += 2)
            {
                const uint16_t b = static_cast<uint16_t>(ref[i] | (ref[i + 1] << 8));
                nz += (b & 0x7fff) != 0;
                nonfin += (b & 0x7f80) == 0x7f80;
            }
            const bool ok = nonfin == 0 && nz > ref.size() / 4;
            printf("%s M=%d k1 reference output: %zu non-finite, %.1f%% nonzero\n", ok ? "PASS" : "FAIL", M, nonfin, 200.0 * nz / ref.size());
            (ok ? g_pass : g_fail)++;
            unsigned long long h = 1469598103934665603ull;
            for (uint8_t b : ref) { h ^= b; h *= 1099511628211ull; }
            printf("digest M=%d %016llx (k1 outputs of the first %d calls; equal across builds/runs = same bytes)\n", M, h, nchk);
        }
        plain(arms[0], got); cmp("k1 run twice", ref, got);
        graphed(arms[0], got); cmp("k1 (graph) == k1 (plain)", ref, got);
        std::vector<char> exact(arms.size(), 1);
        for (size_t i = 1; i < arms.size(); ++i)
        {
            const Arm& a = arms[i];
            if (!a.exact_arm) continue;
            plain(a, got);
            exact[i] = cmp(a.name + " == k1", ref, got);
            if (a.kind == K_CHAIN && a.ovl == 1) { graphed(a, got); exact[i] = cmp(a.name + " (graph) == k1", ref, got) && exact[i]; }
            if (a.name == "tab") { graphed(a, got); exact[i] = cmp(a.name + " (graph) == k1", ref, got) && exact[i]; }
        }
        // --sess R: the batched rows == each session's solo step (R rows, split 0, the engine's fuse form for R rows)
        if (sess > 0 && (M % sess || M == sess)) printf("M=%d: --sess %d skipped (M is not a multiple of R above R)\n", M, sess);
        else if (sess > 0)
        {
            const int nsess = M / sess;
            std::vector<bind::K1> solo(static_cast<size_t>(nlay) * nsess);
            for (int j = 0; j < nsess; ++j)
                for (int l = 0; l < nlay; ++l)
                {
                    bind::Tensor sh[3];
                    if (const char* e = bind::make_k1(Ls[l].T, sess, sc, x2 + static_cast<size_t>(j) * sess * HID, sc.xf_x2, bind::plan_ptrs(dplan),
                                                      out + static_cast<size_t>(j) * sess * HID * 2, 1, solo[static_cast<size_t>(j) * nlay + l], sh, sess == 1, 0, k1v))
                    { printf("M=%d solo R=%d make_k1: %s\n", M, sess, e); exit(2); }
                }
            for (int c = 0; c < nchk; ++c)
            {
                CK(cudaMemsetAsync(out, 0xff, obytes, st));
                for (int j = 0; j < nsess; ++j)
                {
                    glue::k_router_plan<<<1, 256, 0, st>>>(logits_of(c) + static_cast<size_t>(j) * sess * 256, bias, sess, dplan);
                    bind::launch_k1(solo[static_cast<size_t>(j) * nlay + c % nlay], st);
                }
                CK(cudaMemcpyAsync(got.data() + ob * c, out, ob, cudaMemcpyDeviceToHost, st));
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            cmp("k1 == " + std::to_string(nsess) + " solo sessions of " + std::to_string(sess) + " rows (NS2 " + std::to_string(solo[0].a.NS2) + ")", ref, got);
        }

        if (check_only) { fflush(stdout); continue; }

        // ---------------------------------------------------------------- bench
        auto capture = [&](const Arm& a) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            for (int c = 0; c < ncall; ++c) emit(a, c);
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            CK(cudaGraphDestroy(g));
            return ge;
        };
        auto time1 = [&](cudaGraphExec_t ge) {
            CK(cudaGraphLaunch(ge, st));                      // warm (instruction caches, TLB of the scratch); weights stay cold
            CK(cudaStreamSynchronize(st));
            float ms = 0;
            CK(cudaEventRecord(e0, st));
            CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
            return 1000.0 * ms / ncall;
        };
        const int NA = static_cast<int>(arms.size());
        std::vector<cudaGraphExec_t> ge(NA);
        for (int i = 0; i < NA; ++i) ge[i] = capture(arms[i]);
        std::vector<std::vector<double>> t(NA);
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < NA; ++i) t[i].push_back(time1(ge[i]));
        std::vector<double> med(NA);
        for (int i = 0; i < NA; ++i)
        {
            std::sort(t[i].begin(), t[i].end());
            med[i] = t[i][t[i].size() / 2];
            CK(cudaGraphExecDestroy(ge[i]));
        }
        CK(cudaGetLastError());
        auto find = [&](const std::string& n) { for (int i = 0; i < NA; ++i) if (arms[i].name == n) return i; return -1; };
        const double k1us = med[0], gbs = bcall / (k1us * 1e-6) / 1e9;
        printf("bench M=%d: us per call = graph of %d calls / %d, median of %d (min..max)\n", M, ncall, ncall, reps);
        for (int i = 0; i < NA; ++i)
        {
            const Arm& a = arms[i];
            const double b = a.kind == K_CHAIN ? bcall + brouter : a.kind == K_SPLIT ? 0 : bcall;
            printf("bench M=%d %-16s %-5s us/call=%7.1f (%7.1f..%7.1f) vs k1 %+6.1f us (%+5.1f%%) = %+.2f ms/step@75",
                   M, a.name.c_str(), !a.exact_arm ? "" : exact[i] ? "exact" : "DIFF", med[i], t[i].front(), t[i].back(), med[i] - k1us,
                   100 * (med[i] - k1us) / k1us, 75 * (med[i] - k1us) / 1e3);
            if (b > 0) printf("  GB/s=%.1f", b / (med[i] * 1e-6) / 1e9);
            printf("\n");
        }
        const int ip = find("split:plan"), i1 = find(fuse_engine ? "split:+fused" : "split:+fc1"), i2 = find("split:+fc2");
        if (ip >= 0 && i1 >= 0)
        {
            const double plan = med[ip], f1 = med[i1] - med[ip];
            if (fuse_engine)
                printf("split M=%d: plan %.1f us | fused fc1+fc2 %.1f us (%.2f MB, %.1f GB/s) | fin %.1f us | k1 %.1f us = %.2f ms/step@75\n", M, plan, f1,
                       (bf1 + bf2) / 1e6, (bf1 + bf2) / (f1 * 1e-6) / 1e9, k1us - med[i1], k1us, 75 * k1us / 1e3);
            else
            {
                const double f2 = med[i2] - med[i1];
                printf("split M=%d: plan %.1f us | fc1 %.1f us (%.2f MB, %.1f GB/s) | fc2 %.1f us (%.2f MB, %.1f GB/s) | fin %.1f us | k1 %.1f us = %.2f ms/step@75\n",
                       M, plan, f1, bf1 / 1e6, bf1 / (f1 * 1e-6) / 1e9, f2, bf2 / 1e6, bf2 / (f2 * 1e-6) / 1e9, k1us - med[i2], k1us, 75 * k1us / 1e3);
            }
        }
        const int ir = find("read1");
        const double rgbs = ir >= 0 ? bcall / (med[ir] * 1e-6) / 1e9 : 0;
        int ibest = -1;
        for (int i = 1; i < NA; ++i)
            if (arms[i].exact_arm && exact[i] && arms[i].kind == K_K1 && (ibest < 0 || med[i] < med[ibest])) ibest = i;
        printf("verdict M=%d: k1 %.1f us/call, %.1f GB/s over plan+fc1+fc2+fin = %.1f%% of the same-bytes read (%.1f GB/s) and %.1f%% of the stream floor (%.1f GB/s)",
               M, k1us, gbs, rgbs > 0 ? 100 * gbs / rgbs : 0, rgbs, stream_gbs > 0 ? 100 * gbs / stream_gbs : 0, stream_gbs);
        const bool closed = gbs >= 240 || (rgbs > 0 && gbs >= 0.96 * rgbs);
        printf(" -> %s", closed ? "AT FLOOR (>= 240 GB/s or >= 96% of read1): close" : gbs >= 235 ? "NEAR (235-240 GB/s): knobs only if one is >= 1% faster and exact"
                                                                                                    : "BELOW (< 235 GB/s): see the exact arms");
        if (ibest >= 0)
            printf("; best exact runtime arm %s %+.1f us (%+.1f%%)", arms[ibest].name.c_str(), med[ibest] - k1us, 100 * (med[ibest] - k1us) / k1us);
        printf("\n");
        // NS2 pick: the fastest exact fc2 n-split incl. the bound one. Adopt only at >= 1 ms per step over the
        // 75 MoE layers (13.3 us per call) with every ns2 arm exact; else keep the release split.
        {
            int bn = keng[0].a.NS2, ndiff = 0;
            double bt = k1us;
            for (int i = 1; i < NA; ++i)
                if (!arms[i].name.compare(0, 4, "ns2="))
                {
                    if (!exact[i]) { ++ndiff; continue; }
                    if (med[i] < bt) { bt = med[i]; bn = arms[i].ks[0].a.NS2; }
                }
            const int rows = split ? split : M;
            const bool win = bn != keng[0].a.NS2 && ndiff == 0 && 75 * (bt - k1us) <= -1000;
            printf("ns2 pick M=%d class %d (split_m %d): bound NS2 %d %.1f us/call, best exact NS2 %d %.1f us/call %+.1f us (%+.2f%%) = %+.2f ms/step@75, %d ns2 arms DIFF -> %s",
                   M, rows == 1 ? 1 : rows <= 3 ? 3 : 5, split, keng[0].a.NS2, k1us, bn, bt, bt - k1us, 100 * (bt - k1us) / k1us, 75 * (bt - k1us) / 1e3, ndiff,
                   ndiff ? "KILL (not exact)" : win ? "ADOPT" : "KEEP (< 1 ms/step)");
            if (win) printf(" GR8_K1_SPLIT entry %d:/%d", M, bn);
            printf("\n");
        }
        fflush(stdout);
    }
    if (tlb)
    {
        std::vector<Region> rall, rlast;
        for (auto& L : All)
            for (Region r : {Region{reinterpret_cast<uintptr_t>(L.fc1), reinterpret_cast<uintptr_t>(L.fc1) + L.o1},
                             Region{reinterpret_cast<uintptr_t>(L.fc2), reinterpret_cast<uintptr_t>(L.fc2) + L.o2}})
            {
                rall.push_back(r);
                if (&L == &All.back()) rlast.push_back(r);
            }
        std::sort(rall.begin(), rall.end(), [](const Region& x, const Region& y) { return x.a < y.a; });
        std::sort(rlast.begin(), rlast.end(), [](const Region& x, const Region& y) { return x.a < y.a; });
        tlb_probe("held", rall, st);
        tlb_probe("last-layer", rlast, st);
    }
    for (void* p : mem) cudaFree(p);
    arena.end();
    printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
