// ptab_cold: study/151-table-vs-vmm. The two EL3 (#151) designs for paged routed-expert weights, measured COLD: a
// decode step that walks NL distinct synthetic MoE layers (ptab_test's layer shape: 256 routed experts at widths 3 / 4
// with every 5th in tr3.kmix, row 256 = the trellis shared expert, EB = g6nr::EB blocks per expert) in one CUDA graph,
// so a layer's piece table and its address translations are evicted by the other layers' weight streams before the
// layer comes round again, as in a real 78-layer step. The PR's ptab_test replays ONE layer, so everything stays hot.
//
// Arms (decode, per M; prefill at 1024 / 4096 rows):
//   pack       the pack layout (one contiguous tr3.fc1 / tr3.fc2 per layer; base + tr3.table offset)        baseline
//   img-tab    PR #267 mode 1: the table's entries point into the pack image                                   indirection only
//   scatter    PR #267 mode 2: every piece in its own slab, scrambled order                                    indirection + locality loss
//   bm-tab     block-major layout in a VMM range (group g = block g of every row, 2 MB-padded, the rank's EB
//              groups mapped into a range reserved for all 16), read through a table                           bm locality + indirection
//   bm-lin     the same range read as base + (tr[5] + b) * group stride + tr[0 | 1]: Matt's design, no table  bm locality only
//   img-tab+pf the image table, its 16 KB prefetched into L2 by the plan kernel of the same layer              mitigation
//   scatter+pf the scatter table, the same prefetch                                                            mitigation
//   img-tab+L2 the image tables of every layer in one buffer held in L2 with a persisting access-policy window mitigation
// --tlbw KB runs every arm with G6_K1_TLBW's translation warm at that granule.
//
//   ptab_cold [--layers 20] [--M 1,2,4,8,20] [--pf-M 1024,4096] [--sets 24] [--passes 4] [--rounds 7] [--tlbw KB]
//             [--no-prefill] [--no-decode] [--no-switch] [--first-block 4]
// Exit: 0 all checks passed, 1 a check failed, 2 CUDA error / usage.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../glue.cuh"
#include "../bind.cuh"
#include "../ptab.cuh"
#include "../bind_prefill.cuh"

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
#define CU(x)                                                                                                   \
    do                                                                                                          \
    {                                                                                                           \
        const CUresult r_ = (x);                                                                                \
        if (r_ != CUDA_SUCCESS)                                                                                 \
        {                                                                                                       \
            const char* s_ = nullptr; cuGetErrorString(r_, &s_);                                                \
            printf("CUDA driver error %s at %s:%d: %d %s\n", #x, __FILE__, __LINE__, (int)r_, s_ ? s_ : "?");    \
            exit(2);                                                                                            \
        }                                                                                                       \
    } while (0)

static int g_pass = 0, g_fail = 0;
constexpr size_t GRAN = size_t(2) << 20;          // the mapping granule GB10 reports (cuMemGetAllocationGranularity)
constexpr size_t UNIT = static_cast<size_t>(KSH) * 8 * 32;   // bytes of one fc1 matrix / fc2 block per bit of width

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

// ---------------------------------------------------------------- VMM: a block-major range (the rank's groups mapped
// into a range reserved for all EXP_BLOCKS groups)
struct BmRange
{
    CUdeviceptr base = 0;
    size_t gs = 0;                                  // group stride (bytes, a multiple of GRAN)
    int first = 0, n = 0;                           // groups [first, first + n) mapped
    std::vector<CUmemGenericAllocationHandle> h;    // one handle per mapped group
};
static CUmemAllocationProp bm_prop()
{
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;
    return p;
}
static void bm_map_group(BmRange& r, int g)
{
    const CUmemAllocationProp p = bm_prop();
    CUmemGenericAllocationHandle h;
    CU(cuMemCreate(&h, r.gs, &p, 0));
    CU(cuMemMap(r.base + static_cast<size_t>(g) * r.gs, r.gs, 0, h, 0));
    CUmemAccessDesc d = {};
    d.location = p.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(cuMemSetAccess(r.base + static_cast<size_t>(g) * r.gs, r.gs, &d, 1));
    r.h[g - r.first] = h;
}
static void bm_unmap_group(BmRange& r, int g)
{
    CU(cuMemUnmap(r.base + static_cast<size_t>(g) * r.gs, r.gs));
    CU(cuMemRelease(r.h[g - r.first]));
    r.h[g - r.first] = 0;
}
static BmRange bm_reserve(size_t group_bytes, int first, int n)
{
    BmRange r;
    r.gs = (group_bytes + GRAN - 1) / GRAN * GRAN;
    r.first = first; r.n = n;
    r.h.assign(n, 0);
    CU(cuMemAddressReserve(&r.base, r.gs * geom::EXP_BLOCKS, GRAN, 0, 0));
    for (int g = first; g < first + n; ++g) bm_map_group(r, g);
    return r;
}

// ---------------------------------------------------------------- the synthetic layer
struct SynLayer
{
    bind::LayerTensors T;             // the pack layout
    std::vector<long long> tab, kmix; // [257][6], [257]
    std::vector<long long> tab_bm;    // the block-major twin of tr3.table: [0] / [1] = offset inside a group, [5] = first global block
    long long* d_tab_bm = nullptr;
    double ebytes[257];
    bind::Tensor router;
    float* bias = nullptr;
    ptab::Table img, sct, bmt;        // mode 1, mode 2, and the table over the block-major range
    BmRange bm1, bm2;                 // block-major fc1 / fc2 ranges
    size_t fc1_bytes = 0, fc2_bytes = 0, grp1 = 0, grp2 = 0;   // pack bytes; block-major group bytes before padding
};

static void build_layer(SynLayer& L, int k4mod, uint32_t seed, int first_block, std::vector<void*>& mem)
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
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = bits; r[5] = first_block;
        long long km = 0;
        size_t f1 = 0;
        for (int b = 0; b < nb; ++b)
        {
            const int g = mixed ? 2 : bits, u = mixed ? 3 : bits;
            if (mixed) km |= (static_cast<long long>(g) << (8 * b)) | (static_cast<long long>(u) << (8 * b + 4));
            f1 += static_cast<size_t>(g + u) * UNIT;
        }
        L.kmix[e] = km;
        o1 += f1;
        o2 += static_cast<size_t>(nb) * UNIT * bits;
        L.ebytes[e] = static_cast<double>(f1) + static_cast<double>(nb) * UNIT * bits;
        row += nb;
    }
    L.fc1_bytes = o1; L.fc2_bytes = o2;
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
    if (!T.sh_trellis || !T.k2) { printf("synthetic layer: expected the trellis shared expert and 2-bit units\n"); exit(2); }
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

// The block-major copy of the layer: groups first_block .. first_block + EB - 1 of fc1 and fc2 in two VMM ranges, every
// row's pieces at the same offset in every group (a row's pieces all have one size: true of every shipped manifest).
static void build_bm(SynLayer& L, int first_block, cudaStream_t st, std::vector<void*>& mem)
{
    L.tab_bm = L.tab;
    size_t o1 = 0, o2 = 0;
    for (int e = 0; e < 257; ++e)
    {
        const long long* tr = &L.tab[static_cast<size_t>(e) * 6];
        long long* r = &L.tab_bm[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[5] = first_block;
        o1 += tr3_piece_fc1_bytes(L.kmix[e], tr, 0);
        o2 += tr3_piece_fc2_bytes(tr);
    }
    L.grp1 = o1; L.grp2 = o2;
    L.bm1 = bm_reserve(o1, first_block, g6nr::EB);
    L.bm2 = bm_reserve(o2, first_block, g6nr::EB);
    CK(cudaMalloc(&L.d_tab_bm, L.tab_bm.size() * 8));
    mem.push_back(L.d_tab_bm);
    CK(cudaMemcpy(L.d_tab_bm, L.tab_bm.data(), L.tab_bm.size() * 8, cudaMemcpyHostToDevice));
    // the pieces, after the wide reorder, piece by piece (one D2D copy each: what a switch's loader would do)
    const uint8_t* fc1 = static_cast<const uint8_t*>(L.T.t[bind::T_TR3_FC1].p);
    const uint8_t* fc2 = static_cast<const uint8_t*>(L.T.t[bind::T_TR3_FC2].p);
    for (int e = 0; e < 257; ++e)
    {
        const long long* tr = &L.tab[static_cast<size_t>(e) * 6];
        const long long* rb = &L.tab_bm[static_cast<size_t>(e) * 6];
        for (int b = 0; b < tr[3]; ++b)
        {
            const size_t f1 = tr3_piece_fc1_bytes(L.kmix[e], tr, b), f2 = tr3_piece_fc2_bytes(tr);
            const uint8_t* s1 = fc1 + ptab::unit_off(L.kmix[e], tr, 2 * b);
            const uint8_t* s2 = fc2 + tr[1] + static_cast<size_t>(b) * f2;
            uint8_t* d1 = reinterpret_cast<uint8_t*>(L.bm1.base) + static_cast<size_t>(rb[5] + b) * L.bm1.gs + rb[0];
            uint8_t* d2 = reinterpret_cast<uint8_t*>(L.bm2.base) + static_cast<size_t>(rb[5] + b) * L.bm2.gs + rb[1];
            CK(cudaMemcpyAsync(d1, s1, f1, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(d2, s2, f2, cudaMemcpyDeviceToDevice, st));
        }
    }
    CK(cudaStreamSynchronize(st));
    // the table over the block-major range (arm bm-tab)
    ptab::Table& t = L.bmt;
    t.rows = 257; t.eb = g6nr::EB; t.mode = ptab::IMAGE;
    t.host.assign(t.n(), nullptr);
    t.bytes.assign(t.n(), 0);
    for (int e = 0; e < 257; ++e)
    {
        const long long* tr = &L.tab[static_cast<size_t>(e) * 6];
        const long long* rb = &L.tab_bm[static_cast<size_t>(e) * 6];
        for (int b = 0; b < tr[3]; ++b)
        {
            t.host[t.slot(e, b)] = reinterpret_cast<const uint8_t*>(L.bm1.base) + static_cast<size_t>(rb[5] + b) * L.bm1.gs + rb[0];
            t.host[t.slot(e, b) + 1] = reinterpret_cast<const uint8_t*>(L.bm2.base) + static_cast<size_t>(rb[5] + b) * L.bm2.gs + rb[1];
            t.bytes[t.slot(e, b)] = tr3_piece_fc1_bytes(L.kmix[e], tr, b);
            t.bytes[t.slot(e, b) + 1] = tr3_piece_fc2_bytes(tr);
        }
    }
    CK(cudaMalloc(&t.dev, t.n() * sizeof(void*)));
    if (const char* e = ptab::upload(t, st)) { printf("%s\n", e); exit(2); }
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
static bool cmp_bytes(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, bool expect_equal = true)
{
    size_t nbad = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (!nbad) first = i; ++nbad; }
    const bool equal = nbad == 0 && a.size() == b.size(), ok = equal == expect_equal;
    printf("%s %s: %s", ok ? "PASS" : "FAIL", what, equal ? "identical" : "differ");
    if (!equal) printf(" (%zu bytes, first at %zu)", nbad, first);
    if (!expect_equal) printf(" [expected to differ]");
    printf("\n");
    (ok ? g_pass : g_fail)++;
    return ok;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// the arms
enum { A_PACK = 0, A_IMG, A_SCT, A_BMT, A_BML, A_IMGPF, A_SCTPF, A_IMGL2, NARMS };
static const char* arm_name[NARMS] = {"pack", "img-tab", "scatter", "bm-tab", "bm-lin", "img-tab+pf", "scatter+pf", "img-tab+L2"};
static bool arm_table(int v) { return v == A_IMG || v == A_SCT || v == A_BMT || v == A_IMGPF || v == A_SCTPF || v == A_IMGL2; }

int main(int argc, char** argv)
{
    std::vector<int> Ms = {1, 2, 4, 8, 20}, PMs = {1024, 4096};
    int sets = 24, nlay = 20, passes = 4, rounds = 7, k4mod = 4, tlbw = 0, prefill = 1, decode = 1, dosw = 1, first_block = 4;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { printf("missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--M") Ms = parse_list(nx());
        else if (a == "--pf-M") PMs = parse_list(nx());
        else if (a == "--sets") sets = atoi(nx());
        else if (a == "--layers") nlay = atoi(nx());
        else if (a == "--passes") passes = atoi(nx());
        else if (a == "--rounds") rounds = atoi(nx());
        else if (a == "--tlbw") tlbw = atoi(nx());
        else if (a == "--first-block") first_block = atoi(nx());
        else if (a == "--no-prefill") prefill = 0;
        else if (a == "--no-decode") decode = 0;
        else if (a == "--no-switch") dosw = 0;
        else { printf("unknown arg %s\n", a.c_str()); return 2; }
    }
    for (int M : Ms) if (M < 1 || M > G6_MX) { printf("M=%d outside 1..%d\n", M, G6_MX); return 2; }
    if (first_block < 0 || first_block + g6nr::EB > geom::EXP_BLOCKS) { printf("--first-block outside 0..%d\n", geom::EXP_BLOCKS - g6nr::EB); return 2; }
    if (tlbw && (tlbw & (tlbw - 1))) { printf("--tlbw must be a power of two (KB)\n"); return 2; }
    CK(cudaSetDevice(0));
    CK(bind::init());
    CU(cuInit(0));
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("ptab_cold: %s sm_%d%d %d SMs, clock attr %d MHz, L2 %d MB, G6_NR=%d EB=%d G6_MX=%d, layers %d, sets %d, passes %d, rounds %d, tlbw %d KB, first block %d\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount, clk / 1000, pr.l2CacheSize >> 20, g6nr::NR, g6nr::EB, G6_MX, nlay, sets, passes, rounds, tlbw, first_block);
    {
        size_t gmin = 0, grec = 0;
        const CUmemAllocationProp p = bm_prop();
        CU(cuMemGetAllocationGranularity(&gmin, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        CU(cuMemGetAllocationGranularity(&grec, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        printf("vmm granularity: min %zu KB, recommended %zu KB (the bench pads groups to %zu KB)\n", gmin >> 10, grec >> 10, GRAN >> 10);
        if (gmin > GRAN) { printf("granularity above the bench's GRAN\n"); return 2; }
    }

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::vector<void*> mem;
    std::vector<SynLayer> Ls(nlay);
    ptab::Pool pool;
    const auto t_build0 = std::chrono::steady_clock::now();
    for (int l = 0; l < nlay; ++l)
    {
        SynLayer& L = Ls[l];
        build_layer(L, k4mod, 1000u + 17u * l, first_block, mem);
        const bind::LayerTensors& T = L.T;
        auto bld = [&](ptab::Table& t, ptab::Mode m) {
            if (const char* e = ptab::build(t, T.t[bind::T_TR3_TAB].p, T.t[bind::T_TR3_KMIX].p, static_cast<const uint8_t*>(T.t[bind::T_TR3_FC1].p),
                                            static_cast<const uint8_t*>(T.t[bind::T_TR3_FC2].p), 257, g6nr::EB, m, pool, st))
            { printf("ptab::build: %s\n", e); exit(2); }
        };
        bld(L.img, ptab::IMAGE);
        bld(L.sct, ptab::SCATTER);
        build_bm(L, first_block, st, mem);
    }
    const double t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build0).count();
    // every layer's image table in ONE buffer (arm img-tab+L2: one persisting access-policy window over them)
    const size_t tab_bytes = Ls[0].img.n() * sizeof(void*);
    uint8_t* tabs_l2 = nullptr;
    CK(cudaMalloc(&tabs_l2, tab_bytes * nlay));
    mem.push_back(tabs_l2);
    for (int l = 0; l < nlay; ++l) CK(cudaMemcpy(tabs_l2 + tab_bytes * l, Ls[l].img.host.data(), tab_bytes, cudaMemcpyHostToDevice));
    {
        const SynLayer& L = Ls[0];
        printf("layer: pack fc1 %.1f MB + fc2 %.1f MB = %.1f MB; block-major group fc1 %.2f MB (padded %.1f MB, waste %.2f MB) x %d, fc2 %.2f MB (padded %.1f MB, waste %.2f MB) x %d; table %zu entries = %zu KB; scatter slabs %.1f MB\n",
               L.fc1_bytes / 1e6, L.fc2_bytes / 1e6, (L.fc1_bytes + L.fc2_bytes) / 1e6, L.grp1 / 1e6, L.bm1.gs / 1e6, (L.bm1.gs - L.grp1) / 1e6, g6nr::EB,
               L.grp2 / 1e6, L.bm2.gs / 1e6, (L.bm2.gs - L.grp2) / 1e6, g6nr::EB, L.img.n(), tab_bytes >> 10, L.sct.slab_bytes / 1e6);
        printf("pool: %.1f MB held in %zu chunks; build %.1f s\n", pool.held / 1e6, pool.chunks.size(), t_build);
    }
    auto alloc = [&](size_t n) { void* p; CK(cudaMalloc(&p, n)); mem.push_back(p); return p; };
    const size_t xb = static_cast<size_t>(G6_MX) * HID * 2;
    __half* x2 = static_cast<__half*>(alloc(xb));
    __half* x2sets = static_cast<__half*>(alloc(xb * sets));
    k_fill_x2<<<1024, 256>>>(x2sets, sets, G6_MX, 0.6f, 41u);
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
    uint8_t* outs = static_cast<uint8_t*>(alloc(obytes * nlay));
    CK(cudaDeviceSynchronize());
    {
        size_t fr = 0, to = 0;
        CK(cudaMemGetInfo(&fr, &to));
        printf("device memory in use %.2f GB of %.2f GB (%d layers x [pack + scatter + block-major] + scratch %.0f MB)\n", (to - fr) / 1e9, to / 1e9, nlay, sb / 1e6);
    }

    // the layer's tensors through arm v
    auto tens = [&](int l, int v) {
        bind::LayerTensors T = Ls[l].T;
        T.ptab = nullptr; T.bm1 = T.bm2 = 0;
        if (v == A_IMG || v == A_IMGPF) T.ptab = Ls[l].img.dev;
        else if (v == A_IMGL2) T.ptab = reinterpret_cast<const uint8_t* const*>(tabs_l2 + tab_bytes * l);
        else if (v == A_SCT || v == A_SCTPF) T.ptab = Ls[l].sct.dev;
        else if (v == A_BMT || v == A_BML)
        {
            T.t[bind::T_TR3_TAB].p = Ls[l].d_tab_bm;
            T.t[bind::T_TR3_FC1].p = reinterpret_cast<const void*>(Ls[l].bm1.base);
            T.t[bind::T_TR3_FC2].p = reinterpret_cast<const void*>(Ls[l].bm2.base);
            if (v == A_BMT) T.ptab = Ls[l].bmt.dev;
            else { T.bm1 = Ls[l].bm1.gs; T.bm2 = Ls[l].bm2.gs; }
        }
        return T;
    };
    struct Bound { bind::K1 k1; bind::Gv router; };
    auto make = [&](int M, int v, std::vector<Bound>& bs) -> const char* {
        bs.assign(nlay, Bound());
        for (int l = 0; l < nlay; ++l)
        {
            bind::Tensor sh[3];
            const bind::LayerTensors T = tens(l, v);
            if (const char* e = bind::make_k1(T, M, sc, x2, sc.xf_x2, bind::plan_ptrs(dplan), out, 1, bs[l].k1, sh, 1, 0)) return e;
            if (tlbw) bs[l].k1.a.tlbw = 10 + __builtin_ctz(static_cast<unsigned>(tlbw));
            if (const char* e = bind::make_gv(Ls[l].router, bind::BF16W, 2, 16, M, sc.xb_x2, logits, gv::O_F32, bs[l].router)) return e;
        }
        return nullptr;
    };
    // one MoE layer call: x2 copy, xfrag, router GEMV, plan (with the table prefetch in the +pf arms), K1
    auto call = [&](const std::vector<Bound>& bs, int M, int s, int l, int v) {
        CK(cudaMemcpyAsync(x2, x2sets + static_cast<size_t>(s) * G6_MX * HID, static_cast<size_t>(M) * HID * 2, cudaMemcpyDeviceToDevice, st));
        bind::Xfrag xf;
        xf.on = 1; xf.src = x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = sc.xf_x2; xf.bf16 = sc.xb_x2;
        bind::launch(xf, st);
        bind::launch(bs[l].router, st);
        const void* warm = (v == A_IMGPF || v == A_SCTPF) ? static_cast<const void*>(bs[l].k1.a.t_ptab) : nullptr;
        glue::k_router_plan<<<1, 256, 0, st>>>(logits, Ls[l].bias, M, dplan, 0u, nullptr, warm, warm ? static_cast<int>((tab_bytes + 127) / 128) : 0);
        bind::launch_k1(bs[l].k1, st);
    };
    // plain run over every layer (set l): out of layer l into h[l]
    auto plain = [&](const std::vector<Bound>& bs, int M, int v, std::vector<uint8_t>& h) {
        const size_t ob = static_cast<size_t>(M) * HID * 2;
        h.assign(ob * nlay, 0);
        for (int l = 0; l < nlay; ++l)
        {
            CK(cudaMemsetAsync(out, 0xff, obytes, st));
            call(bs, M, l % sets, l, v);
            CK(cudaMemcpyAsync(h.data() + ob * l, out, ob, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
    };
    // the persisting L2 window over the tables (arm img-tab+L2), applied to every kernel node of a captured graph
    auto apply_l2 = [&](cudaGraph_t g) {
        size_t n = 0;
        CK(cudaGraphGetNodes(g, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        CK(cudaGraphGetNodes(g, nodes.data(), &n));
        cudaLaunchAttributeValue v = {};
        v.accessPolicyWindow.base_ptr = tabs_l2;
        v.accessPolicyWindow.num_bytes = tab_bytes * nlay;
        v.accessPolicyWindow.hitRatio = 1.f;
        v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        v.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
        int applied = 0;
        for (cudaGraphNode_t nd : nodes)
        {
            cudaGraphNodeType ty;
            CK(cudaGraphNodeGetType(nd, &ty));
            if (ty != cudaGraphNodeTypeKernel) continue;
            if (cudaGraphKernelNodeSetAttribute(nd, cudaLaunchAttributeAccessPolicyWindow, &v) == cudaSuccess) ++applied;
            else cudaGetLastError();
        }
        return applied;
    };
    {
        // the persisting set-aside: the tables of every layer (and no more, so the weight stream keeps the rest of L2)
        const size_t want = std::min<size_t>(tab_bytes * nlay, static_cast<size_t>(pr.persistingL2CacheMaxSize));
        if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) != cudaSuccess) { cudaGetLastError(); printf("persisting L2 limit: not settable\n"); }
        size_t got = 0;
        cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize);
        printf("persisting L2 set-aside: %zu KB (tables of %d layers = %zu KB)\n", got >> 10, nlay, (tab_bytes * nlay) >> 10);
    }

    // ---------------------------------------------------------------- decode: every arm == pack, plain and graph
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = nlay * passes;
    if (decode)
    {
        printf("== decode check: every arm's output == the pack layout's, every layer (M = %d)\n", Ms[0]);
        const int M = Ms[0];
        std::vector<Bound> bs[NARMS];
        std::vector<uint8_t> ref, got;
        for (int v = 0; v < NARMS; ++v) if (const char* e = make(M, v, bs[v])) { printf("make %s: %s\n", arm_name[v], e); return 2; }
        plain(bs[A_PACK], M, A_PACK, ref);
        {
            size_t nz = 0;
            for (size_t i = 0; i + 1 < ref.size(); i += 2) nz += (ref[i] | (ref[i + 1] & 0x7f)) != 0;
            printf("%s pack output nonzero %.1f%%\n", nz > ref.size() / 4 ? "PASS" : "FAIL", 200.0 * nz / ref.size());
            (nz > ref.size() / 4 ? g_pass : g_fail)++;
        }
        for (int v = 1; v < NARMS; ++v)
        {
            plain(bs[v], M, v, got);
            cmp_bytes((std::string("decode M=") + std::to_string(M) + " " + arm_name[v] + " == pack (plain, all layers)").c_str(), ref, got);
        }
        // the block-major linear arm as a graph, replayed after a group was unmapped and remapped to NEW physical memory
        if (dosw)
        {
            printf("== graph validity across cuMemUnmap / cuMemMap (arm bm-lin, M = %d)\n", M);
            cudaGraph_t g;
            cudaGraphExec_t ge;
            const size_t ob = static_cast<size_t>(M) * HID * 2;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < nlay; ++l)
            {
                call(bs[A_BML], M, l % sets, l, A_BML);
                CK(cudaMemcpyAsync(outs + ob * l, out, ob, cudaMemcpyDeviceToDevice, st));
            }
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            CK(cudaGraphDestroy(g));
            auto replay = [&](std::vector<uint8_t>& h) {
                h.assign(ob * nlay, 0);
                CK(cudaMemsetAsync(outs, 0xff, obytes * nlay, st));
                CK(cudaGraphLaunch(ge, st));
                CK(cudaMemcpyAsync(h.data(), outs, ob * nlay, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaGetLastError());
            };
            std::vector<uint8_t> r0, r1;
            replay(r0);
            cmp_bytes("graph bm-lin replay == plain pack run", ref, r0);
            // layer 0, fc1 group first_block: save, unmap, map new memory, restore the bytes, replay
            SynLayer& L = Ls[0];
            const int g0 = first_block;
            uint8_t* save = static_cast<uint8_t*>(alloc(L.bm1.gs));
            uint8_t* at = reinterpret_cast<uint8_t*>(L.bm1.base) + static_cast<size_t>(g0) * L.bm1.gs;
            CK(cudaMemcpy(save, at, L.bm1.gs, cudaMemcpyDeviceToDevice));
            CK(cudaDeviceSynchronize());
            bm_unmap_group(L.bm1, g0);
            bm_map_group(L.bm1, g0);
            CK(cudaMemcpy(at, save, L.bm1.gs, cudaMemcpyDeviceToDevice));
            replay(r1);
            cmp_bytes("graph replay after the group was unmapped and remapped to new physical memory (same bytes) == before", r0, r1);
            // the new memory filled with another layer's group: the graph reads the new physical memory
            uint8_t* other = reinterpret_cast<uint8_t*>(Ls[1 % nlay].bm1.base) + static_cast<size_t>(g0) * Ls[1 % nlay].bm1.gs;
            CK(cudaMemcpy(at, other, L.bm1.gs, cudaMemcpyDeviceToDevice));
            replay(r1);
            cmp_bytes("graph replay with another layer's bytes mapped there differs from before", r0, r1, false);
            CK(cudaMemcpy(at, save, L.bm1.gs, cudaMemcpyDeviceToDevice));
            replay(r1);
            cmp_bytes("graph replay after the bytes were restored == before", r0, r1);
            CK(cudaGraphExecDestroy(ge));
        }
    }

    // ---------------------------------------------------------------- decode bench
    if (decode)
    {
        printf("== decode bench: us per MoE layer call = graph of %d x [x2 copy, xfrag, router GEMV, plan, K1] over %d layers in turn / %d; median of %d interleaved rounds; min..max\n",
               reps, nlay, reps, rounds);
        for (int M : Ms)
        {
            std::vector<Bound> bs[NARMS];
            bool ok = true;
            for (int v = 0; v < NARMS && ok; ++v)
                if (const char* e = make(M, v, bs[v])) { printf("M=%d make %s: %s\n", M, arm_name[v], e); ok = false; }
            if (!ok) continue;
            // the bytes a layer call reads (the routed experts of the set plus the shared row), averaged over the sets of layer 0
            double nex = 0, bytes = 0;
            for (int s = 0; s < sets; ++s)
            {
                call(bs[A_PACK], M, s, 0, A_PACK);
                glue::RoutePlan P;
                CK(cudaMemcpyAsync(&P, dplan, sizeof P, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::vector<int> ex(P.slot_expert, P.slot_expert + P.n_slots);
                std::sort(ex.begin(), ex.end());
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
                for (int i = 0; i < reps; ++i) call(bs[v], M, i % sets, i % nlay, v);
                CK(cudaStreamEndCapture(st, &g));
                if (v == A_IMGL2) apply_l2(g);
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
            cudaGraphExec_t g[NARMS];
            for (int v = 0; v < NARMS; ++v) g[v] = cap(v);
            std::vector<double> t[NARMS];
            for (int r = 0; r < rounds; ++r)
                for (int v = 0; v < NARMS; ++v) t[v].push_back(time(g[v]));
            double med[NARMS];
            for (int v = 0; v < NARMS; ++v)
            {
                med[v] = median(t[v]);
                CK(cudaGraphExecDestroy(g[v]));
            }
            CK(cudaGetLastError());
            for (int v = 0; v < NARMS; ++v)
            {
                std::vector<double> s = t[v];
                std::sort(s.begin(), s.end());
                printf("bench decode M=%2d %-11s form=%-8s experts=%.1f MB=%.2f us/layer=%7.1f (min %7.1f max %7.1f)  vs pack %+6.1f us (%+5.2f%%) = %+.2f ms/step @75\n",
                       M, arm_name[v], bs[v][0].k1.fuse ? "fused" : "separate", nex, bytes / 1e6, med[v], s.front(), s.back(), med[v] - med[A_PACK],
                       100 * (med[v] - med[A_PACK]) / med[A_PACK], 75 * (med[v] - med[A_PACK]) / 1e3);
            }
            fflush(stdout);
        }
    }

    // ---------------------------------------------------------------- prefill: check and bench, layers in turn
    if (prefill)
    {
        for (int M : PMs)
        {
            void* pblk;
            CK(cudaMalloc(&pblk, bindp::scratch_bytes(M)));
            CK(cudaMemset(pblk, 0, bindp::scratch_bytes(M)));
            const bindp::Scratch psc = bindp::scratch_at(pblk, M);
            __half* px = static_cast<__half*>(alloc(static_cast<size_t>(M) * HID * 2));
            fill_f16(px, static_cast<size_t>(M) * HID, 1.f, 77u + M);
            float* plog = static_cast<float*>(alloc(static_cast<size_t>(M) * 256 * 4));
            k_fill_f32<<<1024, 256>>>(plog, static_cast<size_t>(M) * 256, 1.f, 78u + M);
            CK(cudaGetLastError());
            uint16_t* pout = static_cast<uint16_t*>(alloc(static_cast<size_t>(M) * HID * 2));
            auto pf_make = [&](int l, int v, bindp::Expert& A, bindp::Expert& S) {
                const bind::LayerTensors D = tens(l, v);
                const bind::Tensor &tab = D.t[bind::T_TR3_TAB], &v2 = D.t[bind::T_TR3_V2];
                A = bindp::Expert();
                A.on = 1; A.M = M; A.maxseg = bindp::max_segments(M);
                A.logits = plog; A.bias = Ls[l].bias;
                A.tok_e = psc.tok_e; A.tok_w = psc.tok_w; A.cnt = psc.cnt;
                A.rr_row = psc.rr_row; A.rr_e = psc.rr_e; A.rr_sp = psc.rr_sp; A.rr_w = psc.rr_w; A.pos = psc.pos; A.segs = psc.segs; A.n_seg = psc.n_seg;
                A.t_tab = static_cast<const long long*>(tab.p);
                A.t_kmix = static_cast<const long long*>(D.t[bind::T_TR3_KMIX].p);
                A.k2 = D.k2; A.cb = D.cb; A.bs = 1;
                A.t_sg = static_cast<const __half*>(D.t[bind::T_TR3_SG].p);
                A.t_v1b = static_cast<const __half*>(D.t[bind::T_TR3_V1B].p);
                A.t_v3b = static_cast<const __half*>(D.t[bind::T_TR3_V3B].p);
                A.t_fc1 = static_cast<const uint8_t*>(D.t[bind::T_TR3_FC1].p); A.t_wide = D.fc1_wide;
                A.t_fc2 = static_cast<const uint8_t*>(D.t[bind::T_TR3_FC2].p); A.t_wide2 = D.fc2_wide;
                A.t_ptab = D.ptab; A.t_bm1 = D.bm1; A.t_bm2 = D.bm2;
                A.t_v2 = static_cast<const __half*>(v2.p);
                A.x = px; A.ldx = HID;
                A.z1 = psc.z1; A.x2 = psc.x2; A.yw = psc.yw; A.ysh = psc.ysh; A.out = pout;
#ifdef PF_F12
                A.f12 = 1;
#endif
                S = A;
                S.nroutes = M; S.maxseg = (M + bindp::RS - 1) / bindp::RS;
                S.rr_row = psc.sh_rr_row; S.rr_e = psc.sh_rr_e; S.rr_sp = psc.sh_rr_sp; S.rr_w = psc.sh_rr_w;
                S.segs = psc.sh_segs; S.n_seg = psc.sh_n_seg; S.pos = nullptr; S.yw = psc.ysh;
#ifdef PF_F12
                S.f12 = 0;
#endif
                const size_t row = static_cast<size_t>(D.sh_tr[2]);
                A.sh_svh = A.t_sg + HID; A.sh_sc = A.t_v3b + row * (HID / 128);
            };
            auto pf_run = [&](int l, int v, std::vector<uint8_t>* h) {
                bindp::Expert A, S;
                pf_make(l, v, A, S);
                CK(cudaMemsetAsync(pout, 0xff, static_cast<size_t>(M) * HID * 2, st));
                bindp::launch_shared(S, st);
                bindp::launch_experts(A, st);
                if (h)
                {
                    h->assign(static_cast<size_t>(M) * HID * 2, 0);
                    CK(cudaMemcpyAsync(h->data(), pout, h->size(), cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    CK(cudaGetLastError());
                }
            };
            // check: every arm == pack on layers 0 and nlay - 1 (the +pf / +L2 arms are the decode plan kernel's: not prefill arms)
            const int parms[5] = {A_PACK, A_IMG, A_SCT, A_BMT, A_BML};
            for (int l : {0, nlay - 1})
            {
                std::vector<uint8_t> ref, got;
                pf_run(l, A_PACK, &ref);
                for (int i = 1; i < 5; ++i)
                {
                    pf_run(l, parms[i], &got);
                    cmp_bytes((std::string("prefill M=") + std::to_string(M) + " layer " + std::to_string(l) + " " + arm_name[parms[i]] + " == pack").c_str(), ref, got);
                }
            }
            // bench: plain launches, layers in turn (cold), interleaved rounds
            std::vector<double> t[5];
            const int preps = nlay * std::max(1, passes / 2);
            for (int r = 0; r < std::max(5, rounds); ++r)
                for (int i = 0; i < 5; ++i)
                {
                    const int v = parms[i];
                    for (int l = 0; l < nlay; ++l) pf_run(l, v, nullptr);
                    CK(cudaStreamSynchronize(st));
                    CK(cudaEventRecord(e0, st));
                    for (int k = 0; k < preps; ++k) pf_run(k % nlay, v, nullptr);
                    CK(cudaEventRecord(e1, st));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    t[i].push_back(1000.0 * ms / preps);
                }
            CK(cudaGetLastError());
            double med[5];
            for (int i = 0; i < 5; ++i) med[i] = median(t[i]);
            for (int i = 0; i < 5; ++i)
            {
                std::vector<double> s = t[i];
                std::sort(s.begin(), s.end());
                printf("bench prefill M=%d %-11s us/layer=%8.1f (min %8.1f max %8.1f)  vs pack %+7.1f us (%+5.2f%%)\n", M, arm_name[parms[i]], med[i], s.front(), s.back(),
                       med[i] - med[0], 100 * (med[i] - med[0]) / med[0]);
            }
            fflush(stdout);
            CK(cudaFree(pblk));
        }
    }

    // ---------------------------------------------------------------- the switch's primitives, timed
    if (dosw)
    {
        printf("== switch primitives (host wall time per call, median of 20; device sync after each batch)\n");
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        const CUmemAllocationProp p = bm_prop();
        CUmemAccessDesc d = {};
        d.location = p.location;
        d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        for (size_t gsz : {Ls[0].bm1.gs, Ls[0].bm2.gs, GRAN})
        {
            const int n = gsz == GRAN ? 400 : 20;
            CUdeviceptr base;
            CU(cuMemAddressReserve(&base, gsz * n, GRAN, 0, 0));
            std::vector<CUmemGenericAllocationHandle> hs(n);
            std::vector<double> tc, tm, ta, tu, tr, tcp;
            uint8_t* src = static_cast<uint8_t*>(alloc(gsz));
            for (int i = 0; i < n; ++i)
            {
                auto a0 = now(); CU(cuMemCreate(&hs[i], gsz, &p, 0)); auto a1 = now();
                CU(cuMemMap(base + gsz * i, gsz, 0, hs[i], 0)); auto a2 = now();
                CU(cuMemSetAccess(base + gsz * i, gsz, &d, 1)); auto a3 = now();
                tc.push_back(us(a0, a1)); tm.push_back(us(a1, a2)); ta.push_back(us(a2, a3));
            }
            CK(cudaDeviceSynchronize());
            for (int i = 0; i < n; ++i)
            {
                auto a0 = now();
                CK(cudaMemcpyAsync(reinterpret_cast<void*>(base + gsz * i), src, gsz, cudaMemcpyDeviceToDevice, st));
                CK(cudaStreamSynchronize(st));
                tcp.push_back(us(a0, now()));
            }
            for (int i = 0; i < n; ++i)
            {
                auto a0 = now(); CU(cuMemUnmap(base + gsz * i, gsz)); auto a1 = now();
                CU(cuMemRelease(hs[i])); auto a2 = now();
                tu.push_back(us(a0, a1)); tr.push_back(us(a1, a2));
            }
            CU(cuMemAddressFree(base, gsz * n));
            printf("vmm %7.1f MB x %3d: cuMemCreate %7.1f us, cuMemMap %6.1f us, cuMemSetAccess %6.1f us, D2D copy %7.1f us (%.1f GB/s), cuMemUnmap %6.1f us, cuMemRelease %6.1f us  (medians)\n",
                   gsz / 1e6, n, median(tc), median(tm), median(ta), median(tcp), gsz / median(tcp) / 1e3, median(tu), median(tr));
        }
        // the table design's primitives: ptab::set (16 B H2D) and slab copies at piece size
        {
            std::vector<double> ts, tc1, tc2;
            SynLayer& L = Ls[0];
            const size_t f1 = tr3_piece_fc1_bytes(0, &L.tab[0], 0), f2 = tr3_piece_fc2_bytes(&L.tab[0]);
            const char* err;
            uint8_t* s1 = pool.take(f1, err);
            uint8_t* s2 = pool.take(f2, err);
            for (int i = 0; i < 200; ++i)
            {
                auto a0 = now();
                if (const char* e = ptab::set(L.sct, i % 256, i % g6nr::EB, L.sct.host[L.sct.slot(i % 256, i % g6nr::EB)], L.sct.host[L.sct.slot(i % 256, i % g6nr::EB) + 1], st)) { printf("%s\n", e); return 2; }
                ts.push_back(us(a0, now()));
            }
            CK(cudaStreamSynchronize(st));
            for (int i = 0; i < 200; ++i)
            {
                auto a0 = now();
                CK(cudaMemcpyAsync(s1, L.img.host[L.img.slot(i % 256, 0)], f1, cudaMemcpyDeviceToDevice, st));
                tc1.push_back(us(a0, now()));
                auto a1 = now();
                CK(cudaMemcpyAsync(s2, L.img.host[L.img.slot(i % 256, 0) + 1], f2, cudaMemcpyDeviceToDevice, st));
                tc2.push_back(us(a1, now()));
            }
            auto a0 = now();
            CK(cudaStreamSynchronize(st));
            const double drain = us(a0, now());
            printf("table: ptab::set %.1f us issue (16 B H2D); piece copies issue fc1 %zu KB %.1f us, fc2 %zu KB %.1f us; 400 copies drained in %.0f us = %.1f GB/s\n",
                   median(ts), f1 >> 10, median(tc1), f2 >> 10, median(tc2), drain, 200.0 * (f1 + f2) / drain / 1e3);
            std::vector<double> tup;
            for (int i = 0; i < 20; ++i)
            {
                auto b0 = now();
                if (const char* e = ptab::upload(L.sct, st)) { printf("%s\n", e); return 2; }
                tup.push_back(us(b0, now()));
            }
            printf("table: whole-table upload + sync (%zu KB) %.1f us\n", tab_bytes >> 10, median(tup));
        }
    }
    for (void* p : mem) cudaFree(p);
    printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
