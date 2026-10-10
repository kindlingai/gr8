// Row-budget unit test: one GB10, no fabric, no pack files, synthetic weights.
//   row_budget_test [--only gv|k1] [--time [reps]] [--time-v2 [reps]] [--time-multi [reps]]
// Claim under test: for every M <= G6_MX, the dense GEMVs (bind::make_gv / launch) and the K1 expert stream (router plan with
// slots of <= 8 tokens, make_k1 / launch_k1) give, for each row, the same bits as the same rows run in chunks of 8 (the M <= 8
// path). Weights are synthetic (records of the pack formats, hash-filled), so the claim is bit equality against the 8-row path,
// not agreement with a model. Also checked: outputs finite and not all zero, rows >= M not written, the route plan itself.
// K1 runs in the TR3 form (int4 / MXFP8 shared expert records) and the exlr8 v2 form (E4M3 experts, the trellis
// shared expert as tr3.table row 256 in one slot per 8 rows, block-scale hidden-side vectors).
// Default build (G6_MX = 8): the M list stops at 8 and each case compares a run with itself (smoke + plan check).
// --time: times launch_k1 over M = 6..24 and the K1 split tables (for tuning k1_split; prints one line per point).
// --time-v2: launch_k1 on the exlr8 v2 layer per M, bytes read and GB/s (K1 against the memory floor).
// Build: scripts/c1_tests.sh <bin_dir>      Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../kernels/glue.cuh"
#include "../../kernels/bind.cuh"

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

static cudaStream_t const ST = nullptr;      // everything runs on the legacy default stream
static int g_pass = 0, g_fail = 0;

static void report(bool ok, const char* fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("%s %s\n", ok ? "PASS" : "FAIL", buf);
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

static std::string fmt_s(const char* fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

struct Arena
{
    std::vector<void*> v;
    template <class T> T* alloc(size_t n)
    {
        void* p = nullptr;
        CK(cudaMalloc(&p, n * sizeof(T)));
        v.push_back(p);
        return static_cast<T*>(p);
    }
    ~Arena() { for (void* p : v) cudaFree(p); }
};

// ------------------------------------------------------------------------------------------ synthetic data
__device__ __forceinline__ uint32_t tst_rnd(size_t i, uint32_t seed)
{
    uint32_t x = static_cast<uint32_t>(i) ^ ((static_cast<uint32_t>(i >> 32) + seed) * 0x9e3779b9u) ^ (seed * 0x85ebca6bu);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float tst_u(size_t i, uint32_t seed) { return (tst_rnd(i, seed) >> 8) * (1.f / 16777216.f); }

// records of a GEMV / shared-expert codec, 16-bit words: int4 288 B (256 B nibbles + 16 fp16 scales), mxfp8 528 B
// (512 B e4m3 without the NaN codes + 16 e8m0 scales, 2^-11 .. 2^-8), bf16 512 B (small values)
__global__ void k_fill_rec(uint16_t* p, size_t nh, int codec, uint32_t seed)
{
    const int GR = codec == bind::INT4 ? 288 : codec == bind::MXFP8 ? 528 : 512;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nh; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const int b = static_cast<int>((i * 2) % GR);
        const uint32_t h = tst_rnd(i, seed);
        uint16_t v;
        if (codec == bind::INT4)
            v = b < 256 ? static_cast<uint16_t>(h & 0xffffu) : __half_as_ushort(__float2half(0.005f + 0.015f * tst_u(i, seed + 1)));
        else if (codec == bind::MXFP8)
        {
            if (b < 512)
            {
                uint32_t lo = h & 0xffu, hi = (h >> 8) & 0xffu;
                if ((lo & 0x7fu) == 0x7fu) lo &= 0xbfu;
                if ((hi & 0x7fu) == 0x7fu) hi &= 0xbfu;
                v = static_cast<uint16_t>(lo | (hi << 8));
            }
            else v = static_cast<uint16_t>((116u + (h & 3u)) | ((116u + ((h >> 2) & 3u)) << 8));
        }
        else v = __bfloat16_as_ushort(__float2bfloat16_rn((tst_u(i, seed + 1) - 0.5f) * 0.1f));
        p[i] = v;
    }
}
__global__ void k_fill_f16(__half* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = __float2half((2.f * tst_u(i, seed) - 1.f) * amp);
}
__global__ void k_fill_f32(float* p, size_t n, float amp, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = (2.f * tst_u(i, seed) - 1.f) * amp;
}
__global__ void k_fill_words(uint32_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = tst_rnd(i, seed);
}
static int fill_grid(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 4096)); }
static void fill_rec(void* p, size_t bytes, int codec, uint32_t seed)
{
    k_fill_rec<<<fill_grid(bytes / 2), 256>>>(static_cast<uint16_t*>(p), bytes / 2, codec, seed);
    CK(cudaGetLastError());
}
static void fill_f16(__half* p, size_t n, float amp, uint32_t seed)
{
    k_fill_f16<<<fill_grid(n), 256>>>(p, n, amp, seed);
    CK(cudaGetLastError());
}
static void fill_f32(float* p, size_t n, float amp, uint32_t seed)
{
    k_fill_f32<<<fill_grid(n), 256>>>(p, n, amp, seed);
    CK(cudaGetLastError());
}
static void fill_bytes(void* p, size_t bytes, uint32_t seed)
{
    k_fill_words<<<fill_grid(bytes / 4), 256>>>(static_cast<uint32_t*>(p), bytes / 4, seed);
    CK(cudaGetLastError());
}

static bind::Tensor mk(const void* p, int nd, long long a, long long b = 0, long long c = 0)
{
    bind::Tensor t;
    t.p = p; t.nd = nd; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    return t;
}

static std::vector<int> row_list()
{
    const int all[] = {1, 2, 3, 5, 7, 8, 9, 12, 16, 17, 24, 31, 32};
    std::vector<int> v;
    for (int m : all)
        if (m <= G6_MX) v.push_back(m);
    return v;
}

// bits of one output cell finite? odt: gv::O_F16 / O_BF16 / O_F32
static bool cell_finite(const uint8_t* p, size_t i, int odt)
{
    if (odt == gv::O_F32)
    {
        uint32_t b;
        memcpy(&b, p + 4 * i, 4);
        return (b & 0x7f800000u) != 0x7f800000u;
    }
    uint16_t b;
    memcpy(&b, p + 2 * i, 2);
    return odt == gv::O_F16 ? (b & 0x7c00u) != 0x7c00u : (b & 0x7f80u) != 0x7f80u;
}

// full = one run of M rows, chunk = the same rows in chunks of 8; buffers hold G6_MX rows of row_bytes, pre-filled with 0xFF.
static std::string cmp_out(const std::vector<uint8_t>& f, const std::vector<uint8_t>& c, int M, size_t row_bytes, int odt)
{
    const size_t es = odt == gv::O_F32 ? 4 : 2, n = static_cast<size_t>(M) * row_bytes;
    for (size_t i = 0; i < n; ++i)
        if (f[i] != c[i]) return fmt_s("full run != chunks (row %zu, byte %zu)", i / row_bytes, i % row_bytes);
    bool nz = false;
    for (size_t i = 0; i < n / es; ++i)
    {
        if (!cell_finite(f.data(), i, odt)) return fmt_s("non-finite or unwritten cell (row %zu, cell %zu)", i * es / row_bytes, (i * es % row_bytes) / es);
        for (size_t j = 0; j < es; ++j) nz |= f[i * es + j] != 0;
    }
    if (!nz) return "all outputs zero";
    for (size_t i = n; i < f.size(); ++i)
        if (f[i] != 0xff || c[i] != 0xff) return fmt_s("row >= M written (row %zu)", i / row_bytes);
    return "";
}

// ------------------------------------------------------------------------------------------ dense GEMVs
struct GvCase { const char* name; int codec, O, K, tpc, ks, odt; };

static void run_gv(const GvCase& c, const std::vector<int>& Ms, int chunk = 8)
{
    Arena A;
    const int GR = c.codec == bind::INT4 ? 288 : c.codec == bind::MXFP8 ? 528 : 512, gk = c.codec == bind::BF16W ? 16 : 32;
    const int NG = c.K / gk, OT = c.O / 16;
    const size_t es = c.odt == gv::O_F32 ? 4 : 2, bytes = static_cast<size_t>(OT) * NG * GR, row_bytes = static_cast<size_t>(c.O) * es;
    uint8_t* w = A.alloc<uint8_t>(bytes);
    fill_rec(w, bytes, c.codec, 1000u + c.O);
    __half* x = A.alloc<__half>(static_cast<size_t>(G6_MX) * c.K);
    fill_f16(x, static_cast<size_t>(G6_MX) * c.K, 1.f, 7u + c.K);
    const size_t tbytes = static_cast<size_t>(c.K / 16) * 64 * 4 * g6mx::NT;
    u32* tf = A.alloc<u32>(tbytes / 4);
    u32* tc = A.alloc<u32>(tbytes / 4);
    uint8_t* yf = A.alloc<uint8_t>(row_bytes * G6_MX);
    uint8_t* yc = A.alloc<uint8_t>(row_bytes * G6_MX);
    const bind::Tensor t = mk(w, 3, OT, NG, GR);
    {
        bind::Gv g0;
        const char* err = bind::make_gv(t, c.codec, c.tpc, c.ks, 1, tc, yc, c.odt, g0);
        if (err) { report(false, "gv %s: make_gv: %s", c.name, err); return; }
#if DEC_I4PAIR
        if (c.codec == bind::INT4) launch_i4pair(w, OT, NG, g0.ks, ST);     // the pair records the int4 kernel reads
#endif
    }
    std::string fails;
    int ncase = 0;
    for (int M : Ms)
    {
        ++ncase;
        CK(cudaMemset(yf, 0xff, row_bytes * G6_MX));
        CK(cudaMemset(yc, 0xff, row_bytes * G6_MX));
        bind::Xfrag xf;
        xf.on = 1; xf.src = x; xf.ld = c.K; xf.K = c.K; xf.M = M;
        (c.codec == bind::BF16W ? xf.bf16 : xf.f16) = tf;
        bind::launch(xf, ST);
        bind::Gv g;
        const char* err = bind::make_gv(t, c.codec, c.tpc, c.ks, M, tf, yf, c.odt, g);
        if (err) { fails += fmt_s("M=%d make_gv: %s; ", M, err); continue; }
        if (g.K != c.K || g.O != c.O) { fails += fmt_s("M=%d shape K=%d O=%d; ", M, g.K, g.O); continue; }
        bind::launch(g, ST);
        for (int c0 = 0; c0 * chunk < M; ++c0)
        {
            const int Mc = std::min(chunk, M - chunk * c0);
            bind::Xfrag xc = xf;
            xc.src = x + static_cast<size_t>(chunk * c0) * c.K; xc.M = Mc;
            xc.f16 = c.codec == bind::BF16W ? nullptr : tc;
            xc.bf16 = c.codec == bind::BF16W ? tc : nullptr;
            bind::launch(xc, ST);
            bind::Gv gc;
            bind::make_gv(t, c.codec, c.tpc, c.ks, Mc, tc, yc + static_cast<size_t>(chunk * c0) * row_bytes, c.odt, gc);
            bind::launch(gc, ST);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<uint8_t> hf(row_bytes * G6_MX), hc(row_bytes * G6_MX);
        CK(cudaMemcpy(hf.data(), yf, hf.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), yc, hc.size(), cudaMemcpyDeviceToHost));
        const std::string r = cmp_out(hf, hc, M, row_bytes, c.odt);
        if (!r.empty()) fails += fmt_s("M=%d %s; ", M, r.c_str());
    }
    report(fails.empty(), "gv %-8s chunk=%d codec=%d O=%d K=%d tpc=%d ks=%d: %d row counts%s%s", c.name, chunk, c.codec, c.O, c.K, c.tpc, c.ks, ncase,
           fails.empty() ? "" : " | ", fails.c_str());
}

// ------------------------------------------------------------------------------------------ K1
struct MoeW
{
    uint8_t *fc1 = nullptr, *fc2 = nullptr;
    long long* tab = nullptr;
    __half *v1 = nullptr, *v2 = nullptr, *v3 = nullptr;
    size_t fc1_bytes = 0, fc2_bytes = 0;
    int R = 0;
};

// 256 experts, 3-bit (even ids) and 4-bit (odd ids) K3 records, EB blocks (EB - 1 for every third expert), hash-filled
static void build_moe(Arena& A, MoeW& m)
{
    std::vector<long long> tab(256 * 6);
    size_t o1 = 0, o2 = 0;
    int row = 0;
    for (int e = 0; e < 256; ++e)
    {
        const int bits = e % 2 ? 4 : 3, nb = e % 3 == 0 ? g6nr::EB - 1 : g6nr::EB;
        long long* r = &tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = bits; r[5] = 0;
        o1 += static_cast<size_t>(nb) * 2 * KSH * 8 * 32 * bits;
        o2 += static_cast<size_t>(nb) * KSH * 8 * 32 * bits;
        row += nb;
    }
    m.R = row; m.fc1_bytes = o1; m.fc2_bytes = o2;
    m.tab = A.alloc<long long>(tab.size());
    CK(cudaMemcpy(m.tab, tab.data(), tab.size() * sizeof(long long), cudaMemcpyHostToDevice));
    m.fc1 = A.alloc<uint8_t>(o1); fill_bytes(m.fc1, o1, 31u);
    m.fc2 = A.alloc<uint8_t>(o2); fill_bytes(m.fc2, o2, 32u);
    m.v1 = A.alloc<__half>(static_cast<size_t>(row) * 2 * HID); fill_f16(m.v1, static_cast<size_t>(row) * 2 * HID, 0.05f, 33u);
    m.v2 = A.alloc<__half>(static_cast<size_t>(row) * 3 * 128); fill_f16(m.v2, static_cast<size_t>(row) * 3 * 128, 0.5f, 34u);
    m.v3 = A.alloc<__half>(static_cast<size_t>(row) * HID);     fill_f16(m.v3, static_cast<size_t>(row) * HID, 0.5f, 35u);
}

// shared expert / dense MLP records: int4 a = fc1 [nch][192][576], c = fc2 [384][nch/2][288];
// mxfp8 a = gate, b = up [nch][192][528], c = down [384][nch/2][528]
struct ShW { uint8_t *a = nullptr, *b = nullptr, *c = nullptr; int nch = 0, mx = 0; };
static ShW build_shared(Arena& A, int nch, int mx, uint32_t seed)
{
    ShW s;
    s.nch = nch; s.mx = mx;
    const int codec = mx ? bind::MXFP8 : bind::INT4;
    const size_t fc1 = static_cast<size_t>(nch) * 192 * (mx ? 528 : 576), fc2 = static_cast<size_t>(KSH) * (nch / 2) * (mx ? 528 : 288);
    s.a = A.alloc<uint8_t>(fc1); fill_rec(s.a, fc1, codec, seed);
    if (mx) { s.b = A.alloc<uint8_t>(fc1); fill_rec(s.b, fc1, codec, seed + 1); }
    s.c = A.alloc<uint8_t>(fc2); fill_rec(s.c, fc2, codec, seed + 2);
    return s;
}

// moe == nullptr: dense layer (shared slot filled as T_D_FC1 / T_D_FC2)
static bind::LayerTensors make_T(const MoeW* moe, const ShW& s)
{
    bind::LayerTensors T;
    const long long nch = s.nch;
    if (moe)
    {
        T.t[bind::T_TR3_TAB] = mk(moe->tab, 2, 256, 6);
        T.t[bind::T_TR3_FC1] = mk(moe->fc1, 1, static_cast<long long>(moe->fc1_bytes));
        T.t[bind::T_TR3_FC2] = mk(moe->fc2, 1, static_cast<long long>(moe->fc2_bytes));
        T.t[bind::T_TR3_V1] = mk(moe->v1, 3, moe->R, 2, HID);
        T.t[bind::T_TR3_V2] = mk(moe->v2, 3, moe->R, 3, 128);
        T.t[bind::T_TR3_V3] = mk(moe->v3, 2, moe->R, HID);
    }
    if (s.mx)
    {
        T.t[bind::T_S_GATE_MX] = mk(s.a, 3, nch, 192, 528);
        T.t[bind::T_S_UP_MX] = mk(s.b, 3, nch, 192, 528);
        T.t[bind::T_S_DOWN_MX] = mk(s.c, 3, KSH, nch / 2, 528);
    }
    else if (moe)
    {
        T.t[bind::T_S_FC1] = mk(s.a, 3, nch, 192, 576);
        T.t[bind::T_S_FC2] = mk(s.c, 3, KSH, nch / 2, 288);
    }
    else
    {
        T.t[bind::T_D_FC1] = mk(s.a, 3, nch, 192, 576);
        T.t[bind::T_D_FC2] = mk(s.c, 3, KSH, nch / 2, 288);
    }
    return T;
}

// copies of tr3.fc1 / fc2 reordered into the wide order (bind::widen_fc1 / widen_fc2), tensors of `moe`
static bind::LayerTensors make_T_wide(Arena& A, const MoeW& moe, const ShW& s)
{
    uint8_t* w1 = A.alloc<uint8_t>(moe.fc1_bytes);
    uint8_t* w2 = A.alloc<uint8_t>(moe.fc2_bytes);
    CK(cudaMemcpy(w1, moe.fc1, moe.fc1_bytes, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(w2, moe.fc2, moe.fc2_bytes, cudaMemcpyDeviceToDevice));
    bind::LayerTensors T = make_T(&moe, s);
    T.t[bind::T_TR3_FC1].p = w1;
    T.t[bind::T_TR3_FC2].p = w2;
    bind::widen_fc1(T, ST);
    bind::widen_fc2(T, ST);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    return T;
}

// exlr8 v2 layer: exlr8-e4m3 experts, tr3.table row 256 = the trellis shared expert (one slot per 8 rows, k1.cuh
// k1_slot), hidden-side vectors in block-scale form (tr3.sg / v1b / v3b, no tr3.v1 / v3). k2: every fourth expert has 2-bit
// units (the K2 kernels); otherwise widths 3 and 4 only (the K2 = false SH kernels, k1_tr3_fused_sh's other instantiation).
struct MoeV2
{
    uint8_t *fc1 = nullptr, *fc2 = nullptr;
    long long* tab = nullptr;
    __half *v2 = nullptr, *sg = nullptr, *v1b = nullptr, *v3b = nullptr;
    size_t fc1_bytes = 0, fc2_bytes = 0;
    int R = 0;
};
static void build_moe_v2(Arena& A, MoeV2& m, bool k2)
{
    std::vector<long long> tab(257 * 6);
    size_t o1 = 0, o2 = 0;
    int row = 0;
    for (int e = 0; e < 257; ++e)
    {
        const int bits = e == 256 ? 4 : (k2 && e % 4 == 2) ? 2 : e % 2 ? 4 : 3, nb = e < 256 && e % 3 == 0 ? g6nr::EB - 1 : g6nr::EB;
        long long* r = &tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(o1); r[1] = static_cast<long long>(o2); r[2] = row; r[3] = nb; r[4] = bits; r[5] = 0;
        o1 += static_cast<size_t>(nb) * 2 * KSH * 8 * 32 * bits;
        o2 += static_cast<size_t>(nb) * KSH * 8 * 32 * bits;
        row += nb;
    }
    m.R = row; m.fc1_bytes = o1; m.fc2_bytes = o2;
    m.tab = A.alloc<long long>(tab.size());
    CK(cudaMemcpy(m.tab, tab.data(), tab.size() * sizeof(long long), cudaMemcpyHostToDevice));
    m.fc1 = A.alloc<uint8_t>(o1); fill_bytes(m.fc1, o1, 51u);
    m.fc2 = A.alloc<uint8_t>(o2); fill_bytes(m.fc2, o2, 52u);
    m.v2 = A.alloc<__half>(static_cast<size_t>(row) * 3 * 128); fill_f16(m.v2, static_cast<size_t>(row) * 3 * 128, 0.5f, 54u);
    std::vector<__half> sg(2 * HID);
    for (int i = 0; i < 2 * HID; ++i) sg[i] = __float2half(((i * 2654435761u) >> 13) & 1 ? 1.f : -1.f);
    m.sg = A.alloc<__half>(sg.size());
    CK(cudaMemcpy(m.sg, sg.data(), sg.size() * sizeof(__half), cudaMemcpyHostToDevice));
    m.v1b = A.alloc<__half>(static_cast<size_t>(row) * 2 * (HID / 128)); fill_f16(m.v1b, static_cast<size_t>(row) * 2 * (HID / 128), 0.05f, 55u);
    m.v3b = A.alloc<__half>(static_cast<size_t>(row) * (HID / 128));     fill_f16(m.v3b, static_cast<size_t>(row) * (HID / 128), 0.5f, 56u);
}
static bind::LayerTensors make_T_v2(Arena& A, const MoeV2& m, bool wide)
{
    bind::LayerTensors T;
    T.t[bind::T_TR3_TAB] = mk(m.tab, 2, 257, 6);
    T.t[bind::T_TR3_FC1] = mk(m.fc1, 1, static_cast<long long>(m.fc1_bytes));
    T.t[bind::T_TR3_FC2] = mk(m.fc2, 1, static_cast<long long>(m.fc2_bytes));
    T.t[bind::T_TR3_V2] = mk(m.v2, 3, m.R, 3, 128);
    T.t[bind::T_TR3_SG] = mk(m.sg, 2, 2, HID);
    T.t[bind::T_TR3_V1B] = mk(m.v1b, 3, m.R, 2, HID / 128);
    T.t[bind::T_TR3_V3B] = mk(m.v3b, 2, m.R, HID / 128);
    for (int i : {bind::T_TR3_TAB, bind::T_TR3_FC1, bind::T_TR3_FC2, bind::T_TR3_V2}) T.t[i].fmt = bind::FMT_EXLR8_E4M3;
    T.e4m3 = 1;
    if (wide)
    {
        uint8_t* w1 = A.alloc<uint8_t>(m.fc1_bytes);
        uint8_t* w2 = A.alloc<uint8_t>(m.fc2_bytes);
        CK(cudaMemcpy(w1, m.fc1, m.fc1_bytes, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(w2, m.fc2, m.fc2_bytes, cudaMemcpyDeviceToDevice));
        T.t[bind::T_TR3_FC1].p = w1;
        T.t[bind::T_TR3_FC2].p = w2;
    }
    if (const char* e = bind::scan_widths(T)) { printf("scan_widths: %s\n", e); exit(2); }
    if (!T.sh_trellis) { printf("make_T_v2: no trellis shared expert after scan_widths\n"); exit(2); }
    if (wide) { bind::widen_fc1(T, ST); bind::widen_fc2(T, ST); }
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    return T;
}

struct Env
{
    __half* x2 = nullptr;             // [G6_MX][HID] mlp_norm output
    float* logits = nullptr;          // [G6_MX][256]
    float* bias = nullptr;            // [256]
    glue::RoutePlan* dplan = nullptr;
    void* of = nullptr;               // [G6_MX][HID] 2 B, full run
    void* oc = nullptr;               //   chunks of 8
    bind::Scratch sc;
};
// forced: experts 5, 77 and 200 get a +8 selection bias, so every token routes to them (M > 8: experts with > 8 tokens)
static Env make_env(Arena& A, bool forced)
{
    Env E;
    E.x2 = A.alloc<__half>(static_cast<size_t>(G6_MX) * HID);
    fill_f16(E.x2, static_cast<size_t>(G6_MX) * HID, 1.f, 41u);
    E.logits = A.alloc<float>(static_cast<size_t>(G6_MX) * 256);
    fill_f32(E.logits, static_cast<size_t>(G6_MX) * 256, 0.5f, 42u);
    E.bias = A.alloc<float>(256);
    fill_f32(E.bias, 256, 0.05f, 43u);
    if (forced)
        for (int e : {5, 77, 200})
        {
            const float eight = 8.f;
            CK(cudaMemcpy(E.bias + e, &eight, sizeof eight, cudaMemcpyHostToDevice));
        }
    E.dplan = A.alloc<glue::RoutePlan>(1);
    E.of = A.alloc<uint8_t>(static_cast<size_t>(G6_MX) * HID * 2);
    E.oc = A.alloc<uint8_t>(static_cast<size_t>(G6_MX) * HID * 2);
    const size_t sb = bind::scratch_bytes();
    uint8_t* blk = A.alloc<uint8_t>(sb);
    CK(cudaMemset(blk, 0, sb));
    E.sc = bind::scratch_at(blk);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    return E;
}

// the plan of a full run of M tokens: every route lands in a slot of <= 8 tokens, columns and slot arrays agree with it
static std::string check_plan(const glue::RoutePlan& P, int M, const bind::Scratch& sc, bool forced)
{
    const int n = P.n_slots;
    if (n < 1 || n > SMAX || n > sc.smax) return fmt_s("n_slots %d outside 1..min(SMAX %d, scratch %d)", n, SMAX, sc.smax);
    std::vector<int> cnt(256, 0), nslots(256, 0), full(256, 0);
    for (int s = 0; s < n; ++s)
    {
        const int e = P.slot_expert[s], mask = P.slot_mask[s];
        if (e < 0 || e > 255) return fmt_s("slot %d expert %d", s, e);
        if (s && e < P.slot_expert[s - 1]) return fmt_s("slot %d: expert ids not ascending", s);
        if (mask & ~0xff) return fmt_s("slot %d mask 0x%x beyond column 7", s, mask);
        int tokens = 0;
        for (int col = 0; col < 8; ++col)
        {
            const int tk = P.slot_tok[s][col];
            const bool used = (mask >> col) & 1;
            if (used != (tk >= 0)) return fmt_s("slot %d col %d: mask bit %d, slot_tok %d", s, col, used, tk);
            if (tk >= M) return fmt_s("slot %d col %d: token %d >= M", s, col, tk);
            tokens += used;
            if (!used && P.slot_w[s][col] != 0.f) return fmt_s("slot %d col %d: weight on an unused column", s, col);
        }
        if (M > 8 && mask != (1 << tokens) - 1) return fmt_s("slot %d: mask 0x%x is not the low %d columns", s, mask, tokens);
        ++nslots[e];
        cnt[e] += tokens;
        full[e] += tokens == 8;
    }
    for (int m = 0; m < M; ++m)
    {
        bool seen[256] = {false};
        for (int k = 0; k < 8; ++k)
        {
            const int s = P.tok_slot[m][k], col = P.tok_col[m][k], e = P.tok_expert[m][k];
            if (s < 0 || s >= n || col < 0 || col > 7) return fmt_s("token %d route %d: slot %d col %d", m, k, s, col);
            if (e < 0 || e > 255 || seen[e]) return fmt_s("token %d route %d: expert %d missing or repeated", m, k, e);
            seen[e] = true;
            if (P.slot_expert[s] != e) return fmt_s("token %d route %d: slot expert %d != %d", m, k, P.slot_expert[s], e);
            if (P.slot_tok[s][col] != m) return fmt_s("token %d route %d: slot_tok %d", m, k, P.slot_tok[s][col]);
            if (P.slot_w[s][col] != P.tok_w[m][k]) return fmt_s("token %d route %d: slot_w != tok_w", m, k);
            if (!((P.slot_mask[s] >> col) & 1)) return fmt_s("token %d route %d: column %d not in the slot mask", m, k, col);
            if (M <= 8 && col != m) return fmt_s("token %d route %d: column %d (M <= 8 keeps column = token)", m, k, col);
            if (!(P.tok_w[m][k] > 0.f)) return fmt_s("token %d route %d: weight %g", m, k, P.tok_w[m][k]);
        }
    }
    for (int m = M; m < G6_MX; ++m)
        for (int k = 0; k < 8; ++k)
            if (P.tok_slot[m][k] != -1) return fmt_s("token %d >= M has slot %d", m, P.tok_slot[m][k]);
    int total = 0, maxcnt = 0;
    for (int e = 0; e < 256; ++e)
    {
        int c2 = 0;
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < 8; ++k) c2 += P.tok_expert[m][k] == e;
        if (c2 != cnt[e]) return fmt_s("expert %d: %d routes, %d in slots", e, c2, cnt[e]);
        const int want = M <= 8 ? (c2 > 0) : (c2 + 7) / 8;
        if (nslots[e] != want) return fmt_s("expert %d: %d tokens in %d slots (want %d)", e, c2, nslots[e], want);
        if (M > 8 && nslots[e] && full[e] != nslots[e] - (cnt[e] % 8 != 0)) return fmt_s("expert %d: slots other than the last are not full", e);
        total += c2;
        maxcnt = std::max(maxcnt, c2);
    }
    if (total != M * 8) return fmt_s("%d routes, want %d", total, M * 8);
    if (forced && M > 8 && maxcnt <= 8) return fmt_s("forced routing gave at most %d tokens per expert (no expert with > 8)", maxcnt);
    return "";
}

static inline void route(const Env& E, int tok0, int M)
{
    glue::k_router_plan<<<1, 256>>>(E.logits + static_cast<size_t>(tok0) * 256, E.bias, M, E.dplan);
    CK(cudaGetLastError());
}

// one K1 configuration over the M list: full M-row run vs chunks of 8 (same scratch, same split_m = 8 so the k split is the
// same), plus the plan check on the full run.
// sess (S2 session check): chunk = rows per session, the full run binds split_m = chunk (the engine's row class, s2_layer)
// and each chunk binds split_m = 0 (today's solo step of chunk rows, layer_for): a row's bits must not depend on the batch.
static void run_k1(const std::string& name, const bind::LayerTensors& T, bool moe, int fuse, int out_bf16, const Env& E, bool forced,
                   const std::vector<int>& Ms, int chunk = 8, bool sess = false)
{
    const int split_full = sess ? chunk : 8, split_chunk = sess ? 0 : 8;
    const int odt = out_bf16 ? gv::O_BF16 : gv::O_F16;
    const size_t row_bytes = static_cast<size_t>(HID) * 2;
    std::string fails;
    int ncase = 0, maxsl = 0;
    for (int M : Ms)
    {
        ++ncase;
        CK(cudaMemset(E.of, 0xff, row_bytes * G6_MX));
        CK(cudaMemset(E.oc, 0xff, row_bytes * G6_MX));
        if (moe) route(E, 0, M);
        bind::Xfrag xf;
        xf.on = 1; xf.src = E.x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = E.sc.xf_x2;
        bind::launch(xf, ST);
        bind::K1 k;
        bind::Tensor sh[3];
        const char* err = bind::make_k1(T, M, E.sc, E.x2, E.sc.xf_x2, moe ? bind::plan_ptrs(E.dplan) : bind::PlanPtrs(), E.of, out_bf16, k, sh, fuse, split_full);
        if (err) { fails += fmt_s("M=%d make_k1: %s; ", M, err); continue; }
        bind::launch_k1(k, ST);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        if (moe)
        {
            glue::RoutePlan* hp = new glue::RoutePlan;
            CK(cudaMemcpy(hp, E.dplan, sizeof(glue::RoutePlan), cudaMemcpyDeviceToHost));
            const std::string r = check_plan(*hp, M, E.sc, forced);
            maxsl = std::max(maxsl, hp->n_slots);
            delete hp;
            if (!r.empty()) fails += fmt_s("M=%d plan: %s; ", M, r.c_str());
        }
        for (int c0 = 0; c0 * chunk < M; ++c0)
        {
            const int Mc = std::min(chunk, M - chunk * c0);
            const __half* x2c = E.x2 + static_cast<size_t>(chunk * c0) * HID;
            if (moe) route(E, chunk * c0, Mc);
            bind::Xfrag xc = xf;
            xc.src = x2c; xc.M = Mc;
            bind::launch(xc, ST);
            bind::K1 kc;
            bind::Tensor shc[3];
            err = bind::make_k1(T, Mc, E.sc, x2c, E.sc.xf_x2, moe ? bind::plan_ptrs(E.dplan) : bind::PlanPtrs(),
                                static_cast<uint8_t*>(E.oc) + static_cast<size_t>(chunk * c0) * row_bytes, out_bf16, kc, shc, fuse, split_chunk);
            if (err) { fails += fmt_s("M=%d chunk %d make_k1: %s; ", M, c0, err); break; }
            bind::launch_k1(kc, ST);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<uint8_t> hf(row_bytes * G6_MX), hc(row_bytes * G6_MX);
        CK(cudaMemcpy(hf.data(), E.of, hf.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), E.oc, hc.size(), cudaMemcpyDeviceToHost));
        const std::string r = cmp_out(hf, hc, M, row_bytes, odt);
        if (!r.empty()) fails += fmt_s("M=%d %s; ", M, r.c_str());
    }
    report(fails.empty(), "k1 %s: %d row counts, max slots %d%s%s", name.c_str(), ncase, maxsl, fails.empty() ? "" : " | ", fails.c_str());
}

static void test_k1(const std::vector<int>& Ms)
{
    const int nch = 2 * g6nr::HMAX, nch_dense = g6nr::DENSE_CH / 16;
    {
        Arena A;
        MoeW moe;
        build_moe(A, moe);
        const ShW s4 = build_shared(A, nch, 0, 100u), smx = build_shared(A, nch, 1, 200u);
        Env E = make_env(A, true);
        const bind::LayerTensors Tn4 = make_T(&moe, s4), Tnmx = make_T(&moe, smx);
        const bind::LayerTensors Tw4 = make_T_wide(A, moe, s4), Twmx = make_T_wide(A, moe, smx);
        for (int wide = 0; wide < 2; ++wide)
            for (int fuse = 0; fuse < 3; ++fuse)
                for (int bf = 0; bf < 2; ++bf)
                    run_k1(fmt_s("moe int4-shared wide=%d fuse=%d out_bf16=%d", wide, fuse, bf), wide ? Tw4 : Tn4, true, fuse, bf, E, true, Ms);
        for (int wide = 0; wide < 2; ++wide)
            for (int fuse = 0; fuse < 3; ++fuse)
                run_k1(fmt_s("moe mxfp8-shared wide=%d fuse=%d out_bf16=1", wide, fuse), wide ? Twmx : Tnmx, true, fuse, 1, E, true, Ms);
        // routing as it comes (no forced experts): M = 1..32 tokens spread over the experts
        for (int fuse = 0; fuse < 2; ++fuse)
        {
            Env E2 = E;
            const float zero = 0.f;
            for (int e : {5, 77, 200}) CK(cudaMemcpy(E2.bias + e, &zero, sizeof zero, cudaMemcpyHostToDevice));
            run_k1(fmt_s("moe int4-shared natural routing wide=1 fuse=%d out_bf16=1", fuse), Tw4, true, fuse, 1, E2, false, Ms);
        }
        // S2 sessions: M rows as sessions of 3 (or 1) rows, batched (split_m = class) vs each session alone (solo bind)
        for (int sr : {3, 1})
            for (int fuse = 0; fuse < 2; ++fuse)
                for (int wide = 0; wide < 2; ++wide)
                {
                    Env E2 = E;
                    const float zero = 0.f;
                    for (int e : {5, 77, 200}) CK(cudaMemcpy(E2.bias + e, &zero, sizeof zero, cudaMemcpyHostToDevice));
                    std::vector<int> Ms2;
                    for (int M : Ms) if (M % sr == 0 && M > sr) Ms2.push_back(M);
                    run_k1(fmt_s("S2 sessions of %d rows: batched vs solo, natural routing wide=%d fuse=%d", sr, wide, fuse), wide ? Tw4 : Tn4,
                           true, fuse, 1, E2, false, Ms2, sr, true);
                }
    }
    for (int k2 = 0; k2 < 2; ++k2)
    {
        Arena A;
        MoeV2 m;
        build_moe_v2(A, m, k2);
        Env E = make_env(A, true);
        const bind::LayerTensors Tn = make_T_v2(A, m, false), Tw = make_T_v2(A, m, true);
        for (int wide = 0; wide < 2; ++wide)
            for (int fuse = 0; fuse < 3; ++fuse)
                run_k1(fmt_s("moe v2 trellis-shared k2=%d wide=%d fuse=%d out_bf16=1", k2, wide, fuse), wide ? Tw : Tn, true, fuse, 1, E, true, Ms);
        for (int sr : {3, 1})
            for (int fuse = 0; fuse < 2; ++fuse)
            {
                Env E2 = E;
                const float zero = 0.f;
                for (int e : {5, 77, 200}) CK(cudaMemcpy(E2.bias + e, &zero, sizeof zero, cudaMemcpyHostToDevice));
                std::vector<int> Ms2;
                for (int M : Ms) if (M % sr == 0 && M > sr) Ms2.push_back(M);
                run_k1(fmt_s("S2 sessions of %d rows, v2 trellis-shared k2=%d: batched vs solo, natural routing wide=1 fuse=%d", sr, k2, fuse), Tw,
                       true, fuse, 1, E2, false, Ms2, sr, true);
            }
    }
    {
        Arena A;
        const ShW d = build_shared(A, nch_dense, 0, 300u);
        Env E = make_env(A, false);
        const bind::LayerTensors T = make_T(nullptr, d);
        for (int fuse = 0; fuse < 3; ++fuse)
            for (int bf = 0; bf < 2; ++bf)
                run_k1(fmt_s("dense fuse=%d out_bf16=%d", fuse, bf), T, false, fuse, bf, E, false, Ms);
    }
}

// the layer-input GEMVs (q_a, kv_a, idx_k, idx_w) as one gv_multi launch against one launch each, M = 1..8: identical
// bytes. reps > 0: also the time of both forms (cudaEvent, the same weights every call, so partly from L2).
static void run_gv_multi(int reps)
{
    Arena A;
    const GvCase cs[] = {
        {"q_a",   bind::INT4,  2048, 6144, 4, 8,  gv::O_F16},
        {"kv_a",  bind::MXFP8, 576,  6144, 2, 16, gv::O_F16},
        {"idx_k", bind::BF16W, 128,  6144, 1, 32, gv::O_F16},
        {"idx_w", bind::BF16W, 32,   6144, 1, 32, gv::O_F16},
    };
    const int K = 6144;
    __half* x = A.alloc<__half>(static_cast<size_t>(8) * K);
    fill_f16(x, static_cast<size_t>(8) * K, 1.f, 77u);
    const size_t tbytes = static_cast<size_t>(K / 16) * 64 * 4;
    u32* tf = A.alloc<u32>(tbytes / 4);
    u32* tb = A.alloc<u32>(tbytes / 4);
    bind::Tensor t[4];
    uint8_t *y1[4], *y2[4];
    size_t yb[4];
    for (int i = 0; i < 4; ++i)
    {
        const GvCase& c = cs[i];
        const int GR = c.codec == bind::INT4 ? 288 : c.codec == bind::MXFP8 ? 528 : 512, gk = c.codec == bind::BF16W ? 16 : 32;
        const size_t bytes = static_cast<size_t>(c.O / 16) * (K / gk) * GR;
        uint8_t* w = A.alloc<uint8_t>(bytes);
        fill_rec(w, bytes, c.codec, 2000u + i);
        t[i] = mk(w, 3, c.O / 16, K / gk, GR);
#if DEC_I4PAIR
        if (c.codec == bind::INT4) launch_i4pair(w, c.O / 16, K / gk, c.ks, ST);
#endif
        yb[i] = static_cast<size_t>(8) * c.O * 2;
        y1[i] = A.alloc<uint8_t>(yb[i]);
        y2[i] = A.alloc<uint8_t>(yb[i]);
    }
    std::string fails;
    for (int M = 1; M <= 8; ++M)
    {
        bind::Xfrag xf;
        xf.on = 1; xf.src = x; xf.ld = K; xf.K = K; xf.M = M; xf.f16 = tf; xf.bf16 = tb;
        bind::launch(xf, ST);
        bind::Gv g1[4], g2[4];
        for (int i = 0; i < 4; ++i)
        {
            const GvCase& c = cs[i];
            const u32* tab = c.codec == bind::BF16W ? tb : tf;
            bind::make_gv(t[i], c.codec, c.tpc, c.ks, M, tab, y1[i], c.odt, g1[i]);
            bind::make_gv(t[i], c.codec, c.tpc, c.ks, M, tab, y2[i], c.odt, g2[i]);
            CK(cudaMemset(y1[i], 0xff, yb[i]));
            CK(cudaMemset(y2[i], 0xff, yb[i]));
        }
        GvMulti mu;
        bind::make_gv_multi({&g2[0], &g2[1], &g2[2], &g2[3]}, M, mu);
        if (mu.n != 4) { fails += fmt_s("M=%d make_gv_multi bound %d parts; ", M, mu.n); continue; }
        for (int i = 0; i < 4; ++i) bind::launch(g1[i], ST);
        bind::launch(mu, ST);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        for (int i = 0; i < 4; ++i)
        {
            std::vector<uint8_t> h1(yb[i]), h2(yb[i]);
            CK(cudaMemcpy(h1.data(), y1[i], yb[i], cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(), y2[i], yb[i], cudaMemcpyDeviceToHost));
            if (h1 != h2) fails += fmt_s("M=%d %s differs; ", M, cs[i].name);
        }
        if (reps > 0 && (M == 1 || M == 3 || M == 6 || M == 8))
        {
            cudaEvent_t e0, e1, e2;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
            CK(cudaEventRecord(e0, ST));
            for (int r = 0; r < reps; ++r) for (int i = 0; i < 4; ++i) bind::launch(g1[i], ST);
            CK(cudaEventRecord(e1, ST));
            for (int r = 0; r < reps; ++r) bind::launch(mu, ST);
            CK(cudaEventRecord(e2, ST));
            CK(cudaEventSynchronize(e2));
            float a = 0, b = 0;
            CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&b, e1, e2));
            printf("  gv_multi M=%d: four launches %.1f us, one launch %.1f us (%d CTAs)\n", M, 1e3 * a / reps, 1e3 * b / reps, mu.ctas);
        }
    }
    report(fails.empty(), "gv_multi q_a+kv_a+idx_k+idx_w vs one launch each, M = 1..8%s%s", fails.empty() ? "" : " | ", fails.c_str());
}

static void test_gv(const std::vector<int>& Ms)
{
    const int QBW = g6nr::QBW;
    const GvCase cases[] = {
        {"q_a",     bind::INT4,  2048, 6144, 4, 8,  gv::O_F16},
        {"kv_a",    bind::MXFP8, 576,  6144, 2, 16, gv::O_F16},
        {"q_b",     bind::MXFP8, QBW,  2048, 1, 16, gv::O_F16},
        {"o_proj",  bind::INT4,  6144, QBW,  2, 8,  gv::O_BF16},
        {"router",  bind::BF16W, 256,  6144, 2, 16, gv::O_F32},
        {"lm_head", bind::MXFP8, 1024, 6144, 1, 32, gv::O_F32},
        {"eh",      bind::MXFP8, 6144, 12288, 2, 16, gv::O_F16},
    };
    for (const GvCase& c : cases) run_gv(c, Ms);
    run_gv_multi(0);
    // S2 sessions: the rows of several sessions in one GEMV vs each session of 3 / 1 rows alone
    for (int sr : {3, 1})
    {
        std::vector<int> Ms2;
        for (int M : Ms) if (M % sr == 0 && M > sr) Ms2.push_back(M);
        for (const GvCase& c : cases) run_gv(c, Ms2, sr);
    }
}

// ------------------------------------------------------------------------------------------ timing
static int time_mode(int reps)
{
    Arena A;
    MoeW moe;
    build_moe(A, moe);
    const ShW s4 = build_shared(A, 2 * g6nr::HMAX, 0, 100u);
    Env E = make_env(A, false);
    const bind::LayerTensors T = make_T_wide(A, moe, s4);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("K1 time per launch_k1 (fc1 + fc2 + finalize), unforced routing, us; split_m = the row class passed to k1_split\n");
    for (int M = 6; M <= std::min(24, G6_MX); ++M)
    {
        route(E, 0, M);
        bind::Xfrag xf;
        xf.on = 1; xf.src = E.x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = E.sc.xf_x2;
        bind::launch(xf, ST);
        double best = 1e30;
        int bs = 0, bf = 0;
        for (int split_m : {1, 3, 8})
            for (int fuse = 0; fuse < 2; ++fuse)
            {
                bind::K1 k;
                bind::Tensor sh[3];
                const char* err = bind::make_k1(T, M, E.sc, E.x2, E.sc.xf_x2, bind::plan_ptrs(E.dplan), E.of, 1, k, sh, fuse, split_m);
                if (err) { printf("M=%d split_m=%d fuse=%d: %s\n", M, split_m, fuse, err); continue; }
                for (int i = 0; i < 3; ++i) bind::launch_k1(k, ST);
                CK(cudaEventRecord(e0, ST));
                for (int i = 0; i < reps; ++i) bind::launch_k1(k, ST);
                CK(cudaEventRecord(e1, ST));
                CK(cudaEventSynchronize(e1));
                CK(cudaGetLastError());
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double us = 1000.0 * ms / reps;
                printf("M=%2d split_m=%d fuse=%d us=%.1f\n", M, split_m, fuse, us);
                if (us < best) { best = us; bs = split_m; bf = fuse; }
            }
        printf("M=%2d best: split_m=%d fuse=%d us=%.1f\n", M, bs, bf, best);
        fflush(stdout);
    }
    return 0;
}

// --time-v2 [reps]: launch_k1 on the exlr8 v2 layer (E4M3 experts at widths 3 / 4, the trellis shared expert, the wide fc
// layout the engine binds), per M: mean distinct routed experts, the bytes of their units on this rank plus the shared
// expert's, time per call and GB/s (bytes / time). Each call routes a new set of tokens (as each layer reads other experts
// in the engine; the same experts back to back would be served partly from L2); the routing's own time is subtracted.
static int time_v2(int reps)
{
    Arena A;
    MoeV2 m;
    build_moe_v2(A, m, false);
    // layers with their own (wide) weights, used in turn, so no call finds the previous one's experts in L2
    constexpr int NLAY = 6;
    std::vector<bind::LayerTensors> Ts;
    for (int l = 0; l < NLAY; ++l) Ts.push_back(make_T_v2(A, m, true));
    const int nlay = NLAY;
    Env E = make_env(A, false);
    std::vector<long long> tab(257 * 6);
    CK(cudaMemcpy(tab.data(), m.tab, tab.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    auto ebytes = [&](int e) { const long long* r = &tab[static_cast<size_t>(e) * 6]; return 3.0 * r[3] * KSH * 8 * 32 * r[4]; };
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("K1 v2 time per launch_k1 (fc1 + fc2 + finalize), natural routing\n");
    constexpr int NSET = 64;                     // token sets, each its own random logits
    float* lg = A.alloc<float>(static_cast<size_t>(NSET) * G6_MX * 256);
    fill_f32(lg, static_cast<size_t>(NSET) * G6_MX * 256, 0.5f, 77u);
    for (int M : {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32})
    {
        if (M > G6_MX) break;
        double bytes = 0, nex = 0;
        for (int j = 0; j < NSET; ++j)
        {
            glue::k_router_plan<<<1, 256>>>(lg + static_cast<size_t>(j) * G6_MX * 256, E.bias, M, E.dplan);
            glue::RoutePlan P;
            CK(cudaMemcpy(&P, E.dplan, sizeof P, cudaMemcpyDeviceToHost));
            std::vector<int> ex(P.slot_expert, P.slot_expert + P.n_slots);
            std::sort(ex.begin(), ex.end());
            ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
            bytes += ebytes(256);
            for (int e : ex) bytes += ebytes(e);
            nex += ex.size();
        }
        bytes /= NSET; nex /= NSET;
        auto plan = [&](int i) { glue::k_router_plan<<<1, 256>>>(lg + static_cast<size_t>(i % NSET) * G6_MX * 256, E.bias, M, E.dplan); };
        CK(cudaEventRecord(e0, ST));
        for (int i = 0; i < reps; ++i) plan(i);
        CK(cudaEventRecord(e1, ST));
        CK(cudaEventSynchronize(e1));
        float pms = 0.f;
        CK(cudaEventElapsedTime(&pms, e0, e1));
        bind::Xfrag xf;
        xf.on = 1; xf.src = E.x2; xf.ld = HID; xf.K = HID; xf.M = M; xf.f16 = E.sc.xf_x2;
        bind::launch(xf, ST);
        for (int fuse = 0; fuse < 2; ++fuse)
        {
            std::vector<bind::K1> ks(nlay);
            bind::Tensor sh[3];
            const char* err = nullptr;
            for (int l = 0; l < nlay && !err; ++l) err = bind::make_k1(Ts[l], M, E.sc, E.x2, E.sc.xf_x2, bind::plan_ptrs(E.dplan), E.of, 1, ks[l], sh, fuse);
            if (err) { printf("M=%d fuse=%d: %s\n", M, fuse, err); continue; }
            const bind::K1& k = ks[0];
            for (int i = 0; i < 3; ++i) { plan(i); bind::launch_k1(ks[i % nlay], ST); }
            CK(cudaEventRecord(e0, ST));
            for (int i = 0; i < reps; ++i) { plan(i); bind::launch_k1(ks[i % nlay], ST); }
            CK(cudaEventRecord(e1, ST));
            CK(cudaEventSynchronize(e1));
            CK(cudaGetLastError());
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = 1000.0 * (ms - pms) / reps;
            printf("M=%2d experts=%5.1f MB=%6.2f fuse=%d us=%7.1f GB/s=%6.1f grid=%d\n", M, nex, bytes / 1e6, fuse, us, bytes / us / 1e3,
                   fuse ? k.gf : 0);
        }
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv)
{
    CK(cudaSetDevice(0));
    CK(bind::init());
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("row_budget_test: %s sm_%d%d, G6_MX=%d G6_NR=%d SMAX=%d scratch=%.1f MB\n", pr.name, pr.major, pr.minor, G6_MX, g6nr::NR, SMAX,
           static_cast<double>(bind::scratch_bytes()) / 1e6);
    bool do_gv = true, do_k1 = true;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--time")) return time_mode(i + 1 < argc ? std::max(1, atoi(argv[i + 1])) : 50);
        if (!strcmp(argv[i], "--time-v2")) return time_v2(i + 1 < argc ? std::max(1, atoi(argv[i + 1])) : 50);
        if (!strcmp(argv[i], "--time-multi"))
        {
            run_gv_multi(i + 1 < argc ? std::max(1, atoi(argv[i + 1])) : 200);
            return g_fail ? 1 : 0;
        }
        if (!strcmp(argv[i], "--only") && i + 1 < argc)
        {
            do_gv = !strcmp(argv[i + 1], "gv");
            do_k1 = !strcmp(argv[i + 1], "k1");
            ++i;
        }
    }
    const std::vector<int> Ms = row_list();
    if (do_gv) test_gv(Ms);
    if (do_k1) test_k1(Ms);
    printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
