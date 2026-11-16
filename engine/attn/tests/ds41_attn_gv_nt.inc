// The decode GEMV at M = 9..32 (ds41_attn_gv.cuh: k_xfrag_q_nt, k_gv_mx_nt<2..4>, k_gv_mx_q<1..4>) and its row cap (ds41a::g_gv_rows_max).
// Synthetic data, no fixture, no model. Include at GLOBAL scope after ds41_attn.cuh and ds41_attn_gv.cuh; ds41a::gvnt::run_all() runs every gate,
// prints one marker line per gate and exits 1 on the first failure. Run by ds41_attn_test --gv-nt and at the end of ds41_v2_attn_test.
// Every ds41a / ds41dsp global it changes (g_use_gv, g_use_tc, g_gv_rows_max, g_fn_xfrag, g_fn_gemv, g_src_fast, g_gv_quad, g_dsp_gv, g_gv_tpc,
// g_gv_ks) is saved first and restored before it returns, and every device buffer it allocates is freed.
//
//   ATTN_GV_NT_BITWISE_PASS        every real projection shape (wq_a, wq_b, wkv, grouped wo_a at G = 2 and 4, wo_b) at M = 1..32, stock records
//                                  (tpc 1 / default ks and tpc 2 / ks 8) and quad records (gv_quad_bind + gv_gemv_launch_q at tpc 1): one pass
//                                  vs the 8-row passes (k_xfrag_q + k_gv_mx), 0 bf16 bit mismatches, canaries around the output and the table,
//                                  output rows >= M untouched; a perturbed row of one group at M = 24 changes only that row of that group;
//                                  idx wq_b (stock records) on the xf_qr table the quad wq_b's proj_mm built; quad shapes that do not fit stay stock.
//   ATTN_GV_NT_PREFILL_UNCHANGED   attn_proj + attn_out hook calls: cap 8 (eager / prefill) takes the GEMV at n = 1, 6, 8 and none at
//                                  n = 9, 16, 29, 32, 33; cap 32 (decode / verify) takes it at 9..32 and none at 33; DSpark stays at 8 rows;
//                                  v2 attn_proj_own (1-tile InputBuf::xf) stays at 8 rows at cap 32 (n = 8: 1 + 2 hook calls, n = 16: 0 + 0).
#pragma once
#include "../ds41_attn_dsp.cuh"
#include "../ds41_attn_v2.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_set>

namespace ds41a
{
namespace gvnt
{
#define GVNT_CK(x)                                                                                                  \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess)                                                                                      \
        {                                                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                \
            exit(2);                                                                                                \
        }                                                                                                           \
    } while (0)

inline void fail(const char* what)
{
    fprintf(stderr, "FAIL gv_nt: %s\n", what);
    printf("FAIL gv_nt: %s\n", what);
    exit(1);
}
inline void require(bool ok, const char* what) { if (!ok) fail(what); }

// every allocation of one gate, freed by ~Pool
struct Pool
{
    std::vector<void*> p;
    template <class T> T* alloc(size_t n)
    {
        void* q = nullptr;
        GVNT_CK(cudaMalloc(&q, (n ? n : 1) * sizeof(T)));
        p.push_back(q);
        return (T*)q;
    }
    template <class T> T* up(const std::vector<T>& h)
    {
        T* d = alloc<T>(h.size());
        GVNT_CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
        return d;
    }
    ~Pool() { for (void* q : p) cudaFree(q); }
};

struct Rng
{
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
};
inline uint8_t fp8(Rng& r) { uint8_t b = (uint8_t)r.next(); return (b & 0x7f) == 0x7f ? (uint8_t)(b ^ 1) : b; }   // no E4M3 NaN
inline uint8_t e8m0(Rng& r) { return (uint8_t)(118 + r.next() % 12); }                                              // 2^-9 .. 2^2

// MXFP8 GEMV records [rows/16][K/32][528] of codes [rows][K] + E8M0 scales [rows][K/32] (the k_mx_record layout of ds41_attn_test.cu, on the host)
inline std::vector<uint8_t> records(const std::vector<uint8_t>& codes, const std::vector<uint8_t>& sc, long rows, int K)
{
    const int ng = K / 32;
    std::vector<uint8_t> out((size_t)(rows / 16) * ng * 528);
    for (long t = 0; t < rows / 16; ++t)
        for (int g = 0; g < ng; ++g)
        {
            uint8_t* rec = out.data() + ((size_t)t * ng + g) * 528;
            for (int i = 0; i < 256; ++i)
            {
                const int l = i >> 3, e = i & 7, r = l >> 2, c = l & 3;
                const int row = r + ((e & 2) ? 8 : 0), col = 2 * c + (e & 1) + ((e & 4) ? 8 : 0);
                for (int h = 0; h < 2; ++h) rec[h * 256 + i] = codes[(size_t)(16 * t + row) * K + 32 * g + 16 * h + col];
            }
            for (int i = 0; i < 8; ++i)
            {
                rec[512 + 2 * i] = sc[(size_t)(16 * t + i) * ng + g];
                rec[512 + 2 * i + 1] = sc[(size_t)(16 * t + i + 8) * ng + g];
            }
        }
    return out;
}
// a synthetic weight of `groups` blocks of N x K (w, s, gv all set; the gv records are the w / s records)
inline Fp8Mat mat(Pool& P, int N, int K, int groups, uint32_t seed)
{
    Rng r(seed);
    const long rows = (long)groups * N;
    std::vector<uint8_t> w((size_t)rows * K), s((size_t)rows * (K / 32));
    for (auto& b : w) b = fp8(r);
    for (auto& b : s) b = e8m0(r);
    Fp8Mat m{};
    m.w = P.up(w);
    m.s = P.up(s);
    m.N = N;
    m.K = K;
    m.gv = P.up(records(w, s, rows, K));
    return m;
}
// a copy of W's records (for the quad repack, which works in place)
inline Fp8Mat gv_copy(Pool& P, const Fp8Mat& W, int groups)
{
    const size_t b = (size_t)groups * (W.N / 16) * (W.K / 32) * 528;
    uint8_t* d = P.alloc<uint8_t>(b);
    GVNT_CK(cudaMemcpy(d, W.gv, b, cudaMemcpyDeviceToDevice));
    Fp8Mat m = W;
    m.gv = d;
    return m;
}

// the globals this file changes, restored by ~Saved
struct Saved
{
    bool use_gv = g_use_gv, use_tc = g_use_tc;
    int rows_max = g_gv_rows_max, src_fast = g_src_fast, tpc = g_gv_tpc, ks = g_gv_ks, dsp_gv = ds41dsp::g_dsp_gv;
    XfragFn fx = g_fn_xfrag;
    GemvFn fg = g_fn_gemv;
    std::unordered_set<const uint8_t*> quad = g_gv_quad;
    ~Saved()
    {
        cudaDeviceSynchronize();
        g_use_gv = use_gv; g_use_tc = use_tc; g_gv_rows_max = rows_max; g_src_fast = src_fast; g_gv_tpc = tpc; g_gv_ks = ks;
        ds41dsp::g_dsp_gv = dsp_gv; g_fn_xfrag = fx; g_fn_gemv = fg; g_gv_quad = quad;
    }
};

constexpr uint32_t XF_CANARY = 0xA5C3E1F7u;
constexpr uint16_t Y_CANARY = 0x7FA5;   // a bf16 NaN pattern the GEMV never writes
constexpr size_t GUARD = 256;           // canary elements before and after every checked buffer

template <class T> struct Guarded
{
    T* base = nullptr;
    size_t n = 0;
    T* p() const { return base + GUARD; }
};
template <class T> Guarded<T> guarded(Pool& P, size_t n, T fill)
{
    Guarded<T> g;
    g.n = n;
    g.base = P.alloc<T>(n + 2 * GUARD);
    std::vector<T> h(n + 2 * GUARD, fill);
    GVNT_CK(cudaMemcpy(g.base, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return g;
}
template <class T> void refill(const Guarded<T>& g, T fill)
{
    std::vector<T> h(g.n + 2 * GUARD, fill);
    GVNT_CK(cudaMemcpy(g.base, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> host(const Guarded<T>& g)
{
    std::vector<T> h(g.n + 2 * GUARD);
    GVNT_CK(cudaMemcpy(h.data(), g.base, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
// guards (and elements [from, n) of the payload) still hold the fill
template <class T> bool canaries_ok(const std::vector<T>& h, size_t n, size_t from, T fill)
{
    for (size_t i = 0; i < GUARD; ++i)
        if (h[i] != fill || h[GUARD + n + i] != fill) return false;
    for (size_t i = from; i < n; ++i)
        if (h[GUARD + i] != fill) return false;
    return true;
}

// one projection case: activations of 32 rows (codes [32][groups*K], scales [32][groups*K/32]); M takes the first M rows
struct Acts
{
    uint8_t *codes, *sc;
    int ldc, lds;
};
inline Acts acts(Pool& P, int K, int groups, uint32_t seed)
{
    Rng r(seed);
    const int ldc = groups * K, lds = groups * K / 32;
    std::vector<uint8_t> c((size_t)32 * ldc), s((size_t)32 * lds);
    for (auto& b : c) b = fp8(r);
    for (auto& b : s) b = e8m0(r);
    return {P.up(c), P.up(s), ldc, lds};
}

// the reference: the 8-row passes of today's kernels (k_xfrag_q + k_gv_mx, stock records) into yref [32][groups*N]
inline void ref_passes(const Fp8Mat& Wstock, const Acts& a, int M, int groups, uint32_t* xf8, bf16* yref, int tpc, int ks, cudaStream_t st)
{
    const int K = Wstock.K, ldy = groups * Wstock.N;
    for (int r0 = 0; r0 < M; r0 += 8)
    {
        const int m = M - r0 < 8 ? M - r0 : 8;
        gv_xfrag_launch(a.codes + (size_t)r0 * a.ldc, a.ldc, a.sc + (size_t)r0 * a.lds, a.lds, m, K, groups, K, K / 32, xf8, st);
        gv_gemv_launch(xf8, Wstock, yref + (size_t)r0 * ldy, ldy, m, groups, Wstock.N, st, tpc, ks);
    }
}

enum Mode { STOCK = 0, QUAD = 1 };
struct Tally
{
    long mismatches = 0, checks = 0;
};

// one pass (gv_xfrag_launch + the GemvFn) vs the 8-row passes at every M = 1..32; plus the negative control at M = 24
inline void proj_case(const char* name, const Fp8Mat& Wstock, const Fp8Mat& Wrun, int groups, Mode mode, int tpc, int ks, Tally& T, bool& neg_ok)
{
    Pool P;
    cudaStream_t st = 0;
    const int K = Wstock.K, N = Wstock.N, ldy = groups * N;
    const Acts a = acts(P, K, groups, 1000u + (uint32_t)N + (uint32_t)K + 7u * (uint32_t)groups);
    const size_t xf_n = (size_t)4 * groups * K * 4, y_n = (size_t)32 * ldy;
    uint32_t* xf8 = P.alloc<uint32_t>((size_t)groups * K * 4);
    bf16* yref = P.alloc<bf16>(y_n);
    Guarded<uint32_t> xf = guarded<uint32_t>(P, xf_n, XF_CANARY);
    Guarded<uint16_t> y = guarded<uint16_t>(P, y_n, Y_CANARY);
    std::vector<uint16_t> href(y_n);
    char what[192];
    for (int M = 1; M <= 32; ++M)
    {
        const int nt = (M + 7) / 8;
        refill(xf, XF_CANARY);
        refill(y, Y_CANARY);
        GVNT_CK(cudaMemset(yref, 0, y_n * 2));
        ref_passes(Wstock, a, M, groups, xf8, yref, tpc, ks, st);
        gv_xfrag_launch(a.codes, a.ldc, a.sc, a.lds, M, K, groups, K, K / 32, xf.p(), st);
        if (mode == QUAD) gv_gemv_launch_q(xf.p(), Wrun, (bf16*)y.p(), ldy, M, groups, N, st, tpc, ks);
        else gv_gemv_launch(xf.p(), Wrun, (bf16*)y.p(), ldy, M, groups, N, st, tpc, ks);
        GVNT_CK(cudaDeviceSynchronize());
        GVNT_CK(cudaGetLastError());
        GVNT_CK(cudaMemcpy(href.data(), yref, y_n * 2, cudaMemcpyDeviceToHost));
        const std::vector<uint16_t> hy = host(y);
        long mm = 0;
        for (size_t i = 0; i < (size_t)M * ldy; ++i) mm += hy[GUARD + i] != href[i];
        T.mismatches += mm;
        T.checks += (long)M * ldy;
        snprintf(what, sizeof what, "%s groups=%d %s tpc=%d ks=%d M=%d: %ld bf16 bit mismatches vs the 8-row passes", name, groups, mode == QUAD ? "quad" : "stock",
                 tpc, ks, M, mm);
        require(mm == 0, what);
        snprintf(what, sizeof what, "%s groups=%d M=%d: output canary / rows >= M overwritten", name, groups, M);
        require(canaries_ok(hy, y_n, (size_t)M * ldy, Y_CANARY), what);
        snprintf(what, sizeof what, "%s groups=%d M=%d: fragment table canary / tiles >= nt overwritten", name, groups, M);
        require(canaries_ok(host(xf), xf_n, (size_t)groups * nt * K * 4, XF_CANARY), what);
    }
    // negative control: M = 24, row 13 of the last group negated -> only that row's outputs of that group may move (and some must)
    {
        const int M = 24, row = 13, g = groups - 1;
        GVNT_CK(cudaMemset(yref, 0, y_n * 2));
        ref_passes(Wstock, a, M, groups, xf8, yref, tpc, ks, st);
        uint8_t* crow = a.codes + (size_t)row * a.ldc + (size_t)g * K;
        std::vector<uint8_t> hrow(K), hneg(K);
        GVNT_CK(cudaMemcpy(hrow.data(), crow, K, cudaMemcpyDeviceToHost));
        for (int k = 0; k < K; ++k) hneg[k] = (uint8_t)(hrow[k] ^ 0x80u);
        GVNT_CK(cudaMemcpy(crow, hneg.data(), K, cudaMemcpyHostToDevice));
        refill(y, Y_CANARY);
        gv_xfrag_launch(a.codes, a.ldc, a.sc, a.lds, M, K, groups, K, K / 32, xf.p(), st);
        if (mode == QUAD) gv_gemv_launch_q(xf.p(), Wrun, (bf16*)y.p(), ldy, M, groups, N, st, tpc, ks);
        else gv_gemv_launch(xf.p(), Wrun, (bf16*)y.p(), ldy, M, groups, N, st, tpc, ks);
        GVNT_CK(cudaDeviceSynchronize());
        GVNT_CK(cudaGetLastError());
        GVNT_CK(cudaMemcpy(crow, hrow.data(), K, cudaMemcpyHostToDevice));
        GVNT_CK(cudaMemcpy(href.data(), yref, y_n * 2, cudaMemcpyDeviceToHost));
        const std::vector<uint16_t> hy = host(y);
        long inside = 0, outside = 0;
        for (int m = 0; m < M; ++m)
            for (int c = 0; c < ldy; ++c)
            {
                const bool d = hy[GUARD + (size_t)m * ldy + c] != href[(size_t)m * ldy + c];
                if (m == row && c / N == g) inside += d;
                else outside += d;
            }
        snprintf(what, sizeof what, "%s groups=%d negative control: %ld changed in row %d group %d, %ld elsewhere", name, groups, inside, row, g, outside);
        if (inside == 0 || outside != 0) neg_ok = false;
        require(inside > 0 && outside == 0, what);
    }
}

// idx wq_b (stock records [4096][1280]) in its engine launch shape (ds41_idx_ops.cuh: gv_gemv_launch(..., tpc 1, ks 0)) on the S.xf_qr table that
// the quad-repacked W.wq_b's proj_mm built at the same M, vs its own 8-row passes
inline void idx_wqb_case(Tally& T)
{
    Pool P;
    cudaStream_t st = 0;
    const int HEADS = 16, NQ = HEADS * HD, NIDX = 4096;
    const Fp8Mat Wqb_stock = mat(P, NQ, QA, 1, 41);
    Fp8Mat Wqb = gv_copy(P, Wqb_stock, 1);
    require(gv_quad_bind(Wqb, 1, 1, 0, st) == 1, "idx wq_b: W.wq_b quad repack refused");
    const Fp8Mat Widx = mat(P, NIDX, QA, 1, 42);
    const Acts a = acts(P, QA, 1, 43);
    g_fn_xfrag = gv_xfrag_launch;
    g_fn_gemv = gv_gemv_launch_q;
    g_use_gv = true;
    g_gv_tpc = 1;
    g_gv_ks = 0;
    g_gv_rows_max = 32;
    uint32_t* xf_qr = P.alloc<uint32_t>((size_t)Scratch::GV_TILES * QA * 4);
    uint32_t* xf8 = P.alloc<uint32_t>((size_t)QA * 4);
    bf16* q = P.alloc<bf16>((size_t)32 * NQ);
    bf16* yref = P.alloc<bf16>((size_t)32 * NIDX);
    Guarded<uint16_t> y = guarded<uint16_t>(P, (size_t)32 * NIDX, Y_CANARY);
    std::vector<uint16_t> href((size_t)32 * NIDX);
    char what[160];
    for (int M = 1; M <= 32; ++M)
    {
        refill(y, Y_CANARY);
        require(use_gv(Wqb, M), "idx wq_b: W.wq_b does not take the GEMV at cap 32");
        proj_mm(Wqb, a.codes, QA, a.sc, QA / 32, xf_qr, false, q, NQ, M, 1, 0, 0, 0, st);   // attn_proj_q's call: builds xf_qr
        gv_gemv_launch(xf_qr, Widx, (bf16*)y.p(), NIDX, M, 1, 0, st, 1, 0);                 // ds41_idx_ops.cuh's call
        ref_passes(Widx, a, M, 1, xf8, yref, 1, 0, st);
        GVNT_CK(cudaDeviceSynchronize());
        GVNT_CK(cudaGetLastError());
        GVNT_CK(cudaMemcpy(href.data(), yref, href.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<uint16_t> hy = host(y);
        long mm = 0;
        for (size_t i = 0; i < (size_t)M * NIDX; ++i) mm += hy[GUARD + i] != href[i];
        T.mismatches += mm;
        T.checks += (long)M * NIDX;
        snprintf(what, sizeof what, "idx wq_b M=%d: %ld bf16 bit mismatches vs the 8-row passes", M, mm);
        require(mm == 0, what);
        snprintf(what, sizeof what, "idx wq_b M=%d: output canary / rows >= M overwritten", M);
        require(canaries_ok(hy, y.n, (size_t)M * NIDX, Y_CANARY), what);
    }
    g_gv_rows_max = 8;
}

// project_rows_nt's GemvFn on quad records (gv_gemv_nt_launch -> one gv_gemv_launch_q pass) vs the 8-row passes
inline void nt_hook_quad_case(const char* name, const Fp8Mat& Wstock, const Fp8Mat& Wq, Tally& T)
{
    Pool P;
    cudaStream_t st = 0;
    const int K = Wstock.K, N = Wstock.N;
    const Acts a = acts(P, K, 1, 77u + (uint32_t)N);
    uint32_t* xf = P.alloc<uint32_t>((size_t)4 * K * 4);
    uint32_t* xf8 = P.alloc<uint32_t>((size_t)K * 4);
    bf16* y = P.alloc<bf16>((size_t)32 * N);
    bf16* yref = P.alloc<bf16>((size_t)32 * N);
    std::vector<uint16_t> hy((size_t)32 * N), href((size_t)32 * N);
    char what[160];
    for (int M = 1; M <= 32; ++M)
    {
        gv_xfrag_nt_launch(a.codes, a.ldc, a.sc, a.lds, M, K, xf, st);
        gv_gemv_nt_launch(xf, Wq, y, N, M, 1, 0, st, 1, 0);
        ref_passes(Wstock, a, M, 1, xf8, yref, 1, 0, st);
        GVNT_CK(cudaDeviceSynchronize());
        GVNT_CK(cudaGetLastError());
        GVNT_CK(cudaMemcpy(hy.data(), y, hy.size() * 2, cudaMemcpyDeviceToHost));
        GVNT_CK(cudaMemcpy(href.data(), yref, href.size() * 2, cudaMemcpyDeviceToHost));
        long mm = 0;
        for (size_t i = 0; i < (size_t)M * N; ++i) mm += hy[i] != href[i];
        T.mismatches += mm;
        T.checks += (long)M * N;
        snprintf(what, sizeof what, "%s gv_gemv_nt_launch quad M=%d: %ld bf16 bit mismatches vs the 8-row passes", name, M, mm);
        require(mm == 0, what);
    }
}

inline void gate_bitwise()
{
    Saved keep;
    Pool P;
    cudaStream_t st = 0;
    g_gv_quad.clear();
    struct Shape { const char* name; int N, K, groups; };
    const Shape shapes[] = {{"wq_a", QA, HID, 1},         {"wq_b", 16 * HD, QA, 1},     {"wkv", HD, HID, 1},
                            {"wo_a", OLORA, GROUP_IN, 2}, {"wo_a", OLORA, GROUP_IN, 4}, {"wo_b", HID, 2 * OLORA, 1}};
    Tally T;
    bool neg_ok = true;
    int quad_cases = 0, quad_fallbacks = 0;
    uint32_t seed = 11;
    for (const Shape& s : shapes)
    {
        const Fp8Mat W = mat(P, s.N, s.K, s.groups, seed++);
        proj_case(s.name, W, W, s.groups, STOCK, 1, 0, T, neg_ok);
        proj_case(s.name, W, W, s.groups, STOCK, 2, 8, T, neg_ok);
        Fp8Mat Wq = gv_copy(P, W, s.groups);
        // the bind rule: tpc 2 never takes quads; tpc 1 / default ks takes them exactly when gv_quad_fits
        require(gv_quad_bind(Wq, s.groups, 2, 0, st) == 0, "gv_quad_bind accepted tpc 2");
        const bool fits = gv_quad_fits(s.K, 1, 0);
        const int bound = gv_quad_bind(Wq, s.groups, 1, 0, st);
        require(bound == (fits ? 1 : 0), "gv_quad_bind disagrees with gv_quad_fits");
        GVNT_CK(cudaDeviceSynchronize());
        // quad records -> k_gv_mx_q<NT>; a shape that does not fit is not repacked and gv_gemv_launch_q runs the stock kernels (no abort)
        proj_case(s.name, W, Wq, s.groups, QUAD, 1, 0, T, neg_ok);
        if (bound)
        {
            ++quad_cases;
            if (s.groups == 1) nt_hook_quad_case(s.name, W, Wq, T);
            g_gv_quad.erase(Wq.gv);
        }
        else ++quad_fallbacks;
    }
    require(quad_cases == 4 && quad_fallbacks == 2, "quad coverage: expected wq_b, wo_a x2, wo_b repacked and wq_a, wkv on the stock fallback");
    idx_wqb_case(T);
    printf("# gv_nt bitwise: %ld outputs compared, %d quad record sets, %d non-fitting quad shapes on the stock path\n", T.checks, quad_cases, quad_fallbacks);
    require(neg_ok, "neg ctrl");
    printf("ATTN_GV_NT_BITWISE_PASS M=1..32 groups=1,2,4 quad=0,1 idx_wqb=PASS bit_mismatches=%ld neg_ctrl=%s\n", T.mismatches, neg_ok ? "PASS" : "FAIL");
}

// ---- H1 / H2: which rows take the GEMV, counted at the hooks (nothing is timed)
inline int g_cnt_xfrag = 0, g_cnt_gemv = 0;
inline void cnt_xfrag(const uint8_t* codes, int ldc, const uint8_t* sc, int lds, int M, int K, int groups, size_t gc, size_t gs, uint32_t* xf, cudaStream_t st)
{
    ++g_cnt_xfrag;
    gv_xfrag_launch(codes, ldc, sc, lds, M, K, groups, gc, gs, xf, st);
}
inline void cnt_gemv(const uint32_t* xf, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups, size_t y_gs, cudaStream_t st, int tpc, int ks)
{
    ++g_cnt_gemv;
    gv_gemv_launch_q(xf, W, Y, ldy, M, groups, y_gs, st, tpc, ks);
}

inline void gate_prefill_unchanged()
{
    Saved keep;
    Pool P;
    cudaStream_t st = 0;
    g_gv_quad.clear();
    const int HEADS = 16, G = HEADS / HPG, NMAX = 33, RR = 127 + NMAX;
    Weights W{};
    W.wq_a = mat(P, QA, HID, 1, 61);
    W.wq_b = mat(P, HEADS * HD, QA, 1, 62);
    W.wkv = mat(P, HD, HID, 1, 63);
    W.wo_a = mat(P, OLORA, GROUP_IN, G, 64);
    W.wo_b = mat(P, HID, G * OLORA, 1, 65);
    // the stock launch's quad records (GR8_DS41_GV_QUAD=1 at tpc 1): the three that fit
    require(gv_quad_bind(W.wq_b, 1, 1, 0, st) + gv_quad_bind(W.wo_a, G, 1, 0, st) + gv_quad_bind(W.wo_b, 1, 1, 0, st) == 3, "prefill gate: quad repack refused");
    {
        W.q_norm = (const bf16*)P.up(std::vector<uint16_t>(QA, 0x3F80));   // bf16 1.0
        W.kv_norm = (const bf16*)P.up(std::vector<uint16_t>(HD, 0x3F80));
        W.sink = P.up(std::vector<float>(HEADS, 0.f));
        float inv[32];
        for (int i = 0; i < 32; ++i) inv[i] = 1.0f / (float)(1 << (i % 16));
        W.inv = P.up(std::vector<float>(inv, inv + 32));
    }
    void* sbase = P.alloc<uint8_t>(Scratch::bytes(NMAX, HEADS));
    GVNT_CK(cudaMemset(sbase, 0, Scratch::bytes(NMAX, HEADS)));
    Scratch S;
    S.carve(sbase, NMAX, HEADS);
    Cache C{};
    C.ring = P.alloc<bf16>((size_t)RR * HD);
    C.ring_rows = RR;
    GVNT_CK(cudaMemset(C.ring, 0, (size_t)RR * HD * 2));
    bf16* y = P.alloc<bf16>((size_t)NMAX * HID);
    bf16* out = P.alloc<bf16>((size_t)NMAX * HID);
    {
        Rng r(66);
        std::vector<uint16_t> hy((size_t)NMAX * HID);
        for (auto& v : hy) v = (uint16_t)(0x3C00 + (r.next() % 0x0400)) ^ (uint16_t)((r.next() & 1) << 15);   // |v| in [2^-7, 2^-6)
        GVNT_CK(cudaMemcpy(y, hy.data(), hy.size() * 2, cudaMemcpyHostToDevice));   // S.o (attn_out's input) stays 0: the route does not depend on it
    }
    GVNT_CK(cudaDeviceSynchronize());
    g_fn_xfrag = cnt_xfrag;
    g_fn_gemv = cnt_gemv;
    g_use_gv = true;
    g_use_tc = true;
    g_gv_tpc = 1;
    g_gv_ks = 0;
    ds41dsp::g_dsp_gv = 1;
    char what[192];
    // one attn_proj + attn_out at n rows; the hook calls it made
    auto run = [&](int n, int& nx, int& ng) {
        g_cnt_xfrag = g_cnt_gemv = 0;
        attn_proj(W, C, S, y, n, 0, HEADS, st);
        attn_out(W, S, n, 0, HEADS, out, st);
        GVNT_CK(cudaDeviceSynchronize());
        GVNT_CK(cudaGetLastError());
        nx = g_cnt_xfrag;
        ng = g_cnt_gemv;
    };
    // GEMV at n: wq_a, wq_b, wo_a, wo_b build a table (wkv reuses wq_a's) = 4 xfrag calls; 5 GEMV calls. GEMM at n: 0 and 0.
    auto expect = [&](int n, bool gv, const char* mode) {
        int nx = 0, ng = 0;
        run(n, nx, ng);
        snprintf(what, sizeof what, "%s n=%d: %d xfrag + %d gemv hook calls (want %s)", mode, n, nx, ng, gv ? "4 + 5, the GEMV" : "0 + 0, the GEMM");
        require(gv ? (nx == 4 && ng == 5) : (nx == 0 && ng == 0), what);
        snprintf(what, sizeof what, "%s n=%d: ds41_attn_gv_ready's rule (use_gv(W.wq_b, n)) disagrees with attn_proj", mode, n);
        require(use_gv(W.wq_b, n) == gv, what);
    };
    // prefill / eager: the default cap
    require(g_gv_rows_max == 8, "the GEMV row cap is not 8 outside the decode / verify captures");
    for (int n : {1, 6, 8}) expect(n, true, "cap 8");
    for (int n : {9, 16, 29, 32, 33}) expect(n, false, "cap 8");
    // decode / verify: the rank TU's ds41_attn_set_gv_rows_max(32)
    set_gv_rows_max(32);
    require(g_gv_rows_max == 32, "set_gv_rows_max(32)");
    for (int n : {1, 8, 9, 16, 29, 32}) expect(n, true, "cap 32");
    expect(33, false, "cap 32");
    // v2 attn_proj_own: its InputBuf::xf holds one row tile, so both its proj_mm calls keep the 8-row cap at g_gv_rows_max 32
    {
        v2::InputBuf o;
        o.carve(P.alloc<uint8_t>(v2::InputBuf::bytes(32, 32)), 32, 32);
        auto v2_expect = [&](int n, int want_x, int want_g) {
            g_cnt_xfrag = g_cnt_gemv = 0;
            v2::attn_proj_own(W, o, y, nullptr, nullptr, 0, n, false, st);
            GVNT_CK(cudaDeviceSynchronize());
            GVNT_CK(cudaGetLastError());
            snprintf(what, sizeof what, "v2 attn_proj_own cap 32 n=%d: %d xfrag + %d gemv hook calls (want %d + %d)", n, g_cnt_xfrag, g_cnt_gemv, want_x, want_g);
            require(g_cnt_xfrag == want_x && g_cnt_gemv == want_g, what);
        };
        v2_expect(8, 1, 2);    // wq_a builds the table, wkv reuses it: the hooks are live
        v2_expect(16, 0, 0);   // past the 8-row cap: the GEMM, nothing written to o.xf
    }
    // DSpark: the explicit 8-row cap whatever g_gv_rows_max is
    bool dsp_ok = true;
    for (int n : {1, 6, 8}) dsp_ok = dsp_ok && ds41dsp::dsp_use_gv(W.wq_a, n) && ds41dsp::dsp_use_gv(W.wo_a, n);
    for (int n : {9, 16, 29, 32}) dsp_ok = dsp_ok && !ds41dsp::dsp_use_gv(W.wq_a, n) && !ds41dsp::dsp_use_gv(W.wq_b, n) && !ds41dsp::dsp_use_gv(W.wkv, n) &&
                                           !ds41dsp::dsp_use_gv(W.wo_a, n) && !ds41dsp::dsp_use_gv(W.wo_b, n);
    require(dsp_ok, "DSpark dsp_use_gv left the 8-row cap at g_gv_rows_max 32");
    // the clamp: 8..32
    set_gv_rows_max(64);
    require(g_gv_rows_max == 32, "set_gv_rows_max(64) not clamped to 32");
    set_gv_rows_max(1);
    require(g_gv_rows_max == 8, "set_gv_rows_max(1) not clamped to 8");
    set_gv_rows_max(8);
    for (int n : {9, 32}) expect(n, false, "cap 8 again");
    printf("ATTN_GV_NT_PREFILL_UNCHANGED n=9,16,29,32,33 dsp_cap8=PASS v2_cap8=PASS\n");
}

inline void run_all()
{
    gate_bitwise();
    gate_prefill_unchanged();
}
#undef GVNT_CK
}   // namespace gvnt
}   // namespace ds41a
