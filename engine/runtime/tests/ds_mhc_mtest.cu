// M-invariance gate for the ds_mhc ops and the BF16-trunk gate: an M-row call must equal M single-row calls,
// bit for bit, and the BF16-stored trunk must give the same bits as the FP32-stored trunk.
//   ds_mhc_mtest [--start S] [--reps R] [--H 5120] M [M ...]      M = 1..2048 (a prefill chunk width)
// One line per (op, trunk, M): "op M=<m> start=<s> us=<launch us> us_row=<us per row> bitexact=<yes|rounding|no>[(detail)]". The FP32-trunk lines
// keep the plain op name ("embed"); the BF16-trunk lines carry a suffix ("embed[bf16]"). Both trunk types run in this one binary, in lockstep, on the
// same inputs (ds_mhc::BuffersT<float> and BuffersT<__nv_bfloat16>), whatever -DDS_MHC_TRUNK_BF16 says about the engine build.
//   bitexact is a diagnostic: the pass bar is the model's top-1 against the reference model; the line still tells whether
//   a wide call changed any bit, and "rounding" (1-ulp BF16 only) is the expected size of a non-exact change. us_row is the cost per row.
//   Timing runs after all comparisons on the chain's final state (values drift; launch cost only).
//   bitexact: rows [0,M) of the M-row chain == the chain run one row at a time (bytes), and the canary holds: rows >= M of every buffer
//   still hold the 0x7B fill (so no op read-then-wrote, or wrote, a row it was not given), and the tile counters are back at zero.
//   rounding = only 1-ulp BF16 differences; no = anything else. The chain is embed, prepare(F32 fn), consume, prepare, consume_prepare,
//   prepare(BF16 fn), finish. "start" is the prompt position of row 0: token ids are derived from it (rows never depend on it).
//   trunkcmp: after each op, "trunkcmp <op> M=<m> identical=<yes|no> us_row_f32=<> us_row_bf16=<>" compares the FP32-trunk run with the
//   BF16-trunk run bit for bit: the trunk widened to FP32 (exact), mix / pre / post / comb, y, out, xpre, and the head fragment table words. The
//   claim under test is "the BF16 trunk is bit-identical" (the trunk only ever holds BF16-valued numbers); the gate is every trunkcmp line yes.
//   A line "trunkcmp chain M=<m> us_row_f32=<> us_row_bf16=<> speedup=<>" sums the six chain ops (embed .. finish): the BF16 trunk's measured gain.
//   finish_row0 (nothing in the engine calls it yet): finish with grid 1 and row0 = M-1 equals row M-1 of the full finish: out, xpre and
//   the fragment row 0 of tile 0 against fragment row (M-1)%8 of tile (M-1)/8; pad rows 1..7 are zero; tile 1 is untouched.
//   finish_frag: the head fragment table holds the BF16 pairs of out at the Xfrag index, pad rows to a multiple of 8 are zero, the tiles
//   past ceil(M/8) are untouched.
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I engine -o ds_mhc_mtest ds_mhc_mtest.cu
// Needs a GPU. The reference comparison at these M is ds_mhc_gpu_test with ds_ref.py case files.
// Device memory: per trunk type A (M+32 rows, canary margin) + B (M rows) + the block outputs, about 0.33 GB per 1000 rows for FP32 and 0.2 GB for BF16
// (1.1 GB together at M=2048); sets are freed per M.
#include "../../mhc/ds_mhc.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr int MXR = 2048, CANARY_ROWS = 32, NSB = 3, V = 2048;
constexpr unsigned char FILL = 0x7B;
static int g_bad = 0;

template <class T> static T* dev(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, FILL, n * sizeof(T))); return d; }
template <class T> static T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

// One buffer set of `rows` rows for trunk element type TT (float, or __nv_bfloat16 = the BF16 trunk). Every tensor is filled with FILL bytes first;
// done is zero. A (the wide call) gets M + CANARY_ROWS rows so an op that touches rows past M lands on FILL and is caught; B (the row-at-a-time
// reference) gets M rows and no fragment table.
template <class TT> struct SetT {
    int H, rows; ds_mhc::BuffersT<TT> b{}; __nv_bfloat16 *y, *out, *xpre; unsigned* frag = nullptr; size_t frag_words = 0;
    unsigned* done; int dw, done_n;   // dw chunk counters (done_words(rows)), then one one-row counter per row
    SetT(int H_, int rows_, bool with_frag) : H(H_), rows(rows_) {
        b.streams = dev<TT>((size_t)rows * 4 * H); b.mix = dev<float>((size_t)rows * 24); b.pre = dev<float>((size_t)rows * 4);
        b.post = dev<float>((size_t)rows * 4); b.comb = dev<float>((size_t)rows * 16);
        b.part = dev<float>((size_t)rows * ds_mhc::NV * (H / ds_mhc::PW));
        dw = ds_mhc::done_words(rows); done_n = dw + rows;
        CK(cudaMalloc(&done, (size_t)done_n * 4)); CK(cudaMemset(done, 0, (size_t)done_n * 4)); b.done = done;
        y = dev<__nv_bfloat16>((size_t)rows * H); out = dev<__nv_bfloat16>((size_t)rows * H); xpre = dev<__nv_bfloat16>((size_t)rows * H);
        if (with_frag) { frag_words = (size_t)((rows + 7) / 8) * (H / 16) * 64; frag = dev<unsigned>(frag_words); }
    }
    SetT(const SetT&) = delete; SetT& operator=(const SetT&) = delete;
    ~SetT() {
        cudaFree(b.streams); cudaFree(b.mix); cudaFree(b.pre); cudaFree(b.post); cudaFree(b.comb); cudaFree(b.part); cudaFree(done);
        cudaFree(y); cudaFree(out); cudaFree(xpre); if (frag) cudaFree(frag);
    }
    // row m as a one-row problem (its own tile counter)
    ds_mhc::BuffersT<TT> row(int m) const {
        ds_mhc::BuffersT<TT> r = b;
        r.streams += (size_t)m * 4 * H; r.mix += m * 24; r.pre += m * 4; r.post += m * 4; r.comb += m * 16;
        r.part += (size_t)m * ds_mhc::NV * (H / ds_mhc::PW); r.done = done + dw + m;
        return r;
    }
};
typedef SetT<float> SetF;
typedef SetT<__nv_bfloat16> SetB;
struct Cmp { size_t n = 0, mism = 0; double maxrel = 0; bool canary = true; };
static void cmp_f32(const float* a, const float* b, size_t n, Cmp& c) {
    for (size_t i = 0; i < n; ++i) if (memcmp(a + i, b + i, 4)) {
        ++c.mism; const double d = fabs((double)a[i] - b[i]), s = fmax(fabs((double)a[i]), fabs((double)b[i]));
        c.maxrel = fmax(c.maxrel, s > 0 ? d / s : 0);
    }
    c.n += n;
}
static void cmp_bf(const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n, Cmp& c) {
    for (size_t i = 0; i < n; ++i) if (memcmp(a + i, b + i, 2)) {
        ++c.mism; const double x = __bfloat162float(a[i]), z = __bfloat162float(b[i]), d = fabs(x - z), s = fmax(fabs(x), fabs(z));
        c.maxrel = fmax(c.maxrel, s > 0 ? d / s : 0);
    }
    c.n += n;
}
// the first n elements of a trunk tensor as FP32 (the BF16 widening is exact)
static std::vector<float> down_wide(const float* d, size_t n) { return down(d, n); }
static std::vector<float> down_wide(const __nv_bfloat16* d, size_t n) {
    const auto v = down(d, n); std::vector<float> w(n); for (size_t i = 0; i < n; ++i) w[i] = __bfloat162float(v[i]); return w;
}
static bool is_fill(const void* p, size_t bytes) { const unsigned char* q = (const unsigned char*)p; for (size_t i = 0; i < bytes; ++i) if (q[i] != FILL) return false; return true; }
// rows [M, rows) of A must still be FILL
template <class T> static void canary(const T* d, size_t per_row, int M, int rows, Cmp& c) {
    auto v = down(d + (size_t)M * per_row, (size_t)(rows - M) * per_row);
    if (!is_fill(v.data(), v.size() * sizeof(T))) c.canary = false;
}
// What the op wrote: compare rows [0,M) of A with B, check the canary and the counters. `which` selects the tensors the op touches.
enum { T_STREAMS = 1, T_STATE = 2, T_Y = 4, T_OUT = 8, T_XPRE = 16 };
template <class TT> static Cmp compare(const SetT<TT>& A, const SetT<TT>& B, int M, int H, int which) {
    Cmp c; const int R = A.rows;
    if (which & T_STREAMS) {
        auto a = down_wide(A.b.streams, (size_t)M * 4 * H), b = down_wide(B.b.streams, (size_t)M * 4 * H);
        cmp_f32(a.data(), b.data(), (size_t)M * 4 * H, c); canary(A.b.streams, (size_t)4 * H, M, R, c);
    }
    if (which & T_STATE) {
        float* pa[4] = {A.b.mix, A.b.pre, A.b.post, A.b.comb}; float* pb[4] = {B.b.mix, B.b.pre, B.b.post, B.b.comb};
        const int w[4] = {24, 4, 4, 16};
        for (int k = 0; k < 4; ++k) { auto a = down(pa[k], (size_t)R * w[k]), b = down(pb[k], (size_t)M * w[k]); cmp_f32(a.data(), b.data(), (size_t)M * w[k], c); canary(pa[k], w[k], M, R, c); }
    }
    if (which & T_Y) { auto a = down(A.y, (size_t)R * H), b = down(B.y, (size_t)M * H); cmp_bf(a.data(), b.data(), (size_t)M * H, c); canary(A.y, H, M, R, c); }
    if (which & T_OUT) { auto a = down(A.out, (size_t)R * H), b = down(B.out, (size_t)M * H); cmp_bf(a.data(), b.data(), (size_t)M * H, c); canary(A.out, H, M, R, c); }
    if (which & T_XPRE) { auto a = down(A.xpre, (size_t)R * H), b = down(B.xpre, (size_t)M * H); cmp_bf(a.data(), b.data(), (size_t)M * H, c); canary(A.xpre, H, M, R, c); }
    for (unsigned x : down(A.done, A.done_n)) if (x) c.canary = false;   // chunk counters (and the unused one-row ones) back at zero
    for (unsigned x : down(B.done, B.done_n)) if (x) c.canary = false;   // so are B's one-row ones (and its unused chunk ones)
    return c;
}
// The FP32-trunk wide run F against the BF16-trunk wide run G, bit for bit on rows [0,M): trunk widened to FP32, mix / pre / post / comb, y, out,
// xpre; frag also compares the whole head fragment table (pad rows and untouched tiles included).
static Cmp trunkcmp_sets(const SetF& F, const SetB& G, int M, int H, int which, bool frag) {
    Cmp c;
    if (which & T_STREAMS) { auto a = down_wide(F.b.streams, (size_t)M * 4 * H), b = down_wide(G.b.streams, (size_t)M * 4 * H); cmp_f32(a.data(), b.data(), a.size(), c); }
    if (which & T_STATE) {
        const float* pa[4] = {F.b.mix, F.b.pre, F.b.post, F.b.comb}; const float* pb[4] = {G.b.mix, G.b.pre, G.b.post, G.b.comb};
        const int w[4] = {24, 4, 4, 16};
        for (int k = 0; k < 4; ++k) { auto a = down(pa[k], (size_t)M * w[k]), b = down(pb[k], (size_t)M * w[k]); cmp_f32(a.data(), b.data(), a.size(), c); }
    }
    if (which & T_Y) { auto a = down(F.y, (size_t)M * H), b = down(G.y, (size_t)M * H); cmp_bf(a.data(), b.data(), a.size(), c); }
    if (which & T_OUT) { auto a = down(F.out, (size_t)M * H), b = down(G.out, (size_t)M * H); cmp_bf(a.data(), b.data(), a.size(), c); }
    if (which & T_XPRE) { auto a = down(F.xpre, (size_t)M * H), b = down(G.xpre, (size_t)M * H); cmp_bf(a.data(), b.data(), a.size(), c); }
    if (frag) {
        auto a = down(F.frag, F.frag_words), b = down(G.frag, G.frag_words);
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++c.mism;
        c.n += a.size();
    }
    return c;
}
struct Res { std::string op; Cmp c; bool extra_ok; std::string extra; std::function<void()> launch; };
// one op on both trunk types: f = FP32 trunk, b = BF16 trunk, x = their bit-for-bit comparison (has_x false: not compared)
struct Pair { std::string op; Res f, b; Cmp x; bool has_x, chain; };
static int g_xbad = 0;
static void report(const Res& r, int M, int start, double us) {
    const Cmp& c = r.c;
    const char* v = (c.mism == 0 && c.canary && r.extra_ok) ? "yes" : (c.canary && r.extra_ok && c.maxrel <= 0.0079) ? "rounding" : "no";
    char det[200] = "";
    if (strcmp(v, "yes")) snprintf(det, sizeof det, "(mism=%zu/%zu maxrel=%.2e canary=%s%s)", c.mism, c.n, c.maxrel, c.canary ? "ok" : "BROKEN", r.extra.c_str());
    printf("%-18s M=%-4d start=%-6d us=%-9.2f us_row=%-8.3f bitexact=%s%s\n", r.op.c_str(), M, start, us, us / M, v, det);
    if (strcmp(v, "yes")) ++g_bad;
}
static void report_x(const Pair& p, int M, double uf, double ub) {
    const bool same = p.x.mism == 0;
    char det[120] = "";
    if (!same) snprintf(det, sizeof det, " (mism=%zu/%zu maxrel=%.2e)", p.x.mism, p.x.n, p.x.maxrel);
    printf("trunkcmp %-18s M=%-4d identical=%s us_row_f32=%-8.3f us_row_bf16=%-8.3f%s\n", p.op.c_str(), M, same ? "yes" : "no", uf / M, ub / M, det);
    if (!same) ++g_xbad;
}
// The head fragment table: BF16 pairs of out at the Xfrag index (the same formula as flash_mhc::write_frag_pair), zero pad rows to a multiple of
// 8, the tiles past ceil(M/8) untouched. fidx = index of pair h of row r inside its tile.
static size_t fidx(int r, int h) { return (size_t)((h / 16) * 32 + (r % 8) * 4 + (h % 8) / 2) * 2 + (h / 8) % 2; }
template <class TT> static Res check_frag(const SetT<TT>& A, int M, int H, const char* name) {
    const size_t tile_words = (size_t)(H / 16) * 64;
    auto outv = down(A.out, (size_t)A.rows * H); auto fr = down(A.frag, A.frag_words);
    const size_t used = (size_t)((M + 7) / 8);
    size_t bad = 0, pad_bad = 0, untouched_bad = 0;
    for (int m = 0; m < (int)used * 8; ++m) for (int h = 0; h < H; h += 2) {
        const size_t dst = (size_t)(m / 8) * tile_words + fidx(m, h);
        unsigned want = 0;
        if (m < M) want = (unsigned)__bfloat16_as_ushort(outv[(size_t)m * H + h]) | ((unsigned)__bfloat16_as_ushort(outv[(size_t)m * H + h + 1]) << 16);
        if (fr[dst] != want) { if (m < M) ++bad; else ++pad_bad; }
    }
    for (size_t w = used * tile_words; w < A.frag_words; ++w) if (fr[w] != 0x7B7B7B7Bu) ++untouched_bad;
    Cmp c; c.canary = true; c.n = (size_t)M * H / 2; c.mism = bad + pad_bad + untouched_bad;
    char ex[96]; snprintf(ex, sizeof ex, " words=%zu pad=%zu untouched=%zu", bad, pad_bad, untouched_bad);
    return {name, c, true, ex, nullptr};
}
// finish_row0 against row M-1 of the full finish: out, xpre, fragment row 0 of tile 0 vs row (M-1)%8 of tile (M-1)/8, pad rows, tile 1.
// out1 / xpre1 / frag1 (2 tiles) are the one-row outputs of the row0 call.
template <class TT> static Res check_row0(const SetT<TT>& A, const __nv_bfloat16* out1, const __nv_bfloat16* xpre1, const unsigned* frag1, int M, int H, const char* name) {
    const size_t tile_words = (size_t)(H / 16) * 64;
    auto outv = down(A.out, (size_t)A.rows * H); auto fr = down(A.frag, A.frag_words);
    auto o1 = down(out1, H), x1 = down(xpre1, H), xa = down(A.xpre, (size_t)A.rows * H); auto f1 = down(frag1, 2 * tile_words);
    Cmp c; cmp_bf(o1.data(), outv.data() + (size_t)(M - 1) * H, H, c); const size_t out_bad = c.mism;
    cmp_bf(x1.data(), xa.data() + (size_t)(M - 1) * H, H, c); const size_t xpre_bad = c.mism - out_bad;
    size_t frag_bad = 0, pad_bad = 0, untouched_bad = 0;
    for (int h = 0; h < H; h += 2) {
        if (f1[fidx(0, h)] != fr[(size_t)((M - 1) / 8) * tile_words + fidx((M - 1) % 8, h)]) ++frag_bad;
        for (int r = 1; r < 8; ++r) if (f1[fidx(r, h)]) ++pad_bad;
    }
    for (size_t w = tile_words; w < 2 * tile_words; ++w) if (f1[w] != 0x7B7B7B7Bu) ++untouched_bad;
    c.n += H / 2; c.mism += frag_bad + pad_bad + untouched_bad;
    char ex[128]; snprintf(ex, sizeof ex, " out=%zu xpre=%zu words=%zu pad=%zu untouched=%zu", out_bad, xpre_bad, frag_bad, pad_bad, untouched_bad);
    return {name, c, true, ex, nullptr};
}

int main(int argc, char** argv) {
    int H = 5120, start = 0, reps = 200; std::vector<int> Ms;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--H") && i + 1 < argc) H = atoi(argv[++i]);
        else Ms.push_back(atoi(argv[i]));
    }
    if (Ms.empty()) { fprintf(stderr, "usage: ds_mhc_mtest [--start S] [--reps R] [--H 5120] M [M ...]   (M = 1..%d)\n", MXR); return 2; }
    int R = 1; for (const int M : Ms) { if (M < 1 || M > MXR) { printf("M=%d outside 1..%d\n", M, MXR); return 2; } R = M > R ? M : R; }
    ds_mhc::check_h(H);
    std::mt19937 rng(12345 + start); std::normal_distribution<float> nd(0.f, 1.f);
    // weights: 3 sub-layers of fn F32 / BF16, base, scale, norm; embedding table; 3 block outputs (R rows); final norm
    std::vector<float> fn[3], base[3], scale[3], blk[NSB]; std::vector<__nv_bfloat16> fnb[3], nrm[3], table((size_t)V * H), fnorm(H);
    for (int s = 0; s < 3; ++s) {
        fn[s].resize((size_t)24 * 4 * H); for (auto& x : fn[s]) x = 0.01f * nd(rng);
        fnb[s].resize(fn[s].size()); for (size_t i = 0; i < fn[s].size(); ++i) fnb[s][i] = __float2bfloat16_rn(fn[s][i]);
        base[s].resize(24); for (auto& x : base[s]) x = nd(rng);
        scale[s] = {0.2f + 0.1f * s, 0.3f, 0.5f};
        nrm[s].resize(H); for (auto& x : nrm[s]) x = __float2bfloat16_rn(1.f + 0.1f * nd(rng));
    }
    for (int s = 0; s < NSB; ++s) { blk[s].resize((size_t)R * H); for (auto& x : blk[s]) x = 2.f * nd(rng); }
    for (auto& x : table) x = __float2bfloat16_rn(nd(rng));
    for (auto& x : fnorm) x = __float2bfloat16_rn(1.f + 0.1f * nd(rng));
    float *dfn[3], *dbase[3], *dscale[3], *dblk[NSB]; __nv_bfloat16 *dfnb[3], *dnrm[3];
    for (int s = 0; s < 3; ++s) { dfn[s] = up(fn[s]); dfnb[s] = up(fnb[s]); dbase[s] = up(base[s]); dscale[s] = up(scale[s]); dnrm[s] = up(nrm[s]); }
    for (int s = 0; s < NSB; ++s) { dblk[s] = up(blk[s]); std::vector<float>().swap(blk[s]); }
    __nv_bfloat16* dtable = up(table); __nv_bfloat16* dfnorm = up(fnorm);
    const float eps = 1e-20f;
    ds_mhc::Weights<float> W[3]; ds_mhc::Weights<__nv_bfloat16> WB[3];
    for (int s = 0; s < 3; ++s) { W[s] = {dfn[s], dbase[s], dscale[s], dnrm[s]}; WB[s] = {dfnb[s], dbase[s], dscale[s], dnrm[s]}; }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (const int M : Ms) {
        std::vector<int> tok(M); for (int m = 0; m < M; ++m) tok[m] = (int)(((long long)(start + m) * 7919 + 13) % V);
        int* dtok = up(tok);
        std::vector<Pair> res;
        {   // scope: the sets are freed before the next M
        SetF AF(H, M + CANARY_ROWS, true), BF(H, M, false);       // FP32 trunk: wide call, row-at-a-time reference
        SetB AB(H, M + CANARY_ROWS, true), BB(H, M, false);       // BF16 trunk
        // ---- timing helper: the op on the chain's final state after every comparison is done (values drift; launch cost only)
        auto timeit = [&](const std::function<void()>& launch) {
            for (int i = 0; i < 5; ++i) launch();
            CK(cudaEventRecord(e0)); for (int i = 0; i < reps; ++i) launch(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return 1000.0 * ms / reps;
        };
        // ---- one op on both trunk types: body(A, B) runs the M-row call on A and M one-row calls on B, and returns the launch for timing
        auto run = [&](const char* name, int which, bool frag_cross, bool chain, auto&& body) {
            std::function<void()> lf = body(AF, BF), lb = body(AB, BB);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            res.push_back({name, {name, compare(AF, BF, M, H, which), true, "", lf}, {std::string(name) + "[bf16]", compare(AB, BB, M, H, which), true, "", lb},
                           trunkcmp_sets(AF, AB, M, H, which, frag_cross), true, chain});
        };
        run("embed", T_STREAMS | T_STATE, false, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::embed(A.b, dtok, dtable, M, H, 0);
            for (int m = 0; m < M; ++m) ds_mhc::embed(B.row(m), dtok + m, dtable, 1, H, 0);
            return [&] { ds_mhc::embed(A.b, dtok, dtable, M, H, 0); };
        });
        // ---- prepare (F32 fn), sub-layer 0
        run("prepare", T_STREAMS | T_STATE | T_Y, false, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::prepare<float>(A.b, W[0], M, H, eps, A.y, 0);
            for (int m = 0; m < M; ++m) ds_mhc::prepare<float>(B.row(m), W[0], 1, H, eps, B.y + (size_t)m * H, 0);
            return [&] { ds_mhc::prepare<float>(A.b, W[0], M, H, eps, A.y, 0); };
        });
        // ---- consume block 0 (separate kernel)
        run("consume", T_STREAMS | T_STATE, false, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::consume(A.b, flash_mhc::Fp32Block{dblk[0]}, M, H, 0);
            for (int m = 0; m < M; ++m) ds_mhc::consume(B.row(m), flash_mhc::Fp32Block{dblk[0] + (size_t)m * H}, 1, H, 0);
            return [&] { ds_mhc::consume(A.b, flash_mhc::Fp32Block{dblk[0]}, M, H, 0); };
        });
        // ---- prepare sub-layer 1, then the fused consume(block 1) + prepare sub-layer 2 (the prepare itself is not reported, as before; its
        // result is checked through consume_prepare, which reads what it wrote)
        run("consume_prepare", T_STREAMS | T_STATE | T_Y, false, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::prepare<float>(A.b, W[1], M, H, eps, A.y, 0);
            for (int m = 0; m < M; ++m) ds_mhc::prepare<float>(B.row(m), W[1], 1, H, eps, B.y + (size_t)m * H, 0);
            ds_mhc::consume_prepare<float>(A.b, flash_mhc::Fp32Block{dblk[1]}, W[2], M, H, eps, A.y, 0);
            for (int m = 0; m < M; ++m)
                ds_mhc::consume_prepare<float>(B.row(m), flash_mhc::Fp32Block{dblk[1] + (size_t)m * H}, W[2], 1, H, eps, B.y + (size_t)m * H, 0);
            return [&] { ds_mhc::consume_prepare<float>(A.b, flash_mhc::Fp32Block{dblk[1]}, W[1], M, H, eps, A.y, 0); };
        });
        // ---- prepare with the BF16 fn (the precision knob)
        run("prepare_bf16fn", T_STREAMS | T_STATE | T_Y, false, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::prepare<__nv_bfloat16>(A.b, WB[0], M, H, eps, A.y, 0);
            for (int m = 0; m < M; ++m) ds_mhc::prepare<__nv_bfloat16>(B.row(m), WB[0], 1, H, eps, B.y + (size_t)m * H, 0);
            return [&] { ds_mhc::prepare<__nv_bfloat16>(A.b, WB[0], M, H, eps, A.y, 0); };
        });
        // ---- finish_row0 (prefill stage C): grid 1, row0 = M-1, store=false so the state stays for the full finish. One-row outputs.
        const size_t tile_words = (size_t)(H / 16) * 64;
        __nv_bfloat16 *out1F = dev<__nv_bfloat16>(H), *xpre1F = dev<__nv_bfloat16>(H), *out1B = dev<__nv_bfloat16>(H), *xpre1B = dev<__nv_bfloat16>(H);
        unsigned *frag1F = dev<unsigned>(2 * tile_words), *frag1B = dev<unsigned>(2 * tile_words);   // tile 1 = untouched check
        ds_mhc::finish(AF.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, 1, H, eps, false, out1F, 0, nullptr, frag1F, xpre1F, M - 1);
        ds_mhc::finish(AB.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, 1, H, eps, false, out1B, 0, nullptr, frag1B, xpre1B, M - 1);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        // ---- finish: last consume + collapse + final norm (+ head fragments, + xpre)
        run("finish", T_STREAMS | T_OUT | T_XPRE, true, true, [&](auto& A, auto& B) -> std::function<void()> {
            ds_mhc::finish(A.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, M, H, eps, true, A.out, 0, nullptr, A.frag, A.xpre);
            for (int m = 0; m < M; ++m)
                ds_mhc::finish(B.row(m), flash_mhc::Fp32Block{dblk[2] + (size_t)m * H}, dfnorm, 1, H, eps, true, B.out + (size_t)m * H, 0, nullptr, nullptr, B.xpre + (size_t)m * H);
            return [&] { ds_mhc::finish(A.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, M, H, eps, false, A.out, 0, nullptr, A.frag); };
        });
        // ---- fragment table of the head (finish_frag) and finish_row0 against row M-1 of the full finish, per trunk type
        res.push_back({"finish_frag", check_frag(AF, M, H, "finish_frag"), check_frag(AB, M, H, "finish_frag[bf16]"), Cmp(), false, false});
        {
            Pair p{"finish_row0", check_row0(AF, out1F, xpre1F, frag1F, M, H, "finish_row0"), check_row0(AB, out1B, xpre1B, frag1B, M, H, "finish_row0[bf16]"), Cmp(), true, false};
            p.f.launch = [&] { ds_mhc::finish(AF.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, 1, H, eps, false, nullptr, 0, nullptr, frag1F, nullptr, M - 1); };
            p.b.launch = [&] { ds_mhc::finish(AB.b, flash_mhc::Fp32Block{dblk[2]}, dfnorm, 1, H, eps, false, nullptr, 0, nullptr, frag1B, nullptr, M - 1); };
            auto of = down(out1F, H), ob = down(out1B, H), xf = down(xpre1F, H), xb = down(xpre1B, H); auto ff = down(frag1F, 2 * tile_words), fb = down(frag1B, 2 * tile_words);
            cmp_bf(of.data(), ob.data(), H, p.x); cmp_bf(xf.data(), xb.data(), H, p.x);
            for (size_t i = 0; i < ff.size(); ++i) if (ff[i] != fb[i]) ++p.x.mism;
            p.x.n += ff.size();
            res.push_back(std::move(p));
        }
        CK(cudaFree(out1F)); CK(cudaFree(xpre1F)); CK(cudaFree(out1B)); CK(cudaFree(xpre1B));
        // ---- timing, then one line per op and trunk type, then the trunkcmp line
        double sf = 0, sb = 0;
        for (const Pair& p : res) {
            const double uf = p.f.launch ? timeit(p.f.launch) : 0.0, ub = p.b.launch ? timeit(p.b.launch) : 0.0;
            report(p.f, M, start, uf); report(p.b, M, start, ub);
            if (p.has_x) report_x(p, M, uf, ub);
            if (p.chain) { sf += uf; sb += ub; }
        }
        printf("trunkcmp %-18s M=%-4d us_row_f32=%-8.3f us_row_bf16=%-8.3f speedup=%.2fx\n", "chain", M, sf / M, sb / M, sb > 0 ? sf / sb : 0.0);
        CK(cudaFree(frag1F)); CK(cudaFree(frag1B));
        }
        fflush(stdout);
        CK(cudaFree(dtok));
    }
    printf("%s (%d line%s not bitexact=yes, %d trunkcmp line%s not identical=yes)\n", (g_bad || g_xbad) ? "FAIL" : "PASS", g_bad, g_bad == 1 ? "" : "s", g_xbad, g_xbad == 1 ? "" : "s");
    return (g_bad || g_xbad) ? 1 : 0;
}
