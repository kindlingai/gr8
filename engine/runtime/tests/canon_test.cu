// Session state independent of the layout (#152), on the engine's own kernels. One GPU, synthetic weights.
//   canon_test [--seed S]
// The canonical form (engine/kvman-rs/src/canon.rs, g6kv_canon_*) claims that a token's MLA row and indexer key row are
// the same bytes whichever rank and layout wrote them, so pages convert between layouts by copying rows. Checked here
// on rows the append path writes (g6_attn_append), then on the attention a decode step runs over them:
//   1. same layout: a session's caches -> canonical -> a fresh binding on the same pages is byte-identical, and the
//      blob's byte form is what the C interface returns;
//   2. another page list: canonical -> a binding whose block table is a random permutation of the pages; one decode
//      step (g6_attn_bind_run, full-index layer, then the shared-index layer on its lists) gives the same output bits,
//      index lists and appended rows as the step on the source session;
//   3. another layout (G6_CP builds): the same tokens appended on NR CP bindings (rank r stores the replicated prefix
//      and its pages past R), exported from all NR ranks, give the replicated binding's blob byte for byte; imported
//      into a replicated binding they equal its own caches, and a decode step (on the replicated bindings) after it
//      matches; and back: the replicated caches imported into NR CP bindings equal the CP bindings' own caches. The CP
//      run here is the CP map as built (list 0, 1, ..); other page lists past R are the Rust tests' (tests/canon.rs);
//   4. a live migration's two parts: rows [0, B) then [B, T) imported one after the other equal one import of [0, T).
// Build and run: TEST=canon_test scripts/c1_tests.sh <bin_dir> (links engine/kvman-rs's libg6kvman.a)
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../attn/bind.h"
#include "../../attn/kvfmt.h"
#include "../../g6nr.h"
#include "../../kvman-rs/g6kvman.h"

#define CK(x)                                                                                                         \
    do {                                                                                                              \
        const cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                      \
            printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));                   \
            exit(2);                                                                                                  \
        }                                                                                                             \
    } while (0)
#define CKI(x)                                                                                                        \
    do {                                                                                                              \
        const int e_ = (x);                                                                                           \
        if (e_ != 0) {                                                                                                \
            printf("CUDA error %d from %s at %s:%d\n", e_, #x, __FILE__, __LINE__);                                   \
            exit(2);                                                                                                  \
        }                                                                                                             \
    } while (0)

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const std::string& what)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 12345) {}
    uint64_t next()
    {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uni() { return (next() >> 40) * (1.0f / 16777216.0f); }
    float gauss() { return (uni() + uni() + uni() + uni() - 2.f) * 1.7320508f; }
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
};

struct Arena {
    std::vector<void*> v;
    template <class T> T* alloc(size_t n)
    {
        T* p = nullptr;
        CK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
        v.push_back(p);
        return p;
    }
    template <class T> T* up(const std::vector<T>& h)
    {
        T* p = alloc<T>(h.size());
        if (!h.empty()) CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    ~Arena() { for (void* p : v) cudaFree(p); }
};

static std::vector<__half> rand_h(size_t n, float sd, Rng& r)
{
    std::vector<__half> h(n);
    for (auto& x : h) x = __float2half(r.gauss() * sd);
    return h;
}
static std::vector<__nv_bfloat16> rand_b(size_t n, float sd, float mean, Rng& r)
{
    std::vector<__nv_bfloat16> h(n);
    for (auto& x : h) x = __float2bfloat16(mean + r.gauss() * sd);
    return h;
}

static cudaStream_t const ST = nullptr;
constexpr int NRK = g6nr::NR, NL = 2, MXR = 32, KVA = 576, IDXQ = 4096, IDXK = 128, IDXW = 32;
constexpr int LAYER[NL] = {6, 7};   // 6: full index, 7: shared index (reuses 6's lists)
constexpr int ROW = g6kv::ROW, IXR = g6kv::IDX_ROW;
constexpr uint8_t FLAGS_REP[NL] = {1, 0}, FLAGS_CP[NL] = {3, 2};

struct Pack {
    int h = 0;
    G6AttnPack p{};
};
static Pack make_pack(Arena& A, Rng& r, int rank, int layer)
{
    Pack k;
    const int group = (rank - layer % NRK + NRK) % NRK;
    k.h = g6nr::width(g6nr::HEADS, group);
    k.p.uk = G6AttnTensor{A.up(rand_b((size_t)k.h * 192 * 512, 0.1f, 0.f, r)), 3, {k.h, 192, 512}};
    k.p.uv = G6AttnTensor{A.up(rand_b((size_t)k.h * 256 * 512, 0.05f, 0.f, r)), 3, {k.h, 256, 512}};
    k.p.kv_norm = G6AttnTensor{A.up(rand_b(512, 0.1f, 1.f, r)), 1, {512, 0, 0}};
    if (layer == LAYER[0]) {
        k.p.index_norm = G6AttnTensor{A.up(rand_b(128, 0.1f, 1.f, r)), 1, {128, 0, 0}};
        k.p.index_bias = G6AttnTensor{A.up(rand_b(128, 0.1f, 0.f, r)), 1, {128, 0, 0}};
    }
    return k;
}

// tokens 0..n-1 of every layer: kv_a, index_k (layer 0)
struct Hist {
    int n = 0;
    int* pos = nullptr;
    __half* kva[NL] = {};
    __half* ik = nullptr;
};
static Hist make_hist(Arena& A, Rng& r, int n)
{
    Hist h;
    h.n = n;
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    h.pos = A.up(p);
    for (int l = 0; l < NL; ++l) h.kva[l] = A.up(rand_h((size_t)n * KVA, 1.f, r));
    h.ik = A.up(rand_h((size_t)n * IDXK, 1.f, r));
    return h;
}
// tokens [from, to) appended to a binding pair (layer 0, layer 1) by the engine's append path
static void append(G6AttnBinding* b, const Hist& h, int from, int to)
{
    for (int l = 0; l < NL; ++l)
        for (int off = from; off < to; off += MXR) {
            G6Attn a = b[l].args;
            a.rows = std::min(MXR, to - off);
            a.positions = h.pos + off;
            a.kv_a = h.kva[l] + (size_t)off * KVA;
            a.index_k = l == 0 ? h.ik + (size_t)off * IDXK : nullptr;
            a.reuse_indices = l == 0 ? nullptr : b[0].args.indices;
            a.reuse_row_map = nullptr;
            a.reuse_rows = l == 0 ? 0 : a.rows;
            CKI(g6_attn_append(&a, ST));
        }
    CK(cudaStreamSynchronize(ST));
}

// a binding's caches on the host
struct HostCache {
    std::vector<uint8_t> mla, idx;
};
static HostCache down(const G6AttnBinding& b)
{
    HostCache c;
    c.mla.resize((size_t)b.args.page_count * 64 * ROW);
    CK(cudaMemcpy(c.mla.data(), b.args.mla_cache, c.mla.size(), cudaMemcpyDeviceToHost));
    if (b.args.index_cache) {
        c.idx.resize((size_t)b.args.page_count * 64 * IXR);
        CK(cudaMemcpy(c.idx.data(), b.args.index_cache, c.idx.size(), cudaMemcpyDeviceToHost));
    }
    return c;
}
static void up(G6AttnBinding& b, const HostCache& c)
{
    CK(cudaMemcpy(b.args.mla_cache, c.mla.data(), c.mla.size(), cudaMemcpyHostToDevice));
    if (b.args.index_cache) CK(cudaMemcpy(b.args.index_cache, c.idx.data(), c.idx.size(), cudaMemcpyHostToDevice));
}

// export: caches[r][l] of `ranks` ranks at (nr, R), the run's list 0.. (place = the whole list)
static std::vector<uint8_t> export_rows(std::vector<std::vector<HostCache>>& caches, const uint8_t* flags, int nr, int R,
                                        const std::vector<uint32_t>& list, uint64_t first, uint64_t rows)
{
    const size_t ranks = caches.size();
    std::vector<const uint8_t*> m(ranks * NL), x(ranks * NL);
    std::vector<size_t> mb(ranks * NL), xb(ranks * NL);
    for (size_t r = 0; r < ranks; ++r)
        for (int l = 0; l < NL; ++l) {
            m[r * NL + l] = caches[r][l].mla.data();
            mb[r * NL + l] = caches[r][l].mla.size();
            x[r * NL + l] = caches[r][l].idx.empty() ? nullptr : caches[r][l].idx.data();
            xb[r * NL + l] = caches[r][l].idx.size();
        }
    size_t need = 0;
    int rc = g6kv_canon_export(ROW, NL, flags, nr, R, list.data(), list.size(), UINT32_MAX, m.data(), mb.data(), x.data(),
                               xb.data(), first, rows, 1, nullptr, 0, &need);
    if (rc != 1) { printf("FAIL export size: %s\n", g6kv_canon_error()); exit(1); }
    std::vector<uint8_t> blob(need);
    rc = g6kv_canon_export(ROW, NL, flags, nr, R, list.data(), list.size(), UINT32_MAX, m.data(), mb.data(), x.data(),
                           xb.data(), first, rows, 1, blob.data(), blob.size(), &need);
    if (rc != 0) { printf("FAIL export: %s\n", g6kv_canon_error()); exit(1); }
    return blob;
}
static uint64_t import_rows(const std::vector<uint8_t>& blob, std::vector<HostCache>& caches, const uint8_t* flags, int nr,
                            int R, const std::vector<uint32_t>& list, int rank)
{
    std::vector<uint8_t*> m(NL), x(NL);
    std::vector<size_t> mb(NL), xb(NL);
    for (int l = 0; l < NL; ++l) {
        m[l] = caches[l].mla.data();
        mb[l] = caches[l].mla.size();
        x[l] = caches[l].idx.empty() ? nullptr : caches[l].idx.data();
        xb[l] = caches[l].idx.size();
    }
    uint64_t w = 0;
    if (g6kv_canon_import(blob.data(), blob.size(), ROW, NL, flags, nr, R, list.data(), list.size(), UINT32_MAX, rank,
                          m.data(), mb.data(), x.data(), xb.data(), &w)) {
        printf("FAIL import: %s\n", g6kv_canon_error());
        exit(1);
    }
    return w;
}

// one decode step of M rows at positions ctx - M .. ctx - 1 on a binding pair: layer 0, then layer 1 on its lists
struct StepOut {
    std::vector<uint8_t> out[NL], idx;
    unsigned err = 0;
};
static StepOut step(G6AttnBinding* b, Arena& A, Rng& r, int M, int ctx, const int* heads, uint64_t seed)
{
    Rng in(seed);
    (void)r;
    std::vector<int> p(M);
    for (int i = 0; i < M; ++i) p[i] = ctx - M + i;
    const int* pos = A.up(p);
    StepOut o;
    for (int l = 0; l < NL; ++l) {
        G6AttnStep s{};
        s.rows = M;
        s.context = ctx;
        s.q_b = A.up(rand_h((size_t)M * heads[l] * 256, 2.f, in));
        s.kv_a = A.up(rand_h((size_t)M * KVA, 1.f, in));
        s.positions = pos;
        if (l == 0) {
            s.index_q = A.up(rand_h((size_t)M * IDXQ, 1.f, in));
            s.index_k = A.up(rand_h((size_t)M * IDXK, 1.f, in));
            s.index_weight = A.up(rand_h((size_t)M * IDXW, 0.5f, in));
        } else {
            s.reuse_indices = b[0].args.indices;
            s.reuse_rows = M;
        }
        CK(cudaMemset(b[l].args.error, 0, 4));
        CKI(g6_attn_bind_run(&b[l], &s, ST));
        CK(cudaStreamSynchronize(ST));
        o.out[l].resize((size_t)M * heads[l] * 256 * 2);
        CK(cudaMemcpy(o.out[l].data(), b[l].args.output, o.out[l].size(), cudaMemcpyDeviceToHost));
        unsigned e = 0;
        CK(cudaMemcpy(&e, b[l].args.error, 4, cudaMemcpyDeviceToHost));
        o.err |= e;
    }
    o.idx.resize((size_t)M * 2048 * 4);
    CK(cudaMemcpy(o.idx.data(), b[0].args.indices, o.idx.size(), cudaMemcpyDeviceToHost));
    return o;
}
static bool same(const StepOut& a, const StepOut& b)
{
    return a.out[0] == b.out[0] && a.out[1] == b.out[1] && a.idx == b.idx && !a.err && !b.err;
}

static void set_table(G6AttnBinding& b, const std::vector<int>& t)
{
    CK(cudaMemcpy(const_cast<int*>(b.args.block_table), t.data(), t.size() * 4, cudaMemcpyHostToDevice));
}

static void run(uint64_t seed, int T, int B, int M)
{
    Rng rng(seed);
    Arena A;
    const int P = (T + M + 63) / 64 + 2, cap = P * 64;
    char nm[160];
    snprintf(nm, sizeof nm, "seed %llu, %d tokens, step of %d rows, %d pages, NR %d, %s row", (unsigned long long)seed, T, M,
             P, NRK, g6kv::NAME);
    Pack pk[NRK][NL];
    for (int r = 0; r < NRK; ++r)
        for (int l = 0; l < NL; ++l) {
            pk[r][l] = make_pack(A, rng, r, LAYER[l]);
            // the norms are replicated weights, the same on every rank (only uk / uv split by head)
            if (r) {
                pk[r][l].p.kv_norm = pk[0][l].p.kv_norm;
                pk[r][l].p.index_norm = pk[0][l].p.index_norm;
                pk[r][l].p.index_bias = pk[0][l].p.index_bias;
            }
        }
    const int heads[NL] = {pk[0][0].h, pk[0][1].h};
    const size_t sb = std::max(g6_attn_bind_scratch_need(g6nr::HMIN, cap, MXR, 32), g6_attn_bind_scratch_need(g6nr::HMAX, cap, MXR, 32));
    void* scr = A.alloc<char>(sb);
    auto make = [&](G6AttnBinding* b, int rank) {
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_create_ext(&b[l], &pk[rank][l].p, rank, LAYER[l], cap, MXR, 32, ST, scr, sb));
        CK(cudaStreamSynchronize(ST));
    };
    const Hist h = make_hist(A, rng, T);
    std::vector<uint32_t> ident(P);
    for (int i = 0; i < P; ++i) ident[i] = i;
    const int Tp = (T + 63) / 64;
    std::vector<uint32_t> run_list(ident.begin(), ident.begin() + Tp);

    // source session: tokens 0..T-1 on a replicated binding (identity table)
    G6AttnBinding src[NL] = {};
    make(src, 0);
    append(src, h, 0, T);
    std::vector<std::vector<HostCache>> sc(1, std::vector<HostCache>{down(src[0]), down(src[1])});

    // 1. same layout, same pages
    {
        const auto blob = export_rows(sc, FLAGS_REP, 0, 0, run_list, 0, T);
        std::vector<HostCache> dst{down(src[0]), down(src[1])};
        for (auto& c : dst) { std::fill(c.mla.begin(), c.mla.end(), 0); std::fill(c.idx.begin(), c.idx.end(), 0); }
        const uint64_t w = import_rows(blob, dst, FLAGS_REP, 0, 0, run_list, 0);
        report(dst[0].mla == sc[0][0].mla && dst[0].idx == sc[0][0].idx && dst[1].mla == sc[0][1].mla && w == 2ull * T,
               std::string("same layout: export then import is byte-identical, ") + nm);
    }

    // 2. another page list: canonical -> a binding whose table is a permutation; the decode step matches
    {
        std::vector<int> perm(P);
        for (int i = 0; i < P; ++i) perm[i] = i;
        for (int i = P - 1; i > 0; --i) std::swap(perm[i], perm[rng.range(0, i)]);
        std::vector<uint32_t> plist(perm.begin(), perm.begin() + Tp);
        const auto blob = export_rows(sc, FLAGS_REP, 0, 0, run_list, 0, T);
        G6AttnBinding dst[NL] = {};
        make(dst, 0);
        for (int l = 0; l < NL; ++l) set_table(dst[l], perm);
        std::vector<HostCache> dc{down(dst[0]), down(dst[1])};
        import_rows(blob, dc, FLAGS_REP, 0, 0, plist, 0);
        for (int l = 0; l < NL; ++l) up(dst[l], dc[l]);
        const StepOut a = step(src, A, rng, M, T + M, heads, seed * 31 + 1);
        const StepOut b = step(dst, A, rng, M, T + M, heads, seed * 31 + 1);
        // the step appended rows T..T+M-1 on both: export both again, the same canonical form
        std::vector<std::vector<HostCache>> s2(1, std::vector<HostCache>{down(src[0]), down(src[1])});
        std::vector<std::vector<HostCache>> d2(1, std::vector<HostCache>{down(dst[0]), down(dst[1])});
        const int Tq = (T + M + 63) / 64;
        std::vector<uint32_t> l1(ident.begin(), ident.begin() + Tq), l2(perm.begin(), perm.begin() + Tq);
        const bool rows_same = export_rows(s2, FLAGS_REP, 0, 0, l1, 0, T + M) == export_rows(d2, FLAGS_REP, 0, 0, l2, 0, T + M);
        report(same(a, b) && rows_same, std::string("another page list: decode after import matches the source, ") + nm);
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&dst[l]));
        // the source took a step: rebuild it at T tokens for the cases below
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&src[l]));
        make(src, 0);
        append(src, h, 0, T);
    }

    // 4. bulk [0, B) then tail [B, T) == one copy of [0, T)
    {
        const auto b1 = export_rows(sc, FLAGS_REP, 0, 0, run_list, 0, B);
        const auto b2 = export_rows(sc, FLAGS_REP, 0, 0, run_list, B, T - B);
        const auto all = export_rows(sc, FLAGS_REP, 0, 0, run_list, 0, T);
        std::vector<HostCache> x{down(src[0]), down(src[1])}, y = x;
        for (auto* v : {&x, &y})
            for (auto& c : *v) { std::fill(c.mla.begin(), c.mla.end(), 0xA5); std::fill(c.idx.begin(), c.idx.end(), 0xA5); }
        const uint64_t w = import_rows(b1, x, FLAGS_REP, 0, 0, run_list, 0) + import_rows(b2, x, FLAGS_REP, 0, 0, run_list, 0);
        const uint64_t w2 = import_rows(all, y, FLAGS_REP, 0, 0, run_list, 0);
        report(w == w2 && x[0].mla == y[0].mla && x[0].idx == y[0].idx && x[1].mla == y[1].mla,
               std::string("bulk then tail equals one copy (B = ") + std::to_string(B) + "), " + nm);
    }

#if defined(G6_CP) && G6_CP
    // 3. another layout: NR CP bindings (R = 128 rows) <-> the replicated binding
    {
        const int R = 128;
        G6AttnBinding cp[NRK][NL] = {};
        for (int r = 0; r < NRK; ++r) {
            for (int l = 0; l < NL; ++l)
                CKI(g6_attn_bind_create_cp(&cp[r][l], &pk[r][l].p, r, LAYER[l], cap, MXR, 32, R, NRK, ST, scr, sb));
            CK(cudaStreamSynchronize(ST));
            append(cp[r], h, 0, T);   // each rank appends the rows of the pages it stores (CP_REMOTE_ROW skips the rest)
        }
        std::vector<std::vector<HostCache>> cc(NRK);
        for (int r = 0; r < NRK; ++r) cc[r] = {down(cp[r][0]), down(cp[r][1])};
        // the CP map as built is the run's list 0, 1, .. (bind.cu)
        const auto from_cp = export_rows(cc, FLAGS_CP, NRK, R, run_list, 0, T);
        const auto from_rep = export_rows(sc, FLAGS_REP, 0, 0, run_list, 0, T);
        // the canonical form does not depend on the layout: the same blob, byte for byte
        report(from_cp == from_rep, std::string("another layout: CP (R 128) and replicated export the same blob, ") + nm);
        // CP -> replicated: equal the replicated binding's own caches, and the decode step after it matches (the
        // decode runs on replicated bindings; the CP ranks are compared byte for byte below and in the other direction)
        const std::vector<uint8_t>& as_rep = from_cp;
        G6AttnBinding dst[NL] = {};
        make(dst, 0);
        std::vector<HostCache> dc{down(dst[0]), down(dst[1])};
        import_rows(as_rep, dc, FLAGS_REP, 0, 0, run_list, 0);
        for (int l = 0; l < NL; ++l) up(dst[l], dc[l]);
        const bool caches_same = dc[0].mla == sc[0][0].mla && dc[0].idx == sc[0][0].idx && dc[1].mla == sc[0][1].mla;
        const StepOut a = step(src, A, rng, M, T + M, heads, seed * 31 + 2);
        const StepOut b = step(dst, A, rng, M, T + M, heads, seed * 31 + 2);
        report(caches_same && same(a, b), std::string("CP -> replicated: the caches and the decode step match, ") + nm);
        // replicated -> CP: each rank's import equals that rank's own caches
        const std::vector<uint8_t>& as_cp = from_rep;
        bool all_same = true;
        for (int r = 0; r < NRK; ++r) {
            std::vector<HostCache> z = cc[r];
            for (auto& c : z) { std::fill(c.mla.begin(), c.mla.end(), 0); std::fill(c.idx.begin(), c.idx.end(), 0); }
            import_rows(as_cp, z, FLAGS_CP, NRK, R, run_list, r);
            all_same = all_same && z[0].mla == cc[r][0].mla && z[0].idx == cc[r][0].idx && z[1].mla == cc[r][1].mla;
        }
        report(all_same, std::string("replicated -> CP: every rank's caches equal its own, ") + nm);
        for (int r = 0; r < NRK; ++r)
            for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&cp[r][l]));
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&dst[l]));
    }
#endif
    for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&src[l]));
}

int main(int argc, char** argv)
{
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, dev));
    printf("canon_test: %s sm_%d%d, NR %d\n", pr.name, pr.major, pr.minor, NRK);
    // tokens, bulk split, step rows: under and past a page, past R = 128 by many pages, a long one past 2048 (top-k)
    run(seed, 100, 37, 1);
    run(seed + 1, 1000, 640, 4);
    run(seed + 2, 3000, 2999, 8);
    run(seed + 3, 4500, 64, 3);
    printf("canon_test: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
