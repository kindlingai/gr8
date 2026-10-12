// Batched attention unit test (S2): one GPU, no fabric, no pack files, synthetic weights.
//   s2_attn_test [--only rep|cp] [--seeds N] [--no-long]
// Claim under test: ONE attention call per layer for the rows of several sessions (per-row bounds and block table in device
// data, written by k_s2_cells from S2In) gives, for every row, the same bits as today's per-session calls (one replay entry
// per session on its own view of the pool, the output copied to the concatenated rows), and leaves the same KV bytes.
//   rep: replicated path. Pool of K runs (random base pages, runs shorter than the table), 2 layers: a full-index layer and a
//        shared-index layer that reuses it. Per-session calls (reference: view per slot, replay entry per session with the
//        session's own bucket) vs the batched call (batch view, one entry at the bucket of the longest session). Compared:
//        output rows, index lists, the whole mla / index caches of the two pools, error words (all 0). Plus validation
//        checks of the batched cell (a row at/over its own context, a repeated position inside one session -> error bits;
//        the same position in two sessions is legal). Then the same step on a copy of the pool with every page moved by a
//        random permutation, so each session's table is a shuffled, non-contiguous page list: per-session calls (in G6_CP
//        builds with the index cache moved by a second permutation, read through index_table) and the batched call must
//        give the reference's bits, and their caches must be the reference's caches, page for page.
//   cp:  context-parallel decode (cp_front / cp_index / cp_merge_index / cp_group / cp_finish), query tile 32. All NR ranks run
//        one after the other on this GPU (the exchange is plain device buffers), each with its own binding. Reference: one
//        standalone CP binding per (session, rank) in the plain form; batched: one CP binding per rank holding every
//        session's pages (per-row table with the session's physical pages, G6_CP_REMOTE for pages of other ranks; index keys on
//        the same pages). Compared: output rows of every rank's head group, merged index lists, latent and index caches.
// KV is filled by the append path itself (random fp16 inputs through g6_attn_append), the step's own rows are appended by the
// step. Row i of a session sits at position ctx - rows + i (a decode / verify step), so every row of a session sees its own
// causal bound.
// Build: scripts/c1_tests.sh <bin_dir> (EXTRA=-DDEC_S2: the test is skipped, exit 0, without it)
// Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef DEC_S2
int main()
{
    printf("SKIP s2_attn_test: build with -DDEC_S2 (EXTRA=-DDEC_S2 scripts/c1_tests.sh <bin_dir>)\n");
    return 0;
}
#else
#include "../../attn/bind.h"
#include "../../g6nr.h"
#include "../s2_cells.cuh"

using namespace g6rt;

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
#define CKI(x)                                                                                                  \
    do                                                                                                          \
    {                                                                                                           \
        const int e_ = (x);                                                                                     \
        if (e_ != 0)                                                                                            \
        {                                                                                                       \
            printf("CUDA error %d from %s at %s:%d\n", e_, #x, __FILE__, __LINE__);                             \
            exit(2);                                                                                            \
        }                                                                                                       \
    } while (0)

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const char* fmt, ...)
{
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("%s %s\n", ok ? "PASS" : "FAIL", buf);
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

constexpr int MXR = 32;               // rows of a step (G6_MX of the 32-row build) = the attention query tile
constexpr int TOPK = 2048;

struct Rng
{
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
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }     // inclusive
};

struct Arena
{
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
    void clear()
    {
        for (void* p : v) cudaFree(p);
        v.clear();
    }
    ~Arena() { clear(); }
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

constexpr int NRK = g6nr::NR;
constexpr int NL = 2;
constexpr int LAYER[NL] = {6, 7};     // layer 6: full index, layer 7: shared index (reuses layer 6's lists)
constexpr int IDXQ = 4096, IDXK = 128, IDXW = 32, KVA = 576;
constexpr int KV_B = 656, IX_B = 132;  // bytes per cached key (MLA, indexer)

// ---- synthetic attention pack of (rank, layer): random weights, shapes from the head-group rule
struct Pack
{
    int h = 0;
    __nv_bfloat16 *uk = nullptr, *uv = nullptr, *kvn = nullptr, *idn = nullptr, *idb = nullptr;
    G6AttnPack p{};
};
static Pack make_pack(Arena& A, Rng& r, int rank, int layer)
{
    Pack k;
    const int group = (rank - layer % NRK + NRK) % NRK;
    k.h = g6nr::width(g6nr::HEADS, group);
    k.uk = A.up(rand_b((size_t)k.h * 192 * 512, 0.1f, 0.f, r));
    k.uv = A.up(rand_b((size_t)k.h * 256 * 512, 0.05f, 0.f, r));
    k.kvn = A.up(rand_b(512, 0.1f, 1.f, r));
    k.p.uk = G6AttnTensor{k.uk, 3, {k.h, 192, 512}};
    k.p.uv = G6AttnTensor{k.uv, 3, {k.h, 256, 512}};
    k.p.kv_norm = G6AttnTensor{k.kvn, 1, {512, 0, 0}};
    if (layer == LAYER[0])
    {
        k.idn = A.up(rand_b(128, 0.1f, 1.f, r));
        k.idb = A.up(rand_b(128, 0.1f, 0.f, r));
        k.p.index_norm = G6AttnTensor{k.idn, 1, {128, 0, 0}};
        k.p.index_bias = G6AttnTensor{k.idb, 1, {128, 0, 0}};
    }
    return k;
}

static int bucket_of(int ctx, int cap)         // the engine's bucket list: 2048, 4096, ... then the capacity
{
    int b = 2048;
    while (b < ctx) b *= 2;
    return std::min(b, cap);
}

static bool dev_eq(const void* a, const void* b, size_t n, size_t* first = nullptr)
{
    std::vector<uint8_t> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n, cudaMemcpyDeviceToHost));
    if (!memcmp(x.data(), y.data(), n)) return true;
    size_t i = 0;
    while (x[i] == y[i]) ++i;
    if (first) *first = i;
    return false;
}
static bool sane(const __half* d, size_t n)    // all finite, not all zero
{
    std::vector<__half> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost));
    bool nz = false;
    for (const __half& x : h)
    {
        const float f = __half2float(x);
        if (!std::isfinite(f)) return false;
        nz |= f != 0.f;
    }
    return nz;
}
static unsigned dev_word(const unsigned* p)
{
    unsigned v = 0;
    CK(cudaMemcpy(&v, p, 4, cudaMemcpyDeviceToHost));
    return v;
}

// page g of src -> page perm[g] of dst (page-major caches of perm.size() pages, `bytes` per page)
static void permute_pages(void* dst, const void* src, const std::vector<int>& perm, size_t bytes)
{
    std::vector<uint8_t> a(perm.size() * bytes), b(a.size());
    CK(cudaMemcpy(a.data(), src, a.size(), cudaMemcpyDeviceToHost));
    for (size_t g = 0; g < perm.size(); ++g) memcpy(b.data() + (size_t)perm[g] * bytes, a.data() + g * bytes, bytes);
    CK(cudaMemcpy(dst, b.data(), b.size(), cudaMemcpyHostToDevice));
}
// page g of a == page perm[g] of b, for every g
static bool perm_eq(const void* a, const void* b, const std::vector<int>& perm, size_t bytes)
{
    std::vector<uint8_t> x(perm.size() * bytes), y(x.size());
    CK(cudaMemcpy(x.data(), a, x.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, y.size(), cudaMemcpyDeviceToHost));
    for (size_t g = 0; g < perm.size(); ++g)
        if (memcmp(x.data() + g * bytes, y.data() + (size_t)perm[g] * bytes, bytes)) return false;
    return true;
}

struct Chk
{
    std::string bad;
    void need(bool ok, const std::string& what)
    {
        if (!ok) bad += " " + what;
    }
    // rows of rowbytes each: first differing row, if any
    void rows(const void* a, const void* b, int n, size_t rowbytes, const std::string& what)
    {
        size_t at = 0;
        if (dev_eq(a, b, (size_t)n * rowbytes, &at)) return;
        bad += " " + what + "(row " + std::to_string(at / rowbytes) + ")";
    }
};

// random fp16 history of a session and layer, appended to a binding with g6_attn_append (the engine's own append path)
struct Hist
{
    int n = 0;
    int* pos = nullptr;
    __half *kva = nullptr, *ik = nullptr;
};
static Hist make_hist(Arena& A, Rng& r, int n, bool full)
{
    Hist h;
    h.n = n;
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    h.pos = A.up(p);
    h.kva = A.up(rand_h((size_t)n * KVA, 1.f, r));
    if (full) h.ik = A.up(rand_h((size_t)n * IDXK, 1.f, r));
    return h;
}
static void append_hist(const G6Attn& base, const Hist& h, bool full, int* dummy_reuse)
{
    for (int off = 0; off < h.n; off += MXR)
    {
        G6Attn a = base;
        a.rows = std::min(MXR, h.n - off);
        a.positions = h.pos + off;
        a.kv_a = h.kva + (size_t)off * KVA;
        a.index_k = full ? h.ik + (size_t)off * IDXK : nullptr;
        a.reuse_indices = full ? nullptr : dummy_reuse;
        a.reuse_row_map = nullptr;
        a.reuse_rows = full ? 0 : a.rows;
        CKI(g6_attn_append(&a, ST));
    }
}

// the step's inputs (rows of all sessions, session-major), one set per layer
struct Inputs
{
    __half* qb[NL] = {};
    __half* kva[NL] = {};
    __half *iq = nullptr, *ik = nullptr, *iw = nullptr;
    int* pos = nullptr;
};
static Inputs make_inputs(Arena& A, Rng& r, const int* heads, const std::vector<int>& pos)
{
    Inputs in;
    const int M = (int)pos.size();
    for (int l = 0; l < NL; ++l)
    {
        in.qb[l] = A.up(rand_h((size_t)M * heads[l] * 256, 2.f, r));
        in.kva[l] = A.up(rand_h((size_t)M * KVA, 1.f, r));
    }
    in.iq = A.up(rand_h((size_t)M * IDXQ, 1.f, r));
    in.ik = A.up(rand_h((size_t)M * IDXK, 1.f, r));
    in.iw = A.up(rand_h((size_t)M * IDXW, 0.5f, r));
    in.pos = A.up(pos);
    return in;
}
// rows r0..r0+n-1 of the step on layer l (l = 1: shared, reuses `reuse`)
static G6AttnStep mkstep(const Inputs& in, int l, int r0, int n, int ctx, int heads, const int* reuse)
{
    G6AttnStep s{};
    s.rows = n;
    s.context = ctx;
    s.q_b = in.qb[l] + (size_t)r0 * heads * 256;
    s.kv_a = in.kva[l] + (size_t)r0 * KVA;
    s.positions = in.pos + r0;
    if (l == 0)
    {
        s.index_q = in.iq + (size_t)r0 * IDXQ;
        s.index_k = in.ik + (size_t)r0 * IDXK;
        s.index_weight = in.iw + (size_t)r0 * IDXW;
    }
    else
    {
        s.reuse_indices = reuse;
        s.reuse_rows = n;
    }
    return s;
}

struct Case
{
    std::string name;
    int P;                                   // pages of one session's run (the view's table)
    std::vector<std::pair<int, int>> s;      // per session: rows, context (the step's rows sit at ctx - rows .. ctx - 1)
};

// ======================= replicated path =======================
// pool = one binding per layer (capacity K runs of P pages, identity table) + views: A: one per slot (per-session calls),
// B: one batch view per layer (output = cat). Same shape as S2Model (s2.cuh s2_init).
struct Pool
{
    G6AttnBinding ab[NL] = {};
    G6AttnBinding* sv[NL][S2N] = {};
    G6AttnBinding* bv[NL] = {};
    std::vector<std::unique_ptr<G6AttnBinding>> views;
    void release()
    {
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&ab[l]));
    }
};
static G6AttnBinding* make_view(Pool& pl, Arena& A, const G6AttnBinding& pool, int P, void* output)
{
    pl.views.emplace_back(new G6AttnBinding());
    G6AttnBinding* v = pl.views.back().get();
    *v = pool;
    v->storage_bytes = 0;
    v->max_rows = MXR;
    G6Attn& a = v->args;
    a.rows = MXR;
    a.context = pool.capacity;
    a.table_size = P;
    a.block_table = A.alloc<int>(P);
    a.indices = A.alloc<int>((size_t)MXR * 2048);
    a.error = A.alloc<unsigned>(1);
    a.output = output ? output : (void*)A.alloc<__half>((size_t)MXR * a.heads * 256);
    CK(cudaMemset(a.indices, 0xff, (size_t)MXR * 2048 * 4));
    return v;
}
static size_t scratch_for(int cap)
{
    return std::max(g6_attn_bind_scratch_need(g6nr::HMIN, cap, MXR, 32), g6_attn_bind_scratch_need(g6nr::HMAX, cap, MXR, 32));
}
static std::unique_ptr<Pool> make_pool(Arena& A, const Pack* pk, int rank, int K, int P, void* scr, size_t sb, __half* const* cat)
{
    std::unique_ptr<Pool> pl(new Pool());
    const int cap = K * P * 64;
    for (int l = 0; l < NL; ++l)
    {
        CKI(g6_attn_bind_create_ext(&pl->ab[l], &pk[l].p, rank, LAYER[l], cap, MXR, 32, ST, scr, sb));
        if (cat) pl->bv[l] = make_view(*pl, A, pl->ab[l], P, cat[l]);
        else
            for (int q = 0; q < K; ++q) pl->sv[l][q] = make_view(*pl, A, pl->ab[l], P, nullptr);
    }
    CK(cudaStreamSynchronize(ST));
    return pl;
}

static void run_rep(const Case& c, uint64_t seed)
{
    Rng rng(seed);
    Arena A;
    const int ns = (int)c.s.size(), P = c.P;
    const int K = std::min(S2N, ns + 2);              // pool slots; the sessions take an ascending subset of them
    std::vector<int> all(K), slots;
    for (int i = 0; i < K; ++i) all[i] = i;
    for (int i = K - 1; i > 0; --i) std::swap(all[i], all[rng.range(0, i)]);
    slots.assign(all.begin(), all.begin() + ns);
    std::sort(slots.begin(), slots.end());
    for (int i = K - 1; i > 0; --i) std::swap(all[i], all[rng.range(0, i)]);     // run bases: a permutation of K runs
    std::vector<int> base(K), grant(K, 0);
    for (int q = 0; q < K; ++q) base[q] = all[q] * P;
    int M = 0, maxctx = 0;
    for (const auto& x : c.s) { M += x.first; maxctx = std::max(maxctx, x.second); }
    char nm[256];
    snprintf(nm, sizeof nm, "rep %s: %d sessions, %d rows, P %d, bucket %d", c.name.c_str(), ns, M, P, bucket_of(maxctx, K * P * 64));

    Pack pk[NL];
    int heads[NL];
    for (int l = 0; l < NL; ++l) { pk[l] = make_pack(A, rng, 0, LAYER[l]); heads[l] = pk[l].h; }
    const int cap = K * P * 64;
    const size_t sb = scratch_for(cap);
    void* scr = A.alloc<char>(sb);
    __half* catA[NL];
    __half* catB[NL];
    for (int l = 0; l < NL; ++l) { catA[l] = A.alloc<__half>((size_t)MXR * heads[l] * 256); catB[l] = A.alloc<__half>((size_t)MXR * heads[l] * 256); }
    auto PA = make_pool(A, pk, 0, K, P, scr, sb, nullptr);
    auto PB = make_pool(A, pk, 0, K, P, scr, sb, catB);

    // runs: slot q's table = base + i over its grant, -1 after; slots without a session are unbound
    for (int j = 0; j < ns; ++j) grant[slots[j]] = std::min(P, (c.s[j].second + 63) / 64 + rng.range(0, 2));
    std::vector<const int*> tabs((size_t)S2N * NL, nullptr);
    for (int q = 0; q < K; ++q)
        for (int l = 0; l < NL; ++l)
        {
            std::vector<int> t(P);
            for (int i = 0; i < P; ++i) t[i] = i < grant[q] ? base[q] + i : -1;
            CK(cudaMemcpy(const_cast<int*>(PA->sv[l][q]->args.block_table), t.data(), P * 4, cudaMemcpyHostToDevice));
            tabs[(size_t)q * NL + l] = PA->sv[l][q]->args.block_table;
        }

    // history: every session's positions 0 .. ctx - rows - 1 through the append path of its A view, then A's caches -> B's
    for (int j = 0; j < ns; ++j)
        for (int l = 0; l < NL; ++l)
        {
            const Hist h = make_hist(A, rng, c.s[j].second - c.s[j].first, l == 0);
            if (h.n) append_hist(PA->sv[l][slots[j]]->args, h, l == 0, PA->sv[0][slots[j]]->args.indices);
        }
    for (int l = 0; l < NL; ++l)
    {
        CK(cudaMemcpy(PB->ab[l].args.mla_cache, PA->ab[l].args.mla_cache, (size_t)K * P * 64 * KV_B, cudaMemcpyDeviceToDevice));
        if (l == 0) CK(cudaMemcpy(PB->ab[l].args.index_cache, PA->ab[l].args.index_cache, (size_t)K * P * 64 * IX_B, cudaMemcpyDeviceToDevice));
    }
    // shuffled pages: pool page g of A sits at page pm[g] of C (per-session calls) and D (batched). C's index cache sits at
    // pi[g] when index_table exists (G6_CP), else at pm[g] as D's.
    std::vector<int> pm(K * P), pi(K * P);
    for (int g = 0; g < K * P; ++g) pm[g] = pi[g] = g;
    for (int g = K * P - 1; g > 0; --g) std::swap(pm[g], pm[rng.range(0, g)]);
#if defined(G6_CP) && G6_CP
    for (int g = K * P - 1; g > 0; --g) std::swap(pi[g], pi[rng.range(0, g)]);
#else
    pi = pm;
#endif
    __half* catC[NL];
    __half* catD[NL];
    for (int l = 0; l < NL; ++l) { catC[l] = A.alloc<__half>((size_t)MXR * heads[l] * 256); catD[l] = A.alloc<__half>((size_t)MXR * heads[l] * 256); }
    auto PC = make_pool(A, pk, 0, K, P, scr, sb, nullptr);
    auto PD = make_pool(A, pk, 0, K, P, scr, sb, catD);
    std::vector<const int*> tabsC((size_t)S2N * NL, nullptr);
    for (int q = 0; q < K; ++q)
    {
        std::vector<int> t(P, -1), ti(P, -1);
        for (int i = 0; i < grant[q]; ++i) { t[i] = pm[base[q] + i]; ti[i] = pi[base[q] + i]; }
        const int* dt = A.up(t);
        const int* di = A.up(ti);
        for (int l = 0; l < NL; ++l)
        {
            PC->sv[l][q]->args.block_table = dt;
#if defined(G6_CP) && G6_CP
            PC->sv[l][q]->args.index_table = di;
#endif
            tabsC[(size_t)q * NL + l] = dt;
        }
        (void)di;
    }
    for (int l = 0; l < NL; ++l)
        for (auto* X : {PC.get(), PD.get()})
        {
            permute_pages(X->ab[l].args.mla_cache, PA->ab[l].args.mla_cache, pm, (size_t)64 * KV_B);
            if (l == 0) permute_pages(X->ab[l].args.index_cache, PA->ab[l].args.index_cache, X == PC.get() ? pi : pm, (size_t)64 * IX_B);
        }

    // the step
    S2In h{};
    h.ns = ns;
    h.cls = 1;
    std::vector<int> pos;
    for (int j = 0; j < ns; ++j)
    {
        h.sess[j] = slots[j]; h.row0[j] = (int)pos.size(); h.rows[j] = c.s[j].first; h.ctx[j] = c.s[j].second;
        for (int i = 0; i < c.s[j].first; ++i) pos.push_back(c.s[j].second - c.s[j].first + i);
    }
    const Inputs in = make_inputs(A, rng, heads, pos);
    S2In* d_in = A.alloc<S2In>(1);
    CK(cudaMemcpy(d_in, &h, sizeof h, cudaMemcpyHostToDevice));
    G6AttnStepCell* cellsA = A.alloc<G6AttnStepCell>((size_t)S2N * NL);
    G6AttnRow* rowc = A.alloc<G6AttnRow>((size_t)NL * MXR);
    std::vector<G6AttnStepCell> bc(NL);
    memset(bc.data(), 0, bc.size() * sizeof(G6AttnStepCell));
    for (int l = 0; l < NL; ++l) bc[l].rows = rowc + (size_t)l * MXR;
    G6AttnStepCell* cellsB = A.up(bc);
    const int* d_shared = A.up(std::vector<int>{0, 1});
    const int** d_tabs = A.up(tabs);
    auto cells = [&]() {
        k_s2_cells<<<1, 128, 0, ST>>>(S2Cells{reinterpret_cast<const uint32_t*>(d_in), cellsA, d_shared, NL, cellsB, rowc, d_tabs, MXR, K * P});
        CK(cudaGetLastError());
    };
    cells();

    // A: today's per-session calls, one replay entry per session and layer at the session's own bucket, output copied to cat
    std::vector<std::array<G6AttnReplay, NL>> rpA(ns);
    for (int j = 0; j < ns; ++j)
        for (int l = 0; l < NL; ++l)
        {
            const int q = slots[j];
            const G6AttnStep st = mkstep(in, l, h.row0[j], h.rows[j], h.ctx[j], heads[l], PA->sv[0][q]->args.indices);
            memset(&rpA[j][l], 0, sizeof(G6AttnReplay));
            CKI(g6_attn_bind_replay_init(PA->sv[l][q], &st, cellsA + (size_t)q * NL + l, bucket_of(h.ctx[j], cap), &rpA[j][l], ST));
        }
    for (int j = 0; j < ns; ++j)
        for (int l = 0; l < NL; ++l)
        {
            CKI(g6_attn_replay_run(&rpA[j][l], ST));
            const size_t W = (size_t)heads[l] * 256;
            CK(cudaMemcpyAsync(catA[l] + (size_t)h.row0[j] * W, PA->sv[l][slots[j]]->args.output, (size_t)h.rows[j] * W * 2, cudaMemcpyDeviceToDevice, ST));
        }
    // B: one call per layer for every row, one entry at the bucket of the longest session
    std::array<G6AttnReplay, NL> rpB;
    for (int l = 0; l < NL; ++l)
    {
        G6AttnStep st = mkstep(in, l, 0, M, M, heads[l], PB->bv[0]->args.indices);
        st.s2_batched = 1;   // as s2_bstep: the batched cell, fixed-K decode kernels
        memset(&rpB[l], 0, sizeof(G6AttnReplay));
        CKI(g6_attn_bind_replay_init(PB->bv[l], &st, cellsB + l, bucket_of(maxctx, cap), &rpB[l], ST));
    }
    for (int l = 0; l < NL; ++l) CKI(g6_attn_replay_run(&rpB[l], ST));
    CK(cudaStreamSynchronize(ST));
    CK(cudaGetLastError());

    Chk k;
    for (int l = 0; l < NL; ++l)
    {
        const size_t W = (size_t)heads[l] * 256 * 2;
        k.rows(catA[l], catB[l], M, W, "out" + std::to_string(l));
        k.need(sane(catB[l], (size_t)M * heads[l] * 256), "out" + std::to_string(l) + "-finite");
        k.need(dev_eq(PA->ab[l].args.mla_cache, PB->ab[l].args.mla_cache, (size_t)K * P * 64 * KV_B), "mla" + std::to_string(l));
        if (l == 0) k.need(dev_eq(PA->ab[l].args.index_cache, PB->ab[l].args.index_cache, (size_t)K * P * 64 * IX_B), "idx-cache");
        k.need(dev_word(PB->bv[l]->args.error) == 0, "errB" + std::to_string(l));
    }
    for (int j = 0; j < ns; ++j)
    {
        k.rows(PA->sv[0][slots[j]]->args.indices, PB->bv[0]->args.indices + (size_t)h.row0[j] * 2048, h.rows[j], 2048 * 4, "lists(s" + std::to_string(j) + ")");
        for (int l = 0; l < NL; ++l) k.need(dev_word(PA->sv[l][slots[j]]->args.error) == 0, "errA");
    }

    // C: per-session calls through the shuffled tables; D: the batched call through them
    {
        G6AttnStepCell* cellsC = A.alloc<G6AttnStepCell>((size_t)S2N * NL);
        G6AttnRow* rowcD = A.alloc<G6AttnRow>((size_t)NL * MXR);
        for (int l = 0; l < NL; ++l) bc[l].rows = rowcD + (size_t)l * MXR;
        G6AttnStepCell* cellsD = A.up(bc);
        const int** d_tabsC = A.up(tabsC);
        k_s2_cells<<<1, 128, 0, ST>>>(S2Cells{reinterpret_cast<const uint32_t*>(d_in), cellsC, d_shared, NL, cellsD, rowcD, d_tabsC, MXR, K * P});
        CK(cudaGetLastError());
        std::vector<std::array<G6AttnReplay, NL>> rpC(ns);
        for (int j = 0; j < ns; ++j)
            for (int l = 0; l < NL; ++l)
            {
                const int q = slots[j];
                const G6AttnStep st = mkstep(in, l, h.row0[j], h.rows[j], h.ctx[j], heads[l], PC->sv[0][q]->args.indices);
                memset(&rpC[j][l], 0, sizeof(G6AttnReplay));
                CKI(g6_attn_bind_replay_init(PC->sv[l][q], &st, cellsC + (size_t)q * NL + l, bucket_of(h.ctx[j], cap), &rpC[j][l], ST));
            }
        for (int j = 0; j < ns; ++j)
            for (int l = 0; l < NL; ++l)
            {
                CKI(g6_attn_replay_run(&rpC[j][l], ST));
                const size_t W = (size_t)heads[l] * 256;
                CK(cudaMemcpyAsync(catC[l] + (size_t)h.row0[j] * W, PC->sv[l][slots[j]]->args.output, (size_t)h.rows[j] * W * 2, cudaMemcpyDeviceToDevice, ST));
            }
        std::array<G6AttnReplay, NL> rpD;
        for (int l = 0; l < NL; ++l)
        {
            G6AttnStep st = mkstep(in, l, 0, M, M, heads[l], PD->bv[0]->args.indices);
            st.s2_batched = 1;
            memset(&rpD[l], 0, sizeof(G6AttnReplay));
            CKI(g6_attn_bind_replay_init(PD->bv[l], &st, cellsD + l, bucket_of(maxctx, cap), &rpD[l], ST));
        }
        for (int l = 0; l < NL; ++l) CKI(g6_attn_replay_run(&rpD[l], ST));
        CK(cudaStreamSynchronize(ST));
        CK(cudaGetLastError());
        Chk ks;
        for (int l = 0; l < NL; ++l)
        {
            const size_t W = (size_t)heads[l] * 256 * 2;
            const std::string L = std::to_string(l);
            ks.rows(catA[l], catC[l], M, W, "outC" + L);
            ks.rows(catA[l], catD[l], M, W, "outD" + L);
            ks.need(perm_eq(PA->ab[l].args.mla_cache, PC->ab[l].args.mla_cache, pm, (size_t)64 * KV_B), "mlaC" + L);
            ks.need(perm_eq(PA->ab[l].args.mla_cache, PD->ab[l].args.mla_cache, pm, (size_t)64 * KV_B), "mlaD" + L);
            if (l == 0)
            {
                ks.need(perm_eq(PA->ab[l].args.index_cache, PC->ab[l].args.index_cache, pi, (size_t)64 * IX_B), "idx-cacheC");
                ks.need(perm_eq(PA->ab[l].args.index_cache, PD->ab[l].args.index_cache, pm, (size_t)64 * IX_B), "idx-cacheD");
            }
            ks.need(dev_word(PD->bv[l]->args.error) == 0, "errD" + L);
        }
        for (int j = 0; j < ns; ++j)
        {
            const std::string J = std::to_string(j);
            ks.rows(PA->sv[0][slots[j]]->args.indices, PC->sv[0][slots[j]]->args.indices, h.rows[j], 2048 * 4, "listsC(s" + J + ")");
            ks.rows(PA->sv[0][slots[j]]->args.indices, PD->bv[0]->args.indices + (size_t)h.row0[j] * 2048, h.rows[j], 2048 * 4, "listsD(s" + J + ")");
            for (int l = 0; l < NL; ++l) ks.need(dev_word(PC->sv[l][slots[j]]->args.error) == 0, "errC");
        }
        report(ks.bad.empty(), "%s, shuffled pages%s", nm, ks.bad.c_str());
    }

    // validation of the batched cell: a row at its own context; a repeated position inside one session
    {
        S2In g = h;
        g.ctx[0] -= 1;
        CK(cudaMemcpy(d_in, &g, sizeof g, cudaMemcpyHostToDevice));
        cells();
        CKI(g6_attn_replay_run(&rpB[0], ST));
        CK(cudaStreamSynchronize(ST));
        k.need((dev_word(PB->bv[0]->args.error) & 1) != 0, "neg-ctx");
        CK(cudaMemcpy(d_in, &h, sizeof h, cudaMemcpyHostToDevice));
        for (int j = 0; j < ns; ++j)
            if (h.rows[j] >= 2)
            {
                std::vector<int> p2 = pos;
                p2[h.row0[j] + 1] = p2[h.row0[j]];
                CK(cudaMemcpy(in.pos, p2.data(), p2.size() * 4, cudaMemcpyHostToDevice));
                cells();
                CKI(g6_attn_replay_run(&rpB[0], ST));
                CK(cudaStreamSynchronize(ST));
                k.need((dev_word(PB->bv[0]->args.error) & 1) != 0, "neg-dup");
                CK(cudaMemcpy(in.pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
                break;
            }
        cells();
        CKI(g6_attn_replay_run(&rpB[0], ST));
        CK(cudaStreamSynchronize(ST));
        k.need(dev_word(PB->bv[0]->args.error) == 0, "restored");
    }
    report(k.bad.empty(), "%s%s", nm, k.bad.c_str());
    PA->release();
    PB->release();
    PC->release();
    PD->release();
}

#if defined(G6_CP) && G6_CP
// ======================= context-parallel decode =======================
// NR ranks one after the other on this GPU. Per rank: binding per layer (head group of the rank), one scratch, plain buffers
// for what the fabric carries (cand: every rank's candidate lists, q / rope: each rank's absorbed query, part: the partials
// of a head group in shard order).
constexpr int CP_R = 256;

struct CpWorld
{
    G6AttnBinding b[NRK][NL] = {};
    G6AttnReplay rp[NRK][NL] = {};
    void release()
    {
        for (int r = 0; r < NRK; ++r)
            for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&b[r][l]));
    }
};
struct CpBuf
{
    float *q[NRK] = {}, *rope[NRK] = {}, *part[NRK] = {};
    uint64_t* cand = nullptr;
};
static size_t cp_blk(int M, int H) { return ((size_t)M * H * 514 + 3) & ~(size_t)3; }

// one layer of one world (a = fixed args per rank, cell = replay cell per rank or nullptr: plain form)
static void cp_layer(int l, G6Attn* const* a, const G6AttnStepCell* const* cell, void* const* scr, size_t sb, int M, CpBuf& bf)
{
    for (int r = 0; r < NRK; ++r) CKI(g6_attn_cp_front(a[r], cell[r], scr[r], sb, ST));
    for (int r = 0; r < NRK; ++r)
    {
        const float *q = nullptr, *rope = nullptr;
        CKI(g6_attn_cp_qabs(a[r], scr[r], &q, &rope));
        const int H = a[r]->heads;
        CK(cudaMemcpyAsync(bf.q[r], q, (size_t)M * H * 512 * 4, cudaMemcpyDeviceToDevice, ST));
        CK(cudaMemcpyAsync(bf.rope[r], rope, (size_t)M * H * 64 * 4, cudaMemcpyDeviceToDevice, ST));
    }
    for (int r = 0; r < NRK; ++r) CKI(g6_attn_cp_index(a[r], cell[r], scr[r], sb, bf.cand, ST));
    if (l == 0)
        for (int r = 0; r < NRK; ++r) CKI(g6_attn_cp_merge_index(a[r], cell[r], bf.cand, ST));
    for (int d = 0; d < NRK; ++d)
    {
        const int H = a[d]->heads;
        const size_t blk = cp_blk(M, H);
        for (int s = 0; s < NRK; ++s)
        {
            G6Attn ad = *a[s];
            ad.heads = H;
            CKI(g6_attn_cp_group(&ad, cell[s], scr[s], sb, bf.q[d], bf.rope[d], bf.part[d] + (size_t)s * blk, ST));
        }
    }
    for (int d = 0; d < NRK; ++d) CKI(g6_attn_cp_finish(a[d], cell[d], scr[d], sb, bf.part[d], ST));
}

static void run_cp(const Case& c, uint64_t seed)
{
    Rng rng(seed);
    Arena A;
    const int ns = (int)c.s.size(), P = c.P;
    int M = 0, maxctx = 0;
    for (const auto& x : c.s) { M += x.first; maxctx = std::max(maxctx, x.second); }
    char nm[256];
    snprintf(nm, sizeof nm, "cp %s: %d sessions, %d rows, P %d, R %d, %d ranks", c.name.c_str(), ns, M, P, CP_R, NRK);

    Pack pk[NRK][NL];
    for (int r = 0; r < NRK; ++r)
        for (int l = 0; l < NL; ++l) pk[r][l] = make_pack(A, rng, r, LAYER[l]);
    const int capR = P * 64, capB = ns * P * NRK * 64;
    const size_t sb = std::max(scratch_for(capR), scratch_for(capB));
    void* scr[NRK];
    for (int r = 0; r < NRK; ++r) scr[r] = A.alloc<char>(sb);
    std::vector<std::unique_ptr<CpWorld>> ref;
    for (int s = 0; s < ns; ++s) ref.emplace_back(new CpWorld());
    std::unique_ptr<CpWorld> bw(new CpWorld());
    for (int r = 0; r < NRK; ++r)
        for (int l = 0; l < NL; ++l)
        {
            for (int s = 0; s < ns; ++s)
                CKI(g6_attn_bind_create_cp(&ref[s]->b[r][l], &pk[r][l].p, r, LAYER[l], capR, MXR, 32, CP_R, NRK, ST, scr[r], sb));
            CKI(g6_attn_bind_create_cp(&bw->b[r][l], &pk[r][l].p, r, LAYER[l], capB, MXR, 32, CP_R, NRK, ST, scr[r], sb));
        }
    CK(cudaStreamSynchronize(ST));

    // history on the reference bindings (each rank appends the pages it stores), then copied into the batched binding:
    // MLA and index pages of session s at physical base s * PH (the reference map + base; index keys sit on the latent's pages)
    int PH[NRK];
    for (int r = 0; r < NRK; ++r)
    {
        PH[r] = ref[0]->b[r][0].args.page_count;
        if (ns * PH[r] > bw->b[r][0].args.page_count) { printf("FAIL %s: batched binding too small\n", nm); exit(1); }
    }
    for (int s = 0; s < ns; ++s)
        for (int l = 0; l < NL; ++l)
        {
            const Hist h = make_hist(A, rng, c.s[s].second - c.s[s].first, l == 0);
            if (h.n)
                for (int r = 0; r < NRK; ++r) append_hist(ref[s]->b[r][l].args, h, l == 0, ref[s]->b[r][0].args.indices);
        }
    CK(cudaStreamSynchronize(ST));
    for (int s = 0; s < ns; ++s)
        for (int r = 0; r < NRK; ++r)
            for (int l = 0; l < NL; ++l)
            {
                CK(cudaMemcpy((char*)bw->b[r][l].args.mla_cache + (size_t)s * PH[r] * 64 * KV_B, ref[s]->b[r][l].args.mla_cache,
                              (size_t)PH[r] * 64 * KV_B, cudaMemcpyDeviceToDevice));
                if (l == 0)
                    CK(cudaMemcpy((char*)bw->b[r][l].args.index_cache + (size_t)s * PH[r] * 64 * IX_B, ref[s]->b[r][l].args.index_cache,
                                  (size_t)PH[r] * 64 * IX_B, cudaMemcpyDeviceToDevice));
            }

    // the step
    int heads[NRK][NL];
    for (int r = 0; r < NRK; ++r)
        for (int l = 0; l < NL; ++l) heads[r][l] = pk[r][l].h;
    std::vector<int> pos, row0(ns);
    for (int s = 0; s < ns; ++s)
    {
        row0[s] = (int)pos.size();
        for (int i = 0; i < c.s[s].first; ++i) pos.push_back(c.s[s].second - c.s[s].first + i);
    }
    // activations are replicated over the ranks (kv_a, index inputs, positions); q_b is the rank's own heads
    Inputs in[NRK];
    {
        const int h0[NL] = {heads[0][0], heads[0][1]};
        in[0] = make_inputs(A, rng, h0, pos);
        for (int r = 1; r < NRK; ++r)
        {
            in[r] = in[0];
            for (int l = 0; l < NL; ++l) in[r].qb[l] = A.up(rand_h((size_t)M * heads[r][l] * 256, 2.f, rng));
        }
    }
    CpBuf bf;
    for (int d = 0; d < NRK; ++d)
    {
        const int H = std::max(heads[d][0], heads[d][1]);
        bf.q[d] = A.alloc<float>((size_t)MXR * H * 512);
        bf.rope[d] = A.alloc<float>((size_t)MXR * H * 64);
        bf.part[d] = A.alloc<float>((size_t)NRK * cp_blk(MXR, g6nr::HMAX));
    }
    bf.cand = A.alloc<uint64_t>((size_t)NRK * MXR * 2048);

    // reference: per session, plain form
    for (int s = 0; s < ns; ++s)
        for (int l = 0; l < NL; ++l)
        {
            G6Attn* a[NRK];
            const G6AttnStepCell* cell[NRK];
            for (int r = 0; r < NRK; ++r)
            {
                const G6AttnStep st = mkstep(in[r], l, row0[s], c.s[s].first, c.s[s].second, heads[r][l], ref[s]->b[r][0].args.indices);
                CKI(g6_attn_bind_prepare(&ref[s]->b[r][l], &st, ST));
                a[r] = &ref[s]->b[r][l].args;
                cell[r] = nullptr;
            }
            cp_layer(l, a, cell, scr, sb, c.s[s].first, bf);
        }
    // batched: per rank one entry over every row; per-row table = the reference map + the session's physical base
    std::vector<int*> dtab(NRK * ns);
    std::vector<G6AttnStepCell*> dcell(NRK * NL);
    for (int r = 0; r < NRK; ++r)
    {
        std::vector<int> t(P);
        CK(cudaMemcpy(t.data(), ref[0]->b[r][0].args.block_table, P * 4, cudaMemcpyDeviceToHost));
        std::vector<G6AttnRow> rows(M);
        int maxpages = 0;
        for (int s = 0; s < ns; ++s)
        {
            std::vector<int> ts(t);
            for (int& x : ts) if (x >= 0) x += s * PH[r];
            dtab[(size_t)r * ns + s] = A.up(ts);
            maxpages = std::max(maxpages, (c.s[s].second + 63) / 64);
            for (int i = 0; i < c.s[s].first; ++i)
                rows[row0[s] + i] = G6AttnRow{dtab[(size_t)r * ns + s], c.s[s].second, PH[r], (c.s[s].second + 63) / 64, s * P, s};
        }
        const G6AttnRow* drows = A.up(rows);
        for (int l = 0; l < NL; ++l)
        {
            G6AttnStepCell x{};
            x.context = maxctx; x.page_count = PH[r]; x.table_size = maxpages; x.reuse_rows = l ? M : 0; x.valid = 0; x.rows = drows;
            dcell[(size_t)r * NL + l] = A.up(std::vector<G6AttnStepCell>(1, x));
        }
    }
    for (int l = 0; l < NL; ++l)
    {
        G6Attn* a[NRK];
        const G6AttnStepCell* cell[NRK];
        for (int r = 0; r < NRK; ++r)
        {
            G6AttnStep st = mkstep(in[r], l, 0, M, M, heads[r][l], bw->b[r][0].args.indices);
            st.s2_batched = 1;
            memset(&bw->rp[r][l], 0, sizeof(G6AttnReplay));
            CKI(g6_attn_bind_replay_init(&bw->b[r][l], &st, dcell[(size_t)r * NL + l], capR, &bw->rp[r][l], ST));
            a[r] = &bw->rp[r][l].fixed;
            cell[r] = dcell[(size_t)r * NL + l];
        }
        cp_layer(l, a, cell, scr, sb, M, bf);
    }
    CK(cudaStreamSynchronize(ST));
    CK(cudaGetLastError());

    Chk k;
    for (int s = 0; s < ns; ++s)
        for (int r = 0; r < NRK; ++r)
            for (int l = 0; l < NL; ++l)
            {
                const std::string at = "(s" + std::to_string(s) + "r" + std::to_string(r) + "l" + std::to_string(l) + ")";
                const size_t W = (size_t)heads[r][l] * 256;
                k.rows((const __half*)bw->b[r][l].args.output + (size_t)row0[s] * W, ref[s]->b[r][l].args.output, c.s[s].first, W * 2, "out" + at);
                k.need(dev_word(ref[s]->b[r][l].args.error) == 0, "errA" + at);
                k.need(dev_eq((const char*)bw->b[r][l].args.mla_cache + (size_t)s * PH[r] * 64 * KV_B, ref[s]->b[r][l].args.mla_cache,
                              (size_t)PH[r] * 64 * KV_B), "mla" + at);
                if (l == 0)
                {
                    k.rows(bw->b[r][l].args.indices + (size_t)row0[s] * 2048, ref[s]->b[r][l].args.indices, c.s[s].first, 2048 * 4, "lists" + at);
                    k.need(dev_eq((const char*)bw->b[r][l].args.index_cache + (size_t)s * PH[r] * 64 * IX_B, ref[s]->b[r][l].args.index_cache,
                                  (size_t)PH[r] * 64 * IX_B), "idx-cache" + at);
                }
            }
    for (int r = 0; r < NRK; ++r)
        for (int l = 0; l < NL; ++l)
        {
            k.need(dev_word(bw->b[r][l].args.error) == 0, "errB(r" + std::to_string(r) + ")");
            k.need(sane((const __half*)bw->b[r][l].args.output, (size_t)M * heads[r][l] * 256), "out-finite(r" + std::to_string(r) + ")");
        }
    report(k.bad.empty(), "%s%s", nm, k.bad.c_str());
    for (auto& w : ref) w->release();
    bw->release();
}
#endif

int main(int argc, char** argv)
{
    std::string only;
    int seeds = 6;
    bool longcase = true;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-long")) longcase = false;
    }
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, dev));
    printf("s2_attn_test: %s sm_%d%d, NR %d\n", pr.name, pr.major, pr.minor, NRK);

    // rows, context per session; the step's rows sit at ctx - rows .. ctx - 1
    std::vector<Case> rep = {
        {"two sessions, 1 + 1 rows (small decode)", 16, {{1, 300}, {1, 700}}},
        {"2 + 1 rows, context under 2048", 32, {{2, 900}, {1, 1500}}},
        {"three sessions, mixed rows and contexts", 80, {{4, 1500}, {1, 200}, {3, 3000}}},
        {"four sessions, equal rows and context", 32, {{8, 1000}, {8, 1000}, {8, 1000}, {8, 1000}}},
        {"context across the 2048 bucket", 80, {{3, 130}, {2, 2100}, {5, 4097}}},
        {"short sessions in long runs (partial runs)", 64, {{1, 70}, {2, 200}, {1, 65}}},
        {"one row each, four sessions", 40, {{1, 64}, {1, 1999}, {1, 2049}, {1, 600}}},
        {"32 rows", 32, {{16, 900}, {16, 2000}}},
    };
    if (longcase)
    {
        rep.push_back({"long context, above 32768, both sessions", 520, {{1, 32900}, {4, 33200}}});
        rep.push_back({"long context with a short session", 520, {{2, 200}, {1, 33000}}});
    }
    for (int i = 0; i < seeds; ++i)
    {
        Rng r(1000 + i);
        const int ns = r.range(2, 4);
        Case c{"random seed " + std::to_string(1000 + i), 80, {}};
        int left = MXR;
        for (int j = 0; j < ns; ++j)
        {
            const int n = std::min(left - (ns - 1 - j), r.range(1, 8));
            c.s.push_back({n, r.range(std::max(n, 64), 80 * 64)});
            left -= n;
        }
        rep.push_back(c);
    }
    if (only.empty() || only == "rep")
        for (size_t i = 0; i < rep.size(); ++i) run_rep(rep[i], 7 + i);
#if defined(G6_CP) && G6_CP
    if (only.empty() || only == "cp")
    {
        std::vector<Case> cp = {
            {"two sessions", 16, {{1, 300}, {2, 1000}}},
            {"three sessions, mixed", 16, {{3, 200}, {1, 700}, {8, 1024}}},
            {"four sessions, equal", 16, {{2, 600}, {2, 600}, {2, 600}, {2, 600}}},
        };
        for (int i = 0; i < std::min(seeds, 3); ++i)
        {
            Rng r(2000 + i);
            const int ns = r.range(2, 4);
            Case c{"random seed " + std::to_string(2000 + i), 16, {}};
            for (int j = 0; j < ns; ++j)
            {
                const int n = r.range(1, 8);
                c.s.push_back({n, r.range(std::max(n, 64), 16 * 64)});
            }
            cp.push_back(c);
        }
        for (size_t i = 0; i < cp.size(); ++i) run_cp(cp[i], 31 + i);
    }
#endif
    printf("s2_attn_test: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
#endif
