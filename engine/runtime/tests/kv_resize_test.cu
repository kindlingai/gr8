// A KV pool that shrinks and grows at runtime (#153), on the engine's own kernels. One GPU, synthetic weights.
//   kv_resize_test [--seed S]
// The S2 page allocator (engine/kvman-rs, Pool::chunks / shrink / grow_pool) decides which chunks of pages are online,
// and engine/attn/kvchunks.cu backs exactly those in reserved virtual memory. Checked here:
//   1. two live sessions hold pages in chunks 0 and 2 (a third, retained, in chunk 3); shrink(2) takes chunks 1 and 3
//      offline, evicts the retained run and leaves the live ones; the caches lose the granules only those chunks held;
//   2. decode steps on both sessions before the shrink, while shrunk and after the grow give the same output bits,
//      index lists and appended rows as the same steps on plain cudaMalloc caches that never resized;
//   3. the live sessions' pages are byte-identical before and after the shrink and the grow, and pages that come back
//      read as zero;
//   4. a shrink while a live session holds a page in every chunk it could take waits (nothing unmapped).
// Build and run: TEST=kv_resize_test scripts/c1_tests.sh <bin_dir> (links engine/kvman-rs's libg6kvman.a and -lcuda)
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
#include "../../attn/ops.h"
#include "../../attn/kvchunks.h"
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
            printf("error %d from %s at %s:%d\n", e_, #x, __FILE__, __LINE__);                                        \
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
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 777) {}
    uint64_t next()
    {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uni() { return (next() >> 40) * (1.0f / 16777216.0f); }
    float gauss() { return (uni() + uni() + uni() + uni() - 2.f) * 1.7320508f; }
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
constexpr int NL = 2, MXR = 32, KVA = 576, IDXQ = 4096, IDXK = 128, IDXW = 32;
constexpr int LAYER[NL] = {6, 7};   // 6: full index, 7: shared index (reuses 6's lists)
constexpr int ROW = g6kv::ROW, IXR = g6kv::IDX_ROW;
constexpr uint32_t PAGES = 400, CHUNK = 100, NCHUNK = PAGES / CHUNK;   // ~4 MiB of MLA rows a chunk
constexpr size_t MLA_PAGE = 64 * (size_t)ROW, IDX_PAGE = 64 * (size_t)IXR;

struct Pack {
    int h = 0;
    G6AttnPack p{};
};
static Pack make_pack(Arena& A, Rng& r, int layer)
{
    Pack k;
    const int group = (0 - layer % g6nr::NR + g6nr::NR) % g6nr::NR;
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

// one session on one set of caches: a binding per layer whose MLA (and index) cache is the shared region, and whose
// block table is the session's page list
struct Sess {
    G6AttnBinding b[NL] = {};
};
static void bind(Sess& s, Pack* pk, void* scr, size_t sb, void* const* mla, void* idx, const std::vector<uint32_t>& list)
{
    std::vector<int> t(PAGES, (int)list[0]);
    for (size_t i = 0; i < list.size(); ++i) t[i] = (int)list[i];
    for (int l = 0; l < NL; ++l) {
        CKI(g6_attn_bind_create_ext(&s.b[l], &pk[l].p, 0, LAYER[l], PAGES * 64, MXR, 32, ST, scr, sb));
        s.b[l].args.mla_cache = mla[l];
        if (l == 0) s.b[l].args.index_cache = idx;
        CK(cudaMemcpy(const_cast<int*>(s.b[l].args.block_table), t.data(), t.size() * 4, cudaMemcpyHostToDevice));
    }
}
// tokens [0, n) appended by the engine's append path
static void prefill(Sess& s, Arena& A, Rng& r, int n)
{
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    const int* pos = A.up(p);
    const __half* kva[NL] = {A.up(rand_h((size_t)n * KVA, 1.f, r)), A.up(rand_h((size_t)n * KVA, 1.f, r))};
    const __half* ik = A.up(rand_h((size_t)n * IDXK, 1.f, r));
    for (int l = 0; l < NL; ++l)
        for (int off = 0; off < n; off += MXR) {
            G6Attn a = s.b[l].args;
            a.rows = std::min(MXR, n - off);
            a.positions = pos + off;
            a.kv_a = kva[l] + (size_t)off * KVA;
            a.index_k = l == 0 ? ik + (size_t)off * IDXK : nullptr;
            a.reuse_indices = l == 0 ? nullptr : s.b[0].args.indices;
            a.reuse_row_map = nullptr;
            a.reuse_rows = l == 0 ? 0 : a.rows;
            CKI(g6_attn_append(&a, ST));
        }
    CK(cudaStreamSynchronize(ST));
}

struct StepOut {
    std::vector<uint8_t> out[NL], idx;
    unsigned err = 0;
    bool operator==(const StepOut& o) const
    {
        return out[0] == o.out[0] && out[1] == o.out[1] && idx == o.idx && !err && !o.err;
    }
};
// one decode step of M rows ending at ctx, inputs from `seed`
static StepOut step(Sess& s, Arena&, const int* heads, int M, int ctx, uint64_t seed)
{
    Arena A;   // the step's inputs, freed when it returns (it synchronizes before)
    Rng in(seed);
    std::vector<int> p(M);
    for (int i = 0; i < M; ++i) p[i] = ctx - M + i;
    const int* pos = A.up(p);
    StepOut o;
    for (int l = 0; l < NL; ++l) {
        G6AttnStep st{};
        st.rows = M;
        st.context = ctx;
        st.q_b = A.up(rand_h((size_t)M * heads[l] * 256, 2.f, in));
        st.kv_a = A.up(rand_h((size_t)M * KVA, 1.f, in));
        st.positions = pos;
        if (l == 0) {
            st.index_q = A.up(rand_h((size_t)M * IDXQ, 1.f, in));
            st.index_k = A.up(rand_h((size_t)M * IDXK, 1.f, in));
            st.index_weight = A.up(rand_h((size_t)M * IDXW, 0.5f, in));
        } else {
            st.reuse_indices = s.b[0].args.indices;
            st.reuse_rows = M;
        }
        CKI(g6_attn_bind_run(&s.b[l], &st, ST));
        CK(cudaStreamSynchronize(ST));
        o.out[l].resize((size_t)M * heads[l] * 256 * 2);
        CK(cudaMemcpy(o.out[l].data(), s.b[l].args.output, o.out[l].size(), cudaMemcpyDeviceToHost));
        unsigned e = 0;
        CK(cudaMemcpy(&e, s.b[l].args.error, 4, cudaMemcpyDeviceToHost));
        o.err |= e;
    }
    o.idx.resize((size_t)M * 2048 * 4);
    CK(cudaMemcpy(o.idx.data(), s.b[0].args.indices, o.idx.size(), cudaMemcpyDeviceToHost));
    return o;
}

// the bytes of `pages` pages of every cache (MLA of both layers, index of layer 0)
static std::vector<uint8_t> pages_of(void* const* mla, void* idx, const std::vector<uint32_t>& pages)
{
    std::vector<uint8_t> v;
    for (int c = 0; c < 3; ++c) {
        const size_t pb = c < 2 ? MLA_PAGE : IDX_PAGE;
        const char* base = (const char*)(c < 2 ? mla[c] : idx);
        for (uint32_t p : pages) {
            const size_t at = v.size();
            v.resize(at + pb);
            CK(cudaMemcpy(v.data() + at, base + p * pb, pb, cudaMemcpyDeviceToHost));
        }
    }
    return v;
}

static std::vector<uint8_t> online(g6kv_pool* pool)
{
    std::vector<uint8_t> o(NCHUNK);
    for (uint32_t k = 0; k < NCHUNK; ++k) o[k] = (uint8_t)g6kv_chunk_online(pool, k);
    return o;
}

static void run(uint64_t seed)
{
    Rng rng(seed);
    Arena A;
    Pack pk[NL] = {make_pack(A, rng, LAYER[0]), make_pack(A, rng, LAYER[1])};
    const int heads[NL] = {pk[0].h, pk[1].h};
    const size_t sb = std::max(g6_attn_bind_scratch_need(g6nr::HMIN, PAGES * 64, MXR, 32),
                               g6_attn_bind_scratch_need(g6nr::HMAX, PAGES * 64, MXR, 32));
    void* scr = A.alloc<char>(sb);

    // the caches that resize (virtual memory, backed per granule) and the reference that never does (cudaMalloc)
    G6KvChunks* vm[3] = {};
    for (int c = 0; c < 3; ++c) CKI(g6_kvc_create(&vm[c], PAGES * (c < 2 ? MLA_PAGE : IDX_PAGE), 0));
    void* vmla[NL] = {g6_kvc_ptr(vm[0]), g6_kvc_ptr(vm[1])};
    void* vidx = g6_kvc_ptr(vm[2]);
    void* rmla[NL] = {A.alloc<char>(PAGES * MLA_PAGE), A.alloc<char>(PAGES * MLA_PAGE)};
    void* ridx = A.alloc<char>(PAGES * IDX_PAGE);
    size_t full = 0;
    for (auto* c : vm) full += g6_kvc_backed(c);

    // the pool: R = 64 pages; chunks of 100 pages
    g6kv_pool* pool = nullptr;
    CKI(g6kv_new(PAGES, PAGES, 64 * 64, &pool));
    g6kv_keep(pool, 0, 1, 0);
    CKI(g6kv_chunks(pool, CHUNK));
    auto list = [&](const g6kv_run& r) {
        const uint32_t* p; size_t n; int ex;
        CKI(g6kv_list(pool, &r, &p, &n, &ex));
        return std::vector<uint32_t>(p, p + n);
    };
    // session a: 2,600 tokens + room for 3 steps of 8 (pages 0..40); a filler to page 229; session b in chunk 2
    // (230..253); a filler to 299; a retained run in chunk 3 (300..339); then the fillers go, so chunk 1 is free
    const int TA = 2600, TB = 1500, M = 8;
    g6kv_run ra, rb, rk;
    std::vector<g6kv_run> fills;
    auto filler = [&](uint32_t pages) {   // runs below R (64 pages at most each)
        for (uint32_t left = pages; left;) {
            const uint32_t n = std::min(left, 64u);
            fills.emplace_back();
            CKI(g6kv_reserve(pool, n * 64, n * 64, &fills.back()));
            left -= n;
        }
    };
    CKI(g6kv_reserve(pool, TA + 3 * M, 64 * 64, &ra));
    filler(230 - 41);
    CKI(g6kv_reserve(pool, TB + 3 * M, 64 * 64, &rb));
    filler(300 - 254);
    CKI(g6kv_reserve(pool, 40 * 64, 40 * 64, &rk));
    const std::vector<uint32_t> la = list(ra), lb = list(rb), lk = list(rk);
    std::vector<int> hist(40 * 64 - 1);
    for (size_t i = 0; i < hist.size(); ++i) hist[i] = (int)i;
    CKI(g6kv_retain(pool, &rk, hist.data(), hist.size(), 40 * 64 - 2, 40 * 64 - 1, 40 * 64 - 2));
    for (const g6kv_run& x : fills) CKI(g6kv_release(pool, &x));
    if (la.size() != 41 || la.front() != 0 || lb.front() != 230 || lb.size() != 24 || lk.front() != 300) {
        printf("FAIL pool layout: a %u+%zu b %u+%zu kept %u\n", la.front(), la.size(), lb.front(), lb.size(), lk.front());
        exit(1);
    }
    char nm[200];
    snprintf(nm, sizeof nm, "seed %llu, a pages %u..%u, b pages %u..%u, retained %u..%u, %s row", (unsigned long long)seed,
             la.front(), la.back(), lb.front(), lb.back(), lk.front(), lk.back(), g6kv::NAME);

    Sess va, vb, wa, wb;   // v: resizing caches, w: reference
    bind(va, pk, scr, sb, vmla, vidx, la);
    bind(vb, pk, scr, sb, vmla, vidx, lb);
    bind(wa, pk, scr, sb, rmla, ridx, la);
    bind(wb, pk, scr, sb, rmla, ridx, lb);
    {
        Rng pa(seed * 3 + 1), pb(seed * 3 + 2);
        prefill(va, A, pa, TA);
        prefill(vb, A, pb, TB);
    }
    {
        Rng pa(seed * 3 + 1), pb(seed * 3 + 2);
        prefill(wa, A, pa, TA);
        prefill(wb, A, pb, TB);
    }
    std::vector<uint32_t> live = la;
    live.insert(live.end(), lb.begin(), lb.end());

    // before the shrink
    bool steps_same = step(va, A, heads, M, TA + M, seed * 11 + 1) == step(wa, A, heads, M, TA + M, seed * 11 + 1);
    steps_same = steps_same && step(vb, A, heads, M, TB + M, seed * 11 + 2) == step(wb, A, heads, M, TB + M, seed * 11 + 2);
    const auto before = pages_of(vmla, vidx, live);

    // 4. a shrink of 3 chunks: chunks 0 and 2 hold live pages, so it waits and nothing changes
    uint32_t got[NCHUNK];
    size_t n = 0;
    int done = 1;
    CKI(g6kv_shrink(pool, 3, got, NCHUNK, &n, &done));
    report(!done && online(pool) == std::vector<uint8_t>(NCHUNK, 1), std::string("a shrink into live chunks waits, ") + nm);

    // 1. shrink(2): chunks 1 and 3; the retained run in chunk 3 goes, the live runs stay
    CKI(g6kv_shrink(pool, 2, got, NCHUNK, &n, &done));
    const bool picked = done && n == 2 && got[0] == 1 && got[1] == 3 && g6kv_kept_runs(pool) == 0 && list(ra) == la &&
                        list(rb) == lb;
    const auto on = online(pool);
    size_t released = 0, rel = 0;
    for (int c = 0; c < 3; ++c) {
        CKI(g6_kvc_sync(vm[c], c < 2 ? MLA_PAGE : IDX_PAGE, CHUNK, PAGES, on.data(), NCHUNK, ST, nullptr, &rel));
        released += rel;
    }
    size_t shrunk = 0;
    for (auto* c : vm) shrunk += g6_kvc_backed(c);
    // independently: a granule stays when it holds a byte of an online chunk (0 or 2), in each cache
    const size_t g = g6_kvc_granule(vm[0]);
    size_t want = 0;
    for (int c = 0; c < 3; ++c) {
        const size_t pb = c < 2 ? MLA_PAGE : IDX_PAGE, bytes = PAGES * pb;
        for (size_t at = 0; at < bytes; at += g) {
            bool keep = false;
            for (uint32_t k : {0u, 2u}) {
                const size_t lo = k * CHUNK * pb, hi = (k + 1) * CHUNK * pb;
                keep = keep || (at < hi && at + g > lo);
            }
            want += keep ? g : 0;
        }
    }
    report(picked && shrunk == full - released && shrunk == want && released > 0,
           std::string("shrink(2) takes chunks 1 and 3 offline and leaves the live runs, ") + nm + ", backed " +
               std::to_string(full >> 20) + " -> " + std::to_string(shrunk >> 20) + " MiB (granule " +
               std::to_string(g >> 20) + " MiB)");
    const bool kept_shrunk = pages_of(vmla, vidx, live) == before;

    // 2. steps while shrunk
    steps_same = steps_same && step(va, A, heads, M, TA + 2 * M, seed * 11 + 3) == step(wa, A, heads, M, TA + 2 * M, seed * 11 + 3);
    steps_same = steps_same && step(vb, A, heads, M, TB + 2 * M, seed * 11 + 4) == step(wb, A, heads, M, TB + 2 * M, seed * 11 + 4);
    const auto mid = pages_of(vmla, vidx, live);

    // grow: the memory first, then the pool hands the chunks out
    std::vector<uint8_t> all(NCHUNK, 1);
    size_t mapped = 0, mp = 0;
    for (int c = 0; c < 3; ++c) {
        CKI(g6_kvc_sync(vm[c], c < 2 ? MLA_PAGE : IDX_PAGE, CHUNK, PAGES, all.data(), NCHUNK, ST, &mp, nullptr));
        mapped += mp;
    }
    CKI(g6kv_grow_pool(pool, 2, got, NCHUNK));
    size_t grown = 0;
    for (auto* c : vm) grown += g6_kvc_backed(c);
    const bool regrown = got[0] == 1 && got[1] == 3 && grown == full && mapped == released && online(pool) == all;
    const bool kept_grown = pages_of(vmla, vidx, live) == mid;
    // pages that came back read as zero: every page of chunk 3 that starts in a granule past the one holding the
    // last byte of chunk 2 (that granule stayed backed), in each cache
    bool zero = true;
    size_t zpages = 0;
    for (int c = 0; c < 3; ++c) {
        const size_t pb = c < 2 ? MLA_PAGE : IDX_PAGE;
        const char* base = (const char*)(c < 2 ? vmla[c] : vidx);
        for (uint32_t p = 3 * CHUNK; p < PAGES; ++p) {
            if ((size_t)p * pb / g <= ((size_t)3 * CHUNK * pb - 1) / g) continue;
            std::vector<uint8_t> v(pb);
            CK(cudaMemcpy(v.data(), base + p * pb, pb, cudaMemcpyDeviceToHost));
            zero = zero && std::all_of(v.begin(), v.end(), [](uint8_t x) { return x == 0; });
            ++zpages;
        }
    }
    zero = zero && zpages > 0;
    report(regrown && zero, std::string("grow(2) brings chunks 1 and 3 back, their pages zero, ") + nm);

    // 3. steps after the grow
    steps_same = steps_same && step(va, A, heads, M, TA + 3 * M, seed * 11 + 5) == step(wa, A, heads, M, TA + 3 * M, seed * 11 + 5);
    steps_same = steps_same && step(vb, A, heads, M, TB + 3 * M, seed * 11 + 6) == step(wb, A, heads, M, TB + 3 * M, seed * 11 + 6);
    report(kept_shrunk && kept_grown, std::string("live pages byte-identical across the shrink and the grow, ") + nm);
    report(steps_same && pages_of(vmla, vidx, live) == pages_of(rmla, ridx, live),
           std::string("decode before, during and after: output, lists and rows equal the caches that never resized, ") + nm);

    for (Sess* s : {&va, &vb, &wa, &wb})
        for (int l = 0; l < NL; ++l) CKI(g6_attn_bind_release(&s->b[l]));
    CKI(g6kv_release(pool, &ra));
    CKI(g6kv_release(pool, &rb));
    g6kv_free(pool);
    for (auto* c : vm) g6_kvc_free(c);
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
    printf("kv_resize_test: %s sm_%d%d, NR %d\n", pr.name, pr.major, pr.minor, g6nr::NR);
    run(seed);
    run(seed + 1);
    printf("kv_resize_test: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
