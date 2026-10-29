// GR8_FLASH_KV_CHUNKS (D-331 M7 part B): a Flash rank's MLA and index caches in kvchunks regions under one chunk map
// (runtime/flash_kv.h), on the Flash attention's own kernels. One GPU, synthetic weights, two DSA bindings (layers 3
// and 7) that share the map. Checked here:
//   1. masked boot: one chunk online backs exactly one chunk's granules in every region;
//   2. at every count of chunks online (1..4), decode steps on the chunked bindings give the same output bits and
//      index lists as the same steps on plain bindings (one cudaMalloc, GR8_FLASH_KV_CHUNKS=0), and a chunk that comes
//      online reads as zero; at the end every cache byte is the same;
//   3. a live session through shrink -> grow: decode, shrink to the chunks it uses, decode, grow, decode into the
//      chunks that came back: the same bits as the plain bindings that never resized;
//   4. with a host ledger: a shrink moves exactly the released bytes GPU -> node, a grow waits for the node to release
//      and moves them back; the book balances after each;
//   5. a crash after begin, between the sync and released, and between released and commit (a child process,
//      flashkv::Map::crash, which GR8_FLASH_KV_CRASH_AT sets), in both directions: the book balances, no byte is counted on both sides, and it recovers.
//   6. FLASH_CAP_POLICY=borrow-idle (s2/flash_lend.h) with the ledger booked as the launcher books it (the GPU's
//      lendable chunks): after a session, idle, the lend takes the map to the kept chunk; a short request is admitted
//      at once, a long one waits for a recall (timed); then a fresh session on the recalled chunks gives the same bits
//      as the plain bindings. A recall the node does not release in time leaves the request waiting: the map stays at
//      the kept chunk, and the rule still refuses it, so no step reaches an offline page.
// Build and run: TEST=flash_kv_chunks_test scripts/c1_tests.sh <bin_dir>, or by hand (scripts/c1_tests.sh has the line)
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../attn/bind.h"
#include "../../attn/ops.h"
#include "../../attn/kvchunks.h"
#include "../../attn/kvfmt.h"
#include "../../g6nr.h"
#include "../../knobs-rs/gr8_knobs.h"
#include "../../model_geom.h"
#include "../flash_kv.h"
#include "../s2/flash_lend.h"
#include <chrono>

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
static bool g_nonzero = false;   // a step's output had a nonzero byte (the bits compared are not trivially equal)
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
    float uni() { return (float)((next() >> 40) * (1.0 / (1ull << 24))) * 2.f - 1.f; }
};

constexpr int NB = 2, LAYERS[NB] = {3, 7}, RANK = 0, MR = 16, QT = 16, CHUNKS = 4;
constexpr int PAGE = g6kv::PAGE;

template <class T> static T* upload(const std::vector<T>& v)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static std::vector<__nv_bfloat16> bf(size_t n, Rng& r, float scale)
{
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16_rn(r.uni() * scale);
    return v;
}
static std::vector<__half> hf(size_t n, Rng& r, float scale)
{
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half_rn(r.uni() * scale);
    return v;
}
static G6AttnTensor tensor(void* p, std::initializer_list<int64_t> dims)
{
    G6AttnTensor t{};
    t.data = p;
    t.ndim = (int)dims.size();
    int j = 0;
    for (auto d : dims) t.shape[j++] = d;
    return t;
}

// a layer's weights (synthetic) and its step inputs for `rows` rows at positions from `pos`
struct Layer {
    G6AttnPack pack{};
    int heads = 0;
};
static Layer make_layer(int l, Rng& r)
{
    const int group = (RANK - l % g6nr::NR + g6nr::NR) % g6nr::NR;
    const int h = g6nr::width(g6nr::HEADS, group);
    Layer y;
    y.heads = h;
    y.pack.uk = tensor(upload(bf((size_t)h * geom::QK_NOPE * geom::KV_LORA, r, 0.05f)), {h, geom::QK_NOPE, geom::KV_LORA});
    y.pack.uv = tensor(upload(bf((size_t)h * geom::V_HEAD * geom::KV_LORA, r, 0.05f)), {h, geom::V_HEAD, geom::KV_LORA});
    y.pack.kv_norm = tensor(upload(bf(geom::KV_LORA, r, 1.f)), {geom::KV_LORA});
    y.pack.index_norm = tensor(upload(bf(g6kv::IDX_DIM, r, 1.f)), {g6kv::IDX_DIM});
    y.pack.index_bias = tensor(upload(bf(g6kv::IDX_DIM, r, 0.1f)), {g6kv::IDX_DIM});
    y.pack.index_ape = tensor(upload(bf((size_t)g6kv::KPOOL * g6kv::IDX_DIM, r, 0.1f)), {g6kv::KPOOL, g6kv::IDX_DIM});
    return y;
}

struct Inputs {
    void *q_b, *kv_a, *iq, *ik, *iw, *ig;
    int* pos;
};
static Inputs inputs(int heads)
{
    Inputs x{};
    CK(cudaMalloc(&x.q_b, (size_t)MR * heads * 256 * 2));
    CK(cudaMalloc(&x.kv_a, (size_t)MR * geom::KV_LORA * 2));
    CK(cudaMalloc(&x.iq, (size_t)MR * g6kv::IDX_HEADS * g6kv::IDX_DIM * 2));
    CK(cudaMalloc(&x.ik, (size_t)MR * g6kv::IDX_DIM * 2));
    CK(cudaMalloc(&x.iw, (size_t)MR * g6kv::IDX_HEADS * 2));
    CK(cudaMalloc(&x.ig, (size_t)MR * g6kv::IDX_DIM * 2));
    CK(cudaMalloc(&x.pos, MR * sizeof(int)));
    return x;
}
// random inputs for rows at positions pos..pos+rows-1 (one seed per position, so both binding sets see the same)
static void fill(const Inputs& x, int heads, int pos, int rows, int layer)
{
    Rng r(1000003ull * pos + layer);
    auto put = [&](void* d, size_t n, float s) {
        auto v = hf(n, r, s);
        CK(cudaMemcpy(d, v.data(), n * 2, cudaMemcpyHostToDevice));
    };
    put(x.q_b, (size_t)rows * heads * 256, 1.f);
    put(x.kv_a, (size_t)rows * geom::KV_LORA, 1.f);
    put(x.iq, (size_t)rows * g6kv::IDX_HEADS * g6kv::IDX_DIM, 1.f);
    put(x.ik, (size_t)rows * g6kv::IDX_DIM, 1.f);
    put(x.iw, (size_t)rows * g6kv::IDX_HEADS, 1.f);
    put(x.ig, (size_t)rows * g6kv::IDX_DIM, 1.f);
    std::vector<int> p(rows);
    for (int i = 0; i < rows; ++i) p[i] = pos + i;
    CK(cudaMemcpy(x.pos, p.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
}

struct Set {
    G6AttnBinding b[NB]{};
};

static std::vector<uint8_t> d2h(const void* p, size_t n)
{
    std::vector<uint8_t> v(n);
    CK(cudaMemcpy(v.data(), p, n, cudaMemcpyDeviceToHost));
    return v;
}

// decode tokens [from, to) on both sets, MR rows a step; every step's output and index lists compared bit for bit
static bool decode(Set& plain, Set& chunked, Layer* ly, Inputs* in, int from, int to, cudaStream_t st, std::string& why)
{
    for (int pos = from; pos < to; pos += MR)
    {
        const int rows = std::min(MR, to - pos);
        for (int k = 0; k < NB; ++k)
        {
            fill(in[k], ly[k].heads, pos, rows, LAYERS[k]);
            G6AttnStep s{};
            s.rows = rows;
            s.context = pos + rows;
            s.q_b = in[k].q_b;
            s.kv_a = in[k].kv_a;
            s.index_q = in[k].iq;
            s.index_k = in[k].ik;
            s.index_weight = in[k].iw;
            s.index_gate = in[k].ig;
            s.positions = in[k].pos;
            for (Set* set : {&plain, &chunked}) CKI(g6_attn_bind_run(&set->b[k], &s, st));
            CK(cudaStreamSynchronize(st));
            const size_t ob = (size_t)rows * ly[k].heads * geom::V_HEAD * 2, ib = (size_t)rows * g6kv::SEL * 4;
            const std::vector<uint8_t> po = d2h(plain.b[k].args.output, ob);
            for (size_t j = 0; !g_nonzero && j < po.size(); ++j) g_nonzero = po[j] != 0;
            if (po != d2h(chunked.b[k].args.output, ob) ||
                d2h(plain.b[k].args.indices, ib) != d2h(chunked.b[k].args.indices, ib))
            {
                why = "layer " + std::to_string(LAYERS[k]) + " step at position " + std::to_string(pos);
                return false;
            }
        }
    }
    return true;
}
// every cache byte of pages [0, pages) the same
static bool caches_equal(Set& plain, Set& chunked, uint32_t pages)
{
    for (int k = 0; k < NB; ++k)
    {
        if (d2h(plain.b[k].args.mla_cache, pages * flashkv::MLA_PAGE) != d2h(chunked.b[k].args.mla_cache, pages * flashkv::MLA_PAGE))
            return false;
        if (d2h(plain.b[k].args.index_cache, pages * flashkv::IDX_PAGE) != d2h(chunked.b[k].args.index_cache, pages * flashkv::IDX_PAGE))
            return false;
    }
    return true;
}
static bool zero_pages(const Set& chunked, uint32_t from, uint32_t to)
{
    for (int k = 0; k < NB; ++k)
    {
        auto m = d2h((const char*)chunked.b[k].args.mla_cache + from * flashkv::MLA_PAGE, (to - from) * flashkv::MLA_PAGE);
        auto i = d2h((const char*)chunked.b[k].args.index_cache + from * flashkv::IDX_PAGE, (to - from) * flashkv::IDX_PAGE);
        for (uint8_t b : m) if (b) return false;
        for (uint8_t b : i) if (b) return false;
    }
    return true;
}

// the ledger's book: gpu, node, free, the pending transfer's phase and bytes
struct Bk {
    uint64_t v[9];
    uint64_t gpu() const { return v[2]; }
    uint64_t node() const { return v[3]; }
    uint64_t free() const { return v[4]; }
    uint64_t pend() const { return v[6]; }
    uint64_t phase() const { return v[7]; }
    // every byte counted once: gpu + node + free + what is in flight (released, not committed) = total
    bool balanced() const { return v[2] + v[3] + v[4] + (v[7] == 2 ? v[8] : 0) == v[1]; }
};
static Bk book(const std::string& path)
{
    Bk b{};
    CKI(g6kv_ledger_show(path.c_str(), b.v));
    return b;
}
static std::string ledger_bin() { return knob::kvman_ledger(); }
static void ledger_init(const std::string& path, uint64_t total, uint64_t gpu, uint64_t node)
{
    // the book file as kvman-ledger init writes it (src/ledger.rs Book::to_text)
    std::string cmd = ledger_bin() + " " + path + " init --total " + std::to_string(total) + " --gpu " +
                      std::to_string(gpu) + " --node " + std::to_string(node) + " > /dev/null";
    if (system(cmd.c_str()) != 0) { printf("error: %s\n", cmd.c_str()); exit(2); }
}
// the KV node's side (kvman-remote --ledger releases a node -> gpu transfer once its leases fit): here, at once
struct FakeNode {
    std::string path;
    std::atomic<bool> stop{false};
    std::thread t;
    explicit FakeNode(std::string p) : path(std::move(p))
    {
        t = std::thread([this] {
            while (!stop)
            {
                uint64_t v[9];
                if (!g6kv_ledger_show(path.c_str(), v) && v[6] && v[7] == 1)
                {
                    // node -> gpu only; a gpu -> node transfer is the engine's to release
                    // (the book does not say the direction: the test starts this only around grows)
                    g6kv_ledger_released(path.c_str(), v[6], v[8]);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~FakeNode()
    {
        stop = true;
        t.join();
    }
};

// the child of a crash test: one binding's regions (all online), a ledger, then a resize that ends at the crash point
static int child(const char* point, const char* path, int dir)
{
    flashkv::Map map;
    CKI(map.init(CHUNKS * 256, CHUNKS, 0));
    G6KvChunks *m, *i;
    CKI(map.add(&m, &i));
    map.ledger = path;
    if (dir)
    {
        // grow: shrink first (no crash), then the crash in the grow
        CKI(map.resize(1, nullptr));
        map.crash = point;
        if (!strcmp(point, "begun")) map.resize(CHUNKS, nullptr);   // no node: the crash comes before any release
        else {
            FakeNode node(path);
            map.resize(CHUNKS, nullptr);
        }
    }
    else
    {
        map.crash = point;
        map.resize(1, nullptr);
    }
    return 0;   // not reached when the point is hit
}

int main(int argc, char** argv)
{
    if (argc == 5 && !strcmp(argv[1], "--child")) return child(argv[2], argv[3], atoi(argv[4]));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    Rng r(7);
    Layer ly[NB];
    Inputs in[NB];
    for (int k = 0; k < NB; ++k)
    {
        ly[k] = make_layer(LAYERS[k], r);
        in[k] = inputs(ly[k].heads);
    }
    // 1. the map, one chunk online at boot
    flashkv::Map map;
    CKI(map.init(CHUNKS * 256, 1, 0));
    report(map.chunk_pages == 256 || map.gran != (2u << 20),
           "chunk " + std::to_string(map.chunk_pages) + " pages at a granule of " + std::to_string(map.gran));
    const uint32_t cp = map.chunk_pages, pages = map.pages;
    const int capacity = (int)(pages * PAGE), cpt = (int)(cp * PAGE);
    Set plain, chunked;
    for (int k = 0; k < NB; ++k)
    {
        CKI(g6_attn_bind_create_ext(&plain.b[k], &ly[k].pack, RANK, LAYERS[k], capacity, MR, QT, st, nullptr, 0));
        G6KvChunks *m, *i;
        CKI(map.add(&m, &i));
        CKI(g6_attn_bind_create_chunked(&chunked.b[k], &ly[k].pack, RANK, LAYERS[k], capacity, MR, QT, st, m, i));
    }
    CK(cudaStreamSynchronize(st));
    report(map.backed() == map.bytes_at(1) && map.bytes_at(1) == (uint64_t)NB * cp * (flashkv::MLA_PAGE + flashkv::IDX_PAGE),
           "masked boot backs one chunk: " + std::to_string(map.backed()) + " bytes of " +
               std::to_string(map.bytes_at(CHUNKS)));
    report(plain.b[0].storage_bytes > chunked.b[0].storage_bytes + (size_t)pages * (flashkv::MLA_PAGE + flashkv::IDX_PAGE) - 1024,
           "the chunked binding's storage leaves the caches out");
    // 2. every count of chunks online
    std::string why;
    for (uint32_t n = 1; n <= CHUNKS; ++n)
    {
        if (n > 1)
        {
            CKI(map.resize(n, st));
            report(map.backed() == map.bytes_at(n) && zero_pages(chunked, (n - 1) * cp, n * cp),
                   "chunk " + std::to_string(n - 1) + " online, backed and zero");
        }
        const bool ok = decode(plain, chunked, ly, in, (int)(n - 1) * cpt, (int)n * cpt, st, why);
        report(ok, std::to_string(n) + " chunks online: decode to " + std::to_string(n * cpt) + " tokens, same bits" +
                       (ok ? "" : " (differs: " + why + ")"));
    }
    report(caches_equal(plain, chunked, pages), "all chunks: every cache byte the same");
    report(g_nonzero, "the outputs compared are not all zero");
    // 3. a live session through shrink -> grow
    for (int k = 0; k < NB; ++k)
    {
        CKI(g6_attn_bind_reset(&plain.b[k], st));
        CKI(g6_attn_bind_reset(&chunked.b[k], st));
    }
    bool ok = decode(plain, chunked, ly, in, 0, cpt + 77, st, why);
    CKI(map.resize(2, st));
    report(map.backed() == map.bytes_at(2) && map.tokens() == 2ull * cpt, "shrunk to 2 chunks under a live session");
    ok = ok && decode(plain, chunked, ly, in, cpt + 77, 2 * cpt - 5, st, why);
    CKI(map.resize(CHUNKS, st));
    const bool back_zero = zero_pages(chunked, 2 * cp, pages);
    ok = ok && decode(plain, chunked, ly, in, 2 * cpt - 5, 3 * cpt + 500, st, why);
    report(ok && back_zero, std::string("decode, shrink, decode, grow, decode: same bits") + (back_zero ? "" : " (grown pages not zero)") +
                                (ok ? "" : " (differs: " + why + ")"));
    report(caches_equal(plain, chunked, pages), "after the shrink and grow: every cache byte the same");
    // 4. with a ledger
    char dir[] = "/tmp/flash_kv_chunks_XXXXXX";
    if (!mkdtemp(dir)) { printf("error: mkdtemp\n"); return 2; }
    const std::string path = std::string(dir) + "/ledger";
    const uint64_t all = map.bytes_at(CHUNKS);
    ledger_init(path, all + (64ull << 20), all, 0);
    map.ledger = path;
    CKI(map.resize(1, st));
    Bk b = book(path);
    report(b.gpu() == map.bytes_at(1) && b.node() == all - map.bytes_at(1) && !b.pend() && b.balanced(),
           "ledger shrink to 1 chunk: " + std::to_string(b.node()) + " bytes to the node");
    {
        FakeNode node(path);
        CKI(map.resize(3, st));
    }
    b = book(path);
    report(b.gpu() == map.bytes_at(3) && b.node() == all - map.bytes_at(3) && !b.pend() && b.balanced() &&
               map.backed() == map.bytes_at(3),
           "ledger grow to 3 chunks: the node released, the GPU holds " + std::to_string(b.gpu()));
    map.wait_ms = 200;
    const int e = map.resize(CHUNKS, st);
    b = book(path);
    report(e && map.online == 3 && !b.pend() && b.gpu() == map.bytes_at(3) && b.balanced(),
           "a grow the node does not release in time is aborted: " + map.err);
    // 6. borrow-idle: lend while idle, recall before a request that needs the chunks
    {
        map.ledger.clear();
        map.wait_ms = 10000;
        CKI(map.resize(CHUNKS, st));
        g6s2::FlashLend f;
        f.idle_ms = 50; f.keep = 1; f.chunk_rows = (uint64_t)cpt; f.chunks = CHUNKS;
        const uint64_t lendable = map.bytes_at(CHUNKS) - map.bytes_at(f.keep);
        const std::string lp = std::string(dir) + "/ledger-borrow";
        ledger_init(lp, lendable + (64ull << 20), lendable, 64ull << 20);   // as the supervisor: node n, gpu lendable
        map.ledger = lp;
        for (int k = 0; k < NB; ++k)
        {
            CKI(g6_attn_bind_reset(&plain.b[k], st));
            CKI(g6_attn_bind_reset(&chunked.b[k], st));
        }
        bool same = decode(plain, chunked, ly, in, 0, cpt + 300, st, why);   // a session, then it ends: idle
        const auto t0 = g6s2::FlashLend::Clock::now();
        const bool early = !f.poll(true, t0) && !f.poll(true, t0 + std::chrono::milliseconds(49));
        const std::optional<uint32_t> lend = f.poll(true, t0 + std::chrono::milliseconds(50));
        CKI(map.resize(lend ? *lend : CHUNKS, st));
        Bk b = book(lp);
        report(early && lend == std::optional<uint32_t>(1) && map.online == 1 && map.backed() == map.bytes_at(1) &&
                   b.gpu() == 0 && b.node() == (64ull << 20) + lendable && !b.pend() && b.balanced(),
               "idle 50 ms: the lend takes the map to 1 chunk, " + std::to_string(lendable) + " bytes to the node");
        const uint64_t worst = 3ull * cpt + 500;   // a request that may reach 3 chunks and 500 rows: needs 4
        report(f.recall(1000, map.online) == 0 && f.recall(worst, map.online) == 4,
               "a short request is admitted on the kept chunk; a long one waits for a recall to 4 chunks");
        double recall_ms = 0;
        {
            FakeNode node(lp);
            const auto r0 = std::chrono::steady_clock::now();
            CKI(map.resize(f.recall(worst, map.online), st));
            recall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
        }
        b = book(lp);
        const bool back = map.online == 4 && f.recall(worst, map.online) == 0 && b.node() == (64ull << 20) + lendable -
                          (map.bytes_at(4) - map.bytes_at(1)) && !b.pend() && b.balanced() && zero_pages(chunked, cp, 4 * cp);
        char ms[32];
        snprintf(ms, sizeof ms, "%.1f", recall_ms);
        report(back, std::string("the recall to 4 chunks took ") + ms + " ms (node released at once); the request is admitted");
        for (int k = 0; k < NB; ++k)
        {
            CKI(g6_attn_bind_reset(&plain.b[k], st));
            CKI(g6_attn_bind_reset(&chunked.b[k], st));
        }
        same = same && decode(plain, chunked, ly, in, 0, (int)worst, st, why);
        report(same, std::string("idle lend -> request -> recall -> decode to ") + std::to_string(worst) +
                         " tokens: same bits as GR8_FLASH_KV_CHUNKS=0" + (same ? "" : " (differs: " + why + ")"));
        // a recall the node does not release in time
        CKI(map.resize(1, st));
        map.wait_ms = 200;
        const auto r1 = std::chrono::steady_clock::now();
        const int e6 = map.resize(f.recall(worst, map.online), st);
        const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r1).count();
        b = book(lp);
        snprintf(ms, sizeof ms, "%.0f", waited);
        report(e6 && map.online == 1 && map.backed() == map.bytes_at(1) && map.tokens() < worst &&
                   f.recall(worst, map.online) == 4 && !b.pend() && b.balanced(),
               std::string("a recall the node does not release in ") + ms + " ms: 1 chunk online, the request still waits (" +
                   map.err + ")");
        map.wait_ms = 10000;
        map.ledger.clear();
        CKI(map.resize(CHUNKS, st));
    }
    // 5. crashes: a child process ends at each point; the parent reads the book and recovers
    const char* points[] = {"begun", "synced", "released"};
    for (int d = 0; d < 2; ++d)
        for (const char* p : points)
        {
            if (d && !strcmp(p, "released")) continue;   // a grow's release is the node's: crash after it is "synced"
            const std::string cp_path = std::string(dir) + "/crash-" + p + std::to_string(d);
            // the child's one binding, every chunk online: whole granules (chunks of 256 pages at 2 MiB)
            const uint64_t full = (uint64_t)CHUNKS * 256 * (flashkv::MLA_PAGE + flashkv::IDX_PAGE);
            ledger_init(cp_path, full + (64ull << 20), full, 0);
            const pid_t pid = fork();
            if (pid == 0)
            {
                execl(argv[0], argv[0], "--child", p, cp_path.c_str(), d ? "1" : "0", (char*)nullptr);
                _exit(4);
            }
            int status = 0;
            waitpid(pid, &status, 0);
            Bk c = book(cp_path);
            const bool crashed = WIFEXITED(status) && WEXITSTATUS(status) == 3;
            const bool pending = c.pend() && c.phase() == (!strcmp(p, "begun") ? 1u : d ? 2u : !strcmp(p, "synced") ? 1u : 2u);
            const bool bal = c.balanced();
            // kvman-ledger recover aborts the transfer: what was in flight goes to free, credited to neither side
            const bool recovered = system((ledger_bin() + " " + cp_path + " recover --stale-s 0 > /dev/null").c_str()) == 0;
            Bk after = book(cp_path);
            const bool rec = recovered && !after.pend() && after.balanced() && after.gpu() == c.gpu() &&
                             after.node() == c.node();
            report(crashed && pending && bal && rec,
                   std::string(d ? "grow" : "shrink") + " crash at " + p + ": the book balances (gpu " +
                       std::to_string(c.gpu()) + ", node " + std::to_string(c.node()) + ", free " +
                       std::to_string(c.free()) + ") and recovers (free " + std::to_string(after.free()) + ")");
        }
    for (int k = 0; k < NB; ++k)
    {
        CKI(g6_attn_bind_release(&plain.b[k]));
        CKI(g6_attn_bind_release(&chunked.b[k]));
    }
    map.release();
    printf("flash_kv_chunks_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
