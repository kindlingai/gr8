// G6_FLASH_CACHE's device half (s2/flash_checkpoint.cuh) on one GPU, Flash's TP=2 geometry: 12 bindings of 64 x 1024 B
// latent rows and 40,960 B of indexer keys a page, and the KDA state of 34 layers x 2 slots x 32 heads.
//   usage: test_flash_ckpt STORE...   (mem:, ssd:FILE, remote:HOST:PORT; the corruption cases need ssd:)
// For each store: a checkpoint survives poisoning every device byte (pages and KDA state come back bit for bit, the rows
// and slots it does not own stay as they were), A -> B -> overwrite -> A at short and long context, more pages than one
// copy batch, a slot other than 0, a capture off the page grid refused unless terminal (a
// partial last page), least-recently-used drops, and (ssd:) a torn
// manifest, a changed page byte and a changed state byte refused.
#include "../runtime/s2/flash_checkpoint.cuh"
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
using namespace g6s2;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
constexpr int NB = 12, PAGES = 100, LAYERS = 34, SLOTS = 2, NH = 32, DK = 128;
constexpr size_t MLA = 64 * 1024, IDX = 40960, S_SLOT = (size_t)NH * DK * DK, C_SLOT = (size_t)3 * 3 * NH * DK;

__global__ void k_fill(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = uint32_t(i) * 2654435761u ^ seed * 0x9E3779B9u;
        x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12;
        p[i] = x;
    }
}
struct Dev {
    std::vector<void*> mla, idx;
    float *S = nullptr, *conv = nullptr;
    std::vector<std::pair<void*, size_t>> all() const {
        std::vector<std::pair<void*, size_t>> v;
        for (int b = 0; b < NB; ++b) { v.push_back({mla[b], PAGES * MLA}); v.push_back({idx[b], PAGES * IDX}); }
        v.push_back({S, LAYERS * SLOTS * S_SLOT * 4}); v.push_back({conv, LAYERS * SLOTS * C_SLOT * 4});
        return v;
    }
    void fill(uint32_t seed) {
        uint32_t k = 0;
        for (auto& [p, n] : all()) k_fill<<<1024, 256>>>((uint32_t*)p, n / 4, seed * 1000 + ++k);
        CK(cudaDeviceSynchronize());
    }
    void poison() { for (auto& [p, n] : all()) CK(cudaMemset(p, 0xA5, n)); CK(cudaDeviceSynchronize()); }
    std::vector<std::vector<uint8_t>> snap() const {
        std::vector<std::vector<uint8_t>> v;
        for (auto& [p, n] : all()) { v.emplace_back(n); CK(cudaMemcpy(v.back().data(), p, n, cudaMemcpyDeviceToHost)); }
        return v;
    }
};
// pages [0, np) of every binding and KDA slot `to` equal `want` (slot `from` of it); every other byte equals `rest`
static bool check(const Dev& d, const std::vector<std::vector<uint8_t>>& want, const std::vector<std::vector<uint8_t>>& rest,
                  uint32_t np, int from, int to) {
    const auto now = d.snap();
    for (int b = 0; b < 2 * NB; ++b) {
        const size_t page = b % 2 ? IDX : MLA, own = np * page;
        if (memcmp(now[b].data(), want[b].data(), own)) { fprintf(stderr, "binding piece %d: kept pages differ\n", b); return false; }
        if (memcmp(now[b].data() + own, rest[b].data() + own, PAGES * page - own)) { fprintf(stderr, "binding piece %d: pages past the entry changed\n", b); return false; }
    }
    for (int k = 0; k < 2; ++k) {   // S, conv
        const size_t slot = (k ? C_SLOT : S_SLOT) * 4;
        const auto &n = now[2 * NB + k], &w = want[2 * NB + k], &r = rest[2 * NB + k];
        for (int l = 0; l < LAYERS; ++l)
            for (int s = 0; s < SLOTS; ++s) {
                const uint8_t* got = n.data() + (size_t)(l * SLOTS + s) * slot;
                const uint8_t* exp = s == to ? w.data() + (size_t)(l * SLOTS + from) * slot : r.data() + (size_t)(l * SLOTS + s) * slot;
                if (memcmp(got, exp, slot)) { fprintf(stderr, "%s layer %d slot %d differs\n", k ? "conv" : "S", l, s); return false; }
            }
    }
    return true;
}
static std::vector<int> toks(uint32_t n, int salt) { std::vector<int> v(n); for (uint32_t i = 0; i < n; ++i) v[i] = int((i * 31 + salt) % 50000); return v; }
template <class F> static std::string throws(F f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "";
}
static void flip(const std::string& file, uint64_t off) {
    const int fd = open(file.c_str(), O_RDWR);
    assert(fd >= 0);
    uint8_t b = 0;
    assert(pread(fd, &b, 1, off) == 1);
    b ^= 0x40;
    assert(pwrite(fd, &b, 1, off) == 1);
    fsync(fd); close(fd);
}
static void zero_block(const std::string& file, uint64_t off, size_t n) {
    const int fd = open(file.c_str(), O_RDWR);
    std::vector<uint8_t> z(n, 0);
    assert(fd >= 0 && pwrite(fd, z.data(), n, off) == (ssize_t)n);
    fsync(fd); close(fd);
}

static void run(const std::string& spec, int n_store) {
    Dev d;
    for (int b = 0; b < NB; ++b) { d.mla.push_back(nullptr); d.idx.push_back(nullptr); CK(cudaMalloc(&d.mla[b], PAGES * MLA)); CK(cudaMalloc(&d.idx[b], PAGES * IDX)); }
    CK(cudaMalloc(&d.S, LAYERS * SLOTS * S_SLOT * 4)); CK(cudaMalloc(&d.conv, LAYERS * SLOTS * C_SLOT * 4));
    FlashStateView v;
    for (int b = 0; b < NB; ++b) v.bindings.push_back({(uint4*)d.mla[b], (uint4*)d.idx[b], MLA / 16, IDX / 16});
    v.S = d.S; v.conv = d.conv; v.layers = LAYERS; v.slots = SLOTS; v.s_slot = S_SLOT; v.c_slot = C_SLOT;
    assert(v.page_bytes() == 1277952 && v.state_bytes() == 76316672);   // DESIGN M5's 1,277,952 B block, 76.3 MB a slot at TP=2
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const std::string name = "flash-ckpt-test-" + std::to_string(getpid()) + "-" + std::to_string(n_store);
    // 300 blocks: room for the 40-page and 1-page entries (101 + 62 with 60 state blocks), not for those and a 100-page one
    FlashCheckpoints ck(v, spec, name, 300ull * 1277952, "test compat");
    const uint64_t sb = (76316672 + ck.slot_bytes() - 1) / ck.slot_bytes();
    printf("%s: %llu blocks of %zu B, state %llu blocks\n", ck.describe().c_str(), (unsigned long long)ck.blocks(), ck.slot_bytes(), (unsigned long long)sb);

    // 1. poison and restore: 40 pages (> the 32-page copy batch), slot 0
    d.fill(1);
    const auto A = d.snap();
    const auto tA = toks(40 * 64 + 1, 1);
    const uint64_t gA = ck.capture(40 * 64, 0, tA, {}, st);
    assert(gA == 1);
    printf("  capture 40 pages: %llu blocks %.1f ms\n", (unsigned long long)ck.last.blocks, ck.last.ms);
    d.poison();
    const auto P = d.snap();
    ck.restore(gA, 0, st);
    printf("  restore 40 pages: %.1f ms\n", ck.last.ms);
    assert(check(d, A, P, 40, 0, 0));
    // 2. A -> B (short, 1 page; from slot 1 into an entry) -> overwrite -> A, then B into slot 0
    d.fill(2);
    const auto B = d.snap();
    const auto tB = toks(64 + 1, 2);
    const uint64_t gB = ck.capture(64, 1, tB, {}, st);
    assert(gB == 2);
    d.fill(3);
    const auto C = d.snap();
    ck.restore(gA, 0, st);
    assert(check(d, A, C, 40, 0, 0));
    d.fill(3);
    ck.restore(gB, 0, st);
    assert(check(d, B, C, 1, 1, 0));
    // the same tokens again: the kept entry, nothing written
    assert(ck.capture(40 * 64, 0, tA, {}, st) == gA && ck.index().size() == 2);
    // 3. off the page grid, or the wrong token count: refused
    assert(!throws([&] { ck.capture(65, 0, toks(66, 4), {}, st); }).empty());
    assert(!throws([&] { ck.capture(128, 0, toks(128, 4), {}, st); }).empty());
    // 4. long context: 100 pages (every page of the binding). A, B and it do not fit 300 blocks: B, the least recently
    // used (A was restored and captured again after it), is dropped, and A stays
    d.fill(5);
    const auto L = d.snap();
    const auto tL = toks(100 * 64 + 1, 5);
    const uint64_t gL = ck.capture(100 * 64, 0, tL, {}, st);
    printf("  capture 100 pages: %llu blocks %.1f ms (%.2f GB/s)\n", (unsigned long long)ck.last.blocks, ck.last.ms,
           ck.last.blocks * ck.slot_bytes() / ck.last.ms / 1e6);
    assert(gL == 3);
    d.poison();
    const auto P2 = d.snap();
    ck.restore(gL, 0, st);
    printf("  restore 100 pages: %.1f ms (%.2f GB/s)\n", ck.last.ms, ck.last.blocks * ck.slot_bytes() / ck.last.ms / 1e6);
    assert(check(d, L, P2, 100, 0, 0));
    assert((1 + 40 + sb) + (1 + 1 + sb) + (1 + 100 + sb) > ck.blocks());
    assert(ck.index().find(gA) && !ck.index().find(gB) && ck.index().dropped == 1);
    assert(throws([&] { ck.restore(gB, 0, st); }).find("no entry") != std::string::npos);
    d.poison();
    const auto P3 = d.snap();
    ck.restore(gA, 0, st);   // the survivor still restores after the store moved around it
    assert(check(d, A, P3, 40, 0, 0));
    // 4b. terminal: a position mid-page (10 pages + 37 rows, the whole 11th page kept) from ring slot 1 into slot 0. The
    // store drops L (least recently used: A was restored after it); A stays
    d.fill(7);
    const auto T = d.snap();
    const auto tT = toks(10 * 64 + 37 + 1, 7);
    assert(!throws([&] { ck.capture(10 * 64 + 37, 1, tT, {}, st); }).empty());   // not terminal: refused
    const uint64_t gT = ck.capture(10 * 64 + 37, 1, tT, {}, st, true);
    assert(gT && ck.index().find(gT)->terminal && ck.index().find(gT)->blocks == 1 + 11 + sb);
    assert(ck.index().find(gA) && !ck.index().find(gL));
    d.poison();
    const auto P4 = d.snap();
    ck.restore(gT, 0, st);
    printf("  terminal 10 pages + 37 rows: %llu blocks, restore %.1f ms\n", (unsigned long long)ck.last.blocks, ck.last.ms);
    assert(check(d, T, P4, 11, 1, 0));
    // 5. the bytes in the store must be the entry's (ssd: the file is ours to change)
    if (spec.rfind("ssd:", 0) == 0) {
        const std::string file = spec.substr(4);
        const size_t sl = ck.slot_bytes();
        d.fill(6);
        const uint64_t g6 = ck.capture(2 * 64, 0, toks(2 * 64 + 1, 6), {}, st);
        const FlashEntry* x = ck.index().find(g6);
        assert(x);
        flip(file, (x->base + 1) * sl + 4096 + 3);                          // a byte of page 0
        std::string why = throws([&] { ck.restore(g6, 0, st); });
        printf("  page byte changed: %s\n", why.c_str());
        assert(why.find("page checksum") != std::string::npos);
        flip(file, (x->base + 1) * sl + 4096 + 3);                          // back: the entry restores again
        ck.restore(g6, 0, st);
        flip(file, (x->base + 3) * sl + 12345);                             // a byte of the state (after 2 pages)
        why = throws([&] { ck.restore(g6, 0, st); });
        printf("  state byte changed: %s\n", why.c_str());
        assert(why.find("KDA state checksum") != std::string::npos);
        flip(file, (x->base + 3) * sl + 12345);
        zero_block(file, x->base * sl, sl);                                 // torn: the data without its manifest
        why = throws([&] { ck.restore(g6, 0, st); });
        printf("  manifest missing: %s\n", why.c_str());
        assert(why.find("no manifest") != std::string::npos);
    }
    CK(cudaStreamDestroy(st));
    for (auto& [p, n] : d.all()) cudaFree(p);
    printf("  %s: PASS\n", spec.c_str());
}
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_flash_ckpt STORE...\n"); return 2; }
    for (int i = 1; i < argc; ++i) run(argv[i], i);
    puts("flash_ckpt: PASS (poison and restore bit for bit, A->B->overwrite->A, batches, slots, grid refusal, terminal partial page, drops, corruption refused)");
}
