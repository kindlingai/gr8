// G6_FLASH_CACHE on a group lease (--kv-store ...,group=; s2/flash_checkpoint.cuh share_ask / share_report / apply, D-331
// M8 (c)) against a loopback KV node with 2 MiB segments, one GPU, a small geometry (2 bindings, 16 pages, 2 KDA layers).
//   usage: flash_ckpt_group_test remote:HOST:PORT
//   (node: kvman-remote --gb 0.05 --seg-mb 2 --margin-s 0 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --no-rdma)
// Cluster A is two ranks in lock-step (one device state, two leases); cluster B one raw group lease. A fills its floor
// and drops an entry: its head asks for room, the node grants, both ranks grow at the same Resize and an entry too big
// for the floor now fits. A grows to its cap; B asks, the node asks A to give a segment, A's head offers its ages, both
// ranks shrink: the least recently used entry goes, the entries above the new end move down, and every survivor
// restores bit for bit from where it moved. Then back to the floor with no transaction. The ranks' digests agree after
// every step.
#include "../s2/flash_checkpoint.cuh"
#include <cstdio>
#include <memory>
#include <thread>
#include <unistd.h>
using namespace g6s2;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
static long fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 20) printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
constexpr int NB = 2, PAGES = 16, LAYERS = 2, SLOTS = 2, NH = 2, DK = 128;
constexpr size_t MLA = 64 * 1024, IDX = 40960, S_SLOT = (size_t)NH * DK * DK, C_SLOT = (size_t)3 * 3 * NH * DK;
constexpr uint64_t SEG = 2 << 20;

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
        for (auto& [p, n] : all()) k_fill<<<256, 256>>>((uint32_t*)p, n / 4, seed * 1000 + ++k);
        CK(cudaDeviceSynchronize());
    }
    void poison() { for (auto& [p, n] : all()) CK(cudaMemset(p, 0xA5, n)); CK(cudaDeviceSynchronize()); }
    std::vector<std::vector<uint8_t>> snap() const {
        std::vector<std::vector<uint8_t>> v;
        for (auto& [p, n] : all()) { v.emplace_back(n); CK(cudaMemcpy(v.back().data(), p, n, cudaMemcpyDeviceToHost)); }
        return v;
    }
};
// pages [0, np) of every binding and KDA slot 0 equal `want`'s (captured from slot 0)
static bool same(const Dev& d, const std::vector<std::vector<uint8_t>>& want, uint32_t np) {
    const auto now = d.snap();
    for (int b = 0; b < 2 * NB; ++b)
        if (memcmp(now[b].data(), want[b].data(), np * (b % 2 ? IDX : MLA))) return false;
    for (int k = 0; k < 2; ++k) {
        const size_t slot = (k ? C_SLOT : S_SLOT) * 4;
        for (int l = 0; l < LAYERS; ++l)
            if (memcmp(now[2 * NB + k].data() + (size_t)l * SLOTS * slot, want[2 * NB + k].data() + (size_t)l * SLOTS * slot, slot))
                return false;
    }
    return true;
}
static std::vector<int> toks(uint32_t n, int salt) { std::vector<int> v(n); for (uint32_t i = 0; i < n; ++i) v[i] = int((i * 31 + salt) % 50000); return v; }

struct Kept { uint64_t gen; uint32_t pages; std::vector<std::vector<uint8_t>> bytes; };

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: flash_ckpt_group_test remote:HOST:PORT\n"); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string spec = argv[1];
    Dev d;
    for (int b = 0; b < NB; ++b) { d.mla.push_back(nullptr); d.idx.push_back(nullptr); CK(cudaMalloc(&d.mla[b], PAGES * MLA)); CK(cudaMalloc(&d.idx[b], PAGES * IDX)); }
    CK(cudaMalloc(&d.S, LAYERS * SLOTS * S_SLOT * 4)); CK(cudaMalloc(&d.conv, LAYERS * SLOTS * C_SLOT * 4));
    FlashStateView v;
    for (int b = 0; b < NB; ++b) v.bindings.push_back({(uint4*)d.mla[b], (uint4*)d.idx[b], MLA / 16, IDX / 16});
    v.S = d.S; v.conv = d.conv; v.layers = LAYERS; v.slots = SLOTS; v.s_slot = S_SLOT; v.c_slot = C_SLOT;
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const std::string tag = std::to_string(getpid());
    const uint64_t sg = SEG / v.page_bytes() * v.page_bytes();   // a segment's bytes: the node counts whole blocks a segment

    g6kv_tier* b = nullptr;
    require(g6kv_tier_open_group(spec.c_str(), ("fckg-b-" + tag).c_str(), ("fckg-b-" + tag).c_str(), 1, 0, sg, 4 * sg, 1000,
                                 v.page_bytes(), &b) == 0, g6kv_tier_error());
    std::unique_ptr<FlashCheckpoints> a[2];
    for (uint32_t k = 0; k < 2; ++k) {
        const FlashGroup g{"fckg-a-" + tag, 2, k, 12 * sg, 1000};
        a[k].reset(new FlashCheckpoints(v, spec, "fckg-a" + std::to_string(k) + "-" + tag, 2 * sg, "test compat", &g));
    }
    const uint64_t per = g6kv_store_seg_blocks(b), sb = (v.state_bytes() + a[0]->slot_bytes() - 1) / a[0]->slot_bytes();
    const auto blocks = [&](uint32_t pages) { return 1 + pages + sb; };
    printf("%s: %llu blocks a segment, %llu segments, %llu blocks of %zu B, state %llu blocks\n", a[0]->describe().c_str(),
           (unsigned long long)per, (unsigned long long)a[0]->segments(), (unsigned long long)a[0]->blocks(), a[0]->slot_bytes(),
           (unsigned long long)sb);
    CHECK(a[0]->group() && a[1]->group() && a[0]->segments() == 2 && a[0]->blocks() == 2 * per);
    std::vector<Kept> kept;
    const auto capture = [&](uint32_t pages, uint32_t seed) -> uint64_t {
        d.fill(seed);
        const auto t = toks(pages * 64 + 1, int(seed));
        const uint64_t g0 = a[0]->capture(pages * 64, 0, t, {}, st), g1 = a[1]->capture(pages * 64, 0, t, {}, st);
        CHECK(g0 == g1);
        if (g0) kept.push_back({g0, pages, d.snap()});
        return g0;
    };
    const auto agree = [&] { return a[0]->index().digest() == a[1]->index().digest(); };
    const auto apply = [&](const Resize& r, std::vector<uint64_t>* gone = nullptr) {
        std::vector<uint64_t> g[2];
        const bool ok0 = a[0]->apply(r, &g[0]), ok1 = a[1]->apply(r, &g[1]);
        CHECK(ok0 == ok1 && g[0] == g[1] && agree());
        if (gone) *gone = g[0];
        return ok0;
    };
    // every entry the index still holds restores bit for bit on both ranks
    const auto restores = [&] {
        long n = 0;
        for (const auto& x : kept)
            for (int k = 0; k < 2; ++k)
                if (a[k]->index().find(x.gen)) {
                    d.poison();
                    a[k]->restore(x.gen, 0, st);
                    CHECK(same(d, x.bytes, x.pages));
                    ++n;
                }
        CHECK(agree());
        return n;
    };

    // 1. the floor (2 segments) holds two 4-page entries; a third drops the least recently used. The head's report
    // from those counters asks for the dropped blocks' segments; an entry bigger than the whole lease is counted
    const uint64_t g1 = capture(4, 1), g2 = capture(4, 2);
    CHECK(g1 && g2 && 2 * blocks(4) <= a[0]->blocks() && 3 * blocks(4) > a[0]->blocks());
    auto ask = a[0]->share_ask(false);
    CHECK(ask.first > 0 && ask.second.empty());   // two captures and too little left for a third: ask before a drop
    CHECK(capture(4, 3) && !a[0]->index().find(g1) && a[0]->index().totals().full == 1);
    CHECK(a[0]->index().totals().dropped_pages == blocks(4));
    CHECK(!capture(16, 4) && a[0]->index().oversize == 1 && a[0]->index().oversize_last == blocks(16));
    ask = a[0]->share_ask(false);
    CHECK(ask.first == (blocks(16) - a[0]->index().totals().free_slots + per - 1) / per);
    a[1]->share_ask(false);   // the ranks' report state stays alike (only the head's is used)
    // 2. the node grants it (nobody holds the rest): both ranks grow at the same Resize, and the 16-page entry fits
    const auto grow = a[0]->share_report(FLASH_CKPT_TIER, ask.first, ask.second);
    CHECK(grow && grow->tier == FLASH_CKPT_TIER && grow->segs == 2 + ask.first && grow->slots == grow->segs * per);
    if (grow) CHECK(apply(*grow));
    CHECK(a[0]->blocks() == a[0]->segments() * per && a[1]->segments() == a[0]->segments());
    CHECK(capture(16, 5));
    CHECK(restores() == 2 * 3);
    // 3. to the cap (12 segments a rank, the node's 25 then all held), filled with five 16-page entries and a 4-page one
    const auto cap = a[0]->share_report(FLASH_CKPT_TIER, 12, {});
    CHECK(cap && cap->segs == 12);
    if (cap) CHECK(apply(*cap));
    CHECK(a[0]->blocks() == 12 * per);
    for (const auto& x : kept) for (int k = 0; k < 2; ++k) if (a[k]->index().find(x.gen)) a[k]->index().drop(x.gen);
    kept.clear();
    uint64_t e[6];
    for (int i = 0; i < 5; ++i) e[i] = capture(16, 10 + i);
    e[5] = capture(4, 20);
    const uint64_t used = 5 * blocks(16) + blocks(4);
    CHECK(a[0]->index().used_blocks() == used && used > 11 * per && used - blocks(16) <= 11 * per);
    CHECK(a[0]->index().find(e[5])->base + blocks(4) > 11 * per);   // the last one ends past 11 segments
    // e[0] used again: e[1] is now the least recently used
    for (int k = 0; k < 2; ++k) a[k]->restore(e[0], 0, st);
    // 4. A's head offers its ages (every entry at least 20 ms old; less than a segment free, so no UINT64_MAX). B asks
    // for 2 segments; none is free, so the node asks A to give one a rank at A's next report
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ask = a[0]->share_ask(true);
    a[1]->share_ask(true);
    CHECK(ask.first > 0);   // 6 blocks free, less than an average entry: it asks, but A is at its cap
    CHECK(!ask.second.empty() && ask.second[0] != UINT64_MAX && ask.second[0] >= 20);
    for (size_t i = 1; i < ask.second.size(); ++i) CHECK(ask.second[i] <= ask.second[i - 1]);
    CHECK(!a[0]->share_report(FLASH_CKPT_TIER, ask.first, ask.second));
    g6kv_report rb{};
    CHECK(g6kv_store_report(b, 2, nullptr, 0, &rb) == 0 && rb.grant == 0);
    ask = a[0]->share_ask(true);
    a[1]->share_ask(true);
    const auto give = a[0]->share_report(FLASH_CKPT_TIER, ask.first, ask.second);
    CHECK(give && give->segs == 11 && give->slots == 11 * per);
    std::vector<uint64_t> gone;
    if (give) CHECK(apply(*give, &gone));
    printf("give: dropped %zu entr%s, %zu kept, %llu blocks\n", gone.size(), gone.size() == 1 ? "y" : "ies", a[0]->index().size(),
           (unsigned long long)a[0]->blocks());
    CHECK(gone == std::vector<uint64_t>{e[1]} && a[0]->blocks() == 11 * per && a[0]->segments() == 11);
    for (const auto& x : a[0]->index().entries()) CHECK(x.base + x.blocks <= 11 * per);
    CHECK(a[0]->index().find(e[5])->base == 4 * blocks(16));   // packed: below the four 16-page entries left
    CHECK(restores() == 2 * 5);
    // the segment A gave is B's now
    uint64_t bb = 0;
    CHECK(g6kv_store_report(b, 2, nullptr, 0, &rb) == 0 && rb.grant == 2);
    CHECK(g6kv_store_grow(b, 1 + rb.grant, rb.txn, &bb) == 0 && bb == 3 * per);
    // 5. back to the floor on the group's own (transaction 0): least recently used first until the rest fit 2 segments
    CHECK(apply(Resize{FLASH_CKPT_TIER, 2, 2 * per, 0}, &gone));
    CHECK(a[0]->segments() == 2 && a[0]->blocks() == 2 * per && a[0]->index().used_blocks() <= 2 * per);
    CHECK(a[0]->index().size() == 1 && a[0]->index().find(e[5]));   // the 4-page entry, the most recently captured
    CHECK(restores() == 2);
    // a capture still works at the floor, and a stale grow (a transaction the node does not hold) is refused alike
    CHECK(capture(4, 30));
    CHECK(!apply(Resize{FLASH_CKPT_TIER, 3, 3 * per, 999999}));
    CHECK(a[0]->segments() == 2 && restores() == 2 * 2);

    g6kv_tier_free(b);
    a[0].reset(); a[1].reset();
    CK(cudaStreamDestroy(st));
    for (auto& [p, n] : d.all()) cudaFree(p);
    printf("flash_ckpt_group: %s (ask before a drop, grant, grow, oversize fits, give with ages, LRU drop + pack, restores "
           "bit for bit after moves, floor, refused transaction)\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
