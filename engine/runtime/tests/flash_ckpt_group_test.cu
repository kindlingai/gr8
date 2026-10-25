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

    // 6. D-331 M8 E(c) defect A: a cycle of K prompts through a lease that holds fewer than K entries restores nothing
    // (each capture drops the least recently used, the one the cycle needs next: eviction, by design). Fixed two ways:
    // (a) with no cap= (cap 0, "the node's limit") the lease asks, the node grants and the cycle restores. Before the
    // node took cap 0 as no cap it took it as cap = floor and never granted. (b) a floor that holds K entries restores
    // with no Resize.
    {
        constexpr int K = 3;   // 4-page prompts; the 2-segment floor holds 2
        std::vector<std::vector<std::vector<uint8_t>>> want(K);
        // one request as serve runs it: a hit restores (bit for bit), a miss fills the state and keeps it
        const auto lap = [&](FlashCheckpoints& c) {
            int hits = 0;
            for (int i = 0; i < K; ++i) {
                const auto prompt = toks(4 * 64 + 100, 40 + i);
                const FlashHit h = c.index().match(prompt, int(prompt.size()) - 1, {});
                if (h.hit) {
                    d.poison();
                    c.restore(h.generation, 0, st);
                    CHECK(h.position == 4 * 64 && same(d, want[i], 4));
                    ++hits;
                } else {
                    d.fill(40 + i);
                    want[i] = d.snap();
                    CHECK(c.capture(4 * 64, 0, std::vector<int>(prompt.begin(), prompt.begin() + 4 * 64 + 1), {}, st));
                }
            }
            return hits;
        };
        const FlashGroup gc{"fckg-c-" + tag, 1, 0, 0, 1000};   // cap 0: no cap=
        FlashCheckpoints c(v, spec, "fckg-c0-" + tag, 2 * sg, "test compat", &gc);
        CHECK(c.segments() == 2 && 2 * blocks(4) <= c.blocks() && K * blocks(4) > c.blocks());
        int h0 = lap(c), h1 = lap(c);
        printf("cap 0 at the floor (%llu blocks, %d prompts of %llu blocks): laps hit %d, %d\n", (unsigned long long)c.blocks(), K,
               (unsigned long long)blocks(4), h0, h1);
        CHECK(h0 == 0 && h1 == 0 && c.index().size() == 2);
        const auto a6 = c.share_ask(false);
        CHECK(a6.first > 0);
        const auto g6 = c.share_report(FLASH_CKPT_TIER, a6.first, a6.second);
        CHECK(g6 && g6->segs > 2);   // fails when the node takes cap 0 as cap = floor
        if (g6) CHECK(c.apply(*g6));
        CHECK(c.blocks() >= K * blocks(4));
        h0 = lap(c); h1 = lap(c);
        printf("cap 0 after the grant (%llu segments, %llu blocks): laps hit %d, %d\n", (unsigned long long)c.segments(),
               (unsigned long long)c.blocks(), h0, h1);
        CHECK(h0 == K - 1 && h1 == K);
        // (b) the floor sized for the cycle: 3 segments, no Resize
        const FlashGroup gf{"fckg-f-" + tag, 1, 0, 3 * sg, 1000};
        FlashCheckpoints f(v, spec, "fckg-f0-" + tag, 3 * sg, "test compat", &gf);
        CHECK(f.segments() == 3 && K * blocks(4) <= f.blocks());
        h0 = lap(f); h1 = lap(f);
        printf("floor of 3 segments (%llu blocks): laps hit %d, %d\n", (unsigned long long)f.blocks(), h0, h1);
        CHECK(h0 == 0 && h1 == K && f.segments() == 3);
    }

    // 7. D-331 M8 Flash grow: the grant lands BEFORE the capture that would drop. The head asks once, one entry ahead
    // (share_want: less than an average entry left), while the engine is busy (no ages); the node has no free segment,
    // so it asks an idle group (old blocks) to give. That group gives; the Flash head's next report asks for nothing
    // new (its counters did not change), and the node grants the room it asked for anyway, so the next capture fits
    // without a drop. Before, the node granted only on a report that asked, so the given segments sat free until
    // the capture dropped an entry and the head asked again (box run 10-24: a give at 11:37:06, the drop at ~11:37:18,
    // the grow at 11:37:28).
    {
        constexpr int K = 3;   // 4-page prompts; the 2-segment floor holds 2
        const FlashGroup g7{"fckg-g-" + tag, 1, 0, 0, 1000};
        FlashCheckpoints c(v, spec, "fckg-g0-" + tag, 2 * sg, "test compat", &g7);
        CHECK(c.segments() == 2 && 2 * blocks(4) <= c.blocks() && 3 * blocks(4) > c.blocks());
        // an idle group holding every segment left, its blocks old
        g6kv_tier* idle = nullptr;
        require(g6kv_tier_open_group(spec.c_str(), ("fckg-i-" + tag).c_str(), ("fckg-i-" + tag).c_str(), 1, 0, sg, 0, 1000,
                                     v.page_bytes(), &idle) == 0, g6kv_tier_error());
        g6kv_report ri{};
        CHECK(g6kv_store_report(idle, 0, nullptr, 0, &ri) == 0);
        const uint64_t left = ri.unclaimed;
        uint64_t ib = 0;
        if (left) {
            CHECK(g6kv_store_report(idle, left, nullptr, 0, &ri) == 0 && ri.grant == left);
            CHECK(g6kv_store_grow(idle, 1 + left, ri.txn, &ib) == 0);
        }
        const std::vector<uint64_t> old(8, 600000);
        CHECK(g6kv_store_report(idle, 0, old.data(), old.size(), &ri) == 0 && ri.unclaimed == 0);
        std::vector<std::vector<std::vector<uint8_t>>> want(K);
        uint64_t gens[K] = {};
        const auto cap = [&](int i) {
            const auto prompt = toks(4 * 64 + 100, 70 + i);
            d.fill(70 + i);
            want[i] = d.snap();
            gens[i] = c.capture(4 * 64, 0, std::vector<int>(prompt.begin(), prompt.begin() + 4 * 64 + 1), {}, st);
            CHECK(gens[i]);
        };
        cap(0); cap(1);
        CHECK(c.index().dropped == 0 && c.index().size() == 2);
        // the head, busy: it asks one entry ahead, the node asks the idle group to give
        auto q = c.share_ask(false);
        CHECK(q.first > 0 && q.second.empty());
        CHECK(!c.share_report(FLASH_CKPT_TIER, q.first, q.second));
        CHECK(g6kv_store_report(idle, 0, old.data(), old.size(), &ri) == 0 && ri.give > 0 && ri.txn);
        CHECK(g6kv_store_shrink(idle, 1 + left - ri.give, ri.txn, &ib) == 0);
        // the head's next report asks for nothing new; the node grants what the give made for it
        q = c.share_ask(false);
        CHECK(q.first == 0);
        const auto gr = c.share_report(FLASH_CKPT_TIER, q.first, q.second);
        printf("ask once, idle group gave %llu: next report %s\n", (unsigned long long)ri.give,
               gr ? ("granted to " + std::to_string(gr->segs) + " segments").c_str() : "got nothing (the grant waits for a drop)");
        CHECK(gr && gr->segs > 2);   // fails when the node grants only on a report that asks
        if (gr) CHECK(c.apply(*gr));
        // the capture that would have dropped the least recently used entry fits
        cap(2);
        printf("third capture: %zu entries, %llu dropped\n", c.index().size(), (unsigned long long)c.index().dropped);
        CHECK(c.index().dropped == 0 && c.index().size() == K);
        for (int i = 0; i < K; ++i) {
            d.poison();
            if (c.index().find(gens[i])) { c.restore(gens[i], 0, st); CHECK(same(d, want[i], 4)); }
        }
        g6kv_tier_free(idle);
    }

    // 8. the same when a give left the store full (box run 10-24, p3 -> p4: Flash gave back to 2 segments, 5 entries
    // and 95 blocks free, and p4's first capture dropped an entry before the head asked). No capture since the last
    // report, less than an average entry free: an idle head asks nothing (an idle group never takes room), a busy one
    // (a request resident) asks one entry ahead, the idle group gives, and the next capture fits with no drop.
    {
        constexpr int K = 3;
        const FlashGroup g8{"fckg-h-" + tag, 1, 0, 0, 1000};
        FlashCheckpoints c(v, spec, "fckg-h0-" + tag, 2 * sg, "test compat", &g8);
        g6kv_tier* idle = nullptr;
        require(g6kv_tier_open_group(spec.c_str(), ("fckg-j-" + tag).c_str(), ("fckg-j-" + tag).c_str(), 1, 0, sg, 0, 1000,
                                     v.page_bytes(), &idle) == 0, g6kv_tier_error());
        // it takes every segment left (section 7's leases close as they go)
        g6kv_report ri{};
        uint64_t isegs = 1, ib = 0;
        const std::vector<uint64_t> old(8, 600000);
        for (int i = 0; i < 40; ++i) {
            CHECK(g6kv_store_report(idle, 0, old.data(), old.size(), &ri) == 0);
            if (!ri.unclaimed) break;
            CHECK(g6kv_store_report(idle, ri.unclaimed, old.data(), old.size(), &ri) == 0);
            if (ri.grant) { CHECK(g6kv_store_grow(idle, isegs + ri.grant, ri.txn, &ib) == 0); isegs += ri.grant; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        CHECK(ri.unclaimed == 0 && isegs > 1);
        uint64_t gens[K] = {};
        std::vector<std::vector<std::vector<uint8_t>>> want(K);
        const auto cap = [&](int i) {
            const auto prompt = toks(4 * 64 + 100, 80 + i);
            d.fill(80 + i);
            want[i] = d.snap();
            gens[i] = c.capture(4 * 64, 0, std::vector<int>(prompt.begin(), prompt.begin() + 4 * 64 + 1), {}, st);
            CHECK(gens[i]);
        };
        cap(0); cap(1);
        c.share_ask(false);   // a report the node could not serve: the store is as a give leaves it, full, no capture since
        CHECK(c.share_ask(true, false).first == 0);   // idle: nothing
        auto q = c.share_ask(false, true);            // busy: one entry ahead
        printf("store left full, no capture since: a busy head asks %llu\n", (unsigned long long)q.first);
        CHECK(q.first > 0);   // fails when only a capture since the last report makes the head ask
        CHECK(!c.share_report(FLASH_CKPT_TIER, q.first, q.second));
        CHECK(g6kv_store_report(idle, 0, old.data(), old.size(), &ri) == 0 && ri.give > 0 && ri.txn);
        CHECK(g6kv_store_shrink(idle, isegs - ri.give, ri.txn, &ib) == 0);
        q = c.share_ask(false, true);
        const auto gr = c.share_report(FLASH_CKPT_TIER, q.first, q.second);
        CHECK(gr && gr->segs > 2);
        if (gr) CHECK(c.apply(*gr));
        cap(2);
        printf("capture after the grant: %zu entries, %llu dropped\n", c.index().size(), (unsigned long long)c.index().dropped);
        CHECK(c.index().dropped == 0 && c.index().size() == K);
        for (int i = 0; i < K; ++i) {
            d.poison();
            if (c.index().find(gens[i])) { c.restore(gens[i], 0, st); CHECK(same(d, want[i], 4)); }
        }
        g6kv_tier_free(idle);
    }

    // 9. (c) the capture at the boundary after the quantum (flash_native.cuh h.before_work): a grow the head announced
    // during the prefill commits first (box3 p4 k0). Two stores alike, full at 2 entries, the same device state: one
    // captures as before (inside the quantum, before the grow: an entry dropped), the other grows, then captures (none
    // dropped). The grow's apply leaves the device state alone, and the two entries restore the same bytes.
    {
        const FlashGroup go{"fckg-o-" + tag, 1, 0, 0, 1000}, gn{"fckg-n-" + tag, 1, 0, 0, 1000};
        FlashCheckpoints o(v, spec, "fckg-o0-" + tag, 2 * sg, "test compat", &go), n(v, spec, "fckg-n0-" + tag, 2 * sg, "test compat", &gn);
        const auto prompt = [&](int i) { const auto p = toks(4 * 64 + 100, 90 + i); return std::vector<int>(p.begin(), p.begin() + 4 * 64 + 1); };
        for (int i = 0; i < 2; ++i) {
            d.fill(90 + i);
            CHECK(o.capture(4 * 64, 0, prompt(i), {}, st) && n.capture(4 * 64, 0, prompt(i), {}, st));
        }
        CHECK(o.index().size() == 2 && n.index().size() == 2 && 3 * blocks(4) > o.blocks());
        d.fill(92);
        const auto x = d.snap();
        const uint64_t eo = o.capture(4 * 64, 0, prompt(2), {}, st);   // before the grow
        const auto q = n.share_ask(false);
        const auto gr = n.share_report(FLASH_CKPT_TIER, q.first, q.second);
        CHECK(q.first > 0 && gr && gr->segs > 2);
        if (gr) CHECK(n.apply(*gr));
        CHECK(same(d, x, PAGES));                                      // the commit did not touch the state
        const uint64_t en = n.capture(4 * 64, 0, prompt(2), {}, st);   // after it
        printf("capture in the quantum: %zu entries, %llu dropped; after the grow: %zu entries, %llu dropped\n",
               o.index().size(), (unsigned long long)o.index().dropped, n.index().size(), (unsigned long long)n.index().dropped);
        CHECK(eo && o.index().size() == 2 && o.index().dropped == 1);
        CHECK(en && n.index().size() == 3 && n.index().dropped == 0);
        d.poison(); o.restore(eo, 0, st);
        const auto ro = d.snap();
        CHECK(same(d, x, 4));
        d.poison(); n.restore(en, 0, st);
        CHECK(same(d, x, 4) && same(d, ro, 4));                        // byte-identical restores
    }

    g6kv_tier_free(b);
    a[0].reset(); a[1].reset();
    CK(cudaStreamDestroy(st));
    for (auto& [p, n] : d.all()) cudaFree(p);
    printf("flash_ckpt_group: %s (ask before a drop, grant, grow, oversize fits, give with ages, LRU drop + pack, restores "
           "bit for bit after moves, floor, refused transaction, cap 0 grows a cycling lease, a floor sized for the cycle, a grant before the drop, a busy store a give left full asks, a capture after the grow restores the same bytes)\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
