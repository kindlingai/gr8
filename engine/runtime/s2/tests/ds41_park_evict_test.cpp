// GR8_DS_PARK eviction vs the grant's own snapshot (ds41_native.cuh bind, Ds41PrefixCache::capture keep). A new prompt that
// matches a prompt-end snapshot PE (claimed by its grant) but not the parked chat's request-end tokens RE, all slots full, so
// the bind evicts that parked slot: the capture of RE must not evict PE, which the bind restores next. Before the fix the
// capture only moved PE to the back of the LRU (touch), so a budget that fits PE or RE but not both evicted PE and the
// bind aborted ("the evicted chat's capture evicted the grant's own snapshot"). Astra's fixture (review of 70b7282):
// 1 MiB budget, one attention layer of 256 ring rows; BF16 PE 774144 + RE 876544 B, FP4 PE 521344 + RE 550144 B.
// Host only: the CUDA calls are stubs over host memory with an optional allocation limit.
// Build (one line): g++ -std=c++17 -O1 -g -fsanitize=address,undefined -DG6_MODEL_DS41=1 -I engine/runtime engine/runtime/s2/tests/ds41_park_evict_test.cpp -o /tmp/ds41_park_evict_test <knobs lib>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "prefix_reuse.h"
#include "../engram/engram_ring.h"
#include "../knobs-rs/gr8_knobs.h"

using __nv_bfloat16 = uint16_t;
using cudaStream_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyDeviceToDevice = 0, cudaMemcpyHostToDevice = 1;
static std::unordered_map<void*, size_t> mem;
static size_t live_bytes = 0, alloc_limit = SIZE_MAX;   // alloc_limit: device memory pressure (cudaMalloc fails above it)
static int cudaMalloc(void** p, size_t n) {
    if (n > alloc_limit - live_bytes) return 2;
    *p = std::malloc(n); mem[*p] = n; live_bytes += n; return cudaSuccess;
}
static int cudaFree(void* p) { if (!p) return 0; auto i = mem.find(p); assert(i != mem.end()); live_bytes -= i->second; mem.erase(i); std::free(p); return 0; }
static int cudaGetLastError() { return 0; }
static int cudaStreamSynchronize(int) { return 0; }
static int cudaMemcpyAsync(void* d, const void* s, size_t n, int, int) { std::memcpy(d, s, n); return 0; }
#define G6RT_CK(x) assert((x) == 0)
namespace ds41a { constexpr int HD = 512, CQ_ROW = 288; }
namespace g6rt {
constexpr size_t DS41_CA_ROW = 288, DS41_ICA_ROW = 68;
constexpr int ENG_RING = 64;
struct Cache { uint16_t *ring = nullptr, *comp = nullptr; uint8_t* compq = nullptr; float *kvs = nullptr, *scs = nullptr; int ring_rows = 256; size_t comp_ss = 0; };
struct Ds41AttnLayer { Cache C; };
struct Ds41IdxModel { uint8_t *ca[4]{}, *ica[4]{}; uint16_t* shadow = nullptr; int rows_kv[4]{}, cap = 2048, rcap = 8; };
struct Model { std::vector<Ds41AttnLayer> layers; Ds41IdxModel* idx = nullptr; int* ehist = nullptr; int rank = 0; };
static Ds41AttnLayer* ds41_attn_state(Model& m, int i) { return &m.layers[size_t(i)]; }
[[maybe_unused]] static const Ds41AttnLayer* ds41_attn_state(const Model& m, int i) { return &m.layers[size_t(i)]; }
namespace ds41_idx_detail {
static Ds41IdxModel* find(Model& m, bool) { return m.idx; }
[[maybe_unused]] static const Ds41IdxModel* find(const Model& m, bool) { return m.idx; }
}
}
namespace g6s2 { struct Ds41Dspark { g6rt::Model draft; }; }
#define GR8_DS41_PREFIX_CACHE_STANDALONE 1
#include "s2/ds41_prefix_cache.cuh"
#include "s2/driver.h"
#include <array>
#include <map>
#include <memory>

using g6s2::Ds41PrefixCache;
static g6rt::CacheNamespace ns(int n) { g6rt::CacheNamespace s{}; s[0] = n; return s; }
static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)

struct Fixture {
    g6rt::Model m;
    std::vector<uint16_t> ring, comp;
    std::vector<uint8_t> packed;
    int hist[3 * 64]{};
    explicit Fixture(bool fp4) : ring(3 * 256 * 512, 7), comp(3 * 1024 * 512, 7), packed(3 * 1024 * 288, 7) {
        m.layers.resize(1); m.ehist = hist;
        auto& c = m.layers[0].C; c.ring = ring.data(); c.comp_ss = 1024 * size_t(fp4 ? 288 : 512);
        if (fp4) c.compq = packed.data(); else c.comp = comp.data();
    }
};

// the bind's eviction of a parked slot, as ds41_native.cuh runs it (keep = the grant's claimed generation)
static uint64_t evict_parked(Ds41PrefixCache* ck, int q, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& n, uint64_t keep) {
    const uint64_t gen = ck->capture(q, pos, tokens, n, 0, keep);
    g6s2::require(!keep || ck->find(keep), "DS41 park: the evicted chat's capture evicted the grant's own snapshot");
    return gen;
}

static void tight_budget(bool fp4, bool pressure) {
    Fixture f(fp4);
    Ds41PrefixCache cache(f.m, nullptr, 1);   // 1 MiB
    std::vector<int> t(2048); for (int i = 0; i < 2048; ++i) t[size_t(i)] = i + 1;
    const uint32_t pe_pos = fp4 ? 1800u : 1000u, re_pos = pe_pos + 200;
    const size_t pe_b = fp4 ? 521344 : 774144, re_b = fp4 ? 550144 : 876544;
    CHECK(cache.entry_bytes(pe_pos) == pe_b && cache.entry_bytes(re_pos) == re_b);
    CHECK(pe_b + re_b > cache.budget_bytes() && re_b < cache.budget_bytes());
    const uint64_t pe = cache.capture(0, pe_pos, t, ns(1), 0);
    CHECK(pe != 0);
    std::vector<int> next = t; next[pe_pos + 10]++;   // matches PE, not the parked RE
    const auto hit = cache.match(next, int(re_pos), ns(1));
    CHECK(hit.hit && hit.generation == pe);
    if (pressure) alloc_limit = live_bytes + re_b - 1;   // the budget would fit after evicting PE; device memory does not
    bool threw = false;
    uint64_t gen = 1;
    try { gen = evict_parked(&cache, 0, re_pos, t, ns(1), pe); } catch (const std::runtime_error& e) { threw = true; fprintf(stderr, "abort: %s\n", e.what()); }
    alloc_limit = SIZE_MAX;
    CHECK(!threw);
    CHECK(gen == 0);                       // the optional capture is skipped
    CHECK(cache.find(pe) != nullptr);      // the grant's snapshot survives and restores
    CHECK(cache.evictions() == 0 && cache.used_bytes() == pe_b && cache.used_bytes() <= cache.budget_bytes());
    cache.restore(pe, 0, 0);
    printf("ds41_park_evict_test %s%s: budget %zu PE %zu RE %zu -> capture skipped, PE kept and restored\n", fp4 ? "FP4" : "BF16",
           pressure ? " alloc-pressure" : "", cache.budget_bytes(), pe_b, re_b);
}

// keep only protects one entry: an older entry is still evicted for the capture, and keep = 0 is the knob-off eviction
static void others_evicted(bool fp4) {
    Fixture f(fp4);
    // PE + another entry fit, PE + RE fit, all three do not: BF16 2 MiB with 774144 + 774144 + 876544 B, FP4 1 MiB with
    // 400096 + 400096 + 406144 B
    Ds41PrefixCache cache(f.m, nullptr, fp4 ? 1 : 2);
    std::vector<int> t(2048); for (int i = 0; i < 2048; ++i) t[size_t(i)] = i + 1;
    std::vector<int> u = t; u[0] = 99999;
    const uint32_t pe_pos = fp4 ? 958u : 1000u, re_pos = fp4 ? 1000u : 1200u;
    CHECK(2 * cache.entry_bytes(pe_pos) <= cache.budget_bytes() && cache.entry_bytes(pe_pos) + cache.entry_bytes(re_pos) <= cache.budget_bytes() &&
          2 * cache.entry_bytes(pe_pos) + cache.entry_bytes(re_pos) > cache.budget_bytes());
    const uint64_t pe = cache.capture(0, pe_pos, t, ns(1), 0);
    const uint64_t other = cache.capture(0, pe_pos, u, ns(1), 0);   // newer than PE
    CHECK(pe && other);
    const uint64_t gen = evict_parked(&cache, 0, re_pos, t, ns(1), pe);
    CHECK(gen != 0 && cache.find(pe) && !cache.find(other) && cache.evictions() == 1);
    // keep = 0: plain LRU (the oldest goes first), as before this change
    const uint64_t gen2 = cache.capture(0, re_pos, u, ns(1), 0);
    CHECK(gen2 != 0 && !cache.find(pe) && cache.find(gen) && cache.evictions() == 2);
    printf("ds41_park_evict_test %s: keep spares only the claimed entry (evictions %llu)\n", fp4 ? "FP4" : "BF16", (unsigned long long)cache.evictions());
}


// The same through admission (Astra's regression): Driver + Scheduler with GR8_DS_PARK on and 2 slots, a test native layer
// that plays ds41_native.cuh (prompt-end capture, park at finish, the bind's eviction of a parked slot and the restore of
// the grant's hit). Chats A and B finish and park, so both slots are occupied; chat C's prompt matches A's prompt-end
// snapshot PE but not A's parked tokens (thinking on, no tools); the grant lands on A's parked slot (the oldest), whose
// eviction capture of A's request end RE does not fit beside PE.
static int tok(uint32_t chat, uint32_t i) { return int((chat * 2654435761u ^ i * 40503u) % 5); }
static void admission(bool fp4, bool pressure)
{
    using namespace g6s2;
    Fixture f(fp4);
    Ds41PrefixCache cache(f.m, nullptr, 1), *ck = &cache;
    const uint32_t len = fp4 ? 1901 : 520;   // PE + RE over 1 MiB, each under it
    Config c;
    c.epoch = 1; c.mtp = 1; c.max_rows = 8; c.max_sessions = 2; c.prefill_rows = 64; c.two_lanes = false; c.park = true;
    struct Parked { bool on = false; std::vector<int> tokens; uint32_t pos = 0; };
    std::array<Parked, 2> parked{};
    std::map<uint64_t, std::vector<int>> prompt, out;
    std::map<uint64_t, uint32_t> maxt;
    uint64_t claimed = 0, pe = 0, re_gen = 1;
    bool threw = false;
    std::unique_ptr<Driver> d;
    DriverHooks h;
    h.bind = [&](const Grant& g) {   // ds41_native.cuh h.bind with park, then the restore of the grant's hit
        const int q = int(g.key.slot);
        Parked& k = parked[size_t(q)];
        try {
            if (k.on) re_gen = evict_parked(ck, q, k.pos, k.tokens, ns(1), claimed);
            k = Parked{};
            if (g.reuse) cache.restore(claimed, q, 0);
        } catch (const std::runtime_error& e) { threw = true; fprintf(stderr, "abort: %s\n", e.what()); }
        claimed = 0;
    };
    h.retire = [](const Slot&) {};
    h.prefill = [&](const Slot& sl, const Payload&, uint32_t first, uint32_t rows) {   // prompt end: the PE capture of chat A
        if (sl.req.id == 1 && first + rows == sl.req.prompt_tokens - 1) pe = cache.capture(int(sl.key.slot), first + rows, prompt.at(1), ns(1), 0);
    };
    h.run = [&](const s2::Step& st) { std::vector<int> ids; for (const auto& r : st.r) ids.push_back(tok(r.pos, uint32_t(r.tok))); return ids; };
    h.emit = [&](Key k, const int* ids, int n, int) {
        const Slot& sl = d->state().slots()[k.slot];
        auto& o = out[sl.req.id];
        const int cnt = std::min<int>(n, int(sl.req.max_tokens - o.size()));
        o.insert(o.end(), ids, ids + cnt);
        return Emission{uint32_t(cnt), o.size() == sl.req.max_tokens};
    };
    h.finished = [&](const Slot& sl) {   // the park hook: the slot keeps tokens [0, pos)
        Parked& k = parked[sl.key.slot];
        k.on = true; k.tokens = prompt.at(sl.req.id);
        k.tokens.insert(k.tokens.end(), out.at(sl.req.id).begin(), out.at(sl.req.id).end());
        k.pos = uint32_t(k.tokens.size()) - 1;
    };
    d = std::make_unique<Driver>(c, h);
    auto admit = [&](uint64_t id, std::vector<int> pr, uint32_t mt, uint32_t reuse) {
        const Scheduler& sc = d->state();
        if (!sc.resident()) d->idle_boundary(sc.boundary() + NOTICE_MIN);
        prompt[id] = pr; maxt[id] = mt;
        const Request rq{id, uint32_t(pr.size()), mt};
        Commit cm; cm.epoch = 1; cm.event = sc.next_event(); cm.boundary = sc.boundary(); cm.before_hash = sc.digest();
        cm.admit = Grant{sc.next_key(), rq, Run{0, reserve_pages(c, rq), 1, id * 77}, reuse};
        Payload p; p.prompt = pr; p.chunk = 4;
        d->apply(cm, &p);
        return cm.admit->key.slot;
    };
    auto drain = [&] { for (int g = 0; d->state().resident() && g < 100000; ++g) d->step(); };
    std::vector<int> a(len), b(len / 2);
    for (uint32_t i = 0; i < len; ++i) a[i] = tok(1, i);
    for (uint32_t i = 0; i < len / 2; ++i) b[i] = tok(2, i);
    admit(1, a, 3, 0); drain();
    admit(2, b, 3, 0); drain();
    CHECK(pe != 0 && parked[0].on && parked[1].on);
    const uint32_t pe_pos = len - 1;
    std::vector<int> cp(a.begin(), a.begin() + pe_pos);   // A's prompt, then a new turn that drops A's reply (thinking)
    for (int i = 0; i < 40; ++i) cp.push_back((tok(3, uint32_t(i)) + 1) % 5);
    const auto hit = cache.match(cp, int(cp.size()) - 1, ns(1));
    CHECK(hit.hit && hit.generation == pe && hit.position == pe_pos);
    CHECK(g6rt::PrefixReuse::common_prefix(parked[0].tokens, cp) < int(parked[0].pos));   // not a resume of A
    CHECK(cache.entry_bytes(pe_pos) + cache.entry_bytes(parked[0].pos) > cache.budget_bytes());
    claimed = pe;
    if (pressure) alloc_limit = live_bytes + cache.entry_bytes(parked[0].pos) - 1;
    const unsigned slot = admit(3, cp, 4, pe_pos);
    alloc_limit = SIZE_MAX;
    CHECK(slot == 0);   // the occupied slot: A's, parked first
    CHECK(!threw && re_gen == 0 && cache.find(pe) && cache.evictions() == 0 && cache.hits() == 1);
    drain();
    CHECK(out.at(3).size() == 4 && d->state().slots()[0].req.id == 3);
    printf("ds41_park_evict_test admission %s%s: chat C on A's parked slot %u, A's capture skipped, PE restored, C served\n",
           fp4 ? "FP4" : "BF16", pressure ? " alloc-pressure" : "", slot);
}

int main() {
    for (bool fp4 : {false, true}) {
        tight_budget(fp4, false);
        tight_budget(fp4, true);
        others_evicted(fp4);
        admission(fp4, false);
        admission(fp4, true);
    }
    CHECK(mem.empty());
    if (fails) { fprintf(stderr, "ds41_park_evict_test: %d FAIL\n", fails); return 1; }
    puts("ds41_park_evict_test: ok (10 cases)");
    return 0;
}
