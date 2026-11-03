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

int main() {
    for (bool fp4 : {false, true}) {
        tight_budget(fp4, false);
        tight_budget(fp4, true);
        others_evicted(fp4);
    }
    CHECK(mem.empty());
    if (fails) { fprintf(stderr, "ds41_park_evict_test: %d FAIL\n", fails); return 1; }
    puts("ds41_park_evict_test: ok (6 cases)");
    return 0;
}
