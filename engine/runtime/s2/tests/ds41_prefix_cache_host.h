// Host-memory CUDA substitutes for prefix-cache bookkeeping tests; no GPU runtime is linked.
#pragma once
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
#include "kv_state_host_tiers.h"
#define GR8_DS41_PREFIX_CACHE_STANDALONE 1
#include "s2/ds41_prefix_cache.cuh"
#include "s2/driver.h"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>

using g6s2::Ds41PrefixCache;
static g6rt::CacheNamespace ns(int n) { g6rt::CacheNamespace s{}; s[0] = n; return s; }
static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)

struct Fixture {
    g6rt::Model m;
    std::vector<uint16_t> ring, comp;
    std::vector<uint8_t> packed;
    std::vector<int> hist;
    explicit Fixture(bool fp4, int slots = 3) : ring(slots * 256 * 512, 7), comp(slots * 1024 * 512, 7), packed(slots * 1024 * 288, 7), hist(slots * 64) {
        m.layers.resize(1); m.ehist = hist.data();
        auto& c = m.layers[0].C; c.ring = ring.data(); c.comp_ss = 1024 * size_t(fp4 ? 288 : 512);
        if (fp4) c.compq = packed.data(); else c.comp = comp.data();
    }
};
