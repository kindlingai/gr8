#pragma once
// fc2 (down-projection) adapter for the sweep bench: the served v2 routed-expert down projection
//   mxtc::k_fc2_tc<576,5120,4,2,8,8,4>   (engine/kernels/ds_moe_tc.cuh), launched once per column group (4 launches).
// All values are SYNTHETIC; shapes and constants are copied from engine/runtime/ds41_v2_ffn.cuh (args(), down_group()).
// The timed region is the four fc2 launches only: no combine, no shared GEMM, no plan build (all outside timing).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../scratch_guard.cuh"
#include "kernels/ds41_v2_moe.cuh"   // via -I engine: PlanMT, plan kernels, mxtc::Args, k_fc2_tc

#define SWEEP_ADAPTER_HAS_METADATA 1
// Opt in to bench.cu's row-mask compare: k_fc2_tc legitimately leaves masked rows unwritten, so live rows must match
// byte for byte and every dead row must still hold the harness poison (a write into a dead row is a mismatch).
#define SWEEP_ADAPTER_HAS_ROW_MASK 1

namespace sweep_adapter {
// ---- constants mirrored from engine/runtime/ds41_v2_ffn.cuh (test_fc2_adapter.py checks them against that file)
constexpr int kH = 5120, kI = 576, kNE = 384, kTK = 6, kMM = 8192, kColGroups = 4;
constexpr int kW2 = 4, kTW = 2, kTPC = 8, kST2 = 8, kGY2 = 4, kOS2 = 5;
constexpr int kNG = kI / mxtc::KG, kOT = kH / 16;
static_assert(mxtc::GR == 544 && kNG == 9 && kOT == 320 && kMM == dsr::RO_PLAN_MAX, "fc2 record geometry");
using Plan = dsr::PlanMT<kTK, kMM>;
struct Params { mxtc::Args a; int M; };

struct Shape { int M = 0, experts = kNE; std::string pattern = "uniform"; int holes = 0; };
inline Shape shape(const std::vector<std::string>& a) {
    if (a.empty() || (a.size() - 1) % 2) throw std::runtime_error("fc2 needs M followed by option/value pairs");
    Shape s; s.M = std::stoi(a[0]);
    for (size_t i = 1; i < a.size(); i += 2) {
        const auto& k = a[i]; const auto& v = a[i + 1];
        if (k == "--experts") s.experts = std::stoi(v);
        else if (k == "--pattern") s.pattern = v;
        else if (k == "--holes") { if (v != "0" && v != "1") throw std::runtime_error("holes must be 0 or 1"); s.holes = v == "1"; }
        else throw std::runtime_error("unknown fc2 option " + k);
    }
    if (s.M < 1 || s.M > kMM || s.experts < kTK || s.experts > kNE || (s.pattern != "uniform" && s.pattern != "skew"))
        throw std::runtime_error("invalid fc2 shape: M 1..8192, experts 6..384, pattern uniform|skew");
    return s;
}
inline int tile_bound(const Shape& s) { return dsr::pw_tile_bound(s.M * kTK, s.experts); }
inline size_t weight_bytes(const Shape& s) { return size_t(s.experts) * kOT * kNG * mxtc::GR; }
inline size_t plan_bytes() { return sizeof(Plan); }
inline size_t output_rows(const Shape& s) { return size_t(tile_bound(s)) * 8; }
// Upper bound: output buffers and aq/as use the tile BOUND (any fixture can serve as another fixture's output slot).
inline size_t required_bytes(const std::vector<std::string>& args) {
    const auto s = shape(args); const size_t rows = output_rows(s), rz = 2 * sweep_memory::redzone_bytes;
    const size_t temporaries = size_t(s.M) * kTK * 8 + size_t(kNE) * (kMM / 32) * 4 + kNE * 4;
    return weight_bytes(s) + rz + rows * kI + rz + rows * mxtc::ASP + rz + plan_bytes() + rz + rows + rz +
           2 * (rows * kH * 2 + rz) + sweep_memory::digest_bytes(weight_bytes(s)) + rz + temporaries + (1u << 20);
}

struct Fixture {
    Shape s; uint64_t seed = 0;
    uint8_t *fc2 = nullptr, *aq = nullptr, *as = nullptr, *live = nullptr; Plan* plan = nullptr; __half *out_a = nullptr, *out_b = nullptr;
    size_t rows = 0, aq_rows = 0; int n_slots = 0, n_tiles = 0, bound = 0, live_routes = 0, distinct = 0, min_n = 0, max_n = 0;
    double mean_n = 0; mxtc::Args a{};
    std::unique_ptr<sweep_memory::Registry> memory;
};
inline void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        if (e == cudaErrorMemoryAllocation) { fprintf(stderr, "LOW_MEMORY: %s\n", cudaGetErrorString(e)); std::exit(77); }
        throw std::runtime_error(cudaGetErrorString(e));
    }
}
// ---- deterministic synthetic data. Data ranges keep every product finite in fp16 (same ranges as the engine harness).
__host__ __device__ inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
// weights: 544-byte records = 512 hashed data bytes, then 32 E8M0 scale bytes 117..122
static __global__ void k_fill_fc2(uint64_t* w, size_t words, uint64_t key) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
        uint64_t h = mix(key + i);
        if (i % (mxtc::GR / 8) >= mxtc::SOFF / 8) { uint64_t o = 0; for (int b = 0; b < 8; ++b) o |= uint64_t(117 + ((h >> (8 * b)) & 255) % 6) << (8 * b); h = o; }
        w[i] = h;
    }
}
// activations: E4M3 bytes with sign + exponent <= 13 (finite, bounded); scales E8M0 118..123
static __global__ void k_fill_aq(uint64_t* w, size_t words, uint64_t key) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
        const uint64_t h = mix(key + i); uint64_t o = 0;
        for (int b = 0; b < 8; ++b) { const uint64_t v = (h >> (8 * b)) & 255; o |= ((v & 0x80) | (v % 0x70)) << (8 * b); }
        w[i] = o;
    }
}
static __global__ void k_fill_as(uint64_t* w, size_t words, uint64_t key) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
        const uint64_t h = mix(key + i); uint64_t o = 0;
        for (int b = 0; b < 8; ++b) o |= uint64_t(118 + ((h >> (8 * b)) & 255) % 6) << (8 * b);
        w[i] = o;
    }
}
// live[r] = 1 iff output row r (tile r/8, column r%8) is written by the kernel: tile < n_tiles and its mask bit is set
static __global__ void k_row_live(const Plan* p, uint8_t* live, size_t rows) {
    const size_t r = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < rows) { const size_t t = r >> 3; live[r] = t < size_t(p->n_tiles) && ((p->tile_mask[t] >> (r & 7)) & 1); }
}
inline void fill(Fixture& f, uint64_t key, cudaStream_t st) {
    auto grid = [](size_t words) { return unsigned(std::min<size_t>((words + 255) / 256, 1u << 20)); };
    const size_t ww = weight_bytes(f.s) / 8, aw = f.aq_rows * kI / 8, sw = f.aq_rows * mxtc::ASP / 8;
    k_fill_fc2<<<grid(ww), 256, 0, st>>>(reinterpret_cast<uint64_t*>(f.fc2), ww, key ^ 0x243f6a8885a308d3ULL);
    k_fill_aq<<<grid(aw), 256, 0, st>>>(reinterpret_cast<uint64_t*>(f.aq), aw, key ^ 0x13198a2e03707344ULL);
    k_fill_as<<<grid(sw), 256, 0, st>>>(reinterpret_cast<uint64_t*>(f.as), sw, key ^ 0xa4093822299f31d0ULL);
    ck(cudaGetLastError());
}

// ---- routing (host): top-6 WITHOUT replacement per token. uniform = equal odds; skew = power-law popularity (synthetic stress,
// not a measured router distribution) drawn by an exponential race. holes=1 applies the engine harness' cut rule (idx -1, weight 0).
struct Rng { uint64_t s; uint64_t next() { return mix(s++); } double unit() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); } };
inline void route(const Shape& s, uint64_t seed, std::vector<int>& idx, std::vector<float>& wt) {
    idx.assign(size_t(s.M) * kTK, -1); wt.assign(size_t(s.M) * kTK, 0.f);
    Rng rng{seed * 0x9e3779b97f4a7c15ULL + 0x5851f42d4c957f2dULL};
    std::vector<double> inv(s.experts);
    for (int e = 0; e < s.experts; ++e) inv[e] = std::pow(double(e + 1), 0.8);   // 1/popularity
    std::vector<std::pair<double, int>> race(s.experts);
    for (int m = 0; m < s.M; ++m) {
        int pick[kTK];
        if (s.pattern == "skew") {
            for (int e = 0; e < s.experts; ++e) race[e] = {-std::log(rng.unit()) * inv[e], e};
            std::partial_sort(race.begin(), race.begin() + kTK, race.end());
            for (int k = 0; k < kTK; ++k) pick[k] = race[k].second;
        } else {
            for (int k = 0; k < kTK; ++k) for (;;) {
                const int e = int(rng.next() % s.experts); bool dup = false;
                for (int j = 0; j < k; ++j) dup |= pick[j] == e;
                if (!dup) { pick[k] = e; break; }
            }
        }
        for (int k = 0; k < kTK; ++k) {
            const bool cut = s.holes && (m % 19 == 0 || (m + k) % 5 == 2 || (m % 7 == 1 && k >= 3));
            const uint64_t h = rng.next();   // drawn for cut routes too, so holes do not shift the later tokens' routes
            if (cut) continue;
            idx[size_t(m) * kTK + k] = pick[k]; wt[size_t(m) * kTK + k] = float(1 + h % 31) / 32.f;
        }
    }
}
template <class T> T* alloc(size_t count) { return static_cast<T*>(sweep_memory::allocate(count * sizeof(T))); }

inline Fixture make_case(const std::vector<std::string>& args, uint64_t seed) {
    Fixture f; f.s = shape(args); f.seed = seed; const auto& s = f.s;
    f.bound = tile_bound(s); f.rows = output_rows(s);
    f.memory = std::make_unique<sweep_memory::Registry>();
    sweep_memory::AllocationScope scope(*f.memory);
    // Plan: the engine's own kernels, in the served order (ds41_v2_ffn.cuh up()).
    std::vector<int> idx; std::vector<float> wt; route(s, seed, idx, wt);
    int *d_idx, *d_tile0; float* d_wt; unsigned* d_masks;
    ck(cudaMalloc(&d_idx, idx.size() * 4)); ck(cudaMalloc(&d_wt, wt.size() * 4));
    ck(cudaMalloc(&d_masks, size_t(kNE) * (kMM / 32) * 4)); ck(cudaMalloc(&d_tile0, kNE * 4));
    ck(cudaMemcpy(d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice)); ck(cudaMemcpy(d_wt, wt.data(), wt.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(d_masks, 0, size_t(kNE) * (kMM / 32) * 4)); ck(cudaMemset(d_tile0, 0, kNE * 4));
    f.plan = alloc<Plan>(1);
    ds41v2moe::k_import_routes<kNE, kTK, kMM><<<(s.M * 8 + 255) / 256, 256>>>(d_idx, d_wt, s.M, f.plan, d_masks);
    dsr::k_scan_ro<kNE, kTK, kMM><<<1, kNE>>>(f.plan, d_masks, d_tile0);
    ds41v2moe::k_place_routes<kTK, kMM><<<(s.M * kTK + 255) / 256, 256>>>(f.plan, d_masks, d_tile0, s.M);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    int head[2]; ck(cudaMemcpy(head, f.plan, sizeof(head), cudaMemcpyDeviceToHost));
    f.n_slots = head[0]; f.n_tiles = head[1];
    ck(cudaFree(d_idx)); ck(cudaFree(d_wt)); ck(cudaFree(d_masks)); ck(cudaFree(d_tile0));
    if (f.n_slots < 1 || f.n_tiles < 1 || f.n_tiles > f.bound || f.n_slots > dsr::ro_slots_bound(s.M * kTK, kNE))
        throw std::runtime_error("fc2 plan out of bounds (n_slots " + std::to_string(f.n_slots) + ", n_tiles " + std::to_string(f.n_tiles) + ")");
    // Routing statistics from the host copy of the same routes.
    std::vector<int> per(s.experts, 0);
    for (int v : idx) if (v >= 0) { ++per[v]; ++f.live_routes; }
    f.min_n = 1 << 30; int total = 0;
    for (int n : per) if (n) { ++f.distinct; f.min_n = std::min(f.min_n, n); f.max_n = std::max(f.max_n, n); total += n; }
    f.mean_n = f.distinct ? double(total) / f.distinct : 0;
    if (f.distinct != f.n_slots) throw std::runtime_error("fc2 plan slot count disagrees with host routing");
    f.aq_rows = size_t(f.n_tiles) * 8;
    f.fc2 = alloc<uint8_t>(weight_bytes(s)); f.aq = alloc<uint8_t>(f.aq_rows * kI); f.as = alloc<uint8_t>(f.aq_rows * mxtc::ASP);
    f.live = alloc<uint8_t>(f.rows);
    f.out_a = alloc<__half>(f.rows * kH); f.out_b = alloc<__half>(f.rows * kH);
    fill(f, 0, nullptr);
    k_row_live<<<unsigned((f.rows + 255) / 256), 256>>>(f.plan, f.live, f.rows);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    auto& a = f.a;   // mirrors ds41_v2_ffn.cuh args()
    a.fc2 = f.fc2; a.slot_expert = f.plan->slot_expert; a.slot_tile0 = f.plan->slot_tile0; a.slot_nt = f.plan->slot_nt;
    a.tile_mask = f.plan->tile_mask; a.tile_w = &f.plan->tile_w[0][0]; a.tile_tok = &f.plan->tile_tok[0][0];
    a.n_slots_dev = &f.plan->n_slots; a.n_tiles_dev = &f.plan->n_tiles; a.n_slots = dsr::ro_slots_bound(s.M * kTK, kNE);
    a.gy2 = kGY2; a.os2 = kOS2; a.aq = f.aq; a.as = f.as; a.kc = kTK;
    auto& m = *f.memory;
    m.describe(f.fc2, "fc2_weights"); m.describe(f.aq, "act_q"); m.describe(f.as, "act_scales"); m.describe(f.plan, "plan");
    m.describe(f.live, "row_live"); m.describe(f.out_a, "reference_output", false); m.describe(f.out_b, "candidate_output", false);
    m.prepare();
    return f;
}
inline Params params(Fixture& f, bool ref) { Params p{f.a, f.s.M}; p.a.ypart = ref ? f.out_a : f.out_b; return p; }
inline Params params_with_output(Fixture& in, Fixture& out, bool ref) { Params p{in.a, in.s.M}; p.a.ypart = ref ? out.out_a : out.out_b; return p; }
inline void* output(Fixture& f, bool ref) { return ref ? f.out_a : f.out_b; }
inline size_t output_bytes(const Fixture& f) { return f.rows * kH * 2; }
inline size_t row_bytes(const Fixture&) { return size_t(kH) * 2; }
inline const unsigned char* row_live(Fixture& f) { return f.live; }
// New synthetic weights / activations / scales; routing and plan stay fixed per fixture. Snapshot afterwards.
inline void refresh_inputs(Fixture& f, uint64_t nonce, cudaStream_t st) {
    f.memory->invalidate_snapshot();
    fill(f, nonce ^ (f.seed * 0x9e3779b97f4a7c15ULL), st);
}
inline void snapshot_inputs(Fixture& f, cudaStream_t st) { f.memory->snapshot(st); }
inline size_t check_integrity(Fixture& f, cudaStream_t st) { return f.memory->check(st) + sweep_memory::check_private_scratch(st); }
inline std::string metadata_json(const Fixture& f) {
    std::ostringstream o;
    o << "{\"input_provenance\":\"synthetic\",\"M\":" << f.s.M << ",\"experts\":" << f.s.experts << ",\"pattern\":\"" << f.s.pattern
      << "\",\"holes\":" << f.s.holes << ",\"n_slots\":" << f.n_slots << ",\"n_tiles\":" << f.n_tiles << ",\"tile_bound\":" << f.bound
      << ",\"launch_slots\":" << f.a.n_slots << ",\"live_routes\":" << f.live_routes << ",\"distinct_experts\":" << f.distinct
      << ",\"tokens_per_expert_min\":" << f.min_n << ",\"tokens_per_expert_mean\":" << f.mean_n << ",\"tokens_per_expert_max\":" << f.max_n
      << ",\"weight_bytes\":" << weight_bytes(f.s) << ",\"output_rows\":" << f.rows << ",\"column_groups\":" << kColGroups
      << ",\"timed_region\":\"4x k_fc2_tc col_group0..3, unfused (no combine), plan and operands outside timing\""
      << ",\"private_scratch_guarded_device_bytes\":" << sweep_memory::private_scratch().device_bytes() << ",\"decision_rows\":false}";
    return o.str();
}
inline void destroy(Fixture& f) {
    for (void* p : {static_cast<void*>(f.fc2), static_cast<void*>(f.aq), static_cast<void*>(f.as), static_cast<void*>(f.live),
                    static_cast<void*>(f.plan), static_cast<void*>(f.out_a), static_cast<void*>(f.out_b)}) sweep_memory::deallocate(p);
    if (f.memory) f.memory->release_workspace();
    f.memory.reset();
}
}  // namespace sweep_adapter
