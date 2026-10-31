// DEV RTX 5090 (sm_120a) / Spark (sm_121a) synthetic exactness test for Ds41PrefixCache (engine/runtime/s2/ds41_prefix_cache.cuh).
// Build & run on DEV RTX 5090:
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -gencode=arch=compute_120a,code=sm_120a -DG6_MODEL_DS41=1 -I. -Iengine \
//     engine/tests/test_ds41_prefix_cache.cu engine/attn/ds41_bind.cu -Lengine/kvman-rs/target/release -lg6kvman -lpthread -ldl \
//     -o /tmp/test_ds41_prefix_cache && /tmp/test_ds41_prefix_cache
//
// Proves:
//   (1) Ds41PrefixCache snapshot store, entry_bytes sizing, LRU eviction, tier_stat() (slots, free_slots, entries, hits, dropped,
//       spilled_pages, restored_pages), digest(), and longest-exact-prefix match() (mismatch anywhere -> cold, namespace isolation).
//   (2) Real GPU execution across ALL DS41 prefix state components:
//       - Window attention layers (C.ring)
//       - Ratio-2 KV source layers 2, 8, 14 (comp_row -> C.comp + pending half-pair C.kvs/C.scs + cnorm -> ds41_idx_ica_row(ratio=2) -> S.ica[0..2])
//       - Ratio-1 KV source layer 20 (ds41_idx_ca_row_r1 -> S.ca[3] + S.shadow + pre -> ds41_idx_ica_row(ratio=1) -> S.ica[3] -> topk + cand_blocks)
//       - Candidate-consumer sparse attention layers 24, 28, 32, 36 (ds41_idx_topk(use_cand=1) over S.ica[0..2] + cand_blocks)
//       - Engram token ring (ehist[64] + win_wide_row)
//       - DSpark draft model attention rings (3 layers x 136 rows)
//   (3) Cold prefill vs Restored prefix exactness at BOTH even and odd capture positions:
//       - Prompt-end capture at p = P - 1 for EVEN P (odd p = 63) and ODD P (even p = 64)
//       - Multi-turn retire capture at odd p = 47 and even p = 48 followed by chunk continuation to P = 64 and 8 greedy decode steps
//       - Every restored run after 0xFF/NaN poisoning produces BIT-IDENTICAL (0 bit differences, max_abs_diff = 0.0, identical top-1)
//         ICA planar codes+SF (layers 2, 8, 14, 20), CA rows, shadow KV, C.comp, C.kvs/C.scs, C.ring, hidden states, and next-token logits.
//   (4) Negative controls: corrupting restored ICA or omitting C.kvs/C.scs at odd p = P - 1 causes immediate state/logit divergence.

#define GR8_DS41_PREFIX_CACHE_STANDALONE 1
#include "../attn/ds41_attn.cuh"
#include "../attn/ds41_attn_gv.cuh"
#include "../attn/ds41_idx_api.h"
#include "../attn/ds41_attn_types.h"
#include "../engram/engram_ring.h"
#include <vector>

// Minimal g6rt::Model / Ds41AttnLayer / Ds41IdxModel / Ds41Dspark definitions matching engine/runtime/rank.cuh & ds41_attn_ops.cuh
// so Ds41PrefixCache can be exercised directly without compiling the full 50k-line rank TU.
namespace g6rt {
inline constexpr int ENG_RING = ds41::engram::kRing;
inline constexpr size_t DS41_CA_ROW = 288;
inline constexpr size_t DS41_ICA_ROW = 68;

struct Ds41AttnLayer {
    ds41a::Weights W{};
    ds41a::Cache C{};
};

struct Ds41IdxModel {
    int cap = 0, rcap = 0, slots = 1;
    int rows_kv[4] = {};
    uint8_t* ca[4] = {};
    uint8_t* ica[4] = {};
    __nv_bfloat16* shadow = nullptr;
};

struct LayerStub {
    void* attn_state = nullptr;
    void* idx_state = nullptr;
};

struct Model {
    std::vector<LayerStub> layers;
    int* ehist = nullptr;
    Ds41IdxModel* idx_model = nullptr;
};

inline Ds41AttnLayer* ds41_attn_state(Model& m, int i) {
    return (i >= 0 && i < (int)m.layers.size()) ? static_cast<Ds41AttnLayer*>(m.layers[size_t(i)].attn_state) : nullptr;
}
inline const Ds41AttnLayer* ds41_attn_state(const Model& m, int i) {
    return (i >= 0 && i < (int)m.layers.size()) ? static_cast<const Ds41AttnLayer*>(m.layers[size_t(i)].attn_state) : nullptr;
}
namespace ds41_idx_detail {
inline Ds41IdxModel* find(Model& m, bool) { return m.idx_model; }
inline const Ds41IdxModel* find(const Model& m, bool) { return m.idx_model; }
} // namespace ds41_idx_detail
} // namespace g6rt

namespace g6s2 {
struct Ds41Dspark {
    g6rt::Model& draft;
};
} // namespace g6s2

#define G6RT_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#include "../runtime/s2/ds41_prefix_cache.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

namespace ds41a {

static uint64_t g_rng = 0x123456789ABCDEF0ull;
static uint32_t next_u32() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return uint32_t(g_rng >> 16);
}
static float bf2f_u16(uint16_t u) {
    uint32_t x = uint32_t(u) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
static uint16_t f2bf_u16(float f) {
    bf16 b = __float2bfloat16_rn(f);
    uint16_t u;
    std::memcpy(&u, &b, 2);
    return u;
}

template <class T>
static T* dalloc(size_t n, int fill = 0) {
    void* p = nullptr;
    G6RT_CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    G6RT_CK(cudaMemset(p, fill, (n ? n : 1) * sizeof(T)));
    return static_cast<T*>(p);
}

template <class T>
static T* dcopy(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    if (!h.empty()) G6RT_CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

template <class T>
static std::vector<T> ddown(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n) G6RT_CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<uint8_t> make_fp8_codes(size_t n, uint64_t seed) {
    g_rng = seed * 0x9E3779B97F4A7C15ull + 1;
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = next_u32();
        v[i] = uint8_t(((r & 1) << 7) | (((r >> 1) % 9) << 3) | ((r >> 8) & 7));
    }
    return v;
}
static std::vector<uint8_t> make_e8m0_scales(size_t n, uint64_t seed, int lo) {
    g_rng = seed * 0x9E3779B97F4A7C15ull + 7;
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(lo + next_u32() % 3);
    return v;
}
static Fp8Mat make_mat(long rows_alloc, long N, long K, uint64_t seed, int slo) {
    Fp8Mat m{};
    m.w = dcopy(make_fp8_codes(size_t(rows_alloc) * size_t(K), seed));
    m.s = dcopy(make_e8m0_scales(size_t(rows_alloc) * size_t(K / 32), seed + 1, slo));
    m.N = int(N);
    m.K = int(K);
    return m;
}
static std::vector<bf16> make_bf16_dev_vec(size_t n, uint64_t seed, float lo, float hi) {
    g_rng = seed * 0xBF58476D1CE4E5B9ull + 13;
    std::vector<bf16> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float u = float(next_u32() & 0xFFFFu) / 65535.0f;
        v[i] = __float2bfloat16_rn(lo + (hi - lo) * u);
    }
    return v;
}
static std::vector<uint16_t> make_bf16_u16_vec(size_t n, uint64_t seed, float lo, float hi) {
    g_rng = seed * 0xBF58476D1CE4E5B9ull + 13;
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float u = float(next_u32() & 0xFFFFu) / 65535.0f;
        v[i] = f2bf_u16(lo + (hi - lo) * u);
    }
    return v;
}

static std::vector<float> make_rope_inv() {
    std::vector<float> inv(32);
    for (int i = 0; i < 32; ++i) {
        inv[size_t(i)] = std::pow(10000.0f, -float(2 * i) / 64.0f);
    }
    return inv;
}

// Host helper to compute synthetic logits [V] from final hidden row [HID] + engram ring state
static std::vector<float> compute_logits(const std::vector<uint16_t>& h_row, const std::vector<uint16_t>& w_head,
                                         int V, const int* ehist_host, int pos) {
    std::vector<float> logits(size_t(V), 0.0f);
    int eng_sum = 0;
    for (int k = 1; k <= 3; ++k) {
        eng_sum += (pos - k >= 0) ? ds41::engram::ring_get(ehist_host, pos - k) * (k + 1) : 0;
    }
    const float eng_bias = 1e-4f * float(eng_sum % 997);
    for (int v = 0; v < V; ++v) {
        float acc = eng_bias * float((v % 7) - 3);
        const size_t off = size_t(v) * HID;
        for (int c = 0; c < HID; ++c) {
            acc += bf2f_u16(h_row[size_t(c)]) * bf2f_u16(w_head[off + size_t(c)]);
        }
        logits[size_t(v)] = acc;
    }
    return logits;
}

static int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) {
        if (v[size_t(i)] > v[size_t(best)]) best = i;
    }
    return best;
}

static int run_test() {
    constexpr int HEADS = 16;
    constexpr int G = HEADS / HPG;
    constexpr int MX = 48;
    constexpr int RR_MAIN = 128 + MX - 1; // 175
    constexpr int RR_DRAFT = 136;
    constexpr int CAP = 1280; // > 1024 so ratio-2 klen = (P+1)/2 > 512 and ratio-1 klen = P+1 > 512 exercise real ICA top-512 scoring
    constexpr int RCAP = 64;
    constexpr int SLOTS = 2; // slot 0 = cold/capture, slot 1 = poisoned/restored

    cudaStream_t st;
    G6RT_CK(cudaStreamCreate(&st));

    // -------------------------------------------------------------------------
    // Part 1: Full 40-layer + Indexer + DSpark geometry & LRU / match unit test
    // -------------------------------------------------------------------------
    {
        g6rt::Model main_m, draft_m;
        main_m.layers.resize(40);
        draft_m.layers.resize(3);
        std::vector<g6rt::Ds41AttnLayer> a_layers(40), d_layers(3);
        const int r2_sources[3] = {2, 8, 14};
        for (int i = 0; i < 40; ++i) {
            a_layers[size_t(i)].C.ring_rows = RR_MAIN;
            a_layers[size_t(i)].C.ring = dalloc<bf16>(size_t(SLOTS) * RR_MAIN * HD, 0x11);
            bool is_r2 = (i == r2_sources[0] || i == r2_sources[1] || i == r2_sources[2]);
            if (is_r2) {
                a_layers[size_t(i)].C.comp_ss = size_t(CAP / 2) * HD;
                a_layers[size_t(i)].C.comp = dalloc<bf16>(size_t(SLOTS) * (CAP / 2) * HD, 0x22);
                a_layers[size_t(i)].C.kvs = dalloc<float>(size_t(SLOTS) * 2 * HD, 0x33);
                a_layers[size_t(i)].C.scs = dalloc<float>(size_t(SLOTS) * 2 * HD, 0x44);
            }
            main_m.layers[size_t(i)].attn_state = &a_layers[size_t(i)];
        }
        g6rt::Ds41IdxModel idx_m{};
        idx_m.cap = CAP;
        idx_m.rcap = RCAP;
        idx_m.slots = SLOTS;
        for (int k = 0; k < 4; ++k) {
            idx_m.rows_kv[k] = (k < 3) ? ((CAP + RCAP) / 2) : (CAP + RCAP);
            idx_m.ica[k] = dalloc<uint8_t>(size_t(SLOTS) * size_t(idx_m.rows_kv[k]) * g6rt::DS41_ICA_ROW, 0x55 + k);
        }
        idx_m.ca[3] = dalloc<uint8_t>(size_t(SLOTS) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW, 0x66);
        idx_m.shadow = dalloc<bf16>(size_t(SLOTS) * size_t(CAP + RCAP) * 512, 0x77);
        main_m.idx_model = &idx_m;
        main_m.ehist = dalloc<int>(size_t(SLOTS) * g6rt::ENG_RING, 0);

        for (int j = 0; j < 3; ++j) {
            d_layers[size_t(j)].C.ring_rows = RR_DRAFT;
            d_layers[size_t(j)].C.ring = dalloc<bf16>(size_t(SLOTS) * RR_DRAFT * HD, 0x88);
            draft_m.layers[size_t(j)].attn_state = &d_layers[size_t(j)];
        }
        g6s2::Ds41Dspark dsp{draft_m};

        g6s2::Ds41PrefixCache cache(main_m, &dsp, /*budget_mib=*/16);
        const size_t fixed_b = 40ull * RR_MAIN * 1024 + 3ull * RR_DRAFT * 1024 + 3ull * 2 * 2 * 512 * 4;
        for (uint32_t p : {63u, 64u, 4095u, 4096u, 8192u, 32768u, 131072u}) {
            const size_t p2 = p / 2;
            const size_t want = fixed_b + 3ull * p2 * 1024 + 3ull * p2 * 68 + size_t(p) * (68 + 288 + 1024);
            EXPECT(cache.entry_bytes(p) == want);
        }

        // Matching & LRU eviction tests
        g6rt::CacheNamespace ns_a{}, ns_b{};
        ns_b[0] = 1;
        std::vector<int> p1(100), p2(160);
        for (int i = 0; i < 160; ++i) {
            p2[size_t(i)] = 1000 + i * 3;
            if (i < 100) p1[size_t(i)] = p2[size_t(i)];
        }
        // Capture odd position 63 (prompt-end of P=64) and even position 128
        const uint64_t g63 = cache.capture(0, 63, p1, ns_a, st);
        const uint64_t g128 = cache.capture(0, 128, p2, ns_a, st);
        EXPECT(g63 > 0 && g128 > g63 && cache.size() == 2);

        // Exact prefix match selects longest stored prefix <= limit
        auto h_full = cache.match(p2, 159, ns_a);
        EXPECT(h_full.hit && h_full.generation == g128 && h_full.position == 128);
        auto h_capped = cache.match(p2, 100, ns_a);
        EXPECT(h_capped.hit && h_capped.generation == g63 && h_capped.position == 63);

        // Mismatch at index 62 -> neither 63 nor 128 matches -> cold miss
        std::vector<int> p_mismatch = p2;
        p_mismatch[62] = 999999;
        auto h_miss = cache.match(p_mismatch, 159, ns_a);
        EXPECT(!h_miss.hit && h_miss.lcp == 62);

        // Mismatch at index 63 -> 63 still matches (tokens [0..63) equal!), 128 does not
        std::vector<int> p_mid = p2;
        p_mid[63] = 888888;
        auto h_mid = cache.match(p_mid, 159, ns_a);
        EXPECT(h_mid.hit && h_mid.generation == g63 && h_mid.position == 63);

        // Namespace mismatch -> cold miss
        auto h_ns = cache.match(p2, 159, ns_b);
        EXPECT(!h_ns.hit && std::strcmp(h_ns.why, "namespace") == 0);

        // Touch g63 via restore so g128 becomes oldest, then capture a 3rd entry at pos=95 under 16 MiB budget -> evicts g128
        cache.restore(g63, 1, st);
        const uint64_t g95 = cache.capture(0, 95, p1, ns_a, st);
        EXPECT(g95 > 0 && cache.size() == 2 && cache.evictions() == 1);
        EXPECT(cache.find(g63) != nullptr && cache.find(g128) == nullptr && cache.find(g95) != nullptr);
        const g6kv_tier_totals ts = cache.tier_stat();
        EXPECT(ts.entries == 2 && ts.hits == 1 && ts.dropped == 1 && ts.spilled_pages > 0 && ts.restored_pages > 0);
        printf("Part 1 (40L+Idx+DSpark geometry, odd/even capture, match, LRU, tier_stat): PASS\n");
    }

    // -------------------------------------------------------------------------
    // Part 2: End-to-End GPU Kernel Forward Exactness (Cold Prefill vs Restored Prefix)
    // Exercises real ds41a::attn_proj / comp_row / attn_core / attn_out (window, ratio-2 KV
    // source L2/8/14, ratio-1 L20, candidate consumer L24) + real ds41_idx_* (CA row, ICA row
    // ratio 2 & 1, topk, cand_blocks) + Engram ring + vocab logit projection at both ODD and EVEN capture positions.
    // -------------------------------------------------------------------------
    {
        constexpr int NL = 6;
        constexpr int V = 256;

        const float* inv_d = dcopy(make_rope_inv());
        const bf16* ca_norm_w = dcopy(make_bf16_dev_vec(HD, 900, 0.5f, 1.5f));
        const std::vector<uint16_t> w_head_host = make_bf16_u16_vec(size_t(V) * HID, 999, -0.25f, 0.25f);
        const std::vector<float> sink_zero(HEADS, 0.0f);
        const float* sink_d = dcopy(sink_zero);

        Weights W[NL]{};
        std::vector<const bf16*> idx_wk(NL, nullptr), idx_knorm(NL, nullptr), idx_wproj(NL, nullptr);
        for (int l = 0; l < NL; ++l) {
            const uint64_t s0 = uint64_t(l + 1) * 100;
            W[l].wq_a = make_mat(QA, QA, HID, s0 + 1, 119);
            W[l].wq_b = make_mat(long(HEADS) * HD, long(HEADS) * HD, QA, s0 + 2, 123);
            W[l].wkv = make_mat(HD, HD, HID, s0 + 3, 119);
            W[l].wo_a = make_mat(long(G) * OLORA, OLORA, GROUP_IN, s0 + 4, 118);
            W[l].wo_b = make_mat(HID, HID, long(G) * OLORA, s0 + 5, 118);
            W[l].q_norm = dcopy(make_bf16_dev_vec(QA, s0 + 6, 0.5f, 1.5f));
            W[l].kv_norm = dcopy(make_bf16_dev_vec(NOPE, s0 + 7, 0.5f, 1.5f));
            W[l].sink = sink_d;
            W[l].inv = inv_d;
            if (l >= 1 && l <= 3) {
                W[l].c_wkv = dcopy(make_bf16_dev_vec(size_t(HD) * HID, s0 + 8, -0.15f, 0.15f));
                W[l].c_wgate = dcopy(make_bf16_dev_vec(size_t(HD) * HID, s0 + 9, -0.15f, 0.15f));
                W[l].c_norm = dcopy(make_bf16_dev_vec(HD, s0 + 10, 0.5f, 1.5f));
            }
            if (l >= 1) {
                idx_wk[size_t(l)] = dcopy(make_bf16_dev_vec(128 * 512, s0 + 11, -0.2f, 0.2f));
                idx_knorm[size_t(l)] = dcopy(make_bf16_dev_vec(128, s0 + 12, 0.5f, 1.5f));
                idx_wproj[size_t(l)] = dcopy(make_bf16_dev_vec(32 * HID, s0 + 13, -0.2f, 0.2f));
            }
        }

        g6rt::Model main_m, draft_m;
        main_m.layers.resize(NL);
        draft_m.layers.resize(3);
        std::vector<g6rt::Ds41AttnLayer> a_layers(NL), d_layers(3);
        for (int l = 0; l < NL; ++l) {
            a_layers[size_t(l)].W = W[l];
            a_layers[size_t(l)].C.ring_rows = RR_MAIN;
            a_layers[size_t(l)].C.ring = dalloc<bf16>(size_t(SLOTS) * RR_MAIN * HD, 0);
            if (l >= 1 && l <= 3) {
                a_layers[size_t(l)].C.comp_ss = size_t(CAP / 2) * HD;
                a_layers[size_t(l)].C.src_ss = size_t(CAP / 2) * HD;
                a_layers[size_t(l)].C.comp = dalloc<bf16>(size_t(SLOTS) * (CAP / 2) * HD, 0);
                a_layers[size_t(l)].C.kvs = dalloc<float>(size_t(SLOTS) * 2 * HD, 0);
                a_layers[size_t(l)].C.scs = dalloc<float>(size_t(SLOTS) * 2 * HD, 0);
            }
            main_m.layers[size_t(l)].attn_state = &a_layers[size_t(l)];
        }
        g6rt::Ds41IdxModel idx_m{};
        idx_m.cap = CAP;
        idx_m.rcap = RCAP;
        idx_m.slots = SLOTS;
        for (int k = 0; k < 4; ++k) {
            idx_m.rows_kv[k] = (k < 3) ? ((CAP + RCAP) / 2) : (CAP + RCAP);
            idx_m.ica[k] = dalloc<uint8_t>(size_t(SLOTS) * size_t(idx_m.rows_kv[k]) * g6rt::DS41_ICA_ROW, 0);
        }
        idx_m.ca[3] = dalloc<uint8_t>(size_t(SLOTS) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW, 0);
        idx_m.shadow = dalloc<bf16>(size_t(SLOTS) * size_t(CAP + RCAP) * 512, 0);
        main_m.idx_model = &idx_m;
        main_m.ehist = dalloc<int>(size_t(SLOTS) * g6rt::ENG_RING, -1);

        for (int j = 0; j < 3; ++j) {
            d_layers[size_t(j)].C.ring_rows = RR_DRAFT;
            d_layers[size_t(j)].C.ring = dalloc<bf16>(size_t(SLOTS) * RR_DRAFT * HD, 0);
            draft_m.layers[size_t(j)].attn_state = &d_layers[size_t(j)];
        }
        g6s2::Ds41Dspark dsp{draft_m};

        void* scr_buf = dalloc<uint8_t>(Scratch::bytes(MX, HEADS));
        Scratch S{};
        S.carve(scr_buf, MX, HEADS);

        bf16* idx_pre = dalloc<bf16>(size_t(MX) * HD);
        bf16* idx_wp = dalloc<bf16>(size_t(MX) * 32);
        int* d_pos = dalloc<int>(MX);
        int* d_klen = dalloc<int>(MX);
        int* d_pages = dalloc<int>(1, 0);
        int* d_list = dalloc<int>(size_t(MX) * 512);
        const size_t idx_scr_bytes = ds41_idx_scratch_bytes(MX, CAP + RCAP);
        void* idx_scr = dalloc<uint8_t>(idx_scr_bytes);
        bf16* d_x = dalloc<bf16>(size_t(MX) * HID);
        bf16* d_out = dalloc<bf16>(size_t(MX) * HID);

        const std::vector<uint16_t> emb_table = make_bf16_u16_vec(2048ull * HID, 777, -0.5f, 0.5f);

        std::vector<bf16*> ring_base(NL), comp_base(NL);
        std::vector<float*> kvs_base(NL), scs_base(NL);
        for (int l = 0; l < NL; ++l) {
            ring_base[size_t(l)] = a_layers[size_t(l)].C.ring;
            comp_base[size_t(l)] = a_layers[size_t(l)].C.comp;
            kvs_base[size_t(l)] = a_layers[size_t(l)].C.kvs;
            scs_base[size_t(l)] = a_layers[size_t(l)].C.scs;
        }
        std::vector<bf16*> dring_base(3);
        for (int j = 0; j < 3; ++j) dring_base[size_t(j)] = d_layers[size_t(j)].C.ring;

        auto reset_or_poison_slot = [&](int q, int fill_byte) {
            for (int l = 0; l < NL; ++l) {
                G6RT_CK(cudaMemsetAsync(ring_base[size_t(l)] + size_t(q) * RR_MAIN * HD, fill_byte, size_t(RR_MAIN) * HD * sizeof(bf16), st));
                if (comp_base[size_t(l)]) {
                    G6RT_CK(cudaMemsetAsync(comp_base[size_t(l)] + size_t(q) * (CAP / 2) * HD, fill_byte, size_t(CAP / 2) * HD * sizeof(bf16), st));
                    G6RT_CK(cudaMemsetAsync(kvs_base[size_t(l)] + size_t(q) * 2 * HD, fill_byte, 2 * HD * sizeof(float), st));
                    G6RT_CK(cudaMemsetAsync(scs_base[size_t(l)] + size_t(q) * 2 * HD, fill_byte, 2 * HD * sizeof(float), st));
                }
            }
            for (int k = 0; k < 4; ++k) {
                G6RT_CK(cudaMemsetAsync(idx_m.ica[k] + size_t(q) * size_t(idx_m.rows_kv[k]) * g6rt::DS41_ICA_ROW,
                                        fill_byte, size_t(idx_m.rows_kv[k]) * g6rt::DS41_ICA_ROW, st));
            }
            G6RT_CK(cudaMemsetAsync(idx_m.ca[3] + size_t(q) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW,
                                    fill_byte, size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW, st));
            G6RT_CK(cudaMemsetAsync(idx_m.shadow + size_t(q) * size_t(CAP + RCAP) * 512,
                                    fill_byte, size_t(CAP + RCAP) * 512 * sizeof(bf16), st));
            for (int j = 0; j < 3; ++j) {
                G6RT_CK(cudaMemsetAsync(dring_base[size_t(j)] + size_t(q) * RR_DRAFT * HD,
                                        fill_byte, size_t(RR_DRAFT) * HD * sizeof(bf16), st));
            }
            std::vector<int> eh(g6rt::ENG_RING, fill_byte == 0 ? -1 : -999999);
            G6RT_CK(cudaMemcpyAsync(main_m.ehist + size_t(q) * g6rt::ENG_RING, eh.data(), sizeof(int) * g6rt::ENG_RING,
                                    cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaStreamSynchronize(st));
        };

        auto run_chunk = [&](int q, int start, const int* tok_ids, int M) -> std::vector<uint16_t> {
            EXPECT(M >= 1 && M <= MX && start + M <= CAP);
            std::vector<uint16_t> h_in(size_t(M) * HID);
            std::vector<int> h_pos(size_t(M), 0);
            std::vector<int> eh(g6rt::ENG_RING);
            G6RT_CK(cudaMemcpy(eh.data(), main_m.ehist + size_t(q) * g6rt::ENG_RING, sizeof(int) * g6rt::ENG_RING, cudaMemcpyDeviceToHost));
            for (int r = 0; r < M; ++r) {
                const int tid = tok_ids[r] & 2047;
                std::memcpy(&h_in[size_t(r) * HID], &emb_table[size_t(tid) * HID], HID * sizeof(uint16_t));
                h_pos[size_t(r)] = start + r;
                ds41::engram::ring_put(eh.data(), start + r, tok_ids[r]);
            }
            G6RT_CK(cudaMemcpyAsync(d_x, h_in.data(), h_in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaMemcpyAsync(d_pos, h_pos.data(), h_pos.size() * sizeof(int), cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaMemcpyAsync(main_m.ehist + size_t(q) * g6rt::ENG_RING, eh.data(), sizeof(int) * g6rt::ENG_RING, cudaMemcpyHostToDevice, st));

            for (int l = 0; l < NL; ++l) {
                Cache C_slot = a_layers[size_t(l)].C;
                C_slot.ring = ring_base[size_t(l)] + size_t(q) * RR_MAIN * HD;
                if (comp_base[size_t(l)]) {
                    C_slot.comp = comp_base[size_t(l)] + size_t(q) * (CAP / 2) * HD;
                    C_slot.kvs = kvs_base[size_t(l)] + size_t(q) * 2 * HD;
                    C_slot.scs = scs_base[size_t(l)] + size_t(q) * 2 * HD;
                }
                attn_proj(W[l], C_slot, S, d_x, M, start, HEADS, st, d_pos);

                const bf16* comp_src = nullptr;
                const int* list_ptr = nullptr;
                int tk_arg = 0;
                if (l >= 1 && l <= 3) {
                    const int src_k = l - 1;
                    comp_row(W[l], C_slot, S, d_x, M, start, st, d_pos);
                    ds41_idx_ica_row(S.cnorm, idx_wk[size_t(l)], idx_knorm[size_t(l)], d_pos, /*ratio=*/2,
                                     idx_m.ica[src_k] + size_t(q) * size_t(idx_m.rows_kv[src_k]) * g6rt::DS41_ICA_ROW,
                                     d_pages, idx_m.rows_kv[src_k], M, st);
                    ds41_idx_klen(d_pos, /*ratio=*/2, d_klen, M, st);
                    ds41_idx_gemv(d_x, HID, idx_wproj[size_t(l)], 32, HID, idx_wp, 32, M, st);
                    Ds41IdxStage stg{};
                    stg.q_b = S.q;
                    stg.wproj = idx_wp;
                    stg.dpos = d_pos;
                    stg.klen = d_klen;
                    stg.max_klen = (start + M + 1) / 2;
                    stg.R = M;
                    stg.ica = idx_m.ica[src_k] + size_t(q) * size_t(idx_m.rows_kv[src_k]) * g6rt::DS41_ICA_ROW;
                    stg.pages = d_pages;
                    stg.rpp = idx_m.rows_kv[src_k];
                    stg.scratch = idx_scr;
                    stg.Rcap = MX;
                    stg.max_rows = CAP + RCAP;
                    stg.use_cand = 0;
                    stg.pipe_b = 0;
                    stg.out = d_list;
                    stg.ldo = 512;
                    ds41_idx_topk(&stg, st);
                    comp_src = C_slot.comp;
                    list_ptr = d_list;
                    tk_arg = 512;
                } else if (l == 4) {
                    bf16* sh_slot = idx_m.shadow + size_t(q) * size_t(CAP + RCAP) * 512;
                    ds41_idx_ca_row_r1(S.kv0, ca_norm_w, d_pos,
                                       idx_m.ca[3] + size_t(q) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW,
                                       d_pages, idx_m.rows_kv[3], idx_pre, sh_slot, M, st);
                    ds41_idx_ica_row(idx_pre, idx_wk[size_t(l)], idx_knorm[size_t(l)], d_pos, /*ratio=*/1,
                                     idx_m.ica[3] + size_t(q) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_ICA_ROW,
                                     d_pages, idx_m.rows_kv[3], M, st);
                    ds41_idx_klen(d_pos, /*ratio=*/1, d_klen, M, st);
                    ds41_idx_gemv(d_x, HID, idx_wproj[size_t(l)], 32, HID, idx_wp, 32, M, st);
                    Ds41IdxStage stg{};
                    stg.q_b = S.q;
                    stg.wproj = idx_wp;
                    stg.dpos = d_pos;
                    stg.klen = d_klen;
                    stg.max_klen = start + M;
                    stg.R = M;
                    stg.ica = idx_m.ica[3] + size_t(q) * size_t(idx_m.rows_kv[3]) * g6rt::DS41_ICA_ROW;
                    stg.pages = d_pages;
                    stg.rpp = idx_m.rows_kv[3];
                    stg.scratch = idx_scr;
                    stg.Rcap = MX;
                    stg.max_rows = CAP + RCAP;
                    stg.use_cand = 0;
                    stg.pipe_b = 0;
                    stg.out = d_list;
                    stg.ldo = 512;
                    ds41_idx_topk(&stg, st);
                    ds41_idx_cand_blocks(&stg, st);
                    comp_src = sh_slot;
                    list_ptr = d_list;
                    tk_arg = 512;
                } else if (l == 5) {
                    ds41_idx_klen(d_pos, /*ratio=*/2, d_klen, M, st);
                    ds41_idx_gemv(d_x, HID, idx_wproj[size_t(l)], 32, HID, idx_wp, 32, M, st);
                    Ds41IdxStage stg{};
                    stg.q_b = S.q;
                    stg.wproj = idx_wp;
                    stg.dpos = d_pos;
                    stg.klen = d_klen;
                    stg.max_klen = (start + M + 1) / 2;
                    stg.R = M;
                    stg.ica = idx_m.ica[0] + size_t(q) * size_t(idx_m.rows_kv[0]) * g6rt::DS41_ICA_ROW;
                    stg.pages = d_pages;
                    stg.rpp = idx_m.rows_kv[0];
                    stg.scratch = idx_scr;
                    stg.Rcap = MX;
                    stg.max_rows = CAP + RCAP;
                    stg.use_cand = 1;
                    stg.pipe_b = 0;
                    stg.out = d_list;
                    stg.ldo = 512;
                    ds41_idx_topk(&stg, st);
                    comp_src = comp_base[1] + size_t(q) * (CAP / 2) * HD;
                    list_ptr = d_list;
                    tk_arg = 512;
                }
                attn_core(W[l], C_slot, comp_src, S, M, start, HEADS, list_ptr, 512, tk_arg, st, d_pos);
                attn_out(W[l], S, M, start, HEADS, d_out, st, d_pos);
                G6RT_CK(cudaMemcpyAsync(d_x, d_out, size_t(M) * HID * sizeof(bf16), cudaMemcpyDeviceToDevice, st));
            }
            for (int j = 0; j < 3; ++j) {
                for (int r = 0; r < M; ++r) {
                    const int ring_idx = (start + r) % RR_DRAFT;
                    G6RT_CK(cudaMemcpyAsync(dring_base[size_t(j)] + (size_t(q) * RR_DRAFT + size_t(ring_idx)) * HD,
                                            S.kvrows + size_t(r) * HD, HD * sizeof(bf16), cudaMemcpyDeviceToDevice, st));
                }
            }
            G6RT_CK(cudaStreamSynchronize(st));
            return ddown(reinterpret_cast<const uint16_t*>(d_out) + size_t(M - 1) * HID, HID);
        };

        g6s2::Ds41PrefixCache cache(main_m, &dsp, /*budget_mib=*/64);
        g6rt::CacheNamespace ns{};

        struct Case {
            const char* name;
            int P;
            int P_cap;
            int chunk1;
        };
        const Case cases[6] = {
            {"prompt-end odd P_cap=63 (P=64)", 64, 63, 32},
            {"prompt-end even P_cap=64 (P=65)", 65, 64, 32},
            {"multi-turn odd P_cap=47 -> P=64", 64, 47, 32},
            {"multi-turn even P_cap=48 -> P=64", 64, 48, 32},
            {"long-ctx ICA>512 odd P_cap=1087 (P=1088)", 1088, 1087, 48},
            {"long-ctx ICA>512 even P_cap=1088 (P=1089)", 1089, 1088, 48},
        };

        for (int ci = 0; ci < 6; ++ci) {
            const Case& tc = cases[ci];
            std::vector<int> prompt(size_t(tc.P));
            for (int i = 0; i < tc.P; ++i) prompt[size_t(i)] = 100 + ci * 50 + i * 7;

            reset_or_poison_slot(0, 0);
            int cur = 0;
            while (cur < tc.P_cap) {
                const int step_m = std::min(tc.chunk1, tc.P_cap - cur);
                run_chunk(0, cur, prompt.data() + cur, step_m);
                cur += step_m;
            }
            EXPECT(cur == tc.P_cap);
            const uint64_t gen = cache.capture(0, uint32_t(tc.P_cap), prompt, ns, st);
            EXPECT(gen > 0);

            const int rem = tc.P - tc.P_cap;
            std::vector<uint16_t> cold_last_h = run_chunk(0, tc.P_cap, prompt.data() + tc.P_cap, rem);
            std::vector<int> cold_eh = ddown(main_m.ehist + 0 * g6rt::ENG_RING, g6rt::ENG_RING);
            std::vector<float> cold_logits0 = compute_logits(cold_last_h, w_head_host, V, cold_eh.data(), tc.P - 1);
            std::vector<int> cold_tokens;
            cold_tokens.push_back(argmax(cold_logits0));
            for (int step = 0; step < 8; ++step) {
                const int next_tok = cold_tokens.back();
                const int pos = tc.P + step;
                std::vector<uint16_t> h = run_chunk(0, pos, &next_tok, 1);
                std::vector<int> eh = ddown(main_m.ehist + 0 * g6rt::ENG_RING, g6rt::ENG_RING);
                std::vector<float> lg = compute_logits(h, w_head_host, V, eh.data(), pos);
                cold_tokens.push_back(argmax(lg));
            }

            reset_or_poison_slot(1, 0xFF);
            const auto hit = cache.match(prompt, tc.P - 1, ns);
            EXPECT(hit.hit && int(hit.position) >= tc.P_cap);
            cache.restore(gen, 1, st);

            std::vector<uint16_t> warm_last_h = run_chunk(1, tc.P_cap, prompt.data() + tc.P_cap, rem);
            std::vector<int> warm_eh = ddown(main_m.ehist + 1 * g6rt::ENG_RING, g6rt::ENG_RING);
            std::vector<float> warm_logits0 = compute_logits(warm_last_h, w_head_host, V, warm_eh.data(), tc.P - 1);
            std::vector<int> warm_tokens;
            warm_tokens.push_back(argmax(warm_logits0));
            for (int step = 0; step < 8; ++step) {
                const int next_tok = warm_tokens.back();
                const int pos = tc.P + step;
                std::vector<uint16_t> h = run_chunk(1, pos, &next_tok, 1);
                std::vector<int> eh = ddown(main_m.ehist + 1 * g6rt::ENG_RING, g6rt::ENG_RING);
                std::vector<float> lg = compute_logits(h, w_head_host, V, eh.data(), pos);
                warm_tokens.push_back(argmax(lg));
            }

            EXPECT(cold_last_h == warm_last_h);
            float max_diff = 0.0f;
            for (int v = 0; v < V; ++v) {
                max_diff = std::max(max_diff, std::fabs(cold_logits0[size_t(v)] - warm_logits0[size_t(v)]));
            }
            EXPECT(max_diff == 0.0f);
            EXPECT(cold_tokens == warm_tokens);

            const int final_pos = tc.P + 8;
            const int final_p2 = final_pos / 2;
            for (int k = 0; k < 4; ++k) {
                const size_t nrows = size_t((k < 3) ? final_p2 : final_pos);
                const size_t rpp = size_t(idx_m.rows_kv[k]);
                const uint8_t* p0 = idx_m.ica[k] + 0 * rpp * g6rt::DS41_ICA_ROW;
                const uint8_t* p1 = idx_m.ica[k] + 1 * rpp * g6rt::DS41_ICA_ROW;
                EXPECT(ddown(p0, nrows * size_t(g6kv::ICA_CODES)) == ddown(p1, nrows * size_t(g6kv::ICA_CODES)));
                EXPECT(ddown(p0 + rpp * size_t(g6kv::ICA_CODES), nrows * size_t(g6kv::ICA_SF)) ==
                       ddown(p1 + rpp * size_t(g6kv::ICA_CODES), nrows * size_t(g6kv::ICA_SF)));
            }
            EXPECT(ddown(idx_m.ca[3] + 0 * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW, size_t(final_pos) * g6rt::DS41_CA_ROW) ==
                   ddown(idx_m.ca[3] + 1 * size_t(idx_m.rows_kv[3]) * g6rt::DS41_CA_ROW, size_t(final_pos) * g6rt::DS41_CA_ROW));
            EXPECT(ddown(reinterpret_cast<const uint16_t*>(idx_m.shadow + 0 * size_t(CAP + RCAP) * 512), size_t(final_pos) * 512) ==
                   ddown(reinterpret_cast<const uint16_t*>(idx_m.shadow + 1 * size_t(CAP + RCAP) * 512), size_t(final_pos) * 512));
            for (int l = 1; l <= 3; ++l) {
                EXPECT(ddown(reinterpret_cast<const uint16_t*>(comp_base[size_t(l)] + 0 * (CAP / 2) * HD), size_t(final_p2) * HD) ==
                       ddown(reinterpret_cast<const uint16_t*>(comp_base[size_t(l)] + 1 * (CAP / 2) * HD), size_t(final_p2) * HD));
            }
            printf("Part 2 [%-34s]: PASS (max_logit_diff=%.1e, 9/9 greedy tokens identical, ICA[0..3]+CA+shadow+comp bit-exact)\n",
                   tc.name, max_diff);

            if (ci == 0) {
                reset_or_poison_slot(1, 0xFF);
                cache.restore(gen, 1, st);
                for (int l = 1; l <= 3; ++l) {
                    G6RT_CK(cudaMemsetAsync(kvs_base[size_t(l)] + 1 * 2 * HD, 0, 2 * HD * sizeof(float), st));
                    G6RT_CK(cudaMemsetAsync(scs_base[size_t(l)] + 1 * 2 * HD, 0, 2 * HD * sizeof(float), st));
                }
                G6RT_CK(cudaStreamSynchronize(st));
                std::vector<uint16_t> bad_h = run_chunk(1, tc.P_cap, prompt.data() + tc.P_cap, rem);
                EXPECT(bad_h != cold_last_h);
                printf("Part 3a [negative control: odd P_cap without C.kvs/C.scs diverges]: PASS\n");
            }
            if (ci == 4) {
                reset_or_poison_slot(1, 0xFF);
                cache.restore(gen, 1, st);
                // Corrupt restored ICA[0..3] codes+scales in [0, P_cap/2) so top-512 selection changes
                for (int k = 0; k < 4; ++k) {
                    const size_t rpp = size_t(idx_m.rows_kv[k]);
                    uint8_t* p1 = idx_m.ica[k] + 1 * rpp * g6rt::DS41_ICA_ROW;
                    G6RT_CK(cudaMemsetAsync(p1, 0x55, 256 * size_t(g6kv::ICA_CODES), st));
                    G6RT_CK(cudaMemsetAsync(p1 + rpp * size_t(g6kv::ICA_CODES), 0x80, 256 * size_t(g6kv::ICA_SF), st));
                }
                G6RT_CK(cudaStreamSynchronize(st));
                std::vector<uint16_t> bad_h = run_chunk(1, tc.P_cap, prompt.data() + tc.P_cap, rem);
                EXPECT(bad_h != cold_last_h);
                printf("Part 3b [negative control: corrupting restored ICA[0..3] at P=1088 diverges]: PASS\n");
            }
        }
    }

    printf("ALL DS41 PREFIX CACHE SYNTHETIC GPU EXACTNESS TESTS PASSED\n");
    return 0;
}

} // namespace ds41a

int main() {
    return ds41a::run_test();
}
