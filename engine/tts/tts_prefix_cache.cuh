// SHA-256 GPU Prefix State Cache for Drama 3 (g6tts Wave 3.6 Pass 2).
//
// Caches the exact GPU prefill state after rows `0..P-1`:
//   - Backbone FullAttn KV caches `0..P-1` (10 layers, BF16)
//   - Backbone GDN recurrent state `S` & `conv` at position `P-1` (30 layers, FP32)
//   - Tower KV caches `0..P-2` (13 layers, BF16)
//   - Backbone final hidden state `bb_hid[P-1]` (BF16, `[H]`) so partial hits (`P < T`) can
//     bridge tower position `P-1` with `(bb_hid[P-1], rows[P-1], rows[P])` bit-identically.
//
// All entries are lazily allocated on first save (LRU eviction, default max 8 entries).
// Restores & saves use async D2D copies on the engine CUDA stream with zero host syncs.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tts_backbone.cuh"
#include "tts_config.h"
#include "tts_tower.cuh"

namespace g6tts {

using Sha256Digest = std::array<uint8_t, 32>;

// Minimal self-contained FIPS 180-4 SHA-256 for prefix row hashing (zero external dependencies).
inline Sha256Digest sha256_bytes(const uint8_t* data, size_t len) {
    static constexpr uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t { return (x >> n) | (x << (32u - n)); };
    auto compress = [&](const uint8_t blk[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(blk[i * 4 + 0]) << 24) | (uint32_t(blk[i * 4 + 1]) << 16) |
                   (uint32_t(blk[i * 4 + 2]) << 8) | uint32_t(blk[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };

    size_t pos = 0;
    while (pos + 64 <= len) {
        compress(data + pos);
        pos += 64;
    }
    uint8_t tail[128] = {};
    const size_t rem = len - pos;
    if (rem > 0) std::memcpy(tail, data + pos, rem);
    tail[rem] = 0x80u;
    const size_t pad_len = (rem < 56) ? 64 : 128;
    const uint64_t bit_len = uint64_t(len) * 8ull;
    for (int i = 0; i < 8; ++i) {
        tail[pad_len - 1 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xffu);
    }
    compress(tail);
    if (pad_len == 128) compress(tail + 64);

    Sha256Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[size_t(i * 4 + 0)] = static_cast<uint8_t>((h[i] >> 24) & 0xffu);
        out[size_t(i * 4 + 1)] = static_cast<uint8_t>((h[i] >> 16) & 0xffu);
        out[size_t(i * 4 + 2)] = static_cast<uint8_t>((h[i] >> 8) & 0xffu);
        out[size_t(i * 4 + 3)] = static_cast<uint8_t>(h[i] & 0xffu);
    }
    return out;
}

inline Sha256Digest hash_prefix_rows(const int32_t* rows, int num_rows, int rw) {
    return sha256_bytes(reinterpret_cast<const uint8_t*>(rows), size_t(num_rows) * size_t(rw) * sizeof(int32_t));
}

struct PrefixCacheEntry {
    bool valid = false;
    Sha256Digest key{};
    int prefix_len = 0;
    int kv_cap = 0;
    uint64_t lru_tick = 0;

    std::vector<__nv_bfloat16*> bb_k; // [n_bb_attn], each [kv_cap, bb_kvw]
    std::vector<__nv_bfloat16*> bb_v; // [n_bb_attn]
    std::vector<float*> gdn_S;        // [n_gdn], each [gdn_vh * gdn_kd * gdn_vd]
    std::vector<float*> gdn_conv;     // [n_gdn], each [(gdn_conv - 1) * qkv_dim]
    std::vector<__nv_bfloat16*> tw_k; // [n_tw], each [kv_cap, tw_kvw]
    std::vector<__nv_bfloat16*> tw_v; // [n_tw]
    __nv_bfloat16* last_hid = nullptr; // [H]

    void release() {
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        for (auto*& p : bb_k) freep(p);
        for (auto*& p : bb_v) freep(p);
        for (auto*& p : gdn_S) freep(p);
        for (auto*& p : gdn_conv) freep(p);
        for (auto*& p : tw_k) freep(p);
        for (auto*& p : tw_v) freep(p);
        bb_k.clear(); bb_v.clear();
        gdn_S.clear(); gdn_conv.clear();
        tw_k.clear(); tw_v.clear();
        freep(last_hid);
        valid = false;
        prefix_len = 0;
        kv_cap = 0;
    }
};

struct TtsPrefixCache {
    static constexpr int MIN_PREFIX_LEN = 4;
    static constexpr size_t MAX_RECENT_PROMPTS = 16;

    TtsConfig cfg;
    int max_entries = 8;
    std::vector<int> bb_attn_layers;
    std::vector<int> gdn_layers;
    int n_tw_layers = 0;

    std::vector<PrefixCacheEntry> entries;
    std::vector<std::vector<int32_t>> recent_prompts;
    size_t recent_wr = 0;
    uint64_t tick = 0;
    uint64_t hits_full = 0;
    uint64_t hits_partial = 0;
    uint64_t misses = 0;

    ~TtsPrefixCache() { release(); }

    void release() {
        for (auto& e : entries) e.release();
        entries.clear();
        recent_prompts.clear();
        recent_wr = 0;
        tick = 0;
    }

    void init(const TtsConfig& c, int max_ent = 8) {
        release();
        cfg = c;
        max_entries = std::max(0, max_ent);
        bb_attn_layers.clear();
        gdn_layers.clear();
        for (int l = 0; l < c.num_layers; ++l) {
            if (c.is_full_attn[size_t(l)]) bb_attn_layers.push_back(l);
            else gdn_layers.push_back(l);
        }
        n_tw_layers = c.tower_shared + c.tower_res;
        entries.resize(static_cast<size_t>(max_entries));
    }

    static int next_kv_cap(int need, int max_cap) {
        int c = 64;
        while (c < need && c < max_cap) c <<= 1;
        return std::min(c, max_cap);
    }

    void ensure_entry_alloc(PrefixCacheEntry& e, int need_len, int max_cap) {
        const size_t n_att = bb_attn_layers.size();
        const size_t n_gdn = gdn_layers.size();
        const size_t n_tw = static_cast<size_t>(n_tw_layers);
        const int qkv_dim = (2 * cfg.gdn_kh + cfg.gdn_vh) * cfg.gdn_kd;
        const size_t s_bytes = size_t(cfg.gdn_vh) * cfg.gdn_kd * cfg.gdn_vd * sizeof(float);
        const size_t cv_bytes = size_t(cfg.gdn_conv - 1) * qkv_dim * sizeof(float);

        if (!e.last_hid) {
            TTS_CK(cudaMalloc(&e.last_hid, size_t(cfg.hidden_size) * sizeof(__nv_bfloat16)));
            e.gdn_S.assign(n_gdn, nullptr);
            e.gdn_conv.assign(n_gdn, nullptr);
            for (size_t i = 0; i < n_gdn; ++i) {
                TTS_CK(cudaMalloc(&e.gdn_S[i], s_bytes));
                TTS_CK(cudaMalloc(&e.gdn_conv[i], cv_bytes));
            }
            e.bb_k.assign(n_att, nullptr);
            e.bb_v.assign(n_att, nullptr);
            e.tw_k.assign(n_tw, nullptr);
            e.tw_v.assign(n_tw, nullptr);
        }

        if (e.kv_cap < need_len) {
            const int new_cap = next_kv_cap(need_len, max_cap);
            const size_t bb_kv_b = size_t(new_cap) * cfg.num_kv_heads * cfg.head_dim * sizeof(__nv_bfloat16);
            const size_t tw_kv_b = size_t(new_cap) * cfg.tower_kv_heads * cfg.tower_head_dim * sizeof(__nv_bfloat16);
            for (size_t i = 0; i < n_att; ++i) {
                if (e.bb_k[i]) cudaFree(e.bb_k[i]);
                if (e.bb_v[i]) cudaFree(e.bb_v[i]);
                TTS_CK(cudaMalloc(&e.bb_k[i], bb_kv_b));
                TTS_CK(cudaMalloc(&e.bb_v[i], bb_kv_b));
            }
            for (size_t i = 0; i < n_tw; ++i) {
                if (e.tw_k[i]) cudaFree(e.tw_k[i]);
                if (e.tw_v[i]) cudaFree(e.tw_v[i]);
                TTS_CK(cudaMalloc(&e.tw_k[i], tw_kv_b));
                TTS_CK(cudaMalloc(&e.tw_v[i], tw_kv_b));
            }
            e.kv_cap = new_cap;
        }
    }

    // Finds the longest cached prefix `P <= T` matching `rows[0..P-1]`. Returns entry index or -1.
    int find_longest_prefix(const int32_t* rows, int T, int rw, int& out_prefix_len) {
        out_prefix_len = 0;
        if (max_entries <= 0 || T < MIN_PREFIX_LEN) return -1;

        // Collect distinct candidate lengths <= T sorted descending so first hit is longest.
        std::vector<int> cand_lens;
        cand_lens.reserve(entries.size());
        for (const auto& e : entries) {
            if (!e.valid || e.prefix_len > T) continue;
            if (std::find(cand_lens.begin(), cand_lens.end(), e.prefix_len) == cand_lens.end()) {
                cand_lens.push_back(e.prefix_len);
            }
        }
        if (cand_lens.empty()) return -1;
        std::sort(cand_lens.begin(), cand_lens.end(), std::greater<int>());

        for (int len : cand_lens) {
            const Sha256Digest d = hash_prefix_rows(rows, len, rw);
            for (size_t i = 0; i < entries.size(); ++i) {
                auto& e = entries[i];
                if (e.valid && e.prefix_len == len && e.key == d) {
                    e.lru_tick = ++tick;
                    out_prefix_len = len;
                    if (len == T) ++hits_full;
                    else ++hits_partial;
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }

    // Finds longest common row prefix `4 <= L < T` against recent prompts so shared voice/system
    // prefixes are automatically discovered and cached on the second request.
    int find_recent_common_prefix(const int32_t* rows, int T, int rw) const {
        if (max_entries <= 0 || T <= MIN_PREFIX_LEN) return 0;
        int best = 0;
        for (const auto& prev : recent_prompts) {
            const int prev_T = static_cast<int>(prev.size()) / rw;
            const int lim = std::min(T - 1, prev_T);
            int lcp = 0;
            while (lcp < lim &&
                   std::memcmp(rows + size_t(lcp) * rw, prev.data() + size_t(lcp) * rw,
                               size_t(rw) * sizeof(int32_t)) == 0) {
                ++lcp;
            }
            if (lcp > best) best = lcp;
        }
        return (best >= MIN_PREFIX_LEN) ? best : 0;
    }

    void record_prompt(const int32_t* rows, int T, int rw) {
        if (max_entries <= 0 || T < MIN_PREFIX_LEN) return;
        const int keep_T = std::min(T, 2048);
        std::vector<int32_t> copy(rows, rows + size_t(keep_T) * rw);
        if (recent_prompts.size() < MAX_RECENT_PROMPTS) {
            recent_prompts.push_back(std::move(copy));
        } else {
            recent_prompts[recent_wr % MAX_RECENT_PROMPTS] = std::move(copy);
        }
        ++recent_wr;
    }

    // Restores cached state for `0..P-1` into physical `slot` (async D2D, zero host sync).
    void restore_to_slot(int entry_idx, int slot, BackboneRunner& bb, TowerRunner& tower,
                         __nv_bfloat16* d_last_hid_out, cudaStream_t stream) const {
        const auto& e = entries[size_t(entry_idx)];
        const int P = e.prefix_len;
        const size_t bb_kvw = size_t(cfg.num_kv_heads) * cfg.head_dim;
        const size_t tw_kvw = size_t(cfg.tower_kv_heads) * cfg.tower_head_dim;
        const size_t bb_kv_b = size_t(P) * bb_kvw * sizeof(__nv_bfloat16);
        const size_t tw_kv_b = size_t(std::max(0, P - 1)) * tw_kvw * sizeof(__nv_bfloat16);
        const int qkv_dim = (2 * cfg.gdn_kh + cfg.gdn_vh) * cfg.gdn_kd;
        const size_t s_elems = size_t(cfg.gdn_vh) * cfg.gdn_kd * cfg.gdn_vd;
        const size_t cv_elems = size_t(cfg.gdn_conv - 1) * qkv_dim;

        for (size_t i = 0; i < bb_attn_layers.size(); ++i) {
            const int l = bb_attn_layers[i];
            const size_t off = size_t(slot) * bb.cap * bb_kvw;
            TTS_CK(cudaMemcpyAsync(bb.k_caches[size_t(l)] + off, e.bb_k[i], bb_kv_b, cudaMemcpyDeviceToDevice, stream));
            TTS_CK(cudaMemcpyAsync(bb.v_caches[size_t(l)] + off, e.bb_v[i], bb_kv_b, cudaMemcpyDeviceToDevice, stream));
        }
        for (size_t i = 0; i < gdn_layers.size(); ++i) {
            const int l = gdn_layers[i];
            TTS_CK(cudaMemcpyAsync(bb.gdn_S[size_t(l)] + size_t(slot) * s_elems, e.gdn_S[i],
                                   s_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            TTS_CK(cudaMemcpyAsync(bb.gdn_conv[size_t(l)] + size_t(slot) * cv_elems, e.gdn_conv[i],
                                   cv_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        if (tw_kv_b > 0) {
            for (int i = 0; i < n_tw_layers; ++i) {
                const size_t off = size_t(slot) * tower.cap * tw_kvw;
                TTS_CK(cudaMemcpyAsync(tower.k_caches[size_t(i)] + off, e.tw_k[size_t(i)], tw_kv_b,
                                       cudaMemcpyDeviceToDevice, stream));
                TTS_CK(cudaMemcpyAsync(tower.v_caches[size_t(i)] + off, e.tw_v[size_t(i)], tw_kv_b,
                                       cudaMemcpyDeviceToDevice, stream));
            }
        }
        TTS_CK(cudaMemcpyAsync(d_last_hid_out, e.last_hid,
                               size_t(cfg.hidden_size) * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
    }

    // Saves current GPU state of `slot` after rows `0..P-1` into the LRU cache (async D2D).
    void save_from_slot(const int32_t* rows, int P, int rw, int slot,
                        const BackboneRunner& bb, const TowerRunner& tower,
                        const __nv_bfloat16* d_last_hid_in, cudaStream_t stream) {
        if (max_entries <= 0 || P < MIN_PREFIX_LEN || P > bb.cap) return;
        const Sha256Digest d = hash_prefix_rows(rows, P, rw);

        // Already cached? Just refresh LRU timestamp.
        for (auto& e : entries) {
            if (e.valid && e.prefix_len == P && e.key == d) {
                e.lru_tick = ++tick;
                return;
            }
        }

        // Select first unused entry or LRU victim.
        size_t victim = 0;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (!entries[i].valid) {
                victim = i;
                break;
            }
            if (entries[i].lru_tick < oldest) {
                oldest = entries[i].lru_tick;
                victim = i;
            }
        }

        auto& e = entries[victim];
        ensure_entry_alloc(e, P, bb.cap);

        const size_t bb_kvw = size_t(cfg.num_kv_heads) * cfg.head_dim;
        const size_t tw_kvw = size_t(cfg.tower_kv_heads) * cfg.tower_head_dim;
        const size_t bb_kv_b = size_t(P) * bb_kvw * sizeof(__nv_bfloat16);
        const size_t tw_kv_b = size_t(std::max(0, P - 1)) * tw_kvw * sizeof(__nv_bfloat16);
        const int qkv_dim = (2 * cfg.gdn_kh + cfg.gdn_vh) * cfg.gdn_kd;
        const size_t s_elems = size_t(cfg.gdn_vh) * cfg.gdn_kd * cfg.gdn_vd;
        const size_t cv_elems = size_t(cfg.gdn_conv - 1) * qkv_dim;

        for (size_t i = 0; i < bb_attn_layers.size(); ++i) {
            const int l = bb_attn_layers[i];
            const size_t off = size_t(slot) * bb.cap * bb_kvw;
            TTS_CK(cudaMemcpyAsync(e.bb_k[i], bb.k_caches[size_t(l)] + off, bb_kv_b, cudaMemcpyDeviceToDevice, stream));
            TTS_CK(cudaMemcpyAsync(e.bb_v[i], bb.v_caches[size_t(l)] + off, bb_kv_b, cudaMemcpyDeviceToDevice, stream));
        }
        for (size_t i = 0; i < gdn_layers.size(); ++i) {
            const int l = gdn_layers[i];
            TTS_CK(cudaMemcpyAsync(e.gdn_S[i], bb.gdn_S[size_t(l)] + size_t(slot) * s_elems,
                                   s_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            TTS_CK(cudaMemcpyAsync(e.gdn_conv[i], bb.gdn_conv[size_t(l)] + size_t(slot) * cv_elems,
                                   cv_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        if (tw_kv_b > 0) {
            for (int i = 0; i < n_tw_layers; ++i) {
                const size_t off = size_t(slot) * tower.cap * tw_kvw;
                TTS_CK(cudaMemcpyAsync(e.tw_k[size_t(i)], tower.k_caches[size_t(i)] + off, tw_kv_b,
                                       cudaMemcpyDeviceToDevice, stream));
                TTS_CK(cudaMemcpyAsync(e.tw_v[size_t(i)], tower.v_caches[size_t(i)] + off, tw_kv_b,
                                       cudaMemcpyDeviceToDevice, stream));
            }
        }
        TTS_CK(cudaMemcpyAsync(e.last_hid, d_last_hid_in,
                               size_t(cfg.hidden_size) * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));

        e.valid = true;
        e.key = d;
        e.prefix_len = P;
        e.lru_tick = ++tick;
    }
};

} // namespace g6tts
