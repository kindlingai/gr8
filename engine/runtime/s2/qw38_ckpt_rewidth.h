// GR8_KV_REWIDTH for Qwen3.8-Flash-Next (`./gr8 qw38 scale`): rewrites a persisted aligned checkpoint store from an
// old TP width (`old_nr in 1..4`) to a new TP width (`nr in 1..4`) for `rank` over `kvman-peer`. Host only.
//
// What changes across TP widths in Qwen3.8:
//   - Full-attention KV pages (12 layers, 2 KV heads, d=256 BF16):
//       At TP=1 rank 0 holds KV heads [0, 1]; at TP=2 rank 0 holds [0], rank 1 holds [1];
//       at TP=3 ranks hold [0], [0,1], [1]; at TP=4 ranks hold [0], [0], [1], [1].
//       Per 64-token page: [2 (K,V)][64][nh_kv][256] BF16 -> re-sliced per KV head from old owners!
//   - QSA compressed K pages (12 layers, 1 KV head, 16 blocks x 128 BF16 = 4096 B/page): replicated, copied verbatim.
//   - Gated DeltaNet recurrent state (36 layers):
//       S[nh_v][128][128] FP32 (V heads [v0..v1)) and conv1d[3][(2*nh_k + nh_v)*128] BF16
//       (Q heads [k0..k1), K heads [k0..k1), V heads [v0..v1)): re-sharded head-by-head from old owners!
//   - PLE conv1d history (layer 1, [3][10240] BF16): replicated, copied verbatim.
#pragma once
#include "flash_ckpt_index.h"
#include "flash_ckpt_rewidth.h"
#include "../../model_geom_qw38.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace g6s2 {

struct Qw38CkptLayout {
    int nr = 1, rank = 0;
    int n_full = geom::N_FULL;       // 12 (or 13 with MTP)
    int n_gdn = geom::N_GDN;         // 36
    size_t slot = 4096;

    int nh_kv() const { return geom::nh_kv(nr, rank); }
    int kv0() const { return geom::kv0(nr, rank); }
    int nh_v() const { return geom::gdn_vh(nr); }
    int v0() const { return geom::gdn_v0(nr, rank); }
    int nh_k() const { return geom::gdn_kh(nr, rank); }
    int k0() const { return geom::gdn_k0(nr, rank); }
    int qkv_local() const { return geom::gdn_qkv_local(nr, rank); }

    // Per full-attention layer page bytes: KV [2][64][nh_kv][256] BF16 + QSA [16][128] BF16
    size_t kv_layer_page_bytes() const { return size_t(2) * 64 * size_t(nh_kv()) * geom::HEAD_DIM * 2; }
    static constexpr size_t idx_layer_page_bytes() { return size_t(16) * geom::IDX_DIM * 2; } // 4096 B
    size_t page_bytes() const { return size_t(n_full) * (kv_layer_page_bytes() + idx_layer_page_bytes()); }
    uint64_t page_slot_blocks() const { return (page_bytes() + slot - 1) / slot; }

    // Per GDN layer state bytes: S FP32 + conv BF16
    size_t gdn_s_bytes() const { return size_t(nh_v()) * geom::GDN_DK * geom::GDN_DV * 4; }
    size_t gdn_c_bytes() const { return size_t(geom::GDN_NHIST) * size_t(qkv_local()) * 2; }
    static constexpr size_t ple_conv_bytes() { return size_t(geom::PLE_NHIST) * geom::HC_D * 2 + size_t(geom::PLE_NGRAM_CTX) * 4; } // 184,328 B
    size_t state_bytes() const {
        return size_t(n_gdn) * (gdn_s_bytes() + gdn_c_bytes()) + ple_conv_bytes();
    }
    uint64_t state_blocks() const { return (state_bytes() + slot - 1) / slot; }
    uint64_t entry_blocks(uint32_t pos) const {
        const uint64_t np = (uint64_t(pos) + 63) / 64;
        return 1 + np * page_slot_blocks() + state_blocks();
    }
};

// First old rank in 0..old_nr that owns global full-attention KV head `gkv`
inline int qw38_old_kv_owner(int old_nr, int gkv, int& local_kv) {
    for (int q = 0; q < old_nr; ++q) {
        const int a = geom::kv0(old_nr, q), b = geom::kv1(old_nr, q);
        if (gkv >= a && gkv < b) { local_kv = gkv - a; return q; }
    }
    return -1;
}
// First old rank in 0..old_nr that owns global GDN V head `gv`
inline int qw38_old_v_owner(int old_nr, int gv, int& local_v) {
    for (int q = 0; q < old_nr; ++q) {
        const int a = geom::gdn_v0(old_nr, q), b = a + geom::gdn_vh(old_nr);
        if (gv >= a && gv < b) { local_v = gv - a; return q; }
    }
    return -1;
}
// First old rank in 0..old_nr that owns global GDN K/Q head `gk`
inline int qw38_old_k_owner(int old_nr, int gk, int& local_k) {
    for (int q = 0; q < old_nr; ++q) {
        const int a = geom::gdn_k0(old_nr, q), b = geom::gdn_k1(old_nr, q);
        if (gk >= a && gk < b) { local_k = gk - a; return q; }
    }
    return -1;
}

// Rewrites dir's persisted Qwen3.8 checkpoint store from old_nr -> nr for `rank`.
// Uses the same `.prev` convention and `kvman-peer` protocol as `flash_ckpt_rewidth.h`.
inline int64_t qw38_rewidth(const std::string& dir, const std::string& spec_text, const std::string& compat,
                            const std::function<std::string(int, int, size_t, size_t, const std::string&)>& old_compat,
                            const Qw38CkptLayout& L, uint64_t store_blocks, uint32_t grid, uint64_t pool_pages,
                            std::string& why) {
    FlashRewidthSpec spec;
    if (const char* e = flash_rewidth_spec(spec_text, spec)) { why = e; return -1; }
    if (!geom::rank_count_ok(spec.old_nr) || !geom::rank_count_ok(L.nr)) { why = "unsupported TP width"; return -1; }
    const std::string out_blocks = dir + "/flash-ckpt", out_index = dir + "/flash-ckpt.index", out_tokens = dir + "/flash-ckpt.tokens";
    auto cleanup = [&] {
        std::error_code ec;
        std::filesystem::remove(out_blocks, ec);
        std::filesystem::remove(out_index, ec);
        std::filesystem::remove_all(out_tokens, ec);
    };
    try {
        std::string token;
        {
            std::ifstream t(spec.token_file);
            if (!t || !std::getline(t, token) || token.empty()) frw::fail("no token in " + spec.token_file);
            while (!token.empty() && std::isspace((unsigned char)token.back())) token.pop_back();
        }
        std::vector<Qw38CkptLayout> oldL(size_t(spec.old_nr));
        std::vector<uint64_t> old_compats(size_t(spec.old_nr));
        for (int q = 0; q < spec.old_nr; ++q) {
            oldL[size_t(q)] = Qw38CkptLayout{spec.old_nr, q, L.n_full, L.n_gdn, L.slot};
            const std::string c = old_compat(spec.old_nr, q, oldL[size_t(q)].page_bytes(), oldL[size_t(q)].state_bytes(), spec.old_set);
            old_compats[size_t(q)] = fnv64(c.data(), c.size());
        }
        frw::Src src(spec, dir, token);
        const int from = spec.old_rank >= 0 ? spec.old_rank : 0;
        bool missing = false;
        const std::string ib = src.read_all(from, "flash-ckpt.index.prev", &missing);
        if (missing) { why = "no flash-ckpt.index.prev"; return 0; }
        FlashIndexImage img;
        if (const char* e = flash_index_decode(ib, img)) { why = std::string("old index: ") + e; return 0; }
        if (img.compat != old_compats[size_t(from)]) { why = "another engine or layout"; return 0; }

        // Read every old rank's index so we know each old rank's entry `base`
        std::vector<FlashIndexImage> old_imgs(size_t(spec.old_nr));
        old_imgs[size_t(from)] = img;
        for (int q = 0; q < spec.old_nr; ++q) {
            if (q == from) continue;
            const std::string qb = src.read_all(q, "flash-ckpt.index.prev");
            if (const char* e = flash_index_decode(qb, old_imgs[size_t(q)])) frw::fail(std::string("peer ") + std::to_string(q) + " index: " + e);
            if (old_imgs[size_t(q)].compat != old_compats[size_t(q)]) frw::fail("peer " + std::to_string(q) + " compat differs");
        }
        auto find_old = [&](int q, uint64_t gen) -> const FlashEntry* {
            for (const auto& e : old_imgs[size_t(q)].entries)
                if (e.generation == gen) return &e;
            return nullptr;
        };

        std::vector<FlashEntry> want;
        for (const auto& x : img.entries) {
            if (!x.terminal && grid && x.position % grid == 0 && (x.position + 63) / 64 <= pool_pages &&
                x.manifest.page_bytes == oldL[size_t(from)].page_bytes() &&
                x.manifest.state_bytes == oldL[size_t(from)].state_bytes())
                want.push_back(x);
        }
        std::sort(want.begin(), want.end(), [](const FlashEntry& a, const FlashEntry& b) { return a.used > b.used; });
        uint64_t at = 0;
        size_t fit = 0;
        for (; fit < want.size(); ++fit) {
            const uint64_t need = L.entry_blocks(want[fit].position);
            if (at + need > store_blocks) break;
            at += need;
        }
        want.resize(fit);

        const int fd = ::open(out_blocks.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) frw::fail("open " + out_blocks + ": " + std::strerror(errno));
        struct Close { int fd; ~Close() { ::close(fd); } } closer{fd};

        std::vector<FlashEntry> kept;
        const size_t new_page_slot_bytes = size_t(L.page_slot_blocks() * L.slot);
        std::vector<uint8_t> new_page(new_page_slot_bytes, 0);
        std::vector<uint8_t> new_state(size_t(L.state_blocks() * L.slot), 0);
        std::vector<std::vector<uint8_t>> old_pages(size_t(spec.old_nr));
        std::vector<std::vector<uint8_t>> old_states(size_t(spec.old_nr));
        at = 0;

        for (const auto& x : want) {
            bool ok = true;
            std::vector<const FlashEntry*> peers_e(size_t(spec.old_nr), nullptr);
            for (int q = 0; q < spec.old_nr; ++q) {
                peers_e[size_t(q)] = find_old(q, x.generation);
                if (!peers_e[size_t(q)] || peers_e[size_t(q)]->position != x.position) { ok = false; break; }
            }
            if (!ok) continue;

            const uint64_t np = (uint64_t(x.position) + 63) / 64;
            // Read and verify every old rank's state stream
            for (int q = 0; q < spec.old_nr; ++q) {
                const auto& oL = oldL[size_t(q)];
                old_states[size_t(q)].resize(oL.state_bytes());
                const uint64_t st_off = (peers_e[size_t(q)]->base + 1 + np * oL.page_slot_blocks()) * L.slot;
                src.read(q, "flash-ckpt.prev", st_off, oL.state_bytes(), old_states[size_t(q)].data());
                if (ckpt_sum_host(old_states[size_t(q)].data(), oL.state_bytes()) != peers_e[size_t(q)]->manifest.state_sum) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            // Rebuild each 64-token page: slice KV heads [kv0..kv0+nh_kv) from old owners; copy QSA verbatim
            uint64_t new_page_sum = 0;
            for (uint64_t p = 0; p < np; ++p) {
                for (int q = 0; q < spec.old_nr; ++q) {
                    const auto& oL = oldL[size_t(q)];
                    old_pages[size_t(q)].resize(oL.page_bytes());
                    const uint64_t po = (peers_e[size_t(q)]->base + 1 + p * oL.page_slot_blocks()) * L.slot;
                    src.read(q, "flash-ckpt.prev", po, oL.page_bytes(), old_pages[size_t(q)].data());
                }
                std::memset(new_page.data(), 0, new_page.size());
                uint8_t* dst_layer = new_page.data();
                const size_t head_bytes = size_t(geom::HEAD_DIM) * 2; // 512 B per KV head per token
                for (int l = 0; l < L.n_full; ++l) {
                    const size_t new_kv_b = L.kv_layer_page_bytes();
                    const size_t idx_b = Qw38CkptLayout::idx_layer_page_bytes();
                    // Re-shard KV: [2 (K,V)][64 tokens][nh_kv][256 BF16]
                    for (int kv_l = 0; kv_l < L.nh_kv(); ++kv_l) {
                        const int gkv = L.kv0() + kv_l;
                        int olk = 0;
                        const int oq = qw38_old_kv_owner(spec.old_nr, gkv, olk);
                        const auto& oL = oldL[size_t(oq)];
                        const uint8_t* src_layer = old_pages[size_t(oq)].data() + size_t(l) * (oL.kv_layer_page_bytes() + idx_b);
                        for (int kv_plane = 0; kv_plane < 2; ++kv_plane) {
                            for (int tok = 0; tok < 64; ++tok) {
                                const size_t s_off = ((size_t(kv_plane) * 64 + size_t(tok)) * size_t(oL.nh_kv()) + size_t(olk)) * head_bytes;
                                const size_t d_off = ((size_t(kv_plane) * 64 + size_t(tok)) * size_t(L.nh_kv()) + size_t(kv_l)) * head_bytes;
                                std::memcpy(dst_layer + d_off, src_layer + s_off, head_bytes);
                            }
                        }
                    }
                    // Copy QSA compressed K verbatim from `from`
                    const auto& fL = oldL[size_t(from)];
                    const uint8_t* src_idx = old_pages[size_t(from)].data() + size_t(l) * (fL.kv_layer_page_bytes() + idx_b) + fL.kv_layer_page_bytes();
                    std::memcpy(dst_layer + new_kv_b, src_idx, idx_b);
                    dst_layer += new_kv_b + idx_b;
                }
                new_page_sum += ckpt_sum_host(new_page.data(), L.page_bytes(), p * (L.page_bytes() / 8));
                frw::pwrite_all(fd, new_page.data(), new_page_slot_bytes, (at + 1 + p * L.page_slot_blocks()) * L.slot, out_blocks);
            }

            // Rebuild GDN recurrent states + conv rings + PLE conv ring
            std::memset(new_state.data(), 0, new_state.size());
            const size_t dk_dv_b = size_t(geom::GDN_DK) * geom::GDN_DV * 4;
            const size_t head_bf16_b = size_t(geom::GDN_DK) * 2;
            for (int li = 0; li < L.n_gdn; ++li) {
                uint8_t* dst_s = new_state.data() + size_t(li) * (L.gdn_s_bytes() + L.gdn_c_bytes());
                uint8_t* dst_c = dst_s + L.gdn_s_bytes();
                // 1. S [nh_v][128][128] FP32
                for (int lv = 0; lv < L.nh_v(); ++lv) {
                    int olv = 0;
                    const int oq = qw38_old_v_owner(spec.old_nr, L.v0() + lv, olv);
                    const auto& oL = oldL[size_t(oq)];
                    const uint8_t* src_s = old_states[size_t(oq)].data() + size_t(li) * (oL.gdn_s_bytes() + oL.gdn_c_bytes());
                    std::memcpy(dst_s + size_t(lv) * dk_dv_b, src_s + size_t(olv) * dk_dv_b, dk_dv_b);
                }
                // 2. conv [3][(2*nh_k + nh_v)*128] BF16: Q [nh_k], K [nh_k], V [nh_v] per history tap
                const size_t new_tap_b = size_t(L.qkv_local()) * 2;
                for (int tap = 0; tap < geom::GDN_NHIST; ++tap) {
                    uint8_t* d_tap = dst_c + size_t(tap) * new_tap_b;
                    for (int lk = 0; lk < L.nh_k(); ++lk) {
                        int olk = 0;
                        const int oq = qw38_old_k_owner(spec.old_nr, L.k0() + lk, olk);
                        const auto& oL = oldL[size_t(oq)];
                        const uint8_t* s_tap = old_states[size_t(oq)].data()
                            + size_t(li) * (oL.gdn_s_bytes() + oL.gdn_c_bytes()) + oL.gdn_s_bytes()
                            + size_t(tap) * size_t(oL.qkv_local()) * 2;
                        // Q head
                        std::memcpy(d_tap + size_t(lk) * head_bf16_b, s_tap + size_t(olk) * head_bf16_b, head_bf16_b);
                        // K head
                        std::memcpy(d_tap + size_t(L.nh_k() + lk) * head_bf16_b, s_tap + size_t(oL.nh_k() + olk) * head_bf16_b, head_bf16_b);
                    }
                    for (int lv = 0; lv < L.nh_v(); ++lv) {
                        int olv = 0;
                        const int oq = qw38_old_v_owner(spec.old_nr, L.v0() + lv, olv);
                        const auto& oL = oldL[size_t(oq)];
                        const uint8_t* s_tap = old_states[size_t(oq)].data()
                            + size_t(li) * (oL.gdn_s_bytes() + oL.gdn_c_bytes()) + oL.gdn_s_bytes()
                            + size_t(tap) * size_t(oL.qkv_local()) * 2;
                        // V head
                        std::memcpy(d_tap + size_t(2 * L.nh_k() + lv) * head_bf16_b, s_tap + size_t(2 * oL.nh_k() + olv) * head_bf16_b, head_bf16_b);
                    }
                }
            }
            // 3. PLE conv history copied verbatim from `from`
            const size_t ple_b = Qw38CkptLayout::ple_conv_bytes();
            std::memcpy(new_state.data() + L.state_bytes() - ple_b,
                        old_states[size_t(from)].data() + oldL[size_t(from)].state_bytes() - ple_b, ple_b);

            frw::pwrite_all(fd, new_state.data(), new_state.size(), (at + 1 + np * L.page_slot_blocks()) * L.slot, out_blocks);

            // Seal manifest block
            FlashEntry ne = x;
            ne.base = at;
            ne.blocks = L.entry_blocks(x.position);
            ne.manifest.compat = fnv64(compat.data(), compat.size());
            ne.manifest.page_bytes = L.page_bytes();
            ne.manifest.state_bytes = L.state_bytes();
            ne.manifest.page_sum = new_page_sum;
            ne.manifest.state_sum = ckpt_sum_host(new_state.data(), L.state_bytes());
            ne.manifest.seal();
            std::vector<uint8_t> mb(L.slot, 0);
            std::memcpy(mb.data(), &ne.manifest, sizeof ne.manifest);
            frw::pwrite_all(fd, mb.data(), mb.size(), at * L.slot, out_blocks);

            // Carry token file and populate ne.tokens (required by FlashIndexFiles::save)
            const std::string tname = std::to_string(x.generation);
            const std::string tb = src.read_all(from, "flash-ckpt.tokens.prev/" + tname);
            if (tb.size() != x.manifest.tokens * sizeof(int) || fnv64(tb.data(), tb.size()) != x.manifest.token_hash)
                continue;
            ne.tokens.resize(x.manifest.tokens);
            std::memcpy(ne.tokens.data(), tb.data(), tb.size());
            std::filesystem::create_directories(out_tokens);
            fidx::write_durable(out_tokens + "/" + tname, tb.data(), tb.size());

            at += ne.blocks;
            kept.push_back(std::move(ne));
        }

        if (::fsync(fd) != 0) frw::fail("fsync " + out_blocks + ": " + std::strerror(errno));
        std::sort(kept.begin(), kept.end(), [](const FlashEntry& a, const FlashEntry& b) { return a.generation < b.generation; });
        FlashIndex out_ix(store_blocks);
        out_ix.adopt(img.next, img.clock, kept);
        FlashIndexFiles(out_index).save(fnv64(compat.data(), compat.size()), out_ix);
        return int64_t(kept.size());
    } catch (const std::exception& e) {
        cleanup();
        why = e.what();
        return -1;
    }
}

} // namespace g6s2
