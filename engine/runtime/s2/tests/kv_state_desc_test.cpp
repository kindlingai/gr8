// Host end-to-end test for `KvStateDesc`, `StashPool`, `kvt::next_batch`, and `g6kv_tier` (`mem:` and `ssd:` with
// `persist=on` + O_DIRECT), simulating DeepSeek-V4.1's exact `COMP_FP4=1` planar `ica` + `compq` + `ca[3]` page layout
// (`56,960 B` / `57,344 B` slot) plus `C.ring` + `kvs/scs` + `ehist` + `DSpark` boundary state across:
//   - sub-64 and odd token positions (`pos = 37`, `pos = 201`, `pos = 253`, `pos = 389`)
//   - zero-duplicate-page PromptEnd + RequestEnd sibling snapshots (`keep_from = true`)
//   - reasoning-stripped fallback to PromptEnd vs full-history hit on RequestEnd
//   - `1.00x` write-only-new delta page extension across turns
//   - `keep_id` park-eviction pinning, steering `CacheNamespace` isolation, and SSD restart recovery
//   - corruption detection via `KvStateHeader32`'s per-block FNV-1a checksum
#include "s2/kv_plan.h"
#include "s2/kv_state_desc.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace g6s2;

struct SimDs41Buffers {
    static constexpr uint32_t SLOTS = 4;
    static constexpr uint32_t CAP = 512;      // 8 pages max per slot
    static constexpr uint32_t MAX_PAGES = CAP / 64;
    static constexpr int LAYERS = 40;
    static constexpr int COMP_SRC_LAYERS = 3; // KV-sharing ratio-2 source layers {2, 8, 14}
    static constexpr size_t COMP_SS = (CAP / 2) * 288;
    static constexpr size_t ICA_HALF_RPP = CAP / 2 + 1;
    static constexpr size_t ICA_FULL_RPP = CAP + 1;

    std::vector<std::vector<uint8_t>> compq; // 3 ratio-2 source layers {2, 8, 14}
    std::vector<std::vector<uint8_t>> rings; // 40 main + 3 dspark layers
    std::vector<std::vector<uint8_t>> kvs;   // 3 ratio-2 sources
    std::vector<std::vector<uint8_t>> scs;   // 3 ratio-2 sources
    std::vector<std::vector<uint8_t>> ica;   // 4 planar [codes | sf]
    std::vector<uint8_t> ca3;
    std::vector<uint8_t> ehist;

    SimDs41Buffers()
        : compq(COMP_SRC_LAYERS, std::vector<uint8_t>(SLOTS * COMP_SS, 0)),
          rings(LAYERS + 3, std::vector<uint8_t>(SLOTS * 24 * 1024, 0)),
          kvs(3, std::vector<uint8_t>(SLOTS * 4096, 0)),
          scs(3, std::vector<uint8_t>(SLOTS * 4096, 0)),
          ica(4),
          ca3(SLOTS * ICA_FULL_RPP * 288, 0),
          ehist(SLOTS * 64 * sizeof(int), 0) {
        for (int k = 0; k < 3; ++k) ica[k].assign(SLOTS * ICA_HALF_RPP * 68, 0);
        ica[3].assign(SLOTS * ICA_FULL_RPP * 68, 0);
    }

    KvStateDesc build_desc() {
        KvStateDesc d;
        d.model_tag = "ds41-fp4-sim";
        d.max_slots = SLOTS;
        d.max_pages_per_slot = MAX_PAGES;
        d.mtp_lookahead = false;
        d.allow_partial_pages = true;

        for (int s = 0; s < COMP_SRC_LAYERS; ++s) {
            d.page_planes.push_back(KvPagePlane{compq[s].data(), COMP_SS, 32, 288, 2});
        }
        for (int i = 0; i < LAYERS; ++i) {
            d.tail_extents.push_back(KvTailExtent{rings[i].data(), 24 * 1024, 24 * 1024});
        }
        for (int s = 0; s < 3; ++s) {
            d.tail_extents.push_back(KvTailExtent{kvs[s].data(), 4096, 4096});
            d.tail_extents.push_back(KvTailExtent{scs[s].data(), 4096, 4096});
        }
        for (int k = 0; k < 4; ++k) {
            const uint32_t ratio = (k < 3) ? 2u : 1u;
            const uint32_t rpp_page = 64u / ratio;
            const size_t rpp_slot = (k < 3) ? ICA_HALF_RPP : ICA_FULL_RPP;
            const uint64_t stride = rpp_slot * 68u;
            d.page_planes.push_back(KvPagePlane{ica[k].data(), stride, rpp_page, 64, ratio});
            d.page_planes.push_back(KvPagePlane{ica[k].data() + rpp_slot * 64u, stride, rpp_page, 4, ratio});
        }
        d.page_planes.push_back(KvPagePlane{ca3.data(), ICA_FULL_RPP * 288u, 64, 288, 1});
        d.tail_extents.push_back(KvTailExtent{ehist.data(), 256, 256});
        for (int j = 0; j < 3; ++j) {
            d.tail_extents.push_back(KvTailExtent{rings[LAYERS + j].data(), 24 * 1024, 24 * 1024});
        }
        d.validate();
        return d;
    }

    static uint8_t pattern(uint64_t seed, size_t idx) {
        uint64_t x = seed ^ (idx * 0x9E3779B97F4A7C15ull);
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;
        return uint8_t((x >> 24) & 0xFFu);
    }

    // Deterministic synthetic truth for prefix `tokens[0..pos)` in `slot` (plus namespace salt).
    void write_truth(uint32_t slot, const std::vector<int>& tokens, uint32_t pos, uint8_t ns_salt = 0) {
        const uint32_t p2 = pos / 2;
        // Position-indexed rows depend ONLY on the tokens up to that row!
        for (uint32_t r = 0; r < p2; ++r) {
            const uint64_t rseed = uint64_t(uint32_t(tokens[2 * r])) * 1315423911ull +
                                   uint64_t(uint32_t(tokens[2 * r + 1])) * 2654435761ull + ns_salt * 977ull;
            for (int i = 0; i < COMP_SRC_LAYERS; ++i) {
                uint8_t* dst = compq[i].data() + slot * COMP_SS + size_t(r) * 288;
                for (size_t b = 0; b < 288; ++b) dst[b] = pattern(rseed + uint64_t(i) * 17, b);
            }
            for (int k = 0; k < 3; ++k) {
                uint8_t* c_dst = ica[k].data() + slot * ICA_HALF_RPP * 68 + size_t(r) * 64;
                uint8_t* s_dst = ica[k].data() + slot * ICA_HALF_RPP * 68 + ICA_HALF_RPP * 64 + size_t(r) * 4;
                for (size_t b = 0; b < 64; ++b) c_dst[b] = pattern(rseed + 1000 + k, b);
                for (size_t b = 0; b < 4; ++b) s_dst[b] = pattern(rseed + 2000 + k, b);
            }
        }
        for (uint32_t r = 0; r < pos; ++r) {
            const uint64_t rseed = uint64_t(uint32_t(tokens[r])) * 2246822519ull + r * 3266489917ull + ns_salt * 977ull;
            uint8_t* c_dst = ica[3].data() + slot * ICA_FULL_RPP * 68 + size_t(r) * 64;
            uint8_t* s_dst = ica[3].data() + slot * ICA_FULL_RPP * 68 + ICA_FULL_RPP * 64 + size_t(r) * 4;
            uint8_t* a_dst = ca3.data() + slot * ICA_FULL_RPP * 288 + size_t(r) * 288;
            for (size_t b = 0; b < 64; ++b) c_dst[b] = pattern(rseed + 3000, b);
            for (size_t b = 0; b < 4; ++b) s_dst[b] = pattern(rseed + 4000, b);
            for (size_t b = 0; b < 288; ++b) a_dst[b] = pattern(rseed + 5000, b);
        }
        // Rolling boundary state depends on exact `pos` and trailing tokens.
        const uint64_t bseed = kv_fnv1a_bytes(tokens.data(), pos * sizeof(int)) ^ (uint64_t(ns_salt) << 40) ^ pos;
        for (size_t i = 0; i < rings.size(); ++i) {
            uint8_t* dst = rings[i].data() + slot * 24 * 1024;
            for (size_t b = 0; b < 24 * 1024; ++b) dst[b] = pattern(bseed + i * 31, b);
        }
        for (size_t s = 0; s < 3; ++s) {
            uint8_t* kd = kvs[s].data() + slot * 4096;
            uint8_t* sd = scs[s].data() + slot * 4096;
            for (size_t b = 0; b < 4096; ++b) {
                kd[b] = pattern(bseed + 700 + s, b);
                sd[b] = pattern(bseed + 800 + s, b);
            }
        }
        uint8_t* ed = ehist.data() + slot * 256;
        for (size_t b = 0; b < 256; ++b) ed[b] = pattern(bseed + 999, b);
    }

    void poison_slot(uint32_t slot) {
        for (auto& v : compq) std::memset(v.data() + slot * COMP_SS, 0xA5, COMP_SS);
        for (auto& v : rings) std::memset(v.data() + slot * 24 * 1024, 0x5A, 24 * 1024);
        for (auto& v : kvs) std::memset(v.data() + slot * 4096, 0x3C, 4096);
        for (auto& v : scs) std::memset(v.data() + slot * 4096, 0xC3, 4096);
        for (int k = 0; k < 3; ++k) std::memset(ica[k].data() + slot * ICA_HALF_RPP * 68, 0xF0, ICA_HALF_RPP * 68);
        std::memset(ica[3].data() + slot * ICA_FULL_RPP * 68, 0x0F, ICA_FULL_RPP * 68);
        std::memset(ca3.data() + slot * ICA_FULL_RPP * 288, 0x69, ICA_FULL_RPP * 288);
        std::memset(ehist.data() + slot * 256, 0x96, 256);
    }

    void assert_matches(const SimDs41Buffers& ref, uint32_t slot, uint32_t ref_slot, uint32_t pos) const {
        const uint32_t p2 = pos / 2;
        for (int i = 0; i < COMP_SRC_LAYERS; ++i) {
            assert(std::memcmp(compq[i].data() + slot * COMP_SS,
                               ref.compq[i].data() + ref_slot * COMP_SS,
                               size_t(p2) * 288) == 0);
        }
        for (int k = 0; k < 3; ++k) {
            const uint8_t* a = ica[k].data() + slot * ICA_HALF_RPP * 68;
            const uint8_t* b = ref.ica[k].data() + ref_slot * ICA_HALF_RPP * 68;
            assert(std::memcmp(a, b, size_t(p2) * 64) == 0);
            assert(std::memcmp(a + ICA_HALF_RPP * 64, b + ICA_HALF_RPP * 64, size_t(p2) * 4) == 0);
        }
        {
            const uint8_t* a = ica[3].data() + slot * ICA_FULL_RPP * 68;
            const uint8_t* b = ref.ica[3].data() + ref_slot * ICA_FULL_RPP * 68;
            assert(std::memcmp(a, b, size_t(pos) * 64) == 0);
            assert(std::memcmp(a + ICA_FULL_RPP * 64, b + ICA_FULL_RPP * 64, size_t(pos) * 4) == 0);
        }
        assert(std::memcmp(ca3.data() + slot * ICA_FULL_RPP * 288,
                           ref.ca3.data() + ref_slot * ICA_FULL_RPP * 288,
                           size_t(pos) * 288) == 0);
        for (size_t i = 0; i < rings.size(); ++i) {
            assert(std::memcmp(rings[i].data() + slot * 24 * 1024,
                               ref.rings[i].data() + ref_slot * 24 * 1024,
                               24 * 1024) == 0);
        }
        for (size_t s = 0; s < 3; ++s) {
            assert(std::memcmp(kvs[s].data() + slot * 4096, ref.kvs[s].data() + ref_slot * 4096, 4096) == 0);
            assert(std::memcmp(scs[s].data() + slot * 4096, ref.scs[s].data() + ref_slot * 4096, 4096) == 0);
        }
        assert(std::memcmp(ehist.data() + slot * 256, ref.ehist.data() + ref_slot * 256, 256) == 0);
    }
};

struct HostTierRunner {
    const KvStateDesc& desc;
    std::vector<g6kv_tier*> tiers;
    std::vector<uint8_t> staging;
    int batch = 8; // small batch forces multi-batch splitting across 23+ state blocks!
    uint64_t writes = 0, reads = 0;

    HostTierRunner(const KvStateDesc& d, int b = 8)
        : desc(d), staging(size_t(b) * d.slot_bytes(), 0), batch(b) {}
    ~HostTierRunner() { for (auto* t : tiers) g6kv_tier_free(t); }

    uint64_t add_tier(const std::string& spec, uint64_t blocks) {
        g6kv_tier* t = nullptr;
        if (g6kv_tier_open(spec.c_str(), "rank0", desc.raw_block_bytes(), blocks, &t) != 0) {
            throw std::runtime_error(std::string("tier_open failed: ") + g6kv_tier_error());
        }
        assert(g6kv_tier_block_bytes(t) == desc.slot_bytes());
        tiers.push_back(t);
        return blocks;
    }

    uint8_t* hslot_ptr(int i) { return staging.data() + size_t(i) * desc.slot_bytes(); }

    void run(StashPool& pool) {
        const std::vector<g6kv_io> io = pool.io();
        if (io.empty()) return;
        size_t i = 0;
        while (i < io.size()) {
            const kvt::Batch x = kvt::next_batch(io, i, batch, 0, batch);
            if (x.kind == 0) {
                for (const auto& it : x.items) {
                    if (it.part == 0) desc.pack_page_host(uint32_t(it.page), hslot_ptr(it.hslot));
                    else if (it.part == 3) desc.pack_state_block_host(uint32_t(it.page), uint32_t(it.sub), hslot_ptr(it.hslot));
                    else assert(false);
                }
                many(x.from, x.slots, true);
                writes += x.slots.size();
            } else if (x.kind == 1) {
                many(x.from, x.slots, false);
                many(x.to, x.to_slots, true);
            } else {
                many(x.from, x.slots, false);
                for (const auto& it : x.items) {
                    if (it.part == 0) desc.unpack_page_host(uint32_t(it.page), hslot_ptr(it.hslot));
                    else if (it.part == 3) desc.unpack_state_block_host(uint32_t(it.page), uint32_t(it.sub), hslot_ptr(it.hslot));
                    else assert(false);
                }
                reads += x.slots.size();
            }
        }
        pool.io_done();
    }

    void many(const std::vector<uint32_t>& ts, const std::vector<uint64_t>& blocks, bool write) {
        for (uint32_t ti = 0; ti < tiers.size(); ++ti) {
            std::vector<uint64_t> b;
            std::vector<void*> p;
            for (size_t j = 0; j < ts.size(); ++j) {
                if (ts[j] == ti) {
                    b.push_back(blocks[j]);
                    p.push_back(hslot_ptr(int(j)));
                }
            }
            if (b.empty()) continue;
            const int rc = write ? g6kv_tier_write_many(tiers[ti], b.size(), b.data(), p.data())
                                 : g6kv_tier_read_many(tiers[ti], b.size(), b.data(), p.data());
            if (rc != 0) throw std::runtime_error(std::string("tier io failed: ") + g6kv_tier_error());
        }
    }
};

std::vector<uint32_t> slot_pages(uint32_t slot, uint32_t max_pages, uint32_t n) {
    std::vector<uint32_t> out(n);
    for (uint32_t i = 0; i < n; ++i) out[i] = pack_page_handle(slot, i, max_pages);
    return out;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("kv-state-desc-test-" + std::to_string(int(::getpid())));
    fs::remove_all(root);
    fs::create_directories(root / "ssd");
    const std::string ssd_dir = (root / "ssd").string();
    const std::string ssd_spec = "ssd:" + (root / "ssd" / "blocks.bin").string();

    SimDs41Buffers sim, truth;
    const KvStateDesc desc = sim.build_desc();
    // Exact DeepSeek-V4.1 COMP_FP4=1 TP=4 KV-sharing geometry check:
    //   3 * (32 * 288) + 3 * (32 * 68) + 1 * (64 * 68 + 64 * 288) = 27,648 + 6,528 + 22,784 = 56,960 B
    //   4 KiB O_DIRECT aligned slot_bytes() = 57,344 B (56 KiB).
    assert(desc.page_block_bytes() == 56960u);
    assert(desc.slot_bytes() == 57344u);
    assert(desc.state_blocks() > 0);

    std::vector<int> chat(450);
    for (size_t i = 0; i < chat.size(); ++i) chat[i] = int(10000 + i * 7);

    const g6rt::CacheNamespace stock_ns{};
    g6rt::CacheNamespace steered_ns{};
    steered_ns[0] = 1;
    steered_ns[1] = 0x42;

    uint64_t pf1_id = 0, req1_id = 0, steered_id = 0;
    std::string saved_summary;

    {
        StashPool pool(2, 128, desc.state_blocks(), desc.mtp_lookahead);
        pool.set_rules(false, true, 0);
        HostTierRunner runner(desc, /*batch=*/7); // prime batch size tests cross-batch splitting!
        const uint64_t blocks = runner.add_tier(ssd_spec, 256);
        const std::string compat = "ds41-test-" + std::to_string(desc.format_hash());
        pool.add_tier(blocks, /*keep=all + whole*/ 2 | 4, ssd_dir.c_str(), compat.c_str());
        assert(pool.tiers_loaded() == 0);
        pool.set_policy(0, 0, 0, desc.slot_bytes());

        // 1. Odd sub-64 prompt (37 tokens -> 0 full pages) captured and restored cleanly.
        std::vector<int> p37(chat.begin(), chat.begin() + 37);
        sim.write_truth(0, p37, 37);
        truth.write_truth(0, p37, 37);
        const uint64_t id37 = pool.capture(p37, 37, 37, 37, {}, pack_state_handle(0, 37), 1, stock_ns);
        assert(id37 != 0);
        runner.run(pool);

        sim.poison_slot(1);
        std::vector<int> p201(chat.begin(), chat.begin() + 201);
        const g6kv_hit h37 = pool.match(p201, 200, stock_ns);
        assert(h37.stored == 1 && h37.skip == 37 && h37.age == id37);
        const auto r37 = pool.restore(h37.tier, h37.age, {}, pack_state_handle(1, 37));
        assert(r37.valid == 37 && r37.full_pages == 0);
        runner.run(pool);
        sim.assert_matches(truth, 1, 0, 37);

        // 2. Turn 1 PromptEnd at odd pos = 201 (3 full pages + boundary state), delta-extending id37.
        sim.write_truth(1, p201, 201);
        truth.write_truth(1, p201, 201);
        pf1_id = pool.capture(p201, 201, 201, 201, slot_pages(1, SimDs41Buffers::MAX_PAGES, 3),
                              pack_state_handle(1, 201), 2, stock_ns, id37, 0, /*keep_from=*/false);
        assert(pf1_id != 0 && !pool.has(id37));
        runner.run(pool);

        // 3. Turn 1 RequestEnd at odd pos = 253 (still 3 full pages) with keep_from = true:
        //    writes ZERO full pages (only `state_blocks()` blocks)!
        std::vector<int> p253(chat.begin(), chat.begin() + 253);
        sim.write_truth(1, p253, 253);
        truth.write_truth(2, p253, 253);
        const uint64_t w_before = runner.writes;
        req1_id = pool.capture(p253, 253, 253, 201, slot_pages(1, SimDs41Buffers::MAX_PAGES, 3),
                               pack_state_handle(1, 253), 2, stock_ns, pf1_id, 3, /*keep_from=*/true);
        assert(req1_id != 0 && pool.has(pf1_id));
        runner.run(pool);
        assert(runner.writes - w_before == desc.state_blocks());

        // 4. Steered session on the same tokens is isolated in its own namespace.
        sim.write_truth(3, p201, 201, /*ns_salt=*/7);
        truth.write_truth(3, p201, 201, /*ns_salt=*/7);
        steered_id = pool.capture(p201, 201, 201, 201, slot_pages(3, SimDs41Buffers::MAX_PAGES, 3),
                                  pack_state_handle(3, 201), 1, steered_ns);
        assert(steered_id != 0);
        runner.run(pool);

        saved_summary = pool.stored_summary();
    }

    // 5. Reopen from SSD (`persist=on` index + O_DIRECT block file) with zeroed memory buffers!
    SimDs41Buffers cold;
    const KvStateDesc cold_desc = cold.build_desc();
    {
        StashPool pool(2, 128, cold_desc.state_blocks(), cold_desc.mtp_lookahead);
        pool.set_rules(false, true, 0);
        HostTierRunner runner(cold_desc, /*batch=*/5);
        const uint64_t blocks = runner.add_tier(ssd_spec, 256);
        const std::string compat = "ds41-test-" + std::to_string(cold_desc.format_hash());
        pool.add_tier(blocks, 2 | 4, ssd_dir.c_str(), compat.c_str());
        assert(pool.tiers_loaded() == 0);
        assert(pool.stored_summary() == saved_summary);
        pool.keep_agreed(saved_summary);

        // Reasoning-stripped Turn 2 (diverges right after prompt end 201) hits PromptEnd (201) and restores exact bytes!
        std::vector<int> stripped(chat.begin(), chat.begin() + 201);
        for (int i = 0; i < 40; ++i) stripped.push_back(900000 + i);
        const g6kv_hit h_pf = pool.match(stripped, int(stripped.size()) - 1, stock_ns);
        assert(h_pf.stored == 1 && h_pf.skip == 201 && h_pf.age == pf1_id);
        cold.poison_slot(0);
        pool.restore(h_pf.tier, h_pf.age, slot_pages(0, SimDs41Buffers::MAX_PAGES, 3), pack_state_handle(0, 201));
        runner.run(pool);
        cold.assert_matches(truth, 0, 1, 201);

        // Steered query hits steered_id (201) and restores steered bytes (never stock bytes!).
        const g6kv_hit h_st = pool.match(stripped, int(stripped.size()) - 1, steered_ns);
        assert(h_st.stored == 1 && h_st.skip == 201 && h_st.age == steered_id);
        cold.poison_slot(3);
        pool.restore(h_st.tier, h_st.age, slot_pages(3, SimDs41Buffers::MAX_PAGES, 3), pack_state_handle(3, 201));
        runner.run(pool);
        cold.assert_matches(truth, 3, 3, 201);

        // Full-history Turn 2 (389 tokens -> 6 full pages) hits RequestEnd (253), restores 3 shared pages + state,
        // and delta-extends to 389 writing ONLY 3 new full pages + state_blocks()!
        std::vector<int> t2(chat.begin(), chat.begin() + 389);
        const g6kv_hit h_req = pool.match(t2, int(t2.size()) - 1, stock_ns);
        assert(h_req.stored == 1 && h_req.skip == 253 && h_req.age == req1_id);
        cold.poison_slot(2);
        const auto r_req = pool.restore(h_req.tier, h_req.age, slot_pages(2, SimDs41Buffers::MAX_PAGES, 6),
                                        pack_state_handle(2, 253));
        assert(r_req.valid == 253 && r_req.full_pages == 3);
        runner.run(pool);
        cold.assert_matches(truth, 2, 2, 253);

        cold.write_truth(2, t2, 389);
        truth.write_truth(0, t2, 389);
        assert(pool.drop_entry(pf1_id));
        const uint64_t w_before = runner.writes;
        const uint64_t pf2_id = pool.capture(t2, 389, 389, 389, slot_pages(2, SimDs41Buffers::MAX_PAGES, 6),
                                             pack_state_handle(2, 389), 3, stock_ns, req1_id, 3, /*keep_from=*/false);
        assert(pf2_id != 0);
        runner.run(pool);
        assert(runner.writes - w_before == 3u + cold_desc.state_blocks());

        // Verify restored 389-token snapshot matches truth bit-for-bit across all 6 pages + boundary state.
        cold.poison_slot(1);
        pool.restore(0, pf2_id, slot_pages(1, SimDs41Buffers::MAX_PAGES, 6), pack_state_handle(1, 389));
        runner.run(pool);
        cold.assert_matches(truth, 1, 0, 389);

        // 6. Flipping a payload byte in a packed state block must trip checksum verification!
        std::vector<uint8_t> blk(cold_desc.slot_bytes(), 0);
        cold_desc.pack_state_block_host(pack_state_handle(1, 389), 0, blk.data());
        blk[sizeof(KvStateHeader32) + 7] ^= 0x01u;
        bool caught = false;
        try {
            cold_desc.unpack_state_block_host(pack_state_handle(1, 389), 0, blk.data());
        } catch (const std::runtime_error&) {
            caught = true;
        }
        assert(caught);
    }

    // RAM tiers, even boundaries, sessions admission and disagreement removal.
    {
        StashPool pool(2, 64, desc.state_blocks(), false);
        HostTierRunner runner(desc);
        pool.add_tier(runner.add_tier("mem:", 256), 2 | 4);
        pool.set_policy(0, 0, 1, desc.slot_bytes());
        std::vector<int> p(chat.begin(), chat.begin() + 200);
        sim.write_truth(1, p, 200);
        truth.write_truth(0, p, 200);
        auto pages = slot_pages(1, SimDs41Buffers::MAX_PAGES, 3);
        assert(pool.capture(p, 200, 200, 200, pages, pack_state_handle(1, 200), 0, stock_ns) == 0);
        auto id = pool.capture(p, 200, 200, 200, pages, pack_state_handle(1, 200), 1, stock_ns);
        assert(id);
        runner.run(pool);
        sim.poison_slot(2);
        auto r = pool.restore(0, id, slot_pages(2, SimDs41Buffers::MAX_PAGES, 3), pack_state_handle(2, 200));
        assert(r.hits == 1);
        runner.run(pool);
        sim.assert_matches(truth, 2, 0, 200);
        assert(pool.drop_entry(id));

    }

    fs::remove_all(root);
    std::printf("kv_state_desc_test: PASS\n");
    return 0;
}
