// The engine's half of the pool's block store (kvman-rs src/stash.rs, pool.h store): it runs the copies the pool queues,
// between pool pages in device memory and blocks in a store (g6kv_tier: a file on this box's SSD, or a lease on a KV
// node over RDMA or TCP).
//
// A page block is one pool page of every layer binding, main model then draft: per binding 64 rows of the MLA cache
// (64 x 656 B), then, on full-index layers, 64 rows of the indexer keys (64 x 132 B). The full GLM 5.3 has 79 MLA
// caches and 22 index caches: 3,502,592 B. Each binding stores page p at p x its page bytes, so one kernel copies a batch
// of pages between the bindings and mapped pinned host memory, and the store reads and writes that memory.
// Spills run before restores: a restore may write a page that a spill of the same call still has to read.
//
// Past R (a CP binding, cp_nr > 0) a binding's attention cache holds this rank's every nr-th logical page at physical
// pages R/64 + j, and its indexer keys every logical page. A run's tail past R goes to the store in two parts
// (g6kv_io part): 1 = this rank's shard, the attention pages of the CP bindings at a physical page, in the page block's
// layout; 2 = a logical page's replicated parts (the indexer keys, and every page of the bindings that are not CP: the
// MTP layer), rep_bytes() each, per_slot() to a block.
#pragma once
#include "../rank.cuh"
#include "../../kvman-rs/g6kvman.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6s2 {
// The page block's format: bump it whenever what a block holds, or how, changes (a stored block of another format is
// not this engine's). Stores keyed on it also carry the engine binary's stamp (exe_stamp).
constexpr int KV_BLOCK_FORMAT = 2;

// FNV-1a of this process's executable, hex: two builds that may compute different KV bits get different stamps
inline std::string exe_stamp() {
    FILE* f = fopen("/proc/self/exe", "rb");
    if (!f) return "unknown";
    uint64_t h = 14695981039346656037ull;
    std::vector<unsigned char> buf(1 << 20);
    for (size_t n; (n = fread(buf.data(), 1, buf.size(), f)) > 0;)
        for (size_t i = 0; i < n; ++i) h = (h ^ buf[i]) * 1099511628211ull;
    fclose(f);
    char s[17];
    snprintf(s, sizeof s, "%016llx", (unsigned long long)h);
    return s;
}

namespace kvt {
constexpr size_t MLA16 = 64 * 656 / 16, IDX16 = 64 * 132 / 16;
// off16: the binding's place in a page block; rep16: in a replicated record. phys: its attention cache's pages (the
// last physical page differs by rank past R). cp: its attention cache is sharded past R
struct Piece { uint4* mla; uint4* idx; unsigned long long off16, rep16; int phys; int cp; };
// one copy of a batch: page, part (g6kv_io), sub (part 2), and the host slot it goes to or comes from
struct Item { int page, part, sub, hslot; };
__device__ __forceinline__ void move16(uint4* h, uint4* d, size_t n, int out)
{
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) { if (out) h[k] = d[k]; else d[k] = h[k]; }
}
// grid (items, bindings): item i of the batch, binding b. out: device -> host, else host -> device
__global__ void k_page_blocks(const Item* items, const Piece* pieces, uint4* host, unsigned long long slot16,
                              unsigned long long rep16, int out)
{
    const Piece pc = pieces[blockIdx.y];
    const Item it = items[blockIdx.x];
    const size_t p = (size_t)it.page;
    uint4* slot = host + (size_t)it.hslot * slot16;
    if (it.part == 0) {
        move16(slot + pc.off16, pc.mla + p * MLA16, MLA16, out);
        if (pc.idx) move16(slot + pc.off16 + MLA16, pc.idx + p * IDX16, IDX16, out);
    } else if (it.part == 1) {
        if (pc.cp && it.page < pc.phys) move16(slot + pc.off16, pc.mla + p * MLA16, MLA16, out);   // a rank's shorter shard
    } else {
        uint4* h = slot + (size_t)it.sub * rep16 + pc.rep16;
        if (!pc.cp) { move16(h, pc.mla + p * MLA16, MLA16, out); h += MLA16; }
        if (pc.idx) move16(h, pc.idx + p * IDX16, IDX16, out);
    }
}
} // namespace kvt

class PageBlocks {
    std::vector<kvt::Piece> pieces_;
    kvt::Piece* d_pieces_ = nullptr;
    size_t block_ = 0, slot_ = 0, rep_ = 0;
    int batch_ = 0, items_cap_ = 0;
    uint8_t* host_ = nullptr;   // mapped pinned: batch_ slots
    uint8_t* host_dev_ = nullptr;
    kvt::Item* items_ = nullptr;   // mapped pinned: items_cap_ copies
    kvt::Item* items_dev_ = nullptr;
public:
    // bindings: every layer binding of the pool, main model then draft, in an order every rank shares
    PageBlocks(const std::vector<const G6AttnBinding*>& bindings, int batch = 32) : batch_(batch) {
        size_t off = 0, rep = 0;
        for (const G6AttnBinding* b : bindings) {
            const bool cp = b->args.cp_nr != 0;
            pieces_.push_back({static_cast<uint4*>(b->args.mla_cache), static_cast<uint4*>(b->args.index_cache), off / 16, rep / 16,
                               b->args.page_count, cp});
            off += 64 * 656 + (b->args.index_cache ? 64 * 132 : 0);
            rep += (cp ? 0 : 64 * 656) + (b->args.index_cache ? 64 * 132 : 0);
        }
        block_ = off;
        rep_ = rep;
        slot_ = (block_ + 4095) / 4096 * 4096;
        items_cap_ = batch_ * int(per_slot() ? per_slot() : 1);
        G6RT_CK(cudaMalloc(&d_pieces_, pieces_.size() * sizeof(kvt::Piece)));
        G6RT_CK(cudaMemcpy(d_pieces_, pieces_.data(), pieces_.size() * sizeof(kvt::Piece), cudaMemcpyHostToDevice));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&host_), slot_ * batch_, cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev_), host_, 0));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&items_), sizeof(kvt::Item) * items_cap_, cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&items_dev_), items_, 0));
    }
    ~PageBlocks() { cudaFree(d_pieces_); cudaFreeHost(host_); cudaFreeHost(items_); }
    PageBlocks(const PageBlocks&) = delete;
    PageBlocks& operator=(const PageBlocks&) = delete;
    size_t block_bytes() const { return block_; }
    size_t slot_bytes() const { return slot_; }
    // a logical page's replicated parts past R, and how many fit a block (0: no binding is CP)
    size_t rep_bytes() const { return rep_; }
    uint32_t per_slot() const {
        bool cp = false;
        for (const auto& p : pieces_) cp = cp || p.cp;
        return cp && rep_ ? uint32_t(slot_ / rep_) : 0;
    }
    int batch() const { return batch_; }
    int items() const { return items_cap_; }
    uint8_t* host(int i) const { return host_ + (size_t)i * slot_; }
    size_t host_bytes() const { return slot_ * batch_; }
    // whole pages (at most batch()) <-> host(0..n-1); synchronous on st
    void copy(const std::vector<int>& pages, bool out, cudaStream_t st) {
        std::vector<kvt::Item> items;
        for (size_t i = 0; i < pages.size(); ++i) items.push_back({pages[i], 0, 0, int(i)});
        copy(items, out, st);
    }
    // copies (at most items(), host slots below batch()) <-> their host slots; synchronous on st
    void copy(const std::vector<kvt::Item>& items, bool out, cudaStream_t st) {
        if (items.empty()) return;
        if ((int)items.size() > items_cap_) throw std::runtime_error("page block batch too large");
        G6RT_CK(cudaStreamSynchronize(st));   // the previous batch's host reads or writes are done before items_ changes
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].hslot < 0 || items[i].hslot >= batch_ || (items[i].part == 2 && uint32_t(items[i].sub) >= per_slot()))
                throw std::runtime_error("page block copy out of its batch");
            items_[i] = items[i];
        }
        kvt::k_page_blocks<<<dim3((unsigned)items.size(), (unsigned)pieces_.size()), 256, 0, st>>>(
            items_dev_, d_pieces_, reinterpret_cast<uint4*>(host_dev_), slot_ / 16, rep_ / 16, out);
        G6RT_CK(cudaGetLastError());
        G6RT_CK(cudaStreamSynchronize(st));
    }
};

// The stores of the pool's tiers (--kv-tier, in flag order) and the copies between them and the pool pages: run(io)
// after every pool call that may queue them (reserve, take, grow, retain), before the pages are written. Every rank
// opens its own stores, each the same number of blocks on every rank.
class KvTiers {
    PageBlocks pb_;
    std::vector<g6kv_tier*> t_;
    std::vector<uint64_t> slots_;
    static void ck(int rc, const char* what) {
        if (rc) throw std::runtime_error(std::string("KV store ") + what + ": " + g6kv_tier_error());
    }
public:
    uint64_t spilled = 0, moved = 0, restored = 0;
    double ms = 0, last_ms = 0, last_device_ms = 0;
    explicit KvTiers(const std::vector<const G6AttnBinding*>& bindings) : pb_(bindings) {}
    ~KvTiers() { for (auto* t : t_) g6kv_tier_free(t); }
    KvTiers(const KvTiers&) = delete;
    KvTiers& operator=(const KvTiers&) = delete;
    PageBlocks& blocks() { return pb_; }
    // a store (g6kv_tier_open's spec: "ssd:FILE", "remote:HOST:PORT", "mem:") of `bytes`, in whole page blocks. name:
    // this rank's lease on a KV node. Returns the number of blocks.
    uint64_t add(const std::string& spec, const std::string& name, uint64_t bytes) {
        const uint64_t n = bytes / pb_.slot_bytes();
        if (!n) throw std::runtime_error("KV store " + spec + " smaller than one page block");
        g6kv_tier* t = nullptr;
        ck(g6kv_tier_open(spec.c_str(), name.c_str(), pb_.block_bytes(), n, &t), "open");
        t_.push_back(t); slots_.push_back(n);
        if (g6kv_tier_block_bytes(t) != pb_.slot_bytes()) throw std::runtime_error("KV store block size differs from the page block's");
        ck(g6kv_tier_register(t, pb_.host(0), pb_.host_bytes()), "register");
        return n;
    }
    size_t size() const { return t_.size(); }
    std::string describe(size_t i) const { char b[256]; g6kv_tier_describe(t_[i], b, sizeof b); return b; }
    // the pool's copies, in order: runs of spills (pages -> blocks), moves (block -> block) and restores (blocks ->
    // pages), each a batch of up to batch() blocks. A batch's store reads and writes go out at once, per tier. The
    // copies of one block (a tail's replicated parts) stay in one batch.
    void run(const std::vector<g6kv_io>& io, cudaStream_t st) {
        if (io.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        double dev = 0;
        size_t i = 0;
        // blocks[j] of tier t[j] from or into host(j)
        auto many = [&](const std::vector<uint32_t>& t, const std::vector<uint64_t>& blocks, bool write) {
            for (uint32_t tier = 0; tier < t_.size(); ++tier) {
                std::vector<uint64_t> b; std::vector<void*> p;
                for (size_t j = 0; j < t.size(); ++j) if (t[j] == tier) { b.push_back(blocks[j]); p.push_back(pb_.host((int)j)); }
                if (b.empty()) continue;
                ck(write ? g6kv_tier_write_many(t_[tier], b.size(), b.data(), p.data())
                         : g6kv_tier_read_many(t_[tier], b.size(), b.data(), p.data()), write ? "write" : "read");
            }
        };
        auto device = [&](const std::vector<kvt::Item>& items, bool out) {
            const auto d0 = std::chrono::steady_clock::now();
            pb_.copy(items, out, st);
            dev += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
        };
        while (i < io.size()) {
            const uint32_t kind = io[i].kind;
            std::vector<kvt::Item> items;   // spills and restores: each copy and its host slot
            std::vector<uint32_t> from, to; std::vector<uint64_t> slots, to_slots;   // per host slot
            while (i < io.size() && io[i].kind == kind) {
                const g6kv_io& x = io[i];
                // an entry passed on twice in one call: the first move writes a block, then the second move reads it. A
                // batch reads before it writes, so the second move starts the next batch
                if (kind == 1) {
                    bool dep = false;
                    for (size_t j = 0; j < slots.size() && !dep; ++j) dep = to[j] == x.tier && to_slots[j] == x.slot;
                    if (dep || (int)slots.size() == pb_.batch()) break;
                    from.push_back(x.tier); to.push_back(x.to); slots.push_back(x.slot); to_slots.push_back(x.to_slot); ++i;
                    continue;
                }
                // a block's copies are next to each other: one host slot for them all
                const bool same = !slots.empty() && from.back() == x.tier && slots.back() == x.slot;
                if (!same && ((int)slots.size() == pb_.batch() || (int)items.size() + (int)pb_.per_slot() + 1 > pb_.items())) break;
                if (!same) { from.push_back(x.tier); slots.push_back(x.slot); }
                items.push_back({(int)x.page, (int)x.part, (int)x.sub, (int)slots.size() - 1}); ++i;
            }
            if (kind == 0) { device(items, true); many(from, slots, true); spilled += slots.size(); }
            else if (kind == 1) { many(from, slots, false); many(to, to_slots, true); moved += slots.size(); }
            else { many(from, slots, false); device(items, false); restored += slots.size(); }
        }
        const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ms += t;
        last_ms = t; last_device_ms = dev;
    }
    // spills pages to tier 0 and reads them back, with `between` run in the middle (the round-trip check)
    void round_trip(const std::vector<int>& pages, cudaStream_t st, const std::function<void()>& between) {
        std::vector<g6kv_io> out, in;
        for (size_t i = 0; i < pages.size(); ++i) {
            out.push_back(g6kv_io{0, (uint32_t)pages[i], 0, 0, i, 0, 0, 0});
            in.push_back(g6kv_io{2, (uint32_t)pages[i], 0, 0, i, 0, 0, 0});
        }
        run(out, st);
        between();
        run(in, st);
    }
};

// One --kv-store value: STORE,size=N(M|G|T)[,persist=on|off]. STORE is ssd:PATH (a folder) or remote:HOST:PORT
// (remote:auto is the launcher's to resolve). Sizes are binary: G = 2^30, T = 2^40. persist=on (SSD only) keeps the
// folder's system prefixes and counts across restarts; off (the default) clears it at boot. The deprecated --kv-tier
// (old) also takes keep=evicted|shared|all, which g6m4 maps to --kv-keep.
struct TierSpec {
    std::string store;
    uint64_t bytes = 0;
    int keep = 0;   // --kv-tier only: 0 evicted, 1 shared, 2 all
    bool persist = false;
    static TierSpec parse(const std::string& v, bool old = false) {
        TierSpec t;
        size_t at = v.find(',');
        t.store = v.substr(0, at);
        while (at != std::string::npos) {
            const size_t next = v.find(',', at + 1);
            const std::string kv = v.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
            const size_t eq = kv.find('=');
            const std::string k = kv.substr(0, eq), x = eq == std::string::npos ? "" : kv.substr(eq + 1);
            if (k == "size") {
                char* end = nullptr;
                const double n = strtod(x.c_str(), &end);
                const std::string unit = end ? end : "";
                const double mul = unit == "T" ? 1099511627776.0 : unit == "G" ? 1073741824.0 : unit == "M" ? 1048576.0 : 0.0;
                if (!(n > 0) || !mul) throw std::runtime_error("--kv-store " + v + ": size=N with M, G or T");
                t.bytes = uint64_t(n * mul);
            } else if (k == "persist") {
                if (x != "on" && x != "off") throw std::runtime_error("--kv-store " + v + ": persist=on or off");
                t.persist = x == "on";
            } else if (k == "keep" && !old) {
                throw std::runtime_error("--kv-store " + v + ": what the stores keep is --kv-keep system|sessions|all");
            } else if (k == "keep") {
                if (x == "evicted") t.keep = 0; else if (x == "shared") t.keep = 1; else if (x == "all") t.keep = 2;
                else throw std::runtime_error("--kv-tier " + v + ": keep=evicted, shared or all");
            } else throw std::runtime_error("--kv-store " + v + ": unknown option " + k);
            at = next;
        }
        const bool ssd = t.store.rfind("ssd:", 0) == 0 && t.store.size() > 4;
        const bool remote = t.store.rfind("remote:", 0) == 0 && t.store.size() > 7 && t.store != "remote:auto";
        if (!ssd && !remote && t.store != "mem:")
            throw std::runtime_error("--kv-store " + v + ": the store is ssd:PATH or remote:HOST:PORT (remote:auto needs the launcher)");
        if (!t.bytes) throw std::runtime_error("--kv-store " + v + ": size=N(G|T) is needed");
        if (t.persist && !ssd) throw std::runtime_error("--kv-store " + v + ": persist=on is for ssd: stores");
        return t;
    }
    bool ssd() const { return store.rfind("ssd:", 0) == 0; }
};
} // namespace g6s2
