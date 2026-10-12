// The engine's half of the pool's block store (kvman-rs src/stash.rs, pool.h store): it runs the copies the pool queues,
// between pool pages in device memory and blocks in a store (g6kv_tier: a file on this box's SSD, or a lease on a KV
// node over RDMA or TCP).
//
// A page block is one pool page of every layer binding, main model then draft: per binding 64 rows of the MLA cache
// (64 x 656 B), then, on full-index layers, 64 rows of the indexer keys (64 x 132 B). The full GLM 5.3 has 79 MLA
// caches and 22 index caches: 3,502,592 B. Each binding stores page p at p x its page bytes, so one kernel copies a batch
// of pages between the bindings and mapped pinned host memory, and the store reads and writes that memory.
// Spills run before restores: a restore may write a page that a spill of the same call still has to read.
#pragma once
#include "../rank.cuh"
#include "../../kvman-rs/g6kvman.h"
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6s2 {
namespace kvt {
constexpr size_t MLA16 = 64 * 656 / 16, IDX16 = 64 * 132 / 16;
struct Piece { uint4* mla; uint4* idx; unsigned long long off16; };
// grid (pages, bindings): page i of the batch, binding b. out: device -> host, else host -> device
__global__ void k_page_blocks(const int* pages, const Piece* pieces, uint4* host, unsigned long long slot16, int out)
{
    const Piece pc = pieces[blockIdx.y];
    const size_t p = (size_t)pages[blockIdx.x];
    uint4* h = host + (size_t)blockIdx.x * slot16 + pc.off16;
    uint4* d = pc.mla + p * MLA16;
    for (size_t k = threadIdx.x; k < MLA16; k += blockDim.x) { if (out) h[k] = d[k]; else d[k] = h[k]; }
    if (!pc.idx) return;
    h += MLA16; d = pc.idx + p * IDX16;
    for (size_t k = threadIdx.x; k < IDX16; k += blockDim.x) { if (out) h[k] = d[k]; else d[k] = h[k]; }
}
} // namespace kvt

class PageBlocks {
    std::vector<kvt::Piece> pieces_;
    kvt::Piece* d_pieces_ = nullptr;
    size_t block_ = 0, slot_ = 0;
    int batch_ = 0;
    uint8_t* host_ = nullptr;   // mapped pinned: batch_ slots
    uint8_t* host_dev_ = nullptr;
    int* pages_ = nullptr;      // mapped pinned: batch_ page numbers
    int* pages_dev_ = nullptr;
public:
    // bindings: every layer binding of the pool, main model then draft, in an order every rank shares
    PageBlocks(const std::vector<const G6AttnBinding*>& bindings, int batch = 32) : batch_(batch) {
        size_t off = 0;
        for (const G6AttnBinding* b : bindings) {
            pieces_.push_back({static_cast<uint4*>(b->args.mla_cache), static_cast<uint4*>(b->args.index_cache), off / 16});
            off += 64 * 656 + (b->args.index_cache ? 64 * 132 : 0);
        }
        block_ = off;
        slot_ = (block_ + 4095) / 4096 * 4096;
        G6RT_CK(cudaMalloc(&d_pieces_, pieces_.size() * sizeof(kvt::Piece)));
        G6RT_CK(cudaMemcpy(d_pieces_, pieces_.data(), pieces_.size() * sizeof(kvt::Piece), cudaMemcpyHostToDevice));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&host_), slot_ * batch_, cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev_), host_, 0));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&pages_), sizeof(int) * batch_, cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pages_dev_), pages_, 0));
    }
    ~PageBlocks() { cudaFree(d_pieces_); cudaFreeHost(host_); cudaFreeHost(pages_); }
    PageBlocks(const PageBlocks&) = delete;
    PageBlocks& operator=(const PageBlocks&) = delete;
    size_t block_bytes() const { return block_; }
    size_t slot_bytes() const { return slot_; }
    int batch() const { return batch_; }
    uint8_t* host(int i) const { return host_ + (size_t)i * slot_; }
    size_t host_bytes() const { return slot_ * batch_; }
    // pages (at most batch()) <-> host(0..n-1); synchronous on st
    void copy(const std::vector<int>& pages, bool out, cudaStream_t st) {
        if (pages.empty()) return;
        if ((int)pages.size() > batch_) throw std::runtime_error("page block batch too large");
        G6RT_CK(cudaStreamSynchronize(st));   // the previous batch's host reads or writes are done before pages_ changes
        for (size_t i = 0; i < pages.size(); ++i) pages_[i] = pages[i];
        kvt::k_page_blocks<<<dim3((unsigned)pages.size(), (unsigned)pieces_.size()), 256, 0, st>>>(
            pages_dev_, d_pieces_, reinterpret_cast<uint4*>(host_dev_), slot_ / 16, out);
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
    // pages), each a batch of up to batch() blocks. A batch's store reads and writes go out at once, per tier.
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
        auto device = [&](const std::vector<int>& pages, bool out) {
            const auto d0 = std::chrono::steady_clock::now();
            pb_.copy(pages, out, st);
            dev += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
        };
        while (i < io.size()) {
            const uint32_t kind = io[i].kind;
            std::vector<int> pages; std::vector<uint32_t> from, to; std::vector<uint64_t> slots, to_slots;
            while (i < io.size() && io[i].kind == kind && (int)pages.size() < pb_.batch()) {
                // an entry passed on twice in one call: the first move writes a block, then the second move reads it. A
                // batch reads before it writes, so the second move starts the next batch
                if (kind == 1) {
                    bool dep = false;
                    for (size_t j = 0; j < pages.size() && !dep; ++j) dep = to[j] == io[i].tier && to_slots[j] == io[i].slot;
                    if (dep) break;
                }
                pages.push_back((int)io[i].page); from.push_back(io[i].tier); to.push_back(io[i].to);
                slots.push_back(io[i].slot); to_slots.push_back(io[i].to_slot); ++i;
            }
            if (kind == 0) { device(pages, true); many(from, slots, true); spilled += pages.size(); }
            else if (kind == 1) { many(from, slots, false); many(to, to_slots, true); moved += pages.size(); }
            else { many(from, slots, false); device(pages, false); restored += pages.size(); }
        }
        const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ms += t;
        last_ms = t; last_device_ms = dev;
    }
    // spills pages to tier 0 and reads them back, with `between` run in the middle (the round-trip check)
    void round_trip(const std::vector<int>& pages, cudaStream_t st, const std::function<void()>& between) {
        std::vector<g6kv_io> out, in;
        for (size_t i = 0; i < pages.size(); ++i) {
            out.push_back(g6kv_io{0, (uint32_t)pages[i], 0, 0, i, 0});
            in.push_back(g6kv_io{2, (uint32_t)pages[i], 0, 0, i, 0});
        }
        run(out, st);
        between();
        run(in, st);
    }
};

// One --kv-tier value: STORE[,size=N(G|T)][,keep=evicted|shared|all]. STORE is ssd:PATH (a folder) or
// remote:HOST:PORT (remote:auto is the launcher's to resolve). Sizes are binary: G = 2^30, T = 2^40.
struct TierSpec {
    std::string store;
    uint64_t bytes = 0;
    int keep = 0;   // 0 evicted, 1 shared, 2 all
    static TierSpec parse(const std::string& v) {
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
                if (!(n > 0) || !mul) throw std::runtime_error("--kv-tier " + v + ": size=N with M, G or T");
                t.bytes = uint64_t(n * mul);
            } else if (k == "keep") {
                if (x == "evicted") t.keep = 0; else if (x == "shared") t.keep = 1; else if (x == "all") t.keep = 2;
                else throw std::runtime_error("--kv-tier " + v + ": keep=evicted, shared or all");
            } else throw std::runtime_error("--kv-tier " + v + ": unknown option " + k);
            at = next;
        }
        const bool ssd = t.store.rfind("ssd:", 0) == 0 && t.store.size() > 4;
        const bool remote = t.store.rfind("remote:", 0) == 0 && t.store.size() > 7 && t.store != "remote:auto";
        if (!ssd && !remote && t.store != "mem:")
            throw std::runtime_error("--kv-tier " + v + ": the store is ssd:PATH or remote:HOST:PORT (remote:auto needs the launcher)");
        if (!t.bytes) throw std::runtime_error("--kv-tier " + v + ": size=N(G|T) is needed");
        return t;
    }
    bool ssd() const { return store.rfind("ssd:", 0) == 0; }
};
} // namespace g6s2
