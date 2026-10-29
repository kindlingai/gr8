// Flash session checkpoints (GR8_GLM53F_CACHE), the device half: the copies between a Flash rank's state and one block
// store (g6kv_tier: mem:, an SSD file, or a lease on a KV node). flash_checkpoint.h has the index, the manifest and why
// a checkpoint is exact.
//
// An entry is blocks [base, base + 1 + pages + state_blocks) of the store:
//   base             the manifest (FlashManifest), written last
//   base + 1 + i     page i of every binding, main model then draft: per binding 64 rows of its latent cache, then its
//                    64 rows of indexer keys and pools (g6kv::ROW, g6kv::IDX_PAGE), the pool's page block layout
//   then             the KDA state of one slot, layer by layer: S [nhmax][128][128] fp32, then the conv history, cut
//                    into blocks (the last one padded with zeros)
// Capture computes a checksum of the device bytes before they leave; restore computes it again on the device bytes it
// wrote and refuses the entry when they differ (a torn, reused or corrupted range), as it does when the manifest is not
// the entry's. Both run on the serve stream, synchronously: the copies sit between prefill quanta or before a bind.
#pragma once
#include "flash_checkpoint.h"
#include "kv_share.h"
#include "../../kvman-rs/g6kvman.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace g6s2 {
namespace fck {
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("Flash checkpoint ") + what + ": " + cudaGetErrorString(e));
}
// one binding's caches: page p of the latent cache at mla + p * mla_page, of the indexer keys at idx + p * idx_page
struct Binding { uint4* mla; uint4* idx; unsigned long long mla_page16, idx_page16; };
struct Extent { void* p; size_t bytes; };
// grid (pages, bindings): page items[i] of binding b <-> host slot i
__global__ void k_pages(const int* pages, const Binding* b, const unsigned long long* off16, uint4* host,
                        unsigned long long slot16, int out) {
    const Binding x = b[blockIdx.y];
    const size_t p = (size_t)pages[blockIdx.x];
    uint4* h = host + (size_t)blockIdx.x * slot16 + off16[blockIdx.y];
    uint4* m = x.mla + p * x.mla_page16;
    for (size_t k = threadIdx.x; k < x.mla_page16; k += blockDim.x) { if (out) h[k] = m[k]; else m[k] = h[k]; }
    if (!x.idx) return;
    h += x.mla_page16;
    uint4* d = x.idx + p * x.idx_page16;
    for (size_t k = threadIdx.x; k < x.idx_page16; k += blockDim.x) { if (out) h[k] = d[k]; else d[k] = h[k]; }
}
__device__ __forceinline__ uint64_t mix(uint64_t w, uint64_t i) {
    uint64_t x = w ^ (i * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ull; x ^= x >> 33;
    return x;
}
// out += sum of mix(word, first + i) over the extent (flash_checkpoint.h ckpt_sum_host)
__global__ void k_sum(const uint64_t* p, size_t words, unsigned long long first, unsigned long long* out) {
    uint64_t s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) s += mix(p[i], first + i);
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, (unsigned long long)s);
}
} // namespace fck

// What one rank checkpoints: its attention bindings (main, then draft) and the KDA state (kda.cuh KdaState: S
// [N_KDA][slots][nhmax][128][128], conv [N_KDA][slots][NHIST][3][nhmax*128], fp32).
struct FlashStateView {
    std::vector<fck::Binding> bindings;   // mla / idx device pointers; page sizes in 16-byte units
    float *S = nullptr, *conv = nullptr;
    int layers = 0, slots = 0;
    size_t s_slot = 0, c_slot = 0;        // floats of one slot's S / conv history in one layer
    // slot k of every layer, in the entry's order
    std::vector<fck::Extent> state(int k) const {
        std::vector<fck::Extent> v;
        for (int l = 0; l < layers; ++l) {
            v.push_back({S + (size_t)l * slots * s_slot + (size_t)k * s_slot, s_slot * 4});
            v.push_back({conv + (size_t)l * slots * c_slot + (size_t)k * c_slot, c_slot * 4});
        }
        return v;
    }
    size_t state_bytes() const { return (size_t)layers * (s_slot + c_slot) * 4; }
    size_t page_bytes() const {
        size_t n = 0;
        for (const auto& b : bindings) n += (b.mla_page16 + (b.idx ? b.idx_page16 : 0)) * 16;
        return n;
    }
};

// a checkpoint store on a group lease (--kv-store ...,group=): every rank of the cluster gives the same group, ranks,
// cap and weight (g6kv_tier_open_group); the lease starts at the store's size=
struct FlashGroup { std::string group; uint32_t ranks = 1, rank = 0; uint64_t cap = 0; uint32_t weight = 0; };

class FlashCheckpoints {
    FlashStateView v_;
    g6kv_tier* t_ = nullptr;
    FlashIndex index_;
    size_t block_ = 0, slot_ = 0, state_blocks_ = 0;
    int batch_ = 0;
    uint64_t compat_ = 0;
    g6kv_tier_totals seen_{};   // a group lease, the head: the index counters at its last report
    uint8_t *host_ = nullptr, *host_dev_ = nullptr;   // mapped pinned: batch_ slots, registered with the store
    int* pages_ = nullptr; int* pages_dev_ = nullptr; // mapped pinned: batch_ page numbers
    fck::Binding* d_bind_ = nullptr;
    unsigned long long *d_off_ = nullptr, *d_sum_ = nullptr;
    static void tk(int rc, const char* what) {
        if (rc) throw std::runtime_error(std::string("Flash checkpoint store ") + what + ": " + g6kv_tier_error());
    }
    uint8_t* host(int i) const { return host_ + (size_t)i * slot_; }
    uint64_t sum(const std::vector<fck::Extent>& ex, cudaStream_t st) {
        fck::ck(cudaMemsetAsync(d_sum_, 0, 8, st), "sum");
        uint64_t first = 0;
        for (const auto& e : ex) {
            if (e.bytes % 8) throw std::runtime_error("Flash checkpoint: extent not whole words");
            const size_t w = e.bytes / 8;
            if (w) fck::k_sum<<<(unsigned)std::min<size_t>(1024, (w + 255) / 256), 256, 0, st>>>(static_cast<const uint64_t*>(e.p), w, first, d_sum_);
            first += w;
        }
        fck::ck(cudaGetLastError(), "sum");
        uint64_t s = 0;
        fck::ck(cudaMemcpyAsync(&s, d_sum_, 8, cudaMemcpyDeviceToHost, st), "sum");
        fck::ck(cudaStreamSynchronize(st), "sum");
        return s;
    }
    std::vector<fck::Extent> page_extents(uint64_t pages) const {
        std::vector<fck::Extent> v;
        for (const auto& b : v_.bindings) {
            v.push_back({b.mla, (size_t)pages * b.mla_page16 * 16});
            if (b.idx) v.push_back({b.idx, (size_t)pages * b.idx_page16 * 16});
        }
        return v;
    }
    // pages [first, first + n) <-> host slots 0..n-1 (n <= batch_)
    void pages(uint64_t first, int n, bool out, cudaStream_t st) {
        fck::ck(cudaStreamSynchronize(st), "pages");
        for (int i = 0; i < n; ++i) pages_[i] = int(first + i);
        fck::k_pages<<<dim3((unsigned)n, (unsigned)v_.bindings.size()), 256, 0, st>>>(pages_dev_, d_bind_, d_off_,
            reinterpret_cast<uint4*>(host_dev_), slot_ / 16, out);
        fck::ck(cudaGetLastError(), "pages");
        fck::ck(cudaStreamSynchronize(st), "pages");
    }
    void io(uint64_t first_block, int n, bool write) {
        std::vector<uint64_t> b(n); std::vector<void*> p(n);
        for (int i = 0; i < n; ++i) { b[i] = first_block + i; p[i] = host(i); }
        tk(write ? g6kv_tier_write_many(t_, n, b.data(), p.data()) : g6kv_tier_read_many(t_, n, b.data(), p.data()),
           write ? "write" : "read");
    }
    // the state stream (extents in order) <-> blocks [first_block, ..), batch_ blocks a call
    void state(const std::vector<fck::Extent>& ex, uint64_t first_block, bool out, cudaStream_t st) {
        const size_t total = v_.state_bytes();
        size_t at = 0, e = 0, in_e = 0;   // stream offset; extent e, offset in it
        uint64_t block = first_block;
        while (at < total) {
            const int n = (int)std::min<size_t>(batch_, (total - at + slot_ - 1) / slot_);
            if (!out) io(block, n, false);
            else std::memset(host_, 0, (size_t)n * slot_);   // the last block's tail: zeros, not an old entry's bytes
            size_t room = (size_t)n * slot_, h = 0;
            while (room && e < ex.size()) {
                const size_t c = std::min(room, ex[e].bytes - in_e);
                uint8_t* d = static_cast<uint8_t*>(ex[e].p) + in_e;
                fck::ck(out ? cudaMemcpyAsync(host_ + h, d, c, cudaMemcpyDeviceToHost, st)
                            : cudaMemcpyAsync(d, host_ + h, c, cudaMemcpyHostToDevice, st), "state copy");
                h += c; room -= c; at += c; in_e += c;
                if (in_e == ex[e].bytes) { ++e; in_e = 0; }
            }
            fck::ck(cudaStreamSynchronize(st), "state copy");
            if (out) io(block, n, true);
            block += n;
        }
    }
public:
    // the last capture or restore: blocks, bytes and milliseconds (the log lines)
    struct Last { uint64_t blocks = 0; double ms = 0; } last;
    bool log = true;
    // spec / name / bytes: as KvTiers::add (name: this rank's lease, never the pool stash's). compat: what makes an entry
    // this engine's (model set, TP, rank, layout, engine build)
    // group: a group lease of `bytes` to start (FlashGroup), else a store of `bytes`
    FlashCheckpoints(FlashStateView v, const std::string& spec, const std::string& name, uint64_t bytes,
                     const std::string& compat, const FlashGroup* group = nullptr, int batch = 32) : v_(std::move(v)), batch_(batch) {
        if (v_.bindings.empty() || !v_.S || !v_.conv || v_.layers <= 0 || v_.slots <= 0) throw std::runtime_error("Flash checkpoint: no state to keep");
        block_ = v_.page_bytes();
        slot_ = (block_ + 4095) / 4096 * 4096;
        state_blocks_ = (v_.state_bytes() + slot_ - 1) / slot_;
        uint64_t n = bytes / slot_;
        if (n < 2 + state_blocks_) throw std::runtime_error("Flash checkpoint store " + spec + " smaller than one entry's state");
        if (group) {
            tk(g6kv_tier_open_group(spec.c_str(), name.c_str(), group->group.c_str(), group->ranks, group->rank, bytes, group->cap,
                                    group->weight, block_, &t_), "open group");
            n = g6kv_store_blocks(t_);
            if (n < 2 + state_blocks_) throw std::runtime_error("Flash checkpoint group lease " + spec + " smaller than one entry's state");
        } else tk(g6kv_tier_open(spec.c_str(), name.c_str(), block_, n, &t_), "open");
        if (g6kv_tier_block_bytes(t_) != slot_) throw std::runtime_error("Flash checkpoint store block size differs from the page block's");
        index_ = FlashIndex(n);
        compat_ = fnv64(compat.data(), compat.size());
        std::vector<unsigned long long> off;
        size_t o = 0;
        for (const auto& b : v_.bindings) { off.push_back(o / 16); o += (b.mla_page16 + (b.idx ? b.idx_page16 : 0)) * 16; }
        fck::ck(cudaMalloc(&d_bind_, v_.bindings.size() * sizeof(fck::Binding)), "alloc");
        fck::ck(cudaMemcpy(d_bind_, v_.bindings.data(), v_.bindings.size() * sizeof(fck::Binding), cudaMemcpyHostToDevice), "alloc");
        fck::ck(cudaMalloc(&d_off_, off.size() * 8), "alloc");
        fck::ck(cudaMemcpy(d_off_, off.data(), off.size() * 8, cudaMemcpyHostToDevice), "alloc");
        fck::ck(cudaMalloc(&d_sum_, 8), "alloc");
        fck::ck(cudaHostAlloc(reinterpret_cast<void**>(&host_), slot_ * batch_, cudaHostAllocMapped), "alloc");
        fck::ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev_), host_, 0), "alloc");
        fck::ck(cudaHostAlloc(reinterpret_cast<void**>(&pages_), sizeof(int) * batch_, cudaHostAllocMapped), "alloc");
        fck::ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pages_dev_), pages_, 0), "alloc");
        tk(g6kv_tier_register(t_, host_, slot_ * batch_), "register");
    }
    ~FlashCheckpoints() {
        if (t_) g6kv_tier_free(t_);
        cudaFree(d_bind_); cudaFree(d_off_); cudaFree(d_sum_); cudaFreeHost(host_); cudaFreeHost(pages_);
    }
    FlashCheckpoints(const FlashCheckpoints&) = delete;
    FlashCheckpoints& operator=(const FlashCheckpoints&) = delete;
    const FlashIndex& index() const { return index_; }
    FlashIndex& index() { return index_; }
    uint64_t blocks() const { return index_.blocks(); }
    size_t slot_bytes() const { return slot_; }
    uint64_t entry_blocks(uint32_t position) const { return 1 + (position + 63) / 64 + state_blocks_; }
    std::string describe() const { char b[256]; g6kv_tier_describe(t_, b, sizeof b); return b; }
    // a group lease (its segments move between the clusters on its node)
    bool group() const { return g6kv_store_seg_blocks(t_) != 0; }
    uint64_t segments() const { return g6kv_store_segs(t_); }
    // the head's report, in two halves so the index lock is not held over the node's round trip. ask (index locked):
    // the segments to ask for (s2/kv_share.h share_want, from the counters since the last report) and, when the head
    // offers (GR8_KV_SHARE_WHEN), what giving would drop (FlashIndex::coldest). busy: the engine has a request resident
    // or queued (share_want: it asks one entry ahead even with no capture since)
    std::pair<uint64_t, std::vector<uint64_t>> share_ask(bool offer, bool busy = false, size_t most = 8) {
        const uint64_t per = g6kv_store_seg_blocks(t_);
        if (!per) throw std::runtime_error("Flash checkpoint: the store is not a group lease");
        const g6kv_tier_totals now = index_.totals();
        const uint64_t want = share_want(seen_, now, per, busy);
        seen_ = now;
        return {want, offer ? index_.coldest(per, most) : std::vector<uint64_t>{}};
    }
    // report: the node's answer as the Resize to announce on `tier`, or nothing (kv_share.h share_poll)
    std::optional<Resize> share_report(uint32_t tier, uint64_t want, const std::vector<uint64_t>& ages) {
        const uint64_t per = g6kv_store_seg_blocks(t_), segs = g6kv_store_segs(t_);
        g6kv_report r{};
        tk(g6kv_store_report(t_, want, ages.data(), ages.size(), &r), "report");
        if (r.grant) return Resize{tier, segs + r.grant, (segs + r.grant) * per, r.txn};
        if (r.give && r.give < segs) return Resize{tier, segs - r.give, (segs - r.give) * per, r.txn};
        return std::nullopt;
    }
    // every rank at the Resize commit, index locked (kv_share.h share_apply). Grow: the lease, then the index. Shrink:
    // the index drops and packs (FlashIndex::shrink), the moves run, then the lease gives its top segments back. false:
    // the node refused the transaction (G6KV_STORE_REFUSED, every rank alike) and the lease keeps its size; a refused
    // shrink leaves the index smaller than the lease, which is safe. gone: the generations dropped
    bool apply(const Resize& r, std::vector<uint64_t>* gone = nullptr) {
        const uint64_t segs = g6kv_store_segs(t_);
        uint64_t blocks = 0;
        if (r.segs > segs) {
            const int rc = g6kv_store_grow(t_, r.segs, r.txn, &blocks);
            if (rc == G6KV_STORE_REFUSED) return false;
            tk(rc, "grow");
            if (blocks != r.slots) throw std::runtime_error("Flash checkpoint: the lease grew to a different size than the event's");
            index_.grow(blocks);
        } else if (r.segs < segs) {
            for (const auto& m : index_.shrink(r.slots, gone))
                for (uint64_t i = 0; i < m.blocks; i += batch_) {
                    const int n = (int)std::min<uint64_t>(batch_, m.blocks - i);
                    io(m.from + i, n, false);
                    io(m.to + i, n, true);
                }
            const int rc = g6kv_store_shrink(t_, r.segs, r.txn, &blocks);
            if (rc == G6KV_STORE_REFUSED) return false;
            tk(rc, "shrink");
            if (blocks != r.slots) throw std::runtime_error("Flash checkpoint: the lease shrank to a different size than the event's");
        }
        return true;
    }
    // keep rows [0, position) of the live sequence and KDA slot `slot` (the state after them) as an entry for `tokens`
    // (position + 1 of them). Returns the entry's generation, or 0: the store cannot hold it. The same tokens already
    // kept: that entry is marked used and nothing is written. terminal: position may end mid-page (the whole last page
    // is kept); otherwise it must be on the page grid.
    uint64_t capture(uint32_t position, int slot, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns, cudaStream_t st,
                     bool terminal = false) {
        if (!position || (!terminal && position % 64) || tokens.size() != size_t(position) + 1 || slot < 0 || slot >= v_.slots)
            throw std::runtime_error("Flash checkpoint: capture at " + std::to_string(position) + " with " + std::to_string(tokens.size()) + " tokens");
        if (const FlashEntry* x = index_.same(tokens, ns)) { index_.touch(x->generation); return x->generation; }
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t np = (position + 63) / 64, need = entry_blocks(position);
        uint64_t base = 0;
        if (!index_.reserve(need, base)) return 0;
        FlashEntry x;
        x.generation = index_.next_generation(); x.base = base; x.blocks = need; x.position = position; x.terminal = terminal;
        x.tokens = tokens; x.ns = ns;
        x.manifest = flash_manifest(x.generation, compat_, position, tokens, ns, np, block_, v_.state_bytes());
        const auto sx = v_.state(slot);
        x.manifest.page_sum = sum(page_extents(np), st);
        x.manifest.state_sum = sum(sx, st);
        for (uint64_t p = 0; p < np; p += batch_) {
            const int n = (int)std::min<uint64_t>(batch_, np - p);
            pages(p, n, true, st);
            io(base + 1 + p, n, true);
        }
        state(sx, base + 1 + np, true, st);
        x.manifest.seal();
        std::memset(host_, 0, slot_);
        std::memcpy(host_, &x.manifest, sizeof x.manifest);
        io(base, 1, true);
        const uint64_t g = x.generation;
        index_.add(std::move(x));
        last = Last{need, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()};
        return g;
    }
    // entry `generation` back into rows [0, position) and KDA slot `slot`; throws when the stored bytes are not the
    // entry's (the serve fails: the other ranks restore too, so one rank cannot fall back to a cold prefill alone)
    void restore(uint64_t generation, int slot, cudaStream_t st) {
        const FlashEntry* x = index_.find(generation);
        if (!x) throw std::runtime_error("Flash checkpoint: no entry " + std::to_string(generation));
        if (slot < 0 || slot >= v_.slots) throw std::runtime_error("Flash checkpoint: no KDA slot " + std::to_string(slot));
        const auto t0 = std::chrono::steady_clock::now();
        io(x->base, 1, false);
        FlashManifest m;
        std::memcpy(&m, host_, sizeof m);
        if (const char* why = m.differs(x->manifest)) throw std::runtime_error(std::string("Flash checkpoint ") + std::to_string(generation) + ": " + why);
        const uint64_t np = (x->position + 63) / 64;
        for (uint64_t p = 0; p < np; p += batch_) {
            const int n = (int)std::min<uint64_t>(batch_, np - p);
            io(x->base + 1 + p, n, false);
            pages(p, n, false, st);
        }
        const auto sx = v_.state(slot);
        state(sx, x->base + 1 + np, false, st);
        if (sum(page_extents(np), st) != m.page_sum) throw std::runtime_error("Flash checkpoint " + std::to_string(generation) + ": page checksum");
        if (sum(sx, st) != m.state_sum) throw std::runtime_error("Flash checkpoint " + std::to_string(generation) + ": KDA state checksum");
        last = Last{x->blocks, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()};
        index_.touch(generation);
    }
};
} // namespace g6s2
