// Flash session checkpoints (GR8_GLM53F_CACHE), the device half: the copies between a Flash rank's state and one block
// store (g6kv_tier: mem:, an SSD file, or a lease on a KV node). flash_checkpoint.h has the index, the manifest and why
// a checkpoint is exact.
//
// An entry is blocks [base, base + 1 + pages + state_blocks) of the store:
//   base             the manifest (FlashManifest), written last
//   base + 1 + i     logical page i of the run (the run's i-th pool page) of every binding, main model then draft: per
//                    binding 64 rows of its latent cache, then its 16 pooled index keys (g6kv::ROW, g6kv::IDX_PAGE),
//                    the pool's page block layout. A CP binding (GR8_CP) keeps only the pages this rank holds, at its
//                    own place for each logical page (the cp list); another rank's page is zeros here
//   then             the KDA state of one slot, layer by layer: S [nhmax][128][128] fp32, then the conv history; then
//                    the session's raw index ring of every binding (g6kv::RAW_RING x IDX_RAW, attn/kvfmt.h: the raw
//                    rows of the pool the position may end inside); all cut into blocks (the last one padded with zeros)
// Capture computes a checksum of the device bytes before they leave; restore computes it again on the device bytes it
// wrote and refuses the entry when they differ (a torn, reused or corrupted range), as it does when the manifest is not
// the entry's. Both run on the serve stream, synchronously: the copies sit between prefill quanta or before a bind.
#pragma once
#include "flash_checkpoint.h"
#include "flash_ckpt_index.h"
#include "../../attn/kvfmt.h"
#include "kv_share.h"
#include "../../kvman-rs/g6kvman.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace g6s2 {
namespace fck {
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("Flash checkpoint ") + what + ": " + cudaGetErrorString(e));
}
// one binding's caches: page p of the latent cache at mla + p * mla_page, of the indexer keys at idx + p * idx_page.
// cp: a CP binding, whose pages are the cp list's (-1: another rank holds the page)
struct Binding { uint4* mla; uint4* idx; unsigned long long mla_page16, idx_page16; int cp; };
struct Extent { void* p; size_t bytes; };
// grid (pages, bindings): pool page pages[i] (a CP binding: cps[i]) of binding b <-> host slot i; a page another rank
// holds is zeros in the slot and is not written back
__global__ void k_pages(const int* pages, const int* cps, const Binding* b, const unsigned long long* off16, uint4* host,
                        unsigned long long slot16, int out) {
    const Binding x = b[blockIdx.y];
    const int pg = x.cp ? cps[blockIdx.x] : pages[blockIdx.x];
    uint4* h = host + (size_t)blockIdx.x * slot16 + off16[blockIdx.y];
    if (pg < 0) {
        if (out) for (size_t k = threadIdx.x; k < x.mla_page16 + (x.idx ? x.idx_page16 : 0); k += blockDim.x) h[k] = make_uint4(0, 0, 0, 0);
        return;
    }
    const size_t p = (size_t)pg;
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
// the same over n pool pages of one cache (w words a page): word i is word i % w of page pages[i / w], so a run's pages
// sum as its logical pages [0, n) laid end to end (and the identity run as the contiguous extent). A page -1 (another
// rank's, a CP binding) sums as zeros, the words its block holds
__global__ void k_sum_pages(const uint64_t* base, const int* pages, size_t n, size_t w, unsigned long long first,
                            unsigned long long* out) {
    uint64_t s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n * w; i += (size_t)gridDim.x * blockDim.x) {
        const size_t j = i / w;
        const int pg = pages[j];
        s += mix(pg < 0 ? 0 : base[(size_t)pg * w + (i - j * w)], first + i);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, (unsigned long long)s);
}
} // namespace fck

// What one rank checkpoints: its attention bindings (main, then draft) and the KDA state (kda.cuh KdaState: S
// [N_KDA][slots][nhmax][128][128], conv [N_KDA][slots][NHIST][3][nhmax*128], fp32).
struct FlashStateView {
    std::vector<fck::Binding> bindings;   // mla / idx device pointers; page sizes in 16-byte units
    std::vector<uint8_t*> rings;          // each binding's raw index ring (G6Attn::index_raw), in the bindings' order
    size_t ring_bytes = 0;                // bytes of one session's ring in one binding
    float *S = nullptr, *conv = nullptr;
    int layers = 0, slots = 0, pool_pages = 0;   // pool_pages: the pages every binding holds (a run's pages are below it)
    size_t s_slot = 0, c_slot = 0;        // floats of one slot's S / conv history in one layer
    // KDA slot k of every layer, then session q's ring of every binding, in the entry's order
    std::vector<fck::Extent> state(int k, int q) const {
        std::vector<fck::Extent> v;
        for (int l = 0; l < layers; ++l) {
            v.push_back({S + (size_t)l * slots * s_slot + (size_t)k * s_slot, s_slot * 4});
            v.push_back({conv + (size_t)l * slots * c_slot + (size_t)k * c_slot, c_slot * 4});
        }
        for (uint8_t* r : rings) v.push_back({r + (size_t)q * ring_bytes, ring_bytes});
        return v;
    }
    size_t state_bytes() const { return (size_t)layers * (s_slot + c_slot) * 4 + rings.size() * ring_bytes; }
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
    FlashIndexFiles files_;     // persist=on: the index and token files beside the store's blocks, else off
    bool persist_ = false;      // persist=on at boot
    uint8_t *host_ = nullptr, *host_dev_ = nullptr;   // mapped pinned: batch_ slots, registered with the store
    int* d_list_ = nullptr;                           // the run's pool pages, pool_pages of them at most
    int* d_cp_ = nullptr;                             // the CP bindings' pages of the run's logical pages (-1: another rank's)
    fck::Binding* d_bind_ = nullptr;
    unsigned long long *d_off_ = nullptr, *d_sum_ = nullptr;
    static void tk(int rc, const char* what) {
        if (rc) throw std::runtime_error(std::string("Flash checkpoint store ") + what + ": " + g6kv_tier_error());
    }
    uint8_t* host(int i) const { return host_ + (size_t)i * slot_; }
    uint64_t read_sum(cudaStream_t st) {
        fck::ck(cudaGetLastError(), "sum");
        uint64_t s = 0;
        fck::ck(cudaMemcpyAsync(&s, d_sum_, 8, cudaMemcpyDeviceToHost, st), "sum");
        fck::ck(cudaStreamSynchronize(st), "sum");
        return s;
    }
    uint64_t sum(const std::vector<fck::Extent>& ex, cudaStream_t st) {
        fck::ck(cudaMemsetAsync(d_sum_, 0, 8, st), "sum");
        uint64_t first = 0;
        for (const auto& e : ex) {
            if (e.bytes % 8) throw std::runtime_error("Flash checkpoint: extent not whole words");
            const size_t w = e.bytes / 8;
            if (w) fck::k_sum<<<(unsigned)std::min<size_t>(1024, (w + 255) / 256), 256, 0, st>>>(static_cast<const uint64_t*>(e.p), w, first, d_sum_);
            first += w;
        }
        return read_sum(st);
    }
    // the run's logical pages [0, np) of every binding (set_pages), in the entry's order
    uint64_t page_sum(uint64_t np, cudaStream_t st) {
        fck::ck(cudaMemsetAsync(d_sum_, 0, 8, st), "sum");
        uint64_t first = 0;
        auto one = [&](const uint4* p, unsigned long long page16, const int* list) {
            const size_t w = page16 * 2, words = np * w;
            if (words) fck::k_sum_pages<<<(unsigned)std::min<size_t>(1024, (words + 255) / 256), 256, 0, st>>>(
                reinterpret_cast<const uint64_t*>(p), list, np, w, first, d_sum_);
            first += words;
        };
        for (const auto& b : v_.bindings) {
            const int* list = b.cp ? d_cp_ : d_list_;
            one(b.mla, b.mla_page16, list);
            if (b.idx) one(b.idx, b.idx_page16, list);
        }
        return read_sum(st);
    }
    // the run's first np pool pages to the device (validated: the copies write device memory at these pages); cp: the CP
    // bindings' pages of the same logical pages (-1: another rank's), nullptr = the run's
    void set_pages(const std::vector<int>& run, uint64_t np, cudaStream_t st, const std::vector<int>* cp = nullptr) {
        if (np > run.size() || (cp && np > cp->size()))
            throw std::runtime_error("Flash checkpoint: " + std::to_string(np) + " pages from a run of " + std::to_string(run.size()));
        for (uint64_t i = 0; i < np; ++i)
            if (run[i] < 0 || run[i] >= v_.pool_pages || (cp && ((*cp)[i] < -1 || (*cp)[i] >= v_.pool_pages)))
                throw std::runtime_error("Flash checkpoint: pool page " + std::to_string(run[i]) + " out of range");
        fck::ck(cudaMemcpyAsync(d_list_, run.data(), np * sizeof(int), cudaMemcpyHostToDevice, st), "pages");
        fck::ck(cudaMemcpyAsync(d_cp_, (cp ? *cp : run).data(), np * sizeof(int), cudaMemcpyHostToDevice, st), "pages");
        fck::ck(cudaStreamSynchronize(st), "pages");
    }
    // logical pages [first, first + n) (set_pages) <-> host slots 0..n-1 (n <= batch_)
    void pages(uint64_t first, int n, bool out, cudaStream_t st) {
        fck::k_pages<<<dim3((unsigned)n, (unsigned)v_.bindings.size()), 256, 0, st>>>(d_list_ + first, d_cp_ + first, d_bind_, d_off_,
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
    // persist=on: the index and token files (flash_ckpt_index.h). A full or read-only disk turns persistence off and
    // the serve goes on: the index goes, and this boot keeps its entries in the store without one
    void save() {
        if (!files_.on()) return;
        try { files_.save(compat_, index_); }
        catch (const std::exception& e) {
            fprintf(stderr, "# flash ckpt persist: cannot write %s (%s): persistence off for this boot\n", files_.path().c_str(), e.what());
            files_.remove();
            files_ = FlashIndexFiles();
        }
    }
    // persist=on at boot: the index on disk (read in the constructor), its entries checked on the verify_ thread
    // (flash_entry_check: manifest block, page and KDA state checksums of the stored bytes), published by settle()
    FlashLayout layout_;
    FlashIndexImage found_;               // the index read at boot, then the entries that passed verify_
    std::string checked_;                 // what verify_ dropped and how long it took (the boot log)
    std::thread verify_;
    void verify() {
        const auto t0 = std::chrono::steady_clock::now();
        const FlashRead read = [this](uint64_t first, int n, uint8_t* dst) {
            std::vector<uint64_t> b(static_cast<size_t>(n)); std::vector<void*> p(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) { b[size_t(i)] = first + uint64_t(i); p[size_t(i)] = dst + size_t(i) * slot_; }
            tk(g6kv_tier_read_many(t_, size_t(n), b.data(), p.data()), "read");
        };
        const int threads = int(std::max(1u, std::min(8u, std::thread::hardware_concurrency() / 2)));
        std::vector<FlashEntry> kept;
        std::string dropped;
        uint64_t bytes = 0;
        for (auto& x : found_.entries) {
            const char* why = nullptr;
            std::string err;
            try { why = flash_entry_check(x, compat_, index_.blocks(), layout_, read, batch_, threads); }
            catch (const std::exception& e) { err = e.what(); why = err.c_str(); }
            if (why) { dropped += (dropped.empty() ? "" : ", ") + std::to_string(x.generation) + " (" + why + ")"; continue; }
            bytes += x.blocks * slot_;
            kept.push_back(std::move(x));
        }
        const size_t n = found_.entries.size();
        found_.entries = std::move(kept);
        char b[160];
        snprintf(b, sizeof b, "%zu of %zu entries verified (%.2f GiB) in %.2f s", found_.entries.size(), n, bytes / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        checked_ = b + (dropped.empty() ? std::string() : "; dropped " + dropped);
    }
public:
    // the last capture or restore: blocks, bytes and milliseconds (the log lines)
    struct Last { uint64_t blocks = 0; double ms = 0; } last;
    bool log = true;
    // spec / name / bytes: as KvTiers::add (name: this rank's lease, never the pool stash's). compat: what makes an entry
    // this engine's (model set, TP, rank, layout, engine build)
    // group: a group lease of `bytes` to start (FlashGroup), else a store of `bytes`. index_path: persist=on, the file
    // the index is kept in (loaded here, rewritten after every capture and resize)
    FlashCheckpoints(FlashStateView v, const std::string& spec, const std::string& name, uint64_t bytes,
                     const std::string& compat, const FlashGroup* group = nullptr, const std::string& index_path = {},
                     int batch = 32) : v_(std::move(v)), batch_(batch), files_(index_path) {
        if (v_.bindings.empty() || !v_.S || !v_.conv || v_.layers <= 0 || v_.slots <= 0 || v_.pool_pages <= 0) throw std::runtime_error("Flash checkpoint: no state to keep");
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
        fck::ck(cudaMalloc(&d_list_, sizeof(int) * (size_t)v_.pool_pages), "alloc");
        fck::ck(cudaMalloc(&d_cp_, sizeof(int) * (size_t)v_.pool_pages), "alloc");
        fck::ck(cudaHostAlloc(reinterpret_cast<void**>(&host_), slot_ * batch_, cudaHostAllocMapped), "alloc");
        fck::ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev_), host_, 0), "alloc");
        tk(g6kv_tier_register(t_, host_, slot_ * batch_), "register");
        layout_.slot = slot_; layout_.state_bytes = v_.state_bytes();
        for (const auto& b : v_.bindings) layout_.parts.push_back({size_t(b.mla_page16) * 16, b.idx ? size_t(b.idx_page16) * 16 : 0});
        persist_ = files_.on();
        if (files_.on()) {
            uint64_t stale = 0;
            if (const char* why = files_.load(compat_, index_.blocks(), found_, stale)) loaded = std::string(why) + ", starting empty";
            else {
                loaded = std::to_string(found_.entries.size()) + " entries to verify" +
                    (stale ? ", " + std::to_string(stale) + " without their token file dropped" : "");
                verify_ = std::thread([this] { verify(); });
            }
        }
    }
    std::string loaded;   // persist=on: what the boot found in the index file (the boot log)
    // persist=on at boot (true even after a failed write turned it off): this rank joins the bootstrap's agreement
    bool persisted() const { return persist_; }
    // persist=on: waits for the verification and publishes the entries that passed. Returns the boot log text.
    // Call before anything reads the index (the bootstrap's agreement, the serve): until then it is empty.
    std::string settle() {
        if (!verify_.joinable()) return loaded;
        verify_.join();
        for (auto& x : found_.entries) x.used_ms = index_.now_ms();
        try { index_.adopt(found_.next, found_.clock, std::move(found_.entries)); }
        catch (const std::exception& e) { checked_ += std::string("; ") + e.what() + ", starting empty"; }
        found_ = FlashIndexImage{};
        return checked_;
    }
    // the bootstrap's agreement (flash_ckpt_index.h): this rank's lines, then the entries every rank kept. agree()
    // rewrites the index and returns the boot log text
    std::string agree_summary() const { return flash_agree_summary(index_); }
    std::string agree(const std::string& agreed) {
        const size_t n = index_.size();
        const std::vector<uint64_t> gone = flash_keep_agreed(index_, agreed);
        save();
        std::string s = std::to_string(index_.size()) + " of " + std::to_string(n) + " entries kept by every rank, next " +
            std::to_string(index_.next_generation());
        for (size_t i = 0; i < gone.size(); ++i) s += (i ? ", " : "; dropped ") + std::to_string(gone[i]);
        return s;
    }
    ~FlashCheckpoints() {
        if (verify_.joinable()) verify_.join();
        if (t_) g6kv_tier_free(t_);
        cudaFree(d_bind_); cudaFree(d_off_); cudaFree(d_sum_); cudaFree(d_list_); cudaFree(d_cp_); cudaFreeHost(host_);
    }
    FlashCheckpoints(const FlashCheckpoints&) = delete;
    FlashCheckpoints& operator=(const FlashCheckpoints&) = delete;
    const FlashIndex& index() const { return index_; }
    FlashIndex& index() { return index_; }
    uint64_t blocks() const { return index_.blocks(); }
    size_t slot_bytes() const { return slot_; }
    uint64_t entry_blocks(uint32_t position) const { return layout_.entry_blocks(position); }
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
            // the moves run before the index is saved (capture saves first): a crash between them leaves an index whose
            // entries the load checks and the agreement drop, never a wrong restore
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
        save();
        return true;
    }
    // keep the run's logical rows [0, position) (run: its pool pages in order), KDA slot `slot` (the state after them)
    // and session sess's raw index rings as an entry for `tokens` (position + 1 of them). Returns the entry's generation,
    // or 0: the store cannot hold it. The same tokens already kept: that entry is marked used and nothing is written.
    // terminal: position may end mid-page (the whole last page is kept); otherwise it must be on the page grid.
    // cp: the CP bindings' pages of the run's logical pages (FlashCheckpoints::set_pages), nullptr = no CP binding
    uint64_t capture(uint32_t position, int slot, int sess, const std::vector<int>& run, const std::vector<int>& tokens,
                     const g6rt::CacheNamespace& ns, cudaStream_t st, bool terminal = false, const std::vector<int>* cp = nullptr) {
        if (!position || (!terminal && position % 64) || tokens.size() != size_t(position) + 1 || slot < 0 || slot >= v_.slots ||
            sess < 0 || sess >= g6kv::RAW_SESS)
            throw std::runtime_error("Flash checkpoint: capture at " + std::to_string(position) + " with " + std::to_string(tokens.size()) + " tokens");
        if (const FlashEntry* x = index_.same(tokens, ns)) { index_.touch(x->generation); return x->generation; }
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t np = (position + 63) / 64, need = entry_blocks(position);
        uint64_t base = 0;
        std::vector<uint64_t> gone;
        if (!index_.reserve(need, base, &gone)) return 0;
        // the entries reserve dropped leave the index on disk before their blocks are overwritten, so a crash during the
        // capture never leaves an index naming a range that holds part of another entry
        if (!gone.empty()) save();
        FlashEntry x;
        x.generation = index_.next_generation(); x.base = base; x.blocks = need; x.position = position; x.terminal = terminal;
        x.tokens = tokens; x.ns = ns;
        x.manifest = flash_manifest(x.generation, compat_, position, tokens, ns, np, block_, v_.state_bytes());
        const auto sx = v_.state(slot, sess);
        set_pages(run, np, st, cp);
        x.manifest.page_sum = page_sum(np, st);
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
        save();
        last = Last{need, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()};
        return g;
    }
    // a restore of `generation` read bytes that are not the entry's: the whole index file goes, so this rank's next boot
    // starts with no index and the bootstrap's agreement drops the other ranks' entries too. They may hold the entry
    // this rank lost, and one shard of an entry is no use.
    [[noreturn]] void fail(uint64_t generation, const char* why) {
        index_.drop(generation);
        if (files_.on()) { files_.remove(); files_ = FlashIndexFiles(); }
        throw std::runtime_error("Flash checkpoint " + std::to_string(generation) + ": " + why);
    }
    // entry `generation` back into the run's logical rows [0, position), KDA slot `slot` and session sess's raw index
    // rings; throws when the stored bytes are not the entry's (the serve fails: the other ranks restore too, so one rank
    // cannot fall back to a cold prefill alone)
    void restore(uint64_t generation, int slot, int sess, const std::vector<int>& run, cudaStream_t st, const std::vector<int>* cp = nullptr) {
        const FlashEntry* x = index_.find(generation);
        if (!x) throw std::runtime_error("Flash checkpoint: no entry " + std::to_string(generation));
        if (slot < 0 || slot >= v_.slots) throw std::runtime_error("Flash checkpoint: no KDA slot " + std::to_string(slot));
        if (sess < 0 || sess >= g6kv::RAW_SESS) throw std::runtime_error("Flash checkpoint: no session " + std::to_string(sess));
        const auto t0 = std::chrono::steady_clock::now();
        io(x->base, 1, false);
        FlashManifest m;
        std::memcpy(&m, host_, sizeof m);
        if (const char* why = m.differs(x->manifest)) fail(generation, why);
        const uint64_t np = (x->position + 63) / 64;
        set_pages(run, np, st, cp);
        for (uint64_t p = 0; p < np; p += batch_) {
            const int n = (int)std::min<uint64_t>(batch_, np - p);
            io(x->base + 1 + p, n, false);
            pages(p, n, false, st);
        }
        const auto sx = v_.state(slot, sess);
        state(sx, x->base + 1 + np, false, st);
        if (page_sum(np, st) != m.page_sum) fail(generation, "page checksum");
        if (sum(sx, st) != m.state_sum) fail(generation, "KDA state / raw ring checksum");
        last = Last{x->blocks, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()};
        index_.touch(generation);
    }
};
} // namespace g6s2
