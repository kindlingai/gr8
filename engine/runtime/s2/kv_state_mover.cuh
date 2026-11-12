// Generic CUDA mover and tiered store runner for `KvStateDesc` (`kv_state_desc.h`).
// Moves full 64-token pages (`Part::Full`) via 16-byte vectorized `uint4` loads/stores and arbitrary-alignment
// boundary-state slices (`Part::State(sub)`) via a single batched span kernel plus 32-byte FNV-1a header seal/verify,
// then dispatches batched O_DIRECT / RDMA / RAM I/O through `g6kv_tier_write_many` / `g6kv_tier_read_many`.
#pragma once

#include "kv_state_desc.h"
#include <cuda_runtime.h>
#include <chrono>
#include <functional>

namespace g6s2 {
namespace kvs {

inline void ck_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(err));
    }
}
#define G6S2_CUDA_CK(call) ::g6s2::kvs::ck_cuda((call), __FILE__, __LINE__)

struct DevPlane {
    uint8_t* base;
    uint64_t slot_stride;
    uint64_t page_bytes;
    uint64_t offset;
};

// A planar sub-array or another slot need not be 16-byte aligned.
__device__ inline void copy_bytes(uint8_t* h, uint8_t* d, size_t n, int out) {
    if (((uintptr_t(h) | uintptr_t(d) | n) & 15u) == 0) {
        auto* hv = reinterpret_cast<uint4*>(h);
        auto* dv = reinterpret_cast<uint4*>(d);
        for (size_t k = threadIdx.x; k < n / 16; k += blockDim.x) {
            if (out) hv[k] = dv[k]; else dv[k] = hv[k];
        }
    } else if (((uintptr_t(h) | uintptr_t(d) | n) & 3u) == 0) {
        auto* hv = reinterpret_cast<uint32_t*>(h);
        auto* dv = reinterpret_cast<uint32_t*>(d);
        for (size_t k = threadIdx.x; k < n / 4; k += blockDim.x) {
            if (out) hv[k] = dv[k]; else dv[k] = hv[k];
        }
    } else {
        for (size_t k = threadIdx.x; k < n; k += blockDim.x) {
            if (out) h[k] = d[k]; else d[k] = h[k];
        }
    }
}

__global__ void k_state_pages(const kvt::Item* items, const DevPlane* planes, uint8_t* host,
                              unsigned long long slot_bytes, uint32_t max_pages_per_slot, int out) {
    const DevPlane pl = planes[blockIdx.y];
    const kvt::Item it = items[blockIdx.x];
    const uint32_t handle = uint32_t(it.page);
    const uint32_t q = handle / max_pages_per_slot;
    const uint32_t p = handle % max_pages_per_slot;
    uint8_t* h = host + size_t(it.hslot) * slot_bytes + pl.offset;
    uint8_t* d = pl.base + uint64_t(q) * pl.slot_stride + size_t(p) * pl.page_bytes;
    copy_bytes(h, d, pl.page_bytes, out);
}

__global__ void k_state_spans(const KvCopySpan* spans, uint8_t* host, unsigned long long slot_bytes, int out) {
    const KvCopySpan sp = spans[blockIdx.x];
    uint8_t* h = host + size_t(sp.hslot) * slot_bytes + sp.hoff;
    uint8_t* d = sp.ptr;
    const size_t n = sp.len;
    copy_bytes(h, d, n, out);
}

} // namespace kvs

class KvStateMover {
    KvStateDesc desc_;
    std::vector<kvs::DevPlane> planes_;
    kvs::DevPlane* d_planes_ = nullptr;
    size_t raw_block_ = 0, slot_ = 0;
    int batch_ = 32;
    size_t spans_cap_ = 0;
    uint8_t* host_ = nullptr;
    uint8_t* host_dev_ = nullptr;
    kvt::Item* page_items_ = nullptr;
    kvt::Item* page_items_dev_ = nullptr;
    KvCopySpan* spans_ = nullptr;
    KvCopySpan* spans_dev_ = nullptr;

public:
    explicit KvStateMover(KvStateDesc desc, int batch = 32) : desc_(std::move(desc)), batch_(batch) {
        desc_.validate();
        if (batch_ <= 0) throw std::runtime_error("KV state batch must be positive");
        raw_block_ = desc_.raw_block_bytes();
        slot_ = desc_.slot_bytes();
        size_t off = 0;
        for (const auto& p : desc_.page_planes) {
            planes_.push_back(kvs::DevPlane{p.base, p.slot_stride, p.page_bytes(), off});
            off += size_t(p.page_bytes());
        }
        const size_t segs_per_state = desc_.tail_extents.size() + desc_.page_planes.size() + 1;
        spans_cap_ = std::max<size_t>(size_t(batch_) * segs_per_state, 64);

        if (!planes_.empty()) {
            G6S2_CUDA_CK(cudaMalloc(&d_planes_, planes_.size() * sizeof(kvs::DevPlane)));
            G6S2_CUDA_CK(cudaMemcpy(d_planes_, planes_.data(), planes_.size() * sizeof(kvs::DevPlane), cudaMemcpyHostToDevice));
        }
        G6S2_CUDA_CK(cudaHostAlloc(reinterpret_cast<void**>(&host_), slot_ * size_t(batch_), cudaHostAllocMapped));
        G6S2_CUDA_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev_), host_, 0));
        G6S2_CUDA_CK(cudaHostAlloc(reinterpret_cast<void**>(&page_items_), sizeof(kvt::Item) * size_t(batch_), cudaHostAllocMapped));
        G6S2_CUDA_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&page_items_dev_), page_items_, 0));
        G6S2_CUDA_CK(cudaHostAlloc(reinterpret_cast<void**>(&spans_), sizeof(KvCopySpan) * spans_cap_, cudaHostAllocMapped));
        G6S2_CUDA_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&spans_dev_), spans_, 0));
    }

    ~KvStateMover() {
        if (d_planes_) cudaFree(d_planes_);
        if (host_) cudaFreeHost(host_);
        if (page_items_) cudaFreeHost(page_items_);
        if (spans_) cudaFreeHost(spans_);
    }
    KvStateMover(const KvStateMover&) = delete;
    KvStateMover& operator=(const KvStateMover&) = delete;

    const KvStateDesc& desc() const { return desc_; }
    size_t block_bytes() const { return raw_block_; }
    size_t slot_bytes() const { return slot_; }
    int batch() const { return batch_; }
    uint8_t* host(int i) const { return host_ + size_t(i) * slot_; }
    size_t host_bytes() const { return slot_ * size_t(batch_); }

    void copy(const std::vector<kvt::Item>& items, bool out, cudaStream_t st) {
        if (items.empty()) return;
        if ((int)items.size() > batch_) throw std::runtime_error("KV state copy batch too large");
        G6S2_CUDA_CK(cudaStreamSynchronize(st));

        size_t n_pages = 0;
        std::vector<KvCopySpan> span_vec;
        struct StateHdrCheck {
            int hslot;
            uint32_t pos;
            uint32_t sub;
            uint32_t slice_len;
        };
        std::vector<StateHdrCheck> hdrs;

        for (const auto& it : items) {
            if (it.hslot < 0 || it.hslot >= batch_) {
                throw std::runtime_error("KV state hslot out of range");
            }
            if (it.part == 0) {
                uint32_t q = 0, p = 0;
                unpack_page_handle(uint32_t(it.page), desc_.max_pages_per_slot, q, p);
                if (q >= desc_.max_slots || p >= desc_.max_pages_per_slot)
                    throw std::runtime_error("KV state page handle out of range");
                page_items_[n_pages++] = it;
                if (out && raw_block_ < slot_) {
                    std::memset(host(it.hslot) + raw_block_, 0, slot_ - raw_block_);
                }
            } else if (it.part == 3) {
                uint32_t q = 0, pos = 0;
                unpack_state_handle(uint32_t(it.page), q, pos);
                const uint32_t sub = uint32_t(it.sub);
                const uint32_t slice_len = desc_.append_state_spans(q, pos, sub, uint32_t(it.hslot), span_vec);
                if (!out) {
                    desc_.verify_state_block_header(host(it.hslot), pos, sub, slice_len);
                }
                hdrs.push_back(StateHdrCheck{it.hslot, pos, sub, slice_len});
            } else {
                throw std::runtime_error("KV state mover: unsupported part kind");
            }
        }

        if (span_vec.size() > spans_cap_) {
            cudaFreeHost(spans_);
            spans_cap_ = span_vec.size() * 2;
            G6S2_CUDA_CK(cudaHostAlloc(reinterpret_cast<void**>(&spans_), sizeof(KvCopySpan) * spans_cap_, cudaHostAllocMapped));
            G6S2_CUDA_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&spans_dev_), spans_, 0));
        }
        if (!span_vec.empty()) {
            std::memcpy(spans_, span_vec.data(), span_vec.size() * sizeof(KvCopySpan));
        }

        if (n_pages > 0 && !planes_.empty()) {
            kvs::k_state_pages<<<dim3((unsigned)n_pages, (unsigned)planes_.size()), 256, 0, st>>>(
                page_items_dev_, d_planes_, host_dev_, slot_,
                desc_.max_pages_per_slot, out ? 1 : 0);
            G6S2_CUDA_CK(cudaGetLastError());
        }
        if (!span_vec.empty()) {
            kvs::k_state_spans<<<(unsigned)span_vec.size(), 256, 0, st>>>(spans_dev_, host_dev_, slot_, out ? 1 : 0);
            G6S2_CUDA_CK(cudaGetLastError());
        }
        G6S2_CUDA_CK(cudaStreamSynchronize(st));

        if (out) {
            for (const auto& h : hdrs) {
                desc_.seal_state_block_header(host(h.hslot), h.pos, h.sub, h.slice_len);
            }
        }
    }
};

// Tiered block store runner over `KvStateMover` (the boundary-state twin of `KvTiers` in `kv_tier.cuh`).
class KvStateTiers {
    KvStateMover mv_;
    std::vector<g6kv_tier*> t_;
    std::vector<uint64_t> slots_;

    static void ck(int rc, const char* what) {
        if (rc) throw std::runtime_error(std::string("KV store ") + what + ": " + g6kv_tier_error());
    }

public:
    uint64_t spilled = 0, moved = 0, restored = 0;
    double ms = 0, last_ms = 0, last_device_ms = 0;
    struct Io { uint64_t wblocks = 0, rblocks = 0, calls = 0; double wms = 0, rms = 0; };
    std::vector<Io> last_io;

    explicit KvStateTiers(KvStateDesc desc, int batch = 32) : mv_(std::move(desc), batch) {}
    ~KvStateTiers() { for (auto* t : t_) g6kv_tier_free(t); }
    KvStateTiers(const KvStateTiers&) = delete;
    KvStateTiers& operator=(const KvStateTiers&) = delete;

    KvStateMover& mover() { return mv_; }
    const KvStateDesc& desc() const { return mv_.desc(); }
    size_t size() const { return t_.size(); }
    const g6kv_tier* store(size_t i) const { return t_.at(i); }
    bool group(size_t i) const { return g6kv_store_seg_blocks(t_.at(i)) != 0; }
    std::string describe(size_t i) const { char b[256]; g6kv_tier_describe(t_[i], b, sizeof b); return b; }

    std::string last_io_text() const {
        std::string out;
        for (size_t t = 0; t < last_io.size(); ++t) {
            const Io& x = last_io[t];
            if (!x.calls) continue;
            char b[160];
            snprintf(b, sizeof b, "%st%zu w %llu (%.1f MB) %.1f ms r %llu (%.1f MB) %.1f ms, %llu calls",
                     out.empty() ? "" : "; ", t,
                     (unsigned long long)x.wblocks, x.wblocks * mv_.slot_bytes() / 1e6, x.wms,
                     (unsigned long long)x.rblocks, x.rblocks * mv_.slot_bytes() / 1e6, x.rms,
                     (unsigned long long)x.calls);
            out += b;
            uint64_t ram[3];
            if (g6kv_tier_ram_stats(t_[t], ram) == 0) {
                snprintf(b, sizeof b, ", spare RAM %llu hits %llu misses %llu skipped",
                         (unsigned long long)ram[0], (unsigned long long)ram[1], (unsigned long long)ram[2]);
                out += b;
            }
        }
        return out;
    }

    uint64_t add(const std::string& spec, const std::string& name, uint64_t bytes) {
        const uint64_t n = bytes / mv_.slot_bytes();
        if (!n) throw std::runtime_error("KV store " + spec + " smaller than one block");
        g6kv_tier* t = nullptr;
        ck(g6kv_tier_open(spec.c_str(), name.c_str(), mv_.block_bytes(), n, &t), "open");
        t_.push_back(t);
        slots_.push_back(n);
        if (g6kv_tier_block_bytes(t) != mv_.slot_bytes()) {
            throw std::runtime_error("KV store block size differs from KvStateMover slot size");
        }
        ck(g6kv_tier_register(t, mv_.host(0), mv_.host_bytes()), "register");
        return n;
    }

    uint64_t add_group(const std::string& spec, const std::string& name, const std::string& group, uint32_t nr,
                       uint32_t rank, uint64_t floor, uint64_t cap, uint32_t weight) {
        g6kv_tier* t = nullptr;
        ck(g6kv_tier_open_group(spec.c_str(), name.c_str(), group.c_str(), nr, rank, floor, cap, weight,
                                mv_.block_bytes(), &t),
           "open group");
        const uint64_t n = g6kv_store_blocks(t);
        t_.push_back(t);
        slots_.push_back(n);
        if (g6kv_tier_block_bytes(t) != mv_.slot_bytes()) {
            throw std::runtime_error("KV store block size differs from KvStateMover slot size");
        }
        ck(g6kv_tier_register(t, mv_.host(0), mv_.host_bytes()), "register");
        return n;
    }

    void run(const std::vector<g6kv_io>& io, cudaStream_t st) {
        if (io.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        double dev = 0;
        last_io.assign(t_.size(), Io{});
        size_t i = 0;

        auto many = [&](const std::vector<uint32_t>& t, const std::vector<uint64_t>& blocks, bool write) {
            for (uint32_t tier = 0; tier < t_.size(); ++tier) {
                std::vector<uint64_t> b;
                std::vector<void*> p;
                for (size_t j = 0; j < t.size(); ++j) {
                    if (t[j] == tier) {
                        b.push_back(blocks[j]);
                        p.push_back(mv_.host((int)j));
                    }
                }
                if (b.empty()) continue;
                const auto c0 = std::chrono::steady_clock::now();
                ck(write ? g6kv_tier_write_many(t_[tier], b.size(), b.data(), p.data())
                         : g6kv_tier_read_many(t_[tier], b.size(), b.data(), p.data()),
                   write ? "write" : "read");
                const double c = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                Io& x = last_io[tier];
                ++x.calls;
                if (write) { x.wblocks += b.size(); x.wms += c; }
                else { x.rblocks += b.size(); x.rms += c; }
            }
        };

        auto device = [&](const std::vector<kvt::Item>& items, bool out) {
            const auto d0 = std::chrono::steady_clock::now();
            mv_.copy(items, out, st);
            dev += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
        };

        while (i < io.size()) {
            const kvt::Batch x = kvt::next_batch(io, i, mv_.batch(), 0, mv_.batch());
            if (x.kind == 0) { device(x.items, true); many(x.from, x.slots, true); spilled += x.slots.size(); }
            else if (x.kind == 1) { many(x.from, x.slots, false); many(x.to, x.to_slots, true); moved += x.slots.size(); }
            else { many(x.from, x.slots, false); device(x.items, false); restored += x.slots.size(); }
        }
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ms += elapsed;
        last_ms = elapsed;
        last_device_ms = dev;
    }

    void round_trip_page(uint32_t page_handle, cudaStream_t st, const std::function<void()>& between) {
        std::vector<g6kv_io> out{g6kv_io{0, page_handle, 0, 0, 0, 0, 0, 0}};
        std::vector<g6kv_io> in{g6kv_io{2, page_handle, 0, 0, 0, 0, 0, 0}};
        run(out, st);
        between();
        run(in, st);
    }
};

} // namespace g6s2
