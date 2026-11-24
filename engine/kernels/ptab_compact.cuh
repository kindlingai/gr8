// CUDA wrapper for EL3b (#164) online per-step weight defragmentation / compaction.
// Wires `ptab::Compactor<ptab::Pool, cudaEvent_t>` (ptab_compact.h) to `cudaMemcpyAsync` (D2D on the serving stream),
// `ptab::set` (fixed-address device table update keeping captured graphs valid), `bind::refresh_shared_pieces`
// (whenever shared-expert row NEXP moves), and `cudaEventRecord` / `cudaEventQuery` (ensuring old slabs are handed to
// `Pool::give` + `reclaim_empty()` only after every prior launch and captured graph replay has completed).
#pragma once
#include "ptab.cuh"
#include "ptab_compact.h"
#include "bind.cuh"
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace ptab
{

struct CudaCompactLayer
{
    int layer_id = 0;
    int shared_row = -1;            // NEXP if this layer holds a trellis shared expert; -1 = none
    ptab::Table* table = nullptr;
    const bind::LayerTensors* tensors = nullptr; // optional: for bind::refresh_shared_pieces
    bind::Layer* binding = nullptr;              // optional: for bind::refresh_shared_pieces
};

class CudaCompactor
{
    std::vector<CudaCompactLayer> bind_layers_;
    std::vector<CompactLayerView> views_;
    cudaStream_t cur_st_ = nullptr;
    Compactor<Pool, cudaEvent_t> inner_;

    static CompactorOps<Pool, cudaEvent_t> make_ops(CudaCompactor* self, std::function<bool(size_t)> borrow_fn)
    {
        CompactorOps<Pool, cudaEvent_t> ops;
        ops.copy_d2d = [self](uint8_t* dst, const uint8_t* src, size_t n) -> const char* {
            return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, self->cur_st_) == cudaSuccess
                       ? nullptr
                       : "ptab::CudaCompactor: D2D copy failed";
        };
        ops.set_piece = [self](size_t li, int e, int b, const uint8_t* f1, size_t f1b,
                               const uint8_t* f2, size_t f2b) -> const char* {
            return ptab::set(*self->bind_layers_[li].table, e, b, f1, f1b, f2, f2b, self->cur_st_);
        };
        ops.refresh_shared = [self](size_t li) -> const char* {
            auto& bl = self->bind_layers_[li];
            if (bl.tensors && bl.binding) return bind::refresh_shared_pieces(*bl.tensors, *bl.binding);
            return nullptr;
        };
        ops.record_fence = [self](cudaEvent_t& ev) -> const char* {
            if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
                return "ptab::CudaCompactor: cudaEventCreateWithFlags failed";
            if (cudaEventRecord(ev, self->cur_st_) != cudaSuccess)
            {
                cudaEventDestroy(ev);
                ev = nullptr;
                return "ptab::CudaCompactor: cudaEventRecord failed";
            }
            return nullptr;
        };
        ops.fence_ready = [](const cudaEvent_t& ev) -> bool {
            return !ev || cudaEventQuery(ev) == cudaSuccess;
        };
        ops.destroy_fence = [](cudaEvent_t& ev) {
            if (ev) { cudaEventDestroy(ev); ev = nullptr; }
        };
        ops.borrow_headroom = std::move(borrow_fn);
        return ops;
    }

public:
    explicit CudaCompactor(CompactConfig cfg = {}, std::function<bool(size_t)> borrow_fn = {})
        : inner_(cfg, make_ops(this, std::move(borrow_fn))) {}

    void set_layers(std::vector<CudaCompactLayer> layers)
    {
        bind_layers_ = std::move(layers);
        views_.clear();
        views_.reserve(bind_layers_.size());
        for (const auto& bl : bind_layers_)
        {
            CompactLayerView lv;
            lv.layer_id = bl.layer_id;
            lv.shared_row = bl.shared_row;
            if (bl.table)
            {
                lv.rows = bl.table->rows;
                lv.eb = bl.table->eb;
                lv.host = &bl.table->host;
                lv.bytes = &bl.table->bytes;
            }
            views_.push_back(lv);
        }
    }

    // Performs up to `idle ? idle_experts_per_step : active_experts_per_step` expert compactions on `st`
    // and retires any old slab batches whose prior CUDA events have completed.
    const char* step(Pool& pool, bool idle, cudaStream_t st)
    {
        cur_st_ = st;
        return inner_.step(views_, pool, idle);
    }

    size_t poll_retire(Pool& pool) { return inner_.poll_retire(pool); }

    // Synchronizes `st` and drains every pending retirement batch (for shutdown or test verification).
    const char* drain(Pool& pool, cudaStream_t st)
    {
        if (cudaStreamSynchronize(st) != cudaSuccess) return "ptab::CudaCompactor: drain sync failed";
        inner_.poll_retire(pool);
        return nullptr;
    }

    const CompactStats& stats() const { return inner_.stats(); }
    size_t pending_batches() const { return inner_.pending_batches(); }
    bool done() const { return inner_.done(); }
};

}   // namespace ptab
