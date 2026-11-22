// gr8 M4 tensor provider: pack image in host memory -> per-tensor device copies on first use, as the
// g6rt::Find that rank.cuh binds from. One DevStore may serve several packs (m4sim's six simulated ranks): tensors
// with the same sha256 and size share one device copy, so replicated tensors are held once.
// M5 replaces this by the builder's g6::DevImage (the pack streamed straight into one device allocation).
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include "rank.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "../loader/g6pack.h"
#include "../loader/devimage.h"
#include "../vmm.h"
#ifdef G6_RS_LOADER
#include "weight_check.cuh"
#endif

namespace g6rt
{

struct DevStore
{
    std::map<std::string, void*> by_sha;
    size_t bytes = 0, shared_bytes = 0;

    const void* get(const g6::Pack& pack, const g6::Entry& e)
    {
        const std::string key = std::string((const char*)e.sha256, 32) + ":" + std::to_string(e.nbytes);
        auto it = by_sha.find(key);
        if (it != by_sha.end()) { shared_bytes += e.nbytes; return it->second; }
        void* d;
        G6RT_CK(cudaMalloc(&d, e.nbytes));
        G6RT_CK(g6h2d::copy(d, pack.base() + e.offset, e.nbytes));   // pageable source: pinned staging (h2d.h)
        by_sha[key] = d;
        bytes += e.nbytes;
        return d;
    }
};

// the pack and the store must outlive the returned function
inline Find pack_find(const g6::Pack& pack, DevStore& store)
{
    return [&pack, &store](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = store.get(pack, *e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}

// the same from a device image of the whole pack (no host copy, no per-tensor copies): what a Spark rank uses
inline Find image_find(const g6::Pack& pack, const g6::DevImage& img)
{
    return [&pack, &img](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = img.ptr(*e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}

// EL3 (#151) direct slab load (GR8_PTAB=3): non-expert tensors live in a compact DevImage at dense_off[name], while
// routed-expert tensors (L%03d.tr3.fc1 / tr3.fc2) stream straight from the Rust loader into contiguous-per-expert
// slabs inside `pool`, without ever allocating tr3.fc1 / tr3.fc2 in DevImage.
struct DirectPtab
{
    std::shared_ptr<ptab::Pool> pool;
    std::map<int, std::shared_ptr<ptab::Table>> tables;
    std::unordered_map<std::string, uint64_t> dense_off;
    uint64_t dense_bytes = 0, expert_bytes = 0;
    bool active() const { return !tables.empty(); }
};

inline Find direct_image_find(const g6::Pack& pack, const g6::DevImage& img, const DirectPtab& direct)
{
    return [&pack, &img, &direct](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        int lyr = -1; bool is_fc2 = false;
        if (ptab::parse_tr3_layer_name(name, lyr, is_fc2) && direct.tables.count(lyr))
        {
            t.p = nullptr;   // bytes live in direct.tables[lyr] slabs, not in DevImage
        }
        else
        {
            auto it = direct.dense_off.find(name);
            t.p = it != direct.dense_off.end() ? (img.base() + it->second) : img.ptr(*e);
        }
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}

#ifdef G6_RS_LOADER
#include <mutex>
#include "../loader-rs/g6load.h"
#if G6_MODEL_DS41
#include "../loader-rs/ds41.h"
#endif

// Routed Rust-loader sink for GR8_PTAB=3 direct slab loading: dispatches each staged chunk [off, off + n) through
// ptab::RouteMap directly to either the compact dense DevImage or each routed expert's ptab::Pool slab.
struct RoutedRsSink
{
    const ptab::RouteMap* route = nullptr;
    std::mutex mu;
    std::vector<cudaStream_t> streams;

    static void* alloc(void*, size_t n)
    {
        void* p = nullptr;
        return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
    }
    static void release(void*, void* p) { cudaFreeHost(p); }
    static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n)
    {
        RoutedRsSink& s = *(RoutedRsSink*)ctx;
        cudaStream_t st;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            while ((int)s.streams.size() <= w)
            {
                cudaStream_t x;
                if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return 2;
                s.streams.push_back(x);
            }
            st = s.streams[w];
        }
        const int rc = s.route->dispatch(off, static_cast<const uint8_t*>(src), n, [&](uint8_t* dst, const uint8_t* p, size_t len) {
            return cudaMemcpyAsync(dst, p, len, cudaMemcpyHostToDevice, st) == cudaSuccess ? 0 : 3;
        });
        if (rc) return rc;
        return cudaStreamSynchronize(st) == cudaSuccess ? 0 : 4;
    }
};

// Direct slab load of a planned pack (GR8_PTAB=3, EL3 #151): reads each MoE layer's tiny tr3.table / tr3.kmix through
// read_fn, allocates a compact DevImage for non-expert tensors only and contiguous-per-expert slabs in direct.pool,
// then runs load_fn through RoutedRsSink so routed-expert bytes never enter DevImage.
template <class ReadFn, class LoadFn>
void rs_fill_direct_ptab(const g6::Pack& pack, g6::DevImage& img, DirectPtab& direct, int rank, const char* what,
                         ReadFn read_fn, LoadFn load_fn)
{
    struct LayerSpec { const g6::Entry *fc1 = nullptr, *fc2 = nullptr, *tab = nullptr, *kmix = nullptr; };
    std::map<int, LayerSpec> moe_layers;
    for (const g6::Entry& e : pack.entries())
    {
        int lyr = -1; bool is_fc2 = false;
        if (ptab::parse_tr3_layer_name(e.name, lyr, is_fc2))
        {
            LayerSpec& sp = moe_layers[lyr];
            if (is_fc2) sp.fc2 = &e; else sp.fc1 = &e;
        }
    }
    for (auto it = moe_layers.begin(); it != moe_layers.end(); )
    {
        char nm[64];
        snprintf(nm, sizeof nm, "L%03d.tr3.table", it->first);
        it->second.tab = pack.find(nm);
        snprintf(nm, sizeof nm, "L%03d.tr3.kmix", it->first);
        it->second.kmix = pack.find(nm);
        if (!it->second.fc1 || !it->second.fc2 || !it->second.tab || it->second.tab->shape.size() != 2 || it->second.tab->shape[1] != 6)
            it = moe_layers.erase(it);
        else ++it;
    }

    // Compact dense image: every tensor EXCEPT L%03d.tr3.fc1 / tr3.fc2 of routed MoE layers.
    uint64_t dense_bytes = 0;
    direct.dense_off.clear();
    for (const g6::Entry& e : pack.entries())
    {
        int lyr = -1; bool is_fc2 = false;
        if (ptab::parse_tr3_layer_name(e.name, lyr, is_fc2) && moe_layers.count(lyr)) continue;
        dense_bytes = (dense_bytes + 255ull) & ~255ull;
        direct.dense_off[e.name] = dense_bytes;
        dense_bytes += e.nbytes;
    }
    dense_bytes = std::max<uint64_t>((dense_bytes + 255ull) & ~255ull, 256ull);
    direct.dense_bytes = dense_bytes;

    void* dev = nullptr;
    void (*release)(void*) = nullptr;
    static const long vmm = (long)knob::img_vmm();
    if (vmm > 0)
    {
        std::string why;
        const size_t chunk = vmm > 1 ? (size_t)vmm << 20 : 0;
        if ((dev = g6vmm::alloc(dense_bytes, chunk, why))) release = [](void* p) { g6vmm::release(p); };
    }
    if (!dev && cudaMalloc(&dev, dense_bytes) != cudaSuccess)
        die("GR8_PTAB=3 direct load: cudaMalloc of compact dense image (" + std::to_string(dense_bytes) + " B) failed");

    ptab::RouteMap route;
    for (const g6::Entry& e : pack.entries())
    {
        auto it = direct.dense_off.find(e.name);
        if (it != direct.dense_off.end()) route.add(e.offset, e.nbytes, static_cast<uint8_t*>(dev) + it->second);
    }

    if (!direct.pool) direct.pool = std::make_shared<ptab::Pool>();
    direct.pool->vmm = vmm > 0;
    direct.tables.clear();
    direct.expert_bytes = 0;
    cudaStream_t st = nullptr;
    G6RT_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (const auto& [lyr, sp] : moe_layers)
    {
        const int rows = static_cast<int>(sp.tab->shape[0]);
        std::vector<long long> h(static_cast<size_t>(rows) * 6, 0);
        std::vector<long long> k;
        read_fn(sp.tab->offset, sp.tab->nbytes, h.data());
        if (sp.kmix && sp.kmix->nbytes)
        {
            k.assign(static_cast<size_t>(rows), 0);
            read_fn(sp.kmix->offset, sp.kmix->nbytes, k.data());
        }
        auto t = std::make_shared<ptab::Table>();
        if (const char* e = ptab::alloc_direct(*t, h, k, rows, g6nr::EB, *direct.pool, sp.fc1->offset, sp.fc2->offset, route, st))
            die("GR8_PTAB=3 direct load layer " + std::to_string(lyr) + ": " + e);
        direct.expert_bytes += t->slab_bytes;
        direct.tables[lyr] = std::move(t);
    }
    G6RT_CK(cudaStreamDestroy(st));

    std::string rerr;
    if (!route.finalize(rerr)) die(rerr);

    RoutedRsSink sink;
    sink.route = &route;
    const g6l_sink s{&sink, &RoutedRsSink::alloc, &RoutedRsSink::release, &RoutedRsSink::copy};
    g6l_stats stats{};
    char err[1024] = "";
    if (load_fn(&s, &stats, err, sizeof err)) die(std::string("Rust loader (direct ptab): ") + err);
    for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
    img.adopt(pack, static_cast<uint8_t*>(dev), dense_bytes, release);
    fprintf(stderr, "# rank %d: %s -> GR8_PTAB=3 direct slab load: compact dense %.2f GB + %zu MoE layers %.2f GB slabs "
            "(pool held %.2f GB, VMM %d; %s, %u threads) in %.1f s (%.2f GB/s)\n",
            rank, what, dense_bytes / 1e9, direct.tables.size(), direct.expert_bytes / 1e9, direct.pool->held / 1e9,
            direct.pool->vmm ? 1 : 0, stats.mode ? "read" : "mmap", stats.threads, stats.seconds,
            stats.image_bytes / 1e9 / std::max(stats.seconds, 1e-6));
}

// EL3 (#151 Stage 3) staged target weights for an in-process width switch: shares `direct.pool` with the active width,
// keeps all resident expert slabs untouched, holds only the newly-prefetched slabs (`added_slabs`) + compact repacked
// dense/attn/vocab image (`dense_img`), and tracks `dropped_slabs` to free via Pool::give + reclaim_empty after cutover.
struct StagedDeltaPtab
{
    DirectPtab direct;
    g6::DevImage dense_img;
    std::vector<ptab::SlabRef> added_slabs;
    std::vector<ptab::SlabRef> dropped_slabs;
    size_t kept_pieces = 0, added_pieces = 0, dropped_pieces = 0;
    size_t kept_bytes = 0, added_bytes = 0, dropped_bytes = 0;

    size_t rollback(size_t* chunks_freed = nullptr)
    {
        direct.tables.clear();
        dense_img = g6::DevImage{};
        dropped_slabs.clear();
        return direct.pool ? ptab::rollback_delta_slabs(*direct.pool, added_slabs, chunks_freed) : 0;
    }
    size_t retire_dropped(size_t* chunks_freed = nullptr)
    {
        added_slabs.clear();
        return direct.pool ? ptab::retire_dropped_slabs(*direct.pool, dropped_slabs, chunks_freed) : 0;
    }
};

// Prefetches only dense/attn/vocab tensors + missing routed-expert pieces for a target layout (`plan_handle` + `delta`)
// into `staged`, reusing all resident slabs in `current` (may be empty on a joiner rank) without touching them.
// Returns true on success; on failure rolls back newly allocated slabs and sets `err_out`.
inline bool rs_prefetch_delta_ptab(const g6::Pack& target_pack, void* plan_handle, const void* delta_handle,
                                   const DirectPtab& current, int target_eb, int rank, StagedDeltaPtab& staged,
                                   std::string& err_out)
{
    staged = StagedDeltaPtab{};
    staged.direct.pool = current.pool ? current.pool : std::make_shared<ptab::Pool>();
    static const long vmm = (long)knob::img_vmm();
    staged.direct.pool->vmm = vmm > 0;

    struct LayerSpec { const g6::Entry *fc1 = nullptr, *fc2 = nullptr, *tab = nullptr, *kmix = nullptr; };
    std::map<int, LayerSpec> moe_layers;
    for (const g6::Entry& e : target_pack.entries())
    {
        int lyr = -1; bool is_fc2 = false;
        if (ptab::parse_tr3_layer_name(e.name, lyr, is_fc2))
        {
            LayerSpec& sp = moe_layers[lyr];
            if (is_fc2) sp.fc2 = &e; else sp.fc1 = &e;
        }
    }
    for (auto it = moe_layers.begin(); it != moe_layers.end(); )
    {
        char nm[64];
        snprintf(nm, sizeof nm, "L%03d.tr3.table", it->first);
        it->second.tab = target_pack.find(nm);
        snprintf(nm, sizeof nm, "L%03d.tr3.kmix", it->first);
        it->second.kmix = target_pack.find(nm);
        if (!it->second.fc1 || !it->second.fc2 || !it->second.tab || it->second.tab->shape.size() != 2 || it->second.tab->shape[1] != 6)
            it = moe_layers.erase(it);
        else ++it;
    }

    uint64_t dense_bytes = 0;
    for (const g6::Entry& e : target_pack.entries())
    {
        int lyr = -1; bool is_fc2 = false;
        if (ptab::parse_tr3_layer_name(e.name, lyr, is_fc2) && moe_layers.count(lyr)) continue;
        dense_bytes = (dense_bytes + 255ull) & ~255ull;
        staged.direct.dense_off[e.name] = dense_bytes;
        dense_bytes += e.nbytes;
    }
    dense_bytes = std::max<uint64_t>((dense_bytes + 255ull) & ~255ull, 256ull);
    staged.direct.dense_bytes = dense_bytes;

    void* dev = nullptr;
    void (*release)(void*) = nullptr;
    if (vmm > 0)
    {
        std::string why;
        const size_t chunk = vmm > 1 ? (size_t)vmm << 20 : 0;
        if ((dev = g6vmm::alloc(dense_bytes, chunk, why))) release = [](void* p) { g6vmm::release(p); };
    }
    if (!dev && cudaMalloc(&dev, dense_bytes) != cudaSuccess)
    {
        err_out = "delta prefetch: cudaMalloc of target dense image failed";
        return false;
    }
    staged.dense_img.adopt(target_pack, static_cast<uint8_t*>(dev), dense_bytes, release);

    ptab::RouteMap route;
    for (const g6::Entry& e : target_pack.entries())
    {
        auto it = staged.direct.dense_off.find(e.name);
        if (it != staged.direct.dense_off.end()) route.add(e.offset, e.nbytes, static_cast<uint8_t*>(dev) + it->second);
    }

    cudaStream_t st = nullptr;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
    {
        staged.rollback();
        err_out = "delta prefetch: cudaStreamCreateWithFlags failed";
        return false;
    }

    char rerr_buf[512] = "";
    for (const auto& [lyr, sp] : moe_layers)
    {
        const int rows = static_cast<int>(sp.tab->shape[0]);
        std::vector<long long> h(static_cast<size_t>(rows) * 6, 0);
        std::vector<long long> k;
        if (g6l_plan_read(plan_handle, sp.tab->offset, sp.tab->nbytes, h.data(), rerr_buf, sizeof rerr_buf))
        {
            cudaStreamDestroy(st);
            staged.rollback();
            err_out = std::string("delta prefetch read tr3.table: ") + rerr_buf;
            return false;
        }
        if (sp.kmix && sp.kmix->nbytes)
        {
            k.assign(static_cast<size_t>(rows), 0);
            if (g6l_plan_read(plan_handle, sp.kmix->offset, sp.kmix->nbytes, k.data(), rerr_buf, sizeof rerr_buf))
            {
                cudaStreamDestroy(st);
                staged.rollback();
                err_out = std::string("delta prefetch read tr3.kmix: ") + rerr_buf;
                return false;
            }
        }
        std::vector<ptab::ExpertDelta> deltas(static_cast<size_t>(rows));
        for (int e = 0; e < rows; ++e)
        {
            uint8_t ko[16], kn[16], ad[16], dr[16];
            uint32_t nk = 0, na = 0, nd = 0;
            if (g6l_delta_halves(delta_handle, static_cast<uint32_t>(lyr), static_cast<uint32_t>(e),
                                 ko, kn, &nk, ad, &na, dr, &nd, rerr_buf, sizeof rerr_buf))
            {
                cudaStreamDestroy(st);
                staged.rollback();
                err_out = std::string("delta prefetch halves: ") + rerr_buf;
                return false;
            }
            ptab::ExpertDelta& d = deltas[static_cast<size_t>(e)];
            d.row = e;
            for (uint32_t i = 0; i < nk; ++i) d.kept.push_back({ko[i], kn[i]});
            for (uint32_t i = 0; i < na; ++i) d.added.push_back(ad[i]);
            for (uint32_t i = 0; i < nd; ++i) d.dropped.push_back(dr[i]);
        }
        const auto oit = current.tables.find(lyr);
        const ptab::Table* old_t = oit != current.tables.end() ? oit->second.get() : nullptr;
        auto dst_t = std::make_shared<ptab::Table>();
        ptab::TableDeltaResult res;
        if (const char* pe = ptab::stage_delta(old_t, h, k, rows, target_eb, deltas, *staged.direct.pool,
                                               sp.fc1->offset, sp.fc2->offset, route, *dst_t, res, st))
        {
            cudaStreamDestroy(st);
            staged.rollback();
            err_out = std::string("delta prefetch stage layer ") + std::to_string(lyr) + ": " + pe;
            return false;
        }
        staged.direct.expert_bytes += dst_t->slab_bytes;
        staged.direct.tables[lyr] = std::move(dst_t);
        staged.added_slabs.insert(staged.added_slabs.end(), res.added_slabs.begin(), res.added_slabs.end());
        staged.dropped_slabs.insert(staged.dropped_slabs.end(), res.dropped_slabs.begin(), res.dropped_slabs.end());
        staged.kept_pieces += res.kept_pieces;
        staged.added_pieces += res.added_pieces;
        staged.dropped_pieces += res.dropped_pieces;
        staged.kept_bytes += res.kept_bytes;
        staged.added_bytes += res.added_bytes;
        staged.dropped_bytes += res.dropped_bytes;
    }
    cudaStreamDestroy(st);

    if (!route.finalize(err_out))
    {
        staged.rollback();
        return false;
    }
    const auto exts_p = ptab::route_extents(route);
    std::vector<g6l_extent> keep;
    keep.reserve(exts_p.size());
    for (const auto& [off, nb] : exts_p) keep.push_back(g6l_extent{off, nb});

    RoutedRsSink sink;
    sink.route = &route;
    const g6l_sink s{&sink, &RoutedRsSink::alloc, &RoutedRsSink::release, &RoutedRsSink::copy};
    g6l_stats stats{};
    if (g6l_plan_load_ranges(plan_handle, keep.data(), keep.size(), &s, &stats, rerr_buf, sizeof rerr_buf))
    {
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        staged.rollback();
        err_out = std::string("delta prefetch load: ") + rerr_buf;
        return false;
    }
    for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
    fprintf(stderr, "# rank %d: EL3 delta prefetch: kept %zu pieces (%.2f GB), added %zu pieces (%.2f GB), "
            "dropped %zu pieces (%.2f GB), dense %.2f GB, read %.2f GB in %.2f s\n",
            rank, staged.kept_pieces, staged.kept_bytes / 1e9, staged.added_pieces, staged.added_bytes / 1e9,
            staged.dropped_pieces, staged.dropped_bytes / 1e9, dense_bytes / 1e9, stats.read_bytes / 1e9, stats.seconds);
    return true;
}

// The Rust loader's sink: pinned staging buffers and one stream per worker into the device image. Each chunk is summed
// in staging for the weight check (weight_check.cuh).
struct RsSink
{
    uint8_t* dev = nullptr;
    std::mutex mu;
    std::vector<cudaStream_t> streams;
    wcheck::Loaded sums;
    bool check = false;

    static void* alloc(void*, size_t n)
    {
        void* p = nullptr;
        return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
    }
    static void release(void*, void* p) { cudaFreeHost(p); }
    static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n)
    {
        RsSink& s = *(RsSink*)ctx;
        cudaStream_t st;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            while ((int)s.streams.size() <= w)
            {
                cudaStream_t x;
                if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return 2;
                s.streams.push_back(x);
            }
            st = s.streams[w];
        }
        if (s.check) s.sums.add(off, src, n);
        if (cudaMemcpyAsync(s.dev + off, src, n, cudaMemcpyHostToDevice, st) != cudaSuccess) return 3;
        return cudaStreamSynchronize(st) == cudaSuccess ? 0 : 4;
    }
};

// pack -> one device image through the Rust loader; load(sink, stats, err) runs the load
template <class F> void rs_fill(const g6::Pack& pack, g6::DevImage& img, int rank, const char* what, F load)
{
    RsSink sink;
    void* dev = nullptr;
    // GR8_IMG_VMM=1 (A/B): the image as cuMemCreate chunks of the recommended granularity mapped into one range
    // (engine/vmm.h), =N: chunks of N MB. Same bytes at the same offsets; only the physical
    // backing (the GPU's page size) can differ. Falls back to cudaMalloc, with a line saying why.
    static const long vmm = (long)knob::img_vmm();
    void (*release)(void*) = nullptr;
    if (vmm > 0)
    {
        int d = 0;
        size_t gmin = 0, grec = 0;
        cudaGetDevice(&d);
        g6vmm::granularity(d, gmin, grec);
        std::string why;
        const size_t chunk = vmm > 1 ? (size_t)vmm << 20 : 0;
        if ((dev = g6vmm::alloc(pack.bytes(), chunk, why))) release = [](void* p) { g6vmm::release(p); };
        fprintf(stderr, "# rank %d: GR8_IMG_VMM=%ld: %s (granularity min %zu KB, recommended %zu KB)\n", rank, vmm,
                dev ? ("image in chunks of " + std::to_string((chunk ? (chunk + gmin - 1) / gmin * gmin : grec) >> 20) + " MB").c_str()
                    : ("cudaMalloc instead: " + why).c_str(), gmin >> 10, grec >> 10);
    }
    if (!dev && cudaMalloc(&dev, pack.bytes()) != cudaSuccess) die("cudaMalloc of " + std::to_string(pack.bytes()) + " bytes failed");
    sink.dev = (uint8_t*)dev;
    sink.check = wcheck::enabled();
    if (sink.check) sink.sums.begin(pack.bytes());
    const g6l_sink s{&sink, &RsSink::alloc, &RsSink::release, &RsSink::copy};
    g6l_stats st{};
    char err[1024] = "";
    if (load(&s, &st, err, sizeof err)) die(std::string("Rust loader: ") + err);
    for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
    img.adopt(pack, (uint8_t*)dev, pack.bytes(), release);
    fprintf(stderr, "# rank %d: %s %.2f GB -> device image (Rust loader, %s, %u threads, %llu ranges, %.2f GB read)"
            " in %.1f s (%.2f GB/s)\n", rank, what, st.image_bytes / 1e9, st.mode ? "read" : "mmap", st.threads,
            (unsigned long long)st.ranges, st.read_bytes / 1e9, st.seconds, st.image_bytes / 1e9 / st.seconds);
    if (sink.check) wcheck::record(what, pack, (const uint8_t*)dev, sink.sums, rank);
}

// The pack -> one device image with the Rust loader: parallel reads, no checksum (the files are checked offline).
// GR8_LOADER=cpp, or a pack streamed on stdin, takes the C++ path instead.
inline bool rs_load_image(g6::Pack& pack, g6::DevImage& img, const std::string& file, int rank,
                          DirectPtab* direct = nullptr)
{
    if (knob::loader() == "cpp" || file == "-") return false;
    std::vector<g6l_extent> ext;
    for (const g6::Entry& e : pack.entries()) ext.push_back({e.offset, e.nbytes});
#if !G6_MODEL_DS41
    if (direct && ptab::mode_env() == ptab::CONTIG && !wcheck::enabled())
    {
        FILE* fp = fopen(file.c_str(), "rb");
        if (!fp) die(file + ": " + strerror(errno));
        rs_fill_direct_ptab(pack, img, *direct, rank, "pack",
            [&](uint64_t off, uint64_t n, void* dst) {
                if (fseeko(fp, static_cast<off_t>(off), SEEK_SET) != 0 || fread(dst, 1, n, fp) != n)
                    die(file + ": short read of tr3 metadata");
            },
            [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
                return g6l_load(file.c_str(), pack.bytes(), ext.data(), ext.size(), "", "", s, st, err, n);
            });
        fclose(fp);
        return true;
    }
#else
    (void)direct;
#endif
    rs_fill(pack, img, rank, "pack", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_load(file.c_str(), pack.bytes(), ext.data(), ext.size(), "", "", s, st, err, n);
    });
    return true;
}

// The planned manifest of a rank image: a new file of this process in $TMPDIR (default /tmp), made with mkstemps, so
// another account on the box can neither hold the name nor swap the file between the loader's write and our read.
// The caller removes it once it has been read.
inline std::string planned_manifest(int rank, const char* what)
{
    const std::string sfx = ".manifest.json";
    std::string p = knob::tmpdir() + "/g6m4-rank" + std::to_string(rank) + "." + what +
                    ".XXXXXX" + sfx;
    const int fd = mkstemps(&p[0], (int)sfx.size());
    if (fd < 0) die(p + ": " + strerror(errno));
    close(fd);
    return p;
}

// The base pack with its routed experts replaced by exlr8 v2 experts read from the raw width files (--v2): the Rust
// loader plans the image from the base manifest, the index slices and the width manifest, and writes the image's
// manifest to $TMPDIR (default /tmp) for `pack` to read. Its sha256 fields are zero, so only this loader loads it.
inline void rs_load_v2(g6::Pack& pack, g6::DevImage& img, const std::string& manifest, const std::string& file, int rank,
                       const V2Src& v2, DirectPtab* direct = nullptr)
{
    const std::string planned = planned_manifest(rank, "v2");
    char err[1024] = "";
    void* h = g6l_v2_plan(manifest.c_str(), file.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(),
                          planned.c_str(), err, sizeof err);
    if (!h) { unlink(planned.c_str()); die(std::string("v2 plan: ") + err); }
    std::string e;
    const bool opened = pack.open_manifest(planned, e);
    unlink(planned.c_str());
    if (!opened) die(planned + ": " + e);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
#if !G6_MODEL_DS41
    if (direct && ptab::mode_env() == ptab::CONTIG && !wcheck::enabled())
    {
        rs_fill_direct_ptab(pack, img, *direct, rank, "base pack + v2 files",
            [&](uint64_t off, uint64_t n, void* dst) {
                char rerr[1024] = "";
                if (g6l_plan_read(h, off, n, dst, rerr, sizeof rerr)) die(std::string("v2 plan read: ") + rerr);
            },
            [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
                return g6l_plan_load(h, s, st, err, n);
            });
        g6l_plan_free(h);
        return;
    }
#else
    (void)direct;
#endif
    rs_fill(pack, img, rank, "base pack + v2 files", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_plan_load(h, s, st, err, n);
    });
    g6l_plan_free(h);
}

// Plans rank `rank`'s image from the release alone (--v2-dense) and opens its manifest in pack. Returns the plan handle.
inline void* rs_plan_release(g6::Pack& pack, int rank, const V2Src& v2)
{
    const std::string planned = planned_manifest(rank, "release");
    char err[1024] = "";
#if G6_MODEL_DS41
    // DeepSeek-V4.1-Flash-1017: --v2-dense names the exlr8 native release; the planner also writes the Engram tables'
    // runs, to GR8_DS_ENGRAM_PREFIX.engram.json for the Engram boot, or to a file it removes again
    const auto eprefix = knob::ds_engram_prefix();
    const std::string engram = eprefix ? *eprefix + ".engram.json" : planned_manifest(rank, "engram");
    void* h = g6l_ds41_plan(v2.dense.c_str(), g6nr::NR, rank, planned.c_str(), engram.c_str(), err, sizeof err);
    if (!eprefix) unlink(engram.c_str());
#else
    void* h = g6l_release_plan(v2.dense.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(), g6nr::NR, rank,
                               v2.qo, v2.trellis, v2.layout.c_str(), planned.c_str(), err, sizeof err);
#endif
    if (!h) { unlink(planned.c_str()); die(std::string("release plan: ") + err); }
    std::string e;
    const bool opened = pack.open_manifest(planned, e);
    unlink(planned.c_str());
    if (!opened) die(planned + ": " + e);
    return h;
}

inline void rs_load_release(g6::Pack& pack, g6::DevImage& img, int rank, const V2Src& v2, DirectPtab* direct = nullptr)
{
    void* h = rs_plan_release(pack, rank, v2);
#if !G6_MODEL_DS41
    if (direct && ptab::mode_env() == ptab::CONTIG && !wcheck::enabled())
    {
        rs_fill_direct_ptab(pack, img, *direct, rank, "exlr8 release",
            [&](uint64_t off, uint64_t n, void* dst) {
                char rerr[1024] = "";
                if (g6l_plan_read(h, off, n, dst, rerr, sizeof rerr)) die(std::string("release plan read: ") + rerr);
            },
            [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
                return g6l_plan_load(h, s, st, err, n);
            });
        g6l_plan_free(h);
        return;
    }
#else
    (void)direct;
#endif
    rs_fill(pack, img, rank, "exlr8 release", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_plan_load(h, s, st, err, n);
    });
    g6l_plan_free(h);
}

// m4sim: one planned image per simulated rank, its tensors read from the release one at a time on first use. Tensors
// with the same bytes share one device copy (the replicated ones), as DevStore does for packs.
struct PlanStore
{
    std::vector<void*> plans;
    std::map<std::string, void*> by_name, by_content;
    size_t bytes = 0, shared_bytes = 0;

    const void* get(int r, const g6::Entry& e)
    {
        void*& got = by_name[std::to_string(r) + ":" + e.name];
        if (!got) got = read(r, e);
        return got;
    }
    void* read(int r, const g6::Entry& e)
    {
        std::vector<uint8_t> h(e.nbytes);
        char err[1024] = "";
        if (e.nbytes && g6l_plan_read(plans[r], e.offset, e.nbytes, h.data(), err, sizeof err)) die(e.name + ": " + err);
        uint64_t x = 0xcbf29ce484222325ull;       // FNV-1a over the bytes, plus the size, as the sharing key
        for (uint8_t b : h) x = (x ^ b) * 0x100000001b3ull;
        const std::string key = std::to_string(x) + ":" + std::to_string(e.nbytes);
        auto it = by_content.find(key);
        if (it != by_content.end()) { shared_bytes += e.nbytes; return it->second; }
        void* d;
        G6RT_CK(cudaMalloc(&d, e.nbytes ? e.nbytes : 1));
        G6RT_CK(g6h2d::copy(d, h.data(), e.nbytes));
        by_content[key] = d;
        bytes += e.nbytes;
        return d;
    }
};

inline Find plan_find(const g6::Pack& pack, PlanStore& store, int r)
{
    return [&pack, &store, r](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = store.get(r, *e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}
#endif

// manifest + pack stream ("-" = stdin) -> device image
inline void load_image(g6::Pack& pack, g6::DevImage& img, const std::string& manifest, const std::string& file, int rank,
                       bool hash = true, const V2Src& v2 = {}, DirectPtab* direct = nullptr)
{
#if G6_MODEL_DS41
    if (!v2.dir.empty() || !v2.dense.empty())   // DS41: --v2-dense RELEASE_DIR alone (rs_plan_release -> g6l_ds41_plan)
#else
    if (!v2.dir.empty())
#endif
    {
#ifdef G6_RS_LOADER
        if (!v2.dense.empty()) return rs_load_release(pack, img, rank, v2, direct);
        return rs_load_v2(pack, img, manifest, file, rank, v2, direct);
#else
        die("--v2 needs the Rust loader (scripts/build.sh links it when cargo is installed)");
#endif
    }
    std::string err;
    if (!pack.open_manifest(manifest, err)) die(err);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
#ifdef G6_RS_LOADER
    if (rs_load_image(pack, img, file, rank, direct)) return;
#endif
    g6::LoadOpts lo;
    g6::LoadStats st;
    if (!hash) lo.hash_threads = 0;
    if (!img.load(pack, file, lo, st, err)) die(err);
    fprintf(stderr, "# rank %d: pack %.2f GB -> device image%s in", rank, st.bytes / 1e9, hash ? ", sha256-checked," : " (NO checksum)");
    fprintf(stderr, " %.1f s (%.2f GB/s; copy %.1f s, stall %.1f s)\n", st.seconds, st.gb_per_s, st.copy_seconds, st.stall_seconds);
}

inline void load_pack(g6::Pack& pack, const std::string& manifest, const std::string& file, int rank)
{
    std::string err;
    if (!pack.open_manifest(manifest, err)) die(err);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
    g6::LoadOpts lo;
    g6::LoadStats st;
    if (!pack.load(file, lo, st, err)) die(err);
    fprintf(stderr, "# rank %d: pack %.2f GB loaded + sha256-checked in %.1f s (%.2f GB/s)\n", rank, st.bytes / 1e9, st.seconds,
            st.gb_per_s);
}


}   // namespace g6rt
