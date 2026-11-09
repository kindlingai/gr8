#include "kvchunks.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <utility>
#include <vector>

// The driver calls below come through cudaGetDriverEntryPointByVersion (as engine/vmm.h), not linked with -lcuda: a
// binary that links libcuda.so.1 does not start where there is no driver (a build container), even with
// GR8_GLM53F_KV_CHUNKS=0, which never calls them.
namespace {
#define G6_KVC_DRIVER(X) X(cuMemCreate) X(cuMemMap) X(cuMemUnmap) X(cuMemRelease) X(cuMemSetAccess) \
    X(cuMemsetD8Async) X(cuMemGetAllocationGranularity) X(cuMemAddressReserve) X(cuMemAddressFree) \
    X(cuStreamSynchronize) X(cuCtxSynchronize)
struct Driver {
#define G6_KVC_PTR(f) decltype(&::f) f = nullptr;
    G6_KVC_DRIVER(G6_KVC_PTR)
#undef G6_KVC_PTR
    CUresult status = CUDA_ERROR_NOT_INITIALIZED;
};
// every entry point, or the first one missing (its status, CUDA_ERROR_NOT_FOUND)
const Driver& drv() {
    static Driver d;
    static std::once_flag once;
    std::call_once(once, [] {
        d.status = CUDA_SUCCESS;
#define G6_KVC_GET(f)                                                                                                  \
        if (d.status == CUDA_SUCCESS) {                                                                                \
            void* p = nullptr;                                                                                         \
            cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;                                    \
            if (cudaGetDriverEntryPointByVersion(#f, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||               \
                q != cudaDriverEntryPointSuccess || !p)                                                                \
                d.status = CUDA_ERROR_NOT_FOUND;                                                                       \
            else d.f = reinterpret_cast<decltype(d.f)>(p);                                                             \
        }
        G6_KVC_DRIVER(G6_KVC_GET)
#undef G6_KVC_GET
    });
    return d;
}
#undef G6_KVC_DRIVER
}

struct G6KvChunks {
    CUdeviceptr base = 0;
    size_t bytes = 0, reserved = 0, gran = 0;
    int device = 0;
    std::vector<CUmemGenericAllocationHandle> h;   // one per granule, 0 = not backed
    std::vector<uint8_t> on;
};

namespace {
CUmemAllocationProp prop(int device) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    return p;
}
int back(G6KvChunks* c, size_t g, CUstream s) {
    CUmemAllocationProp p = prop(c->device);
    CUmemGenericAllocationHandle h = 0;
    CUresult r = drv().cuMemCreate(&h, c->gran, &p, 0);
    if (r != CUDA_SUCCESS) return (int)r;
    const CUdeviceptr at = c->base + g * c->gran;
    r = drv().cuMemMap(at, c->gran, 0, h, 0);
    if (r != CUDA_SUCCESS) { drv().cuMemRelease(h); return (int)r; }
    CUmemAccessDesc a{};
    a.location = p.location;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = drv().cuMemSetAccess(at, c->gran, &a, 1);
    if (r == CUDA_SUCCESS) r = drv().cuMemsetD8Async(at, 0, c->gran, s);
    if (r != CUDA_SUCCESS) { drv().cuMemUnmap(at, c->gran); drv().cuMemRelease(h); return (int)r; }
    c->h[g] = h;
    c->on[g] = 1;
    return 0;
}
int drop(G6KvChunks* c, size_t g) {
    const CUdeviceptr at = c->base + g * c->gran;
    CUresult r = drv().cuMemUnmap(at, c->gran);
    if (r != CUDA_SUCCESS) return (int)r;
    // unmapped: the granule is gone from the region whatever the release says (a failed release leaks the handle)
    c->on[g] = 0;
    r = drv().cuMemRelease(std::exchange(c->h[g], 0));
    return (int)r;
}
}

namespace {
// the region reserved; every granule backed (all) or none
int create(G6KvChunks** out, size_t bytes, int device, bool all) {
    if (!out || !bytes) return (int)CUDA_ERROR_INVALID_VALUE;
    cudaFree(nullptr);   // the runtime's primary context, current on this thread
    if (drv().status != CUDA_SUCCESS) return (int)drv().status;
    G6KvChunks* c = new G6KvChunks;
    c->bytes = bytes;
    c->device = device;
    CUmemAllocationProp p = prop(device);
    CUresult r = drv().cuMemGetAllocationGranularity(&c->gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (r == CUDA_SUCCESS) {
        c->reserved = (bytes + c->gran - 1) / c->gran * c->gran;
        r = drv().cuMemAddressReserve(&c->base, c->reserved, c->gran, 0, 0);
    }
    if (r != CUDA_SUCCESS) { delete c; return (int)r; }
    const size_t n = c->reserved / c->gran;
    c->h.assign(n, 0);
    c->on.assign(n, 0);
    for (size_t g = 0; all && g < n; ++g) {
        int e = back(c, g, 0);
        if (e) { g6_kvc_free(c); return e; }
    }
    r = drv().cuStreamSynchronize(0);
    if (r != CUDA_SUCCESS) { g6_kvc_free(c); return (int)r; }
    *out = c;
    return 0;
}
}

extern "C" int g6_kvc_create(G6KvChunks** out, size_t bytes, int device) { return create(out, bytes, device, true); }

extern "C" int g6_kvc_create_masked(G6KvChunks** out, size_t bytes, int device, size_t page_bytes, uint32_t chunk_pages,
                                    uint32_t place, const uint8_t* online, uint32_t chunks) {
    G6KvChunks* c = nullptr;
    int e = create(&c, bytes, device, false);
    if (!e) e = g6_kvc_sync(c, page_bytes, chunk_pages, place, online, chunks, nullptr, nullptr, nullptr);
    if (e) { g6_kvc_free(c); return e; }
    *out = c;
    return 0;
}

extern "C" int g6_kvc_zero(G6KvChunks* c, void* stream) {
    if (!c) return (int)CUDA_ERROR_INVALID_VALUE;
    // runs of backed granules
    for (size_t g = 0; g < c->on.size();) {
        if (!c->on[g]) { ++g; continue; }
        size_t e = g;
        while (e < c->on.size() && c->on[e]) ++e;
        CUresult r = drv().cuMemsetD8Async(c->base + g * c->gran, 0, (e - g) * c->gran, (CUstream)stream);
        if (r != CUDA_SUCCESS) return (int)r;
        g = e;
    }
    return 0;
}

extern "C" void g6_kvc_free(G6KvChunks* c) {
    if (!c) return;
    drv().cuCtxSynchronize();
    for (size_t g = 0; g < c->h.size(); ++g)
        if (c->on[g]) drop(c, g);
    if (c->base) drv().cuMemAddressFree(c->base, c->reserved);
    delete c;
}

extern "C" void* g6_kvc_ptr(const G6KvChunks* c) { return c ? (void*)c->base : nullptr; }
extern "C" size_t g6_kvc_bytes(const G6KvChunks* c) { return c ? c->bytes : 0; }
extern "C" size_t g6_kvc_granule(const G6KvChunks* c) { return c ? c->gran : 0; }
extern "C" size_t g6_kvc_device_granule(int device) {
    cudaFree(nullptr);
    if (drv().status != CUDA_SUCCESS) return 0;
    CUmemAllocationProp p = prop(device);
    size_t g = 0;
    return drv().cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS ? g : 0;
}
extern "C" size_t g6_kvc_backed(const G6KvChunks* c) {
    size_t n = 0;
    if (c)
        for (uint8_t o : c->on) n += o;
    return n * (c ? c->gran : 0);
}

extern "C" int g6_kvc_sync(G6KvChunks* c, size_t page_bytes, uint32_t chunk_pages, uint32_t place, const uint8_t* online,
                           uint32_t chunks, void* stream, size_t* mapped, size_t* released) {
    if (!c || !page_bytes || !chunk_pages || (!online && chunks) || chunks != (place + chunk_pages - 1) / chunk_pages)
        return (int)CUDA_ERROR_INVALID_VALUE;
    const size_t n = c->h.size();
    std::vector<uint8_t> need(n, 0);
    // pages from place on: always backed
    for (size_t g = (size_t)place * page_bytes / c->gran; (size_t)place * page_bytes < c->bytes && g < n; ++g) need[g] = 1;
    for (uint32_t k = 0; k < chunks; ++k) {
        if (!online[k]) continue;
        const uint64_t lo = (uint64_t)k * chunk_pages, hi = lo + chunk_pages < place ? lo + chunk_pages : place;
        const size_t blo = lo * page_bytes, bhi = hi * page_bytes;
        if (blo >= c->bytes || blo == bhi) continue;
        const size_t end = bhi < c->bytes ? bhi : c->bytes;
        for (size_t g = blo / c->gran; g * c->gran < end; ++g) need[g] = 1;
    }
    CUstream s = (CUstream)stream;
    size_t m = 0, d = 0;
    if (mapped) *mapped = 0;
    if (released) *released = 0;
    // nothing on any stream may still read a granule that goes
    CUresult r = drv().cuCtxSynchronize();
    if (r != CUDA_SUCCESS) return (int)r;
    for (size_t g = 0; g < n; ++g) {
        if (need[g] && !c->on[g]) {
            int e = back(c, g, s);
            if (e) return e;   // what changed so far stays changed: g6_kvc_backed says what is backed
            m += c->gran;
            if (mapped) *mapped = m;
        } else if (!need[g] && c->on[g]) {
            int e = drop(c, g);
            if (!c->on[g]) d += c->gran;   // unmapped, even when the release failed
            if (released) *released = d;
            if (e) return e;
        }
    }
    return (int)drv().cuStreamSynchronize(s);
}
