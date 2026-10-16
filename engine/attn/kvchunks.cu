#include "kvchunks.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    CUresult r = cuMemCreate(&h, c->gran, &p, 0);
    if (r != CUDA_SUCCESS) return (int)r;
    const CUdeviceptr at = c->base + g * c->gran;
    r = cuMemMap(at, c->gran, 0, h, 0);
    if (r != CUDA_SUCCESS) { cuMemRelease(h); return (int)r; }
    CUmemAccessDesc a{};
    a.location = p.location;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(at, c->gran, &a, 1);
    if (r == CUDA_SUCCESS) r = cuMemsetD8Async(at, 0, c->gran, s);
    if (r != CUDA_SUCCESS) { cuMemUnmap(at, c->gran); cuMemRelease(h); return (int)r; }
    c->h[g] = h;
    c->on[g] = 1;
    return 0;
}
int drop(G6KvChunks* c, size_t g) {
    const CUdeviceptr at = c->base + g * c->gran;
    CUresult r = cuMemUnmap(at, c->gran);
    if (r == CUDA_SUCCESS) r = cuMemRelease(c->h[g]);
    if (r != CUDA_SUCCESS) return (int)r;
    c->h[g] = 0;
    c->on[g] = 0;
    return 0;
}
}

extern "C" int g6_kvc_create(G6KvChunks** out, size_t bytes, int device) {
    if (!out || !bytes) return (int)CUDA_ERROR_INVALID_VALUE;
    cudaFree(nullptr);   // the runtime's primary context, current on this thread
    G6KvChunks* c = new G6KvChunks;
    c->bytes = bytes;
    c->device = device;
    CUmemAllocationProp p = prop(device);
    CUresult r = cuMemGetAllocationGranularity(&c->gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (r == CUDA_SUCCESS) {
        c->reserved = (bytes + c->gran - 1) / c->gran * c->gran;
        r = cuMemAddressReserve(&c->base, c->reserved, c->gran, 0, 0);
    }
    if (r != CUDA_SUCCESS) { delete c; return (int)r; }
    const size_t n = c->reserved / c->gran;
    c->h.assign(n, 0);
    c->on.assign(n, 0);
    for (size_t g = 0; g < n; ++g) {
        int e = back(c, g, 0);
        if (e) { g6_kvc_free(c); return e; }
    }
    r = cuStreamSynchronize(0);
    if (r != CUDA_SUCCESS) { g6_kvc_free(c); return (int)r; }
    *out = c;
    return 0;
}

extern "C" void g6_kvc_free(G6KvChunks* c) {
    if (!c) return;
    cuCtxSynchronize();
    for (size_t g = 0; g < c->h.size(); ++g)
        if (c->on[g]) drop(c, g);
    if (c->base) cuMemAddressFree(c->base, c->reserved);
    delete c;
}

extern "C" void* g6_kvc_ptr(const G6KvChunks* c) { return c ? (void*)c->base : nullptr; }
extern "C" size_t g6_kvc_granule(const G6KvChunks* c) { return c ? c->gran : 0; }
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
    // nothing may still read a granule that goes
    CUresult r = cuStreamSynchronize(s);
    if (r != CUDA_SUCCESS) return (int)r;
    size_t m = 0, d = 0;
    for (size_t g = 0; g < n; ++g) {
        if (need[g] && !c->on[g]) {
            int e = back(c, g, s);
            if (e) return e;
            m += c->gran;
        } else if (!need[g] && c->on[g]) {
            int e = drop(c, g);
            if (e) return e;
            d += c->gran;
        }
    }
    r = cuStreamSynchronize(s);
    if (mapped) *mapped = m;
    if (released) *released = d;
    return (int)r;
}
