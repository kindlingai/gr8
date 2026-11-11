// Device memory through the CUDA VMM API (cuMemCreate + cuMemMap into one reserved range) in place of one cudaMalloc:
// the weight image under GR8_IMG_VMM (runtime/packfind.h rs_fill). Same bytes, same offsets; only the physical backing
// differs: the range is built from chunks of `chunk` bytes (a multiple of the device's allocation granularity, 2 MB on
// GPUs with large pages), each a physical allocation of its own. Under GR8_ATTIC_SOCK (engine/attic.h) one span of the
// image is a slice of the display carveout instead, and the rest a few large physical allocations (alloc_around).
// The driver API comes through cudaGetDriverEntryPointByVersion (no -lcuda). Not for use inside a stream capture.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace g6vmm
{

struct Api
{
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*free_va)(CUdeviceptr, size_t) = nullptr;
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (*gran)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    bool ok = false;
};
inline const Api& api()
{
    static Api a;
    static std::once_flag once;
    std::call_once(once, [] {
        auto get = [](const char* s, auto& fn) {
            void* p = nullptr;
            cudaDriverEntryPointQueryResult q;
            if (cudaGetDriverEntryPointByVersion(s, &p, 12000, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess) return false;
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
            return true;
        };
        a.ok = get("cuMemCreate", a.create) && get("cuMemRelease", a.release) && get("cuMemAddressReserve", a.reserve) &&
               get("cuMemAddressFree", a.free_va) && get("cuMemMap", a.map) && get("cuMemUnmap", a.unmap) &&
               get("cuMemSetAccess", a.access) && get("cuMemGetAllocationGranularity", a.gran);
    });
    return a;
}
inline CUmemAllocationProp prop(int dev)
{
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
// minimum and recommended granularity of device `dev`; 0, 0 without VMM support
inline void granularity(int dev, size_t& gmin, size_t& grec)
{
    gmin = grec = 0;
    if (!api().ok) return;             // without VMM support the granularity query fails: 0, 0
    const CUmemAllocationProp p = prop(dev);
    if (api().gran(&gmin, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) gmin = 0;
    if (api().gran(&grec, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) grec = gmin;
}

inline std::mutex& mu() { static std::mutex m; return m; }
inline std::map<uintptr_t, std::pair<size_t, size_t>>& spans() { static std::map<uintptr_t, std::pair<size_t, size_t>> s; return s; }   // base -> bytes, chunk
// cuMemUnmap chunk by chunk (one call per mapping, as the driver documents it)
inline void unmap(CUdeviceptr base, size_t n, size_t chunk)
{
    for (size_t o = 0; o < n; o += chunk) api().unmap(base + o, chunk);
}

// bytes of device memory on the current device as chunks of `chunk` bytes (0 = the recommended granularity; rounded
// up to a multiple of it), mapped contiguously at an address aligned to the chunk. nullptr + why on failure (nothing
// is left allocated then).
inline void* alloc(size_t bytes, size_t chunk, std::string& why)
{
    int dev = 0;
    cudaGetDevice(&dev);
    size_t gmin, grec;
    granularity(dev, gmin, grec);
    if (!gmin) { why = "no VMM support (cudaDevAttrVirtualMemoryManagementSupported / driver entry points)"; return nullptr; }
    if (!chunk) chunk = grec;
    chunk = (chunk + gmin - 1) / gmin * gmin;
    const size_t total = (bytes + chunk - 1) / chunk * chunk;
    const Api& a = api();
    CUdeviceptr base = 0;
    if (CUresult r = a.reserve(&base, total, chunk, 0, 0)) { why = "cuMemAddressReserve " + std::to_string(r); return nullptr; }
    const CUmemAllocationProp p = prop(dev);
    size_t done = 0;
    CUresult r = CUDA_SUCCESS;
    for (; done < total; done += chunk)
    {
        CUmemGenericAllocationHandle h;
        if ((r = a.create(&h, chunk, &p, 0))) { why = "cuMemCreate " + std::to_string(r); break; }
        r = a.map(base + done, chunk, 0, h, 0);
        a.release(h);                        // the mapping keeps the memory until cuMemUnmap
        if (r) { why = "cuMemMap " + std::to_string(r); break; }
    }
    CUmemAccessDesc d = {};
    d.location = p.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!r && a.access(base, total, &d, 1))       // one call over the range; else chunk by chunk
        for (size_t o = 0; o < total && !r; o += chunk)
            if ((r = a.access(base + o, chunk, &d, 1))) why = "cuMemSetAccess " + std::to_string(r);
    if (r)
    {
        unmap(base, done, chunk);
        a.free_va(base, total);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(mu());
    spans()[static_cast<uintptr_t>(base)] = {total, chunk};
    return reinterpret_cast<void*>(base);
}

// GR8_ATTIC_SOCK (engine/attic.h): base -> the offset and bytes of each mapping of an alloc_around range
inline std::map<uintptr_t, std::vector<std::pair<size_t, size_t>>>& maps()
{
    static std::map<uintptr_t, std::vector<std::pair<size_t, size_t>>> s;
    return s;
}
// the mappings of an alloc_around range (0 for any other pointer)
inline size_t mappings(void* p)
{
    std::lock_guard<std::mutex> lk(mu());
    auto it = maps().find(reinterpret_cast<uintptr_t>(p));
    return it == maps().end() ? 0 : it->second.size();
}

// A range of `bytes` like alloc's, except that [lo, hi) maps `h` (a handle the caller brings, of hi - lo bytes: a slice
// of the display carveout imported from its descriptor) in one mapping, and the rest is physical allocations cut at lo,
// hi and the multiples of `piece` (a power of two times the granularity; the range is reserved at that alignment). Few
// large allocations, not chunks of the granularity: on a GB10 each physical allocation costs the kernel ~25 KB of slab
// (an 81 GB image in 2 MB chunks: ~1 GiB of ordinary memory, half of what the slice frees), and 1 GiB pieces cost what
// one cudaMalloc does. lo and hi are multiples of the granularity. h is released here whatever happens (a mapping keeps
// its memory until cuMemUnmap). nullptr + why on failure (nothing is left mapped then).
inline void* alloc_around(size_t bytes, size_t lo, size_t hi, CUmemGenericAllocationHandle h, size_t piece, std::string& why)
{
    const Api& a = api();
    int dev = 0;
    cudaGetDevice(&dev);
    size_t gmin, chunk;
    granularity(dev, gmin, chunk);
    if (!gmin)
    {
        why = "no VMM support (cudaDevAttrVirtualMemoryManagementSupported / driver entry points)";
        if (a.release) a.release(h);
        return nullptr;
    }
    chunk = (chunk + gmin - 1) / gmin * gmin;
    const size_t total = (std::max(bytes, hi) + chunk - 1) / chunk * chunk;
    if (lo >= hi || lo % chunk || hi % chunk)
    {
        why = "the lent span is not whole " + std::to_string(chunk >> 10) + " KB chunks";
        a.release(h);
        return nullptr;
    }
    if (piece < chunk || piece % chunk || (piece & (piece - 1)))
    {
        why = "pieces of " + std::to_string(piece) + " bytes are not a power of two times the granularity";
        a.release(h);
        return nullptr;
    }
    CUdeviceptr base = 0;
    if (CUresult r = a.reserve(&base, total, piece, 0, 0))
    {
        why = "cuMemAddressReserve " + std::to_string(r);
        a.release(h);
        return nullptr;
    }
    std::vector<std::pair<size_t, size_t>> done;   // offset, bytes of each mapping made
    CUresult r = a.map(base + lo, hi - lo, 0, h, 0);
    a.release(h);
    if (r) why = "cuMemMap of the lent slice " + std::to_string(r);
    else done.push_back({lo, hi - lo});
    const CUmemAllocationProp p = prop(dev);
    for (size_t o = 0; !r && o < total;)
    {
        if (o == lo) { o = hi; continue; }
        const size_t n = std::min(o / piece * piece + piece, o < lo ? lo : total) - o;
        CUmemGenericAllocationHandle c;
        if ((r = a.create(&c, n, &p, 0))) { why = "cuMemCreate of " + std::to_string(n >> 20) + " MiB " + std::to_string(r); break; }
        r = a.map(base + o, n, 0, c, 0);
        a.release(c);
        if (r) { why = "cuMemMap " + std::to_string(r); break; }
        done.push_back({o, n});
        o += n;
    }
    CUmemAccessDesc d = {};
    d.location = p.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!r && (r = a.access(base, total, &d, 1))) why = "cuMemSetAccess " + std::to_string(r);
    if (r)
    {
        for (const auto& m : done) a.unmap(base + m.first, m.second);
        a.free_va(base, total);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(mu());
    spans()[static_cast<uintptr_t>(base)] = {total, piece};
    maps()[static_cast<uintptr_t>(base)] = std::move(done);
    return reinterpret_cast<void*>(base);
}

// frees what alloc or alloc_around returned; false if p is not one of their ranges
inline bool release(void* p)
{
    std::pair<size_t, size_t> n;
    std::vector<std::pair<size_t, size_t>> m;
    {
        std::lock_guard<std::mutex> lk(mu());
        auto it = spans().find(reinterpret_cast<uintptr_t>(p));
        if (it == spans().end()) return false;
        n = it->second;
        spans().erase(it);
        auto jt = maps().find(reinterpret_cast<uintptr_t>(p));
        if (jt != maps().end()) { m = std::move(jt->second); maps().erase(jt); }
    }
    cudaDeviceSynchronize();
    const CUdeviceptr base = reinterpret_cast<CUdeviceptr>(p);
    if (m.empty()) unmap(base, n.first, n.second);
    for (const auto& x : m) api().unmap(base + x.first, x.second);
    api().free_va(base, n.first);
    return true;
}

}   // namespace g6vmm
