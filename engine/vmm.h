// Device memory through the CUDA VMM API (cuMemCreate + cuMemMap into one reserved range) in place of one cudaMalloc:
// the weight image under GR8_IMG_VMM (runtime/packfind.h rs_fill). Same bytes, same offsets; only the physical backing
// differs: the range is built from chunks of `chunk` bytes (a multiple of the device's allocation granularity, 2 MB on
// GPUs with large pages), each a physical allocation of its own.
// The driver API comes through cudaGetDriverEntryPointByVersion (no -lcuda). Not for use inside a stream capture.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
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
// frees what alloc returned; false if p is not one of its ranges
inline bool release(void* p)
{
    std::pair<size_t, size_t> n;
    {
        std::lock_guard<std::mutex> lk(mu());
        auto it = spans().find(reinterpret_cast<uintptr_t>(p));
        if (it == spans().end()) return false;
        n = it->second;
        spans().erase(it);
    }
    cudaDeviceSynchronize();
    unmap(reinterpret_cast<CUdeviceptr>(p), n.first, n.second);
    api().free_va(reinterpret_cast<CUdeviceptr>(p), n.first);
    return true;
}

}   // namespace g6vmm
