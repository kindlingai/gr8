// Display-memory KV backing, with an ordinary device-memory tail when a lease ends mid-binding.
#pragma once
#include "dispram_client.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <map>
#include <mutex>
#include <type_traits>

namespace g6dispram {
struct VmmApi {
    decltype(&cuMemCreate) create=nullptr;
    decltype(&cuMemRelease) release=nullptr;
    decltype(&cuMemImportFromShareableHandle) import_fd=nullptr;
    decltype(&cuMemAddressReserve) reserve=nullptr;
    decltype(&cuMemAddressFree) free_va=nullptr;
    decltype(&cuMemMap) map=nullptr;
    decltype(&cuMemUnmap) unmap=nullptr;
    decltype(&cuMemSetAccess) access=nullptr;
    decltype(&cuMemGetAllocationGranularity) gran=nullptr;
    bool ok=false;
};
inline const VmmApi& vmm_api() {
    static const VmmApi api=[] {
        VmmApi a;
        auto get=[](const char* name,auto& fn) {
            void* p=nullptr; cudaDriverEntryPointQueryResult q;
            if (cudaGetDriverEntryPointByVersion(name,&p,12000,cudaEnableDefault,&q)!=cudaSuccess || q!=cudaDriverEntryPointSuccess) return false;
            fn=reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p); return true;
        };
        a.ok=get("cuMemCreate",a.create) && get("cuMemRelease",a.release) &&
            get("cuMemImportFromShareableHandle",a.import_fd) && get("cuMemAddressReserve",a.reserve) &&
            get("cuMemAddressFree",a.free_va) && get("cuMemMap",a.map) && get("cuMemUnmap",a.unmap) &&
            get("cuMemSetAccess",a.access) && get("cuMemGetAllocationGranularity",a.gran);
        return a;
    }();
    return api;
}
struct CudaBackend {
    using Err=cudaError_t;
    static constexpr Err ok_err() { return cudaSuccess; }
    static constexpr Err invalid_err() { return cudaErrorInvalidValue; }
    CUmemAllocationProp prop{};
    bool granularity(size_t* gmin) {
        int dev=0; const auto& vmm=vmm_api();
        prop={}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
        if (!vmm.ok || cudaGetDevice(&dev)!=cudaSuccess) return false;
        prop.location.id=dev;
        return vmm.gran(gmin,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM)==CUDA_SUCCESS && *gmin>0;
    }
    bool map_span(int lease_fd,const SpanPlan& plan,size_t gran,int fail_after,size_t live_count,uintptr_t* out_va) {
        const auto& vmm=vmm_api();
        CUmemGenericAllocationHandle hc{}; CUdeviceptr va=0;
        bool carve_mapped=false; size_t tail_mapped=0;
        CUresult cr=CUDA_ERROR_OUT_OF_MEMORY;
        if (fail_after<0 || static_cast<int>(live_count)<fail_after)
            cr=vmm.import_fd(&hc,reinterpret_cast<void*>(static_cast<uintptr_t>(lease_fd)),CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        ::close(lease_fd);
        if (cr==CUDA_SUCCESS) {
            cr=vmm.reserve(&va,plan.total_bytes,gran,0,0);
            if (cr==CUDA_SUCCESS) { cr=vmm.map(va,plan.carve_bytes,0,hc,0); carve_mapped=cr==CUDA_SUCCESS; }
            vmm.release(hc);
        }
        for (size_t o=plan.carve_bytes; cr==CUDA_SUCCESS && o<plan.total_bytes; o+=gran) {
            CUmemGenericAllocationHandle ht{};
            cr=vmm.create(&ht,gran,&prop,0);
            if (cr==CUDA_SUCCESS) {
                cr=vmm.map(va+o,gran,0,ht,0); vmm.release(ht);
                if (cr==CUDA_SUCCESS) tail_mapped+=gran;
            }
        }
        CUmemAccessDesc access{}; access.location=prop.location; access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cr==CUDA_SUCCESS) {
            cr=vmm.access(va,plan.total_bytes,&access,1);
            if (cr!=CUDA_SUCCESS) {
                cr=vmm.access(va,plan.carve_bytes,&access,1);
                for (size_t o=plan.carve_bytes;cr==CUDA_SUCCESS && o<plan.total_bytes;o+=gran) cr=vmm.access(va+o,gran,&access,1);
            }
        }
        if (cr==CUDA_SUCCESS) { *out_va=static_cast<uintptr_t>(va); return true; }
        if (carve_mapped) vmm.unmap(va,plan.carve_bytes);
        for (size_t o=0;o<tail_mapped;o+=gran) vmm.unmap(va+plan.carve_bytes+o,gran);
        if (va) vmm.free_va(va,plan.total_bytes);
        return false;
    }
    void sync() { (void)cudaDeviceSynchronize(); }
    void unmap_span(uintptr_t va,const Span& sp) {
        const auto& vmm=vmm_api();
        if (sp.carve) vmm.unmap(static_cast<CUdeviceptr>(va),sp.carve);
        for (size_t o=0;o+sp.carve<sp.total;o+=sp.gran) vmm.unmap(static_cast<CUdeviceptr>(va+sp.carve+o),sp.gran);
        vmm.free_va(static_cast<CUdeviceptr>(va),sp.total);
    }
    Err fallback_alloc(void** p,size_t n) { return cudaMalloc(p,n); }
    Err fallback_free(void* p) { return cudaFree(p); }
};
inline Session<CudaBackend>& state() { static CudaBackend b; static Session<CudaBackend> s(b); return s; }
inline size_t claimed_bytes() { return state().claimed_bytes(); }
inline size_t mapped_bytes() { return state().mapped_bytes(); }
inline size_t claimed_spans() { return state().claimed_spans(); }
inline bool has_open_conn_for_test() { return state().has_open_conn_for_test(); }
inline void set_fail_vmm_after_spans_for_test(int n) { state().set_fail_vmm_after_spans_for_test(n); }
inline void set_timeout_ms_for_test(int ms) { state().set_timeout_ms_for_test(ms); }
inline cudaError_t alloc_kv(void** p,size_t n) { return state().alloc_kv(p,n); }
inline cudaError_t free_kv(void* p) { return state().free_kv(p); }
inline void release_all() { state().release_all(); }
}
