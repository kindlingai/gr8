#pragma once
#include "memory_guard.cuh"
// Variant-owned mutable scratch uses the same guards as fixtures, without
// immutable hashing. Process-lifetime host registry avoids static destructor
// order issues in variant-owned RAII scratch objects.
namespace sweep_memory {
inline Registry& private_scratch() { static auto* r=new Registry; return *r; }
inline cudaError_t scratch_alloc(void** output,size_t bytes) {
    auto& r=private_scratch();AllocationScope scope(r);
    *output=allocate(bytes);r.describe(*output,"variant_private_scratch",false);
    r.snapshotted=true;return cudaSuccess;
}
inline cudaError_t scratch_free(void* p) {
    if(!p)return cudaSuccess;
    auto& r=private_scratch();
    for(auto it=r.allocations.begin();it!=r.allocations.end();++it)if(it->logical==p) {
        // Return errors to init callers; do not exit from a static destructor
        // after CUDA runtime teardown (the process context frees its memory).
        const auto e=cudaFree(static_cast<unsigned char*>(p)-redzone_bytes);
        if(e==cudaSuccess)r.allocations.erase(it);
        return e;
    }
    return cudaErrorInvalidValue;
}
inline size_t check_private_scratch(cudaStream_t st) {
    auto& r=private_scratch();r.snapshotted=true;return r.check(st);
}
}
