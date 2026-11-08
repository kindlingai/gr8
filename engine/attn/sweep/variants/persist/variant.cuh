#include "kernel.cuh"
namespace sweep_variant {
// Set outside timing by init; no device query, allocation, or synchronization
// is part of launch. The sweep initializes each case on its selected device.
inline int sm_count = 1;
inline const void* kernel(const ds41a::AttnP& p) {
    if (p.floor) return (const void*)sweep_persist::k_persist_t<true,true>;
    return p.TK ? (const void*)sweep_persist::k_persist_t<false,true>
                : (const void*)sweep_persist::k_persist_t<false,false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    return p.floor ? "k_persist_tILb1ELb1EE" : p.TK ? "k_persist_tILb0ELb1EE" : "k_persist_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    int device;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return e;
    e = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (e != cudaSuccess) return e;
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p, cudaStream_t stream) {
    dim3 grid(p.n < sm_count ? p.n : sm_count, p.heads / 16);
    if (p.floor) sweep_persist::k_persist_t<true,true><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
    else if (p.TK) sweep_persist::k_persist_t<false,true><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
    else sweep_persist::k_persist_t<false,false><<<grid,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,stream>>>(p);
}
}
