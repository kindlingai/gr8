#pragma once
#include "kernel.cuh"

namespace sweep_variant {
inline const void* kernel(const ds41a::AttnP& p)
{
    if (p.floor != 0) return (const void*)fragdb_detail::k_attn_fragdb_t<true, true>;
    if (p.TK != 0) return (const void*)fragdb_detail::k_attn_fragdb_t<false, true>;
    return (const void*)fragdb_detail::k_attn_fragdb_t<false, false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p)
{
    if (p.floor != 0) return "k_attn_fragdb_tILb1ELb1EE";
    if (p.TK != 0) return "k_attn_fragdb_tILb0ELb1EE";
    return "k_attn_fragdb_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p)
{
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p, cudaStream_t stream)
{
    const dim3 grid(p.n, p.heads / 16);
    if (p.floor != 0)
        fragdb_detail::k_attn_fragdb_t<true, true><<<grid, ds41a::AT_THREADS, dynamic_smem(p), stream>>>(p);
    else if (p.TK != 0)
        fragdb_detail::k_attn_fragdb_t<false, true><<<grid, ds41a::AT_THREADS, dynamic_smem(p), stream>>>(p);
    else
        fragdb_detail::k_attn_fragdb_t<false, false><<<grid, ds41a::AT_THREADS, dynamic_smem(p), stream>>>(p);
}
} // namespace sweep_variant
