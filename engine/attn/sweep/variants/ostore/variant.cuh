#include "kernel.cuh"

namespace sweep_variant {
inline const void* kernel(const ds41a::AttnP& p) {
    if (p.floor) return (const void*)sweep_ostore::k_attn_ostore_t<true, true>;
    return p.TK ? (const void*)sweep_ostore::k_attn_ostore_t<false, true>
                : (const void*)sweep_ostore::k_attn_ostore_t<false, false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    return p.floor ? "k_attn_ostore_tILb1ELb1EE"
                  : p.TK ? "k_attn_ostore_tILb0ELb1EE"
                         : "k_attn_ostore_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p, cudaStream_t stream) {
    dim3 grid(p.n, p.heads / 16);
    if (p.floor)
        sweep_ostore::k_attn_ostore_t<true, true><<<grid, ds41a::AT_THREADS, ds41a::AT_SMEM_CPA, stream>>>(p);
    else if (p.TK)
        sweep_ostore::k_attn_ostore_t<false, true><<<grid, ds41a::AT_THREADS, ds41a::AT_SMEM_CPA, stream>>>(p);
    else
        sweep_ostore::k_attn_ostore_t<false, false><<<grid, ds41a::AT_THREADS, ds41a::AT_SMEM_CPA, stream>>>(p);
}
} // namespace sweep_variant
