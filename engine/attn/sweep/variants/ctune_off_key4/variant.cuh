#include "kernel.cuh"
// Our served cp.async kernel; shared baseline also used by unsupported cases.
namespace sweep_variant {
inline const void* kernel(const ds41a::AttnP& p) {
    if (p.floor) return (const void*)detail_ctune_off_key4::k_ctune_off_key4<true, true>;
    return p.TK ? (const void*)detail_ctune_off_key4::k_ctune_off_key4<false, true> : (const void*)detail_ctune_off_key4::k_ctune_off_key4<false, false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    return p.floor ? "k_ctune_off_key4ILb1ELb1EE" : p.TK ? "k_ctune_off_key4ILb0ELb1EE" : "k_ctune_off_key4ILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP&) { return ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p, cudaStream_t st) {
    dim3 g(p.n, p.heads / 16);
    if (p.floor) detail_ctune_off_key4::k_ctune_off_key4<true,true><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else if (p.TK) detail_ctune_off_key4::k_ctune_off_key4<false,true><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else detail_ctune_off_key4::k_ctune_off_key4<false,false><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
}
}
