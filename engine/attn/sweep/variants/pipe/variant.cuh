// Built-in adapter over our 1ed4501c pipeline kernel.
namespace sweep_variant {
inline bool eligible(const ds41a::AttnP& p) { return p.TK && !p.floor && sweep_adapter::pipe_enabled(p); }
inline const void* kernel(const ds41a::AttnP& p) {
    if (eligible(p)) return (const void*)ds41a::k_attn_cpa_pipe_q<false>;
    if (p.floor) return (const void*)ds41a::k_attn_cpa_t<true,true>;
    return p.TK ? (const void*)ds41a::k_attn_cpa_t<false,true> : (const void*)ds41a::k_attn_cpa_t<false,false>;
}
inline const char* ptxas_key(const ds41a::AttnP& p) {
    return eligible(p) ? "k_attn_cpa_pipe_qILb0EE" : p.floor ? "k_attn_cpa_tILb1ELb1EE" : p.TK ? "k_attn_cpa_tILb0ELb1EE" : "k_attn_cpa_tILb0ELb0EE";
}
inline size_t dynamic_smem(const ds41a::AttnP& p) { return eligible(p) ? ds41a::AT_SMEM_PIPE : ds41a::AT_SMEM_CPA; }
inline cudaError_t init(const ds41a::AttnP& p) {
    return cudaFuncSetAttribute(kernel(p),cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_smem(p));
}
inline void launch(const ds41a::AttnP& p,cudaStream_t st) {
    dim3 g(p.n,p.heads/16);
    if (eligible(p)) ds41a::k_attn_cpa_pipe_q<false><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_PIPE,st>>>(p);
    else if (p.floor) ds41a::k_attn_cpa_t<true,true><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else if (p.TK) ds41a::k_attn_cpa_t<false,true><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
    else ds41a::k_attn_cpa_t<false,false><<<g,ds41a::AT_THREADS,ds41a::AT_SMEM_CPA,st>>>(p);
}
}
