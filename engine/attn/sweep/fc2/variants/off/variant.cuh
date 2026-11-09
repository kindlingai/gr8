#pragma once
// The served kernel, untouched: mxtc::k_fc2_tc<576,5120,4,2,8,8,4> launched once per column group (ds41_v2_ffn.cuh down_group()).
#define SWEEP_VARIANT_HAS_BLOCK_THREADS 1
namespace sweep_variant {
constexpr int kBlock = 4 * 32;   // W2 warps
inline const void* kernel(const sweep_adapter::Params&) { return (const void*)mxtc::k_fc2_tc<576, 5120, 4, 2, 8, 8, 4>; }
inline const char* ptxas_key(const sweep_adapter::Params&) { return "k_fc2_tcILi576ELi5120ELi4ELi2ELi8ELi8ELi4E"; }
inline size_t dynamic_smem(const sweep_adapter::Params&) { return mxtc::fc2_smem<4, 8, 8>(); }
inline int block_threads(const sweep_adapter::Params&) { return kBlock; }
inline cudaError_t init(const sweep_adapter::Params& p) {
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, int(dynamic_smem(p)));
}
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    for (int c = 0; c < sweep_adapter::kColGroups; ++c) {
        auto a = p.a; a.col_group = c;
        mxtc::k_fc2_tc<576, 5120, 4, 2, 8, 8, 4><<<a.n_slots * a.gy2 * a.os2, kBlock, dynamic_smem(p), st>>>(a);
    }
}
}  // namespace sweep_variant
