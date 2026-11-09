#pragma once
// Copy-me template: a private copy of the served fc2 kernel, launched once per column group like down_group().
// Bit-exact to `off` (it is the same code). Edit kernel.cuh, keep this launch contract.
#include "kernel.cuh"
#define SWEEP_VARIANT_HAS_BLOCK_THREADS 1   // opt in to the occupancy columns (needs block_threads below)
namespace sweep_variant {
constexpr int kBlock = 4 * 32;   // W2 warps
inline const void* kernel(const sweep_adapter::Params&) { return (const void*)fc2_tmpl::k_fc2_tmpl<576, 5120, 4, 2, 8, 8, 4>; }
// Unique substring of the MANGLED ptxas entry name of the kernel above.
inline const char* ptxas_key(const sweep_adapter::Params&) { return "k_fc2_tmplILi576ELi5120ELi4ELi2ELi8ELi8ELi4E"; }
inline size_t dynamic_smem(const sweep_adapter::Params&) { return mxtc::fc2_smem<4, 8, 8>(); }
inline int block_threads(const sweep_adapter::Params&) { return kBlock; }
inline cudaError_t init(const sweep_adapter::Params& p) {
    return cudaFuncSetAttribute(kernel(p), cudaFuncAttributeMaxDynamicSharedMemorySize, int(dynamic_smem(p)));
}
inline void launch(const sweep_adapter::Params& p, cudaStream_t st) {
    for (int c = 0; c < sweep_adapter::kColGroups; ++c) {
        auto a = p.a; a.col_group = c;   // only ypart may be written; everything else is the immutable fixture
        fc2_tmpl::k_fc2_tmpl<576, 5120, 4, 2, 8, 8, 4><<<a.n_slots * a.gy2 * a.os2, kBlock, dynamic_smem(p), st>>>(a);
    }
}
}  // namespace sweep_variant
