// Compile unit for the DS41 attention B (indexer) kernels (engine/attn/ds41_idx.cuh): explicit instantiations of the templated launchers so the
// engine links one copy. The kernels themselves are header-only (a step driver or test may include the header directly).
#include "ds41_idx.cuh"

namespace ds41i
{
template void launch_topk<TOPK>(const bf16*, long, const int32_t*, int, int32_t*, int, int, cudaStream_t);
template void launch_topk<CAND_BLOCKS>(const bf16*, long, const int32_t*, int, int32_t*, int, int, cudaStream_t);
}
