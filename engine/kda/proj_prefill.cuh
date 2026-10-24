// Full-M f_a/g_a/b projection over the decode packed records. No activation requantization.
// Each CTA covers one output tile and NR rows; all 32 K-partitions and their reduction
// order are identical to the decode projection. The M dimension is a grid dimension,
// not a host loop. Larger row tiles reuse each decoded weight across more rows.
#pragma once
#include "kda.cuh"
namespace kda {
// Select the last tile's actual row count: no loads from nonexistent padded row tiles.
// This also keeps the decode reference's treatment of partial M independent of the full tile size.
template<class C, int NT>
__device__ __forceinline__ void tail_tile(const ProjPart& p, const u32* xf, float* y, int rows, int ld, int cta, float4 (&red)[32][32])
{
    if constexpr (NT > 1) {
        if (rows <= (NT - 1) * 8) { tail_tile<C, NT - 1>(p, xf, y, rows, ld, cta, red); return; }
    }
    gv_gemv_body<C, NT>(p.w, xf, p.K, p.ot, p.tpc, p.ks, rows, y, ld, gv::O_F32, red, cta);
}
template<int NT>
__global__ void __launch_bounds__(1024) k_kda_tail_gemm(ProjArgs a)
{
    __shared__ float4 red[32][32];
    const int m0 = blockIdx.y * (8 * NT);
    const int rows = min(8 * NT, a.M - m0);
    int i = 0;
    while (i + 1 < a.n && static_cast<int>(blockIdx.x) >= a.p[i + 1].cta0) ++i;
    const ProjPart& p = a.p[i];
    const u32* xf = p.xfrag + static_cast<size_t>(m0 / 8) * p.K * 4;
    float* y = static_cast<float*>(p.y) + static_cast<size_t>(m0) * a.ld;
    if (p.codec == 1)
        tail_tile<Mxfp8, NT>(p, xf, y, rows, a.ld, blockIdx.x - p.cta0, red);
    else
        tail_tile<Bf16w, NT>(p, xf, y, rows, a.ld, blockIdx.x - p.cta0, red);
}
inline void launch_tail_gemm(const ProjArgs& a, int row_tile, cudaStream_t st)
{
    if (row_tile == 64)
        k_kda_tail_gemm<8><<<dim3(a.ctas, (a.M + 63) / 64), 1024, 0, st>>>(a);
    else
        k_kda_tail_gemm<4><<<dim3(a.ctas, (a.M + 31) / 32), 1024, 0, st>>>(a);
}
} // namespace kda
