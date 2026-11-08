// Opt-in v2 CQ pipeline. Derived from our k_attn_cpa_t; no external code.
// Stage 0: 64 bf16 K rows; stage 1: 64 packed rows. Q's 32 MMA
// fragments per lane live in registers after the prologue, freeing its smem.
constexpr size_t AT_PIPE_STAGE = 64 * CQ_ROW;
constexpr size_t AT_SMEM_PIPE = AT_SMEM_CPA - 16 * AT_QLD * sizeof(bf16) + AT_PIPE_STAGE;
static_assert(AT_SMEM_PIPE <= 99 * 1024, "pipeline fits SM shared memory");
__device__ __forceinline__ void pipe_store(bf16* Ks, const uint8_t* packed, int w, int lane)
{
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const uint8_t* r = packed + (w + 8 * k) * CQ_ROW;
        const uint2 cd = ((const uint2*)r)[lane];
        const __nv_bfloat162 sc = __float2bfloat162_rn(f_e4m3(r[CQ_CODES + lane]));
        uint32_t o[8];
        cq_dq4(cd.x, sc, o[0], o[1]); cq_dq4(cd.x >> 16, sc, o[2], o[3]);
        cq_dq4(cd.y, sc, o[4], o[5]); cq_dq4(cd.y >> 16, sc, o[6], o[7]);
        uint4* dst = (uint4*)(Ks + (w + 8 * k) * AT_KLD + lane * 16);
        dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
        dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
}
// Keep the body in the kernel scope so launch bounds retain their optimizer facts.
__global__ void __launch_bounds__(AT_THREADS, 1) k_attn_cpa_pipe_q(AttnP p)
{
#include "ds41_attn_pipe_body.cuh"
}
// Register-cap arm from our ctune_pipe_cap sweep; no arithmetic change.
__global__ void __maxnreg__(240) k_attn_cpa_pipe_cap_q(AttnP p)
{
#include "ds41_attn_pipe_body.cuh"
}
