// gr8 DeepSeek-V4.1-Flash shared expert + MoE combine. Our own code; standalone (no glue.cuh).
//
// Shared expert (checkpoint ffn.shared_experts, MXFP8 = E4M3 codes + one E8M0 scale per output row per 32 k), TP slice Iw = 2304 / nr:
//   act[m][i] = fp16( swiglu(w1 x, w3 x) ),  swiglu clamps before silu: gate <= L, up in [-L, L], L = swiglu_limit (10) -- the
//                                             shared expert clamps like a routed one (model.py:847-850)
//   out[m][n] = fp32( w2 act )                no route weight and no routed_scaling 1.5 (model.py:905: the shared expert is added unscaled)
// Weights are the mma A operand, the tokens (<= 8) the B columns, as in mxfp4.cuh / densev.cuh. The record is the loader's `P.mx.gv`
// (records::mxfp8_record, GvGroup<Mxfp8> of densev.cuh): 16 rows x 32 k = 528 B. Three matrices, three streams:
//   sh.w1.mx.gv  [Iw/16][K1/32][528]   gate      sh.w3.mx.gv  [Iw/16][K1/32][528]   up      sh.w2.mx.gv  [NOUT/16][Iw/32][528]   down
// The fc1 B table is mx4::build_xfrag's (the routed fc1 reads the same one: build it once per layer). W8A16 like the routed path:
// not bit-identical to the reference (it fp8-quantizes the activations).
// The k-split partials of fc1 are summed in smem in fixed order: deterministic.
//
// Combine, M <= 8 (decode), fixed order, no atomics:
//   y[m][n] = ( sum_{k < TOPK} ypart[tok_slot[m][k]][tok_col[m][k]][n]  +  shared[m][n] )  rounded once to fp16 / bf16 / fp32
// ypart = k_fc2 of mxfp4.cuh (fp16, route weight already applied), shared = k_sh_fc2 (fp32), the plan = ds_router.cuh's.
// y is this rank's partial; the TP all-reduce adds the other ranks'. A token with no route (tok_slot -1) gets the shared expert only.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o ds_shared_test tests/ds_shared_test.cu
#pragma once
#include "mxfp4.cuh"
#include "densev.cuh"
#include "ds_router.cuh"

namespace dsh
{
constexpr int GR = Mxfp8::GR;   // 528
constexpr int KG = 32;          // k per record

struct Args
{
    const uint8_t* w1;          // gate [I/16][K1/32][GR]
    const uint8_t* w3;          // up   [I/16][K1/32][GR]
    const uint8_t* w2;          // down [NOUT/16][I/32][GR]
    const u32* xfrag;           // fc1 B table [K1/16][32][2] (mx4::build_xfrag)
    int mask;                   // bit c = column c is a live token (fc2 writes only those)
    float limit;                // swiglu_limit (10)
    __half* act;                // [8][I]
    float* out;                 // [8][NOUT]
};

// ---------------------------------------------------------------------------------------------------- fc1
// One warp part: gate and up rows of one 16-row chunk over NGP records (k16 step ks0 = 2 per record). sg, su = the chunk's
// gate / up streams at the first record. DEPTH records of each are in flight per lane.
template <int NGP, int DEPTH>
__device__ __forceinline__ void fc1_part(const uint8_t* sg, const uint8_t* su, const u32* bf, int lane, int ks0, float (&ag)[4], float (&au)[4])
{
    GvGroup<Mxfp8> ng[DEPTH], nu[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NGP) { ng[p].load(sg + p * GR, lane); nu[p].load(su + p * GR, lane); }
    for (int g = 0; g < NGP; g += DEPTH)
    {
        GvGroup<Mxfp8> cg[DEPTH], cu[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NGP) { ng[p].load(sg + (g + DEPTH + p) * GR, lane); nu[p].load(su + (g + DEPTH + p) * GR, lane); }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NGP) { cg[p].run(bf, ks0 + (g + p) * 2, ag); cu[p].run(bf, ks0 + (g + p) * 2, au); }
    }
}

// CTA = CW chunks x KS k parts (a warp each). grid = I / 16 / CW.
// PDLM = 1: the whole-layer PDL chain instance (launched under k_plan_p): tail wait before the act stores.
template <int K1, int I, int KS, int CW, int DEPTH, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32) k_sh_fc1(Args A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "shared fc1 tiling");
    __shared__ float4 ps[KS * CW * 32 * 2];
    mx4::pdl_trigger();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = blockIdx.x * CW + cl;
    const size_t off = (static_cast<size_t>(chunk) * NG + part * NGP) * GR;
    float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
    fc1_part<NGP, DEPTH>(A.w1 + off, A.w3 + off, A.xfrag + lane * 2, lane, part * NGP * 2, ag, au);
    if (KS > 1)
    {
        float4* p = ps + (warp * 32 + lane) * 2;
        p[0] = make_float4(ag[0], ag[1], ag[2], ag[3]);
        p[1] = make_float4(au[0], au[1], au[2], au[3]);
        __syncthreads();
        if (part == 0)
            for (int q = 1; q < KS; ++q)      // fixed order
            {
                const float4 g = ps[((warp + q) * 32 + lane) * 2], u = ps[((warp + q) * 32 + lane) * 2 + 1];
                ag[0] += g.x; ag[1] += g.y; ag[2] += g.z; ag[3] += g.w;
                au[0] += u.x; au[1] += u.y; au[2] += u.z; au[3] += u.w;
            }
    }
    if constexpr (PDLM != 0) mx4::pdl_wait();   // tail wait: only the act stores need the kernel before (the plan) to be done
    if (part) return;
    // C regs: (row r, col 2c), (row r, col 2c + 1), (row r + 8, col 2c), (row r + 8, col 2c + 1)
    const int r = lane >> 2, c = lane & 3;
    __half* a0 = A.act + static_cast<size_t>(2 * c) * I + chunk * 16 + r;
    a0[0] = __float2half_rn(mx4::swiglu_lim(ag[0], au[0], A.limit));
    a0[I] = __float2half_rn(mx4::swiglu_lim(ag[1], au[1], A.limit));
    a0[8] = __float2half_rn(mx4::swiglu_lim(ag[2], au[2], A.limit));
    a0[I + 8] = __float2half_rn(mx4::swiglu_lim(ag[3], au[3], A.limit));
}

// ---------------------------------------------------------------------------------------------------- fc2
// CTA = WARPS output tiles (a warp each, all k). The activations become a B fragment table in smem. grid = NOUT / 16 / WARPS.
// PDL (see mxfp4.cuh): the first weight loads are issued before griddepcontrol.wait, which waits for fc1 to flush act.
// PDLM = 1: the whole-layer PDL chain instance: triggers after its wait.
template <int I, int NOUT, int WARPS, int DEPTH, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_sh_fc2(Args A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "shared fc2 tiling");
    __shared__ __align__(16) u32 sb[KSI * 64];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = blockIdx.x * WARPS + warp;
    const uint8_t* s = A.w2 + static_cast<size_t>(ot) * NG * GR;
    GvGroup<Mxfp8> nx[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NG) nx[p].load(s + p * GR, lane);
    mx4::pdl_wait();
    if constexpr (PDLM != 0) mx4::pdl_trigger();   // only after the wait: the routed fc1 behind reads the plan and xfrag, finished two kernels back
    for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
    {
        const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
        sb[i] = __ldcg(reinterpret_cast<const u32*>(A.act + static_cast<size_t>(l >> 2) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
    }
    __syncthreads();
    const u32* bf = sb + lane * 2;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int g = 0; g < NG; g += DEPTH)
    {
        GvGroup<Mxfp8> cur[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NG) nx[p].load(s + (g + DEPTH + p) * GR, lane);
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NG) cur[p].run(bf, (g + p) * 2, acc);
    }
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
    if ((A.mask >> (2 * c)) & 1)
    {
        A.out[(2 * c) * NOUT + o] = acc[0];
        A.out[(2 * c) * NOUT + o + 8] = acc[2];
    }
    if ((A.mask >> (2 * c + 1)) & 1)
    {
        A.out[(2 * c + 1) * NOUT + o] = acc[1];
        A.out[(2 * c + 1) * NOUT + o + 8] = acc[3];
    }
}

// ---------------------------------------------------------------------------------------------------- multi-tile shared expert (M 9..32)
// No plan: tokens are the columns of NT n=8 tiles (tile t = tokens 8t..8t+7); GvGroup::run<NT> (densev.cuh) decodes each weight record once
// and runs the stock mma sequence per tile, so a token's bits equal the M <= 8 path's. The M <= 8 kernels above are unchanged.
struct ArgsMT
{
    const uint8_t* w1;          // gate [I/16][K1/32][GR]
    const uint8_t* w3;          // up   [I/16][K1/32][GR]
    const uint8_t* w2;          // down [NOUT/16][I/32][GR]
    const u32* xfrag;           // fc1 B tables, NT x [K1/16][32][2] (mx4::build_xfrag_mt); tile stride K1 * 4 words
    int rows;                   // M: fc2 writes only rows < M
    float limit;                // swiglu_limit (10)
    __half* act;                // [NT * 8][I]
    float* out;                 // [NT * 8][NOUT]
};

template <int NGP, int DEPTH, int NT>
__device__ __forceinline__ void fc1_part_mt(const uint8_t* sg, const uint8_t* su, const u32* bf, int ts, int lane, int ks0, float (&ag)[NT][4], float (&au)[NT][4])
{
    GvGroup<Mxfp8> ng[DEPTH], nu[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NGP) { ng[p].load(sg + p * GR, lane); nu[p].load(su + p * GR, lane); }
    for (int g = 0; g < NGP; g += DEPTH)
    {
        GvGroup<Mxfp8> cg[DEPTH], cu[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NGP) { ng[p].load(sg + (g + DEPTH + p) * GR, lane); nu[p].load(su + (g + DEPTH + p) * GR, lane); }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NGP) { cg[p].template run<NT>(bf, ks0 + (g + p) * 2, ag, ts); cu[p].template run<NT>(bf, ks0 + (g + p) * 2, au, ts); }
    }
}

// CTA = CW chunks x KS k parts (a warp each). grid = I / 16 / CW.
// PDLM = 1: the whole-layer PDL chain instance (tail wait before the act stores), as sh_fc1_body.
template <int K1, int I, int KS, int CW, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32) k_sh_fc1_mt(ArgsMT A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "shared fc1 tiling");
    __shared__ float4 ps[KS > 1 ? NT : 1][KS * CW * 32 * 2];
    mx4::pdl_trigger();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = blockIdx.x * CW + cl;
    const size_t off = (static_cast<size_t>(chunk) * NG + part * NGP) * GR;
    float ag[NT][4], au[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
    fc1_part_mt<NGP, DEPTH, NT>(A.w1 + off, A.w3 + off, A.xfrag + lane * 2, K1 * 4, lane, part * NGP * 2, ag, au);
    if (KS > 1)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            float4* p = ps[t] + (warp * 32 + lane) * 2;
            p[0] = make_float4(ag[t][0], ag[t][1], ag[t][2], ag[t][3]);
            p[1] = make_float4(au[t][0], au[t][1], au[t][2], au[t][3]);
        }
        __syncthreads();
        if (part == 0)
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                for (int q = 1; q < KS; ++q)      // fixed order
                {
                    const float4 g = ps[t][((warp + q) * 32 + lane) * 2], u = ps[t][((warp + q) * 32 + lane) * 2 + 1];
                    ag[t][0] += g.x; ag[t][1] += g.y; ag[t][2] += g.z; ag[t][3] += g.w;
                    au[t][0] += u.x; au[t][1] += u.y; au[t][2] += u.z; au[t][3] += u.w;
                }
    }
    if constexpr (PDLM != 0) mx4::pdl_wait();
    if (part) return;
    const int r = lane >> 2, c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        __half* a0 = A.act + static_cast<size_t>(t * 8 + 2 * c) * I + chunk * 16 + r;
        a0[0] = __float2half_rn(mx4::swiglu_lim(ag[t][0], au[t][0], A.limit));
        a0[I] = __float2half_rn(mx4::swiglu_lim(ag[t][1], au[t][1], A.limit));
        a0[8] = __float2half_rn(mx4::swiglu_lim(ag[t][2], au[t][2], A.limit));
        a0[I + 8] = __float2half_rn(mx4::swiglu_lim(ag[t][3], au[t][3], A.limit));
    }
}

// A copy of k_sh_fc1_mt (so the existing instances' SASS stays) whose act stores skip the rows >= A.rows (a tile of 8 rows would otherwise write up to NT * 8 - M
// rows past the live rows). Same arithmetic, same order; the rows < M are the bytes of k_sh_fc1_mt. Used by shared_pw_1pass only. grid = I / 16 / CW.
// PDLM = 1: the whole-layer PDL chain instance (tail wait before the act stores), as sh_fc1_body.
template <int K1, int I, int KS, int CW, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32) k_sh_fc1_mt_m(ArgsMT A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "shared fc1 tiling");
    __shared__ float4 ps[KS > 1 ? NT : 1][KS * CW * 32 * 2];
    mx4::pdl_trigger();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = blockIdx.x * CW + cl;
    const size_t off = (static_cast<size_t>(chunk) * NG + part * NGP) * GR;
    float ag[NT][4], au[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
    fc1_part_mt<NGP, DEPTH, NT>(A.w1 + off, A.w3 + off, A.xfrag + lane * 2, K1 * 4, lane, part * NGP * 2, ag, au);
    if (KS > 1)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            float4* p = ps[t] + (warp * 32 + lane) * 2;
            p[0] = make_float4(ag[t][0], ag[t][1], ag[t][2], ag[t][3]);
            p[1] = make_float4(au[t][0], au[t][1], au[t][2], au[t][3]);
        }
        __syncthreads();
        if (part == 0)
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                for (int q = 1; q < KS; ++q)      // fixed order
                {
                    const float4 g = ps[t][((warp + q) * 32 + lane) * 2], u = ps[t][((warp + q) * 32 + lane) * 2 + 1];
                    ag[t][0] += g.x; ag[t][1] += g.y; ag[t][2] += g.z; ag[t][3] += g.w;
                    au[t][0] += u.x; au[t][1] += u.y; au[t][2] += u.z; au[t][3] += u.w;
                }
    }
    if constexpr (PDLM != 0) mx4::pdl_wait();
    if (part) return;
    const int r = lane >> 2, c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int row = t * 8 + 2 * c;
        __half* a0 = A.act + static_cast<size_t>(row) * I + chunk * 16 + r;
        if (row < A.rows)
        {
            a0[0] = __float2half_rn(mx4::swiglu_lim(ag[t][0], au[t][0], A.limit));
            a0[8] = __float2half_rn(mx4::swiglu_lim(ag[t][2], au[t][2], A.limit));
        }
        if (row + 1 < A.rows)
        {
            a0[I] = __float2half_rn(mx4::swiglu_lim(ag[t][1], au[t][1], A.limit));
            a0[I + 8] = __float2half_rn(mx4::swiglu_lim(ag[t][3], au[t][3], A.limit));
        }
    }
}

// CTA = WARPS output tiles (a warp each, all k); NT activation tiles become smem B tables (stride KSI * 64 words). grid = NOUT / 16 / WARPS.
// PDLM = 1: the whole-layer PDL chain instance (trigger after the wait), as sh_fc2_body.
template <int I, int NOUT, int WARPS, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_sh_fc2_mt(ArgsMT A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "shared fc2 tiling");
    __shared__ __align__(16) u32 sb[NT][KSI * 64];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = blockIdx.x * WARPS + warp;
    const uint8_t* s = A.w2 + static_cast<size_t>(ot) * NG * GR;
    GvGroup<Mxfp8> nx[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NG) nx[p].load(s + p * GR, lane);
    mx4::pdl_wait();
    if constexpr (PDLM != 0) mx4::pdl_trigger();
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
        {
            const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
            sb[t][i] = __ldcg(reinterpret_cast<const u32*>(A.act + static_cast<size_t>(t * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
        }
    __syncthreads();
    const u32* bf = &sb[0][0] + lane * 2;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int g = 0; g < NG; g += DEPTH)
    {
        GvGroup<Mxfp8> cur[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NG) nx[p].load(s + (g + DEPTH + p) * GR, lane);
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NG) cur[p].template run<NT>(bf, (g + p) * 2, acc, KSI * 64);
    }
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int row0 = t * 8 + 2 * c;
        if (row0 < A.rows)
        {
            A.out[static_cast<size_t>(row0) * NOUT + o] = acc[t][0];
            A.out[static_cast<size_t>(row0) * NOUT + o + 8] = acc[t][2];
        }
        if (row0 + 1 < A.rows)
        {
            A.out[static_cast<size_t>(row0 + 1) * NOUT + o] = acc[t][1];
            A.out[static_cast<size_t>(row0 + 1) * NOUT + o + 8] = acc[t][3];
        }
    }
}

// k_sh_fc2_mt with the NT smem B tables in DYNAMIC shared memory (launch with NT * KSI * 64 * 4 bytes + cudaFuncSetAttribute MaxDynamicSharedMemorySize): the static limit is 48 KB,
// NT = 6 needs 55296 B. A copy, not an edit, of k_sh_fc2_mt: the existing instances' SASS stays as it is (ab3ro). Same arithmetic, same order.
// PDLM = 1: the whole-layer PDL chain instance (trigger after the wait), as sh_fc2_body.
template <int I, int NOUT, int WARPS, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_sh_fc2_mt_dyn(ArgsMT A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "shared fc2 tiling");
    extern __shared__ __align__(16) u32 sbd[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = blockIdx.x * WARPS + warp;
    const uint8_t* s = A.w2 + static_cast<size_t>(ot) * NG * GR;
    GvGroup<Mxfp8> nx[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NG) nx[p].load(s + p * GR, lane);
    mx4::pdl_wait();
    if constexpr (PDLM != 0) mx4::pdl_trigger();
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
        {
            const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
            sbd[t * (KSI * 64) + i] = __ldcg(reinterpret_cast<const u32*>(A.act + static_cast<size_t>(t * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
        }
    __syncthreads();
    const u32* bf = sbd + lane * 2;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int g = 0; g < NG; g += DEPTH)
    {
        GvGroup<Mxfp8> cur[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NG) nx[p].load(s + (g + DEPTH + p) * GR, lane);
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NG) cur[p].template run<NT>(bf, (g + p) * 2, acc, KSI * 64);
    }
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const int row0 = t * 8 + 2 * c;
        if (row0 < A.rows)
        {
            A.out[static_cast<size_t>(row0) * NOUT + o] = acc[t][0];
            A.out[static_cast<size_t>(row0) * NOUT + o + 8] = acc[t][2];
        }
        if (row0 + 1 < A.rows)
        {
            A.out[static_cast<size_t>(row0 + 1) * NOUT + o] = acc[t][1];
            A.out[static_cast<size_t>(row0 + 1) * NOUT + o + 8] = acc[t][3];
        }
    }
}

// ---------------------------------------------------------------------------------------------------- combine
template <int TK = dsr::TOPK>
struct CombineArgsT
{
    const dsr::PlanT<TK>* plan;
    const __half* ypart;        // [slot][8][NOUT] fp16 (mx4::k_fc2)
    const float* shared;        // [8][NOUT] fp32 (k_sh_fc2); nullptr = no shared term
    void* out;                  // [M][NOUT] of odt
    int M;
};
using CombineArgs = CombineArgsT<>;

// out = the sum above. Thread = 8 consecutive columns of one token (one 16-byte ypart load per route). odt = gv::O_F16 / O_BF16 / O_F32.
// grid = M * NOUT / (8 * CT), block CT.
template <int NOUT, int CT, int ODT, int TK = dsr::TOPK>
__global__ void __launch_bounds__(CT) k_combine(CombineArgsT<TK> A)
{
    constexpr int TPT = NOUT / 8, BPT = TPT / CT;
    static_assert(NOUT % (8 * CT) == 0, "combine tiling");
    mx4::pdl_wait();
    const int m = blockIdx.x / BPT;
    const int n0 = ((blockIdx.x % BPT) * CT + threadIdx.x) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int sl[TK], co[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k) { sl[k] = A.plan->tok_slot[m][k]; co[k] = A.plan->tok_col[m][k]; }
    uint4 v[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k)       // all route loads in flight, summed afterwards in route order
        if (sl[k] >= 0) v[k] = __ldcg(reinterpret_cast<const uint4*>(A.ypart + (static_cast<size_t>(sl[k]) * 8 + co[k]) * NOUT + n0));
    float4 sh0 = make_float4(0.f, 0.f, 0.f, 0.f), sh1 = sh0;
    if (A.shared)
    {
        sh0 = __ldcg(reinterpret_cast<const float4*>(A.shared + static_cast<size_t>(m) * NOUT + n0));
        sh1 = __ldcg(reinterpret_cast<const float4*>(A.shared + static_cast<size_t>(m) * NOUT + n0) + 1);
    }
    #pragma unroll
    for (int k = 0; k < TK; ++k)
    {
        if (sl[k] < 0) continue;
        const u32 w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float2 f = __half22float2(as_h2(w[i]));
            acc[2 * i] += f.x; acc[2 * i + 1] += f.y;
        }
    }
    acc[0] += sh0.x; acc[1] += sh0.y; acc[2] += sh0.z; acc[3] += sh0.w;
    acc[4] += sh1.x; acc[5] += sh1.y; acc[6] += sh1.z; acc[7] += sh1.w;
    if (ODT == gv::O_F32)
    {
        float4* o = reinterpret_cast<float4*>(static_cast<float*>(A.out) + static_cast<size_t>(m) * NOUT + n0);
        o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    else
    {
        uint4 o;
        o.x = pack_o2(acc[0], acc[1], ODT == gv::O_BF16); o.y = pack_o2(acc[2], acc[3], ODT == gv::O_BF16);
        o.z = pack_o2(acc[4], acc[5], ODT == gv::O_BF16); o.w = pack_o2(acc[6], acc[7], ODT == gv::O_BF16);
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(A.out) + static_cast<size_t>(m) * NOUT + n0) = o;
    }
}

template <int TK = dsr::TOPK, int MM = dsr::MT_MAX>
struct CombineArgsMT
{
    const dsr::PlanMT<TK, MM>* plan;
    const __half* ypart;        // [tile][8][NOUT] fp16 (mx4::k_fc2_mt)
    const float* shared;        // [M][NOUT] fp32 (k_sh_fc2_mt); nullptr = no shared term
    void* out;                  // [M][NOUT] of odt
    int M;
};

// Multi-tile (M 9..32): a copy of k_combine for the PlanMT plan (kept separate so k_combine's SASS stays identical to the M <= 8 build).
// ypart is [tile][8][NOUT] and the plan's tok_slot is the global tile; shared [M][NOUT]. Same sum, same order, same single rounding.
// grid = M * NOUT / (8 * CT), block CT.
template <int NOUT, int CT, int ODT, int TK = dsr::TOPK, int MM = dsr::MT_MAX>
__global__ void __launch_bounds__(CT) k_combine_mt(CombineArgsMT<TK, MM> A)
{
    constexpr int TPT = NOUT / 8, BPT = TPT / CT;
    static_assert(NOUT % (8 * CT) == 0, "combine tiling");
    mx4::pdl_wait();
    const int m = blockIdx.x / BPT;
    const int n0 = ((blockIdx.x % BPT) * CT + threadIdx.x) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int sl[TK], co[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k) { sl[k] = A.plan->tok_slot[m][k]; co[k] = A.plan->tok_col[m][k]; }
    uint4 v[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k)       // all route loads in flight, summed afterwards in route order
        if (sl[k] >= 0) v[k] = __ldcg(reinterpret_cast<const uint4*>(A.ypart + (static_cast<size_t>(sl[k]) * 8 + co[k]) * NOUT + n0));
    float4 sh0 = make_float4(0.f, 0.f, 0.f, 0.f), sh1 = sh0;
    if (A.shared)
    {
        sh0 = __ldcg(reinterpret_cast<const float4*>(A.shared + static_cast<size_t>(m) * NOUT + n0));
        sh1 = __ldcg(reinterpret_cast<const float4*>(A.shared + static_cast<size_t>(m) * NOUT + n0) + 1);
    }
    #pragma unroll
    for (int k = 0; k < TK; ++k)
    {
        if (sl[k] < 0) continue;
        const u32 w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float2 f = __half22float2(as_h2(w[i]));
            acc[2 * i] += f.x; acc[2 * i + 1] += f.y;
        }
    }
    acc[0] += sh0.x; acc[1] += sh0.y; acc[2] += sh0.z; acc[3] += sh0.w;
    acc[4] += sh1.x; acc[5] += sh1.y; acc[6] += sh1.z; acc[7] += sh1.w;
    if (ODT == gv::O_F32)
    {
        float4* o = reinterpret_cast<float4*>(static_cast<float*>(A.out) + static_cast<size_t>(m) * NOUT + n0);
        o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    else
    {
        uint4 o;
        o.x = pack_o2(acc[0], acc[1], ODT == gv::O_BF16); o.y = pack_o2(acc[2], acc[3], ODT == gv::O_BF16);
        o.z = pack_o2(acc[4], acc[5], ODT == gv::O_BF16); o.w = pack_o2(acc[6], acc[7], ODT == gv::O_BF16);
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(A.out) + static_cast<size_t>(m) * NOUT + n0) = o;
    }
}
// The tensor-core shared expert (GR8_DS_SH_TC=1, the read-once path only): a copy of k_combine_mt whose shared term is bf16 [M][NOUT] (the TC shared expert's Y, aliased in sh_out;
// A.shared carries it cast to float*). Kept separate so k_combine_mt's SASS stays identical (a shared body changes it). Same sum, same order,
// same single rounding; bf16 -> fp32 is exact.
template <int NOUT, int CT, int ODT, int TK = dsr::TOPK, int MM = dsr::MT_MAX>
__global__ void __launch_bounds__(CT) k_combine_mt_shb(CombineArgsMT<TK, MM> A)
{
    constexpr int TPT = NOUT / 8, BPT = TPT / CT;
    static_assert(NOUT % (8 * CT) == 0, "combine tiling");
    mx4::pdl_wait();
    const int m = blockIdx.x / BPT;
    const int n0 = ((blockIdx.x % BPT) * CT + threadIdx.x) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int sl[TK], co[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k) { sl[k] = A.plan->tok_slot[m][k]; co[k] = A.plan->tok_col[m][k]; }
    uint4 v[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k)       // all route loads in flight, summed afterwards in route order
        if (sl[k] >= 0) v[k] = __ldcg(reinterpret_cast<const uint4*>(A.ypart + (static_cast<size_t>(sl[k]) * 8 + co[k]) * NOUT + n0));
    float4 sh0 = make_float4(0.f, 0.f, 0.f, 0.f), sh1 = sh0;
    if (A.shared)                      // the only change: the shared term is bf16 [M][NOUT] (exact widening to fp32)
    {
        const uint4 b = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(A.shared) + static_cast<size_t>(m) * NOUT + n0));
        sh0 = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u), __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
        sh1 = make_float4(__uint_as_float(b.z << 16), __uint_as_float(b.z & 0xffff0000u), __uint_as_float(b.w << 16), __uint_as_float(b.w & 0xffff0000u));
    }
    #pragma unroll
    for (int k = 0; k < TK; ++k)
    {
        if (sl[k] < 0) continue;
        const u32 w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float2 f = __half22float2(as_h2(w[i]));
            acc[2 * i] += f.x; acc[2 * i + 1] += f.y;
        }
    }
    acc[0] += sh0.x; acc[1] += sh0.y; acc[2] += sh0.z; acc[3] += sh0.w;
    acc[4] += sh1.x; acc[5] += sh1.y; acc[6] += sh1.z; acc[7] += sh1.w;
    if (ODT == gv::O_F32)
    {
        float4* o = reinterpret_cast<float4*>(static_cast<float*>(A.out) + static_cast<size_t>(m) * NOUT + n0);
        o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    else
    {
        uint4 o;
        o.x = pack_o2(acc[0], acc[1], ODT == gv::O_BF16); o.y = pack_o2(acc[2], acc[3], ODT == gv::O_BF16);
        o.z = pack_o2(acc[4], acc[5], ODT == gv::O_BF16); o.w = pack_o2(acc[6], acc[7], ODT == gv::O_BF16);
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(A.out) + static_cast<size_t>(m) * NOUT + n0) = o;
    }
}
}   // namespace dsh
