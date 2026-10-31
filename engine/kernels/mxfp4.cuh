// gr8 DeepSeek-V4.1-Flash routed-expert decode GEMV. Our own code.
//
// Weights: MXFP4 = E2M1 codes with one E8M0 scale (a power of two) per output row per 32 k, the checkpoint's own format.
// Activations: fp16. Accumulate: fp32 on the fp16 tensor-core path. W4A16, not the reference's fp8-activation path, so a
// layer is not bit-identical to the reference model (the activation rounding differs).
//
// The weights are the mma A operand (16 output rows x 16 k per tile), the tokens (<= 8) are the B columns, as in
// rowtile.cuh. A tile is 128 B in lane order: lane l = 4 r + c reads one u32 {row r byte c, row r+8 byte c, row r byte c+4,
// row r+8 byte c+4}; byte j of a row holds k 2j (low nibble) and 2j+1 (high nibble). The code decode is one
// cvt.rn.f16x2.e2m1x2 per byte. E2M1 x fp16 is exact in fp32, and a power-of-two scale on the group's fp32 partial is exact.
//
// Record "mx4g" = 16 rows x 64 k = 544 B (a multiple of 16, so every lane load is aligned):
//   [  0, 512)  4 tiles (k16 steps 0..3): lane l at byte 16 l holds u32 tile 0, 1, 2, 3  (one 128-bit load per lane)
//   [512, 544)  E8M0 bytes, row r (0..7) at 512 + 4 r: {k32 group 0 row r, group 0 row r+8, group 1 row r, group 1 row r+8}
// The 64-k group needs K % 64 == 0 (hidden 5120, TP slices 576 / 384 / 768 / 1152; TP8's 288 would need a zero-padded tail).
//
// Streams (per expert, back to back):
//   fc1  [chunk of 16 intermediate rows][K1/64][gate mx4g | up mx4g]      (one warp part reads gate and up of a chunk together)
//   fc2  [output tile of 16 rows][I/64][mx4g]
//
// Kernels, M <= 8 tokens (one n=8 tile), slot-driven like K1 (slot_expert ascending, slot_mask bit c = column c used):
//   k_mx4_fc1: act[slot][c][I] = fp16(swiglu(gate, up)) , swiglu clamps before silu (gate <= L, up in [-L, L])
//   k_mx4_fc2: ypart[slot][c][NOUT] = fp16(slot_w[slot][c] * (down @ act))     (the routing weight carries routed_scaling 1.5)
// The k-split partials of fc1 are summed in smem in fixed order, so the result is deterministic.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o mxfp4_gemv_test tests/mxfp4_gemv_test.cu
#pragma once
#include "common.cuh"
#include "ds_plan_mt.h"
#include <cuda_bf16.h>
#include "swiglu_lim.cuh"

namespace mx4
{
constexpr int KG = 64;        // k per record group
constexpr int GR = 544;       // bytes of one 16-row x 64-k record
constexpr int SOFF = 512;     // scale bytes inside it

__device__ __forceinline__ void ld4(const void* p, u32 (&v)[4])
{
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));
}
// GR8_DS_MOE_L2HINT: the two weight loads of Grp::load with an L2 evict_last hint. Each asm makes its own policy (createpolicy ... 1.0), so no 64-bit policy is held in a
// register across the loop. Reached only through Grp::loadh<true>; nothing else calls them.
__device__ __forceinline__ void ld4_last(const void* p, u32 (&v)[4])
{
    asm volatile("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                 "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], pol;\n\t}"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));
}
__device__ __forceinline__ u32 ld_nc_last(const void* p)
{
    u32 v;
    asm volatile("{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                 "ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], pol;\n\t}"
                 : "=r"(v) : "l"(p));
    return v;
}
// Programmatic dependent launch: a no-op unless the kernel was launched with cudaLaunchAttributeProgrammaticStreamSerialization.
// fc1 lets fc2 start once its last CTA has started; fc2 issues its first weight loads (independent of fc1) and then waits for
// fc1 to finish and flush before it reads the activations.
// The whole-layer chain (ds41_moe_ops.cuh) uses the PDLM = 1 instances / _p twins below: build_xfrag_p and k_fc2_p trigger at the start,
// k_fc1_p waits just before its act stores (tail wait), and every CTA of a PDL-launched kernel reaches a pdl_wait even on its early exits,
// so each kernel's completion implies its predecessor's. The PDLM = 0 kernels keep their shipped codegen.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ float e8(u32 b) { b &= 0xffu; return __uint_as_float(b ? b << 23 : 0x00400000u); }   // 2^(b - 127), as Mxfp8::e8m0

// One lane's share of a 16-row x 64-k record: 4 code words (k16 steps 0..3) and the 4 scale bytes of its rows r, r + 8.
struct Grp
{
    u32 c[4];
    u32 s;
    __device__ __forceinline__ void load(const uint8_t* g, int lane)
    {
        ld4(g + lane * 16, c);
        s = ld_nc(g + SOFF + (lane >> 2) * 4);
    }
    // L2H = false: load (same code). L2H = true: the same two loads with the L2 evict_last hint (GR8_DS_MOE_L2HINT, the read-once expert loop only).
    template <bool L2H>
    __device__ __forceinline__ void loadh(const uint8_t* g, int lane)
    {
        if constexpr (L2H) { ld4_last(g + lane * 16, c); s = ld_nc_last(g + SOFF + (lane >> 2) * 4); }
        else load(g, lane);
    }
    // acc += scales * (codes @ B); bf = fragment table [k16 step][lane][2] already offset to this lane; ks0 = step of tile 0
    template <bool PROBE>
    __device__ __forceinline__ void run(const u32* bf, int ks0, float (&acc)[4]) const
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)         // the two k32 scale groups
        {
            float g[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                const u32 w = c[2 * h + i];
                const uint2 b = *reinterpret_cast<const uint2*>(bf + (ks0 + 2 * h + i) * 64);
#ifndef MX4_PROBE_ALWAYS
                if constexpr (!PROBE)
                    mma16816(g, e2m1x2(w & 0xffu), e2m1x2((w >> 8) & 0xffu), e2m1x2((w >> 16) & 0xffu), e2m1x2(w >> 24), b.x, b.y);
                else
#endif
                    g[0] += __int_as_float((w ^ b.x ^ b.y) & 0x3fffffff);   // cost probe: the same loads, no dequant, no mma
            }
            const float sl = e8(s >> (16 * h)), sh = e8(s >> (16 * h + 8));
            acc[0] = fmaf(sl, g[0], acc[0]); acc[1] = fmaf(sl, g[1], acc[1]);
            acc[2] = fmaf(sh, g[2], acc[2]); acc[3] = fmaf(sh, g[3], acc[3]);
        }
    }
    // Multi-tile (M 9..32): the record is decoded once per k16 step and fed to the first nt (<= NT, CTA-uniform) n=8 tiles. xb + boff[t] = tile t's
    // fragment stream for this lane (fc1: a per-lane gather over the token tables; fc2: the smem tile t). Each live tile runs the stock
    // mma / scale / accumulate sequence of run<false> on its own columns, so a column's bits do not depend on nt or NT.
    template <int NT>
    __device__ __forceinline__ void run_mt(const u32* xb, const int (&boff)[NT], int ks0, int nt, float (&acc)[NT][4]) const
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            float g[NT][4];
            #pragma unroll
            for (int t = 0; t < NT; ++t) g[t][0] = g[t][1] = g[t][2] = g[t][3] = 0.f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                const u32 w = c[2 * h + i];
                const u32 a0 = e2m1x2(w & 0xffu), a1 = e2m1x2((w >> 8) & 0xffu), a2 = e2m1x2((w >> 16) & 0xffu), a3 = e2m1x2(w >> 24);
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                    if (t < nt)
                    {
                        const uint2 b = *reinterpret_cast<const uint2*>(xb + boff[t] + (ks0 + 2 * h + i) * 64);
                        mma16816(g[t], a0, a1, a2, a3, b.x, b.y);
                    }
            }
            const float sl = e8(s >> (16 * h)), sh = e8(s >> (16 * h + 8));
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t < nt)
                {
                    acc[t][0] = fmaf(sl, g[t][0], acc[t][0]); acc[t][1] = fmaf(sl, g[t][1], acc[t][1]);
                    acc[t][2] = fmaf(sh, g[t][2], acc[t][2]); acc[t][3] = fmaf(sh, g[t][3], acc[t][3]);
                }
        }
    }
};

struct Args
{
    const uint8_t* fc1;        // [E][I/16][K1/64][2][GR]
    const uint8_t* fc2;        // [E][NOUT/16][I/64][GR]
    const u32* xfrag;          // fc1 B table [K1/16][32][2] (mx4::build_xfrag)
    const int* slot_expert;    // [slot], ascending
    const int* slot_mask;      // [slot]
    const float* slot_w;       // [slot][8] routing weight of column c
    const int* n_slots_dev;    // nullptr = the grid is exact; else CTAs of slots >= *n_slots_dev exit
    int n_slots;               // slots the grids are launched for
    float limit;               // swiglu_limit (10)
    __half* act;               // [slot][8][I]
    __half* ypart;             // [slot][8][NOUT]
};

// x [M][K1] bf16 (the sub-layer input after the norm) -> fp16 fragment table, tokens as B columns, columns >= M zero.
// bf16 -> fp16 is exact inside the fp16 range; it saturates at +-65504 (a normed activation never gets near).
// grid = K1 / 16 * 64 / 256, block 256.
template <int PDLM>
__device__ __forceinline__ void build_xfrag_body(const __nv_bfloat16* x, int M, int K1, u32* xfrag)
{
    if constexpr (PDLM != 0) pdl_trigger();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K1 / 16 * 64) return;
    const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1, t = l >> 2;
    u32 v = 0;
    if (t < M)
    {
        const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(x + static_cast<size_t>(t) * K1 + ks * 16 + reg * 8 + 2 * (l & 3));
        const float2 f = __bfloat1622float2(p);
        v = pack_h2(fminf(fmaxf(f.x, -65504.f), 65504.f), fminf(fmaxf(f.y, -65504.f), 65504.f));
    }
    xfrag[i] = v;
}
__global__ void build_xfrag(const __nv_bfloat16* x, int M, int K1, u32* xfrag) { build_xfrag_body<0>(x, M, K1, xfrag); }
// PDL twin: launched plain (it is the first kernel of the layer's MoE chain), triggers at once so the gate can start under it.
__global__ void build_xfrag_p(const __nv_bfloat16* x, int M, int K1, u32* xfrag) { build_xfrag_body<1>(x, M, K1, xfrag); }

// ---------------------------------------------------------------------------------------------------- fc1
// One warp part: gate and up rows of one 16-row chunk over NGP records starting at record g0 (k16 step ks0).
// s = the chunk's stream at record g0, records 2 GR apart. DEPTH records of gate and of up are in flight per lane.
template <int NGP, int DEPTH, bool PROBE>
__device__ __forceinline__ void fc1_part(const uint8_t* s, const u32* bf, int lane, int ks0, float (&ag)[4], float (&au)[4])
{
    Grp ng[DEPTH], nu[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NGP) { ng[p].load(s + p * 2 * GR, lane); nu[p].load(s + p * 2 * GR + GR, lane); }
    for (int g = 0; g < NGP; g += DEPTH)
    {
        Grp cg[DEPTH], cu[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NGP) { ng[p].load(s + (g + DEPTH + p) * 2 * GR, lane); nu[p].load(s + (g + DEPTH + p) * 2 * GR + GR, lane); }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NGP) { cg[p].template run<PROBE>(bf, ks0 + (g + p) * 4, ag); cu[p].template run<PROBE>(bf, ks0 + (g + p) * 4, au); }
    }
}

// swiglu_lim(gate, up, lim): swiglu_lim.cuh (one definition, shared with the attention TU's tensor-core shared expert)

// CTA = one slot x CW chunks x KS k parts (a warp each). grid = n_slots * (I / 16 / CW).
// PDLM = 1: the whole-layer PDL chain instance (tail wait before the act stores, a wait on the early exit).
template <int K1, int I, int KS, int CW, int DEPTH, bool PROBE, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32) k_fc1(Args A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS, UPS = NCH / CW;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "fc1 tiling");
    __shared__ float4 ps[KS * CW * 32 * 2];
    pdl_trigger();
    const int slot = blockIdx.x / UPS, cg = blockIdx.x % UPS;
    if (A.n_slots_dev && slot >= *A.n_slots_dev)
    {
        if constexpr (PDLM != 0) pdl_wait();
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = cg * CW + cl;
    const size_t eb = static_cast<size_t>(A.slot_expert[slot]) * NCH * NG * 2 * GR;
    const uint8_t* s = A.fc1 + eb + (static_cast<size_t>(chunk) * NG + part * NGP) * 2 * GR;
    float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
    fc1_part<NGP, DEPTH, PROBE>(s, A.xfrag + lane * 2, lane, part * NGP * 4, ag, au);
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
    if constexpr (PDLM != 0) pdl_wait();      // tail wait: the act stores below must not run before the kernel before (shared fc2) is done
    if (part) return;
    // C regs: (row r, col 2c), (row r, col 2c + 1), (row r + 8, col 2c), (row r + 8, col 2c + 1)
    const int r = lane >> 2, c = lane & 3;
    __half* a0 = A.act + (static_cast<size_t>(slot) * 8 + 2 * c) * I + chunk * 16 + r;
    a0[0] = __float2half_rn(swiglu_lim(ag[0], au[0], A.limit));
    a0[I] = __float2half_rn(swiglu_lim(ag[1], au[1], A.limit));
    a0[8] = __float2half_rn(swiglu_lim(ag[2], au[2], A.limit));
    a0[I + 8] = __float2half_rn(swiglu_lim(ag[3], au[3], A.limit));
}

// ---------------------------------------------------------------------------------------------------- fc2
// CTA = one slot x WARPS output tiles (a warp each, all k). The slot's activations become a B fragment table in smem.
// grid = n_slots * (NOUT / 16 / WARPS).
// PDLM = 1: the whole-layer PDL chain instance (trigger at the start; the wait after the first weight loads is the kernel's own).
template <int I, int NOUT, int WARPS, int DEPTH, bool PROBE, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_fc2(Args A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, UPS = OT / WARPS, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "fc2 tiling");
    __shared__ __align__(16) u32 sb[KSI * 64];
    if constexpr (PDLM != 0) pdl_trigger();   // the combine behind starts early and waits at its top for this grid
    const int slot = blockIdx.x / UPS, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (A.n_slots_dev && slot >= *A.n_slots_dev)
    {
        if constexpr (PDLM != 0) pdl_wait();
        return;
    }
    const int ot = (blockIdx.x % UPS) * WARPS + warp;
    const size_t eb = static_cast<size_t>(A.slot_expert[slot]) * OT * NG * GR;
    const uint8_t* s = A.fc2 + eb + static_cast<size_t>(ot) * NG * GR;
    // issue the first records before the table build so their latency overlaps it
    Grp nx[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NG) nx[p].load(s + p * GR, lane);
    pdl_wait();
    for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
    {
        const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
        sb[i] = __ldcg(reinterpret_cast<const u32*>(A.act + (static_cast<size_t>(slot) * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
    }
    __syncthreads();
    const u32* bf = sb + lane * 2;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int g = 0; g < NG; g += DEPTH)
    {
        Grp cur[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NG) nx[p].load(s + (g + DEPTH + p) * GR, lane);
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NG) cur[p].template run<PROBE>(bf, (g + p) * 4, acc);
    }
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r, mask = A.slot_mask[slot];
    const float* w = A.slot_w + slot * 8;
    __half* y = A.ypart + static_cast<size_t>(slot) * 8 * NOUT;
    if ((mask >> (2 * c)) & 1)
    {
        y[(2 * c) * NOUT + o] = __float2half_rn(w[2 * c] * acc[0]);
        y[(2 * c) * NOUT + o + 8] = __float2half_rn(w[2 * c] * acc[2]);
    }
    if ((mask >> (2 * c + 1)) & 1)
    {
        y[(2 * c + 1) * NOUT + o] = __float2half_rn(w[2 * c + 1] * acc[1]);
        y[(2 * c + 1) * NOUT + o + 8] = __float2half_rn(w[2 * c + 1] * acc[3]);
    }
}

// ================================================================================================ multi-tile (M 9..32)
// Slot = one distinct expert (dsr::PlanMT, ds_plan_mt.h); its nt = ceil(n / 8) <= NT token tiles run over weight records read once.
// act is indexed by GLOBAL tile: act[tile][8][I], ypart[tile][8][NOUT]; tile = slot_tile0[slot] + j. The M <= 8 kernels above are unchanged.
struct ArgsMT
{
    const uint8_t* fc1;        // [E][I/16][K1/64][2][GR]
    const uint8_t* fc2;        // [E][NOUT/16][I/64][GR]
    const u32* xfrag;          // fc1 B tables, NT x [K1/16][32][2] (build_xfrag_mt); tile stride K1 * 4 words
    const int* slot_expert;    // [slot], ascending
    const int* slot_tile0;     // [slot] first global tile
    const int* slot_nt;        // [slot] tiles of the slot
    const int* tile_mask;      // [tile]
    const float* tile_w;       // [tile][8] routing weight of column c
    const int* tile_tok;       // [tile][8] token of column c, -1 = unused
    const int* n_slots_dev;    // nullptr = the grid is exact; else CTAs of slots >= *n_slots_dev exit
    int n_slots;               // slots the grids are launched for
    float limit;               // swiglu_limit (10)
    __half* act;               // [tile][8][I]
    __half* ypart;             // [tile][8][NOUT]
    const int* order;          // k_fc1_ro / k_fc2_ro with ORD: CTA group `pos` works slot order[pos] (k_order_ro); nullptr elsewhere. Added after the members above: their offsets are unchanged
    int la;                    // k_fc1_ro / k_fc2_ro with LAON: CTAs of lookahead for the L2 prefetch (ds_ro.h ro_pf_target); 0 elsewhere. LAST member
};

// x [M][K1] bf16 -> NT fp16 fragment tables [tile][K1/16][32][2]; tile t holds tokens 8t..8t+7 as B columns, tokens >= M zero. Tile 0 = build_xfrag's table.
// grid = ceil(NT * K1 * 4 / 256), block 256.
template <int PDLM = 0>
__global__ void build_xfrag_mt(const __nv_bfloat16* x, int M, int K1, int NT, u32* xfrag)
{
    if constexpr (PDLM != 0) pdl_trigger();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NT * K1 * 4) return;
    const dsr::MtFrag f = dsr::mt_frag_decode(i, K1);
    const int tok = f.tile * 8 + f.col;
    u32 v = 0;
    if (tok < M)
    {
        const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(x + static_cast<size_t>(tok) * K1 + f.k0);
        const float2 fl = __bfloat1622float2(p);
        v = pack_h2(fminf(fmaxf(fl.x, -65504.f), 65504.f), fminf(fmaxf(fl.y, -65504.f), 65504.f));
    }
    xfrag[i] = v;
}

// fc1_part for NT tiles: the B stream of tile t, lane = xb + boff[t] (+ 64 words per k16 step). L2H: the weight loads carry the L2 evict_last hint (k_fc1_ro, GR8_DS_MOE_L2HINT).
template <int NGP, int DEPTH, int NT, bool L2H = false>
__device__ __forceinline__ void fc1_part_mt(const uint8_t* s, const u32* xb, const int (&boff)[NT], int nt, int lane, int ks0, float (&ag)[NT][4], float (&au)[NT][4])
{
    Grp ng[DEPTH], nu[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NGP) { ng[p].template loadh<L2H>(s + p * 2 * GR, lane); nu[p].template loadh<L2H>(s + p * 2 * GR + GR, lane); }
    for (int g = 0; g < NGP; g += DEPTH)
    {
        Grp cg[DEPTH], cu[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NGP) { ng[p].template loadh<L2H>(s + (g + DEPTH + p) * 2 * GR, lane); nu[p].template loadh<L2H>(s + (g + DEPTH + p) * 2 * GR + GR, lane); }
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NGP)
            {
                cg[p].template run_mt<NT>(xb, boff, ks0 + (g + p) * 4, nt, ag);
                cu[p].template run_mt<NT>(xb, boff, ks0 + (g + p) * 4, nt, au);
            }
    }
}

// Unroll depth the adapters and the gate use for k_fc1_mt at NT tiles. NT = 4 at DEPTH 2 is 132 regs / 15.2 us per slot (80.7% of the stream);
// DEPTH 1 is 98 regs, no spills, 13.2 us per slot (92.6%), act bit-identical (ds_shared_test mtbench fc1 variants). NT 2 and 3
// are at the stream rate either way and keep DEPTH 2.
template <int NT> constexpr int FC1_MT_DEPTH = NT >= 4 ? 1 : 2;

// CTA = one slot x CW chunks x KS k parts (a warp each); the slot's nt <= NT tiles share every weight load. grid = n_slots * (I / 16 / CW).
// PDLM = 1: the whole-layer PDL chain instance (tail wait before the act stores, a wait on the early exit), as fc1_body.
template <int K1, int I, int KS, int CW, int DEPTH, int NT, int MINB = 0, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32, MINB) k_fc1_mt(ArgsMT A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS, UPS = NCH / CW;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "fc1 tiling");
    __shared__ float4 ps[KS > 1 ? NT : 1][KS * CW * 32 * 2];
    pdl_trigger();
    const int slot = blockIdx.x / UPS, cg = blockIdx.x % UPS;
    if (A.n_slots_dev && slot >= *A.n_slots_dev)
    {
        if constexpr (PDLM != 0) pdl_wait();
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = cg * CW + cl;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot];
    int boff[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        boff[t] = t < nt ? dsr::mt_xoff(A.tile_tok[(tile0 + t) * 8 + (lane >> 2)], lane, K1 * 4) : 0;
    const size_t eb = static_cast<size_t>(A.slot_expert[slot]) * NCH * NG * 2 * GR;
    const uint8_t* s = A.fc1 + eb + (static_cast<size_t>(chunk) * NG + part * NGP) * 2 * GR;
    float ag[NT][4], au[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
    fc1_part_mt<NGP, DEPTH, NT>(s, A.xfrag, boff, nt, lane, part * NGP * 4, ag, au);
    if (KS > 1)
    {
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < nt)
            {
                float4* p = ps[t] + (warp * 32 + lane) * 2;
                p[0] = make_float4(ag[t][0], ag[t][1], ag[t][2], ag[t][3]);
                p[1] = make_float4(au[t][0], au[t][1], au[t][2], au[t][3]);
            }
        __syncthreads();
        if (part == 0)
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t < nt)
                    for (int q = 1; q < KS; ++q)      // fixed order
                    {
                        const float4 g = ps[t][((warp + q) * 32 + lane) * 2], u = ps[t][((warp + q) * 32 + lane) * 2 + 1];
                        ag[t][0] += g.x; ag[t][1] += g.y; ag[t][2] += g.z; ag[t][3] += g.w;
                        au[t][0] += u.x; au[t][1] += u.y; au[t][2] += u.z; au[t][3] += u.w;
                    }
    }
    if constexpr (PDLM != 0) pdl_wait();
    if (part) return;
    const int r = lane >> 2, c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        if (t < nt)
        {
            __half* a0 = A.act + (static_cast<size_t>(tile0 + t) * 8 + 2 * c) * I + chunk * 16 + r;
            a0[0] = __float2half_rn(swiglu_lim(ag[t][0], au[t][0], A.limit));
            a0[I] = __float2half_rn(swiglu_lim(ag[t][1], au[t][1], A.limit));
            a0[8] = __float2half_rn(swiglu_lim(ag[t][2], au[t][2], A.limit));
            a0[I + 8] = __float2half_rn(swiglu_lim(ag[t][3], au[t][3], A.limit));
        }
}

// CTA = one slot x WARPS output tiles (a warp each, all k). The slot's nt <= NT activation tiles become B fragment tables in smem
// (tile stride KSI * 64 words). grid = n_slots * (NOUT / 16 / WARPS).
// PDLM = 1: the whole-layer PDL chain instance (trigger at the start, a wait on the early exit), as fc2_body.
template <int I, int NOUT, int WARPS, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_fc2_mt(ArgsMT A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, UPS = OT / WARPS, KSI = I / 16;
    static_assert(I % KG == 0 && OT % WARPS == 0, "fc2 tiling");
    __shared__ __align__(16) u32 sb[NT][KSI * 64];
    if constexpr (PDLM != 0) pdl_trigger();
    const int slot = blockIdx.x / UPS, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (A.n_slots_dev && slot >= *A.n_slots_dev)
    {
        if constexpr (PDLM != 0) pdl_wait();
        return;
    }
    const int ot = (blockIdx.x % UPS) * WARPS + warp;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot];
    const size_t eb = static_cast<size_t>(A.slot_expert[slot]) * OT * NG * GR;
    const uint8_t* s = A.fc2 + eb + static_cast<size_t>(ot) * NG * GR;
    Grp nx[DEPTH];
    #pragma unroll
    for (int p = 0; p < DEPTH; ++p)
        if (p < NG) nx[p].load(s + p * GR, lane);
    pdl_wait();
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        if (t < nt)
            for (int i = threadIdx.x; i < KSI * 64; i += WARPS * 32)
            {
                const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
                sb[t][i] = __ldcg(reinterpret_cast<const u32*>(A.act + (static_cast<size_t>(tile0 + t) * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
            }
    __syncthreads();
    int boff[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) boff[t] = t * KSI * 64 + lane * 2;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    const u32* xb = &sb[0][0];
    for (int g = 0; g < NG; g += DEPTH)
    {
        Grp cur[DEPTH];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + DEPTH + p < NG) nx[p].load(s + (g + DEPTH + p) * GR, lane);
        #pragma unroll
        for (int p = 0; p < DEPTH; ++p)
            if (g + p < NG) cur[p].template run_mt<NT>(xb, boff, (g + p) * 4, nt, acc);
    }
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        if (t < nt)
        {
            const int mask = A.tile_mask[tile0 + t];
            const float* w = A.tile_w + (tile0 + t) * 8;
            __half* y = A.ypart + static_cast<size_t>(tile0 + t) * 8 * NOUT;
            if ((mask >> (2 * c)) & 1)
            {
                y[(2 * c) * NOUT + o] = __float2half_rn(w[2 * c] * acc[t][0]);
                y[(2 * c) * NOUT + o + 8] = __float2half_rn(w[2 * c] * acc[t][2]);
            }
            if ((mask >> (2 * c + 1)) & 1)
            {
                y[(2 * c + 1) * NOUT + o] = __float2half_rn(w[2 * c + 1] * acc[t][1]);
                y[(2 * c + 1) * NOUT + o + 8] = __float2half_rn(w[2 * c + 1] * acc[t][3]);
            }
        }
}
}   // namespace mx4
