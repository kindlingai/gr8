// gr8 DeepSeek-V4.1-Flash shared expert, group loop.
// shared_ro runs the shared expert in 32-row chunks, 2 launches per chunk (k_sh_fc1_mt / k_sh_fc2_mt of ds_shared.cuh): 2 x W/32 launches per layer, and every launch is a
// 36- or 40-CTA grid of 8 warps. These two kernels do the same work in two launches per layer: grid.y = RG row groups, a CTA loops over the chunks of its group.
//
// Per chunk a CTA runs, unchanged, the same bodies as the chunk kernels: fc1_part_mt<NGP, DEPTH, NT> (same k groups in the same order, same GvGroup::run<NT>), the same KS
// partial reduce in smem in the same fixed order q = 1..KS-1, the same swiglu clamp and fp16 store; fc2 the same act staging and the same GvGroup::run<NT> loop. The
// accumulators are zeroed at the top of every chunk, nothing is carried from one chunk to the next. A row's value therefore has the same sequence of operations as in
// k_sh_fc1_mt / k_sh_fc2_mt, whatever RG, NT, CW or WARPS is (the NT-independence of GvGroup::run is the comment at densev.cuh:145). The gate is the GPU test `check`
// (bitwise vs the token-alone M = 1 path, every token): ds_prefill_ro_test.cu.
//
// Differences from the chunk kernels (none of them touches a value):
//   * a chunk is NT * 8 rows (NT = 4: the same 32 rows as the chunk kernels); chunk c covers rows [c * NT * 8, +NT * 8); rows = the total row count M of the call.
//   * a tile whose first row is >= rows is not stored (fc1) / its output rows are not written (fc2): the chunk kernels wrote the whole last 32-row chunk of act.
//     Caller contract: the xfrag table holds ceil(rows / (NT * 8)) * NT tiles (tiles >= build_xfrag_mt's are read but only feed discarded columns) and act holds
//     ceil(rows / (NT * 8)) * NT * 8 rows.
//   * the smem of both kernels is dynamic (NT = 8 does not fit the 48 KB static limit); launch_shared_gl sets the attribute when needed.
#pragma once
#include "ds_shared.cuh"

namespace dsh
{
struct ArgsGL
{
    const uint8_t* w1;          // gate [I/16][K1/32][GR]
    const uint8_t* w3;          // up   [I/16][K1/32][GR]
    const uint8_t* w2;          // down [NOUT/16][I/32][GR]
    const u32* xfrag;           // fc1 B tables of ALL rows: tile t = rows 8t..8t+7 at t * K1 * 4 words (mx4::build_xfrag_mt)
    int rows;                   // M, total
    float limit;                // swiglu_limit (10)
    __half* act;                // [rows rounded up to NT * 8][I]
    float* out;                 // [rows][NOUT]
};

// CTA = CW chunks x KS k parts (a warp each), loops over its row groups' chunks. grid = (I / 16 / CW, RG). dynamic smem = fc1_smem<KS, CW, NT>().
// PDLM = 1: the whole-layer PDL chain instance (tail wait before the act stores), as k_sh_fc1_mt.
template <int KS, int CW, int NT> constexpr size_t fc1_gl_smem() { return KS > 1 ? static_cast<size_t>(NT) * KS * CW * 32 * 2 * sizeof(float4) : 0; }
template <int K1, int I, int KS, int CW, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(KS * CW * 32) k_sh_fc1_gl(ArgsGL A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, NGP = NG / KS, RC = NT * 8, PSN = KS * CW * 32 * 2;
    static_assert(K1 % KG == 0 && NG % KS == 0 && NCH % CW == 0, "shared fc1 tiling");
    extern __shared__ __align__(16) unsigned char gl_smem1[];
    float4* ps = reinterpret_cast<float4*>(gl_smem1);                // [NT][PSN] (KS > 1 only)
    mx4::pdl_trigger();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / KS, part = warp % KS, chunk = blockIdx.x * CW + cl;
    const size_t off = (static_cast<size_t>(chunk) * NG + part * NGP) * GR;
    const int nch = (A.rows + RC - 1) / RC;
    for (int c = blockIdx.y; c < nch; c += gridDim.y)
    {
        const int row0 = c * RC;
        const u32* xf = A.xfrag + static_cast<size_t>(c) * NT * (K1 * 4);
        float ag[NT][4], au[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
        fc1_part_mt<NGP, DEPTH, NT>(A.w1 + off, A.w3 + off, xf + lane * 2, K1 * 4, lane, part * NGP * 2, ag, au);
        if (KS > 1)
        {
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                float4* p = ps + t * PSN + (warp * 32 + lane) * 2;
                p[0] = make_float4(ag[t][0], ag[t][1], ag[t][2], ag[t][3]);
                p[1] = make_float4(au[t][0], au[t][1], au[t][2], au[t][3]);
            }
            __syncthreads();
            if (part == 0)
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                    for (int q = 1; q < KS; ++q)      // fixed order
                    {
                        const float4 g = ps[t * PSN + ((warp + q) * 32 + lane) * 2], u = ps[t * PSN + ((warp + q) * 32 + lane) * 2 + 1];
                        ag[t][0] += g.x; ag[t][1] += g.y; ag[t][2] += g.z; ag[t][3] += g.w;
                        au[t][0] += u.x; au[t][1] += u.y; au[t][2] += u.z; au[t][3] += u.w;
                    }
        }
        if constexpr (PDLM != 0) mx4::pdl_wait();
        if (part == 0)
        {
            const int r = lane >> 2, cc = lane & 3;
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                if (row0 + t * 8 >= A.rows) continue;
                __half* a0 = A.act + static_cast<size_t>(row0 + t * 8 + 2 * cc) * I + chunk * 16 + r;
                a0[0] = __float2half_rn(mx4::swiglu_lim(ag[t][0], au[t][0], A.limit));
                a0[I] = __float2half_rn(mx4::swiglu_lim(ag[t][1], au[t][1], A.limit));
                a0[8] = __float2half_rn(mx4::swiglu_lim(ag[t][2], au[t][2], A.limit));
                a0[I + 8] = __float2half_rn(mx4::swiglu_lim(ag[t][3], au[t][3], A.limit));
            }
        }
        if (KS > 1) __syncthreads();                  // part 0 has read ps: the next chunk may overwrite it
    }
}

// CTA = WARPS output tiles (a warp each, all k); NT activation tiles become smem B tables (stride KSI * 64 words), loops over its row groups' chunks. grid = (NOUT / 16 / WARPS, RG).
// dynamic smem = NT * (I / 16) * 64 * 4 B. PDLM = 1: the whole-layer PDL chain instance (trigger after the wait), as k_sh_fc2_mt.
template <int I, int NT> constexpr size_t fc2_gl_smem() { return static_cast<size_t>(NT) * (I / 16) * 64 * sizeof(u32); }
template <int I, int NOUT, int WARPS, int DEPTH, int NT, int PDLM = 0>
__global__ void __launch_bounds__(WARPS * 32) k_sh_fc2_gl(ArgsGL A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, KSI = I / 16, RC = NT * 8;
    static_assert(I % KG == 0 && OT % WARPS == 0, "shared fc2 tiling");
    extern __shared__ __align__(16) unsigned char gl_smem2[];
    u32* sb = reinterpret_cast<u32*>(gl_smem2);                       // [NT][KSI * 64]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ot = blockIdx.x * WARPS + warp;
    const uint8_t* s = A.w2 + static_cast<size_t>(ot) * NG * GR;
    const int nch = (A.rows + RC - 1) / RC;
    GvGroup<Mxfp8> nx[DEPTH];
    for (int c = blockIdx.y; c < nch; c += gridDim.y)
    {
        const int row0 = c * RC;
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
                sb[t * (KSI * 64) + i] = __ldcg(reinterpret_cast<const u32*>(A.act + static_cast<size_t>(row0 + t * 8 + (l >> 2)) * I + ks * 16 + reg * 8 + 2 * (l & 3)));
            }
        __syncthreads();
        const u32* bf = sb + lane * 2;
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
        const int r = lane >> 2, cc = lane & 3, o = ot * 16 + r;
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const int gr0 = row0 + t * 8 + 2 * cc;
            if (gr0 < A.rows)
            {
                A.out[static_cast<size_t>(gr0) * NOUT + o] = acc[t][0];
                A.out[static_cast<size_t>(gr0) * NOUT + o + 8] = acc[t][2];
            }
            if (gr0 + 1 < A.rows)
            {
                A.out[static_cast<size_t>(gr0 + 1) * NOUT + o] = acc[t][1];
                A.out[static_cast<size_t>(gr0 + 1) * NOUT + o + 8] = acc[t][3];
            }
        }
        __syncthreads();                              // every warp is done with sb: the next chunk may restage it
    }
}

// Host launcher shared by the adapter and the GPU test: fc1 then fc2, plain launches. rg_max = the most row groups (grid.y); the grid uses min(rg_max, chunks), so a CTA always has a chunk.
// Returns cudaGetLastError(). The smem attributes are set only when the dynamic size exceeds 48 KB (host call, no stream work).
template <int K1, int I, int NOUT, int KS, int CW, int D1, int W2, int D2, int NT, int PDLM = 0>
inline int launch_shared_gl(const ArgsGL& A, int rg_max, cudaStream_t st)
{
    constexpr int RC = NT * 8;
    constexpr size_t S1 = fc1_gl_smem<KS, CW, NT>(), S2 = fc2_gl_smem<I, NT>();
    auto* k1 = k_sh_fc1_gl<K1, I, KS, CW, D1, NT, PDLM>;
    auto* k2 = k_sh_fc2_gl<I, NOUT, W2, D2, NT, PDLM>;
    if (S1 > 48 * 1024) cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S1));
    if (S2 > 48 * 1024) cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S2));
    const int nch = (A.rows + RC - 1) / RC;
    const int rg = rg_max < nch ? (rg_max > 0 ? rg_max : 1) : nch;
    k1<<<dim3(I / 16 / CW, rg), dim3(KS * CW * 32), S1, st>>>(A);
    k2<<<dim3(NOUT / 16 / W2, rg), dim3(W2 * 32), S2, st>>>(A);
    return static_cast<int>(cudaGetLastError());
}
}   // namespace dsh
