// GR8_DS_ATTN_TC_TILE=2: k_gemm_persist_ldm_epi_bid, launch_gemm_tc's 128 x 128 tile as a persistent kernel. Included by ds41_attn.cuh inside namespace ds41a,
// after k_gemm_mx8t_sw / TcCfgSw / tc_mma (no namespace and no include guard of its own). Only the work distribution, the fragment loads and the stores differ from
// k_gemm_mx8t_sw; per output element the mma sequence is the same (acc = +0, K stages in order, 4 ks per stage, same fragments, same scale bytes), so Y is
// k_gemm_mx8t_sw's, and k_gemm_mx8t's, bit for bit (tests/ds41_gemm_persist_test.cu, tests/ds41_gemm_tc_bench.cu).
//   persistent: grid = min(SM count, tiles) CTAs; CTA b walks tiles b, b + grid, b + 2 grid, ... in launch order (N tile fastest, then M tile, then group).
//     The cp.async ring is ONE flat sequence of (tile, k-stage) loads that does not drain at a tile boundary: during the last ST-1 stages of tile i the loads
//     for the first stages of tile i+1 are already in flight.
//   ldmatrix.x4 fragment loads: per stage per warp 16 A + 8 B ldmatrix.x4 instead of 128 4-byte LDS (same fragments).
//   byte-id scale select: the E8M0 byte of k-step ks is chosen by the mma byte-id operand (an immediate) instead of a shift and mask per mma.
//   epilogue: the bf16 tile is staged in smem and written with 16-B stores along full rows (needs ldy % 8 == 0, 16-B aligned Y; launch_tc_persist checks).
__device__ __forceinline__ void tc_ldm_x4(uint32_t (&r)[4], uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr) : "memory");
}
// Same instruction as ds41a::tc_mma, but the scale operands are the whole 4-byte words and BID (an immediate 0..3) is the byte-id operand.
template <int BID> __device__ __forceinline__ void tc_mma_bid(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "n"(BID), "h"(z), "r"(sb), "n"(BID), "h"(z));
}
// ks is a compile-time constant after the full unroll of the ks loop; the switch folds to one instruction.
__device__ __forceinline__ void tc_mma_ks(int ks, float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
    switch (ks)
    {
    case 0: tc_mma_bid<0>(d, a, b0, b1, sa, sb); break;
    case 1: tc_mma_bid<1>(d, a, b0, b1, sa, sb); break;
    case 2: tc_mma_bid<2>(d, a, b0, b1, sa, sb); break;
    default: tc_mma_bid<3>(d, a, b0, b1, sa, sb); break;
    }
}
// A tile walker: (row tile, column tile, group) in stock launch order (N tile fastest, then M tile, then group). tc_pr_adv steps it by the grid
// size G tiles with no division: G = dz * (nt * mt) + dm * nt + dn is split once at kernel start.
struct TcPrWalk { int mi, ni, z; };
struct TcPrStep { int dz, dm, dn; };
__device__ __forceinline__ TcPrWalk tc_pr_decode(int tile, int nt, int tpg)
{
    const int z = tile / tpg, r = tile - z * tpg, mi = r / nt;
    return TcPrWalk{mi, r - mi * nt, z};
}
__device__ __forceinline__ void tc_pr_adv(TcPrWalk& w, const TcPrStep& s, int nt, int mt)
{
    int ni = w.ni + s.dn, carry = 0;
    if (ni >= nt) { ni -= nt; carry = 1; }
    int mi = w.mi + s.dm + carry, zc = 0;
    if (mi >= mt) { mi -= mt; zc = 1; }
    w.ni = ni;
    w.mi = mi;
    w.z += s.dz + zc;
}

// p.mfast is ignored (the walk is N tile fastest). groups = the launch's group count (GemmP has none); launch_tc_persist sets the grid.
template <int BM, int BN, int ST> __global__ void __launch_bounds__(256) k_gemm_persist_ldm_epi_bid(GemmP p, int groups)
{
    using C = TcCfgSw<BM, BN, ST>;
    static_assert(BM * 8 % 256 == 0 && BN * 8 % 256 == 0 && BM <= 256 && BN <= 256, "issue() assumes whole 256-thread passes");
    constexpr int NA = BM * 8 / 256, NB = BN * 8 / 256;
    extern __shared__ __align__(16) uint8_t tsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    const int nt = p.N / BN, mt = (p.M + BM - 1) / BM, tpg = nt * mt, total = tpg * groups, G = gridDim.x, NK = p.K >> 7, KB = p.K >> 5;
    const TcPrStep step = [&]
    {
        const int dz = G / tpg, r = G - dz * tpg, dm = r / nt;
        return TcPrStep{dz, dm, r - dm * nt};
    }();
    const int my_tiles = (total - (int)blockIdx.x + G - 1) / G;   // blockIdx.x < total by the grid choice

    // ---- load cursor: the flat (tile, k stage) sequence this CTA's ring is filled from. Per-thread source pointers advance with the stage.
    TcPrWalk lw = tc_pr_decode(blockIdx.x, nt, tpg);
    int lkt = 0;
    const uint8_t *pa[NA], *pw[NB], *psa = nullptr, *psw = nullptr;
    auto set_ptrs = [&]
    {
        const int lm0 = lw.mi * BM, ln0 = lw.ni * BN;
        const uint8_t* A = p.A + (size_t)lw.z * p.a_gs;
        const uint8_t* W = p.W + (size_t)lw.z * p.w_gs;
#pragma unroll
        for (int i = 0; i < NA; i++) pa[i] = A + (size_t)min(lm0 + (tid >> 3) + 32 * i, p.M - 1) * p.lda + (tid & 7) * 16;
#pragma unroll
        for (int i = 0; i < NB; i++) pw[i] = W + (size_t)(ln0 + (tid >> 3) + 32 * i) * p.K + (tid & 7) * 16;
        if (tid < BM) psa = p.SA + (size_t)lw.z * p.sa_gs + (size_t)min(lm0 + tid, p.M - 1) * p.lsa;
        if (tid < BN) psw = p.SW + (size_t)lw.z * p.sw_gs + (size_t)(ln0 + tid) * KB;
    };
    set_ptrs();
    auto issue = [&](int s)   // one stage of the cursor's tile into ring slot s, one commit group (empty past the last tile), advance the cursor
    {
        if (lw.z < groups)
        {
            uint8_t* b = tsm + (size_t)s * C::STAGE;
#pragma unroll
            for (int i = 0; i < NA; i++)
            {
                tc_cp16(b + tc_swz((tid >> 3) + 32 * i, tid & 7), pa[i]);
                pa[i] += 128;
            }
#pragma unroll
            for (int i = 0; i < NB; i++)
            {
                tc_cp16(b + C::A_B + tc_swz((tid >> 3) + 32 * i, tid & 7), pw[i]);
                pw[i] += 128;
            }
            if (tid < BM) { tc_cp4(b + C::A_B + C::W_B + tid * 4, psa); psa += 4; }
            if (tid < BN) { tc_cp4(b + C::A_B + C::W_B + C::SA_B + tid * 4, psw); psw += 4; }
        }
        tc_commit();
        if (++lkt == NK)
        {
            lkt = 0;
            tc_pr_adv(lw, step, nt, mt);
            if (lw.z < groups) set_ptrs();
        }
    };

    int ls = ST - 1;   // ring slot the next load goes to; the prologue fills slots 0 .. ST-2
#pragma unroll
    for (int s = 0; s < ST - 1; s++) issue(s);

    // ldmatrix lane addressing. A: matrix i = rows (g, g+8 -> i&1) x chunk (2ks + (i>>1)); B: matrix i = n-tile (i>>1) x chunk (2ks + (i&1)).
    // Lane l addresses row (l & 7) (+ 8 for matrices 1 and 3) of matrix l >> 3; (row & 7) == (l & 7) for every mb / nt, so the swizzle is one xor per ks.
    const uint32_t smem0 = (uint32_t)__cvta_generic_to_shared(tsm);
    const int lr = lane & 7, xa = ((lane >> 4) & 1) ^ lr, xb = ((lane >> 3) & 1) ^ lr;
    const uint32_t offA = (uint32_t)((wm * C::WM + lr + (((lane >> 3) & 1) << 3)) * 128);
    const uint32_t offB = (uint32_t)((wn * C::WN + lr + (((lane >> 4) & 1) << 3)) * 128);
    int cs = 0;   // ring slot of the stage computed next
    TcPrWalk cw = tc_pr_decode(blockIdx.x, nt, tpg);   // tile being computed
    for (int it = 0; it < my_tiles; it++)
    {
        const int m0 = cw.mi * BM, n0 = cw.ni * BN, cz = cw.z;
        tc_pr_adv(cw, step, nt, mt);
        float acc[C::MB][C::NT][4];
#pragma unroll
        for (int i = 0; i < C::MB; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
        for (int kt = 0; kt < NK; kt++)
        {
            tc_wait<ST - 2>();
            __syncthreads();
            issue(ls);
            ls = (ls + 1 == ST) ? 0 : ls + 1;
            const uint8_t* b = tsm + (size_t)cs * C::STAGE;
            const uint32_t sA_l = smem0 + (uint32_t)(cs * C::STAGE) + offA, sW_l = smem0 + (uint32_t)(cs * C::STAGE + C::A_B) + offB;
            cs = (cs + 1 == ST) ? 0 : cs + 1;
            const uint8_t* sW = b + C::A_B;
            const uint8_t* sSA = sW + C::W_B;
            const uint8_t* sSW = sSA + C::SA_B;
            uint32_t wsa[C::MB], wsw[C::NT];
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++) wsa[mb] = *(const uint32_t*)(sSA + (wm * C::WM + mb * 16 + g + ((lane & 1) ? 8 : 0)) * 4);
#pragma unroll
            for (int nt_ = 0; nt_ < C::NT; nt_++) wsw[nt_] = *(const uint32_t*)(sSW + (wn * C::WN + nt_ * 8 + g) * 4);
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
            {
                uint32_t a[C::MB][4], bb[C::NT][2];
                const uint32_t qa = (uint32_t)((2 * ks) ^ xa) << 4, qb = (uint32_t)((2 * ks) ^ xb) << 4;
#pragma unroll
                for (int mb = 0; mb < C::MB; mb++) tc_ldm_x4(a[mb], sA_l + mb * (16 * 128) + qa);
#pragma unroll
                for (int np = 0; np < C::NT / 2; np++)
                {
                    uint32_t r4[4];
                    tc_ldm_x4(r4, sW_l + np * (16 * 128) + qb);
                    bb[2 * np][0] = r4[0], bb[2 * np][1] = r4[1], bb[2 * np + 1][0] = r4[2], bb[2 * np + 1][1] = r4[3];
                }
#pragma unroll
                for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
                    for (int nt_ = 0; nt_ < C::NT; nt_++) tc_mma_ks(ks, acc[mb][nt_], a[mb], bb[nt_][0], bb[nt_][1], wsa[mb], wsw[nt_]);
            }
        }
        // epilogue: the bf16 tile is staged in smem (256-B rows, 16-B chunk c of row r at chunk c ^ (r & 7)) and written to Y with 16-B coalesced stores along full rows.
        // Ring hazard (the ring is NOT drained here, the next tile's stages 0 and 1 are already loaded or in flight in the other two slots):
        // the only free slot is `ls_last`, the one of this tile's last stage. (1) Warps are still reading it in their last ks steps, so a barrier must come before the staging stores.
        // (2) The next tile's first iteration issues stage 2 into this very slot right after its __syncthreads; that barrier comes after every thread's staging loads
        // (a thread reaches it only after its loads fed its stores), so no extra barrier is needed on that side. (3) No cp.async is in flight into the slot: its stage was
        // waited for at the start of this iteration and the next write to it is the issue above.
        static_assert(BM * BN * 2 <= C::STAGE, "staged tile must fit one ring slot");
        const int ls_last = (cs == 0) ? ST - 1 : cs - 1;
        __syncthreads();   // (1) all warps finished reading the last stage
        uint8_t* const es = tsm + (size_t)ls_last * C::STAGE;
#pragma unroll
        for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
            for (int nt_ = 0; nt_ < C::NT; nt_++)
            {
                const int c = (wn * C::WN + nt_ * 8) / 8, rl = wm * C::WM + mb * 16 + g;   // 16-B chunk (8 bf16) index in the row, row in the tile
                *(__nv_bfloat162*)(es + rl * (BN * 2) + ((c ^ (rl & 7)) << 4) + 4 * t) = __floats2bfloat162_rn(acc[mb][nt_][0], acc[mb][nt_][1]);
                *(__nv_bfloat162*)(es + (rl + 8) * (BN * 2) + ((c ^ (rl & 7)) << 4) + 4 * t) = __floats2bfloat162_rn(acc[mb][nt_][2], acc[mb][nt_][3]);
            }
        __syncthreads();
        bf16* Y = p.Y + (size_t)cz * p.y_gs;
        constexpr int CPR = BN / 8;   // 16-B chunks per row
        static_assert((BM * CPR) % 256 == 0, "whole 256-thread passes");
#pragma unroll
        for (int i = 0; i < (BM * CPR) / 256; i++)
        {
            const int c = tid + i * 256, rl = c / CPR, cc = c % CPR;
            const uint4 v = *(const uint4*)(es + rl * (BN * 2) + ((cc ^ (rl & 7)) << 4));
            if (m0 + rl < p.M) *(uint4*)(Y + (size_t)(m0 + rl) * p.ldy + n0 + cc * 8) = v;
        }
        // no barrier here: the next tile's first iteration starts with tc_wait + __syncthreads before any cp.async write into `es`
    }
    tc_wait<0>();   // only empty groups are left; keeps the CTA from exiting with groups pending
}
