#pragma once
// PRIVATE COPY of mxtc::k_fc2_tc (engine/kernels/ds_moe_tc.cuh:455-607), verbatim except the symbol name.
// To start a variant: `cp -r variants/tmpl variants/NAME && sed -i 's/tmpl/NAME/g' variants/NAME/*`, then edit the kernel here.
// Helpers (cp16, Rec, mma_mx, combine_token, ...) still come from the engine header through `using namespace mxtc`.
#include "kernels/ds41_v2_moe.cuh"
namespace fc2_tmpl {
using namespace mxtc;
template <int I, int NOUT, int WARPS, int TW, int TPC, int ST, int COL_GROUPS = 1>
__global__ void __launch_bounds__(WARPS * 32, 2) k_fc2_tmpl(Args A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, NB = OT / TPC, NTG = WARPS * TW, NTH = WARPS * 32, AST = TPC * GR, ACH = GR / 16, NA = (TPC * ACH + NTH - 1) / NTH;
    constexpr int NSW = (NG + 1) / 2;                                        // scale words per row (2 records each)
    static_assert(I % KG == 0 && OT % TPC == 0 && I % 16 == 0 && ST >= 2 && ST - 1 <= NG && NSW * 4 <= ASP, "fc2 tc tiling");
    static_assert(TPC * 16 <= 128 && (TPC * 16) % 64 == 0 && YP % 8 == 0, "fc2 tc output staging: a row of 16-B pieces; a block = whole 128-B slot lines");
    extern __shared__ __align__(16) uint8_t sm[];
    __half* ys = reinterpret_cast<__half*>(sm + ST * AST) + (threadIdx.x >> 5) * 8 * YP;   // the warp's output stage [8 rows][YP]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, c = lane & 3, tid = threadIdx.x;
    const int GY = A.gy2, OS = A.os2, gy = blockIdx.x % GY, slot = (blockIdx.x / GY) % A.n_slots, os = blockIdx.x / (GY * A.n_slots);
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    if (gy >= ng) return;
    static_assert(NB % COL_GROUPS == 0, "fc2 column groups must contain whole output blocks");
    constexpr int GNB = NB / COL_GROUPS;
    const int gb = COL_GROUPS == 1 ? 0 : A.col_group * GNB;
    const int b0 = gb + os * GNB / OS, nb = (os + 1) * GNB / OS - os * GNB / OS, nstg = nb * NG;
    if (nb == 0) return;
    const uint8_t* s = A.fc2 + dsr::ro_fc2_rec(A.slot_expert[slot], 0, 0, OT, NG) * GR;   // tile ot record r at s + (ot NG + r) GR
    const uint8_t* srca[NA];
    u32 dsta[NA];
    #pragma unroll
    for (int k = 0; k < NA; ++k)
    {
        const int i = tid + k * NTH, j = i / ACH, q = i % ACH;
        srca[k] = s + static_cast<size_t>(j) * NG * GR + q * 16;
        dsta[k] = j * GR + q * 16;
    }
    auto issue = [&](int st, int gs)                                         // stage gs = record gs % NG of tiles (b0 + gs / NG) TPC .. + TPC - 1
    {
        const size_t off = (static_cast<size_t>(b0 + gs / NG) * TPC * NG + gs % NG) * GR;
        #pragma unroll
        for (int k = 0; k < NA; ++k)
            if (tid + k * NTH < TPC * ACH) cp16(sm + st * AST + dsta[k], srca[k] + off);
    };
    #pragma unroll 1
    for (int grp = gy; grp < ng; grp += GY)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, grp), t0 = dsr::ro_tile(tile0, grp, NTG, 0);
        #pragma unroll
        for (int st = 0; st < ST - 1; ++st) { issue(st, st); cp_commit(); }
        // the warp's rows: records and scale words, routing weights and the live mask of its tiles
        uint4 bq[TW][NG];
        u32 sw[TW][NSW];
        float w0[TW], w1[TW];
        int mask[TW];
        const int nlive = min(max(ntg - warp * TW, 0), TW);                  // the warp's live tiles (the first nlive): a dead tile runs no mma
        #pragma unroll
        for (int tw = 0; tw < TW; ++tw)
        {
            const int t = warp * TW + tw;
            const bool live = t < ntg;
            const size_t row = (static_cast<size_t>(t0) + t) * 8 + g;
            #pragma unroll
            for (int r = 0; r < NG; ++r) bq[tw][r] = live ? __ldg(reinterpret_cast<const uint4*>(A.aq + row * I + r * XREC + c * 16)) : make_uint4(0, 0, 0, 0);
            #pragma unroll
            for (int k = 0; k < NSW; ++k) sw[tw][k] = live ? __ldg(reinterpret_cast<const u32*>(A.as + row * ASP + k * 4)) : 0u;
            mask[tw] = live ? A.tile_mask[t0 + t] : 0;
            w0[tw] = live ? A.tile_w[(t0 + t) * 8 + 2 * c] : 0.f;
            w1[tw] = live ? A.tile_w[(t0 + t) * 8 + 2 * c + 1] : 0.f;
        }
        int base = 0;                                                        // ring slot of the block's record 0
        #pragma unroll 1
        for (int bb = 0; bb < nb; ++bb)
        {
            const int b = b0 + bb;
            float acc[TPC][TW][4];
            #pragma unroll
            for (int j = 0; j < TPC; ++j)
                #pragma unroll
                for (int tw = 0; tw < TW; ++tw) acc[j][tw][0] = acc[j][tw][1] = acc[j][tw][2] = acc[j][tw][3] = 0.f;
            static_for<NG>([&](auto rc)
            {
                constexpr int r = decltype(rc)::value;
                cp_wait<ST - 2>();
                __syncthreads();
                const int gs = bb * NG + r;
                if (gs + ST - 1 < nstg) issue((base + r + ST - 1) % ST, gs + ST - 1);
                cp_commit();
                const uint8_t* wa = sm + ((base + r) % ST) * AST;
                #pragma unroll
                for (int j = 0; j < TPC; ++j)
                {
                    Rec R;
                    R.load(wa + j * GR, lane);
                    const u32 a0 = x4lo(R.c[0]), a1 = x4hi(R.c[0]), a2 = x4lo(R.c[1]), a3 = x4hi(R.c[1]);
                    const u32 a4 = x4lo(R.c[2]), a5 = x4hi(R.c[2]), a6 = x4lo(R.c[3]), a7 = x4hi(R.c[3]);
                    #pragma unroll
                    for (int tw = 0; tw < TW; ++tw)
                        if (tw < nlive)
                        {
                            mma_mx<0, (2 * r) & 3>(acc[j][tw], a0, a1, a2, a3, bq[tw][r].x, bq[tw][r].y, R.s, sw[tw][r / 2]);
                            mma_mx<2, (2 * r + 1) & 3>(acc[j][tw], a4, a5, a6, a7, bq[tw][r].z, bq[tw][r].w, R.s, sw[tw][r / 2]);
                        }
                }
            });
            base = (base + NG) % ST;
            #pragma unroll
            for (int tw = 0; tw < TW; ++tw)
            {
                if (tw >= nlive) break;
                #pragma unroll
                for (int j = 0; j < TPC; ++j)                                // fragment order into the stage: (row 2c | 2c+1, column j 16 + g | + 8)
                {
                    __half* r0 = ys + (2 * c) * YP + j * 16 + g;
                    r0[0] = __float2half_rn(w0[tw] * acc[j][tw][0]);
                    r0[8] = __float2half_rn(w0[tw] * acc[j][tw][2]);
                    r0[YP] = __float2half_rn(w1[tw] * acc[j][tw][1]);
                    r0[YP + 8] = __float2half_rn(w1[tw] * acc[j][tw][3]);
                }
                __syncwarp();
                __half* y = A.ypart + (static_cast<size_t>(t0) + warp * TW + tw) * 8 * NOUT + b * (TPC * 16);
                constexpr int PR = TPC * 2;                                  // 16-B pieces per row
                #pragma unroll
                for (int i = lane; i < 8 * PR; i += 32)
                {
                    const int row = i / PR, pc = i % PR;
                    if ((mask[tw] >> row) & 1)
                    {
                        const uint4 v = *reinterpret_cast<const uint4*>(ys + row * YP + pc * 8);
                        if constexpr (MXTC_ABL == 7) { if (v.x == 0x12345678u) y[row * NOUT] = __float2half_rn(1.f); }
                        else *reinterpret_cast<uint4*>(y + static_cast<size_t>(row) * NOUT + pc * 8) = v;
                    }
                }
                __syncwarp();                                               // the stage is rewritten by the next tile / block
            }
        }
        cp_wait<0>();
        if (A.cnt)
        {
            __threadfence();                                                // the slice's ypart stores are visible before the counts
            __syncwarp();                                                   // every lane's fence before the quad leader's count
            #pragma unroll
            for (int tw = 0; tw < TW; ++tw)
            {
                if (tw >= nlive) break;
                const int t = warp * TW + tw;
                const int tok = A.tile_tok[(t0 + t) * 8 + g];
                int last = 0;
                if (c == 0 && ((mask[tw] >> g) & 1))
                    last = atomicAdd(A.cnt + static_cast<size_t>(tok) * OS + os, 1) + 1 == A.kc;
                const unsigned lastg = __ballot_sync(0xffffffffu, last);   // bit 4 g: the warp completes token g's slice
                if (lastg) __threadfence();                                 // the other routes' stores happened before their counts
                for (unsigned bits = lastg; bits; bits &= bits - 1)
                {
                    const int gg = __ffs(bits) - 1;
                    combine_token<NOUT>(A, A.tile_tok[(t0 + t) * 8 + (gg >> 2)], b0 * (TPC * 16), nb * (TPC * 16), lane);   // the CTA's own columns
                }
            }
        }
        __syncthreads();                                                    // the ring restarts for the next group
    }
}
}  // namespace fc2_tmpl
