// Bit-exact prefill dense GEMM fast path (GR8_PF_DENSE_OPT, default off).
// Provides:
//   - k_gemm_mx8f_u32s<BM, RS, F32, CONST_S>: <=128-reg (2 CTAs/SM) MXFP8 / dtr::make_fp8 prefill GEMM
//     with uint32_t 2-stage paired scale loads (K % 128 == 0), CONST_S=true scale elision for dtr::make_fp8
//     (scales == 127), branchless full-tile loadAB, fa double-buffering, unguarded full_cols tail-row path,
//     and full RS / F32 / osvh / rgrp support.
//   - k_gemm_tr16_clean<16, RS, RR=true>: unguarded full-tile MMA + branchless loadAB/decode + by-fastest /
//     rgrp=4 grouped CTA rasterization for L2 weight reuse on dense trellis prefill GEMMs (o_proj).
// Every kernel executes the identical block-scaled E4M3/UE8M0 MMA sequence and K reduction order as stock
// bindp::k_gemm_mx8b / k_gemm_mx8f / k_gemm_tr (0 diff_bytes).
#pragma once

namespace pf_dense_opt
{

template <int BM, bool RS = false, bool F32 = false, bool CONST_S = false>
__global__ void __launch_bounds__(256, 2) k_gemm_mx8f_u32s(Gemm g0, RsOut rs = RsOut())
{
    Gemm g = g0;
    const bool rr = gridDim.y <= 8;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    if (rr)
    {
        by = lin % gridDim.y;
        bx = lin / gridDim.y;
    }
    else if (g0.rgrp > 1)
    {
        const int per = g0.rgrp * gridDim.x, grp = lin / per, gy0 = grp * g0.rgrp, gh = min(g0.rgrp, gridDim.y - gy0);
        const int r = lin - grp * per;
        by = gy0 + r % gh;
        bx = r / gh;
    }
    if constexpr (!RS)
    {
        if (g0.w2)
        {
            const int nb1 = (g0.N + 127) / 128;
            if (bx >= nb1)
            {
                bx -= nb1;
                g.w = g0.w2; g.ws = g0.ws2; g.N = g0.N2; g.y = g0.y2; g.ldy = g0.ldy2;
                g.bf16 = g0.bf162; g.osvh = g0.osvh2;
                if (g0.x8_2) g.x8 = g0.x8_2;
            }
        }
    }
    using C = G2<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS;
    const int c0 = bx * 128, rb0 = by * BM;
    const int nb = min(BM, RB - rb0), ncols = min(128, g.N - c0);
    const bool full_cols = (ncols == 128);
    const bool full = (nb == BM) && full_cols;

    const int bn0 = t >> 2, bch = t & 3;
    const int b_sm0 = bn0 * 64 + ((bch ^ ((bn0 >> 1) & 3)) << 4);
    const uint8_t* w_base0 = g.w + static_cast<size_t>(c0 + bn0) * g.K + bch * 16;
    const size_t w_half_stride = static_cast<size_t>(64) * g.K;

    const int a0_mb = (BM == 8) ? (t / 33) : 0;
    const int a0_ch = (BM == 8) ? (t % 33) : 0;
    const int a1_j = (BM == 8) ? ((t >= 8) ? 1 : 0) : 0;
    const int a1_rem = (BM == 8) ? ((t >= 8) ? (t - 8) : (t + 256)) : 0;
    const int a1_mb = (BM == 8) ? (a1_rem / 33) : 0;
    const int a1_ch = (BM == 8) ? (a1_rem % 33) : 0;

    auto loadAB = [&](int slot, int st)
    {
        unsigned char* a = sm + slot * C::STAGE;
        if constexpr (BM == 8)
        {
            if (full)
            {
                const int ks = st * G2_KS;
                pf::cp16(a + t * 16, g.x8 + (static_cast<size_t>(rb0 + a0_mb) * S + ks) * REC + a0_ch * 16);
                pf::cp16(a + (t + 256) * 16, g.x8 + (static_cast<size_t>(rb0 + a1_mb) * S + ks + a1_j) * REC + a1_ch * 16);
                if (t < 16)
                    pf::cp16(a + (t + 512) * 16, g.x8 + (static_cast<size_t>(rb0 + 7) * S + ks + 1) * REC + (17 + t) * 16);
                unsigned char* b = a + C::A;
                const uint8_t* wp = w_base0 + st * (G2_KS * 32);
                pf::cp16(b + b_sm0, wp);
                pf::cp16(b + b_sm0 + 4096, wp + w_half_stride);
                return;
            }
        }
        for (int c = t; c < G2_KS * BM * 33; c += 256)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        unsigned char* b = a + C::A;
        const uint8_t* wp = w_base0 + st * (G2_KS * 32);
        if (full_cols)
        {
            pf::cp16(b + b_sm0, wp);
            pf::cp16(b + b_sm0 + 4096, wp + w_half_stride);
        }
        else
        {
            if (bn0 < ncols) pf::cp16(b + b_sm0, wp);
            if (bn0 + 64 < ncols) pf::cp16(b + b_sm0 + 4096, wp + w_half_stride);
        }
    };

    const int colw = wn * C::WC;
    const int nvalid = min(NT, max(0, (ncols - colw) / 8));
    const uint8_t* sp = CONST_S ? nullptr : (g.ws + static_cast<size_t>(c0 + colw + (lane >> 2)) * S);
    const bool use_u32s = !CONST_S && ((S & 3) == 0) && ((reinterpret_cast<uintptr_t>(g.ws) & 3u) == 0u);

    auto loadS4 = [&](int st, u32 (&v)[NT])
    {
        if constexpr (!CONST_S)
        {
            const int ks = st * G2_KS;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                if (nt < nvalid)
                    v[nt] = __ldg(reinterpret_cast<const u32*>(sp + static_cast<size_t>(nt) * 8 * S + ks));
                else
                    v[nt] = 0x7f7f7f7fu;
            }
        }
    };
    auto loadS2 = [&](int st, u32 (&v)[NT])
    {
        if constexpr (!CONST_S)
        {
            const int ks = st * G2_KS;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                if (nt < nvalid)
                    v[nt] = static_cast<u32>(__ldg(reinterpret_cast<const uint16_t*>(sp + static_cast<size_t>(nt) * 8 * S + ks)));
                else
                    v[nt] = 0x7f7fu;
            }
        }
    };

    #pragma unroll
    for (int p = 0; p < G2_ST - 1; ++p)
    {
        if (p < NST) loadAB(p, p);
        pf::cp_commit();
    }
    u32 ns[NT];
    u32 cs[NT];
    if constexpr (!CONST_S)
    {
        if (use_u32s) loadS4(0, ns);
        else loadS2(0, ns);
    }

    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;
    const int wm4 = wm * 4;
    const int mb_lim = min(4, max(0, nb - wm4));

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<G2_ST - 2>();
        __syncthreads();
        if (st + G2_ST - 1 < NST) loadAB((st + G2_ST - 1) % G2_ST, st + G2_ST - 1);
        pf::cp_commit();
        if constexpr (!CONST_S)
        {
            if (use_u32s)
            {
                if ((st & 1) == 0)
                {
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) cs[nt] = ns[nt];
                    if (st + 2 < NST) loadS4(st + 2, ns);
                }
                else
                {
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) cs[nt] >>= 16;
                }
            }
            else
            {
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) cs[nt] = ns[nt];
                if (st + 1 < NST) loadS2(st + 1, ns);
            }
        }
        const unsigned char* a = sm + (st % G2_ST) * C::STAGE;
        const unsigned char* b = a + C::A;
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            if (full)
            {
                const unsigned char* rec0 = a + (j * BM + wm4) * REC;
                uint4 fa = *reinterpret_cast<const uint4*>(rec0 + lane * 16);
                u32 sa = rec0[512 + srow];
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    uint4 fa_nx;
                    u32 sa_nx = 0;
                    if (mb + 1 < 4)
                    {
                        const unsigned char* rnx = a + (j * BM + wm4 + mb + 1) * REC;
                        fa_nx = *reinterpret_cast<const uint4*>(rnx + lane * 16);
                        sa_nx = rnx[512 + srow];
                    }
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        const u32 sb = CONST_S ? 127u : (j == 0 ? (cs[nt] & 0xffu) : ((cs[nt] >> 8) & 0xffu));
                        mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, sb);
                    }
                    if (mb + 1 < 4) { fa = fa_nx; sa = sa_nx; }
                }
            }
            else if (full_cols)
            {
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    if (mb >= mb_lim) break;
                    const unsigned char* rec = a + (j * BM + wm4 + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        const u32 sb = CONST_S ? 127u : (j == 0 ? (cs[nt] & 0xffu) : ((cs[nt] >> 8) & 0xffu));
                        mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, sb);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    if (mb >= mb_lim) break;
                    const unsigned char* rec = a + (j * BM + wm4 + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        if (nt < nvalid)
                        {
                            const u32 sb = CONST_S ? 127u : (j == 0 ? (cs[nt] & 0xffu) : ((cs[nt] >> 8) & 0xffu));
                            mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, sb);
                        }
                    }
                }
            }
        }
    }
    pf::cp_wait<0>();
    if (g.osvh)
    {
        gemm_rot_store<BM, NT>(acc, sm, g.osvh, c0, rb0, nb, g.M, wm, colw, [&](int row, int col, const float (&v)[4]) {
            if constexpr (F32) *reinterpret_cast<float4*>(static_cast<float*>(g.y) + static_cast<size_t>(row) * g.ldy + col) = make_float4(v[0], v[1], v[2], v[3]);
            else if constexpr (RS) *reinterpret_cast<uint2*>(rs_row(rs, row, g.ldy) + col) = pack_o4(v, g.bf16);
            else *reinterpret_cast<uint2*>(static_cast<uint16_t*>(g.y) + static_cast<size_t>(row) * g.ldy + col) = pack_o4(v, g.bf16);
        });
        return;
    }

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm4 + mb;
            if (m >= nb || nt >= nvalid) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            if constexpr (F32)
            {
                float* yf = static_cast<float*>(g.y);
                if (row < g.M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * g.ldy + col) = make_float2(acc[mb][nt][0], acc[mb][nt][1]);
                if (row + 8 < g.M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * g.ldy + col) = make_float2(acc[mb][nt][2], acc[mb][nt][3]);
            }
            else if constexpr (RS)
            {
                if (row < g.M) *reinterpret_cast<u32*>(rs_row(rs, row, g.ldy) + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
                if (row + 8 < g.M) *reinterpret_cast<u32*>(rs_row(rs, row + 8, g.ldy) + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
            }
            else
            {
                uint16_t* y = static_cast<uint16_t*>(g.y);
                if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
                if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
            }
        }
}

template <int BM, bool RS, bool RR>
__global__ void __launch_bounds__(GT<BM>::NTH, 1) k_gemm_tr16_clean(Gemm g, RsOut rs)
{
    static_assert(BM == 16, "k_gemm_tr16_clean is specialized for BM=16");
    using T = GT<BM>;
    constexpr int NT = T::NT, ST = T::ST;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = S / G2_KS;

    int bx = blockIdx.x, by = blockIdx.y;
    if constexpr (RR)
    {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        if (gridDim.y <= 8)
        {
            by = lin % gridDim.y;
            bx = lin / gridDim.y;
        }
        else
        {
            constexpr int RGRP = 4;
            const int per = RGRP * gridDim.x, grp = lin / per, gy0 = grp * RGRP, gh = min(RGRP, gridDim.y - gy0);
            const int r = lin - grp * per;
            by = gy0 + r % gh;
            bx = r / gh;
        }
    }
    const int c0 = bx * 128, rb0 = by * BM;
    const int nb = min(BM, RB - rb0);
    const bool full = (nb == BM);
    unsigned char* sdec = sm + ST * T::STG;

    const int a0_mb = t / 33, a0_ch = t % 33;
    const int a1_j = (t >= 16) ? 1 : 0;
    const int a1_rem = (t >= 16) ? (t - 16) : (t + 512);
    const int a1_mb = a1_rem / 33, a1_ch = a1_rem % 33;
    const int r_ti = t >> 3, r_j = r_ti >> 3, r_c = r_ti & 7, r_ch = (t & 7) * 16;
    const uint8_t* tw_bx = g.tw + bx * g.tsB;

    auto loadAB = [&](int st)
    {
        unsigned char* a = sm + (st % ST) * T::STG;
        const int ks_a = st * G2_KS;
        if (full)
        {
            pf::cp16(a + t * 16, g.x8 + (static_cast<size_t>(rb0 + a0_mb) * S + ks_a) * REC + a0_ch * 16);
            pf::cp16(a + (t + 512) * 16, g.x8 + (static_cast<size_t>(rb0 + a1_mb) * S + ks_a + a1_j) * REC + a1_ch * 16);
            if (t < 32)
                pf::cp16(a + (t + 1024) * 16, g.x8 + (static_cast<size_t>(rb0 + 15) * S + ks_a + 1) * REC + (1 + t) * 16);
        }
        else
        {
            if (a0_mb < nb)
                pf::cp16(a + t * 16, g.x8 + (static_cast<size_t>(rb0 + a0_mb) * S + ks_a) * REC + a0_ch * 16);
            if (a1_mb < nb)
                pf::cp16(a + (t + 512) * 16, g.x8 + (static_cast<size_t>(rb0 + a1_mb) * S + ks_a + a1_j) * REC + a1_ch * 16);
            if (t < 32 && 15 < nb)
                pf::cp16(a + (t + 1024) * 16, g.x8 + (static_cast<size_t>(rb0 + 15) * S + ks_a + 1) * REC + (1 + t) * 16);
        }
        if (t < 256)
        {
            const int ks = st * 4 + r_j;
            pf::cp16(a + T::A + t * 16, tw_bx + (ks >> 3) * g.tsG + ((ks & 7) * 8 + r_c) * 128 + r_ch);
        }
    };

    const int r8 = lane >> 2, kc = 2 * (lane & 3);
    Tr3Lane<4, false> tl;
    tl.init(lane);
    auto at = [&](int n, int k) { return n * (G2_KS * 32) + (((k >> 4) ^ ((n >> 1) & 3)) << 4) + (k & 15); };

    const int ti0 = warp, j0 = ti0 >> 3, c0_d = ti0 & 7;
    const int n0_d = c0_d * 16 + r8, k0_d = j0 * 16 + kc;
    const int ti1 = warp + 16, j1 = ti1 >> 3, c1_d = ti1 & 7;
    const int n1_d = c1_d * 16 + r8, k1_d = j1 * 16 + kc;

    auto decode = [&](int st)
    {
        const unsigned char* raw = sm + (st % ST) * T::STG + T::A;
        unsigned char* d = sdec + (st & 1) * T::B;
        {
            const u32* tile = reinterpret_cast<const u32*>(raw + ti0 * 128);
            u32 f0[2], f1[2];
            tl.decode(tile[tl.i0], tile[tl.i1], f0, f1);
            *reinterpret_cast<uint16_t*>(d + at(n0_d, k0_d)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[0]));
            *reinterpret_cast<uint16_t*>(d + at(n0_d, k0_d + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[1]));
            *reinterpret_cast<uint16_t*>(d + at(n0_d + 8, k0_d)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[0]));
            *reinterpret_cast<uint16_t*>(d + at(n0_d + 8, k0_d + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[1]));
        }
        {
            const u32* tile = reinterpret_cast<const u32*>(raw + ti1 * 128);
            u32 f0[2], f1[2];
            tl.decode(tile[tl.i0], tile[tl.i1], f0, f1);
            *reinterpret_cast<uint16_t*>(d + at(n1_d, k1_d)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[0]));
            *reinterpret_cast<uint16_t*>(d + at(n1_d, k1_d + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[1]));
            *reinterpret_cast<uint16_t*>(d + at(n1_d + 8, k1_d)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[0]));
            *reinterpret_cast<uint16_t*>(d + at(n1_d + 8, k1_d + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[1]));
        }
    };

    #pragma unroll
    for (int p = 0; p < ST - 1; ++p)
    {
        if (p < NST) loadAB(p);
        pf::cp_commit();
    }
    pf::cp_wait<ST - 2>();
    __syncthreads();
    decode(0);

    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;

    const int colw = wn * T::WC;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;
    const int wm4 = wm * 4;
    const int mb_lim = min(4, max(0, nb - wm4));

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<ST - 3>();
        __syncthreads();
        if (st + ST - 1 < NST) loadAB(st + ST - 1);
        pf::cp_commit();
        if (st + 1 < NST) decode(st + 1);
        const unsigned char* a = sm + (st % ST) * T::STG;
        const unsigned char* b = sdec + (st & 1) * T::B;
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            if (full)
            {
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    const unsigned char* rec = a + (j * BM + wm4 + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, 127u);
                }
            }
            else
            {
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    if (mb >= mb_lim) break;
                    const unsigned char* rec = a + (j * BM + wm4 + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, 127u);
                }
            }
        }
    }
    pf::cp_wait<0>();
    if (g.osvh)
    {
        gemm_rot_store<BM, NT, T::NW>(acc, sm, g.osvh, c0, rb0, nb, g.M, wm, colw, [&](int row, int col, const float (&v)[4]) {
            if constexpr (RS) *reinterpret_cast<uint2*>(rs_row(rs, row, g.ldy) + col) = pack_o4(v, g.bf16);
            else *reinterpret_cast<uint2*>(static_cast<uint16_t*>(g.y) + static_cast<size_t>(row) * g.ldy + col) = pack_o4(v, g.bf16);
        });
        return;
    }
    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm4 + mb;
            if (m >= nb) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            if constexpr (RS)
            {
                if (row < g.M) *reinterpret_cast<u32*>(rs_row(rs, row, g.ldy) + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
                if (row + 8 < g.M) *reinterpret_cast<u32*>(rs_row(rs, row + 8, g.ldy) + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
            }
            else
            {
                uint16_t* y = static_cast<uint16_t*>(g.y);
                if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
                if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
            }
        }
}

inline bool try_launch(const Gemm& g0, const RsOut& rs, bool big, int nsm, cudaStream_t st)
{
    if (g0.K % 64 != 0) return false;
    const int RB = (g0.M + 15) / 16;
    if (g0.tw)
    {
#if G6_MODEL_FLASH
        if (g0.h16 || g0.bf16 == 2) return false;
#endif
        if (!big || knob::pf_tr_bm() == 8 || g0.w2 || g0.bf16 == 2 || g0.N % 128 != 0) return false;
        static bool attr_tr = false;
        if (!attr_tr)
        {
            cudaFuncSetAttribute(k_gemm_tr16_clean<16, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GT<16>::SMEM);
            cudaFuncSetAttribute(k_gemm_tr16_clean<16, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GT<16>::SMEM);
            attr_tr = true;
        }
        const dim3 grid((g0.N + 127) / 128, (RB + 15) / 16);
        if (rs.base) k_gemm_tr16_clean<16, true, true><<<grid, GT<16>::NTH, GT<16>::SMEM, st>>>(g0, rs);
        else k_gemm_tr16_clean<16, false, true><<<grid, GT<16>::NTH, GT<16>::SMEM, st>>>(g0, rs);
        return true;
    }

    const bool f32 = (g0.bf16 == 2);
    if (f32 && !knob::pf_gemmf32()) return false;

    static bool attr_mx = false;
    if (!attr_mx)
    {
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<8, false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<8, true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<8, false, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<8, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<8, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<4, false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<4, true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<4, false, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<4, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8f_u32s<4, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        attr_mx = true;
    }

    Gemm g = with_rgrp(g0);
    const int NB = (g.N + 127) / 128 + (g.w2 ? (g.N2 + 127) / 128 : 0);
    const bool use_bm8 = (RB >= 8) && (NB * ((RB + 7) / 8) >= nsm);
    const bool const_s = !f32 && (g.const_s != 0) && (!g.w2 || g.const_s2 != 0);

    if (use_bm8)
    {
        const dim3 grid(NB, (RB + 7) / 8);
        if (rs.base)
        {
            if (const_s) k_gemm_mx8f_u32s<8, true, false, true><<<grid, 256, G2<8>::SMEM, st>>>(g, rs);
            else k_gemm_mx8f_u32s<8, true, false, false><<<grid, 256, G2<8>::SMEM, st>>>(g, rs);
        }
        else if (f32)
        {
            k_gemm_mx8f_u32s<8, false, true, false><<<grid, 256, G2<8>::SMEM, st>>>(g, rs);
        }
        else if (const_s)
        {
            k_gemm_mx8f_u32s<8, false, false, true><<<grid, 256, G2<8>::SMEM, st>>>(g, rs);
        }
        else
        {
            k_gemm_mx8f_u32s<8, false, false, false><<<grid, 256, G2<8>::SMEM, st>>>(g, rs);
        }
    }
    else
    {
        const dim3 grid(NB, (RB + 3) / 4);
        if (rs.base)
        {
            if (const_s) k_gemm_mx8f_u32s<4, true, false, true><<<grid, 256, G2<4>::SMEM, st>>>(g, rs);
            else k_gemm_mx8f_u32s<4, true, false, false><<<grid, 256, G2<4>::SMEM, st>>>(g, rs);
        }
        else if (f32)
        {
            k_gemm_mx8f_u32s<4, false, true, false><<<grid, 256, G2<4>::SMEM, st>>>(g, rs);
        }
        else if (const_s)
        {
            k_gemm_mx8f_u32s<4, false, false, true><<<grid, 256, G2<4>::SMEM, st>>>(g, rs);
        }
        else
        {
            k_gemm_mx8f_u32s<4, false, false, false><<<grid, 256, G2<4>::SMEM, st>>>(g, rs);
        }
    }
    return true;
}

}   // namespace pf_dense_opt
