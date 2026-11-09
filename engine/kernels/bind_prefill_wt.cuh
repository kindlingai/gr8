// Routed prefill experts over expert tiles (GR8_PF_WT; included by bind_prefill.cuh inside namespace bindp).
//
// The segment kernels (k_pf_fc1f, k_pf_fc2g) decode an expert's trellis weights once per 64-row segment and quantize
// the fc1 input once per (segment, matrix, block pair). At 4096-row lanes that is ~40 instructions per QMMA and half the
// launched segment work is padding, so the tensor pipe idles. Here a CTA takes a tile of up to WT_R = 128 routes of one
// expert (k_pf_fill's wtiles) and one 128-column weight block, so every decoded weight tile feeds up to 16 QMMAs, and the
// fc1 input is quantized once per token (k_pf_hq8):
//   hq8 / hs8  mx8(f16(H128(x * sg))) per token row, E4M3 bytes [M][HID] in the TR3 k order + one E8M0 byte per 32 k
//   k_pf_fc1t  CTA = (tile, matrix, block): z1 = sum over the 128-k blocks kb of  v1b[kb] * (hq8_kb @ e4m3(W_kb)), so
//              the block scale of the input rotation (sign x scale) multiplies the block's fp32 partial product instead
//              of the values before quantization (the Flash GR8_PF_Q8 arithmetic). The accumulators hold the running sum
//              in units of the current block's scale (acc *= v1b[kb - 1] / v1b[kb] between blocks), which keeps the row
//              blocks of accumulators in registers with no second set.
//   k_pf_fc2t  CTA = (tile, 128-column group): one Hadamard per group over the fp32 sum of the rank's blocks, each scaled
//              by its v3b entry (the Flash GR8_PF_FC2ONE arithmetic, the same running-units trick), then yw = f16(w * y).
//              With the fused combine (Expert::wtc, GR8_PF_WT_FUSE) the route partials never make the trip through DRAM and
//              back through k_pf_fin: after a warp has written a route's slice it counts the route in cnt_tg[token][group],
//              and the warp whose count completes the token sums its routes in route order (k_pf_fin's arithmetic: fp32
//              adds in selection order, then the shared expert's slice), writes the token's bf16 slice of out (or the RS
//              send slot) and drops the partial lines from the L2 unwritten (discard.global.L2). Integer counters only, a
//              fixed summation order: deterministic. Steering needs a whole-row dot product, so a steered pass keeps fin.
// The tile is 128 rows: 8 row blocks of accumulators keep a warp under 128 registers, so 16 warps (four per scheduler)
// fit an SM. The trellis decode's dependent integer chains need that many warps: with 16 row blocks (250 registers, 8
// warps per SM) fc2 ran as slowly as the segment kernel. A CTA is 16 warps = two 8-warp halves that share one A stage and
// compute two weight units from it (fc1: the gate and up matrices of a block, fc2: two 128-column groups). The staged A
// bytes were the kernels' largest traffic (cp.async from L2 at ~0.6 TB/s chip-wide).
// Not byte-equal to the segment kernels (the fc1 input is quantized on a different grid); deterministic: every sum has a
// fixed order and no atomics. Block-scale layers only (Expert::bs). The segment kernels stay as GR8_PF_WT=0.
#pragma once

constexpr int WT_NM = WT_R / 16;                      // row blocks of a tile (WT_R: bind_prefill.cuh)
constexpr int WT_SA = WT_R * 128;                     // fc1 A stage: WT_R rows x 128 E4M3 bytes, 16-B chunk k of row r at slot k ^ (r & 7)
constexpr int WT_SS = WT_R * 4;                       // fc1 scale stage: the row's 4 E8M0 bytes of the 128-k block
constexpr int FC1T_SMEM = 2 * (WT_SA + WT_SS) + WT_R * 4;
constexpr int F2T_SA = WT_NM * 4 * REC;               // fc2 A stage: the tile's x2 records of one expert block
constexpr int F2T_SZ = WT_R * 256;                    // fc2 output tile of one group: WT_R rows x 128 fp16
// WT_R = 128 (the default): 16-warp CTAs = two 8-warp halves sharing one A stage (NM <= 8 row blocks of accumulators, 128
// registers a thread). WT_R = 256 (-DPF_WT_R=256): 8-warp CTAs, one unit each, NM <= 16 (the registers of one CTA per SM).
constexpr int WT_HALVES = WT_R == 128 ? 2 : 1;
constexpr int WT_NTH = 256 * WT_HALVES;               // threads
constexpr int FC2T_SMEM = 2 * F2T_SA > WT_HALVES * F2T_SZ ? 2 * F2T_SA : WT_HALVES * F2T_SZ;   // the halves' output tiles alias the A stages
static_assert(WT_R == 128 || WT_R == 256, "PF_WT_R is 128 or 256");

__device__ __forceinline__ void wt_cp4(void* s, const void* g)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(static_cast<u32>(__cvta_generic_to_shared(s))), "l"(g));
}
__device__ __forceinline__ void wt_ldsm4(u32 (&r)[4], const void* p)
{
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// The fc1 input records: grid ceil(M / 8) * (HID / 128) warps in blocks of 8, warp = (8 rows, 128-k block), lane = (row,
// k32 step). The values are x * sg rotated (had128_l32) and rounded to fp16, as the segment prologue computes them before
// its block scale. The E4M3 bytes of a k32 step land in the TR3 k order (word c: k 2c, 2c+1, 2c+16, 2c+17; word 4 + c:
// + 8). The decoded B fragments use that order, so ldmatrix of a row's 16-B chunks gives the A fragment.
__global__ void __launch_bounds__(256) k_pf_hq8(Expert A)
{
    constexpr int NB = HID / 128;
    const int w = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int rb = w / NB, kb = w - rb * NB, r = rb * 8 + (lane >> 2), q = lane & 3;
    if (rb * 8 >= A.M) return;                   // warp-uniform
    const bool on = r < A.M;
    const __half* xr = A.x + static_cast<size_t>(on ? r : 0) * A.ldx + kb * 128 + q * 32;
    const __half* sp = A.t_sg + kb * 128 + q * 32;
    float v[32];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const uint4 xv = *reinterpret_cast<const uint4*>(xr + j * 8);
        const uint4 sv = *reinterpret_cast<const uint4*>(sp + j * 8);
        const u32 xw[4] = {xv.x, xv.y, xv.z, xv.w}, sw[4] = {sv.x, sv.y, sv.z, sv.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float2 x2 = __half22float2(as_h2(xw[e])), s2 = __half22float2(as_h2(sw[e]));
            v[j * 8 + e * 2] = __fmul_rn(x2.x, s2.x);
            v[j * 8 + e * 2 + 1] = __fmul_rn(x2.y, s2.y);
        }
    }
    had128_l32(v, lane);
    if (!on) return;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        v[i] = rnd_h(v[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    float inv;
    const u32 code = mx_scale(amax, inv);
    u32 o[8];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        o[c] = f2_e4m3x2(v[2 * c] * inv, v[2 * c + 1] * inv) | (f2_e4m3x2(v[2 * c + 16] * inv, v[2 * c + 17] * inv) << 16);
        o[4 + c] = f2_e4m3x2(v[2 * c + 8] * inv, v[2 * c + 9] * inv) | (f2_e4m3x2(v[2 * c + 24] * inv, v[2 * c + 25] * inv) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(A.hq8 + static_cast<size_t>(r) * HID + kb * 128 + q * 32);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    A.hs8[static_cast<size_t>(r) * (HID / 32) + kb * 4 + q] = static_cast<uint8_t>(code);
}

// acc *= f, warp-uniform f
template <int NM> __device__ __forceinline__ void wt_scale(float (&acc)[NM][2][4], float f)
{
    #pragma unroll
    for (int mb = 0; mb < NM; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] *= f;
}

// The decoded B fragments of k32 step s (two TR3 tiles) of the warp's 16-col tile. cw = the lane's wide-order words of
// the 128-k group. cv = word `lane` of each of its 8 tiles (Tr3Lane::loadw: one coalesced load per tile, and shuffles hand
// each lane the two words it decodes).
template <int BITS, bool WIDE, int CB>
__device__ __forceinline__ void wt_bfrag(const Tr3Lane<BITS, false, CB>& ld, const u64 (&cw)[3], const u32 (&cv)[8], int s, int lane, u32 (&bf)[4])
{
    if constexpr (WIDE)
    {
        const u32 c0 = trw_chunk<BITS>(cw, 2 * s), c1 = trw_chunk<BITS>(cw, 2 * s + 1);
        const u32 b0 = __shfl_sync(0xffffffffu, c0, (lane + 31) & 31), b1 = __shfl_sync(0xffffffffu, c1, (lane + 31) & 31);
        u32 p0[2], p1[2], q0[2], q1[2];
        Tr3Lane<BITS, false, CB>::decode_m((static_cast<u64>(b0) << (8 * BITS)) | c0, 7 * BITS, p0, p1);
        Tr3Lane<BITS, false, CB>::decode_m((static_cast<u64>(b1) << (8 * BITS)) | c1, 7 * BITS, q0, q1);
        tr3_b8f(p0, p1, q0, q1, bf);
    }
    else
    {
        u32 p0[2], p1[2], q0[2], q1[2];
        ld.decodew(cv[2 * s], p0, p1);
        ld.decodew(cv[2 * s + 1], q0, q1);
        tr3_b8f(p0, p1, q0, q1, bf);
    }
}

// fc1 of one (tile, matrix, block) over NM row blocks (NM >= the tile's). The stage of 128-k block kb + 1 is copied
// while block kb runs, and the weight words of kb + 1 are loaded into registers at the top of block kb (one barrier per
// block). A row past the tile's rows stages row 0 and stores nothing.
// Both halves of the CTA run this with their own matrix, weights and scales and the same NM. The barriers inside line up
// because both loop over the same 128-k blocks with one barrier each. The stage copy is split over all WT_NTH threads.
template <int BITS, bool WIDE, int NM, int CB, int EB>
__device__ __forceinline__ void fc1t_body(const Expert& A, const uint8_t* mw, const __half* v1b, int r0, int rows, int b, int mat,
                                          const int* sRow, unsigned char* sA, unsigned char* sS)
{
    constexpr int TB = 32 * BITS, NB = HID / 128;
    constexpr size_t WG = static_cast<size_t>(2048) * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = (t >> 5) & 7;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    Tr3Lane<BITS, false, CB> ld;
    ld.init(lane);
    float acc[NM][2][4];
    #pragma unroll
    for (int mb = 0; mb < NM; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    const uint8_t* wg = mw + warp * 256 + lane * 8;        // wide order: the lane's first u64 of a group
    const uint8_t* wb = mw + warp * TB;                     // tile order: the warp's tile column
    u32 nv[8];
    u64 nw[3] = {0, 0, 0};
    if constexpr (WIDE)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + n * 2048);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) nv[i] = ld.loadw(wb + static_cast<size_t>(i) * 8 * TB, lane);
    }
    auto stage = [&](int kb) {                               // the tile's rows of 128-k block kb: 8 chunks of 16 B a row
        unsigned char* a = sA + (kb & 1) * WT_SA;
        #pragma unroll
        for (int ch = t; ch < NM * 128; ch += WT_NTH)
        {
            const int r = ch >> 3, k = ch & 7;
            if (r < rows) pf::cp16(a + r * 128 + ((k ^ (r & 7)) << 4), A.hq8 + static_cast<size_t>(sRow[r]) * HID + kb * 128 + k * 16);
        }
        if (t < NM * 16 && t < rows) wt_cp4(sS + (kb & 1) * WT_SS + t * 4, A.hs8 + static_cast<size_t>(sRow[t]) * (HID / 32) + kb * 4);
        pf::cp_commit();
    };
    stage(0);
    float cprev = 1.f;
    for (int kb = 0; kb < NB; ++kb)
    {
        pf::cp_wait<0>();
        __syncthreads();                                     // block kb landed, and block kb - 1 is done with the other stage
        if (kb + 1 < NB) stage(kb + 1);
        u32 cv[8];
        u64 cw[3];
        if constexpr (WIDE)
        {
            #pragma unroll
            for (int n = 0; n < 3; ++n) cw[n] = nw[n];
            if (kb + 1 < NB)
            {
                #pragma unroll
                for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + static_cast<size_t>(kb + 1) * WG + n * 2048);
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = nv[i];
            if (kb + 1 < NB)
            {
                #pragma unroll
                for (int i = 0; i < 8; ++i) nv[i] = ld.loadw(wb + static_cast<size_t>((kb + 1) * 8 + i) * 8 * TB, lane);
            }
        }
        const float c = __half2float(v1b[kb]);
        if (c == 0.f) continue;                              // a zero block scale adds nothing (never seen in a pack). The scales carry signs.
        if (c != cprev) wt_scale(acc, cprev / c);            // the running sum into units of this block's scale
        cprev = c;
        const unsigned char* xs = sA + (kb & 1) * WT_SA;
        const unsigned char* ss = sS + (kb & 1) * WT_SS;
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            u32 bf[4];
            wt_bfrag<BITS, WIDE, CB>(ld, cw, cv, s, lane, bf);
            #pragma unroll
            for (int mb = 0; mb < NM; ++mb)
            {
                const int Rl = mb * 16 + (lane & 15), ch = 2 * s + (lane >> 4);
                u32 a[4];
                wt_ldsm4(a, xs + Rl * 128 + ((ch ^ (Rl & 7)) << 4));
                const u32 sa = ss[(mb * 16 + srow) * 4 + s];
                mma_mx8(acc[mb][0], make_uint4(a[0], a[1], a[2], a[3]), bf[0], bf[1], sa, 127u);
                mma_mx8(acc[mb][1], make_uint4(a[0], a[1], a[2], a[3]), bf[2], bf[3], sa, 127u);
            }
        }
    }
    wt_scale(acc, cprev);
    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < NM; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int col = warp * 16 + h * 8 + 2 * c, row = mb * 16 + r;
            __half* z = A.z1 + (static_cast<size_t>(r0 + row) * (2 * EB) + b * 2 + mat) * 128 + col;
            if (row < rows) *reinterpret_cast<u32*>(z) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
            if (row + 8 < rows) *reinterpret_cast<u32*>(z + 8 * (2 * EB) * 128) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
        }
}

// the body for the tile's row-block count, rounded up to an even NM. One body per even count keeps the row-block loop
// free of branches between the mma (a branch between them cost the Flash kernels ~30 dead cycles per 4 mma).
template <int BITS, bool WIDE, int CB, int EB>
__device__ __forceinline__ void fc1t_nm(const Expert& A, const uint8_t* mw, const __half* v1b, int r0, int rows, int b, int mat,
                                        const int* sRow, unsigned char* sA, unsigned char* sS)
{
    switch ((rows + 31) / 32)
    {
        case 1: fc1t_body<BITS, WIDE, 2, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break;
        case 2: fc1t_body<BITS, WIDE, 4, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break;
        case 3: fc1t_body<BITS, WIDE, 6, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break;
        case 4: fc1t_body<BITS, WIDE, 8, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break;
        case 5: if constexpr (WT_NM > 8) { fc1t_body<BITS, WIDE, 10, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break; } [[fallthrough]];
        case 6: if constexpr (WT_NM > 8) { fc1t_body<BITS, WIDE, 12, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break; } [[fallthrough]];
        case 7: if constexpr (WT_NM > 8) { fc1t_body<BITS, WIDE, 14, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break; } [[fallthrough]];
        default: if constexpr (WT_NM > 8) fc1t_body<BITS, WIDE, 16, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); else fc1t_body<BITS, WIDE, 8, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS); break;
    }
}

// grid (EB blocks, tiles), block WT_NTH (warps 0-7 the gate matrix, 8-15 up; WT_R = 256: grid (2 EB, tiles), x = mat EB +
// block), FC1T_SMEM dynamic smem. K2 and K3 units come in the wide order (the loader's order on the runtime path), K4 in
// the tile order. make_layer keeps the segment kernels for anything else.
template <int EB = g6nr::EB, bool K2 = false, int CB = TR3_MCG>
__global__ void __launch_bounds__(WT_NTH, 1) k_pf_fc1t(Expert A)
{
    extern __shared__ __align__(16) unsigned char smt[];
    unsigned char* const sA = smt;
    unsigned char* const sS = sA + 2 * WT_SA;
    int* const sRow = reinterpret_cast<int*>(sS + 2 * WT_SS);
    if (static_cast<int>(blockIdx.y) >= *A.n_wtiles) return;
    const int* w = A.wtiles + static_cast<size_t>(blockIdx.y) * 3;
    const long long* tr = A.t_tab + static_cast<size_t>(w[0]) * 6;
    const int mat = WT_HALVES == 2 ? threadIdx.x >> 8 : blockIdx.x / EB, b = WT_HALVES == 2 ? blockIdx.x : blockIdx.x - (blockIdx.x / EB) * EB;
    if (b >= static_cast<int>(tr[3])) return;
    const int r0 = w[1], rows = w[2];
    if (threadIdx.x < WT_R) sRow[threadIdx.x] = static_cast<int>(threadIdx.x) < rows ? A.rr_row[r0 + threadIdx.x] : 0;
    __syncthreads();
    size_t off;
    const int k = tr3_unit(A.t_kmix, w[0], tr, b * 2 + mat, off);
    const uint8_t* mw = A.t_fc1 + off;
    const __half* v1b = A.t_v1b + ((static_cast<size_t>(tr[2]) + b) * 2 + mat) * (HID / 128);
    if (k == 4) fc1t_nm<4, false, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS);
    else if (k == 3) fc1t_nm<3, true, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS);
    else if constexpr (K2) fc1t_nm<2, true, CB, EB>(A, mw, v1b, r0, rows, b, mat, sRow, sA, sS);
}

__device__ __forceinline__ void wt_combine(const Expert& A, int G, int tok, bool act, int q, int lane);

// fc2 of one (tile, 128-col group) over NM row blocks: the records of expert block b + 1 are copied while block b runs.
// y = H128(sum_b v3b[b] z_b) * sign (one Hadamard per group), yw = f16(w * y).
template <int BITS, bool WIDE, int NM, int CB, int EB>
__device__ __forceinline__ void fc2t_body(const Expert& A, const long long* tr, int wt, int r0, int rows, int G0, unsigned char* sA)
{
    constexpr int TB = 32 * BITS, SA = F2T_SA, NCH = NM * 4 * (REC / 16);
    const int t = threadIdx.x, lane = t & 31, warp = (t >> 5) & 7, half = WT_HALVES == 2 ? t >> 8 : 0, G = G0 + half;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0), c = lane & 3, r = lane >> 2;
    const int nb = static_cast<int>(tr[3]);
    Tr3Lane<BITS, false, CB> ld;
    ld.init(lane);
    auto wptr = [&](int b) {
        return A.t_fc2 + tr[1] + static_cast<size_t>(b) * (static_cast<size_t>(KSH) * 8 * TB) + static_cast<size_t>(G * 8 + warp) * 8 * TB;
    };
    const uint8_t* xsrc = A.x2 + static_cast<size_t>(wt) * WT_NM * EB * 4 * REC;       // row block mb, block b at + (mb EB + b) 4 REC
    auto stage = [&](int b) {
        unsigned char* d = sA + (b & 1) * SA;
        for (int ch = t; ch < NCH; ch += WT_NTH)
        {
            const int mb = ch / (4 * (REC / 16)), rem = ch - mb * (4 * (REC / 16));
            pf::cp16(d + ch * 16, xsrc + (static_cast<size_t>(mb) * EB + b) * 4 * REC + rem * 16);
        }
        pf::cp_commit();
    };
    float acc[NM][2][4];
    #pragma unroll
    for (int mb = 0; mb < NM; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    stage(0);
    u32 nv[8];
    u64 nw[3] = {0, 0, 0};
    if constexpr (WIDE)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wptr(0) + n * 256 + lane * 8);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) nv[i] = ld.loadw(wptr(0) + i * TB, lane);
    }
    float sprev = 1.f;
    for (int b = 0; b < nb; ++b)
    {
        pf::cp_wait<0>();
        __syncthreads();
        if (b + 1 < nb) stage(b + 1);
        u32 cv[8];
        u64 cw[3];
        if constexpr (WIDE)
        {
            #pragma unroll
            for (int n = 0; n < 3; ++n) cw[n] = nw[n];
            if (b + 1 < nb)
            {
                #pragma unroll
                for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wptr(b + 1) + n * 256 + lane * 8);
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = nv[i];
            if (b + 1 < nb)
            {
                #pragma unroll
                for (int i = 0; i < 8; ++i) nv[i] = ld.loadw(wptr(b + 1) + i * TB, lane);
            }
        }
        const float sc = __half2float(A.t_v3b[(static_cast<size_t>(tr[2]) + b) * (HID / 128) + G]);
        if (sc == 0.f) continue;
        if (sc != sprev) wt_scale(acc, sprev / sc);
        sprev = sc;
        const unsigned char* sAb = sA + (b & 1) * SA;
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            u32 bf[4];
            wt_bfrag<BITS, WIDE, CB>(ld, cw, cv, s, lane, bf);
            #pragma unroll
            for (int mb = 0; mb < NM; ++mb)
            {
                const unsigned char* rec = sAb + (mb * 4 + s) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                mma_mx8(acc[mb][0], fa, bf[0], bf[1], sa, 127u);
                mma_mx8(acc[mb][1], fa, bf[2], bf[3], sa, 127u);
            }
        }
    }
    wt_scale(acc, sprev);
    __syncthreads();                                          // every warp is done with the A stages: the output tiles take their place
    unsigned char* const sZb = sA + half * F2T_SZ;
    #pragma unroll
    for (int mb = 0; mb < NM; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int row = mb * 16 + r, ch = warp * 2 + h;
            *reinterpret_cast<u32*>(sZb + row * 256 + swz16(ch, row) * 16 + 4 * c) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
            *reinterpret_cast<u32*>(sZb + (row + 8) * 256 + swz16(ch, row + 8) * 16 + 4 * c) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
        }
    __syncthreads();
    // 32 rows a pass: warp w rows 4 w .. 4 w + 3, a lane one row's 16 columns of eighth q (had128_l16)
    const int pr = warp * 4 + (lane >> 3), q = lane & 7;
    const __half* sgp = A.t_sg + HID + G * 128 + q * 16;     // the output signs
    float sv[16];
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        const uint4 sr = *reinterpret_cast<const uint4*>(sgp + j * 8);
        const u32 sw[4] = {sr.x, sr.y, sr.z, sr.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float2 s2 = __half22float2(as_h2(sw[e]));
            sv[j * 8 + e * 2] = s2.x; sv[j * 8 + e * 2 + 1] = s2.y;
        }
    }
    #pragma unroll 1
    for (int p = 0; p < NM / 2; ++p)
    {
        if (warp * 4 + p * 32 >= rows) break;                  // warp-uniform
        const int rp = pr + p * 32;
        const bool act = rp < rows;
        float v[16];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint4 zr = act ? *reinterpret_cast<const uint4*>(sZb + rp * 256 + swz16(q * 2 + j, rp) * 16) : make_uint4(0u, 0u, 0u, 0u);
            const u32 zw[4] = {zr.x, zr.y, zr.z, zr.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float2 z2 = __half22float2(as_h2(zw[e]));
                v[j * 8 + e * 2] = z2.x; v[j * 8 + e * 2 + 1] = z2.y;
            }
        }
        had128_l16(v, lane);
        if (act)
        {
            const float w = A.rr_w[r0 + rp];
            __half* yp = A.yw + static_cast<size_t>(r0 + rp) * HID + G * 128 + q * 16;
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                uint4 o;
                o.x = pack_h2(w * (v[j * 8] * sv[j * 8]), w * (v[j * 8 + 1] * sv[j * 8 + 1]));
                o.y = pack_h2(w * (v[j * 8 + 2] * sv[j * 8 + 2]), w * (v[j * 8 + 3] * sv[j * 8 + 3]));
                o.z = pack_h2(w * (v[j * 8 + 4] * sv[j * 8 + 4]), w * (v[j * 8 + 5] * sv[j * 8 + 5]));
                o.w = pack_h2(w * (v[j * 8 + 6] * sv[j * 8 + 6]), w * (v[j * 8 + 7] * sv[j * 8 + 7]));
                *reinterpret_cast<uint4*>(yp + j * 8) = o;
            }
        }
        if (A.wtc) wt_combine(A, G, act ? A.rr_row[r0 + rp] : 0, act, q, lane);
    }
}

__device__ __forceinline__ void wt_discard(const void* p) { asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory"); }
__device__ __forceinline__ uint4 wt_ldcg(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }   // from the L2: other SMs wrote the line

// The fused combine of one 32-row epilogue pass: lanes 8 i .. 8 i + 7 hold row i (token tok, live when act); after their yw
// slice is written (and fenced), the row's leader counts the route in cnt_tg[tok][G]; a row whose count reaches topk sums the
// token's routes in route order (A.pos), adds the shared expert's slice as k_pf_fin does (SHX: H128(ysh) x svh x scale,
// rounded to fp16), writes the bf16 slice of out / the RS slot, and drops the routes' partial lines from the L2. Every
// lane runs the Hadamard (warp shuffles); only the completing rows' lanes load, store and discard.
__device__ __forceinline__ void wt_combine(const Expert& A, int G, int tok, bool act, int q, int lane)
{
    __threadfence();                                           // this lane's partial stores are visible device-wide
    __syncwarp();                                              // every lane of the row fenced before the leader counts
    int last = 0;
    if (act && q == 0) last = atomicAdd(A.cnt_tg + static_cast<size_t>(tok) * (HID / 128) + G, 1) + 1 == A.topk;
    last = __shfl_sync(0xffffffffu, last, lane & 24);
    if (!__any_sync(0xffffffffu, last)) return;
    if (last) __threadfence();                                 // the other routes' stores happened before their counts
    float a[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) a[j] = 0.f;
    const size_t col = static_cast<size_t>(G) * 128 + q * 16;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        const int pk = last ? A.pos[tok * TOPK + k] : -1;
        if (pk < 0) continue;                                  // a dropped route (GR8_MOE_TOPK), or not a completing row
        const __half* src = A.yw + static_cast<size_t>(pk) * HID + col;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint4 r = wt_ldcg(src + j * 8);
            const u32 rw[4] = {r.x, r.y, r.z, r.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float2 f = __half22float2(as_h2(rw[e]));
                a[j * 8 + e * 2] += f.x; a[j * 8 + e * 2 + 1] += f.y;
            }
        }
    }
    __syncwarp();                                              // every lane's partial reads are done before a line is dropped
    if (last)
        if (const int pk = A.pos[tok * TOPK + q]; pk >= 0)     // lane q drops route q's two 128-B lines
        {
            wt_discard(A.yw + static_cast<size_t>(pk) * HID + G * 128);
            wt_discard(A.yw + static_cast<size_t>(pk) * HID + G * 128 + 64);
        }
    // the shared expert's slice (k_pf_fin: SHX rotates ysh, else adds it)
    const __half* ys = A.ysh + static_cast<size_t>(tok) * HID + col;
    float sh[16];
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        const uint4 r = last ? *reinterpret_cast<const uint4*>(ys + j * 8) : make_uint4(0u, 0u, 0u, 0u);
        const u32 rw[4] = {r.x, r.y, r.z, r.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float2 f = __half22float2(as_h2(rw[e]));
            sh[j * 8 + e * 2] = f.x; sh[j * 8 + e * 2 + 1] = f.y;
        }
    }
    if (A.sh_svh)
    {
        had128_l16(sh, lane);
        const float c = A.sh_sc ? __half2float(A.sh_sc[G]) : 1.f;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint4 r = *reinterpret_cast<const uint4*>(A.sh_svh + col + j * 8);
            const u32 rw[4] = {r.x, r.y, r.z, r.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float2 f = __half22float2(as_h2(rw[e]));
                a[j * 8 + e * 2] += rnd_h(sh[j * 8 + e * 2] * (f.x * c));          // fin: sv = svh * c, then rnd_h(v * sv)
                a[j * 8 + e * 2 + 1] += rnd_h(sh[j * 8 + e * 2 + 1] * (f.y * c));
            }
        }
    }
    else
    {
        #pragma unroll
        for (int j = 0; j < 16; ++j) a[j] += sh[j];
    }
    if (!last) return;
    uint16_t* dst = (A.rs.base ? rs_row(A.rs, tok, HID) : A.out + static_cast<size_t>(tok) * HID) + col;
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        uint4 o;
        o.x = pack_bf2(a[j * 8], a[j * 8 + 1]); o.y = pack_bf2(a[j * 8 + 2], a[j * 8 + 3]);
        o.z = pack_bf2(a[j * 8 + 4], a[j * 8 + 5]); o.w = pack_bf2(a[j * 8 + 6], a[j * 8 + 7]);
        *reinterpret_cast<uint4*>(dst + j * 8) = o;
    }
}

template <int BITS, bool WIDE, int CB, int EB>
__device__ __forceinline__ void fc2t_nm(const Expert& A, const long long* tr, int wt, int r0, int rows, int G, unsigned char* sA)
{
    switch ((rows + 31) / 32)
    {
        case 1: fc2t_body<BITS, WIDE, 2, CB, EB>(A, tr, wt, r0, rows, G, sA); break;
        case 2: fc2t_body<BITS, WIDE, 4, CB, EB>(A, tr, wt, r0, rows, G, sA); break;
        case 3: fc2t_body<BITS, WIDE, 6, CB, EB>(A, tr, wt, r0, rows, G, sA); break;
        case 4: fc2t_body<BITS, WIDE, 8, CB, EB>(A, tr, wt, r0, rows, G, sA); break;
        case 5: if constexpr (WT_NM > 8) { fc2t_body<BITS, WIDE, 10, CB, EB>(A, tr, wt, r0, rows, G, sA); break; } [[fallthrough]];
        case 6: if constexpr (WT_NM > 8) { fc2t_body<BITS, WIDE, 12, CB, EB>(A, tr, wt, r0, rows, G, sA); break; } [[fallthrough]];
        case 7: if constexpr (WT_NM > 8) { fc2t_body<BITS, WIDE, 14, CB, EB>(A, tr, wt, r0, rows, G, sA); break; } [[fallthrough]];
        default: if constexpr (WT_NM > 8) fc2t_body<BITS, WIDE, 16, CB, EB>(A, tr, wt, r0, rows, G, sA); else fc2t_body<BITS, WIDE, 8, CB, EB>(A, tr, wt, r0, rows, G, sA); break;
    }
}

// grid (HID / 256 group pairs, tiles), or (tiles, HID / 256) when A.wt_gmajor (WT_R = 256: single groups, HID / 128) (the groups of a token's partials written close
// in time, at the price of re-reading x2 once per group pair): block WT_NTH (warps 0-7 the pair's first group, 8-15 the
// second), FC2T_SMEM dynamic smem. The expert's fc2 width is tr[4].
template <int EB = g6nr::EB, bool K2 = false, int CB = TR3_MCG>
__global__ void __launch_bounds__(WT_NTH, 1) k_pf_fc2t(Expert A)
{
    extern __shared__ __align__(16) unsigned char smt[];
    const int wt = A.wt_gmajor ? blockIdx.x : blockIdx.y, G = WT_HALVES * (A.wt_gmajor ? blockIdx.y : blockIdx.x);
    if (wt >= *A.n_wtiles) return;
    const int* w = A.wtiles + static_cast<size_t>(wt) * 3;
    const long long* tr = A.t_tab + static_cast<size_t>(w[0]) * 6;
    const int r0 = w[1], rows = w[2];
    if (tr[4] == 4) fc2t_nm<4, false, CB, EB>(A, tr, wt, r0, rows, G, smt);
    else if (tr[4] == 3) fc2t_nm<3, true, CB, EB>(A, tr, wt, r0, rows, G, smt);
    else if constexpr (K2) fc2t_nm<2, true, CB, EB>(A, tr, wt, r0, rows, G, smt);
}

inline int wt_tiles(const Expert& A) { return max_wtiles_t(A.M); }   // the grid's tile bound (CTAs past *n_wtiles exit)

template <int EB = g6nr::EB> inline void launch_fc1t(const Expert& A, cudaStream_t st)
{
    if (!A.hq8_ready) k_pf_hq8<<<((A.M + 7) / 8 * (HID / 128) + 7) / 8, 256, 0, st>>>(A);
    tr3_cb_dispatch(A.cb, [&](auto C) {
        constexpr int CB = decltype(C)::value;
        static bool attr = false;
        if (!attr)
        {
            if (cudaFuncSetAttribute(k_pf_fc1t<EB, false, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, FC1T_SMEM) != cudaSuccess ||
                cudaFuncSetAttribute(k_pf_fc1t<EB, true, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, FC1T_SMEM) != cudaSuccess)
                fprintf(stderr, "bindp: k_pf_fc1t cannot take %d B of dynamic shared memory: %s\n", FC1T_SMEM, cudaGetErrorString(cudaGetLastError()));
            attr = true;
        }
        const dim3 g(EB * (2 / WT_HALVES), wt_tiles(A));
        if (A.k2) k_pf_fc1t<EB, true, CB><<<g, WT_NTH, FC1T_SMEM, st>>>(A);
        else k_pf_fc1t<EB, false, CB><<<g, WT_NTH, FC1T_SMEM, st>>>(A);
    });
}
// the fused combine runs unless the pass steers (k_pf_fin's whole-row dot product)
inline bool wt_fused(const Expert& A) { return A.wtc && !(A.steer_dir && A.steer_scale != 0.f && A.steer_alpha != 0.f); }
template <int EB = g6nr::EB> inline void launch_fc2t(const Expert& A0, cudaStream_t st)
{
    Expert A = A0;
    A.wtc = wt_fused(A0) ? 1 : 0;
    if (A.wtc) cudaMemsetAsync(A.cnt_tg, 0, static_cast<size_t>(A.M) * (HID / 128) * sizeof(int), st);
    tr3_cb_dispatch(A.cb, [&](auto C) {
        constexpr int CB = decltype(C)::value;
        static bool attr = false;
        if (!attr)
        {
            if (cudaFuncSetAttribute(k_pf_fc2t<EB, false, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, FC2T_SMEM) != cudaSuccess ||
                cudaFuncSetAttribute(k_pf_fc2t<EB, true, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, FC2T_SMEM) != cudaSuccess)
                fprintf(stderr, "bindp: k_pf_fc2t cannot take %d B of dynamic shared memory: %s\n", FC2T_SMEM, cudaGetErrorString(cudaGetLastError()));
            attr = true;
        }
        const dim3 g = A.wt_gmajor ? dim3(wt_tiles(A), HID / (128 * WT_HALVES)) : dim3(HID / (128 * WT_HALVES), wt_tiles(A));
        if (A.k2) k_pf_fc2t<EB, true, CB><<<g, WT_NTH, FC2T_SMEM, st>>>(A);
        else k_pf_fc2t<EB, false, CB><<<g, WT_NTH, FC2T_SMEM, st>>>(A);
    });
}
