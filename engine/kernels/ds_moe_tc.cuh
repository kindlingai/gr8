// gr8 DeepSeek-V4.1-Flash read-once prefill MoE on the block-scaled tensor cores (GR8_DS_MOE_TC=1). Our own code.
//
// The same plan, scratch and combine as ds_prefill_ro.cuh (k_route_ro / k_scan_ro / k_place_ro, act [tile][8][I], ypart [tile][8][NOUT],
// dsh::k_combine_mt); only the two expert loops change:
//   weights     the MXFP4 records as stored (mxfp4.cuh "mx4g": 16 rows x 64 k, lane l's 16 B = 4 k16 words, 4 scale bytes). A record word is the
//               fp16 kernels' A fragment; here its nibbles are spread into 8-bit containers (x4lo / x4hi: the E2M1 code sits in bits 5..2 of the
//               container, which is where sm_121a reads it under kind::mxf8f6f4, probed) and fed as the e2m1 A operand of
//               mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 with the record's E8M0 byte as the A scale. The k order inside a k32
//               block is the record's (lane c: k 2c, 2c+1, 2c+8, 2c+9, then + 16), which is free as long as B follows it.
//   activations MXFP8 (E4M3 per 32 k with one E8M0 scale, the ATTN_TC / SH_TC quantizer's formula: scale = 2^ceil(log2(amax / 448)), amax floored
//               at 1e-4), written by k_xq (the layer input x) and k_aq (fc1's fp16 act) in XREC records: per row and 64-k record 64 B, lane c of a
//               quad owns bytes [16 c, 16 c + 16) = {k32 group 0 b0, b1, group 1 b0, b1} in the k order above, so a lane reads one 16-B word per
//               (row, record). Scales are plain [row][K / 32] bytes (k_aq pads the row to ASP bytes).
//   fc1         k_fc1_tc: CTA = (expert slot, CW chunks of 16 intermediate rows), a warp per chunk = gate + up (two A tiles) x NTG token tiles;
//               the expert's tiles are walked in groups of NTG as k_fc1_ro does; K streams through an ST-deep cp.async ring of 128-k stages of the
//               group's rows (gathered by tile_tok). fp32 accumulate over all of K in one warp (no k split), swiglu_lim, fp16 act as k_fc1_ro.
//   fc2         k_fc2_tc: CTA = (expert slot, WARPS x WT output tiles of 16 rows), the group's act records (all 576 k) staged in smem once per
//               group; a warp runs its WT tiles x NTG token tiles. Epilogue = k_fc2_ro's (tile_w, tile_mask, fp16 ypart).
// Not bit-equal to the fp16 path (the activations are rounded to E4M3 and the tensor cores sum in their own order); accepted as a configuration,
// like GR8_DS_ATTN_TC. Plain launches only. Microbench + accuracy gate: tests/ds_moe_tc_bench.cu.
#pragma once
#include "mxfp4.cuh"
#include "ds_ro.h"

namespace mxtc
{
using mx4::GR;
using mx4::KG;
using mx4::SOFF;
constexpr int XREC = 64;          // bytes of one row's 64-k activation record
constexpr int ASP = 32;           // k_aq scale row pitch (I / 32 = 18 bytes used)

struct Args
{
    const uint8_t* fc1;           // [E][I/16][K1/64][2][GR]
    const uint8_t* fc2;           // [E][NOUT/16][I/64][GR]
    const uint8_t* xq;            // [M][K1 / 64][XREC]  the layer input, quantized (k_xq)
    const uint8_t* xs;            // [M][K1 / 32]        its E8M0 bytes
    const int* slot_expert;       // plan (ds_plan_mt.h), as mx4::ArgsMT
    const int* slot_tile0;
    const int* slot_nt;
    const int* tile_mask;
    const float* tile_w;
    const int* tile_tok;
    const int* n_slots_dev;
    const int* n_tiles_dev;       // the plan's tile count: k_aq quantizes rows < 8 * *n_tiles_dev (nullptr = every row of its grid)
    int n_slots;
    int gy1, gy2;                 // token-group split of k_fc1_tc / k_fc2_tc: CTA (x, gy) walks groups gy, gy + GY, ..; gy is the fastest index of blockIdx.x
    float limit;
    __half* act;                  // [tile][8][I] fp16 (fc1 out)
    uint8_t* aq;                  // [tile * 8][I / 64][XREC]  act quantized (k_aq)
    uint8_t* as;                  // [tile * 8][ASP]
    __half* ypart;                // [tile][8][NOUT]
};

// ---------------------------------------------------------------------------------------------------- operands
#ifndef MXTC_ABL
#define MXTC_ABL 0                // bench ablation: 1 = the mma replaced by an xor (every load and address stays)
#endif
// d += 2^(sa - 127) 2^(sb - 127) (A @ B): A 16 rows x 32 k e2m1 containers, B 32 k x 8 cols e4m3. Scale words (probed on sm_121a, tests/.. sel probe):
// byte BA of the word of lane 4 g scales row g, byte BA of lane 4 g + 1 scales row g + 8; byte BB of lane 4 g's sb word scales column g.
template <int BA, int BB>
__device__ __forceinline__ void mma_mx(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1, u32 sa, u32 sb)
{
    if constexpr (MXTC_ABL == 1) { d[0] += __uint_as_float((a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ sa ^ sb) & 0x3fffffffu); return; }
    static_assert(BA >= 0 && BA < 4 && BB >= 0 && BB < 4, "scale byte");
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "n"(BA), "n"(0), "r"(sb), "n"(BB), "n"(0));
}
// A record word w = {row r byte c, row r+8 byte c, row r byte c+4, row r+8 byte c+4} (mxfp4.cuh) -> the four containers of row r (x4lo) and
// row r+8 (x4hi): nibbles (byte c low, byte c high, byte c+4 low, byte c+4 high) at bits 5..2 of containers 0..3.
__device__ __forceinline__ u32 x4lo(u32 w) { return ((w << 2) & 0x003C003Cu) | ((w << 6) & 0x3C003C00u); }
__device__ __forceinline__ u32 x4hi(u32 w) { return ((w >> 6) & 0x003C003Cu) | ((w >> 2) & 0x3C003C00u); }
// One lane's share of a record in smem: the 4 k16 words and its scale word, pre-shifted so byte 2 h is the lane's row scale of k32 group h
// (the record word is {g0 row r, g0 row r+8, g1 row r, g1 row r+8}: an even lane scales row r, an odd lane row r + 8).
struct Rec
{
    u32 c[4];
    u32 s;
    __device__ __forceinline__ void load(const uint8_t* rec, int lane)
    {
        const uint4 v = *reinterpret_cast<const uint4*>(rec + lane * 16);
        c[0] = v.x; c[1] = v.y; c[2] = v.z; c[3] = v.w;
        s = *reinterpret_cast<const u32*>(rec + SOFF + (lane >> 2) * 4) >> (8 * (lane & 1));
    }
};
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((u32)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// One k32 step of a warp's tile row against NT token tiles from the smem stage: A words from record R, B 16-B words at bb + (t 8 + g) XREC + c 16, scale
// words sb[t] (byte BB). acc[t] += A(R) x B(t).
template <int H, int BB, int NT>
__device__ __forceinline__ void mma_row(const Rec& R, const uint8_t* bb, int g, int c, const u32 (&sb)[NT], int nt, float (&acc)[NT][4])
{
    const u32 a0 = x4lo(R.c[2 * H]), a1 = x4hi(R.c[2 * H]), a2 = x4lo(R.c[2 * H + 1]), a3 = x4hi(R.c[2 * H + 1]);
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        if (t < nt)
        {
            const uint4 bv = *reinterpret_cast<const uint4*>(bb + (t * 8 + g) * XREC + c * 16);
            mma_mx<2 * H, BB>(acc[t], a0, a1, a2, a3, H ? bv.z : bv.x, H ? bv.w : bv.y, R.s, sb[t]);
        }
}

// ---------------------------------------------------------------------------------------------------- quantizers
// E8M0 code of a 32-block with absolute max amax (floored at 1e-4): e = ceil(log2(amax / 448)), code = e + 127; inv = 2^-e.
__device__ __forceinline__ u32 mx_scale(float amax, float& inv)
{
    amax = fmaxf(amax, 1e-4f);
    const u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f));
    const int e = static_cast<int>(u >> 23) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    inv = __uint_as_float(static_cast<u32>(127 - e) << 23);
    return static_cast<u32>(e + 127);
}
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d)   // four floats -> four E4M3 bytes (a = low), RN, saturating
{
    u32 lo, hi;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(lo) : "f"(a), "f"(b));
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(hi) : "f"(c), "f"(d));
    return lo | (hi << 16);
}
// 32 values v[0..31] of one row's 32-block j (physical k = 32 j + i) -> the 8 words of the record layout at rec = row's record (j / 2), h = j & 1:
// for lane c: word (c, h, b0) = codes of k 2c, 2c+1, 2c+8, 2c+9; word (c, h, b1) = codes of k 16 + those. dst = the row's record base.
__device__ __forceinline__ void store_block(uint8_t* rec, int h, const float (&v)[32], float inv)
{
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        u32* p = reinterpret_cast<u32*>(rec + c * 16 + h * 8);
        p[0] = e4m3x4(v[2 * c] * inv, v[2 * c + 1] * inv, v[2 * c + 8] * inv, v[2 * c + 9] * inv);
        p[1] = e4m3x4(v[16 + 2 * c] * inv, v[17 + 2 * c] * inv, v[24 + 2 * c] * inv, v[25 + 2 * c] * inv);
    }
}
// x [M][K] bf16 -> xq [M][K/64][XREC], xs [M][K/32]. One thread per (row, 32-block); grid = ceil(M * K / 32 / 256), block 256. K % 64 == 0.
__global__ void __launch_bounds__(256) k_xq(const __nv_bfloat16* __restrict__ x, int M, int K, uint8_t* __restrict__ xq, uint8_t* __restrict__ xs)
{
    const int kb = K >> 5;
    const long i = static_cast<long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long>(M) * kb) return;
    const int row = static_cast<int>(i / kb), j = static_cast<int>(i % kb);
    const uint4* p = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * K + j * 32);
    float v[32];
    float amax = 0.f;
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        const uint4 w = p[q];
        const u32 ww[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ww[r]));
            v[q * 8 + 2 * r] = f.x; v[q * 8 + 2 * r + 1] = f.y;
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    float inv;
    const u32 code = mx_scale(amax, inv);
    store_block(xq + (static_cast<size_t>(row) * (K / 64) + (j >> 1)) * XREC, j & 1, v, inv);
    xs[static_cast<size_t>(row) * kb + j] = static_cast<uint8_t>(code);
}
// act [rows][I] fp16 (rows = tiles * 8, cut to the plan's 8 * n_tiles) -> aq [rows][I/64][XREC], as [rows][ASP] (bytes I/32.. of a row are not written).
// grid = ceil(rows * I / 32 / 256).
__global__ void __launch_bounds__(256) k_aq(const __half* __restrict__ act, int rows, const int* __restrict__ n_tiles_dev, int I, uint8_t* __restrict__ aq,
                                            uint8_t* __restrict__ as)
{
    const int kb = I >> 5;
    const long i = static_cast<long>(blockIdx.x) * 256 + threadIdx.x;
    if (n_tiles_dev) rows = min(rows, *n_tiles_dev * 8);
    if (i >= static_cast<long>(rows) * kb) return;
    const int row = static_cast<int>(i / kb), j = static_cast<int>(i % kb);
    const uint4* p = reinterpret_cast<const uint4*>(act + static_cast<size_t>(row) * I + j * 32);
    float v[32];
    float amax = 0.f;
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        const uint4 w = p[q];
        const u32 ww[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            const float2 f = __half22float2(as_h2(ww[r]));
            v[q * 8 + 2 * r] = f.x; v[q * 8 + 2 * r + 1] = f.y;
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    float inv;
    const u32 code = mx_scale(amax, inv);
    store_block(aq + (static_cast<size_t>(row) * (I / 64) + (j >> 1)) * XREC, j & 1, v, inv);
    as[static_cast<size_t>(row) * ASP + j] = static_cast<uint8_t>(code);
}

// ---------------------------------------------------------------------------------------------------- fc1
// CTA = (slot, CW chunks, gy), warp w = chunk cg * CW + w. grid = n_slots * (I / 16 / CW) * GY (gy fastest, so the GY CTAs that share an expert's weights
// are neighbours in launch order and read them from DRAM about once: an expert's second token group read its weights again from DRAM after the L2 had
// dropped them when every CTA walked all the groups), block CW * 32, dynamic smem fc1_smem<CW, NTG, ST>(). CTA gy walks the expert's token groups
// gy, gy + GY, ...
// A stage = one 64-k record: the CW chunks' gate + up records (2 GR contiguous bytes per chunk) and the group's NTOK = NTG * 8 rows of xq
// ([NTOK][XREC], gathered by tile_tok), by cp.async in an ST-deep ring with per-thread pointers fixed per group (one add per copy per stage), so the
// bytes in flight per SM are (ST - 1) stages whatever the warp count. The rows' E8M0 bytes ([NTOK][K1 / 32]) are staged once per group; the mma's
// byte selector picks the record's two k32 bytes from a 16-bit word. fp32 accumulate over all of K in one warp, swiglu_lim, fp16 act as k_fc1_ro.
template <int K1, int CW, int NTG, int ST> constexpr size_t fc1_smem() { return static_cast<size_t>(ST) * (CW * 2 * GR + NTG * 8 * XREC) + NTG * 8 * (K1 / 32) + NTG * 8 * 4; }
template <int K1, int I, int CW, int NTG, int ST>
__global__ void __launch_bounds__(CW * 32) k_fc1_tc(Args A)
{
    constexpr int NCH = I / 16, NG = K1 / KG, UPS = NCH / CW, NTOK = NTG * 8, NTH = CW * 32;
    constexpr int AREC = 2 * GR, AST = CW * AREC, STAGE = AST + NTOK * XREC, ACH = AREC / 16;
    constexpr int NA = (CW * ACH + NTH - 1) / NTH, NB = (NTOK * 4 + NTH - 1) / NTH;   // copies per thread per stage
    static_assert(K1 % KG == 0 && NCH % CW == 0 && ST >= 2 && ST - 1 <= NG && AREC % 16 == 0, "fc1 tc tiling");
    constexpr int XSR = K1 / 32, XS_B = NTOK * XSR;                          // the group's scale rows
    static_assert(XSR % 16 == 0, "fc1 tc: xs rows copy in 16-B pieces");
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* sxs = sm + ST * STAGE;
    int* stok = reinterpret_cast<int*>(sxs + XS_B);                          // the group's row -> token
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, c = lane & 3, tid = threadIdx.x;
    const int GY = A.gy1, gy = blockIdx.x % GY, slot = blockIdx.x / (UPS * GY), chunk0 = ((blockIdx.x / GY) % UPS) * CW;
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc1 + dsr::ro_fc1_rec(A.slot_expert[slot], chunk0, 0, 0, NCH, NG) * GR;   // chunk chunk0 + i, record ks: s + i NG AREC + ks AREC
    // the thread's weight copies: chunk cc, 16-B piece q of the record (fixed), source advances AREC per stage
    const uint8_t* srca[NA];
    u32 dsta[NA];
    #pragma unroll
    for (int k = 0; k < NA; ++k)
    {
        const int i = tid + k * NTH, cc = i / ACH, q = i % ACH;
        srca[k] = s + static_cast<size_t>(cc) * NG * AREC + q * 16;
        dsta[k] = cc * AREC + q * 16;
    }
    #pragma unroll 1
    for (int grp = gy; grp < ng; grp += GY)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, grp), t0 = dsr::ro_tile(tile0, grp, NTG, 0), nrow = ntg * 8;
        for (int i = tid; i < NTOK; i += NTH)
        {
            const int tok = i < nrow ? A.tile_tok[(t0 + i / 8) * 8 + (i & 7)] : 0;
            stok[i] = tok < 0 ? 0 : tok;
        }
        __syncthreads();
        for (int i = tid; i < nrow * (XSR / 16); i += NTH)                   // the rows' scale bytes, one cp.async group ahead of the ring
            cp16(sxs + i * 16, A.xs + static_cast<size_t>(stok[i / (XSR / 16)]) * XSR + (i % (XSR / 16)) * 16);
        cp_commit();
        // the thread's activation copies: row q, piece cc (fixed per group), source advances XREC per stage; rows past nrow copy token 0 (never read)
        const uint8_t* srcb[NB];
        u32 dstb[NB];
        #pragma unroll
        for (int k = 0; k < NB; ++k)
        {
            const int i = (tid + k * NTH) % (NTOK * 4), q = i >> 2, cc = i & 3;
            srcb[k] = A.xq + static_cast<size_t>(stok[q]) * (K1 / 64) * XREC + cc * 16;
            dstb[k] = AST + q * XREC + cc * 16;
        }
        auto issue = [&](int st, int ks)
        {
            uint8_t* b = sm + st * STAGE;
            #pragma unroll
            for (int k = 0; k < NA; ++k)
                if (tid + k * NTH < CW * ACH) cp16(b + dsta[k], srca[k] + static_cast<size_t>(ks) * AREC);
            #pragma unroll
            for (int k = 0; k < NB; ++k)
                if (tid + k * NTH < NTOK * 4) cp16(b + dstb[k], srcb[k] + ks * XREC);
        };
        float ag[NTG][4], au[NTG][4];
        #pragma unroll
        for (int t = 0; t < NTG; ++t) { ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.f; au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f; }
        #pragma unroll
        for (int st = 0; st < ST - 1; ++st) { issue(st, st); cp_commit(); }
        #pragma unroll 1
        for (int ks = 0; ks < NG; ++ks)
        {
            cp_wait<ST - 2>();
            __syncthreads();
            if (ks + ST - 1 < NG) issue((ks + ST - 1) % ST, ks + ST - 1);
            cp_commit();
            u32 sb[NTG];
            #pragma unroll
            for (int t = 0; t < NTG; ++t) sb[t] = *reinterpret_cast<const unsigned short*>(sxs + (t * 8 + g) * XSR + 2 * ks);
            const uint8_t* b = sm + (ks % ST) * STAGE;
            const uint8_t* bb = b + AST;
            Rec wg, wu;
            wg.load(b + warp * AREC, lane);
            wu.load(b + warp * AREC + GR, lane);
            mma_row<0, 0, NTG>(wg, bb, g, c, sb, ntg, ag);
            mma_row<0, 0, NTG>(wu, bb, g, c, sb, ntg, au);
            mma_row<1, 1, NTG>(wg, bb, g, c, sb, ntg, ag);
            mma_row<1, 1, NTG>(wu, bb, g, c, sb, ntg, au);
        }
        cp_wait<0>();
        #pragma unroll
        for (int t = 0; t < NTG; ++t)
            if (t < ntg)
            {
                __half* a0 = A.act + (static_cast<size_t>(t0 + t) * 8 + 2 * c) * I + (chunk0 + warp) * 16 + g;
                a0[0] = __float2half_rn(mx4::swiglu_lim(ag[t][0], au[t][0], A.limit));
                a0[I] = __float2half_rn(mx4::swiglu_lim(ag[t][1], au[t][1], A.limit));
                a0[8] = __float2half_rn(mx4::swiglu_lim(ag[t][2], au[t][2], A.limit));
                a0[I + 8] = __float2half_rn(mx4::swiglu_lim(ag[t][3], au[t][3], A.limit));
            }
        __syncthreads();                                                    // the ring and stok are rewritten by the next group
    }
}

// ---------------------------------------------------------------------------------------------------- fc2
// CTA = (slot, TPC = WARPS * WT output tiles, gy), warp w = tiles ot0 + w WT .. + WT - 1. grid = n_slots * (NOUT / 16 / TPC) * GY (gy fastest, as k_fc1_tc),
// block WARPS * 32, dynamic smem fc2_smem<I, WARPS, WT, NTG, ST>(): the group's rows of aq [NTOK][I] and as [NTOK][ASP] (staged once per group) and an
// ST-deep ring of A stages, a stage = one record of each of the CTA's TPC tiles (cp.async, as k_fc1_tc). The record loop is unrolled (NG = 9), so the
// ring slot and the scale byte are constants.
template <int I, int WARPS, int WT, int NTG, int ST> constexpr size_t fc2_smem() { return static_cast<size_t>(NTG) * 8 * (I + ASP) + static_cast<size_t>(ST) * WARPS * WT * GR; }
template <int I, int NOUT, int WARPS, int WT, int NTG, int ST>
__global__ void __launch_bounds__(WARPS * 32) k_fc2_tc(Args A)
{
    constexpr int NG = I / KG, OT = NOUT / 16, TPC = WARPS * WT, UPS = OT / TPC, NTOK = NTG * 8, SA_B = NTOK * I, BREG = NTOK * (I + ASP), AST = TPC * GR, NTH = WARPS * 32;
    constexpr int ACH = GR / 16, NA = (TPC * ACH + NTH - 1) / NTH;
    static_assert(I % KG == 0 && OT % TPC == 0 && I % 16 == 0 && ST >= 2 && ST - 1 <= NG && NG % 2 == 1, "fc2 tc tiling");
    extern __shared__ __align__(16) uint8_t sm[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, c = lane & 3, tid = threadIdx.x;
    const int GY = A.gy2, gy = blockIdx.x % GY, slot = blockIdx.x / (UPS * GY), ot0 = ((blockIdx.x / GY) % UPS) * TPC;
    if (A.n_slots_dev && slot >= *A.n_slots_dev) return;
    const int nt = A.slot_nt[slot], tile0 = A.slot_tile0[slot], ng = dsr::ro_ngroups(nt, NTG);
    const uint8_t* s = A.fc2 + dsr::ro_fc2_rec(A.slot_expert[slot], ot0, 0, OT, NG) * GR;   // tile ot0 + j record r at s + (j NG + r) GR
    const uint8_t* srca[NA];
    u32 dsta[NA];
    #pragma unroll
    for (int k = 0; k < NA; ++k)
    {
        const int i = tid + k * NTH, j = i / ACH, q = i % ACH;
        srca[k] = s + static_cast<size_t>(j) * NG * GR + q * 16;
        dsta[k] = BREG + j * GR + q * 16;
    }
    auto issue = [&](int st, int r)
    {
        #pragma unroll
        for (int k = 0; k < NA; ++k)
            if (tid + k * NTH < TPC * ACH) cp16(sm + st * AST + dsta[k], srca[k] + r * GR);
    };
    #pragma unroll 1
    for (int grp = gy; grp < ng; grp += GY)
    {
        const int ntg = dsr::ro_group_nt(nt, NTG, grp), t0 = dsr::ro_tile(tile0, grp, NTG, 0), nrow = ntg * 8;
        const size_t row0 = static_cast<size_t>(t0) * 8;
        for (int i = tid; i < nrow * (I / 16); i += NTH) cp16(sm + i * 16, A.aq + row0 * I + static_cast<size_t>(i) * 16);
        for (int i = tid; i < nrow * (ASP / 16); i += NTH) cp16(sm + SA_B + i * 16, A.as + row0 * ASP + static_cast<size_t>(i) * 16);
        cp_commit();
        #pragma unroll
        for (int st = 0; st < ST - 1; ++st) { issue(st, st); cp_commit(); }
        float acc[WT][NTG][4];
        #pragma unroll
        for (int j = 0; j < WT; ++j)
            #pragma unroll
            for (int t = 0; t < NTG; ++t) acc[j][t][0] = acc[j][t][1] = acc[j][t][2] = acc[j][t][3] = 0.f;
        u32 sb[NTG];
        #pragma unroll
        for (int r = 0; r < NG; ++r)
        {
            cp_wait<ST - 2>();
            __syncthreads();
            if (r + ST - 1 < NG) issue((r + ST - 1) % ST, r + ST - 1);
            cp_commit();
            if ((r & 1) == 0)                                                // the scale word of records r, r + 1: bytes 2 (r & 1) + h
                #pragma unroll
                for (int t = 0; t < NTG; ++t) sb[t] = *reinterpret_cast<const u32*>(sm + SA_B + (t * 8 + g) * ASP + (r >> 1) * 4);
            const uint8_t* wa = sm + BREG + (r % ST) * AST + warp * WT * GR;
            const uint8_t* bb = sm + r * XREC;                                // row q's record r at bb + q I: mma_row's XREC pitch is I here
            #pragma unroll
            for (int j = 0; j < WT; ++j)
            {
                Rec R;
                R.load(wa + j * GR, lane);
                const u32 a0 = x4lo(R.c[0]), a1 = x4hi(R.c[0]), a2 = x4lo(R.c[1]), a3 = x4hi(R.c[1]);
                const u32 b4 = x4lo(R.c[2]), b5 = x4hi(R.c[2]), b6 = x4lo(R.c[3]), b7 = x4hi(R.c[3]);
                #pragma unroll
                for (int t = 0; t < NTG; ++t)
                    if (t < ntg)
                    {
                        const uint4 bv = *reinterpret_cast<const uint4*>(bb + (t * 8 + g) * I + c * 16);
                        if ((r & 1) == 0)
                        {
                            mma_mx<0, 0>(acc[j][t], a0, a1, a2, a3, bv.x, bv.y, R.s, sb[t]);
                            mma_mx<2, 1>(acc[j][t], b4, b5, b6, b7, bv.z, bv.w, R.s, sb[t]);
                        }
                        else
                        {
                            mma_mx<0, 2>(acc[j][t], a0, a1, a2, a3, bv.x, bv.y, R.s, sb[t]);
                            mma_mx<2, 3>(acc[j][t], b4, b5, b6, b7, bv.z, bv.w, R.s, sb[t]);
                        }
                    }
            }
        }
        cp_wait<0>();
        #pragma unroll
        for (int j = 0; j < WT; ++j)
        {
            const int o = (ot0 + warp * WT + j) * 16 + g;
            #pragma unroll
            for (int t = 0; t < NTG; ++t)
                if (t < ntg)
                {
                    const int mask = A.tile_mask[t0 + t];
                    const float* w = A.tile_w + (t0 + t) * 8;
                    __half* y = A.ypart + static_cast<size_t>(t0 + t) * 8 * NOUT;
                    if ((mask >> (2 * c)) & 1)
                    {
                        y[(2 * c) * NOUT + o] = __float2half_rn(w[2 * c] * acc[j][t][0]);
                        y[(2 * c) * NOUT + o + 8] = __float2half_rn(w[2 * c] * acc[j][t][2]);
                    }
                    if ((mask >> (2 * c + 1)) & 1)
                    {
                        y[(2 * c + 1) * NOUT + o] = __float2half_rn(w[2 * c + 1] * acc[j][t][1]);
                        y[(2 * c + 1) * NOUT + o + 8] = __float2half_rn(w[2 * c + 1] * acc[j][t][3]);
                    }
                }
        }
        __syncthreads();                                                    // sm is restaged by the next group
    }
}

// ---------------------------------------------------------------------------------------------------- host launch
// The four launches of one layer's routed experts on st: k_xq (x -> xq / xs), k_fc1_tc, k_aq (act -> aq / as), k_fc2_tc. sb = the slot bound the
// grids are sized for (CTAs of slots >= *n_slots_dev exit), tiles_bound = rows of act / aq to quantize (the plan's tile bound x 8).
// cudaFuncSetAttribute is a host call without stream work (the first call per instantiation).
template <int K1, int I, int NOUT, int CW, int NTG1, int ST1, int W2, int WT, int NTG2, int ST2>
inline cudaError_t launch_layer(const Args& A, const __nv_bfloat16* x, int M, int sb, int tiles_bound, uint8_t* xq, uint8_t* xs, cudaStream_t st)
{
    constexpr size_t S1 = fc1_smem<K1, CW, NTG1, ST1>(), S2 = fc2_smem<I, W2, WT, NTG2, ST2>();
    auto* k1 = k_fc1_tc<K1, I, CW, NTG1, ST1>;
    auto* k2 = k_fc2_tc<I, NOUT, W2, WT, NTG2, ST2>;
    static const cudaError_t attr = [&] {
        cudaError_t e = cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S1));
        const cudaError_t e2 = cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S2));
        return e != cudaSuccess ? e : e2;
    }();
    if (attr != cudaSuccess) return attr;
    k_xq<<<(unsigned)((static_cast<long>(M) * (K1 / 32) + 255) / 256), 256, 0, st>>>(x, M, K1, xq, xs);
    k1<<<sb * (I / 16 / CW) * A.gy1, CW * 32, S1, st>>>(A);
    const long arows = static_cast<long>(tiles_bound) * 8;
    k_aq<<<(unsigned)((arows * (I / 32) + 255) / 256), 256, 0, st>>>(A.act, static_cast<int>(arows), A.n_tiles_dev, I, A.aq, A.as);
    k2<<<sb * (NOUT / 16 / (W2 * WT)) * A.gy2, W2 * 32, S2, st>>>(A);
    return cudaGetLastError();
}
}   // namespace mxtc
