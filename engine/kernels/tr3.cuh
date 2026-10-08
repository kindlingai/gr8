// TR3 (EXL3 trellis) decode-path expert kernels. Our own code, written from the format:
//   tile = 16x16 weights, tail-biting bit stream of 256*K bits, MSB-first over LE uint32 words;
//   weight t = lane*8+j has the 16-bit state ending at stream bit (t+1)*K; value = 3INST "mcg" hash.
//   lane l of a warp decodes 8 weights = the mma A fragment (rows = output cols l/4 and l/4+8) of the tile.
//   linear: y = H128(f16(H128(x*suh)) @ What) * svh, H128 scaled 1/sqrt(128), per 128 block.
#pragma once
#include "k1args.cuh"
#include "tr3_wide.cuh"

// two 16-bit states -> half2. E4 (exlr8-e4m3 packs): each value rounded to FP8 E4M3 and back (RN-even, saturating to
// finite), as the encoder rounded its codebook and as the prefill path rounds every tile (bind_prefill.cuh tr3_b8).
template <bool E4 = false>
__device__ __forceinline__ u32 tr3_cb2(u32 w0, u32 w1)
{
    u32 x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
    x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
    const u32 lo = __byte_perm(x0, x1, 0x5410);
    const u32 hi = __byte_perm(x0, x1, 0x7632);
    u32 r = as_u32(__hadd2(as_h2(lo), as_h2(hi)));
    if constexpr (E4)
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

template <int BITS, bool E4 = false>
struct Tr3Lane
{
    int i0, i1, sh;   // word indices (bytes offsets /4) and base shift
    __device__ __forceinline__ void init(int lane)
    {
        const int b0 = (lane * 8 + 1) * BITS - 16 + 256 * BITS;   // first bit of the lane's first state
        i0 = (b0 / 32) % (8 * BITS);
        i1 = (i0 + 1) % (8 * BITS);
        sh = 48 - (b0 % 32);                                      // state j = (m >> (sh - BITS*j)) & 0xffff
    }
    __device__ __forceinline__ void load(const uint8_t* tile, u32& a, u32& b) const
    {
        a = ld_nc(tile + i0 * 4);
        b = ld_nc(tile + i1 * 4);
    }
    // One coalesced load per tile (lane w reads word w), words handed to their lanes by two shuffles at
    // decode time: half the load instructions of load(), and no word is fetched twice.
    __device__ __forceinline__ u32 loadw(const uint8_t* tile, int lane) const { return ld_nc(tile + (lane % (8 * BITS)) * 4); }
    __device__ __forceinline__ void decodew(u32 raw, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        const u32 a = __shfl_sync(0xffffffffu, raw, i0), b = __shfl_sync(0xffffffffu, raw, i1);
        decode(a, b, f0, f1);
    }
    // -> fragments for cols 0-7 (f0) and 8-15 (f1) of the tile
    __device__ __forceinline__ void decode(u32 a, u32 b, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        decode_m((static_cast<u64>(a) << 32) | b, sh, f0, f1);
    }
    // m = the lane's stream bits, state j = (m >> (s - BITS*j)) & 0xffff
    __device__ static __forceinline__ void decode_m(u64 m, int s, u32 (&f0)[2], u32 (&f1)[2])
    {
        u32 w[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (s - BITS * j)) & 0xffffu;
        f0[0] = tr3_cb2<E4>(w[0], w[1]);
        f0[1] = tr3_cb2<E4>(w[2], w[3]);
        f1[0] = tr3_cb2<E4>(w[4], w[5]);
        f1[1] = tr3_cb2<E4>(w[6], w[7]);
    }
};

// Width and byte offset in tr3.fc1 of fc1 unit u = 2b + mat (block b, gate | up) of expert e, table row tr.
// kmix = tr3.kmix (exlr8 v2 packs, #65) or nullptr. kmix[e] == 0 (and no kmix) = every unit has the table width tr[4]
// and the matrices sit back to back at that width (TR3, exlr8 v1); else 4 bits per unit at bit 4u give its width,
// and the matrices sit back to back at theirs.
__device__ __forceinline__ int tr3_unit(const long long* kmix, int e, const long long* tr, int u, size_t& off)
{
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;      // bytes of one fc1 matrix per bit of width
    const long long km = kmix ? kmix[e] : 0;
    if (!km)
    {
        const int k = static_cast<int>(tr[4]);
        off = static_cast<size_t>(tr[0]) + static_cast<size_t>(u) * k * MB;
        return k;
    }
    off = static_cast<size_t>(tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * MB;
    return static_cast<int>((km >> (4 * u)) & 15);
}

#ifndef TR3_UNR
#define TR3_UNR 8    // tiles loaded ahead per warp (bytes in flight per warp = TR3_UNR * tile bytes)
#endif
#ifndef TR3_EARLY
#define TR3_EARLY 1  // wide order: request a CTA's first weight bytes before its prologue (0 = after it; A/B builds)
#endif

// ---------------------------------------------------------------------------------------------
// FC1 unit = (slot, block b, matrix gate|up, k-split h). 8*WG warps: warp = (group q, tile j).
// Reads 8 tiles (128 output cols) x its k range; writes partial fp32 sums gpart[..][h*WG+q][m][128].
// tr: the slot's expert row of t_tab. mw: the first tile of the unit's matrix (block b, gate|up). BITS: its width
// (tr3_unit); units of blocks the expert does not have exit.
// The 4 hidden-side values at k of block row `row`: the stored vector, or BS (block-scale form, exlr8 v2, #65): the
// layer sign times the row's scale of k's 128-block. v2 vectors are exactly sign x block scale, so both give the same
// bits.
template <bool BS>
__device__ __forceinline__ uint2 tr3_hid4(const __half* vec, const __half* sg, const __half* blk, size_t row, int k)
{
    if (!BS) return *reinterpret_cast<const uint2*>(vec + row * HID + k);
    const uint2 s = *reinterpret_cast<const uint2*>(sg + k);
    const __half2 c = __half2half2(blk[row * (HID / 128) + k / 128]);
    uint2 r;
    r.x = as_u32(__hmul2(as_h2(s.x), c));
    r.y = as_u32(__hmul2(as_h2(s.y), c));
    return r;
}

// mask = the slot's tokens (A.slot_mask, or A.sh_mask for the trellis shared expert)
template <int BITS, bool E4 = false, bool BS = false>
__device__ __forceinline__ void tr3_fc1_body(const K1Args& A, int plan, int unit, const long long* tr, const uint8_t* mw, int mask, u32* smem)
{
    constexpr int TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int h = unit % A.KS1;
    const int mat = (unit / A.KS1) & 1;
    const int b = (unit / (A.KS1 * 2)) % A.B;
    const int slot = unit / (A.KS1 * 2 * A.B);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int KSU = KSH / A.KS1;              // k16 steps of this unit
    const int ks0 = h * KSU;
    const int q = warp >> 3, j = warp & 7;
    const int steps = KSU / A.WG;
    const int ksq = q * steps;                // first step of this warp group inside the unit
    const uint8_t* base = mw + (static_cast<size_t>(ks0 + ksq) * 8 + j) * TB;
#if !defined(TR3_PROBE)
    // wide order: the first group is requested here, before the prologue, so its DRAM latency overlaps the prologue's
    const bool wide = trw_ok<BITS>() && A.t_wide;                 // K3, and K2 (#65)
    const uint8_t* g0 = base - static_cast<size_t>(j) * TB + j * 256 + lane * 8;
    u64 nw[3] = {0, 0, 0};
    if (wide && TR3_EARLY)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(g0 + n * 2048);
    }
#endif

    // prologue: xa = f16(H128(x * suh)) for the unit's k range, as token fragments [ks][lane][2].
    // Tasks = (active token, 128 block), spread over all warps, two per iteration so the cold suh loads overlap.
    // Rows of inactive tokens stay unwritten: an mma output column depends only on its own token column.
    {
        const __half* suh = A.t_v1 + (row * 2 + mat) * HID;
        const int nact = __popc(mask & 0xff), ntask = nact * (KSU / 8);
        for (int task = warp; task < ntask; task += 2 * nwarps)
        {
            uint2 xr[2], sr[2];
            int m[2], blk[2];
            #pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                const int tk = min(task + u * nwarps, ntask - 1);
                m[u] = __fns(mask, 0, tk % nact + 1);
                blk[u] = tk / nact;
                const int k = ks0 * 16 + blk[u] * 128 + lane * 4;
                xr[u] = *reinterpret_cast<const uint2*>(A.x + m[u] * HID + k);
                if (BS) sr[u] = tr3_hid4<true>(nullptr, A.t_sg, A.t_v1b, row * 2 + mat, k);
                else sr[u] = *reinterpret_cast<const uint2*>(suh + k);
            }
            #pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                if (task + u * nwarps >= ntask) break;
                const float2 x01 = __half22float2(as_h2(xr[u].x)), x23 = __half22float2(as_h2(xr[u].y));
                const float2 s01 = __half22float2(as_h2(sr[u].x)), s23 = __half22float2(as_h2(sr[u].y));
                float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
                had128(v, lane);
                store_frag_row(smem, blk[u] * 8, m[u], lane, v);
            }
        }
    }
    __syncthreads();

    const u32* xa = smem + (ksq * 32 + lane) * 2;
    Tr3Lane<BITS, E4> ld;
    ld.init(lane);

    // One mma per tile: the tile is the A operand (16 output cols x 16 k, exactly the trellis lane order),
    // the tokens are the B operand (8 cols).  acc: c0,c1 = (col l/4, tokens 2c, 2c+1), c2,c3 = (col l/4+8, same).
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#if defined(TR3_PROBE) && TR3_PROBE == 3
    // cost probe: loads only, wide. The group's 8 warps read the same bytes as the tile loads (8 k-steps = 64 tiles =
    // 2048*BITS bytes) with BITS 64-bit loads per warp (256 B per instruction) instead of 8 32-bit ones (4*8*BITS B).
    {
        const uint8_t* g0 = base - static_cast<size_t>(j) * TB;
        u64 nw[BITS];
        #pragma unroll
        for (int p = 0; p < BITS; ++p) nw[p] = ld_nc64(g0 + static_cast<size_t>(p * 8 + j) * 256 + lane * 8);
        for (int s = 0; s < steps; s += 8)
        {
            u32 r = 0;
            #pragma unroll
            for (int p = 0; p < BITS; ++p) r ^= static_cast<u32>(nw[p]) ^ static_cast<u32>(nw[p] >> 32);
            if (s + 8 < steps)
            {
                #pragma unroll
                for (int p = 0; p < BITS; ++p) nw[p] = ld_nc64(g0 + static_cast<size_t>(s + 8) * 8 * TB + static_cast<size_t>(p * 8 + j) * 256 + lane * 8);
            }
            #pragma unroll
            for (int p = 0; p < 8; ++p) acc[0] += __int_as_float((r ^ xa[(s + p) * 64] ^ xa[(s + p) * 64 + 1]) & 0x3fffffff);
        }
    }
#else
#if !defined(TR3_PROBE)
    if (wide)
    {
        // wide order (tr3_wide.cuh): the warp's 8 tiles of 8 k-steps in three 64-bit loads; the lane's 24 bits per
        // tile come out of its own words, the 13 bits before them from lane - 1. Same states, same mma order as below.
        if (!TR3_EARLY)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(g0 + n * 2048);
        }
        for (int s = 0; s < steps; s += 8)
        {
            const u64 w[3] = {nw[0], nw[1], nw[2]};
            if (s + 8 < steps)
            {
                #pragma unroll
                for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(g0 + static_cast<size_t>(s + 8) * 8 * TB + n * 2048);
            }
            #pragma unroll
            for (int p = 0; p < 8; ++p)
            {
                const u32 c = trw_chunk<BITS>(w, p);
                const u32 pc = __shfl_sync(0xffffffffu, c, (lane + 31) & 31);
                u32 f0[2], f1[2];
                Tr3Lane<BITS, E4>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | c, 7 * BITS, f0, f1);
                mma16816(acc, f0[0], f1[0], f0[1], f1[1], xa[(s + p) * 64], xa[(s + p) * 64 + 1]);
            }
        }
    }
    else
    {
#endif
    u32 na[TR3_UNR];
    #pragma unroll
    for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(base + static_cast<size_t>(p) * 8 * TB, lane);
    for (int s = 0; s < steps; s += TR3_UNR)
    {
        u32 wa[TR3_UNR];
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p) wa[p] = na[p];
        if (s + TR3_UNR < steps)
        {
            #pragma unroll
            for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(base + static_cast<size_t>(s + TR3_UNR + p) * 8 * TB, lane);
        }
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p)
        {
            u32 f0[2], f1[2];
#if !defined(TR3_PROBE) || TR3_PROBE == 1
            ld.decodew(wa[p], f0, f1);
#endif
            const u32 x0 = xa[(s + p) * 64], x1 = xa[(s + p) * 64 + 1];
#if !defined(TR3_PROBE)
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], x0, x1);
#elif TR3_PROBE == 1   // cost probe: decode, no mma
            acc[0] += __int_as_float((f0[0] ^ f1[0] ^ f0[1] ^ f1[1] ^ x0 ^ x1) & 0x3fffffff);
#else                  // cost probe: loads only
            acc[0] += __int_as_float((wa[p] ^ x0 ^ x1) & 0x3fffffff);
#endif
        }
    }
#if !defined(TR3_PROBE)
    }
#endif
#endif

    // epilogue: cols j*16 + l/4 (+8), tokens 2c and 2c+1
    const int nl = lane >> 2, c = lane & 3;
    float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX + h * A.WG + q) * 8 * 128 + j * 16 + nl;
    if ((mask >> (2 * c)) & 1) { gp[2 * c * 128] = acc[0]; gp[2 * c * 128 + 8] = acc[2]; }
    if ((mask >> (2 * c + 1)) & 1) { gp[(2 * c + 1) * 128] = acc[1]; gp[(2 * c + 1) * 128 + 8] = acc[3]; }
}

// ---------------------------------------------------------------------------------------------
// FC2 unit = (slot, block b, n-split). Prologue: g,u partial sums -> H*svh -> silu(g)*u -> H(act*suh_d)
// for the block's 128 intermediate columns; then streams the block's down tiles for its n range.
// Writes z[slot][b][m][n] fp16 (before the output Hadamard; finalize applies H*svh per block).
template <int BITS, bool E4 = false>
__device__ __forceinline__ void tr3_fc2_body(const K1Args& A, int plan, int unit, const long long* tr, int mask, u32* smem)
{
    constexpr int TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int ns = unit % A.NS2;
    const int b = (unit / A.NS2) % A.B;
    const int slot = unit / (A.NS2 * A.B);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int GP = A.KS1 * A.WG;
    const int NTU = KSH / A.NS2;                      // n-tiles of this unit
    const int nt0 = ns * NTU + warp;
    const int ntiles = (NTU - warp + nwarps - 1) / nwarps;
    const uint8_t* base = A.t_fc2 + tr[1] + static_cast<size_t>(b) * (static_cast<size_t>(KSH) * 8 * TB);
    // wide order (tr3_wide.cuh, k_tr3_widen2): the 8 k-tiles of an n-tile in three 64-bit loads. The first n-tile is
    // requested here, before the prologue, so its DRAM latency overlaps the prologue's.
    const bool wide = trw_ok<BITS>() && A.t_wide2;                // K3, and K2 (#65)
    const uint8_t* wg = base + static_cast<size_t>(nt0) * 8 * TB + lane * 8;
    u64 nw[3] = {0, 0, 0};
    if (wide && ntiles > 0 && TR3_EARLY)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + n * 256);
    }

    {
        const __half* v2 = A.t_v2 + row * 3 * 128 + lane * 4;
        for (int m = warp; m < 8; m += nwarps)
        {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if ((mask >> m) & 1)
            {
                float gu[2][4];
                #pragma unroll
                for (int mat = 0; mat < 2; ++mat)
                {
                    const float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX * 8) * 128 + m * 128 + lane * 4;
                    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int p = 0; p < GP; ++p)
                    {
                        const float4 w = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p) * 8 * 128));
                        s.x += w.x; s.y += w.y; s.z += w.z; s.w += w.w;
                    }
                    float y[4] = {rnd_h(s.x), rnd_h(s.y), rnd_h(s.z), rnd_h(s.w)};
                    had128(y, lane);
                    const uint2 sr = *reinterpret_cast<const uint2*>(v2 + mat * 128);
                    const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
                    gu[mat][0] = rnd_h(y[0] * s01.x); gu[mat][1] = rnd_h(y[1] * s01.y);
                    gu[mat][2] = rnd_h(y[2] * s23.x); gu[mat][3] = rnd_h(y[3] * s23.y);
                }
                const uint2 sr = *reinterpret_cast<const uint2*>(v2 + 2 * 128);
                const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
                v[0] = rnd_h(silu_f(gu[0][0]) * gu[1][0]) * s01.x;
                v[1] = rnd_h(silu_f(gu[0][1]) * gu[1][1]) * s01.y;
                v[2] = rnd_h(silu_f(gu[0][2]) * gu[1][2]) * s23.x;
                v[3] = rnd_h(silu_f(gu[0][3]) * gu[1][3]) * s23.y;
                had128(v, lane);
            }
            store_frag_row(smem, 0, m, lane, v);
        }
    }
    __syncthreads();

    const u32* ha = smem + lane * 2;
    Tr3Lane<BITS, E4> ld;
    ld.init(lane);
    const int nl = lane >> 2, c = lane & 3;
    const bool wr0 = (mask >> (2 * c)) & 1, wr1 = (mask >> (2 * c + 1)) & 1;
    __half* zb = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + 2 * c) * HID + nl;   // token 2c; token 2c+1 is +HID

    auto tile_ptr = [&](int q) { return base + (static_cast<size_t>(nt0 + (q >> 3) * nwarps) * 8 + (q & 7)) * TB; };
    const int Q = ntiles * 8;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (wide)
    {
        // same states, same mma order and the same z stores as the tile loop below
        if (!TR3_EARLY && ntiles > 0)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + n * 256);
        }
        for (int i = 0; i < ntiles; ++i)
        {
            const u64 w[3] = {nw[0], nw[1], nw[2]};
            if (i + 1 < ntiles)
            {
                #pragma unroll
                for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + static_cast<size_t>(i + 1) * nwarps * 8 * TB + n * 256);
            }
            #pragma unroll
            for (int p = 0; p < 8; ++p)
            {
                const u32 ch = trw_chunk<BITS>(w, p);
                const u32 pc = __shfl_sync(0xffffffffu, ch, (lane + 31) & 31);
                u32 f0[2], f1[2];
                Tr3Lane<BITS, E4>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
                mma16816(acc, f0[0], f1[0], f0[1], f1[1], ha[p * 64], ha[p * 64 + 1]);
            }
            __half* zp = zb + (nt0 + i * nwarps) * 16;
            if (wr0) { zp[0] = __float2half_rn(acc[0]); zp[8] = __float2half_rn(acc[2]); }
            if (wr1) { zp[HID] = __float2half_rn(acc[1]); zp[HID + 8] = __float2half_rn(acc[3]); }
            #pragma unroll
            for (int k = 0; k < 4; ++k) acc[k] = 0.f;
        }
        return;
    }
    u32 na[TR3_UNR];
    if (Q > 0)
    {
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(tile_ptr(p), lane);
    }
    for (int s = 0; s < Q; s += TR3_UNR)
    {
        u32 wa[TR3_UNR];
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p) wa[p] = na[p];
        if (s + TR3_UNR < Q)
        {
            #pragma unroll
            for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(tile_ptr(s + TR3_UNR + p), lane);
        }
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p)
        {
            u32 f0[2], f1[2];
            ld.decodew(wa[p], f0, f1);
            const int ks = (s + p) & 7;
            const u32 x0 = ha[ks * 64], x1 = ha[ks * 64 + 1];
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], x0, x1);
        }
        if (((s + TR3_UNR) & 7) == 0)
        {
            __half* zp = zb + (nt0 + (s >> 3) * nwarps) * 16;
            if (wr0) { zp[0] = __float2half_rn(acc[0]); zp[8] = __float2half_rn(acc[2]); }
            if (wr1) { zp[HID] = __float2half_rn(acc[1]); zp[HID + 8] = __float2half_rn(acc[3]); }
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = 0.f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Finalize: out[m] = f16 or bf16( sum_k w[m][k] * sum_b svh_d[e][b] * H128(z[slot][b][m]) + shared[m] ),
// fixed k, b order. One CTA = (token, one 128-col block), 4 warps x 2 routes each, summed through smem.
// SH (exlr8 v2 trellis shared expert, #65): shared[m] is the trellis expert of scratch slot n_slots with weight 1. A fifth
// warp sums it (the CTA has 160 threads), and the s_* records' fp16 ypart row is not read.
template <bool SH = false>
__device__ __forceinline__ void tr3_fin_body(const K1Args& A, int plan)
{
    constexpr int NW = SH ? 5 : 4;
    constexpr bool BSF = SH;          // v2 layers: block-scale hidden-side vectors (tr3.sg, tr3.v3b)
    __shared__ float4 red[NW][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m = blockIdx.x / 48;
    const int n0 = (blockIdx.x % 48) * 128 + lane * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (warp < 4)
    {
        const int k = warp * 2;
        uint2 zr[2][g6nr::EB], sr[2][g6nr::EB];
        float w[2];
        bool on[2];
        int nb[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u)
        {
            const int slot = A.tok_slot[(plan * 8 + m) * 8 + k + u];
            on[u] = slot >= 0;
            w[u] = A.tok_w[(plan * 8 + m) * 8 + k + u];
            const int sl = on[u] ? slot : 0;
            const long long* tr = A.t_tab + 6 * A.slot_expert[plan * SMAX + sl];
            nb[u] = on[u] ? static_cast<int>(tr[3]) : 0;
            const size_t row = static_cast<size_t>(tr[2]);
            #pragma unroll
            for (int b = 0; b < g6nr::EB; ++b)
                if (b < nb[u])
                {
                    zr[u][b] = *reinterpret_cast<const uint2*>(A.z + ((static_cast<size_t>(sl) * A.B + b) * 8 + m) * HID + n0);
                    sr[u][b] = BSF ? tr3_hid4<true>(nullptr, A.t_sg + HID, A.t_v3b, row + b, n0)
                                   : *reinterpret_cast<const uint2*>(A.t_v3 + (row + b) * HID + n0);
                }
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u)
            #pragma unroll
            for (int b = 0; b < g6nr::EB; ++b)
                if (b < nb[u])
                {
                    const float2 z01 = __half22float2(as_h2(zr[u][b].x)), z23 = __half22float2(as_h2(zr[u][b].y));
                    float v[4] = {z01.x, z01.y, z23.x, z23.y};
                    had128(v, lane);
                    const float2 s01 = __half22float2(as_h2(sr[u][b].x)), s23 = __half22float2(as_h2(sr[u][b].y));
                    acc[0] += w[u] * (v[0] * s01.x); acc[1] += w[u] * (v[1] * s01.y);
                    acc[2] += w[u] * (v[2] * s23.x); acc[3] += w[u] * (v[3] * s23.y);
                }
    }
    if (SH && warp == 4)
    {
        const long long* tr = A.t_tab + 6 * A.sh_e;
        const int nb = static_cast<int>(tr[3]);
        const size_t row = static_cast<size_t>(tr[2]);
        uint2 zr[g6nr::EB], sr[g6nr::EB];
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < nb)
            {
                zr[b] = *reinterpret_cast<const uint2*>(A.z + ((static_cast<size_t>(A.n_slots) * A.B + b) * 8 + m) * HID + n0);
                sr[b] = BSF ? tr3_hid4<true>(nullptr, A.t_sg + HID, A.t_v3b, row + b, n0)
                            : *reinterpret_cast<const uint2*>(A.t_v3 + (row + b) * HID + n0);
            }
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < nb)
            {
                const float2 z01 = __half22float2(as_h2(zr[b].x)), z23 = __half22float2(as_h2(zr[b].y));
                float v[4] = {z01.x, z01.y, z23.x, z23.y};
                had128(v, lane);
                const float2 s01 = __half22float2(as_h2(sr[b].x)), s23 = __half22float2(as_h2(sr[b].y));
                acc[0] += v[0] * s01.x; acc[1] += v[1] * s01.y;
                acc[2] += v[2] * s23.x; acc[3] += v[3] * s23.y;
            }
    }
    red[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (warp) return;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    #pragma unroll
    for (int q = 0; q < NW; ++q)
    {
        const float4 v = red[q][lane];
        acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
    }
    const uint2 yr = SH ? make_uint2(0u, 0u) : *reinterpret_cast<const uint2*>(A.ypart + (static_cast<size_t>(SMAX) * 8 + m) * HID + n0);
    const float2 y01 = __half22float2(as_h2(yr.x)), y23 = __half22float2(as_h2(yr.y));
    uint2 o;
    o.x = pack_o2(acc[0] + y01.x, acc[1] + y01.y, A.out_bf16);
    o.y = pack_o2(acc[2] + y23.x, acc[3] + y23.y, A.out_bf16);
    *reinterpret_cast<uint2*>(K1_OUT(A) + m * HID + n0) = o;
}
