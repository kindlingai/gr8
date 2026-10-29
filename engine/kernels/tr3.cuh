// TR3 (EXL3 trellis) decode-path expert kernels. Our own code, written from the format:
//   tile = 16x16 weights, tail-biting bit stream of 256*K bits, MSB-first over LE uint32 words;
//   weight t = lane*8+j has the 16-bit state ending at stream bit (t+1)*K; value = the codebook's hash of it (below).
//   lane l of a warp decodes 8 weights = the mma A fragment (rows = output cols l/4 and l/4+8) of the tile.
//   linear: y = H128(f16(H128(x*suh)) @ What) * svh, H128 scaled 1/sqrt(128), per 128 block.
#pragma once
#include <type_traits>
#include "k1args.cuh"
#include "tr3_wide.cuh"

// The codebook turns trellis states into fp16 values. It is chosen per layer (bind::LayerTensors::cb, from the
// manifest's expert_format) and is a template parameter of every kernel that decodes routed experts:
//   TR3_MCG    the 16-bit state of a weight times 0xCBAC1FED, lop3 mask 0x8fff8fff xor 0x3b603b60, the two fp16
//              halves added. TR3 and every exlr8 release before the other codebooks.
//   TR3_MUL1   the 16-bit state times 0x83DCD12D, the four bytes of the product summed (s = 0..1020), then one fp16
//              fma: (1024 + s) * c1 + c0 rounded once, c1 = 0.00676727294921875, c0 = -10.3828125.
//   TR3_PMCGX  pair trellis: weights 2p and 2p + 1 share one 20-bit state, the 20 stream bits ending where weight
//              2p + 1's 16-bit state would end (each step brings 2K new bits, so bits per weight are unchanged).
//              m1 = mask(w * 0xCBAC1FED), m2 = mask(w * 0xBCA14C51) with mcg's mask and xor; weight 2p is
//              m1.lo + m2.hi and weight 2p + 1 is m1.hi + m2.lo, fp16 adds.
// E4 (exlr8-e4m3 layers) rounds every value to FP8 E4M3 and back (RN-even, saturating to finite), as the encoder
// rounded its codebook inside the search. The tile layout, the bit packing and the tail-biting rule are the same for
// all three.
enum { TR3_MCG = 0, TR3_MUL1 = 1, TR3_PMCGX = 2, TR3_NCB = 3 };

// The codebooks a build decodes, as a bit mask over TR3_*: a codebook outside it has no kernels, and the binder
// refuses a layer that uses it. All three by default; a shorter mask only shortens a development build.
#ifndef G6_TR3_CBS
#define G6_TR3_CBS 7
#endif
static_assert((G6_TR3_CBS & 1) != 0, "G6_TR3_CBS must keep mcg (bit 0)");
__host__ __device__ constexpr bool tr3_cb_built(int cb) { return cb >= 0 && cb < TR3_NCB && ((G6_TR3_CBS >> cb) & 1); }

// f(std::integral_constant<int, CB>) for the run-time codebook cb (TR3_MCG for anything not built: binders refuse
// those first, tr3_cb_built)
template <class F> inline void tr3_cb_dispatch(int cb, F&& f)
{
    if constexpr (tr3_cb_built(TR3_MUL1))
        if (cb == TR3_MUL1) { f(std::integral_constant<int, TR3_MUL1>()); return; }
    if constexpr (tr3_cb_built(TR3_PMCGX))
        if (cb == TR3_PMCGX) { f(std::integral_constant<int, TR3_PMCGX>()); return; }
    f(std::integral_constant<int, TR3_MCG>());
}

template <bool E4>
__device__ __forceinline__ u32 tr3_e4(u32 r)
{
    if constexpr (E4)
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

// two 16-bit states -> half2 (mcg). E4 (exlr8-e4m3 packs): each value rounded to FP8 E4M3 and back (RN-even,
// saturating to finite), as the encoder rounded its codebook and as the prefill path rounds every tile (bind_prefill.cuh
// tr3_b8).
template <bool E4 = false>
__device__ __forceinline__ u32 tr3_cb2(u32 w0, u32 w1)
{
    u32 x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const u32 lo = __byte_perm(x0, x1, 0x5410);
    const u32 hi = __byte_perm(x0, x1, 0x7632);
    u32 r = as_u32(__hadd2(as_h2(lo), as_h2(hi)));
    if constexpr (E4)
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

// two 16-bit states -> half2 (mul1). The byte sum is __dp4a against 0x01010101 (one native instruction; vabsdiff4 gives
// the same sums but is emulated on sm_121a). Its accumulator 0x6400 makes the low 16 bits the fp16 pattern of
// 1024 + s, exact since s <= 1020, so the fma rounds once, as the encoder's codebook does.
template <bool E4 = false>
__device__ __forceinline__ u32 tr3_mul1x2(u32 w0, u32 w1)
{
    const u32 s0 = __dp4a(w0 * 0x83DCD12Du, 0x01010101u, 0x6400u), s1 = __dp4a(w1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 v = __hfma2(as_h2(__byte_perm(s0, s1, 0x5410)), as_h2(0x1eee1eeeu), as_h2(0xc931c931u));   // c1, c0
    return tr3_e4<E4>(as_u32(v));
}

// one 20-bit pmcgx state -> half2 of its two weights (low half = weight 2p)
template <bool E4 = false>
__device__ __forceinline__ u32 tr3_pmcgx(u32 w)
{
    u32 x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    return tr3_e4<E4>(as_u32(__hadd2(as_h2(x0), as_h2(__byte_perm(x1, 0, 0x1032)))));
}

// Stream weight t of one tile at width bits (codebook CB, E4M3-rounded when E4) -> its fp16 bits in the low half (the
// high half is another value), read bit by bit from the tile's words: the per-weight form of Tr3Lane, for one-off
// decodes (bind.cuh k_shared_fp8, tests).
template <int CB, bool E4>
__device__ __forceinline__ u32 tr3_weight(const uint8_t* tile, int t, int bits)
{
    const int L = CB == TR3_PMCGX ? 20 : 16;
    const int end = (CB == TR3_PMCGX ? (t | 1) + 1 : t + 1) * bits;     // the state's last stream bit + 1
    const int nb = 256 * bits;
    u32 st = 0;
    for (int q = 0; q < L; ++q)
    {
        const int bit = ((end - L + q) % nb + nb) % nb;                  // tail-biting
        const u32 word = *reinterpret_cast<const u32*>(tile + (bit >> 5) * 4);
        st = (st << 1) | ((word >> (31 - (bit & 31))) & 1u);
    }
    if constexpr (CB == TR3_PMCGX) return tr3_pmcgx<E4>(st) >> (16 * (t & 1));
    else if constexpr (CB == TR3_MUL1) return tr3_mul1x2<E4>(st, 0u);
    else return tr3_cb2<E4>(st, 0u);
}

// BITS = width, E4 = E4M3 rounding, CB = codebook (TR3_*)
template <int BITS, bool E4 = false, int CB = TR3_MCG>
struct Tr3Lane
{
    static_assert(CB >= 0 && CB < TR3_NCB, "CB is a codebook (TR3_*)");
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
    __device__ __forceinline__ u32 loadw(const uint8_t* tile, int lane) const
    {
        if constexpr (BITS == 4) return ld_nc(tile + lane * 4);
        else return ld_nc(tile + (lane % (8 * BITS)) * 4);
    }
    __device__ __forceinline__ void decodew(u32 raw, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        if constexpr (BITS == 4)
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0);
            decode_m((static_cast<u64>(a) << 32) | raw, 28, f0, f1);
        }
        else
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0), b = __shfl_sync(0xffffffffu, raw, i1);
            decode(a, b, f0, f1);
        }
    }
    // -> fragments for cols 0-7 (f0) and 8-15 (f1) of the tile
    __device__ __forceinline__ void decode(u32 a, u32 b, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        decode_m((static_cast<u64>(a) << 32) | b, BITS == 4 ? 28 : sh, f0, f1);
    }
    // m = the lane's stream bits, state j = (m >> (s - BITS*j)) & 0xffff. pmcgx: the state of the pair of weights 2p,
    // 2p + 1 ends where weight 2p + 1's would, (m >> (s - BITS*(2p+1))) & 0xfffff. Its first state starts 4 - BITS bits
    // before weight 0's 16-bit state: every caller's m holds them (the top bit, s + 19 - BITS, is at most 63 for every
    // lane of load / decode, and 6 BITS + 19 < 16 BITS in the wide order's (chunk[l-1] << 8 BITS) | chunk[l]).
    __device__ static __forceinline__ void decode_m(u64 m, int s, u32 (&f0)[2], u32 (&f1)[2])
    {
        if constexpr (CB == TR3_PMCGX)
        {
            u32 w[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) w[p] = static_cast<u32>(m >> (s - BITS * (2 * p + 1))) & 0xfffffu;
            f0[0] = tr3_pmcgx<E4>(w[0]);
            f0[1] = tr3_pmcgx<E4>(w[1]);
            f1[0] = tr3_pmcgx<E4>(w[2]);
            f1[1] = tr3_pmcgx<E4>(w[3]);
        }
        else
        {
            u32 w[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (s - BITS * j)) & 0xffffu;
            if constexpr (CB == TR3_MUL1)
            {
                f0[0] = tr3_mul1x2<E4>(w[0], w[1]);
                f0[1] = tr3_mul1x2<E4>(w[2], w[3]);
                f1[0] = tr3_mul1x2<E4>(w[4], w[5]);
                f1[1] = tr3_mul1x2<E4>(w[6], w[7]);
            }
            else
            {
                f0[0] = tr3_cb2<E4>(w[0], w[1]);
                f0[1] = tr3_cb2<E4>(w[2], w[3]);
                f1[0] = tr3_cb2<E4>(w[4], w[5]);
                f1[1] = tr3_cb2<E4>(w[6], w[7]);
            }
        }
    }
};

// Width and byte offset in tr3.fc1 of fc1 unit u = 2b + mat (block b, gate | up) of expert e, table row tr.
// kmix = tr3.kmix (exlr8 v2 packs) or nullptr. kmix[e] == 0 (and no kmix) = every unit has the table width tr[4]
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

// EL3 (#151) stage 1, the piece table: a piece is one 128-column block of one expert row (its gate and up rows, its
// down columns). ptab[(e * EB + b) * 2] is the slab holding the piece's fc1 bytes ([gate][up] back to back, each at
// its unit's width, the order they have inside tr3.fc1) and ptab[(e * EB + b) * 2 + 1] its fc2 bytes (the block's
// down tiles); entries of blocks the row does not have on this rank are null. The table sits at one fixed device
// address per layer, so a captured graph stays valid when its entries change (bind::PieceTable, ptab.cuh). ptab ==
// nullptr: the pack layout, every piece at its tr3.table offset in one tr3.fc1 / tr3.fc2 (tr3_unit). Either way one
// table read per CTA, before the tile loops.
__device__ __forceinline__ int tr3_ptab_slot(int e, int b) { return (e * g6nr::EB + b) * 2; }

// Width and first tile (mw) of fc1 unit u = 2b + mat of expert row e (table row tr).
__device__ __forceinline__ int tr3_fc1_unit(const uint8_t* fc1, const uint8_t* const* ptab, const long long* kmix, int e,
                                            const long long* tr, int u, const uint8_t*& mw)
{
    if (!ptab)
    {
        size_t off;
        const int k = tr3_unit(kmix, e, tr, u, off);
        mw = fc1 + off;
        return k;
    }
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;
    const long long km = kmix ? kmix[e] : 0;
    const uint8_t* p = ptab[tr3_ptab_slot(e, u >> 1)];
    if (!km)
    {
        const int k = static_cast<int>(tr[4]);
        mw = p + static_cast<size_t>(u & 1) * k * MB;
        return k;
    }
    mw = p + ((u & 1) ? static_cast<size_t>((km >> (4 * (u & ~1))) & 15) * MB : 0);
    return static_cast<int>((km >> (4 * u)) & 15);
}

// First tile of fc2 block b of expert row e (table row tr) at width bits.
__device__ __forceinline__ const uint8_t* tr3_fc2_block(const uint8_t* fc2, const uint8_t* const* ptab, int e, const long long* tr, int b, int bits)
{
    if (!ptab) return fc2 + tr[1] + static_cast<size_t>(b) * (static_cast<size_t>(KSH) * 8 * 32 * bits);
    return ptab[tr3_ptab_slot(e, b) + 1];
}

// Bytes of a piece's fc1 slab (its two units) and of its fc2 slab, from the table row and tr3.kmix.
__host__ __device__ __forceinline__ size_t tr3_piece_fc1_bytes(long long km, const long long* tr, int b)
{
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;
    if (!km) return 2 * static_cast<size_t>(tr[4]) * MB;
    return (static_cast<size_t>((km >> (8 * b)) & 15) + static_cast<size_t>((km >> (8 * b + 4)) & 15)) * MB;
}
__host__ __device__ __forceinline__ size_t tr3_piece_fc2_bytes(const long long* tr)
{
    return static_cast<size_t>(tr[4]) * (static_cast<size_t>(KSH) * 8 * 32);
}

#ifndef TR3_UNR
#define TR3_UNR 8    // tiles loaded ahead per warp (bytes in flight per warp = TR3_UNR * tile bytes)
#endif
#ifndef TR3_XCOMP
#define TR3_XCOMP 0   // fc1 x table compact [M][KSU*16+8] halfs instead of the 8-column fragment table (window1)
#endif
#ifndef TR3_PU
#define TR3_PU 2      // window2: fc1 prologue tasks in flight per warp (W2 build: 4)
#endif
#ifndef TR3_EARLY
#define TR3_EARLY 1  // wide order: request a CTA's first weight bytes before its prologue (0 = after it; A/B builds)
#endif

// ---------------------------------------------------------------------------------------------
// FC1 unit = (slot, block b, matrix gate|up, k-split h). 8*WG warps: warp = (group q, tile j).
// Reads 8 tiles (128 output cols) x its k range; writes partial fp32 sums gpart[..][h*WG+q][m][128].
// tr: the slot's expert row of t_tab. mw: the first tile of the unit's matrix (block b, gate|up). BITS: its width
// (tr3_unit); units of blocks the expert does not have exit.
// The 4 hidden-side values at k of block row `row`: the stored vector, or BS (block-scale form, exlr8 v2): the
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

// mask = the slot's columns (A.slot_mask, or the rows of a trellis shared-expert tile, k1.cuh k1_slot). CB = the
// layer's codebook (TR3_*).
template <int BITS, bool E4 = false, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc1_body(const K1Args& A, int plan, int unit, const long long* tr, const uint8_t* mw, int mask, u32* smem)
{
    constexpr int TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int s_ks1 = __ffs(A.KS1) - 1, s_wg = __ffs(A.WG) - 1;
    const int h = unit & (A.KS1 - 1);
    const int u_ks1 = unit >> s_ks1;
    const int mat = u_ks1 & 1;
    const int u_b = u_ks1 >> 1;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int KSU = KSH >> s_ks1;             // k16 steps of this unit
    const int ks0 = h * KSU;
    const int q = warp >> 3, j = warp & 7;
    const int steps = KSU >> s_wg;
    const int ksq = q * steps;                // first step of this warp group inside the unit
    const uint8_t* base = mw + (static_cast<size_t>(ks0 + ksq) * 8 + j) * TB;
#if !defined(TR3_PROBE)
    // wide order: the first group is requested here, before the prologue, so its DRAM latency overlaps the prologue's
    const bool wide = trw_ok<BITS>() && A.t_wide;                 // K3, and K2
    const uint8_t* g0 = base - static_cast<size_t>(j) * TB + j * 256 + lane * 8;
    u64 nw[3] = {0, 0, 0};
    Tr3Lane<BITS, E4, CB> ld;
    ld.init(lane);
    u32 na[TR3_UNR];
    if constexpr (trw_ok<BITS>() && TR3_EARLY)
    {
        if (wide)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(g0 + n * 2048);
        }
    }
    else if constexpr (!trw_ok<BITS>() && TR3_EARLY)
    {
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(base + static_cast<size_t>(p) * 8 * TB, lane);
    }
#endif

    // prologue: xa = f16(H128(x * suh)) for the unit's k range, as token fragments [ks][lane][2].
    // Tasks = (active token, 128 block), spread over all warps, two per iteration so the cold suh loads overlap.
    // Rows of inactive columns stay unwritten: an mma output column depends only on its own token column. A slot holds at
    // most 8 tokens in columns 0..7 (the router splits an expert with more tokens into several slots); slot_tok = the token of
    // a column (M <= 8: column c = token c). The trellis shared expert's scratch slots n_slots + t (SH, one per 8 rows) are
    // not in the route plan: column c of tile t is token 8 t + c.
    {
        const __half* suh = A.t_v1 + (row * 2 + mat) * HID;
        const int nact = __popc(mask & 0xff), ntask = nact * (KSU >> 3);
        const int m0 = __ffs(mask) - 1;
        for (int task = warp; task < ntask; task += TR3_PU * nwarps)
        {
            uint2 xr[TR3_PU], sr[TR3_PU];
            int m[TR3_PU], blk[TR3_PU];
            #pragma unroll
            for (int u = 0; u < TR3_PU; ++u)
            {
                if (task + u * nwarps < ntask)
                {
                    const int tk = task + u * nwarps;
                    if (nact == 1) { m[u] = m0; blk[u] = tk; }
                    else { m[u] = __fns(mask, 0, tk % nact + 1); blk[u] = tk / nact; }
                    const int tok = A.M <= 8 ? m[u] : (slot < A.n_slots ? A.slot_tok[(plan * SMAX + slot) * 8 + m[u]] : (slot - A.n_slots) * 8 + m[u]);
                    const int k = ks0 * 16 + blk[u] * 128 + lane * 4;
                    xr[u] = *reinterpret_cast<const uint2*>(A.x + tok * HID + k);
                    if (BS) sr[u] = tr3_hid4<true>(nullptr, A.t_sg, A.t_v1b, row * 2 + mat, k);
                    else sr[u] = *reinterpret_cast<const uint2*>(suh + k);
                }
            }
            #pragma unroll
            for (int u = 0; u < TR3_PU; ++u)
            {
                if (task + u * nwarps >= ntask) break;
                const float2 x01 = __half22float2(as_h2(xr[u].x)), x23 = __half22float2(as_h2(xr[u].y));
                const float2 s01 = __half22float2(as_h2(sr[u].x)), s23 = __half22float2(as_h2(sr[u].y));
                float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
                had128(v, lane);
                if (TR3_XCOMP && A.xc)
                {
                    u32* xp = smem + (m[u] * (KSU * 16 + 8) + blk[u] * 128 + lane * 4) / 2;
                    xp[0] = pack_h2(v[0], v[1]);
                    xp[1] = pack_h2(v[2], v[3]);
                }
                else store_frag_row(smem, blk[u] * 8, m[u], lane, v);
            }
        }
    }
    __syncthreads();

    const bool xcmp = TR3_XCOMP && A.xc;
    const u32* xa = xcmp ? smem + min(lane >> 2, A.M - 1) * (KSU * 8 + 4) + ksq * 8 + (lane & 3) : smem + (ksq * 32 + lane) * 2;
    const int xs = xcmp ? 8 : 64, xo = xcmp ? 4 : 1;
#define XA0(ks) xa[(ks) * xs]
#define XA1(ks) xa[(ks) * xs + xo]
#if defined(TR3_PROBE)
    Tr3Lane<BITS, E4, CB> ld;
    ld.init(lane);
#endif

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
            for (int p = 0; p < 8; ++p) acc[0] += __int_as_float((r ^ XA0(s + p) ^ XA1(s + p)) & 0x3fffffff);
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
                Tr3Lane<BITS, E4, CB>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | c, 7 * BITS, f0, f1);
                mma16816(acc, f0[0], f1[0], f0[1], f1[1], XA0(s + p), XA1(s + p));
            }
        }
    }
    else
    {
#endif
#if defined(TR3_PROBE)
    u32 na[TR3_UNR];
#endif
    if (trw_ok<BITS>() || !TR3_EARLY)
    {
        #pragma unroll
        for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(base + static_cast<size_t>(p) * 8 * TB, lane);
    }
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
            const u32 x0 = XA0(s + p), x1 = XA1(s + p);
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

#undef XA0
#undef XA1
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
template <int BITS, bool E4 = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc2_body(const K1Args& A, int plan, int unit, const long long* tr, int mask, u32* smem)
{
    constexpr int TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int s_ns2 = __ffs(A.NS2) - 1, s_nw = __ffs(nwarps) - 1;
    const int ns = unit & (A.NS2 - 1);
    const int u_b = unit >> s_ns2;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int GP = A.KS1 * A.WG;
    const int NTU = KSH >> s_ns2;                     // n-tiles of this unit
    const int nt0 = ns * NTU + warp;
    const int ntiles = (NTU - warp + nwarps - 1) >> s_nw;
    const int e = static_cast<int>((tr - A.t_tab) / 6);
    const uint8_t* base = tr3_fc2_block(A.t_fc2, A.t_ptab, e, tr, b, BITS);
    // wide order (tr3_wide.cuh, k_tr3_widen2): the 8 k-tiles of an n-tile in three 64-bit loads. The first n-tile is
    // requested here, before the prologue, so its DRAM latency overlaps the prologue's.
    const bool wide = trw_ok<BITS>() && A.t_wide2;                // K3, and K2
    const uint8_t* wg = base + static_cast<size_t>(nt0) * 8 * TB + lane * 8;
    u64 nw[3] = {0, 0, 0};
    Tr3Lane<BITS, E4, CB> ld;
    ld.init(lane);
    auto tile_ptr = [&](int q) { return base + (static_cast<size_t>(nt0 + (q >> 3) * nwarps) * 8 + (q & 7)) * TB; };
    const int Q = ntiles * 8;
    u32 na[TR3_UNR];
    if constexpr (trw_ok<BITS>() && TR3_EARLY)
    {
        if (wide && ntiles > 0)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + n * 256);
        }
    }
    else if constexpr (!trw_ok<BITS>() && TR3_EARLY)
    {
        if (Q > 0)
        {
            #pragma unroll
            for (int p = 0; p < TR3_UNR; ++p) na[p] = ld.loadw(tile_ptr(p), lane);
        }
    }

    {
        const __half* v2 = A.t_v2 + row * 3 * 128 + lane * 4;
        const int m = warp & 7, mat = warp >> 3;
        const bool active = ((mask >> m) & 1) != 0;
        float gu0[4] = {0.f, 0.f, 0.f, 0.f};
        if (warp < 16 && active)
        {
            const float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX * 8) * 128 + m * 128 + lane * 4;
            float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
            if (GP == 4)
            {
                float4 w[4];
                #pragma unroll
                for (int p = 0; p < 4; ++p) w[p] = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p) * 8 * 128));
                #pragma unroll
                for (int p = 0; p < 4; ++p) { s.x += w[p].x; s.y += w[p].y; s.z += w[p].z; s.w += w[p].w; }
            }
            else if (GP == 8)
            {
                for (int p0 = 0; p0 < 8; p0 += 4)
                {
                    float4 w[4];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) w[p] = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p0 + p) * 8 * 128));
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) { s.x += w[p].x; s.y += w[p].y; s.z += w[p].z; s.w += w[p].w; }
                }
            }
            else
            {
                for (int p = 0; p < GP; ++p)
                {
                    const float4 w = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p) * 8 * 128));
                    s.x += w.x; s.y += w.y; s.z += w.z; s.w += w.w;
                }
            }
            float y[4] = {rnd_h(s.x), rnd_h(s.y), rnd_h(s.z), rnd_h(s.w)};
            had128(y, lane);
            const uint2 sr = *reinterpret_cast<const uint2*>(v2 + mat * 128);
            const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
            gu0[0] = rnd_h(y[0] * s01.x); gu0[1] = rnd_h(y[1] * s01.y);
            gu0[2] = rnd_h(y[2] * s23.x); gu0[3] = rnd_h(y[3] * s23.y);
            if (mat == 1)
                reinterpret_cast<float4*>(smem + 512)[m * 32 + lane] = make_float4(gu0[0], gu0[1], gu0[2], gu0[3]);
        }
        __syncthreads();
        if (warp < 8)
        {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (active)
            {
                const float4 gu1 = reinterpret_cast<const float4*>(smem + 512)[m * 32 + lane];
                const uint2 sr = *reinterpret_cast<const uint2*>(v2 + 2 * 128);
                const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
                v[0] = rnd_h(swiglu_f(gu0[0], gu1.x)) * s01.x;
                v[1] = rnd_h(swiglu_f(gu0[1], gu1.y)) * s01.y;
                v[2] = rnd_h(swiglu_f(gu0[2], gu1.z)) * s23.x;
                v[3] = rnd_h(swiglu_f(gu0[3], gu1.w)) * s23.y;
                had128(v, lane);
            }
            store_frag_row(smem, 0, m, lane, v);
        }
    }
    __syncthreads();

    const u32* ha = smem + lane * 2;
    const int nl = lane >> 2, c = lane & 3;
    const bool wr0 = (mask >> (2 * c)) & 1, wr1 = (mask >> (2 * c + 1)) & 1;
    __half* zb = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + 2 * c) * HID + nl;   // token 2c; token 2c+1 is +HID
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
                Tr3Lane<BITS, E4, CB>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
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
    if ((trw_ok<BITS>() || !TR3_EARLY) && Q > 0)
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
// fixed k, b order. One CTA = (token, one 128-col block), 8 warps x 1 route each (+ warp 8 for SH), summed through smem
// in the exact (2q, 2q+1) pair order of the 4-warp kernel.
template <bool SH = false>
// True only in the CTA that corrected the whole row: all its threads may have stored output.
__device__ __forceinline__ bool tr3_fin_body(const K1Args& A, int plan)
{
    constexpr int NW = SH ? 5 : 4;
    constexpr bool BSF = SH;          // v2 layers: block-scale hidden-side vectors (tr3.sg, tr3.v3b)
    __shared__ float4 red[NW][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m = blockIdx.x / geom::HB, hb = blockIdx.x % geom::HB;
    const int n0 = hb * 128 + lane * 4;
    const uint2 sg = BSF ? *reinterpret_cast<const uint2*>(A.t_sg + HID + n0) : make_uint2(0u, 0u);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float4 vs[g6nr::EB];
    float w = 0.f;
    int nb = 0;
    if (warp < 8)
    {
        const int idx = (plan * G6_MX + m) * 8 + warp;
        const int slot = A.tok_slot[idx];
        const bool on = slot >= 0;
        w = A.tok_w[idx];
        const int col = A.M <= 8 ? m : (on ? A.tok_col[idx] : 0);      // the token's column in its slot
        const int sl = on ? slot : 0;
        const long long* tr = A.t_tab + 6 * A.slot_expert[plan * SMAX + sl];
        nb = on ? static_cast<int>(tr[3]) : 0;
        const size_t row = static_cast<size_t>(tr[2]);
        uint2 zr[g6nr::EB], sr[g6nr::EB];
        __half sc[g6nr::EB];
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < nb)
            {
                zr[b] = *reinterpret_cast<const uint2*>(A.z + ((static_cast<size_t>(sl) * A.B + b) * 8 + col) * HID + n0);
                if constexpr (BSF) sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
                else sr[b] = *reinterpret_cast<const uint2*>(A.t_v3 + (row + b) * HID + n0);
            }
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < nb)
            {
                const float2 z01 = __half22float2(as_h2(zr[b].x)), z23 = __half22float2(as_h2(zr[b].y));
                float v[4] = {z01.x, z01.y, z23.x, z23.y};
                had128(v, lane);
                uint2 s2;
                if constexpr (BSF)
                {
                    const __half2 c = __half2half2(sc[b]);
                    s2 = make_uint2(as_u32(__hmul2(as_h2(sg.x), c)), as_u32(__hmul2(as_h2(sg.y), c)));
                }
                else s2 = sr[b];
                const float2 s01 = __half22float2(as_h2(s2.x)), s23 = __half22float2(as_h2(s2.y));
                if ((warp & 1) == 0)
                {
                    acc[0] += w * (v[0] * s01.x); acc[1] += w * (v[1] * s01.y);
                    acc[2] += w * (v[2] * s23.x); acc[3] += w * (v[3] * s23.y);
                }
                else
                {
                    vs[b] = make_float4(v[0] * s01.x, v[1] * s01.y, v[2] * s23.x, v[3] * s23.y);
                }
            }
    }
    else if (SH && warp == 8)
    {
        const long long* tr = A.t_tab + 6 * A.sh_e;
        const int snb = static_cast<int>(tr[3]);
        const size_t row = static_cast<size_t>(tr[2]);
        uint2 zr[g6nr::EB], sr[g6nr::EB];
        __half sc[g6nr::EB];
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < snb)
            {
                zr[b] = *reinterpret_cast<const uint2*>(A.z + ((static_cast<size_t>(A.n_slots + (m >> 3)) * A.B + b) * 8 + (m & 7)) * HID + n0);
                if constexpr (BSF) sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
                else sr[b] = *reinterpret_cast<const uint2*>(A.t_v3 + (row + b) * HID + n0);
            }
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < snb)
            {
                const float2 z01 = __half22float2(as_h2(zr[b].x)), z23 = __half22float2(as_h2(zr[b].y));
                float v[4] = {z01.x, z01.y, z23.x, z23.y};
                had128(v, lane);
                uint2 s2;
                if constexpr (BSF)
                {
                    const __half2 c = __half2half2(sc[b]);
                    s2 = make_uint2(as_u32(__hmul2(as_h2(sg.x), c)), as_u32(__hmul2(as_h2(sg.y), c)));
                }
                else s2 = sr[b];
                const float2 s01 = __half22float2(as_h2(s2.x)), s23 = __half22float2(as_h2(s2.y));
                acc[0] += v[0] * s01.x; acc[1] += v[1] * s01.y;
                acc[2] += v[2] * s23.x; acc[3] += v[3] * s23.y;
            }
    }
    if ((warp & 1) == 0) red[warp >> 1][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (warp < 8 && (warp & 1) == 1)
    {
        float4 a = red[warp >> 1][lane];
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
            if (b < nb)
            {
                a.x += w * vs[b].x; a.y += w * vs[b].y;
                a.z += w * vs[b].z; a.w += w * vs[b].w;
            }
        red[warp >> 1][lane] = a;
    }
    __syncthreads();
    const float scale = A.steer_scale ? A.steer_scale[m] : 0.f;
    const int slot = A.steer_slot ? A.steer_slot[m] : 0;
    const float alpha = A.steer_slot ? A.steer_slot_alpha[slot] : A.steer_alpha;
    const float* sdir = A.steer_dir + (size_t)slot * HID;
    const bool steer = A.steer_dir && scale != 0.f && alpha != 0.f;
    __shared__ int steer_last;
    __shared__ float steer_kappa;
    if (warp == 0)
    {
        acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        #pragma unroll
        for (int q = 0; q < NW; ++q)
        {
            const float4 v = red[q][lane];
            acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
        }
        const uint2 yr = SH ? make_uint2(0u, 0u) : *reinterpret_cast<const uint2*>(A.ypart + (static_cast<size_t>(A.ysh) + m) * HID + n0);
        const float2 y01 = __half22float2(as_h2(yr.x)), y23 = __half22float2(as_h2(yr.y));
        const float4 shared = SH ? red[NW - 1][lane] : make_float4(y01.x, y01.y, y23.x, y23.y);
        uint2 o;
        o.x = pack_o2(acc[0] + y01.x, acc[1] + y01.y, A.out_bf16);
        o.y = pack_o2(acc[2] + y23.x, acc[3] + y23.y, A.out_bf16);
        *reinterpret_cast<uint2*>(K1_OUT(A) + m * HID + n0) = o;

        // Project only the shared expert contribution. The routed sum in `o` never participates in the dot. One fixed
        // partial per 128-column CTA avoids nondeterministic floating-point atomics; whichever CTA arrives last sums
        // the 48 partials in column order. All five resident warps then share the correction instead of serializing it
        // through warp 0; the existing finalize launch and slot publish remain unchanged.
        if (steer)
        {
            const float4 r = *reinterpret_cast<const float4*>(sdir + n0);
            float dot = shared.x * r.x + shared.y * r.y + shared.z * r.z + shared.w * r.w;
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
            // Bypass stale per-SM cache lines when this scratch is reused by the next layer/step.
            volatile float* partial = A.steer_part + (size_t)m * 48;
            if (lane == 0) partial[blockIdx.x % 48] = dot;
            // Every output-writing lane fences its stores before lane 0 signals this CTA's arrival.
            __threadfence();
            __syncwarp();
            if (lane == 0)
            {
                steer_last = atomicAdd(A.steer_count + m, 1u) == 47u;
                if (steer_last)
                {
                    float total = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 48; ++j) total += partial[j];
                    steer_kappa = (scale * alpha) * total;
                }
            }
        }
    }
    if (!steer) return false;
    __syncthreads();
    if (steer_last)
    {
        const float kappa = steer_kappa;
        for (int c = threadIdx.x * 4; c < HID; c += blockDim.x * 4)
        {
            const float4 d = *reinterpret_cast<const float4*>(sdir + c);
            if (A.out_bf16)
            {
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(K1_OUT(A) + (size_t)m * HID + c);
                const volatile u32* words = reinterpret_cast<const volatile u32*>(dst);
                const uint2 q = make_uint2(words[0], words[1]);
                const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&q);
                *reinterpret_cast<uint2*>(dst) = make_uint2(pack_bf2(__bfloat162float(b[0]) - kappa * d.x,
                                                                     __bfloat162float(b[1]) - kappa * d.y),
                                                            pack_bf2(__bfloat162float(b[2]) - kappa * d.z,
                                                                     __bfloat162float(b[3]) - kappa * d.w));
            }
            else
            {
                __half* dst = K1_OUT(A) + (size_t)m * HID + c;
                const volatile u32* words = reinterpret_cast<const volatile u32*>(dst);
                const uint2 q = make_uint2(words[0], words[1]);
                const float2 x01 = __half22float2(as_h2(q.x)), x23 = __half22float2(as_h2(q.y));
                *reinterpret_cast<uint2*>(dst) = make_uint2(pack_h2(x01.x - kappa * d.x, x01.y - kappa * d.y),
                                                           pack_h2(x23.x - kappa * d.z, x23.y - kappa * d.w));
            }
        }
        // The caller passes every correcting thread to the existing slot publisher, so each writer performs its own
        // system fence before the doorbell. Without a slot, kernel completion provides the boundary.
        __syncthreads();
        if (threadIdx.x == 0) A.steer_count[m] = 0u;
    }
    return steer_last;
}
