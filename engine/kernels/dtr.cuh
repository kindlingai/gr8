// Dense trellis matrices of the exlr8 release (#116): q_b, o_proj and the dense MLP of layers 0-2, stored as trellis K4
// with the E4M3 codebook, 128-block Hadamards and per-channel suh / svh. Decode GEMV on the release's own entries.
// Our own code, written from the release format (HF card, Loader section).
//
// The release stores W (in, out) as diag(suh) Hb T Hb diag(svh). One rank's slice is the release's entries of M.trellis
// for that rank, as stored (u8 [E][entry bytes], 128 channels each), plus the matching slices of M.suh and M.svh:
//   side 1 (q_b, mlp gate and up, record field 4 = 1): entry e = output columns 128e .. 128e + 127,
//          tiles [k16 = 0 .. K/16 - 1][c = 0 .. 7], the order of an expert gate / up half
//   side 2 (o_proj, mlp down, field 4 = 2):        entry e = input rows 128e .. 128e + 127,
//          tiles [b = 0 .. N/128 - 1][r = 0 .. 7][c = 0 .. 7], the order of an expert down half
// Both follow one rule: the tile of output block nb, k16 step s and column tile c (16 output columns) is at
//   w + nb * sB + (s / 8) * sG + ((s % 8) * 8 + c) * TB,   TB = 32 K bytes
//   side 1: sB = entry bytes, sG = 64 TB          side 2: sB = 64 TB, sG = entry bytes
// so the kernels read the entries without a re-layout.
//
// k_dtr: y = H128(H128(x * suh) @ T) * svh for M <= G6_MX rows. CTA = (output block nb, matrix, k part p), 8 * SUB warps,
// warp = (sub-part q, column tile c). The rotations follow the routed experts (tr3.cuh):
//   prologue  xa = f16(H128(x * suh)) for the CTA's k range into shared memory as mma B fragments (ROT), or the
//             fragment tables of an input another launch rotated already (!ROT: mlp down, written by the GLU epilogue)
//   loop      each decoded tile is the A operand of NT mma calls, one per 8 rows, fp32 accumulate
//   reduce    the sub-parts through shared memory, the CTA's fp32 partial [M][128] to part, then the last CTA of block nb
//             sums the k parts in order in fp32 (the experts round this sum to fp16), applies H128 and svh:
//   EPI_OUT   y [M][ldy] fp16, BF16 (fabric partials) or fp32
//   EPI_GLU   gate and up in one launch (matrix 0, 1); the last CTA of block nb forms g = f16(H128(sum_g) * svh_g),
//             u likewise, act = f16(silu(g) * u), and writes f16(H128(act * suh_d)) as down's input fragment tables
//             at k16 steps 8 nb .. 8 nb + 7 (the K1 fc2 prologue's arithmetic on fp32 sums)
// The reduction order is fixed, and the results do not depend on timing. The last CTA zeroes its counter for the next
// launch.
// DEC_SLOT: with Args::slot.send set, EPI_OUT writes BF16 into this rank's send slot and (DEC_SLOT >= 2) every CTA
// takes part in the doorbell (decslot.cuh).
// Prefill: o_proj runs on its tiles (bind_prefill.cuh k_gemm_tr decodes each stage into the bytes the FP8 copy would
// hold, so the result is the copy path's to the bit). q_b and the dense MLP run on make_fp8 copies (the tiles decoded
// once into E4M3 codes for the MXFP8 GEMMs), where reading the tiles measured slower. The prefill binder applies the
// rotations around the GEMMs (Xq::suh, Gemm::osvh).
#pragma once
#include "tr3.cuh"
#ifdef DEC_SLOT
#include "decslot.cuh"
#endif

namespace dtr
{
enum { SIDE_OUT = 1, SIDE_IN = 2 };
enum { EPI_OUT = 0, EPI_GLU = 1 };
enum { O_F16 = 0, O_BF16 = 1, O_F32 = 2 };
constexpr int BITS = 4;               // the release stores every dense trellis matrix at K4
constexpr int UNR = 16;               // tiles in flight per warp
constexpr int SMEM_MAX = 98304;       // dynamic shared memory attribute (init)

struct Args
{
    const uint8_t* w[2] = {};         // the matrix's entries (EPI_GLU: gate, up)
    long long sB = 0, sG = 0;         // tile addressing, bytes
    int K = 0, N = 0;                 // input and output width on this rank
    int ks = 1;                       // k parts = CTAs per output block and matrix
    int M = 0;
    const __half* x = nullptr;        // ROT: input rows [M][ldx] fp16
    int ldx = 0;
    const __half* suh[2] = {};        // ROT: input-side vector [K] per matrix
    const u32* xa = nullptr;          // !ROT: the rotated input as fragment tables, one [K/16][32][2] per 8 rows
    const __half* svh[2] = {};        // output-side vector [N] per matrix
    float* part = nullptr;            // [N/128][matrices][ks][M][128] fp32 partials
    int* cnt = nullptr;               // [N/128] arrivals per output block, zero between launches
    void* y = nullptr;                // EPI_OUT: [M][ldy] of odt
    int ldy = 0, odt = O_F16;
    const __half* suh_d = nullptr;    // EPI_GLU: down's input-side vector [N]
    u32* xd = nullptr;                // EPI_GLU: down's input fragment tables [ceil(M/8)][N/16][32][2]
#ifdef DEC_SLOT
    dslot::Slot slot;                 // EPI_OUT, send != nullptr: the output goes as BF16 to the send slot of slot.c (y unused)
#endif
};

__device__ __forceinline__ void ld4h(const __half* p, float (&v)[4])
{
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(as_h2(r.x)), b = __half22float2(as_h2(r.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

template <int NT, int SUB, bool ROT, int EPI>
__global__ void __launch_bounds__(256 * SUB) k_dtr(Args a)
{
    constexpr int TB = 32 * BITS, NM = EPI == EPI_GLU ? 2 : 1, NW = 8 * SUB;
    extern __shared__ __align__(16) u32 sm[];
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, q = warp >> 3;
    const int p = blockIdx.x % a.ks, mat = NM == 2 ? (blockIdx.x / a.ks) & 1 : 0, nb = blockIdx.x / (a.ks * NM);
    const int K16 = a.K / 16, sc = K16 / a.ks, sw = sc / SUB, s0c = p * sc, s0 = s0c + q * sw;
    const uint8_t* wb = a.w[mat] + nb * a.sB + j * TB;
    auto tile = [&](int s) { return wb + (s >> 3) * a.sG + (s & 7) * 8 * TB; };
    Tr3Lane<BITS, true> ld;
    ld.init(lane);
    // the first weight bytes are requested before the prologue, so their DRAM latency overlaps it
    u32 na[UNR];
    #pragma unroll
    for (int u = 0; u < UNR; ++u) na[u] = u < sw ? ld.loadw(tile(s0 + u), lane) : 0u;

    const int tabw = ROT ? NT * sc * 64 : 0;          // u32 of the prologue's fragment tables
    if constexpr (ROT)
    {
        // xa = f16(H128(x * suh)) for k = 16 s0c .. 16 (s0c + sc), task = (row, 128 block), one warp each. Rows >= M of a
        // tile stay unwritten: an mma output column depends only on its own row's column.
        const __half* suh = a.suh[mat];
        const int ntask = a.M * (sc / 8);
        for (int t = warp; t < ntask; t += NW)
        {
            const int m = t % a.M, blk = t / a.M, k = s0c * 16 + blk * 128 + lane * 4;
            float x[4], s[4];
            ld4h(a.x + static_cast<size_t>(m) * a.ldx + k, x);
            ld4h(suh + k, s);
            float v[4] = {x[0] * s[0], x[1] * s[1], x[2] * s[2], x[3] * s[3]};
            had128(v, lane);
            store_frag_row(sm + (m >> 3) * sc * 64, blk * 8, m & 7, lane, v);
        }
        __syncthreads();
    }
    const u32* xb = ROT ? sm + (q * sw) * 64 + lane * 2 : a.xa + static_cast<size_t>(s0) * 64 + lane * 2;
    const int xts = ROT ? sc * 64 : K16 * 64;         // u32 between two row tiles' tables

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int s = 0; s < sw; s += UNR)
    {
        u32 wa[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u) wa[u] = na[u];
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
            if (s + UNR + u < sw) na[u] = ld.loadw(tile(s0 + s + UNR + u), lane);
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
        {
            if (s + u >= sw) break;
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                const u32* xg = xb + t * xts + (s + u) * 64;
                u32 b0, b1;
                if (ROT) { b0 = xg[0]; b1 = xg[1]; }
                else { b0 = __ldcg(xg); b1 = __ldcg(xg + 1); }
                mma16816(acc[t], f0[0], f1[0], f0[1], f1[1], b0, b1);
            }
        }
    }

    // sub-parts 1 .. SUB-1 -> sub-part 0, in order
    if constexpr (SUB > 1)
    {
        float4* red = reinterpret_cast<float4*>(sm + tabw);
        if (q)
            #pragma unroll
            for (int t = 0; t < NT; ++t) red[(((q - 1) * 8 + j) * NT + t) * 32 + lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        __syncthreads();
        if (!q)
            for (int r = 1; r < SUB; ++r)
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const float4 v = red[(((r - 1) * 8 + j) * NT + t) * 32 + lane];
                    acc[t][0] += v.x; acc[t][1] += v.y; acc[t][2] += v.z; acc[t][3] += v.w;
                }
    }
    // the CTA's partial part[nb][mat][p][m][128]; acc[t]: (col 16 j + l/4, rows 8t + 2c, + 1), (col + 8, same rows)
    const int M = a.M;
    if (!q)
    {
        float* pp = a.part + ((static_cast<size_t>(nb) * NM + mat) * a.ks + p) * M * 128;
        const int col = j * 16 + (lane >> 2), c = lane & 3;
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const int m0 = t * 8 + 2 * c;
            if (m0 < M) { pp[m0 * 128 + col] = acc[t][0]; pp[m0 * 128 + col + 8] = acc[t][2]; }
            if (m0 + 1 < M) { pp[(m0 + 1) * 128 + col] = acc[t][1]; pp[(m0 + 1) * 128 + col + 8] = acc[t][3]; }
        }
    }
    const int arrivals = NM * a.ks;
    if (arrivals > 1) __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = arrivals == 1 || atomicAdd(a.cnt + nb, 1) == arrivals - 1;
    __syncthreads();
#if DEC_SLOT >= 2
    const bool pub = EPI == EPI_OUT && a.slot.send;
#endif
    if (!last)
    {
#if DEC_SLOT >= 2
        if (pub) dslot::publish(a.slot, false);
#endif
        return;
    }
    if (arrivals > 1) __threadfence();
#ifdef DEC_SLOT
    void* const yo = EPI == EPI_OUT && a.slot.send ? dslot::ptr(a.slot) : a.y;
    const int odt = EPI == EPI_OUT && a.slot.send ? O_BF16 : a.odt;
#else
    void* const yo = a.y;
    const int odt = a.odt;
#endif

    const int n0 = nb * 128 + lane * 4;
    auto sum = [&](int mt, int m, float (&v)[4]) {
        const float* pb = a.part + (static_cast<size_t>(nb) * NM + mt) * a.ks * M * 128 + m * 128 + lane * 4;
        v[0] = v[1] = v[2] = v[3] = 0.f;
        for (int r = 0; r < a.ks; ++r)
        {
            const float4 w = __ldcg(reinterpret_cast<const float4*>(pb + static_cast<size_t>(r) * M * 128));
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
    };
    for (int m = warp; m < M; m += NW)
    {
        if constexpr (EPI == EPI_OUT)
        {
            float v[4], s[4];
            sum(0, m, v);
            had128(v, lane);
            ld4h(a.svh[0] + n0, s);
            #pragma unroll
            for (int e = 0; e < 4; ++e) v[e] *= s[e];
            const size_t o = static_cast<size_t>(m) * a.ldy + n0;
            if (odt == O_F32) *reinterpret_cast<float4*>(static_cast<float*>(yo) + o) = make_float4(v[0], v[1], v[2], v[3]);
            else if (odt == O_BF16) *reinterpret_cast<uint2*>(static_cast<uint16_t*>(yo) + o) = make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]));
            else *reinterpret_cast<uint2*>(static_cast<__half*>(yo) + o) = make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
        }
        else
        {
            float gu[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                float s[4];
                sum(mt, m, gu[mt]);
                had128(gu[mt], lane);
                ld4h(a.svh[mt] + n0, s);
                #pragma unroll
                for (int e = 0; e < 4; ++e) gu[mt][e] = rnd_h(gu[mt][e] * s[e]);
            }
            float sd[4], v[4];
            ld4h(a.suh_d + n0, sd);
            #pragma unroll
            for (int e = 0; e < 4; ++e) v[e] = rnd_h(silu_f(gu[0][e]) * gu[1][e]) * sd[e];
            had128(v, lane);
            store_frag_row(a.xd + static_cast<size_t>(m >> 3) * (a.N / 16) * 64, nb * 8, m & 7, lane, v);
        }
    }
    if (arrivals > 1 && threadIdx.x == 0) a.cnt[nb] = 0;
#if DEC_SLOT >= 2
    if (pub) dslot::publish(a.slot, true);
#endif
}

// ------------------------------------------------------------------------------------------ binding
// One rank's slice of a dense trellis matrix: entries u8 [E][entry bytes], suh f16 [K], svh f16 [N]. T is any tensor
// view with p, nd and shape (bind::Tensor, g6rt::TensorRef).
struct Mat
{
    const uint8_t* w = nullptr;
    const __half *suh = nullptr, *svh = nullptr;
    int K = 0, N = 0, side = 0;
    long long sB = 0, sG = 0;
    long long bytes() const { return static_cast<long long>(K) * N / 2; }    // K4
};

template <class T> inline const char* make_mat(const T& ent, const T& suh, const T& svh, int side, Mat& m)
{
    m = Mat();
    if (!ent.p) return nullptr;
    if (!suh.p || !svh.p || ent.nd != 2 || suh.nd != 1 || svh.nd != 1) return "dense trellis matrix is not entries [E][bytes] + suh [K] + svh [N]";
    const int K = static_cast<int>(suh.shape[0]), N = static_cast<int>(svh.shape[0]), E = static_cast<int>(ent.shape[0]);
    const long long eb = ent.shape[1];
    if (K % 128 || N % 128) return "dense trellis matrix: in and out widths must be multiples of 128";
    if (side != SIDE_OUT && side != SIDE_IN) return "dense trellis matrix: side must be 1 (output entries) or 2 (input entries)";
    if ((side == SIDE_OUT ? N : K) != E * 128) return "dense trellis matrix: entry count does not match the sharded side's width";
    const long long tb = 32 * BITS, want = static_cast<long long>(side == SIDE_OUT ? K : N) / 16 * 8 * tb;
    if (eb != want) return "dense trellis matrix: entry bytes are not a K4 entry of this shape (only K4 is supported)";
    m.w = static_cast<const uint8_t*>(ent.p);
    m.suh = static_cast<const __half*>(suh.p);
    m.svh = static_cast<const __half*>(svh.p);
    m.K = K; m.N = N; m.side = side;
    m.sB = side == SIDE_OUT ? eb : 64 * tb;
    m.sG = side == SIDE_OUT ? 64 * tb : eb;
    return nullptr;
}

// One layer's dense trellis matrices under the names the loader gives them (find("L001.q_b.trellis") -> bind::Tensor,
// p = nullptr: absent). Per matrix NAME in q_b, o_proj, dense.gate, dense.up, dense.down:
//   LLLL.NAME.trellis  u8 [E][entry bytes]  the rank's rows of the release's M.trellis, as stored
//   LLLL.NAME.suh      f16 [K]              M.suh, the rank's slice on side 2 (o_proj, down), whole on side 1
//   LLLL.NAME.svh      f16 [N]              M.svh, the rank's slice on side 1 (q_b, gate, up), whole on side 2
// The rank's channels follow g6's split: q_b / o_proj the rank's heads (g6nr::start / width of 64, 256 channels each),
// the dense MLP DENSE_CH contiguous channels.
constexpr int NMAT = 5;
static const char* const MAT_NAMES[NMAT] = {"q_b", "o_proj", "dense.gate", "dense.up", "dense.down"};
static const int MAT_SIDE[NMAT] = {SIDE_OUT, SIDE_IN, SIDE_OUT, SIDE_OUT, SIDE_IN};
struct LayerMats
{
    Mat q_b, o_proj, gate, up, down;
    Mat* at(int i) { Mat* m[NMAT] = {&q_b, &o_proj, &gate, &up, &down}; return m[i]; }
    bool any() const { return q_b.w || o_proj.w || gate.w; }
};
template <class F> inline const char* layer_mats(int layer, F find, LayerMats& L)
{
    L = LayerMats();
    char n[3][64];
    for (int i = 0; i < NMAT; ++i)
    {
        snprintf(n[0], 64, "L%03d.%s.trellis", layer, MAT_NAMES[i]);
        snprintf(n[1], 64, "L%03d.%s.suh", layer, MAT_NAMES[i]);
        snprintf(n[2], 64, "L%03d.%s.svh", layer, MAT_NAMES[i]);
        if (const char* e = make_mat(find(n[0]), find(n[1]), find(n[2]), MAT_SIDE[i], *L.at(i))) return e;
    }
    if (!L.gate.w != !L.up.w || !L.gate.w != !L.down.w) return "dense trellis MLP needs gate, up and down together";
    if (L.q_b.w && L.o_proj.w && L.o_proj.K != L.q_b.N) return "dense trellis q_b width is not o_proj's input width";
    return nullptr;
}

// Scratch of the launches: fp32 partials and the block counters. Zero the counters once. Every launch leaves them zero.
// Launches on one stream may share it.
struct Scratch
{
    float* part = nullptr;
    size_t part_n = 0;                // floats
    int* cnt = nullptr;
    int cnt_n = 0;
};
// floats of part for a launch: N / 128 blocks x matrices x ks x M x 128
inline size_t part_floats(int N, int nm, int ks, int M) { return static_cast<size_t>(N) / 128 * nm * ks * M * 128; }

struct Gemv
{
    int on = 0, nt = 1, sub = 2, rot = 1, epi = EPI_OUT, side = 0;
    Args a;
    unsigned grid = 0;
    size_t smem = 0;
};

// the instantiation of a launch (NT 1 .. g6mx::NT, SUB 1 | 2; ROT with EPI_OUT or EPI_GLU, !ROT with EPI_OUT)
template <int NT, int SUB> inline const void* pick3(int rot, int epi)
{
    if (!rot) return reinterpret_cast<const void*>(k_dtr<NT, SUB, false, EPI_OUT>);
    return epi == EPI_GLU ? reinterpret_cast<const void*>(k_dtr<NT, SUB, true, EPI_GLU>) : reinterpret_cast<const void*>(k_dtr<NT, SUB, true, EPI_OUT>);
}
template <int NT> inline const void* pick2(int sub, int rot, int epi) { return sub == 2 ? pick3<NT, 2>(rot, epi) : pick3<NT, 1>(rot, epi); }
inline const void* kernel_of(int nt, int sub, int rot, int epi)
{
    const void* f = nullptr;
    g6mx::nt_dispatch(nt, [&](auto N) { f = pick2<decltype(N)::value>(sub, rot, epi); });
    return f;
}

inline void launch(const Gemv& g, cudaStream_t st)
{
    if (!g.on) return;
    void* args[] = {const_cast<Args*>(&g.a)};
    cudaLaunchKernel(kernel_of(g.nt, g.sub, g.rot, g.epi), dim3(g.grid), dim3(256 * g.sub), args, g.smem, st);
}

// once per process, before the first launch or graph capture: every instantiation may use up to SMEM_MAX
inline cudaError_t init()
{
    for (int nt = 1; nt <= g6mx::NT; ++nt)
        for (int sub = 1; sub <= 2; ++sub)
            for (int v = 0; v < 3; ++v)
            {
                const cudaError_t e = cudaFuncSetAttribute(kernel_of(nt, sub, v != 0, v == 2 ? EPI_GLU : EPI_OUT),
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
                if (e != cudaSuccess) return e;
            }
    return cudaSuccess;
}

// (ks, sub) per shape, best of a sweep on one GB10 at the 2200 MHz cap, TP4 rank shapes, M = 1, 3, 8 (dtr_test --bench
// --sweep 1). Rows above 8 take the 8-row choice (not tuned). Other shapes: ks 4, sub 1, adjusted by fill.
inline void split_for(int K, int side, int epi, int rot, int M, int& ks, int& sub)
{
    ks = 4; sub = 1;
    if (epi == EPI_GLU) { ks = 16; sub = M <= 3 ? 2 : 1; }                    // mlp gate + up, K 6144
    else if (!rot) { ks = M <= 3 ? 24 : 12; sub = M <= 3 ? 1 : 2; }          // mlp down on the GLU's tables, K 3072
    else if (side == SIDE_OUT && K == 2048) ks = 16;                         // q_b
    else if (side == SIDE_OUT) ks = 6;                                       // gate or up alone, K 6144
    else if (K >= 4096) ks = M == 1 ? 32 : 4;                                // o_proj
    else ks = M <= 3 ? 24 : 3;                                               // down with its own rotation
}

// k parts and sub-parts (ks <= 0: split_for): ks is halved until the CTA's k range is whole 128 blocks per sub-part pair and fits SMEM_MAX
inline const char* fill(Gemv& g, int nm, int M, int ks, int sub, int rot, const Scratch& sc, int cnt_off)
{
    const int K16 = g.a.K / 16, NT = g6mx::tiles(M);
    if (ks <= 0) split_for(g.a.K, g.side, g.epi, rot, M, ks, sub);
    if (M < 1 || M > G6_MX) return "dense trellis GEMV: M must be 1..G6_MX";
    if (sub != 1 && sub != 2) return "dense trellis GEMV: sub must be 1 or 2";
    auto smem = [&](int k) { return static_cast<size_t>(rot ? NT * (K16 / k) * 256 : 0) + static_cast<size_t>(sub - 1) * 8 * NT * 32 * 16; };
    while (ks > 1 && (K16 % ks || (K16 / ks) % 8 || (K16 / ks) % sub)) ks >>= 1;
    while (smem(ks) > SMEM_MAX && ks * 2 <= K16 / 8 && K16 % (ks * 2) == 0) ks *= 2;
    if (K16 % ks || (K16 / ks) % 8 || (K16 / ks) % sub || smem(ks) > SMEM_MAX) return "dense trellis GEMV: no k split fits this shape";
    const int nblk = g.a.N / 128;
    if (part_floats(g.a.N, nm, ks, M) > sc.part_n) return "dense trellis GEMV: scratch partials too small";
    if (cnt_off + nblk > sc.cnt_n) return "dense trellis GEMV: scratch counters too few";
    g.a.ks = ks; g.a.M = M;
    g.a.part = sc.part; g.a.cnt = sc.cnt + cnt_off;
    g.nt = NT; g.sub = sub; g.rot = rot;
    g.grid = static_cast<unsigned>(nblk * nm * ks);
    g.smem = smem(ks);
    g.on = 1;
    return nullptr;
}

// y [M][ldy] (odt) = the matrix applied to x [M][ldx] fp16 (rot = 1), or to the rotated fragment tables xa (rot = 0).
// ks <= 0: the split from split_for.
// cnt_off: the launch's counters start there (N / 128 of them); launches that may overlap need their own.
inline const char* make_out(const Mat& m, int M, const __half* x, int ldx, const u32* xa, void* y, int ldy, int odt,
                            const Scratch& sc, int cnt_off, int ks, int sub, Gemv& g)
{
    g = Gemv();
    if (!m.w) return nullptr;
    Args& a = g.a;
    a.w[0] = m.w; a.sB = m.sB; a.sG = m.sG; a.K = m.K; a.N = m.N;
    a.x = x; a.ldx = ldx; a.suh[0] = m.suh; a.xa = xa; a.svh[0] = m.svh;
    a.y = y; a.ldy = ldy; a.odt = odt;
    if (x && ldx % 4) return "dense trellis GEMV: ldx must be a multiple of 4";
    if (ldy % 4) return "dense trellis GEMV: ldy must be a multiple of 4";
    g.epi = EPI_OUT;
    g.side = m.side;
    return fill(g, 1, M, ks, sub, x ? 1 : 0, sc, cnt_off);
}

// The dense MLP's gate and up in one launch on x [M][ldx]; writes down's input fragment tables xd [ceil(M/8)][I/16][64]
// (I = gate / up width = down's K). Run the down matrix after it with make_out(down, M, nullptr, 0, xd, ...).
inline const char* make_glu(const Mat& gate, const Mat& up, const Mat& down, int M, const __half* x, int ldx, u32* xd,
                            const Scratch& sc, int cnt_off, int ks, int sub, Gemv& g)
{
    g = Gemv();
    if (!gate.w) return nullptr;
    if (!up.w || !down.w) return "dense trellis MLP without up or down";
    if (gate.side != SIDE_OUT || up.side != SIDE_OUT || down.side != SIDE_IN || gate.K != up.K || gate.N != up.N || down.K != gate.N ||
        gate.sB != up.sB || gate.sG != up.sG)
        return "dense trellis MLP shapes do not chain (gate, up [K -> I] side 1, down [I -> N] side 2)";
    if (ldx % 4) return "dense trellis GEMV: ldx must be a multiple of 4";
    Args& a = g.a;
    a.w[0] = gate.w; a.w[1] = up.w; a.sB = gate.sB; a.sG = gate.sG; a.K = gate.K; a.N = gate.N;
    a.x = x; a.ldx = ldx; a.suh[0] = gate.suh; a.suh[1] = up.suh; a.svh[0] = gate.svh; a.svh[1] = up.svh;
    a.suh_d = down.suh; a.xd = xd;
    g.epi = EPI_GLU;
    g.side = SIDE_OUT;
    return fill(g, 2, M, ks, sub, 1, sc, cnt_off);
}

// ------------------------------------------------------------------------------------------ prefill: the FP8 copy
// grid ceil(N * K / 256), block 256: one thread per weight (n, k), k fastest
__global__ void __launch_bounds__(256) k_dtr_fp8(const uint8_t* w, long long sB, long long sG, int K, int N, uint8_t* codes)
{
    constexpr int TB = 32 * BITS;
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(N) * K) return;
    const int n = static_cast<int>(i / K), k = static_cast<int>(i % K);
    const int s = k >> 4, kk = k & 15, c = (n & 127) >> 4, nn = n & 15;
    const uint8_t* p = w + (n >> 7) * sB + (s >> 3) * sG + ((s & 7) * 8 + c) * TB;
    // tile element (kk, nn) is stream weight t = lane * 8 + j of the mma A fragment order (tr3.cuh Tr3Lane)
    const int lane = (nn & 7) * 4 + ((kk & 7) >> 1), jj = (kk & 1) | (((kk >> 3) & 1) << 1) | ((nn >> 3) << 2), t = lane * 8 + jj;
    // state of weight t: the 16 stream bits ending at bit (t + 1) K, MSB first over LE u32 words, tail-biting
    constexpr int NB = 256 * BITS;
    u32 st = 0;
    for (int b = 0; b < 16; ++b)
    {
        const int bit = (((t + 1) * BITS - 16 + b) % NB + NB) % NB;
        const u32 word = *reinterpret_cast<const u32*>(p + (bit >> 5) * 4);
        st = (st << 1) | ((word >> (31 - (bit & 31))) & 1u);
    }
    const u32 h2 = tr3_cb2<true>(st, 0u);
    unsigned short e;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e) : "r"(h2));
    codes[i] = static_cast<uint8_t>(e & 0xff);
}

// The matrix's tiles T (still rotated) as E4M3 codes [N][K] + E8M0 scales [N][K/32] all 127: the MXFP8 GEMM operands
// (bindp::Gemm). The codebook values are E4M3 already, so the copy is exact. Allocates N K + N K / 32 bytes. Synchronous.
struct Fp8
{
    uint8_t* codes = nullptr;
    uint8_t* scales = nullptr;
    int K = 0, N = 0;
};
inline const char* make_fp8(const Mat& m, cudaStream_t st, Fp8& F)
{
    F = Fp8();
    if (!m.w) return nullptr;
    const size_t nc = static_cast<size_t>(m.N) * m.K;
    if (cudaMalloc(&F.codes, nc) != cudaSuccess || cudaMalloc(&F.scales, nc / 32) != cudaSuccess) return "dtr make_fp8: cudaMalloc failed";
    F.K = m.K; F.N = m.N;
    cudaMemsetAsync(F.scales, 127, nc / 32, st);
    k_dtr_fp8<<<static_cast<unsigned>((nc + 255) / 256), 256, 0, st>>>(m.w, m.sB, m.sG, m.K, m.N, F.codes);
    return cudaStreamSynchronize(st) == cudaSuccess ? nullptr : "dtr make_fp8: decode failed";
}
inline void free_fp8(Fp8& F)
{
    cudaFree(F.codes);
    cudaFree(F.scales);
    F = Fp8();
}

}   // namespace dtr
