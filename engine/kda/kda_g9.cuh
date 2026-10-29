// G9: KDA scan + conv prefill on CH16 tensor cores. ON path, opt-in (G9_KDA_SCAN != 0);
// the OFF path (kda_prefill.cuh) is untouched. Everything here lives in kda::g9: templates, inline functions and
// kernels only, no __device__ globals, no macros.
//
// Scratch reuse (doc sec 4.3): ksc.C (MP x 3dl fp32) holds six bf16 planes [MP][dl] at plane k = byte k * MP * dl * 2:
//   0 Qn, 1 Kn, 2 V, 3 G16, 4 Z16, 5 O16.
//   kda::g9::conv16  Pq|k|v fp32 -> Qn, Kn (l2-normalized, bf16, QSCALE applied later in K1), V (bf16)
//   kda::g9::out16   O16, Z16 -> y = o * r * onorm * sigm(z) to OF (frag table, bf16) and/or OH (fp16), as out_2
#pragma once
#include "kda_prefill.cuh"
#include "../knobs-rs/gr8_knobs.h"

namespace kda
{
namespace g9
{
constexpr int CH16 = 16;               // rows per chunk
constexpr int CV_ROWS = 8;             // conv16: rows per CTA
constexpr int CV_T = 3 * DK;           // conv16: threads = 3 streams x 128 channels
constexpr int OUT_ROWS = 8;            // out16: rows per CTA, one warp per row
constexpr int NPLANE = 6;

enum Plane { QN = 0, KN = 1, VV = 2, G16 = 3, Z16 = 4, O16 = 5 };

// the six bf16 planes in a.C; MP = rows of the scratch (ksc.rows), M <= MP
struct Planes
{
    uint16_t* p[NPLANE] = {};
    int MP = 0;
};
inline Planes planes(const Args& a, int MP)
{
    Planes pl;
    pl.MP = MP;
    uint16_t* base = reinterpret_cast<uint16_t*>(a.C);
    for (int k = 0; k < NPLANE; ++k) pl.p[k] = base + static_cast<size_t>(k) * MP * a.dl();
    return pl;
}

struct G9Args
{
    Args a;
    Planes pl;
    float* dbg = nullptr;   // test only: conv16 stores its pre-rounding fp32 c here, layout of Args::C ([m][3][dl]); null in the engine; only stored when built with -DG9_CONV_DBG
};

// ------------------------------------------------------------------------------------------------------ conv16
// CTA (8-row block, head), thread = (stream, channel d). 4-tap sliding window over the thread's rows in registers, the op
// order of k_kda_conv_pf (acc = x0 w0, += x1 w1, += x2 w2, += x3 w3, silu). Rows < 3 of the call take the slot's conv
// history (k_kda_conv_end_pf, launched after this kernel, writes the new one). Row sums of squares of q and k: warp
// shuffle, then the 4 warps of the stream through smem. All loads and stores are predicated on row < M.
__global__ void __launch_bounds__(CV_T) k_conv16(G9Args g)
{
    __shared__ float red[2][4][CV_ROWS];
    const Args& a = g.a;
    const int dl = a.dl(), ld = a.pl().ld(), slot = a.row_slot[0], M = a.M;
    const int t = threadIdx.x, stream = t / DK, d = t % DK, h = blockIdx.y, warp = (t % DK) / 32, lane = t % 32;
    const int ch = h * DK + d, rem = stream * dl + ch, m0 = blockIdx.x * CV_ROWS;
    const uint16_t* w = a.conv_w[stream] + static_cast<size_t>(ch) * CONV;
    float wt[CONV];
    for (int j = 0; j < CONV; ++j) wt[j] = bf2f(w[j]);
    auto x = [&](int r) { return r >= 0 ? a.P[static_cast<size_t>(r) * ld + rem] : a.cst[a.c_off(slot, NHIST + r, stream, ch)]; };
    float win[CONV];                    // win[j] = x(m - NHIST + j)
    win[0] = x(m0 - NHIST); win[1] = x(m0 - NHIST + 1); win[2] = x(m0 - NHIST + 2);
    float c[CV_ROWS];
    for (int i = 0; i < CV_ROWS; ++i)
    {
        const int m = m0 + i;
        c[i] = 0.f;
        if (m < M)
        {
            win[3] = x(m);
            // the exact FMUL + 3 x FFMA chain of k_kda_conv_pf (SASS of the base kernel), pinned with intrinsics
            float acc = __fmul_rn(win[0], wt[0]);
            acc = __fmaf_rn(win[1], wt[1], acc);
            acc = __fmaf_rn(win[2], wt[2], acc);
            acc = __fmaf_rn(win[3], wt[3], acc);
            c[i] = silu(acc);
#ifdef G9_CONV_DBG
            if (g.dbg) g.dbg[static_cast<size_t>(m) * 3 * dl + rem] = c[i];
#endif
            win[0] = win[1]; win[1] = win[2]; win[2] = win[3];
        }
    }
    if (stream < 2)
    {
        float ss[CV_ROWS];
        for (int i = 0; i < CV_ROWS; ++i)
        {
            float v = c[i] * c[i];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            ss[i] = v;
        }
        if (lane == 0)
            for (int i = 0; i < CV_ROWS; ++i) red[stream][warp][i] = ss[i];
    }
    __syncthreads();
    uint16_t* dst = g.pl.p[stream == 0 ? QN : stream == 1 ? KN : VV];
    for (int i = 0; i < CV_ROWS; ++i)
    {
        const int m = m0 + i;
        if (m >= M) break;
        float y = c[i];
        if (stream < 2)
        {
            const float s = (red[stream][0][i] + red[stream][1][i]) + (red[stream][2][i] + red[stream][3][i]);
            y = c[i] / sqrtf(s + L2_EPS);
        }
        dst[static_cast<size_t>(m) * dl + ch] = f2bf_rn(y);
    }
}

inline void launch_conv16(const G9Args& g, cudaStream_t st)
{
    const Args& a = g.a;
    k_conv16<<<dim3((a.M + CV_ROWS - 1) / CV_ROWS, a.nh), CV_T, 0, st>>>(g);
    k_kda_conv_end_pf<<<(3 * a.dl() + 255) / 256, 256, 0, st>>>(a);
}

// ------------------------------------------------------------------------------------------------------ out16
// CTA (8-row block, head), warp per row; lane holds 4 consecutive v. y = o * r * onorm * sigm(z), r = 1 / sqrt(mean o^2
// + eps) (fp32 shuffle sum). Writes exactly what out_2 writes minus OG: OF frag table (f2bf_rn) when a.OF, OH (f2h_rn)
// when a.OH. Rows >= M are not touched.
__device__ __forceinline__ void out16_row(const G9Args& g, int m, int h, int lane, uint2 zv)
{
    const Args& a = g.a;
    const int dl = a.dl();
    const size_t i0 = static_cast<size_t>(m) * dl + h * DK + lane * 4;
    const uint2 ov = *reinterpret_cast<const uint2*>(g.pl.p[O16] + i0);
    const float o[4] = {bf2f(static_cast<uint16_t>(ov.x)), bf2f(static_cast<uint16_t>(ov.x >> 16)),
                        bf2f(static_cast<uint16_t>(ov.y)), bf2f(static_cast<uint16_t>(ov.y >> 16))};
    const float z[4] = {bf2f(static_cast<uint16_t>(zv.x)), bf2f(static_cast<uint16_t>(zv.x >> 16)),
                        bf2f(static_cast<uint16_t>(zv.y)), bf2f(static_cast<uint16_t>(zv.y >> 16))};
    float ss = (o[0] * o[0] + o[1] * o[1]) + (o[2] * o[2] + o[3] * o[3]);
    for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    const float r = 1.f / sqrtf(ss / DV + RMS_EPS);
    uint16_t hv[4];
    for (int j = 0; j < 4; ++j)
    {
        const int t = lane * 4 + j;
        const float y = o[j] * r * bf2f(a.onorm[t]) * sigm(z[j]);
        if (a.OF) a.OF[frag_u16(m, h * DK + t, dl)] = f2bf_rn(y);
        hv[j] = f2h_rn(y);
    }
    if (a.OH)
    {
        uint2 hw;
        hw.x = hv[0] | (static_cast<uint32_t>(hv[1]) << 16);
        hw.y = hv[2] | (static_cast<uint32_t>(hv[3]) << 16);
        *reinterpret_cast<uint2*>(a.OH + i0) = hw;
    }
}
__global__ void __launch_bounds__(OUT_ROWS * 32) k_out16(G9Args g)
{
    const int M = g.a.M, h = blockIdx.y, lane = threadIdx.x % 32, m = blockIdx.x * OUT_ROWS + threadIdx.x / 32;
    if (m >= M) return;
    const size_t i0 = static_cast<size_t>(m) * g.a.dl() + h * DK + lane * 4;
    out16_row(g, m, h, lane, *reinterpret_cast<const uint2*>(g.pl.p[Z16] + i0));
}

inline void launch_out16(const G9Args& g, cudaStream_t st)
{
    const Args& a = g.a;
    k_out16<<<dim3((a.M + OUT_ROWS - 1) / OUT_ROWS, a.nh), OUT_ROWS * 32, 0, st>>>(g);
}

// ------------------------------------------------------------------------------------------------------ PTX helpers
__device__ inline uint32_t pack_bf16(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// D = A (16x16 bf16, row) * B (16x8 bf16, col) + D, fp32
__device__ inline void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ inline uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ inline void cp_async16(uint32_t dst, const void* src, bool valid)   // valid == false: 16 zero bytes, src not read
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ inline void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ inline void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ inline void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ inline uint32_t lds32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// the workspace of K1 -> K2 (kps.QE = kd, kps.KD = qd, kps.W = kr as bf16 [MP][dl]; kps.U = per (head, tile) blocks)
struct Tape
{
    uint16_t *kd = nullptr, *qd = nullptr, *kr = nullptr;
    uint8_t* U = nullptr;               // block (h * NT + t) * 1536 B: INVb bf16 [16][16] +0, Mqk bf16 [16][16] +512, e^gt fp32 [128] +1024
    int NT = 0;
};
constexpr int U_BLOCK = 1536;

// ------------------------------------------------------------------------------------------------------ gz16
// G16 = bf16(f_b (f_a x)), Z16 = bf16(g_b (g_a x)) on tensor cores: CTA (64-row tile, 64-channel tile, mat), 8 warps
// (4 row slices x 2 channel halves), K = 128 in smem. A = P columns fp32 -> bf16 on load (rows >= M zero-filled, never
// read), B = f_b / g_b bf16 [channel][k]. Gate and dt_bias are applied in K1. Stores predicated on row < M.
constexpr int GZ_PAD = 136;
__global__ void __launch_bounds__(256) k_gz16(G9Args g)
{
    __shared__ __align__(16) uint16_t As[64][GZ_PAD];
    __shared__ __align__(16) uint16_t Bs[64][GZ_PAD];
    const Args& a = g.a;
    const int dl = a.dl(), ld = a.pl().ld(), M = a.M, mat = blockIdx.z, m0 = blockIdx.x * 64, c0 = blockIdx.y * 64;
    const int col = mat ? a.pl().ga() : a.pl().fa();
    const uint16_t* wsrc = mat ? a.gb : a.fb;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, gi = lane >> 2, ti = lane & 3;
    for (int q = t; q < 64 * 32; q += 256)
    {
        const int r = q / 32, c4 = q % 32;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m0 + r < M) v = *reinterpret_cast<const float4*>(a.P + static_cast<size_t>(m0 + r) * ld + col + 4 * c4);
        uint2 w;
        w.x = pack_bf16(v.x, v.y); w.y = pack_bf16(v.z, v.w);
        *reinterpret_cast<uint2*>(&As[r][4 * c4]) = w;
    }
    for (int q = t; q < 64 * 16; q += 256)
    {
        const int r = q / 16, c = q % 16;
        *reinterpret_cast<uint4*>(&Bs[r][8 * c]) = *reinterpret_cast<const uint4*>(wsrc + static_cast<size_t>(c0 + r) * DK + 8 * c);
    }
    __syncthreads();
    const int wm = warp & 3, wn = warp >> 2;
    float acc[4][4] = {};
    for (int ks = 0; ks < 8; ++ks)
    {
        uint32_t af[4];
        const int r0 = wm * 16 + gi, k0 = 16 * ks + 2 * ti;
        af[0] = *reinterpret_cast<const uint32_t*>(&As[r0][k0]);
        af[1] = *reinterpret_cast<const uint32_t*>(&As[r0 + 8][k0]);
        af[2] = *reinterpret_cast<const uint32_t*>(&As[r0][k0 + 8]);
        af[3] = *reinterpret_cast<const uint32_t*>(&As[r0 + 8][k0 + 8]);
        for (int nt = 0; nt < 4; ++nt)
        {
            const int n = wn * 32 + nt * 8 + gi;
            mma16816(acc[nt], af, *reinterpret_cast<const uint32_t*>(&Bs[n][k0]), *reinterpret_cast<const uint32_t*>(&Bs[n][k0 + 8]));
        }
    }
    uint16_t* dst = g.pl.p[mat ? Z16 : G16];
    for (int nt = 0; nt < 4; ++nt)
    {
        const int c = c0 + wn * 32 + nt * 8 + 2 * ti;
        const int ra = m0 + wm * 16 + gi, rb = ra + 8;
        if (ra < M) *reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(ra) * dl + c) = pack_bf16(acc[nt][0], acc[nt][1]);
        if (rb < M) *reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(rb) * dl + c) = pack_bf16(acc[nt][2], acc[nt][3]);
    }
}

// ------------------------------------------------------------------------------------------------------ GZF
// GR8_GLM53F_PF_KDA_GZF: no gz16 pass and no G16 / Z16 planes. K1 builds its tile's G16 values and k_out16z its tile's Z16 values
// with gz_tile: gz16's A conversion (P fp32 -> pack_bf16, rows >= M zero), gz16's B fragments (f_b / g_b [channel][k],
// read straight from global), the same mma16816 calls in the same k order per output and the same pack_bf16: the bits
// gz16 stores. Tile = rows m0 .. m0 + 15 x head h's 128 channels; NW warps split its 16 n8 tiles.
template <int NW>
__device__ __forceinline__ void gz_tile(const Args& a, int mat, int m0, int h, uint16_t (*As)[GZ_PAD], uint16_t (*dst)[GZ_PAD], int t)
{
    const int ld = a.pl().ld(), M = a.M, col = mat ? a.pl().ga() : a.pl().fa();
    const uint16_t* wsrc = (mat ? a.gb : a.fb) + static_cast<size_t>(h) * DK * DK;
    for (int q = t; q < 16 * 32; q += NW * 32)
    {
        const int r = q / 32, c4 = q % 32;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m0 + r < M) v = *reinterpret_cast<const float4*>(a.P + static_cast<size_t>(m0 + r) * ld + col + 4 * c4);
        uint2 w;
        w.x = pack_bf16(v.x, v.y); w.y = pack_bf16(v.z, v.w);
        *reinterpret_cast<uint2*>(&As[r][4 * c4]) = w;
    }
    __syncthreads();
    constexpr int TPW = 16 / NW;
    const int warp = t / 32, lane = t % 32, gi = lane >> 2, ti = lane & 3;
    float acc[TPW][4] = {};
    for (int ks = 0; ks < 8; ++ks)
    {
        uint32_t af[4];
        const int k0 = 16 * ks + 2 * ti;
        af[0] = *reinterpret_cast<const uint32_t*>(&As[gi][k0]);
        af[1] = *reinterpret_cast<const uint32_t*>(&As[gi + 8][k0]);
        af[2] = *reinterpret_cast<const uint32_t*>(&As[gi][k0 + 8]);
        af[3] = *reinterpret_cast<const uint32_t*>(&As[gi + 8][k0 + 8]);
#pragma unroll
        for (int j = 0; j < TPW; ++j)
        {
            const uint16_t* wr = wsrc + static_cast<size_t>((warp * TPW + j) * 8 + gi) * DK + k0;
            mma16816(acc[j], af, __ldg(reinterpret_cast<const unsigned*>(wr)), __ldg(reinterpret_cast<const unsigned*>(wr + 8)));
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j)
    {
        const int c = (warp * TPW + j) * 8 + 2 * ti;
        *reinterpret_cast<uint32_t*>(&dst[gi][c]) = pack_bf16(acc[j][0], acc[j][1]);
        *reinterpret_cast<uint32_t*>(&dst[gi + 8][c]) = pack_bf16(acc[j][2], acc[j][3]);
    }
    __syncthreads();
}
// GZF out16: CTA (16-row tile, head), 8 warps: the Z16 tile (gz_tile), then rows i and i + 8 per warp through out16_row
__global__ void __launch_bounds__(256) k_out16z(G9Args g)
{
    __shared__ __align__(16) uint16_t As[16][GZ_PAD], Zs[16][GZ_PAD];
    const int m0 = blockIdx.x * 16, h = blockIdx.y, t = threadIdx.x, warp = t / 32, lane = t % 32;
    gz_tile<8>(g.a, 1, m0, h, As, Zs, t);
    for (int i = warp; i < 16; i += 8)
        if (m0 + i < g.a.M) out16_row(g, m0 + i, h, lane, *reinterpret_cast<const uint2*>(&Zs[i][lane * 4]));
}

// ------------------------------------------------------------------------------------------------------ K1
// CTA (16-row tile, head), thread = channel d. Gate, fp32 cumsum down the 16 rows, pivot p = gc[7], the decay factors
// (__expf = ex2.approx.ftz), bf16 operands. Warps: w & 1 selects L (kdp kip^T) or Mqk (qdp kip^T), w >> 1 the n8 half.
// Rows >= M: kn = qn = 0, beta = 0, g = 0 (loads predicated, never dereferenced). Writes kd, qd, kr rows < M and the
// full U block (pad rows are zeros). The "gt" slot of the block holds e^gt, which is exp(gc[15]) already in a register.
// GZF: the G16 tile first (gz_tile, A in kdp, the values in qdp; thread d reads only column d of qdp, all of it before
// it writes qdp[.][d], so the second loop may overwrite it)
template <bool GZF = false>
__global__ void __launch_bounds__(128) k_k1(G9Args g, Tape tp)
{
    __shared__ __align__(16) uint16_t kdp[16][GZ_PAD], qdp[16][GZ_PAD], kip[16][GZ_PAD];
    __shared__ float Ls[16][17], Ms[16][17], beta_s[16];
    __shared__ __align__(16) uint16_t invs[16][16], mqs[16][16];
    __shared__ __align__(16) float egs[128];
    const Args& a = g.a;
    const int dl = a.dl(), ld = a.pl().ld(), M = a.M, h = blockIdx.y, tile = blockIdx.x, d = threadIdx.x, ch = h * DK + d, m0 = tile * 16;
    const int n = M - m0 < 16 ? M - m0 : 16;
    if constexpr (GZF) gz_tile<4>(a, 0, m0, h, kdp, qdp, d);
    if (d < 16) beta_s[d] = d < n ? sigm(a.P[static_cast<size_t>(m0 + d) * ld + a.pl().b() + a.h0 + h]) : 0.f;
    const float ea = expf(a.A_log[h]), dtb = a.dt_bias[ch];
    float kn[16], qn[16], gc[16];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const size_t o = static_cast<size_t>(m0 + i) * dl + ch;
        const bool v = i < n;
        kn[i] = v ? bf2f(g.pl.p[KN][o]) : 0.f;
        qn[i] = v ? bf2f(g.pl.p[QN][o]) * QSCALE : 0.f;
        acc += v ? GATE_LO * sigm(ea * (bf2f(GZF ? qdp[i][d] : g.pl.p[G16][o]) + dtb)) : 0.f;
        gc[i] = acc;
    }
    const float gt = gc[15], pv = gc[7];
    (void)gt;
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const float ep = __expf(gc[i] - pv), en = __expf(pv - gc[i]), eg = __expf(gc[i]), er = __expf(gc[15] - gc[i]);
        kdp[i][d] = f2bf_rn(kn[i] * ep); qdp[i][d] = f2bf_rn(qn[i] * ep); kip[i][d] = f2bf_rn(kn[i] * en);
        if (i < n)
        {
            const size_t o = static_cast<size_t>(m0 + i) * dl + ch;
            tp.kd[o] = f2bf_rn(kn[i] * eg); tp.qd[o] = f2bf_rn(qn[i] * eg); tp.kr[o] = f2bf_rn(kn[i] * er);
        }
        if (i == 15) egs[d] = eg;
    }
    __syncthreads();
    {
        const int warp = d / 32, lane = d % 32, gi = lane >> 2, ti = lane & 3, mat = warp & 1, nh8 = warp >> 1;
        uint16_t (*A)[GZ_PAD] = mat ? qdp : kdp;
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        for (int ks = 0; ks < 8; ++ks)
        {
            const int k0 = 16 * ks + 2 * ti, nb = nh8 * 8 + gi;
            uint32_t af[4];
            af[0] = *reinterpret_cast<const uint32_t*>(&A[gi][k0]);
            af[1] = *reinterpret_cast<const uint32_t*>(&A[gi + 8][k0]);
            af[2] = *reinterpret_cast<const uint32_t*>(&A[gi][k0 + 8]);
            af[3] = *reinterpret_cast<const uint32_t*>(&A[gi + 8][k0 + 8]);
            mma16816(c, af, *reinterpret_cast<const uint32_t*>(&kip[nb][k0]), *reinterpret_cast<const uint32_t*>(&kip[nb][k0 + 8]));
        }
        float(*D)[17] = mat ? Ms : Ls;
        const int j = nh8 * 8 + 2 * ti;
        D[gi][j] = c[0]; D[gi][j + 1] = c[1]; D[gi + 8][j] = c[2]; D[gi + 8][j + 1] = c[3];
    }
    __syncthreads();
    for (int e = d; e < 256; e += 128)
    {
        const int i = e / 16, j = e % 16;
        Ls[i][j] = j < i ? beta_s[i] * Ls[i][j] : 0.f;
        mqs[i][j] = f2bf_rn(j <= i ? Ms[i][j] : 0.f);
    }
    __syncthreads();
    if (d < 16)   // column d of (I + L)^-1 by forward substitution, fp32, then diag(beta)
    {
        float X[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            float x = i == d ? 1.f : 0.f;
#pragma unroll
            for (int k = 0; k < i; ++k) x -= Ls[i][k] * X[k];
            X[i] = x;
            invs[i][d] = i >= d ? f2bf_rn(x * beta_s[d]) : static_cast<uint16_t>(0);
        }
    }
    __syncthreads();
    uint8_t* blk = tp.U + static_cast<size_t>(h * tp.NT + tile) * U_BLOCK;
    reinterpret_cast<uint32_t*>(blk)[d] = reinterpret_cast<const uint32_t*>(invs)[d];
    reinterpret_cast<uint32_t*>(blk + 512)[d] = reinterpret_cast<const uint32_t*>(mqs)[d];
    reinterpret_cast<float*>(blk + 1024)[d] = egs[d];
}

// ------------------------------------------------------------------------------------------------------ K2
// CTA (v slice vs of VSW = 128 / VS columns, head h): 8 / VS math warps own 16 v rows each of the transposed state
// S^T [v][d] (fp32 mma accumulators, 64 regs/thread), one producer warp streams the chunk tape through a 3-stage cp.async
// ring (named barriers: full = 1 + s, empty = 4 + s; no smem for sync). Per chunk and warp, with bf16(S^T) as the A fragment:
//   X = S^T kd^T, Y = S^T qd^T;  r = v - X;  U^T = r INVb^T;  O^T = Y + U^T Mqk^T;  S^T = S^T e^gt + U^T kr.
// kd, qd, kr stages are [16][128] bf16 with the 16 B chunk index XORed with (row & 7) (conflict-free fragment loads
// without padding); kr is read with ldmatrix.trans (B operand with k = row).
template <int VS> struct K2Cfg
{
    static constexpr int NW = 8 / VS, VSW = 128 / VS, THREADS = 32 * (NW + 1);
    static constexpr int KD = 0, QD = 4096, KR = 8192, V = 12288, INV = V + 16 * VSW * 2, MQK = INV + 512, EG = MQK + 512, STAGE = EG + 512;
    static constexpr int SMEM = 3 * STAGE + 128;   // + 128 B to align the ring base by hand
};

template <int VS>
__global__ void __launch_bounds__(K2Cfg<VS>::THREADS, VS == 1 ? 1 : 2) k_k2(G9Args g, Tape tp)
{
    using C = K2Cfg<VS>;
    extern __shared__ uint8_t ring_raw[];   // default alignment: an __align__(128) extern would pad the static smem of every kernel in the TU
    uint8_t* const ring = ring_raw + ((128u - (smem_u32(ring_raw) & 127u)) & 127u);
    const Args& a = g.a;
    const int dl = a.dl(), M = a.M, NT = tp.NT, h = blockIdx.y, vs = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr int NB = C::THREADS;
    const uint32_t ring_s = smem_u32(ring);
    if (warp == C::NW)   // producer
    {
        auto issue = [&](int tc) {
            const uint32_t sb = ring_s + (tc % 3) * C::STAGE;
            const int m0 = tc * 16;
            for (int x = 0; x < 3; ++x)
            {
                const uint16_t* src = x == 0 ? tp.kd : x == 1 ? tp.qd : tp.kr;
                for (int k = 0; k < 8; ++k)
                {
                    const int pc = lane + 32 * k, row = pc >> 4, cj = pc & 15, m = m0 + row;
                    cp_async16(sb + x * 4096 + row * 256 + ((cj ^ (row & 7)) << 4), src + static_cast<size_t>(m < M ? m : 0) * dl + h * DK + cj * 8, m < M);
                }
            }
            for (int pc = lane; pc < 2 * C::VSW; pc += 32)
            {
                const int row = pc / (C::VSW / 8), p = pc % (C::VSW / 8), m = m0 + row;
                cp_async16(sb + C::V + row * (2 * C::VSW) + p * 16,
                           g.pl.p[VV] + static_cast<size_t>(m < M ? m : 0) * dl + h * DK + vs * C::VSW + p * 8, m < M);
            }
            const uint8_t* blk = tp.U + static_cast<size_t>(h * NT + tc) * U_BLOCK;
            cp_async16(sb + C::INV + lane * 16, blk + lane * 16, true);
            cp_async16(sb + C::MQK + lane * 16, blk + 512 + lane * 16, true);
            cp_async16(sb + C::EG + lane * 16, blk + 1024 + lane * 16, true);
        };
        for (int p = 0; p < (NT < 2 ? NT : 2); ++p) { issue(p); cp_async_commit(); }
        for (int t = 0; t < NT; ++t)
        {
            if (t + 1 < NT) cp_async_wait<1>(); else cp_async_wait<0>();
            __threadfence_block();
            bar_arrive(1 + t % 3, NB);
            if (t + 2 < NT)
            {
                if (t >= 1) bar_sync(4 + (t + 2) % 3, NB);   // consumers are done with chunk t - 1 (same stage)
                issue(t + 2);
                cp_async_commit();
            }
        }
        return;
    }
    const int gi = lane >> 2, ti = lane & 3;
    const int vrow0 = vs * C::VSW + warp * 16;
    float* Sg = a.S + a.s_off(a.row_slot[0], h);
    float S[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
    {
        const int dd = nt * 8 + 2 * ti;
        S[nt][0] = Sg[static_cast<size_t>(dd) * DV + vrow0 + gi];
        S[nt][1] = Sg[static_cast<size_t>(dd + 1) * DV + vrow0 + gi];
        S[nt][2] = Sg[static_cast<size_t>(dd) * DV + vrow0 + gi + 8];
        S[nt][3] = Sg[static_cast<size_t>(dd + 1) * DV + vrow0 + gi + 8];
    }
    for (int t = 0; t < NT; ++t)
    {
        const int s = t % 3;
        const uint8_t* sb = ring + s * C::STAGE;
        bar_sync(1 + s, NB);
        uint32_t Af[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
        {
            Af[ks][0] = pack_bf16(S[2 * ks][0], S[2 * ks][1]);
            Af[ks][1] = pack_bf16(S[2 * ks][2], S[2 * ks][3]);
            Af[ks][2] = pack_bf16(S[2 * ks + 1][0], S[2 * ks + 1][1]);
            Af[ks][3] = pack_bf16(S[2 * ks + 1][2], S[2 * ks + 1][3]);
        }
        float X[2][4] = {}, Y[2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                const int row = nt * 8 + gi;
                const int o0 = row * 256 + (((2 * ks) ^ gi) << 4) + 4 * ti, o1 = row * 256 + (((2 * ks + 1) ^ gi) << 4) + 4 * ti;
                mma16816(X[nt], Af[ks], lds32(sb + C::KD + o0), lds32(sb + C::KD + o1));
                mma16816(Y[nt], Af[ks], lds32(sb + C::QD + o0), lds32(sb + C::QD + o1));
            }
        // r = v - X, as the A fragment of the 16 x 16 products
        uint32_t Ar[4];
        {
            float r[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const int i = nt * 8 + 2 * ti + (e & 1), vl = warp * 16 + gi + (e >> 1) * 8;
                    r[nt][e] = bf2f(*reinterpret_cast<const uint16_t*>(sb + C::V + i * (2 * C::VSW) + vl * 2)) - X[nt][e];
                }
            Ar[0] = pack_bf16(r[0][0], r[0][1]); Ar[1] = pack_bf16(r[0][2], r[0][3]);
            Ar[2] = pack_bf16(r[1][0], r[1][1]); Ar[3] = pack_bf16(r[1][2], r[1][3]);
        }
        float U[2][4] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            const int o = (nt * 8 + gi) * 32 + 4 * ti;
            mma16816(U[nt], Ar, lds32(sb + C::INV + o), lds32(sb + C::INV + o + 16));
        }
        uint32_t Au[4];
        Au[0] = pack_bf16(U[0][0], U[0][1]); Au[1] = pack_bf16(U[0][2], U[0][3]);
        Au[2] = pack_bf16(U[1][0], U[1][1]); Au[3] = pack_bf16(U[1][2], U[1][3]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            const int o = (nt * 8 + gi) * 32 + 4 * ti;
            mma16816(Y[nt], Au, lds32(sb + C::MQK + o), lds32(sb + C::MQK + o + 16));   // Y is now O^T
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int m = t * 16 + nt * 8 + 2 * ti + (e & 1), v = vrow0 + gi + (e >> 1) * 8;
                if (m < M) g.pl.p[O16][static_cast<size_t>(m) * dl + h * DK + v] = f2bf_rn(Y[nt][e]);
            }
        // S^T = S^T e^gt + U^T kr
#pragma unroll
        for (int nt = 0; nt < 16; nt += 2)
        {
            uint32_t b[4];
            const int mi = lane >> 3, rr = lane & 7, row = (mi & 1) * 8 + rr, tile = nt + (mi >> 1);
            ldmatrix_x4_trans(b, ring_s + s * C::STAGE + C::KR + row * 256 + ((tile ^ rr) << 4));
#pragma unroll
            for (int q = 0; q < 2; ++q)
            {
                const float2 e = *reinterpret_cast<const float2*>(sb + C::EG + ((nt + q) * 8 + 2 * ti) * 4);
                S[nt + q][0] *= e.x; S[nt + q][1] *= e.y; S[nt + q][2] *= e.x; S[nt + q][3] *= e.y;
                mma16816(S[nt + q], Au, b[2 * q], b[2 * q + 1]);
            }
        }
        if (t + 3 < NT) bar_arrive(4 + s, NB);
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
    {
        const int dd = nt * 8 + 2 * ti;
        Sg[static_cast<size_t>(dd) * DV + vrow0 + gi] = S[nt][0];
        Sg[static_cast<size_t>(dd + 1) * DV + vrow0 + gi] = S[nt][1];
        Sg[static_cast<size_t>(dd) * DV + vrow0 + gi + 8] = S[nt][2];
        Sg[static_cast<size_t>(dd + 1) * DV + vrow0 + gi + 8] = S[nt][3];
    }
}

template <int VS> inline void launch_k2(const G9Args& g, const Tape& tp, cudaStream_t st)
{
    using C = K2Cfg<VS>;
    static const cudaError_t attr = cudaFuncSetAttribute(k_k2<VS>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    (void)attr;
    k_k2<VS><<<dim3(VS, g.a.nh), C::THREADS, C::SMEM, st>>>(g, tp);
}

// ------------------------------------------------------------------------------------------------------ ON path
// conv16 -> gz16 -> K1 -> K2 -> out16 on projections already in a.P; scratch ps (rows = MP) holds the tape, a.C the six planes.
// Returns an error string (nothing launched) or nullptr. VS in {1, 2, 4}.
// GR8_GLM53F_PF_KDA_GZF (default 1, read once; =0: the gz16 pass and the G16 / Z16 planes): K1 and k_out16z build the G16 / Z16
// tiles (gz_tile): the same bits without the planes' DRAM round trip (gzf < 0: the env; 0 / 1: forced, the tests).
// kda/tests/kda_g9_gzf_test.cu: byte-equal, 12 cases. TP3 8K A/B: prefill 2592 -> 2627 tok/s, rank-0 8K -39 ms, T=0
// hashes unchanged.
inline int gzf_env() { return knob::glm53f_pf_kda_gzf(); }
inline const char* launch(const Args& a, const KdaPrefillScratch& ps, int VS, cudaStream_t st, int gzf = -1)
{
    if (gzf < 0) gzf = gzf_env();
    const int NT = (a.M + CH16 - 1) / CH16;
    if (a.M < 1 || a.M > ps.rows) return "kda::g9::launch: M out of range of the scratch";
    if (3 * NT > ps.rows) return "kda::g9::launch: 3 * ceil(M / 16) > MP (U block capacity)";
    if (a.dl() % 128 || a.nh > a.nhmax) return "kda::g9::launch: dl or nh";
    if (VS != 1 && VS != 2 && VS != 4) return "kda::g9::launch: VS must be 1, 2 or 4";
    const G9Args g{a, planes(a, ps.rows)};
    Tape tp;
    tp.kd = reinterpret_cast<uint16_t*>(ps.QE); tp.qd = reinterpret_cast<uint16_t*>(ps.KD); tp.kr = reinterpret_cast<uint16_t*>(ps.W);
    tp.U = reinterpret_cast<uint8_t*>(ps.U); tp.NT = NT;
    launch_conv16(g, st);
    if (gzf) k_k1<true><<<dim3(NT, a.nh), 128, 0, st>>>(g, tp);
    else
    {
        k_gz16<<<dim3((a.M + 63) / 64, a.dl() / 64, 2), 256, 0, st>>>(g);
        k_k1<false><<<dim3(NT, a.nh), 128, 0, st>>>(g, tp);
    }
    if (VS == 1) launch_k2<1>(g, tp, st);
    else if (VS == 2) launch_k2<2>(g, tp, st);
    else launch_k2<4>(g, tp, st);
    if (gzf) k_out16z<<<dim3(NT, a.nh), 256, 0, st>>>(g);
    else launch_out16(g, st);
    return nullptr;
}
}   // namespace g9
}   // namespace kda
