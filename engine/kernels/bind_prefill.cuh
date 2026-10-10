// g6 prefill binder: one rank's pack tensors of one layer -> kernel arguments + launch helpers for the
// prefill path, M >= 256 rows (M3b). Same style as bind.cuh (plain tensor struct in, launch helpers out); all
// buffers are DEVICE memory. Not meant for a CUDA graph (the expert grids are sized for the worst case of M).
//
//   bindp::Tensors T = bindp::layer_tensors(l, find);             find("L003.q_a.mx.codes") -> bind::Tensor
//   bindp::Scratch sc = bindp::scratch_at(block, Mmax);           block = bindp::scratch_bytes(Mmax) of device memory
//   bindp::Layer L;  const char* err = bindp::make_layer(T, M, sc, io, blas, L);      err == nullptr: ready
//
// Numeric path (the old build's large-M arm, FACTS-M0 "Small-M vs large-M"):
//   dense   W8A8: activations -> MXFP8 (E4M3 codes, one E8M0 scale per row per 32 k, scale = 2^ceil(log2(amax/448))),
//           weights = the pack's *.mx.codes / *.mx.scales read row-major as they are, block-scaled FP8 mma
//           (m16n8k32, scale_vec 1X on both operands), fp32 accumulate.
//   BF16    router, idx.wk, idx.weights_proj: cuBLAS gemm on BF16 rows (caller's handle), fp32 out.
//   experts TR3 tiles decoded in the kernel and rounded to E4M3 (no weight scale), activations MXFP8 as above:
//           y = sum over the rank's blocks b of  H128(f16(X2_b @ Wd_b)) * svh_d,
//           X2_b = mx8(f16(H128(act_b * suh_d))),  act_b = f16(silu(g) * u),
//           g = f16(H128(f16(mx8(f16(H128(x * suh_g))) @ Wg_b)) * svh_g),  u likewise;
//           route rows are grouped by expert (segments of <= 64 rows), per route yw = f16(route weight * y),
//           out = bf16(sum of the token's 8 yw in selection order + f16 shared expert output).
//
// Step order (each line = launches on the stream; `->` = where the result lands):
//   launch(L.xq_x, st)                        io.x -> MXFP8 rows
//   launch(L.q_a, st), launch(L.kv_a, st)     -> io.q_a fp16 [M][2048], io.kv_a fp16 [M][576]
//   launch(L.xb_x, st), launch(L.idx_k, st), launch(L.idx_w, st)     indexer layers -> io.idx_k [M][128], io.idx_w [M][32] fp16
//   ... q_a_norm ...
//   launch(L.xq_qa, st), launch(L.q_b, st), launch(L.idx_q, st)      -> io.q_b fp16 [M][H*256], io.idx_q fp16 [M][4096]
//   ... attention ...
//   launch(L.xq_attn, st), launch(L.o_proj, st)                      io.attn [M][H*256] -> io.attn_part BF16 [M][6144]
//   ... all-reduce 1, mlp_norm ...
//   launch(L.xq_x2, st), launch(L.gate, st), launch(L.up, st), launch(L.act, st), launch(L.xq_act, st), launch(L.down, st)
//                                             shared expert (-> fp16 scratch) or dense MLP (-> io.mlp_part BF16 [M][6144])
//   launch(L.xb_x2, st), launch(L.router, st) MoE -> io.logits fp32 [M][256]
//   launch_shared(L.exs, st)                  exlr8 v2 trellis shared expert (#65; then the five shared launches above are off)
//   launch_experts(L.ex, st)                  MoE: route, group, fc1, mid, fc2, final sum -> io.mlp_part BF16 [M][6144]
// A launch of a member with on == 0 does nothing. launch() of a Blas member returns the cuBLAS status.
// tr3.fc1 may be in the wide order (bind::widen_fc1 on the decode tensors of the same device memory): set
// Tensors::fc1_wide = bind::LayerTensors::fc1_wide before make_layer and fc1 reads that order. Same results bit for bit.
// The same for tr3.fc2: bind::widen_fc2, Tensors::fc2_wide = bind::LayerTensors::fc2_wide.
#pragma once
#include <cublas_v2.h>
#include "bind.cuh"
#include "gemm_x.cuh"

namespace bindp
{
using bind::Tensor;

constexpr int MB = 4;                 // 16-row blocks per CTA
constexpr int RS = MB * 16;           // rows per expert segment
constexpr int REC = 528;              // MXFP8 activation record of (16 rows, 32 k): [32 lanes][16 B] A fragments + 16 scale bytes
constexpr int NEXP = 256, TOPK = 8;

enum
{
    T_Q_A, T_KV_A, T_Q_B, T_O_PROJ, T_IDX_WQ_B, T_S_GATE, T_S_UP, T_S_DOWN, T_D_GATE, T_D_UP, T_D_DOWN, NMX,    // .mx.codes + .mx.scales
    T_IDX_WK = NMX, T_IDX_WPROJ, T_ROUTER, T_ROUTER_BIAS, T_TR3_FC1, T_TR3_FC2, T_TR3_V1, T_TR3_V2, T_TR3_V3, T_TR3_TAB,
    T_TR3_KMIX,                       // exlr8 v2 packs only (#65)
    T_TR3_SG, T_TR3_V1B, T_TR3_V3B,   // v2 layers with the trellis shared expert: hidden-side signs + block scales
    NT
};
static const char* const NAMES[NT] = {
    "q_a.mx", "kv_a.mx", "q_b.mx", "o_proj.mx", "idx.wq_b.mx", "shared.gate.mx", "shared.up.mx", "shared.down.mx",
    "dense.gate.mx", "dense.up.mx", "dense.down.mx",
    "idx.wk", "idx.weights_proj", "router.w", "router.bias", "tr3.fc1", "tr3.fc2", "tr3.v1", "tr3.v2", "tr3.v3", "tr3.table",
    "tr3.kmix", "tr3.sg", "tr3.v1b", "tr3.v3b"};

struct Tensors
{
    Tensor t[NT];                     // MXFP8 entries: the .codes tensor [N][K]
    Tensor s[NMX];                    //                the .scales tensor [N][K/32]
    int fc1_wide = 0;                 // set if bind::widen_fc1 ran on this tr3.fc1 (K3 experts in the wide order, tr3_wide.cuh)
    int fc2_wide = 0;                 // set if bind::widen_fc2 ran on this tr3.fc2
    int k2 = 1;                       // 2-bit units in the layer (bind::LayerTensors::k2 after bind::scan_widths, #65)
    int sh_trellis = 0;               // shared expert as trellis row 256 (bind::LayerTensors::sh_trellis, #65)
    int sh_fp8 = 0;                   // t/s[T_S_GATE, T_S_UP, T_S_DOWN] hold its FP8 copy (bind::make_shared_fp8)
    long long sh_tr[6] = {};          // tr3.table row 256
    int qo_cover = 0;                 // G6_PF_OWN=3: q_b / o_proj of this layer are prefilled from the QO pack (never from these tensors)
    int heads = 0;                    // attention head count, used in place of q_b.N / 256 when the q_b tensor is deduplicated away
    // dense trellis matrices (dtr.cuh, in dtr::MAT_NAMES order: q_b, o_proj, dense gate, up, down): their input and output
    // vectors. Set, the matrix's .mx codes are its FP8 tile copy (dtr::make_fp8, still rotated), and make_layer wraps the
    // GEMM in the rotations. nullptr: plain MXFP8 weights.
    const __half* tsuh[dtr::NMAT] = {};
    const __half* tsvh[dtr::NMAT] = {};
    // the matrices themselves (dtr::layer_mats): o_proj without an FP8 copy runs on its tiles (make_gemm_tr, k_gemm_tr)
    dtr::LayerMats tm;
};

template <class F> inline Tensors layer_tensors(int layer, F find)
{
    Tensors T;
    char n[80];
    for (int i = 0; i < NT; ++i)
    {
        snprintf(n, sizeof n, "L%03d.%s%s", layer, NAMES[i], i < NMX ? ".codes" : "");
        T.t[i] = find(n);
        if (i < NMX)
        {
            snprintf(n, sizeof n, "L%03d.%s.scales", layer, NAMES[i]);
            T.s[i] = find(n);
        }
    }
    for (int i = 0; i < dtr::NMAT; ++i)
    {
        snprintf(n, sizeof n, "L%03d.%s.trellis", layer, dtr::MAT_NAMES[i]);
        if (!find(n).p) continue;
        snprintf(n, sizeof n, "L%03d.%s.suh", layer, dtr::MAT_NAMES[i]);
        T.tsuh[i] = static_cast<const __half*>(find(n).p);
        snprintf(n, sizeof n, "L%03d.%s.svh", layer, dtr::MAT_NAMES[i]);
        T.tsvh[i] = static_cast<const __half*>(find(n).p);
    }
    if (dtr::layer_mats(layer, find, T.tm)) T.tm = dtr::LayerMats();   // make_layer reports a bad matrix by its missing codes
    return T;
}

// ------------------------------------------------------------------------------------------ MXFP8 helpers
// d += 2^(sa-127) 2^(sb-127) (A @ B), A 16 rows x 32 k, B 32 k x 8 cols, E4M3 bytes. Scale operands (probed on
// sm_120a): sa = the scale byte of row lane/4 (even lanes) or lane/4 + 8 (odd lanes), sb = the scale byte of col lane/4.
__device__ __forceinline__ void mma_mx8(float (&d)[4], const uint4& a, u32 b0, u32 b1, u32 sa, u32 sb)
{
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}

// E8M0 scale code of a block of fp16 values with absolute max amax, and inv = 2^-(code-127):
// v = fl32(amax * fl32(1/448)), code = 127 + exponent(v) + (mantissa(v) != 0); amax == 0 -> 127. Finite inputs.
__device__ __forceinline__ u32 mx_scale(float amax, float& inv)
{
    const u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f));
    int e = static_cast<int>(u >> 23) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    if (amax == 0.f) e = 0;
    inv = __uint_as_float(static_cast<u32>(127 - e) << 23);
    return static_cast<u32>(e + 127);
}
__device__ __forceinline__ u32 f2_e4m3x2(float a, float b)   // two floats -> two E4M3 bytes (a = low byte), RN, saturating
{
    u32 r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
// The same scale code plus the exponent itself (inv = 2^-e).
__device__ __forceinline__ u32 mx_scale_e(float amax, int& e)
{
    const u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f));
    e = static_cast<int>(u >> 23) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    if (amax == 0.f) e = 0;
    return static_cast<u32>(e + 127);
}
// f16x2 -> two E4M3 bytes (low half = low byte), RN, saturating, in one convert: the same bytes as gx::h2_e4m3x2
// (fp16 -> fp32 is exact, so rounding the fp16 value directly is the same rounding).
__device__ __forceinline__ u32 h2_e4m3x2_d(u32 h2)
{
    u32 r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "r"(h2));
    return r;
}
__device__ __forceinline__ u32 h2_pow2(int k) { return (static_cast<u32>(k + 15) << 10) | (static_cast<u32>(k + 15) << 26); }   // {2^k, 2^k} as f16x2, k in [-14, 15]
__device__ __forceinline__ u32 hmul2_u(u32 a, u32 b) { return as_u32(__hmul2(as_h2(a), as_h2(b))); }

// fp16 rows [M][ld] -> MXFP8 records dst [ceil(M/16)][K/32][REC], natural k order:
// lane (row%8)*4 + c, register half*2 + (row%16 >= 8), byte i = code of k = 32 s + 16 half + 4 c + i; byte 512 + row%16 =
// the row's scale. Rows >= M of the last block are zero. ld % 8 == 0, 16-byte aligned rows, K % 32 == 0.
struct Xq
{
    int on = 0;
    const __half* src = nullptr;
    int ld = 0, K = 0, M = 0;
    uint8_t* dst = nullptr;
    const __half* suh = nullptr;      // dense trellis input (dtr.cuh): records of H128(src * suh) per 128 block (k_xq8_rot)
};

#ifdef PF_XQ8C
// PF_XQ8C (served window arm E; default off; results/xq8): one CTA = one 16-row block x 16 consecutive k-groups s =
// 16 whole records, which are contiguous in dst: thread (rr = t / 16, s = s0 + t % 16) quantizes (row, s) exactly as before into a shared copy of the records, then the CTA
// writes them out as coalesced 16-B stores (the per-thread 4-B scatter into the records throttled the LSU queue).
__global__ void __launch_bounds__(256) k_xq8(Xq a)
{
    __shared__ __align__(16) uint8_t sr[16 * REC];
    const int S = a.K / 32, ST = (S + 15) / 16;
    const int rb = blockIdx.x / ST, s0 = (blockIdx.x % ST) * 16;
    const int sl = threadIdx.x & 15, rr = threadIdx.x >> 4, s = s0 + sl, row = rb * 16 + rr;
    if (s < S)
    {
        uint8_t* rec = sr + sl * REC;
        u32 c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, code = 127u;
        if (row < a.M)
        {
            const uint4* p = reinterpret_cast<const uint4*>(a.src + static_cast<size_t>(row) * a.ld + s * 32);
            const uint4 q[4] = {p[0], p[1], p[2], p[3]};
            float2 f[16];
            float amax = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                f[4 * j] = __half22float2(as_h2(q[j].x)); f[4 * j + 1] = __half22float2(as_h2(q[j].y));
                f[4 * j + 2] = __half22float2(as_h2(q[j].z)); f[4 * j + 3] = __half22float2(as_h2(q[j].w));
            }
            #pragma unroll
            for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
            float inv;
            code = mx_scale(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                c[j] = f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) | (f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc)
                *reinterpret_cast<u32*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
        rec[512 + rr] = static_cast<uint8_t>(code);
    }
    __syncthreads();
    const int nv = min(16, S - s0) * (REC / 16);
    uint4* d = reinterpret_cast<uint4*>(a.dst + (static_cast<size_t>(rb) * S + s0) * REC);
    const uint4* v = reinterpret_cast<const uint4*>(sr);
    for (int i = threadIdx.x; i < nv; i += 256) d[i] = v[i];
}
#else
__global__ void __launch_bounds__(256) k_xq8(Xq a)
{
    const int S = a.K / 32;
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    const int row = static_cast<int>(i / S), s = static_cast<int>(i % S);
    if (row >= (a.M + 15) / 16 * 16) return;
    const int rr = row & 15;
    uint8_t* rec = a.dst + (static_cast<size_t>(row >> 4) * S + s) * REC;
    u32 c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, code = 127u;
    if (row < a.M)
    {
        const uint4* p = reinterpret_cast<const uint4*>(a.src + static_cast<size_t>(row) * a.ld + s * 32);
        const uint4 q[4] = {p[0], p[1], p[2], p[3]};
        float2 f[16];
        float amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            f[4 * j] = __half22float2(as_h2(q[j].x)); f[4 * j + 1] = __half22float2(as_h2(q[j].y));
            f[4 * j + 2] = __half22float2(as_h2(q[j].z)); f[4 * j + 3] = __half22float2(as_h2(q[j].w));
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
        float inv;
        code = mx_scale(amax, inv);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            c[j] = f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) | (f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc)
            *reinterpret_cast<u32*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
    rec[512 + rr] = static_cast<uint8_t>(code);
}
#endif

// Xq::suh: x [M][ld] -> the k_xq8 records of H128(x * suh), quantized from fp32 (the rule of k_shx_rotq).
// grid ceil(ceil(M/16)*16 * K/128 / 8), block 256: warp = (row, 128 block)
__device__ __forceinline__ void shx_store_mx8(const float (&v)[4], int lane, int row, int M, int S, int s, uint8_t* x8);
__global__ void __launch_bounds__(256) k_xq8_rot(Xq a)
{
    const int lane = threadIdx.x & 31, nbk = a.K / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nbk, blk = w % nbk;
    if (row >= (a.M + 15) / 16 * 16) return;
    const int k0 = blk * 128 + lane * 4;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < a.M)
    {
        const uint2 xr = *reinterpret_cast<const uint2*>(a.src + static_cast<size_t>(row) * a.ld + k0);
        const uint2 sr = *reinterpret_cast<const uint2*>(a.suh + k0);
        const float2 x01 = __half22float2(as_h2(xr.x)), x23 = __half22float2(as_h2(xr.y));
        const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
        v[0] = __fmul_rn(x01.x, s01.x); v[1] = __fmul_rn(x01.y, s01.y); v[2] = __fmul_rn(x23.x, s23.x); v[3] = __fmul_rn(x23.y, s23.y);
    }
    had128(v, lane);
    shx_store_mx8(v, lane, row, a.M, a.K / 32, blk * 4 + (lane >> 3), a.dst);
}

inline void launch(const Xq& a, cudaStream_t st)
{
    if (!a.on) return;
    if (a.suh)
    {
        k_xq8_rot<<<static_cast<unsigned>(((a.M + 15) / 16 * 16 * static_cast<long long>(a.K / 128) + 7) / 8), 256, 0, st>>>(a);
        return;
    }
#ifdef PF_XQ8C
    const int S = a.K / 32;
    k_xq8<<<static_cast<unsigned>((a.M + 15) / 16 * ((S + 15) / 16)), 256, 0, st>>>(a);
#else
    const long long n = static_cast<long long>((a.M + 15) / 16 * 16) * (a.K / 32);
    k_xq8<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(a);
#endif
}

// ------------------------------------------------------------------------------------------ dense MXFP8 GEMM
// Y[M][N] = X8 @ W^T. CTA = (128 output cols, MB row blocks); X records through smem (cp.async), W codes and
// scales straight from the pack's row-major tensors (4 x 4-byte loads + 2 scale bytes per lane per k32 step).
// G6_PF_RSDIRECT (prefill.cuh): an [M][ld] BF16 output written straight into the reduce-scatter send slot instead of a
// staging buffer + k_put_rs: row j of owner p = (j + 1) * nr - 1) / M lands at slot(step) + p * ss + (j - p * M / nr) * ld,
// slot(step) = base + (step & 1) * pstride (the step parity is read from the device step cell). base == nullptr: off.
struct RsOut
{
    char* base = nullptr;
    size_t pstride = 0, ss = 0;
    const uint64_t* stepp = nullptr;
    int M = 0, nr = 0;
};
__device__ __forceinline__ uint16_t* rs_row(const RsOut& o, int row, int ld)
{
    const int p = ((row + 1) * o.nr - 1) / o.M;
    return reinterpret_cast<uint16_t*>(o.base + (*o.stepp & 1) * o.pstride + p * o.ss) + static_cast<size_t>(row - p * o.M / o.nr) * ld;
}

struct Gemm
{
    int on = 0;
    const uint8_t* x8 = nullptr;      // [ceil(M/16)][K/32][REC]
    const uint8_t* w = nullptr;       // codes  [N][K]
    const uint8_t* ws = nullptr;      // scales [N][K/32]
    int K = 0, N = 0, M = 0;
    void* y = nullptr;                // [M][ldy] fp16, or BF16 bits when bf16 == 1, or fp32 when bf16 == 2 (v2 only)
    int ldy = 0, bf16 = 0;
    // v2 pair launch (launch_pair): a second tensor of the same K on the same x8, its CTAs after this one's
    const uint8_t *w2 = nullptr, *ws2 = nullptr;
    int N2 = 0, ldy2 = 0, bf162 = 0;
    void* y2 = nullptr;
    const uint8_t* x8_2 = nullptr;    // the second tensor's records when they differ from x8 (the trellis shared expert's
                                      // gate and up inputs, #65); nullptr = x8
    // dense trellis matrices (dtr.cuh): pre writes x8 (rotated, Xq::suh) right before the GEMM; osvh (osvh2: the second
    // tensor's) applies H128 and svh to every row's 128 columns of a CTA in its epilogue (gemm_rot_store), N % 128 == 0
    Xq pre;
    const __half *osvh = nullptr, *osvh2 = nullptr;
    // tw set: the weights are a dense trellis matrix's K4 tiles (dtr.cuh addressing: tile (column block, k16 step s,
    // col tile c) at tw + block * tsB + (s / 8) * tsG + ((s % 8) * 8 + c) * 128), decoded in the GEMM (k_gemm_tr);
    // w / ws unused. The bits equal the GEMM on the matrix's FP8 copy (dtr::make_fp8).
    const uint8_t* tw = nullptr;
    long long tsB = 0, tsG = 0;
};

// Gemm::osvh: the CTA's 128 output columns of each row go through H128 and svh before the store. The input is the sums
// rounded to fp16, as the unrotated fp16 output would be, so the bits are those of an fp16 GEMM output rotated by a
// separate pass. The fp16 tile [rows][136] reuses the shared memory (34.8 KB at BM = 8, 17.4 KB at BM = 4, 69.6 KB at
// BM = 16 in k_gemm_tr). NW = the CTA's warps.
// store(row, col, v[4]) writes columns col .. col + 3 of the row.
template <int BM, int NT, int NW = 8, class St>
__device__ __forceinline__ void gemm_rot_store(const float (&acc)[4][NT][4], unsigned char* sm, const __half* svh, int c0, int rb0,
                                               int nb, int M, int wm, int colw, St store)
{
    constexpr int LD = 136;
    __half* tile = reinterpret_cast<__half*>(sm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, c = lane & 3, r = lane >> 2;
    __syncthreads();                                  // every warp is done with the stage buffers
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
    {
        const int m = wm * 4 + mb;
        if (m >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int rr = m * 16 + r, col = colw + nt * 8 + 2 * c;
            *reinterpret_cast<__half2*>(tile + rr * LD + col) = __floats2half2_rn(acc[mb][nt][0], acc[mb][nt][1]);
            *reinterpret_cast<__half2*>(tile + (rr + 8) * LD + col) = __floats2half2_rn(acc[mb][nt][2], acc[mb][nt][3]);
        }
    }
    __syncthreads();
    const uint2 sr = *reinterpret_cast<const uint2*>(svh + c0 + lane * 4);
    const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
    for (int rr = warp; rr < nb * 16; rr += NW)
    {
        const int row = rb0 * 16 + rr;
        if (row >= M) break;
        const uint2 tr = *reinterpret_cast<const uint2*>(tile + rr * LD + lane * 4);
        const float2 a = __half22float2(as_h2(tr.x)), b = __half22float2(as_h2(tr.y));
        float v[4] = {a.x, a.y, b.x, b.y};
        had128(v, lane);
        v[0] = __fmul_rn(v[0], s01.x); v[1] = __fmul_rn(v[1], s01.y); v[2] = __fmul_rn(v[2], s23.x); v[3] = __fmul_rn(v[3], s23.y);
        store(row, c0 + lane * 4, v);
    }
}
__device__ __forceinline__ uint2 pack_o4(const float (&v)[4], int bf16)
{
    return make_uint2(pack_o2(v[0], v[1], bf16), pack_o2(v[2], v[3], bf16));
}

constexpr int G_STAGES = 4, G_KS = 2, G_STAGE = G_KS * MB * REC;

__global__ void __launch_bounds__(256) k_gemm_mx8(Gemm g)
{
    __shared__ __align__(16) unsigned char sm[G_STAGES * G_STAGE];
    constexpr int CH = G_STAGE / 16;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = g.K / 32, NT16 = g.N / 16, RB = (g.M + 15) / 16;
    const int tile = blockIdx.x * 8 + warp, rb0 = blockIdx.y * MB;
    const int nb = min(MB, RB - rb0), NST = (S + G_KS - 1) / G_KS;
    const bool on = tile < NT16;

    auto loadA = [&](int slot, int st)
    {
        for (int c = t; c < CH; c += 256)
        {
            const int j = c / (MB * 33), rem = c % (MB * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G_KS + j < S)
                pf::cp16(sm + slot * G_STAGE + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G_KS + j) * REC + ch * 16);
        }
    };
    const int n0 = (on ? tile : 0) * 16 + (lane >> 2);
    const uint8_t* w0 = g.w + static_cast<size_t>(n0) * g.K + (lane & 3) * 4;
    const uint8_t* w1 = w0 + static_cast<size_t>(8) * g.K;
    const uint8_t* s0 = g.ws + static_cast<size_t>(n0) * S;
    const uint8_t* s1 = s0 + static_cast<size_t>(8) * S;
    auto loadB = [&](int ks, u32 (&v)[4], u32 (&sc)[2])
    {
        v[0] = v[1] = v[2] = v[3] = 0u; sc[0] = sc[1] = 127u;
        if (!on || ks >= S) return;
        v[0] = ld_nc(w0 + ks * 32); v[1] = ld_nc(w0 + ks * 32 + 16);
        v[2] = ld_nc(w1 + ks * 32); v[3] = ld_nc(w1 + ks * 32 + 16);
        sc[0] = s0[ks]; sc[1] = s1[ks];
    };

    #pragma unroll
    for (int p = 0; p < G_STAGES - 1; ++p)
    {
        if (p < NST) loadA(p, p);
        pf::cp_commit();
    }
    u32 nv[G_KS][4], ns[G_KS][2];
    #pragma unroll
    for (int j = 0; j < G_KS; ++j) loadB(j, nv[j], ns[j]);
    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<G_STAGES - 2>();
        __syncthreads();
        if (st + G_STAGES - 1 < NST) loadA((st + G_STAGES - 1) % G_STAGES, st + G_STAGES - 1);
        pf::cp_commit();
        u32 cv[G_KS][4], cs[G_KS][2];
        #pragma unroll
        for (int j = 0; j < G_KS; ++j)
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) cv[j][i] = nv[j][i];
            cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1];
        }
        if (st + 1 < NST)
        {
            #pragma unroll
            for (int j = 0; j < G_KS; ++j) loadB((st + 1) * G_KS + j, nv[j], ns[j]);
        }
        const unsigned char* a = sm + (st % G_STAGES) * G_STAGE;
        #pragma unroll
        for (int j = 0; j < G_KS; ++j)
        {
            if (!on || st * G_KS + j >= S) continue;
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb >= nb) continue;
                const unsigned char* rec = a + (j * MB + mb) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                mma_mx8(acc[mb][0], fa, cv[j][0], cv[j][1], sa, cs[j][0]);
                mma_mx8(acc[mb][1], fa, cv[j][2], cv[j][3], sa, cs[j][1]);
            }
        }
    }
    pf::cp_wait<0>();
    if (!on) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            if (mb >= nb) continue;
            const int col = tile * 16 + h * 8 + 2 * c, row = (rb0 + mb) * 16 + r;
            uint16_t* y = static_cast<uint16_t*>(g.y);
            if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][h][0], acc[mb][h][1], g.bf16);
            if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][h][2], acc[mb][h][3], g.bf16);
        }
}

// v2: the same GEMM with W codes through smem too (cp.async, 16-byte chunks, xor-swizzled rows read by ldmatrix)
// and wider warp tiles. CTA = (128 output cols, BM row blocks), 8 warps as (BM/4) x (8/(BM/4)); a warp owns 64 rows x
// NT*8 cols. Every output element gets the same mma_mx8 calls with the same operands in the same k order as
// k_gemm_mx8, so the results are the same bits. W scales: one byte per (col, k32) straight from global, prefetched
// one stage ahead. Rows / cols past M / N are neither loaded nor multiplied.
constexpr int G2_KS = 2, G2_ST = 3;
template <int BM> struct G2
{
    static constexpr int WM = BM / 4, WN = 8 / WM, WC = 128 / WN, NT = WC / 8;
    static constexpr int A = G2_KS * BM * REC, B = 128 * G2_KS * 32, STAGE = A + B, SMEM = G2_ST * STAGE;
};
__device__ __forceinline__ void ldsm_x4(const void* p, u32& r0, u32& r1, u32& r2, u32& r3)
{
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
// RS: rows go to the RS send slot (rs, G6_PF_RSDIRECT); a separate instantiation so the default kernel is unchanged
// F32: fp32 output (Gemm.bf16 == 2, G6_PF_OWN=3 o_proj); F32 = false instantiations are the served code, unchanged
template <int BM, bool RS = false, bool F32 = false> __global__ void __launch_bounds__(256) k_gemm_mx8b(Gemm g0, RsOut rs = RsOut())
{
    Gemm g = g0;
#ifdef PF_GEMMR
    // PF_GEMMR: row block fastest in launch order, so the CTAs that share a W column tile are resident together and W
    // is read from DRAM about once instead of once per wave (o_proj RS 48 x 6 at OWN=3: 327 -> ~139 MB). Same tiles,
    // same k order, same mma operands per output element: the same bits.
    const int lin = blockIdx.y * gridDim.x + blockIdx.x, by = lin % gridDim.y;
    int bx = lin / gridDim.y;
#else
    int bx = blockIdx.x;
#endif
    if (g0.w2)
    {
        const int nb1 = (g0.N + 127) / 128;
        if (bx >= nb1) { bx -= nb1; g.w = g0.w2; g.ws = g0.ws2; g.N = g0.N2; g.y = g0.y2; g.ldy = g0.ldy2; g.bf16 = g0.bf162; g.osvh = g0.osvh2; if (g0.x8_2) g.x8 = g0.x8_2; }
    }
    using C = G2<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS;
#ifdef PF_GEMMR
    const int c0 = bx * 128, rb0 = by * BM;
#else
    const int c0 = bx * 128, rb0 = blockIdx.y * BM;
#endif
    const int nb = min(BM, RB - rb0), ncols = min(128, g.N - c0);

    auto loadAB = [&](int slot, int st)
    {
        unsigned char* a = sm + slot * C::STAGE;
        for (int c = t; c < G2_KS * BM * 33; c += 256)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        unsigned char* b = a + C::A;
        for (int c = t; c < 128 * G2_KS * 2; c += 256)
        {
            const int n = c / (G2_KS * 2), ch = c % (G2_KS * 2);
            if (n < ncols && st * G2_KS + ch / 2 < S)
                pf::cp16(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16),
                         g.w + static_cast<size_t>(c0 + n) * g.K + st * (G2_KS * 32) + ch * 16);
        }
    };
    const int colw = wn * C::WC;                           // the warp's first col in the CTA
    const int nvalid = min(NT, max(0, (ncols - colw) / 8)); // n8 tiles of the warp inside N
    const uint8_t* sp = g.ws + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, u32 (&v)[G2_KS][NT])
    {
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                v[j][nt] = (nt < nvalid && st * G2_KS + j < S) ? static_cast<u32>(__ldg(sp + static_cast<size_t>(nt) * 8 * S + st * G2_KS + j)) : 127u;
    };

    #pragma unroll
    for (int p = 0; p < G2_ST - 1; ++p)
    {
        if (p < NST) loadAB(p, p);
        pf::cp_commit();
    }
    u32 ns[G2_KS][NT];
    loadS(0, ns);
    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<G2_ST - 2>();
        __syncthreads();
        if (st + G2_ST - 1 < NST) loadAB((st + G2_ST - 1) % G2_ST, st + G2_ST - 1);
        pf::cp_commit();
        u32 cs[G2_KS][NT];
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) cs[j][nt] = ns[j][nt];
        if (st + 1 < NST) loadS(st + 1, ns);
        const unsigned char* a = sm + (st % G2_ST) * C::STAGE;
        const unsigned char* b = a + C::A;
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            if (st * G2_KS + j >= S) continue;
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                const int m = wm * 4 + mb;
                if (m >= nb) continue;
                const unsigned char* rec = a + (j * BM + m) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    if (nt < nvalid) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, cs[j][nt]);
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
    uint16_t* y = static_cast<uint16_t*>(g.y);
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm * 4 + mb;
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
                if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
                if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
            }
        }
}

#ifdef PF_GEMMW
// PF_GEMMW: k_gemm_mx8b with a GW_CW = 256-col CTA (16 warps, 512 threads, 1 CTA/SM at 3 x 24.6 KB smem). Each warp
// keeps the BM=8 warp tile (64 rows x 32 cols, NT = 4) and the same mma_mx8 sequence per output element, so the bits
// are those of k_gemm_mx8b; the A records are staged once per 256 cols instead of once per 128 (fewer L2 re-reads).
constexpr int GW_CW = 256;
#ifndef PF_GEMMW_ST
#define PF_GEMMW_ST 3
#endif
constexpr int GW_ST = PF_GEMMW_ST;     // pipeline stages (4 x 24.6 KB = 98.6 KB fits the 99 KB opt-in)
template <int BM> struct G2W
{
    static constexpr int NW = 8 * GW_CW / 128, NTH = NW * 32, WM = BM / 4, WN = NW / WM, WC = GW_CW / WN, NT = WC / 8;
    static constexpr int A = G2_KS * BM * REC, B = GW_CW * G2_KS * 32, STAGE = A + B, SMEM = GW_ST * STAGE;
};
template <int BM> __global__ void __launch_bounds__(G2W<BM>::NTH) k_gemm_mx8w(Gemm g0)
{
    Gemm g = g0;
    int bx = blockIdx.x;
    if (g0.w2)
    {
        const int nb1 = (g0.N + GW_CW - 1) / GW_CW;
        if (bx >= nb1) { bx -= nb1; g.w = g0.w2; g.ws = g0.ws2; g.N = g0.N2; g.y = g0.y2; g.ldy = g0.ldy2; g.bf16 = g0.bf162; g.osvh = g0.osvh2; if (g0.x8_2) g.x8 = g0.x8_2; }
    }
    using C = G2W<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS;
    const int c0 = bx * GW_CW, rb0 = blockIdx.y * BM;
    const int nb = min(BM, RB - rb0), ncols = min(GW_CW, g.N - c0);

    auto loadAB = [&](int slot, int st)
    {
        unsigned char* a = sm + slot * C::STAGE;
        for (int c = t; c < G2_KS * BM * 33; c += C::NTH)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        unsigned char* b = a + C::A;
        for (int c = t; c < GW_CW * G2_KS * 2; c += C::NTH)
        {
            const int n = c / (G2_KS * 2), ch = c % (G2_KS * 2);
            if (n < ncols && st * G2_KS + ch / 2 < S)
                pf::cp16(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16),
                         g.w + static_cast<size_t>(c0 + n) * g.K + st * (G2_KS * 32) + ch * 16);
        }
    };
    const int colw = wn * C::WC;                           // the warp's first col in the CTA
    const int nvalid = min(NT, max(0, (ncols - colw) / 8)); // n8 tiles of the warp inside N
    const uint8_t* sp = g.ws + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, u32 (&v)[G2_KS][NT])
    {
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                v[j][nt] = (nt < nvalid && st * G2_KS + j < S) ? static_cast<u32>(__ldg(sp + static_cast<size_t>(nt) * 8 * S + st * G2_KS + j)) : 127u;
    };

    #pragma unroll
    for (int p = 0; p < GW_ST - 1; ++p)
    {
        if (p < NST) loadAB(p, p);
        pf::cp_commit();
    }
    u32 ns[G2_KS][NT];
    loadS(0, ns);
    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<GW_ST - 2>();
        __syncthreads();
        if (st + GW_ST - 1 < NST) loadAB((st + GW_ST - 1) % GW_ST, st + GW_ST - 1);
        pf::cp_commit();
        u32 cs[G2_KS][NT];
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) cs[j][nt] = ns[j][nt];
        if (st + 1 < NST) loadS(st + 1, ns);
        const unsigned char* a = sm + (st % GW_ST) * C::STAGE;
        const unsigned char* b = a + C::A;
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            if (st * G2_KS + j >= S) continue;
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                const int m = wm * 4 + mb;
                if (m >= nb) continue;
                const unsigned char* rec = a + (j * BM + m) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    if (nt < nvalid) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, cs[j][nt]);
            }
        }
    }
    pf::cp_wait<0>();

    const int c = lane & 3, r = lane >> 2;
    uint16_t* y = static_cast<uint16_t*>(g.y);
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm * 4 + mb;
            if (m >= nb || nt >= nvalid) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
            if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
        }
}

#endif

#ifdef PF_GEMMP
// PF_GEMMP: k_gemm_mx8b with the next fragments loaded before the current mma (software pipeline inside a stage).
// 1: next k-step's B fragments + next row block's A record; 2: all A and B fragments of the next k-step. Same mma_mx8
// calls with the same operands in the same k order per output element, so the bits are those of k_gemm_mx8b.
template <int BM> __global__ void __launch_bounds__(256) k_gemm_mx8p(Gemm g0)
{
    Gemm g = g0;
    int bx = blockIdx.x;
    if (g0.w2)
    {
        const int nb1 = (g0.N + 127) / 128;
        if (bx >= nb1) { bx -= nb1; g.w = g0.w2; g.ws = g0.ws2; g.N = g0.N2; g.y = g0.y2; g.ldy = g0.ldy2; g.bf16 = g0.bf162; g.osvh = g0.osvh2; if (g0.x8_2) g.x8 = g0.x8_2; }
    }
    using C = G2<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS;
    const int c0 = bx * 128, rb0 = blockIdx.y * BM;
    const int nb = min(BM, RB - rb0), ncols = min(128, g.N - c0);

    auto loadAB = [&](int slot, int st)
    {
        unsigned char* a = sm + slot * C::STAGE;
        for (int c = t; c < G2_KS * BM * 33; c += 256)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        unsigned char* b = a + C::A;
        for (int c = t; c < 128 * G2_KS * 2; c += 256)
        {
            const int n = c / (G2_KS * 2), ch = c % (G2_KS * 2);
            if (n < ncols && st * G2_KS + ch / 2 < S)
                pf::cp16(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16),
                         g.w + static_cast<size_t>(c0 + n) * g.K + st * (G2_KS * 32) + ch * 16);
        }
    };
    const int colw = wn * C::WC;                           // the warp's first col in the CTA
    const int nvalid = min(NT, max(0, (ncols - colw) / 8)); // n8 tiles of the warp inside N
    const uint8_t* sp = g.ws + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, u32 (&v)[G2_KS][NT])
    {
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                v[j][nt] = (nt < nvalid && st * G2_KS + j < S) ? static_cast<u32>(__ldg(sp + static_cast<size_t>(nt) * 8 * S + st * G2_KS + j)) : 127u;
    };

    #pragma unroll
    for (int p = 0; p < G2_ST - 1; ++p)
    {
        if (p < NST) loadAB(p, p);
        pf::cp_commit();
    }
    u32 ns[G2_KS][NT];
    loadS(0, ns);
    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<G2_ST - 2>();
        __syncthreads();
        if (st + G2_ST - 1 < NST) loadAB((st + G2_ST - 1) % G2_ST, st + G2_ST - 1);
        pf::cp_commit();
        u32 cs[G2_KS][NT];
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) cs[j][nt] = ns[j][nt];
        if (st + 1 < NST) loadS(st + 1, ns);
        const unsigned char* a = sm + (st % G2_ST) * C::STAGE;
        const unsigned char* b = a + C::A;
        // fragments of k-step j: B by ldmatrix, A record row + scale per row block; loaded one step ahead of the mma
        auto ldb = [&](int jj, u32 (&bf)[NT][2])
        {
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = jj * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
        };
        auto lda = [&](int jj, int mb, uint4& fa, u32& sa)
        {
            const int m = wm * 4 + mb;
            if (m >= nb) { fa = make_uint4(0u, 0u, 0u, 0u); sa = 0u; return; }
            const unsigned char* rec = a + (jj * BM + m) * REC;
            fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
            sa = rec[512 + srow];
        };
#if PF_GEMMP >= 2
        u32 bfr[2][NT][2];
        uint4 far[2][4];
        u32 sar[2][4];
        ldb(0, bfr[0]);
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb) lda(0, mb, far[0][mb], sar[0][mb]);
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            if (j + 1 < G2_KS && st * G2_KS + j + 1 < S)
            {
                ldb(j + 1, bfr[(j + 1) & 1]);
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb) lda(j + 1, mb, far[(j + 1) & 1][mb], sar[(j + 1) & 1][mb]);
            }
            if (st * G2_KS + j >= S) continue;
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                if (wm * 4 + mb >= nb) continue;
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    if (nt < nvalid) mma_mx8(acc[mb][nt], far[j & 1][mb], bfr[j & 1][nt][0], bfr[j & 1][nt][1], sar[j & 1][mb], cs[j][nt]);
            }
        }
#else
        u32 bfr[2][NT][2];
        uint4 fa[2];
        u32 sa[2];
        ldb(0, bfr[0]);
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            if (j + 1 < G2_KS && st * G2_KS + j + 1 < S) ldb(j + 1, bfr[(j + 1) & 1]);
            if (st * G2_KS + j >= S) continue;
            lda(j, 0, fa[0], sa[0]);
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                if (mb + 1 < 4) lda(j, mb + 1, fa[(mb + 1) & 1], sa[(mb + 1) & 1]);
                if (wm * 4 + mb >= nb) continue;
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    if (nt < nvalid) mma_mx8(acc[mb][nt], fa[mb & 1], bfr[j & 1][nt][0], bfr[j & 1][nt][1], sa[mb & 1], cs[j][nt]);
            }
        }
#endif
    }
    pf::cp_wait<0>();

    const int c = lane & 3, r = lane >> 2;
    uint16_t* y = static_cast<uint16_t*>(g.y);
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm * 4 + mb;
            if (m >= nb || nt >= nvalid) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
            if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
        }
}
#endif

#ifdef PF_GEMMF
// PF_GEMMF: k_gemm_mx8b with an unguarded mma loop for full tiles (nb == BM, ncols == 128; a CTA-uniform test): the
// per-mma guards made nvcc wrap each predicated QMMA in WARPSYNC / BSSY / BSYNC (results/peak/gemmsass). Same mma
// calls, operands and k order per output, so the bits are those of k_gemm_mx8b.
template <int BM> __global__ void __launch_bounds__(256) k_gemm_mx8f(Gemm g0)
{
    Gemm g = g0;
#ifdef PF_GEMMRF
    // PF_GEMMRF: owner-row launches only (gridDim.y <= 8): row block fastest in launch order, as PF_GEMMR in k_gemm_mx8b
    // (q_b 128 x 6 at OWN=3: W read once instead of once per row block, 205 -> ~46 MB). Full-M launches keep the raster.
    // Same tiles, same k order, same mma operands per output element: the same bits.
    const bool rr = gridDim.y <= 8;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x, by = rr ? lin % gridDim.y : blockIdx.y;
    int bx = rr ? lin / gridDim.y : blockIdx.x;
#else
    int bx = blockIdx.x;
#endif
    if (g0.w2)
    {
        const int nb1 = (g0.N + 127) / 128;
        if (bx >= nb1) { bx -= nb1; g.w = g0.w2; g.ws = g0.ws2; g.N = g0.N2; g.y = g0.y2; g.ldy = g0.ldy2; g.bf16 = g0.bf162; g.osvh = g0.osvh2; if (g0.x8_2) g.x8 = g0.x8_2; }
    }
    using C = G2<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS;
#ifdef PF_GEMMRF
    const int c0 = bx * 128, rb0 = by * BM;
#else
    const int c0 = bx * 128, rb0 = blockIdx.y * BM;
#endif
    const int nb = min(BM, RB - rb0), ncols = min(128, g.N - c0);
    const bool full = nb == BM && ncols == 128;

#if PF_GEMMF >= 2
    // full tiles: chunk offsets precomputed once (A: chunks t, t+256, t+512 < 2*BM*33; B: t, t+256 < 512), the stage
    // adds st * G2_KS records (A) / st * 64 bytes (B); only the K-end guard is left
    constexpr int CA = G2_KS * BM * 33;
    u32 sa_off[3], sb_off[2];
    size_t ga_off[3], gb_off[2];
    int ja[3], jb[2];
    if (full)
    {
        #pragma unroll
        for (int q = 0; q < 3; ++q)
        {
            const int c = t + q * 256, j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            sa_off[q] = c * 16; ja[q] = c < CA ? j : 1 << 20;
            ga_off[q] = (static_cast<size_t>(rb0 + mb) * S + j) * REC + ch * 16;
        }
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int c = t + q * 256, n = c / (G2_KS * 2), ch = c % (G2_KS * 2);
            sb_off[q] = n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16); jb[q] = ch / 2;
            gb_off[q] = static_cast<size_t>(c0 + n) * g.K + ch * 16;
        }
    }
#endif
    auto loadAB = [&](int slot, int st)
    {
#if PF_GEMMF >= 2
        if (full)
        {
            unsigned char* a = sm + slot * C::STAGE;
            #pragma unroll
            for (int q = 0; q < 3; ++q)
                if (st * G2_KS + ja[q] < S) pf::cp16(a + sa_off[q], g.x8 + ga_off[q] + static_cast<size_t>(st) * G2_KS * REC);
            #pragma unroll
            for (int q = 0; q < 2; ++q)
                if (st * G2_KS + jb[q] < S) pf::cp16(a + C::A + sb_off[q], g.w + gb_off[q] + st * (G2_KS * 32));
            return;
        }
#endif
        unsigned char* a = sm + slot * C::STAGE;
        for (int c = t; c < G2_KS * BM * 33; c += 256)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        unsigned char* b = a + C::A;
        for (int c = t; c < 128 * G2_KS * 2; c += 256)
        {
            const int n = c / (G2_KS * 2), ch = c % (G2_KS * 2);
            if (n < ncols && st * G2_KS + ch / 2 < S)
                pf::cp16(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16),
                         g.w + static_cast<size_t>(c0 + n) * g.K + st * (G2_KS * 32) + ch * 16);
        }
    };
    const int colw = wn * C::WC;                           // the warp's first col in the CTA
    const int nvalid = min(NT, max(0, (ncols - colw) / 8)); // n8 tiles of the warp inside N
    const uint8_t* sp = g.ws + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, u32 (&v)[G2_KS][NT])
    {
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                v[j][nt] = (nt < nvalid && st * G2_KS + j < S) ? static_cast<u32>(__ldg(sp + static_cast<size_t>(nt) * 8 * S + st * G2_KS + j)) : 127u;
    };

    #pragma unroll
    for (int p = 0; p < G2_ST - 1; ++p)
    {
        if (p < NST) loadAB(p, p);
        pf::cp_commit();
    }
    u32 ns[G2_KS][NT];
    loadS(0, ns);
    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<G2_ST - 2>();
        __syncthreads();
        if (st + G2_ST - 1 < NST) loadAB((st + G2_ST - 1) % G2_ST, st + G2_ST - 1);
        pf::cp_commit();
        u32 cs[G2_KS][NT];
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) cs[j][nt] = ns[j][nt];
        if (st + 1 < NST) loadS(st + 1, ns);
        const unsigned char* a = sm + (st % G2_ST) * C::STAGE;
        const unsigned char* b = a + C::A;
        #pragma unroll
        for (int j = 0; j < G2_KS; ++j)
        {
            if (st * G2_KS + j >= S) continue;
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            if (full)       // CTA-uniform: every row block and every warp's n8 tiles valid, no per-mma guard
            {
                #pragma unroll
                for (int mb = 0; mb < 4; ++mb)
                {
                    const unsigned char* rec = a + (j * BM + wm * 4 + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, cs[j][nt]);
                }
                continue;
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                const int m = wm * 4 + mb;
                if (m >= nb) continue;
                const unsigned char* rec = a + (j * BM + m) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    if (nt < nvalid) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, cs[j][nt]);
            }
        }
    }
    pf::cp_wait<0>();
    if (g.osvh)
    {
        gemm_rot_store<BM, NT>(acc, sm, g.osvh, c0, rb0, nb, g.M, wm, colw, [&](int row, int col, const float (&v)[4]) {
            *reinterpret_cast<uint2*>(static_cast<uint16_t*>(g.y) + static_cast<size_t>(row) * g.ldy + col) = pack_o4(v, g.bf16);
        });
        return;
    }

    const int c = lane & 3, r = lane >> 2;
    uint16_t* y = static_cast<uint16_t*>(g.y);
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            const int m = wm * 4 + mb;
            if (m >= nb || nt >= nvalid) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            if (row < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row) * g.ldy + col) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
            if (row + 8 < g.M) *reinterpret_cast<u32*>(y + static_cast<size_t>(row + 8) * g.ldy + col) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
        }
}
#endif

// ------------------------------------------------------------------------------------------ dense trellis GEMM
// k_gemm_tr: k_gemm_mx8b's tiling (CTA = 128 output cols x BM 16-row blocks, A records through a cp.async ring, ldsm B
// fragments, mma_mx8) with the weights read as a dense trellis matrix's K4 tiles (Gemm::tw): a stage brings the raw
// tiles of its 128 cols x 64 k (4 KB), and the warps decode them into sdec, the stage's E4M3 bytes at loadAB's
// swizzled [n][k] places. So the bits equal k_gemm_mx8b on the matrix's FP8 copy (dtr::make_fp8, scales 127).
// Overlap: stage st + 1 is decoded into the other sdec buffer while stage st's mma runs (one barrier per stage; the
// loads run ST - 2 stages ahead of the decode). Bigger CTAs decode less per row: BM = 16 (256 rows, 16 warps of 64 x
// 32, 512 threads) for large launches; BM = 4 and 8 keep k_gemm_mx8b's warp shapes.
template <int BM> struct GT
{
    static constexpr int WM = BM / 4, WN = BM >= 8 ? 4 : 8, NW = WM * WN, NTH = 32 * NW, WC = 128 / WN, NT = WC / 8;
    static constexpr int A = G2_KS * BM * REC, RAW = 32 * 128, STG = A + RAW, B = 128 * G2_KS * 32;
    static constexpr int ST = BM == 16 ? 4 : 3, SMEM = ST * STG + 2 * B;     // BM = 16: 98.3 KB, 1 CTA per SM
};
template <int BM, bool RS> __global__ void __launch_bounds__(GT<BM>::NTH) k_gemm_tr(Gemm g, RsOut rs)
{
    using T = GT<BM>;
    constexpr int NT = T::NT, ST = T::ST;
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int S = g.K / 32, RB = (g.M + 15) / 16, NST = (S + G2_KS - 1) / G2_KS, K16 = g.K / 16;
    const int bx = blockIdx.x, c0 = bx * 128, rb0 = blockIdx.y * BM;
    const int nb = min(BM, RB - rb0);
    unsigned char* sdec = sm + ST * T::STG;

    auto loadAB = [&](int st)
    {
        unsigned char* a = sm + (st % ST) * T::STG;
        for (int c = t; c < G2_KS * BM * 33; c += T::NTH)
        {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * G2_KS + j < S)
                pf::cp16(a + c * 16, g.x8 + (static_cast<size_t>(rb0 + mb) * S + st * G2_KS + j) * REC + ch * 16);
        }
        // tile ti = c / 8 of the stage: k16 step 4 st + ti / 8, column tile ti % 8; 16 B a thread
        for (int c = t; c < 256; c += T::NTH)
        {
            const int ti = c >> 3, ks = st * 4 + (ti >> 3);
            if (ks < K16)
                pf::cp16(a + T::A + c * 16, g.tw + bx * g.tsB + (ks >> 3) * g.tsG + ((ks & 7) * 8 + (ti & 7)) * 128 + (c & 7) * 16);
        }
    };
    const int r8 = lane >> 2, kc = 2 * (lane & 3);
    Tr3Lane<4, false> tl;             // fp16 codebook values: the one E4M3 rounding is the store's (h2_e4m3x2_d), same bytes
    tl.init(lane);
    auto at = [&](int n, int k) { return n * (G2_KS * 32) + (((k >> 4) ^ ((n >> 1) & 3)) << 4) + (k & 15); };
    // stage st's raw tiles -> sdec buffer st & 1. Warp w takes tiles w, w + NW, .. (unrolled, so the tiles' dependent
    // chains interleave); lane holds (col 16 c + l/4 (+8), k 16 j + 2 (l%4) (+8)) pairs.
    auto decode = [&](int st)
    {
        const unsigned char* raw = sm + (st % ST) * T::STG + T::A;
        unsigned char* d = sdec + (st & 1) * T::B;
        #pragma unroll
        for (int u = 0; u < 32 / T::NW; ++u)
        {
            const int ti = warp + u * T::NW, j = ti >> 3, c = ti & 7;
            if (st * 4 + j >= K16) continue;
            const u32* tile = reinterpret_cast<const u32*>(raw + ti * 128);
            u32 f0[2], f1[2];
            tl.decode(tile[tl.i0], tile[tl.i1], f0, f1);
            const int n = c * 16 + r8, k = j * 16 + kc;
            *reinterpret_cast<uint16_t*>(d + at(n, k)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[0]));
            *reinterpret_cast<uint16_t*>(d + at(n, k + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f0[1]));
            *reinterpret_cast<uint16_t*>(d + at(n + 8, k)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[0]));
            *reinterpret_cast<uint16_t*>(d + at(n + 8, k + 8)) = static_cast<uint16_t>(h2_e4m3x2_d(f1[1]));
        }
    };
    // loads of stages 0 .. ST - 2 (one commit group each), then stage 0 decoded
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
    const bool full = nb == BM;

    for (int st = 0; st < NST; ++st)
    {
        // stage st + 1 landed; sdec[st & 1] decoded; every warp past stage st - 1's mma (its sdec buffer and ring slot free)
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
            if (st * G2_KS + j >= S) continue;
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr)
            {
                const int n = colw + (pr * 2 + (mat >> 1)) * 8 + lr, ch = j * 2 + (mat & 1);
                ldsm_x4(b + n * (G2_KS * 32) + ((ch ^ ((n >> 1) & 3)) * 16), bf[2 * pr][0], bf[2 * pr][1], bf[2 * pr + 1][0], bf[2 * pr + 1][1]);
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb)
            {
                const int m = wm * 4 + mb;
                if (!full && m >= nb) continue;
                const unsigned char* rec = a + (j * BM + m) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma_mx8(acc[mb][nt], fa, bf[nt][0], bf[nt][1], sa, 127u);
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
            const int m = wm * 4 + mb;
            if (m >= nb) continue;
            const int col = c0 + colw + nt * 8 + 2 * c, row = (rb0 + m) * 16 + r;
            auto out = [&](int rw) { return RS ? rs_row(rs, rw, g.ldy) + col : static_cast<uint16_t*>(g.y) + static_cast<size_t>(rw) * g.ldy + col; };
            if (row < g.M) *reinterpret_cast<u32*>(out(row)) = pack_o2(acc[mb][nt][0], acc[mb][nt][1], g.bf16);
            if (row + 8 < g.M) *reinterpret_cast<u32*>(out(row + 8)) = pack_o2(acc[mb][nt][2], acc[mb][nt][3], g.bf16);
        }
}
template <int BM, bool RS> inline void launch_tr_bm(const Gemm& g, const RsOut& rs, cudaStream_t st)
{
    static bool attr = false;
    if (!attr)
    {
        cudaFuncSetAttribute(k_gemm_tr<BM, RS>, cudaFuncAttributeMaxDynamicSharedMemorySize, GT<BM>::SMEM);
        attr = true;
    }
    const int RB = (g.M + 15) / 16;
    k_gemm_tr<BM, RS><<<dim3((g.N + 127) / 128, (RB + BM - 1) / BM), GT<BM>::NTH, GT<BM>::SMEM, st>>>(g, rs);
}
// G6_PF_TR_BM (default 16): the row blocks of a large launch's CTA (8 or 16); small launches take 4, as k_gemm_mx8b
inline void launch_tr(const Gemm& g, const RsOut& rs, bool big, cudaStream_t st)
{
    static const int bmw = getenv("G6_PF_TR_BM") ? atoi(getenv("G6_PF_TR_BM")) : 16;
    const int bm = !big ? 4 : bmw == 8 ? 8 : 16;
    if (rs.base) { if (bm == 16) launch_tr_bm<16, true>(g, rs, st); else if (bm == 8) launch_tr_bm<8, true>(g, rs, st); else launch_tr_bm<4, true>(g, rs, st); }
    else if (bm == 16) launch_tr_bm<16, false>(g, rs, st);
    else if (bm == 8) launch_tr_bm<8, false>(g, rs, st);
    else launch_tr_bm<4, false>(g, rs, st);
}

inline void die_gemm_pair() { fprintf(stderr, "bindp: a paired Gemm needs the v2 kernel\n"); exit(1); }
// G6_GEMM: 1 = k_gemm_mx8 (v1), 2 = v2 with 128-row CTAs, 3 = v2 with 64-row CTAs, unset/0 = v2, row blocks by grid size
inline int gemm_mode()
{
    static const int m = [] { const char* e = getenv("G6_GEMM"); return e ? atoi(e) : 0; }();
    return m;
}
// rs.base set (G6_PF_RSDIRECT): y is unused, rows go to the RS send slot (unpaired v2 kernel only)
inline void launch(const Gemm& g, cudaStream_t st, int mode, const RsOut& rs = RsOut())
{
    if (!g.on) return;
    launch(g.pre, st);
#if defined(PF_GEMMW) || defined(PF_GEMMP)
    if (g.osvh || g.osvh2) { fprintf(stderr, "bindp: Gemm::osvh needs the v2 or PF_GEMMF kernel (not PF_GEMMW / PF_GEMMP)\n"); exit(1); }
#endif
    if ((g.osvh || g.osvh2 || g.tw) && mode == 1) { fprintf(stderr, "bindp: Gemm::osvh / tw need the v2 kernel (G6_GEMM != 1)\n"); exit(1); }
    const int RB = (g.M + 15) / 16, NB = (g.N + 127) / 128 + (g.w2 ? (g.N2 + 127) / 128 : 0);
    if (rs.base && (mode == 1 || g.w2)) { fprintf(stderr, "bindp: an RS-slot Gemm needs the unpaired v2 kernel\n"); exit(1); }
    const bool f32 = g.bf16 == 2;
    if (f32 && (mode == 1 || rs.base || (g.w2 && g.bf162 != 2))) { fprintf(stderr, "bindp: fp32 Gemm output needs the v2 kernel, unpaired or both fp32, no RS slot\n"); exit(1); }
    if (mode == 1)
    {
        if (g.w2) die_gemm_pair();
        const dim3 grid((g.N / 16 + 7) / 8, (RB + MB - 1) / MB);
        k_gemm_mx8<<<grid, 256, 0, st>>>(g);
        return;
    }
    static bool attr = false;
    if (!attr)
    {
        cudaFuncSetAttribute(k_gemm_mx8b<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8b<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8b<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8b<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8b<8, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM);
        cudaFuncSetAttribute(k_gemm_mx8b<4, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<4>::SMEM);
        attr = true;
    }
    static int nsm = 0;
    if (!nsm) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d); }
    const bool big = mode == 2 || (mode != 3 && NB * ((RB + 7) / 8) >= 2 * nsm);
    if (g.tw)                         // dense trellis tiles (k_gemm_tr): never paired, fp16 / BF16 out
    {
        if (g.w2 || f32) { fprintf(stderr, "bindp: a trellis Gemm is unpaired with fp16 / BF16 output\n"); exit(1); }
        launch_tr(g, rs, big, st);
        return;
    }
    if (rs.base)
    {
        if (big) k_gemm_mx8b<8, true><<<dim3(NB, (RB + 7) / 8), 256, G2<8>::SMEM, st>>>(g, rs);
        else k_gemm_mx8b<4, true><<<dim3(NB, (RB + 3) / 4), 256, G2<4>::SMEM, st>>>(g, rs);
        return;
    }
    if (f32)
    {
        if (big) k_gemm_mx8b<8, false, true><<<dim3(NB, (RB + 7) / 8), 256, G2<8>::SMEM, st>>>(g);
        else k_gemm_mx8b<4, false, true><<<dim3(NB, (RB + 3) / 4), 256, G2<4>::SMEM, st>>>(g);
        return;
    }
#ifdef PF_GEMMW
    static bool attrw = false;
    if (!attrw) { cudaFuncSetAttribute(k_gemm_mx8w<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2W<8>::SMEM); attrw = true; }
    const int NBW = (g.N + GW_CW - 1) / GW_CW + (g.w2 ? (g.N2 + GW_CW - 1) / GW_CW : 0);
    if (big) k_gemm_mx8w<8><<<dim3(NBW, (RB + 7) / 8), G2W<8>::NTH, G2W<8>::SMEM, st>>>(g);
#elif defined(PF_GEMMF)
    static bool attrf = false;
    if (!attrf) { cudaFuncSetAttribute(k_gemm_mx8f<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM); attrf = true; }
    if (big) k_gemm_mx8f<8><<<dim3(NB, (RB + 7) / 8), 256, G2<8>::SMEM, st>>>(g);
#elif defined(PF_GEMMP)
    static bool attrp = false;
    if (!attrp) { cudaFuncSetAttribute(k_gemm_mx8p<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, G2<8>::SMEM); attrp = true; }
    if (big) k_gemm_mx8p<8><<<dim3(NB, (RB + 7) / 8), 256, G2<8>::SMEM, st>>>(g);
#else
    if (big) k_gemm_mx8b<8><<<dim3(NB, (RB + 7) / 8), 256, G2<8>::SMEM, st>>>(g);
#endif
    else k_gemm_mx8b<4><<<dim3(NB, (RB + 3) / 4), 256, G2<4>::SMEM, st>>>(g);
}
inline void launch(const Gemm& g, cudaStream_t st, const RsOut& rs = RsOut()) { launch(g, st, gemm_mode(), rs); }
// a then b, or one v2 launch for both when they read the same records (same x8, K, M): each output element gets the
// same mma calls as in its own launch, so the bits do not change
inline void launch_pair(const Gemm& a, const Gemm& b, cudaStream_t st, int mode)
{
    if (mode == 1 || !a.on || !b.on || a.K != b.K || a.M != b.M || a.w2 || b.w2 || a.tw || b.tw || a.bf16 != b.bf16 && (a.bf16 == 2 || b.bf16 == 2))
    {
        launch(a, st, mode);
        launch(b, st, mode);
        return;
    }
    Gemm p = a;
    p.w2 = b.w; p.ws2 = b.ws; p.N2 = b.N; p.y2 = b.y; p.ldy2 = b.ldy; p.bf162 = b.bf16; p.osvh2 = b.osvh;
    p.x8_2 = b.x8 != a.x8 ? b.x8 : nullptr;
    launch(b.pre, st);
    launch(p, st, mode);
}
inline void launch_pair(const Gemm& a, const Gemm& b, cudaStream_t st) { launch_pair(a, b, st, gemm_mode()); }

// codes [N][K] + scales [N][K/32] -> GEMM of M rows from the records x8 into y [M][N]
inline const char* make_gemm(const Tensor& codes, const Tensor& scales, int M, const uint8_t* x8, void* y, int bf16, Gemm& g)
{
    g = Gemm();
    if (!codes.p) return nullptr;
    if (!scales.p || codes.nd != 2 || scales.nd != 2 || codes.shape[0] != scales.shape[0] || codes.shape[1] != scales.shape[1] * 32 ||
        codes.shape[0] % 16)
        return "MXFP8 tensor is not codes [N][K] + scales [N][K/32] with N % 16 == 0";
    g.on = 1;
    g.x8 = x8;
    g.w = static_cast<const uint8_t*>(codes.p); g.ws = static_cast<const uint8_t*>(scales.p);
    g.N = static_cast<int>(codes.shape[0]); g.K = static_cast<int>(codes.shape[1]); g.M = M;
    g.y = y; g.ldy = g.N; g.bf16 = bf16;
    return nullptr;
}

// a dense trellis matrix's tiles (dtr::Mat, K4) -> GEMM of M rows from the records x8 into y [M][N] (Gemm::tw)
inline const char* make_gemm_tr(const dtr::Mat& m, int M, const uint8_t* x8, void* y, int bf16, Gemm& g)
{
    g = Gemm();
    if (!m.w) return nullptr;
    if (m.N % 128 || m.K % 64 || reinterpret_cast<uintptr_t>(m.w) % 16 || m.sB % 16 || m.sG % 16)
        return "trellis Gemm: N % 128, K % 64 and 16-byte aligned tiles";
    g.on = 1;
    g.x8 = x8;
    g.tw = m.w; g.tsB = m.sB; g.tsG = m.sG;
    g.N = m.N; g.K = m.K; g.M = M;
    g.y = y; g.ldy = g.N; g.bf16 = bf16;
    return nullptr;
}

// ------------------------------------------------------------------------------------------ small elementwise passes
struct ToBf { int on = 0; const __half* src = nullptr; int ld = 0, K = 0, M = 0; uint16_t* dst = nullptr; };   // dst [M][K] BF16
__global__ void __launch_bounds__(256) k_f16_bf16(ToBf a)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    const int row = static_cast<int>(i / (a.K / 2)), k = static_cast<int>(i % (a.K / 2)) * 2;
    if (row >= a.M) return;
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(a.src + static_cast<size_t>(row) * a.ld + k));
    *reinterpret_cast<u32*>(a.dst + static_cast<size_t>(row) * a.K + k) = pack_bf2(f.x, f.y);
}
inline void launch(const ToBf& a, cudaStream_t st)
{
    if (a.on) k_f16_bf16<<<static_cast<unsigned>((static_cast<long long>(a.M) * (a.K / 2) + 255) / 256), 256, 0, st>>>(a);
}

__global__ void __launch_bounds__(256) k_f32_f16(const float* src, long long n, __half* dst)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

// act = f16(silu(g) * u), rows [M][I]. Dense trellis MLP (svh_g set): g, u are the rotated GEMM outputs, and the pass
// writes the down GEMM's records x8 of H128(f16(silu(g) u) * suh_d), g = f16(H128(g') svh_g), u likewise (k_pf_mid's
// arithmetic with per-channel vectors) in place of act and xq_act.
struct Act
{
    int on = 0;
    const __half* g = nullptr;
    const __half* u = nullptr;
    __half* act = nullptr;
    int I = 0, M = 0;
    const __half *svh_g = nullptr, *svh_u = nullptr, *suh_d = nullptr;
    uint8_t* x8 = nullptr;
};
__global__ void __launch_bounds__(256) k_pf_act(Act a)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(a.M) * (a.I / 2)) return;
    const float2 g = __half22float2(*reinterpret_cast<const __half2*>(a.g + 2 * i));
    const float2 u = __half22float2(*reinterpret_cast<const __half2*>(a.u + 2 * i));
    *reinterpret_cast<u32*>(a.act + 2 * i) = pack_h2(silu_f(g.x) * u.x, silu_f(g.y) * u.y);
}
// grid ceil(ceil(M/16)*16 * I/128 / 8), block 256: warp = (row, 128 block)
__global__ void __launch_bounds__(256) k_pf_dtr_mid(Act a)
{
    const int lane = threadIdx.x & 31, nbk = a.I / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nbk, b = w % nbk;
    if (row >= (a.M + 15) / 16 * 16) return;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < a.M)
    {
        const int c0 = b * 128 + lane * 4;
        const size_t o = static_cast<size_t>(row) * a.I + c0;
        auto f4 = [](const __half* p, float (&r)[4]) {
            const uint2 q = *reinterpret_cast<const uint2*>(p);
            const float2 x = __half22float2(as_h2(q.x)), y = __half22float2(as_h2(q.y));
            r[0] = x.x; r[1] = x.y; r[2] = y.x; r[3] = y.y;
        };
        float gv[4], uv[4], sg[4], su[4], sd[4];
        f4(a.g + o, gv); f4(a.u + o, uv); f4(a.svh_g + c0, sg); f4(a.svh_u + c0, su); f4(a.suh_d + c0, sd);
        had128(gv, lane);
        had128(uv, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float gg = rnd_h(__fmul_rn(gv[e], sg[e])), uu = rnd_h(__fmul_rn(uv[e], su[e]));
            v[e] = __fmul_rn(rnd_h(silu_f(gg) * uu), sd[e]);
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = rnd_h(v[e]);
    }
    shx_store_mx8(v, lane, row, a.M, a.I / 32, b * 4 + (lane >> 3), a.x8);
}
inline void launch(const Act& a, cudaStream_t st)
{
    if (!a.on) return;
    if (a.svh_g) k_pf_dtr_mid<<<static_cast<unsigned>(((a.M + 15) / 16 * 16 * static_cast<long long>(a.I / 128) + 7) / 8), 256, 0, st>>>(a);
    else k_pf_act<<<static_cast<unsigned>((static_cast<long long>(a.M) * (a.I / 2) + 255) / 256), 256, 0, st>>>(a);
}

// ------------------------------------------------------------------------------------------ BF16 GEMM (cuBLAS)
// y32 [M][O] fp32 = xb [M][K] (BF16) @ w [O][K]^T (BF16 row-major from the pack); y16 != nullptr: also rounded to fp16.
struct Blas
{
    int on = 0;
    cublasHandle_t h = nullptr;
    const uint16_t* xb = nullptr;
    const uint16_t* w = nullptr;
    int K = 0, O = 0, M = 0;
    float* y32 = nullptr;
    __half* y16 = nullptr;
};
inline cublasStatus_t launch(const Blas& b, cudaStream_t st)
{
    if (!b.on) return CUBLAS_STATUS_SUCCESS;
    const float one = 1.f, zero = 0.f;
    cublasStatus_t s = cublasSetStream(b.h, st);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    // row-major Y = X W^T is column-major Y' = W'^T X' with W' (K x O), X' (K x M), Y' (O x M)
    s = cublasGemmEx(b.h, CUBLAS_OP_T, CUBLAS_OP_N, b.O, b.M, b.K, &one, b.w, CUDA_R_16BF, b.K, b.xb, CUDA_R_16BF, b.K, &zero,
                     b.y32, CUDA_R_32F, b.O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    const long long n = static_cast<long long>(b.M) * b.O;
    if (b.y16) k_f32_f16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(b.y32, n, b.y16);
    return s;
}
inline const char* make_blas(const Tensor& w, int K, int M, cublasHandle_t h, const uint16_t* xb, float* y32, __half* y16, Blas& b)
{
    b = Blas();
    if (!w.p) return nullptr;
    if (w.nd != 2 || w.shape[1] != K) return "BF16 tensor is not row-major [O][K]";
    b.on = 1; b.h = h; b.xb = xb;
    b.w = static_cast<const uint16_t*>(w.p);
    b.K = K; b.O = static_cast<int>(w.shape[0]); b.M = M;
    b.y32 = y32; b.y16 = y16;
    return nullptr;
}

// ------------------------------------------------------------------------------------------ routed experts
struct Expert
{
    int on = 0, M = 0, maxseg = 0;
    int nroutes = 0;                  // routes of the pass: 0 = M * TOPK; the trellis shared pass (#65) has M (one per row)
    const __half* sh_svh = nullptr;   // set: ysh holds the trellis shared expert's down output before its output rotation;
                                      // k_pf_fin adds f16(H128(ysh) x sh_svh) (bindp::ShX, #65)
    const __half* sh_sc = nullptr;    // block-scale form: sh_svh holds the output signs and these the 48 scales
    // routing
    const float* logits = nullptr;    // [M][256] fp32
    const float* bias = nullptr;      // [256] fp32 (router.bias)
    float route_scale = 2.5f;
    int* tok_e = nullptr;             // [M][8] selected experts, selection order
    float* tok_w = nullptr;           // [M][8] routing weights
    int* cnt = nullptr;               // [256] route rows per expert
    int* rr_row = nullptr;            // [8M] token row of route i; routes sorted by (expert, row)
    int* rr_e = nullptr;              // [8M] expert of route i
    int* rr_sp = nullptr;             // [8M] segment * RS + position in the segment
    float* rr_w = nullptr;            // [8M] routing weight
    int* pos = nullptr;               // [M][8] route index of (token, k)
    int* segs = nullptr;              // [maxseg][3] expert, first route, rows (<= RS)
    int* n_seg = nullptr;             // [1]
    // weights (pack layout, as K1Args)
    const long long* t_tab = nullptr;
    const long long* t_kmix = nullptr;    // tr3.kmix (exlr8 v2 packs): fc1 width per (block, matrix); nullptr = all at bits
    const uint8_t* t_fc1 = nullptr;
    int k2 = 0;                       // the layer has 2-bit units: the K2 instantiations of fc1 / fc2 (#65)
    int bs = 0;                       // hidden-side vectors as signs x block scales (t_sg, t_v1b, t_v3b): the BS instantiations
    const __half *t_sg = nullptr, *t_v1b = nullptr, *t_v3b = nullptr;   // [2][6144] signs (in, out), [blk][2][48], [blk][48]
    int t_wide = 0;                   // K3 experts of t_fc1 are in the wide order
    const uint8_t* t_fc2 = nullptr;
    int t_wide2 = 0;                  // K3 experts of t_fc2 are in the wide order
    const __half* t_v1 = nullptr;
    const __half* t_v2 = nullptr;
    const __half* t_v3 = nullptr;
    // data
    const __half* x = nullptr;        // [M][ldx] mlp_norm output
    int ldx = 0;
    __half* z1 = nullptr;             // [8M][EB blocks][gate, up][128] fp16 (EB = 3 for the TP6 packs, 4 for TP4)
    uint8_t* x2 = nullptr;            // [maxseg][MB][EB blocks][4][REC]  fc2 inputs, TR3 k order
    __half* yw = nullptr;             // [8M][6144] fp16 weighted expert output per route
    const __half* ysh = nullptr;      // [M][6144] fp16 shared expert output
    uint16_t* out = nullptr;          // [M][6144] BF16
    RsOut rs;                         // set: k_pf_fin writes its rows to the RS send slot instead of out
#ifdef PF_AOP
    uint8_t* aop = nullptr;           // PF_AOP: [maxseg][AOP_NS][gate, up][AOPS] fragment-ready fc1 A records (k_pf_aop); nullptr = off
#endif
#ifdef PF_F12
    int f12 = 0;                      // PF_F12: 1 = fc1 runs as k_pf_fc1f (fat CTA: F12_NE adjacent expert blocks share one quantized A record); default off
#endif
};

inline int max_segments(int M) { return (M * TOPK + RS - 1) / RS + NEXP; }

// grid ceil(M/8), block 256: warp = one token. Rule of glue::k_router_plan: key = sigmoid(logit) + bias, ties ->
// lowest expert id; weights = sigmoid of the selected / their sum * route_scale.
__global__ void __launch_bounds__(256) k_pf_route(Expert A)
{
    const int l = threadIdx.x & 31, m = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (m >= A.M) return;
    float s[8], key[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int e = l + 32 * i;
        s[i] = 1.f / (1.f + expf(-A.logits[static_cast<size_t>(m) * NEXP + e]));
        key[i] = s[i] + A.bias[e];
    }
    unsigned taken = 0;
    int se[TOPK];
    float sw[TOPK], sum = 0.f;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        float bk = -INFINITY;
        int be = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (!(taken >> i & 1) && key[i] > bk) { bk = key[i]; be = l + 32 * i; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ok > bk || (ok == bk && oe < be)) { bk = ok; be = oe; }
        }
        float sv = 0.f;
        if ((be & 31) == l) { taken |= 1u << (be >> 5); sv = s[be >> 5]; }
        sv = __shfl_sync(0xffffffffu, sv, be & 31);
        se[k] = be; sw[k] = sv;
        sum += sv;
    }
    if (l < TOPK)
    {
        A.tok_e[m * TOPK + l] = se[l];
        A.tok_w[m * TOPK + l] = sw[l] / sum * A.route_scale;
    }
}

// grid 256 (expert), block 256
__global__ void __launch_bounds__(256) k_pf_count(Expert A)
{
    __shared__ int tot;
    const int e = blockIdx.x, t = threadIdx.x;
    if (t == 0) tot = 0;
    __syncthreads();
    int c = 0;
    for (int i = t; i < A.M * TOPK; i += 256) c += A.tok_e[i] == e;
    atomicAdd(&tot, c);
    __syncthreads();
    if (t == 0) A.cnt[e] = tot;
}

// grid 256 (expert), block 256: the expert's routes in token order, its segments, the inverse map.
__global__ void __launch_bounds__(256) k_pf_fill(Expert A)
{
    __shared__ int c[256], base, segb;
    const int e = blockIdx.x, t = threadIdx.x;
    const int C = (A.M + 255) / 256, m0 = min(A.M, t * C), m1 = min(A.M, m0 + C);
    int n = 0;
    for (int i = m0 * TOPK; i < m1 * TOPK; ++i) n += A.tok_e[i] == e;
    c[t] = n;
    if (t == 0)
    {
        int off = 0, sb = 0;
        for (int i = 0; i < e; ++i) { off += A.cnt[i]; sb += (A.cnt[i] + RS - 1) / RS; }
        base = off; segb = sb;
        if (e == NEXP - 1) *A.n_seg = sb + (A.cnt[e] + RS - 1) / RS;
    }
    __syncthreads();
    int p = 0;
    for (int i = 0; i < t; ++i) p += c[i];
    for (int i = m0 * TOPK; i < m1 * TOPK; ++i)
        if (A.tok_e[i] == e)
        {
            const int idx = base + p;
            A.rr_row[idx] = i / TOPK; A.rr_e[idx] = e; A.rr_w[idx] = A.tok_w[i];
            A.rr_sp[idx] = (segb + p / RS) * RS + p % RS;
            A.pos[i] = idx;
            ++p;
        }
    const int ne = A.cnt[e], nseg = (ne + RS - 1) / RS;
    for (int i = t; i < nseg; i += 256)
    {
        int* sg = A.segs + static_cast<size_t>(segb + i) * 3;
        sg[0] = e; sg[1] = base + i * RS; sg[2] = min(RS, ne - i * RS);
    }
}

// One row of a 128 block, 4 consecutive values per lane (k = lane*4 + e): MXFP8-quantize (scale per 32 k = 8 lanes)
// and store into A fragment records rec4 [4 k32 steps][..] in the TR3 k order (gemm_x.cuh): register half*2 + (rr >= 8)
// of lane (rr%8)*4 + c, byte i = k32 offset 16 (i>>1) + 2 c + (i&1) + 8 half. rec(s) = record of k32 step s for this row.
template <class R>
__device__ __forceinline__ void store_mx8_tr3(const float (&v)[4], int lane, int rr, R rec)
{
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
    float inv;
    const u32 code = mx_scale(am, inv);
    const u32 p01 = f2_e4m3x2(v[0] * inv, v[1] * inv), p23 = f2_e4m3x2(v[2] * inv, v[3] * inv);
    const int L8 = lane & 7;
    unsigned char* r = rec(lane >> 3);
    unsigned char* f = r + ((rr & 7) * 4 + 2 * (L8 & 1)) * 16 + (((L8 >> 1) & 1) * 2 + (rr >> 3)) * 4 + 2 * ((L8 >> 2) & 1);
    *reinterpret_cast<uint16_t*>(f) = static_cast<uint16_t>(p01);
    *reinterpret_cast<uint16_t*>(f + 16) = static_cast<uint16_t>(p23);
    if (L8 == 0) r[512 + rr] = static_cast<unsigned char>(code);
}

// 128-point Hadamard (Sylvester order), the stage order of had128 (strides 1, 2, .., 64; low = a + b, high = a - b), a row
// held by 4 lanes (q = lane & 3: elements q*32 + j) as 32 consecutive floats per lane: strides 1..16 in registers, 32 and
// 64 by xor-shuffle. The same fp32 values as had128 bit for bit.
__device__ __forceinline__ void had128_l32(float (&v)[32], int lane)
{
    #pragma unroll
    for (int st = 1; st < 32; st <<= 1)
        #pragma unroll
        for (int j = 0; j < 32; ++j)
            if (!(j & st)) { const float a = v[j], b = v[j + st]; v[j] = a + b; v[j + st] = a - b; }
    #pragma unroll
    for (int m = 1; m < 4; m <<= 1)
    {
        const float sg = (lane & m) ? -1.f : 1.f;    // v * sg is exact, so fmaf(v, sg, p) = RN(p -+ v): the bits of p - v / v + p
        #pragma unroll
        for (int j = 0; j < 32; ++j)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = fmaf(v[j], sg, p);
        }
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) v[j] *= SQ128;
}

// 128-point Hadamard of a row held by 16 lanes, 8 consecutive values each (element 8 (lane % 16) + j), 1/sqrt(128):
// strides 1-4 in registers, 8-64 by xor-shuffle over lane bits 0-3
__device__ __forceinline__ void had128_l8(float (&v)[8], int lane)
{
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            if (!(j & h)) { const float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
    #pragma unroll
    for (int q = 1; q < 16; q <<= 1)
    {
        const bool hi = lane & q;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], q);
            v[j] = hi ? o - v[j] : v[j] + o;
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] *= SQ128;
}

// The same as had128_l32 for a row held by 8 lanes (q = lane & 7: elements q*16 + j) as 16 consecutive floats per lane:
// strides 1..8 in registers, 16, 32 and 64 by xor-shuffle. The same fp32 values as had128 bit for bit.
__device__ __forceinline__ void had128_l16(float (&v)[16], int lane)
{
    #pragma unroll
    for (int st = 1; st < 16; st <<= 1)
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            if (!(j & st)) { const float a = v[j], b = v[j + st]; v[j] = a + b; v[j + st] = a - b; }
    #pragma unroll
    for (int m = 1; m < 8; m <<= 1)
    {
        const float sg = (lane & m) ? -1.f : 1.f;    // v * sg is exact, so fmaf(v, sg, p) = RN(p -+ v): the bits of p - v / v + p
        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = fmaf(v[j], sg, p);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; ++j) v[j] *= SQ128;
}

// 16 B chunk c (0..15) of a 256 B smem row r -> its swizzled slot: the 8 lanes of a 128-bit access (rows r, r + 1, q = 0..3,
// chunk q * 4 + j) hit 8 different bank groups. Bit 3 of c is kept, so a slot stays inside the row.
__device__ __forceinline__ int swz16(int c, int r) { return c ^ (r & 7) ^ ((c >> 3) << 1); }

// A lane holds 32 consecutive k (one k32 step) of one row of a 128 block as fp32: round to fp16 (packed pairs),
// MXFP8-quantize (amax of the rounded values in the lane) and store the row's 32 codes into the step's record `rec`
// (layout of store_mx8_tr3) as 8 u32 words, plus the scale byte. The words are stored in a per-quad rotated order so
// that the 8 quads of the warp hit 16 different banks per store.
// The scale multiply runs in packed fp16 on the rounded values h (the same bytes as fp32 h * inv then E4M3): h has 11
// significant bits and h * 2^-e <= 448, so the product is exact whenever it is a normal fp16; below 2^-14 it may round
// but converts to E4M3 zero either way. 2^-e fits fp16 for e >= -15 (amax >= 448 * 2^-15); a smaller amax scales up
// in two more steps (scaling a subnormal up is exact; e >= -32 for fp16 amax).
__device__ __forceinline__ void store_mx8_l32(const float (&v)[32], int lane, int rr, unsigned char* rec)
{
    u32 h[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = pack_h2(v[2 * j], v[2 * j + 1]);
    u32 am[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) am[j] = as_u32(__hmax2(__habs2(as_h2(h[2 * j])), __habs2(as_h2(h[2 * j + 1]))));
    #pragma unroll
    for (int w = 4; w > 0; w >>= 1)
        #pragma unroll
        for (int j = 0; j < w; ++j) am[j] = as_u32(__hmax2(as_h2(am[j]), as_h2(am[j + w])));
    const float amax = fmaxf(__low2float(as_h2(am[0])), __high2float(as_h2(am[0])));
    int e;
    const u32 code = mx_scale_e(amax, e);
    const u32 s0 = h2_pow2(min(-e, 15));
    #pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = hmul2_u(h[j], s0);
    if (e < -15)
    {
        const int k1 = min(-e - 15, 15);
        const u32 s1 = h2_pow2(k1), s2 = h2_pow2(-e - 15 - k1);
        #pragma unroll
        for (int j = 0; j < 16; ++j) h[j] = hmul2_u(hmul2_u(h[j], s1), s2);
    }
    u32 w[8];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
            w[c * 2 + hh] = h2_e4m3x2_d(h[c + 4 * hh]) | (h2_e4m3x2_d(h[c + 4 * hh + 8]) << 16);
    const int rot = (lane >> 2) & 7;            // w'[k] = w[(k + rot) & 7], barrel rotate (no dynamic register indexing)
    #pragma unroll
    for (int st = 1; st < 8; st <<= 1)
    {
        const bool on = (rot & st) != 0;
        u32 t[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) t[k] = on ? w[(k + st) & 7] : w[k];
        #pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = t[k];
    }
    unsigned char* base = rec + (rr & 7) * 64 + (rr >> 3) * 4;
    #pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        const int idx = (k + rot) & 7;          // = c * 2 + h
        *reinterpret_cast<u32*>(base + (idx >> 1) * 16 + (idx & 1) * 8) = w[k];
    }
    rec[512 + rr] = static_cast<unsigned char>(code);
}

// decoded fp16 fragments of two TR3 tiles (p: k16 step 2s, q: 2s+1; x0 = cols 0-7, x1 = cols 8-15) -> E4M3 B fragments
__device__ __forceinline__ void tr3_b8f(const u32 (&p0)[2], const u32 (&p1)[2], const u32 (&q0)[2], const u32 (&q1)[2], u32 (&b)[4])
{
    b[0] = h2_e4m3x2_d(p0[0]) | (h2_e4m3x2_d(q0[0]) << 16);
    b[1] = h2_e4m3x2_d(p0[1]) | (h2_e4m3x2_d(q0[1]) << 16);
    b[2] = h2_e4m3x2_d(p1[0]) | (h2_e4m3x2_d(q1[0]) << 16);
    b[3] = h2_e4m3x2_d(p1[1]) | (h2_e4m3x2_d(q1[1]) << 16);
}

// two TR3 tiles (k16 steps 2s, 2s+1 of one 16-col tile) -> E4M3 B fragments of the k32 step: b[0], b[1] cols 0-7, b[2], b[3] cols 8-15
template <int BITS>
__device__ __forceinline__ void tr3_b8(const Tr3Lane<BITS>& ld, const u32 (&p)[2], const u32 (&q)[2], u32 (&b)[4])
{
    u32 p0[2], p1[2], q0[2], q1[2];
    ld.decode(p[0], p[1], p0, p1);
    ld.decode(q[0], q[1], q0, q1);
    tr3_b8f(p0, p1, q0, q1, b);
}

// fc1: CTA = (segment, block b, gate | up): z1 = f16(mx8(f16(H128(x * suh))) @ e4m3(What)), 128 output cols.
// The 128-k blocks go in groups of G (2 for segments of <= 32 rows, nmb = 16-row blocks <= 2; else 1). Per group the
// segment's x rows of the G blocks are staged into smem (sX, two stages of RS x 256 B, block g's row r at stage row
// g << rsh | r) by cp.async one group ahead; then the 2 nmb warps of block g rotate and quantize its rows into sA, a lane
// holding one row's 32 k of one k32 step (had128_l32, store_mx8_l32; suh read from global), then each warp runs the
// 4 G k32 steps of its 16-col tile over the nmb row blocks, blocks in k order.
// Two barriers per group: after the stage landed (also frees sA and the other sX stage), after sA is written.
// Record (g, s, mb) of the group is sA + ((g * 4 + s) * nmb + mb) * REC: 4 G nmb <= 4 MB records.
// K3 expert in the wide order (A.t_wide, tr3_wide.cuh): a 128-k block is one wide group, the warp's 8 tiles of it come
// in three 64-bit loads; the lane's 24 bits per tile are in its own words, the 13 bits before them in lane - 1's.
// Wide weights with G = 2 are fetched a full group ahead: the even block of the next group by cp.async into sW (one
// 6 KB slot in the global group layout, so each warp's 3 x 256 B are warp-private), the odd block by ld_nc64 into
// registers (nw); with G = 1 (and the non-wide paths) the register set is loaded one block ahead as before.
constexpr int FC1_XS = RS * 256;      // bytes of one sX stage: RS rows x 128 halves (A.ldx * 2 must be a multiple of 16)
// PF_AOP (A-operand share, default off): k_pf_aop runs the x stage + prologue (x * suh, had128, MXFP8) ONCE per (segment,
// matrix, distinct prologue input) and writes the sA records of every 128-k group to global scratch (AOP_GRP bytes per
// group, the sA layout of fc1_body); the fc1 block-CTAs whose prologue inputs (suh row, or v1b scale row in BS layers) are
// byte-equal to that slot's block stage the records by cp.async instead of recomputing them. Same lane code -> same bytes.
// A block whose input is the AOP_NS+1-th distinct one of its expert runs the in-CTA prologue as before.
constexpr int AOP_GRP = 4 * MB * REC;                         // one group's records (4 G rst <= 16)
constexpr size_t AOPS = static_cast<size_t>(HID / 128) * AOP_GRP;   // one (segment, slot, matrix)
#ifndef PF_AOP_NS
#define PF_AOP_NS 1
#endif
constexpr int AOP_NS = PF_AOP_NS;     // distinct prologue inputs per (expert, matrix) with a scratch slot
#ifndef PF_AOP_NCH
#define PF_AOP_NCH 16
#endif
#ifndef PF_AOP_ST
#define PF_AOP_ST 2
#endif
#ifndef PF_AOP_MINB
#define PF_AOP_MINB 1
#endif
constexpr int AOP_ST = PF_AOP_ST;     // k_pf_aop x stages (1: no in-CTA prefetch, 25 KB smem -> more CTAs per SM)
constexpr int AOP_NCH = PF_AOP_NCH;   // k_pf_aop CTAs per (segment, slot, matrix): the 128-k groups split in AOP_NCH chunks

#ifdef PF_CLK
// clock probe (build with -DPF_CLK, not a production path): per-warp SM cycles by phase, [path: K3 wide / K3 / K4]
// [warp active in the prologue / idle] [phase], summed by the last CTA of a launch and printed, then zeroed.
__device__ unsigned long long g_pf_clk[3][2][10];
__device__ unsigned g_pf_warps[3][2];
__device__ unsigned long long g_pf_cta_cyc;
__device__ unsigned g_pf_done;
#define PF_CK(i) { const unsigned c1_ = clock(); ck_[i] += c1_ - c0_; c0_ = c1_; }
#else
#define PF_CK(i)
#endif

// PF_SKIPPRO and PF_HOIST read table[5] as the first of a contiguous run of blocks, which holds for TR3 and v1 packs
// only. In an exlr8 v2 pack table[5] is the first owned half and the halves are not contiguous.
// PF_SKIPPRO (timing-only build, wrong values, not a production path): a block-CTA runs the x stage and the prologue
// (x * suh, had128, MXFP8 into sA) only if it is the first block of its 4-block suh fragment on this rank (b == 0 or
// global block tr[5] + b a multiple of 4); the others run the MMA on whatever sA holds. Upper bound of hoisting the
// prologue to once per (segment, fragment, matrix).
#ifdef PF_SKIPPRO
#define PF_PRO_ARG , bool pro
#define PF_PRO pro
#define PF_PRO_CALL , pro
#else
#define PF_PRO_ARG
#define PF_PRO true
#define PF_PRO_CALL
#endif
// NBL > 1 (PF_HOIST, k_pf_fc1h): the CTA has NBL groups of 8 warps; group j computes local block b + j (j < nbg; a
// group past nbg repeats block b and stores nothing). Only group 0 stages x and writes sA (the prologue, with block
// b's suh = the fragment's suh); every group runs its block's MMA on that sA, exactly as a 256-thread CTA of its block
// would, so z1 is byte-equal. sW: TR3W_GROUP bytes per group.
// mw = the first tile of the unit's matrix (tr3_unit) when NBL == 1; NBL > 1 computes it from the table width (one
// width per expert: k_pf_fc1h refuses tr3.kmix experts).
template <int BITS, int G, int EB, int NBL = 1, bool BS = false, bool AOP = false>
__device__ __forceinline__ void fc1_body(const Expert& A, const long long* tr, int b, int mat, int r0, int rows, unsigned char* sA,
                                         unsigned char* sX, unsigned char* sW, const int* sRow PF_PRO_ARG, const uint8_t* mw, int nbg = 1,
                                         const uint8_t* ga = nullptr)
{
#ifdef PF_CLK
    unsigned ck_[10] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    unsigned c0_ = clock();
#endif
    constexpr int TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = NBL == 1 ? t >> 5 : (t >> 5) & 7;
    const int grp = NBL == 1 ? 0 : t >> 8;
    const bool lead = grp == 0, act = grp < nbg;
    const size_t brow = static_cast<size_t>(tr[2]) + b;   // suh of the lead block (the same for the fragment's blocks)
    if (NBL > 1)
    {
        b += act ? grp : 0;
        sW += grp * TR3W_GROUP;
    }
    const __half* suh = A.t_v1 + (brow * 2 + mat) * HID;
    const uint8_t* wb = (NBL == 1 ? mw : A.t_fc1 + tr[0] + static_cast<size_t>(b * 2 + mat) * (static_cast<size_t>(KSH) * 8 * TB))
                        + static_cast<size_t>(warp) * TB;
    Tr3Lane<BITS> ld;
    ld.init(lane);
    constexpr int NB = HID / 128;
    const int nmb = (rows + 15) / 16;
    constexpr int rsh = G == 2 ? 5 : 6;                   // log2 of the padded rows of a block in the stage
    const int rst = G == 1 ? MB : nmb;                    // record stride of a k32 step: (g * 4 + s) * rst + mb
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    const bool wide = trw_ok<BITS>() && A.t_wide;                 // K3, and K2 (#65)
    constexpr size_t WG = static_cast<size_t>(2048) * BITS;         // bytes of one wide group (TR3W_GROUP at K3)
    const uint8_t* wg = wb - static_cast<size_t>(warp) * TB + warp * 256 + lane * 8;      // the lane's first u64 of group 0
    u32 nv[8][2];
    u64 nw[3] = {0, 0, 0};
    // wide weights of 128-k block hb -> sW (the lane's 3 u64 at warp * 256 + lane * 8 + n * 2048, as in global): the warp
    // copies its own 3 x 16 chunks of 16 B, so only a __syncwarp separates the slot's last read from the next copy
    const uint8_t* wgrp = wb - static_cast<size_t>(warp) * TB;
    auto stage_w = [&](int hb) {
        const int o = warp * 256 + (lane & 15) * 16 + (lane >> 4) * 2048;
        const uint8_t* src = wgrp + static_cast<size_t>(hb) * WG;
        pf::cp16(sW + o, src + o);
        if (BITS == 3 && lane < 16) pf::cp16(sW + o + 4096, src + o + 4096);
    };
    if (wide)
    {
        if (G == 2) stage_w(0);
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + (G == 2 ? WG : 0) + n * 2048);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) ld.load(wb + static_cast<size_t>(i) * 8 * TB, nv[i][0], nv[i][1]);
    }
    // x rows of the G blocks of group hg -> sX stage hg & 1: 16 chunks of 16 B per row, RS * 16 / 256 chunks per thread:
    // chunk j of thread t is (stage row t / 16 + 16 j = block g << rsh | row r, chunk t % 16); its swizzled slot depends on
    // row & 7 = (t / 16) & 7 only, so the 4 chunks share one slot offset (+ j * 16 rows); the source offset is 32-bit
    // (halves; rows * HID < 2^32).
    const int sc = t & 15, sr0 = t >> 4;
    const int sofs = sr0 * 256 + swz16(sc, sr0) * 16;
    auto stage = [&](int hg) {
        #pragma unroll
        for (int j = 0; j < RS * 16 / 256; ++j)
        {
            const int R = sr0 + 16 * j, g = R >> rsh, r = R & ((1 << rsh) - 1);
            if (r < rows)
                pf::cp16(sX + (hg & 1) * FC1_XS + sofs + j * 4096, A.x + (static_cast<u32>(sRow[r]) * static_cast<u32>(A.ldx) + static_cast<u32>((hg * G + g) * 128 + sc * 8)));
        }
        pf::cp_commit();
    };
    // AOP: the group's records ga + hg * AOP_GRP -> sX buffer hg & 1 (4 G rst records)
    auto stage_a = [&](int hg) {
        const int nch = 4 * G * rst * (REC / 16);
        const uint8_t* src = ga + static_cast<size_t>(hg) * AOP_GRP;
        unsigned char* dst = sX + (hg & 1) * AOP_GRP;
        for (int i = t; i < nch; i += 256) pf::cp16(dst + i * 16, src + i * 16);
        pf::cp_commit();
    };
    if (AOP) stage_a(0);
    else if (PF_PRO && lead) stage(0);
    else pf::cp_commit();         // a group that does not stage x still commits its wide stage_w(0) copies
    constexpr int NG = NB / G;
    const int pg = G == 1 ? 0 : warp / (2 * nmb), pw = warp - pg * 2 * nmb;    // prologue: this warp's block of the group and its 8 rows
    const int pr = pw * 8 + (lane >> 2), q = lane & 3;
    const bool pact = pg < G && pr < rows;

    for (int hg = 0; hg < NG; ++hg)
    {
        PF_CK(0)
        pf::cp_wait<0>();
        __syncthreads();
        PF_CK(1)
        if (AOP) { if (hg + 1 < NG) stage_a(hg + 1); }
        else if (PF_PRO && lead && hg + 1 < NG) stage(hg + 1);
        PF_CK(2)
        if (!AOP && PF_PRO && lead && pg < G && pw * 8 < rows)            // warp-uniform: block pg, lane = (row pr, k32 step q)
        {
            const int R = (pg << rsh) + pr;
            const unsigned char* xs = sX + (hg & 1) * FC1_XS + R * 256;
            const __half* sp = (BS ? A.t_sg : suh) + (hg * G + pg) * 128 + q * 32;
            const __half2 scl = BS ? __half2half2(A.t_v1b[(brow * 2 + mat) * (HID / 128) + hg * G + pg]) : __half2half2(__float2half(1.f));
            float v[32];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const uint4 xr = pact ? *reinterpret_cast<const uint4*>(xs + swz16(q * 4 + j, R) * 16) : make_uint4(0u, 0u, 0u, 0u);
                uint4 sr = *reinterpret_cast<const uint4*>(sp + j * 8);
                if (BS)       // sign x block scale: the same fp16 values as the stored vector (#65)
                {
                    sr.x = as_u32(__hmul2(as_h2(sr.x), scl)); sr.y = as_u32(__hmul2(as_h2(sr.y), scl));
                    sr.z = as_u32(__hmul2(as_h2(sr.z), scl)); sr.w = as_u32(__hmul2(as_h2(sr.w), scl));
                }
                const u32 xw[4] = {xr.x, xr.y, xr.z, xr.w}, sw[4] = {sr.x, sr.y, sr.z, sr.w};
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const float2 x2 = __half22float2(as_h2(xw[e])), s2 = __half22float2(as_h2(sw[e]));
                    v[j * 8 + e * 2] = __fmul_rn(x2.x, s2.x);
                    v[j * 8 + e * 2 + 1] = __fmul_rn(x2.y, s2.y);
                }
            }
            PF_CK(3)
            had128_l32(v, lane);
            PF_CK(4)
            if (pact) store_mx8_l32(v, lane, pr & 15, sA + ((pg * 4 + q) * rst + (pr >> 4)) * REC);
            PF_CK(5)
        }
        if (!AOP) __syncthreads();
        PF_CK(6)
        for (int g = 0; g < G; ++g)
        {
        const int hb = hg * G + g;
        u32 cv[8][2];
        u64 cw[3];
        if (wide)
        {
            if (G == 2 && g == 0)
            {
                #pragma unroll
                for (int n = 0; n < trw_nw<BITS>(); ++n) cw[n] = *reinterpret_cast<const u64*>(sW + warp * 256 + lane * 8 + n * 2048);
                __syncwarp();
                if (hb + 2 < NB) stage_w(hb + 2);
                pf::cp_commit();
            }
            else
            {
                #pragma unroll
                for (int n = 0; n < 3; ++n) cw[n] = nw[n];
                if (hb + G < NB)
                {
                    #pragma unroll
                    for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wg + static_cast<size_t>(hb + G) * WG + n * 2048);
                }
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) { cv[i][0] = nv[i][0]; cv[i][1] = nv[i][1]; }
            if (hb + 1 < NB)
            {
                #pragma unroll
                for (int i = 0; i < 8; ++i) ld.load(wb + static_cast<size_t>((hb + 1) * 8 + i) * 8 * TB, nv[i][0], nv[i][1]);
            }
        }
        PF_CK(7)
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            u32 bf[4];
            if (wide)
            {
                const u32 c0 = trw_chunk<BITS>(cw, 2 * s), c1 = trw_chunk<BITS>(cw, 2 * s + 1);
                const u32 b0 = __shfl_sync(0xffffffffu, c0, (lane + 31) & 31), b1 = __shfl_sync(0xffffffffu, c1, (lane + 31) & 31);
                u32 p0[2], p1[2], q0[2], q1[2];
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b0) << (8 * BITS)) | c0, 7 * BITS, p0, p1);
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b1) << (8 * BITS)) | c1, 7 * BITS, q0, q1);
                tr3_b8f(p0, p1, q0, q1, bf);
            }
            else tr3_b8<BITS>(ld, cv[2 * s], cv[2 * s + 1], bf);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb >= nmb) continue;
                const unsigned char* rec = (AOP ? sX + (hg & 1) * AOP_GRP : sA) + ((g * 4 + s) * rst + mb) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                mma_mx8(acc[mb][0], fa, bf[0], bf[1], sa, 127u);
                mma_mx8(acc[mb][1], fa, bf[2], bf[3], sa, 127u);
            }
        }
        PF_CK(8)
        }
    }
#ifdef PF_CLK
    {
        const int path = wide ? 0 : (BITS == 3 ? 1 : 2), idle = (pg < G && pw * 8 < rows) ? 0 : 1;
        if (lane == 0)
        {
            #pragma unroll
            for (int i = 0; i < 9; ++i) atomicAdd(&g_pf_clk[path][idle][i], static_cast<unsigned long long>(ck_[i]));
            atomicAdd(&g_pf_warps[path][idle], 1u);
        }
    }
#endif
    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            if (mb >= nmb) continue;
            const int col = warp * 16 + h * 8 + 2 * c, row = mb * 16 + r;
            __half* z = A.z1 + (static_cast<size_t>(r0 + row) * (2 * EB) + b * 2 + mat) * 128 + col;
            if (act && row < rows) *reinterpret_cast<u32*>(z) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
            if (act && row + 8 < rows) *reinterpret_cast<u32*>(z + 8 * (2 * EB) * 128) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
        }
}

#if defined(PF_AOP) || defined(PF_F12)
// CTA-uniform: blocks b1, b2 of expert row tr have byte-equal prologue inputs for matrix mat (BS: the v1b scale row, the
// sign vector is shared; else the suh row of t_v1)
template <bool BS> __device__ __forceinline__ bool aop_same(const Expert& A, const long long* tr, int b1, int b2, int mat)
{
    const size_t r1 = static_cast<size_t>(tr[2]) + b1, r2 = static_cast<size_t>(tr[2]) + b2;
    constexpr int N16 = BS ? (HID / 128) * 2 / 16 : HID * 2 / 16;
    const uint4* p1 = reinterpret_cast<const uint4*>(BS ? A.t_v1b + (r1 * 2 + mat) * (HID / 128) : A.t_v1 + (r1 * 2 + mat) * HID);
    const uint4* p2 = reinterpret_cast<const uint4*>(BS ? A.t_v1b + (r2 * 2 + mat) * (HID / 128) : A.t_v1 + (r2 * 2 + mat) * HID);
    int diff = 0;
    for (int i = threadIdx.x; i < N16; i += blockDim.x)
    {
        const uint4 a = p1[i], c = p2[i];
        diff |= (a.x != c.x) | (a.y != c.y) | (a.z != c.z) | (a.w != c.w);
    }
    return !__syncthreads_or(diff);
}
// CTA-uniform: the slot of block b (rank of its first byte-equal block among the expert's distinct inputs), or -1
template <bool BS> __device__ __forceinline__ int aop_slot(const Expert& A, const long long* tr, int b, int mat)
{
    int k = 0;
    for (int c = 0; c <= b; ++c)
    {
        bool canon = true;
        for (int d = 0; d < c && canon; ++d) canon = !aop_same<BS>(A, tr, d, c, mat);
        if (!canon) continue;
        if (aop_same<BS>(A, tr, c, b, mat)) return k < AOP_NS ? k : -1;
        ++k;
    }
    return -1;
}
// the slot's block, or -1 (CTA-uniform)
template <bool BS> __device__ __forceinline__ int aop_block(const Expert& A, const long long* tr, int k, int mat)
{
    const int nb = static_cast<int>(tr[3]);
    int n = 0;
    for (int c = 0; c < nb; ++c)
    {
        bool canon = true;
        for (int d = 0; d < c && canon; ++d) canon = !aop_same<BS>(A, tr, d, c, mat);
        if (!canon) continue;
        if (n == k) return c;
        ++n;
    }
    return -1;
}
// x stage + prologue of fc1_body (lead group, NBL = 1) for groups [h0, h1), records to dst + hg * AOP_GRP
template <int G, bool BS>
__device__ __forceinline__ void aop_body(const Expert& A, const long long* tr, int b, int mat, int rows, unsigned char* sX,
                                         const int* sRow, uint8_t* dst, int h0, int h1, unsigned char* sA)
{
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const __half* suh = A.t_v1 + (brow * 2 + mat) * HID;
    const int nmb = (rows + 15) / 16;
    constexpr int rsh = G == 2 ? 5 : 6;
    const int rst = G == 1 ? MB : nmb;
    const int sc = t & 15, sr0 = t >> 4;
    const int sofs = sr0 * 256 + swz16(sc, sr0) * 16;
    auto stage = [&](int hg) {
        #pragma unroll
        for (int j = 0; j < RS * 16 / 256; ++j)
        {
            const int R = sr0 + 16 * j, g = R >> rsh, r = R & ((1 << rsh) - 1);
            if (r < rows)
                pf::cp16(sX + (hg % AOP_ST) * FC1_XS + sofs + j * 4096, A.x + (static_cast<u32>(sRow[r]) * static_cast<u32>(A.ldx) + static_cast<u32>((hg * G + g) * 128 + sc * 8)));
        }
        pf::cp_commit();
    };
    const int pg = G == 1 ? 0 : warp / (2 * nmb), pw = warp - pg * 2 * nmb;
    const int pr = pw * 8 + (lane >> 2), q = lane & 3;
    const bool pact = pg < G && pr < rows;
    if (h0 < h1) stage(h0);
    for (int hg = h0; hg < h1; ++hg)
    {
        if (AOP_ST == 1 && hg > h0) stage(hg);     // the copy-out barrier of hg - 1 freed the single stage
        pf::cp_wait<0>();
        __syncthreads();
        if (AOP_ST > 1 && hg + 1 < h1) stage(hg + 1);
        if (pg < G && pw * 8 < rows)
        {
            const int R = (pg << rsh) + pr;
            const unsigned char* xs = sX + (hg % AOP_ST) * FC1_XS + R * 256;
            const __half* sp = (BS ? A.t_sg : suh) + (hg * G + pg) * 128 + q * 32;
            const __half2 scl = BS ? __half2half2(A.t_v1b[(brow * 2 + mat) * (HID / 128) + hg * G + pg]) : __half2half2(__float2half(1.f));
            float v[32];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const uint4 xr = pact ? *reinterpret_cast<const uint4*>(xs + swz16(q * 4 + j, R) * 16) : make_uint4(0u, 0u, 0u, 0u);
                uint4 sr = *reinterpret_cast<const uint4*>(sp + j * 8);
                if (BS)
                {
                    sr.x = as_u32(__hmul2(as_h2(sr.x), scl)); sr.y = as_u32(__hmul2(as_h2(sr.y), scl));
                    sr.z = as_u32(__hmul2(as_h2(sr.z), scl)); sr.w = as_u32(__hmul2(as_h2(sr.w), scl));
                }
                const u32 xw[4] = {xr.x, xr.y, xr.z, xr.w}, sw[4] = {sr.x, sr.y, sr.z, sr.w};
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const float2 x2 = __half22float2(as_h2(xw[e])), s2 = __half22float2(as_h2(sw[e]));
                    v[j * 8 + e * 2] = __fmul_rn(x2.x, s2.x);
                    v[j * 8 + e * 2 + 1] = __fmul_rn(x2.y, s2.y);
                }
            }
            had128_l32(v, lane);
            if (pact) store_mx8_l32(v, lane, pr & 15, sA + ((pg * 4 + q) * rst + (pr >> 4)) * REC);
        }
        __syncthreads();      // sA complete: coalesced 16 B copy to the group records in global (unwritten rows are never read back)
        {
            const int nch = 4 * G * rst * (REC / 16);
            uint4* gd = reinterpret_cast<uint4*>(dst + static_cast<size_t>(hg) * AOP_GRP);
            const uint4* sd = reinterpret_cast<const uint4*>(sA);
            for (int i = t; i < nch; i += 256) gd[i] = sd[i];
        }
        if (AOP_ST == 1) __syncthreads();   // sX and sA free for the next group
    }
}
// grid (2 AOP_NS AOP_NCH = (slot * AOP_NCH + chunk) * 2 + matrix, maxseg), block 256
template <bool BS>
__global__ void __launch_bounds__(256, PF_AOP_MINB) k_pf_aop(Expert A)
{
    __shared__ __align__(16) unsigned char sX[AOP_ST * FC1_XS];
    __shared__ __align__(16) unsigned char sA[AOP_GRP];
    __shared__ int sRow[RS];
    const int seg = blockIdx.y;
    if (seg >= *A.n_seg) return;
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const int r0 = sg[1], rows = sg[2];
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    const int mat = blockIdx.x & 1, ch = (blockIdx.x >> 1) % AOP_NCH, k = (blockIdx.x >> 1) / AOP_NCH;
    const int b = aop_block<BS>(A, tr, k, mat);
    if (b < 0) return;
    if (threadIdx.x < rows) sRow[threadIdx.x] = A.rr_row[r0 + threadIdx.x];
    __syncthreads();
    uint8_t* dst = A.aop + ((static_cast<size_t>(seg) * AOP_NS + k) * 2 + mat) * AOPS;
    if (rows <= 32)
    {
        constexpr int NG = HID / 128 / 2;
        aop_body<2, BS>(A, tr, b, mat, rows, sX, sRow, dst, ch * NG / AOP_NCH, (ch + 1) * NG / AOP_NCH, sA);
    }
    else
    {
        constexpr int NG = HID / 128;
        aop_body<1, BS>(A, tr, b, mat, rows, sX, sRow, dst, ch * NG / AOP_NCH, (ch + 1) * NG / AOP_NCH, sA);
    }
}
#endif

// grid (2 EB = block * 2 + matrix, maxseg), block 256. EB = blocks per expert the scratch and grid are laid out for
// (g6nr::EB: 3 = the TP6 packs, 4 = TP4; every expert of the table must have tr[3] <= EB).
template <int EB = g6nr::EB, bool K2 = false, bool BS = false, bool AOPK = false>
__global__ void __launch_bounds__(256, 2) k_pf_fc1(Expert A)
{
    __shared__ __align__(16) unsigned char sA[4 * MB * REC];
    __shared__ __align__(16) unsigned char sX[2 * FC1_XS];
    __shared__ __align__(16) unsigned char sW[TR3W_GROUP];
    __shared__ int sRow[RS];
    const int seg = blockIdx.y;
#ifdef PF_CLK
    const unsigned kc0 = static_cast<unsigned>(clock());
    auto pf_clk_end = [&](bool ran) {
        __syncthreads();
        if (threadIdx.x != 0) return;
        if (ran) atomicAdd(&g_pf_cta_cyc, static_cast<unsigned long long>(static_cast<unsigned>(clock()) - kc0));
        __threadfence();
        const unsigned total = gridDim.x * gridDim.y;
        if (atomicAdd(&g_pf_done, 1u) + 1u == total)
        {
            __threadfence();
            printf("PF_CLK ctas %u cta_cyc_sum %llu\n", total, g_pf_cta_cyc);
            for (int p = 0; p < 3; ++p) for (int a = 0; a < 2; ++a)
            {
                const unsigned w = g_pf_warps[p][a];
                if (!w) continue;
                printf("PF_CLK path %d idle %d warps %u cyc/warp: top %llu wait1 %llu stage %llu xload %llu had %llu quant %llu wait2 %llu wld %llu mma %llu\n",
                       p, a, w, g_pf_clk[p][a][0] / w, g_pf_clk[p][a][1] / w, g_pf_clk[p][a][2] / w, g_pf_clk[p][a][3] / w,
                       g_pf_clk[p][a][4] / w, g_pf_clk[p][a][5] / w, g_pf_clk[p][a][6] / w, g_pf_clk[p][a][7] / w, g_pf_clk[p][a][8] / w);
                for (int i = 0; i < 10; ++i) g_pf_clk[p][a][i] = 0ull;
                g_pf_warps[p][a] = 0u;
            }
            g_pf_cta_cyc = 0ull; g_pf_done = 0u;
        }
    };
#endif
    if (seg >= *A.n_seg) {
#ifdef PF_CLK
        pf_clk_end(false);
#endif
        return;
    }
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const int r0 = sg[1], rows = sg[2];
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    const int mat = blockIdx.x / EB, b = blockIdx.x - mat * EB;     // x = mat * EB + b: the EB blocks of one (segment, matrix) are adjacent on x (cluster-able)
    if (b >= static_cast<int>(tr[3])) {
#ifdef PF_CLK
        pf_clk_end(false);
#endif
        return;
    }
    if (threadIdx.x < rows) sRow[threadIdx.x] = A.rr_row[r0 + threadIdx.x];
    __syncthreads();
#ifdef PF_SKIPPRO
    const bool pro = b == 0 || (static_cast<int>(tr[5]) + b) % 4 == 0;
#endif
    size_t off;
    const int k = tr3_unit(A.t_kmix, sg[0], tr, b * 2 + mat, off);
    const uint8_t* mw = A.t_fc1 + off;
#ifdef PF_AOP
    if (AOPK && A.aop)      // AOPK = false: the stock instantiation (same code as a build without PF_AOP)
    {
        const int sl = aop_slot<BS>(A, tr, b, mat);
        if (sl >= 0)
        {
            const uint8_t* ga = A.aop + ((static_cast<size_t>(seg) * AOP_NS + sl) * 2 + mat) * AOPS;
            if (rows <= 32)
            {
                if (k == 3) fc1_body<3, 2, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
                else if (!K2 || k == 4) fc1_body<4, 2, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
                else fc1_body<K2 ? 2 : 4, 2, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
            }
            else
            {
                if (k == 3) fc1_body<3, 1, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
                else if (!K2 || k == 4) fc1_body<4, 1, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
                else fc1_body<K2 ? 2 : 4, 1, EB, 1, BS, true>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw, 1, ga);
            }
            return;
        }
    }
#endif
    if (rows <= 32)                       // G = 2: two 128-k blocks per group
    {
        if (k == 3) fc1_body<3, 2, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
        else if (!K2 || k == 4) fc1_body<4, 2, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
        else fc1_body<K2 ? 2 : 4, 2, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
    }
    else
    {
        if (k == 3) fc1_body<3, 1, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
        else if (!K2 || k == 4) fc1_body<4, 1, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
        else fc1_body<K2 ? 2 : 4, 1, EB, 1, BS>(A, tr, b, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, mw);
    }
#ifdef PF_CLK
    pf_clk_end(true);
#endif
}

#ifdef PF_F12
#ifndef F12_NE
#define F12_NE 2
#endif
#endif
#ifdef PF_F12
// PF-F12 "fat CTA": one CTA = (segment, matrix, NE adjacent expert blocks). Each 128-k group's A record is quantized ONCE
// per CTA (the stock prologue lane code -> the same record bytes) into the local sA, then the MMA of the NE blocks runs on it
// (group-major, NE accumulator sets live, per block the same op order as fc1_body -> byte-equal z1). No cluster, no DSMEM.
// x stage: ONE stage (the stage of group hg + 1 is issued after the quantize of hg, overlapping the MMA loop), so smem =
// sA | sX | NE x sW | sRow. WIDE = trw_ok<BITS>() is a compile-time path choice (the host launches f12 only if A.t_wide).
template <int BITS, int G, int NE, bool BS>
__device__ __forceinline__ void fc1f_body(const Expert& A, const long long* tr, int e0, int mat, int r0, int rows, unsigned char* sA,
                                          unsigned char* sX, unsigned char* sW, const int* sRow, const uint8_t* const* mwp)
{
    constexpr int TB = 32 * BITS;
    constexpr bool wide = trw_ok<BITS>();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const size_t brow = static_cast<size_t>(tr[2]) + e0;           // suh of the lead block (byte-equal in the CTA's blocks, checked by the caller)
    const __half* suh = A.t_v1 + (brow * 2 + mat) * HID;
    constexpr int NB = HID / 128, NG = NB / G;
    const int nmb = (rows + 15) / 16;
    constexpr int rsh = G == 2 ? 5 : 6;
    const int rst = G == 1 ? MB : nmb;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    float acc[NE][MB][2][4];
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[e][mb][h][i] = 0.f;
    constexpr size_t WG = static_cast<size_t>(2048) * BITS;
    Tr3Lane<BITS> ld;
    ld.init(lane);
    u32 nv[NE][8][2];
    u64 nw[NE][3];
    #pragma unroll
    for (int e = 0; e < NE; ++e) { nw[e][0] = 0; nw[e][1] = 0; nw[e][2] = 0; }
    auto stage_w = [&](int e, int hb) {
        const int o = warp * 256 + (lane & 15) * 16 + (lane >> 4) * 2048;
        const uint8_t* src = mwp[e] + static_cast<size_t>(hb) * WG;
        unsigned char* dw = sW + e * TR3W_GROUP;
        pf::cp16(dw + o, src + o);
        if (BITS == 3 && lane < 16) pf::cp16(dw + o + 4096, src + o + 4096);
    };
    if (wide)
    {
        #pragma unroll
        for (int e = 0; e < NE; ++e)
        {
            if (G == 2) stage_w(e, 0);
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[e][n] = ld_nc64(mwp[e] + warp * 256 + lane * 8 + (G == 2 ? WG : 0) + n * 2048);
        }
    }
    else
    {
        #pragma unroll
        for (int e = 0; e < NE; ++e)
            #pragma unroll
            for (int i = 0; i < 8; ++i) ld.load(mwp[e] + warp * TB + static_cast<size_t>(i) * 8 * TB, nv[e][i][0], nv[e][i][1]);
    }
    const int sc = t & 15, sr0 = t >> 4;
    const int sofs = sr0 * 256 + swz16(sc, sr0) * 16;
    auto stage = [&](int hg) {
        #pragma unroll
        for (int j = 0; j < RS * 16 / 256; ++j)
        {
            const int R = sr0 + 16 * j, g = R >> rsh, r = R & ((1 << rsh) - 1);
            if (r < rows)
                pf::cp16(sX + sofs + j * 4096, A.x + (static_cast<u32>(sRow[r]) * static_cast<u32>(A.ldx) + static_cast<u32>((hg * G + g) * 128 + sc * 8)));
        }
        pf::cp_commit();
    };
    stage(0);       // also commits the wide stage_w(e, 0) copies
    const int pg = G == 1 ? 0 : warp / (2 * nmb), pw = warp - pg * 2 * nmb;
    const int pr = pw * 8 + (lane >> 2), q = lane & 3;
    const bool pact = pg < G && pr < rows;
    for (int hg = 0; hg < NG; ++hg)
    {
        pf::cp_wait<0>();
        __syncthreads();                      // x of group hg landed; sA / sX free (the MMA of hg - 1 is done)
        if (pg < G && pw * 8 < rows)          // the stock prologue (fc1_body), once for the NE blocks
        {
            const int R = (pg << rsh) + pr;
            const unsigned char* xs = sX + R * 256;
            const __half* sp = (BS ? A.t_sg : suh) + (hg * G + pg) * 128 + q * 32;
            const __half2 scl = BS ? __half2half2(A.t_v1b[(brow * 2 + mat) * (HID / 128) + hg * G + pg]) : __half2half2(__float2half(1.f));
            float v[32];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const uint4 xr = pact ? *reinterpret_cast<const uint4*>(xs + swz16(q * 4 + j, R) * 16) : make_uint4(0u, 0u, 0u, 0u);
                uint4 sr = *reinterpret_cast<const uint4*>(sp + j * 8);
                if (BS)
                {
                    sr.x = as_u32(__hmul2(as_h2(sr.x), scl)); sr.y = as_u32(__hmul2(as_h2(sr.y), scl));
                    sr.z = as_u32(__hmul2(as_h2(sr.z), scl)); sr.w = as_u32(__hmul2(as_h2(sr.w), scl));
                }
                const u32 xw[4] = {xr.x, xr.y, xr.z, xr.w}, sw[4] = {sr.x, sr.y, sr.z, sr.w};
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const float2 x2 = __half22float2(as_h2(xw[e])), s2 = __half22float2(as_h2(sw[e]));
                    v[j * 8 + e * 2] = __fmul_rn(x2.x, s2.x);
                    v[j * 8 + e * 2 + 1] = __fmul_rn(x2.y, s2.y);
                }
            }
            had128_l32(v, lane);
            if (pact) store_mx8_l32(v, lane, pr & 15, sA + ((pg * 4 + q) * rst + (pr >> 4)) * REC);
        }
        __syncthreads();                      // sA complete, sX free
        if (hg + 1 < NG) stage(hg + 1);       // x of the next group lands while the MMA below runs
        for (int g = 0; g < G; ++g)
        {
            const int hb = hg * G + g;
            u32 cv[NE][8][2];
            u64 cw[NE][3];
            #pragma unroll
            for (int e = 0; e < NE; ++e)
            {
                if (wide)
                {
                    if (G == 2 && g == 0)
                    {
                        #pragma unroll
                        for (int n = 0; n < trw_nw<BITS>(); ++n) cw[e][n] = *reinterpret_cast<const u64*>(sW + e * TR3W_GROUP + warp * 256 + lane * 8 + n * 2048);
                        __syncwarp();
                        if (hb + 2 < NB) stage_w(e, hb + 2);
                    }
                    else
                    {
                        #pragma unroll
                        for (int n = 0; n < 3; ++n) cw[e][n] = nw[e][n];
                        if (hb + G < NB)
                        {
                            #pragma unroll
                            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[e][n] = ld_nc64(mwp[e] + warp * 256 + lane * 8 + static_cast<size_t>(hb + G) * WG + n * 2048);
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) { cv[e][i][0] = nv[e][i][0]; cv[e][i][1] = nv[e][i][1]; }
                    if (hb + 1 < NB)
                    {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) ld.load(mwp[e] + warp * TB + static_cast<size_t>((hb + 1) * 8 + i) * 8 * TB, nv[e][i][0], nv[e][i][1]);
                    }
                }
            }
            if (wide && G == 2 && g == 0) pf::cp_commit();
            #pragma unroll
            for (int s = 0; s < 4; ++s)
            {
                u32 bf[NE][4];
                #pragma unroll
                for (int e = 0; e < NE; ++e)
                {
                    if (wide)
                    {
                        const u32 c0 = trw_chunk<BITS>(cw[e], 2 * s), c1 = trw_chunk<BITS>(cw[e], 2 * s + 1);
                        const u32 b0 = __shfl_sync(0xffffffffu, c0, (lane + 31) & 31), b1 = __shfl_sync(0xffffffffu, c1, (lane + 31) & 31);
                        u32 p0[2], p1[2], q0[2], q1[2];
                        Tr3Lane<BITS>::decode_m((static_cast<u64>(b0) << (8 * BITS)) | c0, 7 * BITS, p0, p1);
                        Tr3Lane<BITS>::decode_m((static_cast<u64>(b1) << (8 * BITS)) | c1, 7 * BITS, q0, q1);
                        tr3_b8f(p0, p1, q0, q1, bf[e]);
                    }
                    else tr3_b8<BITS>(ld, cv[e][2 * s], cv[e][2 * s + 1], bf[e]);
                }
                #pragma unroll
                for (int mb = 0; mb < MB; ++mb)
                {
                    if (mb >= nmb) continue;
                    const unsigned char* rec = sA + ((g * 4 + s) * rst + mb) * REC;
                    const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                    const u32 sa = rec[512 + srow];
                    #pragma unroll
                    for (int e = 0; e < NE; ++e)
                    {
                        mma_mx8(acc[e][mb][0], fa, bf[e][0], bf[e][1], sa, 127u);
                        mma_mx8(acc[e][mb][1], fa, bf[e][2], bf[e][3], sa, 127u);
                    }
                }
            }
        }
    }
    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                if (mb >= nmb) continue;
                const int col = warp * 16 + h * 8 + 2 * c, row = mb * 16 + r;
                __half* z = A.z1 + (static_cast<size_t>(r0 + row) * (2 * g6nr::EB) + (e0 + e) * 2 + mat) * 128 + col;
                if (row < rows) *reinterpret_cast<u32*>(z) = pack_h2(acc[e][mb][h][0], acc[e][mb][h][1]);
                if (row + 8 < rows) *reinterpret_cast<u32*>(z + 8 * (2 * g6nr::EB) * 128) = pack_h2(acc[e][mb][h][2], acc[e][mb][h][3]);
            }
}

template <bool K2, bool BS, int NEX>
__device__ __forceinline__ void f12_run(const Expert& A, const long long* tr, int e0, int mat, int r0, int rows, int k, unsigned char* sA,
                                        unsigned char* sX, unsigned char* sW, const int* sRow, const uint8_t* const* mwp)
{
    if constexpr (NEX > 1)
    {
        // fat bodies are wide-order only (k == 3, or k == 2 on K2 tables): the non-wide K4 body would keep NE x 2 x (nv + cv) = 64
        // more live registers, so the kernel sends K4 groups through the NE = 1 bodies (fat is false for them, see k_pf_fc1f)
        if (rows <= 32) { if (k == 3) fc1f_body<3, 2, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp); else if constexpr (K2) fc1f_body<2, 2, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp); }
        else            { if (k == 3) fc1f_body<3, 1, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp); else if constexpr (K2) fc1f_body<2, 1, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp); }
        return;
    }
    if (rows <= 32)
    {
        if (k == 3) fc1f_body<3, 2, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
        else if (!K2 || k == 4) fc1f_body<4, 2, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
        else fc1f_body<K2 ? 2 : 4, 2, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
    }
    else
    {
        if (k == 3) fc1f_body<3, 1, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
        else if (!K2 || k == 4) fc1f_body<4, 1, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
        else fc1f_body<K2 ? 2 : 4, 1, NEX, BS>(A, tr, e0, mat, r0, rows, sA, sX, sW, sRow, mwp);
    }
}

constexpr int F12_SMEM(int ne) { return 4 * MB * REC + FC1_XS + ne * TR3W_GROUP + RS * 4; }
// grid (2 * EB / NE: x = mat * (EB / NE) + pair, maxseg), block 256, dynamic smem F12_SMEM(NE). Blocks [pair * NE, pair * NE + NE)
// of the (segment, matrix): all NE present, one width and byte-equal prologue inputs -> the fat body; else one block at a time
// through the NE = 1 body (every block quantizes its own record, i.e. the stock arithmetic).
template <int EB = g6nr::EB, bool K2 = false, bool BS = false, int NE = F12_NE>
__global__ void __launch_bounds__(256, 2) k_pf_fc1f(Expert A)
{
    extern __shared__ __align__(16) unsigned char smf[];
    unsigned char* const sA = smf;
    unsigned char* const sX = sA + 4 * MB * REC;
    unsigned char* const sW = sX + FC1_XS;
    int* const sRow = reinterpret_cast<int*>(sW + NE * TR3W_GROUP);
    const int seg = blockIdx.y;
    if (seg >= *A.n_seg) return;
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const int r0 = sg[1], rows = sg[2];
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    constexpr int NP = EB / NE;
    const int mat = blockIdx.x / NP, e0 = (blockIdx.x - mat * NP) * NE;
    const int nb = static_cast<int>(tr[3]);
    if (e0 >= nb) return;
    if (threadIdx.x < rows) sRow[threadIdx.x] = A.rr_row[r0 + threadIdx.x];
    __syncthreads();
    const int nblk = min(NE, nb - e0);
    int kk[NE];
    const uint8_t* mw[NE];
    #pragma unroll
    for (int e = 0; e < NE; ++e)
    {
        kk[e] = 0; mw[e] = nullptr;
        if (e < nblk) { size_t off; kk[e] = tr3_unit(A.t_kmix, sg[0], tr, (e0 + e) * 2 + mat, off); mw[e] = A.t_fc1 + off; }
    }
    bool fat = nblk == NE;
    fat = fat && (kk[0] == 3 || (K2 && kk[0] == 2));        // wide-order groups only (see f12_run)
    #pragma unroll
    for (int e = 1; e < NE; ++e) fat = fat && kk[e] == kk[0];
    for (int e = 1; e < NE && fat; ++e) fat = aop_same<BS>(A, tr, e0, e0 + e, mat);     // CTA-uniform
    if (fat)
    {
        f12_run<K2, BS, NE>(A, tr, e0, mat, r0, rows, kk[0], sA, sX, sW, sRow, mw);
        return;
    }
    for (int e = 0; e < nblk; ++e)
    {
        __syncthreads();
        pf::cp_wait<0>();
        const uint8_t* m1[1] = {mw[e]};
        f12_run<K2, BS, 1>(A, tr, e0 + e, mat, r0, rows, kk[e], sA, sX, sW, sRow, m1);
    }
}
#endif
#ifdef PF_HOIST
// fc1 with the prologue hoisted (frontier-2 #1, measurement build): CTA = (segment, pair of local blocks inside one
// 4-block suh fragment, matrix), 512 threads = 2 groups of 8 warps (fc1_body NBL = 2). The x stage and prologue run
// once per pair instead of once per block; same z1 bytes. A fragment's local blocks [a, e) give ceil((e - a) / 2)
// pairs, fragments in order; grid x = 2 * FC1H_PAIRS(EB). (4 blocks per CTA would need 1024 threads = 64 registers;
// fc1 uses 128.)
constexpr int FC1H_SMEM = 4 * MB * REC + 2 * FC1_XS + 2 * TR3W_GROUP + RS * 4;
__host__ __device__ constexpr int fc1h_pairs(int eb) { return eb / 2 + 2; }   // >= pairs of any eb-block run
template <int EB>
__global__ void __launch_bounds__(512, 1) k_pf_fc1h(Expert A)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sX = sA + 4 * MB * REC;
    unsigned char* sW = sX + 2 * FC1_XS;
    int* sRow = reinterpret_cast<int*>(sW + 2 * TR3W_GROUP);
    const int seg = blockIdx.y;
    if (seg >= *A.n_seg) return;
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const int r0 = sg[1], rows = sg[2];
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    const int gb0 = static_cast<int>(tr[5]), nb = static_cast<int>(tr[3]), mat = blockIdx.x & 1;
    int c = blockIdx.x >> 1, bs = -1, n = 0;
    for (int f = gb0 >> 2; nb > 0 && f <= (gb0 + nb - 1) >> 2 && bs < 0; ++f)
    {
        const int a = max(0, 4 * f - gb0), e = min(nb, 4 * f + 4 - gb0), k = (e - a + 1) / 2;
        if (c < k) { bs = a + 2 * c; n = min(2, e - bs); }
        else c -= k;
    }
    if (bs < 0) return;
    if (threadIdx.x < rows) sRow[threadIdx.x] = A.rr_row[r0 + threadIdx.x];
    __syncthreads();
    if (rows <= 32)
    {
        if (tr[4] == 3) fc1_body<3, 2, EB, 2>(A, tr, bs, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, nullptr, n);
        else fc1_body<4, 2, EB, 2>(A, tr, bs, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, nullptr, n);
    }
    else
    {
        if (tr[4] == 3) fc1_body<3, 1, EB, 2>(A, tr, bs, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, nullptr, n);
        else fc1_body<4, 1, EB, 2>(A, tr, bs, mat, r0, rows, sA, sX, sW, sRow PF_PRO_CALL, nullptr, n);
    }
}
#endif

// mid: warp = (route, block): g, u from z1 -> act -> rotated, quantized fc2 input record.
// grid ceil(8M * EB / 8), block 256
template <int EB = g6nr::EB>
__global__ void __launch_bounds__(256) k_pf_mid(Expert A)
{
    const int lane = threadIdx.x & 31;
    const long long g = static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int route = static_cast<int>(g / EB), b = static_cast<int>(g % EB);
    if (route >= (A.nroutes ? A.nroutes : A.M * TOPK)) return;
    const long long* tr = A.t_tab + static_cast<size_t>(A.rr_e[route]) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const __half* v2 = A.t_v2 + (static_cast<size_t>(tr[2]) + b) * 3 * 128 + lane * 4;
    const __half* z = A.z1 + (static_cast<size_t>(route) * (2 * EB) + b * 2) * 128 + lane * 4;
    const uint2 zg = *reinterpret_cast<const uint2*>(z), zu = *reinterpret_cast<const uint2*>(z + 128);
    const uint2 sg = *reinterpret_cast<const uint2*>(v2), su = *reinterpret_cast<const uint2*>(v2 + 128), sd = *reinterpret_cast<const uint2*>(v2 + 256);
    auto f4 = [](const uint2& q, float (&o)[4]) {
        const float2 a = __half22float2(as_h2(q.x)), c = __half22float2(as_h2(q.y));
        o[0] = a.x; o[1] = a.y; o[2] = c.x; o[3] = c.y;
    };
    float gv[4], uv[4], svg[4], svu[4], sud[4], v[4];
    f4(zg, gv); f4(zu, uv); f4(sg, svg); f4(su, svu); f4(sd, sud);
    had128(gv, lane);
    had128(uv, lane);
    #pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        const float gg = rnd_h(__fmul_rn(gv[e], svg[e])), uu = rnd_h(__fmul_rn(uv[e], svu[e]));
        v[e] = __fmul_rn(rnd_h(silu_f(gg) * uu), sud[e]);
    }
    had128(v, lane);
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = rnd_h(v[e]);
    const int sp = A.rr_sp[route], p = sp % RS;
    uint8_t* rec = A.x2 + ((static_cast<size_t>(sp / RS) * MB + (p >> 4)) * EB + b) * 4 * REC;
    store_mx8_tr3(v, lane, p & 15, [&](int s) { return rec + s * REC; });
}

// fc2: CTA = (segment, 128 output cols); loops over the expert's blocks b: z = f16(X2_b @ e4m3(Wd_b)) into smem,
// y += H128(z) * svh_d; then yw = f16(route weight * y). The block's X2 records and its svh slice are staged by cp.async
// one block ahead (sA, sV, two stages) and the wide weight words of the next block are loaded before this block's
// epilogue. In the epilogue warp w takes rows 4w..4w+3 (and 4w+32.. when rows > 32), a lane one row's 16 columns of one
// eighth (had128_l16), so all 8 warps rotate when rows >= 29; sZ rows are chunk-swizzled (swz16) so the tile writes and
// the row reads are bank-conflict free. Two barriers per block.
template <int BITS, int EB, bool BS = false>
__device__ __forceinline__ void fc2_body(const Expert& A, const long long* tr, int seg, int r0, int rows, unsigned char* sA, __half* sZ,
                                         unsigned char* sV)
{
    constexpr int TB = 32 * BITS;
    constexpr int SA = MB * 4 * REC;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int G = blockIdx.x, nb = static_cast<int>(tr[3]), nmb = (rows + 15) / 16;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0), c = lane & 3, r = lane >> 2;
    Tr3Lane<BITS> ld;
    ld.init(lane);
    // K3 (or K2) expert in the wide order (A.t_wide2, tr3_wide.cuh): the n-tile's 8 k-tiles in three (two) 64-bit loads
    const bool wide = trw_ok<BITS>() && A.t_wide2;                // K3, and K2 (#65)
    unsigned char* sZb = reinterpret_cast<unsigned char*>(sZ);
    const int pr = warp * 4 + (lane >> 3), q = lane & 7;     // epilogue: this lane's rows pr, pr + 32 of the segment and column eighth
    const int np = rows > 32 ? 2 : 1;                         // row passes (warp-uniform)
    float ys[2][16];
    #pragma unroll
    for (int p = 0; p < 2; ++p)
        #pragma unroll
        for (int j = 0; j < 16; ++j) ys[p][j] = 0.f;
    auto wptr = [&](int b) {
        return A.t_fc2 + tr[1] + static_cast<size_t>(b) * (static_cast<size_t>(KSH) * 8 * TB) + static_cast<size_t>(G * 8 + warp) * 8 * TB;
    };
    // X2 records and the svh slice of block b -> stage b & 1
    auto stage = [&](int b) {
        for (int ch = t; ch < MB * 4 * 33; ch += 256)
        {
            const int mb = ch / (4 * 33), rem = ch % (4 * 33);
            pf::cp16(sA + (b & 1) * SA + ch * 16, A.x2 + ((static_cast<size_t>(seg) * MB + mb) * EB + b) * 4 * REC + rem * 16);
        }
        if (t < 16) pf::cp16(sV + (b & 1) * 256 + t * 16, BS ? A.t_sg + HID + G * 128 + t * 8 : A.t_v3 + (static_cast<size_t>(tr[2]) + b) * HID + G * 128 + t * 8);
        pf::cp_commit();
    };
    stage(0);
    u64 cw[3] = {0, 0, 0};
    if (wide)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) cw[n] = ld_nc64(wptr(0) + n * 256 + lane * 8);
    }
    for (int b = 0; b < nb; ++b)
    {
        u32 cv[8][2];
        if (!wide)
        {
            const uint8_t* wb = wptr(b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) ld.load(wb + i * TB, cv[i][0], cv[i][1]);
        }
        pf::cp_wait<0>();
        __syncthreads();
        if (b + 1 < nb) stage(b + 1);
        u64 nw[3] = {0, 0, 0};
        if (wide && b + 1 < nb)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wptr(b + 1) + n * 256 + lane * 8);
        }
        const unsigned char* sAb = sA + (b & 1) * SA;
        float acc[MB][2][4];
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            u32 bf[4];
            if (wide)
            {
                const u32 c0 = trw_chunk<BITS>(cw, 2 * s), c1 = trw_chunk<BITS>(cw, 2 * s + 1);
                const u32 b0 = __shfl_sync(0xffffffffu, c0, (lane + 31) & 31), b1 = __shfl_sync(0xffffffffu, c1, (lane + 31) & 31);
                u32 p0[2], p1[2], q0[2], q1[2];
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b0) << (8 * BITS)) | c0, 7 * BITS, p0, p1);
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b1) << (8 * BITS)) | c1, 7 * BITS, q0, q1);
                tr3_b8f(p0, p1, q0, q1, bf);
            }
            else tr3_b8<BITS>(ld, cv[2 * s], cv[2 * s + 1], bf);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb >= nmb) continue;
                const unsigned char* rec = sAb + (mb * 4 + s) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                mma_mx8(acc[mb][0], fa, bf[0], bf[1], sa, 127u);
                mma_mx8(acc[mb][1], fa, bf[2], bf[3], sa, 127u);
            }
        }
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                if (mb >= nmb) continue;
                const int row = mb * 16 + r, ch = warp * 2 + h;
                *reinterpret_cast<u32*>(sZb + row * 256 + swz16(ch, row) * 16 + 4 * c) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
                *reinterpret_cast<u32*>(sZb + (row + 8) * 256 + swz16(ch, row + 8) * 16 + 4 * c) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
            }
        __syncthreads();
        const __half* svp = reinterpret_cast<const __half*>(sV + (b & 1) * 256) + q * 16;
        const float scl = BS ? __half2float(A.t_v3b[(static_cast<size_t>(tr[2]) + b) * (HID / 128) + G]) : 1.f;   // BS: sV holds signs
        float sv[16];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint4 sr = *reinterpret_cast<const uint4*>(svp + j * 8);
            const u32 sw[4] = {sr.x, sr.y, sr.z, sr.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float2 s2 = __half22float2(as_h2(sw[e]));
                if (BS) { sv[j * 8 + e * 2] = s2.x * scl; sv[j * 8 + e * 2 + 1] = s2.y * scl; }
                else { sv[j * 8 + e * 2] = s2.x; sv[j * 8 + e * 2 + 1] = s2.y; }
            }
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            if (p >= np || warp * 4 + p * 32 >= rows) continue;      // warp-uniform
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
            #pragma unroll
            for (int j = 0; j < 16; ++j) ys[p][j] += v[j] * sv[j];
        }
        #pragma unroll
        for (int n = 0; n < 3; ++n) cw[n] = nw[n];
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        const int rp = pr + p * 32;
        if (p < np && rp < rows)
        {
            const float w = A.rr_w[r0 + rp];
            __half* yp = A.yw + static_cast<size_t>(r0 + rp) * HID + G * 128 + q * 16;
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                uint4 o;
                o.x = pack_h2(w * ys[p][j * 8], w * ys[p][j * 8 + 1]);
                o.y = pack_h2(w * ys[p][j * 8 + 2], w * ys[p][j * 8 + 3]);
                o.z = pack_h2(w * ys[p][j * 8 + 4], w * ys[p][j * 8 + 5]);
                o.w = pack_h2(w * ys[p][j * 8 + 6], w * ys[p][j * 8 + 7]);
                *reinterpret_cast<uint4*>(yp + j * 8) = o;
            }
        }
    }
}

// grid (48, maxseg), block 256
template <int EB = g6nr::EB, bool K2 = false, bool BS = false>
__global__ void __launch_bounds__(256, 2) k_pf_fc2(Expert A)
{
    __shared__ __align__(16) unsigned char sA[2 * MB * 4 * REC];
    __shared__ __align__(16) __half sZ[RS * 128];
    __shared__ __align__(16) unsigned char sV[2 * 256];
    const int seg = blockIdx.y;
    if (seg >= *A.n_seg) return;
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    if (tr[4] == 3) fc2_body<3, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
    else if (!K2 || tr[4] == 4) fc2_body<4, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
    else fc2_body<K2 ? 2 : 4, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
}

#ifdef PF_FC2GL
// PF_FC2GL = GL (2, 3, 4, 6, 8): fc2 group loop. CTA = (segment, GL consecutive 128-col groups); the CTA runs the
// fc2_body block loop for each of its groups in turn as one flattened pipeline it = gi * nb + b, so the X2/svh
// cp.async stage and the wide weight words of the next (group, block) are issued during the current one and the
// cold first loads happen once per CTA instead of once per group. Per group the arithmetic and its order are those
// of fc2_body (byte-equal yw). Measurement build; the default build does not compile this.
constexpr int FC2GL = PF_FC2GL;
static_assert(HID / 128 % FC2GL == 0, "PF_FC2GL must divide 48");
template <int BITS, int EB, bool BS = false>
__device__ __forceinline__ void fc2_body_gl(const Expert& A, const long long* tr, int seg, int r0, int rows, unsigned char* sA, __half* sZ,
                                            unsigned char* sV)
{
    constexpr int TB = 32 * BITS;
    constexpr int SA = MB * 4 * REC;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int G0 = blockIdx.x * FC2GL, nb = static_cast<int>(tr[3]), nmb = (rows + 15) / 16, nit = FC2GL * nb;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0), c = lane & 3, r = lane >> 2;
    Tr3Lane<BITS> ld;
    ld.init(lane);
    const bool wide = trw_ok<BITS>() && A.t_wide2;                // K3, and K2 (#65)
    unsigned char* sZb = reinterpret_cast<unsigned char*>(sZ);
    const int pr = warp * 4 + (lane >> 3), q = lane & 7;
    const int np = rows > 32 ? 2 : 1;
    float ys[2][16];
    #pragma unroll
    for (int p = 0; p < 2; ++p)
        #pragma unroll
        for (int j = 0; j < 16; ++j) ys[p][j] = 0.f;
    auto wptr = [&](int G, int b) {
        return A.t_fc2 + tr[1] + static_cast<size_t>(b) * (static_cast<size_t>(KSH) * 8 * TB) + static_cast<size_t>(G * 8 + warp) * 8 * TB;
    };
    // X2 records of block b and the svh slice of (block b, group G) -> stage it & 1
    auto stage = [&](int it, int G, int b) {
        for (int ch = t; ch < MB * 4 * 33; ch += 256)
        {
            const int mb = ch / (4 * 33), rem = ch % (4 * 33);
            pf::cp16(sA + (it & 1) * SA + ch * 16, A.x2 + ((static_cast<size_t>(seg) * MB + mb) * EB + b) * 4 * REC + rem * 16);
        }
        if (t < 16) pf::cp16(sV + (it & 1) * 256 + t * 16, BS ? A.t_sg + HID + G * 128 + t * 8 : A.t_v3 + (static_cast<size_t>(tr[2]) + b) * HID + G * 128 + t * 8);
        pf::cp_commit();
    };
    stage(0, G0, 0);
    u64 cw[3] = {0, 0, 0};
    if (wide)
    {
        #pragma unroll
        for (int n = 0; n < trw_nw<BITS>(); ++n) cw[n] = ld_nc64(wptr(G0, 0) + n * 256 + lane * 8);
    }
    int G = G0, b = 0;
    for (int it = 0; it < nit; ++it)
    {
        const int Gn = b + 1 < nb ? G : G + 1, bn = b + 1 < nb ? b + 1 : 0;
        u32 cv[8][2];
        if (!wide)
        {
            const uint8_t* wb = wptr(G, b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) ld.load(wb + i * TB, cv[i][0], cv[i][1]);
        }
        pf::cp_wait<0>();
        __syncthreads();
        if (it + 1 < nit) stage(it + 1, Gn, bn);
        u64 nw[3] = {0, 0, 0};
        if (wide && it + 1 < nit)
        {
            #pragma unroll
            for (int n = 0; n < trw_nw<BITS>(); ++n) nw[n] = ld_nc64(wptr(Gn, bn) + n * 256 + lane * 8);
        }
        const unsigned char* sAb = sA + (it & 1) * SA;
        float acc[MB][2][4];
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            u32 bf[4];
            if (wide)
            {
                const u32 c0 = trw_chunk<BITS>(cw, 2 * s), c1 = trw_chunk<BITS>(cw, 2 * s + 1);
                const u32 b0 = __shfl_sync(0xffffffffu, c0, (lane + 31) & 31), b1 = __shfl_sync(0xffffffffu, c1, (lane + 31) & 31);
                u32 p0[2], p1[2], q0[2], q1[2];
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b0) << (8 * BITS)) | c0, 7 * BITS, p0, p1);
                Tr3Lane<BITS>::decode_m((static_cast<u64>(b1) << (8 * BITS)) | c1, 7 * BITS, q0, q1);
                tr3_b8f(p0, p1, q0, q1, bf);
            }
            else tr3_b8<BITS>(ld, cv[2 * s], cv[2 * s + 1], bf);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb >= nmb) continue;
                const unsigned char* rec = sAb + (mb * 4 + s) * REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const u32 sa = rec[512 + srow];
                mma_mx8(acc[mb][0], fa, bf[0], bf[1], sa, 127u);
                mma_mx8(acc[mb][1], fa, bf[2], bf[3], sa, 127u);
            }
        }
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                if (mb >= nmb) continue;
                const int row = mb * 16 + r, ch = warp * 2 + h;
                *reinterpret_cast<u32*>(sZb + row * 256 + swz16(ch, row) * 16 + 4 * c) = pack_h2(acc[mb][h][0], acc[mb][h][1]);
                *reinterpret_cast<u32*>(sZb + (row + 8) * 256 + swz16(ch, row + 8) * 16 + 4 * c) = pack_h2(acc[mb][h][2], acc[mb][h][3]);
            }
        __syncthreads();
        const __half* svp = reinterpret_cast<const __half*>(sV + (it & 1) * 256) + q * 16;
        const float scl = BS ? __half2float(A.t_v3b[(static_cast<size_t>(tr[2]) + b) * (HID / 128) + G]) : 1.f;   // BS: sV holds signs
        float sv[16];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint4 sr = *reinterpret_cast<const uint4*>(svp + j * 8);
            const u32 sw[4] = {sr.x, sr.y, sr.z, sr.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float2 s2 = __half22float2(as_h2(sw[e]));
                if (BS) { sv[j * 8 + e * 2] = s2.x * scl; sv[j * 8 + e * 2 + 1] = s2.y * scl; }
                else { sv[j * 8 + e * 2] = s2.x; sv[j * 8 + e * 2 + 1] = s2.y; }
            }
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            if (p >= np || warp * 4 + p * 32 >= rows) continue;      // warp-uniform
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
            #pragma unroll
            for (int j = 0; j < 16; ++j) ys[p][j] += v[j] * sv[j];
        }
        #pragma unroll
        for (int n = 0; n < 3; ++n) cw[n] = nw[n];
        if (b + 1 == nb)      // group G done: store it, restart the accumulators for the next group
        {
            #pragma unroll
            for (int p = 0; p < 2; ++p)
            {
                const int rp = pr + p * 32;
                if (p < np && rp < rows)
                {
                    const float w = A.rr_w[r0 + rp];
                    __half* yp = A.yw + static_cast<size_t>(r0 + rp) * HID + G * 128 + q * 16;
                    #pragma unroll
                    for (int j = 0; j < 2; ++j)
                    {
                        uint4 o;
                        o.x = pack_h2(w * ys[p][j * 8], w * ys[p][j * 8 + 1]);
                        o.y = pack_h2(w * ys[p][j * 8 + 2], w * ys[p][j * 8 + 3]);
                        o.z = pack_h2(w * ys[p][j * 8 + 4], w * ys[p][j * 8 + 5]);
                        o.w = pack_h2(w * ys[p][j * 8 + 6], w * ys[p][j * 8 + 7]);
                        *reinterpret_cast<uint4*>(yp + j * 8) = o;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 16; ++j) ys[p][j] = 0.f;
            }
        }
        G = Gn; b = bn;
    }
}

// grid (48 / GL, maxseg), block 256
template <int EB = g6nr::EB, bool K2 = false, bool BS = false>
__global__ void __launch_bounds__(256, 2) k_pf_fc2g(Expert A)
{
    __shared__ __align__(16) unsigned char sA[2 * MB * 4 * REC];
    __shared__ __align__(16) __half sZ[RS * 128];
    __shared__ __align__(16) unsigned char sV[2 * 256];
    const int seg = blockIdx.y;
    if (seg >= *A.n_seg) return;
    const int* sg = A.segs + static_cast<size_t>(seg) * 3;
    const long long* tr = A.t_tab + static_cast<size_t>(sg[0]) * 6;
    if (tr[4] == 3) fc2_body_gl<3, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
    else if (!K2 || tr[4] == 4) fc2_body_gl<4, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
    else fc2_body_gl<K2 ? 2 : 4, EB, BS>(A, tr, seg, sg[1], sg[2], sA, sZ, sV);
}
#endif

// out[m] = bf16(sum_k yw[pos[m][k]] (k = 0..7 in order) + ysh[m]), fp32 sum. grid M, block 256.
template <bool RS = false, bool SHX = false> __global__ void __launch_bounds__(256) k_pf_fin(Expert A)
{
    const int m = blockIdx.x;
    int p[TOPK];
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) p[k] = A.pos[m * TOPK + k];
    for (int i = threadIdx.x; i < HID / 8; i += 256)
    {
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        auto add = [&](const __half* src) {
            const uint4 q = *reinterpret_cast<const uint4*>(src + i * 8);
            const float2 f0 = __half22float2(as_h2(q.x)), f1 = __half22float2(as_h2(q.y)), f2 = __half22float2(as_h2(q.z)), f3 = __half22float2(as_h2(q.w));
            a[0] += f0.x; a[1] += f0.y; a[2] += f1.x; a[3] += f1.y; a[4] += f2.x; a[5] += f2.y; a[6] += f3.x; a[7] += f3.y;
        };
        #pragma unroll
        for (int k = 0; k < TOPK; ++k)
#ifdef PF_FINLOCAL
            if (p[k] >= 0)        // measurement build: an expert-parallel rank sums only its own routes (pos < 0 = not here)
#endif
            add(A.yw + static_cast<size_t>(p[k]) * HID);
        if constexpr (SHX)
        {
            const uint4 q = *reinterpret_cast<const uint4*>(A.ysh + static_cast<size_t>(m) * HID + i * 8);
            const uint4 sq = *reinterpret_cast<const uint4*>(A.sh_svh + i * 8);
            const u32 qw[4] = {q.x, q.y, q.z, q.w}, sw[4] = {sq.x, sq.y, sq.z, sq.w};
            float v[8], sv[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f = __half22float2(as_h2(qw[j])), g = __half22float2(as_h2(sw[j]));
                v[2 * j] = f.x; v[2 * j + 1] = f.y; sv[2 * j] = g.x; sv[2 * j + 1] = g.y;
            }
            if (A.sh_sc)              // sh_svh = the output signs: times the block's scale
            {
                const float c = __half2float(A.sh_sc[(i * 8) / 128]);
                #pragma unroll
                for (int j = 0; j < 8; ++j) sv[j] *= c;
            }
            had128_l8(v, threadIdx.x & 31);            // 16 lanes x 8 columns = one 128-block (i = threadIdx.x + 256 n)
            #pragma unroll
            for (int j = 0; j < 8; ++j) a[j] += rnd_h(v[j] * sv[j]);
        }
        else add(A.ysh + static_cast<size_t>(m) * HID);
        uint4 o;
        o.x = pack_bf2(a[0], a[1]); o.y = pack_bf2(a[2], a[3]); o.z = pack_bf2(a[4], a[5]); o.w = pack_bf2(a[6], a[7]);
        if constexpr (RS) *reinterpret_cast<uint4*>(rs_row(A.rs, m, HID) + i * 8) = o;
        else *reinterpret_cast<uint4*>(A.out + static_cast<size_t>(m) * HID + i * 8) = o;
    }
}

inline void launch_route(const Expert& A, cudaStream_t st)     // logits -> tok_e, tok_w, the grouped route list, segments
{
    if (!A.on) return;
    k_pf_route<<<(A.M + 7) / 8, 256, 0, st>>>(A);
    k_pf_count<<<NEXP, 256, 0, st>>>(A);
    k_pf_fill<<<NEXP, 256, 0, st>>>(A);
}
// EB: blocks per expert of the scratch layout (scratch_bytes / scratch_at with the same eb); g6nr::EB = 3 (TP6) / 4 (TP4)
#ifdef PF_HOIST
template <int EB = g6nr::EB> inline void launch_fc1(const Expert& A, cudaStream_t st)
{
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(k_pf_fc1h<EB>, cudaFuncAttributeMaxDynamicSharedMemorySize, FC1H_SMEM); attr = true; }
    if (A.on) k_pf_fc1h<EB><<<dim3(2 * fc1h_pairs(EB), A.maxseg), 512, FC1H_SMEM, st>>>(A);
}
#else
template <int EB = g6nr::EB> inline void launch_fc1(const Expert& A, cudaStream_t st)
{
    if (!A.on) return;
    const dim3 g(2 * EB, A.maxseg);
#ifdef PF_F12
    if (A.f12 && A.t_wide)
    {
        static_assert(EB % F12_NE == 0, "F12_NE must divide EB");
        const dim3 gf(2 * (EB / F12_NE), A.maxseg);
        static bool attr = false;
        if (!attr)
        {
            cudaFuncSetAttribute(k_pf_fc1f<EB, false, false, F12_NE>, cudaFuncAttributeMaxDynamicSharedMemorySize, F12_SMEM(F12_NE));
            cudaFuncSetAttribute(k_pf_fc1f<EB, true, false, F12_NE>, cudaFuncAttributeMaxDynamicSharedMemorySize, F12_SMEM(F12_NE));
            cudaFuncSetAttribute(k_pf_fc1f<EB, false, true, F12_NE>, cudaFuncAttributeMaxDynamicSharedMemorySize, F12_SMEM(F12_NE));
            cudaFuncSetAttribute(k_pf_fc1f<EB, true, true, F12_NE>, cudaFuncAttributeMaxDynamicSharedMemorySize, F12_SMEM(F12_NE));
            attr = true;
        }
        if (A.bs) { if (A.k2) k_pf_fc1f<EB, true, true, F12_NE><<<gf, 256, F12_SMEM(F12_NE), st>>>(A); else k_pf_fc1f<EB, false, true, F12_NE><<<gf, 256, F12_SMEM(F12_NE), st>>>(A); }
        else if (A.k2) k_pf_fc1f<EB, true, false, F12_NE><<<gf, 256, F12_SMEM(F12_NE), st>>>(A);
        else k_pf_fc1f<EB, false, false, F12_NE><<<gf, 256, F12_SMEM(F12_NE), st>>>(A);
        return;
    }
#endif
#ifdef PF_AOP
    // below AOP_MMIN rows the record write traffic (8 routes per token, padded segments) costs more than the prologue
    // it saves (1-GPU home L5: +5..10% at M <= 1024, ~0 at 2048, -3..5% at 3072-4096): those passes run the stock fc1 (also byte-exact)
    static const int mmin = [] { const char* e = getenv("G6_PF_AOP_MMIN"); return e ? atoi(e) : 3072; }();
    if (A.aop && A.M >= mmin)
    {
        const dim3 ga(2 * AOP_NS * AOP_NCH, A.maxseg);
        if (A.bs) k_pf_aop<true><<<ga, 256, 0, st>>>(A);
        else k_pf_aop<false><<<ga, 256, 0, st>>>(A);
        if (A.bs) { if (A.k2) k_pf_fc1<EB, true, true, true><<<g, 256, 0, st>>>(A); else k_pf_fc1<EB, false, true, true><<<g, 256, 0, st>>>(A); }
        else if (A.k2) k_pf_fc1<EB, true, false, true><<<g, 256, 0, st>>>(A);
        else k_pf_fc1<EB, false, false, true><<<g, 256, 0, st>>>(A);
        return;
    }
#endif
    if (A.bs) { if (A.k2) k_pf_fc1<EB, true, true><<<g, 256, 0, st>>>(A); else k_pf_fc1<EB, false, true><<<g, 256, 0, st>>>(A); }
    else if (A.k2) k_pf_fc1<EB, true><<<g, 256, 0, st>>>(A);
    else k_pf_fc1<EB><<<g, 256, 0, st>>>(A);
}
#endif
template <int EB = g6nr::EB> inline void launch_mid(const Expert& A, cudaStream_t st)
{
    if (A.on) k_pf_mid<EB><<<((A.nroutes ? A.nroutes : A.M * TOPK) * EB + 7) / 8, 256, 0, st>>>(A);
}
#ifdef PF_FC2GL
template <int EB = g6nr::EB> inline void launch_fc2(const Expert& A, cudaStream_t st)
{
    if (!A.on) return;
    const dim3 g(HID / 128 / FC2GL, A.maxseg);
    if (A.bs) { if (A.k2) k_pf_fc2g<EB, true, true><<<g, 256, 0, st>>>(A); else k_pf_fc2g<EB, false, true><<<g, 256, 0, st>>>(A); }
    else if (A.k2) k_pf_fc2g<EB, true><<<g, 256, 0, st>>>(A);
    else k_pf_fc2g<EB><<<g, 256, 0, st>>>(A);
}
#else
template <int EB = g6nr::EB> inline void launch_fc2(const Expert& A, cudaStream_t st)
{
    if (!A.on) return;
    const dim3 g(HID / 128, A.maxseg);
    if (A.bs) { if (A.k2) k_pf_fc2<EB, true, true><<<g, 256, 0, st>>>(A); else k_pf_fc2<EB, false, true><<<g, 256, 0, st>>>(A); }
    else if (A.k2) k_pf_fc2<EB, true><<<g, 256, 0, st>>>(A);
    else k_pf_fc2<EB><<<g, 256, 0, st>>>(A);
}
#endif
inline void launch_fin(const Expert& A, cudaStream_t st)
{
    if (!A.on) return;
    if (A.sh_svh)
    {
        if (A.rs.base) k_pf_fin<true, true><<<A.M, 256, 0, st>>>(A);
        else k_pf_fin<false, true><<<A.M, 256, 0, st>>>(A);
    }
    else if (A.rs.base) k_pf_fin<true><<<A.M, 256, 0, st>>>(A);
    else k_pf_fin<<<A.M, 256, 0, st>>>(A);
}
// ------------------------------------------------------------------------------------------ trellis shared expert
// With its FP8 copy (bind::make_shared_fp8: the decoded tiles, still rotated), the shared expert runs on the MXFP8 GEMMs
// of the dense shared path, wrapped in its rotations. k_shx_rotq writes the records of H128(x * suh_gate) and
// H128(x * suh_up) in one pass. The gate and up GEMMs run as one pair. k_shx_mid (the arithmetic of k_pf_mid) writes
// the down input records, the down GEMM writes ysh, and k_pf_fin adds f16(H128(ysh) * svh_down).
struct ShX
{
    int on = 0, M = 0, I = 0;
    const __half* x = nullptr;        // [M][6144] mlp_norm output
    uint8_t *x8g = nullptr, *x8u = nullptr;   // MXFP8 records of the rotated gate and up inputs (gate's: the GEMMs' x8_big)
    uint8_t* x8a = nullptr;           // MXFP8 records of the down input (x8_small)
    const __half *suh_g = nullptr, *suh_u = nullptr;    // [6144] input-side vectors (block row 0 of row 256), or with sc_g:
    const __half *sc_g = nullptr, *sc_u = nullptr;      // the signs (suh_g = suh_u) and the 48 block scales of gate and up
    const __half* v2 = nullptr;       // [I / 128][3][128] svh gate, svh up, suh down of row 256's blocks
    const __half *g = nullptr, *u = nullptr;            // gate, up GEMM outputs [M][I]
};

// One warp's 4 values (k = 32 s + 4 (lane % 8) + i of a 32-k group, s = the warp's group lane / 8) of row `row` ->
// MXFP8 records [16-row block][K/32][REC] (the k_xq8 layout: u32 of lane (row % 8) * 4 + c, register half * 2 + (row % 16
// >= 8); scale byte 512 + row % 16). Rows >= M write zero codes (the GEMMs read the whole last block).
__device__ __forceinline__ void shx_store_mx8(const float (&v)[4], int lane, int row, int M, int S, int s, uint8_t* x8)
{
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
    float inv;
    u32 code = mx_scale(am, inv), c = f2_e4m3x2(v[0] * inv, v[1] * inv) | (f2_e4m3x2(v[2] * inv, v[3] * inv) << 16);
    if (row >= M) { c = 0u; code = 127u; }
    const int j = lane & 7, rr = row & 15;
    uint8_t* rec = x8 + (static_cast<size_t>(row >> 4) * S + s) * REC;
    *reinterpret_cast<u32*>(rec + ((rr & 7) * 4 + (j & 3)) * 16 + ((j >> 2) * 2 + (rr >> 3)) * 4) = c;
    if (j == 0) rec[512 + rr] = static_cast<uint8_t>(code);
}

// x -> MXFP8 records of H128(x * suh_g) and H128(x * suh_u), quantized from fp32 with k_xq8's scale rule. The trellis
// fc1 prologue quantizes from fp16 values with its own rule, so the two paths differ by FP8 activation rounding
// (cos 0.99976 at M = 20). Against the fp16-activation reference this one is as close or closer (#65).
// grid ceil(ceil(M/16)*16 * 48 / 8), block 256: warp = (row, 128-block)
__global__ void __launch_bounds__(256) k_shx_rotq(ShX a)
{
    const int lane = threadIdx.x & 31;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / 48, blk = w % 48;
    if (row >= (a.M + 15) / 16 * 16) return;
    const int k0 = blk * 128 + lane * 4;
    float x[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < a.M)
    {
        const uint2 xr = *reinterpret_cast<const uint2*>(a.x + static_cast<size_t>(row) * HID + k0);
        const float2 x01 = __half22float2(as_h2(xr.x)), x23 = __half22float2(as_h2(xr.y));
        x[0] = x01.x; x[1] = x01.y; x[2] = x23.x; x[3] = x23.y;
    }
    #pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        uint2 sr = *reinterpret_cast<const uint2*>((j ? a.suh_u : a.suh_g) + k0);
        if (a.sc_g)                   // signs x block scale: the stored vector's fp16 values
        {
            const __half2 c = __half2half2((j ? a.sc_u : a.sc_g)[blk]);
            sr.x = as_u32(__hmul2(as_h2(sr.x), c)); sr.y = as_u32(__hmul2(as_h2(sr.y), c));
        }
        const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
        float v[4] = {__fmul_rn(x[0], s01.x), __fmul_rn(x[1], s01.y), __fmul_rn(x[2], s23.x), __fmul_rn(x[3], s23.y)};
        had128(v, lane);
        shx_store_mx8(v, lane, row, a.M, HID / 32, blk * 4 + (lane >> 3), j ? a.x8u : a.x8g);
    }
}

// g, u [M][I] -> the down input records: the arithmetic of k_pf_mid, quantized from its fp16-rounded values.
// grid ceil(ceil(M/16)*16 * I/128 / 8), block 256: warp = (row, 128-block)
__global__ void __launch_bounds__(256) k_shx_mid(ShX a)
{
    const int lane = threadIdx.x & 31, nb = a.I / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nb, b = w % nb;
    if (row >= (a.M + 15) / 16 * 16) return;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < a.M)
    {
        const size_t o = static_cast<size_t>(row) * a.I + b * 128 + lane * 4;
        const __half* v2 = a.v2 + static_cast<size_t>(b) * 3 * 128 + lane * 4;
        auto f4 = [](const void* p, float (&r)[4]) {
            const uint2 q = *reinterpret_cast<const uint2*>(p);
            const float2 x = __half22float2(as_h2(q.x)), y = __half22float2(as_h2(q.y));
            r[0] = x.x; r[1] = x.y; r[2] = y.x; r[3] = y.y;
        };
        float gv[4], uv[4], svg[4], svu[4], sud[4];
        f4(a.g + o, gv); f4(a.u + o, uv); f4(v2, svg); f4(v2 + 128, svu); f4(v2 + 256, sud);
        had128(gv, lane);
        had128(uv, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float gg = rnd_h(__fmul_rn(gv[e], svg[e])), uu = rnd_h(__fmul_rn(uv[e], svu[e]));
            v[e] = __fmul_rn(rnd_h(silu_f(gg) * uu), sud[e]);
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = rnd_h(v[e]);
    }
    shx_store_mx8(v, lane, row, a.M, a.I / 32, b * 4 + (lane >> 3), a.x8a);
}

// The trellis shared expert (exlr8 v2 tr3.table row 256, #65) without an FP8 copy runs as its own pass of the routed kernels. Every row is
// one route of expert 256 with weight 1, in segments of RS rows. fc2 writes the rows straight into ysh (S.yw = ysh).
// k_pf_fin adds ysh as it adds the MXFP8 shared output. grid ceil(M / 256), block 256.
__global__ void __launch_bounds__(256) k_pf_sh_routes(Expert S)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < S.M) { S.rr_row[i] = i; S.rr_e[i] = NEXP; S.rr_w[i] = 1.f; S.rr_sp[i] = i; }
    const int nseg = (S.M + RS - 1) / RS;
    if (i < nseg)
    {
        int* sg = S.segs + static_cast<size_t>(i) * 3;
        sg[0] = NEXP; sg[1] = i * RS; sg[2] = min(RS, S.M - i * RS);
    }
    if (i == 0) *S.n_seg = nseg;
}
template <int EB = g6nr::EB> inline void launch_shared(const Expert& S, cudaStream_t st)
{
    if (!S.on) return;
    k_pf_sh_routes<<<(S.M + 255) / 256, 256, 0, st>>>(S);
    launch_fc1<EB>(S, st);
    launch_mid<EB>(S, st);
    launch_fc2<EB>(S, st);
}

template <int EB = g6nr::EB> inline void launch_experts(const Expert& A, cudaStream_t st)
{
    launch_route(A, st);
    launch_fc1<EB>(A, st);
    launch_mid<EB>(A, st);
    launch_fc2<EB>(A, st);
    launch_fin(A, st);
}

// ------------------------------------------------------------------------------------------ scratch (device memory)
struct Scratch
{
    int Mmax = 0;
    uint8_t* x8_big;                  // MXFP8 records of a 6144-wide input (x, then x2)
    uint8_t* x8_up;                   // the same size: the trellis shared expert's rotated up input (#65)
    uint8_t* x8_small;                // same, up to SMALLW wide (q_a_norm output, attention output, MLP activation; 2816 / 4096)
    uint16_t* xb;                     // [M][6144] BF16 copy of x / x2
    float* y32;                       // [M][256] fp32 staging of idx.wk / idx.weights_proj
    __half *g, *u, *act;              // [M][2048] shared / dense gate, up, activation
    __half* ysh;                      // [M][6144] shared expert output
    int *tok_e, *cnt, *rr_row, *rr_e, *rr_sp, *pos, *segs, *n_seg;
    float *tok_w, *rr_w;
    int *sh_rr_row, *sh_rr_e, *sh_rr_sp, *sh_segs, *sh_n_seg;      // the trellis shared pass's routes (#65)
    float* sh_rr_w;
    __half* z1;
    uint8_t* x2;
    __half* yw;
#ifdef PF_AOP
    uint8_t* aop;                     // [maxseg][AOP_NS][2][AOPS] (Expert::aop)
#endif
};

inline size_t scratch_layout(Scratch* s, char* base, int M, int eb = g6nr::EB)     // eb = blocks per expert (EB of the launches)
{
    size_t o = 0;
    auto take = [&](size_t bytes) { char* p = base ? base + o : nullptr; o += (bytes + 255) / 256 * 256; return p; };
    const size_t RB = (static_cast<size_t>(M) + 15) / 16, m = static_cast<size_t>(M), R = m * TOPK, ms = static_cast<size_t>(max_segments(M));
    Scratch t;
    t.Mmax = M;
    t.x8_big = reinterpret_cast<uint8_t*>(take(RB * (HID / 32) * REC));
    t.x8_up = reinterpret_cast<uint8_t*>(take(RB * (HID / 32) * REC));
    t.x8_small = reinterpret_cast<uint8_t*>(take(RB * (g6nr::SMALLW / 32) * REC));
    t.xb = reinterpret_cast<uint16_t*>(take(m * HID * 2));
    t.y32 = reinterpret_cast<float*>(take(m * 256 * 4));
    t.g = reinterpret_cast<__half*>(take(m * SACTW * 2));
    t.u = reinterpret_cast<__half*>(take(m * SACTW * 2));
    t.act = reinterpret_cast<__half*>(take(m * SACTW * 2));
    t.ysh = reinterpret_cast<__half*>(take(m * HID * 2));
    t.tok_e = reinterpret_cast<int*>(take(R * 4));
    t.tok_w = reinterpret_cast<float*>(take(R * 4));
    t.cnt = reinterpret_cast<int*>(take(NEXP * 4));
    t.rr_row = reinterpret_cast<int*>(take(R * 4));
    t.rr_e = reinterpret_cast<int*>(take(R * 4));
    t.rr_sp = reinterpret_cast<int*>(take(R * 4));
    t.rr_w = reinterpret_cast<float*>(take(R * 4));
    t.pos = reinterpret_cast<int*>(take(R * 4));
    t.segs = reinterpret_cast<int*>(take(ms * 3 * 4));
    t.n_seg = reinterpret_cast<int*>(take(4));
    t.sh_rr_row = reinterpret_cast<int*>(take(m * 4));
    t.sh_rr_e = reinterpret_cast<int*>(take(m * 4));
    t.sh_rr_sp = reinterpret_cast<int*>(take(m * 4));
    t.sh_rr_w = reinterpret_cast<float*>(take(m * 4));
    t.sh_segs = reinterpret_cast<int*>(take((m + RS - 1) / RS * 3 * 4));
    t.sh_n_seg = reinterpret_cast<int*>(take(4));
    t.z1 = reinterpret_cast<__half*>(take(R * 2 * eb * 128 * 2));
    t.x2 = reinterpret_cast<uint8_t*>(take(ms * MB * eb * 4 * REC));
    t.yw = reinterpret_cast<__half*>(take(R * HID * 2));
#ifdef PF_AOP
    t.aop = reinterpret_cast<uint8_t*>(take(ms * AOP_NS * 2 * AOPS));
#endif
    if (s) *s = t;
    return o;
}
// One block per rank for prefill sub-batches of up to Mmax rows: about 161 KB per row + 6.6 MB (1024 rows: 172 MB).
inline size_t scratch_bytes(int Mmax, int eb = g6nr::EB) { return scratch_layout(nullptr, nullptr, Mmax, eb); }
inline Scratch scratch_at(void* block, int Mmax, int eb = g6nr::EB)
{
    Scratch s;
    scratch_layout(&s, static_cast<char*>(block), Mmax, eb);
    return s;
}

// ------------------------------------------------------------------------------------------ one layer of one rank
struct Io                             // step buffers of one rank, device memory, rows contiguous at the stated width
{
    const __half* x = nullptr;        // in  [M][6144] attn_norm output
    __half* q_a = nullptr;            // out [M][2048]
    const __half* q_a_n = nullptr;    // in  [M][2048] q_a_norm output
    __half* kv_a = nullptr;           // out [M][576]
    __half* q_b = nullptr;            // out [M][H*256]
    __half* idx_q = nullptr;          // out [M][4096]   idx.wq_b   (indexer layers)
    __half* idx_k = nullptr;          // out [M][128]    idx.wk
    __half* idx_w = nullptr;          // out [M][32]     idx.weights_proj
    const __half* attn = nullptr;     // in  [M][H*256] attention output
    uint16_t* attn_part = nullptr;    // out [M][6144] BF16   o_proj partial
    const __half* x2 = nullptr;       // in  [M][6144] mlp_norm output
    float* logits = nullptr;          // out [M][256] fp32 router logits (MoE layers)
    uint16_t* mlp_part = nullptr;     // out [M][6144] BF16   MLP partial
};

struct Layer
{
    int M = 0, moe = 0, indexer = 0, heads = 0, inter = 0;      // inter = shared / dense MLP width on this rank
    Xq xq_x, xq_qa, xq_attn, xq_x2, xq_act;
    Gemm q_a, kv_a, q_b, idx_q, o_proj, gate, up, down;
    RsOut rs_o, rs_down;              // G6_PF_RSDIRECT (prefill.cuh): o_proj / dense down rows -> RS send slot
    ToBf xb_x, xb_x2;
    Blas idx_k, idx_w, router;
    Act act;
    Expert ex;
    Expert exs;                       // trellis shared expert pass (launch_shared, #65); on = 0 otherwise
    ShX shx;                          // the same from its FP8 copy on the MXFP8 GEMMs (gate, up, down above); on = 0 otherwise
};

// the shared expert of an exlr8 v2 layer (#65): FP8 copy on the GEMMs, or the trellis pass. No-op for other layers.
// Their shared / dense MLP runs from the five launches of the step order above.
template <int EB = g6nr::EB> inline void launch_shared(const Layer& L, cudaStream_t st)
{
    if (L.shx.on)
    {
        const ShX& a = L.shx;
        const int rows = (a.M + 15) / 16 * 16;
        k_shx_rotq<<<(rows * 48 + 7) / 8, 256, 0, st>>>(a);
        launch_pair(L.gate, L.up, st);    // up reads its own records (Gemm::x8_2)
        k_shx_mid<<<(rows * (a.I / 128) + 7) / 8, 256, 0, st>>>(a);
        launch(L.down, st);           // -> ysh, which k_pf_fin rotates and scales (Expert::sh_svh)
    }
    else launch_shared<EB>(L.exs, st);
}

// Dense trellis matrices (Tensors::tsuh): the GEMMs on their FP8 tile copies inside the rotations. q_b's rotated input
// goes to x8_up, and xq_qa stays unrotated for idx_q. xq_attn rotates o_proj's input. Dense MLP: gate on xq_x2's
// records, up on its own (x8_up), the mid pass writes down's records (Act::svh_g), down rotates its output.
inline const char* dense_trellis(const Tensors& T, int M, const Scratch& sc, const Io& io, Layer& L)
{
    const bool any = T.tsuh[0] || T.tsuh[1] || T.tsuh[2] || T.tsuh[3] || T.tsuh[4];
    if (!any) return nullptr;
    for (int i = 0; i < dtr::NMAT; ++i)
        if (!T.tsuh[i] != !T.tsvh[i]) return "dense trellis matrix without its suh / svh";
    auto rot = [&](const __half* src, int ld, int K, uint8_t* dst, const __half* suh) {
        Xq q; q.on = 1; q.src = src; q.ld = ld; q.K = K; q.M = M; q.dst = dst; q.suh = suh;
        return q;
    };
    if (T.tsuh[0] && L.q_b.on)
    {
        if (L.q_b.N % 128 || L.q_b.K % 128) return "dense trellis q_b: widths must be multiples of 128";
        L.q_b.pre = rot(io.q_a_n, L.q_b.K, L.q_b.K, sc.x8_up, T.tsuh[0]);
        L.q_b.x8 = sc.x8_up;
        L.q_b.osvh = T.tsvh[0];
    }
    if (T.tsuh[1] && L.o_proj.on)
    {
        if (L.o_proj.N % 128 || L.o_proj.K % 128) return "dense trellis o_proj: widths must be multiples of 128";
        L.xq_attn.suh = T.tsuh[1];
        L.o_proj.osvh = T.tsvh[1];
    }
    if (T.tsuh[2] || T.tsuh[3] || T.tsuh[4])
    {
        if (L.moe || !T.tsuh[2] || !T.tsuh[3] || !T.tsuh[4] || !L.gate.on || !L.up.on || !L.down.on)
            return "dense trellis MLP: gate, up and down together, on a dense layer";
        if (L.inter % 128) return "dense trellis MLP: width must be a multiple of 128";
        L.xq_x2.suh = T.tsuh[2];
        L.up.pre = rot(io.x2, HID, HID, sc.x8_up, T.tsuh[3]);
        L.up.x8 = sc.x8_up;
        L.act.svh_g = T.tsvh[2]; L.act.svh_u = T.tsvh[3]; L.act.suh_d = T.tsuh[4]; L.act.x8 = sc.x8_small;
        L.xq_act.on = 0;
        L.down.osvh = T.tsvh[4];
    }
    return nullptr;
}

inline const char* make_layer(const Tensors& T, int M, const Scratch& sc, const Io& io, cublasHandle_t blas, Layer& L)
{
    L = Layer();
    if (M < 1 || M > sc.Mmax) return "M must be 1..Mmax of the scratch";
    L.M = M;
    L.moe = T.t[T_TR3_TAB].p != nullptr;
    L.indexer = T.t[T_IDX_WQ_B].p != nullptr;
    const int GT = L.moe ? T_S_GATE : T_D_GATE, UT = L.moe ? T_S_UP : T_D_UP, DT = L.moe ? T_S_DOWN : T_D_DOWN;
    const char* e = nullptr;
    //                  codes            scales           records       output                              BF16
    if ((e = make_gemm(T.t[T_Q_A],      T.s[T_Q_A],      M, sc.x8_big,   io.q_a,                             0, L.q_a))) return e;
    if ((e = make_gemm(T.t[T_KV_A],     T.s[T_KV_A],     M, sc.x8_big,   io.kv_a,                            0, L.kv_a))) return e;
    if ((e = make_gemm(T.t[T_Q_B],      T.s[T_Q_B],      M, sc.x8_small, io.q_b,                             0, L.q_b))) return e;
    if ((e = make_gemm(T.t[T_IDX_WQ_B], T.s[T_IDX_WQ_B], M, sc.x8_small, io.idx_q,                           0, L.idx_q))) return e;
    if ((e = make_gemm(T.t[T_O_PROJ],   T.s[T_O_PROJ],   M, sc.x8_small, io.attn_part,                       1, L.o_proj))) return e;
    // a dense trellis o_proj without an FP8 copy (dtr_find.h): the GEMM decodes its tiles (bit-equal to the copy)
    if (!L.o_proj.on && T.tm.o_proj.w && (e = make_gemm_tr(T.tm.o_proj, M, sc.x8_small, io.attn_part, 1, L.o_proj))) return e;
    if ((e = make_gemm(T.t[GT],         T.s[GT],         M, sc.x8_big,   sc.g,                               0, L.gate))) return e;
    if ((e = make_gemm(T.t[UT],         T.s[UT],         M, sc.x8_big,   sc.u,                               0, L.up))) return e;
    if ((e = make_gemm(T.t[DT],         T.s[DT],         M, sc.x8_small, L.moe ? static_cast<void*>(sc.ysh) : static_cast<void*>(io.mlp_part), !L.moe, L.down))) return e;
#if PF_QO_DEDUP
    // slim pack (tools/convert/slimpack.py): q_b / o_proj .mx.codes + .scales are left out where the QO pack covers the layer.
    // Both absent together is fine only then; anything else fails loud.
    const bool qo_dedup = !L.q_b.on && !L.o_proj.on && T.qo_cover && T.heads > 0;
    if (!L.q_a.on || !L.kv_a.on || (!qo_dedup && (!L.q_b.on || !L.o_proj.on)))
        return !L.q_b.on && !L.o_proj.on && !T.qo_cover
                   ? "layer without q_b / o_proj .mx.codes and no QO pack covers it (slim pack booted without G6_PF_OWN=3 / qo pack?)"
                   : "layer without q_a / kv_a / q_b / o_proj .mx.codes";
#else
    const bool qo_dedup = false;
    if (!L.q_a.on || !L.kv_a.on || !L.q_b.on || !L.o_proj.on) return "layer without q_a / kv_a / q_b / o_proj .mx.codes";
#endif
    const bool shtr = L.moe && T.sh_trellis;          // shared expert as trellis row 256: gate / up / down may be absent
    if ((!L.gate.on || !L.up.on || !L.down.on) && !shtr)
        return L.moe ? "MoE layer without shared.gate/up/down.mx" : "layer has neither tr3.table nor dense.gate/up/down.mx";
    if (L.q_a.K != HID || L.kv_a.K != HID || L.q_a.N > g6nr::SMALLW)
        return "attention projection shapes do not chain (q_a -> q_b, q_b width = o_proj K <= QBW)";
    if (!qo_dedup && (L.q_b.K != L.q_a.N || L.o_proj.N != HID || L.o_proj.K != L.q_b.N || L.q_b.N % 256 || L.q_b.N > g6nr::QBW))
        return "attention projection shapes do not chain (q_a -> q_b, q_b width = o_proj K <= QBW)";
    if (L.indexer && L.idx_q.K != L.q_a.N) return "idx.wq_b input width is not the q_a width";
    L.heads = qo_dedup ? T.heads : L.q_b.N / 256;
    L.inter = L.gate.N;
    if (L.gate.on && (L.gate.K != HID || L.up.K != HID || L.up.N != L.inter || L.down.K != L.inter || L.down.N != HID || L.inter > SACTW ||
                      L.inter % 32))
        return "shared / dense MLP shapes are not gate, up [I][6144] + down [6144][I] with I <= SACTW";

    auto xq = [&](Xq& q, const __half* src, int K, uint8_t* dst) { q.on = 1; q.src = src; q.ld = K; q.K = K; q.M = M; q.dst = dst; };
    xq(L.xq_x, io.x, HID, sc.x8_big);
    xq(L.xq_qa, io.q_a_n, L.q_a.N, sc.x8_small);
    xq(L.xq_attn, io.attn, qo_dedup ? T.heads * 256 : L.o_proj.K, sc.x8_small);   // not launched on the QO path
    xq(L.xq_x2, io.x2, HID, sc.x8_big);
    xq(L.xq_act, sc.act, L.inter, sc.x8_small);
    L.act.on = 1; L.act.g = sc.g; L.act.u = sc.u; L.act.act = sc.act; L.act.I = L.inter; L.act.M = M;
    if (const char* e = dense_trellis(T, M, sc, io, L)) return e;

    if ((e = make_blas(T.t[T_IDX_WK],    HID, M, blas, sc.xb, sc.y32,                         io.idx_k, L.idx_k))) return e;
    if ((e = make_blas(T.t[T_IDX_WPROJ], HID, M, blas, sc.xb, sc.y32 + static_cast<size_t>(M) * 128, io.idx_w, L.idx_w))) return e;
    if ((e = make_blas(T.t[T_ROUTER],    HID, M, blas, sc.xb, io.logits,                      nullptr,  L.router))) return e;
    if (L.moe != L.router.on) return "router.w and tr3.table must come together";
    if (L.idx_k.on && (L.idx_k.O != 128 || !L.idx_w.on || L.idx_w.O != 32)) return "idx.wk / idx.weights_proj are not [128][6144] + [32][6144]";
    if (L.router.on && L.router.O != NEXP) return "router.w is not [256][6144]";
    L.xb_x.on = L.idx_k.on;  L.xb_x.src = io.x;   L.xb_x.ld = HID;  L.xb_x.K = HID;  L.xb_x.M = M;  L.xb_x.dst = sc.xb;
    L.xb_x2.on = L.moe;      L.xb_x2.src = io.x2; L.xb_x2.ld = HID; L.xb_x2.K = HID; L.xb_x2.M = M; L.xb_x2.dst = sc.xb;

    if (L.moe)
    {
        const Tensor &tab = T.t[T_TR3_TAB], &v1 = T.t[T_TR3_V1], &v2 = T.t[T_TR3_V2], &v3 = T.t[T_TR3_V3], &bias = T.t[T_ROUTER_BIAS];
        const Tensor &sgn = T.t[T_TR3_SG], &v1b = T.t[T_TR3_V1B], &v3b = T.t[T_TR3_V3B];
        const bool bsf = v1b.p != nullptr;           // block-scale hidden-side vectors (exlr8 v2, with the trellis shared expert)
        if (!T.t[T_TR3_FC1].p || !T.t[T_TR3_FC2].p || !v2.p || !bias.p || (bsf ? !sgn.p || !v3b.p : !v1.p || !v3.p))
            return "MoE layer without tr3.fc1/fc2/v2, v1/v3 (or sg/v1b/v3b) or router.bias";
        if (bsf != (T.sh_trellis != 0)) return "block-scale hidden-side vectors go with the trellis shared expert, and only with it";
        const long long nrow = v2.shape[0];
        const Tensor& km = T.t[T_TR3_KMIX];
        if (km.p && (km.nd != 1 || km.shape[0] != tab.shape[0])) return "tr3.kmix is not one entry per tr3.table row";
#ifdef PF_HOIST
        if (km.p || T.k2 || T.t[T_TR3_V1B].p) return "PF_HOIST (k_pf_fc1h) takes TR3 / v1 layers only (no tr3.kmix, 2-bit or block-scale layers)";
#endif
        // 257 rows: an exlr8 v2 pack with the shared expert as trellis row 256 (#65)
        if (tab.nd != 2 || (tab.shape[0] != NEXP && tab.shape[0] != NEXP + 1) || tab.shape[1] != 6 || v2.nd != 3 || v2.shape[1] != 3 ||
            v2.shape[2] != 128 || bias.shape[0] != NEXP ||
            (bsf ? sgn.nd != 2 || sgn.shape[0] != 2 || sgn.shape[1] != HID || v1b.nd != 3 || v1b.shape[0] != nrow || v1b.shape[1] != 2 ||
                       v1b.shape[2] != HID / 128 || v3b.nd != 2 || v3b.shape[0] != nrow || v3b.shape[1] != HID / 128
                 : v1.nd != 3 || v1.shape[1] != 2 || v1.shape[2] != HID || v3.nd != 2 || v3.shape[1] != HID || v1.shape[0] != nrow ||
                       v3.shape[0] != nrow))
            return "tr3 table / v1 / v2 / v3 (or sg / v1b / v3b) / router.bias shapes do not match the pack layout";
        Expert& A = L.ex;
        A.on = 1; A.M = M; A.maxseg = max_segments(M);
        A.logits = io.logits; A.bias = static_cast<const float*>(bias.p);
        A.tok_e = sc.tok_e; A.tok_w = sc.tok_w; A.cnt = sc.cnt;
        A.rr_row = sc.rr_row; A.rr_e = sc.rr_e; A.rr_sp = sc.rr_sp; A.rr_w = sc.rr_w; A.pos = sc.pos; A.segs = sc.segs; A.n_seg = sc.n_seg;
        A.t_tab = static_cast<const long long*>(tab.p);
        A.t_kmix = static_cast<const long long*>(km.p);
        A.k2 = T.k2;
        A.t_fc1 = static_cast<const uint8_t*>(T.t[T_TR3_FC1].p); A.t_wide = T.fc1_wide; A.t_fc2 = static_cast<const uint8_t*>(T.t[T_TR3_FC2].p); A.t_wide2 = T.fc2_wide;
        A.t_v1 = static_cast<const __half*>(v1.p); A.t_v2 = static_cast<const __half*>(v2.p); A.t_v3 = static_cast<const __half*>(v3.p);
        A.bs = bsf;
        A.t_sg = static_cast<const __half*>(sgn.p); A.t_v1b = static_cast<const __half*>(v1b.p); A.t_v3b = static_cast<const __half*>(v3b.p);
        A.x = io.x2; A.ldx = HID;
        A.z1 = sc.z1; A.x2 = sc.x2; A.yw = sc.yw; A.ysh = sc.ysh; A.out = io.mlp_part;
#ifdef PF_AOP
        { const char* ev = getenv("G6_PF_AOP"); A.aop = (ev && atoi(ev) == 0) ? nullptr : sc.aop; }   // compiled in: on unless G6_PF_AOP=0
#endif
#ifdef PF_F12
        { const char* ev = getenv("G6_PF_F12"); A.f12 = ev ? atoi(ev) : 0; }   // default off
#endif
        if (T.sh_trellis && T.sh_fp8)
        {
            if (tab.shape[0] != NEXP + 1) return "trellis shared expert without tr3.table row 256";
            if (!L.gate.on || L.inter != static_cast<int>(T.sh_tr[3]) * 128) return "shared FP8 copy does not match tr3.table row 256";
            ShX& X = L.shx;
            X.on = 1; X.M = M; X.I = L.inter;
            X.x = io.x2; X.x8g = sc.x8_big; X.x8u = sc.x8_up; X.x8a = sc.x8_small;
            const size_t row = static_cast<size_t>(T.sh_tr[2]);
            if (bsf)
            {
                X.suh_g = X.suh_u = A.t_sg; X.sc_g = A.t_v1b + row * 2 * (HID / 128); X.sc_u = X.sc_g + HID / 128;
                A.sh_svh = A.t_sg + HID; A.sh_sc = A.t_v3b + row * (HID / 128);
            }
            else
            {
                X.suh_g = A.t_v1 + row * 2 * HID; X.suh_u = X.suh_g + HID;
                A.sh_svh = A.t_v3 + row * HID;    // k_pf_fin: ysh = f16(H128(down output) x svh_down)
            }
            X.v2 = A.t_v2 + row * 3 * 128;
            X.g = sc.g; X.u = sc.u;
            L.up.x8 = sc.x8_up;
            L.xq_x2.on = 0; L.act.on = 0; L.xq_act.on = 0;
        }
        else if (T.sh_trellis)
        {
            if (tab.shape[0] != NEXP + 1) return "trellis shared expert without tr3.table row 256";
            Expert& S = L.exs;
            S = A;
            S.nroutes = M; S.maxseg = (M + RS - 1) / RS;
            S.rr_row = sc.sh_rr_row; S.rr_e = sc.sh_rr_e; S.rr_sp = sc.sh_rr_sp; S.rr_w = sc.sh_rr_w;
            S.segs = sc.sh_segs; S.n_seg = sc.sh_n_seg; S.pos = nullptr;
#ifdef PF_AOP
            S.aop = nullptr;          // the trellis shared pass keeps the in-CTA prologue (no shared scratch slot)
#endif
#ifdef PF_F12
            S.f12 = 0;
#endif
            S.yw = sc.ysh;                // weight 1: the pass writes the shared output rows
            // ysh comes from the pass above, so the MXFP8 shared GEMMs stay off
            L.xq_x2.on = 0; L.gate.on = 0; L.up.on = 0; L.act.on = 0; L.xq_act.on = 0; L.down.on = 0;
        }
    }
    return nullptr;
}

}   // namespace bindp
