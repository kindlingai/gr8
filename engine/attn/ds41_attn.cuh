// DeepSeek-V4.1-Flash-1017 attention block kernels (attention A): q_a/q_b, window KV (fp8-valued ring), the ratio-2
// compressor, sparse attention with a learned per-head sink, grouped o_a, o_b.
//
// Semantics are the release's own inference code (Attention, Compressor, sparse_attn, act_quant, fp4_act_quant, fp8_gemm);
// engine/ref/ds41/attn_ref.py is the torch twin and every stage here has a buffer it can be compared against
// (engine/attn/tests/ds41_attn_test.cu). Header-only; no gr8 runtime dependency, so a step driver includes it and calls
// ds41a::layer_forward, or the kernels one by one.
//
// Conventions
//   y          bf16 [M, 5120], the sub-layer input already RMS-normed (eps 1e-20) by the caller.
//   fp8 matrix = E4M3 codes u8 [N][K] row-major + E8M0 scale bytes u8 [N][K/32], the block-row scale repeated over its 32 rows
//                (the pack's form: P.mx.codes / P.mx.scales; the checkpoint stores [N/32][K/32]).
//   activation = per row, per 32 values: scale 2^ceil(log2(max(amax, 1e-4) / 448)) (an E8M0 byte), E4M3 codes.
//   fp8 gemm   = per 32-wide K block: fp32 partial sum of exact fp8 products, times scale_a * scale_w, added into an fp32
//                accumulator in K order. Correct, not fast: the engine's GEMV records replace it where decode speed counts.
//   q, o       bf16 [rows, heads, 512] (heads = this rank's heads, a multiple of 16; the o groups are heads / 8).
//   window KV  bf16 ring [R][512] per layer (Cache::ring_rows, R >= 128), slot = position % R, values already quantized to fp8 and
//   back. R = 128 is exact for single-row decode; a speculative verify chunk of M rows writes M slots before acceptance, so it needs
//   R >= 128 + M - 1 (the evicted position of a rejected row is still inside the window of the next query).
//   compressed bf16 [R][512] per KV-source layer (2, 8, 14, 20), row j = group j, fp4-valued; consumers pass the SOURCE
//   KV         layer's cache.
//   topk       int32 [rows][ldtk], compressed-row ids ascending, -1 only at the tail (produced by the indexer, ds41_idx.cuh).
// Rows are positions start .. start+n-1: prefill from start == 0 (any n), or n >= 1 rows at start > 0 (one decode row, or a verify
// chunk: row t sees window positions start+t-127 .. start+t, so a chunk equals n sequential decode steps).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <math.h>

#include "../knobs-rs/gr8_knobs.h"
#include "ds41_attn_types.h"
#include "ds41_ring_keep.h"
#include "ds41_src_fast.cuh"
#include "ds41_attn_split_math.h"

namespace ds41a
{
// ------------------------------------------------------------------------------------------------ small device helpers
__device__ __forceinline__ float f_e4m3(uint8_t c)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
    return __half2float(__half(h));
}
__device__ __forceinline__ float f_e8m0(uint8_t b) { return __uint_as_float(b ? ((uint32_t)b << 23) : 0x00400000u); }
__device__ __forceinline__ float pow2i(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }
__device__ __forceinline__ int log2ceil_bits(float x)
{
    uint32_t b = __float_as_uint(x);
    return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0);
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float warp_max(float v)
{
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}
// sum over all threads of a block; sh = 32 floats of scratch; every thread gets the total
__device__ __forceinline__ float block_sum(float v, float* sh)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < nw; i++) t += sh[i];
    return t;
}
// Quantize one value per lane with its warp as the 32-block: returns the dequantized value; code / exponent byte out.
__device__ __forceinline__ float act_quant_lane(float v, uint8_t& code, uint8_t& ebyte)
{
    float a = warp_max(fabsf(v));
    a = fmaxf(a, 1e-4f);
    const int e = log2ceil_bits(a * INV448);
    code = (uint8_t)__nv_cvt_float_to_fp8(v * pow2i(-e), __NV_SATFINITE, __NV_E4M3);
    ebyte = (uint8_t)(e + 127);
    return f_e4m3(code) * pow2i(e);
}
// rotate the pair (a, b) by angle ang; plain mul/add, no FMA contraction (the reference's complex multiply)
__device__ __forceinline__ void rot_pair(float a, float b, float ang, bool inverse, float& ra, float& rb)
{
    float s, c;
    sincosf(ang, &s, &c);
    if (inverse) s = -s;
    ra = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    rb = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}
__device__ __forceinline__ float e2m1_rne(float v)
{
    const float a = fminf(fabsf(v), 6.0f);
    float o = 0.0f;
    if (a > 0.25f) o = 0.5f;
    if (a >= 0.75f) o = 1.0f;
    if (a > 1.25f) o = 1.5f;
    if (a >= 1.75f) o = 2.0f;
    if (a > 2.5f) o = 3.0f;
    if (a >= 3.5f) o = 4.0f;
    if (a > 5.0f) o = 6.0f;
    return copysignf(o, v);
}

// ------------------------------------------------------------------------------------------------ activation quantizer
// x [M][K] bf16 (row stride ldx) -> codes u8 [M][K] (stride ldc), E8M0 bytes [M][K/32] (stride lds). One warp per block of 32.
__global__ void k_act_quant(const bf16* __restrict__ x, int ldx, uint8_t* __restrict__ codes, int ldc, uint8_t* __restrict__ sc, int lds,
                            int M, int K)
{
    const int lane = threadIdx.x & 31;
    const int kb = K >> 5;
    const long w = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= (long)M * kb) return;
    const int r = (int)(w / kb), b = (int)(w % kb);
    const float v = __bfloat162float(x[(size_t)r * ldx + b * 32 + lane]);
    uint8_t c, e;
    act_quant_lane(v, c, e);
    codes[(size_t)r * ldc + b * 32 + lane] = c;
    if (lane == 0) sc[(size_t)r * lds + b] = e;
}

// ------------------------------------------------------------------------------------------------ block-scaled fp8 gemm
struct GemmP
{
    const uint8_t *A, *SA, *W, *SW;   // activation codes / scale bytes, weight codes [N][K] / scale bytes [N][K/32]
    bf16* Y;
    int M, N, K;                      // per group
    int lda, lsa, ldy;                // A row stride (bytes), A scale row stride (bytes), Y row stride (elements)
    size_t a_gs, sa_gs, w_gs, sw_gs, y_gs;   // group strides (bytes / bytes / bytes / bytes / elements)
    int mfast;                        // k_gemm_mx8t only, set by launch_tc_t: 1 = the grid is (M tiles, N tiles, groups), so CTAs that share a weight tile are neighbours
};
constexpr int GBM = 64, GBN = 64;
__global__ void __launch_bounds__(256) k_gemm_mxfp8(GemmP p)
{
    __shared__ float lut[256];
    __shared__ float As[GBM][33];
    __shared__ float Ws[GBN][33];
    __shared__ float sa[GBM], sw[GBN];
    const int g = blockIdx.z, tid = threadIdx.x;
    const uint8_t* A = p.A + g * p.a_gs;
    const uint8_t* SA = p.SA + g * p.sa_gs;
    const uint8_t* W = p.W + g * p.w_gs;
    const uint8_t* SW = p.SW + g * p.sw_gs;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
    const int KB = p.K >> 5;
    lut[tid] = f_e4m3((uint8_t)tid);
    __syncthreads();
    const int ty = tid >> 4, tx = tid & 15;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int kb = 0; kb < KB; kb++)
    {
        {
            const int r = tid >> 2, c8 = (tid & 3) * 8;
            uint2 ua = make_uint2(0, 0), uw = make_uint2(0, 0);
            if (m0 + r < p.M) ua = *(const uint2*)(A + (size_t)(m0 + r) * p.lda + kb * 32 + c8);
            if (n0 + r < p.N) uw = *(const uint2*)(W + (size_t)(n0 + r) * p.K + kb * 32 + c8);
            const uint8_t* pa = (const uint8_t*)&ua;
            const uint8_t* pw = (const uint8_t*)&uw;
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                As[r][c8 + i] = lut[pa[i]];
                Ws[r][c8 + i] = lut[pw[i]];
            }
        }
        if (tid < GBM) sa[tid] = (m0 + tid < p.M) ? f_e8m0(SA[(size_t)(m0 + tid) * p.lsa + kb]) : 0.0f;
        else if (tid < GBM + GBN)
        {
            const int n = n0 + tid - GBM;
            sw[tid - GBM] = (n < p.N) ? f_e8m0(SW[(size_t)n * KB + kb]) : 0.0f;
        }
        __syncthreads();
        float part[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) part[i][j] = 0.0f;
#pragma unroll 8
        for (int k = 0; k < 32; k++)
        {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[ty * 4 + i][k];
#pragma unroll
            for (int j = 0; j < 4; j++) w[j] = Ws[tx * 4 + j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = fmaf(a[i], w[j], part[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = fmaf(part[i][j], sa[ty * 4 + i] * sw[tx * 4 + j], acc[i][j]);
        __syncthreads();
    }
    bf16* Y = p.Y + g * p.y_gs;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        const int m = m0 + ty * 4 + i;
        if (m >= p.M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int n = n0 + tx * 4 + j;
            if (n < p.N) Y[(size_t)m * p.ldy + n] = __float2bfloat16_rn(acc[i][j]);
        }
    }
}


// ------------------------------------------------------------------------------------------------ tiled tensor-core block-scaled fp8 gemm
// Same operands and meaning as k_gemm_mxfp8 (A codes [M][lda] + E8M0 bytes [M][lsa], W codes [N][K] + E8M0 bytes [N][K/32], per group), but on the
// block-scaled tensor-core mma (sm_120a / 121a kind::mxf8f6f4 scale_vec::1X m16n8k32 ue8m0, fp32 accumulate): Y = sum over the 32-blocks of the
// block dot product scaled by 2^(sa-127) 2^(sw-127). The fragment and scale-operand convention is the one engine/kernels/bind_prefill.cuh probed on
// sm_120a (mma_mx8 there; copied, that header is not included here): a0..a3 = (row g, k 4t..4t+3), (g+8, 4t..), (g, 16+4t..), (g+8, 16+4t..);
// b0 = (k 4t..4t+3, col g), b1 = (k 16+4t.., col g); the A scale operand of a lane is the byte of row g (even lane) or g+8 (odd lane), the B scale
// operand the byte of col g (g = lane / 4, t = lane % 4). A THIRD numeric order (the hardware's block accumulation), not bit-equal to k_gemm_mxfp8
// or to the GEMV: the quality bar is top-1 agreement, measured by the oracle, not here.
// CTA = BM x BN output tile, 8 warps as 2 (rows) x 4 (cols), K in stages of 128 (cp.async, STAGES-deep ring); smem rows 144 B (conflict-free 4-B reads).
// Needs K % 128 == 0, N % BN == 0, lda / lsa multiples of 16 / 4 bytes; rows >= M read row M - 1 and are not stored.
__device__ __forceinline__ void tc_cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void tc_cp4(void* s, const void* g) { asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(s)), "l"(g)); }
__device__ __forceinline__ void tc_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void tc_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void tc_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}
template <int BM, int BN, int ST> struct TcCfg
{
    static constexpr int WM = BM / 2, WN = BN / 4, MB = WM / 16, NT = WN / 8, PITCH = 144;
    static constexpr int A_B = BM * PITCH, W_B = BN * PITCH, SA_B = BM * 4, SW_B = BN * 4, STAGE = A_B + W_B + SA_B + SW_B;
    static constexpr size_t SMEM = (size_t)STAGE * ST;
};
template <int BM, int BN, int ST> __global__ void __launch_bounds__(256) k_gemm_mx8t(GemmP p)
{
    using C = TcCfg<BM, BN, ST>;
    extern __shared__ __align__(16) uint8_t tsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    const uint8_t* A = p.A + blockIdx.z * p.a_gs;
    const uint8_t* SA = p.SA + blockIdx.z * p.sa_gs;
    const uint8_t* W = p.W + blockIdx.z * p.w_gs;
    const uint8_t* SW = p.SW + blockIdx.z * p.sw_gs;
    const int m0 = (p.mfast ? blockIdx.x : blockIdx.y) * BM, n0 = (p.mfast ? blockIdx.y : blockIdx.x) * BN, NK = p.K >> 7, KB = p.K >> 5;
    auto issue = [&](int s, int kt)
    {
        uint8_t* b = tsm + (size_t)s * C::STAGE;
        for (int c = tid; c < BM * 8; c += 256)
            tc_cp16(b + (c >> 3) * C::PITCH + (c & 7) * 16, A + (size_t)min(m0 + (c >> 3), p.M - 1) * p.lda + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BN * 8; c += 256)
            tc_cp16(b + C::A_B + (c >> 3) * C::PITCH + (c & 7) * 16, W + (size_t)(n0 + (c >> 3)) * p.K + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BM; c += 256) tc_cp4(b + C::A_B + C::W_B + c * 4, SA + (size_t)min(m0 + c, p.M - 1) * p.lsa + kt * 4);
        for (int c = tid; c < BN; c += 256) tc_cp4(b + C::A_B + C::W_B + C::SA_B + c * 4, SW + (size_t)(n0 + c) * KB + kt * 4);
    };
    float acc[C::MB][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MB; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
#pragma unroll
    for (int s = 0; s < ST - 1; s++)
    {
        if (s < NK) issue(s, s);
        tc_commit();
    }
    for (int kt = 0; kt < NK; kt++)
    {
        tc_wait<ST - 2>();
        __syncthreads();
        if (kt + ST - 1 < NK) issue((kt + ST - 1) % ST, kt + ST - 1);
        tc_commit();
        const uint8_t* b = tsm + (size_t)(kt % ST) * C::STAGE;
        const uint8_t* sA = b;
        const uint8_t* sW = b + C::A_B;
        const uint8_t* sSA = sW + C::W_B;
        const uint8_t* sSW = sSA + C::SA_B;
        uint32_t wsa[C::MB], wsw[C::NT];
#pragma unroll
        for (int mb = 0; mb < C::MB; mb++) wsa[mb] = *(const uint32_t*)(sSA + (wm * C::WM + mb * 16 + g + ((lane & 1) ? 8 : 0)) * 4);
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++) wsw[nt] = *(const uint32_t*)(sSW + (wn * C::WN + nt * 8 + g) * 4);
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            uint32_t a[C::MB][4], bb[C::NT][2];
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
            {
                const uint8_t* r0 = sA + (wm * C::WM + mb * 16 + g) * C::PITCH + ks * 32 + 4 * t;
                a[mb][0] = *(const uint32_t*)r0;
                a[mb][1] = *(const uint32_t*)(r0 + 8 * C::PITCH);
                a[mb][2] = *(const uint32_t*)(r0 + 16);
                a[mb][3] = *(const uint32_t*)(r0 + 8 * C::PITCH + 16);
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt++)
            {
                const uint8_t* c0 = sW + (wn * C::WN + nt * 8 + g) * C::PITCH + ks * 32 + 4 * t;
                bb[nt][0] = *(const uint32_t*)c0;
                bb[nt][1] = *(const uint32_t*)(c0 + 16);
            }
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
                for (int nt = 0; nt < C::NT; nt++) tc_mma(acc[mb][nt], a[mb], bb[nt][0], bb[nt][1], (wsa[mb] >> (8 * ks)) & 0xFFu, (wsw[nt] >> (8 * ks)) & 0xFFu);
        }
    }
    tc_wait<0>();
    bf16* Y = p.Y + blockIdx.z * p.y_gs;
#pragma unroll
    for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++)
        {
            const int col = n0 + wn * C::WN + nt * 8 + 2 * t, row = m0 + wm * C::WM + mb * 16 + g;
            if (row < p.M) *(__nv_bfloat162*)(Y + (size_t)row * p.ldy + col) = __floats2bfloat162_rn(acc[mb][nt][0], acc[mb][nt][1]);
            if (row + 8 < p.M) *(__nv_bfloat162*)(Y + (size_t)(row + 8) * p.ldy + col) = __floats2bfloat162_rn(acc[mb][nt][2], acc[mb][nt][3]);
        }
}

// Compressor FP8 arm: same block-scaled E4M3/UE8M0 MMA and K order as k_gemm_mx8t, but preserves the compressor's FP32 output contract.
template <int BM, int BN, int ST> __global__ void __launch_bounds__(256) k_gemm_mx8t_f32(GemmP p, float* Y32)
{
    using C = TcCfg<BM, BN, ST>;
    extern __shared__ __align__(16) uint8_t tsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    const uint8_t* A = p.A + blockIdx.z * p.a_gs;
    const uint8_t* SA = p.SA + blockIdx.z * p.sa_gs;
    const uint8_t* W = p.W + blockIdx.z * p.w_gs;
    const uint8_t* SW = p.SW + blockIdx.z * p.sw_gs;
    const int m0 = (p.mfast ? blockIdx.x : blockIdx.y) * BM, n0 = (p.mfast ? blockIdx.y : blockIdx.x) * BN, NK = p.K >> 7, KB = p.K >> 5;
    auto issue = [&](int s, int kt)
    {
        uint8_t* b = tsm + (size_t)s * C::STAGE;
        for (int c = tid; c < BM * 8; c += 256)
            tc_cp16(b + (c >> 3) * C::PITCH + (c & 7) * 16, A + (size_t)min(m0 + (c >> 3), p.M - 1) * p.lda + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BN * 8; c += 256)
            tc_cp16(b + C::A_B + (c >> 3) * C::PITCH + (c & 7) * 16, W + (size_t)(n0 + (c >> 3)) * p.K + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BM; c += 256) tc_cp4(b + C::A_B + C::W_B + c * 4, SA + (size_t)min(m0 + c, p.M - 1) * p.lsa + kt * 4);
        for (int c = tid; c < BN; c += 256) tc_cp4(b + C::A_B + C::W_B + C::SA_B + c * 4, SW + (size_t)(n0 + c) * KB + kt * 4);
    };
    float acc[C::MB][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MB; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
#pragma unroll
    for (int s = 0; s < ST - 1; s++)
    {
        if (s < NK) issue(s, s);
        tc_commit();
    }
    for (int kt = 0; kt < NK; kt++)
    {
        tc_wait<ST - 2>();
        __syncthreads();
        if (kt + ST - 1 < NK) issue((kt + ST - 1) % ST, kt + ST - 1);
        tc_commit();
        const uint8_t* b = tsm + (size_t)(kt % ST) * C::STAGE;
        const uint8_t* sA = b;
        const uint8_t* sW = b + C::A_B;
        const uint8_t* sSA = sW + C::W_B;
        const uint8_t* sSW = sSA + C::SA_B;
        uint32_t wsa[C::MB], wsw[C::NT];
#pragma unroll
        for (int mb = 0; mb < C::MB; mb++) wsa[mb] = *(const uint32_t*)(sSA + (wm * C::WM + mb * 16 + g + ((lane & 1) ? 8 : 0)) * 4);
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++) wsw[nt] = *(const uint32_t*)(sSW + (wn * C::WN + nt * 8 + g) * 4);
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            uint32_t a[C::MB][4], bb[C::NT][2];
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
            {
                const uint8_t* r0 = sA + (wm * C::WM + mb * 16 + g) * C::PITCH + ks * 32 + 4 * t;
                a[mb][0] = *(const uint32_t*)r0;
                a[mb][1] = *(const uint32_t*)(r0 + 8 * C::PITCH);
                a[mb][2] = *(const uint32_t*)(r0 + 16);
                a[mb][3] = *(const uint32_t*)(r0 + 8 * C::PITCH + 16);
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt++)
            {
                const uint8_t* c0 = sW + (wn * C::WN + nt * 8 + g) * C::PITCH + ks * 32 + 4 * t;
                bb[nt][0] = *(const uint32_t*)c0;
                bb[nt][1] = *(const uint32_t*)(c0 + 16);
            }
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
                for (int nt = 0; nt < C::NT; nt++) tc_mma(acc[mb][nt], a[mb], bb[nt][0], bb[nt][1], (wsa[mb] >> (8 * ks)) & 0xFFu, (wsw[nt] >> (8 * ks)) & 0xFFu);
        }
    }
    tc_wait<0>();
    float* Y = Y32 + blockIdx.z * p.y_gs;
#pragma unroll
    for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++)
        {
            const int col = n0 + wn * C::WN + nt * 8 + 2 * t, row = m0 + wm * C::WM + mb * 16 + g;
            if (row < p.M) { Y[(size_t)row * p.ldy + col] = acc[mb][nt][0]; Y[(size_t)row * p.ldy + col + 1] = acc[mb][nt][1]; }
            if (row + 8 < p.M) { Y[(size_t)(row + 8) * p.ldy + col] = acc[mb][nt][2]; Y[(size_t)(row + 8) * p.ldy + col + 1] = acc[mb][nt][3]; }
        }
}


// ------------------------------------------------------------------------------------------------ rmsnorm
// y = bf16(w * (x * rsqrt(mean(x^2) + eps))), fp32 math; one block per row; y may alias x.
__global__ void k_rmsnorm(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ w, bf16* y, int ldy, int D, float eps)
{
    __shared__ float sh[32];
    const bf16* xr = x + (size_t)blockIdx.x * ldx;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float v = __bfloat162float(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / (float)D + eps);
    bf16* yr = y + (size_t)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = __float2bfloat16_rn(__bfloat162float(w[i]) * (__bfloat162float(xr[i]) * r));
}

// The first position of a call: a host value (tests, prefill) or, in the engine step, the device word m.pos[0] (graph-safe; the
// rows of a decode step are contiguous, pos[r] = pos[0] + r).
__device__ __forceinline__ int start_of(int start, const int* dstart) { return dstart ? __ldcv(dstart) : start; }
// Row r's position. SESS > 1 (dsess = Model::sess, rows of several sessions): the device array dstart[r]; else start + r (start from start_of).
__device__ __forceinline__ int pos_of(int start, const int* dstart, const int* dsess, int r) { return dsess ? __ldcv(dstart + r) : start + r; }

// ------------------------------------------------------------------------------------------------ rope on the last 64 dims
// x [rows][heads][512] bf16 in place (row stride ld elements); interleaved pairs; angle = fp32(pos) * inv[i], pos = start + row.
__global__ void k_rope_tail(bf16* x, long ld, int rows, int heads, int start0, const int* dstart, const float* __restrict__ inv, int inverse,
                            const int* dsess = nullptr)
{
    const int start = start_of(start0, dstart);
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)rows * heads * (ROPE / 2)) return;
    const int i = (int)(t % (ROPE / 2));
    const long rh = t / (ROPE / 2);
    const int h = (int)(rh % heads), r = (int)(rh / heads);
    bf16* p = x + (size_t)r * ld + (size_t)h * HD + NOPE + 2 * i;
    float ra, rb;
    rot_pair(__bfloat162float(p[0]), __bfloat162float(p[1]), (float)pos_of(start, dstart, dsess, r) * inv[i], inverse != 0, ra, rb);
    p[0] = __float2bfloat16_rn(ra);
    p[1] = __float2bfloat16_rn(rb);
}

// ------------------------------------------------------------------------------------------------ window KV prep
// kv0 [n][512] (wkv out) -> kv_norm -> rope tail -> fp8 quantize-dequantize over all 512 (blocks of 32).
// Writes kvrows [n][512] (the chunk's rows) and the ring slot (pos % R) for the last R positions (per slot run at SESS > 1:
// ds41_ring_keep.h). One block of 512 per row.
__global__ void __launch_bounds__(512) k_kv_prep(const bf16* __restrict__ kv0, const bf16* __restrict__ nw, const float* __restrict__ inv,
                                                 int start0, const int* dstart, int n, bf16* kvrows, bf16* ring, int R, const int* dsess = nullptr)
{
    const int start = start_of(start0, dstart);
    __shared__ float sh[32];
    const int row = blockIdx.x, c = threadIdx.x;
    const float v = __bfloat162float(kv0[(size_t)row * HD + c]);
    const float ss = block_sum(v * v, sh);
    const float r = rsqrtf(ss / (float)HD + EPS);
    float x = bf16r(__bfloat162float(nw[c]) * (v * r));
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)pos_of(start, dstart, dsess, row) * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { rot_pair(x, partner, ang, false, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, false, ra, rb); x = bf16r(rb); }
    }
    uint8_t code, e;
    const bf16 o = __float2bfloat16_rn(act_quant_lane(x, code, e));
    if (kvrows) kvrows[(size_t)row * HD + c] = o;
    const int pos = pos_of(start, dstart, dsess, row);
    if (ring && ring_keep(row, n, R, dsess))   // SESS > 1: the row's slot ring, last R rows of the row's own slot run
        ring[(dsess ? (size_t)dsess[row] * R * HD : 0) + (size_t)(pos % R) * HD + c] = o;
}

// ------------------------------------------------------------------------------------------------ o prep (inverse rope + quantize)
// o [rows][heads][512] bf16 -> inverse rope on the tail, then act-quant per 32 along each head: codes u8 [rows][heads*512],
// E8M0 bytes [rows][heads*16]. One block of 512 per (row, head).
__global__ void __launch_bounds__(512) k_o_prep(const bf16* __restrict__ o, int heads, int start0, const int* dstart,
                                                const float* __restrict__ inv, bf16* o_rot, uint8_t* codes, uint8_t* sc, const int* dsess = nullptr)
{
    const int start = start_of(start0, dstart);
    const int row = blockIdx.x / heads, h = blockIdx.x % heads, c = threadIdx.x;
    const size_t base = ((size_t)row * heads + h) * HD;
    float x = __bfloat162float(o[base + c]);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)pos_of(start, dstart, dsess, row) * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { rot_pair(x, partner, ang, true, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, true, ra, rb); x = bf16r(rb); }
    }
    if (o_rot) o_rot[base + c] = __float2bfloat16_rn(x);
    uint8_t code, e;
    act_quant_lane(x, code, e);
    codes[base + c] = code;
    if ((c & 31) == 0) sc[((size_t)row * heads + h) * (HD / 32) + (c >> 5)] = e;
}

// ------------------------------------------------------------------------------------------------ sparse attention core
// One CTA = one row x 16 heads. Slot list per row = [window | compressed top-k] in blocks of 64 slots; scores by tensor core
// (bf16 in, fp32 accumulate), running max / rescale per block exactly like the release's sparse_attn: P = exp(s - max) in fp32,
// the denominator sums fp32 P, the PV product uses bf16 P, the sink joins the denominator once at the end.
struct AttnP
{
    const bf16* q;       // [n][heads][512], rope applied
    const bf16* kvw;     // window rows: prefill = this chunk's rows [n][512]; decode = the ring [ring_rows][512]
    const bf16* ck;      // compressed cache rows of the source layer [R][512] (null when TK == 0)
    const int* topk;     // [n][ldtk] (null when TK == 0)
    const float* sink;   // [heads]
    bf16* out;           // [n][heads][512]
    int ldtk, n, heads, start;
    const int* dstart;   // device word with the first position (overrides start; the engine's m.pos), or null
    int decode;          // 0: prefill from 0, 1: n rows at positions start .. start+n-1, start > 0
    int WL;              // window slots: min(n, 128) prefill, 128 decode
    int ring_rows;       // decode: slots in the ring (slot = position % ring_rows); 0 = 128
    int TK;              // compressed slots: min(512, (start + n) / ratio), 0 when the layer has none
    float scale;         // 512^-0.5
    int floor = 0;       // decode: window positions below this are masked (bounded replay: ring slots the chunk's tail layers never wrote); 0 = none
    const int* dsess = nullptr;   // SESS > 1 (decode): device [n] S2 slot of each row; row t at position dstart[t], its ring = kvw + slot * ring_rows * 512,
    size_t ck_ss = 0;             // its compressed rows = ck + slot * ck_ss. nullptr: one session (row t at start + t)
    float* split = nullptr;       // optional [n][heads][2][max,sum,HD accum], shared prefill scratch
};
constexpr int AT_THREADS = 256, AT_QLD = HD + 8, AT_KLD = HD + 8, AT_SLD = 64 + 4, AT_PLD = 64 + 8;
constexpr size_t AT_SMEM = (size_t)16 * AT_QLD * 2 + (size_t)64 * AT_KLD * 2 + (size_t)16 * AT_SLD * 4 + (size_t)16 * AT_PLD * 2 +
                           64 * sizeof(void*) + 32 * 4;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// FL = false: the plain kernel (pos >= 0). FL = true: also masks window positions below p.floor (bounded replay, k_attn_floor).
template <bool FL>
__global__ void __launch_bounds__(AT_THREADS, 1) k_attn_t(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);
    float* sc_s = (float*)(rowp + 64);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
    {
        const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
        for (int c = tid; c < 16 * HD / 8; c += AT_THREADS)
            *(uint4*)(Qs + (c >> 6) * AT_QLD + (c & 63) * 8) = src[c];
    }
    float m_run = -1e30f, sum_run = 0.0f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int start = start_of(p.start, p.dstart);
    const int L = p.WL + p.TK;
    const int nblk = (L + 63) >> 6;
    const int R = p.ring_rows ? p.ring_rows : WIN;
    const int tpos = pos_of(start, p.dstart, p.dsess, t);   // this row's position (decode)
    const bf16 *kvr = p.kvw, *ckr = p.ck;                   // this row's ring and compressed rows (its session's slab at SESS > 1)
    if (p.dsess)
    {
        const int k = p.dsess[t];
        kvr += (size_t)k * R * HD;
        if (ckr) ckr += (size_t)k * p.ck_ss;
    }
    for (int blk = 0; blk < nblk; blk++)
    {
        int has = 0;
        if (tid < 64)
        {
            const int i = blk * 64 + tid;
            const bf16* rp = nullptr;
            if (i < p.WL)
            {
                if (!p.decode)
                {
                    const int id = max(t - (WIN - 1), 0) + i;
                    if (id <= t) rp = p.kvw + (size_t)id * HD;
                }
                else
                {
                    const int pos = tpos - (WIN - 1) + i;   // list slot i holds position tpos - 127 + i; negative = masked
                    if (pos >= (FL ? p.floor : 0)) rp = kvr + (size_t)(pos % R) * HD;   // FL (bounded replay): positions below p.floor masked
                }
            }
            else if (i < L)
            {
                const int j = p.topk[(size_t)t * p.ldtk + (i - p.WL)];
                if (j >= 0) rp = ckr + (size_t)j * HD;
            }
            rowp[tid] = rp;
            has = rp != nullptr;
        }
        // a block with no live slot (the -1 tail of a short top-k list, the masked front of a young window) changes nothing:
        // P = 0, scale = exp(0) = 1 -> skip it (uniform for the CTA)
        if (!__syncthreads_or(has)) continue;
        for (int c = tid; c < 64 * 64; c += AT_THREADS)
        {
            const bf16* rp = rowp[c >> 6];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (rp) v = ((const uint4*)rp)[c & 63];
            *(uint4*)(Ks + (c >> 6) * AT_KLD + (c & 63) * 8) = v;
        }
        __syncthreads();
        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks++)
            {
                uint32_t a[4];
                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                mma16816(cc, a, b0, b1);
            }
            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
            const bool v0 = rowp[col] != nullptr, v1 = rowp[col + 1] != nullptr;
            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
        }
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
            }
#pragma unroll
            for (int kj = 0; kj < 4; kj++)
            {
                uint32_t a[4];
                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; np++)
                {
                    uint32_t b[4];
                    const int mi = lane >> 3, r = lane & 7;
                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    if (sl == 0) fin_s[sh] = sum_run + expf(p.sink[hg * 16 + sh] - m_run);
    __syncthreads();
    {
        const int r0 = lane >> 2;
        const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
        bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
        bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);
            *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);
        }
    }
}
// the plain kernel and the bounded replay one under plain names (tests and launch_attn take their address / launch them)
inline void (*const k_attn)(AttnP) = k_attn_t<false>;
inline void (*const k_attn_floor)(AttnP) = k_attn_t<true>;

// ------------------------------------------------------------------------------------------------ sparse attention core, cp.async gather
// k_attn_cpa (GR8_DS_ATTN_CPA=1) = k_attn with the same arithmetic in the same order and a different way to fill smem: the 64 K rows of a
// block (64 KB) are all in flight at once as 16 cp.async per thread (a null slot is zero-filled by src-size 0, the same as k_attn's make_uint4(0)), Q
// joins the first group, the NEXT live block's slot list (the top-k index load) is built while the K rows land, and its rows are
// prefetched into L2. k_attn's loop does 16 dependent global-load + smem-store pairs per thread per block (~12 us per 64-slot block).
// Output bytes == k_attn.
constexpr size_t AT_SMEM_CPA = AT_SMEM + 64 * sizeof(void*);   // a second slot list (prefetch of the next block)
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }

// FL = false: the plain kernel (pos >= 0). FL = true: also masks window positions below p.floor (bounded replay, k_attn_cpa_floor).
template <bool FL>
__global__ void __launch_bounds__(AT_THREADS, 1) k_attn_cpa_t(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);   // two lists of 64 slots: the block in use and the next live one
    float* sc_s = (float*)(rowp + 128);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
    {
        const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
        for (int c = tid; c < 16 * HD / 8; c += AT_THREADS)
            cp_async16(Qs + (c >> 6) * AT_QLD + (c & 63) * 8, src + c, 16);
    }
    float m_run = -1e30f, sum_run = 0.0f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int start = start_of(p.start, p.dstart);
    const int L = p.WL + p.TK;
    const int nblk = (L + 63) >> 6;
    const int R = p.ring_rows ? p.ring_rows : WIN;
    const int tpos = pos_of(start, p.dstart, p.dsess, t);   // this row's position (decode)
    const bf16 *kvr = p.kvw, *ckr = p.ck;                   // this row's ring and compressed rows (its session's slab at SESS > 1)
    if (p.dsess)
    {
        const int k = p.dsess[t];
        kvr += (size_t)k * R * HD;
        if (ckr) ckr += (size_t)k * p.ck_ss;
    }
    // slot list of block blk into lst; the return value (uniform) says whether any slot is live. A block with no live slot (the -1 tail
    // of a short top-k list, the masked front of a young window) changes nothing: P = 0, scale = exp(0) = 1 -> skipped
    auto build = [&](int blk, const bf16** lst) -> int
    {
        int has = 0;
        if (tid < 64)
        {
            const int i = blk * 64 + tid;
            const bf16* rp = nullptr;
            if (i < p.WL)
            {
                if (!p.decode)
                {
                    const int id = max(t - (WIN - 1), 0) + i;
                    if (id <= t) rp = p.kvw + (size_t)id * HD;
                }
                else
                {
                    const int pos = tpos - (WIN - 1) + i;   // list slot i holds position tpos - 127 + i; negative = masked
                    if (pos >= (FL ? p.floor : 0)) rp = kvr + (size_t)(pos % R) * HD;   // FL (bounded replay): positions below p.floor masked
                }
            }
            else if (i < L)
            {
                const int j = p.topk[(size_t)t * p.ldtk + (i - p.WL)];
                if (j >= 0) rp = ckr + (size_t)j * HD;
            }
            lst[tid] = rp;
            has = rp != nullptr;
        }
        return __syncthreads_or(has);
    };
    int cblk = 0, cur = 0;
    while (cblk < nblk && !build(cblk, rowp)) cblk++;
    while (cblk < nblk)
    {
        const bf16** cl = rowp + 64 * cur;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            const int c = tid + i * AT_THREADS;
            const bf16* rp = cl[c >> 6];
            cp_async16(Ks + (c >> 6) * AT_KLD + (c & 63) * 8, rp ? (const void*)((const uint4*)rp + (c & 63)) : (const void*)p.q, rp ? 16 : 0);
        }
        cp_async_commit();
        int nb = cblk + 1;
        const int nx = cur ^ 1;
        while (nb < nblk && !build(nb, rowp + 64 * nx)) nb++;
        if (nb < nblk)
        {
            const bf16* rp = rowp[64 * nx + (tid >> 2)];
            if (rp)
            {
                prefetch_l2((const char*)rp + (tid & 3) * 256);
                prefetch_l2((const char*)rp + (tid & 3) * 256 + 128);
            }
        }
        cp_async_wait0();
        __syncthreads();
        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks++)
            {
                uint32_t a[4];
                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                mma16816(cc, a, b0, b1);
            }
            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
            const bool v0 = cl[col] != nullptr, v1 = cl[col + 1] != nullptr;
            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
        }
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
            }
#pragma unroll
            for (int kj = 0; kj < 4; kj++)
            {
                uint32_t a[4];
                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; np++)
                {
                    uint32_t b[4];
                    const int mi = lane >> 3, r = lane & 7;
                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
        cblk = nb;
        cur = nx;
    }
    cp_async_commit();
    cp_async_wait0();   // Q's copies when no block was live
    if (sl == 0) fin_s[sh] = sum_run + expf(p.sink[hg * 16 + sh] - m_run);
    __syncthreads();
    {
        const int r0 = lane >> 2;
        const float d0 = fin_s[r0], d1 = fin_s[r0 + 8];
        bf16* o0 = p.out + ((size_t)t * p.heads + hg * 16 + r0) * HD;
        bf16* o1 = p.out + ((size_t)t * p.heads + hg * 16 + r0 + 8) * HD;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            *(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);
            *(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);
        }
    }
}
// Split-key-block variant of the existing cp.async + mma.sync core; default-off.
template <bool FL>
__global__ void __launch_bounds__(AT_THREADS, 1) k_attn_cpa_split_t(AttnP p)
{
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;
    bf16* Ks = Qs + 16 * AT_QLD;
    float* Ss = (float*)(Ks + 64 * AT_KLD);
    bf16* Ps = (bf16*)(Ss + 16 * AT_SLD);
    const bf16** rowp = (const bf16**)(Ps + 16 * AT_PLD);   // two lists of 64 slots: the block in use and the next live one
    float* sc_s = (float*)(rowp + 128);
    float* fin_s = sc_s + 16;

    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sh = tid >> 4, sl = tid & 15;   // softmax role: head sh, slots sl*4 .. sl*4+3
    {
        const uint4* src = (const uint4*)(p.q + ((size_t)t * p.heads + (size_t)hg * 16) * HD);
        for (int c = tid; c < 16 * HD / 8; c += AT_THREADS)
            cp_async16(Qs + (c >> 6) * AT_QLD + (c & 63) * 8, src + c, 16);
    }
    float m_run = -1e30f, sum_run = 0.0f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int start = start_of(p.start, p.dstart);
    const int L = p.WL + p.TK;
    const int nblk = (L + 63) >> 6;
    const int part = blockIdx.z;
    const int blk_lo = (nblk * part) / 2, blk_hi = (nblk * (part + 1)) / 2;
    const int R = p.ring_rows ? p.ring_rows : WIN;
    const int tpos = pos_of(start, p.dstart, p.dsess, t);   // this row's position (decode)
    const bf16 *kvr = p.kvw, *ckr = p.ck;                   // this row's ring and compressed rows (its session's slab at SESS > 1)
    if (p.dsess)
    {
        const int k = p.dsess[t];
        kvr += (size_t)k * R * HD;
        if (ckr) ckr += (size_t)k * p.ck_ss;
    }
    // slot list of block blk into lst; the return value (uniform) says whether any slot is live. A block with no live slot (the -1 tail
    // of a short top-k list, the masked front of a young window) changes nothing: P = 0, scale = exp(0) = 1 -> skipped
    auto build = [&](int blk, const bf16** lst) -> int
    {
        int has = 0;
        if (tid < 64)
        {
            const int i = blk * 64 + tid;
            const bf16* rp = nullptr;
            if (i < p.WL)
            {
                if (!p.decode)
                {
                    const int id = max(t - (WIN - 1), 0) + i;
                    if (id <= t) rp = p.kvw + (size_t)id * HD;
                }
                else
                {
                    const int pos = tpos - (WIN - 1) + i;   // list slot i holds position tpos - 127 + i; negative = masked
                    if (pos >= (FL ? p.floor : 0)) rp = kvr + (size_t)(pos % R) * HD;   // FL (bounded replay): positions below p.floor masked
                }
            }
            else if (i < L)
            {
                const int j = p.topk[(size_t)t * p.ldtk + (i - p.WL)];
                if (j >= 0) rp = ckr + (size_t)j * HD;
            }
            lst[tid] = rp;
            has = rp != nullptr;
        }
        return __syncthreads_or(has);
    };
    int cblk = blk_lo, cur = 0;
    while (cblk < blk_hi && !build(cblk, rowp)) cblk++;
    while (cblk < blk_hi)
    {
        const bf16** cl = rowp + 64 * cur;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            const int c = tid + i * AT_THREADS;
            const bf16* rp = cl[c >> 6];
            cp_async16(Ks + (c >> 6) * AT_KLD + (c & 63) * 8, rp ? (const void*)((const uint4*)rp + (c & 63)) : (const void*)p.q, rp ? 16 : 0);
        }
        cp_async_commit();
        int nb = cblk + 1;
        const int nx = cur ^ 1;
        while (nb < blk_hi && !build(nb, rowp + 64 * nx)) nb++;
        if (nb < blk_hi)
        {
            const bf16* rp = rowp[64 * nx + (tid >> 2)];
            if (rp)
            {
                prefetch_l2((const char*)rp + (tid & 3) * 256);
                prefetch_l2((const char*)rp + (tid & 3) * 256 + 128);
            }
        }
        cp_async_wait0();
        __syncthreads();
        {   // S[16 heads][8 slots of this warp] = Q . K^T over 512
            float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bf16* krow = Ks + (w * 8 + (lane >> 2)) * AT_KLD + 2 * (lane & 3);
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ks++)
            {
                uint32_t a[4];
                ldsm_x4(a, Qs + (lane & 15) * AT_QLD + ks * 16 + (lane >> 4) * 8);
                const uint32_t b0 = *(const uint32_t*)(krow + ks * 16);
                const uint32_t b1 = *(const uint32_t*)(krow + ks * 16 + 8);
                mma16816(cc, a, b0, b1);
            }
            const int r0 = lane >> 2, col = w * 8 + 2 * (lane & 3);
            const bool v0 = cl[col] != nullptr, v1 = cl[col + 1] != nullptr;
            Ss[r0 * AT_SLD + col] = v0 ? cc[0] * p.scale : -INFINITY;
            Ss[r0 * AT_SLD + col + 1] = v1 ? cc[1] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col] = v0 ? cc[2] * p.scale : -INFINITY;
            Ss[(r0 + 8) * AT_SLD + col + 1] = v1 ? cc[3] * p.scale : -INFINITY;
        }
        __syncthreads();
        {   // online softmax, head sh, 16 threads x 4 slots
            const float4 s4 = *(const float4*)(Ss + sh * AT_SLD + sl * 4);
            float mb = fmaxf(fmaxf(s4.x, s4.y), fmaxf(s4.z, s4.w));
            for (int o = 8; o; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xFFFFFFFFu, mb, o));
            const float m_new = fmaxf(m_run, mb);
            const float scl = expf(m_run - m_new);
            const float p0 = expf(s4.x - m_new), p1 = expf(s4.y - m_new), p2 = expf(s4.z - m_new), p3 = expf(s4.w - m_new);
            float ps = (p0 + p1) + (p2 + p3);
            for (int o = 8; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFu, ps, o);
            sum_run = sum_run * scl + ps;
            m_run = m_new;
            bf16* pr = Ps + sh * AT_PLD + sl * 4;
            pr[0] = __float2bfloat16_rn(p0);
            pr[1] = __float2bfloat16_rn(p1);
            pr[2] = __float2bfloat16_rn(p2);
            pr[3] = __float2bfloat16_rn(p3);
            if (sl == 0) sc_s[sh] = scl;
        }
        __syncthreads();
        {   // acc[16 heads][64 dims of this warp] = acc * scale + P . V   (V = the same K rows)
            const float sc0 = sc_s[lane >> 2], sc1 = sc_s[(lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                acc[nt][0] *= sc0;
                acc[nt][1] *= sc0;
                acc[nt][2] *= sc1;
                acc[nt][3] *= sc1;
            }
#pragma unroll
            for (int kj = 0; kj < 4; kj++)
            {
                uint32_t a[4];
                ldsm_x4(a, Ps + (lane & 15) * AT_PLD + kj * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; np++)
                {
                    uint32_t b[4];
                    const int mi = lane >> 3, r = lane & 7;
                    ldsm_x4_t(b, Ks + (kj * 16 + (mi & 1) * 8 + r) * AT_KLD + w * 64 + (np * 2 + (mi >> 1)) * 8);
                    mma16816(acc[np * 2], a, b[0], b[1]);
                    mma16816(acc[np * 2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
        cblk = nb;
        cur = nx;
    }
    cp_async_commit();
    cp_async_wait0();   // Q's copy when this partition has no live block
    // Each partition publishes one independent online-softmax state. It does not add the sink:
    // the merge adds that once, after rescaling both states to their common maximum.
    if (sl == 0) {
        float* state = p.split + (((size_t)t * p.heads + hg * 16 + sh) * 2 + part) * (HD + 2);
        state[0] = m_run;
        state[1] = sum_run;
    }
    {
        const int r0 = lane >> 2;
        float* state0 = p.split + (((size_t)t * p.heads + hg * 16 + r0) * 2 + part) * (HD + 2) + 2;
        float* state1 = p.split + (((size_t)t * p.heads + hg * 16 + r0 + 8) * 2 + part) * (HD + 2) + 2;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int d = w * 64 + nt * 8 + 2 * (lane & 3);
            state0[d] = acc[nt][0]; state0[d + 1] = acc[nt][1];
            state1[d] = acc[nt][2]; state1[d + 1] = acc[nt][3];
        }
    }
}

// One CTA per row/head-group merges the two partition states. The source softmax's BF16 P
// rounding inside each key block is retained; only cross-partition FP32 accumulation order changes.
__global__ void __launch_bounds__(AT_THREADS) k_attn_cpa_split_merge(AttnP p)
{
    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x;
    for (int i = tid; i < 16 * HD; i += AT_THREADS) {
        const int h = hg * 16 + i / HD, d = i % HD;
        const float* a = p.split + (((size_t)t * p.heads + h) * 2) * (HD + 2);
        const float* b = a + HD + 2;
        const SplitMergeWeights weights = split_merge_weights(a[0], a[1], b[0], b[1], p.sink[h]);
        const float value = split_merge_value(a[d + 2], b[d + 2], weights);
        p.out[((size_t)t * p.heads + h) * HD + d] = __float2bfloat16_rn(value);
    }
}

// the plain kernel and the bounded replay one under plain names (tests and launch_attn take their address / launch them)
inline void (*const k_attn_cpa)(AttnP) = k_attn_cpa_t<false>;
inline void (*const k_attn_cpa_floor)(AttnP) = k_attn_cpa_t<true>;

// ------------------------------------------------------------------------------------------------ ratio-2 compressor
// x bf16 [M][K=5120] . W bf16 [N=512][K]^T for two weight matrices (wkv, wgate) -> fp32 [M][512] each. fp32 math; the weights are
// the checkpoint's bf16 (the release promotes them to fp32, same values).
constexpr int FB_M = 32, FB_N = 64;
__global__ void __launch_bounds__(256) k_f32gemm2(const bf16* __restrict__ x, const bf16* __restrict__ W0, const bf16* __restrict__ W1,
                                                  float* __restrict__ Y0, float* __restrict__ Y1, int M, int N, int K)
{
    __shared__ float As[FB_M][33];
    __shared__ float Ws[FB_N][33];
    const bf16* W = blockIdx.z ? W1 : W0;
    float* Y = blockIdx.z ? Y1 : Y0;
    const int m0 = blockIdx.y * FB_M, n0 = blockIdx.x * FB_N, tid = threadIdx.x;
    const int ty = tid >> 4, tx = tid & 15;
    float acc[2][4] = {};
    for (int k0 = 0; k0 < K; k0 += 32)
    {
        for (int i = tid; i < FB_M * 32; i += 256)
            As[i >> 5][i & 31] = (m0 + (i >> 5) < M) ? __bfloat162float(x[(size_t)(m0 + (i >> 5)) * K + k0 + (i & 31)]) : 0.0f;
        for (int i = tid; i < FB_N * 32; i += 256)
            Ws[i >> 5][i & 31] = (n0 + (i >> 5) < N) ? __bfloat162float(W[(size_t)(n0 + (i >> 5)) * K + k0 + (i & 31)]) : 0.0f;
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < 32; k++)
        {
            float a0 = As[ty * 2][k], a1 = As[ty * 2 + 1][k];
            float w[4];
#pragma unroll
            for (int j = 0; j < 4; j++) w[j] = Ws[tx * 4 + j][k];
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                acc[0][j] = fmaf(a0, w[j], acc[0][j]);
                acc[1][j] = fmaf(a1, w[j], acc[1][j]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            if (m0 + ty * 2 + i < M && n0 + tx * 4 + j < N) Y[(size_t)(m0 + ty * 2 + i) * N + n0 + tx * 4 + j] = acc[i][j];
}

// Pool pairs of tokens into compressed rows. Prefill (start == 0): block j pools tokens 2j, 2j+1 -> row j; with n odd, block n/2
// parks the last token in the state slot 0. Decode (n == 1, start > 0): the token fills state slot start % 2 and, when it
// completes a pair, row start / 2 is produced from the two slots. Row = softmax pool -> bf16 -> RMSNorm -> rope at position 2j
// -> fp4 quantize-dequantize (blocks of 16, E4M3 scale). One block of 512 per row. nrm (optional) receives the post-norm,
// pre-rope row at index grp - first group of this call (the indexer's K input).
__global__ void __launch_bounds__(512) k_comp_pool(const float* __restrict__ kvf, const float* __restrict__ scf, int n, int start,
                                                   const bf16* __restrict__ nw, const float* __restrict__ inv, bf16* ck, float* kvs,
                                                   float* scs, bf16* lat_dbg, bf16* nrm)
{
    __shared__ float sh[32];
    const int c = threadIdx.x;
    float k0, k1, s0, s1;
    int grp;
    if (start == 0)
    {
        const int j = blockIdx.x;
        if (2 * j + 1 >= n)
        {
            kvs[c] = kvf[(size_t)(n - 1) * HD + c];
            scs[c] = scf[(size_t)(n - 1) * HD + c];
            return;
        }
        k0 = kvf[(size_t)(2 * j) * HD + c];
        k1 = kvf[(size_t)(2 * j + 1) * HD + c];
        s0 = scf[(size_t)(2 * j) * HD + c];
        s1 = scf[(size_t)(2 * j + 1) * HD + c];
        grp = j;
    }
    else
    {
        const int slot = start & 1;
        kvs[slot * HD + c] = kvf[c];
        scs[slot * HD + c] = scf[c];
        if (slot == 0) return;
        k0 = kvs[c];
        k1 = kvs[HD + c];
        s0 = scs[c];
        s1 = scs[HD + c];
        grp = start >> 1;
    }
    const float mx = fmaxf(s0, s1);
    const float e0 = expf(s0 - mx), e1 = expf(s1 - mx), den = e0 + e1;
    const float pool = __fadd_rn(__fmul_rn(k0, e0 / den), __fmul_rn(k1, e1 / den));
    float x0 = bf16r(pool);
    const float ss = block_sum(x0 * x0, sh);
    const float r = rsqrtf(ss / (float)HD + EPS);
    float x = bf16r(__bfloat162float(nw[c]) * (x0 * r));
    if (nrm) nrm[(size_t)(grp - (start == 0 ? 0 : (start >> 1))) * HD + c] = __float2bfloat16_rn(x);   // post-norm, pre-rope (the indexer's K row input)
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)(2 * grp) * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { rot_pair(x, partner, ang, false, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, false, ra, rb); x = bf16r(rb); }
    }
    if (lat_dbg) lat_dbg[(size_t)grp * HD + c] = __float2bfloat16_rn(x);
    float a = fabsf(x);
    for (int o = 8; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
    a = fmaxf(a, 6.0f * 0.001953125f);   // 6 * 2^-9
    const float sc = f_e4m3((uint8_t)__nv_cvt_float_to_fp8(a / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float q = e2m1_rne(fminf(fmaxf(x / sc, -6.0f), 6.0f));
    ck[(size_t)grp * HD + c] = __float2bfloat16_rn(q * sc);
}

// Device-start form (the engine step): rows are positions start .. start+n-1 with start read from the device. Block k pools the k-th
// completed pair of the call (a pair completes at an odd position: row r = 2k when start is odd, 2k+1 when even; the earlier token of
// the pair is the state slot 0 when r == 0, else row r-1 of this call). Same arithmetic as k_comp_pool. nrm row k = completion k.
// The state update (k_comp_state) runs after it: it reads the state, so it cannot be in the same launch.
// SESS > 1: grid y = sessions; session j = rows j*rps .. (rps = n / gridDim.y) of slot dsess[j*rps] (its ck / kvs / scs slab), at
// positions dstart[j*rps] ..; nrm row j*((rps+1)/2) + k. gridDim.y = 1 and dsess = nullptr: the one-session form.
__global__ void __launch_bounds__(512) k_comp_pool_dev(const float* __restrict__ kvf, const float* __restrict__ scf, int n, const int* dstart,
                                                       const bf16* __restrict__ nw, const float* __restrict__ inv, bf16* ck,
                                                       const float* __restrict__ kvs, const float* __restrict__ scs, bf16* nrm,
                                                       const int* dsess = nullptr, size_t ck_ss = 0)
{
    __shared__ float sh[32];
    const int c = threadIdx.x, k = blockIdx.x;
    const int rps = n / (int)gridDim.y, row0 = (int)blockIdx.y * rps;
    const int start = dsess ? __ldcv(dstart + row0) : start_of(0, dstart);
    const int r = (start & 1) ? 2 * k : 2 * k + 1;
    if (r >= rps) return;
    kvf += (size_t)row0 * HD;
    scf += (size_t)row0 * HD;
    if (nrm) nrm += (size_t)blockIdx.y * ((rps + 1) / 2) * HD;
    if (dsess)
    {
        const int s = dsess[row0];
        ck += (size_t)s * ck_ss;
        kvs += (size_t)s * 2 * HD;
        scs += (size_t)s * 2 * HD;
    }
    const int grp = (start + r) >> 1;
    float k0, s0;
    if (r == 0) { k0 = kvs[c]; s0 = scs[c]; }
    else { k0 = kvf[(size_t)(r - 1) * HD + c]; s0 = scf[(size_t)(r - 1) * HD + c]; }
    const float k1 = kvf[(size_t)r * HD + c], s1 = scf[(size_t)r * HD + c];
    const float mx = fmaxf(s0, s1);
    const float e0 = expf(s0 - mx), e1 = expf(s1 - mx), den = e0 + e1;
    const float pool = __fadd_rn(__fmul_rn(k0, e0 / den), __fmul_rn(k1, e1 / den));
    float x0 = bf16r(pool);
    const float ss = block_sum(x0 * x0, sh);
    const float rr = rsqrtf(ss / (float)HD + EPS);
    float x = bf16r(__bfloat162float(nw[c]) * (x0 * rr));
    if (nrm) nrm[(size_t)k * HD + c] = __float2bfloat16_rn(x);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)(2 * grp) * inv[i];
        float ra, rb;
        if ((c & 1) == 0) { rot_pair(x, partner, ang, false, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, false, ra, rb); x = bf16r(rb); }
    }
    float a = fabsf(x);
    for (int o = 8; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
    a = fmaxf(a, 6.0f * 0.001953125f);
    const float sc = f_e4m3((uint8_t)__nv_cvt_float_to_fp8(a / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float q = e2m1_rne(fminf(fmaxf(x / sc, -6.0f), 6.0f));
    ck[(size_t)grp * HD + c] = __float2bfloat16_rn(q * sc);
}
// After the call: when its last position is even, that token is the pending half of the next pair -> state slot 0.
// SESS > 1: one block per session (grid = sessions, rows as k_comp_pool_dev), each parks into its slot's kvs / scs.
__global__ void __launch_bounds__(512) k_comp_state(const float* __restrict__ kvf, const float* __restrict__ scf, int n, const int* dstart,
                                                    float* kvs, float* scs, const int* dsess = nullptr)
{
    const int c = threadIdx.x;
    const int rps = n / (int)gridDim.x, row0 = (int)blockIdx.x * rps;
    const int start = dsess ? __ldcv(dstart + row0) : start_of(0, dstart);
    const size_t so = dsess ? (size_t)dsess[row0] * 2 * HD : 0;
    if (((start + rps - 1) & 1) == 0)
    {
        kvs[so + c] = kvf[(size_t)(row0 + rps - 1) * HD + c];
        scs[so + c] = scf[(size_t)(row0 + rps - 1) * HD + c];
    }
}
// DSpark / MTP verify chunk, after the accept: dq = the last accepted position Pa, dacc = the accepted drafts a = Pa - start (device words;
// the accept kernel's q and acc). k_comp_state parked row n - 1 (or nothing) from all the chunk's rows, accepted or not. When Pa is even the
// next step starts odd and its first row pairs with state slot 0, which must hold the row of position Pa = row a of this chunk. Pa odd: nothing
// is pending. Reads this layer's S.ckv / S.csc, which stay intact until the next comp_row: launch after the accept, before the next step.
// SESS > 1: one block per session j of the draft step (grid = sessions): dq[j], dacc[j]; its verify rows start at row drow0[j] of the chunk
// and its state is slot dslot[j]'s kvs / scs slab. drow0 = dslot = nullptr, grid 1: the one-session form.
__global__ void __launch_bounds__(512) k_comp_commit(const int* dq, const int* dacc, const float* __restrict__ kvf, const float* __restrict__ scf,
                                                     float* kvs, float* scs, const int* drow0 = nullptr, const int* dslot = nullptr)
{
    const int j = blockIdx.x;
    if (__ldcv(dq + j) & 1) return;
    const int c = threadIdx.x, a = __ldcv(dacc + j) + (drow0 ? __ldcv(drow0 + j) : 0);
    const size_t so = dslot ? (size_t)__ldcv(dslot + j) * 2 * HD : 0;
    kvs[so + c] = kvf[(size_t)a * HD + c];
    scs[so + c] = scf[(size_t)a * HD + c];
}


// ------------------------------------------------------------------------------------------------ host side
// RoPE inverse frequencies [32] with the release's operation order in fp32 (pow in double rounded to fp32, then 1/x, then the YaRN
// blend in fp32 steps): bit-equal to torch's precompute_freqs_cis for theta 10000 (orig_len 0) and 160000 + YaRN (orig_len 65536).
// The intermediates are volatile so the compiler cannot fuse them into FMAs.
inline void inv_freqs_host(double base, int orig_len, float out[ROPE / 2], double factor = 16.0, double beta_fast = 32.0, double beta_slow = 1.0)
{
    const int dim = ROPE;
    for (int i = 0; i < dim / 2; i++) out[i] = 1.0f / (float)pow(base, (double)(2 * i) / dim);
    if (orig_len > 0)
    {
        auto cd = [&](double rot) { return dim * log(orig_len / (rot * 2 * M_PI)) / (2 * log(base)); };
        double low = floor(cd(beta_fast)), high = ceil(cd(beta_slow));
        if (low < 0) low = 0;
        if (high > dim - 1) high = dim - 1;
        const float den = (float)fmax(high - low, 1e-3);
        for (int i = 0; i < dim / 2; i++)
        {
            volatile float ramp = ((float)i - (float)low) / den;
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            volatile float smooth = 1.0f - ramp;
            volatile float a = out[i] / (float)factor;
            volatile float b = 1.0f - smooth;
            volatile float c = a * b;
            volatile float d = out[i] * smooth;
            out[i] = c + d;
        }
    }
}

// Tensor-core GEMM switch (GR8_DS_ATTN_TC, default off): launch_gemm sends M >= 9 to k_gemm_mx8t where the shapes allow (K % 128, N % 64, lda % 16, lsa % 4).
inline bool g_use_tc = false;
inline int g_tc_mfast = 0;   // launch order (GR8_DS_ATTN_TC_MF, default 0 = N tile fastest). 1 = the M tile is the fastest grid dimension for every launch, so the ~M/BM CTAs that read the same weight tile run together and the weights come from DRAM about once.
                             // 2 = auto: that order only for a weight of >= TC_MF_AUTO_BYTES (measured: x2.49 on the 157 MB Engram wkv at M 1024; 0.87-1.04 on the 3-10 MB attention weights, so never there)
constexpr size_t TC_MF_AUTO_BYTES = (size_t)100 << 20;   // fp8 weight bytes N * K * groups; between the largest attention weight (10.5 MB) and the Engram wkv (157 MB), clear of the 78.6 MB DSpark main_proj that nobody has timed at M > 128
inline bool tc_mfast_pick(int mode, int N, int K, int groups, unsigned mt) { return mode == 1 || (mode == 2 && mt >= 2 && (size_t)N * K * groups >= TC_MF_AUTO_BYTES); }
inline bool tc_ok(const GemmP& p) { return p.K % 128 == 0 && p.N % 64 == 0 && p.lda % 16 == 0 && p.lsa % 4 == 0 && p.M >= 1; }
template <int BM, int BN, int ST> inline void launch_tc_t(const GemmP& p, int groups, cudaStream_t st)
{
    static bool init = false;
    if (!init)
    {
        cudaFuncSetAttribute(k_gemm_mx8t<BM, BN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TcCfg<BM, BN, ST>::SMEM);
        init = true;
    }
    GemmP q = p;
    const unsigned nt = p.N / BN, mt = (p.M + BM - 1) / BM;
    q.mfast = tc_mfast_pick(g_tc_mfast, p.N, p.K, groups, mt) ? 1 : 0;
    k_gemm_mx8t<BM, BN, ST><<<q.mfast ? dim3(mt, nt, groups) : dim3(nt, mt, groups), 256, TcCfg<BM, BN, ST>::SMEM, st>>>(q);
}
// tile pick: the tallest row tile the M fills, the wider column tile when it still leaves >= 96 CTAs (48 SMs x 2)
inline void launch_gemm_tc(const GemmP& p, int groups, cudaStream_t st)
{
    const int BM = p.M >= 128 ? 128 : (p.M >= 64 ? 64 : 32);
    const long ctas128 = (long)(p.N / 128) * ((p.M + BM - 1) / BM) * groups;
    const bool wide = p.N % 128 == 0 && ctas128 >= 96;
    if (BM == 128) { if (wide) launch_tc_t<128, 128, 2>(p, groups, st); else launch_tc_t<128, 64, 3>(p, groups, st); }
    else if (BM == 64) { if (wide) launch_tc_t<64, 128, 3>(p, groups, st); else launch_tc_t<64, 64, 3>(p, groups, st); }
    else launch_tc_t<32, 64, 3>(p, groups, st);
}
inline void launch_comp_fp8(const uint8_t* A, const uint8_t* SA, const Fp8Mat& W, float* Y, int M, cudaStream_t st)
{
    constexpr int BM = 128, BN = 64, ST = 3;
    static bool init = false;
    if (!init)
    {
        if (cudaFuncSetAttribute(k_gemm_mx8t_f32<BM, BN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)TcCfg<BM, BN, ST>::SMEM) != cudaSuccess)
        {
            fprintf(stderr, "DS41 compressor FP8 admission RED: dynamic shared memory\n");
            abort();
        }
        init = true;
    }
    GemmP p{};
    p.A = A; p.SA = SA; p.W = W.w; p.SW = W.s;
    p.M = M; p.N = W.N; p.K = W.K;
    p.lda = HID; p.lsa = HID / 32; p.ldy = HD;
    p.mfast = 0;
    k_gemm_mx8t_f32<BM, BN, ST><<<dim3(HD / BN, (M + BM - 1) / BM), 256, TcCfg<BM, BN, ST>::SMEM, st>>>(p, Y);
}
inline void launch_act_quant(const bf16* x, int ldx, uint8_t* codes, int ldc, uint8_t* sc, int lds, int M, int K, cudaStream_t st)
{
    const long warps = (long)M * (K / 32);
    k_act_quant<<<(unsigned)((warps + 7) / 8), 256, 0, st>>>(x, ldx, codes, ldc, sc, lds, M, K);
}
// Y[M][N] = fp8gemm(act(codes, scales), W): groups > 1 for the block-diagonal o_a (group g reads columns g*K.. of A, rows g*N.. of W)
inline void launch_gemm(const uint8_t* A, int lda, const uint8_t* SA, int lsa, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups,
                        size_t a_gs, size_t sa_gs, size_t y_gs, cudaStream_t st)
{
    GemmP p;
    p.A = A; p.SA = SA; p.W = W.w; p.SW = W.s; p.Y = Y;
    p.M = M; p.N = W.N; p.K = W.K;
    p.lda = lda; p.lsa = lsa; p.ldy = ldy;
    p.a_gs = a_gs; p.sa_gs = sa_gs; p.y_gs = y_gs;
    p.w_gs = (size_t)W.N * W.K; p.sw_gs = (size_t)W.N * (W.K / 32);
    if (g_use_tc && M >= 9 && tc_ok(p)) { launch_gemm_tc(p, groups, st); return; }
    dim3 grid((W.N + GBN - 1) / GBN, (M + GBM - 1) / GBM, groups);
    k_gemm_mxfp8<<<grid, 256, 0, st>>>(p);
}

// Decode GEMV switch (a whole-process setting, read when a stage is launched): off by default; the weights need Fp8Mat::gv.
// The GEMV kernels are not in this TU: they wrap densev.cuh, whose non-inline __global__ symbols the rank TU (m4sim.cu) already defines, so
// ds41_attn_gv.cuh is included by the rank-side glue (ds41_attn_ops.cuh) and the single-TU test, and reaches this TU as two launcher hooks.
inline bool g_use_gv = false;
inline bool g_comp_fp8 = false; // GR8_DS_COMP_FP8, default off; lossy, quality not measured
inline int g_src_fast = 0;   // GR8_DS_SRC_FAST bits, set by ds41_attn_set_src_fast: 1 = comp_row uses k_f32gemv2 (bit-identical to k_f32gemm2); 2 = the rank TU routes idx wq_b through the GEMV
inline int g_win_floor = 0;  // ds41_attn_set_win_floor: the window floor attn_core puts in AttnP (bounded replay, eager prefill only); 0 = none
inline int g_attn_cpa = 0;   // GR8_DS_ATTN_CPA, set by ds41_attn_set_cpasync: 1 = launch_attn runs k_attn_cpa (cp.async K gather, bit-identical to k_attn)
inline int g_attn_split = 0; // GR8_DS_ATTN_SPLIT=1, default off; two CTAs per row/head-group plus online-softmax merge
inline int g_gv_tpc = 1, g_gv_ks = 0;   // output tiles per CTA; k-parts per tile (0 = the default pick, gv_pick_ks in ds41_attn_gv.cuh)
inline XfragFn g_fn_xfrag = nullptr;
inline GemvFn g_fn_gemv = nullptr;
inline bool use_gv(const Fp8Mat& W, int n) { return g_use_gv && n >= 1 && n <= 8 && W.gv != nullptr && g_fn_xfrag && g_fn_gemv; }
inline void launch_xfrag(const uint8_t* codes, int ldc, const uint8_t* sc, int lds, int M, int K, int groups, size_t gc, size_t gs, uint32_t* xf, cudaStream_t st)
{
    g_fn_xfrag(codes, ldc, sc, lds, M, K, groups, gc, gs, xf, st);
}
inline void launch_gemv(const uint32_t* xf, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups, size_t y_gs, cudaStream_t st, int tpc = 0, int ks = 0)
{
    g_fn_gemv(xf, W, Y, ldy, M, groups, y_gs, st, tpc ? tpc : g_gv_tpc, ks ? ks : g_gv_ks);
}
// one projection: act codes/scales A/SA already quantized -> Y (groups > 1: block-diagonal, as launch_gemm). gv_ready: xf already holds A's table.
inline void proj_mm(const Fp8Mat& W, const uint8_t* A, int lda, const uint8_t* SA, int lsa, uint32_t* xf, bool gv_ready, bf16* Y, int ldy, int M, int groups,
                    size_t a_gs, size_t sa_gs, size_t y_gs, cudaStream_t st)
{
    if (use_gv(W, M))
    {
        if (!gv_ready) launch_xfrag(A, lda, SA, lsa, M, W.K, groups, a_gs, sa_gs, xf, st);
        launch_gemv(xf, W, Y, ldy, M, groups, y_gs, st);
    }
    else launch_gemm(A, lda, SA, lsa, W, Y, ldy, M, groups, a_gs, sa_gs, y_gs, st);
}
// y[rows][ldy] = fp8gemm(act_quant(x[rows][K]), W) through the GEMV, any row count (8-row passes over the same records; <= 8 is one
// weight pass, the decode and verify case). x bf16 [rows][K]; codes u8 [8][K], scales u8 [8][K/32], xf u32 [K*4] are scratch; W.gv required.
inline int project_rows(const Fp8Mat& W, const bf16* x, int rows, bf16* y, int ldy, uint8_t* codes, uint8_t* scales, uint32_t* xf, cudaStream_t st)
{
    if (!W.gv || !g_fn_xfrag || !g_fn_gemv || rows < 1 || (W.K & 31) || (W.N & 15)) return 1;
    for (int r0 = 0; r0 < rows; r0 += 8)
    {
        const int m = rows - r0 < 8 ? rows - r0 : 8;
        launch_act_quant(x + (size_t)r0 * W.K, W.K, codes, W.K, scales, W.K / 32, m, W.K, st);
        launch_xfrag(codes, W.K, scales, W.K / 32, m, W.K, 1, 0, 0, xf, st);
        launch_gemv(xf, W, y + (size_t)r0 * ldy, ldy, m, 1, 0, st);
    }
    return 0;
}
inline void launch_attn(const AttnP& p, cudaStream_t st)
{
    static bool init = false;
    if (!init)
    {
        cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM);
        cudaFuncSetAttribute(k_attn_cpa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM_CPA);
        cudaFuncSetAttribute(k_attn_floor, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM);
        cudaFuncSetAttribute(k_attn_cpa_floor, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM_CPA);
        cudaFuncSetAttribute(k_attn_cpa_split_t<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM_CPA);
        init = true;
    }
    if (p.floor > 0) {   // bounded replay range B (ds41_attn_set_win_floor): the _floor kernels; floor 0 runs the plain kernels
        if (g_attn_cpa) k_attn_cpa_floor<<<dim3(p.n, p.heads / 16), AT_THREADS, AT_SMEM_CPA, st>>>(p);
        else k_attn_floor<<<dim3(p.n, p.heads / 16), AT_THREADS, AT_SMEM, st>>>(p);
        return;
    }
    // W-prefill passes a device dstart and therefore uses the decode-form ring path even at 4096 rows.
    // A width threshold, not p.decode, separates wide prefill from 8/32-session decode batches.
    if (g_attn_split && g_attn_cpa && p.split && split_eligible(p.n, p.WL, p.TK)) {
        k_attn_cpa_split_t<false><<<dim3(p.n, p.heads / 16, 2), AT_THREADS, AT_SMEM_CPA, st>>>(p);
        k_attn_cpa_split_merge<<<dim3(p.n, p.heads / 16), AT_THREADS, 0, st>>>(p);
    } else if (g_attn_cpa) k_attn_cpa<<<dim3(p.n, p.heads / 16), AT_THREADS, AT_SMEM_CPA, st>>>(p);
    else k_attn<<<dim3(p.n, p.heads / 16), AT_THREADS, AT_SMEM, st>>>(p);
}

// ---- the four stages of the step plan (runtime/ds41_ops.h): each is async on `st` and reads/writes only what its comment lists.
// Rows are positions start .. start+n-1 (prefill from 0, or one decode row); y = bf16 [n][5120], the sub-layer input.
//
// ST_ATTN_PROJ: S.qa, S.qr (the indexer reads qr), S.q (rope applied), S.kvrows (this chunk's window rows, fp8-valued), C.ring.
inline void attn_proj(const Weights& W, Cache& C, const Scratch& S, const bf16* y, int n, int start, int heads, cudaStream_t st,
                      const int* dstart = nullptr, const uint8_t* gathered_codes = nullptr, const uint8_t* gathered_scales = nullptr)
{
    if (!gathered_codes) launch_act_quant(y, HID, S.yq, HID, S.ys, HID / 32, n, HID, st);
    const uint8_t* yq = gathered_codes ? gathered_codes : S.yq;
    const uint8_t* ys = gathered_scales ? gathered_scales : S.ys;
    proj_mm(W.wq_a, yq, HID, ys, HID / 32, S.xf_y, false, S.qa, QA, n, 1, 0, 0, 0, st);
    k_rmsnorm<<<n, 256, 0, st>>>(S.qa, QA, W.q_norm, S.qr, QA, QA, EPS);
    launch_act_quant(S.qr, QA, S.qrq, QA, S.qrs, QA / 32, n, QA, st);
    proj_mm(W.wq_b, S.qrq, QA, S.qrs, QA / 32, S.xf_qr, false, S.q, heads * HD, n, 1, 0, 0, 0, st);
    k_rope_tail<<<(unsigned)(((long)n * heads * 32 + 255) / 256), 256, 0, st>>>(S.q, (long)heads * HD, n, heads, start, dstart, W.inv, 0, S.sess);
    proj_mm(W.wkv, yq, HID, ys, HID / 32, S.xf_y, use_gv(W.wq_a, n), S.kv0, HD, n, 1, 0, 0, 0, st);   // wq_a built y's table
    k_kv_prep<<<n, 512, 0, st>>>(S.kv0, W.kv_norm, W.inv, start, dstart, n, S.kvrows, C.ring, C.ring_rows, S.sess);
}
// ST_CA_ROW at the ratio-2 KV sources (layers 2, 8, 14; W.c_* set): the compressed row(s) completed by this chunk go to C.comp
// (row j = group j), the pending token to C.kvs / C.scs. Uses S.ckv / S.csc as scratch. Needs only y (not attn_proj's outputs).
// S.cnorm row k = the post-norm pre-rope row k completed by this call (the first row it completes is row 0). n > 1 at start > 0
// (verify chunk) pools the rows one position at a time, so it equals n single-row calls.
inline void comp_row(const Weights& W, Cache& C, const Scratch& S, const bf16* y, int n, int start, cudaStream_t st,
                     const int* dstart = nullptr)
{
    if (g_comp_fp8 && n >= 9)
    {
        if (!W.c_wkv8.w || !W.c_wkv8.s || !W.c_wgate8.w || !W.c_wgate8.s)
        {
            fprintf(stderr, "DS41 compressor FP8 prepack RED: missing codes/scales\n");
            abort();
        }
        // Quantize each BF16 source row once; the two FP8 projections consume the same codes/scales.
        launch_act_quant(y, HID, S.yq, HID, S.ys, HID / 32, n, HID, st);
        launch_comp_fp8(S.yq, S.ys, W.c_wkv8, S.ckv, n, st);
        launch_comp_fp8(S.yq, S.ys, W.c_wgate8, S.csc, n, st);
    }
    else if (!(g_src_fast & 1) || !launch_f32gemv2(y, W.c_wkv, W.c_wgate, S.ckv, S.csc, n, HD, HID, st))
        k_f32gemm2<<<dim3(HD / FB_N, (n + FB_M - 1) / FB_M, 2), 256, 0, st>>>(y, W.c_wkv, W.c_wgate, S.ckv, S.csc, n, HD, HID);
    if (dstart)
    {
        const int ns = S.sess ? S.ns : 1;   // sessions of n / ns rows each (the op checks n % ns == 0)
        k_comp_pool_dev<<<dim3((n / ns + 1) / 2, ns), 512, 0, st>>>(S.ckv, S.csc, n, dstart, W.c_norm, W.inv, C.comp, C.kvs, C.scs, S.cnorm,
                                                                  S.sess, C.comp_ss);
        k_comp_state<<<ns, 512, 0, st>>>(S.ckv, S.csc, n, dstart, C.kvs, C.scs, S.sess);
        return;
    }
    if (start == 0)
    {
        k_comp_pool<<<(n + 1) / 2, 512, 0, st>>>(S.ckv, S.csc, n, 0, W.c_norm, W.inv, C.comp, C.kvs, C.scs, nullptr, S.cnorm);
        return;
    }
    int done = 0;
    for (int r = 0; r < n; r++)
    {
        const int pos = start + r;
        k_comp_pool<<<1, 512, 0, st>>>(S.ckv + (size_t)r * HD, S.csc + (size_t)r * HD, 1, pos, W.c_norm, W.inv, C.comp, C.kvs, C.scs, nullptr,
                                       S.cnorm + (size_t)done * HD);
        if (pos & 1) done++;
    }
}
// k_comp_commit for one source layer (2, 8, 14); see the kernel
inline void comp_commit(const Cache& C, const Scratch& S, const int* dq, const int* dacc, cudaStream_t st, const int* drow0 = nullptr,
                        const int* dslot = nullptr, int ns = 1)
{
    k_comp_commit<<<ns, 512, 0, st>>>(dq, dacc, S.ckv, S.csc, C.kvs, C.scs, drow0, dslot);
}
// ST_ATTN_CORE: S.o = sparse attention of S.q over [window | topk list]. comp_src = the cache the compressed rows are read from
// (its own C.comp at a source, else the source layer's); topk / ldtk / TK = the indexer's list (TK = 0 on layers 0, 1: topk and
// comp_src unused). Reads S.q, S.kvrows (prefill) or C.ring (decode).
inline void attn_core(const Weights& W, const Cache& C, const bf16* comp_src, const Scratch& S, int n, int start, int heads,
                      const int* topk, int ldtk, int TK, cudaStream_t st, const int* dstart = nullptr)
{
    AttnP a;
    const bool decode = dstart != nullptr || start != 0;
    a.q = S.q; a.kvw = decode ? C.ring : S.kvrows; a.ck = comp_src; a.topk = topk; a.sink = W.sink; a.out = S.o;
    a.ldtk = ldtk; a.n = n; a.heads = heads; a.start = start; a.dstart = dstart;
    a.decode = decode; a.WL = decode ? WIN : (n < WIN ? n : WIN); a.ring_rows = C.ring_rows; a.TK = TK; a.scale = SOFTMAX_SCALE;
    a.floor = decode ? g_win_floor : 0;
    a.dsess = decode && dstart ? S.sess : nullptr; a.ck_ss = C.src_ss;
    a.split = S.attn_split;
    launch_attn(a, st);
}
// c2: the coalesced-epilogue TC GEMM for wo_b writes into direct-send slots (GR8_DS_PF_DIRECT=1, M=4096, N=5120, K=2048):
// Identical K-loop, cp.async staging, and tc_mma accumulation as k_gemm_mx8t<BM,BN,ST>. Stages the [128, 128] bf16 tile
// through existing dynamic shared memory (EP_STRIDE = BN + 8 = 136 bf16, conflict-free in 75,776 B tsm) and emits 16-byte
// (uint4) row-coalesced stores (256 B contiguous per half-warp row) into mapped memory.
template <int BM, int BN, int ST> __global__ void __launch_bounds__(256) k_gemm_mx8t_c2(GemmP p)
{
    using C = TcCfg<BM, BN, ST>;
    extern __shared__ __align__(16) uint8_t tsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    const uint8_t* A = p.A + blockIdx.z * p.a_gs;
    const uint8_t* SA = p.SA + blockIdx.z * p.sa_gs;
    const uint8_t* W = p.W + blockIdx.z * p.w_gs;
    const uint8_t* SW = p.SW + blockIdx.z * p.sw_gs;
    const int m0 = (p.mfast ? blockIdx.x : blockIdx.y) * BM, n0 = (p.mfast ? blockIdx.y : blockIdx.x) * BN, NK = p.K >> 7, KB = p.K >> 5;
    auto issue = [&](int s, int kt)
    {
        uint8_t* b = tsm + (size_t)s * C::STAGE;
        for (int c = tid; c < BM * 8; c += 256)
            tc_cp16(b + (c >> 3) * C::PITCH + (c & 7) * 16, A + (size_t)min(m0 + (c >> 3), p.M - 1) * p.lda + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BN * 8; c += 256)
            tc_cp16(b + C::A_B + (c >> 3) * C::PITCH + (c & 7) * 16, W + (size_t)(n0 + (c >> 3)) * p.K + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BM; c += 256) tc_cp4(b + C::A_B + C::W_B + c * 4, SA + (size_t)min(m0 + c, p.M - 1) * p.lsa + kt * 4);
        for (int c = tid; c < BN; c += 256) tc_cp4(b + C::A_B + C::W_B + C::SA_B + c * 4, SW + (size_t)(n0 + c) * KB + kt * 4);
    };
    float acc[C::MB][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MB; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
#pragma unroll
    for (int s = 0; s < ST - 1; s++)
    {
        if (s < NK) issue(s, s);
        tc_commit();
    }
    for (int kt = 0; kt < NK; kt++)
    {
        tc_wait<ST - 2>();
        __syncthreads();
        if (kt + ST - 1 < NK) issue((kt + ST - 1) % ST, kt + ST - 1);
        tc_commit();
        const uint8_t* b = tsm + (size_t)(kt % ST) * C::STAGE;
        const uint8_t* sA = b;
        const uint8_t* sW = b + C::A_B;
        const uint8_t* sSA = sW + C::W_B;
        const uint8_t* sSW = sSA + C::SA_B;
        uint32_t wsa[C::MB], wsw[C::NT];
#pragma unroll
        for (int mb = 0; mb < C::MB; mb++) wsa[mb] = *(const uint32_t*)(sSA + (wm * C::WM + mb * 16 + g + ((lane & 1) ? 8 : 0)) * 4);
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++) wsw[nt] = *(const uint32_t*)(sSW + (wn * C::WN + nt * 8 + g) * 4);
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            uint32_t a[C::MB][4], bb[C::NT][2];
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
            {
                const uint8_t* r0 = sA + (wm * C::WM + mb * 16 + g) * C::PITCH + ks * 32 + 4 * t;
                a[mb][0] = *(const uint32_t*)r0;
                a[mb][1] = *(const uint32_t*)(r0 + 8 * C::PITCH);
                a[mb][2] = *(const uint32_t*)(r0 + 16);
                a[mb][3] = *(const uint32_t*)(r0 + 8 * C::PITCH + 16);
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt++)
            {
                const uint8_t* c0 = sW + (wn * C::WN + nt * 8 + g) * C::PITCH + ks * 32 + 4 * t;
                bb[nt][0] = *(const uint32_t*)c0;
                bb[nt][1] = *(const uint32_t*)(c0 + 16);
            }
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
                for (int nt = 0; nt < C::NT; nt++) tc_mma(acc[mb][nt], a[mb], bb[nt][0], bb[nt][1], (wsa[mb] >> (8 * ks)) & 0xFFu, (wsw[nt] >> (8 * ks)) & 0xFFu);
        }
    }
    tc_wait<0>();
    __syncthreads();
    constexpr int EP_STRIDE = BN + 8;
    static_assert((size_t)BM * EP_STRIDE * sizeof(bf16) <= C::SMEM, "C2 epilogue staging fits in existing TcCfg SMEM");
    bf16* sY = reinterpret_cast<bf16*>(tsm);
#pragma unroll
    for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++)
        {
            const int c_loc = wn * C::WN + nt * 8 + 2 * t, r_loc = wm * C::WM + mb * 16 + g;
            *(__nv_bfloat162*)(sY + r_loc * EP_STRIDE + c_loc) = __floats2bfloat162_rn(acc[mb][nt][0], acc[mb][nt][1]);
            *(__nv_bfloat162*)(sY + (r_loc + 8) * EP_STRIDE + c_loc) = __floats2bfloat162_rn(acc[mb][nt][2], acc[mb][nt][3]);
        }
    __syncthreads();
    bf16* Y = p.Y + blockIdx.z * p.y_gs;
    const int col_u4 = tid & 15, row_base = tid >> 4;
#pragma unroll
    for (int r_step = 0; r_step < BM; r_step += 16)
    {
        const int r_loc = row_base + r_step, row = m0 + r_loc;
        if (row < p.M)
            *(uint4*)(Y + (size_t)row * p.ldy + n0 + col_u4 * 8) = *(const uint4*)(sY + r_loc * EP_STRIDE + col_u4 * 8);
    }
}
// Persistent wo_b arm (GR8_DS_WOB_PERSIST, default off); same mapped direct epilogue as c2, with 2-stage cp.async under CC12.x 99 KiB cap.
template <int BM, int BN, int ST> __global__ void __launch_bounds__(256) k_gemm_mx8t_c2_persist(GemmP p)
{
    using C = TcCfg<BM, BN, ST>;
    // CC 12.x permits at most 99 KiB per block. The sm_121a compiler's static
    // shared allocation for this kernel is 1024 B; the build gate checks the
    // actual cubin, and the launch gate checks cudaFuncGetAttributes as well.
    static_assert(C::SMEM + 1024 <= 99 * 1024, "persistent C2 exceeds CC12.x per-block shared-memory cap");
    extern __shared__ __align__(16) uint8_t tsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    const uint8_t* A = p.A + blockIdx.z * p.a_gs;
    const uint8_t* SA = p.SA + blockIdx.z * p.sa_gs;
    const uint8_t* W = p.W + blockIdx.z * p.w_gs;
    const uint8_t* SW = p.SW + blockIdx.z * p.sw_gs;
    const int mt = (p.M + BM - 1) / BM, nt = p.N / BN, NK = p.K >> 7, KB = p.K >> 5;
    // All CTAs with one n tile retain that weight column across their M tiles. A persistent
    // CTA owns (n, m0), (n, m0 + peers), ...; peers counts CTAs assigned to this n tile.
    const int n_tile = (int)blockIdx.x % nt;
    const int peer = (int)blockIdx.x / nt;
    const int peers = ((int)gridDim.x + nt - 1 - n_tile) / nt;
    for (int m_tile = peer; m_tile < mt; m_tile += peers)
    {
    const int m0 = m_tile * BM, n0 = n_tile * BN;
    auto issue = [&](int s, int kt)
    {
        uint8_t* b = tsm + (size_t)s * C::STAGE;
        for (int c = tid; c < BM * 8; c += 256)
            tc_cp16(b + (c >> 3) * C::PITCH + (c & 7) * 16, A + (size_t)min(m0 + (c >> 3), p.M - 1) * p.lda + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BN * 8; c += 256)
            tc_cp16(b + C::A_B + (c >> 3) * C::PITCH + (c & 7) * 16, W + (size_t)(n0 + (c >> 3)) * p.K + kt * 128 + (c & 7) * 16);
        for (int c = tid; c < BM; c += 256) tc_cp4(b + C::A_B + C::W_B + c * 4, SA + (size_t)min(m0 + c, p.M - 1) * p.lsa + kt * 4);
        for (int c = tid; c < BN; c += 256) tc_cp4(b + C::A_B + C::W_B + C::SA_B + c * 4, SW + (size_t)(n0 + c) * KB + kt * 4);
    };
    float acc[C::MB][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MB; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
#pragma unroll
    for (int s = 0; s < ST - 1; s++)
    {
        if (s < NK) issue(s, s);
        tc_commit();
    }
    for (int kt = 0; kt < NK; kt++)
    {
        tc_wait<ST - 2>();
        __syncthreads();
        if (kt + ST - 1 < NK) issue((kt + ST - 1) % ST, kt + ST - 1);
        tc_commit();
        const uint8_t* b = tsm + (size_t)(kt % ST) * C::STAGE;
        const uint8_t* sA = b;
        const uint8_t* sW = b + C::A_B;
        const uint8_t* sSA = sW + C::W_B;
        const uint8_t* sSW = sSA + C::SA_B;
        uint32_t wsa[C::MB], wsw[C::NT];
#pragma unroll
        for (int mb = 0; mb < C::MB; mb++) wsa[mb] = *(const uint32_t*)(sSA + (wm * C::WM + mb * 16 + g + ((lane & 1) ? 8 : 0)) * 4);
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++) wsw[nt] = *(const uint32_t*)(sSW + (wn * C::WN + nt * 8 + g) * 4);
#pragma unroll
        for (int ks = 0; ks < 4; ks++)
        {
            uint32_t a[C::MB][4], bb[C::NT][2];
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
            {
                const uint8_t* r0 = sA + (wm * C::WM + mb * 16 + g) * C::PITCH + ks * 32 + 4 * t;
                a[mb][0] = *(const uint32_t*)r0;
                a[mb][1] = *(const uint32_t*)(r0 + 8 * C::PITCH);
                a[mb][2] = *(const uint32_t*)(r0 + 16);
                a[mb][3] = *(const uint32_t*)(r0 + 8 * C::PITCH + 16);
            }
#pragma unroll
            for (int nt = 0; nt < C::NT; nt++)
            {
                const uint8_t* c0 = sW + (wn * C::WN + nt * 8 + g) * C::PITCH + ks * 32 + 4 * t;
                bb[nt][0] = *(const uint32_t*)c0;
                bb[nt][1] = *(const uint32_t*)(c0 + 16);
            }
#pragma unroll
            for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
                for (int nt = 0; nt < C::NT; nt++) tc_mma(acc[mb][nt], a[mb], bb[nt][0], bb[nt][1], (wsa[mb] >> (8 * ks)) & 0xFFu, (wsw[nt] >> (8 * ks)) & 0xFFu);
        }
    }
    tc_wait<0>();
    __syncthreads();
    constexpr int EP_STRIDE = BN + 8;
    static_assert((size_t)BM * EP_STRIDE * sizeof(bf16) <= C::SMEM, "C2 epilogue staging fits in existing TcCfg SMEM");
    bf16* sY = reinterpret_cast<bf16*>(tsm);
#pragma unroll
    for (int mb = 0; mb < C::MB; mb++)
#pragma unroll
        for (int nt = 0; nt < C::NT; nt++)
        {
            const int c_loc = wn * C::WN + nt * 8 + 2 * t, r_loc = wm * C::WM + mb * 16 + g;
            *(__nv_bfloat162*)(sY + r_loc * EP_STRIDE + c_loc) = __floats2bfloat162_rn(acc[mb][nt][0], acc[mb][nt][1]);
            *(__nv_bfloat162*)(sY + (r_loc + 8) * EP_STRIDE + c_loc) = __floats2bfloat162_rn(acc[mb][nt][2], acc[mb][nt][3]);
        }
    __syncthreads();
    bf16* Y = p.Y + blockIdx.z * p.y_gs;
    const int col_u4 = tid & 15, row_base = tid >> 4;
#pragma unroll
    for (int r_step = 0; r_step < BM; r_step += 16)
    {
        const int r_loc = row_base + r_step, row = m0 + r_loc;
        if (row < p.M)
            *(uint4*)(Y + (size_t)row * p.ldy + n0 + col_u4 * 8) = *(const uint4*)(sY + r_loc * EP_STRIDE + col_u4 * 8);
    }
    __syncthreads(); // all coalesced stores read sY before the next tile overwrites tsm
    }
}

// c2 dispatch trace. Host side only: no device code, no kernel, no SASS change. Off by default; with GR8_DS_C2_TRACE=1 every
// launch of the c2 epilogue is counted, so an A/B run shows which arm took the c2 branch. A caller prints
// ds41a::c2_dispatch_count() at stop.
inline unsigned long long& c2_dispatch_count_ref()
{
    static unsigned long long n = 0;
    return n;
}
inline unsigned long long c2_dispatch_count() { return c2_dispatch_count_ref(); }
inline bool c2_persist_on()
{
    static const bool on = knob::ds_wob_persist();
    return on;
}
inline bool c2_trace_on()
{
    static const bool on = knob::ds_c2_trace();
    return on;
}
template <int BM, int BN, int ST> inline void launch_tc_t_c2(const GemmP& p, int groups, cudaStream_t st)
{
    // The only call site of this launcher is the taken c2 branch in attn_out (after tc_ok(p)), so counting here counts exactly the
    // dispatches that ran, and the sealed source guard on that branch line stays byte-identical.
    if (c2_trace_on()) ++c2_dispatch_count_ref();
    static bool init = false;
    if (!init)
    {
        cudaFuncSetAttribute(k_gemm_mx8t_c2<BM, BN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TcCfg<BM, BN, ST>::SMEM);
        init = true;
    }
    if (c2_persist_on() && groups == 1 && BM == 128 && BN == 128 && ST == 2)
    {
        static bool persist_init = false;
        static unsigned persist_sm = 0;
        if (!persist_init)
        {
            int dev = 0, active = 0;
            cudaDeviceProp prop{};
            cudaFuncAttributes attr{};
            const cudaError_t e0 = cudaGetDevice(&dev);
            const cudaError_t e1 = e0 == cudaSuccess ? cudaGetDeviceProperties(&prop, dev) : e0;
            const cudaError_t e2 = e1 == cudaSuccess ? cudaFuncGetAttributes(&attr, k_gemm_mx8t_c2_persist<128, 128, 2>) : e1;
            if (e2 != cudaSuccess || attr.sharedSizeBytes + TcCfg<128, 128, 2>::SMEM > 99 * 1024)
            {
                fprintf(stderr, "DS41 wo_b persistent admission RED: static=%zu dynamic=%zu CC12.x cap=%d cuda=%d\n",
                        (size_t)attr.sharedSizeBytes, TcCfg<128, 128, 2>::SMEM, 99 * 1024, (int)e2);
                abort();
            }
            const cudaError_t e3 = cudaFuncSetAttribute(k_gemm_mx8t_c2_persist<128, 128, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TcCfg<128, 128, 2>::SMEM);
            const cudaError_t e4 = e3 == cudaSuccess ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, k_gemm_mx8t_c2_persist<128, 128, 2>, 256, TcCfg<128, 128, 2>::SMEM) : e3;
            if (e4 != cudaSuccess || active < 1 || prop.multiProcessorCount < 1)
            {
                fprintf(stderr, "DS41 wo_b persistent admission RED: cuda=%d CTAs/SM=%d SMs=%d\n", (int)e4, active, prop.multiProcessorCount);
                abort();
            }
            persist_sm = (unsigned)prop.multiProcessorCount;
            fprintf(stderr, "# DS41 wo_b persistent: CTAs/SM=%d SMs=%u dynamic_smem/CTA=%zu static_smem/CTA=%zu\n",
                    active, persist_sm, TcCfg<128, 128, 2>::SMEM, (size_t)attr.sharedSizeBytes);
            persist_init = true;
        }
        const unsigned mt = (p.M + 127) / 128, nt = p.N / 128;
        const unsigned sm = persist_sm;
        // At least one CTA per N tile is required by the fixed-column ownership map.
        const unsigned grid = sm < nt ? nt : (sm < mt * nt ? sm : mt * nt);
        k_gemm_mx8t_c2_persist<128, 128, 2><<<grid ? grid : mt * nt, 256, TcCfg<128, 128, 2>::SMEM, st>>>(p);
        return;
    }
    GemmP q = p;
    const unsigned nt = p.N / BN, mt = (p.M + BM - 1) / BM;
    q.mfast = tc_mfast_pick(g_tc_mfast, p.N, p.K, groups, mt) ? 1 : 0;
    k_gemm_mx8t_c2<BM, BN, ST><<<q.mfast ? dim3(mt, nt, groups) : dim3(nt, mt, groups), 256, TcCfg<BM, BN, ST>::SMEM, st>>>(q);
}
// ST_ATTN_OUT: out [n][5120] bf16 = this rank's o_b partial (sum across ranks at TP > 1) from S.o.
inline void attn_out(const Weights& W, const Scratch& S, int n, int start, int heads, bf16* out, cudaStream_t st,
                     const int* dstart = nullptr, bool direct_c2 = false)
{
    const int G = heads / HPG;
    k_o_prep<<<n * heads, 512, 0, st>>>(S.o, heads, start, dstart, W.inv, nullptr, S.oq, S.os, S.sess);
    proj_mm(W.wo_a, S.oq, heads * HD, S.os, heads * HD / 32, S.xf_o, false, S.oa, G * OLORA, n, G, GROUP_IN, GROUP_IN / 32, OLORA, st);
    launch_act_quant(S.oa, G * OLORA, S.oaq, G * OLORA, S.oas, G * OLORA / 32, n, G * OLORA, st);
    if (direct_c2 && g_use_tc && n == 4096 && W.wo_b.N == 5120 && W.wo_b.K == 2048 && G * OLORA == 2048 && !use_gv(W.wo_b, n))
    {
        GemmP p{};
        p.A = S.oaq; p.SA = S.oas; p.W = W.wo_b.w; p.SW = W.wo_b.s; p.Y = out;
        p.M = n; p.N = W.wo_b.N; p.K = W.wo_b.K;
        p.lda = G * OLORA; p.lsa = G * OLORA / 32; p.ldy = HID;
        p.a_gs = 0; p.sa_gs = 0; p.y_gs = 0;
        p.w_gs = (size_t)W.wo_b.N * W.wo_b.K; p.sw_gs = (size_t)W.wo_b.N * (W.wo_b.K / 32);
        if (tc_ok(p)) { launch_tc_t_c2<128, 128, 2>(p, 1, st); return; }
    }
    proj_mm(W.wo_b, S.oaq, G * OLORA, S.oas, G * OLORA / 32, S.xf_oa, false, out, HID, n, 1, 0, 0, 0, st);
}
// The whole layer in plan order (what the test and a standalone caller use). In the engine the index stage (ds41_idx.cuh) runs between
// comp_row and attn_core and produces topk.
inline void layer_forward(const Weights& W, Cache& C, const bf16* comp_src, const Scratch& S, const bf16* y, int n, int start, int heads,
                          const int* topk, int ldtk, int TK, bf16* out, cudaStream_t st, const int* dstart = nullptr)
{
    attn_proj(W, C, S, y, n, start, heads, st, dstart);
    if (W.c_wkv) comp_row(W, C, S, y, n, start, st, dstart);
    attn_core(W, C, comp_src, S, n, start, heads, topk, ldtk, TK, st, dstart);
    attn_out(W, S, n, start, heads, out, st, dstart);
}
}   // namespace ds41a

// Non-inline link targets for a TU that cannot include this header (the device kernels here would be defined twice in one
// binary, e.g. engine/runtime/g6m4.cu). Defined once in engine/attn/ds41_attn.cu; a caller declares them and prints them at stop.
unsigned long long ds41_c2_dispatch_count();
bool ds41_c2_trace_on();
