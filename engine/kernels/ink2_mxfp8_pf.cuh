// Blackwell (sm_121a) Native MXFP8 Engine for Inkling-2-Mini:
//   1. Dense projections (wq/wr/gate/sink/wk/wv/wo, loop ctrl down, optional lm_head):
//      - M > 64 (prefill): 8-warp 128x64 cp.async double-buffered mma.m16n8k32.e4m3 (418 TFLOPS)
//      - 16 <= M <= 64 (C=6 verify): 4-warp 64x(MB*16) mma.m16n8k32.e4m3
//      - 1 <= M <= 15 (solo decode/verify): coalesced 128-bit FP8 E4M3 GEMV (2x less weight BW than BF16!)
//   2. Cached Physical-Layer MXFP8 Grouped MoE Prefill (M >= 128, GR8_INK2_PF_MX8_MOE=1|2):
//      - Decodes exlr8 PMCGX trellis experts (including H128 + suh + svh) ONCE at boot into MXFP8.
//      - Runtime prefill executes pure grouped mma.m16n8k32.e4m3 (fused gate+up+softcapped SwiGLU,
//        then down * rw + atomicAdd combine — zero Hadamard, zero trellis unpack, zero moe_combine!).
//      - Decode/verify (M < 128) stays 100% untouched W*A16 exlr8 trellis!
#pragma once
#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "ink2_tr3_moe.cuh"
#include "../runtime/ink2_prof.cuh"

namespace ink2 {
namespace mx8 {

constexpr int MX8_REC = 528; // 16 rows x 32 k: 512 B E4M3 A-fragments (32 lanes x 16 B) + 16 B UE8M0 row scales

struct Mx8Mat {
    uint8_t* q = nullptr;    // [N, K] E4M3
    uint8_t* sc = nullptr;   // [N, K / 32] UE8M0
    int N = 0;
    int K = 0;
    bool ready() const { return q != nullptr && sc != nullptr && N > 0 && K > 0; }
};

struct MoeMx8Layer {
    uint8_t* gu_q = nullptr;    // [NE, 2*I, H] E4M3 (rows 0..I-1 = gate, rows I..2*I-1 = up)
    uint8_t* gu_sc = nullptr;   // [NE, 2*I, H/32] UE8M0
    uint8_t* dn_q = nullptr;    // [NE, H, I] E4M3
    uint8_t* dn_sc = nullptr;   // [NE, H, I/32] UE8M0
    int NE = 0, H = 0, I = 0;
    bool ready() const { return gu_q != nullptr && dn_q != nullptr; }
};

// Per-CTA descriptor for grouped MXFP8 MoE: up to 4 consecutive 16-row blocks (<=64 tokens) of ONE expert.
struct MoeTileMap4 {
    int expert;       // 0..NE-1
    int rb0;          // starting 16-row block index in x8 / a8 / tok_list
    int nb;           // 1..4 (number of 16-row blocks owned by this CTA)
    int m_here;       // actual valid tokens in [1 .. nb*16]
};

struct MoeMx8Scratch {
    uint8_t* x8 = nullptr;          // [max_rb][H/32][528]
    uint16_t* act_bf16 = nullptr;   // [max_rb * 16][I]
    uint8_t* a8 = nullptr;          // [max_rb][I/32][528]
    float* out_f32 = nullptr;       // [max_M][H]
    int* tok_list = nullptr;        // [max_rb * 16]
    float* rw_list = nullptr;       // [max_rb * 16]
    MoeTileMap4* map4 = nullptr;    // [max_cta_y]
    int* d_counts = nullptr;        // [2]: {total_rb, total_cta_y}
    int max_rb = 0;
    int max_cta_y = 0;
    int max_M = 0;
};

__device__ __forceinline__ void mx8_cp16(void* dst_smem, const void* src_gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(s), "l"(src_gmem));
#else
    *static_cast<uint4*>(dst_smem) = *static_cast<const uint4*>(src_gmem);
#endif
}
__device__ __forceinline__ void mx8_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void mx8_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
__device__ __forceinline__ void mx8_ldsm_x4(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#endif
}
__device__ __forceinline__ void mx8_mma(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1,
                                        uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1),
                   "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
#else
    (void)d; (void)a; (void)b0; (void)b1; (void)sa; (void)sb;
#endif
}

__device__ __forceinline__ uint32_t mx8_scale_ue8m0(float amax, float& inv) {
    if (amax <= 1e-30f) { inv = 0.0f; return 127u; }
    // Smallest power of 2 >= amax / 448.0f
    const float raw = amax * (1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(raw);
    uint32_t exp = (bits >> 23) & 0xffu;
    if ((bits & 0x7fffffu) != 0u) ++exp;
    if (exp < 1u) exp = 1u;
    if (exp > 254u) exp = 254u;
    const uint32_t inv_exp = 254u - exp;
    inv = __uint_as_float(inv_exp << 23);
    return exp;
}

__device__ __forceinline__ uint32_t mx8_f2_e4m3x2(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(a), "f"(b));
    return r;
#else
    const uint32_t lo = static_cast<uint8_t>(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3));
    const uint32_t hi = static_cast<uint8_t>(__nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3));
    return lo | (hi << 16);
#endif
}

__device__ __forceinline__ uint32_t mx8_pack_bf16x2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float mx8_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float mx8_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------- Boot Weight Quantization (BF16 -> MXFP8)
// Quantize BF16 W[N, K] -> E4M3 q[N, K] + UE8M0 sc[N, K/32]. Grid: ceil(N*(K/32)/8), Block: 256.
__global__ void __launch_bounds__(256) k_ink2_bf16_to_mx8_w(
    const uint16_t* __restrict__ W,
    uint8_t* __restrict__ q,
    uint8_t* __restrict__ sc,
    int N, int K) {
    const int nb_row = K >> 5;
    const size_t total = static_cast<size_t>(N) * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total) return;

    const size_t base = blk * 32 + lane;
    const float w = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(W + base));
    const float amax = mx8_warp_max(fabsf(w));
    float inv;
    const uint32_t code = mx8_scale_ue8m0(amax, inv);
    if (lane == 0) sc[blk] = static_cast<uint8_t>(code);
    q[base] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(w * inv, __NV_SATFINITE, __NV_E4M3));
}

inline void quantize_w(const uint16_t* W, Mx8Mat& dst, int N, int K, cudaStream_t st = 0) {
    dst.N = N; dst.K = K;
    const size_t total = static_cast<size_t>(N) * (K >> 5);
    k_ink2_bf16_to_mx8_w<<<static_cast<unsigned>((total + 7) / 8), 256, 0, st>>>(W, dst.q, dst.sc, N, K);
}

// ------------------------------------------------------------------------------------------- Runtime Activation Quantization (BF16 -> MX8_REC)
__global__ void __launch_bounds__(256) k_ink2_bf16_to_mx8_a(
    const uint16_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int M, int K) {
    __shared__ __align__(16) uint8_t sr[16 * MX8_REC];
    const int S = K >> 5;
    const int ST = (S + 15) >> 4;
    const int rb = blockIdx.x / ST;
    const int s0 = (blockIdx.x % ST) << 4;
    const int sl = threadIdx.x & 15;
    const int rr = threadIdx.x >> 4;
    const int s = s0 + sl;
    const int row = (rb << 4) + rr;

    if (s < S) {
        uint8_t* rec = sr + sl * MX8_REC;
        uint32_t c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        uint32_t code = 127u;
        if (row < M) {
            const uint4* p = reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * K + (s << 5));
            const uint4 q[4] = {p[0], p[1], p[2], p[3]};
            float2 f[16];
            float amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
                f[4 * j]     = __bfloat1622float2(bp[0]);
                f[4 * j + 1] = __bfloat1622float2(bp[1]);
                f[4 * j + 2] = __bfloat1622float2(bp[2]);
                f[4 * j + 3] = __bfloat1622float2(bp[3]);
            }
            #pragma unroll
            for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
            float inv;
            code = mx8_scale_ue8m0(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = mx8_f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                       (mx8_f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                *reinterpret_cast<uint32_t*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
            }
        }
        rec[512 + rr] = static_cast<uint8_t>(code);
    }
    __syncthreads();

    const int nv = min(16, S - s0) * (MX8_REC / 16);
    uint4* d = reinterpret_cast<uint4*>(dst + (static_cast<size_t>(rb) * S + s0) * MX8_REC);
    const uint4* v = reinterpret_cast<const uint4*>(sr);
    for (int i = threadIdx.x; i < nv; i += 256) d[i] = v[i];
}

inline void bf16_to_mx8_a(const uint16_t* src, uint8_t* dst, int M, int K, cudaStream_t st) {
    const int RB = (M + 15) >> 4;
    const int ST = ((K >> 5) + 15) >> 4;
    k_ink2_bf16_to_mx8_a<<<RB * ST, 256, 0, st>>>(src, dst, M, K);
}

// Fused 3-input quantizer for looped/global layers (ki, vi, qi -> x8_k, x8_v, x8_q in 1 launch!)
__global__ void __launch_bounds__(256) k_ink2_bf16_to_mx8_a3(
    const uint16_t* __restrict__ src0,
    const uint16_t* __restrict__ src1,
    const uint16_t* __restrict__ src2,
    uint8_t* __restrict__ dst0,
    uint8_t* __restrict__ dst1,
    uint8_t* __restrict__ dst2,
    int M, int K) {
    __shared__ __align__(16) uint8_t sr[16 * MX8_REC];
    const int S = K >> 5;
    const int ST = (S + 15) >> 4;
    const int which = blockIdx.y;
    const uint16_t* src = (which == 0) ? src0 : ((which == 1) ? src1 : src2);
    uint8_t* dst = (which == 0) ? dst0 : ((which == 1) ? dst1 : dst2);
    const int rb = blockIdx.x / ST;
    const int s0 = (blockIdx.x % ST) << 4;
    const int sl = threadIdx.x & 15;
    const int rr = threadIdx.x >> 4;
    const int s = s0 + sl;
    const int row = (rb << 4) + rr;

    if (s < S) {
        uint8_t* rec = sr + sl * MX8_REC;
        uint32_t c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        uint32_t code = 127u;
        if (row < M) {
            const uint4* p = reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * K + (s << 5));
            const uint4 q[4] = {p[0], p[1], p[2], p[3]};
            float2 f[16];
            float amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
                f[4 * j]     = __bfloat1622float2(bp[0]);
                f[4 * j + 1] = __bfloat1622float2(bp[1]);
                f[4 * j + 2] = __bfloat1622float2(bp[2]);
                f[4 * j + 3] = __bfloat1622float2(bp[3]);
            }
            #pragma unroll
            for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
            float inv;
            code = mx8_scale_ue8m0(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = mx8_f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                       (mx8_f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                *reinterpret_cast<uint32_t*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
            }
        }
        rec[512 + rr] = static_cast<uint8_t>(code);
    }
    __syncthreads();

    const int nv = min(16, S - s0) * (MX8_REC / 16);
    uint4* d = reinterpret_cast<uint4*>(dst + (static_cast<size_t>(rb) * S + s0) * MX8_REC);
    const uint4* v = reinterpret_cast<const uint4*>(sr);
    for (int i = threadIdx.x; i < nv; i += 256) d[i] = v[i];
}

inline void bf16_to_mx8_a3(const uint16_t* s0, const uint16_t* s1, const uint16_t* s2,
                           uint8_t* d0, uint8_t* d1, uint8_t* d2, int M, int K, cudaStream_t st) {
    const int RB = (M + 15) >> 4;
    const int ST = ((K >> 5) + 15) >> 4;
    k_ink2_bf16_to_mx8_a3<<<dim3(RB * ST, 3), 256, 0, st>>>(s0, s1, s2, d0, d1, d2, M, K);
}

__device__ __forceinline__ float mx8_h16_sum(float v) {
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------- Dense MXFP8 GEMV (M = 1..15)
__device__ __forceinline__ void mx8_fp8x4_unpack_raw(uint32_t v, float2& f0, float2& f1) {
    uint32_t r0, r1;
    const uint16_t lo = static_cast<uint16_t>(v & 0xffffu);
    const uint16_t hi = static_cast<uint16_t>(v >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r0) : "h"(lo));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r1) : "h"(hi));
    f0 = __half22float2(*reinterpret_cast<const __half2*>(&r0));
    f1 = __half22float2(*reinterpret_cast<const __half2*>(&r1));
}

__device__ __forceinline__ float mx8_dot_bf16x16(uint4 a0, uint4 a1, const float2 (&wf)[8]) {
    const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
    const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(p0[i]);
        s = fmaf(fa.x, wf[i].x, fmaf(fa.y, wf[i].y, s));
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(p1[i]);
        s = fmaf(fa.x, wf[4 + i].x, fmaf(fa.y, wf[4 + i].y, s));
    }
    return s;
}

// Coalesced 16-lane/row (8 rows/CTA) 2x-pipelined MXFP8 GEMV for M <= 15 (exact K16%16==0 on H=2304 & nq=4096!).
template <int MAX_M, bool EXACT_M, bool OUT_F32 = false>
__global__ void __launch_bounds__(128) k_ink2_mx8_gemv(
    const uint16_t* __restrict__ X,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    void* __restrict__ C,
    int M, int N, int K) {
    const int hwarp = threadIdx.x >> 4;
    const int hlane = threadIdx.x & 15;
    const int o = blockIdx.x * 8 + hwarp;
    if (o >= N) return;

    const int K16 = K >> 4;
    const int NB = K >> 5;
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(o) * K);
    const uint8_t* srow = Bsc + static_cast<size_t>(o) * NB;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    int i = hlane;
    uint4 wv = (i < K16) ? __ldcs(qrow + i) : make_uint4(0, 0, 0, 0);
    uint8_t sv = (i < K16) ? __ldcs(srow + (i >> 1)) : 127u;
    for (; i < K16; i += 16) {
        const int ni = i + 16;
        uint4 wv_next;
        uint8_t sv_next = 127u;
        if (ni < K16) {
            wv_next = __ldcs(qrow + ni);
            sv_next = __ldcs(srow + (ni >> 1));
        }
        const float sc = __uint_as_float(static_cast<uint32_t>(sv) << 23);
        float2 wf[8];
        mx8_fp8x4_unpack_raw(wv.x, wf[0], wf[1]);
        mx8_fp8x4_unpack_raw(wv.y, wf[2], wf[3]);
        mx8_fp8x4_unpack_raw(wv.z, wf[4], wf[5]);
        mx8_fp8x4_unpack_raw(wv.w, wf[6], wf[7]);

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K);
                acc[m] = fmaf(sc, mx8_dot_bf16x16(__ldg(xm + 2 * i), __ldg(xm + 2 * i + 1), wf), acc[m]);
            }
        }
        wv = wv_next;
        sv = sv_next;
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = mx8_h16_sum(acc[m]);
            if (hlane == 0) {
                if constexpr (OUT_F32) static_cast<float*>(C)[static_cast<size_t>(m) * N + o] = tot;
                else {
                    __nv_bfloat16 bh = __float2bfloat16(tot);
                    static_cast<uint16_t*>(C)[static_cast<size_t>(m) * N + o] = *reinterpret_cast<uint16_t*>(&bh);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------- Dense MXFP8 TC GEMM (M >= 16)
template <int MB, bool OUT_F32 = false>
__global__ void __launch_bounds__(128) k_ink2_gemm_mx8_small(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    void* __restrict__ C,
    int M, int N, int K) {
    constexpr int KS = 4;
    constexpr int ST = 2;
    constexpr int A_BYTES = KS * MB * MX8_REC;
    constexpr int B_BYTES = 64 * KS * 32;
    constexpr int STAGE = A_BYTES + B_BYTES;
    __shared__ __align__(16) uint8_t sm[ST * STAGE];

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = K >> 5, RB = (M + 15) >> 4, NST = (S + KS - 1) / KS;
    const int c0 = blockIdx.x * 64, rb0 = blockIdx.y * MB;
    const int nb = min(MB, RB - rb0), ncols = min(64, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * STAGE;
        for (int c = t; c < KS * MB * 33; c += 128) {
            const int j = c / (MB * 33), rem = c % (MB * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + A_BYTES;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 128 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                         Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    const uint8_t* sp = Bsc + static_cast<size_t>(min(N - 1, c0 + colw + (lane >> 2))) * S;
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = c0 + colw + (lane >> 2);
            v[j][0] = (r0 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };
    (void)sp;

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * STAGE;
        const uint8_t* b = a + A_BYTES;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = colw + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * MB + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = c0 + colw + h * 8 + 2 * c;
            if (col >= N) continue;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][h][0], acc[mb][h][1]);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][h][2], acc[mb][h][3]);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = mx8_pack_bf16x2(acc[mb][h][0], acc[mb][h][1]);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = mx8_pack_bf16x2(acc[mb][h][2], acc[mb][h][3]);
            }
        }
    }
}

struct Mx8LargeCfg {
    static constexpr int BM = 4;
    static constexpr int KS = 4;
    static constexpr int ST = 2;
    static constexpr int A = KS * BM * MX8_REC;  // 8448 B
    static constexpr int B = 128 * KS * 32;      // 16384 B
    static constexpr int STAGE = A + B;          // 24832 B
    static constexpr int SMEM = ST * STAGE;      // 49664 B (48.5 KB -> 2 CTAs/SM!)
};

template <bool OUT_F32 = false>
__global__ void __launch_bounds__(256, 2) k_ink2_gemm_mx8_large(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    void* __restrict__ C,
    int M, int N, int K) {
    using Cfg = Mx8LargeCfg;
    constexpr int BM = Cfg::BM, KS = Cfg::KS;
    extern __shared__ __align__(16) uint8_t sm[];

    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int by = lin % gridDim.y;
    const int bx = lin / gridDim.y;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = K >> 5, RB = (M + 15) >> 4, NST = (S + KS - 1) / KS;
    const int c0 = bx * 128, rb0 = by * BM;
    const int nb = min(BM, RB - rb0), ncols = min(128, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * Cfg::STAGE;
        for (int c = t; c < KS * BM * 33; c += 256) {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                         Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = c0 + colw + (lane >> 2);
            v[j][0] = (r0 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[BM][2][4];
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * Cfg::STAGE;
        const uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = colw + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = c0 + colw + nt * 8 + 2 * c;
            if (col >= N) continue;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][nt][0], acc[mb][nt][1]);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][nt][2], acc[mb][nt][3]);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = mx8_pack_bf16x2(acc[mb][nt][0], acc[mb][nt][1]);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = mx8_pack_bf16x2(acc[mb][nt][2], acc[mb][nt][3]);
            }
        }
    }
}

inline bool mx8_supported() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    return major >= 12;
}

inline int& mx8_tc_min_m_ref() { static int v = 9; return v; }
inline int mx8_tc_min_m() { return mx8_tc_min_m_ref(); }

template <bool OUT_F32 = false>
inline void gemm_mx8(const uint16_t* X_bf16, const uint8_t* x8, const Mx8Mat& B, void* C, int M, cudaStream_t st) {
    if (M < mx8_tc_min_m() && X_bf16 != nullptr) {
        const unsigned g = static_cast<unsigned>((B.N + 7) / 8);
        if (M == 1) k_ink2_mx8_gemv<1, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 1, B.N, B.K);
        else if (M == 2) k_ink2_mx8_gemv<2, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 2, B.N, B.K);
        else if (M == 3) k_ink2_mx8_gemv<3, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 3, B.N, B.K);
        else if (M == 4) k_ink2_mx8_gemv<4, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 4, B.N, B.K);
        else if (M == 8) k_ink2_mx8_gemv<8, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 8, B.N, B.K);
        else if (M <= 8) k_ink2_mx8_gemv<8, false, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, M, B.N, B.K);
        else {
            k_ink2_mx8_gemv<8, true, OUT_F32><<<g, 128, 0, st>>>(X_bf16, B.q, B.sc, C, 8, B.N, B.K);
            const int rem = M - 8;
            void* C2 = OUT_F32
                ? static_cast<void*>(static_cast<float*>(C) + static_cast<size_t>(8) * B.N)
                : static_cast<void*>(static_cast<uint16_t*>(C) + static_cast<size_t>(8) * B.N);
            k_ink2_mx8_gemv<8, false, OUT_F32><<<g, 128, 0, st>>>(X_bf16 + static_cast<size_t>(8) * B.K, B.q, B.sc, C2, rem, B.N, B.K);
        }
        return;
    }
    const int RB = (M + 15) >> 4;
    if (M <= 16) {
        k_ink2_gemm_mx8_small<1, OUT_F32><<<dim3((B.N + 63) / 64, RB), 128, 0, st>>>(x8, B.q, B.sc, C, M, B.N, B.K);
    } else if (M <= 32) {
        k_ink2_gemm_mx8_small<2, OUT_F32><<<dim3((B.N + 63) / 64, (RB + 1) / 2), 128, 0, st>>>(x8, B.q, B.sc, C, M, B.N, B.K);
    } else if (M <= 64) {
        k_ink2_gemm_mx8_small<4, OUT_F32><<<dim3((B.N + 63) / 64, (RB + 3) / 4), 128, 0, st>>>(x8, B.q, B.sc, C, M, B.N, B.K);
    } else {
        k_ink2_gemm_mx8_large<OUT_F32><<<dim3((B.N + 127) / 128, (RB + 3) / 4), 256, Mx8LargeCfg::SMEM, st>>>(x8, B.q, B.sc, C, M, B.N, B.K);
    }
}

struct Mx8MultiItem {
    const uint16_t* X;
    const uint8_t* x8;
    const uint8_t* Bq;
    const uint8_t* Bsc;
    uint16_t* C;
    int N;
    int off0;
};
struct Mx8MultiArgs {
    Mx8MultiItem t[6];
    int nt;
    int total_n;
};

template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_ink2_mx8_gemv_multi(
    Mx8MultiArgs a,
    int M, int K) {
    const int hwarp = threadIdx.x >> 4;
    const int hlane = threadIdx.x & 15;
    const int g_o = blockIdx.x * 8 + hwarp;
    if (g_o >= a.total_n) return;

    int ti = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (i < a.nt && g_o >= a.t[i].off0) ti = i;
    }
    const int o = g_o - a.t[ti].off0;
    const int N = a.t[ti].N;
    const uint16_t* X = a.t[ti].X;
    const int K16 = K >> 4;
    const int NB = K >> 5;
    const uint4* qrow = reinterpret_cast<const uint4*>(a.t[ti].Bq + static_cast<size_t>(o) * K);
    const uint8_t* srow = a.t[ti].Bsc + static_cast<size_t>(o) * NB;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    int i = hlane;
    uint4 wv = (i < K16) ? __ldcs(qrow + i) : make_uint4(0, 0, 0, 0);
    uint8_t sv = (i < K16) ? __ldcs(srow + (i >> 1)) : 127u;
    for (; i < K16; i += 16) {
        const int ni = i + 16;
        uint4 wv_next;
        uint8_t sv_next = 127u;
        if (ni < K16) {
            wv_next = __ldcs(qrow + ni);
            sv_next = __ldcs(srow + (ni >> 1));
        }
        const float sc = __uint_as_float(static_cast<uint32_t>(sv) << 23);
        float2 wf[8];
        mx8_fp8x4_unpack_raw(wv.x, wf[0], wf[1]);
        mx8_fp8x4_unpack_raw(wv.y, wf[2], wf[3]);
        mx8_fp8x4_unpack_raw(wv.z, wf[4], wf[5]);
        mx8_fp8x4_unpack_raw(wv.w, wf[6], wf[7]);

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K);
                acc[m] = fmaf(sc, mx8_dot_bf16x16(__ldg(xm + 2 * i), __ldg(xm + 2 * i + 1), wf), acc[m]);
            }
        }
        wv = wv_next;
        sv = sv_next;
    }
    uint16_t* C = a.t[ti].C;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = mx8_h16_sum(acc[m]);
            if (hlane == 0) {
                __nv_bfloat16 bh = __float2bfloat16(tot);
                C[static_cast<size_t>(m) * N + o] = *reinterpret_cast<uint16_t*>(&bh);
            }
        }
    }
}

// Fused multi-target small TC GEMM (1 launch for all QKV/gate/sink/wr targets on M=8..64, supports distinct x8 per target!)
template <int MB>
__global__ void __launch_bounds__(128) k_ink2_gemm_mx8_small_multi(
    Mx8MultiArgs a_args,
    int M, int K) {
    constexpr int KS = 4, ST = 2;
    constexpr int A_BYTES = KS * MB * MX8_REC;
    constexpr int B_BYTES = 64 * KS * 32;
    constexpr int STAGE = A_BYTES + B_BYTES;
    __shared__ __align__(16) uint8_t sm[ST * STAGE];

    const int gx = blockIdx.x;
    int ti = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (i < a_args.nt && gx >= a_args.t[i].off0) ti = i;
    }
    const uint8_t* x8 = a_args.t[ti].x8;
    const uint8_t* Bq = a_args.t[ti].Bq;
    const uint8_t* Bsc = a_args.t[ti].Bsc;
    uint16_t* C = a_args.t[ti].C;
    const int N = a_args.t[ti].N;
    const int c0 = (gx - a_args.t[ti].off0) * 64;
    if (c0 >= N) return;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = K >> 5, RB = (M + 15) >> 4, NST = (S + KS - 1) / KS;
    const int rb0 = blockIdx.y * MB;
    const int nb = min(MB, RB - rb0), ncols = min(64, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * STAGE;
        for (int c = t; c < KS * MB * 33; c += 128) {
            const int j = c / (MB * 33), rem = c % (MB * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + A_BYTES;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 128 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                         Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = c0 + colw + (lane >> 2);
            v[j][0] = (r0 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * STAGE;
        const uint8_t* b = a + A_BYTES;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = colw + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * MB + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = c0 + colw + h * 8 + 2 * c;
            if (col >= N) continue;
            const int row = (rb0 + mb) * 16 + r;
            if (row < M) *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row) * N + col) = mx8_pack_bf16x2(acc[mb][h][0], acc[mb][h][1]);
            if (row + 8 < M) *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row + 8) * N + col) = mx8_pack_bf16x2(acc[mb][h][2], acc[mb][h][3]);
        }
    }
}

// Fused multi-target large TC GEMM (1 launch for all QKV/gate/sink/wr targets on prefill M > 64, distinct x8 per target!)
__global__ void __launch_bounds__(256, 2) k_ink2_gemm_mx8_large_multi(
    Mx8MultiArgs a_args,
    int M, int K) {
    using Cfg = Mx8LargeCfg;
    constexpr int BM = Cfg::BM, KS = Cfg::KS;
    extern __shared__ __align__(16) uint8_t sm[];

    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int by = lin % gridDim.y;
    const int bx = lin / gridDim.y;

    int ti = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (i < a_args.nt && bx >= a_args.t[i].off0) ti = i;
    }
    const uint8_t* x8 = a_args.t[ti].x8;
    const uint8_t* Bq = a_args.t[ti].Bq;
    const uint8_t* Bsc = a_args.t[ti].Bsc;
    uint16_t* C = a_args.t[ti].C;
    const int N = a_args.t[ti].N;
    const int c0 = (bx - a_args.t[ti].off0) * 128;
    if (c0 >= N) return;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = K >> 5, RB = (M + 15) >> 4, NST = (S + KS - 1) / KS;
    const int rb0 = by * BM;
    const int nb = min(BM, RB - rb0), ncols = min(128, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * Cfg::STAGE;
        for (int c = t; c < KS * BM * 33; c += 256) {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                         Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = c0 + colw + (lane >> 2);
            v[j][0] = (r0 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < N && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[BM][2][4];
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * Cfg::STAGE;
        const uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = colw + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = c0 + colw + nt * 8 + 2 * c;
            if (col >= N) continue;
            const int row = (rb0 + mb) * 16 + r;
            if (row < M) *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row) * N + col) = mx8_pack_bf16x2(acc[mb][nt][0], acc[mb][nt][1]);
            if (row + 8 < M) *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row + 8) * N + col) = mx8_pack_bf16x2(acc[mb][nt][2], acc[mb][nt][3]);
        }
    }
}

struct Mx8MultiSpec {
    const Mx8Mat* B;
    uint16_t* C;
    const uint16_t* X = nullptr;
    const uint8_t* x8 = nullptr;
};

inline void gemm_mx8_multi_dispatch(const Mx8MultiSpec* specs, int nt, int M, int K, cudaStream_t st) {
    if (nt <= 0) return;
    if (M < mx8_tc_min_m()) {
        Mx8MultiArgs a{};
        a.nt = nt;
        int cur = 0;
        for (int i = 0; i < nt; ++i) {
            a.t[i] = {specs[i].X, nullptr, specs[i].B->q, specs[i].B->sc, specs[i].C, specs[i].B->N, cur};
            cur += specs[i].B->N;
        }
        a.total_n = cur;
        const unsigned g = static_cast<unsigned>((cur + 7) / 8);
        if (M == 1) k_ink2_mx8_gemv_multi<1, true><<<g, 128, 0, st>>>(a, 1, K);
        else if (M == 2) k_ink2_mx8_gemv_multi<2, true><<<g, 128, 0, st>>>(a, 2, K);
        else if (M == 3) k_ink2_mx8_gemv_multi<3, true><<<g, 128, 0, st>>>(a, 3, K);
        else if (M == 4) k_ink2_mx8_gemv_multi<4, true><<<g, 128, 0, st>>>(a, 4, K);
        else if (M == 8) k_ink2_mx8_gemv_multi<8, true><<<g, 128, 0, st>>>(a, 8, K);
        else if (M <= 8) k_ink2_mx8_gemv_multi<8, false><<<g, 128, 0, st>>>(a, M, K);
        else {
            k_ink2_mx8_gemv_multi<8, true><<<g, 128, 0, st>>>(a, 8, K);
            Mx8MultiArgs a2 = a;
            for (int i = 0; i < nt; ++i) {
                a2.t[i].X += static_cast<size_t>(8) * K;
                a2.t[i].C += static_cast<size_t>(8) * a2.t[i].N;
            }
            k_ink2_mx8_gemv_multi<8, false><<<g, 128, 0, st>>>(a2, M - 8, K);
        }
        return;
    }
    const int RB = (M + 15) >> 4;
    if (M <= 64) {
        Mx8MultiArgs a{};
        a.nt = nt;
        int cur_blk = 0;
        for (int i = 0; i < nt; ++i) {
            a.t[i] = {nullptr, specs[i].x8, specs[i].B->q, specs[i].B->sc, specs[i].C, specs[i].B->N, cur_blk};
            cur_blk += (specs[i].B->N + 63) / 64;
        }
        a.total_n = cur_blk;
        if (M <= 16) k_ink2_gemm_mx8_small_multi<1><<<dim3(cur_blk, RB), 128, 0, st>>>(a, M, K);
        else if (M <= 32) k_ink2_gemm_mx8_small_multi<2><<<dim3(cur_blk, (RB + 1) / 2), 128, 0, st>>>(a, M, K);
        else k_ink2_gemm_mx8_small_multi<4><<<dim3(cur_blk, (RB + 3) / 4), 128, 0, st>>>(a, M, K);
    } else {
        Mx8MultiArgs a{};
        a.nt = nt;
        int cur_blk = 0;
        for (int i = 0; i < nt; ++i) {
            a.t[i] = {nullptr, specs[i].x8, specs[i].B->q, specs[i].B->sc, specs[i].C, specs[i].B->N, cur_blk};
            cur_blk += (specs[i].B->N + 127) / 128;
        }
        a.total_n = cur_blk;
        k_ink2_gemm_mx8_large_multi<<<dim3(cur_blk, (RB + 3) / 4), 256, Mx8LargeCfg::SMEM, st>>>(a, M, K);
    }
}

inline void gemm_mx8_multi(const uint16_t* X_bf16, uint8_t* x8_buf, const Mx8MultiSpec* specs, int nt, int M, int K, cudaStream_t st) {
    if (nt <= 0) return;
    if (M >= mx8_tc_min_m()) bf16_to_mx8_a(X_bf16, x8_buf, M, K, st);
    Mx8MultiSpec s2[6];
    for (int i = 0; i < nt; ++i) {
        s2[i] = {specs[i].B, specs[i].C, X_bf16, x8_buf};
    }
    gemm_mx8_multi_dispatch(s2, nt, M, K, st);
}

inline void gemm_mx8_multi_3x(const uint16_t* ki, const uint16_t* vi, const uint16_t* qi,
                              uint8_t* x8_buf, size_t x8_stride,
                              const Mx8Mat& wk, const Mx8Mat& wv, const Mx8Mat& wq,
                              const Mx8Mat& wr, const Mx8Mat& gate, const Mx8Mat& sink,
                              uint16_t* kp, uint16_t* vp, uint16_t* q_raw,
                              uint16_t* r_raw, uint16_t* gate_out, uint16_t* sink_out,
                              int M, int K, cudaStream_t st) {
    uint8_t* x8_k = x8_buf;
    uint8_t* x8_v = x8_buf + x8_stride;
    uint8_t* x8_q = x8_buf + 2 * x8_stride;
    if (M >= mx8_tc_min_m()) bf16_to_mx8_a3(ki, vi, qi, x8_k, x8_v, x8_q, M, K, st);
    const Mx8MultiSpec specs[6] = {
        {&wk, kp, ki, x8_k},
        {&wv, vp, vi, x8_v},
        {&wq, q_raw, qi, x8_q},
        {&wr, r_raw, qi, x8_q},
        {&gate, gate_out, qi, x8_q},
        {&sink, sink_out, qi, x8_q},
    };
    gemm_mx8_multi_dispatch(specs, 6, M, K, st);
}

__global__ void __launch_bounds__(256) k_ink2_transpose_bf16(
    const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int rows, int cols) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    const size_t tot = static_cast<size_t>(rows) * cols;
    if (idx >= tot) return;
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx % cols);
    dst[static_cast<size_t>(c) * rows + r] = src[idx];
}

inline void transpose_bf16(const uint16_t* src, uint16_t* dst, int rows, int cols, cudaStream_t st = 0) {
    const size_t tot = static_cast<size_t>(rows) * cols;
    k_ink2_transpose_bf16<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(src, dst, rows, cols);
}

struct Mx8AttnMats {
    Mx8Mat wk, wv, wq, wr, gate, sink, wo;
    uint16_t* rel_proj_t = nullptr;  // [rel_win][rel_dim] transposed for coalesced decode attention
    uint8_t* x8_buf = nullptr;
    size_t x8_stride = 0;
    int mode = 0; // 0 = off, 1 = all M, 2 = prefill only (M >= 64)
    bool active(int M) const {
        if (mode == 0 || !wq.ready() || !wo.ready()) return false;
        return (mode == 1) || (mode == 2 && M >= 64);
    }
};

__global__ void __launch_bounds__(256) k_ink2_mx8_softcap_mask(float* __restrict__ logits, int M, int rows, int v0, int valid_vocab) {
    const size_t tot = static_cast<size_t>(M) * rows;
    const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (idx >= tot) return;
    const int o = static_cast<int>(idx % static_cast<size_t>(rows));
    if (v0 + o >= valid_vocab) logits[idx] = -INFINITY;
    else logits[idx] = 30.0f * tanhf(logits[idx] * (1.0f / 30.0f));
}

inline void lm_head_mx8(const uint16_t* hf, uint8_t* x8_buf, const Mx8Mat& lm_mx8, int M, int v0, int valid_vocab, float* logits, cudaStream_t st) {
    if (M >= mx8_tc_min_m()) bf16_to_mx8_a(hf, x8_buf, M, lm_mx8.K, st);
    gemm_mx8<true>(hf, x8_buf, lm_mx8, logits, M, st);
    const size_t tot = static_cast<size_t>(M) * lm_mx8.N;
    k_ink2_mx8_softcap_mask<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(logits, M, lm_mx8.N, v0, valid_vocab);
}

// ------------------------------------------------------------------------------------------- Boot Trellis -> MXFP8 MoE Decoder
// Decodes one 128x128 block (b, hb) of pre-widened tr3 FC1 (mat=0 gate, mat=1 up) -> exact W = diag(svh) H128 W_tr H128 diag(suh)
// and quantizes directly to MXFP8 gu_q[e][mat*I + b*128 + r][hb*128 + k] + gu_sc!
// Grid: (NE * 2 * eb * HB), Block: 256 (8 warps). Dynamic SMEM: 128 * 132 * 4 = 67584 B.
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_decode_fc1_mx8(
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1,      // [nblk][2][H] suh
    const __half* __restrict__ v2,      // [nblk][3][128] svh gate(0), up(1)
    const long long* __restrict__ tab,  // [NE][6]
    const long long* __restrict__ kmix,
    uint8_t* __restrict__ gu_q,         // [NE][2*I][H]
    uint8_t* __restrict__ gu_sc,        // [NE][2*I][H/32]
    int eb, int H) {
    extern __shared__ float sW[];       // [128][132]
    const int HB = H / 128;
    const int I = eb * 128;
    const int hb = blockIdx.x % HB;
    const int bm = (blockIdx.x / HB) % (2 * eb);
    const int e = blockIdx.x / (HB * 2 * eb);
    const int b = bm >> 1;
    const int mat = bm & 1;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const size_t brow = static_cast<size_t>(tr[2]) + b;

    const uint8_t* mw = nullptr;
    const int bits = tr3::fc1_unit(fc1, kmix, e, tr, bm, H, mw);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;

    auto decode_tile_to_smem = [&](auto tag_bits) {
        constexpr int BITS = decltype(tag_bits)::value;
        tr3::LaneDec<BITS, CB> tl;
        tl.init(lane);
        const size_t tb = static_cast<size_t>(32 * BITS);
        const int ks0 = hb * 8;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint8_t* tile = mw + (static_cast<size_t>(ks0 + p) * 8 + warp) * tb;
            const uint32_t raw = tl.loadw(tile, lane);
            uint32_t f0[2], f1[2];
            tl.decodew(raw, f0, f1);
            const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&f0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&f1[0]));
            const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&f0[1]));
            const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&f1[1]));
            const int rA = warp * 16 + row0;
            const int rB = rA + 8;
            const int c0 = p * 16 + 2 * col;
            sW[rA * 132 + c0]     = a0.x; sW[rA * 132 + c0 + 1] = a0.y;
            sW[rB * 132 + c0]     = a1.x; sW[rB * 132 + c0 + 1] = a1.y;
            sW[rA * 132 + c0 + 8] = a2.x; sW[rA * 132 + c0 + 9] = a2.y;
            sW[rB * 132 + c0 + 8] = a3.x; sW[rB * 132 + c0 + 9] = a3.y;
        }
    };
    if (bits == 3) decode_tile_to_smem(std::integral_constant<int, 3>{});
    else if (bits == 4) decode_tile_to_smem(std::integral_constant<int, 4>{});
    else decode_tile_to_smem(std::integral_constant<int, 2>{});
    __syncthreads();

    // 1. Row-wise H128 * suh (each warp handles 16 rows)
    const __half* suh = v1 + (brow * 2 + mat) * H + hb * 128 + lane * 4;
    float s_in[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s_in[i] = __half2float(suh[i]);
    for (int r = warp; r < 128; r += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = sW[r * 132 + lane * 4 + i];
        tr3::had128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) sW[r * 132 + lane * 4 + i] = v[i] * s_in[i];
    }
    __syncthreads();

    // 2. Col-wise H128 * svh (each warp handles 16 cols)
    const __half* svh = v2 + (brow * 3 + mat) * 128 + lane * 4;
    float s_out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s_out[i] = __half2float(svh[i]);
    for (int k = warp; k < 128; k += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = sW[(lane * 4 + i) * 132 + k];
        tr3::had128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) sW[(lane * 4 + i) * 132 + k] = v[i] * s_out[i];
    }
    __syncthreads();

    // 3. Quantize each row r=0..127 (4 blocks of 32 along k=0..127) to MXFP8
    const int S = H >> 5;
    for (int r = warp; r < 128; r += 8) {
        const size_t grow = (static_cast<size_t>(e) * (2 * I) + mat * I + b * 128 + r);
        uint8_t* qdst = gu_q + grow * H + hb * 128;
        uint8_t* sdst = gu_sc + grow * S + hb * 4;
        #pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            const float val = sW[r * 132 + blk * 32 + lane];
            const float amax = mx8_warp_max(fabsf(val));
            float inv;
            const uint32_t code = mx8_scale_ue8m0(amax, inv);
            if (lane == 0) sdst[blk] = static_cast<uint8_t>(code);
            qdst[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val * inv, __NV_SATFINITE, __NV_E4M3));
        }
    }
}

// Same for FC2: decodes [NE, H, I] (rows = hb*128 + r, cols = b*128 + k) -> dn_q[NE][H][I] + dn_sc[NE][H][I/32].
// Grid: (NE * eb * HB), Block: 256. Dynamic SMEM: 67584 B.
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_decode_fc2_mx8(
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ v2,      // [nblk][3][128] suh_down(2)
    const __half* __restrict__ v3,      // [nblk][H] svh_down
    const long long* __restrict__ tab,
    uint8_t* __restrict__ dn_q,         // [NE][H][I]
    uint8_t* __restrict__ dn_sc,        // [NE][H][I/32]
    int eb, int H) {
    extern __shared__ float sW[];       // [128][132]
    const int HB = H / 128;
    const int I = eb * 128;
    const int hb = blockIdx.x % HB;
    const int b = (blockIdx.x / HB) % eb;
    const int e = blockIdx.x / (HB * eb);
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int bits = static_cast<int>(tr[4]);
    const uint8_t* mw = tr3::fc2_block(fc2, tr, b, bits, H);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    auto decode_tile_to_smem = [&](auto tag_bits) {
        constexpr int BITS = decltype(tag_bits)::value;
        tr3::LaneDec<BITS, CB> tl;
        tl.init(lane);
        const size_t tb = static_cast<size_t>(32 * BITS);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
            const uint32_t raw = tl.loadw(tile, lane);
            uint32_t f0[2], f1[2];
            tl.decodew(raw, f0, f1);
            const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&f0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&f1[0]));
            const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&f0[1]));
            const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&f1[1]));
            const int rA = warp * 16 + row0;
            const int rB = rA + 8;
            const int c0 = ks * 16 + 2 * col;
            sW[rA * 132 + c0]     = a0.x; sW[rA * 132 + c0 + 1] = a0.y;
            sW[rB * 132 + c0]     = a1.x; sW[rB * 132 + c0 + 1] = a1.y;
            sW[rA * 132 + c0 + 8] = a2.x; sW[rA * 132 + c0 + 9] = a2.y;
            sW[rB * 132 + c0 + 8] = a3.x; sW[rB * 132 + c0 + 9] = a3.y;
        }
    };
    if (bits == 3) decode_tile_to_smem(std::integral_constant<int, 3>{});
    else if (bits == 4) decode_tile_to_smem(std::integral_constant<int, 4>{});
    else decode_tile_to_smem(std::integral_constant<int, 2>{});
    __syncthreads();

    // 1. Row-wise H128 * suh_down
    const __half* suh = v2 + (brow * 3 + 2) * 128 + lane * 4;
    float s_in[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s_in[i] = __half2float(suh[i]);
    for (int r = warp; r < 128; r += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = sW[r * 132 + lane * 4 + i];
        tr3::had128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) sW[r * 132 + lane * 4 + i] = v[i] * s_in[i];
    }
    __syncthreads();

    // 2. Col-wise H128 * svh_down
    const __half* svh = v3 + brow * H + hb * 128 + lane * 4;
    float s_out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s_out[i] = __half2float(svh[i]);
    for (int k = warp; k < 128; k += 8) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = sW[(lane * 4 + i) * 132 + k];
        tr3::had128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) sW[(lane * 4 + i) * 132 + k] = v[i] * s_out[i];
    }
    __syncthreads();

    // 3. Quantize to MXFP8 dn_q[e][hb*128 + r][b*128 + k]
    const int S = I >> 5;
    for (int r = warp; r < 128; r += 8) {
        const size_t grow = static_cast<size_t>(e) * H + hb * 128 + r;
        uint8_t* qdst = dn_q + grow * I + b * 128;
        uint8_t* sdst = dn_sc + grow * S + b * 4;
        #pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            const float val = sW[r * 132 + blk * 32 + lane];
            const float amax = mx8_warp_max(fabsf(val));
            float inv;
            const uint32_t code = mx8_scale_ue8m0(amax, inv);
            if (lane == 0) sdst[blk] = static_cast<uint8_t>(code);
            qdst[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val * inv, __NV_SATFINITE, __NV_E4M3));
        }
    }
}

// Runtime Post-Widened Trellis -> MXFP8 MoE Decoders (for layer-wise streaming scratch GR8_INK2_PF_MX8_MOE=3|4).
// Uses 32 KiB XOR-swizzled __half[128][128] SMEM (3 CTAs/SM on sm_120a, 0 SMEM bank conflicts across all 4 phases),
// dual-column Hadamard, and 128B coalesced uint32_t FP8 + uint32_t scale stores.
__device__ __forceinline__ int tr3_sw_u32(int r) {
    const int b = (r >> 2) & 31;
    const int rot = (b >> 1) | ((b & 1) << 4);
    return rot ^ ((r & 3) << 2);
}

__device__ __forceinline__ void tr3_quant_row_128(
    const uint32_t* __restrict__ sW_row, int sw_r,
    uint8_t* __restrict__ qdst, uint8_t* __restrict__ sdst, int lane) {
    const uint32_t u01 = sW_row[lane ^ sw_r];
    const uint32_t u23 = sW_row[(lane + 32) ^ sw_r];
    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&u01));
    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&u23));
    // Lanes 0..15 hold blk 0 (in f01) & blk 2 (in f23); lanes 16..31 hold blk 1 (in f01) & blk 3 (in f23).
    float amax01 = fmaxf(fabsf(f01.x), fabsf(f01.y));
    float amax23 = fmaxf(fabsf(f23.x), fabsf(f23.y));
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        amax01 = fmaxf(amax01, __shfl_xor_sync(0xffffffffu, amax01, o));
        amax23 = fmaxf(amax23, __shfl_xor_sync(0xffffffffu, amax23, o));
    }
    float inv01, inv23;
    const uint32_t code01 = mx8_scale_ue8m0(amax01, inv01);
    const uint32_t code23 = mx8_scale_ue8m0(amax23, inv23);
    const uint32_t q01 = mx8_f2_e4m3x2(f01.x * inv01, f01.y * inv01);
    const uint32_t q23 = mx8_f2_e4m3x2(f23.x * inv23, f23.y * inv23);
    const int src_lo = (lane & 15) * 2;
    const uint32_t a_lo = __shfl_sync(0xffffffffu, q01, src_lo);
    const uint32_t a_hi = __shfl_sync(0xffffffffu, q01, src_lo + 1);
    const uint32_t b_lo = __shfl_sync(0xffffffffu, q23, src_lo);
    const uint32_t b_hi = __shfl_sync(0xffffffffu, q23, src_lo + 1);
    reinterpret_cast<uint32_t*>(qdst)[lane] = (lane < 16) ? (a_lo | (a_hi << 16)) : (b_lo | (b_hi << 16));
    const uint32_t c1 = __shfl_sync(0xffffffffu, code01, 16);
    const uint32_t c3 = __shfl_sync(0xffffffffu, code23, 16);
    if (lane == 0) {
        *reinterpret_cast<uint32_t*>(sdst) = code01 | (c1 << 8) | (code23 << 16) | (c3 << 24);
    }
}

__global__ void __launch_bounds__(256, 3) k_ink2_tr3_wide_decode_fc1_mx8(
    const uint8_t* __restrict__ fc1,
    const __half* __restrict__ v1,      // [nblk][2][H] suh
    const __half* __restrict__ v2,      // [nblk][3][128] svh gate(0), up(1)
    const long long* __restrict__ tab,  // [NE][6]
    const long long* __restrict__ kmix,
    uint8_t* __restrict__ gu_q,         // [NE][2*I][H]
    uint8_t* __restrict__ gu_sc,        // [NE][2*I][H/32]
    int eb, int H) {
    extern __shared__ uint32_t sW_h2[]; // [128][64] of __half2 (32 KiB, XOR-swizzled)
    const int HB = H / 128;
    const int I = eb * 128;
    const int hb = blockIdx.x % HB;
    const int bm = (blockIdx.x / HB) % (2 * eb);
    const int e = blockIdx.x / (HB * 2 * eb);
    const int b = bm >> 1;
    const int mat = bm & 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    const int S = H >> 5;
    if (b >= static_cast<int>(tr[3])) {
        for (int r = warp; r < 128; r += 8) {
            const size_t grow = (static_cast<size_t>(e) * (2 * I) + mat * I + b * 128 + r);
            reinterpret_cast<uint32_t*>(gu_q + grow * H + hb * 128)[lane] = 0u;
            if (lane == 0) *reinterpret_cast<uint32_t*>(gu_sc + grow * S + hb * 4) = 0x7f7f7f7fu;
        }
        return;
    }
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const uint8_t* mw = nullptr;
    const int bits = tr3::fc1_unit(fc1, kmix, e, tr, bm, H, mw);
    const int row0 = lane >> 2, col = lane & 3;
    const int prev_lane = (lane + 31) & 31;
    const int rA = warp * 16 + row0;
    const int rB = rA + 8;
    const int swA = tr3_sw_u32(rA);
    const int swB = tr3_sw_u32(rB);

    auto store_pair = [&](int kp, const uint32_t (&f0)[2], const uint32_t (&f1)[2]) {
        const int c2_lo = kp * 8 + col;
        const int c2_hi = c2_lo + 4;
        sW_h2[rA * 64 + (c2_lo ^ swA)] = f0[0];
        sW_h2[rB * 64 + (c2_lo ^ swB)] = f1[0];
        sW_h2[rA * 64 + (c2_hi ^ swA)] = f0[1];
        sW_h2[rB * 64 + (c2_hi ^ swB)] = f1[1];
    };

    if (bits == 4) {
        const uint8_t* wg = mw + static_cast<size_t>(hb) * 8192 + warp * 1024 + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 wh = tr3::ld_nc128_tr3(wg + h * 512);
            const uint32_t w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t ch = w[p];
                const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                uint32_t f0[2], f1[2];
                tr3::decode_pmcgx_m<4>((static_cast<uint64_t>(pc) << 32) | ch, 28, f0, f1);
                store_pair(h * 4 + p, f0, f1);
            }
        }
    } else if (bits == 3) {
        const uint8_t* wg = mw + static_cast<size_t>(hb) * 6144 + warp * 256 + lane * 8;
        uint64_t w[3] = {tr3::ld_nc64_tr3(wg), tr3::ld_nc64_tr3(wg + 2048), tr3::ld_nc64_tr3(wg + 4096)};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t ch = tr3::trw_chunk_rt<3>(w, p);
            const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            uint32_t f0[2], f1[2];
            tr3::decode_pmcgx_m<3>((static_cast<uint64_t>(pc) << 24) | ch, 21, f0, f1);
            store_pair(p, f0, f1);
        }
    } else {
        const uint8_t* wg = mw + static_cast<size_t>(hb) * 4096 + warp * 256 + lane * 8;
        uint64_t w[3] = {tr3::ld_nc64_tr3(wg), tr3::ld_nc64_tr3(wg + 2048), 0ull};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t ch = tr3::trw_chunk_rt<2>(w, p);
            const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            uint32_t f0[2], f1[2];
            tr3::decode_pmcgx_m<2>((static_cast<uint64_t>(pc) << 16) | ch, 14, f0, f1);
            store_pair(p, f0, f1);
        }
    }
    __syncthreads();

    // 1. Row-wise H128 * suh (strided-64 mapping: lane holds {2*lane, 2*lane+1, 64+2*lane, 64+2*lane+1}, 0 bank conflicts)
    const __half2* suh2 = reinterpret_cast<const __half2*>(v1 + (brow * 2 + mat) * H + hb * 128);
    const float2 si0 = __half22float2(suh2[lane]);
    const float2 si1 = __half22float2(suh2[lane + 32]);
    for (int r = warp; r < 128; r += 8) {
        const int sw_r = tr3_sw_u32(r);
        const int i0 = r * 64 + (lane ^ sw_r);
        const int i1 = r * 64 + ((lane + 32) ^ sw_r);
        const uint32_t u0 = sW_h2[i0], u1 = sW_h2[i1];
        const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u0));
        const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&u1));
        float v[4] = {a0.x, a0.y, a1.x, a1.y};
        tr3::had128(v, lane);
        const __half2 h0 = __floats2half2_rn(v[0] * si0.x, v[1] * si0.y);
        const __half2 h1 = __floats2half2_rn(v[2] * si1.x, v[3] * si1.y);
        sW_h2[i0] = *reinterpret_cast<const uint32_t*>(&h0);
        sW_h2[i1] = *reinterpret_cast<const uint32_t*>(&h1);
    }
    __syncthreads();

    // 2. Dual-column H128 * svh (2 adjacent cols per __half2, 0 bank conflicts)
    const __half* svh = v2 + (brow * 3 + mat) * 128 + lane * 4;
    float s_out[4];
    int sw_row[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        s_out[i] = __half2float(svh[i]);
        sw_row[i] = tr3_sw_u32(lane * 4 + i);
    }
    for (int k2 = warp; k2 < 64; k2 += 8) {
        float v0[4], v1[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t u = sW_h2[(lane * 4 + i) * 64 + (k2 ^ sw_row[i])];
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
            v0[i] = f.x;
            v1[i] = f.y;
        }
        tr3::had128(v0, lane);
        tr3::had128(v1, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 h = __floats2half2_rn(v0[i] * s_out[i], v1[i] * s_out[i]);
            sW_h2[(lane * 4 + i) * 64 + (k2 ^ sw_row[i])] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
    __syncthreads();

    // 3. Quantize each row r=0..127 to MXFP8 with 128B coalesced uint32_t stores + 4B scale store
    for (int r = warp; r < 128; r += 8) {
        const size_t grow = (static_cast<size_t>(e) * (2 * I) + mat * I + b * 128 + r);
        tr3_quant_row_128(sW_h2 + r * 64, tr3_sw_u32(r), gu_q + grow * H + hb * 128, gu_sc + grow * S + hb * 4, lane);
    }
}

__global__ void __launch_bounds__(256, 3) k_ink2_tr3_wide_decode_fc2_mx8(
    const uint8_t* __restrict__ fc2,
    const __half* __restrict__ v2,      // [nblk][3][128] suh_down(2)
    const __half* __restrict__ v3,      // [nblk][H] svh_down
    const long long* __restrict__ tab,
    uint8_t* __restrict__ dn_q,         // [NE][H][I]
    uint8_t* __restrict__ dn_sc,        // [NE][H][I/32]
    int eb, int H) {
    extern __shared__ uint32_t sW_h2[]; // [128][64] of __half2 (32 KiB, XOR-swizzled)
    const int HB = H / 128;
    const int I = eb * 128;
    const int hb = blockIdx.x % HB;
    const int b = (blockIdx.x / HB) % eb;
    const int e = blockIdx.x / (HB * eb);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long* tr = tab + static_cast<size_t>(e) * 6;
    const int S = I >> 5;
    if (b >= static_cast<int>(tr[3])) {
        for (int r = warp; r < 128; r += 8) {
            const size_t grow = static_cast<size_t>(e) * H + hb * 128 + r;
            reinterpret_cast<uint32_t*>(dn_q + grow * I + b * 128)[lane] = 0u;
            if (lane == 0) *reinterpret_cast<uint32_t*>(dn_sc + grow * S + b * 4) = 0x7f7f7f7fu;
        }
        return;
    }
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int bits = static_cast<int>(tr[4]);
    const uint8_t* mw = tr3::fc2_block(fc2, tr, b, bits, H);
    const int row0 = lane >> 2, col = lane & 3;
    const int prev_lane = (lane + 31) & 31;
    const int n_tile = hb * 8 + warp;
    const int rA = warp * 16 + row0;
    const int rB = rA + 8;
    const int swA = tr3_sw_u32(rA);
    const int swB = tr3_sw_u32(rB);

    auto store_pair = [&](int ks, const uint32_t (&f0)[2], const uint32_t (&f1)[2]) {
        const int c2_lo = ks * 8 + col;
        const int c2_hi = c2_lo + 4;
        sW_h2[rA * 64 + (c2_lo ^ swA)] = f0[0];
        sW_h2[rB * 64 + (c2_lo ^ swB)] = f1[0];
        sW_h2[rA * 64 + (c2_hi ^ swA)] = f0[1];
        sW_h2[rB * 64 + (c2_hi ^ swB)] = f1[1];
    };

    if (bits == 4) {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * 1024 + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 wh = tr3::ld_nc128_tr3(wg + h * 512);
            const uint32_t w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t ch = w[p];
                const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                uint32_t f0[2], f1[2];
                tr3::decode_pmcgx_m<4>((static_cast<uint64_t>(pc) << 32) | ch, 28, f0, f1);
                store_pair(h * 4 + p, f0, f1);
            }
        }
    } else if (bits == 3) {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * 768 + lane * 8;
        uint64_t w[3] = {tr3::ld_nc64_tr3(wg), tr3::ld_nc64_tr3(wg + 256), tr3::ld_nc64_tr3(wg + 512)};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t ch = tr3::trw_chunk_rt<3>(w, p);
            const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            uint32_t f0[2], f1[2];
            tr3::decode_pmcgx_m<3>((static_cast<uint64_t>(pc) << 24) | ch, 21, f0, f1);
            store_pair(p, f0, f1);
        }
    } else {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * 512 + lane * 8;
        uint64_t w[3] = {tr3::ld_nc64_tr3(wg), tr3::ld_nc64_tr3(wg + 256), 0ull};
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t ch = tr3::trw_chunk_rt<2>(w, p);
            const uint32_t pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            uint32_t f0[2], f1[2];
            tr3::decode_pmcgx_m<2>((static_cast<uint64_t>(pc) << 16) | ch, 14, f0, f1);
            store_pair(p, f0, f1);
        }
    }
    __syncthreads();

    // 1. Row-wise H128 * suh_down (strided-64 mapping, 0 bank conflicts)
    const __half2* suh2 = reinterpret_cast<const __half2*>(v2 + (brow * 3 + 2) * 128);
    const float2 si0 = __half22float2(suh2[lane]);
    const float2 si1 = __half22float2(suh2[lane + 32]);
    for (int r = warp; r < 128; r += 8) {
        const int sw_r = tr3_sw_u32(r);
        const int i0 = r * 64 + (lane ^ sw_r);
        const int i1 = r * 64 + ((lane + 32) ^ sw_r);
        const uint32_t u0 = sW_h2[i0], u1 = sW_h2[i1];
        const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u0));
        const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&u1));
        float v[4] = {a0.x, a0.y, a1.x, a1.y};
        tr3::had128(v, lane);
        const __half2 h0 = __floats2half2_rn(v[0] * si0.x, v[1] * si0.y);
        const __half2 h1 = __floats2half2_rn(v[2] * si1.x, v[3] * si1.y);
        sW_h2[i0] = *reinterpret_cast<const uint32_t*>(&h0);
        sW_h2[i1] = *reinterpret_cast<const uint32_t*>(&h1);
    }
    __syncthreads();

    // 2. Dual-column H128 * svh_down (2 adjacent cols per __half2, 0 bank conflicts)
    const __half* svh = v3 + brow * H + hb * 128 + lane * 4;
    float s_out[4];
    int sw_row[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        s_out[i] = __half2float(svh[i]);
        sw_row[i] = tr3_sw_u32(lane * 4 + i);
    }
    for (int k2 = warp; k2 < 64; k2 += 8) {
        float v0[4], v1[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t u = sW_h2[(lane * 4 + i) * 64 + (k2 ^ sw_row[i])];
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
            v0[i] = f.x;
            v1[i] = f.y;
        }
        tr3::had128(v0, lane);
        tr3::had128(v1, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 h = __floats2half2_rn(v0[i] * s_out[i], v1[i] * s_out[i]);
            sW_h2[(lane * 4 + i) * 64 + (k2 ^ sw_row[i])] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
    __syncthreads();

    // 3. Quantize to MXFP8 dn_q[e][hb*128 + r][b*128 + k] with 128B coalesced uint32_t stores + 4B scale store
    for (int r = warp; r < 128; r += 8) {
        const size_t grow = static_cast<size_t>(e) * H + hb * 128 + r;
        tr3_quant_row_128(sW_h2 + r * 64, tr3_sw_u32(r), dn_q + grow * I + b * 128, dn_sc + grow * S + b * 4, lane);
    }
}

inline void launch_tr3_decode_mx8(
    const uint8_t* tr3_fc1, const uint8_t* tr3_fc2,
    const __half* tr3_v1, const __half* tr3_v2, const __half* tr3_v3,
    const long long* tr3_tab, const long long* tr3_kmix, bool wide,
    const MoeMx8Layer& dst, int NE, int H, int I, cudaStream_t st) {
    const int HB = H / 128;
    const int eb = I / 128;
    if (wide) {
        constexpr int DEC_WIDE_SMEM = 128 * 64 * sizeof(uint32_t); // 32 KiB -> 3 CTAs/SM on sm_120a
        k_ink2_tr3_wide_decode_fc1_mx8<<<static_cast<unsigned>(NE * 2 * eb * HB), 256, DEC_WIDE_SMEM, st>>>(
            tr3_fc1, tr3_v1, tr3_v2, tr3_tab, tr3_kmix, dst.gu_q, dst.gu_sc, eb, H);
        k_ink2_tr3_wide_decode_fc2_mx8<<<static_cast<unsigned>(NE * eb * HB), 256, DEC_WIDE_SMEM, st>>>(
            tr3_fc2, tr3_v2, tr3_v3, tr3_tab, dst.dn_q, dst.dn_sc, eb, H);
    } else {
        constexpr int DEC_SMEM = 128 * 132 * sizeof(float);
        k_ink2_tr3_decode_fc1_mx8<tr3::CB_PMCGX><<<static_cast<unsigned>(NE * 2 * eb * HB), 256, DEC_SMEM, st>>>(
            tr3_fc1, tr3_v1, tr3_v2, tr3_tab, tr3_kmix, dst.gu_q, dst.gu_sc, eb, H);
        k_ink2_tr3_decode_fc2_mx8<tr3::CB_PMCGX><<<static_cast<unsigned>(NE * eb * HB), 256, DEC_SMEM, st>>>(
            tr3_fc2, tr3_v2, tr3_v3, tr3_tab, dst.dn_q, dst.dn_sc, eb, H);
    }
}

// ------------------------------------------------------------------------------------------- Grouped Blackwell MXFP8 MoE Prefill
// Step 1: Parallel TileMap4 + flat tok_list/rw_list builder from ink2::Plan (1 CTA, 128 threads, ~5 us!).
__global__ void __launch_bounds__(128) k_ink2_moe_mx8_build_map(
    Plan plan,
    int* __restrict__ tok_list,
    float* __restrict__ rw_list,
    MoeTileMap4* __restrict__ map4,
    int* __restrict__ d_counts) {
    __shared__ int s_nrb[128], s_ncta[128];
    __shared__ int s_rb0[128], s_cta0[128];
    const int tid = threadIdx.x;
    const int nslots = plan.counts[0];

    int cnt = 0;
    if (tid < nslots) {
        const int t0 = plan.slot_tile0[tid];
        const int nt = plan.slot_nt[tid];
        for (int tg = 0; tg < nt; ++tg) cnt += __popc(plan.tile_mask[t0 + tg] & 0xff);
    }
    const int nrb = (cnt + 15) >> 4;
    const int ncta = (nrb + 3) >> 2;
    s_nrb[tid] = nrb;
    s_ncta[tid] = ncta;
    __syncthreads();

    if (tid == 0) {
        int rb_cur = 0, cta_cur = 0;
        for (int s = 0; s < nslots; ++s) {
            s_rb0[s] = rb_cur;
            s_cta0[s] = cta_cur;
            rb_cur += s_nrb[s];
            cta_cur += s_ncta[s];
        }
        d_counts[0] = rb_cur;
        d_counts[1] = cta_cur;
    }
    __syncthreads();

    if (tid < nslots && cnt > 0) {
        const int e = plan.slot_expert[tid];
        const int t0 = plan.slot_tile0[tid];
        const int nt = plan.slot_nt[tid];
        const int rb0 = s_rb0[tid];
        const int cta0 = s_cta0[tid];
        const int base_row = rb0 * 16;
        int c = 0;
        for (int tg = 0; tg < nt; ++tg) {
            const int tile = t0 + tg;
            const int mask = plan.tile_mask[tile];
            for (int lane = 0; lane < 8; ++lane) {
                if ((mask >> lane) & 1) {
                    tok_list[base_row + c] = plan.tile_tok[tile * 8 + lane];
                    rw_list[base_row + c] = plan.tile_w[tile * 8 + lane];
                    ++c;
                }
            }
        }
        for (int pad = cnt; pad < nrb * 16; ++pad) {
            tok_list[base_row + pad] = -1;
            rw_list[base_row + pad] = 0.0f;
        }
        int ci = 0;
        for (int off = 0; off < nrb; off += 4) {
            const int nb = min(4, nrb - off);
            const int m_here = min(nb * 16, cnt - off * 16);
            map4[cta0 + ci++] = MoeTileMap4{e, rb0 + off, nb, m_here};
        }
    }
}

// Step 2: Gather xm[tok_list[row]] and quantize directly into MX8_REC x8[rb][H/32][528].
// Grid: (max_rb * ((H/32 + 15)/16)), Block: 256.
__global__ void __launch_bounds__(256) k_ink2_moe_mx8_gather_x8(
    const uint16_t* __restrict__ xm,
    const int* __restrict__ tok_list,
    const int* __restrict__ d_counts,
    uint8_t* __restrict__ x8,
    int H) {
    __shared__ __align__(16) uint8_t sr[16 * MX8_REC];
    const int S = H >> 5;
    const int ST = (S + 15) >> 4;
    const int rb = blockIdx.x / ST;
    if (rb >= d_counts[0]) return;
    const int s0 = (blockIdx.x % ST) << 4;
    const int sl = threadIdx.x & 15;
    const int rr = threadIdx.x >> 4;
    const int s = s0 + sl;
    const int tok = tok_list[(rb << 4) + rr];

    if (s < S) {
        uint8_t* rec = sr + sl * MX8_REC;
        uint32_t c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        uint32_t code = 127u;
        if (tok >= 0) {
            const uint4* p = reinterpret_cast<const uint4*>(xm + static_cast<size_t>(tok) * H + (s << 5));
            const uint4 q[4] = {p[0], p[1], p[2], p[3]};
            float2 f[16];
            float amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
                f[4 * j]     = __bfloat1622float2(bp[0]);
                f[4 * j + 1] = __bfloat1622float2(bp[1]);
                f[4 * j + 2] = __bfloat1622float2(bp[2]);
                f[4 * j + 3] = __bfloat1622float2(bp[3]);
            }
            #pragma unroll
            for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
            float inv;
            code = mx8_scale_ue8m0(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = mx8_f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                       (mx8_f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                *reinterpret_cast<uint32_t*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
            }
        }
        rec[512 + rr] = static_cast<uint8_t>(code);
    }
    __syncthreads();
    const int nv = min(16, S - s0) * (MX8_REC / 16);
    uint4* d = reinterpret_cast<uint4*>(x8 + (static_cast<size_t>(rb) * S + s0) * MX8_REC);
    const uint4* v = reinterpret_cast<const uint4*>(sr);
    for (int i = threadIdx.x; i < nv; i += 256) d[i] = v[i];
}

__device__ __forceinline__ float ink2_swiglu_f32(float g, float u) {
    // Match ink2_moe.cuh exact scalar: sigmoid sees the RAW gate g (NOT g_cap)!
    const float gc = 4.0f * tanhf(g * 0.25f);
    const float uc = 25.0f * tanhf(u * 0.04f);
    return (gc / (1.0f + __expf(-g))) * uc;
}

// Step 3: Grouped Blackwell MXFP8 Fused FC1 (Gate + Up + Inkling Softcapped SwiGLU -> directly quantized MX8_REC a8!).
// Zero act_bf16 DRAM writes/reads and zero separate quant_a8 kernel launch!
__global__ void __launch_bounds__(256, 2) k_ink2_moe_mx8_fc1(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ gu_q,    // [NE][2*I][H]
    const uint8_t* __restrict__ gu_sc,   // [NE][2*I][H/32]
    const MoeTileMap4* __restrict__ map4,
    const int* __restrict__ d_counts,
    uint8_t* __restrict__ a8,            // [max_rb][I/32][528]
    int H, int I) {
    using Cfg = Mx8LargeCfg;
    constexpr int BM = Cfg::BM, KS = Cfg::KS;
    extern __shared__ __align__(16) uint8_t sm[];

    const int cta_y = blockIdx.y;
    if (cta_y >= d_counts[1]) return;
    const MoeTileMap4 tm = map4[cta_y];
    const int e = tm.expert, rb0 = tm.rb0, nb = tm.nb, m_here = tm.m_here;
    const int c0 = blockIdx.x * 64; // 64 cols of gate AND 64 cols of up = 128 B-rows!
    if (c0 >= I) return;
    const int ncols = min(64, I - c0);

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = H >> 5, NST = (S + KS - 1) / KS;
    const uint8_t* Bq_g = gu_q + (static_cast<size_t>(e) * (2 * I) + c0) * H;
    const uint8_t* Bq_u = gu_q + (static_cast<size_t>(e) * (2 * I) + I + c0) * H;
    const uint8_t* Bsc_g = gu_sc + (static_cast<size_t>(e) * (2 * I) + c0) * S;
    const uint8_t* Bsc_u = gu_sc + (static_cast<size_t>(e) * (2 * I) + I + c0) * S;

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * Cfg::STAGE;
        for (int c = t; c < KS * BM * 33; c += 256) {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A; // rows 0..63 = gate, rows 64..127 = up
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3, ch = c & 7;
            const int is_up = (n >= 64);
            const int nr = is_up ? (n - 64) : n;
            if (nr < ncols && st * KS + (ch >> 1) < S) {
                const uint8_t* src = (is_up ? Bq_u : Bq_g) + static_cast<size_t>(nr) * H + st * 128 + (ch << 4);
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4), src);
            }
        }
    };

    const bool is_up_w = (warp >= 4);
    const int colw = (warp & 3) * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    const uint8_t* sp_base = is_up_w ? Bsc_u : Bsc_g;
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = colw + (lane >> 2);
            v[j][0] = (r0 < ncols && ks < S) ? static_cast<uint32_t>(__ldg(sp_base + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < ncols && ks < S) ? static_cast<uint32_t>(__ldg(sp_base + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[BM][2][4];
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;
    const int b_row_base = warp * 16;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * Cfg::STAGE;
        const uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = b_row_base + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    __syncthreads();

    // Epilogue: reuse sm as float s_gate[64][64] (16 KB) + float s_up[64][64] (16 KB) = 32 KB < 48.5 KB!
    float* s_gate = reinterpret_cast<float*>(sm);
    float* s_up = s_gate + 64 * 64;
    float* s_dst = is_up_w ? s_up : s_gate;
    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int lc = colw + nt * 8 + 2 * c;
            if (lc >= ncols) continue;
            const int lr0 = mb * 16 + r;
            s_dst[lr0 * 64 + lc]     = acc[mb][nt][0];
            s_dst[lr0 * 64 + lc + 1] = acc[mb][nt][1];
            s_dst[(lr0 + 8) * 64 + lc]     = acc[mb][nt][2];
            s_dst[(lr0 + 8) * 64 + lc + 1] = acc[mb][nt][3];
        }
    }
    __syncthreads();

    // Directly quantize SwiGLU(s_gate, s_up) -> MX8_REC a8!
    // 8 warps = 4 row-blocks (mb = 0..3) x 2 K-blocks of 32 cols (sb = 0..1).
    // Uses s_rec[8][528] placed right after s_up (32768 + 4224 = 36992 B < 49664 B).
    uint8_t* s_rec_base = sm + 32768;
    const int mb_w = warp >> 1;
    const int sb_w = warp & 1;
    if (mb_w < nb && sb_w * 32 < ncols) {
        uint8_t* s_rec = s_rec_base + warp * MX8_REC;
        const int col_local = sb_w * 32 + lane;
        const int h_half = lane >> 4;
        const int cc = (lane >> 2) & 3;
        const int b_byte = lane & 3;
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            const int row = mb_w * 16 + rr;
            const float y = (row < m_here && col_local < ncols)
                ? ink2_swiglu_f32(s_gate[row * 64 + col_local], s_up[row * 64 + col_local])
                : 0.0f;
            const float amax = mx8_warp_max(fabsf(y));
            float inv;
            const uint32_t code = mx8_scale_ue8m0(amax, inv);
            const uint8_t qv = static_cast<uint8_t>(__nv_cvt_float_to_fp8(y * inv, __NV_SATFINITE, __NV_E4M3));
            const int off = ((rr & 7) * 4 + cc) * 16 + (h_half * 2 + (rr >> 3)) * 4 + b_byte;
            s_rec[off] = qv;
            if (lane == 0) s_rec[512 + rr] = static_cast<uint8_t>(code);
        }
        __syncwarp();
        const int S_i = I >> 5;
        const int s_idx = (c0 >> 5) + sb_w;
        uint4* dst = reinterpret_cast<uint4*>(a8 + (static_cast<size_t>(rb0 + mb_w) * S_i + s_idx) * MX8_REC);
        const uint4* src = reinterpret_cast<const uint4*>(s_rec);
        dst[lane] = src[lane];
        if (lane == 0) dst[32] = src[32];
    }
}

// Step 4: Quantize act_bf16[total_rb * 16, I] -> a8[total_rb][I/32][528]
__global__ void __launch_bounds__(256) k_ink2_moe_mx8_quant_a8(
    const uint16_t* __restrict__ act_bf16,
    const int* __restrict__ d_counts,
    uint8_t* __restrict__ a8,
    int I) {
    __shared__ __align__(16) uint8_t sr[16 * MX8_REC];
    const int S = I >> 5;
    const int ST = (S + 15) >> 4;
    const int rb = blockIdx.x / ST;
    if (rb >= d_counts[0]) return;
    const int s0 = (blockIdx.x % ST) << 4;
    const int sl = threadIdx.x & 15;
    const int rr = threadIdx.x >> 4;
    const int s = s0 + sl;
    const int row = (rb << 4) + rr;

    if (s < S) {
        uint8_t* rec = sr + sl * MX8_REC;
        uint32_t c[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        const uint4* p = reinterpret_cast<const uint4*>(act_bf16 + static_cast<size_t>(row) * I + (s << 5));
        const uint4 q[4] = {p[0], p[1], p[2], p[3]};
        float2 f[16];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
            f[4 * j]     = __bfloat1622float2(bp[0]);
            f[4 * j + 1] = __bfloat1622float2(bp[1]);
            f[4 * j + 2] = __bfloat1622float2(bp[2]);
            f[4 * j + 3] = __bfloat1622float2(bp[3]);
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
        float inv;
        const uint32_t code = mx8_scale_ue8m0(amax, inv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            c[j] = mx8_f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                   (mx8_f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                *reinterpret_cast<uint32_t*>(rec + ((rr & 7) * 4 + cc) * 16 + (h * 2 + (rr >> 3)) * 4) = c[h * 4 + cc];
            }
        }
        rec[512 + rr] = static_cast<uint8_t>(code);
    }
    __syncthreads();
    const int nv = min(16, S - s0) * (MX8_REC / 16);
    uint4* d = reinterpret_cast<uint4*>(a8 + (static_cast<size_t>(rb) * S + s0) * MX8_REC);
    const uint4* v = reinterpret_cast<const uint4*>(sr);
    for (int i = threadIdx.x; i < nv; i += 256) d[i] = v[i];
}

// Step 5: Grouped Blackwell MXFP8 FC2 (down * rw + atomicAdd directly into out_f32[tok][H] -> ZERO moe_combine!).
// Grid: dim3(ceil(H / 128), max_cta_y), Block: 256.
__global__ void __launch_bounds__(256, 2) k_ink2_moe_mx8_fc2(
    const uint8_t* __restrict__ a8,
    const uint8_t* __restrict__ dn_q,    // [NE][H][I]
    const uint8_t* __restrict__ dn_sc,   // [NE][H][I/32]
    const MoeTileMap4* __restrict__ map4,
    const int* __restrict__ tok_list,
    const float* __restrict__ rw_list,
    const int* __restrict__ d_counts,
    float* __restrict__ out_f32,         // [M][H] zeroed before launch
    int H, int I) {
    using Cfg = Mx8LargeCfg;
    constexpr int BM = Cfg::BM, KS = Cfg::KS;
    extern __shared__ __align__(16) uint8_t sm[];

    const int cta_y = blockIdx.y;
    if (cta_y >= d_counts[1]) return;
    const MoeTileMap4 tm = map4[cta_y];
    const int e = tm.expert, rb0 = tm.rb0, nb = tm.nb, m_here = tm.m_here;
    const int c0 = blockIdx.x * 128;
    if (c0 >= H) return;
    const int ncols = min(128, H - c0);

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int S = I >> 5, NST = (S + KS - 1) / KS;
    const uint8_t* Bq = dn_q + (static_cast<size_t>(e) * H + c0) * I;
    const uint8_t* Bsc = dn_sc + (static_cast<size_t>(e) * H + c0) * S;

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * Cfg::STAGE;
        for (int c = t; c < KS * BM * 33; c += 256) {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                mx8_cp16(a + c * 16, a8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                mx8_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4), Bq + static_cast<size_t>(n) * I + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw + 7) / 8));
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            const int r0 = colw + (lane >> 2);
            v[j][0] = (r0 < ncols && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0) * S + ks)) : 127u;
            v[j][1] = (r0 + 8 < ncols && ks < S) ? static_cast<uint32_t>(__ldg(Bsc + static_cast<size_t>(r0 + 8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    mx8_cp_commit();
    uint32_t ns[KS][2];
    loadS(0, ns);

    float acc[BM][2][4];
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.0f;

    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int mat = lane >> 3, lr = lane & 7;

    for (int st = 0; st < NST; ++st) {
        mx8_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        mx8_cp_commit();

        uint32_t cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j) { cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1]; }
        if (st + 1 < NST) loadS(st + 1, ns);

        const uint8_t* a = sm + (st & 1) * Cfg::STAGE;
        const uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            if (st * KS + j >= S || nvalid == 0) continue;
            uint32_t bf00, bf01, bf10, bf11;
            const int n = colw + (mat >> 1) * 8 + lr;
            const int ch = j * 2 + (mat & 1);
            mx8_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mx8_mma(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mx8_mma(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    mx8_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        const int rA = mb * 16 + r;
        const int rB = rA + 8;
        const int tokA = (rA < m_here) ? tok_list[rb0 * 16 + rA] : -1;
        const int tokB = (rB < m_here) ? tok_list[rb0 * 16 + rB] : -1;
        const float rwA = (tokA >= 0) ? rw_list[rb0 * 16 + rA] : 0.0f;
        const float rwB = (tokB >= 0) ? rw_list[rb0 * 16 + rB] : 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = c0 + colw + nt * 8 + 2 * c;
            if (col >= H) continue;
            if (tokA >= 0) {
                float* dst = out_f32 + static_cast<size_t>(tokA) * H + col;
                atomicAdd(dst, acc[mb][nt][0] * rwA);
                atomicAdd(dst + 1, acc[mb][nt][1] * rwA);
            }
            if (tokB >= 0) {
                float* dst = out_f32 + static_cast<size_t>(tokB) * H + col;
                atomicAdd(dst, acc[mb][nt][2] * rwB);
                atomicAdd(dst + 1, acc[mb][nt][3] * rwB);
            }
        }
    }
}

__global__ void __launch_bounds__(256) k_ink2_f32_to_bf16_vec4(const float* __restrict__ src, uint16_t* __restrict__ dst, size_t n4) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n4) return;
    const float4 v = reinterpret_cast<const float4*>(src)[i];
    uint2 ov;
    ov.x = mx8_pack_bf16x2(v.x, v.y);
    ov.y = mx8_pack_bf16x2(v.z, v.w);
    reinterpret_cast<uint2*>(dst)[i] = ov;
}

inline void init_mx8_attrs() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_ink2_gemm_mx8_large<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_ink2_gemm_mx8_large<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_ink2_gemm_mx8_large_multi,  cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_ink2_moe_mx8_fc1,           cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_ink2_moe_mx8_fc2,           cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    constexpr int DEC_SMEM = 128 * 132 * sizeof(float);
    cudaFuncSetAttribute(k_ink2_tr3_decode_fc1_mx8<tr3::CB_PMCGX>, cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_SMEM);
    cudaFuncSetAttribute(k_ink2_tr3_decode_fc2_mx8<tr3::CB_PMCGX>, cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_SMEM);
    constexpr int DEC_WIDE_SMEM = 128 * 64 * sizeof(uint32_t);
    cudaFuncSetAttribute(k_ink2_tr3_wide_decode_fc1_mx8,           cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_WIDE_SMEM);
    cudaFuncSetAttribute(k_ink2_tr3_wide_decode_fc2_mx8,           cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_WIDE_SMEM);
}

inline void launch_moe_mx8_prefill(
    const MoeMx8Layer& ly, const MoeMx8Scratch& sc,
    const uint16_t* xm, const Plan& plan, int M, int TK,
    uint16_t* out_bf16, cudaStream_t st, Prof* prof = nullptr) {
    const int H = ly.H, I = ly.I, NE = ly.NE;
    // Upper bound on 16-row blocks and 64-row CTAs across NE experts:
    const int max_rb_here = std::min(sc.max_rb, (M * TK + 15) / 16 + NE);
    const int max_cta_here = std::min(sc.max_cta_y, (max_rb_here + 3) / 4 + NE);
    {
        ProfScope _ps(prof, PS_MOE_FC1, st);
        k_ink2_moe_mx8_build_map<<<1, 128, 0, st>>>(plan, sc.tok_list, sc.rw_list, sc.map4, sc.d_counts);
        const int st_h = ((H >> 5) + 15) >> 4;
        k_ink2_moe_mx8_gather_x8<<<max_rb_here * st_h, 256, 0, st>>>(xm, sc.tok_list, sc.d_counts, sc.x8, H);
        k_ink2_moe_mx8_fc1<<<dim3((I + 63) / 64, max_cta_here), 256, Mx8LargeCfg::SMEM, st>>>(
            sc.x8, ly.gu_q, ly.gu_sc, sc.map4, sc.d_counts, sc.a8, H, I);
    }
    {
        ProfScope _ps(prof, PS_MOE_FC2, st);
        cudaMemsetAsync(sc.out_f32, 0, static_cast<size_t>(M) * H * sizeof(float), st);
        k_ink2_moe_mx8_fc2<<<dim3((H + 127) / 128, max_cta_here), 256, Mx8LargeCfg::SMEM, st>>>(
            sc.a8, ly.dn_q, ly.dn_sc, sc.map4, sc.tok_list, sc.rw_list, sc.d_counts, sc.out_f32, H, I);
    }
    {
        ProfScope _ps(prof, PS_MOE_COMBINE, st);
        const size_t n4 = static_cast<size_t>(M) * (H >> 2);
        k_ink2_f32_to_bf16_vec4<<<static_cast<unsigned>((n4 + 255) / 256), 256, 0, st>>>(sc.out_f32, out_bf16, n4);
    }
}

}  // namespace mx8
}  // namespace ink2
#endif
