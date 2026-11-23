// Block-32 FP8 E4M3 on-load dense quantization + coalesced 128-bit decode GEMV kernels for Qwen3.8-Flash-Next.
//   - Format: q[N, K] (uint8_t FP8 E4M3, 1 B/weight) + sc[N, K/32] (__half, max_abs / 448.0f per 32-weight block).
//   - Every uint4 load (16 FP8 weights) lies inside ONE single 32-weight block -> 1 scale per uint4 load.
//   - Quantized on GPU in ~0.15 s at Engine::init() when GR8_QW38_DENSE_FP8=1; prefill (M > 8) stays on BF16 cuBLAS.
//   - Native sm_121a cvt.rn.f16x2.e4m3x2 decode (1 instruction per 2 weights), FP32 accumulation.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace qw38 {

struct Fp8Mat {
    const uint8_t* q = nullptr;     // [N, K] FP8 E4M3
    const __half* sc = nullptr;     // [N, K / 32] FP16 block-32 scale (when !mx_sc)
    const uint8_t* mx_sc = nullptr; // [N, K / 32] OCP MXFP8 UE8M0 block-32 scale (when GR8_QW38_PF_MXFP8=1)
    int N = 0;
    int K = 0;
};

__device__ __forceinline__ uint32_t qw38_mx_scale(float amax, float& inv) {
    const uint32_t u = __float_as_uint(__fmul_rn(amax, 0.002232142857f)); // amax / 448
    int e = static_cast<int>(u >> 23) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    if (amax == 0.0f) e = 0;
    if (e < -126) e = -126;
    if (e > 127) e = 127;
    inv = __uint_as_float(static_cast<uint32_t>(127 - e) << 23);
    return static_cast<uint32_t>(e + 127);
}

template <bool MX_SC>
__device__ __forceinline__ float load_fp8_scale(const __half* sc, const uint8_t* mx_sc, size_t idx) {
    if constexpr (MX_SC) {
        return __uint_as_float(static_cast<uint32_t>(__ldg(mx_sc + idx)) << 23);
    } else {
        return __half2float(__ldg(sc + idx));
    }
}

__device__ __forceinline__ float fp8_warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float fp8_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Quantize BF16 W[N, K] (K multiple of 32) -> block-32 FP8 E4M3 q[N, K] + FP16 sc or UE8M0 mx_sc.
// 8 warps/CTA, 1 warp = 1 block of 32 weights.
template <bool USE_MX>
__global__ void k_quantize_bf16_to_fp8_b32(const __nv_bfloat16* __restrict__ W,
                                           uint8_t* __restrict__ q,
                                           __half* __restrict__ sc,
                                           uint8_t* __restrict__ mx_sc,
                                           int N, int K) {
    const int nb_row = K >> 5;
    const size_t total_blks = static_cast<size_t>(N) * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blks) return;

    const size_t base = blk * 32 + lane;
    const float w = __bfloat162float(W[base]);
    const float mx = fp8_warp_max(fabsf(w));
    float inv_s = 0.0f;
    if constexpr (USE_MX) {
        const uint32_t code = qw38_mx_scale(mx, inv_s);
        if (lane == 0) mx_sc[blk] = static_cast<uint8_t>(code);
    } else {
        const float s_raw = (mx > 1e-12f) ? (mx / 448.0f) : 1.0f;
        const __half sh = __float2half_rn(s_raw);
        const float s = __half2float(sh);
        inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
        if (lane == 0) sc[blk] = sh;
    }
    const __nv_fp8_storage_t qv = __nv_cvt_float_to_fp8(w * inv_s, __NV_SATFINITE, __NV_E4M3);
    q[base] = static_cast<uint8_t>(qv);
}

inline void quantize_bf16_to_fp8_b32(const __nv_bfloat16* W, uint8_t* q, __half* sc, uint8_t* mx_sc,
                                     int N, int K, cudaStream_t st) {
    const size_t total_blks = static_cast<size_t>(N) * (K >> 5);
    const unsigned grid = static_cast<unsigned>((total_blks + 7) / 8);
    if (mx_sc) k_quantize_bf16_to_fp8_b32<true><<<grid, 256, 0, st>>>(W, q, nullptr, mx_sc, N, K);
    else k_quantize_bf16_to_fp8_b32<false><<<grid, 256, 0, st>>>(W, q, sc, nullptr, N, K);
}

// Unpack 4 FP8 E4M3 bytes (u32) -> 2x float2 UNSCALED using native cvt.rn.f16x2.e4m3x2.
__device__ __forceinline__ void fp8x4_unpack_raw(uint32_t v, float2& f0, float2& f1) {
    uint32_t r0, r1;
    const uint16_t lo = static_cast<uint16_t>(v & 0xffffu);
    const uint16_t hi = static_cast<uint16_t>(v >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r0) : "h"(lo));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r1) : "h"(hi));
    f0 = __half22float2(*reinterpret_cast<const __half2*>(&r0));
    f1 = __half22float2(*reinterpret_cast<const __half2*>(&r1));
}

__device__ __forceinline__ void fp8x16_unpack_raw(uint4 wv, float2 (&wf)[8]) {
    fp8x4_unpack_raw(wv.x, wf[0], wf[1]);
    fp8x4_unpack_raw(wv.y, wf[2], wf[3]);
    fp8x4_unpack_raw(wv.z, wf[4], wf[5]);
    fp8x4_unpack_raw(wv.w, wf[6], wf[7]);
}

// Unpack 4 FP8 E4M3 bytes (u32) -> 2x float2 multiplied by `scale`, using native cvt.rn.f16x2.e4m3x2.
__device__ __forceinline__ void fp8x4_unpack(uint32_t v, float scale, float2& f0, float2& f1) {
    uint32_t r0, r1;
    const uint16_t lo = static_cast<uint16_t>(v & 0xffffu);
    const uint16_t hi = static_cast<uint16_t>(v >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r0) : "h"(lo));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r1) : "h"(hi));
    const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&r0));
    const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&r1));
    f0.x = h0.x * scale; f0.y = h0.y * scale;
    f1.x = h1.x * scale; f1.y = h1.y * scale;
}

// Dot 16 BF16 activations (two uint4s) with 8 pre-unpacked float2 weights.
__device__ __forceinline__ float dot_bf16x16_f2x8(uint4 a0, uint4 a1, const float2 (&wf)[8]) {
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

__device__ __forceinline__ void fp8x16_unpack(uint4 wv, float scale, float2 (&wf)[8]) {
    fp8x4_unpack(wv.x, scale, wf[0], wf[1]);
    fp8x4_unpack(wv.y, scale, wf[2], wf[3]);
    fp8x4_unpack(wv.z, scale, wf[4], wf[5]);
    fp8x4_unpack(wv.w, scale, wf[6], wf[7]);
}

// Quantize HC up [4, 2560, 320] BF16 -> transposed [2560, 4, 320] block-32 FP8 E4M3 so all 4 streams of dim d are contiguous!
template <bool USE_MX>
__global__ void k_quantize_hc_up_fp8_b32(const __nv_bfloat16* __restrict__ W,
                                         uint8_t* __restrict__ q,
                                         __half* __restrict__ sc,
                                         uint8_t* __restrict__ mx_sc,
                                         int HID, int RANK) {
    const int nb_row = RANK >> 5;
    const size_t total_blks = static_cast<size_t>(4) * HID * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blks) return;

    const int b_in_row = static_cast<int>(blk % nb_row);
    const size_t row_dst = blk / nb_row; // [0 .. HID*4 - 1] = d * 4 + s
    const int s = static_cast<int>(row_dst & 3);
    const int d = static_cast<int>(row_dst >> 2);
    const size_t row_src = static_cast<size_t>(s) * HID + d;

    const float w = __bfloat162float(W[row_src * RANK + b_in_row * 32 + lane]);
    const float mx = fp8_warp_max(fabsf(w));
    float inv_s = 0.0f;
    if constexpr (USE_MX) {
        const uint32_t code = qw38_mx_scale(mx, inv_s);
        if (lane == 0) mx_sc[blk] = static_cast<uint8_t>(code);
    } else {
        const float s_raw = (mx > 1e-12f) ? (mx / 448.0f) : 1.0f;
        const __half sh = __float2half_rn(s_raw);
        const float sv = __half2float(sh);
        inv_s = (sv > 0.0f) ? (1.0f / sv) : 0.0f;
        if (lane == 0) sc[blk] = sh;
    }
    q[blk * 32 + lane] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(w * inv_s, __NV_SATFINITE, __NV_E4M3));
}

inline void quantize_hc_up_fp8_b32(const __nv_bfloat16* W, uint8_t* q, __half* sc, uint8_t* mx_sc,
                                   int HID, int RANK, cudaStream_t st) {
    const size_t total_blks = static_cast<size_t>(4) * HID * (RANK >> 5);
    const unsigned grid = static_cast<unsigned>((total_blks + 7) / 8);
    if (mx_sc) k_quantize_hc_up_fp8_b32<true><<<grid, 256, 0, st>>>(W, q, nullptr, mx_sc, HID, RANK);
    else k_quantize_hc_up_fp8_b32<false><<<grid, 256, 0, st>>>(W, q, sc, nullptr, HID, RANK);
}

// Coalesced 128-bit FP8 E4M3 GEMV for small decode batches (M <= 20):
//   C[M, N] = A[M, K] (BF16) @ B[N, K]^T (FP8 b32 with FP16 or MXFP8 UE8M0 scales).
template <int MAX_M, bool OUT_F32 = false, bool MX_SC = false>
__global__ void __launch_bounds__(128) k_fp8_gemv_1warp(const __nv_bfloat16* __restrict__ A,
                                 const uint8_t* __restrict__ Bq,
                                 const __half* __restrict__ Bsc,
                                 const uint8_t* __restrict__ Bmx,
                                 void* __restrict__ C,
                                 int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int k16 = K >> 4;
    const int nb_row = K >> 5;
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(n) * K);
    const size_t sbase = static_cast<size_t>(n) * nb_row;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k16; i += 32) {
        const uint4 wv = __ldg(qrow + i);
        const float sc = load_fp8_scale<MX_SC>(Bsc, Bmx, sbase + (i >> 1));
        float2 wf[8];
        fp8x16_unpack(wv, sc, wf);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* am = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K);
                acc[m] += dot_bf16x16_f2x8(am[2 * i], am[2 * i + 1], wf);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float tot = fp8_warp_sum(acc[m]);
            if (lane == 0) {
                if constexpr (OUT_F32) {
                    static_cast<float*>(C)[static_cast<size_t>(m) * N + n] = tot;
                } else {
                    static_cast<__nv_bfloat16*>(C)[static_cast<size_t>(m) * N + n] = __float2bfloat16(tot);
                }
            }
        }
    }
}

template <bool OUT_F32, bool MX_SC>
inline void launch_fp8_gemv_smallm_impl(const __nv_bfloat16* A, const Fp8Mat& B, void* C, int M, cudaStream_t st) {
    const int grid = (B.N + 3) / 4;
    if (M == 1) k_fp8_gemv_1warp<1, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else if (M <= 4) k_fp8_gemv_1warp<4, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else if (M == 5) k_fp8_gemv_1warp<5, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else if (M <= 8) k_fp8_gemv_1warp<8, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else if (M <= 12) k_fp8_gemv_1warp<12, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else if (M <= 16) k_fp8_gemv_1warp<16, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
    else k_fp8_gemv_1warp<20, OUT_F32, MX_SC><<<grid, 128, 0, st>>>(A, B.q, B.sc, B.mx_sc, C, M, B.N, B.K);
}

template <bool OUT_F32 = false>
inline void launch_fp8_gemv_smallm(const __nv_bfloat16* A, const Fp8Mat& B, void* C, int M, cudaStream_t st) {
    if (B.mx_sc) launch_fp8_gemv_smallm_impl<OUT_F32, true>(A, B, C, M, st);
    else launch_fp8_gemv_smallm_impl<OUT_F32, false>(A, B, C, M, st);
}

// Single-allocation aligned device arena for all on-load FP8/MXFP8 weights & scales.
struct Fp8Arena {
    uint8_t* base = nullptr;
    size_t cap = 0;
    size_t used = 0;
    struct Pending {
        const __nv_bfloat16* src;
        Fp8Mat* dst;
        int N, K;
        bool is_hc_up = false;
    };
    std::vector<Pending> pend;

    ~Fp8Arena() { if (base) cudaFree(base); }

    static size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

    void add(const __nv_bfloat16* src, Fp8Mat& dst, int N, int K, bool is_hc_up = false) {
        if (!src || N <= 0 || K <= 0 || (K & 31) != 0) return;
        pend.push_back({src, &dst, N, K, is_hc_up});
    }

    size_t build(cudaStream_t st, bool use_mxfp8 = false) {
        if (pend.empty()) return 0;
        const size_t sc_elem_b = use_mxfp8 ? sizeof(uint8_t) : sizeof(__half);
        size_t total = 0;
        for (const auto& p : pend) {
            total += align256(static_cast<size_t>(p.N) * p.K);
            total += align256(static_cast<size_t>(p.N) * (p.K >> 5) * sc_elem_b);
        }
        void* ptr = nullptr;
        cudaError_t e = cudaMalloc(&ptr, total);
        if (e != cudaSuccess) return 0;
        base = static_cast<uint8_t*>(ptr);
        cap = total;
        used = 0;
        for (const auto& p : pend) {
            uint8_t* qp = base + used;
            used += align256(static_cast<size_t>(p.N) * p.K);
            uint8_t* sp_raw = base + used;
            used += align256(static_cast<size_t>(p.N) * (p.K >> 5) * sc_elem_b);
            __half* sp = use_mxfp8 ? nullptr : reinterpret_cast<__half*>(sp_raw);
            uint8_t* mxp = use_mxfp8 ? sp_raw : nullptr;
            if (p.is_hc_up) quantize_hc_up_fp8_b32(p.src, qp, sp, mxp, p.N / 4, p.K, st);
            else quantize_bf16_to_fp8_b32(p.src, qp, sp, mxp, p.N, p.K, st);
            p.dst->q = qp;
            p.dst->sc = sp;
            p.dst->mx_sc = mxp;
            p.dst->N = p.N;
            p.dst->K = p.K;
        }
        pend.clear();
        return total;
    }
};

} // namespace qw38
