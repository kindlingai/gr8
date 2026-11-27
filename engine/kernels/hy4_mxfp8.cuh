// Blackwell (sm_120a / sm_121a) Native MXFP8 (E4M3 + UE8M0, 1x32 block scale) Dense Linear Engine for Hy4-flash.
//   - Boot quantization converts BF16 [N, K] dense matrices -> contiguous MXFP8 arena (~51.6% of BF16 VRAM)
//     so the BF16 dense staging image can be freed before KV pool allocation (~5.1 GiB VRAM saved on TP=1!).
//   - Decode / DFly2 verify (M <= 15): 16-lane/row coalesced 2x-pipelined MXFP8 GEMV (2x less weight BW than BF16),
//     plus fused MXFP8 q_absorb_rope and v_expand_gate kernels.
//   - Medium batch & prefill (M >= 16): Blackwell native mma.sync.aligned.kind::mxf8f6f4.m16n8k32.e4m3
//     (cp.async double-buffered 128x64 tiles for M > 64, 64x(MB*16) tiles for 16 <= M <= 64).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../model_geom_hy4.h"
#include "hy4_hc.cuh"

namespace hy4 {
namespace mx8 {

constexpr int MX8_REC = 528; // 16 rows x 32 k: 512 B E4M3 A-fragments (32 lanes x 16 B) + 16 B UE8M0 row scales

struct Mx8Mat {
    uint8_t* q  = nullptr; // [N, K] E4M3
    uint8_t* sc = nullptr; // [N, K / 32] UE8M0
    int N = 0;
    int K = 0;
    bool ready() const { return q != nullptr && sc != nullptr && N > 0 && K > 0; }
};

// Unified linear weight descriptor: either native MXFP8 or exact BF16 (rollback).
struct LinW {
    const __nv_bfloat16* bf16 = nullptr;
    Mx8Mat mx8{};
    bool is_mx8() const { return mx8.ready(); }
    static LinW from_bf16(const __nv_bfloat16* p, int /*N*/, int /*K*/) {
        LinW w{};
        w.bf16 = p;
        return w;
    }
    static LinW from_mx8(Mx8Mat m) {
        LinW w{};
        w.mx8 = m;
        return w;
    }
};

inline size_t x8_bytes(int M, int K) {
    return static_cast<size_t>((M + 15) / 16) * static_cast<size_t>((K + 31) / 32) * MX8_REC;
}

struct Mx8Scratch {
    uint8_t* x8 = nullptr; // [max_RB * max_S * MX8_REC]
    size_t bytes = 0;
};

__device__ __forceinline__ void cp16(void* dst_smem, const void* src_gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(s), "l"(src_gmem));
#else
    *static_cast<uint4*>(dst_smem) = *static_cast<const uint4*>(src_gmem);
#endif
}
__device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    (void)p; r0 = r1 = r2 = r3 = 0;
#endif
}
__device__ __forceinline__ void mma_mxfp8(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1,
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

__device__ __forceinline__ uint32_t scale_ue8m0(float amax, float& inv) {
    if (amax == 0.0f) { inv = 0.0f; return 127u; }
    const float raw = amax * (1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(raw);
    uint32_t exp = (bits >> 23) & 0xffu;
    if ((bits & 0x7fffffu) != 0u) ++exp;
    if (exp < 1u) exp = 1u;
    if (exp > 254u) exp = 254u;
    inv = __uint_as_float((254u - exp) << 23);
    return exp;
}

__device__ __forceinline__ uint32_t f2_e4m3x2(float a, float b) {
    uint16_t r = 0;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(a), "f"(b));
    return static_cast<uint32_t>(r);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float h16_sum(float v) {
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void fp8x4_unpack(uint32_t v, float2& f0, float2& f1) {
    uint32_t r0, r1;
    const uint16_t lo = static_cast<uint16_t>(v & 0xffffu);
    const uint16_t hi = static_cast<uint16_t>(v >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r0) : "h"(lo));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r1) : "h"(hi));
    f0 = __half22float2(*reinterpret_cast<const __half2*>(&r0));
    f1 = __half22float2(*reinterpret_cast<const __half2*>(&r1));
}

__device__ __forceinline__ float dot_bf16x16(uint4 a0, uint4 a1, const float2 (&wf)[8]) {
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

// ------------------------------------------------------------------------------------------- Boot Weight Quantization
__global__ void __launch_bounds__(256) k_hy4_bf16_to_mx8_w(
    const __nv_bfloat16* __restrict__ W,
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
    const float w = __bfloat162float(W[base]);
    const float amax = warp_max(fabsf(w));
    float inv;
    const uint32_t code = scale_ue8m0(amax, inv);
    if (lane == 0) sc[blk] = static_cast<uint8_t>(code);
    q[base] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(w * inv, __NV_SATFINITE, __NV_E4M3));
}

inline void quantize_w(const __nv_bfloat16* W, Mx8Mat& dst, int N, int K, cudaStream_t st = 0) {
    dst.N = N; dst.K = K;
    const size_t total = static_cast<size_t>(N) * (K >> 5);
    k_hy4_bf16_to_mx8_w<<<static_cast<unsigned>((total + 7) / 8), 256, 0, st>>>(W, dst.q, dst.sc, N, K);
}

// ------------------------------------------------------------------------------------------- Activation Quantization
__global__ void __launch_bounds__(256) k_hy4_bf16_to_mx8_a(
    const __nv_bfloat16* __restrict__ src,
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
            code = scale_ue8m0(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                       (f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
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

inline void bf16_to_mx8_a(const __nv_bfloat16* src, uint8_t* dst, int M, int K, cudaStream_t st) {
    const int RB = (M + 15) >> 4;
    const int ST = ((K >> 5) + 15) >> 4;
    k_hy4_bf16_to_mx8_a<<<RB * ST, 256, 0, st>>>(src, dst, M, K);
}

// ------------------------------------------------------------------------------------------- Small-M MXFP8 GEMV
template <int MAX_M, bool EXACT_M, bool OUT_F32 = false, bool HAS_BIAS = false>
__global__ void __launch_bounds__(128) k_hy4_mx8_gemv(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    const __nv_bfloat16* __restrict__ bias,
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
        fp8x4_unpack(wv.x, wf[0], wf[1]);
        fp8x4_unpack(wv.y, wf[2], wf[3]);
        fp8x4_unpack(wv.z, wf[4], wf[5]);
        fp8x4_unpack(wv.w, wf[6], wf[7]);

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const uint4* xm = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K);
                acc[m] = fmaf(sc, dot_bf16x16(__ldg(xm + 2 * i), __ldg(xm + 2 * i + 1), wf), acc[m]);
            }
        }
        wv = wv_next;
        sv = sv_next;
    }
    const float bv = HAS_BIAS ? __bfloat162float(bias[o]) : 0.0f;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = h16_sum(acc[m]) + bv;
            if (hlane == 0) {
                if constexpr (OUT_F32) static_cast<float*>(C)[static_cast<size_t>(m) * N + o] = tot;
                else static_cast<__nv_bfloat16*>(C)[static_cast<size_t>(m) * N + o] = __float2bfloat16(tot);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------- TC MXFP8 GEMM (M >= 16)
template <int MB, bool OUT_F32 = false, bool HAS_BIAS = false>
__global__ void __launch_bounds__(128) k_hy4_gemm_mx8_small(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    const __nv_bfloat16* __restrict__ bias,
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
                cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + A_BYTES;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 128 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
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
    cp_commit();
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
        cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        cp_commit();

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
            ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * MB + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mma_mxfp8(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mma_mxfp8(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = c0 + colw + h * 8 + 2 * c;
            if (col >= N) continue;
            const float b0 = HAS_BIAS ? __bfloat162float(bias[col]) : 0.0f;
            const float b1 = HAS_BIAS ? __bfloat162float(bias[col + 1]) : 0.0f;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][h][0] + b0, acc[mb][h][1] + b1);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][h][2] + b0, acc[mb][h][3] + b1);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = pack_bf16x2(acc[mb][h][0] + b0, acc[mb][h][1] + b1);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = pack_bf16x2(acc[mb][h][2] + b0, acc[mb][h][3] + b1);
            }
        }
    }
}

struct Mx8LargeCfg {
    static constexpr int BM = 4;
    static constexpr int KS = 4;
    static constexpr int ST = 2;
    static constexpr int A = KS * BM * MX8_REC;
    static constexpr int B = 128 * KS * 32;
    static constexpr int STAGE = A + B;
    static constexpr int SMEM = ST * STAGE;
};

template <bool OUT_F32 = false, bool HAS_BIAS = false>
__global__ void __launch_bounds__(256, 2) k_hy4_gemm_mx8_large(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    const __nv_bfloat16* __restrict__ bias,
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
                cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3, ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
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
    cp_commit();
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
        cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        cp_commit();

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
            ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                mma_mxfp8(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) mma_mxfp8(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = c0 + colw + nt * 8 + 2 * c;
            if (col >= N) continue;
            const float b0 = HAS_BIAS ? __bfloat162float(bias[col]) : 0.0f;
            const float b1 = HAS_BIAS ? __bfloat162float(bias[col + 1]) : 0.0f;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][nt][0] + b0, acc[mb][nt][1] + b1);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][nt][2] + b0, acc[mb][nt][3] + b1);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = pack_bf16x2(acc[mb][nt][0] + b0, acc[mb][nt][1] + b1);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = pack_bf16x2(acc[mb][nt][2] + b0, acc[mb][nt][3] + b1);
            }
        }
    }
}

inline void ensure_large_smem_attr() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_hy4_gemm_mx8_large<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_hy4_gemm_mx8_large<false, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_hy4_gemm_mx8_large<true,  false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_hy4_gemm_mx8_large<true,  true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
}

template <bool OUT_F32 = false, bool HAS_BIAS = false>
inline void gemm_mx8_impl(const __nv_bfloat16* X, const uint8_t* x8, const Mx8Mat& B,
                          const __nv_bfloat16* bias, void* C, int M, cudaStream_t st) {
    if (M <= 15 && X != nullptr) {
        const unsigned g = static_cast<unsigned>((B.N + 7) / 8);
        if (M == 1)      k_hy4_mx8_gemv<1, true,  OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 1, B.N, B.K);
        else if (M == 2) k_hy4_mx8_gemv<2, true,  OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 2, B.N, B.K);
        else if (M == 4) k_hy4_mx8_gemv<4, true,  OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 4, B.N, B.K);
        else if (M == 6) k_hy4_mx8_gemv<6, true,  OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 6, B.N, B.K);
        else if (M == 8) k_hy4_mx8_gemv<8, true,  OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 8, B.N, B.K);
        else if (M <  8) k_hy4_mx8_gemv<8, false, OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, M, B.N, B.K);
        else {
            k_hy4_mx8_gemv<8, true, OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X, B.q, B.sc, bias, C, 8, B.N, B.K);
            const int rem = M - 8;
            void* C2 = OUT_F32
                ? static_cast<void*>(static_cast<float*>(C) + static_cast<size_t>(8) * B.N)
                : static_cast<void*>(static_cast<__nv_bfloat16*>(C) + static_cast<size_t>(8) * B.N);
            k_hy4_mx8_gemv<8, false, OUT_F32, HAS_BIAS><<<g, 128, 0, st>>>(X + static_cast<size_t>(8) * B.K, B.q, B.sc, bias, C2, rem, B.N, B.K);
        }
        return;
    }
    const int RB = (M + 15) >> 4;
    if (M <= 16) {
        k_hy4_gemm_mx8_small<1, OUT_F32, HAS_BIAS><<<dim3((B.N + 63) / 64, RB), 128, 0, st>>>(x8, B.q, B.sc, bias, C, M, B.N, B.K);
    } else if (M <= 32) {
        k_hy4_gemm_mx8_small<2, OUT_F32, HAS_BIAS><<<dim3((B.N + 63) / 64, (RB + 1) / 2), 128, 0, st>>>(x8, B.q, B.sc, bias, C, M, B.N, B.K);
    } else if (M <= 64) {
        k_hy4_gemm_mx8_small<4, OUT_F32, HAS_BIAS><<<dim3((B.N + 63) / 64, (RB + 3) / 4), 128, 0, st>>>(x8, B.q, B.sc, bias, C, M, B.N, B.K);
    } else {
        ensure_large_smem_attr();
        k_hy4_gemm_mx8_large<OUT_F32, HAS_BIAS><<<dim3((B.N + 127) / 128, (RB + 3) / 4), 256, Mx8LargeCfg::SMEM, st>>>(x8, B.q, B.sc, bias, C, M, B.N, B.K);
    }
}

// Unified linear dispatch: uses MXFP8 when B.is_mx8(), else exact BF16 (small-M GEMV or cuBLAS).
// When prequant_x8 != nullptr and M >= 16, reuses already-quantized x8 (zero redundant quantization!).
inline void linear_w(cublasHandle_t blas, const __nv_bfloat16* A, const LinW& W,
                     __nv_bfloat16* C, int M, int N, int K,
                     uint8_t* x8_scratch, const uint8_t* prequant_x8, cudaStream_t st,
                     const __nv_bfloat16* bias = nullptr) {
    if (W.is_mx8()) {
        const uint8_t* x8 = prequant_x8;
        if (M >= 16 && x8 == nullptr) {
            bf16_to_mx8_a(A, x8_scratch, M, K, st);
            x8 = x8_scratch;
        }
        if (bias) gemm_mx8_impl<false, true> (A, x8, W.mx8, bias, C, M, st);
        else      gemm_mx8_impl<false, false>(A, x8, W.mx8, nullptr, C, M, st);
    } else {
        linear_bf16(blas, A, W.bf16, C, M, N, K, st, bias);
    }
}

// Batched head-wise MXFP8 dequant to temporary BF16 (only used for prefill M > 12 batched w_uk_t / w_uv GEMM,
// or we run per-head gemm_mx8 directly!). Let's support direct per-head MXFP8 GEMM for M > 12:
// w_batched: [nh, N_h, K_h] contiguous MXFP8 -> runs nh head slices against x[nh, M, K_h] -> y[nh, M, N_h]!
// Actually for M > 12, dequantizing w_uk_t [nh*512, 192] (~6 MB max) OR running a single batched MXFP8 kernel is instant!
__global__ void __launch_bounds__(256) k_hy4_mx8_dequant_bf16(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ sc,
    __nv_bfloat16* __restrict__ out,
    size_t total_blocks) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blocks) return;
    const float s = __uint_as_float(static_cast<uint32_t>(sc[blk]) << 23);
    const size_t idx = blk * 32 + lane;
    const __nv_fp8_e4m3 f = *reinterpret_cast<const __nv_fp8_e4m3*>(q + idx);
    out[idx] = __float2bfloat16(static_cast<float>(f) * s);
}

inline void dequant_to_bf16(const Mx8Mat& m, __nv_bfloat16* dst, cudaStream_t st) {
    const size_t nblk = static_cast<size_t>(m.N) * (m.K >> 5);
    k_hy4_mx8_dequant_bf16<<<static_cast<unsigned>((nblk + 7) / 8), 256, 0, st>>>(m.q, m.sc, dst, nblk);
}

// Contiguous arena allocator for boot-time MXFP8 dense quantization.
struct Mx8Arena {
    uint8_t* base = nullptr;
    size_t total_bytes = 0;
    size_t cursor = 0;

    static size_t align256(size_t x) { return (x + 255ull) & ~255ull; }
    static size_t mat_bytes(int N, int K) {
        return align256(static_cast<size_t>(N) * K) + align256(static_cast<size_t>(N) * (K >> 5));
    }

    void reserve(int N, int K) { total_bytes += mat_bytes(N, K); }

    bool alloc() {
        if (total_bytes == 0) return true;
        if (cudaMalloc(&base, total_bytes) != cudaSuccess) { base = nullptr; return false; }
        cursor = 0;
        return true;
    }

    Mx8Mat add(const __nv_bfloat16* W_bf16, int N, int K, cudaStream_t st) {
        Mx8Mat m{};
        m.q = base + cursor;
        cursor += align256(static_cast<size_t>(N) * K);
        m.sc = base + cursor;
        cursor += align256(static_cast<size_t>(N) * (K >> 5));
        quantize_w(W_bf16, m, N, K, st);
        return m;
    }

    void free_all() {
        if (base) { cudaFree(base); base = nullptr; }
        total_bytes = cursor = 0;
    }
    ~Mx8Arena() { free_all(); }
};

} // namespace mx8
} // namespace hy4
