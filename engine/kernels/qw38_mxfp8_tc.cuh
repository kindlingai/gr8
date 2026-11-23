// Blackwell (sm_120a / sm_121a) native MXFP8 Tensor Core GEMM for Qwen3.8-Flash-Next (GR8_QW38_PF_MXFP8=1).
// Uses mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
// directly on the already-resident Fp8Mat weights (UE8M0 scales -> ZERO extra VRAM, -131 MB vs FP16 scales!).
//
// Full 128-byte coalesced B loads (KS=4 = 128 k-bytes/row = 8 x 16B cp.async per row) + bank-conflict-free
// XOR-swizzled ldmatrix.x4 for BOTH:
//   - Small/Medium-M verify batches (M = 16..64: c4/c8 MTP verify): 4 warps = 64 cols x MB*16 rows/CTA (20-32 KB smem)
//   - Large-M prefill chunks (M > 64): 8 warps = 128 cols x 64 rows/CTA (48.5 KB smem, 2 CTAs/SM, L2 grid swizzle)
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "qw38_fp8.cuh"

namespace qw38 {

constexpr int MX8_REC = 528; // 16 rows x 32 k: 512 B E4M3 A-fragments (32 lanes x 16 B) + 16 B UE8M0 row scales

__device__ __forceinline__ void qw38_cp16(void* dst_smem, const void* src_gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(s), "l"(src_gmem));
#else
    *static_cast<uint4*>(dst_smem) = *static_cast<const uint4*>(src_gmem);
#endif
}

__device__ __forceinline__ void qw38_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void qw38_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void qw38_ldsm_x4(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#endif
}

// d += 2^(sa-127) * 2^(sb-127) * (A @ B), A 16x32 row-major E4M3, B 32x8 col-major E4M3.
__device__ __forceinline__ void qw38_mma_mx8(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1,
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

__device__ __forceinline__ uint32_t qw38_f2_e4m3x2(float a, float b) {
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

__device__ __forceinline__ uint32_t qw38_pack_bf16x2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Quantize BF16 A[M, K] -> coalesced MXFP8 A-records dst[ceil(M/16)][K/32][528].
// 1 CTA = 16 rows x 16 consecutive k32 groups (512 k), written via coalesced 16-B stores from smem.
__global__ void __launch_bounds__(256) k_qw38_bf16_to_mx8(
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
            code = qw38_mx_scale(amax, inv);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c[j] = qw38_f2_e4m3x2(f[2 * j].x * inv, f[2 * j].y * inv) |
                       (qw38_f2_e4m3x2(f[2 * j + 1].x * inv, f[2 * j + 1].y * inv) << 16);
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

inline void qw38_bf16_to_mx8(const __nv_bfloat16* src, uint8_t* dst, int M, int K, cudaStream_t st) {
    const int RB = (M + 15) >> 4;
    const int ST = ((K >> 5) + 15) >> 4;
    k_qw38_bf16_to_mx8<<<RB * ST, 256, 0, st>>>(src, dst, M, K);
}

// Small/Medium-M MXFP8 GEMM (M = 16..64: c4/c8 MTP verify).
// 4 warps/CTA = 64 output cols x MB*16 rows, KS = 4 (128 B/row = 100% coalesced 128-B DRAM bursts!),
// 2-stage cp.async A+B pipeline + bank-conflict-free ldmatrix.x4 + mma_mx8. Static smem: 20-33 KB (3-4 CTAs/SM!).
template <int MB, bool OUT_F32 = false>
__global__ void __launch_bounds__(128) k_qw38_gemm_mx8_small(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    void* __restrict__ C,
    int M, int N, int K) {
    constexpr int KS = 4;                  // 4 x 32 = 128 k-bytes per row per stage (exact 128-B cache line!)
    constexpr int ST = 2;                  // double-buffered cp.async
    constexpr int A_BYTES = KS * MB * MX8_REC;
    constexpr int B_BYTES = 64 * KS * 32;  // 8192 B
    constexpr int STAGE = A_BYTES + B_BYTES;
    __shared__ __align__(16) uint8_t sm[ST * STAGE];

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5; // 0..3 (each warp owns 16 cols = 2 x n8 tiles)
    const int S = K >> 5;
    const int RB = (M + 15) >> 4;
    const int NST = (S + KS - 1) / KS;
    const int c0 = blockIdx.x * 64;
    const int rb0 = blockIdx.y * MB;
    const int nb = min(MB, RB - rb0);
    const int ncols = min(64, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * STAGE;
        for (int c = t; c < KS * MB * 33; c += 128) {
            const int j = c / (MB * 33);
            const int rem = c % (MB * 33);
            const int mb = rem / 33;
            const int ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                qw38_cp16(a + c * 16,
                          x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + A_BYTES;
        // 64 rows x 8 16-B chunks = 512 chunks -> 8 consecutive threads read 128 contiguous bytes of row n!
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 128 + t;
            const int n = c >> 3;
            const int ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                qw38_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                          Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw) / 8));
    const uint8_t* sp = Bsc + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            v[j][0] = (nvalid > 0 && ks < S) ? static_cast<uint32_t>(__ldg(sp + ks)) : 127u;
            v[j][1] = (nvalid > 1 && ks < S) ? static_cast<uint32_t>(__ldg(sp + static_cast<size_t>(8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    qw38_cp_commit();
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
        qw38_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        qw38_cp_commit();

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
            qw38_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * MB + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                qw38_mma_mx8(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) qw38_mma_mx8(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    qw38_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (h >= nvalid) continue;
            const int col = c0 + colw + h * 8 + 2 * c;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][h][0], acc[mb][h][1]);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][h][2], acc[mb][h][3]);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = qw38_pack_bf16x2(acc[mb][h][0], acc[mb][h][1]);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = qw38_pack_bf16x2(acc[mb][h][2], acc[mb][h][3]);
            }
        }
    }
}

// Large-M MXFP8 Tensor Core GEMM (prefill M > 64):
// 8 warps = 128 output cols x 64 rows/CTA (BM=4), KS = 4 (128 B/row = 100% coalesced 128-B DRAM bursts!),
// 2-stage cp.async A+B pipeline (48.5 KB smem -> 2 CTAs/SM on sm_121a!) + bank-conflict-free ldmatrix.x4 + L2 grid swizzle!
struct Mx8LargeCfg {
    static constexpr int BM = 4;                 // 64 rows/CTA (2x CTA parallelism on TP=2!)
    static constexpr int KS = 4;                 // 128 k-bytes/row = exact 128-B cache line!
    static constexpr int ST = 2;                 // double-buffered cp.async
    static constexpr int A = KS * BM * MX8_REC;  // 8448 B
    static constexpr int B = 128 * KS * 32;      // 16384 B
    static constexpr int STAGE = A + B;          // 24832 B
    static constexpr int SMEM = ST * STAGE;      // 49664 B (48.5 KB -> 2 CTAs/SM on sm_121a!)
};

template <bool OUT_F32 = false>
__global__ void __launch_bounds__(256, 2) k_qw38_gemm_mx8_large(
    const uint8_t* __restrict__ x8,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsc,
    void* __restrict__ C,
    int M, int N, int K) {
    using Cfg = Mx8LargeCfg;
    constexpr int BM = Cfg::BM;
    constexpr int KS = Cfg::KS;
    extern __shared__ __align__(16) uint8_t sm[];

    // Row-block-fastest grid swizzle for L2 weight reuse across row blocks
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int by = lin % gridDim.y;
    const int bx = lin / gridDim.y;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5; // 8 warps = 1x8 (each warp: 64 rows x 16 cols)
    const int S = K >> 5, RB = (M + 15) >> 4, NST = (S + KS - 1) / KS;
    const int c0 = bx * 128, rb0 = by * BM;
    const int nb = min(BM, RB - rb0), ncols = min(128, N - c0);

    auto loadAB = [&](int slot, int st) {
        uint8_t* a = sm + slot * Cfg::STAGE;
        for (int c = t; c < KS * BM * 33; c += 256) {
            const int j = c / (BM * 33), rem = c % (BM * 33), mb = rem / 33, ch = rem % 33;
            if (mb < nb && st * KS + j < S) {
                qw38_cp16(a + c * 16, x8 + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * MX8_REC + ch * 16);
            }
        }
        uint8_t* b = a + Cfg::A;
        // 128 rows x 8 16-B chunks = 1024 chunks -> 4 iters/thread, 100% coalesced 128-B cache lines!
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = it * 256 + t;
            const int n = c >> 3;
            const int ch = c & 7;
            if (n < ncols && st * KS + (ch >> 1) < S) {
                qw38_cp16(b + n * 128 + ((ch ^ (n & 7)) << 4),
                          Bq + static_cast<size_t>(c0 + n) * K + st * 128 + (ch << 4));
            }
        }
    };

    const int colw = warp * 16;
    const int nvalid = min(2, max(0, (ncols - colw) / 8));
    const uint8_t* sp = Bsc + static_cast<size_t>(c0 + colw + (lane >> 2)) * S;
    auto loadS = [&](int st, uint32_t (&v)[KS][2]) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int ks = st * KS + j;
            v[j][0] = (nvalid > 0 && ks < S) ? static_cast<uint32_t>(__ldg(sp + ks)) : 127u;
            v[j][1] = (nvalid > 1 && ks < S) ? static_cast<uint32_t>(__ldg(sp + static_cast<size_t>(8) * S + ks)) : 127u;
        }
    };

    if (NST > 0) loadAB(0, 0);
    qw38_cp_commit();
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
        qw38_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NST) loadAB((st + 1) & 1, st + 1);
        qw38_cp_commit();

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
            qw38_ldsm_x4(b + n * 128 + ((ch ^ (n & 7)) << 4), bf00, bf01, bf10, bf11);
            #pragma unroll
            for (int mb = 0; mb < BM; ++mb) {
                if (mb >= nb) continue;
                const uint8_t* rec = a + (j * BM + mb) * MX8_REC;
                const uint4 fa = *reinterpret_cast<const uint4*>(rec + lane * 16);
                const uint32_t sa = rec[512 + srow];
                qw38_mma_mx8(acc[mb][0], fa, bf00, bf01, sa, cs[j][0]);
                if (nvalid > 1) qw38_mma_mx8(acc[mb][1], fa, bf10, bf11, sa, cs[j][1]);
            }
        }
    }
    qw38_cp_wait<0>();
    if (nvalid == 0) return;

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < BM; ++mb) {
        if (mb >= nb) continue;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            if (nt >= nvalid) continue;
            const int col = c0 + colw + nt * 8 + 2 * c;
            const int row = (rb0 + mb) * 16 + r;
            if constexpr (OUT_F32) {
                float* yf = static_cast<float*>(C);
                if (row < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row) * N + col) = make_float2(acc[mb][nt][0], acc[mb][nt][1]);
                if (row + 8 < M) *reinterpret_cast<float2*>(yf + static_cast<size_t>(row + 8) * N + col) = make_float2(acc[mb][nt][2], acc[mb][nt][3]);
            } else {
                uint16_t* y = static_cast<uint16_t*>(C);
                if (row < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row) * N + col) = qw38_pack_bf16x2(acc[mb][nt][0], acc[mb][nt][1]);
                if (row + 8 < M) *reinterpret_cast<uint32_t*>(y + static_cast<size_t>(row + 8) * N + col) = qw38_pack_bf16x2(acc[mb][nt][2], acc[mb][nt][3]);
            }
        }
    }
}

// Call once at Engine::init() (outside CUDA graph capture) to raise dynamic smem limit for large prefill kernels.
inline void qw38_mxfp8_init_attrs() {
    cudaFuncSetAttribute(k_qw38_gemm_mx8_large<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
    cudaFuncSetAttribute(k_qw38_gemm_mx8_large<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, Mx8LargeCfg::SMEM);
}

template <bool OUT_F32 = false>
inline void qw38_gemm_mx8_nt(const uint8_t* x8, const Fp8Mat& B, void* C, int M, cudaStream_t st) {
    const int RB = (M + 15) >> 4;
    if (M <= 16) {
        const dim3 grid((B.N + 63) / 64, RB);
        k_qw38_gemm_mx8_small<1, OUT_F32><<<grid, 128, 0, st>>>(x8, B.q, B.mx_sc, C, M, B.N, B.K);
    } else if (M <= 32) {
        const dim3 grid((B.N + 63) / 64, (RB + 1) / 2);
        k_qw38_gemm_mx8_small<2, OUT_F32><<<grid, 128, 0, st>>>(x8, B.q, B.mx_sc, C, M, B.N, B.K);
    } else if (M <= 64) {
        const dim3 grid((B.N + 63) / 64, (RB + 3) / 4);
        k_qw38_gemm_mx8_small<4, OUT_F32><<<grid, 128, 0, st>>>(x8, B.q, B.mx_sc, C, M, B.N, B.K);
    } else {
        const dim3 grid((B.N + 127) / 128, (RB + 3) / 4);
        k_qw38_gemm_mx8_large<OUT_F32><<<grid, 256, Mx8LargeCfg::SMEM, st>>>(x8, B.q, B.mx_sc, C, M, B.N, B.K);
    }
}

} // namespace qw38
