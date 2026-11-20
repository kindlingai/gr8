// Exact-input BF16 projections for sequence-parallel prefill; include globally.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ds41v2tc {
constexpr int BM = 64, BN = 64, BK = 64, ST = 3;
constexpr int TC_SMEM_BYTES = ST * (BM + BN) * BK * sizeof(__nv_bfloat16);

__device__ __forceinline__ void cp16_zfill(void* smem_ptr, const void* gmem_ptr, bool valid) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int src_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr), "l"(gmem_ptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16_m16n8k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template<class OutT, bool PAIRED>
__global__ __launch_bounds__(128, 2) void k_gemm_bf16_tc(
    const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W0, const __nv_bfloat16* W1,
    int N, int K, OutT* Y0, OutT* Y1, int ldy, int M) {
    const int grid_n = (N + BN - 1) / BN;
    const int inner = PAIRED ? (grid_n * 2) : grid_n;
    const int m_tile = blockIdx.x / inner;
    const int rem = blockIdx.x % inner;
    const int z = PAIRED ? (rem & 1) : 0;
    const int n_tile = PAIRED ? (rem >> 1) : rem;
    const int m0 = m_tile * BM;
    const int n0 = n_tile * BN;
    if (m0 >= M || n0 >= N) return;
    const auto* W = z ? W1 : W0;
    auto* Y = z ? Y1 : Y0;
    extern __shared__ __align__(16) unsigned char storage[];
    auto* stages = reinterpret_cast<uint4*>(storage);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int tiles = K / BK;
    auto issue = [&](int s, int kt) {
        if (kt < tiles) {
            auto* sa = stages + s * 1024;
            auto* sb = sa + 512;
            const int k_base = kt * BK;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = tid + j * 128, r = t >> 3, c8 = t & 7;
                const int c8_swz = c8 ^ (r & 7);
                const int gm = m0 + r;
                const __nv_bfloat16* gx = X + static_cast<size_t>(gm < M ? gm : 0) * ldx + k_base + c8 * 8;
                cp16_zfill(sa + r * 8 + c8_swz, gx, gm < M);
                const int gn = n0 + r;
                const __nv_bfloat16* gw = W + static_cast<size_t>(gn < N ? gn : 0) * K + k_base + c8 * 8;
                cp16_zfill(sb + r * 8 + c8_swz, gw, gn < N);
            }
        }
        // Empty tail groups keep wait_group<2> pointed at the current stage.
        cp_commit();
    };
    issue(0, 0);
    issue(1, 1);
    float acc[2][4][4] = {};
    const int a_r_sub = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_c_sub = (lane >> 4) & 1;
    const int b_r_sub = ((lane >> 4) * 8) + (lane & 7);
    const int b_c_sub = (lane >> 3) & 1;
#pragma unroll 1
    for (int kt = 0; kt < tiles; ++kt) {
        issue((kt + 2) % ST, kt + 2);
        cp_wait<2>();
        __syncthreads();
        const auto* sa = stages + (kt % ST) * 1024;
        const auto* sb = sa + 512;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t af[2][4], bf[2][4];
            const int c8_base = ks * 2;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm + mi * 16 + a_r_sub;
                ldmatrix_x4(af[mi], sa + r * 8 + ((c8_base + a_c_sub) ^ (r & 7)));
            }
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int r = wn + ni * 16 + b_r_sub;
                ldmatrix_x4(bf[ni], sb + r * 8 + ((c8_base + b_c_sub) ^ (r & 7)));
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_bf16_m16n8k16(acc[mi][ni], af[mi], bf[ni / 2][(ni & 1) * 2], bf[ni / 2][(ni & 1) * 2 + 1]);
            }
        }
        __syncthreads();
    }
    cp_wait<0>();
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r0 = m0 + wm + mi * 16 + g, r1 = r0 + 8;
            const int c0 = n0 + wn + ni * 8 + t * 2;
            if (c0 < N) {
                if constexpr (std::is_same_v<OutT, float>) {
                    if (r0 < M) *reinterpret_cast<float2*>(Y + static_cast<size_t>(r0) * ldy + c0) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
                    if (r1 < M) *reinterpret_cast<float2*>(Y + static_cast<size_t>(r1) * ldy + c0) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
                } else {
                    if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(r0) * ldy + c0) = __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
                    if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(r1) * ldy + c0) = __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
        }
    }
}
inline cudaError_t prepare() {
    cudaError_t e = cudaFuncSetAttribute(k_gemm_bf16_tc<float, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM_BYTES);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(k_gemm_bf16_tc<float, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM_BYTES);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(k_gemm_bf16_tc<__nv_bfloat16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM_BYTES);
    return e;
}
inline bool can_launch_f32(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W, int N, int K, const float* Y, int ldy, int M) {
    return X && W && Y && M >= 1 && N >= 2 && (N & 1) == 0 && K >= 64 && (K & 63) == 0 &&
           ldx >= K && (ldx & 7) == 0 && ldy >= N && (ldy & 1) == 0 &&
           ((uintptr_t(X) | uintptr_t(W)) & 15) == 0 && (uintptr_t(Y) & 7) == 0;
}
inline bool can_launch_f32_pair(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W0, const __nv_bfloat16* W1, int N, int K, const float* Y0, const float* Y1, int ldy, int M) {
    return can_launch_f32(X, ldx, W0, N, K, Y0, ldy, M) && can_launch_f32(X, ldx, W1, N, K, Y1, ldy, M);
}
inline bool can_launch_bf16(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W, int N, int K, const __nv_bfloat16* Y, int ldy, int M) {
    return X && W && Y && M >= 1 && N >= 2 && (N & 1) == 0 && K >= 64 && (K & 63) == 0 &&
           ldx >= K && (ldx & 7) == 0 && ldy >= N && (ldy & 1) == 0 &&
           ((uintptr_t(X) | uintptr_t(W)) & 15) == 0 && (uintptr_t(Y) & 3) == 0;
}
inline cudaError_t launch_gemm_f32(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W, int N, int K, float* Y, int ldy, int M, cudaStream_t st) {
    if (!can_launch_f32(X, ldx, W, N, K, Y, ldy, M)) return cudaErrorInvalidValue;
    const int m_tiles = (M + 63) / 64, n_tiles = (N + 63) / 64;
    k_gemm_bf16_tc<float, false><<<m_tiles * n_tiles, 128, TC_SMEM_BYTES, st>>>(X, ldx, W, nullptr, N, K, Y, nullptr, ldy, M);
    return cudaGetLastError();
}
inline cudaError_t launch_gemm2_f32(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W0, const __nv_bfloat16* W1, int N, int K, float* Y0, float* Y1, int ldy, int M, cudaStream_t st) {
    if (!can_launch_f32_pair(X, ldx, W0, W1, N, K, Y0, Y1, ldy, M)) return cudaErrorInvalidValue;
    const int m_tiles = (M + 63) / 64, n_tiles = (N + 63) / 64;
    k_gemm_bf16_tc<float, true><<<m_tiles * n_tiles * 2, 128, TC_SMEM_BYTES, st>>>(X, ldx, W0, W1, N, K, Y0, Y1, ldy, M);
    return cudaGetLastError();
}
inline cudaError_t launch_gemm_bf16(const __nv_bfloat16* X, int ldx, const __nv_bfloat16* W, int N, int K, __nv_bfloat16* Y, int ldy, int M, cudaStream_t st) {
    if (!can_launch_bf16(X, ldx, W, N, K, Y, ldy, M)) return cudaErrorInvalidValue;
    const int m_tiles = (M + 63) / 64, n_tiles = (N + 63) / 64;
    k_gemm_bf16_tc<__nv_bfloat16, false><<<m_tiles * n_tiles, 128, TC_SMEM_BYTES, st>>>(X, ldx, W, nullptr, N, K, Y, nullptr, ldy, M);
    return cudaGetLastError();
}
} // namespace ds41v2tc
