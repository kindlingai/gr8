// DeepSeek-V4.1-Flash source-layer fast path, part (b): the ratio-2 compressor projections (wkv, wgate) for M <= 8 rows as a thread-per-output GEMV
// that is bit-identical to k_f32gemm2 (ds41_attn.cuh): every output element is one sequential fp32 FMA chain over k = 0..K-1 starting at 0.0f, with
// the same operands (bf16 -> fp32 is exact), so only the schedule changes. GR8_DS_SRC_FAST bit 0 (=1) turns it on, default off.
//
// k_f32gemm2 runs 16 CTAs of 256 threads and 160 serial K steps of 32 (scalar 2-byte loads, two __syncthreads per step) over 10.5 MB of weights
// (floor 41 us at 255.8 GB/s). Here: grid (N/32, 2) CTAs, lane = output column n of one 32-wide group, the WHOLE weight row of that column is read
// by that lane in 16-byte loads (64-k tiles, next tile prefetched into registers), x (M rows, <= 80 KB bf16) is staged in shared memory once. The
// R = 3 rows per warp split keeps the per-warp instruction count at 3 rows (M = 6 -> 2 warps, M = 8 -> 3 warps); every (row, n) chain is still
// one thread's sequential FMA chain, so the order of additions does not depend on M, R or the schedule.
//
// src_gemv2_lane is __host__ __device__ so engine/attn/tests/ds41_src_fast_test.cpp runs the very same loop on the CPU against a transcription of
// k_f32gemm2 and compares the fp32 bit patterns.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <math.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define DS41_SF_HD __host__ __device__ __forceinline__
#else
#define DS41_SF_HD inline
#endif
#ifdef __CUDA_ARCH__
#define DS41_SF_UNROLL _Pragma("unroll")
#else
#define DS41_SF_UNROLL
#endif

namespace ds41a
{
struct alignas(16) SfV4 { uint32_t v[4]; };   // 8 bf16, element 2q in the low half of word q (little endian)
DS41_SF_HD float sf_u2f(uint32_t u)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}
constexpr int SF_R = 3;       // rows per warp
constexpr int SF_MAXM = 8;    // rows of one pass (the decode and verify case)

// One lane = one output column. wr = its weight row as SfV4 (K/8 of them); xs = staged x, row r at xs + r * xld8 (SfV4 units); rows r0 .. r0+R-1 < M
// are summed into acc in k order. nt = K / 64.
template <int R>
DS41_SF_HD void src_gemv2_lane(const SfV4* __restrict__ wr, const SfV4* __restrict__ xs, int xld8, int r0, int M, int nt, float (&acc)[R])
{
    SfV4 wc[8], wn[8];
DS41_SF_UNROLL
    for (int g = 0; g < 8; g++) wc[g] = wr[g];
    for (int t = 0; t < nt; t++)
    {
        if (t + 1 < nt)
        {
DS41_SF_UNROLL
            for (int g = 0; g < 8; g++) wn[g] = wr[(size_t)(t + 1) * 8 + g];
        }
DS41_SF_UNROLL
        for (int g = 0; g < 8; g++)
        {
            float wf[8];
DS41_SF_UNROLL
            for (int q = 0; q < 4; q++)
            {
                wf[2 * q] = sf_u2f(wc[g].v[q] << 16);
                wf[2 * q + 1] = sf_u2f(wc[g].v[q] & 0xFFFF0000u);
            }
DS41_SF_UNROLL
            for (int r = 0; r < R; r++)
            {
                if (r0 + r < M)
                {
                    const SfV4 xv = xs[(size_t)(r0 + r) * xld8 + (size_t)t * 8 + g];
DS41_SF_UNROLL
                    for (int q = 0; q < 4; q++)
                    {
                        acc[r] = fmaf(sf_u2f(xv.v[q] << 16), wf[2 * q], acc[r]);
                        acc[r] = fmaf(sf_u2f(xv.v[q] & 0xFFFF0000u), wf[2 * q + 1], acc[r]);
                    }
                }
            }
        }
        if (t + 1 < nt)
        {
DS41_SF_UNROLL
            for (int g = 0; g < 8; g++) wc[g] = wn[g];
        }
    }
}

#ifdef __CUDACC__
// x bf16 [M][K] (row stride K), W0 / W1 bf16 [N][K], Y0 / Y1 fp32 [M][N] (the k_f32gemm2 contract). grid (N / 32, 2), block 32 * ceil(M / R), dynamic smem M * K * 2.
template <int R>
__global__ void __launch_bounds__(96) k_f32gemv2(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
                                                 float* __restrict__ Y0, float* __restrict__ Y1, int M, int N, int K)
{
    extern __shared__ __align__(16) unsigned char sf_smem[];
    SfV4* xs = (SfV4*)sf_smem;
    const __nv_bfloat16* W = blockIdx.y ? W1 : W0;
    float* Y = blockIdx.y ? Y1 : Y0;
    const int xld8 = K >> 3;
    {
        const SfV4* src = (const SfV4*)x;
        const int tot = M * xld8;
#pragma unroll 4
        for (int i = threadIdx.x; i < tot; i += blockDim.x) xs[i] = src[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 32 + lane;
    const int r0 = warp * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
    src_gemv2_lane<R>((const SfV4*)(W + (size_t)n * K), xs, xld8, r0, M, K >> 6, acc);
#pragma unroll
    for (int r = 0; r < R; r++)
        if (r0 + r < M) Y[(size_t)(r0 + r) * N + n] = acc[r];
}

// false = shape / alignment / smem outside the fast path's contract (the caller launches k_f32gemm2)
inline bool launch_f32gemv2(const __nv_bfloat16* x, const __nv_bfloat16* W0, const __nv_bfloat16* W1, float* Y0, float* Y1, int M, int N, int K,
                            cudaStream_t st)
{
    if (M < 1 || M > SF_MAXM) return false;   // prefill chunks: k_f32gemm2, silently
    static bool warned = false;
    const size_t sm = (size_t)M * K * 2;
    if ((N & 31) || (K & 63) || K < 64 || (((uintptr_t)x | (uintptr_t)W0 | (uintptr_t)W1) & 15) || sm > 96 * 1024)
    {
        if (!warned) std::fprintf(stderr, "DS41 GR8_DS_SRC_FAST: compressor GEMV refused (N=%d K=%d M=%d x=%p W0=%p W1=%p): k_f32gemm2 stays\n", N, K, M, (const void*)x, (const void*)W0, (const void*)W1);
        warned = true;
        return false;
    }
    static size_t set_sm = 0;
    if (sm > set_sm)
    {
        if (cudaFuncSetAttribute(k_f32gemv2<SF_R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) != cudaSuccess)
        {
            if (!warned) std::fprintf(stderr, "DS41 GR8_DS_SRC_FAST: cudaFuncSetAttribute(%zu B) failed: k_f32gemm2 stays\n", sm);
            warned = true;
            return false;
        }
        set_sm = sm;
    }
    const int warps = (M + SF_R - 1) / SF_R;
    k_f32gemv2<SF_R><<<dim3(N / 32, 2), warps * 32, sm, st>>>(x, W0, W1, Y0, Y1, M, N, K);
    return true;
}
#endif
}   // namespace ds41a
