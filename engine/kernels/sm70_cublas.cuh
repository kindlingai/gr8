// SM70 cuBLAS BF16 compatibility helper.
// On compute capability >= 8.0 (Ampere+), forwards directly to cublasGemmEx with CUDA_R_16BF.
// On SM70 (Volta / V100), cuBLAS has no native BF16 Tensor Core support: converts BF16 A and B
// into FP16 scratch (clamped to +/-65504) and executes cublasGemmEx with CUDA_R_16F + CUBLAS_COMPUTE_32F
// + CUBLAS_GEMM_DEFAULT_TENSOR_OP (full 125 TFLOP/s Volta FP16 Tensor Cores), optionally rounding
// FP32 output back to BF16 when Ctype == CUDA_R_16BF.
#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace gr8_sm70
{
static __global__ void k_sm70_bf16_to_f16_mat(const uint16_t* __restrict__ src, int ld_src,
                                             __half* __restrict__ dst, int rows, int cols)
{
    const size_t total = static_cast<size_t>(rows) * cols;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / cols, c = i - r * cols;
    const uint32_t b = src[r * ld_src + c];
    const float f = __uint_as_float(b << 16);
    const float cl = fminf(fmaxf(f, -65504.0f), 65504.0f);
    dst[i] = __float2half_rn(cl);
}

static __global__ void k_sm70_f32_to_bf16_mat(const float* __restrict__ src, int ld_src,
                                             uint16_t* __restrict__ dst, int ld_dst, int rows, int cols)
{
    const size_t total = static_cast<size_t>(rows) * cols;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / cols, c = i - r * cols;
    uint32_t u = __float_as_uint(src[r * ld_src + c]);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    dst[r * ld_dst + c] = static_cast<uint16_t>(u >> 16);
}

inline int current_cc_major()
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 8;
    int maj = 8;
    if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 8;
    return maj;
}

struct Sm70BlasScratch
{
    __half* a = nullptr;
    size_t a_cap = 0;
    __half* b = nullptr;
    size_t b_cap = 0;
    float* c = nullptr;
    size_t c_cap = 0;
    int dev = -1;

    bool ensure(size_t na, size_t nb, size_t nc)
    {
        int cur = 0;
        if (cudaGetDevice(&cur) != cudaSuccess) return false;
        if (dev != cur)
        {
            if (a) cudaFree(a);
            if (b) cudaFree(b);
            if (c) cudaFree(c);
            a = nullptr; b = nullptr; c = nullptr;
            a_cap = b_cap = c_cap = 0;
            dev = cur;
        }
        if (na > a_cap)
        {
            if (a) cudaFree(a);
            a = nullptr; a_cap = 0;
            if (cudaMalloc(&a, na * sizeof(__half)) != cudaSuccess) return false;
            a_cap = na;
        }
        if (nb > b_cap)
        {
            if (b) cudaFree(b);
            b = nullptr; b_cap = 0;
            if (cudaMalloc(&b, nb * sizeof(__half)) != cudaSuccess) return false;
            b_cap = nb;
        }
        if (nc > c_cap)
        {
            if (c) cudaFree(c);
            c = nullptr; c_cap = 0;
            if (cudaMalloc(&c, nc * sizeof(float)) != cudaSuccess) return false;
            c_cap = nc;
        }
        return true;
    }
};

// Drop-in replacement for cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha,
//   W[N, ldw] BF16, ldw, X[M, ldx] BF16, ldx, beta, Y[M, ldy] (F32 or BF16), ldy, CUBLAS_COMPUTE_32F, algo).
inline cublasStatus_t cublas_gemm_bf16_tn(cublasHandle_t h, int N, int M, int K,
                                          const float* alpha, const void* W, int ldw,
                                          const void* X, int ldx, const float* beta,
                                          void* Y, cudaDataType_t y_type, int ldy,
                                          cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
{
    if (current_cc_major() >= 8)
    {
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha,
                            W, CUDA_R_16BF, ldw, X, CUDA_R_16BF, ldx,
                            beta, Y, y_type, ldy, CUBLAS_COMPUTE_32F, algo);
    }
    cudaStream_t st = nullptr;
    cublasGetStream(h, &st);
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
        return CUBLAS_STATUS_NOT_SUPPORTED;

    static std::mutex mu;
    static Sm70BlasScratch sc;
    std::lock_guard<std::mutex> lk(mu);
    const size_t na = static_cast<size_t>(N) * K;
    const size_t nb = static_cast<size_t>(M) * K;
    const size_t nc = (y_type == CUDA_R_16BF) ? static_cast<size_t>(M) * N : 0;
    if (!sc.ensure(na, nb, nc)) return CUBLAS_STATUS_ALLOC_FAILED;

    k_sm70_bf16_to_f16_mat<<<static_cast<unsigned>((na + 255) / 256), 256, 0, st>>>(
        static_cast<const uint16_t*>(W), ldw, sc.a, N, K);
    k_sm70_bf16_to_f16_mat<<<static_cast<unsigned>((nb + 255) / 256), 256, 0, st>>>(
        static_cast<const uint16_t*>(X), ldx, sc.b, M, K);

    void* dst = (y_type == CUDA_R_16BF) ? static_cast<void*>(sc.c) : Y;
    const int ld_dst = (y_type == CUDA_R_16BF) ? N : ldy;
    const cublasStatus_t rc = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha,
        sc.a, CUDA_R_16F, K, sc.b, CUDA_R_16F, K,
        beta, dst, CUDA_R_32F, ld_dst, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (rc != CUBLAS_STATUS_SUCCESS) return rc;

    if (y_type == CUDA_R_16BF)
    {
        k_sm70_f32_to_bf16_mat<<<static_cast<unsigned>((nc + 255) / 256), 256, 0, st>>>(
            sc.c, N, static_cast<uint16_t*>(Y), ldy, M, N);
    }
    return CUBLAS_STATUS_SUCCESS;
}
}   // namespace gr8_sm70
#endif   // __CUDACC__
