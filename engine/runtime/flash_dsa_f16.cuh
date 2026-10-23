// Opt-in FP16 cuBLAS arm for the DSA input projections (G6_PF_DSA_F16GEMM=1, default 0): q_a, kv_a, idx_k, idx_q
// (MXFP8 records, fp16 GEMM) and idx_w + idx_g (BF16W records, one concatenated bf16 GEMM).
//
// Today both run as MXFP8 GV records (densev.cuh Mxfp8): E4M3 weight codes decoded to fp16, fp16 activation fragments,
// fp16 MMA with fp32 accumulate, one E8M0 scale per (row, 32 inputs) multiplying the group's fp32 partial (exact, a
// power of two). This arm decodes each record once, at bind, into a plain fp16 matrix W'[O][K] = code * s / R with one
// power-of-two R per matrix (R = s_max * 2^-7), and runs cublasGemmEx (fp16 x fp16, fp32 compute, alpha = R) on the
// plain fp16 activation rows that hold the fragment tables' values (q_a: P.x, idx_q: q_a_n). cuBLAS writes fp32, and a
// second pass rounds to fp16: with fp16 output, cuBLAS's split-K kernels (it picks them for some row counts) keep
// partial sums in fp16 before alpha, and the first token's partials overflow.
// W' is exact when every product code * s / R is an fp16 value (scale spread within a matrix <= 2^22); the bind
// counts the elements where it is not and keeps that matrix on the GV kernel. Operands are then the same numbers as
// the GV kernel's; only the fp32 summation order differs (near-byte, not byte-equal).
// BF16W records (bf16 x bf16 MMA in the GV kernel) are copied to a plain bf16 matrix (always exact); the activation
// rows are unpacked from the same bf16 fragment table the GV kernel reads; cuBLAS writes fp32, rounded to fp16 rows.
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace g6rt
{
namespace flash_dsa_f16
{

inline bool on()
{
    static const bool v = getenv("G6_PF_DSA_F16GEMM") && atoi(getenv("G6_PF_DSA_F16GEMM")) != 0;
    return v;
}

// byte offsets in an MXFP8 GV record tensor [O/16][NG = K/32][528] (densev.cuh Mxfp8): two 16x16 tiles of 256 B in
// m16n8k16 A-fragment order (lane l = 4r + c holds {row r k 2c,2c+1 | row r+8 k 2c,2c+1} then the same at k + 8), then
// 16 scale bytes {s[r], s[r+8]} for r = 0..7.
__host__ __device__ inline size_t mx_code_offset(int o, int k, int NG)
{
    const int ot = o >> 4, rr = o & 15, r = rr & 7, hf = rr >> 3;
    const int g = k >> 5, i = (k >> 4) & 1, kk = k & 15, c = (kk & 7) >> 1, word = kk >> 3;
    return (static_cast<size_t>(ot) * NG + g) * 528 + i * 256 + (r * 4 + c) * 8 + word * 4 + hf * 2 + (kk & 1);
}
__host__ __device__ inline size_t mx_scale_offset(int o, int k, int NG)
{
    const int ot = o >> 4, rr = o & 15;
    return (static_cast<size_t>(ot) * NG + (k >> 5)) * 528 + 512 + (rr & 7) * 2 + (rr >> 3);
}

// byte offset of element (o, k) in a BF16W GV record tensor [O/16][NG = K/16][512] (densev.cuh Bf16w): one 16x16 tile
// in m16n8k16 A-fragment order, lane l = 4r + c holds a[0..3] = {row r | row r+8} x {k 2c,2c+1 | k 2c+8,2c+9}
__host__ __device__ inline size_t bf_offset(int o, int k, int NG)
{
    const int ot = o >> 4, rr = o & 15, r = rr & 7, hf = rr >> 3;
    const int kk = k & 15, c = (kk & 7) >> 1, hi = kk >> 3;
    return (static_cast<size_t>(ot) * NG + (k >> 4)) * 512 + (r * 4 + c) * 16 + (hi * 2 + hf) * 4 + (kk & 1) * 2;
}

__global__ void k_mx_emax(const uint8_t* rec, size_t groups, int* emax)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups * 16) return;
    atomicMax(emax, static_cast<int>(rec[(i >> 4) * 528 + 512 + (i & 15)]));
}

// W[o][k] = code * 2^(s - emax + 7) as fp16; bad += elements whose value is not exactly that
__global__ void k_mx_to_f16(const uint8_t* rec, int O, int K, const int* emax, __half* W, unsigned* bad)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(O) * K) return;
    const int o = static_cast<int>(i / K), k = static_cast<int>(i % K), NG = K / 32;
    const u32 h2 = e4m3x2(rec[mx_code_offset(o, k, NG)]);
    const float v = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
    const int shift = static_cast<int>(rec[mx_scale_offset(o, k, NG)]) - *emax + 7;
    const float vs = ldexpf(v, shift);
    const __half h = __float2half_rn(vs);
    if (!(__half2float(h) == vs && ldexpf(vs, -shift) == v)) atomicAdd(bad, 1u);
    W[i] = h;
}

// W[o][k] = the BF16W record's value (a copy)
__global__ void k_bf_to_rows(const uint8_t* rec, int O, int K, __nv_bfloat16* W)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(O) * K) return;
    const int o = static_cast<int>(i / K), k = static_cast<int>(i % K);
    W[i] = *reinterpret_cast<const __nv_bfloat16*>(rec + bf_offset(o, k, K / 16));
}

// plain rows [M][H] from a fragment table (k_mhc_fragments / bind::Xfrag layout; one u32 = columns h, h + 1)
__global__ void k_frag_rows(const u32* frag, int M, int H, u32* rows)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(M) * (H / 2)) return;
    const int m = static_cast<int>(i / (H / 2)), h = 2 * static_cast<int>(i % (H / 2));
    rows[i] = frag[static_cast<size_t>(m / 8) * (H / 16) * 64 + ((h / 16) * 32 + (m % 8) * 4 + (h % 8) / 2) * 2 + (h / 8) % 2];
}

// fp32 [M][O] -> fp16 columns 0..Oa-1 into ya[M][lda], Oa..O-1 into yb[M][ldb] (round to nearest)
__global__ void k_split_f16(const float* y, int M, int O, int Oa, __half* ya, int lda, __half* yb, int ldb)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(M) * O) return;
    const int m = static_cast<int>(i / O), o = static_cast<int>(i % O);
    const __half h = __float2half_rn(y[i]);
    if (o < Oa) ya[static_cast<size_t>(m) * lda + o] = h;
    else yb[static_cast<size_t>(m) * ldb + o - Oa] = h;
}

struct Mat
{
    void* w = nullptr;     // [O][K] fp16 (bf: bf16), null: this matrix stays on the GV kernel
    float alpha = 0.f;
    int O = 0, K = 0;
    bool bf = false;       // bf16 weights and rows, fp32 out (BF16W records)
    int Oa = 0;            // bf: rows 0..Oa-1 come from the first record, Oa..O-1 from the second
};

// decode one MXFP8 GV matrix; returns false (and frees) when any element is not exact or the scale is out of range
inline bool build(const bind::Gv& g, Mat& out, const char* name, int layer, cudaStream_t st)
{
    out = Mat();
    if (!g.on || g.codec != bind::MXFP8 || g.K % 32 || g.O % 16) return false;
    const size_t groups = static_cast<size_t>(g.O / 16) * (g.K / 32);
    int* emax = dmalloc<int>(1);
    unsigned* bad = dmalloc<unsigned>(1);
    G6RT_CK(cudaMemsetAsync(emax, 0, sizeof(int), st));
    G6RT_CK(cudaMemsetAsync(bad, 0, sizeof(unsigned), st));
    __half* w = dmalloc<__half>(static_cast<size_t>(g.O) * g.K);
    k_mx_emax<<<static_cast<unsigned>((groups * 16 + 255) / 256), 256, 0, st>>>(g.w, groups, emax);
    k_mx_to_f16<<<static_cast<unsigned>((static_cast<size_t>(g.O) * g.K + 255) / 256), 256, 0, st>>>(g.w, g.O, g.K, emax, w, bad);
    int e = 0;
    unsigned b = 0;
    G6RT_CK(cudaMemcpyAsync(&e, emax, sizeof e, cudaMemcpyDeviceToHost, st));
    G6RT_CK(cudaMemcpyAsync(&b, bad, sizeof b, cudaMemcpyDeviceToHost, st));
    G6RT_CK(cudaStreamSynchronize(st));
    cudaFree(emax);
    cudaFree(bad);
    if (b || e - 134 < -126)   // alpha = 2^(e - 127 - 7) must be a normal fp32
    {
        std::fprintf(stderr, "# DSA f16 gemm: layer %d %s stays on GV (%u inexact elements, scale code max %d)\n", layer, name, b, e);
        cudaFree(w);
        return false;
    }
    out.w = w;
    out.alpha = ldexpf(1.f, e - 134);
    out.O = g.O;
    out.K = g.K;
    return true;
}

// two BF16W GV matrices with the same K stacked into one bf16 matrix [Oa + Ob][K]; false when either is not BF16W
inline bool build_bf_cat(const bind::Gv& a, const bind::Gv& b, Mat& out, cudaStream_t st)
{
    out = Mat();
    for (const bind::Gv* g : {&a, &b})
        if (!g->on || g->codec != bind::BF16W || g->K % 16 || g->O % 16) return false;
    if (a.K != b.K) return false;
    const size_t na = static_cast<size_t>(a.O) * a.K, nb = static_cast<size_t>(b.O) * b.K;
    __nv_bfloat16* w = dmalloc<__nv_bfloat16>(na + nb);
    k_bf_to_rows<<<static_cast<unsigned>((na + 255) / 256), 256, 0, st>>>(a.w, a.O, a.K, w);
    k_bf_to_rows<<<static_cast<unsigned>((nb + 255) / 256), 256, 0, st>>>(b.w, b.O, b.K, w + na);
    G6RT_CK(cudaGetLastError());
    out.w = w;
    out.alpha = 1.f;
    out.O = a.O + b.O;
    out.K = a.K;
    out.bf = true;
    out.Oa = a.O;
    return true;
}

// y[M][O] (fp32) = x[M][ldx] (bf16) @ W^T, W a bf Mat
inline void gemm_bf(cublasHandle_t h, const Mat& w, const __nv_bfloat16* x, int ldx, int M, float* y, cudaStream_t st)
{
    const float zero = 0.f;
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) die("flash_dsa_f16: cublasSetStream failed");
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, w.O, M, w.K, &w.alpha, w.w, CUDA_R_16BF, w.K, x, CUDA_R_16BF, ldx, &zero,
                     y, CUDA_R_32F, w.O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        die("flash_dsa_f16: cublasGemmEx (bf16) failed");
}

// the bf path end to end: rows from the bf16 fragment table, GEMM, split into the two fp16 outputs
inline void gemm_bf_split(cublasHandle_t h, const Mat& w, const u32* xb_frag, int M, __nv_bfloat16* rows, float* y32,
                          __half* ya, int lda, __half* yb, int ldb, cudaStream_t st)
{
    const size_t pairs = static_cast<size_t>(M) * (w.K / 2), outs = static_cast<size_t>(M) * w.O;
    k_frag_rows<<<static_cast<unsigned>((pairs + 255) / 256), 256, 0, st>>>(xb_frag, M, w.K, reinterpret_cast<u32*>(rows));
    gemm_bf(h, w, rows, w.K, M, y32, st);
    k_split_f16<<<static_cast<unsigned>((outs + 255) / 256), 256, 0, st>>>(y32, M, w.O, w.Oa, ya, lda, yb, ldb);
}

// y[M][ldy] (fp16) = alpha * x[M][ldx] (fp16) @ W'^T, through y32 [M][O] fp32
inline void gemm(cublasHandle_t h, const Mat& w, const __half* x, int ldx, int M, float* y32, __half* y, int ldy,
                 cudaStream_t st)
{
    const float zero = 0.f;
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) die("flash_dsa_f16: cublasSetStream failed");
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, w.O, M, w.K, &w.alpha, w.w, CUDA_R_16F, w.K, x, CUDA_R_16F, ldx, &zero,
                     y32, CUDA_R_32F, w.O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        die("flash_dsa_f16: cublasGemmEx failed");
    const size_t outs = static_cast<size_t>(M) * w.O;
    k_split_f16<<<static_cast<unsigned>((outs + 255) / 256), 256, 0, st>>>(y32, M, w.O, w.O, y, ldy, nullptr, 0);
}

}   // namespace flash_dsa_f16
}   // namespace g6rt
