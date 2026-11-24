// Inkling-2-Small norms, depth attention (AttnRes), residual steps, the BF16 GEMM and the softcapped head (sm_121a).
// Our own code. The twins of ink2ref::Model::wnorm, attn_res and the head of forward() (engine/ref/ink2/ink2_cpu_ref.h).
//
//   rmsnorm      out = bf16(w * (x * r)), r = 1 / sqrt(mean(x^2) + 1e-6), the products in fp32, one rounding
//   embed_norm   the same on the embedding row of a token; an image token's feature row is copied and skips the norm
//   attn_res     per token over D <= 9 depth rows (the finished blocks' states, then the partial residual): U_d = row_d * r_d
//                in fp32 (never rounded); logit = query . U_d / sqrt(H); out = bf16(sum_d softmax_d * U_d), for Q = 1 or 3
//                queries. With a norm weight the mixture is rounded to BF16 first and that value is normalised, as the
//                reference does (attn_res then wnorm).
//   gemm         y [M][N] = x [M][K] W [N][K]^T on mma m16n8k16, BF16 operands and an fp32 sum; y BF16, fp32, or the head's
//                30 tanh(z / 30) with -inf from valid_vocab on
//   add / copy / sum_ranks   the residual steps and the sum of per-rank fp32 partials in rank order
//
// Sums over H run as one partial sum per thread and a tree over the CTA, and the GEMM sums in the order of the tensor
// core, so a result agrees with the reference to fp32 rounding of the sum, not bit for bit; every product, quotient and
// rounding that is not such a sum is the reference's own operation (explicit round-to-nearest, no fused multiply-add).
// A row's result does not depend on M or on the row's place in the batch.
//
// Shapes are arguments. H is any multiple of 16; M >= 1. This header also compiles as plain C++ (the scalar helpers).
#pragma once
#include "ink2_router.cuh"

namespace ink2
{
constexpr float RMS_EPS = 1e-6f;
constexpr int MAX_DEPTH = 9;   // depth rows of one depth attention: 8 finished blocks + the partial residual
constexpr int MAX_Q = 3;       // queries of one depth attention: q, k and v on a full-attention layer

// ---------------------------------------------------------------------------------------------- exact scalars
// One IEEE operation each: the device compiler may not fuse them into an fma (the host is built with -ffp-contract=off).
INK2_HD float mul_rn(float a, float b) { return nv4::mul_rn(a, b); }
INK2_HD float add_rn(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
INK2_HD float div_rn(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
INK2_HD float bf(uint16_t h) { return nv4::u2f(static_cast<u32>(h) << 16); }              // BF16 cell -> fp32, exact
INK2_HD uint16_t bf16(float f) { return static_cast<uint16_t>(nv4::bf16_rne(f)); }         // fp32 -> BF16 cell, RNE
INK2_HD float rbf(float f) { return bf(bf16(f)); }
INK2_HD float rsq(float x) { return div_rn(1.0f, sqrtf(x)); }                              // the reference's rsqrtf
// 30 tanh(z / 30)
INK2_HD float softcap30(float z) { return mul_rn(30.0f, tanhf(div_rn(z, 30.0f))); }

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------------- CTA reductions
// One value per thread -> the sum (or the maximum) over the CTA's nt threads (a power of two), the same bits on every
// thread. sm [nt] shared. Bit-exact with the full binary tree while using only 2 __syncthreads() (down from 9).
__device__ __forceinline__ float blk_sum(float v, float* sm, int nt)
{
    const int t = threadIdx.x;
    sm[t] = v;
    __syncthreads();
    if (nt == 256)
    {
        if (t < 32)
        {
            const float b0 = add_rn(add_rn(sm[t], sm[t + 128]), add_rn(sm[t + 64], sm[t + 192]));
            const float b1 = add_rn(add_rn(sm[t + 32], sm[t + 160]), add_rn(sm[t + 96], sm[t + 224]));
            float x = add_rn(b0, b1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x = add_rn(x, __shfl_down_sync(0xffffffffu, x, off));
            if (t == 0) sm[0] = x;
        }
        __syncthreads();
        const float r = sm[0];
        __syncthreads();
        return r;
    }
    if (nt == 128)
    {
        if (t < 32)
        {
            float x = add_rn(add_rn(sm[t], sm[t + 64]), add_rn(sm[t + 32], sm[t + 96]));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x = add_rn(x, __shfl_down_sync(0xffffffffu, x, off));
            if (t == 0) sm[0] = x;
        }
        __syncthreads();
        const float r = sm[0];
        __syncthreads();
        return r;
    }
    for (int s = nt >> 1; s > 0; s >>= 1)
    {
        if (t < s) sm[t] = add_rn(sm[t], sm[t + s]);
        __syncthreads();
    }
    const float r = sm[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float blk_max(float v, float* sm, int nt)
{
    const int t = threadIdx.x;
    sm[t] = v;
    __syncthreads();
    if (nt == 256)
    {
        if (t < 32)
        {
            const float b0 = fmaxf(fmaxf(sm[t], sm[t + 128]), fmaxf(sm[t + 64], sm[t + 192]));
            const float b1 = fmaxf(fmaxf(sm[t + 32], sm[t + 160]), fmaxf(sm[t + 96], sm[t + 224]));
            float x = fmaxf(b0, b1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
            if (t == 0) sm[0] = x;
        }
        __syncthreads();
        const float r = sm[0];
        __syncthreads();
        return r;
    }
    if (nt == 128)
    {
        if (t < 32)
        {
            float x = fmaxf(fmaxf(sm[t], sm[t + 64]), fmaxf(sm[t + 32], sm[t + 96]));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
            if (t == 0) sm[0] = x;
        }
        __syncthreads();
        const float r = sm[0];
        __syncthreads();
        return r;
    }
    for (int s = nt >> 1; s > 0; s >>= 1)
    {
        if (t < s) sm[t] = fmaxf(sm[t], sm[t + s]);
        __syncthreads();
    }
    const float r = sm[0];
    __syncthreads();
    return r;
}
// n <= V values per thread at once. sm [nt][V] shared. Bit-exact with the full binary tree.
template <int V>
__device__ __forceinline__ void blk_sum_v(float (&v)[V], int n, float* sm, int nt)
{
    const int t = threadIdx.x;
    for (int i = 0; i < n; ++i) sm[t * V + i] = v[i];
    __syncthreads();
    if (nt == 256)
    {
        const int warp = t >> 5, lane = t & 31;
        for (int i = warp; i < n; i += 8)
        {
            const float b0 = add_rn(add_rn(sm[lane * V + i], sm[(lane + 128) * V + i]), add_rn(sm[(lane + 64) * V + i], sm[(lane + 192) * V + i]));
            const float b1 = add_rn(add_rn(sm[(lane + 32) * V + i], sm[(lane + 160) * V + i]), add_rn(sm[(lane + 96) * V + i], sm[(lane + 224) * V + i]));
            float x = add_rn(b0, b1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x = add_rn(x, __shfl_down_sync(0xffffffffu, x, off));
            if (lane == 0) sm[i] = x;
        }
        __syncthreads();
        for (int i = 0; i < n; ++i) v[i] = sm[i];
        __syncthreads();
        return;
    }
    if (nt == 128)
    {
        if (t < 32)
        {
            for (int i = 0; i < n; ++i)
            {
                float x = add_rn(add_rn(sm[t * V + i], sm[(t + 64) * V + i]), add_rn(sm[(t + 32) * V + i], sm[(t + 96) * V + i]));
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) x = add_rn(x, __shfl_down_sync(0xffffffffu, x, off));
                if (t == 0) sm[i] = x;
            }
        }
        __syncthreads();
        for (int i = 0; i < n; ++i) v[i] = sm[i];
        __syncthreads();
        return;
    }
    for (int s = nt >> 1; s > 0; s >>= 1)
    {
        if (t < s)
            for (int i = 0; i < n; ++i) sm[t * V + i] = add_rn(sm[t * V + i], sm[(t + s) * V + i]);
        __syncthreads();
    }
    for (int i = 0; i < n; ++i) v[i] = sm[i];
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------- norms
constexpr int NRT = 256;   // threads of a CTA that owns one token row

// CTA = row. out may be x.
__global__ void __launch_bounds__(NRT) k_ink2_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, int H)
{
    __shared__ float sm[NRT];
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    float s = 0.0f;
    for (int c = threadIdx.x; c < H; c += NRT)
    {
        const float v = bf(x[off + c]);
        s = add_rn(s, mul_rn(v, v));
    }
    const float r = rsq(add_rn(div_rn(blk_sum(s, sm, NRT), static_cast<float>(H)), RMS_EPS));
    for (int c = threadIdx.x; c < H; c += NRT) out[off + c] = bf16(mul_rn(bf(w[c]), mul_rn(bf(x[off + c]), r)));
}

// CTA = token. toks [M]; embed [vocab][H]; w = embed_norm [H]; vis [M][H] or nullptr: the row of an image token is
// copied as it is.
__global__ void __launch_bounds__(NRT) k_ink2_embed_norm(const int* toks, const uint16_t* embed, const uint16_t* w, const uint16_t* vis,
                                                          int image_token, uint16_t* out, int H)
{
    __shared__ float sm[NRT];
    const int m = blockIdx.x, tok = toks[m];
    const size_t off = static_cast<size_t>(m) * H;
    if (vis != nullptr && tok == image_token)
    {
        for (int c = threadIdx.x; c < H; c += NRT) out[off + c] = vis[off + c];
        return;
    }
    const uint16_t* x = embed + static_cast<size_t>(tok) * H;
    float s = 0.0f;
    for (int c = threadIdx.x; c < H; c += NRT)
    {
        const float v = bf(x[c]);
        s = add_rn(s, mul_rn(v, v));
    }
    const float r = rsq(add_rn(div_rn(blk_sum(s, sm, NRT), static_cast<float>(H)), RMS_EPS));
    for (int c = threadIdx.x; c < H; c += NRT) out[off + c] = bf16(mul_rn(bf(w[c]), mul_rn(bf(x[c]), r)));
}

// ---------------------------------------------------------------------------------------------- depth attention
struct AttnResArgs
{
    const uint16_t* rows[MAX_DEPTH];   // D rows [M][H]: the finished blocks' states in block order, then the partial residual
    const float* query;                // [Q][H] fp32
    const uint16_t* norm[MAX_Q];       // [H]: the weight of the norm that follows query q; nullptr = out is the mixture
    uint16_t* out[MAX_Q];              // [M][H] each, none of them one of the rows
    int D, Q, H;
};

// CTA = token.
__global__ void __launch_bounds__(NRT) k_ink2_attn_res(AttnResArgs a)
{
    __shared__ float sm[NRT * MAX_DEPTH];
    const int t = threadIdx.x, H = a.H, D = a.D;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    float v[MAX_DEPTH], r[MAX_DEPTH];
    for (int d = 0; d < D; ++d)
    {
        const uint16_t* row = a.rows[d] + off;
        float s = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float x = bf(__ldg(row + c));
            s = add_rn(s, mul_rn(x, x));
        }
        v[d] = s;
    }
    blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
    for (int d = 0; d < D; ++d) r[d] = rsq(add_rn(div_rn(v[d], static_cast<float>(H)), RMS_EPS));
    const float root_h = sqrtf(static_cast<float>(H));
    if (a.Q == 1)
    {
        const float* qr = a.query;
        for (int d = 0; d < D; ++d)
        {
            const uint16_t* row = a.rows[d] + off;
            float s = 0.0f;
            for (int c = t; c < H; c += NRT) s = add_rn(s, mul_rn(__ldg(qr + c), mul_rn(bf(__ldg(row + c)), r[d])));
            v[d] = s;
        }
        blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
        float mx = -INFINITY, den = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            v[d] = div_rn(v[d], root_h);
            mx = fmaxf(mx, v[d]);
        }
        for (int d = 0; d < D; ++d)
        {
            v[d] = expf(v[d] - mx);
            den = add_rn(den, v[d]);
        }
        for (int d = 0; d < D; ++d) v[d] = div_rn(v[d], den);
        uint16_t* o = a.out[0] + off;
        float ss = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s = add_rn(s, mul_rn(v[d], mul_rn(bf(__ldg(a.rows[d] + off + c)), r[d])));
            const uint16_t cell = bf16(s);
            o[c] = cell;
            ss = add_rn(ss, mul_rn(bf(cell), bf(cell)));
        }
        if (a.norm[0] != nullptr)
        {
            const float rr = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
            const uint16_t* nw = a.norm[0];
            for (int c = t; c < H; c += NRT) o[c] = bf16(mul_rn(bf(__ldg(nw + c)), mul_rn(bf(o[c]), rr)));
        }
        return;
    }
    // Multi-query fused path (Q = 2..3): reads rows[d] ONCE for all Q dots and ONCE for all Q outputs (100% bit-exact!).
    float wq[MAX_Q][MAX_DEPTH];
    for (int d = 0; d < D; ++d)
    {
        const uint16_t* row = a.rows[d] + off;
        float sq[MAX_Q];
        #pragma unroll
        for (int q = 0; q < MAX_Q; ++q) sq[q] = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float u = mul_rn(bf(__ldg(row + c)), r[d]);
            for (int q = 0; q < a.Q; ++q)
                sq[q] = add_rn(sq[q], mul_rn(__ldg(a.query + static_cast<size_t>(q) * H + c), u));
        }
        for (int q = 0; q < a.Q; ++q) wq[q][d] = sq[q];
    }
    for (int q = 0; q < a.Q; ++q)
    {
        blk_sum_v<MAX_DEPTH>(wq[q], D, sm, NRT);
        float mx = -INFINITY, den = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            wq[q][d] = div_rn(wq[q][d], root_h);
            mx = fmaxf(mx, wq[q][d]);
        }
        for (int d = 0; d < D; ++d)
        {
            wq[q][d] = expf(wq[q][d] - mx);
            den = add_rn(den, wq[q][d]);
        }
        for (int d = 0; d < D; ++d) wq[q][d] = div_rn(wq[q][d], den);
    }
    float ss_q[MAX_Q];
    #pragma unroll
    for (int q = 0; q < MAX_Q; ++q) ss_q[q] = 0.0f;
    for (int c = t; c < H; c += NRT)
    {
        float acc[MAX_Q];
        #pragma unroll
        for (int q = 0; q < MAX_Q; ++q) acc[q] = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            const float u = mul_rn(bf(__ldg(a.rows[d] + off + c)), r[d]);
            for (int q = 0; q < a.Q; ++q) acc[q] = add_rn(acc[q], mul_rn(wq[q][d], u));
        }
        for (int q = 0; q < a.Q; ++q)
        {
            const uint16_t cell = bf16(acc[q]);
            a.out[q][off + c] = cell;
            ss_q[q] = add_rn(ss_q[q], mul_rn(bf(cell), bf(cell)));
        }
    }
    bool any_norm = false;
    for (int q = 0; q < a.Q; ++q)
        if (a.norm[q] != nullptr) any_norm = true;
    if (any_norm)
    {
        blk_sum_v<MAX_Q>(ss_q, a.Q, sm, NRT);
        for (int q = 0; q < a.Q; ++q)
        {
            if (a.norm[q] == nullptr) continue;
            const float rr = rsq(add_rn(div_rn(ss_q[q], static_cast<float>(H)), RMS_EPS));
            uint16_t* o = a.out[q] + off;
            const uint16_t* nw = a.norm[q];
            for (int c = t; c < H; c += NRT) o[c] = bf16(mul_rn(bf(__ldg(nw + c)), mul_rn(bf(o[c]), rr)));
        }
    }
}

// ---------------------------------------------------------------------------------------------- residual steps
// out = bf16(a + b); out may be a or b. A thread per cell.
__global__ void __launch_bounds__(256) k_ink2_add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] = bf16(add_rn(bf(a[i]), bf(b[i])));
}
__global__ void __launch_bounds__(256) k_ink2_copy_bf16(const uint16_t* a, uint16_t* out, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] = a[i];
}
// parts [R][n] fp32, one per rank -> out = bf16(parts[0] + parts[1] + ...), in rank order.
__global__ void __launch_bounds__(256) k_ink2_sum_ranks(const float* parts, int R, size_t n, uint16_t* out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    float s = parts[i];
    for (int r = 1; r < R; ++r) s = add_rn(s, parts[static_cast<size_t>(r) * n + i]);
    out[i] = bf16(s);
}

// ---------------------------------------------------------------------------------------------- GEMM
constexpr int GEMM_BF16 = 0, GEMM_F32 = 1, GEMM_SOFTCAP = 2;

INK2_HD u32 ld32(const uint16_t* p) { return *reinterpret_cast<const u32*>(p); }

// y [M][N] = x [M][K] W [N][K]^T. A warp owns 16 rows of W and NT tiles of 8 tokens; W is read once per 8 NT tokens.
// N need not divide by 16: the rows past N - 1 read row N - 1 and are not written. mode GEMM_SOFTCAP: row n is
// vocabulary id v0 + n, y = softcap30(z) below valid and -inf from there on.
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, int v0, int valid)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int n0 = (blockIdx.x * RW_CTA + (threadIdx.x >> 5)) * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* w0 = w + static_cast<size_t>(min(n0 + r, N - 1)) * K + 2 * c;
    const uint16_t* w1 = w + static_cast<size_t>(min(n0 + r + 8, N - 1)) * K + 2 * c;
    for (int k = 0; k < K; k += 16)
    {
        const u32 a0 = nv4::rd32(w0 + k), a1 = nv4::rd32(w1 + k), a2 = nv4::rd32(w0 + k + 8), a3 = nv4::rd32(w1 + k + 8);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + k), ld32(xr[t] + k + 8));
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const size_t o = static_cast<size_t>(m) * N + n;
            const float z = acc[t][j];
            if (mode == GEMM_BF16) static_cast<uint16_t*>(y)[o] = bf16(z);
            else if (mode == GEMM_F32) static_cast<float*>(y)[o] = z;
            else static_cast<float*>(y)[o] = v0 + n < valid ? softcap30(z) : -INFINITY;
        }
}

// Coalesced 128B-line Tensor Core BF16 GEMM (K % 64 == 0, 16B-aligned w):
// Reads 1 full 128B cache line per weight row via uint4 into zero-bank-conflict per-warp smem,
// then runs 4 back-to-back mma_bf16 steps in EXACT stock k_ink2_gemm order (100% bit-identical!).
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_c128(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                                                int M, int K, int N, void* __restrict__ y, int mode, int v0, int valid)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][36];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * RW_CTA + warp) * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = 0; k < K; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 4]) = v;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const size_t o = static_cast<size_t>(m) * N + n;
            const float z = acc[t][j];
            if (mode == GEMM_BF16) static_cast<uint16_t*>(y)[o] = bf16(z);
            else if (mode == GEMM_F32) static_cast<float*>(y)[o] = z;
            else static_cast<float*>(y)[o] = v0 + n < valid ? softcap30(z) : -INFINITY;
        }
}

// ---------------------------------------------------------------------------------------------- launches
inline unsigned cells_grid(size_t n) { return static_cast<unsigned>((n + 255) / 256); }

inline void rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, int M, int H, cudaStream_t st)
{
    k_ink2_rmsnorm<<<M, NRT, 0, st>>>(x, w, out, H);
}
inline void embed_norm(const int* toks, const uint16_t* embed, const uint16_t* w, const uint16_t* vis, int image_token, uint16_t* out, int M, int H, cudaStream_t st)
{
    k_ink2_embed_norm<<<M, NRT, 0, st>>>(toks, embed, w, vis, image_token, out, H);
}
inline void attn_res(const AttnResArgs& a, int M, cudaStream_t st)
{
    if (a.D < 1 || a.D > MAX_DEPTH || a.Q < 1 || a.Q > MAX_Q || M < 1) throw std::runtime_error("ink2::attn_res: D, Q or M");
    k_ink2_attn_res<<<M, NRT, 0, st>>>(a);
}
inline void add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n, cudaStream_t st)
{
    k_ink2_add_bf16<<<cells_grid(n), 256, 0, st>>>(a, b, out, n);
}
inline void copy_bf16(const uint16_t* a, uint16_t* out, size_t n, cudaStream_t st)
{
    k_ink2_copy_bf16<<<cells_grid(n), 256, 0, st>>>(a, out, n);
}
inline void sum_ranks(const float* parts, int R, size_t n, uint16_t* out, cudaStream_t st)
{
    k_ink2_sum_ranks<<<cells_grid(n), 256, 0, st>>>(parts, R, n, out);
}
// The residual after the attention of layer l, a = attn_sconv(attention): at the first layer of a block the layer
// input h becomes the next depth row (state, [M][H]) and mid = a; on the other layers mid = bf16(h + a).
inline void attn_block_step(const uint16_t* h, const uint16_t* a, bool block_start, uint16_t* state, uint16_t* mid, size_t n, cudaStream_t st)
{
    if (block_start)
    {
        copy_bf16(h, state, n, st);
        copy_bf16(a, mid, n, st);
    }
    else add_bf16(h, a, mid, n, st);
}
// A step-scoped replacement for the GEMM_BF16 / GEMM_F32 products (the engine sets it for a prefill chunk and
// clears it after; null = the kernels of this file). It returns false when it did not do the product.
using GemmHook = bool (*)(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, cudaStream_t st);
inline GemmHook& gemm_hook()
{
    static thread_local GemmHook hk = nullptr;
    return hk;
}
// Coalesced 128-bit BF16 GEMV (4 warps/CTA = 4 output rows/CTA, 100% batch-size invariant across all M).
// Every lane loads uint4 (8 BF16s = 16 B), 32 lanes = 512 B contiguous per load, identical K-sum order for M=1..MAX_M.
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink2_gemv_1warp(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                                         int M, int K, int N, void* __restrict__ y, int mode, int v0, int valid)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;
    const int m0 = blockIdx.y * MAX_M;
    const int m_rem = min(MAX_M, M - m0);
    if (m_rem <= 0) return;

    const int k8 = K >> 3;
    const uint4* wrow = reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * K);

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k8; i += 32)
    {
        const uint4 wv = __ldg(wrow + i);
        const float w0 = bf(static_cast<uint16_t>(wv.x)), w1 = bf(static_cast<uint16_t>(wv.x >> 16));
        const float w2 = bf(static_cast<uint16_t>(wv.y)), w3 = bf(static_cast<uint16_t>(wv.y >> 16));
        const float w4 = bf(static_cast<uint16_t>(wv.z)), w5 = bf(static_cast<uint16_t>(wv.z >> 16));
        const float w6 = bf(static_cast<uint16_t>(wv.w)), w7 = bf(static_cast<uint16_t>(wv.w >> 16));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m)
        {
            if (m < m_rem)
            {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m0 + m) * K) + i);
                float s = acc[m];
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.x)), w0));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.x >> 16)), w1));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.y)), w2));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.y >> 16)), w3));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.z)), w4));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.z >> 16)), w5));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.w)), w6));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.w >> 16)), w7));
                acc[m] = s;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m)
    {
        if (m < m_rem)
        {
            float z = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) z = add_rn(z, __shfl_xor_sync(0xffffffffu, z, off));
            if (lane == 0)
            {
                const size_t o = static_cast<size_t>(m0 + m) * N + n;
                if (mode == GEMM_BF16) static_cast<uint16_t*>(y)[o] = bf16(z);
                else if (mode == GEMM_F32) static_cast<float*>(y)[o] = z;
                else static_cast<float*>(y)[o] = v0 + n < valid ? softcap30(z) : -INFINITY;
            }
        }
    }
}

inline bool fast_gemv_enabled()
{
    static const bool on = []() {
        const char* e = std::getenv("INK2_FAST_GEMV");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

inline bool gemm_c128_enabled()
{
    static const bool on = []() {
        const char* e = std::getenv("INK2_GEMM_C128");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

inline bool fused_qkv_enabled()
{
    static const bool on = []() {
        const char* e = std::getenv("INK2_FUSED_QKV");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

struct GemmTarget
{
    const uint16_t* w;
    void* y;
    int N;
    int mode;
    int tile0;   // starting tile index in the combined grid
    int col0;    // starting output column in Split-K partial buffer
};

struct GemmMultiArgs
{
    GemmTarget t[6];
    int nt;
    int total_tiles;
    int total_n;
};

// Split-K 128B-coalesced Tensor Core BF16 GEMM (1 warp/CTA, grid (ceil(N/16), 1, SK)).
// Eliminates SM starvation on small/medium N (down_w N=64, wk/wv N=512, wo N=2304, etc.).
template <int NT>
__global__ void __launch_bounds__(32) k_ink2_gemm_c128_splitk(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                                              int M, int K, int N, int SK, float* __restrict__ partial)
{
    __shared__ alignas(16) uint32_t sm_w[16][36];
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int sk_id = blockIdx.z;
    const int K64 = K >> 6;
    const int k0 = ((sk_id * K64) / SK) << 6;
    const int k1 = (((sk_id + 1) * K64) / SK) << 6;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = k0; k < k1; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 4]) = v;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[r][p * 8 + c];
            const u32 a1 = sm_w[r + 8][p * 8 + c];
            const u32 a2 = sm_w[r][p * 8 + c + 4];
            const u32 a3 = sm_w[r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    float* pout = partial + static_cast<size_t>(sk_id) * M * N;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m < M && n < N) pout[static_cast<size_t>(m) * N + n] = acc[t][j];
        }
}

__global__ void __launch_bounds__(256) k_ink2_splitk_reduce(const float* __restrict__ partial, void* __restrict__ y,
                                                            int M, int N, int SK, int mode, int v0, int valid)
{
    const size_t mn = static_cast<size_t>(M) * N;
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= mn) return;
    float z = partial[i];
    for (int s = 1; s < SK; ++s) z += partial[static_cast<size_t>(s) * mn + i];
    if (mode == GEMM_BF16) static_cast<uint16_t*>(y)[i] = bf16(z);
    else if (mode == GEMM_F32) static_cast<float*>(y)[i] = z;
    else
    {
        const int n = static_cast<int>(i % N);
        static_cast<float*>(y)[i] = v0 + n < valid ? softcap30(z) : -INFINITY;
    }
}

// Split-K multi-target 128B-coalesced Tensor Core BF16 GEMM (1 warp/CTA, grid (total_tiles, 1, SK)).
template <int NT>
__global__ void __launch_bounds__(32) k_ink2_gemm_multi_c128_splitk(const uint16_t* __restrict__ x, GemmMultiArgs a,
                                                                    int M, int K, int SK, float* __restrict__ partial)
{
    __shared__ alignas(16) uint32_t sm_w[16][36];
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x;
    if (tile >= a.total_tiles) return;
    int idx = a.nt - 1;
    #pragma unroll
    for (int i = 0; i < 6; ++i)
    {
        if (i + 1 < a.nt && tile < a.t[i + 1].tile0) { idx = i; break; }
    }
    const GemmTarget tg = a.t[idx];
    const int n0 = (tile - tg.tile0) * 16;
    const int N = tg.N;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int sk_id = blockIdx.z;
    const int K64 = K >> 6;
    const int k0 = ((sk_id * K64) / SK) << 6;
    const int k1 = (((sk_id + 1) * K64) / SK) << 6;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = tg.w + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = k0; k < k1; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 4]) = v;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[r][p * 8 + c];
            const u32 a1 = sm_w[r + 8][p * 8 + c];
            const u32 a2 = sm_w[r][p * 8 + c + 4];
            const u32 a3 = sm_w[r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    float* pout = partial + static_cast<size_t>(sk_id) * M * a.total_n;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m < M && n < N) pout[static_cast<size_t>(m) * a.total_n + tg.col0 + n] = acc[t][j];
        }
}

__global__ void __launch_bounds__(256) k_ink2_splitk_reduce_multi(const float* __restrict__ partial, GemmMultiArgs a, int M, int SK)
{
    const size_t mn = static_cast<size_t>(M) * a.total_n;
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= mn) return;
    float z = partial[i];
    for (int s = 1; s < SK; ++s) z += partial[static_cast<size_t>(s) * mn + i];
    const int m = static_cast<int>(i / a.total_n);
    const int col = static_cast<int>(i % a.total_n);
    int idx = a.nt - 1;
    #pragma unroll
    for (int j = 0; j < 6; ++j)
    {
        if (j + 1 < a.nt && col < a.t[j + 1].col0) { idx = j; break; }
    }
    const GemmTarget tg = a.t[idx];
    const size_t o = static_cast<size_t>(m) * tg.N + (col - tg.col0);
    if (tg.mode == GEMM_BF16) static_cast<uint16_t*>(tg.y)[o] = bf16(z);
    else static_cast<float*>(tg.y)[o] = z;
}

// Per-row symmetric INT8 quantizer for drafter-only lm_head & fold_w (4 warps/CTA = 4 rows/CTA).
__global__ void __launch_bounds__(128) k_ink2_quant_row_i8(const uint16_t* __restrict__ w, int8_t* __restrict__ w_i8,
                                                           float* __restrict__ scales, int N, int K)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;
    const uint16_t* row = w + static_cast<size_t>(n) * K;
    int8_t* dst = w_i8 + static_cast<size_t>(n) * K;
    float amax = 0.0f;
    for (int k = lane; k < K; k += 32)
    {
        const float v = fabsf(bf(row[k]));
        if (v > amax) amax = v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float sc = amax > 0.0f ? (amax / 127.0f) : 1.0f;
    const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
    if (lane == 0) scales[n] = sc;
    for (int k = lane; k < K; k += 32)
    {
        const float q = fminf(fmaxf(bf(row[k]) * inv, -127.0f), 127.0f);
        dst[k] = static_cast<int8_t>(__float2int_rn(q));
    }
}

__device__ __forceinline__ u32 i8x2_to_bf16x2(int a, int b)
{
    return static_cast<u32>(bf16(static_cast<float>(a))) | (static_cast<u32>(bf16(static_cast<float>(b))) << 16);
}

__device__ __forceinline__ u32 i8x2_lut(const uint16_t* lut, u32 pair16)
{
    return static_cast<u32>(lut[pair16 & 0xffu]) | (static_cast<u32>(lut[(pair16 >> 8) & 0xffu]) << 16);
}

// Fast 128B-line uint4 + smem-LUT INT8 Tensor Core GEMM (K % 128 == 0).
// Drafter-only (mtp_body + mtp_base_lm); trunk always uses 100% exact BF16 weights.
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_c128_i8_k128(const uint16_t* __restrict__ x, const int8_t* __restrict__ w_i8,
                                                                        const float* __restrict__ scales, int M, int K, int N, uint16_t* __restrict__ y)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][68];
    __shared__ uint16_t s_lut[256];
    for (int i = threadIdx.x; i < 256; i += RW_CTA * 32)
        s_lut[i] = bf16(static_cast<float>(static_cast<int8_t>(i)));
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * RW_CTA + warp) * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const int8_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w_i8 + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = 0; k < K; k += 128)
    {
        uint4 v16[4];
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
            v16[rr] = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            uint4 bv0, bv1;
            bv0.x = i8x2_lut(s_lut, v16[rr].x);
            bv0.y = i8x2_lut(s_lut, v16[rr].x >> 16);
            bv0.z = i8x2_lut(s_lut, v16[rr].y);
            bv0.w = i8x2_lut(s_lut, v16[rr].y >> 16);
            bv1.x = i8x2_lut(s_lut, v16[rr].z);
            bv1.y = i8x2_lut(s_lut, v16[rr].z >> 16);
            bv1.z = i8x2_lut(s_lut, v16[rr].w);
            bv1.w = i8x2_lut(s_lut, v16[rr].w >> 16);
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 8]) = bv0;
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 8 + 4]) = bv1;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    const float sc0 = __ldg(scales + min(n0 + r, N - 1));
    const float sc1 = __ldg(scales + min(n0 + r + 8, N - 1));
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const float sc = (j < 2) ? sc0 : sc1;
            y[static_cast<size_t>(m) * N + n] = bf16(acc[t][j] * sc);
        }
}

// Split-K 128B-line uint4 + smem-LUT INT8 Tensor Core GEMM (1 warp/CTA, K % 128 == 0, drafter-only).
template <int NT>
__global__ void __launch_bounds__(32) k_ink2_gemm_c128_i8_splitk(const uint16_t* __restrict__ x, const int8_t* __restrict__ w_i8,
                                                                 int M, int K, int N, int SK, float* __restrict__ partial)
{
    __shared__ alignas(16) uint32_t sm_w[16][68];
    __shared__ uint16_t s_lut[256];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = lane; i < 256; i += 32)
        s_lut[i] = bf16(static_cast<float>(static_cast<int8_t>(i)));
    __syncwarp();

    const int n0 = blockIdx.x * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int sk_id = blockIdx.z;
    const int K128 = K >> 7;
    const int k0 = ((sk_id * K128) / SK) << 7;
    const int k1 = (((sk_id + 1) * K128) / SK) << 7;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const int8_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w_i8 + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = k0; k < k1; k += 128)
    {
        uint4 v16[4];
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
            v16[rr] = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            uint4 bv0, bv1;
            bv0.x = i8x2_lut(s_lut, v16[rr].x);
            bv0.y = i8x2_lut(s_lut, v16[rr].x >> 16);
            bv0.z = i8x2_lut(s_lut, v16[rr].y);
            bv0.w = i8x2_lut(s_lut, v16[rr].y >> 16);
            bv1.x = i8x2_lut(s_lut, v16[rr].z);
            bv1.y = i8x2_lut(s_lut, v16[rr].z >> 16);
            bv1.z = i8x2_lut(s_lut, v16[rr].w);
            bv1.w = i8x2_lut(s_lut, v16[rr].w >> 16);
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 8]) = bv0;
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 8 + 4]) = bv1;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[r][p * 8 + c];
            const u32 a1 = sm_w[r + 8][p * 8 + c];
            const u32 a2 = sm_w[r][p * 8 + c + 4];
            const u32 a3 = sm_w[r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    float* pout = partial + static_cast<size_t>(sk_id) * M * N;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m < M && n < N) pout[static_cast<size_t>(m) * N + n] = acc[t][j];
        }
}

__global__ void __launch_bounds__(256) k_ink2_splitk_reduce_i8(const float* __restrict__ partial, const float* __restrict__ scales,
                                                               uint16_t* __restrict__ y, int M, int N, int SK)
{
    const size_t mn = static_cast<size_t>(M) * N;
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= mn) return;
    float z = partial[i];
    for (int s = 1; s < SK; ++s) z += partial[static_cast<size_t>(s) * mn + i];
    const int n = static_cast<int>(i % N);
    y[i] = bf16(z * __ldg(scales + n));
}

// Multi-target 128B-line uint4 + smem-LUT INT8 Tensor Core GEMM (drafter-only: mtp_body qkv & gate/up).
struct GemmTargetI8
{
    const int8_t* w_i8;
    const float* scales;
    uint16_t* y;
    int N;
    int tile0;
};

struct GemmMultiI8Args
{
    GemmTargetI8 t[4];
    int nt;
    int total_tiles;
};

template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_multi_c128_i8_k128(const uint16_t* __restrict__ x, GemmMultiI8Args a, int M, int K)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][68];
    __shared__ uint16_t s_lut[256];
    for (int i = threadIdx.x; i < 256; i += RW_CTA * 32)
        s_lut[i] = bf16(static_cast<float>(static_cast<int8_t>(i)));
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = blockIdx.x * RW_CTA + warp;
    if (tile >= a.total_tiles) return;
    int idx = a.nt - 1;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (i + 1 < a.nt && tile < a.t[i + 1].tile0) { idx = i; break; }
    }
    const GemmTargetI8 tg = a.t[idx];
    const int n0 = (tile - tg.tile0) * 16;
    const int N = tg.N;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const int8_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = tg.w_i8 + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = 0; k < K; k += 128)
    {
        uint4 v16[4];
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
            v16[rr] = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            uint4 bv0, bv1;
            bv0.x = i8x2_lut(s_lut, v16[rr].x);
            bv0.y = i8x2_lut(s_lut, v16[rr].x >> 16);
            bv0.z = i8x2_lut(s_lut, v16[rr].y);
            bv0.w = i8x2_lut(s_lut, v16[rr].y >> 16);
            bv1.x = i8x2_lut(s_lut, v16[rr].z);
            bv1.y = i8x2_lut(s_lut, v16[rr].z >> 16);
            bv1.z = i8x2_lut(s_lut, v16[rr].w);
            bv1.w = i8x2_lut(s_lut, v16[rr].w >> 16);
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 8]) = bv0;
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 8 + 4]) = bv1;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    const float sc0 = __ldg(tg.scales + min(n0 + r, N - 1));
    const float sc1 = __ldg(tg.scales + min(n0 + r + 8, N - 1));
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const float sc = (j < 2) ? sc0 : sc1;
            tg.y[static_cast<size_t>(m) * N + n] = bf16(acc[t][j] * sc);
        }
}

// Drafter-only INT8 per-row 64B-coalesced Tensor Core GEMM (fallback when K % 128 != 0).
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_c128_i8(const uint16_t* __restrict__ x, const int8_t* __restrict__ w_i8,
                                                                   const float* __restrict__ scales, int M, int K, int N, uint16_t* __restrict__ y)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][36];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * RW_CTA + warp) * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const int8_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w_i8 + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = 0; k < K; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint2 v8 = __ldg(reinterpret_cast<const uint2*>(wr[rr] + k) + lc);
            uint4 bv;
            bv.x = i8x2_to_bf16x2(static_cast<int8_t>(v8.x & 0xffu), static_cast<int8_t>((v8.x >> 8) & 0xffu));
            bv.y = i8x2_to_bf16x2(static_cast<int8_t>((v8.x >> 16) & 0xffu), static_cast<int8_t>((v8.x >> 24) & 0xffu));
            bv.z = i8x2_to_bf16x2(static_cast<int8_t>(v8.y & 0xffu), static_cast<int8_t>((v8.y >> 8) & 0xffu));
            bv.w = i8x2_to_bf16x2(static_cast<int8_t>((v8.y >> 16) & 0xffu), static_cast<int8_t>((v8.y >> 24) & 0xffu));
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 4]) = bv;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    const float sc0 = __ldg(scales + min(n0 + r, N - 1));
    const float sc1 = __ldg(scales + min(n0 + r + 8, N - 1));
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const float sc = (j < 2) ? sc0 : sc1;
            y[static_cast<size_t>(m) * N + n] = bf16(acc[t][j] * sc);
        }
}

inline void gemm_bf16_i8(const uint16_t* x, const int8_t* w_i8, const float* scales, int M, int K, int N, uint16_t* y, cudaStream_t st)
{
    if ((K % 128 == 0) && ((reinterpret_cast<uintptr_t>(w_i8) & 15u) == 0))
    {
        if (M <= 64 && N <= 4096)
        {
            const unsigned gx1 = static_cast<unsigned>((N + 15) / 16);
            int sk = pick_splitk(static_cast<int>(gx1), K >> 1, M, static_cast<size_t>(N));
            if (sk > (K >> 7)) sk = (K >> 7);
            if (sk > 1)
            {
                float* pbuf = splitk_buf();
                if (M <= 8) k_ink2_gemm_c128_i8_splitk<1><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w_i8, M, K, N, sk, pbuf);
                else if (M <= 16) k_ink2_gemm_c128_i8_splitk<2><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w_i8, M, K, N, sk, pbuf);
                else if (M <= 32) k_ink2_gemm_c128_i8_splitk<4><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w_i8, M, K, N, sk, pbuf);
                else k_ink2_gemm_c128_i8_splitk<8><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w_i8, M, K, N, sk, pbuf);
                k_ink2_splitk_reduce_i8<<<cells_grid(static_cast<size_t>(M) * N), 256, 0, st>>>(pbuf, scales, y, M, N, sk);
                return;
            }
            if (M <= 8) k_ink2_gemm_c128_i8_k128<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
            else if (M <= 16) k_ink2_gemm_c128_i8_k128<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
            else if (M <= 32) k_ink2_gemm_c128_i8_k128<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
            else k_ink2_gemm_c128_i8_k128<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
            return;
        }
        const unsigned gx = static_cast<unsigned>((N + 16 * RW - 1) / (16 * RW));
        if (M <= 8) k_ink2_gemm_c128_i8_k128<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
        else if (M <= 16) k_ink2_gemm_c128_i8_k128<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
        else if (M <= 32) k_ink2_gemm_c128_i8_k128<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
        else k_ink2_gemm_c128_i8_k128<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
        return;
    }
    const unsigned gx = static_cast<unsigned>((N + 16 * RW - 1) / (16 * RW));
    if (M <= 8) k_ink2_gemm_c128_i8<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
    else if (M <= 16) k_ink2_gemm_c128_i8<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
    else if (M <= 32) k_ink2_gemm_c128_i8<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
    else k_ink2_gemm_c128_i8<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, w_i8, scales, M, K, N, y);
}

inline void gemm_multi_bf16_i8(const uint16_t* x, const GemmTargetI8* targets, int nt, int M, int K, cudaStream_t st)
{
    GemmMultiI8Args a{};
    a.nt = nt;
    int cur = 0;
    for (int i = 0; i < nt; ++i)
    {
        a.t[i] = targets[i];
        a.t[i].tile0 = cur;
        cur += (targets[i].N + 15) / 16;
    }
    a.total_tiles = cur;
    if (M <= 64 && cur <= 512)
    {
        const unsigned gx1 = static_cast<unsigned>(cur);
        if (M <= 8) k_ink2_gemm_multi_c128_i8_k128<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else if (M <= 16) k_ink2_gemm_multi_c128_i8_k128<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else if (M <= 32) k_ink2_gemm_multi_c128_i8_k128<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else k_ink2_gemm_multi_c128_i8_k128<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        return;
    }
    const unsigned gx = static_cast<unsigned>((cur + RW - 1) / RW);
    if (M <= 8) k_ink2_gemm_multi_c128_i8_k128<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else if (M <= 16) k_ink2_gemm_multi_c128_i8_k128<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else if (M <= 32) k_ink2_gemm_multi_c128_i8_k128<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else k_ink2_gemm_multi_c128_i8_k128<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, a, M, K);
}

// Fused multi-target Tensor Core BF16 GEMM sharing input x: 100% bit-identical to separate k_ink2_gemm calls.
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_multi(const uint16_t* __restrict__ x, GemmMultiArgs a, int M, int K)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int tile = blockIdx.x * RW_CTA + (threadIdx.x >> 5);
    if (tile >= a.total_tiles) return;
    int idx = a.nt - 1;
    #pragma unroll
    for (int i = 0; i < 6; ++i)
    {
        if (i + 1 < a.nt && tile < a.t[i + 1].tile0) { idx = i; break; }
    }
    const GemmTarget tg = a.t[idx];
    const int n0 = (tile - tg.tile0) * 16;
    const int N = tg.N;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* w0 = tg.w + static_cast<size_t>(min(n0 + r, N - 1)) * K + 2 * c;
    const uint16_t* w1 = tg.w + static_cast<size_t>(min(n0 + r + 8, N - 1)) * K + 2 * c;
    for (int k = 0; k < K; k += 16)
    {
        const u32 a0 = nv4::rd32(w0 + k), a1 = nv4::rd32(w1 + k), a2 = nv4::rd32(w0 + k + 8), a3 = nv4::rd32(w1 + k + 8);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + k), ld32(xr[t] + k + 8));
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const size_t o = static_cast<size_t>(m) * N + n;
            const float z = acc[t][j];
            if (tg.mode == GEMM_BF16) static_cast<uint16_t*>(tg.y)[o] = bf16(z);
            else static_cast<float*>(tg.y)[o] = z;
        }
}

// Fused multi-target 128B-coalesced Tensor Core BF16 GEMM sharing input x: 100% bit-identical to stock k_ink2_gemm.
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_ink2_gemm_multi_c128(const uint16_t* __restrict__ x, GemmMultiArgs a, int M, int K)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][36];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = blockIdx.x * RW_CTA + warp;
    if (tile >= a.total_tiles) return;
    int idx = a.nt - 1;
    #pragma unroll
    for (int i = 0; i < 6; ++i)
    {
        if (i + 1 < a.nt && tile < a.t[i + 1].tile0) { idx = i; break; }
    }
    const GemmTarget tg = a.t[idx];
    const int n0 = (tile - tg.tile0) * 16;
    const int N = tg.N;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = tg.w + static_cast<size_t>(min(n0 + rr * 4 + lg, N - 1)) * K;
    for (int k = 0; k < K; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 4]) = v;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m >= M || n >= N) continue;
            const size_t o = static_cast<size_t>(m) * N + n;
            const float z = acc[t][j];
            if (tg.mode == GEMM_BF16) static_cast<uint16_t*>(tg.y)[o] = bf16(z);
            else static_cast<float*>(tg.y)[o] = z;
        }
}

// Fused multi-target coalesced 128-bit GEMV sharing input x: 100% bit-identical to separate k_ink2_gemv_1warp calls.
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink2_gemv_1warp_multi(const uint16_t* __restrict__ x, GemmMultiArgs a, int M, int K)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_g = blockIdx.x * 4 + warp;
    if (row_g >= a.total_tiles) return;
    int idx = a.nt - 1;
    #pragma unroll
    for (int i = 0; i < 6; ++i)
    {
        if (i + 1 < a.nt && row_g < a.t[i + 1].tile0) { idx = i; break; }
    }
    const GemmTarget tg = a.t[idx];
    const int n = row_g - tg.tile0;
    const int N = tg.N;
    if (n >= N) return;
    const int m0 = blockIdx.y * MAX_M;
    const int m_rem = min(MAX_M, M - m0);
    if (m_rem <= 0) return;

    const int k8 = K >> 3;
    const uint4* wrow = reinterpret_cast<const uint4*>(tg.w + static_cast<size_t>(n) * K);

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k8; i += 32)
    {
        const uint4 wv = __ldg(wrow + i);
        const float w0 = bf(static_cast<uint16_t>(wv.x)), w1 = bf(static_cast<uint16_t>(wv.x >> 16));
        const float w2 = bf(static_cast<uint16_t>(wv.y)), w3 = bf(static_cast<uint16_t>(wv.y >> 16));
        const float w4 = bf(static_cast<uint16_t>(wv.z)), w5 = bf(static_cast<uint16_t>(wv.z >> 16));
        const float w6 = bf(static_cast<uint16_t>(wv.w)), w7 = bf(static_cast<uint16_t>(wv.w >> 16));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m)
        {
            if (m < m_rem)
            {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m0 + m) * K) + i);
                float s = acc[m];
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.x)), w0));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.x >> 16)), w1));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.y)), w2));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.y >> 16)), w3));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.z)), w4));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.z >> 16)), w5));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.w)), w6));
                s = add_rn(s, mul_rn(bf(static_cast<uint16_t>(xv.w >> 16)), w7));
                acc[m] = s;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m)
    {
        if (m < m_rem)
        {
            float z = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) z = add_rn(z, __shfl_xor_sync(0xffffffffu, z, off));
            if (lane == 0)
            {
                const size_t o = static_cast<size_t>(m0 + m) * N + n;
                if (tg.mode == GEMM_BF16) static_cast<uint16_t*>(tg.y)[o] = bf16(z);
                else static_cast<float*>(tg.y)[o] = z;
            }
        }
    }
}

// Set once at runtime startup; standalone kernel users retain the original dispatch.
inline bool& narrow_mma_cta()
{
    static bool enabled = false;
    return enabled;
}
inline void gemm(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, int v0, int valid, cudaStream_t st)
{
    if (M < 1 || N < 1 || K < 16 || K % 16) throw std::runtime_error("ink2::gemm: M < 1, N < 1 or K % 16");
    if (mode != GEMM_SOFTCAP)
        if (GemmHook hk = gemm_hook())
            if (hk(x, w, M, K, N, y, mode, st)) return;
    // Keep each warp's m16n8k16 operand and accumulation sequence; spread small
    // grids across more SMs by scheduling one independent output tile per CTA.
    if (narrow_mma_cta() && M <= 56 && N <= 4096 && N > 16) {
        const unsigned nx = (N + 15) / 16;
        if (M <= 8) k_ink2_gemm<1,1><<<dim3(nx,1),32,0,st>>>(x,w,M,K,N,y,mode,v0,valid);
        else k_ink2_gemm<4,1><<<dim3(nx,(M+31)/32),32,0,st>>>(x,w,M,K,N,y,mode,v0,valid);
        return;
    }
    if (fast_gemv_enabled() && M <= 64 && (((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 15u) == 0))
    {
        const unsigned gx = static_cast<unsigned>((N + 3) / 4);
        if (M == 1) k_ink2_gemv_1warp<1><<<dim3(gx, 1), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 4) k_ink2_gemv_1warp<4><<<dim3(gx, 1), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 8) k_ink2_gemv_1warp<8><<<dim3(gx, 1), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 16) k_ink2_gemv_1warp<16><<<dim3(gx, 1), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 32) k_ink2_gemv_1warp<32><<<dim3(gx, 1), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else k_ink2_gemv_1warp<32><<<dim3(gx, (M + 31) / 32), 128, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        return;
    }
    const bool use_c128 = gemm_c128_enabled() && (K % 64 == 0) && ((reinterpret_cast<uintptr_t>(w) & 15u) == 0);
    // Small-N / small-M exact Tensor Core dispatch: 1 warp/CTA + deterministic Split-K fills all SMs for small N
    // (e.g. down_w N=64, wk/wv N=512, wo N=2304) while remaining 100% batch-size invariant across M <= 64!
    if (M <= 64 && N <= 4096)
    {
        const unsigned gx1 = static_cast<unsigned>((N + 15) / 16);
        if (use_c128)
        {
            const int sk = pick_splitk(static_cast<int>(gx1), K, M, static_cast<size_t>(N));
            if (sk > 1)
            {
                float* pbuf = splitk_buf();
                if (M <= 8) k_ink2_gemm_c128_splitk<1><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w, M, K, N, sk, pbuf);
                else if (M <= 16) k_ink2_gemm_c128_splitk<2><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w, M, K, N, sk, pbuf);
                else if (M <= 32) k_ink2_gemm_c128_splitk<4><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w, M, K, N, sk, pbuf);
                else k_ink2_gemm_c128_splitk<8><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, w, M, K, N, sk, pbuf);
                k_ink2_splitk_reduce<<<cells_grid(static_cast<size_t>(M) * N), 256, 0, st>>>(pbuf, y, M, N, sk, mode, v0, valid);
                return;
            }
            if (M <= 8) k_ink2_gemm_c128<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
            else if (M <= 16) k_ink2_gemm_c128<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
            else if (M <= 32) k_ink2_gemm_c128<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
            else k_ink2_gemm_c128<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
            return;
        }
        if (M <= 8) k_ink2_gemm<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 16) k_ink2_gemm<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 32) k_ink2_gemm<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else k_ink2_gemm<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        return;
    }
    const unsigned gx = static_cast<unsigned>((N + 16 * RW - 1) / (16 * RW));
    if (use_c128)
    {
        if (M <= 8) k_ink2_gemm_c128<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 16) k_ink2_gemm_c128<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else if (M <= 32) k_ink2_gemm_c128<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        else k_ink2_gemm_c128<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
        return;
    }
    if (M <= 8) k_ink2_gemm<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
    else if (M <= 16) k_ink2_gemm<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
    else if (M <= 32) k_ink2_gemm<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
    else k_ink2_gemm<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
}

inline void gemm_multi(const uint16_t* x, const GemmTarget* targets, int nt, int M, int K, cudaStream_t st)
{
    if (nt <= 0) return;
    if (!fused_qkv_enabled() || gemm_hook() != nullptr || nt == 1)
    {
        for (int i = 0; i < nt; ++i) gemm(x, targets[i].w, M, K, targets[i].N, targets[i].y, targets[i].mode, 0, 0, st);
        return;
    }
    bool aligned = (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
    for (int i = 0; i < nt; ++i)
        if ((reinterpret_cast<uintptr_t>(targets[i].w) & 15u) != 0) aligned = false;
    GemmMultiArgs a{};
    a.nt = nt;
    if (fast_gemv_enabled() && M <= 64 && aligned)
    {
        int cur = 0;
        for (int i = 0; i < nt; ++i)
        {
            a.t[i] = targets[i];
            a.t[i].tile0 = cur;
            cur += targets[i].N;
        }
        a.total_tiles = cur;
        const unsigned gx = static_cast<unsigned>((cur + 3) / 4);
        if (M == 1) k_ink2_gemv_1warp_multi<1><<<dim3(gx, 1), 128, 0, st>>>(x, a, M, K);
        else if (M <= 4) k_ink2_gemv_1warp_multi<4><<<dim3(gx, 1), 128, 0, st>>>(x, a, M, K);
        else if (M <= 8) k_ink2_gemv_1warp_multi<8><<<dim3(gx, 1), 128, 0, st>>>(x, a, M, K);
        else if (M <= 16) k_ink2_gemv_1warp_multi<16><<<dim3(gx, 1), 128, 0, st>>>(x, a, M, K);
        else if (M <= 32) k_ink2_gemv_1warp_multi<32><<<dim3(gx, 1), 128, 0, st>>>(x, a, M, K);
        else k_ink2_gemv_1warp_multi<32><<<dim3(gx, (M + 31) / 32), 128, 0, st>>>(x, a, M, K);
        return;
    }
    int cur = 0, cur_n = 0;
    for (int i = 0; i < nt; ++i)
    {
        a.t[i] = targets[i];
        a.t[i].tile0 = cur;
        a.t[i].col0 = cur_n;
        cur += (targets[i].N + 15) / 16;
        cur_n += targets[i].N;
    }
    a.total_tiles = cur;
    a.total_n = cur_n;
    const bool use_c128 = gemm_c128_enabled() && (K % 64 == 0) && aligned;
    if (M <= 64 && cur <= 512)
    {
        const unsigned gx1 = static_cast<unsigned>(cur);
        if (use_c128)
        {
            const int sk = pick_splitk(cur, K, M, static_cast<size_t>(cur_n));
            if (sk > 1)
            {
                float* pbuf = splitk_buf();
                if (M <= 8) k_ink2_gemm_multi_c128_splitk<1><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, a, M, K, sk, pbuf);
                else if (M <= 16) k_ink2_gemm_multi_c128_splitk<2><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, a, M, K, sk, pbuf);
                else if (M <= 32) k_ink2_gemm_multi_c128_splitk<4><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, a, M, K, sk, pbuf);
                else k_ink2_gemm_multi_c128_splitk<8><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, a, M, K, sk, pbuf);
                k_ink2_splitk_reduce_multi<<<cells_grid(static_cast<size_t>(M) * cur_n), 256, 0, st>>>(pbuf, a, M, sk);
                return;
            }
            if (M <= 8) k_ink2_gemm_multi_c128<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
            else if (M <= 16) k_ink2_gemm_multi_c128<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
            else if (M <= 32) k_ink2_gemm_multi_c128<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
            else k_ink2_gemm_multi_c128<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
            return;
        }
        if (M <= 8) k_ink2_gemm_multi<1, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else if (M <= 16) k_ink2_gemm_multi<2, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else if (M <= 32) k_ink2_gemm_multi<4, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        else k_ink2_gemm_multi<8, 1><<<dim3(gx1, 1), 32, 0, st>>>(x, a, M, K);
        return;
    }
    const unsigned gx = static_cast<unsigned>((cur + RW - 1) / RW);
    if (use_c128)
    {
        if (M <= 8) k_ink2_gemm_multi_c128<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
        else if (M <= 16) k_ink2_gemm_multi_c128<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
        else if (M <= 32) k_ink2_gemm_multi_c128<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
        else k_ink2_gemm_multi_c128<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, a, M, K);
        return;
    }
    if (M <= 8) k_ink2_gemm_multi<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else if (M <= 16) k_ink2_gemm_multi<2><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else if (M <= 32) k_ink2_gemm_multi<4><<<dim3(gx, 1), RW * 32, 0, st>>>(x, a, M, K);
    else k_ink2_gemm_multi<8><<<dim3(gx, (M + 63) / 64), RW * 32, 0, st>>>(x, a, M, K);
}
inline void gemm_bf16(const uint16_t* x, const uint16_t* w, int M, int K, int N, uint16_t* y, cudaStream_t st) { gemm(x, w, M, K, N, y, GEMM_BF16, 0, 0, st); }
inline void gemm_f32(const uint16_t* x, const uint16_t* w, int M, int K, int N, float* y, cudaStream_t st) { gemm(x, w, M, K, N, y, GEMM_F32, 0, 0, st); }
// logits [M][rows] of one vocabulary slice: lm_head rows v0 .. v0 + rows, hf [M][H] = final_norm's output.
inline void lm_head(const uint16_t* hf, const uint16_t* w, int M, int H, int rows, int v0, int valid_vocab, float* logits, cudaStream_t st)
{
    gemm(hf, w, M, H, rows, logits, GEMM_SOFTCAP, v0, valid_vocab, st);
}
#endif   // __CUDACC__
}   // namespace ink2
