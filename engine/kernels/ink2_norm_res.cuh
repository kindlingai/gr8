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
// thread. sm [nt] shared.
__device__ __forceinline__ float blk_sum(float v, float* sm, int nt)
{
    const int t = threadIdx.x;
    sm[t] = v;
    __syncthreads();
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
    for (int s = nt >> 1; s > 0; s >>= 1)
    {
        if (t < s) sm[t] = fmaxf(sm[t], sm[t + s]);
        __syncthreads();
    }
    const float r = sm[0];
    __syncthreads();
    return r;
}
// n <= V values per thread at once. sm [nt][V] shared.
template <int V>
__device__ __forceinline__ void blk_sum_v(float (&v)[V], int n, float* sm, int nt)
{
    const int t = threadIdx.x;
    for (int i = 0; i < n; ++i) sm[t * V + i] = v[i];
    __syncthreads();
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
        float s = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float x = bf(a.rows[d][off + c]);
            s = add_rn(s, mul_rn(x, x));
        }
        v[d] = s;
    }
    blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
    for (int d = 0; d < D; ++d) r[d] = rsq(add_rn(div_rn(v[d], static_cast<float>(H)), RMS_EPS));
    const float root_h = sqrtf(static_cast<float>(H));
    for (int q = 0; q < a.Q; ++q)
    {
        const float* qr = a.query + static_cast<size_t>(q) * H;
        for (int d = 0; d < D; ++d)
        {
            float s = 0.0f;
            for (int c = t; c < H; c += NRT) s = add_rn(s, mul_rn(qr[c], mul_rn(bf(a.rows[d][off + c]), r[d])));
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
        uint16_t* o = a.out[q] + off;
        float ss = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s = add_rn(s, mul_rn(v[d], mul_rn(bf(a.rows[d][off + c]), r[d])));
            const uint16_t cell = bf16(s);
            o[c] = cell;
            ss = add_rn(ss, mul_rn(bf(cell), bf(cell)));
        }
        if (a.norm[q] == nullptr) continue;
        // the norm reads the rounded mixture
        const float rr = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
        for (int c = t; c < H; c += NRT) o[c] = bf16(mul_rn(bf(a.norm[q][c]), mul_rn(bf(o[c]), rr)));
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
template <int NT>
__global__ void __launch_bounds__(RW * 32) k_ink2_gemm(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, int v0, int valid)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int n0 = (blockIdx.x * RW + (threadIdx.x >> 5)) * 16;
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
inline void gemm(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, int v0, int valid, cudaStream_t st)
{
    if (M < 1 || N < 1 || K < 16 || K % 16) throw std::runtime_error("ink2::gemm: M < 1, N < 1 or K % 16");
    const unsigned gx = static_cast<unsigned>((N + 16 * RW - 1) / (16 * RW));
    if (M <= 8) k_ink2_gemm<1><<<dim3(gx, 1), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
    else k_ink2_gemm<4><<<dim3(gx, (M + 31) / 32), RW * 32, 0, st>>>(x, w, M, K, N, y, mode, v0, valid);
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
