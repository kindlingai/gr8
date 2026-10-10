// Dense trellis matrices (#116, dtr.cuh) on the prefill path: the decoded tiles as an FP8 copy on today's MXFP8 GEMMs
// (bind_prefill.cuh), wrapped in their rotations, as the trellis shared expert runs there (#65).
//
//   make_fp8     once per matrix at load: T (still rotated) as E4M3 codes [N][K] + E8M0 scales [N][K/32] all 127, the
//                bindp::Gemm weight layout. The codebook values are E4M3 already, so the copy is exact.
//   k_dtr_rotq   x [M][ldx] -> MXFP8 records of H128(x * suh) (the k_xq8 layout, quantized from fp32 as k_shx_rotq)
//   GEMM         bindp::Gemm on those records -> y' fp16 [M][N]
//   k_dtr_post   y = H128(y') * svh -> fp16 or BF16 [M][ldo]
//   k_dtr_pmid   mlp: g', u' fp16 [M][I] -> MXFP8 records of the down input H128(f16(silu(g) u) * suh_d),
//                g = f16(H128(g') * svh_g), u likewise (the arithmetic of k_shx_mid with per-channel vectors)
// Step order for one matrix: rotq, GEMM, post. MLP: rotq with gate's suh (a second rotq when up's differs), gate + up GEMMs
// (launch_pair), pmid, down GEMM, post.
#pragma once
#include "dtr.cuh"
#include "bind_prefill.cuh"

namespace dtr
{

// grid ceil(N * K / 256), block 256: one thread per weight (n, k), k fastest
__global__ void __launch_bounds__(256) k_dtr_fp8(const uint8_t* w, long long sB, long long sG, int K, int N, uint8_t* codes)
{
    constexpr int TB = 32 * BITS;
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(N) * K) return;
    const int n = static_cast<int>(i / K), k = static_cast<int>(i % K);
    const int s = k >> 4, kk = k & 15, c = (n & 127) >> 4, nn = n & 15;
    const uint8_t* p = w + (n >> 7) * sB + (s >> 3) * sG + ((s & 7) * 8 + c) * TB;
    // tile element (kk, nn) is stream weight t = lane * 8 + j of the mma A fragment order (tr3.cuh Tr3Lane)
    const int lane = (nn & 7) * 4 + ((kk & 7) >> 1), jj = (kk & 1) | (((kk >> 3) & 1) << 1) | ((nn >> 3) << 2), t = lane * 8 + jj;
    // state of weight t: the 16 stream bits ending at bit (t + 1) K, MSB first over LE u32 words, tail-biting
    constexpr int NB = 256 * BITS;
    u32 st = 0;
    for (int b = 0; b < 16; ++b)
    {
        const int bit = (((t + 1) * BITS - 16 + b) % NB + NB) % NB;
        const u32 word = *reinterpret_cast<const u32*>(p + (bit >> 5) * 4);
        st = (st << 1) | ((word >> (31 - (bit & 31))) & 1u);
    }
    const u32 h2 = tr3_cb2<true>(st, 0u);
    unsigned short e;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e) : "r"(h2));
    codes[i] = static_cast<uint8_t>(e & 0xff);
}

struct Fp8
{
    uint8_t* codes = nullptr;         // [N][K]
    uint8_t* scales = nullptr;        // [N][K/32], all 127
    int K = 0, N = 0;
    bind::Tensor tc() const { bind::Tensor t; t.p = codes; t.nd = 2; t.shape[0] = N; t.shape[1] = K; return t; }
    bind::Tensor ts() const { bind::Tensor t; t.p = scales; t.nd = 2; t.shape[0] = N; t.shape[1] = K / 32; return t; }
};

// Allocates F (N K + N K / 32 bytes of device memory) and decodes m into it. Synchronous.
inline const char* make_fp8(const Mat& m, cudaStream_t st, Fp8& F)
{
    F = Fp8();
    if (!m.w) return nullptr;
    const size_t nc = static_cast<size_t>(m.N) * m.K;
    if (cudaMalloc(&F.codes, nc) != cudaSuccess || cudaMalloc(&F.scales, nc / 32) != cudaSuccess) return "dtr make_fp8: cudaMalloc failed";
    F.K = m.K; F.N = m.N;
    cudaMemsetAsync(F.scales, 127, nc / 32, st);
    k_dtr_fp8<<<static_cast<unsigned>((nc + 255) / 256), 256, 0, st>>>(m.w, m.sB, m.sG, m.K, m.N, F.codes);
    return cudaStreamSynchronize(st) == cudaSuccess ? nullptr : "dtr make_fp8: decode failed";
}
inline void free_fp8(Fp8& F)
{
    cudaFree(F.codes);
    cudaFree(F.scales);
    F = Fp8();
}

// grid ceil(ceil(M/16)*16 * K/128 / 8), block 256: warp = (row, 128 block)
__global__ void __launch_bounds__(256) k_dtr_rotq(const __half* x, int ldx, int K, int M, const __half* suh, uint8_t* x8)
{
    const int lane = threadIdx.x & 31, nbk = K / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nbk, blk = w % nbk;
    if (row >= (M + 15) / 16 * 16) return;
    const int k0 = blk * 128 + lane * 4;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < M)
    {
        float xv[4], s[4];
        ld4h(x + static_cast<size_t>(row) * ldx + k0, xv);
        ld4h(suh + k0, s);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = __fmul_rn(xv[e], s[e]);
    }
    had128(v, lane);
    bindp::shx_store_mx8(v, lane, row, M, K / 32, blk * 4 + (lane >> 3), x8);
}

// grid ceil(M * N/128 / 8), block 256: warp = (row, 128 block)
__global__ void __launch_bounds__(256) k_dtr_post(const __half* yp, int N, int M, const __half* svh, void* out, int ldo, int bf16)
{
    const int lane = threadIdx.x & 31, nbk = N / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nbk, blk = w % nbk;
    if (row >= M) return;
    const int n0 = blk * 128 + lane * 4;
    float v[4], s[4];
    ld4h(yp + static_cast<size_t>(row) * N + n0, v);
    had128(v, lane);
    ld4h(svh + n0, s);
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = __fmul_rn(v[e], s[e]);
    const size_t o = static_cast<size_t>(row) * ldo + n0;
    if (bf16) *reinterpret_cast<uint2*>(static_cast<uint16_t*>(out) + o) = make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]));
    else *reinterpret_cast<uint2*>(static_cast<__half*>(out) + o) = make_uint2(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]));
}

// grid ceil(ceil(M/16)*16 * I/128 / 8), block 256: warp = (row, 128 block)
__global__ void __launch_bounds__(256) k_dtr_pmid(const __half* g, const __half* u, int I, int M, const __half* svh_g,
                                                  const __half* svh_u, const __half* suh_d, uint8_t* x8)
{
    const int lane = threadIdx.x & 31, nbk = I / 128;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), row = w / nbk, b = w % nbk;
    if (row >= (M + 15) / 16 * 16) return;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (row < M)
    {
        const int c0 = b * 128 + lane * 4;
        const size_t o = static_cast<size_t>(row) * I + c0;
        float gv[4], uv[4], sg[4], su[4], sd[4];
        ld4h(g + o, gv); ld4h(u + o, uv); ld4h(svh_g + c0, sg); ld4h(svh_u + c0, su); ld4h(suh_d + c0, sd);
        had128(gv, lane);
        had128(uv, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float gg = rnd_h(__fmul_rn(gv[e], sg[e])), uu = rnd_h(__fmul_rn(uv[e], su[e]));
            v[e] = __fmul_rn(rnd_h(silu_f(gg) * uu), sd[e]);
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = rnd_h(v[e]);
    }
    bindp::shx_store_mx8(v, lane, row, M, I / 32, b * 4 + (lane >> 3), x8);
}

inline unsigned warps_grid(long long warps) { return static_cast<unsigned>((warps + 7) / 8); }
inline long long rows16(int M) { return (M + 15) / 16 * 16; }

// One matrix on M prefill rows: x [M][ldx] -> out [M][ldo] fp16 or BF16. x8 = MXFP8 records scratch
// (ceil(M/16) * K/32 * 528 bytes), yp = fp16 [M][N] scratch.
struct PfMat
{
    int on = 0, M = 0, bf16 = 0, ldo = 0;
    const Mat* m = nullptr;
    const __half* x = nullptr;
    int ldx = 0;
    uint8_t* x8 = nullptr;
    __half* yp = nullptr;
    void* out = nullptr;
    bindp::Gemm gemm;
};
inline const char* make_pf(const Mat& m, const Fp8& F, int M, const __half* x, int ldx, uint8_t* x8, __half* yp, void* out, int ldo, int bf16, PfMat& P)
{
    P = PfMat();
    if (!m.w) return nullptr;
    if (F.K != m.K || F.N != m.N) return "dtr make_pf: FP8 copy does not match the matrix";
    if (const char* e = bindp::make_gemm(F.tc(), F.ts(), M, x8, yp, 0, P.gemm)) return e;
    P.on = 1; P.M = M; P.bf16 = bf16; P.ldo = ldo; P.m = &m; P.x = x; P.ldx = ldx; P.x8 = x8; P.yp = yp; P.out = out;
    return nullptr;
}
inline void launch(const PfMat& P, cudaStream_t st)
{
    if (!P.on) return;
    const Mat& m = *P.m;
    k_dtr_rotq<<<warps_grid(rows16(P.M) * (m.K / 128)), 256, 0, st>>>(P.x, P.ldx, m.K, P.M, m.suh, P.x8);
    bindp::launch(P.gemm, st);
    k_dtr_post<<<warps_grid(static_cast<long long>(P.M) * (m.N / 128)), 256, 0, st>>>(P.yp, m.N, P.M, m.svh, P.out, P.ldo, P.bf16);
}

// The dense MLP on M prefill rows: x [M][ldx] -> out [M][ldo]. Scratch: x8 (K/32 records), x8u (up's, when its suh
// differs from gate's), x8d (I/32 records), g, u fp16 [M][I], yp fp16 [M][N].
struct PfMlp
{
    int on = 0, M = 0, own_up = 0;
    const Mat *gm = nullptr, *um = nullptr, *dm = nullptr;
    const __half* x = nullptr;
    int ldx = 0;
    uint8_t *x8 = nullptr, *x8u = nullptr, *x8d = nullptr;
    __half *g = nullptr, *u = nullptr;
    bindp::Gemm gate, up;
    PfMat down;                       // k_dtr_pmid writes its input records (its x and rotq are unused)
};
inline const char* make_pf_mlp(const Mat& gm, const Mat& um, const Mat& dm, const Fp8& Fg, const Fp8& Fu, const Fp8& Fd, int M,
                               const __half* x, int ldx, uint8_t* x8, uint8_t* x8u, uint8_t* x8d, __half* g, __half* u,
                               __half* yp, void* out, int ldo, int bf16, int up_same_rotation, PfMlp& P)
{
    P = PfMlp();
    if (!gm.w) return nullptr;
    if (!um.w || !dm.w || gm.K != um.K || gm.N != um.N || dm.K != gm.N) return "dtr make_pf_mlp: shapes do not chain";
    const char* e = nullptr;
    if ((e = bindp::make_gemm(Fg.tc(), Fg.ts(), M, x8, g, 0, P.gate))) return e;
    if ((e = bindp::make_gemm(Fu.tc(), Fu.ts(), M, up_same_rotation ? x8 : x8u, u, 0, P.up))) return e;
    if ((e = make_pf(dm, Fd, M, nullptr, 0, x8d, yp, out, ldo, bf16, P.down))) return e;
    P.on = 1; P.M = M; P.own_up = !up_same_rotation;
    P.gm = &gm; P.um = &um; P.dm = &dm; P.x = x; P.ldx = ldx; P.x8 = x8; P.x8u = x8u; P.x8d = x8d; P.g = g; P.u = u;
    return nullptr;
}
inline void launch(const PfMlp& P, cudaStream_t st)
{
    if (!P.on) return;
    const unsigned gk = warps_grid(rows16(P.M) * (P.gm->K / 128));
    k_dtr_rotq<<<gk, 256, 0, st>>>(P.x, P.ldx, P.gm->K, P.M, P.gm->suh, P.x8);
    if (P.own_up) k_dtr_rotq<<<gk, 256, 0, st>>>(P.x, P.ldx, P.um->K, P.M, P.um->suh, P.x8u);
    bindp::launch_pair(P.gate, P.up, st);
    const int I = P.gm->N;
    k_dtr_pmid<<<warps_grid(rows16(P.M) * (I / 128)), 256, 0, st>>>(P.g, P.u, I, P.M, P.gm->svh, P.um->svh, P.dm->suh, P.x8d);
    bindp::launch(P.down.gemm, st);
    k_dtr_post<<<warps_grid(static_cast<long long>(P.M) * (P.dm->N / 128)), 256, 0, st>>>(P.down.yp, P.dm->N, P.M, P.dm->svh, P.down.out,
                                                                                          P.down.ldo, P.down.bf16);
}

}   // namespace dtr
