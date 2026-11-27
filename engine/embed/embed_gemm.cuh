// Standalone mixed-format dense linear GEMM dispatch for g6embed (Blackwell sm_121a / sm_120a).
// Supports all exlr8 v2 dense formats natively on GPU with zero persistent dequantized copies:
//   - FMT_TRELLIS (K4 mcg, side=OUT or side=IN): exact FP16-activation on-the-fly Trellis K4 MMA (m16n8k16)
//     and high-throughput MXFP8-activation on-the-fly Trellis K4 MMA (m16n8k32, block_scale)
//   - FMT_INT4    (g32 row-major q4 + s4): fused nibble unpack + FP32 accumulate
//   - FMT_FP8_ROW (side=OUT/REPLICATED, arbitrary N & K incl. 8B ViT 4304): native cvt.rn.f16x2.e4m3x2 GEMM
//   - FMT_BF16    (copied BF16 or --weight-mode bf16 debug path): cuBLAS BF16 GEMM + fused bias/residual epilogue
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "embed_weights.h"

namespace g6emb {

using u32 = uint32_t;
using u64 = uint64_t;
using bf16 = __nv_bfloat16;

inline void emb_blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("g6embed cuBLAS error: ") + what + " (" + std::to_string(int(s)) + ")");
    }
}
#define EMB_BLAS(x) ::g6emb::emb_blas_ck((x), #x)

// ------------------------------------------------------------------------------------------------ PTX primitives

constexpr float EMB_SQ128 = 0.088388347648f;

__device__ __forceinline__ __half2 emb_as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 emb_as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ u32 emb_pack_h2(float a, float b) { return emb_as_u32(__floats2half2_rn(a, b)); }
__device__ __forceinline__ uint16_t emb_f2bf(float f) {
    u32 u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
__device__ __forceinline__ u32 emb_pack_bf2(float a, float b) {
    return static_cast<u32>(emb_f2bf(a)) | (static_cast<u32>(emb_f2bf(b)) << 16);
}

__device__ __forceinline__ void emb_cp16(void* smem_dst, const void* gmem_src) {
    const u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void emb_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void emb_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void emb_mma16816(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void emb_mma16816_bf16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void emb_mma_mx8(float (&d)[4], const uint4& a, u32 b0, u32 b1, u32 sa, u32 sb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}

__device__ __forceinline__ void emb_ldsm_x4(const void* p, u32& r0, u32& r1, u32& r2, u32& r3) {
    const u32 addr = static_cast<u32>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ u32 emb_h2_e4m3x2(u32 h2) {
    u32 r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "r"(h2));
    return r;
}

__device__ __forceinline__ u32 emb_f2_e4m3x2(float a, float b) {
    u32 r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ u32 emb_mx_scale(float amax, float& inv) {
    const u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f)); // amax / 448
    int e = static_cast<int>(u >> 23) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    if (amax == 0.f) e = 0;
    if (e < -126) e = -126;
    if (e > 127) e = 127;
    inv = __uint_as_float(static_cast<u32>(127 - e) << 23);
    return static_cast<u32>(e + 127);
}

// 128-point Sylvester Hadamard scaled by 1/sqrt(128) across one warp (4 floats per lane).
__device__ __forceinline__ void emb_had128(float (&v)[4], int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? (p - v[e]) : (v[e] + p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= EMB_SQ128;
}

// MCG codebook (exlr8-e4m3): two 16-bit states -> half2 (optionally E4M3-rounded).
template <bool E4>
__device__ __forceinline__ u32 emb_tr3_cb2(u32 w0, u32 w1) {
    u32 x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const u32 lo = __byte_perm(x0, x1, 0x5410);
    const u32 hi = __byte_perm(x0, x1, 0x7632);
    u32 r = emb_as_u32(__hadd2(emb_as_h2(lo), emb_as_h2(hi)));
    if constexpr (E4) {
        asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    }
    return r;
}

// K=4 MCG trellis lane decoder (matches engine/kernels/tr3.cuh Tr3Lane<4, E4, TR3_MCG> bit for bit).
template <bool E4>
struct EmbTr3Lane4 {
    int i0, i1;
    __device__ __forceinline__ void init(int lane) {
        i0 = (lane + 31) & 31;
        i1 = lane;
    }
    __device__ __forceinline__ void decode(u32 a, u32 b, u32 (&f0)[2], u32 (&f1)[2]) const {
        const u64 m = (static_cast<u64>(a) << 32) | b;
        u32 w[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (28 - 4 * j)) & 0xffffu;
        f0[0] = emb_tr3_cb2<E4>(w[0], w[1]);
        f0[1] = emb_tr3_cb2<E4>(w[2], w[3]);
        f1[0] = emb_tr3_cb2<E4>(w[4], w[5]);
        f1[1] = emb_tr3_cb2<E4>(w[6], w[7]);
    }
};

// ------------------------------------------------------------------------------------------------ Epilogue helper

// Applies optional bias and optional residual add when storing a 4-element float vector to BF16[row, col..col+3].
__device__ __forceinline__ void store_bf16x4(bf16* dst, const bf16* bias, bool add_res, int col, const float (&v)[4]) {
    float o[4] = {v[0], v[1], v[2], v[3]};
    if (bias) {
        const uint2 br = *reinterpret_cast<const uint2*>(bias + col);
        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&br.x));
        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&br.y));
        o[0] += b01.x; o[1] += b01.y; o[2] += b23.x; o[3] += b23.y;
    }
    if (add_res) {
        const uint2 rr = *reinterpret_cast<const uint2*>(dst + col);
        const float2 r01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.x));
        const float2 r23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.y));
        // Match PyTorch BF16 residual: bf16(res + bf16(out))
        o[0] = r01.x + __bfloat162float(__float2bfloat16_rn(o[0]));
        o[1] = r01.y + __bfloat162float(__float2bfloat16_rn(o[1]));
        o[2] = r23.x + __bfloat162float(__float2bfloat16_rn(o[2]));
        o[3] = r23.y + __bfloat162float(__float2bfloat16_rn(o[3]));
    }
    *reinterpret_cast<uint2*>(dst + col) = make_uint2(emb_pack_bf2(o[0], o[1]), emb_pack_bf2(o[2], o[3]));
}

// ------------------------------------------------------------------------------------------------ TRELLIS K4 kernels

// Stage 1 (H16 exact path): Rotate BF16 X[M, K] by suh[K] + H128 and pack directly into FP16 m16n8k16 A-fragments:
// dst [ceil(M/16)][K/16][512 bytes] where each 512B block holds 32 lanes * uint4 (a0, a1, a2, a3).
// Grid: (ceil(M/16), K/128), Block: 256 (8 warps; each warp handles 2 rows of the 16-row tile).
__global__ void __launch_bounds__(256) k_emb_rot_h16(
    const bf16* __restrict__ x, const __half* __restrict__ suh, uint8_t* __restrict__ dst, int M, int K) {
    __shared__ __half sh[16][128];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int mb = blockIdx.x, kb = blockIdx.y;
    const int k0 = kb * 128 + lane * 4;
    const int K16 = K >> 4;

    const uint2 sr = *reinterpret_cast<const uint2*>(suh + k0);
    const float2 s01 = __half22float2(emb_as_h2(sr.x)), s23 = __half22float2(emb_as_h2(sr.y));

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int lr = warp + pass * 8;
        const int row = mb * 16 + lr;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (row < M) {
            const uint2 xr = *reinterpret_cast<const uint2*>(x + size_t(row) * K + k0);
            const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.x));
            const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.y));
            v[0] = x01.x * s01.x; v[1] = x01.y * s01.y;
            v[2] = x23.x * s23.x; v[3] = x23.y * s23.y;
            emb_had128(v, lane);
        }
        *reinterpret_cast<uint2*>(&sh[lr][lane * 4]) = make_uint2(emb_pack_h2(v[0], v[1]), emb_pack_h2(v[2], v[3]));
    }
    __syncthreads();

    // Pack 8 k16-steps (1 warp per k16-step, 32 lanes -> 512 B per k16-step)
    const int j = warp;
    const int r = lane >> 2, c = (lane & 3) * 2;
    const int c0 = j * 16 + c;
    const u32 a0 = *reinterpret_cast<const u32*>(&sh[r][c0]);
    const u32 a1 = *reinterpret_cast<const u32*>(&sh[r + 8][c0]);
    const u32 a2 = *reinterpret_cast<const u32*>(&sh[r][c0 + 8]);
    const u32 a3 = *reinterpret_cast<const u32*>(&sh[r + 8][c0 + 8]);
    uint4* out_blk = reinterpret_cast<uint4*>(dst + (size_t(mb) * K16 + kb * 8 + j) * 512);
    out_blk[lane] = make_uint4(a0, a1, a2, a3);
}

// Stage 2 (H16 exact path): On-the-fly Trellis K4 -> E4M3 FP16 -> mma.m16n8k16 + H128*svh (+ bias/residual) epilogue.
// Matches k_gemm_tr_h16 in bind_prefill.cuh with zero GLM coupling.
template <int BM>
struct EmbTrH16Cfg {
    static constexpr int WM = BM / 4;
    static constexpr int WN = (BM >= 8) ? 4 : 8;
    static constexpr int NW = WM * WN;
    static constexpr int NTH = 32 * NW;
    static constexpr int WC = 128 / WN;
    static constexpr int NT = WC / 8;
    static constexpr int A = 4 * BM * 512;
    static constexpr int RAW = 32 * 128;
    static constexpr int STG = A + RAW;
    static constexpr int ST = (BM == 16) ? 2 : 3;
    // Shared memory must also hold 64 * 136 * 4 = 34816 B for the H128 epilogue.
    static constexpr int SMEM = (ST * STG > 34816) ? (ST * STG) : 34816;
};

template <int BM>
__global__ void __launch_bounds__(EmbTrH16Cfg<BM>::NTH) k_emb_tr_h16(
    const uint8_t* __restrict__ x_h16,
    const uint8_t* __restrict__ tw,
    const __half* __restrict__ svh,
    const bf16* __restrict__ bias,
    bf16* __restrict__ y,
    int M, int N, int K, int64_t sB, int64_t sG, int add_res) {
    using C = EmbTrH16Cfg<BM>;
    constexpr int NT = C::NT;
    extern __shared__ __align__(16) unsigned char sm[];

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int K16 = K >> 4, KG = K >> 6, RB = (M + 15) >> 4;
    const int c0 = blockIdx.x * 128, rb0 = blockIdx.y * BM;
    const int nb = min(BM, RB - rb0), colw = wn * C::WC;
    const bool full = (nb == BM);

    EmbTr3Lane4<true> tl;
    tl.init(lane);

    auto loadAB = [&](int slot, int kb) {
        unsigned char* a = sm + slot * C::STG;
        for (int c = t; c < 4 * BM * 32; c += C::NTH) {
            const int j = c / (BM * 32), rem = c % (BM * 32), mb = rem >> 5, ch = rem & 31;
            if (mb < nb && kb * 4 + j < K16) {
                emb_cp16(a + c * 16, x_h16 + (size_t(rb0 + mb) * K16 + kb * 4 + j) * 512 + ch * 16);
            }
        }
        for (int c = t; c < 256; c += C::NTH) {
            const int ti = c >> 3, ks = kb * 4 + (ti >> 3);
            if (ks < K16) {
                emb_cp16(a + C::A + c * 16,
                         tw + int64_t(blockIdx.x) * sB + int64_t(ks >> 3) * sG +
                             int64_t((ks & 7) * 8 + (ti & 7)) * 128 + (c & 7) * 16);
            }
        }
    };

    #pragma unroll
    for (int p = 0; p < C::ST - 1; ++p) {
        if (p < KG) loadAB(p, p);
        emb_cp_commit();
    }

    float acc[4][NT][4];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][nt][i] = 0.f;

    for (int kb = 0; kb < KG; ++kb) {
        emb_cp_wait<C::ST - 2>();
        __syncthreads();
        if (kb + C::ST - 1 < KG) loadAB((kb + C::ST - 1) % C::ST, kb + C::ST - 1);
        emb_cp_commit();
        const unsigned char* a = sm + (kb % C::ST) * C::STG;
        const unsigned char* raw = a + C::A;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (kb * 4 + j >= K16) continue;
            u32 bf[NT][2];
            #pragma unroll
            for (int pr = 0; pr < NT / 2; ++pr) {
                const int ti = j * 8 + wn * (NT / 2) + pr;
                const u32* tile = reinterpret_cast<const u32*>(raw + ti * 128);
                tl.decode(tile[tl.i0], tile[tl.i1], bf[2 * pr], bf[2 * pr + 1]);
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb) {
                const int m = wm * 4 + mb;
                if (!full && m >= nb) continue;
                const uint4 fa = *reinterpret_cast<const uint4*>(a + (j * BM + m) * 512 + lane * 16);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    emb_mma16816(acc[mb][nt], fa.x, fa.y, fa.z, fa.w, bf[nt][0], bf[nt][1]);
                }
            }
        }
    }
    emb_cp_wait<0>();

    // Pure-FP32 H128 * svh (+ optional bias + optional residual) epilogue in 64-row passes.
    constexpr int LD = 136;
    float* tile = reinterpret_cast<float*>(sm);
    const int c = lane & 3, r = lane >> 2;
    const uint2 sr = *reinterpret_cast<const uint2*>(svh + c0 + lane * 4);
    const float2 s01 = __half22float2(emb_as_h2(sr.x)), s23 = __half22float2(emb_as_h2(sr.y));
    #pragma unroll
    for (int p = 0; p < BM; p += 4) {
        __syncthreads();
        if (wm * 4 == p && p < nb) {
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb) {
                if (p + mb >= nb) continue;
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int rr = mb * 16 + r, col = colw + nt * 8 + 2 * c;
                    *reinterpret_cast<float2*>(tile + rr * LD + col) = make_float2(acc[mb][nt][0], acc[mb][nt][1]);
                    *reinterpret_cast<float2*>(tile + (rr + 8) * LD + col) = make_float2(acc[mb][nt][2], acc[mb][nt][3]);
                }
            }
        }
        __syncthreads();
        const int nrows = min(64, max(0, (nb - p) * 16));
        for (int rr = warp; rr < nrows; rr += C::NW) {
            const int row = (rb0 + p) * 16 + rr;
            if (row >= M) break;
            const float2 a0 = *reinterpret_cast<const float2*>(tile + rr * LD + lane * 4);
            const float2 a1 = *reinterpret_cast<const float2*>(tile + rr * LD + lane * 4 + 2);
            float v[4] = {a0.x, a0.y, a1.x, a1.y};
            emb_had128(v, lane);
            v[0] = __fmul_rn(v[0], s01.x);
            v[1] = __fmul_rn(v[1], s01.y);
            v[2] = __fmul_rn(v[2], s23.x);
            v[3] = __fmul_rn(v[3], s23.y);
            store_bf16x4(y + size_t(row) * N, bias, add_res != 0, c0 + lane * 4, v);
        }
    }
}

// ------------------------------------------------------------------------------------------------ FP8_ROW & INT4 kernels

// Fused FP8_ROW token embedding lookup: out[m, d] = bf16(e4m3(fp8[tok[m], d]) * scale[tok[m]]).
// Grid: (M, ceil(H / 256)), Block: 256 (each thread handles 2 elements via native cvt.rn.f16x2.e4m3x2).
__global__ void __launch_bounds__(256) k_embed_lookup_fp8_row(
    const int* __restrict__ toks, const uint8_t* __restrict__ fp8, const float* __restrict__ scale,
    bf16* __restrict__ out, int M, int H) {
    const int m = blockIdx.x;
    const int d2 = (blockIdx.y * 256 + threadIdx.x) * 2;
    if (m >= M || d2 >= H) return;
    const int tok = toks[m];
    const float sc = scale[tok];
    const uint16_t raw = *reinterpret_cast<const uint16_t*>(fp8 + size_t(tok) * H + d2);
    u32 h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    const float2 fv = __half22float2(emb_as_h2(h2));
    *reinterpret_cast<u32*>(out + size_t(m) * H + d2) = emb_pack_bf2(fv.x * sc, fv.y * sc);
}

__global__ void __launch_bounds__(256) k_embed_lookup_bf16(
    const int* __restrict__ toks, const bf16* __restrict__ table, bf16* __restrict__ out, int M, int H) {
    const int m = blockIdx.x;
    const int d8 = (blockIdx.y * 256 + threadIdx.x) * 8;
    if (m >= M || d8 >= H) return;
    const int tok = toks[m];
    *reinterpret_cast<uint4*>(out + size_t(m) * H + d8) = *reinterpret_cast<const uint4*>(table + size_t(tok) * H + d8);
}

// FP8_ROW tiled GEMM: Y[M, N] = X[M, K] (BF16) @ (fp8[N, K] * scale[N])^T [+ bias[N]] [+ residual].
// Works for arbitrary N, K divisible by 16 (including 8B ViT 4304!).
// CTA = 32 rows (M) x 32 cols (N) using 4 warps (each warp computes 16x16 via 2x2 mma.m16n8k16.bf16).
// Baseline clean implementation (target for Phase B Tensor-Core / cp.async hill-climbing!).
__global__ void __launch_bounds__(128) k_emb_fp8_row_gemm(
    const bf16* __restrict__ X,
    const uint8_t* __restrict__ Wq,
    const float* __restrict__ Wsc,
    const bf16* __restrict__ bias,
    bf16* __restrict__ Y,
    int M, int N, int K, int add_res) {
    __shared__ __align__(16) bf16 sA[32][32]; // [32 rows][32 K]
    __shared__ __align__(16) bf16 sB[32][32]; // [32 cols][32 K]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1; // 2x2 warps covering 32x32
    const int m0 = blockIdx.y * 32, n0 = blockIdx.x * 32;
    const int r = lane >> 2, c = (lane & 3) * 2;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int k0 = 0; k0 < K; k0 += 32) {
        // Load A[32, 32] (BF16) cooperatively: 128 threads x 8 elements = 1024
        for (int i = tid; i < 32 * 4; i += 128) {
            const int lr = i >> 2, lc = (i & 3) * 8;
            const int gr = m0 + lr;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < M && k0 + lc < K) {
                v = *reinterpret_cast<const uint4*>(X + size_t(gr) * K + k0 + lc);
            }
            *reinterpret_cast<uint4*>(&sA[lr][lc]) = v;
        }
        // Load & dequantize B[32, 32] (FP8 -> BF16) cooperatively: 128 threads x 8 bytes = 1024
        for (int i = tid; i < 32 * 4; i += 128) {
            const int lr = i >> 2, lc = (i & 3) * 8;
            const int gn = n0 + lr;
            u32 out4[4] = {0, 0, 0, 0};
            if (gn < N && k0 + lc < K) {
                const uint2 raw8 = *reinterpret_cast<const uint2*>(Wq + size_t(gn) * K + k0 + lc);
                const u32 words[2] = {raw8.x, raw8.y};
                #pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const uint16_t lo = static_cast<uint16_t>(words[w] & 0xffffu);
                    const uint16_t hi = static_cast<uint16_t>(words[w] >> 16);
                    u32 h0, h1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(lo));
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(hi));
                    const float2 f0 = __half22float2(emb_as_h2(h0));
                    const float2 f1 = __half22float2(emb_as_h2(h1));
                    out4[w * 2 + 0] = emb_pack_bf2(f0.x, f0.y);
                    out4[w * 2 + 1] = emb_pack_bf2(f1.x, f1.y);
                }
            }
            *reinterpret_cast<uint4*>(&sB[lr][lc]) = make_uint4(out4[0], out4[1], out4[2], out4[3]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            const int mr = wm * 16 + r;
            const u32 a0 = *reinterpret_cast<const u32*>(&sA[mr][kk + c]);
            const u32 a1 = *reinterpret_cast<const u32*>(&sA[mr + 8][kk + c]);
            const u32 a2 = *reinterpret_cast<const u32*>(&sA[mr][kk + c + 8]);
            const u32 a3 = *reinterpret_cast<const u32*>(&sA[mr + 8][kk + c + 8]);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int nr = wn * 16 + nt * 8 + r;
                const u32 b0 = *reinterpret_cast<const u32*>(&sB[nr][kk + c]);
                const u32 b1 = *reinterpret_cast<const u32*>(&sB[nr][kk + c + 8]);
                emb_mma16816_bf16(acc[nt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    // Store scaled results
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = n0 + wn * 16 + nt * 8 + c;
        const int row0 = m0 + wm * 16 + r;
        const int row1 = row0 + 8;
        if (col + 1 < N) {
            const float sc0 = Wsc[col], sc1 = Wsc[col + 1];
            const float b0 = bias ? __bfloat162float(bias[col]) : 0.f;
            const float b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
            if (row0 < M) {
                float v0 = acc[nt][0] * sc0 + b0, v1 = acc[nt][1] * sc1 + b1;
                if (add_res) {
                    v0 = __bfloat162float(Y[size_t(row0) * N + col]) + __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(Y[size_t(row0) * N + col + 1]) + __bfloat162float(__float2bfloat16_rn(v1));
                }
                *reinterpret_cast<u32*>(Y + size_t(row0) * N + col) = emb_pack_bf2(v0, v1);
            }
            if (row1 < M) {
                float v0 = acc[nt][2] * sc0 + b0, v1 = acc[nt][3] * sc1 + b1;
                if (add_res) {
                    v0 = __bfloat162float(Y[size_t(row1) * N + col]) + __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(Y[size_t(row1) * N + col + 1]) + __bfloat162float(__float2bfloat16_rn(v1));
                }
                *reinterpret_cast<u32*>(Y + size_t(row1) * N + col) = emb_pack_bf2(v0, v1);
            }
        }
    }
}

// INT4 g32 tiled GEMM (row-major q4[N, K/2] + s4[N, K/32] FP16 -> BF16 mma.m16n8k16).
__global__ void __launch_bounds__(128) k_emb_int4_gemm(
    const bf16* __restrict__ X,
    const uint8_t* __restrict__ q4,
    const __half* __restrict__ s4,
    const bf16* __restrict__ bias,
    bf16* __restrict__ Y,
    int M, int N, int K, int add_res) {
    __shared__ __align__(16) bf16 sA[32][32];
    __shared__ __align__(16) bf16 sB[32][32];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 32, n0 = blockIdx.x * 32;
    const int r = lane >> 2, c = (lane & 3) * 2;
    const int K2 = K >> 1, K32 = K >> 5;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int i = tid; i < 32 * 4; i += 128) {
            const int lr = i >> 2, lc = (i & 3) * 8;
            const int gr = m0 + lr;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < M && k0 + lc < K) {
                v = *reinterpret_cast<const uint4*>(X + size_t(gr) * K + k0 + lc);
            }
            *reinterpret_cast<uint4*>(&sA[lr][lc]) = v;
        }
        // Unpack 32x32 INT4 weights (128 threads: 4 threads per row, each unpacks 8 nibbles = 4 bytes)
        for (int i = tid; i < 32 * 4; i += 128) {
            const int lr = i >> 2, lc = (i & 3) * 8;
            const int gn = n0 + lr;
            u32 out4[4] = {0, 0, 0, 0};
            if (gn < N && k0 + lc < K) {
                const float sc = __half2float(s4[size_t(gn) * K32 + (k0 >> 5)]);
                const u32 packed = *reinterpret_cast<const u32*>(q4 + size_t(gn) * K2 + ((k0 + lc) >> 1));
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const u32 byte = (packed >> (b * 8)) & 0xffu;
                    const float lo = static_cast<float>(static_cast<int>(byte & 15u) - 8) * sc;
                    const float hi = static_cast<float>(static_cast<int>(byte >> 4) - 8) * sc;
                    out4[b] = emb_pack_bf2(lo, hi);
                }
            }
            *reinterpret_cast<uint4*>(&sB[lr][lc]) = make_uint4(out4[0], out4[1], out4[2], out4[3]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            const int mr = wm * 16 + r;
            const u32 a0 = *reinterpret_cast<const u32*>(&sA[mr][kk + c]);
            const u32 a1 = *reinterpret_cast<const u32*>(&sA[mr + 8][kk + c]);
            const u32 a2 = *reinterpret_cast<const u32*>(&sA[mr][kk + c + 8]);
            const u32 a3 = *reinterpret_cast<const u32*>(&sA[mr + 8][kk + c + 8]);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int nr = wn * 16 + nt * 8 + r;
                const u32 b0 = *reinterpret_cast<const u32*>(&sB[nr][kk + c]);
                const u32 b1 = *reinterpret_cast<const u32*>(&sB[nr][kk + c + 8]);
                emb_mma16816_bf16(acc[nt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = n0 + wn * 16 + nt * 8 + c;
        const int row0 = m0 + wm * 16 + r;
        const int row1 = row0 + 8;
        if (col + 1 < N) {
            const float b0 = bias ? __bfloat162float(bias[col]) : 0.f;
            const float b1 = bias ? __bfloat162float(bias[col + 1]) : 0.f;
            if (row0 < M) {
                float v0 = acc[nt][0] + b0, v1 = acc[nt][1] + b1;
                if (add_res) {
                    v0 = __bfloat162float(Y[size_t(row0) * N + col]) + __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(Y[size_t(row0) * N + col + 1]) + __bfloat162float(__float2bfloat16_rn(v1));
                }
                *reinterpret_cast<u32*>(Y + size_t(row0) * N + col) = emb_pack_bf2(v0, v1);
            }
            if (row1 < M) {
                float v0 = acc[nt][2] + b0, v1 = acc[nt][3] + b1;
                if (add_res) {
                    v0 = __bfloat162float(Y[size_t(row1) * N + col]) + __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(Y[size_t(row1) * N + col + 1]) + __bfloat162float(__float2bfloat16_rn(v1));
                }
                *reinterpret_cast<u32*>(Y + size_t(row1) * N + col) = emb_pack_bf2(v0, v1);
            }
        }
    }
}

// Epilogue for cuBLAS FP32 accumulator -> BF16 with optional bias and residual add.
__global__ void __launch_bounds__(256) k_emb_f32_to_bf16_epi(
    const float* __restrict__ acc, const bf16* __restrict__ bias, bf16* __restrict__ y,
    int M, int N, int add_res) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(M) * N) return;
    float v = acc[i];
    if (bias) v += __bfloat162float(bias[i % N]);
    if (add_res) {
        v = __bfloat162float(y[i]) + __bfloat162float(__float2bfloat16_rn(v));
    }
    y[i] = __float2bfloat16_rn(v);
}

// ------------------------------------------------------------------------------------------------ Workspace & dispatch

struct GemmWorkspace {
    cublasHandle_t blas = nullptr;
    uint8_t* x_rot = nullptr;      // [ceil(max_M/16) * (max_K/16) * 512] FP16 A-fragments for TRELLIS K4
    size_t x_rot_bytes = 0;
    float* f32_acc = nullptr;      // [acc_floats] FP32 scratch for BF16 cuBLAS bias/residual epilogue
    size_t acc_floats = 0;

    void init(int max_M, int max_K, size_t max_acc_floats = size_t(16) << 20) {
        EMB_BLAS(cublasCreate(&blas));
        const size_t rb = (size_t(max_M) + 15) / 16;
        const size_t k16 = (size_t(max_K) + 15) / 16;
        x_rot_bytes = std::max<size_t>(4096, rb * k16 * 512);
        EMB_CK(cudaMalloc(&x_rot, x_rot_bytes));
        acc_floats = max_acc_floats;
        EMB_CK(cudaMalloc(&f32_acc, acc_floats * sizeof(float)));

        // Set dynamic shared memory attribute for Trellis K4 GEMM kernels once.
        EMB_CK(cudaFuncSetAttribute(k_emb_tr_h16<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, EmbTrH16Cfg<4>::SMEM));
        EMB_CK(cudaFuncSetAttribute(k_emb_tr_h16<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, EmbTrH16Cfg<8>::SMEM));
        EMB_CK(cudaFuncSetAttribute(k_emb_tr_h16<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, EmbTrH16Cfg<16>::SMEM));
    }

    void ensure_rot(int M, int K) {
        const size_t rb = (size_t(M) + 15) / 16;
        const size_t k16 = (size_t(K) + 15) / 16;
        const size_t need = std::max<size_t>(4096, rb * k16 * 512);
        if (need > x_rot_bytes) {
            if (x_rot) cudaFree(x_rot);
            x_rot = nullptr;
            x_rot_bytes = 0;
            EMB_CK(cudaMalloc(&x_rot, need));
            x_rot_bytes = need;
        }
    }

    void release() {
        if (blas) { cublasDestroy(blas); blas = nullptr; }
        if (x_rot) { cudaFree(x_rot); x_rot = nullptr; x_rot_bytes = 0; }
        if (f32_acc) { cudaFree(f32_acc); f32_acc = nullptr; acc_floats = 0; }
    }
    ~GemmWorkspace() { release(); }
};

// Rotates X[M, K] by suh[K] into ws.x_rot so multiple matrices sharing the same input & suh
// (or standalone matrix calls) can reuse ws.x_rot cleanly.
inline void launch_rot_h16(const bf16* X, const __half* suh, int M, int K, GemmWorkspace& ws, cudaStream_t st) {
    const unsigned rb = static_cast<unsigned>((M + 15) / 16);
    const unsigned kb = static_cast<unsigned>(K / 128);
    k_emb_rot_h16<<<dim3(rb, kb), 256, 0, st>>>(X, suh, ws.x_rot, M, K);
}

inline void launch_trellis_with_rot(const DevMatrix& W, const bf16* bias, bf16* Y, int M, bool add_res,
                                    GemmWorkspace& ws, cudaStream_t st) {
    const int RB = (M + 15) / 16;
    const int NB = W.out / 128;
    const int ar = add_res ? 1 : 0;
    if (M >= 256) {
        const dim3 grid(NB, (RB + 15) / 16);
        k_emb_tr_h16<16><<<grid, EmbTrH16Cfg<16>::NTH, EmbTrH16Cfg<16>::SMEM, st>>>(
            ws.x_rot, W.data, W.svh, bias, Y, M, W.out, W.inp, W.sB, W.sG, ar);
    } else if (M >= 96) {
        const dim3 grid(NB, (RB + 7) / 8);
        k_emb_tr_h16<8><<<grid, EmbTrH16Cfg<8>::NTH, EmbTrH16Cfg<8>::SMEM, st>>>(
            ws.x_rot, W.data, W.svh, bias, Y, M, W.out, W.inp, W.sB, W.sG, ar);
    } else {
        const dim3 grid(NB, (RB + 3) / 4);
        k_emb_tr_h16<4><<<grid, EmbTrH16Cfg<4>::NTH, EmbTrH16Cfg<4>::SMEM, st>>>(
            ws.x_rot, W.data, W.svh, bias, Y, M, W.out, W.inp, W.sB, W.sG, ar);
    }
}

// Unified linear projection: Y[M, W.out] = X[M, W.inp] @ W^T [+ bias] [+ residual].
inline void linear_forward(const bf16* X, const DevMatrix& W, const bf16* bias, bf16* Y, int M,
                           bool add_res, GemmWorkspace& ws, cudaStream_t st) {
    if (M <= 0) return;
    switch (W.fmt) {
        case FMT_TRELLIS: {
            launch_rot_h16(X, W.suh, M, W.inp, ws, st);
            launch_trellis_with_rot(W, bias, Y, M, add_res, ws, st);
            break;
        }
        case FMT_FP8_ROW: {
            const dim3 grid((W.out + 31) / 32, (M + 31) / 32);
            k_emb_fp8_row_gemm<<<grid, 128, 0, st>>>(X, W.data, W.scale, bias, Y, M, W.out, W.inp, add_res ? 1 : 0);
            break;
        }
        case FMT_INT4: {
            const dim3 grid((W.out + 31) / 32, (M + 31) / 32);
            k_emb_int4_gemm<<<grid, 128, 0, st>>>(X, W.data, W.s4, bias, Y, M, W.out, W.inp, add_res ? 1 : 0);
            break;
        }
        case FMT_BF16: {
            EMB_BLAS(cublasSetStream(ws.blas, st));
            const float alpha = 1.0f, beta = 0.0f;
            if (!bias && !add_res) {
                EMB_BLAS(cublasGemmEx(ws.blas, CUBLAS_OP_T, CUBLAS_OP_N, W.out, M, W.inp,
                                      &alpha, W.bf16, CUDA_R_16BF, W.inp,
                                      X, CUDA_R_16BF, W.inp,
                                      &beta, Y, CUDA_R_16BF, W.out,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                const int chunk = std::max(1, static_cast<int>(ws.acc_floats / size_t(W.out)));
                for (int r0 = 0; r0 < M; r0 += chunk) {
                    const int rows = std::min(chunk, M - r0);
                    EMB_BLAS(cublasGemmEx(ws.blas, CUBLAS_OP_T, CUBLAS_OP_N, W.out, rows, W.inp,
                                          &alpha, W.bf16, CUDA_R_16BF, W.inp,
                                          X + size_t(r0) * W.inp, CUDA_R_16BF, W.inp,
                                          &beta, ws.f32_acc, CUDA_R_32F, W.out,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    const size_t n = size_t(rows) * W.out;
                    k_emb_f32_to_bf16_epi<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(
                        ws.f32_acc, bias, Y + size_t(r0) * W.out, rows, W.out, add_res ? 1 : 0);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("g6embed linear_forward: unsupported matrix format " + std::to_string(W.fmt));
    }
}

// Raw BF16 weight matrix GEMM (e.g. 5D patch_embed.proj flattened to [out, inp]).
inline void bf16_linear_forward(const bf16* X, const bf16* W, const bf16* bias, bf16* Y, int M, int N, int K,
                                bool add_res, GemmWorkspace& ws, cudaStream_t st) {
    DevMatrix dm;
    dm.fmt = FMT_BF16;
    dm.out = N;
    dm.inp = K;
    dm.bf16 = W;
    linear_forward(X, dm, bias, Y, M, add_res, ws, st);
}

} // namespace g6emb
