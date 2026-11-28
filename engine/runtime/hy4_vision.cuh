// Tencent Hy4-flash vision tower (HYV4FlashVisionTransformer, dense/vision-bf16.safetensors) and S2 prefill staging.
// Every TP rank holds the whole 27-layer BF16 ViT (~890 MiB) and encodes images locally (no cross-rank communication).
//
// In: smart-resized RGB uint8 [vit_h * 16][vit_w * 16][3] (hy4_vision_plan.h).
// Out: BF16 [(vit_h / 2) * (vit_w / 2 + 1)][3072], one row per image_token_id (120120) in row-major merged order
//      with vit.perceive.image_newline appended to every merged row.
//
// Pipeline (verified verbatim against HF PR 50903 modeling_hy_v4_flash.py):
//   patchify  row-major patch (r, c) -> [3, 16, 16] BF16 normalized (u / 255.0 - 0.5) / 0.5
//   embed     x = bf16(patch_embedding(patches) + bias) + bf16(bilinear(pos_embed[1..16384], scale=(size+0.1)/128, align_corners=False))
//   27 layers h = x;
//             h1 = LayerNorm(h, eps=1e-5)
//             q/k/v = linear(h1, q/k/v_proj) [N, 16, 72] (no ViT RoPE)
//             bidirectional SDPA (16 heads x 72 dim, scale = 1/sqrt(72)) -> o_proj -> h += out
//             h += dense_4h_to_h(gelu_exact(dense_h_to_4h(LayerNorm(h, eps=1e-5))))  (1152 -> 4304 -> 1152)
//   merger    y = RMSNorm(h, eps=1e-5) [vit_h, vit_w, 1152]
//             2x2 Conv2d(1152 -> 4608, k=2, s=2) + bias -> gelu_exact -> 1x1 Conv2d(4608 -> 3072) + bias
//             append vit.perceive.image_newline [3072] at end of each merged row -> [(vit_h/2)*(vit_w/2 + 1), 3072]
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hy4_vision_plan.h"
#include "image_input.h"
#include "vision_id.h"
#include "vision_weights.h"

namespace hy4rt {

#define HY4V_CUDA(x) do { cudaError_t ce_ = (x); if (ce_ != cudaSuccess) throw std::runtime_error(std::string("hy4 vision: ") + #x + ": " + cudaGetErrorString(ce_)); } while (0)
#define HY4V_BLAS(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("hy4 vision: ") + #x + " failed: " + std::to_string((int)st_)); } while (0)

namespace vis {
using bf16 = __nv_bfloat16;

constexpr int VD = 1152;                              // vision_config.hidden_size
constexpr int VHEADS = 16;                            // vision_config.num_attention_heads
constexpr int VHD = 72;                               // VD / VHEADS
constexpr int VL = 27;                                // vision_config.num_hidden_layers
constexpr int VMLP = 4304;                            // vision_config.intermediate_size
constexpr int VGRID = 128;                            // max_image_size (2048) / patch_size (16)
constexpr int VOUT = 3072;                            // vision_config.out_hidden_size
constexpr int VMRG = VD * hy4v::MERGE * hy4v::MERGE;  // 4608
constexpr int VPK = 3 * hy4v::PATCH * hy4v::PATCH;    // 768
constexpr float VEPS = 1e-5f;                         // vision_config.rms_norm_eps & layer_norm_eps
constexpr size_t ACC_FLOATS = size_t(8) << 20;        // 32 MiB FP32 GEMM accumulator

// Row-major patchify: pixels [vit_h*16, vit_w*16, 3] uint8 -> patches [N, 3, 16, 16] BF16.
// Grid: (N), Block: 256.
__global__ void __launch_bounds__(256) k_hy4v_patchify(const uint8_t* __restrict__ px, int vit_w, bf16* __restrict__ out) {
    const int n = blockIdx.x;
    const int row = n / vit_w;
    const int col = n % vit_w;
    const size_t W = size_t(vit_w) * hy4v::PATCH;
    bf16* dst = out + size_t(n) * VPK;
    for (int i = threadIdx.x; i < VPK; i += 256) {
        const int ch = i >> 8;        // 0..2
        const int rem = i & 255;
        const int py = rem >> 4;      // 0..15
        const int pxx = rem & 15;     // 0..15
        const int gy = row * hy4v::PATCH + py;
        const int gx = col * hy4v::PATCH + pxx;
        const float u = static_cast<float>(px[(size_t(gy) * W + gx) * 3 + ch]);
        dst[i] = __float2bfloat16_rn(__fdiv_rn(__fsub_rn(__fmul_rn(u, 1.0f / 255.0f), 0.5f), 0.5f));
    }
}

// Bilinear interpolation matching PyTorch F.interpolate(pos[128,128], scale_factor=((h+0.1)/128, (w+0.1)/128),
// mode="bilinear", align_corners=False) with recompute_scale_factor=None (real_scale = 1 / scale_factor).
__device__ __forceinline__ void pos_lerp_hy4(int i, int n, int g, int& i0, int& i1, float& w1) {
    const double scale_factor = (static_cast<double>(n) + 0.1) / static_cast<double>(g);
    const double real_scale = 1.0 / scale_factor;
    double src = real_scale * (static_cast<double>(i) + 0.5) - 0.5;
    if (src < 0.0) src = 0.0;
    const int f = static_cast<int>(floor(src));
    i0 = min(max(f, 0), g - 1);
    i1 = min(i0 + 1, g - 1);
    w1 = static_cast<float>(src - static_cast<double>(f));
    if (i0 == i1) w1 = 0.0f;
}

// x[N, 1152] += bf16(bilinear(pos_table[1..16384], vit_h, vit_w)). Grid: (N), Block: 256.
__global__ void __launch_bounds__(256) k_hy4v_pos_add(bf16* __restrict__ x, const bf16* __restrict__ table_with_cls,
                                                      int vit_h, int vit_w) {
    const int n = blockIdx.x;
    const int row = n / vit_w;
    const int col = n % vit_w;
    int y0, y1, x0, x1;
    float wy1, wx1;
    pos_lerp_hy4(row, vit_h, VGRID, y0, y1, wy1);
    pos_lerp_hy4(col, vit_w, VGRID, x0, x1, wx1);
    const float wy0 = 1.0f - wy1, wx0 = 1.0f - wx1;
    const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
    const bf16* table = table_with_cls + VD;  // skip row 0 (unused CLS token)
    const bf16* a = table + size_t(y0 * VGRID + x0) * VD;
    const bf16* b = table + size_t(y0 * VGRID + x1) * VD;
    const bf16* c = table + size_t(y1 * VGRID + x0) * VD;
    const bf16* e = table + size_t(y1 * VGRID + x1) * VD;
    bf16* xr = x + size_t(n) * VD;
    for (int d = threadIdx.x; d < VD; d += 256) {
        float p = w00 * __bfloat162float(a[d]) + w01 * __bfloat162float(b[d]) +
                  w10 * __bfloat162float(c[d]) + w11 * __bfloat162float(e[d]);
        p = __bfloat162float(__float2bfloat16_rn(p));
        xr[d] = __float2bfloat16_rn(__bfloat162float(xr[d]) + p);
    }
}

__device__ __forceinline__ float sum128(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float t = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return t;
}

// LayerNorm over VD=1152 (9 elements per thread across 128 threads, eps=1e-5). Grid: (N), Block: 128.
__global__ void __launch_bounds__(128) k_hy4v_layernorm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                        const bf16* __restrict__ b, bf16* __restrict__ y) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const bf16* xr = x + size_t(r) * VD;
    bf16* yr = y + size_t(r) * VD;
    float v[9], s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        v[j] = __bfloat162float(xr[threadIdx.x + 128 * j]);
        s += v[j];
    }
    const float mean = sum128(s, red) * (1.0f / float(VD));
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        const float d = v[j] - mean;
        ss += d * d;
    }
    const float rs = rsqrtf(sum128(ss, red) * (1.0f / float(VD)) + VEPS);
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        const int k = threadIdx.x + 128 * j;
        const float t = (v[j] - mean) * rs * __bfloat162float(w[k]) + __bfloat162float(b[k]);
        yr[k] = __float2bfloat16_rn(t);
    }
}

// RMSNorm over VD=1152 (vit.perceive.norm, eps=1e-5). Grid: (N), Block: 128.
__global__ void __launch_bounds__(128) k_hy4v_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                      bf16* __restrict__ y) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const bf16* xr = x + size_t(r) * VD;
    bf16* yr = y + size_t(r) * VD;
    float v[9], ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        v[j] = __bfloat162float(xr[threadIdx.x + 128 * j]);
        ss += v[j] * v[j];
    }
    const float rs = rsqrtf(sum128(ss, red) * (1.0f / float(VD)) + VEPS);
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        const int k = threadIdx.x + 128 * j;
        const float xn = __bfloat162float(__float2bfloat16_rn(v[j] * rs));
        yr[k] = __float2bfloat16_rn(xn * __bfloat162float(w[k]));
    }
}

// Bidirectional tiled BF16 Tensor Core FlashAttention (mma.sync.aligned.m16n8k16) for HD=72, 16 heads.
// Grid: (ceil(N / 64), 16), Block: 128 (4 warps; each warp handles 16 query rows, tiling K/V in 32-row chunks).
// Padded HD=80 (5 x 16) in shared memory for bank-conflict-free 16x8x16 MMA loads (~20 KiB smem/CTA).
constexpr int ATTN_BM = 64;
constexpr int ATTN_BN = 32;
constexpr int ATTN_HD_PAD = 80;

__device__ __forceinline__ void hy4v_mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t hy4v_pack_bf2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__global__ void __launch_bounds__(128) k_hy4v_attn_tc(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                                      const bf16* __restrict__ V, bf16* __restrict__ out, int N) {
    __shared__ bf16 sQ[ATTN_BM][ATTN_HD_PAD];
    __shared__ bf16 sK[ATTN_BN][ATTN_HD_PAD];
    __shared__ bf16 sV[ATTN_BN][ATTN_HD_PAD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.y;
    const int q_cta = blockIdx.x * ATTN_BM;
    const int q_warp = warp * 16;
    const int r0 = lane >> 2;      // 0..7
    const int r1 = r0 + 8;         // 8..15
    const int c2 = (lane & 3) * 2; // 0, 2, 4, 6
    const float scale = 0.11785113019775793f;  // 1 / sqrt(72)

    // Load 64 x 72 Q tile (zero-padded to 80) into shared memory
    for (int idx = tid; idx < ATTN_BM * ATTN_HD_PAD; idx += 128) {
        const int r = idx / ATTN_HD_PAD;
        const int c = idx % ATTN_HD_PAD;
        const int gq = q_cta + r;
        sQ[r][c] = (gq < N && c < VHD) ? Q[size_t(gq) * VD + h * VHD + c] : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();

    // Preload Q fragments for this warp's 16 queries across all 5 k-tiles (5 x 16 = 80)
    uint32_t qa[5][4];
    #pragma unroll
    for (int kt = 0; kt < 5; ++kt) {
        const int k0 = kt * 16;
        qa[kt][0] = *reinterpret_cast<const uint32_t*>(&sQ[q_warp + r0][k0 + c2]);
        qa[kt][1] = *reinterpret_cast<const uint32_t*>(&sQ[q_warp + r1][k0 + c2]);
        qa[kt][2] = *reinterpret_cast<const uint32_t*>(&sQ[q_warp + r0][k0 + c2 + 8]);
        qa[kt][3] = *reinterpret_cast<const uint32_t*>(&sQ[q_warp + r1][k0 + c2 + 8]);
    }

    // Output accumulator: 9 n=8 tiles = 72 dims
    float oacc[9][4];
    #pragma unroll
    for (int t = 0; t < 9; ++t) { oacc[t][0] = oacc[t][1] = oacc[t][2] = oacc[t][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;

    for (int kv0 = 0; kv0 < N; kv0 += ATTN_BN) {
        const int valid_kv = min(ATTN_BN, N - kv0);
        for (int idx = tid; idx < ATTN_BN * ATTN_HD_PAD; idx += 128) {
            const int r = idx / ATTN_HD_PAD;
            const int c = idx % ATTN_HD_PAD;
            if (r < valid_kv && c < VHD) {
                const size_t off = size_t(kv0 + r) * VD + h * VHD + c;
                sK[r][c] = K[off];
                sV[r][c] = V[off];
            } else {
                sK[r][c] = __float2bfloat16_rn(0.0f);
                sV[r][c] = __float2bfloat16_rn(0.0f);
            }
        }
        __syncthreads();

        // 1. S = Q[16, 80] @ K[32, 80]^T -> 4 tiles of 16x8
        float S[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            S[nt][0] = S[nt][1] = S[nt][2] = S[nt][3] = 0.0f;
            const int kn = nt * 8 + r0;  // 0..31
            #pragma unroll
            for (int kt = 0; kt < 5; ++kt) {
                const int k0 = kt * 16;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[kn][k0 + c2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[kn][k0 + c2 + 8]);
                hy4v_mma16816(S[nt], qa[kt][0], qa[kt][1], qa[kt][2], qa[kt][3], b0, b1);
            }
            const int col0 = nt * 8 + c2;
            const int col1 = col0 + 1;
            S[nt][0] = (col0 < valid_kv) ? (S[nt][0] * scale) : -INFINITY;
            S[nt][1] = (col1 < valid_kv) ? (S[nt][1] * scale) : -INFINITY;
            S[nt][2] = (col0 < valid_kv) ? (S[nt][2] * scale) : -INFINITY;
            S[nt][3] = (col1 < valid_kv) ? (S[nt][3] * scale) : -INFINITY;
        }

        // 2. Online softmax across 32 keys (4 threads per query row: lane & 3)
        float rm0 = -INFINITY, rm1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            rm0 = fmaxf(rm0, fmaxf(S[nt][0], S[nt][1]));
            rm1 = fmaxf(rm1, fmaxf(S[nt][2], S[nt][3]));
        }
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 1));
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 2));
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 1));
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 2));

        const float nm0 = fmaxf(m0, rm0);
        const float nm1 = fmaxf(m1, rm1);
        const float a0 = __expf(m0 - nm0);
        const float a1 = __expf(m1 - nm1);

        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            oacc[t][0] *= a0; oacc[t][1] *= a0;
            oacc[t][2] *= a1; oacc[t][3] *= a1;
        }

        float rl0 = 0.0f, rl1 = 0.0f;
        uint32_t pf[2][4];  // 2 k-tiles of 16 along the 32-key axis
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p00 = __expf(S[nt][0] - nm0);
            const float p01 = __expf(S[nt][1] - nm0);
            const float p10 = __expf(S[nt][2] - nm1);
            const float p11 = __expf(S[nt][3] - nm1);
            rl0 += p00 + p01;
            rl1 += p10 + p11;
            const int pk = nt >> 1;      // 0 or 1
            const int half = nt & 1;     // 0 (a0,a1) or 1 (a2,a3)
            pf[pk][half * 2 + 0] = hy4v_pack_bf2(p00, p01);
            pf[pk][half * 2 + 1] = hy4v_pack_bf2(p10, p11);
        }
        rl0 += __shfl_xor_sync(0xffffffffu, rl0, 1);
        rl0 += __shfl_xor_sync(0xffffffffu, rl0, 2);
        rl1 += __shfl_xor_sync(0xffffffffu, rl1, 1);
        rl1 += __shfl_xor_sync(0xffffffffu, rl1, 2);
        l0 = l0 * a0 + rl0;
        l1 = l1 * a1 + rl1;
        m0 = nm0;
        m1 = nm1;

        // 3. O += P[16, 32] @ V[32, 72] -> 9 n=8 tiles x 2 k=16 tiles
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int dcol = t * 8 + r0;  // 0..71
            #pragma unroll
            for (int pk = 0; pk < 2; ++pk) {
                const int k0 = pk * 16;
                // Pack 2 BF16 values along the k (key-row) axis into uint32_t for row.col B-fragment
                __nv_bfloat162 v01 = make_bfloat162(sV[k0 + c2][dcol], sV[k0 + c2 + 1][dcol]);
                __nv_bfloat162 v23 = make_bfloat162(sV[k0 + c2 + 8][dcol], sV[k0 + c2 + 9][dcol]);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&v01);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&v23);
                hy4v_mma16816(oacc[t], pf[pk][0], pf[pk][1], pf[pk][2], pf[pk][3], b0, b1);
            }
        }
        __syncthreads();
    }

    const float inv0 = 1.0f / l0;
    const float inv1 = 1.0f / l1;
    const int gq0 = q_cta + q_warp + r0;
    const int gq1 = q_cta + q_warp + r1;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int d0 = t * 8 + c2;
        if (gq0 < N) {
            out[size_t(gq0) * VD + h * VHD + d0]     = __float2bfloat16_rn(oacc[t][0] * inv0);
            out[size_t(gq0) * VD + h * VHD + d0 + 1] = __float2bfloat16_rn(oacc[t][1] * inv0);
        }
        if (gq1 < N) {
            out[size_t(gq1) * VD + h * VHD + d0]     = __float2bfloat16_rn(oacc[t][2] * inv1);
            out[size_t(gq1) * VD + h * VHD + d0 + 1] = __float2bfloat16_rn(oacc[t][3] * inv1);
        }
    }
}

enum Epilogue { E_BIAS = 0, E_RES = 1, E_GELU_EXACT = 2 };

__global__ void __launch_bounds__(256) k_hy4v_epilogue(const float* __restrict__ acc, const bf16* __restrict__ bias,
                                                       bf16* __restrict__ out, int rows, int cols, int mode) {
    const size_t total = size_t(rows) * size_t(cols);
    for (size_t idx = size_t(blockIdx.x) * 256 + threadIdx.x; idx < total; idx += size_t(gridDim.x) * 256) {
        const int c = static_cast<int>(idx % size_t(cols));
        float v = acc[idx] + __bfloat162float(bias[c]);
        if (mode == E_RES) {
            v = __bfloat162float(out[idx]) + __bfloat162float(__float2bfloat16_rn(v));
        } else if (mode == E_GELU_EXACT) {
            const float x = __bfloat162float(__float2bfloat16_rn(v));
            v = 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
        }
        out[idx] = __float2bfloat16_rn(v);
    }
}

// Gather 2x2 spatial neighborhoods in PyTorch Conv2d(1152, 4608, k=2, s=2) layout: [C=1152, ky=2, kx=2] = 4608.
// Input y: [vit_h, vit_w, 1152]. Output dst: [(vit_h/2)*(vit_w/2), 4608].
// Grid: ((vit_h/2) * (vit_w/2)), Block: 256.
__global__ void __launch_bounds__(256) k_hy4v_merge_gather(const bf16* __restrict__ y, bf16* __restrict__ dst,
                                                           int vit_h, int vit_w) {
    const int mh = vit_h >> 1;
    const int mw = vit_w >> 1;
    const int m = blockIdx.x;
    if (m >= mh * mw) return;
    const int mr = m / mw;
    const int mc = m % mw;
    const bf16* r00 = y + size_t((2 * mr + 0) * vit_w + (2 * mc + 0)) * VD;
    const bf16* r01 = y + size_t((2 * mr + 0) * vit_w + (2 * mc + 1)) * VD;
    const bf16* r10 = y + size_t((2 * mr + 1) * vit_w + (2 * mc + 0)) * VD;
    const bf16* r11 = y + size_t((2 * mr + 1) * vit_w + (2 * mc + 1)) * VD;
    bf16* out = dst + size_t(m) * VMRG;
    for (int c = threadIdx.x; c < VD; c += 256) {
        out[c * 4 + 0] = r00[c];
        out[c * 4 + 1] = r01[c];
        out[c * 4 + 2] = r10[c];
        out[c * 4 + 3] = r11[c];
    }
}

// Scatter merged rows [mh, mw, 3072] + image_newline [3072] at end of each row -> [mh * (mw + 1), 3072].
// Grid: (mh * (mw + 1)), Block: 256.
__global__ void __launch_bounds__(256) k_hy4v_append_newline(const bf16* __restrict__ proj,
                                                             const bf16* __restrict__ newline,
                                                             bf16* __restrict__ out, int mh, int mw) {
    const int row_out = blockIdx.x;
    const int stride = mw + 1;
    if (row_out >= mh * stride) return;
    const int mr = row_out / stride;
    const int mc = row_out % stride;
    const bf16* src = (mc < mw) ? (proj + size_t(mr * mw + mc) * VOUT) : newline;
    bf16* dst = out + size_t(row_out) * VOUT;
    for (int d = threadIdx.x; d < VOUT; d += 256) {
        dst[d] = src[d];
    }
}

}  // namespace vis

class Hy4Vision {
public:
    using bf16 = vis::bf16;
    bool log = false;

    bool ready() const { return blob_.dev != nullptr; }
    size_t device_bytes() const { return blob_.bytes + ws_bytes_ + pool_bytes_ + stage_bytes_; }
    const std::string& id() const { return id_; }

    void load(const std::string& path, int pool_rows) {
        id_ = g6vis::file_id(path);
        blob_ = g6vis::load_blob(path, "hy4 vision");
        auto G = [&](const std::string& name, const std::vector<int64_t>& shape) -> const bf16* {
            return static_cast<const bf16*>(blob_.get(name, "BF16", shape, path, "hy4 vision"));
        };
        w_.patch_w = G("vit.embeddings.patch_embedding.weight", {vis::VD, 3, hy4v::PATCH, hy4v::PATCH});
        w_.patch_b = G("vit.embeddings.patch_embedding.bias", {vis::VD});
        w_.pos = G("vit.embeddings.position_embedding.weight", {vis::VGRID * vis::VGRID + 1, vis::VD});
        w_.layers.resize(vis::VL);
        for (int i = 0; i < vis::VL; ++i) {
            const std::string p = "vit.layers." + std::to_string(i) + ".";
            auto& y = w_.layers[size_t(i)];
            y.ln1_w = G(p + "input_layernorm.weight", {vis::VD});
            y.ln1_b = G(p + "input_layernorm.bias", {vis::VD});
            y.q_w   = G(p + "self_attn.q_proj.weight", {vis::VD, vis::VD});
            y.q_b   = G(p + "self_attn.q_proj.bias", {vis::VD});
            y.k_w   = G(p + "self_attn.k_proj.weight", {vis::VD, vis::VD});
            y.k_b   = G(p + "self_attn.k_proj.bias", {vis::VD});
            y.v_w   = G(p + "self_attn.v_proj.weight", {vis::VD, vis::VD});
            y.v_b   = G(p + "self_attn.v_proj.bias", {vis::VD});
            y.o_w   = G(p + "self_attn.o_proj.weight", {vis::VD, vis::VD});
            y.o_b   = G(p + "self_attn.o_proj.bias", {vis::VD});
            y.ln2_w = G(p + "post_attention_layernorm.weight", {vis::VD});
            y.ln2_b = G(p + "post_attention_layernorm.bias", {vis::VD});
            y.fc1_w = G(p + "mlp.dense_h_to_4h.weight", {vis::VMLP, vis::VD});
            y.fc1_b = G(p + "mlp.dense_h_to_4h.bias", {vis::VMLP});
            y.fc2_w = G(p + "mlp.dense_4h_to_h.weight", {vis::VD, vis::VMLP});
            y.fc2_b = G(p + "mlp.dense_4h_to_h.bias", {vis::VD});
        }
        w_.mrg_norm_w = G("vit.perceive.norm.weight", {vis::VD});
        w_.mrg_p0_w   = G("vit.perceive.proj.0.weight", {vis::VMRG, vis::VD, hy4v::MERGE, hy4v::MERGE});
        w_.mrg_p0_b   = G("vit.perceive.proj.0.bias", {vis::VMRG});
        w_.mrg_p2_w   = G("vit.perceive.proj.2.weight", {vis::VOUT, vis::VMRG, 1, 1});
        w_.mrg_p2_b   = G("vit.perceive.proj.2.bias", {vis::VOUT});
        w_.newline    = G("vit.perceive.image_newline", {vis::VOUT});

        HY4V_BLAS(cublasCreate(&blas_));
        if (pool_rows > 0) {
            pool_rows_ = pool_rows;
            pool_bytes_ = size_t(pool_rows) * vis::VOUT * sizeof(bf16) + size_t(pool_rows) * sizeof(int);
            HY4V_CUDA(cudaMalloc(&pool_, size_t(pool_rows) * vis::VOUT * sizeof(bf16)));
            HY4V_CUDA(cudaMalloc(&d_row_map_, size_t(pool_rows) * sizeof(int)));
        }
    }

    // Encode one smart-resized RGB image [vit_h*16, vit_w*16, 3] into out [(vit_h/2)*(vit_w/2 + 1), 3072] BF16.
    void encode(const uint8_t* rgb, int vit_h, int vit_w, bf16* out, cudaStream_t st) {
        const int span = hy4v::span_of(vit_h, vit_w);
        if (span <= 0) throw std::runtime_error("hy4 vision: invalid grid");
        const int N = vit_h * vit_w;
        const int mh = vit_h / hy4v::MERGE;
        const int mw = vit_w / hy4v::MERGE;
        const int cells = mh * mw;
        ensure_ws(N, cells);

        const size_t px_bytes = size_t(vit_h) * hy4v::PATCH * size_t(vit_w) * hy4v::PATCH * 3;
        HY4V_CUDA(cudaMemcpyAsync(d_px_, rgb, px_bytes, cudaMemcpyHostToDevice, st));

        // 1. Patchify + Patch Embed + Interpolated Learned Position Embedding
        vis::k_hy4v_patchify<<<N, 256, 0, st>>>(d_px_, vit_w, patches_);
        linear(patches_, w_.patch_w, w_.patch_b, x_, N, vis::VD, vis::VPK, vis::E_BIAS, st);
        vis::k_hy4v_pos_add<<<N, 256, 0, st>>>(x_, w_.pos, vit_h, vit_w);

        // 2. 27 ViT layers
        const int mlp_rows = std::max(1, static_cast<int>(vis::ACC_FLOATS / vis::VMLP));
        const dim3 attn_grid((N + vis::ATTN_BM - 1) / vis::ATTN_BM, vis::VHEADS);
        for (int i = 0; i < vis::VL; ++i) {
            const auto& y = w_.layers[size_t(i)];
            vis::k_hy4v_layernorm<<<N, 128, 0, st>>>(x_, y.ln1_w, y.ln1_b, t_);
            linear(t_, y.q_w, y.q_b, q_, N, vis::VD, vis::VD, vis::E_BIAS, st);
            linear(t_, y.k_w, y.k_b, k_, N, vis::VD, vis::VD, vis::E_BIAS, st);
            linear(t_, y.v_w, y.v_b, v_, N, vis::VD, vis::VD, vis::E_BIAS, st);
            vis::k_hy4v_attn_tc<<<attn_grid, 128, 0, st>>>(q_, k_, v_, t_, N);
            linear(t_, y.o_w, y.o_b, x_, N, vis::VD, vis::VD, vis::E_RES, st);

            vis::k_hy4v_layernorm<<<N, 128, 0, st>>>(x_, y.ln2_w, y.ln2_b, t_);
            for (int r0 = 0; r0 < N; r0 += mlp_rows) {
                const int rows = std::min(mlp_rows, N - r0);
                linear(t_ + size_t(r0) * vis::VD, y.fc1_w, y.fc1_b, m_, rows, vis::VMLP, vis::VD, vis::E_GELU_EXACT, st);
                linear(m_, y.fc2_w, y.fc2_b, x_ + size_t(r0) * vis::VD, rows, vis::VD, vis::VMLP, vis::E_RES, st);
            }
        }

        // 3. Patch Merger (vit.perceive): RMSNorm -> 2x2 Conv2d -> GELU -> 1x1 Conv2d -> append image_newline
        vis::k_hy4v_rmsnorm<<<N, 128, 0, st>>>(x_, w_.mrg_norm_w, t_);
        vis::k_hy4v_merge_gather<<<cells, 256, 0, st>>>(t_, mrg_in_, vit_h, vit_w);
        const int mrg_rows = std::max(1, static_cast<int>(vis::ACC_FLOATS / vis::VMRG));
        for (int r0 = 0; r0 < cells; r0 += mrg_rows) {
            const int rows = std::min(mrg_rows, cells - r0);
            linear(mrg_in_ + size_t(r0) * vis::VMRG, w_.mrg_p0_w, w_.mrg_p0_b, m_, rows, vis::VMRG, vis::VMRG, vis::E_GELU_EXACT, st);
            linear(m_, w_.mrg_p2_w, w_.mrg_p2_b, mrg_out_ + size_t(r0) * vis::VOUT, rows, vis::VOUT, vis::VMRG, vis::E_BIAS, st);
        }
        vis::k_hy4v_append_newline<<<span, 256, 0, st>>>(mrg_out_, w_.newline, out, mh, mw);
        HY4V_CUDA(cudaGetLastError());
        HY4V_CUDA(cudaStreamSynchronize(st));
    }

    int stage(const std::vector<g6rt::ImageIn>& images, int first, int rows, cudaStream_t st,
              int rank, unsigned long long request) {
        if (images.empty()) return 0;
        if (!ready()) throw std::runtime_error("hy4 vision: image request without vision tower");
        if (rows < 1 || rows > pool_rows_) throw std::runtime_error("hy4 vision: launch rows exceed pool");
        map_.assign(size_t(rows), -1);
        int pool_row = 0;
        for (size_t k = 0; k < images.size(); ++k) {
            const g6rt::ImageIn& im = images[k];
            const int a = std::max(im.pos, first);
            const int b = std::min(im.pos + im.span, first + rows);
            if (a >= b) continue;
            if (im.span != hy4v::span_of(im.vit_h, im.vit_w) || !im.rgb ||
                im.rgb->size() != size_t(im.vit_h) * im.vit_w * hy4v::PATCH * hy4v::PATCH * 3)
                throw std::runtime_error("hy4 vision: image span or RGB bytes mismatch grid");
            ensure_stage(im.span);
            if (!(last_ == im.rgb && last_h_ == im.vit_h && last_w_ == im.vit_w)) {
                const auto t0 = std::chrono::steady_clock::now();
                last_.reset();
                encode(im.rgb->data(), im.vit_h, im.vit_w, stage_, st);
                last_ = im.rgb;
                last_h_ = im.vit_h;
                last_w_ = im.vit_w;
                if (log && rank == 0) {
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    std::fprintf(stderr, "# hy4 vision: request %llu image %zu: %dx%d patches, %d tokens, tower %.1f ms\n",
                                 request, k, im.vit_h, im.vit_w, im.span, ms);
                }
            }
            HY4V_CUDA(cudaMemcpyAsync(pool_ + size_t(pool_row) * vis::VOUT,
                                      stage_ + size_t(a - im.pos) * vis::VOUT,
                                      size_t(b - a) * vis::VOUT * sizeof(bf16),
                                      cudaMemcpyDeviceToDevice, st));
            for (int p = a; p < b; ++p) map_[size_t(p - first)] = pool_row + (p - a);
            pool_row += (b - a);
        }
        if (pool_row > 0) {
            HY4V_CUDA(cudaMemcpyAsync(d_row_map_, map_.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
        }
        return pool_row;
    }

    const bf16* pool() const { return pool_; }
    const int* row_map() const { return d_row_map_; }

    Hy4Vision() = default;
    Hy4Vision(const Hy4Vision&) = delete;
    Hy4Vision& operator=(const Hy4Vision&) = delete;
    ~Hy4Vision() {
        if (blas_) cublasDestroy(blas_);
        if (ws_) cudaFree(ws_);
        if (pool_) cudaFree(pool_);
        if (d_row_map_) cudaFree(d_row_map_);
        if (stage_) cudaFree(stage_);
        if (blob_.dev) cudaFree(blob_.dev);
    }

private:
    struct LayerW {
        const bf16 *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
        const bf16 *ln2_w, *ln2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    };
    struct Weights {
        const bf16 *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr;
        const bf16 *mrg_norm_w = nullptr, *mrg_p0_w = nullptr, *mrg_p0_b = nullptr;
        const bf16 *mrg_p2_w = nullptr, *mrg_p2_b = nullptr, *newline = nullptr;
        std::vector<LayerW> layers;
    };

    g6vis::Blob blob_{};
    Weights w_{};
    std::string id_;
    cublasHandle_t blas_ = nullptr;

    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    int ws_n_ = 0;
    uint8_t* d_px_ = nullptr;
    bf16 *patches_ = nullptr, *x_ = nullptr, *t_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
    bf16 *m_ = nullptr, *mrg_in_ = nullptr, *mrg_out_ = nullptr;
    float* acc_ = nullptr;

    bf16* pool_ = nullptr;
    int* d_row_map_ = nullptr;
    int pool_rows_ = 0;
    size_t pool_bytes_ = 0;
    std::vector<int> map_;

    bf16* stage_ = nullptr;
    int stage_rows_ = 0;
    size_t stage_bytes_ = 0;
    std::shared_ptr<const std::vector<uint8_t>> last_;
    int last_h_ = 0, last_w_ = 0;

    static size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

    void ensure_ws(int N, int cells) {
        if (N <= ws_n_ && ws_) return;
        if (ws_) { cudaFree(ws_); ws_ = nullptr; ws_bytes_ = 0; ws_n_ = 0; }
        const size_t px_b   = align256(size_t(N) * vis::VPK);
        const size_t pk_b   = align256(size_t(N) * vis::VPK * sizeof(bf16));
        const size_t d_b    = align256(size_t(N) * vis::VD * sizeof(bf16));
        const int max_mrows = std::max(1, static_cast<int>(vis::ACC_FLOATS / vis::VMLP));
        const size_t m_b    = align256(size_t(max_mrows) * vis::VMRG * sizeof(bf16));
        const size_t mi_b   = align256(size_t(cells) * vis::VMRG * sizeof(bf16));
        const size_t mo_b   = align256(size_t(cells) * vis::VOUT * sizeof(bf16));
        const size_t acc_b  = align256(vis::ACC_FLOATS * sizeof(float));
        ws_bytes_ = px_b + pk_b + 5 * d_b + m_b + mi_b + mo_b + acc_b;
        HY4V_CUDA(cudaMalloc(&ws_, ws_bytes_));
        char* p = static_cast<char*>(ws_);
        auto take = [&](size_t b) -> void* { void* r = p; p += b; return r; };
        d_px_    = static_cast<uint8_t*>(take(px_b));
        patches_ = static_cast<bf16*>(take(pk_b));
        x_       = static_cast<bf16*>(take(d_b));
        t_       = static_cast<bf16*>(take(d_b));
        q_       = static_cast<bf16*>(take(d_b));
        k_       = static_cast<bf16*>(take(d_b));
        v_       = static_cast<bf16*>(take(d_b));
        m_       = static_cast<bf16*>(take(m_b));
        mrg_in_  = static_cast<bf16*>(take(mi_b));
        mrg_out_ = static_cast<bf16*>(take(mo_b));
        acc_     = static_cast<float*>(take(acc_b));
        ws_n_    = N;
    }

    void ensure_stage(int span) {
        if (span <= stage_rows_ && stage_) return;
        if (stage_) { cudaFree(stage_); stage_ = nullptr; stage_bytes_ = 0; stage_rows_ = 0; }
        stage_bytes_ = size_t(span) * vis::VOUT * sizeof(bf16);
        HY4V_CUDA(cudaMalloc(&stage_, stage_bytes_));
        stage_rows_ = span;
    }

    void linear(const bf16* A, const bf16* W, const bf16* bias, bf16* out,
                int rows, int cols, int kdim, vis::Epilogue ep, cudaStream_t st) {
        HY4V_BLAS(cublasSetStream(blas_, st));
        const int max_r = std::max(1, static_cast<int>(vis::ACC_FLOATS / size_t(cols)));
        const float alpha = 1.0f, beta = 0.0f;
        for (int r0 = 0; r0 < rows; r0 += max_r) {
            const int r = std::min(max_r, rows - r0);
            HY4V_BLAS(cublasGemmEx(
                blas_, CUBLAS_OP_T, CUBLAS_OP_N,
                cols, r, kdim, &alpha,
                W, CUDA_R_16BF, kdim,
                A + size_t(r0) * kdim, CUDA_R_16BF, kdim,
                &beta,
                acc_, CUDA_R_32F, cols,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            const size_t tot = size_t(r) * size_t(cols);
            const unsigned grid = static_cast<unsigned>(std::min<size_t>((tot + 255) / 256, 65535));
            vis::k_hy4v_epilogue<<<grid, 256, 0, st>>>(acc_, bias, out + size_t(r0) * cols, r, cols, int(ep));
        }
    }
};

}  // namespace hy4rt
