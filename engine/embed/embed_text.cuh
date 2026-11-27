// Native Qwen3-VL text prefill-only CUDA transformer forward + last-token pooling + MRL + FP32 L2 norm.
// Ephemeral K/V scratch (0 bytes persistent KV cache), packed variable-length `cu_seqlens` batches,
// Qwen3VLTextRMSNorm (plain w, eps=1e-6), QK-norm, interleaved 3D M-RoPE [24, 20, 20] (theta=5e6),
// DeepStack residual scatter-add after LLM layers 0..2, and optional per-kernel CUDA event profiler.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "embed_config.h"
#include "embed_gemm.cuh"
#include "embed_weights.h"

namespace g6emb {

// ------------------------------------------------------------------------------------------------ Text CUDA kernels

__device__ __forceinline__ float emb_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float emb_block_sum256(float v, float* red) {
    v = emb_warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float tot = 0.f;
    if (threadIdx.x < 32) {
        tot = (threadIdx.x < 8) ? red[threadIdx.x] : 0.f;
        tot = emb_warp_sum(tot);
        if (threadIdx.x == 0) red[0] = tot;
    }
    __syncthreads();
    return red[0];
}

// Qwen3VLTextRMSNorm: y = w * bf16(x_f32 * rsqrt(mean(x_f32^2) + eps)). Grid: (M), Block: 256.
// Single-pass uint4 (8 BF16/thread) register-cached fast path for H <= 4096 (covers 2B H=2048 and 8B H=4096),
// eliminating the second DRAM read of x.
__global__ void __launch_bounds__(256) k_q3_rmsnorm(
    const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ y,
    int M, int H, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;
    const bf16* xr = x + size_t(m) * H;
    bf16* yr = y + size_t(m) * H;

    if (H <= 4096 && (H & 7) == 0) {
        // Up to 2 uint4 loads (16 BF16) per thread -> stays in registers, single DRAM read of x!
        uint4 xr_buf[2];
        int nvec = 0;
        float ss = 0.f;
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int d = (threadIdx.x + pass * 256) * 8;
            if (d < H) {
                const uint4 r = *reinterpret_cast<const uint4*>(xr + d);
                xr_buf[pass] = r;
                nvec = pass + 1;
                const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
                const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
                const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.z));
                const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.w));
                ss += (a0.x * a0.x + a0.y * a0.y) + (a1.x * a1.x + a1.y * a1.y) +
                      (a2.x * a2.x + a2.y * a2.y) + (a3.x * a3.x + a3.y * a3.y);
            }
        }
        const float rs = rsqrtf(emb_block_sum256(ss, red) / float(H) + eps);
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            if (pass < nvec) {
                const int d = (threadIdx.x + pass * 256) * 8;
                const uint4 r = xr_buf[pass];
                const uint4 wr = *reinterpret_cast<const uint4*>(w + d);
                const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
                const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
                const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.z));
                const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.w));
                const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.x));
                const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.y));
                const float2 w2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.z));
                const float2 w3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.w));
                auto norm_mul = [&](float v, float wt) {
                    return __bfloat162float(__float2bfloat16_rn(v * rs)) * wt;
                };
                *reinterpret_cast<uint4*>(yr + d) = make_uint4(
                    emb_pack_bf2(norm_mul(a0.x, w0.x), norm_mul(a0.y, w0.y)),
                    emb_pack_bf2(norm_mul(a1.x, w1.x), norm_mul(a1.y, w1.y)),
                    emb_pack_bf2(norm_mul(a2.x, w2.x), norm_mul(a2.y, w2.y)),
                    emb_pack_bf2(norm_mul(a3.x, w3.x), norm_mul(a3.y, w3.y)));
            }
        }
        return;
    }

    float ss = 0.f;
    for (int d = threadIdx.x * 4; d < H; d += 256 * 4) {
        const uint2 r = *reinterpret_cast<const uint2*>(xr + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        ss += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    }
    const float rs = rsqrtf(emb_block_sum256(ss, red) / float(H) + eps);
    for (int d = threadIdx.x * 4; d < H; d += 256 * 4) {
        const uint2 r = *reinterpret_cast<const uint2*>(xr + d);
        const uint2 wr = *reinterpret_cast<const uint2*>(w + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        const float2 wa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.x));
        const float2 wb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.y));
        const float o0 = __bfloat162float(__float2bfloat16_rn(a.x * rs)) * wa.x;
        const float o1 = __bfloat162float(__float2bfloat16_rn(a.y * rs)) * wa.y;
        const float o2 = __bfloat162float(__float2bfloat16_rn(b.x * rs)) * wb.x;
        const float o3 = __bfloat162float(__float2bfloat16_rn(b.y * rs)) * wb.y;
        *reinterpret_cast<uint2*>(yr + d) = make_uint2(emb_pack_bf2(o0, o1), emb_pack_bf2(o2, o3));
    }
}

// Qwen3-VL interleaved 3D M-RoPE selector for head_dim=128 (64 pairs), mrope_section=[24, 20, 20]:
//   pair_idx % 3 == 1 && pair_idx < s1*3 -> H (thw.y)
//   pair_idx % 3 == 2 && pair_idx < s2*3 -> W (thw.z)
//   otherwise                            -> T (thw.x)
__device__ __forceinline__ int q3_mrope_coord(int3 thw, int pair_idx, int s1_lim, int s2_lim) {
    const int rem = pair_idx % 3;
    if (rem == 1 && pair_idx < s1_lim) return thw.y;
    if (rem == 2 && pair_idx < s2_lim) return thw.z;
    return thw.x;
}

// In-place per-head QK-RMSNorm (plain w, eps=1e-6) + full-head interleaved 3D M-RoPE on Q[M, nq, hd] or K[M, nkv, hd].
// Grid: (M * nheads), Block: 64 (each thread handles pair (j, j + half_hd) for head_dim=128 -> half_hd=64).
__global__ void __launch_bounds__(64) k_q3_head_norm_mrope(
    bf16* __restrict__ qk, const bf16* __restrict__ norm_w, const int3* __restrict__ mrope_thw,
    int M, int nheads, int hd, float eps, float theta, int s1_lim, int s2_lim) {
    __shared__ float sm[2];
    const int idx = blockIdx.x;
    if (idx >= M * nheads) return;
    const int m = idx / nheads;
    const int j = threadIdx.x;
    const int half = hd >> 1;
    if (j >= half) return;

    bf16* row = qk + size_t(idx) * hd;
    float a = __bfloat162float(row[j]);
    float b = __bfloat162float(row[j + half]);
    float ss = a * a + b * b;
    ss = emb_warp_sum(ss);
    if ((j & 31) == 0) sm[j >> 5] = ss;
    __syncthreads();
    const float tot = sm[0] + (half > 32 ? sm[1] : 0.f);
    const float rs = rsqrtf(tot / float(hd) + eps);

    // Per-head Qwen3VLTextRMSNorm rounded to BF16 before RoPE
    a = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(a * rs)) * __bfloat162float(norm_w[j])));
    b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(b * rs)) * __bfloat162float(norm_w[j + half])));

    const int pos = q3_mrope_coord(mrope_thw[m], j, s1_lim, s2_lim);
    const float inv_freq = 1.0f / powf(theta, static_cast<float>(2 * j) / static_cast<float>(hd));
    const float ang = static_cast<float>(pos) * inv_freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    row[j] = __float2bfloat16_rn(a * cs - b * sn);
    row[j + half] = __float2bfloat16_rn(b * cs + a * sn);
}

// Fused Q + K per-head RMSNorm + interleaved 3D M-RoPE with precomputed inv_freq[hd/2] (zero powf per token!).
// Grid: (M, (nq + nkv + 3) / 4), Block: 256 (4 heads per CTA, 64 threads per head for hd=128).
__global__ void __launch_bounds__(256) k_q3_qk_norm_mrope_fused(
    bf16* __restrict__ q, bf16* __restrict__ k,
    const bf16* __restrict__ q_norm_w, const bf16* __restrict__ k_norm_w,
    const int3* __restrict__ mrope_thw, const float* __restrict__ inv_freq_tbl,
    int M, int nq, int nkv, int hd, float eps, int s1_lim, int s2_lim) {
    __shared__ float sm[8]; // 2 warps per head * 4 heads per CTA
    const int m = blockIdx.x;
    const int h_sub = threadIdx.x >> 6; // 0..3
    const int j = threadIdx.x & 63;     // 0..63
    const int h_glob = blockIdx.y * 4 + h_sub;
    const int half = hd >> 1;
    if (m >= M || h_glob >= nq + nkv || j >= half) return;

    const bool is_k = (h_glob >= nq);
    const int h = is_k ? (h_glob - nq) : h_glob;
    bf16* row = is_k ? (k + (size_t(m) * nkv + h) * hd) : (q + (size_t(m) * nq + h) * hd);
    const bf16* nw = is_k ? k_norm_w : q_norm_w;

    float a = __bfloat162float(row[j]);
    float b = __bfloat162float(row[j + half]);
    float ss = a * a + b * b;
    ss = emb_warp_sum(ss);
    const int warp_id = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) sm[warp_id] = ss;
    __syncthreads();
    const float tot = sm[h_sub * 2] + (half > 32 ? sm[h_sub * 2 + 1] : 0.f);
    const float rs = rsqrtf(tot / float(hd) + eps);

    a = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(a * rs)) * __bfloat162float(nw[j])));
    b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(b * rs)) * __bfloat162float(nw[j + half])));

    const int pos = q3_mrope_coord(mrope_thw[m], j, s1_lim, s2_lim);
    float sn, cs;
    __sincosf(static_cast<float>(pos) * inv_freq_tbl[j], &sn, &cs);
    row[j] = __float2bfloat16_rn(a * cs - b * sn);
    row[j + half] = __float2bfloat16_rn(b * cs + a * sn);
}

// Packed variable-length causal GQA FlashAttention (head_dim=128, ephemeral K/V scratch, zero persistent KV cache).
// Grid: (ceil(M / 4), nq), Block: 128 (4 warps; 1 warp = 1 query row, each lane holds 4 dims: lane + 0, 32, 64, 96).
// Baseline clean implementation — prime target for Phase B Tensor-Core FlashAttention hill-climbing!
constexpr int TEXT_FA_QW = 4;
constexpr int TEXT_FA_KT = 32;

__global__ void __launch_bounds__(128) k_q3_causal_gqa_attn(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ seq_of_row,
    int M, int nq, int nkv, int hd, bool causal) {
    __shared__ bf16 sK[TEXT_FA_KT][128];
    __shared__ bf16 sV[TEXT_FA_KT][128];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hq = blockIdx.y;
    const int hkv = hq / (nq / nkv);
    const int q_row = blockIdx.x * TEXT_FA_QW + warp;
    const float scale = rsqrtf(float(hd));

    const int q_base_row = blockIdx.x * TEXT_FA_QW;
    const int q_end_row = min(M, q_base_row + TEXT_FA_QW);
    // Determine contiguous K range needed by any active warp in this CTA.
    const int first_seq = (q_base_row < M) ? seq_of_row[q_base_row] : 0;
    const int last_seq = (q_end_row > 0) ? seq_of_row[q_end_row - 1] : 0;
    const int cta_k_start = (q_base_row < M) ? cu_seqlens[first_seq] : 0;
    const int cta_k_end = (q_base_row < M) ? cu_seqlens[last_seq + 1] : 0;

    int my_start = 0, my_end = 0;
    float qv[4] = {0.f, 0.f, 0.f, 0.f};
    if (q_row < M) {
        const int s = seq_of_row[q_row];
        my_start = cu_seqlens[s];
        my_end = causal ? (q_row + 1) : cu_seqlens[s + 1];
        const bf16* qp = Q + (size_t(q_row) * nq + hq) * hd;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + i * 32 < hd) qv[i] = __bfloat162float(qp[lane + i * 32]);
        }
    }

    float ov[4] = {0.f, 0.f, 0.f, 0.f};
    float m_max = -INFINITY;
    float l_sum = 0.f;

    for (int k0 = cta_k_start; k0 < cta_k_end; k0 += TEXT_FA_KT) {
        const int kt = min(TEXT_FA_KT, cta_k_end - k0);
        // Load kt x hd K and V tiles into shared memory
        for (int idx = tid; idx < kt * (hd >> 3); idx += 128) {
            const int r = idx / (hd >> 3);
            const int c = (idx % (hd >> 3)) * 8;
            const size_t off = (size_t(k0 + r) * nkv + hkv) * hd + c;
            *reinterpret_cast<uint4*>(&sK[r][c]) = *reinterpret_cast<const uint4*>(K + off);
            *reinterpret_cast<uint4*>(&sV[r][c]) = *reinterpret_cast<const uint4*>(V + off);
        }
        __syncthreads();

        if (q_row < M && k0 < my_end && k0 + kt > my_start) {
            const int r_lo = max(0, my_start - k0);
            const int r_hi = min(kt, my_end - k0);
            for (int r = r_lo; r < r_hi; ++r) {
                float dot = 0.f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (lane + i * 32 < hd) dot = fmaf(qv[i], __bfloat162float(sK[r][lane + i * 32]), dot);
                }
                dot = emb_warp_sum(dot);
                const float sc = dot * scale;
                const float nmax = fmaxf(m_max, sc);
                const float alpha = __expf(m_max - nmax);
                const float p = __expf(sc - nmax);
                m_max = nmax;
                l_sum = l_sum * alpha + p;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (lane + i * 32 < hd) ov[i] = ov[i] * alpha + p * __bfloat162float(sV[r][lane + i * 32]);
                }
            }
        }
        __syncthreads();
    }

    if (q_row < M) {
        const float inv = (l_sum > 0.f) ? (1.0f / l_sum) : 0.f;
        bf16* op = O + (size_t(q_row) * nq + hq) * hd;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + i * 32 < hd) op[lane + i * 32] = __float2bfloat16_rn(ov[i] * inv);
        }
    }
}

// Tensor-Core BF16 FlashAttention-2 for packed variable-length causal/bidirectional GQA (head_dim = 128).
// BQ = 64 (4 warps x 16 Q rows), BK = 64 KV rows, Q kept 100% in registers, online softmax via 4-lane quad shuffles.
// Grid: (num_q_tiles, nq), Block: 128.
constexpr int TEXT_TC_BQ = 64;
constexpr int TEXT_TC_BK = 64;

__global__ void __launch_bounds__(128, 2) k_q3_flash_attn_tc(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O,
    const int* __restrict__ cu_seqlens,
    const int2* __restrict__ q_tiles,
    int num_q_tiles, int nq, int nkv, bool causal) {
    __shared__ __align__(16) bf16 sK[TEXT_TC_BK][136];
    __shared__ __align__(16) bf16 sV[TEXT_TC_BK][136];

    const int tile_idx = blockIdx.x;
    if (tile_idx >= num_q_tiles) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r = lane >> 2;
    const int c = (lane & 3) * 2;

    const int2 qt = q_tiles[tile_idx];
    const int seq = qt.x;
    const int q_local_base = qt.y;
    const int s_start = cu_seqlens[seq];
    const int seq_len = cu_seqlens[seq + 1] - s_start;
    const int hq = blockIdx.y;
    const int hkv = hq / (nq / nkv);
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    const int q_row0 = q_local_base + warp * 16 + r;
    const int q_row1 = q_row0 + 8;

    u32 q_frag[8][4];
    const bf16* qp0 = (q_row0 < seq_len) ? (Q + (size_t(s_start + q_row0) * nq + hq) * 128) : nullptr;
    const bf16* qp1 = (q_row1 < seq_len) ? (Q + (size_t(s_start + q_row1) * nq + hq) * 128) : nullptr;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int col = ks * 16 + c;
        q_frag[ks][0] = qp0 ? *reinterpret_cast<const u32*>(qp0 + col) : 0u;
        q_frag[ks][1] = qp1 ? *reinterpret_cast<const u32*>(qp1 + col) : 0u;
        q_frag[ks][2] = qp0 ? *reinterpret_cast<const u32*>(qp0 + col + 8) : 0u;
        q_frag[ks][3] = qp1 ? *reinterpret_cast<const u32*>(qp1 + col + 8) : 0u;
    }

    float O_acc[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        O_acc[dt][0] = O_acc[dt][1] = O_acc[dt][2] = O_acc[dt][3] = 0.f;
    }
    float m_max0 = -1e30f, m_max1 = -1e30f;
    float l_sum0 = 0.f, l_sum1 = 0.f;

    const int kv_end = causal ? min(seq_len, q_local_base + TEXT_TC_BQ) : seq_len;

    for (int k0 = 0; k0 < kv_end; k0 += TEXT_TC_BK) {
        const int kt = min(TEXT_TC_BK, kv_end - k0);
        // 64 * 128 / 8 = 1024 uint4 loads -> 8 per thread
        #pragma unroll
        for (int idx = tid; idx < TEXT_TC_BK * 16; idx += 128) {
            const int kr = idx >> 4;
            const int kc = (idx & 15) * 8;
            if (kr < kt) {
                const size_t off = (size_t(s_start + k0 + kr) * nkv + hkv) * 128 + kc;
                *reinterpret_cast<uint4*>(&sK[kr][kc]) = *reinterpret_cast<const uint4*>(K + off);
                *reinterpret_cast<uint4*>(&sV[kr][kc]) = *reinterpret_cast<const uint4*>(V + off);
            } else {
                *reinterpret_cast<uint4*>(&sK[kr][kc]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sV[kr][kc]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Skip warps completely before the causal boundary of this KV tile
        const int warp_q_hi = q_local_base + warp * 16 + 15;
        if (!causal || k0 <= warp_q_hi) {
            float S[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) S[nt][0] = S[nt][1] = S[nt][2] = S[nt][3] = 0.f;

            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int kcol = ks * 16 + c;
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const int kr = nt * 8 + r;
                    const u32 b0 = *reinterpret_cast<const u32*>(&sK[kr][kcol]);
                    const u32 b1 = *reinterpret_cast<const u32*>(&sK[kr][kcol + 8]);
                    emb_mma16816_bf16(S[nt], q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], b0, b1);
                }
            }

            const bool need_mask = (k0 + TEXT_TC_BK > seq_len) || (causal && (k0 + TEXT_TC_BK > q_local_base + warp * 16));
            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int kp0 = k0 + nt * 8 + c;
                const int kp1 = kp0 + 1;
                float s0 = S[nt][0] * scale, s1 = S[nt][1] * scale;
                float s2 = S[nt][2] * scale, s3 = S[nt][3] * scale;
                if (need_mask) {
                    if (kp0 >= seq_len || (causal && kp0 > q_row0)) s0 = -1e30f;
                    if (kp1 >= seq_len || (causal && kp1 > q_row0)) s1 = -1e30f;
                    if (kp0 >= seq_len || (causal && kp0 > q_row1)) s2 = -1e30f;
                    if (kp1 >= seq_len || (causal && kp1 > q_row1)) s3 = -1e30f;
                }
                S[nt][0] = s0; S[nt][1] = s1; S[nt][2] = s2; S[nt][3] = s3;
                rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
                rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
            }
            // Quad-reduce max across 4 lanes holding the same query row
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

            const float m_new0 = fmaxf(m_max0, rmax0);
            const float m_new1 = fmaxf(m_max1, rmax1);
            const float alpha0 = (m_max0 > -1e20f) ? __expf(m_max0 - m_new0) : 0.f;
            const float alpha1 = (m_max1 > -1e20f) ? __expf(m_max1 - m_new1) : 0.f;

            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                O_acc[dt][0] *= alpha0; O_acc[dt][1] *= alpha0;
                O_acc[dt][2] *= alpha1; O_acc[dt][3] *= alpha1;
            }
            l_sum0 *= alpha0;
            l_sum1 *= alpha1;

            float rsum0 = 0.f, rsum1 = 0.f;
            u32 pf[4][4];
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int n0 = ks * 2, n1 = ks * 2 + 1;
                const float p00 = (S[n0][0] > -1e20f) ? __expf(S[n0][0] - m_new0) : 0.f;
                const float p01 = (S[n0][1] > -1e20f) ? __expf(S[n0][1] - m_new0) : 0.f;
                const float p02 = (S[n0][2] > -1e20f) ? __expf(S[n0][2] - m_new1) : 0.f;
                const float p03 = (S[n0][3] > -1e20f) ? __expf(S[n0][3] - m_new1) : 0.f;
                const float p10 = (S[n1][0] > -1e20f) ? __expf(S[n1][0] - m_new0) : 0.f;
                const float p11 = (S[n1][1] > -1e20f) ? __expf(S[n1][1] - m_new0) : 0.f;
                const float p12 = (S[n1][2] > -1e20f) ? __expf(S[n1][2] - m_new1) : 0.f;
                const float p13 = (S[n1][3] > -1e20f) ? __expf(S[n1][3] - m_new1) : 0.f;
                rsum0 += (p00 + p01) + (p10 + p11);
                rsum1 += (p02 + p03) + (p12 + p13);
                pf[ks][0] = emb_pack_bf2(p00, p01);
                pf[ks][1] = emb_pack_bf2(p02, p03);
                pf[ks][2] = emb_pack_bf2(p10, p11);
                pf[ks][3] = emb_pack_bf2(p12, p13);
            }
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
            l_sum0 += rsum0;
            l_sum1 += rsum1;
            m_max0 = m_new0;
            m_max1 = m_new1;

            // O_acc += P @ sV
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int krow = ks * 16 + c;
                #pragma unroll
                for (int dt = 0; dt < 16; ++dt) {
                    const int dc = dt * 8 + r;
                    const uint16_t v0 = *reinterpret_cast<const uint16_t*>(&sV[krow][dc]);
                    const uint16_t v1 = *reinterpret_cast<const uint16_t*>(&sV[krow + 1][dc]);
                    const uint16_t v2 = *reinterpret_cast<const uint16_t*>(&sV[krow + 8][dc]);
                    const uint16_t v3 = *reinterpret_cast<const uint16_t*>(&sV[krow + 9][dc]);
                    const u32 b0 = static_cast<u32>(v0) | (static_cast<u32>(v1) << 16);
                    const u32 b1 = static_cast<u32>(v2) | (static_cast<u32>(v3) << 16);
                    emb_mma16816_bf16(O_acc[dt], pf[ks][0], pf[ks][1], pf[ks][2], pf[ks][3], b0, b1);
                }
            }
        }
        __syncthreads();
    }

    const float inv0 = (l_sum0 > 0.f) ? (1.0f / l_sum0) : 0.f;
    const float inv1 = (l_sum1 > 0.f) ? (1.0f / l_sum1) : 0.f;
    bf16* op0 = (q_row0 < seq_len) ? (O + (size_t(s_start + q_row0) * nq + hq) * 128) : nullptr;
    bf16* op1 = (q_row1 < seq_len) ? (O + (size_t(s_start + q_row1) * nq + hq) * 128) : nullptr;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        const int col = dt * 8 + c;
        if (op0) *reinterpret_cast<u32*>(op0 + col) = emb_pack_bf2(O_acc[dt][0] * inv0, O_acc[dt][1] * inv0);
        if (op1) *reinterpret_cast<u32*>(op1 + col) = emb_pack_bf2(O_acc[dt][2] * inv1, O_acc[dt][3] * inv1);
    }
}

// SwiGLU activation: out = bf16(bf16(silu(gate)) * up). Grid: ceil(M * I / 256), Block: 256.
__global__ void __launch_bounds__(256) k_q3_swiglu(
    const bf16* __restrict__ gate, const bf16* __restrict__ up, bf16* __restrict__ out, size_t n) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    const float s = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g))));
    out[i] = __float2bfloat16_rn(s * u);
}

// Scatter visual tokens from main merger output into hidden[img_tok_pos[k]] (overwrite).
__global__ void __launch_bounds__(256) k_q3_scatter_visual(
    bf16* __restrict__ hidden, const bf16* __restrict__ vis, const int* __restrict__ img_pos, int n_img, int H) {
    const int k = blockIdx.x;
    if (k >= n_img) return;
    const int dst_row = img_pos[k];
    for (int d = threadIdx.x * 8; d < H; d += 256 * 8) {
        *reinterpret_cast<uint4*>(hidden + size_t(dst_row) * H + d) =
            *reinterpret_cast<const uint4*>(vis + size_t(k) * H + d);
    }
}

// In-place DeepStack residual scatter-add after LLM layers 0..2: hidden[img_pos[k]] += ds[k].
__global__ void __launch_bounds__(256) k_q3_scatter_add_deepstack(
    bf16* __restrict__ hidden, const bf16* __restrict__ ds, const int* __restrict__ img_pos, int n_img, int H) {
    const int k = blockIdx.x;
    if (k >= n_img) return;
    const int dst_row = img_pos[k];
    bf16* hr = hidden + size_t(dst_row) * H;
    const bf16* dr = ds + size_t(k) * H;
    for (int d = threadIdx.x * 2; d < H; d += 256 * 2) {
        const u32 hv = *reinterpret_cast<const u32*>(hr + d);
        const u32 dv = *reinterpret_cast<const u32*>(dr + d);
        const float2 hf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hv));
        const float2 df = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&dv));
        *reinterpret_cast<u32*>(hr + d) = emb_pack_bf2(hf.x + df.x, hf.y + df.y);
    }
}

// Last-token (or mean) pooling + MRL slice + optional FP32 L2 normalization.
// Grid: (B), Block: 256. Each sequence b writes out_emb[b * max_dim .. b * max_dim + req_dims[b]].
__global__ void __launch_bounds__(256) k_q3_pool_l2(
    const bf16* __restrict__ hidden,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ req_dims,
    const int* __restrict__ req_norm,
    float* __restrict__ out_emb,
    int B, int H, int max_dim, int last_token_pool) {
    __shared__ float red[8];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int lo = cu_seqlens[b];
    const int hi = cu_seqlens[b + 1];
    const int dim = req_dims[b];
    const bool do_norm = (req_norm[b] != 0);
    float* dst = out_emb + size_t(b) * max_dim;

    float ss = 0.f;
    if (last_token_pool) {
        const bf16* row = hidden + size_t(hi - 1) * H;
        for (int d = threadIdx.x; d < dim; d += 256) {
            const float v = __bfloat162float(row[d]);
            dst[d] = v;
            ss += v * v;
        }
    } else {
        const float inv_len = 1.0f / float(max(1, hi - lo));
        for (int d = threadIdx.x; d < dim; d += 256) {
            float acc = 0.f;
            for (int r = lo; r < hi; ++r) acc += __bfloat162float(hidden[size_t(r) * H + d]);
            const float v = acc * inv_len;
            dst[d] = v;
            ss += v * v;
        }
    }
    const float norm2 = emb_block_sum256(ss, red);
    if (do_norm) {
        const float inv = rsqrtf(fmaxf(norm2, 1e-12f));
        for (int d = threadIdx.x; d < dim; d += 256) {
            dst[d] *= inv;
        }
    }
}

// ------------------------------------------------------------------------------------------------ Profiler

struct EmbedProfiler {
    bool enabled = false;
    double embed_ms = 0.0;
    double rmsnorm_ms = 0.0;
    double qkv_ms = 0.0;
    double qknorm_mrope_ms = 0.0;
    double attn_ms = 0.0;
    double o_proj_ms = 0.0;
    double mlp_gate_up_ms = 0.0;
    double swiglu_ms = 0.0;
    double mlp_down_ms = 0.0;
    double deepstack_ms = 0.0;
    double vision_ms = 0.0;
    double pool_ms = 0.0;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    void init(bool on) {
        enabled = on;
        if (enabled && !ev0) {
            EMB_CK(cudaEventCreate(&ev0));
            EMB_CK(cudaEventCreate(&ev1));
        }
    }
    void reset() {
        embed_ms = rmsnorm_ms = qkv_ms = qknorm_mrope_ms = attn_ms = o_proj_ms = 0.0;
        mlp_gate_up_ms = swiglu_ms = mlp_down_ms = deepstack_ms = vision_ms = pool_ms = 0.0;
    }
    void release() {
        if (ev0) { cudaEventDestroy(ev0); ev0 = nullptr; }
        if (ev1) { cudaEventDestroy(ev1); ev1 = nullptr; }
    }
    ~EmbedProfiler() { release(); }

    template <class Fn>
    void record(double& bucket, cudaStream_t st, Fn&& fn) {
        if (!enabled) { fn(); return; }
        if (!ev0) init(true);
        EMB_CK(cudaEventRecord(ev0, st));
        fn();
        EMB_CK(cudaEventRecord(ev1, st));
        EMB_CK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        EMB_CK(cudaEventElapsedTime(&ms, ev0, ev1));
        bucket += double(ms);
    }

    double total_ms() const {
        return embed_ms + rmsnorm_ms + qkv_ms + qknorm_mrope_ms + attn_ms + o_proj_ms +
               mlp_gate_up_ms + swiglu_ms + mlp_down_ms + deepstack_ms + vision_ms + pool_ms;
    }
};

// ------------------------------------------------------------------------------------------------ TextRunner

// Packed vision inputs produced by VisionRunner (Commit 3) and consumed by TextRunner.
struct PackedVisionFeatures {
    int n_img_tokens = 0;
    const bf16* main_vis = nullptr;              // [n_img_tokens, H] on device
    std::vector<const bf16*> deepstack_vis;      // up to 3 pointers [n_img_tokens, H] on device
    std::vector<int> img_token_positions;        // length n_img_tokens, packed global token index in [0, M)
};

class TextRunner {
public:
    const EmbedWeights* W = nullptr;
    GemmWorkspace ws;
    EmbedProfiler prof;
    int max_tokens = 0;
    int max_batch = 256;

    // Ephemeral activations (zero persistent KV cache!)
    bf16* d_hidden = nullptr;    // [max_tokens, H]
    bf16* d_normed = nullptr;    // [max_tokens, H]
    bf16* d_q = nullptr;         // [max_tokens, nq * hd]
    bf16* d_k = nullptr;         // [max_tokens, nkv * hd]
    bf16* d_v = nullptr;         // [max_tokens, nkv * hd]
    bf16* d_attn = nullptr;      // [max_tokens, nq * hd]
    bf16* d_gate = nullptr;      // [max_tokens, I]
    bf16* d_up = nullptr;        // [max_tokens, I]
    bf16* d_act = nullptr;       // [max_tokens, I]

    int* d_toks = nullptr;       // [max_tokens]
    int3* d_mrope = nullptr;     // [max_tokens]
    int* d_cu_seqlens = nullptr; // [max_batch + 1]
    int* d_seq_of_row = nullptr; // [max_tokens]
    int2* d_q_tiles = nullptr;   // [max_tokens / 64 + max_batch + 16]
    int* d_img_pos = nullptr;    // [max_tokens]
    int* d_req_dims = nullptr;   // [max_batch]
    int* d_req_norm = nullptr;   // [max_batch]
    float* d_inv_freq = nullptr; // [head_dim / 2] precomputed RoPE inverse frequencies
    float* d_out_emb = nullptr;  // [max_batch, H]

    // Reusable pinned host staging buffers (64K-page safe, zero per-batch heap alloc or pageable copy stall)
    int* h_toks_pin = nullptr;
    int3* h_mrope_pin = nullptr;
    int* h_cu_pin = nullptr;
    int* h_seq_pin = nullptr;
    int2* h_qtiles_pin = nullptr;
    int* h_dims_pin = nullptr;
    int* h_norm_pin = nullptr;
    float* h_out_pin = nullptr;
    std::vector<MropePos> tmp_pos;

    void init(const EmbedWeights& weights, int max_batched_tokens, int max_batch_seqs = 256, bool profile = false) {
        W = &weights;
        max_tokens = std::max(64, max_batched_tokens);
        max_batch = std::max(16, max_batch_seqs);
        const auto& c = W->cfg;
        const int max_K = std::max({c.hidden_size, c.intermediate_size, c.num_q_heads * c.head_dim,
                                    c.has_vision ? std::max(c.vit_hidden * 4, c.vit_intermediate) : 0});
        ws.init(max_tokens, max_K);
        prof.init(profile);

        auto alloc = [](auto** p, size_t n) {
            EMB_CK(cudaMalloc(p, std::max<size_t>(1, n) * sizeof(**p)));
        };
        alloc(&d_hidden, size_t(max_tokens) * c.hidden_size);
        alloc(&d_normed, size_t(max_tokens) * c.hidden_size);
        alloc(&d_q, size_t(max_tokens) * c.num_q_heads * c.head_dim);
        alloc(&d_k, size_t(max_tokens) * c.num_kv_heads * c.head_dim);
        alloc(&d_v, size_t(max_tokens) * c.num_kv_heads * c.head_dim);
        alloc(&d_attn, size_t(max_tokens) * c.num_q_heads * c.head_dim);
        alloc(&d_gate, size_t(max_tokens) * c.intermediate_size);
        alloc(&d_up, size_t(max_tokens) * c.intermediate_size);
        // Only allocate d_act if any layer's down_proj is non-Trellis (saves 201 MiB on 2B / 402 MiB on 8B in native mode!).
        bool need_act = false;
        for (const auto& L : weights.text_layers) {
            if (L.down_proj.fmt != FMT_TRELLIS) { need_act = true; break; }
        }
        if (need_act) alloc(&d_act, size_t(max_tokens) * c.intermediate_size);

        alloc(&d_toks, size_t(max_tokens));
        alloc(&d_mrope, size_t(max_tokens));
        alloc(&d_cu_seqlens, size_t(max_batch + 1));
        alloc(&d_seq_of_row, size_t(max_tokens));
        alloc(&d_q_tiles, size_t(max_tokens / TEXT_TC_BQ + max_batch + 16));
        alloc(&d_img_pos, size_t(max_tokens));
        alloc(&d_req_dims, size_t(max_batch));
        alloc(&d_req_norm, size_t(max_batch));
        alloc(&d_inv_freq, size_t(c.head_dim / 2));
        alloc(&d_out_emb, size_t(max_batch) * c.embedding_dim);

        std::vector<float> h_inv(size_t(c.head_dim / 2));
        for (int j = 0; j < c.head_dim / 2; ++j) {
            h_inv[size_t(j)] = static_cast<float>(1.0 / std::pow(double(c.rope_theta), double(2 * j) / double(c.head_dim)));
        }
        EMB_CK(cudaMemcpy(d_inv_freq, h_inv.data(), h_inv.size() * sizeof(float), cudaMemcpyHostToDevice));

        auto alloc_pin = [](auto** p, size_t n) {
            EMB_CK(cudaHostAlloc(p, std::max<size_t>(1, n) * sizeof(**p), cudaHostAllocDefault));
        };
        alloc_pin(&h_toks_pin, size_t(max_tokens));
        alloc_pin(&h_mrope_pin, size_t(max_tokens));
        alloc_pin(&h_cu_pin, size_t(max_batch + 1));
        alloc_pin(&h_seq_pin, size_t(max_tokens));
        alloc_pin(&h_qtiles_pin, size_t(max_tokens / TEXT_TC_BQ + max_batch + 16));
        alloc_pin(&h_dims_pin, size_t(max_batch));
        alloc_pin(&h_norm_pin, size_t(max_batch));
        alloc_pin(&h_out_pin, size_t(max_batch) * c.embedding_dim);
    }

    size_t workspace_bytes() const {
        if (!W) return 0;
        const auto& c = W->cfg;
        size_t b = ws.x_rot_bytes + ws.acc_floats * sizeof(float);
        const int n_mlp_bufs = d_act ? 3 : 2;
        b += size_t(max_tokens) * (2 * c.hidden_size + 2 * c.num_q_heads * c.head_dim +
                                   2 * c.num_kv_heads * c.head_dim + n_mlp_bufs * c.intermediate_size) * sizeof(bf16);
        b += size_t(max_batch) * c.embedding_dim * sizeof(float);
        return b;
    }

    void release() {
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_hidden); fr(d_normed); fr(d_q); fr(d_k); fr(d_v); fr(d_attn);
        fr(d_gate); fr(d_up); fr(d_act);
        fr(d_toks); fr(d_mrope); fr(d_cu_seqlens); fr(d_seq_of_row); fr(d_q_tiles); fr(d_img_pos);
        fr(d_req_dims); fr(d_req_norm); fr(d_inv_freq); fr(d_out_emb);
        auto fr_pin = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        fr_pin(h_toks_pin); fr_pin(h_mrope_pin); fr_pin(h_cu_pin); fr_pin(h_seq_pin);
        fr_pin(h_qtiles_pin); fr_pin(h_dims_pin); fr_pin(h_norm_pin); fr_pin(h_out_pin);
        ws.release();
        prof.release();
    }
    ~TextRunner() { release(); }

    // Runs a packed batch of EmbedRequests (with optional precomputed vision/DeepStack features)
    // and returns one EmbedOutput per request.
    std::vector<EmbedOutput> forward_batch(const std::vector<EmbedRequest>& batch,
                                           const PackedVisionFeatures* vis,
                                           cudaStream_t st) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& c = W->cfg;
        const int B = static_cast<int>(batch.size());
        if (B == 0) return {};
        if (B > max_batch) throw std::runtime_error("g6embed: batch size exceeds max_batch");

        int total_M = 0;
        for (int b = 0; b < B; ++b) {
            const int len = static_cast<int>(batch[size_t(b)].prompt.size());
            h_cu_pin[b] = total_M;
            total_M += len;
            h_dims_pin[b] = batch[size_t(b)].dimensions > 0 ? batch[size_t(b)].dimensions : c.embedding_dim;
            h_norm_pin[b] = batch[size_t(b)].normalize ? 1 : 0;
        }
        h_cu_pin[B] = total_M;
        if (total_M > max_tokens) {
            throw std::runtime_error("g6embed: packed batch tokens (" + std::to_string(total_M) +
                                     ") exceed max_batched_tokens (" + std::to_string(max_tokens) + ")");
        }

        int row_idx = 0;
        int num_q_tiles = 0;
        for (int b = 0; b < B; ++b) {
            const auto& req = batch[size_t(b)];
            const int len = static_cast<int>(req.prompt.size());
            tmp_pos.resize(size_t(len));
            build_qwen3vl_mrope_positions(len, req.images, tmp_pos.data());
            for (int p = 0; p < len; ++p) {
                h_toks_pin[row_idx] = req.prompt[size_t(p)];
                h_mrope_pin[row_idx] = make_int3(tmp_pos[size_t(p)].t, tmp_pos[size_t(p)].h, tmp_pos[size_t(p)].w);
                h_seq_pin[row_idx] = b;
                ++row_idx;
            }
            for (int ql = 0; ql < len; ql += TEXT_TC_BQ) {
                h_qtiles_pin[num_q_tiles++] = make_int2(b, ql);
            }
        }

        EMB_CK(cudaMemcpyAsync(d_toks, h_toks_pin, size_t(total_M) * sizeof(int), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_mrope, h_mrope_pin, size_t(total_M) * sizeof(int3), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_cu_seqlens, h_cu_pin, size_t(B + 1) * sizeof(int), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_seq_of_row, h_seq_pin, size_t(total_M) * sizeof(int), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_q_tiles, h_qtiles_pin, size_t(num_q_tiles) * sizeof(int2), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_req_dims, h_dims_pin, size_t(B) * sizeof(int), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_req_norm, h_norm_pin, size_t(B) * sizeof(int), cudaMemcpyHostToDevice, st));

        const bool has_vis = (vis && vis->n_img_tokens > 0 && vis->main_vis != nullptr);
        if (has_vis) {
            EMB_CK(cudaMemcpyAsync(d_img_pos, vis->img_token_positions.data(),
                                   size_t(vis->n_img_tokens) * sizeof(int), cudaMemcpyHostToDevice, st));
        }

        // 1. Token embedding lookup
        prof.record(prof.embed_ms, st, [&] {
            if (W->embed_tokens.fmt == FMT_FP8_ROW) {
                const dim3 grid(total_M, (c.hidden_size / 2 + 255) / 256);
                k_embed_lookup_fp8_row<<<grid, 256, 0, st>>>(
                    d_toks, W->embed_tokens.data, W->embed_tokens.scale, d_hidden, total_M, c.hidden_size);
            } else if (W->embed_tokens.fmt == FMT_BF16) {
                const dim3 grid(total_M, (c.hidden_size / 8 + 255) / 256);
                k_embed_lookup_bf16<<<grid, 256, 0, st>>>(d_toks, W->embed_tokens.bf16, d_hidden, total_M, c.hidden_size);
            } else {
                throw std::runtime_error("g6embed: unsupported embed_tokens format");
            }
            if (has_vis) {
                k_q3_scatter_visual<<<vis->n_img_tokens, 256, 0, st>>>(
                    d_hidden, vis->main_vis, d_img_pos, vis->n_img_tokens, c.hidden_size);
            }
        });

        const int s1_lim = c.mrope_section[1] * 3;
        const int s2_lim = c.mrope_section[2] * 3;
        const bool is_causal = (c.profile == EmbedProfile::Qwen3Vl);

        // 2. Text transformer layers
        for (int l = 0; l < c.num_text_layers; ++l) {
            const TextLayerW& L = W->text_layers[size_t(l)];

            bool fused_qkv = false;
            prof.record(prof.qkv_ms, st, [&] {
                fused_qkv = rmsnorm_linear_forward_triple(
                    d_hidden, L.input_ln, c.rms_norm_eps,
                    L.q_proj, L.k_proj, L.v_proj, d_q, d_k, d_v, total_M, ws, st);
            });
            if (!fused_qkv) {
                prof.record(prof.rmsnorm_ms, st, [&] {
                    k_q3_rmsnorm<<<total_M, 256, 0, st>>>(d_hidden, L.input_ln, d_normed, total_M, c.hidden_size, c.rms_norm_eps);
                });
                prof.record(prof.qkv_ms, st, [&] {
                    linear_forward_triple(d_normed, L.q_proj, L.k_proj, L.v_proj, d_q, d_k, d_v, total_M, ws, st);
                });
            }

            prof.record(prof.qknorm_mrope_ms, st, [&] {
                const dim3 qk_grid(total_M, (c.num_q_heads + c.num_kv_heads + 3) / 4);
                k_q3_qk_norm_mrope_fused<<<qk_grid, 256, 0, st>>>(
                    d_q, d_k, L.q_norm, L.k_norm, d_mrope, d_inv_freq,
                    total_M, c.num_q_heads, c.num_kv_heads, c.head_dim,
                    c.rms_norm_eps, s1_lim, s2_lim);
            });

            prof.record(prof.attn_ms, st, [&] {
                if (c.head_dim == 128) {
                    const dim3 grid(num_q_tiles, c.num_q_heads);
                    k_q3_flash_attn_tc<<<grid, 128, 0, st>>>(
                        d_q, d_k, d_v, d_attn, d_cu_seqlens, d_q_tiles,
                        num_q_tiles, c.num_q_heads, c.num_kv_heads, is_causal);
                } else {
                    const dim3 grid((total_M + TEXT_FA_QW - 1) / TEXT_FA_QW, c.num_q_heads);
                    k_q3_causal_gqa_attn<<<grid, 128, 0, st>>>(
                        d_q, d_k, d_v, d_attn, d_cu_seqlens, d_seq_of_row,
                        total_M, c.num_q_heads, c.num_kv_heads, c.head_dim, is_causal);
                }
            });

            prof.record(prof.o_proj_ms, st, [&] {
                linear_forward(d_attn, L.o_proj, nullptr, d_hidden, total_M, true, ws, st);
            });

            bool fused_gu = false;
            prof.record(prof.mlp_gate_up_ms, st, [&] {
                fused_gu = rmsnorm_linear_forward_dual(
                    d_hidden, L.post_attn_ln, c.rms_norm_eps,
                    L.gate_proj, L.up_proj, d_gate, d_up, total_M, ws, st);
            });
            if (!fused_gu) {
                prof.record(prof.rmsnorm_ms, st, [&] {
                    k_q3_rmsnorm<<<total_M, 256, 0, st>>>(d_hidden, L.post_attn_ln, d_normed, total_M, c.hidden_size, c.rms_norm_eps);
                });
                prof.record(prof.mlp_gate_up_ms, st, [&] {
                    linear_forward_dual(d_normed, L.gate_proj, L.up_proj, d_gate, d_up, total_M, ws, st);
                });
            }

            if (L.down_proj.fmt == FMT_TRELLIS) {
                prof.record(prof.swiglu_ms, st, [&] {
                    launch_swiglu_rot_h16(d_gate, d_up, L.down_proj.suh, total_M, c.intermediate_size, ws, st);
                });
                prof.record(prof.mlp_down_ms, st, [&] {
                    launch_trellis_with_rot(L.down_proj, nullptr, d_hidden, total_M, true, ws, st);
                });
            } else {
                prof.record(prof.swiglu_ms, st, [&] {
                    const size_t n = size_t(total_M) * c.intermediate_size;
                    k_q3_swiglu<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(d_gate, d_up, d_act, n);
                });
                prof.record(prof.mlp_down_ms, st, [&] {
                    linear_forward(d_act, L.down_proj, nullptr, d_hidden, total_M, true, ws, st);
                });
            }

            // DeepStack injection after LLM layers 0, 1, 2
            if (has_vis && size_t(l) < vis->deepstack_vis.size() && vis->deepstack_vis[size_t(l)] != nullptr) {
                prof.record(prof.deepstack_ms, st, [&] {
                    k_q3_scatter_add_deepstack<<<vis->n_img_tokens, 256, 0, st>>>(
                        d_hidden, vis->deepstack_vis[size_t(l)], d_img_pos, vis->n_img_tokens, c.hidden_size);
                });
            }
        }

        // 3. Final RMSNorm + Pooling + MRL + FP32 L2 norm
        prof.record(prof.pool_ms, st, [&] {
            k_q3_rmsnorm<<<total_M, 256, 0, st>>>(d_hidden, W->final_norm, d_normed, total_M, c.hidden_size, c.rms_norm_eps);
            k_q3_pool_l2<<<B, 256, 0, st>>>(
                d_normed, d_cu_seqlens, d_req_dims, d_req_norm, d_out_emb,
                B, c.hidden_size, c.embedding_dim, c.last_token_pool ? 1 : 0);
        });

        EMB_CK(cudaMemcpyAsync(h_out_pin, d_out_emb, size_t(B) * c.embedding_dim * sizeof(float), cudaMemcpyDeviceToHost, st));
        EMB_CK(cudaStreamSynchronize(st));

        const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<EmbedOutput> outs;
        outs.reserve(size_t(B));
        for (int b = 0; b < B; ++b) {
            EmbedOutput o;
            o.id = batch[size_t(b)].id;
            o.prompt_tokens = static_cast<int>(batch[size_t(b)].prompt.size());
            const int dim = h_dims_pin[b];
            o.embedding.assign(h_out_pin + size_t(b) * c.embedding_dim,
                               h_out_pin + size_t(b) * c.embedding_dim + dim);
            o.prefill_s = prefill_s;
            outs.push_back(std::move(o));
        }
        return outs;
    }
};

} // namespace g6emb
