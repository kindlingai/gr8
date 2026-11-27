// ERNIE-5.5-Flash-Preview 37x HCA C128 Compressed Attention (all layers except 7, 14, 21, 28, 35, 42).
//
// Exact ERNIE 5.5 math invariants verified against config.json & CPU reference:
//   - Projections:
//       q_lora = rms_norm(q_down(x), q_norm)                   [M, 1024]
//       kv     = rms_norm(kv_proj(x), kv_norm)                 [M, 512]
//       gate   = sigmoid(gate_proj(q_lora))                    [M, ng_loc * 1024]
//       q      = rms_norm(q_up(q_lora).view(M, nh_loc, 512), None)
//   - YaRN Interleaved RoPE on trailing 64 dims [448..511] (theta=160k, factor=16, orig=65536, bf=32, bs=1):
//       q[..., 448:] = rope_interleaved(q[..., 448:], pos)
//       swa_kv[..., 448:] = rope_interleaved(kv[..., 448:], pos) -> written to SWA ring
//   - Non-overlapping C128 Compressor (compress_ratio = 128):
//       c_kv = comp_wkv(x) [F32 512], c_sc = comp_wgate(x) [F32 512]
//       On every 128-token boundary (pos + 1) % 128 == 0:
//         ps = softmax_k(part_sc[0..127] + ape[0..127])        [128, 512] F32
//         comp = rms_norm(sum_k part_kv[k] * ps[k], comp_norm) [512] BF16
//         comp[448:] = rope_interleaved(comp[448:], blk * 128)
//         comp_kv[slot, blk] = comp
//   - Attention (Tensor Core mma.m16n8k16, K == V == 512 dims, scale = 512^-0.5):
//       Visible keys at pos = [comp_kv[0 .. (pos+1)/128 - 1] | swa_ring[max(0, pos-127) .. pos]]
//       Online softmax initialized with F32 attn_sink (m = sink[h], l = 1.0)
//   - Post-Attention:
//       o = VHAPostMix(o, I + V @ U^T)                         [M, nh_loc, 512]
//       o[..., 448:] = rope_interleaved_inverse(o[..., 448:], pos)
//       z = grouped_linear(o.view(M, ng_loc, 4096), wo_a)      [M, ng_loc * 1024]
//       out = o_proj(z * gate)                                 [M, 4096]
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "../model_geom_ernie55.h"
#include "../kernels/ernie55_mhc.cuh"
#include "../kernels/ernie55_mxfp8.cuh"

namespace ernie55 {

constexpr int HCA_MAX_SPLITS = 32;

__device__ __forceinline__ void mma16816_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Compute exact FP64 YaRN inverse frequencies (32 pairs for rope_dim=64) on host at boot.
inline std::vector<float> build_hca_yarn_inv_freq() {
    constexpr int D = geom::HCA_ROPE;
    constexpr int HALF = D / 2;
    const double theta = static_cast<double>(geom::HCA_ROPE_THETA);
    const double factor = static_cast<double>(geom::HCA_YARN_FACTOR);
    const double orig = static_cast<double>(geom::HCA_YARN_ORIG);
    const double pi = 3.14159265358979323846;
    auto corr_dim = [&](double nrot) {
        return (D * std::log(orig / (nrot * 2.0 * pi))) / (2.0 * std::log(theta));
    };
    double low = std::max(std::floor(corr_dim(geom::HCA_YARN_BETA_FAST)), 0.0);
    double high = std::min(std::ceil(corr_dim(geom::HCA_YARN_BETA_SLOW)), static_cast<double>(D - 1));
    if (low == high) high += 0.001;
    std::vector<float> out(HALF);
    for (int p = 0; p < HALF; ++p) {
        const double inv_extrap = 1.0 / std::pow(theta, static_cast<double>(2 * p) / static_cast<double>(D));
        const double inv_interp = inv_extrap / factor;
        double ramp = (static_cast<double>(p) - low) / (high - low);
        ramp = std::max(0.0, std::min(1.0, ramp));
        const double mask = 1.0 - ramp;
        out[static_cast<size_t>(p)] = static_cast<float>(inv_interp * (1.0 - mask) + inv_extrap * mask);
    }
    return out;
}

__device__ __forceinline__ void hca_rope_pair(float& x0, float& x1, int pos, float inv_freq, bool inverse) {
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    float y0, y1;
    if (inverse) {
        y0 = x0 * c + x1 * s;
        y1 = x1 * c - x0 * s;
    } else {
        y0 = x0 * c - x1 * s;
        y1 = x1 * c + x0 * s;
    }
    x0 = y0;
    x1 = y1;
}

// Precompute F32 VHA postmix matrix [64, 64] = I_64 + V @ U^T from BF16 U[64, 16], V[64, 16].
__global__ void __launch_bounds__(256) k_ernie55_build_vha_mix(
    const __nv_bfloat16* __restrict__ U,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ mix) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= geom::HEADS * geom::HEADS) return;
    const int h = idx / geom::HEADS;
    const int g = idx % geom::HEADS;
    float s = (h == g) ? 1.0f : 0.0f;
    for (int r = 0; r < geom::VHA_RANK; ++r) {
        s = fmaf(__bfloat162float(V[h * geom::VHA_RANK + r]),
                 __bfloat162float(U[g * geom::VHA_RANK + r]), s);
    }
    mix[idx] = s;
}

struct HcaWeights {
    int nh_loc = 0;                               // 64 / nr
    int ng_loc = 0;                               // 8 / nr
    int h_off  = 0;                               // rank * nh_loc
    mx8::LinW            q_down{};                // [1024, 4096]
    const __nv_bfloat16* q_norm     = nullptr;    // [1024]
    mx8::LinW            q_up{};                  // [nh_loc * 512, 1024]
    mx8::LinW            kv_proj{};               // [512, 4096]
    const __nv_bfloat16* kv_norm    = nullptr;    // [512]
    mx8::LinW            comp_wkv{};              // [512, 4096]
    mx8::LinW            comp_wgate{};            // [512, 4096]
    const float*         comp_ape   = nullptr;    // [128, 512] F32
    const __nv_bfloat16* comp_norm  = nullptr;    // [512]
    const float*         sinks      = nullptr;    // [nh_loc] F32
    const float*         vha_mix    = nullptr;    // [64, 64] F32 (precomputed I + V @ U^T)
    mx8::LinW            wo_a[geom::HCA_O_GROUPS]{}; // ng_loc matrices of [1024, 4096]
    mx8::LinW            gate_proj{};             // [ng_loc * 1024, 1024]
    mx8::LinW            o_proj{};                // [4096, ng_loc * 1024]
};

struct HcaLayerKv {
    __nv_bfloat16* swa_ring = nullptr; // [slots][HCA_SWA_RING][512] BF16
    __nv_bfloat16* comp_kv  = nullptr; // [slots][max_comp_blks][512] BF16
    float*         part_kv  = nullptr; // [slots][128][512] F32 (pending partial block)
    float*         part_sc  = nullptr; // [slots][128][512] F32 (pending scores + ape)
    int max_comp_blks = 0;
};

struct HcaScratch {
    __nv_bfloat16* q_lora   = nullptr; // [max_M, 1024]
    __nv_bfloat16* q_raw    = nullptr; // [max_M, nh_loc, 512]
    __nv_bfloat16* kv_raw   = nullptr; // [max_M, 512]
    float*         c_kv     = nullptr; // [max_M, 512] F32
    float*         c_sc     = nullptr; // [max_M, 512] F32
    __nv_bfloat16* attn_o   = nullptr; // [max_M, nh_loc, 512]
    __nv_bfloat16* post_o   = nullptr; // [max_M, ng_loc, 4096] (after VHA + inv-RoPE, grouped!)
    __nv_bfloat16* o_grp_in = nullptr; // [max_M, 4096] (staging for group g when ng_loc > 1)
    __nv_bfloat16* o_grp_z  = nullptr; // [max_M, 1024]
    __nv_bfloat16* z_cat    = nullptr; // [max_M, ng_loc * 1024]
    __nv_bfloat16* gate     = nullptr; // [max_M, ng_loc * 1024]

    float* split_m = nullptr; // [max_dec_M, HCA_MAX_SPLITS, nh_loc]
    float* split_l = nullptr; // [max_dec_M, HCA_MAX_SPLITS, nh_loc]
    float* split_o = nullptr; // [max_dec_M, HCA_MAX_SPLITS, nh_loc, 512]
};

// Weightless per-head RMSNorm (512 dims) + trailing-64 YaRN interleaved RoPE on q_raw in-place.
// Grid: (M * nh), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_hca_q_norm_rope(
    __nv_bfloat16* __restrict__ q,
    const int* __restrict__ d_pos,
    const float* __restrict__ inv_freq,
    int M, int nh) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    __nv_bfloat16* qh = q + static_cast<size_t>(idx) * geom::HCA_HD;

    const float v0 = __bfloat162float(qh[tid]);
    const float v1 = __bfloat162float(qh[tid + 256]);
    float ss = warp_sum(v0 * v0 + v1 * v1);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < 8) ? sm[tid] : 0.0f;
        tot = warp_sum(tot);
        if (tid == 0) sm[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::HCA_HD)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];

    __shared__ float s_q[geom::HCA_HD];
    s_q[tid]       = __bfloat162float(__float2bfloat16(v0 * inv));
    s_q[tid + 256] = __bfloat162float(__float2bfloat16(v1 * inv));
    __syncthreads();

    if (tid < geom::HCA_NOPE / 2) {
        qh[2 * tid]     = __float2bfloat16(s_q[2 * tid]);
        qh[2 * tid + 1] = __float2bfloat16(s_q[2 * tid + 1]);
    } else if (tid < geom::HCA_NOPE / 2 + geom::HCA_ROPE / 2) {
        const int p = tid - geom::HCA_NOPE / 2;
        float x0 = s_q[geom::HCA_NOPE + 2 * p];
        float x1 = s_q[geom::HCA_NOPE + 2 * p + 1];
        hca_rope_pair(x0, x1, d_pos[m], inv_freq[p], false);
        qh[geom::HCA_NOPE + 2 * p]     = __float2bfloat16(x0);
        qh[geom::HCA_NOPE + 2 * p + 1] = __float2bfloat16(x1);
    }
}

// KV RMSNorm (512 dims) + trailing-64 YaRN RoPE -> write into HCA SWA ring.
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_hca_kv_write(
    const __nv_bfloat16* __restrict__ kv_raw,
    const __nv_bfloat16* __restrict__ kv_norm_w,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ swa_ring,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* src = kv_raw + static_cast<size_t>(m) * geom::HCA_HD;

    const float v0 = __bfloat162float(src[tid]);
    const float v1 = __bfloat162float(src[tid + 256]);
    float ss = warp_sum(v0 * v0 + v1 * v1);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < 8) ? sm[tid] : 0.0f;
        tot = warp_sum(tot);
        if (tid == 0) sm[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::HCA_HD)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];

    __shared__ float s_kv[geom::HCA_HD];
    s_kv[tid]       = __bfloat162float(__float2bfloat16(v0 * inv * __bfloat162float(kv_norm_w[tid])));
    s_kv[tid + 256] = __bfloat162float(__float2bfloat16(v1 * inv * __bfloat162float(kv_norm_w[tid + 256])));
    __syncthreads();

    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    __nv_bfloat16* dst = swa_ring + (static_cast<size_t>(slot) * geom::HCA_SWA_RING + (pos & (geom::HCA_SWA_RING - 1))) * geom::HCA_HD;
    if (tid < geom::HCA_NOPE / 2) {
        dst[2 * tid]     = __float2bfloat16(s_kv[2 * tid]);
        dst[2 * tid + 1] = __float2bfloat16(s_kv[2 * tid + 1]);
    } else if (tid < geom::HCA_NOPE / 2 + geom::HCA_ROPE / 2) {
        const int p = tid - geom::HCA_NOPE / 2;
        float x0 = s_kv[geom::HCA_NOPE + 2 * p];
        float x1 = s_kv[geom::HCA_NOPE + 2 * p + 1];
        hca_rope_pair(x0, x1, pos, inv_freq[p], false);
        dst[geom::HCA_NOPE + 2 * p]     = __float2bfloat16(x0);
        dst[geom::HCA_NOPE + 2 * p + 1] = __float2bfloat16(x1);
    }
}

// Helper: commit one completed C128 block (512 threads = 1 thread per dim d=0..511) -> RMSNorm -> YaRN RoPE at blk_pos.
__device__ __forceinline__ void hca_commit_c128_dim(
    float pooled, int d, int blk_pos,
    const __nv_bfloat16* __restrict__ comp_norm_w,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ dst_comp,
    float (&s_comp)[geom::HCA_HD]) {
    float ss = warp_sum(pooled * pooled);
    __shared__ float sm[16];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    if (d < 32) {
        float tot = (d < 16) ? sm[d] : 0.0f;
        tot = warp_sum(tot);
        if (d == 0) sm[0] = rsqrtf(tot * (1.0f / static_cast<float>(geom::HCA_HD)) + geom::RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    s_comp[d] = __bfloat162float(__float2bfloat16(pooled * inv * __bfloat162float(comp_norm_w[d])));
    __syncthreads();

    if (d < geom::HCA_NOPE) {
        dst_comp[d] = __float2bfloat16(s_comp[d]);
    } else {
        const int p = (d - geom::HCA_NOPE) >> 1;
        float x0 = s_comp[geom::HCA_NOPE + 2 * p];
        float x1 = s_comp[geom::HCA_NOPE + 2 * p + 1];
        hca_rope_pair(x0, x1, blk_pos, inv_freq[p], false);
        dst_comp[d] = __float2bfloat16(((d & 1) == 0) ? x0 : x1);
    }
}

// Device-driven small-M C128 compressor update (100% CUDA-Graph-capture-safe!).
// Grid: 1, Block: 512.
__global__ void __launch_bounds__(512) k_ernie55_hca_comp_smallm(
    const float* __restrict__ c_kv,
    const float* __restrict__ c_sc,
    const float* __restrict__ comp_ape,
    const __nv_bfloat16* __restrict__ comp_norm_w,
    const float* __restrict__ inv_freq,
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    float* __restrict__ part_kv,
    float* __restrict__ part_sc,
    __nv_bfloat16* __restrict__ comp_kv,
    int M, int max_comp_blks) {
    const int d = threadIdx.x; // 0..511
    __shared__ float s_comp[geom::HCA_HD];
    for (int m = 0; m < M; ++m) {
        const int slot = d_slot[m];
        const int pos  = d_pos[m];
        const int sub  = pos & (geom::HCA_CR - 1);
        float* pkv = part_kv + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;
        float* psc = part_sc + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;
        pkv[static_cast<size_t>(sub) * geom::HCA_HD + d] = c_kv[static_cast<size_t>(m) * geom::HCA_HD + d];
        psc[static_cast<size_t>(sub) * geom::HCA_HD + d] =
            c_sc[static_cast<size_t>(m) * geom::HCA_HD + d] + comp_ape[static_cast<size_t>(sub) * geom::HCA_HD + d];
        __syncthreads();
        if (sub == geom::HCA_CR - 1) {
            float mx = -INFINITY;
            for (int k = 0; k < geom::HCA_CR; ++k) mx = fmaxf(mx, psc[static_cast<size_t>(k) * geom::HCA_HD + d]);
            float denom = 0.0f, num = 0.0f;
            for (int k = 0; k < geom::HCA_CR; ++k) {
                const float w = expf(psc[static_cast<size_t>(k) * geom::HCA_HD + d] - mx);
                denom += w;
                num = fmaf(w, pkv[static_cast<size_t>(k) * geom::HCA_HD + d], num);
            }
            const int blk = pos >> 7;
            __nv_bfloat16* dst = comp_kv + (static_cast<size_t>(slot) * max_comp_blks + blk) * geom::HCA_HD;
            hca_commit_c128_dim(num / denom, d, blk * geom::HCA_CR, comp_norm_w, inv_freq, dst, s_comp);
            __syncthreads();
        }
    }
}

// Prefill parallel C128 block compressor: 1 CTA per completed 128-token block inside [pos0, pos0 + len).
// Grid: (n_complete_blks), Block: 512.
__global__ void __launch_bounds__(512) k_ernie55_hca_comp_seg_complete(
    const float* __restrict__ c_kv,
    const float* __restrict__ c_sc,
    const float* __restrict__ comp_ape,
    const __nv_bfloat16* __restrict__ comp_norm_w,
    const float* __restrict__ inv_freq,
    const float* __restrict__ part_kv,
    const float* __restrict__ part_sc,
    __nv_bfloat16* __restrict__ comp_kv,
    int max_comp_blks, int slot, int row0, int pos0, int len) {
    const int d = threadIdx.x; // 0..511
    const int blk = (pos0 >> 7) + blockIdx.x;
    const int bp0 = blk * geom::HCA_CR;
    if (bp0 + geom::HCA_CR > pos0 + len) return;

    const float* pkv = part_kv + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;
    const float* psc = part_sc + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;

    float mx = -INFINITY;
    for (int k = 0; k < geom::HCA_CR; ++k) {
        const int p = bp0 + k;
        const float sc = (p < pos0)
            ? psc[static_cast<size_t>(k) * geom::HCA_HD + d]
            : (c_sc[static_cast<size_t>(row0 + (p - pos0)) * geom::HCA_HD + d] + comp_ape[static_cast<size_t>(k) * geom::HCA_HD + d]);
        mx = fmaxf(mx, sc);
    }
    float denom = 0.0f, num = 0.0f;
    for (int k = 0; k < geom::HCA_CR; ++k) {
        const int p = bp0 + k;
        const float sc = (p < pos0)
            ? psc[static_cast<size_t>(k) * geom::HCA_HD + d]
            : (c_sc[static_cast<size_t>(row0 + (p - pos0)) * geom::HCA_HD + d] + comp_ape[static_cast<size_t>(k) * geom::HCA_HD + d]);
        const float kv = (p < pos0)
            ? pkv[static_cast<size_t>(k) * geom::HCA_HD + d]
            : c_kv[static_cast<size_t>(row0 + (p - pos0)) * geom::HCA_HD + d];
        const float w = expf(sc - mx);
        denom += w;
        num = fmaf(w, kv, num);
    }
    __shared__ float s_comp[geom::HCA_HD];
    __nv_bfloat16* dst = comp_kv + (static_cast<size_t>(slot) * max_comp_blks + blk) * geom::HCA_HD;
    hca_commit_c128_dim(num / denom, d, bp0, comp_norm_w, inv_freq, dst, s_comp);
}

// Prefill tail copy: save remaining partial tokens [tail_b0 .. pos0 + len - 1] into part_kv/part_sc.
// Grid: (tail_len), Block: 512.
__global__ void __launch_bounds__(512) k_ernie55_hca_comp_seg_tail(
    const float* __restrict__ c_kv,
    const float* __restrict__ c_sc,
    const float* __restrict__ comp_ape,
    float* __restrict__ part_kv,
    float* __restrict__ part_sc,
    int slot, int row0, int pos0, int len) {
    const int d = threadIdx.x;
    const int seg_end = pos0 + len;
    const int tail_b0 = (seg_end >> 7) << 7;
    const int first_p = max(pos0, tail_b0);
    const int p = first_p + blockIdx.x;
    if (p >= seg_end) return;
    const int sub = p & (geom::HCA_CR - 1);
    const int r = row0 + (p - pos0);
    float* pkv = part_kv + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;
    float* psc = part_sc + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD;
    pkv[static_cast<size_t>(sub) * geom::HCA_HD + d] = c_kv[static_cast<size_t>(r) * geom::HCA_HD + d];
    psc[static_cast<size_t>(sub) * geom::HCA_HD + d] =
        c_sc[static_cast<size_t>(r) * geom::HCA_HD + d] + comp_ape[static_cast<size_t>(sub) * geom::HCA_HD + d];
}

// =================================================================================================
// Tensor-Core HCA Attention (mma.m16n8k16): 16 query heads x 32 keys, 8 warps split 512 dims (64/warp).
// Supports BOTH decode (with split-KV NS) and prefill (1 CTA per (m, hg) when NS == 1)!
// Grid: (M * (nh / 16) * NS), Block: 256.
// =================================================================================================
constexpr int HCA_TC_SMEM = 32 * 520 * 2 + 8 * 16 * 32 * 4; // 33,280 + 16,384 = 49,664 B

__global__ void __launch_bounds__(256) k_ernie55_hca_attn_tc(
    const __nv_bfloat16* __restrict__ q,        // [M, nh, 512]
    const float* __restrict__ sinks,            // [nh] F32
    const __nv_bfloat16* __restrict__ comp_kv,  // [slots, max_comp_blks, 512]
    const __nv_bfloat16* __restrict__ swa_ring, // [slots, HCA_SWA_RING, 512]
    const int* __restrict__ d_slot,
    const int* __restrict__ d_pos,
    __nv_bfloat16* __restrict__ out,            // [M, nh, 512] (NS == 1)
    float* __restrict__ split_m,                // [M, NS, nh] (NS > 1)
    float* __restrict__ split_l,                // [M, NS, nh] (NS > 1)
    float* __restrict__ split_o,                // [M, NS, nh, 512] (NS > 1)
    int M, int nh, int max_comp_blks, int NS) {
    const int n_hg = nh >> 4;
    const int idx = blockIdx.x;
    const int sp = idx % NS;
    const int rem = idx / NS;
    const int hg = rem % n_hg;
    const int m  = rem / n_hg;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5; // 0..7 -> owns dims [warp*64 .. warp*64+63]
    const int lane = tid & 31;
    const int rg = lane >> 2;  // 0..7
    const int tc = lane & 3;   // 0..3
    const int h0 = hg * 16;

    const int slot = d_slot[m];
    const int pos  = d_pos[m];
    const int n_comp  = (pos + 1) >> 7;
    const int w_start = max(0, pos - geom::SLIDING_WIN + 1);
    const int n_swa   = pos - w_start + 1;
    const int total_k = n_comp + n_swa;

    const int per = (total_k + NS - 1) / NS;
    const int k_start = sp * per;
    const int k_end   = min(total_k, k_start + per);

    // Preload Q[16, 64] for this warp into registers
    const __nv_bfloat16* qp_a = q + (static_cast<size_t>(m) * nh + h0 + rg) * geom::HCA_HD;
    const __nv_bfloat16* qp_b = q + (static_cast<size_t>(m) * nh + h0 + rg + 8) * geom::HCA_HD;
    uint32_t qn[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c = warp * 64 + ks * 16 + 2 * tc;
        qn[ks][0] = *reinterpret_cast<const uint32_t*>(qp_a + c);
        qn[ks][1] = *reinterpret_cast<const uint32_t*>(qp_b + c);
        qn[ks][2] = *reinterpret_cast<const uint32_t*>(qp_a + c + 8);
        qn[ks][3] = *reinterpret_cast<const uint32_t*>(qp_b + c + 8);
    }

    float m_a = (sp == 0) ? sinks[h0 + rg]     : -INFINITY;
    float m_b = (sp == 0) ? sinks[h0 + rg + 8] : -INFINITY;
    float l_a = (sp == 0) ? 1.0f : 0.0f;
    float l_b = (sp == 0) ? 1.0f : 0.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 (*sKV)[520] = reinterpret_cast<__nv_bfloat16 (*)[520]>(sm_raw);
    float (*sPart)[16][32] = reinterpret_cast<float (*)[16][32]>(sm_raw + 32 * 520 * 2);

    constexpr int TILE_K = 32;
    constexpr float ATTN_SCALE = 0.04419417382415922f; // 512^-0.5

    const __nv_bfloat16* comp_slot = comp_kv + static_cast<size_t>(slot) * max_comp_blks * geom::HCA_HD;
    const __nv_bfloat16* swa_slot  = swa_ring + static_cast<size_t>(slot) * geom::HCA_SWA_RING * geom::HCA_HD;

    for (int k0 = k_start; k0 < k_end; k0 += TILE_K) {
        const int nk = min(TILE_K, k_end - k0);
        const int total_u4 = nk * 64; // 512 bf16 = 64 uint4
        for (int idx4 = tid; idx4 < total_u4; idx4 += 256) {
            const int tk = idx4 >> 6;
            const int u4 = idx4 & 63;
            const int ki = k0 + tk;
            const __nv_bfloat16* src = (ki < n_comp)
                ? (comp_slot + static_cast<size_t>(ki) * geom::HCA_HD)
                : (swa_slot + static_cast<size_t>((w_start + (ki - n_comp)) & (geom::HCA_SWA_RING - 1)) * geom::HCA_HD);
            *reinterpret_cast<uint4*>(sKV[tk] + u4 * 8) = reinterpret_cast<const uint4*>(src)[u4];
        }
        __syncthreads();

        float s_loc[4][4];
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) s_loc[kt][0] = s_loc[kt][1] = s_loc[kt][2] = s_loc[kt][3] = 0.0f;

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int tk = kt * 8 + rg;
            const bool valid_tk = (tk < nk);
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int c = warp * 64 + ks * 16 + 2 * tc;
                const uint32_t b0 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c)     : 0u;
                const uint32_t b1 = valid_tk ? *reinterpret_cast<const uint32_t*>(sKV[tk] + c + 8) : 0u;
                mma16816_bf16(s_loc[kt], qn[ks][0], qn[ks][1], qn[ks][2], qn[ks][3], b0, b1);
            }
            sPart[warp][rg][kt * 8 + 2 * tc]         = s_loc[kt][0];
            sPart[warp][rg][kt * 8 + 2 * tc + 1]     = s_loc[kt][1];
            sPart[warp][rg + 8][kt * 8 + 2 * tc]     = s_loc[kt][2];
            sPart[warp][rg + 8][kt * 8 + 2 * tc + 1] = s_loc[kt][3];
        }
        __syncthreads();

        float sc_a[8], sc_b[8];
        float tmax_a = -INFINITY, tmax_b = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int c0 = kt * 8 + 2 * tc;
            const int c1 = c0 + 1;
            float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sa0 += sPart[w][rg][c0];     sa1 += sPart[w][rg][c1];
                sb0 += sPart[w][rg + 8][c0]; sb1 += sPart[w][rg + 8][c1];
            }
            sc_a[2 * kt]     = (c0 < nk) ? (sa0 * ATTN_SCALE) : -INFINITY;
            sc_a[2 * kt + 1] = (c1 < nk) ? (sa1 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt]     = (c0 < nk) ? (sb0 * ATTN_SCALE) : -INFINITY;
            sc_b[2 * kt + 1] = (c1 < nk) ? (sb1 * ATTN_SCALE) : -INFINITY;
            tmax_a = fmaxf(tmax_a, fmaxf(sc_a[2 * kt], sc_a[2 * kt + 1]));
            tmax_b = fmaxf(tmax_b, fmaxf(sc_b[2 * kt], sc_b[2 * kt + 1]));
        }
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 1));
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 2));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 1));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 2));

        const float m_new_a = fmaxf(m_a, tmax_a);
        const float m_new_b = fmaxf(m_b, tmax_b);
        const float alpha_a = (m_a > -1e30f) ? __expf(m_a - m_new_a) : 0.0f;
        const float alpha_b = (m_b > -1e30f) ? __expf(m_b - m_new_b) : 0.0f;

        float l_tile_a = 0.0f, l_tile_b = 0.0f;
        float pa[8], pb[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa[i] = (sc_a[i] > -1e30f) ? __expf(sc_a[i] - m_new_a) : 0.0f;
            pb[i] = (sc_b[i] > -1e30f) ? __expf(sc_b[i] - m_new_b) : 0.0f;
            l_tile_a += pa[i];
            l_tile_b += pb[i];
        }
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 1);
        l_tile_a += __shfl_xor_sync(0xffffffffu, l_tile_a, 2);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 1);
        l_tile_b += __shfl_xor_sync(0xffffffffu, l_tile_b, 2);

        l_a = l_a * alpha_a + l_tile_a;
        l_b = l_b * alpha_b + l_tile_b;
        m_a = m_new_a;
        m_b = m_new_b;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha_a; o_acc[nt][1] *= alpha_a;
            o_acc[nt][2] *= alpha_b; o_acc[nt][3] *= alpha_b;
        }

        const uint32_t pf0[4] = { pack_bf2(pa[0], pa[1]), pack_bf2(pb[0], pb[1]), pack_bf2(pa[2], pa[3]), pack_bf2(pb[2], pb[3]) };
        const uint32_t pf1[4] = { pack_bf2(pa[4], pa[5]), pack_bf2(pb[4], pb[5]), pack_bf2(pa[6], pa[7]), pack_bf2(pb[6], pb[7]) };

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int dcol = warp * 64 + nt * 8 + rg;
            const int k00 = 2 * tc, k01 = k00 + 1, k02 = k00 + 8, k03 = k02 + 1;
            const uint32_t vb0 = pack_bf2(
                k00 < nk ? __bfloat162float(sKV[k00][dcol]) : 0.0f,
                k01 < nk ? __bfloat162float(sKV[k01][dcol]) : 0.0f);
            const uint32_t vb1 = pack_bf2(
                k02 < nk ? __bfloat162float(sKV[k02][dcol]) : 0.0f,
                k03 < nk ? __bfloat162float(sKV[k03][dcol]) : 0.0f);
            mma16816_bf16(o_acc[nt], pf0[0], pf0[1], pf0[2], pf0[3], vb0, vb1);

            if (nk > 16) {
                const int k10 = 16 + 2 * tc, k11 = k10 + 1, k12 = k10 + 8, k13 = k12 + 1;
                const uint32_t vb2 = pack_bf2(
                    k10 < nk ? __bfloat162float(sKV[k10][dcol]) : 0.0f,
                    k11 < nk ? __bfloat162float(sKV[k11][dcol]) : 0.0f);
                const uint32_t vb3 = pack_bf2(
                    k12 < nk ? __bfloat162float(sKV[k12][dcol]) : 0.0f,
                    k13 < nk ? __bfloat162float(sKV[k13][dcol]) : 0.0f);
                mma16816_bf16(o_acc[nt], pf1[0], pf1[1], pf1[2], pf1[3], vb2, vb3);
            }
        }
        __syncthreads();
    }

    if (NS == 1) {
        const float inv_a = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
        const float inv_b = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
        __nv_bfloat16* da = out + (static_cast<size_t>(m) * nh + h0 + rg)     * geom::HCA_HD + warp * 64;
        __nv_bfloat16* db = out + (static_cast<size_t>(m) * nh + h0 + rg + 8) * geom::HCA_HD + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            *reinterpret_cast<uint32_t*>(da + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][0] * inv_a, o_acc[nt][1] * inv_a);
            *reinterpret_cast<uint32_t*>(db + nt * 8 + 2 * tc) = pack_bf2(o_acc[nt][2] * inv_b, o_acc[nt][3] * inv_b);
        }
    } else {
        const size_t sia = (static_cast<size_t>(m) * NS + sp) * nh + (h0 + rg);
        const size_t sib = (static_cast<size_t>(m) * NS + sp) * nh + (h0 + rg + 8);
        if (warp == 0 && tc == 0) {
            split_m[sia] = m_a; split_l[sia] = l_a;
            split_m[sib] = m_b; split_l[sib] = l_b;
        }
        float* soa = split_o + sia * geom::HCA_HD + warp * 64;
        float* sob = split_o + sib * geom::HCA_HD + warp * 64;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            soa[nt * 8 + 2 * tc]     = o_acc[nt][0];
            soa[nt * 8 + 2 * tc + 1] = o_acc[nt][1];
            sob[nt * 8 + 2 * tc]     = o_acc[nt][2];
            sob[nt * 8 + 2 * tc + 1] = o_acc[nt][3];
        }
    }
}

inline void ensure_hca_tc_smem() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(k_ernie55_hca_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, HCA_TC_SMEM);
}

__global__ void __launch_bounds__(256) k_ernie55_hca_split_reduce(
    const float* __restrict__ split_m,
    const float* __restrict__ split_l,
    const float* __restrict__ split_o,
    __nv_bfloat16* __restrict__ out,
    int M, int nh, int NS) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int tid = threadIdx.x;

    float gm = -INFINITY;
    for (int sp = 0; sp < NS; ++sp) {
        gm = fmaxf(gm, split_m[(static_cast<size_t>(m) * NS + sp) * nh + h]);
    }
    float gl = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int sp = 0; sp < NS; ++sp) {
        const size_t sidx = (static_cast<size_t>(m) * NS + sp) * nh + h;
        const float sl = split_l[sidx];
        if (sl <= 0.0f) continue;
        const float w = __expf(split_m[sidx] - gm);
        gl += sl * w;
        const float* so = split_o + sidx * geom::HCA_HD;
        o0 = fmaf(so[tid],       w, o0);
        o1 = fmaf(so[tid + 256], w, o1);
    }
    const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    __nv_bfloat16* dst = out + (static_cast<size_t>(m) * nh + h) * geom::HCA_HD;
    dst[tid]       = __float2bfloat16(o0 * inv_l);
    dst[tid + 256] = __float2bfloat16(o1 * inv_l);
}

// Fused VHAPostMix (I + V @ U^T) + Inverse YaRN RoPE on trailing 64 dims.
// o_in [M, nh, 512] -> o_out [M, nh, 512] (each block owns 1 token m and 1 output head h).
// Grid: (M * nh), Block: 256.
__global__ void __launch_bounds__(256) k_ernie55_hca_vha_invrope(
    const __nv_bfloat16* __restrict__ o_in,
    const float* __restrict__ vha_mix, // [64, 64]
    const int* __restrict__ d_pos,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ o_out,
    int M, int nh, int h_off) {
    const int idx = blockIdx.x;
    if (idx >= M * nh) return;
    const int m = idx / nh;
    const int h = idx % nh;
    const int tid = threadIdx.x;

    const __nv_bfloat16* om = o_in + static_cast<size_t>(m) * nh * geom::HCA_HD;
    const float* mrow = vha_mix + static_cast<size_t>(h_off + h) * geom::HEADS + h_off;

    float v0 = 0.0f, v1 = 0.0f;
    for (int g = 0; g < nh; ++g) {
        const float w = mrow[g];
        const __nv_bfloat16* og = om + static_cast<size_t>(g) * geom::HCA_HD;
        v0 = fmaf(w, __bfloat162float(og[tid]), v0);
        v1 = fmaf(w, __bfloat162float(og[tid + 256]), v1);
    }

    __shared__ float s_o[geom::HCA_HD];
    s_o[tid]       = __bfloat162float(__float2bfloat16(v0));
    s_o[tid + 256] = __bfloat162float(__float2bfloat16(v1));
    __syncthreads();

    __nv_bfloat16* dst = o_out + static_cast<size_t>(idx) * geom::HCA_HD;
    if (tid < geom::HCA_NOPE / 2) {
        dst[2 * tid]     = __float2bfloat16(s_o[2 * tid]);
        dst[2 * tid + 1] = __float2bfloat16(s_o[2 * tid + 1]);
    } else if (tid < geom::HCA_NOPE / 2 + geom::HCA_ROPE / 2) {
        const int p = tid - geom::HCA_NOPE / 2;
        float x0 = s_o[geom::HCA_NOPE + 2 * p];
        float x1 = s_o[geom::HCA_NOPE + 2 * p + 1];
        hca_rope_pair(x0, x1, d_pos[m], inv_freq[p], true);
        dst[geom::HCA_NOPE + 2 * p]     = __float2bfloat16(x0);
        dst[geom::HCA_NOPE + 2 * p + 1] = __float2bfloat16(x1);
    }
}

// Gather group g [M, 4096] from post_o [M, ng_loc, 4096] -> o_grp_in [M, 4096].
__global__ void __launch_bounds__(256) k_ernie55_hca_gather_grp(
    const __nv_bfloat16* __restrict__ post_o,
    __nv_bfloat16* __restrict__ grp_in,
    int M, int ng_loc, int g) {
    const int m = blockIdx.x / 16;
    const int d = (blockIdx.x % 16) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HCA_HPG * geom::HCA_HD) return;
    grp_in[static_cast<size_t>(m) * 4096 + d] =
        post_o[(static_cast<size_t>(m) * ng_loc + g) * 4096 + d];
}

// Scatter z_g [M, 1024] * sigmoid(gate[m, g*1024..]) -> z_cat[m, g*1024..].
__global__ void __launch_bounds__(256) k_ernie55_hca_scatter_gate(
    const __nv_bfloat16* __restrict__ z_g,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ z_cat,
    int M, int ng_loc, int g) {
    const int m = blockIdx.x / 4;
    const int d = (blockIdx.x % 4) * 256 + threadIdx.x;
    if (m >= M || d >= geom::HCA_O_LORA) return;
    const size_t cat_idx = (static_cast<size_t>(m) * ng_loc + g) * geom::HCA_O_LORA + d;
    const float zv = __bfloat162float(z_g[static_cast<size_t>(m) * geom::HCA_O_LORA + d]);
    const float gv = sigmoid_f(__bfloat162float(gate[cat_idx]));
    z_cat[cat_idx] = __float2bfloat16(zv * gv);
}

// Full HCA C128 Forward -> out_part [M, 4096] BF16.
inline void hca_forward(
    cublasHandle_t blas, const HcaWeights& w, HcaScratch& sc, HcaLayerKv& kv,
    const __nv_bfloat16* x_norm, __nv_bfloat16* out_part,
    const StepSeg* segs, int nsegs,
    const int* d_slot, const int* d_pos, const float* d_yarn_inv_freq,
    int M, int max_pos, uint8_t* x8_scratch, cudaStream_t st, int graph_ns = 0) {
    // 1. Pre-quantize x_norm once for M >= 16 MXFP8 reuse across q_down, kv_proj, comp_wkv, comp_wgate!
    const uint8_t* x_x8 = nullptr;
    if (M >= 16 && w.q_down.is_mx8() && x8_scratch != nullptr) {
        mx8::bf16_to_mx8_a(x_norm, x8_scratch, M, geom::HID, st);
        x_x8 = x8_scratch;
    }

    // 2. Q_lora + RMSNorm, KV + RMSNorm + RoPE -> SWA ring, Compressor inputs (F32)
    mx8::linear_w(blas, x_norm, w.q_down, sc.q_lora, M, geom::HCA_Q_LORA, geom::HID, x8_scratch, x_x8, st);
    k_ernie55_rmsnorm<<<M, 256, 0, st>>>(sc.q_lora, w.q_norm, sc.q_lora, M, geom::HCA_Q_LORA, geom::RMS_EPS);

    mx8::linear_w(blas, x_norm, w.kv_proj, sc.kv_raw, M, geom::HCA_HD, geom::HID, x8_scratch, x_x8, st);
    k_ernie55_hca_kv_write<<<M, 256, 0, st>>>(sc.kv_raw, w.kv_norm, d_slot, d_pos, d_yarn_inv_freq, kv.swa_ring, M);

    mx8::linear_w_f32(blas, x_norm, w.comp_wkv,   sc.c_kv, M, geom::HCA_HD, geom::HID, x8_scratch, x_x8, st);
    mx8::linear_w_f32(blas, x_norm, w.comp_wgate, sc.c_sc, M, geom::HCA_HD, geom::HID, x8_scratch, x_x8, st);

    // 3. C128 compressor update (device-driven for small-M/graph, parallel per-block for prefill)
    if (graph_ns > 0 || M <= 12) {
        k_ernie55_hca_comp_smallm<<<1, 512, 0, st>>>(
            sc.c_kv, sc.c_sc, w.comp_ape, w.comp_norm, d_yarn_inv_freq,
            d_slot, d_pos, kv.part_kv, kv.part_sc, kv.comp_kv, M, kv.max_comp_blks);
    } else {
        for (int s = 0; s < nsegs; ++s) {
            const StepSeg& sg = segs[s];
            if (sg.len <= 0) continue;
            const int b_first = sg.pos0 >> 7;
            const int b_end   = (sg.pos0 + sg.len) >> 7;
            if (b_end > b_first) {
                k_ernie55_hca_comp_seg_complete<<<b_end - b_first, 512, 0, st>>>(
                    sc.c_kv, sc.c_sc, w.comp_ape, w.comp_norm, d_yarn_inv_freq,
                    kv.part_kv, kv.part_sc, kv.comp_kv, kv.max_comp_blks,
                    sg.slot, sg.row0, sg.pos0, sg.len);
            }
            const int seg_end = sg.pos0 + sg.len;
            const int tail_b0 = (seg_end >> 7) << 7;
            const int first_p = std::max(sg.pos0, tail_b0);
            if (first_p < seg_end) {
                k_ernie55_hca_comp_seg_tail<<<seg_end - first_p, 512, 0, st>>>(
                    sc.c_kv, sc.c_sc, w.comp_ape, kv.part_kv, kv.part_sc,
                    sg.slot, sg.row0, sg.pos0, sg.len);
            }
        }
    }

    // 4. Q_up + weightless RMSNorm + YaRN RoPE, and gate_proj(q_lora)
    const uint8_t* ql_x8 = nullptr;
    if (M >= 16 && w.q_up.is_mx8() && x8_scratch != nullptr) {
        mx8::bf16_to_mx8_a(sc.q_lora, x8_scratch, M, geom::HCA_Q_LORA, st);
        ql_x8 = x8_scratch;
    }
    mx8::linear_w(blas, sc.q_lora, w.q_up,      sc.q_raw, M, w.nh_loc * geom::HCA_HD, geom::HCA_Q_LORA, x8_scratch, ql_x8, st);
    mx8::linear_w(blas, sc.q_lora, w.gate_proj, sc.gate,  M, w.ng_loc * geom::HCA_O_LORA, geom::HCA_Q_LORA, x8_scratch, ql_x8, st);
    k_ernie55_hca_q_norm_rope<<<M * w.nh_loc, 256, 0, st>>>(sc.q_raw, d_pos, d_yarn_inv_freq, M, w.nh_loc);

    // 5. Tensor-Core HCA Attention
    ensure_hca_tc_smem();
    const int n_hg = w.nh_loc >> 4;
    const int est_k = ((max_pos + 1) >> 7) + std::min(max_pos + 1, geom::SLIDING_WIN);
    int NS = 1;
    if (graph_ns > 0) {
        NS = graph_ns;
    } else if (M <= 12 && est_k >= 256) {
        NS = std::min(HCA_MAX_SPLITS, std::max(1, std::min(16 / M, est_k / 128)));
    }
    k_ernie55_hca_attn_tc<<<M * n_hg * NS, 256, HCA_TC_SMEM, st>>>(
        sc.q_raw, w.sinks, kv.comp_kv, kv.swa_ring, d_slot, d_pos,
        sc.attn_o, sc.split_m, sc.split_l, sc.split_o, M, w.nh_loc, kv.max_comp_blks, NS);
    if (NS > 1) {
        k_ernie55_hca_split_reduce<<<M * w.nh_loc, 256, 0, st>>>(
            sc.split_m, sc.split_l, sc.split_o, sc.attn_o, M, w.nh_loc, NS);
    }

    // 6. VHAPostMix + Inverse YaRN RoPE
    k_ernie55_hca_vha_invrope<<<M * w.nh_loc, 256, 0, st>>>(
        sc.attn_o, w.vha_mix, d_pos, d_yarn_inv_freq, sc.post_o, M, w.nh_loc, w.h_off);

    // 7. Grouped wo_a (ng_loc groups of [1024, 4096]) * sigmoid(gate) -> o_proj
    for (int g = 0; g < w.ng_loc; ++g) {
        const __nv_bfloat16* gin = sc.post_o;
        if (w.ng_loc > 1) {
            k_ernie55_hca_gather_grp<<<M * 16, 256, 0, st>>>(sc.post_o, sc.o_grp_in, M, w.ng_loc, g);
            gin = sc.o_grp_in;
        }
        mx8::linear_w(blas, gin, w.wo_a[g], sc.o_grp_z, M, geom::HCA_O_LORA, geom::HCA_HPG * geom::HCA_HD, x8_scratch, nullptr, st);
        k_ernie55_hca_scatter_gate<<<M * 4, 256, 0, st>>>(sc.o_grp_z, sc.gate, sc.z_cat, M, w.ng_loc, g);
    }
    mx8::linear_w(blas, sc.z_cat, w.o_proj, out_part, M, geom::HID, w.ng_loc * geom::HCA_O_LORA, x8_scratch, nullptr, st);
}

} // namespace ernie55
