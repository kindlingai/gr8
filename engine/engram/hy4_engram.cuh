// Tencent Hy4-flash Device Engram Kernels (layers 1, 11, 21, 31).
// Verified verbatim against HF PR 50903 modeling_hy_v4_flash.py (HYV4FlashEngramLayer) & PyTorch 2.15:
//   - PyTorch nn.RMSNorm(dim, eps=None) uses finfo(float32).eps = FLT_EPSILON = 1.1920928955078125e-07f!
//   1. Dequantize 128-byte raw row (96 FP8-E4M3FN bytes + BF16 scale at offset 96) -> exact _fp8_round_trip BF16 [M, 16, 96].
//   2. Per-head nn.RMSNorm(96, eps = ENGRAM_RMS_EPS) -> flatten [M, 1536] BF16.
//   3. val = value_proj(emb) + value_bias          [M, 3072] BF16
//      key[h] = key_projs[h](emb) + key_bias[h]    [M, 3072] BF16, h = 0..3
//   4. main_signal = unified_pre_gamma * unified_rmsnorm(val)   [M, 3072] BF16
//   5. For each hc in 0..3:
//        kn = norm_key[hc](key[hc]), qn = norm_query[hc](hyper[:, hc, :])
//        d  = (kn * qn).sum(-1) * rsqrt(3072)
//        g  = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
//        hyper[:, hc, :] += g * main_signal + unified_per_hc_scales[hc] * main_signal
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "../model_geom_hy4.h"
#include "../kernels/hy4_hc.cuh"
#include "../kernels/hy4_mxfp8.cuh"

namespace hy4 {

// Empirically verified against PyTorch 2.15 nn.RMSNorm(96, dtype=bf16, eps=None): uses FLT_EPSILON!
constexpr float ENGRAM_RMS_EPS = 1.1920928955078125e-07f;

struct EngramWeights {
    const __nv_bfloat16* head_norms[16] = {};     // [16] x [96] BF16
    mx8::LinW            value_w{};               // [3072, 1536] BF16 or MXFP8
    const __nv_bfloat16* value_b        = nullptr;// [3072] BF16
    mx8::LinW            key_w[4]{};              // [4] x [3072, 1536] BF16 or MXFP8
    const __nv_bfloat16* key_b[4]       = {};     // [4] x [3072] BF16
    const __nv_bfloat16* norm_key[4]    = {};     // [4] x [3072] BF16
    const __nv_bfloat16* norm_query[4]  = {};     // [4] x [3072] BF16
    const __nv_bfloat16* pre_gamma      = nullptr;// [3072] BF16
    const __nv_bfloat16* hc_scales[4]   = {};     // [4] x [3072] BF16
};

struct EngramScratch {
    uint8_t*       raw_rows = nullptr; // [max_M * 16 * 128] U8
    __nv_bfloat16* emb      = nullptr; // [max_M, 1536] BF16
    __nv_bfloat16* val      = nullptr; // [max_M, 3072] BF16
    __nv_bfloat16* main_sig = nullptr; // [max_M, 3072] BF16
    __nv_bfloat16* key      = nullptr; // [4, max_M, 3072] BF16
};

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t u) {
    const __nv_fp8_e4m3 f = *reinterpret_cast<const __nv_fp8_e4m3*>(&u);
    return static_cast<float>(f);
}

struct EngramHeadNormPtrs {
    const __nv_bfloat16* w[16];
};

// Kernel 1: Dequantize 128-byte rows (96 FP8 + BF16 scale) AND apply per-head nn.RMSNorm(96).
// Grid: (M * 16), Block: 96 threads (3 warps = exact head_dim 96).
__global__ void __launch_bounds__(96) k_hy4_engram_dequant_norm(
    const uint8_t* __restrict__ raw_rows,
    EngramHeadNormPtrs hn,
    __nv_bfloat16* __restrict__ emb_out,
    int M) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::ENGRAM_HEADS) return;
    const int h = idx % geom::ENGRAM_HEADS;
    const int d = threadIdx.x; // 0..95
    const uint8_t* row = raw_rows + static_cast<size_t>(idx) * geom::ENGRAM_ROW_BYTES;
    const uint16_t sc_bits = *reinterpret_cast<const uint16_t*>(row + 96);
    const float sc = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sc_bits));
    const float qbf = __bfloat162float(__float2bfloat16(e4m3fn_to_f32(row[d])));
    const float x = __bfloat162float(__float2bfloat16(qbf * sc));

    float ss = warp_sum(x * x);
    __shared__ float sm[3];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1] + sm[2]) / 96.0f + ENGRAM_RMS_EPS);
    const float w = __bfloat162float(hn.w[h][d]);
    emb_out[static_cast<size_t>(idx) * geom::ENGRAM_HEAD_DIM + d] = __float2bfloat16(x * inv * w);
}

// Kernel 2: unified_rmsnorm(val) * unified_pre_gamma -> main_sig [M, 3072] BF16.
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_engram_main_sig(
    const __nv_bfloat16* __restrict__ val,
    const __nv_bfloat16* __restrict__ pre_gamma,
    __nv_bfloat16* __restrict__ main_sig,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* vm = val + static_cast<size_t>(m) * geom::HID;
    float xs[12];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const float v = __bfloat162float(vm[tid + k * 256]);
        xs[k] = v;
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[8];
    ss = warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot / static_cast<float>(geom::HID) + ENGRAM_RMS_EPS);
    }
    __syncthreads();
    const float inv = sm[0];
    __nv_bfloat16* out = main_sig + static_cast<size_t>(m) * geom::HID;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float normed = __bfloat162float(__float2bfloat16(xs[k] * inv));
        out[d] = __float2bfloat16(__bfloat162float(pre_gamma[d]) * normed);
    }
}

struct EngramHcPtrs {
    const __nv_bfloat16* nk[4];
    const __nv_bfloat16* nq[4];
    const __nv_bfloat16* sc[4];
};

// Kernel 3: Per-HC norm_key + norm_query + sqrt-signed sigmoid gate + add to hyper[m, hc, :].
// Grid: (M * 4), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_engram_gate_add(
    const __nv_bfloat16* __restrict__ key_all, // [4, M, 3072]
    const __nv_bfloat16* __restrict__ main_sig,// [M, 3072]
    EngramHcPtrs p,
    __nv_bfloat16* __restrict__ hyper,         // [M, 4, 3072]
    int M,
    const uint8_t* __restrict__ d_eng_mask = nullptr) {
    const int idx = blockIdx.x;
    if (idx >= M * geom::HC) return;
    const int m = idx / geom::HC;
    if (d_eng_mask && !d_eng_mask[m]) return;
    const int hc = idx % geom::HC;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const __nv_bfloat16* km = key_all + (static_cast<size_t>(hc) * M + m) * geom::HID;
    __nv_bfloat16* qm = hyper + (static_cast<size_t>(m) * geom::HC + hc) * geom::HID;
    const __nv_bfloat16* ms = main_sig + static_cast<size_t>(m) * geom::HID;
    const __nv_bfloat16* nkw = p.nk[hc];
    const __nv_bfloat16* nqw = p.nq[hc];
    const __nv_bfloat16* hsc = p.sc[hc];

    float kv[12], qv[12];
    float kss = 0.0f, qss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        kv[k] = __bfloat162float(km[d]);
        qv[k] = __bfloat162float(qm[d]);
        kss = fmaf(kv[k], kv[k], kss);
        qss = fmaf(qv[k], qv[k], qss);
    }
    __shared__ float sm_k[8], sm_q[8], sm_dp[8];
    kss = warp_sum(kss);
    qss = warp_sum(qss);
    if (lane == 0) { sm_k[warp] = kss; sm_q[warp] = qss; }
    __syncthreads();
    if (tid == 0) {
        float tk = 0.0f, tq = 0.0f;
        for (int w = 0; w < 8; ++w) { tk += sm_k[w]; tq += sm_q[w]; }
        sm_k[0] = rsqrtf(tk / static_cast<float>(geom::HID) + ENGRAM_RMS_EPS);
        sm_q[0] = rsqrtf(tq / static_cast<float>(geom::HID) + ENGRAM_RMS_EPS);
    }
    __syncthreads();
    const float inv_k = sm_k[0], inv_q = sm_q[0];

    float dp = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float kn = __bfloat162float(__float2bfloat16(kv[k] * inv_k * __bfloat162float(nkw[d])));
        const float qn = __bfloat162float(__float2bfloat16(qv[k] * inv_q * __bfloat162float(nqw[d])));
        dp = fmaf(kn, qn, dp);
    }
    dp = warp_sum(dp);
    if (lane == 0) sm_dp[warp] = dp;
    __syncthreads();
    if (tid == 0) {
        float dsum = 0.0f;
        for (int w = 0; w < 8; ++w) dsum += sm_dp[w];
        const float ds = dsum * 0.018042195912175808f; // 3072^-0.5
        const float sgn = (ds > 0.0f) ? 1.0f : ((ds < 0.0f) ? -1.0f : 0.0f);
        const float g_in = sqrtf(fmaxf(fabsf(ds), 1e-6f)) * sgn;
        sm_dp[0] = sigmoid_f(g_in);
    }
    __syncthreads();
    const float g = sm_dp[0];

    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int d = tid + k * 256;
        const float msv = __bfloat162float(ms[d]);
        const float g_ms = __bfloat162float(__float2bfloat16(g * msv));
        const float br   = __bfloat162float(__float2bfloat16(__bfloat162float(hsc[d]) * msv));
        const float add  = __bfloat162float(__float2bfloat16(g_ms + br));
        qm[d] = __float2bfloat16(qv[k] + add);
    }
}

inline void engram_forward(cublasHandle_t blas, const EngramWeights& w, EngramScratch& sc,
                           __nv_bfloat16* hyper, int M, uint8_t* x8_scratch, cudaStream_t st,
                           const uint8_t* d_eng_mask = nullptr) {
    EngramHeadNormPtrs hn{};
    for (int h = 0; h < 16; ++h) hn.w[h] = w.head_norms[h];
    k_hy4_engram_dequant_norm<<<M * geom::ENGRAM_HEADS, geom::ENGRAM_HEAD_DIM, 0, st>>>(
        sc.raw_rows, hn, sc.emb, M);

    const uint8_t* emb_x8 = nullptr;
    if (M >= 16 && w.value_w.is_mx8() && x8_scratch != nullptr) {
        mx8::bf16_to_mx8_a(sc.emb, x8_scratch, M, geom::ENGRAM_DIM, st);
        emb_x8 = x8_scratch;
    }
    mx8::linear_w(blas, sc.emb, w.value_w, sc.val, M, geom::HID, geom::ENGRAM_DIM, x8_scratch, emb_x8, st, w.value_b);
    for (int hc = 0; hc < geom::HC; ++hc) {
        mx8::linear_w(blas, sc.emb, w.key_w[hc], sc.key + static_cast<size_t>(hc) * M * geom::HID,
                      M, geom::HID, geom::ENGRAM_DIM, x8_scratch, emb_x8, st, w.key_b[hc]);
    }
    k_hy4_engram_main_sig<<<M, 256, 0, st>>>(sc.val, w.pre_gamma, sc.main_sig, M);
    EngramHcPtrs p{};
    for (int hc = 0; hc < geom::HC; ++hc) {
        p.nk[hc] = w.norm_key[hc];
        p.nq[hc] = w.norm_query[hc];
        p.sc[hc] = w.hc_scales[hc];
    }
    k_hy4_engram_gate_add<<<M * geom::HC, 256, 0, st>>>(sc.key, sc.main_sig, p, hyper, M, d_eng_mask);
}

} // namespace hy4
