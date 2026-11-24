// QSpine-DF2: Native Qwen3.8-Flash-Next Parallel Block Speculative Drafter (DFlash2 + head2_qw38 + Calibrated Viterbi DP).
//
// Directly loads Apache-2.0 incoai/Qwen3.8-Flash-Next-DFlash2 DF2P packs (q4_0 body + bf16 selector codebooks)
// as well as Stage-A adapted DF2P packs (optional dp_cal [4,7] f32 + slot1.down/up [64,2560]/[2560,64] f32).
//
// Key design invariants:
//   - Always executes frozen bidirectional B = 8 block forward; truncates proposals/verify to K in 1..7.
//   - Body is replicated across TP ranks (zero body collectives, 100% identical activations across TP=1..4).
//   - Draft lm_head is TP-sharded (head2_qw38 2-bit screen + exact block-32 FP8 rescore -> local sorted top-16),
//     merged globally via a single 896 B/rank exchange (K=7 x 16 x 8 B) in k_walk_dp.
//   - Context KV ring is FP16 [L][NKV][RING][HD] (80.0 MiB at RING=4096), updated ONLY on committed target tokens;
//     block K/V stay in a tiny [B][NKV][HD] scratch buffer so rejected drafts never pollute the ring.
//   - Supports BOTH stock myopic greedy selector walk (use_dp=0) AND Calibrated Expected-Reach Viterbi DP (use_dp=1)
//     with OOV/"none-of-these" residual mass calibration feeding DflOut::lp for S2 KStop / --spec-adaptive.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../model_geom_qw38.h"
#include "../kernels/qw38_fp8.cuh"
#include "../kernels/qw38_mtp_head2.cuh"

namespace qw38 {
namespace dfl {

constexpr int L = 5;
constexpr int NT = 5;
constexpr int TAPS[NT] = {5, 15, 25, 35, 45};
constexpr int HID = geom::HID;           // 2560
constexpr int NH = 32;
constexpr int NKV = 8;
constexpr int GQA = NH / NKV;            // 4
constexpr int HD = 128;
constexpr int INTER = 7680;
constexpr int B = 8;                     // frozen bidirectional block size
constexpr int K_MAX = B - 1;             // 7 max drafts
constexpr int WIN = 2048;
constexpr int RING = 4096;               // power-of-two FP16 ring (80.0 MiB)
constexpr int GS = 16;
constexpr int NG = HID / GS;             // 160
constexpr int CPO = 4 * NG;              // 640 (2 sides x 2 taps x 160 groups)
constexpr int TOPK = 16;
constexpr int SELR = 256;
constexpr int MASK_ID = 248077;
constexpr float ROPE_THETA = 10000.0f;
constexpr float LOG_ROPE_THETA = 9.210340371976184f; // log(10000.0)
constexpr float RMS_EPS = 1e-6f;
constexpr int ATTN_SPLIT = 4;

struct Q4Mat {
    const uint8_t* qs = nullptr;   // [rows, cols/2] 16-byte aligned (1 uint4 = 32 4-bit weights: lo=0..15, hi=16..31)
    const __half* sc = nullptr;    // [rows, cols/32] FP16 block-32 scales
    int rows = 0;
    int cols = 0;
};

struct DpCal {
    float gamma[K_MAX];  // temperature per offset
    float beta0[K_MAX];  // OOV intercept
    float beta1[K_MAX];  // OOV coeff on max logit
    float beta2[K_MAX];  // OOV coeff on (top1 - top2) margin
    DpCal() {
        // Empirical 6-domain bit300 calibration (overridden by qspine.dp_cal tensor when present in pack)
        const float dg[K_MAX]  = {0.882f, 0.722f, 0.668f, 0.639f, 0.634f, 0.610f, 0.597f};
        const float db0[K_MAX] = {4.576f, 2.586f, 3.436f, 3.713f, 3.598f, 3.886f, 3.724f};
        const float db1[K_MAX] = {0.543f, 0.578f, 0.505f, 0.486f, 0.494f, 0.463f, 0.454f};
        const float db2[K_MAX] = {-0.200f, -0.435f, -0.334f, -0.401f, -0.358f, -0.379f, -0.275f};
        for (int i = 0; i < K_MAX; ++i) {
            gamma[i] = dg[i];
            beta0[i] = db0[i];
            beta1[i] = db1[i];
            beta2[i] = db2[i];
        }
    }
};

struct LayerWeights {
    Q4Mat q, k, v, o;
    Q4Mat gate, up, down;
    Q4Mat aconv_kp, mconv_kp;
    const float* in_norm = nullptr;      // [HID]
    const float* post_norm = nullptr;    // [HID]
    const float* q_norm = nullptr;       // [HD]
    const float* k_norm = nullptr;       // [HD]
    const float* aconv_base = nullptr;   // [2][2][HID]
    const float* mconv_base = nullptr;   // [2][2][HID]
};

struct DraftResult {
    int tok[K_MAX];
    float lp[K_MAX];
    int K = K_MAX;
};

// ------------------------------------------------------------------------------------------- Utility Kernels
__device__ __forceinline__ float dfl_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float dfl_silu(float x) { return x / (1.0f + __expf(-x)); }

// Extract 4-stream HC mean from target hyper [M, 4, 2560] BF16 -> FP16 taps [M, NT, HID] at slot `tap_idx`.
// Matches incoai patches/qwen38-dflash2-capture.patch verbatim: unflatten(-1, (4, 2560)).mean(dim=-2).
// Grid: (HID / 256, M), Block: 256.
__global__ void __launch_bounds__(256) k_tap_hc_mean(const __nv_bfloat16* __restrict__ hyper,
                                                     __half* __restrict__ taps_out,
                                                     int tap_idx) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const __nv_bfloat16* h = hyper + static_cast<size_t>(m) * 4 * HID + c;
    const float s = (__bfloat162float(h[0]) + __bfloat162float(h[HID]) +
                     __bfloat162float(h[2 * HID]) + __bfloat162float(h[3 * HID])) * 0.25f;
    taps_out[(static_cast<size_t>(m) * NT + tap_idx) * HID + c] = __float2half_rn(s);
}

// Prefill tap extraction into [WIN, NT, HID] ring window [lo, hi) for rows whose pos in [lo, hi).
__global__ void __launch_bounds__(256) k_pf_tap_hc_mean(const __nv_bfloat16* __restrict__ hyper,
                                                        const int* __restrict__ d_pos,
                                                        int lo, int hi,
                                                        __half* __restrict__ pf_feat,
                                                        int tap_idx) {
    const int m = blockIdx.y;
    const int p = d_pos[m];
    if (p < lo || p >= hi) return;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const __nv_bfloat16* h = hyper + static_cast<size_t>(m) * 4 * HID + c;
    const float s = (__bfloat162float(h[0]) + __bfloat162float(h[HID]) +
                     __bfloat162float(h[2 * HID]) + __bfloat162float(h[3 * HID])) * 0.25f;
    pf_feat[(static_cast<size_t>(p - lo) * NT + tap_idx) * HID + c] = __float2half_rn(s);
}

// ------------------------------------------------------------------------------------------- Fast __dp4a Q8_1 x Q4_0 GEMV (M <= 8)
// Quantizes tiny activation A[M, K] (<= 8 x 12800) to per-32-block int8 + fp32 scale + int32 sum (Q8_1),
// then executes hardware __dp4a (4 INT8xINT8 MACs/instruction, 2 instructions per 8 weights!) over
// 128-bit coalesced SoA Q4_0 weights qs[O, K/2] + sc[O, K/32]. Zero register spills, 5-8x faster than scalar FP32!
struct Q8Act {
    int8_t* q = nullptr;   // [max_M * max_K]
    float* sc = nullptr;   // [max_M * (max_K / 32)]
    int* isum = nullptr;   // [max_M * (max_K / 32)] sum of 32 int8 activations (for exact -8*sum offset!)
};

__device__ __forceinline__ void dfl_quant_warp_q8(float v, size_t blk, int lane,
                                                  int8_t* __restrict__ q_out,
                                                  float* __restrict__ sc_out,
                                                  int* __restrict__ sum_out) {
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = (amax > 1e-12f) ? (amax / 127.0f) : 0.0f;
    const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    const int qi = (sc > 0.0f) ? max(-127, min(127, __float2int_rn(v * inv))) : 0;

    q_out[blk * 32 + lane] = static_cast<int8_t>(qi);

    int hs = qi;
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) hs += __shfl_xor_sync(0xffffffffu, hs, o);
    if (lane == 0) {
        sc_out[blk] = sc;
        sum_out[blk * 2] = hs << 3; // pre-multiplied by 8 for zero-mul GEMV offset!
    } else if (lane == 16) {
        sum_out[blk * 2 + 1] = hs << 3;
    }
}

template <bool IN_F16>
__global__ void __launch_bounds__(256) k_quant_act_q8(const void* __restrict__ A,
                                                      int8_t* __restrict__ q_out,
                                                      float* __restrict__ sc_out,
                                                      int* __restrict__ sum_out,
                                                      int M, int K) {
    const int nb_row = K >> 5;
    const int total = M * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + warp;
    if (blk >= total) return;

    const size_t base = static_cast<size_t>(blk) * 32 + lane;
    float v;
    if constexpr (IN_F16) v = __half2float(static_cast<const __half*>(A)[base]);
    else v = static_cast<const float*>(A)[base];

    dfl_quant_warp_q8(v, static_cast<size_t>(blk), lane, q_out, sc_out, sum_out);
}

// Convert ggml Q4_0 [total_blocks x 18B] -> coalesced qs [total_blocks x 16B] + sc [total_blocks] FP16.
__global__ void __launch_bounds__(256) k_repack_q4_soa(const uint8_t* __restrict__ src,
                                                       uint8_t* __restrict__ dst_qs,
                                                       __half* __restrict__ dst_sc,
                                                       size_t total_blocks) {
    const size_t b = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (b >= total_blocks) return;
    const uint8_t* s = src + b * 18;
    const uint16_t db = static_cast<uint16_t>(s[0]) | (static_cast<uint16_t>(s[1]) << 8);
    dst_sc[b] = __ushort_as_half(db);
    uint8_t* d = dst_qs + b * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) d[i] = s[2 + i];
}

// Ultra-fast 4-warp, 2-lanes-per-Q4_0-block __dp4a GEMV (zero SMEM, zero barriers, zero integer divisions!):
//   - Even lanes process lo 16 nibbles (0..15), odd lanes process hi 16 nibbles (16..31).
//   - Wqs: 16 consecutive blocks = 256B coalesced across warp.
//   - Aq[m]: 32 consecutive int4s = 512B 100% coalesced across all 32 lanes!
//   - Half the register pressure -> 100% SM occupancy on GB10!
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_q4_dp4a_gemv(const int8_t* __restrict__ Aq,
                                                      const float* __restrict__ Asc,
                                                      const int* __restrict__ Asum,
                                                      const uint8_t* __restrict__ Wqs,
                                                      const __half* __restrict__ Wsc,
                                                      float* __restrict__ C,
                                                      int M, int O, int K) {
    const int warp = threadIdx.x >> 5;   // 0..3 -> 4 output rows per CTA
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= O) return;

    const int nb = K >> 5;
    const int bi0 = lane >> 1;           // 0..15
    const int hi = lane & 1;             // 0 (lo 0..15) or 1 (hi 16..31)
    const int shift = hi << 2;
    const uint4* qrow = reinterpret_cast<const uint4*>(Wqs + static_cast<size_t>(o) * (K >> 1));
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const size_t a_stride_i4 = static_cast<size_t>(K) >> 4;
    const size_t sum_stride = static_cast<size_t>(nb) << 1;

    for (int b = bi0; b < nb; b += 16) {
        const uint4 qv = __ldcs(qrow + b);
        const float d = __half2float(__ldcs(srow + b));
        const int w0 = static_cast<int>((qv.x >> shift) & 0x0F0F0F0Fu);
        const int w1 = static_cast<int>((qv.y >> shift) & 0x0F0F0F0Fu);
        const int w2 = static_cast<int>((qv.z >> shift) & 0x0F0F0F0Fu);
        const int w3 = static_cast<int>((qv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
                int dot = 0;
                dot = __dp4a(w0, a.x, dot);
                dot = __dp4a(w1, a.y, dot);
                dot = __dp4a(w2, a.z, dot);
                dot = __dp4a(w3, a.w, dot);
                const float val = static_cast<float>(dot - __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx));
                acc[m] = fmaf(val, d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = dfl_warp_sum(acc[m]);
            if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
        }
    }
}

// Fused Q + K + V projection for B = 8 (1 launch instead of 3, shared L2 residency of Aq!).
__global__ void __launch_bounds__(128) k_q4_dp4a_qkv8(const int8_t* __restrict__ Aq,
                                                      const float* __restrict__ Asc,
                                                      const int* __restrict__ Asum,
                                                      const uint8_t* __restrict__ Qqs, const __half* __restrict__ Qsc,
                                                      const uint8_t* __restrict__ Kqs, const __half* __restrict__ Ksc,
                                                      const uint8_t* __restrict__ Vqs, const __half* __restrict__ Vsc,
                                                      float* __restrict__ Qout, float* __restrict__ Kout, float* __restrict__ Vout) {
    constexpr int OQ = NH * HD;   // 4096
    constexpr int OK = NKV * HD;  // 1024
    constexpr int K = HID;        // 2560
    constexpr int nb = K >> 5;    // 80
    constexpr size_t a_stride_i4 = K >> 4;
    constexpr size_t sum_stride = nb << 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int go = blockIdx.x * 4 + warp;
    if (go >= OQ + OK + OK) return;

    const uint8_t* Wqs;
    const __half* Wsc;
    float* C;
    int O, o;
    if (go < OQ) { Wqs = Qqs; Wsc = Qsc; C = Qout; O = OQ; o = go; }
    else if (go < OQ + OK) { Wqs = Kqs; Wsc = Ksc; C = Kout; O = OK; o = go - OQ; }
    else { Wqs = Vqs; Wsc = Vsc; C = Vout; O = OK; o = go - OQ - OK; }

    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;
    const uint4* qrow = reinterpret_cast<const uint4*>(Wqs + static_cast<size_t>(o) * (K >> 1));
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int b = bi0; b < nb; b += 16) {
        const uint4 qv = __ldcs(qrow + b);
        const float d = __half2float(__ldcs(srow + b));
        const int w0 = static_cast<int>((qv.x >> shift) & 0x0F0F0F0Fu);
        const int w1 = static_cast<int>((qv.y >> shift) & 0x0F0F0F0Fu);
        const int w2 = static_cast<int>((qv.z >> shift) & 0x0F0F0F0Fu);
        const int w3 = static_cast<int>((qv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            int dot = 0;
            dot = __dp4a(w0, a.x, dot);
            dot = __dp4a(w1, a.y, dot);
            dot = __dp4a(w2, a.z, dot);
            dot = __dp4a(w3, a.w, dot);
            const float val = static_cast<float>(dot - __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx));
            acc[m] = fmaf(val, d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const float tot = dfl_warp_sum(acc[m]);
        if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
    }
}

// Fused K + V projection for context ingest (M <= 8, 1 launch instead of 2 per layer!).
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_q4_dp4a_kv(const int8_t* __restrict__ Aq,
                                                    const float* __restrict__ Asc,
                                                    const int* __restrict__ Asum,
                                                    const uint8_t* __restrict__ Kqs, const __half* __restrict__ Ksc,
                                                    const uint8_t* __restrict__ Vqs, const __half* __restrict__ Vsc,
                                                    float* __restrict__ Kout, float* __restrict__ Vout,
                                                    int M) {
    constexpr int OK = NKV * HD; // 1024
    constexpr int K = HID;       // 2560
    constexpr int nb = K >> 5;   // 80
    constexpr size_t a_stride_i4 = K >> 4;
    constexpr size_t sum_stride = nb << 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int go = blockIdx.x * 4 + warp;
    if (go >= 2 * OK) return;

    const bool is_v = (go >= OK);
    const int o = is_v ? (go - OK) : go;
    const uint4* qrow = reinterpret_cast<const uint4*>((is_v ? Vqs : Kqs) + static_cast<size_t>(o) * (K >> 1));
    const __half* srow = (is_v ? Vsc : Ksc) + static_cast<size_t>(o) * nb;
    float* C = is_v ? Vout : Kout;

    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    #pragma unroll
    for (int b = bi0; b < nb; b += 16) {
        const uint4 qv = __ldcs(qrow + b);
        const float d = __half2float(__ldcs(srow + b));
        const int w0 = static_cast<int>((qv.x >> shift) & 0x0F0F0F0Fu);
        const int w1 = static_cast<int>((qv.y >> shift) & 0x0F0F0F0Fu);
        const int w2 = static_cast<int>((qv.z >> shift) & 0x0F0F0F0Fu);
        const int w3 = static_cast<int>((qv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
                int dot = 0;
                dot = __dp4a(w0, a.x, dot);
                dot = __dp4a(w1, a.y, dot);
                dot = __dp4a(w2, a.z, dot);
                dot = __dp4a(w3, a.w, dot);
                const float val = static_cast<float>(dot - __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx));
                acc[m] = fmaf(val, d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = dfl_warp_sum(acc[m]);
            if (lane == 0) C[static_cast<size_t>(m) * OK + o] = tot;
        }
    }
}

// Fused Gate + Up + SwiGLU for B = 8: reads Aq ONCE for both gate & up, zero gate/up DRAM buffers!
__global__ void __launch_bounds__(128) k_q4_dp4a_gate_up_swiglu8(const int8_t* __restrict__ Aq,
                                                                 const float* __restrict__ Asc,
                                                                 const int* __restrict__ Asum,
                                                                 const uint8_t* __restrict__ Gqs, const __half* __restrict__ Gsc,
                                                                 const uint8_t* __restrict__ Uqs, const __half* __restrict__ Usc,
                                                                 float* __restrict__ act_out) {
    constexpr int K = HID;     // 2560
    constexpr int nb = K >> 5; // 80
    constexpr size_t a_stride_i4 = K >> 4;
    constexpr size_t sum_stride = nb << 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= INTER) return;

    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;
    const uint4* gq = reinterpret_cast<const uint4*>(Gqs + static_cast<size_t>(o) * (K >> 1));
    const __half* gs = Gsc + static_cast<size_t>(o) * nb;
    const uint4* uq = reinterpret_cast<const uint4*>(Uqs + static_cast<size_t>(o) * (K >> 1));
    const __half* us = Usc + static_cast<size_t>(o) * nb;

    float ag[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float au[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (int b = bi0; b < nb; b += 16) {
        const uint4 qgv = __ldcs(gq + b);
        const float dg = __half2float(__ldcs(gs + b));
        const uint4 quv = __ldcs(uq + b);
        const float du = __half2float(__ldcs(us + b));
        const int wg0 = static_cast<int>((qgv.x >> shift) & 0x0F0F0F0Fu);
        const int wg1 = static_cast<int>((qgv.y >> shift) & 0x0F0F0F0Fu);
        const int wg2 = static_cast<int>((qgv.z >> shift) & 0x0F0F0F0Fu);
        const int wg3 = static_cast<int>((qgv.w >> shift) & 0x0F0F0F0Fu);
        const int wu0 = static_cast<int>((quv.x >> shift) & 0x0F0F0F0Fu);
        const int wu1 = static_cast<int>((quv.y >> shift) & 0x0F0F0F0Fu);
        const int wu2 = static_cast<int>((quv.z >> shift) & 0x0F0F0F0Fu);
        const int wu3 = static_cast<int>((quv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            const int s8 = __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx);
            const float asc = __ldg(Asc + static_cast<size_t>(m) * nb + b);
            int dg_dot = 0, du_dot = 0;
            dg_dot = __dp4a(wg0, a.x, dg_dot); du_dot = __dp4a(wu0, a.x, du_dot);
            dg_dot = __dp4a(wg1, a.y, dg_dot); du_dot = __dp4a(wu1, a.y, du_dot);
            dg_dot = __dp4a(wg2, a.z, dg_dot); du_dot = __dp4a(wu2, a.z, du_dot);
            dg_dot = __dp4a(wg3, a.w, dg_dot); du_dot = __dp4a(wu3, a.w, du_dot);
            ag[m] = fmaf(static_cast<float>(dg_dot - s8), dg * asc, ag[m]);
            au[m] = fmaf(static_cast<float>(du_dot - s8), du * asc, au[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const float tg = dfl_warp_sum(ag[m]);
        const float tu = dfl_warp_sum(au[m]);
        if (lane == 0) act_out[static_cast<size_t>(m) * INTER + o] = dfl_silu(tg) * tu;
    }
}

inline void quant_act_q8(const float* A, Q8Act& qa, int M, int K, cudaStream_t st) {
    const int total = M * (K >> 5);
    k_quant_act_q8<false><<<(total + 7) / 8, 256, 0, st>>>(A, qa.q, qa.sc, qa.isum, M, K);
}

inline void quant_act_q8_f16(const __half* A, Q8Act& qa, int M, int K, cudaStream_t st) {
    const int total = M * (K >> 5);
    k_quant_act_q8<true><<<(total + 7) / 8, 256, 0, st>>>(A, qa.q, qa.sc, qa.isum, M, K);
}

inline void launch_q4_gemv_q8(const Q8Act& qa, const Q4Mat& W, float* C, int M, cudaStream_t st) {
    const int g = (W.rows + 3) / 4;
    if (M == 8) k_q4_dp4a_gemv<8, true><<<g, 128, 0, st>>>(qa.q, qa.sc, qa.isum, W.qs, W.sc, C, 8, W.rows, W.cols);
    else if (M == 7) k_q4_dp4a_gemv<7, true><<<g, 128, 0, st>>>(qa.q, qa.sc, qa.isum, W.qs, W.sc, C, 7, W.rows, W.cols);
    else if (M == 1) k_q4_dp4a_gemv<1, true><<<g, 128, 0, st>>>(qa.q, qa.sc, qa.isum, W.qs, W.sc, C, 1, W.rows, W.cols);
    else if (M <= 4) k_q4_dp4a_gemv<4, false><<<g, 128, 0, st>>>(qa.q, qa.sc, qa.isum, W.qs, W.sc, C, M, W.rows, W.cols);
    else k_q4_dp4a_gemv<8, false><<<g, 128, 0, st>>>(qa.q, qa.sc, qa.isum, W.qs, W.sc, C, M, W.rows, W.cols);
}

// ------------------------------------------------------------------------------------------- Layer Building Blocks
// Fused RMSNorm + Q8_1 quantization (optional F32 and/or BF16 output). Grid: M, Block: 256 (8 warps).
template <bool WRITE_F32, bool WRITE_BF16>
__global__ void __launch_bounds__(256) k_df2_rmsnorm_q8(const float* __restrict__ x,
                                                        const float* __restrict__ w,
                                                        float* __restrict__ f32_out,
                                                        __nv_bfloat16* __restrict__ bf16_out,
                                                        int8_t* __restrict__ q_out,
                                                        float* __restrict__ sc_out,
                                                        int* __restrict__ sum_out,
                                                        float eps) {
    const int m = blockIdx.x, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const float* xr = x + static_cast<size_t>(m) * HID;
    float ss = 0.0f;
    for (int c = t; c < HID; c += 256) {
        const float v = xr[c];
        ss = fmaf(v, v, ss);
    }
    ss = dfl_warp_sum(ss);
    __shared__ float sw[8];
    if (lane == 0) sw[warp] = ss;
    __syncthreads();
    if (t < 32) {
        float tot = (t < 8) ? sw[t] : 0.0f;
        tot = dfl_warp_sum(tot);
        if (t == 0) sw[0] = rsqrtf(tot / static_cast<float>(HID) + eps);
    }
    __syncthreads();
    const float inv = sw[0];
    constexpr int NB = HID >> 5; // 80 blocks of 32
    for (int b = warp; b < NB; b += 8) {
        const int c = (b << 5) | lane;
        const float val = xr[c] * inv * w[c];
        if constexpr (WRITE_F32) f32_out[static_cast<size_t>(m) * HID + c] = val;
        if constexpr (WRITE_BF16) bf16_out[static_cast<size_t>(m) * HID + c] = __float2bfloat16(val);
        dfl_quant_warp_q8(val, static_cast<size_t>(m) * NB + b, lane, q_out, sc_out, sum_out);
    }
}

// Fused side-0 prepare conv + Q8_1 quantization directly in registers (zero xc DRAM traffic!).
// Grid: (HID / 256 = 10, B), Block: 256 (8 warps = 8 contiguous Q8_1 blocks of 32).
__global__ void __launch_bounds__(256) k_df2_conv0_q8(const float* __restrict__ x,
                                                      const float* __restrict__ dyn,
                                                      const float* __restrict__ base,
                                                      int8_t* __restrict__ q_out,
                                                      float* __restrict__ sc_out,
                                                      int* __restrict__ sum_out) {
    const int m = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int g = c / GS;
    const float* drow = dyn + static_cast<size_t>(m) * 4 * NG;
    const float k0 = base[c] + drow[g];
    float y = k0 * x[static_cast<size_t>(m) * HID + c];
    if (m > 0) {
        const float k1 = base[HID + c] + drow[NG + g];
        y = fmaf(k1, x[static_cast<size_t>(m - 1) * HID + c], y);
    }
    const size_t blk = static_cast<size_t>(m) * (HID >> 5) + blockIdx.x * 8 + warp;
    dfl_quant_warp_q8(y, blk, lane, q_out, sc_out, sum_out);
}

// Embed B = 8 block rows: row 0 = d_meta[0] (anchor_tok), rows 1..7 = MASK_ID. Grid: B, Block: 256.
__global__ void __launch_bounds__(256) k_df2_embed(const __nv_bfloat16* __restrict__ emb,
                                                   const int* __restrict__ d_meta,
                                                   float* __restrict__ out) {
    const int m = blockIdx.x;
    const int tok = (m == 0) ? d_meta[0] : MASK_ID;
    const __nv_bfloat16* src = emb + static_cast<size_t>(tok) * HID;
    float* dst = out + static_cast<size_t>(m) * HID;
    for (int c = threadIdx.x; c < HID; c += 256) dst[c] = __bfloat162float(src[c]);
}

// Dynamic 2-tap grouped conv (GS = 16, ks = 2) over B = 8 rows.
// dyn [B, 2 sides, 2 taps, 160 groups], base [2 sides, 2 taps, HID].
// side = 0 (prepare): out = conv(x); side = 1 (finish): resid += conv(x).
// Grid: (HID / 256, B), Block: 256.
template <bool ADD_RESID>
__global__ void __launch_bounds__(256) k_df2_conv(const float* __restrict__ x,
                                                  const float* __restrict__ dyn,
                                                  const float* __restrict__ base,
                                                  int side,
                                                  float* __restrict__ dst) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int g = c / GS;
    const float* drow = dyn + (static_cast<size_t>(m) * 4 + side * 2) * NG;
    const float* brow = base + static_cast<size_t>(side * 2) * HID;

    const float k0 = brow[c] + drow[g];
    float y = k0 * x[static_cast<size_t>(m) * HID + c];
    if (m > 0) {
        const float k1 = brow[HID + c] + drow[NG + g];
        y = fmaf(k1, x[static_cast<size_t>(m - 1) * HID + c], y);
    }
    const size_t idx = static_cast<size_t>(m) * HID + c;
    if constexpr (ADD_RESID) dst[idx] += y;
    else dst[idx] = y;
}

// SwiGLU activation: out[m, c] = silu(gate[m, c]) * up[m, c]. Grid: (INTER / 256, B), Block: 256.
__global__ void __launch_bounds__(256) k_df2_swiglu(const float* __restrict__ gate,
                                                    const float* __restrict__ up,
                                                    float* __restrict__ out) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const size_t idx = static_cast<size_t>(m) * INTER + c;
    out[idx] = dfl_silu(gate[idx]) * up[idx];
}

// Fast FP32 Head RMSNorm + NeoX RoPE (theta = 10000; avoids 1/64-rate SM121 FP64 transcendentals!).
__device__ __forceinline__ void df2_rope_cs(int pos, int d, float& c, float& s) {
    const float inv = __expf(-static_cast<float>(d & 63) * (2.0f / static_cast<float>(HD) * LOG_ROPE_THETA));
    __sincosf(static_cast<float>(pos) * inv, &s, &c);
}

// Write context K (norm + RoPE) and V into FP16 ring at positions pos0..pos0+M-1. Grid: (M * NKV), Block: 128.
__global__ void __launch_bounds__(128) k_df2_ctx_kv_write(const float* __restrict__ k_in,
                                                          const float* __restrict__ v_in,
                                                          const float* __restrict__ k_norm_w,
                                                          __half* __restrict__ kring,
                                                          __half* __restrict__ vring,
                                                          int pos0, float eps) {
    const int idx = blockIdx.x;
    const int m = idx / NKV;
    const int h = idx % NKV;
    const int d = threadIdx.x; // 0..127
    const float* kr = k_in + (static_cast<size_t>(m) * NKV + h) * HD;
    const float* vr = v_in + (static_cast<size_t>(m) * NKV + h) * HD;

    const float kv = kr[d];
    float ss = dfl_warp_sum(kv * kv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float kn = kv * inv * k_norm_w[d];
    rot[d] = kn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(pos0 + m, d, cs, sn);
    const float o = rot[d ^ 64];
    const float krot = (d < 64) ? (kn * cs - o * sn) : (kn * cs + o * sn);

    const int slot_pos = (pos0 + m) & (RING - 1);
    const size_t roff = (static_cast<size_t>(h) * RING + slot_pos) * HD + d;
    kring[roff] = __float2half_rn(krot);
    vring[roff] = __float2half_rn(vr[d]);
}

// Q norm + RoPE for B = 8 block rows at q0..q0+7 (q0 = d_meta[1]). Grid: (B * NH), Block: 128.
__global__ void __launch_bounds__(128) k_df2_q_norm_rope(const float* __restrict__ q_in,
                                                         const float* __restrict__ q_norm_w,
                                                         float* __restrict__ q_out,
                                                         const int* __restrict__ d_meta, float eps) {
    const int q0 = d_meta[1];
    const int idx = blockIdx.x;
    const int m = idx / NH;
    const int h = idx % NH;
    const int d = threadIdx.x;
    const float* qr = q_in + (static_cast<size_t>(m) * NH + h) * HD;

    const float qv = qr[d];
    float ss = dfl_warp_sum(qv * qv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float qn = qv * inv * q_norm_w[d];
    rot[d] = qn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(q0 + m, d, cs, sn);
    const float o = rot[d ^ 64];
    q_out[(static_cast<size_t>(m) * NH + h) * HD + d] = (d < 64) ? (qn * cs - o * sn) : (qn * cs + o * sn);
}

// Block K norm + RoPE into isolated scratch kblk [NKV][B][HD] and vblk [NKV][B][HD] (never mutates committed ring!).
// Grid: (B * NKV), Block: 128.
__global__ void __launch_bounds__(128) k_df2_blk_kv(const float* __restrict__ k_in,
                                                    const float* __restrict__ v_in,
                                                    const float* __restrict__ k_norm_w,
                                                    __half* __restrict__ kblk,
                                                    __half* __restrict__ vblk,
                                                    const int* __restrict__ d_meta, float eps) {
    const int q0 = d_meta[1];
    const int idx = blockIdx.x;
    const int m = idx / NKV;
    const int h = idx % NKV;
    const int d = threadIdx.x;
    const float* kr = k_in + (static_cast<size_t>(m) * NKV + h) * HD;
    const float* vr = v_in + (static_cast<size_t>(m) * NKV + h) * HD;

    const float kv = kr[d];
    float ss = dfl_warp_sum(kv * kv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float kn = kv * inv * k_norm_w[d];
    rot[d] = kn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(q0 + m, d, cs, sn);
    const float o = rot[d ^ 64];
    const float krot = (d < 64) ? (kn * cs - o * sn) : (kn * cs + o * sn);

    const size_t boff = (static_cast<size_t>(h) * B + m) * HD + d;
    kblk[boff] = __float2half_rn(krot);
    vblk[boff] = __float2half_rn(vr[d]);
}

// Split-KV bidirectional B = 8 attention over FP16 context ring [ctx_lo, q0) + isolated block scratch [0..7].
// Matches dflash2_ref.py:142 visibility: query m at q0+m sees ctx pos p iff (q0 + m - p) < WIN, and ALL 8 block rows!
// Pass 1: Grid (NH, ATTN_SPLIT), Block 128 (d = 0..127). Writes apart [NH][ATTN_SPLIT][B][HD + 2] (o[0..127], m, l).
__global__ void __launch_bounds__(128) k_df2_attn_split(const float* __restrict__ qf,
                                                        const __half* __restrict__ kring,
                                                        const __half* __restrict__ vring,
                                                        const __half* __restrict__ kblk,
                                                        const __half* __restrict__ vblk,
                                                        const int* __restrict__ d_meta,
                                                        float* __restrict__ apart) {
    const int q0 = d_meta[1];
    const int ctx_lo = d_meta[2];
    const int h = blockIdx.x;
    const int sp = blockIdx.y;
    const int d = threadIdx.x;
    const int kh = h / GQA;
    constexpr float scale = 0.08838834764831845f; // 1 / sqrt(128)

    float qv[B];
    #pragma unroll
    for (int m = 0; m < B; ++m) qv[m] = qf[(static_cast<size_t>(m) * NH + h) * HD + d];

    float rm[B], rl[B], ro[B];
    #pragma unroll
    for (int m = 0; m < B; ++m) { rm[m] = -INFINITY; rl[m] = 0.0f; ro[m] = 0.0f; }

    __shared__ float sw[B][4];
    auto step_key = [&](const __half* kp, const __half* vp, int kpos, bool is_blk) {
        const float kf = __half2float(kp[d]);
        const float vf = __half2float(vp[d]);
        #pragma unroll
        for (int m = 0; m < B; ++m) {
            const float dot = dfl_warp_sum(qv[m] * kf);
            if ((d & 31) == 0) sw[m][d >> 5] = dot;
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < B; ++m) {
            if (!is_blk && (q0 + m - kpos) >= WIN) continue;
            const float sc = (sw[m][0] + sw[m][1] + sw[m][2] + sw[m][3]) * scale;
            const float nm = fmaxf(rm[m], sc);
            const float a = (rm[m] == -INFINITY) ? 0.0f : __expf(rm[m] - nm);
            const float b = __expf(sc - nm);
            ro[m] = ro[m] * a + vf * b;
            rl[m] = rl[m] * a + b;
            rm[m] = nm;
        }
        __syncthreads();
    };

    const int lo = max(ctx_lo, q0 - WIN + 1);
    const int nctx = max(0, q0 - lo);
    const int c_per = (nctx + ATTN_SPLIT - 1) / ATTN_SPLIT;
    const int p0 = lo + sp * c_per;
    const int p1 = min(q0, p0 + c_per);
    const __half* kh_ring = kring + static_cast<size_t>(kh) * RING * HD;
    const __half* vh_ring = vring + static_cast<size_t>(kh) * RING * HD;

    for (int p = p0; p < p1; ++p) {
        const int r = p & (RING - 1);
        step_key(kh_ring + static_cast<size_t>(r) * HD, vh_ring + static_cast<size_t>(r) * HD, p, false);
    }
    if (sp == 0) {
        const __half* kh_blk = kblk + static_cast<size_t>(kh) * B * HD;
        const __half* vh_blk = vblk + static_cast<size_t>(kh) * B * HD;
        for (int j = 0; j < B; ++j) {
            step_key(kh_blk + static_cast<size_t>(j) * HD, vh_blk + static_cast<size_t>(j) * HD, q0 + j, true);
        }
    }

    float* dst = apart + ((static_cast<size_t>(h) * ATTN_SPLIT + sp) * B) * (HD + 2);
    #pragma unroll
    for (int m = 0; m < B; ++m) {
        float* row = dst + m * (HD + 2);
        row[d] = ro[m];
        if (d == 0) { row[HD] = rm[m]; row[HD + 1] = rl[m]; }
    }
}

// Pass 2: merge ATTN_SPLIT partials -> ao [B, NH * HD]. Grid: (B * NH), Block: 128.
__global__ void __launch_bounds__(128) k_df2_attn_merge(const float* __restrict__ apart, float* __restrict__ ao) {
    const int idx = blockIdx.x;
    const int m = idx / NH;
    const int h = idx % NH;
    const int d = threadIdx.x;

    float gm = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
        const float* row = apart + (((static_cast<size_t>(h) * ATTN_SPLIT + sp) * B + m) * (HD + 2));
        gm = fmaxf(gm, row[HD]);
    }
    float gl = 0.0f, go = 0.0f;
    for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
        const float* row = apart + (((static_cast<size_t>(h) * ATTN_SPLIT + sp) * B + m) * (HD + 2));
        const float sm = row[HD];
        if (sm > -INFINITY) {
            const float w = __expf(sm - gm);
            gl += row[HD + 1] * w;
            go += row[d] * w;
        }
    }
    ao[(static_cast<size_t>(m) * NH + h) * HD + d] = (gl > 0.0f) ? (go / gl) : 0.0f;
}

// Optional Stage-A rank-64 slot-1 residual adapter on row 1 (offset +1): h[1] += (silu(h[1] @ W_down^T)) @ W_up^T.
__global__ void __launch_bounds__(256) k_df2_slot1_adapter(float* __restrict__ h_row1,
                                                           const float* __restrict__ w_down, // [R, HID]
                                                           const float* __restrict__ w_up,   // [HID, R]
                                                           int R) {
    __shared__ float mid[128];
    const int t = threadIdx.x;
    for (int r = t; r < R; r += 256) {
        float acc = 0.0f;
        const float* wr = w_down + static_cast<size_t>(r) * HID;
        for (int c = 0; c < HID; ++c) acc = fmaf(h_row1[c], wr[c], acc);
        mid[r] = dfl_silu(acc);
    }
    __syncthreads();
    for (int c = t; c < HID; c += 256) {
        float acc = 0.0f;
        const float* wu = w_up + static_cast<size_t>(c) * R;
        for (int r = 0; r < R; ++r) acc = fmaf(mid[r], wu[r], acc);
        h_row1[c] += acc;
    }
}

// ------------------------------------------------------------------------------------------- Global Top-16 Merge + Calibrated Expected-Reach Viterbi DP
// Single CTA (256 threads = 8 warps):
//   1. Merges nr ranks' sorted local top-16 (Cand16 [nr][K][16]) deterministically -> global s_l[K][16], s_i[K][16].
//   2. If use_dp == 0: exact stock DFlash2 1-step greedy walk.
//   3. If use_dp == 1: exact backward Expected-Reach Viterbi DP over the K x 16 candidate lattice with
//      calibrated OOV ("none-of-these") residual mass z_OOV(s) = beta0 + beta1*max_l + beta2*(l1 - l2).
__device__ __forceinline__ bool cand_better(float l0, int i0, float l1, int i1) {
    return (l0 > l1) || (l0 == l1 && i0 < i1);
}

__global__ void __launch_bounds__(256) k_df2_walk_dp(const Cand16* __restrict__ rank_cands, // [nr][K_MAX][16]
                                                     int nr, int K, int anchor_tok,
                                                     const float* __restrict__ hp,          // [K_MAX][SELR]
                                                     const __nv_bfloat16* __restrict__ pred,// [VOCAB][SELR]
                                                     const __nv_bfloat16* __restrict__ succ,// [VOCAB][SELR]
                                                     DpCal cal, float sel_alpha, int use_dp,
                                                     float tv_base_ms, float tv_step_ms,
                                                     int* __restrict__ tok_out,             // [K_MAX + 1] (tok_out[K_MAX] = K_eff)
                                                     float* __restrict__ lp_out) {          // [K_MAX]
    __shared__ float s_l[K_MAX][TOPK];
    __shared__ int s_i[K_MAX][TOPK];
    // Transition probabilities p_trans[s][p][c] for s=1..K-1, and p0[c] for s=0
    __shared__ float s_p0[TOPK];
    __shared__ float s_p[K_MAX][TOPK][TOPK];
    __shared__ float s_V[2][TOPK];
    __shared__ int s_pi[K_MAX][TOPK];
    // SMEM-cached codebook rows for current offset (16 KB total -> 16x fewer random DRAM misses!)
    __shared__ __nv_bfloat16 s_pred[TOPK][SELR];
    __shared__ __nv_bfloat16 s_succ[TOPK][SELR];
    __shared__ float s_hp[SELR];

    const int t = threadIdx.x, w = t >> 5, l = t & 31;

    // 1. Deterministic global top-16 merge (1 warp per offset s = 0..K-1)
    if (w < K && l == 0) {
        float* bl = s_l[w];
        int* bi = s_i[w];
        int n = 0;
        for (int p = 0; p < nr; ++p) {
            const Cand16* src = rank_cands + (static_cast<size_t>(p) * K_MAX + w) * TOPK;
            for (int k = 0; k < TOPK; ++k) {
                const Cand16 x = src[k];
                if (n == TOPK && !cand_better(x.logit, x.id, bl[TOPK - 1], bi[TOPK - 1])) break;
                int at = (n < TOPK) ? n++ : (TOPK - 1);
                while (at > 0 && cand_better(x.logit, x.id, bl[at - 1], bi[at - 1])) {
                    bl[at] = bl[at - 1]; bi[at] = bi[at - 1]; --at;
                }
                bl[at] = x.logit; bi[at] = x.id;
            }
        }
        for (int k = 0; k < TOPK; ++k) {
            if (static_cast<unsigned>(bi[k]) >= static_cast<unsigned>(geom::VOCAB)) {
                bi[k] = 0;
                bl[k] = -INFINITY;
            }
        }
    }
    __syncthreads();

    auto load_cb16 = [&](const __nv_bfloat16* table, const int* ids, __nv_bfloat16 dst[TOPK][SELR]) {
        // 16 rows x 256 bf16 = 512 uint4 (16B each) -> 2 coalesced 128-bit loads per thread!
        for (int idx = t; idx < TOPK * (SELR / 8); idx += 256) {
            const int row = idx >> 5;
            const int vec = idx & 31;
            const uint4* src = reinterpret_cast<const uint4*>(table + static_cast<size_t>(ids[row]) * SELR);
            reinterpret_cast<uint4*>(dst[row])[vec] = __ldg(src + vec);
        }
    };

    auto bilinear_dot_smem = [&](int p_slot, int c_slot) -> float {
        float acc = 0.0f;
        const __nv_bfloat16* pr = s_pred[p_slot];
        const __nv_bfloat16* sr = s_succ[c_slot];
        #pragma unroll
        for (int i = l; i < SELR; i += 32) {
            acc = fmaf(__bfloat162float(pr[i]) * s_hp[i], __bfloat162float(sr[i]), acc);
        }
        return dfl_warp_sum(acc);
    };

    if (use_dp == 0) {
        // Stock myopic greedy walk (matches dflash2_ref.py:175-182 verbatim)
        __shared__ float sc[TOPK];
        __shared__ int s_prev;
        if (t == 0) s_prev = anchor_tok;
        __syncthreads();
        for (int s = 0; s < K; ++s) {
            s_hp[t] = hp[static_cast<size_t>(s) * SELR + t];
            s_pred[0][t] = pred[static_cast<size_t>(s_prev) * SELR + t];
            load_cb16(succ, s_i[s], s_succ);
            __syncthreads();
            for (int cc = w; cc < TOPK; cc += 8) {
                const float dot = bilinear_dot_smem(0, cc);
                if (l == 0) sc[cc] = s_l[s][cc] + sel_alpha * dot;
            }
            __syncthreads();
            if (t == 0) {
                int bk = 0;
                float best = sc[0], mx = sc[0];
                for (int k = 1; k < TOPK; ++k) {
                    if (sc[k] > best) { best = sc[k]; bk = k; }
                    mx = fmaxf(mx, sc[k]);
                }
                float z = 0.0f;
                for (int k = 0; k < TOPK; ++k) z += expf(sc[k] - mx);
                s_prev = s_i[s][bk];
                tok_out[s] = s_i[s][bk];
                lp_out[s] = sc[bk] - mx - logf(z);
            }
            __syncthreads();
        }
        if (t == 0) tok_out[K_MAX] = K;
        return;
    }

    // 2. Calibrated Expected-Reach Viterbi DP over the K x 16 candidate lattice
    // Step 2a: offset s = 0 (from anchor_tok -> 16 candidates)
    {
        __shared__ float sc0[TOPK];
        s_hp[t] = hp[t];
        s_pred[0][t] = pred[static_cast<size_t>(anchor_tok) * SELR + t];
        load_cb16(succ, s_i[0], s_succ);
        __syncthreads();
        for (int cc = w; cc < TOPK; cc += 8) {
            const float dot = bilinear_dot_smem(0, cc);
            if (l == 0) sc0[cc] = s_l[0][cc] + sel_alpha * dot;
        }
        __syncthreads();
        if (t == 0) {
            float m1 = -INFINITY, m2 = -INFINITY;
            for (int k = 0; k < TOPK; ++k) {
                const float v = sc0[k];
                if (v > m1) { m2 = m1; m1 = v; }
                else if (v > m2) m2 = v;
            }
            const float g = cal.gamma[0];
            const float z_oov = cal.beta0[0] + cal.beta1[0] * m1 + cal.beta2[0] * fmaxf(0.0f, m1 - m2);
            const float mx = fmaxf(g * m1, z_oov);
            float z = expf(z_oov - mx);
            for (int k = 0; k < TOPK; ++k) z += expf(g * sc0[k] - mx);
            const float inv_z = 1.0f / z;
            for (int k = 0; k < TOPK; ++k) s_p0[k] = expf(g * sc0[k] - mx) * inv_z;
        }
        __syncthreads();
    }

    // Step 2b: offsets s = 1..K-1 (all 16 x 16 transitions in parallel from SMEM-cached codebooks!)
    for (int s = 1; s < K; ++s) {
        s_hp[t] = hp[static_cast<size_t>(s) * SELR + t];
        load_cb16(pred, s_i[s - 1], s_pred);
        load_cb16(succ, s_i[s], s_succ);
        __syncthreads();
        for (int pair = w; pair < TOPK * TOPK; pair += 8) {
            const int p = pair >> 4;
            const int c = pair & 15;
            const float dot = bilinear_dot_smem(p, c);
            if (l == 0) s_p[s][p][c] = s_l[s][c] + sel_alpha * dot;
        }
        __syncthreads();
        // Calibrate each predecessor row p = 0..15 into probabilities p_hat(c | p)
        if (t < TOPK) {
            const int p = t;
            float m1 = -INFINITY, m2 = -INFINITY;
            for (int c = 0; c < TOPK; ++c) {
                const float v = s_p[s][p][c];
                if (v > m1) { m2 = m1; m1 = v; }
                else if (v > m2) m2 = v;
            }
            const float g = cal.gamma[s];
            const float z_oov = cal.beta0[s] + cal.beta1[s] * m1 + cal.beta2[s] * fmaxf(0.0f, m1 - m2);
            const float mx = fmaxf(g * m1, z_oov);
            float z = expf(z_oov - mx);
            for (int c = 0; c < TOPK; ++c) z += expf(g * s_p[s][p][c] - mx);
            const float inv_z = 1.0f / z;
            for (int c = 0; c < TOPK; ++c) s_p[s][p][c] = expf(g * s_p[s][p][c] - mx) * inv_z;
        }
        __syncthreads();
    }

    // Step 2c: Backward Expected-Reach DP. If use_dp == 2, evaluate every truncation kk = 1..K
    // and pick K* = argmax_{kk in 1..K} E[tau(kk)] / (tv_base_ms + tv_step_ms * kk)!
    __shared__ float s_best_tps;
    __shared__ int s_best_k;
    if (t == 0) { s_best_tps = -1.0f; s_best_k = K; }
    __syncthreads();

    const int k_lo = (use_dp >= 2) ? 1 : K;
    for (int kk = k_lo; kk <= K; ++kk) {
        if (t < TOPK) s_V[0][t] = 0.0f;
        __syncthreads();
        int cur = 0;
        for (int s = kk - 1; s >= 1; --s) {
            const int nxt = cur ^ 1;
            if (t < TOPK) {
                const int p = t;
                float best = -1.0f;
                int bc = 0;
                for (int c = 0; c < TOPK; ++c) {
                    const float val = s_p[s][p][c] * (1.0f + s_V[cur][c]);
                    if (val > best) { best = val; bc = c; }
                }
                s_V[nxt][p] = best;
                s_pi[s][p] = bc;
            }
            __syncthreads();
            cur = nxt;
        }
        if (t == 0) {
            float best = -1.0f;
            int c = 0;
            for (int k = 0; k < TOPK; ++k) {
                const float val = s_p0[k] * (1.0f + s_V[cur][k]);
                if (val > best) { best = val; c = k; }
            }
            const float etps = (1.0f + best) / (tv_base_ms + tv_step_ms * static_cast<float>(kk));
            if (use_dp < 2 || etps > s_best_tps) {
                s_best_tps = etps;
                s_best_k = kk;
                tok_out[0] = s_i[0][c];
                lp_out[0] = logf(fmaxf(s_p0[c], 1e-30f));
                for (int s = 1; s < kk; ++s) {
                    const int nc = s_pi[s][c];
                    tok_out[s] = s_i[s][nc];
                    lp_out[s] = logf(fmaxf(s_p[s][c][nc], 1e-30f));
                    c = nc;
                }
            }
        }
        __syncthreads();
    }
    if (t == 0) tok_out[K_MAX] = s_best_k;
}

// Exact FP8/BF16 fallback top-16 extraction for GR8_QW38_TRUNK_HEAD2=0 / A/B testing on QSpine-DF2.
__global__ void __launch_bounds__(256) k_df2_sort_top16_from_y(const float* __restrict__ y, int ldy, int v0,
                                                               const int* __restrict__ cand16,
                                                               Cand16* __restrict__ out) {
    const int m = blockIdx.x, t = threadIdx.x;
    __shared__ float sv[16];
    __shared__ int si[16];
    if (t < 16) {
        const int idx = cand16[static_cast<size_t>(m) * 16 + t];
        sv[t] = y[static_cast<size_t>(m) * ldy + idx];
        si[t] = v0 + idx;
    }
    __syncthreads();
    if (t < 16) {
        const float li = sv[t];
        const int ii = si[t];
        int r = 0;
        for (int j = 0; j < 16; ++j) {
            if (sv[j] > li || (sv[j] == li && si[j] < ii)) ++r;
        }
        out[static_cast<size_t>(m) * 16 + r] = Cand16{li, ii};
    }
}

// ------------------------------------------------------------------------------------------- Host Drafter Class
struct Drafter {
    uint8_t* dev_blob = nullptr;
    uint8_t* dev_soa_blob = nullptr; // coalesced SoA Q4_0 weights (qs + sc)
    size_t blob_bytes = 0;

    Q4Mat fc, hproj;
    const float* hidden_norm = nullptr;
    const float* final_norm = nullptr;
    const __nv_bfloat16* pred_cb = nullptr;
    const __nv_bfloat16* succ_cb = nullptr;
    LayerWeights ly[L];

    // Optional Stage-A overlays (loaded automatically if present in DF2P pack)
    DpCal dp_cal;
    const float* slot1_down = nullptr;
    const float* slot1_up = nullptr;
    int slot1_rank = 0;

    // FP16 Context KV ring [L][NKV][RING][HD] (80.0 MiB) + prompt prefill tap staging [WIN][NT][HID] (50.0 MiB)
    __half* kring = nullptr;
    __half* vring = nullptr;
    __half* pf_feat = nullptr;
    __half* step_feat = nullptr;     // [B][NT][HID] committed decode taps

    // Scratch buffers
    float* ctx_fc = nullptr;         // [B][HID]
    float* ctx_norm = nullptr;       // [B][HID]
    float* ctx_k = nullptr;          // [B][NKV * HD]
    float* ctx_v = nullptr;          // [B][NKV * HD]
    float* h = nullptr;              // [B][HID]
    float* xn = nullptr;             // [B][HID]
    float* xc = nullptr;             // [B][HID]
    float* dyn = nullptr;            // [B][CPO]
    float* q_buf = nullptr;          // [B][NH * HD]
    float* qf = nullptr;             // [B][NH * HD]
    float* k_buf = nullptr;          // [B][NKV * HD]
    float* v_buf = nullptr;          // [B][NKV * HD]
    __half* kblk = nullptr;          // [NKV][B][HD]
    __half* vblk = nullptr;          // [NKV][B][HD]
    float* apart = nullptr;          // [NH][ATTN_SPLIT][B][HD + 2]
    float* ao = nullptr;             // [B][NH * HD]
    float* o_buf = nullptr;          // [B][HID]
    float* gate_buf = nullptr;       // [B][INTER]
    float* up_buf = nullptr;         // [B][INTER]
    float* act_buf = nullptr;        // [B][INTER]
    Q8Act qa;                        // Q8_1 quantized activation scratch [B * NT * HID]
    __nv_bfloat16* out_bf16 = nullptr; // [K_MAX][HID] final-normed draft rows 1..7 for lm_head/head2
    float* hp = nullptr;             // [K_MAX][SELR]
    Cand16* d_rank_cands = nullptr; // [4][K_MAX][TOPK]
    int* d_tok_out = nullptr;        // [K_MAX + 1]
    float* d_lp_out = nullptr;       // [K_MAX]
    int* d_blk_meta = nullptr;       // [3]: {anchor_tok, q0, ctx_lo} for CUDA-Graph forward_block
    int* h_tok_pinned = nullptr;
    float* h_lp_pinned = nullptr;
    int* h_blk_meta = nullptr;
    cudaGraphExec_t blk_graph = nullptr;
    const __nv_bfloat16* blk_graph_embed = nullptr;

    bool loaded = false;
    int ctx_lo = 0;

    ~Drafter() { release(); }

    void release() {
        if (blk_graph) { cudaGraphExecDestroy(blk_graph); blk_graph = nullptr; blk_graph_embed = nullptr; }
        auto cf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        cf(dev_blob); cf(dev_soa_blob); cf(kring); cf(vring); cf(pf_feat); cf(step_feat);
        cf(ctx_fc); cf(ctx_norm); cf(ctx_k); cf(ctx_v);
        cf(h); cf(xn); cf(xc); cf(dyn); cf(q_buf); cf(qf); cf(k_buf); cf(v_buf);
        cf(kblk); cf(vblk); cf(apart); cf(ao); cf(o_buf); cf(gate_buf); cf(up_buf); cf(act_buf);
        cf(qa.q); cf(qa.sc); cf(qa.isum);
        cf(out_bf16); cf(hp); cf(d_rank_cands); cf(d_tok_out); cf(d_lp_out); cf(d_blk_meta);
        if (h_tok_pinned) { cudaFreeHost(h_tok_pinned); h_tok_pinned = nullptr; }
        if (h_lp_pinned) { cudaFreeHost(h_lp_pinned); h_lp_pinned = nullptr; }
        if (h_blk_meta) { cudaFreeHost(h_blk_meta); h_blk_meta = nullptr; }
        loaded = false;
    }

    bool load_df2p(const std::string& path, bool alloc_pf_feat, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "cannot open DF2P pack: " + path; return false; }
        char magic[4] = {};
        uint32_t ver = 0, idx_len = 0;
        f.read(magic, 4);
        f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&idx_len), 4);
        if (std::memcmp(magic, "DF2P", 4) != 0 || ver != 1) {
            err = "bad DF2P header/version in " + path;
            return false;
        }
        std::string idx(idx_len, '\0');
        f.read(&idx[0], idx_len);
        const size_t data_base = 12 + idx_len;
        f.seekg(0, std::ios::end);
        const size_t file_size = static_cast<size_t>(f.tellg());
        if (file_size <= data_base) { err = "truncated DF2P pack"; return false; }
        blob_bytes = file_size - data_base;

        struct TInfo { std::string dtype; int rows, cols; size_t off, nbytes; };
        std::unordered_map<std::string, TInfo> tmap;
        std::istringstream ss(idx);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag == "end") break;
            if (tag == "t") {
                std::string name, dt;
                int r = 0, c = 0;
                size_t off = 0, nb = 0;
                ls >> name >> dt >> r >> c >> off >> nb;
                tmap[name] = TInfo{dt, r, c, off, nb};
            }
        }

        std::vector<uint8_t> hblob(blob_bytes);
        f.seekg(static_cast<std::streamoff>(data_base), std::ios::beg);
        f.read(reinterpret_cast<char*>(hblob.data()), static_cast<std::streamsize>(blob_bytes));
        if (!f) { err = "failed reading DF2P payload"; return false; }

        if (cudaMalloc(&dev_blob, blob_bytes) != cudaSuccess ||
            cudaMemcpy(dev_blob, hblob.data(), blob_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "cudaMalloc/Memcpy failed for DF2P blob";
            return false;
        }

        // Repack all Q4_0 matrices into coalesced 128-bit aligned SoA (qs + sc) for 10x faster GEMV!
        struct Q4Job { Q4Mat* dst; const uint8_t* src; int rows, cols; size_t off_qs, off_sc; };
        std::vector<Q4Job> q4_jobs;
        size_t soa_bytes = 0;
        auto a256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        auto get_q4 = [&](const std::string& n, Q4Mat& m) -> bool {
            auto it = tmap.find(n);
            if (it == tmap.end() || it->second.dtype != "q4_0") { err = "missing q4_0 tensor: " + n; return false; }
            m.rows = it->second.rows;
            m.cols = it->second.cols;
            const size_t nblks = static_cast<size_t>(m.rows) * (m.cols >> 5);
            const size_t oq = soa_bytes; soa_bytes += a256(nblks * 16);
            const size_t os = soa_bytes; soa_bytes += a256(nblks * sizeof(__half));
            q4_jobs.push_back({&m, dev_blob + it->second.off, m.rows, m.cols, oq, os});
            return true;
        };
        auto get_f32 = [&](const std::string& n, const float*& p) -> bool {
            auto it = tmap.find(n);
            if (it == tmap.end() || it->second.dtype != "f32") { err = "missing f32 tensor: " + n; return false; }
            p = reinterpret_cast<const float*>(dev_blob + it->second.off);
            return true;
        };
        auto get_bf16 = [&](const std::string& n, const __nv_bfloat16*& p) -> bool {
            auto it = tmap.find(n);
            if (it == tmap.end() || it->second.dtype != "bf16") { err = "missing bf16 tensor: " + n; return false; }
            p = reinterpret_cast<const __nv_bfloat16*>(dev_blob + it->second.off);
            return true;
        };

        if (!get_q4("fc.weight", fc)) return false;
        if (!get_f32("hidden_norm.weight", hidden_norm)) return false;
        if (!get_f32("norm.weight", final_norm)) return false;
        for (int l = 0; l < L; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            if (!get_q4(p + "self_attn.q_proj.weight", ly[l].q)) return false;
            if (!get_q4(p + "self_attn.k_proj.weight", ly[l].k)) return false;
            if (!get_q4(p + "self_attn.v_proj.weight", ly[l].v)) return false;
            if (!get_q4(p + "self_attn.o_proj.weight", ly[l].o)) return false;
            if (!get_q4(p + "mlp.gate_proj.weight", ly[l].gate)) return false;
            if (!get_q4(p + "mlp.up_proj.weight", ly[l].up)) return false;
            if (!get_q4(p + "mlp.down_proj.weight", ly[l].down)) return false;
            if (!get_q4(p + "attention_conv.kernel_projection.weight", ly[l].aconv_kp)) return false;
            if (!get_q4(p + "mlp_conv.kernel_projection.weight", ly[l].mconv_kp)) return false;
            if (!get_f32(p + "input_layernorm.weight", ly[l].in_norm)) return false;
            if (!get_f32(p + "post_attention_layernorm.weight", ly[l].post_norm)) return false;
            if (!get_f32(p + "self_attn.q_norm.weight", ly[l].q_norm)) return false;
            if (!get_f32(p + "self_attn.k_norm.weight", ly[l].k_norm)) return false;
            if (!get_f32(p + "attention_conv.base_kernel", ly[l].aconv_base)) return false;
            if (!get_f32(p + "mlp_conv.base_kernel", ly[l].mconv_base)) return false;
        }
        if (!get_q4("candidate_selector.hidden_projection.weight", hproj)) return false;
        if (!get_bf16("candidate_selector.predecessor_codebook", pred_cb)) return false;
        if (!get_bf16("candidate_selector.successor_codebook", succ_cb)) return false;

        if (soa_bytes > 0) {
            if (cudaMalloc(&dev_soa_blob, soa_bytes) != cudaSuccess) {
                err = "cudaMalloc failed for SoA Q4 weights";
                release();
                return false;
            }
            for (const auto& j : q4_jobs) {
                uint8_t* qs = dev_soa_blob + j.off_qs;
                __half* sc = reinterpret_cast<__half*>(dev_soa_blob + j.off_sc);
                const size_t nblks = static_cast<size_t>(j.rows) * (j.cols >> 5);
                k_repack_q4_soa<<<static_cast<unsigned>((nblks + 255) / 256), 256>>>(j.src, qs, sc, nblks);
                j.dst->qs = qs;
                j.dst->sc = sc;
            }
            cudaDeviceSynchronize();
        }

        // Optional Stage-A overlays
        if (auto it = tmap.find("qspine.dp_cal"); it != tmap.end() && it->second.nbytes == 4 * K_MAX * sizeof(float)) {
            const float* src = reinterpret_cast<const float*>(hblob.data() + it->second.off);
            std::memcpy(dp_cal.gamma, src + 0 * K_MAX, K_MAX * sizeof(float));
            std::memcpy(dp_cal.beta0, src + 1 * K_MAX, K_MAX * sizeof(float));
            std::memcpy(dp_cal.beta1, src + 2 * K_MAX, K_MAX * sizeof(float));
            std::memcpy(dp_cal.beta2, src + 3 * K_MAX, K_MAX * sizeof(float));
        }
        if (auto it_d = tmap.find("qspine.slot1.down"); it_d != tmap.end()) {
            if (auto it_u = tmap.find("qspine.slot1.up"); it_u != tmap.end()) {
                slot1_down = reinterpret_cast<const float*>(dev_blob + it_d->second.off);
                slot1_up = reinterpret_cast<const float*>(dev_blob + it_u->second.off);
                slot1_rank = it_d->second.rows;
            }
        }

        // Allocate FP16 KV rings + scratch
        auto da = [](auto*& ptr, size_t n) -> bool {
            void* p = nullptr;
            if (cudaMalloc(&p, (n ? n : 1) * sizeof(*ptr)) != cudaSuccess) return false;
            cudaMemset(p, 0, (n ? n : 1) * sizeof(*ptr));
            ptr = static_cast<std::remove_reference_t<decltype(ptr)>>(p);
            return true;
        };
        bool ok = da(kring, static_cast<size_t>(L) * NKV * RING * HD) &&
                  da(vring, static_cast<size_t>(L) * NKV * RING * HD) &&
                  (!alloc_pf_feat || da(pf_feat, static_cast<size_t>(WIN) * NT * HID)) &&
                  da(step_feat, static_cast<size_t>(B) * NT * HID) &&
                  da(ctx_fc, B * HID) && da(ctx_norm, B * HID) &&
                  da(ctx_k, B * NKV * HD) && da(ctx_v, B * NKV * HD) &&
                  da(h, B * HID) && da(xn, B * HID) && da(xc, B * HID) && da(dyn, B * CPO) &&
                  da(q_buf, B * NH * HD) && da(qf, B * NH * HD) &&
                  da(k_buf, B * NKV * HD) && da(v_buf, B * NKV * HD) &&
                  da(kblk, NKV * B * HD) && da(vblk, NKV * B * HD) &&
                  da(apart, static_cast<size_t>(NH) * ATTN_SPLIT * B * (HD + 2)) &&
                  da(ao, B * NH * HD) && da(o_buf, B * HID) &&
                  da(gate_buf, B * INTER) && da(up_buf, B * INTER) && da(act_buf, B * INTER) &&
                  da(qa.q, B * NT * HID) && da(qa.sc, B * (NT * HID / 32)) && da(qa.isum, B * (NT * HID / 16)) &&
                  da(out_bf16, K_MAX * HID) && da(hp, K_MAX * SELR) &&
                  da(d_rank_cands, 4 * K_MAX * TOPK) &&
                  da(d_tok_out, K_MAX + 1) && da(d_lp_out, K_MAX) && da(d_blk_meta, 3);
        if (!ok ||
            cudaHostAlloc(&h_tok_pinned, (K_MAX + 1) * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&h_lp_pinned, K_MAX * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&h_blk_meta, 3 * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
            err = "cudaMalloc failed for QSpine-DF2 rings/scratch";
            release();
            return false;
        }
        loaded = true;
        return true;
    }

    // Ingest n <= B contiguous committed context tap rows (FP16 [n, NT, HID]) at positions [pos0, pos0 + n).
    void ingest_context(const __half* feats_fp16, int n, int pos0, cudaStream_t st) {
        if (n <= 0) return;
        constexpr int kv_grid = (2 * NKV * HD + 3) / 4; // 512 CTAs for fused K+V
        for (int off = 0; off < n; off += B) {
            const int chunk = std::min(B, n - off);
            const __half* src = feats_fp16 + static_cast<size_t>(off) * NT * HID;
            quant_act_q8_f16(src, qa, chunk, NT * HID, st);
            launch_q4_gemv_q8(qa, fc, ctx_fc, chunk, st);
            k_df2_rmsnorm_q8<false, false><<<chunk, 256, 0, st>>>(ctx_fc, hidden_norm, nullptr, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
            for (int l = 0; l < L; ++l) {
                if (chunk == 1) k_q4_dp4a_kv<1, true><<<kv_grid, 128, 0, st>>>(qa.q, qa.sc, qa.isum, ly[l].k.qs, ly[l].k.sc, ly[l].v.qs, ly[l].v.sc, ctx_k, ctx_v, 1);
                else if (chunk <= 4) k_q4_dp4a_kv<4, false><<<kv_grid, 128, 0, st>>>(qa.q, qa.sc, qa.isum, ly[l].k.qs, ly[l].k.sc, ly[l].v.qs, ly[l].v.sc, ctx_k, ctx_v, chunk);
                else k_q4_dp4a_kv<8, false><<<kv_grid, 128, 0, st>>>(qa.q, qa.sc, qa.isum, ly[l].k.qs, ly[l].k.sc, ly[l].v.qs, ly[l].v.sc, ctx_k, ctx_v, chunk);
                __half* kr = kring + static_cast<size_t>(l) * NKV * RING * HD;
                __half* vr = vring + static_cast<size_t>(l) * NKV * RING * HD;
                k_df2_ctx_kv_write<<<chunk * NKV, 128, 0, st>>>(ctx_k, ctx_v, ly[l].k_norm, kr, vr, pos0 + off, RMS_EPS);
            }
        }
    }

    // Execute frozen bidirectional B = 8 block forward at q0 (anchor token at q0, masks at q0+1..q0+7).
    // Produces final-normed out_bf16 [7, 2560] and selector hidden projections hp [7, 256].
    void forward_block(const __nv_bfloat16* target_embed, int anchor_tok, int q0, cudaStream_t st, bool use_graph = true) {
        h_blk_meta[0] = anchor_tok;
        h_blk_meta[1] = q0;
        h_blk_meta[2] = ctx_lo;
        cudaMemcpyAsync(d_blk_meta, h_blk_meta, 3 * sizeof(int), cudaMemcpyHostToDevice, st);

        auto launch_body = [&]() {
            constexpr int qkv_grid = (NH * HD + 2 * NKV * HD + 3) / 4; // 1536 CTAs
            constexpr int mlp_grid = (INTER + 3) / 4;                  // 1920 CTAs
            k_df2_embed<<<B, 256, 0, st>>>(target_embed, d_blk_meta, h);
            for (int l = 0; l < L; ++l) {
                // Attention sublayer (fused rmsnorm+q8, fused conv0+q8, fused QKV!)
                k_df2_rmsnorm_q8<true, false><<<B, 256, 0, st>>>(h, ly[l].in_norm, xn, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
                launch_q4_gemv_q8(qa, ly[l].aconv_kp, dyn, B, st);
                k_df2_conv0_q8<<<dim3(HID / 256, B), 256, 0, st>>>(xn, dyn, ly[l].aconv_base, qa.q, qa.sc, qa.isum);
                k_q4_dp4a_qkv8<<<qkv_grid, 128, 0, st>>>(qa.q, qa.sc, qa.isum,
                                                         ly[l].q.qs, ly[l].q.sc, ly[l].k.qs, ly[l].k.sc, ly[l].v.qs, ly[l].v.sc,
                                                         q_buf, k_buf, v_buf);
                k_df2_q_norm_rope<<<B * NH, 128, 0, st>>>(q_buf, ly[l].q_norm, qf, d_blk_meta, RMS_EPS);
                k_df2_blk_kv<<<B * NKV, 128, 0, st>>>(k_buf, v_buf, ly[l].k_norm, kblk, vblk, d_blk_meta, RMS_EPS);
                const __half* kr = kring + static_cast<size_t>(l) * NKV * RING * HD;
                const __half* vr = vring + static_cast<size_t>(l) * NKV * RING * HD;
                k_df2_attn_split<<<dim3(NH, ATTN_SPLIT), 128, 0, st>>>(qf, kr, vr, kblk, vblk, d_blk_meta, apart);
                k_df2_attn_merge<<<B * NH, 128, 0, st>>>(apart, ao);
                quant_act_q8(ao, qa, B, NH * HD, st);
                launch_q4_gemv_q8(qa, ly[l].o, o_buf, B, st);
                k_df2_conv<true><<<dim3(HID / 256, B), 256, 0, st>>>(o_buf, dyn, ly[l].aconv_base, 1, h);

                // MLP sublayer (fused rmsnorm+q8, fused conv0+q8, fused gate+up+swiglu!)
                k_df2_rmsnorm_q8<true, false><<<B, 256, 0, st>>>(h, ly[l].post_norm, xn, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
                launch_q4_gemv_q8(qa, ly[l].mconv_kp, dyn, B, st);
                k_df2_conv0_q8<<<dim3(HID / 256, B), 256, 0, st>>>(xn, dyn, ly[l].mconv_base, qa.q, qa.sc, qa.isum);
                k_q4_dp4a_gate_up_swiglu8<<<mlp_grid, 128, 0, st>>>(qa.q, qa.sc, qa.isum,
                                                                    ly[l].gate.qs, ly[l].gate.sc, ly[l].up.qs, ly[l].up.sc,
                                                                    act_buf);
                quant_act_q8(act_buf, qa, B, INTER, st);
                launch_q4_gemv_q8(qa, ly[l].down, o_buf, B, st);
                k_df2_conv<true><<<dim3(HID / 256, B), 256, 0, st>>>(o_buf, dyn, ly[l].mconv_base, 1, h);
            }
            // Final norm on rows 1..7 (offsets +1..+7): fused out_bf16 + Q8_1 for hproj in ONE launch!
            if (slot1_down && slot1_up && slot1_rank > 0) {
                k_df2_slot1_adapter<<<1, 256, 0, st>>>(h + HID, slot1_down, slot1_up, slot1_rank);
            }
            k_df2_rmsnorm_q8<false, true><<<K_MAX, 256, 0, st>>>(h + HID, final_norm, nullptr, out_bf16, qa.q, qa.sc, qa.isum, RMS_EPS);
            launch_q4_gemv_q8(qa, hproj, hp, K_MAX, st);
        };

        if (use_graph) {
            if (!blk_graph || blk_graph_embed != target_embed) {
                if (blk_graph) { cudaGraphExecDestroy(blk_graph); blk_graph = nullptr; }
                cudaGraph_t g = nullptr;
                cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
                launch_body();
                cudaStreamEndCapture(st, &g);
                cudaGraphInstantiate(&blk_graph, g, nullptr, nullptr, 0);
                cudaGraphDestroy(g);
                blk_graph_embed = target_embed;
            }
            cudaGraphLaunch(blk_graph, st);
        } else {
            launch_body();
        }
    }

    // Run local top-16 extraction for rank index `rank_slot` (0 for single-rank / rank0, or i in m4sim).
    // Reuses PR #599's already-resident MtpHead2 2-bit screen + exact FP8 rescore at ZERO extra VRAM!
    void extract_local_top16(int rank_slot, int K, bool use_head2,
                             const MtpHead2* h2, const Fp8Mat* fp8_lm, const __nv_bfloat16* bf16_lm,
                             int v0, int vn, float* head_logits, MtpHead2Scratch& h2_sc, cudaStream_t st) {
        Cand16* dst = d_rank_cands + static_cast<size_t>(rank_slot) * K_MAX * TOPK;
        if (use_head2 && h2 && h2->ready() && fp8_lm && fp8_lm->q) {
            mtp_head2_top16(*h2, *fp8_lm, out_bf16, head_logits, h2_sc, dst, K, st);
        } else {
            if (fp8_lm && fp8_lm->q) {
                launch_fp8_gemv_smallm<true>(out_bf16, *fp8_lm, head_logits, K, st);
            } else {
                launch_bf16_gemv_smallm<true>(out_bf16, bf16_lm, head_logits, K, vn, HID, st);
            }
            k_qw38_head2_top<16, false><<<K, 1024, 0, st>>>(head_logits, vn, h2_sc.cand);
            k_df2_sort_top16_from_y<<<K, 256, 0, st>>>(head_logits, vn, v0, h2_sc.cand, dst);
        }
    }

    // Merge all `nr` ranks' sorted top-16 and select K_eff in 1..K draft tokens via Greedy (0), Viterbi DP (1), or Dynamic-K* DP (2).
    void select_chain(int nr, int K, int anchor_tok, float sel_alpha, int use_dp,
                      DraftResult& res, cudaStream_t st) {
        // Empirical PR #599 GB10 round cost (base_verify + draft_round + gdn_commit, plus per-draft verify slope)
        const float tv_base = (nr == 1) ? 30.5f : 20.5f;
        const float tv_step = (nr == 1) ? 2.75f : 1.75f;
        k_df2_walk_dp<<<1, 256, 0, st>>>(d_rank_cands, nr, K, anchor_tok, hp, pred_cb, succ_cb,
                                         dp_cal, sel_alpha, use_dp, tv_base, tv_step, d_tok_out, d_lp_out);
        cudaMemcpyAsync(h_tok_pinned, d_tok_out, static_cast<size_t>(K_MAX + 1) * sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_lp_pinned, d_lp_out, static_cast<size_t>(K) * sizeof(float), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        const int keff = std::max(1, std::min(K, h_tok_pinned[K_MAX]));
        res.K = keff;
        for (int i = 0; i < keff; ++i) {
            res.tok[i] = h_tok_pinned[i];
            res.lp[i] = h_lp_pinned[i];
        }
    }
};

} // namespace dfl
} // namespace qw38
