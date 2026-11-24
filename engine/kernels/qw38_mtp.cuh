// Qwen3.8-Flash-Next 1-layer Multi-Token Prediction (MTP, pack layer L048) kernels.
// Architecture (verified against exlr8-mtp.safetensors tensor shapes):
//   - MTP.pre_fc_norm_embedding [2560]  (1 + w RMSNorm on next token embedding)
//   - MTP.pre_fc_norm_hidden    [10240] (1 + w 4-stream grouped RMSNorm on trunk 4-stream hyper state)
//   - MTP.fc_embedding          [2560, 2560]
//   - MTP.fc_hidden             [2560, 2560] (applied per stream s=0..3 -> [M, 4, 2560])
//   - L048: 1 full-attention + QSA + 512 routed experts (INT4/FP8/BF16) + gated shared + 4-stream HC layer
//   - MTP.hc_final: 4-stream -> 2560 mixer before shared lm_head (2-bit screen + top-64 exact rescore).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include "qw38_hc.cuh"
#include "qw38_fp8.cuh"
#include "qw38_moe.cuh"
#include "qw38_mtp_head2.cuh"
#include "../attn/qw38_attn.cuh"

namespace qw38 {

// ------------------------------------------------------------------------------------------------
// Block-32 Symmetric Signed INT4 ([-8..7] + 8, interleaved even/odd nibbles per uint32) for MTP L048.
// 1 uint4 (16 bytes) = 32 INT4 weights = 1 exact block-32 = 1 FP16 scale (4.5 bits/weight).
// ------------------------------------------------------------------------------------------------
struct Int4Mat {
    const uint8_t* q = nullptr;   // [N, K / 2] interleaved INT4
    const __half* sc = nullptr;   // [N, K / 32] FP16 scale = amax / 7.0f
    int N = 0;
    int K = 0;
};

__global__ void __launch_bounds__(256) k_quantize_bf16_to_int4_b32(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ q_out,
    __half* __restrict__ sc_out,
    int N, int K) {
    const int nb_row = K >> 5;
    const size_t total_blks = static_cast<size_t>(N) * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blks) return;

    const float w = __bfloat162float(W[blk * 32 + lane]);
    const float mx = fp8_warp_max(fabsf(w));
    const __half sh = __float2half_rn(mx / 7.0f);
    const float s = __half2float(sh);
    const uint32_t qv = (s > 0.0f)
        ? static_cast<uint32_t>(min(max(__float2int_rn(w / s), -8), 7) + 8)
        : 8u;
    if (lane == 0) sc_out[blk] = sh;

    // Pack 8 consecutive weights into one uint32: even indices in bits 0..15, odd indices in bits 16..31.
    const int sub = lane & 7;
    const int bit_pos = 4 * (sub >> 1) + 16 * (sub & 1);
    uint32_t part = qv << bit_pos;
    part |= __shfl_xor_sync(0xffffffffu, part, 1);
    part |= __shfl_xor_sync(0xffffffffu, part, 2);
    part |= __shfl_xor_sync(0xffffffffu, part, 4);
    if (sub == 0) {
        reinterpret_cast<uint32_t*>(q_out)[blk * 4 + (lane >> 3)] = part;
    }
}

inline void quantize_bf16_to_int4_b32(const __nv_bfloat16* W, uint8_t* q, __half* sc,
                                      int N, int K, cudaStream_t st) {
    const size_t total_blks = static_cast<size_t>(N) * (K >> 5);
    const unsigned grid = static_cast<unsigned>((total_blks + 7) / 8);
    k_quantize_bf16_to_int4_b32<<<grid, 256, 0, st>>>(W, q, sc, N, K);
}

// Dot 32 BF16 activations (4 x uint4) with 1 block-32 of interleaved INT4 weights (1 x uint4), unscaled FP32.
__device__ __forceinline__ float dot_bf16x32_int4_raw(uint4 a0, uint4 a1, uint4 a2, uint4 a3, uint4 wv) {
    const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
    const uint4 as[4] = {a0, a1, a2, a3};
    const __half2 off = __float2half2_rn(1032.0f);
    float s = 0.0f;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&as[p]);
        const uint32_t wp = ws[p];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t h2 = ((wp >> (4 * i)) & 0x000f000fu) | 0x64006400u;
            const __half2 wf = __hsub2(*reinterpret_cast<const __half2*>(&h2), off);
            const float2 fw = __half22float2(wf);
            const float2 fa = __bfloat1622float2(ap[i]);
            s = fmaf(fa.x, fw.x, fmaf(fa.y, fw.y, s));
        }
    }
    return s;
}

struct Int4Arena {
    uint8_t* base = nullptr;
    size_t cap = 0;
    size_t used = 0;
    struct Pending {
        const __nv_bfloat16* src;
        Int4Mat* dst;
        int N, K;
    };
    std::vector<Pending> pend;

    ~Int4Arena() { if (base) cudaFree(base); }

    static size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

    void add(const __nv_bfloat16* src, Int4Mat& dst, int N, int K) {
        if (!src || N <= 0 || K <= 0 || (K & 31) != 0) return;
        pend.push_back({src, &dst, N, K});
    }

    size_t build(cudaStream_t st) {
        if (pend.empty()) return 0;
        size_t total = 0;
        for (const auto& p : pend) {
            total += align256(static_cast<size_t>(p.N) * (p.K >> 1));
            total += align256(static_cast<size_t>(p.N) * (p.K >> 5) * sizeof(__half));
        }
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, total) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        base = static_cast<uint8_t*>(ptr);
        cap = total;
        used = 0;
        for (const auto& p : pend) {
            uint8_t* qp = base + used;
            used += align256(static_cast<size_t>(p.N) * (p.K >> 1));
            __half* sp = reinterpret_cast<__half*>(base + used);
            used += align256(static_cast<size_t>(p.N) * (p.K >> 5) * sizeof(__half));
            quantize_bf16_to_int4_b32(p.src, qp, sp, p.N, p.K, st);
            p.dst->q = qp;
            p.dst->sc = sp;
            p.dst->N = p.N;
            p.dst->K = p.K;
        }
        pend.clear();
        return total;
    }
};

struct MtpWeights {
    const __nv_bfloat16* pre_fc_norm_emb = nullptr;  // [2560]
    const __nv_bfloat16* pre_fc_norm_hid = nullptr;  // [10240]
    const __nv_bfloat16* fc_emb = nullptr;           // [2560, 2560]
    const __nv_bfloat16* fc_hid = nullptr;           // [2560, 2560]
    Fp8Mat fp8_fc_emb;
    Fp8Mat fp8_fc_hid;
    HcWeights attn_hc;
    HcWeights mlp_hc;
    HcWeights final_hc;
    // Per-rank L048 weights (index 0..nloc-1)
    std::vector<FullAttnWeights> attn;
    std::vector<FullLayerKv> kv;
    std::vector<MoeWeights> moe;
    // Quantized MTP 512 routed experts per local rank (INT4 default when GR8_QW38_MTP_FAST=1, FP8 fallback):
    std::vector<Fp8Mat> fp8_exp_gu;
    std::vector<Fp8Mat> fp8_exp_dn;
    std::vector<Int4Mat> i4_exp_gu;
    std::vector<Int4Mat> i4_exp_dn;
};

struct MtpScratch {
    __nv_bfloat16* emb_raw = nullptr;     // [max_S, 2560]
    __nv_bfloat16* emb_norm = nullptr;    // [max_S, 2560]
    __nv_bfloat16* emb_proj = nullptr;    // [max_S, 2560]
    __nv_bfloat16* hid_norm = nullptr;    // [max_S, 10240]
    __nv_bfloat16* hid_proj = nullptr;    // [max_S, 10240]
    __nv_bfloat16* mtp_hyper = nullptr;   // [max_S, 10240]
    __nv_bfloat16* exp_mid = nullptr;     // [max_S * 10, 640] BF16
    float* exp_dn_part = nullptr;         // [max_S * 10, 2560] FP32
};

__global__ void k_gather_embed(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ embed,
                               __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const uint4* src = reinterpret_cast<const uint4*>(embed + static_cast<size_t>(tok[m]) * geom::HID);
    uint4* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * geom::HID);
    constexpr int D8 = geom::HID >> 3;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) dst[i] = src[i];
}

// Graph-safe gather of last_hyper[d_slot[m]] -> mtp_hid_in[m] (works for any S2 slot subset!).
__global__ void k_mtp_gather_hid(const int* __restrict__ d_slot, const __nv_bfloat16* __restrict__ last_hyper,
                                 __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int slot = d_slot[m];
    const uint4* src = reinterpret_cast<const uint4*>(last_hyper + static_cast<size_t>(slot) * geom::HC_D);
    uint4* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * geom::HC_D);
    constexpr int H8 = geom::HC_D >> 3; // 1280
    for (int i = threadIdx.x; i < H8; i += blockDim.x) dst[i] = src[i];
}

// Graph-safe save/restore of L048 idx_raw & idx_raw_thw across speculative k >= 1 draft steps!
template <bool SAVE>
__global__ void k_mtp_idx_snap(const int* __restrict__ d_slot,
                               __nv_bfloat16* __restrict__ idx_raw,
                               __nv_bfloat16* __restrict__ idx_raw_snap,
                               int3* __restrict__ idx_raw_thw,
                               int3* __restrict__ idx_raw_thw_snap,
                               int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int slot = d_slot[m];
    const int tid = threadIdx.x; // 0..63 -> 64 x uint4 = 512 BF16 = 4 * 128 EXACT!
    const size_t off_u4 = static_cast<size_t>(slot) * 64 + tid;
    if (idx_raw && idx_raw_snap && tid < 64) {
        if constexpr (SAVE) {
            reinterpret_cast<uint4*>(idx_raw_snap)[off_u4] = reinterpret_cast<const uint4*>(idx_raw)[off_u4];
        } else {
            reinterpret_cast<uint4*>(idx_raw)[off_u4] = reinterpret_cast<const uint4*>(idx_raw_snap)[off_u4];
        }
    }
    if (tid == 0 && idx_raw_thw && idx_raw_thw_snap) {
        if constexpr (SAVE) idx_raw_thw_snap[slot] = idx_raw_thw[slot];
        else idx_raw_thw[slot] = idx_raw_thw_snap[slot];
    }
}

// On-GPU TP=1 greedy MTP draft chain: writes draft token into d_draft_out[m*K + k] (and optional lp) and advances d_tok/d_pos/d_rope!
__global__ void k_mtp_chain_tp1(const HeadArgmax* __restrict__ argmax,
                                int* __restrict__ d_draft_out,
                                int* __restrict__ d_tok,
                                int* __restrict__ d_pos,
                                int3* __restrict__ d_rope,
                                int M, int K, int k,
                                const float* __restrict__ d_lp = nullptr,
                                float* __restrict__ d_draft_lp = nullptr) {
    const int m = threadIdx.x;
    if (m >= M) return;
    const int tok = argmax[m].id;
    d_draft_out[m * K + k] = tok;
    if (d_lp && d_draft_lp) d_draft_lp[m * K + k] = d_lp[m];
    if (k + 1 < K) {
        d_tok[m] = tok;
        d_pos[m] += 1;
        int3 rp = d_rope[m];
        rp.x += 1; rp.y += 1; rp.z += 1;
        d_rope[m] = rp;
    }
}

// Per-stream fuse: out[m, s, d] = bf16(h_proj[m, s, d] + e_proj[m, d])
__global__ void k_mtp_fuse_stream4(const __nv_bfloat16* __restrict__ h_proj,
                                   const __nv_bfloat16* __restrict__ e_proj,
                                   __nv_bfloat16* __restrict__ out4, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const uint4* ep = reinterpret_cast<const uint4*>(e_proj + static_cast<size_t>(m) * geom::HID);
    constexpr int D8 = geom::HID >> 3;
    for (int i = threadIdx.x; i < D8; i += blockDim.x) {
        const uint4 ev = ep[i];
        const __nv_bfloat162* e2 = reinterpret_cast<const __nv_bfloat162*>(&ev);
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) {
            const uint4 hv = reinterpret_cast<const uint4*>(h_proj + static_cast<size_t>(m) * geom::HC_D + s * geom::HID)[i];
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hv);
            uint4 ov;
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 fh = __bfloat1622float2(h2[k]);
                const float2 fe = __bfloat1622float2(e2[k]);
                o2[k] = __floats2bfloat162_rn(fh.x + fe.x, fh.y + fe.y);
            }
            reinterpret_cast<uint4*>(out4 + static_cast<size_t>(m) * geom::HC_D + s * geom::HID)[i] = ov;
        }
    }
}

// Fast GPU-native MTP Routed MoE FC1 (gate+up+silu) in block-32 INT4!
// 16 lanes per channel (2 channels/warp, 8 channels/CTA), NB=80 / 16 = 5 EXACT iterations (0% idle lanes!).
// Streaming __ldcs on weights, 1x read of x for both gate & up, 1 fmaf scale per 32 weights!
__global__ void __launch_bounds__(128) k_mtp_exp_fc1_int4(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ topk_idx,
    const uint8_t* __restrict__ gu_q, const __half* __restrict__ gu_sc,
    __nv_bfloat16* __restrict__ mid_out,
    int M, int sh_i) {
    const int n_blks = (sh_i + 7) >> 3;
    const int route = blockIdx.x / n_blks;
    const int blk = blockIdx.x % n_blks;
    const int m = route / geom::TOPK;
    if (m >= M) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub_r = lane >> 4; // 0..1 (2 channels/warp)
    const int sub_l = lane & 15; // 0..15 (16 lanes/channel)
    const int n = blk * 8 + warp * 2 + sub_r;
    if (n >= sh_i) return;

    const int e = topk_idx[route];
    const size_t exp_rows = static_cast<size_t>(2) * sh_i;
    const size_t g_row = static_cast<size_t>(e) * exp_rows + n;
    const size_t u_row = g_row + sh_i;

    constexpr int NB = geom::HID >> 5; // 80 = 5 * 16 EXACT
    const uint4* gq = reinterpret_cast<const uint4*>(gu_q + g_row * (geom::HID >> 1));
    const uint4* uq = reinterpret_cast<const uint4*>(gu_q + u_row * (geom::HID >> 1));
    const __half* gs = gu_sc + g_row * NB;
    const __half* us = gu_sc + u_row * NB;
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);

    float ag = 0.0f, au = 0.0f;
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int b = it * 16 + sub_l;
        const uint4 a0 = xm[4 * b], a1 = xm[4 * b + 1], a2 = xm[4 * b + 2], a3 = xm[4 * b + 3];
        const uint4 gw = __ldcs(gq + b);
        const uint4 uw = __ldcs(uq + b);
        const float g_sc = __half2float(__ldcs(gs + b));
        const float u_sc = __half2float(__ldcs(us + b));
        ag = fmaf(dot_bf16x32_int4_raw(a0, a1, a2, a3, gw), g_sc, ag);
        au = fmaf(dot_bf16x32_int4_raw(a0, a1, a2, a3, uw), u_sc, au);
    }
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        ag += __shfl_xor_sync(0xffffffffu, ag, off);
        au += __shfl_xor_sync(0xffffffffu, au, off);
    }
    if (sub_l == 0) {
        const float g = __bfloat162float(__float2bfloat16(ag));
        const float u = __bfloat162float(__float2bfloat16(au));
        mid_out[static_cast<size_t>(route) * sh_i + n] = __float2bfloat16(silu_f(g) * u);
    }
}

// Fast GPU-native MTP Routed MoE FC2 (down) in block-32 INT4!
// Caches mid_in[route, 0..sh_i-1] (<= 1.28 KB) in shared memory ONCE per CTA across 64 output dims!
// Grid: (M * TOPK * (2560 / 64)) = M * 400 CTAs (16x fewer CTAs than 6,400!), Block: 256 (4 lanes/dim).
__global__ void __launch_bounds__(256) k_mtp_exp_fc2_int4(
    const __nv_bfloat16* __restrict__ mid_in,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const uint8_t* __restrict__ dn_q, const __half* __restrict__ dn_sc,
    float* __restrict__ dn_part,
    int M, int sh_i) {
    constexpr int D_BLKS = geom::HID / 64; // 40
    const int route = blockIdx.x / D_BLKS;
    const int blk = blockIdx.x % D_BLKS;
    const int m = route / geom::TOPK;
    if (m >= M) return;

    __shared__ uint4 s_mid[geom::SHARED_I >> 3]; // max 80 uint4 = 1280 B
    const int k8 = sh_i >> 3;
    const uint4* am = reinterpret_cast<const uint4*>(mid_in + static_cast<size_t>(route) * sh_i);
    if (threadIdx.x < k8) s_mid[threadIdx.x] = am[threadIdx.x];
    __syncthreads();

    const int dim_loc = threadIdx.x >> 2; // 0..63
    const int sub_l = threadIdx.x & 3;    // 0..3 (4 lanes per output dim)
    const int d = blk * 64 + dim_loc;
    const int e = topk_idx[route];
    const size_t row = static_cast<size_t>(e) * geom::HID + d;
    const int nb = sh_i >> 5;

    const uint4* dq = reinterpret_cast<const uint4*>(dn_q + row * (sh_i >> 1));
    const __half* ds = dn_sc + row * nb;

    float acc = 0.0f;
    for (int b = sub_l; b < nb; b += 4) {
        const uint4 wv = __ldcs(dq + b);
        const float sc = __half2float(__ldcs(ds + b));
        acc = fmaf(dot_bf16x32_int4_raw(s_mid[4 * b], s_mid[4 * b + 1], s_mid[4 * b + 2], s_mid[4 * b + 3], wv), sc, acc);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if (sub_l == 0) dn_part[static_cast<size_t>(route) * geom::HID + d] = topk_w[route] * acc;
}

// Fast GPU-native MTP Routed MoE FC1 (gate+up+silu) in block-32 FP8/MXFP8 (fallback when GR8_QW38_MTP_FAST=0).
template <bool MX_SC = false>
__global__ void __launch_bounds__(128) k_mtp_exp_fc1_fp8(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ topk_idx,
    const uint8_t* __restrict__ gu_q, const __half* __restrict__ gu_sc, const uint8_t* __restrict__ gu_mx,
    __nv_bfloat16* __restrict__ mid_out,
    int M, int sh_i) {
    const int n_blks = (sh_i + 3) >> 2;
    const int route = blockIdx.x / n_blks;
    const int blk = blockIdx.x % n_blks;
    const int m = route / geom::TOPK;
    if (m >= M) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blk * 4 + warp;
    if (n >= sh_i) return;

    const int e = topk_idx[route];
    const size_t exp_rows = static_cast<size_t>(2) * sh_i;
    const size_t g_row = static_cast<size_t>(e) * exp_rows + n;
    const size_t u_row = g_row + sh_i;

    constexpr int K16 = geom::HID >> 4; // 160
    constexpr int NB = geom::HID >> 5;  // 80
    const uint4* gq = reinterpret_cast<const uint4*>(gu_q + g_row * geom::HID);
    const uint4* uq = reinterpret_cast<const uint4*>(gu_q + u_row * geom::HID);
    const size_t gsb = g_row * NB;
    const size_t usb = u_row * NB;
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);

    float ag = 0.0f, au = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const uint4 a0 = xm[2 * i], a1 = xm[2 * i + 1];
        float2 gf[8], uf[8];
        fp8x16_unpack(__ldg(gq + i), load_fp8_scale<MX_SC>(gu_sc, gu_mx, gsb + (i >> 1)), gf);
        fp8x16_unpack(__ldg(uq + i), load_fp8_scale<MX_SC>(gu_sc, gu_mx, usb + (i >> 1)), uf);
        ag += dot_bf16x16_f2x8(a0, a1, gf);
        au += dot_bf16x16_f2x8(a0, a1, uf);
    }
    const float g = __bfloat162float(__float2bfloat16(warp_sum(ag)));
    const float u = __bfloat162float(__float2bfloat16(warp_sum(au)));
    if (lane == 0) mid_out[static_cast<size_t>(route) * sh_i + n] = __float2bfloat16(silu_f(g) * u);
}

// Fast GPU-native MTP Routed MoE FC2 (down) in block-32 FP8/MXFP8 (fallback when GR8_QW38_MTP_FAST=0).
template <bool MX_SC = false>
__global__ void __launch_bounds__(128) k_mtp_exp_fc2_fp8(
    const __nv_bfloat16* __restrict__ mid_in,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const uint8_t* __restrict__ dn_q, const __half* __restrict__ dn_sc, const uint8_t* __restrict__ dn_mx,
    float* __restrict__ dn_part,
    int M, int sh_i) {
    constexpr int D_BLKS = geom::HID >> 2; // 640
    const int route = blockIdx.x / D_BLKS;
    const int blk = blockIdx.x % D_BLKS;
    const int m = route / geom::TOPK;
    if (m >= M) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = blk * 4 + warp;

    const int e = topk_idx[route];
    const size_t row = static_cast<size_t>(e) * geom::HID + d;
    const int k16 = sh_i >> 4;
    const int nb = sh_i >> 5;
    const uint4* dq = reinterpret_cast<const uint4*>(dn_q + row * sh_i);
    const size_t dsb = row * nb;
    const uint4* am = reinterpret_cast<const uint4*>(mid_in + static_cast<size_t>(route) * sh_i);

    float acc = 0.0f;
    for (int i = lane; i < k16; i += 32) {
        float2 wf[8];
        fp8x16_unpack(__ldg(dq + i), load_fp8_scale<MX_SC>(dn_sc, dn_mx, dsb + (i >> 1)), wf);
        acc += dot_bf16x16_f2x8(am[2 * i], am[2 * i + 1], wf);
    }
    const float tot = warp_sum(acc);
    if (lane == 0) dn_part[static_cast<size_t>(route) * geom::HID + d] = topk_w[route] * tot;
}

// Fast BF16 fallback when GR8_QW38_DENSE_FP8=0 (still GPU-native, zero host sync!).
__global__ void __launch_bounds__(128) k_mtp_exp_fc1_bf16(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ topk_idx,
    const __nv_bfloat16* __restrict__ gu_w,
    __nv_bfloat16* __restrict__ mid_out,
    int M, int sh_i) {
    const int n_blks = (sh_i + 3) >> 2;
    const int route = blockIdx.x / n_blks;
    const int blk = blockIdx.x % n_blks;
    const int m = route / geom::TOPK;
    if (m >= M) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 4 + warp;
    if (n >= sh_i) return;
    const int e = topk_idx[route];
    const size_t g_row = static_cast<size_t>(e) * (2 * sh_i) + n;
    const uint4* gw = reinterpret_cast<const uint4*>(gu_w + g_row * geom::HID);
    const uint4* uw = reinterpret_cast<const uint4*>(gu_w + (g_row + sh_i) * geom::HID);
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);
    float ag = 0.0f, au = 0.0f;
    constexpr int K8 = geom::HID >> 3;
    for (int i = lane; i < K8; i += 32) {
        const uint4 xv = xm[i];
        ag += dot_bf16x8(xv, __ldg(gw + i));
        au += dot_bf16x8(xv, __ldg(uw + i));
    }
    const float g = __bfloat162float(__float2bfloat16(warp_sum(ag)));
    const float u = __bfloat162float(__float2bfloat16(warp_sum(au)));
    if (lane == 0) mid_out[static_cast<size_t>(route) * sh_i + n] = __float2bfloat16(silu_f(g) * u);
}

__global__ void __launch_bounds__(128) k_mtp_exp_fc2_bf16(
    const __nv_bfloat16* __restrict__ mid_in,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const __nv_bfloat16* __restrict__ dn_w,
    float* __restrict__ dn_part,
    int M, int sh_i) {
    constexpr int D_BLKS = geom::HID >> 2;
    const int route = blockIdx.x / D_BLKS;
    const int blk = blockIdx.x % D_BLKS;
    const int m = route / geom::TOPK;
    if (m >= M) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int d = blk * 4 + warp;
    const int e = topk_idx[route];
    const size_t row = static_cast<size_t>(e) * geom::HID + d;
    const uint4* dw = reinterpret_cast<const uint4*>(dn_w + row * sh_i);
    const uint4* am = reinterpret_cast<const uint4*>(mid_in + static_cast<size_t>(route) * sh_i);
    float acc = 0.0f;
    const int k8 = sh_i >> 3;
    for (int i = lane; i < k8; i += 32) acc += dot_bf16x8(am[i], __ldg(dw + i));
    const float tot = warp_sum(acc);
    if (lane == 0) dn_part[static_cast<size_t>(route) * geom::HID + d] = topk_w[route] * tot;
}

// Fast MTP Layer-48 MoE forward (uses INT4 GPU-native routed experts when GR8_QW38_MTP_FAST=1, else FP8/BF16!).
inline void mtp_moe_forward(cublasHandle_t blas, const MoeWeights& w,
                            const Int4Mat& i4_gu, const Int4Mat& i4_dn,
                            const Fp8Mat& fp8_gu, const Fp8Mat& fp8_dn,
                            MoeScratch& sc, MtpScratch& msc, const __nv_bfloat16* x, __nv_bfloat16* out_part,
                            int M, cudaStream_t st) {
    // 1. Router + shared expert (same fast path as trunk MoE!)
    if (M <= 8) {
        const int rgrid = (geom::NEXP + 1 + 3) / 4;
        const int sgrid = (w.sh_i + 3) / 4;
        if (M == 1) {
            k_qw38_router_and_shgate<1><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) launch_fp8_sh_gate_up_silu<1>(x, w.fp8_sh_gate, w.fp8_sh_up, sc.sh_gate_buf, M, w.sh_i, st);
            else k_qw38_sh_gate_up_silu<1><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else if (M <= 4) {
            k_qw38_router_and_shgate<4><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) launch_fp8_sh_gate_up_silu<4>(x, w.fp8_sh_gate, w.fp8_sh_up, sc.sh_gate_buf, M, w.sh_i, st);
            else k_qw38_sh_gate_up_silu<4><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else {
            k_qw38_router_and_shgate<8><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) launch_fp8_sh_gate_up_silu<8>(x, w.fp8_sh_gate, w.fp8_sh_up, sc.sh_gate_buf, M, w.sh_i, st);
            else k_qw38_sh_gate_up_silu<8><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        }
        k_qw38_router_topk10<<<M, 256, 0, st>>>(sc.router_logits, sc.topk_idx, sc.topk_w, M);
        if (w.fp8_sh_down.q) launch_fp8_gemv_smallm<false>(sc.sh_gate_buf, w.fp8_sh_down, sc.sh_out, M, st);
        else launch_bf16_gemv_smallm<false>(sc.sh_gate_buf, w.sh_down, sc.sh_out, M, geom::HID, w.sh_i, st);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt_f32(blas, M, geom::NEXP, geom::HID, x, w.router, sc.router_logits);
        k_qw38_router_topk10<<<M, 256, 0, st>>>(sc.router_logits, sc.topk_idx, sc.topk_w, M);
        gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_gate, sc.sh_gate_buf);
        gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_up, sc.sh_up_buf);
        k_silu_mul_bf16<<<(M * w.sh_i + 255) / 256, 256, 0, st>>>(sc.sh_gate_buf, sc.sh_up_buf, M * w.sh_i);
        gemm_bf16_nt(blas, M, geom::HID, w.sh_i, sc.sh_gate_buf, w.sh_down, sc.sh_out);
        gemm_bf16_nt_f32(blas, M, 1, geom::HID, x, w.sh_exp_gate, sc.sh_scalar_gate);
    }

    // 2. Routed experts (GPU-native INT4 / FP8 / BF16, zero host sync) + deterministic combine!
    const int n_routes = M * geom::TOPK;
    if (i4_gu.q && i4_dn.q) {
        const int fc1_grid = n_routes * ((w.sh_i + 7) / 8);
        const int fc2_grid = n_routes * (geom::HID / 64);
        k_mtp_exp_fc1_int4<<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, i4_gu.q, i4_gu.sc, msc.exp_mid, M, w.sh_i);
        k_mtp_exp_fc2_int4<<<fc2_grid, 256, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, i4_dn.q, i4_dn.sc, msc.exp_dn_part, M, w.sh_i);
    } else if (fp8_gu.q && fp8_dn.q) {
        const int fc1_grid = n_routes * ((w.sh_i + 3) / 4);
        const int fc2_grid = n_routes * (geom::HID / 4);
        if (fp8_gu.mx_sc) {
            k_mtp_exp_fc1_fp8<true><<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, fp8_gu.q, fp8_gu.sc, fp8_gu.mx_sc, msc.exp_mid, M, w.sh_i);
            k_mtp_exp_fc2_fp8<true><<<fc2_grid, 128, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, fp8_dn.q, fp8_dn.sc, fp8_dn.mx_sc, msc.exp_dn_part, M, w.sh_i);
        } else {
            k_mtp_exp_fc1_fp8<false><<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, fp8_gu.q, fp8_gu.sc, fp8_gu.mx_sc, msc.exp_mid, M, w.sh_i);
            k_mtp_exp_fc2_fp8<false><<<fc2_grid, 128, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, fp8_dn.q, fp8_dn.sc, fp8_dn.mx_sc, msc.exp_dn_part, M, w.sh_i);
        }
    } else {
        const int fc1_grid = n_routes * ((w.sh_i + 3) / 4);
        const int fc2_grid = n_routes * (geom::HID / 4);
        k_mtp_exp_fc1_bf16<<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, w.mtp_gate_up, msc.exp_mid, M, w.sh_i);
        k_mtp_exp_fc2_bf16<<<fc2_grid, 128, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, w.mtp_down, msc.exp_dn_part, M, w.sh_i);
    }
    k_qw38_moe_combine_det_m10<<<M * 10, 64, 0, st>>>(msc.exp_dn_part, sc.sh_out, sc.sh_scalar_gate, out_part, M);
}

} // namespace qw38
