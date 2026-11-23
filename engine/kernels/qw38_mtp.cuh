// Qwen3.8-Flash-Next 1-layer Multi-Token Prediction (MTP, pack layer L048) kernels.
// Architecture (verified against exlr8-mtp.safetensors tensor shapes):
//   - MTP.pre_fc_norm_embedding [2560]  (1 + w RMSNorm on next token embedding)
//   - MTP.pre_fc_norm_hidden    [10240] (1 + w 4-stream grouped RMSNorm on trunk 4-stream hyper state)
//   - MTP.fc_embedding          [2560, 2560]
//   - MTP.fc_hidden             [2560, 2560] (applied per stream s=0..3 -> [M, 4, 2560])
//   - L048: 1 full-attention + QSA + 512 routed experts (FP8/BF16) + gated shared + 4-stream HC layer
//   - MTP.hc_final: 4-stream -> 2560 mixer before shared lm_head.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "qw38_hc.cuh"
#include "qw38_fp8.cuh"
#include "qw38_moe.cuh"
#include "../attn/qw38_attn.cuh"

namespace qw38 {

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
    // Quantized FP8 MTP 512 routed experts per local rank:
    //   gate_up: [512, 2*sh_i, 2560] FP8 + scales
    //   down:    [512, 2560, sh_i]   FP8 + scales
    std::vector<Fp8Mat> fp8_exp_gu;
    std::vector<Fp8Mat> fp8_exp_dn;
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

// Fast GPU-native MTP Routed MoE FC1 (gate+up+silu) in block-32 FP8 (ZERO host sync, ZERO cuBLAS!).
// Grid: (M * TOPK * ((sh_i + 3) / 4)), Block: 128 (4 warps = 4 output channels of expert e).
__global__ void __launch_bounds__(128) k_mtp_exp_fc1_fp8(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ topk_idx,
    const uint8_t* __restrict__ gu_q, const __half* __restrict__ gu_sc,
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
    const __half* gs = gu_sc + g_row * NB;
    const __half* us = gu_sc + u_row * NB;
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);

    float ag = 0.0f, au = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const uint4 a0 = xm[2 * i], a1 = xm[2 * i + 1];
        float2 gf[8], uf[8];
        fp8x16_unpack(__ldg(gq + i), __half2float(__ldg(gs + (i >> 1))), gf);
        fp8x16_unpack(__ldg(uq + i), __half2float(__ldg(us + (i >> 1))), uf);
        ag += dot_bf16x16_f2x8(a0, a1, gf);
        au += dot_bf16x16_f2x8(a0, a1, uf);
    }
    const float g = __bfloat162float(__float2bfloat16(warp_sum(ag)));
    const float u = __bfloat162float(__float2bfloat16(warp_sum(au)));
    if (lane == 0) mid_out[static_cast<size_t>(route) * sh_i + n] = __float2bfloat16(silu_f(g) * u);
}

// Fast GPU-native MTP Routed MoE FC2 (down) in block-32 FP8 -> writes weighted partial into exp_dn_part[route, 2560].
// Grid: (M * TOPK * (2560 / 4)), Block: 128 (4 warps = 4 hidden dims of expert e).
__global__ void __launch_bounds__(128) k_mtp_exp_fc2_fp8(
    const __nv_bfloat16* __restrict__ mid_in,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const uint8_t* __restrict__ dn_q, const __half* __restrict__ dn_sc,
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
    const __half* ds = dn_sc + row * nb;
    const uint4* am = reinterpret_cast<const uint4*>(mid_in + static_cast<size_t>(route) * sh_i);

    float acc = 0.0f;
    for (int i = lane; i < k16; i += 32) {
        float2 wf[8];
        fp8x16_unpack(__ldg(dq + i), __half2float(__ldg(ds + (i >> 1))), wf);
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

// Fast MTP Layer-48 MoE forward (uses FP8 GPU-native routed experts when available, zero host sync!).
inline void mtp_moe_forward(cublasHandle_t blas, const MoeWeights& w, const Fp8Mat& fp8_gu, const Fp8Mat& fp8_dn,
                            MoeScratch& sc, MtpScratch& msc, const __nv_bfloat16* x, __nv_bfloat16* out_part,
                            int M, cudaStream_t st) {
    // 1. Router + shared expert (same fast path as trunk MoE!)
    if (M <= 8) {
        const int rgrid = (geom::NEXP + 1 + 3) / 4;
        const int sgrid = (w.sh_i + 3) / 4;
        if (M == 1) {
            k_qw38_router_and_shgate<1><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) k_fp8_sh_gate_up_silu<1><<<sgrid, 128, 0, st>>>(x, w.fp8_sh_gate.q, w.fp8_sh_gate.sc, w.fp8_sh_up.q, w.fp8_sh_up.sc, sc.sh_gate_buf, M, w.sh_i);
            else k_qw38_sh_gate_up_silu<1><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else if (M <= 4) {
            k_qw38_router_and_shgate<4><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) k_fp8_sh_gate_up_silu<4><<<sgrid, 128, 0, st>>>(x, w.fp8_sh_gate.q, w.fp8_sh_gate.sc, w.fp8_sh_up.q, w.fp8_sh_up.sc, sc.sh_gate_buf, M, w.sh_i);
            else k_qw38_sh_gate_up_silu<4><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else {
            k_qw38_router_and_shgate<8><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            if (w.fp8_sh_gate.q) k_fp8_sh_gate_up_silu<8><<<sgrid, 128, 0, st>>>(x, w.fp8_sh_gate.q, w.fp8_sh_gate.sc, w.fp8_sh_up.q, w.fp8_sh_up.sc, sc.sh_gate_buf, M, w.sh_i);
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

    // 2. Routed experts (GPU-native, zero host sync) + deterministic combine!
    const int n_routes = M * geom::TOPK;
    const int fc1_grid = n_routes * ((w.sh_i + 3) / 4);
    const int fc2_grid = n_routes * (geom::HID / 4);
    if (fp8_gu.q && fp8_dn.q) {
        k_mtp_exp_fc1_fp8<<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, fp8_gu.q, fp8_gu.sc, msc.exp_mid, M, w.sh_i);
        k_mtp_exp_fc2_fp8<<<fc2_grid, 128, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, fp8_dn.q, fp8_dn.sc, msc.exp_dn_part, M, w.sh_i);
    } else {
        k_mtp_exp_fc1_bf16<<<fc1_grid, 128, 0, st>>>(x, sc.topk_idx, w.mtp_gate_up, msc.exp_mid, M, w.sh_i);
        k_mtp_exp_fc2_bf16<<<fc2_grid, 128, 0, st>>>(msc.exp_mid, sc.topk_idx, sc.topk_w, w.mtp_down, msc.exp_dn_part, M, w.sh_i);
    }
    k_qw38_moe_combine_det<<<M, 256, 0, st>>>(msc.exp_dn_part, sc.sh_out, sc.sh_scalar_gate, out_part, M);
}

} // namespace qw38
