// Qwen3.8-Flash-Next Layer-1 PLE GPU kernels (Qwen4ExpTextPLELayer).
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py:
//   key_normed   = norm_key(key_proj(ple_emb)).view(4, 2560)      (Qwen4ExpTextRMSNorm, 1+w)
//   value        = value_proj(ple_emb)                             [M, 2560]
//   query_normed = norm_query(hyper_in).view(4, 2560)              (Qwen4ExpTextRMSNorm, 1+w)
//   gate         = sum_d(key_normed * query_normed) / sqrt(2560)   [M, 4, 1]
//   gate         = sqrt(max(|gate|, 1e-6)) * sign(gate)
//   gated_value  = sigmoid(gate) * value.unsqueeze(-2)             [M, 4, 2560]
//   gv_normed    = norm_conv(gated_value.flatten(-2))              (Qwen4ExpTextRMSNorm, 1+w)
//   conv_out     = silu(causal_dilated_conv1d(gv_normed, k=4, dilation=3))  (history len = 9!)
//   hyper_in    += gated_value + conv_out
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "../kernels/qw38_hc.cuh"
#include "../kernels/qw38_gdn.cuh"

namespace qw38 {

struct PleWeights {
    const __nv_bfloat16* key_proj = nullptr;   // [10240, 2560]
    const __nv_bfloat16* value_proj = nullptr; // [2560, 2560]
    const __nv_bfloat16* conv1d = nullptr;     // [10240, 4]
    const __nv_bfloat16* norm_query = nullptr; // [10240] (1 + w)
    const __nv_bfloat16* norm_key = nullptr;   // [10240] (1 + w)
    const __nv_bfloat16* norm_conv = nullptr;  // [10240] (1 + w)
};

struct PleState {
    __nv_bfloat16* conv_hist = nullptr;      // [slots, 9, 10240] BF16 (oldest..newest: [t-9..t-1])
    __nv_bfloat16* conv_hist_snap = nullptr; // optional [slots, max_K, 9, 10240] BF16 (for MTP rollback)
};

struct PleScratch {
    __nv_bfloat16* emb = nullptr;       // [max_M, 2560] BF16 (copied from host PleStore)
    __nv_bfloat16* key = nullptr;       // [max_M, 10240] BF16
    __nv_bfloat16* query = nullptr;     // [max_M, 10240] BF16
    __nv_bfloat16* val = nullptr;       // [max_M, 2560] BF16
    __nv_bfloat16* gv = nullptr;        // [max_M, 10240] BF16 (gated_value)
    __nv_bfloat16* gv_norm = nullptr;   // [max_M, 10240] BF16 (norm_conv(gv))
};

// Compute signed-sqrt dot gate between key_normed[m, s, :] and query_normed[m, s, :],
// then gated_value[m, s, d] = bf16(sigmoid(gate) * value[m, d]).
// Grid: (M * 4), Block: 256 threads.
__global__ void k_ple_gate(const __nv_bfloat16* __restrict__ key_n,
                           const __nv_bfloat16* __restrict__ query_n,
                           const __nv_bfloat16* __restrict__ val,
                           __nv_bfloat16* __restrict__ gv_out,
                           int M) {
    const int ms = blockIdx.x;
    const int m = ms / geom::HC;
    if (m >= M) return;
    const __nv_bfloat16* kr = key_n + static_cast<size_t>(ms) * geom::HID;
    const __nv_bfloat16* qr = query_n + static_cast<size_t>(ms) * geom::HID;
    const __nv_bfloat16* vr = val + static_cast<size_t>(m) * geom::HID;
    __nv_bfloat16* dst = gv_out + static_cast<size_t>(ms) * geom::HID;

    float dot = 0.0f;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        dot += __bfloat162float(kr[d]) * __bfloat162float(qr[d]);
    }
    dot = warp_sum(dot);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? sm[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) {
            constexpr float inv_sqrt_h = 0.01976423537605237f; // 1 / sqrt(2560)
            const float g = tot * inv_sqrt_h;
            const float ag = fmaxf(fabsf(g), 1e-6f);
            const float sgn = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);
            const float sg = sqrtf(ag) * sgn;
            sm[0] = sigmoid_f(sg);
        }
    }
    __syncthreads();
    const float gw = sm[0];
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        dst[d] = __float2bfloat16(gw * __bfloat162float(vr[d]));
    }
}

// Causal dilated-3 depthwise conv1d (kernel=4, dilation=3 => taps at t-9, t-6, t-3, t) + SiLU,
// then adds gated_value + conv_out directly into hyper_in [M, 10240]!
// Updates conv_hist[slot, 0..8, c] when update_state == true.
__global__ void k_ple_dilated_conv_add_seg(const __nv_bfloat16* __restrict__ gv,
                                           const __nv_bfloat16* __restrict__ gv_norm,
                                           const __nv_bfloat16* __restrict__ w,
                                           __nv_bfloat16* __restrict__ conv_hist,
                                           __nv_bfloat16* __restrict__ conv_snap,
                                           __nv_bfloat16* __restrict__ hyper_in,
                                           int slot, int row0, int pos0, int len, int snap_k, bool update_state) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= geom::HC_D) return;
    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * geom::PLE_NHIST * geom::HC_D;
    float h[geom::PLE_NHIST]; // h[0] = t-9, ..., h[8] = t-1
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) {
        h[k] = (pos0 > 0) ? __bfloat162float(hbase[static_cast<size_t>(k) * geom::HC_D + c]) : 0.0f;
    }

    const int snap_stride = snap_k + 1;
    if (conv_snap && snap_k > 0) {
        __nv_bfloat16* sb0 = conv_snap + (static_cast<size_t>(slot) * snap_stride) * geom::PLE_NHIST * geom::HC_D;
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST; ++k) sb0[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
    }

    for (int t = 0; t < len; ++t) {
        const size_t idx = static_cast<size_t>(row0 + t) * geom::HC_D + c;
        const float x = __bfloat162float(gv_norm[idx]);
        // Dilation 3: taps at [t-9]=h[0], [t-6]=h[3], [t-3]=h[6], [t]=x
        const float conv = h[0] * w0 + h[3] * w1 + h[6] * w2 + x * w3;
        const float conv_act = __bfloat162float(__float2bfloat16(silu_f(conv)));
        const float ple_out = __bfloat162float(gv[idx]) + conv_act;
        hyper_in[idx] = __float2bfloat16(__bfloat162float(hyper_in[idx]) + ple_out);

        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST - 1; ++k) h[k] = h[k + 1];
        h[geom::PLE_NHIST - 1] = x;

        if (conv_snap && t < len - 1 && t < snap_k) {
            __nv_bfloat16* sb = conv_snap + (static_cast<size_t>(slot) * snap_stride + (t + 1)) * geom::PLE_NHIST * geom::HC_D;
            #pragma unroll
            for (int k = 0; k < geom::PLE_NHIST; ++k) sb[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
        }
    }

    if (update_state) {
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST; ++k) {
            hbase[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
        }
    }
}

// Batched single-token decode dilated-3 conv1d + residual add across all M rows in ONE graph-safe launch!
// Grid: ((M * HC_D + 255) / 256), Block: 256.
__global__ void k_ple_dilated_conv_add_dec_batch(const __nv_bfloat16* __restrict__ gv,
                                                 const __nv_bfloat16* __restrict__ gv_norm,
                                                 const __nv_bfloat16* __restrict__ w,
                                                 __nv_bfloat16* __restrict__ conv_hist,
                                                 __nv_bfloat16* __restrict__ hyper_in,
                                                 const int* __restrict__ d_slot,
                                                 const int* __restrict__ d_pos,
                                                 int M, bool update_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = M * geom::HC_D;
    if (idx >= tot) return;
    const int m = idx / geom::HC_D;
    const int c = idx % geom::HC_D;
    const int slot = d_slot[m];
    const int pos = d_pos[m];

    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * geom::PLE_NHIST * geom::HC_D;
    float h[geom::PLE_NHIST];
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) {
        h[k] = (pos > 0) ? __bfloat162float(hbase[static_cast<size_t>(k) * geom::HC_D + c]) : 0.0f;
    }

    const float x = __bfloat162float(gv_norm[idx]);
    const float conv = h[0] * w0 + h[3] * w1 + h[6] * w2 + x * w3;
    const float conv_act = __bfloat162float(__float2bfloat16(silu_f(conv)));
    const float ple_out = __bfloat162float(gv[idx]) + conv_act;
    hyper_in[idx] = __float2bfloat16(__bfloat162float(hyper_in[idx]) + ple_out);

    if (update_state) {
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST - 1; ++k) hbase[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k + 1]);
        hbase[static_cast<size_t>(geom::PLE_NHIST - 1) * geom::HC_D + c] = __float2bfloat16(x);
    }
}

// Graph-safe batched MTP verify dilated-3 conv1d + residual add across S sessions x V tokens/session!
__global__ void k_ple_dilated_conv_add_verify_batch(const __nv_bfloat16* __restrict__ gv,
                                                    const __nv_bfloat16* __restrict__ gv_norm,
                                                    const __nv_bfloat16* __restrict__ w,
                                                    __nv_bfloat16* __restrict__ conv_hist,
                                                    __nv_bfloat16* __restrict__ conv_snap,
                                                    __nv_bfloat16* __restrict__ hyper_in,
                                                    const int* __restrict__ d_slot,
                                                    const int* __restrict__ d_pos,
                                                    int S, int V, int snap_k, bool update_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * geom::HC_D) return;
    const int s = idx / geom::HC_D;
    const int c = idx % geom::HC_D;
    const int row0 = s * V;
    const int slot = d_slot[row0];
    const int pos0 = d_pos[row0];

    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * geom::PLE_NHIST * geom::HC_D;
    float h[geom::PLE_NHIST];
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) {
        h[k] = (pos0 > 0) ? __bfloat162float(hbase[static_cast<size_t>(k) * geom::HC_D + c]) : 0.0f;
    }

    const int snap_stride = snap_k + 1;
    if (conv_snap && snap_k > 0) {
        __nv_bfloat16* sb0 = conv_snap + (static_cast<size_t>(slot) * snap_stride) * geom::PLE_NHIST * geom::HC_D;
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST; ++k) sb0[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
    }

    for (int t = 0; t < V; ++t) {
        const size_t ri = static_cast<size_t>(row0 + t) * geom::HC_D + c;
        const float x = __bfloat162float(gv_norm[ri]);
        const float conv = h[0] * w0 + h[3] * w1 + h[6] * w2 + x * w3;
        const float conv_act = __bfloat162float(__float2bfloat16(silu_f(conv)));
        const float ple_out = __bfloat162float(gv[ri]) + conv_act;
        hyper_in[ri] = __float2bfloat16(__bfloat162float(hyper_in[ri]) + ple_out);

        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST - 1; ++k) h[k] = h[k + 1];
        h[geom::PLE_NHIST - 1] = x;

        if (conv_snap && t < V - 1 && t < snap_k) {
            __nv_bfloat16* sb = conv_snap + (static_cast<size_t>(slot) * snap_stride + (t + 1)) * geom::PLE_NHIST * geom::HC_D;
            #pragma unroll
            for (int k = 0; k < geom::PLE_NHIST; ++k) sb[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
        }
    }

    if (update_state) {
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST; ++k) {
            hbase[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
        }
    }
}

// Fused norm_key + norm_query + signed-sqrt gate + norm_conv in ONE 320-thread kernel per (m, s).
// Eliminates global write/read of key_norm and query_norm; directly outputs gv and gv_norm.
__global__ void __launch_bounds__(320) k_ple_fused_norm_gate_norm(
    const __nv_bfloat16* __restrict__ key_raw,
    const __nv_bfloat16* __restrict__ hyper_in,
    const __nv_bfloat16* __restrict__ val,
    const __nv_bfloat16* __restrict__ w_nk,
    const __nv_bfloat16* __restrict__ w_nq,
    const __nv_bfloat16* __restrict__ w_nc,
    __nv_bfloat16* __restrict__ gv_out,
    __nv_bfloat16* __restrict__ gv_norm_out,
    int M, float eps) {
    const int ms = blockIdx.x;
    const int m = ms >> 2, s = ms & 3;
    if (m >= M) return;
    const int tid = threadIdx.x; // 0..319 -> 1 uint4 = 8 BF16s exact (HID = 2560)
    const int lane = tid & 31, warp = tid >> 5;

    const uint4 kv = reinterpret_cast<const uint4*>(key_raw + static_cast<size_t>(ms) * geom::HID)[tid];
    const uint4 qv = reinterpret_cast<const uint4*>(hyper_in + static_cast<size_t>(ms) * geom::HID)[tid];
    const uint4 wkv = __ldg(reinterpret_cast<const uint4*>(w_nk + static_cast<size_t>(s) * geom::HID) + tid);
    const uint4 wqv = __ldg(reinterpret_cast<const uint4*>(w_nq + static_cast<size_t>(s) * geom::HID) + tid);

    float kf[8], qf[8], wkf[8], wqf[8];
    float ssk = 0.0f, ssq = 0.0f;
    const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kv);
    const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qv);
    const __nv_bfloat162* wkp = reinterpret_cast<const __nv_bfloat162*>(&wkv);
    const __nv_bfloat162* wqp = reinterpret_cast<const __nv_bfloat162*>(&wqv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fk = __bfloat1622float2(kp[i]);
        const float2 fq = __bfloat1622float2(qp[i]);
        const float2 fwk = __bfloat1622float2(wkp[i]);
        const float2 fwq = __bfloat1622float2(wqp[i]);
        kf[2 * i] = fk.x; kf[2 * i + 1] = fk.y;
        qf[2 * i] = fq.x; qf[2 * i + 1] = fq.y;
        wkf[2 * i] = 1.0f + fwk.x; wkf[2 * i + 1] = 1.0f + fwk.y;
        wqf[2 * i] = 1.0f + fwq.x; wqf[2 * i + 1] = 1.0f + fwq.y;
        ssk = fmaf(fk.x, fk.x, fmaf(fk.y, fk.y, ssk));
        ssq = fmaf(fq.x, fq.x, fmaf(fq.y, fq.y, ssq));
    }
    ssk = warp_sum(ssk);
    ssq = warp_sum(ssq);
    __shared__ float smk[10], smq[10];
    if (lane == 0) { smk[warp] = ssk; smq[warp] = ssq; }
    __syncthreads();
    if (warp == 0) {
        const float tk = warp_sum((lane < 10) ? smk[lane] : 0.0f);
        const float tq = warp_sum((lane < 10) ? smq[lane] : 0.0f);
        if (lane == 0) {
            smk[0] = rsqrtf(tk / float(geom::HID) + eps);
            smq[0] = rsqrtf(tq / float(geom::HID) + eps);
        }
    }
    __syncthreads();
    const float ik = smk[0], iq = smq[0];

    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float kn = __bfloat162float(__float2bfloat16(kf[i] * ik * wkf[i]));
        const float qn = __bfloat162float(__float2bfloat16(qf[i] * iq * wqf[i]));
        dot = fmaf(kn, qn, dot);
    }
    dot = warp_sum(dot);
    if (lane == 0) smk[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        const float tot = warp_sum((lane < 10) ? smk[lane] : 0.0f);
        if (lane == 0) {
            constexpr float inv_sqrt_h = 0.01976423537605237f; // 1 / sqrt(2560)
            const float g = tot * inv_sqrt_h;
            const float ag = fmaxf(fabsf(g), 1e-6f);
            const float sgn = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);
            smk[0] = sigmoid_f(sqrtf(ag) * sgn);
        }
    }
    __syncthreads();
    const float gw = smk[0];

    const uint4 vv = reinterpret_cast<const uint4*>(val + static_cast<size_t>(m) * geom::HID)[tid];
    const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vv);
    uint4 gvv;
    __nv_bfloat162* gvp = reinterpret_cast<__nv_bfloat162*>(&gvv);
    float gvf[8], ssg = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fv = __bfloat1622float2(vp[i]);
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(gw * fv.x, gw * fv.y);
        gvp[i] = b2;
        const float2 fb = __bfloat1622float2(b2);
        gvf[2 * i] = fb.x; gvf[2 * i + 1] = fb.y;
        ssg = fmaf(fb.x, fb.x, fmaf(fb.y, fb.y, ssg));
    }
    reinterpret_cast<uint4*>(gv_out + static_cast<size_t>(ms) * geom::HID)[tid] = gvv;

    ssg = warp_sum(ssg);
    if (lane == 0) smk[warp] = ssg;
    __syncthreads();
    if (warp == 0) {
        const float tg = warp_sum((lane < 10) ? smk[lane] : 0.0f);
        if (lane == 0) smk[0] = rsqrtf(tg / float(geom::HID) + eps);
    }
    __syncthreads();
    const float ig = smk[0];
    const uint4 wcv = __ldg(reinterpret_cast<const uint4*>(w_nc + static_cast<size_t>(s) * geom::HID) + tid);
    const __nv_bfloat162* wcp = reinterpret_cast<const __nv_bfloat162*>(&wcv);
    uint4 gnv;
    __nv_bfloat162* gnp = reinterpret_cast<__nv_bfloat162*>(&gnv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fwc = __bfloat1622float2(wcp[i]);
        gnp[i] = __floats2bfloat162_rn(gvf[2 * i] * ig * (1.0f + fwc.x),
                                       gvf[2 * i + 1] * ig * (1.0f + fwc.y));
    }
    reinterpret_cast<uint4*>(gv_norm_out + static_cast<size_t>(ms) * geom::HID)[tid] = gnv;
}

// 2D vectorized prefill dilated-3 conv1d + residual add (4 channels/thread uint2, 16 rows/CTA).
__global__ void __launch_bounds__(256) k_ple_dilated_conv_add_pf(
    const __nv_bfloat16* __restrict__ gv,
    const __nv_bfloat16* __restrict__ gv_norm,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ conv_hist,
    __nv_bfloat16* __restrict__ hyper_in,
    int slot, int row0, int pos0, int len) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= (geom::HC_D >> 2)) return;
    const int c = c4 << 2;
    const int t0 = blockIdx.y * 16;
    const int t1 = min(len, t0 + 16);

    float w0[4], w1[4], w2[4], w3[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint2 wv = *reinterpret_cast<const uint2*>(w + static_cast<size_t>(c + k) * 4);
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
        w0[k] = f0.x; w1[k] = f0.y; w2[k] = f1.x; w3[k] = f1.y;
    }
    const __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * geom::PLE_NHIST * geom::HC_D;
    auto load4 = [&](int t, float (&dst)[4]) {
        uint2 v;
        if (t >= 0) v = reinterpret_cast<const uint2*>(gv_norm + static_cast<size_t>(row0 + t) * geom::HC_D)[c4];
        else if (pos0 > 0) v = reinterpret_cast<const uint2*>(hbase + static_cast<size_t>(geom::PLE_NHIST + t) * geom::HC_D)[c4];
        else { dst[0] = dst[1] = dst[2] = dst[3] = 0.0f; return; }
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        dst[0] = f0.x; dst[1] = f0.y; dst[2] = f1.x; dst[3] = f1.y;
    };

    float h[geom::PLE_NHIST][4];
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) load4(t0 - geom::PLE_NHIST + k, h[k]);

    for (int t = t0; t < t1; ++t) {
        float x[4];
        load4(t, x);
        const size_t ri4 = static_cast<size_t>(row0 + t) * (geom::HC_D >> 2) + c4;
        const uint2 gvv = reinterpret_cast<const uint2*>(gv)[ri4];
        const uint2 hv  = reinterpret_cast<const uint2*>(hyper_in)[ri4];
        const float2 gv0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gvv.x));
        const float2 gv1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gvv.y));
        const float2 h0  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hv.x));
        const float2 h1  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hv.y));
        const float gva[4] = {gv0.x, gv0.y, gv1.x, gv1.y};
        const float ha[4]  = {h0.x, h0.y, h1.x, h1.y};
        float out[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float conv = h[0][k] * w0[k] + h[3][k] * w1[k] + h[6][k] * w2[k] + x[k] * w3[k];
            const float conv_act = __bfloat162float(__float2bfloat16(silu_f(conv)));
            out[k] = ha[k] + (gva[k] + conv_act);
            #pragma unroll
            for (int r = 0; r < geom::PLE_NHIST - 1; ++r) h[r][k] = h[r + 1][k];
            h[geom::PLE_NHIST - 1][k] = x[k];
        }
        uint2 ov;
        *reinterpret_cast<__nv_bfloat162*>(&ov.x) = __floats2bfloat162_rn(out[0], out[1]);
        *reinterpret_cast<__nv_bfloat162*>(&ov.y) = __floats2bfloat162_rn(out[2], out[3]);
        reinterpret_cast<uint2*>(hyper_in)[ri4] = ov;
    }
}

// Vectorized 9-slot PLE conv history tail update for prefill segments.
__global__ void __launch_bounds__(256) k_ple_conv_hist_pf(
    const __nv_bfloat16* __restrict__ gv_norm,
    __nv_bfloat16* __restrict__ conv_hist,
    int slot, int row0, int pos0, int len) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= (geom::HC_D >> 2)) return;
    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * geom::PLE_NHIST * geom::HC_D;
    uint2 vals[geom::PLE_NHIST];
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) {
        const int t = len - geom::PLE_NHIST + k;
        if (t >= 0) vals[k] = reinterpret_cast<const uint2*>(gv_norm + static_cast<size_t>(row0 + t) * geom::HC_D)[c4];
        else if (pos0 > 0) vals[k] = reinterpret_cast<const uint2*>(hbase + static_cast<size_t>(geom::PLE_NHIST + t) * geom::HC_D)[c4];
        else vals[k] = make_uint2(0, 0);
    }
    #pragma unroll
    for (int k = 0; k < geom::PLE_NHIST; ++k) {
        reinterpret_cast<uint2*>(hbase + static_cast<size_t>(k) * geom::HC_D)[c4] = vals[k];
    }
}

// Run Layer-1 PLE in place on `hyper_in` [M, 10240] BF16, assuming `sc.emb` [M, 2560] BF16 is already populated.
// Graph-safe when M <= 20 and d_slot/d_pos are provided for decode or MTP verify steps!
inline void ple_forward(cublasHandle_t blas, const PleWeights& w, PleState& st_buf, PleScratch& sc,
                        __nv_bfloat16* hyper_in, const StepSeg* segs, int nsegs, int M,
                        bool update_state, cudaStream_t st,
                        const int* d_slot = nullptr, const int* d_pos = nullptr, int snap_k = 0) {
    if (M <= 12) {
        launch_bf16_gemv_smallm<false>(sc.emb, w.key_proj, sc.key, M, geom::HC_D, geom::PLE_DIM, st);
        launch_bf16_gemv_smallm<false>(sc.emb, w.value_proj, sc.val, M, geom::HID, geom::PLE_DIM, st);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, geom::HC_D, geom::PLE_DIM, sc.emb, w.key_proj, sc.key);
        gemm_bf16_nt(blas, M, geom::HID, geom::PLE_DIM, sc.emb, w.value_proj, sc.val);
    }

    k_ple_fused_norm_gate_norm<<<M * geom::HC, 320, 0, st>>>(
        sc.key, hyper_in, sc.val, w.norm_key, w.norm_query, w.norm_conv, sc.gv, sc.gv_norm, M, geom::RMS_EPS);

    bool all_len1 = (d_slot != nullptr && d_pos != nullptr && nsegs == M);
    if (all_len1) {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len != 1 || segs[s].row0 != s) { all_len1 = false; break; }
        }
    }
    const int V = snap_k + 1;
    bool all_verify = (!all_len1 && snap_k > 0 && d_slot != nullptr && d_pos != nullptr && nsegs > 0 && nsegs * V == M);
    if (all_verify) {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len != V || segs[s].row0 != s * V) { all_verify = false; break; }
        }
    }
    if (all_len1) {
        k_ple_dilated_conv_add_dec_batch<<<(M * geom::HC_D + 255) / 256, 256, 0, st>>>(
            sc.gv, sc.gv_norm, w.conv1d, st_buf.conv_hist, hyper_in, d_slot, d_pos, M, update_state);
    } else if (all_verify) {
        k_ple_dilated_conv_add_verify_batch<<<(nsegs * geom::HC_D + 255) / 256, 256, 0, st>>>(
            sc.gv, sc.gv_norm, w.conv1d, st_buf.conv_hist, st_buf.conv_hist_snap, hyper_in,
            d_slot, d_pos, nsegs, V, snap_k, update_state);
    } else {
        for (int s = 0; s < nsegs; ++s) {
            if (segs[s].len <= 0) continue;
            if (segs[s].len >= 16 && snap_k == 0) {
                k_ple_dilated_conv_add_pf<<<dim3(((geom::HC_D >> 2) + 255) / 256, (segs[s].len + 15) / 16), 256, 0, st>>>(
                    sc.gv, sc.gv_norm, w.conv1d, st_buf.conv_hist, hyper_in,
                    segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                if (update_state) {
                    k_ple_conv_hist_pf<<<((geom::HC_D >> 2) + 255) / 256, 256, 0, st>>>(
                        sc.gv_norm, st_buf.conv_hist, segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len);
                }
            } else {
                k_ple_dilated_conv_add_seg<<<(geom::HC_D + 255) / 256, 256, 0, st>>>(
                    sc.gv, sc.gv_norm, w.conv1d, st_buf.conv_hist, snap_k > 0 ? st_buf.conv_hist_snap : nullptr, hyper_in,
                    segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len, snap_k, update_state);
            }
        }
    }
}

} // namespace qw38
