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
    __nv_bfloat16* conv_hist = nullptr; // [slots, 9, 10240] BF16 (oldest..newest: [t-9..t-1])
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
                                           __nv_bfloat16* __restrict__ hyper_in,
                                           int slot, int row0, int pos0, int len, bool update_state) {
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
    }

    if (update_state) {
        #pragma unroll
        for (int k = 0; k < geom::PLE_NHIST; ++k) {
            hbase[static_cast<size_t>(k) * geom::HC_D + c] = __float2bfloat16(h[k]);
        }
    }
}

// Run Layer-1 PLE in place on `hyper_in` [M, 10240] BF16, assuming `sc.emb` [M, 2560] BF16 is already populated.
inline void ple_forward(cublasHandle_t blas, const PleWeights& w, PleState& st_buf, PleScratch& sc,
                        __nv_bfloat16* hyper_in, const StepSeg* segs, int nsegs, int M,
                        bool update_state, cudaStream_t st) {
    cublasSetStream(blas, st);
    gemm_bf16_nt(blas, M, geom::HC_D, geom::PLE_DIM, sc.emb, w.key_proj, sc.key);
    gemm_bf16_nt(blas, M, geom::HID, geom::PLE_DIM, sc.emb, w.value_proj, sc.val);

    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(sc.key, w.norm_key, sc.key, M, geom::HC, geom::RMS_EPS);
    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(hyper_in, w.norm_query, sc.query, M, geom::HC, geom::RMS_EPS);
    k_ple_gate<<<M * geom::HC, 256, 0, st>>>(sc.key, sc.query, sc.val, sc.gv, M);
    k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, st>>>(sc.gv, w.norm_conv, sc.gv_norm, M, geom::HC, geom::RMS_EPS);

    for (int s = 0; s < nsegs; ++s) {
        if (segs[s].len <= 0) continue;
        k_ple_dilated_conv_add_seg<<<(geom::HC_D + 255) / 256, 256, 0, st>>>(
            sc.gv, sc.gv_norm, w.conv1d, st_buf.conv_hist, hyper_in,
            segs[s].slot, segs[s].row0, segs[s].pos0, segs[s].len, update_state);
    }
}

} // namespace qw38
