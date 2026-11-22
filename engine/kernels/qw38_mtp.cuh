// Qwen3.8-Flash-Next 1-layer Multi-Token Prediction (MTP, pack layer L048) helpers.
// Verified tensor layout from model.safetensors.index.json:
//   mtp.pre_fc_norm_embedding.weight [2560]     (1 + w RMSNorm on next token embedding)
//   mtp.pre_fc_norm_hidden.weight    [10240]    (1 + w grouped RMSNorm on 4-stream trunk hidden state)
//   mtp.fc_embedding.weight          [2560, 2560]
//   mtp.fc_hidden.weight             [2560, 2560]  (Wait: check if input is mixed 2560 or mean of 4 streams!
//                                                   We verify this against checkpoint / MTP forward before enabling MTP!)
//   mtp.layers.0.*                   (1 full-attention + QSA + 512 BF16 MoE + gated shared + 4-stream HC layer)
//   mtp.hyper_connection_mixer.*     (final 4-stream -> 2560 mixer before shared lm_head)
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "qw38_hc.cuh"
#include "qw38_moe.cuh"
#include "../attn/qw38_attn.cuh"

namespace qw38 {

struct MtpWeights {
    const __nv_bfloat16* pre_fc_norm_emb = nullptr;  // [2560]
    const __nv_bfloat16* pre_fc_norm_hid = nullptr;  // [10240]
    const __nv_bfloat16* fc_emb = nullptr;           // [2560, 2560]
    const __nv_bfloat16* fc_hid = nullptr;           // [2560, 2560]
    HcWeights attn_hc;
    HcWeights mlp_hc;
    HcWeights final_hc;
    FullAttnWeights attn;
    MoeWeights moe;
};

// Fuse normalized embedding and normalized 4-stream hidden state into 4-stream MTP input:
//   e_n = rmsnorm_1pw(embed[tok])
//   h_n = mean_s(rmsnorm_1pw_grouped(hyper_hid))
//   fused = fc_emb(e_n) + fc_hid(h_n), replicated across 4 streams -> mtp_hyper [M, 10240].
__global__ void k_mtp_mean4_bf16(const __nv_bfloat16* __restrict__ h4, __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* src = h4 + static_cast<size_t>(m) * geom::HC_D;
    __nv_bfloat16* dst = out + static_cast<size_t>(m) * geom::HID;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) acc += __bfloat162float(src[s * geom::HID + d]);
        dst[d] = __float2bfloat16(acc * 0.25f);
    }
}

__global__ void k_mtp_add_rep4(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                               __nv_bfloat16* __restrict__ out4, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* ar = a + static_cast<size_t>(m) * geom::HID;
    const __nv_bfloat16* br = b + static_cast<size_t>(m) * geom::HID;
    __nv_bfloat16* dst = out4 + static_cast<size_t>(m) * geom::HC_D;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        const __nv_bfloat16 v = __float2bfloat16(__bfloat162float(ar[d]) + __bfloat162float(br[d]));
        #pragma unroll
        for (int s = 0; s < geom::HC; ++s) dst[s * geom::HID + d] = v;
    }
}

__global__ void k_gather_embed(const int* __restrict__ tok, const __nv_bfloat16* __restrict__ embed,
                               __nv_bfloat16* __restrict__ out, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* src = embed + static_cast<size_t>(tok[m]) * geom::HID;
    __nv_bfloat16* dst = out + static_cast<size_t>(m) * geom::HID;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) dst[d] = src[d];
}

} // namespace qw38
