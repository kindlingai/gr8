// Qwen3.8-Flash-Next Gated DeltaNet linear attention (Qwen4ExpTextGatedDeltaNet, 36 layers l % 4 != 3).
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py & config.json:
//   - 16 K/Q heads, 48 V heads (3 V per K), d_k = d_v = 128, causal conv1d kernel = 4 + SiLU.
//   - output_gate_type = "sigmoid": Qwen4ExpTextRMSNormGated uses self.weight directly (NOT 1 + w)
//     and multiplies by sigmoid(z.float()), then casts back to BF16.
//   - Decay: g = -exp(A_log.float()) * softplus(a.float() + dt_bias.float()), beta = sigmoid(b.float()).
//   - Recurrent update in FP32 state S[nh_v, 128, 128]:
//       q = l2norm(q, 1e-6) * rsqrt(128),  k = l2norm(k, 1e-6)
//       S *= exp(g)
//       kv_mem = sum_k S[k, v] * k[k]
//       delta = (v - kv_mem) * beta
//       S[k, v] += k[k] * delta[v]
//       out[v] = sum_k S[k, v] * q[k]
//   - Supports multi-session batched steps (Rows) and staged rollback/commit for MTP verification.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "qw38_hc.cuh"

namespace qw38 {

// Compact description of a batched step's rows grouped by session slot (same spirit as ink2::Rows).
struct StepSeg {
    int slot = 0;    // session slot index in 0..max_slots-1
    int row0 = 0;    // starting row in the flat [0..M) step tensors
    int pos0 = 0;    // sequence position of row0 (0 = fresh reset)
    int len = 0;     // number of consecutive tokens for this session in this step
};

struct GdnWeights {
    const __nv_bfloat16* in_proj_qkv = nullptr; // [qkv_local, 2560]
    const __nv_bfloat16* conv1d = nullptr;      // [qkv_local, 4]
    const __nv_bfloat16* in_proj_z = nullptr;   // [nh_v * 128, 2560]
    const __nv_bfloat16* in_proj_a = nullptr;   // [nh_v, 2560]
    const __nv_bfloat16* in_proj_b = nullptr;   // [nh_v, 2560]
    const __nv_bfloat16* A_log = nullptr;       // [nh_v]
    const __nv_bfloat16* dt_bias = nullptr;     // [nh_v]
    const __nv_bfloat16* norm = nullptr;        // [128] (direct scale, NOT 1+w)
    const __nv_bfloat16* out_proj = nullptr;    // [2560, nh_v * 128]
};

// Per-layer recurrent & conv state across all session slots.
// For MTP rollback without modifying committed state until commit():
//   - S / conv hold the committed state of each slot.
//   - S_work / conv_work hold working copies during multi-row verify steps when speculative = true,
//     or when speculative = false (MTP=0 / prefill) kernels update S / conv directly in place.
struct GdnLayerState {
    float* S = nullptr;            // [slots, nh_v, 128, 128] FP32
    __nv_bfloat16* conv = nullptr; // [slots, 3, qkv_local] BF16 (ring ordered oldest..newest: [t-3, t-2, t-1])
};

struct GdnScratch {
    __nv_bfloat16* qkv_raw = nullptr;   // [max_M, qkv_local] BF16
    __nv_bfloat16* qkv_conv = nullptr;  // [max_M, qkv_local] BF16
    __nv_bfloat16* z = nullptr;         // [max_M, nh_v * 128] BF16
    __nv_bfloat16* a = nullptr;         // [max_M, nh_v] BF16
    __nv_bfloat16* b = nullptr;         // [max_M, nh_v] BF16
    __nv_bfloat16* core = nullptr;      // [max_M, nh_v * 128] BF16
};

// Causal depthwise conv1d (kernel=4, dilation=1) + SiLU over one segment (slot, row0, pos0, len).
// Updates conv_hist[slot] in place with the last 3 raw inputs when update_state == true.
// When pos0 == 0, history is treated as zeros (exact match to HF causal_conv1d_fn padding=3).
__global__ void k_gdn_conv1d_seg(const __nv_bfloat16* __restrict__ qkv_raw,
                                 const __nv_bfloat16* __restrict__ w,
                                 __nv_bfloat16* __restrict__ conv_hist,
                                 __nv_bfloat16* __restrict__ qkv_out,
                                 int qkv_dim, int slot, int row0, int pos0, int len, bool update_state) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) return;
    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * 3 * qkv_dim;
    float h0 = (pos0 > 0) ? __bfloat162float(hbase[0 * qkv_dim + c]) : 0.0f;
    float h1 = (pos0 > 0) ? __bfloat162float(hbase[1 * qkv_dim + c]) : 0.0f;
    float h2 = (pos0 > 0) ? __bfloat162float(hbase[2 * qkv_dim + c]) : 0.0f;

    for (int t = 0; t < len; ++t) {
        const size_t idx = static_cast<size_t>(row0 + t) * qkv_dim + c;
        const float x = __bfloat162float(qkv_raw[idx]);
        const float y = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        qkv_out[idx] = __float2bfloat16(silu_f(y));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    if (update_state) {
        hbase[0 * qkv_dim + c] = __float2bfloat16(h0);
        hbase[1 * qkv_dim + c] = __float2bfloat16(h1);
        hbase[2 * qkv_dim + c] = __float2bfloat16(h2);
    }
}

__device__ __forceinline__ float softplus_f(float x) {
    // Stable softplus matching F.softplus(x, beta=1, threshold=20)
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

// Recurrent Gated Delta Rule over one segment (slot, row0, pos0, len).
// Grid: (nh_v) blocks, Block: 128 threads (one thread per v column in 0..127!).
// Each thread holds its entire column S[0..127, v] in registers for zero HBM traffic between tokens!
// Note: 128 floats/thread is fine on sm_121a (255 regs/thread); we use 2-way v split (256 threads,
// or 128 threads with smem for q/k and 128 floats split across 2 threads -> 64 floats/thread!) so
// occupancy stays high and register spilling is ZERO.
// Block = 256 threads: tid = v_col (0..127) + 128 * half_k (0 or 1, handling k in [0..63] or [64..127]).
__global__ void k_gdn_recurrent_seg(const __nv_bfloat16* __restrict__ qkv_conv,
                                    const __nv_bfloat16* __restrict__ a_in,
                                    const __nv_bfloat16* __restrict__ b_in,
                                    const __nv_bfloat16* __restrict__ A_log,
                                    const __nv_bfloat16* __restrict__ dt_bias,
                                    float* __restrict__ S_all,
                                    __nv_bfloat16* __restrict__ core_out,
                                    int nh_k, int nh_v, int k0_global, int v0_global,
                                    int slot, int row0, int pos0, int len, bool update_state) {
    const int lv = blockIdx.x;
    if (lv >= nh_v) return;
    const int gv = v0_global + lv;
    const int lk = gv / geom::GDN_V_PER_K - k0_global;
    const int qkv_dim = (2 * nh_k + nh_v) * geom::GDN_DK;

    const int tid = threadIdx.x;       // 0..255
    const int v = tid & 127;           // 0..127
    const int kh = tid >> 7;           // 0 => k in [0..63], 1 => k in [64..127]
    const int k_base = kh * 64;

    float* S_head = S_all + (static_cast<size_t>(slot) * nh_v + lv) * (128 * 128);
    float s_col[64];
    if (pos0 == 0) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = S_head[(k_base + i) * 128 + v];
    }

    __shared__ float sq[128];
    __shared__ float sk[128];
    __shared__ float sp[128]; // partial reduction between kh=0 and kh=1
    __shared__ float s_meta[2]; // [0]=exp(g), [1]=beta

    const float neg_A = -expf(__bfloat162float(A_log[lv]));
    const float dtb = __bfloat162float(dt_bias[lv]);
    constexpr float inv_sqrt_dk = 0.08838834764831845f; // 1 / sqrt(128)

    for (int t = 0; t < len; ++t) {
        const int r = row0 + t;
        const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(r) * qkv_dim;
        const __nv_bfloat16* q_ptr = qkv + static_cast<size_t>(lk) * 128;
        const __nv_bfloat16* k_ptr = qkv + static_cast<size_t>(nh_k + lk) * 128;
        const __nv_bfloat16* v_ptr = qkv + static_cast<size_t>(2 * nh_k + lv) * 128;

        // Load q (kh=0) and k (kh=1) into shared memory and compute L2 norms
        if (kh == 0) {
            sq[v] = __bfloat162float(q_ptr[v]);
        } else {
            sk[v] = __bfloat162float(k_ptr[v]);
        }
        if (tid == 0) {
            const float av = __bfloat162float(a_in[static_cast<size_t>(r) * nh_v + lv]);
            const float bv = __bfloat162float(b_in[static_cast<size_t>(r) * nh_v + lv]);
            s_meta[0] = expf(neg_A * softplus_f(av + dtb));
            s_meta[1] = sigmoid_f(bv);
        }
        __syncthreads();

        // L2-normalize sq (first 4 warps) and sk (second 4 warps)
        if (kh == 0) {
            float qv = sq[v];
            float q2 = warp_sum(qv * qv);
            __shared__ float qwarp[4];
            if ((v & 31) == 0) qwarp[v >> 5] = q2;
            __syncthreads(); // all threads participate below
            const float qsum = qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3];
            sq[v] = qv * rsqrtf(qsum + 1e-6f) * inv_sqrt_dk;
        } else {
            float kv = sk[v];
            float k2 = warp_sum(kv * kv);
            __shared__ float kwarp[4];
            if ((v & 31) == 0) kwarp[v >> 5] = k2;
            __syncthreads();
            const float ksum = kwarp[0] + kwarp[1] + kwarp[2] + kwarp[3];
            sk[v] = kv * rsqrtf(ksum + 1e-6f);
        }
        __syncthreads();

        const float decay = s_meta[0];
        const float beta = s_meta[1];

        // 1. Decay state and compute partial kv_mem = sum_k S[k, v] * k[k]
        float kv_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            s_col[i] *= decay;
            kv_part += s_col[i] * sk[k_base + i];
        }
        if (kh == 1) sp[v] = kv_part;
        __syncthreads();
        if (kh == 0) {
            const float v_val = __bfloat162float(v_ptr[v]);
            sp[v] = (v_val - (kv_part + sp[v])) * beta;
        }
        __syncthreads();
        const float delta_v = sp[v];

        // 2. Rank-1 update S[k, v] += k[k] * delta[v] and compute partial out[v] = sum_k S[k, v] * q[k]
        float out_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            s_col[i] += sk[k_base + i] * delta_v;
            out_part += s_col[i] * sq[k_base + i];
        }
        if (kh == 1) sp[v] = out_part;
        __syncthreads();
        if (kh == 0) {
            core_out[(static_cast<size_t>(r) * nh_v + lv) * 128 + v] = __float2bfloat16(out_part + sp[v]);
        }
        __syncthreads();
    }

    if (update_state) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) S_head[(k_base + i) * 128 + v] = s_col[i];
    }
}

// Qwen4ExpTextRMSNormGated (output_gate_type = "sigmoid", weight applied directly, NOT 1+w):
//   hs = bf16(hs.float() * rsqrt(mean(hs^2) + eps)) * weight
//   out = bf16(hs * sigmoid(z.float()))
// Grid: (M * nh_v), Block: 128 threads (1 head of 128 dims per CTA).
__global__ void k_gdn_norm_gate(__nv_bfloat16* __restrict__ core, const __nv_bfloat16* __restrict__ z,
                                const __nv_bfloat16* __restrict__ w, int M, int nh_v, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_v) return;
    const int d = threadIdx.x; // 0..127
    __nv_bfloat16* hr = core + static_cast<size_t>(idx) * 128;
    const __nv_bfloat16* zr = z + static_cast<size_t>(idx) * 128;

    const float hv = __bfloat162float(hr[d]);
    float ss = warp_sum(hv * hv);
    __shared__ float sm[4];
    if ((d & 31) == 0) sm[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / 128.0f + eps);
    const float normed_bf = __bfloat162float(__float2bfloat16(hv * inv));
    const float scaled = __bfloat162float(__float2bfloat16(normed_bf * __bfloat162float(w[d])));
    const float gate = sigmoid_f(__bfloat162float(zr[d]));
    hr[d] = __float2bfloat16(scaled * gate);
}

// Execute one GDN layer on rank `rank` of `nr` over `M` rows divided into `segs`.
// Writes this rank's partial output [M, 2560] BF16 into `out_part`.
inline void gdn_forward(cublasHandle_t blas, const GdnWeights& w, GdnLayerState& st_buf, GdnScratch& sc,
                        const __nv_bfloat16* x, __nv_bfloat16* out_part,
                        const StepSeg* segs, int nsegs, int M, int nr, int rank,
                        bool update_state, cudaStream_t st) {
    const int nh_v = geom::gdn_vh(nr);
    const int nh_k = geom::gdn_kh(nr, rank);
    const int k0 = geom::gdn_k0(nr, rank);
    const int v0 = geom::gdn_v0(nr, rank);
    const int qkv_dim = geom::gdn_qkv_local(nr, rank);

    cublasSetStream(blas, st);
    gemm_bf16_nt(blas, M, qkv_dim, geom::HID, x, w.in_proj_qkv, sc.qkv_raw);
    gemm_bf16_nt(blas, M, nh_v * geom::GDN_DV, geom::HID, x, w.in_proj_z, sc.z);
    gemm_bf16_nt(blas, M, nh_v, geom::HID, x, w.in_proj_a, sc.a);
    gemm_bf16_nt(blas, M, nh_v, geom::HID, x, w.in_proj_b, sc.b);

    for (int s = 0; s < nsegs; ++s) {
        const StepSeg& g = segs[s];
        if (g.len <= 0) continue;
        k_gdn_conv1d_seg<<<(qkv_dim + 255) / 256, 256, 0, st>>>(
            sc.qkv_raw, w.conv1d, st_buf.conv, sc.qkv_conv, qkv_dim, g.slot, g.row0, g.pos0, g.len, update_state);
        k_gdn_recurrent_seg<<<nh_v, 256, 0, st>>>(
            sc.qkv_conv, sc.a, sc.b, w.A_log, w.dt_bias, st_buf.S, sc.core,
            nh_k, nh_v, k0, v0, g.slot, g.row0, g.pos0, g.len, update_state);
    }

    k_gdn_norm_gate<<<M * nh_v, 128, 0, st>>>(sc.core, sc.z, w.norm, M, nh_v, geom::RMS_EPS);
    gemm_bf16_nt(blas, M, geom::HID, nh_v * geom::GDN_DV, sc.core, w.out_proj, out_part);
}

} // namespace qw38
