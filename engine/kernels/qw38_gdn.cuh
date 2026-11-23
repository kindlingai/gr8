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
//   - Fast decode path (M <= 8): FP8/BF16 coalesced uint4 GEMVs + fused in_proj_ab +
//     batched conv1d + fused recurrent+RMSNorm+sigmoid(z) kernel (1 launch for all M sessions!).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "qw38_hc.cuh"
#include "qw38_fp8.cuh"

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
    // Optional on-load block-32 FP8 decode weights (GR8_QW38_DENSE_FP8=1)
    Fp8Mat fp8_qkv;
    Fp8Mat fp8_z;
    Fp8Mat fp8_out;
};

struct GdnLayerState {
    float* S = nullptr;            // [slots, nh_v, 128, 128] FP32 (active committed state)
    float* S_snap = nullptr;       // optional [slots, max_K, nh_v, 128, 128] FP32 (for MTP=1..3 exact rollback!)
    __nv_bfloat16* conv = nullptr; // [slots, 3, qkv_local] BF16 (ring ordered oldest..newest: [t-3, t-2, t-1])
    __nv_bfloat16* conv_snap = nullptr; // optional [slots, max_K, 3, qkv_local] BF16
};

struct GdnScratch {
    __nv_bfloat16* qkv_raw = nullptr;   // [max_M, qkv_local] BF16
    __nv_bfloat16* qkv_conv = nullptr;  // [max_M, qkv_local] BF16
    __nv_bfloat16* z = nullptr;         // [max_M, nh_v * 128] BF16
    __nv_bfloat16* a = nullptr;         // [max_M, nh_v] BF16
    __nv_bfloat16* b = nullptr;         // [max_M, nh_v] BF16
    __nv_bfloat16* core = nullptr;      // [max_M, nh_v * 128] BF16
};

// Fused BF16 GEMV for in_proj_a [nh_v, 2560] + in_proj_b [nh_v, 2560] (M <= 8).
// 4 warps/CTA = 4 rows/CTA over [0 .. 2*nh_v).
template <int MAX_M>
__global__ void k_gdn_in_proj_ab(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ wa,
                                 const __nv_bfloat16* __restrict__ wb,
                                 __nv_bfloat16* __restrict__ out_a,
                                 __nv_bfloat16* __restrict__ out_b,
                                 int M, int nh_v) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + warp;
    if (idx >= 2 * nh_v) return;
    const bool is_b = (idx >= nh_v);
    const int n = is_b ? (idx - nh_v) : idx;
    const uint4* wrow = reinterpret_cast<const uint4*>((is_b ? wb : wa) + static_cast<size_t>(n) * geom::HID);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    constexpr int K8 = geom::HID >> 3;
    for (int i = lane; i < K8; i += 32) {
        const uint4 wv = __ldg(wrow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID)[i];
                acc[m] += dot_bf16x8(av, wv);
            }
        }
    }
    __nv_bfloat16* dst = is_b ? out_b : out_a;
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float tot = warp_sum(acc[m]);
            if (lane == 0) dst[static_cast<size_t>(m) * nh_v + n] = __float2bfloat16(tot);
        }
    }
}

// Batched single-token decode conv1d (kernel=4, SiLU) across all M decode rows in ONE launch!
// Grid: ((M * qkv_dim + 255) / 256), Block: 256.
__global__ void k_gdn_conv1d_dec_batch(const __nv_bfloat16* __restrict__ qkv_raw,
                                       const __nv_bfloat16* __restrict__ w,
                                       __nv_bfloat16* __restrict__ conv_hist,
                                       __nv_bfloat16* __restrict__ qkv_out,
                                       const int* __restrict__ d_slot,
                                       const int* __restrict__ d_pos,
                                       int M, int qkv_dim, bool update_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = M * qkv_dim;
    if (idx >= tot) return;
    const int m = idx / qkv_dim;
    const int c = idx % qkv_dim;
    const int slot = d_slot[m];
    const int pos = d_pos[m];

    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * 3 * qkv_dim;
    const float h0 = (pos > 0) ? __bfloat162float(hbase[0 * qkv_dim + c]) : 0.0f;
    const float h1 = (pos > 0) ? __bfloat162float(hbase[1 * qkv_dim + c]) : 0.0f;
    const float h2 = (pos > 0) ? __bfloat162float(hbase[2 * qkv_dim + c]) : 0.0f;

    const float x = __bfloat162float(qkv_raw[idx]);
    const float y = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
    qkv_out[idx] = __float2bfloat16(silu_f(y));
    if (update_state) {
        hbase[0 * qkv_dim + c] = __float2bfloat16(h1);
        hbase[1 * qkv_dim + c] = __float2bfloat16(h2);
        hbase[2 * qkv_dim + c] = __float2bfloat16(x);
    }
}

// Causal depthwise conv1d (kernel=4, dilation=1) + SiLU over one segment (slot, row0, pos0, len).
// When conv_snap != nullptr (MTP verify step), saves post-token-t conv ring state into conv_snap[slot, t] for t < len - 1!
__global__ void k_gdn_conv1d_seg(const __nv_bfloat16* __restrict__ qkv_raw,
                                 const __nv_bfloat16* __restrict__ w,
                                 __nv_bfloat16* __restrict__ conv_hist,
                                 __nv_bfloat16* __restrict__ conv_snap,
                                 __nv_bfloat16* __restrict__ qkv_out,
                                 int qkv_dim, int slot, int row0, int pos0, int len, int snap_stride_slots, bool update_state) {
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
        if (conv_snap && t < len - 1 && t < snap_stride_slots) {
            __nv_bfloat16* sb = conv_snap + (static_cast<size_t>(slot) * snap_stride_slots + t) * 3 * qkv_dim;
            sb[0 * qkv_dim + c] = __float2bfloat16(h0);
            sb[1 * qkv_dim + c] = __float2bfloat16(h1);
            sb[2 * qkv_dim + c] = __float2bfloat16(h2);
        }
    }
    if (update_state) {
        hbase[0 * qkv_dim + c] = __float2bfloat16(h0);
        hbase[1 * qkv_dim + c] = __float2bfloat16(h1);
        hbase[2 * qkv_dim + c] = __float2bfloat16(h2);
    }
}

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

// Fused batched single-token decode: GDN recurrent update + RMSNorm(direct weight) * sigmoid(z) in ONE launch!
// Eliminates core_out HBM round-trip and handles all M decode sessions in parallel.
// Grid: (M * nh_v), Block: 256 threads (v = 0..127, kh = 0..1).
__global__ void k_gdn_rec_norm_gate_batch(const __nv_bfloat16* __restrict__ qkv_conv,
                                          const __nv_bfloat16* __restrict__ a_in,
                                          const __nv_bfloat16* __restrict__ b_in,
                                          const __nv_bfloat16* __restrict__ z_in,
                                          const __nv_bfloat16* __restrict__ A_log,
                                          const __nv_bfloat16* __restrict__ dt_bias,
                                          const __nv_bfloat16* __restrict__ norm_w,
                                          float* __restrict__ S_all,
                                          __nv_bfloat16* __restrict__ core_out,
                                          const int* __restrict__ d_slot,
                                          const int* __restrict__ d_pos,
                                          int M, int nh_k, int nh_v, int k0_global, int v0_global,
                                          float eps, bool update_state) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_v) return;
    const int m = idx / nh_v;
    const int lv = idx % nh_v;
    const int slot = d_slot[m];
    const int pos = d_pos[m];

    const int gv = v0_global + lv;
    const int lk = gv / geom::GDN_V_PER_K - k0_global;
    const int qkv_dim = (2 * nh_k + nh_v) * geom::GDN_DK;

    const int tid = threadIdx.x;
    const int v = tid & 127;
    const int kh = tid >> 7;
    const int k_base = kh * 64;

    __shared__ float sq[128];
    __shared__ float sk[128];
    __shared__ float sp[128];
    __shared__ float s_meta[2];
    __shared__ float qwarp[4];
    __shared__ float kwarp[4];

    const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(m) * qkv_dim;
    const __nv_bfloat16* q_ptr = qkv + static_cast<size_t>(lk) * 128;
    const __nv_bfloat16* k_ptr = qkv + static_cast<size_t>(nh_k + lk) * 128;
    const __nv_bfloat16* v_ptr = qkv + static_cast<size_t>(2 * nh_k + lv) * 128;

    if (kh == 0) sq[v] = __bfloat162float(q_ptr[v]);
    else sk[v] = __bfloat162float(k_ptr[v]);

    if (tid == 0) {
        const float neg_A = -expf(__bfloat162float(A_log[lv]));
        const float dtb = __bfloat162float(dt_bias[lv]);
        const float av = __bfloat162float(a_in[static_cast<size_t>(m) * nh_v + lv]);
        const float bv = __bfloat162float(b_in[static_cast<size_t>(m) * nh_v + lv]);
        s_meta[0] = expf(neg_A * softplus_f(av + dtb));
        s_meta[1] = sigmoid_f(bv);
    }
    __syncthreads();

    constexpr float inv_sqrt_dk = 0.08838834764831845f;
    if (kh == 0) {
        const float qv = sq[v];
        const float q2 = warp_sum(qv * qv);
        if ((v & 31) == 0) qwarp[v >> 5] = q2;
        __syncthreads();
        const float qsum = qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3];
        sq[v] = qv * rsqrtf(qsum + 1e-6f) * inv_sqrt_dk;
    } else {
        const float kv = sk[v];
        const float k2 = warp_sum(kv * kv);
        if ((v & 31) == 0) kwarp[v >> 5] = k2;
        __syncthreads();
        const float ksum = kwarp[0] + kwarp[1] + kwarp[2] + kwarp[3];
        sk[v] = kv * rsqrtf(ksum + 1e-6f);
    }
    __syncthreads();

    const float decay = s_meta[0];
    const float beta = s_meta[1];

    float* S_head = S_all + (static_cast<size_t>(slot) * nh_v + lv) * (128 * 128);
    float s_col[64];
    float kv_part = 0.0f;
    if (pos == 0) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) s_col[i] = 0.0f;
    } else {
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float sv = S_head[(k_base + i) * 128 + v] * decay;
            s_col[i] = sv;
            kv_part = fmaf(sv, sk[k_base + i], kv_part);
        }
    }
    if (kh == 1) sp[v] = kv_part;
    __syncthreads();
    if (kh == 0) {
        const float v_val = __bfloat162float(v_ptr[v]);
        sp[v] = (v_val - (kv_part + sp[v])) * beta;
    }
    __syncthreads();
    const float delta_v = sp[v];

    float out_part = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const float sv = fmaf(sk[k_base + i], delta_v, s_col[i]);
        if (update_state) S_head[(k_base + i) * 128 + v] = sv;
        out_part = fmaf(sv, sq[k_base + i], out_part);
    }
    if (kh == 1) sp[v] = out_part;
    __syncthreads();

    // Fused Qwen4ExpTextRMSNormGated (direct norm_w * sigmoid(z)) on kh == 0 (all 128 v-dims!)
    if (kh == 0) {
        const float hv = __bfloat162float(__float2bfloat16(out_part + sp[v]));
        const float ss = warp_sum(hv * hv);
        if ((v & 31) == 0) qwarp[v >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
        const float normed_bf = __bfloat162float(__float2bfloat16(hv * inv));
        const float scaled = __bfloat162float(__float2bfloat16(normed_bf * __bfloat162float(norm_w[v])));
        const float gate = sigmoid_f(__bfloat162float(z_in[static_cast<size_t>(idx) * 128 + v]));
        core_out[static_cast<size_t>(idx) * 128 + v] = __float2bfloat16(scaled * gate);
    } else {
        __syncthreads();
    }
}

// Recurrent Gated Delta Rule over one multi-token segment (slot, row0, pos0, len).
// When S_snap != nullptr (MTP verify step), saves post-token-t state into S_snap[slot, t] for t < len - 1!
__global__ void k_gdn_recurrent_seg(const __nv_bfloat16* __restrict__ qkv_conv,
                                    const __nv_bfloat16* __restrict__ a_in,
                                    const __nv_bfloat16* __restrict__ b_in,
                                    const __nv_bfloat16* __restrict__ A_log,
                                    const __nv_bfloat16* __restrict__ dt_bias,
                                    float* __restrict__ S_all,
                                    float* __restrict__ S_snap,
                                    __nv_bfloat16* __restrict__ core_out,
                                    int nh_k, int nh_v, int k0_global, int v0_global,
                                    int slot, int row0, int pos0, int len, int snap_k, bool update_state) {
    const int lv = blockIdx.x;
    if (lv >= nh_v) return;
    const int gv = v0_global + lv;
    const int lk = gv / geom::GDN_V_PER_K - k0_global;
    const int qkv_dim = (2 * nh_k + nh_v) * geom::GDN_DK;

    const int tid = threadIdx.x;
    const int v = tid & 127;
    const int kh = tid >> 7;
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
    __shared__ float sp[128];
    __shared__ float s_meta[2];

    const float neg_A = -expf(__bfloat162float(A_log[lv]));
    const float dtb = __bfloat162float(dt_bias[lv]);
    constexpr float inv_sqrt_dk = 0.08838834764831845f;

    for (int t = 0; t < len; ++t) {
        const int r = row0 + t;
        const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(r) * qkv_dim;
        const __nv_bfloat16* q_ptr = qkv + static_cast<size_t>(lk) * 128;
        const __nv_bfloat16* k_ptr = qkv + static_cast<size_t>(nh_k + lk) * 128;
        const __nv_bfloat16* v_ptr = qkv + static_cast<size_t>(2 * nh_k + lv) * 128;

        if (kh == 0) sq[v] = __bfloat162float(q_ptr[v]);
        else sk[v] = __bfloat162float(k_ptr[v]);
        if (tid == 0) {
            const float av = __bfloat162float(a_in[static_cast<size_t>(r) * nh_v + lv]);
            const float bv = __bfloat162float(b_in[static_cast<size_t>(r) * nh_v + lv]);
            s_meta[0] = expf(neg_A * softplus_f(av + dtb));
            s_meta[1] = sigmoid_f(bv);
        }
        __syncthreads();

        if (kh == 0) {
            float qv = sq[v];
            float q2 = warp_sum(qv * qv);
            __shared__ float qwarp[4];
            if ((v & 31) == 0) qwarp[v >> 5] = q2;
            __syncthreads();
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
        if (S_snap && t < len - 1 && t < snap_k) {
            float* S_dst = S_snap + ((static_cast<size_t>(slot) * snap_k + t) * nh_v + lv) * (128 * 128);
            #pragma unroll
            for (int i = 0; i < 64; ++i) S_dst[(k_base + i) * 128 + v] = s_col[i];
        }
        __syncthreads();
    }

    if (update_state) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) S_head[(k_base + i) * 128 + v] = s_col[i];
    }
}

__global__ void k_gdn_norm_gate(__nv_bfloat16* __restrict__ core, const __nv_bfloat16* __restrict__ z,
                                const __nv_bfloat16* __restrict__ w, int M, int nh_v, float eps) {
    const int idx = blockIdx.x;
    if (idx >= M * nh_v) return;
    const int d = threadIdx.x;
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

// Graph-safe batched MTP verify conv1d across S sessions x V tokens/session (V = snap_k + 1, row0 = s * V).
// Reads slot & pos0 from device arrays d_slot/d_pos (ZERO host-baked segment metadata!).
__global__ void k_gdn_conv1d_verify_batch(const __nv_bfloat16* __restrict__ qkv_raw,
                                          const __nv_bfloat16* __restrict__ w,
                                          __nv_bfloat16* __restrict__ conv_hist,
                                          __nv_bfloat16* __restrict__ conv_snap,
                                          __nv_bfloat16* __restrict__ qkv_out,
                                          const int* __restrict__ d_slot,
                                          const int* __restrict__ d_pos,
                                          int S, int V, int snap_k, int qkv_dim, bool update_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * qkv_dim) return;
    const int s = idx / qkv_dim;
    const int c = idx % qkv_dim;
    const int row0 = s * V;
    const int slot = d_slot[row0];
    const int pos0 = d_pos[row0];

    const __nv_bfloat16* wc = w + static_cast<size_t>(c) * 4;
    const float w0 = __bfloat162float(wc[0]);
    const float w1 = __bfloat162float(wc[1]);
    const float w2 = __bfloat162float(wc[2]);
    const float w3 = __bfloat162float(wc[3]);

    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * 3 * qkv_dim;
    float h0 = (pos0 > 0) ? __bfloat162float(hbase[0 * qkv_dim + c]) : 0.0f;
    float h1 = (pos0 > 0) ? __bfloat162float(hbase[1 * qkv_dim + c]) : 0.0f;
    float h2 = (pos0 > 0) ? __bfloat162float(hbase[2 * qkv_dim + c]) : 0.0f;

    for (int t = 0; t < V; ++t) {
        const size_t ri = static_cast<size_t>(row0 + t) * qkv_dim + c;
        const float x = __bfloat162float(qkv_raw[ri]);
        const float y = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        qkv_out[ri] = __float2bfloat16(silu_f(y));
        h0 = h1;
        h1 = h2;
        h2 = x;
        if (conv_snap && t < V - 1 && t < snap_k) {
            __nv_bfloat16* sb = conv_snap + (static_cast<size_t>(slot) * snap_k + t) * 3 * qkv_dim;
            sb[0 * qkv_dim + c] = __float2bfloat16(h0);
            sb[1 * qkv_dim + c] = __float2bfloat16(h1);
            sb[2 * qkv_dim + c] = __float2bfloat16(h2);
        }
    }
    if (update_state) {
        hbase[0 * qkv_dim + c] = __float2bfloat16(h0);
        hbase[1 * qkv_dim + c] = __float2bfloat16(h1);
        hbase[2 * qkv_dim + c] = __float2bfloat16(h2);
    }
}

// Graph-safe batched MTP verify GDN recurrence + fused NormGate across S sessions x V tokens/session!
// Grid: (S * nh_v), Block: 256 threads (v = 0..127, kh = 0..1).
__global__ void k_gdn_rec_verify_batch(const __nv_bfloat16* __restrict__ qkv_conv,
                                       const __nv_bfloat16* __restrict__ a_in,
                                       const __nv_bfloat16* __restrict__ b_in,
                                       const __nv_bfloat16* __restrict__ z_in,
                                       const __nv_bfloat16* __restrict__ A_log,
                                       const __nv_bfloat16* __restrict__ dt_bias,
                                       const __nv_bfloat16* __restrict__ norm_w,
                                       float* __restrict__ S_all,
                                       float* __restrict__ S_snap,
                                       __nv_bfloat16* __restrict__ core_out,
                                       const int* __restrict__ d_slot,
                                       const int* __restrict__ d_pos,
                                       int S, int V, int snap_k, int nh_k, int nh_v, int k0_global, int v0_global,
                                       float eps, bool update_state) {
    const int idx = blockIdx.x;
    if (idx >= S * nh_v) return;
    const int s = idx / nh_v;
    const int lv = idx % nh_v;
    const int row0 = s * V;
    const int slot = d_slot[row0];
    const int pos0 = d_pos[row0];

    const int gv = v0_global + lv;
    const int lk = gv / geom::GDN_V_PER_K - k0_global;
    const int qkv_dim = (2 * nh_k + nh_v) * geom::GDN_DK;

    const int tid = threadIdx.x;
    const int v = tid & 127;
    const int kh = tid >> 7;
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
    __shared__ float sp[128];
    __shared__ float s_meta[2];
    __shared__ float qwarp[4];
    __shared__ float kwarp[4];

    const float neg_A = -expf(__bfloat162float(A_log[lv]));
    const float dtb = __bfloat162float(dt_bias[lv]);
    constexpr float inv_sqrt_dk = 0.08838834764831845f;

    for (int t = 0; t < V; ++t) {
        const int r = row0 + t;
        const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(r) * qkv_dim;
        const __nv_bfloat16* q_ptr = qkv + static_cast<size_t>(lk) * 128;
        const __nv_bfloat16* k_ptr = qkv + static_cast<size_t>(nh_k + lk) * 128;
        const __nv_bfloat16* v_ptr = qkv + static_cast<size_t>(2 * nh_k + lv) * 128;

        if (kh == 0) sq[v] = __bfloat162float(q_ptr[v]);
        else sk[v] = __bfloat162float(k_ptr[v]);
        if (tid == 0) {
            const float av = __bfloat162float(a_in[static_cast<size_t>(r) * nh_v + lv]);
            const float bv = __bfloat162float(b_in[static_cast<size_t>(r) * nh_v + lv]);
            s_meta[0] = expf(neg_A * softplus_f(av + dtb));
            s_meta[1] = sigmoid_f(bv);
        }
        __syncthreads();

        if (kh == 0) {
            const float qv = sq[v];
            const float q2 = warp_sum(qv * qv);
            if ((v & 31) == 0) qwarp[v >> 5] = q2;
            __syncthreads();
            const float qsum = qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3];
            sq[v] = qv * rsqrtf(qsum + 1e-6f) * inv_sqrt_dk;
        } else {
            const float kv = sk[v];
            const float k2 = warp_sum(kv * kv);
            if ((v & 31) == 0) kwarp[v >> 5] = k2;
            __syncthreads();
            const float ksum = kwarp[0] + kwarp[1] + kwarp[2] + kwarp[3];
            sk[v] = kv * rsqrtf(ksum + 1e-6f);
        }
        __syncthreads();

        const float decay = s_meta[0];
        const float beta = s_meta[1];

        float kv_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            s_col[i] *= decay;
            kv_part = fmaf(s_col[i], sk[k_base + i], kv_part);
        }
        if (kh == 1) sp[v] = kv_part;
        __syncthreads();
        if (kh == 0) {
            const float v_val = __bfloat162float(v_ptr[v]);
            sp[v] = (v_val - (kv_part + sp[v])) * beta;
        }
        __syncthreads();
        const float delta_v = sp[v];

        float out_part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            s_col[i] = fmaf(sk[k_base + i], delta_v, s_col[i]);
            out_part = fmaf(s_col[i], sq[k_base + i], out_part);
        }
        if (kh == 1) sp[v] = out_part;
        __syncthreads();

        if (kh == 0) {
            const float hv = __bfloat162float(__float2bfloat16(out_part + sp[v]));
            const float ss = warp_sum(hv * hv);
            if ((v & 31) == 0) qwarp[v >> 5] = ss;
            __syncthreads();
            const float inv = rsqrtf((qwarp[0] + qwarp[1] + qwarp[2] + qwarp[3]) / 128.0f + eps);
            const float normed_bf = __bfloat162float(__float2bfloat16(hv * inv));
            const float scaled = __bfloat162float(__float2bfloat16(normed_bf * __bfloat162float(norm_w[v])));
            const size_t oi = (static_cast<size_t>(r) * nh_v + lv) * 128 + v;
            const float gate = sigmoid_f(__bfloat162float(z_in[oi]));
            core_out[oi] = __float2bfloat16(scaled * gate);
        } else {
            __syncthreads();
        }

        if (S_snap && t < V - 1 && t < snap_k) {
            float* S_dst = S_snap + ((static_cast<size_t>(slot) * snap_k + t) * nh_v + lv) * (128 * 128);
            #pragma unroll
            for (int i = 0; i < 64; ++i) S_dst[(k_base + i) * 128 + v] = s_col[i];
        }
        __syncthreads();
    }

    if (update_state) {
        #pragma unroll
        for (int i = 0; i < 64; ++i) S_head[(k_base + i) * 128 + v] = s_col[i];
    }
}

// Execute one GDN layer on rank `rank` of `nr` over `M` rows divided into `segs`.
// When `d_slot != nullptr && d_pos != nullptr && nsegs == M` (all single-token decode segments),
// runs the fused single-launch batched conv1d + fused recurrent-norm-gate path!
inline void gdn_forward(cublasHandle_t blas, const GdnWeights& w, GdnLayerState& st_buf, GdnScratch& sc,
                        const __nv_bfloat16* x, __nv_bfloat16* out_part,
                        const StepSeg* segs, int nsegs, int M, int nr, int rank,
                        bool update_state, cudaStream_t st,
                        const int* d_slot = nullptr, const int* d_pos = nullptr, int snap_k = 0) {
    const int nh_v = geom::gdn_vh(nr);
    const int nh_k = geom::gdn_kh(nr, rank);
    const int k0 = geom::gdn_k0(nr, rank);
    const int v0 = geom::gdn_v0(nr, rank);
    const int qkv_dim = geom::gdn_qkv_local(nr, rank);

    // 1. Input projections (coalesced FP8/BF16 uint4 GEMV for M <= 12, cuBLAS tensor cores for M > 12)
    if ((M <= 12 && w.fp8_qkv.q) || M <= 8) {
        if (w.fp8_qkv.q) {
            launch_fp8_gemv_smallm<false>(x, w.fp8_qkv, sc.qkv_raw, M, st);
            launch_fp8_gemv_smallm<false>(x, w.fp8_z, sc.z, M, st);
        } else {
            launch_bf16_gemv_smallm<false>(x, w.in_proj_qkv, sc.qkv_raw, M, qkv_dim, geom::HID, st);
            launch_bf16_gemv_smallm<false>(x, w.in_proj_z, sc.z, M, nh_v * geom::GDN_DV, geom::HID, st);
        }
        const int ab_grid = (2 * nh_v + 3) / 4;
        if (M == 1) k_gdn_in_proj_ab<1><<<ab_grid, 128, 0, st>>>(x, w.in_proj_a, w.in_proj_b, sc.a, sc.b, M, nh_v);
        else if (M <= 4) k_gdn_in_proj_ab<4><<<ab_grid, 128, 0, st>>>(x, w.in_proj_a, w.in_proj_b, sc.a, sc.b, M, nh_v);
        else if (M <= 8) k_gdn_in_proj_ab<8><<<ab_grid, 128, 0, st>>>(x, w.in_proj_a, w.in_proj_b, sc.a, sc.b, M, nh_v);
        else k_gdn_in_proj_ab<12><<<ab_grid, 128, 0, st>>>(x, w.in_proj_a, w.in_proj_b, sc.a, sc.b, M, nh_v);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, qkv_dim, geom::HID, x, w.in_proj_qkv, sc.qkv_raw);
        gemm_bf16_nt(blas, M, nh_v * geom::GDN_DV, geom::HID, x, w.in_proj_z, sc.z);
        gemm_bf16_nt(blas, M, nh_v, geom::HID, x, w.in_proj_a, sc.a);
        gemm_bf16_nt(blas, M, nh_v, geom::HID, x, w.in_proj_b, sc.b);
    }

    // 2. Conv1d + GDN Recurrence + NormGate
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
        k_gdn_conv1d_dec_batch<<<(M * qkv_dim + 255) / 256, 256, 0, st>>>(
            sc.qkv_raw, w.conv1d, st_buf.conv, sc.qkv_conv, d_slot, d_pos, M, qkv_dim, update_state);
        k_gdn_rec_norm_gate_batch<<<M * nh_v, 256, 0, st>>>(
            sc.qkv_conv, sc.a, sc.b, sc.z, w.A_log, w.dt_bias, w.norm, st_buf.S, sc.core,
            d_slot, d_pos, M, nh_k, nh_v, k0, v0, geom::RMS_EPS, update_state);
    } else if (all_verify) {
        k_gdn_conv1d_verify_batch<<<(nsegs * qkv_dim + 255) / 256, 256, 0, st>>>(
            sc.qkv_raw, w.conv1d, st_buf.conv, st_buf.conv_snap, sc.qkv_conv,
            d_slot, d_pos, nsegs, V, snap_k, qkv_dim, update_state);
        k_gdn_rec_verify_batch<<<nsegs * nh_v, 256, 0, st>>>(
            sc.qkv_conv, sc.a, sc.b, sc.z, w.A_log, w.dt_bias, w.norm, st_buf.S, st_buf.S_snap, sc.core,
            d_slot, d_pos, nsegs, V, snap_k, nh_k, nh_v, k0, v0, geom::RMS_EPS, update_state);
    } else {
        for (int s = 0; s < nsegs; ++s) {
            const StepSeg& g = segs[s];
            if (g.len <= 0) continue;
            k_gdn_conv1d_seg<<<(qkv_dim + 255) / 256, 256, 0, st>>>(
                sc.qkv_raw, w.conv1d, st_buf.conv, snap_k > 0 ? st_buf.conv_snap : nullptr, sc.qkv_conv,
                qkv_dim, g.slot, g.row0, g.pos0, g.len, snap_k, update_state);
            k_gdn_recurrent_seg<<<nh_v, 256, 0, st>>>(
                sc.qkv_conv, sc.a, sc.b, w.A_log, w.dt_bias, st_buf.S, snap_k > 0 ? st_buf.S_snap : nullptr, sc.core,
                nh_k, nh_v, k0, v0, g.slot, g.row0, g.pos0, g.len, snap_k, update_state);
        }
        k_gdn_norm_gate<<<M * nh_v, 128, 0, st>>>(sc.core, sc.z, w.norm, M, nh_v, geom::RMS_EPS);
    }

    // 3. Output projection
    if ((M <= 12 && w.fp8_out.q) || M <= 8) {
        if (w.fp8_out.q) {
            launch_fp8_gemv_smallm<false>(sc.core, w.fp8_out, out_part, M, st);
        } else {
            launch_bf16_gemv_smallm<false>(sc.core, w.out_proj, out_part, M, geom::HID, nh_v * geom::GDN_DV, st);
        }
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt(blas, M, geom::HID, nh_v * geom::GDN_DV, sc.core, w.out_proj, out_part);
    }
}

} // namespace qw38
