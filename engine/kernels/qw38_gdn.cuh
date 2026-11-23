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
    // Chunked WY Tensor-Core prefill scratch (sm_80+, allocated for max_M rows)
    uint16_t* wy_qe = nullptr;          // [nh_v, max_M, 128] BF16
    uint16_t* wy_kd = nullptr;          // [nh_v, max_M, 128] BF16
    uint16_t* wy_w = nullptr;           // [nh_v, max_M, 128] BF16
    uint16_t* wy_u = nullptr;           // [nh_v, max_M, 128] BF16
    uint16_t* wy_aq = nullptr;          // [nh_v, max_M, 64] BF16
    float* wy_glast = nullptr;          // [nh_v, (max_M + 63) / 64] FP32
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

// ------------------------------------------------------------------------------------------------
// Chunked WY Tensor-Core GDN Prefill (sm_80+ BF16 MMA m16n8k16) + 2D Conv1d + Warp NormGate
// ------------------------------------------------------------------------------------------------
constexpr int GDN_WY_CH = 64;
constexpr int GDN_WY_VB = 64;

__device__ __forceinline__ void gdn_mma_bf16_m16n8k16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void gdn_ldsm4(uint32_t (&a)[4], const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
#endif
}

__device__ __forceinline__ void gdn_ldsm2(uint32_t (&b)[2], const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(s));
#endif
}

__device__ __forceinline__ uint32_t gdn_pack2_bf16(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float2 gdn_unpack2_bf16(uint32_t u) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}

__device__ __forceinline__ uint16_t gdn_pack1_bf16(float a) {
    __nv_bfloat16 v = __float2bfloat16(a);
    return *reinterpret_cast<const uint16_t*>(&v);
}

// 2D-parallel vectorized causal conv1d (kernel=4, SiLU) over a prefill segment (4 channels/thread, 16 tokens/CTA-y).
__global__ void __launch_bounds__(256) k_gdn_conv1d_pf(
    const __nv_bfloat16* __restrict__ qkv_raw,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ conv_hist,
    __nv_bfloat16* __restrict__ qkv_out,
    int qkv_dim, int slot, int row0, int pos0, int len) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int qkv4 = qkv_dim >> 2;
    if (c4 >= qkv4) return;
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
    const __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * 3 * qkv_dim;

    auto load4 = [&](int t, float (&dst)[4]) {
        uint2 v;
        if (t >= 0) v = reinterpret_cast<const uint2*>(qkv_raw + static_cast<size_t>(row0 + t) * qkv_dim)[c4];
        else if (pos0 > 0) v = reinterpret_cast<const uint2*>(hbase + static_cast<size_t>(3 + t) * qkv_dim)[c4];
        else { dst[0] = dst[1] = dst[2] = dst[3] = 0.0f; return; }
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        dst[0] = f0.x; dst[1] = f0.y; dst[2] = f1.x; dst[3] = f1.y;
    };

    float h0[4], h1[4], h2[4], x[4];
    load4(t0 - 3, h0);
    load4(t0 - 2, h1);
    load4(t0 - 1, h2);
    for (int t = t0; t < t1; ++t) {
        load4(t, x);
        float y[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            y[k] = silu_f(h0[k] * w0[k] + h1[k] * w1[k] + h2[k] * w2[k] + x[k] * w3[k]);
            h0[k] = h1[k]; h1[k] = h2[k]; h2[k] = x[k];
        }
        uint2 ov;
        *reinterpret_cast<__nv_bfloat162*>(&ov.x) = __floats2bfloat162_rn(y[0], y[1]);
        *reinterpret_cast<__nv_bfloat162*>(&ov.y) = __floats2bfloat162_rn(y[2], y[3]);
        reinterpret_cast<uint2*>(qkv_out + static_cast<size_t>(row0 + t) * qkv_dim)[c4] = ov;
    }
}

__global__ void __launch_bounds__(256) k_gdn_conv1d_hist_pf(
    const __nv_bfloat16* __restrict__ qkv_raw,
    __nv_bfloat16* __restrict__ conv_hist,
    int qkv_dim, int slot, int row0, int pos0, int len) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c4 >= (qkv_dim >> 2)) return;
    __nv_bfloat16* hbase = conv_hist + static_cast<size_t>(slot) * 3 * qkv_dim;
    auto get4 = [&](int t) -> uint2 {
        if (t >= 0) return reinterpret_cast<const uint2*>(qkv_raw + static_cast<size_t>(row0 + t) * qkv_dim)[c4];
        if (pos0 > 0) return reinterpret_cast<const uint2*>(hbase + static_cast<size_t>(3 + t) * qkv_dim)[c4];
        return make_uint2(0, 0);
    };
    const uint2 v0 = get4(len - 3), v1 = get4(len - 2), v2 = get4(len - 1);
    reinterpret_cast<uint2*>(hbase + 0 * qkv_dim)[c4] = v0;
    reinterpret_cast<uint2*>(hbase + 1 * qkv_dim)[c4] = v1;
    reinterpret_cast<uint2*>(hbase + 2 * qkv_dim)[c4] = v2;
}

// Phase 1 (Chunk-parallel WY intra-chunk prep, 64 tokens/chunk):
// Computes cumulative decay G, L2-normalized q/k, QE/KD, intra-chunk causal AQ,
// lower-triangular (I + A)^-1 via forward substitution, and WY factors W and U via BF16 MMA.
// Grid: dim3(nh_v, nchunks), Block: 256.
__global__ void __launch_bounds__(256) k_gdn_wy_intra(
    const __nv_bfloat16* __restrict__ qkv_conv,
    const __nv_bfloat16* __restrict__ a_in,
    const __nv_bfloat16* __restrict__ b_in,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    uint16_t* __restrict__ QE,
    uint16_t* __restrict__ KD,
    uint16_t* __restrict__ W_out,
    uint16_t* __restrict__ U_out,
    uint16_t* __restrict__ AQ_out,
    float* __restrict__ G_last_out,
    int nh_k, int nh_v, int k0_global, int v0_global,
    int row0, int len, int nchunks, int stride_m) {
    const int lv = blockIdx.x, c = blockIdx.y;
    if (lv >= nh_v || c >= nchunks) return;
    const int gv = v0_global + lv;
    const int lk = gv / geom::GDN_V_PER_K - k0_global;
    const int qkv_dim = (2 * nh_k + nh_v) * 128;
    const int c_off = c * GDN_WY_CH;
    const int n = min(GDN_WY_CH, len - c_off);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const size_t h_base_128 = (static_cast<size_t>(lv) * stride_m + c_off) * 128;
    const size_t h_base_64  = (static_cast<size_t>(lv) * stride_m + c_off) * GDN_WY_CH;

    __shared__ float s_G[GDN_WY_CH], s_beta[GDN_WY_CH], s_qinv[GDN_WY_CH], s_kinv[GDN_WY_CH];
    __shared__ float s_expG[GDN_WY_CH], s_expGd[GDN_WY_CH], s_ksc[GDN_WY_CH];
    __shared__ float s_A[GDN_WY_CH][GDN_WY_CH + 1];
    __shared__ uint16_t s_t0[GDN_WY_CH][72];
    __shared__ uint16_t s_t1[GDN_WY_CH][72];

    // Warp-0 parallel inclusive scan for cumulative decay G[0..n-1]
    if (warp == 0) {
        const float neg_A = -expf(__bfloat162float(A_log[lv]));
        const float dtb = __bfloat162float(dt_bias[lv]);
        float g0 = 0.0f, g1 = 0.0f;
        if (lane < n) {
            const int r = row0 + c_off + lane;
            g0 = neg_A * softplus_f(__bfloat162float(a_in[static_cast<size_t>(r) * nh_v + lv]) + dtb);
            s_beta[lane] = sigmoid_f(__bfloat162float(b_in[static_cast<size_t>(r) * nh_v + lv]));
        } else {
            s_beta[lane] = 0.0f;
        }
        if (lane + 32 < n) {
            const int r = row0 + c_off + lane + 32;
            g1 = neg_A * softplus_f(__bfloat162float(a_in[static_cast<size_t>(r) * nh_v + lv]) + dtb);
            s_beta[lane + 32] = sigmoid_f(__bfloat162float(b_in[static_cast<size_t>(r) * nh_v + lv]));
        } else {
            s_beta[lane + 32] = 0.0f;
        }
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float t0 = __shfl_up_sync(0xffffffffu, g0, off);
            const float t1 = __shfl_up_sync(0xffffffffu, g1, off);
            if (lane >= off) { g0 += t0; g1 += t1; }
        }
        const float sum_lo = __shfl_sync(0xffffffffu, g0, 31);
        g1 += sum_lo;
        s_G[lane] = g0;
        s_G[lane + 32] = g1;
        const float g_last = (n <= 32) ? __shfl_sync(0xffffffffu, g0, n - 1)
                                       : __shfl_sync(0xffffffffu, g1, n - 33);
        s_expG[lane] = __expf(g0);
        s_expG[lane + 32] = __expf(g1);
        s_expGd[lane] = __expf(g_last - g0);
        s_expGd[lane + 32] = __expf(g_last - g1);
        if (lane == 0) G_last_out[static_cast<size_t>(lv) * nchunks + c] = g_last;
    }

    constexpr float inv_sqrt_dk = 0.08838834764831845f;
    for (int i = warp; i < GDN_WY_CH; i += 8) {
        if (i < n) {
            const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(row0 + c_off + i) * qkv_dim;
            const uint2 qv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(lk) * 128 + lane * 4);
            const uint2 kv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(nh_k + lk) * 128 + lane * 4);
            const float2 q0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qv.x));
            const float2 q1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qv.y));
            const float2 k0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.x));
            const float2 k1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.y));
            const float qs = warp_sum(q0.x * q0.x + q0.y * q0.y + q1.x * q1.x + q1.y * q1.y);
            const float ks = warp_sum(k0.x * k0.x + k0.y * k0.y + k1.x * k1.x + k1.y * k1.y);
            if (lane == 0) {
                s_qinv[i] = rsqrtf(qs + 1e-6f) * inv_sqrt_dk;
                s_kinv[i] = rsqrtf(ks + 1e-6f);
            }
        } else if (lane == 0) {
            s_qinv[i] = 0.0f;
            s_kinv[i] = 0.0f;
        }
    }
    __syncthreads();
    if (tid < GDN_WY_CH) s_ksc[tid] = s_kinv[tid] * s_beta[tid] * s_expG[tid];
    __syncthreads();

    const int wm = (warp >> 1) * 16;
    const int wn = (warp & 1) * 32;
    const int a_r = lane & 15, a_c = (lane >> 4) * 8;
    const int b_r = lane & 7,  b_c = ((lane >> 3) & 1) * 8;
    float kk_acc[4][4] = {}, qk_acc[4][4] = {};

    for (int d0 = 0; d0 < 128; d0 += 64) {
        for (int idx = tid; idx < GDN_WY_CH * 16; idx += 256) {
            const int i = idx >> 4, d4 = idx & 15;
            uint2 q_out = make_uint2(0, 0), k_out = make_uint2(0, 0);
            if (i < n) {
                const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(row0 + c_off + i) * qkv_dim;
                const uint2 qv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(lk) * 128 + d0 + d4 * 4);
                const uint2 kv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(nh_k + lk) * 128 + d0 + d4 * 4);
                const float2 q0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qv.x));
                const float2 q1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qv.y));
                const float2 k0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.x));
                const float2 k1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.y));
                const float qi = s_qinv[i], ki = s_kinv[i];
                const float qe_s = qi * s_expG[i], kd_s = ki * s_expGd[i];
                q_out.x = gdn_pack2_bf16(q0.x * qi, q0.y * qi);
                q_out.y = gdn_pack2_bf16(q1.x * qi, q1.y * qi);
                k_out.x = gdn_pack2_bf16(k0.x * ki, k0.y * ki);
                k_out.y = gdn_pack2_bf16(k1.x * ki, k1.y * ki);
                uint2 qe_v, kd_v;
                qe_v.x = gdn_pack2_bf16(q0.x * qe_s, q0.y * qe_s);
                qe_v.y = gdn_pack2_bf16(q1.x * qe_s, q1.y * qe_s);
                kd_v.x = gdn_pack2_bf16(k0.x * kd_s, k0.y * kd_s);
                kd_v.y = gdn_pack2_bf16(k1.x * kd_s, k1.y * kd_s);
                *reinterpret_cast<uint2*>(QE + h_base_128 + static_cast<size_t>(i) * 128 + d0 + d4 * 4) = qe_v;
                *reinterpret_cast<uint2*>(KD + h_base_128 + static_cast<size_t>(i) * 128 + d0 + d4 * 4) = kd_v;
            }
            *reinterpret_cast<uint2*>(s_t0[i] + d4 * 4) = q_out;
            *reinterpret_cast<uint2*>(s_t1[i] + d4 * 4) = k_out;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int dk = kk * 16;
            uint32_t qa[4], ka[4];
            gdn_ldsm4(qa, s_t0[wm + a_r] + dk + a_c);
            gdn_ldsm4(ka, s_t1[wm + a_r] + dk + a_c);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t kb[2];
                gdn_ldsm2(kb, s_t1[wn + j * 8 + b_r] + dk + b_c);
                gdn_mma_bf16_m16n8k16(kk_acc[j], ka, kb);
                gdn_mma_bf16_m16n8k16(qk_acc[j], qa, kb);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r_a = wm + gid, r_b = r_a + 8;
        const int c_0 = wn + j * 8 + 2 * tid4;
        auto do_pair = [&](int r, float k0, float k1, float q0, float q1) {
            const bool v0 = (r < n && c_0 <= r), v1 = (r < n && c_0 + 1 <= r);
            const float d0 = v0 ? __expf(s_G[r] - s_G[c_0]) : 0.0f;
            const float d1 = v1 ? __expf(s_G[r] - s_G[c_0 + 1]) : 0.0f;
            s_A[r][c_0]     = (r < n && c_0 < r)     ? (-s_beta[r] * d0 * k0) : 0.0f;
            s_A[r][c_0 + 1] = (r < n && c_0 + 1 < r) ? (-s_beta[r] * d1 * k1) : 0.0f;
            if (r < n) {
                *reinterpret_cast<uint32_t*>(AQ_out + h_base_64 + static_cast<size_t>(r) * GDN_WY_CH + c_0) =
                    gdn_pack2_bf16(v0 ? d0 * q0 : 0.0f, v1 ? d1 * q1 : 0.0f);
            }
        };
        do_pair(r_a, kk_acc[j][0], kk_acc[j][1], qk_acc[j][0], qk_acc[j][1]);
        do_pair(r_b, kk_acc[j][2], kk_acc[j][3], qk_acc[j][2], qk_acc[j][3]);
    }
    __syncthreads();

    // Forward substitution: invert (I - L) in-place in s_A
    for (int i = 2; i < n; ++i) {
        float b = 0.0f;
        if (tid < i - 1) {
            b = s_A[i][tid];
            for (int m = tid + 1; m < i; ++m) b = fmaf(s_A[i][m], s_A[m][tid], b);
        }
        __syncthreads();
        if (tid < i - 1) s_A[i][tid] = b;
        __syncthreads();
    }
    if (tid < GDN_WY_CH) s_A[tid][tid] = (tid < n) ? 1.0f : 0.0f;
    __syncthreads();

    // Compute WY factors W and U via BF16 MMA (2 tiles of 64 cols)
    for (int d0 = 0; d0 < 128; d0 += 64) {
        for (int idx = tid; idx < GDN_WY_CH * 16; idx += 256) {
            const int j = idx >> 4, d4 = idx & 15;
            if (j < n) {
                const __nv_bfloat16* qkv = qkv_conv + static_cast<size_t>(row0 + c_off + j) * qkv_dim;
                const uint2 kv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(nh_k + lk) * 128 + d0 + d4 * 4);
                const uint2 vv = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(2 * nh_k + lv) * 128 + d0 + d4 * 4);
                const float2 k0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.x));
                const float2 k1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.y));
                const float2 v0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv.x));
                const float2 v1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv.y));
                const float ksc = s_ksc[j];
                const float bj = s_beta[j];
                s_t0[d4 * 4 + 0][j] = gdn_pack1_bf16(k0.x * ksc);
                s_t0[d4 * 4 + 1][j] = gdn_pack1_bf16(k0.y * ksc);
                s_t0[d4 * 4 + 2][j] = gdn_pack1_bf16(k1.x * ksc);
                s_t0[d4 * 4 + 3][j] = gdn_pack1_bf16(k1.y * ksc);
                s_t1[d4 * 4 + 0][j] = gdn_pack1_bf16(v0.x * bj);
                s_t1[d4 * 4 + 1][j] = gdn_pack1_bf16(v0.y * bj);
                s_t1[d4 * 4 + 2][j] = gdn_pack1_bf16(v1.x * bj);
                s_t1[d4 * 4 + 3][j] = gdn_pack1_bf16(v1.y * bj);
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) { s_t0[d4 * 4 + k][j] = 0; s_t1[d4 * 4 + k][j] = 0; }
            }
        }
        __syncthreads();

        float wc[4][4] = {}, uc[4][4] = {};
        const int k_lim = min(n, wm + 16);
        for (int k0 = 0; k0 < k_lim; k0 += 16) {
            uint32_t af[4];
            af[0] = gdn_pack2_bf16(s_A[wm + gid][k0 + 2 * tid4],     s_A[wm + gid][k0 + 2 * tid4 + 1]);
            af[1] = gdn_pack2_bf16(s_A[wm + gid + 8][k0 + 2 * tid4], s_A[wm + gid + 8][k0 + 2 * tid4 + 1]);
            af[2] = gdn_pack2_bf16(s_A[wm + gid][k0 + 2 * tid4 + 8], s_A[wm + gid][k0 + 2 * tid4 + 9]);
            af[3] = gdn_pack2_bf16(s_A[wm + gid + 8][k0 + 2 * tid4 + 8], s_A[wm + gid + 8][k0 + 2 * tid4 + 9]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t bk[2], bv[2];
                gdn_ldsm2(bk, s_t0[wn + j * 8 + b_r] + k0 + b_c);
                gdn_ldsm2(bv, s_t1[wn + j * 8 + b_r] + k0 + b_c);
                gdn_mma_bf16_m16n8k16(wc[j], af, bk);
                gdn_mma_bf16_m16n8k16(uc[j], af, bv);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r_a = wm + gid, r_b = r_a + 8;
            const int c_0 = d0 + wn + j * 8 + 2 * tid4;
            if (r_a < n) {
                const size_t oa = h_base_128 + static_cast<size_t>(r_a) * 128 + c_0;
                *reinterpret_cast<uint32_t*>(W_out + oa) = gdn_pack2_bf16(wc[j][0], wc[j][1]);
                *reinterpret_cast<uint32_t*>(U_out + oa) = gdn_pack2_bf16(uc[j][0], uc[j][1]);
            }
            if (r_b < n) {
                const size_t ob = h_base_128 + static_cast<size_t>(r_b) * 128 + c_0;
                *reinterpret_cast<uint32_t*>(W_out + ob) = gdn_pack2_bf16(wc[j][2], wc[j][3]);
                *reinterpret_cast<uint32_t*>(U_out + ob) = gdn_pack2_bf16(uc[j][2], uc[j][3]);
            }
        }
        __syncthreads();
    }
}

// Phase 2 (Chunk-sequential FP32 state recurrence via BF16 MMA, split 2x across V=128):
// Keeps recurrent state S[64][128] in FP32 shared memory and executes chunked WY updates on Tensor Cores.
// Grid: dim3(nh_v, 2), Block: 256 (48 KB static smem -> 2 CTAs/SM).
__global__ void __launch_bounds__(256) k_gdn_wy_state(
    const uint16_t* __restrict__ QE,
    const uint16_t* __restrict__ KD,
    const uint16_t* __restrict__ W_in,
    const uint16_t* __restrict__ U_in,
    const uint16_t* __restrict__ AQ_in,
    const float* __restrict__ G_last_in,
    float* __restrict__ S_all,
    __nv_bfloat16* __restrict__ core_out,
    int nh_v, int slot, int row0, int pos0, int len, int nchunks, int stride_m, bool update_state) {
    const int lv = blockIdx.x, vb = blockIdx.y;
    if (lv >= nh_v) return;
    const int v0 = vb * GDN_WY_VB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    const int a_r = lane & 15, a_c = (lane >> 4) * 8;
    const int b_r = lane & 7,  b_c = ((lane >> 3) & 1) * 8;

    __shared__ float s_S[GDN_WY_VB][128];     // 32 KB
    __shared__ uint16_t s_VN[GDN_WY_VB][64];  // 8 KB
    __shared__ uint16_t s_B0[GDN_WY_CH][64];  // 8 KB

    float* S_head = S_all + (static_cast<size_t>(slot) * nh_v + lv) * (128 * 128);
    if (pos0 == 0) {
        for (int idx = tid; idx < GDN_WY_VB * 128; idx += 256) s_S[idx >> 7][idx & 127] = 0.0f;
    } else {
        for (int idx = tid; idx < 128 * (GDN_WY_VB / 4); idx += 256) {
            const int k = idx >> 4, v4 = (idx & 15) << 2;
            const float4 sv = *reinterpret_cast<const float4*>(S_head + k * 128 + v0 + v4);
            s_S[v4 + 0][k] = sv.x; s_S[v4 + 1][k] = sv.y; s_S[v4 + 2][k] = sv.z; s_S[v4 + 3][k] = sv.w;
        }
    }
    __syncthreads();

    const int wm = (warp >> 1) * 16;
    const int wn = (warp & 1) * 32;

    for (int c = 0; c < nchunks; ++c) {
        const int c_off = c * GDN_WY_CH;
        const int n = min(GDN_WY_CH, len - c_off);
        const size_t h_base_128 = (static_cast<size_t>(lv) * stride_m + c_off) * 128;
        const size_t h_base_64  = (static_cast<size_t>(lv) * stride_m + c_off) * GDN_WY_CH;

        // Phase A1: ws = W @ S^T
        float ws[4][4] = {};
        for (int k0 = 0; k0 < 128; k0 += 64) {
            for (int idx = tid; idx < GDN_WY_CH * 8; idx += 256) {
                const int i = idx >> 3, k8 = (idx & 7) << 3;
                *reinterpret_cast<uint4*>(s_B0[i] + k8) = (i < n)
                    ? *reinterpret_cast<const uint4*>(W_in + h_base_128 + static_cast<size_t>(i) * 128 + k0 + k8)
                    : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int dk = kk * 16;
                uint32_t wa[4];
                gdn_ldsm4(wa, s_B0[wm + a_r] + dk + a_c);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int vc = wn + j * 8 + gid;
                    uint32_t sb[2];
                    sb[0] = gdn_pack2_bf16(s_S[vc][k0 + dk + 2 * tid4],     s_S[vc][k0 + dk + 2 * tid4 + 1]);
                    sb[1] = gdn_pack2_bf16(s_S[vc][k0 + dk + 2 * tid4 + 8], s_S[vc][k0 + dk + 2 * tid4 + 9]);
                    gdn_mma_bf16_m16n8k16(ws[j], wa, sb);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r_a = wm + gid, r_b = r_a + 8;
            const int vc = wn + j * 8 + 2 * tid4;
            if (r_a < n) {
                const float2 uf = gdn_unpack2_bf16(*reinterpret_cast<const uint32_t*>(U_in + h_base_128 + static_cast<size_t>(r_a) * 128 + v0 + vc));
                s_VN[vc][r_a]     = gdn_pack1_bf16(uf.x - ws[j][0]);
                s_VN[vc + 1][r_a] = gdn_pack1_bf16(uf.y - ws[j][1]);
            } else { s_VN[vc][r_a] = 0; s_VN[vc + 1][r_a] = 0; }
            if (r_b < n) {
                const float2 uf = gdn_unpack2_bf16(*reinterpret_cast<const uint32_t*>(U_in + h_base_128 + static_cast<size_t>(r_b) * 128 + v0 + vc));
                s_VN[vc][r_b]     = gdn_pack1_bf16(uf.x - ws[j][2]);
                s_VN[vc + 1][r_b] = gdn_pack1_bf16(uf.y - ws[j][3]);
            } else { s_VN[vc][r_b] = 0; s_VN[vc + 1][r_b] = 0; }
        }
        __syncthreads();

        // Phase A2 + B: O = QE @ S^T + AQ @ VN^T
        float qs[4][4] = {};
        for (int k0 = 0; k0 < 128; k0 += 64) {
            for (int idx = tid; idx < GDN_WY_CH * 8; idx += 256) {
                const int i = idx >> 3, k8 = (idx & 7) << 3;
                *reinterpret_cast<uint4*>(s_B0[i] + k8) = (i < n)
                    ? *reinterpret_cast<const uint4*>(QE + h_base_128 + static_cast<size_t>(i) * 128 + k0 + k8)
                    : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int dk = kk * 16;
                uint32_t qa[4];
                gdn_ldsm4(qa, s_B0[wm + a_r] + dk + a_c);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int vc = wn + j * 8 + gid;
                    uint32_t sb[2];
                    sb[0] = gdn_pack2_bf16(s_S[vc][k0 + dk + 2 * tid4],     s_S[vc][k0 + dk + 2 * tid4 + 1]);
                    sb[1] = gdn_pack2_bf16(s_S[vc][k0 + dk + 2 * tid4 + 8], s_S[vc][k0 + dk + 2 * tid4 + 9]);
                    gdn_mma_bf16_m16n8k16(qs[j], qa, sb);
                }
            }
            __syncthreads();
        }
        for (int idx = tid; idx < GDN_WY_CH * 8; idx += 256) {
            const int i = idx >> 3, j8 = (idx & 7) << 3;
            *reinterpret_cast<uint4*>(s_B0[i] + j8) = (i < n)
                ? *reinterpret_cast<const uint4*>(AQ_in + h_base_64 + static_cast<size_t>(i) * GDN_WY_CH + j8)
                : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        const int j_lim = min(n, wm + 16);
        for (int j0 = 0; j0 < j_lim; j0 += 16) {
            uint32_t ma[4];
            gdn_ldsm4(ma, s_B0[wm + a_r] + j0 + a_c);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t vb_r[2];
                gdn_ldsm2(vb_r, s_VN[wn + j * 8 + b_r] + j0 + b_c);
                gdn_mma_bf16_m16n8k16(qs[j], ma, vb_r);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r_a = wm + gid, r_b = r_a + 8;
            const int vc = wn + j * 8 + 2 * tid4;
            if (r_a < n) {
                const size_t oa = (static_cast<size_t>(row0 + c_off + r_a) * nh_v + lv) * 128 + v0 + vc;
                *reinterpret_cast<__nv_bfloat162*>(core_out + oa) = __floats2bfloat162_rn(qs[j][0], qs[j][1]);
            }
            if (r_b < n) {
                const size_t ob = (static_cast<size_t>(row0 + c_off + r_b) * nh_v + lv) * 128 + v0 + vc;
                *reinterpret_cast<__nv_bfloat162*>(core_out + ob) = __floats2bfloat162_rn(qs[j][2], qs[j][3]);
            }
        }
        __syncthreads();

        // Phase C: S = exp(G_last) * S + VN @ KD
        const float eg = __expf(G_last_in[static_cast<size_t>(lv) * nchunks + c]);
        for (int k0 = 0; k0 < 128; k0 += 64) {
            for (int idx = tid; idx < GDN_WY_CH * 16; idx += 256) {
                const int t = idx >> 4, k4 = (idx & 15) << 2;
                if (t < n) {
                    const uint2 kv = *reinterpret_cast<const uint2*>(KD + h_base_128 + static_cast<size_t>(t) * 128 + k0 + k4);
                    const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kv);
                    s_B0[k4 + 0][t] = kp[0]; s_B0[k4 + 1][t] = kp[1]; s_B0[k4 + 2][t] = kp[2]; s_B0[k4 + 3][t] = kp[3];
                } else {
                    s_B0[k4 + 0][t] = s_B0[k4 + 1][t] = s_B0[k4 + 2][t] = s_B0[k4 + 3][t] = 0;
                }
            }
            __syncthreads();

            float ds[4][4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kk = k0 + wn + j * 8 + 2 * tid4;
                ds[j][0] = s_S[wm + gid][kk]         * eg;
                ds[j][1] = s_S[wm + gid][kk + 1]     * eg;
                ds[j][2] = s_S[wm + gid + 8][kk]     * eg;
                ds[j][3] = s_S[wm + gid + 8][kk + 1] * eg;
            }
            for (int t0 = 0; t0 < n; t0 += 16) {
                uint32_t va[4];
                gdn_ldsm4(va, s_VN[wm + a_r] + t0 + a_c);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t kb[2];
                    gdn_ldsm2(kb, s_B0[wn + j * 8 + b_r] + t0 + b_c);
                    gdn_mma_bf16_m16n8k16(ds[j], va, kb);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kk = k0 + wn + j * 8 + 2 * tid4;
                s_S[wm + gid][kk]         = ds[j][0];
                s_S[wm + gid][kk + 1]     = ds[j][1];
                s_S[wm + gid + 8][kk]     = ds[j][2];
                s_S[wm + gid + 8][kk + 1] = ds[j][3];
            }
            __syncthreads();
        }
    }

    if (update_state) {
        for (int idx = tid; idx < 128 * (GDN_WY_VB / 4); idx += 256) {
            const int k = idx >> 4, v4 = (idx & 15) << 2;
            *reinterpret_cast<float4*>(S_head + k * 128 + v0 + v4) =
                make_float4(s_S[v4 + 0][k], s_S[v4 + 1][k], s_S[v4 + 2][k], s_S[v4 + 3][k]);
        }
    }
}

// Warp-level vectorized RMSNormGated (4 heads/CTA, uint2 loads/stores, exact match to k_gdn_norm_gate).
__global__ void __launch_bounds__(128) k_gdn_norm_gate_fast(
    __nv_bfloat16* __restrict__ core, const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ w, int M, int nh_v, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + warp;
    if (idx >= M * nh_v) return;
    __nv_bfloat16* hr = core + static_cast<size_t>(idx) * 128;
    const __nv_bfloat16* zr = z + static_cast<size_t>(idx) * 128;
    const uint2 hv = reinterpret_cast<const uint2*>(hr)[lane];
    const uint2 zv = reinterpret_cast<const uint2*>(zr)[lane];
    const uint2 wv = reinterpret_cast<const uint2*>(w)[lane];
    const float2 h0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hv.x));
    const float2 h1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hv.y));
    const float2 z0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&zv.x));
    const float2 z1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&zv.y));
    const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
    const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
    const float ss = warp_sum(h0.x * h0.x + h0.y * h0.y + h1.x * h1.x + h1.y * h1.y);
    const float inv = rsqrtf(ss * (1.0f / 128.0f) + eps);
    auto do1 = [&](float h, float ww, float zz) {
        const float n = __bfloat162float(__float2bfloat16(h * inv));
        const float s = __bfloat162float(__float2bfloat16(n * ww));
        return s * sigmoid_f(zz);
    };
    uint2 ov;
    *reinterpret_cast<__nv_bfloat162*>(&ov.x) = __floats2bfloat162_rn(do1(h0.x, w0.x, z0.x), do1(h0.y, w0.y, z0.y));
    *reinterpret_cast<__nv_bfloat162*>(&ov.y) = __floats2bfloat162_rn(do1(h1.x, w1.x, z1.x), do1(h1.y, w1.y, z1.y));
    reinterpret_cast<uint2*>(hr)[lane] = ov;
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
            if (g.len >= 16 && snap_k == 0 && sc.wy_qe != nullptr) {
                const int nchunks = (g.len + GDN_WY_CH - 1) / GDN_WY_CH;
                k_gdn_conv1d_pf<<<dim3(((qkv_dim >> 2) + 255) / 256, (g.len + 15) / 16), 256, 0, st>>>(
                    sc.qkv_raw, w.conv1d, st_buf.conv, sc.qkv_conv, qkv_dim, g.slot, g.row0, g.pos0, g.len);
                if (update_state) {
                    k_gdn_conv1d_hist_pf<<<((qkv_dim >> 2) + 255) / 256, 256, 0, st>>>(
                        sc.qkv_raw, st_buf.conv, qkv_dim, g.slot, g.row0, g.pos0, g.len);
                }
                k_gdn_wy_intra<<<dim3(nh_v, nchunks), 256, 0, st>>>(
                    sc.qkv_conv, sc.a, sc.b, w.A_log, w.dt_bias,
                    sc.wy_qe, sc.wy_kd, sc.wy_w, sc.wy_u, sc.wy_aq, sc.wy_glast,
                    nh_k, nh_v, k0, v0, g.row0, g.len, nchunks, g.len);
                k_gdn_wy_state<<<dim3(nh_v, 2), 256, 0, st>>>(
                    sc.wy_qe, sc.wy_kd, sc.wy_w, sc.wy_u, sc.wy_aq, sc.wy_glast,
                    st_buf.S, sc.core, nh_v, g.slot, g.row0, g.pos0, g.len, nchunks, g.len, update_state);
            } else {
                k_gdn_conv1d_seg<<<(qkv_dim + 255) / 256, 256, 0, st>>>(
                    sc.qkv_raw, w.conv1d, st_buf.conv, snap_k > 0 ? st_buf.conv_snap : nullptr, sc.qkv_conv,
                    qkv_dim, g.slot, g.row0, g.pos0, g.len, snap_k, update_state);
                k_gdn_recurrent_seg<<<nh_v, 256, 0, st>>>(
                    sc.qkv_conv, sc.a, sc.b, w.A_log, w.dt_bias, st_buf.S, snap_k > 0 ? st_buf.S_snap : nullptr, sc.core,
                    nh_k, nh_v, k0, v0, g.slot, g.row0, g.pos0, g.len, snap_k, update_state);
            }
        }
        k_gdn_norm_gate_fast<<<(M * nh_v + 3) / 4, 128, 0, st>>>(sc.core, sc.z, w.norm, M, nh_v, geom::RMS_EPS);
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
