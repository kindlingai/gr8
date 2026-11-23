// Qwen3.8-Flash-Next MoE (Qwen4ExpTextSparseMoeBlock):
//   - 512 routed experts, top-10, EXP_I = 640 (5 x 128-col pmcgx blocks, exlr8-e4m3 v2 block-scale form)
//   - Full weight-paging support via engine/kernels/ptab.cuh (GR8_PTAB=0|1|2: OFF, IMAGE, SCATTER)
//   - Wide-load in-place K2/K3/K4 weight order (tr3_wide.cuh, GR8_QW38_WIDE=1) + grouped <=8-token MMA buckets
//   - Deterministic per-route reduction -> 0 byte diffs across GR8_PTAB=0/1/2 and m4sim == g6m4
//   - Exact router: logits = x @ W_router^T (FP32) -> softmax(FP32) -> topk(10) -> normalize -> BF16
//   - BF16 gated shared expert (640 channels, 32-aligned TP sharded):
//       sigmoid(x @ shared_expert_gate^T) * (down_proj(silu(gate_proj(x)) * up_proj(x)))
//   - Also supports BF16 routed experts for MTP layer L048 (mtp_exp.{gate_up, down}).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "tr3.cuh"
#include "ptab.cuh"
#include "qw38_hc.cuh"

namespace qw38 {

struct MoeWeights {
    const __nv_bfloat16* router = nullptr;       // [512, 2560] BF16
    const __nv_bfloat16* sh_gate = nullptr;      // [sh_i, 2560] BF16
    const __nv_bfloat16* sh_up = nullptr;        // [sh_i, 2560] BF16
    const __nv_bfloat16* sh_down = nullptr;      // [2560, sh_i] BF16
    const __nv_bfloat16* sh_exp_gate = nullptr;  // [1, 2560] BF16 (scalar sigmoid gate)
    int sh_i = 0;                                // local shared intermediate width (640 / 320 / 224|192 / 160)

    // Routed experts: exlr8 v2 pmcgx (layers 0..47)
    bool is_tr3 = true;
    const uint8_t* t_fc1 = nullptr;
    const uint8_t* t_fc2 = nullptr;
    const long long* t_tab = nullptr;            // [rows, 6] (rows = 512 or 513)
    const long long* t_kmix = nullptr;           // [rows]
    const __half* t_sg = nullptr;                // [2, 2560] (in, out signs)
    const __half* t_v1b = nullptr;               // [n_blks, 2, 20] (gate/up block scales over HB=20)
    const __half* t_v2 = nullptr;                // [n_blks, 3, 128] (svh gate, svh up, suh down)
    const __half* t_v3b = nullptr;               // [n_blks, 20] (down svh block scales over HB=20)
    const uint8_t* const* t_ptab = nullptr;      // GR8_PTAB=1|2 per-layer fixed-address piece table!
    int t_rows = 0;
    int t_eb = g6nr::EB;
    int fc1_wide = 0;
    int fc2_wide = 0;

    // Routed experts: BF16 3D tensors (MTP layer L048)
    const __nv_bfloat16* mtp_gate_up = nullptr;  // [512, 2 * sh_i, 2560]
    const __nv_bfloat16* mtp_down = nullptr;     // [512, 2560, sh_i]
};

struct MoeScratch {
    float* router_logits = nullptr;    // [max_M, 512] FP32
    int* topk_idx = nullptr;           // [max_M, 10] INT32
    float* topk_w = nullptr;           // [max_M, 10] FP32
    // Grouped routing buffers for PMCGX routed experts (up to max_M * 10 groups of <= 8 tokens):
    int* d_n_groups = nullptr;         // [1] INT32
    int* grp_exp = nullptr;            // [max_M * 10] INT32
    int* grp_cnt = nullptr;            // [max_M * 10] INT32
    int* grp_routes = nullptr;         // [max_M * 10 * 8] INT32
    // Per-route intermediate activations & outputs:
    float* fc1_part = nullptr;         // [8 * 10 * g6nr::EB * 2 * 2 * 128] FP32 (for M <= 8 KS=2 path)
    __half* fc1_act = nullptr;         // [max_M * 10, g6nr::EB, 128] FP16 (post silu(g)*u, H128(act*suh_d) ready!)
    float* fc2_part = nullptr;         // [max_M * 10, 2560] FP32 (deterministic per-route sum over b=0..nb-1)
    float* routed_acc = nullptr;       // [max_M, 2560] FP32 (MTP BF16 path)
    // Shared expert scratch:
    __nv_bfloat16* sh_gate_buf = nullptr; // [max_M, 640] BF16
    __nv_bfloat16* sh_up_buf = nullptr;   // [max_M, 640] BF16
    __nv_bfloat16* sh_out = nullptr;      // [max_M, 2560] BF16
    float* sh_scalar_gate = nullptr;      // [max_M] FP32
    // MTP BF16 expert scratch:
    __nv_bfloat16* mtp_gu_buf = nullptr;  // [max_M * 10, 2 * 640] BF16
    __nv_bfloat16* mtp_mid_buf = nullptr; // [max_M * 10, 640] BF16
    __nv_bfloat16* mtp_dn_buf = nullptr;  // [max_M * 10, 2560] BF16
};

// Fused router [512, 2560] + shared scalar gate [1, 2560] for small decode batches (M <= 8).
// 4 warps/CTA = 4 rows/CTA -> 129 CTAs (rows 0..511 = router, row 512 = sh_exp_gate).
template <int MAX_M>
__global__ void k_qw38_router_and_shgate(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ router,
                                         const __nv_bfloat16* __restrict__ sh_exp_gate,
                                         float* __restrict__ router_logits,
                                         float* __restrict__ sh_scalar_gate,
                                         int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n > geom::NEXP) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(
        (n < geom::NEXP) ? (router + static_cast<size_t>(n) * geom::HID) : sh_exp_gate);
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    constexpr int K8 = geom::HID >> 3; // 320
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
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float tot = warp_sum(acc[m]);
            if (lane == 0) {
                if (n < geom::NEXP) router_logits[static_cast<size_t>(m) * geom::NEXP + n] = tot;
                else sh_scalar_gate[m] = tot;
            }
        }
    }
}

// Fused shared expert gate + up + silu*mul for small decode batches (M <= 8).
// 4 warps/CTA = 4 output channels/CTA -> 160 CTAs at TP=1, 80 CTAs at TP=2.
template <int MAX_M>
__global__ void k_qw38_sh_gate_up_silu(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ sh_gate,
                                       const __nv_bfloat16* __restrict__ sh_up,
                                       __nv_bfloat16* __restrict__ out,
                                       int M, int sh_i) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= sh_i) return;
    const uint4* grow = reinterpret_cast<const uint4*>(sh_gate + static_cast<size_t>(n) * geom::HID);
    const uint4* urow = reinterpret_cast<const uint4*>(sh_up + static_cast<size_t>(n) * geom::HID);
    float ag[MAX_M], au[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }

    constexpr int K8 = geom::HID >> 3; // 320
    for (int i = lane; i < K8; i += 32) {
        const uint4 gw = __ldg(grow + i);
        const uint4 uw = __ldg(urow + i);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID)[i];
                ag[m] += dot_bf16x8(av, gw);
                au[m] += dot_bf16x8(av, uw);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float g = __bfloat162float(__float2bfloat16(warp_sum(ag[m])));
            const float u = __bfloat162float(__float2bfloat16(warp_sum(au[m])));
            if (lane == 0) out[static_cast<size_t>(m) * sh_i + n] = __float2bfloat16(silu_f(g) * u);
        }
    }
}

// Exact Qwen4ExpTextTopKRouter: softmax(FP32) over 512 logits -> top-10 -> normalize by sum -> round to BF16.
// Grid: (M), Block: 256 threads (each thread handles 2 experts: tid and tid + 256).
__global__ void k_qw38_router_topk10(const float* __restrict__ logits,
                                     int* __restrict__ topk_idx,
                                     float* __restrict__ topk_w, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* row = logits + static_cast<size_t>(m) * geom::NEXP;
    const int tid = threadIdx.x; // 0..255
    const int lane = tid & 31, warp = tid >> 5;

    float v0 = row[tid];
    float v1 = row[tid + 256];
    float mx = fmaxf(v0, v1);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? sm[lane] : -1e30f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, off));
        if (lane == 0) sm[0] = t;
    }
    __syncthreads();
    const float rmax = sm[0];

    const float e0 = expf(v0 - rmax);
    const float e1 = expf(v1 - rmax);
    float sum = warp_sum(e0 + e1);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? sm[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) sm[0] = 1.0f / t;
    }
    __syncthreads();
    const float inv_sum = sm[0];

    __shared__ float probs[geom::NEXP];
    probs[tid] = e0 * inv_sum;
    probs[tid + 256] = e1 * inv_sum;
    __syncthreads();

    __shared__ float sel_v[geom::TOPK];
    __shared__ int sel_i[geom::TOPK];
    for (int k = 0; k < geom::TOPK; ++k) {
        float p0 = probs[tid], p1 = probs[tid + 256];
        float bv = (p0 >= p1) ? p0 : p1;
        int bi = (p0 >= p1) ? tid : (tid + 256);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __shared__ float wv[8];
        __shared__ int wi[8];
        if (lane == 0) { wv[warp] = bv; wi[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float mv = wv[0]; int mi = wi[0];
            for (int w = 1; w < 8; ++w) {
                if (wv[w] > mv || (wv[w] == mv && wi[w] < mi)) { mv = wv[w]; mi = wi[w]; }
            }
            sel_v[k] = mv;
            sel_i[k] = mi;
            probs[mi] = -1.0f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < geom::TOPK; ++k) wsum += sel_v[k];
        const float inv = (wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
        for (int k = 0; k < geom::TOPK; ++k) {
            topk_idx[m * geom::TOPK + k] = sel_i[k];
            topk_w[m * geom::TOPK + k] = __bfloat162float(__float2bfloat16(sel_v[k] * inv));
        }
    }
}

// Group `n_routes = M * 10` routes by expert into chunks of <= 8 routes sharing the same expert.
// Single CTA of 512 threads (1 thread per expert), zero host sync!
__global__ void k_qw38_moe_build_groups(const int* __restrict__ topk_idx,
                                        int* __restrict__ d_n_groups,
                                        int* __restrict__ grp_exp,
                                        int* __restrict__ grp_cnt,
                                        int* __restrict__ grp_routes,
                                        int n_routes) {
    const int tid = threadIdx.x; // 0..511 = expert id
    __shared__ int cnt[geom::NEXP];
    __shared__ int g_off[geom::NEXP];
    __shared__ int cur[geom::NEXP];

    cnt[tid] = 0;
    cur[tid] = 0;
    __syncthreads();

    for (int r = tid; r < n_routes; r += geom::NEXP) {
        atomicAdd(&cnt[topk_idx[r]], 1);
    }
    __syncthreads();

    // Exclusive prefix sum of ceil(cnt[e] / 8) over 512 experts (16 warps x 32 lanes)
    const int ng = (cnt[tid] + 7) >> 3;
    const int lane = tid & 31, warp = tid >> 5;
    int val = ng;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, val, off);
        if (lane >= off) val += n;
    }
    __shared__ int wsum[16];
    if (lane == 31) wsum[warp] = val;
    __syncthreads();
    if (warp == 0) {
        int wv = (lane < 16) ? wsum[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            int n = __shfl_up_sync(0xffffffffu, wv, off);
            if (lane >= off) wv += n;
        }
        if (lane < 16) wsum[lane] = wv;
    }
    __syncthreads();
    const int base = (warp > 0 ? wsum[warp - 1] : 0) + (val - ng);
    g_off[tid] = base;
    if (tid == geom::NEXP - 1) d_n_groups[0] = base + ng;

    // Emit group metadata for this expert
    const int c_tot = cnt[tid];
    for (int gi = 0; gi < ng; ++gi) {
        const int g = base + gi;
        const int rem = c_tot - gi * 8;
        grp_exp[g] = tid;
        grp_cnt[g] = (rem < 8) ? rem : 8;
    }
    __syncthreads();

    // Scatter routes into their expert's group slots
    for (int r = tid; r < n_routes; r += geom::NEXP) {
        const int e = topk_idx[r];
        const int pos = atomicAdd(&cur[e], 1);
        const int g = g_off[e] + (pos >> 3);
        grp_routes[g * 8 + (pos & 7)] = r;
    }
}

// Fast 1-token group builder for M == 1 (10 distinct experts -> 10 single-route groups, 1 warp!).
__global__ void k_qw38_moe_groups_m1(const int* __restrict__ topk_idx,
                                     int* __restrict__ d_n_groups,
                                     int* __restrict__ grp_exp,
                                     int* __restrict__ grp_cnt,
                                     int* __restrict__ grp_routes) {
    const int k = threadIdx.x;
    if (k < geom::TOPK) {
        grp_exp[k] = topk_idx[k];
        grp_cnt[k] = 1;
        grp_routes[k * 8] = k;
    }
    if (k == 0) d_n_groups[0] = geom::TOPK;
}

// =================================================================================================
// PMCGX Wide-Load & Stock-Order MMA Helpers (100% bit-exact with tr3.cuh / tr3_wide.cuh!)
// =================================================================================================
template <int BITS>
__device__ __forceinline__ void pmcgx_fc1_hb_stock(const uint8_t* mw, int hb, int warp, int lane,
                                                   const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const size_t tb = static_cast<size_t>(32 * BITS);
    Tr3Lane<BITS, true, TR3_PMCGX> tl;
    tl.init(lane);
    const int ks0 = hb * 8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t* tile = mw + (static_cast<size_t>(ks0 + p) * 8 + warp) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
        mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

template <int BITS>
__device__ __forceinline__ void pmcgx_fc1_hb_wide(const uint8_t* mw, int hb, int warp, int lane,
                                                  const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane + 31) & 31;
    if constexpr (BITS == 4) {
        const uint8_t* g4 = mw + static_cast<size_t>(hb) * TR4W_GROUP + (warp * 2) * 512 + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 wh = tr4w_ld128(g4 + h * 512);
            const u32 w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const u32 ch = w[p];
                const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                u32 f0[2], f1[2];
                Tr3Lane<4, true, TR3_PMCGX>::decode_m((static_cast<u64>(pc) << 32) | ch, 28, f0, f1);
                const int kp = h * 4 + p;
                const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + kp * 16 + 2 * col) : 0u;
                const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + kp * 16 + 2 * col + 8) : 0u;
                mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
            }
        }
    } else {
        const uint8_t* g0 = mw + static_cast<size_t>(hb) * (2048 * BITS) + static_cast<size_t>(warp) * 256 + lane * 8;
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = ld_nc64(g0 + n * 2048);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<BITS, true, TR3_PMCGX>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

__device__ __forceinline__ void pmcgx_fc1_hb(const uint8_t* mw, int bits, int wide, int hb, int warp, int lane,
                                             const __half (&x_rot)[8][136], int cnt, float (&c)[4]) {
    if (wide) {
        if (bits == 3) pmcgx_fc1_hb_wide<3>(mw, hb, warp, lane, x_rot, cnt, c);
        else if (bits == 4) pmcgx_fc1_hb_wide<4>(mw, hb, warp, lane, x_rot, cnt, c);
        else pmcgx_fc1_hb_wide<2>(mw, hb, warp, lane, x_rot, cnt, c);
    } else {
        if (bits == 3) pmcgx_fc1_hb_stock<3>(mw, hb, warp, lane, x_rot, cnt, c);
        else if (bits == 4) pmcgx_fc1_hb_stock<4>(mw, hb, warp, lane, x_rot, cnt, c);
        else pmcgx_fc1_hb_stock<2>(mw, hb, warp, lane, x_rot, cnt, c);
    }
}

template <int BITS>
__device__ __forceinline__ void pmcgx_fc2_ntile_stock(const uint8_t* mw, int n_tile, int lane,
                                                      const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const size_t tb = static_cast<size_t>(32 * BITS);
    Tr3Lane<BITS, true, TR3_PMCGX> tl;
    tl.init(lane);
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col + 8) : 0u;
        mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

template <int BITS>
__device__ __forceinline__ void pmcgx_fc2_ntile_wide(const uint8_t* mw, int n_tile, int lane,
                                                     const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = (row0 < cnt);
    const int prev_lane = (lane + 31) & 31;
    if constexpr (BITS == 4) {
        const uint8_t* wg4 = mw + static_cast<size_t>(n_tile) * TR4W_NT + lane * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 wh = tr4w_ld128(wg4 + h * 512);
            const u32 w[4] = {wh.x, wh.y, wh.z, wh.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const u32 ch = w[p];
                const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
                u32 f0[2], f1[2];
                Tr3Lane<4, true, TR3_PMCGX>::decode_m((static_cast<u64>(pc) << 32) | ch, 28, f0, f1);
                const int ks = h * 4 + p;
                const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col) : 0u;
                const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col + 8) : 0u;
                mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
            }
        }
    } else {
        const uint8_t* wg = mw + static_cast<size_t>(n_tile) * (8 * 32 * BITS) + lane * 8;
        u64 w[3] = {0, 0, 0};
        #pragma unroll
        for (int n = 0; n < BITS; ++n) w[n] = ld_nc64(wg + n * 256);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const u32 ch = trw_chunk<BITS>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<BITS, true, TR3_PMCGX>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
            const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col) : 0u;
            const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + p * 16 + 2 * col + 8) : 0u;
            mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
        }
    }
}

__device__ __forceinline__ void pmcgx_fc2_ntile(const uint8_t* mw, int bits, int wide, int n_tile, int lane,
                                                const __half (&x2)[8][136], int cnt, float (&c)[4]) {
    if (wide) {
        if (bits == 3) pmcgx_fc2_ntile_wide<3>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 4) pmcgx_fc2_ntile_wide<4>(mw, n_tile, lane, x2, cnt, c);
        else pmcgx_fc2_ntile_wide<2>(mw, n_tile, lane, x2, cnt, c);
    } else {
        if (bits == 3) pmcgx_fc2_ntile_stock<3>(mw, n_tile, lane, x2, cnt, c);
        else if (bits == 4) pmcgx_fc2_ntile_stock<4>(mw, n_tile, lane, x2, cnt, c);
        else pmcgx_fc2_ntile_stock<2>(mw, n_tile, lane, x2, cnt, c);
    }
}

// =================================================================================================
// Grouped FC1 Path 1 (Small M <= 8, KS = 2, separate gate/up CTAs -> saturates all 48 SMs on TP=1..4!)
// Grid: (grid_G * eb * 4), Block: 256 (8 warps = 8 n-tiles).
// =================================================================================================
__global__ void __launch_bounds__(256) k_qw38_moe_fc1_ks2(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ t_tab,
    const long long* __restrict__ t_kmix,
    const uint8_t* __restrict__ t_fc1,
    const uint8_t* const* __restrict__ t_ptab,
    const __half* __restrict__ t_sg,
    const __half* __restrict__ t_v1b,
    float* __restrict__ fc1_part,
    int eb, int wide) {
    const int idx = blockIdx.x;
    const int ks = idx & 1;
    const int mat = (idx >> 1) & 1;
    const int rem = idx >> 2;
    const int b = rem % eb;
    const int g = rem / eb;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const int cnt = grp_cnt[g];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ int s_tok[8];
    __shared__ int s_route[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_tok[tid] = r / geom::TOPK;
    }
    __syncthreads();

    const uint8_t* mw = nullptr;
    const int bits = tr3_fc1_unit(t_fc1, t_ptab, t_kmix, e, tr, 2 * b + mat, mw);

    __shared__ __half x_rot[8][136];
    float c[4] = {0.f, 0.f, 0.f, 0.f};

    const int hb0 = ks * (geom::HB / 2);
    const int hb1 = hb0 + (geom::HB / 2);
    for (int hb = hb0; hb < hb1; ++hb) {
        if (warp < cnt) {
            const float bscale = __half2float(t_v1b[(brow * 2 + mat) * geom::HB + hb]);
            const __nv_bfloat16* xm = x + static_cast<size_t>(s_tok[warp]) * geom::HID + hb * 128 + lane * 4;
            const __half* sgp = t_sg + hb * 128 + lane * 4;
            const uint2 xv = *reinterpret_cast<const uint2*>(xm);
            const uint2 sv = *reinterpret_cast<const uint2*>(sgp);
            const float2 xf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
            const float2 xf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
            const float2 sf0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
            const float2 sf1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
            float v[4] = {xf0.x * (sf0.x * bscale), xf0.y * (sf0.y * bscale),
                          xf1.x * (sf1.x * bscale), xf1.y * (sf1.y * bscale)};
            had128(v, lane);
            __half* dst = x_rot[warp] + lane * 4;
            dst[0] = __float2half_rn(v[0]);
            dst[1] = __float2half_rn(v[1]);
            dst[2] = __float2half_rn(v[2]);
            dst[3] = __float2half_rn(v[3]);
        }
        __syncthreads();
        pmcgx_fc1_hb(mw, bits, wide, hb, warp, lane, x_rot, cnt, c);
        __syncthreads();
    }

    const int row0 = lane >> 2, col = lane & 3;
    const int t0 = 2 * col, t1 = 2 * col + 1;
    if (t0 < cnt) {
        const size_t base = (((static_cast<size_t>(s_route[t0]) * eb + b) * 2 + mat) * 2 + ks) * 128;
        fc1_part[base + warp * 16 + row0] = c[0];
        fc1_part[base + warp * 16 + row0 + 8] = c[2];
    }
    if (t1 < cnt) {
        const size_t base = (((static_cast<size_t>(s_route[t1]) * eb + b) * 2 + mat) * 2 + ks) * 128;
        fc1_part[base + warp * 16 + row0] = c[1];
        fc1_part[base + warp * 16 + row0 + 8] = c[3];
    }
}

// Epilogue for KS=2 FC1: sum 2 splits -> had128 * svh -> silu(g)*u * suh_d -> had128 -> fc1_act.
// Grid: (M * TOPK * eb), Block: 32 (1 warp per (route, b)).
__global__ void k_qw38_moe_fc1_epilogue_ks2(const float* __restrict__ fc1_part,
                                            const int* __restrict__ topk_idx,
                                            const long long* __restrict__ t_tab,
                                            const __half* __restrict__ t_v2,
                                            __half* __restrict__ fc1_act,
                                            int M, int eb) {
    const int idx = blockIdx.x;
    const int b = idx % eb;
    const int route = idx / eb;
    if (route >= M * geom::TOPK) return;
    const int e = topk_idx[route];
    const long long* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int lane = threadIdx.x;
    const size_t gbase = ((static_cast<size_t>(route) * eb + b) * 2 + 0) * 2 * 128;
    const size_t ubase = ((static_cast<size_t>(route) * eb + b) * 2 + 1) * 2 * 128;

    const float4 g0 = *reinterpret_cast<const float4*>(fc1_part + gbase + lane * 4);
    const float4 g1 = *reinterpret_cast<const float4*>(fc1_part + gbase + 128 + lane * 4);
    const float4 u0 = *reinterpret_cast<const float4*>(fc1_part + ubase + lane * 4);
    const float4 u1 = *reinterpret_cast<const float4*>(fc1_part + ubase + 128 + lane * 4);

    float vg[4] = {__half2float(__float2half_rn(g0.x + g1.x)), __half2float(__float2half_rn(g0.y + g1.y)),
                   __half2float(__float2half_rn(g0.z + g1.z)), __half2float(__float2half_rn(g0.w + g1.w))};
    float vu[4] = {__half2float(__float2half_rn(u0.x + u1.x)), __half2float(__float2half_rn(u0.y + u1.y)),
                   __half2float(__float2half_rn(u0.z + u1.z)), __half2float(__float2half_rn(u0.w + u1.w))};
    had128(vg, lane);
    had128(vu, lane);

    const __half* svh_g = t_v2 + (brow * 3 + 0) * 128 + lane * 4;
    const __half* svh_u = t_v2 + (brow * 3 + 1) * 128 + lane * 4;
    const __half* suh_d = t_v2 + (brow * 3 + 2) * 128 + lane * 4;
    float va[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float g = __half2float(__float2half_rn(vg[i] * __half2float(svh_g[i])));
        const float u = __half2float(__float2half_rn(vu[i] * __half2float(svh_u[i])));
        const float act = __half2float(__float2half_rn(silu_f(g) * u));
        va[i] = act * __half2float(suh_d[i]);
    }
    had128(va, lane);
    __half* dst = fc1_act + (static_cast<size_t>(route) * eb + b) * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(va[i]);
}

// =================================================================================================
// Grouped FC1 Path 2 (M > 8, e.g. prefill): 1 CTA per (group, b), <= 8 tokens per weight load!
// Grid: (grid_G * eb), Block: 256 (8 warps = 8 n-tiles). Zero fc1_part HBM traffic!
// =================================================================================================
__global__ void __launch_bounds__(256) k_qw38_moe_fc1_grp(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const long long* __restrict__ t_tab,
    const long long* __restrict__ t_kmix,
    const uint8_t* __restrict__ t_fc1,
    const uint8_t* const* __restrict__ t_ptab,
    const __half* __restrict__ t_sg,
    const __half* __restrict__ t_v1b,
    const __half* __restrict__ t_v2,
    __half* __restrict__ fc1_act,
    int eb, int wide) {
    const int idx = blockIdx.x;
    const int b = idx % eb;
    const int g = idx / eb;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const int cnt = grp_cnt[g];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = lane >> 2, col = lane & 3;

    __shared__ int s_tok[8];
    __shared__ int s_route[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_tok[tid] = r / geom::TOPK;
    }
    __syncthreads();

    __shared__ __half x_rot[8][136];
    __shared__ float y_gate[8][128];
    __shared__ float y_up[8][128];

    for (int mat = 0; mat < 2; ++mat) {
        const uint8_t* mw = nullptr;
        const int bits = tr3_fc1_unit(t_fc1, t_ptab, t_kmix, e, tr, 2 * b + mat, mw);
        float c[4] = {0.f, 0.f, 0.f, 0.f};

        for (int hb = 0; hb < geom::HB; ++hb) {
            if (warp < cnt) {
                const float bscale = __half2float(t_v1b[(brow * 2 + mat) * geom::HB + hb]);
                const __nv_bfloat16* xm = x + static_cast<size_t>(s_tok[warp]) * geom::HID + hb * 128 + lane * 4;
                const __half* sgp = t_sg + hb * 128 + lane * 4;
                const uint2 xv = *reinterpret_cast<const uint2*>(xm);
                const uint2 sv = *reinterpret_cast<const uint2*>(sgp);
                const float2 xf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
                const float2 xf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
                const float2 sf0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
                const float2 sf1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
                float v[4] = {xf0.x * (sf0.x * bscale), xf0.y * (sf0.y * bscale),
                              xf1.x * (sf1.x * bscale), xf1.y * (sf1.y * bscale)};
                had128(v, lane);
                __half* dst = x_rot[warp] + lane * 4;
                dst[0] = __float2half_rn(v[0]);
                dst[1] = __float2half_rn(v[1]);
                dst[2] = __float2half_rn(v[2]);
                dst[3] = __float2half_rn(v[3]);
            }
            __syncthreads();
            pmcgx_fc1_hb(mw, bits, wide, hb, warp, lane, x_rot, cnt, c);
            __syncthreads();
        }

        float (*ydst)[128] = (mat == 0) ? y_gate : y_up;
        const int t0 = 2 * col, t1 = 2 * col + 1;
        if (t0 < cnt) {
            ydst[t0][warp * 16 + row0] = __half2float(__float2half_rn(c[0]));
            ydst[t0][warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[2]));
        }
        if (t1 < cnt) {
            ydst[t1][warp * 16 + row0] = __half2float(__float2half_rn(c[1]));
            ydst[t1][warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[3]));
        }
        __syncthreads();
    }

    if (warp < cnt) {
        float vg[4], vu[4];
        const __half* svh_g = t_v2 + (brow * 3 + 0) * 128 + lane * 4;
        const __half* svh_u = t_v2 + (brow * 3 + 1) * 128 + lane * 4;
        const __half* suh_d = t_v2 + (brow * 3 + 2) * 128 + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = y_gate[warp][lane * 4 + i];
            vu[i] = y_up[warp][lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        float va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float g = __half2float(__float2half_rn(vg[i] * __half2float(svh_g[i])));
            const float u = __half2float(__float2half_rn(vu[i] * __half2float(svh_u[i])));
            const float act = __half2float(__float2half_rn(silu_f(g) * u));
            va[i] = act * __half2float(suh_d[i]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128 + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(va[i]);
    }
}

// =================================================================================================
// Unified Grouped FC2 (down projection): 1 CTA per (group g, hb in 0..19).
// Loops b = 0..nb-1 in exact ascending order, loading each wide n-tile ONCE for all <= 8 tokens,
// accumulating sum_b(rw * y_b) in registers and writing fc2_part[route, hb*128..+127] ONCE (5x less HBM!).
// =================================================================================================
__global__ void __launch_bounds__(256) k_qw38_moe_fc2_grp(
    const __half* __restrict__ fc1_act,
    const int* __restrict__ d_n_groups,
    const int* __restrict__ grp_exp,
    const int* __restrict__ grp_cnt,
    const int* __restrict__ grp_routes,
    const float* __restrict__ topk_w,
    const long long* __restrict__ t_tab,
    const uint8_t* __restrict__ t_fc2,
    const uint8_t* const* __restrict__ t_ptab,
    const __half* __restrict__ t_sg,
    const __half* __restrict__ t_v3b,
    float* __restrict__ fc2_part,
    int eb, int wide) {
    const int idx = blockIdx.x;
    const int hb = idx % geom::HB;
    const int g = idx / geom::HB;
    if (g >= d_n_groups[0]) return;

    const int e = grp_exp[g];
    const long long* tr = t_tab + 6 * e;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int cnt = grp_cnt[g];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    __shared__ int s_route[8];
    __shared__ float s_rw[8];
    if (tid < cnt) {
        const int r = grp_routes[g * 8 + tid];
        s_route[tid] = r;
        s_rw[tid] = topk_w[r];
    }
    __syncthreads();

    float sg_v[4];
    if (warp < cnt) {
        const __half* sgp = t_sg + geom::HID + hb * 128 + lane * 4;
        const uint2 sv = *reinterpret_cast<const uint2*>(sgp);
        const float2 sf0 = __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
        const float2 sf1 = __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
        sg_v[0] = sf0.x; sg_v[1] = sf0.y; sg_v[2] = sf1.x; sg_v[3] = sf1.y;
    }

    __shared__ __half x2_sm[8][136];
    __shared__ float y_raw[8][128];
    float sum_v[4] = {0.f, 0.f, 0.f, 0.f};

    for (int b = 0; b < nb; ++b) {
        if (warp < cnt) {
            const __half* x2 = fc1_act + (static_cast<size_t>(s_route[warp]) * eb + b) * 128 + lane * 4;
            *reinterpret_cast<uint2*>(x2_sm[warp] + lane * 4) = *reinterpret_cast<const uint2*>(x2);
        }
        __syncthreads();

        const uint8_t* mw = tr3_fc2_block(t_fc2, t_ptab, e, tr, b, bits);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        pmcgx_fc2_ntile(mw, bits, wide, n_tile, lane, x2_sm, cnt, c);

        const int t0 = 2 * col, t1 = 2 * col + 1;
        if (t0 < cnt) {
            y_raw[t0][warp * 16 + row0] = __half2float(__float2half_rn(c[0]));
            y_raw[t0][warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[2]));
        }
        if (t1 < cnt) {
            y_raw[t1][warp * 16 + row0] = __half2float(__float2half_rn(c[1]));
            y_raw[t1][warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[3]));
        }
        __syncthreads();

        if (warp < cnt) {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
            had128(v, lane);
            const size_t brow = static_cast<size_t>(tr[2]) + b;
            const float bscale = __half2float(t_v3b[brow * geom::HB + hb]);
            const float rw = s_rw[warp];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum_v[i] += rw * (v[i] * (sg_v[i] * bscale));
            }
        }
        __syncthreads();
    }

    if (warp < cnt) {
        float* dst = fc2_part + static_cast<size_t>(s_route[warp]) * geom::HID + hb * 128 + lane * 4;
        *reinterpret_cast<float4*>(dst) = make_float4(sum_v[0], sum_v[1], sum_v[2], sum_v[3]);
    }
}

// Deterministic vectorized reduction of per-route fc2_part[m*10 + 0..9] + sigmoid(sh_scalar_gate) * sh_out -> out_part [M, 2560] BF16.
__global__ void k_qw38_moe_combine_det(const float* __restrict__ fc2_part,
                                       const __nv_bfloat16* __restrict__ sh_out,
                                       const float* __restrict__ sh_scalar_gate,
                                       __nv_bfloat16* __restrict__ out_part,
                                       int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float sg = sigmoid_f(sh_scalar_gate[m]);
    const __nv_bfloat16* s = sh_out + static_cast<size_t>(m) * geom::HID;
    __nv_bfloat16* o = out_part + static_cast<size_t>(m) * geom::HID;
    const float* base = fc2_part + static_cast<size_t>(m * geom::TOPK) * geom::HID;

    constexpr int D4 = geom::HID >> 2; // 640
    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int k = 0; k < geom::TOPK; ++k) {
            const float4 v = reinterpret_cast<const float4*>(base + static_cast<size_t>(k) * geom::HID)[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        const uint2 sv = reinterpret_cast<const uint2*>(s)[i];
        const float2 sf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sv.x));
        const float2 sf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sv.y));
        const float s0 = __bfloat162float(__float2bfloat16(sg * sf0.x));
        const float s1 = __bfloat162float(__float2bfloat16(sg * sf0.y));
        const float s2 = __bfloat162float(__float2bfloat16(sg * sf1.x));
        const float s3 = __bfloat162float(__float2bfloat16(sg * sf1.y));
        uint2 ov;
        *reinterpret_cast<__nv_bfloat162*>(&ov.x) = __floats2bfloat162_rn(acc.x + s0, acc.y + s1);
        *reinterpret_cast<__nv_bfloat162*>(&ov.y) = __floats2bfloat162_rn(acc.z + s2, acc.w + s3);
        reinterpret_cast<uint2*>(o)[i] = ov;
    }
}

// Shared expert activation: sh_gate = bf16(silu(sh_gate) * sh_up) (for large M > 8 cuBLAS path)
__global__ void k_silu_mul_bf16(__nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        gate[i] = __float2bfloat16(silu_f(g) * u);
    }
}

// Combine routed_acc (FP32) + sigmoid(sh_scalar_gate) * sh_out (BF16) -> out_part [M, 2560] BF16 (MTP path)
__global__ void k_qw38_moe_combine(const float* __restrict__ routed_acc,
                                   const __nv_bfloat16* __restrict__ sh_out,
                                   const float* __restrict__ sh_scalar_gate,
                                   __nv_bfloat16* __restrict__ out_part, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float sg = sigmoid_f(sh_scalar_gate[m]);
    const float* r = routed_acc + static_cast<size_t>(m) * geom::HID;
    const __nv_bfloat16* s = sh_out + static_cast<size_t>(m) * geom::HID;
    __nv_bfloat16* o = out_part + static_cast<size_t>(m) * geom::HID;
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        const float sv = __bfloat162float(__float2bfloat16(sg * __bfloat162float(s[d])));
        o[d] = __float2bfloat16(r[d] + sv);
    }
}

// MTP BF16 routed expert gather-activate-add (for layer L048 where experts are BF16 3D tensors)
__global__ void k_qw38_mtp_exp_act_add(const __nv_bfloat16* __restrict__ gu_out,
                                       __nv_bfloat16* __restrict__ mid_out,
                                       int sh_i) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sh_i) {
        const float g = __bfloat162float(gu_out[i]);
        const float u = __bfloat162float(gu_out[sh_i + i]);
        mid_out[i] = __float2bfloat16(silu_f(g) * u);
    }
}

__global__ void k_qw38_mtp_exp_accum(const __nv_bfloat16* __restrict__ dn_out,
                                     const float* __restrict__ topk_w,
                                     float* __restrict__ routed_acc,
                                     int route, int m) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < geom::HID) {
        routed_acc[static_cast<size_t>(m) * geom::HID + d] += topk_w[route] * __bfloat162float(dn_out[d]);
    }
}

// Execute one Qwen3.8 MoE block (512 routed + gated shared expert) -> out_part [M, 2560] BF16.
inline void moe_forward(cublasHandle_t blas, const MoeWeights& w, MoeScratch& sc,
                        const __nv_bfloat16* x, __nv_bfloat16* out_part, int M, cudaStream_t st) {
    // 1. Router logits + shared expert (fused coalesced uint4 GEMV for small decode M <= 8, cuBLAS for prefill)
    if (M <= 8) {
        const int rgrid = (geom::NEXP + 1 + 3) / 4; // 129 CTAs
        const int sgrid = (w.sh_i + 3) / 4;
        if (M == 1) {
            k_qw38_router_and_shgate<1><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            k_qw38_sh_gate_up_silu<1><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else if (M <= 4) {
            k_qw38_router_and_shgate<4><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            k_qw38_sh_gate_up_silu<4><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        } else {
            k_qw38_router_and_shgate<8><<<rgrid, 128, 0, st>>>(x, w.router, w.sh_exp_gate, sc.router_logits, sc.sh_scalar_gate, M);
            k_qw38_sh_gate_up_silu<8><<<sgrid, 128, 0, st>>>(x, w.sh_gate, w.sh_up, sc.sh_gate_buf, M, w.sh_i);
        }
        k_qw38_router_topk10<<<M, 256, 0, st>>>(sc.router_logits, sc.topk_idx, sc.topk_w, M);
        launch_bf16_gemv_smallm<false>(sc.sh_gate_buf, w.sh_down, sc.sh_out, M, geom::HID, w.sh_i, st);
    } else {
        cublasSetStream(blas, st);
        gemm_bf16_nt_f32(blas, M, geom::NEXP, geom::HID, x, w.router, sc.router_logits);
        k_qw38_router_topk10<<<M, 256, 0, st>>>(sc.router_logits, sc.topk_idx, sc.topk_w, M);
        gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_gate, sc.sh_gate_buf);
        gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_up, sc.sh_up_buf);
        {
            const int n = M * w.sh_i;
            k_silu_mul_bf16<<<(n + 255) / 256, 256, 0, st>>>(sc.sh_gate_buf, sc.sh_up_buf, n);
        }
        gemm_bf16_nt(blas, M, geom::HID, w.sh_i, sc.sh_gate_buf, w.sh_down, sc.sh_out);
        gemm_bf16_nt_f32(blas, M, 1, geom::HID, x, w.sh_exp_gate, sc.sh_scalar_gate);
    }

    // 2. Routed experts + 3. Combine routed + gated shared -> out_part [M, 2560] BF16
    if (w.is_tr3) {
        const int n_routes = M * geom::TOPK;
        int grid_G = 0;
        if (M == 1) {
            k_qw38_moe_groups_m1<<<1, 32, 0, st>>>(sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes);
            grid_G = geom::TOPK;
        } else {
            k_qw38_moe_build_groups<<<1, geom::NEXP, 0, st>>>(
                sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes, n_routes);
            grid_G = std::min(n_routes, geom::NEXP + (n_routes + 7) / 8);
        }

        if (M <= 8 && sc.fc1_part) {
            k_qw38_moe_fc1_ks2<<<grid_G * w.t_eb * 4, 256, 0, st>>>(
                x, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
                w.t_tab, w.t_kmix, w.t_fc1, w.t_ptab, w.t_sg, w.t_v1b, sc.fc1_part, w.t_eb, w.fc1_wide);
            k_qw38_moe_fc1_epilogue_ks2<<<n_routes * w.t_eb, 32, 0, st>>>(
                sc.fc1_part, sc.topk_idx, w.t_tab, w.t_v2, sc.fc1_act, M, w.t_eb);
        } else {
            k_qw38_moe_fc1_grp<<<grid_G * w.t_eb, 256, 0, st>>>(
                x, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
                w.t_tab, w.t_kmix, w.t_fc1, w.t_ptab, w.t_sg, w.t_v1b, w.t_v2, sc.fc1_act, w.t_eb, w.fc1_wide);
        }

        k_qw38_moe_fc2_grp<<<grid_G * geom::HB, 256, 0, st>>>(
            sc.fc1_act, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_routes,
            sc.topk_w, w.t_tab, w.t_fc2, w.t_ptab, w.t_sg, w.t_v3b, sc.fc2_part, w.t_eb, w.fc2_wide);

        k_qw38_moe_combine_det<<<M, 256, 0, st>>>(sc.fc2_part, sc.sh_out, sc.sh_scalar_gate, out_part, M);
    } else if (w.mtp_gate_up && w.mtp_down) {
        cublasSetStream(blas, st);
        cudaMemsetAsync(sc.routed_acc, 0, static_cast<size_t>(M) * geom::HID * sizeof(float), st);
        std::vector<int> h_idx(static_cast<size_t>(M) * geom::TOPK);
        cudaMemcpyAsync(h_idx.data(), sc.topk_idx, h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        for (int m = 0; m < M; ++m) {
            const __nv_bfloat16* xm = x + static_cast<size_t>(m) * geom::HID;
            for (int k = 0; k < geom::TOPK; ++k) {
                const int route = m * geom::TOPK + k;
                const int e = h_idx[static_cast<size_t>(route)];
                const __nv_bfloat16* gu_w = w.mtp_gate_up + static_cast<size_t>(e) * (2 * w.sh_i) * geom::HID;
                const __nv_bfloat16* dn_w = w.mtp_down + static_cast<size_t>(e) * geom::HID * w.sh_i;
                gemm_bf16_nt(blas, 1, 2 * w.sh_i, geom::HID, xm, gu_w, sc.mtp_gu_buf);
                k_qw38_mtp_exp_act_add<<<(w.sh_i + 255) / 256, 256, 0, st>>>(sc.mtp_gu_buf, sc.mtp_mid_buf, w.sh_i);
                gemm_bf16_nt(blas, 1, geom::HID, w.sh_i, sc.mtp_mid_buf, dn_w, sc.mtp_dn_buf);
                k_qw38_mtp_exp_accum<<<(geom::HID + 255) / 256, 256, 0, st>>>(sc.mtp_dn_buf, sc.topk_w, sc.routed_acc, route, m);
            }
        }
        k_qw38_moe_combine<<<M, 256, 0, st>>>(sc.routed_acc, sc.sh_out, sc.sh_scalar_gate, out_part, M);
    } else {
        cudaMemsetAsync(sc.routed_acc, 0, static_cast<size_t>(M) * geom::HID * sizeof(float), st);
        k_qw38_moe_combine<<<M, 256, 0, st>>>(sc.routed_acc, sc.sh_out, sc.sh_scalar_gate, out_part, M);
    }
}

} // namespace qw38
