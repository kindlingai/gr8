// Qwen3.8-Flash-Next MoE (Qwen4ExpTextSparseMoeBlock):
//   - 512 routed experts, top-10, EXP_I = 640 (5 x 128-col pmcgx blocks, exlr8-e4m3 v2 block-scale form)
//   - Full weight-paging support via engine/kernels/ptab.cuh (GR8_PTAB=0|1|2: OFF, IMAGE, SCATTER)
//   - Exact router: logits = x @ W_router^T (FP32) -> softmax(FP32) -> topk(10) -> normalize -> BF16
//   - BF16 gated shared expert (640 channels, 32-aligned TP sharded):
//       sigmoid(x @ shared_expert_gate^T) * (down_proj(silu(gate_proj(x)) * up_proj(x)))
//   - Also supports BF16 routed experts for MTP layer L048 (mtp_exp.{gate_up, down}).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
    // Grouped routing buffers for PMCGX routed experts:
    // Up to max_M * 10 routes grouped by expert -> at most max_M * 10 tasks of (expert, tok_idx, weight)
    // For fast batched execution: per-expert buckets
    int* exp_counts = nullptr;         // [512] INT32
    int* exp_offsets = nullptr;        // [513] INT32
    int* active_exps = nullptr;        // [512] INT32
    int* n_active = nullptr;           // [1] INT32
    int* route_tok = nullptr;          // [max_M * 10] INT32
    float* route_w = nullptr;          // [max_M * 10] FP32
    // Per-route intermediate activations & outputs:
    // fc1_out[route, block, 128] FP16 -> act[route, block, 128] FP16 -> fc2_part[route, block, 2560] FP32
    __half* fc1_act = nullptr;         // [max_M * 10, g6nr::EB, 128] FP16 (post silu(g)*u, H128(act*suh_d) ready!)
    float* fc2_part = nullptr;         // [max_M * 10, g6nr::EB, 2560] FP32 (deterministic non-atomic per-(route,b) partials)
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

    // Exact top-10 selection on thread 0 (or parallel 10-pass warp reduction)
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
            // Match HF: router_top_value.to(router_logits.dtype) -> BF16
            topk_w[m * geom::TOPK + k] = __bfloat162float(__float2bfloat16(sel_v[k] * inv));
        }
    }
}

// =================================================================================================
// PMCGX Routed Expert Kernels (exlr8-e4m3 v2 block-scale + GR8_PTAB=0|1|2 piece table!)
//
// Exact linear op per 128-col block b of expert e (matching engine/kernels/tr3.cuh):
//   H128(v) = Sylvester Hadamard(128) scaled by 1/sqrt(128).
//   FC1 (gate: mat=0, up: mat=1):
//     x_rot[k] = fp16(H128(x * suh_m)[k]),  where suh_m[k] = sg[0, k] * v1b[row, mat, k/128]
//     out_raw[n] = sum_{k=0..2559} fp16(x_rot[k]) * W_pmcgx_e4m3[k, n]    (16x16 tiles, 160 k-steps x 8 n-tiles)
//     y_m = fp16(H128(fp16(out_raw)) * svh_m)   (svh_gate = v2[row, 0], svh_up = v2[row, 1])
//   Mid:
//     act = fp16(silu(y_gate) * y_up)
//     x2_rot = fp16(H128(act * suh_down))       (suh_down = v2[row, 2])
//   FC2 (down):
//     out_raw[n] = sum_{k=0..127} fp16(x2_rot[k]) * W_down_pmcgx_e4m3[k, n]   (8 k-steps x 160 n-tiles)
//     y_b[n] = H128(fp16(out_raw))[n] * svh_down[n],  svh_down[n] = sg[1, n] * v3b[row, n/128]
//     routed_acc[m, n] += route_w * y_b[n]
//
// Notice: reads every piece through tr3_fc1_unit(t_fc1, t_ptab, t_kmix, ...) and
// tr3_fc2_block(t_fc2, t_ptab, ...) — so GR8_PTAB=0, 1, 2 are 100% natively supported!
// =================================================================================================

// Decode one 16x16 PMCGX (E4M3-rounded) tile at width `BITS` using Tr3Lane<BITS, true, TR3_PMCGX>
// and accumulate `C[4] += A(tile) @ B(x_frag)` or direct dot with 16-element x vector.
// Note on tile layout (tr3.cuh):
//   In a 16x16 tile, lane l (0..31) decodes 8 weights:
//     f0[0] = half2(w[0], w[1]), f0[1] = half2(w[2], w[3]) -> cols 0..7?
//   EXACT mma16816 A-fragment layout (tr3.cuh line 4 & common.cuh line 63):
//     row0 = lane / 4  (0..7),  row1 = lane / 4 + 8  (8..15),  c = lane % 4  (0..3)
//     f0[0] = (row0, k=2c..2c+1)   [a0]
//     f0[1] = (row1, k=2c..2c+1)   [a1]
//     f1[0] = (row0, k=2c+8..2c+9) [a2]
//     f1[1] = (row1, k=2c+8..2c+9) [a3]
//   WAIT: in tr3.cuh line 4: "rows = output cols l/4 and l/4+8" of the tile!
//   That means for tile (ks, nj):
//     A-operand row r (0..15) is OUTPUT column `nj * 16 + r`, and A-operand k (0..15) is INPUT index `ks * 16 + k`!
//   Let's double check! For a single vector x[16] in shared memory:
//     b0 = half2(x[2c], x[2c+1]), b1 = half2(x[2c+8], x[2c+9])!
//     Then sum over c=0..3 (4 lanes with same lane/4) of dot(f0[0], b0) + dot(f1[0], b1) gives output row0 = lane/4,
//     and dot(f0[1], b0) + dot(f1[1], b1) gives output row1 = lane/4 + 8!
//   OR even better: use mma16816 directly when batching up to 8 tokens, AND for 1 token use warp dot or mma16816!
//   Using mma16816 directly (with B fragment = up to 8 tokens) guarantees BIT-IDENTICAL FP16->FP32 MMA math
//   with gr8's tr3.cuh AND runs at Tensor Core speed!!

template <int BITS>
__device__ __forceinline__ void pmcgx_tile_mma(const uint8_t* tile, int lane, const Tr3Lane<BITS, true, TR3_PMCGX>& tl,
                                               u32 b0, u32 b1, float (&c)[4]) {
    u32 raw = tl.loadw(tile, lane);
    u32 f0[2], f1[2];
    tl.decodew(raw, f0, f1);
    mma16816(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
}

// FC1 + SiLU*Up + H128(act * suh_down) for one (route = m*10 + k, block b).
// Grid: (M * TOPK * eb), Block: 256 threads (8 warps: warp j handles output n-tile j in 0..7 -> 128 output cols!).
// Each warp loops over ks = 0..159 (160 k16 steps) using mma16816 on token m (in B col 0).
__global__ void k_qw38_moe_fc1(const __nv_bfloat16* __restrict__ x,
                               const int* __restrict__ topk_idx,
                               const long long* __restrict__ t_tab,
                               const long long* __restrict__ t_kmix,
                               const uint8_t* __restrict__ t_fc1,
                               const uint8_t* const* __restrict__ t_ptab,
                               const __half* __restrict__ t_sg,
                               const __half* __restrict__ t_v1b,
                               const __half* __restrict__ t_v2,
                               __half* __restrict__ fc1_act,
                               int M, int eb) {
    const int idx = blockIdx.x;
    const int b = idx % eb;
    const int route = idx / eb;
    const int m = route / geom::TOPK;
    if (m >= M) return;
    const int e = topk_idx[route];
    const long long* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;   // 0..7 -> n-tile j (16 output cols each = 128 cols total!)
    const int lane = tid & 31;
    const int row0 = lane >> 2;  // 0..7
    const int col = lane & 3;    // 0..3

    const __nv_bfloat16* xm = x + static_cast<size_t>(m) * geom::HID;
    __shared__ __half x_rot[geom::HID]; // 5120 bytes: H128(x * suh_m) in fp16
    __shared__ float y_gate[128];
    __shared__ float y_up[128];

    Tr3Lane<2, true, TR3_PMCGX> tl2; tl2.init(lane);
    Tr3Lane<3, true, TR3_PMCGX> tl3; tl3.init(lane);
    Tr3Lane<4, true, TR3_PMCGX> tl4; tl4.init(lane);

    // Run mat=0 (gate) then mat=1 (up)
    for (int mat = 0; mat < 2; ++mat) {
        // 1. Compute x_rot = fp16(H128(x * suh_mat)) across HB=20 blocks of 128 (8 warps loop over 20 blocks)
        for (int hb = warp; hb < geom::HB; hb += 8) {
            const float bscale = __half2float(t_v1b[(brow * 2 + mat) * geom::HB + hb]);
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k = hb * 128 + lane * 4 + i;
                const float sg = __half2float(t_sg[k]); // sg[0][k] = input sign
                v[i] = __bfloat162float(xm[k]) * (sg * bscale);
            }
            had128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                x_rot[hb * 128 + lane * 4 + i] = __float2half_rn(v[i]);
            }
        }
        __syncthreads();

        // 2. Matvec against PMCGX tiles for block b, matrix mat (reading through t_ptab when GR8_PTAB=1|2!)
        const uint8_t* mw = nullptr;
        const int bits = tr3_fc1_unit(t_fc1, t_ptab, t_kmix, e, tr, 2 * b + mat, mw);
        const size_t tb = static_cast<size_t>(32 * bits);
        float c[4] = {0.f, 0.f, 0.f, 0.f};

        for (int ks = 0; ks < geom::KSH; ++ks) {
            const __half* xk = x_rot + ks * 16;
            const u32 b0 = *reinterpret_cast<const u32*>(xk + 2 * col);
            const u32 b1 = *reinterpret_cast<const u32*>(xk + 2 * col + 8);
            const uint8_t* tile = mw + (static_cast<size_t>(ks) * 8 + warp) * tb;
            if (bits == 3) pmcgx_tile_mma<3>(tile, lane, tl3, b0, b1, c);
            else if (bits == 4) pmcgx_tile_mma<4>(tile, lane, tl4, b0, b1, c);
            else pmcgx_tile_mma<2>(tile, lane, tl2, b0, b1, c);
        }
        // In mma16816, when B has token m in col 0 (lane/4 == 0 => only lanes 0..3 hold col 0 in B?
        // WAIT! Let's check common.cuh line 64:
        //   B regs: b0 (k 2c..2c+1, col l/4), b1 (k+8, col l/4)
        //   C regs: c0,c1 (row l/4, col 2c, 2c+1), c2,c3 (row l/4+8, col 2c, 2c+1)
        // Since we passed the SAME xk for ALL l/4 in 0..7, ALL 8 columns of B equal xk!
        // Therefore c[0] = (row0, col 2c) and c[2] = (row0+8, col 2c) hold the EXACT dot product for col=0 (when c==0)!
        float* ydst = (mat == 0) ? y_gate : y_up;
        if (col == 0) {
            ydst[warp * 16 + row0] = __half2float(__float2half_rn(c[0]));
            ydst[warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[2]));
        }
        __syncthreads();
    }

    // 3. Apply H128 * svh on y_gate & y_up, compute silu(g)*u, multiply by suh_down, and H128 into fc1_act!
    if (warp == 0) {
        float vg[4], vu[4];
        const __half* svh_g = t_v2 + (brow * 3 + 0) * 128;
        const __half* svh_u = t_v2 + (brow * 3 + 1) * 128;
        const __half* suh_d = t_v2 + (brow * 3 + 2) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = y_gate[lane * 4 + i];
            vu[i] = y_up[lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        float va[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float g = __half2float(__float2half_rn(vg[i] * __half2float(svh_g[lane * 4 + i])));
            const float u = __half2float(__float2half_rn(vu[i] * __half2float(svh_u[lane * 4 + i])));
            const float act = __half2float(__float2half_rn(silu_f(g) * u));
            va[i] = act * __half2float(suh_d[lane * 4 + i]);
        }
        had128(va, lane);
        __half* dst = fc1_act + (static_cast<size_t>(route) * eb + b) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(va[i]);
    }
}

// FC2 (down projection) for one (route = m*10 + k, block b, hidden_block hb in 0..19).
// Grid: (M * TOPK * eb * HB), Block: 256 threads (8 warps: warp j handles output n-tile j of this 128-col hidden block!).
// Accumulates `route_w * (H128(out_raw) * svh_down)` atomically into routed_acc[m, hb*128 .. hb*128+127].
__global__ void k_qw38_moe_fc2(const __half* __restrict__ fc1_act,
                               const int* __restrict__ topk_idx,
                               const float* __restrict__ topk_w,
                               const long long* __restrict__ t_tab,
                               const uint8_t* __restrict__ t_fc2,
                               const uint8_t* const* __restrict__ t_ptab,
                               const __half* __restrict__ t_sg,
                               const __half* __restrict__ t_v3b,
                               float* __restrict__ routed_acc,
                               int M, int eb) {
    const int idx = blockIdx.x;
    const int hb = idx % geom::HB;
    const int rem = idx / geom::HB;
    const int b = rem % eb;
    const int route = rem / eb;
    const int m = route / geom::TOPK;
    if (m >= M) return;
    const int e = topk_idx[route];
    const long long* tr = t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;

    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int bits = static_cast<int>(tr[4]);
    const uint8_t* mw = tr3_fc2_block(t_fc2, t_ptab, e, tr, b, bits);
    const size_t tb = static_cast<size_t>(32 * bits);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;   // 0..7 -> n-tile within this 128-wide hidden block hb
    const int lane = tid & 31;
    const int row0 = lane >> 2;
    const int col = lane & 3;

    const __half* x2 = fc1_act + (static_cast<size_t>(route) * eb + b) * 128;
    Tr3Lane<2, true, TR3_PMCGX> tl2; tl2.init(lane);
    Tr3Lane<3, true, TR3_PMCGX> tl3; tl3.init(lane);
    Tr3Lane<4, true, TR3_PMCGX> tl4; tl4.init(lane);

    // Note on tr3.fc2 tile indexing (tr3.cuh):
    //   fc2 has K = 128 (8 k16 steps), N = HID (KSH = 160 n16 tiles).
    //   Tile layout in tr3.fc2: [n_tile (0..159)][k_step (0..7)][TB]!!
    //   Let's verify: in tr3.cuh line 607: tile = base + (n_tile * 8 + ks) * TB.
    const int n_tile = hb * 8 + warp;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ks = 0; ks < 8; ++ks) {
        const u32 b0 = *reinterpret_cast<const u32*>(x2 + ks * 16 + 2 * col);
        const u32 b1 = *reinterpret_cast<const u32*>(x2 + ks * 16 + 2 * col + 8);
        const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
        if (bits == 3) pmcgx_tile_mma<3>(tile, lane, tl3, b0, b1, c);
        else if (bits == 4) pmcgx_tile_mma<4>(tile, lane, tl4, b0, b1, c);
        else pmcgx_tile_mma<2>(tile, lane, tl2, b0, b1, c);
    }

    __shared__ float y_raw[128];
    if (col == 0) {
        y_raw[warp * 16 + row0] = __half2float(__float2half_rn(c[0]));
        y_raw[warp * 16 + row0 + 8] = __half2float(__float2half_rn(c[2]));
    }
    __syncthreads();

    if (warp == 0) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = y_raw[lane * 4 + i];
        had128(v, lane);
        const float bscale = __half2float(t_v3b[brow * geom::HB + hb]);
        const float rw = topk_w[route];
        float* part_dst = routed_acc + (static_cast<size_t>(route) * eb + b) * geom::HID + hb * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = hb * 128 + lane * 4 + i;
            const float sg = __half2float(t_sg[geom::HID + d]); // sg[1][d] = output sign
            const float y = v[i] * (sg * bscale);
            part_dst[lane * 4 + i] = rw * y;
        }
    }
}

// Deterministic reduction of fc2_part[m, 0..TOPK-1, 0..nb-1, d] + sigmoid(sh_scalar_gate) * sh_out -> out_part [M, 2560] BF16
__global__ void k_qw38_moe_combine_det(const float* __restrict__ fc2_part,
                                       const int* __restrict__ topk_idx,
                                       const long long* __restrict__ t_tab,
                                       const __nv_bfloat16* __restrict__ sh_out,
                                       const float* __restrict__ sh_scalar_gate,
                                       __nv_bfloat16* __restrict__ out_part,
                                       int M, int eb) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float sg = sigmoid_f(sh_scalar_gate[m]);
    const __nv_bfloat16* s = sh_out + static_cast<size_t>(m) * geom::HID;
    __nv_bfloat16* o = out_part + static_cast<size_t>(m) * geom::HID;
    __shared__ int s_nb[geom::TOPK];
    if (threadIdx.x < geom::TOPK) {
        const int e = topk_idx[m * geom::TOPK + threadIdx.x];
        s_nb[threadIdx.x] = static_cast<int>(t_tab[6 * e + 3]);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < geom::HID; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < geom::TOPK; ++k) {
            const int nb = s_nb[k];
            const size_t rbase = static_cast<size_t>(m * geom::TOPK + k) * eb;
            for (int b = 0; b < nb; ++b) {
                acc += fc2_part[(rbase + b) * geom::HID + d];
            }
        }
        const float sv = __bfloat162float(__float2bfloat16(sg * __bfloat162float(s[d])));
        o[d] = __float2bfloat16(acc + sv);
    }
}

// Shared expert activation: sh_gate = bf16(silu(sh_gate) * sh_up)
__global__ void k_silu_mul_bf16(__nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        gate[i] = __float2bfloat16(silu_f(g) * u);
    }
}

// Combine routed_acc (FP32) + sigmoid(sh_scalar_gate) * sh_out (BF16) -> out_part [M, 2560] BF16
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
    cublasSetStream(blas, st);

    // 1. Router logits -> top-10 normalized weights
    gemm_bf16_nt_f32(blas, M, geom::NEXP, geom::HID, x, w.router, sc.router_logits);
    k_qw38_router_topk10<<<M, 256, 0, st>>>(sc.router_logits, sc.topk_idx, sc.topk_w, M);

    // 2. Shared expert (BF16, sh_i channels) + scalar sigmoid gate
    gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_gate, sc.sh_gate_buf);
    gemm_bf16_nt(blas, M, w.sh_i, geom::HID, x, w.sh_up, sc.sh_up_buf);
    {
        const int n = M * w.sh_i;
        k_silu_mul_bf16<<<(n + 255) / 256, 256, 0, st>>>(sc.sh_gate_buf, sc.sh_up_buf, n);
    }
    gemm_bf16_nt(blas, M, geom::HID, w.sh_i, sc.sh_gate_buf, w.sh_down, sc.sh_out);
    gemm_bf16_nt_f32(blas, M, 1, geom::HID, x, w.sh_exp_gate, sc.sh_scalar_gate);

    // 3. Routed experts + 4. Combine routed + gated shared -> out_part [M, 2560] BF16
    if (w.is_tr3) {
        const int n_routes = M * geom::TOPK;
        k_qw38_moe_fc1<<<n_routes * w.t_eb, 256, 0, st>>>(
            x, sc.topk_idx, w.t_tab, w.t_kmix, w.t_fc1, w.t_ptab, w.t_sg, w.t_v1b, w.t_v2, sc.fc1_act, M, w.t_eb);
        k_qw38_moe_fc2<<<n_routes * w.t_eb * geom::HB, 256, 0, st>>>(
            sc.fc1_act, sc.topk_idx, sc.topk_w, w.t_tab, w.t_fc2, w.t_ptab, w.t_sg, w.t_v3b, sc.fc2_part, M, w.t_eb);
        k_qw38_moe_combine_det<<<M, 256, 0, st>>>(
            sc.fc2_part, sc.topk_idx, w.t_tab, sc.sh_out, sc.sh_scalar_gate, out_part, M, w.t_eb);
    } else if (w.mtp_gate_up && w.mtp_down) {
        cudaMemsetAsync(sc.routed_acc, 0, static_cast<size_t>(M) * geom::HID * sizeof(float), st);
        // BF16 routed experts for MTP layer L048 (small M, typically 1..8 during draft)
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
