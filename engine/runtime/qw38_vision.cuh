// Qwen3.8-Flash-Next vision tower (Qwen4ExpVisionModel, dense/vision-bf16.safetensors) and S2 prefill staging.
// Every TP rank holds the whole 27-layer BF16 ViT (~898 MB) and encodes images locally (no cross-rank communication).
//
// In: smart-resized RGB uint8 [vit_h * 16][vit_w * 16][3] (qw38_vision_plan.h).
// Out: BF16 [vit_h * vit_w / 4][2560], one row per <|image_pad|> token in row-major merged-token order.
//
// Pipeline (verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py):
//   patchify  patch n in 2x2 spatial-merge-block order -> [3, 2, 16, 16] BF16 normalized (u / 255 - 0.5) / 0.5
//   embed     x = bf16(patch_embed.proj(patches) + bias) + bf16(bilinear_align_corners(pos_embed[48x48]))
//   27 layers h = x;
//             qkv = linear(LayerNorm(h, 1e-6), qkv_w, qkv_b)          [N, 3, 16, 72]
//             axial 2D rotate_half RoPE on Q and K (theta=10000, 18 h-freqs + 18 w-freqs)
//             bidirectional SDPA (16 heads x 72 dim, scale = 1/sqrt(72)) -> proj -> h += out
//             h += fc2(gelu_pytorch_tanh(fc1(LayerNorm(h, 1e-6))))    (1152 -> 4304 -> 1152)
//   merger    y = LayerNorm(h, 1e-6)                                  [N, 1152] (use_postshuffle_norm=False)
//             reshape [N/4, 4608] -> fc1(4608->4608) -> exact GELU -> fc2(4608->2560)
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_input.h"
#include "qw38_vision_plan.h"
#include "vision_id.h"
#include "vision_weights.h"

namespace qw38rt {

#define QW38V_CUDA(x) do { cudaError_t ce_ = (x); if (ce_ != cudaSuccess) throw std::runtime_error(std::string("qw38 vision: ") + #x + ": " + cudaGetErrorString(ce_)); } while (0)
#define QW38V_BLAS(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("qw38 vision: ") + #x + " failed: " + std::to_string((int)st_)); } while (0)

namespace vis {
using bf16 = __nv_bfloat16;

constexpr int VD = 1152;                                      // vision_config.hidden_size
constexpr int VHEADS = 16;                                    // vision_config.num_heads
constexpr int VHD = 72;                                       // VD / VHEADS
constexpr int VRF = VHD / 4;                                  // 18 axial frequencies per axis (h, w)
constexpr int VL = 27;                                        // vision_config.depth
constexpr int VMLP = 4304;                                    // vision_config.intermediate_size
constexpr int VGRID = 48;                                     // sqrt(num_position_embeddings = 2304)
constexpr int VOUT = 2560;                                    // vision_config.out_hidden_size
constexpr int VMRG = VD * qw38v::MERGE * qw38v::MERGE;        // 4608
constexpr int VPK = 3 * qw38v::TEMPORAL * qw38v::PATCH * qw38v::PATCH; // 1536
constexpr float VEPS = 1e-6f;
constexpr float VTHETA = 10000.0f;
constexpr size_t ACC_FLOATS = size_t(8) << 20;                // 32 MiB FP32 GEMM accumulator

// Spatial-merge-block patch order: 4 patches of each 2x2 merged token are consecutive.
__device__ __forceinline__ void patch_rc(int n, int vit_w, int& row, int& col) {
    const int blk = n >> 2;
    const int in = n & 3;
    const int cols = vit_w / qw38v::MERGE;
    row = (blk / cols) * qw38v::MERGE + (in >> 1);
    col = (blk % cols) * qw38v::MERGE + (in & 1);
}

// pixels [vit_h*16, vit_w*16, 3] uint8 -> patches [N, 3, 2, 16, 16] BF16 in merge-block order.
// Grid: (N), Block: 256.
__global__ void k_qw38v_patchify(const uint8_t* __restrict__ px, int vit_w, bf16* __restrict__ out) {
    const int n = blockIdx.x;
    int row, col;
    patch_rc(n, vit_w, row, col);
    const size_t W = size_t(vit_w) * qw38v::PATCH;
    bf16* dst = out + size_t(n) * VPK;
    for (int i = threadIdx.x; i < VPK; i += blockDim.x) {
        const int ch = i / 512;
        const int rem = i & 511;
        const int py = (rem & 255) >> 4;
        const int pxx = rem & 15;
        const int gy = row * qw38v::PATCH + py;
        const int gx = col * qw38v::PATCH + pxx;
        const float u = static_cast<float>(px[(size_t(gy) * W + gx) * 3 + ch]);
        // (u / 255 - 0.5) / 0.5 = u * (2 / 255) - 1
        dst[i] = __float2bfloat16_rn(__fdiv_rn(__fsub_rn(__fmul_rn(u, 1.0f / 255.0f), 0.5f), 0.5f));
    }
}

// Bilinear interpolation with align_corners=True (matches Qwen4ExpVisionModel pos_embed resampling).
__device__ __forceinline__ void pos_lerp_ac(int i, int n, int g, int& i0, int& i1, float& w1) {
    if (n <= 1) {
        i0 = i1 = 0;
        w1 = 0.0f;
        return;
    }
    const float s = static_cast<float>(i) * static_cast<float>(g - 1) / static_cast<float>(n - 1);
    const float f = floorf(s);
    i0 = min(max(static_cast<int>(f), 0), g - 1);
    i1 = min(i0 + 1, g - 1);
    w1 = s - f;
}

// x[N, 1152] += bf16(bilinear_align_corners(pos_table[48*48, 1152])). Grid: (N), Block: 256.
__global__ void k_qw38v_pos_add(bf16* __restrict__ x, const bf16* __restrict__ table, int vit_h, int vit_w) {
    const int n = blockIdx.x;
    int row, col, y0, y1, x0, x1;
    float wy1, wx1;
    patch_rc(n, vit_w, row, col);
    pos_lerp_ac(row, vit_h, VGRID, y0, y1, wy1);
    pos_lerp_ac(col, vit_w, VGRID, x0, x1, wx1);
    const float wy0 = 1.0f - wy1, wx0 = 1.0f - wx1;
    const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
    const bf16* a = table + size_t(y0 * VGRID + x0) * VD;
    const bf16* b = table + size_t(y0 * VGRID + x1) * VD;
    const bf16* c = table + size_t(y1 * VGRID + x0) * VD;
    const bf16* e = table + size_t(y1 * VGRID + x1) * VD;
    bf16* xr = x + size_t(n) * VD;
    for (int d = threadIdx.x; d < VD; d += blockDim.x) {
        float p = w00 * __bfloat162float(a[d]) + w01 * __bfloat162float(b[d]) +
                  w10 * __bfloat162float(c[d]) + w11 * __bfloat162float(e[d]);
        p = __bfloat162float(__float2bfloat16_rn(p));
        xr[d] = __float2bfloat16_rn(__bfloat162float(xr[d]) + p);
    }
}

__device__ __forceinline__ float sum128(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float t = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return t;
}

// LayerNorm over VD=1152 (9 elements per thread across 128 threads, eps=1e-6). Grid: (N), Block: 128.
__global__ void __launch_bounds__(128) k_qw38v_layernorm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                         const bf16* __restrict__ b, bf16* __restrict__ y) {
    __shared__ float red[4];
    const int r = blockIdx.x;
    const bf16* xr = x + size_t(r) * VD;
    bf16* yr = y + size_t(r) * VD;
    float v[9], s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        v[j] = __bfloat162float(xr[threadIdx.x + 128 * j]);
        s += v[j];
    }
    const float mean = sum128(s, red) * (1.0f / float(VD));
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        const float d = v[j] - mean;
        ss += d * d;
    }
    const float rs = rsqrtf(sum128(ss, red) * (1.0f / float(VD)) + VEPS);
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        const int k = threadIdx.x + 128 * j;
        const float t = (v[j] - mean) * rs * __bfloat162float(w[k]) + __bfloat162float(b[k]);
        yr[k] = __float2bfloat16_rn(t);
    }
}

// Axial 2D rotate_half RoPE on Q and K inside qkv [N, 3, 16, 72].
// Grid: (N), Block: dim3(36, 16) (576 threads: x = pair 0..35, y = head 0..15).
__global__ void k_qw38v_rope(bf16* __restrict__ qkv, int vit_w) {
    const int s = blockIdx.x;
    const int j = threadIdx.x; // 0..35
    const int h = threadIdx.y; // 0..15
    int row, col;
    patch_rc(s, vit_w, row, col);
    const int m = (j < VRF) ? j : (j - VRF);
    const float inv = 1.0f / powf(VTHETA, static_cast<float>(2 * m) / static_cast<float>(VHD / 2));
    const float ang = static_cast<float>(j < VRF ? row : col) * inv;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    bf16* base = qkv + size_t(s) * (3 * VD) + h * VHD;
    #pragma unroll
    for (int qk = 0; qk < 2; ++qk) {
        bf16* p = base + qk * VD;
        const float a = __bfloat162float(p[j]);
        const float b = __bfloat162float(p[j + 36]);
        p[j] = __float2bfloat16_rn(a * cs - b * sn);
        p[j + 36] = __float2bfloat16_rn(b * cs + a * sn);
    }
}

// Bidirectional tiled FlashAttention for HD=72, 16 heads.
// Grid: (ceil(N / 4), 16), Block: 128 (4 warps; warp w handles query row q = blockIdx.x * 4 + w).
// Each thread in a warp holds 3 dims: lane, lane+32, lane+64 (only lanes 0..7 active for +64).
constexpr int ATTN_QW = 4;
constexpr int ATTN_KT = 32;

__global__ void __launch_bounds__(128) k_qw38v_attn(const bf16* __restrict__ qkv, bf16* __restrict__ out, int N) {
    __shared__ bf16 sK[ATTN_KT][VHD];
    __shared__ bf16 sV[ATTN_KT][VHD];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.y;
    const int q = blockIdx.x * ATTN_QW + warp;
    const size_t ld = size_t(3) * VD;
    const float scale = 0.11785113019775793f; // 1 / sqrt(72)

    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f;
    if (q < N) {
        const bf16* qp = qkv + size_t(q) * ld + h * VHD;
        q0 = __bfloat162float(qp[lane]);
        q1 = __bfloat162float(qp[lane + 32]);
        if (lane < 8) q2 = __bfloat162float(qp[lane + 64]);
    }

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f;
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k0 = 0; k0 < N; k0 += ATTN_KT) {
        const int kt = min(ATTN_KT, N - k0);
        // Load kt x 72 K and V tiles into shared memory
        for (int idx = tid; idx < kt * VHD; idx += 128) {
            const int r = idx / VHD;
            const int c = idx % VHD;
            const bf16* src = qkv + size_t(k0 + r) * ld + h * VHD + c;
            sK[r][c] = src[VD];
            sV[r][c] = src[2 * VD];
        }
        __syncthreads();

        if (q < N) {
            for (int r = 0; r < kt; ++r) {
                float dot = q0 * __bfloat162float(sK[r][lane]) + q1 * __bfloat162float(sK[r][lane + 32]);
                if (lane < 8) dot += q2 * __bfloat162float(sK[r][lane + 64]);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
                const float sc = dot * scale;
                const float nmax = fmaxf(m_max, sc);
                const float alpha = __expf(m_max - nmax);
                const float p = __expf(sc - nmax);
                m_max = nmax;
                l_sum = l_sum * alpha + p;
                o0 = o0 * alpha + p * __bfloat162float(sV[r][lane]);
                o1 = o1 * alpha + p * __bfloat162float(sV[r][lane + 32]);
                if (lane < 8) o2 = o2 * alpha + p * __bfloat162float(sV[r][lane + 64]);
            }
        }
        __syncthreads();
    }

    if (q < N) {
        const float inv = 1.0f / l_sum;
        bf16* dst = out + size_t(q) * VD + h * VHD;
        dst[lane] = __float2bfloat16_rn(o0 * inv);
        dst[lane + 32] = __float2bfloat16_rn(o1 * inv);
        if (lane < 8) dst[lane + 64] = __float2bfloat16_rn(o2 * inv);
    }
}

enum Epi { E_BIAS = 0, E_RES = 1, E_GELU_TANH = 2, E_GELU_EXACT = 3 };

__global__ void k_qw38v_epi(const float* __restrict__ acc, const bf16* __restrict__ bias,
                            bf16* __restrict__ out, int rows, int N, int mode) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * N) return;
    const int c = static_cast<int>(i % N);
    float v = acc[i];
    if (bias) v += __bfloat162float(bias[c]);
    if (mode == E_BIAS) {
        out[i] = __float2bfloat16_rn(v);
    } else if (mode == E_RES) {
        out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) + __bfloat162float(__float2bfloat16_rn(v)));
    } else if (mode == E_GELU_TANH) {
        v = __bfloat162float(__float2bfloat16_rn(v));
        const float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        out[i] = __float2bfloat16_rn(0.5f * v * (1.0f + t));
    } else {
        v = __bfloat162float(__float2bfloat16_rn(v));
        out[i] = __float2bfloat16_rn(0.5f * v * (1.0f + erff(v * 0.7071067811865475244f)));
    }
}

} // namespace vis

class Qw38Vision {
public:
    using bf16 = vis::bf16;

    long encode_calls = 0;
    bool log = false;

    bool ready() const { return blob_.dev != nullptr; }
    const std::string& id() const { return id_; }
    size_t device_bytes() const {
        return blob_.bytes + ws_bytes_ + size_t(pool_rows_) * vis::VOUT * sizeof(bf16) +
               size_t(stage_cap_) * vis::VOUT * sizeof(bf16);
    }

    // Load dense/vision-bf16.safetensors and allocate the prefill launch pool (pool_rows rows).
    // Workspace and stage buffer grow on demand up to qw38v::MAX_PATCHES so small images use little memory.
    void load(const std::string& path, int pool_rows) {
        if (pool_rows < 1) throw std::runtime_error("qw38 vision: pool_rows < 1");
        id_ = g6vis::file_id(path);
        if (id_.empty()) throw std::runtime_error("qw38 vision: cannot read header of " + path);
        blob_ = g6vis::load_blob(path, "qw38 vision");
        auto g = [&](const std::string& name, const std::vector<int64_t>& shape) -> const bf16* {
            return reinterpret_cast<const bf16*>(blob_.get(name, "BF16", shape, path, "qw38 vision"));
        };
        w_.patch_w = g("model.visual.patch_embed.proj.weight", {vis::VD, 3, qw38v::TEMPORAL, qw38v::PATCH, qw38v::PATCH});
        w_.patch_b = g("model.visual.patch_embed.proj.bias", {vis::VD});
        w_.pos = g("model.visual.pos_embed.weight", {vis::VGRID * vis::VGRID, vis::VD});
        w_.layers.resize(vis::VL);
        for (int l = 0; l < vis::VL; ++l) {
            const std::string p = "model.visual.blocks." + std::to_string(l) + ".";
            LayerW& y = w_.layers[size_t(l)];
            y.ln1_w = g(p + "norm1.weight", {vis::VD});
            y.ln1_b = g(p + "norm1.bias", {vis::VD});
            y.qkv_w = g(p + "attn.qkv.weight", {3 * vis::VD, vis::VD});
            y.qkv_b = g(p + "attn.qkv.bias", {3 * vis::VD});
            y.proj_w = g(p + "attn.proj.weight", {vis::VD, vis::VD});
            y.proj_b = g(p + "attn.proj.bias", {vis::VD});
            y.ln2_w = g(p + "norm2.weight", {vis::VD});
            y.ln2_b = g(p + "norm2.bias", {vis::VD});
            y.fc1_w = g(p + "mlp.linear_fc1.weight", {vis::VMLP, vis::VD});
            y.fc1_b = g(p + "mlp.linear_fc1.bias", {vis::VMLP});
            y.fc2_w = g(p + "mlp.linear_fc2.weight", {vis::VD, vis::VMLP});
            y.fc2_b = g(p + "mlp.linear_fc2.bias", {vis::VD});
        }
        w_.mrg_ln_w = g("model.visual.merger.norm.weight", {vis::VD});
        w_.mrg_ln_b = g("model.visual.merger.norm.bias", {vis::VD});
        w_.mrg_fc1_w = g("model.visual.merger.linear_fc1.weight", {vis::VMRG, vis::VMRG});
        w_.mrg_fc1_b = g("model.visual.merger.linear_fc1.bias", {vis::VMRG});
        w_.mrg_fc2_w = g("model.visual.merger.linear_fc2.weight", {vis::VOUT, vis::VMRG});
        w_.mrg_fc2_b = g("model.visual.merger.linear_fc2.bias", {vis::VOUT});

        QW38V_BLAS(cublasCreate(&blas_));
        pool_rows_ = pool_rows;
        QW38V_CUDA(cudaMalloc(&pool_, size_t(pool_rows_) * vis::VOUT * sizeof(bf16)));
        QW38V_CUDA(cudaMalloc(&d_row_map_, size_t(pool_rows_) * sizeof(int)));
        ensure_capacity(4096); // default 1024 tokens (~55 MB workspace), grows on demand up to 16384 tokens
    }

    // Encode one RGB image [vit_h * 16][vit_w * 16][3] -> BF16 [tokens][2560] on `out` (device pointer).
    void encode(const uint8_t* rgb, int vit_h, int vit_w, bf16* out, cudaStream_t st = nullptr) {
        if (!ready()) throw std::runtime_error("qw38 vision: encode before load");
        const int tokens = qw38v::span_of(vit_h, vit_w);
        if (!rgb || !tokens) throw std::runtime_error("qw38 vision: invalid grid " + std::to_string(vit_h) + "x" + std::to_string(vit_w));
        const int N = vit_h * vit_w;
        ensure_capacity(N);
        ++encode_calls;

        QW38V_BLAS(cublasSetStream(blas_, st));
        const size_t rgb_bytes = size_t(N) * qw38v::PATCH * qw38v::PATCH * 3;
        QW38V_CUDA(cudaMemcpyAsync(pix_, rgb, rgb_bytes, cudaMemcpyHostToDevice, st));

        // 1. Patchify into qkv_ [N, 1536] -> Conv3d GEMM -> x_ [N, 1152] + bilinear 48x48 pos_embed
        vis::k_qw38v_patchify<<<N, 256, 0, st>>>(pix_, vit_w, qkv_);
        linear(qkv_, w_.patch_w, w_.patch_b, x_, N, vis::VD, vis::VPK, vis::E_BIAS, st);
        vis::k_qw38v_pos_add<<<N, 256, 0, st>>>(x_, w_.pos, vit_h, vit_w);

        // 2. 27 ViT layers
        const int mlp_rows = std::max(1, static_cast<int>(vis::ACC_FLOATS / vis::VMLP));
        for (int l = 0; l < vis::VL; ++l) {
            const LayerW& y = w_.layers[size_t(l)];
            vis::k_qw38v_layernorm<<<N, 128, 0, st>>>(x_, y.ln1_w, y.ln1_b, t_);
            linear(t_, y.qkv_w, y.qkv_b, qkv_, N, 3 * vis::VD, vis::VD, vis::E_BIAS, st);
            vis::k_qw38v_rope<<<N, dim3(36, vis::VHEADS), 0, st>>>(qkv_, vit_w);
            vis::k_qw38v_attn<<<dim3((N + vis::ATTN_QW - 1) / vis::ATTN_QW, vis::VHEADS), 128, 0, st>>>(qkv_, t_, N);
            linear(t_, y.proj_w, y.proj_b, x_, N, vis::VD, vis::VD, vis::E_RES, st);

            vis::k_qw38v_layernorm<<<N, 128, 0, st>>>(x_, y.ln2_w, y.ln2_b, t_);
            for (int r0 = 0; r0 < N; r0 += mlp_rows) {
                const int rows = std::min(mlp_rows, N - r0);
                linear(t_ + size_t(r0) * vis::VD, y.fc1_w, y.fc1_b, m_, rows, vis::VMLP, vis::VD, vis::E_GELU_TANH, st);
                linear(m_, y.fc2_w, y.fc2_b, x_ + size_t(r0) * vis::VD, rows, vis::VD, vis::VMLP, vis::E_RES, st);
            }
        }

        // 3. Patch merger: LayerNorm(1152) per patch BEFORE 2x2 concat (use_postshuffle_norm=False)
        vis::k_qw38v_layernorm<<<N, 128, 0, st>>>(x_, w_.mrg_ln_w, w_.mrg_ln_b, t_);
        const int mrg_rows = std::max(1, static_cast<int>(vis::ACC_FLOATS / vis::VMRG));
        for (int r0 = 0; r0 < tokens; r0 += mrg_rows) {
            const int rows = std::min(mrg_rows, tokens - r0);
            linear(t_ + size_t(r0) * vis::VMRG, w_.mrg_fc1_w, w_.mrg_fc1_b, m_, rows, vis::VMRG, vis::VMRG, vis::E_GELU_EXACT, st);
            linear(m_, w_.mrg_fc2_w, w_.mrg_fc2_b, out + size_t(r0) * vis::VOUT, rows, vis::VOUT, vis::VMRG, vis::E_BIAS, st);
        }
        QW38V_CUDA(cudaGetLastError());
        QW38V_CUDA(cudaStreamSynchronize(st));
    }

    // Stage overlapping image embeddings for a prefill launch of `rows` tokens at prompt positions [first, first+rows).
    // Populates pool_ and d_row_map_ (-1 for text rows, 0..n-1 into pool_ for image rows). Returns count of image rows.
    int stage(const std::vector<g6rt::ImageIn>& images, int first, int rows, cudaStream_t st,
              int rank, unsigned long long request) {
        if (images.empty()) return 0;
        if (!ready()) throw std::runtime_error("qw38 vision: image request without vision tower");
        if (rows < 1 || rows > pool_rows_) throw std::runtime_error("qw38 vision: launch rows exceed pool");
        map_.assign(size_t(rows), -1);
        int pool_row = 0;
        for (size_t k = 0; k < images.size(); ++k) {
            const g6rt::ImageIn& im = images[k];
            const int a = std::max(im.pos, first);
            const int b = std::min(im.pos + im.span, first + rows);
            if (a >= b) continue;
            if (im.span != qw38v::span_of(im.vit_h, im.vit_w) || !im.rgb ||
                im.rgb->size() != size_t(im.vit_h) * im.vit_w * qw38v::PATCH * qw38v::PATCH * 3)
                throw std::runtime_error("qw38 vision: image span or RGB bytes mismatch grid");
            ensure_stage(im.span);
            if (!(last_ == im.rgb && last_h_ == im.vit_h && last_w_ == im.vit_w)) {
                const auto t0 = std::chrono::steady_clock::now();
                last_.reset();
                encode(im.rgb->data(), im.vit_h, im.vit_w, stage_, st);
                last_ = im.rgb;
                last_h_ = im.vit_h;
                last_w_ = im.vit_w;
                if (log && rank == 0) {
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    std::fprintf(stderr, "# qw38 vision: request %llu image %zu: %dx%d patches, %d tokens, tower %.1f ms\n",
                                 request, k, im.vit_h, im.vit_w, im.span, ms);
                }
            }
            QW38V_CUDA(cudaMemcpyAsync(pool_ + size_t(pool_row) * vis::VOUT,
                                       stage_ + size_t(a - im.pos) * vis::VOUT,
                                       size_t(b - a) * vis::VOUT * sizeof(bf16),
                                       cudaMemcpyDeviceToDevice, st));
            for (int p = a; p < b; ++p) map_[size_t(p - first)] = pool_row + (p - a);
            pool_row += (b - a);
        }
        if (pool_row > 0) {
            QW38V_CUDA(cudaMemcpyAsync(d_row_map_, map_.data(), size_t(rows) * sizeof(int), cudaMemcpyHostToDevice, st));
        }
        return pool_row;
    }

    const bf16* pool() const { return pool_; }
    const int* row_map() const { return d_row_map_; }

    Qw38Vision() = default;
    Qw38Vision(const Qw38Vision&) = delete;
    Qw38Vision& operator=(const Qw38Vision&) = delete;
    ~Qw38Vision() {
        if (blas_) cublasDestroy(blas_);
        if (ws_) cudaFree(ws_);
        if (pool_) cudaFree(pool_);
        if (d_row_map_) cudaFree(d_row_map_);
        if (stage_) cudaFree(stage_);
        if (blob_.dev) cudaFree(blob_.dev);
    }

private:
    struct LayerW {
        const bf16 *ln1_w, *ln1_b, *qkv_w, *qkv_b, *proj_w, *proj_b;
        const bf16 *ln2_w, *ln2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    };
    struct Weights {
        const bf16 *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr;
        const bf16 *mrg_ln_w = nullptr, *mrg_ln_b = nullptr;
        const bf16 *mrg_fc1_w = nullptr, *mrg_fc1_b = nullptr, *mrg_fc2_w = nullptr, *mrg_fc2_b = nullptr;
        std::vector<LayerW> layers;
    };

    g6vis::Blob blob_;
    std::string id_;
    Weights w_;
    cublasHandle_t blas_ = nullptr;

    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    int ws_patches_ = 0;
    uint8_t* pix_ = nullptr;
    bf16 *x_ = nullptr, *t_ = nullptr, *qkv_ = nullptr, *m_ = nullptr;
    float* acc_ = nullptr;

    bf16* pool_ = nullptr;
    int* d_row_map_ = nullptr;
    int pool_rows_ = 0;
    bf16* stage_ = nullptr;
    int stage_cap_ = 0;
    std::shared_ptr<const std::vector<uint8_t>> last_;
    int last_h_ = 0, last_w_ = 0;
    std::vector<int> map_;

    void ensure_capacity(int patches) {
        if (patches <= ws_patches_) return;
        if (ws_) { cudaFree(ws_); ws_ = nullptr; ws_bytes_ = 0; ws_patches_ = 0; }
        const size_t N = size_t(patches);
        const size_t m_cells = std::max(size_t(vis::ACC_FLOATS / vis::VMLP) * vis::VMLP,
                                        size_t(vis::ACC_FLOATS / vis::VMRG) * vis::VMRG);
        size_t off = 0;
        auto take = [&](size_t bytes) {
            const size_t o = off;
            off += (bytes + 255) & ~size_t(255);
            return o;
        };
        const size_t o_pix = take(N * qw38v::PATCH * qw38v::PATCH * 3);
        const size_t o_x = take(N * vis::VD * sizeof(bf16));
        const size_t o_t = take(N * vis::VD * sizeof(bf16));
        const size_t o_qkv = take(std::max(N * 3 * vis::VD, N * vis::VPK) * sizeof(bf16));
        const size_t o_m = take(m_cells * sizeof(bf16));
        const size_t o_acc = take(vis::ACC_FLOATS * sizeof(float));
        QW38V_CUDA(cudaMalloc(&ws_, off));
        ws_bytes_ = off;
        ws_patches_ = patches;
        char* b = static_cast<char*>(ws_);
        pix_ = reinterpret_cast<uint8_t*>(b + o_pix);
        x_ = reinterpret_cast<bf16*>(b + o_x);
        t_ = reinterpret_cast<bf16*>(b + o_t);
        qkv_ = reinterpret_cast<bf16*>(b + o_qkv);
        m_ = reinterpret_cast<bf16*>(b + o_m);
        acc_ = reinterpret_cast<float*>(b + o_acc);
    }

    void ensure_stage(int tokens) {
        if (tokens <= stage_cap_) return;
        if (stage_) { cudaFree(stage_); stage_ = nullptr; stage_cap_ = 0; }
        QW38V_CUDA(cudaMalloc(&stage_, size_t(tokens) * vis::VOUT * sizeof(bf16)));
        stage_cap_ = tokens;
    }

    void linear(const bf16* x, const bf16* W, const bf16* bias, bf16* y,
                int M, int N, int K, int mode, cudaStream_t st) {
        const float one = 1.0f, zero = 0.0f;
        const int pass = std::max(1, std::min(M, static_cast<int>(vis::ACC_FLOATS / size_t(N))));
        for (int r0 = 0; r0 < M; r0 += pass) {
            const int rows = std::min(pass, M - r0);
            QW38V_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K,
                                    &one, W, CUDA_R_16BF, K,
                                    x + size_t(r0) * K, CUDA_R_16BF, K,
                                    &zero, acc_, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            const size_t tot = size_t(rows) * N;
            vis::k_qw38v_epi<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(
                acc_, bias, y + size_t(r0) * N, rows, N, mode);
        }
    }
};

} // namespace qw38rt
