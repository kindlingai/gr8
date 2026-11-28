// GLM-5.3 vision tower (Glm5Next ViT + 6144-dim merger + learned row_break_embed, trained by exlr8-vproj), for image
// input on GLM-5.3 (GR8_GLM53_VISION=1 / GR8_GLM53_VISION_PATH). Every TP rank runs it on the same pixels (the token
// embedding table is whole on every rank), so the ranks exchange nothing for its rows.
//
// In: the padded RGB the front end sends (uint8 [vit_h * 14][vit_w * 14][3]: Glm5NextImageProcessor content resize and
// zero pad). Out: bf16 [(vit_h / 2) * (vit_w / 2 + 1)][6144], one row per spliced <|image|> token in row-major order
// with `model.visual.row_break_embed` inserted at column W_p of every merged row (exact parity with exlr8-vproj's
// `k_row_break_splice_fwd`). The prefill embed (Glm53Vision, glm53_vision_stage.cuh) reads these rows into the FP32
// residual stream in place of the <|image|> embedding row.
//
// Weights: `vision-glm53-bf16.safetensors` (348 BF16 tensors, `model.visual.*`: the 346 Glm5Next ViT/merger tensors,
// `model.visual.merger.down_proj.weight` [6144, 10240], and `model.visual.row_break_embed` [1, 6144]), read whole into
// one device allocation at boot (~1.17 GB). Workspace stays near ~0.6 GB by running MLPs in row passes and aliasing the
// final [L][6144] merger output inside `qkv_` before expanding row breaks into `out`.
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

#include "glm53_vision_plan.h"
#include "vision_weights.h"

namespace glm53v {

constexpr int VD = 1024;          // vision_config.hidden_size
constexpr int VL = 24;            // vision_config.depth
constexpr int VH = 16;            // vision_config.num_heads
constexpr int VHD = VD / VH;      // 64
constexpr int VI = 4096;          // vision_config.intermediate_size
constexpr int TPS = 2;            // vision_config.temporal_patch_size: one frame repeated over 2 slots
constexpr int PK = 3 * TPS * PATCH * PATCH;   // 1176: patch embed input (channel, frame, y, x)
constexpr int OD_MID = 4096;      // downsample + merger.proj + post_projection_norm width
constexpr int OD_OUT = 6144;      // GLM-5.3 hidden_size (merger.down_proj out + row_break_embed width)
constexpr int MI = 10240;         // merger SwiGLU intermediate width
constexpr int MK = OD_MID;        // downsample input: 2 x 2 block of patches (4 x 1024)
constexpr float SWIGLU_LIMIT = 10.f;
constexpr float NORM_EPS = 1e-5f;
constexpr float LN_EPS = 1e-5f;
constexpr float ROPE_THETA = 10000.f;
constexpr int RF = VHD / 4;       // 16 rotary frequencies per axis (head_dim / 2 = 32 spatial dims, step 2)
// CLIP image_mean / image_std (identical to Glm5NextImageProcessor and exlr8-vproj)
constexpr float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float STDV[3] = {0.26862954f, 0.26130258f, 0.27577711f};
// FP32 budget of the GEMM accumulator
constexpr size_t ACC_FLOATS = (size_t)16 << 20;   // 64 MiB
constexpr int MLP_ROWS = (int)(ACC_FLOATS / (2 * VI));   // 2048: rows of one block-MLP pass
constexpr int MRG_ROWS = (int)(ACC_FLOATS / (2 * MI));   // 819: rows of one merger-MLP pass

#define GLM53V_CUDA(x) do { cudaError_t glm53v_ce_ = (x); if (glm53v_ce_ != cudaSuccess) throw std::runtime_error(std::string("glm53 vision: ") + #x + ": " + cudaGetErrorString(glm53v_ce_)); } while (0)
#define GLM53V_BLAS(x) do { cublasStatus_t glm53v_st_ = (x); if (glm53v_st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("glm53 vision: ") + #x + " failed: " + std::to_string((int)glm53v_st_)); } while (0)

typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------------------ kernels

// The patch at index n of the processor's order: 2 x 2 merge blocks row-major, the 4 patches of a block row-major.
__device__ __forceinline__ void patch_at(int n, int vit_w, int& row, int& col)
{
    const int blk = n >> 2, in = n & 3, bw = vit_w / MERGE;
    row = (blk / bw) * MERGE + (in >> 1);
    col = (blk % bw) * MERGE + (in & 1);
}

// pixels -> patches: element i = (c, t, py, px) of patch n, x = bf16((u - fl(mean * 255)) / fl(std * 255)) in fp32;
// the frame repeats over t. grid(n), block 256.
__global__ void k_patchify(const uint8_t* __restrict__ px, int vit_w, float4 ms0, float2 ms1, bf16* __restrict__ out)
{
    const int n = blockIdx.x;
    int row, col;
    patch_at(n, vit_w, row, col);
    const size_t W = (size_t)vit_w * PATCH;
    for (int i = threadIdx.x; i < PK; i += blockDim.x) {
        const int c = i / (TPS * PATCH * PATCH), py = (i / PATCH) % PATCH, pxx = i % PATCH;
        const float u = (float)px[(((size_t)row * PATCH + py) * W + (size_t)col * PATCH + pxx) * 3 + c];
        const float m = c == 0 ? ms0.x : c == 1 ? ms0.y : ms0.z, s = c == 0 ? ms0.w : c == 1 ? ms1.x : ms1.y;
        out[(size_t)n * PK + i] = __float2bfloat16_rn(__fdiv_rn(__fsub_rn(u, m), s));
    }
}

__device__ __forceinline__ float block_sum256(float v, float* red)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 8; ++k) t += red[k];
    __syncthreads();
    return t;
}

// RMSNorm over VD: y = bf16(w * (x * rsqrt(mean(x^2) + eps))). grid(rows), block 256.
__global__ void k_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ y)
{
    __shared__ float red[8];
    const int r = blockIdx.x;
    const bf16* xr = x + (size_t)r * VD;
    float v[VD / 256], ss = 0.f;
    for (int j = 0; j < VD / 256; ++j) { v[j] = __bfloat162float(xr[threadIdx.x + 256 * j]); ss += v[j] * v[j]; }
    const float rs = rsqrtf(block_sum256(ss, red) / (float)VD + NORM_EPS);
    for (int j = 0; j < VD / 256; ++j) {
        const int h = threadIdx.x + 256 * j;
        y[(size_t)r * VD + h] = __float2bfloat16_rn(__bfloat162float(w[h]) * (v[j] * rs));
    }
}

// LayerNorm over OD_MID (weight and bias), then exact GELU: h = bf16(ln(x)), y = bf16(gelu(h)). grid(rows), block 256.
__global__ void k_layernorm_gelu(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b, bf16* __restrict__ y)
{
    __shared__ float red[8];
    const int r = blockIdx.x;
    const bf16* xr = x + (size_t)r * OD_MID;
    float v[OD_MID / 256], s = 0.f;
    for (int j = 0; j < OD_MID / 256; ++j) { v[j] = __bfloat162float(xr[threadIdx.x + 256 * j]); s += v[j]; }
    const float mean = block_sum256(s, red) / (float)OD_MID;
    float ss = 0.f;
    for (int j = 0; j < OD_MID / 256; ++j) { const float d = v[j] - mean; ss += d * d; }
    const float rs = rsqrtf(block_sum256(ss, red) / (float)OD_MID + LN_EPS);
    for (int j = 0; j < OD_MID / 256; ++j) {
        const int k = threadIdx.x + 256 * j;
        const float h = __bfloat162float(__float2bfloat16_rn((v[j] - mean) * rs * __bfloat162float(w[k]) + __bfloat162float(b[k])));
        y[(size_t)r * OD_MID + k] = __float2bfloat16_rn(0.5f * h * (1.f + erff(h * 0.70710678118654752f)));
    }
}

enum Epi { E_BIAS = 0, E_BIAS_RES = 1, E_SWIGLU = 2 };

// GEMM epilogue on an fp32 [rows][N] block.
__global__ void k_epi(const float* __restrict__ acc, const bf16* __restrict__ b, bf16* __restrict__ out, int rows, int N, int mode)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int on = mode == E_SWIGLU ? N / 2 : N;
    if (i >= (size_t)rows * on) return;
    const int r = (int)(i / on), c = (int)(i % on);
    if (mode == E_SWIGLU) {
        float g = acc[(size_t)r * N + c], u = acc[(size_t)r * N + on + c];
        if (b) { g += __bfloat162float(b[c]); u += __bfloat162float(b[on + c]); }
        g = fminf(__bfloat162float(__float2bfloat16_rn(g)), SWIGLU_LIMIT);
        u = fminf(fmaxf(__bfloat162float(__float2bfloat16_rn(u)), -SWIGLU_LIMIT), SWIGLU_LIMIT);
        const float s = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
        out[i] = __float2bfloat16_rn(s * u);
        return;
    }
    float v = acc[i];
    if (b) v += __bfloat162float(b[c]);
    if (mode == E_BIAS) { out[i] = __float2bfloat16_rn(v); return; }
    const float o = __bfloat162float(__float2bfloat16_rn(v));
    out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) + o);
}

// Per-head RMSNorm of q and k (weights of 64), then axial 2-D rotary on the whole head, in place in qkv [n][3 VD].
// grid(n), block (32, 2 * VH).
__global__ void k_qk_norm_rope(bf16* __restrict__ qkv, const bf16* __restrict__ qw, const bf16* __restrict__ kw, int vit_w)
{
    const int n = blockIdx.x, j = threadIdx.x, hh = threadIdx.y;
    const bool isq = hh < VH;
    bf16* x = qkv + (size_t)n * 3 * VD + (size_t)(isq ? hh : VH + (hh - VH)) * VHD;
    const bf16* w = isq ? qw : kw;
    float a = __bfloat162float(x[j]), b = __bfloat162float(x[j + 32]);
    float ss = a * a + b * b;
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss / (float)VHD + NORM_EPS);
    a = __bfloat162float(__float2bfloat16_rn(__bfloat162float(w[j]) * (a * rs)));
    b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(w[j + 32]) * (b * rs)));
    int row, col;
    patch_at(n, vit_w, row, col);
    const int fi = j % RF;
    const float inv = 1.f / powf(ROPE_THETA, (float)(2 * fi) / (float)(2 * RF));
    const float ang = (float)(j < RF ? row : col) * inv;
    float s, c;
    sincosf(ang, &s, &c);
    x[j] = __float2bfloat16_rn(a * c - b * s);
    x[j + 32] = __float2bfloat16_rn(b * c + a * s);
}

// Fused unmasked FlashAttention over one image's patches per head (identical to glm5v::k_flash_attn and exlr8-vproj).
constexpr int FA_T = 64;
constexpr int FA_LD = VHD + 8;

__device__ __forceinline__ void fa_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t fa_pack(float lo, float hi)
{
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__global__ void __launch_bounds__(128) k_flash_attn(const bf16* __restrict__ qkv, bf16* __restrict__ o, int n)
{
    __shared__ __align__(16) bf16 sK[FA_T * FA_LD];
    __shared__ __align__(16) bf16 sV[FA_T * FA_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, ld = 3 * VD;
    const int qa = blockIdx.x * FA_T + warp * 16 + g, qb = qa + 8;
    uint32_t qf[4][4];
    {
        const bf16* qpa = qkv + (size_t)qa * ld + h * VHD;
        const bf16* qpb = qkv + (size_t)qb * ld + h * VHD;
        for (int kk = 0; kk < 4; ++kk) {
            const int c = kk * 16 + 2 * t;
            qf[kk][0] = qa < n ? *reinterpret_cast<const uint32_t*>(qpa + c) : 0u;
            qf[kk][1] = qb < n ? *reinterpret_cast<const uint32_t*>(qpb + c) : 0u;
            qf[kk][2] = qa < n ? *reinterpret_cast<const uint32_t*>(qpa + c + 8) : 0u;
            qf[kk][3] = qb < n ? *reinterpret_cast<const uint32_t*>(qpb + c + 8) : 0u;
        }
    }
    float oc[8][4];
    for (int j = 0; j < 8; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
    const float sc = 0.125f * 1.4426950408889634f;
    for (int k0 = 0; k0 < n; k0 += FA_T) {
        __syncthreads();
        for (int v = tid; v < FA_T * VHD / 8; v += 128) {
            const int r = v / (VHD / 8), c = (v % (VHD / 8)) * 8;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = make_uint4(0, 0, 0, 0);
            if (k0 + r < n) {
                const bf16* row = qkv + (size_t)(k0 + r) * ld + h * VHD + c;
                kx = *reinterpret_cast<const uint4*>(row + VD);
                vx = *reinterpret_cast<const uint4*>(row + 2 * VD);
            }
            *reinterpret_cast<uint4*>(sK + r * FA_LD + c) = kx;
            const bf16* ve = reinterpret_cast<const bf16*>(&vx);
            for (int i = 0; i < 8; ++i) sV[(c + i) * FA_LD + r] = ve[i];
        }
        __syncthreads();
        float sc_[8][4];
        for (int j = 0; j < 8; ++j) {
            sc_[j][0] = sc_[j][1] = sc_[j][2] = sc_[j][3] = 0.f;
            const bf16* kr = sK + (j * 8 + g) * FA_LD + 2 * t;
            for (int kk = 0; kk < 4; ++kk)
                fa_mma(sc_[j], qf[kk], *reinterpret_cast<const uint32_t*>(kr + kk * 16), *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 8));
        }
        const int valid = n - k0;
        float xa = -INFINITY, xb = -INFINITY;
        for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 2; ++e) {
                const int col = j * 8 + 2 * t + e;
                const bool ok = col < valid;
                sc_[j][e] = ok ? sc_[j][e] * sc : -INFINITY;
                sc_[j][2 + e] = ok ? sc_[j][2 + e] * sc : -INFINITY;
                xa = fmaxf(xa, sc_[j][e]);
                xb = fmaxf(xb, sc_[j][2 + e]);
            }
        for (int o2 = 1; o2 < 4; o2 <<= 1) {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
        }
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        const float aa = exp2f(ma - na), ab = exp2f(mb - nb);
        ma = na; mb = nb;
        la *= aa; lb *= ab;
        for (int j = 0; j < 8; ++j) { oc[j][0] *= aa; oc[j][1] *= aa; oc[j][2] *= ab; oc[j][3] *= ab; }
        uint32_t pf[4][4];
        for (int kk = 0; kk < 4; ++kk)
            for (int half = 0; half < 2; ++half) {
                const float* c = sc_[2 * kk + half];
                const __nv_bfloat162 pa = __floats2bfloat162_rn(exp2f(c[0] - na), exp2f(c[1] - na));
                const __nv_bfloat162 pb = __floats2bfloat162_rn(exp2f(c[2] - nb), exp2f(c[3] - nb));
                la += __low2float(pa) + __high2float(pa);
                lb += __low2float(pb) + __high2float(pb);
                pf[kk][half * 2] = *reinterpret_cast<const uint32_t*>(&pa);
                pf[kk][half * 2 + 1] = *reinterpret_cast<const uint32_t*>(&pb);
            }
        for (int j = 0; j < 8; ++j) {
            const bf16* vr = sV + (j * 8 + g) * FA_LD + 2 * t;
            for (int kk = 0; kk < 4; ++kk)
                fa_mma(oc[j], pf[kk], *reinterpret_cast<const uint32_t*>(vr + kk * 16), *reinterpret_cast<const uint32_t*>(vr + kk * 16 + 8));
        }
    }
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, o2);
        lb += __shfl_xor_sync(0xffffffffu, lb, o2);
    }
    const float ia = 1.f / la, ib = 1.f / lb;
    for (int j = 0; j < 8; ++j) {
        const int c = h * VHD + j * 8 + 2 * t;
        if (qa < n) *reinterpret_cast<uint32_t*>(o + (size_t)qa * VD + c) = fa_pack(oc[j][0] * ia, oc[j][1] * ia);
        if (qb < n) *reinterpret_cast<uint32_t*>(o + (size_t)qb * VD + c) = fa_pack(oc[j][2] * ib, oc[j][3] * ib);
    }
}

// Downsample weight [OD_MID][VD][2][2] -> [OD_MID][(i * 2 + j) * VD + c]. grid(OD_MID), block 256.
__global__ void k_perm_downsample(const bf16* __restrict__ w, bf16* __restrict__ out)
{
    const int o = blockIdx.x;
    for (int k = threadIdx.x; k < MK; k += blockDim.x) {
        const int ij = k / VD, c = k % VD;
        out[(size_t)o * MK + k] = w[(size_t)o * MK + c * 4 + ij];
    }
}

// Expands merger output [hp * wp][OD_OUT] into [hp * (wp + 1)][OD_OUT], inserting `row_break` [OD_OUT] at column wp
// of every merged row (exact parity with exlr8-vproj `k_row_break_splice_fwd`). grid(hp * (wp + 1)), block 256.
__global__ void k_row_break_expand(const bf16* __restrict__ merged, const bf16* __restrict__ row_break,
                                   bf16* __restrict__ out, int hp, int wp)
{
    const int idx = blockIdx.x;
    const int stride = wp + 1;
    const int pr = idx / stride, pc = idx % stride;
    const bf16* src = (pc == wp) ? row_break : (merged + (size_t)(pr * wp + pc) * OD_OUT);
    bf16* dst = out + (size_t)idx * OD_OUT;
    for (int d = threadIdx.x; d < OD_OUT; d += blockDim.x) dst[d] = src[d];
}

// ------------------------------------------------------------------------------------------------------------ weights

struct Block { const bf16 *n1, *wqkv, *bqkv, *qn, *kn, *wo, *bo, *n2, *wg, *wu, *bgu, *wd, *bd; };

struct Weights {
    void* blob = nullptr;
    size_t bytes = 0;
    bf16* bias_gu = nullptr;  // [VL][2 VI]: gate bias then up bias per block
    const bf16 *patch_w = nullptr, *patch_b = nullptr, *post_norm = nullptr;
    Block blk[VL];
    const bf16 *ds_w = nullptr, *ds_b = nullptr;
    const bf16 *m_proj = nullptr, *m_lnw = nullptr, *m_lnb = nullptr, *m_gate = nullptr, *m_up = nullptr, *m_down = nullptr;
    const bf16 *row_break = nullptr;   // [1, 6144]: model.visual.row_break_embed
};

// Reads the GLM-5.3 vision safetensors file (348 BF16 tensors) into device memory and checks every tensor's dtype and shape.
inline Weights load_weights(const std::string& path)
{
    const g6vis::Blob blob = g6vis::load_blob(path, "glm53 vision");
    Weights w;
    w.blob = blob.dev;
    w.bytes = blob.bytes;
    auto get = [&](const std::string& name, std::vector<int64_t> shape) -> const bf16* {
        return (const bf16*)blob.get("model.visual." + name, "BF16", shape, path, "glm53 vision");
    };
    w.patch_w = get("patch_embed.proj.weight", {VD, 3, TPS, PATCH, PATCH});
    w.patch_b = get("patch_embed.proj.bias", {VD});
    for (int l = 0; l < VL; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        Block& b = w.blk[l];
        b.n1 = get(p + "norm1.weight", {VD});
        b.wqkv = get(p + "attn.qkv.weight", {3 * VD, VD});
        b.bqkv = get(p + "attn.qkv.bias", {3 * VD});
        b.qn = get(p + "attn.q_norm.weight", {VHD});
        b.kn = get(p + "attn.k_norm.weight", {VHD});
        b.wo = get(p + "attn.proj.weight", {VD, VD});
        b.bo = get(p + "attn.proj.bias", {VD});
        b.n2 = get(p + "norm2.weight", {VD});
        b.wg = get(p + "mlp.gate_proj.weight", {VI, VD});
        b.wu = get(p + "mlp.up_proj.weight", {VI, VD});
        b.wd = get(p + "mlp.down_proj.weight", {VD, VI});
        b.bd = get(p + "mlp.down_proj.bias", {VD});
    }
    w.post_norm = get("post_layernorm.weight", {VD});
    w.ds_w = get("downsample.weight", {OD_MID, VD, MERGE, MERGE});
    w.ds_b = get("downsample.bias", {OD_MID});
    w.m_proj = get("merger.proj.weight", {OD_MID, OD_MID});
    w.m_lnw = get("merger.post_projection_norm.weight", {OD_MID});
    w.m_lnb = get("merger.post_projection_norm.bias", {OD_MID});
    w.m_gate = get("merger.gate_proj.weight", {MI, OD_MID});
    w.m_up = get("merger.up_proj.weight", {MI, OD_MID});
    w.m_down = get("merger.down_proj.weight", {OD_OUT, MI});
    w.row_break = get("row_break_embed", {1, OD_OUT});
    if (blob.hdr.size() != 348 || blob.hdr.count("model.visual.blocks." + std::to_string(VL) + ".norm1.weight"))
        throw std::runtime_error("glm53 vision: " + path + " is not the 24-block GLM-5.3 vision tower (348 tensors)");
    GLM53V_CUDA(cudaMalloc(&w.bias_gu, (size_t)VL * 2 * VI * 2));
    for (int l = 0; l < VL; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        w.blk[l].bgu = w.bias_gu + (size_t)l * 2 * VI;
        GLM53V_CUDA(cudaMemcpy(w.bias_gu + (size_t)l * 2 * VI, get(p + "mlp.gate_proj.bias", {VI}), (size_t)VI * 2, cudaMemcpyDeviceToDevice));
        GLM53V_CUDA(cudaMemcpy(w.bias_gu + (size_t)l * 2 * VI + VI, get(p + "mlp.up_proj.bias", {VI}), (size_t)VI * 2, cudaMemcpyDeviceToDevice));
    }
    bf16* tmp = nullptr;
    GLM53V_CUDA(cudaMalloc(&tmp, (size_t)OD_MID * MK * 2));
    GLM53V_CUDA(cudaMemcpy(tmp, w.ds_w, (size_t)OD_MID * MK * 2, cudaMemcpyDeviceToDevice));
    k_perm_downsample<<<OD_MID, 256>>>(tmp, const_cast<bf16*>(w.ds_w));
    GLM53V_CUDA(cudaGetLastError());
    GLM53V_CUDA(cudaDeviceSynchronize());
    GLM53V_CUDA(cudaFree(tmp));
    return w;
}

// ------------------------------------------------------------------------------------------------------------ the tower

class Tower {
public:
    Weights w;
    bool loaded() const { return w.blob != nullptr; }

    void load(const std::string& path) { w = load_weights(path); }

    // Allocates the workspace for the largest image (MAX_PATCHES rows), at boot so the KV pool is sized after it.
    void ensure_workspace()
    {
        if (ws_) return;
        GLM53V_BLAS(cublasCreate(&blas_));
        const size_t n = (size_t)MAX_PATCHES;
        size_t off = 0;
        auto take = [&](size_t bytes) { const size_t o = off; off += (bytes + 255) & ~size_t(255); return o; };
        static_assert((size_t)MAX_PATCHES * PK <= (size_t)MAX_PATCHES * 3 * VD, "patches fit in qkv_");
        static_assert((size_t)MAX_TOKENS * OD_MID <= (size_t)MAX_PATCHES * VD, "merger intermediate rows fit in x_ and o_");
        static_assert((size_t)MAX_TOKENS * OD_OUT <= (size_t)MAX_PATCHES * 3 * VD, "merger [L][6144] output fits in qkv_");
        const size_t o_pix = take(n * PATCH * PATCH * 3), o_x = take(n * VD * 2), o_h = take(n * VD * 2),
                     o_qkv = take(n * 3 * VD * 2), o_o = take(n * VD * 2),
                     o_m = take((size_t)std::max(MLP_ROWS * VI, MRG_ROWS * MI) * 2), o_acc = take(ACC_FLOATS * 4);
        GLM53V_CUDA(cudaMalloc(&ws_, off));
        ws_bytes_ = off;
        char* b = (char*)ws_;
        pix_ = (uint8_t*)(b + o_pix); x_ = (bf16*)(b + o_x); h_ = (bf16*)(b + o_h); qkv_ = (bf16*)(b + o_qkv);
        o_ = (bf16*)(b + o_o); m_ = (bf16*)(b + o_m); acc_ = (float*)(b + o_acc);
    }
    size_t workspace_bytes() const { return ws_bytes_; }

    // Runs the tower on host pixels (uint8 [vit_h * 14][vit_w * 14][3]) and writes the row-break-expanded rows
    // (bf16 [(vit_h / 2) * (vit_w / 2 + 1)][6144]) to device pointer `out`. Synchronous on `st` on return.
    void encode(const uint8_t* host_px, int vit_h, int vit_w, bf16* out, cudaStream_t st)
    {
        const int span = span_of(vit_h, vit_w);
        if (!span)
            throw std::runtime_error("glm53 vision: patch grid " + std::to_string(vit_h) + "x" + std::to_string(vit_w) + " is not a processor grid");
        ensure_workspace();
        GLM53V_BLAS(cublasSetStream(blas_, st));
        const int n = vit_h * vit_w, hp = vit_h / MERGE, wp = vit_w / MERGE, L = hp * wp;
        GLM53V_CUDA(cudaMemcpyAsync(pix_, host_px, (size_t)n * PATCH * PATCH * 3, cudaMemcpyHostToDevice, st));
        float M[3], S[3];
        for (int c = 0; c < 3; ++c) { M[c] = MEAN[c] * 255.f; S[c] = STDV[c] * 255.f; }
        bf16* pat = qkv_;
        k_patchify<<<n, 256, 0, st>>>(pix_, vit_w, make_float4(M[0], M[1], M[2], S[0]), make_float2(S[1], S[2]), pat);
        linear(pat, w.patch_w, w.patch_b, x_, n, VD, PK, E_BIAS, st);
        for (int l = 0; l < VL; ++l) {
            const Block& b = w.blk[l];
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n1, h_);
            linear(h_, b.wqkv, b.bqkv, qkv_, n, 3 * VD, VD, E_BIAS, st);
            k_qk_norm_rope<<<n, dim3(32, 2 * VH), 0, st>>>(qkv_, b.qn, b.kn, vit_w);
            attention(n, st);
            linear(o_, b.wo, b.bo, x_, n, VD, VD, E_BIAS_RES, st);
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n2, h_);
            mlp(h_, VD, b.wg, b.wu, b.bgu, VI, b.wd, b.bd, x_, VD, n, MLP_ROWS, E_BIAS_RES, st);
        }
        k_rmsnorm<<<n, 256, 0, st>>>(x_, w.post_norm, h_);
        // Downsample [L][4096] -> merger.proj [L][4096] -> LayerNorm+GELU [L][4096] -> merger MLP [L][6144] -> row-break expand [span][6144]
        linear(h_, w.ds_w, w.ds_b, qkv_, L, OD_MID, MK, E_BIAS, st);
        linear(qkv_, w.m_proj, nullptr, o_, L, OD_MID, OD_MID, E_BIAS, st);
        k_layernorm_gelu<<<L, 256, 0, st>>>(o_, w.m_lnw, w.m_lnb, x_);
        mlp(x_, OD_MID, w.m_gate, w.m_up, nullptr, MI, w.m_down, nullptr, qkv_, OD_OUT, L, MRG_ROWS, E_BIAS, st);
        k_row_break_expand<<<span, 256, 0, st>>>(qkv_, w.row_break, out, hp, wp);
        GLM53V_CUDA(cudaGetLastError());
        GLM53V_CUDA(cudaStreamSynchronize(st));
    }

    ~Tower()
    {
        if (blas_) cublasDestroy(blas_);
        if (ws_) cudaFree(ws_);
        if (w.bias_gu) cudaFree(w.bias_gu);
        if (w.blob) cudaFree(w.blob);
    }

private:
    cublasHandle_t blas_ = nullptr;
    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    uint8_t* pix_ = nullptr;
    bf16 *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *o_ = nullptr, *m_ = nullptr;
    float* acc_ = nullptr;

    void gemm(const bf16* x, int ldx, const bf16* W, int ldw, float* acc, int ldc, int rows, int N, int K)
    {
        const float one = 1.f, zero = 0.f;
        GLM53V_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, ldw, x, CUDA_R_16BF, ldx,
                                 &zero, acc, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    void linear(const bf16* x, const bf16* W, const bf16* bias, bf16* y, int M, int N, int K, int mode, cudaStream_t st)
    {
        const int rch = (int)std::min<size_t>(M, ACC_FLOATS / (size_t)N);
        for (int r0 = 0; r0 < M; r0 += rch) {
            const int rows = std::min(rch, M - r0);
            gemm(x + (size_t)r0 * K, K, W, K, acc_, N, rows, N, K);
            const size_t tot = (size_t)rows * N;
            k_epi<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(acc_, bias, y + (size_t)r0 * N, rows, N, mode);
        }
    }

    void mlp(const bf16* x, int K, const bf16* wg, const bf16* wu, const bf16* bgu, int I, const bf16* wd, const bf16* bd,
             bf16* y, int N, int M, int pass, int mode, cudaStream_t st)
    {
        for (int r0 = 0; r0 < M; r0 += pass) {
            const int rows = std::min(pass, M - r0);
            gemm(x + (size_t)r0 * K, K, wg, K, acc_, 2 * I, rows, I, K);
            gemm(x + (size_t)r0 * K, K, wu, K, acc_ + I, 2 * I, rows, I, K);
            const size_t tot = (size_t)rows * I;
            k_epi<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(acc_, bgu, m_, rows, 2 * I, E_SWIGLU);
            linear(m_, wd, bd, y + (size_t)r0 * N, rows, N, I, mode, st);
        }
    }

    void attention(int n, cudaStream_t st)
    {
        k_flash_attn<<<dim3((n + FA_T - 1) / FA_T, VH), 128, 0, st>>>(qkv_, o_, n);
    }
};

}  // namespace glm53v
