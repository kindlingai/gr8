// GLM-5.3-Flash vision tower (`glm5_next_vision`, transformers `Glm5NextVisionModel`, docs/FLASH-VISION.md), for image
// input. Every TP rank runs it on the same pixels (the token embedding table is whole on every rank), so the ranks
// exchange nothing for its rows.
//
// In: the padded RGB the front end sends (uint8 [vit_h * 14][vit_w * 14][3]: the processor's content resize and zero
// pad). Out: bf16 [vit_h * vit_w / 4][4096], one row per image token in the merged grid's row-major order. The prefill
// embed (FlashVision below) puts these rows into the 4 mHC streams in place of the <|image|> embedding row.
//
// Numerics follow the reference's bf16 run: bf16 activations between ops, every GEMM bf16 x bf16 with fp32 accumulation
// (cuBLAS), bias / residual / activation epilogues in fp32 rounded once to bf16, the norms and the rotary in fp32,
// attention scores and the online softmax in fp32 with bf16 probabilities (k_flash_attn). The golden test (tests/flash_vision_test.cu) compares
// the output with the reference's fp32 run (engine/ref/flash/vision_golden.py).
//
// Weights: the release's dense/vision-bf16.safetensors (347 bf16 tensors, `model.visual.*`), read whole into one device
// allocation at boot (1.13 GB). The workspace holds the largest image (8000 tokens, 32000 patches); the MLPs run in row
// passes, so it stays near 0.6 GB.
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

#include "flash_vision_plan.h"
#include "vision_weights.h"

namespace glm5v {

constexpr int VD = 1024;          // [cfg] vision_config.hidden_size
constexpr int VL = 24;            // [cfg] vision_config.depth
constexpr int VH = 16;            // [cfg] vision_config.num_heads
constexpr int VHD = VD / VH;      // 64
constexpr int VI = 4096;          // [cfg] vision_config.intermediate_size
constexpr int TPS = 2;            // [cfg] vision_config.temporal_patch_size: the one frame repeated over 2 slots
constexpr int PK = 3 * TPS * PATCH * PATCH;   // 1176, the patch embed's input (channel, frame, y, x)
constexpr int OD = 4096;          // [cfg] vision_config.out_hidden_size (the text hidden size)
constexpr int MI = 10240;         // [cfg] vision_config.projection_intermediate_size
constexpr int MK = OD;            // the downsample's input: a 2 x 2 block of patches (4 x 1024)
constexpr float SWIGLU_LIMIT = 10.f;     // [cfg] vision_config.swiglu_limit
constexpr float NORM_EPS = 1e-5f;        // [cfg] vision_config.rms_norm_eps: norm1, norm2, q/k norm, post_layernorm
constexpr float LN_EPS = 1e-5f;          // nn.LayerNorm's default: the merger's post_projection_norm
constexpr float ROPE_THETA = 10000.f;    // the axial rope's default theta
constexpr int RF = VHD / 4;              // 16 rotary frequencies per axis (head_dim / 2 = 32 spatial dims, step 2)
// processor_config.json image_mean / image_std (CLIP)
constexpr float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float STDV[3] = {0.26862954f, 0.26130258f, 0.27577711f};
// the fp32 budget of the GEMM accumulator
constexpr size_t ACC_FLOATS = (size_t)16 << 20;   // 64 MiB
constexpr int MLP_ROWS = (int)(ACC_FLOATS / (2 * VI));   // 2048: rows of one block-MLP pass
constexpr int MRG_ROWS = (int)(ACC_FLOATS / (2 * MI));   // 819: rows of one merger-MLP pass

#define GLM5V_CUDA(x) do { cudaError_t glm5v_ce_ = (x); if (glm5v_ce_ != cudaSuccess) throw std::runtime_error(std::string("flash vision: ") + #x + ": " + cudaGetErrorString(glm5v_ce_)); } while (0)
#define GLM5V_BLAS(x) do { cublasStatus_t glm5v_st_ = (x); if (glm5v_st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("flash vision: ") + #x + " failed: " + std::to_string((int)glm5v_st_)); } while (0)

typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------------------ kernels

// The patch at index n of the processor's order: 2 x 2 merge blocks row-major, the 4 patches of a block row-major.
__device__ __forceinline__ void patch_at(int n, int vit_w, int& row, int& col)
{
    const int blk = n >> 2, in = n & 3, bw = vit_w / MERGE;
    row = (blk / bw) * MERGE + (in >> 1);
    col = (blk % bw) * MERGE + (in & 1);
}

// pixels -> patches: element i = (c, t, py, px) of patch n, x = bf16((u - fl(mean * 255)) / fl(std * 255)) in fp32 (the
// torchvision backend's fused rescale + normalize); the frame repeats over t. grid(n), block 256.
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

// LayerNorm over OD (weight and bias), then exact GELU: h = bf16(ln(x)), y = bf16(gelu(h)). grid(rows), block 256.
__global__ void k_layernorm_gelu(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b, bf16* __restrict__ y)
{
    __shared__ float red[8];
    const int r = blockIdx.x;
    const bf16* xr = x + (size_t)r * OD;
    float v[OD / 256], s = 0.f;
    for (int j = 0; j < OD / 256; ++j) { v[j] = __bfloat162float(xr[threadIdx.x + 256 * j]); s += v[j]; }
    const float mean = block_sum256(s, red) / (float)OD;
    float ss = 0.f;
    for (int j = 0; j < OD / 256; ++j) { const float d = v[j] - mean; ss += d * d; }
    const float rs = rsqrtf(block_sum256(ss, red) / (float)OD + LN_EPS);
    for (int j = 0; j < OD / 256; ++j) {
        const int k = threadIdx.x + 256 * j;
        const float h = __bfloat162float(__float2bfloat16_rn((v[j] - mean) * rs * __bfloat162float(w[k]) + __bfloat162float(b[k])));
        y[(size_t)r * OD + k] = __float2bfloat16_rn(0.5f * h * (1.f + erff(h * 0.70710678118654752f)));
    }
}

enum Epi { E_BIAS = 0, E_BIAS_RES = 1, E_SWIGLU = 2 };

// The GEMM epilogue on an fp32 [rows][N] block. E_BIAS: out = bf16(acc (+ b)). E_BIAS_RES: o = bf16(acc + b),
// out = bf16(out + o) (the residual add of two bf16 tensors). E_SWIGLU (N = 2 I, out [rows][I], b [2 I] or null):
// g = min(bf16(acc[:I] + b[:I]), 10), u = clamp(bf16(acc[I:] + b[I:]), -10, 10), out = bf16(bf16(silu(g)) * u).
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

// Per-head RMSNorm of q and k (weights of 64), then the axial 2-D rotary on the whole head, in place in qkv [n][3 VD]:
// for the patch at (row, col), angle j < 16 is row * f_j and 16 <= j < 32 is col * f_(j-16), f_i = 1 / theta^(2i / 32);
// x1 = x[:32], x2 = x[32:], out = [x1 cos - x2 sin, x2 cos + x1 sin]. The norm's output is rounded to bf16 first, as the
// reference's q_norm / k_norm return it. grid(n), block (32, 2 * VH): warp y = q heads 0..15, then k heads 16..31.
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

// Fused attention over one image's patches (no mask: the tower attends to the whole image), per head:
// o = softmax(q k^T / 8) v, with the online softmax (a running row max and sum) over key tiles of FA_T, so no score
// matrix reaches device memory. Tensor-core mma.m16n8k16 (bf16 in, fp32 accumulate) with the accumulators in registers:
// warp w owns query rows 16w..16w+15 of the block's FA_T. The probabilities are rounded to bf16 for the P v product, and
// the row sum adds those rounded values, so o is normalised by what the product used. grid(ceil(n / FA_T), VH), block 128.
// In the m16n8k16 layout lane = 4 g + t holds rows g and g + 8, columns 2t and 2t + 1 of each 16 x 8 accumulator.
constexpr int FA_T = 64;                 // query rows per block, key rows per tile
constexpr int FA_LD = VHD + 8;           // bf16 row stride of the K and transposed-V tiles in shared memory

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
    __shared__ __align__(16) bf16 sK[FA_T * FA_LD];    // [key][dim]
    __shared__ __align__(16) bf16 sV[FA_T * FA_LD];    // [dim][key]: V transposed, so a B fragment is one 32-bit load
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, ld = 3 * VD;
    const int qa = blockIdx.x * FA_T + warp * 16 + g, qb = qa + 8;   // this lane's two query rows
    // Q as A fragments over the head's 4 k-steps of 16 dims (zero past row n)
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
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;   // running max (exp2 domain) and this lane's part of the sum
    const float sc = 0.125f * 1.4426950408889634f;              // 1/8, in the exp2 domain
    for (int k0 = 0; k0 < n; k0 += FA_T) {
        __syncthreads();   // every warp is done with the last tile
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
        // S [16][64] = Q K^T: 8 blocks of 8 keys
        float sc_[8][4];
        for (int j = 0; j < 8; ++j) {
            sc_[j][0] = sc_[j][1] = sc_[j][2] = sc_[j][3] = 0.f;
            const bf16* kr = sK + (j * 8 + g) * FA_LD + 2 * t;
            for (int kk = 0; kk < 4; ++kk)
                fa_mma(sc_[j], qf[kk], *reinterpret_cast<const uint32_t*>(kr + kk * 16), *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 8));
        }
        // the online softmax over this tile's valid keys
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
        const float aa = exp2f(ma - na), ab = exp2f(mb - nb);   // 0 on the first tile
        ma = na; mb = nb;
        la *= aa; lb *= ab;
        for (int j = 0; j < 8; ++j) { oc[j][0] *= aa; oc[j][1] *= aa; oc[j][2] *= ab; oc[j][3] *= ab; }
        // P as A fragments (keys = k), rounded to bf16; the sums add the rounded values
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
        // o [16][64] += P V: 8 blocks of 8 dims
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

// The downsample's weight [OD][VD][2][2] (Conv2d: out, channel, i, j) -> [OD][(i * 2 + j) * VD + c], so a 2 x 2 block of
// patches, 4 consecutive rows of the patch order, is one contiguous 4096-wide input row. grid(OD), block 256.
__global__ void k_perm_downsample(const bf16* __restrict__ w, bf16* __restrict__ out)
{
    const int o = blockIdx.x;
    for (int k = threadIdx.x; k < MK; k += blockDim.x) {
        const int ij = k / VD, c = k % VD;
        out[(size_t)o * MK + k] = w[(size_t)o * MK + c * 4 + ij];
    }
}

// ------------------------------------------------------------------------------------------------------------ weights

struct Block { const bf16 *n1, *wqkv, *bqkv, *qn, *kn, *wo, *bo, *n2, *wg, *wu, *bgu, *wd, *bd; };

struct Weights {
    void* blob = nullptr;     // the file's data section, whole
    size_t bytes = 0;
    bf16* bias_gu = nullptr;  // [VL][2 VI]: each block's gate bias then up bias, side by side as the SwiGLU epilogue reads them
    const bf16 *patch_w = nullptr, *patch_b = nullptr, *post_norm = nullptr;
    Block blk[VL];
    const bf16 *ds_w = nullptr, *ds_b = nullptr;          // downsample, ds_w permuted in place (k_perm_downsample)
    const bf16 *m_proj = nullptr, *m_lnw = nullptr, *m_lnb = nullptr, *m_gate = nullptr, *m_up = nullptr, *m_down = nullptr;
};

// Reads the vision file into device memory and checks every tensor's dtype and shape against the reference module.
inline Weights load_weights(const std::string& path)
{
    const g6vis::Blob blob = g6vis::load_blob(path, "flash vision");
    Weights w;
    w.blob = blob.dev;
    w.bytes = blob.bytes;
    auto get = [&](const std::string& name, std::vector<int64_t> shape) -> const bf16* {
        return (const bf16*)blob.get("model.visual." + name, "BF16", shape, path, "flash vision");
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
    w.ds_w = get("downsample.weight", {OD, VD, MERGE, MERGE});
    w.ds_b = get("downsample.bias", {OD});
    w.m_proj = get("merger.proj.weight", {OD, OD});
    w.m_lnw = get("merger.post_projection_norm.weight", {OD});
    w.m_lnb = get("merger.post_projection_norm.bias", {OD});
    w.m_gate = get("merger.gate_proj.weight", {MI, OD});
    w.m_up = get("merger.up_proj.weight", {MI, OD});
    w.m_down = get("merger.down_proj.weight", {OD, MI});
    if (blob.hdr.size() != 347 || blob.hdr.count("model.visual.blocks." + std::to_string(VL) + ".norm1.weight"))
        throw std::runtime_error("flash vision: " + path + " is not the 24-block glm5_next_vision tower (347 tensors)");
    // the gate and up biases side by side
    GLM5V_CUDA(cudaMalloc(&w.bias_gu, (size_t)VL * 2 * VI * 2));
    for (int l = 0; l < VL; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        w.blk[l].bgu = w.bias_gu + (size_t)l * 2 * VI;
        GLM5V_CUDA(cudaMemcpy(w.bias_gu + (size_t)l * 2 * VI, get(p + "mlp.gate_proj.bias", {VI}), (size_t)VI * 2, cudaMemcpyDeviceToDevice));
        GLM5V_CUDA(cudaMemcpy(w.bias_gu + (size_t)l * 2 * VI + VI, get(p + "mlp.up_proj.bias", {VI}), (size_t)VI * 2, cudaMemcpyDeviceToDevice));
    }
    // the downsample weight, permuted in place through a temporary copy
    bf16* tmp = nullptr;
    GLM5V_CUDA(cudaMalloc(&tmp, (size_t)OD * MK * 2));
    GLM5V_CUDA(cudaMemcpy(tmp, w.ds_w, (size_t)OD * MK * 2, cudaMemcpyDeviceToDevice));
    k_perm_downsample<<<OD, 256>>>(tmp, const_cast<bf16*>(w.ds_w));
    GLM5V_CUDA(cudaGetLastError());
    GLM5V_CUDA(cudaDeviceSynchronize());
    GLM5V_CUDA(cudaFree(tmp));
    return w;
}

// ------------------------------------------------------------------------------------------------------------ the tower

class Tower {
public:
    Weights w;
    bool loaded() const { return w.blob != nullptr; }

    void load(const std::string& path) { w = load_weights(path); }

    // The workspace of the largest image (MAX_PATCHES rows), at boot so the KV pool is sized after it.
    void ensure_workspace()
    {
        if (ws_) return;
        GLM5V_BLAS(cublasCreate(&blas_));
        const size_t n = (size_t)MAX_PATCHES, L = (size_t)MAX_TOKENS;
        size_t off = 0;
        auto take = [&](size_t bytes) { const size_t o = off; off += (bytes + 255) & ~size_t(255); return o; };
        // The patches live in qkv_ before the first block. The merger's [L][OD] rows live in qkv_ and o_ after the
        // last block.
        static_assert((size_t)MAX_PATCHES * PK <= (size_t)MAX_PATCHES * 3 * VD, "the patches fit in qkv_");
        static_assert((size_t)MAX_TOKENS * OD <= (size_t)MAX_PATCHES * VD, "a merger row set fits in o_ (and in qkv_)");
        const size_t o_pix = take(n * PATCH * PATCH * 3), o_x = take(n * VD * 2), o_h = take(n * VD * 2),
                     o_qkv = take(n * 3 * VD * 2), o_o = take(n * VD * 2),
                     o_m = take((size_t)std::max(MLP_ROWS * VI, MRG_ROWS * MI) * 2), o_acc = take(ACC_FLOATS * 4);
        (void)L;
        GLM5V_CUDA(cudaMalloc(&ws_, off));
        ws_bytes_ = off;
        char* b = (char*)ws_;
        pix_ = (uint8_t*)(b + o_pix); x_ = (bf16*)(b + o_x); h_ = (bf16*)(b + o_h); qkv_ = (bf16*)(b + o_qkv);
        o_ = (bf16*)(b + o_o); m_ = (bf16*)(b + o_m); acc_ = (float*)(b + o_acc);
    }
    size_t workspace_bytes() const { return ws_bytes_; }

    // Runs the tower on host pixels (uint8 [vit_h * 14][vit_w * 14][3]) and writes its rows (bf16 [tokens][4096]) to the
    // device pointer `out`. Synchronous on `st` when it returns (the pixels are copied first).
    void encode(const uint8_t* host_px, int vit_h, int vit_w, bf16* out, cudaStream_t st)
    {
        if (!span_of(vit_h, vit_w))
            throw std::runtime_error("flash vision: patch grid " + std::to_string(vit_h) + "x" + std::to_string(vit_w) + " is not a processor grid");
        ensure_workspace();
        GLM5V_BLAS(cublasSetStream(blas_, st));
        const int n = vit_h * vit_w, L = n / (MERGE * MERGE);
        GLM5V_CUDA(cudaMemcpyAsync(pix_, host_px, (size_t)n * PATCH * PATCH * 3, cudaMemcpyHostToDevice, st));
        float M[3], S[3];
        for (int c = 0; c < 3; ++c) { M[c] = MEAN[c] * 255.f; S[c] = STDV[c] * 255.f; }
        bf16* pat = qkv_;
        k_patchify<<<n, 256, 0, st>>>(pix_, vit_w, make_float4(M[0], M[1], M[2], S[0]), make_float2(S[1], S[2]), pat);
        linear(pat, w.patch_w, w.patch_b, x_, n, VD, PK, E_BIAS, st);           // x = patch_embed(patches)
        for (int l = 0; l < VL; ++l) {
            const Block& b = w.blk[l];
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n1, h_);
            linear(h_, b.wqkv, b.bqkv, qkv_, n, 3 * VD, VD, E_BIAS, st);
            k_qk_norm_rope<<<n, dim3(32, 2 * VH), 0, st>>>(qkv_, b.qn, b.kn, vit_w);
            attention(n, st);
            linear(o_, b.wo, b.bo, x_, n, VD, VD, E_BIAS_RES, st);       // x = x + proj(o)
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n2, h_);
            mlp(h_, VD, b.wg, b.wu, b.bgu, VI, b.wd, b.bd, x_, VD, n, MLP_ROWS, E_BIAS_RES, st);   // x = x + mlp(.)
        }
        k_rmsnorm<<<n, 256, 0, st>>>(x_, w.post_norm, h_);
        // the downsample reads h_ as [L][4096]: a 2 x 2 block is 4 consecutive patch rows
        bf16* d = qkv_;
        bf16* a = o_;
        linear(h_, w.ds_w, w.ds_b, d, L, OD, MK, E_BIAS, st);
        linear(d, w.m_proj, nullptr, a, L, OD, OD, E_BIAS, st);
        k_layernorm_gelu<<<L, 256, 0, st>>>(a, w.m_lnw, w.m_lnb, d);
        mlp(d, OD, w.m_gate, w.m_up, nullptr, MI, w.m_down, nullptr, out, OD, L, MRG_ROWS, E_BIAS, st);
        GLM5V_CUDA(cudaGetLastError());
        GLM5V_CUDA(cudaStreamSynchronize(st));
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

    // acc [rows][N] (row stride ldc) = x [rows][K] (row stride ldx) * W [N][K]^T (row stride ldw)
    void gemm(const bf16* x, int ldx, const bf16* W, int ldw, float* acc, int ldc, int rows, int N, int K)
    {
        const float one = 1.f, zero = 0.f;
        GLM5V_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, ldw, x, CUDA_R_16BF, ldx,
                                &zero, acc, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // y [M][N] (epilogue) = x [M][K] * W [N][K]^T, in passes through the accumulator
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

    // y (epilogue `mode` of down) = down(silu(min(gate(x), 10)) * clamp(up(x), -10, 10)) in passes of `pass` rows: gate
    // and up land side by side in the accumulator ([rows][2 I]), the SwiGLU product in m_ ([rows][I]). bgu: the gate
    // bias then the up bias ([2 I]), or null.
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

    // o [n][VD] = softmax(q k^T / 8) v per head (k_flash_attn); q, k, v are the three VD-wide parts of qkv_
    void attention(int n, cudaStream_t st)
    {
        k_flash_attn<<<dim3((n + FA_T - 1) / FA_T, VH), 128, 0, st>>>(qkv_, o_, n);
    }
};

}  // namespace glm5v
