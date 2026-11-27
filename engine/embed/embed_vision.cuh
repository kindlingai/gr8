// Native Qwen3-VL CUDA vision tower + main/DeepStack mergers for g6embed (2B and 8B).
// Matches transformers Qwen3VLVisionModel:
//   - 2x2 merge-block patchify (patch=16, temporal=2, (u/255 - 0.5)/0.5)
//   - patch_embed.proj BF16 linear [N, 1536] -> [N, H_v] + bias
//   - fast_pos_embed_interpolate (align_corners=True bilinear from [48*48, H_v])
//   - L ViT blocks (24 on 2B H_v=1024 I_v=4096 hd=64; 27 on 8B H_v=1152 I_v=4304 hd=72):
//       LayerNorm(eps=1e-6) -> QKV -> 2D axial RoPE (theta=10000) -> packed bidirectional attn -> proj + res
//       LayerNorm(eps=1e-6) -> FC1 -> gelu_pytorch_tanh -> FC2 + res
//   - Main merger (use_postshuffle_norm=False) & DeepStack mergers at [5,11,17] (2B) / [8,16,24] (8B) (postshuffle=True)
//     with exact GELU (erff).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "embed_config.h"
#include "embed_gemm.cuh"
#include "embed_proto.h"
#include "embed_text.cuh"
#include "embed_weights.h"

namespace g6emb {

// ------------------------------------------------------------------------------------------------ Vision CUDA kernels

// Map flat patch index p (in 2x2 spatial-merge block order) to (patch_row, patch_col).
__device__ __forceinline__ void q3v_patch_rc(int p, int vit_w, int& pr, int& pc) {
    const int bw = vit_w >> 1;
    const int blk = p >> 2;
    const int sub = p & 3;
    pr = (blk / bw) * 2 + (sub >> 1);
    pc = (blk % bw) * 2 + (sub & 1);
}

// Patchify one RGB image [vit_h*16, vit_w*16, 3] into [N_img, 1536] BF16 (C=3, T=2 repeated, Ph=16, Pw=16).
// Grid: (N_img), Block: 256.
__global__ void __launch_bounds__(256) k_q3v_patchify(
    const uint8_t* __restrict__ rgb, bf16* __restrict__ out, int vit_h, int vit_w) {
    const int p = blockIdx.x;
    const int n_img = vit_h * vit_w;
    if (p >= n_img) return;
    int pr = 0, pc = 0;
    q3v_patch_rc(p, vit_w, pr, pc);
    const int W_px = vit_w * 16;
    bf16* dst = out + size_t(p) * 1536;

    // 1536 = 3 * 2 * 16 * 16 -> 6 elements per thread for blockDim.x = 256
    #pragma unroll
    for (int i = threadIdx.x; i < 1536; i += 256) {
        const int px = i & 15;
        const int py = (i >> 4) & 15;
        // t = (i >> 8) & 1 is repeated (temporal_patch_size = 2)
        const int c = i >> 9; // 0, 1, 2
        const int y = pr * 16 + py;
        const int x = pc * 16 + px;
        const float u = static_cast<float>(rgb[(size_t(y) * W_px + x) * 3 + c]);
        const float v = (u * (1.0f / 255.0f) - 0.5f) * 2.0f;
        dst[i] = __float2bfloat16_rn(v);
    }
}

// Bilinear pos_embed interpolation (align_corners=True, matching Qwen3VLVisionModel.fast_pos_embed_interpolate).
// Grid: (N_img), Block: 256.
__global__ void __launch_bounds__(256) k_q3v_pos_add(
    bf16* __restrict__ x, const bf16* __restrict__ pos_embed,
    int vit_h, int vit_w, int grid, int H_v) {
    const int p = blockIdx.x;
    if (p >= vit_h * vit_w) return;
    int pr = 0, pc = 0;
    q3v_patch_rc(p, vit_w, pr, pc);

    const float rh = (vit_h > 1) ? static_cast<float>(pr) * static_cast<float>(grid - 1) / static_cast<float>(vit_h - 1) : 0.0f;
    const float rw = (vit_w > 1) ? static_cast<float>(pc) * static_cast<float>(grid - 1) / static_cast<float>(vit_w - 1) : 0.0f;
    const int r0 = static_cast<int>(floorf(rh));
    const int c0 = static_cast<int>(floorf(rw));
    const int r1 = min(r0 + 1, grid - 1);
    const int c1 = min(c0 + 1, grid - 1);
    const float dr = rh - static_cast<float>(r0);
    const float dc = rw - static_cast<float>(c0);

    const float w00 = (1.0f - dr) * (1.0f - dc);
    const float w01 = (1.0f - dr) * dc;
    const float w10 = dr * (1.0f - dc);
    const float w11 = dr * dc;

    const bf16* p00 = pos_embed + size_t(r0 * grid + c0) * H_v;
    const bf16* p01 = pos_embed + size_t(r0 * grid + c1) * H_v;
    const bf16* p10 = pos_embed + size_t(r1 * grid + c0) * H_v;
    const bf16* p11 = pos_embed + size_t(r1 * grid + c1) * H_v;
    bf16* row = x + size_t(p) * H_v;

    for (int d = threadIdx.x; d < H_v; d += 256) {
        const float pe = w00 * __bfloat162float(p00[d]) + w01 * __bfloat162float(p01[d]) +
                         w10 * __bfloat162float(p10[d]) + w11 * __bfloat162float(p11[d]);
        const float pe_bf = __bfloat162float(__float2bfloat16_rn(pe));
        row[d] = __float2bfloat16_rn(__bfloat162float(row[d]) + pe_bf);
    }
}

// Standard nn.LayerNorm(D, eps=1e-6) with weight and bias (works for D = 1024, 1152, 4096, 4608, etc.).
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_q3v_layernorm(
    const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
    bf16* __restrict__ y, int M, int D, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    if (m >= M) return;
    const bf16* xr = x + size_t(m) * D;
    bf16* yr = y + size_t(m) * D;

    float sum = 0.0f;
    for (int d = threadIdx.x * 4; d < D; d += 256 * 4) {
        const uint2 r = *reinterpret_cast<const uint2*>(xr + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        sum += a.x + a.y + c.x + c.y;
    }
    const float mean = emb_block_sum256(sum, red) / static_cast<float>(D);

    float vs = 0.0f;
    for (int d = threadIdx.x * 4; d < D; d += 256 * 4) {
        const uint2 r = *reinterpret_cast<const uint2*>(xr + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        const float d0 = a.x - mean, d1 = a.y - mean, d2 = c.x - mean, d3 = c.y - mean;
        vs += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    const float rstd = rsqrtf(emb_block_sum256(vs, red) / static_cast<float>(D) + eps);

    for (int d = threadIdx.x * 4; d < D; d += 256 * 4) {
        const uint2 r = *reinterpret_cast<const uint2*>(xr + d);
        const uint2 wr = *reinterpret_cast<const uint2*>(w + d);
        const uint2 br = *reinterpret_cast<const uint2*>(b + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        const float2 wa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.x));
        const float2 wc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wr.y));
        const float2 ba = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&br.x));
        const float2 bc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&br.y));
        const float o0 = (a.x - mean) * rstd * wa.x + ba.x;
        const float o1 = (a.y - mean) * rstd * wa.y + ba.y;
        const float o2 = (c.x - mean) * rstd * wc.x + bc.x;
        const float o3 = (c.y - mean) * rstd * wc.y + bc.y;
        *reinterpret_cast<uint2*>(yr + d) = make_uint2(emb_pack_bf2(o0, o1), emb_pack_bf2(o2, o3));
    }
}

// 2D axial rotate_half RoPE applied in-place to Q and K inside qkv[N_img, 3, heads, hd].
// Matches Qwen3VLVisionRotaryEmbedding (dim = hd / 2, theta = 10000): first hd/4 pairs use row, next hd/4 use col.
// Grid: (N_img, heads), Block: 64 (threads j < hd/2 handle pair (j, j + hd/2) for both Q and K).
__global__ void __launch_bounds__(64) k_q3v_rope(
    bf16* __restrict__ qkv, int vit_h, int vit_w, int heads, int hd) {
    const int p = blockIdx.x;
    const int h = blockIdx.y;
    const int j = threadIdx.x;
    const int half = hd >> 1;
    if (p >= vit_h * vit_w || h >= heads || j >= half) return;

    int pr = 0, pc = 0;
    q3v_patch_rc(p, vit_w, pr, pc);
    const int rf = hd >> 2; // 16 for hd=64 (2B), 18 for hd=72 (8B)
    const int freq_idx = (j < rf) ? j : (j - rf);
    const int coord = (j < rf) ? pr : pc;
    const float inv_freq = 1.0f / powf(10000.0f, static_cast<float>(2 * freq_idx) / static_cast<float>(half));
    float s = 0.0f, c = 0.0f;
    __sincosf(static_cast<float>(coord) * inv_freq, &s, &c);

    const int H_v = heads * hd;
    bf16* row = qkv + size_t(p) * (3 * H_v);
    #pragma unroll
    for (int qk = 0; qk < 2; ++qk) {
        bf16* vec = row + qk * H_v + h * hd;
        const float x0 = __bfloat162float(vec[j]);
        const float x1 = __bfloat162float(vec[j + half]);
        vec[j] = __float2bfloat16_rn(x0 * c - x1 * s);
        vec[j + half] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

// Bidirectional packed ViT self-attention over images (cu_img_patches).
// Reads Q, K, V from qkv[total_patches, 3 * H_v] and writes out[total_patches, H_v].
// Supports arbitrary head_dim <= 128 (e.g. 64 on 2B, 72 on 8B).
// 4 warps/CTA, each warp handles 1 query patch. Grid: (ceil(total_patches / 4), heads), Block: 128.
constexpr int VIT_FA_QW = 4;

__global__ void __launch_bounds__(128) k_q3v_attn(
    const bf16* __restrict__ qkv,
    bf16* __restrict__ out,
    const int* __restrict__ cu_patches,
    const int* __restrict__ img_of_patch,
    int total_patches,
    int heads,
    int hd) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_idx = blockIdx.x * VIT_FA_QW + warp;
    const int h = blockIdx.y;
    if (q_idx >= total_patches || h >= heads) return;

    const int img = img_of_patch[q_idx];
    const int p_start = cu_patches[img];
    const int p_end = cu_patches[img + 1];
    const int H_v = heads * hd;
    const float scale = rsqrtf(static_cast<float>(hd));

    const bf16* q_ptr = qkv + size_t(q_idx) * (3 * H_v) + h * hd;
    float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float o_reg[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (d < hd) q_reg[i] = __bfloat162float(q_ptr[d]) * scale;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;
    for (int k_idx = p_start; k_idx < p_end; ++k_idx) {
        const bf16* kv_row = qkv + size_t(k_idx) * (3 * H_v);
        const bf16* k_ptr = kv_row + H_v + h * hd;
        const bf16* v_ptr = kv_row + 2 * H_v + h * hd;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (d < hd) dot += q_reg[i] * __bfloat162float(k_ptr[d]);
        }
        dot = emb_warp_sum(dot);

        const float m_new = fmaxf(m_max, dot);
        const float alpha = __expf(m_max - m_new);
        const float p_w = __expf(dot - m_new);
        l_sum = l_sum * alpha + p_w;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (d < hd) o_reg[i] = o_reg[i] * alpha + p_w * __bfloat162float(v_ptr[d]);
        }
        m_max = m_new;
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    bf16* o_ptr = out + size_t(q_idx) * H_v + h * hd;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (d < hd) o_ptr[d] = __float2bfloat16_rn(o_reg[i] * inv_l);
    }
}

// gelu_pytorch_tanh in-place for Qwen3VLVisionMLP.
__global__ void __launch_bounds__(256) k_q3v_gelu_tanh(bf16* __restrict__ x, size_t n) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    const float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = __float2bfloat16_rn(0.5f * v * (1.0f + tanhf(c)));
}

// Exact GELU (erff) in-place for Qwen3VLVisionPatchMerger.
__global__ void __launch_bounds__(256) k_q3v_gelu_exact(bf16* __restrict__ x, size_t n) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16_rn(0.5f * v * (1.0f + erff(v * 0.7071067811865475f)));
}

// ------------------------------------------------------------------------------------------------ VisionRunner

struct VisionRunner {
    const EmbedWeights* W = nullptr;
    int cap_patches = 0;
    size_t cap_rgb_bytes = 0;

    uint8_t* d_rgb = nullptr;
    int* d_cu_patches = nullptr;
    int* d_img_of_patch = nullptr;
    int cap_imgs = 0;

    bf16* d_patches = nullptr; // [cap_patches, 1536]
    bf16* d_x = nullptr;       // [cap_patches, H_v]
    bf16* d_ln = nullptr;      // [cap_patches, H_v]
    bf16* d_qkv = nullptr;     // [cap_patches, 3 * H_v]
    bf16* d_attn = nullptr;    // [cap_patches, H_v]
    bf16* d_mlp = nullptr;     // [cap_patches, max(I_v, 4 * H_v)]
    bf16* d_m_ln = nullptr;    // [cap_tokens, 4 * H_v]
    bf16* d_m_fc1 = nullptr;   // [cap_tokens, 4 * H_v]
    bf16* d_main_out = nullptr;// [cap_tokens, H_text]
    std::vector<bf16*> d_ds_out; // [n_ds][cap_tokens, H_text]

    void init(const EmbedWeights& weights) {
        W = &weights;
        d_ds_out.assign(weights.deepstack_mergers.size(), nullptr);
    }

    void release() {
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fre(d_rgb);
        fre(d_cu_patches);
        fre(d_img_of_patch);
        fre(d_patches);
        fre(d_x);
        fre(d_ln);
        fre(d_qkv);
        fre(d_attn);
        fre(d_mlp);
        fre(d_m_ln);
        fre(d_m_fc1);
        fre(d_main_out);
        for (auto*& p : d_ds_out) fre(p);
        cap_patches = 0;
        cap_rgb_bytes = 0;
        cap_imgs = 0;
    }
    ~VisionRunner() { release(); }

    void ensure_capacity(int need_patches, size_t need_rgb_bytes, int need_imgs) {
        const EmbedConfig& c = W->cfg;
        if (need_patches > cap_patches) {
            auto realloc_bf = [](bf16*& p, size_t elems) {
                if (p) cudaFree(p);
                p = nullptr;
                EMB_CK(cudaMalloc(&p, std::max<size_t>(256, elems * sizeof(bf16))));
            };
            const int np = std::max(need_patches, 256);
            const int nt = (np + 3) / 4;
            const int H_v = c.vit_hidden;
            const int I_v = std::max(c.vit_intermediate, 4 * H_v);
            realloc_bf(d_patches, size_t(np) * 1536);
            realloc_bf(d_x, size_t(np) * H_v);
            realloc_bf(d_ln, size_t(np) * H_v);
            realloc_bf(d_qkv, size_t(np) * 3 * H_v);
            realloc_bf(d_attn, size_t(np) * H_v);
            realloc_bf(d_mlp, size_t(np) * I_v);
            realloc_bf(d_m_ln, size_t(nt) * 4 * H_v);
            realloc_bf(d_m_fc1, size_t(nt) * 4 * H_v);
            realloc_bf(d_main_out, size_t(nt) * c.hidden_size);
            for (auto*& p : d_ds_out) realloc_bf(p, size_t(nt) * c.hidden_size);
            if (d_img_of_patch) cudaFree(d_img_of_patch);
            EMB_CK(cudaMalloc(&d_img_of_patch, size_t(np) * sizeof(int)));
            cap_patches = np;
        }
        if (need_rgb_bytes > cap_rgb_bytes) {
            if (d_rgb) cudaFree(d_rgb);
            cap_rgb_bytes = std::max<size_t>(need_rgb_bytes, 4096);
            EMB_CK(cudaMalloc(&d_rgb, cap_rgb_bytes));
        }
        if (need_imgs + 1 > cap_imgs) {
            if (d_cu_patches) cudaFree(d_cu_patches);
            cap_imgs = std::max(need_imgs + 1, 16);
            EMB_CK(cudaMalloc(&d_cu_patches, size_t(cap_imgs) * sizeof(int)));
        }
    }

    void run_merger(const bf16* src_x, const MergerW& M, bf16* dst_out,
                    int total_patches, int total_tokens, GemmWorkspace& ws, cudaStream_t st) {
        const EmbedConfig& c = W->cfg;
        const int H_v = c.vit_hidden;
        const int D_m = 4 * H_v;
        if (M.use_postshuffle_norm) {
            k_q3v_layernorm<<<total_tokens, 256, 0, st>>>(src_x, M.norm_w, M.norm_b, d_m_ln, total_tokens, D_m, 1e-6f);
        } else {
            // Patches are already ordered in 2x2 merge blocks, so [total_patches, H_v] IS [total_tokens, 4*H_v]!
            k_q3v_layernorm<<<total_patches, 256, 0, st>>>(src_x, M.norm_w, M.norm_b, d_m_ln, total_patches, H_v, 1e-6f);
        }
        linear_forward(d_m_ln, M.fc1_w, M.fc1_b, d_m_fc1, total_tokens, false, ws, st);
        const size_t n_fc1 = size_t(total_tokens) * D_m;
        k_q3v_gelu_exact<<<static_cast<unsigned>((n_fc1 + 255) / 256), 256, 0, st>>>(d_m_fc1, n_fc1);
        linear_forward(d_m_fc1, M.fc2_w, M.fc2_b, dst_out, total_tokens, false, ws, st);
    }

    // Encodes all images across `batch`, returning PackedVisionFeatures whose device pointers remain valid
    // until the next encode_batch() call. Returns empty PackedVisionFeatures if the batch has no images.
    PackedVisionFeatures encode_batch(const std::vector<EmbedRequest>& batch, GemmWorkspace& ws,
                                      EmbedProfiler* prof, cudaStream_t st) {
        PackedVisionFeatures out;
        if (!W || !W->cfg.has_vision) return out;

        struct ImgRef {
            const EmbedImage* span = nullptr;
            int patch_off = 0;
            size_t rgb_off = 0;
        };
        std::vector<ImgRef> imgs;
        int total_patches = 0;
        int total_tokens = 0;
        size_t total_rgb = 0;
        int tok_base = 0;

        for (const auto& req : batch) {
            for (const auto& im : req.images) {
                if (im.vit_h <= 0 || im.vit_w <= 0 || (im.vit_h & 1) || (im.vit_w & 1)) {
                    throw std::runtime_error("g6embed vision: invalid vit_h/vit_w grid");
                }
                const int n_p = im.vit_h * im.vit_w;
                const int n_t = n_p / 4;
                const size_t need_rgb = size_t(im.vit_h) * 16 * size_t(im.vit_w) * 16 * 3;
                if (im.rgb.size() != need_rgb) {
                    throw std::runtime_error("g6embed vision: RGB byte length mismatch");
                }
                imgs.push_back(ImgRef{&im, total_patches, total_rgb});
                for (int k = 0; k < n_t; ++k) {
                    out.img_token_positions.push_back(tok_base + im.pos + k);
                }
                total_patches += n_p;
                total_tokens += n_t;
                total_rgb += need_rgb;
            }
            tok_base += static_cast<int>(req.prompt.size());
        }
        if (imgs.empty()) return out;

        const EmbedConfig& c = W->cfg;
        const int H_v = c.vit_hidden;
        const int I_v = c.vit_intermediate;
        ensure_capacity(total_patches, total_rgb, static_cast<int>(imgs.size()));
        ws.ensure_rot(total_patches, std::max({H_v, I_v, 4 * H_v}));

        // Pack RGB bytes and image index tables on host, upload once
        std::vector<uint8_t> h_rgb(total_rgb);
        std::vector<int> h_cu(imgs.size() + 1, 0);
        std::vector<int> h_iof(size_t(total_patches), 0);
        for (size_t i = 0; i < imgs.size(); ++i) {
            const EmbedImage& im = *imgs[i].span;
            std::memcpy(h_rgb.data() + imgs[i].rgb_off, im.rgb.data(), im.rgb.size());
            h_cu[i] = imgs[i].patch_off;
            const int n_p = im.vit_h * im.vit_w;
            for (int p = 0; p < n_p; ++p) h_iof[size_t(imgs[i].patch_off + p)] = static_cast<int>(i);
        }
        h_cu[imgs.size()] = total_patches;

        EMB_CK(cudaMemcpyAsync(d_rgb, h_rgb.data(), total_rgb, cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_cu_patches, h_cu.data(), h_cu.size() * sizeof(int), cudaMemcpyHostToDevice, st));
        EMB_CK(cudaMemcpyAsync(d_img_of_patch, h_iof.data(), h_iof.size() * sizeof(int), cudaMemcpyHostToDevice, st));

        auto run_body = [&] {
            // 1. Patchify each image
            for (const auto& ir : imgs) {
                const int n_p = ir.span->vit_h * ir.span->vit_w;
                k_q3v_patchify<<<n_p, 256, 0, st>>>(
                    d_rgb + ir.rgb_off, d_patches + size_t(ir.patch_off) * 1536, ir.span->vit_h, ir.span->vit_w);
            }
            // 2. Patch embedding linear + bias
            bf16_linear_forward(d_patches, W->patch_proj_w, W->patch_proj_b, d_x, total_patches, H_v, 1536, false, ws, st);
            // 3. Bilinear positional embedding add
            for (const auto& ir : imgs) {
                const int n_p = ir.span->vit_h * ir.span->vit_w;
                k_q3v_pos_add<<<n_p, 256, 0, st>>>(
                    d_x + size_t(ir.patch_off) * H_v, W->pos_embed,
                    ir.span->vit_h, ir.span->vit_w, c.vit_grid_size, H_v);
            }
            // 4. ViT transformer blocks + DeepStack hooks
            for (int l = 0; l < c.vit_layers; ++l) {
                const VisionBlockW& B = W->vision_blocks[size_t(l)];
                k_q3v_layernorm<<<total_patches, 256, 0, st>>>(d_x, B.norm1_w, B.norm1_b, d_ln, total_patches, H_v, 1e-6f);
                linear_forward(d_ln, B.qkv_w, B.qkv_b, d_qkv, total_patches, false, ws, st);
                for (const auto& ir : imgs) {
                    const int n_p = ir.span->vit_h * ir.span->vit_w;
                    k_q3v_rope<<<dim3(n_p, c.vit_heads), 64, 0, st>>>(
                        d_qkv + size_t(ir.patch_off) * (3 * H_v), ir.span->vit_h, ir.span->vit_w,
                        c.vit_heads, c.vit_head_dim);
                }
                const dim3 ag((total_patches + VIT_FA_QW - 1) / VIT_FA_QW, c.vit_heads);
                k_q3v_attn<<<ag, 128, 0, st>>>(
                    d_qkv, d_attn, d_cu_patches, d_img_of_patch, total_patches, c.vit_heads, c.vit_head_dim);
                linear_forward(d_attn, B.proj_w, B.proj_b, d_x, total_patches, true, ws, st);

                k_q3v_layernorm<<<total_patches, 256, 0, st>>>(d_x, B.norm2_w, B.norm2_b, d_ln, total_patches, H_v, 1e-6f);
                linear_forward(d_ln, B.fc1_w, B.fc1_b, d_mlp, total_patches, false, ws, st);
                const size_t n_mlp = size_t(total_patches) * I_v;
                k_q3v_gelu_tanh<<<static_cast<unsigned>((n_mlp + 255) / 256), 256, 0, st>>>(d_mlp, n_mlp);
                linear_forward(d_mlp, B.fc2_w, B.fc2_b, d_x, total_patches, true, ws, st);

                for (size_t ds = 0; ds < c.deepstack_indexes.size(); ++ds) {
                    if (c.deepstack_indexes[ds] == l) {
                        run_merger(d_x, W->deepstack_mergers[ds], d_ds_out[ds], total_patches, total_tokens, ws, st);
                    }
                }
            }
            // 5. Main patch merger
            run_merger(d_x, W->main_merger, d_main_out, total_patches, total_tokens, ws, st);
        };

        if (prof) {
            prof->record(prof->vision_ms, st, run_body);
        } else {
            run_body();
        }

        out.n_img_tokens = total_tokens;
        out.main_vis = d_main_out;
        out.deepstack_vis.resize(d_ds_out.size());
        for (size_t ds = 0; ds < d_ds_out.size(); ++ds) out.deepstack_vis[ds] = d_ds_out[ds];
        return out;
    }
};

} // namespace g6emb
