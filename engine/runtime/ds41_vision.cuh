// DS41 vision tower + aligner (DeepSeek-V4.1-Flash-1017 `inference/vision.py`, `model.py` encode_image /
// merge_image_embeddings), for image input. Every TP rank runs it on the same pixels (the token embedding table is
// already whole on every rank), so its rows need no fabric traffic.
//
// In: the padded RGB the front end sends (uint8 [vit_h * 14][vit_w * 14][3], the output of the official plan_image_grid +
// ImageOps.pad). Out: the image span, bf16 [n_llm_h * (n_llm_w + 1) + 2][5120]: image_start, then each aligner row of
// n_llm_w rows followed by image_newline, then image_end (image_processor.image_token_types). The trunk embed copies
// these rows into the 4 mHC streams in place of the image token's embedding row.
//
// Numerics follow the official bf16 run: bf16 activations between ops, every GEMM bf16 x bf16 with fp32 accumulation
// (cuBLAS), bias / residual / activation epilogues in fp32 rounded once to bf16, RMSNorm and rotary in fp32, attention
// scores and softmax in fp32 with bf16 probabilities. The golden test (tests/ds41_vision_test.cu) compares the span
// against the official code's fp32 run.
//
// Weights: the release's dense/exlr8-extra-vision.safetensors (bf16, names as in the official checkpoint), read whole
// into one device allocation at boot (0.98 GB). The workspace is sized for the largest image (1024 tokens) and
// allocated on the first image (ensure_workspace), so a boot that never sees an image keeps it.
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"
#include "ds41_vision_plan.h"
#include "ds41_vision_id.h"

namespace ds41v {

constexpr int VD = 1024;          // [cfg] vision_config.hidden_size
constexpr int VL = 32;            // [cfg] vision_config.num_hidden_layers
constexpr int VH = 16;            // [cfg] vision_config.num_attention_heads
constexpr int VHD = VD / VH;      // 64
constexpr int VI = 2816;          // [cfg] vision_config.intermediate_size
constexpr int RD = VHD / 2;       // rotary dims per axis pair: vision.py rope_dim = dim // heads // 2 = 32
constexpr int PK = 3 * PATCH * PATCH;   // 588, the patch embed's input
constexpr int AIN = VD * DOWN * DOWN;   // 9216, the aligner's input
constexpr int OD = 5120;          // [cfg] text hidden_size
constexpr float ROPE_THETA = 10000.f;   // [cfg] vision_config.rope_theta
constexpr float NORM_EPS = 1e-6f;       // vision.py RMSNorm default
constexpr int RCH = 2048;         // rows per GEMM pass: the fp32 accumulator holds RCH x (2 * VI)
constexpr int QT = 256;           // attention query rows per pass: scores are VH x QT x n fp32

#define DS41V_CUDA(x) do { cudaError_t ds41v_ce_ = (x); if (ds41v_ce_ != cudaSuccess) throw std::runtime_error(std::string("ds41 vision: ") + #x + ": " + cudaGetErrorString(ds41v_ce_)); } while (0)
#define DS41V_BLAS(x) do { cublasStatus_t ds41v_st_ = (x); if (ds41v_st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("ds41 vision: ") + #x + " failed: " + std::to_string((int)ds41v_st_)); } while (0)

typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------------------ kernels

// pixels -> patches: patch (ph, pw) row-major, inside it (c, py, px); x = bf16(((u / 255) - 0.5) / 0.5), the float32 ops of
// image_processor.load_image. grid(n), block 588.
__global__ void k_patchify(const uint8_t* __restrict__ px, int vit_w, bf16* __restrict__ out)
{
    const int n = blockIdx.x, i = threadIdx.x;
    if (i >= PK) return;
    const int ph = n / vit_w, pw = n % vit_w;
    const int c = i / (PATCH * PATCH), py = (i / PATCH) % PATCH, pxx = i % PATCH;
    const size_t W = (size_t)vit_w * PATCH;
    const float u = (float)px[(((size_t)ph * PATCH + py) * W + (size_t)pw * PATCH + pxx) * 3 + c];
    const float x = __fdiv_rn(__fsub_rn(__fdiv_rn(u, 255.f), 0.5f), 0.5f);
    out[(size_t)n * PK + i] = __float2bfloat16_rn(x);
}

// RMSNorm over VD: y = bf16(w * (x * rsqrt(mean(x^2) + eps))). grid(rows), block 256.
__global__ void k_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ y)
{
    const int r = blockIdx.x;
    const bf16* xr = x + (size_t)r * VD;
    float v[VD / 256], ss = 0.f;
    for (int j = 0; j < VD / 256; ++j) { v[j] = __bfloat162float(xr[threadIdx.x + 256 * j]); ss += v[j] * v[j]; }
    __shared__ float red[8];
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 8; ++k) t += red[k];
    const float rs = rsqrtf(t / (float)VD + NORM_EPS);
    for (int j = 0; j < VD / 256; ++j) {
        const int h = threadIdx.x + 256 * j;
        y[(size_t)r * VD + h] = __float2bfloat16_rn(__bfloat162float(w[h]) * (v[j] * rs));
    }
}

enum Epi { E_BIAS = 0, E_BIAS_RES = 1, E_RES = 2, E_SWIGLU = 3, E_BIAS_GELU = 4 };

// The GEMM epilogue on an fp32 [rows][N] block. E_BIAS: out = bf16(acc + b). E_BIAS_RES / E_RES: o = bf16(acc (+ b)),
// out = bf16(out + o) (the residual add of two bf16 tensors). E_SWIGLU (N = 2 I, out [rows][I]): g = bf16(acc[:I]),
// u = bf16(acc[I:]), out = bf16(bf16(silu(g)) * u). E_BIAS_GELU: h = bf16(acc + b), out = bf16(gelu_erf(h)).
__global__ void k_epi(const float* __restrict__ acc, const bf16* __restrict__ b, bf16* __restrict__ out, int rows, int N, int mode)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int on = mode == E_SWIGLU ? N / 2 : N;
    if (i >= (size_t)rows * on) return;
    const int r = (int)(i / on), c = (int)(i % on);
    if (mode == E_SWIGLU) {
        const float g = __bfloat162float(__float2bfloat16_rn(acc[(size_t)r * N + c]));
        const float u = __bfloat162float(__float2bfloat16_rn(acc[(size_t)r * N + on + c]));
        const float s = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
        out[i] = __float2bfloat16_rn(s * u);
        return;
    }
    float v = acc[i];
    if (b) v += __bfloat162float(b[c]);
    if (mode == E_BIAS) { out[i] = __float2bfloat16_rn(v); return; }
    if (mode == E_BIAS_GELU) {
        const float h = __bfloat162float(__float2bfloat16_rn(v));
        out[i] = __float2bfloat16_rn(0.5f * h * (1.f + erff(h * 0.70710678118654752f)));
        return;
    }
    const float o = __bfloat162float(__float2bfloat16_rn(v));
    out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) + o);
}

// 2-D rotary on q and k inside qkv [n][3 VD] in place (vision.py get_vision_cos_sin / apply_rotary): for patch (ph, pw),
// angle j < 16 is ph * f_j, 16 <= j < 32 is pw * f_(j-16), f_i = 1 / theta^(2i / 32); x1 = x[:32], x2 = x[32:],
// out = [x1 cos - x2 sin, x2 cos + x1 sin]. grid(n), block (32, 2 * VH): y = q heads 0..15, k heads 16..31.
__global__ void k_rope(bf16* __restrict__ qkv, int vit_w)
{
    const int n = blockIdx.x, j = threadIdx.x, hh = threadIdx.y;
    const int ph = n / vit_w, pw = n % vit_w;
    const int fi = j % (RD / 2);
    const float inv = 1.f / powf(ROPE_THETA, (float)(2 * fi) / (float)RD);
    const float ang = (float)(j < RD / 2 ? ph : pw) * inv;
    float s, c;
    sincosf(ang, &s, &c);
    bf16* x = qkv + (size_t)n * 3 * VD + (size_t)(hh < VH ? hh : VD / VHD + (hh - VH)) * VHD;
    const float x1 = __bfloat162float(x[j]), x2 = __bfloat162float(x[j + RD]);
    x[j] = __float2bfloat16_rn(x1 * c - x2 * s);
    x[j + RD] = __float2bfloat16_rn(x2 * c + x1 * s);
}

// softmax rows of fp32 scores (scale 1/8 = 1/sqrt(64), SDPA's default) into bf16 probabilities. grid(rows), block 256.
__global__ void k_softmax(const float* __restrict__ s, bf16* __restrict__ p, int n)
{
    const float* sr = s + (size_t)blockIdx.x * n;
    bf16* pr = p + (size_t)blockIdx.x * n;
    __shared__ float red[8];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, sr[i]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (int k = 1; k < 8; ++k) m = fmaxf(m, red[k]);
    __syncthreads();
    const float sc = 0.125f;
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) z += expf((sr[i] - m) * sc);
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = z;
    __syncthreads();
    z = 0.f;
    for (int k = 0; k < 8; ++k) z += red[k];
    const float iz = 1.f / z;
    for (int i = threadIdx.x; i < n; i += 256) pr[i] = __float2bfloat16_rn(expf((sr[i] - m) * sc) * iz);
}

// The aligner's 3 x 3 unfold (vision.py Aligner: view (h, w, C), pad right / bottom with zeros to multiples of 3,
// F.unfold(3, stride 3)): row t = (bh, bw) of the llm grid, column c * 9 + ki * 3 + kj = channel c of patch
// (3 bh + ki, 3 bw + kj). grid(L), block 256.
__global__ void k_unfold(const bf16* __restrict__ y, int vit_h, int vit_w, int llm_w, bf16* __restrict__ u)
{
    const int t = blockIdx.x, bh = t / llm_w, bw = t % llm_w;
    for (int col = threadIdx.x; col < AIN; col += blockDim.x) {
        const int c = col / 9, k = col % 9, r = 3 * bh + k / 3, q = 3 * bw + k % 3;
        u[(size_t)t * AIN + col] = (r < vit_h && q < vit_w) ? y[((size_t)r * vit_w + q) * VD + c] : __float2bfloat16_rn(0.f);
    }
}

// The span rows: [start] + ([aligner row] * llm_w + [newline]) * llm_h + [end]. grid(span), block 256.
__global__ void k_span(const bf16* __restrict__ al, const bf16* __restrict__ st, const bf16* __restrict__ nl, const bf16* __restrict__ en,
                       int llm_h, int llm_w, bf16* __restrict__ out)
{
    const int s = blockIdx.x, last = llm_h * (llm_w + 1) + 1;
    const bf16* src;
    if (s == 0) src = st;
    else if (s == last) src = en;
    else {
        const int k = s - 1, r = k / (llm_w + 1), c = k % (llm_w + 1);
        src = c == llm_w ? nl : al + ((size_t)r * llm_w + c) * OD;
    }
    for (int h = threadIdx.x; h < OD; h += blockDim.x) out[(size_t)s * OD + h] = src[h];
}

// ------------------------------------------------------------------------------------------------------------ weights

struct StTensor { std::string dtype; std::vector<int64_t> shape; size_t begin = 0, end = 0; };

// The safetensors header of `path`: tensor name -> dtype, shape, byte range inside the data section; data_off = where
// the data section starts in the file.
inline std::map<std::string, StTensor> st_header(const std::string& path, size_t& data_off, size_t& file_bytes)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ds41 vision: cannot open " + path);
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1 || n > (64u << 20)) { std::fclose(f); throw std::runtime_error("ds41 vision: bad safetensors header in " + path); }
    std::string h(n, '\0');
    if (std::fread(&h[0], 1, n, f) != n) { std::fclose(f); throw std::runtime_error("ds41 vision: short header in " + path); }
    std::fseek(f, 0, SEEK_END);
    file_bytes = (size_t)std::ftell(f);
    std::fclose(f);
    data_off = 8 + n;
    const g6v::Json j = g6v::Json::parse(h);
    std::map<std::string, StTensor> out;
    for (const auto& kv : j.o) {
        if (kv.first == "__metadata__") continue;
        StTensor t;
        t.dtype = kv.second.at("dtype").str();
        for (const auto& d : kv.second.at("shape").a) t.shape.push_back(d.i64());
        t.begin = (size_t)kv.second.at("data_offsets").at(0).i64();
        t.end = (size_t)kv.second.at("data_offsets").at(1).i64();
        if (t.end < t.begin || data_off + t.end > file_bytes) throw std::runtime_error("ds41 vision: tensor " + kv.first + " past the file end");
        out[kv.first] = t;
    }
    return out;
}

struct Block { const bf16 *n1, *wqkv, *bqkv, *wo, *bo, *n2, *w1, *w2; };

struct Weights {
    void* blob = nullptr;     // the file's data section, whole
    size_t bytes = 0;
    const bf16 *patch_w = nullptr, *patch_b = nullptr, *norm = nullptr;
    Block blk[VL];
    const bf16 *a_w1 = nullptr, *a_b1 = nullptr, *a_w2 = nullptr, *a_b2 = nullptr;
    const bf16 *img_start = nullptr, *img_end = nullptr, *img_nl = nullptr;
};

// Reads the vision file into device memory and checks every tensor's dtype and shape against the official module.
inline Weights load_weights(const std::string& path)
{
    size_t off = 0, fbytes = 0;
    const auto hdr = st_header(path, off, fbytes);
    Weights w;
    w.bytes = fbytes - off;
    DS41V_CUDA(cudaMalloc(&w.blob, w.bytes));
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("ds41 vision: cannot open " + path);
        std::fseek(f, (long)off, SEEK_SET);
        const size_t CH = 64u << 20;
        void* host = nullptr;
        DS41V_CUDA(cudaMallocHost(&host, CH));
        size_t done = 0;
        while (done < w.bytes) {
            const size_t k = std::min(CH, w.bytes - done);
            if (std::fread(host, 1, k, f) != k) { std::fclose(f); cudaFreeHost(host); throw std::runtime_error("ds41 vision: short read of " + path); }
            DS41V_CUDA(cudaMemcpy((char*)w.blob + done, host, k, cudaMemcpyHostToDevice));
            done += k;
        }
        std::fclose(f);
        cudaFreeHost(host);
    }
    auto get = [&](const std::string& name, std::vector<int64_t> shape) -> const bf16* {
        auto it = hdr.find(name);
        if (it == hdr.end()) throw std::runtime_error("ds41 vision: " + path + " has no tensor " + name);
        if (it->second.dtype != "BF16" || it->second.shape != shape) throw std::runtime_error("ds41 vision: tensor " + name + " has the wrong dtype or shape");
        return (const bf16*)((const char*)w.blob + it->second.begin);
    };
    w.patch_w = get("vision.patch_embed.proj.weight", {VD, PK});
    w.patch_b = get("vision.patch_embed.proj.bias", {VD});
    for (int l = 0; l < VL; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        Block& b = w.blk[l];
        b.n1 = get(p + "norm1.weight", {VD});
        b.wqkv = get(p + "attn.wqkv.weight", {3 * VD, VD});
        b.bqkv = get(p + "attn.wqkv.bias", {3 * VD});
        b.wo = get(p + "attn.wo.weight", {VD, VD});
        b.bo = get(p + "attn.wo.bias", {VD});
        b.n2 = get(p + "norm2.weight", {VD});
        b.w1 = get(p + "mlp.w1.weight", {2 * VI, VD});
        b.w2 = get(p + "mlp.w2.weight", {VD, VI});
    }
    w.norm = get("vision.norm.weight", {VD});
    w.a_w1 = get("aligner.w1.weight", {OD, AIN});
    w.a_b1 = get("aligner.w1.bias", {OD});
    w.a_w2 = get("aligner.w2.weight", {OD, OD});
    w.a_b2 = get("aligner.w2.bias", {OD});
    w.img_start = get("image_start", {OD});
    w.img_end = get("image_end", {OD});
    w.img_nl = get("image_newline", {OD});
    if (hdr.count("vision.blocks." + std::to_string(VL) + ".norm1.weight")) throw std::runtime_error("ds41 vision: more than 32 vision blocks");
    return w;
}

// ------------------------------------------------------------------------------------------------------------ the tower

class Tower {
public:
    Weights w;
    bool loaded() const { return w.blob != nullptr; }

    void load(const std::string& path) { w = load_weights(path); }

    // The workspace of the largest image (max_patches rows); on the first image, or at boot by a caller that wants it
    // counted before the KV pool is sized.
    void ensure_workspace()
    {
        if (ws_) return;
        DS41V_BLAS(cublasCreate(&blas_));
        const size_t n = (size_t)MAX_PATCHES;
        size_t off = 0;
        auto take = [&](size_t bytes) { const size_t o = off; off += (bytes + 255) & ~size_t(255); return o; };
        const size_t o_pix = take(n * PATCH * PATCH * 3), o_pat = take(n * PK * 2), o_x = take(n * VD * 2), o_h = take(n * VD * 2),
                     o_qkv = take(n * 3 * VD * 2), o_o = take(n * VD * 2), o_m = take(n * VI * 2),
                     o_acc = take((size_t)RCH * 2 * VI * 4), o_s = take((size_t)VH * QT * n * 4), o_p = take((size_t)VH * QT * n * 2),
                     o_u = take((size_t)MAX_LLM * AIN * 2), o_a1 = take((size_t)MAX_LLM * OD * 2), o_a2 = take((size_t)MAX_LLM * OD * 2);
        DS41V_CUDA(cudaMalloc(&ws_, off));
        ws_bytes_ = off;
        char* b = (char*)ws_;
        pix_ = (uint8_t*)(b + o_pix); pat_ = (bf16*)(b + o_pat); x_ = (bf16*)(b + o_x); h_ = (bf16*)(b + o_h);
        qkv_ = (bf16*)(b + o_qkv); o_ = (bf16*)(b + o_o); m_ = (bf16*)(b + o_m); acc_ = (float*)(b + o_acc);
        s_ = (float*)(b + o_s); p_ = (bf16*)(b + o_p); u_ = (bf16*)(b + o_u); a1_ = (bf16*)(b + o_a1); a2_ = (bf16*)(b + o_a2);
    }
    size_t workspace_bytes() const { return ws_bytes_; }

    // Runs the tower on host pixels (uint8 [vit_h * 14][vit_w * 14][3]) and writes the span (bf16 [span_tokens][5120]) to
    // the device pointer `span`. Synchronous on `st` when it returns (the pixels are copied first).
    void encode(const uint8_t* host_px, int vit_h, int vit_w, bf16* span, cudaStream_t st)
    {
        const Grid g = grid_of(vit_h, vit_w);
        if (!g.ok) throw std::runtime_error("ds41 vision: patch grid " + std::to_string(vit_h) + "x" + std::to_string(vit_w) + " is not a valid image plan");
        ensure_workspace();
        DS41V_BLAS(cublasSetStream(blas_, st));
        const int n = vit_h * vit_w;
        DS41V_CUDA(cudaMemcpyAsync(pix_, host_px, (size_t)n * PATCH * PATCH * 3, cudaMemcpyHostToDevice, st));
        k_patchify<<<n, 608, 0, st>>>(pix_, vit_w, pat_);
        linear(pat_, PK, w.patch_w, w.patch_b, x_, n, VD, PK, E_BIAS, st);   // x = patch_embed(patches)
        for (int l = 0; l < VL; ++l) {
            const Block& b = w.blk[l];
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n1, h_);
            linear(h_, VD, b.wqkv, b.bqkv, qkv_, n, 3 * VD, VD, E_BIAS, st);
            k_rope<<<n, dim3(RD, 2 * VH), 0, st>>>(qkv_, vit_w);
            attention(n, st);
            linear(o_, VD, b.wo, b.bo, x_, n, VD, VD, E_BIAS_RES, st);       // x = x + wo(o)
            k_rmsnorm<<<n, 256, 0, st>>>(x_, b.n2, h_);
            linear(h_, VD, b.w1, nullptr, m_, n, 2 * VI, VD, E_SWIGLU, st);  // silu(gate) * up
            linear(m_, VI, b.w2, nullptr, x_, n, VD, VI, E_RES, st);         // x = x + w2(.)
        }
        k_rmsnorm<<<n, 256, 0, st>>>(x_, w.norm, h_);
        const int L = g.llm_h * g.llm_w;
        k_unfold<<<L, 256, 0, st>>>(h_, vit_h, vit_w, g.llm_w, u_);
        linear(u_, AIN, w.a_w1, w.a_b1, a1_, L, OD, AIN, E_BIAS_GELU, st);
        linear(a1_, OD, w.a_w2, w.a_b2, a2_, L, OD, OD, E_BIAS, st);
        k_span<<<g.span(), 256, 0, st>>>(a2_, w.img_start, w.img_nl, w.img_end, g.llm_h, g.llm_w, span);
        DS41V_CUDA(cudaGetLastError());
        DS41V_CUDA(cudaStreamSynchronize(st));
    }

    // the aligner rows (bf16 [llm_h * llm_w][5120]) of the last encode, for the golden test
    const bf16* aligner_rows() const { return a2_; }

    ~Tower()
    {
        if (blas_) cublasDestroy(blas_);
        if (ws_) cudaFree(ws_);
        if (w.blob) cudaFree(w.blob);
    }

private:
    cublasHandle_t blas_ = nullptr;
    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    uint8_t* pix_ = nullptr;
    bf16 *pat_ = nullptr, *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *o_ = nullptr, *m_ = nullptr, *p_ = nullptr, *u_ = nullptr, *a1_ = nullptr, *a2_ = nullptr;
    float *acc_ = nullptr, *s_ = nullptr;

    // y [M][N] (epilogue) = x [M][K] (row stride ldx) * W [N][K]^T, in passes of RCH rows through the fp32 accumulator
    void linear(const bf16* x, int ldx, const bf16* W, const bf16* bias, bf16* y, int M, int N, int K, int mode, cudaStream_t st)
    {
        const float one = 1.f, zero = 0.f;
        const int on = mode == E_SWIGLU ? N / 2 : N;
        for (int r0 = 0; r0 < M; r0 += RCH) {
            const int rows = std::min(RCH, M - r0);
            DS41V_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, K, x + (size_t)r0 * ldx, CUDA_R_16BF, ldx,
                                    &zero, acc_, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            const size_t tot = (size_t)rows * on;
            k_epi<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(acc_, bias, y + (size_t)r0 * on, rows, N, mode);
        }
    }

    // o [n][VD] = softmax(q k^T / 8) v per head, in passes of QT query rows; q, k, v are the three VD-wide parts of qkv_
    void attention(int n, cudaStream_t st)
    {
        const float one = 1.f, zero = 0.f;
        const int ld = 3 * VD;
        for (int q0 = 0; q0 < n; q0 += QT) {
            const int tq = std::min(QT, n - q0);
            // S_h [tq][n] = Q_h [tq][64] K_h [n][64]^T, batched over the 16 heads (column-major: S^T = K Q^T)
            DS41V_BLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n, tq, VHD, &one,
                                                  qkv_ + VD, CUDA_R_16BF, ld, VHD,
                                                  qkv_ + (size_t)q0 * ld, CUDA_R_16BF, ld, VHD,
                                                  &zero, s_, CUDA_R_32F, n, (long long)tq * n, VH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            k_softmax<<<VH * tq, 256, 0, st>>>(s_, p_, n);
            // O_h [tq][64] = P_h [tq][n] V_h [n][64] into o_ rows q0.., columns h * 64 (column-major: O^T = V^T P^T)
            DS41V_BLAS(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, VHD, tq, n, &one,
                                                  qkv_ + 2 * VD, CUDA_R_16BF, ld, VHD,
                                                  p_, CUDA_R_16BF, n, (long long)tq * n,
                                                  &zero, o_ + (size_t)q0 * VD, CUDA_R_16BF, VD, VHD, VH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }
};

}  // namespace ds41v
