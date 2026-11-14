// Inkling-2-Small vision tower (the checkpoint's model.visual.* tensors, planned as V.*), for image input. Every TP rank
// holds the whole tower and runs it on the same pixels, so the ranks exchange nothing for its rows.
//
// In: the padded RGB the front end sends (uint8 [vit_h * 20][vit_w * 20][3]). Out: BF16 [vit_h * vit_w / 4][H], one row
// per image token, tokens row by row. The step's embed (kernels/ink2_norm_res.cuh k_ink2_embed_norm) copies such a row
// in place of the image token's embedding.
//
//   pixel   bf16((u / 255 - 0.5) / 0.5); a pixel past the content counts as u = -1
//   x       patch n: bf16(bf16(patch_embedding . pixels) + position embedding);  class row: bf16(class + cls position)
//   h       ln_pre(x)
//   layer   q, k, v = bf16(W ln_1(h) + b); rotary on q and k (class row untouched, a patch at (row + 1, col + 1));
//           a = bf16(softmax(q . k / 8) v) over every row;  h = bf16(h + bf16(ls_1 * bf16(W_o a + b_o)))
//           h = bf16(h + bf16(ls_2 * bf16(W_proj gelu(bf16(W_fc ln_2(h) + b_fc)) + b_proj)))
//   merger  the patch rows, 4 at a time as one row [4 VD]:  out = bf16(proj_out . gelu(bf16(proj_in . row)))
//
// Numerics: BF16 activations between the operators, every GEMM BF16 x BF16 with fp32 accumulation (cuBLAS), the
// epilogues, the norms and the rotary in fp32 rounded once to BF16. The attention keeps its scores and the online softmax
// in fp32 and rounds the probabilities to BF16 for the product with v; the CPU reference (engine/ref/ink2) keeps them in
// fp32, so tests/ink2_vision_test.cu compares within a tolerance.
//
// The dimensions come from the checkpoint's config.json, so a miniature tower runs through the same kernels. The
// kernels need a head of 64 cells and a width that is a multiple of 256, at most 2048.
//
// The second half of the file is the staging of a prefill launch (Ink2Vision::stage): the images whose spans overlap
// the launch's rows run through the tower, and their rows are copied to a pool the embed reads. The tower's last output
// is kept, so an image that straddles two launches runs once.
#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_input.h"
#include "ink2_vision_plan.h"

namespace ink2rt
{

#define INK2V_CUDA(x) do { cudaError_t ink2v_ce_ = (x); if (ink2v_ce_ != cudaSuccess) throw std::runtime_error(std::string("ink2 vision: ") + #x + ": " + cudaGetErrorString(ink2v_ce_)); } while (0)
#define INK2V_BLAS(x) do { cublasStatus_t ink2v_st_ = (x); if (ink2v_st_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("ink2 vision: ") + #x + " failed: " + std::to_string((int)ink2v_st_)); } while (0)

namespace vis
{
typedef __nv_bfloat16 bf16;

constexpr int HD = 64;                  // cells of a head (the attention and the rotary are written for it)
constexpr int RF = HD / 4;              // 16 rotary frequencies per axis
constexpr int MAX_VD = 2048;            // the norm keeps a row in registers: VD / 256 cells a thread
constexpr int PK = ink2v::PATCH * ink2v::PATCH * 3;   // 1200, the patch embedding's input (y, x, channel)
constexpr size_t ACC_FLOATS = static_cast<size_t>(8) << 20;   // 32 MiB, the fp32 accumulator of one GEMM pass

// ------------------------------------------------------------------------------------------------------------ kernels

// Patch n of a vit_w wide grid: the 4 patches of an image token are consecutive, tokens row by row, the patches of a
// token row by row.
__device__ __forceinline__ void patch_rc(int n, int vit_w, int& row, int& col)
{
    const int blk = n >> 2, in = n & 3, cols = vit_w / ink2v::MERGE;
    row = (blk / cols) * ink2v::MERGE + (in >> 1);
    col = (blk % cols) * ink2v::MERGE + (in & 1);
}

// pixels -> patches [n][PK]: cell (py * 20 + px) * 3 + channel. grid(n), block 256.
__global__ void k_ink2v_patchify(const uint8_t* __restrict__ px, int vit_w, int content_h, int content_w, bf16* __restrict__ out)
{
    const int n = blockIdx.x, P = ink2v::PATCH;
    int row, col;
    patch_rc(n, vit_w, row, col);
    const size_t W = static_cast<size_t>(vit_w) * P;
    for (int i = threadIdx.x; i < PK; i += blockDim.x)
    {
        const int ch = i % 3, pxx = (i / 3) % P, py = i / (3 * P);
        const int gy = row * P + py, gx = col * P + pxx;
        const float u = gy < content_h && gx < content_w ? static_cast<float>(px[(static_cast<size_t>(gy) * W + gx) * 3 + ch]) : -1.f;
        out[static_cast<size_t>(n) * PK + i] = __float2bfloat16_rn(__fdiv_rn(__fsub_rn(__fmul_rn(u, 1.f / 255.f), 0.5f), 0.5f));
    }
}

// One side of the position table read at patch i of n: half-pixel centres, clamped at the border.
__device__ __forceinline__ void pos_lerp(int i, int n, int grid, int& i0, int& i1, float& w1)
{
    const float s = __fsub_rn(__fdiv_rn(__fmul_rn(__fadd_rn(static_cast<float>(i), 0.5f), static_cast<float>(grid)), static_cast<float>(n)), 0.5f);
    const float f = floorf(s);
    const int a = static_cast<int>(f);
    i0 = min(max(a, 0), grid - 1);
    i1 = min(max(a + 1, 0), grid - 1);
    w1 = __fsub_rn(s, f);
}

// x [1 + n][VD] in place: row 0 = bf16(class + cls position); row 1 + p = bf16(x + bf16(position of patch p)), the
// position read from the table [grid * grid][VD] with bilinear weights. grid(1 + n), block 256.
__global__ void k_ink2v_pos_add(bf16* __restrict__ x, const bf16* __restrict__ cls, const bf16* __restrict__ cls_pos, const bf16* __restrict__ table,
                                int vit_h, int vit_w, int grid, int VD)
{
    const int s = blockIdx.x;
    bf16* xr = x + static_cast<size_t>(s) * VD;
    if (s == 0)
    {
        for (int d = threadIdx.x; d < VD; d += blockDim.x) xr[d] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(cls[d]), __bfloat162float(cls_pos[d])));
        return;
    }
    int row, col, y0, y1, x0, x1;
    float wy1, wx1;
    patch_rc(s - 1, vit_w, row, col);
    pos_lerp(row, vit_h, grid, y0, y1, wy1);
    pos_lerp(col, vit_w, grid, x0, x1, wx1);
    const float wy0 = __fsub_rn(1.f, wy1), wx0 = __fsub_rn(1.f, wx1);
    const float w00 = __fmul_rn(wy0, wx0), w01 = __fmul_rn(wy0, wx1), w10 = __fmul_rn(wy1, wx0), w11 = __fmul_rn(wy1, wx1);
    const bf16 *a = table + static_cast<size_t>(y0 * grid + x0) * VD, *b = table + static_cast<size_t>(y0 * grid + x1) * VD;
    const bf16 *c = table + static_cast<size_t>(y1 * grid + x0) * VD, *e = table + static_cast<size_t>(y1 * grid + x1) * VD;
    for (int d = threadIdx.x; d < VD; d += blockDim.x)
    {
        float p = __fmul_rn(w00, __bfloat162float(a[d]));
        p = __fadd_rn(p, __fmul_rn(w01, __bfloat162float(b[d])));
        p = __fadd_rn(p, __fmul_rn(w10, __bfloat162float(c[d])));
        p = __fadd_rn(p, __fmul_rn(w11, __bfloat162float(e[d])));
        p = __bfloat162float(__float2bfloat16_rn(p));
        xr[d] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(xr[d]), p));
    }
}

__device__ __forceinline__ float sum256(float v, float* red)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 8; ++k) t += red[k];
    __syncthreads();
    return t;
}

// LayerNorm over VD: y = bf16((x - mean) * rsqrt(var + eps) * w + b), the variance over VD. grid(rows), block 256.
__global__ void __launch_bounds__(256) k_ink2v_layernorm(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                                         bf16* __restrict__ y, int VD, float eps)
{
    __shared__ float red[8];
    const int r = blockIdx.x, per = VD >> 8;
    const bf16* xr = x + static_cast<size_t>(r) * VD;
    float v[MAX_VD / 256], s = 0.f;
    for (int j = 0; j < MAX_VD / 256; ++j)
    {
        v[j] = j < per ? __bfloat162float(xr[threadIdx.x + 256 * j]) : 0.f;
        s += v[j];
    }
    const float mean = sum256(s, red) / static_cast<float>(VD);
    float ss = 0.f;
    for (int j = 0; j < MAX_VD / 256; ++j)
    {
        const float d = j < per ? __fsub_rn(v[j], mean) : 0.f;
        ss = __fadd_rn(ss, __fmul_rn(d, d));
    }
    const float rs = rsqrtf(__fadd_rn(sum256(ss, red) / static_cast<float>(VD), eps));
    for (int j = 0; j < per; ++j)
    {
        const int k = threadIdx.x + 256 * j;
        const float t = __fmul_rn(__fmul_rn(__fsub_rn(v[j], mean), rs), __bfloat162float(w[k]));
        y[static_cast<size_t>(r) * VD + k] = __float2bfloat16_rn(__fadd_rn(t, __bfloat162float(b[k])));
    }
}

enum Epi { E_BIAS = 0, E_LS_RES = 1, E_GELU = 2 };

// The epilogue of a GEMM on its fp32 [rows][N] block; out has rows of ldo cells; b [N] or null.
//   E_BIAS    out = bf16(acc + b)
//   E_LS_RES  o = bf16(acc + b), out = bf16(out + bf16(ls * o))     (the scaled residual branch)
//   E_GELU    out = bf16(gelu(bf16(acc + b))), the exact GELU
__global__ void k_ink2v_epi(const float* __restrict__ acc, const bf16* __restrict__ b, const bf16* __restrict__ ls, bf16* __restrict__ out, int rows,
                            int N, int ldo, int mode)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * N) return;
    const int r = static_cast<int>(i / N), c = static_cast<int>(i % N);
    bf16* o = out + static_cast<size_t>(r) * ldo + c;
    float v = acc[i];
    if (b) v = __fadd_rn(v, __bfloat162float(b[c]));
    v = __bfloat162float(__float2bfloat16_rn(v));
    if (mode == E_BIAS) *o = __float2bfloat16_rn(v);
    else if (mode == E_LS_RES)
    {
        const float sres = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(ls[c]), v)));
        *o = __float2bfloat16_rn(__fadd_rn(__bfloat162float(*o), sres));
    }
    else *o = __float2bfloat16_rn(__fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.f, erff(__fmul_rn(v, 0.7071067811865475244f)))));
}

// The 2-D rotary on q and k, in place in qkv [S][3 VD]: pair p = cells 2p and 2p + 1 of a head; the first 16 pairs
// turn with the column, the next 16 with the row, frequency m of an axis = 1 / theta^(2m / 32). Row 0 is the class
// row (angle 0, untouched); row 1 + n is patch n at (row + 1, col + 1). grid(S), block (32, 2 * heads): y = the heads
// of q, then the heads of k.
__global__ void k_ink2v_rope(bf16* __restrict__ qkv, int vit_w, int VD, int heads, float theta)
{
    const int s = blockIdx.x, p = threadIdx.x, hh = threadIdx.y;
    if (s == 0) return;
    int row, col;
    patch_rc(s - 1, vit_w, row, col);
    bf16* x = qkv + static_cast<size_t>(s) * 3 * VD + (hh < heads ? hh * HD : VD + (hh - heads) * HD) + 2 * p;
    const int m = p < RF ? p : p - RF;
    const float inv = 1.f / powf(theta, static_cast<float>(2 * m) / static_cast<float>(HD / 2));
    const float ang = __fmul_rn(static_cast<float>(p < RF ? col + 1 : row + 1), inv);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    const float a = __bfloat162float(x[0]), b = __bfloat162float(x[1]);
    x[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(a, cs), __fmul_rn(b, sn)));
    x[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, cs), __fmul_rn(a, sn)));
}

// Attention over every row of one image (no mask), per head: o = softmax(q k^T / 8) v, with the online softmax (a
// running row maximum and sum) over key tiles of AT_T rows, so no score matrix reaches device memory. Tensor-core
// mma.m16n8k16 (BF16 in, fp32 accumulate) with the accumulators in registers: warp w owns query rows 16w .. 16w + 15
// of the block's AT_T. The probabilities are rounded to BF16 for the product with v, and the row sum adds the rounded
// values, so o is normalised by what the product used. grid(ceil(n / AT_T), heads), block 128.
// In the m16n8k16 layout lane = 4 g + t holds rows g and g + 8, columns 2t and 2t + 1 of each 16 x 8 accumulator.
constexpr int AT_T = 64;                // query rows of a block, key rows of a tile
constexpr int AT_LD = HD + 8;           // row stride of the K and transposed-V tiles in shared memory

__device__ __forceinline__ void at_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t at_pack(float lo, float hi)
{
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__global__ void __launch_bounds__(128) k_ink2v_attn(const bf16* __restrict__ qkv, bf16* __restrict__ o, int n, int VD)
{
    __shared__ __align__(16) bf16 sK[AT_T * AT_LD];    // [key][cell]
    __shared__ __align__(16) bf16 sV[AT_T * AT_LD];    // [cell][key]: V transposed, so a B fragment is one 32-bit load
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y;
    const size_t ld = static_cast<size_t>(3) * VD;
    const int qa = blockIdx.x * AT_T + warp * 16 + g, qb = qa + 8;   // this lane's two query rows
    // q as A fragments over the head's 4 steps of 16 cells (zero past row n)
    uint32_t qf[4][4];
    {
        const bf16* qpa = qkv + static_cast<size_t>(qa) * ld + h * HD;
        const bf16* qpb = qkv + static_cast<size_t>(qb) * ld + h * HD;
        for (int kk = 0; kk < 4; ++kk)
        {
            const int c = kk * 16 + 2 * t;
            qf[kk][0] = qa < n ? *reinterpret_cast<const uint32_t*>(qpa + c) : 0u;
            qf[kk][1] = qb < n ? *reinterpret_cast<const uint32_t*>(qpb + c) : 0u;
            qf[kk][2] = qa < n ? *reinterpret_cast<const uint32_t*>(qpa + c + 8) : 0u;
            qf[kk][3] = qb < n ? *reinterpret_cast<const uint32_t*>(qpb + c + 8) : 0u;
        }
    }
    float oc[8][4];
    for (int j = 0; j < 8; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;   // the running maximum (exp2 domain) and this lane's part of the sum
    const float sc = 0.125f * 1.4426950408889634f;              // 1 / sqrt(64), in the exp2 domain
    for (int k0 = 0; k0 < n; k0 += AT_T)
    {
        __syncthreads();   // every warp is done with the last tile
        for (int v = tid; v < AT_T * HD / 8; v += 128)
        {
            const int r = v / (HD / 8), c = (v % (HD / 8)) * 8;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = make_uint4(0, 0, 0, 0);
            if (k0 + r < n)
            {
                const bf16* row = qkv + static_cast<size_t>(k0 + r) * ld + h * HD + c;
                kx = *reinterpret_cast<const uint4*>(row + VD);
                vx = *reinterpret_cast<const uint4*>(row + 2 * VD);
            }
            *reinterpret_cast<uint4*>(sK + r * AT_LD + c) = kx;
            const bf16* ve = reinterpret_cast<const bf16*>(&vx);
            for (int i = 0; i < 8; ++i) sV[(c + i) * AT_LD + r] = ve[i];
        }
        __syncthreads();
        // the scores [16][64] = q k^T: 8 blocks of 8 keys
        float sr[8][4];
        for (int j = 0; j < 8; ++j)
        {
            sr[j][0] = sr[j][1] = sr[j][2] = sr[j][3] = 0.f;
            const bf16* kr = sK + (j * 8 + g) * AT_LD + 2 * t;
            for (int kk = 0; kk < 4; ++kk)
                at_mma(sr[j], qf[kk], *reinterpret_cast<const uint32_t*>(kr + kk * 16), *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 8));
        }
        // the online softmax over this tile's valid keys
        const int valid = n - k0;
        float xa = -INFINITY, xb = -INFINITY;
        for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 2; ++e)
            {
                const bool ok = j * 8 + 2 * t + e < valid;
                sr[j][e] = ok ? sr[j][e] * sc : -INFINITY;
                sr[j][2 + e] = ok ? sr[j][2 + e] * sc : -INFINITY;
                xa = fmaxf(xa, sr[j][e]);
                xb = fmaxf(xb, sr[j][2 + e]);
            }
        for (int o2 = 1; o2 < 4; o2 <<= 1)
        {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
        }
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        const float aa = exp2f(ma - na), ab = exp2f(mb - nb);   // 0 on the first tile
        ma = na;
        mb = nb;
        la *= aa;
        lb *= ab;
        for (int j = 0; j < 8; ++j)
        {
            oc[j][0] *= aa;
            oc[j][1] *= aa;
            oc[j][2] *= ab;
            oc[j][3] *= ab;
        }
        // the probabilities as A fragments (keys = k), rounded to BF16; the sums add the rounded values
        uint32_t pf[4][4];
        for (int kk = 0; kk < 4; ++kk)
            for (int half = 0; half < 2; ++half)
            {
                const float* c = sr[2 * kk + half];
                const __nv_bfloat162 pa = __floats2bfloat162_rn(exp2f(c[0] - na), exp2f(c[1] - na));
                const __nv_bfloat162 pb = __floats2bfloat162_rn(exp2f(c[2] - nb), exp2f(c[3] - nb));
                la += __low2float(pa) + __high2float(pa);
                lb += __low2float(pb) + __high2float(pb);
                pf[kk][half * 2] = *reinterpret_cast<const uint32_t*>(&pa);
                pf[kk][half * 2 + 1] = *reinterpret_cast<const uint32_t*>(&pb);
            }
        // o [16][64] += p v: 8 blocks of 8 cells
        for (int j = 0; j < 8; ++j)
        {
            const bf16* vr = sV + (j * 8 + g) * AT_LD + 2 * t;
            for (int kk = 0; kk < 4; ++kk)
                at_mma(oc[j], pf[kk], *reinterpret_cast<const uint32_t*>(vr + kk * 16), *reinterpret_cast<const uint32_t*>(vr + kk * 16 + 8));
        }
    }
    for (int o2 = 1; o2 < 4; o2 <<= 1)
    {
        la += __shfl_xor_sync(0xffffffffu, la, o2);
        lb += __shfl_xor_sync(0xffffffffu, lb, o2);
    }
    const float ia = 1.f / la, ib = 1.f / lb;
    for (int j = 0; j < 8; ++j)
    {
        const int c = h * HD + j * 8 + 2 * t;
        if (qa < n) *reinterpret_cast<uint32_t*>(o + static_cast<size_t>(qa) * VD + c) = at_pack(oc[j][0] * ia, oc[j][1] * ia);
        if (qb < n) *reinterpret_cast<uint32_t*>(o + static_cast<size_t>(qb) * VD + c) = at_pack(oc[j][2] * ib, oc[j][3] * ib);
    }
}

}   // namespace vis

// ------------------------------------------------------------------------------------------------------------ the tower

// The tower's dimensions (config.json vision_config) and the trunk's hidden size.
struct VisDims
{
    int vd = 1024, layers = 23, heads = 16, mlp = 4096;
    int patch = 20, merge = 2, grid = 32, proj = 4096;
    int H = 0;
    float eps = 1e-5f, theta = 10000.f;
};

class Ink2Vision
{
public:
    typedef vis::bf16 bf16;
    // The device pointer of the BF16 tensor V.<name> with this shape; it refuses (throws or exits) any other tensor.
    typedef std::function<const uint16_t*(const std::string& name, const std::vector<int64_t>& shape)> Lookup;

    VisDims d;
    long encode_calls = 0;    // towers run (a test reads it)
    bool log = false;         // one line per tower run on stderr

    bool ready() const { return ws_ != nullptr; }
    size_t device_bytes() const { return ws_bytes_ + (static_cast<size_t>(pool_rows_) + ink2v::MAX_TOKENS) * d.H * sizeof(bf16); }

    // Binds the weights, and allocates the workspace of the largest image, the rows of one image (the stage) and a pool
    // of pool_rows rows (the rows of one launch).
    void init(const VisDims& dims, const Lookup& get, int pool_rows)
    {
        d = dims;
        if (d.heads < 1 || d.vd != d.heads * vis::HD) fail("the kernels need heads of 64 cells");
        if (d.vd % 256 || d.vd > vis::MAX_VD) fail("the kernels need a width that is a multiple of 256, at most 2048");
        if (d.patch != ink2v::PATCH || d.merge != ink2v::MERGE) fail("the patch or merge size is not the image processor's");
        if (d.layers < 1 || d.mlp < 1 || d.grid < 1 || d.proj < 1 || d.H < 1 || pool_rows < 1) fail("dimensions");
        const int64_t VD = d.vd, VI = d.mlp, PJ = d.proj;
        const auto g = [&](const std::string& n, std::vector<int64_t> shape) { return reinterpret_cast<const bf16*>(get(n, shape)); };
        const std::string e = "backbone.embeddings.", t = "backbone.transformer.";
        w_.cls = g(e + "class_embedding", {VD});
        w_.cls_pos = g(e + "cls_position_embedding", {VD});
        w_.patch = g(e + "patch_embedding.weight", {VD, vis::PK});
        w_.pos = g(e + "position_embedding.weight", {static_cast<int64_t>(d.grid) * d.grid, VD});
        w_.ln_pre_w = g(t + "ln_pre.weight", {VD});
        w_.ln_pre_b = g(t + "ln_pre.bias", {VD});
        w_.layers.resize(static_cast<size_t>(d.layers));
        for (int l = 0; l < d.layers; ++l)
        {
            const std::string p = t + "layers." + std::to_string(l) + ".";
            LayerW& y = w_.layers[static_cast<size_t>(l)];
            y.ln1_w = g(p + "ln_1.weight", {VD});
            y.ln1_b = g(p + "ln_1.bias", {VD});
            y.ln2_w = g(p + "ln_2.weight", {VD});
            y.ln2_b = g(p + "ln_2.bias", {VD});
            y.ls1 = g(p + "ls_1.gamma", {VD});
            y.ls2 = g(p + "ls_2.gamma", {VD});
            y.q_w = g(p + "self_attn.q_proj.weight", {VD, VD});
            y.q_b = g(p + "self_attn.q_proj.bias", {VD});
            y.k_w = g(p + "self_attn.k_proj.weight", {VD, VD});
            y.k_b = g(p + "self_attn.k_proj.bias", {VD});
            y.v_w = g(p + "self_attn.v_proj.weight", {VD, VD});
            y.v_b = g(p + "self_attn.v_proj.bias", {VD});
            y.o_w = g(p + "self_attn.out_proj.weight", {VD, VD});
            y.o_b = g(p + "self_attn.out_proj.bias", {VD});
            y.fc_w = g(p + "mlp.c_fc.weight", {VI, VD});
            y.fc_b = g(p + "mlp.c_fc.bias", {VI});
            y.pj_w = g(p + "mlp.c_proj.weight", {VD, VI});
            y.pj_b = g(p + "mlp.c_proj.bias", {VD});
        }
        w_.proj_in = g("proj_in.weight", {PJ, 4 * VD});
        w_.proj_out = g("proj_out.weight", {static_cast<int64_t>(d.H), PJ});

        INK2V_BLAS(cublasCreate(&blas_));
        const size_t N = ink2v::MAX_PATCHES, S = N + 1, vd = static_cast<size_t>(d.vd);
        mlp_rows_ = static_cast<int>(std::max<size_t>(1, vis::ACC_FLOATS / static_cast<size_t>(d.mlp)));
        mrg_rows_ = static_cast<int>(std::max<size_t>(1, vis::ACC_FLOATS / static_cast<size_t>(std::max(d.proj, d.H))));
        const size_t m_cells = std::max(static_cast<size_t>(mlp_rows_) * d.mlp, static_cast<size_t>(mrg_rows_) * d.proj);
        size_t off = 0;
        const auto take = [&](size_t bytes) {
            const size_t o = off;
            off += (bytes + 255) & ~static_cast<size_t>(255);
            return o;
        };
        // the patches [N][PK] stand in qkv_ before the first layer
        const size_t o_pix = take(N * ink2v::PATCH * ink2v::PATCH * 3), o_x = take(S * vd * 2), o_t = take(S * vd * 2),
                     o_qkv = take(std::max(S * 3 * vd, N * static_cast<size_t>(vis::PK)) * 2), o_m = take(m_cells * 2),
                     o_acc = take(vis::ACC_FLOATS * 4);
        INK2V_CUDA(cudaMalloc(&ws_, off));
        ws_bytes_ = off;
        char* b = static_cast<char*>(ws_);
        pix_ = reinterpret_cast<uint8_t*>(b + o_pix);
        x_ = reinterpret_cast<bf16*>(b + o_x);
        t_ = reinterpret_cast<bf16*>(b + o_t);
        qkv_ = reinterpret_cast<bf16*>(b + o_qkv);
        m_ = reinterpret_cast<bf16*>(b + o_m);
        acc_ = reinterpret_cast<float*>(b + o_acc);
        pool_rows_ = pool_rows;
        INK2V_CUDA(cudaMalloc(&pool_, static_cast<size_t>(pool_rows) * d.H * sizeof(bf16)));
        INK2V_CUDA(cudaMalloc(&stage_, static_cast<size_t>(ink2v::MAX_TOKENS) * d.H * sizeof(bf16)));
    }

    // Runs the tower on host pixels (uint8 [vit_h * 20][vit_w * 20][3]) and writes its rows (BF16 [tokens][H]) to the
    // device pointer `out`. Pixels at y >= content_h or x >= content_w are padding (-1 = none). Done when it returns.
    void encode(const uint8_t* rgb, int vit_h, int vit_w, uint16_t* out, int content_h = -1, int content_w = -1)
    {
        if (!ready()) fail("encode before init");
        if (!rgb || !ink2v::span_of(vit_h, vit_w)) fail("patch grid " + std::to_string(vit_h) + "x" + std::to_string(vit_w) + " is not a processor grid");
        const int P = ink2v::PATCH, VD = d.vd, N = vit_h * vit_w, S = N + 1, L = N / (ink2v::MERGE * ink2v::MERGE);
        if (content_h < 0) content_h = vit_h * P;
        if (content_w < 0) content_w = vit_w * P;
        ++encode_calls;
        INK2V_BLAS(cublasSetStream(blas_, 0));
        INK2V_CUDA(cudaMemcpy(pix_, rgb, static_cast<size_t>(N) * P * P * 3, cudaMemcpyHostToDevice));
        // the embeddings in t_, then h = ln_pre in x_
        vis::k_ink2v_patchify<<<N, 256>>>(pix_, vit_w, content_h, content_w, qkv_);
        linear(qkv_, vis::PK, w_.patch, nullptr, nullptr, t_ + VD, VD, N, VD, vis::PK, vis::E_BIAS);
        vis::k_ink2v_pos_add<<<S, 256>>>(t_, w_.cls, w_.cls_pos, w_.pos, vit_h, vit_w, d.grid, VD);
        vis::k_ink2v_layernorm<<<S, 256>>>(t_, w_.ln_pre_w, w_.ln_pre_b, x_, VD, d.eps);
        for (const LayerW& y : w_.layers)
        {
            vis::k_ink2v_layernorm<<<S, 256>>>(x_, y.ln1_w, y.ln1_b, t_, VD, d.eps);
            linear(t_, VD, y.q_w, y.q_b, nullptr, qkv_, 3 * VD, S, VD, VD, vis::E_BIAS);
            linear(t_, VD, y.k_w, y.k_b, nullptr, qkv_ + VD, 3 * VD, S, VD, VD, vis::E_BIAS);
            linear(t_, VD, y.v_w, y.v_b, nullptr, qkv_ + 2 * VD, 3 * VD, S, VD, VD, vis::E_BIAS);
            vis::k_ink2v_rope<<<S, dim3(32, 2 * d.heads)>>>(qkv_, vit_w, VD, d.heads, d.theta);
            vis::k_ink2v_attn<<<dim3((S + vis::AT_T - 1) / vis::AT_T, d.heads), 128>>>(qkv_, t_, S, VD);
            linear(t_, VD, y.o_w, y.o_b, y.ls1, x_, VD, S, VD, VD, vis::E_LS_RES);        // h += ls_1 * out_proj(a)
            vis::k_ink2v_layernorm<<<S, 256>>>(x_, y.ln2_w, y.ln2_b, t_, VD, d.eps);
            for (int r0 = 0; r0 < S; r0 += mlp_rows_)                                      // h += ls_2 * c_proj(gelu(c_fc(.)))
            {
                const int rows = std::min(mlp_rows_, S - r0);
                linear(t_ + static_cast<size_t>(r0) * VD, VD, y.fc_w, y.fc_b, nullptr, m_, d.mlp, rows, d.mlp, VD, vis::E_GELU);
                linear(m_, d.mlp, y.pj_w, y.pj_b, y.ls2, x_ + static_cast<size_t>(r0) * VD, VD, rows, VD, d.mlp, vis::E_LS_RES);
            }
        }
        // the merger reads the patch rows of x_ as [L][4 VD]: the 4 patches of an image token are consecutive
        bf16* o = reinterpret_cast<bf16*>(out);
        for (int r0 = 0; r0 < L; r0 += mrg_rows_)
        {
            const int rows = std::min(mrg_rows_, L - r0);
            linear(x_ + VD + static_cast<size_t>(r0) * 4 * VD, 4 * VD, w_.proj_in, nullptr, nullptr, m_, d.proj, rows, d.proj, 4 * VD, vis::E_GELU);
            linear(m_, d.proj, w_.proj_out, nullptr, nullptr, o + static_cast<size_t>(r0) * d.H, d.H, rows, d.H, d.proj, vis::E_BIAS);
        }
        INK2V_CUDA(cudaGetLastError());
        INK2V_CUDA(cudaDeviceSynchronize());
    }

    // Before a launch of `rows` rows at prompt positions first ..: runs the tower on every image whose span overlaps
    // them and copies its rows there to the pool, at the launch row of each (pool row m = position first + m). The rows
    // of the pool that are not an image's are zeros. Returns the pool, or null when no image overlaps the launch.
    const uint16_t* stage(const std::vector<g6rt::ImageIn>& images, int first, int rows, unsigned long long request)
    {
        if (images.empty()) return nullptr;
        if (!ready()) fail("an image request without the tower");
        if (rows < 1 || rows > pool_rows_) fail("a launch of " + std::to_string(rows) + " rows does not fit the image pool");
        const size_t H = static_cast<size_t>(d.H);
        int n = 0;
        for (size_t k = 0; k < images.size(); ++k)
        {
            const g6rt::ImageIn& im = images[k];
            const int a = std::max(im.pos, first), b = std::min(im.pos + im.span, first + rows);
            if (a >= b) continue;
            if (im.span != ink2v::span_of(im.vit_h, im.vit_w) || !im.rgb ||
                im.rgb->size() != static_cast<size_t>(im.vit_h) * im.vit_w * ink2v::PATCH * ink2v::PATCH * 3)
                fail("an image's span or pixels do not match its grid");
            if (n == 0) INK2V_CUDA(cudaMemset(pool_, 0, static_cast<size_t>(rows) * H * sizeof(bf16)));
            if (!(last_ == im.rgb && last_h_ == im.vit_h && last_w_ == im.vit_w))
            {
                const auto t0 = std::chrono::steady_clock::now();
                last_.reset();
                encode(im.rgb->data(), im.vit_h, im.vit_w, reinterpret_cast<uint16_t*>(stage_));
                last_ = im.rgb;
                last_h_ = im.vit_h;
                last_w_ = im.vit_w;
                if (log)
                    fprintf(stderr, "# ink2 vision: request %llu image %zu: %dx%d patches, %d tokens, tower %.1f ms\n", request, k, im.vit_h, im.vit_w,
                            im.span, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            INK2V_CUDA(cudaMemcpy(pool_ + static_cast<size_t>(a - first) * H, stage_ + static_cast<size_t>(a - im.pos) * H,
                                  static_cast<size_t>(b - a) * H * sizeof(bf16), cudaMemcpyDeviceToDevice));
            n += b - a;
        }
        return n ? reinterpret_cast<const uint16_t*>(pool_) : nullptr;
    }

    Ink2Vision() = default;
    Ink2Vision(const Ink2Vision&) = delete;
    Ink2Vision& operator=(const Ink2Vision&) = delete;
    ~Ink2Vision()
    {
        if (blas_) cublasDestroy(blas_);
        if (ws_) cudaFree(ws_);
        if (pool_) cudaFree(pool_);
        if (stage_) cudaFree(stage_);
    }

private:
    struct LayerW
    {
        const bf16 *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ls1, *ls2, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b, *fc_w, *fc_b, *pj_w, *pj_b;
    };
    struct Weights
    {
        const bf16 *cls = nullptr, *cls_pos = nullptr, *patch = nullptr, *pos = nullptr, *ln_pre_w = nullptr, *ln_pre_b = nullptr;
        const bf16 *proj_in = nullptr, *proj_out = nullptr;
        std::vector<LayerW> layers;
    };

    Weights w_;
    cublasHandle_t blas_ = nullptr;
    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    uint8_t* pix_ = nullptr;
    bf16 *x_ = nullptr, *t_ = nullptr, *qkv_ = nullptr, *m_ = nullptr;
    float* acc_ = nullptr;
    int mlp_rows_ = 0, mrg_rows_ = 0;
    bf16 *pool_ = nullptr, *stage_ = nullptr;   // [pool_rows][H]; the last image's rows [tokens][H]
    int pool_rows_ = 0;
    std::shared_ptr<const std::vector<uint8_t>> last_;   // whose rows stage_ holds (kept alive, so the pointer stays unique)
    int last_h_ = 0, last_w_ = 0;

    [[noreturn]] static void fail(const std::string& m) { throw std::runtime_error("ink2 vision: " + m); }

    // y [M][N] in rows of ldy cells (epilogue `mode`) = x [M][K] in rows of ldx cells * W [N][K]^T, in passes through
    // the accumulator.
    void linear(const bf16* x, int ldx, const bf16* W, const bf16* bias, const bf16* ls, bf16* y, int ldy, int M, int N, int K, int mode)
    {
        const float one = 1.f, zero = 0.f;
        const int pass = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(M), vis::ACC_FLOATS / static_cast<size_t>(N))));
        for (int r0 = 0; r0 < M; r0 += pass)
        {
            const int rows = std::min(pass, M - r0);
            INK2V_BLAS(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, K, x + static_cast<size_t>(r0) * ldx, CUDA_R_16BF,
                                    ldx, &zero, acc_, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            const size_t tot = static_cast<size_t>(rows) * N;
            vis::k_ink2v_epi<<<static_cast<unsigned>((tot + 255) / 256), 256>>>(acc_, bias, ls, y + static_cast<size_t>(r0) * ldy, rows, N, ldy, mode);
        }
    }
};

}   // namespace ink2rt
