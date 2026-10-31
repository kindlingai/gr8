// DeepSeek-V4.1-Flash-1017 attention B (indexer) kernels: the compressed-row producers of the ratio-1 source layer (ca row, ica row),
// the indexer (q prep, MQA logits, top-512), and the candidate blocks (layer 20 scores them, layers 24/28/32/36 read them).
//
// Semantics are the release's own inference code (Compressor, Indexer, select_candidate_blocks, RMSNorm, apply_rotary_emb,
// fp4_act_quant); engine/ref/ds41/idx_ref.py is the torch twin and engine/tests/test_ds41_idx_ref.py proves it against a line by line
// transcription of the official Indexer. Header-only and standalone: plain pointers, no gr8 runtime dependency (the Flash/GLM
// select.cuh / kpool.cuh are GLM-specific), so the step driver includes it and calls the stage functions at the bottom, or the
// kernels one by one. Notation: R query rows, rows start at position pos0; klen[r] = number of compressed rows visible to row r
// ((pos+1)/ratio; ratio 1 includes the row of the query's own token, so ca_row_r1 + ica_row must run before the logits).
//
// Row formats (engine/attn/kvfmt.h, namespace g6kv, DS41 branch):
//   ca row  288 B = [256 B FP4 e2m1 codes][32 B E4M3 scales, one per 16 dims]   all 512 dims, rope tail included. Element 2i is
//           the LOW nibble of byte i. Value = e2m1(code) * e4m3(scale). Sign bit is nibble bit 3, magnitude index 0..7 of
//           0, .5, 1, 1.5, 2, 3, 4, 6.
//   ica row  68 B = [64 B FP4 codes][4 B E8M0 scales, one per 32 dims]. Stored planar in the page (all codes of the page's rows,
//           then all scale words) so both planes stay 16 B / 4 B aligned; ica_map() builds the two plane views.
//   One 64-token page = 32 rows (ratio 2) or 64 rows (ratio 1) of each; the page table maps a sequence's logical page to a pool page.
// Numerics notes (all checked in idx_ref.py against the official code):
//   RMSNorm rounds ONCE: bf16(w * (x * rsqrt(mean(x^2) + eps))). Rope: fp32 complex product (plain mul/add, no FMA), rounded to
//   bf16; the angle is the fp32 product (float)pos * inv[i] with the official fp32 frequency table (kInvFreq below).
//   E4M3 scale = e4m3(max(amax, 6*2^-9) / 6); E8M0 scale = 2^ceil(log2(max(amax, 6*2^-126) * fp32(1/6))); code = e2m1(x / scale)
//   ties to the even code. Logits: p = bf16(sum_h bf16(relu(bf16(q_h . k_j)) * w_h)), fp32 sums; the dot is exact in int (dp4a) per
//   group of 32 dims and summed in fp32 over the 4 groups.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <math.h>

namespace ds41i
{
typedef __nv_bfloat16 bf16;

constexpr int D = 512, NOPE = 448, ROPE = 64;          // ca row width, rope tail
constexpr int CA_CODES = D / 2, CA_SF = D / 16, CA_ROW = CA_CODES + CA_SF;   // 256, 32, 288
constexpr int IH = 32, ID = 128;                      // index heads, index head dim
constexpr int IK_CODES = ID / 2, IK_SF = ID / 32, ICA_ROW = IK_CODES + IK_SF;   // 64, 4, 68
constexpr int TOPK = 512, CAND_BLOCKS = 2048, CAND_BS = 8, CAND_COLS = CAND_BLOCKS * CAND_BS;   // 16384
constexpr float EPS = 1e-20f;                         // geom::RMS_EPS (norm_eps)
constexpr float W_SCALE = 0.015625f;                  // 2^-6 = index_head_dim^-0.5 * index_n_heads^-0.5, exact
constexpr float INV6 = 0.16666667163372040f;          // float32(1/6), the fp4 quantizer's constant

// RoPE inverse frequencies of the compressed-row rope (theta 160000, YaRN factor 16 / 65536 / beta 32, 1), float32, computed by the
// official precompute_freqs_cis in fp32 (torch). A double-precision route differs in 9 of the 32 entries (last ulp).
struct RopeInv { float v[ROPE / 2]; };
inline RopeInv rope_inv_ds41()
{
    static const unsigned bits[32] = {
        0x3f800000u, 0x3f300a3au, 0x3ef21c1fu, 0x3ea67d01u, 0x3e64f92eu, 0x3e1d7475u, 0x3dd88cb4u, 0x3d94e963u,
        0x3d4ccccdu, 0x3d0cd4fbu, 0x3cc1b019u, 0x3c8530ceu, 0x3c372dbfu, 0x3bfbed88u, 0x3bad3d5eu, 0x3b6e4237u,
        0x3b147ae1u, 0x3ab714e0u, 0x3a5ebdb6u, 0x3a0530ceu, 0x399bb3afu, 0x39305978u, 0x38be904du, 0x383e9b5fu,
        0x37a3d70cu, 0x36b443d0u, 0x3677eba6u, 0x362a7be8u, 0x35ea77ffu, 0x35a13bdcu, 0x355dbf30u, 0x35187c4du};
    RopeInv r;
    for (int i = 0; i < 32; i++) { unsigned b = bits[i]; r.v[i] = *reinterpret_cast<float*>(&b); }
    return r;
}

// ------------------------------------------------------------------------------------------------ row addressing
// One cache plane of one sequence: row j sits at base + pages[j / rows_per_page] * page_stride + (j % rows_per_page) * row_stride.
struct PagedRows
{
    uint8_t* base;
    const int32_t* pages;
    int rows_per_page;
    int row_stride;
    size_t page_stride;
    __host__ __device__ __forceinline__ uint8_t* row(int j) const
    {
        return base + (size_t)pages[j / rows_per_page] * page_stride + (size_t)(j % rows_per_page) * row_stride;
    }
};
// ca rows of a layer: PagedRows with row_stride CA_ROW (288), rows_per_page 32 (ratio 2) or 64 (ratio 1).
inline PagedRows ca_map(void* page0, const int32_t* pages, int rows_per_page, size_t page_stride)
{
    return PagedRows{(uint8_t*)page0, pages, rows_per_page, CA_ROW, page_stride};
}
// ica rows: the page holds rows_per_page * 64 B of codes, then rows_per_page * 4 B of scale words.
struct IcaMap
{
    PagedRows codes, sf;
};
inline IcaMap ica_map(void* page0, const int32_t* pages, int rows_per_page, size_t page_stride)
{
    IcaMap m;
    m.codes = PagedRows{(uint8_t*)page0, pages, rows_per_page, IK_CODES, page_stride};
    m.sf = PagedRows{(uint8_t*)page0 + (size_t)rows_per_page * IK_CODES, pages, rows_per_page, IK_SF, page_stride};
    return m;
}

// ------------------------------------------------------------------------------------------------ small device helpers
__device__ __forceinline__ float f_e4m3(uint8_t c)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
    return __half2float(__half(h));
}
__device__ __forceinline__ float f_e8m0(uint8_t b) { return __uint_as_float(b ? ((uint32_t)b << 23) : 0x00400000u); }
__device__ __forceinline__ float pow2i(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }
__device__ __forceinline__ int log2ceil_bits(float x)
{
    uint32_t b = __float_as_uint(x);
    return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0);
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float warp_max(float v)
{
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}
// sum over all threads of a block (all must call); sh = 32 floats of scratch; every thread gets the total
__device__ __forceinline__ float block_sum(float v, float* sh)
{
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < nw; i++) t += sh[i];
    return t;
}
// rotate the pair (a, b) by angle ang with plain mul/add (the reference's complex multiply, no FMA contraction)
__device__ __forceinline__ void rot_pair(float a, float b, float ang, float& ra, float& rb)
{
    // cos / sin of the fp32 angle evaluated in double and rounded to fp32 (correctly rounded, which is what the oracle's table holds;
    // sincosf is 1-2 ulp). 32 angles per row, so the fp64 cost does not matter.
    double sd, cd;
    sincos((double)ang, &sd, &cd);
    const float s = (float)sd, c = (float)cd;
    ra = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    rb = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}
// e2m1 code (sign bit 3, magnitude index 0..7) of v: nearest, ties to the even code, saturating (midpoints .25 .75 1.25 1.75 2.5 3.5 5)
__device__ __forceinline__ uint32_t e2m1_code(float v)
{
    const float a = fabsf(v);
    const uint32_t m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return m | ((v < 0.0f && m) ? 8u : 0u);
}
// value of an e2m1 code (sign bit 3, magnitude index 0..7)
__device__ __forceinline__ float e2m1_value(uint32_t code)
{
    const uint32_t m = code & 7;
    const float v = m <= 4 ? 0.5f * (float)m : (m == 5 ? 3.0f : (m == 6 ? 4.0f : 6.0f));
    return (code & 8) ? -v : v;
}
// Four fp4 codes (the low 16 bits of sel, element j in nibble j) -> four int8 holding 2 * value (0, 1, 2, 3, 4, 6, 8, 12), one per
// byte, element j in byte j. Magnitude by byte permute, sign as a per-byte two's-complement negate.
__device__ __forceinline__ uint32_t fp4x4_to_i8x4(uint32_t sel)
{
    const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel & 0x7777u);
    const uint32_t neg = __byte_perm(0u, 0xFFFFFFFFu, (sel >> 1) & 0x4444u);
    return __vsub4(mag ^ neg, neg);
}

// ------------------------------------------------------------------------------------------------ ca row, ratio 1 (layer 20)
struct CaR1P
{
    const bf16* y;        // [T][512] wkv(x), the bf16 projection output
    const bf16* norm_w;   // [512] compressor.norm.weight
    RopeInv inv;
    int pos0;             // token t has position pos0 + t: its ca row index and its rope position
    PagedRows ca;         // destination
    bf16* pre;            // optional [T][512]: the post-norm PRE-rope latent (the ica row's input)
    bf16* roped;          // optional [T][512] debug: the post-rope latent before fp4
    const int* dpos;      // optional device positions [T] (engine step, graph-safe): position of token t = dpos[t] instead of pos0 + t
    bf16* shadow;         // optional [capacity][512] bf16 copy of the stored row (e2m1 * e4m3, exact), row index = position
};
// One block of 512 per row (thread = dim): y -> RMSNorm -> [pre] -> rope tail -> FP4 e4m3/16 -> 288 B row.
__global__ void __launch_bounds__(512) k_ca_row_r1(CaR1P p)
{
    __shared__ float sh[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const int pos = p.dpos ? p.dpos[t] : p.pos0 + t;
    const float v = __bfloat162float(p.y[(size_t)t * D + c]);
    const float ss = block_sum(v * v, sh);
    const float r = rsqrtf(ss / (float)D + EPS);
    float x = bf16r(__bfloat162float(p.norm_w[c]) * (v * r));
    if (p.pre) p.pre[(size_t)t * D + c] = __float2bfloat16_rn(x);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (c >= NOPE)
    {
        const int i = (c - NOPE) >> 1;
        const float ang = (float)pos * p.inv.v[i];
        float ra, rb;
        if ((c & 1) == 0) { rot_pair(x, partner, ang, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, ra, rb); x = bf16r(rb); }
    }
    if (p.roped) p.roped[(size_t)t * D + c] = __float2bfloat16_rn(x);
    float a = fabsf(x);
    for (int o = 8; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));   // the 16-lane group
    a = fmaxf(a, 6.0f * 0.001953125f);                                              // 6 * 2^-9
    const uint8_t sc8 = (uint8_t)__nv_cvt_float_to_fp8(__fdiv_rn(a, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float sc = f_e4m3(sc8);
    const uint32_t code = e2m1_code(fminf(fmaxf(__fdiv_rn(x, sc), -6.0f), 6.0f));
    const uint32_t nb = __shfl_xor_sync(0xFFFFFFFFu, code, 1);
    uint8_t* row = p.ca.row(pos);
    if ((c & 1) == 0) row[c >> 1] = (uint8_t)(code | (nb << 4));
    if ((c & 15) == 0) row[CA_CODES + (c >> 4)] = sc8;
    // the shadow row sits where the ca row does: page pages[pos / rpp] (SESS > 1: the session's slot slab; one session: page 0, row = pos)
    if (p.shadow)
        p.shadow[((size_t)p.ca.pages[pos / p.ca.rows_per_page] * p.ca.rows_per_page + pos % p.ca.rows_per_page) * D + c] =
            __float2bfloat16_rn(e2m1_value(code) * sc);
}

// ------------------------------------------------------------------------------------------------ fp4 e8m0/32 quantizer (warp = one group of 32)
// x: this lane's value (already bf16-valued). Returns the code of x; ebyte = the group's E8M0 byte.
__device__ __forceinline__ uint32_t quant_e8m0_lane(float x, uint32_t& ebyte)
{
    float a = warp_max(fabsf(x));
    a = fmaxf(a, 6.0f * 1.1754943508222875e-38f);   // 6 * 2^-126
    const int e = log2ceil_bits(__fmul_rn(a, INV6));
    ebyte = (uint32_t)min(max(e + 127, 0), 255);
    return e2m1_code(fminf(fmaxf(__fdiv_rn(x, pow2i(e)), -6.0f), 6.0f));
}

// ------------------------------------------------------------------------------------------------ ica row (index key)
struct IcaP
{
    const bf16* pre;      // [T][512] post-norm pre-rope ca latent of rows row0 .. row0 + T - 1
    const bf16* wk;       // [128][512] indexer.wk
    const bf16* knorm_w;  // [128] indexer.k_norm.weight
    RopeInv inv;
    int row0, ratio;      // row j has rope position j * ratio
    IcaMap ica;           // destination rows row0 + t
    bf16* kdbg;           // optional [T][128] debug: k after norm + rope (bf16)
    const int* dpos;      // optional device positions [T] (engine step): token t has position dpos[t]; ratio 2: only odd positions make a row
                          // (row = pos / 2, input slot (t + (dpos[0] & 1)) / 2 of pre), ratio 1: row = pos, slot t
};
// One block of 256 per row. k = bf16(pre . wk^T) (fp32 accumulate, fixed order), RMSNorm, rope tail (dims 64..127), FP4 e8m0/32.
__global__ void __launch_bounds__(256) k_ica_row(IcaP p)
{
    __shared__ float spre[D];
    __shared__ float sk[ID];
    __shared__ float sh[32];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int row = p.row0 + t, slot = t;
    if (p.dpos)
    {
        const int pos = p.dpos[t];
        if (p.ratio == 2)
        {
            if (!(pos & 1)) return;   // block-uniform: this token completes no pair
            row = pos >> 1;
            slot = (t + (p.dpos[0] & 1)) >> 1;
        }
        else { row = pos / p.ratio; slot = t; }
    }
    for (int i = tid; i < D; i += 256) spre[i] = __bfloat162float(p.pre[(size_t)slot * D + i]);
    __syncthreads();
    // warp w: output dims w*16 .. w*16+15; lane handles k = h*256 + lane*8 .. +7 (coalesced 16 B per lane)
    for (int d = w * 16; d < w * 16 + 16; d++)
    {
        float acc = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            const uint4 raw = *reinterpret_cast<const uint4*>(p.wk + (size_t)d * D + h * 256 + lane * 8);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                const float2 f = __bfloat1622float2(b2[j]);
                const int k = h * 256 + lane * 8 + 2 * j;
                acc = fmaf(spre[k], f.x, acc);
                acc = fmaf(spre[k + 1], f.y, acc);
            }
        }
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        if (lane == 0) sk[d] = bf16r(acc);
    }
    __syncthreads();
    const float kv = tid < ID ? sk[tid] : 0.0f;
    const float ss = block_sum(kv * kv, sh);
    const float r = rsqrtf(ss / (float)ID + EPS);
    if (tid >= ID) return;
    float x = bf16r(__bfloat162float(p.knorm_w[tid]) * (kv * r));
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (tid >= ID - ROPE)
    {
        const int i = (tid - (ID - ROPE)) >> 1;
        const float ang = (float)(row * p.ratio) * p.inv.v[i];
        float ra, rb;
        if ((tid & 1) == 0) { rot_pair(x, partner, ang, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, ra, rb); x = bf16r(rb); }
    }
    if (p.kdbg) p.kdbg[(size_t)t * ID + tid] = __float2bfloat16_rn(x);
    uint32_t eb;
    const uint32_t code = quant_e8m0_lane(x, eb);
    const uint32_t nb = __shfl_xor_sync(0xFFFFFFFFu, code, 1);
    if ((tid & 1) == 0) p.ica.codes.row(row)[tid >> 1] = (uint8_t)(code | (nb << 4));
    if (lane == 0) p.ica.sf.row(row)[w] = (uint8_t)eb;
}

// ------------------------------------------------------------------------------------------------ index q (rope + fp4) and weights
struct IdxQP
{
    const bf16* q_b;      // [R][32 * 128] indexer.wq_b(q_a) output, bf16
    const bf16* wproj;    // [R][32] indexer.weights_proj(x) output, bf16
    RopeInv inv;
    int pos0;             // row r has query position pos0 + r (the rope position of the query)
    uint8_t* q_pack;      // [R][32][64]
    uint8_t* q_exp;       // [R][32][4]
    bf16* w;              // [R][32] bf16(wproj * 2^-6)
    const int* dpos = nullptr;   // optional device positions [R]: query position of row r = dpos[r] instead of pos0 + r
    float2* trig = nullptr; // optional [R][32] once-per-row fp64 sincos, x=cos, y=sin
};
// The angle expression and fp64 call match k_index_q exactly. One CTA owns a row;
// 32 lanes write the 32 pairs that all 32 head CTAs subsequently read.
__global__ void k_index_q_trig(IdxQP p)
{
    const int r = blockIdx.x, i = threadIdx.x;
    const float ang = (float)(p.dpos ? p.dpos[r] : p.pos0 + r) * p.inv.v[i];
    double sd, cd;
    sincos((double)ang, &sd, &cd);
    p.trig[(size_t)r * 32 + i] = make_float2((float)cd, (float)sd);
}
// grid (R, 32 heads), 128 threads (thread = dim).
__global__ void __launch_bounds__(128) k_index_q(IdxQP p)
{
    const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    float x = __bfloat162float(p.q_b[((size_t)r * IH + h) * ID + tid]);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (tid >= ID - ROPE)
    {
        const int i = (tid - (ID - ROPE)) >> 1;
        const float ang = (float)(p.dpos ? p.dpos[r] : p.pos0 + r) * p.inv.v[i];
        float ra, rb;
        if ((tid & 1) == 0) { rot_pair(x, partner, ang, ra, rb); x = bf16r(ra); }
        else { rot_pair(partner, x, ang, ra, rb); x = bf16r(rb); }
    }
    uint32_t eb;
    const uint32_t code = quant_e8m0_lane(x, eb);
    const uint32_t nb = __shfl_xor_sync(0xFFFFFFFFu, code, 1);
    if ((tid & 1) == 0) p.q_pack[((size_t)r * IH + h) * IK_CODES + (tid >> 1)] = (uint8_t)(code | (nb << 4));
    if (lane == 0) p.q_exp[((size_t)r * IH + h) * IK_SF + (tid >> 5)] = (uint8_t)eb;
    if (h == 0 && tid < IH) p.w[(size_t)r * IH + tid] = __float2bfloat16_rn(__bfloat162float(p.wproj[(size_t)r * IH + tid]) * W_SCALE);
}
// Same quantization and arithmetic as k_index_q, with only the trigonometry
// moved into k_index_q_trig. k_index_q runs when the knob is off.
__global__ void __launch_bounds__(128) k_index_q_cached(IdxQP p)
{
    const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    float x = __bfloat162float(p.q_b[((size_t)r * IH + h) * ID + tid]);
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, x, 1);
    if (tid >= ID - ROPE)
    {
        const int i = (tid - (ID - ROPE)) >> 1;
        const float2 cs = p.trig[(size_t)r * 32 + i];
        float ra, rb;
        if ((tid & 1) == 0) { ra = __fsub_rn(__fmul_rn(x, cs.x), __fmul_rn(partner, cs.y)); x = bf16r(ra); }
        else { rb = __fadd_rn(__fmul_rn(partner, cs.y), __fmul_rn(x, cs.x)); x = bf16r(rb); }
    }
    uint32_t eb;
    const uint32_t code = quant_e8m0_lane(x, eb);
    const uint32_t nb = __shfl_xor_sync(0xFFFFFFFFu, code, 1);
    if ((tid & 1) == 0) p.q_pack[((size_t)r * IH + h) * IK_CODES + (tid >> 1)] = (uint8_t)(code | (nb << 4));
    if (lane == 0) p.q_exp[((size_t)r * IH + h) * IK_SF + (tid >> 5)] = (uint8_t)eb;
    if (h == 0 && tid < IH) p.w[(size_t)r * IH + tid] = __float2bfloat16_rn(__bfloat162float(p.wproj[(size_t)r * IH + tid]) * W_SCALE);
}

// ------------------------------------------------------------------------------------------------ MQA logits
struct LogitsP
{
    const uint8_t* q_pack;   // [R][32][64]
    const uint8_t* q_exp;    // [R][32][4]
    const bf16* w;           // [R][32]
    IcaMap k;                // index keys (ica rows of the index-K source layer), indexed by token (= compressed row) id
    const int32_t* klen;     // [R] visible rows per query row
    const int32_t* cand;     // [R][ldc] candidate block ids ascending, -1 pad (GATHER only)
    int ldc;
    bf16* p;                 // [R][ldp]
    long ldp;
    int ncols;               // columns to produce per row: dense = max klen (rounded up to a multiple of 128 by the grid), gather = CAND_COLS
    int R;
};
// Dense: column c is token c. Gather (layers 24/28/32/36): column slot * 8 + o is token cand[r][slot] * 8 + o. Columns the row does
// not see (c >= klen, a -1 slot) are written -inf. Block = 128 keys (thread = key) x RB query rows (dense) or 1 query row (gather).
template <int RB, bool GATHER>
__global__ void __launch_bounds__(128) k_logits(LogitsP p)
{
    __shared__ __align__(16) int32_t sq[RB][IH][32];
    __shared__ float ssc[RB][IH][4];
    __shared__ float sw[RB][IH];
    const int tid = threadIdx.x;
    const int r0 = GATHER ? (int)blockIdx.y : (int)blockIdx.y * RB;
    const int nr = GATHER ? 1 : min(RB, p.R - r0);
    for (int i = tid; i < nr * IH * 32; i += 128)
    {
        const int rr = i / (IH * 32), h = (i >> 5) & 31, wd = i & 31;
        const uint16_t two = *reinterpret_cast<const uint16_t*>(p.q_pack + ((size_t)(r0 + rr) * IH + h) * IK_CODES + wd * 2);
        sq[rr][h][wd] = (int32_t)fp4x4_to_i8x4(two);
    }
    for (int i = tid; i < nr * IH * 4; i += 128)
    {
        const int rr = i >> 7, h = (i >> 2) & 31, g = i & 3;
        ssc[rr][h][g] = f_e8m0(p.q_exp[((size_t)(r0 + rr) * IH + h) * IK_SF + g]);
    }
    for (int i = tid; i < nr * IH; i += 128) sw[i / IH][i % IH] = __bfloat162float(p.w[(size_t)r0 * IH + i]);
    __syncthreads();

    const int c = blockIdx.x * 128 + tid;
    if (c >= p.ncols) return;
    int tok = c;
    if (GATHER)
    {
        const int b = p.cand[(size_t)r0 * p.ldc + (c >> 3)];
        tok = b >= 0 ? b * CAND_BS + (c & 7) : -1;
    }
    int kmax = 0;
    for (int rr = 0; rr < nr; rr++) kmax = max(kmax, p.klen[r0 + rr]);
    const bf16 ninf = __float2bfloat16_rn(-INFINITY);
    if (tok < 0 || tok >= kmax)
    {
        for (int rr = 0; rr < nr; rr++) p.p[(size_t)(r0 + rr) * p.ldp + c] = ninf;
        return;
    }
    int32_t k4[32];
    {
        const uint4* kc = reinterpret_cast<const uint4*>(p.k.codes.row(tok));
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            const uint4 u = kc[i];
            k4[i * 8 + 0] = (int32_t)fp4x4_to_i8x4(u.x & 0xFFFFu);
            k4[i * 8 + 1] = (int32_t)fp4x4_to_i8x4(u.x >> 16);
            k4[i * 8 + 2] = (int32_t)fp4x4_to_i8x4(u.y & 0xFFFFu);
            k4[i * 8 + 3] = (int32_t)fp4x4_to_i8x4(u.y >> 16);
            k4[i * 8 + 4] = (int32_t)fp4x4_to_i8x4(u.z & 0xFFFFu);
            k4[i * 8 + 5] = (int32_t)fp4x4_to_i8x4(u.z >> 16);
            k4[i * 8 + 6] = (int32_t)fp4x4_to_i8x4(u.w & 0xFFFFu);
            k4[i * 8 + 7] = (int32_t)fp4x4_to_i8x4(u.w >> 16);
        }
    }
    const uint32_t ke = *reinterpret_cast<const uint32_t*>(p.k.sf.row(tok));
    float sk[4];
#pragma unroll
    for (int g = 0; g < 4; g++) sk[g] = f_e8m0((uint8_t)(ke >> (8 * g))) * 0.25f;   // 1/4: the codes are doubled twice
#pragma unroll 1
    for (int rr = 0; rr < nr; rr++)
    {
        if (tok >= p.klen[r0 + rr])
        {
            p.p[(size_t)(r0 + rr) * p.ldp + c] = ninf;
            continue;
        }
        float s = 0.0f;
#pragma unroll 1
        for (int h = 0; h < IH; h++)
        {
            const uint4* qv = reinterpret_cast<const uint4*>(&sq[rr][h][0]);
            float dot = 0.0f;
#pragma unroll
            for (int g = 0; g < 4; g++)
            {
                const uint4 a0 = qv[2 * g], a1 = qv[2 * g + 1];
                int acc = __dp4a((int)a0.x, k4[8 * g + 0], 0);
                acc = __dp4a((int)a0.y, k4[8 * g + 1], acc);
                acc = __dp4a((int)a0.z, k4[8 * g + 2], acc);
                acc = __dp4a((int)a0.w, k4[8 * g + 3], acc);
                acc = __dp4a((int)a1.x, k4[8 * g + 4], acc);
                acc = __dp4a((int)a1.y, k4[8 * g + 5], acc);
                acc = __dp4a((int)a1.z, k4[8 * g + 6], acc);
                acc = __dp4a((int)a1.w, k4[8 * g + 7], acc);
                dot = dot + (float)acc * (ssc[rr][h][g] * sk[g]);
            }
            const float db = bf16r(dot);
            s += bf16r((db > 0.0f ? db : 0.0f) * sw[rr][h]);
        }
        p.p[(size_t)(r0 + rr) * p.ldp + c] = __float2bfloat16_rn(s);
    }
}


// ------------------------------------------------------------------------------------------------ dense logits on the tensor cores
// The same logits as k_logits (dense), computed with mma.sync m16n8k64 kind::mxf4nvf4 block_scale (e2m1 x e2m1, ue8m0 per 32, fp32 acc),
// the instruction CUTLASS wraps as SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue8m0, 32>. D[16 heads of query row r][8 keys] over K = 128 dims
// is two MMAs (dims 0..63, 64..127), the second accumulating onto the first; the epilogue is the reference's per-head rounding
// (bf16 of the dot, relu, times the head weight, bf16) and an fp32 sum over the 32 heads (4 terms per lane, then a xor tree over the 8
// lanes of equal lane % 4: a different add order from k_logits' head loop, so it may differ in the last fp32 bit before the final bf16 round).
// Fragment layout (CUTLASS mma_traits_sm120.hpp SM120_16x8x64_TN_VS): lane = 4 m + t; A regs a0..a3 hold, for rows m (a0, a2) and m + 8
// (a1, a3), 8 consecutive dims k = 8 t .. 8 t + 7 of K-block 0 (a0, a1) or K-block 1 (a2, a3), low nibble first; B regs b0, b1: key n = m,
// the same 8 dims of K-block 0 / 1; scale regs hold two ue8m0 bytes (K-block 0, 1): A of row m + 8 (lane & 1), B of key m. C: c0, c1 =
// (row m, keys 2 t, 2 t + 1), c2, c3 = (row m + 8, same keys). A K-block is one 32-dim group of the index row, so no k permutation is needed.
__device__ __forceinline__ void mma_mxf4(float* d, const uint4 a, uint32_t b0, uint32_t b1, const float* c, uint32_t sfa, uint32_t sfb)
{
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa), "h"(z), "h"(z),
          "r"(sfb), "h"(z), "h"(z));
#else
    __trap();
#endif
}
constexpr int MMA_RB = 8;    // query rows per CTA
constexpr int MMA_TP = 36;   // EXACT mode: row pitch (floats) of the per-warp head-term transpose buffer (2-way store conflicts at worst)
// EXACT = false: one MMA per 64 dims with the E8M0 scales applied inside the tensor core, head terms summed in a shuffle tree. Fast, but
//   its fp32 rounding differs from k_logits when the 4 groups' scales are far apart (wide-scale synthetic data: ~1e-5 of the outputs).
// EXACT = true: one MMA per 32-dim group with unit scales (the group dot is a small exact integer / 4, so the tensor core adds nothing
//   inexact), then k_logits' own fp32 recurrence per head, dot = fma(group dot, q scale * k scale, dot) over g = 0..3, the same bf16
//   roundings, and the head terms summed in head order 0..31 from a shared-memory transpose: bit-identical to k_logits on every input.
// One warp = 32 consecutive keys per iteration (4 tiles of 8), a CTA = 4 warps, a persistent grid strides over the key units.
template <bool EXACT>
__global__ void __launch_bounds__(128) k_logits_mma(LogitsP p)
{
    __shared__ __align__(16) uint4 sA[MMA_RB * 4 * 32];
    __shared__ uint32_t sSF[MMA_RB * 4 * 32];
    __shared__ float sw[MMA_RB][IH];
    __shared__ float ssq[EXACT ? MMA_RB : 1][EXACT ? IH : 1][EXACT ? 4 : 1];
    __shared__ float sT[EXACT ? 4 : 1][EXACT ? IH * MMA_TP : 1];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.y * MMA_RB, nr = min(MMA_RB, p.R - r0);
    for (int i = tid; i < nr * 4 * 32; i += 128)
    {
        const int l = i & 31, kh = (i >> 5) & 1, rt = (i >> 6) & 1, rr = i >> 7;
        const int t = l & 3, m = l >> 2;
        const uint8_t* qa = p.q_pack + ((size_t)(r0 + rr) * IH + rt * 16 + m) * IK_CODES;
        const uint8_t* qb = qa + 8 * IK_CODES;
        uint4 a;
        a.x = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 4 * t);
        a.y = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 4 * t);
        a.z = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 16 + 4 * t);
        a.w = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 16 + 4 * t);
        sA[i] = a;
        const int hs = rt * 16 + (l >> 2) + 8 * (l & 1);
        sSF[i] = *reinterpret_cast<const uint16_t*>(p.q_exp + ((size_t)(r0 + rr) * IH + hs) * IK_SF + 2 * kh);
    }
    for (int i = tid; i < nr * IH; i += 128) sw[i / IH][i % IH] = __bfloat162float(p.w[(size_t)r0 * IH + i]);
    if constexpr (EXACT)
        for (int i = tid; i < nr * IH * 4; i += 128) ssq[i >> 7][(i >> 2) & 31][i & 3] = f_e8m0(p.q_exp[(size_t)r0 * IH * IK_SF + i]);
    __syncthreads();
    int kmax = 0;
    for (int rr = 0; rr < nr; rr++) kmax = max(kmax, p.klen[r0 + rr]);
    kmax = min(kmax, p.ncols);
    const bf16 ninf = __float2bfloat16_rn(-INFINITY);
    const int t = lane & 3, n = lane >> 2;
    const int units = (p.ncols + 31) / 32;
    for (int u = blockIdx.x * 4 + warp; u < units; u += gridDim.x * 4)
    {
        const int c0 = u * 32;
        if (c0 >= kmax)
        {
            for (int rr = 0; rr < nr; rr++)
                if (c0 + lane < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + c0 + lane] = ninf;
            continue;
        }
        uint32_t bw[4][4], sfb[4][2];
#pragma unroll
        for (int tl = 0; tl < 4; tl++)
        {
            const int tok = c0 + tl * 8 + n;
            if (tok < kmax)
            {
                const uint8_t* cr = p.k.codes.row(tok);
                bw[tl][0] = *reinterpret_cast<const uint32_t*>(cr + 4 * t);
                bw[tl][1] = *reinterpret_cast<const uint32_t*>(cr + 16 + 4 * t);
                bw[tl][2] = *reinterpret_cast<const uint32_t*>(cr + 32 + 4 * t);
                bw[tl][3] = *reinterpret_cast<const uint32_t*>(cr + 48 + 4 * t);
                const uint32_t se = *reinterpret_cast<const uint32_t*>(p.k.sf.row(tok));
                sfb[tl][0] = se & 0xFFFFu;
                sfb[tl][1] = se >> 16;
            }
            else
            {
                bw[tl][0] = bw[tl][1] = bw[tl][2] = bw[tl][3] = 0u;
                sfb[tl][0] = sfb[tl][1] = 0x7F7Fu;
            }
        }
        if constexpr (EXACT)
        {
            float skf[4][2][4];   // scales of keys 2t, 2t+1 of tile tl (the D columns this lane holds), group g
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
            {
                const uint32_t se = sfb[tl][0] | (sfb[tl][1] << 16);
#pragma unroll
                for (int jj = 0; jj < 2; jj++)
                {
                    const uint32_t wk = __shfl_sync(0xFFFFFFFFu, se, 4 * (2 * t + jj));
#pragma unroll
                    for (int g = 0; g < 4; g++) skf[tl][jj][g] = f_e8m0((uint8_t)(wk >> (8 * g)));
                }
            }
            float* T = sT[warp];
#pragma unroll 1
            for (int rr = 0; rr < nr; rr++)
            {
#pragma unroll
                for (int rt = 0; rt < 2; rt++)
                {
                    const uint4 a0 = sA[((rr * 2 + rt) * 2 + 0) * 32 + lane], a1 = sA[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                    float sqv[2][4], wv[2];
#pragma unroll
                    for (int hr = 0; hr < 2; hr++)
                    {
#pragma unroll
                        for (int g = 0; g < 4; g++) sqv[hr][g] = ssq[rr][rt * 16 + hr * 8 + n][g];
                        wv[hr] = sw[rr][rt * 16 + hr * 8 + n];
                    }
#pragma unroll
                    for (int tl = 0; tl < 4; tl++)
                    {
                        const float z4[4] = {0.f, 0.f, 0.f, 0.f};
                        float d[4][4];
                        mma_mxf4(d[0], make_uint4(a0.x, a0.y, 0u, 0u), bw[tl][0], bw[tl][1], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[1], make_uint4(0u, 0u, a0.z, a0.w), bw[tl][0], bw[tl][1], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[2], make_uint4(a1.x, a1.y, 0u, 0u), bw[tl][2], bw[tl][3], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[3], make_uint4(0u, 0u, a1.z, a1.w), bw[tl][2], bw[tl][3], z4, 0x7F7Fu, 0x7F7Fu);
#pragma unroll
                        for (int e = 0; e < 4; e++)
                        {
                            const int hr = e >> 1, jj = e & 1;
                            float dot = 0.0f;
#pragma unroll
                            for (int g = 0; g < 4; g++) dot = fmaf(d[g][e], sqv[hr][g] * skf[tl][jj][g], dot);
                            const float db = bf16r(dot);
                            T[(rt * 16 + hr * 8 + n) * MMA_TP + tl * 8 + 2 * t + jj] = bf16r((db > 0.0f ? db : 0.0f) * wv[hr]);
                        }
                    }
                }
                __syncwarp();
                float sum = 0.0f;
#pragma unroll
                for (int h = 0; h < IH; h++) sum += T[h * MMA_TP + lane];
                const int col = c0 + lane;
                if (col < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + col] = col < p.klen[r0 + rr] ? __float2bfloat16_rn(sum) : ninf;
                __syncwarp();
            }
            continue;
        }
#pragma unroll 1
        for (int rr = 0; rr < nr; rr++)
        {
            float ps[4][2];
#pragma unroll
            for (int tl = 0; tl < 4; tl++) ps[tl][0] = ps[tl][1] = 0.0f;
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
            {
                const uint4 a0 = sA[((rr * 2 + rt) * 2 + 0) * 32 + lane], a1 = sA[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const uint32_t f0 = sSF[((rr * 2 + rt) * 2 + 0) * 32 + lane], f1 = sSF[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const float w_lo = sw[rr][rt * 16 + n], w_hi = sw[rr][rt * 16 + n + 8];
#pragma unroll
                for (int tl = 0; tl < 4; tl++)
                {
                    float acc[4] = {0.f, 0.f, 0.f, 0.f}, d1[4], d2[4];
                    mma_mxf4(d1, a0, bw[tl][0], bw[tl][1], acc, f0, sfb[tl][0]);
                    mma_mxf4(d2, a1, bw[tl][2], bw[tl][3], d1, f1, sfb[tl][1]);
#pragma unroll
                    for (int j = 0; j < 2; j++)
                    {
                        const float x0 = bf16r(d2[j]), x1 = bf16r(d2[2 + j]);
                        ps[tl][j] += bf16r((x0 > 0.0f ? x0 : 0.0f) * w_lo);
                        ps[tl][j] += bf16r((x1 > 0.0f ? x1 : 0.0f) * w_hi);
                    }
                }
            }
            const int kl = p.klen[r0 + rr];
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    float v = ps[tl][j];
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
                    const int col = c0 + tl * 8 + 2 * t + j;
                    if (n == 0 && col < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + col] = col < kl ? __float2bfloat16_rn(v) : ninf;
                }
        }
    }
}

// Pipelined dense index logits: native FP4/UE8M0, persistent CTA, two staged K tiles. The plain launcher stays the default.
// EXACT = true (arm A) keeps the plain kernel's exact numerical order; EXACT = false (arm B) uses the real scales inside the same SM121a MMA.
constexpr int PIPE_RB = 4;
template <bool EXACT>
__global__ void __launch_bounds__(256) k_logits_pipe(LogitsP p)
{
    __shared__ __align__(16) uint4 sA[PIPE_RB * 4 * 32];
    __shared__ uint32_t sSF[PIPE_RB * 4 * 32];
    __shared__ float sw[PIPE_RB][IH];
    __shared__ float ssq[EXACT ? PIPE_RB : 1][EXACT ? IH : 1][EXACT ? 4 : 1];
    __shared__ float sT[EXACT ? 4 : 1][EXACT ? IH * MMA_TP : 1];
    __shared__ uint32_t sKC[2][128][16];   // 128 rows x 64 packed bytes, two stages
    __shared__ uint32_t sKE[2][128];       // 4 UE8M0 bytes per key, planar source
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const bool math = wid < 4;
    const int warp = wid;                  // math warps 0..3, IO warps 4..7
    const int tiles = (p.ncols + 127) / 128;
    const int tasks = ((p.R + PIPE_RB - 1) / PIPE_RB) * tiles;
    const bf16 ninf = __float2bfloat16_rn(-INFINITY);
    int task = blockIdx.x, stage = 0;
    // Producer loads one row per IO lane; no new PTX opcode, no TMA or cp.async.
    if (!math && task < tasks)
    {
        const int tok = (task % tiles) * 128 + (wid - 4) * 32 + lane;
        const int ix = (wid - 4) * 32 + lane;
        if (tok < p.ncols)
        {
            const uint8_t* cr = p.k.codes.row(tok);
#pragma unroll
            for (int g = 0; g < 4; g++)
            {
                const uint4 v = *reinterpret_cast<const uint4*>(cr + g * 16);
                sKC[stage][ix][4*g+0] = v.x; sKC[stage][ix][4*g+1] = v.y;
                sKC[stage][ix][4*g+2] = v.z; sKC[stage][ix][4*g+3] = v.w;
            }
            sKE[stage][ix] = *reinterpret_cast<const uint32_t*>(p.k.sf.row(tok));
        }
        else { for (int g = 0; g < 16; g++) sKC[stage][ix][g] = 0; sKE[stage][ix] = 0x7F7F7F7Fu; }
    }
    __syncthreads();
    for (; task < tasks; task += gridDim.x)
    {
        const int r0 = (task / tiles) * PIPE_RB;
        const int nr = min(PIPE_RB, p.R - r0);
        const int base = (task % tiles) * 128;
        // Both roles cooperatively stage Q while the current K tile is already ready.
        for (int i = tid; i < nr * 4 * 32; i += 256)
        {
            const int l = i & 31, kh = (i >> 5) & 1, rt = (i >> 6) & 1, rr = i >> 7;
            const int t = l & 3, m = l >> 2;
            const uint8_t* qa = p.q_pack + ((size_t)(r0 + rr) * IH + rt * 16 + m) * IK_CODES;
            const uint8_t* qb = qa + 8 * IK_CODES;
            uint4 a;
            a.x = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 4 * t);
            a.y = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 4 * t);
            a.z = *reinterpret_cast<const uint32_t*>(qa + 32 * kh + 16 + 4 * t);
            a.w = *reinterpret_cast<const uint32_t*>(qb + 32 * kh + 16 + 4 * t);
            sA[i] = a;
            const int hs = rt * 16 + (l >> 2) + 8 * (l & 1);
            sSF[i] = *reinterpret_cast<const uint16_t*>(p.q_exp + ((size_t)(r0 + rr) * IH + hs) * IK_SF + 2 * kh);
        }
        for (int i = tid; i < nr * IH; i += 256) sw[i / IH][i % IH] = __bfloat162float(p.w[(size_t)r0 * IH + i]);
        if constexpr (EXACT)
            for (int i = tid; i < nr * IH * 4; i += 256) ssq[i >> 7][(i >> 2) & 31][i & 3] = f_e8m0(p.q_exp[(size_t)r0 * IH * IK_SF + i]);
        __syncthreads();
        const int next = task + gridDim.x, other = stage ^ 1;
        if (!math && next < tasks)
        {
            const int tok = (next % tiles) * 128 + (wid - 4) * 32 + lane;
            const int ix = (wid - 4) * 32 + lane;
            if (tok < p.ncols)
            {
                const uint8_t* cr = p.k.codes.row(tok);
#pragma unroll
                for (int g = 0; g < 4; g++)
                {
                    const uint4 v = *reinterpret_cast<const uint4*>(cr + g * 16);
                    sKC[other][ix][4*g+0] = v.x; sKC[other][ix][4*g+1] = v.y;
                    sKC[other][ix][4*g+2] = v.z; sKC[other][ix][4*g+3] = v.w;
                }
                sKE[other][ix] = *reinterpret_cast<const uint32_t*>(p.k.sf.row(tok));
            }
            else { for (int g = 0; g < 16; g++) sKC[other][ix][g] = 0; sKE[other][ix] = 0x7F7F7F7Fu; }
        }
        if (math)
        {
            int kmax = 0;
            for (int rr = 0; rr < nr; rr++) kmax = max(kmax, p.klen[r0 + rr]);
            kmax = min(kmax, p.ncols);
            const int t = lane & 3, n = lane >> 2;
            const int c0 = base + warp * 32;
            uint32_t bw[4][4], sfb[4][2];
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
            {
                const int tok = c0 + tl * 8 + n;
                if (tok < kmax)
                {
                    const int ix = warp * 32 + tl * 8 + n;
#pragma unroll
                    for (int g = 0; g < 4; g++) bw[tl][g] = sKC[stage][ix][4*g+t];
                    const uint32_t se = sKE[stage][ix];
                    sfb[tl][0] = se & 0xFFFFu; sfb[tl][1] = se >> 16;
                }
                else { bw[tl][0] = bw[tl][1] = bw[tl][2] = bw[tl][3] = 0u; sfb[tl][0] = sfb[tl][1] = 0x7F7Fu; }
            }
        if constexpr (EXACT)
        {
            float skf[4][2][4];   // scales of keys 2t, 2t+1 of tile tl (the D columns this lane holds), group g
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
            {
                const uint32_t se = sfb[tl][0] | (sfb[tl][1] << 16);
#pragma unroll
                for (int jj = 0; jj < 2; jj++)
                {
                    const uint32_t wk = __shfl_sync(0xFFFFFFFFu, se, 4 * (2 * t + jj));
#pragma unroll
                    for (int g = 0; g < 4; g++) skf[tl][jj][g] = f_e8m0((uint8_t)(wk >> (8 * g)));
                }
            }
            float* T = sT[warp];
#pragma unroll 1
            for (int rr = 0; rr < nr; rr++)
            {
#pragma unroll
                for (int rt = 0; rt < 2; rt++)
                {
                    const uint4 a0 = sA[((rr * 2 + rt) * 2 + 0) * 32 + lane], a1 = sA[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                    float sqv[2][4], wv[2];
#pragma unroll
                    for (int hr = 0; hr < 2; hr++)
                    {
#pragma unroll
                        for (int g = 0; g < 4; g++) sqv[hr][g] = ssq[rr][rt * 16 + hr * 8 + n][g];
                        wv[hr] = sw[rr][rt * 16 + hr * 8 + n];
                    }
#pragma unroll
                    for (int tl = 0; tl < 4; tl++)
                    {
                        const float z4[4] = {0.f, 0.f, 0.f, 0.f};
                        float d[4][4];
                        mma_mxf4(d[0], make_uint4(a0.x, a0.y, 0u, 0u), bw[tl][0], bw[tl][1], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[1], make_uint4(0u, 0u, a0.z, a0.w), bw[tl][0], bw[tl][1], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[2], make_uint4(a1.x, a1.y, 0u, 0u), bw[tl][2], bw[tl][3], z4, 0x7F7Fu, 0x7F7Fu);
                        mma_mxf4(d[3], make_uint4(0u, 0u, a1.z, a1.w), bw[tl][2], bw[tl][3], z4, 0x7F7Fu, 0x7F7Fu);
#pragma unroll
                        for (int e = 0; e < 4; e++)
                        {
                            const int hr = e >> 1, jj = e & 1;
                            float dot = 0.0f;
#pragma unroll
                            for (int g = 0; g < 4; g++) dot = fmaf(d[g][e], sqv[hr][g] * skf[tl][jj][g], dot);
                            const float db = bf16r(dot);
                            T[(rt * 16 + hr * 8 + n) * MMA_TP + tl * 8 + 2 * t + jj] = bf16r((db > 0.0f ? db : 0.0f) * wv[hr]);
                        }
                    }
                }
                __syncwarp();
                float sum = 0.0f;
#pragma unroll
                for (int h = 0; h < IH; h++) sum += T[h * MMA_TP + lane];
                const int col = c0 + lane;
                if (col < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + col] = col < p.klen[r0 + rr] ? __float2bfloat16_rn(sum) : ninf;
                __syncwarp();
            }
        }
        else
        {
#pragma unroll 1
        for (int rr = 0; rr < nr; rr++)
        {
            float ps[4][2];
#pragma unroll
            for (int tl = 0; tl < 4; tl++) ps[tl][0] = ps[tl][1] = 0.0f;
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
            {
                const uint4 a0 = sA[((rr * 2 + rt) * 2 + 0) * 32 + lane], a1 = sA[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const uint32_t f0 = sSF[((rr * 2 + rt) * 2 + 0) * 32 + lane], f1 = sSF[((rr * 2 + rt) * 2 + 1) * 32 + lane];
                const float w_lo = sw[rr][rt * 16 + n], w_hi = sw[rr][rt * 16 + n + 8];
#pragma unroll
                for (int tl = 0; tl < 4; tl++)
                {
                    float acc[4] = {0.f, 0.f, 0.f, 0.f}, d1[4], d2[4];
                    mma_mxf4(d1, a0, bw[tl][0], bw[tl][1], acc, f0, sfb[tl][0]);
                    mma_mxf4(d2, a1, bw[tl][2], bw[tl][3], d1, f1, sfb[tl][1]);
#pragma unroll
                    for (int j = 0; j < 2; j++)
                    {
                        const float x0 = bf16r(d2[j]), x1 = bf16r(d2[2 + j]);
                        ps[tl][j] += bf16r((x0 > 0.0f ? x0 : 0.0f) * w_lo);
                        ps[tl][j] += bf16r((x1 > 0.0f ? x1 : 0.0f) * w_hi);
                    }
                }
            }
            const int kl = p.klen[r0 + rr];
#pragma unroll
            for (int tl = 0; tl < 4; tl++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    float v = ps[tl][j];
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
                    const int col = c0 + tl * 8 + 2 * t + j;
                    if (n == 0 && col < p.ncols) p.p[(size_t)(r0 + rr) * p.ldp + col] = col < kl ? __float2bfloat16_rn(v) : ninf;
                }
        }
        }
        }
        __syncthreads();  // current math done and next IO tile ready
        stage = other;
    }
}
#include "ds41_idx_pipe_qr.cuh"
inline void launch_logits_pipe(const LogitsP& p, cudaStream_t s, bool exact)
{
    const int tasks = ((p.R + PIPE_RB - 1) / PIPE_RB) * ((p.ncols + 127) / 128);
    const int gx = max(1, min(exact ? 48 : 144, tasks));
    if (exact) k_logits_pipe<true><<<gx, 256, 0, s>>>(p);
    else k_logits_pipe<false><<<gx, 256, 0, s>>>(p);
}

// ------------------------------------------------------------------------------------------------ candidate block scores (layer 20)
// sc[r][b] = max of p over block b's visible tokens (-inf past the last visible block); the last visible block is +inf so it is
// always chosen. nvalid[r] = number of visible blocks. Requires ldp % 8 == 0 and p 16 B aligned.
__global__ void k_block_scores(const bf16* __restrict__ p, long ldp, const int32_t* __restrict__ klen, int nb, bf16* __restrict__ sc,
                               long ldsc, int32_t* __restrict__ nvalid)
{
    const int r = blockIdx.y, b = blockIdx.x * blockDim.x + threadIdx.x;
    const int kl = klen[r], vb = min((kl + CAND_BS - 1) / CAND_BS, nb);
    if (blockIdx.x == 0 && threadIdx.x == 0) nvalid[r] = vb;
    if (b >= nb) return;
    float m = -INFINITY;
    if (b < vb)
    {
        const uint4 raw = *reinterpret_cast<const uint4*>(p + (size_t)r * ldp + (size_t)b * CAND_BS);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const float2 f = __bfloat1622float2(b2[j]);
            if (b * CAND_BS + 2 * j < kl) m = fmaxf(m, f.x);
            if (b * CAND_BS + 2 * j + 1 < kl) m = fmaxf(m, f.y);
        }
        if (b == vb - 1) m = INFINITY;
    }
    sc[(size_t)r * ldsc + b] = __float2bfloat16_rn(m);
}

// ------------------------------------------------------------------------------------------------ top-k select (sorted ascending by index)
// One CTA of 1024 per row. out[r][0..K) = the K largest v[r][0 .. min(end[r], ncap)), indices ASCENDING, -1 pad when fewer than K
// columns. Ties at the cut go to the lower index (torch's radix select emits the equal-to-kth elements in index order). Order key =
// torch's float ordering of a bf16 (-0 < +0, NaN largest). Two radix passes (high byte, low byte) find the K-th key T and the quota q
// of keys equal to T; a single in-order scan then writes every selected index at its rank, which is why the output is sorted.
__device__ __forceinline__ uint32_t okey(bf16 x)
{
    const uint32_t b = __bfloat16_as_ushort(x);
    if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
template <int K>
__global__ void __launch_bounds__(1024) k_topk(const bf16* __restrict__ v, long ldv, const int32_t* __restrict__ end, int ncap,
                                               int32_t* __restrict__ out, int ldo)
{
    __shared__ int hist[256];
    __shared__ uint32_t wsum[33];
    __shared__ int s_a, s_b;
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const bf16* vr = v + (size_t)r * ldv;
    int32_t* o = out + (size_t)r * ldo;
    const int n = max(0, min(end[r], ncap));
    if (n <= K)
    {
        for (int i = tid; i < K; i += 1024) o[i] = i < n ? i : -1;
        return;
    }
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += 1024) atomicAdd(&hist[okey(vr[i]) >> 8], 1);
    __syncthreads();
    if (tid == 0)
    {
        int cum = 0, b = 255;
        for (; b > 0; --b)
        {
            if (cum + hist[b] >= K) break;
            cum += hist[b];
        }
        s_b = b;
        s_a = cum;
    }
    __syncthreads();
    const uint32_t hi = (uint32_t)s_b;
    const int cum1 = s_a;
    __syncthreads();
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += 1024)
    {
        const uint32_t k = okey(vr[i]);
        if ((k >> 8) == hi) atomicAdd(&hist[k & 255u], 1);
    }
    __syncthreads();
    if (tid == 0)
    {
        int cum = cum1, l = 255;
        for (; l > 0; --l)
        {
            if (cum + hist[l] >= K) break;
            cum += hist[l];
        }
        s_b = (int)((hi << 8) | (uint32_t)l);
        s_a = cum;
    }
    __syncthreads();
    const uint32_t T = (uint32_t)s_b;
    const int q = K - s_a;   // equal-to-T elements taken, lowest indices first
    int gtb = 0, eqb = 0;
    for (int base = 0; base < n; base += 1024)
    {
        const int i = base + tid;
        uint32_t gt = 0, eq = 0;
        if (i < n)
        {
            const uint32_t k = okey(vr[i]);
            gt = k > T;
            eq = k == T;
        }
        const uint32_t pv = gt | (eq << 16);
        uint32_t x = pv;
        for (int d = 1; d < 32; d <<= 1)
        {
            const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) wsum[w] = x;
        __syncthreads();
        if (w == 0)
        {
            const uint32_t t = wsum[lane];
            uint32_t inc = t;
            for (int d = 1; d < 32; d <<= 1)
            {
                const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, inc, d);
                if (lane >= d) inc += y;
            }
            wsum[lane] = inc - t;
            if (lane == 31) wsum[32] = inc;
        }
        __syncthreads();
        const uint32_t ex = x - pv + wsum[w];
        const uint32_t tot = wsum[32];
        const int gex = gtb + (int)(ex & 0xFFFFu), eex = eqb + (int)(ex >> 16);
        if (gt || (eq && eex < q)) o[gex + min(eex, q)] = i;
        gtb += (int)(tot & 0xFFFFu);
        eqb += (int)(tot >> 16);
        if (gtb + min(eqb, q) >= K) break;   // block-uniform: every selected index has been written
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ multi-CTA top-k select
// Same contract and same output as k_topk (the K largest of v[r][0 .. min(end[r], ncap)), indices ascending, -1 pad, ties at the cut go to
// the lower index), for long rows: the row is split into chunks of CH elements (a multiple of 2048), one CTA of 256 per chunk, grid
// (G, R), five launches: [memset of the two histograms] -> hist of the high key byte -> hist of the low byte of the keys in the cut bin
// -> per-chunk counts of keys > T and == T -> ordered write. All sums are integer, so the result does not depend on CTA order. Every
// CTA re-derives the cut (T, quota q) from the two global 256-bin histograms (a 256-lane scan), so no extra launch is needed for it.
// Requires v 16 B aligned and ldv % 8 == 0 (the launcher falls back to k_topk otherwise). ws: ints [R][256] hist1, [R][256] hist2,
// then int2 [R][G] counts.
constexpr int MC_T = 256, MC_TILE = MC_T * 8, MC_MIN_N = 24576, MC_CTAS = 288;
__device__ __forceinline__ uint32_t okey_raw(uint32_t b)
{
    if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
// Largest bin b with sum(h[b..255]) >= need (need >= 1); above = sum(h[b+1..255]). All 256 threads of the block call it; res[0] = b,
// res[1] = above on return (visible to the block). wsm: 8 ints.
__device__ __forceinline__ void cut256(const int* __restrict__ h, int need, int* wsm, int* res)
{
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int c = h[255 - tid];
    int x = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        const int y = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) wsm[w] = x;
    __syncthreads();
    int base = 0;
    for (int i = 0; i < w; i++) base += wsm[i];
    const int P = x + base;
    if (P >= need && P - c < need)
    {
        res[0] = 255 - tid;
        res[1] = P - c;
    }
    __syncthreads();
}
// 8 consecutive elements of a row starting at i0 (a multiple of 8): okey of each, ok[j] = (i0 + j < hi)
__device__ __forceinline__ void mc_load8(const bf16* __restrict__ vr, int i0, int hi, uint32_t* key, bool* ok)
{
    if (i0 + 8 <= hi)
    {
        const uint4 u = *reinterpret_cast<const uint4*>(vr + i0);
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            key[2 * j] = okey_raw(w4[j] & 0xFFFFu);
            key[2 * j + 1] = okey_raw(w4[j] >> 16);
            ok[2 * j] = ok[2 * j + 1] = true;
        }
    }
    else
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            ok[j] = i0 + j < hi;
            key[j] = ok[j] ? okey_raw(__bfloat16_as_ushort(vr[i0 + j])) : 0u;
        }
    }
}
// shared prologue of the chunk kernels: row r, chunk g, valid length n, chunk [lo, hi)
#define MC_ROW(K_)                                                                                                         \
    const int r = blockIdx.y, g = blockIdx.x, tid = threadIdx.x, lane = tid & 31;                                          \
    const int n = max(0, min(end[r], ncap));                                                                               \
    const int lo = g * CH, hi = min(lo + CH, n);                                                                           \
    const bf16* vr = v + (size_t)r * ldv;                                                                                  \
    (void)lane; (void)vr; (void)hi; (void)lo; (void)n; (void)g; (void)r; (void)tid;

template <int K>
__global__ void __launch_bounds__(MC_T) k_tkm_h1(const bf16* __restrict__ v, long ldv, const int32_t* __restrict__ end, int ncap, int CH,
                                                 int32_t* __restrict__ ws)
{
    MC_ROW(K)
    __shared__ int hist[256];
    if (n <= K || lo >= n) return;
    hist[tid] = 0;
    __syncthreads();
    for (int base = lo; base < hi; base += MC_TILE)
    {
        uint32_t key[8];
        bool ok[8];
        mc_load8(vr, base + tid * 8, hi, key, ok);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const uint32_t bin = ok[j] ? (key[j] >> 8) : 0xFFFFFFFFu;
            const uint32_t m = __match_any_sync(0xFFFFFFFFu, bin);
            if (ok[j] && lane == __ffs(m) - 1) atomicAdd(&hist[bin], __popc(m));
        }
    }
    __syncthreads();
    if (hist[tid]) atomicAdd(&ws[(size_t)r * 256 + tid], hist[tid]);
}
template <int K>
__global__ void __launch_bounds__(MC_T) k_tkm_h2(const bf16* __restrict__ v, long ldv, const int32_t* __restrict__ end, int ncap, int CH,
                                                 int32_t* __restrict__ ws, int R)
{
    MC_ROW(K)
    __shared__ int hist[256], wsm[8], res[2];
    if (n <= K || lo >= n) return;
    cut256(ws + (size_t)r * 256, K, wsm, res);
    const uint32_t hib = (uint32_t)res[0];
    hist[tid] = 0;
    __syncthreads();
    for (int base = lo; base < hi; base += MC_TILE)
    {
        uint32_t key[8];
        bool ok[8];
        mc_load8(vr, base + tid * 8, hi, key, ok);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const bool sel = ok[j] && (key[j] >> 8) == hib;
            const uint32_t bin = sel ? (key[j] & 255u) : 0xFFFFFFFFu;
            const uint32_t m = __match_any_sync(0xFFFFFFFFu, bin);
            if (sel && lane == __ffs(m) - 1) atomicAdd(&hist[bin], __popc(m));
        }
    }
    __syncthreads();
    if (hist[tid]) atomicAdd(&ws[(size_t)(R + r) * 256 + tid], hist[tid]);
}
// T = the K-th largest key (16 bit), q = number of keys equal to T that are taken (lowest indices first)
__device__ __forceinline__ void mc_cut(const int32_t* __restrict__ ws, int R, int r, int K, int* wsm, int* res, uint32_t& T, int& q)
{
    cut256(ws + (size_t)r * 256, K, wsm, res);
    const uint32_t hib = (uint32_t)res[0];
    const int cum1 = res[1];
    cut256(ws + (size_t)(R + r) * 256, K - cum1, wsm, res);
    T = (hib << 8) | (uint32_t)res[0];
    q = K - (cum1 + res[1]);
}
template <int K>
__global__ void __launch_bounds__(MC_T) k_tkm_cnt(const bf16* __restrict__ v, long ldv, const int32_t* __restrict__ end, int ncap, int CH,
                                                  const int32_t* __restrict__ ws, int R, int G)
{
    MC_ROW(K)
    __shared__ int wsm[8], res[2], red[16];
    if (n <= K) return;
    int2* cnt = reinterpret_cast<int2*>(const_cast<int32_t*>(ws) + (size_t)R * 512) + (size_t)r * G;
    if (lo >= n)
    {
        if (tid == 0) cnt[g] = make_int2(0, 0);
        return;
    }
    uint32_t T;
    int q;
    mc_cut(ws, R, r, K, wsm, res, T, q);
    int ngt = 0, neq = 0;
    for (int base = lo; base < hi; base += MC_TILE)
    {
        uint32_t key[8];
        bool ok[8];
        mc_load8(vr, base + tid * 8, hi, key, ok);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            ngt += ok[j] && key[j] > T;
            neq += ok[j] && key[j] == T;
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
    {
        ngt += __shfl_xor_sync(0xFFFFFFFFu, ngt, o);
        neq += __shfl_xor_sync(0xFFFFFFFFu, neq, o);
    }
    if (lane == 0) { red[tid >> 5] = ngt; red[8 + (tid >> 5)] = neq; }
    __syncthreads();
    if (tid == 0)
    {
        int a = 0, b = 0;
        for (int i = 0; i < 8; i++) { a += red[i]; b += red[8 + i]; }
        cnt[g] = make_int2(a, b);
    }
}
template <int K>
__global__ void __launch_bounds__(MC_T) k_tkm_wr(const bf16* __restrict__ v, long ldv, const int32_t* __restrict__ end, int ncap, int CH,
                                                 const int32_t* __restrict__ ws, int R, int G, int32_t* __restrict__ out, int ldo)
{
    MC_ROW(K)
    __shared__ int wsm[8], res[2], red[16];
    __shared__ uint32_t wsum[9];
    int32_t* o = out + (size_t)r * ldo;
    if (n <= K)
    {
        if (g == 0)
            for (int i = tid; i < K; i += MC_T) o[i] = i < n ? i : -1;
        return;
    }
    if (lo >= n) return;
    uint32_t T;
    int q;
    mc_cut(ws, R, r, K, wsm, res, T, q);
    const int2* cnt = reinterpret_cast<const int2*>(ws + (size_t)R * 512) + (size_t)r * G;
    int a = 0, b = 0;
    for (int i = tid; i < g; i += MC_T) { const int2 c = cnt[i]; a += c.x; b += c.y; }
#pragma unroll
    for (int d = 16; d; d >>= 1)
    {
        a += __shfl_xor_sync(0xFFFFFFFFu, a, d);
        b += __shfl_xor_sync(0xFFFFFFFFu, b, d);
    }
    if (lane == 0) { red[tid >> 5] = a; red[8 + (tid >> 5)] = b; }
    __syncthreads();
    int gtb = 0, eqb = 0;
    for (int i = 0; i < 8; i++) { gtb += red[i]; eqb += red[8 + i]; }
    for (int base = lo; base < hi; base += MC_TILE)
    {
        const int i0 = base + tid * 8;
        uint32_t key[8];
        bool ok[8];
        mc_load8(vr, i0, hi, key, ok);
        uint32_t pv = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) pv += (ok[j] && key[j] > T ? 1u : 0u) + (ok[j] && key[j] == T ? 0x10000u : 0u);
        uint32_t x = pv;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
            const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) wsum[tid >> 5] = x;
        __syncthreads();
        if (tid == 0)
        {
            uint32_t run = 0;
            for (int i = 0; i < 8; i++) { const uint32_t t = wsum[i]; wsum[i] = run; run += t; }
            wsum[8] = run;
        }
        __syncthreads();
        const uint32_t ex = x - pv + wsum[tid >> 5];
        int gex = gtb + (int)(ex & 0xFFFFu), eex = eqb + (int)(ex >> 16);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            if (!ok[j]) continue;
            if (key[j] > T) { o[gex + min(eex, q)] = i0 + j; gex++; }
            else if (key[j] == T) { if (eex < q) o[gex + eex] = i0 + j; eex++; }
        }
        const uint32_t tot = wsum[8];
        gtb += (int)(tot & 0xFFFFu);
        eqb += (int)(tot >> 16);
        __syncthreads();
    }
}
// ------------------------------------------------------------------------------------------------ candidate consumers: compact lens, restore
// clen[r] = number of visible tokens in the candidate blocks (a prefix of the compact columns: only the last visible block is partial)
__global__ void k_compact_lens(const int32_t* __restrict__ cand, int ldc, const int32_t* __restrict__ klen, int32_t* __restrict__ clen)
{
    __shared__ int sh[32];
    const int r = blockIdx.x;
    int n = 0;
    for (int s = threadIdx.x; s < CAND_BLOCKS; s += blockDim.x)
    {
        const int b = cand[(size_t)r * ldc + s];
        if (b >= 0) n += min(max(klen[r] - b * CAND_BS, 0), CAND_BS);
    }
    for (int o = 16; o; o >>= 1) n += __shfl_xor_sync(0xFFFFFFFFu, n, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = n;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int t = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += sh[i];
        clen[r] = t;
    }
}
// compact column ci -> token cand[ci / 8] * 8 + ci % 8 (still ascending); -1 stays -1
__global__ void k_restore(const int32_t* __restrict__ ci, int ldci, const int32_t* __restrict__ cand, int ldc, int32_t* __restrict__ out,
                          int ldo)
{
    const int r = blockIdx.x;
    for (int j = threadIdx.x; j < TOPK; j += blockDim.x)
    {
        const int c = ci[(size_t)r * ldci + j];
        int res = -1;
        if (c >= 0 && c < CAND_COLS)
        {
            const int b = cand[(size_t)r * ldc + (c >> 3)];
            if (b >= 0) res = b * CAND_BS + (c & 7);
        }
        out[(size_t)r * ldo + j] = res;
    }
}

// ------------------------------------------------------------------------------------------------ ca row dequantizer (reader helper)
// rows row0 .. row0 + n - 1 of a ca plane -> bf16 [n][512] (fp4 * e4m3 is exact in bf16). One block of 512 per row.
__global__ void __launch_bounds__(512) k_ca_dequant(PagedRows ca, int row0, bf16* __restrict__ out)
{
    const int t = blockIdx.x, c = threadIdx.x;
    const uint8_t* row = ca.row(row0 + t);
    const uint32_t code = (row[c >> 1] >> ((c & 1) * 4)) & 0xFu;
    const uint32_t m = code & 7;
    float v = m <= 4 ? 0.5f * (float)m : (m == 5 ? 3.0f : (m == 6 ? 4.0f : 6.0f));
    if (code & 8) v = -v;
    out[(size_t)t * D + c] = __float2bfloat16_rn(v * f_e4m3(row[CA_CODES + (c >> 4)]));
}

// ------------------------------------------------------------------------------------------------ engine-step helpers (device positions)
// klen[r] = number of compressed rows visible to the query at position dpos[r] = (pos + 1) / ratio
__global__ void k_klen(const int* __restrict__ dpos, int ratio, int32_t* __restrict__ klen, int R)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R) klen[r] = (__ldcv(dpos + r) + 1) / ratio;
}

// Y[r][n] = bf16(sum_k X[r][k] * W[n][k]), fp32 accumulate (fixed order: lane-strided over K, then a xor-butterfly). X bf16 [R][ldx],
// W bf16 [N][K] (K % 256 == 0). grid (ceil(N / 8), ceil(R / 8)), 256 threads: warp = one output column, 8 rows per block row.
__global__ void __launch_bounds__(256) k_gemv_bf16(const bf16* __restrict__ X, int ldx, const bf16* __restrict__ W, int N, int K,
                                                   bf16* __restrict__ Y, int ldy, int R)
{
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int r0 = blockIdx.y * 8;
    if (n >= N) return;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; r++) acc[r] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 256)
    {
        const int k = k0 + lane * 8;
        const uint4 wr = *reinterpret_cast<const uint4*>(W + (size_t)n * K + k);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wr);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 4; j++) { const float2 f = __bfloat1622float2(w2[j]); wf[2 * j] = f.x; wf[2 * j + 1] = f.y; }
#pragma unroll
        for (int r = 0; r < 8; r++)
        {
            if (r0 + r >= R) continue;
            const uint4 xr = *reinterpret_cast<const uint4*>(X + (size_t)(r0 + r) * ldx + k);
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xr);
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                const float2 f = __bfloat1622float2(x2[j]);
                acc[r] = fmaf(f.x, wf[2 * j], acc[r]);
                acc[r] = fmaf(f.y, wf[2 * j + 1], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        float v = acc[r];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if (lane == 0 && r0 + r < R) Y[(size_t)(r0 + r) * ldy + n] = __float2bfloat16_rn(v);
    }
}

// Block-scaled fp8 GEMM for the indexer q projection: Y[M][N] = bf16 over K blocks of 32 of (exact e4m3 products in fmaf order) * (E8M0
// activation scale * E8M0 weight scale), accumulated in K order -- the arithmetic of attention A's k_gemm_mxfp8 (ds41_attn.cuh), so the
// indexer sees the same q_b rounding the attention does. A codes [M][lda], SA scale bytes [M][lsa], W codes [N][K], SW bytes [N][K/32].
struct GemmMxP
{
    const uint8_t *A, *SA, *W, *SW;
    bf16* Y;
    int M, N, K, lda, lsa, ldy;
};
constexpr int GBM = 64, GBN = 64;
__global__ void __launch_bounds__(256) k_gemm_mx(GemmMxP p)
{
    __shared__ float lut[256];
    __shared__ float As[GBM][33];
    __shared__ float Ws[GBN][33];
    __shared__ float sa[GBM], sw[GBN];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
    const int KB = p.K >> 5;
    lut[tid] = f_e4m3((uint8_t)tid);
    __syncthreads();
    const int ty = tid >> 4, tx = tid & 15;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int kb = 0; kb < KB; kb++)
    {
        {
            const int r = tid >> 2, c8 = (tid & 3) * 8;
            uint2 ua = make_uint2(0, 0), uw = make_uint2(0, 0);
            if (m0 + r < p.M) ua = *(const uint2*)(p.A + (size_t)(m0 + r) * p.lda + kb * 32 + c8);
            if (n0 + r < p.N) uw = *(const uint2*)(p.W + (size_t)(n0 + r) * p.K + kb * 32 + c8);
            const uint8_t* pa = (const uint8_t*)&ua;
            const uint8_t* pw = (const uint8_t*)&uw;
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                As[r][c8 + i] = lut[pa[i]];
                Ws[r][c8 + i] = lut[pw[i]];
            }
        }
        if (tid < GBM) sa[tid] = (m0 + tid < p.M) ? f_e8m0(p.SA[(size_t)(m0 + tid) * p.lsa + kb]) : 0.0f;
        else if (tid < GBM + GBN)
        {
            const int n = n0 + tid - GBM;
            sw[tid - GBM] = (n < p.N) ? f_e8m0(p.SW[(size_t)n * KB + kb]) : 0.0f;
        }
        __syncthreads();
        float part[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) part[i][j] = 0.0f;
#pragma unroll 8
        for (int k = 0; k < 32; k++)
        {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[ty * 4 + i][k];
#pragma unroll
            for (int j = 0; j < 4; j++) w[j] = Ws[tx * 4 + j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = fmaf(a[i], w[j], part[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = fmaf(part[i][j], sa[ty * 4 + i] * sw[tx * 4 + j], acc[i][j]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        const int m = m0 + ty * 4 + i;
        if (m >= p.M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int n = n0 + tx * 4 + j;
            if (n < p.N) p.Y[(size_t)m * p.ldy + n] = __float2bfloat16_rn(acc[i][j]);
        }
    }
}

// ================================================================================================ host launchers
inline void launch_ca_row_r1(const CaR1P& p, int T, cudaStream_t s) { k_ca_row_r1<<<T, 512, 0, s>>>(p); }
inline void launch_ica_row(const IcaP& p, int T, cudaStream_t s) { k_ica_row<<<T, 256, 0, s>>>(p); }
inline void launch_index_q(const IdxQP& p, int R, cudaStream_t s) { k_index_q<<<dim3(R, IH), 128, 0, s>>>(p); }
inline void launch_index_q_cached(const IdxQP& p, int R, cudaStream_t s)
{
    k_index_q_trig<<<R, 32, 0, s>>>(p);
    k_index_q_cached<<<dim3(R, IH), 128, 0, s>>>(p);
}
inline void launch_logits_dense(const LogitsP& p, cudaStream_t s)
{
    constexpr int RB = 8;
    k_logits<RB, false><<<dim3((p.ncols + 127) / 128, (p.R + RB - 1) / RB), 128, 0, s>>>(p);
}
inline int sm_count()
{
    static int cache[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 48;
    if (!cache[dev]) cudaDeviceGetAttribute(&cache[dev], cudaDevAttrMultiProcessorCount, dev);
    return cache[dev] > 0 ? cache[dev] : 48;
}
// exact = true: bit-identical to launch_logits_dense on every input; false: the faster variant with its own fp32 rounding (see k_logits_mma)
inline void launch_logits_dense_mma(const LogitsP& p, cudaStream_t s, bool exact = true)
{
    const int units = (p.ncols + 31) / 32;
    const int gx = max(1, min((units + 3) / 4, sm_count() * 6));
    const dim3 grid(gx, (p.R + MMA_RB - 1) / MMA_RB);
    if (exact) k_logits_mma<true><<<grid, 128, 0, s>>>(p);
    else k_logits_mma<false><<<grid, 128, 0, s>>>(p);
}
inline void launch_logits_gather(const LogitsP& p, cudaStream_t s)
{
    k_logits<1, true><<<dim3((p.ncols + 127) / 128, p.R), 128, 0, s>>>(p);
}
inline void launch_block_scores(const bf16* p, long ldp, const int32_t* klen, int nb, bf16* sc, long ldsc, int32_t* nvalid, int R,
                                cudaStream_t s)
{
    k_block_scores<<<dim3((nb + 255) / 256, R), 256, 0, s>>>(p, ldp, klen, nb, sc, ldsc, nvalid);
}
template <int K>
inline void launch_topk(const bf16* v, long ldv, const int32_t* end, int ncap, int32_t* out, int ldo, int R, cudaStream_t s)
{
    k_topk<K><<<R, 1024, 0, s>>>(v, ldv, end, ncap, out, ldo);
}
// ints of workspace for R rows over rows of at most max_rows elements
inline size_t topk_mc_ws_ints(int R, long max_rows) { return (size_t)R * (512 + 2 * ((max_rows + MC_TILE - 1) / MC_TILE)); }
// ctas = target total CTAs of the grid (R rows x G chunks); the chunk size follows from ncap (host bound of end[]). Falls back to
// the single-CTA kernel for short rows or unaligned input.
template <int K>
inline void launch_topk_mc(const bf16* v, long ldv, const int32_t* end, int ncap, int32_t* out, int ldo, int R, int32_t* ws, int ctas,
                           cudaStream_t s)
{
    if (ncap < MC_MIN_N || (ldv & 7) || (reinterpret_cast<uintptr_t>(v) & 15) || ws == nullptr)
    {
        launch_topk<K>(v, ldv, end, ncap, out, ldo, R, s);
        return;
    }
    const int gper = max(1, ctas / R);
    const int CH = MC_TILE * ((ncap + MC_TILE * gper - 1) / (MC_TILE * gper));
    const int G = (ncap + CH - 1) / CH;
    cudaMemsetAsync(ws, 0, (size_t)R * 512 * sizeof(int32_t), s);
    const dim3 grid(G, R);
    k_tkm_h1<K><<<grid, MC_T, 0, s>>>(v, ldv, end, ncap, CH, ws);
    k_tkm_h2<K><<<grid, MC_T, 0, s>>>(v, ldv, end, ncap, CH, ws, R);
    k_tkm_cnt<K><<<grid, MC_T, 0, s>>>(v, ldv, end, ncap, CH, ws, R, G);
    k_tkm_wr<K><<<grid, MC_T, 0, s>>>(v, ldv, end, ncap, CH, ws, R, G, out, ldo);
}

inline void launch_compact_lens(const int32_t* cand, int ldc, const int32_t* klen, int32_t* clen, int R, cudaStream_t s)
{
    k_compact_lens<<<R, 256, 0, s>>>(cand, ldc, klen, clen);
}
inline void launch_restore(const int32_t* ci, int ldci, const int32_t* cand, int ldc, int32_t* out, int ldo, int R, cudaStream_t s)
{
    k_restore<<<R, 256, 0, s>>>(ci, ldci, cand, ldc, out, ldo);
}
inline void launch_klen(const int* dpos, int ratio, int32_t* klen, int R, cudaStream_t s) { k_klen<<<(R + 63) / 64, 64, 0, s>>>(dpos, ratio, klen, R); }
inline void launch_gemv_bf16(const bf16* X, int ldx, const bf16* W, int N, int K, bf16* Y, int ldy, int R, cudaStream_t s)
{
    k_gemv_bf16<<<dim3((N + 7) / 8, (R + 7) / 8), 256, 0, s>>>(X, ldx, W, N, K, Y, ldy, R);
}
inline void launch_gemm_mx(const GemmMxP& p, cudaStream_t s)
{
    k_gemm_mx<<<dim3((p.N + GBN - 1) / GBN, (p.M + GBM - 1) / GBM), 256, 0, s>>>(p);
}
inline void launch_ca_dequant(const PagedRows& ca, int row0, int n, bf16* out, cudaStream_t s) { k_ca_dequant<<<n, 512, 0, s>>>(ca, row0, out); }

// ================================================================================================ stages (what the step driver calls)
// Scratch for R query rows over at most max_rows compressed rows (decode: R <= 8; the step driver carves it once per session).
struct Scratch
{
    uint8_t *q_pack, *q_exp;   // [R][32][64], [R][32][4]
    bf16* w;                   // [R][32]
    bf16* p;                   // [R][ldp] logits, ldp = max_rows rounded up to a multiple of 128
    long ldp;
    bf16* sc;                  // [R][ldsc] block scores, ldsc = ldp / 8
    long ldsc;
    int32_t* nvalid;           // [R]
    int32_t* cand;             // [R][2048] candidate block ids
    bf16* cp;                  // [R][16384] compact logits
    int32_t* clen;             // [R]
    int32_t* ci;               // [R][512] compact top-k before restore
    int32_t* tkw;              // multi-CTA top-k workspace, topk_mc_ws_ints(R, ldp) ints
    float2* trig;              // [R][32] row-shared cosine/sine, default-off
    static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
    static long ld_of(int max_rows) { return ((long)max_rows + 127) / 128 * 128; }
    static size_t bytes(int R, int max_rows)
    {
        const long ld = ld_of(max_rows);
        return al((size_t)R * IH * IK_CODES) + al((size_t)R * IH * IK_SF) + al((size_t)R * IH * 2) + al((size_t)R * ld * 2) +
               al((size_t)R * (ld / 8) * 2) + al((size_t)R * 4) + al((size_t)R * CAND_BLOCKS * 4) + al((size_t)R * CAND_COLS * 2) +
               al((size_t)R * 4) + al((size_t)R * TOPK * 4) + al(topk_mc_ws_ints(R, ld) * 4) + al((size_t)R * 32 * sizeof(float2));
    }
    static Scratch carve(void* base, int R, int max_rows)
    {
        Scratch s;
        uint8_t* b = (uint8_t*)base;
        auto take = [&](size_t n) { uint8_t* r = b; b += al(n); return r; };
        s.ldp = ld_of(max_rows);
        s.ldsc = s.ldp / 8;
        s.q_pack = take((size_t)R * IH * IK_CODES);
        s.q_exp = take((size_t)R * IH * IK_SF);
        s.w = (bf16*)take((size_t)R * IH * 2);
        s.p = (bf16*)take((size_t)R * s.ldp * 2);
        s.sc = (bf16*)take((size_t)R * s.ldsc * 2);
        s.nvalid = (int32_t*)take((size_t)R * 4);
        s.cand = (int32_t*)take((size_t)R * CAND_BLOCKS * 4);
        s.cp = (bf16*)take((size_t)R * CAND_COLS * 2);
        s.clen = (int32_t*)take((size_t)R * 4);
        s.ci = (int32_t*)take((size_t)R * TOPK * 4);
        s.tkw = (int32_t*)take(topk_mc_ws_ints(R, s.ldp) * 4);
        s.trig = (float2*)take((size_t)R * 32 * sizeof(float2));
        return s;
    }
};

// Inputs of an index stage. klen is a device array [R]; max_klen the host-side bound used to size the grid.
struct IndexIn
{
    const bf16* q_b;       // [R][4096] indexer.wq_b output
    const bf16* wproj;     // [R][32] indexer.weights_proj output
    RopeInv inv;           // this layer's rope table (the same for all layers 2..39)
    int pos0, R;
    const int32_t* klen;   // device [R]
    int max_klen;          // host: max over rows of klen
    IcaMap k;              // index-K rows of the index-K source layer
    const int* dpos = nullptr;   // optional device positions [R] (engine step): the query rope position of row r
    int logits_mma = 0;          // dense logits: 0 = the dp4a reference kernel, 1 = tensor cores bit-identical to it, 2 = tensor cores, fast variant
    bool pipe_b = false;         // served opt-in; never enables the exact slow arm A
    bool pipe_qr = false;        // default-off 16-row Q-stationary cp.async dense logits
    bool q_trig = false;         // default-off once-per-row query rope trigonometry
    int topk_mc = 1;             // 1: multi-CTA top-k for long rows, 0: the single-CTA kernel
};

// ST_INDEX_TOPK at an index source (dense, layers 2, 8, 14, 20; also 24.. when the candidate stage is off): q prep, logits over all
// visible rows, top-512. out [R][ldo >= 512] int32 ascending, -1 tail. Leaves the logits in sc.p (the candidate stage reads them at 20).
inline void stage_index_dense(const IndexIn& in, const Scratch& sc, int32_t* out, int ldo, cudaStream_t s)
{
    IdxQP q{in.q_b, in.wproj, in.inv, in.pos0, sc.q_pack, sc.q_exp, sc.w, in.dpos, sc.trig};
    if (in.q_trig) launch_index_q_cached(q, in.R, s);
    else launch_index_q(q, in.R, s);
    LogitsP l{};
    l.q_pack = sc.q_pack; l.q_exp = sc.q_exp; l.w = sc.w; l.k = in.k; l.klen = in.klen;
    l.p = sc.p; l.ldp = sc.ldp; l.ncols = in.max_klen; l.R = in.R;
    if (in.pipe_qr) launch_logits_pipe_qr(l, s);
    else if (in.pipe_b) launch_logits_pipe(l, s, false);
    else if (in.logits_mma) launch_logits_dense_mma(l, s, in.logits_mma == 1);
    else launch_logits_dense(l, s);
    const int ncap = max(1, min(in.max_klen, (int)sc.ldp));
    if (in.topk_mc) launch_topk_mc<TOPK>(sc.p, sc.ldp, in.klen, ncap, out, ldo, in.R, sc.tkw, MC_CTAS, s);
    else launch_topk<TOPK>(sc.p, sc.ldp, in.klen, (int)sc.ldp, out, ldo, in.R, s);
}
// ST_CAND_BLOCKS (layer 20, after stage_index_dense): block scores from the dense logits, top-2048 blocks -> sc.cand
inline void stage_cand_blocks(const IndexIn& in, const Scratch& sc, cudaStream_t s)
{
    const int nb = (int)sc.ldsc;
    launch_block_scores(sc.p, sc.ldp, in.klen, nb, sc.sc, sc.ldsc, sc.nvalid, in.R, s);
    if (in.topk_mc)
        launch_topk_mc<CAND_BLOCKS>(sc.sc, sc.ldsc, sc.nvalid, max(1, min(nb, (in.max_klen + CAND_BS - 1) / CAND_BS)), sc.cand, CAND_BLOCKS, in.R,
                                    sc.tkw, MC_CTAS, s);
    else launch_topk<CAND_BLOCKS>(sc.sc, sc.ldsc, sc.nvalid, nb, sc.cand, CAND_BLOCKS, in.R, s);
}
// ST_INDEX_TOPK at a candidate consumer (layers 24, 28, 32, 36): logits over the candidate tokens only, top-512 over their prefix,
// restored to token ids. cand: the layer-20 candidates [R][2048] (sc.cand of that stage).
inline void stage_index_cand(const IndexIn& in, const Scratch& sc, const int32_t* cand, int32_t* out, int ldo, cudaStream_t s)
{
    IdxQP q{in.q_b, in.wproj, in.inv, in.pos0, sc.q_pack, sc.q_exp, sc.w, in.dpos, sc.trig};
    if (in.q_trig) launch_index_q_cached(q, in.R, s);
    else launch_index_q(q, in.R, s);
    LogitsP l{};
    l.q_pack = sc.q_pack; l.q_exp = sc.q_exp; l.w = sc.w; l.k = in.k; l.klen = in.klen;
    l.cand = cand; l.ldc = CAND_BLOCKS; l.p = sc.cp; l.ldp = CAND_COLS; l.ncols = CAND_COLS; l.R = in.R;
    launch_logits_gather(l, s);
    launch_compact_lens(cand, CAND_BLOCKS, in.klen, sc.clen, in.R, s);
    launch_topk<TOPK>(sc.cp, CAND_COLS, sc.clen, CAND_COLS, sc.ci, TOPK, in.R, s);
    launch_restore(sc.ci, TOPK, cand, CAND_BLOCKS, out, ldo, in.R, s);
}
}   // namespace ds41i
