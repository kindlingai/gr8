// V2 attention kernels: split projection from gr8 PR2, lossless input staging, and split
// wo_a/wo_b. Included only by the attention TU and the local device test.
#pragma once
#include "ds41_attn.cuh"
#include "ds41_attn_v2_types.h"
namespace ds41a { namespace v2 {
inline void attn_proj_own(const Weights& W, InputBuf& o, const bf16* y, const uint8_t* codes, const uint8_t* scales, int row0,
                          int n, bool pair, cudaStream_t st)
{
    const uint8_t* yq = codes ? codes : o.yq;
    const uint8_t* ys = scales ? scales : o.ys;
    if (!codes) launch_act_quant(y, HID, o.yq, HID, o.ys, HID / 32, n, HID, st);
    proj_mm(W.wq_a, yq, HID, ys, HID / 32, o.xf, false, o.qa + (size_t)row0 * QA, QA, n, 1, 0, 0, 0, st);
    proj_mm(W.wkv, yq, HID, ys, HID / 32, o.xf, use_gv(W.wq_a, n), o.kv0 + (size_t)row0 * HD, HD, n, 1, 0, 0, 0, st);   // wq_a built y's table
    if (!pair) return;
    // with an MXFP8 y form the caller pre-dequantized the bf16 y rows, so comp_row_mm's quantization sees the same bytes the
    // full-row path's comp_row sees after its gather-side dequant
    comp_row_mm(W, y, o.yq, o.ys, o.ckv + (size_t)row0 * HD, o.csc + (size_t)row0 * HD, n, st);
}
// attn_proj minus the y-wide GEMMs (their outputs are the gathered InputBuf qa / kv0 rows): the per-row remainder on every rank,
// through attn_proj's own halves. This path always runs at start = 0.
inline void attn_proj_rest(const Weights& W, Cache& C, const Scratch& S, const InputBuf& o, int n, int heads, cudaStream_t st,
                           const int* dstart)
{
    attn_proj_q(W, S, o.qa, n, 0, heads, st, dstart);
    attn_proj_kv(W, C, S, o.kv0, n, 0, st, dstart);
}
// comp_row minus its two GEMMs (the gathered InputBuf ckv / csc rows): comp_row's pooling half, per row, on every rank.
inline void comp_row_rest(const Weights& W, Cache& C, const Scratch& S, const InputBuf& o, int n, int start, cudaStream_t st,
                          const int* dstart)
{
    comp_row_pool(W, C, S, o.ckv, o.csc, n, start, st, dstart);
}

// attn_out's first half (attn_out_a); the second half runs after the row-owner all-to-all rather than producing a full-M
// hidden partial.
inline void output_a(const Weights& W, const Scratch& S, int rows, int heads, cudaStream_t st, const int* pos)
{
    if (!rows) return;
    attn_out_a(W, S, rows, 0, heads, st, pos);
}
__global__ void k_pack_output(const uint8_t* codes, const uint8_t* scales, uint8_t* wire, int rows, int padded, int K)
{
    const int stride = K * 33 / 32;
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= size_t(padded) * stride) return;
    const int r = x / stride, c = x % stride;
    wire[x] = r >= rows ? 0 : c < K ? codes[size_t(r) * K + c] : scales[size_t(r) * (K / 32) + c - K];
}
inline void pack_output(const Scratch& S, uint8_t* wire, int rows, int padded, int heads, cudaStream_t st)
{
    if (padded) k_pack_output<<<(size_t(padded) * output_stride(heads) + 255) / 256, 256, 0, st>>>(S.oaq, S.oas, wire, rows, padded, output_width(heads));
}
inline void output_b(const Fp8Mat& W, const uint8_t* wire, bf16* out, int rows, uint32_t* xf, cudaStream_t st)
{
    if (!rows) return;
    const int stride = W.K * 33 / 32;
    proj_mm(W, wire, stride, wire + W.K, stride, xf, false, out, W.N, rows, 1, 0, 0, 0, st);
}
inline void output_b_split(const Fp8Mat& W, const uint8_t* codes, const uint8_t* scales,
                           bf16* out, int rows, uint32_t* xf, cudaStream_t st) {
    if (!rows) return;
    proj_mm(W, codes, W.K, scales, W.K / 32, xf, false, out, W.N, rows, 1, 0, 0, 0, st);
}
__global__ void k_sum_output(const bf16* parts, bf16* out, int rows, int q, int nr)
{
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= size_t(rows) * HID) return;
    float sum = 0;
    for (int p = 0; p < nr; ++p) sum += __bfloat162float(parts[size_t(p) * q * HID + x]);
    out[x] = __float2bfloat16_rn(sum);
}
inline void sum_output(const bf16* parts, bf16* out, int rows, int q, int nr, cudaStream_t st)
{
    if (rows) k_sum_output<<<(size_t(rows) * HID + 255) / 256, 256, 0, st>>>(parts, out, rows, q, nr);
}
}} // namespace ds41a::v2
