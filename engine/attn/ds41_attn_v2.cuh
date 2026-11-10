// V2 attention kernels: split projection from gr8 PR2, compact FP8 wire, and split
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

// Per-row record: K E4M3 codes followed by K/32 E8M0 scales. A rank owns
// a contiguous row interval; padded suffix records are zero-valued, never cached.
__global__ void k_pack_input(InputBuf in, uint8_t* wire, int owned, int q, int mode)
{
    const int K = input_width(mode), stride = input_stride(mode);
    const int lane = threadIdx.x & 31;
    const size_t b = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int r = b / (K / 32), c = (b % (K / 32)) * 32 + lane;
    if (r >= q) return;
    float v = 0;
    if (r < owned) {
        if (c < QA) v = __bfloat162float(in.qa[size_t(r) * QA + c]);
        else if (c < QA + HD) v = __bfloat162float(in.kv0[size_t(r) * HD + c - QA]);
        else if (mode == 1) v = __bfloat162float(in.ckv1[size_t(r) * HD + c - QA - HD]);
        else if (c < QA + 2 * HD) v = in.ckv[size_t(r) * HD + c - QA - HD];
        else v = in.csc[size_t(r) * HD + c - QA - 2 * HD];
    }
    uint8_t code, exponent;
    act_quant_lane(v, code, exponent);
    wire[size_t(r) * stride + c] = code;
    if (!lane) wire[size_t(r) * stride + K + c / 32] = exponent;
}
__global__ void k_unpack_input(const uint8_t* wire, InputBuf out, int rows, int mode)
{
    const int K = input_width(mode), stride = input_stride(mode);
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= size_t(rows) * K) return;
    const int r = x / K, c = x % K;
    const uint8_t* row = wire + size_t(r) * stride;
    const float v = f_e4m3(row[c]) * f_e8m0(row[K + c / 32]);
    if (c < QA) out.qa[size_t(r) * QA + c] = __float2bfloat16_rn(v);
    else if (c < QA + HD) out.kv0[size_t(r) * HD + c - QA] = __float2bfloat16_rn(v);
    else if (mode == 1) out.ckv1[size_t(r) * HD + c - QA - HD] = __float2bfloat16_rn(v);
    else if (c < QA + 2 * HD) out.ckv[size_t(r) * HD + c - QA - HD] = v;
    else out.csc[size_t(r) * HD + c - QA - 2 * HD] = v;
}
inline void pack_input(const InputBuf& in, uint8_t* wire, int owned, int q, int mode, cudaStream_t st)
{
    // Initialize alignment bytes too: the fixed-size collective reads the full records.
    cudaMemsetAsync(wire, 0, size_t(q) * input_stride(mode), st);
    const size_t warps = size_t(q) * input_width(mode) / 32;
    if (warps) k_pack_input<<<(warps + 7) / 8, 256, 0, st>>>(in, wire, owned, q, mode);
}
inline void unpack_input(const uint8_t* wire, const InputBuf& out, int rows, int mode, cudaStream_t st)
{
    if (rows) k_unpack_input<<<(size_t(rows) * input_width(mode) + 255) / 256, 256, 0, st>>>(wire, out, rows, mode);
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
