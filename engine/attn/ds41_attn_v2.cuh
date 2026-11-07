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
    if (g_comp_fp8 && n >= 9)
    {
        if (!W.c_wkv8.w || !W.c_wkv8.s || !W.c_wgate8.w || !W.c_wgate8.s)
        {
            fprintf(stderr, "DS41 compressor FP8 prepack RED: missing codes/scales\n");
            abort();
        }
        // with an MXFP8 y form the caller pre-dequantized the bf16 y rows, so this quantization sees the same bytes the
        // full-row path's comp_row sees after its gather-side dequant
        launch_act_quant(y, HID, o.yq, HID, o.ys, HID / 32, n, HID, st);
        launch_comp_fp8(o.yq, o.ys, W.c_wkv8, o.ckv + (size_t)row0 * HD, n, st);
        launch_comp_fp8(o.yq, o.ys, W.c_wgate8, o.csc + (size_t)row0 * HD, n, st);
    }
    else if (!(g_src_fast & 1) || !launch_f32gemv2(y, W.c_wkv, W.c_wgate, o.ckv + (size_t)row0 * HD, o.csc + (size_t)row0 * HD, n, HD, HID, st))
        k_f32gemm2<<<dim3(HD / FB_N, (n + FB_M - 1) / FB_M, 2), 256, 0, st>>>(y, W.c_wkv, W.c_wgate, o.ckv + (size_t)row0 * HD,
                                                                           o.csc + (size_t)row0 * HD, n, HD, HID);
}
// gr8 PR2: attn_proj minus the y-wide GEMMs (their outputs are the gathered InputBuf qa / kv0 rows): the per-row remainder on
// every rank. Verbatim from attn_proj; this path always runs at start = 0.
inline void attn_proj_rest(const Weights& W, Cache& C, const Scratch& S, const InputBuf& o, int n, int heads, cudaStream_t st,
                           const int* dstart)
{
    k_rmsnorm<<<n, 256, 0, st>>>(o.qa, QA, W.q_norm, S.qr, QA, QA, EPS);
    launch_act_quant(S.qr, QA, S.qrq, QA, S.qrs, QA / 32, n, QA, st);
    proj_mm(W.wq_b, S.qrq, QA, S.qrs, QA / 32, S.xf_qr, false, S.q, heads * HD, n, 1, 0, 0, 0, st);
    k_rope_tail<<<(unsigned)(((long)n * heads * 32 + 255) / 256), 256, 0, st>>>(S.q, (long)heads * HD, n, heads, 0, dstart, W.inv, 0, S.sess);
    k_kv_prep<<<n, 512, 0, st>>>(o.kv0, W.kv_norm, W.inv, 0, dstart, n, S.kvrows, C.ring, C.ring_rows, S.sess);
}
// gr8 PR2: comp_row minus its two GEMMs (the gathered InputBuf ckv / csc rows): the pooling / state part is per-row and runs on
// every rank unchanged. Verbatim from comp_row.
inline void comp_row_rest(const Weights& W, Cache& C, const Scratch& S, const InputBuf& o, int n, int start, cudaStream_t st,
                          const int* dstart)
{
    if (dstart)
    {
        const int ns = S.sess ? S.ns : 1;   // sessions of n / ns rows each (the op checks n % ns == 0)
        k_comp_pool_dev<<<dim3((n / ns + 1) / 2, ns), 512, 0, st>>>(o.ckv, o.csc, n, dstart, W.c_norm, W.inv, C.comp, C.kvs, C.scs, S.cnorm,
                                                                  S.sess, C.comp_ss, C.compq);
        k_comp_state<<<ns, 512, 0, st>>>(o.ckv, o.csc, n, dstart, C.kvs, C.scs, S.sess);
        return;
    }
    if (start == 0)
    {
        k_comp_pool<<<(n + 1) / 2, 512, 0, st>>>(o.ckv, o.csc, n, 0, W.c_norm, W.inv, C.comp, C.kvs, C.scs, nullptr, S.cnorm, C.compq);
        return;
    }
    int done = 0;
    for (int r = 0; r < n; r++)
    {
        const int pos = start + r;
        k_comp_pool<<<1, 512, 0, st>>>(o.ckv + (size_t)r * HD, o.csc + (size_t)r * HD, 1, pos, W.c_norm, W.inv, C.comp, C.kvs, C.scs,
                                       nullptr, S.cnorm + (size_t)done * HD, C.compq);
        if (pos & 1) done++;
    }
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
// This is the unchanged first half of gr8 attn_out; the second half runs after
// the row-owner all-to-all rather than producing a full-M hidden partial.
inline void output_a(const Weights& W, const Scratch& S, int rows, int heads, cudaStream_t st, const int* pos)
{
    const int G = heads / HPG;
    if (!rows) return;
    k_o_prep<<<rows * heads, 512, 0, st>>>(S.o, heads, 0, pos, W.inv, nullptr, S.oq, S.os, S.sess);
    proj_mm(W.wo_a, S.oq, heads * HD, S.os, heads * HD / 32, S.xf_o, false, S.oa, G * OLORA, rows, G,
            GROUP_IN, GROUP_IN / 32, OLORA, st);
    launch_act_quant(S.oa, G * OLORA, S.oaq, G * OLORA, S.oas, G * OLORA / 32, rows, G * OLORA, st);
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
