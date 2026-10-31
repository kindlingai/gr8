// DeepSeek-V4.1-Flash-1017 attention A: the plain types (no kernels) shared by the attention TU (ds41_attn.cu, kernels in
// ds41_attn.cuh) and the step-driver glue (ds41_attn_ops.cuh), so the glue sees the structs without compiling the kernels twice.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

namespace ds41a
{
typedef __nv_bfloat16 bf16;

constexpr int HID = 5120, QA = 1280, HD = 512, ROPE = 64, NOPE = 448, WIN = 128, OLORA = 1024, HPG = 8;   // HPG: heads per o group
constexpr int GROUP_IN = HPG * HD;                      // 4096, one o group's input width
constexpr float EPS = 1e-20f;
constexpr float INV448 = 0.002232142857142857f;         // fp32(1/448), the activation scale's constant
constexpr float SOFTMAX_SCALE = 0.04419417382415922f;   // 512^-0.5 (fp32 below)

// gv: the matrix as MXFP8 GEMV records [N/16][K/32][528] (the pack's .mx.gv; N, K are one group's, groups are contiguous), or null
struct Fp8Mat { const uint8_t* w; const uint8_t* s; int N, K; const uint8_t* gv = nullptr; };
// Decode GEMV launcher hooks (ds41_attn_gv.cuh defines them; the attention TU cannot include densev.cuh, see ds41_attn.cuh g_fn_xfrag).
// xfrag: fragment table of `groups` activation blocks; gemv: y = records x table, tpc / ks already resolved (0 = pick).
typedef void (*XfragFn)(const uint8_t* codes, int ldc, const uint8_t* sc, int lds, int M, int K, int groups, size_t gc, size_t gs, uint32_t* xf,
                        cudaStream_t st);
typedef void (*GemvFn)(const uint32_t* xf, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups, size_t y_gs, cudaStream_t st, int tpc, int ks);
struct Weights
{
    Fp8Mat wq_a, wq_b, wkv, wo_a, wo_b;           // wq_b [heads*512][1280]; wo_a N = 1024, K = 4096 (one group; the pointers cover all groups); wo_b [5120][groups*1024]
    const bf16 *q_norm, *kv_norm;
    const float* sink;                            // [heads]
    const float* inv;                             // device [32]: this layer's rope table (theta 10000 plain, or 160000 + YaRN)
    const bf16 *c_wkv, *c_wgate, *c_norm;         // compressor of a ratio-2 KV-source layer (2, 8, 14); null elsewhere
    Fp8Mat c_wkv8, c_wgate8;                     // optional bind-time E4M3/UE8M0 prepack; null when GR8_DS_COMP_FP8 is off
};
struct Cache
{
    bf16* ring;        // [ring_rows][512] this layer's window rows
    int ring_rows = WIN;   // >= 128; 128 + (verify chunk rows - 1) when rows are written before acceptance
    bf16* comp;        // [R][512] compressed rows: written by a source layer, read (source layer's cache) by every compressed layer
    float *kvs, *scs;  // [2][512] compressor state, source layers only
    // SESS > 1 (Scratch::sess set): one slab per S2 slot. ring: slot k at ring + k * ring_rows * 512; kvs / scs: + k * 2 * 512;
    size_t comp_ss = 0;   // comp: + k * comp_ss elements (this layer's own rows, comp_rows * 512)
    size_t src_ss = 0;    // the comp_src this layer's attention core reads: + k * src_ss (its source's comp_ss, or the indexer's shadow stride)
};
// Scratch for M rows (carve one allocation of Scratch::bytes(M, heads)).
struct Scratch
{
    uint8_t *yq, *ys, *qrq, *qrs, *oq, *os, *oaq, *oas;
    bf16 *qa, *qr, *q, *kv0, *kvrows, *o, *oa, *cnorm;   // cnorm [M][512]: post-norm pre-rope compressed rows of this call (one session: rows 0..(M+1)/2-1)
    float *ckv, *csc;
    uint32_t *xf_y, *xf_qr, *xf_o, *xf_oa;   // decode GEMV: fp16 B-fragment tables (8 tokens) of the dequantized y / qr / o / oa activations
    float* attn_split = nullptr;  // optional, shared W-prefill scratch: [row][head][2 partitions][max,sum,512 partial output]
    // The call's sessions (SESS > 1; set by the op before each call, not by carve). sess = nullptr: one session, slot 0, row r at position
    // start + r. sess != nullptr: row r is at position dstart[r] of slot sess[r]; ns sessions of n / ns rows each, contiguous
    // (session j = rows j*(n/ns) ..), the rows of one session at consecutive positions.
    const int* sess = nullptr;
    int ns = 1;
    static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
    static size_t split_bytes(int M, int heads) { return al((size_t)M * heads * 2 * (HD + 2) * sizeof(float)); }
    static size_t bytes(int M, int heads)
    {
        const int G = heads / HPG;
        return al((size_t)M * HID) + al((size_t)M * HID / 32) + al((size_t)M * QA) + al((size_t)M * QA / 32) + al((size_t)M * heads * HD) +
               al((size_t)M * heads * HD / 32) + al((size_t)M * G * OLORA) + al((size_t)M * G * OLORA / 32) + 3 * al((size_t)M * QA * 2) +
               2 * al((size_t)M * HD * 2) + 2 * al((size_t)M * heads * HD * 2) + al((size_t)M * G * OLORA * 2) + 2 * al((size_t)M * HD * 4) + al((size_t)M * HD * 2) +
               al((size_t)HID * 16) + al((size_t)QA * 16) + al((size_t)heads * HD * 16) + al((size_t)G * OLORA * 16);
    }
    void carve(void* base, int M, int heads)
    {
        uint8_t* p = (uint8_t*)base;
        const int G = heads / HPG;
        auto take = [&](size_t b) { uint8_t* r = p; p += al(b); return r; };
        yq = take((size_t)M * HID);
        ys = take((size_t)M * HID / 32);
        qrq = take((size_t)M * QA);
        qrs = take((size_t)M * QA / 32);
        oq = take((size_t)M * heads * HD);
        os = take((size_t)M * heads * HD / 32);
        oaq = take((size_t)M * G * OLORA);
        oas = take((size_t)M * G * OLORA / 32);
        qa = (bf16*)take((size_t)M * QA * 2);
        qr = (bf16*)take((size_t)M * QA * 2);
        q = (bf16*)take((size_t)M * heads * HD * 2);
        kv0 = (bf16*)take((size_t)M * HD * 2);
        kvrows = (bf16*)take((size_t)M * HD * 2);
        o = (bf16*)take((size_t)M * heads * HD * 2);
        oa = (bf16*)take((size_t)M * G * OLORA * 2);
        ckv = (float*)take((size_t)M * HD * 4);
        csc = (float*)take((size_t)M * HD * 4);
        cnorm = (bf16*)take((size_t)M * HD * 2);
        xf_y = (uint32_t*)take((size_t)HID * 16);
        xf_qr = (uint32_t*)take((size_t)QA * 16);
        xf_o = (uint32_t*)take((size_t)heads * HD * 16);
        xf_oa = (uint32_t*)take((size_t)G * OLORA * 16);
    }
};

}   // namespace ds41a
