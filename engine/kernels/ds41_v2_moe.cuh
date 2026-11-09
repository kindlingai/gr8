// Sequence-parallel prefill FFN layout adapters. New gr8 code; no external kernel source.
#pragma once
#include "ds_prefill_ro.cuh"
#include "ds_moe_tc.cuh"

namespace ds41v2moe
{
// Packed six-wide wire routes, versus the stock plan's eight-wide rows.
// Padding is explicit even on ranks that own no live rows.
template <int TK, int MM>
__global__ void k_export_routes(const dsr::PlanMT<TK, MM>* p, int owned, int Q, int* idx, float* wt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q * TK) return;
    const int m = i / TK, k = i % TK;
    idx[i] = m < owned ? p->tok_expert[m][k] : -1;
    wt[i] = m < owned ? p->tok_w[m][k] : 0.f;
}
// Rebuild token masks from globally gathered routes; zero masks before launch.
// Cuts remain -1/0. k_scan_ro + k_place_ro consume this in the normal gr8 format.
template <int NE, int TK, int MM>
__global__ void k_import_routes(const int* idx, const float* wt, int M,
                                dsr::PlanMT<TK, MM>* p, unsigned* masks)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * 8) return;
    const int m = i / 8, k = i % 8;
    const int e = k < TK ? idx[m * TK + k] : -1;
    p->tok_expert[m][k] = e;
    p->tok_w[m][k] = e >= 0 ? wt[m * TK + k] : 0.f;
    p->tok_slot[m][k] = -1;
    p->tok_col[m][k] = 0;
    if (e >= 0 && e < NE) atomicOr(masks + e * (MM / 32) + m / 32, 1u << (m & 31));
}
// k_place_ro assumes all first TK routes live. This variant imports optional cuts
// using the same stock rank-in-mask formula, without reading an expert -1.
template <int TK, int MM>
__global__ void k_place_routes(dsr::PlanMT<TK, MM>* p, const unsigned* masks, const int* tile0, int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * TK) return;
    const int m = i / TK, k = i % TK, e = p->tok_expert[m][k];
    if (e < 0) return;
    int t, c;
    dsr::mt_route_pos_w(tile0[e], masks + e * (MM / 32), m, t, c);
    p->tok_slot[m][k] = t;
    p->tok_col[m][k] = c;
    p->tile_w[t][c] = p->tok_w[m][k];
    p->tile_tok[t][c] = m;
}
// No quantization: the same E4M3 bytes in two layouts. Scales are already plain.
template <bool TO_RECORDS>
__global__ void k_relayout(const uint8_t* src, uint8_t* dst, int M, int K)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(M) * K) return;
    const int k = i % K;
    const size_t r = i - k + mxtc::aq_byte_off(k);
    if constexpr (TO_RECORDS) dst[r] = src[i];
    else dst[i] = src[r];
}
// One independent packed output per column group. Half expert partials already
// include routing weights (stock fc2 epilogue); shared is bf16, compact [M][cols].
// FP32 output lets the fabric sum ranks before the final bf16 rounding.
template <int NOUT, int TK, int MM, typename Out>
__global__ void k_combine_group(const dsr::PlanMT<TK, MM>* p, const __half* partial,
                                const __nv_bfloat16* shared, Out* out,
                                int M, int padded, int col0, int cols)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(padded) * cols) return;
    const int m = i / cols, n = i % cols;
    float v = 0.f;
    if (m < M)
    {
        #pragma unroll
        for (int k = 0; k < TK; ++k)
        {
            const int t = p->tok_slot[m][k];
            if (t >= 0) v += __half2float(partial[(static_cast<size_t>(t) * 8 + p->tok_col[m][k]) * NOUT + col0 + n]);
        }
        if (shared) v += __bfloat162float(shared[i]);
    }
    out[i] = static_cast<Out>(v);
}
} // namespace ds41v2moe
