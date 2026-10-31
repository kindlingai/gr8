// DSpark drafter of DeepSeek-V4.1-Flash-1017: the device kernels and launchers. docs/DEEPSEEK-DSPARK.md describes the design;
// ds41_dspark.h has the host rules and ds41_dspark_dev.h the pure functions both sides share. Raw pointers only (no Model, no fabric):
// ds41_dspark_ops.cuh binds these to the step driver's op slots.
//
//   ST_DS_TAP    k_ds_tap          streams [rows][4][5120] FP32 -> dsp_tap[:, slot, :] BF16 mean, at the entry of main layers 37/38/39
//   ST_DS_MAIN   ds_main           main_proj(tap) -> main_norm -> 3 x wkv -> kv_norm -> rope -> fp8 round trip -> the DSpark rings
//   ST_DS_DRAFT_IN  k_ds_draft_in  verify result -> q, golden, accepted count; the draft block's tokens [golden, noise x4] and positions
//   ST_DS_MARKOV    k_markov_add x5 (with the head merge between steps, in the ops file) + k_conf
//
// Layer maths of the draft block itself is attn/ds41_attn_dsp.cuh (block-mode attention) plus the shared mHC / MoE ops.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ds41_dspark_api.h"
#include "../attn/ds41_attn_dsp.cuh"
#include "ds41_dspark_dev.h"

namespace ds41dsp
{
namespace d = ds41::dspd;

// ---------------------------------------------------------------------------------------------------------------- tap
// streams [rows][4][5120] (the mHC trunk: float, or __nv_bfloat16 under -DDS_MHC_TRUNK_BF16; BF16-valued either way, so the widening is exact),
// tap BF16 [rows][3][5120]; writes slot `slot` of every row. Thread = 4 elements.
__device__ __forceinline__ float4 tap_ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float4 tap_ld4(const __nv_bfloat16* p)
{
    const uint2 u = *(const uint2*)p;
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
template <class TT>
__global__ void __launch_bounds__(256) k_ds_tap(const TT* __restrict__ streams, __nv_bfloat16* __restrict__ tap, int slot, int rows)
{
    const int row = blockIdx.y;
    const int e = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (row >= rows || e >= d::HID) return;
    const TT* s = streams + (size_t)row * 4 * d::HID + e;
    const float4 a = tap_ld4(s);
    const float4 b = tap_ld4(s + d::HID);
    const float4 c = tap_ld4(s + 2 * d::HID);
    const float4 g = tap_ld4(s + 3 * d::HID);
    __nv_bfloat162 lo = __floats2bfloat162_rn(d::tap_mean_f32(a.x, b.x, c.x, g.x), d::tap_mean_f32(a.y, b.y, c.y, g.y));
    __nv_bfloat162 hi = __floats2bfloat162_rn(d::tap_mean_f32(a.z, b.z, c.z, g.z), d::tap_mean_f32(a.w, b.w, c.w, g.w));
    uint2 v;
    v.x = *(uint32_t*)&lo;
    v.y = *(uint32_t*)&hi;
    *(uint2*)(tap + ((size_t)row * d::TAPS + slot) * d::HID + e) = v;
}
void ds_tap(const float* streams, __nv_bfloat16* tap, int slot, int rows, cudaStream_t st)
{
    k_ds_tap<float><<<dim3((d::HID / 4 + 255) / 256, rows), 256, 0, st>>>(streams, tap, slot, rows);
}
void ds_tap(const __nv_bfloat16* streams, __nv_bfloat16* tap, int slot, int rows, cudaStream_t st)
{
    k_ds_tap<__nv_bfloat16><<<dim3((d::HID / 4 + 255) / 256, rows), 256, 0, st>>>(streams, tap, slot, rows);
}

bool dsp_gv_enabled() { return dsp_gv_on(); }
bool dsp_gv_ok(const ds41a::Fp8Mat& W, int rows) { return dsp_use_gv(W, rows); }

// ---------------------------------------------------------------------------------------------------------------- main cache
// M verify rows, rows_per_sess = k + 1; pos [M] = their absolute positions. Writes the main key of every row into the 3 rings.
void ds_main(const MainW& W, const MainS& S, const bf16* tap, int M, int rows_per_sess, const int* pos, cudaStream_t st, bool wide, const int* dsess)
{
    using namespace ds41a;
    launch_act_quant(tap, d::TAPS * d::HID, S.tq, d::TAPS * d::HID, S.ts, d::TAPS * d::HID / 32, M, d::TAPS * d::HID, st);
    dsp_mm(W.main_proj, S.tq, d::TAPS * d::HID, S.ts, d::TAPS * d::HID / 32, S.xf_tap, false, S.mh0, HID, M, 1, 0, 0, 0, st, wide);
    k_rmsnorm<<<M, 256, 0, st>>>(S.mh0, HID, W.main_norm, S.mh, HID, HID, EPS);
    launch_act_quant(S.mh, HID, S.hq, HID, S.hs, HID / 32, M, HID, st);
    bool h_ready = false;   // GR8_DS_DSP_GV: main_x's GEMV table, built by the first wkv that takes the GEMV and shared by the other two
    for (int j = 0; j < 3; j++)
    {
        dsp_mm(W.wkv[j], S.hq, HID, S.hs, HID / 32, S.xf_h, h_ready, S.mk, HD, M, 1, 0, 0, 0, st, wide);
        h_ready = h_ready || dsp_use_gv(W.wkv[j], M);
        k_kv_prep_pos<<<M, 512, 0, st>>>(S.mk, W.kv_norm[j], W.inv, pos, rows_per_sess, W.ring[j].base, W.ring[j].sess_stride, W.ring[j].ring_rows, dsess);
    }
}

// Prefill: `rows` consecutive rows of one session (tap [rows][3][5120], pos [rows]) in launches of at most `cap` rows, ascending.
// cap <= min(MainS rows, ring_rows): one launch must not hold two rows of the same ring slot (k_kv_prep_pos blocks run in any order); launches
// on one stream give "newest position wins". Every launch is the plain ds_main with rows_per_sess = its own row count (session 0).
// wide = true: a launch of more than 8 rows takes the GEMM even with GR8_DS_DSP_GV=1 (the GEMV is the decode route, 1..8 rows).
void ds_main_rows(const MainW& W, const MainS& S, const bf16* tap, int rows, const int* pos, int cap, cudaStream_t st, const int* dsess)
{
    for (int r0 = 0; r0 < rows; r0 += cap)
    {
        const int m = rows - r0 < cap ? rows - r0 : cap;
        ds_main(W, S, tap + (size_t)r0 * d::TAPS * d::HID, m, m, pos + r0, st, /*wide=*/true, dsess ? dsess + r0 : nullptr);
    }
}

// ---------------------------------------------------------------------------------------------------------------- accept + draft input
// One thread per session. Reads the verify step (vtok = the rows' input tokens, vtgt = their argmax, vpos = their positions; rows_per_sess =
// k + 1) and writes q (last accepted position), golden (the token at q + 1), the accepted count, and the draft block's tokens / positions
// (tok, pos: int [nsess][5], the draft Model's m.tok / m.pos). The device twin of ds41::dsp::accept_greedy.
__global__ void k_ds_draft_in(const int* __restrict__ vtok, const int* __restrict__ vtgt, const int* __restrict__ vpos, int rows_per_sess, int nsess,
                              int* __restrict__ q_out, int* __restrict__ golden_out, int* __restrict__ acc_out, int* __restrict__ tok, int* __restrict__ pos)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nsess) return;
    int q, g;
    if (vtok)
    {
        const int r0 = s * rows_per_sess;
        int a;
        d::accept_session(vtok + r0, vtgt + r0, rows_per_sess, vpos[r0], a, q, g);
        q_out[s] = q;
        golden_out[s] = g;
        if (acc_out) acc_out[s] = a;
    }
    else { q = q_out[s]; g = golden_out[s]; }   // vtok == nullptr: the host wrote q / golden / acc (Dsp::host_accept)
    for (int r = 0; r < d::BLOCK; r++)
    {
        tok[s * d::BLOCK + r] = d::block_token(g, r);
        pos[s * d::BLOCK + r] = d::block_pos(q, r);
    }
}
void ds_draft_in(const int* vtok, const int* vtgt, const int* vpos, int rows_per_sess, int nsess, int* q, int* golden, int* acc, int* tok, int* pos,
                        cudaStream_t st)
{
    k_ds_draft_in<<<(nsess + 31) / 32, 32, 0, st>>>(vtok, vtgt, vpos, rows_per_sess, nsess, q, golden, acc, tok, pos);
}

// ---------------------------------------------------------------------------------------------------------------- markov head
// Step i: hlog[(s*5 + i) * vn + v] += dot(markov_head.head[v][:256], markov_embed[prev][:256]) for every session s and every vocab id v of this
// rank's shard (mhead = the shard's rows [vn][256] BF16). prev = golden[s] at i == 0, else the previous merge's draft (dtok i-major [5][nsess]).
// One warp per vocab row: the row is read once and dotted with all sessions' w1. FP32 accumulate, FP32 add into the logits.
__global__ void __launch_bounds__(256) k_markov_add(const bf16* __restrict__ membed, const bf16* __restrict__ mhead, const int* __restrict__ golden,
                                                    const int* __restrict__ dtok, int i, int nsess, float* hlog, long ld_row, int vn)
{
    __shared__ float w1[MARKOV_MAXS][d::MARKOV_RANK];
    for (int e = threadIdx.x; e < nsess * d::MARKOV_RANK; e += 256)
    {
        const int s = e / d::MARKOV_RANK, c = e % d::MARKOV_RANK;
        w1[s][c] = __bfloat162float(membed[(size_t)d::markov_prev(golden, dtok, nsess, s, i) * d::MARKOV_RANK + c]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + warp;
    if (v >= vn) return;
    const uint4 hv = *(const uint4*)(mhead + (size_t)v * d::MARKOV_RANK + lane * 8);
    float h[8];
    {
        const __nv_bfloat162* p = (const __nv_bfloat162*)&hv;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const float2 f = __bfloat1622float2(p[j]);
            h[2 * j] = f.x;
            h[2 * j + 1] = f.y;
        }
    }
    for (int s = 0; s < nsess; s++)
    {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) sum = fmaf(h[j], w1[s][lane * 8 + j], sum);
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
        if (lane == 0) hlog[(size_t)(s * d::BLOCK + i) * ld_row + v] += sum;
    }
}
void markov_add(const bf16* membed, const bf16* mhead, const int* golden, const int* dtok, int i, int nsess, float* hlog, long ld_row, int vn,
                       cudaStream_t st)
{
    k_markov_add<<<(vn + 7) / 8, 256, 0, st>>>(membed, mhead, golden, dtok, i, nsess, hlog, ld_row, vn);
}

// Confidence score and the draft id vector. One block per draft row (s*5 + i): score = dot(concat(x_row [5120], w1 [256]), proj [5376]),
// FP32 (the release returns this raw dot, the confidence head of the reference model.py; sigmoid and the temperature are DeepInfer's and apply on use). Also moves
// the i-major merge output into the session-major draft vector the next verify step reads: draft[s*5 + i] = dtok[i*nsess + s].
__global__ void __launch_bounds__(256) k_conf(const bf16* __restrict__ x, const bf16* __restrict__ membed, const bf16* __restrict__ proj,
                                              const int* __restrict__ golden, const int* __restrict__ dtok, int nsess, float* __restrict__ conf,
                                              int* __restrict__ draft)
{
    __shared__ float sh[32];
    const int row = blockIdx.x, s = row / d::BLOCK, i = row % d::BLOCK;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < d::HID; c += 256) acc = fmaf(__bfloat162float(x[(size_t)row * d::HID + c]), __bfloat162float(proj[c]), acc);
    const int prev = d::markov_prev(golden, dtok, nsess, s, i);
    for (int c = threadIdx.x; c < d::MARKOV_RANK; c += 256)
        acc = fmaf(__bfloat162float(membed[(size_t)prev * d::MARKOV_RANK + c]), __bfloat162float(proj[d::HID + c]), acc);
    const float tot = ds41a::block_sum(acc, sh);
    if (threadIdx.x == 0)
    {
        conf[row] = tot;
        draft[row] = dtok[i * nsess + s];
    }
}
void conf_head(const bf16* x, const bf16* membed, const bf16* proj, const int* golden, const int* dtok, int nsess, float* conf, int* draft,
                      cudaStream_t st)
{
    k_conf<<<nsess * d::BLOCK, 256, 0, st>>>(x, membed, proj, golden, dtok, nsess, conf, draft);
}

// ---------------------------------------------------------------------------------------------------------------- next verify rows
// The next round's verify step from the draft block: row s * (k + 1) + r of the step = token [golden, draft_0 .. draft_{k-1}][r] at position
// q + 1 + r. One thread per row. A host-driven round can read draft / q / golden instead; a graph-captured round launches this after the
// draft's markov stage and before the main step's embed.
__global__ void k_ds_verify_in(const int* __restrict__ golden, const int* __restrict__ draft, const int* __restrict__ q, int k, int nsess,
                               int* __restrict__ tok, int* __restrict__ pos)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nsess * (k + 1)) return;
    const int s = e / (k + 1), r = e % (k + 1);
    tok[e] = r == 0 ? golden[s] : draft[s * d::BLOCK + r - 1];
    pos[e] = d::block_pos(q[s], r);
}
void ds_verify_in(const int* golden, const int* draft, const int* q, int k, int nsess, int* tok, int* pos, cudaStream_t st)
{
    k_ds_verify_in<<<(nsess * (k + 1) + 31) / 32, 32, 0, st>>>(golden, draft, q, k, nsess, tok, pos);
}

// ---------------------------------------------------------------------------------------------------------------- the per-Model bundle
// Everything the DSpark ops of one rank touch, as raw device pointers (no Model, no fabric, no rank.cuh). One instance lives in each
// g6rt::Model (`ds41dsp::Dsp dsp;`): the main Model and the draft Model carry copies filled from the same allocations, so an op reads
// m.dsp whichever Model it runs on. ds41_dspark_ops.cuh binds these to the stage slots and has the allocator.
}   // namespace ds41dsp
