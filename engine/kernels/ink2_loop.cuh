// Inkling-2-Mini looped core kernels: unrounded fp32 depth attention (BlockAttentionResidual / MessageAttentionResidual)
// and fused MessageGate + RMSNorm (Inkling2LoopMessageGate). Our own code.
//
// Twins of ink2ref::Model::attn_res_f32 and ink2ref::Model::msg_gate (engine/ref/ink2/ink2_cpu_ref.h):
//   attn_res_f32     per token over D <= MAX_DEPTH BF16 rows: U_d = row_d * r_d in fp32, logit = query . U_d / sqrt(H),
//                    out[q] = sum_d softmax_d * U_d in fp32 (no BF16 rounding), for Q in 1 .. 4 queries.
//   msg_gate_prep    cat[m] = [bf16(unweighted_rms(h_f32[m])), bf16(unweighted_rms(msg_f32[m]))] in [M][2*H] BF16.
//   msg_gate_norm    g = bf16(cat @ down_w^T) [M][R], then fused per channel:
//                      fg = tanh(forget_w[c] . g), mg = tanh(gain_w[c] . g),
//                      val = (1 - fg) * h_f32[m, c] + mg * msg_f32[m, c],
//                      out_bf16[m, c] = bf16(norm_w[c] * (val * rsqrt(mean(val^2) + eps))).
//
// Shapes are runtime arguments. H is any multiple of 16 (<= 5120 for the register-cached fused norm path; larger H
// falls back cleanly to a two-pass loop), R <= 256 (Inkling-2-Mini uses R = 64).
#pragma once
#include "ink2_norm_res.cuh"

namespace ink2
{
constexpr int LOOP_MAX_Q = 4;        // q, k, v, mlp messages
constexpr int LOOP_MAX_RANK = 256;   // max gate_rank supported in shared memory (Inkling-2-Mini = 64)

#ifdef __CUDACC__
struct AttnResF32Args
{
    const uint16_t* rows[MAX_DEPTH];   // D rows [M][H] BF16
    const float* query;                // [Q][H] fp32
    float* out[LOOP_MAX_Q];            // [M][H] fp32 each (Q in 1..4)
    int D, Q, H;
};

// CTA = token. Unrounded fp32 depth attention (used for message_input Q=4, and pass>=1 attention_input / mlp_input).
// Fuses all Q queries into a single read of rows[d] for dots and a single read for outputs (100% bit-exact!).
__global__ void __launch_bounds__(NRT) k_ink2_attn_res_f32(AttnResF32Args a)
{
    __shared__ float sm[NRT * MAX_DEPTH];
    const int t = threadIdx.x, H = a.H, D = a.D;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    float v[MAX_DEPTH], r[MAX_DEPTH];
    for (int d = 0; d < D; ++d)
    {
        const uint16_t* row = a.rows[d] + off;
        float s = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float x = bf(__ldg(row + c));
            s = add_rn(s, mul_rn(x, x));
        }
        v[d] = s;
    }
    blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
    for (int d = 0; d < D; ++d) r[d] = rsq(add_rn(div_rn(v[d], static_cast<float>(H)), RMS_EPS));
    const float root_h = sqrtf(static_cast<float>(H));
    if (a.Q == 1)
    {
        const float* qr = a.query;
        for (int d = 0; d < D; ++d)
        {
            const uint16_t* row = a.rows[d] + off;
            float s = 0.0f;
            for (int c = t; c < H; c += NRT) s = add_rn(s, mul_rn(__ldg(qr + c), mul_rn(bf(__ldg(row + c)), r[d])));
            v[d] = s;
        }
        blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
        float mx = -INFINITY, den = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            v[d] = div_rn(v[d], root_h);
            mx = fmaxf(mx, v[d]);
        }
        for (int d = 0; d < D; ++d)
        {
            v[d] = expf(v[d] - mx);
            den = add_rn(den, v[d]);
        }
        for (int d = 0; d < D; ++d) v[d] = div_rn(v[d], den);
        float* o = a.out[0] + off;
        for (int c = t; c < H; c += NRT)
        {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s = add_rn(s, mul_rn(v[d], mul_rn(bf(__ldg(a.rows[d] + off + c)), r[d])));
            o[c] = s;
        }
        return;
    }
    __shared__ float s_wq[LOOP_MAX_Q][MAX_DEPTH];
    for (int d = 0; d < D; ++d)
    {
        const uint16_t* row = a.rows[d] + off;
        float sq[LOOP_MAX_Q];
        #pragma unroll
        for (int q = 0; q < LOOP_MAX_Q; ++q) sq[q] = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float u = mul_rn(bf(__ldg(row + c)), r[d]);
            for (int q = 0; q < a.Q; ++q)
                sq[q] = add_rn(sq[q], mul_rn(__ldg(a.query + static_cast<size_t>(q) * H + c), u));
        }
        blk_sum_v<LOOP_MAX_Q>(sq, a.Q, sm, NRT);
        if (t < a.Q) s_wq[t][d] = sq[t];
    }
    __syncthreads();
    if (t < a.Q)
    {
        float mx = -INFINITY, den = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            s_wq[t][d] = div_rn(s_wq[t][d], root_h);
            mx = fmaxf(mx, s_wq[t][d]);
        }
        for (int d = 0; d < D; ++d)
        {
            s_wq[t][d] = expf(s_wq[t][d] - mx);
            den = add_rn(den, s_wq[t][d]);
        }
        for (int d = 0; d < D; ++d) s_wq[t][d] = div_rn(s_wq[t][d], den);
    }
    __syncthreads();
    for (int c = t; c < H; c += NRT)
    {
        float acc[LOOP_MAX_Q];
        #pragma unroll
        for (int q = 0; q < LOOP_MAX_Q; ++q) acc[q] = 0.0f;
        for (int d = 0; d < D; ++d)
        {
            const float u = mul_rn(bf(__ldg(a.rows[d] + off + c)), r[d]);
            for (int q = 0; q < a.Q; ++q) acc[q] = add_rn(acc[q], mul_rn(s_wq[q][d], u));
        }
        for (int q = 0; q < a.Q; ++q) a.out[q][off + c] = acc[q];
    }
}

// CTA = token. Builds cat_bf16 [M][2*H] = [bf16(unweighted_rms(h_f32)), bf16(unweighted_rms(msg_f32))].
__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_prep(const float* h_f32, const float* msg_f32, uint16_t* cat_bf16, int H)
{
    __shared__ float sm[NRT * 2];
    const int t = threadIdx.x;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    const size_t coff = static_cast<size_t>(blockIdx.x) * (2 * H);
    float ss[2] = {0.0f, 0.0f};
    if (H <= NRT * 9)
    {
        float hr[9], mr[9];
        int idx = 0;
        for (int c = t; c < H; c += NRT, ++idx)
        {
            const float hv = h_f32[off + c], mv = msg_f32[off + c];
            hr[idx] = hv;
            mr[idx] = mv;
            ss[0] = add_rn(ss[0], mul_rn(hv, hv));
            ss[1] = add_rn(ss[1], mul_rn(mv, mv));
        }
        blk_sum_v<2>(ss, 2, sm, NRT);
        const float rh = rsq(add_rn(div_rn(ss[0], static_cast<float>(H)), RMS_EPS));
        const float rm = rsq(add_rn(div_rn(ss[1], static_cast<float>(H)), RMS_EPS));
        idx = 0;
        for (int c = t; c < H; c += NRT, ++idx)
        {
            cat_bf16[coff + c] = bf16(mul_rn(hr[idx], rh));
            cat_bf16[coff + H + c] = bf16(mul_rn(mr[idx], rm));
        }
        return;
    }
    for (int c = t; c < H; c += NRT)
    {
        const float hv = h_f32[off + c], mv = msg_f32[off + c];
        ss[0] = add_rn(ss[0], mul_rn(hv, hv));
        ss[1] = add_rn(ss[1], mul_rn(mv, mv));
    }
    blk_sum_v<2>(ss, 2, sm, NRT);
    const float rh = rsq(add_rn(div_rn(ss[0], static_cast<float>(H)), RMS_EPS));
    const float rm = rsq(add_rn(div_rn(ss[1], static_cast<float>(H)), RMS_EPS));
    for (int c = t; c < H; c += NRT)
    {
        cat_bf16[coff + c] = bf16(mul_rn(h_f32[off + c], rh));
        cat_bf16[coff + H + c] = bf16(mul_rn(msg_f32[off + c], rm));
    }
}

// Coalesced 128B-line warp-cooperative MessageGate apply+norm for R == 64 (and H <= 4096).
// Every warp processes channels c = warp, warp+8, ...; all 32 lanes load 2 BF16s (4B) each = 128B coalesced line!
__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_apply_norm_r64(const float* __restrict__ h_f32, const float* __restrict__ msg_f32,
                                                                      const uint16_t* __restrict__ g_bf16,
                                                                      const uint16_t* __restrict__ forget_w, const uint16_t* __restrict__ gain_w,
                                                                      const uint16_t* __restrict__ norm_w,
                                                                      uint16_t* __restrict__ out_bf16, int H)
{
    __shared__ float s_val[4096];
    __shared__ float sm[NRT];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    const uint32_t gpair = reinterpret_cast<const uint32_t*>(g_bf16 + static_cast<size_t>(blockIdx.x) * 64)[lane];
    const float g0 = bf(static_cast<uint16_t>(gpair));
    const float g1 = bf(static_cast<uint16_t>(gpair >> 16));

    for (int c = warp; c < H; c += 8)
    {
        const uint32_t fp = __ldg(reinterpret_cast<const uint32_t*>(forget_w + static_cast<size_t>(c) * 64) + lane);
        const uint32_t gp = __ldg(reinterpret_cast<const uint32_t*>(gain_w + static_cast<size_t>(c) * 64) + lane);
        float sf = add_rn(mul_rn(bf(static_cast<uint16_t>(fp)), g0), mul_rn(bf(static_cast<uint16_t>(fp >> 16)), g1));
        float sm_ = add_rn(mul_rn(bf(static_cast<uint16_t>(gp)), g0), mul_rn(bf(static_cast<uint16_t>(gp >> 16)), g1));
        #pragma unroll
        for (int off_ = 16; off_ > 0; off_ >>= 1)
        {
            sf = add_rn(sf, __shfl_xor_sync(0xffffffffu, sf, off_));
            sm_ = add_rn(sm_, __shfl_xor_sync(0xffffffffu, sm_, off_));
        }
        if (lane == 0)
        {
            const float fg = tanhf(sf);
            const float mg = tanhf(sm_);
            s_val[c] = add_rn(mul_rn(add_rn(1.0f, -fg), h_f32[off + c]), mul_rn(mg, msg_f32[off + c]));
        }
    }
    __syncthreads();
    float ss = 0.0f;
    for (int c = t; c < H; c += NRT)
    {
        const float v = s_val[c];
        ss = add_rn(ss, mul_rn(v, v));
    }
    const float rn = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    for (int c = t; c < H; c += NRT) out_bf16[off + c] = bf16(mul_rn(bf(norm_w[c]), mul_rn(s_val[c], rn)));
}

// Batched dual-output coalesced GEMV for R == 64: reads forget_w[c] and gain_w[c] ONCE for up to MAX_M tokens
// using the EXACT SAME 32-lane coalesced dot + warp reduction order as k_ink2_msg_gate_apply_norm_r64!
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink2_msg_gate_fg_r64(const uint16_t* __restrict__ g_bf16,
                                                              const uint16_t* __restrict__ forget_w,
                                                              const uint16_t* __restrict__ gain_w,
                                                              float* __restrict__ sf_out,
                                                              float* __restrict__ sm_out,
                                                              int M, int H)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + warp;
    if (c >= H) return;
    const int m0 = blockIdx.y * MAX_M;
    const int m_rem = min(MAX_M, M - m0);
    if (m_rem <= 0) return;

    const uint32_t fp = __ldg(reinterpret_cast<const uint32_t*>(forget_w + static_cast<size_t>(c) * 64) + lane);
    const uint32_t gp = __ldg(reinterpret_cast<const uint32_t*>(gain_w + static_cast<size_t>(c) * 64) + lane);
    const float f0 = bf(static_cast<uint16_t>(fp)), f1 = bf(static_cast<uint16_t>(fp >> 16));
    const float g0w = bf(static_cast<uint16_t>(gp)), g1w = bf(static_cast<uint16_t>(gp >> 16));

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m)
    {
        if (m < m_rem)
        {
            const uint32_t gpair = __ldg(reinterpret_cast<const uint32_t*>(g_bf16 + static_cast<size_t>(m0 + m) * 64) + lane);
            const float x0 = bf(static_cast<uint16_t>(gpair));
            const float x1 = bf(static_cast<uint16_t>(gpair >> 16));
            float sf = add_rn(mul_rn(f0, x0), mul_rn(f1, x1));
            float sm_ = add_rn(mul_rn(g0w, x0), mul_rn(g1w, x1));
            #pragma unroll
            for (int off_ = 16; off_ > 0; off_ >>= 1)
            {
                sf = add_rn(sf, __shfl_xor_sync(0xffffffffu, sf, off_));
                sm_ = add_rn(sm_, __shfl_xor_sync(0xffffffffu, sm_, off_));
            }
            if (lane == 0)
            {
                const size_t o = static_cast<size_t>(m0 + m) * H + c;
                sf_out[o] = sf;
                sm_out[o] = sm_;
            }
        }
    }
}

// CTA = token. Finish MessageGate apply + weighted RMSNorm from precomputed sf_in / sm_in [M][H].
__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_finish(const float* __restrict__ h_f32, const float* __restrict__ msg_f32,
                                                              const float* __restrict__ sf_in, const float* __restrict__ sm_in,
                                                              const uint16_t* __restrict__ norm_w,
                                                              uint16_t* __restrict__ out_bf16, int H)
{
    __shared__ float sm[NRT];
    const int t = threadIdx.x;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    constexpr int MAX_REG = 20;
    float vals[MAX_REG];
    const bool in_reg = (H <= NRT * MAX_REG);
    float ss = 0.0f;
    int idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        const float fg = tanhf(sf_in[off + c]);
        const float mg = tanhf(sm_in[off + c]);
        const float v = add_rn(mul_rn(add_rn(1.0f, -fg), h_f32[off + c]), mul_rn(mg, msg_f32[off + c]));
        if (in_reg) vals[idx] = v;
        ss = add_rn(ss, mul_rn(v, v));
    }
    const float rn = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        const float v = in_reg ? vals[idx] : add_rn(mul_rn(add_rn(1.0f, -tanhf(sf_in[off + c])), h_f32[off + c]), mul_rn(tanhf(sm_in[off + c]), msg_f32[off + c]));
        out_bf16[off + c] = bf16(mul_rn(bf(norm_w[c]), mul_rn(v, rn)));
    }
}

// CTA = token. Fused MessageGate apply + weighted RMSNorm -> out_bf16 [M][H].
// g_bf16 [M][R], forget_w [H][R] BF16, gain_w [H][R] BF16, norm_w [H] BF16.
__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_apply_norm(const float* h_f32, const float* msg_f32, const uint16_t* g_bf16,
                                                                  const uint16_t* forget_w, const uint16_t* gain_w, const uint16_t* norm_w,
                                                                  uint16_t* out_bf16, int H, int R)
{
    __shared__ float sm[NRT];
    __shared__ float sg[LOOP_MAX_RANK];
    const int t = threadIdx.x;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    const uint16_t* gm = g_bf16 + static_cast<size_t>(blockIdx.x) * R;
    for (int r = t; r < R; r += NRT) sg[r] = bf(gm[r]);
    __syncthreads();

    constexpr int MAX_REG = 20;   // covers H up to 256 * 20 = 5120 in registers (Inkling-2-Mini H = 2304 -> 9 regs)
    float vals[MAX_REG];
    const bool in_reg = (H <= NRT * MAX_REG);
    float ss = 0.0f;
    int idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        const uint16_t* fw = forget_w + static_cast<size_t>(c) * R;
        const uint16_t* gw = gain_w + static_cast<size_t>(c) * R;
        float sf = 0.0f, sm_ = 0.0f;
        for (int r = 0; r < R; ++r)
        {
            const float gr = sg[r];
            sf = add_rn(sf, mul_rn(bf(fw[r]), gr));
            sm_ = add_rn(sm_, mul_rn(bf(gw[r]), gr));
        }
        const float fg = tanhf(sf);
        const float mg = tanhf(sm_);
        const float v = add_rn(mul_rn(add_rn(1.0f, -fg), h_f32[off + c]), mul_rn(mg, msg_f32[off + c]));
        if (in_reg) vals[idx] = v;
        ss = add_rn(ss, mul_rn(v, v));
    }
    const float rn = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        float v;
        if (in_reg) v = vals[idx];
        else
        {
            const uint16_t* fw = forget_w + static_cast<size_t>(c) * R;
            const uint16_t* gw = gain_w + static_cast<size_t>(c) * R;
            float sf = 0.0f, sm_ = 0.0f;
            for (int r = 0; r < R; ++r)
            {
                const float gr = sg[r];
                sf = add_rn(sf, mul_rn(bf(fw[r]), gr));
                sm_ = add_rn(sm_, mul_rn(bf(gw[r]), gr));
            }
            v = add_rn(mul_rn(add_rn(1.0f, -tanhf(sf)), h_f32[off + c]), mul_rn(tanhf(sm_), msg_f32[off + c]));
        }
        out_bf16[off + c] = bf16(mul_rn(bf(norm_w[c]), mul_rn(v, rn)));
    }
}

inline bool& msg_bf16_enabled()
{
    static bool enabled = true;
    return enabled;
}
// 1-pass dynamic-SMEM zero-spill vectorized uint4 unrounded fp32 depth attention (QC = 1..4).
template <int QC>
__global__ void __launch_bounds__(NRT) k_ink2_attn_res_f32_fast(AttnResF32Args a)
{
    extern __shared__ __align__(16) uint4 s_rows[];
    __shared__ float s_red[8 * (QC + 1)];
    __shared__ float s_vss[MAX_DEPTH];
    __shared__ float s_sq[QC][MAX_DEPTH];
    __shared__ float s_weff[QC][MAX_DEPTH];

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int H = a.H, D = a.D;
    const int H8 = H >> 3;
    const size_t off8 = static_cast<size_t>(blockIdx.x) * H8;
    const float inv_root_h = rsqrtf(static_cast<float>(H));

    for (int d = 0; d < D; ++d)
    {
        const uint4* r8 = reinterpret_cast<const uint4*>(a.rows[d]) + off8;
        float s = 0.0f;
        float qacc[QC];
        #pragma unroll
        for (int q = 0; q < QC; ++q) qacc[q] = 0.0f;

        for (int i = t; i < H8; i += NRT)
        {
            const uint4 xv = __ldg(r8 + i);
            s_rows[d * H8 + i] = xv;
            const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float f[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const float2 fp = __bfloat1622float2(bp[k]);
                f[2 * k] = fp.x; f[2 * k + 1] = fp.y;
                s = fmaf(fp.x, fp.x, fmaf(fp.y, fp.y, s));
            }
            #pragma unroll
            for (int q = 0; q < QC; ++q)
            {
                const float4* q4 = reinterpret_cast<const float4*>(a.query + static_cast<size_t>(q) * H) + (i << 1);
                const float4 qa = __ldg(q4), qb = __ldg(q4 + 1);
                float dq = qacc[q];
                dq = fmaf(qa.x, f[0], dq); dq = fmaf(qa.y, f[1], dq);
                dq = fmaf(qa.z, f[2], dq); dq = fmaf(qa.w, f[3], dq);
                dq = fmaf(qb.x, f[4], dq); dq = fmaf(qb.y, f[5], dq);
                dq = fmaf(qb.z, f[6], dq); dq = fmaf(qb.w, f[7], dq);
                qacc[q] = dq;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            s += __shfl_xor_sync(0xffffffffu, s, o);
            #pragma unroll
            for (int q = 0; q < QC; ++q) qacc[q] += __shfl_xor_sync(0xffffffffu, qacc[q], o);
        }
        if (lane == 0)
        {
            s_red[warp] = s;
            #pragma unroll
            for (int q = 0; q < QC; ++q) s_red[(q + 1) * 8 + warp] = qacc[q];
        }
        __syncthreads();
        if (t == 0)
        {
            s_vss[d] = s_red[0] + s_red[1] + s_red[2] + s_red[3] + s_red[4] + s_red[5] + s_red[6] + s_red[7];
            #pragma unroll
            for (int q = 0; q < QC; ++q)
            {
                const float* p = s_red + (q + 1) * 8;
                s_sq[q][d] = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
            }
        }
        __syncthreads();
    }

    if (t == 0)
    {
        float r[MAX_DEPTH];
        for (int d = 0; d < D; ++d) r[d] = rsqrtf(s_vss[d] / static_cast<float>(H) + RMS_EPS);
        #pragma unroll
        for (int q = 0; q < QC; ++q)
        {
            float mx = -INFINITY, den = 0.0f;
            float lg[MAX_DEPTH];
            for (int d = 0; d < D; ++d)
            {
                lg[d] = s_sq[q][d] * r[d] * inv_root_h;
                mx = fmaxf(mx, lg[d]);
            }
            for (int d = 0; d < D; ++d) { lg[d] = __expf(lg[d] - mx); den += lg[d]; }
            const float iden = 1.0f / den;
            for (int d = 0; d < D; ++d) s_weff[q][d] = lg[d] * iden * r[d];
        }
    }
    __syncthreads();

    for (int i = t; i < H8; i += NRT)
    {
        float acc[QC][8];
        #pragma unroll
        for (int q = 0; q < QC; ++q)
            #pragma unroll
            for (int k = 0; k < 8; ++k) acc[q][k] = 0.0f;

        for (int d = 0; d < D; ++d)
        {
            const uint4 xv = s_rows[d * H8 + i];
            const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float f[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const float2 fp = __bfloat1622float2(bp[k]);
                f[2 * k] = fp.x; f[2 * k + 1] = fp.y;
            }
            #pragma unroll
            for (int q = 0; q < QC; ++q)
            {
                const float w = s_weff[q][d];
                #pragma unroll
                for (int k = 0; k < 8; ++k) acc[q][k] = fmaf(w, f[k], acc[q][k]);
            }
        }
        #pragma unroll
        for (int q = 0; q < QC; ++q)
        {
            float4* o4 = reinterpret_cast<float4*>(a.out[q] + static_cast<size_t>(blockIdx.x) * H) + (i << 1);
            o4[0] = make_float4(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
            o4[1] = make_float4(acc[q][4], acc[q][5], acc[q][6], acc[q][7]);
        }
    }
}

// Zero-spill SMEM vectorized float4/uint2 MessageGate finish (reads BF16 sf/sm -> 2x less DRAM, STACK:0!)
__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_finish_bf16(
    const float* __restrict__ h_f32, const float* __restrict__ msg_f32,
    const uint16_t* __restrict__ sf_in, const uint16_t* __restrict__ sm_in,
    const uint16_t* __restrict__ norm_w, uint16_t* __restrict__ out_bf16, int H)
{
    __shared__ float4 s_v[1024]; // up to H=4096 (16 KB SMEM -> 4 CTAs/SM, STACK:0!)
    __shared__ float s_red[8];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int H4 = H >> 2;
    const size_t off4 = static_cast<size_t>(blockIdx.x) * H4;
    const float4* hf4 = reinterpret_cast<const float4*>(h_f32) + off4;
    const float4* mf4 = reinterpret_cast<const float4*>(msg_f32) + off4;
    const uint2* sf2 = reinterpret_cast<const uint2*>(sf_in) + off4;
    const uint2* sm2 = reinterpret_cast<const uint2*>(sm_in) + off4;
    const uint2* nw2 = reinterpret_cast<const uint2*>(norm_w);
    uint2* out2 = reinterpret_cast<uint2*>(out_bf16) + off4;

    float ss = 0.0f;
    for (int i = t; i < H4; i += NRT)
    {
        const float4 hv = __ldg(hf4 + i), mv = __ldg(mf4 + i);
        const uint2 svf = __ldg(sf2 + i), svm = __ldg(sm2 + i);
        const float2 fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svf.x));
        const float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svf.y));
        const float2 ma = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svm.x));
        const float2 mb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svm.y));
        const float4 vv = make_float4(
            fmaf(1.0f - tanhf(fa.x), hv.x, tanhf(ma.x) * mv.x),
            fmaf(1.0f - tanhf(fa.y), hv.y, tanhf(ma.y) * mv.y),
            fmaf(1.0f - tanhf(fb.x), hv.z, tanhf(mb.x) * mv.z),
            fmaf(1.0f - tanhf(fb.y), hv.w, tanhf(mb.y) * mv.w));
        s_v[i] = vv;
        ss = fmaf(vv.x, vv.x, fmaf(vv.y, vv.y, fmaf(vv.z, vv.z, fmaf(vv.w, vv.w, ss))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (t == 0)
    {
        const float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3] + s_red[4] + s_red[5] + s_red[6] + s_red[7];
        s_red[0] = rsqrtf(tot / static_cast<float>(H) + RMS_EPS);
    }
    __syncthreads();
    const float rn = s_red[0];
    for (int i = t; i < H4; i += NRT)
    {
        const float4 vv = s_v[i];
        const uint2 nv = __ldg(nw2 + i);
        const float2 na = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nv.x));
        const float2 nb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nv.y));
        const __nv_bfloat162 oa = __floats2bfloat162_rn(na.x * vv.x * rn, na.y * vv.y * rn);
        const __nv_bfloat162 ob = __floats2bfloat162_rn(nb.x * vv.z * rn, nb.y * vv.w * rn);
        uint2 ov;
        ov.x = *reinterpret_cast<const uint32_t*>(&oa);
        ov.y = *reinterpret_cast<const uint32_t*>(&ob);
        out2[i] = ov;
    }
}

inline void attn_res_f32(const AttnResF32Args& a, int M, cudaStream_t st)
{
    if (a.D < 1 || a.D > MAX_DEPTH || a.Q < 1 || a.Q > LOOP_MAX_Q || M < 1) throw std::runtime_error("ink2::attn_res_f32: D, Q or M");
    const bool fast_res = fast_res_enabled();
    const size_t smem = static_cast<size_t>(a.D) * a.H * sizeof(uint16_t);
    if (fast_res && (a.H & 7) == 0 && smem <= 65536)
    {
        static const bool attr_ok = []() {
            cudaFuncSetAttribute(k_ink2_attn_res_f32_fast<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(k_ink2_attn_res_f32_fast<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(k_ink2_attn_res_f32_fast<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            cudaFuncSetAttribute(k_ink2_attn_res_f32_fast<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            return true;
        }();
        (void)attr_ok;
        if (a.Q == 1) k_ink2_attn_res_f32_fast<1><<<M, NRT, smem, st>>>(a);
        else if (a.Q == 2) k_ink2_attn_res_f32_fast<2><<<M, NRT, smem, st>>>(a);
        else if (a.Q == 3) k_ink2_attn_res_f32_fast<3><<<M, NRT, smem, st>>>(a);
        else k_ink2_attn_res_f32_fast<4><<<M, NRT, smem, st>>>(a);
        return;
    }
    k_ink2_attn_res_f32<<<M, NRT, 0, st>>>(a);
}

// Batched 3-controller MessageGate kernels (q, k, v) for M <= 64, R == 64:
// Cuts 240 kernel launches per forward pass (15 -> 5 per gated layer) while staying 100% bit-exact!
struct MsgGate3Args
{
    const float* h_f32[3];
    const float* msg_f32[3];
    const uint16_t* down_w[3];
    const uint16_t* forget_w[3];
    const uint16_t* gain_w[3];
    const uint16_t* norm_w[3];
    uint16_t* out_bf16[3];
};

__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_prep_3(MsgGate3Args a, uint16_t* __restrict__ cat_bf16, int M, int H)
{
    __shared__ float sm[NRT * 2];
    const int t = threadIdx.x;
    const int m = blockIdx.x, ctrl = blockIdx.y;
    const size_t off = static_cast<size_t>(m) * H;
    const size_t coff = (static_cast<size_t>(ctrl) * M + m) * (2 * H);
    const float* hf = a.h_f32[ctrl] + off;
    const float* mf = a.msg_f32[ctrl] + off;
    float ss[2] = {0.0f, 0.0f};
    if (H <= NRT * 9)
    {
        float hr[9], mr[9];
        int idx = 0;
        for (int c = t; c < H; c += NRT, ++idx)
        {
            const float hv = __ldg(hf + c), mv = __ldg(mf + c);
            hr[idx] = hv;
            mr[idx] = mv;
            ss[0] = add_rn(ss[0], mul_rn(hv, hv));
            ss[1] = add_rn(ss[1], mul_rn(mv, mv));
        }
        blk_sum_v<2>(ss, 2, sm, NRT);
        const float rh = rsq(add_rn(div_rn(ss[0], static_cast<float>(H)), RMS_EPS));
        const float rm = rsq(add_rn(div_rn(ss[1], static_cast<float>(H)), RMS_EPS));
        idx = 0;
        for (int c = t; c < H; c += NRT, ++idx)
        {
            cat_bf16[coff + c] = bf16(mul_rn(hr[idx], rh));
            cat_bf16[coff + H + c] = bf16(mul_rn(mr[idx], rm));
        }
        return;
    }
    for (int c = t; c < H; c += NRT)
    {
        const float hv = __ldg(hf + c), mv = __ldg(mf + c);
        ss[0] = add_rn(ss[0], mul_rn(hv, hv));
        ss[1] = add_rn(ss[1], mul_rn(mv, mv));
    }
    blk_sum_v<2>(ss, 2, sm, NRT);
    const float rh = rsq(add_rn(div_rn(ss[0], static_cast<float>(H)), RMS_EPS));
    const float rm = rsq(add_rn(div_rn(ss[1], static_cast<float>(H)), RMS_EPS));
    for (int c = t; c < H; c += NRT)
    {
        cat_bf16[coff + c] = bf16(mul_rn(__ldg(hf + c), rh));
        cat_bf16[coff + H + c] = bf16(mul_rn(__ldg(mf + c), rm));
    }
}

template <int NT>
__global__ void __launch_bounds__(32) k_ink2_msg_gate_down3_c128_splitk(const uint16_t* __restrict__ cat_bf16, MsgGate3Args a,
                                                                        int M, int K, int SK, float* __restrict__ partial)
{
    __shared__ alignas(16) uint32_t sm_w[16][36];
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * 16;
    const int ctrl = blockIdx.y;
    const int sk_id = blockIdx.z;
    const int K64 = K >> 6;
    const int k0 = ((sk_id * K64) / SK) << 6;
    const int k1 = (((sk_id + 1) * K64) / SK) << 6;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    const uint16_t* x = cat_bf16 + static_cast<size_t>(ctrl) * M * K;
    const uint16_t* w = a.down_w[ctrl];
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(t * 8 + r, M - 1)) * K + 2 * c;
    }
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = w + static_cast<size_t>(n0 + rr * 4 + lg) * K;
    for (int k = k0; k < k1; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 4]) = v;
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
            const u32 a0 = sm_w[r][p * 8 + c];
            const u32 a1 = sm_w[r + 8][p * 8 + c];
            const u32 a2 = sm_w[r][p * 8 + c + 4];
            const u32 a3 = sm_w[r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t * 8 < M) mma_bf16(acc[t], a0, a1, a2, a3, ld32(xr[t] + kp), ld32(xr[t] + kp + 8));
        }
        __syncwarp();
    }
    float* pout = partial + (static_cast<size_t>(ctrl) * SK + sk_id) * M * 64;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = t * 8 + 2 * c + (j & 1), n = n0 + r + 8 * (j >> 1);
            if (m < M && n < 64) pout[static_cast<size_t>(m) * 64 + n] = acc[t][j];
        }
}

__global__ void __launch_bounds__(256) k_ink2_msg_gate_down3_reduce(const float* __restrict__ partial, uint16_t* __restrict__ g_bf16, int M, int SK)
{
    const size_t mn = static_cast<size_t>(M) * 64;
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= 3 * mn) return;
    const int ctrl = static_cast<int>(i / mn);
    const size_t loc = i - static_cast<size_t>(ctrl) * mn;
    const float* base = partial + static_cast<size_t>(ctrl) * SK * mn + loc;
    float z = base[0];
    for (int s = 1; s < SK; ++s) z += base[static_cast<size_t>(s) * mn];
    g_bf16[i] = bf16(z);
}

template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink2_msg_gate_fg_r64_3(const uint16_t* __restrict__ g_bf16, MsgGate3Args a,
                                                                float* __restrict__ sf_out, float* __restrict__ sm_out,
                                                                int M, int H)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + warp;
    if (c >= H) return;
    const int ctrl = blockIdx.z;
    const int m0 = blockIdx.y * MAX_M;
    const int m_rem = min(MAX_M, M - m0);
    if (m_rem <= 0) return;

    const uint32_t fp = __ldg(reinterpret_cast<const uint32_t*>(a.forget_w[ctrl] + static_cast<size_t>(c) * 64) + lane);
    const uint32_t gp = __ldg(reinterpret_cast<const uint32_t*>(a.gain_w[ctrl] + static_cast<size_t>(c) * 64) + lane);
    const float f0 = bf(static_cast<uint16_t>(fp)), f1 = bf(static_cast<uint16_t>(fp >> 16));
    const float g0w = bf(static_cast<uint16_t>(gp)), g1w = bf(static_cast<uint16_t>(gp >> 16));
    const uint16_t* g_ctrl = g_bf16 + static_cast<size_t>(ctrl) * M * 64;
    float* sf_ctrl = sf_out + static_cast<size_t>(ctrl) * M * H;
    float* sm_ctrl = sm_out + static_cast<size_t>(ctrl) * M * H;

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m)
    {
        if (m < m_rem)
        {
            const uint32_t gpair = __ldg(reinterpret_cast<const uint32_t*>(g_ctrl + static_cast<size_t>(m0 + m) * 64) + lane);
            const float x0 = bf(static_cast<uint16_t>(gpair));
            const float x1 = bf(static_cast<uint16_t>(gpair >> 16));
            float sf = add_rn(mul_rn(f0, x0), mul_rn(f1, x1));
            float sm_ = add_rn(mul_rn(g0w, x0), mul_rn(g1w, x1));
            #pragma unroll
            for (int off_ = 16; off_ > 0; off_ >>= 1)
            {
                sf = add_rn(sf, __shfl_xor_sync(0xffffffffu, sf, off_));
                sm_ = add_rn(sm_, __shfl_xor_sync(0xffffffffu, sm_, off_));
            }
            if (lane == 0)
            {
                const size_t o = static_cast<size_t>(m0 + m) * H + c;
                sf_ctrl[o] = sf;
                sm_ctrl[o] = sm_;
            }
        }
    }
}

__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_finish_3(MsgGate3Args a, const float* __restrict__ sf_in,
                                                                const float* __restrict__ sm_in, int M, int H)
{
    __shared__ float sm[NRT];
    const int t = threadIdx.x;
    const int m = blockIdx.x, ctrl = blockIdx.y;
    const size_t off = static_cast<size_t>(m) * H;
    const size_t soff = (static_cast<size_t>(ctrl) * M + m) * H;
    const float* hf = a.h_f32[ctrl] + off;
    const float* mf = a.msg_f32[ctrl] + off;
    const float* sf = sf_in + soff;
    const float* sm_ = sm_in + soff;
    const uint16_t* nw = a.norm_w[ctrl];
    uint16_t* out = a.out_bf16[ctrl] + off;
    constexpr int MAX_REG = 20;
    float vals[MAX_REG];
    const bool in_reg = (H <= NRT * MAX_REG);
    float ss = 0.0f;
    int idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        const float fg = tanhf(__ldg(sf + c));
        const float mg = tanhf(__ldg(sm_ + c));
        const float v = add_rn(mul_rn(add_rn(1.0f, -fg), __ldg(hf + c)), mul_rn(mg, __ldg(mf + c)));
        if (in_reg) vals[idx] = v;
        ss = add_rn(ss, mul_rn(v, v));
    }
    const float rn = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    idx = 0;
    for (int c = t; c < H; c += NRT, ++idx)
    {
        const float v = in_reg ? vals[idx] : add_rn(mul_rn(add_rn(1.0f, -tanhf(__ldg(sf + c))), __ldg(hf + c)), mul_rn(tanhf(__ldg(sm_ + c)), __ldg(mf + c)));
        out[c] = bf16(mul_rn(bf(__ldg(nw + c)), mul_rn(v, rn)));
    }
}

__global__ void __launch_bounds__(NRT) k_ink2_msg_gate_finish_bf16_3(
    MsgGate3Args a, const uint16_t* __restrict__ sf_in, const uint16_t* __restrict__ sm_in, int M, int H)
{
    __shared__ float4 s_v[1024]; // up to H=4096 (16 KB SMEM -> 4 CTAs/SM, STACK:0!)
    __shared__ float s_red[8];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int m = blockIdx.x, ctrl = blockIdx.y;
    const int H4 = H >> 2;
    const size_t off4 = static_cast<size_t>(m) * H4;
    const size_t soff4 = (static_cast<size_t>(ctrl) * M + m) * H4;
    const float4* hf4 = reinterpret_cast<const float4*>(a.h_f32[ctrl]) + off4;
    const float4* mf4 = reinterpret_cast<const float4*>(a.msg_f32[ctrl]) + off4;
    const uint2* sf2 = reinterpret_cast<const uint2*>(sf_in) + soff4;
    const uint2* sm2 = reinterpret_cast<const uint2*>(sm_in) + soff4;
    const uint2* nw2 = reinterpret_cast<const uint2*>(a.norm_w[ctrl]);
    uint2* out2 = reinterpret_cast<uint2*>(a.out_bf16[ctrl]) + off4;

    float ss = 0.0f;
    for (int i = t; i < H4; i += NRT)
    {
        const float4 hv = __ldg(hf4 + i), mv = __ldg(mf4 + i);
        const uint2 svf = __ldg(sf2 + i), svm = __ldg(sm2 + i);
        const float2 fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svf.x));
        const float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svf.y));
        const float2 ma = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svm.x));
        const float2 mb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&svm.y));
        const float4 vv = make_float4(
            fmaf(1.0f - tanhf(fa.x), hv.x, tanhf(ma.x) * mv.x),
            fmaf(1.0f - tanhf(fa.y), hv.y, tanhf(ma.y) * mv.y),
            fmaf(1.0f - tanhf(fb.x), hv.z, tanhf(mb.x) * mv.z),
            fmaf(1.0f - tanhf(fb.y), hv.w, tanhf(mb.y) * mv.w));
        s_v[i] = vv;
        ss = fmaf(vv.x, vv.x, fmaf(vv.y, vv.y, fmaf(vv.z, vv.z, fmaf(vv.w, vv.w, ss))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (t == 0)
    {
        const float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3] + s_red[4] + s_red[5] + s_red[6] + s_red[7];
        s_red[0] = rsqrtf(tot / static_cast<float>(H) + RMS_EPS);
    }
    __syncthreads();
    const float rn = s_red[0];
    for (int i = t; i < H4; i += NRT)
    {
        const float4 vv = s_v[i];
        const uint2 nv = __ldg(nw2 + i);
        const float2 na = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nv.x));
        const float2 nb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nv.y));
        const __nv_bfloat162 oa = __floats2bfloat162_rn(na.x * vv.x * rn, na.y * vv.y * rn);
        const __nv_bfloat162 ob = __floats2bfloat162_rn(nb.x * vv.z * rn, nb.y * vv.w * rn);
        uint2 ov;
        ov.x = *reinterpret_cast<const uint32_t*>(&oa);
        ov.y = *reinterpret_cast<const uint32_t*>(&ob);
        out2[i] = ov;
    }
}

inline void msg_gate_norm_3(const MsgGate3Args& a, uint16_t* cat_bf16, uint16_t* g_bf16,
                            float* sf_buf, float* sm_buf, int M, int H, cudaStream_t st)
{
    k_ink2_msg_gate_prep_3<<<dim3(static_cast<unsigned>(M), 3), NRT, 0, st>>>(a, cat_bf16, M, H);
    const int K = 2 * H;
    const int sk = (gemm_c128_enabled() && (K % 64 == 0)) ? pick_splitk(4, K, M, 64) : 1;
    if (sk > 1 && splitk_buf() != nullptr && static_cast<size_t>(3) * sk * 64 * 64 <= splitk_cap())
    {
        float* pbuf = splitk_buf();
        if (M <= 8) k_ink2_msg_gate_down3_c128_splitk<1><<<dim3(4, 3, static_cast<unsigned>(sk)), 32, 0, st>>>(cat_bf16, a, M, K, sk, pbuf);
        else if (M <= 16) k_ink2_msg_gate_down3_c128_splitk<2><<<dim3(4, 3, static_cast<unsigned>(sk)), 32, 0, st>>>(cat_bf16, a, M, K, sk, pbuf);
        else if (M <= 32) k_ink2_msg_gate_down3_c128_splitk<4><<<dim3(4, 3, static_cast<unsigned>(sk)), 32, 0, st>>>(cat_bf16, a, M, K, sk, pbuf);
        else k_ink2_msg_gate_down3_c128_splitk<8><<<dim3(4, 3, static_cast<unsigned>(sk)), 32, 0, st>>>(cat_bf16, a, M, K, sk, pbuf);
        k_ink2_msg_gate_down3_reduce<<<cells_grid(static_cast<size_t>(3) * M * 64), 256, 0, st>>>(pbuf, g_bf16, M, sk);
    }
    else
    {
        for (int c = 0; c < 3; ++c)
            gemm_bf16(cat_bf16 + static_cast<size_t>(c) * M * K, a.down_w[c], M, K, 64, g_bf16 + static_cast<size_t>(c) * M * 64, st);
    }
    if (M > 64)
    {
        const bool msg_bf16 = msg_bf16_enabled();
        if (msg_bf16 && (H & 3) == 0 && H <= 4096)
        {
            uint16_t* sf_b = reinterpret_cast<uint16_t*>(sf_buf);
            uint16_t* sm_b = reinterpret_cast<uint16_t*>(sm_buf);
            for (int c = 0; c < 3; ++c)
            {
                gemm_bf16(g_bf16 + static_cast<size_t>(c) * M * 64, a.forget_w[c], M, 64, H, sf_b + static_cast<size_t>(c) * M * H, st);
                gemm_bf16(g_bf16 + static_cast<size_t>(c) * M * 64, a.gain_w[c], M, 64, H, sm_b + static_cast<size_t>(c) * M * H, st);
            }
            k_ink2_msg_gate_finish_bf16_3<<<dim3(static_cast<unsigned>(M), 3), NRT, 0, st>>>(a, sf_b, sm_b, M, H);
            return;
        }
        for (int c = 0; c < 3; ++c)
        {
            gemm_f32(g_bf16 + static_cast<size_t>(c) * M * 64, a.forget_w[c], M, 64, H, sf_buf + static_cast<size_t>(c) * M * H, st);
            gemm_f32(g_bf16 + static_cast<size_t>(c) * M * 64, a.gain_w[c], M, 64, H, sm_buf + static_cast<size_t>(c) * M * H, st);
        }
        k_ink2_msg_gate_finish_3<<<dim3(static_cast<unsigned>(M), 3), NRT, 0, st>>>(a, sf_buf, sm_buf, M, H);
        return;
    }
    const unsigned gx = static_cast<unsigned>((H + 3) / 4);
    if (M == 1) k_ink2_msg_gate_fg_r64_3<1><<<dim3(gx, 1, 3), 128, 0, st>>>(g_bf16, a, sf_buf, sm_buf, M, H);
    else if (M <= 4) k_ink2_msg_gate_fg_r64_3<4><<<dim3(gx, 1, 3), 128, 0, st>>>(g_bf16, a, sf_buf, sm_buf, M, H);
    else if (M <= 8) k_ink2_msg_gate_fg_r64_3<8><<<dim3(gx, 1, 3), 128, 0, st>>>(g_bf16, a, sf_buf, sm_buf, M, H);
    else if (M <= 16) k_ink2_msg_gate_fg_r64_3<16><<<dim3(gx, 1, 3), 128, 0, st>>>(g_bf16, a, sf_buf, sm_buf, M, H);
    else k_ink2_msg_gate_fg_r64_3<16><<<dim3(gx, static_cast<unsigned>((M + 15) / 16), 3), 128, 0, st>>>(g_bf16, a, sf_buf, sm_buf, M, H);
    k_ink2_msg_gate_finish_3<<<dim3(static_cast<unsigned>(M), 3), NRT, 0, st>>>(a, sf_buf, sm_buf, M, H);
}

// Opt-in experimental replay of the stock serial-rank reduction, not a global exactness guarantee.
inline bool& stock_msg_gate_enabled()
{
    static bool enabled = false;
    return enabled;
}
inline bool& msg_prefill_fuse_enabled()
{
    static bool enabled = false;
    return enabled;
}
inline void msg_gate_norm(const float* h_f32, const float* msg_f32, const uint16_t* down_w, const uint16_t* forget_w,
                          const uint16_t* gain_w, const uint16_t* norm_w, uint16_t* cat_bf16, uint16_t* g_bf16,
                          uint16_t* out_bf16, int M, int H, int R, cudaStream_t st, float* sf_buf = nullptr, float* sm_buf = nullptr)
{
    if (M < 1 || R < 1 || R > LOOP_MAX_RANK || (2 * H) % 16 != 0) throw std::runtime_error("ink2::msg_gate_norm: M, H or R");
    k_ink2_msg_gate_prep<<<M, NRT, 0, st>>>(h_f32, msg_f32, cat_bf16, H);
    gemm_bf16(cat_bf16, down_w, M, 2 * H, R, g_bf16, st);
    if (!stock_msg_gate_enabled() && R == 64 && H <= 4096)
    {
        const bool prefill_fuse = msg_prefill_fuse_enabled();
        if (sf_buf == nullptr || sm_buf == nullptr || (M > 64 && prefill_fuse))
        {
            k_ink2_msg_gate_apply_norm_r64<<<M, NRT, 0, st>>>(h_f32, msg_f32, g_bf16, forget_w, gain_w, norm_w, out_bf16, H);
            return;
        }
        if (M > 64)
        {
            if (msg_bf16_enabled() && (H & 3) == 0 && H <= 4096)
            {
                uint16_t* sf_b = reinterpret_cast<uint16_t*>(sf_buf);
                uint16_t* sm_b = reinterpret_cast<uint16_t*>(sm_buf);
                gemm_bf16(g_bf16, forget_w, M, R, H, sf_b, st);
                gemm_bf16(g_bf16, gain_w, M, R, H, sm_b, st);
                k_ink2_msg_gate_finish_bf16<<<M, NRT, 0, st>>>(h_f32, msg_f32, sf_b, sm_b, norm_w, out_bf16, H);
                return;
            }
            gemm_f32(g_bf16, forget_w, M, R, H, sf_buf, st);
            gemm_f32(g_bf16, gain_w, M, R, H, sm_buf, st);
            k_ink2_msg_gate_finish<<<M, NRT, 0, st>>>(h_f32, msg_f32, sf_buf, sm_buf, norm_w, out_bf16, H);
            return;
        }
        const unsigned gx = static_cast<unsigned>((H + 3) / 4);
        if (M == 1) k_ink2_msg_gate_fg_r64<1><<<dim3(gx, 1), 128, 0, st>>>(g_bf16, forget_w, gain_w, sf_buf, sm_buf, M, H);
        else if (M <= 4) k_ink2_msg_gate_fg_r64<4><<<dim3(gx, 1), 128, 0, st>>>(g_bf16, forget_w, gain_w, sf_buf, sm_buf, M, H);
        else if (M <= 8) k_ink2_msg_gate_fg_r64<8><<<dim3(gx, 1), 128, 0, st>>>(g_bf16, forget_w, gain_w, sf_buf, sm_buf, M, H);
        else if (M <= 16) k_ink2_msg_gate_fg_r64<16><<<dim3(gx, 1), 128, 0, st>>>(g_bf16, forget_w, gain_w, sf_buf, sm_buf, M, H);
        else k_ink2_msg_gate_fg_r64<16><<<dim3(gx, (M + 15) / 16), 128, 0, st>>>(g_bf16, forget_w, gain_w, sf_buf, sm_buf, M, H);
        k_ink2_msg_gate_finish<<<M, NRT, 0, st>>>(h_f32, msg_f32, sf_buf, sm_buf, norm_w, out_bf16, H);
        return;
    }
    k_ink2_msg_gate_apply_norm<<<M, NRT, 0, st>>>(h_f32, msg_f32, g_bf16, forget_w, gain_w, norm_w, out_bf16, H, R);
}
#endif   // __CUDACC__
}   // namespace ink2
