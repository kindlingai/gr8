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
__global__ void __launch_bounds__(NRT) k_ink2_attn_res_f32(AttnResF32Args a)
{
    __shared__ float sm[NRT * MAX_DEPTH];
    const int t = threadIdx.x, H = a.H, D = a.D;
    const size_t off = static_cast<size_t>(blockIdx.x) * H;
    float v[MAX_DEPTH], r[MAX_DEPTH];
    for (int d = 0; d < D; ++d)
    {
        float s = 0.0f;
        for (int c = t; c < H; c += NRT)
        {
            const float x = bf(a.rows[d][off + c]);
            s = add_rn(s, mul_rn(x, x));
        }
        v[d] = s;
    }
    blk_sum_v<MAX_DEPTH>(v, D, sm, NRT);
    for (int d = 0; d < D; ++d) r[d] = rsq(add_rn(div_rn(v[d], static_cast<float>(H)), RMS_EPS));
    const float root_h = sqrtf(static_cast<float>(H));
    for (int q = 0; q < a.Q; ++q)
    {
        const float* qr = a.query + static_cast<size_t>(q) * H;
        for (int d = 0; d < D; ++d)
        {
            float s = 0.0f;
            for (int c = t; c < H; c += NRT) s = add_rn(s, mul_rn(qr[c], mul_rn(bf(a.rows[d][off + c]), r[d])));
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
        float* o = a.out[q] + off;
        for (int c = t; c < H; c += NRT)
        {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s = add_rn(s, mul_rn(v[d], mul_rn(bf(a.rows[d][off + c]), r[d])));
            o[c] = s;
        }
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

inline void attn_res_f32(const AttnResF32Args& a, int M, cudaStream_t st)
{
    if (a.D < 1 || a.D > MAX_DEPTH || a.Q < 1 || a.Q > LOOP_MAX_Q || M < 1) throw std::runtime_error("ink2::attn_res_f32: D, Q or M");
    k_ink2_attn_res_f32<<<M, NRT, 0, st>>>(a);
}

inline void msg_gate_norm(const float* h_f32, const float* msg_f32, const uint16_t* down_w, const uint16_t* forget_w,
                          const uint16_t* gain_w, const uint16_t* norm_w, uint16_t* cat_bf16, uint16_t* g_bf16,
                          uint16_t* out_bf16, int M, int H, int R, cudaStream_t st)
{
    if (M < 1 || R < 1 || R > LOOP_MAX_RANK || (2 * H) % 16 != 0) throw std::runtime_error("ink2::msg_gate_norm: M, H or R");
    k_ink2_msg_gate_prep<<<M, NRT, 0, st>>>(h_f32, msg_f32, cat_bf16, H);
    gemm_bf16(cat_bf16, down_w, M, 2 * H, R, g_bf16, st);
    k_ink2_msg_gate_apply_norm<<<M, NRT, 0, st>>>(h_f32, msg_f32, g_bf16, forget_w, gain_w, norm_w, out_bf16, H, R);
}
#endif   // __CUDACC__
}   // namespace ink2
