// Coalesced Warp-per-Head Decode/Verify Attention for Inkling-2-Mini.
// 4 warps (128 threads) per (row, query_head, span), coalesced uint2 K/V loads
// + coalesced transposed rel_proj_t[dist * rd + lane], zero __syncthreads() inside the K/V loop.
#pragma once
#ifdef __CUDACC__

__device__ __forceinline__ float dec_warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(ACT) k_ink2_attn_core_fast(AttnCoreArgs a, const uint16_t* __restrict__ rel_proj_t)
{
    __shared__ float qs[ATTN_MAX_HD];
    __shared__ float rs[ATTN_MAX_REL];
    __shared__ float s_aw[4][ATTN_MAX_HD];
    __shared__ float s_mw[4];
    __shared__ float s_dw[4];
    __shared__ float s_red[4];

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int hd = a.hd;   // expected 128
    const int rd = a.rel_dim;
    const int row = a.m0 + blockIdx.x / a.nh_q;
    const int lq = blockIdx.x % a.nh_q;
    const int sp = blockIdx.y;
    const int p = a.rw.pos[row];
    const int lo = attn_lo(a.global != 0, a.window, p);
    const int ns = attn_splits(a.global != 0, a.window, p);
    if (sp >= ns) return;

    const float tq = a.global ? attn_tau(p, a.base) : 1.0f;
    const size_t qo = (static_cast<size_t>(row) * a.nh_q + lq) * hd;

    // 1. Parallel Q norm + clip (128 threads)
    const float qtmp = (t < hd) ? mul_rn(bf(__ldg(a.q_raw + qo + t)), bf(__ldg(a.q_clip + lq * hd + t))) : 0.0f;
    float qss = dec_warp_sum(qtmp * qtmp);
    if (lane == 0) s_red[warp] = qss;
    __syncthreads();
    const float qf = l2_clip(s_red[0] + s_red[1] + s_red[2] + s_red[3], hd);
    const float inv_root_hd = rsqrtf(static_cast<float>(hd));
    if (t < hd)
    {
        float q = rbf(mul_rn(qtmp, qf));
        if (a.global) q = rbf(mul_rn(q, tq));
        qs[t] = q * inv_root_hd;
    }

    // 2. Parallel R norm + clip
    const uint16_t* rr = a.r_raw + (static_cast<size_t>(row) * a.nh_q + lq) * rd;
    const float rtmp = (t < rd) ? mul_rn(bf(__ldg(rr + t)), bf(__ldg(a.rel_clip + lq * rd + t))) : 0.0f;
    float rss = dec_warp_sum(rtmp * rtmp);
    __syncthreads();
    if (lane == 0) s_red[warp] = rss;
    __syncthreads();
    const float rf = l2_clip(s_red[0] + s_red[1] + s_red[2] + s_red[3], rd);
    if (t < rd)
    {
        float r = rbf(mul_rn(rtmp, rf));
        if (a.global) r = rbf(mul_rn(r, tq));
        rs[t] = r;
    }
    __syncthreads();

    // Load 4 Q components per lane into registers (hd == 128 -> 32 lanes * 4 = 128)
    const float q0 = qs[lane * 4 + 0];
    const float q1 = qs[lane * 4 + 1];
    const float q2 = qs[lane * 4 + 2];
    const float q3 = qs[lane * 4 + 3];
    const float r0 = (lane < rd) ? rs[lane] : 0.0f;
    const float r1 = (lane + 32 < rd) ? rs[lane + 32] : 0.0f;

    const int lkv = (a.q0 + lq) / a.gqa - a.kv0;
    const int k0 = lo + sp * ATTN_SPAN;
    const int k1 = min(k0 + ATTN_SPAN, p + 1);
    const int rel_win = a.rel_win;
    const uint16_t* rel_proj = a.rel_proj;

    float m_w = -INFINITY;
    float d_w = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // 3. Coalesced warp-stride key/value loop — zero __syncthreads()!
    for (int kpos = k0 + warp; kpos < k1; kpos += 4)
    {
        size_t voff;
        const uint16_t* kr = attn_key(a, row, kpos, lkv, voff);
        const uint2 kv = __ldg(reinterpret_cast<const uint2*>(kr + lane * 4));
        const float2 kf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.x));
        const float2 kf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kv.y));
        float z = fmaf(q0, kf0.x, fmaf(q1, kf0.y, fmaf(q2, kf1.x, q3 * kf1.y)));
        z = dec_warp_sum(z);

        const int dist = p - kpos;
        if (dist < rel_win)
        {
            float b = 0.0f;
            if (rel_proj_t != nullptr && rd <= 32)
            {
                if (lane < rd) b = r0 * bf(__ldg(rel_proj_t + static_cast<size_t>(dist) * rd + lane));
            }
            else if (rd <= 64)
            {
                if (lane < rd) b = fmaf(r0, bf(__ldg(rel_proj + static_cast<size_t>(lane) * rel_win + dist)), b);
                if (lane + 32 < rd) b = fmaf(r1, bf(__ldg(rel_proj + static_cast<size_t>(lane + 32) * rel_win + dist)), b);
            }
            else
            {
                for (int d = lane; d < rd; d += 32)
                    b = fmaf(rs[d], bf(__ldg(rel_proj + static_cast<size_t>(d) * rel_win + dist)), b);
            }
            z += dec_warp_sum(b);
        }

        const uint2 vv = __ldg(reinterpret_cast<const uint2*>(kr + voff + lane * 4));
        const float2 vf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv.x));
        const float2 vf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vv.y));

        const float mn = fmaxf(m_w, z);
        const float s_old = (d_w > 0.0f) ? __expf(m_w - mn) : 0.0f;
        const float p_cur = __expf(z - mn);
        d_w = fmaf(d_w, s_old, p_cur);
        a0 = fmaf(p_cur, vf0.x, a0 * s_old);
        a1 = fmaf(p_cur, vf0.y, a1 * s_old);
        a2 = fmaf(p_cur, vf1.x, a2 * s_old);
        a3 = fmaf(p_cur, vf1.y, a3 * s_old);
        m_w = mn;
    }

    // 4. Write warp partials to SMEM & merge across 4 warps
    if (lane == 0)
    {
        s_mw[warp] = m_w;
        s_dw[warp] = d_w;
    }
    s_aw[warp][lane * 4 + 0] = a0;
    s_aw[warp][lane * 4 + 1] = a1;
    s_aw[warp][lane * 4 + 2] = a2;
    s_aw[warp][lane * 4 + 3] = a3;
    __syncthreads();

    float mx = -INFINITY, den = 0.0f, acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        const float dw = s_dw[w];
        if (dw > 0.0f)
        {
            const float mw = s_mw[w];
            const float aw = (t < hd) ? s_aw[w][t] : 0.0f;
            if (den == 0.0f)
            {
                mx = mw;
                den = dw;
                acc = aw;
            }
            else
            {
                attn_merge(mx, den, acc, mw, dw, aw);
            }
        }
    }

    if (ns > 1)
    {
        float* pt = a.part + (static_cast<size_t>(blockIdx.x) * a.smax + sp) * (hd + 2);
        if (t == 0)
        {
            pt[0] = mx;
            pt[1] = den;
        }
        if (t < hd) pt[2 + t] = acc;
        return;
    }
    if (t >= hd) return;
    float m2 = bf(__ldg(a.sink + static_cast<size_t>(row) * a.nh_q + lq));
    float d2 = 1.0f, a2_out = 0.0f;
    if (den > 0.0f) attn_merge(m2, d2, a2_out, mx, den, acc);
    a.og[qo + t] = attn_out(a2_out, d2, __ldg(a.gate + qo + t));
}

#endif
