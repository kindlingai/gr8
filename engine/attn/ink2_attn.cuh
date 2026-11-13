// Inkling-2-Small trunk attention on one rank (sm_121a). Our own code. The twin of ink2ref::Model::attention
// (engine/ref/ink2/ink2_cpu_ref.h); the shapes, weights and state are in ink2_attn.h.
//
//   attn_forward   qi, ki, vi [M][H] BF16 (the normed inputs; ki = vi = qi on a local layer) ->
//     1. kp, vp, q_raw, r_raw, gate, sink = bf16(input W^T)                                    (ink2::gemm)
//     2. kc, vc = the K / V short convolution of kp, vp (static on a local layer, dynamic on a global one)
//     3. K = bf16(kc * min(1, 2 sqrt(hd) / (|kc| + eps))), V = bf16(vc / sqrt(mean(vc^2) + eps)) per K / V head,
//        into the batch's staging rows, and on a global layer into the slot's pages
//     4. per (row, query head): q = bf16(q_raw * q_clip * min(1, 2 sqrt(hd) / (| . | + eps))), r the same over rel_dim;
//        on a global layer both times tau(pos), rounded again; score(key) = q . K / sqrt(hd) + r . rel_proj[:, dist]
//        (dist < rel_win); softmax over the visible keys with the sink logit in the denominator only;
//        og = bf16(bf16(sum p V) * sigmoid(gate))
//     5. out = og wo^T: BF16 [M][H], or the rank's fp32 partial for a sum over ranks (ink2::sum_ranks)
//   attn_commit    after the forward, for the accepted rows: the staged K / V into the ring of a local layer, and the
//                  K / V convolution inputs into hist_k / hist_v
//
// Rows are ink2::Rows (ink2_sconv.cuh): any mix of slots, each a run of consecutive positions.
//
// The forward writes no state a later step can see: a local layer reads the keys of the row's own run from the staging
// rows and older ones from the ring, and a global layer writes only page entries at the batch's own positions, which
// the next step from the accepted position writes again before it reads them. So a speculative step is forward, then
// commit with the accepted rows; nothing is saved or restored, at any position and for any number of accepted rows
// (0 = no commit at all). The AttnWork of a layer must stay as the forward left it until its commit.
//
// The core. A CTA = (row, query head, span of 256 keys), 128 threads. A thread scores one key (the dot product and
// the relative logit in index order, as the reference), the CTA folds 128 keys at a time into a running (maximum,
// sum of exp, sum of exp * V) with thread d holding component d of V. One span: the CTA merges the sink and writes og.
// More spans (a global layer past 256 positions, a local one with a window over 256): every CTA writes its partial
// and a merge kernel folds them in span order after the sink. Scores are the reference's bits given the same K; the
// softmax sums differ from it in order only.
#pragma once
#include "ink2_attn.h"
#include "../kernels/ink2_sconv.cuh"

namespace ink2
{
// Keys a query at position p reads: lo .. p.
INK2_HD int attn_lo(bool global, int window, int p) { return global || p < window ? 0 : p - (window - 1); }
// CTAs (spans of ATTN_SPAN keys) of a query at position p.
INK2_HD int attn_splits(bool global, int window, int p) { return (p - attn_lo(global, window, p) + ATTN_SPAN) / ATTN_SPAN; }
INK2_HD float attn_tau(int p, int base)
{
    return add_rn(1.0f, mul_rn(0.1f, logf(fmaxf(1.0f, div_rn(static_cast<float>(p + 1), static_cast<float>(base))))));
}
// min(1, 2 sqrt(n) / (sqrt(ss) + eps))
INK2_HD float l2_clip(float ss, int n)
{
    return fminf(1.0f, div_rn(mul_rn(2.0f, sqrtf(static_cast<float>(n))), add_rn(sqrtf(ss), RMS_EPS)));
}
// (mx, den, acc) += a partial softmax (mx2, den2, acc2): sums of exp(z - maximum) and of exp(z - maximum) * V.
INK2_HD void attn_merge(float& mx, float& den, float& acc, float mx2, float den2, float acc2)
{
    const float mn = fmaxf(mx, mx2), s1 = expf(mx - mn), s2 = expf(mx2 - mn);
    den = add_rn(mul_rn(den, s1), mul_rn(den2, s2));
    acc = add_rn(mul_rn(acc, s1), mul_rn(acc2, s2));
    mx = mn;
}
INK2_HD uint16_t attn_out(float acc, float den, uint16_t gate) { return bf16(mul_rn(rbf(div_rn(acc, den)), sigmoid(bf(gate)))); }

#ifdef __CUDACC__
constexpr int ACT = 128;   // threads of an attention CTA

// ---------------------------------------------------------------------------------------------- K / V head norms
// CTA = (row, K / V head). kv_new [M][2][nk]. pages != nullptr: also into the row's page.
__global__ void __launch_bounds__(ACT) k_ink2_kv_prep(const uint16_t* kc, const uint16_t* vc, uint16_t* kv_new, uint16_t* pages, const int* page_table, int max_pps, Rows rw, int nkv, int hd)
{
    __shared__ float sm[ACT * 2];
    const int t = threadIdx.x, m = blockIdx.x / nkv, h = blockIdx.x % nkv, nk = nkv * hd;
    const size_t o = static_cast<size_t>(m) * nk + h * hd + t;
    const float k = t < hd ? bf(kc[o]) : 0.0f, v = t < hd ? bf(vc[o]) : 0.0f;
    float ss[2] = {mul_rn(k, k), mul_rn(v, v)};
    blk_sum_v<2>(ss, 2, sm, ACT);
    if (t >= hd) return;
    const uint16_t kcell = bf16(mul_rn(k, l2_clip(ss[0], hd)));
    const uint16_t vcell = bf16(mul_rn(v, rsq(add_rn(div_rn(ss[1], static_cast<float>(hd)), RMS_EPS))));
    uint16_t* dst = kv_new + static_cast<size_t>(m) * 2 * nk + h * hd + t;
    dst[0] = kcell;
    dst[nk] = vcell;
    if (pages == nullptr) return;
    const int p = rw.pos[m];
    const size_t pg = static_cast<size_t>(page_table[static_cast<size_t>(rw.slot[m]) * max_pps + p / KV_PAGE_TOKENS]);
    uint16_t* pd = pages + (pg * 2 * KV_PAGE_TOKENS + p % KV_PAGE_TOKENS) * nk + h * hd + t;
    pd[0] = kcell;
    pd[static_cast<size_t>(KV_PAGE_TOKENS) * nk] = vcell;
}

// ---------------------------------------------------------------------------------------------- the core
struct AttnCoreArgs
{
    const uint16_t *q_raw, *r_raw, *gate, *sink;   // [M][nh_q * hd], [M][nh_q * rel_dim], [M][nh_q * hd], [M][nh_q]
    const uint16_t *q_clip, *rel_clip, *rel_proj;
    const uint16_t *kv_new, *ring, *pages;
    const int* page_table;
    Rows rw;
    uint16_t* og;      // [M][nh_q * hd]
    float* part;       // [rows of the launch][nh_q][smax][hd + 2] = maximum, sum of exp, sum of exp * V
    int m0;            // first row of the launch
    int nh_q, q0, gqa, kv0, nk, hd, rel_dim, rel_win, window, base, global, max_pps, smax;
};

// K row of position kpos for the query row `row`, K / V head lkv; voff = cells from it to the V row.
__device__ __forceinline__ const uint16_t* attn_key(const AttnCoreArgs& a, int row, int kpos, int lkv, size_t& voff)
{
    const size_t nk = static_cast<size_t>(a.nk), slot = static_cast<size_t>(a.rw.slot[row]);
    if (a.global)
    {
        const size_t pg = static_cast<size_t>(a.page_table[slot * a.max_pps + kpos / KV_PAGE_TOKENS]);
        voff = KV_PAGE_TOKENS * nk;
        return a.pages + (pg * 2 * KV_PAGE_TOKENS + kpos % KV_PAGE_TOKENS) * nk + lkv * a.hd;
    }
    const int r0 = a.rw.run0[row], p0 = a.rw.pos[r0];
    if (kpos >= p0)
    {
        voff = nk;
        return a.kv_new + static_cast<size_t>(r0 + kpos - p0) * 2 * nk + lkv * a.hd;
    }
    voff = static_cast<size_t>(a.window) * nk;
    return a.ring + (slot * 2 * a.window + kpos % a.window) * nk + lkv * a.hd;
}

// grid (rows * nh_q, smax)
__global__ void __launch_bounds__(ACT) k_ink2_attn_core(AttnCoreArgs a)
{
    __shared__ float qs[ATTN_MAX_HD], rs[ATTN_MAX_REL], es[ACT], red[ACT];
    __shared__ const uint16_t* vrow[ACT];
    const int t = threadIdx.x, hd = a.hd, rd = a.rel_dim;
    const int row = a.m0 + blockIdx.x / a.nh_q, lq = blockIdx.x % a.nh_q, sp = blockIdx.y;
    const int p = a.rw.pos[row];
    const int lo = attn_lo(a.global != 0, a.window, p), ns = attn_splits(a.global != 0, a.window, p);
    if (sp >= ns) return;
    const float tq = a.global ? attn_tau(p, a.base) : 1.0f;
    const size_t qo = (static_cast<size_t>(row) * a.nh_q + lq) * hd;

    // q of the head (thread d = component d) and r (thread 0, in index order)
    const float tmp = t < hd ? mul_rn(bf(a.q_raw[qo + t]), bf(a.q_clip[lq * hd + t])) : 0.0f;
    const float qf = l2_clip(blk_sum(mul_rn(tmp, tmp), red, ACT), hd);
    if (t < hd)
    {
        float q = rbf(mul_rn(tmp, qf));
        if (a.global) q = rbf(mul_rn(q, tq));
        qs[t] = q;
    }
    if (t == 0)
    {
        const uint16_t* rr = a.r_raw + (static_cast<size_t>(row) * a.nh_q + lq) * rd;
        float ss = 0.0f;
        for (int d = 0; d < rd; ++d)
        {
            rs[d] = mul_rn(bf(rr[d]), bf(a.rel_clip[lq * rd + d]));
            ss = add_rn(ss, mul_rn(rs[d], rs[d]));
        }
        const float rf = l2_clip(ss, rd);
        for (int d = 0; d < rd; ++d)
        {
            float r = rbf(mul_rn(rs[d], rf));
            if (a.global) r = rbf(mul_rn(r, tq));
            rs[d] = r;
        }
    }
    __syncthreads();

    const int lkv = (a.q0 + lq) / a.gqa - a.kv0;
    const int k0 = lo + sp * ATTN_SPAN, k1 = min(k0 + ATTN_SPAN, p + 1);
    const float root_hd = sqrtf(static_cast<float>(hd));
    float mx = -INFINITY, den = 0.0f, acc = 0.0f;
    for (int kb = k0; kb < k1; kb += ACT)
    {
        const int n = min(ACT, k1 - kb);
        float z = -INFINITY;
        if (t < n)
        {
            const int kpos = kb + t, dist = p - kpos;
            size_t voff;
            const uint16_t* kr = attn_key(a, row, kpos, lkv, voff);
            float s = 0.0f;
            for (int d = 0; d < hd; ++d) s = add_rn(s, mul_rn(qs[d], bf(kr[d])));
            z = div_rn(s, root_hd);
            if (dist < a.rel_win)
            {
                float b = 0.0f;
                for (int d = 0; d < rd; ++d) b = add_rn(b, mul_rn(rs[d], bf(a.rel_proj[static_cast<size_t>(d) * a.rel_win + dist])));
                z = add_rn(z, b);
            }
            vrow[t] = kr + voff;
        }
        const float mn = fmaxf(mx, blk_max(z, red, ACT)), sc = expf(mx - mn);
        es[t] = t < n ? expf(z - mn) : 0.0f;
        __syncthreads();
        den = mul_rn(den, sc);
        acc = mul_rn(acc, sc);
        for (int j = 0; j < n; ++j)
        {
            den = add_rn(den, es[j]);
            if (t < hd) acc = add_rn(acc, mul_rn(es[j], bf(vrow[j][t])));
        }
        mx = mn;
        __syncthreads();
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
    float m2 = bf(a.sink[static_cast<size_t>(row) * a.nh_q + lq]), d2 = 1.0f, a2 = 0.0f;
    attn_merge(m2, d2, a2, mx, den, acc);
    a.og[qo + t] = attn_out(a2, d2, a.gate[qo + t]);
}

// The rows with more than one span: the sink, then the partials in span order. grid rows * nh_q.
__global__ void __launch_bounds__(ACT) k_ink2_attn_merge(AttnCoreArgs a)
{
    const int t = threadIdx.x, hd = a.hd;
    const int row = a.m0 + blockIdx.x / a.nh_q, lq = blockIdx.x % a.nh_q;
    const int ns = attn_splits(a.global != 0, a.window, a.rw.pos[row]);
    if (ns <= 1 || t >= hd) return;
    const size_t qo = (static_cast<size_t>(row) * a.nh_q + lq) * hd;
    const float* pt = a.part + static_cast<size_t>(blockIdx.x) * a.smax * (hd + 2);
    float mx = bf(a.sink[static_cast<size_t>(row) * a.nh_q + lq]), den = 1.0f, acc = 0.0f;
    for (int sp = 0; sp < ns; ++sp, pt += hd + 2) attn_merge(mx, den, acc, pt[0], pt[1], pt[2 + t]);
    a.og[qo + t] = attn_out(acc, den, a.gate[qo + t]);
}

// The staged K / V of the accepted rows into the ring. A thread per (row, cell of nk).
__global__ void __launch_bounds__(256) k_ink2_attn_commit(const uint16_t* kv_new, uint16_t* ring, Rows rw, const uint8_t* acc, int M, int nk, int win)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * nk) return;
    const int m = static_cast<int>(i / nk), c = static_cast<int>(i % nk);
    if (!ring_writer(rw, acc, m, M, win)) return;
    const uint16_t* src = kv_new + static_cast<size_t>(m) * 2 * nk + c;
    uint16_t* dst = ring + (static_cast<size_t>(rw.slot[m]) * 2 * win + rw.pos[m] % win) * nk + c;
    dst[0] = src[0];
    dst[static_cast<size_t>(win) * nk] = src[nk];
}

// ---------------------------------------------------------------------------------------------- buffers
// The work buffers of one layer on one rank for up to M rows, one cudaMalloc each.
struct AttnWork
{
    int M = 0;
    size_t part_floats = 0;
    uint16_t *kp = nullptr, *vp = nullptr, *kc = nullptr, *vc = nullptr, *kv_new = nullptr;
    uint16_t *q_raw = nullptr, *r_raw = nullptr, *gate = nullptr, *sink = nullptr, *og = nullptr;
    float *dynk = nullptr, *dynv = nullptr, *part = nullptr;
    std::vector<void*> owned;

    template <class T>
    T* get(size_t n)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, (n ? n : 1) * sizeof(T)) != cudaSuccess) throw std::runtime_error("ink2::AttnWork: cudaMalloc");
        owned.push_back(p);
        return static_cast<T*>(p);
    }
    // part_floats: room for the split partials, rows * nh_q * spans * (hd + 2) floats; the forward takes as many rows
    // per launch as fit. 0 = a layer that never has more than one span.
    void alloc(const AttnGeom& g, int M_, size_t part_floats_)
    {
        M = M_;
        part_floats = part_floats_;
        const size_t m = static_cast<size_t>(M), nk = static_cast<size_t>(g.nh_kv) * g.hd, nq = static_cast<size_t>(g.nh_q) * g.hd;
        kp = get<uint16_t>(m * nk);
        vp = get<uint16_t>(m * nk);
        kc = get<uint16_t>(m * nk);
        vc = get<uint16_t>(m * nk);
        kv_new = get<uint16_t>(m * 2 * nk);
        q_raw = get<uint16_t>(m * nq);
        r_raw = get<uint16_t>(m * g.nh_q * g.rel_dim);
        gate = get<uint16_t>(m * nq);
        sink = get<uint16_t>(m * g.nh_q);
        og = get<uint16_t>(m * nq);
        dynk = get<float>(g.global ? m * g.nh_kv * SCONV_TAPS : 0);
        dynv = get<float>(g.global ? m * g.nh_kv * SCONV_TAPS : 0);
        part = get<float>(part_floats);
    }
    void release()
    {
        for (void* p : owned) cudaFree(p);
        owned.clear();
    }
};

// ---------------------------------------------------------------------------------------------- launches
inline void attn_check(const AttnGeom& g)
{
    if (g.H < 16 || g.H % 16 || g.hd < 16 || g.hd % 16 || g.hd > ATTN_MAX_HD || g.rel_dim < 1 || g.rel_dim > ATTN_MAX_REL || g.rel_win < 1 || g.gqa < 1 ||
        g.nh_q < 1 || g.nh_kv < 1 || g.window < 1 || g.log_scale_base < 1 || g.q0 / g.gqa < g.kv0 || (g.q0 + g.nh_q - 1) / g.gqa >= g.kv0 + g.nh_kv)
        throw std::runtime_error("ink2 attention: geometry");
}

// rw: device arrays of the M rows; max_pos = the largest position among them. out [M][H] BF16, or out == nullptr and
// out_f32 [M][H] = the rank's partial of the wo sum.
inline void attn_forward(const AttnGeom& g, const AttnWeights& w, const AttnState& s, AttnWork& wk, const Rows& rw, int max_pos, const uint16_t* qi,
                         const uint16_t* ki, const uint16_t* vi, int M, uint16_t* out, float* out_f32, cudaStream_t st)
{
    attn_check(g);
    if (M < 1 || M > wk.M) throw std::runtime_error("ink2::attn_forward: rows");
    if (g.global && max_pos / KV_PAGE_TOKENS >= s.max_pps) throw std::runtime_error("ink2::attn_forward: position past the slot's page table");
    const int H = g.H, hd = g.hd, nk = g.nh_kv * hd, nq = g.nh_q * hd;
    gemm_bf16(ki, w.wk, M, H, nk, wk.kp, st);
    gemm_bf16(vi, w.wv, M, H, nk, wk.vp, st);
    gemm_bf16(qi, w.wq, M, H, nq, wk.q_raw, st);
    gemm_bf16(qi, w.wr, M, H, g.nh_q * g.rel_dim, wk.r_raw, st);
    gemm_bf16(qi, w.gate, M, H, nq, wk.gate, st);
    gemm_bf16(qi, w.sink, M, H, g.nh_q, wk.sink, st);
    if (g.global)
    {
        gemm_f32(ki, w.k_wproj, M, H, g.nh_kv * SCONV_TAPS, wk.dynk, st);
        gemm_f32(vi, w.v_wproj, M, H, g.nh_kv * SCONV_TAPS, wk.dynv, st);
        sconv_dynamic(wk.kp, s.hist_k, rw, w.k_static, wk.dynk, M, nk, hd, wk.kc, st);
        sconv_dynamic(wk.vp, s.hist_v, rw, w.v_static, wk.dynv, M, nk, hd, wk.vc, st);
    }
    else
    {
        sconv_static(wk.kp, s.hist_k, rw, w.k_sconv, M, nk, wk.kc, st);
        sconv_static(wk.vp, s.hist_v, rw, w.v_sconv, M, nk, wk.vc, st);
    }
    k_ink2_kv_prep<<<M * g.nh_kv, ACT, 0, st>>>(wk.kc, wk.vc, wk.kv_new, g.global ? s.pages : nullptr, s.page_table, s.max_pps, rw, g.nh_kv, hd);

    const int smax = attn_splits(g.global, g.window, max_pos);
    const size_t per_row = static_cast<size_t>(g.nh_q) * smax * (hd + 2);
    const int mb = smax == 1 ? M : static_cast<int>(wk.part_floats / per_row < static_cast<size_t>(M) ? wk.part_floats / per_row : static_cast<size_t>(M));
    if (mb < 1) throw std::runtime_error("ink2::attn_forward: AttnWork::part_floats is too small for one row at this position");
    AttnCoreArgs a{wk.q_raw, wk.r_raw, wk.gate, wk.sink, w.q_clip, w.rel_clip, w.rel_proj, wk.kv_new, s.ring, s.pages, s.page_table, rw, wk.og, wk.part,
                   0, g.nh_q, g.q0, g.gqa, g.kv0, nk, hd, g.rel_dim, g.rel_win, g.window, g.log_scale_base, g.global ? 1 : 0, s.max_pps, smax};
    for (int m0 = 0; m0 < M; m0 += mb)
    {
        const int rows = M - m0 < mb ? M - m0 : mb;
        a.m0 = m0;
        k_ink2_attn_core<<<dim3(rows * g.nh_q, smax), ACT, 0, st>>>(a);
        if (smax > 1) k_ink2_attn_merge<<<rows * g.nh_q, ACT, 0, st>>>(a);
    }
    if (out != nullptr) gemm_bf16(wk.og, w.wo, M, nq, H, out, st);
    else gemm_f32(wk.og, w.wo, M, nq, H, out_f32, st);
}

// The rows a forward of this layer ran on, and the accepted ones (acc [M] bytes on the device; nullptr = all).
inline void attn_commit(const AttnGeom& g, const AttnState& s, const AttnWork& wk, const Rows& rw, const uint8_t* acc, int M, cudaStream_t st)
{
    const int nk = g.nh_kv * g.hd;
    if (!g.global) k_ink2_attn_commit<<<cells_grid(static_cast<size_t>(M) * nk), 256, 0, st>>>(wk.kv_new, s.ring, rw, acc, M, nk, g.window);
    sconv_advance(wk.kp, s.hist_k, rw, acc, M, nk, st);
    sconv_advance(wk.vp, s.hist_v, rw, acc, M, nk, st);
}
#endif   // __CUDACC__
}   // namespace ink2
