// Inkling-2-Small MTP drafter kernels (sm_121a). Our own code. The twins of ink2ref::Model::mix_norm, head_norm_rope,
// mtp_context and mtp_draft (engine/ref/ink2/ink2_cpu_ref.h).
//
// The drafter reads taps of the trunk (the hidden state at the entry of a few layers and the final_norm output). Per
// drafter layer every committed position has one context K / V row made of a softmax mix of its taps. A draft runs a
// block of B rows (row 0 = the last sampled token, the others a mask row) through the drafter layers: every block row
// reads the context rows [lo, nctx) and all B rows of its block. A recurrent head then turns block row step + 1 into
// the logits of draft step.
//
//   mix_norm        out = bf16(w * (x * r)), x = sum_t taps[t] * alpha[t] in tap order (fp32, never rounded)
//   head_norm_rope  per head: n = bf16(w * (x * r)), then the half-split rotation at the row's position
//   ctx_store       the K / V rows of the accepted rows into the layer's ring (a local layer) or pages (a global one)
//   block attention softmax over the context rows then the block rows, score = q . K * (1 / sqrt(hd)); no sink, no bias,
//                   no gate; out = bf16(sum / den)
//   swiglu          bf16((g * sigmoid(g)) * u)
//   head            gate / candidate / latent of the recurrence, and logits = softcap30(bf16(base + bf16(fold_b + dot)))
//   fold            lm_head . hidden_out in fp64 (products of BF16 values are exact there), one rounding to BF16
//
// As in the trunk, sums over H or over keys run in the order of the kernel, so a result agrees with the reference to
// fp32 rounding of the sum; the fold is the reference's bits. A row's result does not depend on the rows beside it.
#pragma once
#include "../attn/ink2_attn.cuh"

namespace ink2
{
// Context rows a block reads: lo .. nctx - 1.
INK2_HD int mtp_lo(bool global, int window, int nctx) { return global || nctx < window ? 0 : nctx - window; }
// CTAs (spans of ATTN_SPAN keys) of a block row: its context rows and the B block rows.
INK2_HD int mtp_splits(bool global, int window, int nctx, int B) { return (nctx - mtp_lo(global, window, nctx) + B + ATTN_SPAN - 1) / ATTN_SPAN; }

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------------- taps
// CTA = output row b. Tap t of its source row r (rows == nullptr: r = b, else rows[b]) is taps + r * row_stride +
// t * tap_stride, H cells. alpha [ntaps], w [H]; out [rows][H].
__global__ void __launch_bounds__(NRT) k_ink2_mtp_mix_norm(const uint16_t* taps, size_t row_stride, size_t tap_stride, const int* rows, const uint16_t* alpha,
                                                            const uint16_t* w, uint16_t* out, int ntaps, int H)
{
    __shared__ float sm[NRT];
    const size_t r = rows == nullptr ? static_cast<size_t>(blockIdx.x) : static_cast<size_t>(rows[blockIdx.x]);
    const uint16_t* src = taps + r * row_stride;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < H; c += NRT)
    {
        float x = 0.0f;
        for (int t = 0; t < ntaps; ++t) x = add_rn(x, mul_rn(bf(src[static_cast<size_t>(t) * tap_stride + c]), bf(alpha[t])));
        ss = add_rn(ss, mul_rn(x, x));
    }
    const float rr = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    uint16_t* o = out + static_cast<size_t>(blockIdx.x) * H;
    for (int c = threadIdx.x; c < H; c += NRT)
    {
        float x = 0.0f;
        for (int t = 0; t < ntaps; ++t) x = add_rn(x, mul_rn(bf(src[static_cast<size_t>(t) * tap_stride + c]), bf(alpha[t])));
        o[c] = bf16(mul_rn(bf(w[c]), mul_rn(x, rr)));
    }
}

// The taps of the accepted rows' last one per run into last [slots][ntaps][H]. taps [ntaps][tap_rows][H]. A thread
// per (row, tap, cell).
__global__ void __launch_bounds__(256) k_ink2_mtp_last_taps(const uint16_t* taps, size_t tap_stride, uint16_t* last, Rows rw, const uint8_t* acc, int M, int ntaps, int H)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x, per = static_cast<size_t>(ntaps) * H;
    if (i >= static_cast<size_t>(M) * per) return;
    const int m = static_cast<int>(i / per);
    if (!ring_writer(rw, acc, m, M, 1)) return;
    const size_t tc = i % per, t = tc / H, c = tc % H;
    last[static_cast<size_t>(rw.slot[m]) * per + tc] = taps[t * tap_stride + static_cast<size_t>(m) * H + c];
}

// ---------------------------------------------------------------------------------------------- head norm + rotation
// CTA = (row, head). x [rows][heads * hd] in place; w [hd]; pos [rows]; inv_freq [hd / 2].
__global__ void __launch_bounds__(ACT) k_ink2_mtp_head_norm_rope(uint16_t* x, const uint16_t* w, const int* pos, const float* inv_freq, int heads, int hd)
{
    __shared__ float sm[ACT], nv[ATTN_MAX_HD];
    const int t = threadIdx.x, row = blockIdx.x / heads, half = hd / 2;
    uint16_t* v = x + static_cast<size_t>(blockIdx.x) * hd;
    const float xv = t < hd ? bf(v[t]) : 0.0f;
    const float r = rsq(add_rn(div_rn(blk_sum(mul_rn(xv, xv), sm, ACT), static_cast<float>(hd)), RMS_EPS));
    if (t < hd) nv[t] = rbf(mul_rn(bf(w[t]), mul_rn(xv, r)));
    __syncthreads();
    if (t >= hd) return;
    const int j = t < half ? t : t - half;
    const float ang = mul_rn(static_cast<float>(pos[row]), inv_freq[j]), cs = cosf(ang), sn = sinf(ang);
    const float a = nv[j], b = nv[j + half];
    v[t] = t < half ? bf16(add_rn(mul_rn(a, cs), -mul_rn(b, sn))) : bf16(add_rn(mul_rn(b, cs), mul_rn(a, sn)));
}

// ---------------------------------------------------------------------------------------------- context rows
// k, v [M][nk] of the rows of a forward -> the ring (pages == nullptr) or the pages, for the accepted rows. A thread
// per (row, cell).
__global__ void __launch_bounds__(256) k_ink2_mtp_ctx_store(const uint16_t* k, const uint16_t* v, uint16_t* ring, uint16_t* pages, const int* page_table,
                                                             int max_pps, Rows rw, const uint8_t* acc, int M, int nk, int win)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * nk) return;
    const int m = static_cast<int>(i / nk), c = static_cast<int>(i % nk), p = rw.pos[m];
    const size_t slot = static_cast<size_t>(rw.slot[m]);
    if (pages != nullptr)
    {
        if (acc != nullptr && !acc[m]) return;
        const size_t pg = static_cast<size_t>(page_table[slot * max_pps + p / KV_PAGE_TOKENS]);
        uint16_t* dst = pages + (pg * 2 * KV_PAGE_TOKENS + p % KV_PAGE_TOKENS) * nk + c;
        dst[0] = k[i];
        dst[static_cast<size_t>(KV_PAGE_TOKENS) * nk] = v[i];
        return;
    }
    if (!ring_writer(rw, acc, m, M, win)) return;
    uint16_t* dst = ring + (slot * 2 * win + p % win) * nk + c;
    dst[0] = k[i];
    dst[static_cast<size_t>(win) * nk] = v[i];
}

// ---------------------------------------------------------------------------------------------- the block
// CTA = block row s * B + b. Row 0 of a block: the normed embedding of the session's token; the others: mask [H].
__global__ void __launch_bounds__(NRT) k_ink2_mtp_block_init(const int* toks, const uint16_t* embed, const uint16_t* w, const uint16_t* mask, uint16_t* out, int B, int H)
{
    __shared__ float sm[NRT];
    const int s = blockIdx.x / B, b = blockIdx.x % B;
    uint16_t* o = out + static_cast<size_t>(blockIdx.x) * H;
    if (b > 0)
    {
        for (int c = threadIdx.x; c < H; c += NRT) o[c] = mask[c];
        return;
    }
    const uint16_t* x = embed + static_cast<size_t>(toks[s]) * H;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < H; c += NRT)
    {
        const float v = bf(x[c]);
        ss = add_rn(ss, mul_rn(v, v));
    }
    const float r = rsq(add_rn(div_rn(blk_sum(ss, sm, NRT), static_cast<float>(H)), RMS_EPS));
    for (int c = threadIdx.x; c < H; c += NRT) o[c] = bf16(mul_rn(bf(w[c]), mul_rn(bf(x[c]), r)));
}

struct MtpAttnArgs
{
    const uint16_t *q, *kb, *vb;       // the block: [rows][nh_q * hd] after norm and rotation, [rows][nk], [rows][nk]
    const uint16_t *ring, *pages;      // the layer's context rows
    const int* page_table;
    const int *slot, *nctx;            // [sessions]: the slot of a block and its count of context rows
    uint16_t* og;                      // [rows][nh_q * hd]
    float* part;                       // [rows of the launch][nh_q][smax][hd + 2] = maximum, sum of exp, sum of exp * V
    int m0, B, nh_q, q0, gqa, kv0, nk, hd, window, global, max_pps, smax;
};

// grid (rows of the launch * nh_q, smax). Key i of a row: context row lo + i while i < nctx - lo, then block row
// i - (nctx - lo).
__global__ void __launch_bounds__(ACT) k_ink2_mtp_attn(MtpAttnArgs a)
{
    __shared__ float qs[ATTN_MAX_HD], es[ACT], red[ACT];
    __shared__ const uint16_t* vrow[ACT];
    const int t = threadIdx.x, hd = a.hd;
    const int row = a.m0 + blockIdx.x / a.nh_q, lq = blockIdx.x % a.nh_q, sp = blockIdx.y;
    const int s = row / a.B, nctx = a.nctx[s];
    const int lo = mtp_lo(a.global != 0, a.window, nctx), nc = nctx - lo, nkeys = nc + a.B;
    const int ns = (nkeys + ATTN_SPAN - 1) / ATTN_SPAN;
    if (sp >= ns) return;
    const size_t qo = (static_cast<size_t>(row) * a.nh_q + lq) * hd, nk = static_cast<size_t>(a.nk), slot = static_cast<size_t>(a.slot[s]);
    if (t < hd) qs[t] = bf(a.q[qo + t]);
    __syncthreads();

    const int lkv = (a.q0 + lq) / a.gqa - a.kv0;
    const int k0 = sp * ATTN_SPAN, k1 = min(k0 + ATTN_SPAN, nkeys);
    const float scale = div_rn(1.0f, sqrtf(static_cast<float>(hd)));
    float mx = -INFINITY, den = 0.0f, acc = 0.0f;
    for (int kb = k0; kb < k1; kb += ACT)
    {
        const int n = min(ACT, k1 - kb);
        float z = -INFINITY;
        if (t < n)
        {
            const int i = kb + t;
            const uint16_t *kr, *vr;
            if (i >= nc)
            {
                const size_t o = (static_cast<size_t>(s) * a.B + (i - nc)) * nk + lkv * hd;
                kr = a.kb + o;
                vr = a.vb + o;
            }
            else if (a.global)
            {
                const int kpos = lo + i;
                const size_t pg = static_cast<size_t>(a.page_table[slot * a.max_pps + kpos / KV_PAGE_TOKENS]);
                kr = a.pages + (pg * 2 * KV_PAGE_TOKENS + kpos % KV_PAGE_TOKENS) * nk + lkv * hd;
                vr = kr + static_cast<size_t>(KV_PAGE_TOKENS) * nk;
            }
            else
            {
                kr = a.ring + (slot * 2 * a.window + (lo + i) % a.window) * nk + lkv * hd;
                vr = kr + static_cast<size_t>(a.window) * nk;
            }
            float d = 0.0f;
            for (int j = 0; j < hd; ++j) d = add_rn(d, mul_rn(qs[j], bf(kr[j])));
            z = mul_rn(d, scale);
            vrow[t] = vr;
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
    if (t < hd) a.og[qo + t] = bf16(div_rn(acc, den));
}

// Coalesced GQA+block-grouped MTP attention: one CTA per (session, kv_head, span) handles ALL B * gqa <= 32
// queries sharing the exact same context + block KV rows! 128-bit coalesced K loads, 24x less KV DRAM traffic,
// bit-exact FP32 arithmetic order matching k_ink2_mtp_attn.
constexpr int MTP_GRP_Q = 32;
__global__ void __launch_bounds__(ACT) k_ink2_mtp_attn_grp(MtpAttnArgs a)
{
    __shared__ float qs[MTP_GRP_Q][ATTN_MAX_HD], zs[MTP_GRP_Q][ACT];
    __shared__ alignas(16) uint16_t ks[GRP_KT][ATTN_MAX_HD + 8];
    __shared__ float mx_s[MTP_GRP_Q], den_s[MTP_GRP_Q], sc_s[MTP_GRP_Q];

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int hd = a.hd, sp = static_cast<int>(blockIdx.y), lkv = static_cast<int>(blockIdx.z);
    const int s = a.m0 / a.B + static_cast<int>(blockIdx.x);
    const int nctx = a.nctx[s];
    const int lo = mtp_lo(a.global != 0, a.window, nctx), nc = nctx - lo, nkeys = nc + a.B;
    const int ns = (nkeys + ATTN_SPAN - 1) / ATTN_SPAN;
    if (sp >= ns) return;

    const int lq0 = max(0, (a.kv0 + lkv) * a.gqa - a.q0);
    const int lq1 = min(a.nh_q, (a.kv0 + lkv + 1) * a.gqa - a.q0);
    const int gq = lq1 - lq0;
    if (gq <= 0) return;
    const int qcnt = a.B * gq;
    if (qcnt > MTP_GRP_Q) return;

    if (t < qcnt)
    {
        mx_s[t] = -INFINITY;
        den_s[t] = 0.0f;
        sc_s[t] = 1.0f;
    }
    for (int i = t; i < qcnt * hd; i += ACT)
    {
        const int q = i / hd, d = i % hd;
        const int row = s * a.B + (q / gq);
        const int lq = lq0 + (q % gq);
        qs[q][d] = bf(a.q[(static_cast<size_t>(row) * a.nh_q + lq) * hd + d]);
    }
    __syncthreads();

    float acc[MTP_GRP_Q];
    #pragma unroll
    for (int q = 0; q < MTP_GRP_Q; ++q) acc[q] = 0.0f;

    const size_t nk = static_cast<size_t>(a.nk), slot = static_cast<size_t>(a.slot[s]);
    const int k0 = sp * ATTN_SPAN, k1 = min(k0 + ATTN_SPAN, nkeys);
    const float scale = div_rn(1.0f, sqrtf(static_cast<float>(hd)));
    const int grp8 = t >> 3, sub8 = t & 7;
    const int q0w = warp * 8, qnw = min(8, max(0, qcnt - q0w));

    for (int kb = k0; kb < k1; kb += ACT)
    {
        const int n = min(ACT, k1 - kb);
        for (int kt = 0; kt < n; kt += GRP_KT)
        {
            const int nt = min(GRP_KT, n - kt);
            for (int key = grp8; key < nt; key += 16)
            {
                const int i = kb + kt + key;
                const uint16_t* kr;
                if (i >= nc)
                {
                    kr = a.kb + (static_cast<size_t>(s) * a.B + (i - nc)) * nk + lkv * hd;
                }
                else if (a.global)
                {
                    const int kpos = lo + i;
                    const size_t pg = static_cast<size_t>(a.page_table[slot * a.max_pps + kpos / KV_PAGE_TOKENS]);
                    kr = a.pages + (pg * 2 * KV_PAGE_TOKENS + kpos % KV_PAGE_TOKENS) * nk + lkv * hd;
                }
                else
                {
                    kr = a.ring + (slot * 2 * a.window + (lo + i) % a.window) * nk + lkv * hd;
                }
                if ((hd & 7) == 0 && (reinterpret_cast<uintptr_t>(kr) & 15u) == 0)
                {
                    const uint4* kr128 = reinterpret_cast<const uint4*>(kr);
                    uint4* ks128 = reinterpret_cast<uint4*>(&ks[key][0]);
                    for (int c8 = sub8; c8 < (hd >> 3); c8 += 8) ks128[c8] = __ldg(kr128 + c8);
                }
                else
                {
                    const uint32_t* kr32 = reinterpret_cast<const uint32_t*>(kr);
                    uint32_t* ks32 = reinterpret_cast<uint32_t*>(&ks[key][0]);
                    for (int c2 = sub8; c2 < (hd >> 1); c2 += 8) ks32[c2] = kr32[c2];
                }
            }
            __syncthreads();

            if (lane < nt && qnw > 0)
            {
                float sc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                const uint32_t* ks32 = reinterpret_cast<const uint32_t*>(&ks[lane][0]);
                for (int d = 0; d < hd; d += 2)
                {
                    const uint32_t pair = ks32[d >> 1];
                    const float k0f = bf(static_cast<uint16_t>(pair));
                    const float k1f = bf(static_cast<uint16_t>(pair >> 16));
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        sc8[j] = add_rn(sc8[j], mul_rn(qs[q0w + j][d], k0f));
                        sc8[j] = add_rn(sc8[j], mul_rn(qs[q0w + j][d + 1], k1f));
                    }
                }
                for (int j = 0; j < qnw; ++j) zs[q0w + j][kt + lane] = mul_rn(sc8[j], scale);
            }
            __syncthreads();
        }

        for (int q = warp; q < qcnt; q += 4)
        {
            const int j0 = lane, j1 = lane + 32, j2 = lane + 64, j3 = lane + 96;
            const float z0 = j0 < n ? zs[q][j0] : -INFINITY;
            const float z1 = j1 < n ? zs[q][j1] : -INFINITY;
            const float z2 = j2 < n ? zs[q][j2] : -INFINITY;
            const float z3 = j3 < n ? zs[q][j3] : -INFINITY;
            float m = fmaxf(fmaxf(z0, z2), fmaxf(z1, z3));
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
            m = __shfl_sync(0xffffffffu, m, 0);
            const float mx_old = mx_s[q];
            const float mn = fmaxf(mx_old, m), sc = expf(mx_old - mn);
            zs[q][j0] = j0 < n ? expf(z0 - mn) : 0.0f;
            zs[q][j1] = j1 < n ? expf(z1 - mn) : 0.0f;
            zs[q][j2] = j2 < n ? expf(z2 - mn) : 0.0f;
            zs[q][j3] = j3 < n ? expf(z3 - mn) : 0.0f;
            __syncwarp();
            if (lane == 0)
            {
                float den = mul_rn(den_s[q], sc);
                for (int j = 0; j < n; ++j) den = add_rn(den, zs[q][j]);
                den_s[q] = den;
                mx_s[q] = mn;
                sc_s[q] = sc;
            }
        }
        __syncthreads();

        if (t < hd)
        {
            for (int q = 0; q < qcnt; ++q) acc[q] = mul_rn(acc[q], sc_s[q]);
            for (int j = 0; j < n; ++j)
            {
                const int i = kb + j;
                const uint16_t* vr;
                if (i >= nc)
                {
                    vr = a.vb + (static_cast<size_t>(s) * a.B + (i - nc)) * nk + lkv * hd;
                }
                else if (a.global)
                {
                    const int kpos = lo + i;
                    const size_t pg = static_cast<size_t>(a.page_table[slot * a.max_pps + kpos / KV_PAGE_TOKENS]);
                    vr = a.pages + (pg * 2 * KV_PAGE_TOKENS + KV_PAGE_TOKENS + kpos % KV_PAGE_TOKENS) * nk + lkv * hd;
                }
                else
                {
                    vr = a.ring + (slot * 2 * a.window + a.window + (lo + i) % a.window) * nk + lkv * hd;
                }
                const float vv = bf(__ldg(vr + t));
                for (int q = 0; q < qcnt; ++q) acc[q] = add_rn(acc[q], mul_rn(zs[q][j], vv));
            }
        }
        __syncthreads();
    }

    for (int q = 0; q < qcnt; ++q)
    {
        const int row = s * a.B + (q / gq), lq = lq0 + (q % gq);
        if (ns > 1)
        {
            const size_t b = static_cast<size_t>(row - a.m0) * a.nh_q + lq;
            float* pt = a.part + (b * a.smax + sp) * (hd + 2);
            if (t == 0)
            {
                pt[0] = mx_s[q];
                pt[1] = den_s[q];
            }
            if (t < hd) pt[2 + t] = acc[q];
        }
        else if (t < hd)
        {
            a.og[(static_cast<size_t>(row) * a.nh_q + lq) * hd + t] = bf16(div_rn(acc[q], den_s[q]));
        }
    }
}

// The rows with more than one span: the partials in span order. grid rows of the launch * nh_q.
__global__ void __launch_bounds__(ACT) k_ink2_mtp_attn_merge(MtpAttnArgs a)
{
    const int t = threadIdx.x, hd = a.hd;
    const int row = a.m0 + blockIdx.x / a.nh_q, lq = blockIdx.x % a.nh_q;
    const int ns = mtp_splits(a.global != 0, a.window, a.nctx[row / a.B], a.B);
    if (ns <= 1 || t >= hd) return;
    const float* pt = a.part + static_cast<size_t>(blockIdx.x) * a.smax * (hd + 2);
    float mx = pt[0], den = pt[1], acc = pt[2 + t];
    pt += hd + 2;
    for (int sp = 1; sp < ns; ++sp, pt += hd + 2) attn_merge(mx, den, acc, pt[0], pt[1], pt[2 + t]);
    a.og[(static_cast<size_t>(row) * a.nh_q + lq) * hd + t] = bf16(div_rn(acc, den));
}

// g = bf16((g * sigmoid(g)) * u), a thread per cell.
__global__ void __launch_bounds__(256) k_ink2_mtp_swiglu(uint16_t* g, const uint16_t* u, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float x = bf(g[i]);
    g[i] = bf16(mul_rn(mul_rn(x, sigmoid(x)), bf(u[i])));
}

// ---------------------------------------------------------------------------------------------- the head
// hid [S][K][H] = block rows 1 .. K of hh [S][B][H]. A thread per cell.
__global__ void __launch_bounds__(256) k_ink2_mtp_pick_rows(const uint16_t* hh, uint16_t* hid, int S, int B, int K, int H)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x, per = static_cast<size_t>(K) * H;
    if (i >= static_cast<size_t>(S) * per) return;
    hid[i] = hh[(i / per) * B * H + H + i % per];
}

// inp [S][2 R] = state [S][R] | w1 [prev[s]][R]. A thread per cell.
__global__ void __launch_bounds__(256) k_ink2_mtp_inp(const uint16_t* state, const uint16_t* w1, const int* prev, uint16_t* inp, int S, int R)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= S * 2 * R) return;
    const int s = i / (2 * R), c = i % (2 * R);
    inp[i] = c < R ? state[s * R + c] : w1[static_cast<size_t>(prev[s]) * R + (c - R)];
}

// One step of the recurrence, a thread per (session, r). raw = ((bias + hidp) + frp) + rec with hidp [S][K][3 R] (the
// step's row), frp [S][3 R], rec [S][3 R] fp32; state, latent [S][R] BF16.
__global__ void __launch_bounds__(256) k_ink2_mtp_gate(const uint16_t* bias, const float* hidp, const float* frp, const float* rec, uint16_t* state, uint16_t* latent,
                                                        int S, int K, int step, int R)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= S * R) return;
    const int s = i / R, r = i % R, G = 3 * R;
    const float* hp = hidp + (static_cast<size_t>(s) * K + step) * G;
    const float *fp = frp + static_cast<size_t>(s) * G, *rp = rec + static_cast<size_t>(s) * G;
    float raw[3];
    for (int j = 0; j < 3; ++j)
    {
        const int g = j * R + r;
        raw[j] = rbf(add_rn(add_rn(add_rn(bf(bias[g]), hp[g]), fp[g]), rp[g]));
    }
    const float gate = rbf(sigmoid(raw[0])), cand = rbf(tanhf(raw[1]));
    state[i] = bf16(add_rn(rbf(mul_rn(gate, bf(state[i]))), rbf(mul_rn(rbf(add_rn(1.0f, -gate)), cand))));
    latent[i] = bf16(tanhf(raw[2]));
}

// lg [S][vr]: in the latent's dot with fold_w, out the logits of the step. base [S][K][vr], fold_b [vr]; row v is
// vocabulary id v0 + v. A thread per (session, v).
__global__ void __launch_bounds__(256) k_ink2_mtp_logits(const uint16_t* base, const uint16_t* fold_b, float* lg, int S, int K, int step, int vr, int v0, int valid)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(S) * vr) return;
    const size_t s = i / vr, v = i % vr;
    const float bias = rbf(add_rn(bf(fold_b[v]), lg[i]));
    const float z = rbf(add_rn(bf(base[(s * K + step) * vr + v]), bias));
    lg[i] = v0 + static_cast<int>(v) < valid ? softcap30(z) : -INFINITY;
}

// fp64 -> the nearest BF16 cell, ties to even, with one rounding (the twin of ink2ref::rbf_f64).
__device__ __forceinline__ uint16_t bf16_f64(double d)
{
    float f = static_cast<float>(d);
    u32 u = nv4::f2u(f);
    if ((u & 0x7f800000u) != 0x7f800000u && (u & 0xffffu) == 0x8000u && static_cast<double>(f) != d)
    {
        // f sits on a BF16 tie that d is not on: step off the tie toward d
        u = fabs(d) > fabs(static_cast<double>(f)) ? u + 1 : u - 1;
        f = nv4::u2f(u);
    }
    return bf16(f);
}

// CTA = a row v of the rank's lm_head [vr][H]. fold_w [vr][R] = lm_head . hidden_out weight [H][R], fold_b [vr] =
// lm_head . hidden_out bias [H]; the sum over h in order, in fp64.
__global__ void __launch_bounds__(256) k_ink2_mtp_fold(const uint16_t* lm, const uint16_t* how, const uint16_t* hob, uint16_t* fold_w, uint16_t* fold_b, int H, int R)
{
    const uint16_t* row = lm + static_cast<size_t>(blockIdx.x) * H;
    for (int r = threadIdx.x; r <= R; r += 256)
    {
        double s = 0.0;
        if (r < R)
            for (int h = 0; h < H; ++h) s = __dadd_rn(s, __dmul_rn(static_cast<double>(bf(row[h])), static_cast<double>(bf(how[static_cast<size_t>(h) * R + r]))));
        else
            for (int h = 0; h < H; ++h) s = __dadd_rn(s, __dmul_rn(static_cast<double>(bf(row[h])), static_cast<double>(bf(hob[h]))));
        if (r < R) fold_w[static_cast<size_t>(blockIdx.x) * R + r] = bf16_f64(s);
        else fold_b[blockIdx.x] = bf16_f64(s);
    }
}

// Fused TP=1 greedy MTP fold GEMV + bias + base + softcap30 + CTA-local argmax.
// Eliminates w.lg HBM materialization, k_ink2_mtp_logits, k_ink2_head_pack, and per-step host syncs!
// Grid: (NG, S), Block: 128 (4 warps = 4 vocab rows in parallel). Requires R % 8 == 0, R <= 512.
__global__ void __launch_bounds__(128) k_ink2_mtp_fold_argmax_part(
    const uint16_t* __restrict__ latent,
    const uint16_t* __restrict__ fold_w,
    const uint16_t* __restrict__ fold_b,
    const uint16_t* __restrict__ base,
    float* __restrict__ part_val,
    int* __restrict__ part_id,
    int S, int K, int step, int R, int vr, int valid)
{
    __shared__ float s_lat[512];
    __shared__ float s_val[4];
    __shared__ int s_id[4];
    const int s = blockIdx.y;
    if (s >= S) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint16_t* lat_row = latent + static_cast<size_t>(s) * R;
    for (int r = tid; r < R; r += 128) s_lat[r] = bf(lat_row[r]);
    __syncthreads();

    const int r8 = R >> 3;
    const uint16_t* base_row = base + (static_cast<size_t>(s) * K + step) * vr;
    const int vlim = vr < valid ? vr : valid;
    float best_v = -INFINITY;
    int best_id = -1;

    for (int v = blockIdx.x * 4 + warp; v < vlim; v += gridDim.x * 4)
    {
        const uint4* wrow = reinterpret_cast<const uint4*>(fold_w + static_cast<size_t>(v) * R);
        float acc = 0.0f;
        for (int i = lane; i < r8; i += 32)
        {
            const uint4 wv = __ldg(wrow + i);
            const int b0 = i * 8;
            acc = fmaf(bf(static_cast<uint16_t>(wv.x)), s_lat[b0 + 0], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.x >> 16)), s_lat[b0 + 1], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.y)), s_lat[b0 + 2], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.y >> 16)), s_lat[b0 + 3], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.z)), s_lat[b0 + 4], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.z >> 16)), s_lat[b0 + 5], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.w)), s_lat[b0 + 6], acc);
            acc = fmaf(bf(static_cast<uint16_t>(wv.w >> 16)), s_lat[b0 + 7], acc);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0)
        {
            const float bias = rbf(add_rn(bf(__ldg(fold_b + v)), acc));
            const float z = rbf(add_rn(bf(__ldg(base_row + v)), bias));
            const float lg = softcap30(z);
            if (best_id < 0 || lg > best_v || (lg == best_v && v < best_id))
            {
                best_v = lg;
                best_id = v;
            }
        }
    }
    if (lane == 0)
    {
        s_val[warp] = best_v;
        s_id[warp] = best_id;
    }
    __syncthreads();
    if (tid == 0)
    {
        float mv = s_val[0];
        int mi = s_id[0];
        for (int w = 1; w < 4; ++w)
        {
            if (s_id[w] >= 0 && (mi < 0 || s_val[w] > mv || (s_val[w] == mv && s_id[w] < mi)))
            {
                mv = s_val[w];
                mi = s_id[w];
            }
        }
        const size_t o = static_cast<size_t>(s) * gridDim.x + blockIdx.x;
        part_val[o] = mv;
        part_id[o] = mi;
    }
}

// Grid: (S), Block: 256. Reduces NG <= 256 CTA winners into m_prev[s] and draft_out[s*K + step].
__global__ void __launch_bounds__(256) k_ink2_mtp_fold_argmax_reduce(
    const float* __restrict__ part_val,
    const int* __restrict__ part_id,
    int* __restrict__ m_prev,
    int* __restrict__ draft_out,
    int NG, int K, int step)
{
    __shared__ float sv[8];
    __shared__ int si[8];
    const int s = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t base = static_cast<size_t>(s) * NG;
    float bv = -INFINITY;
    int bi = -1;
    for (int i = tid; i < NG; i += 256)
    {
        const float v = part_val[base + i];
        const int id = part_id[base + i];
        if (id >= 0 && (bi < 0 || v > bv || (v == bv && id < bi)))
        {
            bv = v;
            bi = id;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi)))
        {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0)
    {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (tid == 0)
    {
        float mv = sv[0];
        int mi = si[0];
        for (int w = 1; w < 8; ++w)
        {
            if (si[w] >= 0 && (mi < 0 || sv[w] > mv || (sv[w] == mv && si[w] < mi)))
            {
                mv = sv[w];
                mi = si[w];
            }
        }
        if (mi < 0) mi = 0;
        m_prev[s] = mi;
        draft_out[s * K + step] = mi;
    }
}

// Batched fused TP=1 greedy MTP fold GEMV + bias + base + softcap30 + CTA-local argmax (BF16 fold_w).
// Reads fold_w and fold_b ONCE for up to MAX_S sessions (8x less DRAM traffic at c=8, 4x at c=4)!
// Lanes 0..ns-1 own sessions s0..s0+ns-1 in parallel for base load + softcap30 + argmax.
// 100% bit-identical to k_ink2_mtp_fold_argmax_part!
template <int MAX_S>
__global__ void __launch_bounds__(128) k_ink2_mtp_fold_argmax_batch(
    const uint16_t* __restrict__ latent,
    const uint16_t* __restrict__ fold_w,
    const uint16_t* __restrict__ fold_b,
    const uint16_t* __restrict__ base,
    float* __restrict__ part_val,
    int* __restrict__ part_id,
    int S, int K, int step, int R, int vr, int valid)
{
    __shared__ float s_lat[MAX_S][512];
    __shared__ float s_val[4][MAX_S];
    __shared__ int s_id[4][MAX_S];
    const int s0 = blockIdx.y * MAX_S;
    const int ns = min(MAX_S, S - s0);
    if (ns <= 0) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < ns * R; i += 128)
    {
        const int si = i / R, r = i % R;
        s_lat[si][r] = bf(latent[static_cast<size_t>(s0 + si) * R + r]);
    }
    __syncthreads();

    const int r8 = R >> 3;
    const int vlim = vr < valid ? vr : valid;
    float best_v = -INFINITY;
    int best_id = -1;
    const uint16_t* my_base = (lane < ns) ? (base + (static_cast<size_t>(s0 + lane) * K + step) * vr) : nullptr;

    for (int v = blockIdx.x * 4 + warp; v < vlim; v += gridDim.x * 4)
    {
        const uint4* wrow = reinterpret_cast<const uint4*>(fold_w + static_cast<size_t>(v) * R);
        float acc[MAX_S];
        #pragma unroll
        for (int si = 0; si < MAX_S; ++si) acc[si] = 0.0f;
        for (int i = lane; i < r8; i += 32)
        {
            const uint4 wv = __ldg(wrow + i);
            const float w0 = bf(static_cast<uint16_t>(wv.x)), w1 = bf(static_cast<uint16_t>(wv.x >> 16));
            const float w2 = bf(static_cast<uint16_t>(wv.y)), w3 = bf(static_cast<uint16_t>(wv.y >> 16));
            const float w4 = bf(static_cast<uint16_t>(wv.z)), w5 = bf(static_cast<uint16_t>(wv.z >> 16));
            const float w6 = bf(static_cast<uint16_t>(wv.w)), w7 = bf(static_cast<uint16_t>(wv.w >> 16));
            const int b0 = i * 8;
            #pragma unroll
            for (int si = 0; si < MAX_S; ++si)
            {
                if (si < ns)
                {
                    float a = acc[si];
                    a = fmaf(w0, s_lat[si][b0 + 0], a);
                    a = fmaf(w1, s_lat[si][b0 + 1], a);
                    a = fmaf(w2, s_lat[si][b0 + 2], a);
                    a = fmaf(w3, s_lat[si][b0 + 3], a);
                    a = fmaf(w4, s_lat[si][b0 + 4], a);
                    a = fmaf(w5, s_lat[si][b0 + 5], a);
                    a = fmaf(w6, s_lat[si][b0 + 6], a);
                    a = fmaf(w7, s_lat[si][b0 + 7], a);
                    acc[si] = a;
                }
            }
        }
        #pragma unroll
        for (int si = 0; si < MAX_S; ++si)
        {
            if (si < ns)
            {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc[si] += __shfl_xor_sync(0xffffffffu, acc[si], off);
            }
        }
        if (lane < ns)
        {
            float my_acc = 0.0f;
            #pragma unroll
            for (int si = 0; si < MAX_S; ++si)
                if (lane == si) my_acc = acc[si];
            const float bias = rbf(add_rn(bf(__ldg(fold_b + v)), my_acc));
            const float z = rbf(add_rn(bf(__ldg(my_base + v)), bias));
            const float lg = softcap30(z);
            if (best_id < 0 || lg > best_v || (lg == best_v && v < best_id))
            {
                best_v = lg;
                best_id = v;
            }
        }
    }
    if (lane < ns)
    {
        s_val[warp][lane] = best_v;
        s_id[warp][lane] = best_id;
    }
    __syncthreads();
    if (tid < ns)
    {
        float mv = s_val[0][tid];
        int mi = s_id[0][tid];
        for (int w = 1; w < 4; ++w)
        {
            if (s_id[w][tid] >= 0 && (mi < 0 || s_val[w][tid] > mv || (s_val[w][tid] == mv && s_id[w][tid] < mi)))
            {
                mv = s_val[w][tid];
                mi = s_id[w][tid];
            }
        }
        const size_t o = static_cast<size_t>(s0 + tid) * gridDim.x + blockIdx.x;
        part_val[o] = mv;
        part_id[o] = mi;
    }
}

// Drafter-only per-row symmetric INT8 batched MTP fold argmax (2x less HBM than BF16 fold_w, exact BF16 trunk verification!).
template <int MAX_S>
__global__ void __launch_bounds__(128) k_ink2_mtp_fold_argmax_batch_i8(
    const uint16_t* __restrict__ latent,
    const int8_t* __restrict__ fold_w_i8,
    const float* __restrict__ fold_scale,
    const uint16_t* __restrict__ fold_b,
    const uint16_t* __restrict__ base,
    float* __restrict__ part_val,
    int* __restrict__ part_id,
    int S, int K, int step, int R, int vr, int valid)
{
    __shared__ float s_lat[MAX_S][512];
    __shared__ float s_val[4][MAX_S];
    __shared__ int s_id[4][MAX_S];
    const int s0 = blockIdx.y * MAX_S;
    const int ns = min(MAX_S, S - s0);
    if (ns <= 0) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < ns * R; i += 128)
    {
        const int si = i / R, r = i % R;
        s_lat[si][r] = bf(latent[static_cast<size_t>(s0 + si) * R + r]);
    }
    __syncthreads();

    const int r8 = R >> 3;
    const int vlim = vr < valid ? vr : valid;
    float best_v = -INFINITY;
    int best_id = -1;
    const uint16_t* my_base = (lane < ns) ? (base + (static_cast<size_t>(s0 + lane) * K + step) * vr) : nullptr;

    for (int v = blockIdx.x * 4 + warp; v < vlim; v += gridDim.x * 4)
    {
        const uint2* wrow = reinterpret_cast<const uint2*>(fold_w_i8 + static_cast<size_t>(v) * R);
        float acc[MAX_S];
        #pragma unroll
        for (int si = 0; si < MAX_S; ++si) acc[si] = 0.0f;
        for (int i = lane; i < r8; i += 32)
        {
            const uint2 wv = __ldg(wrow + i);
            const float w0 = static_cast<float>(static_cast<int8_t>(wv.x & 0xffu));
            const float w1 = static_cast<float>(static_cast<int8_t>((wv.x >> 8) & 0xffu));
            const float w2 = static_cast<float>(static_cast<int8_t>((wv.x >> 16) & 0xffu));
            const float w3 = static_cast<float>(static_cast<int8_t>((wv.x >> 24) & 0xffu));
            const float w4 = static_cast<float>(static_cast<int8_t>(wv.y & 0xffu));
            const float w5 = static_cast<float>(static_cast<int8_t>((wv.y >> 8) & 0xffu));
            const float w6 = static_cast<float>(static_cast<int8_t>((wv.y >> 16) & 0xffu));
            const float w7 = static_cast<float>(static_cast<int8_t>((wv.y >> 24) & 0xffu));
            const int b0 = i * 8;
            #pragma unroll
            for (int si = 0; si < MAX_S; ++si)
            {
                if (si < ns)
                {
                    float a = acc[si];
                    a = fmaf(w0, s_lat[si][b0 + 0], a);
                    a = fmaf(w1, s_lat[si][b0 + 1], a);
                    a = fmaf(w2, s_lat[si][b0 + 2], a);
                    a = fmaf(w3, s_lat[si][b0 + 3], a);
                    a = fmaf(w4, s_lat[si][b0 + 4], a);
                    a = fmaf(w5, s_lat[si][b0 + 5], a);
                    a = fmaf(w6, s_lat[si][b0 + 6], a);
                    a = fmaf(w7, s_lat[si][b0 + 7], a);
                    acc[si] = a;
                }
            }
        }
        #pragma unroll
        for (int si = 0; si < MAX_S; ++si)
        {
            if (si < ns)
            {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc[si] += __shfl_xor_sync(0xffffffffu, acc[si], off);
            }
        }
        if (lane < ns)
        {
            float my_acc = 0.0f;
            #pragma unroll
            for (int si = 0; si < MAX_S; ++si)
                if (lane == si) my_acc = acc[si];
            my_acc *= __ldg(fold_scale + v);
            const float bias = rbf(add_rn(bf(__ldg(fold_b + v)), my_acc));
            const float z = rbf(add_rn(bf(__ldg(my_base + v)), bias));
            const float lg = softcap30(z);
            if (best_id < 0 || lg > best_v || (lg == best_v && v < best_id))
            {
                best_v = lg;
                best_id = v;
            }
        }
    }
    if (lane < ns)
    {
        s_val[warp][lane] = best_v;
        s_id[warp][lane] = best_id;
    }
    __syncthreads();
    if (tid < ns)
    {
        float mv = s_val[0][tid];
        int mi = s_id[0][tid];
        for (int w = 1; w < 4; ++w)
        {
            if (s_id[w][tid] >= 0 && (mi < 0 || s_val[w][tid] > mv || (s_val[w][tid] == mv && s_id[w][tid] < mi)))
            {
                mv = s_val[w][tid];
                mi = s_id[w][tid];
            }
        }
        const size_t o = static_cast<size_t>(s0 + tid) * gridDim.x + blockIdx.x;
        part_val[o] = mv;
        part_id[o] = mi;
    }
}

// Fast R==256 uint4 half-warp (2 rows/warp) drafter-only INT8 batched MTP fold argmax.
// Single 16-lane x 16B = 256B coalesced load per vocab row + skips monotonic softcap30 (tanhf) in argmax!
template <int MAX_S>
__global__ void __launch_bounds__(128) k_ink2_mtp_fold_argmax_r256_i8(
    const uint16_t* __restrict__ latent,
    const int8_t* __restrict__ fold_w_i8,
    const float* __restrict__ fold_scale,
    const uint16_t* __restrict__ fold_b,
    const uint16_t* __restrict__ base,
    float* __restrict__ part_val,
    int* __restrict__ part_id,
    int S, int K, int step, int vr, int valid)
{
    __shared__ float s_lat[MAX_S][256];
    __shared__ float s_val[8][MAX_S];
    __shared__ int s_id[8][MAX_S];
    const int s0 = blockIdx.y * MAX_S;
    const int ns = min(MAX_S, S - s0);
    if (ns <= 0) return;
    const int tid = threadIdx.x, hwid = tid >> 4, hlane = tid & 15;
    for (int i = tid; i < ns * 256; i += 128)
        s_lat[i >> 8][i & 255] = bf(latent[static_cast<size_t>(s0 + (i >> 8)) * 256 + (i & 255)]);
    __syncthreads();

    const int vlim = vr < valid ? vr : valid;
    float best_v = -INFINITY;
    int best_id = -1;
    const uint16_t* my_base = (hlane < ns) ? (base + (static_cast<size_t>(s0 + hlane) * K + step) * vr) : nullptr;
    const int b0 = hlane * 16;

    for (int v = blockIdx.x * 8 + hwid; v < vlim; v += gridDim.x * 8)
    {
        const uint4 wv = __ldg(reinterpret_cast<const uint4*>(fold_w_i8 + static_cast<size_t>(v) * 256) + hlane);
        const float w0 = static_cast<float>(static_cast<int8_t>(wv.x & 0xffu));
        const float w1 = static_cast<float>(static_cast<int8_t>((wv.x >> 8) & 0xffu));
        const float w2 = static_cast<float>(static_cast<int8_t>((wv.x >> 16) & 0xffu));
        const float w3 = static_cast<float>(static_cast<int8_t>((wv.x >> 24) & 0xffu));
        const float w4 = static_cast<float>(static_cast<int8_t>(wv.y & 0xffu));
        const float w5 = static_cast<float>(static_cast<int8_t>((wv.y >> 8) & 0xffu));
        const float w6 = static_cast<float>(static_cast<int8_t>((wv.y >> 16) & 0xffu));
        const float w7 = static_cast<float>(static_cast<int8_t>((wv.y >> 24) & 0xffu));
        const float w8 = static_cast<float>(static_cast<int8_t>(wv.z & 0xffu));
        const float w9 = static_cast<float>(static_cast<int8_t>((wv.z >> 8) & 0xffu));
        const float w10 = static_cast<float>(static_cast<int8_t>((wv.z >> 16) & 0xffu));
        const float w11 = static_cast<float>(static_cast<int8_t>((wv.z >> 24) & 0xffu));
        const float w12 = static_cast<float>(static_cast<int8_t>(wv.w & 0xffu));
        const float w13 = static_cast<float>(static_cast<int8_t>((wv.w >> 8) & 0xffu));
        const float w14 = static_cast<float>(static_cast<int8_t>((wv.w >> 16) & 0xffu));
        const float w15 = static_cast<float>(static_cast<int8_t>((wv.w >> 24) & 0xffu));
        float acc[MAX_S];
        #pragma unroll
        for (int si = 0; si < MAX_S; ++si)
        {
            if (si < ns)
            {
                const float* sl = &s_lat[si][b0];
                float a = w0 * sl[0];
                a = fmaf(w1, sl[1], a);
                a = fmaf(w2, sl[2], a);
                a = fmaf(w3, sl[3], a);
                a = fmaf(w4, sl[4], a);
                a = fmaf(w5, sl[5], a);
                a = fmaf(w6, sl[6], a);
                a = fmaf(w7, sl[7], a);
                a = fmaf(w8, sl[8], a);
                a = fmaf(w9, sl[9], a);
                a = fmaf(w10, sl[10], a);
                a = fmaf(w11, sl[11], a);
                a = fmaf(w12, sl[12], a);
                a = fmaf(w13, sl[13], a);
                a = fmaf(w14, sl[14], a);
                a = fmaf(w15, sl[15], a);
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off, 16);
                acc[si] = a;
            }
            else acc[si] = 0.0f;
        }
        if (hlane < ns)
        {
            float my_acc = 0.0f;
            #pragma unroll
            for (int si = 0; si < MAX_S; ++si)
                if (hlane == si) my_acc = acc[si];
            my_acc *= __ldg(fold_scale + v);
            const float bias = rbf(add_rn(bf(__ldg(fold_b + v)), my_acc));
            const float z = rbf(add_rn(bf(__ldg(my_base + v)), bias));
            if (best_id < 0 || z > best_v || (z == best_v && v < best_id))
            {
                best_v = z;
                best_id = v;
            }
        }
    }
    if (hlane < ns)
    {
        s_val[hwid][hlane] = best_v;
        s_id[hwid][hlane] = best_id;
    }
    __syncthreads();
    if (tid < ns)
    {
        float mv = s_val[0][tid];
        int mi = s_id[0][tid];
        for (int hw = 1; hw < 8; ++hw)
        {
            if (s_id[hw][tid] >= 0 && (mi < 0 || s_val[hw][tid] > mv || (s_val[hw][tid] == mv && s_id[hw][tid] < mi)))
            {
                mv = s_val[hw][tid];
                mi = s_id[hw][tid];
            }
        }
        const size_t o = static_cast<size_t>(s0 + tid) * gridDim.x + blockIdx.x;
        part_val[o] = mv;
        part_id[o] = mi;
    }
}

inline void mtp_fold_argmax_launch(
    const uint16_t* latent, const uint16_t* fold_w, const int8_t* fold_w_i8, const float* fold_scale,
    const uint16_t* fold_b, const uint16_t* base,
    float* part_val, int* part_id, int* m_prev, int* draft_out,
    int NG, int S, int K, int step, int R, int vr, int valid, cudaStream_t st)
{
    if (fold_w_i8 != nullptr && fold_scale != nullptr)
    {
        if (R == 256 && ((reinterpret_cast<uintptr_t>(fold_w_i8) & 15u) == 0))
        {
            if (S == 1)
                k_ink2_mtp_fold_argmax_r256_i8<1><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, vr, valid);
            else if (S <= 4)
                k_ink2_mtp_fold_argmax_r256_i8<4><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, vr, valid);
            else
                k_ink2_mtp_fold_argmax_r256_i8<8><<<dim3(NG, (S + 7) / 8), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, vr, valid);
        }
        else if (S == 1)
            k_ink2_mtp_fold_argmax_batch_i8<1><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
        else if (S <= 4)
            k_ink2_mtp_fold_argmax_batch_i8<4><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
        else
            k_ink2_mtp_fold_argmax_batch_i8<8><<<dim3(NG, (S + 7) / 8), 128, 0, st>>>(latent, fold_w_i8, fold_scale, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
    }
    else
    {
        if (S == 1)
            k_ink2_mtp_fold_argmax_batch<1><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
        else if (S <= 4)
            k_ink2_mtp_fold_argmax_batch<4><<<dim3(NG, 1), 128, 0, st>>>(latent, fold_w, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
        else
            k_ink2_mtp_fold_argmax_batch<8><<<dim3(NG, (S + 7) / 8), 128, 0, st>>>(latent, fold_w, fold_b, base, part_val, part_id, S, K, step, R, vr, valid);
    }
    k_ink2_mtp_fold_argmax_reduce<<<S, 256, 0, st>>>(part_val, part_id, m_prev, draft_out, NG, K, step);
}

// ---------------------------------------------------------------------------------------------- launches
inline void mtp_mix_norm(const uint16_t* taps, size_t row_stride, size_t tap_stride, const int* rows, const uint16_t* alpha, const uint16_t* w, uint16_t* out,
                         int M, int ntaps, int H, cudaStream_t st)
{
    k_ink2_mtp_mix_norm<<<M, NRT, 0, st>>>(taps, row_stride, tap_stride, rows, alpha, w, out, ntaps, H);
}
inline void mtp_head_norm_rope(uint16_t* x, const uint16_t* w, const int* pos, const float* inv_freq, int M, int heads, int hd, cudaStream_t st)
{
    if (hd < 2 || hd % 2 || hd > ATTN_MAX_HD) throw std::runtime_error("ink2::mtp_head_norm_rope: head_dim");
    k_ink2_mtp_head_norm_rope<<<M * heads, ACT, 0, st>>>(x, w, pos, inv_freq, heads, hd);
}

// Set once at runtime startup; standalone kernel users retain the default dispatch.
inline bool& mtp_attn_grp_enabled()
{
    static bool on = true;
    return on;
}

// The block attention of MB = S * B rows. part: room for the split partials, part_floats floats; the rows go in as many
// launches as that takes. nctx_max = the largest context count among the sessions.
inline void mtp_attn(MtpAttnArgs a, int MB, int nctx_max, float* part, size_t part_floats, cudaStream_t st)
{
    a.smax = mtp_splits(a.global != 0, a.window, nctx_max, a.B);
    a.part = part;
    const size_t per_row = static_cast<size_t>(a.nh_q) * a.smax * (a.hd + 2);
    const int mb = a.smax == 1 ? MB : static_cast<int>(part_floats / per_row < static_cast<size_t>(MB) ? part_floats / per_row : static_cast<size_t>(MB));
    if (mb < 1) throw std::runtime_error("ink2::mtp_attn: the partial buffer is too small for one row at this position");
    const bool grp_on = mtp_attn_grp_enabled();
    const int nh_kv = a.nk / a.hd;
    int max_gq = 1;
    for (int lkv = 0; lkv < nh_kv; ++lkv)
    {
        const int lq0 = std::max(0, (a.kv0 + lkv) * a.gqa - a.q0);
        const int lq1 = std::min(a.nh_q, (a.kv0 + lkv + 1) * a.gqa - a.q0);
        if (lq1 - lq0 > max_gq) max_gq = lq1 - lq0;
    }
    const bool use_grp = grp_on && a.B * max_gq <= MTP_GRP_Q && a.hd <= ATTN_MAX_HD && mb >= a.B && (MB % a.B == 0);
    if (use_grp)
    {
        const int mb_grp = std::max(1, mb / a.B) * a.B;
        for (int m0 = 0; m0 < MB; m0 += mb_grp)
        {
            const int rows = MB - m0 < mb_grp ? MB - m0 : mb_grp;
            a.m0 = m0;
            k_ink2_mtp_attn_grp<<<dim3(rows / a.B, a.smax, nh_kv), ACT, 0, st>>>(a);
            if (a.smax > 1) k_ink2_mtp_attn_merge<<<rows * a.nh_q, ACT, 0, st>>>(a);
        }
        return;
    }
    for (int m0 = 0; m0 < MB; m0 += mb)
    {
        const int rows = MB - m0 < mb ? MB - m0 : mb;
        a.m0 = m0;
        k_ink2_mtp_attn<<<dim3(rows * a.nh_q, a.smax), ACT, 0, st>>>(a);
        if (a.smax > 1) k_ink2_mtp_attn_merge<<<rows * a.nh_q, ACT, 0, st>>>(a);
    }
}
#endif   // __CUDACC__
}   // namespace ink2
