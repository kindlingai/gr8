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

// The block attention of MB = S * B rows. part: room for the split partials, part_floats floats; the rows go in as many
// launches as that takes. nctx_max = the largest context count among the sessions.
inline void mtp_attn(MtpAttnArgs a, int MB, int nctx_max, float* part, size_t part_floats, cudaStream_t st)
{
    a.smax = mtp_splits(a.global != 0, a.window, nctx_max, a.B);
    a.part = part;
    const size_t per_row = static_cast<size_t>(a.nh_q) * a.smax * (a.hd + 2);
    const int mb = a.smax == 1 ? MB : static_cast<int>(part_floats / per_row < static_cast<size_t>(MB) ? part_floats / per_row : static_cast<size_t>(MB));
    if (mb < 1) throw std::runtime_error("ink2::mtp_attn: the partial buffer is too small for one row at this position");
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
