// Inkling-2-Small causal short convolutions over 4 tokens (sm_121a). Our own code. The twins of
// ink2ref::Model::sconv_static and sconv_dynamic (engine/ref/ink2/ink2_cpu_ref.h).
//
//   static    w [C][4] BF16, tap 3 - lag on token t - lag:    y = bf16((x[t] + sum_lag w[c][3 - lag] x[t - lag][c]) / sqrt(5))
//   dynamic   stat [4][C] BF16 (row = lag) + dyn [M][C / hd][4] fp32 (the weight_proj of the K / V head, per token):
//                                                              y = bf16((x[t] + sum_lag (stat + dyn) x[t - lag][c]) / 3)
// The sum runs over lag = 0 .. 3 while lag <= t, from 0, one product and one sum at a time: the static form gives the
// reference's bits, the dynamic form gives them from the same dyn.
//
// Rows. A batch is M rows; rows of one slot are contiguous and hold consecutive positions (a run). Per row: its slot,
// its position t, and run0 = the index of the first row of its run. Several slots, and prefill chunks of any length,
// go in one batch.
//
// State. hist [slots][3][C] BF16 is a ring by position: entry t % 3 holds the input at position t. A row reads the
// inputs of its own run from the batch and older ones from hist; a lag past position 0 is skipped, so hist needs no
// zero fill and a slot is reset by its position alone.
//
// The forward never writes hist. sconv_advance writes the inputs of the accepted rows afterwards (all rows, or a
// mask; the accepted rows of a run are its first ones). So a speculative step is: forward on M rows, then advance on
// the accepted ones; a rejected row leaves no trace and a later step from the accepted position gives the bits of a
// plain sequential run.
#pragma once
#include "ink2_norm_res.cuh"

namespace ink2
{
constexpr int SCONV_TAPS = 4, SCONV_HIST = SCONV_TAPS - 1;

struct Rows   // [M] each, host or device memory
{
    const int* slot;
    const int* pos;
    const int* run0;
};
struct RowsHost
{
    std::vector<int> slot, pos, run0;
    int max_pos = 0;
    // n rows of slot s at positions p0 .. p0 + n
    void add(int s, int p0, int n)
    {
        const int first = static_cast<int>(slot.size());
        for (int i = 0; i < n; ++i)
        {
            slot.push_back(s);
            pos.push_back(p0 + i);
            run0.push_back(first);
        }
        if (n > 0 && p0 + n - 1 > max_pos) max_pos = p0 + n - 1;
    }
    int rows() const { return static_cast<int>(slot.size()); }
    Rows view() const { return Rows{slot.data(), pos.data(), run0.data()}; }
};

// Row m keeps its state write unless a later accepted row of its run lands on the same ring entry (period rows on).
INK2_HD bool ring_writer(const Rows& rw, const uint8_t* acc, int m, int M, int period)
{
    if (acc != nullptr && !acc[m]) return false;
    const int n = m + period;
    return !(n < M && rw.run0[n] == rw.run0[m] && (acc == nullptr || acc[n]));
}

#ifdef __CUDACC__
// A thread per (row, channel). dyn == nullptr: static, w [C][4]; else dynamic, w = stat [4][C]. y is not x.
__global__ void __launch_bounds__(256) k_ink2_sconv(const uint16_t* x, const uint16_t* hist, Rows rw, const uint16_t* w, const float* dyn, int M, int C, int hd, uint16_t* y)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * C) return;
    const int m = static_cast<int>(i / C), c = static_cast<int>(i % C);
    const int t = rw.pos[m], r0 = rw.run0[m];
    const uint16_t* hs = hist + static_cast<size_t>(rw.slot[m]) * SCONV_HIST * C;
    const float* dm = dyn == nullptr ? nullptr : dyn + (static_cast<size_t>(m) * (C / hd) + c / hd) * SCONV_TAPS;
    float s = 0.0f;
    for (int lag = 0; lag < SCONV_TAPS && lag <= t; ++lag)
    {
        const float xv = bf(m - lag >= r0 ? x[i - static_cast<size_t>(lag) * C] : hs[static_cast<size_t>((t - lag) % SCONV_HIST) * C + c]);
        const float tap = dm == nullptr ? bf(w[c * SCONV_TAPS + SCONV_TAPS - 1 - lag]) : add_rn(bf(w[static_cast<size_t>(lag) * C + c]), dm[lag]);
        s = add_rn(s, mul_rn(tap, xv));
    }
    y[i] = bf16(div_rn(add_rn(bf(x[i]), s), dm == nullptr ? sqrtf(5.0f) : 3.0f));
}

// hist entry pos % 3 of the row's slot = x[row - src_base], for the accepted rows (acc [M] bytes, nullptr = all).
__global__ void __launch_bounds__(256) k_ink2_sconv_advance(const uint16_t* x, uint16_t* hist, Rows rw, const uint8_t* acc, int M, int C, int src_base)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * C) return;
    const int m = static_cast<int>(i / C), c = static_cast<int>(i % C);
    if (!ring_writer(rw, acc, m, M, SCONV_HIST)) return;
    const size_t si = static_cast<size_t>(m - src_base) * C + c;
    hist[(static_cast<size_t>(rw.slot[m]) * SCONV_HIST + rw.pos[m] % SCONV_HIST) * C + c] = x[si];
}

// Save the tail n_tail rows (starting at row0) of att, mo, kp, vp from shared prefill work buffers into compact per-layer commit buffers.
__global__ void __launch_bounds__(256) k_ink2_save_sconv_tail(
    const uint16_t* __restrict__ att_src, const uint16_t* __restrict__ mo_src,
    const uint16_t* __restrict__ kp_src, const uint16_t* __restrict__ vp_src,
    uint16_t* __restrict__ att_dst, uint16_t* __restrict__ mo_dst,
    uint16_t* __restrict__ kp_dst, uint16_t* __restrict__ vp_dst,
    int row0, int n_tail, int H, int nk)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(n_tail) * H) return;
    const int r = static_cast<int>(i / H), c = static_cast<int>(i % H);
    const size_t sh = static_cast<size_t>(row0 + r) * H + c;
    att_dst[i] = att_src[sh];
    mo_dst[i] = mo_src[sh];
    if (c < nk)
    {
        const size_t dk = static_cast<size_t>(r) * nk + c;
        const size_t sk = static_cast<size_t>(row0 + r) * nk + c;
        kp_dst[dk] = kp_src[sk];
        vp_dst[dk] = vp_src[sk];
    }
}

inline void save_sconv_tail(const uint16_t* att_src, const uint16_t* mo_src, const uint16_t* kp_src, const uint16_t* vp_src,
                            uint16_t* att_dst, uint16_t* mo_dst, uint16_t* kp_dst, uint16_t* vp_dst,
                            int row0, int n_tail, int H, int nk, cudaStream_t st)
{
    if (n_tail <= 0) return;
    k_ink2_save_sconv_tail<<<cells_grid(static_cast<size_t>(n_tail) * H), 256, 0, st>>>(
        att_src, mo_src, kp_src, vp_src, att_dst, mo_dst, kp_dst, vp_dst, row0, n_tail, H, nk);
}

inline void sconv_static(const uint16_t* x, const uint16_t* hist, const Rows& rw, const uint16_t* w, int M, int C, uint16_t* y, cudaStream_t st)
{
    k_ink2_sconv<<<cells_grid(static_cast<size_t>(M) * C), 256, 0, st>>>(x, hist, rw, w, nullptr, M, C, C, y);
}
// dyn [M][C / hd][4] fp32 = gemm_f32(norm input [M][H], weight_proj [C / hd * 4][H])
inline void sconv_dynamic(const uint16_t* x, const uint16_t* hist, const Rows& rw, const uint16_t* stat, const float* dyn, int M, int C, int hd, uint16_t* y, cudaStream_t st)
{
    k_ink2_sconv<<<cells_grid(static_cast<size_t>(M) * C), 256, 0, st>>>(x, hist, rw, stat, dyn, M, C, hd, y);
}
// x = the convolution INPUT of the forward that is being committed.
inline void sconv_advance(const uint16_t* x, uint16_t* hist, const Rows& rw, const uint8_t* acc, int M, int C, cudaStream_t st, int src_base = 0)
{
    k_ink2_sconv_advance<<<cells_grid(static_cast<size_t>(M) * C), 256, 0, st>>>(x, hist, rw, acc, M, C, src_base);
}

// Batched multi-layer trunk commit: collapses ~232 per-layer sconv_advance + attn_commit launches into 1 launch!
struct CommitLayerDesc
{
    const uint16_t *att, *mo, *kp, *vp, *kv_new;
    uint16_t *hist_a, *hist_m, *hist_k, *hist_v, *ring;
    int nk, window, is_local;
};

__global__ void __launch_bounds__(256) k_ink2_commit_all(
    const CommitLayerDesc* __restrict__ descs, Rows rw, const uint8_t* __restrict__ acc, int M, int H, int sconv_base, int kv_base)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * H) return;
    const int m = static_cast<int>(i / H), c = static_cast<int>(i % H);
    const CommitLayerDesc d = descs[blockIdx.y];
    const size_t slot = static_cast<size_t>(rw.slot[m]);
    const int pos = rw.pos[m];
    if (ring_writer(rw, acc, m, M, SCONV_HIST))
    {
        const size_t ms = static_cast<size_t>(m - sconv_base);
        const size_t ih = ms * H + c;
        const size_t h3H = (slot * SCONV_HIST + pos % SCONV_HIST) * H + c;
        d.hist_a[h3H] = d.att[ih];
        d.hist_m[h3H] = d.mo[ih];
        if (c < d.nk)
        {
            const size_t nk = static_cast<size_t>(d.nk);
            const size_t ik = ms * nk + c;
            const size_t h3k = (slot * SCONV_HIST + pos % SCONV_HIST) * nk + c;
            d.hist_k[h3k] = d.kp[ik];
            d.hist_v[h3k] = d.vp[ik];
        }
    }
    if (d.is_local && c < d.nk && ring_writer(rw, acc, m, M, d.window))
    {
        const size_t nk = static_cast<size_t>(d.nk);
        const uint16_t* src = d.kv_new + static_cast<size_t>(m - kv_base) * 2 * nk + c;
        uint16_t* dst = d.ring + (slot * 2 * d.window + pos % d.window) * nk + c;
        dst[0] = src[0];
        dst[static_cast<size_t>(d.window) * nk] = src[nk];
    }
}

inline void commit_all(const CommitLayerDesc* d_descs, int NL, const Rows& rw, const uint8_t* acc, int M, int H, cudaStream_t st, int sconv_base = 0, int kv_base = 0)
{
    if (NL <= 0 || M <= 0) return;
    k_ink2_commit_all<<<dim3(cells_grid(static_cast<size_t>(M) * H), static_cast<unsigned>(NL)), 256, 0, st>>>(d_descs, rw, acc, M, H, sconv_base, kv_base);
}
#endif   // __CUDACC__
}   // namespace ink2
