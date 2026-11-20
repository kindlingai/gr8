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

// hist entry pos % 3 of the row's slot = x[row], for the accepted rows (acc [M] bytes, nullptr = all).
__global__ void __launch_bounds__(256) k_ink2_sconv_advance(const uint16_t* x, uint16_t* hist, Rows rw, const uint8_t* acc, int M, int C)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * C) return;
    const int m = static_cast<int>(i / C), c = static_cast<int>(i % C);
    if (!ring_writer(rw, acc, m, M, SCONV_HIST)) return;
    hist[(static_cast<size_t>(rw.slot[m]) * SCONV_HIST + rw.pos[m] % SCONV_HIST) * C + c] = x[i];
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
inline void sconv_advance(const uint16_t* x, uint16_t* hist, const Rows& rw, const uint8_t* acc, int M, int C, cudaStream_t st)
{
    k_ink2_sconv_advance<<<cells_grid(static_cast<size_t>(M) * C), 256, 0, st>>>(x, hist, rw, acc, M, C);
}
#endif   // __CUDACC__
}   // namespace ink2
