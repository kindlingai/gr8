// Fast 2-bit block-32 screen + exact FP8/BF16 rescore for Qwen3.8-Flash-Next MTP draft (top-64) and trunk (top-256) lm_head.
// Requests asking for logprobs bypass trunk_head2 to exact full-vocab FP8 lm_head for unbiased full-vocab LSE.
//
// Format for HID = 2560 (NCH = 40 chunks of 64 weights, NG = 80 groups of 32 weights):
//   - codes[vr, 40]  uint4   (16 B/chunk = 64 x 2-bit codes, 640 B/row)
//   - scales[vr, 40] __half2 (2 x FP16 scales per chunk, 160 B/row)
//   - Total: 800 B/row = 198.7 MB on TP=1 (3.40x smaller than FP8 675 MB, 6.40x smaller than BF16 1271 MB!)
//
// Value per group of 32 weights: w ~ (2*q - 3) * (0.5 * s), q in {0, 1, 2, 3}, s = fp16(amax(|w|) / 1.5).
// Matches engine/kernels/head2.cuh's proven 2-bit encoding + FP16x2 branchless unpack.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "../model_geom_qw38.h"
#include "qw38_fp8.cuh"
#include "qw38_head.cuh"

namespace qw38 {

struct HeadArgmax {
    float logit;
    int id;
};

struct MtpHead2 {
    static constexpr int NCH = geom::HID / 64; // 40
    static constexpr int NG = geom::HID / 32;  // 80
    static constexpr int SL = 64;              // exact-rescored candidates per row (MTP draft)
    static constexpr int SL_TRUNK = 256;       // exact-rescored candidates per row (trunk decode/verify)
    static constexpr int BINS = 2048;
    static constexpr float RANGE = 32.0f;

    uint4* codes = nullptr;     // [vr * NCH]
    __half2* scales = nullptr;  // [vr * NCH]
    int vr = 0;
    int v0 = 0;

    ~MtpHead2() {
        if (codes) cudaFree(codes);
        if (scales) cudaFree(scales);
    }
    MtpHead2() = default;
    MtpHead2(const MtpHead2&) = delete;
    MtpHead2& operator=(const MtpHead2&) = delete;
    MtpHead2(MtpHead2&& o) noexcept : codes(o.codes), scales(o.scales), vr(o.vr), v0(o.v0) {
        o.codes = nullptr; o.scales = nullptr; o.vr = 0; o.v0 = 0;
    }
    MtpHead2& operator=(MtpHead2&& o) noexcept {
        if (this != &o) {
            if (codes) cudaFree(codes);
            if (scales) cudaFree(scales);
            codes = o.codes; scales = o.scales; vr = o.vr; v0 = o.v0;
            o.codes = nullptr; o.scales = nullptr; o.vr = 0; o.v0 = 0;
        }
        return *this;
    }
    bool ready() const { return codes != nullptr && scales != nullptr && vr > 0; }
};

struct MtpHead2Scratch {
    int* cand = nullptr;       // [max_M * SL_TRUNK]
    float* cl = nullptr;       // [max_M * SL_TRUNK]
    double* tail_dsum = nullptr; // [max_M]
    float* screen_max = nullptr; // [max_M]
};

// Boot conversion: BF16 lm_head[vr, 2560] -> 2-bit codes[vr, 40] + scales[vr, 40].
// One thread per (row, chunk of 64). Deterministic.
__global__ void __launch_bounds__(256) k_qw38_lm_head_to_i2(
    const __nv_bfloat16* __restrict__ W,
    uint4* __restrict__ codes,
    __half2* __restrict__ scales,
    int vr) {
    constexpr int NCH = MtpHead2::NCH;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(vr) * NCH) return;
    const int row = static_cast<int>(id / NCH);
    const int c = static_cast<int>(id % NCH);
    const __nv_bfloat16* src = W + static_cast<size_t>(row) * geom::HID + c * 64;

    uint32_t w[4] = {0u, 0u, 0u, 0u};
    __half s16[2];
    #pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
        float v[32];
        float amax = 0.0f;
        const uint4* p = reinterpret_cast<const uint4*>(src + gg * 32);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 qv = p[j];
            const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&qv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __bfloat1622float2(bp[k]);
                v[8 * j + 2 * k] = f.x;
                v[8 * j + 2 * k + 1] = f.y;
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        s16[gg] = __float2half_rn(__fdiv_rn(amax, 1.5f));
        const float s = __half2float(s16[gg]);
        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            const uint32_t q = (s != 0.0f)
                ? static_cast<uint32_t>(min(max(__float2int_rd(__fadd_rn(__fdiv_rn(v[kk], s), 2.0f)), 0), 3))
                : 2u;
            const int e = gg * 32 + kk;
            const int word = e >> 4;
            const int i = e & 15;
            w[word] |= q << (2 * (i >> 1) + 16 * (i & 1));
        }
    }
    codes[id] = make_uint4(w[0], w[1], w[2], w[3]);
    scales[id] = __halves2half2(s16[0], s16[1]);
}

inline size_t build_mtp_head2(const __nv_bfloat16* W, int vr, int v0, MtpHead2& out, cudaStream_t st) {
    constexpr int NCH = MtpHead2::NCH;
    const size_t cb = static_cast<size_t>(vr) * NCH * sizeof(uint4);
    const size_t sb = static_cast<size_t>(vr) * NCH * sizeof(__half2);
    void* cp = nullptr;
    void* sp = nullptr;
    if (cudaMalloc(&cp, cb) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaMalloc(&sp, sb) != cudaSuccess) { (void)cudaGetLastError(); cudaFree(cp); return 0; }
    out.codes = static_cast<uint4*>(cp);
    out.scales = static_cast<__half2*>(sp);
    out.vr = vr;
    out.v0 = v0;
    const long long n = static_cast<long long>(vr) * NCH;
    k_qw38_lm_head_to_i2<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(W, out.codes, out.scales, vr);
    return cb + sb;
}

__device__ __forceinline__ int head2_swz(int u) { return u ^ ((u >> 3) & 7); }

// Stage 1: Persistent 2-bit screen GEMV over all `vr` vocab rows for M <= 8.
// 8 lanes per vocab row (4 rows/warp, 32 rows/CTA): 40 chunks / 8 lanes = 5 exact coalesced 128B-burst iterations!
// Caches x[0..M-1, 0..2559] scaled by 0.25f as FP16 in bank-conflict-free shared memory ONCE per CTA.
template <int MAX_M>
__global__ void __launch_bounds__(256) k_qw38_mtp_head2_screen(
    const uint4* __restrict__ codes,
    const __half2* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y,
    int M, int vr) {
    constexpr int NCH = MtpHead2::NCH;  // 40
    constexpr int K8 = geom::HID / 8;   // 320
    __shared__ uint4 xs4[MAX_M][K8];    // MAX_M * 5 KB (<= 40 KB for MAX_M=8)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;          // 0..7
    const int lane = tid & 31;          // 0..31
    const int sub_r = lane >> 3;        // 0..3 (4 vocab rows per warp)
    const int sub_l = lane & 7;         // 0..7 (8 lanes per vocab row)

    const int tot_u4 = M * K8;
    for (int i = tid; i < tot_u4; i += 256) {
        const int m = i / K8;
        const int u = i % K8;
        const uint4 xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID)[u];
        const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        uint4 hv;
        __half2* hp = reinterpret_cast<__half2*>(&hv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(bp[k]);
            hp[k] = __floats2half2_rn(f.x * 0.25f, f.y * 0.25f);
        }
        xs4[m][head2_swz(u)] = hv;
    }
    __syncthreads();

    const __half2 off = __float2half2_rn(1027.0f);
    for (int v = blockIdx.x * 32 + warp * 4 + sub_r; v < vr; v += gridDim.x * 32) {
        const uint4* crow = codes + static_cast<size_t>(v) * NCH;
        const __half2* srow = scales + static_cast<size_t>(v) * NCH;

        float acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int c = j * 8 + sub_l;
            const uint4 cw = __ldcs(crow + c);
            const __half2 sc = __ldcs(srow + c);
            const uint32_t wd[4] = {cw.x, cw.y, cw.z, cw.w};

            #pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                __half2 p[MAX_M];
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) p[m] = __float2half2_rn(0.0f);

                #pragma unroll
                for (int q = 2 * gg; q < 2 * gg + 2; ++q) {
                    #pragma unroll
                    for (int h8 = 0; h8 < 2; ++h8) {
                        const int u = 8 * c + 2 * q + h8;
                        const int su = head2_swz(u);
                        __half2 wv[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int sh = 2 * (4 * h8 + i);
                            const uint32_t b = ((sh ? (wd[q] >> (sh - 1)) : (wd[q] << 1)) & 0x00060006u) | 0x64006400u;
                            wv[i] = __hsub2(*reinterpret_cast<const __half2*>(&b), off);
                        }
                        #pragma unroll
                        for (int m = 0; m < MAX_M; ++m) {
                            if (m < M) {
                                const uint4 xv = xs4[m][su];
                                const __half2* xp = reinterpret_cast<const __half2*>(&xv);
                                p[m] = __hfma2(wv[0], xp[0], p[m]);
                                p[m] = __hfma2(wv[1], xp[1], p[m]);
                                p[m] = __hfma2(wv[2], xp[2], p[m]);
                                p[m] = __hfma2(wv[3], xp[3], p[m]);
                            }
                        }
                    }
                }
                const float s = 2.0f * (gg ? __high2float(sc) : __low2float(sc));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float2 f = __half22float2(p[m]);
                        acc[m] = fmaf(s, f.x + f.y, acc[m]);
                    }
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], 4);
                acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], 2);
                acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], 1);
                if (sub_l == 0) y[static_cast<size_t>(m) * vr + v] = acc[m];
            }
        }
    }
}

// Stage 2: Deterministic top-SL selection from screen logits y[M, vr] (1 CTA/row, 1024 threads).
// When CALC_TAIL is true, also accumulates sum_{v not in top-SL} exp(y[v] - screen_max) for exact full-vocab LSE.
template <int SL, bool CALC_TAIL = false>
__global__ void __launch_bounds__(1024) k_qw38_head2_top(
    const float* __restrict__ y,
    int vr,
    int* __restrict__ cand,
    double* __restrict__ tail_dsum = nullptr,
    float* __restrict__ screen_max = nullptr) {
    constexpr int BINS = MtpHead2::BINS;
    constexpr float RANGE = MtpHead2::RANGE;

    __shared__ unsigned hist[BINS];
    __shared__ float s_mx[32];
    __shared__ float s_mn[32];
    __shared__ double s_tail[32];
    __shared__ int s_cut, s_lt;
    __shared__ int s_n[2][1024];

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int w = t >> 5, l = t & 31;
    const float* row = y + static_cast<size_t>(m) * vr;

    const int per = (vr + 1023) / 1024;
    const int i0 = min(t * per, vr);
    const int i1 = min(i0 + per, vr);

    float mx = -INFINITY, mn = INFINITY;
    for (int i = i0; i < i1; ++i) {
        const float v = row[i];
        if (isfinite(v)) {
            mx = fmaxf(mx, v);
            mn = fminf(mn, v);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
    }
    if (l == 0) { s_mx[w] = mx; s_mn[w] = mn; }
    for (int i = t; i < BINS; i += 1024) hist[i] = 0u;
    __syncthreads();

    mx = s_mx[0];
    mn = s_mn[0];
    #pragma unroll
    for (int i = 1; i < 32; ++i) {
        mx = fmaxf(mx, s_mx[i]);
        mn = fminf(mn, s_mn[i]);
    }

    // Adapt histogram range to peaked logits (up to 256 logit units) so rank <= SL never collapses into overflow bin BINS - 1.
    const float span = (isfinite(mx) && isfinite(mn) && mx > mn) ? (mx - mn) * 1.001f + 1e-3f : RANGE;
    const float range = fmaxf(RANGE, fminf(span, 256.0f));
    const float inv_step = static_cast<float>(BINS - 1) / range;
    auto bin = [&](float v) -> int {
        if (!isfinite(v)) return BINS - 1;
        const float b = (mx - v) * inv_step;
        return (b < static_cast<float>(BINS - 1)) ? max(0, static_cast<int>(b)) : (BINS - 1);
    };
    for (int i = i0; i < i1; ++i) atomicAdd(&hist[bin(row[i])], 1u);
    __syncthreads();

    if (w == 0) {
        constexpr int PER = BINS / 32;
        unsigned c = 0;
        for (int i = 0; i < PER; ++i) c += hist[l * PER + i];
        unsigned inc = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned z = __shfl_up_sync(0xffffffffu, inc, o);
            if (l >= o) inc += z;
        }
        const unsigned over = __ballot_sync(0xffffffffu, inc >= static_cast<unsigned>(SL));
        const int L = over ? (__ffs(over) - 1) : 31;
        if (l == L) {
            unsigned cum = inc - c;
            int cut = L * PER + PER - 1;
            for (int i = 0; i < PER; ++i) {
                if (cum + hist[L * PER + i] >= static_cast<unsigned>(SL)) { cut = L * PER + i; break; }
                cum += hist[L * PER + i];
            }
            s_cut = cut;
            s_lt = static_cast<int>(cum);
        }
    }
    __syncthreads();

    const int cut = s_cut, lt = s_lt;
    int nlt = 0, neq = 0;
    for (int i = i0; i < i1; ++i) {
        const int b = bin(row[i]);
        nlt += (b < cut);
        neq += (b == cut);
    }
    s_n[0][t] = nlt;
    s_n[1][t] = neq;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int a = (t >= o) ? s_n[0][t - o] : 0;
        const int b = (t >= o) ? s_n[1][t - o] : 0;
        __syncthreads();
        s_n[0][t] += a;
        s_n[1][t] += b;
        __syncthreads();
    }
    int plt = s_n[0][t] - nlt;
    int peq = lt + s_n[1][t] - neq;
    int* out = cand + static_cast<size_t>(m) * SL;
    double tail_a = 0.0;
    for (int i = i0; i < i1; ++i) {
        const float v = row[i];
        const int b = bin(v);
        if (b < cut) out[plt++] = i;
        else if (b == cut && peq < SL) out[peq++] = i;
        else if constexpr (CALC_TAIL) {
            if (ink2::head_key(v) && (mx - v) < 20.0f) {
                tail_a += static_cast<double>(expf(v - mx));
            }
        }
    }
    if constexpr (CALC_TAIL) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tail_a += __shfl_xor_sync(0xffffffffu, tail_a, o);
        if (l == 0) s_tail[w] = tail_a;
        __syncthreads();
        if (w == 0) {
            double ts = s_tail[l];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ts += __shfl_xor_sync(0xffffffffu, ts, o);
            if (l == 0) {
                tail_dsum[m] = ts;
                screen_max[m] = mx;
            }
        }
    }
}

// Stage 3a (MTP draft): Exact FP8/MXFP8 rescore of SL=64 candidates (Grid dim3(M, 8), Block 256).
template <bool MX_SC = false>
__global__ void __launch_bounds__(256) k_qw38_mtp_head2_rescore_fp8(
    const uint8_t* __restrict__ Bq,
    const __half* __restrict__ Bsc,
    const uint8_t* __restrict__ Bmx,
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ cand,
    float* __restrict__ y,
    float* __restrict__ cl,
    int vr, bool write_y) {
    const int m = blockIdx.x;
    const int j = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int v = cand[static_cast<size_t>(m) * MtpHead2::SL + j];

    constexpr int K16 = geom::HID >> 4; // 160
    constexpr int NB = geom::HID >> 5;  // 80
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(v) * geom::HID);
    const size_t sbase = static_cast<size_t>(v) * NB;
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);

    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const uint4 wv = __ldcs(qrow + i);
        const float sc = MX_SC
            ? __uint_as_float(static_cast<uint32_t>(__ldcs(Bmx + sbase + (i >> 1))) << 23)
            : __half2float(__ldcs(Bsc + sbase + (i >> 1)));
        float2 wf[8];
        fp8x16_unpack_raw(wv, wf);
        const float d = dot_bf16x16_f2x8(xm[2 * i], xm[2 * i + 1], wf);
        acc = fmaf(d, sc, acc);
    }
    const float tot = fp8_warp_sum(acc);
    if (lane == 0) {
        cl[static_cast<size_t>(m) * MtpHead2::SL + j] = tot;
        if (write_y) y[static_cast<size_t>(m) * vr + v] = tot;
    }
}

// Stage 3b (Trunk): Bit-exact FP8/MXFP8 rescore of SL=256 candidates matching k_fp8_gemv_1warp arithmetic identically!
// Grid: dim3(M, SL_TRUNK / 8 = 32), Block: 256 (8 candidates/CTA).
template <bool MX_SC = false>
__global__ void __launch_bounds__(256) k_qw38_head2_rescore_fp8_exact(
    const uint8_t* __restrict__ Bq,
    const __half* __restrict__ Bsc,
    const uint8_t* __restrict__ Bmx,
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ cand,
    float* __restrict__ cl) {
    const int m = blockIdx.x;
    const int j = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int v = cand[static_cast<size_t>(m) * MtpHead2::SL_TRUNK + j];

    constexpr int K16 = geom::HID >> 4; // 160
    constexpr int NB = geom::HID >> 5;  // 80
    const uint4* qrow = reinterpret_cast<const uint4*>(Bq + static_cast<size_t>(v) * geom::HID);
    const size_t sbase = static_cast<size_t>(v) * NB;
    const uint4* xm = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * geom::HID);

    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const uint4 wv = __ldg(qrow + i);
        const float sc = load_fp8_scale<MX_SC>(Bsc, Bmx, sbase + (i >> 1));
        float2 wf[8];
        fp8x16_unpack(wv, sc, wf);
        acc += dot_bf16x16_f2x8(xm[2 * i], xm[2 * i + 1], wf);
    }
    const float tot = fp8_warp_sum(acc);
    if (lane == 0) {
        cl[static_cast<size_t>(m) * MtpHead2::SL_TRUNK + j] = tot;
    }
}

// Stage 4c (Trunk): Bitonic sort 256 exact FP8 candidates + full-vocab LSE (top-256 exact + 2-bit tail) -> wire packet!
// Zero full-vocab head_pack_fast scans, zero 2-bit false positives in top-NC! Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_qw38_head2_pack_256(
    const float* __restrict__ cl,
    const int* __restrict__ cand,
    const double* __restrict__ tail_dsum,
    const float* __restrict__ screen_max,
    int v0, int NC,
    uint8_t* __restrict__ out, size_t rb) {
    const int m = blockIdx.x;
    const int t = threadIdx.x; // 0..255
    const int lane = t & 31, warp = t >> 5;

    float v = cl[static_cast<size_t>(m) * MtpHead2::SL_TRUNK + t];
    int id = v0 + cand[static_cast<size_t>(m) * MtpHead2::SL_TRUNK + t];
    const int fin = ink2::head_key(v) ? 1 : 0;
    if (!fin) { v = -INFINITY; id = 0x7fffffff; }

    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ int s_fin[8];
    __shared__ double s_sum[8];

    sv[t] = v;
    si[t] = id;

    int wf = fin;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) wf += __shfl_xor_sync(0xffffffffu, wf, o);
    if (lane == 0) s_fin[warp] = wf;
    __syncthreads();

    // Bitonic sort 256 candidates descending by (logit, -id)
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = t ^ j;
            if (ixj > t) {
                const bool want_desc = ((t & k) == 0);
                const float vt = sv[t], vx = sv[ixj];
                const int it = si[t], ix = si[ixj];
                const bool a_before_b = (vt > vx) || (vt == vx && it < ix);
                if (want_desc ? (!a_before_b) : a_before_b) {
                    sv[t] = vx; si[t] = ix;
                    sv[ixj] = vt; si[ixj] = it;
                }
            }
            __syncthreads();
        }
    }

    const float mx = sv[0];
    double e = (ink2::head_key(sv[t]) && ink2::head_key(mx)) ? static_cast<double>(expf(sv[t] - mx)) : 0.0;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) s_sum[warp] = e;
    __syncthreads();

    uint8_t* o = out + static_cast<size_t>(m) * rb;
    int tot_fin = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) tot_fin += s_fin[w];
    const int n_out = min(NC, tot_fin);

    if (t == 0) {
        double sum256 = 0.0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum256 += s_sum[w];
        double tail = tail_dsum[m];
        const float smx = screen_max[m];
        if (tail > 0.0 && ink2::head_key(smx) && ink2::head_key(mx)) {
            const double shift = static_cast<double>(smx) - static_cast<double>(mx);
            tail *= exp(fmin(shift, 20.0));
        }
        *reinterpret_cast<float*>(o) = mx;
        *reinterpret_cast<int*>(o + 4) = n_out;
        *reinterpret_cast<double*>(o + 8) = sum256 + tail;
    }
    if (t < n_out) {
        *reinterpret_cast<float*>(o + 16 + static_cast<size_t>(t) * 8) = sv[t];
        *reinterpret_cast<int*>(o + 20 + static_cast<size_t>(t) * 8) = si[t];
    }
}

// Stage 4a: Reduce SL=64 exact rescored candidates -> HeadArgmax[M] (Grid M, Block 64).
__global__ void __launch_bounds__(64) k_qw38_mtp_head2_argmax_pair(
    const float* __restrict__ cl,
    const int* __restrict__ cand,
    int v0,
    HeadArgmax* __restrict__ out) {
    const int m = blockIdx.x;
    const int t = threadIdx.x; // 0..63
    const int lane = t & 31, warp = t >> 5;
    float mx = cl[static_cast<size_t>(m) * MtpHead2::SL + t];
    int id = v0 + cand[static_cast<size_t>(m) * MtpHead2::SL + t];
    if (!ink2::head_key(mx)) { mx = -INFINITY; id = 0x7fffffff; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, mx, off);
        const int oi = __shfl_xor_sync(0xffffffffu, id, off);
        head_better(mx, id, om, oi);
    }
    __shared__ float smx[2];
    __shared__ int sid[2];
    if (lane == 0) { smx[warp] = mx; sid[warp] = id; }
    __syncthreads();
    if (t == 0) {
        float fm = smx[0];
        int fi = sid[0];
        head_better(fm, fi, smx[1], sid[1]);
        out[m] = HeadArgmax{fm, fi};
    }
}

// Stage 4b: Reduce SL=64 exact rescored candidates -> wire packet (for single-step greedy mtp_draft_step).
__global__ void __launch_bounds__(64) k_qw38_mtp_head2_argmax_pkt(
    const float* __restrict__ cl,
    const int* __restrict__ cand,
    int v0,
    uint8_t* __restrict__ out, size_t rb) {
    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    float mx = cl[static_cast<size_t>(m) * MtpHead2::SL + t];
    int id = v0 + cand[static_cast<size_t>(m) * MtpHead2::SL + t];
    int fin = ink2::head_key(mx) ? 1 : 0;
    if (!fin) { mx = -INFINITY; id = 0x7fffffff; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, mx, off);
        const int oi = __shfl_xor_sync(0xffffffffu, id, off);
        head_better(mx, id, om, oi);
        fin += __shfl_xor_sync(0xffffffffu, fin, off);
    }
    __shared__ float smx[2];
    __shared__ int sid[2];
    __shared__ int sfn[2];
    if (lane == 0) { smx[warp] = mx; sid[warp] = id; sfn[warp] = fin; }
    __syncthreads();
    if (t == 0) {
        float fm = smx[0];
        int fi = sid[0];
        head_better(fm, fi, smx[1], sid[1]);
        const int n = (sfn[0] + sfn[1] > 0) ? 1 : 0;
        uint8_t* o = out + static_cast<size_t>(m) * rb;
        *reinterpret_cast<float*>(o) = fm;
        *reinterpret_cast<int*>(o + 4) = n;
        *reinterpret_cast<double*>(o + 8) = n ? 1.0 : 0.0;
        if (n) {
            *reinterpret_cast<float*>(o + 16) = fm;
            *reinterpret_cast<int*>(o + 20) = fi;
        }
    }
}

// Fallback full-vocab argmax -> HeadArgmax[M] from blk_max/blk_argmax (used when M > 8 or !MtpHead2).
__global__ void __launch_bounds__(64) k_qw38_head_argmax_to_pair(
    const float* __restrict__ blk_max,
    const int* __restrict__ blk_argmax,
    HeadArgmax* __restrict__ out) {
    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    float mx = blk_max[m * HEAD_BLKS + t];
    int id = blk_argmax[m * HEAD_BLKS + t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, mx, off);
        const int oi = __shfl_xor_sync(0xffffffffu, id, off);
        head_better(mx, id, om, oi);
    }
    __shared__ float smx[2];
    __shared__ int sid[2];
    if (lane == 0) { smx[warp] = mx; sid[warp] = id; }
    __syncthreads();
    if (t == 0) {
        float fm = smx[0];
        int fi = sid[0];
        head_better(fm, fi, smx[1], sid[1]);
        out[m] = HeadArgmax{fm, fi};
    }
}

inline void launch_mtp_head2_screen(const MtpHead2& h2, const __nv_bfloat16* x, float* y, int M, cudaStream_t st) {
    if (M == 1) k_qw38_mtp_head2_screen<1><<<192, 256, 0, st>>>(h2.codes, h2.scales, x, y, M, h2.vr);
    else if (M <= 2) k_qw38_mtp_head2_screen<2><<<192, 256, 0, st>>>(h2.codes, h2.scales, x, y, M, h2.vr);
    else if (M <= 4) k_qw38_mtp_head2_screen<4><<<192, 256, 0, st>>>(h2.codes, h2.scales, x, y, M, h2.vr);
    else k_qw38_mtp_head2_screen<8><<<96, 256, 0, st>>>(h2.codes, h2.scales, x, y, M, h2.vr);
}

inline void mtp_head2_argmax_pair(const MtpHead2& h2, const Fp8Mat& fp8_lm, const __nv_bfloat16* x,
                                  float* head_logits, MtpHead2Scratch& sc, HeadArgmax* out_pair,
                                  int M, cudaStream_t st) {
    launch_mtp_head2_screen(h2, x, head_logits, M, st);
    k_qw38_head2_top<MtpHead2::SL, false><<<M, 1024, 0, st>>>(head_logits, h2.vr, sc.cand);
    if (fp8_lm.mx_sc) {
        k_qw38_mtp_head2_rescore_fp8<true><<<dim3(M, 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, sc.cand, head_logits, sc.cl, h2.vr, false);
    } else {
        k_qw38_mtp_head2_rescore_fp8<false><<<dim3(M, 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, sc.cand, head_logits, sc.cl, h2.vr, false);
    }
    k_qw38_mtp_head2_argmax_pair<<<M, 64, 0, st>>>(sc.cl, sc.cand, h2.v0, out_pair);
}

inline void mtp_head2_pack(const MtpHead2& h2, const Fp8Mat& fp8_lm, const __nv_bfloat16* x,
                           float* head_logits, MtpHead2Scratch& h2sc, HeadPackScratch& hsc,
                           int NC, uint8_t* out_pkt, size_t rb, int M, bool argmax_only, cudaStream_t st) {
    launch_mtp_head2_screen(h2, x, head_logits, M, st);
    k_qw38_head2_top<MtpHead2::SL, false><<<M, 1024, 0, st>>>(head_logits, h2.vr, h2sc.cand);
    if (fp8_lm.mx_sc) {
        k_qw38_mtp_head2_rescore_fp8<true><<<dim3(M, 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, h2sc.cand, head_logits, h2sc.cl, h2.vr, !argmax_only);
    } else {
        k_qw38_mtp_head2_rescore_fp8<false><<<dim3(M, 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, h2sc.cand, head_logits, h2sc.cl, h2.vr, !argmax_only);
    }
    if (argmax_only) {
        k_qw38_mtp_head2_argmax_pkt<<<M, 64, 0, st>>>(h2sc.cl, h2sc.cand, h2.v0, out_pkt, rb);
    } else {
        head_pack_fast(head_logits, M, h2.vr, h2.v0, NC, hsc, out_pkt, rb, st, false);
    }
}

inline void trunk_head2_pack(const MtpHead2& h2, const Fp8Mat& fp8_lm, const __nv_bfloat16* x,
                             float* head_logits, MtpHead2Scratch& h2sc,
                             int NC, uint8_t* out_pkt, size_t rb, int M, cudaStream_t st) {
    launch_mtp_head2_screen(h2, x, head_logits, M, st);
    k_qw38_head2_top<MtpHead2::SL_TRUNK, true><<<M, 1024, 0, st>>>(
        head_logits, h2.vr, h2sc.cand, h2sc.tail_dsum, h2sc.screen_max);
    if (fp8_lm.mx_sc) {
        k_qw38_head2_rescore_fp8_exact<true><<<dim3(M, MtpHead2::SL_TRUNK / 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, h2sc.cand, h2sc.cl);
    } else {
        k_qw38_head2_rescore_fp8_exact<false><<<dim3(M, MtpHead2::SL_TRUNK / 8), 256, 0, st>>>(
            fp8_lm.q, fp8_lm.sc, fp8_lm.mx_sc, x, h2sc.cand, h2sc.cl);
    }
    k_qw38_head2_pack_256<<<M, 256, 0, st>>>(
        h2sc.cl, h2sc.cand, h2sc.tail_dsum, h2sc.screen_max, h2.v0, NC, out_pkt, rb);
}

} // namespace qw38
