// GR8_DRAFT_HEAD2 (two-stage draft head): a 2-bit screen over the rank's vocab slice, then the exact MXFP8
// logits of the screen's best SL rows. Drafts only: the target's head is not touched.
//
//   1. k_screen     y[m][v] ~ x[m] . w[v] from 2-bit g32 records of the slice (made at boot from the MXFP8 head records,
//                   k_mx_to_i2): 1536 code bytes + 384 scale bytes per vocab row (6144 inputs), 74.5 MB per TP4 slice
//                   instead of 134 MB (INT4 g32, GR8_DRAFT_HEAD_I4) or 246 MB (MXFP8)
//   2. k_screen_top per row, the SL best screen logits (ties and the last histogram bin in index order: deterministic)
//   3. k_rescore    per row and candidate, the MXFP8 logit of that vocab row (its E4M3 codes and E8M0 scales read out of
//                   the 16-row tile records the target's head reads), written over the screen logit in y and into the
//                   candidate list
//   4. k_head_local_sl (glue k_head_local over the candidates): the record's best sample score among the candidates,
//                   its log-sum-exp over the whole row (exact at the candidates, screen elsewhere). The head merge is
//                   unchanged, so every rank merges the same bytes.
// The draft of a row is the MXFP8 head's argmax whenever that token is among its slice's SL screen best (offline, 24k
// recorded draft-head inputs: never missed at SL >= 32 in an offline replay). Its log-probability's
// log-sum-exp takes the screen logits outside the candidates (|error| p99 0.07 nats offline; INT4's own is 0.31).
//
// 2-bit record of a vocab row: 96 chunks of 64 inputs (k = 64 c .. 64 c + 63 = groups 2c and 2c+1 of 32), one uint4
// each, chunk-major [vn][96]; word q of a chunk holds k = 64 c + 16 q .. + 15: bits 2i = element 2i, bits 16 + 2i =
// element 2i + 1 (i = 0..7), so ((word << 1 >> 2i) & 0x00060006) | 0x64006400 is the fp16 pair (1024 + 2 q_2i, 1024 + 2 q_2i+1)
// and minus 1027 the exact odd integers 2q - 3 = 2 (q - 1.5).
// Scales [vn][96] half2 (groups 2c, 2c + 1). Value = (q - 1.5) s, per (row, group of 32) from the MXFP8 values v:
//   s = fp16(amax |v| / 1.5) (round to nearest even), q = clamp(floor(v / s + 2), 0, 3) (q = 2 where s == 0)
// Every thread writes only its own (row, chunk): the conversion is deterministic.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "glue.cuh"
#include "head_i4.cuh"

namespace head2
{
constexpr int K = HID, NG = HID / 32, NCH = HID / 64;   // inputs, groups of 32, chunks of 64
constexpr int SL = 64;                                   // candidates per row (the rescored best of the screen)
constexpr int MXM = 8;                                   // rows per call at most (k_screen keeps x of every row in smem)
constexpr int BINS = 2048;                               // k_screen_top: histogram of (max - logit) over RANGE logit units
constexpr float RANGE = 32.f;
static_assert(K % 2048 == 0, "k_screen: three 2048-input sweeps of 32 lanes x 64");

inline size_t code_bytes(int vn) { return (size_t)vn * NCH * 16; }
inline size_t scale_bytes(int vn) { return (size_t)vn * NG * 2; }
inline size_t bytes(int vn) { return code_bytes(vn) + scale_bytes(vn); }

// ------------------------------------------------------------------------------------------- boot conversion
// mx [vn/16][NG][528] -> codes [vn][NCH] uint4, scales [vn][NCH] half2; one thread per (row, chunk); grid ceil(vn*NCH/256)
__global__ void __launch_bounds__(256) k_mx_to_i2(const uint8_t* __restrict__ mx, uint4* __restrict__ codes, __half2* __restrict__ scales, int vn)
{
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)vn * NCH) return;
    const int row = (int)(id / NCH), c = (int)(id % NCH);
    const int tile = row >> 4, r = row & 7, hi = (row >> 3) & 1;
    uint32_t w[4] = {0u, 0u, 0u, 0u};
    __half s16[2];
    #pragma unroll
    for (int gg = 0; gg < 2; ++gg)
    {
        const int g = 2 * c + gg;
        const uint8_t* src = mx + ((size_t)tile * NG + g) * head_i4::MXGR;
        const float sc = head_i4::e8m0_val(src[512 + 2 * r + hi]);
        float v[32];
        float amax = 0.f;
        #pragma unroll
        for (int kk = 0; kk < 32; ++kk)
        {
            const int h = kk >> 4, k = kk & 15, cc = (k & 7) >> 1;
            v[kk] = head_i4::e4m3_val(src[h * 256 + (r * 4 + cc) * 8 + (k >> 3) * 4 + hi * 2 + (k & 1)]) * sc;
            amax = fmaxf(amax, fabsf(v[kk]));
        }
        s16[gg] = __float2half_rn(__fdiv_rn(amax, 1.5f));
        const float s = __half2float(s16[gg]);
        #pragma unroll
        for (int kk = 0; kk < 32; ++kk)
        {
            const uint32_t q = s != 0.f ? (uint32_t)min(max(__float2int_rd(__fadd_rn(__fdiv_rn(v[kk], s), 2.f)), 0), 3) : 2u;
            const int e = gg * 32 + kk, word = e >> 4, i = e & 15;      // element i of word: bits 2 (i / 2) + 16 (i % 2)
            w[word] |= q << (2 * (i >> 1) + 16 * (i & 1));
        }
    }
    codes[id] = make_uint4(w[0], w[1], w[2], w[3]);
    scales[id] = __halves2half2(s16[0], s16[1]);
}
inline void convert(const uint8_t* mx, uint4* codes, __half2* scales, int vn, cudaStream_t st)
{
    const long long n = (long long)vn * NCH;
    k_mx_to_i2<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(mx, codes, scales, vn);
}

// ------------------------------------------------------------------------------------------- 1. screen GEMV
// grid persistent (screen_grid), block 256 = 8 warps, dynamic smem M x K fp16 (x rows, 16-byte units swizzled: unit u of
// a row at u ^ ((u >> 3) & 7), so the 32 lanes' reads of their chunks hit distinct units). A warp takes one vocab row at a
// time; lane l takes chunks l, l + 32, l + 64. Per group of 32: 16 HFMA2 of (2q - 3) x into an fp16 pair from zero, then
// y += (s / 2) * (lo + hi) in fp32; lane sums in chunk order, then an xor tree: the bits do not depend on the grid.
__device__ __forceinline__ int swz(int u) { return u ^ ((u >> 3) & 7); }
template <int M>
__global__ void __launch_bounds__(256) k_screen(const uint4* __restrict__ codes, const __half2* __restrict__ scales, const __half* __restrict__ x,
                                                int ldx, float* __restrict__ y, int ldy, int vn)
{
    extern __shared__ uint4 xs4[];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    for (int i = t; i < M * (K / 8); i += 256)
    {
        const int m = i / (K / 8), u = i % (K / 8);
        xs4[m * (K / 8) + swz(u)] = *reinterpret_cast<const uint4*>(x + (size_t)m * ldx + 8 * u);
    }
    __syncthreads();
    const __half2 off = __float2half2_rn(1027.f);
    for (int v = blockIdx.x * 8 + w; v < vn; v += gridDim.x * 8)
    {
        uint4 cw[3];
        __half2 sc[3];
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            cw[j] = __ldcs(codes + (size_t)v * NCH + 32 * j + l);         // streamed once per pass
            sc[j] = __ldcs(scales + (size_t)v * NCH + 32 * j + l);
        }
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.f;
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            const int c = 32 * j + l;
            const uint32_t wd[4] = {cw[j].x, cw[j].y, cw[j].z, cw[j].w};
            #pragma unroll
            for (int gg = 0; gg < 2; ++gg)
            {
                __half2 p[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) p[m] = __float2half2_rn(0.f);
                #pragma unroll
                for (int q = 2 * gg; q < 2 * gg + 2; ++q)
                {
                    #pragma unroll
                    for (int h8 = 0; h8 < 2; ++h8)                         // 8 elements = one 16-byte unit of x
                    {
                        const int u = 8 * c + 2 * q + h8;
                        __half2 wv[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            const int sh = 2 * (4 * h8 + i);              // pair (element sh / 2 * 2, + 1): codes at bits sh, 16 + sh
                            const uint32_t b = ((sh ? wd[q] >> (sh - 1) : wd[q] << 1) & 0x00060006u) | 0x64006400u;
                            wv[i] = __hsub2(*reinterpret_cast<const __half2*>(&b), off);
                        }
                        #pragma unroll
                        for (int m = 0; m < M; ++m)
                        {
                            const uint4 xv = xs4[m * (K / 8) + swz(u)];
                            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) p[m] = __hfma2(wv[i], *reinterpret_cast<const __half2*>(&xw[i]), p[m]);
                        }
                    }
                }
                const float s = 0.5f * (gg ? __high2float(sc[j]) : __low2float(sc[j]));
                #pragma unroll
                for (int m = 0; m < M; ++m)
                {
                    const float2 f = __half22float2(p[m]);
                    acc[m] = fmaf(s, f.x + f.y, acc[m]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m)
        {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
            if (l == 0) y[(size_t)m * ldy + v] = acc[m];
        }
    }
}
inline size_t screen_smem(int M) { return (size_t)M * K * 2; }
// one wave: the SMs x the CTAs of M rows' smem that fit on one (init); rows go to warps by a fixed stride
inline int& screen_grid(int M) { static int g[MXM + 1] = {}; return g[M]; }
template <int M> inline const char* screen_attr()
{
    if (cudaFuncSetAttribute(k_screen<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)screen_smem(M)) != cudaSuccess)
        return "head2: cannot raise k_screen's shared memory";
    int dev = 0, sms = 0, per = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_screen<M>, 256, screen_smem(M)) != cudaSuccess || per < 1)
        return "head2: k_screen does not fit on an SM";
    screen_grid(M) = sms * per;
    return nullptr;
}
// once before any launch (and outside any capture): the shared memory limits of every instantiation
inline const char* init()
{
    const char* e = nullptr;
    if (!e) e = screen_attr<1>();
    if (!e) e = screen_attr<2>();
    if (!e) e = screen_attr<3>();
    if (!e) e = screen_attr<4>();
    if (!e) e = screen_attr<5>();
    if (!e) e = screen_attr<6>();
    if (!e) e = screen_attr<7>();
    if (!e) e = screen_attr<8>();
    return e;
}
inline void launch_screen(const uint4* codes, const __half2* scales, const __half* x, int ldx, float* y, int ldy, int vn, int M, cudaStream_t st)
{
    const dim3 g(screen_grid(M < MXM ? M : MXM)), b(256);
    const size_t sm = screen_smem(M);
    switch (M)
    {
    case 1: k_screen<1><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 2: k_screen<2><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 3: k_screen<3><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 4: k_screen<4><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 5: k_screen<5><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 6: k_screen<6><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    case 7: k_screen<7><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    default: k_screen<8><<<g, b, sm, st>>>(codes, scales, x, ldx, y, ldy, vn); break;
    }
}

// ------------------------------------------------------------------------------------------- 2. the screen's best SL
// grid M, block 1024. Thread t owns the contiguous indices [t * per, (t + 1) * per). The SL best: every index whose
// bin of (max - y) is below the cut bin, then the cut bin's indices in ascending order until SL (bin = BINS - 1 takes
// everything past RANGE). cand[m][0..SL) = local indices (ascending within the two classes: deterministic).
__global__ void __launch_bounds__(1024) k_screen_top(const float* __restrict__ y, int ldy, int vn, int* __restrict__ cand)
{
    __shared__ unsigned hist[BINS];
    __shared__ float s_mx[32];
    __shared__ int s_cut, s_lt;
    __shared__ int s_n[2][1024];
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const float* row = y + (size_t)m * ldy;
    const int per = (vn + 1023) / 1024, i0 = min(t * per, vn), i1 = min(i0 + per, vn);
    float mx = -INFINITY;
    for (int i = i0; i < i1; ++i) mx = fmaxf(mx, row[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (l == 0) s_mx[w] = mx;
    for (int i = t; i < BINS; i += 1024) hist[i] = 0u;
    __syncthreads();
    mx = s_mx[0];
    for (int i = 1; i < 32; ++i) mx = fmaxf(mx, s_mx[i]);
    auto bin = [&](float v) { const float b = (mx - v) * (BINS / RANGE); return b < (float)(BINS - 1) ? (int)b : BINS - 1; };
    for (int i = i0; i < i1; ++i) atomicAdd(&hist[bin(row[i])], 1u);
    __syncthreads();
    if (w == 0)                          // cut = the first bin where the count of bins 0..cut reaches SL
    {
        constexpr int PER = BINS / 32;
        unsigned c = 0;
        for (int i = 0; i < PER; ++i) c += hist[l * PER + i];
        unsigned inc = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const unsigned z = __shfl_up_sync(0xffffffffu, inc, o); if (l >= o) inc += z; }
        const unsigned over = __ballot_sync(0xffffffffu, inc >= (unsigned)SL);
        const int L = over ? __ffs(over) - 1 : 31;
        if (l == L)
        {
            unsigned cum = inc - c;
            int cut = L * PER + PER - 1;
            for (int i = 0; i < PER; ++i)
            {
                if (cum + hist[L * PER + i] >= (unsigned)SL) { cut = L * PER + i; break; }
                cum += hist[L * PER + i];
            }
            s_cut = cut;
            s_lt = (int)cum;                 // indices in bins below the cut (< SL)
        }
    }
    __syncthreads();
    const int cut = s_cut, lt = s_lt;
    int nlt = 0, neq = 0;
    for (int i = i0; i < i1; ++i) { const int b = bin(row[i]); nlt += b < cut; neq += b == cut; }
    s_n[0][t] = nlt; s_n[1][t] = neq;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1)       // inclusive scans of both counts in thread (= index) order
    {
        const int a = t >= o ? s_n[0][t - o] : 0, b = t >= o ? s_n[1][t - o] : 0;
        __syncthreads();
        s_n[0][t] += a; s_n[1][t] += b;
        __syncthreads();
    }
    int plt = s_n[0][t] - nlt, peq = lt + s_n[1][t] - neq;
    int* out = cand + (size_t)m * SL;
    for (int i = i0; i < i1; ++i)
    {
        const int b = bin(row[i]);
        if (b < cut) out[plt++] = i;
        else if (b == cut && peq < SL) out[peq++] = i;
    }
}

// ------------------------------------------------------------------------------------------- 3. exact rescoring
// grid (M, SL / 8), block 256: warp = one candidate of row m. Lane l takes groups l, l + 32, .. (6 of 192): per group,
// the row's 32 E4M3 codes (8 x 8 bytes of the tile record, 4 of the row's codes in each) x x[m] in fp32 in k order,
// times the group's E8M0 scale; lanes in group order, then an xor tree. The logit goes over the screen's in y and into
// cl[m][j].
__global__ void __launch_bounds__(256) k_rescore(const uint8_t* __restrict__ mx, const __half* __restrict__ x, int ldx, const int* __restrict__ cand,
                                                 float* __restrict__ y, int ldy, float* __restrict__ cl)
{
    const int m = blockIdx.x, j = blockIdx.y * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    const int v = cand[(size_t)m * SL + j];
    const int tile = v >> 4, r = v & 7, hi = (v >> 3) & 1;
    const __half* xr = x + (size_t)m * ldx;
    float acc = 0.f;
    for (int g = l; g < NG; g += 32)
    {
        const uint8_t* src = mx + ((size_t)tile * NG + g) * head_i4::MXGR;
        const float sc = head_i4::e8m0_val(src[512 + 2 * r + hi]);
        float d = 0.f;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const uint2 b = *reinterpret_cast<const uint2*>(src + h * 256 + (r * 4 + c) * 8);
                const uint32_t lo = hi ? b.x >> 16 : b.x, up = hi ? b.y >> 16 : b.y;   // (k 2c, 2c+1) and (2c+8, 2c+9) of half h
                const int k0 = 32 * g + 16 * h + 2 * c;
                const __half2 x01 = *reinterpret_cast<const __half2*>(xr + k0), x89 = *reinterpret_cast<const __half2*>(xr + k0 + 8);
                d = fmaf(head_i4::e4m3_val(lo & 0xffu), __low2float(x01), d);
                d = fmaf(head_i4::e4m3_val((lo >> 8) & 0xffu), __high2float(x01), d);
                d = fmaf(head_i4::e4m3_val(up & 0xffu), __low2float(x89), d);
                d = fmaf(head_i4::e4m3_val((up >> 8) & 0xffu), __high2float(x89), d);
            }
        acc = fmaf(sc, d, acc);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (l == 0)
    {
        y[(size_t)m * ldy + v] = acc;
        cl[(size_t)m * SL + j] = acc;
    }
}

// ------------------------------------------------------------------------------------------- 4. the head record
// glue::k_head_local with the best score taken over the row's SL candidates (exact logits), the log-sum-exp (and the
// top_p / top-k parts, the target id) over the whole row as k_head_local reads it. grid M, block 1024.
__global__ void __launch_bounds__(1024) k_head_local_sl(glue::HeadArgs a, const int* __restrict__ cand, const float* __restrict__ cl)
{
    using namespace glue;
    __shared__ float s_sc[32], s_lg[32], s_mx[32], s_sm[32];
    __shared__ int s_id[32];
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const SampleRow sr = a.rows[m];
    const float* row = a.logits + (size_t)m * a.ldl;
    float bs = -INFINITY, blg = 0.f, mx = -INFINITY, sm = 0.f;
    int bid = 0x7fffffff;
    if (t < SL)
    {
        const float lg = cl[(size_t)m * SL + t];
        const int v = a.vlo + cand[(size_t)m * SL + t];
        bs = sample_score(sr, lg, v); bid = v; blg = lg;
    }
    for (int i = t; i < a.vn; i += 1024)
    {
        const float lg = row[i];
        if (lg > mx) { sm = sm * expf(mx - lg) + 1.f; mx = lg; }
        else sm += expf(lg - mx);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        const float os = __shfl_xor_sync(0xffffffffu, bs, o), ol = __shfl_xor_sync(0xffffffffu, blg, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
        if (os > bs || (os == bs && oi < bid)) { bs = os; bid = oi; blg = ol; }
        const float omx = __shfl_xor_sync(0xffffffffu, mx, o), osm = __shfl_xor_sync(0xffffffffu, sm, o);
        const float nmx = fmaxf(mx, omx);
        sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (omx == -INFINITY ? 0.f : osm * expf(omx - nmx));
        mx = nmx;
    }
    if (l == 0) { s_sc[w] = bs; s_id[w] = bid; s_lg[w] = blg; s_mx[w] = mx; s_sm[w] = sm; }
    __syncthreads();
    if (t == 0)
    {
        HeadRec r;
        r.score = s_sc[0]; r.id = s_id[0]; r.logit = s_lg[0]; r.lmax = s_mx[0]; r.lsum = s_sm[0];
        for (int i = 1; i < 32; ++i)
        {
            if (s_sc[i] > r.score || (s_sc[i] == r.score && s_id[i] < r.id)) { r.score = s_sc[i]; r.id = s_id[i]; r.logit = s_lg[i]; }
            const float nmx = fmaxf(r.lmax, s_mx[i]);
            r.lsum = (r.lmax == -INFINITY ? 0.f : r.lsum * expf(r.lmax - nmx)) +
                     (s_mx[i] == -INFINITY ? 0.f : s_sm[i] * expf(s_mx[i] - nmx));
            r.lmax = nmx;
        }
        r.pad_[0] = r.pad_[1] = r.pad_[2] = 0.f;
        if (a.tgt)
        {
            const int i = a.tgt[m] - a.vlo;
            r.pad_[0] = i >= 0 && i < a.vn ? row[i] : -INFINITY;
        }
        reinterpret_cast<HeadRec*>(send_slot(a.f, *a.stepp, a.c))[m] = r;
    }
    if (a.nc > 0 && topp_on(sr)) head_local_topp(a, m, row, sr, s_mx);
    if (a.tk && sr.topk > 0) head_local_topk(a, m, row);
    const u64 step = *a.stepp;
    publish(a.f, a.c, g6::coll_seq(step, a.c, a.f.ncoll), a.ticket);
}

// GR8_DRAFT_HEAD2=2 (measurement): per row, whether the MXFP8 head's argmax of the slice (ref, the target's GEMV on the
// same rows) is among the row's candidates. cnt[0] rows, cnt[1] misses; every 4096 rows rank prints a line (device
// printf, flushed at the host's next synchronization). grid M, block 256
__global__ void __launch_bounds__(256) k_check(const float* __restrict__ ref, int ldr, int vn, const int* __restrict__ cand, unsigned* cnt, int rank)
{
    __shared__ float s_v[8];
    __shared__ int s_i[8];
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = t; i < vn; i += 256)
    {
        const float v = ref[(size_t)m * ldr + i];
        if (v > bv) { bv = v; bi = i; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (l == 0) { s_v[w] = bv; s_i[w] = bi; }
    __syncthreads();
    if (t == 0)
    {
        for (int i = 1; i < 8; ++i)
            if (s_v[i] > bv || (s_v[i] == bv && s_i[i] < bi)) { bv = s_v[i]; bi = s_i[i]; }
        bool in = false;
        for (int j = 0; j < SL; ++j) in |= cand[(size_t)m * SL + j] == bi;
        const unsigned n = atomicAdd(cnt, 1u) + 1u;
        const unsigned x = in ? cnt[1] : atomicAdd(cnt + 1, 1u) + 1u;
        if (n % 4096 == 0) printf("# head2 check rank %d: %u of %u rows' MXFP8 argmax outside the %d candidates\n", rank, x, n, SL);
    }
}

// the screen, its best, the rescoring: y [M][ldy] then holds the row logits k_head_local_sl reads (a.logits = y)
struct Bufs
{
    const uint4* codes = nullptr;
    const __half2* scales = nullptr;
    const uint8_t* mx = nullptr;      // the MXFP8 head records of the slice (the target's)
    int* cand = nullptr;              // [MXM][SL]
    float* cl = nullptr;              // [MXM][SL]
};
inline void launch_front(const Bufs& b, const __half* x, int ldx, float* y, int ldy, int vn, int M, cudaStream_t st)
{
    launch_screen(b.codes, b.scales, x, ldx, y, ldy, vn, M, st);
    k_screen_top<<<M, 1024, 0, st>>>(y, ldy, vn, b.cand);
    k_rescore<<<dim3(M, SL / 8), 256, 0, st>>>(b.mx, x, ldx, b.cand, y, ldy, b.cl);
}
}   // namespace head2
