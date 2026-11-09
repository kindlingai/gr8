// DeepSeek-V4.1-Flash-1017 hyper-connection trunk kernels (SHIFTED variant, no hc_head). New file; nothing in
// engine/mhc/mhc.cuh changes. Host twin: engine/mhc/ds_cpu.h. Torch reference: engine/mhc/ds_ref.py.
//
// Per sub-layer s (attn then ffn, 80 in all), trunk T = [rows,4,H] FP32 buffer that always holds BF16-valued numbers:
//   embed      T = 4 copies of the embedding row; pre = [1,0,0,0]
//   prepare    mix[o] = dot(T, fn[o]) * rsqrt(mean(T^2) + eps)  (fn F32 or BF16 [24,4H]; both reductions in the same pass)
//              pre_next/post/comb from mix (flash_mhc::coefficients_warp, same Sinkhorn); then
//              y = BF16(RMSNorm(BF16(sum_i pre[i]*T[i])) * norm)   <- collapse uses the CARRIED pre, not pre_next
//              pre <- pre_next (carried to the next sub-layer's collapse)
//   consume    T[j] = BF16(post[j]*BF16(out) + sum_i comb[i][j]*T[i])   (out = reduced block output, BF16 module output)
//   finish     consume of the last block, then x = BF16(sum_i pre[i]*T[i]) with the last carried pre,
//              h = BF16(RMSNorm(x) * norm.weight) -> lm_head input (optional fragment tables as flash_mhc::k_mhc_final)
// Differences from flash_mhc: shifted pre, F32 fn option, BF16 trunk
// writes always, the collapse rounded to BF16 before the norm, no mean of 4 streams at the end, no stored `inv`
// (prepare recomputes the flat sum of squares from the rows it already reads, so consume is a plain elementwise kernel
// that spreads over H/256 x rows CTAs), generic H (the DS H=5120 has no H==4096 fast path to fall into).
// Block getters are the flash_mhc ones: init(), wait(step), operator()(size_t k) with k = row*H + h, as MhcFabricBlock.
// Trunk storage: the trunk only ever holds BF16-valued numbers, so it can be stored as BF16 with no change in any bit of
// any result (every writer rounds with rn16 / copies a BF16 embedding; every reader widens to FP32 first, which is exact). -DDS_MHC_TRUNK_BF16
// makes ds_mhc::trunk_t = __nv_bfloat16 (Model::resid4, Buffers::streams); the default stays FP32. Every kernel and wrapper below is a
// template on the trunk element type, so one binary (ds_mhc_mtest) can run both and compare them; the engine uses ds_mhc::Buffers = BuffersT<trunk_t>.
// The BF16 trunk halves the trunk's DRAM traffic (the prefill mHC is DRAM-bound).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "mhc.cuh"
namespace ds_mhc {
using flash_mhc::THREADS;
using flash_mhc::MIX;
// GR8_DS_DEC_STAGE=1: a Block with `static constexpr bool STAGE = true`, `NSLOT` and `uint4 ld16(int p, size_t elem) const` (16 bytes = 8 BF16 of slot p, elem a
// multiple of 8) gets the staged slot read in k_step / k_consume / k_finish: the CTA loads the slot tile with 16-byte loads into smem, then each thread sums its column
// from smem. Every other Block compiles the plain path: the staged paths sit in `if constexpr (has_stage_v<Block>)` branches.
template <class B, class = void> struct has_stage : std::false_type {};
template <class B> struct has_stage<B, std::void_t<decltype(B::STAGE)>> : std::bool_constant<B::STAGE> {};
template <class B> constexpr bool has_stage_v = has_stage<B>::value;
#ifdef DS_MHC_TRUNK_BF16
using trunk_t = __nv_bfloat16;   // trunk storage of the engine build (see the header): BF16
#else
using trunk_t = float;           // default: BF16-valued FP32
#endif
template <class TT> struct BuffersT {
    TT* streams;               // [rows,4,H]; BF16-valued numbers, stored as TT (float or __nv_bfloat16)
    float* mix;                // [rows,24]
    float* pre;                // [rows,4]  CARRIED: the pre the next collapse uses (one-hot after embed)
    float* post;               // [rows,4]
    float* comb;               // [rows,16] source then destination
    unsigned* done = nullptr;  // [ceil(rows/RT)], zeroed once; atomicInc wraps at H/PW so it resets itself
    float* part = nullptr;     // [rows, 26, H/PW] partial dots / sums of squares of prepare (scratch)
};
using Buffers = BuffersT<trunk_t>;
template <class WT> struct Weights {
    const WT* fn;                     // [24,4H]; float (checkpoint F32) or __nv_bfloat16 (precision knob)
    const float *base, *scale;        // [24], [3]
    const __nv_bfloat16* norm;        // [H] attn_norm / ffn_norm
};
__device__ __forceinline__ float rn16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// Trunk element access: load widens to FP32 (exact for BF16), store rounds to the element type (v is always BF16-valued, so exact).
__device__ __forceinline__ float ld_t(const float* p) { return *p; }
__device__ __forceinline__ float ld_t(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void ld4(const float* p, float (&v)[4]) {
    const float4 t = __ldg(reinterpret_cast<const float4*>(p));
    v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
}
__device__ __forceinline__ void ld4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 t = __ldg(reinterpret_cast<const uint2*>(p));
    v[0] = __uint_as_float(t.x << 16); v[1] = __uint_as_float(t.x & 0xffff0000u);
    v[2] = __uint_as_float(t.y << 16); v[3] = __uint_as_float(t.y & 0xffff0000u);
}
// Two block sums in one pass; the leading barrier makes back-to-back calls safe (flash_mhc::sum_block has no guard).
__device__ inline void sum2(float& a, float& b) {
    __shared__ float sums[2][8];
    __syncthreads();
    for (int d = 16; d; d /= 2) { a += __shfl_down_sync(0xffffffffu, a, d); b += __shfl_down_sync(0xffffffffu, b, d); }
    if (!(threadIdx.x & 31)) { sums[0][threadIdx.x / 32] = a; sums[1][threadIdx.x / 32] = b; }
    __syncthreads();
    a = threadIdx.x < 8 ? sums[0][threadIdx.x] : 0.f;
    b = threadIdx.x < 8 ? sums[1][threadIdx.x] : 0.f;
    if (threadIdx.x < 32) {
        for (int d = 4; d; d /= 2) { a += __shfl_down_sync(0xffffffffu, a, d); b += __shfl_down_sync(0xffffffffu, b, d); }
        if (!threadIdx.x) { sums[0][0] = a; sums[1][0] = b; }
    }
    __syncthreads();
    a = sums[0][0]; b = sums[1][0];
}
__device__ inline float sum1(float a) { float z = 0.f; sum2(a, z); return a; }

// grid(ceil(H/THREADS), rows). Gather + 4 copies; pre = one-hot. tok must be in [0,V).
template <class TT>
__global__ void k_embed(BuffersT<TT> b, const int* tok, const __nv_bfloat16* embed, int H) {
    const int m = blockIdx.y, h = blockIdx.x * THREADS + threadIdx.x;
    if (h < H) {
        const float v = __bfloat162float(embed[(size_t)tok[m] * H + h]);
        for (int j = 0; j < 4; ++j) st_t(b.streams + ((size_t)m * 4 + j) * H + h, v);
    }
    if (blockIdx.x == 0 && threadIdx.x < 4) b.pre[m * 4 + threadIdx.x] = threadIdx.x == 0 ? 1.f : 0.f;
}
// ---- prepare, k-split. grid(H/PW, ceil(rows/RT)), THREADS = PW*RT threads. CTA (c,tile) owns the columns [c*PW,(c+1)*PW) of
// the 4 streams for RT rows. Step A (one thread = one (row, column)): optionally run the PREVIOUS sub-layer's hc_post
// (FUSE: T_new[i] = BF16(post[i]*BF16(out) + sum_j comb[j][i]*T[j]), written back in place), then the BF16 collapse
// x = BF16(sum_i pre_prev[i]*T_new[i]) (the carried pre) and its partial sums of squares. Step B (warp = 3 outputs): the
// partial dots of that column slice with the 24 fn rows; one fn slice is read once for all RT rows. Partials go to
// b.part[row][v][c]. The last CTA of the tile (atomic retirement) adds the partials (v < 24 dots, 24 = sum T^2, 25 = sum
// x^2), runs the Sinkhorn warp for each row, replaces pre by pre_next, and rescales the staged x into y.
constexpr int PW = 64, RT = 4, NV = MIX + 2, YU = 10;   // YU: bf16 pairs per thread per batch in the y rescale
static_assert(PW * RT == THREADS, "one thread per (row, column)");
struct NoBlock {};
// The staged-read tile maps (k16). One 16-byte load = 8 columns of one slot; the kernels and tests/ds41_stage_host_test.cu call the same functions.
//   k_step:    thread t < NSLOT*RT*(PW/8) loads (slot p, tile row rr, segment seg): elements (m0+rr)*H + c*PW + seg*8 .. +7  -> stg[p][rr][seg*8 ..]
//   k_consume: thread t < NSLOT*(THREADS/8) loads (slot p, segment seg):            elements m*H + blockIdx.x*THREADS + seg*8 .. +7 -> stg[p][seg*8 ..]
//   k_finish:  per FCH-column chunk, u = t, t+THREADS, .. < NSLOT*(FCH/8) loads (slot p, segment seg): elements mi*H + h0 + seg*8 .. +7 -> stg[p][seg*8 ..]
constexpr int FCH = 1024;   // k_finish staged chunk, in columns (H % FCH == 0)
struct StageMap { int p, rr, seg; };
__host__ __device__ constexpr StageMap stage_kstep(int t) { return {t / (RT * (PW / 8)), (t % (RT * (PW / 8))) / (PW / 8), t % (PW / 8)}; }
__host__ __device__ constexpr StageMap stage_consume(int t) { return {t / (THREADS / 8), 0, t % (THREADS / 8)}; }
__host__ __device__ constexpr StageMap stage_finish(int u) { return {u / (FCH / 8), 0, u % (FCH / 8)}; }
// Same arithmetic and order as flash_mhc::coefficients_warp, for lanes 0..15 of any warp (that one reads threadIdx.x).
__device__ inline void coeff16(const volatile float* mix, const float* base, const float* scale, float* pre, float* post, float* comb) {
    const int k = threadIdx.x & 31;
    if (k < 16) {
        const int i = k >> 2, j = k & 3;
        if (k < 4) {
            pre[k] = flash_mhc::sigmoid(mix[k] * scale[0] + base[k]) + flash_mhc::HC_EPS;
            post[k] = 2.f * flash_mhc::sigmoid(mix[4 + k] * scale[1] + base[4 + k]);
        }
        float c = mix[8 + k] * scale[2] + base[8 + k];
        float mx = -INFINITY;
        for (int jj = 0; jj < 4; ++jj) mx = fmaxf(mx, __shfl_sync(0xffffu, c, 4 * i + jj));
        c = expf(c - mx);
        float sum = 0.f;
        for (int jj = 0; jj < 4; ++jj) sum += __shfl_sync(0xffffu, c, 4 * i + jj);
        c = c / sum + flash_mhc::HC_EPS;
        for (int iter = 0; iter < flash_mhc::ITERS; ++iter) {
            if (iter) {
                float rsum = 0.f;
                for (int jj = 0; jj < 4; ++jj) rsum += __shfl_sync(0xffffu, c, 4 * i + jj);
                c /= rsum + flash_mhc::HC_EPS;
            }
            float csum = 0.f;
            for (int ii = 0; ii < 4; ++ii) csum += __shfl_sync(0xffffu, c, 4 * ii + j);
            c /= csum + flash_mhc::HC_EPS;
        }
        comb[k] = c;
    }
}
template <class TT, class WT, class Block, bool FUSE>
__global__ void __launch_bounds__(THREADS) k_step(BuffersT<TT> b, Weights<WT> w, Block block, int H, int M, float eps, __nv_bfloat16* y) {
    __shared__ __align__(16) float tsm[RT][4 * PW];      // the new trunk slice of this CTA
    __shared__ float wred[THREADS / 32][2];               // per-warp sum T^2, sum x^2
    __shared__ float vals[RT][NV];
    __shared__ float mixs[RT][MIX];
    __shared__ int sh_last;
    const int NCH = H / PW;
    const int c = blockIdx.x, tile = blockIdx.y, m0 = tile * RT;
    const int nr = min(RT, M - m0);
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int r = t / PW, hh = t % PW, m = m0 + r, h = c * PW + hh;
    const bool valid = r < nr;
    unsigned long long step = 0;
    if constexpr (FUSE) step = block.init();
    // fn slice first: it does not depend on the trunk, so its latency hides behind the trunk loads / the block wait.
    float fa[3][4], fb[3][4];
    {
        const int j0 = lane >> 4, hh0 = (lane & 15) * 4;
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            const WT* fn = w.fn + (size_t)(wid + 8 * q) * 4 * H + (size_t)c * PW + hh0;
            ld4(fn + (size_t)j0 * H, fa[q]);
            ld4(fn + (size_t)(j0 + 2) * H, fb[q]);
        }
    }
    float old[4] = {0.f, 0.f, 0.f, 0.f}, pre[4] = {0.f, 0.f, 0.f, 0.f}, post[4] = {0.f, 0.f, 0.f, 0.f}, comb[16];
    TT* __restrict__ sm = b.streams + (size_t)m * 4 * H;
    #pragma unroll
    for (int k = 0; k < 16; ++k) comb[k] = 0.f;
    if (valid) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) { old[j] = ld_t(sm + (size_t)j * H + h); pre[j] = b.pre[m * 4 + j]; }
        if constexpr (FUSE) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) post[i] = b.post[m * 4 + i];
            #pragma unroll
            for (int k = 0; k < 16; ++k) comb[k] = b.comb[m * 16 + k];
        }
    }
    if constexpr (FUSE) block.wait(step);       // every thread, valid or not (block getters may barrier)
    float nw[4] = {old[0], old[1], old[2], old[3]};
    float sq = 0.f, sqx = 0.f;
    if constexpr (FUSE && has_stage_v<Block>) {   // k16: the CTA's slot tile (NSLOT x nr rows x PW columns) in 16-byte loads, then each thread sums its column from smem.
        // No variable is declared outside this block: a new local in the unstaged instances moves their register allocation (SASS gate), so the nw update below is repeated here.
        constexpr int NS = Block::NSLOT;
        static_assert(NS * RT * (PW / 8) <= THREADS, "one 16-byte load per thread");
        __shared__ __align__(16) unsigned short stg[NS][RT][PW];
        if (t < NS * RT * (PW / 8)) {
            const StageMap q = stage_kstep(t);
            if (q.rr < nr) *reinterpret_cast<uint4*>(&stg[q.p][q.rr][q.seg * 8]) = block.ld16(q.p, (size_t)(m0 + q.rr) * H + (size_t)c * PW + q.seg * 8);
        }
        __syncthreads();
        if (valid) {
            float ys = 0.f;
            #pragma unroll
            for (int p = 0; p < NS; ++p) ys += __uint_as_float((unsigned)stg[p][r][hh] << 16);   // rank order from 0.f, as Ds41Block::operator()
            const float yv = rn16(ys);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float mix = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) mix += comb[j * 4 + i] * old[j];
                nw[i] = rn16(post[i] * yv + mix);
                st_t(sm + (size_t)i * H + h, nw[i]);
            }
        }
    }
    if (valid) {
        if constexpr (FUSE && !has_stage_v<Block>) {
            const float yv = rn16(block((size_t)m * H + h));
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float mix = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) mix += comb[j * 4 + i] * old[j];
                nw[i] = rn16(post[i] * yv + mix);
                st_t(sm + (size_t)i * H + h, nw[i]);
            }
        }
        float xv = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { tsm[r][i * PW + hh] = nw[i]; sq = fmaf(nw[i], nw[i], sq); xv += pre[i] * nw[i]; }
        xv = rn16(xv);
        sqx = xv * xv;
        y[(size_t)m * H + h] = __float2bfloat16_rn(xv);   // staged; the last CTA rescales it
    }
    for (int d = 16; d; d /= 2) { sq += __shfl_down_sync(0xffffffffu, sq, d); sqx += __shfl_down_sync(0xffffffffu, sqx, d); }
    if (!lane) { wred[wid][0] = sq; wred[wid][1] = sqx; }
    __syncthreads();
    // Step B: warp wid computes outputs wid, wid+8, wid+16 for the nr rows of this tile.
    {
        const int e0 = lane * 4;
        float d[3][RT];
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            const float* f0 = fa[q];
            const float* f1 = fb[q];
            #pragma unroll
            for (int rr = 0; rr < RT; ++rr) {
                d[q][rr] = 0.f;
                if (rr < nr) {
                    const float4 a0 = *reinterpret_cast<const float4*>(&tsm[rr][e0]);
                    const float4 a1 = *reinterpret_cast<const float4*>(&tsm[rr][128 + e0]);
                    float v = fmaf(a0.x, f0[0], 0.f);
                    v = fmaf(a0.y, f0[1], v); v = fmaf(a0.z, f0[2], v); v = fmaf(a0.w, f0[3], v);
                    v = fmaf(a1.x, f1[0], v); v = fmaf(a1.y, f1[1], v); v = fmaf(a1.z, f1[2], v); v = fmaf(a1.w, f1[3], v);
                    d[q][rr] = v;
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < 3; ++q)
            #pragma unroll
            for (int rr = 0; rr < RT; ++rr)
                for (int dd = 16; dd; dd /= 2) d[q][rr] += __shfl_down_sync(0xffffffffu, d[q][rr], dd);
        if (!lane) {
            #pragma unroll
            for (int q = 0; q < 3; ++q)
                #pragma unroll
                for (int rr = 0; rr < RT; ++rr)
                    if (rr < nr) b.part[((size_t)(m0 + rr) * NV + wid + 8 * q) * NCH + c] = d[q][rr];
        }
        if (t < RT && t < nr) {
            b.part[((size_t)(m0 + t) * NV + MIX) * NCH + c] = wred[2 * t][0] + wred[2 * t + 1][0];
            b.part[((size_t)(m0 + t) * NV + MIX + 1) * NCH + c] = wred[2 * t][1] + wred[2 * t + 1][1];
        }
    }
    __syncthreads();
    if (!t) {
        __threadfence();
        sh_last = (atomicInc(b.done + tile, NCH - 1) == NCH - 1);
        if (sh_last) __threadfence();
    }
    __syncthreads();
    if (!sh_last) return;
    // Tail (last CTA of the tile). Warp pair (2rr, 2rr+1) reduces row rr's 26 partial sums: 13 values each.
    {
        const int rr = wid >> 1;
        if (rr < nr) {
            const int v0 = (wid & 1) * 13;
            float ld[13][4], acc[13];
            #pragma unroll
            for (int q = 0; q < 13; ++q) {
                const float* pp = b.part + ((size_t)(m0 + rr) * NV + v0 + q) * NCH;
                #pragma unroll
                for (int i = 0; i < 4; ++i) ld[q][i] = lane + 32 * i < NCH ? __ldcg(pp + lane + 32 * i) : 0.f;   // NCH <= 128
            }
            #pragma unroll
            for (int q = 0; q < 13; ++q) acc[q] = ((ld[q][0] + ld[q][1]) + ld[q][2]) + ld[q][3];
            #pragma unroll
            for (int q = 0; q < 13; ++q)
                for (int dd = 16; dd; dd /= 2) acc[q] += __shfl_down_sync(0xffffffffu, acc[q], dd);
            if (!lane) {
                #pragma unroll
                for (int q = 0; q < 13; ++q) vals[rr][v0 + q] = acc[q];
            }
        }
    }
    __syncthreads();
    {
        const int rr = wid >> 1;
        if (!(wid & 1) && rr < nr) {   // warp 2rr: mixes -> coefficients; pre becomes pre_next
            const float rs = rsqrtf(vals[rr][MIX] / (4 * H) + eps);
            if (lane < MIX) { mixs[rr][lane] = vals[rr][lane] * rs; b.mix[(m0 + rr) * MIX + lane] = mixs[rr][lane]; }
            __syncwarp();
            coeff16(mixs[rr], w.base, w.scale, b.pre + 4 * (m0 + rr), b.post + 4 * (m0 + rr), b.comb + 16 * (m0 + rr));
        }
    }
    for (int rr = 0; rr < nr; ++rr) {   // y = BF16(x * inv * norm) from the staged BF16 x
        const float inv = rsqrtf(vals[rr][MIX + 1] / H + eps);
        __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(y + (size_t)(m0 + rr) * H);
        const __nv_bfloat162* n2 = reinterpret_cast<const __nv_bfloat162*>(w.norm);
        for (int p0 = t; p0 < H / 2; p0 += THREADS * YU) {
            __nv_bfloat162 xr[YU], nr2[YU];
            #pragma unroll
            for (int u = 0; u < YU; ++u) {
                const int p = p0 + THREADS * u;
                if (p < H / 2) { xr[u] = __ldcg(y2 + p); nr2[u] = __ldg(n2 + p); }
            }
            #pragma unroll
            for (int u = 0; u < YU; ++u) {
                const int p = p0 + THREADS * u;
                if (p < H / 2) {
                    const float2 xv = __bfloat1622float2(xr[u]), nv = __bfloat1622float2(nr2[u]);
                    y2[p] = __floats2bfloat162_rn(xv.x * inv * nv.x, xv.y * inv * nv.y);
                }
            }
        }
    }
}
// grid(ceil(H/THREADS), rows). Plain elementwise hc_post; every thread calls wait (block getters may barrier).
template <class TT, class Block>
__global__ void __launch_bounds__(THREADS) k_consume(BuffersT<TT> b, Block block, int H) {
    const int m = blockIdx.y, h = blockIdx.x * THREADS + threadIdx.x;
    const unsigned long long step = block.init();
    float post[4], comb[16];
    #pragma unroll
    for (int i = 0; i < 4; ++i) post[i] = b.post[m * 4 + i];
    #pragma unroll
    for (int k = 0; k < 16; ++k) comb[k] = b.comb[m * 16 + k];
    float old[4] = {0.f, 0.f, 0.f, 0.f};
    TT* __restrict__ s = b.streams + (size_t)m * 4 * H;
    if (h < H) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) old[j] = ld_t(s + (size_t)j * H + h);
    }
    block.wait(step);
    [[maybe_unused]] float ystage = 0.f;
    if constexpr (has_stage_v<Block>) {   // k16: the slot tile (NSLOT x THREADS columns) in 16-byte loads before the early return (the barrier needs every thread)
        constexpr int NS = Block::NSLOT;
        static_assert(NS * (THREADS / 8) <= THREADS, "one 16-byte load per thread");
        __shared__ __align__(16) unsigned short stg[NS][THREADS];
        if (threadIdx.x < NS * (THREADS / 8)) {
            const StageMap q = stage_consume(threadIdx.x);
            const int col = blockIdx.x * THREADS + q.seg * 8;
            if (col < H) *reinterpret_cast<uint4*>(&stg[q.p][q.seg * 8]) = block.ld16(q.p, (size_t)m * H + col);
        }
        __syncthreads();
        if (h < H) {
            #pragma unroll
            for (int p = 0; p < NS; ++p) ystage += __uint_as_float((unsigned)stg[p][threadIdx.x] << 16);   // rank order from 0.f
        }
    }
    if (h >= H) return;
    float yv;
    if constexpr (has_stage_v<Block>) yv = rn16(ystage);
    else yv = rn16(block((size_t)m * H + h));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float mix = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) mix += comb[j * 4 + i] * old[j];
        st_t(s + (size_t)i * H + h, rn16(post[i] * yv + mix));
    }
}
// grid(finish rows). Last consume + final collapse with the carried pre + norm.weight. store=true also writes the new trunk.
// out[m,H] BF16 and/or fragment tables (bind::Xfrag, as flash_mhc::k_mhc_final) for the lm_head GEMM.
// xpre (optional, BF16 [M,H]): the collapsed row x = BF16(sum pre*t) before the norm (DSpark confidence head).
// row0 (default 0): finish row m of the grid works on trunk / mHC state / block-output row row0 + m (so a prefill chunk of W rows finishes
// its LAST row with grid 1, row0 = W - 1); out, xpre and the fragment table are indexed by m, the finish row (a one-row head). The block's
// wait still covers the whole collective of the chunk. row0 = 0 is the old kernel byte for byte.
template <class TT, class Block>
__global__ void __launch_bounds__(THREADS) k_finish(BuffersT<TT> b, Block block, const __nv_bfloat16* norm, int H, float eps,
                                                    bool store, __nv_bfloat16* out, unsigned* f16 = nullptr,
                                                    unsigned* bf16 = nullptr, __nv_bfloat16* xpre = nullptr, int row0 = 0) {
    extern __shared__ float row[];
    const int m = blockIdx.x, M = gridDim.x, mi = row0 + m;
    const unsigned long long step = block.init();
    float post[4], comb[16], pre[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { post[i] = b.post[mi * 4 + i]; pre[i] = b.pre[mi * 4 + i]; }
    #pragma unroll
    for (int k = 0; k < 16; ++k) comb[k] = b.comb[mi * 16 + k];
    TT* __restrict__ s = b.streams + (size_t)mi * 4 * H;
    block.wait(step);
    float sq = 0.f;
    if constexpr (has_stage_v<Block>) {   // k16: the same loop, per FCH-column chunk: the slot chunk in 16-byte loads, then each thread sums its columns from smem. H % FCH == 0 (host-checked)
        constexpr int NS = Block::NSLOT;
        __shared__ __align__(16) unsigned short stg[NS][FCH];
        for (int h0 = 0; h0 < H; h0 += FCH) {
            __syncthreads();   // the previous chunk's smem reads are done
            for (int u = threadIdx.x; u < NS * (FCH / 8); u += THREADS) {
                const StageMap q = stage_finish(u);
                *reinterpret_cast<uint4*>(&stg[q.p][q.seg * 8]) = block.ld16(q.p, (size_t)mi * H + h0 + q.seg * 8);
            }
            __syncthreads();
            for (int h = h0 + threadIdx.x; h < h0 + FCH; h += THREADS) {   // h ascends per thread exactly as the plain loop, so sq adds in the same order
                float old[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) old[j] = ld_t(s + (size_t)j * H + h);
                float ys = 0.f;
                #pragma unroll
                for (int p = 0; p < NS; ++p) ys += __uint_as_float((unsigned)stg[p][h - h0] << 16);
                const float yv = rn16(ys);
                float x = 0.f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float mix = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) mix += comb[j * 4 + i] * old[j];
                    const float t = rn16(post[i] * yv + mix);
                    if (store) st_t(s + (size_t)i * H + h, t);
                    x += pre[i] * t;
                }
                x = rn16(x);
                row[h] = x; sq += x * x;
                if (xpre) xpre[(size_t)m * H + h] = __float2bfloat16_rn(x);
            }
        }
    }
    if constexpr (!has_stage_v<Block>)
    for (int h = threadIdx.x; h < H; h += THREADS) {
        float old[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) old[j] = ld_t(s + (size_t)j * H + h);
        const float yv = rn16(block((size_t)mi * H + h));
        float x = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mix = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) mix += comb[j * 4 + i] * old[j];
            const float t = rn16(post[i] * yv + mix);
            if (store) st_t(s + (size_t)i * H + h, t);
            x += pre[i] * t;
        }
        x = rn16(x);
        row[h] = x; sq += x * x;
        if (xpre) xpre[(size_t)m * H + h] = __float2bfloat16_rn(x);
    }
    const float inv = rsqrtf(sum1(sq) / H + eps);
    for (int h = threadIdx.x; h < H; h += THREADS) {
        const float v = rn16(row[h] * inv * __bfloat162float(norm[h]));
        if (out) out[(size_t)m * H + h] = __float2bfloat16_rn(v);
        if (f16 || bf16) {
            const float nb = __shfl_xor_sync(__activemask(), v, 1);
            if (!(threadIdx.x & 1)) flash_mhc::write_frag_pair(m, M, H, h, v, nb, f16, bf16);
        }
    }
}
template <class TT>
inline void embed(BuffersT<TT> b, const int* tok, const __nv_bfloat16* table, int M, int H, cudaStream_t st) {
    k_embed<TT><<<dim3((H + THREADS - 1) / THREADS, M), THREADS, 0, st>>>(b, tok, table, H);
}
inline void check_h(int H) {
    if (H % PW || H / PW > 128 || H % 2) { fprintf(stderr, "ds_mhc: H=%d must be a multiple of %d with H/%d <= 128\n", H, PW, PW); abort(); }
}
template <class WT, class TT>
inline void prepare(BuffersT<TT> b, Weights<WT> w, int M, int H, float eps, __nv_bfloat16* y, cudaStream_t st) {
    check_h(H);
    k_step<TT, WT, NoBlock, false><<<dim3(H / PW, (M + RT - 1) / RT), THREADS, 0, st>>>(b, w, NoBlock{}, H, M, eps, y);
}
// consume(s-1) + prepare(s) in one launch: the trunk is read and written once. Same results as consume() then prepare().
template <class WT, class Block, class TT>
inline void consume_prepare(BuffersT<TT> b, Block block, Weights<WT> w, int M, int H, float eps, __nv_bfloat16* y, cudaStream_t st) {
    check_h(H);
    k_step<TT, WT, Block, true><<<dim3(H / PW, (M + RT - 1) / RT), THREADS, 0, st>>>(b, w, block, H, M, eps, y);
}
template <class Block, class TT>
inline void consume(BuffersT<TT> b, Block block, int M, int H, cudaStream_t st) {
    k_consume<TT, Block><<<dim3((H + THREADS - 1) / THREADS, M), THREADS, 0, st>>>(b, block, H);
}
template <class Block, class TT>
inline void finish(BuffersT<TT> b, Block block, const __nv_bfloat16* norm, int M, int H, float eps, bool store,
                   __nv_bfloat16* out, cudaStream_t st, unsigned* f16 = nullptr, unsigned* bf16 = nullptr,
                   __nv_bfloat16* xpre = nullptr, int row0 = 0) {
    k_finish<TT, Block><<<M, THREADS, H * sizeof(float), st>>>(b, block, norm, H, eps, store, out, f16, bf16, xpre, row0);
}
// Whole-trunk copies to / from FP32 for the diagnostics (GR8_RESID_DUMP, GR8_DS_FEED_DIR): n elements. The float overloads are a plain device copy,
// so the default build runs exactly the old cudaMemcpyAsync. f32_to_trunk rounds to BF16 (exact when the source is BF16-valued, as the oracle's is).
template <class A, class B> __global__ void k_cvt(const A* src, B* dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_t(dst + i, ld_t(src + i));
}
inline void trunk_to_f32(const float* src, float* dst, size_t n, cudaStream_t st) { cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st); }
inline void trunk_to_f32(const __nv_bfloat16* src, float* dst, size_t n, cudaStream_t st) { k_cvt<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(src, dst, n); }
inline void f32_to_trunk(const float* src, float* dst, size_t n, cudaStream_t st) { cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st); }
inline void f32_to_trunk(const float* src, __nv_bfloat16* dst, size_t n, cudaStream_t st) { k_cvt<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(src, dst, n); }
// words of the tile counter Buffers::done needs for a chunk of `rows` rows (one per 4-row tile; prepare / consume_prepare index done + tile)
inline int done_words(int rows) { return (rows + RT - 1) / RT; }
} // namespace ds_mhc
